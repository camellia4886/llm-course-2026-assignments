"""Compare serial punctuation and precomputed batched sliding windows."""
import json
import os
import statistics
import time
from pathlib import Path

import torch
from sklearn.metrics import classification_report
from transformers import BertForTokenClassification, BertTokenizerFast

from punctuation import ID2P, LABEL_NAMES, P2ID, character_labels
from train_punct import read_rows, unique_rows

BASE = Path(__file__).parent
MODEL_DIR = Path(os.environ.get("BERT_PUNCT_DIR", BASE / "results" / "bert_punct"))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
WINDOW, KEEP, LEFT_DROP = 400, 300, 50
STRIDE = KEEP - LEFT_DROP


def load():
    tok = BertTokenizerFast.from_pretrained(MODEL_DIR)
    model = BertForTokenClassification.from_pretrained(MODEL_DIR).to(DEVICE).eval()
    return tok, model


def puncted(chars, labels):
    return "".join(char + ID2P[label] for char, label in zip(chars, labels))


def forward_batch(tok, model, windows, batch):
    """Return an aligned punctuation label for every original character."""
    predictions = []
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16, enabled=DEVICE == "cuda"):
        for start in range(0, len(windows), batch):
            group = windows[start:start + batch]
            encoded = tok(group, padding=True, return_tensors="pt", return_offsets_mapping=True)
            offsets = encoded.pop("offset_mapping").tolist()
            logits = model(**encoded.to(DEVICE)).logits[..., :len(ID2P)]
            rows = logits.argmax(-1).cpu().tolist()
            predictions.extend(character_labels(mapping, row, len(text))
                               for text, mapping, row in zip(group, offsets, rows))
    return predictions


def sync():
    if DEVICE == "cuda":
        torch.cuda.synchronize()


def decode_sequential(tok, model, text):
    """Choose the next window after the last predicted full stop in its first 300 characters."""
    sync()
    start_time = time.perf_counter()
    labels = []
    start = 0
    calls = 0
    while start < len(text):
        window = text[start:start + WINDOW]
        window_labels = forward_batch(tok, model, [window], 1)[0]
        calls += 1
        if start + len(window) >= len(text):
            labels.extend(window_labels)
            break
        cut = max((index + 1 for index, label in enumerate(window_labels[:KEEP])
                   if label == P2ID["。"]), default=KEEP)
        labels.extend(window_labels[:cut])
        start += cut
    text_with_punctuation = puncted(text, labels)
    sync()
    return text_with_punctuation, time.perf_counter() - start_time, calls, [ID2P[label] for label in labels]


def decode_batch(tok, model, text, batch):
    """Precompute all windows and merge their disjoint central character ranges."""
    if batch < 1:
        raise ValueError("batch must be positive")
    sync()
    start_time = time.perf_counter()
    starts = []
    if text:
        starts.append(0)
        while starts[-1] + WINDOW < len(text):
            starts.append(starts[-1] + STRIDE)
    windows = [text[start:start + WINDOW] for start in starts]
    all_labels = forward_batch(tok, model, windows, batch)
    labels = []
    for index, prediction in enumerate(all_labels):
        left = LEFT_DROP if index else 0
        right = KEEP if index + 1 < len(all_labels) else len(prediction)
        labels.extend(prediction[left:right])
    text_with_punctuation = puncted(text, labels)
    sync()
    return text_with_punctuation, time.perf_counter() - start_time, len(windows), [ID2P[label] for label in labels]


def modern_benchmark(length=20000):
    seen = set()
    for filename in ("train.jsonl", "test.jsonl"):
        unique_rows(read_rows(BASE / "data" / filename), seen)
    rows = unique_rows(read_rows(BASE / "data" / "test_march.jsonl"), seen)
    pieces, labels = [], []
    for row in rows:
        pieces.append(row["input"])
        labels.extend(row["labels"])
        if len(labels) >= length:
            break
    return "".join(pieces)[:length], labels[:length]


def ancient_benchmark():
    source = BASE.parent / "task17_domain" / "guwen.tsv"
    texts, labels = [], []
    from build_data import STRIP
    for line in source.read_text(encoding="utf-8").splitlines():
        _, passage = line.split("\t", 1)
        chars, gold = [], []
        for char in passage:
            if char not in STRIP:
                chars.append(char)
                gold.append(0)
            elif char in P2ID and gold and not gold[-1]:
                gold[-1] = P2ID[char]
        texts.extend(chars)
        labels.extend(gold)
    return "".join(texts), labels


def scores(gold, predicted):
    report = classification_report(
        gold, [P2ID[label] for label in predicted], labels=list(range(len(ID2P))),
        target_names=LABEL_NAMES, output_dict=True, zero_division=0,
    )
    return {
        "macro_f1": report["macro avg"]["f1-score"],
        "punctuation_macro_f1": statistics.mean(report[label]["f1-score"] for label in LABEL_NAMES[1:]),
    }


def benchmark(tok, model, text, gold, repeats=3):
    results = {}
    baseline_labels = []
    for batch in (None, 1, 8, 16, 32, 64):
        if batch is None:
            runs = [decode_sequential(tok, model, text) for _ in range(repeats)]
        else:
            runs = [decode_batch(tok, model, text, batch) for _ in range(repeats)]
        _, _, windows, labels = runs[-1]
        if batch is None:
            baseline_labels = labels
        seconds = statistics.median(run[1] for run in runs)
        results["serial" if batch is None else f"batch_{batch}"] = {
            "windows": windows,
            "forward_calls": windows if batch is None else (windows + batch - 1) // batch,
            "median_seconds": seconds,
            "seconds_trials": [run[1] for run in runs],
            "characters_per_second": len(text) / seconds,
            "agreement_with_serial": sum(a == b for a, b in zip(labels, baseline_labels)) / len(text),
            **(scores(gold, labels) if gold is not None else {}),
        }
    return results


def main():
    tok, model = load()
    modern, modern_labels = modern_benchmark()
    ancient, ancient_labels = ancient_benchmark()
    for text in ("中国加入WTO以后继续推进合作", "报告发布于1998年", "甲" * 300 + "ABC" + "乙" * 110):
        for method, kwargs in ((decode_sequential, {}), (decode_batch, {"batch": 8})):
            decoded, _, _, labels = method(tok, model, text, **kwargs)
            assert len(labels) == len(text) and decoded.startswith(text[0])
    forward_batch(tok, model, [modern[:WINDOW]] * 4, 4)
    report = {
        "device": DEVICE, "gpu": torch.cuda.get_device_name() if DEVICE == "cuda" else None,
        "window_characters": WINDOW, "retained_prefix": KEEP, "fixed_stride": STRIDE,
        "test_cases": {
            "march_news": {"characters": len(modern), "results": benchmark(tok, model, modern, modern_labels)},
            "classical": {"characters": len(ancient), "results": benchmark(tok, model, ancient, ancient_labels)},
            "classical_long": {"characters": 3 * len(ancient), "results": benchmark(tok, model, ancient * 3, None)},
        },
        "classical_long_note": "The concatenated public-domain passages are repeated three times for throughput only.",
        "classical_sample": decode_batch(tok, model, ancient[:420], 8)[0],
    }
    path = BASE / "results" / "batch_decode_v2.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for name, case in report["test_cases"].items():
        print(name, [(key, round(item["median_seconds"], 3), item["forward_calls"])
                     for key, item in case["results"].items()])


if __name__ == "__main__":
    main()
