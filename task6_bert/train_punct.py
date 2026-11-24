"""Fine-tune Chinese BERT with separate training, validation and held-out sets."""
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader, Dataset
from transformers import (
    BertForTokenClassification,
    BertTokenizerFast,
    get_linear_schedule_with_warmup,
)

from punctuation import ID2P, LABEL_NAMES, P2ID, token_labels

BASE = Path(__file__).parent
OUT = BASE / "results"
MODEL_NAME = os.environ.get("BERT_BASE_MODEL", "bert-base-chinese")
MAX_LEN = 252
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def read_rows(path):
    with path.open(encoding="utf-8") as source:
        return [json.loads(line) for line in source]


def unique_rows(rows, seen):
    unique = []
    for row in rows:
        text = row["input"]
        if text not in seen:
            seen.add(text)
            unique.append(row)
    return unique


class PunctDataset(Dataset):
    def __init__(self, path_or_rows, tokenizer):
        self.rows = read_rows(Path(path_or_rows)) if isinstance(path_or_rows, (str, Path)) else path_or_rows
        self.tok = tokenizer

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        row = self.rows[i]
        enc = self.tok(
            row["input"], max_length=MAX_LEN, truncation=True,
            return_offsets_mapping=True,
        )
        return (
            torch.tensor(enc["input_ids"]),
            torch.tensor(enc["attention_mask"]),
            torch.tensor(token_labels(enc["offset_mapping"], row["labels"])),
        )


def collate(batch):
    ids, masks, labels = zip(*batch)
    return tuple(
        torch.nn.utils.rnn.pad_sequence(values, batch_first=True, padding_value=padding).to(DEVICE)
        for values, padding in ((ids, 0), (masks, 0), (labels, -100))
    )


@torch.inference_mode()
def evaluate(model, loader):
    model.eval()
    gold, predictions = [], []
    for ids, mask, labels in loader:
        with torch.autocast("cuda", dtype=torch.float16, enabled=DEVICE == "cuda"):
            logits = model(ids, attention_mask=mask).logits
        keep = labels != -100
        gold.extend(labels[keep].cpu().tolist())
        predictions.extend(logits[..., :len(ID2P)].argmax(-1)[keep].cpu().tolist())
    labels = list(range(len(ID2P)))
    report = classification_report(
        gold, predictions, labels=labels, target_names=LABEL_NAMES,
        output_dict=True, zero_division=0,
    )
    return {
        "accuracy": float(np.mean(np.equal(gold, predictions))),
        "macro_f1": report["macro avg"]["f1-score"],
        "punctuation_macro_f1": float(np.mean([report[name]["f1-score"] for name in LABEL_NAMES[1:]])),
        "per_label": {name: report[name] for name in LABEL_NAMES},
        "tokens": len(gold),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--train-limit", type=int, default=8000)
    parser.add_argument("--validation-limit", type=int, default=3000)
    parser.add_argument("--lr", type=float, default=3e-5)
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.train_limit, args.validation_limit) < 1:
        parser.error("epochs, batch size, and split limits must be positive")

    torch.manual_seed(SEED)
    tok = BertTokenizerFast.from_pretrained(MODEL_NAME, local_files_only=True)
    model = BertForTokenClassification.from_pretrained(
        MODEL_NAME, num_labels=len(ID2P), local_files_only=True,
        id2label=dict(enumerate(LABEL_NAMES)),
        label2id={label: index for index, label in enumerate(LABEL_NAMES)},
    ).to(DEVICE)

    seen = set()
    train = unique_rows(read_rows(BASE / "data" / "train.jsonl"), seen)
    validation = unique_rows(read_rows(BASE / "data" / "test.jsonl"), seen)
    heldout = unique_rows(read_rows(BASE / "data" / "test_march.jsonl"), seen)
    train = train[:args.train_limit]
    validation = validation[:args.validation_limit]
    loaders = [
        DataLoader(PunctDataset(rows, tok), batch_size=args.batch_size, shuffle=shuffle, collate_fn=collate)
        for rows, shuffle in ((train, True), (validation, False), (heldout, False))
    ]
    train_loader, validation_loader, heldout_loader = loaders

    out_model = OUT / "bert_punct"
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    steps = len(train_loader) * args.epochs
    sched = get_linear_schedule_with_warmup(opt, int(steps * 0.1), steps)
    scaler = torch.amp.GradScaler("cuda", enabled=DEVICE == "cuda")
    history = []
    best_f1 = -1.0
    best_epoch = None
    if DEVICE == "cuda":
        torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    print(f"device={DEVICE}, train={len(train)} validation={len(validation)} heldout={len(heldout)}", flush=True)
    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum = 0.0
        for step, (ids, masks, labels) in enumerate(train_loader, 1):
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16, enabled=DEVICE == "cuda"):
                loss = model(ids, attention_mask=masks, labels=labels).loss
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            loss_sum += loss.item()
            if step % 50 == 0 or step == len(train_loader):
                print(f"epoch={epoch} step={step}/{len(train_loader)} loss={loss_sum / step:.4f}", flush=True)

        scores = evaluate(model, validation_loader)
        history.append({"epoch": epoch, "train_loss": loss_sum / len(train_loader), "validation": scores})
        print(f"epoch={epoch} validation_macro_f1={scores['macro_f1']:.4f}", flush=True)
        if scores["macro_f1"] > best_f1:
            best_f1 = scores["macro_f1"]
            best_epoch = epoch
            out_model.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(out_model)
            tok.save_pretrained(out_model)

    train_seconds = time.perf_counter() - start
    del model, opt
    model = BertForTokenClassification.from_pretrained(out_model).to(DEVICE)
    test_scores = evaluate(model, heldout_loader)
    result = {
        "configuration": {
            **vars(args), "seed": SEED, "max_tokens": MAX_LEN, "base_model": MODEL_NAME,
            "torch_version": str(torch.__version__), "device": DEVICE,
            "gpu": torch.cuda.get_device_name() if DEVICE == "cuda" else None,
        },
        "splits": {"train_01": len(train), "validation_02": len(validation), "heldout_03": len(heldout)},
        "split_deduplication": "Exact normalized input by chronological split, before limits.",
        "best_epoch": best_epoch, "train_and_validation_seconds": train_seconds,
        "peak_gpu_bytes": torch.cuda.max_memory_allocated() if DEVICE == "cuda" else None,
        "history": history, "heldout": test_scores,
    }
    (OUT / "eval_v2.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"heldout_macro_f1={test_scores['macro_f1']:.4f} punctuation_f1={test_scores['punctuation_macro_f1']:.4f}", flush=True)


if __name__ == "__main__":
    main()
