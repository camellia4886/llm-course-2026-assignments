"""任务17：有限预算下改善领域模型 —— 古文断句标点（真实小规模实验）

基线：task6 微调的 BERT 标点模型（仅现代新闻语料训练）
干预A：精选 SFT（5 篇古文，约 1100 字，继续微调）
干预B：规则后处理（句末语气词补标点，仅修正模型输出 O 的位置）
四臂对比：基线 / 基线+规则 / 精选SFT / 精选SFT+规则
主指标：古文测试（桃花源记、师说，未参与 SFT 的篇目）标点 F1
回归指标：现代文测试（98年2月语料）macro-F1
交付：数据与反馈规则、训练配置、逐样本错误分析、成本账本
"""
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import classification_report, accuracy_score
from torch.utils.data import DataLoader, Dataset
from transformers import (BertForTokenClassification, BertTokenizerFast,
                          get_linear_schedule_with_warmup)

BASE = Path(__file__).parent
TASK6 = BASE.parent / "task6_bert"
sys.path.insert(0, str(TASK6))
from batch_decode import decode_batch, forward_batch, sync
from punctuation import LABEL_NAMES, token_labels
from train_punct import read_rows, unique_rows

MODEL_DIR = TASK6 / "results" / "bert_punct"
ID2P = ["", "，", "。", "、", "；", "：", "？", "！"]
P2ID = {p: i for i, p in enumerate(ID2P)}
STRIP = set("，。、；：？！“”‘’《》〈〉（）【】…—·•・,.;:?!\"'()[]<>-~～…")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_LEN = 512
# 语气词规则：仅在模型预测为 O 的位置补标点
RULE = {c: P2ID["？"] for c in "乎邪耶哉欤"}
RULE.update({c: P2ID["。"] for c in "也矣焉耳"})


def texts_to_samples(tsv_path, names):
    """tsv: 篇名\t带标点原文 -> {input, labels} 列表"""
    rows = {}
    for line in tsv_path.read_text(encoding="utf-8").splitlines():
        name, text = line.split("\t", 1)
        rows[name] = text
    out = []
    for name in names:
        chars, labels = [], []
        for c in rows[name]:
            if c in STRIP:
                if c in P2ID and labels and labels[-1] == 0:
                    labels[-1] = P2ID[c]
                continue
            chars.append(c)
            labels.append(0)
        out.append({"input": "".join(chars), "labels": labels, "name": name})
    return out


class SampleDS(Dataset):
    def __init__(self, samples, tok):
        self.samples = samples
        self.tok = tok

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        r = self.samples[i]
        enc = self.tok(r["input"], truncation=True, max_length=MAX_LEN,
                       add_special_tokens=True, return_offsets_mapping=True)
        labels = token_labels(enc["offset_mapping"], r["labels"])
        return (torch.tensor(enc["input_ids"]), torch.tensor(enc["attention_mask"]),
                torch.tensor(labels))


def collate(batch):
    ids = torch.nn.utils.rnn.pad_sequence([b[0] for b in batch], batch_first=True,
                                          padding_value=0).to(DEVICE)
    mask = torch.nn.utils.rnn.pad_sequence([b[1] for b in batch], batch_first=True,
                                           padding_value=0).to(DEVICE)
    labs = torch.nn.utils.rnn.pad_sequence([b[2] for b in batch], batch_first=True,
                                           padding_value=-100).to(DEVICE)
    return ids, mask, labs


def predict(model, tok, samples, batch=4):
    """返回每样本 (chars, gold_labels, pred_labels)"""
    model.eval()
    short = [sample["input"] for sample in samples if len(sample["input"]) <= 400]
    short_predictions = iter(forward_batch(tok, model, short, batch))
    predictions = []
    for sample in samples:
        text = sample["input"]
        if len(text) <= 400:
            labels = next(short_predictions)
        else:
            _, _, _, punct = decode_batch(tok, model, text, batch)
            labels = [P2ID[label] for label in punct]
        predictions.append((text, sample["labels"], labels))
    return predictions


def apply_rule(chars, labels):
    labels = list(labels)
    for i, c in enumerate(chars):
        if labels[i] == 0 and c in RULE:
            labels[i] = RULE[c]
    return labels


def metrics(samples_pred):
    """samples_pred: (chars, gold, pred) -> acc, macro_f1(全部类), macro_f1(出现类), report"""
    y_true, y_pred = [], []
    for _, g, p in samples_pred:
        y_true.extend(g)
        y_pred.extend(p)
    acc = accuracy_score(y_true, y_pred)
    rep = classification_report(y_true, y_pred,
                                labels=list(range(len(ID2P))),
                                target_names=["O"] + ID2P[1:],
                                output_dict=True, zero_division=0)
    present = sorted(set(y_true))
    names = ["O"] + ID2P[1:]
    f1_present = float(np.mean([rep[names[i]]["f1-score"] for i in present]))
    return acc, rep["macro avg"]["f1-score"], f1_present, rep


def sft(model, tok, samples, epochs=8, lr=2e-5):
    model.train()
    ds = SampleDS(samples, tok)
    loader = DataLoader(ds, batch_size=4, shuffle=True, collate_fn=collate)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total = len(loader) * epochs
    sched = get_linear_schedule_with_warmup(opt, max(1, total // 10), total)
    scaler = torch.amp.GradScaler("cuda", enabled=DEVICE == "cuda")
    for ep in range(epochs):
        for ids, mask, labs in loader:
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16, enabled=DEVICE == "cuda"):
                loss = model(ids, attention_mask=mask, labels=labs).loss
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
    model.eval()


def load_modern_test(tok, n=2000):
    seen = set()
    for filename in ("train.jsonl", "test.jsonl"):
        unique_rows(read_rows(TASK6 / "data" / filename), seen)
    rows = unique_rows(read_rows(TASK6 / "data" / "test_march.jsonl"), seen)[:n]
    return [{"input": r["input"], "labels": r["labels"], "name": "modern"}
            for r in rows]


def sentence_errors(triples):
    """triples: (chars, gold, pred) -> 逐句不一致的句对列表"""
    errors = []
    for ci, (chars, g, p) in enumerate(triples):
        g_sents, p_sents, cur = [], [], []
        for c, gl in zip(chars, g):
            cur.append(c)
            if gl in (2, 6, 7):
                g_sents.append("".join(cur) + ID2P[gl])
                cur = []
        cur = []
        for c, pl in zip(chars, p):
            cur.append(c)
            if pl in (2, 6, 7):
                p_sents.append("".join(cur) + ID2P[pl])
                cur = []
        if len(g_sents) != len(p_sents):
            errors.append((ci, len(g_sents), len(p_sents), "句数不一致"))
            continue
        for gi, (gs, ps) in enumerate(zip(g_sents, p_sents)):
            if gs != ps:
                errors.append((ci, gi, gs, ps))
    return errors


def main():
    torch.manual_seed(42)
    tok = BertTokenizerFast.from_pretrained(MODEL_DIR)
    model = BertForTokenClassification.from_pretrained(MODEL_DIR).to(DEVICE).eval()
    ledger = {}

    guwen = texts_to_samples(BASE / "guwen.tsv",
                             ["岳阳楼记", "醉翁亭记", "爱莲说", "陋室铭", "马说",
                              "桃花源记", "师说"])
    train_gw = [s for s in guwen if s["name"] not in ("桃花源记", "师说")]
    test_gw = [s for s in guwen if s["name"] in ("桃花源记", "师说")]
    modern = load_modern_test(tok)
    variant_file = BASE / "constraint_variants.tsv"
    variant_names = [line.split("\t", 1)[0] for line in variant_file.read_text(encoding="utf-8").splitlines()]
    variants = texts_to_samples(variant_file, variant_names)

    t0 = time.time()
    base_gw = predict(model, tok, test_gw)
    base_mod = predict(model, tok, modern, batch=32)
    base_variants = predict(model, tok, variants)
    sync()
    ledger["eval_baseline_s"] = round(time.time() - t0, 1)

    # SFT
    t0 = time.time()
    sft(model, tok, train_gw)
    sync()
    ledger["sft_train_s"] = round(time.time() - t0, 1)
    t0 = time.time()
    sft_gw = predict(model, tok, test_gw)
    sft_mod = predict(model, tok, modern, batch=32)
    sft_variants = predict(model, tok, variants)
    sync()
    ledger["eval_sft_s"] = round(time.time() - t0, 1)
    (BASE / "results").mkdir(exist_ok=True)
    model.save_pretrained(BASE / "results" / "bert_punct_classical")

    results = {
        "arms": {}, "ledger": ledger, "configuration": {
            "seed": 42, "epochs": 8, "learning_rate": 2e-5, "batch": 4,
            "device": DEVICE, "training": [sample["name"] for sample in train_gw],
            "test": [sample["name"] for sample in test_gw], "modern_holdout": len(modern),
        },
        "train_chars": sum(len(s["input"]) for s in train_gw), "constraint_variants": {},
    }
    for name, gw, mod, checks in [("基线", base_gw, base_mod, base_variants),
                                  ("精选SFT", sft_gw, sft_mod, sft_variants)]:
        acc_g, f1_g, f1p_g, rep_g = metrics(gw)
        acc_m, f1_m, _, _ = metrics(mod)
        # +规则
        gw_r = [(c, g, apply_rule(c, p)) for c, g, p in gw]
        mod_r = [(c, g, apply_rule(c, p)) for c, g, p in mod]
        acc_gr, f1_gr, f1p_gr, rep_gr = metrics(gw_r)
        acc_mr, f1_mr, _, _ = metrics(mod_r)
        results["arms"][name] = {
            "古文_acc": round(acc_g, 4), "古文_macroF1": round(f1_g, 4),
            "古文_出现类macroF1": round(f1p_g, 4),
            "古文_+规则_acc": round(acc_gr, 4), "古文_+规则_macroF1": round(f1_gr, 4),
            "古文_+规则_出现类macroF1": round(f1p_gr, 4),
            "现代文_macroF1": round(f1_m, 4), "现代文_+规则_macroF1": round(f1_mr, 4),
            "古文_per_label": {k: round(v["f1-score"], 4) for k, v in rep_g.items()
                              if k in ["O"] + ID2P[1:]},
            "古文_仅标点macroF1": float(np.mean([rep_g[k]["f1-score"] for k in LABEL_NAMES[1:]])),
            "古文_+规则_仅标点macroF1": float(np.mean([rep_gr[k]["f1-score"] for k in LABEL_NAMES[1:]])),
        }
        variant_results = []
        for variant, (chars, gold, pred) in zip(variants, checks):
            rule_labels = apply_rule(chars, pred)
            render = lambda labels: "".join(c + ID2P[l] for c, l in zip(chars, labels))
            variant_results.append({
                "name": variant["name"], "gold": render(gold), "predicted": render(pred),
                "with_rule": render(rule_labels), "exact": gold == pred,
                "exact_with_rule": gold == rule_labels,
            })
        results["constraint_variants"][name] = variant_results
        if name == "基线":
            results["errors_基线_古文"] = sentence_errors(gw)[:15]
        else:
            results["errors_精选SFT_古文"] = sentence_errors(gw)[:15]

    json.dump(results, open(BASE / "results" / "domain_sft_v2.json", "w",
                            encoding="utf-8"), ensure_ascii=False, indent=2)
    print(json.dumps(results, ensure_ascii=False, indent=2)[:4000])


if __name__ == "__main__":
    main()
