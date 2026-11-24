"""任务17 补充：逐样本错误示例（gold vs pred vs +规则）。"""
import json
from pathlib import Path

import torch
from transformers import BertForTokenClassification, BertTokenizerFast

from run_domain_sft import (BASE, DEVICE, ID2P, TASK6, apply_rule, predict,
                            texts_to_samples)

MODEL_DIRS = {"基线": TASK6 / "results" / "bert_punct",
              "精选SFT": BASE / "results" / "bert_punct_classical"}


def diffs(chars, gold, pred, k=30):
    out = []
    for i in range(len(chars)):
        if gold[i] != pred[i]:
            out.append({"pos": i,
                        "ctx": chars[max(0, i - 6):i + 7],
                        "gold": ID2P[gold[i]] or "O",
                        "pred": ID2P[pred[i]] or "O"})
    return out[:k]


def main():
    tok = BertTokenizerFast.from_pretrained(TASK6 / "results" / "bert_punct")
    test_gw = texts_to_samples(BASE / "guwen.tsv", ["桃花源记", "师说"])
    out = {}
    for arm, mdir in MODEL_DIRS.items():
        model = BertForTokenClassification.from_pretrained(mdir).to(DEVICE).eval()
        triples = predict(model, tok, test_gw, batch=2)
        arm_out = []
        for chars, gold, pred in triples:
            rule = apply_rule(chars, pred)
            arm_out.append({
                "chars": len(chars),
                "gold": "".join(c + (ID2P[g] if g else "") for c, g in zip(chars, gold)),
                "pred": "".join(c + (ID2P[p] if p else "") for c, p in zip(chars, pred)),
                "pred_rule": "".join(c + (ID2P[p] if p else "")
                                    for c, p in zip(chars, rule)),
                "diffs": diffs(chars, gold, pred),
            })
        out[arm] = arm_out
    json.dump(out, open(BASE / "results" / "error_examples_v2.json", "w",
                        encoding="utf-8"), ensure_ascii=False, indent=2)
    for arm, arm_out in out.items():
        for s in arm_out:
            print(f"[{arm}] {s['chars']}字 diff数={len(s['diffs'])}")
            for d in s["diffs"][:12]:
                print(f"  …{d['ctx']}… gold={d['gold']} pred={d['pred']}")
    print("saved results/error_examples_v2.json")


if __name__ == "__main__":
    main()
