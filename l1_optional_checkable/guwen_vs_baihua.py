# -*- coding: utf-8 -*-
"""L1 可选对照实验：古文 vs 白话改写，区分断句失败是句式先验(a)还是整体退化(b)。

设计（见仓库根目录《L1可选-把熟悉项目重做成可检查任务.md》§5）：
- 关键条件变化样本对：同一语义，仅改语体——7 篇古文（guwen.tsv）vs
  7 篇白话改写（baihua.tsv，逐句对齐古文标点结构，金标句数完全一致）；
  预期：若白话标点质量显著高于古文 → 支持 (a) 句式先验解释；
        若两者都差 → 支持 (b) 语义/领域整体退化解释。
- 控制样本对（答案应保持不变）：原句 vs 仅替换专名（等长替换），
  比较非替换位置的标签是否完全一致；若不一致 → 模型依赖词形/词频而非句法结构。
- 模型为 task6 训练的 BERT 断句模型（eval 模式前向是确定性的，无随机源，
  因此不做 seed 重复，改用全 7 篇全文扩大样本量）。
- 长文（>512 token）用 task6 的滑窗批量解码（decode_batch），与任务7 一致。

依赖：task6 训练好的模型（目录不存在时用环境变量 BERT_PUNCT_DIR 指定，
或按 README 命令 `python task6_bert/train_punct.py` 重建）。
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

BASE = Path(__file__).parent
REPO = BASE.parent
sys.path.insert(0, str(REPO / "task17_domain"))
sys.path.insert(0, str(REPO / "task6_bert"))

from run_domain_sft import (ID2P, P2ID, metrics, sentence_errors,  # noqa: E402
                            texts_to_samples)
from batch_decode import decode_batch  # noqa: E402

MODEL_DIR = Path(os.environ.get("BERT_PUNCT_DIR",
                                REPO / "task6_bert" / "results" / "bert_punct"))
NAMES = ["岳阳楼记", "醉翁亭记", "爱莲说", "陋室铭", "马说", "桃花源记", "师说"]

# 控制样本对（§5）：原句 vs 等长专名替换
CONTROL_PAIRS = [
    {"name": "桃花源记开篇", "exclude": [0, 1, 2, 17, 18, 19],
     "c": "武陵人捕鱼为业缘溪行忘路之远近忽逢桃花林",
     "d": "豫章人捕鱼为业缘溪行忘路之远近忽逢梨花林"},
    {"name": "陋室铭尾联", "exclude": [2, 3, 7, 8],
     "c": "南阳诸葛庐西蜀子云亭孔子云何陋之有",
     "d": "南阳司马庐西蜀子房亭孔子云何陋之有"},
]


def decode_int_labels(tok, model, text, batch=8):
    """滑窗解码 -> 与输入等长的 int 标签列表。"""
    _, _, _, lab_str = decode_batch(tok, model, text, batch)
    assert len(lab_str) == len(text), f"{len(lab_str)} != {len(text)}"
    return [P2ID[p] if p in P2ID else 0 for p in lab_str]


def sents_of(labels):
    """句末标点（。？！=2/6/7）计数。"""
    return sum(1 for l in labels if l in (2, 6, 7))


def main():
    t0 = time.time()
    # batch_decode.load() 用其模块内 MODEL_DIR（仓库路径），此处自行加载
    from transformers import BertForTokenClassification, BertTokenizerFast
    tok = BertTokenizerFast.from_pretrained(MODEL_DIR)
    model = BertForTokenClassification.from_pretrained(MODEL_DIR).eval()
    if __import__("torch").cuda.is_available():
        model = model.to("cuda")

    gw = texts_to_samples(REPO / "task17_domain" / "guwen.tsv", NAMES)
    bw = texts_to_samples(BASE / "baihua.tsv", NAMES)
    assert len(gw) == len(bw) == 7

    # 自检：白话金标句数与古文金标句数逐篇一致（改写对齐质量检查）
    mirror_check = {g["name"]: (sents_of(g["labels"]), sents_of(b["labels"]))
                    for g, b in zip(gw, bw)}
    assert all(a == b for a, b in mirror_check.values()), mirror_check

    out = {"setup": {"model": str(MODEL_DIR), "decode": "decode_batch(滑窗)",
                     "mirror_check_gold_sents": mirror_check},
           "per_text": [], "control_pairs": []}

    all_g, all_b = [], []
    for g, b in zip(gw, bw):
        pg = decode_int_labels(tok, model, g["input"])
        pb = decode_int_labels(tok, model, b["input"])
        all_g.append((g["input"], g["labels"], pg))
        all_b.append((b["input"], b["labels"], pb))
        for domain, tup in (("古文", (g["input"], g["labels"], pg)),
                            ("白话", (b["input"], b["labels"], pb))):
            _, _, f1_present, _ = metrics([tup])
            acc = np.mean([x == y for x, y in zip(tup[1], tup[2])])
            out["per_text"].append({
                "name": g["name"], "domain": domain, "chars": len(tup[0]),
                "acc": round(float(acc), 4), "macroF1_出现类": round(f1_present, 4),
                "gold_sents": sents_of(tup[1]), "pred_sents": sents_of(tup[2]),
                "sents_delta": sents_of(tup[2]) - sents_of(tup[1])})

    acc_g, f1_g, f1p_g, _ = metrics(all_g)
    acc_b, f1_b, f1p_b, _ = metrics(all_b)
    out["overall"] = {
        "古文": {"acc": round(float(acc_g), 4), "macroF1": round(f1_g, 4),
                 "macroF1_出现类": round(f1p_g, 4), "chars": sum(len(c) for c, _, _ in all_g)},
        "白话": {"acc": round(float(acc_b), 4), "macroF1": round(f1_b, 4),
                 "macroF1_出现类": round(f1p_b, 4), "chars": sum(len(c) for c, _, _ in all_b)}}
    out["errors_古文"] = sentence_errors(all_g)[:12]
    out["errors_白话"] = sentence_errors(all_b)[:12]

    for cp in CONTROL_PAIRS:
        lc = decode_int_labels(tok, model, cp["c"])
        ld = decode_int_labels(tok, model, cp["d"])
        assert len(lc) == len(ld) == len(cp["c"]) == len(cp["d"])
        mism = [(i, lc[i], ld[i]) for i in range(len(lc))
                if i not in cp["exclude"] and lc[i] != ld[i]]
        out["control_pairs"].append({
            "name": cp["name"], "excluded_pos": cp["exclude"],
            "n_compare": len(lc) - len(cp["exclude"]),
            "mismatch_outside": mism,
            "一致": len(mism) == 0})

    out["ledger_s"] = round(time.time() - t0, 1)
    (BASE / "results").mkdir(exist_ok=True)
    json.dump(out, open(BASE / "results" / "guwen_vs_baihua_v2.json", "w",
                        encoding="utf-8"), ensure_ascii=False, indent=2)
    print(json.dumps({"overall": out["overall"], "ledger_s": out["ledger_s"]},
                     ensure_ascii=False, indent=2))
    print("\n逐篇：")
    for r in out["per_text"]:
        print(f"  [{r['domain']}] {r['name']:5s} acc={r['acc']:.4f} "
              f"f1={r['macroF1_出现类']:.4f} 句数 {r['gold_sents']}->{r['pred_sents']}")
    print("\n控制对：")
    for cp in out["control_pairs"]:
        print(f"  {cp['name']}: 非替换位置不一致 {len(cp['mismatch_outside'])}/{cp['n_compare']}"
              + ("" if cp["一致"] else f" -> {cp['mismatch_outside'][:8]}"))


if __name__ == "__main__":
    main()
