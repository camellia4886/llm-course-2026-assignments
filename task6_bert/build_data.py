"""人民日报语料 -> 断句标点序列标注数据。

标签（每字之后应插入的标点）:
0=O  1=，  2=。  3=、  4=；  5=：  6=？  7=！
引号/书名号/括号等不参与预测，直接从输入中去掉。
一月训练，二月验证，三月独立测试。test.jsonl 保留旧文件名供兼容。
"""
import json
import random
from pathlib import Path

DATA = Path(__file__).resolve().parents[1] / "data"
OUT = Path(__file__).parent / "data"
OUT.mkdir(exist_ok=True)

PUNCT2ID = {"，": 1, "。": 2, "、": 3, "；": 4, "：": 5, "？": 6, "！": 7}
ID2PUNCT = {v: k for k, v in PUNCT2ID.items()}
# 所有要剥离的标点字符（含引号书名号括号等）
STRIP = set("，。、；：？！“”‘’《》〈〉（）【】…—·•・,.;:?!\"'()[]<>-~～…　　")


def build(in_file, out_file, max_len=250):
    rows = []
    with open(in_file, encoding="utf-8") as f:
        for line in f:
            tokens = line.split()
            chars, labels = [], []
            for tok in tokens:
                if tok and all(c in STRIP for c in tok):
                    if tok in PUNCT2ID and labels and labels[-1] == 0:
                        labels[-1] = PUNCT2ID[tok]
                    continue
                for c in tok:
                    if c in STRIP:
                        if c in PUNCT2ID and labels and labels[-1] == 0:
                            labels[-1] = PUNCT2ID[c]
                        continue
                    chars.append(c)
                    labels.append(0)
            if len(chars) >= 8 and len(labels) == len(chars):
                chars = chars[:max_len]
                labels = labels[:max_len]
                rows.append({"input": "".join(chars), "labels": labels})
    random.Random(42).shuffle(rows)
    with open(out_file, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return len(rows)


if __name__ == "__main__":
    n = build(DATA / "01.txt", OUT / "train.jsonl")
    m = build(DATA / "02.txt", OUT / "test.jsonl")
    print(f"train={n} test={m}")
    k = build(DATA / "03.txt", OUT / "test_march.jsonl")
    print(f"独立测试原始样本={k}")
    # 标签分布统计
    from collections import Counter
    for name in ("train", "test", "test_march"):
        c = Counter()
        for l in open(OUT / f"{name}.jsonl", encoding="utf-8"):
            c.update(json.loads(l)["labels"])
        dist = {ID2PUNCT.get(k, "O"): v for k, v in c.most_common()}
        print(name, "labels:", dist)
