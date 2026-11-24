# -*- coding: utf-8 -*-
"""Gensim 表示学习 + FastText 文本分类实验。

1. 在 CMRC 2018 独立语料，或在人民日报 98 年 4–6 月训练表示
2. 表示学习检验：相似词、类比、FastText 的 OOV 能力
3. 文本分类：文章平均向量 + 逻辑回归/线性 SVM，分类数 2 -> 4 -> 6 逐步展开
"""
import argparse
import hashlib
import json
import re
import time
from collections import Counter
from pathlib import Path

import numpy as np
import jieba
from gensim.models import FastText, Word2Vec
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

DATA = Path(__file__).resolve().parents[1] / "data"
ART = Path(__file__).parent / "articles_9801.jsonl"
OUT = Path(__file__).parent / "results"
OUT.mkdir(exist_ok=True)


def read_sentences(files):
    sents = []
    for f in files:
        with open(f, encoding="utf-8") as fin:
            for line in fin:
                words = line.split()
                if words:
                    sents.append(words)
    return sents


def cmrc_sentences(path):
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    paragraphs = dict.fromkeys(p["context"] for d in data["data"] for p in d["paragraphs"])
    sentences = []
    for paragraph in paragraphs:
        for line in re.split(r"(?<=[。！？!?])\s*", paragraph):
            tokens = [word for word in jieba.cut(line, HMM=False) if word.strip()]
            if tokens:
                sentences.append(tokens)
    return sentences, len(paragraphs)


def train_embeddings(save_models=False, cmrc_path=None, revision="v2"):
    january = {tuple(sentence) for sentence in read_sentences([DATA / "01.txt"])}
    if cmrc_path is None:
        files = [DATA / f"{month:02d}.txt" for month in (4, 5, 6)]
        candidates = read_sentences(files)
        source = {"months": [4, 5, 6]}
    else:
        candidates, paragraph_count = cmrc_sentences(cmrc_path)
        source = {"dataset": "CMRC 2018 training contexts",
                  "source_file": cmrc_path.name,
                  "source_sha256": hashlib.sha256(cmrc_path.read_bytes()).hexdigest(),
                  "unique_paragraphs": paragraph_count,
                  "tokenizer": f"jieba {jieba.__version__}, HMM=False"}
    sentences = [sentence for sentence in candidates if tuple(sentence) not in january]
    common = dict(vector_size=100, window=5, min_count=5, sg=1,
                  workers=4, epochs=4, seed=42)
    t0 = time.time()
    w2v = Word2Vec(sentences, **common)
    t_w2v = time.time() - t0
    t0 = time.time()
    ft = FastText(sentences, min_n=2, max_n=4, bucket=300_000, **common)
    t_ft = time.time() - t0
    if save_models:
        w2v.save(str(OUT / f"w2v_{revision}.model"))
        ft.save(str(OUT / f"ft_{revision}.model"))
    print(f"Word2Vec: {t_w2v:.1f}s, vocab={len(w2v.wv)}")
    print(f"FastText: {t_ft:.1f}s, vocab={len(ft.wv)}")
    details = {**source, "excluded_exact_january_sentences": True,
               "dropped_exact_january_sentences": len(candidates) - len(sentences),
               "vocabulary": len(w2v.wv),
               "sentences": len(sentences), "tokens": sum(map(len, sentences)),
               "seconds": {"w2v": t_w2v, "ft": t_ft}, "configuration": common,
               "fasttext_min_n": 2, "fasttext_max_n": 4, "fasttext_bucket": 300_000}
    return w2v, ft, details


def similarity_demos(w2v, ft):
    demos = {
        "w2v_sim": [],
        "ft_sim": [],
        "ft_oov": [],
        "analogy": [],
    }
    for w in ["经济", "改革", "体育", "中国", "发展"]:
        if w in w2v.wv:
            demos["w2v_sim"].append([w, w2v.wv.most_similar(w, topn=5)])
    for w in ["足球", "电脑", "金融"]:
        if w in ft.wv.key_to_index:
            demos["ft_sim"].append([w, ft.wv.most_similar(w, topn=5)])
    # FastText OOV 能力：模型没见过的词（打错字/新词），靠子词拼出向量
    for w in ["改革开放新", "电子计算机化", "足球比赛"]:
        if w not in ft.wv.key_to_index:
            demos["ft_oov"].append([w, ft.wv.most_similar(w, topn=5)])
    # 类比：北京:中国 :: 巴黎:法国 之类
    pairs = [("北京", "中国", "巴黎"), ("北京", "中国", "东京"),
             ("中国", "北京", "法国")]
    for a, b, c in pairs:
        if all(x in w2v.wv for x in (a, b, c)):
            try:
                res = w2v.wv.most_similar(positive=[c, b], negative=[a], topn=3)
                demos["analogy"].append([f"{a}:{b} :: {c}:?", res])
            except KeyError:
                pass
    return demos


def doc_vector(model, text):
    words = [w for w in text.split() if w in model.wv.key_to_index]
    if not words:
        return np.zeros(model.vector_size)
    return np.mean(model.wv[words], axis=0)


def classify(model, n_classes, top_labels, measure_coverage=False):
    rows = [json.loads(l) for l in open(ART, encoding="utf-8")]
    sub = [r for r in rows if r["label"] in top_labels]
    X = np.vstack([doc_vector(model, r["text"]) for r in sub])
    y = np.array([r["label"] for r in sub])
    coverage = {}
    if measure_coverage:
        articles = [r["text"].split() for r in sub]
        known = [sum(word in model.wv.key_to_index for word in words) for words in articles]
        coverage = {"known_token_ratio": sum(known) / sum(map(len, articles)),
                    "documents_without_in_vocab_tokens": known.count(0)}
    results = []
    for name, clf in [("LR", make_pipeline(StandardScaler(), LogisticRegression(
            max_iter=1000, C=1.0))),
                      ("SVM", make_pipeline(StandardScaler(), LinearSVC(C=1.0)))]:
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        f1 = cross_val_score(clf, X, y, cv=cv, scoring="f1_macro")
        results.append({**coverage, "classes": n_classes, "labels": top_labels,
                        "clf": name, "n": len(sub),
                        "f1_mean": round(float(f1.mean()), 4),
                        "f1_std": round(float(f1.std()), 4)})
        print(f"  {n_classes}类 {name}: n={len(sub)} F1={f1.mean():.3f}±{f1.std():.3f}")
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--save-models", action="store_true")
    parser.add_argument("--cmrc-train", type=Path,
                        help="train on independent CMRC contexts and write v3 results")
    args = parser.parse_args()
    revision = "v3" if args.cmrc_train else "v2"
    w2v, ft, details = train_embeddings(args.save_models, args.cmrc_train, revision)
    demos = similarity_demos(w2v, ft)
    json.dump({"training": details, "demos": demos}, open(OUT / f"demos_{revision}.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print("demos saved")
    all_results = []
    # 由少到多：2 类 -> 4 类 -> 6 类（剔除"其他"，仅用主题类）
    label_counts = Counter(json.loads(l)["label"] for l in open(ART, encoding="utf-8"))
    theme = [c for c, n in label_counts.most_common() if c != "其他"]
    for k in (2, 4, 6):
        labels = theme[:k]
        print(f"== {k} 类: {labels}")
        for model, mname in ((w2v, "w2v"), (ft, "ft")):
            res = classify(model, k, labels, measure_coverage=bool(args.cmrc_train))
            for r in res:
                r["vec"] = mname
                all_results.append(r)
    json.dump(all_results, open(OUT / f"classification_{revision}.json", "w",
                                encoding="utf-8"), ensure_ascii=False, indent=2)
    print("classification done")


if __name__ == "__main__":
    main()
