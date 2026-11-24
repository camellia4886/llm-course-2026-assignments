# -*- coding: utf-8 -*-
"""n-gram 作业主实验：训练时间对比 / 困惑度评测 / 文本生成。

输出：
  - results/train_time.tsv     训练时间与规模统计
  - results/ppl.tsv            测试集困惑度
  - results/generation.txt     各种参数下的续写文本
"""
import json
import time
import sys
from pathlib import Path

import jieba

sys.path.insert(0, str(Path(__file__).parent))
from ngram_lm import (DATA_DIR, LaplaceLM, KneserNeyLM, generate, load_tokens,
                      perplexity, train_model)

OUT = Path(__file__).parent / "results"
OUT.mkdir(exist_ok=True)

MONTHS = [DATA_DIR / f"{m:02d}.txt" for m in range(1, 7)]


def log(msg):
    print(msg, flush=True)


def phase_a():
    """训练时间与规模随 n、语料大小、平滑方法的变化。"""
    rows = []
    sizes = {"1个月": MONTHS[:1], "3个月": MONTHS[:3], "6个月": MONTHS[:6]}
    for name, files in sizes.items():
        seqs, w2id = load_tokens(files)
        V = len(w2id)
        words = sum(len(s) for s in seqs)
        log(f"== {name}: {words} 词, V={V}")
        for n in (1, 2, 3):
            for cls in (LaplaceLM, KneserNeyLM):
                thresholds = (1, 2) if n == 3 and name == "6个月" else (1,)
                for min_freq in thresholds:
                    t0 = time.time()
                    lm, t_count, t_est = train_model(cls, seqs, n, V, w2id,
                                                     min_freq=min_freq)
                    wall = time.time() - t0
                    rows.append({
                        "语料": name, "词数": words, "n": n,
                        "平滑": cls.__name__, "V": V,
                        "计数耗时s": round(t_count, 2),
                        "估计耗时s": round(t_est, 2),
                        "总耗时s": round(wall, 2),
                        "n-gram种类": len(lm.counts),
                        "D": round(getattr(lm, "D", -1), 4),
                        "min_freq": min_freq,
                    })
                    log(f"   n={n} {cls.__name__} min_freq={min_freq}: 计数{t_count:.1f}s "
                        f"估计{t_est:.1f}s 种类{len(lm.counts)}")
                    del lm
        del seqs
    with open(OUT / "train_time.json", "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    with open(OUT / "train_time.tsv", "w", encoding="utf-8") as f:
        keys = ["语料", "词数", "n", "平滑", "V", "计数耗时s", "估计耗时s",
                "总耗时s", "n-gram种类", "D", "min_freq"]
        f.write("\t".join(keys) + "\n")
        for r in rows:
            f.write("\t".join(str(r[k]) for k in keys) + "\n")
    log("phase_a done")


def phase_b():
    """用 01-05 训练，06 作测试集测困惑度。"""
    seqs_train, w2id = load_tokens(MONTHS[:5])
    # 测试集按词面重新映射 id，取前 20 万词控制评测时间
    test_sents = []
    with open(MONTHS[5], encoding="utf-8") as f:
        for line in f:
            words = line.split()
            if words:
                test_sents.append(words)
    unk = w2id.get("<unk>", 1)
    seqs_test = []
    n_tok = 0
    for s in test_sents:
        if n_tok >= 200_000:
            break
        seqs_test.append([w2id.get(w, unk) for w in s])
        n_tok += len(s)
    V = len(w2id)
    rows = []
    for n in (1, 2, 3):
        for cls in (LaplaceLM, KneserNeyLM):
            t0 = time.time()
            lm, t_count, t_est = train_model(cls, seqs_train, n, V, w2id,
                                             min_freq=2 if n == 3 else 1)
            ppl = perplexity(lm, seqs_test, n)
            wall = time.time() - t0
            rows.append({"n": n, "平滑": cls.__name__, "PPL": round(ppl, 1),
                         "训练s": round(t_count + t_est, 1), "总s": round(wall, 1)})
            log(f"   n={n} {cls.__name__}: PPL={ppl:.1f} "
                f"(训练{t_count + t_est:.0f}s 评测{wall - t_count - t_est:.0f}s)")
    with open(OUT / "ppl.json", "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    with open(OUT / "ppl.tsv", "w", encoding="utf-8") as f:
        f.write("n\t平滑\tPPL\t训练s\t总s\n")
        for r in rows:
            f.write(f"{r['n']}\t{r['平滑']}\t{r['PPL']}\t{r['训练s']}\t{r['总s']}\n")
    log("phase_b done")


def phase_c():
    """不同 n、平滑、温度下的续写生成。"""
    prompt = "在阳光明媚的五月，我们学校胜利召开了"
    seed_words = [w for w in jieba.cut(prompt) if w.strip()]
    log(f"提示词分词: {' / '.join(seed_words)}")
    lines = [f"提示词：{prompt}", f"分词：{' / '.join(seed_words)}", ""]
    for name, files in {"1个月": MONTHS[:1], "6个月": MONTHS[:6]}.items():
        seqs, w2id = load_tokens(files)
        V = len(w2id)
        for n in (1, 2, 3):
            lm, _, _ = train_model(KneserNeyLM, seqs, n, V, w2id,
                                   min_freq=2 if (n == 3 and name == "6个月") else 1)
            for temp in (0.7, 1.0, 1.3):
                for sd in (1, 2):
                    t0 = time.time()
                    out = generate(lm, seed_words, max_len=80, temperature=temp,
                                   top_k=50, seed=sd)
                    text = "".join(out)
                    lines.append(f"[{name} | n={n} | KN | T={temp} | seed={sd} | "
                                 f"{time.time() - t0:.0f}s]")
                    lines.append(text)
                    lines.append("")
                    log(f"   {name} n={n} T={temp} seed={sd}: {text[:40]}...")
        # Laplace 生成一组对照
        lm, _, _ = train_model(LaplaceLM, seqs, 3, V, w2id)
        out = generate(lm, seed_words, max_len=80, temperature=1.0, top_k=50, seed=1)
        lines.append(f"[{name} | n=3 | Laplace | T=1.0 | seed=1]")
        lines.append("".join(out))
        lines.append("")
        log(f"   {name} Laplace n=3 done")
    with open(OUT / "generation.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    log("phase_c done")


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("a", "all"):
        phase_a()
    if which in ("b", "all"):
        phase_b()
    if which in ("c", "all"):
        phase_c()
    log("ALL DONE")
