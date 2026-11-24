import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from ngram_lm import DATA_DIR, LaplaceLM, KneserNeyLM, load_tokens, train_model, generate, perplexity
import jieba

seqs, w2id = load_tokens([DATA_DIR / "01.txt"])
V = len(w2id)
print(f"1月语料: {sum(len(s) for s in seqs)} 词, V={V}")
for n in (1, 2):
    lm, tc, te = train_model(KneserNeyLM, seqs, n, V, w2id)
    print(f"KN n={n}: count={tc:.1f}s est={te:.1f}s D={lm.D:.3f} types={len(lm.counts)}")
    lm2, tc2, te2 = train_model(LaplaceLM, seqs, n, V, w2id)
    print(f"Lap n={n}: count={tc2:.1f}s est={te2:.1f}s types={len(lm2.counts)}")
    ppl = perplexity(lm, seqs[:300], n)
    print(f"  PPL(train300)={ppl:.0f}")
lm, _, _ = train_model(KneserNeyLM, seqs, 2, V, w2id)
prompt = "在阳光明媚的五月，我们学校胜利召开了"
seed = [w for w in jieba.cut(prompt) if w.strip()]
print("分词:", "/".join(seed))
t0=time.time()
out = generate(lm, seed, max_len=50, temperature=1.0, top_k=50, seed=1)
print(f"生成({time.time()-t0:.0f}s):", "".join(out))
