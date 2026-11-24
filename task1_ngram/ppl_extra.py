import sys, json, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from ngram_lm import DATA_DIR, KneserNeyLM, load_tokens, train_model, perplexity

MONTHS = [DATA_DIR / f"{m:02d}.txt" for m in range(1, 7)]
seqs_train, w2id = load_tokens(MONTHS[:5])
test_sents = []
with open(MONTHS[5], encoding="utf-8") as f:
    for line in f:
        words = line.split()
        if words: test_sents.append(words)
unk = w2id.get("<unk>", 1)
seqs_test, n_tok = [], 0
for s in test_sents:
    if n_tok >= 200_000: break
    seqs_test.append([w2id.get(w, unk) for w in s]); n_tok += len(s)
V = len(w2id)
rows = []
for mf in (1, 2):
    t0 = time.time()
    lm, tc, te = train_model(KneserNeyLM, seqs_train, 3, V, w2id, min_freq=mf)
    ppl = perplexity(lm, seqs_test, 3)
    rows.append({"min_freq": mf, "types": len(lm.counts), "D": round(lm.D,4),
                 "PPL": round(ppl,1), "train_s": round(tc+te,1), "total_s": round(time.time()-t0,1)})
    print(rows[-1], flush=True)
out = Path(__file__).resolve().parent / "results"
out.mkdir(exist_ok=True)
(out / "ppl_trigram_prune.json").write_text(
    json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
)
