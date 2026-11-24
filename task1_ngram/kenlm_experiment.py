"""Train KenLM trigrams on the existing monthly corpus and evaluate held-out June."""

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from time import perf_counter

BASE = Path(__file__).resolve().parent
DATA = BASE.parent / "data"
RESULT = BASE / "results" / "kenlm.json"


def prepare_training_data(months, corpus):
    words = sentences = 0
    with corpus.open("wb") as output:
        for month in range(1, months + 1):
            with (DATA / f"{month:02d}.txt").open("rb") as source:
                for line in source:
                    tokens = line.split()
                    if not tokens:
                        continue
                    output.write(b" ".join(tokens) + b"\n")
                    words += len(tokens)
                    sentences += 1
    return words, sentences


def evaluate(query, model_path):
    words = sentences = 0
    heldout = model_path.with_name("heldout.txt")
    with (DATA / "06.txt").open(encoding="utf-8") as source, heldout.open("wb") as out:
        for line in source:
            if words >= 200_000:
                break
            tokens = line.split()
            if not tokens:
                continue
            out.write((" ".join(tokens) + "\n").encode("utf-8"))
            words += len(tokens)
            sentences += 1
    # On Windows, KenLM expects a seekable input and a relative ASCII model path.
    with heldout.open("rb") as source:
        completed = subprocess.run(
            [str(query), "-v", "summary", model_path.name], cwd=model_path.parent,
            stdin=source, stdout=subprocess.PIPE, text=True, encoding="utf-8", check=True)
    stats = {name.strip(): value.strip() for name, value in
             (line.split(":", 1) for line in completed.stdout.splitlines() if ":" in line)}
    positions = int(stats["Tokens"])
    oov = int(stats["OOVs"])
    assert positions == words + sentences, stats
    return {
        "test_month": "06",
        "word_limit_rule": "stop after the first full sentence reaching 200000 words",
        "words": words,
        "sentences": sentences,
        "scored_positions_including_eos": positions,
        "oov_words": oov,
        "oov_rate": oov / words,
        "perplexity_including_oov_and_eos": float(stats["Perplexity including OOVs"]),
        "perplexity_excluding_oov": float(stats["Perplexity excluding OOVs"]),
    }


def train(executable, months, work, options):
    corpus = work / "train.txt"
    arpa = work / "model.arpa"
    words, sentences = prepare_training_data(months, corpus)
    started = perf_counter()
    with corpus.open("rb") as source, arpa.open("wb") as out:
        subprocess.run([str(executable), *options], cwd=work, stdin=source, stdout=out,
                       check=True)
    elapsed = perf_counter() - started
    counts = {}
    with arpa.open("rb") as model:
        for line in model:
            if line.strip() == b"\\1-grams:":
                break
            match = re.fullmatch(rb"ngram (\d+)=(\d+)", line.strip())
            if match:
                counts[int(match[1])] = int(match[2])
    assert set(counts) == {1, 2, 3}, counts
    return {
        "months": list(range(1, months + 1)),
        "words": words,
        "sentences": sentences,
        "training_seconds": round(elapsed, 3),
        "arpa_bytes": arpa.stat().st_size,
        "arpa_ngram_counts": counts,
    }, arpa


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lmplz", default=shutil.which("lmplz") or shutil.which("lmplz.exe"))
    parser.add_argument("--query", type=Path, help="KenLM query executable, defaults to lmplz's directory")
    parser.add_argument("--memory", default="512M")
    parser.add_argument("--output", type=Path, default=RESULT)
    args = parser.parse_args()
    if args.lmplz is None:
        parser.error("install KenLM's lmplz or pass --lmplz PATH")
    executable = Path(args.lmplz).resolve(strict=True)
    query_name = "query" + executable.suffix
    query = (args.query or executable.with_name(query_name)).resolve(strict=True)
    options = ["-o", "3", "-S", args.memory, "-T", ".", "--discount_fallback"]
    sizes = []
    with tempfile.TemporaryDirectory(prefix="kenlm-", dir=BASE) as temporary:
        work = Path(temporary)
        for months in (1, 3, 6, 5):
            row, arpa = train(executable, months, work, options)
            print(row, flush=True)
            if months == 5:
                evaluation = {**row, **evaluate(query, arpa)}
            else:
                sizes.append(row)
    result = {
        "estimator": "KenLM lmplz",
        "estimator_sha256": hashlib.sha256(executable.read_bytes()).hexdigest(),
        "options": options,
        "scorer": "KenLM query -v summary with sentence boundaries",
        "scorer_sha256": hashlib.sha256(query.read_bytes()).hexdigest(),
        "method": "trigram modified Kneser–Ney, all n-grams retained",
        "training_scale": sizes,
        "heldout_evaluation": evaluation,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                           encoding="utf-8")
    print(f"Held-out PPL {evaluation['perplexity_including_oov_and_eos']:.3f}", flush=True)


if __name__ == "__main__":
    main()
