"""Run the standalone FastText supervised learner on January weak labels."""
import json
import tempfile
import time
from pathlib import Path

import fasttext
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

BASE = Path(__file__).parent


def main():
    rows = [json.loads(line) for line in (BASE / "articles_9801.jsonl").read_text(encoding="utf-8").splitlines()]
    used = ["经济", "政务", "文化", "外交", "科技", "体育"]
    labels = {name: f"__label__{index}" for index, name in enumerate(used)}
    rows = [row for row in rows if row["label"] in labels]
    train, test = train_test_split(rows, test_size=0.2, stratify=[row["label"] for row in rows], random_state=42)
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "train.txt"
        path.write_text("".join(labels[row["label"]] + " " + " ".join(row["text"].split()) + "\n"
                                for row in train), encoding="utf-8")
        params = dict(dim=64, lr=0.25, epoch=15, wordNgrams=2, minCount=2,
                      bucket=100000, thread=1, verbose=0)
        start = time.perf_counter()
        model = fasttext.train_supervised(str(path), **params)
        elapsed = time.perf_counter() - start
    predicted, _ = model.predict([" ".join(row["text"].split()) for row in test])
    predicted = [row[0] for row in predicted]
    truth = [labels[row["label"]] for row in test]
    result = {"implementation": "facebookresearch/fastText via fasttext-wheel 0.9.2",
              "train": len(train), "test": len(test), "label_source": "January 1998 weak labels",
              "seed_split": 42, "settings": params, "training_seconds": elapsed,
              "test_macro_f1": f1_score(truth, predicted, average="macro"),
              "test_accuracy": sum(a == b for a, b in zip(predicted, truth)) / len(test)}
    (BASE / "results" / "official_fasttext.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(result)


if __name__ == "__main__":
    main()
