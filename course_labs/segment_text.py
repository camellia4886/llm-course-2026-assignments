"""Estimate topic boundaries from paired news passages, with article-level holdout."""
import json
import random
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

BASE = Path(__file__).parent
ARTICLES = BASE.parent / "task4_gensim" / "articles_9801.jsonl"
WINDOW = 45


def article_passages():
    records = []
    for line in ARTICLES.read_text(encoding="utf-8").splitlines():
        article = json.loads(line)
        tokens = article["text"].split()
        if len(tokens) >= 2 * WINDOW:
            records.append((" ".join(tokens[:WINDOW]), " ".join(tokens[WINDOW:2 * WINDOW])))
    random.Random(42).shuffle(records)
    return records


def pairs(articles):
    samples = []
    for index, (first, second) in enumerate(articles):
        samples.append((first, second, 0))
        samples.append((first, articles[(index + 1) % len(articles)][1], 1))
    return samples


def feature(vectorizer, samples):
    left = vectorizer.transform([sample[0] for sample in samples])
    right = vectorizer.transform([sample[1] for sample in samples])
    similarity = np.asarray(left.multiply(right).sum(axis=1)).ravel()
    return similarity.reshape(-1, 1)


def main():
    articles = article_passages()
    cutoff = len(articles) * 4 // 5
    train_articles, test_articles = articles[:cutoff], articles[cutoff:]
    train, test = pairs(train_articles), pairs(test_articles)
    model = TfidfVectorizer(analyzer="char", ngram_range=(1, 2), min_df=3,
                            max_features=18000, sublinear_tf=True)
    model.fit([chunk for article in train_articles for chunk in article])
    x, y = feature(model, train), [row[2] for row in train]
    x_test, y_test = feature(model, test), [row[2] for row in test]
    classifier = LogisticRegression(random_state=42).fit(x, y)
    predictions = {"always_boundary": np.ones(len(test), dtype=int),
                   "median_similarity": (x_test.ravel() < np.median(x)).astype(int),
                   "logistic": classifier.predict(x_test)}
    scores = {name: classification_report(y_test, labels, output_dict=True, zero_division=0)
              for name, labels in predictions.items()}
    failures = [{"left": sample[0][-95:], "right": sample[1][:95:],
                 "gold": sample[2], "prediction": int(pred), "cosine": float(similarity[0])}
                for sample, pred, similarity in zip(test, predictions["logistic"], x_test)
                if sample[2] != pred][:10]
    result = {"window_words": WINDOW, "train_articles": len(train_articles),
              "heldout_articles": len(test_articles), "scores": scores, "errors": failures}
    (BASE / "results").mkdir(exist_ok=True)
    (BASE / "results" / "segmentation.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    print({name: result["macro avg"]["f1-score"] for name, result in scores.items()})


if __name__ == "__main__":
    main()
