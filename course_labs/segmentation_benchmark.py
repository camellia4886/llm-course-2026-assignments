"""Evaluate paragraph boundaries and QA evidence retrieval on DRCD and SQuAD."""

import hashlib
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from statistics import median

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import precision_recall_curve, precision_recall_fscore_support

BASE = Path(__file__).resolve().parent
DATA = BASE.parent / "data" / "public_benchmarks"
SOURCES = {
    "zh": ("DRCD_dev.json", "e236df03861ba241bd7b29628cd48b3b8e339c60137e041a4358b5320ed61928"),
    "en": ("squad_dev_v11.json", "95aa6a52d5d6a735563366753ca50492a658031da74f301ac5238b03966972c9"),
}
SEED = 79
TOP_K = (1, 3, 5)
ZH_SENTENCE_END = re.compile(r'[。！？!?]+[”’」』）)]*')
EN_SENTENCE_END = re.compile(r'[.!?]+["”’\')]*\s+(?=["“‘(A-Z0-9])')
ABBREVIATION = re.compile(r'\b(?:(?:[A-Za-z]\.)+|Mr\.|Mrs\.|Dr\.|St\.|Prof\.|Sr\.|Jr\.)$')


@dataclass
class Document:
    identifier: str
    text: str
    starts: list[int]
    paragraph_cuts: set[int]
    questions: list
    paragraph_lengths: list[int]

    def sentences(self):
        return [self.text[a:b].strip() for a, b in zip(self.starts, self.starts[1:])]


def sentence_ends(text, language):
    pattern = ZH_SENTENCE_END if language == "zh" else EN_SENTENCE_END
    for match in pattern.finditer(text):
        if match.end() < len(text) and text[match.end():].strip():
            if language == "en" and ABBREVIATION.search(text[:match.end()].rstrip()):
                continue
            yield match.end()
    yield len(text)


def make_document(article, language, index):
    paragraphs = [p for p in article["paragraphs"] if p["context"].strip()]
    contexts = [p["context"] for p in paragraphs]
    starts = [0]
    cuts = set()
    offset = 0
    questions = []
    for paragraph, context in zip(paragraphs, contexts, strict=True):
        starts.extend(offset + end for end in sentence_ends(context, language))
        cuts.add(len(starts) - 1)
        for question in paragraph["qas"]:
            valid = set()
            for entry in question["answers"]:
                start = entry["answer_start"]
                answer = entry["text"]
                if answer and isinstance(start, int) and context[start:start + len(answer)] == answer:
                    valid.add((offset + start, offset + start + len(answer)))
            questions.append((question["id"], question["question"], sorted(valid)))
        offset += len(context) + 1
    cuts.discard(len(starts) - 1)
    return Document(str(article.get("id", index)), "\n".join(contexts), starts, cuts,
                    questions, list(map(len, contexts)))


def vectorizer(language):
    if language == "zh":
        return TfidfVectorizer(analyzer="char", ngram_range=(2, 3), sublinear_tf=True)
    return TfidfVectorizer(analyzer="word", ngram_range=(1, 2), sublinear_tf=True,
                           token_pattern=r"(?u)\b\w+\b")


def gap_features(doc, vectors):
    sentences = doc.sentences()
    left = [" ".join(sentences[max(0, i - 2):i]) for i in range(1, len(sentences))]
    right = [" ".join(sentences[i:i + 2]) for i in range(1, len(sentences))]
    a, b = vectors.transform(left), vectors.transform(right)
    return (1 - np.asarray(a.multiply(b).sum(axis=1)).ravel()).clip(0, 1)


def labels(doc):
    return [int(i in doc.paragraph_cuts) for i in range(1, len(doc.starts) - 1)]


def chunks(doc, cuts, cap):
    result = []
    first = 0
    units = len(doc.starts) - 1
    for i in range(1, units + 1):
        if i - 1 > first and doc.starts[i] - doc.starts[first] > cap:
            result.append((doc.starts[first], doc.starts[i - 1]))
            first = i - 1
        if i in cuts or i == units:
            result.append((doc.starts[first], doc.starts[i]))
            first = i
    return result


def boundary_metrics(docs, predicted):
    gold = []
    proposals = []
    bad_windows = windows = 0
    for doc, cuts in zip(docs, predicted, strict=True):
        reference = labels(doc)
        estimate = [int(i in cuts) for i in range(1, len(doc.starts) - 1)]
        gold.extend(reference)
        proposals.extend(estimate)
        k = max(1, round((len(doc.starts) - 1) / (2 * (len(doc.paragraph_cuts) + 1))))
        windows += max(0, len(reference) - k + 1)
        bad_windows += sum(sum(reference[i:i + k]) != sum(estimate[i:i + k])
                           for i in range(len(reference) - k + 1))
    precision, recall, f1, _ = precision_recall_fscore_support(
        gold, proposals, average="binary", zero_division=0)
    return {"precision": precision, "recall": recall, "boundary_f1": f1,
            "windowdiff": bad_windows / windows, "gaps": len(gold),
            "gold_boundaries": sum(gold), "predicted_boundaries": sum(proposals)}


def retrieval(docs, chunks_by_doc, search, question_rows, read_budget):
    rows = [(doc_id, start, end) for doc_id, ranges in enumerate(chunks_by_doc)
            for start, end in ranges]
    texts = [docs[doc_id].text[start:end] for doc_id, start, end in rows]
    index = search.transform(texts)
    query_vectors = search.transform([q[2] for q in question_rows])
    recall = dict.fromkeys(TOP_K, 0)
    mrr = retrieved_chars = covered = budget_hits = 0
    errors = []
    for n, (doc_id, qid, _, answers) in enumerate(question_rows):
        def contains(row):
            candidate_id, start, end = row
            return candidate_id == doc_id and any(start <= a and b <= end for a, b in answers)

        oracle = any(contains((doc_id, start, end)) for start, end in chunks_by_doc[doc_id])
        covered += oracle
        scores = (index @ query_vectors[n].T).toarray().ravel()
        candidates = np.argsort(-scores, kind="stable")[:max(TOP_K)]
        ranked = [rows[i] for i in candidates if scores[i] > 0]
        remaining = read_budget
        for candidate_id, start, end in ranked:
            read_end = min(end, start + remaining)
            if contains((candidate_id, start, read_end)):
                budget_hits += 1
                break
            remaining -= read_end - start
            if remaining == 0:
                break
        retrieved_chars += sum(end - start for _, start, end in ranked[:3])
        rank = next((i for i, row in enumerate(ranked, 1) if contains(row)), None)
        if rank is not None:
            mrr += 1 / rank
            for k in TOP_K:
                recall[k] += rank <= k
        elif len(errors) < 8:
            errors.append({"doc": docs[doc_id].identifier, "question_id": qid,
                           "span_covered_by_a_chunk": oracle, "gold_spans": answers})
    size = len(question_rows)
    return {"chunks": len(rows), "queries": size,
            "mean_chunk_chars": sum(map(len, texts)) / len(rows),
            "mean_chars_top_3": retrieved_chars / size,
            "oracle_span_coverage": covered / size,
            "recall": {k: v / size for k, v in recall.items()},
            "recall_at_char_budget_top_5": budget_hits / size,
            "mrr_at_5": mrr / size, "example_misses": errors}


def benchmark(language, filename, expected_hash):
    data = (DATA / filename).read_bytes()
    assert hashlib.sha256(data).hexdigest() == expected_hash, filename
    articles = json.loads(data.decode("utf-8-sig"))["data"]
    docs = [make_document(article, language, i) for i, article in enumerate(articles)]
    docs = [doc for doc in docs if doc.paragraph_cuts]
    random.Random(SEED).shuffle(docs)
    cutoff = 2 * len(docs) // 3
    dev, test = docs[:cutoff], docs[cutoff:]
    boundary_vectorizer = vectorizer(language).fit([s for doc in dev for s in doc.sentences()])
    dev_scores = np.concatenate([gap_features(doc, boundary_vectorizer) for doc in dev])
    dev_gold = [score for doc in dev for score in labels(doc)]
    p, r, thresholds = precision_recall_curve(dev_gold, dev_scores)
    f1 = 2 * p[:-1] * r[:-1] / np.maximum(p[:-1] + r[:-1], np.finfo(float).eps)
    threshold = float(thresholds[f1.argmax()])
    cap = int(median([size for doc in dev for size in doc.paragraph_lengths]))

    lexical_cuts = [{i for i, value in enumerate(gap_features(doc, boundary_vectorizer), 1)
                     if value >= threshold} for doc in test]
    cap_chunks = [chunks(doc, set(), cap) for doc in test]
    fixed_cuts = [{doc.starts.index(end) for _, end in ranges[:-1]}
                  for doc, ranges in zip(test, cap_chunks, strict=True)]
    lexical_chunks = [chunks(doc, cuts, cap) for doc, cuts in zip(test, lexical_cuts, strict=True)]
    natural_chunks = [chunks(doc, doc.paragraph_cuts, cap) for doc in test]
    search = vectorizer(language).fit([doc.text for doc in test])
    question_rows = [(doc_id, qid, text, answers) for doc_id, doc in enumerate(test)
                     for qid, text, answers in doc.questions if answers and text.strip()]
    return {
        "source": filename, "sha256": expected_hash,
        "development_documents": len(dev), "test_documents": len(test),
        "test_document_ids": [doc.identifier for doc in test],
        "test_paragraphs": sum(len(doc.paragraph_lengths) for doc in test),
        "excluded_test_questions": sum(len(doc.questions) for doc in test) - len(question_rows),
        "threshold": threshold, "soft_char_limit_from_development": cap,
        "read_budget_chars": 2 * cap,
        "boundary": {"fixed_length": boundary_metrics(test, fixed_cuts),
                     "lexical_score": boundary_metrics(test, lexical_cuts)},
        "retrieval": {
            name: retrieval(test, candidates, search, question_rows, 2 * cap)
            for name, candidates in [("fixed_length", cap_chunks),
                                     ("lexical_score", lexical_chunks),
                                     ("editor_paragraph", natural_chunks)]
        },
    }


def main():
    result = {"split_seed": SEED, "languages": {
        language: benchmark(language, *source) for language, source in SOURCES.items()
    }}
    out = BASE / "results" / "segmentation_benchmark.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for language, row in result["languages"].items():
        print(language, row["boundary"],
              {name: m["recall"] for name, m in row["retrieval"].items()}, flush=True)


if __name__ == "__main__":
    main()
