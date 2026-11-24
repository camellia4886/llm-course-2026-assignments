"""Run the Hugging Face Datasets FAISS workflow on local Chinese news."""

import json
import os
import tempfile
from pathlib import Path

import faiss
import numpy as np
import torch
from datasets import Dataset, load_dataset
from torch.nn.functional import normalize
from transformers import AutoModel, AutoTokenizer

BASE = Path(__file__).resolve().parent
CORPUS = BASE.parent / "data" / "04.txt"
QUERIES = ("国际合作与外交", "农业科技与农民收入", "学校开展素质教育")
SIZE = 512
MAX_LENGTH = 128
TOP_K = 3


@torch.inference_mode()
def embed(texts, model, tokenizer):
    vectors = []
    for start in range(0, len(texts), 16):
        batch = tokenizer(texts[start:start + 16], max_length=MAX_LENGTH, padding=True,
                          truncation=True, return_tensors="pt",
                          return_special_tokens_mask=True)
        mask = (batch["attention_mask"].bool() & ~batch.pop("special_tokens_mask").bool())
        hidden = model(**batch).last_hidden_state
        pooled = (hidden * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp_min(1)
        vectors.append(normalize(pooled, dim=1).numpy().astype(np.float32))
    return np.ascontiguousarray(np.concatenate(vectors))


def main():
    torch.set_num_threads(min(torch.get_num_threads(), 4))
    model_id = os.environ.get("BERT_BASE_MODEL", "bert-base-chinese")
    tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
    model = AutoModel.from_pretrained(model_id, local_files_only=True).eval()
    with tempfile.TemporaryDirectory() as cache:
        stream = load_dataset("text", data_files={"train": str(CORPUS)},
                              split="train", streaming=True, cache_dir=cache)
        texts = ["".join(row["text"].split()) for row in stream.take(SIZE)]
    assert len(texts) == SIZE and all(texts)
    vectors = embed(texts, model, tokenizer)
    queries = embed(list(QUERIES), model, tokenizer)
    dataset = Dataset.from_dict({"row": list(range(SIZE)), "text": texts})
    dataset.add_faiss_index_from_external_arrays(
        vectors, index_name="embeddings", metric_type=faiss.METRIC_INNER_PRODUCT)

    results = []
    with tempfile.TemporaryDirectory() as directory:
        index = str(Path(directory) / "vectors.faiss")
        dataset.save_faiss_index("embeddings", index)
        dataset.drop_index("embeddings")
        dataset.load_faiss_index("embeddings", index)
        for text, query in zip(QUERIES, queries, strict=True):
            scores, matches = dataset.get_nearest_examples("embeddings", query, k=TOP_K)
            reference = vectors @ query
            expected = np.argsort(-reference)[:TOP_K]
            assert matches["row"] == expected.tolist()
            np.testing.assert_allclose(scores, reference[expected], atol=1e-6)
            results.append({"query": text, "matches": [
                {"row": row, "cosine": float(score), "excerpt": passage[:90]}
                for row, score, passage in zip(matches["row"], scores, matches["text"],
                                                strict=True)
            ]})
    output = {
        "corpus": "data/04.txt",
        "rows": SIZE,
        "read_mode": "datasets streaming, first 512 records",
        "model": Path(model_id).name,
        "embedding": "mean of non-padding, non-special final-layer tokens; L2 normalized",
        "max_tokens": MAX_LENGTH,
        "embedding_dimension": vectors.shape[1],
        "index": "FAISS flat inner product",
        "faiss_version": faiss.__version__,
        "saved_index_reloaded": True,
        "top_k_ids_and_scores_match_numpy": True,
        "queries": results,
    }
    path = BASE / "results" / "faiss_search.json"
    path.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(path)


if __name__ == "__main__":
    main()
