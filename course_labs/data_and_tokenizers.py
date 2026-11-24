"""Offline exercises for HF Datasets, fast tokenizers and vocabulary adaptation."""
import json
import tempfile
from pathlib import Path

from datasets import Dataset, load_from_disk
from sklearn.feature_extraction.text import TfidfVectorizer
from tokenizers import Regex, Tokenizer, decoders, models, normalizers, pre_tokenizers, trainers
from transformers import BertTokenizerFast

BASE = Path(__file__).parent
DATA = BASE.parent / "data"

RARE = ["朙月與明月", "𠮷與吉須保留區別", "古籍異體字㑹與會", "𫖯首而行"]
SEEDS = ["𠮷", "㑹", "𫖯", "朙"]
VOCAB_SIZE = 6000


def make_tokenizer(kind, corpus):
    if kind == "BBPE":
        tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tokenizer.decoder = decoders.ByteLevel()
        trainer = trainers.BpeTrainer(vocab_size=VOCAB_SIZE, min_frequency=2,
                                      special_tokens=["[UNK]"],
                                      initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
    elif kind == "WordPiece":
        tokenizer = Tokenizer(models.WordPiece(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
            pre_tokenizers.Split(Regex(r"\p{Han}"), behavior="isolated"),
            pre_tokenizers.BertPreTokenizer(),
        ])
        tokenizer.decoder = decoders.WordPiece()
        trainer = trainers.WordPieceTrainer(vocab_size=VOCAB_SIZE, min_frequency=2,
                                            special_tokens=["[UNK]"])
    else:
        tokenizer = Tokenizer(models.Unigram())
        tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
        trainer = trainers.UnigramTrainer(vocab_size=VOCAB_SIZE, unk_token="[UNK]",
                                          special_tokens=["[UNK]"], max_piece_length=12)
    tokenizer.normalizer = normalizers.NFC()
    tokenizer.train_from_iterator(corpus, trainer=trainer, length=len(corpus))
    return tokenizer


def describe(tokenizer, text):
    result = tokenizer.encode(text)
    unknown = tokenizer.token_to_id("[UNK]")
    return {"text": text, "tokens": result.tokens, "offsets": result.offsets,
            "count": len(result.ids), "unknown": result.ids.count(unknown),
            "roundtrip": tokenizer.decode(result.ids) == text}


def main():
    lines = (DATA / "04.txt").read_text(encoding="utf-8").splitlines()[:4200]
    dataset = Dataset.from_dict({"text": lines})
    dataset = dataset.filter(lambda row: bool(row["text"].strip()))
    dataset = dataset.map(lambda row: {"words": len(row["text"].split())})
    split = dataset.train_test_split(test_size=0.15, seed=42)
    with tempfile.TemporaryDirectory() as temporary:
        snapshot = Path(temporary) / "split"
        split.save_to_disk(str(snapshot))
        restored = load_from_disk(str(snapshot))
        assert restored["train"][0] == split["train"][0]
    corpus = list(split["train"]["text"])
    samples = ["中国加入WTO以后继续推进合作", "报告发布于1998年", "attention learns from context", *RARE]
    probe = {}
    for name in ("BBPE", "WordPiece", "Unigram"):
        tokenizer = make_tokenizer(name, corpus)
        probe[name] = {"vocab": tokenizer.get_vocab_size(),
                       "cases": [describe(tokenizer, text) for text in samples]}
        if name in ("BBPE", "WordPiece"):
            tokenizer.add_tokens(SEEDS)
            probe[name + "+4 exact entries"] = {
                "vocab": tokenizer.get_vocab_size(),
                "cases": [describe(tokenizer, text) for text in samples],
            }

    bert = BertTokenizerFast.from_pretrained("bert-base-chinese", local_files_only=True)
    adapted = bert.train_new_from_iterator(corpus, vocab_size=VOCAB_SIZE)
    context = "报告明确说明，中国将继续参与WTO相关磋商。"
    encoding = bert("中国将继续参与什么？", context, return_offsets_mapping=True)
    start = context.index("WTO")
    end = start + len("WTO")
    qa_tokens = [
        {"token": encoding.tokens()[i], "offset": [left, right],
         "source": context[left:right]}
        for i, (left, right) in enumerate(encoding["offset_mapping"])
        if encoding.sequence_ids()[i] == 1 and left < end and right > start
    ]
    sample_counts = [len(split["test"][i]["text"].split()) for i in range(len(split["test"]))]
    query = "国际合作与外交"
    vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(1, 2), min_df=2)
    vectors = vectorizer.fit_transform(corpus)
    similarities = (vectors @ vectorizer.transform([query]).T).toarray().ravel()
    nearest = similarities.argsort()[-3:][::-1]
    result = {"dataset": {"source": "data/04.txt", "train": len(split["train"]),
                          "validation": len(split["test"]), "split_seed": 42,
                          "validation_mean_words": sum(sample_counts) / len(sample_counts)},
              "setup": {"normalization": "NFC", "requested_vocabulary": VOCAB_SIZE,
                        "added_tokens": SEEDS, "qa_context": context},
              "retrieval": {"query": query, "method": "character TF-IDF cosine, local exact search",
                            "matches": [{"score": float(similarities[i]), "text": corpus[i][:160]}
                                        for i in nearest]},
              "qa_offsets": qa_tokens, "tokenizer_comparison": probe,
              "retrained_bert_tokenizer": {"vocab": len(adapted), "sample": adapted.tokenize(samples[0])}}
    path = BASE / "results" / "data_and_tokenizers.json"
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(path, {name: sum(case["unknown"] for case in row["cases"]) for name, row in probe.items()})


if __name__ == "__main__":
    main()
