"""Recompute held-out metrics for the revised punctuation checkpoint."""
import argparse
import json

from torch.utils.data import DataLoader
from transformers import BertForTokenClassification, BertTokenizerFast

from train_punct import BASE, DEVICE, OUT, PunctDataset, collate, evaluate, read_rows, unique_rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=24)
    options = parser.parse_args()
    model_dir = OUT / "bert_punct"
    tokenizer = BertTokenizerFast.from_pretrained(model_dir)
    model = BertForTokenClassification.from_pretrained(model_dir).to(DEVICE)
    seen = set()
    for filename in ("train.jsonl", "test.jsonl"):
        unique_rows(read_rows(BASE / "data" / filename), seen)
    rows = unique_rows(read_rows(BASE / "data" / "test_march.jsonl"), seen)
    loader = DataLoader(PunctDataset(rows, tokenizer), batch_size=options.batch_size,
                        shuffle=False, collate_fn=collate)
    print(json.dumps(evaluate(model, loader), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
