"""Compare token/sample weighting and check conversational supervision masks."""
import json
from pathlib import Path

import torch
from torch.nn import functional as F
from transformers import AutoTokenizer

BASE = Path(__file__).parent
MODEL = BASE.parents[1] / "models" / "qwen2.5-1.5b-instruct"


def encode_turns(tok, turns):
    text = tok.apply_chat_template(turns, tokenize=False)
    encoded = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    assistant_spans = []
    cursor = 0
    for turn in turns:
        position = text.index(turn["content"], cursor)
        cursor = position + len(turn["content"])
        if turn["role"] == "assistant":
            assistant_spans.append((position, cursor))
    mask = [any(start <= left and right <= end and right > left
                for start, end in assistant_spans)
            for left, right in encoded["offset_mapping"]]
    ids = encoded["input_ids"]
    return ids, mask, [{"piece": tok.convert_ids_to_tokens(token), "supervised": belongs}
                       for token, belongs in zip(ids, mask)]


def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    conversations = [
        [{"role": "system", "content": "仅根据材料回答。"},
         {"role": "user", "content": "预约号 A3 到期了吗？"},
         {"role": "tool", "content": "A3: 2026-12-01"},
         {"role": "assistant", "content": "没有。"}],
        [{"role": "user", "content": "按材料回答。"},
         {"role": "assistant", "content": "请提供材料。"},
         {"role": "user", "content": "登记时间为周三。"},
         {"role": "assistant", "content": "登记时间是周三，材料没有提供截止时间。"}],
    ]
    examples = [encode_turns(tokenizer, turns) for turns in conversations]
    # A small explicit matrix keeps the pedagogical weighting example inspectable.
    gold = torch.tensor([[1, -100, -100, -100], [1, 0, 1, -100]])
    logits = torch.tensor([[[0., 3.], [2., 0.], [2., 0.], [2., 0.]],
                           [[2., 0.], [0., 3.], [2., 0.], [2., 0.]]])
    losses = F.cross_entropy(logits.transpose(1, 2), gold, reduction="none", ignore_index=-100)
    counts = (gold != -100).sum(dim=1)
    result = {"token_mean": float(losses.sum() / counts.sum()),
              "example_mean": float((losses.sum(dim=1) / counts).mean()),
              "example_target_counts": counts.tolist(), "examples": []}
    max_tokens = max(len(ids) for ids, _, _ in examples)
    for ids, mask, rows in examples:
        rows.extend({"piece": "(padding)", "supervised": False} for _ in range(max_tokens - len(ids)))
        result["examples"].append({"tokens": len(ids), "supervised": sum(mask), "positions": rows})
    out = BASE / "results" / "sft_loss_check.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("token_mean", result["token_mean"], "example_mean", result["example_mean"])


if __name__ == "__main__":
    main()
