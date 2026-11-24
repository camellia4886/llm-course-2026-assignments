"""Small original decoder used to check nanoGPT's tensor and compute accounting."""
import json
import math
import time
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

BASE = Path(__file__).parent
BATCH, CONTEXT, WIDTH, HEADS, LAYERS = 8, 64, 64, 4, 2


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.attention_norm = nn.LayerNorm(WIDTH)
        self.qkv = nn.Linear(WIDTH, 3 * WIDTH)
        self.project = nn.Linear(WIDTH, WIDTH)
        self.ffn_norm = nn.LayerNorm(WIDTH)
        self.ffn = nn.Sequential(nn.Linear(WIDTH, 4 * WIDTH), nn.GELU(), nn.Linear(4 * WIDTH, WIDTH))

    def forward(self, state):
        batch, length, _ = state.shape
        projections = self.qkv(self.attention_norm(state)).split(WIDTH, dim=-1)
        query, key, value = [
            tensor.reshape(batch, length, HEADS, WIDTH // HEADS).transpose(1, 2)
            for tensor in projections
        ]
        attended = F.scaled_dot_product_attention(query, key, value, is_causal=True)
        attended = attended.transpose(1, 2).reshape(batch, length, WIDTH)
        state = state + self.project(attended)
        return state + self.ffn(self.ffn_norm(state))


class Decoder(nn.Module):
    def __init__(self, vocabulary):
        super().__init__()
        self.token = nn.Embedding(vocabulary, WIDTH)
        self.position = nn.Embedding(CONTEXT, WIDTH)
        self.blocks = nn.Sequential(*(Block() for _ in range(LAYERS)))
        self.norm = nn.LayerNorm(WIDTH)
        self.apply(self.initialize)

    @staticmethod
    def initialize(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            nn.init.zeros_(module.bias)

    def forward(self, ids):
        positions = torch.arange(ids.shape[1], device=ids.device)
        state = self.blocks(self.token(ids) + self.position(positions))
        return F.linear(self.norm(state), self.token.weight)


def batch_from(tokens, generator):
    start = torch.randint(len(tokens) - CONTEXT, (BATCH,), generator=generator)
    indices = start[:, None] + torch.arange(CONTEXT)
    return tokens[indices], tokens[indices + 1]


def loss_on(model, inputs, targets):
    logits = model(inputs)
    return F.cross_entropy(logits.flatten(0, 1), targets.flatten())


@torch.inference_mode()
def validation_loss(model, tokens):
    random = torch.Generator().manual_seed(321)
    return sum(loss_on(model, *batch_from(tokens, random)).item() for _ in range(8)) / 8


def main():
    torch.set_num_threads(3)
    torch.manual_seed(42)
    source = BASE.parent / "data"
    train_text = (source / "05.txt").read_text(encoding="utf-8")[:170000]
    heldout_text = (source / "06.txt").read_text(encoding="utf-8")[:35000]
    vocab = {char: index for index, char in enumerate(sorted(set(train_text)), 1)}
    train = torch.tensor([vocab[char] for char in train_text])
    heldout = torch.tensor([vocab.get(char, 0) for char in heldout_text])
    model = Decoder(len(vocab) + 1)
    params = sum(parameter.numel() for parameter in model.parameters())
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.003)
    random = torch.Generator().manual_seed(123)
    before = validation_loss(model, heldout)
    x, _ = batch_from(train, random)
    changed = x.clone()
    changed[:, 31:] = (changed[:, 31:] + 1) % (len(vocab) + 1)
    with torch.no_grad():
        causal_difference = (model(x)[:, :31] - model(changed)[:, :31]).abs().max().item()
    assert causal_difference < 1e-5, "later tokens influenced past logits"

    start = time.perf_counter()
    loss_history = []
    for step in range(120):
        optimizer.zero_grad(set_to_none=True)
        loss = loss_on(model, *batch_from(train, random))
        loss.backward()
        optimizer.step()
        if (step + 1) % 30 == 0:
            loss_history.append({"step": step + 1, "loss": loss.item()})
    seconds = time.perf_counter() - start
    after = validation_loss(model, heldout)
    res = {
        "device": "CPU", "seed": 42, "parameters": params, "train_month": 5, "heldout_month": 6,
        "setup": {"B": BATCH, "s": CONTEXT, "d": WIDTH, "h": HEADS, "L": LAYERS,
                  "train_characters": len(train), "heldout_characters": len(heldout),
                  "heldout_unknown_characters": int((heldout == 0).sum())},
        "full_square_flops_per_block": {
            "qkv": 6 * BATCH * CONTEXT * WIDTH**2,
            "attention_projection": 2 * BATCH * CONTEXT * WIDTH**2,
            "ffn": 16 * BATCH * CONTEXT * WIDTH**2,
            "attention_pairing": 4 * BATCH * CONTEXT**2 * WIDTH,
        },
        "causal_prefix_max_diff": causal_difference, "training_seconds": seconds,
        "heldout_before": before, "heldout_after": after, "heldout_ppl_after": math.exp(after),
        "history": loss_history,
    }
    output = BASE / "results" / "nanogpt_exercise.json"
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps(res, indent=2) + "\n", encoding="utf-8")
    print(f"heldout loss {before:.3f} -> {after:.3f}, seconds={seconds:.2f}")


if __name__ == "__main__":
    main()
