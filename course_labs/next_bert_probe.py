"""Toy ablation of context reach, masked supervision and retrieval feedback."""
import json
import time
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

BASE = Path(__file__).parent
LENGTH, WIDTH, MASK = 64, 32, 3
BATCH, STEPS = 24, 160
VERSIONS = ("local", "global", "global_span", "global_retrieval")


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.words = nn.Embedding(32, WIDTH)
        self.positions = nn.Embedding(LENGTH, WIDTH)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(WIDTH, 4, 64, dropout=0.0, batch_first=True, norm_first=True)
            for _ in range(2)
        ])
        self.answer = nn.Linear(WIDTH, 4)
        self.restore = nn.Linear(WIDTH, 32)
        self.retrieval = nn.Linear(WIDTH, WIDTH, bias=False)
        index = torch.arange(LENGTH)
        self.register_buffer("local_mask", (index[:, None] - index[None, :]).abs() > 8)

    def forward(self, ids, global_layer):
        state = self.words(ids) + self.positions(torch.arange(ids.shape[1]))
        state = self.layers[0](state, src_mask=self.local_mask)
        return self.layers[1](state, src_mask=None if global_layer else self.local_mask)


def make_examples(count, positions, seed):
    rng = torch.Generator().manual_seed(seed)
    ids = torch.randint(10, 32, (count, LENGTH), generator=rng)
    position = positions[torch.randint(len(positions), (count,), generator=rng)]
    target = torch.randint(4, (count,), generator=rng)
    row = torch.arange(count)
    ids[:, 0] = 1
    ids[row, position] = 2
    ids[row, position + 1] = target + 4
    return ids, target, position


def masked_positions(count, contiguous, rng):
    if contiguous:
        start = torch.randint(1, LENGTH - 3, (count, 1), generator=rng)
        return start + torch.arange(4)
    return torch.rand(count, LENGTH - 1, generator=rng).argsort(dim=1)[:, :4] + 1


def train_arm(version, train, test):
    torch.manual_seed(42)
    model = Encoder()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.002)
    rng = torch.Generator().manual_seed(17)
    mask_rng = torch.Generator().manual_seed(98)
    x, y, evidence = train
    use_global = version != "local"
    clock = time.perf_counter()
    for _ in range(STEPS):
        picked = torch.randint(len(x), (BATCH,), generator=rng)
        ids, target, location = x[picked], y[picked], evidence[picked]
        state = model(ids, use_global)
        loss = F.cross_entropy(model.answer(state[:, 0]), target)
        positions = masked_positions(len(picked), version == "global_span", mask_rng)
        corrupt = ids.scatter(1, positions, MASK)
        masked_state = model(corrupt, use_global)
        row = torch.arange(len(picked))[:, None]
        loss = loss + 0.25 * F.cross_entropy(
            model.restore(masked_state[row, positions]).flatten(0, 1),
            ids[row, positions].flatten(),
        )
        if version == "global_retrieval":
            scores = torch.einsum("bd,bsd->bs", model.retrieval(state[:, 0]), state) / WIDTH**0.5
            loss = loss + 0.25 * F.cross_entropy(scores, location)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    model.eval()
    results = {}
    with torch.inference_mode():
        for name, (ids, target, location) in test.items():
            state = model(ids, use_global)
            retrieved = torch.einsum("bd,bsd->bs", model.retrieval(state[:, 0]), state).argmax(1)
            results[name] = {
                "accuracy": (model.answer(state[:, 0]).argmax(1) == target).float().mean().item(),
                "retrieval_at_1": (retrieved == location).float().mean().item(),
                "size": len(ids),
            }
    return {"seconds": time.perf_counter() - clock, "last_loss": loss.item(), "test": results}


def main():
    torch.set_num_threads(2)
    train = make_examples(480, torch.arange(4, 62, 2), 2026)
    test = {"short": make_examples(96, torch.arange(5, 16, 2), 120),
            "long": make_examples(96, torch.arange(35, 62, 2), 121)}
    result = {"configuration": {"max_positions": LENGTH, "width": WIDTH, "training_rows": len(train[0]),
                                "batch": BATCH, "steps": STEPS, "masked_positions_per_row": 4},
              "arms": {arm: train_arm(arm, train, test) for arm in VERSIONS}}
    output = BASE / "results" / "next_bert_probe.json"
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print({arm: {name: out["accuracy"] for name, out in row["test"].items()}
           for arm, row in result["arms"].items()})


if __name__ == "__main__":
    main()
