"""Compare SFT and group-relative policy updates on an exact arithmetic verifier."""
import copy
import json
import time
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

BASE = Path(__file__).parent
STEPS, BATCH, GROUP = 160, 32, 8


class TinyPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.digit = nn.Embedding(12, 12)
        self.predict = nn.Sequential(nn.Linear(24, 48), nn.Tanh(), nn.Linear(48, 3))

    def forward(self, inputs):
        return self.predict(self.digit(inputs).flatten(1))


@torch.inference_mode()
def accuracy(model, inputs, answers):
    return (model(inputs).argmax(-1) == answers).float().mean().item()


def experiment(initial, train, answers, test, test_answers, method):
    torch.manual_seed(42)
    model = copy.deepcopy(initial)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.008)
    generator = torch.Generator().manual_seed(2026)
    start = time.perf_counter()
    history = []
    reward_samples = 0
    for step in range(STEPS):
        indices = torch.randint(len(train), (BATCH,), generator=generator)
        inputs, target = train[indices], answers[indices]
        if method == "sft":
            loss = F.cross_entropy(model(inputs), target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        else:
            with torch.no_grad():
                initial_logp = initial(inputs).log_softmax(-1)
                before_logp = model(inputs).log_softmax(-1)
                samples = torch.multinomial(before_logp.exp(), GROUP, replacement=True)
                reward = (samples == target[:, None]).float()
                advantage = (reward - reward.mean(1, keepdim=True)) / (
                    reward.std(1, keepdim=True, unbiased=False) + 1e-5)
                old = before_logp.gather(1, samples)
                reference = initial_logp.gather(1, samples)
                reward_samples += samples.numel()
            for _ in range(2):
                current = model(inputs).log_softmax(-1).gather(1, samples)
                ratio = (current - old).exp()
                delta = reference - current
                kl = delta.exp() - delta - 1
                surrogate = torch.minimum(ratio * advantage, ratio.clamp(0.8, 1.2) * advantage)
                loss = (-surrogate + 0.04 * kl).mean()
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
        if (step + 1) % 40 == 0:
            history.append({"step": step + 1, "loss": loss.item(),
                            "test_accuracy": accuracy(model, test, test_answers)})
    return {"seconds": time.perf_counter() - start, "history": history,
            "reward_evaluations": reward_samples, "mean_output_tokens": 1.0,
            "parameter_updates": STEPS if method == "sft" else 2 * STEPS,
            "train_accuracy": accuracy(model, train, answers),
            "test_accuracy": accuracy(model, test, test_answers)}


def main():
    torch.manual_seed(17)
    torch.set_num_threads(2)
    inputs = torch.cartesian_prod(torch.arange(12), torch.arange(12))
    target = (inputs[:, 0] + inputs[:, 1]) % 3
    # Holding both orders out prevents learning an exact reversed test pair.
    test_mask = ((inputs[:, 0] * inputs[:, 1] + inputs.sum(1)) % 7 == 0)
    train, test = inputs[~test_mask], inputs[test_mask]
    train_labels, test_labels = target[~test_mask], target[test_mask]
    initial = TinyPolicy()
    report = {"task": "Pick (a + b) modulo 3, a and b from 0 to 11",
              "train_pairs": len(train), "heldout_pairs": len(test),
              "initial_accuracy": accuracy(initial, test, test_labels),
              "configuration": {"device": "CPU", "rounds": STEPS, "batch": BATCH,
                                "group": GROUP, "seed": 42, "head_output_length": 1},
              "arms": {}}
    for name in ("sft", "grpo"):
        report["arms"][name] = experiment(initial, train, train_labels, test, test_labels, name)
    path = BASE / "results" / "tiny_posttrain.json"
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print([(k, r["test_accuracy"]) for k, r in report["arms"].items()])


if __name__ == "__main__":
    main()
