"""Check causal-token NLL against PyTorch's cross-entropy on a padded batch."""

import json
from pathlib import Path

import torch
from torch.nn import functional as F


def main():
    logits = torch.arange(40, dtype=torch.float64).reshape(2, 4, 5) / 5
    labels = torch.tensor([[2, 1, 2, -100], [3, 0, 4, -100]])
    # Predict labels[:, t + 1] from the logit vector at position t.
    flat_logits = logits[:, :-1].reshape(-1, logits.size(-1))
    targets = labels[:, 1:].reshape(-1)
    log_probs = F.log_softmax(flat_logits, dim=-1)
    valid = targets != -100
    per_token = -log_probs[valid].gather(1, targets[valid, None]).squeeze(1)
    cross_entropy = F.cross_entropy(flat_logits, targets, ignore_index=-100)
    nll = F.nll_loss(log_probs, targets, ignore_index=-100)
    torch.testing.assert_close(cross_entropy, nll)
    torch.testing.assert_close(nll, per_token.mean())
    torch.testing.assert_close(
        F.nll_loss(log_probs, targets, ignore_index=-100, reduction="sum"),
        per_token.sum(),
    )
    result = {
        "torch": torch.__version__,
        "logits_shape": list(logits.shape),
        "shifted_targets": targets.tolist(),
        "valid_positions": int(valid.sum()),
        "per_token_nll": per_token.tolist(),
        "mean_cross_entropy": cross_entropy.item(),
        "mean_nll_loss": nll.item(),
        "mean_manual_nll": per_token.mean().item(),
        "sum_nll": per_token.sum().item(),
        "perplexity": nll.exp().item(),
    }
    output = Path(__file__).parent / "results" / "nll_check.json"
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(result)


if __name__ == "__main__":
    main()
