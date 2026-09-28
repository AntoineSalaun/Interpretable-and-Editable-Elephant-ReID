"""Label-free entropy on a frozen, per-image initial identity shortlist."""

import torch


def initial_candidates(identity_scores: torch.Tensor, k: int = 20) -> torch.Tensor:
    if k < 1 or identity_scores.ndim != 2 or identity_scores.shape[1] == 0:
        raise ValueError("Need positive k and a nonempty query-by-identity score matrix")
    # Stable sorting resolves ties by the pooler's sorted identity index.
    return torch.argsort(identity_scores, dim=1, descending=True, stable=True)[:, :k].clone()


def fixed_candidate_entropy(identity_scores, candidates, temperature):
    if temperature <= 0:
        raise ValueError("Temperature must be positive")
    selected = identity_scores.gather(1, candidates.to(identity_scores.device))
    log_probs = torch.log_softmax(selected / temperature, dim=1)
    return -(log_probs.exp() * log_probs).sum(dim=1)
