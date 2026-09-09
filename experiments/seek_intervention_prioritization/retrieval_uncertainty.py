"""Identity-level uncertainty helpers for retrieval scores."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class IdentityScorePooler:
    """Reusable gallery-label index for identity-level max pooling."""

    identities: torch.Tensor
    inverse_indices: torch.Tensor

    @classmethod
    def from_labels(cls, gallery_labels: torch.Tensor, device: torch.device | str | None = None) -> "IdentityScorePooler":
        labels = gallery_labels if device is None else gallery_labels.to(device)
        identities, inverse = torch.unique(labels, sorted=True, return_inverse=True)
        return cls(identities=identities, inverse_indices=inverse)

    def score_matrix(self, similarity: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if similarity.dim() != 2:
            raise ValueError(f"similarity must be 2-D, got {tuple(similarity.shape)}")
        inverse = self.inverse_indices.to(similarity.device)
        identities = self.identities.to(similarity.device)
        if similarity.shape[1] != inverse.numel():
            raise ValueError(
                "similarity columns must match gallery labels, "
                f"got {similarity.shape[1]} and {inverse.numel()}"
            )
        scores = torch.full(
            (similarity.shape[0], identities.numel()),
            -torch.inf,
            device=similarity.device,
            dtype=similarity.dtype,
        )
        index = inverse.unsqueeze(0).expand(similarity.shape[0], -1)
        scores.scatter_reduce_(dim=1, index=index, src=similarity, reduce="amax", include_self=True)
        return identities, scores


def identity_score_matrix(
    similarity: torch.Tensor,
    gallery_labels: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Aggregate gallery-image scores into identity scores with a per-ID max."""

    return IdentityScorePooler.from_labels(gallery_labels, similarity.device).score_matrix(similarity)


def identity_probabilities(
    similarity: torch.Tensor,
    gallery_labels: torch.Tensor,
    temperature: float,
    pooler: IdentityScorePooler | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return identity labels, max-pooled identity scores, and softmax probabilities."""

    if temperature <= 0:
        raise ValueError(f"temperature must be positive, got {temperature}")
    identities, scores = (
        pooler.score_matrix(similarity)
        if pooler is not None
        else identity_score_matrix(similarity, gallery_labels)
    )
    probs = torch.softmax(scores / float(temperature), dim=1)
    return identities, scores, probs


def entropy_from_probs(probs: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    log_probs = torch.log(probs.clamp_min(eps))
    return -(probs * log_probs).sum(dim=1)


def mean_entropy(probs: torch.Tensor, eps: float = 1e-12) -> float:
    return float(entropy_from_probs(probs, eps=eps).mean().item())


def top_two_log_margin(probs: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    if probs.shape[1] == 0:
        raise ValueError("Need at least one identity probability")
    if probs.shape[1] == 1:
        return torch.zeros(probs.shape[0], device=probs.device, dtype=probs.dtype)
    top2 = torch.topk(probs, k=2, dim=1).values
    return torch.log(top2[:, 0].clamp_min(eps)) - torch.log(top2[:, 1].clamp_min(eps))


def mean_top_two_log_margin(probs: torch.Tensor, eps: float = 1e-12) -> float:
    return float(top_two_log_margin(probs, eps=eps).mean().item())


def retrieval_metrics(
    similarity: torch.Tensor,
    query_labels: torch.Tensor,
    gallery_labels: torch.Tensor,
    ks: Iterable[int] = (1, 5),
) -> dict[str, float]:
    """Compute existing gallery-entry top-k recall plus query-image MRR."""

    query_labels = query_labels.to(similarity.device)
    gallery_labels = gallery_labels.to(similarity.device)
    metrics: dict[str, float] = {}
    num_gallery = similarity.shape[1]
    for k in ks:
        k_eff = min(int(k), num_gallery)
        indices = torch.topk(similarity, k=k_eff, dim=1).indices
        hits = (gallery_labels[indices] == query_labels[:, None]).any(dim=1)
        metrics[f"recall_at_{k}"] = float(hits.float().mean().item())

    order = torch.argsort(similarity, dim=1, descending=True)
    matches = gallery_labels[order] == query_labels[:, None]
    first = matches.float().argmax(dim=1)
    has_match = matches.any(dim=1)
    reciprocal = torch.where(
        has_match,
        1.0 / (first.float() + 1.0),
        torch.zeros_like(first, dtype=torch.float32),
    )
    metrics["mrr"] = float(reciprocal.mean().item())
    return metrics


def true_identity_ranking_metrics(
    similarity: torch.Tensor,
    query_labels: torch.Tensor,
    gallery_labels: torch.Tensor,
    pooler: IdentityScorePooler | None = None,
) -> dict[str, float]:
    """Rank the true identity after per-identity gallery max pooling."""

    query_labels = query_labels.to(similarity.device)
    identities, scores = (
        pooler.score_matrix(similarity)
        if pooler is not None
        else identity_score_matrix(similarity, gallery_labels)
    )
    label_to_pos = {int(label.item()): pos for pos, label in enumerate(identities)}
    target_positions = []
    present_rows = []
    missing = 0
    for row, label in enumerate(query_labels):
        pos = label_to_pos.get(int(label.item()))
        if pos is None:
            missing += 1
            continue
        present_rows.append(row)
        target_positions.append(pos)
    if not present_rows:
        return {
            "true_recall_at_1": 0.0,
            "true_mrr": 0.0,
            "true_id_margin": float("-inf"),
            "mean_true_identity_rank": float("inf"),
            "missing_true_identity_count": float(missing),
        }

    row_idx = torch.tensor(present_rows, device=similarity.device, dtype=torch.long)
    target_positions = torch.tensor(target_positions, device=similarity.device, dtype=torch.long)
    present_scores = scores[row_idx]
    order = torch.argsort(present_scores, dim=1, descending=True)
    ranks = (order == target_positions[:, None]).nonzero()[:, 1] + 1
    true_scores = present_scores[torch.arange(present_scores.shape[0], device=scores.device), target_positions]
    incorrect_scores = present_scores.clone()
    incorrect_scores[torch.arange(present_scores.shape[0], device=scores.device), target_positions] = -torch.inf
    margins = true_scores - incorrect_scores.max(dim=1).values
    total = query_labels.numel()

    return {
        "true_recall_at_1": float((ranks == 1).float().sum().item() / total),
        "true_mrr": float((1.0 / ranks.float()).sum().item() / total),
        "true_id_margin": float(margins.mean().item()),
        "mean_true_identity_rank": float(ranks.float().mean().item()),
        "missing_true_identity_count": float(missing),
    }


def fit_temperature_grid(
    similarity: torch.Tensor,
    query_labels: torch.Tensor,
    gallery_labels: torch.Tensor,
    *,
    min_tau: float = 0.01,
    max_tau: float = 100.0,
    steps: int = 81,
    pooler: IdentityScorePooler | None = None,
) -> tuple[float, float]:
    """Fit one scalar temperature by identity-level NLL on non-test data."""

    if min_tau <= 0 or max_tau <= 0 or steps <= 1:
        raise ValueError("temperature grid requires positive bounds and at least two steps")
    query_labels = query_labels.to(similarity.device)
    identities, scores = (
        pooler.score_matrix(similarity)
        if pooler is not None
        else identity_score_matrix(similarity, gallery_labels)
    )
    label_to_pos = {int(label.item()): pos for pos, label in enumerate(identities)}
    keep = torch.tensor([int(label.item()) in label_to_pos for label in query_labels], device=similarity.device)
    if not bool(keep.any().item()):
        return 1.0, float("nan")
    target = torch.tensor(
        [label_to_pos[int(label.item())] for label in query_labels[keep]],
        device=similarity.device,
        dtype=torch.long,
    )
    scores = scores[keep]

    grid = torch.logspace(
        torch.log10(torch.tensor(float(min_tau))),
        torch.log10(torch.tensor(float(max_tau))),
        steps=int(steps),
        device=similarity.device,
    )
    losses = torch.stack([F.cross_entropy(scores / tau, target) for tau in grid])
    best_idx = int(torch.argmin(losses).item())
    return float(grid[best_idx].item()), float(losses[best_idx].item())


def summarize_uncertainty(
    similarity: torch.Tensor,
    gallery_labels: torch.Tensor,
    temperature: float,
    eps: float = 1e-12,
    pooler: IdentityScorePooler | None = None,
) -> dict[str, float]:
    _, _, probs = identity_probabilities(similarity, gallery_labels, temperature, pooler=pooler)
    return {
        "entropy": mean_entropy(probs, eps=eps),
        "margin_log_ratio": mean_top_two_log_margin(probs, eps=eps),
        "temperature": float(temperature),
    }
