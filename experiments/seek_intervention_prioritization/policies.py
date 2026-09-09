"""Intervention policies for prioritized SEEK corrections."""

from __future__ import annotations

import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Mapping, Sequence

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
METHODS_DIR = REPO_ROOT / "methods"
if str(METHODS_DIR) not in sys.path:
    sys.path.insert(0, str(METHODS_DIR))

from seek_code import SEEK  # noqa: E402


MetricDict = Mapping[str, float]
CounterfactualScorer = Callable[[torch.Tensor], MetricDict]
OracleScorer = Callable[[torch.Tensor], MetricDict]


ATTRIBUTE_SLICES: list[slice] = []
_offset = 0
for _name in SEEK.attribute_names:
    _length = SEEK.lengths[_name]
    ATTRIBUTE_SLICES.append(slice(_offset, _offset + _length))
    _offset += _length


@dataclass
class PolicyState:
    """Label-free state passed to deployable policies."""

    current_seek: torch.Tensor
    concept_logits: torch.Tensor
    corrected_mask: torch.Tensor | Sequence[bool]
    costs: torch.Tensor | Sequence[float] | None = None

    def __post_init__(self) -> None:
        if self.current_seek.dim() != 2 or self.current_seek.shape[1] != 63:
            raise ValueError(f"current_seek must have shape (N, 63), got {tuple(self.current_seek.shape)}")
        if self.concept_logits.shape != self.current_seek.shape:
            raise ValueError(
                "concept_logits must have the same shape as current_seek, "
                f"got {tuple(self.concept_logits.shape)} and {tuple(self.current_seek.shape)}"
            )
        self.corrected_mask = torch.as_tensor(self.corrected_mask, dtype=torch.bool)
        if self.corrected_mask.numel() != len(SEEK.attribute_names):
            raise ValueError("corrected_mask must contain one flag per SEEK attribute")
        if self.costs is None:
            self.costs = torch.ones(len(SEEK.attribute_names), dtype=torch.float32)
        else:
            self.costs = torch.as_tensor(self.costs, dtype=torch.float32)
        if self.costs.numel() != len(SEEK.attribute_names):
            raise ValueError("costs must contain one value per SEEK attribute")


@dataclass(frozen=True)
class PolicyDecision:
    attribute_index: int
    attribute_name: str
    score: float
    scores_by_attribute: dict[str, float]
    diagnostics: dict[str, object] = field(default_factory=dict)


def remaining_attribute_indices(corrected_mask: torch.Tensor | Sequence[bool]) -> list[int]:
    mask = torch.as_tensor(corrected_mask, dtype=torch.bool)
    return [idx for idx, is_corrected in enumerate(mask.tolist()) if not is_corrected]


def set_attribute_value(seek_vectors: torch.Tensor, attribute_index: int, value_index: int) -> torch.Tensor:
    """Return a copy with one categorical SEEK attribute set for all rows."""

    edited = seek_vectors.clone()
    attr_slice = ATTRIBUTE_SLICES[attribute_index]
    width = attr_slice.stop - attr_slice.start
    if not 0 <= value_index < width:
        name = SEEK.attribute_names[attribute_index]
        raise ValueError(f"value_index {value_index} is out of range for {name}")
    edited[:, attr_slice] = 0.0
    edited[:, attr_slice.start + value_index] = 1.0
    return edited


def copy_attribute_from_expert(
    seek_vectors: torch.Tensor,
    expert_seek_vectors: torch.Tensor,
    attribute_index: int,
) -> torch.Tensor:
    """Return a copy with one attribute copied from the expert sighting SEEK."""

    if expert_seek_vectors.shape != seek_vectors.shape:
        raise ValueError(
            "expert_seek_vectors must have the same shape as seek_vectors, "
            f"got {tuple(expert_seek_vectors.shape)} and {tuple(seek_vectors.shape)}"
        )
    edited = seek_vectors.clone()
    attr_slice = ATTRIBUTE_SLICES[attribute_index]
    edited[:, attr_slice] = expert_seek_vectors[:, attr_slice]
    return edited


def sighting_attribute_probabilities(concept_logits: torch.Tensor, attribute_index: int) -> torch.Tensor:
    """Average per-image concept-head probabilities into a sighting distribution."""

    attr_slice = ATTRIBUTE_SLICES[attribute_index]
    probs = torch.softmax(concept_logits[:, attr_slice], dim=1).mean(dim=0)
    total = probs.sum().clamp_min(1e-12)
    return probs / total


def value_names(attribute_index: int) -> list[str]:
    return list(SEEK.mappings[SEEK.attribute_names[attribute_index]])


def _choose_best(scores: dict[int, float]) -> int:
    if not scores:
        raise ValueError("No candidate attributes remain")
    return max(scores, key=lambda idx: (scores[idx], -idx))


class RandomPolicy:
    name = "random"
    display_name = "Random"
    uses_ground_truth = False

    def __init__(self, seed: int = 42) -> None:
        self.rng = random.Random(seed)

    def select(self, state: PolicyState, scorer: CounterfactualScorer | None = None) -> PolicyDecision:
        remaining = remaining_attribute_indices(state.corrected_mask)
        chosen = self.rng.choice(remaining)
        return PolicyDecision(
            attribute_index=chosen,
            attribute_name=SEEK.attribute_names[chosen],
            score=0.0,
            scores_by_attribute={SEEK.attribute_names[idx]: 0.0 for idx in remaining},
            diagnostics={"num_remaining": len(remaining)},
        )


class ExpectedEntropyReductionPolicy:
    name = "entropy"
    display_name = "Expected entropy reduction"
    uses_ground_truth = False

    def select(
        self,
        state: PolicyState,
        scorer: CounterfactualScorer,
        current_metrics: MetricDict | None = None,
    ) -> PolicyDecision:
        current = dict(current_metrics or scorer(state.current_seek))
        current_entropy = float(current["entropy"])
        scores: dict[int, float] = {}
        diagnostics: dict[str, object] = {"current_entropy": current_entropy, "values": {}}

        for attr_idx in remaining_attribute_indices(state.corrected_mask):
            probs = sighting_attribute_probabilities(state.concept_logits, attr_idx)
            expected_entropy = 0.0
            attr_values = {}
            for value_idx, prob in enumerate(probs.tolist()):
                counterfactual = set_attribute_value(state.current_seek, attr_idx, value_idx)
                metrics = scorer(counterfactual)
                entropy = float(metrics["entropy"])
                expected_entropy += float(prob) * entropy
                attr_values[value_names(attr_idx)[value_idx]] = {
                    "probability": float(prob),
                    "entropy": entropy,
                }
            cost = float(torch.as_tensor(state.costs)[attr_idx].item())
            scores[attr_idx] = (current_entropy - expected_entropy) / cost
            diagnostics["values"][SEEK.attribute_names[attr_idx]] = attr_values

        chosen = _choose_best(scores)
        return PolicyDecision(
            attribute_index=chosen,
            attribute_name=SEEK.attribute_names[chosen],
            score=float(scores[chosen]),
            scores_by_attribute={SEEK.attribute_names[idx]: float(score) for idx, score in scores.items()},
            diagnostics=diagnostics,
        )


class MarginRatioPolicy:
    name = "margin"
    display_name = "Margin-ratio"
    uses_ground_truth = False

    def __init__(self, margin_lambda: float = 1.0, eps: float = 1e-12) -> None:
        self.margin_lambda = float(margin_lambda)
        self.eps = float(eps)

    def select(
        self,
        state: PolicyState,
        scorer: CounterfactualScorer,
        current_metrics: MetricDict | None = None,
    ) -> PolicyDecision:
        current = dict(current_metrics or scorer(state.current_seek))
        current_margin = float(current["margin_log_ratio"])
        scores: dict[int, float] = {}
        diagnostics: dict[str, object] = {
            "margin_lambda": self.margin_lambda,
            "current_margin_log_ratio": current_margin,
            "attributes": {},
        }

        for attr_idx in remaining_attribute_indices(state.corrected_mask):
            probs = sighting_attribute_probabilities(state.concept_logits, attr_idx)
            concept_uncertainty = float(-(probs * torch.log(probs.clamp_min(self.eps))).sum().item())
            expected_margin = 0.0
            attr_values = {}
            for value_idx, prob in enumerate(probs.tolist()):
                counterfactual = set_attribute_value(state.current_seek, attr_idx, value_idx)
                metrics = scorer(counterfactual)
                margin = float(metrics["margin_log_ratio"])
                expected_margin += float(prob) * margin
                attr_values[value_names(attr_idx)[value_idx]] = {
                    "probability": float(prob),
                    "margin_log_ratio": margin,
                }
            margin_importance = expected_margin - current_margin
            combined_score = concept_uncertainty + self.margin_lambda * margin_importance
            scores[attr_idx] = combined_score
            diagnostics["attributes"][SEEK.attribute_names[attr_idx]] = {
                "margin_concept_uncertainty": concept_uncertainty,
                "margin_importance": margin_importance,
                "margin_combined_score": combined_score,
                "expected_margin_log_ratio": expected_margin,
                "values": attr_values,
            }

        chosen = _choose_best(scores)
        selected = diagnostics["attributes"][SEEK.attribute_names[chosen]]
        diagnostics["selected"] = selected
        return PolicyDecision(
            attribute_index=chosen,
            attribute_name=SEEK.attribute_names[chosen],
            score=float(scores[chosen]),
            scores_by_attribute={SEEK.attribute_names[idx]: float(score) for idx, score in scores.items()},
            diagnostics=diagnostics,
        )


class GreedyOraclePolicy:
    name = "greedy_oracle"
    display_name = "Oracle"
    uses_ground_truth = True

    def select(
        self,
        state: PolicyState,
        expert_seek_vectors: torch.Tensor,
        scorer: OracleScorer,
    ) -> PolicyDecision:
        scores: dict[int, float] = {}
        tie_breakers: dict[int, tuple[float, float, float, int]] = {}
        diagnostics: dict[str, object] = {"values": {}}

        for attr_idx in remaining_attribute_indices(state.corrected_mask):
            counterfactual = copy_attribute_from_expert(state.current_seek, expert_seek_vectors, attr_idx)
            metrics = scorer(counterfactual)
            recall = float(metrics["true_recall_at_1"])
            mrr = float(metrics["true_mrr"])
            margin = float(metrics["true_id_margin"])
            scores[attr_idx] = recall
            tie_breakers[attr_idx] = (recall, mrr, margin, -attr_idx)
            diagnostics["values"][SEEK.attribute_names[attr_idx]] = dict(metrics)

        chosen = max(tie_breakers, key=tie_breakers.get)
        return PolicyDecision(
            attribute_index=chosen,
            attribute_name=SEEK.attribute_names[chosen],
            score=float(scores[chosen]),
            scores_by_attribute={SEEK.attribute_names[idx]: float(score) for idx, score in scores.items()},
            diagnostics=diagnostics,
        )


POLICY_FACTORIES = {
    "random": RandomPolicy,
    "entropy": ExpectedEntropyReductionPolicy,
    "margin": MarginRatioPolicy,
    "greedy_oracle": GreedyOraclePolicy,
    "oracle": GreedyOraclePolicy,
}
