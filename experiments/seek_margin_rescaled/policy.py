"""Concept uncertainty plus an expected raw identity-similarity gap increase."""

import math

import torch

from experiments.seek_intervention_prioritization.policies import (
    SEEK, PolicyDecision, remaining_attribute_indices, set_attribute_value,
    sighting_attribute_probabilities,
)


def rescaled_weight(original_lambda, original_temperature):
    if not math.isfinite(original_temperature) or original_temperature <= 0:
        raise ValueError("The original temperature must be finite and positive")
    if not math.isfinite(original_lambda):
        raise ValueError("The original lambda must be finite")
    return original_lambda / original_temperature


def raw_margin(identity_scores):
    if identity_scores.shape[-1] < 2:
        return torch.zeros(identity_scores.shape[0], device=identity_scores.device)
    top = identity_scores.topk(2, dim=1).values
    return top[:, 0] - top[:, 1]


class RawMarginPolicy:
    def __init__(self, beta, eps=1e-12):
        self.beta = float(beta)
        self.eps = eps

    def select(self, state, score_batch):
        remaining = remaining_attribute_indices(state.corrected_mask)
        if not remaining:
            raise ValueError("No candidate attributes remain")
        variants = [state.current_seek]
        probabilities = {}
        for attr in remaining:
            probabilities[attr] = sighting_attribute_probabilities(state.concept_logits, attr)
            variants.extend(set_attribute_value(state.current_seek, attr, v)
                            for v in range(len(probabilities[attr])))
        metrics = score_batch(variants)
        scores, attributes = {}, {}
        cursor = 1
        for attr in remaining:
            probs = probabilities[attr]
            values = metrics[cursor:cursor + len(probs)]
            cursor += len(probs)
            uncertainty = float(-(probs * probs.clamp_min(self.eps).log()).sum().item())
            importance = sum(p * m["raw_margin"] for p, m in zip(probs.tolist(), values)) - metrics[0]["raw_margin"]
            scores[attr] = uncertainty + self.beta * importance
            attributes[SEEK.attribute_names[attr]] = {
                "concept_uncertainty": uncertainty, "raw_importance": importance,
                "combined_score": scores[attr],
                "probabilities": probs.tolist(), "values": values,
            }
        chosen = max(scores, key=lambda attr: (scores[attr], -attr))
        return PolicyDecision(chosen, SEEK.attribute_names[chosen], scores[chosen],
                              {SEEK.attribute_names[a]: score for a, score in scores.items()},
                              {"margin_beta": self.beta, "current": metrics[0], "attributes": attributes})
