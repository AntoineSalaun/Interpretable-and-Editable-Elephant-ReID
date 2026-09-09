from __future__ import annotations

import torch

from experiments.seek_intervention_prioritization.policies import (
    ExpectedEntropyReductionPolicy,
    GreedyOraclePolicy,
    MarginRatioPolicy,
    PolicyState,
    RandomPolicy,
    copy_attribute_from_expert,
    set_attribute_value,
    sighting_attribute_probabilities,
)
from seek_code import SEEK


def _state() -> PolicyState:
    current = torch.zeros(2, 63)
    logits = torch.zeros(2, 63)
    offset = 0
    lengths = [3, 2, 3, 3, 5, 5, 5, 5, 5, 5, 5, 5, 3, 3, 3, 3]
    for length in lengths:
        current[:, offset] = 1.0
        logits[:, offset : offset + length] = torch.arange(length, dtype=torch.float32)
        offset += length
    return PolicyState(
        current_seek=current,
        concept_logits=logits,
        corrected_mask=torch.zeros(16, dtype=torch.bool),
    )


def test_set_attribute_value_sets_one_block_only() -> None:
    state = _state()

    edited = set_attribute_value(state.current_seek, attribute_index=0, value_index=2)

    assert edited[:, 2].eq(1).all()
    assert edited[:, :3].sum(dim=1).eq(1).all()
    assert torch.equal(edited[:, 3:], state.current_seek[:, 3:])


def test_copy_attribute_from_expert_copies_selected_block() -> None:
    state = _state()
    expert = set_attribute_value(state.current_seek, attribute_index=1, value_index=1)

    edited = copy_attribute_from_expert(state.current_seek, expert, attribute_index=1)

    assert torch.equal(edited[:, 3:5], expert[:, 3:5])
    assert torch.equal(edited[:, :3], state.current_seek[:, :3])


def test_random_policy_selects_remaining_attribute() -> None:
    state = _state()
    state.corrected_mask[0] = True

    decision = RandomPolicy(seed=1).select(state)

    assert decision.attribute_index in range(1, 16)


def test_entropy_and_margin_do_not_change_when_labels_are_shuffled() -> None:
    state = _state()
    labels = torch.tensor([1, 2])
    shuffled_labels = torch.tensor([2, 1])

    def make_label_free_scorer(_labels):
        def scorer(seek_vectors: torch.Tensor):
            signal = float(seek_vectors[:, 0].mean().item())
            return {
                "entropy": 10.0 - signal,
                "margin_log_ratio": signal,
            }

        return scorer

    entropy_a = ExpectedEntropyReductionPolicy().select(state, make_label_free_scorer(labels))
    entropy_b = ExpectedEntropyReductionPolicy().select(state, make_label_free_scorer(shuffled_labels))
    margin_a = MarginRatioPolicy().select(state, make_label_free_scorer(labels))
    margin_b = MarginRatioPolicy().select(state, make_label_free_scorer(shuffled_labels))

    assert entropy_a.attribute_index == entropy_b.attribute_index
    assert entropy_a.scores_by_attribute == entropy_b.scores_by_attribute
    assert margin_a.attribute_index == margin_b.attribute_index
    assert margin_a.scores_by_attribute == margin_b.scores_by_attribute


def test_expected_entropy_reduction_selects_largest_expected_downstream_gain() -> None:
    state = _state()

    def scorer(seek_vectors: torch.Tensor):
        attr0_value2 = float(seek_vectors[:, 2].mean().item())
        return {
            "entropy": 10.0 * (1.0 - attr0_value2),
            "margin_log_ratio": 0.0,
        }

    decision = ExpectedEntropyReductionPolicy().select(state, scorer)
    probs = sighting_attribute_probabilities(state.concept_logits, 0)
    expected_gain = 10.0 * float(probs[2].item())

    assert decision.attribute_index == 0
    assert abs(decision.score - expected_gain) < 1e-6
    assert abs(decision.scores_by_attribute[SEEK.attribute_names[1]]) < 1e-6


def test_expected_entropy_reduction_does_not_score_concept_entropy_alone() -> None:
    state = _state()

    def scorer(_seek_vectors: torch.Tensor):
        return {
            "entropy": 3.0,
            "margin_log_ratio": 0.0,
        }

    decision = ExpectedEntropyReductionPolicy().select(state, scorer)

    assert all(abs(score) < 1e-6 for score in decision.scores_by_attribute.values())


def test_margin_policy_combines_concept_uncertainty_and_margin_importance() -> None:
    state = _state()

    def scorer(seek_vectors: torch.Tensor):
        return {
            "entropy": 0.0,
            "margin_log_ratio": float(seek_vectors[:, 2].mean().item()),
        }

    decision = MarginRatioPolicy(margin_lambda=2.0).select(state, scorer)

    probs = sighting_attribute_probabilities(state.concept_logits, 0)
    expected_uncertainty = float(-(probs * torch.log(probs.clamp_min(1e-12))).sum().item())
    expected_importance = float(probs[2].item())
    expected_score = expected_uncertainty + 2.0 * expected_importance
    diagnostics = decision.diagnostics["attributes"][SEEK.attribute_names[0]]

    assert abs(diagnostics["margin_concept_uncertainty"] - expected_uncertainty) < 1e-6
    assert abs(diagnostics["margin_importance"] - expected_importance) < 1e-6
    assert abs(diagnostics["margin_combined_score"] - expected_score) < 1e-6


def test_oracle_uses_explicit_ground_truth_scorer() -> None:
    state = _state()
    expert = set_attribute_value(state.current_seek, attribute_index=2, value_index=2)

    def scorer(seek_vectors: torch.Tensor):
        score = float(seek_vectors[:, 7].mean().item())
        return {
            "true_recall_at_1": score,
            "true_mrr": score,
            "true_id_margin": score,
        }

    decision = GreedyOraclePolicy().select(state, expert, scorer)

    assert decision.attribute_index == 2
