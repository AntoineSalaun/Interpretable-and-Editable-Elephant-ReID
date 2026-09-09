from __future__ import annotations

import torch

from experiments.seek_intervention_prioritization.retrieval_uncertainty import (
    IdentityScorePooler,
    fit_temperature_grid,
    identity_probabilities,
    identity_score_matrix,
    retrieval_metrics,
    summarize_uncertainty,
    true_identity_ranking_metrics,
)


def test_identity_score_matrix_uses_max_per_identity() -> None:
    similarity = torch.tensor(
        [
            [0.1, 0.8, 0.4, 0.7],
            [0.9, 0.2, 0.5, 0.3],
        ]
    )
    gallery_labels = torch.tensor([1, 1, 2, 2])

    identities, scores = identity_score_matrix(similarity, gallery_labels)

    assert identities.tolist() == [1, 2]
    assert torch.allclose(scores, torch.tensor([[0.8, 0.7], [0.9, 0.5]]))


def test_identity_score_pooler_matches_identity_score_matrix() -> None:
    similarity = torch.tensor(
        [
            [0.1, 0.8, 0.4, 0.7],
            [0.9, 0.2, 0.5, 0.3],
        ]
    )
    gallery_labels = torch.tensor([1, 1, 2, 2])

    expected = identity_score_matrix(similarity, gallery_labels)
    actual = IdentityScorePooler.from_labels(gallery_labels).score_matrix(similarity)

    assert torch.equal(actual[0], expected[0])
    assert torch.allclose(actual[1], expected[1])


def test_identity_probabilities_are_normalized() -> None:
    similarity = torch.tensor([[1.0, 0.0, 0.5]])
    gallery_labels = torch.tensor([3, 4, 4])

    _, _, probs = identity_probabilities(similarity, gallery_labels, temperature=1.0)

    assert probs.shape == (1, 2)
    assert torch.allclose(probs.sum(dim=1), torch.ones(1))


def test_temperature_controls_identity_uncertainty() -> None:
    similarity = torch.tensor([[4.0, 0.0, 1.0]])
    gallery_labels = torch.tensor([3, 4, 4])

    cold = summarize_uncertainty(similarity, gallery_labels, temperature=0.1)
    warm = summarize_uncertainty(similarity, gallery_labels, temperature=10.0)

    assert cold["temperature"] == 0.1
    assert warm["temperature"] == 10.0
    assert warm["entropy"] > cold["entropy"]
    assert warm["margin_log_ratio"] < cold["margin_log_ratio"]


def test_retrieval_metrics_follow_gallery_entry_convention() -> None:
    similarity = torch.tensor(
        [
            [0.9, 0.2, 0.1],
            [0.3, 0.8, 0.7],
        ]
    )
    query_labels = torch.tensor([1, 2])
    gallery_labels = torch.tensor([1, 1, 2])

    metrics = retrieval_metrics(similarity, query_labels, gallery_labels, ks=(1, 2))

    assert metrics["recall_at_1"] == 0.5
    assert metrics["recall_at_2"] == 1.0
    assert metrics["mrr"] == 0.75


def test_true_identity_ranking_metrics_pool_by_identity() -> None:
    similarity = torch.tensor(
        [
            [0.4, 0.7, 0.6],
            [0.5, 0.2, 0.8],
        ]
    )
    query_labels = torch.tensor([1, 2])
    gallery_labels = torch.tensor([1, 1, 2])

    metrics = true_identity_ranking_metrics(similarity, query_labels, gallery_labels)

    assert metrics["true_recall_at_1"] == 1.0
    assert metrics["true_mrr"] == 1.0
    assert metrics["true_id_margin"] > 0


def test_fit_temperature_grid_returns_positive_temperature() -> None:
    similarity = torch.tensor(
        [
            [2.0, 0.0],
            [0.0, 2.0],
        ]
    )
    query_labels = torch.tensor([0, 1])
    gallery_labels = torch.tensor([0, 1])

    tau, nll = fit_temperature_grid(similarity, query_labels, gallery_labels, steps=5)

    assert tau > 0
    assert nll >= 0
