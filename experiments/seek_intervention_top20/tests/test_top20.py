"""Small CPU tests runnable with unittest (no pytest dependency)."""

import unittest
import numpy as np
import torch

from experiments.seek_intervention_top20.topk_entropy import initial_candidates, fixed_candidate_entropy
from experiments.seek_intervention_top20.plot_results import bootstrap_curves, cluster_totals
from experiments.seek_intervention_prioritization.retrieval_uncertainty import entropy_from_probs
from experiments.seek_intervention_prioritization.policies import ExpectedEntropyReductionPolicy, PolicyState


class Top20Tests(unittest.TestCase):
    def test_frozen_candidates_ignore_new_outside_leader(self):
        original = torch.tensor([[4., 3., 2., 1.]])
        candidates = initial_candidates(original, 2)
        edited = torch.tensor([[4., 3., 20., 1.]])
        self.assertTrue(torch.equal(candidates, torch.tensor([[0, 1]])))
        self.assertTrue(torch.equal(fixed_candidate_entropy(original, candidates, 1.), fixed_candidate_entropy(edited, candidates, 1.)))
        self.assertFalse(torch.equal(candidates, initial_candidates(edited, 2)))

    def test_all_candidates_matches_original_entropy(self):
        scores = torch.tensor([[3., 2., 1.], [0., 5., 1.]])
        actual = fixed_candidate_entropy(scores, initial_candidates(scores, 20), .7)
        expected = entropy_from_probs(torch.softmax(scores / .7, dim=1))
        torch.testing.assert_close(actual, expected)

    def test_per_image_selection_and_temperature(self):
        scores = torch.tensor([[4., 3., 0.], [1., 4., 3.]])
        chosen = initial_candidates(scores, 2)
        self.assertTrue(torch.equal(chosen, torch.tensor([[0, 1], [1, 2]])))
        self.assertTrue((fixed_candidate_entropy(scores, chosen, 2.) > fixed_candidate_entropy(scores, chosen, .1)).all())

    def test_top20_policy_prefers_reduction_within_frozen_candidates(self):
        state = PolicyState(torch.zeros(1, 63), torch.zeros(1, 63), [False, False] + [True] * 14)
        candidates = initial_candidates(torch.tensor([[2., 1., 0.]]), 2)
        def scorer(seek):
            # Sex creates an outside leader; age separates the two frozen candidates.
            scores = torch.tensor([[2., 1., 50.]]) if seek[:, :3].sum() else torch.tensor([[6., 1., 0.]])
            return {"entropy": fixed_candidate_entropy(scores, candidates, 1.).item()}
        decision = ExpectedEntropyReductionPolicy().select(state, scorer, {"entropy": 1.})
        self.assertEqual(decision.attribute_name, "age")

    def test_identity_clusters_keep_all_images_together(self):
        hits = np.array([[[1., 1.], [1., 1.], [0., 0.]]])
        totals, sizes = cluster_totals(hits, np.array([0, 0, 1]))
        np.testing.assert_array_equal(sizes, [2, 1])
        np.testing.assert_array_equal(totals[0], [[2, 2], [0, 0]])
        intervals, seed_intervals = bootstrap_curves(hits, np.array([0, 0, 1]), draws=1000)
        np.testing.assert_array_equal(intervals, [[0, 0], [100, 100]])
        np.testing.assert_allclose(seed_intervals, np.full((2, 2), 200/3))

    def test_seed_variation_and_constant_endpoint(self):
        hits = np.array([[[0., 1.], [0., 1.]], [[1., 1.], [1., 1.]]])
        interval, seed_interval = bootstrap_curves(hits, np.array([0, 1]), draws=1000)
        np.testing.assert_array_equal(interval[:, 1], [100, 100])
        np.testing.assert_array_equal(seed_interval[:, 0], [0, 100])


if __name__ == "__main__":
    unittest.main()
