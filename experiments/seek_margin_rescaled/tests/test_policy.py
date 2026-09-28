import unittest
from types import SimpleNamespace

import torch

from experiments.seek_intervention_prioritization.policies import MarginRatioPolicy, PolicyState
from experiments.seek_intervention_prioritization.retrieval_uncertainty import top_two_log_margin
from experiments.seek_margin_rescaled.policy import RawMarginPolicy, raw_margin, rescaled_weight


class RescalingTests(unittest.TestCase):
    def test_softmax_cancels_when_floor_is_inactive(self):
        scores = torch.tensor([[.75, .7, .5], [.3, .5, .1]], dtype=torch.float64)
        tau = .03548133373260498
        torch.testing.assert_close(raw_margin(scores)/tau, top_two_log_margin(torch.softmax(scores/tau, dim=1)))

    def test_floor_can_break_equivalence(self):
        scores = torch.tensor([[1., -1.]])
        self.assertGreater(raw_margin(scores).item()/.01, top_two_log_margin(torch.softmax(scores/.01, dim=1)).item())

    def test_rescaled_policy_matches_original(self):
        torch.manual_seed(42)
        tau, weight = .03548133373260498, .3
        matrix = torch.randn(63, 5, dtype=torch.float64) * .002
        state = PolicyState(torch.zeros(2, 63), torch.randn(2, 63), [False]*16)
        def score(seek):
            scores = seek.double() @ matrix
            return {"raw_margin": raw_margin(scores).mean().item(),
                    "margin_log_ratio": top_two_log_margin(torch.softmax(scores/tau, dim=1)).mean().item()}
        original = MarginRatioPolicy(weight).select(state, score)
        raw = RawMarginPolicy(rescaled_weight(weight, tau)).select(state, lambda variants: [score(v) for v in variants])
        self.assertEqual(original.attribute_index, raw.attribute_index)
        for attr in original.scores_by_attribute:
            self.assertAlmostEqual(original.scores_by_attribute[attr], raw.scores_by_attribute[attr], places=7)

    def test_single_identity_and_invalid_temperature(self):
        torch.testing.assert_close(raw_margin(torch.ones(2, 1)), torch.zeros(2))
        with self.assertRaises(ValueError):
            rescaled_weight(.3, 0.)

    def test_batched_scorer_matches_individual_retrieval(self):
        from experiments.seek_margin_rescaled.run import score_variants, base, IdentityScorePooler
        torch.manual_seed(6)
        projector = SimpleNamespace(layer=torch.nn.Linear(63, 8).eval(), device="cpu", layer_norm=True, alpha=.4)
        retrieval = SimpleNamespace(similarity_matrix=lambda q, g: torch.nn.functional.normalize(q, dim=1) @ torch.nn.functional.normalize(g, dim=1).T)
        visual = torch.randn(3, 8)
        variants = [torch.randn(3, 63) for _ in range(4)]
        gallery = torch.randn(7, 8)
        pooler = IdentityScorePooler.from_labels(torch.tensor([0, 0, 1, 2, 3, 3, 4]))
        audit = dict(clipped_top2_rows=0, min_top2_probability=1., counterfactual_image_rows=0, max_per_image_margin_error=0.)
        actual = score_variants(projector, retrieval, visual, variants, gallery, pooler, .3, 1e-12, audit)
        for variant, metrics in zip(variants, actual):
            embeddings = base.project_from_seek(projector, visual, variant)
            _, scores = pooler.score_matrix(retrieval.similarity_matrix(embeddings, gallery))
            self.assertAlmostEqual(metrics["raw_margin"], raw_margin(scores).mean().item(), places=6)
            self.assertAlmostEqual(metrics["legacy_margin"], top_two_log_margin(torch.softmax(scores/.3, dim=1)).mean().item(), places=6)
        self.assertEqual(audit["counterfactual_image_rows"], 12)


if __name__ == "__main__":
    unittest.main()
