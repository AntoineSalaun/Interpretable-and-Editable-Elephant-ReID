"""Checks for correction semantics and the requested paired estimand."""
import unittest
from unittest.mock import patch

import numpy as np
import torch

from experiments.seek_ear_only import run


class ComparisonTests(unittest.TestCase):
    def test_schema_discovery_uses_names(self):
        expected = set(run.ear_attributes())
        with patch.object(run.base.SEEK, 'attribute_names', list(reversed(run.base.SEEK.attribute_names))):
            self.assertEqual(set(run.ear_attributes()), expected)
        self.assertNotIn('ear_special', expected)
        self.assertNotIn('right_extreme', expected)

    def test_image_specific_unknowns_and_other_predictions_preserved(self):
        predicted = torch.cat([torch.eye(run.base.SEEK.lengths[n])[0].repeat(2, 1)
                               for n in run.base.SEEK.attribute_names], dim=1)
        expert = predicted.clone()
        offset = 0
        for n in run.base.SEEK.attribute_names:
            width = run.base.SEEK.lengths[n]
            expert[:, offset:offset+width] = 0
            expert[0, offset+width-1] = 1
            expert[1, offset+run.base.SEEK.mappings[n].index('_') if '_' in run.base.SEEK.mappings[n] else offset] = 1
            offset += width
        result = run.corrected(predicted, expert, run.ear_attributes())
        offset = 0
        for n in run.base.SEEK.attribute_names:
            width = run.base.SEEK.lengths[n]
            reference = expert if n in run.ear_attributes() else predicted
            self.assertTrue(torch.equal(result[:, offset:offset+width], reference[:, offset:offset+width]))
            offset += width
        self.assertTrue(torch.equal(run.corrected(predicted, expert, run.base.SEEK.attribute_names), expert))

    def test_sighting_weighting_and_shared_bootstrap_draws(self):
        hits = np.array([[0, 1], [1, 1], [1, 1]])
        scores = run.sighting_scores(hits, ['a', 'b', 'b'])
        np.testing.assert_array_equal(scores, [[0, 1], [1, 1]])
        self.assertEqual(scores[:, 0].mean(), 0.5)
        self.assertNotEqual(scores[:, 0].mean(), hits[:, 0].mean())
        shifted = np.column_stack([np.arange(5)/10, np.arange(5)/10 + 0.25])
        means = run.paired_bootstrap(shifted, 107, 2026)
        np.testing.assert_allclose(means[:, 1]-means[:, 0], 25, atol=1e-12)


if __name__ == '__main__':
    unittest.main()
