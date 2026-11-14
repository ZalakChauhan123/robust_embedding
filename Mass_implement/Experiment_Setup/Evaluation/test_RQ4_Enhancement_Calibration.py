"""Training-free checks for the UMLS Table 3 enhancement eligibility rules."""
import itertools
import math
import sys
import unittest
from pathlib import Path

import numpy as np
import torch

MASS_DIR = Path(__file__).resolve().parents[2]
if str(MASS_DIR) not in sys.path:
    sys.path.insert(0, str(MASS_DIR))

from Path_Generation_Module.TransE_Surrogate import Surrogate, TransE
from RQ4_Attack_Enhancements import (
    EnhancementFilter,
    SurrogateScores,
    calibrate_plausibility,
    type_allowed,
)


class EnhancementCalibrationTests(unittest.TestCase):
    def setUp(self):
        self.entities = ["h", "m", "t", "outlier"]
        self.relations = ["a", "b"]
        model = TransE(len(self.entities), len(self.relations), dim=2)
        with torch.no_grad():
            model.entity.weight.copy_(torch.tensor([[0., 0.], [1., 0.], [3., 0.], [100., 2.]]))
            model.relation.weight.copy_(torch.tensor([[0., 0.], [.5, .5]]))
        self.surrogate = Surrogate(
            model,
            {name: index for index, name in enumerate(self.entities)},
            {name: index for index, name in enumerate(self.relations)},
        )
        self.scores = SurrogateScores.from_surrogate(self.surrogate)
        self.clean = [("h", "a", "m"), ("h", "a", "t"), ("m", "a", "t")]
        self.domain = {"a": {"h", "m"}, "b": {"outlier"}}
        self.range_ = {"a": {"m", "t"}, "b": {"h"}}

    def test_score_cube_matches_scalar_negative_l2_for_every_tiny_triple(self):
        self.assertEqual(self.scores.values.shape, (2, 4, 4))
        for head, relation, tail in itertools.product(self.entities, self.relations, self.entities):
            with self.subTest(triple=(head, relation, tail)):
                self.assertAlmostEqual(
                    self.scores.score((head, relation, tail)),
                    self.surrogate.triple_score(head, relation, tail), places=6)
        for triple in (("unknown", "a", "m"), ("h", "unknown", "m"), ("h", "a", "unknown")):
            with self.subTest(unknown=triple), self.assertRaises(ValueError):
                self.scores.score(triple)

    def test_global_median_uses_only_the_supplied_clean_training_facts(self):
        calibration = calibrate_plausibility(self.scores, self.clean)
        self.assertEqual(calibration["quantile"], .5)
        self.assertEqual(calibration["threshold"], -2.)
        self.assertEqual(calibration["num_clean_train_triples"], 3)
        self.assertEqual(calibration["min"], -3.)
        self.assertEqual(calibration["max"], -1.)

        # Scores for non-training triples can change without changing the
        # calibration. Such triples affect it only if explicitly supplied.
        outlier = ("h", "a", "outlier")
        altered = SurrogateScores(self.scores.values.copy(), self.scores.entity_to_id,
                                  self.scores.relation_to_id)
        altered.values[0, 0, 3] = -10000.
        self.assertEqual(calibrate_plausibility(altered, self.clean), calibration)
        included = calibrate_plausibility(altered, self.clean + [outlier])
        self.assertEqual(included["num_clean_train_triples"], 4)
        self.assertEqual(included["threshold"], -2.5)

    def test_quantile_bounds_are_inclusive_and_invalid_calibration_is_rejected(self):
        self.assertEqual(calibrate_plausibility(self.scores, self.clean, quantile=0)["threshold"], -3.)
        self.assertEqual(calibrate_plausibility(self.scores, self.clean, quantile=1)["threshold"], -1.)
        for quantile in (-.01, 1.01, math.nan, math.inf, -math.inf):
            with self.subTest(quantile=quantile), self.assertRaises(ValueError):
                calibrate_plausibility(self.scores, self.clean, quantile=quantile)
        with self.assertRaises(ValueError):
            calibrate_plausibility(self.scores, [])
        for value in (math.nan, math.inf, -math.inf):
            altered = SurrogateScores(self.scores.values.copy(), self.scores.entity_to_id,
                                      self.scores.relation_to_id)
            altered.values[0, 0, 1] = value
            with self.subTest(score=value), self.assertRaises(ValueError):
                calibrate_plausibility(altered, self.clean)

    def test_nonfinite_surrogate_score_cubes_are_rejected(self):
        for value in (math.nan, math.inf):
            with torch.no_grad():
                self.surrogate.model.entity.weight[0, 0] = value
            with self.subTest(weight=value), self.assertRaises(ValueError):
                SurrogateScores.from_surrogate(self.surrogate)

    def test_type_eligibility_uses_marginal_domain_and_range(self):
        # Each allowed head may pair with each allowed tail; there is no
        # mandatory exact entity-pair listing in this type-constraint rule.
        for head, tail in itertools.product(("h", "m"), ("m", "t")):
            self.assertTrue(type_allowed((head, "a", tail), self.domain, self.range_))
        for triple in (("t", "a", "m"), ("h", "a", "h"),
                       ("unknown", "a", "m"), ("h", "a", "unknown"),
                       ("h", "unknown", "m")):
            with self.subTest(triple=triple):
                self.assertFalse(type_allowed(triple, self.domain, self.range_))

    def test_filter_modes_are_independent_and_plausibility_boundary_is_inclusive(self):
        common = dict(domain=self.domain, range_=self.range_, scores=self.scores, threshold=-2.)
        type_filter = EnhancementFilter("type", **common)
        plausibility_filter = EnhancementFilter("plausibility", **common)
        none_filter = EnhancementFilter("none", **common)

        low_plausibility_type_valid = ("h", "a", "t")  # -3 < -2
        self.assertTrue(type_filter(low_plausibility_type_valid))
        self.assertFalse(plausibility_filter(low_plausibility_type_valid))
        self.assertTrue(none_filter(low_plausibility_type_valid))

        type_invalid_at_boundary = ("t", "a", "m")  # -2 == cutoff
        self.assertFalse(type_filter(type_invalid_at_boundary))
        self.assertTrue(plausibility_filter(type_invalid_at_boundary))
        self.assertTrue(none_filter(type_invalid_at_boundary))

        type_invalid_above_boundary = ("m", "a", "h")  # -1 > -2
        self.assertFalse(type_filter(type_invalid_above_boundary))
        self.assertTrue(plausibility_filter(type_invalid_above_boundary))

        for mode in ("unknown", "both"):
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                EnhancementFilter(mode, **common)
        for value in (math.nan, math.inf, -math.inf):
            with self.subTest(threshold=value), self.assertRaises(ValueError):
                EnhancementFilter("plausibility", self.domain, self.range_, self.scores, value)


if __name__ == "__main__":
    unittest.main()
