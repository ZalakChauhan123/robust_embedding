"""Tests that prevent incomplete or invalid runs from becoming RQ4 curves."""
import math
import unittest

from RQ4_Embedding_Plot import aggregate_runs


def run(dimension=32, seed=42, hits=0.4, mrr=0.2, **extra):
    return {"surrogate_dim": dimension, "seed": seed,
            "Hit_at_10": hits, "MRR": mrr, **extra}


class EmbeddingAggregationTests(unittest.TestCase):
    def test_known_means_and_sample_standard_deviations(self):
        rows = [run(32, seed, hits, mrr) for seed, hits, mrr in
                [(7, 0.1, 0.2), (11, 0.3, 0.4), (19, 0.5, 0.6)]]
        rows += [run(128, seed, hits, 0.1, ignored=math.nan) for seed, hits in
                 [(7, 0.4), (11, 0.6), (19, 0.8)]]
        result = aggregate_runs(reversed(rows), dimensions=[128, 32], seeds=[19, 7, 11])
        self.assertEqual([row["surrogate_dim"] for row in result], [32, 128])
        self.assertEqual(result[0]["num_seeds"], 3)
        self.assertAlmostEqual(result[0]["Hit_at_10_mean"], 0.3)
        self.assertAlmostEqual(result[0]["Hit_at_10_std"], 0.2)
        self.assertAlmostEqual(result[0]["MRR_mean"], 0.4)
        self.assertAlmostEqual(result[0]["MRR_std"], 0.2)
        self.assertAlmostEqual(result[1]["Hit_at_10_mean"], 0.6)
        self.assertAlmostEqual(result[1]["Hit_at_10_std"], 0.2)
        self.assertEqual(result[1]["MRR_std"], 0.0)

    def test_one_seed_is_not_given_a_fabricated_zero_standard_deviation(self):
        result = aggregate_runs([run(32), run(64), run(128)], [32, 64, 128], [42])
        for row in result:
            self.assertEqual(row["num_seeds"], 1)
            self.assertIsNone(row["Hit_at_10_std"])
            self.assertIsNone(row["MRR_std"])

    def test_missing_dimension_seed_pair_is_rejected(self):
        rows = [run(32, 1), run(32, 2), run(64, 1)]
        with self.assertRaisesRegex(ValueError, "Incomplete.*64, 2"):
            aggregate_runs(rows, [32, 64], [1, 2])

    def test_duplicate_pair_cannot_substitute_for_a_missing_run(self):
        rows = [run(32, 1), run(32, 1), run(64, 1)]
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            aggregate_runs(rows, [32, 64], [1, 2])

    def test_unrequested_dimension_or_seed_is_rejected(self):
        for extra in (run(128, 42), run(32, 99)):
            with self.subTest(extra=extra):
                with self.assertRaisesRegex(ValueError, "Unexpected"):
                    aggregate_runs([run(), extra], [32], [42])

    def test_metrics_must_be_finite_and_within_unit_interval(self):
        for key in ("Hit_at_10", "MRR"):
            for invalid in (math.nan, math.inf, -math.inf, -0.01, 1.01, None, "bad", True):
                row = run()
                row[key] = invalid
                with self.subTest(metric=key, value=invalid):
                    with self.assertRaisesRegex(ValueError, key):
                        aggregate_runs([row], [32], [42])

    def test_metric_endpoints_are_valid(self):
        result = aggregate_runs([run(hits=1, mrr=0)], [32], [42])
        self.assertEqual(result[0]["Hit_at_10_mean"], 1.0)
        self.assertEqual(result[0]["MRR_mean"], 0.0)

    def test_missing_required_metric_is_rejected(self):
        row = run()
        del row["MRR"]
        with self.assertRaisesRegex(ValueError, "must contain"):
            aggregate_runs([row], [32], [42])

    def test_dimensions_and_seeds_require_unique_nonempty_integer_inputs(self):
        invalid_configurations = [([], [42]), ([32], []), ([32, 32], [42]),
                                  ([32], [42, 42]), ([0], [42]), ([32], [-1]),
                                  ([32.5], [42]), ([True], [42])]
        for dimensions, seeds in invalid_configurations:
            with self.subTest(dimensions=dimensions, seeds=seeds):
                with self.assertRaises(ValueError):
                    aggregate_runs([], dimensions, seeds)


if __name__ == "__main__":
    unittest.main()
