"""Protect timing units, attempted-target denominators, and complete coverage."""
import math
import unittest

from RQ4_Time_Plot import aggregate_timing_runs


def run(budget=2, method="MaSS", repeat=0, elapsed=100_000_000, **extra):
    return {"budget": budget, "method": method, "repeat": repeat,
            "num_targets": 100, "total_elapsed_ns": elapsed,
            "targets_with_injections": 96, "unique_injected": 162, **extra}


class TimingAggregationTests(unittest.TestCase):
    def test_ns_are_converted_to_ms_per_attempted_target(self):
        result = aggregate_timing_runs([run(targets_with_injections=5, unique_injected=10)],
                                       [2], ["MaSS"], 1)[0]
        self.assertEqual(result["mean_ms_per_target"], 1.0)
        self.assertEqual(result["num_targets"], 100)
        self.assertEqual(result["targets_with_injections"], 5)
        self.assertEqual(result["unique_injected"], 10)
        self.assertIsNone(result["std_ms_per_target"])

    def test_known_mean_and_sample_standard_deviation(self):
        runs = [run(repeat=repeat, elapsed=elapsed) for repeat, elapsed in
                enumerate((100_000_000, 200_000_000, 300_000_000))]
        result = aggregate_timing_runs(reversed(runs), [2], ["MaSS"], 3)[0]
        self.assertEqual(result["num_repeats"], 3)
        self.assertAlmostEqual(result["mean_ms_per_target"], 2)
        self.assertAlmostEqual(result["std_ms_per_target"], 1)
        self.assertEqual(result["min_ms_per_target"], 1)
        self.assertEqual(result["max_ms_per_target"], 3)

    def test_complete_cartesian_coverage_is_sorted_by_budget(self):
        budgets = [6, 2]
        methods = ["DirectAdd", "ComAttack", "MaSS"]
        runs = [run(budget, method, repeat, elapsed=(budget + repeat) * 100_000_000)
                for budget in budgets for method in methods for repeat in range(2)]
        result = aggregate_timing_runs(reversed(runs), budgets, methods, 2)
        self.assertEqual([(row["budget"], row["method"]) for row in result],
                         [(budget, method) for budget in [2, 6] for method in methods])

    def test_duplicate_cannot_replace_missing_pass(self):
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            aggregate_timing_runs([run(), run()], [2], ["MaSS"], 2)

    def test_missing_pass_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Incomplete.*MaSS.*1"):
            aggregate_timing_runs([run()], [2], ["MaSS"], 2)

    def test_unrequested_budget_method_or_repeat_is_rejected(self):
        for row in (run(budget=4), run(method="Other"), run(repeat=1)):
            with self.subTest(row=row), self.assertRaisesRegex(ValueError, "Unexpected"):
                aggregate_timing_runs([row], [2], ["MaSS"], 1)

    def test_elapsed_ns_must_be_finite_and_nonnegative(self):
        for value in (math.nan, math.inf, -math.inf, -1, None, True, "bad"):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "total_elapsed_ns"):
                aggregate_timing_runs([run(elapsed=value)], [2], ["MaSS"], 1)

    def test_zero_elapsed_is_valid_and_does_not_create_error_bars(self):
        row = run(elapsed=0, targets_with_injections=0, unique_injected=0)
        result = aggregate_timing_runs([row], [2], ["MaSS"], 1)[0]
        self.assertEqual(result["mean_ms_per_target"], 0)
        self.assertIsNone(result["std_ms_per_target"])

    def test_target_denominator_must_be_positive_integer(self):
        for value in (0, -1, 1.5, None, True):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "num_targets"):
                aggregate_timing_runs([run(num_targets=value)], [2], ["MaSS"], 1)

    def test_target_denominator_must_match_across_all_configurations(self):
        rows = [run(), run(method="DirectAdd", num_targets=101)]
        with self.assertRaisesRegex(ValueError, "same num_targets"):
            aggregate_timing_runs(rows, [2], ["MaSS", "DirectAdd"], 1)

    def test_deterministic_coverage_and_injection_counts_cannot_change(self):
        for changed in ({"targets_with_injections": 95}, {"unique_injected": 161}):
            with self.subTest(changed=changed), self.assertRaisesRegex(ValueError, "changed across repeats"):
                aggregate_timing_runs([run(), run(repeat=1, **changed)], [2], ["MaSS"], 2)

    def test_missing_fields_and_invalid_coverage_are_rejected(self):
        row = run()
        del row["unique_injected"]
        with self.assertRaisesRegex(ValueError, "must contain"):
            aggregate_timing_runs([row], [2], ["MaSS"], 1)
        for changed in ({"targets_with_injections": 101}, {"unique_injected": -1},
                        {"targets_with_injections": 0}, {"unique_injected": 0}):
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                aggregate_timing_runs([run(**changed)], [2], ["MaSS"], 1)

    def test_configuration_must_be_nonempty_unique_and_integer(self):
        for budgets, methods, repeats in (([], ["MaSS"], 1), ([2], [], 1),
                                          ([2, 2], ["MaSS"], 1), ([2], ["MaSS", "MaSS"], 1),
                                          ([0], ["MaSS"], 1), ([True], ["MaSS"], 1),
                                          ([2], ["MaSS"], 0), ([2], ["MaSS"], 1.5),
                                          ([2], ["MaSS"], True), ([2], [""], 1)):
            with self.subTest(budgets=budgets, methods=methods, repeats=repeats):
                with self.assertRaises(ValueError):
                    aggregate_timing_runs([], budgets, methods, repeats)


if __name__ == "__main__":
    unittest.main()
