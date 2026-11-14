"""Protect Table 3 populations, paired baselines, and measured-run coverage."""
import csv
import json
import math
from pathlib import Path
import tempfile
import unittest

from RQ4_Enhancement_Table import ATTACKS, ENHANCEMENTS, METRICS, build_table3, export_table3


def measured_rows(seeds=(42,)):
    # Deliberately separate attack from clean utility and permit worsening.
    values = {
        "DirectAdd": {"none": (0.48, 0.15, 0.72, 0.39),
                      "type": (0.21, 0.18, 0.62, 0.32),
                      "plausibility": (0.35, 0.12, 0.70, 0.37)},
        "ComAttack": {"none": (0.27, 0.10, 0.64, 0.26),
                      "type": (0.29, 0.11, 0.61, 0.25),
                      "plausibility": (0.19, 0.08, 0.67, 0.27)},
    }
    return [{"attack": attack, "enhancement": enhancement, "seed": seed,
             "num_targets": 100, **dict(zip(METRICS, values[attack][enhancement]))}
            for attack in ATTACKS for enhancement in ENHANCEMENTS for seed in seeds]


class EnhancementTableTests(unittest.TestCase):
    def test_type_uses_set_b_and_plausibility_uses_set_a_of_each_variant(self):
        rows = build_table3(reversed(measured_rows()), [42])
        self.assertEqual([row["attack"] for row in rows], list(ATTACKS))
        direct, composition = rows
        self.assertEqual(direct["num_seeds"], 1)
        self.assertEqual([direct[f"type_{period}_{metric}"] for period in ("before", "after")
                          for metric in ("Hit_at_10", "MRR")], [0.48, 0.15, 0.21, 0.18])
        self.assertEqual([direct[f"plausibility_{period}_{metric}"] for period in ("before", "after")
                          for metric in ("Hit_at_10", "MRR")], [0.72, 0.39, 0.70, 0.37])
        self.assertEqual(composition["type_before_Hit_at_10"], 0.27)
        self.assertEqual(composition["plausibility_after_MRR"], 0.27)

    def test_both_before_columns_come_from_unenhanced_poisoned_victim(self):
        runs = measured_rows()
        baseline = runs[0]
        baseline.update(set_B_Hit_at_10=0.02, set_B_MRR=0.01,
                        set_A_Hit_at_10=0.92, set_A_MRR=0.81,
                        clean_Hit_at_10=0.99, clean_MRR=0.98)
        row = build_table3(runs, [42])[0]
        self.assertEqual((row["type_before_Hit_at_10"], row["type_before_MRR"]), (0.02, 0.01))
        self.assertEqual((row["plausibility_before_Hit_at_10"], row["plausibility_before_MRR"]), (0.92, 0.81))

    def test_multiple_seeds_are_averaged_without_rounding(self):
        runs = measured_rows((42, 43))
        for row in runs:
            if row["seed"] == 43:
                for metric in METRICS:
                    row[metric] += 0.123456
        averaged = build_table3(reversed(runs), [43, 42])
        self.assertEqual(averaged[0]["num_seeds"], 2)
        self.assertAlmostEqual(averaged[0]["type_after_MRR"], 0.241728)
        self.assertAlmostEqual(averaged[1]["plausibility_before_Hit_at_10"], 0.701728)

    def test_missing_or_duplicate_run_cannot_hide_incomplete_coverage(self):
        runs = measured_rows()
        with self.assertRaisesRegex(ValueError, "Incomplete.*DirectAdd.*none.*42"):
            build_table3(runs[1:], [42])
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            build_table3(runs[1:] + [runs[1]], [42])

    def test_unexpected_attack_variant_or_seed_is_rejected(self):
        for changed in ({"attack": "MaSS"}, {"enhancement": "type+plausibility"}, {"seed": 43}):
            with self.subTest(changed=changed):
                runs = measured_rows()
                runs[0].update(changed)
                with self.assertRaisesRegex(ValueError, "Unexpected"):
                    build_table3(runs, [42])

    def test_every_metric_must_be_finite_and_in_range(self):
        for metric in METRICS:
            for value in (math.nan, math.inf, -math.inf, -0.01, 1.01, True, None, "bad"):
                with self.subTest(metric=metric, value=value):
                    runs = measured_rows()
                    runs[-1][metric] = value
                    with self.assertRaisesRegex(ValueError, metric):
                        build_table3(runs, [42])

    def test_missing_fields_and_invalid_seed_configuration_are_rejected(self):
        runs = measured_rows()
        del runs[0]["set_A_MRR"]
        with self.assertRaisesRegex(ValueError, "must contain.*set_A_MRR"):
            build_table3(runs, [42])
        for seeds in ([], [42, 42], [-1], [True], [42.0]):
            with self.subTest(seeds=seeds), self.assertRaises(ValueError):
                build_table3(measured_rows(), seeds)
        for value in (-1, True, 42.0):
            runs = measured_rows()
            runs[0]["seed"] = value
            with self.subTest(seed=value), self.assertRaisesRegex(ValueError, "seed"):
                build_table3(runs, [42])

    def test_optional_target_count_is_positive_and_other_metadata_is_ignored(self):
        runs = measured_rows()
        del runs[0]["num_targets"]
        runs[1]["extra_metadata"] = "ignored"
        self.assertEqual(len(build_table3(runs, [42])), 2)
        for value in (0, -1, 1.5, True, None):
            runs = measured_rows()
            runs[0]["num_targets"] = value
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "num_targets"):
                build_table3(runs, [42])

    def test_export_preserves_measured_values_and_labels_populations(self):
        with tempfile.TemporaryDirectory() as temporary:
            result = export_table3(measured_rows(), temporary, seeds=[42])
            self.assertEqual(set(result), {"rows", "paths"})
            self.assertEqual(set(result["paths"]), {"csv", "json", "md", "png", "pdf"})
            paths = result["paths"]
            for extension, path in paths.items():
                self.assertIsInstance(path, Path)
                self.assertEqual(path.name, f"table3_umls.{extension}")
                self.assertGreater(path.stat().st_size, 0)
            self.assertEqual(json.loads(paths["json"].read_text(encoding="utf-8")), result["rows"])
            with paths["csv"].open(newline="", encoding="utf-8") as stream:
                exported = list(csv.DictReader(stream))
            self.assertEqual(exported[0]["attack"], "DirectAdd")
            self.assertEqual(float(exported[0]["plausibility_before_MRR"]), 0.39)
            markdown = paths["md"].read_text(encoding="utf-8")
            for label in ("UMLS", "ConvE", "Set B attack effectiveness", "Set A clean utility",
                          "local interpretation", "0.4800", "0.7000"):
                self.assertIn(label, markdown)
            self.assertTrue(paths["png"].read_bytes().startswith(b"\x89PNG\r\n\x1a\n"))
            self.assertTrue(paths["pdf"].read_bytes().startswith(b"%PDF"))

    def test_invalid_export_creates_no_results(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "uncreated"
            with self.assertRaises(ValueError):
                export_table3([], output, seeds=[42])
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
