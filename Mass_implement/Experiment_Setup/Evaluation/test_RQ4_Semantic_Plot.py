"""Protect score statistics and count histograms sharing the full score range."""
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from RQ4_Semantic_Plot import (GROUP_ORDER, histogram_counts,
                               plot_semantic_information, summarize_scores)


def rows(correct=(-4, 0, 4), poisoned=(-3, 1, 5), noisy=(-20, -12, -4)):
    return [{"group": group, "score": score}
            for group, scores in zip(GROUP_ORDER, (correct, poisoned, noisy))
            for score in scores]


class SemanticScoreTests(unittest.TestCase):
    def test_known_statistics_and_order_do_not_depend_on_input_order(self):
        result = summarize_scores(reversed(rows()))
        self.assertEqual([row["group"] for row in result], list(GROUP_ORDER))
        correct = result[0]
        self.assertEqual(correct["count"], 3)
        self.assertEqual(correct["mean"], 0)
        self.assertEqual(correct["median"], 0)
        self.assertEqual(correct["std"], 4)
        self.assertEqual(correct["min"], -4)
        self.assertEqual(correct["max"], 4)
        self.assertAlmostEqual(correct["p05"], -3.6)
        self.assertAlmostEqual(correct["p95"], 3.6)

    def test_singleton_has_no_sample_standard_deviation(self):
        result = summarize_scores(rows(correct=(2,), poisoned=(1,), noisy=(-1,)))
        for row in result:
            self.assertEqual(row["count"], 1)
            self.assertIsNone(row["std"])
            self.assertEqual(row["mean"], row["median"])
            self.assertEqual(row["min"], row["max"])
            self.assertEqual(row["p05"], row["p95"])

    def test_missing_or_empty_group_is_rejected(self):
        for data in ([], rows(correct=()), rows(poisoned=()), rows(noisy=())):
            for function in (summarize_scores, histogram_counts):
                with self.subTest(data=data, function=function.__name__):
                    with self.assertRaisesRegex(ValueError, "nonempty.*missing"):
                        function(data)

    def test_group_names_are_exact(self):
        for group in ("MaSS", "correct facts", "Noisy facts ", "", None, []):
            data = rows() + [{"group": group, "score": 0}]
            with self.subTest(group=group), self.assertRaisesRegex(ValueError, "unsupported group"):
                summarize_scores(data)

    def test_scores_must_be_finite_numbers(self):
        for score in (math.nan, math.inf, -math.inf, None, "bad", True, np.bool_(False)):
            data = rows() + [{"group": "Noisy facts", "score": score}]
            for function in (summarize_scores, histogram_counts):
                with self.subTest(score=score, function=function.__name__):
                    with self.assertRaisesRegex(ValueError, "score must be finite"):
                        function(data)

    def test_rows_require_score_and_group_fields(self):
        for row in ({"group": "Noisy facts"}, {"score": 0}, None, 1):
            with self.subTest(row=row), self.assertRaisesRegex(ValueError, "contain group and score"):
                summarize_scores(rows() + [row])


class SemanticHistogramTests(unittest.TestCase):
    def test_known_counts_use_shared_bins_and_include_maximum_score(self):
        data = rows(correct=(-4, 0, 4), poisoned=(-2, 0, 2), noisy=(-4, -4, 0))
        edges, counts = histogram_counts(data, bins=4)
        self.assertEqual(edges, [-4, -2, 0, 2, 4])
        self.assertEqual(list(counts), list(GROUP_ORDER))
        self.assertEqual(counts["Correct facts"], [1, 0, 1, 1])
        self.assertEqual(counts["Poisoned triplets"], [0, 1, 1, 1])
        self.assertEqual(counts["Noisy facts"], [2, 0, 1, 0])
        self.assertTrue(all(sum(values) == 3 for values in counts.values()))

    def test_all_measured_scores_are_retained_without_paper_range_clipping(self):
        data = rows(correct=(0, 10, 20), poisoned=(30, 40), noisy=(-140, -100, -80, -70))
        edges, counts = histogram_counts(data, bins=9)
        self.assertEqual(edges[0], -140)
        self.assertEqual(edges[-1], 40)
        self.assertTrue(np.allclose(np.diff(edges), 20))
        self.assertEqual([sum(counts[group]) for group in GROUP_ORDER], [3, 2, 4])
        self.assertEqual(counts["Poisoned triplets"][-1], 2)

    def test_degenerate_equal_scores_receive_a_finite_nonzero_bin_range(self):
        edges, counts = histogram_counts(rows(correct=(2,), poisoned=(2,), noisy=(2,)), bins=1)
        self.assertEqual(edges, [1.5, 2.5])
        self.assertEqual(list(counts.values()), [[1], [1], [1]])

    def test_bins_must_be_a_positive_integer(self):
        for bins in (0, -1, 1.5, "20", None, True, np.bool_(False)):
            with self.subTest(bins=bins), self.assertRaisesRegex(ValueError, "positive integer"):
                histogram_counts(rows(), bins=bins)

    def test_png_and_pdf_exports_are_created(self):
        with tempfile.TemporaryDirectory() as temporary:
            output_dir = Path(temporary) / "plots"
            outputs = plot_semantic_information(rows(), output_dir, bins=4)
            self.assertEqual(set(outputs), {"png", "pdf"})
            for extension, path in outputs.items():
                self.assertEqual(path.parent, output_dir)
                self.assertEqual(path.name, f"figure8_umls_semantic_information.{extension}")
                self.assertGreater(path.stat().st_size, 100)
            self.assertTrue(outputs["png"].read_bytes().startswith(b"\x89PNG\r\n\x1a\n"))
            self.assertTrue(outputs["pdf"].read_bytes().startswith(b"%PDF-"))

    def test_plot_rejects_invalid_data_before_creating_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            output_dir = Path(temporary) / "invalid"
            with self.assertRaisesRegex(ValueError, "nonempty"):
                plot_semantic_information(rows(noisy=()), output_dir)
            self.assertFalse(output_dir.exists())


if __name__ == "__main__":
    unittest.main()
