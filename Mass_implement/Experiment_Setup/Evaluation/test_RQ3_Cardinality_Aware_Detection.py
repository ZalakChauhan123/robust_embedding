"""Regression checks for false contradiction claims and rule coverage."""
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from analyze_relation_cardinality import analyze
from RQ3_Cardinality_Aware_Detection import evaluate, examine, load_rules, validate_rules


class CardinalityDetectionTests(unittest.TestCase):
    def test_multiple_interaction_partners_do_not_establish_a_contradiction(self):
        clean = [("hormone", "interacts_with", "chemical"),
                 ("hormone", "interacts_with", "immunologic_factor")]
        _, stats, hr, rt = analyze(clean)
        row = examine("baseline", ("hormone", "interacts_with", "fungus"), stats, hr, rt, {})
        self.assertTrue(row["strict_overlap_flagged"])
        self.assertTrue(row["forward_multi_valued_evidence"])
        self.assertEqual(row["cardinality_status"], "inconclusive_overlap")
        self.assertFalse(row["rule_assessed"])

    def test_documented_forward_uniqueness_detects_only_forward_conflicts(self):
        # A functional relation may still have several heads for one tail.
        clean = [("alice", "birth_city", "paris"), ("bob", "birth_city", "paris")]
        _, stats, hr, rt = analyze(clean)
        rules = {"birth_city": {"functional": True, "source": "synthetic test specification"}}
        validate_rules(rules, stats)
        forward = examine("baseline", ("alice", "birth_city", "london"), stats, hr, rt, rules)
        inverse = examine("baseline", ("carol", "birth_city", "paris"), stats, hr, rt, rules)
        self.assertEqual(forward["cardinality_status"], "confirmed_rule_conflict")
        self.assertEqual(forward["conflict_direction"], "functional")
        self.assertEqual(inverse["cardinality_status"], "inconclusive_overlap")

    def test_documented_inverse_uniqueness_detects_only_inverse_conflicts(self):
        clean = [("alice", "owns_serial", "serial_a"), ("alice", "owns_serial", "serial_b")]
        _, stats, hr, rt = analyze(clean)
        rules = {"owns_serial": {"inverse_functional": True, "source": "synthetic test specification"}}
        validate_rules(rules, stats)
        row = examine("baseline", ("bob", "owns_serial", "serial_a"), stats, hr, rt, rules)
        self.assertEqual(row["cardinality_status"], "confirmed_rule_conflict")
        self.assertEqual(row["conflict_direction"], "inverse_functional")

    def test_uniqueness_rule_refuted_by_clean_data_is_rejected(self):
        _, stats, _, _ = analyze([("hormone", "interacts_with", "chemical"),
                                  ("hormone", "interacts_with", "fungus")])
        with self.assertRaisesRegex(ValueError, "not functional"):
            validate_rules({"interacts_with": {"functional": True}}, stats)
        with self.assertRaisesRegex(ValueError, "Unknown relation"):
            validate_rules({"interact_with_typo": {"functional": True}}, stats)

    def test_a_single_observation_does_not_create_a_uniqueness_rule(self):
        _, stats, hr, rt = analyze([("alice", "birth_city", "paris")])
        row = examine("baseline", ("alice", "birth_city", "london"), stats, hr, rt, {})
        self.assertFalse(row["rule_assessed"])
        self.assertEqual(row["cardinality_status"], "inconclusive_overlap")

    def test_true_rule_requires_a_source(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rules.json"
            for source in (None, 42, [], "", "   "):
                with self.subTest(source=source):
                    path.write_text(json.dumps({"relations": {
                        "birth_city": {"functional": True, "source": source}}}))
                    with self.assertRaisesRegex(ValueError, "documented source"):
                        load_rules(path)

    def test_no_rule_coverage_is_not_reported_as_zero_percent_detection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train = root / "train.txt"
            train.write_text("hormone\tinteracts_with\tchemical\n", encoding="utf-8")
            poisoning = root / "experiment" / "baseline" / "poisoning"
            poisoning.mkdir(parents=True)
            (poisoning / "injected_triples.tsv").write_text(
                "hormone\tinteracts_with\tfungus\n", encoding="utf-8")
            with contextlib.redirect_stdout(io.StringIO()):
                rows = evaluate(root / "experiment", train=train, output=root / "assessment",
                                methods=("baseline",))
            self.assertEqual(rows[0]["rule_coverage_pct"], 0)
            self.assertIsNone(rows[0]["confirmed_conflict_rate_pct"])
            self.assertIsNone(rows[0]["confirmed_conflict_rate_among_assessed_pct"])
            self.assertEqual(rows[0]["confirmed_rule_conflicts"], 0)
            self.assertEqual(rows[0]["inconclusive_overlaps"], 1)
            self.assertEqual(rows[0]["assessment_status"], "not_assessable_no_applicable_rules")


if __name__ == "__main__":
    unittest.main()
