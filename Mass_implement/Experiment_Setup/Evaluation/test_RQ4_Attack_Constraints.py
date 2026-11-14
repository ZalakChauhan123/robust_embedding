"""Integrity checks for optional constraints on the existing attack generators."""
import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

SETUP_DIR = Path(__file__).resolve().parents[1]
if str(SETUP_DIR) not in sys.path:
    sys.path.insert(0, str(SETUP_DIR))

import construct_attack_baselines as attacks
from Path_Generation_Module.TransE_Surrogate import Surrogate, TransE


class AttackConstraintTests(unittest.TestCase):
    def setUp(self):
        self.entities = ["h", "m", "t"]
        self.relations = ["a", "target"]
        self.target = ("h", "target", "t")
        model = TransE(len(self.entities), len(self.relations), dim=2)
        with torch.no_grad():
            model.entity.weight.copy_(torch.tensor([[0., 0.], [1., .3], [3., 0.]]))
            model.relation.weight.copy_(torch.tensor([[0., 0.], [1., 0.]]))
        self.surrogate = Surrogate(
            model,
            {entity: index for index, entity in enumerate(self.entities)},
            {relation: index for index, relation in enumerate(self.relations)},
        )
        self.common = dict(
            surrogate=self.surrogate, entities=self.entities, relations=self.relations,
            original_known=set(), target_set={self.target}, globally_injected=set(),
        )

    def test_disabled_directadd_filter_keeps_the_existing_legal_candidate(self):
        allowed = ("h", "a", "m")
        all_proposals = {
            edge for relation in self.relations for entity in self.entities
            for edge in (("h", relation, entity), (entity, relation, "t"))
        }
        self.common["original_known"] = all_proposals - {allowed, self.target}
        original, original_counts = attacks.directadd_candidates(self.target, step=.1, **self.common)
        explicit_none, none_counts = attacks.directadd_candidates(
            self.target, step=.1, edge_filter=None, **self.common)
        accept_all, accept_counts = attacks.directadd_candidates(
            self.target, step=.1, edge_filter=lambda edge: True, **self.common)
        self.assertEqual([record["edge"] for record in original], [allowed])
        self.assertEqual(original, explicit_none)
        self.assertEqual(original, accept_all)
        self.assertEqual(original_counts, none_counts)
        self.assertEqual(original_counts, accept_counts)

    def test_directadd_filter_keeps_a_later_ranked_candidate(self):
        baseline, _ = attacks.directadd_candidates(self.target, step=.1, **self.common)
        allowed = baseline[-1]["edge"]
        candidates, rejected = attacks.directadd_candidates(
            self.target, step=.1, edge_filter=lambda edge: edge == allowed, **self.common)
        self.assertEqual(candidates, [baseline[-1]])
        self.assertGreater(rejected["enhancement"], 0)

    def test_directadd_filter_runs_only_after_hard_integrity_guards(self):
        seen = []

        def reject(edge):
            seen.append(edge)
            return False

        candidates, rejected = attacks.directadd_candidates(
            self.target, step=.1, edge_filter=reject, **self.common)
        self.assertEqual(candidates, [])
        self.assertTrue(seen)
        self.assertTrue(all(edge[0] != edge[2] and edge != self.target for edge in seen))
        # DirectAdd counts candidate proposals before symmetric deduplication.
        self.assertEqual(rejected["enhancement"], len(seen))
        self.assertEqual(rejected["enhancement"], 6)

    def test_comattack_rejects_a_partial_body_and_searches_later_pairs(self):
        baseline, baseline_counts = attacks.comattack_candidates(self.target, **self.common)
        accept_all, accept_counts = attacks.comattack_candidates(
            self.target, edge_filter=lambda edge: True, **self.common)
        self.assertEqual(baseline, accept_all)
        self.assertEqual(baseline_counts, accept_counts)
        self.assertEqual((baseline[0]["r1"], baseline[0]["r2"]), ("a", "target"))

        # The first relation pair has only one legal intermediate. Its first
        # edge passes, its second fails. The next pair has a complete body.
        allowed = {("h", "a", "m"), ("h", "target", "m"), ("m", "a", "t")}
        candidates, rejected = attacks.comattack_candidates(
            self.target, edge_filter=lambda edge: edge in allowed, **self.common)
        self.assertEqual(len(candidates), 1)
        body = candidates[0]
        self.assertEqual((body["r1"], body["r2"]), ("target", "a"))
        self.assertIn(body["first_triple"], allowed)
        self.assertIn(body["second_triple"], allowed)
        self.assertEqual(rejected["enhancement"], 1)

    def test_comattack_all_rejected_bodies_are_counted_once_each(self):
        seen = []

        def reject(edge):
            seen.append(edge)
            return False

        candidates, rejected = attacks.comattack_candidates(
            self.target, edge_filter=reject, **self.common)
        self.assertEqual(candidates, [])
        # All four relation pairs have exactly one legal middle entity.
        self.assertEqual(rejected["enhancement"], 4)
        self.assertEqual(len(seen), 4)
        self.assertTrue(all(edge[0] != edge[2] and edge != self.target for edge in seen))

    def test_constructor_forwards_filter_preserves_budget_and_records_configuration(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for name, triples in {
                "train": [("h", "a", "t")], "valid": [("t", "a", "h")],
                "test": [("m", "a", "h")], "targets": [self.target],
            }.items():
                attacks.write_triples(root / f"{name}.tsv", triples)
            (root / "entities.txt").write_text("\n".join(self.entities), encoding="utf-8")
            (root / "relations.txt").write_text("\n".join(self.relations), encoding="utf-8")
            (root / "surrogate.pt").write_bytes(b"test-checkpoint")
            kwargs = dict(
                targets_file=root / "targets.tsv", train_file=root / "train.tsv",
                valid_file=root / "valid.tsv", test_file=root / "test.tsv",
                entities_file=root / "entities.txt", relations_file=root / "relations.txt",
                surrogate_checkpoint=root / "surrogate.pt", budget=2,
            )
            allowed = ("h", "a", "m")
            with patch.object(attacks.Surrogate, "load", return_value=self.surrogate), \
                 contextlib.redirect_stdout(io.StringIO()):
                attacks.construct("DirectAdd", output_dir=root / "baseline", **kwargs)
                attacks.construct("DirectAdd", output_dir=root / "accept_all",
                                  edge_filter=lambda edge: True, **kwargs)
                report = attacks.construct(
                    "DirectAdd", output_dir=root / "filtered",
                    edge_filter=lambda edge: edge == allowed, enhancement="type_constraint",
                    enhancement_metadata={"source": "test-ontology"}, **kwargs)
            self.assertEqual((root / "baseline" / "injected_triples.tsv").read_bytes(),
                             (root / "accept_all" / "injected_triples.tsv").read_bytes())
            self.assertEqual(attacks.read_triples(root / "filtered" / "injected_triples.tsv"), [allowed])
            self.assertEqual(report["num_unique_injected_triples"], 1)
            self.assertEqual(report["num_targets_set_B"], 1)
            self.assertEqual(report["parameters"]["enhancement"], "type_constraint")
            for filename in ("eligible_candidate_paths.json", "selected_poisoning_paths.json"):
                saved = json.loads((root / "filtered" / filename).read_text(encoding="utf-8"))
                self.assertEqual(saved["enhancement"], "type_constraint")
                self.assertEqual(saved["enhancement_metadata"], {"source": "test-ontology"})


if __name__ == "__main__":
    unittest.main()
