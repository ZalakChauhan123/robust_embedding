"""Regression checks for timing MaSS search with preloaded input resources."""
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

MASS_DIR = Path(__file__).resolve().parents[2]
MODULE_DIR = MASS_DIR / "Path_Generation_Module"
for directory in (MASS_DIR, MODULE_DIR):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

import Entity_Selection as selection


class RankingContextTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.checkpoint = MASS_DIR / "outputs" / "Path_Generation" / "transe_umls.pt"
        cls.experiment = MASS_DIR / "outputs" / "Experiments" / "UMLS_feasible_rank11_15_seed42"
        target_lines = (cls.experiment / "setup" / "set_B_target_triples.tsv").read_text(encoding="utf-8-sig").splitlines()
        cls.target = tuple(next(line for line in target_lines if line.strip() and not line.startswith("#")).split("\t"))
        cls.context = selection.load_ranking_context(cls.checkpoint, "supervised", 0.9)
        cls.options = {"checkpoint": cls.checkpoint, "pra_method": "supervised", "threshold": 0.9,
                       "top_k": 10, "top_entities": 10}
        cls.default_result = selection.rank_paths(*cls.target, **cls.options)

    def rank(self, **changes):
        return selection.rank_paths(*self.target, context=self.context, **{**self.options, **changes})

    def test_preloaded_and_default_results_match_exactly(self):
        self.assertEqual(self.rank(), self.default_result)

    def test_ranking_matches_existing_saved_candidate_payload(self):
        saved = json.loads((self.experiment / "feasibility" / "target_001.json").read_text(encoding="utf-8"))
        self.assertEqual(self.rank(), saved)

    def test_loaded_inputs_are_not_reloaded_per_target(self):
        with patch.object(selection.Surrogate, "load", side_effect=AssertionError("checkpoint reload")), \
             patch.object(selection, "load_dataset", side_effect=AssertionError("training data reload")), \
             patch.object(selection, "SupervisedPRA", side_effect=AssertionError("PRA reload")), \
             patch.object(selection, "load_semantic_constraints", side_effect=AssertionError("ontology reload")), \
             patch.object(selection, "load_dataset_entities", side_effect=AssertionError("entity reload")):
            self.assertEqual(self.rank(), self.default_result)

    def test_pra_and_semantic_candidates_are_recomputed_every_call(self):
        with patch.object(self.context.translator, "translate_relation",
                          wraps=self.context.translator.translate_relation) as translate, \
             patch.object(selection, "apply_semantic_constraints_to_pra",
                          wraps=selection.apply_semantic_constraints_to_pra) as semantic:
            first = self.rank()
            first["ranked_candidate_paths"].clear()
            self.assertEqual(self.rank(), self.default_result)
        self.assertEqual(translate.call_count, 2)
        self.assertEqual(semantic.call_count, 2)

    def test_configuration_mismatches_are_rejected(self):
        for changes in ({"threshold": 0.8}, {"pra_method": "empirical"},
                        {"checkpoint": self.checkpoint.with_name("different_surrogate.pt")}):
            with self.subTest(changes=changes), self.assertRaisesRegex(ValueError, "context"):
                self.rank(**changes)

    def test_context_may_supply_checkpoint_when_argument_is_omitted(self):
        self.assertEqual(self.rank(checkpoint=None), self.default_result)


if __name__ == "__main__":
    unittest.main()
