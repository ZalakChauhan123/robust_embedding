"""Check equal sampling, leakage-free corruptions, and raw-logit extraction."""
import math
import unittest

import torch

from Evaluate_RQ4_Semantic_Information import (
    make_noisy_facts,
    score_triples,
    select_fact_samples,
)


class FactSelectionTests(unittest.TestCase):
    def setUp(self):
        self.injected = [(f"p{i}", "r", f"q{i}") for i in range(8)]
        self.correct = [(f"a{i}", "s", f"b{i}") for i in range(12)]

    def test_default_uses_every_injection_with_equal_unique_heldout_sample(self):
        before = (list(self.injected), list(self.correct))
        correct, poisoned = select_fact_samples(self.injected, self.correct, seed=42)
        self.assertEqual(poisoned, self.injected)
        self.assertEqual(len(correct), len(poisoned))
        self.assertEqual(len(set(correct)), len(correct))
        self.assertTrue(set(correct) <= set(self.correct))
        self.assertEqual((self.injected, self.correct), before)

    def test_subsets_are_reproducible_and_unique_in_both_groups(self):
        first = select_fact_samples(self.injected, self.correct, num_facts=4, seed=13)
        self.assertEqual(first, select_fact_samples(self.injected, self.correct, num_facts=4, seed=13))
        for sample, pool in zip(first, (self.correct, self.injected)):
            self.assertEqual(len(sample), 4)
            self.assertEqual(len(set(sample)), 4)
            self.assertTrue(set(sample) <= set(pool))
        self.assertNotEqual(first, select_fact_samples(self.injected, self.correct, num_facts=4, seed=14))

    def test_oversampling_or_nonpositive_counts_are_rejected(self):
        for count in (0, -1, 9, 20):
            with self.subTest(count=count), self.assertRaisesRegex(ValueError, "positive.*fit"):
                select_fact_samples(self.injected, self.correct, num_facts=count)
        with self.assertRaisesRegex(ValueError, "fit"):
            select_fact_samples(self.injected, self.correct[:3])

    def test_duplicate_pools_are_rejected(self):
        cases = ((self.injected + self.injected[:1], self.correct),
                 (self.injected, self.correct + self.correct[:1]))
        for injected, correct in cases:
            with self.subTest(injected=injected, correct=correct), self.assertRaisesRegex(ValueError, "unique"):
                select_fact_samples(injected, correct, num_facts=3)


class NoiseGenerationTests(unittest.TestCase):
    def test_corruptions_are_reproducible_unique_and_disjoint_from_every_exclusion(self):
        entities = list("abcdefgh")
        correct = [("a", "r", "b"), ("c", "r", "d"), ("e", "s", "f"), ("g", "s", "h")]
        train = {("c", "r", "b"), ("a", "r", "d")}
        valid = {("e", "r", "b"), ("c", "r", "f")}
        test = set(correct)
        injected = {("a", "r", "g"), ("e", "s", "a")}
        targets = {("g", "s", "e"), ("b", "s", "h")}
        excluded = train | valid | test | injected | targets
        before = (list(correct), list(entities), set(excluded))
        records = make_noisy_facts(correct, entities, excluded, seed=43)
        self.assertEqual(records, make_noisy_facts(correct, entities, excluded, seed=43))
        self.assertEqual(len(records), len(correct))
        triples = [record["triple"] for record in records]
        self.assertEqual(len(triples), len(set(triples)))
        self.assertFalse(set(triples) & excluded)
        for source, record in zip(correct, records):
            h, r, t = record["triple"]
            self.assertEqual(record["source"], source)
            self.assertEqual(r, source[1])
            self.assertNotEqual(h, t)
            self.assertTrue(h in entities and t in entities)
            self.assertEqual(int(h != source[0]) + int(t != source[2]), 1)
            changed = "head" if h != source[0] else "tail"
            self.assertEqual(record["corrupted_field"], changed)
        self.assertEqual((correct, entities, excluded), before)

    def test_falls_back_to_the_only_legal_direction(self):
        source = ("a", "r", "b")
        for blocked, expected, direction in (
            (("a", "r", "c"), ("c", "r", "b"), "head"),
            (("c", "r", "b"), ("a", "r", "c"), "tail"),
        ):
            for seed in (0, 1, 42):
                with self.subTest(direction=direction, seed=seed):
                    record, = make_noisy_facts([source], list("abc"), {source, blocked}, seed=seed)
                    self.assertEqual(record["triple"], expected)
                    self.assertEqual(record["corrupted_field"], direction)

    def test_both_legal_directions_can_be_selected(self):
        source = ("a", "r", "b")
        directions = {make_noisy_facts([source], list("abc"), {source}, seed=seed)[0]["corrupted_field"]
                      for seed in range(12)}
        self.assertEqual(directions, {"head", "tail"})

    def test_previously_generated_noise_is_not_reused(self):
        source = ("a", "r", "b")
        records = make_noisy_facts([source, source], list("abc"), {source})
        self.assertEqual({record["triple"] for record in records},
                         {("c", "r", "b"), ("a", "r", "c")})

    def test_exhausted_replacements_fail_instead_of_reusing_noise(self):
        source = ("a", "r", "b")
        with self.assertRaisesRegex(ValueError, "No legal one-entity"):
            make_noisy_facts([source] * 3, list("abc"), {source})
        excluded = {source, ("c", "r", "b"), ("a", "r", "c")}
        with self.assertRaisesRegex(ValueError, "No legal one-entity"):
            make_noisy_facts([source], list("abc"), excluded)

    def test_empty_or_duplicate_entity_lists_are_rejected(self):
        for entities in ([], ["a", "a", "b"]):
            with self.subTest(entities=entities), self.assertRaisesRegex(ValueError, "nonempty and unique"):
                make_noisy_facts([("a", "r", "b")], entities, set())


class DummyLogitModel(torch.nn.Module):
    """Tiny analytic scorer: score(h,r,t) = 100*h + 10*r + t - 200."""
    def __init__(self, invalid=None):
        super().__init__()
        self.offset = torch.nn.Parameter(torch.tensor(-200.0))
        self.invalid = invalid
        self.calls = []

    def score_all_tails(self, heads, relations):
        self.calls.append({"batch_size": len(heads), "training": self.training,
                           "grad_enabled": torch.is_grad_enabled(),
                           "inference_enabled": torch.is_inference_mode_enabled()})
        logits = 100 * heads[:, None] + 10 * relations[:, None] + torch.arange(4)[None, :] + self.offset
        if self.invalid is not None:
            logits = torch.full_like(logits, self.invalid)
        return logits


class TripleScoringTests(unittest.TestCase):
    def setUp(self):
        self.entities = {"a": 0, "b": 1, "c": 2, "d": 3}
        self.relations = {"r": 0, "s": 1}
        self.triples = [("a", "r", "d"), ("c", "s", "b"), ("b", "r", "a"),
                        ("d", "s", "c"), ("a", "s", "b")]
        self.expected = [-197.0, 11.0, -100.0, 112.0, -189.0]

    def test_selected_tail_logits_are_exact_and_batch_invariant(self):
        for batch_size in (1, 2, 10):
            model = DummyLogitModel()
            with self.subTest(batch_size=batch_size):
                values = score_triples(model, self.triples, self.entities, self.relations,
                                       batch_size=batch_size)
                self.assertEqual(values, self.expected)
                self.assertEqual(sum(call["batch_size"] for call in model.calls), len(self.triples))
                self.assertLessEqual(max(call["batch_size"] for call in model.calls), batch_size)

    def test_scoring_sets_eval_mode_and_disables_gradient_tracking(self):
        model = DummyLogitModel().train()
        self.assertTrue(model.training)
        score_triples(model, self.triples, self.entities, self.relations, batch_size=2)
        self.assertFalse(model.training)
        self.assertTrue(all(not call["training"] and not call["grad_enabled"]
                            and call["inference_enabled"] for call in model.calls))
        self.assertIsNone(model.offset.grad)

    def test_nonfinite_selected_logits_are_rejected(self):
        for invalid in (math.nan, math.inf, -math.inf):
            with self.subTest(invalid=invalid), self.assertRaisesRegex(ValueError, "nonfinite"):
                score_triples(DummyLogitModel(invalid), self.triples, self.entities, self.relations)

    def test_unknown_head_relation_or_tail_is_rejected_before_inference(self):
        for triple in (("missing", "r", "b"), ("a", "missing", "b"), ("a", "r", "missing")):
            model = DummyLogitModel()
            with self.subTest(triple=triple), self.assertRaisesRegex(ValueError, "Unknown entity or relation"):
                score_triples(model, [triple], self.entities, self.relations)
            self.assertEqual(model.calls, [])

    def test_nonpositive_batch_sizes_are_rejected(self):
        for size in (0, -1):
            with self.subTest(size=size), self.assertRaisesRegex(ValueError, "batch_size must be positive"):
                score_triples(DummyLogitModel(), self.triples, self.entities, self.relations, batch_size=size)


if __name__ == "__main__":
    unittest.main()
