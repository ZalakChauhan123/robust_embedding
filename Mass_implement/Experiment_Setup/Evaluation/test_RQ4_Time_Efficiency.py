"""Attack-selection integrity checks for the generation-only timing runner."""
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import Evaluate_RQ4_Time_Efficiency as timing


def path(first, second):
    return {"first_triple": list(first), "second_triple": list(second)}


class TimingGenerationTests(unittest.TestCase):
    def setUp(self):
        self.target = ("h", "target_relation", "t")
        self.args = SimpleNamespace(checkpoint=Path("surrogate.pt"), threshold=.9,
                                    top_k=10, top_entities=10, delta=.7,
                                    seed=42, promotion_step=.1)
        self.resources = timing.Resources(
            context=SimpleNamespace(surrogate=object()), targets=[self.target],
            clean=set(), held_out=set(), original_known=set(), target_set={self.target},
            entities=["h", "t", "m1", "m2"], relations=["r1", "r2"],
        )

    def test_mass_skips_an_overbudget_path_atomically_and_keeps_shared_edges(self):
        existing = ("m1", "existing", "t")
        self.resources.clean.add(existing)
        self.resources.original_known.add(existing)
        first = path(("h", "r1", "m1"), existing)
        too_large = path(("h", "r3", "m2"), ("m2", "r4", "t"))
        shared = path(("h", "r1", "m1"), ("m1", "r5", "t"))
        candidates = [first, too_large, shared]
        prior = ("old_h", "old_r", "old_t")
        injected = {prior}
        with patch.object(timing, "rank_paths", return_value={"ranked_candidate_paths": candidates}), \
             patch.object(timing, "sample_paths", return_value={
                 "selected_candidate_paths": candidates, "num_selected_paths": 3}) as sample:
            result = timing.generate_target("MaSS", self.target, 7, 2, injected, self.resources, self.args)
        self.assertEqual(result.selected, [first, shared])
        self.assertEqual(result.attributed, {("h", "r1", "m1"), ("m1", "r5", "t")})
        self.assertEqual(injected, result.attributed | {prior})
        self.assertEqual(sample.call_args.kwargs["seed"], 49)
        self.assertIsNone(sample.call_args.kwargs["max_paths"])
        timing.validate_generated("MaSS", result, 2, self.resources)

    def test_comattack_budget_four_includes_two_complete_bodies(self):
        # The ComAttack candidate API returns tuple edges; MaSS JSON paths use lists.
        bodies = [{"first_triple": ("h", "r1", middle), "second_triple": (middle, "r2", "t")}
                  for middle in ("m1", "m2", "m3")]
        injected = set()
        with patch.object(timing, "comattack_candidates", return_value=(bodies, {})):
            result = timing.generate_target("ComAttack", self.target, 0, 4,
                                            injected, self.resources, self.args)
        self.assertEqual(result.selected, bodies[:2])
        self.assertEqual(len(result.attributed), 4)
        self.assertEqual(injected, result.attributed)
        self.assertNotIn(("h", "r1", "m3"), injected)
        timing.validate_generated("ComAttack", result, 4, self.resources)

    def test_directadd_uses_the_current_global_state_and_top_budget(self):
        candidates = [{"edge": ("h", "r1", middle)} for middle in ("m1", "m2", "m3")]
        injected = {("prior", "r1", "t")}
        with patch.object(timing, "directadd_candidates", return_value=(candidates, {})) as generate:
            result = timing.generate_target("DirectAdd", self.target, 0, 2,
                                            injected, self.resources, self.args)
        self.assertIs(generate.call_args.kwargs["globally_injected"], injected)
        self.assertEqual(result.attributed, {item["edge"] for item in candidates[:2]})
        self.assertEqual(len(injected), 3)
        timing.validate_generated("DirectAdd", result, 2, self.resources)

    def test_fresh_global_sets_do_not_share_state_between_runs(self):
        candidates = [{"edge": ("h", "r1", "m1")}]
        first_state, second_state = set(), set()
        with patch.object(timing, "directadd_candidates", return_value=(candidates, {})):
            first = timing.generate_target("DirectAdd", self.target, 0, 2,
                                           first_state, self.resources, self.args)
            self.assertFalse(second_state)
            second = timing.generate_target("DirectAdd", self.target, 0, 2,
                                            second_state, self.resources, self.args)
        self.assertEqual(first.attributed, second.attributed)
        first_state.clear()
        self.assertEqual(second_state, {("h", "r1", "m1")})

    def test_no_candidate_target_remains_a_valid_attempt_with_zero_injections(self):
        with patch.object(timing, "directadd_candidates", return_value=([], {})):
            result = timing.generate_target("DirectAdd", self.target, 0, 2,
                                            set(), self.resources, self.args)
        self.assertEqual(result.attributed, set())
        self.assertEqual(result.candidate_count, 0)
        timing.validate_generated("DirectAdd", result, 2, self.resources)

    def test_partial_composition_body_is_rejected(self):
        body = path(("h", "r1", "m1"), ("m1", "r2", "t"))
        result = timing.GeneratedAttack({("h", "r1", "m1")}, [body], 1, 1)
        with self.assertRaisesRegex(AssertionError, "partially selected"):
            timing.validate_generated("ComAttack", result, 2, self.resources)


if __name__ == "__main__":
    unittest.main()
