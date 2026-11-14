"""Small artifact/resume and experimental-control checks without model training."""
import contextlib
import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

from Evaluate_RQ4_Embedding_Sensitivity import complete_stage, verify_clean_controls


class CompletedStageTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="rq4_stage_test_")
        self.addCleanup(self.temporary.cleanup)
        self.point = Path(self.temporary.name)
        self.state = {"surrogate_dim": 32, "seed": 42, "stages": {}}
        self.stage = "surrogate"

    def complete(self, operation, *, allow_run=True):
        with contextlib.redirect_stdout(io.StringIO()):
            complete_stage(self.point, self.state, self.stage, operation, allow_run=allow_run)

    def write_artifacts(self):
        directory = self.point / self.stage
        directory.mkdir()
        (directory / "checkpoint.bin").write_bytes(b"synthetic checkpoint fixture")
        nested = directory / "diagnostics"
        nested.mkdir()
        (nested / "loss.csv").write_bytes(b"epoch,loss\n1,0.25\n")

    def test_successful_callback_records_and_persists_every_artifact_hash(self):
        operation = Mock(side_effect=self.write_artifacts)
        self.complete(operation)
        operation.assert_called_once_with()
        record = self.state["stages"][self.stage]
        expected = {
            "surrogate/checkpoint.bin": hashlib.sha256(b"synthetic checkpoint fixture").hexdigest(),
            "surrogate/diagnostics/loss.csv": hashlib.sha256(b"epoch,loss\n1,0.25\n").hexdigest(),
        }
        self.assertEqual(record["sha256"], expected)
        self.assertGreaterEqual(record["elapsed_seconds"], 0)
        persisted = json.loads((self.point / "stage_manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(persisted, self.state)

    def test_completed_stage_verifies_files_and_skips_callback_in_both_modes(self):
        self.complete(self.write_artifacts)
        saved_manifest = (self.point / "stage_manifest.json").read_bytes()
        for allow_run in (True, False):
            with self.subTest(allow_run=allow_run):
                operation = Mock(side_effect=AssertionError("completed callback must not run"))
                self.complete(operation, allow_run=allow_run)
                operation.assert_not_called()
                self.assertEqual((self.point / "stage_manifest.json").read_bytes(), saved_manifest)

    def test_changed_artifact_is_rejected_before_callback(self):
        self.complete(self.write_artifacts)
        (self.point / self.stage / "checkpoint.bin").write_bytes(b"changed artifact")
        operation = Mock(side_effect=AssertionError("corrupt stage must not silently rerun"))
        with self.assertRaisesRegex(ValueError, "artifact changed"):
            self.complete(operation)
        operation.assert_not_called()

    def test_missing_artifact_is_rejected_before_callback(self):
        self.complete(self.write_artifacts)
        (self.point / self.stage / "checkpoint.bin").unlink()
        operation = Mock()
        with self.assertRaisesRegex(ValueError, "artifact changed"):
            self.complete(operation)
        operation.assert_not_called()

    def test_failed_callback_does_not_mark_partial_artifacts_complete(self):
        def fail_after_partial_output():
            directory = self.point / self.stage
            directory.mkdir()
            (directory / "partial.bin").write_bytes(b"unfinished")
            raise RuntimeError("fixture operation failed")

        with self.assertRaisesRegex(RuntimeError, "fixture operation failed"):
            self.complete(fail_after_partial_output)
        self.assertNotIn(self.stage, self.state["stages"])
        self.assertFalse((self.point / "stage_manifest.json").exists())
        self.assertTrue((self.point / self.stage / "partial.bin").exists())

    def test_callback_without_artifacts_is_not_marked_complete(self):
        with self.assertRaisesRegex(AssertionError, "produced no artifacts"):
            self.complete(lambda: None)
        self.assertNotIn(self.stage, self.state["stages"])
        self.assertFalse((self.point / "stage_manifest.json").exists())

    def test_read_only_mode_rejects_incomplete_stage_without_running_callback(self):
        operation = Mock(side_effect=AssertionError("read-only mode must not run callback"))
        with self.assertRaisesRegex(ValueError, "Incomplete point"):
            self.complete(operation, allow_run=False)
        operation.assert_not_called()
        self.assertEqual(self.state["stages"], {})
        self.assertFalse((self.point / "run.log").exists())


def control_row(dimension=32, seed=42, **overrides):
    row = {
        "surrogate_dim": dimension, "seed": seed, "victim_dim": 64,
        "victim_initial_state_sha256": f"initial_state_seed_{seed}",
        "clean_victim_state_sha256": f"clean_state_seed_{seed}",
        "clean_Hit_at_10": 0.23, "clean_MRR": 0.10, "num_set_B_targets": 100,
        "Hit_at_10": 0.42, "MRR": 0.16,
    }
    row.update(overrides)
    return row


class CleanControlTests(unittest.TestCase):
    def test_clean_state_or_metrics_cannot_change_with_surrogate_size(self):
        changes = {
            "victim_dim": 128,
            "victim_initial_state_sha256": "different_initialization",
            "clean_victim_state_sha256": "different_clean_state",
            "clean_Hit_at_10": 0.24,
            "clean_MRR": 0.11,
            "num_set_B_targets": 99,
        }
        for field, changed in changes.items():
            with self.subTest(field=field):
                rows = [control_row(), control_row(dimension=128, **{field: changed})]
                with self.assertRaisesRegex(AssertionError, field):
                    verify_clean_controls(rows)

    def test_separate_seeds_may_have_different_clean_states_and_metrics(self):
        seed_43_values = {"clean_Hit_at_10": 0.31, "clean_MRR": 0.14}
        # Interleaving seeds ensures references are grouped by seed, not order.
        rows = [control_row(32, 42), control_row(64, 43, **seed_43_values),
                control_row(128, 42), control_row(32, 43, **seed_43_values)]
        verify_clean_controls(rows)

    def test_attack_metrics_may_vary_while_clean_control_stays_fixed(self):
        rows = [control_row(32), control_row(64, Hit_at_10=0.55, MRR=0.22),
                control_row(128, Hit_at_10=0.38, MRR=0.15)]
        verify_clean_controls(rows)


if __name__ == "__main__":
    unittest.main()
