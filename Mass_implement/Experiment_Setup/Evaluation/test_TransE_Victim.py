"""Compatibility regressions for the dedicated TransE victim module."""
from __future__ import annotations

import argparse
import contextlib
import csv
import io
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import torch
from torch.nn import functional as F

MASS = Path(__file__).resolve().parents[2]
for directory in (MASS, MASS / "Experiment_Setup", Path(__file__).parent):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from Path_Generation_Module.TransE_Surrogate import Surrogate, TransE as LegacyTransE
from Path_Generation_Module.TransE_Victim import TransE, TransEVictim
import train_victim_models as training
import Evaluate_victim_models as evaluation


def checkpoint(model, entities, relations, dimension):
    """The existing attacker/victim checkpoint schema, without class pickles."""
    return {"state_dict": model.state_dict(), "entity_to_id": entities,
            "relation_to_id": relations, "embedding_dim": dimension}


class TransEVictimCompatibilityTests(unittest.TestCase):
    def test_victim_architecture_is_independent_of_surrogate_module(self):
        self.assertEqual(TransE.__bases__, (torch.nn.Module,))
        for cwd, module in ((MASS, "Path_Generation_Module.TransE_Victim"),
                            (MASS / "Path_Generation_Module", "TransE_Victim")):
            with self.subTest(module=module):
                script = (
                    "import sys,importlib.abc; "
                    "\nclass BlockSurrogate(importlib.abc.MetaPathFinder):"
                    "\n def find_spec(self,fullname,path=None,target=None):"
                    "\n  if fullname.split('.')[-1]=='TransE_Surrogate':"
                    "\n   raise ImportError('Victim must not import the attacker surrogate')"
                    "\nsys.meta_path.insert(0,BlockSurrogate())"
                    f"\nfrom {module} import TransE,TransEVictim"
                    "\nimport torch"
                    "\nassert TransE.__bases__==(torch.nn.Module,)"
                    "\nmodel=TransE(3,1,4)"
                    "\nwrapper=TransEVictim(model,{'h':0,'m':1,'t':2},{'r':0})"
                    "\nassert isinstance(wrapper.triple_score('h','r','t'),float)"
                    "\nassert not any(name.split('.')[-1]=='TransE_Surrogate' for name in sys.modules)"
                )
                completed = subprocess.run([sys.executable, "-c", script], cwd=cwd,
                                           capture_output=True, text=True, timeout=60)
                self.assertEqual(completed.returncode, 0, completed.stderr)

    def assert_same_state(self, first, second):
        self.assertEqual(set(first.state_dict()), {"entity.weight", "relation.weight"})
        self.assertEqual(set(first.state_dict()), set(second.state_dict()))
        for key, value in first.state_dict().items():
            torch.testing.assert_close(value, second.state_dict()[key], rtol=0, atol=0)

    def test_initialization_scores_gradients_and_adam_updates_match_at_64_and_128(self):
        heads = torch.tensor([0, 1, 2])
        relations = torch.tensor([0, 1, 0])
        tails = torch.tensor([1, 3, 4])
        corrupt_tails = torch.tensor([4, 2, 0])
        for dimension in (64, 128):
            with self.subTest(dimension=dimension):
                torch.manual_seed(42)
                legacy = LegacyTransE(5, 2, dimension)
                torch.manual_seed(42)
                victim = TransE(5, 2, dimension)
                self.assert_same_state(legacy, victim)
                torch.testing.assert_close(legacy.distance(heads, relations, tails),
                                           victim.distance(heads, relations, tails), rtol=0, atol=0)
                torch.testing.assert_close(legacy.score(heads, relations, tails),
                                           victim.score(heads, relations, tails), rtol=0, atol=0)
                old_optimizer = torch.optim.Adam(legacy.parameters(), lr=.01)
                new_optimizer = torch.optim.Adam(victim.parameters(), lr=.01)
                for model, optimizer in ((legacy, old_optimizer), (victim, new_optimizer)):
                    loss = F.relu(1. + model.distance(heads, relations, tails)
                                  - model.distance(heads, relations, corrupt_tails)).mean()
                    optimizer.zero_grad()
                    loss.backward()
                for old_parameter, new_parameter in zip(legacy.parameters(), victim.parameters()):
                    torch.testing.assert_close(old_parameter.grad, new_parameter.grad, rtol=0, atol=0)
                old_optimizer.step()
                new_optimizer.step()
                for model in (legacy, victim):
                    with torch.no_grad():
                        model.entity.weight.copy_(F.normalize(model.entity.weight, p=2, dim=1))
                self.assert_same_state(legacy, victim)

    def test_legacy_and_new_checkpoints_load_in_both_directions(self):
        entities = {name: i for i, name in enumerate(("A", "B", "C", "D", "E"))}
        relations = {"r": 0, "s": 1}
        with tempfile.TemporaryDirectory(prefix="transe_checkpoint_compat_") as temporary:
            root = Path(temporary)
            for dimension in (64, 128):
                with self.subTest(dimension=dimension):
                    torch.manual_seed(13)
                    old_model = LegacyTransE(5, 2, dimension)
                    legacy_path = root / f"legacy_{dimension}.pt"
                    torch.save(checkpoint(old_model, entities, relations, dimension), legacy_path)
                    new_wrapper = TransEVictim.load(legacy_path)
                    self.assertIsInstance(new_wrapper.model, TransE)
                    self.assertFalse(new_wrapper.model.training)
                    self.assertEqual(new_wrapper.entity_to_id, entities)
                    self.assertEqual(new_wrapper.relation_to_id, relations)
                    self.assert_same_state(old_model, new_wrapper.model)
                    old_wrapper = Surrogate.load(legacy_path)
                    self.assertEqual(old_wrapper.triple_score("A", "r", "E"),
                                     new_wrapper.triple_score("A", "r", "E"))

                    new_path = root / f"victim_{dimension}.pt"
                    torch.save(checkpoint(new_wrapper.model, entities, relations, dimension), new_path)
                    reloaded_old_wrapper = Surrogate.load(new_path)
                    self.assert_same_state(new_wrapper.model, reloaded_old_wrapper.model)
                    self.assertEqual(reloaded_old_wrapper.triple_score("B", "s", "D"),
                                     new_wrapper.triple_score("B", "s", "D"))

    def test_filtered_average_ties_preserve_target_plausibility(self):
        model = TransE(4, 1, 2)
        with torch.no_grad():
            model.entity.weight.zero_()
            model.relation.weight.zero_()
        wrapper = TransEVictim(model, {"A": 0, "B": 1, "C": 2, "D": 3}, {"r": 0})
        candidate_ids = torch.arange(4, dtype=torch.long)
        rank, score = evaluation.tail_rank(wrapper, (0, 0, 2), {}, candidate_ids)
        self.assertEqual((rank, score), (2.5, 0.))
        for known_tails in ({1}, {1, 2}):
            with self.subTest(known_tails=known_tails):
                rank, score = evaluation.tail_rank(wrapper, (0, 0, 2), {(0, 0): known_tails}, candidate_ids)
                self.assertEqual((rank, score), (2., 0.))
                self.assertEqual(evaluation.compute_metrics([rank])["MRR"], .5)

    def test_single_target_metrics_use_reciprocal_rank_and_inclusive_top_ten(self):
        for rank, hit, reciprocal in ((10., 1., .1), (11., 0., 1 / 11), (2.5, 1., .4)):
            with self.subTest(rank=rank):
                metrics = evaluation.compute_metrics([rank])
                self.assertEqual(metrics["count"], 1)
                self.assertEqual(metrics["Hits@10"], hit)
                self.assertEqual(metrics["MRR"], reciprocal)

    def test_package_and_standalone_victim_imports(self):
        for cwd, module in ((MASS, "Path_Generation_Module.TransE_Victim"),
                            (MASS / "Path_Generation_Module", "TransE_Victim")):
            with self.subTest(module=module):
                script = (
                    f"from {module} import TransE, TransEVictim; "
                    "import torch; model=TransE(3,1,4); "
                    "assert model.score(torch.tensor([0]),torch.tensor([0]),torch.tensor([1])).shape==(1,); "
                    f"assert TransE.__module__=={module!r}"
                )
                completed = subprocess.run([sys.executable, "-c", script], cwd=cwd,
                                           capture_output=True, text=True, timeout=60)
                self.assertEqual(completed.returncode, 0, completed.stderr)


class TransEVictimPipelineTests(unittest.TestCase):
    def test_two_epoch_paired_training_and_evaluation_with_tiny_disjoint_graph(self):
        self.assertIs(training.TransE, TransE)
        self.assertEqual(training.TransE.__module__, "Path_Generation_Module.TransE_Victim")
        clean = [("A", "r", "B"), ("B", "r", "C"), ("C", "s", "D"), ("D", "s", "E")]
        injected = [("A", "s", "D"), ("D", "r", "E")]
        valid = [("A", "s", "C")]
        test = [("B", "s", "D"), ("C", "r", "E")]
        targets = [("A", "r", "E")]
        with tempfile.TemporaryDirectory(prefix="transe_victim_pipeline_") as temporary:
            root = Path(temporary)
            paths = {}
            for name, triples in (("train", clean), ("poisoned", clean + injected), ("injected", injected),
                                  ("valid", valid), ("test", test), ("targets", targets)):
                paths[name] = root / f"{name}.tsv"
                with paths[name].open("w", encoding="utf-8", newline="") as stream:
                    csv.writer(stream, delimiter="\t", lineterminator="\n").writerows(triples)
            paths["entities"] = root / "entities.txt"
            paths["entities"].write_text("A\nB\nC\nD\nE\n", encoding="utf-8")
            paths["relations"] = root / "relations.txt"
            paths["relations"].write_text("r\ns\n", encoding="utf-8")
            status_path = root / "poisoning_summary.csv"
            with status_path.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(["target_id", "head", "relation", "tail", "status", "injected_for_target"])
                writer.writerow(["target_001", "A", "r", "E", "injected", 2])
            args = argparse.Namespace(
                clean_train=paths["train"], poisoned_train=paths["poisoned"], injected=paths["injected"],
                valid=paths["valid"], test=paths["test"], targets=paths["targets"],
                entities=paths["entities"], relations=paths["relations"], output_dir=root / "models",
                epochs=2, dim=8, batch_size=2, lr=.01, margin=1., seed=42, log_every=1,
                overwrite=False, attack_name="MaSS regression fixture")
            with contextlib.redirect_stdout(io.StringIO()):
                report = training.run(args)
            self.assertTrue(report["same_initial_embeddings_verified"])
            self.assertEqual(report["datasets"]["clean_training_triples"], 4)
            self.assertEqual(report["datasets"]["poisoned_training_triples"], 6)
            self.assertEqual(report["datasets"]["unique_injected_triples"], 2)
            self.assertEqual(report["datasets"]["set_B_target_triples"], 1)
            states = []
            for condition in ("clean", "poisoned"):
                path = args.output_dir / f"{condition}_transe.pt"
                saved = torch.load(path, map_location="cpu", weights_only=True)
                self.assertEqual(set(saved), {"state_dict", "entity_to_id", "relation_to_id", "embedding_dim"})
                self.assertEqual(saved["embedding_dim"], 8)
                self.assertEqual(saved["state_dict"]["entity.weight"].shape, (5, 8))
                states.append(saved["state_dict"])
                current, legacy = TransEVictim.load(path), Surrogate.load(path)
                self.assertEqual(current.triple_score("A", "r", "E"), legacy.triple_score("A", "r", "E"))
            self.assertTrue(any(not torch.equal(states[0][key], states[1][key]) for key in states[0]))
            eval_args = argparse.Namespace(
                clean_checkpoint=args.output_dir / "clean_transe.pt",
                poisoned_checkpoint=args.output_dir / "poisoned_transe.pt",
                train=paths["train"], poisoned_train=paths["poisoned"], injected=paths["injected"],
                valid=paths["valid"], test=paths["test"], targets=paths["targets"],
                poisoning_summary=status_path, training_summary=args.output_dir / "training_summary.json",
                output_dir=root / "evaluation", attack_name=args.attack_name, overwrite=False)
            with contextlib.redirect_stdout(io.StringIO()):
                summary = evaluation.run(eval_args)
            with (eval_args.output_dir / "set_B_target_rank_comparison.csv").open(encoding="utf-8-sig", newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 1)
            self.assertEqual(tuple(rows[0][key] for key in ("head", "relation", "tail")), targets[0])
            metrics = json.loads((eval_args.output_dir / "set_B_attack_metrics.json").read_text(encoding="utf-8"))["set_B_all_targets"]
            self.assertEqual(summary["num_set_B_targets"], 1)
            for condition in ("clean", "poisoned"):
                model = TransEVictim.load(args.output_dir / f"{condition}_transe.pt")
                rank = float(rows[0][f"{condition}_rank"])
                self.assertEqual(float(rows[0][f"{condition}_score"]), model.triple_score("A", "r", "E"))
                self.assertEqual(metrics[condition]["MRR"], 1 / rank)
                self.assertEqual(metrics[condition]["Hits@10"], int(rank <= 10))


if __name__ == "__main__":
    unittest.main()
