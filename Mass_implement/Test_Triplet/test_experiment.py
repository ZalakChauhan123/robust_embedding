"""Regression checks for isolated MaSS candidate reuse and TransE plausibility."""
from __future__ import annotations

import copy
import csv
import hashlib
import json
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import run_experiment as experiment
from Surrogate_Model.TransE_Surrogate import Surrogate, TransE as AttackerTransE
from Victim_Model.TransE_Victim import TransEVictim


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def write_triples(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        csv.writer(stream, delimiter="\t", lineterminator="\n").writerows(rows)


def read_triples(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return [tuple(row) for row in csv.reader(stream, delimiter="\t") if row]


def candidate(target, middle="D", first_relation="p", second_relation="q", stale=12345.):
    return {
        "r1": first_relation, "r2": second_relation, "intermediate_entity": middle,
        "first_triple": [target[0], first_relation, middle],
        "second_triple": [middle, second_relation, target[2]],
        "pra_confidence": .99, "rank": 77,
        "relation_score": stale, "first_triple_score": stale,
        "second_triple_score": stale, "entity_score": stale,
        "lower_bound_score": stale, "num_new_triples": 999,
        "first_triple_exists_in_train": False, "second_triple_exists_in_train": False,
        "first_triple_is_self_loop": False, "second_triple_is_self_loop": False,
        "has_self_loop": False,
    }


def payload(target, paths):
    return {
        "dataset": "tiny regression graph", "pra_method": "supervised",
        "target_triple": list(target), "pra_threshold": .9,
        "top_k_relation_templates": 10, "top_entities_per_template": 10,
        "num_pra_templates": 1, "num_semantically_valid_templates": 1,
        "num_selected_templates": 1, "num_ranked_paths": len(paths),
        "ranked_candidate_paths": paths,
    }


def graph_facts(path):
    tree = ET.parse(path)
    namespace = {"g": "http://graphml.graphdrawing.org/xmlns"}
    keys = {key.attrib["id"]: key.attrib.get("attr.name", "")
            for key in tree.findall("g:key", namespace)}
    nodes = set()
    labels = {}
    for node in tree.findall(".//g:node", namespace):
        values = {keys.get(item.attrib["key"], item.attrib["key"]): item.text
                  for item in node.findall("g:data", namespace)}
        label = values.get("entity", values.get("label", node.attrib["id"]))
        labels[node.attrib["id"]] = label
        nodes.add(label)
    facts = []
    for edge in tree.findall(".//g:edge", namespace):
        values = {keys.get(item.attrib["key"], item.attrib["key"]): item.text
                  for item in edge.findall("g:data", namespace)}
        relation = values.get("relation", values.get("label"))
        if relation is None:
            raise AssertionError("GraphML edge needs an explicit relation")
        facts.append((labels[edge.attrib["source"]], relation, labels[edge.attrib["target"]]))
    return nodes, facts


def fixture(root, *, sampling_delta=0., budget=2, payloads=None):
    clean = [("A", "r", "B"), ("A", "s", "B"), ("B", "r", "C"),
             ("C", "s", "D"), ("D", "s", "E")]
    valid = [("A", "s", "C")]
    test = [("B", "s", "D")]
    targets = [("A", "r", "E"), ("A", "r", "F")]
    entities = ["A", "B", "C", "D", "E", "F", "Z"]
    relations = ["r", "p", "q", "s"]
    inputs = root / "Input"
    inputs.mkdir(parents=True)
    for name, triples in (("train", clean), ("valid", valid), ("test", test)):
        write_triples(inputs / f"{name}.txt", triples)
    # A custom filename and reversed candidate order check identity-based matching.
    write_triples(inputs / "my_targets.tsv", targets)
    (inputs / "entities.txt").write_text("\n".join(entities) + "\n", encoding="utf-8")
    (inputs / "relations.txt").write_text("\n".join(relations) + "\n", encoding="utf-8")
    items = []
    for i, target in enumerate(targets, 1):
        data = payloads[i - 1] if payloads else payload(target, [candidate(target)])
        source = inputs / f"original_mass_{i}.json"
        write_json(source, data)
        items.append({"source_target_id": f"original_{i:03d}", "target_triple": list(target),
                      "source_file": str(source),
                      "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                      "mass_output": data})
    write_json(inputs / "mass_candidates.json", {"targets": list(reversed(items))})
    config = experiment.default_config()
    paths = config.get("paths", config)
    paths.update({
        "train": "Input/train.txt", "valid": "Input/valid.txt", "test": "Input/test.txt",
        "entities": "Input/entities.txt", "relations": "Input/relations.txt",
        "targets": "Input/my_targets.tsv", "mass_candidates": "Input/mass_candidates.json",
        "poisoned_train": "Input/poisoned_train.txt",
        "surrogate_checkpoint": "Surrogate_Model/surrogate_transe.pt",
        "clean_checkpoint": "Victim_Model/clean_transe.pt",
        "poisoned_checkpoint": "Victim_Model/poisoned_transe.pt",
        "output_dir": "Output", "clean_graph": "Graphs/Clean/clean_graph.graphml",
        "poisoned_graph": "Graphs/Poisoned/poisoned_graph.graphml",
    })
    for role, seed in (("surrogate", 43), ("victim", 42)):
        config[role].update({"dim": 8, "epochs": 2, "batch_size": 2,
                             "lr": .01, "margin": 1., "seed": seed})
    config["attack"].update({"budget": budget, "delta": sampling_delta,
                             "seed": 42, "selection_policy": "baseline"})
    config["torch_threads"] = 1
    config_path = root / "config.json"
    write_json(config_path, config)
    return config_path, config, clean, valid, test, targets, entities


class CandidateTests(unittest.TestCase):
    def test_candidate_bundle_matches_actual_triples_instead_of_input_order(self):
        first, second = ("A", "r", "E"), ("A", "r", "F")
        with tempfile.TemporaryDirectory(prefix="triplet_candidates_") as temporary:
            path = Path(temporary) / "candidates.json"
            write_json(path, {"targets": [
                {"target_triple": list(second), "mass_output": payload(second, [candidate(second)])},
                {"target_triple": list(first), "mass_output": payload(first, [candidate(first)])},
            ]})
            loaded = experiment.load_candidates(path, [first, second])
            for target in (first, second):
                self.assertEqual(loaded[target]["target_triple"], list(target))
            with self.assertRaises(ValueError):
                experiment.load_candidates(path, [("A", "r", "Z")])
            write_json(path, payload(first, [candidate(first)]))
            self.assertEqual(experiment.load_candidates(path, [first])[first]["target_triple"], list(first))

    def test_rescoring_replaces_stale_scores_ranking_and_training_flags_without_mutation(self):
        target = ("A", "r", "E")
        model = AttackerTransE(4, 3, 2)
        with torch.no_grad():
            model.entity.weight.copy_(torch.tensor([[0., 0.], [3., 0.], [.5, 0.], [1., 0.]]))
            model.relation.weight.copy_(torch.tensor([[1., 0.], [.5, 0.], [.5, 0.]]))
        surrogate = Surrogate(model, {"A": 0, "B": 1, "D": 2, "E": 3}, {"r": 0, "p": 1, "q": 2})
        bad, good = candidate(target, "B", stale=999.), candidate(target, "D", stale=-999.)
        original = payload(target, [bad, good])
        before = copy.deepcopy(original)
        rescored = experiment.rescore_candidates(original, surrogate, {("A", "p", "B")})
        self.assertEqual(original, before)
        paths = rescored["ranked_candidate_paths"]
        self.assertEqual(len(paths), 2)
        self.assertEqual([p["intermediate_entity"] for p in paths], ["D", "B"])
        self.assertEqual(paths[0]["lower_bound_score"], 0.)
        self.assertEqual(paths[1]["lower_bound_score"], -5.)
        self.assertTrue(paths[1]["first_triple_exists_in_train"])
        self.assertFalse(paths[1]["second_triple_exists_in_train"])
        self.assertEqual(paths[1]["num_new_triples"], 1)
        self.assertEqual(paths[0]["num_new_triples"], 2)
        for rank, item in enumerate(paths, 1):
            self.assertEqual(item["rank"], rank)
            self.assertEqual(item["pra_confidence"], .99)
            self.assertEqual(item["entity_score"], item["first_triple_score"] + item["second_triple_score"])
            self.assertEqual(item["lower_bound_score"], item["relation_score"] + item["entity_score"])

    def test_graphml_preserves_parallel_relations_direction_and_isolated_entities(self):
        triples = [("A", "r", "B"), ("A", "s", "B"), ("B", "q", "A")]
        with tempfile.TemporaryDirectory(prefix="triplet_graphml_") as temporary:
            path = Path(temporary) / "graph.graphml"
            experiment.export_graphml(path, triples, ["A", "B", "Z"], injected={triples[1]})
            nodes, saved = graph_facts(path)
            self.assertEqual(nodes, {"A", "B", "Z"})
            self.assertEqual(len(saved), 3)
            self.assertEqual(set(saved), set(triples))


class PipelineTests(unittest.TestCase):
    def test_two_targets_share_one_joint_graph_delta_sign_is_clean_minus_poisoned(self):
        with tempfile.TemporaryDirectory(prefix="triplet_joint_pipeline_") as temporary:
            root = Path(temporary)
            config_path, config, clean, valid, test, targets, entities = fixture(root)
            preserved = {path: path.read_bytes() for path in (root / "Input").iterdir() if path.is_file()}
            summary = experiment.run(config_path)
            result = summary["results"]
            self.assertEqual([(r["head"], r["relation"], r["tail"]) for r in result], targets)
            expected_added = {("A", "p", "D"), ("D", "q", "E"), ("D", "q", "F")}
            poisoned = read_triples(root / "Input/poisoned_train.txt")
            self.assertEqual(len(poisoned), len(clean) + 3)
            self.assertEqual(set(poisoned), set(clean) | expected_added)
            self.assertFalse(set(poisoned) & (set(valid) | set(test) | set(targets)))
            self.assertEqual(set(read_triples(root / "Output/injected_triples.tsv")), expected_added)
            self.assertEqual([r["injected_for_target"] for r in result], [2, 2])
            self.assertEqual(sum(r["new_to_global_training"] for r in result), 3)
            for path, expected in preserved.items():
                self.assertEqual(path.read_bytes(), expected)
            clean_model = TransEVictim.load(root / "Victim_Model/clean_transe.pt")
            poisoned_model = TransEVictim.load(root / "Victim_Model/poisoned_transe.pt")
            surrogate = Surrogate.load(root / "Surrogate_Model/surrogate_transe.pt")
            self.assertTrue(any(not torch.equal(value, surrogate.model.state_dict()[key])
                                for key, value in clean_model.model.state_dict().items()))
            with (root / "Output/Final result.csv").open(encoding="utf-8-sig", newline="") as stream:
                csv_rows = list(csv.DictReader(stream))
            self.assertEqual(len(csv_rows), 2)
            for row, csv_row, target in zip(result, csv_rows, targets):
                clean_score = clean_model.triple_score(*target)
                poison_score = poisoned_model.triple_score(*target)
                self.assertEqual(row["clean_plausibility"], clean_score)
                self.assertEqual(row["poisoned_plausibility"], poison_score)
                self.assertEqual(row["delta"], clean_score - poison_score)
                self.assertAlmostEqual(float(csv_row["delta"]), clean_score - poison_score, places=12)
            for kind, expected in (("Clean", clean), ("Poisoned", poisoned)):
                nodes, saved = graph_facts(root / f"Graphs/{kind}/{kind.lower()}_graph.graphml")
                self.assertEqual(nodes, set(entities))
                self.assertEqual(set(saved), set(expected))
                self.assertEqual(len(saved), len(expected))
            self.assertTrue((root / "Output/experiment_manifest.json").is_file())
            checkpoint = root / "Victim_Model/poisoned_transe.pt"
            before = checkpoint.read_bytes()
            resumed = experiment.run(config_path, resume=True)
            self.assertEqual(resumed["results"], result)
            self.assertEqual(checkpoint.read_bytes(), before)
            config["victim"]["lr"] = .02
            write_json(config_path, config)
            with self.assertRaises((ValueError, RuntimeError)):
                experiment.run(config_path, resume=True)

    def test_no_sampled_paths_retains_targets_and_trains_identical_clean_poisoned_victims(self):
        with tempfile.TemporaryDirectory(prefix="triplet_no_injections_") as temporary:
            root = Path(temporary)
            config, _, clean, _, _, targets, _ = fixture(root, sampling_delta=1.)
            summary = experiment.run(config)
            self.assertEqual(len(summary["results"]), len(targets))
            self.assertEqual(read_triples(root / "Input/poisoned_train.txt"), clean)
            self.assertEqual(read_triples(root / "Output/injected_triples.tsv"), [])
            clean_ckpt = torch.load(root / "Victim_Model/clean_transe.pt", map_location="cpu", weights_only=True)
            poison_ckpt = torch.load(root / "Victim_Model/poisoned_transe.pt", map_location="cpu", weights_only=True)
            for key, value in clean_ckpt["state_dict"].items():
                self.assertTrue(torch.equal(value, poison_ckpt["state_dict"][key]))
            for row in summary["results"]:
                self.assertEqual(row["injected_for_target"], 0)
                self.assertEqual(row["delta"], 0.)

    def test_budget_cannot_inject_half_a_two_new_fact_path(self):
        with tempfile.TemporaryDirectory(prefix="triplet_atomic_budget_") as temporary:
            root = Path(temporary)
            config, _, clean, _, _, _, _ = fixture(root, budget=1)
            summary = experiment.run(config)
            self.assertEqual(read_triples(root / "Input/poisoned_train.txt"), clean)
            self.assertTrue(all(row["injected_for_target"] == 0 for row in summary["results"]))

    def test_heldout_edge_rejects_its_whole_path_without_dropping_target(self):
        targets = [("A", "r", "E"), ("A", "r", "F")]
        blocked = payload(targets[0], [candidate(targets[0], "C", first_relation="s")])
        allowed = payload(targets[1], [candidate(targets[1])])
        with tempfile.TemporaryDirectory(prefix="triplet_heldout_filter_") as temporary:
            root = Path(temporary)
            config, _, clean, valid, test, _, _ = fixture(root, payloads=[blocked, allowed])
            summary = experiment.run(config)
            rows = summary["results"]
            self.assertEqual(len(rows), 2)
            self.assertEqual([row["injected_for_target"] for row in rows], [0, 2])
            added = set(read_triples(root / "Input/poisoned_train.txt")) - set(clean)
            self.assertEqual(added, {("A", "p", "D"), ("D", "q", "F")})
            self.assertFalse(added & (set(valid) | set(test)))
            self.assertNotIn(("C", "q", "E"), added)


if __name__ == "__main__":
    unittest.main()
