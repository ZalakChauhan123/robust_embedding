#!/usr/bin/env python3
"""Fresh TransE target-plausibility experiment using frozen MaSS candidates.

Run from the workspace: .venv/Scripts/python.exe Mass_implement/Test_Triplet/run_experiment.py
All paths in config.json are relative to that configuration's directory. Multiple
targets share one poisoned graph; delta is CLEAN score minus POISONED score.
"""
from __future__ import annotations

import argparse
import contextlib
import copy
import csv
import hashlib
import json
import math
import random
import shutil
import sys
import tempfile
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

HERE = Path(__file__).resolve().parent
MASS = HERE.parent
for directory in (MASS, MASS / "Experiment_Setup", HERE):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))
from Surrogate_Model.TransE_Surrogate import TransE as SurrogateTransE, Surrogate
from Victim_Model.TransE_Victim import TransE as VictimTransE, TransEVictim
from construct_poisoned_dataset import construct as construct_mass

GRAPHML = "http://graphml.graphdrawing.org/xmlns"


def default_config():
    return {
        "dataset": "UMLS", "candidate_mode": "frozen_MaSS_paths_and_intermediate_entities",
        "paths": {
            "train": "Input/train.txt", "valid": "Input/valid.txt", "test": "Input/test.txt",
            "entities": "Input/entities.txt", "relations": "Input/relations.txt",
            "targets": "Input/target_triplet.txt", "mass_candidates": "Input/mass_candidates.json",
            "poisoned_train": "Input/poisoned_train.txt",
            "surrogate_checkpoint": "Surrogate_Model/surrogate_transe.pt",
            "clean_checkpoint": "Victim_Model/clean_transe.pt",
            "poisoned_checkpoint": "Victim_Model/poisoned_transe.pt", "output_dir": "Output",
            "clean_graph": "Graphs/Clean/clean_graph.graphml",
            "poisoned_graph": "Graphs/Poisoned/poisoned_graph.graphml",
        },
        "surrogate": {"dim": 128, "epochs": 500, "batch_size": 256, "lr": .01, "margin": 1., "seed": 43},
        "victim": {"dim": 128, "epochs": 500, "batch_size": 256, "lr": .01, "margin": 1., "seed": 42},
        "attack": {"budget": 2, "delta": .7, "seed": 42, "selection_policy": "baseline"},
        "torch_threads": 1,
    }


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(path.name + ".tmp")
    pending.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    pending.replace(path)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_hashes(hashes):
    for filename, expected in hashes.items():
        if not Path(filename).is_file() or sha(filename) != expected:
            raise ValueError(f"Recorded input or artifact changed: {filename}")


def read_triples(path, allow_empty=False):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        rows = [tuple(row) for row in csv.reader(stream, delimiter="\t") if row]
    if ((not rows and not allow_empty) or any(len(r) != 3 or not all(r) for r in rows)
            or len(rows) != len(set(rows))):
        raise ValueError(f"Triples must be nonempty, unique, TAB-separated rows: {path}")
    return rows


def read_names(path):
    names = [line.strip() for line in Path(path).read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    if not names or len(names) != len(set(names)):
        raise ValueError(f"Empty or duplicate entity/relation names: {path}")
    return names


def load_candidates(path, targets):
    """Match by the actual triple, never by source ID or JSON order."""
    document = read_json(path)
    entries = document.get("targets")
    if entries is None and "target_triple" in document:
        entries = [document]
    if not isinstance(entries, list):
        raise ValueError("mass_candidates.json needs a targets list or one raw MaSS target payload")
    by_target = {}
    for entry in entries:
        payload = entry.get("mass_output", entry)
        target = tuple(payload.get("target_triple", ()))
        if len(target) != 3 or not all(isinstance(x, str) and x for x in target):
            raise ValueError("MaSS candidate entry has no valid target_triple")
        if "target_triple" in entry and tuple(entry["target_triple"]) != target:
            raise ValueError("Candidate entry identity differs from its MaSS payload")
        if target in by_target:
            raise ValueError(f"Duplicate MaSS candidate target: {target}")
        if not isinstance(payload.get("ranked_candidate_paths"), list):
            raise ValueError(f"Missing ranked_candidate_paths for {target}")
        by_target[target] = copy.deepcopy(payload)
    missing = [t for t in targets if t not in by_target]
    if missing:
        raise ValueError(f"No matching saved MaSS candidates for targets: {missing}")
    return {target: by_target[target] for target in targets}


def rescore_candidates(payload, surrogate, clean_set):
    """Recompute every embedding-dependent score, retaining the frozen pool."""
    result = copy.deepcopy(payload)
    h, relation, t = result["target_triple"]
    ranked = []
    for source_order, path in enumerate(result["ranked_candidate_paths"]):
        first, second = tuple(path["first_triple"]), tuple(path["second_triple"])
        middle, r1, r2 = path["intermediate_entity"], path["r1"], path["r2"]
        if first != (h, r1, middle) or second != (middle, r2, t):
            raise ValueError("Malformed/disconnected MaSS candidate body")
        if any(e not in surrogate.entity_to_id for e in (h, middle, t)) or any(
                r not in surrogate.relation_to_id for r in (relation, r1, r2)):
            raise ValueError("MaSS candidate references unknown entity/relation")
        confidence = float(path["pra_confidence"])
        if not math.isfinite(confidence) or not 0 <= confidence <= 1:
            raise ValueError("Invalid saved PRA confidence")
        relation_score = surrogate.relation_score(relation, r1, r2)
        first_score, second_score = surrogate.triple_score(*first), surrogate.triple_score(*second)
        entity_score = first_score + second_score
        bound = relation_score + entity_score
        if not all(math.isfinite(v) for v in (relation_score, first_score, second_score, bound)):
            raise ValueError("Nonfinite fresh surrogate score")
        path.update({
            "source_rank": path.get("rank", source_order + 1), "relation_score": relation_score,
            "first_triple_score": first_score, "second_triple_score": second_score,
            "entity_score": entity_score, "lower_bound_score": bound,
            "first_triple_is_self_loop": first[0] == first[2],
            "second_triple_is_self_loop": second[0] == second[2],
            "has_self_loop": first[0] == first[2] or second[0] == second[2],
            "first_triple_exists_in_train": first in clean_set,
            "second_triple_exists_in_train": second in clean_set,
            "num_new_triples": len({first, second} - clean_set),
        })
        ranked.append(path)
    ranked.sort(key=lambda x: -x["lower_bound_score"])
    for index, path in enumerate(ranked, 1):
        path["rank"] = index
    result["ranked_candidate_paths"] = ranked
    result["num_ranked_paths"] = len(ranked)
    result["rescoring"] = "Fresh surrogate scores; original PRA templates and intermediate-entity candidates frozen"
    return result


def state_hash(model):
    digest = hashlib.sha256()
    for key, tensor in sorted(model.state_dict().items()):
        digest.update(key.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def train_model(model_class, triples, entity_names, relation_names, settings, checkpoint, loss_csv, label):
    """The existing TransE training protocol applied to an explicit local class."""
    seed = settings["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    e2id = {name: i for i, name in enumerate(entity_names)}
    r2id = {name: i for i, name in enumerate(relation_names)}
    model = model_class(len(e2id), len(r2id), settings["dim"])
    initial = state_hash(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=settings["lr"])
    encoded = torch.tensor([(e2id[h], r2id[r], e2id[t]) for h, r, t in triples], dtype=torch.long)
    losses = []
    for epoch in range(1, settings["epochs"] + 1):
        model.train()
        total = 0.
        for indices in torch.randperm(len(encoded)).split(settings["batch_size"]):
            positive = encoded[indices]
            negative = positive.clone()
            corrupt_head = torch.rand(len(negative)) < .5
            replacement = torch.randint(len(e2id), (len(negative),))
            negative[corrupt_head, 0] = replacement[corrupt_head]
            negative[~corrupt_head, 2] = replacement[~corrupt_head]
            positive_distance = model.distance(positive[:, 0], positive[:, 1], positive[:, 2])
            negative_distance = model.distance(negative[:, 0], negative[:, 1], negative[:, 2])
            loss = F.relu(settings["margin"] + positive_distance - negative_distance).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                model.entity.weight.copy_(F.normalize(model.entity.weight, p=2, dim=1))
            total += loss.item() * len(indices)
        value = total / len(encoded)
        if not math.isfinite(value):
            raise ValueError(f"{label} training produced nonfinite loss")
        losses.append(value)
        if epoch == 1 or epoch % 50 == 0 or epoch == settings["epochs"]:
            print(f"{label}: epoch {epoch}/{settings['epochs']} loss={value:.8f}", flush=True)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "entity_to_id": e2id, "relation_to_id": r2id,
                "embedding_dim": settings["dim"]}, checkpoint)
    with loss_csv.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["epoch", "training_loss"])
        writer.writerows(enumerate(losses, 1))
    return {"settings": settings, "training_triples": len(triples),
            "model_class": f"{model_class.__module__}.{model_class.__name__}",
            "initial_state_sha256": initial, "final_state_sha256": state_hash(model),
            "checkpoint_sha256": sha(checkpoint), "first_loss": losses[0], "last_loss": losses[-1]}


def export_graphml(path, triples, entity_names, injected=frozenset()):
    """GraphML uses one distinct directed edge per fact, preserving multi-edges."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ET.register_namespace("", GRAPHML)
    root = ET.Element(f"{{{GRAPHML}}}graphml")
    for key, kind in (("relation", "string"), ("injected", "boolean")):
        ET.SubElement(root, f"{{{GRAPHML}}}key", {"id": key, "for": "edge", "attr.name": key, "attr.type": kind})
    graph = ET.SubElement(root, f"{{{GRAPHML}}}graph", {"id": "KG", "edgedefault": "directed"})
    for name in entity_names:
        ET.SubElement(graph, f"{{{GRAPHML}}}node", {"id": name})
    for index, (h, relation, t) in enumerate(triples, 1):
        edge = ET.SubElement(graph, f"{{{GRAPHML}}}edge", {"id": f"e{index}", "source": h, "target": t})
        ET.SubElement(edge, f"{{{GRAPHML}}}data", {"key": "relation"}).text = relation
        ET.SubElement(edge, f"{{{GRAPHML}}}data", {"key": "injected"}).text = str((h, relation, t) in injected).lower()
    ET.indent(root, space="  ")
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)


def graph_triples(path):
    graph = ET.parse(path).getroot().find(f"{{{GRAPHML}}}graph")
    nodes = {n.attrib["id"] for n in graph.findall(f"{{{GRAPHML}}}node")}
    rows = []
    for edge in graph.findall(f"{{{GRAPHML}}}edge"):
        relation = next(d.text for d in edge.findall(f"{{{GRAPHML}}}data") if d.attrib["key"] == "relation")
        rows.append((edge.attrib["source"], relation, edge.attrib["target"]))
    return nodes, rows


def resolve_paths(config_path, config):
    base = config_path.resolve().parent
    expected = set(default_config()["paths"])
    if set(config["paths"]) != expected:
        raise ValueError(f"Configuration paths must contain exactly: {sorted(expected)}")
    paths = {name: (base / value).resolve() for name, value in config["paths"].items()}
    for path in paths.values():
        if not path.is_relative_to(base):
            raise ValueError(f"Experiment inputs/outputs must stay inside the configuration folder: {path}")
    if len(set(paths.values())) != len(paths):
        raise ValueError("Input and generated output paths must be distinct")
    input_files = [paths[k] for k in ("train", "valid", "test", "entities", "relations", "targets", "mass_candidates")]
    if any(p.is_relative_to(paths["output_dir"]) for p in input_files):
        raise ValueError("Input files cannot reside inside the output directory")
    return paths


def validate_config(config):
    if config.get("candidate_mode") != "frozen_MaSS_paths_and_intermediate_entities":
        raise ValueError("This experiment rescores a frozen MaSS candidate pool; select its documented mode")
    for role in ("surrogate", "victim"):
        settings = config[role]
        if any(not isinstance(settings[k], int) or settings[k] < 1 for k in ("dim", "epochs", "batch_size")):
            raise ValueError("Dimensions, epochs and batch size must be positive integers")
        if not isinstance(settings["seed"], int) or settings["seed"] < 0:
            raise ValueError("Model seeds must be nonnegative integers")
        if any(not math.isfinite(settings[k]) or settings[k] <= 0 for k in ("lr", "margin")):
            raise ValueError("Learning rate and margin must be finite and positive")
    if config["surrogate"]["seed"] == config["victim"]["seed"]:
        raise ValueError("Use different surrogate and victim seeds for independent initialization")
    attack = config["attack"]
    if (not isinstance(attack["budget"], int) or attack["budget"] < 1 or not 0 <= attack["delta"] <= 1
            or not isinstance(attack["seed"], int) or attack["seed"] < 0
            or attack["selection_policy"] not in ("baseline", "low_degree")):
        raise ValueError("Invalid MaSS budget, sampling, seed or selection policy")
    if not isinstance(config["torch_threads"], int) or config["torch_threads"] < 1:
        raise ValueError("torch_threads must be a positive integer")


def run(config_path, resume=False):
    config_path = Path(config_path).resolve()
    config = read_json(config_path)
    validate_config(config)
    paths = resolve_paths(config_path, config)
    for name, source in config.get("input_sources", {}).items():
        if name not in paths or sha(paths[name]) != source["source_sha256"]:
            raise ValueError(f"Input snapshot differs from its recorded original source: {name}")
    torch.set_num_threads(config["torch_threads"])
    train, valid, test, targets = [read_triples(paths[k]) for k in ("train", "valid", "test", "targets")]
    entities, relations = read_names(paths["entities"]), read_names(paths["relations"])
    clean, held_out, target_set = set(train), set(valid) | set(test), set(targets)
    if clean & held_out or target_set & (clean | held_out):
        raise ValueError("Clean/held-out overlap or an already-known target is not permitted")
    for h, r, t in train + valid + test + targets:
        if h not in entities or t not in entities or r not in relations:
            raise ValueError("Input triple references an unknown entity or relation")
    candidates = load_candidates(paths["mass_candidates"], targets)
    code = [Path(__file__).resolve(), HERE / "Surrogate_Model/TransE_Surrogate.py",
            HERE / "Victim_Model/TransE_Victim.py", MASS / "Experiment_Setup/construct_poisoned_dataset.py",
            MASS / "Path_Generation_Module/Random_Sampling.py", MASS / "Path_Generation_Module/Stealthiness.py"]
    input_files = [config_path] + [paths[k] for k in ("train", "valid", "test", "entities", "relations", "targets", "mass_candidates")]
    signature = {"input_sha256": {str(p): sha(p) for p in input_files},
                 "code_sha256": {str(p): sha(p) for p in code},
                 "environment": {"python": sys.version, "torch": str(torch.__version__), "numpy": np.__version__}}
    output = paths["output_dir"]
    manifest_path = output / "experiment_manifest.json"
    if manifest_path.exists():
        manifest = read_json(manifest_path)
        if not resume or manifest["signature"] != signature:
            raise ValueError("Existing run requires --resume with unchanged inputs, configuration and code. Use a separate experiment folder for changed inputs.")
        verify_hashes(manifest["external_source_sha256"])
    else:
        generated = [paths[k] for k in ("poisoned_train", "surrogate_checkpoint", "clean_checkpoint", "poisoned_checkpoint", "clean_graph", "poisoned_graph")]
        if (output.exists() and any(output.iterdir())) or any(p.exists() for p in generated):
            raise ValueError("Untracked generated outputs already exist; choose an empty experiment/output folder")
        external = [MASS / "outputs/Final Results/RQ-1 -- Attack Effectiveness -- Final Results.csv"]
        external += [MASS / "dataset_uml" / name for name in ("train.txt", "valid.txt", "test.txt", "entities.txt", "relations.txt")]
        bundle = read_json(paths["mass_candidates"])
        source_entries = bundle.get("targets", [])
        for entry in source_entries:
            if entry.get("source_file"):
                external.append((config_path.parent / entry["source_file"]).resolve())
        external_hashes = {str(p): sha(p) for p in external if p.is_file()}
        manifest = {"created_utc": datetime.now(timezone.utc).isoformat(), "status": "running",
                    "signature": signature, "config": config, "stages": {},
                    "external_source_sha256": external_hashes,
                    "candidate_mode": "Saved PRA paths/entities frozen; fresh embedding-dependent scores and selection",
                    "multiple_target_mode": "joint poisoned graph; one result row per target",
                    "source_candidate_metadata": [{k: entry[k] for k in ("source_file", "source_target_id", "source_sha256", "target_triple") if k in entry} for entry in source_entries]}
        write_json(manifest_path, manifest)
    logs = output / "training_logs"
    logs.mkdir(parents=True, exist_ok=True)
    rescored = output / "rescored_paths"

    def stage(name, operation, artifacts):
        for hashes in (signature["input_sha256"], signature["code_sha256"], manifest["external_source_sha256"]):
            verify_hashes(hashes)
        if name in manifest["stages"]:
            verify_hashes(manifest["stages"][name]["sha256"])
            print(f"{name}: verified saved artifacts", flush=True)
            return
        print(f"{name}: running", flush=True)
        manifest["status"] = "running"
        manifest["active_stage"] = name
        write_json(manifest_path, manifest)
        started = time.perf_counter()
        log_file = logs / f"{name}.log"
        with log_file.open("w", encoding="utf-8") as stream:
            with contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream):
                operation()
        files = list(artifacts()) + [log_file]
        if not files or any(not p.is_file() for p in files):
            raise ValueError(f"Stage missing required artifacts: {name}")
        manifest["stages"][name] = {"elapsed_seconds": time.perf_counter() - started,
                                    "sha256": {str(p): sha(p) for p in files}}
        write_json(manifest_path, manifest)
        print(f"{name}: completed ({manifest['stages'][name]['elapsed_seconds']:.1f}s)", flush=True)

    def surrogate_training():
        report = train_model(SurrogateTransE, train, entities, relations, config["surrogate"],
                             paths["surrogate_checkpoint"], logs / "surrogate_loss.csv", "surrogate")
        write_json(logs / "surrogate_training_summary.json", report)
    stage("surrogate", surrogate_training, lambda: [paths["surrogate_checkpoint"], logs / "surrogate_loss.csv", logs / "surrogate_training_summary.json"])

    def rescore():
        surrogate = Surrogate.load(paths["surrogate_checkpoint"])
        rows = []
        for index, target in enumerate(targets, 1):
            payload = rescore_candidates(candidates[target], surrogate, clean)
            payload["fresh_surrogate_sha256"] = sha(paths["surrogate_checkpoint"])
            target_id = f"target_{index:03d}"
            write_json(rescored / f"{target_id}.json", payload)
            rows.append({"target_id": target_id, "head": target[0], "relation": target[1], "tail": target[2],
                         "num_ranked_paths": len(payload["ranked_candidate_paths"]),
                         "status": "feasible" if payload["ranked_candidate_paths"] else "infeasible"})
        write_json(rescored / "feasibility_summary.json", {"set_B_targets_total": len(targets), "error_targets": 0, "targets": rows,
                   "source_mode": "Frozen MaSS candidates rescored; no PRA retraining or intermediate-entity search",
                   "surrogate_checkpoint": str(paths["surrogate_checkpoint"]), "surrogate_sha256": sha(paths["surrogate_checkpoint"])})
    stage("rescore", rescore, lambda: list(rescored.glob("*.json")))

    def poisoning():
        with tempfile.TemporaryDirectory(prefix="mass_triplet_poisoning_") as temporary:
            directory = Path(temporary)
            construct_mass(targets_file=paths["targets"], feasibility_dir=rescored, output_dir=directory,
                           train_file=paths["train"], valid_file=paths["valid"], test_file=paths["test"], **config["attack"])
            paths["poisoned_train"].parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(directory / "poisoned_train.txt", paths["poisoned_train"])
            for name in ("selected_poisoning_paths.json", "eligible_candidate_paths.json", "injected_triples.tsv", "poisoning_summary.json", "poisoning_summary.csv"):
                shutil.copyfile(directory / name, output / name)
    poison_files = [output / name for name in ("selected_poisoning_paths.json", "eligible_candidate_paths.json", "injected_triples.tsv", "poisoning_summary.json", "poisoning_summary.csv")]
    stage("poisoning", poisoning, lambda: [paths["poisoned_train"], *poison_files])
    poisoned = read_triples(paths["poisoned_train"])
    injected = set(read_triples(output / "injected_triples.tsv", allow_empty=True))
    if set(poisoned) != clean | injected or injected & (clean | held_out | target_set):
        raise ValueError("Poisoned graph is not the legal clean-plus-injected union")

    def graphs():
        export_graphml(paths["clean_graph"], train, entities)
        export_graphml(paths["poisoned_graph"], poisoned, entities, injected)
    stage("graphs", graphs, lambda: [paths["clean_graph"], paths["poisoned_graph"]])

    def victim_training():
        reports = {}
        for label, triples, checkpoint in (("clean", train, paths["clean_checkpoint"]), ("poisoned", poisoned, paths["poisoned_checkpoint"])):
            reports[label] = train_model(VictimTransE, triples, entities, relations, config["victim"], checkpoint, logs / f"{label}_victim_loss.csv", f"{label} victim")
        if reports["clean"]["initial_state_sha256"] != reports["poisoned"]["initial_state_sha256"]:
            raise ValueError("Victim initial embeddings differ")
        surrogate_report = read_json(logs / "surrogate_training_summary.json")
        if surrogate_report["initial_state_sha256"] == reports["clean"]["initial_state_sha256"]:
            raise ValueError("Surrogate and victim have identical initialization")
        write_json(output / "victim_training_summary.json", {"victims": reports,
                   "same_victim_initial_embeddings_verified": True, "surrogate_independent_initialization_verified": True,
                   "training_protocol": "Adam, margin-ranking negative L2, one uniform head/tail negative per positive, unit entity normalization after every step"})
    stage("victims", victim_training, lambda: [paths["clean_checkpoint"], paths["poisoned_checkpoint"], logs / "clean_victim_loss.csv", logs / "poisoned_victim_loss.csv", output / "victim_training_summary.json"])

    def results():
        before, after = TransEVictim.load(paths["clean_checkpoint"]), TransEVictim.load(paths["poisoned_checkpoint"])
        status = read_json(output / "poisoning_summary.json")["targets"]
        rows = []
        for index, (target, record) in enumerate(zip(targets, status), 1):
            if tuple(record[k] for k in ("head", "relation", "tail")) != target:
                raise ValueError("Poisoning summary target order mismatch")
            s_clean, s_poisoned = before.triple_score(*target), after.triple_score(*target)
            rows.append({"target_id": f"target_{index:03d}", "head": target[0], "relation": target[1], "tail": target[2],
                         "clean_plausibility": s_clean, "poisoned_plausibility": s_poisoned, "delta": s_clean - s_poisoned,
                         "poisoning_status": record["status"], "injected_for_target": record["injected_for_target"],
                         "new_to_global_training": record["new_to_global_training"]})
        with (output / "Final result.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        summary = {"results": rows, "delta_definition": "clean_plausibility - poisoned_plausibility",
                   "negative_delta": "Target more plausible to poisoned victim",
                   "num_targets": len(targets), "num_clean_facts": len(train), "num_poisoned_facts": len(poisoned),
                   "num_unique_injected_facts": len(injected), "num_entities": len(entities), "num_relations": len(relations),
                   "multiple_target_mode": "Joint poisoned graph; cross-target effects are included",
                   "candidate_mode": "Existing MaSS templates/intermediate candidates frozen; freshly rescored and reselected",
                   "configuration": config}
        write_json(output / "Final result.json", summary)
    stage("results", results, lambda: [output / "Final result.csv", output / "Final result.json"])

    summary = read_json(output / "Final result.json")
    # Independent arithmetic on checkpoint tensors catches wrapper/scoring mistakes.
    for label, checkpoint in (("clean", paths["clean_checkpoint"]), ("poisoned", paths["poisoned_checkpoint"])):
        data = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if data["embedding_dim"] != config["victim"]["dim"]:
            raise ValueError("Victim embedding dimension differs from config")
        e, r = data["state_dict"]["entity.weight"], data["state_dict"]["relation.weight"]
        for target, row in zip(targets, summary["results"]):
            h, rel, t = target
            actual = float(-torch.linalg.vector_norm(e[data["entity_to_id"][h]] + r[data["relation_to_id"][rel]] - e[data["entity_to_id"][t]], ord=2))
            if actual != row[f"{label}_plausibility"]:
                raise ValueError("Plausibility differs from checkpoint tensor calculation")
    if len(summary["results"]) != len(targets) or any(r["delta"] != r["clean_plausibility"] - r["poisoned_plausibility"] for r in summary["results"]):
        raise ValueError("Result denominator or delta definition is incorrect")
    selected = read_json(output / "selected_poisoning_paths.json")["targets"]
    for entry in selected:
        attributed = {tuple(t) for t in entry["attributed_injected_triples"]}
        if len(attributed) > config["attack"]["budget"] or not attributed <= injected:
            raise ValueError("Poison budget or attribution mismatch")
        for path in entry["stealthiness"]["selected_candidate_paths"]:
            if not {tuple(path[key]) for key in ("first_triple", "second_triple")} <= set(poisoned):
                raise ValueError("An incomplete selected path entered the graph")
    for graph, facts in ((paths["clean_graph"], train), (paths["poisoned_graph"], poisoned)):
        nodes, rows = graph_triples(graph)
        if nodes != set(entities) or set(rows) != set(facts) or len(rows) != len(facts):
            raise ValueError("GraphML lost nodes, parallel relations or facts")
    for step in manifest["stages"].values():
        verify_hashes(step["sha256"])
    for hashes in (signature["input_sha256"], signature["code_sha256"], manifest["external_source_sha256"]):
        verify_hashes(hashes)
    summary["checks"] = {"paired_victim_initialization": True, "independent_surrogate": True,
                         "legal_complete_paths_and_budget": True, "all_targets_retained": True,
                         "checkpoint_scores_and_delta_verified": True, "GraphML_round_trip": True,
                         "original_external_sources_unchanged": True}
    manifest.update({"status": "complete", "completed_utc": datetime.now(timezone.utc).isoformat(), "active_stage": None,
                     "summary": summary})
    write_json(manifest_path, manifest)
    print(f"Completed {len(targets)} target(s). Results: {output / 'Final result.csv'}", flush=True)
    for row in summary["results"]:
        print(f"{row['target_id']}: clean={row['clean_plausibility']:.9f}, poisoned={row['poisoned_plausibility']:.9f}, delta={row['delta']:+.9f}", flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=HERE / "config.json")
    parser.add_argument("--resume", action="store_true", help="Verify and reuse completed stages from unchanged inputs/configuration/code")
    args = parser.parse_args()
    run(args.config, resume=args.resume)


if __name__ == "__main__":
    main()
