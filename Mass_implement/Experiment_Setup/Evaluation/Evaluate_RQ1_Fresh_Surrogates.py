#!/usr/bin/env python3
"""Train fresh TransE attackers at 64/128 dimensions; keep TransE victims at 128.

Regenerate all three local attack implementations on the fixed 100 UMLS targets.
This experiment never writes to the final RQ1 table or existing experiment data.
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import json
import math
import sys
import time
from pathlib import Path

MASS = Path(__file__).resolve().parents[2]
SETUP = MASS / "Experiment_Setup"
for directory in (MASS, SETUP):
    sys.path.insert(0, str(directory))
import torch
from Path_Generation_Module.TransE_Surrogate import train as train_surrogate
from target_feasibility_check import run_feasibility
from construct_poisoned_dataset import construct as construct_mass
from construct_attack_baselines import construct as construct_baseline
from train_victim_models import run as train_victims
from Evaluate_victim_models import run as evaluate_victims

EXP = MASS / "outputs" / "Experiments" / "UMLS_feasible_rank11_15_seed42"
DATA = MASS / "dataset_uml"
ROOT = EXP / "RQ1" / "fresh_surrogates"
TARGETS = EXP / "setup" / "set_B_target_triples.tsv"
FINAL = MASS / "outputs" / "Final Results" / "RQ-1 -- Attack Effectiveness -- Final Results.csv"
ATTACKS = (("MaSS", "baseline"), ("DirectAdd", "DirectAdd"), ("ComAttack", "ComAttack"))


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def triples(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return [tuple(row) for row in csv.reader(stream, delimiter="\t") if row]


def verify_hashes(hashes):
    for filename, expected in hashes.items():
        if sha(Path(filename)) != expected:
            raise ValueError(f"Recorded source/artifact changed: {filename}")


def snapshot():
    paths = [FINAL, TARGETS, EXP / "setup" / "target_setup_summary.json", Path(__file__).resolve()]
    paths += [DATA / name for name in ("train.txt", "valid.txt", "test.txt", "entities.txt", "relations.txt", "Semantic_Constraints/SRSTRE2")]
    paths += [MASS / relative for relative in (
        "outputs/PRA/supervised_pra_umls.joblib", "outputs/Path_Generation/transe_umls.pt",
        "Path_Generation_Module/TransE_Surrogate.py", "Path_Generation_Module/TransE_Victim.py",
        "Path_Generation_Module/Entity_Selection.py",
        "Path_Generation_Module/Random_Sampling.py", "Path_Generation_Module/Stealthiness.py",
        "PRA_Module/PRA_supervised.py", "PRA_Module/Pra_implement_finalVersion.py",
        "Semantic_Constraints_Module/Semantic_Constraints.py",
        "Experiment_Setup/target_feasibility_check.py", "Experiment_Setup/construct_poisoned_dataset.py",
        "Experiment_Setup/construct_attack_baselines.py", "Experiment_Setup/train_victim_models.py",
        "Experiment_Setup/Evaluation/Evaluate_victim_models.py")]
    for _, folder in ATTACKS:
        paths += [EXP / folder / "poisoning" / name for name in ("poisoned_train.txt", "injected_triples.tsv")]
        paths += [EXP / "RQ1" / "transe_dim128" / folder / "seed_42" / "victim_models" / name
                  for name in ("clean_transe.pt", "poisoned_transe.pt", "training_summary.json")]
    paths.append(EXP / "RQ1" / "transe_dim128" / "results_summary.json")
    return {str(path): sha(path) for path in paths}


def stage(point, name, operation, artifact_directory, config):
    verify_hashes(config["input_sha256"])
    point.mkdir(parents=True, exist_ok=True)
    manifest = point / "stage_manifest.json"
    state = read_json(manifest) if manifest.exists() else {"stages": {}}
    if name in state["stages"]:
        verify_hashes(state["stages"][name]["sha256"])
        print(f"{point.relative_to(ROOT)} / {name}: verified saved stage", flush=True)
        return
    print(f"{point.relative_to(ROOT)} / {name}: running", flush=True)
    started = time.perf_counter()
    with (point / f"{name}.log").open("w", encoding="utf-8") as stream:
        with contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream):
            operation()
    files = [path for path in artifact_directory.rglob("*") if path.is_file()]
    if not files:
        raise ValueError(f"Stage produced no artifacts: {name}")
    state["stages"][name] = {"seconds": time.perf_counter() - started,
                              "sha256": {str(path): sha(path) for path in files}}
    write_json(manifest, state)
    print(f"{point.relative_to(ROOT)} / {name}: completed ({state['stages'][name]['seconds']:.1f}s)", flush=True)


def feasibility(checkpoint, output):
    result = run_feasibility(targets_file=TARGETS, output_dir=output, checkpoint=checkpoint,
                             pra_method="supervised", threshold=.9, top_k=10, top_entities=10,
                             metadata_file=EXP / "setup" / "target_setup_summary.json")
    if result["error_targets"]:
        raise RuntimeError("Path generation errors are not legitimate infeasible targets")


def victim_args(point, poison, attack):
    train = argparse.Namespace(
        clean_train=DATA / "train.txt", poisoned_train=poison / "poisoned_train.txt",
        injected=poison / "injected_triples.tsv", valid=DATA / "valid.txt", test=DATA / "test.txt",
        targets=TARGETS, entities=DATA / "entities.txt", relations=DATA / "relations.txt",
        output_dir=point / "victim_models", attack_name=attack,
        epochs=500, dim=128, batch_size=256, seed=42, lr=.01, margin=1., log_every=50, overwrite=True)
    evaluate = argparse.Namespace(
        clean_checkpoint=train.output_dir / "clean_transe.pt",
        poisoned_checkpoint=train.output_dir / "poisoned_transe.pt",
        train=train.clean_train, poisoned_train=train.poisoned_train, injected=train.injected,
        valid=train.valid, test=train.test, targets=TARGETS,
        poisoning_summary=poison / "poisoning_summary.csv",
        training_summary=train.output_dir / "training_summary.json",
        output_dir=point / "evaluation", attack_name=attack, overwrite=True)
    return train, evaluate


def validate(point, dimension, attack, folder, reference_state):
    poison = point / "poisoning"
    summary = read_json(poison / "poisoning_summary.json")
    train = read_json(point / "victim_models" / "training_summary.json")
    if not train["same_initial_embeddings_verified"] or train["parameters"]["embedding_dim"] != 128:
        raise ValueError("Victim pairing/dimension control failed")
    injected = set(triples(poison / "injected_triples.tsv"))
    clean = set(triples(DATA / "train.txt"))
    original = clean | set(triples(DATA / "valid.txt")) | set(triples(DATA / "test.txt"))
    targets = triples(TARGETS)
    if injected & (original | set(targets)) or set(triples(poison / "poisoned_train.txt")) != clean | injected:
        raise ValueError("Poison data integrity failed")
    if any(row["injected_for_target"] > 2 for row in summary["targets"]):
        raise ValueError("Attack budget exceeded")
    if attack != "MaSS" and summary["source_sha256"]["surrogate_checkpoint"] != sha(point.parent / "surrogate" / "transe_umls.pt"):
        raise ValueError("Wrong attacker checkpoint for baseline")
    surrogate = torch.load(point.parent / "surrogate" / "transe_umls.pt", map_location="cpu", weights_only=True)
    if surrogate["embedding_dim"] != dimension:
        raise ValueError("Wrong surrogate dimension")
    with (point / "evaluation" / "set_B_target_rank_comparison.csv").open(encoding="utf-8-sig", newline="") as stream:
        ranks = list(csv.DictReader(stream))
    if len(ranks) != 100 or [tuple(row[k] for k in ("head", "relation", "tail")) for row in ranks] != targets:
        raise ValueError("Targets or evaluation denominator changed")
    result = read_json(point / "evaluation" / "set_B_attack_metrics.json")["set_B_all_targets"]
    checkpoints = {}
    for condition in ("clean", "poisoned"):
        checkpoint = torch.load(point / "victim_models" / f"{condition}_transe.pt", map_location="cpu", weights_only=True)
        if checkpoint["embedding_dim"] != 128 or checkpoint["state_dict"]["entity.weight"].shape != (135, 128):
            raise ValueError("Wrong victim checkpoint shape")
        checkpoints[condition] = checkpoint
        e, r = checkpoint["state_dict"]["entity.weight"], checkpoint["state_dict"]["relation.weight"]
        e2id, r2id = checkpoint["entity_to_id"], checkpoint["relation_to_id"]
        filtered = {}
        for h, rel, t in original:
            filtered.setdefault((h, rel), set()).add(e2id[t])
        measured = []
        for target, saved in zip(targets, ranks):
            h, rel, t = target
            with torch.no_grad():
                scores = -torch.linalg.vector_norm(e[e2id[h]] + r[r2id[rel]] - e, ord=2, dim=1)
                value = scores[e2id[t]].item()
                excluded = filtered.get((h, rel), set()) - {e2id[t]}
                if excluded:
                    scores[sorted(excluded)] = -torch.inf
                rank = 1 + int((scores > value).sum()) + (int((scores == value).sum()) - 1) / 2
            if rank != float(saved[f"{condition}_rank"]):
                raise ValueError("Independent checkpoint rank differs from evaluator output")
            measured.append(rank)
        hit = sum(rank <= 10 for rank in measured) / 100
        mrr = sum(1 / rank for rank in measured) / 100
        if not math.isclose(hit, result[condition]["Hits@10"], abs_tol=1e-12) or not math.isclose(mrr, result[condition]["MRR"], abs_tol=1e-12):
            raise ValueError("Incorrect summary metrics")
    state = checkpoints["clean"]["state_dict"]
    if any(not torch.equal(state[k], reference_state[k]) for k in state):
        raise ValueError("Clean victim changed across attack conditions or from original128 comparison")
    old = read_json(EXP / "RQ1" / "transe_dim128" / folder / "seed_42" / "evaluation" / "set_B_attack_metrics.json")["set_B_all_targets"]
    equal_surrogate = dimension == 128 and all(torch.equal(surrogate["state_dict"][k], state[k]) for k in state)
    return {"surrogate_dim": dimension, "attack": attack, "victim_dim": 128, "seed": 42,
            "Hit_at_10": result["poisoned"]["Hits@10"], "MRR": result["poisoned"]["MRR"],
            "clean_Hit_at_10": result["clean"]["Hits@10"], "clean_MRR": result["clean"]["MRR"],
            "delta_Hit_at_10": result["delta_poisoned_minus_clean"]["Hits@10"],
            "delta_MRR": result["delta_poisoned_minus_clean"]["MRR"],
            "previous_frozen_surrogate_Hit_at_10": old["poisoned"]["Hits@10"],
            "previous_frozen_surrogate_MRR": old["poisoned"]["MRR"],
            "num_targets": 100, "injected_targets": summary["num_targets_with_injections"],
            "unique_injected_triplets": len(injected),
            "surrogate_equals_clean_victim": equal_surrogate,
            "evaluation_directory": str(point / "evaluation")}


def export(rows, config):
    with (ROOT / "fresh_surrogate_results.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_json(ROOT / "fresh_surrogate_results.json", {"configuration": str(ROOT / "sweep_config.json"), "results": rows,
        "final_RQ1_CSV_unchanged": sha(FINAL) == config["input_sha256"][str(FINAL)],
        "checks": ["Frozen100targets", "pairedinitialization", "identicalcleanvictims", "budget2", "noheldoutortargetinjection", "all1200ranksrecomputedfromcheckpoints"],
        "interpretation": "One seed. Fresh128 attacker equals clean128 victim because architecture, seed and training settings match. This is a clear-box condition; it does not establish independent-model transfer."})
    lines = ["# Fresh attacker surrogates: UMLS RQ1", "", "| Attacker dimension | Attack | Hit@10 | MRR | Injected targets | New triplets |", "|---:|---|---:|---:|---:|---:|"]
    for row in rows:
        lines.append(f"| {row['surrogate_dim']} | {row['attack']} | {row['Hit_at_10']:.4f} | {row['MRR']:.8f} | {row['injected_targets']} | {row['unique_injected_triplets']} |")
    lines += ["", "Victim: TransE128; clean Hit@10 = 0.30, MRR = 0.11094261788036644. All100 original targets remain in each evaluation.",
              "Both surrogates and paired victims train from scratch for500epochs, Adam lr0.01, batch256, margin1, seed42. Only surrogate dimension varies between the fresh attacker conditions.",
              "MaSS uses supervisedPRA threshold0.9, top10 templates, top10 entities, delta0.7 and baseline selection. All methods have budget2. DirectAdd and ComAttack retain the project's target-promotion adaptations.",
              "Fresh128 surrogate and clean128 victim have identical weights under these matched settings; this condition is clear-box. Results are from one seed and do not establish general transfer or guaranteed improvement.",
              "Targets were originally selected with64-dimensional TransE ranks11–15; they were not reselected. The final RQ1 CSV is unchanged.",
              "", "Reproduce: `.venv\\Scripts\\python.exe Mass_implement\\Experiment_Setup\\Evaluation\\Evaluate_RQ1_Fresh_Surrogates.py --resume`"]
    (ROOT / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    config = {"dimensions": [64, 128], "surrogate": {"epochs": 500, "batch_size": 256, "lr": .01, "margin": 1., "seed": 42},
              "victim": {"dim": 128, "epochs": 500, "batch_size": 256, "lr": .01, "margin": 1., "seed": 42},
              "attack": {"budget": 2, "delta": .7, "seed": 42, "selection_policy": "baseline", "threshold": .9, "top_k": 10, "top_entities": 10},
              "torch_threads": 1, "torch_version": str(torch.__version__), "input_sha256": snapshot()}
    config_path = ROOT / "sweep_config.json"
    if config_path.exists():
        if not args.resume or read_json(config_path) != config:
            raise ValueError("Existing sweep requires --resume with unchanged configuration and sources")
    else:
        if ROOT.exists() and any(ROOT.iterdir()):
            raise ValueError("Output directory contains untracked results")
        write_json(config_path, config)
    reference_state = torch.load(EXP / "RQ1" / "transe_dim128" / "baseline" / "seed_42" / "victim_models" / "clean_transe.pt", map_location="cpu", weights_only=True)["state_dict"]
    rows = []
    for dimension in config["dimensions"]:
        point = ROOT / f"dim_{dimension}"
        checkpoint = point / "surrogate" / "transe_umls.pt"
        stage(point, "surrogate", lambda: train_surrogate(checkpoint, dim=dimension, **config["surrogate"]), point / "surrogate", config)
        stage(point, "feasibility", lambda: feasibility(checkpoint, point / "feasibility"), point / "feasibility", config)
        for attack, folder in ATTACKS:
            attack_point = point / folder
            poison = attack_point / "poisoning"
            def construct():
                if attack == "MaSS":
                    construct_mass(targets_file=TARGETS, feasibility_dir=point / "feasibility", output_dir=poison,
                                   budget=2, delta=.7, seed=42, selection_policy="baseline")
                else:
                    construct_baseline(attack, targets_file=TARGETS, output_dir=poison,
                                       surrogate_checkpoint=checkpoint, budget=2, seed=42, directadd_step=.1)
            stage(attack_point, "poisoning", construct, poison, config)
            train, evaluate = victim_args(attack_point, poison, attack)
            stage(attack_point, "victim_models", lambda: train_victims(train), train.output_dir, config)
            stage(attack_point, "evaluation", lambda: evaluate_victims(evaluate), evaluate.output_dir, config)
            rows.append(validate(attack_point, dimension, attack, folder, reference_state))
            print(json.dumps(rows[-1]), flush=True)
            export(rows, config)
    verify_hashes(config["input_sha256"])
    print(f"Completed six conditions. Final RQ1 CSV unchanged. Results: {ROOT}", flush=True)


if __name__ == "__main__":
    main()
