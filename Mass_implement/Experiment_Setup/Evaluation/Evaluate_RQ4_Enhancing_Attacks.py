#!/usr/bin/env python3
"""UMLS Table 3: independent type/plausibility enhancements to baseline attacks.

TransE remains the frozen surrogate; matched ConvE victims evaluate every
condition on all fixed Set B targets and the original Set A test triples.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import sys
from pathlib import Path

MASS_DIR = Path(__file__).resolve().parents[2]
SETUP = MASS_DIR / "Experiment_Setup"
for directory in (MASS_DIR, SETUP):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

import numpy as np
import torch

from Path_Generation_Module.ConvE_Victim import ConvE
from Path_Generation_Module.TransE_Surrogate import Surrogate
from construct_attack_baselines import construct, read_triples
from semantic_path_diagnostics import load_ontology
from train_conve_victim_models import run as train_victims
from train_victim_models import sha256_initial_state
from Evaluate_conve_victim_models import run as evaluate_victims
from Evaluate_victim_models import compute_metrics, verify_training_sources
from Evaluate_RQ4_Embedding_Sensitivity import complete_stage, point_arguments
from RQ4_Attack_Enhancements import SurrogateScores, calibrate_plausibility, EnhancementFilter, type_allowed
from RQ4_Enhancement_Table import export_table3

DATA = MASS_DIR / "dataset_uml"
DEFAULT_EXPERIMENT = MASS_DIR / "outputs" / "Experiments" / "UMLS_feasible_rank11_15_seed42"
ATTACKS = ("DirectAdd", "ComAttack")
ENHANCEMENTS = ("none", "type", "plausibility")


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_hashes(hashes):
    for name, expected in hashes.items():
        if not Path(name).is_file() or file_hash(Path(name)) != expected:
            raise ValueError(f"Recorded enhancement input changed: {name}")


def source_snapshot(args):
    files = [DATA / name for name in ("train.txt", "valid.txt", "test.txt", "entities.txt", "relations.txt")]
    files += [args.checkpoint, args.ontology, args.experiment_dir / "setup" / "set_B_target_triples.tsv"]
    files += [MASS_DIR / name for name in (
        "Experiment_Setup/construct_attack_baselines.py", "Experiment_Setup/semantic_path_diagnostics.py",
        "Experiment_Setup/train_conve_victim_models.py", "Experiment_Setup/train_victim_models.py",
        "Path_Generation_Module/TransE_Surrogate.py", "Path_Generation_Module/ConvE_Victim.py",
        "Experiment_Setup/Evaluation/Evaluate_victim_models.py",
        "Experiment_Setup/Evaluation/Evaluate_conve_victim_models.py",
        "Experiment_Setup/Evaluation/Evaluate_RQ4_Embedding_Sensitivity.py",
        "Experiment_Setup/Evaluation/Evaluate_RQ4_Enhancing_Attacks.py",
        "Experiment_Setup/Evaluation/RQ4_Attack_Enhancements.py",
        "Experiment_Setup/Evaluation/RQ4_Enhancement_Table.py",
    )]
    for attack in ATTACKS:
        reference = args.experiment_dir / attack
        for name in ("injected_triples.tsv", "poisoned_train.txt", "poisoning_summary.json"):
            path = reference / "poisoning" / name
            if path.is_file():
                files.append(path)
        if not args.fresh_baselines:
            for seed in args.seeds:
                for name in ("clean_conve.pt", "poisoned_conve.pt", "training_summary.json",
                             "clean_training_loss.csv", "poisoned_training_loss.csv"):
                    path = reference / f"seed_{seed}" / "conve_victim_models" / name
                    if path.is_file():
                        files.append(path)
    return {str(path.resolve()): file_hash(path) for path in files}


def baseline_reference_check(point, attack, seed, args):
    reference = args.experiment_dir / attack / "poisoning"
    summary_path = reference / "poisoning_summary.json"
    if not summary_path.is_file():
        return "no saved reference"
    old = read_json(summary_path)
    expected = {"budget_distinct_new_triples_per_target": args.budget,
                "attack_seed": seed, "directadd_promotion_step": args.promotion_step}
    if (any(old["parameters"].get(key) != value for key, value in expected.items())
            or old["source_sha256"].get("surrogate_checkpoint") != file_hash(args.checkpoint)
            or old["source_sha256"].get("targets") != file_hash(args.experiment_dir / "setup" / "set_B_target_triples.tsv")):
        return "skipped: reference configuration differs"
    for name in ("injected_triples.tsv", "poisoned_train.txt"):
        if file_hash(point / "poisoning" / name) != file_hash(reference / name):
            raise AssertionError(f"Unenhanced {attack} no longer reproduces its saved {name}")
    return "passed"


def train_or_reuse(point, attack, enhancement, seed, args, config, train_args):
    reference = args.experiment_dir / attack / f"seed_{seed}" / "conve_victim_models"
    metadata_path = reference / "training_summary.json"
    if enhancement == "none" and not args.fresh_baselines and metadata_path.is_file():
        metadata = read_json(metadata_path)
        expected = {"epochs": config["victim"]["epochs"], "embedding_dim": config["victim"]["dim"],
                    "random_seed": seed, "learning_rate": config["victim"]["lr"],
                    **{key: config["victim"][key] for key in (
                        "embedding_height", "num_filters", "kernel_size", "input_dropout",
                        "feature_dropout", "hidden_dropout", "label_smoothing", "batch_size")}}
        if metadata.get("victim_model") == "ConvE" and all(metadata["parameters"].get(k) == v for k, v in expected.items()):
            verify_training_sources(
                metadata, reference / "clean_conve.pt", reference / "poisoned_conve.pt",
                train_args.clean_train, train_args.poisoned_train, train_args.injected,
                train_args.valid, train_args.test, train_args.targets,
            )
            destination = point / "conve_victim_models"
            destination.mkdir(parents=True, exist_ok=True)
            for name in ("clean_conve.pt", "poisoned_conve.pt", "clean_training_loss.csv", "poisoned_training_loss.csv"):
                shutil.copy2(reference / name, destination / name)
            metadata["reuse_provenance"] = {"source_training_summary": str(metadata_path),
                                             "source_training_summary_sha256": file_hash(metadata_path),
                                             "settings_and_dataset_hashes_verified": True}
            for label in ("clean", "poisoned"):
                metadata["victims"][label]["checkpoint"] = str(destination / f"{label}_conve.pt")
                metadata["victims"][label]["training_loss_csv"] = str(destination / f"{label}_training_loss.csv")
            metadata["datasets"].update({"clean_train_file": str(train_args.clean_train),
                                         "poisoned_train_file": str(train_args.poisoned_train),
                                         "injected_triples_file": str(train_args.injected)})
            write_json(destination / "training_summary.json", metadata)
            print("Reused verified, byte-identical original baseline ConvE checkpoints.")
            return
    train_victims(train_args)


def collect_case(point, attack, enhancement, seed, targets, score_table, domain, range_, threshold, config):
    summary = read_json(point / "conve_evaluation" / "evaluation_summary.json")
    training = read_json(point / "conve_victim_models" / "training_summary.json")
    with (point / "conve_evaluation" / "set_B_target_rank_comparison.csv").open(encoding="utf-8-sig", newline="") as stream:
        b_rows = list(csv.DictReader(stream))
    if [tuple(row[key] for key in ("head", "relation", "tail")) for row in b_rows] != targets:
        raise AssertionError("Enhancement evaluation changed or dropped the frozen targets")
    with (point / "conve_evaluation" / "set_A_triple_rank_comparison.csv").open(encoding="utf-8-sig", newline="") as stream:
        a_rows = list(csv.DictReader(stream))
    metrics = {}
    for name, rows in (("B", b_rows), ("A", a_rows)):
        calculated = compute_metrics([float(row["poisoned_rank"]) for row in rows])
        for key, metric in (("Hit_at_10", "Hits@10"), ("MRR", "MRR")):
            expected = summary[f"set_{name}_{'Hits_at_10' if key == 'Hit_at_10' else 'MRR'}_poisoned"]
            if not math.isclose(calculated[metric], expected, abs_tol=1e-12):
                raise AssertionError("Table metrics differ from saved ranks")
            metrics[f"set_{name}_{key}"] = calculated[metric]
    injected = read_triples(point / "poisoning" / "injected_triples.tsv", require_unique=True)
    audit = [{"head": h, "relation": r, "tail": t,
              "type_passed": type_allowed((h, r, t), domain, range_),
              "surrogate_score": score_table.score((h, r, t)),
              "plausibility_threshold": threshold,
              "plausibility_passed": score_table.score((h, r, t)) >= threshold} for h, r, t in injected]
    if enhancement == "type" and not all(row["type_passed"] for row in audit):
        raise AssertionError("Type-enhanced attack injected a type-invalid triple")
    if enhancement == "plausibility" and not all(row["plausibility_passed"] for row in audit):
        raise AssertionError("Plausibility-enhanced attack violated its calibrated cutoff")
    write_csv(point / "constraint_audit.csv", audit)
    write_json(point / "constraint_audit.json", audit)
    model, _, _, model_config = ConvE.load(point / "conve_victim_models" / "clean_conve.pt")
    if model_config["embedding_dim"] != config["victim"]["dim"] or not training["same_initial_embeddings_verified"]:
        raise AssertionError("ConvE model or initialization control failed")
    return {
        "attack": attack, "enhancement": enhancement, "seed": seed, **metrics,
        "num_targets": len(targets), "targets_with_injections": summary["num_injected_targets"],
        "unique_injected": len(injected), "type_pass_rate": sum(row["type_passed"] for row in audit) / len(audit),
        "plausibility_pass_rate": sum(row["plausibility_passed"] for row in audit) / len(audit),
        "mean_surrogate_score": float(np.mean([row["surrogate_score"] for row in audit])),
        "set_A_Hit_at_10_clean": summary["set_A_Hits_at_10_clean"], "set_A_MRR_clean": summary["set_A_MRR_clean"],
        "set_B_Hit_at_10_clean": summary["set_B_Hits_at_10_clean"], "set_B_MRR_clean": summary["set_B_MRR_clean"],
        "victim_dim": model_config["embedding_dim"], "clean_state_sha256": sha256_initial_state(model),
        "initial_state_sha256": training["victims"]["clean"]["initial_state_sha256"],
        "reference_victims_reused": "reuse_provenance" in training,
        "injected_sha256": file_hash(point / "poisoning" / "injected_triples.tsv"),
    }


def verify_controls(rows):
    references = {}
    for row in rows:
        previous = references.setdefault(row["seed"], row)
        for key in ("num_targets", "victim_dim", "clean_state_sha256", "initial_state_sha256",
                    "set_A_Hit_at_10_clean", "set_A_MRR_clean", "set_B_Hit_at_10_clean", "set_B_MRR_clean"):
            if row[key] != previous[key]:
                raise AssertionError(f"Clean control changed across Table 3 conditions: {key}")


def run(args):
    args.experiment_dir = args.experiment_dir.resolve()
    args.checkpoint = args.checkpoint.resolve()
    args.ontology = args.ontology.resolve()
    output = (args.output_dir or args.experiment_dir / "RQ4" / "enhancing_attacks").resolve()
    if args.plot_only:
        manifest = read_json(output / "enhancement_manifest.json")
        for name, expected in manifest["measured_artifacts_sha256"].items():
            if file_hash(output / name) != expected:
                raise ValueError(f"Measured enhancement metrics changed: {name}")
        export_table3(read_json(output / "enhancement_metrics.json"), output, seeds=manifest["seeds"])
        print(f"Re-exported measured UMLS Table 3: {output}")
        return
    if (len(set(args.seeds)) != len(args.seeds) or any(seed < 0 for seed in args.seeds)
            or args.budget < 2 or args.budget % 2 or min(args.victim_epochs, args.victim_dim, args.torch_threads) < 1
            or args.victim_dim % 8 or args.victim_dim < 24 or not 0 <= args.plausibility_quantile <= 1
            or args.promotion_step <= 0):
        raise ValueError("Invalid seed, budget, victim configuration or calibration quantile")
    torch.set_num_threads(args.torch_threads)
    sources = source_snapshot(args)
    surrogate = Surrogate.load(args.checkpoint)
    score_table = SurrogateScores.from_surrogate(surrogate)
    clean_train = read_triples(DATA / "train.txt", require_unique=True)
    calibration = calibrate_plausibility(score_table, clean_train, quantile=args.plausibility_quantile)
    domain, range_, _ = load_ontology(args.ontology)
    targets_path = args.experiment_dir / "setup" / "set_B_target_triples.tsv"
    targets = read_triples(targets_path, require_unique=True)
    config = {
        "dataset": "UMLS", "attacks": ATTACKS, "enhancements": ENHANCEMENTS,
        "seeds": sorted(args.seeds), "budget": args.budget, "torch_threads": args.torch_threads,
        "checkpoint": str(args.checkpoint), "ontology": str(args.ontology), "promotion_step": args.promotion_step,
        "fresh_baselines": args.fresh_baselines, "calibration": calibration,
        "victim": {"epochs": args.victim_epochs, "dim": args.victim_dim, "embedding_height": 8,
                   "num_filters": 32, "kernel_size": 3, "input_dropout": 0.2, "feature_dropout": 0.2,
                   "hidden_dropout": 0.3, "label_smoothing": 0.1, "batch_size": 256, "lr": 0.001},
        "input_sha256": sources, "environment": {"python": sys.version, "torch": torch.__version__, "numpy": np.__version__},
        "table_populations": {"type_constraint": "Set B attack targets", "plausibility_constraint": "Set A clean test utility"},
        "population_qualification": "Explicit UMLS interpretation motivated by paper prose; Table 3 does not identify metric populations. Full Set A and B metrics are saved for every condition.",
    }
    config_path = output / "enhancement_config.json"
    if config_path.is_file():
        if not args.resume:
            raise FileExistsError(f"Enhancement experiment exists: {output}; use --resume or --plot-only")
        if read_json(config_path) != json.loads(json.dumps(config)):
            raise ValueError("Saved enhancement configuration or sources differ; choose a new output directory")
    else:
        if args.resume:
            raise FileNotFoundError("No saved enhancement experiment to resume")
        if output.exists() and any(output.iterdir()):
            raise FileExistsError("Output directory is not empty")
        write_json(config_path, config)
        write_json(output / "plausibility_calibration.json", calibration)
        write_csv(output / "plausibility_calibration_scores.csv", [
            {"head": h, "relation": r, "tail": t, "score": score_table.score((h, r, t))} for h, r, t in clean_train])
    print(f"Clean-training q={calibration['quantile']}: TransE cutoff={calibration['threshold']:.6f}", flush=True)
    rows, references = [], []
    for seed in config["seeds"]:
        for attack in ATTACKS:
            for enhancement in ENHANCEMENTS:
                point = output / attack / enhancement / f"seed_{seed}"
                point.mkdir(parents=True, exist_ok=True)
                manifest_path = point / "stage_manifest.json"
                state = read_json(manifest_path) if manifest_path.is_file() else {
                    "attack": attack, "enhancement": enhancement, "seed": seed, "stages": {}}
                if (state["attack"], state["enhancement"], state["seed"]) != (attack, enhancement, seed):
                    raise ValueError("Case manifest identifies different conditions")
                print(f"\nTable 3: {attack}, {enhancement}, seed={seed}", flush=True)
                predicate = EnhancementFilter(enhancement, domain, range_, score_table, calibration["threshold"])
                enhancement_metadata = {
                    "enabled_constraints": [] if enhancement == "none" else [enhancement],
                    "type_rule": "head in Domain(relation) and tail in Range(relation); marginal SRSTRE2 sets",
                    "plausibility_rule": "frozen clean TransE negative L2 >= global clean-training quantile cutoff",
                    "plausibility_quantile": calibration["quantile"], "plausibility_threshold": calibration["threshold"],
                    "calibration_train_sha256": file_hash(DATA / "train.txt"), "ontology_sha256": file_hash(args.ontology),
                    "surrogate_sha256": file_hash(args.checkpoint),
                }
                training_args, evaluation_args = point_arguments(point, targets_path, config, seed)
                training_args.attack_name = evaluation_args.attack_name = attack
                operations = {
                    "poisoning": lambda: construct(
                        attack, targets_file=targets_path, output_dir=point / "poisoning",
                        surrogate_checkpoint=args.checkpoint, budget=args.budget, seed=seed,
                        directadd_step=args.promotion_step, edge_filter=None if enhancement == "none" else predicate,
                        enhancement=enhancement, enhancement_metadata=enhancement_metadata),
                    "conve_victim_models": lambda: train_or_reuse(point, attack, enhancement, seed, args, config, training_args),
                    "conve_evaluation": lambda: evaluate_victims(evaluation_args),
                }
                for stage in ("poisoning", "conve_victim_models", "conve_evaluation"):
                    complete_stage(point, state, stage, operations[stage], allow_run=True)
                    if stage == "poisoning":
                        generated = read_json(point / stage / "poisoning_summary.json")
                        if not generated["num_unique_injected_triples"]:
                            raise RuntimeError("No poisoning meets the declared constraint; do not silently relax the cutoff")
                        if enhancement == "none":
                            references.append({"attack": attack, "seed": seed,
                                               "status": baseline_reference_check(point, attack, seed, args)})
                row = collect_case(point, attack, enhancement, seed, targets, score_table, domain, range_, calibration["threshold"], config)
                rows.append(row)
                verify_controls(rows)
                write_csv(output / "enhancement_metrics.csv", rows)
                write_json(output / "enhancement_metrics.json", rows)
                print(f"  Set B Hit@10={row['set_B_Hit_at_10']:.4f}, MRR={row['set_B_MRR']:.4f}; "
                      f"Set A Hit@10={row['set_A_Hit_at_10']:.4f}, MRR={row['set_A_MRR']:.4f}; "
                      f"injections={row['unique_injected']}, covered targets={row['targets_with_injections']}/{len(targets)}", flush=True)
    verify_hashes(sources)
    export_table3(rows, output, seeds=config["seeds"])
    write_json(output / "enhancement_manifest.json", {
        "dataset": "UMLS", "victim": "ConvE", "surrogate": "TransE", "seeds": config["seeds"],
        "budget": args.budget, "num_targets": len(targets), "calibration": calibration,
        "table_populations": config["table_populations"], "population_qualification": config["population_qualification"],
        "baseline_reference_checks": references, "clean_controls_identical_across_conditions": True,
        "input_sha256": sources,
        "measured_artifacts_sha256": {name: file_hash(output / name) for name in
                                     ("enhancement_metrics.csv", "enhancement_metrics.json")},
        "limitations": [
            "UMLS adaptation with current target-promotion baseline implementations, not the original paper datasets or exact source attack implementations.",
            "Type-only and plausibility-only enhancements are independent; no cumulative constraints or victim-informed threshold tuning.",
            "The global clean-training quantile cutoff is a declared local choice, not a paper-provided numerical threshold.",
            "The table's metric-population interpretation is explicit; use the full metrics to inspect both Set A and Set B for every condition.",
            "Single-seed defaults are preliminary; use --seeds and a new output directory for repeats.",
        ],
    })
    print(f"\nMeasured UMLS Table 3 and full metrics saved to {output}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dir", type=Path, default=DEFAULT_EXPERIMENT)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--checkpoint", type=Path, default=MASS_DIR / "outputs" / "Path_Generation" / "transe_umls.pt")
    parser.add_argument("--ontology", type=Path, default=DATA / "Semantic_Constraints" / "SRSTRE2")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--budget", type=int, default=2)
    parser.add_argument("--promotion-step", type=float, default=0.1)
    parser.add_argument("--plausibility-quantile", type=float, default=0.5)
    parser.add_argument("--victim-epochs", type=int, default=500)
    parser.add_argument("--victim-dim", type=int, default=64)
    parser.add_argument("--torch-threads", type=int, default=4)
    parser.add_argument("--fresh-baselines", action="store_true")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--resume", action="store_true")
    mode.add_argument("--plot-only", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
