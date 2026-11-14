#!/usr/bin/env python3
"""UMLS Figure 9: vary the TransE surrogate size while fixing the ConvE victim.

Run from Mass_implement:
    python Experiment_Setup/Evaluation/Evaluate_RQ4_Embedding_Sensitivity.py

Every point trains a fresh surrogate, regenerates MaSS paths and poisoning,
trains paired clean/poisoned ConvE victims, and evaluates all frozen targets.
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import json
import math
import statistics
import sys
import time
from pathlib import Path

MASS_DIR = Path(__file__).resolve().parents[2]
SETUP_DIR = MASS_DIR / "Experiment_Setup"
for directory in (MASS_DIR, SETUP_DIR):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

import numpy as np
import sklearn
import torch

from Path_Generation_Module.TransE_Surrogate import train as train_surrogate
from Path_Generation_Module.ConvE_Victim import ConvE
from construct_poisoned_dataset import construct as construct_poisoning, read_triples
from target_feasibility_check import run_feasibility
from train_conve_victim_models import run as train_victims
from train_victim_models import sha256_initial_state
from Evaluate_conve_victim_models import run as evaluate_victims
from RQ4_Embedding_Plot import aggregate_runs, plot_embedding_sensitivity

DEFAULT_EXPERIMENT = MASS_DIR / "outputs" / "Experiments" / "UMLS_feasible_rank11_15_seed42"
DATA = MASS_DIR / "dataset_uml"
STAGES = ("surrogate", "feasibility", "poisoning", "conve_victim_models", "conve_evaluation")


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_snapshot(experiment):
    files = [DATA / name for name in ("train.txt", "valid.txt", "test.txt", "entities.txt", "relations.txt")]
    files += [DATA / "Semantic_Constraints" / "SRSTRE2",
              MASS_DIR / "outputs" / "PRA" / "supervised_pra_umls.joblib",
              experiment / "setup" / "set_B_target_triples.tsv",
              experiment / "setup" / "target_setup_summary.json"]
    files += [MASS_DIR / path for path in (
        "Path_Generation_Module/TransE_Surrogate.py", "Path_Generation_Module/Entity_Selection.py",
        "Path_Generation_Module/Random_Sampling.py", "Path_Generation_Module/Stealthiness.py",
        "Path_Generation_Module/ConvE_Victim.py", "PRA_Module/PRA_supervised.py",
        "PRA_Module/Pra_implement_finalVersion.py", "Semantic_Constraints_Module/Semantic_Constraints.py",
        "Experiment_Setup/target_feasibility_check.py", "Experiment_Setup/construct_poisoned_dataset.py",
        "Experiment_Setup/train_conve_victim_models.py", "Experiment_Setup/train_victim_models.py",
        "Experiment_Setup/Evaluation/Evaluate_conve_victim_models.py",
        "Experiment_Setup/Evaluation/Evaluate_victim_models.py",
        "Experiment_Setup/Evaluation/Evaluate_RQ4_Embedding_Sensitivity.py",
        "Experiment_Setup/Evaluation/RQ4_Embedding_Plot.py",
    )]
    return {str(path.resolve()): file_hash(path) for path in files}


def verify_hashes(hashes, root=None):
    for name, expected in hashes.items():
        path = (root / name) if root else Path(name)
        if not path.is_file() or file_hash(path) != expected:
            raise ValueError(f"Recorded input or artifact changed: {path}")


def generate_feasibility(**kwargs):
    result = run_feasibility(**kwargs)
    if result["error_targets"]:
        raise RuntimeError("Feasibility contains execution errors; these are not infeasible targets")
    return result


def complete_stage(point, state, stage, operation, *, allow_run):
    """Only skip an already completed stage after verifying every artifact."""
    if stage in state["stages"]:
        verify_hashes(state["stages"][stage]["sha256"], point)
        print(f"  {stage}: verified saved artifacts", flush=True)
        return
    if not allow_run:
        raise ValueError(f"Incomplete point: {point} / {stage}; use --resume")
    print(f"  {stage}: running...", flush=True)
    start = time.perf_counter()
    with (point / "run.log").open("a", encoding="utf-8") as log:
        with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
            operation()
    artifacts = sorted((point / stage).rglob("*"))
    hashes = {path.relative_to(point).as_posix(): file_hash(path) for path in artifacts if path.is_file()}
    if not hashes:
        raise AssertionError(f"Stage {stage} produced no artifacts")
    state["stages"][stage] = {"elapsed_seconds": time.perf_counter() - start, "sha256": hashes}
    write_json(point / "stage_manifest.json", state)
    print(f"  {stage}: completed in {state['stages'][stage]['elapsed_seconds']:.1f}s", flush=True)


def point_arguments(point, targets, config, seed):
    poisoning = point / "poisoning"
    victim = point / "conve_victim_models"
    training = argparse.Namespace(
        clean_train=DATA / "train.txt", poisoned_train=poisoning / "poisoned_train.txt",
        injected=poisoning / "injected_triples.tsv", valid=DATA / "valid.txt", test=DATA / "test.txt",
        targets=targets, entities=DATA / "entities.txt", relations=DATA / "relations.txt",
        output_dir=victim, attack_name="MaSS", seed=seed, log_every=50, overwrite=True,
        **config["victim"],
    )
    evaluation = argparse.Namespace(
        clean_checkpoint=victim / "clean_conve.pt", poisoned_checkpoint=victim / "poisoned_conve.pt",
        train=DATA / "train.txt", poisoned_train=poisoning / "poisoned_train.txt",
        injected=poisoning / "injected_triples.tsv", valid=DATA / "valid.txt", test=DATA / "test.txt",
        targets=targets, poisoning_summary=poisoning / "poisoning_summary.csv",
        training_summary=victim / "training_summary.json", output_dir=point / "conve_evaluation",
        attack_name="MaSS", overwrite=True,
    )
    return training, evaluation


def collect_point(point, dimension, seed, targets, config):
    """Recompute chart metrics from target ranks and check experimental controls."""
    checkpoint = torch.load(point / "surrogate" / "transe_umls.pt", map_location="cpu", weights_only=True)
    if (checkpoint["embedding_dim"] != dimension
            or checkpoint["state_dict"]["entity.weight"].shape[1] != dimension):
        raise AssertionError("Surrogate checkpoint has the wrong embedding dimension")
    feasibility = read_json(point / "feasibility" / "feasibility_summary.json")
    expected_checkpoint = point / "surrogate" / "transe_umls.pt"
    if Path(feasibility["config"]["transe_checkpoint"]).resolve() != expected_checkpoint.resolve():
        raise AssertionError("Paths were not generated with this point's surrogate")
    poisoning = read_json(point / "poisoning" / "poisoning_summary.json")
    training = read_json(point / "conve_victim_models" / "training_summary.json")
    summary = read_json(point / "conve_evaluation" / "evaluation_summary.json")
    with (point / "conve_evaluation" / "set_B_target_rank_comparison.csv").open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if [tuple(row[key] for key in ("head", "relation", "tail")) for row in rows] != targets:
        raise AssertionError("Evaluation changed or dropped frozen targets")
    ranks = [float(row["poisoned_rank"]) for row in rows]
    if any(not math.isfinite(rank) or rank < 1 for rank in ranks):
        raise AssertionError("Invalid target ranks")
    hit = statistics.mean(rank <= 10 for rank in ranks)
    mrr = statistics.mean(1 / rank for rank in ranks)
    for actual, expected in ((hit, summary["set_B_Hits_at_10_poisoned"]), (mrr, summary["set_B_MRR_poisoned"])):
        if not math.isclose(actual, expected, abs_tol=1e-12):
            raise AssertionError("Summary metrics differ from saved target ranks")
    if summary["num_set_B_targets"] != len(targets) or not training["same_initial_embeddings_verified"]:
        raise AssertionError("Victim initialization or target-count control failed")
    model, _, _, victim_config = ConvE.load(point / "conve_victim_models" / "clean_conve.pt")
    if victim_config["embedding_dim"] != config["victim"]["dim"]:
        raise AssertionError("ConvE dimension varied with surrogate size")
    return {
        "surrogate_dim": dimension, "seed": seed, "Hit_at_10": hit, "MRR": mrr,
        "clean_Hit_at_10": summary["set_B_Hits_at_10_clean"], "clean_MRR": summary["set_B_MRR_clean"],
        "delta_Hit_at_10": hit - summary["set_B_Hits_at_10_clean"],
        "delta_MRR": mrr - summary["set_B_MRR_clean"],
        "num_set_B_targets": len(targets), "num_injected_targets": summary["num_injected_targets"],
        "unique_injected_triples": summary["num_injected_triples"],
        "feasible_targets": feasibility["feasible_targets"],
        "set_A_Hit_at_10_clean": summary["set_A_Hits_at_10_clean"],
        "set_A_Hit_at_10_poisoned": summary["set_A_Hits_at_10_poisoned"],
        "set_A_MRR_clean": summary["set_A_MRR_clean"], "set_A_MRR_poisoned": summary["set_A_MRR_poisoned"],
        "victim_dim": config["victim"]["dim"],
        "victim_initial_state_sha256": training["victims"]["clean"]["initial_state_sha256"],
        "clean_victim_state_sha256": sha256_initial_state(model),
        "surrogate_checkpoint_sha256": file_hash(expected_checkpoint),
        "injected_sha256": file_hash(point / "poisoning" / "injected_triples.tsv"),
        "poisoning_parameters": json.dumps(poisoning["parameters"], sort_keys=True),
    }


def verify_clean_controls(rows):
    references = {}
    for row in rows:
        reference = references.setdefault(row["seed"], row)
        for key in ("victim_dim", "victim_initial_state_sha256", "clean_victim_state_sha256",
                    "clean_Hit_at_10", "clean_MRR", "num_set_B_targets"):
            if row[key] != reference[key]:
                raise AssertionError(f"Clean control varied across surrogate sizes: seed={row['seed']}, {key}")


def run(args):
    if (len(set(args.dimensions)) != len(args.dimensions) or any(d < 1 for d in args.dimensions)
            or len(set(args.seeds)) != len(args.seeds) or any(s < 0 for s in args.seeds)):
        raise ValueError("Dimensions must be distinct positive integers and seeds distinct nonnegative integers")
    if min(args.surrogate_epochs, args.victim_epochs, args.victim_dim, args.torch_threads) < 1:
        raise ValueError("Epochs, victim dimension, and thread count must be positive")
    if args.victim_dim % 8 or args.victim_dim // 8 < 3:
        raise ValueError("ConvE victim dimension must be divisible by 8 and at least 24")
    torch.set_num_threads(args.torch_threads)
    experiment = args.experiment_dir.resolve()
    output = (args.output_dir or experiment / "RQ4" / "embedding_sensitivity").resolve()
    config_path = output / "sweep_config.json"
    config = {
        "dataset": "UMLS", "experiment_dir": str(experiment),
        "dimensions": sorted(args.dimensions), "seeds": sorted(args.seeds), "torch_threads": args.torch_threads,
        "surrogate": {"epochs": args.surrogate_epochs, "batch_size": 256, "lr": 0.01, "margin": 1.0},
        "attack": {"pra_method": "supervised", "threshold": 0.9, "top_k": 10, "top_entities": 10,
                   "delta": 0.7, "budget": 2, "selection_policy": "baseline"},
        "victim": {"epochs": args.victim_epochs, "dim": args.victim_dim, "embedding_height": 8,
                   "num_filters": 32, "kernel_size": 3, "input_dropout": 0.2, "feature_dropout": 0.2,
                   "hidden_dropout": 0.3, "label_smoothing": 0.1, "batch_size": 256, "lr": 0.001},
        "environment": {"python": sys.version, "torch": torch.__version__, "numpy": np.__version__,
                        "sklearn": sklearn.__version__},
        "input_sha256": {} if args.plot_only else source_snapshot(experiment),
        "metric": "Poisoned ConvE filtered tail prediction on ALL frozen Set B targets; MRR bars and Hit@10 line.",
        "limitations": [
            "UMLS adaptation: original paper Figure 9 uses FB15k-237 and WN18RR.",
            "Fixed 100 targets selected previously with a 64-dimensional TransE rank-11--15 protocol.",
            "ConvE dimension remains fixed at the specified local setting; paper default was 128.",
            "Paper reports ten independent attacks; use --seeds for repeats. Single-seed runs are preliminary.",
            "Fixed epoch budgets are controlled local settings, not a proof of convergence.",
        ],
    }
    if args.plot_only:
        config = read_json(config_path)
        verify_hashes(config["input_sha256"])
        torch.set_num_threads(config["torch_threads"])
        experiment = Path(config["experiment_dir"])
    elif config_path.exists():
        if not args.resume:
            raise FileExistsError(f"Sweep already exists: {output}; use --resume or --plot-only")
        if read_json(config_path) != config:
            raise ValueError("Saved configuration, source files, code, or environment differs; choose a new output directory")
    else:
        if args.resume:
            raise FileNotFoundError("No saved sweep configuration to resume")
        if output.exists() and any(output.iterdir()):
            raise FileExistsError(f"Output directory is not empty: {output}")
        write_json(config_path, config)

    targets_path = experiment / "setup" / "set_B_target_triples.tsv"
    targets = read_triples(targets_path, require_unique=True)
    rows = []
    for seed in config["seeds"]:
        for dimension in config["dimensions"]:
            point = output / f"seed_{seed}" / f"dim_{dimension}"
            point.mkdir(parents=True, exist_ok=True)
            manifest_path = point / "stage_manifest.json"
            state = read_json(manifest_path) if manifest_path.is_file() else {
                "surrogate_dim": dimension, "seed": seed, "stages": {}}
            if state["surrogate_dim"] != dimension or state["seed"] != seed:
                raise ValueError("Point manifest identifies a different experiment")
            print(f"\nRQ4: TransE dim={dimension}, seed={seed} -> fixed ConvE dim={config['victim']['dim']}", flush=True)
            training_args, evaluation_args = point_arguments(point, targets_path, config, seed)
            checkpoint = point / "surrogate" / "transe_umls.pt"
            attack = config["attack"]
            operations = {
                "surrogate": lambda: train_surrogate(checkpoint, dim=dimension, seed=seed, **config["surrogate"]),
                "feasibility": lambda: generate_feasibility(
                    targets_file=targets_path, output_dir=point / "feasibility", checkpoint=checkpoint,
                    pra_method=attack["pra_method"], threshold=attack["threshold"], top_k=attack["top_k"],
                    top_entities=attack["top_entities"], metadata_file=experiment / "setup" / "target_setup_summary.json"),
                "poisoning": lambda: construct_poisoning(
                    targets_file=targets_path, feasibility_dir=point / "feasibility", output_dir=point / "poisoning",
                    budget=attack["budget"], delta=attack["delta"], seed=seed, selection_policy=attack["selection_policy"]),
                "conve_victim_models": lambda: train_victims(training_args),
                "conve_evaluation": lambda: evaluate_victims(evaluation_args),
            }
            for stage in STAGES:
                complete_stage(point, state, stage, operations[stage], allow_run=not args.plot_only)
                if stage == "feasibility" and read_json(point / stage / "feasibility_summary.json")["error_targets"]:
                    raise RuntimeError("Feasibility contains execution errors; these are not infeasible targets")
            row = collect_point(point, dimension, seed, targets, config)
            rows.append(row)
            verify_clean_controls(rows)
            print(f"  Measured all {len(targets)} targets: Hit@10={row['Hit_at_10']:.4f}, MRR={row['MRR']:.6f}", flush=True)
            write_csv(output / "embedding_sensitivity_runs.csv", rows)
            write_json(output / "embedding_sensitivity_runs.json", rows)
    aggregate = aggregate_runs(rows, config["dimensions"], config["seeds"])
    write_csv(output / "embedding_sensitivity_summary.csv", aggregate)
    write_json(output / "embedding_sensitivity_summary.json", aggregate)
    plot_embedding_sensitivity(aggregate, output, victim_dim=config["victim"]["dim"])
    verify_hashes(config["input_sha256"])
    print(f"\nFigure 9 and measured metrics saved to {output}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dir", type=Path, default=DEFAULT_EXPERIMENT)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--dimensions", type=int, nargs="+", default=[32, 64, 128])
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--surrogate-epochs", type=int, default=500)
    parser.add_argument("--victim-epochs", type=int, default=500)
    parser.add_argument("--victim-dim", type=int, default=64)
    parser.add_argument("--torch-threads", type=int, default=4)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--resume", action="store_true", help="Verify and reuse completed stages")
    mode.add_argument("--plot-only", action="store_true", help="Verify completed runs and redraw from measured ranks")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
