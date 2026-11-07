#!/usr/bin/env python3
"""MaSS UMLS: policy-controlled paired multi-seed RQ1/RQ2 experiments.

Place at: Mass_implement/Experiment_Setup/Evaluation/run_multi_seed_evaluation.py
Run from Mass_implement/:
    python Experiment_Setup/Evaluation/run_multi_seed_evaluation.py --seeds 42 43 44 45 46 --budgets 2 4

Default (fixed attack): make ONE poisoned dataset per budget, then train 5 independently
initialized clean/poisoned victim PAIRS. --attack-mode vary also varies attack sampling.
Reuses existing construct_poisoned_dataset.py, train_victim_models.py and
Evaluation/Evaluate_victim_models.py; does not modify these implementations.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import statistics
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

MASS_DIR = Path(__file__).resolve().parents[2]
PILOT_DIR = MASS_DIR / "outputs" / "Experiments" / "UMLS_pilot"
DATASET_DIR = MASS_DIR / "dataset_uml"
SETUP_DIR = MASS_DIR / "Experiment_Setup"
CONSTRUCT_SCRIPT = SETUP_DIR / "construct_poisoned_dataset.py"
TRAIN_SCRIPT = SETUP_DIR / "train_victim_models.py"
EVAL_SCRIPT = SETUP_DIR / "Evaluation" / "Evaluate_victim_models.py"

POISON_FILES = ("poisoned_train.txt", "injected_triples.tsv", "poisoning_summary.csv", "poisoning_summary.json")
TRAIN_FILES = ("clean_transe.pt", "poisoned_transe.pt", "training_summary.json", "clean_training_loss.csv", "poisoned_training_loss.csv")
EVAL_FILES = ("set_B_target_rank_comparison.csv", "set_B_attack_metrics.json", "attack_effectiveness_summary.json",
              "set_A_triple_rank_comparison.csv", "set_A_clean_performance.json", "evaluation_summary.json")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    result = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(result, dict):
        raise ValueError(f"JSON object expected: {path}")
    return result


def save_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def ensure_manifest(path: Path, expected: dict[str, Any], *, resume: bool) -> None:
    if path.exists():
        actual = read_json(path)
        if actual != expected:
            raise ValueError(f"Configuration/source mismatch at {path}; use a new output root or --overwrite")
        if not resume:
            raise FileExistsError(f"Existing run: {path}; use --resume or --overwrite")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        save_json(path, expected)


def stage(cmd: list[str], folder: Path, files: tuple[str, ...], *, resume: bool) -> None:
    exists = [(folder / f).is_file() for f in files]
    if all(exists):
        if not resume:
            raise FileExistsError(f"Stage already exists: {folder}; use --resume or --overwrite")
        print(f"[reuse] {folder}", flush=True)
        return
    if any(exists):
        raise RuntimeError(f"Stage is incomplete: {folder}. Use --overwrite to rebuild selected runs.")
    print("[execute] " + subprocess.list2cmdline(cmd), flush=True)
    subprocess.run(cmd, cwd=MASS_DIR, check=True)
    if not all((folder / f).is_file() for f in files):
        raise RuntimeError(f"Stage did not produce all expected artifacts: {folder}")


def source_fingerprints(args: argparse.Namespace) -> dict[str, str]:
    files = {
        "targets": args.targets, "train": DATASET_DIR / "train.txt",
        "valid": DATASET_DIR / "valid.txt", "test": DATASET_DIR / "test.txt",
        "feasibility_summary": args.feasibility_dir / "feasibility_summary.json",
        "construct_script": CONSTRUCT_SCRIPT, "train_script": TRAIN_SCRIPT,
        "evaluate_script": EVAL_SCRIPT,
        "surrogate_module": MASS_DIR / "Path_Generation_Module" / "TransE_Surrogate.py",
        "sampling_module": MASS_DIR / "Path_Generation_Module" / "Random_Sampling.py",
        "stealthiness_module": MASS_DIR / "Path_Generation_Module" / "Stealthiness.py",
    }
    n_targets = len([s for s in args.targets.read_text(encoding="utf-8-sig").splitlines() if s.strip()])
    for i in range(1, n_targets + 1):
        files[f"feasibility_target_{i:03d}"] = args.feasibility_dir / f"target_{i:03d}.json"
    missing = [str(p) for p in files.values() if not p.is_file()]
    if missing:
        raise FileNotFoundError("Missing required sources:\n" + "\n".join(missing))
    return {key: digest(value) for key, value in files.items()}


def poison_command(args: argparse.Namespace, budget: int, attack_seed: int, poison: Path) -> list[str]:
    return [sys.executable, str(CONSTRUCT_SCRIPT), "--targets", str(args.targets),
            "--feasibility-dir", str(args.feasibility_dir), "--output-dir", str(poison),
            "--budget", str(budget), "--delta", str(args.delta), "--seed", str(attack_seed),
            "--selection-policy", args.selection_policy]


def train_command(args: argparse.Namespace, seed: int, poison: Path, victims: Path) -> list[str]:
    return [sys.executable, str(TRAIN_SCRIPT), "--poisoned-train", str(poison / "poisoned_train.txt"),
            "--injected", str(poison / "injected_triples.tsv"), "--targets", str(args.targets),
            "--output-dir", str(victims), "--epochs", str(args.epochs), "--dim", str(args.dim),
            "--batch-size", str(args.batch_size), "--lr", str(args.lr),
            "--margin", str(args.margin), "--seed", str(seed)]


def eval_command(args: argparse.Namespace, poison: Path, victims: Path, evaluation: Path) -> list[str]:
    return [sys.executable, str(EVAL_SCRIPT), "--poisoned-train", str(poison / "poisoned_train.txt"),
            "--injected", str(poison / "injected_triples.tsv"),
            "--poisoning-summary", str(poison / "poisoning_summary.csv"),
            "--clean-checkpoint", str(victims / "clean_transe.pt"),
            "--poisoned-checkpoint", str(victims / "poisoned_transe.pt"),
            "--training-summary", str(victims / "training_summary.json"),
            "--targets", str(args.targets), "--output-dir", str(evaluation)]


def metric_from_csv(rows: list[dict[str, str]]) -> dict[str, dict[str, float]]:
    output = {}
    for label in ("clean", "poisoned"):
        ranks = [float(row[f"{label}_rank"]) for row in rows]
        if not ranks or any(rank < 1 or not math.isfinite(rank) for rank in ranks):
            raise ValueError(f"Invalid {label} ranks")
        output[label] = {
            "count": len(ranks), "MRR": statistics.mean(1 / r for r in ranks),
            "Hits@10": statistics.mean(float(r <= 10) for r in ranks),
            "Mean_Rank": statistics.mean(ranks),
        }
    return output


def close(a: float, b: float) -> bool:
    return math.isclose(float(a), float(b), rel_tol=1e-9, abs_tol=1e-10)


def collect_run(args: argparse.Namespace, budget: int, seed: int, attack_seed: int,
                poison: Path, victims: Path, evaluation: Path) -> tuple[dict[str, Any], list[dict[str, str]]]:
    p = read_json(poison / "poisoning_summary.json")
    t = read_json(victims / "training_summary.json")
    b = read_json(evaluation / "attack_effectiveness_summary.json")
    a = read_json(evaluation / "set_A_clean_performance.json")["set_A_original_test_triples"]
    summary = read_json(evaluation / "evaluation_summary.json")
    pb = p["parameters"]
    if (int(pb["budget_distinct_new_triples_per_target"]) != budget or
        int(pb["seed"]) != attack_seed or not close(pb["delta"], args.delta) or
        pb.get("selection_policy") != args.selection_policy):
        raise ValueError(f"Poisoning configuration mismatch: {poison}")
    for key in ("injected_train_overlap", "injected_valid_overlap", "injected_test_overlap", "injected_set_B_overlap"):
        if p[key] != 0:
            raise ValueError(f"Leakage detected: {key} in {poison}")
    params = t["parameters"]
    if (int(params["random_seed"]) != seed or int(params["epochs"]) != args.epochs or
        int(params["embedding_dim"]) != args.dim or not close(params["learning_rate"], args.lr) or
        int(params["batch_size"]) != args.batch_size or not close(params["margin"], args.margin) or
        t["same_initial_embeddings_verified"] is not True):
        raise ValueError(f"Training settings mismatch: {victims}")
    mapping = {
        "clean_train": DATASET_DIR / "train.txt", "poisoned_train": poison / "poisoned_train.txt",
        "injected_triples": poison / "injected_triples.tsv", "validation": DATASET_DIR / "valid.txt",
        "test": DATASET_DIR / "test.txt", "set_B_targets": args.targets,
    }
    for key, path in mapping.items():
        if t["datasets"]["source_sha256"][key] != digest(path):
            raise ValueError(f"Training source hash mismatch: {path}")
    for model, filename in (("clean", "clean_transe.pt"), ("poisoned", "poisoned_transe.pt")):
        if t["victims"][model]["checkpoint_sha256"] != digest(victims / filename):
            raise ValueError(f"Checkpoint hash mismatch: {victims / filename}")
    if (p["num_unique_injected_triples"] != t["datasets"]["unique_injected_triples"] or
        p["num_unique_injected_triples"] != summary["num_injected_triples"]):
        raise ValueError(f"Injected triple count mismatch: {poison}")

    b_rows = read_csv(evaluation / "set_B_target_rank_comparison.csv")
    a_rows = read_csv(evaluation / "set_A_triple_rank_comparison.csv")
    if len(b_rows) != p["num_targets_set_B"] or len(a_rows) != summary["num_set_A_test_triples"]:
        raise ValueError("Evaluation CSV row-count mismatch")
    b_calc, a_calc = metric_from_csv(b_rows), metric_from_csv(a_rows)
    for label, metrics, report in (("SetB", b_calc, b["set_B_all"]), ("SetA", a_calc, a)):
        for victim in ("clean", "poisoned"):
            for metric in ("MRR", "Hits@10", "Mean_Rank"):
                if not close(metrics[victim][metric], report[victim][metric]):
                    raise ValueError(f"{label} {victim} {metric} differs from individual CSV ranks")

    bv, av = b["set_B_all"], a
    row = {
        "budget": budget, "victim_seed": seed, "attack_seed": attack_seed,
        "selection_policy": args.selection_policy,
        "mean_selected_intermediate_degree": p.get("mean_selected_intermediate_degree"),
        "delta": args.delta, "num_injected_triples": p["num_unique_injected_triples"],
        "num_injected_targets": p["num_targets_with_injections"],
        "rq1_setB_clean_MRR": bv["clean"]["MRR"],
        "rq1_setB_poisoned_MRR": bv["poisoned"]["MRR"],
        "rq1_delta_MRR": bv["delta_poisoned_minus_clean"]["MRR"],
        "rq1_setB_clean_Hits10": bv["clean"]["Hits@10"],
        "rq1_setB_poisoned_Hits10": bv["poisoned"]["Hits@10"],
        "rq1_delta_Hits10": bv["delta_poisoned_minus_clean"]["Hits@10"],
        "rq1_mean_target_rank_improvement": bv["mean_rank_improvement_clean_minus_poisoned"],
        "rq1_num_improved_targets": bv["num_improved_rank"],
        "rq2_setA_clean_MRR": av["clean"]["MRR"],
        "rq2_setA_poisoned_MRR": av["poisoned"]["MRR"],
        "rq2_delta_MRR": av["delta_poisoned_minus_clean"]["MRR"],
        "rq2_setA_clean_Hits10": av["clean"]["Hits@10"],
        "rq2_setA_poisoned_Hits10": av["poisoned"]["Hits@10"],
        "rq2_delta_Hits10": av["delta_poisoned_minus_clean"]["Hits@10"],
        "run_directory": str(evaluation.parent),
    }
    return row, b_rows


RUN_FIELDS = [
    "budget", "victim_seed", "attack_seed", "selection_policy",
    "mean_selected_intermediate_degree", "delta", "num_injected_triples", "num_injected_targets",
    "rq1_setB_clean_MRR", "rq1_setB_poisoned_MRR", "rq1_delta_MRR",
    "rq1_setB_clean_Hits10", "rq1_setB_poisoned_Hits10", "rq1_delta_Hits10",
    "rq1_mean_target_rank_improvement", "rq1_num_improved_targets",
    "rq2_setA_clean_MRR", "rq2_setA_poisoned_MRR", "rq2_delta_MRR",
    "rq2_setA_clean_Hits10", "rq2_setA_poisoned_Hits10", "rq2_delta_Hits10", "run_directory",
]
AGGREGATE_COLUMNS = ["num_injected_triples", "rq1_setB_poisoned_MRR", "rq1_delta_MRR",
                     "rq1_setB_poisoned_Hits10", "rq1_delta_Hits10",
                     "rq1_mean_target_rank_improvement", "rq1_num_improved_targets",
                     "rq2_setA_poisoned_MRR", "rq2_delta_MRR",
                     "rq2_setA_poisoned_Hits10", "rq2_delta_Hits10"]


def summarize_runs(rows: list[dict[str, Any]], budgets: list[int]) -> list[dict[str, Any]]:
    result = []
    for budget in budgets:
        matches = [row for row in rows if row["budget"] == budget]
        record: dict[str, Any] = {"budget": budget, "n_seeds": len(matches)}
        for key in AGGREGATE_COLUMNS:
            samples = [float(row[key]) for row in matches]
            record[f"{key}_mean"] = statistics.mean(samples)
            record[f"{key}_sample_sd"] = statistics.stdev(samples) if len(samples) >= 2 else ""
        result.append(record)
    return result


def paired_budget_summary(rows: list[dict[str, Any]], budgets: list[int]) -> list[dict[str, Any]]:
    if len(budgets) != 2:
        return []
    b_low, b_high = sorted(budgets)
    by_key = {(row["budget"], row["victim_seed"]): row for row in rows}
    samples = []
    seeds = sorted({row["victim_seed"] for row in rows})
    for seed in seeds:
        low = by_key[(b_low, seed)]
        high = by_key[(b_high, seed)]
        samples.append({
            "victim_seed": seed, "lower_budget": b_low, "higher_budget": b_high,
            "rq1_delta_MRR_high_minus_low": high["rq1_delta_MRR"] - low["rq1_delta_MRR"],
            "rq1_delta_Hits10_high_minus_low": high["rq1_delta_Hits10"] - low["rq1_delta_Hits10"],
            "rq1_rank_improvement_high_minus_low": high["rq1_mean_target_rank_improvement"] - low["rq1_mean_target_rank_improvement"],
            "rq2_delta_MRR_high_minus_low": high["rq2_delta_MRR"] - low["rq2_delta_MRR"],
            "rq2_delta_Hits10_high_minus_low": high["rq2_delta_Hits10"] - low["rq2_delta_Hits10"],
        })
    return samples


def summarize_targets(runs: list[tuple[int, int, list[dict[str, str]]]]) -> list[dict[str, Any]]:
    group: dict[tuple[int, str], list[dict[str, str]]] = defaultdict(list)
    for budget, seed, rows in runs:
        for item in rows:
            group[(budget, item["target_id"])].append(item)
    summary = []
    for (budget, target_id), items in sorted(group.items()):
        statuses = {item["poisoning_status"] for item in items}
        clean = [float(item["clean_rank"]) for item in items]
        poisoned = [float(item["poisoned_rank"]) for item in items]
        summary.append({
            "budget": budget, "target_id": target_id, "n_seeds": len(items),
            "poisoning_statuses": ";".join(sorted(statuses)),
            "clean_rank_mean": statistics.mean(clean),
            "poisoned_rank_mean": statistics.mean(poisoned),
            "rank_improvement_mean": statistics.mean(c-p for c,p in zip(clean, poisoned)),
            "poisoned_hits10_fraction": statistics.mean(float(x <= 10) for x in poisoned),
            "num_seeds_improved": sum(c > p for c,p in zip(clean, poisoned)),
        })
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run paired multi-seed MaSS UMLS TransE RQ1/RQ2 experiments")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    parser.add_argument("--budgets", type=int, nargs="+", default=[2, 4])
    parser.add_argument("--attack-mode", choices=("fixed", "vary"), default="fixed",
                        help="fixed: same poison dataset for every victim seed; vary: resample poison per seed")
    parser.add_argument("--attack-seed", type=int, default=42,
                        help="Attack seed used with --attack-mode fixed")
    parser.add_argument("--selection-policy", choices=("baseline", "low_degree"), default="baseline",
                        help="Select original baseline ordering or low-degree priority after random sampling")
    parser.add_argument("--delta", type=float, default=0.7)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--margin", type=float, default=1.0)
    parser.add_argument("--targets", type=Path, default=PILOT_DIR / "setup" / "set_B_target_triples.tsv")
    parser.add_argument("--feasibility-dir", type=Path, default=PILOT_DIR / "feasibility")
    parser.add_argument("--output-root", type=Path, default=PILOT_DIR / "multiseed")
    parser.add_argument("--resume", action="store_true", help="Reuse completed matching stages")
    parser.add_argument("--overwrite", action="store_true", help="Delete ONLY selected run folders before running")
    parser.add_argument("--dry-run", action="store_true", help="Show commands without creating outputs")
    args = parser.parse_args()
    args.targets = args.targets.resolve()
    args.feasibility_dir = args.feasibility_dir.resolve()
    args.output_root = args.output_root.resolve()
    if args.resume and args.overwrite:
        parser.error("Choose --resume OR --overwrite")
    if not args.seeds or len(set(args.seeds)) != len(args.seeds) or min(args.seeds) < 0:
        parser.error("Seeds must be unique non-negative integers")
    if not args.budgets or len(set(args.budgets)) != len(args.budgets) or min(args.budgets) < 1:
        parser.error("Budgets must be unique positive integers")
    if args.attack_seed < 0 or not 0 <= args.delta <= 1:
        parser.error("Attack seed must be nonnegative and delta must be in [0,1]")
    if args.epochs < 1 or args.dim < 1 or args.batch_size < 1 or args.lr <= 0 or args.margin <= 0:
        parser.error("Training hyperparameters must be positive")
    if args.output_root == MASS_DIR or args.output_root == DATASET_DIR or MASS_DIR not in args.output_root.parents:
        # The expected output root stays within the project. Do not allow arbitrary-folder deletion.
        parser.error("--output-root must be a descendant of Mass_implement/ and outside dataset_uml/")
    if DATASET_DIR in args.output_root.parents:
        parser.error("--output-root must not be under dataset_uml/")

    fingerprints = source_fingerprints(args)
    mode_dir = args.output_root / args.selection_policy / (
        f"fixed_attack_seed{args.attack_seed}" if args.attack_mode == "fixed" else "vary_attack_seed"
    )
    plans: list[tuple[int,int,int,Path,Path,Path,Path]] = []
    for budget in args.budgets:
        for seed in args.seeds:
            attack_seed = args.attack_seed if args.attack_mode == "fixed" else seed
            budget_root = mode_dir / f"budget_{budget}"
            run_root = budget_root / f"seed_{seed}"
            poison = (budget_root / "poisoning") if args.attack_mode == "fixed" else (run_root / "poisoning")
            plans.append((budget, seed, attack_seed, run_root, poison, run_root / "victim_models", run_root / "evaluation"))

    print(f"Policy={args.selection_policy} | Mode={args.attack_mode} | seeds={args.seeds} "
          f"| budgets={args.budgets} | delta={args.delta}")
    print(f"Source Set B: {args.targets} (SHA-256: {fingerprints['targets']})")
    if args.dry_run:
        printed_poison = set()
        for b,s,att,root,poison,victims,evaluation in plans:
            if poison not in printed_poison:
                print("\n"+subprocess.list2cmdline(poison_command(args,b,att,poison)))
                printed_poison.add(poison)
            print(subprocess.list2cmdline(train_command(args,s,poison,victims)))
            print(subprocess.list2cmdline(eval_command(args,poison,victims,evaluation)))
        print("\nDry run complete: no output files were written.")
        return

    records: list[dict[str, Any]] = []
    per_target_runs: list[tuple[int, int, list[dict[str,str]]]] = []
    reset_poison = set()
    for index, (budget, seed, attack_seed, run_root, poison, victims, evaluation) in enumerate(plans, 1):
        print(f"\n--- Run {index}/{len(plans)}: budget={budget}, victim_seed={seed}, attack_seed={attack_seed} ---", flush=True)
        if args.overwrite:
            # Only selected run directories and their selected budget's shared poison are removed.
            if run_root.exists():
                shutil.rmtree(run_root)
            if args.attack_mode == "fixed" and poison not in reset_poison:
                if poison.exists():
                    shutil.rmtree(poison)
                reset_poison.add(poison)
        poison_manifest = {"kind": "poison", "budget": budget, "attack_seed": attack_seed,
                           "delta": args.delta, "selection_policy": args.selection_policy,
                           "source_fingerprints": fingerprints}
        ensure_manifest(poison / "multi_seed_manifest.json", poison_manifest,
                        resume=args.resume or (args.attack_mode == "fixed" and index > 1))
        stage(poison_command(args,budget,attack_seed,poison), poison, POISON_FILES,
              resume=args.resume or (args.attack_mode == "fixed" and (poison / "poisoning_summary.json").exists()))
        run_manifest = {"kind": "victim_evaluation", "budget": budget, "victim_seed": seed,
                        "attack_seed": attack_seed, "delta": args.delta, "selection_policy": args.selection_policy,
                        "epochs": args.epochs,
                        "dim": args.dim, "batch_size": args.batch_size, "lr": args.lr,
                        "margin": args.margin, "source_fingerprints": fingerprints,
                        "poison_dataset_sha256": digest(poison / "poisoned_train.txt")}
        ensure_manifest(run_root / "multi_seed_manifest.json", run_manifest, resume=args.resume)
        stage(train_command(args,seed,poison,victims), victims, TRAIN_FILES, resume=args.resume)
        stage(eval_command(args,poison,victims,evaluation), evaluation, EVAL_FILES, resume=args.resume)
        record, ranks = collect_run(args,budget,seed,attack_seed,poison,victims,evaluation)
        records.append(record)
        per_target_runs.append((budget,seed,ranks))
        print(f"[verified] Set B delta MRR={record['rq1_delta_MRR']:+.6f}; "
              f"Set B poisoned Hits@10={record['rq1_setB_poisoned_Hits10']:.3f}; "
              f"Set A delta MRR={record['rq2_delta_MRR']:+.6f}", flush=True)

    summary_dir = mode_dir / "summary"
    run_rows = sorted(records, key=lambda row: (row["budget"],row["victim_seed"]))
    write_csv(summary_dir / "runs.csv", run_rows, RUN_FIELDS)
    aggregate = summarize_runs(run_rows, args.budgets)
    aggregate_fields = ["budget", "n_seeds"] + [f"{field}_{suffix}" for field in AGGREGATE_COLUMNS for suffix in ("mean", "sample_sd")]
    write_csv(summary_dir / "summary_by_budget.csv", aggregate, aggregate_fields)
    paired = paired_budget_summary(run_rows, args.budgets)
    if paired:
        write_csv(summary_dir / "paired_budget_comparison.csv", paired, list(paired[0]))
        paired_stats = {"n_paired_seeds": len(paired),
                        "lower_budget": min(args.budgets), "higher_budget": max(args.budgets),
                        "differences": {}}
        for field in paired[0]:
            if field in ("victim_seed", "lower_budget", "higher_budget"):
                continue
            samples = [float(row[field]) for row in paired]
            paired_stats["differences"][field] = {
                "mean": statistics.mean(samples),
                "sample_sd": statistics.stdev(samples) if len(samples) >= 2 else None,
            }
        save_json(summary_dir / "paired_budget_statistics.json", paired_stats)
    by_target = summarize_targets(per_target_runs)
    if by_target:
        write_csv(summary_dir / "target_rank_summary.csv", by_target, list(by_target[0]))
    save_json(summary_dir / "experiment_summary.json", {
        "purpose": "Paired multi-seed RQ1 (Set B) and preliminary RQ2 (Set A) evaluation",
        "notes": [
            "This experiment measures reliability; multiple seeds do NOT guarantee a better attack.",
            "Same Set A, Set B, clean KG and feasibility inputs for all runs.",
            "Compare policies using matching seeds, budgets, attack seed, and hyperparameters.",
            "fixed attack mode isolates victim-training randomness; vary attack mode also samples path-selection randomness.",
            "Reported sample SD uses n-1 denominator; no claim of statistical significance from 5 seeds.",
            "Stealthiness here means unchanged genuine-fact predictive performance, NOT detection resistance.",
            "TransE-only pilot; not an opaque-box model-agnostic evaluation.",
        ],
        "configuration": {"seeds": args.seeds, "budgets": args.budgets, "attack_mode": args.attack_mode,
                          "fixed_attack_seed": args.attack_seed if args.attack_mode=="fixed" else None,
                          "selection_policy": args.selection_policy,
                          "delta": args.delta, "epochs": args.epochs, "dimension": args.dim,
                          "batch_size": args.batch_size, "lr": args.lr, "margin": args.margin},
        "source_sha256": fingerprints,
        "per_budget": aggregate,
        "num_completed_runs": len(run_rows),
    })
    print("\n=== MULTI-SEED EXPERIMENT COMPLETE ===")
    for item in aggregate:
        print(f"Budget {item['budget']}: n={item['n_seeds']}; "
              f"RQ1 delta MRR={item['rq1_delta_MRR_mean']:+.6f} "
              f"(SD={item['rq1_delta_MRR_sample_sd']}); "
              f"RQ1 Hits@10={item['rq1_setB_poisoned_Hits10_mean']:.3f}; "
              f"RQ2 delta MRR={item['rq2_delta_MRR_mean']:+.6f}")
    print(f"Saved summaries in: {summary_dir}")


if __name__ == "__main__":
    main()
