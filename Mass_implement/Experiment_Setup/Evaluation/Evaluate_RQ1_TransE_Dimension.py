#!/usr/bin/env python3
"""Retrain TransE victims at a new dimension on fixed RQ1 poisoning datasets.

The attacker surrogate, targets and poisoning data remain frozen. Update only
the TransE cells of the final two-header-row RQ1 table after validating all runs.
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import io
import json
import math
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

MASS_DIR = Path(__file__).resolve().parents[2]
SETUP = MASS_DIR / "Experiment_Setup"
sys.path.insert(0, str(SETUP))
import torch
from train_victim_models import run as train_victims
from Evaluate_victim_models import run as evaluate_victims

EXPERIMENT = MASS_DIR / "outputs" / "Experiments" / "UMLS_feasible_rank11_15_seed42"
DATA = MASS_DIR / "dataset_uml"
FINAL_CSV = MASS_DIR / "outputs" / "Final Results" / "RQ-1 -- Attack Effectiveness -- Final Results.csv"
ATTACKS = (("MaSS", "baseline"), ("DirectAdd", "DirectAdd"), ("ComAttack", "ComAttack"))


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_hashes(hashes):
    for name, expected in hashes.items():
        if file_hash(Path(name)) != expected:
            raise ValueError(f"Recorded input or output changed: {name}")


def input_snapshot():
    paths = [DATA / name for name in ("train.txt", "valid.txt", "test.txt", "entities.txt", "relations.txt")]
    paths += [EXPERIMENT / "setup" / name for name in ("set_B_target_triples.tsv", "target_setup_summary.json")]
    paths += [Path(__file__).resolve(), SETUP / "train_victim_models.py",
              Path(__file__).with_name("Evaluate_victim_models.py"),
              MASS_DIR / "Path_Generation_Module" / "TransE_Surrogate.py",
              MASS_DIR / "Path_Generation_Module" / "TransE_Victim.py",
              MASS_DIR / "outputs" / "Path_Generation" / "transe_umls.pt"]
    for _, folder in ATTACKS:
        paths += [EXPERIMENT / folder / "poisoning" / name for name in
                  ("poisoned_train.txt", "injected_triples.tsv", "poisoning_summary.csv", "poisoning_summary.json")]
        paths += [EXPERIMENT / folder / "seed_42" / "victim_models" / name for name in
                  ("training_summary.json", "clean_transe.pt", "poisoned_transe.pt")]
        paths.append(EXPERIMENT / folder / "seed_42" / "evaluation" / "set_B_attack_metrics.json")
    return {str(path): file_hash(path) for path in paths}


def validate_run(point, folder, dimension, expected_clean):
    old = read_json(EXPERIMENT / folder / "seed_42" / "victim_models" / "training_summary.json")
    training = read_json(point / "victim_models" / "training_summary.json")
    expected_parameters = {**old["parameters"], "embedding_dim": dimension}
    if training["parameters"] != expected_parameters:
        raise ValueError("A training parameter other than the requested dimension changed")
    if training["datasets"]["source_sha256"] != old["datasets"]["source_sha256"]:
        raise ValueError("The fixed dataset differs from the original experiment")
    if not training["same_initial_embeddings_verified"]:
        raise ValueError("Clean/poisoned initial embeddings differ")
    for label in ("clean", "poisoned"):
        checkpoint = torch.load(point / "victim_models" / f"{label}_transe.pt", map_location="cpu", weights_only=True)
        if checkpoint["embedding_dim"] != dimension:
            raise ValueError("Incorrect checkpoint dimension")
        state = checkpoint["state_dict"]
        if tuple(state["entity.weight"].shape) != (135, dimension) or tuple(state["relation.weight"].shape) != (46, dimension):
            raise ValueError("Incorrect embedding tensor shapes")
        if label == "clean":
            if expected_clean is not None and any(not torch.equal(state[k], expected_clean[k]) for k in state):
                raise ValueError("Clean model weights differ between attack conditions")
            clean_state = state
    with (point / "evaluation" / "set_B_target_rank_comparison.csv").open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    with (EXPERIMENT / "setup" / "set_B_target_triples.tsv").open(encoding="utf-8-sig", newline="") as stream:
        targets = list(csv.reader(stream, delimiter="\t"))
    if len(rows) != 100 or [[r["head"], r["relation"], r["tail"]] for r in rows] != targets:
        raise ValueError("Evaluation target identity or denominator changed")
    metrics = read_json(point / "evaluation" / "set_B_attack_metrics.json")["set_B_all_targets"]
    for label in ("clean", "poisoned"):
        ranks = [float(r[f"{label}_rank"]) for r in rows]
        for key, actual in (("MRR", sum(1 / r for r in ranks) / 100),
                            ("Hits@10", sum(r <= 10 for r in ranks) / 100)):
            if not math.isclose(actual, metrics[label][key], rel_tol=1e-12, abs_tol=1e-12):
                raise ValueError("Metrics disagree with saved per-target ranks")
    return metrics, clean_state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.dim < 1 or args.threads < 1:
        raise ValueError("Dimension and thread count must be positive")
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    root = EXPERIMENT / "RQ1" / f"transe_dim{args.dim}"
    root.mkdir(parents=True, exist_ok=True)
    config_path = root / "run_config.json"
    config = {"dataset": "UMLS", "victim": "TransE", "dimension": args.dim,
              "torch_threads": args.threads, "torch_version": str(torch.__version__),
              "scope": "Only victim embedding dimension changes; frozen original targets, attacker surrogate and poisoning datasets.",
              "target_selection": "Targets originally qualified using the 64-dimensional clean TransE victim ranks 11-15.",
              "input_sha256": input_snapshot()}
    if config_path.exists():
        if not args.resume or read_json(config_path) != config:
            raise ValueError("Saved run exists; use --resume with unchanged inputs and configuration")
    else:
        write_json(config_path, config)
    backup = root / "final_results_before.csv"
    if not backup.exists():
        backup.write_bytes(FINAL_CSV.read_bytes())
    original_csv = backup.read_bytes()
    clean_state = None
    results = []
    for attack, folder in ATTACKS:
        verify_hashes(config["input_sha256"])
        point = root / folder / "seed_42"
        point.mkdir(parents=True, exist_ok=True)
        manifest = point / "stage_manifest.json"
        if manifest.exists():
            verify_hashes(read_json(manifest)["artifact_sha256"])
            print(f"{attack}: verified saved training/evaluation artifacts", flush=True)
        else:
            poison = EXPERIMENT / folder / "poisoning"
            old = read_json(EXPERIMENT / folder / "seed_42" / "victim_models" / "training_summary.json")["parameters"]
            train_args = argparse.Namespace(
                clean_train=DATA / "train.txt", poisoned_train=poison / "poisoned_train.txt",
                injected=poison / "injected_triples.tsv", valid=DATA / "valid.txt", test=DATA / "test.txt",
                targets=EXPERIMENT / "setup" / "set_B_target_triples.tsv",
                entities=DATA / "entities.txt", relations=DATA / "relations.txt",
                output_dir=point / "victim_models", attack_name=attack, epochs=old["epochs"],
                dim=args.dim, batch_size=old["batch_size"], seed=old["random_seed"],
                lr=old["learning_rate"], margin=old["margin"], log_every=50, overwrite=False)
            eval_args = argparse.Namespace(
                clean_checkpoint=train_args.output_dir / "clean_transe.pt",
                poisoned_checkpoint=train_args.output_dir / "poisoned_transe.pt",
                train=train_args.clean_train, poisoned_train=train_args.poisoned_train,
                injected=train_args.injected, valid=train_args.valid, test=train_args.test,
                targets=train_args.targets, poisoning_summary=poison / "poisoning_summary.csv",
                training_summary=train_args.output_dir / "training_summary.json",
                output_dir=point / "evaluation", attack_name=attack, overwrite=False)
            print(f"{attack}: training paired {args.dim}-dimensional TransE victims (500 epochs each)...", flush=True)
            start = time.perf_counter()
            with (point / "run.log").open("w", encoding="utf-8") as log:
                with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                    train_victims(train_args)
                    evaluate_victims(eval_args)
            artifacts = [p for directory in (train_args.output_dir, eval_args.output_dir) for p in directory.iterdir() if p.is_file()]
            write_json(manifest, {"elapsed_seconds": time.perf_counter() - start,
                                 "artifact_sha256": {str(p): file_hash(p) for p in artifacts}})
        metrics, clean_state = validate_run(point, folder, args.dim, clean_state)
        previous = read_json(EXPERIMENT / folder / "seed_42" / "evaluation" / "set_B_attack_metrics.json")["set_B_all_targets"]
        results.append({"attack": attack, "embedding_dim": args.dim,
                        "clean": metrics["clean"], "poisoned": metrics["poisoned"],
                        "previous_64_dimension_poisoned": previous["poisoned"],
                        "evaluation_directory": str(point / "evaluation"),
                        "training_summary": str(point / "victim_models" / "training_summary.json")})
        print(f"{attack}: Hit@10={metrics['poisoned']['Hits@10']:.4f}, MRR={metrics['poisoned']['MRR']:.8f}; clean Hit@10={metrics['clean']['Hits@10']:.4f}, MRR={metrics['clean']['MRR']:.8f}", flush=True)
    verify_hashes(config["input_sha256"])
    rows = list(csv.reader(io.StringIO(original_csv.decode("utf-8-sig"))))
    if rows[0][2:4] != ["TransE", "TransE"] or rows[1][2:4] != ["Hit@10", "MRR"]:
        raise ValueError("Unexpected final CSV layout")
    replacements = {r["attack"]: [f"{r['poisoned']['Hits@10']:.2f}", f"{r['poisoned']['MRR']:.2f}"] for r in results}
    found = []
    for row in rows[2:]:
        if row[:1] == ["UMLS"] and row[1] in replacements:
            row[2:4] = replacements[row[1]]
            found.append(row[1])
    if sorted(found) != sorted(replacements):
        raise ValueError("Expected exactly one final CSV row for each attack")
    buffer = io.StringIO(newline="")
    csv.writer(buffer, lineterminator="\r\n").writerows(rows)
    new_csv = (b"\xef\xbb\xbf" if original_csv.startswith(b"\xef\xbb\xbf") else b"") + buffer.getvalue().encode("utf-8")
    if FINAL_CSV.read_bytes() not in (original_csv, new_csv):
        raise ValueError("Final CSV changed during training; refusing to replace concurrent edits")
    pending = FINAL_CSV.with_suffix(".csv.tmp")
    pending.write_bytes(new_csv)
    os.replace(pending, FINAL_CSV)
    (root / "final_results_after.csv").write_bytes(new_csv)
    write_json(root / "results_summary.json", {
        "completed_utc": datetime.now(timezone.utc).isoformat(), "configuration": str(config_path),
        "results": results, "final_csv": str(FINAL_CSV),
        "final_csv_before_sha256": hashlib.sha256(original_csv).hexdigest(),
        "final_csv_after_sha256": hashlib.sha256(new_csv).hexdigest(),
        "csv_presentation": "Two decimal places, matching the existing final table; full precision in this summary.",
        "checks": {"all_metrics_verified_from_100_target_ranks": True,
                   "all_original_sources_and_64_dimension_checkpoints_unchanged": True,
                   "clean_weights_identical_across_attacks": True,
                   "paired_initial_embeddings_identical": True},
        "note": "Existing other-model columns retain their previous settings. RQ2 final table is not updated by this RQ1 experiment."})
    print(f"Updated: {FINAL_CSV}", flush=True)
    print(f"Full-precision results and provenance: {root}", flush=True)


if __name__ == "__main__":
    main()
