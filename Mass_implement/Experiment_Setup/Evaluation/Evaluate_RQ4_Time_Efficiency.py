#!/usr/bin/env python3
"""Figure 6 on UMLS: generation time per attempted attack target, in ms.

Models and datasets are loaded once. Every measured target regenerates its
candidates; training, file I/O, and output serialization are outside timers.
"""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import os
import platform
import sys
import time
from dataclasses import dataclass
from pathlib import Path

MASS_DIR = Path(__file__).resolve().parents[2]
SETUP_DIR = MASS_DIR / "Experiment_Setup"
for directory in (MASS_DIR, SETUP_DIR, MASS_DIR / "Path_Generation_Module"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

import numpy as np
import sklearn
import torch
from threadpoolctl import threadpool_info, threadpool_limits

from Entity_Selection import load_ranking_context, rank_paths
from Path_Generation_Module.TransE_Surrogate import read_names
from Path_Generation_Module.Random_Sampling import sample_paths
from construct_attack_baselines import directadd_candidates, comattack_candidates
from construct_poisoned_dataset import read_triples, filter_candidate_paths, parse_candidate_edges
from RQ4_Time_Plot import aggregate_timing_runs, plot_time_efficiency

DATA = MASS_DIR / "dataset_uml"
DEFAULT_EXPERIMENT = MASS_DIR / "outputs" / "Experiments" / "UMLS_feasible_rank11_15_seed42"
METHODS = ("DirectAdd", "ComAttack", "MaSS")


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def triples_hash(triples):
    return hashlib.sha256("".join("\t".join(edge) + "\n" for edge in sorted(triples)).encode()).hexdigest()


@dataclass
class Resources:
    context: object
    targets: list
    clean: set
    held_out: set
    original_known: set
    target_set: set
    entities: list
    relations: list


@dataclass
class GeneratedAttack:
    attributed: set
    selected: list
    candidate_count: int
    sampled_count: int


def load_resources(args):
    context = load_ranking_context(args.checkpoint, pra_method="supervised", threshold=args.threshold)
    targets = read_triples(args.experiment_dir / "setup" / "set_B_target_triples.tsv", require_unique=True)
    clean = set(read_triples(DATA / "train.txt", require_unique=True))
    held_out = set(read_triples(DATA / "valid.txt", require_unique=True)) | set(read_triples(DATA / "test.txt", require_unique=True))
    target_set = set(targets)
    if clean & held_out or target_set & (clean | held_out):
        raise ValueError("Clean/held-out/target split overlap")
    entities, relations = read_names(DATA / "entities.txt"), read_names(DATA / "relations.txt")
    if (context.surrogate.entity_to_id != {name: i for i, name in enumerate(entities)}
            or context.surrogate.relation_to_id != {name: i for i, name in enumerate(relations)}):
        raise ValueError("Checkpoint ID order differs from UMLS dictionaries")
    if context.known_train != clean:
        raise ValueError("MaSS context uses a different clean graph")
    return Resources(context, targets, clean, held_out, clean | held_out, target_set, entities, relations)


def generate_target(method, target, target_index, budget, globally_injected, resources, args):
    """The measured work. All candidate scoring/search is performed afresh."""
    if method == "MaSS":
        payload = rank_paths(
            *target, checkpoint=args.checkpoint, pra_method="supervised", threshold=args.threshold,
            top_k=args.top_k, top_entities=args.top_entities, context=resources.context,
        )
        eligible, _ = filter_candidate_paths(
            payload["ranked_candidate_paths"], resources.clean, resources.held_out, resources.target_set, target,
        )
        sampling = sample_paths({**payload, "ranked_candidate_paths": eligible},
                                delta=args.delta, max_paths=None, seed=args.seed + target_index)
        chosen, attributed = [], set()
        for candidate in sampling["selected_candidate_paths"]:
            required = set(parse_candidate_edges(candidate)) - resources.clean
            additional = required - attributed
            if not additional or len(attributed) + len(additional) > budget:
                continue
            attributed.update(additional)
            chosen.append(candidate)
            if len(attributed) == budget:
                break
        result = GeneratedAttack(attributed, chosen, len(eligible), sampling["num_selected_paths"])
    else:
        common = dict(
            surrogate=resources.context.surrogate, entities=resources.entities, relations=resources.relations,
            original_known=resources.original_known, target_set=resources.target_set,
            globally_injected=globally_injected,
        )
        if method == "DirectAdd":
            candidates, _ = directadd_candidates(target, step=args.promotion_step, **common)
            chosen = candidates[:budget]
            attributed = {item["edge"] for item in chosen}
        elif method == "ComAttack":
            candidates, _ = comattack_candidates(target, **common)
            chosen = candidates[:budget // 2]
            attributed = {item[key] for item in chosen for key in ("first_triple", "second_triple")}
        else:
            raise ValueError(f"Unknown attack: {method}")
        result = GeneratedAttack(attributed, chosen, len(candidates), len(candidates))
    globally_injected.update(result.attributed)
    return result


def validate_generated(method, result, budget, resources):
    """Integrity and audit validation, performed outside measured intervals."""
    if len(result.attributed) > budget:
        raise AssertionError("Per-target injection budget exceeded")
    if result.attributed & (resources.original_known | resources.target_set):
        raise AssertionError("Injection leaked an original fact or target")
    if any(h == t for h, _, t in result.attributed):
        raise AssertionError("Selected injection contains a self-loop")
    if method in ("MaSS", "ComAttack"):
        for candidate in result.selected:
            required = {tuple(candidate[key]) for key in ("first_triple", "second_triple")} - resources.clean
            if not required <= result.attributed:
                raise AssertionError("A composition path was only partially selected")
        if method == "ComAttack" and len(result.attributed) != 2 * len(result.selected):
            raise AssertionError("ComAttack bodies must cost two distinct new triples each")


def source_snapshot(args):
    paths = [DATA / name for name in ("train.txt", "valid.txt", "test.txt", "entities.txt", "relations.txt")]
    paths += [args.checkpoint, args.experiment_dir / "setup" / "set_B_target_triples.tsv",
              DATA / "Semantic_Constraints" / "SRSTRE2", MASS_DIR / "outputs" / "PRA" / "supervised_pra_umls.joblib"]
    paths += [MASS_DIR / name for name in (
        "Path_Generation_Module/Entity_Selection.py", "Path_Generation_Module/TransE_Surrogate.py",
        "Path_Generation_Module/Random_Sampling.py", "PRA_Module/PRA_supervised.py",
        "PRA_Module/Pra_implement_finalVersion.py", "Semantic_Constraints_Module/Semantic_Constraints.py",
        "Experiment_Setup/construct_poisoned_dataset.py", "Experiment_Setup/construct_attack_baselines.py",
        "Experiment_Setup/Evaluation/Evaluate_RQ4_Time_Efficiency.py", "Experiment_Setup/Evaluation/RQ4_Time_Plot.py",
    )]
    return {str(path.resolve()): file_hash(path) for path in paths}


def reference_eligibility(args):
    sweep = args.experiment_dir / "RQ3" / "budget_sweep"
    manifest_path = sweep / "sweep_manifest.json"
    if not manifest_path.is_file() or args.limit is not None:
        return sweep, "skipped: no full matching saved sweep"
    manifest = read_json(manifest_path)
    if (args.seed != manifest["seed"] or args.delta != manifest["delta"]
            or args.promotion_step != manifest["directadd_promotion_step"]
            or (args.threshold, args.top_k, args.top_entities) != (0.9, 10, 10)):
        return sweep, "skipped: attack parameters differ"
    current = {"train": DATA / "train.txt", "valid": DATA / "valid.txt", "test": DATA / "test.txt",
               "targets": args.experiment_dir / "setup" / "set_B_target_triples.tsv",
               "surrogate_checkpoint": args.checkpoint, "ontology": DATA / "Semantic_Constraints" / "SRSTRE2"}
    if any(manifest["sources"][key]["sha256"] != file_hash(path) for key, path in current.items()):
        return sweep, "skipped: attack inputs differ"
    return sweep, "eligible"


def redraw(output):
    manifest = read_json(output / "timing_manifest.json")
    for name, digest in manifest["measured_artifacts_sha256"].items():
        if file_hash(output / name) != digest:
            raise ValueError(f"Measured timing data changed: {name}")
    runs = read_json(output / "timing_runs.json")
    summary = aggregate_timing_runs(runs, manifest["budgets"], METHODS, manifest["repeats"])
    write_csv(output / "timing_summary.csv", summary)
    write_json(output / "timing_summary.json", summary)
    plot_time_efficiency(summary, output)
    return summary


def run(args):
    args.experiment_dir = args.experiment_dir.resolve()
    args.checkpoint = args.checkpoint.resolve()
    output = (args.output_dir or args.experiment_dir / "RQ4" / "time_efficiency").resolve()
    if args.plot_only:
        redraw(output)
        print(f"Replotted saved measured timings: {output}")
        return
    if (len(set(args.budgets)) != len(args.budgets) or any(b < 2 or b % 2 for b in args.budgets)
            or min(args.repeats, args.torch_threads, args.blas_threads, args.top_k, args.top_entities) < 1
            or args.warmup_targets < 0 or args.seed < 0 or not 0 <= args.delta <= 1
            or not 0 <= args.threshold <= 1 or args.promotion_step <= 0):
        raise ValueError("Invalid budgets, repeat count, thread settings, or attack parameters")
    if args.limit is not None and (args.limit < 1 or args.output_dir is None):
        raise ValueError("Limited smoke runs need --limit >= 1 and an explicit --output-dir")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output}; use --plot-only or a new output directory")
    sources = source_snapshot(args)
    torch.set_num_threads(args.torch_threads)
    budgets = sorted(args.budgets)
    samples, runs, order, checks = [], [], [], []
    deterministic_hashes = {}
    reference_dir, reference_status = reference_eligibility(args)
    with threadpool_limits(limits=args.blas_threads, user_api="blas"):
        start = time.perf_counter_ns()
        resources = load_resources(args)
        preload_ms = (time.perf_counter_ns() - start) / 1e6
        targets = resources.targets[:args.limit] if args.limit is not None else resources.targets
        print(f"Loaded {len(resources.targets)} frozen targets; timing {len(targets)}. Resource setup: {preload_ms:.1f} ms", flush=True)
        print("Warm-up: fresh generation for each method (not measured)...", flush=True)
        for method in METHODS:
            injected = set()
            for index, target in enumerate(targets[:args.warmup_targets]):
                result = generate_target(method, target, index, budgets[0], injected, resources, args)
                validate_generated(method, result, budgets[0], resources)
        blas_runtime = threadpool_info()
        for repeat in range(args.repeats):
            ordered_budgets = budgets if repeat % 2 == 0 else list(reversed(budgets))
            for budget_index, budget in enumerate(ordered_budgets):
                rotation = (repeat + budget_index) % len(METHODS)
                method_order = METHODS[rotation:] + METHODS[:rotation]
                for method in method_order:
                    print(f"Repeat {repeat + 1}/{args.repeats}: x={budget}, {method}...", flush=True)
                    order.append({"repeat": repeat, "budget": budget, "method": method})
                    injected, elapsed_total, covered = set(), 0, 0
                    for index, target in enumerate(targets):
                        before = len(injected)
                        started = time.perf_counter_ns()
                        result = generate_target(method, target, index, budget, injected, resources, args)
                        elapsed = time.perf_counter_ns() - started
                        elapsed_total += elapsed
                        # Validation, sample rows, hashing and printing do not
                        # contribute to the elapsed interval above.
                        validate_generated(method, result, budget, resources)
                        covered += bool(result.attributed)
                        samples.append({
                            "repeat": repeat, "budget": budget, "method": method,
                            "target_id": f"target_{index + 1:03d}", "elapsed_ns": elapsed,
                            "elapsed_ms": elapsed / 1e6, "attributed_injected": len(result.attributed),
                            "new_to_global_graph": len(injected) - before,
                            "selected_units": len(result.selected), "candidate_count": result.candidate_count,
                        })
                    digest = triples_hash(injected)
                    key = (budget, method)
                    previous = deterministic_hashes.setdefault(key, digest)
                    if previous != digest:
                        raise AssertionError("Fixed-seed output changed across timing repeats")
                    if repeat == 0 and reference_status == "eligible":
                        folder = "baseline" if method == "MaSS" else method
                        path = reference_dir / f"x_{budget}" / folder / "poisoning" / "injected_triples.tsv"
                        if path.is_file():
                            if set(read_triples(path, require_unique=True)) != injected:
                                raise AssertionError(f"Generated output differs from saved {method} x={budget}")
                            checks.append({"budget": budget, "method": method, "status": "passed"})
                        else:
                            checks.append({"budget": budget, "method": method, "status": "no saved reference"})
                    runs.append({
                        "repeat": repeat, "budget": budget, "method": method,
                        "num_targets": len(targets), "total_elapsed_ns": elapsed_total,
                        "mean_ms_per_target": elapsed_total / len(targets) / 1e6,
                        "targets_with_injections": covered, "unique_injected": len(injected),
                        "injected_set_sha256": digest,
                    })
                    print(f"  {elapsed_total / len(targets) / 1e6:.3f} ms/target; {covered}/{len(targets)} targets with injections", flush=True)
    if any(file_hash(Path(path)) != digest for path, digest in sources.items()):
        raise AssertionError("A benchmark input changed during measurement")
    summary = aggregate_timing_runs(runs, budgets, METHODS, args.repeats)
    output.mkdir(parents=True, exist_ok=True)
    for name, rows in (("timing_samples", samples), ("timing_runs", runs), ("timing_summary", summary)):
        write_csv(output / f"{name}.csv", rows)
        write_json(output / f"{name}.json", rows)
    plot_time_efficiency(summary, output)
    write_json(output / "timing_manifest.json", {
        "dataset": "UMLS", "budgets": budgets, "methods": METHODS, "repeats": args.repeats,
        "seed": args.seed, "checkpoint": str(args.checkpoint),
        "surrogate_dim": resources.context.surrogate.model.entity.weight.shape[1],
        "num_targets_in_source": len(resources.targets), "num_targets_measured": len(targets),
        "attack": {"threshold": args.threshold, "top_k": args.top_k, "top_entities": args.top_entities,
                   "delta": args.delta, "promotion_step": args.promotion_step, "selection_policy": "baseline"},
        "timer": "time.perf_counter_ns; sum of per-target intervals / all attempted targets / 1e6",
        "included": "Fresh candidate generation/scoring/ranking, PRA inference and semantic filtering for MaSS, leakage filtering, sampling, budget selection, global injected-set update.",
        "excluded": "Training; imports/setup; model/data/ontology loading; file output; printing; post-generation integrity checks, audit serialization and degree-only annotations; victim evaluation.",
        "setup_ms_excluded": preload_ms, "warmup_targets_per_method": min(args.warmup_targets, len(targets)),
        "candidate_caching": "None: only trained models, clean data and ontology resources are reused.",
        "method_run_order": order, "reference_check_status": reference_status, "reference_checks": checks,
        "fixed_seed_injected_sets_identical_across_repeats": True, "input_sha256": sources,
        "environment": {"python": sys.version, "torch": torch.__version__, "numpy": np.__version__,
                        "sklearn": sklearn.__version__, "os": platform.platform(), "processor": platform.processor(),
                        "processor_identifier": os.environ.get("PROCESSOR_IDENTIFIER"), "logical_cpus": os.cpu_count(),
                        "torch_threads": torch.get_num_threads(), "torch_interop_threads": torch.get_num_interop_threads(),
                        "blas_threads_requested": args.blas_threads, "threadpool_runtime": blas_runtime,
                        "gc_enabled": gc.isenabled(), "clock_resolution_seconds": time.get_clock_info("perf_counter").resolution},
        "limitations": [
            "Timings measure these local UMLS implementations and machine, not paper hardware or an exact original implementation.",
            "DirectAdd and ComAttack are the project's target-promotion adaptations.",
            "The paper does not explicitly define timer boundaries; this is a declared local generation-only protocol.",
            "Three fixed-seed timing repeats measure runtime variation, not attack-effectiveness variation.",
        ],
        "measured_artifacts_sha256": {name: file_hash(output / name) for name in (
            "timing_samples.csv", "timing_samples.json", "timing_runs.csv", "timing_runs.json")},
    })
    print(f"Figure 6, raw timings and protocol saved to {output}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dir", type=Path, default=DEFAULT_EXPERIMENT)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--checkpoint", type=Path, default=MASS_DIR / "outputs" / "Path_Generation" / "transe_umls.pt")
    parser.add_argument("--budgets", type=int, nargs="+", default=[2, 4, 6, 8, 10, 12])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--torch-threads", type=int, default=4)
    parser.add_argument("--blas-threads", type=int, default=1)
    parser.add_argument("--warmup-targets", type=int, default=3)
    parser.add_argument("--threshold", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--top-entities", type=int, default=10)
    parser.add_argument("--delta", type=float, default=0.7)
    parser.add_argument("--promotion-step", type=float, default=0.1)
    parser.add_argument("--limit", type=int, help="First N targets for a separate smoke run")
    parser.add_argument("--plot-only", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
