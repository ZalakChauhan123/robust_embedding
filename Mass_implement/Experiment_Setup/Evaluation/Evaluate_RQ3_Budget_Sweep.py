#!/usr/bin/env python3
"""Generate UMLS attack budgets and plot RQ3 type filtering as in Figure 5.

Run from Mass_implement:
    python Experiment_Setup/Evaluation/Evaluate_RQ3_Budget_Sweep.py

Uses the existing target-promotion attack adaptations and cached MaSS candidate
pool. Each budget is generated independently from the same clean graph. No
victim training is needed for type-constraint detection.
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import json
import sys
from pathlib import Path

MASS_DIR = Path(__file__).resolve().parents[2]
SETUP_DIR = MASS_DIR / "Experiment_Setup"
if str(SETUP_DIR) not in sys.path:
    sys.path.insert(0, str(SETUP_DIR))

from construct_attack_baselines import construct as construct_baseline
from construct_poisoned_dataset import construct as construct_mass, read_triples
from Evaluate_RQ3 import METHODS, evaluate, write_csv

LABELS = {"baseline": "MaSS", "DirectAdd": "DirectAdd", "ComAttack": "ComAttack"}
DEFAULT_EXPERIMENT = MASS_DIR / "outputs" / "Experiments" / "UMLS_feasible_rank11_15_seed42"


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, payload):
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_generation(report, *, budget, method, args, sources):
    """Also check configuration when --evaluate-only reuses generated files."""
    parameters = report["parameters"]
    expected = {"budget_distinct_new_triples_per_target": budget}
    if method == "baseline":
        expected.update(seed=args.seed, delta=args.delta, selection_policy="baseline")
    else:
        expected.update(attack_seed=args.seed, directadd_promotion_step=args.promotion_step)
    for key, value in expected.items():
        if parameters.get(key) != value:
            raise ValueError(f"x={budget} {method}: saved {key} differs from requested {value}")
    for key in ("targets", "train", "valid", "test"):
        if report["source_sha256"].get(key) != sources[key]["sha256"]:
            raise ValueError(f"x={budget} {method}: source hash differs for {key}")
    key = "feasibility_summary" if method == "baseline" else "surrogate_checkpoint"
    if report["source_sha256"].get(key) != sources[key]["sha256"]:
        raise ValueError(f"x={budget} {method}: source hash differs for {key}")
    if any(row["injected_for_target"] > budget for row in report["targets"]):
        raise ValueError(f"x={budget} {method}: per-target budget exceeded")
    for key in ("injected_train_overlap", "injected_valid_overlap",
                "injected_test_overlap", "injected_set_B_overlap"):
        if report[key]:
            raise ValueError(f"x={budget} {method}: {key} is nonzero")


def compare_original_x2(experiment: Path, budget_dir: Path, method: str, report):
    reference_dir = experiment / method / "poisoning"
    reference_summary = reference_dir / "poisoning_summary.json"
    if not reference_summary.is_file():
        return {"method": LABELS[method], "status": "no original x=2 output"}
    reference = read_json(reference_summary)
    parameter_keys = ("budget_distinct_new_triples_per_target", "seed", "delta", "selection_policy") \
        if method == "baseline" else \
        ("budget_distinct_new_triples_per_target", "attack_seed", "directadd_promotion_step")
    if (any(reference["parameters"].get(k) != report["parameters"].get(k) for k in parameter_keys)
            or reference["source_sha256"] != report["source_sha256"]):
        return {"method": LABELS[method], "status": "skipped: original configuration differs"}
    original = set(read_triples(reference_dir / "injected_triples.tsv", require_unique=True))
    regenerated = set(read_triples(budget_dir / method / "poisoning" / "injected_triples.tsv",
                                    require_unique=True))
    if original != regenerated:
        raise AssertionError(f"{method}: regenerated x=2 differs from original injected triples")
    return {"method": LABELS[method], "status": "passed", "identical_injected_triples": len(original)}


def plot_results(rows, budgets, output: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FormatStrFormatter

    colors = {"MaSS": "#d65a4a", "DirectAdd": "#438db4", "ComAttack": "#9890c6"}
    fig, ax = plt.subplots(figsize=(8.4, 4.5), layout="constrained")
    width = 0.24
    for index, label in enumerate(LABELS.values()):
        by_budget = {row["budget"]: row for row in rows if row["attack"] == label}
        ax.bar([i + (index - 1) * width for i in range(len(budgets))],
               [by_budget[b]["type_remaining_ratio"] for b in budgets],
               width=width, color=colors[label], label=label, edgecolor="white", linewidth=0.5)
    ax.set_xticks(range(len(budgets)), [f"x={b}" for b in budgets])
    ax.set_xlabel("Injection budget per target (new triplets)")
    ax.set_ylabel("Remaining ratio")
    ax.set_ylim(0.0, 1.0)
    ax.set_yticks([i / 5 for i in range(6)])
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    ax.set_title("UMLS: type-constraint filtering (RQ3)", pad=35)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncols=3, frameon=False)
    ax.set_axisbelow(True)
    ax.grid(axis="y", alpha=0.2)
    ax.spines[["top", "right"]].set_visible(False)
    for extension in ("png", "pdf"):
        fig.savefig(output / f"figure5_umls_type_constraints.{extension}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def run(args):
    if not args.budgets or any(b < 2 or b % 2 for b in args.budgets):
        raise ValueError("Budgets must be positive even integers >= 2 for complete ComAttack bodies")
    if len(args.budgets) != len(set(args.budgets)):
        raise ValueError("Budgets must not repeat")
    if not 0 <= args.delta <= 1 or args.promotion_step <= 0:
        raise ValueError("delta must be in [0,1] and promotion-step must be positive")
    budgets = sorted(args.budgets)
    experiment = args.experiment_dir.resolve()
    output = (args.output_dir or experiment / "RQ3" / "budget_sweep").resolve()
    targets = experiment / "setup" / "set_B_target_triples.tsv"
    feasibility = experiment / "feasibility"
    data = MASS_DIR / "dataset_uml"
    ontology = data / "Semantic_Constraints" / "SRSTRE2"
    source_paths = {
        "targets": targets, "train": data / "train.txt", "valid": data / "valid.txt",
        "test": data / "test.txt", "ontology": ontology,
        "feasibility_summary": feasibility / "feasibility_summary.json",
        "surrogate_checkpoint": args.checkpoint.resolve(),
    }
    sources = {key: {"path": str(path), "sha256": file_hash(path)}
               for key, path in source_paths.items()}
    # Record the whole cached candidate pool, not just its summary.
    candidate_hashes = {path.name: file_hash(path) for path in sorted(feasibility.glob("target_*.json"))}
    previous_manifest = None
    if args.evaluate_only:
        previous_manifest = read_json(output / "sweep_manifest.json")
        if (previous_manifest["sources"] != sources
                or previous_manifest["cached_mass_candidate_sha256"] != candidate_hashes):
            raise ValueError("Inputs differ from the saved generation manifest; regenerate the sweep")
        for budget in budgets:
            for method in METHODS:
                relative = f"x_{budget}/{method}/poisoning/injected_triples.tsv"
                if previous_manifest["injected_sha256"].get(relative) != file_hash(output / relative):
                    raise ValueError(f"Saved injection file changed or was not recorded: {relative}")
    output.mkdir(parents=True, exist_ok=True)
    results = []
    regression_checks = []
    for budget in budgets:
        budget_dir = output / f"x_{budget}"
        budget_dir.mkdir(parents=True, exist_ok=True)
        reports = {}
        if not args.evaluate_only:
            with (budget_dir / "generation.log").open("w", encoding="utf-8") as log:
                for method in METHODS:
                    print(f"x={budget}: generating {LABELS[method]}...", flush=True)
                    with contextlib.redirect_stdout(log):
                        if method == "baseline":
                            report = construct_mass(
                                targets_file=targets, feasibility_dir=feasibility,
                                output_dir=budget_dir / method / "poisoning",
                                delta=args.delta, budget=budget, seed=args.seed, selection_policy="baseline",
                            )
                        else:
                            report = construct_baseline(
                                method, targets_file=targets,
                                output_dir=budget_dir / method / "poisoning",
                                surrogate_checkpoint=args.checkpoint, budget=budget,
                                seed=args.seed, directadd_step=args.promotion_step,
                            )
                    reports[method] = report
        for method in METHODS:
            report = reports.get(method) or read_json(budget_dir / method / "poisoning" / "poisoning_summary.json")
            check_generation(report, budget=budget, method=method, args=args, sources=sources)
            reports[method] = report
            if budget == 2:
                regression_checks.append(compare_original_x2(experiment, budget_dir, method, report))

        # Reuse the existing detector; chart only its TYPE results, not mutual
        # exclusion or their union. Separate original RQ3 outputs are untouched.
        summaries = evaluate(budget_dir, ontology=ontology)
        for summary in summaries:
            method = summary["method"]
            report = reports[method]
            n = summary["total_injected"]
            if summary["type_unknown"]:
                raise ValueError(f"x={budget} {method}: unknown relation types; do not plot as confirmed passes")
            if n != report["num_unique_injected_triples"]:
                raise ValueError(f"x={budget} {method}: injected count differs from construction summary")
            detected = summary["type_detected"]
            target_counts = [row["injected_for_target"] for row in report["targets"]]
            results.append({
                "budget": budget, "attack": LABELS[method], "method": method,
                "total_injected": n, "type_detected": detected,
                "type_remaining": n - detected, "type_unknown": summary["type_unknown"],
                "type_remaining_ratio": (n - detected) / n,
                "num_targets": len(target_counts),
                "targets_at_budget": sum(count == budget for count in target_counts),
                "targets_below_budget": sum(count < budget for count in target_counts),
                "max_injected_for_target": max(target_counts),
            })
            print(f"  {LABELS[method]}: {n - detected}/{n} remain ({(n - detected) / n:.4f})", flush=True)

    write_csv(output / "type_constraint_budget_summary.csv", results, list(results[0]))
    write_json(output / "type_constraint_budget_summary.json", results)
    plot_results(results, budgets, output)
    # Check that generation did not alter any inputs, including old candidate files.
    if (any(file_hash(path) != sources[key]["sha256"] for key, path in source_paths.items())
            or any(file_hash(feasibility / name) != value for name, value in candidate_hashes.items())):
        raise AssertionError("An experiment input changed during the sweep")
    write_json(output / "sweep_manifest.json", {
        "dataset": "UMLS", "experiment_dir": str(experiment), "budgets": budgets,
        "seed": args.seed, "delta": args.delta, "directadd_promotion_step": args.promotion_step,
        "selection_policy": "baseline", "sources": sources,
        "cached_mass_candidate_sha256": candidate_hashes,
        "injected_sha256": {
            f"x_{budget}/{method}/poisoning/injected_triples.tsv":
                file_hash(output / f"x_{budget}" / method / "poisoning" / "injected_triples.tsv")
            for budget in budgets for method in METHODS
        },
        "remaining_ratio": "(unique injected triplets - type-detected triplets) / unique injected triplets",
        "original_x2_regression_checks": regression_checks,
        "notes": [
            "Detection only; no victim retraining. Chart excludes mutual-exclusion detection.",
            "Budgets are per-target maxima; actual unique injection counts are reported.",
            "MaSS uses the existing cached candidate pool and fixed random acceptance draws.",
            "DirectAdd and ComAttack are the project's target-promotion adaptations.",
            "ComAttack selects budget/2 complete bodies from its first viable relation pair.",
            "Each budget is regenerated from clean inputs; larger-budget datasets need not nest.",
        ],
    })
    print(f"Saved chart, counts, and manifest to {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dir", type=Path, default=DEFAULT_EXPERIMENT)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--budgets", type=int, nargs="+", default=[2, 4, 6, 8, 10, 12])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--delta", type=float, default=0.7)
    parser.add_argument("--checkpoint", type=Path, default=MASS_DIR / "outputs" / "Path_Generation" / "transe_umls.pt")
    parser.add_argument("--promotion-step", type=float, default=0.1)
    parser.add_argument("--evaluate-only", action="store_true", help="Reuse generated budget directories; rerun detection and plot")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
