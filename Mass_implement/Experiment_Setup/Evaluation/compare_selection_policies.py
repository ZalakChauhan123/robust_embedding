#!/usr/bin/env python3
"""Paired baseline-vs-low_degree comparison for MaSS UMLS RQ1/RQ2 runs.

Run from Mass_implement/ after both policy runs finish.
Requires updated run_multi_seed_evaluation.py output schema.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

METRICS = (
    "rq1_setB_poisoned_MRR",
    "rq1_delta_MRR",
    "rq1_setB_poisoned_Hits10",
    "rq1_delta_Hits10",
    "rq1_mean_target_rank_improvement",
    "rq2_setA_poisoned_MRR",
    "rq2_delta_MRR",
    "rq2_setA_poisoned_Hits10",
    "rq2_delta_Hits10",
)
CLEAN_METRICS = (
    "rq1_setB_clean_MRR", "rq1_setB_clean_Hits10",
    "rq2_setA_clean_MRR", "rq2_setA_clean_Hits10",
)


def read_json(p: Path):
    with p.open(encoding="utf-8-sig") as stream:
        return json.load(stream)


def read_csv(p: Path):
    with p.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(p: Path, rows, fields):
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def load_mode(root: Path, policy: str, mode_name: str):
    directory = root / policy / mode_name
    summary = read_json(directory / "summary" / "experiment_summary.json")
    rows = read_csv(directory / "summary" / "runs.csv")
    if summary["configuration"].get("selection_policy") != policy:
        raise ValueError(f"Incorrect policy in {policy} experiment summary")
    if len(rows) != summary["num_completed_runs"]:
        raise ValueError(f"Run count mismatch in {directory}")
    by_key = {}
    for r in rows:
        if r.get("selection_policy") != policy:
            raise ValueError(f"Incorrect policy in runs.csv: {policy}")
        key = (int(r["budget"]), int(r["victim_seed"]))
        if key in by_key:
            raise ValueError(f"Duplicate budget/seed: {key}")
        by_key[key] = r
    return directory, summary, by_key


def main():
    ap = argparse.ArgumentParser(description="Compare matched MaSS selection-policy experiments")
    ap.add_argument("--output-root", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, default=None)
    ap.add_argument("--attack-mode", choices=("fixed", "vary"), default="fixed")
    ap.add_argument("--attack-seed", type=int, default=42)
    args = ap.parse_args()
    mode_name = (f"fixed_attack_seed{args.attack_seed}" if args.attack_mode == "fixed"
                 else "vary_attack_seed")
    root = args.output_root.resolve()
    directory_a, summary_a, baseline = load_mode(root, "baseline", mode_name)
    directory_b, summary_b, degree = load_mode(root, "low_degree", mode_name)
    config_a = dict(summary_a["configuration"])
    config_b = dict(summary_b["configuration"])
    config_a.pop("selection_policy", None)
    config_b.pop("selection_policy", None)
    if config_a != config_b:
        raise ValueError("Policy runs used different experiment configurations")
    if summary_a["source_sha256"] != summary_b["source_sha256"]:
        raise ValueError("Policy runs used different input/code fingerprints")
    if not baseline or set(baseline) != set(degree):
        raise ValueError("Budget/seed pairs do not match between policies")

    paired = []
    warnings = []
    degree_stats = {}
    for key in sorted(baseline):
        budget, seed = key
        a, b = baseline[key], degree[key]
        if int(a["attack_seed"]) != int(b["attack_seed"]):
            raise ValueError(f"Unpaired attack seeds for {key}")
        for metric in CLEAN_METRICS:
            if not math.isclose(float(a[metric]), float(b[metric]),
                                rel_tol=1e-6, abs_tol=1e-7):
                raise ValueError(f"Clean control differs across policies: {key} {metric}")
        if int(a["num_injected_triples"]) != int(b["num_injected_triples"]):
            warnings.append(f"Budget {budget}, seed {seed}: injected counts differ; "
                            "policy effects may be confounded by different injection counts")
        row = {
            "budget": budget, "victim_seed": seed,
            "attack_seed": int(a["attack_seed"]),
            "baseline_injected_triples": int(a["num_injected_triples"]),
            "low_degree_injected_triples": int(b["num_injected_triples"]),
            "baseline_mean_intermediate_degree": a["mean_selected_intermediate_degree"],
            "low_degree_mean_intermediate_degree": b["mean_selected_intermediate_degree"],
        }
        for metric in METRICS:
            row[f"baseline_{metric}"] = float(a[metric])
            row[f"low_degree_{metric}"] = float(b[metric])
            row[f"low_minus_baseline_{metric}"] = float(b[metric]) - float(a[metric])
        paired.append(row)

    # Check that the poisoning construction was held fixed across victim seeds.
    for budget in sorted({key[0] for key in baseline}):
        records = {}
        for policy, directory in (("baseline", directory_a), ("low_degree", directory_b)):
            poisoning = directory / f"budget_{budget}" / "poisoning" / "poisoning_summary.json"
            # In vary mode each seed has separate poison datasets.
            if args.attack_mode == "fixed":
                obj = read_json(poisoning)
                if obj["parameters"]["selection_policy"] != policy:
                    raise ValueError(f"Poisoning policy mismatch for {policy}, budget={budget}")
                for flag in ("injected_train_overlap", "injected_valid_overlap",
                             "injected_test_overlap", "injected_set_B_overlap"):
                    if obj[flag] != 0:
                        raise ValueError(f"Leakage in {poisoning}: {flag}")
                records[policy] = {
                    "injected_triples": obj["num_unique_injected_triples"],
                    "targets_with_injections": obj["num_targets_with_injections"],
                    "sampled_paths": obj["total_sampled_paths"],
                    "mean_intermediate_degree": obj["mean_selected_intermediate_degree"],
                }
        if args.attack_mode == "fixed" and records["baseline"]["sampled_paths"] != records["low_degree"]["sampled_paths"]:
            raise ValueError(f"Random-sampling count unexpectedly differs at budget={budget}")
        degree_stats[str(budget)] = records

    aggregate = []
    for budget in sorted({x["budget"] for x in paired}):
        subset = [x for x in paired if x["budget"] == budget]
        for metric in METRICS:
            values = [x[f"low_minus_baseline_{metric}"] for x in subset]
            aggregate.append({
                "budget": budget, "metric": metric, "num_seeds": len(values),
                "mean_low_minus_baseline": statistics.mean(values),
                "sample_sd_low_minus_baseline": (
                    statistics.stdev(values) if len(values) > 1 else ""
                ),
            })

    output_dir = (args.output_dir or root / "policy_comparison").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "paired_policy_runs.csv", paired, list(paired[0]))
    write_csv(output_dir / "paired_policy_summary.csv", aggregate, list(aggregate[0]))
    report = {
        "comparison": "low_degree minus baseline; matched budgets and victim seeds",
        "configuration": config_a,
        "num_paired_runs": len(paired),
        "poisoning_by_budget": degree_stats,
        "paired_aggregate": aggregate,
        "warnings": list(dict.fromkeys(warnings)),
        "interpretation": (
            "Positive Set B delta MRR/Hits@10 indicates stronger attack effectiveness. "
            "For Set A, delta closer to zero generally indicates less degradation; "
            "higher is not always equivalent to better stealthiness. "
            "These comparisons are descriptive, not significance tests."
        ),
    }
    (output_dir / "policy_comparison.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print("Paired runs:", len(paired))
    for budget in sorted(degree_stats, key=int):
        print(f"Budget {budget} poisoning: {degree_stats[budget]}")
    for item in aggregate:
        print(f"Budget {item['budget']}: {item['metric']} "
              f"low-baseline mean={item['mean_low_minus_baseline']:+.6f}")
    for warning in report["warnings"]:
        print("WARNING:", warning)
    print("Saved comparison to:", output_dir)


if __name__ == "__main__":
    main()
