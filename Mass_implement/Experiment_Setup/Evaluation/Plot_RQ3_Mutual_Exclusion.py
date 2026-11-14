#!/usr/bin/env python3
"""Plot UMLS mutual-exclusion filtering from the saved RQ3 budget sweep.

Run from Mass_implement:
    python Experiment_Setup/Evaluation/Plot_RQ3_Mutual_Exclusion.py

The Figure-7-style chart uses one explicitly selected budget (default x=2).
A second grouped chart shows all budgets. Filter ratio means the number of
strict clean-training overlap flags divided by unique injected triples.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from Evaluate_RQ3 import MASS_DIR, METHODS, build_clean_indexes, read_triples, write_csv

DEFAULT_SWEEP = (MASS_DIR / "outputs" / "Experiments" / "UMLS_feasible_rank11_15_seed42"
                 / "RQ3" / "budget_sweep")
LABELS = {"baseline": "MaSS", "DirectAdd": "DirectAdd", "ComAttack": "ComAttack"}


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def collect_results(sweep, budgets, train):
    clean = read_triples(train, require_unique=True)
    clean_set = set(clean)
    hr, rt = build_clean_indexes(clean)
    inputs = {str(train): file_hash(train)}
    manifest_path = sweep / "sweep_manifest.json"
    manifest = read_json(manifest_path) if manifest_path.is_file() else None
    if manifest:
        inputs[str(manifest_path)] = file_hash(manifest_path)
        if manifest["sources"]["train"]["sha256"] != inputs[str(train)]:
            raise ValueError("Clean training data differs from the saved generation manifest")
    results = []
    for budget in budgets:
        summary_path = sweep / f"x_{budget}" / "RQ3" / "rq3_summary.json"
        inputs[str(summary_path)] = file_hash(summary_path)
        summaries = read_json(summary_path)
        if len(summaries) != len(METHODS) or {row["method"] for row in summaries} != set(METHODS):
            raise ValueError(f"x={budget}: expected one summary for each of the three attacks")
        by_method = {row["method"]: row for row in summaries}
        for method in METHODS:
            relative = f"x_{budget}/{method}/poisoning/injected_triples.tsv"
            injected_path = sweep / relative
            inputs[str(injected_path)] = file_hash(injected_path)
            if manifest and manifest["injected_sha256"].get(relative) != inputs[str(injected_path)]:
                raise ValueError(f"Injected data differs from the saved generation manifest: {relative}")
            injected = read_triples(injected_path, require_unique=True)
            if clean_set.intersection(injected):
                raise ValueError(f"x={budget} {method}: injected triples overlap clean training data")
            forward_count = inverse_count = both_count = 0
            # Recompute from the injected graph to reject stale saved counts.
            # OR counts each triplet once even if both overlap directions fire.
            for h, r, t in injected:
                forward = bool(hr.get((h, r), set()) - {t})
                inverse = bool(rt.get((r, t), set()) - {h})
                forward_count += forward
                inverse_count += inverse
                both_count += forward and inverse
            filtered = forward_count + inverse_count - both_count
            n = len(injected)
            saved = by_method[method]
            if saved["total_injected"] != n or saved["mutual_detected"] != filtered:
                raise ValueError(f"x={budget} {method}: saved RQ3 counts differ from recomputed overlap flags")
            results.append({
                "budget": budget, "attack": LABELS[method], "method": method,
                "total_injected": n, "mutual_detected": filtered,
                "filter_ratio": filtered / n,
                "mutual_remaining": n - filtered, "mutual_remaining_ratio": (n - filtered) / n,
                "same_head_relation_other_tail": forward_count,
                "same_relation_tail_other_head": inverse_count, "both_directions": both_count,
            })
    return results, inputs


def plot_results(rows, budgets, selected_budget, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FormatStrFormatter

    def format_axes(ax, title):
        ax.set_ylim(0, 1)
        ax.set_yticks([i / 5 for i in range(6)])
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
        ax.set_ylabel("Filter ratio (flagged / injected)")
        ax.set_title(title, pad=15)
        ax.grid(axis="y", alpha=0.2)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)

    def save(fig, name):
        for extension in ("png", "pdf"):
            fig.savefig(output / f"{name}.{extension}", dpi=300, bbox_inches="tight")
        plt.close(fig)

    selected = [row for row in rows if row["budget"] == selected_budget]
    fig, ax = plt.subplots(figsize=(6.4, 4.5), layout="constrained")
    bars = ax.bar([row["attack"] for row in selected], [row["filter_ratio"] for row in selected],
                  width=0.58, color="#168f8b", edgecolor="white", hatch="//", label="UMLS")
    ax.bar_label(bars, labels=[f"{row['filter_ratio']:.2%}\n({row['mutual_detected']}/{row['total_injected']})"
                              for row in selected], label_type="center", color="white", fontweight="bold")
    format_axes(ax, f"UMLS: mutual-exclusion filtering (x={selected_budget})")
    ax.set_xlabel("Attack")
    ax.legend(loc="upper right", frameon=False)
    fig.supxlabel("Strict overlap with clean training facts", fontsize=9)
    save(fig, f"figure7_umls_x{selected_budget}")

    fig, ax = plt.subplots(figsize=(8.4, 4.5), layout="constrained")
    width = 0.24
    for index, (label, color) in enumerate(zip(LABELS.values(), ("#d65a4a", "#438db4", "#9890c6"))):
        by_budget = {row["budget"]: row["filter_ratio"] for row in rows if row["attack"] == label}
        ax.bar([i + (index - 1) * width for i in range(len(budgets))],
               [by_budget[b] for b in budgets], width=width, color=color,
               label=label, edgecolor="white", linewidth=0.5)
    format_axes(ax, "UMLS: mutual-exclusion filtering by budget")
    ax.set_xticks(range(len(budgets)), [f"x={b}" for b in budgets])
    ax.set_xlabel("Injection budget per target (new triplets)")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncols=3, frameon=False)
    ax.set_title("UMLS: mutual-exclusion filtering by budget", pad=35)
    save(fig, "figure7_umls_budget_sweep")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep-dir", type=Path, default=DEFAULT_SWEEP)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--budgets", type=int, nargs="+", default=[2, 4, 6, 8, 10, 12])
    parser.add_argument("--budget", type=int, default=2, help="Budget for the Figure-7-style attack comparison")
    parser.add_argument("--train", type=Path, default=MASS_DIR / "dataset_uml" / "train.txt")
    args = parser.parse_args()
    if (not args.budgets or any(b < 1 for b in args.budgets)
            or len(args.budgets) != len(set(args.budgets)) or args.budget not in args.budgets):
        parser.error("Budgets must be distinct positive integers and include --budget")
    sweep = args.sweep_dir.resolve()
    output = (args.output_dir or sweep / "mutual_exclusion").resolve()
    budgets = sorted(args.budgets)
    rows, inputs = collect_results(sweep, budgets, args.train.resolve())
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "mutual_exclusion_budget_summary.csv", rows, list(rows[0]))
    (output / "mutual_exclusion_budget_summary.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    plot_results(rows, budgets, args.budget, output)
    if any(file_hash(Path(path)) != digest for path, digest in inputs.items()):
        raise AssertionError("An evaluation input changed while plotting")
    (output / "figure7_manifest.json").write_text(json.dumps({
        "dataset": "UMLS", "budgets": budgets, "main_chart_budget": args.budget,
        "filter_ratio": "unique injected triplets flagged by strict overlap / unique injected triplets",
        "rule": "same head and relation with another tail OR same relation and tail with another head in CLEAN TRAIN",
        "input_sha256": inputs, "all_saved_counts_verified": True,
        "notes": [
            "Type detection and combined detection are excluded from these plots.",
            "Overlap flags are not confirmed contradictions for multivalued relations.",
            "The paper's Figure 7 does not explicitly specify a denominator or injection budget.",
            "The UMLS chart defines both explicitly and uses a 0-1 scale to show the full measured ratios.",
        ],
    }, indent=2), encoding="utf-8")
    for row in rows:
        print(f"x={row['budget']:2d} {row['attack']:9s}: {row['mutual_detected']}/{row['total_injected']} "
              f"flagged; filter ratio={row['filter_ratio']:.6f}")
    print(f"Charts and verified counts saved to {output}")


if __name__ == "__main__":
    main()
