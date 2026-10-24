"""10-relation supervised PRA experiment: exclude r* from both path hops.

For each target relation r*, compare the same trained classifier's templates
above the score threshold before and after applying:

    r1 != r* AND r2 != r*

This is an inference-time filtering ablation. It does NOT retrain the model.

From Mass_implement/:
    python PRA_Module/PRA_ExcludeTargetRelation.py --threshold 0.9

From PRA_Module/:
    python PRA_ExcludeTargetRelation.py --threshold 0.9

Output (default):
    Mass_implement/outputs/PRA/pra_exclusion_10_relations.csv
    Mass_implement/outputs/PRA/pra_exclusion_10_relations.json
    Mass_implement/outputs/PRA/pra_exclusion_10_relations.png
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Save charts without requiring a display.
import matplotlib.pyplot as plt

MASS_DIR = Path(__file__).resolve().parents[1]
if str(MASS_DIR) not in sys.path:
    sys.path.insert(0, str(MASS_DIR))

from PRA_Module.PRA_supervised import SupervisedPRA

DEFAULT_MODEL = MASS_DIR / "outputs" / "PRA" / "supervised_pra_umls.joblib"
DEFAULT_OUTPUT_DIR = MASS_DIR / "outputs" / "PRA"
DEFAULT_RELATIONS = [
    "uses",
    "isa",
    "affects",
    "interacts_with",
    "produces",
    "treats",
    "prevents",
    "performs",
    "conceptually_related_to",
    "manages",
]

# Requested 4-column summary: retained_templates is a COUNT in this table.
HEADERS = (
    "r* relation",
    "num_baseline_templates",
    "num_excluded_templates",
    "retained_templates",
)


def serialize_templates(paths: list) -> list[dict]:
    """Convert ((r1, r2), score, traces) records to JSON-friendly objects."""
    return [
        {"r1": r1, "r2": r2, "score": round(float(score), 8)}
        for (r1, r2), score, _ in paths
    ]


def run_experiments(relations: list[str], threshold: float = 0.9,
                    model_path: Path = DEFAULT_MODEL) -> dict:
    """Evaluate 10 target relations using one pretrained SupervisedPRA model."""
    if len(relations) != 10 or len(set(relations)) != 10:
        raise ValueError("Please supply exactly 10 distinct target relations")
    if not 0 <= threshold <= 1:
        raise ValueError("Threshold must lie between 0 and 1")
    if not model_path.is_file():
        raise FileNotFoundError(f"Trained Supervised PRA model not found: {model_path}")

    # Load the classifier just once; this experiment never calls model.fit().
    model = SupervisedPRA(model_path=model_path, threshold=threshold)
    unknown = [r for r in relations if r not in model.relations]
    if unknown:
        raise ValueError(
            f"Relations absent from the trained UMLS model: {unknown}. "
            f"Known relations: {', '.join(sorted(model.relations))}"
        )

    details = []
    for target_relation in relations:
        baseline = model.translate_relation(target_relation)
        retained = model.translate_relation(
            target_relation, exclude_target_relation=True
        )
        excluded = [
            entry for entry in baseline
            if target_relation in entry[0]
        ]

        # Verify that the strict constraint holds and partitions the baseline.
        if any(target_relation in pair for pair, _, _ in retained):
            raise AssertionError(f"Target relation leaked into {target_relation} templates")
        if len(baseline) != len(excluded) + len(retained):
            raise AssertionError(f"Template counts inconsistent for {target_relation}")

        details.append({
            "target_relation": target_relation,
            "num_baseline_templates": len(baseline),
            "num_excluded_templates": len(excluded),
            "num_retained_templates": len(retained),
            "baseline_templates": serialize_templates(baseline),
            "excluded_path_templates": serialize_templates(excluded),
            "retained_path_templates": serialize_templates(retained),
        })

    return {
        "experiment": "supervised_pra_strict_target_relation_exclusion",
        "dataset": "UMLS",
        "threshold": threshold,
        "model_path": str(model_path),
        "target_relation_count": len(relations),
        "exclusion_rule": "r1 != r* AND r2 != r*",
        "model_retrained": False,
        "results": details,
    }


def summary_rows(report: dict) -> list[dict]:
    """Four-column summary with numeric template counts."""
    return [
        {
            "r* relation": item["target_relation"],
            "num_baseline_templates": item["num_baseline_templates"],
            "num_excluded_templates": item["num_excluded_templates"],
            "retained_templates": item["num_retained_templates"],
        }
        for item in report["results"]
    ]


def print_table(rows: list[dict]) -> None:
    widths = {
        col: max(len(col), *(len(str(row[col])) for row in rows))
        for col in HEADERS
    }
    def format_row(row):
        return " | ".join(str(row[col]).ljust(widths[col]) for col in HEADERS)

    print(format_row({col: col for col in HEADERS}))
    print("-+-".join("-" * widths[col] for col in HEADERS))
    for row in rows:
        print(format_row(row))


def save_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=HEADERS)
        writer.writeheader()
        writer.writerows(rows)


def save_bar_graph(rows: list[dict], path: Path, threshold: float) -> None:
    """A stacked bar shows the baseline total split into retained/excluded."""
    labels = [row["r* relation"] for row in rows]
    retained = [row["retained_templates"] for row in rows]
    excluded = [row["num_excluded_templates"] for row in rows]
    baseline = [row["num_baseline_templates"] for row in rows]

    x = list(range(len(rows)))
    fig, ax = plt.subplots(figsize=(14.5, 7.2))
    ax.bar(x, retained, label="Retained (r1 != r*, r2 != r*)")
    ax.bar(x, excluded, bottom=retained, label="Excluded (r1 = r* or r2 = r*)")

    ymax = max(baseline, default=0)
    offset = max(ymax * 0.012, 0.35)
    for i, (num_retained, num_excluded, total) in enumerate(
        zip(retained, excluded, baseline)
    ):
        # The number above each stacked bar is its total baseline size.
        ax.text(i, total + offset, str(total), ha="center", va="bottom", fontsize=9)
        if num_retained:
            ax.text(i, num_retained / 2, str(num_retained),
                    ha="center", va="center", fontsize=9, weight="bold")
        if num_excluded:
            ax.text(i, num_retained + num_excluded / 2, str(num_excluded),
                    ha="center", va="center", fontsize=9)

    ax.set_xticks(x, labels, rotation=35, ha="right")
    ax.set_ylabel("Number of relation-path templates")
    ax.set_xlabel("Target relation r*")
    ax.set_title(
        "Supervised PRA: how many baseline templates are retained?\n"
        f"Target-relation exclusion experiment | score threshold > {threshold}"
    )
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    ax.set_ylim(0, max(1, ymax * 1.17 + 1))
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare target-relation exclusion across 10 UMLS relations"
    )
    parser.add_argument(
        "--relations", nargs="+", default=DEFAULT_RELATIONS,
        help="Exactly 10 distinct target relations (default: 10 built-in UMLS relations)",
    )
    parser.add_argument("--threshold", type=float, default=0.9)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL,
                        help="Trusted trained SupervisedPRA checkpoint")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_DIR / "pra_exclusion_10_relations.json",
                        help="JSON details output path")
    parser.add_argument("--csv", type=Path, default=None,
                        help="Optional CSV path (default: derived from --output)")
    parser.add_argument("--plot", type=Path, default=None,
                        help="Optional PNG graph path (default: derived from --output)")
    args = parser.parse_args()

    try:
        report = run_experiments(args.relations, args.threshold, args.model)
    except (ValueError, FileNotFoundError) as exc:
        parser.error(str(exc))

    rows = summary_rows(report)
    csv_path = args.csv or args.output.with_suffix(".csv")
    plot_path = args.plot or args.output.with_suffix(".png")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    save_csv(rows, csv_path)
    save_bar_graph(rows, plot_path, args.threshold)

    print("\n10-relation strict target-relation exclusion experiment")
    print(f"Rule: {report['exclusion_rule']} | threshold > {args.threshold}\n")
    print_table(rows)
    print(f"\nDetailed JSON: {args.output}")
    print(f"Summary CSV:   {csv_path}")
    print(f"Bar graph PNG: {plot_path}")


if __name__ == "__main__":
    main()
