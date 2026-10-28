"""
-- PURPOSE : Check whether each UMLS Set B target admits MaSS candidate poisoning paths.

Run from Mass_implement/:
    python Experiment_Setup/target_feasibility_check.py

Outputs (default: outputs/Experiments/UMLS_pilot/feasibility/):
    target_001.json ... target_NNN.json  - full entity-selection results
    feasibility_summary.csv               - one row per Set B target
    feasibility_summary.json              - configuration and aggregated results
    feasible_targets.tsv                  - original-order subset with usable paths
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

MASS_DIR = Path(__file__).resolve().parents[1]
PATH_GENERATION_DIR = MASS_DIR / "Path_Generation_Module"
if str(PATH_GENERATION_DIR) not in sys.path:
    sys.path.insert(0, str(PATH_GENERATION_DIR))

from Entity_Selection import rank_paths

DEFAULT_TARGETS = (
    MASS_DIR / "outputs" / "Experiments" / "UMLS_pilot" / "setup"
    / "set_B_target_triples.tsv"
)
DEFAULT_TARGET_METADATA = DEFAULT_TARGETS.parent / "target_setup_summary.json"
DEFAULT_OUTPUT_DIR = (
    MASS_DIR / "outputs" / "Experiments" / "UMLS_pilot" / "feasibility"
)
DEFAULT_CHECKPOINT = MASS_DIR / "outputs" / "Path_Generation" / "transe_umls.pt"
DEFAULT_SUPERVISED_PRA = MASS_DIR / "outputs" / "PRA" / "supervised_pra_umls.joblib"

CSV_COLUMNS = [
    "target_id", "head", "relation", "tail", "clean_filtered_tail_rank",
    "pra_method", "num_pra_templates", "num_semantically_valid_templates",
    "num_selected_templates", "num_ranked_paths", "one_new_triple_paths",
    "two_new_triple_paths", "distinct_new_triples_available",
    "best_lower_bound_score", "status", "reason", "result_json", "error",
]


def read_tsv_triples(path: Path) -> list[tuple[str, str, str]]:
    """Read a headerless, tab-delimited triple file; reject malformed/duplicates."""
    if not path.is_file():
        raise FileNotFoundError(f"Missing triple file: {path}")
    triples = []
    seen = set()
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for line_no, row in enumerate(csv.reader(handle, delimiter="\t"), 1):
            if not row or (len(row) == 1 and not row[0].strip()):
                continue
            if row[0].lstrip().startswith("#"):
                continue
            values = tuple(cell.strip() for cell in row)
            if len(values) != 3 or not all(values):
                raise ValueError(
                    f"Expected 3 nonempty TAB-separated fields at {path}:{line_no}; "
                    f"got {row!r}"
                )
            if values in seen:
                raise ValueError(f"Duplicate triple at {path}:{line_no}: {values!r}")
            seen.add(values)
            triples.append(values)
    if not triples:
        raise ValueError(f"No target triples found in {path}")
    return triples


def read_setup_ranks(path: Path) -> dict[tuple[str, str, str], float]:
    """Optional mapping of fixed Set B targets to their initial clean tail ranks."""
    if not path.is_file():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    result = {}
    for entry in data.get("set_B_rows", []):
        target = entry.get("target_triple")
        rank = entry.get("clean_filtered_tail_rank")
        if isinstance(target, list) and len(target) == 3 and rank is not None:
            result[tuple(target)] = float(rank)
    return result


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def status_reason(result: dict) -> str:
    if result["num_pra_templates"] == 0:
        return "No PRA templates passed threshold"
    if result["num_semantically_valid_templates"] == 0:
        return "No templates survived semantic/endpoint and embedding checks"
    if result["num_ranked_paths"] == 0:
        return "No candidates survived entity/path selection and filtering"
    return "At least one eligible candidate path"


def run_feasibility(
    targets_file: Path = DEFAULT_TARGETS,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    checkpoint: Path = DEFAULT_CHECKPOINT,
    pra_method: str = "supervised",
    threshold: float = 0.9,
    top_k: int = 10,
    top_entities: int = 10,
    metadata_file: Path = DEFAULT_TARGET_METADATA,
    limit: int | None = None,
) -> dict:
    """Run existing Entity Selection on each target and save diagnostics."""
    if pra_method not in ("supervised", "empirical"):
        raise ValueError("pra_method must be 'supervised' or 'empirical'")
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be between 0 and 1")
    if top_k < 1 or top_entities < 1:
        raise ValueError("top_k and top_entities must be positive")
    if limit is not None and limit < 1:
        raise ValueError("limit must be positive")
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Missing TransE checkpoint: {checkpoint}")
    if pra_method == "supervised" and not DEFAULT_SUPERVISED_PRA.is_file():
        raise FileNotFoundError(f"Missing supervised PRA checkpoint: {DEFAULT_SUPERVISED_PRA}")

    all_targets = read_tsv_triples(targets_file)
    targets = all_targets[:limit] if limit is not None else all_targets
    clean_ranks = read_setup_ranks(metadata_file)

    # Ensure that Set B remains disjoint from ALL ORIGINAL KG splits.
    dataset_dir = MASS_DIR / "dataset_uml"
    original_facts = set()
    for split in ("train", "valid", "test"):
        original_facts.update(read_tsv_triples(dataset_dir / f"{split}.txt"))
    overlap = [t for t in targets if t in original_facts]
    if overlap:
        raise ValueError(
            f"Set B contains {len(overlap)} already-known triples; "
            f"first overlap: {overlap[0]!r}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    feasible_targets = []
    for index, (head, relation, tail) in enumerate(targets, 1):
        target_id = f"target_{index:03d}"
        result_path = output_dir / f"{target_id}.json"
        row = {field: "" for field in CSV_COLUMNS}
        row.update({
            "target_id": target_id, "head": head, "relation": relation, "tail": tail,
            "pra_method": pra_method,
            "clean_filtered_tail_rank": clean_ranks.get((head, relation, tail), ""),
            "result_json": result_path.name,
        })
        try:
            # Current rank_paths handles PRA, ontology constraints, TransE scoring,
            # endpoint checks, self-loop filtering and non-new-edge filtering.
            result = rank_paths(
                head, relation, tail,
                threshold=threshold,
                top_k=top_k,
                top_entities=top_entities,
                checkpoint=checkpoint,
                pra_method=pra_method,
            )
            candidates = result["ranked_candidate_paths"]
            new_edges = {
                tuple(candidate[key])
                for candidate in candidates
                for key in ("first_triple", "second_triple")
                if tuple(candidate[key]) not in original_facts
            }
            row.update({
                "num_pra_templates": result["num_pra_templates"],
                "num_semantically_valid_templates": result["num_semantically_valid_templates"],
                "num_selected_templates": result["num_selected_templates"],
                "num_ranked_paths": len(candidates),
                "one_new_triple_paths": sum(c["num_new_triples"] == 1 for c in candidates),
                "two_new_triple_paths": sum(c["num_new_triples"] == 2 for c in candidates),
                "distinct_new_triples_available": len(new_edges),
                "best_lower_bound_score": (
                    candidates[0]["lower_bound_score"] if candidates else ""
                ),
                "status": "feasible" if candidates else "infeasible",
                "reason": status_reason(result),
            })
            write_json(result_path, result)
            if candidates:
                feasible_targets.append((head, relation, tail))
        except Exception as exc:
            # Preserve a record for each target; any errors still produce a
            # nonzero process exit code so they cannot be mistaken for infeasibility.
            row.update({
                "status": "error", "reason": "Entity Selection execution failed",
                "error": f"{type(exc).__name__}: {exc}",
            })
            write_json(result_path, {
                "target_triple": [head, relation, tail],
                "pra_method": pra_method, "status": "error", "error": row["error"],
            })
        rows.append(row)
        print(
            f"[{index:02d}/{len(targets):02d}] {target_id} "
            f"({head}, {relation}, {tail}) | "
            f"PRA={row['num_pra_templates'] or '-'} "
            f"valid={row['num_semantically_valid_templates'] or '-'} "
            f"paths={row['num_ranked_paths'] if row['num_ranked_paths'] != '' else '-'} "
            f"| {row['status']}"
        )

    csv_path = output_dir / "feasibility_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    feasible_path = output_dir / "feasible_targets.tsv"
    with feasible_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerows(feasible_targets)

    counts = Counter(row["status"] for row in rows)
    summary = {
        "experiment": "UMLS MaSS Set B target feasibility",
        "method": "existing Entity_Selection.rank_paths (no Random Sampling)",
        "config": {
            "targets_file": str(targets_file),
            "metadata_file": str(metadata_file),
            "transe_checkpoint": str(checkpoint),
            "pra_method": pra_method,
            "pra_threshold": threshold,
            "top_k": top_k,
            "top_entities_per_template": top_entities,
            "endpoint_compatibility_and_path_filters": "as implemented in Entity_Selection.py",
        },
        "set_B_targets_total": len(targets),
        "feasible_targets": counts.get("feasible", 0),
        "infeasible_targets": counts.get("infeasible", 0),
        "error_targets": counts.get("error", 0),
        "coverage_percent": round(100 * counts.get("feasible", 0) / len(targets), 2),
        "total_ranked_candidate_paths": sum(
            int(row["num_ranked_paths"])
            for row in rows if row["num_ranked_paths"] != ""
        ),
        "note": (
            "Feasibility means at least one path survived the current filters. "
            "It does not guarantee that Random Sampling will accept a path, "
            "that a two-triple injection budget can be satisfied, "
            "or that a victim-model attack will succeed."
        ),
        "output_files": {
            "per_target": "target_001.json ... target_NNN.json",
            "summary_csv": csv_path.name,
            "feasible_targets_tsv": feasible_path.name,
        },
        "targets": rows,
    }
    json_path = output_dir / "feasibility_summary.json"
    write_json(json_path, summary)
    print("\n=== FEASIBILITY SUMMARY ===")
    print(f"Targets: {len(targets)}")
    print(f"Feasible: {summary['feasible_targets']}")
    print(f"Infeasible: {summary['infeasible_targets']}")
    print(f"Errors: {summary['error_targets']}")
    print(f"Coverage: {summary['coverage_percent']:.2f}%")
    print(f"Saved: {csv_path}")
    print(f"Saved: {json_path}")
    print(f"Saved: {feasible_path}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Feasibility check for UMLS MaSS Set B targets")
    parser.add_argument("--targets", type=Path, default=DEFAULT_TARGETS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_TARGET_METADATA,
                        help="Optional target_setup_summary.json for clean ranks")
    parser.add_argument("--pra-method", choices=("supervised", "empirical"),
                        default="supervised")
    parser.add_argument("--threshold", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--top-entities", type=int, default=10)
    parser.add_argument("--limit", type=int, default=None,
                        help="Optional first N targets for a quick smoke test")
    args = parser.parse_args()
    summary = run_feasibility(
        targets_file=args.targets, output_dir=args.output_dir,
        checkpoint=args.checkpoint, pra_method=args.pra_method,
        threshold=args.threshold, top_k=args.top_k,
        top_entities=args.top_entities, metadata_file=args.metadata,
        limit=args.limit,
    )
    if summary["error_targets"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
