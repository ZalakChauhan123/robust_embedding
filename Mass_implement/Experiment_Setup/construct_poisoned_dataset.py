"""
-- Construct a poisoned UMLS training graph from MaSS feasibility outputs.

-- Run from: Mass_implement/
    python Experiment_Setup/construct_poisoned_dataset.py --delta 0.7 --budget 2 --seed 42

Prerequisite: Experiment_Setup/target_feasibility_check.py has generated
outputs/Experiments/UMLS_pilot/feasibility/target_001.json ... target_020.json.

This is a leakage-controlled UMLS PILOT, not an exact reproduction of the MaSS
paper. The existing Stealthiness module annotates degrees but does not filter.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

MASS_DIR = Path(__file__).resolve().parents[1]
PATH_MODULE_DIR = MASS_DIR / "Path_Generation_Module"
if str(PATH_MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(PATH_MODULE_DIR))

from Random_Sampling import sample_paths
from Stealthiness import annotate_stealthiness

DATASET_DIR = MASS_DIR / "dataset_uml"
PILOT_DIR = MASS_DIR / "outputs" / "Experiments" / "UMLS_pilot"
DEFAULT_TARGETS = PILOT_DIR / "setup" / "set_B_target_triples.tsv"
DEFAULT_FEASIBILITY_DIR = PILOT_DIR / "feasibility"
DEFAULT_OUTPUT_DIR = PILOT_DIR / "poisoning"

Triple = tuple[str, str, str]
CSV_FIELDS = [
    "target_id", "head", "relation", "tail", "original_status", "initial_paths",
    "eligible_paths", "rejected_leakage", "rejected_target_overlap",
    "rejected_invalid", "sampled_paths", "selected_paths", "injected_for_target",
    "new_to_global_training", "budget_skipped_paths", "status",
]


def read_triples(path: Path, *, require_unique: bool = False) -> list[Triple]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing input file: {path}")
    triples: list[Triple] = []
    seen: set[Triple] = set()
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for line_no, row in enumerate(csv.reader(handle, delimiter="\t"), 1):
            if not row or (len(row) == 1 and not row[0].strip()):
                continue
            if row[0].lstrip().startswith("#"):
                continue
            triple = tuple(col.strip() for col in row)
            if len(triple) != 3 or not all(triple):
                raise ValueError(f"Invalid triple at {path}:{line_no}: {row!r}")
            if require_unique and triple in seen:
                raise ValueError(f"Duplicate triple at {path}:{line_no}: {triple!r}")
            triples.append(triple)  # Preserve the original training-file order.
            seen.add(triple)
    if not triples:
        raise ValueError(f"Empty triple file: {path}")
    return triples


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing experiment output: {path}")
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return obj


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def write_triples(path: Path, triples: list[Triple]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerows(triples)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_candidate_edges(candidate: dict[str, Any]) -> tuple[Triple, Triple]:
    edges = []
    for key in ("first_triple", "second_triple"):
        value = candidate.get(key)
        if not isinstance(value, list) or len(value) != 3 or not all(
            isinstance(v, str) and v.strip() for v in value
        ):
            raise ValueError(f"Malformed candidate {key}: {value!r}")
        edges.append(tuple(value))
    return edges[0], edges[1]


def filter_candidate_paths(
    paths: list[dict[str, Any]],
    train_set: set[Triple],
    held_out_set: set[Triple],
    target_set: set[Triple],
    target: Triple,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Keep complete paths only; do not partially add a rejected two-hop path.

    Held-out (valid/test) triples may NEVER be newly added to the training graph.
    A target triple may NEVER be directly injected, even if it appears in a path.
    """
    counts: Counter[str] = Counter()
    eligible = []
    for candidate in paths:
        if not isinstance(candidate, dict):
            counts["invalid"] += 1
            continue
        try:
            first, second = parse_candidate_edges(candidate)
        except ValueError:
            counts["invalid"] += 1
            continue
        if (
            candidate.get("has_self_loop", False)
            or first[0] == first[2]
            or second[0] == second[2]
            or first[2] != second[0]
            or first[0] != target[0]
            or second[2] != target[2]
            or (first in train_set and second in train_set)
        ):
            counts["invalid"] += 1
            continue
        if first in target_set or second in target_set:
            counts["target_overlap"] += 1
            continue
        if (first in held_out_set and first not in train_set) or (second in held_out_set and second not in train_set):
            counts["leakage"] += 1
            continue
        # Preserve source ranking and metadata, use only clean-train status for the budget.
        eligible.append(candidate)
    return eligible, dict(counts)


def construct(
    targets_file: Path = DEFAULT_TARGETS,
    feasibility_dir: Path = DEFAULT_FEASIBILITY_DIR,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    train_file: Path = DATASET_DIR / "train.txt",
    valid_file: Path = DATASET_DIR / "valid.txt",
    test_file: Path = DATASET_DIR / "test.txt",
    *,
    delta: float = 0.7,
    budget: int = 2,
    seed: int = 42,
) -> dict[str, Any]:
    if not 0.0 <= delta <= 1.0:
        raise ValueError("delta must be within [0, 1]")
    if budget < 1:
        raise ValueError("budget must be >= 1")

    targets = read_triples(targets_file, require_unique=True)
    train = read_triples(train_file)
    train_set = set(train)
    valid_set = set(read_triples(valid_file))
    test_set = set(read_triples(test_file))
    held_out = valid_set | test_set
    target_set = set(targets)
    original_known = train_set | held_out

    if target_set & original_known:
        example = sorted(target_set & original_known)[0]
        raise ValueError(f"Set B target already appears in an original KG split: {example}")
    if train_set & held_out:
        raise ValueError("Original training and held-out sets overlap; resolve before evaluation")

    summary_path = feasibility_dir / "feasibility_summary.json"
    prior = load_json(summary_path)
    if prior.get("set_B_targets_total") != len(targets):
        raise ValueError("Set B size differs from feasibility run; regenerate feasibility")
    prior_rows = prior.get("targets")
    if not isinstance(prior_rows, list) or len(prior_rows) != len(targets):
        raise ValueError("Missing or inconsistent feasibility summary rows")
    if prior.get("error_targets", 0):
        raise ValueError("Feasibility run contains errors; fix these before injection")

    output_train = output_dir / "poisoned_train.txt"
    if output_train.resolve() == train_file.resolve():
        raise ValueError("The poisoned output must not overwrite the original train.txt")
    if output_dir.resolve() == DATASET_DIR.resolve():
        raise ValueError("Output directory must not be the original dataset directory")

    # Fail early if the per-target files or their target identities don't match.
    source_results = []
    for i, target in enumerate(targets, start=1):
        target_id = f"target_{i:03d}"
        prior_row = prior_rows[i - 1]
        if (prior_row.get("target_id") != target_id or
            tuple(prior_row.get(k, "") for k in ("head", "relation", "tail")) != target):
            raise ValueError(f"Feasibility summary differs from Set B at {target_id}")
        payload = load_json(feasibility_dir / f"{target_id}.json")
        if payload.get("target_triple") != list(target):
            raise ValueError(f"Feasibility JSON target mismatch at {target_id}")
        if prior_row.get("status") == "error" or payload.get("status") == "error":
            raise ValueError(f"Cannot use feasibility error result: {target_id}")
        paths = payload.get("ranked_candidate_paths")
        if not isinstance(paths, list):
            raise ValueError(f"Missing ranked_candidate_paths in {target_id}")
        if len(paths) != int(prior_row["num_ranked_paths"]):
            raise ValueError(f"Candidate count mismatch for {target_id}")
        source_results.append((target_id, target, prior_row, payload))

    all_injected: set[Triple] = set()
    all_selected = []
    all_eligible = []
    rows = []
    for i, (target_id, target, prior_row, payload) in enumerate(source_results):
        candidates = payload["ranked_candidate_paths"]
        eligible, rejected = filter_candidate_paths(candidates, train_set, held_out, target_set, target)
        all_eligible.append({
            "target_id": target_id, "target_triple": list(target),
            "input_paths": len(candidates), "eligible_paths": len(eligible),
            "rejections": rejected,
            "ranked_candidate_paths": eligible,
        })

        # Reuse MaSS Section 4.4.2 sampling. With max_paths=None, apply
        # acceptance probability 1-delta to all eligible paths. Then enforce
        # our separate TRIPLE budget; a path is used atomically.
        sampling = sample_paths(
            {"dataset": "UMLS", "target_triple": list(target),
             "ranked_candidate_paths": eligible},
            delta=delta, max_paths=None, seed=seed + i,
        )
        sampled = sampling["selected_candidate_paths"]
        chosen = []
        attributed: set[Triple] = set()
        skipped_budget = 0
        for candidate in sampled:
            first, second = parse_candidate_edges(candidate)
            required = {e for e in (first, second) if e not in train_set}
            # Charge only distinct non-clean edges, including any already
            # attributed to this target. No partial two-hop path injections.
            additional = required - attributed
            if len(attributed) + len(additional) > budget:
                skipped_budget += 1
                continue
            if not additional:
                # Repeating an already selected path would add no new evidence.
                continue
            attributed.update(additional)
            chosen.append(candidate)
            if len(attributed) == budget:
                break

        sampling_after_budget = {
            **sampling,
            "num_selected_paths_before_budget": sampling["num_selected_paths"],
            "num_selected_paths": len(chosen),
            "selected_candidate_paths": chosen,
        }
        stealth = annotate_stealthiness(sampling_after_budget, train)
        for triple in attributed:
            if triple in held_out or triple in target_set or triple in train_set:
                raise AssertionError(f"Illegal poisoning triple: {triple}")
        globally_new = attributed - all_injected
        all_injected.update(attributed)

        if prior_row["status"] == "infeasible":
            status = "originally_infeasible"
        elif not eligible:
            status = "no_eligible_paths_after_leakage_filter"
        elif not chosen:
            status = "not_selected_by_random_sampling_or_budget"
        else:
            status = "injected"
        row = {
            "target_id": target_id, "head": target[0], "relation": target[1],
            "tail": target[2], "original_status": prior_row["status"],
            "initial_paths": len(candidates), "eligible_paths": len(eligible),
            "rejected_leakage": rejected.get("leakage", 0),
            "rejected_target_overlap": rejected.get("target_overlap", 0),
            "rejected_invalid": rejected.get("invalid", 0),
            "sampled_paths": len(sampled), "selected_paths": len(chosen),
            "injected_for_target": len(attributed),
            "new_to_global_training": len(globally_new),
            "budget_skipped_paths": skipped_budget, "status": status,
        }
        rows.append(row)
        all_selected.append({
            "target_id": target_id, "target_triple": list(target),
            "status": status, "sampling_seed": seed + i,
            "sampling": {k: v for k, v in sampling.items() if k != "selected_candidate_paths"},
            "budget": budget, "budget_skipped_paths": skipped_budget,
            "attributed_injected_triples": [list(t) for t in sorted(attributed)],
            "new_to_global_training": [list(t) for t in sorted(globally_new)],
            "stealthiness": stealth,
        })
        print(f"[{i+1:02d}/{len(targets):02d}] {target_id}: "
              f"eligible={len(eligible)} sampled={len(sampled)} "
              f"selected={len(chosen)} added={len(globally_new)} | {status}")

    poisoned = train + sorted(all_injected)
    if (len(poisoned) != len(train) + len(all_injected) or
        set(poisoned) & held_out or set(all_injected) & target_set):
        raise AssertionError("Poisoned dataset invariants failed")
    if any(r["injected_for_target"] > budget for r in rows):
        raise AssertionError("Per-target budget exceeded")

    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "eligible_candidate_paths.json", {
        "filter": "reject any path injecting original valid/test or Set B facts",
        "targets": all_eligible,
    })
    write_json(output_dir / "selected_poisoning_paths.json", {
        "delta": delta, "budget_per_target": budget,
        "stealthiness_note": "degree annotation only, not degree-aware selection",
        "targets": all_selected,
    })
    write_triples(output_dir / "injected_triples.tsv", sorted(all_injected))
    write_triples(output_train, poisoned)
    csv_path = output_dir / "poisoning_summary.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    report = {
        "experiment": "MaSS UMLS pilot: leakage-controlled poisoning dataset construction",
        "source": "saved feasibility target_XXX.json files, not recomputed Entity Selection",
        "parameters": {"delta": delta, "path_accept_probability": 1 - delta,
                       "budget_distinct_new_triples_per_target": budget,
                       "seed": seed, "seed_per_target": "seed + zero-based Set B index"},
        "source_files": {
            "targets": str(targets_file), "feasibility_summary": str(summary_path),
            "train": str(train_file), "valid": str(valid_file), "test": str(test_file),
        },
        "source_sha256": {
            "targets": sha256(targets_file), "feasibility_summary": sha256(summary_path),
            "train": sha256(train_file), "valid": sha256(valid_file), "test": sha256(test_file),
        },
        "num_targets_set_B": len(targets),
        "num_originally_feasible": sum(r["original_status"] == "feasible" for r in rows),
        "num_after_leakage_feasible": sum(r["eligible_paths"] > 0 for r in rows),
        "num_targets_with_injections": sum(r["status"] == "injected" for r in rows),
        "num_targets_not_injected": sum(r["status"] != "injected" for r in rows),
        "total_input_candidate_paths": sum(r["initial_paths"] for r in rows),
        "total_eligible_candidate_paths": sum(r["eligible_paths"] for r in rows),
        "total_sampled_paths": sum(r["sampled_paths"] for r in rows),
        "total_selected_paths": sum(r["selected_paths"] for r in rows),
        "num_clean_train_triples": len(train),
        "num_unique_injected_triples": len(all_injected),
        "num_poisoned_train_triples": len(poisoned),
        "injected_train_overlap": len(all_injected & train_set),
        "injected_valid_overlap": len(all_injected & valid_set),
        "injected_test_overlap": len(all_injected & test_set),
        "injected_set_B_overlap": len(all_injected & target_set),
        "note": "Counts and outcomes only; attack success requires victim retraining and evaluation."
                " Infeasible Set B targets remain in the final evaluation set.",
        "targets": rows,
    }
    write_json(output_dir / "poisoning_summary.json", report)
    print("\n=== POISONED DATASET CONSTRUCTION ===")
    for key in ("num_targets_set_B", "num_originally_feasible", "num_after_leakage_feasible",
                "num_targets_with_injections", "total_input_candidate_paths",
                "total_eligible_candidate_paths", "total_selected_paths",
                "num_clean_train_triples", "num_unique_injected_triples", "num_poisoned_train_triples"):
        print(f"{key}: {report[key]}")
    print(f"Saved: {output_dir.resolve()}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Construct leakage-controlled UMLS MaSS poisoned training dataset")
    parser.add_argument("--targets", type=Path, default=DEFAULT_TARGETS)
    parser.add_argument("--feasibility-dir", type=Path, default=DEFAULT_FEASIBILITY_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--train", type=Path, default=DATASET_DIR / "train.txt")
    parser.add_argument("--valid", type=Path, default=DATASET_DIR / "valid.txt")
    parser.add_argument("--test", type=Path, default=DATASET_DIR / "test.txt")
    parser.add_argument("--delta", type=float, default=0.7)
    parser.add_argument("--budget", type=int, default=2,
                        help="Maximum distinct new triples injected per target")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    construct(targets_file=args.targets, feasibility_dir=args.feasibility_dir,
              output_dir=args.output_dir, train_file=args.train,
              valid_file=args.valid, test_file=args.test,
              delta=args.delta, budget=args.budget, seed=args.seed)


if __name__ == "__main__":
    main()
