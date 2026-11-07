#!/usr/bin/env python3
"""Read-only RQ3 detection evaluation for the existing UMLS MaSS experiments.

Install at Mass_implement/Experiment_Setup/Evaluation/evaluate_rq3.py.
Run from Mass_implement/:
  python Experiment_Setup/Evaluation/evaluate_rq3.py \
      --experiment-dir outputs/Experiments/UMLS_feasible_rank11_15_seed42

Only newly injected triples are evaluated (not poisoned_train.txt or target triples).
The mutual-exclusion check reproduces Section 5.4's overlap heuristic; it
is NOT a logically sound contradiction test for non-functional relations.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

MASS_DIR = Path(__file__).resolve().parents[2]
EXPERIMENT_SETUP = MASS_DIR / "Experiment_Setup"
if str(EXPERIMENT_SETUP) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_SETUP))

from construct_poisoned_dataset import read_triples
from semantic_path_diagnostics import load_ontology, check_edge

METHODS = ("baseline", "DirectAdd", "ComAttack")


def build_clean_indexes(triples):
    hr = defaultdict(set)
    rt = defaultdict(set)
    for h, r, t in triples:
        hr[(h, r)].add(t)
        rt[(r, t)].add(h)
    return hr, rt


def evaluate_triple(triple, domain, range_, pairs, hr, rt):
    h, r, t = triple
    if r not in domain or r not in range_:
        type_status = "unknown_relation"
        domain_ok = range_ok = exact_pair_listed = ""
    else:
        checked = check_edge(triple, domain, range_, pairs)
        domain_ok = checked["domain_ok"]
        range_ok = checked["range_ok"]
        exact_pair_listed = checked["exact_pair_listed"]
        type_status = "passed" if checked["marginal_ok"] else "detected"

    # Paper's overly strict 'mutually exclusive fact' overlap heuristic.
    same_hr_other_tail = any(x != t for x in hr.get((h, r), ()))
    same_rt_other_head = any(x != h for x in rt.get((r, t), ()))
    mutual_detected = same_hr_other_tail or same_rt_other_head

    return {
        "head": h, "relation": r, "tail": t,
        "type_status": type_status,
        "domain_ok": domain_ok, "range_ok": range_ok,
        "exact_srstre2_pair_listed_diagnostic": exact_pair_listed,
        "same_head_relation_different_tail": same_hr_other_tail,
        "same_relation_tail_different_head": same_rt_other_head,
        "mutual_detected": mutual_detected,
        "combined_detected": type_status == "detected" or mutual_detected,
        "combined_unknown": type_status == "unknown_relation" and not mutual_detected,
    }


def write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def pct(n, total):
    return round(100.0 * n / total, 4) if total else None


def main():
    parser = argparse.ArgumentParser(description="UMLS MaSS RQ3 detection evaluation")
    parser.add_argument("--experiment-dir", type=Path, required=True)
    parser.add_argument("--train", type=Path, default=MASS_DIR / "dataset_uml" / "train.txt")
    parser.add_argument("--ontology", type=Path, default=MASS_DIR / "dataset_uml" / "Semantic_Constraints" / "SRSTRE2")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    args = parser.parse_args()

    experiment_dir = args.experiment_dir.resolve()
    output_dir = (args.output_dir or experiment_dir / "RQ3").resolve()
    domain, range_, pairs = load_ontology(args.ontology)
    clean = read_triples(args.train)
    clean_set = set(clean)
    hr, rt = build_clean_indexes(clean)
    summary = []
    for method in args.methods:
        input_file = experiment_dir / method / "poisoning" / "injected_triples.tsv"
        injected = read_triples(input_file, require_unique=True)
        duplicates = clean_set.intersection(injected)
        if duplicates:
            raise ValueError(f"{method}: {len(duplicates)} injected triples already in clean train")
        rows = [evaluate_triple(t, domain, range_, pairs, hr, rt) for t in injected]
        n = len(rows)
        type_detected = sum(x["type_status"] == "detected" for x in rows)
        type_unknown = sum(x["type_status"] == "unknown_relation" for x in rows)
        mutual_detected = sum(x["mutual_detected"] for x in rows)
        combined_detected = sum(x["combined_detected"] for x in rows)
        combined_unknown = sum(x["combined_unknown"] for x in rows)
        write_csv(output_dir / f"{method}_triplet_detection.csv", rows, list(rows[0]))
        summary.append({
            "method": method, "total_injected": n,
            "type_detected": type_detected,
            "type_unknown": type_unknown,
            "type_detection_rate_pct": pct(type_detected, n),
            "type_remaining_ratio": round((n-type_detected)/n, 4),
            "mutual_detected": mutual_detected,
            "mutual_detection_rate_pct": pct(mutual_detected, n),
            "mutual_remaining_ratio": round((n-mutual_detected)/n, 4),
            "combined_detected_union": combined_detected,
            "combined_detection_rate_pct": pct(combined_detected, n),
            "combined_unknown": combined_unknown,
            "combined_confirmed_undetected": n - combined_detected - combined_unknown,
        })

    write_csv(output_dir / "rq3_summary.csv", summary, list(summary[0]))
    (output_dir / "rq3_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"RQ3 results: {output_dir}")
    for item in summary:
        print(f"{item['method']}: {item['total_injected']} injected; "
              f"type detected={item['type_detected']} (unknown={item['type_unknown']}); "
              f"mutual detected={item['mutual_detected']}; "
              f"combined union={item['combined_detected_union']}")


if __name__ == "__main__":
    main()
