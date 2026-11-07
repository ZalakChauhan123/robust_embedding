#!/usr/bin/env python3
"""RQ3 cardinality-aware mutual-exclusion evaluation (read-only).

Place alongside analyze_relation_cardinality.py in:
  Mass_implement/Experiment_Setup/Evaluation/
Run from Mass_implement:
  python Experiment_Setup/Evaluation/RQ3_Cardinality_Aware_Detection.py \
      --experiment-dir outputs/Experiments/UMLS_feasible_rank11_15_seed42

By default, flags with overlapping facts but no independently established
relation functionality are INCONCLUSIVE, not confirmed contradictions.
SRSTRE2 domain/range constraints do NOT supply cardinality information.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from analyze_relation_cardinality import analyze, read_triples

MASS_DIR = Path(__file__).resolve().parents[2]
METHODS = ("baseline", "DirectAdd", "ComAttack")
FIELDS = (
    "method", "head", "relation", "tail", "strict_overlap_flagged",
    "same_head_relation_different_tail", "same_relation_tail_different_head",
    "relation_clean_triples", "observed_pattern",
    "forward_multi_valued_evidence", "inverse_multi_valued_evidence",
    "functional_rule", "inverse_functional_rule", "cardinality_status",
    "conflict_direction", "reason",
)


def load_rules(path: Path | None) -> dict:
    """Optional manually curated, externally justified relation declarations.

    Format: {"relations": {"relation_name": {"functional": true,
              "inverse_functional": false, "source": "citation/ontology"}}}
    Omitted directions mean unknown. No functionality is inferred from counts.
    """
    if path is None:
        return {}
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict) or not isinstance(payload.get("relations"), dict):
        raise ValueError("Rules JSON must have a 'relations' object")
    rules = payload["relations"]
    for relation, rule in rules.items():
        if not isinstance(rule, dict):
            raise ValueError(f"Invalid rule for {relation}")
        for key in ("functional", "inverse_functional"):
            if key in rule and not isinstance(rule[key], bool):
                raise ValueError(f"{relation}.{key} must be boolean")
        if (rule.get("functional") or rule.get("inverse_functional")) and not str(rule.get("source", "")).strip():
            raise ValueError(f"{relation}: true functionality rules require a documented source")
    return rules


def validate_rules(rules: dict, stats: dict):
    """Do not call a rule confirmed when the clean graph refutes it."""
    for relation, rule in rules.items():
        s = stats.get(relation)
        if s is None:
            continue
        if rule.get("functional") is True and s["functional_counterexample_observed"]:
            raise ValueError(f"Rule conflicts with clean data: {relation} is not functional")
        if rule.get("inverse_functional") is True and s["inverse_functional_counterexample_observed"]:
            raise ValueError(f"Rule conflicts with clean data: {relation} is not inverse-functional")


def examine(method, triple, stats, by_hr, by_rt, rules):
    h, r, t = triple
    other_tails = by_hr.get((h, r), set()) - {t}
    other_heads = by_rt.get((r, t), set()) - {h}
    forward = bool(other_tails)
    inverse = bool(other_heads)
    strict = forward or inverse
    s = stats.get(r, {})
    rule = rules.get(r, {})
    functional = rule.get("functional", None)
    inverse_functional = rule.get("inverse_functional", None)

    conflicts = []
    if forward and functional is True:
        conflicts.append("functional")
    if inverse and inverse_functional is True:
        conflicts.append("inverse_functional")

    if conflicts:
        status = "confirmed_rule_conflict"
        reason = "Overlap contradicts explicitly documented functionality rule(s)"
    elif strict:
        status = "inconclusive_overlap"
        reason = "Overlap without a documented exclusivity rule for a triggered direction"
    else:
        status = "no_overlap"
        reason = "No strict overlap with clean training triples"

    return {
        "method": method, "head": h, "relation": r, "tail": t,
        "strict_overlap_flagged": strict,
        "same_head_relation_different_tail": forward,
        "same_relation_tail_different_head": inverse,
        "relation_clean_triples": s.get("num_triples", 0),
        "observed_pattern": s.get("observed_pattern", "unseen_relation"),
        "forward_multi_valued_evidence": s.get("functional_counterexample_observed", False),
        "inverse_multi_valued_evidence": s.get("inverse_functional_counterexample_observed", False),
        "functional_rule": "unknown" if functional is None else str(functional).lower(),
        "inverse_functional_rule": "unknown" if inverse_functional is None else str(inverse_functional).lower(),
        "cardinality_status": status,
        "conflict_direction": ";".join(conflicts),
        "reason": reason,
    }


def save_csv(path: Path, rows: list[dict], fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dir", type=Path, required=True)
    parser.add_argument("--train", type=Path, default=MASS_DIR / "dataset_uml" / "train.txt")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--rules", type=Path, help="Optional manually verified relation-cardinality rules JSON")
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    args = parser.parse_args()

    experiment = args.experiment_dir.resolve()
    output = (args.output_dir or experiment / "RQ3" / "cardinality_aware").resolve()
    clean = read_triples(args.train)
    clean_set = set(clean)
    _, stats, by_hr, by_rt = analyze(clean)
    rules = load_rules(args.rules)
    validate_rules(rules, stats)

    summaries = []
    for method in args.methods:
        triples = read_triples(experiment / method / "poisoning" / "injected_triples.tsv", unique_required=True)
        if not triples:
            raise ValueError(f"{method}: injected-triple list is empty")
        if overlap := set(triples) & clean_set:
            raise ValueError(f"{method}: injected triples overlap clean training data: {sorted(overlap)[:2]}")
        rows = [examine(method, triple, stats, by_hr, by_rt, rules) for triple in triples]
        save_csv(output / f"{method}_cardinality_detection.csv", rows, FIELDS)
        counts = Counter(row["cardinality_status"] for row in rows)
        strict = sum(row["strict_overlap_flagged"] for row in rows)
        n = len(rows)
        summaries.append({
            "method": method, "total_injected": n,
            "strict_overlap_flagged": strict,
            "strict_overlap_rate_pct": round(100 * strict / n, 4),
            "confirmed_rule_conflicts": counts["confirmed_rule_conflict"],
            "confirmed_conflict_rate_pct": round(100 * counts["confirmed_rule_conflict"] / n, 4),
            "inconclusive_overlaps": counts["inconclusive_overlap"],
            "no_overlaps": counts["no_overlap"],
        })
    fields = list(summaries[0])
    save_csv(output / "cardinality_aware_summary.csv", summaries, fields)
    (output / "cardinality_aware_summary.json").write_text(json.dumps({
        "methodology": "Confirmed requires documented functional/inverse-functional rule; empirical multiplicity alone never establishes contradiction.",
        "rules_file": str(args.rules) if args.rules else None,
        "results": summaries,
    }, indent=2), encoding="utf-8")
    for row in summaries:
        print(f"{row['method']}: strict={row['strict_overlap_flagged']}/{row['total_injected']}, "
              f"confirmed={row['confirmed_rule_conflicts']}, "
              f"inconclusive={row['inconclusive_overlaps']}, no_overlap={row['no_overlaps']}")
    print(f"Outputs: {output}")


if __name__ == "__main__":
    main()
