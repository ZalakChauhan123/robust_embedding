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
With no applicable documented rules, conflict rates are null (not assessable).
Use --sweep-dir to assess all saved RQ3 budgets separately.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
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
    "rule_assessed", "conflict_direction", "reason",
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
        source = rule.get("source")
        if ((rule.get("functional") or rule.get("inverse_functional"))
                and (not isinstance(source, str) or not source.strip())):
            raise ValueError(f"{relation}: true functionality rules require a documented source")
    return rules


def validate_rules(rules: dict, stats: dict, known_relations=None):
    """Do not call a rule confirmed when the clean graph refutes it."""
    known = set(stats) if known_relations is None else set(known_relations)
    for relation, rule in rules.items():
        if relation not in known:
            raise ValueError(f"Unknown relation in cardinality rules: {relation}")
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
        if ((not forward or s.get("functional_counterexample_observed", False))
                and (not inverse or s.get("inverse_functional_counterexample_observed", False))):
            reason = "Clean data shows multiplicity in every triggered direction; overlap alone is insufficient"
        else:
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
        "rule_assessed": functional is True or inverse_functional is True,
        "conflict_direction": ";".join(conflicts),
        "reason": reason,
    }


def save_csv(path: Path, rows: list[dict], fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def evaluate(experiment, *, train=MASS_DIR / "dataset_uml" / "train.txt",
             output=None, rules_file=None, methods=METHODS):
    experiment = experiment.resolve()
    output = (output or experiment / "RQ3" / "cardinality_aware").resolve()
    clean = read_triples(train, unique_required=True)
    clean_set = set(clean)
    relation_rows, stats, by_hr, by_rt = analyze(clean)
    rules = load_rules(rules_file)
    relations_file = MASS_DIR / "dataset_uml" / "relations.txt"
    known_relations = relations_file.read_text(encoding="utf-8-sig").splitlines()
    validate_rules(rules, stats, known_relations)
    save_csv(output / "relation_cardinality.csv", relation_rows, list(relation_rows[0]))
    input_paths = [train, relations_file]
    if rules_file:
        input_paths.append(rules_file)

    summaries = []
    for method in methods:
        injected_path = experiment / method / "poisoning" / "injected_triples.tsv"
        input_paths.append(injected_path)
        triples = read_triples(injected_path, unique_required=True)
        if not triples:
            raise ValueError(f"{method}: injected-triple list is empty")
        if overlap := set(triples) & clean_set:
            raise ValueError(f"{method}: injected triples overlap clean training data: {sorted(overlap)[:2]}")
        rows = [examine(method, triple, stats, by_hr, by_rt, rules) for triple in triples]
        save_csv(output / f"{method}_cardinality_detection.csv", rows, FIELDS)
        counts = Counter(row["cardinality_status"] for row in rows)
        strict = sum(row["strict_overlap_flagged"] for row in rows)
        n = len(rows)
        assessed = sum(row["rule_assessed"] for row in rows)
        confirmed = counts["confirmed_rule_conflict"]
        all_directions_multivalued = sum(
            row["strict_overlap_flagged"]
            and (not row["same_head_relation_different_tail"] or row["forward_multi_valued_evidence"])
            and (not row["same_relation_tail_different_head"] or row["inverse_multi_valued_evidence"])
            for row in rows
        )
        summaries.append({
            "method": method, "total_injected": n,
            "strict_overlap_flagged": strict,
            "strict_overlap_rate_pct": round(100 * strict / n, 4),
            "confirmed_rule_conflicts": confirmed,
            "rule_assessed_triples": assessed,
            "not_rule_assessed_triples": n - assessed,
            "rule_coverage_pct": round(100 * assessed / n, 4),
            "assessment_status": "assessed_with_documented_rules" if assessed else "not_assessable_no_applicable_rules",
            "confirmed_conflict_rate_pct": round(100 * confirmed / n, 4) if assessed else None,
            "confirmed_conflict_rate_among_assessed_pct": round(100 * confirmed / assessed, 4) if assessed else None,
            "inconclusive_overlaps": counts["inconclusive_overlap"],
            "no_overlaps": counts["no_overlap"],
            "overlaps_with_multiplicity_in_all_triggered_directions": all_directions_multivalued,
        })
    fields = list(summaries[0])
    save_csv(output / "cardinality_aware_summary.csv", summaries, fields)
    (output / "cardinality_aware_summary.json").write_text(json.dumps({
        "methodology": "Confirmed requires documented functional/inverse-functional rule; empirical multiplicity alone never establishes contradiction.",
        "rules_file": str(rules_file) if rules_file else None,
        "coverage_definition": "Fraction of injected triples with at least one documented true functional/inverse-functional constraint for their relation.",
        "rate_interpretation": "Rates are null when rule coverage is zero; with partial coverage the all-triples rate counts only known-rule conflicts and is not total contradiction prevalence.",
        "input_sha256": {str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest() for path in input_paths},
        "results": summaries,
    }, indent=2), encoding="utf-8")
    for row in summaries:
        print(f"{row['method']}: strict={row['strict_overlap_flagged']}/{row['total_injected']}, "
              f"confirmed={row['confirmed_rule_conflicts']}, "
              f"inconclusive={row['inconclusive_overlaps']}, no_overlap={row['no_overlaps']}, "
              f"rule_coverage={row['rule_coverage_pct']}% ({row['assessment_status']})")
    print(f"Outputs: {output}")
    return summaries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--experiment-dir", type=Path)
    source.add_argument("--sweep-dir", type=Path, help="Evaluate saved x_2, x_4, ... experiment directories")
    parser.add_argument("--budgets", type=int, nargs="+", default=[2, 4, 6, 8, 10, 12])
    parser.add_argument("--train", type=Path, default=MASS_DIR / "dataset_uml" / "train.txt")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--rules", type=Path, help="Optional manually verified relation-cardinality rules JSON")
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    args = parser.parse_args()
    if args.experiment_dir:
        evaluate(args.experiment_dir, train=args.train, output=args.output_dir,
                 rules_file=args.rules, methods=args.methods)
        return
    if (not args.budgets or any(b < 1 for b in args.budgets)
            or len(set(args.budgets)) != len(args.budgets)):
        parser.error("Budgets must be distinct positive integers")
    sweep = args.sweep_dir.resolve()
    output = (args.output_dir or sweep / "cardinality_aware").resolve()
    summaries = []
    for budget in sorted(args.budgets):
        rows = evaluate(sweep / f"x_{budget}", train=args.train, output=output / f"x_{budget}",
                        rules_file=args.rules, methods=args.methods)
        summaries.extend({"budget": budget, **row} for row in rows)
    save_csv(output / "cardinality_aware_budget_summary.csv", summaries, list(summaries[0]))
    (output / "cardinality_aware_budget_summary.json").write_text(json.dumps({
        "methodology": "Documented relation-specific uniqueness rules only; missing rules do not establish that injected triples are valid.",
        "rules_file": str(args.rules) if args.rules else None,
        "results": summaries,
    }, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
