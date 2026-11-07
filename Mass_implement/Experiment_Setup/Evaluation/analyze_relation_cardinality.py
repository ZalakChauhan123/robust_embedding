#!/usr/bin/env python3
"""Read-only empirical relation-cardinality audit for MaSS RQ3.

Run from Mass_implement/:
  python Experiment_Setup/Evaluation/analyze_relation_cardinality.py \
    --experiment-dir outputs/Experiments/UMLS_feasible_rank11_15_seed42

IMPORTANT: An absence of multiple observations does not prove functionality.
Only clean training triples are used to estimate cardinality.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

MASS_DIR = Path(__file__).resolve().parents[2]
METHODS = ('baseline', 'DirectAdd', 'ComAttack')


def read_triples(path: Path, *, unique_required: bool = False):
    if not path.is_file():
        raise FileNotFoundError(path)
    results, seen = [], set()
    with path.open(encoding='utf-8-sig', newline='') as handle:
        for number, row in enumerate(csv.reader(handle, delimiter='\t'), 1):
            if not row or (len(row) == 1 and not row[0].strip()) or row[0].lstrip().startswith('#'):
                continue
            triple = tuple(piece.strip() for piece in row)
            if len(triple) != 3 or not all(triple):
                raise ValueError(f'{path}:{number}: expected 3 nonempty TSV fields')
            if unique_required and triple in seen:
                raise ValueError(f'{path}:{number}: duplicated triple {triple}')
            seen.add(triple)
            results.append(triple)
    return results


def save_csv(path: Path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8', newline='') as out:
        writer = csv.DictWriter(out, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def ratio(a: int, b: int):
    return round(a / b, 6) if b else 0.0


def analyze(train):
    by_hr, by_rt, relation_edges = defaultdict(set), defaultdict(set), defaultdict(set)
    for h, r, t in set(train):
        by_hr[(h, r)].add(t)
        by_rt[(r, t)].add(h)
        relation_edges[r].add((h, t))

    result = []
    relation_index = {}
    for r in sorted(relation_edges):
        heads = {h for h, t in relation_edges[r]}
        tails = {t for h, t in relation_edges[r]}
        n = len(relation_edges[r])
        multi_head = sum(len(by_hr[(h, r)]) > 1 for h in heads)
        multi_tail = sum(len(by_rt[(r, t)]) > 1 for t in tails)
        if multi_head and multi_tail:
            pattern = 'observed_many_to_many'
        elif multi_head:
            pattern = 'observed_one_to_many_only'
        elif multi_tail:
            pattern = 'observed_many_to_one_only'
        else:
            pattern = 'no_multiplicity_observed'
        row = {
            'relation': r, 'num_triples': n,
            'num_heads': len(heads), 'num_tails': len(tails),
            'avg_tails_per_head': ratio(n, len(heads)),
            'avg_heads_per_tail': ratio(n, len(tails)),
            'heads_with_multiple_tails': multi_head,
            'tails_with_multiple_heads': multi_tail,
            'pct_heads_with_multiple_tails': round(100 * ratio(multi_head, len(heads)), 4),
            'pct_tails_with_multiple_heads': round(100 * ratio(multi_tail, len(tails)), 4),
            'observed_pattern': pattern,
            # Observed multi-valuedness disproves a universal functional constraint
            # in the dataset, assuming the clean data are trustworthy.
            'functional_counterexample_observed': bool(multi_head),
            'inverse_functional_counterexample_observed': bool(multi_tail),
        }
        result.append(row)
        relation_index[r] = row
    return result, relation_index, by_hr, by_rt


def audit_method(method, injected, clean_set, info, by_hr, by_rt):
    audit = []
    for h, r, t in injected:
        if (h, r, t) in clean_set:
            raise ValueError(f'{method}: injected triple already in clean train: {(h,r,t)}')
        same_hr = any(v != t for v in by_hr.get((h, r), ()))
        same_rt = any(v != h for v in by_rt.get((r, t), ()))
        strict = same_hr or same_rt
        stats = info.get(r)
        multi_forward = stats['functional_counterexample_observed'] if stats else False
        multi_inverse = stats['inverse_functional_counterexample_observed'] if stats else False
        # These flags identify an overlap for which the clean graph already
        # demonstrates that relation-wide uniqueness is invalid.
        explained_forward = same_hr and multi_forward
        explained_inverse = same_rt and multi_inverse
        audit.append({
            'method': method, 'head': h, 'relation': r, 'tail': t,
            'strict_overlap_flagged': strict,
            'same_head_relation_different_tail': same_hr,
            'same_relation_tail_different_head': same_rt,
            'relation_clean_triples': stats['num_triples'] if stats else 0,
            'relation_observed_pattern': stats['observed_pattern'] if stats else 'unseen_relation',
            'forward_multi_valued_evidence': multi_forward,
            'inverse_multi_valued_evidence': multi_inverse,
            'forward_overlap_with_multi_valued_evidence': explained_forward,
            'inverse_overlap_with_multi_valued_evidence': explained_inverse,
            'all_triggered_directions_have_multi_valued_evidence': strict and (not same_hr or multi_forward) and (not same_rt or multi_inverse),
            'at_least_one_triggered_direction_has_multi_valued_evidence': explained_forward or explained_inverse,
        })
    return audit


def main():
    parser = argparse.ArgumentParser(description='Read-only UMLS RQ3 cardinality analysis')
    parser.add_argument('--experiment-dir', type=Path, required=True)
    parser.add_argument('--train', type=Path, default=MASS_DIR / 'dataset_uml' / 'train.txt')
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--methods', nargs='+', choices=METHODS, default=list(METHODS))
    args = parser.parse_args()

    exp = args.experiment_dir.resolve()
    out = (args.output_dir or exp / 'RQ3' / 'cardinality').resolve()
    clean = read_triples(args.train)
    relation_rows, index, by_hr, by_rt = analyze(clean)
    save_csv(out / 'relation_cardinality.csv', relation_rows, list(relation_rows[0]))

    summary = []
    for method in args.methods:
        injected = read_triples(exp / method / 'poisoning' / 'injected_triples.tsv', unique_required=True)
        if not injected:
            raise ValueError(f'{method}: no injected triples')
        audited = audit_method(method, injected, set(clean), index, by_hr, by_rt)
        save_csv(out / f'{method}_overlap_audit.csv', audited, list(audited[0]))
        strict = sum(r['strict_overlap_flagged'] for r in audited)
        all_explained = sum(r['all_triggered_directions_have_multi_valued_evidence'] for r in audited)
        any_explained = sum(r['at_least_one_triggered_direction_has_multi_valued_evidence'] for r in audited)
        summary.append({
            'method': method, 'num_injected': len(audited),
            'strict_overlap_flagged': strict,
            'strict_overlap_rate_pct': round(100 * ratio(strict, len(audited)), 4),
            'flagged_all_directions_multi_valued': all_explained,
            'flagged_any_direction_multi_valued': any_explained,
            'flagged_without_multi_valued_evidence_in_all_directions': strict - all_explained,
            'unflagged': len(audited) - strict,
        })
    save_csv(out / 'cardinality_audit_summary.csv', summary, list(summary[0]))
    (out / 'cardinality_audit_summary.json').write_text(json.dumps({
        'source': str(args.train),
        'interpretation': 'Empirical multiplicity is evidence against universal uniqueness, not proof of a specific poisoned triple being true; no observed multiplicity is inconclusive.',
        'methods': summary,
    }, indent=2), encoding='utf-8')
    print(f'Saved cardinality analysis to {out}')
    for row in summary:
        print(f"{row['method']}: strict overlap={row['strict_overlap_flagged']}/{row['num_injected']}; "
              f"all triggered directions have observed multiplicity={row['flagged_all_directions_multi_valued']}")


if __name__ == '__main__':
    main()
