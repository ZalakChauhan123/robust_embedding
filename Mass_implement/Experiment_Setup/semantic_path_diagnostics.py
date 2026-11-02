
"""Priority 2: audit UMLS MaSS path semantics without changing injected triples.

Run from Mass_implement/ after target_feasibility_check.py.
Exact SRSTRE2 pair support is a diagnostic, not a mandatory MaSS rule.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

MASS_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = MASS_DIR / "dataset_uml"


def normalize(value: str) -> str:
    return (
        value.strip()
        .lower()
        .replace(",", "")
        .replace(" ", "_")
    )


def load_ontology(path: Path):
    domain = defaultdict(set)
    range_ = defaultdict(set)
    pairs = defaultdict(set)

    with path.open(encoding="utf-8-sig") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue

            fields = line.rstrip("\r\n").split("|")

            if len(fields) < 3 or not all(fields[:3]):
                raise ValueError(
                    f"Malformed SRSTRE2 line {number}"
                )

            h = normalize(fields[0])
            r = fields[1].strip().lower()
            t = normalize(fields[2])

            domain[r].add(h)
            range_[r].add(t)
            pairs[r].add((h, t))

    return domain, range_, pairs


def read_targets(path: Path):
    rows = []

    with path.open(encoding="utf-8-sig") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue

            fields = line.strip().split("\t")

            if len(fields) != 3:
                raise ValueError(
                    f"Expected 3 TSV columns at {path}:{number}"
                )

            rows.append(tuple(fields))

    if not rows or len(set(rows)) != len(rows):
        raise ValueError("Empty or duplicated Set B targets")

    return rows


def check_edge(edge, domain, range_, pairs):
    h, r, t = edge

    d = h in domain.get(r, ())
    q = t in range_.get(r, ())
    e = (h, t) in pairs.get(r, ())

    return {
        "domain_ok": d,
        "range_ok": q,
        "marginal_ok": d and q,
        "exact_pair_listed": e,
    }


def write_csv(path, rows, fields):
    with path.open(
        "w", newline="", encoding="utf-8-sig"
    ) as handle:

        writer = csv.DictWriter(
            handle, fieldnames=fields
        )
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(
        description="Read-only MaSS SRSTRE2 path audit"
    )

    parser.add_argument(
        "--targets", type=Path, required=True
    )

    parser.add_argument(
        "--feasibility-dir", type=Path, required=True
    )

    parser.add_argument(
        "--srstre2",
        type=Path,
        default=DATA_DIR / "Semantic_Constraints" / "SRSTRE2"
    )

    parser.add_argument(
        "--output-dir", type=Path, required=True
    )

    parser.add_argument(
        "--selected-paths",
        type=Path,
        help="Optional selected_poisoning_paths.json"
    )

    args = parser.parse_args()

    targets = read_targets(args.targets)

    domain, range_, pairs = load_ontology(args.srstre2)

    selected_by_id = {}

    if args.selected_paths is not None:
        obj = json.loads(
            args.selected_paths.read_text(
                encoding="utf-8-sig"
            )
        )

        selected_by_id = {
            x["target_id"]:
                x["stealthiness"]["selected_candidate_paths"]
            for x in obj["targets"]
        }

    path_rows = []
    target_rows = []

    for index, target in enumerate(targets, 1):
        tid = f"target_{index:03d}"

        source = (
            args.feasibility_dir / f"{tid}.json"
        )

        if not source.is_file():
            raise FileNotFoundError(
                f"Missing feasibility output: {source}. "
                "Run target_feasibility_check.py first."
            )

        data = json.loads(
            source.read_text(encoding="utf-8-sig")
        )

        if tuple(data.get("target_triple", [])) != target:
            raise ValueError(
                f"Target mismatch in {source}. "
                "Regenerate feasibility for this Set B."
            )

        candidates = data.get("ranked_candidate_paths")

        if not isinstance(candidates, list):
            raise ValueError(
                f"Missing ranked_candidate_paths in {source}"
            )

        target_check = check_edge(
            target, domain, range_, pairs
        )

        summary = {
            "target_id": tid,
            "status": data.get("status", "unknown"),
            "target_head": target[0],
            "target_relation": target[1],
            "target_tail": target[2],
            "target_marginal_ok":
                target_check["marginal_ok"],
            "target_exact_pair_listed":
                target_check["exact_pair_listed"],
            "candidate_paths": len(candidates),
            "candidate_paths_marginal_ok": 0,
            "candidate_paths_both_exact_pairs": 0,
            "selected_paths": 0,
            "selected_paths_both_exact_pairs": 0,
        }

        stages = (
            ("candidate", candidates),
            ("selected", selected_by_id.get(tid, [])),
        )

        for stage, paths in stages:
            for j, candidate in enumerate(paths, 1):

                edge1 = candidate.get("first_triple", [])
                edge2 = candidate.get("second_triple", [])

                if len(edge1) != 3 or len(edge2) != 3:
                    raise ValueError(
                        f"Malformed path: {tid} {stage} #{j}"
                    )

                e1 = check_edge(
                    edge1, domain, range_, pairs
                )
                e2 = check_edge(
                    edge2, domain, range_, pairs
                )

                shape_ok = (
                    edge1[0] == target[0]
                    and edge2[2] == target[2]
                    and edge1[2] == edge2[0]
                    and edge1[1] == candidate.get("r1")
                    and edge2[1] == candidate.get("r2")
                )

                marginal_ok = (
                    shape_ok
                    and e1["marginal_ok"]
                    and e2["marginal_ok"]
                )

                exact_ok = (
                    shape_ok
                    and e1["exact_pair_listed"]
                    and e2["exact_pair_listed"]
                )

                path_rows.append({
                    "target_id": tid,
                    "stage": stage,
                    "path_index": j,
                    "r1": edge1[1],
                    "intermediate": edge1[2],
                    "r2": edge2[1],
                    "shape_ok": shape_ok,
                    "first_domain_ok": e1["domain_ok"],
                    "first_range_ok": e1["range_ok"],
                    "second_domain_ok": e2["domain_ok"],
                    "second_range_ok": e2["range_ok"],
                    "path_marginal_ok": marginal_ok,
                    "first_exact_pair_listed":
                        e1["exact_pair_listed"],
                    "second_exact_pair_listed":
                        e2["exact_pair_listed"],
                    "both_exact_pairs_listed": exact_ok,
                    "lower_bound_score":
                        candidate.get("lower_bound_score", ""),
                })

                if stage == "candidate":
                    summary[
                        "candidate_paths_marginal_ok"
                    ] += int(marginal_ok)

                    summary[
                        "candidate_paths_both_exact_pairs"
                    ] += int(exact_ok)
                else:
                    summary["selected_paths"] += 1

                    summary[
                        "selected_paths_both_exact_pairs"
                    ] += int(exact_ok)

        target_rows.append(summary)

    totals = {
        "num_targets": len(target_rows),
        "target_marginal_ok": sum(
            x["target_marginal_ok"] for x in target_rows
        ),
        "target_exact_pair_listed": sum(
            x["target_exact_pair_listed"] for x in target_rows
        ),
        "candidate_paths": sum(
            x["candidate_paths"] for x in target_rows
        ),
        "candidate_paths_marginal_ok": sum(
            x["candidate_paths_marginal_ok"]
            for x in target_rows
        ),
        "candidate_paths_both_exact_pairs": sum(
            x["candidate_paths_both_exact_pairs"]
            for x in target_rows
        ),
        "targets_with_candidates": sum(
            x["candidate_paths"] > 0 for x in target_rows
        ),
        "targets_with_exact_pair_paths": sum(
            x["candidate_paths_both_exact_pairs"] > 0
            for x in target_rows
        ),
        "selected_paths": sum(
            x["selected_paths"] for x in target_rows
        ),
        "selected_paths_both_exact_pairs": sum(
            x["selected_paths_both_exact_pairs"]
            for x in target_rows
        ),
    }

    args.output_dir.mkdir(
        parents=True, exist_ok=True
    )

    write_csv(
        args.output_dir / "target_semantic_audit.csv",
        target_rows,
        list(target_rows[0]),
    )

    path_fields = [
        "target_id",
        "stage",
        "path_index",
        "r1",
        "intermediate",
        "r2",
        "shape_ok",
        "first_domain_ok",
        "first_range_ok",
        "second_domain_ok",
        "second_range_ok",
        "path_marginal_ok",
        "first_exact_pair_listed",
        "second_exact_pair_listed",
        "both_exact_pairs_listed",
        "lower_bound_score",
    ]

    write_csv(
        args.output_dir / "path_semantic_audit.csv",
        path_rows,
        path_fields,
    )

    report = {
        "scope": (
            "Diagnostics only. No path selection, "
            "poisoning, or training changed."
        ),
        "interpretation": (
            "An exact SRSTRE2 pair not being listed "
            "does not necessarily mean the triple is false."
        ),
        "totals": totals,
    }

    (args.output_dir / "semantic_audit_summary.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(totals, indent=2))
    print("Saved to:", args.output_dir.resolve())


if __name__ == "__main__":
    main()
