"""MaSS Section 4.4.3 -- degree-aware path-selection experiment.

The paper qualitatively favors lower-degree intermediate entities; it does not
specify a numeric penalty or ranking formula. ``low_degree`` is our explicit
UMLS ablation, NOT a verbatim reproduction of a paper-provided rule.

Policy is applied AFTER MaSS random path sampling and BEFORE the triple budget.
``baseline`` retains the original sample order and therefore unchanged
injection choices for the same random seed and inputs.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Iterable

PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_DIR / "outputs" / "Path_Generation" / "random_sampling_output.json"
DEFAULT_OUTPUT = PROJECT_DIR / "outputs" / "Path_Generation" / "stealthiness_output.json"
POLICIES = ("baseline", "low_degree")


def read_training_triples(path: Path) -> list[tuple[str, str, str]]:
    triples = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip() or line.startswith("#"):
                continue
            fields = line.strip().split("\t")
            if len(fields) != 3:
                raise ValueError(f"Expected 3 TSV columns at {path}:{line_no}")
            triples.append(tuple(fields))
    return triples


def entity_degrees(triples: Iterable[tuple[str, str, str]]) -> Counter:
    """Undirected incident occurrence count on the ORIGINAL clean training KG."""
    degree = Counter()
    for head, _, tail in triples:
        degree[head] += 1
        degree[tail] += 1
    return degree


def mean_intermediate_degree(paths: list[dict], degrees: Counter) -> float | None:
    """Unweighted average degree across selected PATHS; None if none selected."""
    if not paths:
        return None
    return float(mean(degrees[path["intermediate_entity"]] for path in paths))


def order_sampled_paths(paths: list[dict], degrees: Counter,
                        policy: str = "baseline") -> list[dict]:
    """Order already-accepted paths before applying the existing TRIPLE budget.

    Baseline: exactly preserve order produced by Random_Sampling.sample_paths.
    Low degree: prefer smallest clean-KG intermediate degree. If tied,
    prefer the higher TransE lower-bound score, then original candidate rank.
    No candidates are discarded here and the random acceptance rate is unchanged.
    """
    if policy not in POLICIES:
        raise ValueError(f"Unknown policy {policy!r}; choose from {POLICIES}")
    if policy == "baseline":
        return list(paths)
    return sorted(
        paths,
        key=lambda p: (
            degrees[p["intermediate_entity"]],
            -float(p["lower_bound_score"]),
            int(p.get("rank", 0)),
        ),
    )


def annotate_stealthiness(payload: dict,
                           triples: list[tuple[str, str, str]],
                           *, degrees: Counter | None = None,
                           selection_policy: str = "baseline") -> dict:
    """Report degrees of the actual paths chosen under the specified policy."""
    if selection_policy not in POLICIES:
        raise ValueError(f"Unknown selection policy: {selection_policy}")
    degrees = degrees if degrees is not None else entity_degrees(triples)
    paths = payload.get("selected_candidate_paths")
    if not isinstance(paths, list):
        raise ValueError("Missing selected_candidate_paths list")
    annotated = [
        {**path, "intermediate_entity_degree": degrees[path["intermediate_entity"]]}
        for path in paths
    ]
    return {
        "dataset": payload.get("dataset"),
        "target_triple": payload.get("target_triple"),
        "selection_policy": selection_policy,
        "degree_definition": "IN_PLUS_OUT_CLEAN_TRAINING_TRIPLE_OCCURRENCES; SELF_LOOP_COUNTS_TWICE",
        "num_training_triples": len(triples),
        "num_selected_paths": len(annotated),
        "mean_selected_intermediate_degree": mean_intermediate_degree(paths, degrees),
        "selected_candidate_paths": annotated,
        "paths_by_ascending_intermediate_degree": [
            {"rank": p.get("rank"), "intermediate_entity": p["intermediate_entity"],
             "degree": p["intermediate_entity_degree"]}
            for p in sorted(annotated,
                            key=lambda p: (p["intermediate_entity_degree"],
                                           p.get("rank", 0)))
        ],
        "note": (
            "Baseline: degree is annotation only; original MaSS sampling/budget order retained."
            if selection_policy == "baseline" else
            "Low-degree UMLS ablation: accepted paths re-ordered by intermediate degree BEFORE budget allocation."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Report degree of selected MaSS paths")
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--selection-policy", choices=POLICIES, default="baseline")
    args = parser.parse_args()
    payload = json.loads(args.input.read_text(encoding="utf-8-sig"))
    triples = read_training_triples(args.train)
    result = annotate_stealthiness(payload, triples,
                                   selection_policy=args.selection_policy)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Selected paths: {result['num_selected_paths']}")
    print(f"Mean intermediate degree: {result['mean_selected_intermediate_degree']}")
    print("Note: this standalone CLI ANNOTATES paths; for degree-AWARE selection,"
          " run construct_poisoned_dataset.py --selection-policy low_degree")
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
