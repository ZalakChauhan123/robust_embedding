"""
SECTION 4.4.3 -- Stealthiness
"""
import argparse
import json
from collections import Counter
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_DIR / "outputs" / "Path_Generation" / "random_sampling_output.json"
DEFAULT_OUTPUT = PROJECT_DIR / "outputs" / "Path_Generation" / "stealthiness_output.json"


def read_training_triples(path: Path):
    """READ NAME-BASED TRAINING TRIPLES (HEAD RELATION TAIL)."""
    triples = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t") if "\t" in line else line.split()
            if len(parts) != 3:
                raise ValueError(f"Expected three fields at {path}:{line_number}")
            triples.append(tuple(parts))
    return triples


def annotate_stealthiness(payload: dict, triples: list) -> dict:
    """COUNT INCIDENT TRAINING EDGES AND ANNOTATE SELECTED PATHS."""
    degrees = Counter()
    for head, _, tail in triples:
        degrees[head] += 1
        degrees[tail] += 1

    paths = payload.get("selected_candidate_paths")
    if not isinstance(paths, list):
        raise ValueError("Missing selected_candidate_paths list")

    annotated = []
    for path in paths:
        entity = path["intermediate_entity"]
        annotated.append({**path, "intermediate_entity_degree": degrees[entity]})

    return {
        "dataset": payload.get("dataset"),
        "target_triple": payload.get("target_triple"),
        "degree_definition": "IN_PLUS_OUT_TRAINING_TRIPLE_OCCURRENCES; SELF_LOOP_COUNTS_TWICE",
        "num_training_triples": len(triples),
        "num_selected_paths": len(annotated),
        "selected_candidate_paths": annotated,
        "paths_by_ascending_intermediate_degree": [
            {"rank": p.get("rank"), "intermediate_entity": p["intermediate_entity"],
             "degree": p["intermediate_entity_degree"]}
            for p in sorted(annotated, key=lambda p: (p["intermediate_entity_degree"], p.get("rank", 0)))
        ],
        "note": "DEGREE ANNOTATION ONLY; NO PATHS FILTERED OR RESELECTED",
    }


def main():
    parser = argparse.ArgumentParser(description="MASS SECTION 4.4.3 STEALTHINESS ANALYSIS")
    parser.add_argument("--train", type=Path, required=True,
                        help="PATH TO UMLS TRAINING TRIPLES USING ENTITY AND RELATION NAMES")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    with args.input.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    result = annotate_stealthiness(payload, read_training_triples(args.train))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)

    print(f"TRAINING TRIPLES: {result['num_training_triples']}")
    print(f"SELECTED PATHS: {result['num_selected_paths']}")
    for item in result["paths_by_ascending_intermediate_degree"]:
        print(f"RANK {item['rank']}: {item['intermediate_entity']} DEGREE={item['degree']}")
    print(f"OUTPUT: {args.output}")


if __name__ == "__main__":
    main()
