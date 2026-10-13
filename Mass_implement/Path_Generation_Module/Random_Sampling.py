""" RANDOM SAMPLING OF RANKED MASS CANDIDATE PATHS 
    Implement SECTION 4.4.2
"""

import argparse
import json
import random
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_DIR / "outputs" / "Path_Generation" / "entity_selection_output.json"
DEFAULT_OUTPUT = PROJECT_DIR / "outputs" / "Path_Generation" / "random_sampling_output.json"


def sample_paths(payload: dict, delta: float = 0.7, max_paths: int | None = None,
                 seed: int | None = None) -> dict:
    """EXAMINE RANKED PATHS IN ORDER AND ACCEPT EACH WITH PROBABILITY 1 - DELTA."""
    if not 0.0 <= delta <= 1.0:
        raise ValueError("delta must be between 0 and 1")
    if max_paths is not None and max_paths < 1:
        raise ValueError("max_paths must be a positive integer")

    candidates = payload.get("ranked_candidate_paths")
    if not isinstance(candidates, list):
        raise ValueError("Input JSON must contain a 'ranked_candidate_paths' list")

    # PRESERVE THE ENTITY SELECTION RANKING BY SORTING ON LOWER-BOUND SCORE.
    ranked = sorted(candidates, key=lambda p: p["lower_bound_score"], reverse=True)
    rng = random.Random(seed)
    accepted = []
    examined = 0

    for candidate in ranked:
        if max_paths is not None and len(accepted) >= max_paths:
            break
        examined += 1
        if rng.random() < 1.0 - delta:
            # PRESERVE THE COMPLETE CANDIDATE, INCLUDING ORIGINAL RANK AND DIAGNOSTICS.
            accepted.append(candidate.copy())

    return {
        "dataset": payload.get("dataset"),
        "target_triple": payload.get("target_triple"),
        "delta": delta,
        "acceptance_probability": 1.0 - delta,
        "random_seed": seed,
        "requested_max_paths": max_paths,
        "num_input_paths": len(ranked),
        "num_examined_paths": examined,
        "num_selected_paths": len(accepted),
        "selected_candidate_paths": accepted,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="MASS SECTION 4.4.2 RANDOM SAMPLING")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--delta", type=float, default=0.7)
    parser.add_argument("--max-paths", type=int, default=None,
                        help="OPTIONAL PATH LIMIT; NOT THE POISONED-TRIPLE BUDGET")
    parser.add_argument("--seed", type=int, default=None,
                        help="OPTIONAL SEED FOR REPRODUCIBLE SAMPLING")
    args = parser.parse_args()

    with args.input.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    result = sample_paths(payload, delta=args.delta, max_paths=args.max_paths,
                          seed=args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)

    print(f"INPUT PATHS: {result['num_input_paths']}")
    print(f"EXAMINED PATHS: {result['num_examined_paths']}")
    print(f"SELECTED PATHS: {result['num_selected_paths']}")
    print(f"OUTPUT: {args.output}")


if __name__ == "__main__":
    main()