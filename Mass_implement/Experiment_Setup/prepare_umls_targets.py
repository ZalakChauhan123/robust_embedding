"""
-- Prepare UMLS Set B for Experiment.
-- Place at Mass_implement/Experiment_Setup/prepare_umls_targets.py.
-- Run from Mass_implement/:
        python Experiment_Setup/prepare_umls_targets.py --num-targets 20 --target-rank 100 --seed 42

-- Set A is the original UMLS test.txt and is not modified.
-- Set B contains ABSENT / low-plausibility candidate triples
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import torch

MASS_DIR = Path(__file__).resolve().parents[1]
MODULE_DIR = MASS_DIR / "Path_Generation_Module"
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

from TransE_Surrogate import Surrogate, read_triples


def clean_tail_rank(
    surrogate: Surrogate,
    head: str,
    relation: str,
    tail: str,
    true_tails: dict[tuple[str, str], set[str]],
) -> float:
    """Filtered tail rank of a candidate ABSENT target using the clean TransE model."""
    model = surrogate.model
    eid = surrogate.entity_to_id
    rid = surrogate.relation_to_id
    n = len(eid)
    heads = torch.full((n,), eid[head], dtype=torch.long)
    rels = torch.full((n,), rid[relation], dtype=torch.long)
    tails = torch.arange(n, dtype=torch.long)
    with torch.no_grad():
        scores = model.score(heads, rels, tails)
    target_score = scores[eid[tail]].item()
    # Exclude already-known true answers so the candidate's rank isn't inflated
    # by observed facts. The candidate itself is guaranteed absent.
    other_true = true_tails.get((head, relation), set()) - {tail}
    if other_true:
        scores[[eid[t] for t in other_true]] = -torch.inf
    better = (scores > target_score).sum().item()
    tied = (scores == target_score).sum().item()
    return float(1 + better + (tied - 1) / 2)


def prepare(num_targets: int, target_rank: int, min_rank: int, seed: int,
            checkpoint: Path, output_dir: Path) -> dict:
    data_dir = MASS_DIR / "dataset_uml"
    data = {split: read_triples(data_dir / f"{split}.txt")
            for split in ("train", "valid", "test")}
    all_true = set(data["train"] + data["valid"] + data["test"])
    true_tails = defaultdict(set)
    for h, r, t in all_true:
        true_tails[(h, r)].add(t)

    surrogate = Surrogate.load(checkpoint)
    entity_names = list(surrogate.entity_to_id)
    n_entities = len(entity_names)
    if not 1 <= target_rank <= n_entities:
        raise ValueError(f"--target-rank must be between 1 and {n_entities}")
    if not 1 <= min_rank <= target_rank:
        raise ValueError("--min-rank must be >= 1 and <= --target-rank")
    if num_targets < 1:
        raise ValueError("--num-targets must be positive")

    rng = random.Random(seed)
    originals = list(data["test"])
    rng.shuffle(originals)
    selected = []
    used_targets = set()
    for original_h, relation, tail in originals:
        if len(selected) >= num_targets:
            break
        # Search alternate existing source entities for the same relation + tail.
        # Each target must be a missing fact in ALL the original splits.
        alternatives = []
        for new_h in entity_names:
            target = (new_h, relation, tail)
            if new_h == original_h or target in all_true or target in used_targets:
                continue
            rank = clean_tail_rank(surrogate, new_h, relation, tail, true_tails)
            if rank >= min_rank:
                alternatives.append((abs(rank - target_rank), -rank, new_h, rank))
        if not alternatives:
            continue
        alternatives.sort()
        _, _, new_head, clean_rank = alternatives[0]
        target = (new_head, relation, tail)
        used_targets.add(target)
        selected.append({
            "target_triple": list(target),
            "original_test_triple": [original_h, relation, tail],
            "clean_filtered_tail_rank": clean_rank,
        })

    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "set_B_target_triples.tsv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerows(row["target_triple"] for row in selected)
    with (output_dir / "set_B_provenance.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["original_head", "relation", "tail", "new_head", "clean_filtered_tail_rank"])
        for row in selected:
            o = row["original_test_triple"]
            writer.writerow([o[0], o[1], o[2], row["target_triple"][0], row["clean_filtered_tail_rank"]])

    summary = {
        "experiment": "UMLS MaSS Set B pilot preparation",
        "paper_fidelity": "adaptation: TransE-only low-plausibility screening, not TransE+ConvKB",
        "set_A_clean_test_path": str(data_dir / "test.txt"),
        "set_A_size": len(data["test"]),
        "set_B_target_triples_path": str(output_dir / "set_B_target_triples.tsv"),
        "requested_set_B_size": num_targets,
        "actual_set_B_size": len(selected),
        "clean_train_size": len(data["train"]),
        "num_entities": n_entities,
        "desired_target_rank": target_rank,
        "min_acceptable_target_rank": min_rank,
        "surrogate_checkpoint": str(checkpoint),
        "random_seed": seed,
        "all_targets_absent_from_original_train_valid_test": all(
            tuple(row["target_triple"]) not in all_true for row in selected
        ),
        "set_B_rows": selected,
    }
    (output_dir / "target_setup_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(f"Set A: original test.txt ({len(data['test'])} triples)")
    print(f"Set B: {len(selected)}/{num_targets} absent target triples selected")
    print(f"Target ranks: {[x['clean_filtered_tail_rank'] for x in selected]}")
    print(f"Saved: {output_dir / 'set_B_target_triples.tsv'}")
    if len(selected) < num_targets:
        print("NOTE: Not enough test triples yielded a candidate at the requested minimum rank.")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare fixed Set A/Set B for UMLS MaSS pilot")
    parser.add_argument("--num-targets", type=int, default=20)
    parser.add_argument("--target-rank", type=int, default=100,
                        help="Desired approximate CLEAN filtered tail rank (UMLS has 135 entities)")
    parser.add_argument("--min-rank", type=int, default=75,
                        help="Minimum acceptable CLEAN filtered tail rank")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint", type=Path,
                        default=MASS_DIR / "outputs" / "Path_Generation" / "transe_umls.pt")
    parser.add_argument("--output-dir", type=Path,
                        default=MASS_DIR / "outputs" / "Experiments" / "UMLS_pilot" / "setup")
    args = parser.parse_args()
    prepare(args.num_targets, args.target_rank, args.min_rank,
            args.seed, args.checkpoint, args.output_dir)


if __name__ == "__main__":
    main()
