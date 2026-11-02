
"""MaSS UMLS Priority 1: Set B target generation."""

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
sys.path.insert(
    0, str(MASS_DIR / "Path_Generation_Module")
)

from TransE_Surrogate import Surrogate, read_triples


def filtered_tail_rank(surrogate, h, r, t, true_tails):
    """Calculate filtered average-tie rank of an absent target."""

    e2id = surrogate.entity_to_id
    r2id = surrogate.relation_to_id
    n = len(e2id)

    with torch.no_grad():
        scores = surrogate.model.score(
            torch.full((n,), e2id[h], dtype=torch.long),
            torch.full((n,), r2id[r], dtype=torch.long),
            torch.arange(n, dtype=torch.long),
        ).clone()

    if t in true_tails.get((h, r), set()):
        raise ValueError("Target is already a known fact")

    target_score = scores[e2id[t]].item()

    for other in true_tails.get((h, r), set()):
        scores[e2id[other]] = -torch.inf

    better = int((scores > target_score).sum())
    tied = int((scores == target_score).sum())

    return float(1 + better + (tied - 1) / 2)


def prepare(args):

    if not 1 <= args.min_rank <= args.target_rank:
        raise ValueError("Invalid minimum or target rank")

    if args.max_rank is not None:
        if args.max_rank < args.target_rank:
            raise ValueError("max-rank must be >= target-rank")

    data_dir = MASS_DIR / "dataset_uml"

    data = {
        s: read_triples(data_dir / f"{s}.txt")
        for s in ("train", "valid", "test")
    }

    true_facts = set(
        data["train"] + data["valid"] + data["test"]
    )

    true_tails = defaultdict(set)

    for h, r, t in true_facts:
        true_tails[(h, r)].add(t)

    surrogate = Surrogate.load(args.checkpoint)
    entities = list(surrogate.entity_to_id)

    if args.num_targets > len(data["test"]):
        raise ValueError("Too many requested targets")

    if args.target_rank > len(entities):
        raise ValueError("Target rank exceeds entity count")

    if args.max_rank and args.max_rank > len(entities):
        raise ValueError("Maximum rank exceeds entity count")

    output_dir = args.output_dir

    filenames = [
        "set_B_target_triples.tsv",
        "set_B_provenance.csv",
        "target_setup_summary.json",
    ]

    if not args.overwrite:
        if any((output_dir / n).exists() for n in filenames):
            raise FileExistsError(
                "Output exists. Choose another output directory."
            )

    rng = random.Random(args.seed)

    # Randomly sample original test triples.
    anchors = rng.sample(
        data["test"],
        k=args.num_targets
    )

    selected = []
    used = set()

    for original_h, relation, original_t in anchors:

        candidates = []

        for new_entity in entities:

            if args.replacement_mode == "tail":

                if new_entity == original_t:
                    continue

                target = (
                    original_h,
                    relation,
                    new_entity
                )

            else:

                if new_entity == original_h:
                    continue

                target = (
                    new_entity,
                    relation,
                    original_t
                )

            # Do not inject a known KG fact as a target.
            if target in true_facts or target in used:
                continue

            rank = filtered_tail_rank(
                surrogate,
                *target,
                true_tails
            )

            upper = args.max_rank or len(entities)

            if args.min_rank <= rank <= upper:

                candidates.append(
                    (
                        abs(rank - args.target_rank),
                        -rank,
                        new_entity,
                        target
                    )
                )

        if not candidates:
            continue

        # Prefer ranks closest to the requested rank.
        candidates.sort(
            key=lambda x: (x[0], x[1], x[2])
        )

        _, neg_rank, _, target = candidates[0]

        used.add(target)

        selected.append({
            "original_test_triple": [
                original_h, relation, original_t
            ],
            "target_triple": list(target),
            "clean_filtered_tail_rank": -neg_rank,
        })

    if len(selected) != args.num_targets:
        if not args.allow_partial:
            raise ValueError(
                f"Only {len(selected)}/{args.num_targets} "
                "targets found. No files written. "
                "Change seed or rank window."
            )

    if not selected:
        raise ValueError("No valid targets found")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Set B file
    with (output_dir / filenames[0]).open(
        "w", encoding="utf-8", newline=""
    ) as f:

        csv.writer(
            f, delimiter="\t"
        ).writerows(
            row["target_triple"] for row in selected
        )

    # Provenance file
    with (output_dir / filenames[1]).open(
        "w", encoding="utf-8", newline=""
    ) as f:

        writer = csv.writer(f)

        writer.writerow([
            "original_head",
            "relation",
            "original_tail",
            "target_head",
            "target_tail",
            "clean_filtered_tail_rank"
        ])

        for row in selected:

            oh, relation, ot = row["original_test_triple"]
            th, _, tt = row["target_triple"]

            writer.writerow([
                oh, relation, ot, th, tt,
                row["clean_filtered_tail_rank"]
            ])

    # Metadata required by feasibility module
    summary = {
        "experiment": "UMLS MaSS Priority 1",
        "paper_fidelity": (
            "UMLS adaptation with TransE-only target screening"
        ),
        "replacement_mode": args.replacement_mode,
        "screening_models": ["TransE"],
        "set_A_clean_test_path": str(
            data_dir / "test.txt"
        ),
        "set_A_size": len(data["test"]),
        "set_B_target_triples_path": str(
            output_dir / filenames[0]
        ),
        "requested_set_B_size": args.num_targets,
        "actual_set_B_size": len(selected),
        "clean_train_size": len(data["train"]),
        "num_entities": len(entities),
        "desired_target_rank": args.target_rank,
        "min_acceptable_target_rank": args.min_rank,
        "max_acceptable_target_rank": args.max_rank,
        "surrogate_checkpoint": str(args.checkpoint),
        "random_seed": args.seed,
        "all_targets_absent_from_original_train_valid_test": True,
        "set_B_rows": selected,
    }

    (output_dir / filenames[2]).write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8"
    )

    print(
        f"Created {len(selected)}/{args.num_targets} "
        f"Set B targets"
    )

    print(
        "Target ranks:",
        [r["clean_filtered_tail_rank"] for r in selected]
    )

    print("Saved to:", output_dir.resolve())


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--replacement-mode",
        choices=["tail", "head"],
        default="tail"
    )

    parser.add_argument(
        "--num-targets", type=int, default=20
    )

    parser.add_argument(
        "--target-rank", type=int, default=100
    )

    parser.add_argument(
        "--min-rank", type=int, default=75
    )

    parser.add_argument(
        "--max-rank", type=int
    )

    parser.add_argument(
        "--seed", type=int, default=42
    )

    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=MASS_DIR / "outputs" /
                "Path_Generation" / "transe_umls.pt"
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=MASS_DIR / "outputs" /
                "Experiments" / "UMLS_tail_v1" / "setup"
    )

    parser.add_argument(
        "--allow-partial", action="store_true"
    )

    parser.add_argument(
        "--overwrite", action="store_true"
    )

    prepare(parser.parse_args())


if __name__ == "__main__":
    main()
