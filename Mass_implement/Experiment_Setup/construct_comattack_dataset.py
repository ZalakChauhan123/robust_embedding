#!/usr/bin/env python3
"""Command-line constructor for the ComAttack UMLS comparison baseline."""
from __future__ import annotations

import argparse
from pathlib import Path

from construct_attack_baselines import DATASET_DIR, MASS_DIR, construct


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Construct a leakage-controlled ComAttack-style promotion dataset"
    )
    parser.add_argument("--targets", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train", type=Path, default=DATASET_DIR / "train.txt")
    parser.add_argument("--valid", type=Path, default=DATASET_DIR / "valid.txt")
    parser.add_argument("--test", type=Path, default=DATASET_DIR / "test.txt")
    parser.add_argument("--budget", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--surrogate-checkpoint", type=Path,
                        default=MASS_DIR / "outputs" / "Path_Generation" / "transe_umls.pt")
    args = parser.parse_args()
    construct(
        "ComAttack", targets_file=args.targets, output_dir=args.output_dir,
        train_file=args.train, valid_file=args.valid, test_file=args.test,
        budget=args.budget, seed=args.seed, surrogate_checkpoint=args.surrogate_checkpoint,
    )


if __name__ == "__main__":
    main()
