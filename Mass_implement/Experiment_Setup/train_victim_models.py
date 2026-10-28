#!/usr/bin/env python3
"""
-- Train clean/poisoned TransE victims for the UMLS MaSS pilot.
-- Run from Mass_implement/:
    python Experiment_Setup/train_victim_models.py --epochs 500 --dim 64 \
        --batch-size 256 --lr 0.01 --margin 1.0 --seed 42
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F

MASS_DIR = Path(__file__).resolve().parents[1]
PATH_MODULE_DIR = MASS_DIR / "Path_Generation_Module"
if str(PATH_MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(PATH_MODULE_DIR))

from TransE_Surrogate import TransE, read_names, read_triples

DATASET_DIR = MASS_DIR / "dataset_uml"
PILOT_DIR = MASS_DIR / "outputs" / "Experiments" / "UMLS_pilot"
DEFAULT_POISONING_DIR = PILOT_DIR / "poisoning"
DEFAULT_OUTPUT_DIR = PILOT_DIR / "victim_models"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sha256_initial_state(model: torch.nn.Module) -> str:
    """Hash initial embedding weights to prove equal initialization."""
    digest = hashlib.sha256()
    for key, value in sorted(model.state_dict().items()):
        digest.update(key.encode("utf-8"))
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def unique_triples(path: Path) -> list[tuple[str, str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Input file not found: {path}")
    triples = read_triples(path)
    if not triples:
        raise ValueError(f"Empty triple file: {path}")
    if len(triples) != len(set(triples)):
        raise ValueError(f"Duplicate triples found: {path}")
    return triples


def verify_datasets(
    clean_file: Path,
    poisoned_file: Path,
    injected_file: Path,
    valid_file: Path,
    test_file: Path,
    targets_file: Path,
    entities: list[str],
    relations: list[str],
) -> tuple[list[tuple[str, str, str]], list[tuple[str, str, str]], dict[str, Any]]:
    """Check the poisoned graph is exactly clean + the expected new triples."""
    clean = unique_triples(clean_file)
    poisoned = unique_triples(poisoned_file)
    injected = unique_triples(injected_file)
    valid = unique_triples(valid_file)
    test = unique_triples(test_file)
    targets = unique_triples(targets_file)
    clean_set, poisoned_set, injected_set = set(clean), set(poisoned), set(injected)
    held_out_set, target_set = set(valid) | set(test), set(targets)

    if clean_set & held_out_set:
        raise ValueError("Original train.txt overlaps validation/test splits")
    if clean_set & target_set or held_out_set & target_set:
        raise ValueError("Set B targets overlap original clean/validation/test splits")
    if not clean_set <= poisoned_set:
        raise ValueError("Poisoned training set is missing clean training triples")
    if injected_set & clean_set:
        raise ValueError("Injected triples already occur in original clean training data")
    if injected_set & held_out_set:
        raise ValueError("Injected triples overlap validation/test data (data leakage)")
    if injected_set & target_set:
        raise ValueError("A Set B target was injected directly")
    if poisoned_set != clean_set | injected_set:
        raise ValueError("poisoned_train.txt must be EXACTLY clean train + injected_triples.tsv")

    entity_set, relation_set = set(entities), set(relations)
    for label, triples in (("clean", clean), ("poisoned", poisoned)):
        for triple in triples:
            h, r, t = triple
            if h not in entity_set or t not in entity_set or r not in relation_set:
                raise ValueError(f"Unknown entity/relation in {label}: {triple!r}")

    counts = {
        "clean_training_triples": len(clean),
        "poisoned_training_triples": len(poisoned),
        "unique_injected_triples": len(injected),
        "validation_triples": len(valid),
        "test_triples": len(test),
        "set_B_target_triples": len(targets),
        "num_entities": len(entities),
        "num_relations": len(relations),
        "verified_poisoned_equals_clean_union_injected": True,
        "verified_injection_disjoint_from_valid_test_and_set_B": True,
        "source_sha256": {
            "clean_train": sha256_file(clean_file),
            "poisoned_train": sha256_file(poisoned_file),
            "injected_triples": sha256_file(injected_file),
            "validation": sha256_file(valid_file),
            "test": sha256_file(test_file),
            "set_B_targets": sha256_file(targets_file),
        },
    }
    return clean, poisoned, counts


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def encode_triples(
    triples: list[tuple[str, str, str]],
    entity_to_id: dict[str, int],
    relation_to_id: dict[str, int],
) -> torch.Tensor:
    return torch.tensor(
        [(entity_to_id[h], relation_to_id[r], entity_to_id[t]) for h, r, t in triples],
        dtype=torch.long,
    )


def train_one_victim(
    label: str,
    triples: list[tuple[str, str, str]],
    entity_to_id: dict[str, int],
    relation_to_id: dict[str, int],
    *,
    epochs: int,
    dim: int,
    batch_size: int,
    lr: float,
    margin: float,
    seed: int,
    checkpoint_path: Path,
    loss_csv_path: Path,
    log_every: int = 50,
) -> dict[str, Any]:
    """Match the existing surrogate training loop, but use configurable inputs."""
    set_seed(seed)  # Reset before BOTH victims: identical initial embeddings.
    model = TransE(len(entity_to_id), len(relation_to_id), dim)
    initial_state_sha256 = sha256_initial_state(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    encoded = encode_triples(triples, entity_to_id, relation_to_id)
    losses: list[float] = []

    print(f"\n[{label}] Training TransE on {len(encoded)} triples", flush=True)
    for epoch in range(1, epochs + 1):
        model.train()
        indices = torch.randperm(len(encoded))
        total_loss = 0.0
        for batch_indices in indices.split(batch_size):
            positive = encoded[batch_indices]

            # SAME uniform head/tail corruption as TransE_Surrogate.train().
            negative = positive.clone()
            corrupt_head = torch.rand(len(negative)) < 0.5
            replacement = torch.randint(len(entity_to_id), (len(negative),))
            negative[corrupt_head, 0] = replacement[corrupt_head]
            negative[~corrupt_head, 2] = replacement[~corrupt_head]

            positive_distance = model.distance(
                positive[:, 0], positive[:, 1], positive[:, 2]
            )
            negative_distance = model.distance(
                negative[:, 0], negative[:, 1], negative[:, 2]
            )
            loss = F.relu(margin + positive_distance - negative_distance).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                model.entity.weight.copy_(
                    F.normalize(model.entity.weight, p=2, dim=1)
                )
            total_loss += loss.item() * len(batch_indices)
        epoch_loss = total_loss / len(encoded)
        losses.append(epoch_loss)
        if epoch == 1 or epoch == epochs or epoch % log_every == 0:
            print(f"[{label}] Epoch {epoch:4d}/{epochs}: loss={epoch_loss:.6f}", flush=True)

    # Same checkpoint keys as the existing Surrogate.load() implementation.
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": model.state_dict(),
        "entity_to_id": entity_to_id,
        "relation_to_id": relation_to_id,
        "embedding_dim": dim,
    }, checkpoint_path)

    loss_csv_path.parent.mkdir(parents=True, exist_ok=True)
    with loss_csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["epoch", "training_loss"])
        writer.writerows(enumerate(losses, start=1))

    return {
        "checkpoint": str(checkpoint_path),
        "training_loss_csv": str(loss_csv_path),
        "training_triples": len(triples),
        "initial_state_sha256": initial_state_sha256,
        "first_epoch_loss": losses[0],
        "final_epoch_loss": losses[-1],
        "minimum_epoch_loss": min(losses),
        "checkpoint_sha256": sha256_file(checkpoint_path),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    for key in ("epochs", "dim", "batch_size", "log_every"):
        if getattr(args, key) < 1:
            raise ValueError(f"--{key.replace('_', '-')} must be positive")
    if args.lr <= 0 or args.margin <= 0:
        raise ValueError("--lr and --margin must be positive")
    if args.seed < 0:
        raise ValueError("--seed must be nonnegative")

    paths = {
        "clean": args.output_dir / "clean_transe.pt",
        "poisoned": args.output_dir / "poisoned_transe.pt",
        "clean_loss": args.output_dir / "clean_training_loss.csv",
        "poisoned_loss": args.output_dir / "poisoned_training_loss.csv",
        "summary": args.output_dir / "training_summary.json",
    }
    for name, path in paths.items():
        if path.exists() and not args.overwrite:
            raise FileExistsError(f"{name} output already exists: {path}. Pass --overwrite to retrain.")

    entities = read_names(args.entities)
    relations = read_names(args.relations)
    if not entities or len(set(entities)) != len(entities):
        raise ValueError("Entity ID list is empty or contains duplicates")
    if not relations or len(set(relations)) != len(relations):
        raise ValueError("Relation ID list is empty or contains duplicates")
    entity_to_id = {name: i for i, name in enumerate(entities)}
    relation_to_id = {name: i for i, name in enumerate(relations)}
    clean, poisoned, integrity = verify_datasets(
        args.clean_train, args.poisoned_train, args.injected,
        args.valid, args.test, args.targets, entities, relations,
    )

    config = {
        "epochs": args.epochs,
        "embedding_dim": args.dim,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "margin": args.margin,
        "random_seed": args.seed,
        "optimizer": "Adam",
        "loss": "mean ReLU(margin + positive_L2_distance - negative_L2_distance)",
        "negative_sampling": "uniform random entity replacing head/tail with probability 0.5",
        "embedding_normalization": "L2-normalize entity embeddings after each optimizer step",
        "device": "cpu",
        "same_entity_relation_id_mappings": True,
        "independent_training_from_scratch": True,
    }
    print("Dataset integrity passed:",
          f"clean={len(clean)}, poisoned={len(poisoned)},",
          f"injected={integrity['unique_injected_triples']}", flush=True)
    clean_model = train_one_victim(
        "clean", clean, entity_to_id, relation_to_id,
        epochs=args.epochs, dim=args.dim, batch_size=args.batch_size,
        lr=args.lr, margin=args.margin, seed=args.seed,
        checkpoint_path=paths["clean"], loss_csv_path=paths["clean_loss"],
        log_every=args.log_every,
    )
    poisoned_model = train_one_victim(
        "poisoned", poisoned, entity_to_id, relation_to_id,
        epochs=args.epochs, dim=args.dim, batch_size=args.batch_size,
        lr=args.lr, margin=args.margin, seed=args.seed,
        checkpoint_path=paths["poisoned"], loss_csv_path=paths["poisoned_loss"],
        log_every=args.log_every,
    )
    same_initial_weights = (
        clean_model["initial_state_sha256"] == poisoned_model["initial_state_sha256"]
    )
    if not same_initial_weights:
        raise AssertionError("Victims did not start from identical initialized weights")

    report = {
        "experiment": "MaSS UMLS pilot — matched clean and poisoned TransE victims",
        "parameters": config,
        "datasets": {
            "clean_train_file": str(args.clean_train),
            "poisoned_train_file": str(args.poisoned_train),
            "injected_triples_file": str(args.injected),
            "validation_file": str(args.valid),
            "test_file": str(args.test),
            "set_B_file": str(args.targets),
            **integrity,
        },
        "victims": {"clean": clean_model, "poisoned": poisoned_model},
        "same_initial_embeddings_verified": same_initial_weights,
        "notes": [
            "The existing PRA/TransE surrogate checkpoint was not used as a victim initialization.",
            "Identical seeds and initial weights do not imply identical per-epoch permutations after the datasets differ.",
            "Training loss is not an attack-success measure. Next evaluate both victims on Set A and Set B.",
            "This is a TransE-only pilot, not a full reproduction of all MaSS victim architectures.",
        ],
    }
    paths["summary"].parent.mkdir(parents=True, exist_ok=True)
    paths["summary"].write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("\n=== VICTIM TRAINING FINISHED ===")
    print(f"Identical initial embeddings: {same_initial_weights}")
    print(f"Clean final training loss: {clean_model['final_epoch_loss']:.6f}")
    print(f"Poisoned final training loss: {poisoned_model['final_epoch_loss']:.6f}")
    print(f"Output directory: {args.output_dir.resolve()}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Train clean and poisoned UMLS TransE victim models")
    parser.add_argument("--clean-train", type=Path, default=DATASET_DIR / "train.txt")
    parser.add_argument("--poisoned-train", type=Path, default=DEFAULT_POISONING_DIR / "poisoned_train.txt")
    parser.add_argument("--injected", type=Path, default=DEFAULT_POISONING_DIR / "injected_triples.tsv")
    parser.add_argument("--valid", type=Path, default=DATASET_DIR / "valid.txt")
    parser.add_argument("--test", type=Path, default=DATASET_DIR / "test.txt")
    parser.add_argument("--targets", type=Path, default=PILOT_DIR / "setup" / "set_B_target_triples.tsv")
    parser.add_argument("--entities", type=Path, default=DATASET_DIR / "entities.txt")
    parser.add_argument("--relations", type=Path, default=DATASET_DIR / "relations.txt")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--margin", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--overwrite", action="store_true",
                        help="Allow replacing existing victim checkpoints and loss files")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
