#!/usr/bin/env python3
"""Train matched clean and poisoned ConvKB UMLS victims."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import torch
from torch.nn import functional as F

MASS_DIR = Path(__file__).resolve().parents[1]
if str(MASS_DIR) not in sys.path:
    sys.path.insert(0, str(MASS_DIR))

from Path_Generation_Module.ConvKB_Victim import ConvKB  # noqa: E402
from Path_Generation_Module.TransE_Surrogate import read_names  # noqa: E402
from Experiment_Setup.train_victim_models import (  # noqa: E402
    set_seed,
    sha256_file,
    sha256_initial_state,
    verify_datasets,
)

DATASET_DIR = MASS_DIR / "dataset_uml"
PILOT_DIR = MASS_DIR / "outputs" / "Experiments" / "UMLS_pilot"
DEFAULT_POISONING_DIR = PILOT_DIR / "poisoning"
DEFAULT_OUTPUT_DIR = PILOT_DIR / "convkb_victim_models"


def encode_triples(
    triples: list[tuple[str, str, str]], entity_to_id: dict[str, int], relation_to_id: dict[str, int]
) -> torch.Tensor:
    return torch.tensor(
        [(entity_to_id[head], relation_to_id[relation], entity_to_id[tail])
         for head, relation, tail in triples], dtype=torch.long,
    )


def train_one_victim(
    label: str,
    triples: list[tuple[str, str, str]],
    entity_to_id: dict[str, int],
    relation_to_id: dict[str, int],
    *,
    epochs: int,
    dim: int,
    num_filters: int,
    dropout: float,
    batch_size: int,
    lr: float,
    l2_regularization: float,
    seed: int,
    checkpoint_path: Path,
    loss_path: Path,
    log_every: int,
) -> dict[str, Any]:
    set_seed(seed)
    model = ConvKB(len(entity_to_id), len(relation_to_id), dim, num_filters, dropout)
    initial_state = sha256_initial_state(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    encoded = encode_triples(triples, entity_to_id, relation_to_id)
    losses: list[float] = []
    print(f"\n[{label}] Training ConvKB on {len(encoded)} triples", flush=True)
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for batch_indices in torch.randperm(len(encoded)).split(batch_size):
            positive = encoded[batch_indices]
            negative = positive.clone()
            corrupt_head = torch.rand(len(negative)) < 0.5
            replacement = torch.randint(len(entity_to_id), (len(negative),))
            negative[corrupt_head, 0] = replacement[corrupt_head]
            negative[~corrupt_head, 2] = replacement[~corrupt_head]
            positive_score = model.score_triples(positive[:, 0], positive[:, 1], positive[:, 2])
            negative_score = model.score_triples(negative[:, 0], negative[:, 1], negative[:, 2])
            logistic_loss = (F.softplus(-positive_score) + F.softplus(negative_score)).mean()
            l2 = (
                model.entity(positive[:, 0]).pow(2).mean() +
                model.relation(positive[:, 1]).pow(2).mean() +
                model.entity(positive[:, 2]).pow(2).mean()
            )
            loss = logistic_loss + l2_regularization * l2
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * len(batch_indices)
        epoch_loss = total_loss / len(encoded)
        losses.append(epoch_loss)
        if epoch == 1 or epoch == epochs or epoch % log_every == 0:
            print(f"[{label}] Epoch {epoch:4d}/{epochs}: loss={epoch_loss:.6f}", flush=True)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_type": "ConvKB", "state_dict": model.state_dict(),
        "entity_to_id": entity_to_id, "relation_to_id": relation_to_id,
        "model_config": model.model_config(),
    }, checkpoint_path)
    loss_path.parent.mkdir(parents=True, exist_ok=True)
    with loss_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["epoch", "training_loss"])
        writer.writerows(enumerate(losses, start=1))
    return {
        "checkpoint": str(checkpoint_path), "training_loss_csv": str(loss_path),
        "training_triples": len(triples), "initial_state_sha256": initial_state,
        "first_epoch_loss": losses[0], "final_epoch_loss": losses[-1],
        "minimum_epoch_loss": min(losses), "checkpoint_sha256": sha256_file(checkpoint_path),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    for key in ("epochs", "dim", "num_filters", "batch_size", "log_every"):
        if getattr(args, key) < 1:
            raise ValueError(f"--{key.replace('_', '-')} must be positive")
    if args.lr <= 0 or args.l2_regularization < 0 or not 0 <= args.dropout < 1:
        raise ValueError("Invalid ConvKB optimization/dropout value")
    if args.seed < 0:
        raise ValueError("--seed must be nonnegative")
    paths = {
        "clean": args.output_dir / "clean_convkb.pt",
        "poisoned": args.output_dir / "poisoned_convkb.pt",
        "clean_loss": args.output_dir / "clean_training_loss.csv",
        "poisoned_loss": args.output_dir / "poisoned_training_loss.csv",
        "summary": args.output_dir / "training_summary.json",
    }
    for name, path in paths.items():
        if path.exists() and not args.overwrite:
            raise FileExistsError(f"{name} output already exists: {path}. Pass --overwrite to retrain.")
    entities, relations = read_names(args.entities), read_names(args.relations)
    if not entities or len(set(entities)) != len(entities):
        raise ValueError("Entity dictionary is empty or contains duplicates")
    if not relations or len(set(relations)) != len(relations):
        raise ValueError("Relation dictionary is empty or contains duplicates")
    entity_to_id = {name: index for index, name in enumerate(entities)}
    relation_to_id = {name: index for index, name in enumerate(relations)}
    clean, poisoned, integrity = verify_datasets(
        args.clean_train, args.poisoned_train, args.injected,
        args.valid, args.test, args.targets, entities, relations,
    )
    print("Dataset integrity passed:",
          f"clean={len(clean)}, poisoned={len(poisoned)}, injected={integrity['unique_injected_triples']}",
          flush=True)
    common = dict(
        epochs=args.epochs, dim=args.dim, num_filters=args.num_filters, dropout=args.dropout,
        batch_size=args.batch_size, lr=args.lr, l2_regularization=args.l2_regularization,
        seed=args.seed, log_every=args.log_every,
    )
    clean_model = train_one_victim(
        "clean", clean, entity_to_id, relation_to_id,
        checkpoint_path=paths["clean"], loss_path=paths["clean_loss"], **common,
    )
    poisoned_model = train_one_victim(
        "poisoned", poisoned, entity_to_id, relation_to_id,
        checkpoint_path=paths["poisoned"], loss_path=paths["poisoned_loss"], **common,
    )
    same_initial_weights = clean_model["initial_state_sha256"] == poisoned_model["initial_state_sha256"]
    if not same_initial_weights:
        raise AssertionError("ConvKB victims did not start from identical initialized weights")
    report = {
        "experiment": f"{args.attack_name} UMLS pilot — matched clean and poisoned ConvKB victims",
        "attack_name": args.attack_name, "victim_model": "ConvKB",
        "parameters": {
            "model": "ConvKB", "embedding_dim": args.dim, "num_filters": args.num_filters,
            "dropout": args.dropout, "epochs": args.epochs, "batch_size": args.batch_size,
            "learning_rate": args.lr, "l2_regularization": args.l2_regularization,
            "random_seed": args.seed, "optimizer": "Adam",
            "loss": "positive/negative logistic loss plus batch L2 regularization",
            "negative_sampling": "uniform random entity replacing head/tail with probability 0.5",
            "device": "cpu", "same_entity_relation_id_mappings": True,
            "independent_training_from_scratch": True,
            "parameter_source": "Local UMLS defaults; not ConvKB or MaSS paper hyperparameters.",
        },
        "datasets": {
            "clean_train_file": str(args.clean_train), "poisoned_train_file": str(args.poisoned_train),
            "injected_triples_file": str(args.injected), "validation_file": str(args.valid),
            "test_file": str(args.test), "set_B_file": str(args.targets), **integrity,
        },
        "victims": {"clean": clean_model, "poisoned": poisoned_model},
        "same_initial_embeddings_verified": same_initial_weights,
        "notes": [
            "ConvKB is trained from scratch; the TransE attacker surrogate is not a victim initialization.",
            "The frozen Set-B protocol was selected from TransE ranks, not ConvKB ranks.",
            "Use the matching filtered RQ1/RQ2 evaluator for attack results.",
        ],
    }
    paths["summary"].parent.mkdir(parents=True, exist_ok=True)
    paths["summary"].write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("\n=== CONVKB VICTIM TRAINING FINISHED ===")
    print(f"Identical initial embeddings: {same_initial_weights}")
    print(f"Clean final training loss: {clean_model['final_epoch_loss']:.6f}")
    print(f"Poisoned final training loss: {poisoned_model['final_epoch_loss']:.6f}")
    print(f"Output directory: {args.output_dir.resolve()}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Train paired ConvKB UMLS victims")
    parser.add_argument("--clean-train", type=Path, default=DATASET_DIR / "train.txt")
    parser.add_argument("--poisoned-train", type=Path, default=DEFAULT_POISONING_DIR / "poisoned_train.txt")
    parser.add_argument("--injected", type=Path, default=DEFAULT_POISONING_DIR / "injected_triples.tsv")
    parser.add_argument("--valid", type=Path, default=DATASET_DIR / "valid.txt")
    parser.add_argument("--test", type=Path, default=DATASET_DIR / "test.txt")
    parser.add_argument("--targets", type=Path, default=PILOT_DIR / "setup" / "set_B_target_triples.tsv")
    parser.add_argument("--entities", type=Path, default=DATASET_DIR / "entities.txt")
    parser.add_argument("--relations", type=Path, default=DATASET_DIR / "relations.txt")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--attack-name", default="MaSS")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--num-filters", type=int, default=16)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--l2-regularization", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
