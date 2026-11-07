#!/usr/bin/env python3
"""Train matched clean and poisoned ConvE victims for the UMLS attack runs.

The attacker remains the frozen TransE surrogate.  This script trains ConvE
from scratch on the clean and poisoned graphs with identical initial weights,
then records hashes that the ConvE evaluator verifies before reporting RQ1/RQ2.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from torch.nn import functional as F

MASS_DIR = Path(__file__).resolve().parents[1]
if str(MASS_DIR) not in sys.path:
    sys.path.insert(0, str(MASS_DIR))

from Path_Generation_Module.ConvE_Victim import ConvE  # noqa: E402
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
DEFAULT_OUTPUT_DIR = PILOT_DIR / "conve_victim_models"


def build_query_targets(
    triples: list[tuple[str, str, str]],
    entity_to_id: dict[str, int],
    relation_to_id: dict[str, int],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build deterministic (h,r)->multi-hot tail examples for 1-vs-all BCE."""
    query_tails: dict[tuple[int, int], set[int]] = defaultdict(set)
    for head, relation, tail in triples:
        query_tails[(entity_to_id[head], relation_to_id[relation])].add(entity_to_id[tail])
    ordered_queries = sorted(query_tails)
    heads = torch.tensor([head for head, _ in ordered_queries], dtype=torch.long)
    relations = torch.tensor([relation for _, relation in ordered_queries], dtype=torch.long)
    labels = torch.zeros((len(ordered_queries), len(entity_to_id)), dtype=torch.float32)
    for index, query in enumerate(ordered_queries):
        labels[index, sorted(query_tails[query])] = 1.0
    return heads, relations, labels


def train_one_victim(
    label: str,
    triples: list[tuple[str, str, str]],
    entity_to_id: dict[str, int],
    relation_to_id: dict[str, int],
    *,
    epochs: int,
    dim: int,
    embedding_height: int,
    num_filters: int,
    kernel_size: int,
    input_dropout: float,
    feature_dropout: float,
    hidden_dropout: float,
    label_smoothing: float,
    batch_size: int,
    lr: float,
    seed: int,
    checkpoint_path: Path,
    loss_csv_path: Path,
    log_every: int,
) -> dict[str, Any]:
    set_seed(seed)  # Reset before both victims to prove an equal starting point.
    model = ConvE(
        len(entity_to_id), len(relation_to_id), embedding_dim=dim,
        embedding_height=embedding_height, num_filters=num_filters,
        kernel_size=kernel_size, input_dropout=input_dropout,
        feature_dropout=feature_dropout, hidden_dropout=hidden_dropout,
    )
    initial_state_sha256 = sha256_initial_state(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    heads, relations, labels = build_query_targets(triples, entity_to_id, relation_to_id)
    if len(heads) < 2:
        raise ValueError("ConvE requires at least two distinct (head, relation) training queries")
    losses: list[float] = []
    print(f"\n[{label}] Training ConvE on {len(triples)} triples / {len(heads)} queries", flush=True)
    for epoch in range(1, epochs + 1):
        model.train()
        indices = torch.randperm(len(heads))
        total_loss = 0.0
        total_examples = 0
        for batch_indices in indices.split(batch_size):
            # BatchNorm1d needs two examples in training mode.  The selected
            # UMLS configuration has no single-item batch; retain this guard
            # to make alternate batch sizes safe and deterministic.
            if len(batch_indices) == 1:
                continue
            batch_labels = labels[batch_indices]
            if label_smoothing:
                batch_labels = ((1.0 - label_smoothing) * batch_labels +
                                label_smoothing / len(entity_to_id))
            logits = model(heads[batch_indices], relations[batch_indices])
            loss = F.binary_cross_entropy_with_logits(logits, batch_labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * len(batch_indices)
            total_examples += len(batch_indices)
        if not total_examples:
            raise ValueError("No ConvE training batch was large enough for BatchNorm")
        epoch_loss = total_loss / total_examples
        losses.append(epoch_loss)
        if epoch == 1 or epoch == epochs or epoch % log_every == 0:
            print(f"[{label}] Epoch {epoch:4d}/{epochs}: loss={epoch_loss:.6f}", flush=True)

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_type": "ConvE",
        "state_dict": model.state_dict(),
        "entity_to_id": entity_to_id,
        "relation_to_id": relation_to_id,
        "model_config": model.model_config(),
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
        "training_queries": len(heads),
        "initial_state_sha256": initial_state_sha256,
        "first_epoch_loss": losses[0],
        "final_epoch_loss": losses[-1],
        "minimum_epoch_loss": min(losses),
        "checkpoint_sha256": sha256_file(checkpoint_path),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    for key in ("epochs", "dim", "embedding_height", "num_filters", "kernel_size", "batch_size", "log_every"):
        if getattr(args, key) < 1:
            raise ValueError(f"--{key.replace('_', '-')} must be positive")
    if args.dim % args.embedding_height:
        raise ValueError("--dim must be divisible by --embedding-height")
    if args.lr <= 0:
        raise ValueError("--lr must be positive")
    for name in ("input_dropout", "feature_dropout", "hidden_dropout", "label_smoothing"):
        value = getattr(args, name)
        if not 0.0 <= value < 1.0:
            raise ValueError(f"--{name.replace('_', '-')} must be in [0, 1)")
    if args.seed < 0:
        raise ValueError("--seed must be nonnegative")
    paths = {
        "clean": args.output_dir / "clean_conve.pt",
        "poisoned": args.output_dir / "poisoned_conve.pt",
        "clean_loss": args.output_dir / "clean_training_loss.csv",
        "poisoned_loss": args.output_dir / "poisoned_training_loss.csv",
        "summary": args.output_dir / "training_summary.json",
    }
    for name, path in paths.items():
        if path.exists() and not args.overwrite:
            raise FileExistsError(f"{name} output already exists: {path}. Pass --overwrite to retrain.")
    entities, relations = read_names(args.entities), read_names(args.relations)
    if not entities or len(set(entities)) != len(entities):
        raise ValueError("Entity ID list is empty or contains duplicates")
    if not relations or len(set(relations)) != len(relations):
        raise ValueError("Relation ID list is empty or contains duplicates")
    entity_to_id = {name: index for index, name in enumerate(entities)}
    relation_to_id = {name: index for index, name in enumerate(relations)}
    clean, poisoned, integrity = verify_datasets(
        args.clean_train, args.poisoned_train, args.injected,
        args.valid, args.test, args.targets, entities, relations,
    )
    config = {
        "model": "ConvE", "epochs": args.epochs, "embedding_dim": args.dim,
        "embedding_height": args.embedding_height, "embedding_width": args.dim // args.embedding_height,
        "num_filters": args.num_filters, "kernel_size": args.kernel_size,
        "input_dropout": args.input_dropout, "feature_dropout": args.feature_dropout,
        "hidden_dropout": args.hidden_dropout, "label_smoothing": args.label_smoothing,
        "batch_size": args.batch_size, "learning_rate": args.lr, "random_seed": args.seed,
        "optimizer": "Adam", "loss": "1-vs-all binary cross entropy with logits",
        "device": "cpu", "independent_training_from_scratch": True,
        "same_entity_relation_id_mappings": True,
        "parameter_source": "Local UMLS defaults; not ConvE hyperparameters specified by the MaSS paper.",
    }
    print("Dataset integrity passed:",
          f"clean={len(clean)}, poisoned={len(poisoned)}, injected={integrity['unique_injected_triples']}",
          flush=True)
    common = dict(
        epochs=args.epochs, dim=args.dim, embedding_height=args.embedding_height,
        num_filters=args.num_filters, kernel_size=args.kernel_size,
        input_dropout=args.input_dropout, feature_dropout=args.feature_dropout,
        hidden_dropout=args.hidden_dropout, label_smoothing=args.label_smoothing,
        batch_size=args.batch_size, lr=args.lr, seed=args.seed, log_every=args.log_every,
    )
    clean_model = train_one_victim(
        "clean", clean, entity_to_id, relation_to_id,
        checkpoint_path=paths["clean"], loss_csv_path=paths["clean_loss"], **common,
    )
    poisoned_model = train_one_victim(
        "poisoned", poisoned, entity_to_id, relation_to_id,
        checkpoint_path=paths["poisoned"], loss_csv_path=paths["poisoned_loss"], **common,
    )
    same_initial_weights = clean_model["initial_state_sha256"] == poisoned_model["initial_state_sha256"]
    if not same_initial_weights:
        raise AssertionError("ConvE victims did not start from identical initialized weights")
    report = {
        "experiment": f"{args.attack_name} UMLS pilot — matched clean and poisoned ConvE victims",
        "attack_name": args.attack_name, "victim_model": "ConvE", "parameters": config,
        "datasets": {
            "clean_train_file": str(args.clean_train), "poisoned_train_file": str(args.poisoned_train),
            "injected_triples_file": str(args.injected), "validation_file": str(args.valid),
            "test_file": str(args.test), "set_B_file": str(args.targets), **integrity,
        },
        "victims": {"clean": clean_model, "poisoned": poisoned_model},
        "same_initial_embeddings_verified": same_initial_weights,
        "notes": [
            "ConvE is trained from scratch; the TransE attacker surrogate is not a victim initialization.",
            "The frozen Set-B protocol was selected from TransE ranks, not ConvE ranks.",
            "Training loss is not an attack-success metric; use the paired RQ1/RQ2 evaluator.",
        ],
    }
    paths["summary"].parent.mkdir(parents=True, exist_ok=True)
    paths["summary"].write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("\n=== CONVE VICTIM TRAINING FINISHED ===")
    print(f"Identical initial embeddings: {same_initial_weights}")
    print(f"Clean final training loss: {clean_model['final_epoch_loss']:.6f}")
    print(f"Poisoned final training loss: {poisoned_model['final_epoch_loss']:.6f}")
    print(f"Output directory: {args.output_dir.resolve()}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Train matched clean and poisoned ConvE UMLS victims")
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
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--embedding-height", type=int, default=8)
    parser.add_argument("--num-filters", type=int, default=32)
    parser.add_argument("--kernel-size", type=int, default=3)
    parser.add_argument("--input-dropout", type=float, default=0.2)
    parser.add_argument("--feature-dropout", type=float, default=0.2)
    parser.add_argument("--hidden-dropout", type=float, default=0.3)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
