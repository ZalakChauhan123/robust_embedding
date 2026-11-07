#!/usr/bin/env python3
"""Train paired TransH or DistMult victims on clean and poisoned UMLS graphs."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Literal

import torch
from torch.nn import functional as F

MASS_DIR = Path(__file__).resolve().parents[1]
if str(MASS_DIR) not in sys.path:
    sys.path.insert(0, str(MASS_DIR))

from Path_Generation_Module.DistMult_Victim import DistMult  # noqa: E402
from Path_Generation_Module.TransH_Victim import TransH  # noqa: E402
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
DEFAULT_OUTPUT_DIR = PILOT_DIR / "shallow_victim_models"
ModelName = Literal["TransH", "DistMult"]


def build_query_targets(
    triples: list[tuple[str, str, str]],
    entity_to_id: dict[str, int],
    relation_to_id: dict[str, int],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create deterministic multi-label tail data for DistMult 1-vs-all BCE."""
    tails_by_query: dict[tuple[int, int], set[int]] = defaultdict(set)
    for head, relation, tail in triples:
        tails_by_query[(entity_to_id[head], relation_to_id[relation])].add(entity_to_id[tail])
    queries = sorted(tails_by_query)
    heads = torch.tensor([head for head, _ in queries], dtype=torch.long)
    relations = torch.tensor([relation for _, relation in queries], dtype=torch.long)
    labels = torch.zeros((len(queries), len(entity_to_id)), dtype=torch.float32)
    for index, query in enumerate(queries):
        labels[index, sorted(tails_by_query[query])] = 1.0
    return heads, relations, labels


def encode_triples(
    triples: list[tuple[str, str, str]],
    entity_to_id: dict[str, int],
    relation_to_id: dict[str, int],
) -> torch.Tensor:
    return torch.tensor(
        [(entity_to_id[head], relation_to_id[relation], entity_to_id[tail])
         for head, relation, tail in triples],
        dtype=torch.long,
    )


def train_transh(
    triples: list[tuple[str, str, str]],
    entity_to_id: dict[str, int],
    relation_to_id: dict[str, int],
    *,
    dim: int,
    epochs: int,
    batch_size: int,
    lr: float,
    margin: float,
    seed: int,
    log_label: str,
    log_every: int,
) -> tuple[TransH, list[float], str]:
    set_seed(seed)
    model = TransH(len(entity_to_id), len(relation_to_id), dim)
    initial_state = sha256_initial_state(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    encoded = encode_triples(triples, entity_to_id, relation_to_id)
    losses: list[float] = []
    print(f"\n[{log_label}] Training TransH on {len(encoded)} triples", flush=True)
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
            loss = F.relu(margin - positive_score + negative_score).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            model.normalize_embeddings()
            total_loss += float(loss.item()) * len(batch_indices)
        epoch_loss = total_loss / len(encoded)
        losses.append(epoch_loss)
        if epoch == 1 or epoch == epochs or epoch % log_every == 0:
            print(f"[{log_label}] Epoch {epoch:4d}/{epochs}: loss={epoch_loss:.6f}", flush=True)
    return model, losses, initial_state


def train_distmult(
    triples: list[tuple[str, str, str]],
    entity_to_id: dict[str, int],
    relation_to_id: dict[str, int],
    *,
    dim: int,
    epochs: int,
    batch_size: int,
    lr: float,
    label_smoothing: float,
    seed: int,
    log_label: str,
    log_every: int,
) -> tuple[DistMult, list[float], str]:
    set_seed(seed)
    model = DistMult(len(entity_to_id), len(relation_to_id), dim)
    initial_state = sha256_initial_state(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    heads, relations, labels = build_query_targets(triples, entity_to_id, relation_to_id)
    if len(heads) < 1:
        raise ValueError("DistMult requires at least one training query")
    losses: list[float] = []
    print(f"\n[{log_label}] Training DistMult on {len(triples)} triples / {len(heads)} queries", flush=True)
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_examples = 0
        for batch_indices in torch.randperm(len(heads)).split(batch_size):
            batch_labels = labels[batch_indices]
            if label_smoothing:
                batch_labels = ((1.0 - label_smoothing) * batch_labels +
                                label_smoothing / len(entity_to_id))
            logits = model.score_all_tails(heads[batch_indices], relations[batch_indices])
            loss = F.binary_cross_entropy_with_logits(logits, batch_labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * len(batch_indices)
            total_examples += len(batch_indices)
        epoch_loss = total_loss / total_examples
        losses.append(epoch_loss)
        if epoch == 1 or epoch == epochs or epoch % log_every == 0:
            print(f"[{log_label}] Epoch {epoch:4d}/{epochs}: loss={epoch_loss:.6f}", flush=True)
    return model, losses, initial_state


def save_victim(
    model: TransH | DistMult,
    losses: list[float],
    initial_state: str,
    entity_to_id: dict[str, int],
    relation_to_id: dict[str, int],
    checkpoint_path: Path,
    loss_path: Path,
    training_triples: int,
) -> dict[str, Any]:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_type": model.model_config()["model_type"], "state_dict": model.state_dict(),
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
        "training_triples": training_triples, "initial_state_sha256": initial_state,
        "first_epoch_loss": losses[0], "final_epoch_loss": losses[-1],
        "minimum_epoch_loss": min(losses), "checkpoint_sha256": sha256_file(checkpoint_path),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.model not in ("TransH", "DistMult"):
        raise ValueError("--model must be TransH or DistMult")
    for key in ("epochs", "dim", "batch_size", "log_every"):
        if getattr(args, key) < 1:
            raise ValueError(f"--{key.replace('_', '-')} must be positive")
    if args.transh_lr <= 0 or args.distmult_lr <= 0 or args.transh_margin <= 0:
        raise ValueError("Learning rates and TransH margin must be positive")
    if not 0.0 <= args.label_smoothing < 1.0:
        raise ValueError("--label-smoothing must be in [0, 1)")
    if args.seed < 0:
        raise ValueError("--seed must be nonnegative")
    model_key = args.model.lower()
    paths = {
        "clean": args.output_dir / f"clean_{model_key}.pt",
        "poisoned": args.output_dir / f"poisoned_{model_key}.pt",
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
    if args.model == "TransH":
        def train_dataset(label: str, triples: list[tuple[str, str, str]]):
            return train_transh(
                triples, entity_to_id, relation_to_id, dim=args.dim, epochs=args.epochs,
                batch_size=args.batch_size, lr=args.transh_lr, margin=args.transh_margin,
                seed=args.seed, log_label=label, log_every=args.log_every,
            )
        model_parameters: dict[str, Any] = {
            "model": "TransH", "embedding_dim": args.dim, "epochs": args.epochs,
            "batch_size": args.batch_size, "learning_rate": args.transh_lr,
            "margin": args.transh_margin,
            "loss": "mean ReLU(margin - positive_score + negative_score)",
            "negative_sampling": "uniform random entity replacing head/tail with probability 0.5",
            "normalization": "unit L2 entity, relation, and hyperplane-normal vectors after each step",
        }
    else:
        def train_dataset(label: str, triples: list[tuple[str, str, str]]):
            return train_distmult(
                triples, entity_to_id, relation_to_id, dim=args.dim, epochs=args.epochs,
                batch_size=args.batch_size, lr=args.distmult_lr, label_smoothing=args.label_smoothing,
                seed=args.seed, log_label=label, log_every=args.log_every,
            )
        model_parameters = {
            "model": "DistMult", "embedding_dim": args.dim, "epochs": args.epochs,
            "batch_size": args.batch_size, "learning_rate": args.distmult_lr,
            "label_smoothing": args.label_smoothing,
            "loss": "1-vs-all binary cross entropy with logits",
        }
    clean_model, clean_losses, clean_initial = train_dataset("clean", clean)
    poisoned_model, poisoned_losses, poisoned_initial = train_dataset("poisoned", poisoned)
    clean_meta = save_victim(clean_model, clean_losses, clean_initial, entity_to_id, relation_to_id,
                             paths["clean"], paths["clean_loss"], len(clean))
    poisoned_meta = save_victim(poisoned_model, poisoned_losses, poisoned_initial, entity_to_id, relation_to_id,
                                paths["poisoned"], paths["poisoned_loss"], len(poisoned))
    same_initial_weights = clean_initial == poisoned_initial
    if not same_initial_weights:
        raise AssertionError("Victims did not start from identical initialized weights")
    report = {
        "experiment": f"{args.attack_name} UMLS pilot — matched clean and poisoned {args.model} victims",
        "attack_name": args.attack_name, "victim_model": args.model,
        "parameters": {
            **model_parameters, "random_seed": args.seed, "optimizer": "Adam", "device": "cpu",
            "same_entity_relation_id_mappings": True, "independent_training_from_scratch": True,
            "parameter_source": "Local UMLS defaults; not victim hyperparameters specified by the MaSS paper.",
        },
        "datasets": {
            "clean_train_file": str(args.clean_train), "poisoned_train_file": str(args.poisoned_train),
            "injected_triples_file": str(args.injected), "validation_file": str(args.valid),
            "test_file": str(args.test), "set_B_file": str(args.targets), **integrity,
        },
        "victims": {"clean": clean_meta, "poisoned": poisoned_meta},
        "same_initial_embeddings_verified": same_initial_weights,
        "notes": [
            f"{args.model} is trained from scratch; the TransE attacker surrogate is not a victim initialization.",
            "The frozen Set-B protocol was selected from TransE ranks, not this victim model's ranks.",
            "Use the matching filtered RQ1/RQ2 evaluator for attack results.",
        ],
    }
    paths["summary"].parent.mkdir(parents=True, exist_ok=True)
    paths["summary"].write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\n=== {args.model.upper()} VICTIM TRAINING FINISHED ===")
    print(f"Identical initial embeddings: {same_initial_weights}")
    print(f"Clean final training loss: {clean_meta['final_epoch_loss']:.6f}")
    print(f"Poisoned final training loss: {poisoned_meta['final_epoch_loss']:.6f}")
    print(f"Output directory: {args.output_dir.resolve()}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Train paired TransH or DistMult UMLS victims")
    parser.add_argument("--model", choices=("TransH", "DistMult"), required=True)
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
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--transh-lr", type=float, default=0.01)
    parser.add_argument("--transh-margin", type=float, default=1.0)
    parser.add_argument("--distmult-lr", type=float, default=0.001)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
