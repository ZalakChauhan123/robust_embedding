#!/usr/bin/env python3
"""Train a local DeepPath-style two-hop policy victim for UMLS.

This is explicitly a local supervised path-policy reconstruction.  It is not
the original DeepPath TensorFlow/RL environment, which is absent from this
repository.  See the training summary's implementation_variant field.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import torch
from torch.nn import functional as F

MASS_DIR = Path(__file__).resolve().parents[1]
if str(MASS_DIR) not in sys.path:
    sys.path.insert(0, str(MASS_DIR))

from Path_Generation_Module.DeepPath_Victim import DeepPathLocal  # noqa: E402
from Path_Generation_Module.TransE_Surrogate import read_names  # noqa: E402
from Experiment_Setup.train_victim_models import (  # noqa: E402
    set_seed,
    sha256_file,
    verify_datasets,
)

DATASET_DIR = MASS_DIR / "dataset_uml"
PILOT_DIR = MASS_DIR / "outputs" / "Experiments" / "UMLS_pilot"
DEFAULT_POISONING_DIR = PILOT_DIR / "poisoning"
DEFAULT_OUTPUT_DIR = PILOT_DIR / "deeppath_victim_models"


def sha256_trainable_parameters(model: torch.nn.Module) -> str:
    """Hash only trainable state; clean/poison action tables legitimately differ."""
    digest = hashlib.sha256()
    for name, value in sorted(model.named_parameters()):
        digest.update(name.encode("utf-8"))
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def encode_edges(
    triples: list[tuple[str, str, str]], entity_to_id: dict[str, int], relation_to_id: dict[str, int]
) -> list[tuple[int, int, int]]:
    return [
        (entity_to_id[head], relation_to_id[relation], entity_to_id[tail])
        for head, relation, tail in triples
    ]


def build_teacher_examples(
    model: DeepPathLocal,
    triples: list[tuple[int, int, int]],
    *,
    max_paths_per_triple: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Extract deterministic non-trivial two-hop demonstrations from a KG."""
    current_examples: list[int] = []
    target_examples: list[int] = []
    query_examples: list[int] = []
    action_examples: list[int] = []
    for head, query_relation, tail in triples:
        selected_paths = 0
        first_mask = model.action_mask[head]
        for first_index in torch.nonzero(first_mask, as_tuple=False).flatten().tolist():
            middle = int(model.action_destination_ids[head, first_index].item())
            # Exclude direct edges and immediate backtracking demonstrations.
            if middle in (head, tail):
                continue
            second_mask = model.action_mask[middle]
            matching_second = torch.nonzero(
                second_mask & (model.action_destination_ids[middle] == tail), as_tuple=False
            ).flatten().tolist()
            if not matching_second:
                continue
            second_index = matching_second[0]
            current_examples.extend((head, middle))
            target_examples.extend((tail, tail))
            query_examples.extend((query_relation, query_relation))
            action_examples.extend((first_index, second_index))
            selected_paths += 1
            if selected_paths >= max_paths_per_triple:
                break
    if not current_examples:
        raise ValueError("No non-trivial two-hop path demonstrations were found")
    return tuple(torch.tensor(items, dtype=torch.long) for items in (
        current_examples, target_examples, query_examples, action_examples
    ))  # type: ignore[return-value]


def train_one_victim(
    label: str,
    triples: list[tuple[str, str, str]],
    entity_to_id: dict[str, int],
    relation_to_id: dict[str, int],
    *,
    epochs: int,
    dim: int,
    hidden_dim: int,
    dropout: float,
    beam_size: int,
    max_paths_per_triple: int,
    batch_size: int,
    lr: float,
    seed: int,
    checkpoint_path: Path,
    loss_path: Path,
    log_every: int,
) -> dict[str, Any]:
    encoded = encode_edges(triples, entity_to_id, relation_to_id)
    max_actions = DeepPathLocal.max_actions_for_edges(encoded, len(entity_to_id), len(relation_to_id))
    set_seed(seed)
    model = DeepPathLocal(
        len(entity_to_id), len(relation_to_id), dim, hidden_dim, max_actions, dropout, beam_size
    )
    model.set_graph(encoded)
    initial_trainable_state = sha256_trainable_parameters(model)
    current, target, query, action = build_teacher_examples(
        model, encoded, max_paths_per_triple=max_paths_per_triple
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    losses: list[float] = []
    print(
        f"\n[{label}] Training DeepPath-style policy on {len(triples)} triples / "
        f"{len(current)} two-hop demonstrations / max_actions={max_actions}", flush=True,
    )
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for batch_indices in torch.randperm(len(current)).split(batch_size):
            logits = model.policy_logits(current[batch_indices], target[batch_indices], query[batch_indices])
            loss = F.cross_entropy(logits, action[batch_indices])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * len(batch_indices)
        epoch_loss = total_loss / len(current)
        losses.append(epoch_loss)
        if epoch == 1 or epoch == epochs or epoch % log_every == 0:
            print(f"[{label}] Epoch {epoch:4d}/{epochs}: loss={epoch_loss:.6f}", flush=True)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_type": "DeepPathLocal", "state_dict": model.state_dict(),
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
        "training_triples": len(triples), "training_demonstrations": len(current),
        "max_actions": max_actions, "initial_trainable_state_sha256": initial_trainable_state,
        "first_epoch_loss": losses[0], "final_epoch_loss": losses[-1],
        "minimum_epoch_loss": min(losses), "checkpoint_sha256": sha256_file(checkpoint_path),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    for key in ("epochs", "dim", "hidden_dim", "beam_size", "max_paths_per_triple", "batch_size", "log_every"):
        if getattr(args, key) < 1:
            raise ValueError(f"--{key.replace('_', '-')} must be positive")
    if args.lr <= 0 or not 0 <= args.dropout < 1 or args.seed < 0:
        raise ValueError("Invalid DeepPath-style optimizer, dropout, or seed value")
    paths = {
        "clean": args.output_dir / "clean_deeppath.pt",
        "poisoned": args.output_dir / "poisoned_deeppath.pt",
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
        epochs=args.epochs, dim=args.dim, hidden_dim=args.hidden_dim, dropout=args.dropout,
        beam_size=args.beam_size, max_paths_per_triple=args.max_paths_per_triple,
        batch_size=args.batch_size, lr=args.lr, seed=args.seed, log_every=args.log_every,
    )
    clean_model = train_one_victim(
        "clean", clean, entity_to_id, relation_to_id,
        checkpoint_path=paths["clean"], loss_path=paths["clean_loss"], **common,
    )
    poisoned_model = train_one_victim(
        "poisoned", poisoned, entity_to_id, relation_to_id,
        checkpoint_path=paths["poisoned"], loss_path=paths["poisoned_loss"], **common,
    )
    same_initial_weights = (
        clean_model["initial_trainable_state_sha256"] == poisoned_model["initial_trainable_state_sha256"]
    )
    if not same_initial_weights:
        raise AssertionError("DeepPath-style victims did not start from identical trainable weights")
    report = {
        "experiment": f"{args.attack_name} UMLS pilot — paired local DeepPath-style path-policy victims",
        "attack_name": args.attack_name, "victim_model": "DeepPath",
        "implementation_variant": "local_supervised_two_hop_path_policy_not_official_DeepPath",
        "parameters": {
            "embedding_dim": args.dim, "hidden_dim": args.hidden_dim, "dropout": args.dropout,
            "beam_size": args.beam_size, "max_hops": 2,
            "max_teacher_paths_per_training_triple": args.max_paths_per_triple,
            "epochs": args.epochs, "batch_size": args.batch_size, "learning_rate": args.lr,
            "random_seed": args.seed, "optimizer": "Adam", "loss": "masked local-action cross entropy",
            "device": "cpu", "independent_training_from_scratch": True,
            "same_entity_relation_id_mappings": True,
        },
        "datasets": {
            "clean_train_file": str(args.clean_train), "poisoned_train_file": str(args.poisoned_train),
            "injected_triples_file": str(args.injected), "validation_file": str(args.valid),
            "test_file": str(args.test), "set_B_file": str(args.targets), **integrity,
        },
        "victims": {"clean": clean_model, "poisoned": poisoned_model},
        # Evaluator's generic provenance check consumes this standard key.
        "same_initial_embeddings_verified": same_initial_weights,
        "notes": [
            "This local supervised two-hop path policy is not the official TensorFlow/RL DeepPath reproduction.",
            "The frozen Set-B protocol was selected from TransE ranks, not DeepPath-style path-policy ranks.",
            "Graph action-table buffers differ correctly between clean and poisoned graphs; only trainable weights are matched.",
        ],
    }
    paths["summary"].parent.mkdir(parents=True, exist_ok=True)
    paths["summary"].write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("\n=== DEEPPATH-STYLE VICTIM TRAINING FINISHED ===")
    print(f"Identical initial trainable weights: {same_initial_weights}")
    print(f"Clean final training loss: {clean_model['final_epoch_loss']:.6f}")
    print(f"Poisoned final training loss: {poisoned_model['final_epoch_loss']:.6f}")
    print(f"Output directory: {args.output_dir.resolve()}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Train paired local DeepPath-style UMLS path-policy victims")
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
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--beam-size", type=int, default=10)
    parser.add_argument("--max-paths-per-triple", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=0.003)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
