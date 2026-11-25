"""Train and score native DICE knowledge graph embedding models."""

from __future__ import annotations

import os
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from dicee import KGE


MODEL_NAMES = ("TransE", "ComplEx", "RotatE")


class DiceModel:
    """Expose DICE scores and relation composition by graph labels."""

    def __init__(self, kge: KGE, model_name: str):
        self.kge = kge
        self.model_name = model_name
        self.entity_to_id = kge.entity_to_idx
        self.relation_to_id = kge.relation_to_idx

    def triple_scores(self, triples: Sequence[tuple[str, str, str]]) -> list[float]:
        """Return native DICE logits in input order."""
        if not triples:
            return []
        heads, relations, tails = zip(*triples)
        scores = self.kge.triple_score(
            h=list(heads), r=list(relations), t=list(tails), logits=True
        )
        return scores.detach().cpu().tolist()

    def triple_score(self, head: str, relation: str, tail: str) -> float:
        """Use native logits, removing the fixed margin for distance-based models."""
        score = self.triple_scores([(head, relation, tail)])[0]
        if self.model_name in ("TransE", "RotatE"):
            return score - self.kge.model.margin
        return score

    def relation_score(self, target: str, first: str, second: str) -> float:
        """Measure two-relation composition using the trained DICE embeddings."""
        target_vector, first_vector, second_vector = self.kge.get_relation_embeddings(
            [target, first, second]
        ).detach()
        if self.model_name == "ComplEx":
            target_real, target_imag = target_vector.chunk(2)
            first_real, first_imag = first_vector.chunk(2)
            second_real, second_imag = second_vector.chunk(2)
            product_real = first_real * second_real - first_imag * second_imag
            product_imag = first_real * second_imag + first_imag * second_real
            squared_distance = (target_real - product_real).square() + (target_imag - product_imag).square()
            return -squared_distance.sum().sqrt().item()
        difference = target_vector - first_vector - second_vector
        if self.model_name == "RotatE":
            # Chord distance compares the composed and target unit rotations.
            difference = 2 * (difference / 2).sin()
        return -difference.norm(p=2).item()


def train_dice_model(
    dataset_dir: Path,
    checkpoint_dir: Path,
    *,
    model_name: str,
    epochs: int,
    embedding_dim: int,
    batch_size: int,
    lr: float,
    seed: int,
) -> DiceModel:
    """Train a native DICE model and reload its checkpoint for inference."""
    if checkpoint_dir.exists():
        raise FileExistsError(f"Checkpoint directory already exists: {checkpoint_dir}")
    checkpoint_dir.parent.mkdir(parents=True, exist_ok=True)

    # Small CPU batches run faster without parallelizing each tensor operation.
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    from dicee import Execute, KGE
    from dicee.config import Namespace
    from dicee.static_funcs import MODEL_REGISTRY

    if model_name not in MODEL_NAMES or model_name not in MODEL_REGISTRY:
        raise ValueError(f"Native DICE model {model_name!r} is unavailable. Use dicee 0.3.3 for all three models.")

    args = Namespace()
    args.dataset_dir = str(dataset_dir.resolve())
    args.path_to_store_single_run = str(checkpoint_dir.resolve())
    args.model = model_name
    args.trainer = "torchCPUTrainer"
    args.scoring_technique = "NegSample"
    args.neg_ratio = 1
    args.eval_model = None
    args.num_epochs = epochs
    args.embedding_dim = embedding_dim
    args.batch_size = batch_size
    args.lr = lr
    args.random_seed = seed
    args.num_core = 0
    Execute(args).start()
    return DiceModel(KGE(path=args.path_to_store_single_run), model_name)
