"""Local TransE attacker architecture and checkpoint scoring for Test_Triplet.

Training is orchestrated by run_experiment.py using explicit input paths.
Scores are negative L2 distances: higher values indicate greater plausibility.
"""
from __future__ import annotations

from pathlib import Path

import torch
from torch import nn


class TransE(nn.Module):
    """TransE with checkpoint-compatible entity and relation embeddings."""

    def __init__(self, num_entities: int, num_relations: int, dim: int = 64):
        super().__init__()
        self.entity = nn.Embedding(num_entities, dim)
        self.relation = nn.Embedding(num_relations, dim)
        nn.init.xavier_uniform_(self.entity.weight)
        nn.init.xavier_uniform_(self.relation.weight)

    def distance(self, heads: torch.Tensor, relations: torch.Tensor,
                 tails: torch.Tensor) -> torch.Tensor:
        return torch.linalg.vector_norm(
            self.entity(heads) + self.relation(relations) - self.entity(tails),
            ord=2, dim=-1,
        )

    def score(self, heads: torch.Tensor, relations: torch.Tensor,
              tails: torch.Tensor) -> torch.Tensor:
        return -self.distance(heads, relations, tails)


class Surrogate:
    """A trained attacker surrogate with named-triplet and relation-path scores."""

    def __init__(self, model: TransE, entity_to_id: dict[str, int],
                 relation_to_id: dict[str, int]):
        self.model = model.cpu().eval()
        self.entity_to_id = entity_to_id
        self.relation_to_id = relation_to_id

    @classmethod
    def load(cls, checkpoint_path: Path) -> "Surrogate":
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        model = TransE(
            len(checkpoint["entity_to_id"]),
            len(checkpoint["relation_to_id"]),
            checkpoint["embedding_dim"],
        )
        model.load_state_dict(checkpoint["state_dict"])
        return cls(model, checkpoint["entity_to_id"], checkpoint["relation_to_id"])

    @torch.no_grad()
    def relation_score(self, target: str, r1: str, r2: str) -> float:
        """Return -||r - r1 - r2||_2 for a two-relation path template."""
        embeddings = self.model.relation.weight
        target_vector = embeddings[self.relation_to_id[target]]
        first = embeddings[self.relation_to_id[r1]]
        second = embeddings[self.relation_to_id[r2]]
        return float(-torch.linalg.vector_norm(target_vector - first - second, ord=2))

    @torch.no_grad()
    def triple_score(self, head: str, relation: str, tail: str) -> float:
        heads = torch.tensor([self.entity_to_id[head]], dtype=torch.long)
        relations = torch.tensor([self.relation_to_id[relation]], dtype=torch.long)
        tails = torch.tensor([self.entity_to_id[tail]], dtype=torch.long)
        return float(self.model.score(heads, relations, tails).item())
