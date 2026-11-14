"""TransE victim architecture and loader for clean/poisoned checkpoints.

This module owns the victim's embedding initialization, L2 distance and raw
plausibility calculation independently of the attacker surrogate. The numerical
operations and state-dict keys match existing checkpoints exactly. Training
belongs in Experiment_Setup/train_victim_models.py.
"""
from __future__ import annotations

from pathlib import Path

import torch
from torch import nn


class TransE(nn.Module):
    """Victim TransE model with score(h,r,t) = -||h + r - t||_2.

    Keep the original embedding creation and initialization order so seeded
    training and legacy ``entity.weight`` / ``relation.weight`` checkpoints
    remain compatible.
    """

    def __init__(self, num_entities: int, num_relations: int, dim: int = 64):
        super().__init__()
        self.entity = nn.Embedding(num_entities, dim)
        self.relation = nn.Embedding(num_relations, dim)
        nn.init.xavier_uniform_(self.entity.weight)
        nn.init.xavier_uniform_(self.relation.weight)

    def distance(self, heads: torch.Tensor, relations: torch.Tensor,
                 tails: torch.Tensor) -> torch.Tensor:
        """Return ||h + r - t||_2 using this victim's learned embeddings."""
        return torch.linalg.vector_norm(
            self.entity(heads) + self.relation(relations) - self.entity(tails),
            ord=2, dim=-1,
        )

    def score(self, heads: torch.Tensor, relations: torch.Tensor,
              tails: torch.Tensor) -> torch.Tensor:
        """Return the raw plausibility score; higher (closer to zero) is better."""
        return -self.distance(heads, relations, tails)


class TransEVictim:
    """A trained victim and its entity/relation mappings, ready for evaluation."""

    def __init__(self, model: TransE, entity_to_id: dict[str, int],
                 relation_to_id: dict[str, int]) -> None:
        self.model = model.cpu().eval()
        self.entity_to_id = entity_to_id
        self.relation_to_id = relation_to_id

    @classmethod
    def load(cls, checkpoint_path: Path) -> "TransEVictim":
        """Load existing clean/poisoned TransE checkpoint dictionaries unchanged."""
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        model = TransE(
            len(checkpoint["entity_to_id"]),
            len(checkpoint["relation_to_id"]),
            checkpoint["embedding_dim"],
        )
        model.load_state_dict(checkpoint["state_dict"])
        return cls(model, checkpoint["entity_to_id"], checkpoint["relation_to_id"])

    @torch.no_grad()
    def triple_score(self, head: str, relation: str, tail: str) -> float:
        """Return this victim's raw plausibility score for a named triplet."""
        heads = torch.tensor([self.entity_to_id[head]], dtype=torch.long)
        relations = torch.tensor([self.relation_to_id[relation]], dtype=torch.long)
        tails = torch.tensor([self.entity_to_id[tail]], dtype=torch.long)
        return float(self.model.score(heads, relations, tails).item())
