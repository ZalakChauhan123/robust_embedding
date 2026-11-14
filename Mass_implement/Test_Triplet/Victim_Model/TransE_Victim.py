"""Local independent TransE victim architecture and loader for Test_Triplet.

The clean and poisoned victims are trained as separate instances. Initialization,
negative L2 scoring and checkpoint keys match the existing TransE implementation.
"""
from __future__ import annotations

from pathlib import Path

import torch
from torch import nn


class TransE(nn.Module):
    """Victim TransE model with plausibility score -||h + r - t||_2."""

    def __init__(self, num_entities: int, num_relations: int, dim: int = 64):
        super().__init__()
        self.entity = nn.Embedding(num_entities, dim)
        self.relation = nn.Embedding(num_relations, dim)
        nn.init.xavier_uniform_(self.entity.weight)
        nn.init.xavier_uniform_(self.relation.weight)

    def distance(self, heads: torch.Tensor, relations: torch.Tensor,
                 tails: torch.Tensor) -> torch.Tensor:
        """Return the target's L2 translation distance using this victim's weights."""
        return torch.linalg.vector_norm(
            self.entity(heads) + self.relation(relations) - self.entity(tails),
            ord=2, dim=-1,
        )

    def score(self, heads: torch.Tensor, relations: torch.Tensor,
              tails: torch.Tensor) -> torch.Tensor:
        """Return raw plausibility; higher (closer to zero) is better."""
        return -self.distance(heads, relations, tails)


class TransEVictim:
    """A trained victim and its entity/relation mappings, ready for scoring."""

    def __init__(self, model: TransE, entity_to_id: dict[str, int],
                 relation_to_id: dict[str, int]) -> None:
        self.model = model.cpu().eval()
        self.entity_to_id = entity_to_id
        self.relation_to_id = relation_to_id

    @classmethod
    def load(cls, checkpoint_path: Path) -> "TransEVictim":
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
        heads = torch.tensor([self.entity_to_id[head]], dtype=torch.long)
        relations = torch.tensor([self.relation_to_id[relation]], dtype=torch.long)
        tails = torch.tensor([self.entity_to_id[tail]], dtype=torch.long)
        return float(self.model.score(heads, relations, tails).item())
