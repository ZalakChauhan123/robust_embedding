"""Pure DistMult victim model used for opaque-box UMLS attack evaluation."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn


class DistMult(nn.Module):
    """DistMult score(h,r,t)=sum(h*r*t), with higher scores more plausible."""

    def __init__(self, num_entities: int, num_relations: int, embedding_dim: int = 64) -> None:
        super().__init__()
        if num_entities < 1 or num_relations < 1 or embedding_dim < 1:
            raise ValueError("DistMult needs positive entity/relation counts and embedding dimension")
        self.embedding_dim = embedding_dim
        self.entity = nn.Embedding(num_entities, embedding_dim)
        self.relation = nn.Embedding(num_relations, embedding_dim)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.entity.weight)
        nn.init.xavier_uniform_(self.relation.weight)

    def score_triples(self, heads: torch.Tensor, relations: torch.Tensor, tails: torch.Tensor) -> torch.Tensor:
        return (self.entity(heads) * self.relation(relations) * self.entity(tails)).sum(dim=-1)

    def score_all_tails(self, heads: torch.Tensor, relations: torch.Tensor) -> torch.Tensor:
        """Return higher-is-better scores with shape ``[batch, entities]``."""
        query = self.entity(heads) * self.relation(relations)
        return query @ self.entity.weight.transpose(0, 1)

    def model_config(self) -> dict[str, Any]:
        return {"model_type": "DistMult", "embedding_dim": self.embedding_dim}

    @classmethod
    def load(cls, checkpoint_path: Path) -> tuple["DistMult", dict[str, int], dict[str, int], dict[str, Any]]:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if checkpoint.get("model_type") != "DistMult":
            raise ValueError(f"Not a DistMult checkpoint: {checkpoint_path}")
        entity_to_id = checkpoint["entity_to_id"]
        relation_to_id = checkpoint["relation_to_id"]
        config = checkpoint["model_config"]
        model = cls(len(entity_to_id), len(relation_to_id), int(config["embedding_dim"]))
        model.load_state_dict(checkpoint["state_dict"])
        return model.cpu().eval(), entity_to_id, relation_to_id, config
