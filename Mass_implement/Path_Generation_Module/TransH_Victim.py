"""TransH victim model used for opaque-box UMLS attack evaluation."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


class TransH(nn.Module):
    """TransH with relation-specific hyperplane projection and L2 scoring."""

    def __init__(self, num_entities: int, num_relations: int, embedding_dim: int = 64) -> None:
        super().__init__()
        if num_entities < 1 or num_relations < 1 or embedding_dim < 1:
            raise ValueError("TransH needs positive entity/relation counts and embedding dimension")
        self.embedding_dim = embedding_dim
        self.entity = nn.Embedding(num_entities, embedding_dim)
        self.relation = nn.Embedding(num_relations, embedding_dim)
        self.relation_normal = nn.Embedding(num_relations, embedding_dim)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.entity.weight)
        nn.init.xavier_uniform_(self.relation.weight)
        nn.init.xavier_uniform_(self.relation_normal.weight)
        self.normalize_embeddings()

    @staticmethod
    def project(entity: torch.Tensor, normal: torch.Tensor) -> torch.Tensor:
        normal = F.normalize(normal, p=2, dim=-1)
        return entity - (entity * normal).sum(dim=-1, keepdim=True) * normal

    def score_triples(self, heads: torch.Tensor, relations: torch.Tensor, tails: torch.Tensor) -> torch.Tensor:
        normal = self.relation_normal(relations)
        head = self.project(self.entity(heads), normal)
        tail = self.project(self.entity(tails), normal)
        return -torch.linalg.vector_norm(head + self.relation(relations) - tail, ord=2, dim=-1)

    def score_all_tails(self, heads: torch.Tensor, relations: torch.Tensor) -> torch.Tensor:
        """Return higher-is-better tail scores with shape ``[batch, entities]``."""
        normal = F.normalize(self.relation_normal(relations), p=2, dim=-1)
        head = self.entity(heads)
        head = head - (head * normal).sum(dim=-1, keepdim=True) * normal
        tail_embeddings = self.entity.weight.unsqueeze(0)
        tail = tail_embeddings - (tail_embeddings * normal.unsqueeze(1)).sum(dim=-1, keepdim=True) * normal.unsqueeze(1)
        distance = torch.linalg.vector_norm(
            head.unsqueeze(1) + self.relation(relations).unsqueeze(1) - tail,
            ord=2, dim=-1,
        )
        return -distance

    @torch.no_grad()
    def normalize_embeddings(self) -> None:
        """Maintain TransH's unit-vector constraints after each optimizer step."""
        self.entity.weight.copy_(F.normalize(self.entity.weight, p=2, dim=1))
        self.relation.weight.copy_(F.normalize(self.relation.weight, p=2, dim=1))
        self.relation_normal.weight.copy_(F.normalize(self.relation_normal.weight, p=2, dim=1))

    def model_config(self) -> dict[str, Any]:
        return {"model_type": "TransH", "embedding_dim": self.embedding_dim, "distance": "L2"}

    @classmethod
    def load(cls, checkpoint_path: Path) -> tuple["TransH", dict[str, int], dict[str, int], dict[str, Any]]:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if checkpoint.get("model_type") != "TransH":
            raise ValueError(f"Not a TransH checkpoint: {checkpoint_path}")
        entity_to_id = checkpoint["entity_to_id"]
        relation_to_id = checkpoint["relation_to_id"]
        config = checkpoint["model_config"]
        model = cls(len(entity_to_id), len(relation_to_id), int(config["embedding_dim"]))
        model.load_state_dict(checkpoint["state_dict"])
        return model.cpu().eval(), entity_to_id, relation_to_id, config
