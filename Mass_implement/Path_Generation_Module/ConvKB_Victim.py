"""ConvKB victim model for opaque-box UMLS poisoning evaluation."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


class ConvKB(nn.Module):
    """ConvKB scores an embedding matrix whose columns are (head, relation, tail)."""

    def __init__(
        self,
        num_entities: int,
        num_relations: int,
        embedding_dim: int = 64,
        num_filters: int = 16,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if num_entities < 1 or num_relations < 1 or embedding_dim < 1 or num_filters < 1:
            raise ValueError("ConvKB needs positive entity/relation counts, dimension, and filter count")
        self.embedding_dim = embedding_dim
        self.num_filters = num_filters
        self.dropout_probability = dropout
        self.entity = nn.Embedding(num_entities, embedding_dim)
        self.relation = nn.Embedding(num_relations, embedding_dim)
        # Original ConvKB applies 1x3 filters across the [h, r, t] columns.
        self.conv = nn.Conv2d(1, num_filters, kernel_size=(1, 3), bias=True)
        self.batch_norm = nn.BatchNorm2d(num_filters)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters * embedding_dim, 1)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.entity.weight)
        nn.init.xavier_uniform_(self.relation.weight)
        nn.init.xavier_uniform_(self.conv.weight)
        nn.init.zeros_(self.conv.bias)
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def score_triples(self, heads: torch.Tensor, relations: torch.Tensor, tails: torch.Tensor) -> torch.Tensor:
        stacked = torch.stack((self.entity(heads), self.relation(relations), self.entity(tails)), dim=-1)
        x = self.conv(stacked.unsqueeze(1))
        x = self.batch_norm(x)
        x = F.relu(x)
        x = self.dropout(x)
        return self.fc(x.reshape(x.shape[0], -1)).squeeze(-1)

    def score_all_tails(self, heads: torch.Tensor, relations: torch.Tensor) -> torch.Tensor:
        """Return higher-is-better scores with shape ``[batch, entities]``."""
        batch_size = heads.numel()
        num_entities = self.entity.num_embeddings
        all_tails = torch.arange(num_entities, dtype=torch.long, device=heads.device)
        expanded_heads = heads.unsqueeze(1).expand(batch_size, num_entities).reshape(-1)
        expanded_relations = relations.unsqueeze(1).expand(batch_size, num_entities).reshape(-1)
        expanded_tails = all_tails.unsqueeze(0).expand(batch_size, num_entities).reshape(-1)
        return self.score_triples(expanded_heads, expanded_relations, expanded_tails).reshape(batch_size, num_entities)

    def model_config(self) -> dict[str, Any]:
        return {
            "model_type": "ConvKB", "embedding_dim": self.embedding_dim,
            "num_filters": self.num_filters, "dropout": self.dropout_probability,
        }

    @classmethod
    def load(cls, checkpoint_path: Path) -> tuple["ConvKB", dict[str, int], dict[str, int], dict[str, Any]]:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if checkpoint.get("model_type") != "ConvKB":
            raise ValueError(f"Not a ConvKB checkpoint: {checkpoint_path}")
        entity_to_id = checkpoint["entity_to_id"]
        relation_to_id = checkpoint["relation_to_id"]
        config = checkpoint["model_config"]
        model = cls(
            len(entity_to_id), len(relation_to_id), int(config["embedding_dim"]),
            int(config["num_filters"]), float(config["dropout"]),
        )
        model.load_state_dict(checkpoint["state_dict"])
        return model.cpu().eval(), entity_to_id, relation_to_id, config
