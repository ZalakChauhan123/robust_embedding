"""A compact, CPU-friendly ConvE implementation for UMLS victim evaluation.

This is intentionally a victim model rather than an attacker surrogate.  The
attacker remains the frozen TransE surrogate used by MaSS, DirectAdd, and
ComAttack.  Scores are logits: higher values mean a more plausible tail.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


class ConvE(nn.Module):
    """ConvE with 1-vs-all tail scoring and an 8x8-compatible embedding API."""

    def __init__(
        self,
        num_entities: int,
        num_relations: int,
        embedding_dim: int = 64,
        embedding_height: int = 8,
        num_filters: int = 32,
        kernel_size: int = 3,
        input_dropout: float = 0.2,
        feature_dropout: float = 0.2,
        hidden_dropout: float = 0.3,
    ) -> None:
        super().__init__()
        if num_entities < 1 or num_relations < 1:
            raise ValueError("ConvE needs at least one entity and relation")
        if embedding_dim < 1 or embedding_height < 1 or embedding_dim % embedding_height:
            raise ValueError("embedding_dim must be divisible by embedding_height")
        if kernel_size < 1:
            raise ValueError("kernel_size must be positive")
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.embedding_dim = embedding_dim
        self.embedding_height = embedding_height
        self.embedding_width = embedding_dim // embedding_height
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.input_dropout_probability = input_dropout
        self.feature_dropout_probability = feature_dropout
        self.hidden_dropout_probability = hidden_dropout

        conv_height = 2 * embedding_height - kernel_size + 1
        conv_width = self.embedding_width - kernel_size + 1
        if conv_height < 1 or conv_width < 1:
            raise ValueError("kernel_size is too large for the selected embedding reshape")

        self.entity = nn.Embedding(num_entities, embedding_dim)
        self.relation = nn.Embedding(num_relations, embedding_dim)
        self.entity_bias = nn.Parameter(torch.zeros(num_entities))
        self.bn0 = nn.BatchNorm2d(1)
        self.bn1 = nn.BatchNorm2d(num_filters)
        self.bn2 = nn.BatchNorm1d(embedding_dim)
        self.input_dropout = nn.Dropout(input_dropout)
        self.feature_dropout = nn.Dropout2d(feature_dropout)
        self.hidden_dropout = nn.Dropout(hidden_dropout)
        self.conv1 = nn.Conv2d(1, num_filters, (kernel_size, kernel_size), bias=True)
        self.fc = nn.Linear(num_filters * conv_height * conv_width, embedding_dim)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_normal_(self.entity.weight)
        nn.init.xavier_normal_(self.relation.weight)
        nn.init.xavier_normal_(self.conv1.weight)
        nn.init.zeros_(self.conv1.bias)
        nn.init.xavier_normal_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)
        nn.init.zeros_(self.entity_bias)

    def model_config(self) -> dict[str, Any]:
        return {
            "model_type": "ConvE",
            "embedding_dim": self.embedding_dim,
            "embedding_height": self.embedding_height,
            "num_filters": self.num_filters,
            "kernel_size": self.kernel_size,
            "input_dropout": self.input_dropout_probability,
            "feature_dropout": self.feature_dropout_probability,
            "hidden_dropout": self.hidden_dropout_probability,
        }

    def query_representation(self, heads: torch.Tensor, relations: torch.Tensor) -> torch.Tensor:
        """Return ConvE's query vector for one or more (head, relation) pairs."""
        head = self.entity(heads).reshape(-1, 1, self.embedding_height, self.embedding_width)
        relation = self.relation(relations).reshape(-1, 1, self.embedding_height, self.embedding_width)
        stacked = torch.cat((head, relation), dim=2)
        x = self.bn0(stacked)
        x = self.input_dropout(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.feature_dropout(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        x = self.hidden_dropout(x)
        x = self.bn2(x)
        return F.relu(x)

    def score_all_tails(self, heads: torch.Tensor, relations: torch.Tensor) -> torch.Tensor:
        """Return logits with shape ``[batch, num_entities]``."""
        query = self.query_representation(heads, relations)
        return query @ self.entity.weight.transpose(0, 1) + self.entity_bias

    def forward(self, heads: torch.Tensor, relations: torch.Tensor) -> torch.Tensor:
        return self.score_all_tails(heads, relations)

    @classmethod
    def load(cls, checkpoint_path: Path) -> tuple["ConvE", dict[str, int], dict[str, int], dict[str, Any]]:
        """Load a checkpoint created by ``train_conve_victim_models.py``."""
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if checkpoint.get("model_type") != "ConvE":
            raise ValueError(f"Not a ConvE checkpoint: {checkpoint_path}")
        entity_to_id = checkpoint["entity_to_id"]
        relation_to_id = checkpoint["relation_to_id"]
        config = checkpoint["model_config"]
        model = cls(
            len(entity_to_id), len(relation_to_id),
            embedding_dim=int(config["embedding_dim"]),
            embedding_height=int(config["embedding_height"]),
            num_filters=int(config["num_filters"]),
            kernel_size=int(config["kernel_size"]),
            input_dropout=float(config["input_dropout"]),
            feature_dropout=float(config["feature_dropout"]),
            hidden_dropout=float(config["hidden_dropout"]),
        )
        model.load_state_dict(checkpoint["state_dict"])
        return model.cpu().eval(), entity_to_id, relation_to_id, config
