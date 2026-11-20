"""
Surrogate Model (TransE)

INPUT
train.txt , entity/relation.txt

OUTPUT
surrogate_transe.pt

"""
from __future__ import annotations

from pathlib import Path

import torch
from torch import nn


# Defines the model’s embeddings and mathematical calculations.
class TransE(nn.Module):

    # Creates entity and relation embedding tables and initializes their values.
    def __init__(self, num_entities: int, num_relations: int, dim: int = 64):
        super().__init__()
        self.entity = nn.Embedding(num_entities, dim)
        self.relation = nn.Embedding(num_relations, dim)
        nn.init.xavier_uniform_(self.entity.weight)
        nn.init.xavier_uniform_(self.relation.weight)

    # Calculates how far head + relation is from tail.
    def distance(self, heads: torch.Tensor, relations: torch.Tensor,
                 tails: torch.Tensor) -> torch.Tensor:
        return torch.linalg.vector_norm(
            self.entity(heads) + self.relation(relations) - self.entity(tails),
            ord=2, # ord=2 (L2 (Euclidean) distance)
            dim=-1, 
        )

    # Converts that distance into plausibility by making it negative.
    def score(self, heads: torch.Tensor, relations: torch.Tensor,
              tails: torch.Tensor) -> torch.Tensor:
        return -self.distance(heads, relations, tails)


# PURPOSE
# 1. Load Surrogate Model
# 2. Convert Entity & Relation into Numerical IDs
# 3. Calculate Plausibility Score
class Surrogate:

    # Stores the model and name-to-ID mappings
    def __init__(self, model: TransE, entity_to_id: dict[str, int],
                 relation_to_id: dict[str, int]):
        self.model = model.cpu().eval()
        self.entity_to_id = entity_to_id
        self.relation_to_id = relation_to_id

    # Loads learned weights, mappings and embedding dimension
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
    # Measures compatibility between a target relation and a two-relation path.
    @torch.no_grad()

    # Relation score = how well two path relations together represent the target relation.
    # how well interacts_with → affects represents complicates
    def relation_score(self, target: str, r1: str, r2: str) -> float:

        # Return -||r - r1 - r2|| for a two-relation path template
        embeddings = self.model.relation.weight
        target_vector = embeddings[self.relation_to_id[target]]
        first = embeddings[self.relation_to_id[r1]]
        second = embeddings[self.relation_to_id[r2]]
        return float(-torch.linalg.vector_norm(target_vector - first - second, ord=2))

    @torch.no_grad()
    # Triplet score = how plausible one specific triplet is, using its entity and relation embeddings.
    # Example =  (receptor, affects, reptile)
    # FORMULA =  -||h + r - t||
    def triple_score(self, head: str, relation: str, tail: str) -> float:
        heads = torch.tensor([self.entity_to_id[head]], dtype=torch.long)
        relations = torch.tensor([self.relation_to_id[relation]], dtype=torch.long)
        tails = torch.tensor([self.entity_to_id[tail]], dtype=torch.long)
        return float(self.model.score(heads, relations, tails).item())
