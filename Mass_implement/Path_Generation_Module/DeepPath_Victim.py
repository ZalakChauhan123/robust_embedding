"""Local DeepPath-style two-hop path-policy victim for UMLS.

This is a PyTorch reconstruction for this repository, not the original
TensorFlow DeepPath implementation.  It follows the central DeepPath idea:
an entity/relation-conditioned policy selects graph actions to reach a proposed
tail.  It is trained by supervised two-hop path demonstrations because the
original RL environment and task files are absent from this project.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import torch
from torch import nn
from torch.nn import functional as F


class DeepPathLocal(nn.Module):
    """Target-conditioned policy over inverse-KG actions with two-hop scoring."""

    def __init__(
        self,
        num_entities: int,
        num_relations: int,
        embedding_dim: int = 64,
        hidden_dim: int = 128,
        max_actions: int = 1,
        dropout: float = 0.1,
        beam_size: int = 10,
    ) -> None:
        super().__init__()
        if min(num_entities, num_relations, embedding_dim, hidden_dim, max_actions, beam_size) < 1:
            raise ValueError("DeepPathLocal parameters must be positive")
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.max_actions = max_actions
        self.dropout_probability = dropout
        self.beam_size = beam_size
        self.entity = nn.Embedding(num_entities, embedding_dim)
        self.query_relation = nn.Embedding(num_relations, embedding_dim)
        self.action_relation = nn.Embedding(2 * num_relations, embedding_dim)
        self.policy = nn.Sequential(
            nn.Linear(3 * embedding_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, embedding_dim),
        )
        self.register_buffer("action_relation_ids", torch.zeros((num_entities, max_actions), dtype=torch.long))
        self.register_buffer("action_destination_ids", torch.zeros((num_entities, max_actions), dtype=torch.long))
        self.register_buffer("action_mask", torch.zeros((num_entities, max_actions), dtype=torch.bool))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.entity.weight)
        nn.init.xavier_uniform_(self.query_relation.weight)
        nn.init.xavier_uniform_(self.action_relation.weight)
        for module in self.policy:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    @staticmethod
    def inverse_edges(
        edges: Iterable[tuple[int, int, int]], num_relations: int
    ) -> list[tuple[int, int, int]]:
        expanded: set[tuple[int, int, int]] = set()
        for head, relation, tail in edges:
            expanded.add((head, relation, tail))
            expanded.add((tail, relation + num_relations, head))
        return sorted(expanded)

    @classmethod
    def max_actions_for_edges(
        cls, edges: Iterable[tuple[int, int, int]], num_entities: int, num_relations: int
    ) -> int:
        counts = [0] * num_entities
        for head, _, _ in cls.inverse_edges(edges, num_relations):
            counts[head] += 1
        return max(1, max(counts, default=0))

    @torch.no_grad()
    def set_graph(self, edges: Iterable[tuple[int, int, int]]) -> None:
        """Store a sorted inverse-KG action table in checkpointed buffers."""
        outgoing: list[list[tuple[int, int]]] = [[] for _ in range(self.num_entities)]
        for head, relation, tail in self.inverse_edges(edges, self.num_relations):
            outgoing[head].append((relation, tail))
        if max((len(actions) for actions in outgoing), default=0) > self.max_actions:
            raise ValueError("Action table exceeds checkpoint model_config.max_actions")
        relations = torch.zeros_like(self.action_relation_ids)
        destinations = torch.zeros_like(self.action_destination_ids)
        mask = torch.zeros_like(self.action_mask)
        for entity, actions in enumerate(outgoing):
            for index, (relation, tail) in enumerate(actions):
                relations[entity, index] = relation
                destinations[entity, index] = tail
                mask[entity, index] = True
        self.action_relation_ids.copy_(relations)
        self.action_destination_ids.copy_(destinations)
        self.action_mask.copy_(mask)

    def policy_logits(
        self, current_entities: torch.Tensor, target_entities: torch.Tensor, query_relations: torch.Tensor
    ) -> torch.Tensor:
        state = torch.cat((
            self.entity(current_entities), self.entity(target_entities), self.query_relation(query_relations),
        ), dim=-1)
        query = self.policy(state)
        action_relations = self.action_relation_ids[current_entities]
        action_destinations = self.action_destination_ids[current_entities]
        action_vectors = self.action_relation(action_relations) + self.entity(action_destinations)
        logits = (action_vectors * query.unsqueeze(1)).sum(dim=-1)
        return logits.masked_fill(~self.action_mask[current_entities], -1.0e9)

    def score_all_tails(self, heads: torch.Tensor, relations: torch.Tensor) -> torch.Tensor:
        """Return DeepPath-style log success scores for every candidate tail.

        Queries are batched to make complete filtered Set-A evaluation practical
        on CPU.  The calculation remains the exact direct/two-hop beam score.
        """
        device = self.entity.weight.device
        batch_size = heads.numel()
        targets = torch.arange(self.num_entities, dtype=torch.long, device=device).unsqueeze(0).expand(batch_size, -1)
        queries = relations.unsqueeze(1).expand(batch_size, self.num_entities)
        current = heads.unsqueeze(1).expand(batch_size, self.num_entities)
        first_logits = self.policy_logits(current.reshape(-1), targets.reshape(-1), queries.reshape(-1))
        first_logits = first_logits.reshape(batch_size, self.num_entities, self.max_actions)
        first_probabilities = F.softmax(first_logits, dim=2)
        first_destinations = self.action_destination_ids[heads].unsqueeze(1).expand(
            batch_size, self.num_entities, -1
        )
        # The direct term is normally zero for held-out targets, but retains a
        # valid one-hop policy score when evaluating a graph that contains one.
        probability = (first_probabilities * (first_destinations == targets.unsqueeze(2))).sum(dim=2)
        beam = min(self.beam_size, self.max_actions)
        first_probability, first_index = torch.topk(first_probabilities, k=beam, dim=2)
        for beam_index in range(beam):
            next_entities = first_destinations.gather(
                2, first_index[:, :, beam_index:beam_index + 1]
            ).squeeze(2)
            second_logits = self.policy_logits(
                next_entities.reshape(-1), targets.reshape(-1), queries.reshape(-1)
            ).reshape(batch_size, self.num_entities, self.max_actions)
            second_probabilities = F.softmax(second_logits, dim=2)
            second_destinations = self.action_destination_ids[next_entities.reshape(-1)].reshape(
                batch_size, self.num_entities, self.max_actions
            )
            reaches_target = (second_probabilities * (second_destinations == targets.unsqueeze(2))).sum(dim=2)
            probability = probability + first_probability[:, :, beam_index] * reaches_target
        return torch.log(probability.clamp_min(1.0e-12))

    def model_config(self) -> dict[str, Any]:
        return {
            "model_type": "DeepPathLocal", "embedding_dim": self.embedding_dim,
            "hidden_dim": self.hidden_dim, "max_actions": self.max_actions,
            "dropout": self.dropout_probability, "beam_size": self.beam_size,
            "max_hops": 2, "training": "supervised_two_hop_path_demonstrations",
        }

    @classmethod
    def load(cls, checkpoint_path: Path) -> tuple["DeepPathLocal", dict[str, int], dict[str, int], dict[str, Any]]:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if checkpoint.get("model_type") != "DeepPathLocal":
            raise ValueError(f"Not a DeepPathLocal checkpoint: {checkpoint_path}")
        entity_to_id = checkpoint["entity_to_id"]
        relation_to_id = checkpoint["relation_to_id"]
        config = checkpoint["model_config"]
        model = cls(
            len(entity_to_id), len(relation_to_id), int(config["embedding_dim"]),
            int(config["hidden_dim"]), int(config["max_actions"]), float(config["dropout"]),
            int(config["beam_size"]),
        )
        model.load_state_dict(checkpoint["state_dict"])
        return model.cpu().eval(), entity_to_id, relation_to_id, config
