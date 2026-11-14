"""Explicit UMLS Table 3 eligibility rules for the existing attack rankers.

Type and plausibility constraints are separate experiments. Plausibility uses
the frozen clean TransE surrogate, calibrated only on clean training triples.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import torch


def type_allowed(triple, domain, range_):
    h, r, t = triple
    return h in domain.get(r, ()) and t in range_.get(r, ())


@dataclass
class SurrogateScores:
    values: np.ndarray
    entity_to_id: dict
    relation_to_id: dict

    @classmethod
    @torch.no_grad()
    def from_surrogate(cls, surrogate):
        """Cache the small UMLS score cube, one relation at a time.

        This stores 46*135*135 float scores for the default dataset. It uses
        exactly -L2(h+r-t); no victim predictions enter the eligibility test.
        """
        entity = surrogate.model.entity.weight.detach().cpu()
        relation = surrogate.model.relation.weight.detach().cpu()
        values = np.empty((len(relation), len(entity), len(entity)), dtype=np.float32)
        for index in range(len(relation)):
            residual = entity[:, None, :] + relation[index] - entity[None, :, :]
            values[index] = (-torch.linalg.vector_norm(residual, ord=2, dim=-1)).numpy()
        if not np.all(np.isfinite(values)):
            raise ValueError("Surrogate produced nonfinite plausibility scores")
        return cls(values, dict(surrogate.entity_to_id), dict(surrogate.relation_to_id))

    def score(self, triple):
        h, r, t = triple
        try:
            return float(self.values[self.relation_to_id[r], self.entity_to_id[h], self.entity_to_id[t]])
        except KeyError as error:
            raise ValueError(f"Unknown entity or relation in plausibility query: {triple}") from error


def calibrate_plausibility(scores, clean_train, *, quantile=0.5):
    if not math.isfinite(quantile) or not 0 <= quantile <= 1 or not clean_train:
        raise ValueError("Calibration needs clean training facts and a finite quantile in [0,1]")
    values = np.asarray([scores.score(triple) for triple in clean_train], dtype=float)
    if not np.all(np.isfinite(values)):
        raise ValueError("Calibration scores must be finite")
    return {
        "quantile": quantile, "threshold": float(np.quantile(values, quantile, method="linear")),
        "num_clean_train_triples": len(values), "min": float(np.min(values)),
        "median": float(np.median(values)), "max": float(np.max(values)),
        "mean": float(np.mean(values)), "score": "negative TransE L2 distance; larger is more plausible",
        "scope": "global clean-training score quantile; no validation/test/target/victim calibration",
        "interpretation": "Explicit UMLS design choice; Table 3 does not specify a numeric plausibility cutoff.",
    }


@dataclass
class EnhancementFilter:
    mode: str
    domain: dict
    range_: dict
    scores: SurrogateScores
    threshold: float

    def __post_init__(self):
        if self.mode not in ("none", "type", "plausibility") or not math.isfinite(self.threshold):
            raise ValueError("Unknown enhancement or nonfinite threshold")

    def __call__(self, triple):
        if self.mode == "none":
            return True
        if self.mode == "type":
            return type_allowed(triple, self.domain, self.range_)
        return self.scores.score(triple) >= self.threshold
