"""Supervised two-hop relation-path translation for the MaSS pipeline.

Implements the positive/negative relation-label training idea in MaSS Section 4.2.

NOTE :- The LogisticRegression architecture and feature design are implementation
choices; the paper does not prescribe this precise model.

Dependencies: scikit-learn, joblib

Usage (from Mass_implement/):
    python PRA_Module/PRA_Supervised.py --train dataset_uml/train.txt

The trained model is reused for every target relation in the same KG.
"""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import joblib
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline

MASS_DIR = Path(__file__).resolve().parents[1]
DEFAULT_TRAIN = MASS_DIR / "dataset_uml" / "train.txt"
DEFAULT_MODEL = MASS_DIR / "outputs" / "PRA" / "supervised_pra_umls.joblib"
DEFAULT_REPORT = MASS_DIR / "outputs" / "PRA" / "training_report.json"


def load_triples(path: Path) -> list[tuple[str, str, str]]:
    """Load name-based triples, keeping all three fields as strings."""
    triples = []
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t") if "\t" in line else line.split()
            if len(parts) != 3:
                raise ValueError(f"Expected 3 columns at {path}:{number}")
            triples.append(tuple(parts))
    if not triples:
        raise ValueError(f"No training triples found: {path}")
    return triples


def mine_two_hop_paths(triples):
    """Return path -> distinct endpoint pairs and endpoints -> direct relations.

    Deduplication matches the pair-based semantics of empirical PRA.
    """
    graph = defaultdict(lambda: defaultdict(set))
    direct = defaultdict(set)
    relation_set = set()
    for head, rel, tail in triples:
        graph[head][rel].add(tail)
        direct[(head, tail)].add(rel)
        relation_set.add(rel)

    path_pairs = defaultdict(set)
    for source, by_relation in graph.items():
        for r1, middles in by_relation.items():
            for middle in middles:
                for r2, tails in graph.get(middle, {}).items():
                    for target in tails:
                        path_pairs[(r1, r2)].add((source, target))
    return path_pairs, direct, sorted(relation_set)


def features(r1: str, r2: str, candidate_relation: str) -> dict[str, int]:
    """Categorical interactions let the classifier learn path/target association.

    The joint feature is essential: only separate path and relation one-hot
    features would mostly learn which relations are frequent, not which paths
    predict which relations.
    """
    return {
        f"first={r1}": 1,
        f"second={r2}": 1,
        f"path={r1}|{r2}": 1,
        f"target={candidate_relation}": 1,
        f"first_target={r1}|{candidate_relation}": 1,
        f"second_target={r2}|{candidate_relation}": 1,
        f"path_target={r1}|{r2}|{candidate_relation}": 1,
    }


def build_examples(path_pairs, direct, relations, *, negatives_per_positive=1, seed=42):
    """Create (path, candidate relation, binary label) examples.

    For every observed direct positive r+ on a path's endpoint pair,
    randomly draw r- from relations *not* present on that endpoint pair.
    This avoids treating a KNOWN positive as a negative, though missing
    KG facts may still be sampled as imperfect negatives.
    """
    if negatives_per_positive < 1:
        raise ValueError("negatives_per_positive must be >= 1")
    rng = random.Random(seed)
    X, y, group_ids = [], [], []
    for (r1, r2), pairs in sorted(path_pairs.items()):
        for source, target in sorted(pairs):
            positives = direct.get((source, target), set())
            if not positives:
                continue  # Paper samples negatives for observed positive examples
            possible_negatives = [r for r in relations if r not in positives]
            for positive_rel in sorted(positives):
                X.append(features(r1, r2, positive_rel))
                y.append(1)
                group_ids.append((source, target))
                for _ in range(negatives_per_positive):
                    if not possible_negatives:
                        break
                    negative_rel = rng.choice(possible_negatives)
                    X.append(features(r1, r2, negative_rel))
                    y.append(0)
                    group_ids.append((source, target))
    if not X or len(set(y)) < 2:
        raise ValueError("Insufficient positive/negative examples for training")
    return X, np.asarray(y, dtype=int), group_ids


def train_supervised_pra(train_path=DEFAULT_TRAIN, output_path=DEFAULT_MODEL,
                         report_path=DEFAULT_REPORT, seed=42,
                         negatives_per_positive=1, test_size=0.2):
    triples = load_triples(Path(train_path))
    path_pairs, direct, relations = mine_two_hop_paths(triples)
    X, y, group_ids = build_examples(
        path_pairs, direct, relations,
        negatives_per_positive=negatives_per_positive, seed=seed
    )
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1")

    # Prevent the same endpoint pair from appearing in both train and holdout.
    # Caution: the topology used for path mining still includes all KG edges.
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    groups = [f"{h}\0{t}" for h, t in group_ids]
    train_idx, val_idx = next(splitter.split(X, y, groups=groups))

    def make_pipeline():
        return Pipeline([
            ("vectorizer", DictVectorizer(sparse=True)),
            ("classifier", LogisticRegression(
                solver="lbfgs", max_iter=1000, random_state=seed
            )),
        ])

    # Holdout diagnostics, not a claimed end-to-end attack evaluation.
    evaluation_model = make_pipeline()
    evaluation_model.fit([X[i] for i in train_idx], y[train_idx])
    probs = evaluation_model.predict_proba([X[i] for i in val_idx])[:, 1]
    report = {
        "method": "binary_logistic_regression_supervised_pra",
        "train_file": str(train_path),
        "triples": len(triples),
        "relations": len(relations),
        "unique_two_hop_templates": len(path_pairs),
        "examples": len(y),
        "positive_examples": int((y == 1).sum()),
        "negative_examples": int((y == 0).sum()),
        "heldout_endpoint_pairs": len({groups[i] for i in val_idx}),
        "holdout_average_precision": float(average_precision_score(y[val_idx], probs)),
        "holdout_roc_auc": float(roc_auc_score(y[val_idx], probs)),
        "seed": seed,
        "negatives_per_positive": negatives_per_positive,
        "note": "Held-out by entity pair, not a leakage-free disjoint-graph evaluation; "
                "missing KG facts may be sampled as negatives.",
    }

    # Once diagnostics are complete, refit using all available training examples.
    final_model = make_pipeline()
    final_model.fit(X, y)
    payload = {
        "model": final_model,
        "path_templates": sorted(path_pairs),
        "relations": relations,
        "training_report": report,
    }
    output_path, report_path = Path(output_path), Path(report_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(payload, output_path)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Training completed: {len(path_pairs)} templates, {len(y)} examples")
    print(f"Validation AP={report['holdout_average_precision']:.4f} "
          f"ROC-AUC={report['holdout_roc_auc']:.4f}")
    print(f"Model saved to: {output_path}")
    return report


class SupervisedPRA:
    """Drop-in relation translator for Semantic_Constraints.py.

    Returns the same three-tuple structure as empirical PRA:
       [((r1, r2), predicted_confidence, matching_pairs), ...]
    matching_pairs=[] because supervised translation does not compute an
    empirical trace list; downstream Semantic Constraints ignores it.
    """
    def __init__(self, model_path=DEFAULT_MODEL, threshold=0.9):
        if not 0 <= threshold <= 1:
            raise ValueError("threshold must lie in [0, 1]")
        # Only load checkpoints created locally/from trusted sources.
        payload = joblib.load(model_path)
        self.model = payload["model"]
        self.templates = payload["path_templates"]
        self.relations = set(payload["relations"])
        self.threshold = threshold

    def translate_relation(self, target_relation: str, *, exclude_target_relation: bool = False):
        if target_relation not in self.relations:
            raise ValueError(f"Target relation absent from supervised PRA: {target_relation}")
        data = [features(r1, r2, target_relation) for r1, r2 in self.templates]
        probabilities = self.model.predict_proba(data)[:, 1]
        candidates = [
            ((r1, r2), float(score), [])
            for (r1, r2), score in zip(self.templates, probabilities)
            if score > self.threshold
            and (not exclude_target_relation or
                 (r1 != target_relation and r2 != target_relation))
        ]
        candidates.sort(key=lambda x: (-x[1], x[0]))
        return candidates


def main():
    parser = argparse.ArgumentParser(description="Train supervised PRA for MaSS")
    parser.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--output", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--negatives-per-positive", type=int, default=1)
    parser.add_argument("--test-size", type=float, default=0.2)
    args = parser.parse_args()
    train_supervised_pra(
        args.train, args.output, args.report,
        seed=args.seed, negatives_per_positive=args.negatives_per_positive,
        test_size=args.test_size
    )


if __name__ == "__main__":
    main()
