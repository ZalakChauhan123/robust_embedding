"""
-- MaSS Section 4.4.1 : rank semantically valid two-hop candidate paths.
-- Uses 
        existing PRA, 
        Semantic Constraints & 
        trained TransE Surrogate module.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

MASS_DIR = Path(__file__).resolve().parents[1]
if str(MASS_DIR) not in sys.path:
    sys.path.insert(0, str(MASS_DIR))

from PRA_Module.PRA_supervised import SupervisedPRA
from PRA_Module.Pra_implement_finalVersion import MaSSPathTranslator, load_dataset
from Semantic_Constraints_Module.Semantic_Constraints import (
    apply_semantic_constraints_to_pra,
    load_dataset_entities,
    load_semantic_constraints,
)
from TransE_Surrogate import Surrogate

def rank_paths(head: str, relation: str, tail: str, *, threshold: float = 0.8,
               top_k: int = 10, top_entities: int = 10,
               checkpoint: Path | None = None,
               pra_method: str = "empirical") -> dict:
    """Return ranked paths. All scores are higher-is-better (negative distances)."""
    if top_k < 1 or top_entities < 1:
        raise ValueError("top_k and top_entities must be positive integers")

    dataset_dir = MASS_DIR / "dataset_uml"
    checkpoint = checkpoint or (MASS_DIR / "outputs" / "Path_Generation" / "transe_umls.pt")
    surrogate = Surrogate.load(checkpoint)
    for name, mapping, label in (
        (head, surrogate.entity_to_id, "head"),
        (tail, surrogate.entity_to_id, "tail"),
        (relation, surrogate.relation_to_id, "relation"),
    ):
        if name not in mapping:
            raise ValueError(f"Unknown target {label}: {name!r}")

    triples = load_dataset(str(dataset_dir / "train.txt"))
    if pra_method == "supervised":
        translator = SupervisedPRA(
            model_path=MASS_DIR / "outputs" / "PRA" / "supervised_pra_umls.joblib",
            threshold=threshold
        )

    elif pra_method == "empirical":
        translator = MaSSPathTranslator(
            triples,
            threshold=threshold
        )

    else:
        raise ValueError(f"Unknown PRA method: {pra_method}")
    pra_paths = translator.translate_relation(relation)
    constraints = load_semantic_constraints(str(dataset_dir / "Semantic_Constraints" / "SRSTRE2"))
    entities = load_dataset_entities(str(dataset_dir / "entities.txt"))
    semantic_results = apply_semantic_constraints_to_pra(pra_paths, constraints, entities)

    known_train = set(triples)

    # Rank templates by TransE relation composition, not by PRA confidence.
    templates = []
    for item in semantic_results:
        r1, r2 = item["r1"], item["r2"]
        if r1 not in surrogate.relation_to_id or r2 not in surrogate.relation_to_id:
            continue
        candidates = [e for e in item["valid_intermediate_entities"]
                      if e in surrogate.entity_to_id]
        if not candidates:
            continue
        templates.append({
            "r1": r1, "r2": r2,
            "pra_confidence": item["confidence"],
            "relation_score": surrogate.relation_score(relation, r1, r2),
            "candidate_entities": candidates,
        })
    templates.sort(key=lambda x: (-x["relation_score"], x["r1"], x["r2"]))

    selected = templates[:top_k]
    ranked_paths = []
    for template in selected:
        r1, r2 = template["r1"], template["r2"]
        entity_scores = []
        for middle in template["candidate_entities"]:
            score_first = surrogate.triple_score(head, r1, middle)
            score_second = surrogate.triple_score(middle, r2, tail)
            entity_scores.append((score_first + score_second, middle, score_first, score_second))
        entity_scores.sort(key=lambda x: (-x[0], x[1]))
        for entity_score, middle, score_first, score_second in entity_scores[:top_entities]:
            ranked_paths.append({
                "r1": r1,
                "intermediate_entity": middle,
                "r2": r2,
                "first_triple": [head, r1, middle],
                "second_triple": [middle, r2, tail],
                "pra_confidence": template["pra_confidence"],
                "relation_score": template["relation_score"],
                "first_triple_score": score_first,
                "second_triple_score": score_second,
                "entity_score": entity_score,
                "lower_bound_score": template["relation_score"] + entity_score,
                # DIAGNOSTIC 1: DETECT SELF-LOOPS WITHOUT FILTERING CANDIDATE PATHS.
                "first_triple_is_self_loop": head == middle,
                "second_triple_is_self_loop": middle == tail,
                "has_self_loop": head == middle or middle == tail,
                # DIAGNOSTIC 2: CHECK WHETHER EACH TRIPLE ALREADY EXISTS IN TRAIN.TXT.
                "first_triple_exists_in_train": (head, r1, middle) in known_train,
                "second_triple_exists_in_train": (middle, r2, tail) in known_train,
                # COUNT TRIPLES ABSENT FROM THE TRAINING GRAPH (ZERO, ONE, OR TWO).
                "num_new_triples": (
                    int((head, r1, middle) not in known_train)
                    + int((middle, r2, tail) not in known_train)
                ),
            })
    ranked_paths.sort(key=lambda x: (-x["lower_bound_score"], x["r1"], x["intermediate_entity"], x["r2"]))
    for index, item in enumerate(ranked_paths, 1):
        item["rank"] = index
    return {
        "dataset": "UMLS",
        "target_triple": [head, relation, tail],
        "pra_threshold": threshold,
        "top_k_relation_templates": top_k,
        "top_entities_per_template": top_entities,
        "num_pra_templates": len(pra_paths),
        "num_semantically_valid_templates": len(templates),
        "num_selected_templates": len(selected),
        "num_ranked_paths": len(ranked_paths),
        "ranked_candidate_paths": ranked_paths,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Rank MaSS two-hop candidate paths")
    parser.add_argument("--head", default="age_group")
    parser.add_argument("--relation", default="uses")
    parser.add_argument("--tail", default="clinical_drug")
    parser.add_argument("--threshold", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--top-entities", type=int, default=10)
    parser.add_argument("--checkpoint", type=Path, default=MASS_DIR / "outputs" / "Path_Generation" / "transe_umls.pt")
    parser.add_argument("--output", type=Path, default=MASS_DIR / "outputs" / "Path_Generation" / "entity_selection_output.json")
    parser.add_argument("--pra-method", choices=["empirical", "supervised"], default="empirical")
    args = parser.parse_args()
    result = rank_paths(args.head, args.relation, args.tail,
                        threshold=args.threshold, top_k=args.top_k,
                        top_entities=args.top_entities, checkpoint=args.checkpoint, pra_method=args.pra_method)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Ranked {result['num_ranked_paths']} candidate paths")
    for path in result["ranked_candidate_paths"][:5]:
        print(f"  #{path['rank']}: {path['r1']} -> {path['intermediate_entity']} -> {path['r2']} | score={path['lower_bound_score']:.4f}")
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()