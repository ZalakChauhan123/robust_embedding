#!/usr/bin/env python3
"""Filtered RQ1/RQ2 evaluation for paired clean and poisoned ConvE victims."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch

MASS_DIR = Path(__file__).resolve().parents[2]
EVALUATION_DIR = Path(__file__).resolve().parent
if str(MASS_DIR) not in sys.path:
    sys.path.insert(0, str(MASS_DIR))
if str(EVALUATION_DIR) not in sys.path:
    sys.path.insert(0, str(EVALUATION_DIR))

from Path_Generation_Module.ConvE_Victim import ConvE  # noqa: E402
from Evaluate_victim_models import (  # noqa: E402
    DATASET_DIR,
    PILOT_DIR,
    Triple,
    compare_group,
    encode_all,
    load_json,
    load_poisoning_status,
    make_true_tail_index,
    require_triples,
    verify_training_sources,
    write_csv,
    write_json,
)

VICTIM_DIR = PILOT_DIR / "conve_victim_models"
POISONING_DIR = PILOT_DIR / "poisoning"
DEFAULT_OUTPUT_DIR = PILOT_DIR / "conve_evaluation"


@torch.no_grad()
def tail_rank(
    victim: ConvE,
    encoded_target: tuple[int, int, int],
    true_tails_by_hr: dict[tuple[int, int], set[int]],
) -> tuple[float, float]:
    """Filtered tail rank from ConvE logits; higher score is more plausible."""
    head, relation, tail = encoded_target
    scores = victim.score_all_tails(
        torch.tensor([head], dtype=torch.long), torch.tensor([relation], dtype=torch.long)
    )[0]
    target_score = float(scores[tail].item())
    filtered_scores = scores.clone()
    other_true_tails = true_tails_by_hr.get((head, relation), set()) - {tail}
    if other_true_tails:
        filtered_scores[sorted(other_true_tails)] = -torch.inf
    num_better = int((filtered_scores > target_score).sum().item())
    num_tied = int((filtered_scores == target_score).sum().item())
    rank = float(1 + num_better + (num_tied - 1) / 2)
    if not 1 <= rank <= filtered_scores.numel():
        raise AssertionError(f"Invalid ConvE rank for {encoded_target}")
    return rank, target_score


def run(args: argparse.Namespace) -> dict[str, Any]:
    input_paths = [
        args.clean_checkpoint, args.poisoned_checkpoint, args.train, args.poisoned_train,
        args.injected, args.valid, args.test, args.targets, args.poisoning_summary,
        args.training_summary,
    ]
    for path in input_paths:
        if not path.is_file():
            raise FileNotFoundError(f"Missing required input: {path}")
    clean_facts = require_triples(args.train)
    poisoned_facts = require_triples(args.poisoned_train)
    injected_facts = require_triples(args.injected)
    valid_facts = require_triples(args.valid)
    set_a = require_triples(args.test)
    set_b = require_triples(args.targets)
    provenance = load_poisoning_status(args.poisoning_summary, set_b)
    original_set = set(clean_facts) | set(valid_facts) | set(set_a)
    if set(clean_facts) & (set(valid_facts) | set(set_a)):
        raise ValueError("Original training facts overlap held-out facts")
    if set(set_b) & (original_set | set(poisoned_facts)):
        raise ValueError("Set B target must not appear in the original or poisoned KG")
    if set(injected_facts) & original_set:
        raise ValueError("Injected triples overlap original KG (leakage)")
    if set(poisoned_facts) != set(clean_facts) | set(injected_facts):
        raise ValueError("Poisoned train must equal clean train union injected triples")

    training_metadata = load_json(args.training_summary)
    if training_metadata.get("victim_model") != "ConvE":
        raise ValueError("Training summary does not identify a ConvE victim")
    verify_training_sources(
        training_metadata, args.clean_checkpoint, args.poisoned_checkpoint,
        args.train, args.poisoned_train, args.injected, args.valid, args.test, args.targets,
    )
    clean_model, clean_e2id, clean_r2id, _ = ConvE.load(args.clean_checkpoint)
    poisoned_model, poison_e2id, poison_r2id, _ = ConvE.load(args.poisoned_checkpoint)
    if clean_e2id != poison_e2id or clean_r2id != poison_r2id:
        raise ValueError("Clean/poisoned ConvE checkpoint ID mappings differ")
    entity_to_id, relation_to_id = clean_e2id, clean_r2id
    known_original = encode_all(list(original_set), entity_to_id, relation_to_id)
    filter_index = make_true_tail_index(known_original)

    def evaluate(
        triples: list[Triple], *, statuses: list[dict[str, str]] | None = None
    ) -> list[dict[str, Any]]:
        encoded = encode_all(triples, entity_to_id, relation_to_id)
        rows: list[dict[str, Any]] = []
        for index, (triple, triple_ids) in enumerate(zip(triples, encoded), 1):
            clean_rank, clean_score = tail_rank(clean_model, triple_ids, filter_index)
            poisoned_rank, poisoned_score = tail_rank(poisoned_model, triple_ids, filter_index)
            row: dict[str, Any] = {
                "target_id": f"target_{index:03d}" if statuses is not None else f"test_{index:04d}",
                "head": triple[0], "relation": triple[1], "tail": triple[2],
                "clean_rank": clean_rank, "poisoned_rank": poisoned_rank,
                "rank_improvement": clean_rank - poisoned_rank,
                "clean_score": clean_score, "poisoned_score": poisoned_score,
                "score_change": poisoned_score - clean_score,
                "clean_hits_at_10": int(clean_rank <= 10),
                "poisoned_hits_at_10": int(poisoned_rank <= 10),
                "rank_outcome": "improved" if poisoned_rank < clean_rank else (
                    "worsened" if poisoned_rank > clean_rank else "unchanged"),
            }
            if statuses is not None:
                row["poisoning_status"] = statuses[index - 1]["status"]
                row["injected_for_target"] = int(statuses[index - 1]["injected_for_target"])
            rows.append(row)
        return rows

    print(f"Loaded ConvE victim checkpoints; {len(entity_to_id)} entities, {len(relation_to_id)} relations.")
    print(f"Evaluating Set B ({len(set_b)} attacker target triples)...", flush=True)
    attack_rows = evaluate(set_b, statuses=provenance)
    print(f"Evaluating Set A ({len(set_a)} original test triples)...", flush=True)
    clean_rows = evaluate(set_a)
    all_b = compare_group(attack_rows)
    injected_b = compare_group([row for row in attack_rows if row["poisoning_status"] == "injected"])
    not_injected_b = compare_group([row for row in attack_rows if row["poisoning_status"] != "injected"])
    set_a_metrics = compare_group(clean_rows)

    attack_result = {
        "research_question": f"RQ1 — attack effectiveness ({args.attack_name}, ConvE victim)",
        "victim_model": "ConvE",
        "prediction": "TAIL only: (h*, r*, ?) ranked against every known entity",
        "ranking_protocol": "filtered; same original train+valid+test facts for both models",
        "filter_excludes": "injected edges and synthetic Set B targets",
        "tie_policy": "average rank, consistent with the TransE evaluation",
        "interpretation": "Higher poisoned MRR/Hits@10, or positive rank improvement, favors the attacker",
        "set_B_all_targets": all_b,
        "set_B_injected_targets": injected_b,
        "set_B_not_injected_targets": not_injected_b,
        "attack_name": args.attack_name,
        "opaque_box_note": "The frozen attacker surrogate is TransE while the victim is ConvE.",
        "target_protocol_note": "Set B was selected with TransE ranks 11–15, not ConvE ranks.",
    }
    set_a_result = {
        "research_question": "RQ2 — preliminary stealthiness check on Set A",
        "victim_model": "ConvE",
        "prediction": "TAIL only to match the RQ1 protocol",
        "set_A_original_test_triples": set_a_metrics,
        "interpretation": "Small clean-to-poisoned degradation in Hits@10/MRR favors stealthiness",
    }
    evaluation_summary = {
        "experiment": f"{args.attack_name} UMLS pilot: ConvE victim evaluation",
        "attack_name": args.attack_name, "victim_model": "ConvE",
        "input_files": {
            "clean_victim": str(args.clean_checkpoint), "poisoned_victim": str(args.poisoned_checkpoint),
            "set_A": str(args.test), "set_B": str(args.targets),
            "training_summary": str(args.training_summary), "poisoning_summary": str(args.poisoning_summary),
        },
        "num_clean_training_triples": len(clean_facts), "num_poisoned_training_triples": len(poisoned_facts),
        "num_injected_triples": len(injected_facts), "num_entities": len(entity_to_id),
        "num_relations": len(relation_to_id), "num_set_B_targets": len(set_b),
        "num_injected_targets": injected_b["count"], "num_not_injected_targets": not_injected_b["count"],
        "num_set_A_test_triples": len(set_a),
        "set_B_MRR_clean": all_b["clean"]["MRR"], "set_B_MRR_poisoned": all_b["poisoned"]["MRR"],
        "set_B_Hits_at_10_clean": all_b["clean"]["Hits@10"],
        "set_B_Hits_at_10_poisoned": all_b["poisoned"]["Hits@10"],
        "set_A_MRR_clean": set_a_metrics["clean"]["MRR"],
        "set_A_MRR_poisoned": set_a_metrics["poisoned"]["MRR"],
        "set_A_Hits_at_10_clean": set_a_metrics["clean"]["Hits@10"],
        "set_A_Hits_at_10_poisoned": set_a_metrics["poisoned"]["Hits@10"],
        "limitations": [
            "One UMLS dataset and one random seed; results are preliminary.",
            "ConvE is an opaque-box victim relative to the TransE attacker surrogate.",
            "The frozen targets were rank-qualified under TransE, not ConvE.",
        ],
    }

    output_paths = [args.output_dir / filename for filename in (
        "set_B_target_rank_comparison.csv", "set_B_attack_metrics.json",
        "attack_effectiveness_summary.json", "set_A_triple_rank_comparison.csv",
        "set_A_clean_performance.json", "evaluation_summary.json",
    )]
    if not args.overwrite:
        existing = [str(path) for path in output_paths if path.exists()]
        if existing:
            raise FileExistsError("Evaluation outputs already exist; pass --overwrite: " + ", ".join(existing))
    b_columns = [
        "target_id", "head", "relation", "tail", "poisoning_status", "injected_for_target",
        "clean_rank", "poisoned_rank", "rank_improvement", "clean_score", "poisoned_score",
        "score_change", "clean_hits_at_10", "poisoned_hits_at_10", "rank_outcome",
    ]
    a_columns = [key for key in b_columns if key not in ("poisoning_status", "injected_for_target")]
    write_csv(output_paths[0], attack_rows, b_columns)
    write_json(output_paths[1], attack_result)
    write_json(output_paths[2], {
        "set_B_all": all_b, "set_B_injected": injected_b, "set_B_not_injected": not_injected_b,
        "note": "Lower target rank / higher Hits@10 and MRR indicates better attack effectiveness",
    })
    write_csv(output_paths[3], clean_rows, a_columns)
    write_json(output_paths[4], set_a_result)
    write_json(output_paths[5], evaluation_summary)
    print(f"\n=== {args.attack_name} UMLS RQ1 — CONVE VICTIM ===")
    print(f"Set B: {len(set_b)} targets (injected={injected_b['count']}, no injection={not_injected_b['count']})")
    print(f"Hits@10  clean={all_b['clean']['Hits@10']:.4f}  poisoned={all_b['poisoned']['Hits@10']:.4f}  delta={all_b['delta_poisoned_minus_clean']['Hits@10']:+.4f}")
    print(f"MRR      clean={all_b['clean']['MRR']:.4f}  poisoned={all_b['poisoned']['MRR']:.4f}  delta={all_b['delta_poisoned_minus_clean']['MRR']:+.4f}")
    print("\n=== PRELIMINARY RQ2 — CONVE VICTIM ===")
    print(f"Set A: {len(set_a)} original test triples")
    print(f"Hits@10  clean={set_a_metrics['clean']['Hits@10']:.4f}  poisoned={set_a_metrics['poisoned']['Hits@10']:.4f}")
    print(f"MRR      clean={set_a_metrics['clean']['MRR']:.4f}  poisoned={set_a_metrics['poisoned']['MRR']:.4f}")
    print(f"Saved evaluation results: {args.output_dir.resolve()}")
    return evaluation_summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate clean vs poisoned ConvE UMLS victims (RQ1/RQ2)")
    parser.add_argument("--clean-checkpoint", type=Path, default=VICTIM_DIR / "clean_conve.pt")
    parser.add_argument("--poisoned-checkpoint", type=Path, default=VICTIM_DIR / "poisoned_conve.pt")
    parser.add_argument("--train", type=Path, default=DATASET_DIR / "train.txt")
    parser.add_argument("--poisoned-train", type=Path, default=POISONING_DIR / "poisoned_train.txt")
    parser.add_argument("--injected", type=Path, default=POISONING_DIR / "injected_triples.tsv")
    parser.add_argument("--valid", type=Path, default=DATASET_DIR / "valid.txt")
    parser.add_argument("--test", type=Path, default=DATASET_DIR / "test.txt")
    parser.add_argument("--targets", type=Path, default=PILOT_DIR / "setup" / "set_B_target_triples.tsv")
    parser.add_argument("--poisoning-summary", type=Path, default=POISONING_DIR / "poisoning_summary.csv")
    parser.add_argument("--training-summary", type=Path, default=VICTIM_DIR / "training_summary.json")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--attack-name", default="MaSS")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
