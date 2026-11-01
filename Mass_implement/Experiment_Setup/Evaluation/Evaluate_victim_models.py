#!/usr/bin/env python3
"""
NOTE - RQ1 attack effectiveness & preliminary RQ2 stealthiness.

- Run from Mass_implement/:
    python Experiment_Setup/evaluate_victim_models.py

MaSS paper Metric -- filtered TAIL prediction Hits@10 and MRR.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch

MASS_DIR = Path(__file__).resolve().parents[2]
PATH_MODULE_DIR = MASS_DIR / "Path_Generation_Module"
if str(PATH_MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(PATH_MODULE_DIR))

from TransE_Surrogate import Surrogate, read_triples  # Same model/checkpoint format.

DATASET_DIR = MASS_DIR / "dataset_uml"
PILOT_DIR = MASS_DIR / "outputs" / "Experiments" / "UMLS_pilot"
VICTIM_DIR = PILOT_DIR / "victim_models"
POISONING_DIR = PILOT_DIR / "poisoning"
DEFAULT_OUTPUT_DIR = PILOT_DIR / "evaluation"

Triple = tuple[str, str, str]


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require_triples(path: Path) -> list[Triple]:
    if not path.is_file():
        raise FileNotFoundError(f"Required triple file not found: {path}")
    triples = read_triples(path)
    if not triples or len(triples) != len(set(triples)):
        raise ValueError(f"Triples must be nonempty and unique: {path}")
    return triples


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Required JSON file not found: {path}")
    obj = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(obj, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return obj


def load_poisoning_status(path: Path, targets: list[Triple]) -> list[dict[str, str]]:
    """Require one matching provenance record per original-order Set B target."""
    if not path.is_file():
        raise FileNotFoundError(f"Poisoning summary missing: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != len(targets):
        raise ValueError(f"Expected {len(targets)} target rows in {path}, got {len(rows)}")
    for i, (record, triple) in enumerate(zip(rows, targets), 1):
        expected_id = f"target_{i:03d}"
        found = (record.get("head"), record.get("relation"), record.get("tail"))
        if record.get("target_id") != expected_id or found != triple:
            raise ValueError(f"Target identity/order mismatch for {expected_id}: {found}")
        if record.get("status") not in {
            "injected", "originally_infeasible", "no_eligible_paths_after_leakage_filter",
            "not_selected_by_random_sampling_or_budget",
        }:
            raise ValueError(f"Unexpected poisoning status for {expected_id}: {record.get('status')}")
    return rows


def encode_all(triples: list[Triple], e2id: dict[str, int], r2id: dict[str, int]) -> list[tuple[int, int, int]]:
    encoded = []
    for triple in triples:
        h, r, t = triple
        try:
            encoded.append((e2id[h], r2id[r], e2id[t]))
        except KeyError as exc:
            raise ValueError(f"Entity/relation is missing from checkpoint: {triple}") from exc
    return encoded


def make_true_tail_index(original_facts: list[tuple[int, int, int]]) -> dict[tuple[int, int], set[int]]:
    by_hr: dict[tuple[int, int], set[int]] = defaultdict(set)
    for h, r, t in original_facts:
        by_hr[(h, r)].add(t)
    return dict(by_hr)


@torch.no_grad()
def tail_rank(
    victim: Surrogate,
    encoded_target: tuple[int, int, int],
    true_tails_by_hr: dict[tuple[int, int], set[int]],
    candidate_ids: torch.Tensor,
) -> tuple[float, float]:
    """Filtered tail rank and target score (high = plausible).

    Use average ranks for ties, like existing Model_Evaluation.py.
    The true filter set is fixed ORIGINAL train/valid/test for BOTH models.
    """
    h, r, t = encoded_target
    n = candidate_ids.numel()
    heads = torch.full((n,), h, dtype=torch.long)
    relations = torch.full((n,), r, dtype=torch.long)
    scores = victim.model.score(heads, relations, candidate_ids)
    target_score = float(scores[t].item())
    scores = scores.clone()
    others = true_tails_by_hr.get((h, r), set()) - {t}
    if others:
        scores[sorted(others)] = -torch.inf
    n_better = int((scores > target_score).sum().item())
    n_tied = int((scores == target_score).sum().item())
    rank = float(1 + n_better + (n_tied - 1) / 2)
    if not 1 <= rank <= n:
        raise AssertionError(f"Invalid rank {rank} for {encoded_target}")
    return rank, target_score


def compute_metrics(ranks: list[float]) -> dict[str, float | int]:
    if not ranks:
        raise ValueError("Cannot compute metrics for an empty rank list")
    return {
        "count": len(ranks),
        "MRR": sum(1.0 / r for r in ranks) / len(ranks),
        "Hits@1": sum(r <= 1 for r in ranks) / len(ranks),
        "Hits@3": sum(r <= 3 for r in ranks) / len(ranks),
        "Hits@10": sum(r <= 10 for r in ranks) / len(ranks),
        "Mean_Rank": sum(ranks) / len(ranks),
        "Median_Rank": float(sorted(ranks)[len(ranks)//2]) if len(ranks) % 2 else
                       (sorted(ranks)[len(ranks)//2 - 1] + sorted(ranks)[len(ranks)//2]) / 2,
    }


def compare_group(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"count": 0, "note": "No targets in this group"}
    clean = compute_metrics([r["clean_rank"] for r in rows])
    poisoned = compute_metrics([r["poisoned_rank"] for r in rows])
    delta = {name: poisoned[name] - clean[name]
             for name in ("MRR", "Hits@1", "Hits@3", "Hits@10", "Mean_Rank", "Median_Rank")}
    return {
        "count": len(rows),
        "clean": clean,
        "poisoned": poisoned,
        "delta_poisoned_minus_clean": delta,
        "num_improved_rank": sum(r["rank_improvement"] > 0 for r in rows),
        "num_worsened_rank": sum(r["rank_improvement"] < 0 for r in rows),
        "num_unchanged_rank": sum(r["rank_improvement"] == 0 for r in rows),
        "num_new_hits_at_10": sum(r["clean_rank"] > 10 and r["poisoned_rank"] <= 10 for r in rows),
        "num_lost_hits_at_10": sum(r["clean_rank"] <= 10 and r["poisoned_rank"] > 10 for r in rows),
        "mean_rank_improvement_clean_minus_poisoned": (
            sum(r["rank_improvement"] for r in rows) / len(rows)
        ),
    }


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_csv(path: Path, records: list[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(records)


def verify_training_sources(
    training_summary: dict[str, Any], clean_ckpt: Path, poisoned_ckpt: Path,
    clean_train: Path, poisoned_train: Path, injected: Path, valid: Path,
    test: Path, targets: Path,
) -> None:
    """Ensure evaluation references the exact sources used for victim training."""
    victim_meta = training_summary.get("victims", {})
    dataset_meta = training_summary.get("datasets", {})
    expected_files = {
        "clean_train": clean_train, "poisoned_train": poisoned_train,
        "injected_triples": injected, "validation": valid,
        "test": test, "set_B_targets": targets,
    }
    for key, path in expected_files.items():
        expected_hash = dataset_meta.get("source_sha256", {}).get(key)
        if not expected_hash:
            raise ValueError(f"Training summary missing source hash: {key}")
        if sha256_file(path) != expected_hash:
            raise ValueError(f"Source file changed since victim training: {path}")
    for name, path in (("clean", clean_ckpt), ("poisoned", poisoned_ckpt)):
        expected_hash = victim_meta.get(name, {}).get("checkpoint_sha256")
        if not expected_hash or sha256_file(path) != expected_hash:
            raise ValueError(f"Checkpoint differs from training summary: {path}")
    if not training_summary.get("same_initial_embeddings_verified"):
        raise ValueError("Training summary did not verify identical initial embeddings")


def run(args: argparse.Namespace) -> dict[str, Any]:
    input_paths = [args.clean_checkpoint, args.poisoned_checkpoint, args.train,
                   args.poisoned_train, args.injected, args.valid, args.test,
                   args.targets, args.poisoning_summary, args.training_summary]
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
    verify_training_sources(training_metadata, args.clean_checkpoint, args.poisoned_checkpoint,
                            args.train, args.poisoned_train, args.injected, args.valid,
                            args.test, args.targets)

    clean_model = Surrogate.load(args.clean_checkpoint)
    poisoned_model = Surrogate.load(args.poisoned_checkpoint)
    if clean_model.entity_to_id != poisoned_model.entity_to_id or (
        clean_model.relation_to_id != poisoned_model.relation_to_id
    ):
        raise ValueError("Clean/poisoned checkpoint ID mappings differ")
    e2id, r2id = clean_model.entity_to_id, clean_model.relation_to_id
    known_original = encode_all(list(original_set), e2id, r2id)
    filter_index = make_true_tail_index(known_original)
    candidate_ids = torch.arange(len(e2id), dtype=torch.long)

    def evaluate(triples: list[Triple], *, statuses: list[dict[str, str]] | None = None) -> list[dict[str, Any]]:
        encoded = encode_all(triples, e2id, r2id)
        rows: list[dict[str, Any]] = []
        for index, (triple, triple_ids) in enumerate(zip(triples, encoded), 1):
            clean_rank, clean_score = tail_rank(clean_model, triple_ids, filter_index, candidate_ids)
            poison_rank, poison_score = tail_rank(poisoned_model, triple_ids, filter_index, candidate_ids)
            rec = {
                "target_id": f"target_{index:03d}" if statuses is not None else f"test_{index:04d}",
                "head": triple[0], "relation": triple[1], "tail": triple[2],
                "clean_rank": clean_rank, "poisoned_rank": poison_rank,
                "rank_improvement": clean_rank - poison_rank,
                "clean_score": clean_score, "poisoned_score": poison_score,
                "score_change": poison_score - clean_score,
                "clean_hits_at_10": int(clean_rank <= 10),
                "poisoned_hits_at_10": int(poison_rank <= 10),
                "rank_outcome": "improved" if poison_rank < clean_rank else (
                    "worsened" if poison_rank > clean_rank else "unchanged"),
            }
            if statuses is not None:
                rec["poisoning_status"] = statuses[index - 1]["status"]
                rec["injected_for_target"] = int(statuses[index - 1]["injected_for_target"])
            rows.append(rec)
        return rows

    print(f"Loaded victim checkpoints; {len(e2id)} entities, {len(r2id)} relations.")
    print(f"Evaluating Set B ({len(set_b)} attacker target triples)...", flush=True)
    attack_rows = evaluate(set_b, statuses=provenance)
    print(f"Evaluating Set A ({len(set_a)} original test triples)...", flush=True)
    clean_rows = evaluate(set_a)

    all_b = compare_group(attack_rows)
    injected_b = compare_group([r for r in attack_rows if r["poisoning_status"] == "injected"])
    not_injected_b = compare_group([r for r in attack_rows if r["poisoning_status"] != "injected"])
    set_a_metrics = compare_group(clean_rows)

    attack_result = {
        "research_question": "RQ1 — attack effectiveness (TransE-only UMLS pilot)",
        "prediction": "TAIL only: (h*, r*, ?) ranked against every known entity",
        "ranking_protocol": "filtered; same original train+valid+test facts for both models",
        "filter_excludes": "injected edges and synthetic Set B targets",
        "tie_policy": "average rank, consistent with existing Model_Evaluation.py",
        "interpretation": "Higher poisoned MRR/Hits@10, or positive rank improvement, favors the attacker",
        "set_B_all_targets": all_b,
        "set_B_injected_targets": injected_b,
        "set_B_not_injected_targets": not_injected_b,
        "baseline_comparison_note": "No DirectAdd or ComAttack implementation in this pilot",
        "opaque_box_note": "TransE surrogate and TransE victims are same architecture; opaque-box transfer is not tested",
    }
    set_a_result = {
        "research_question": "RQ2 — preliminary stealthiness check on Set A",
        "prediction": "TAIL only to match the paper's evaluation protocol",
        "set_A_original_test_triples": set_a_metrics,
        "interpretation": "Small clean-to-poisoned degradation in Hits@10/MRR favors stealthiness",
    }
    evaluation_summary = {
        "experiment": "MaSS UMLS pilot: TransE victim evaluation",
        "input_files": {
            "clean_victim": str(args.clean_checkpoint),
            "poisoned_victim": str(args.poisoned_checkpoint),
            "set_A": str(args.test), "set_B": str(args.targets),
            "training_summary": str(args.training_summary),
            "poisoning_summary": str(args.poisoning_summary),
        },
        "num_clean_training_triples": len(clean_facts),
        "num_poisoned_training_triples": len(poisoned_facts),
        "num_injected_triples": len(injected_facts),
        "num_entities": len(e2id),
        "num_relations": len(r2id),
        "num_set_B_targets": len(set_b),
        "num_injected_targets": injected_b["count"],
        "num_not_injected_targets": not_injected_b["count"],
        "num_set_A_test_triples": len(set_a),
        "set_B_MRR_clean": all_b["clean"]["MRR"],
        "set_B_MRR_poisoned": all_b["poisoned"]["MRR"],
        "set_B_Hits_at_10_clean": all_b["clean"]["Hits@10"],
        "set_B_Hits_at_10_poisoned": all_b["poisoned"]["Hits@10"],
        "set_A_MRR_clean": set_a_metrics["clean"]["MRR"],
        "set_A_MRR_poisoned": set_a_metrics["poisoned"]["MRR"],
        "set_A_Hits_at_10_clean": set_a_metrics["clean"]["Hits@10"],
        "set_A_Hits_at_10_poisoned": set_a_metrics["poisoned"]["Hits@10"],
        "limitations": [
            "One UMLS dataset and one random seed; results are preliminary.",
            "TransE only: no opaque-box KGE transferability test yet.",
            "This evaluation does not compare against DirectAdd or ComAttack.",
        ],
    }

    output_paths = [args.output_dir / f for f in (
        "set_B_target_rank_comparison.csv", "set_B_attack_metrics.json",
        "attack_effectiveness_summary.json", "set_A_triple_rank_comparison.csv",
        "set_A_clean_performance.json", "evaluation_summary.json",
    )]
    if not args.overwrite:
        existing = [str(p) for p in output_paths if p.exists()]
        if existing:
            raise FileExistsError("Evaluation outputs already exist; pass --overwrite to replace: " + ", ".join(existing))

    b_columns = ["target_id", "head", "relation", "tail", "poisoning_status",
                 "injected_for_target", "clean_rank", "poisoned_rank", "rank_improvement",
                 "clean_score", "poisoned_score", "score_change", "clean_hits_at_10",
                 "poisoned_hits_at_10", "rank_outcome"]
    a_columns = [k for k in b_columns if k not in ("poisoning_status", "injected_for_target")]
    write_csv(output_paths[0], attack_rows, b_columns)
    write_json(output_paths[1], attack_result)
    write_json(output_paths[2], {
        "set_B_all": all_b, "set_B_injected": injected_b,
        "set_B_not_injected": not_injected_b,
        "note": "Lower target rank / higher Hits@10 and MRR indicates better attack effectiveness",
    })
    write_csv(output_paths[3], clean_rows, a_columns)
    write_json(output_paths[4], set_a_result)
    write_json(output_paths[5], evaluation_summary)

    print("\n=== MaSS UMLS RQ1 — ATTACK EFFECTIVENESS ===")
    print(f"Set B: {len(set_b)} targets (injected={injected_b['count']}, no injection={not_injected_b['count']})")
    print(f"Hits@10  clean={all_b['clean']['Hits@10']:.4f}  poisoned={all_b['poisoned']['Hits@10']:.4f}  delta={all_b['delta_poisoned_minus_clean']['Hits@10']:+.4f}")
    print(f"MRR      clean={all_b['clean']['MRR']:.4f}  poisoned={all_b['poisoned']['MRR']:.4f}  delta={all_b['delta_poisoned_minus_clean']['MRR']:+.4f}")
    print(f"Improved targets: {all_b['num_improved_rank']}; worsened: {all_b['num_worsened_rank']}; unchanged: {all_b['num_unchanged_rank']}")
    print("\n=== PRELIMINARY RQ2 — STEALTHINESS ===")
    print(f"Set A: {len(set_a)} original test triples")
    print(f"Hits@10  clean={set_a_metrics['clean']['Hits@10']:.4f}  poisoned={set_a_metrics['poisoned']['Hits@10']:.4f}")
    print(f"MRR      clean={set_a_metrics['clean']['MRR']:.4f}  poisoned={set_a_metrics['poisoned']['MRR']:.4f}")
    print(f"Saved evaluation results: {args.output_dir.resolve()}")
    return evaluation_summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate MaSS UMLS clean vs poisoned TransE victims (RQ1, preliminary RQ2)")
    parser.add_argument("--clean-checkpoint", type=Path, default=VICTIM_DIR / "clean_transe.pt")
    parser.add_argument("--poisoned-checkpoint", type=Path, default=VICTIM_DIR / "poisoned_transe.pt")
    parser.add_argument("--train", type=Path, default=DATASET_DIR / "train.txt")
    parser.add_argument("--poisoned-train", type=Path, default=POISONING_DIR / "poisoned_train.txt")
    parser.add_argument("--injected", type=Path, default=POISONING_DIR / "injected_triples.tsv")
    parser.add_argument("--valid", type=Path, default=DATASET_DIR / "valid.txt")
    parser.add_argument("--test", type=Path, default=DATASET_DIR / "test.txt")
    parser.add_argument("--targets", type=Path, default=PILOT_DIR / "setup" / "set_B_target_triples.tsv")
    parser.add_argument("--poisoning-summary", type=Path, default=POISONING_DIR / "poisoning_summary.csv")
    parser.add_argument("--training-summary", type=Path, default=VICTIM_DIR / "training_summary.json")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing evaluation reports")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
