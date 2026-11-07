#!/usr/bin/env python3
"""Build DirectAdd and ComAttack-style UMLS comparison datasets.

The original MaSS constructor remains unchanged. These constructors use the
same held-out/Set-B leakage protections and B=2 accounting so their fresh
TransE victims can be evaluated by the existing RQ1/RQ2 evaluator.

Both source attacks were originally formulated mainly to degrade a true fact.
This project instead has absent synthetic Set-B facts and measures promotion.
The output metadata therefore calls both constructions target_promotion_adaptation.
"""
from __future__ import annotations

import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Literal

import torch

MASS_DIR = Path(__file__).resolve().parents[1]
if str(MASS_DIR) not in sys.path:
    sys.path.insert(0, str(MASS_DIR))

from Path_Generation_Module.TransE_Surrogate import Surrogate, read_names  # noqa: E402

DATASET_DIR = MASS_DIR / "dataset_uml"
Triple = tuple[str, str, str]
AttackName = Literal["DirectAdd", "ComAttack"]

SUMMARY_FIELDS = [
    "target_id", "head", "relation", "tail", "original_status",
    "initial_paths", "eligible_paths", "rejected_leakage",
    "rejected_target_overlap", "rejected_invalid", "sampled_paths",
    "selected_paths", "injected_for_target", "new_to_global_training",
    "budget_skipped_paths", "status", "selection_policy",
    "mean_sampled_intermediate_degree", "mean_selected_intermediate_degree",
]


def read_triples(path: Path, *, require_unique: bool = False) -> list[Triple]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing input file: {path}")
    triples: list[Triple] = []
    seen: set[Triple] = set()
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        for line_no, row in enumerate(csv.reader(stream, delimiter="\t"), 1):
            if not row or (len(row) == 1 and not row[0].strip()):
                continue
            if row[0].lstrip().startswith("#"):
                continue
            triple = tuple(value.strip() for value in row)
            if len(triple) != 3 or not all(triple):
                raise ValueError(f"Invalid triple at {path}:{line_no}: {row!r}")
            if require_unique and triple in seen:
                raise ValueError(f"Duplicate triple at {path}:{line_no}: {triple!r}")
            triples.append(triple)  # type: ignore[arg-type]
            seen.add(triple)  # type: ignore[arg-type]
    if not triples:
        raise ValueError(f"Empty triple file: {path}")
    return triples


def write_triples(path: Path, triples: Iterable[Triple]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        csv.writer(stream, delimiter="\t", lineterminator="\n").writerows(triples)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def legal_new_edge(
    edge: Triple,
    *,
    original_known: set[Triple],
    target_set: set[Triple],
    globally_injected: set[Triple],
) -> tuple[bool, str | None]:
    """Enforce direct attack integrity without MaSS semantic/PRA filters."""
    head, _, tail = edge
    if head == tail:
        return False, "invalid"
    if edge in target_set:
        return False, "target_overlap"
    if edge in original_known:
        return False, "leakage"
    if edge in globally_injected:
        return False, "global_duplicate"
    return True, None


def directadd_candidates(
    target: Triple,
    *,
    surrogate: Surrogate,
    entities: list[str],
    relations: list[str],
    original_known: set[Triple],
    target_set: set[Triple],
    globally_injected: set[Triple],
    step: float,
) -> tuple[list[dict[str, Any]], Counter[str]]:
    """Rank direct additions using the promotion form of DirectAdd Eq. 3.

    f(h,r,t)=-||h+r-t||. Promotion shifts are eps_h=-s*d/||d|| and
    eps_t=+s*d/||d||, where d=h+r-t. Candidate score is f(shifted)-f(base).
    """
    head, relation, tail = target
    e2id, r2id = surrogate.entity_to_id, surrogate.relation_to_id
    entity_embeddings = surrogate.model.entity.weight.detach()
    relation_embeddings = surrogate.model.relation.weight.detach()
    h = entity_embeddings[e2id[head]]
    r = relation_embeddings[r2id[relation]]
    t = entity_embeddings[e2id[tail]]
    residual = h + r - t
    norm = torch.linalg.vector_norm(residual, ord=2).clamp_min(1e-12)
    eps_h = -step * residual / norm
    eps_t = step * residual / norm

    records: list[dict[str, Any]] = []
    rejected: Counter[str] = Counter()
    for candidate_relation in relations:
        relation_vector = relation_embeddings[r2id[candidate_relation]]
        base_h = -torch.linalg.vector_norm(h + relation_vector - entity_embeddings, ord=2, dim=1)
        shifted_h = -torch.linalg.vector_norm(h + eps_h + relation_vector - entity_embeddings, ord=2, dim=1)
        base_t = -torch.linalg.vector_norm(entity_embeddings + relation_vector - t, ord=2, dim=1)
        shifted_t = -torch.linalg.vector_norm(entity_embeddings + relation_vector - (t + eps_t), ord=2, dim=1)
        for entity_index, entity in enumerate(entities):
            pairs = (
                ((head, candidate_relation, entity), "head_neighborhood", float((shifted_h[entity_index] - base_h[entity_index]).item())),
                ((entity, candidate_relation, tail), "tail_neighborhood", float((shifted_t[entity_index] - base_t[entity_index]).item())),
            )
            for edge, direction, benefit in pairs:
                legal, reason = legal_new_edge(
                    edge, original_known=original_known, target_set=target_set,
                    globally_injected=globally_injected,
                )
                if not legal:
                    rejected[reason or "invalid"] += 1
                    continue
                records.append({"edge": edge, "direction": direction, "directadd_benefit": benefit})
    records.sort(key=lambda item: (-item["directadd_benefit"], item["edge"]))
    # (h, r_j, t) appears once from each symmetric candidate pool.  It is one
    # fact, not two budget units, so retain its deterministic first occurrence.
    unique_records: list[dict[str, Any]] = []
    seen_edges: set[Triple] = set()
    for record in records:
        if record["edge"] not in seen_edges:
            unique_records.append(record)
            seen_edges.add(record["edge"])
    return unique_records, rejected


def comattack_candidates(
    target: Triple,
    *,
    surrogate: Surrogate,
    entities: list[str],
    relations: list[str],
    original_known: set[Triple],
    target_set: set[Triple],
    globally_injected: set[Triple],
) -> tuple[list[dict[str, Any]], Counter[str]]:
    """Construct a ComAttack-style target-promotion composition body.

    Pair order follows ||r-(r1+r2)||. For each pair, a legal z is ranked by
    sigmoid(f(h,r1,z))*sigmoid(f(z,r2,t)); first viable pair is retained.
    """
    head, target_relation, tail = target
    e2id, r2id = surrogate.entity_to_id, surrogate.relation_to_id
    entity_embeddings = surrogate.model.entity.weight.detach()
    relation_embeddings = surrogate.model.relation.weight.detach()
    target_vector = relation_embeddings[r2id[target_relation]]
    pairs: list[tuple[float, str, str]] = []
    for first_relation in relations:
        first_vector = relation_embeddings[r2id[first_relation]]
        for second_relation in relations:
            distance = float(torch.linalg.vector_norm(
                target_vector - first_vector - relation_embeddings[r2id[second_relation]], ord=2
            ).item())
            pairs.append((distance, first_relation, second_relation))
    pairs.sort(key=lambda item: (item[0], item[1], item[2]))

    rejected: Counter[str] = Counter()
    h_vector, t_vector = entity_embeddings[e2id[head]], entity_embeddings[e2id[tail]]
    for composition_distance, first_relation, second_relation in pairs:
        first_scores = -torch.linalg.vector_norm(
            h_vector + relation_embeddings[r2id[first_relation]] - entity_embeddings, ord=2, dim=1
        )
        second_scores = -torch.linalg.vector_norm(
            entity_embeddings + relation_embeddings[r2id[second_relation]] - t_vector, ord=2, dim=1
        )
        plausibility = torch.sigmoid(first_scores) * torch.sigmoid(second_scores)
        records: list[dict[str, Any]] = []
        for entity_index, middle in enumerate(entities):
            first = (head, first_relation, middle)
            second = (middle, second_relation, tail)
            first_ok, first_reason = legal_new_edge(
                first, original_known=original_known, target_set=target_set,
                globally_injected=globally_injected,
            )
            second_ok, second_reason = legal_new_edge(
                second, original_known=original_known, target_set=target_set,
                globally_injected=globally_injected,
            )
            if not first_ok:
                rejected[first_reason or "invalid"] += 1
                continue
            if not second_ok:
                rejected[second_reason or "invalid"] += 1
                continue
            records.append({
                "first_triple": first, "second_triple": second,
                "r1": first_relation, "r2": second_relation, "intermediate_entity": middle,
                "composition_distance": composition_distance,
                "body_plausibility": float(plausibility[entity_index].item()),
                "first_triple_score": float(first_scores[entity_index].item()),
                "second_triple_score": float(second_scores[entity_index].item()),
            })
        if records:
            records.sort(key=lambda item: (-item["body_plausibility"], item["intermediate_entity"]))
            return records, rejected
    return [], rejected


def construct(
    attack: AttackName,
    *,
    targets_file: Path,
    output_dir: Path,
    train_file: Path = DATASET_DIR / "train.txt",
    valid_file: Path = DATASET_DIR / "valid.txt",
    test_file: Path = DATASET_DIR / "test.txt",
    entities_file: Path = DATASET_DIR / "entities.txt",
    relations_file: Path = DATASET_DIR / "relations.txt",
    surrogate_checkpoint: Path = MASS_DIR / "outputs" / "Path_Generation" / "transe_umls.pt",
    budget: int = 2,
    seed: int = 42,
    directadd_step: float = 0.1,
) -> dict[str, Any]:
    if attack not in ("DirectAdd", "ComAttack"):
        raise ValueError(f"Unsupported attack: {attack}")
    if budget != 2:
        raise ValueError("These predeclared comparison baselines use exactly B=2")
    if directadd_step <= 0:
        raise ValueError("directadd_step must be positive")
    output_dir = output_dir.resolve()
    if output_dir == DATASET_DIR.resolve() or (output_dir / "poisoned_train.txt").resolve() == train_file.resolve():
        raise ValueError("Output directory must not overwrite the source UMLS training data")

    targets = read_triples(targets_file, require_unique=True)
    train = read_triples(train_file, require_unique=True)
    valid = read_triples(valid_file, require_unique=True)
    test = read_triples(test_file, require_unique=True)
    entities, relations = read_names(entities_file), read_names(relations_file)
    train_set, valid_set, test_set, target_set = set(train), set(valid), set(test), set(targets)
    original_known = train_set | valid_set | test_set
    if train_set & (valid_set | test_set) or target_set & original_known:
        raise ValueError("Original data splits or Set B targets overlap")
    surrogate = Surrogate.load(surrogate_checkpoint)
    if set(entities) != set(surrogate.entity_to_id) or set(relations) != set(surrogate.relation_to_id):
        raise ValueError("Surrogate mappings differ from current UMLS dictionaries")

    globally_injected: set[Triple] = set()
    summary_rows: list[dict[str, Any]] = []
    candidate_audit: list[dict[str, Any]] = []
    selected_audit: list[dict[str, Any]] = []
    for target_index, target in enumerate(targets, 1):
        target_id = f"target_{target_index:03d}"
        if attack == "DirectAdd":
            candidates, rejected = directadd_candidates(
                target, surrogate=surrogate, entities=entities, relations=relations,
                original_known=original_known, target_set=target_set,
                globally_injected=globally_injected, step=directadd_step,
            )
            chosen = candidates[:budget]
            injected = [item["edge"] for item in chosen]
            selected = [
                {"edge": list(item["edge"]), "direction": item["direction"],
                 "directadd_benefit": item["directadd_benefit"]} for item in chosen
            ]
            preview = [
                {"edge": list(item["edge"]), "direction": item["direction"],
                 "directadd_benefit": item["directadd_benefit"]} for item in candidates[:20]
            ]
        else:
            candidates, rejected = comattack_candidates(
                target, surrogate=surrogate, entities=entities, relations=relations,
                original_known=original_known, target_set=target_set,
                globally_injected=globally_injected,
            )
            chosen = candidates[:1]  # Each selected composition body costs two new facts.
            injected = [] if not chosen else [chosen[0]["first_triple"], chosen[0]["second_triple"]]
            selected = [
                {"first_triple": list(item["first_triple"]), "second_triple": list(item["second_triple"]),
                 "r1": item["r1"], "r2": item["r2"], "intermediate_entity": item["intermediate_entity"],
                 "composition_distance": item["composition_distance"], "body_plausibility": item["body_plausibility"]}
                for item in chosen
            ]
            preview = [
                {"first_triple": list(item["first_triple"]), "second_triple": list(item["second_triple"]),
                 "r1": item["r1"], "r2": item["r2"], "intermediate_entity": item["intermediate_entity"],
                 "composition_distance": item["composition_distance"], "body_plausibility": item["body_plausibility"]}
                for item in candidates[:20]
            ]
        attributed = set(injected)
        if len(attributed) > budget:
            raise AssertionError("Per-target budget exceeded")
        for edge in attributed:
            legal, reason = legal_new_edge(
                edge, original_known=original_known, target_set=target_set,
                globally_injected=globally_injected,
            )
            if not legal:
                raise AssertionError(f"Illegal selected edge ({reason}): {edge}")
        globally_injected.update(attributed)
        status = "injected" if attributed else "no_eligible_paths_after_leakage_filter"
        row = {
            "target_id": target_id, "head": target[0], "relation": target[1], "tail": target[2],
            "original_status": "feasible" if candidates else "infeasible",
            "initial_paths": len(candidates), "eligible_paths": len(candidates),
            "rejected_leakage": rejected.get("leakage", 0),
            "rejected_target_overlap": rejected.get("target_overlap", 0),
            "rejected_invalid": rejected.get("invalid", 0),
            "sampled_paths": len(candidates), "selected_paths": len(chosen),
            "injected_for_target": len(attributed), "new_to_global_training": len(attributed),
            "budget_skipped_paths": max(0, len(candidates) - len(chosen)), "status": status,
            "selection_policy": attack,
            "mean_sampled_intermediate_degree": 0.0, "mean_selected_intermediate_degree": 0.0,
        }
        summary_rows.append(row)
        candidate_audit.append({
            "target_id": target_id, "target_triple": list(target), "num_eligible_candidates": len(candidates),
            "rejections": dict(rejected), "top_candidates": preview,
        })
        selected_audit.append({
            "target_id": target_id, "target_triple": list(target), "status": status,
            "attributed_injected_triples": [list(edge) for edge in sorted(attributed)], "selected": selected,
        })
        print(f"[{target_index:03d}/{len(targets):03d}] {target_id}: candidates={len(candidates)} "
              f"added={len(attributed)} | {status}", flush=True)

    poisoned = train + sorted(globally_injected)
    if set(poisoned) != train_set | globally_injected:
        raise AssertionError("Poisoned train must be exact clean union injected")
    output_dir.mkdir(parents=True, exist_ok=True)
    write_triples(output_dir / "poisoned_train.txt", poisoned)
    write_triples(output_dir / "injected_triples.tsv", sorted(globally_injected))
    with (output_dir / "poisoning_summary.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(summary_rows)
    write_json(output_dir / "eligible_candidate_paths.json", {
        "attack": attack, "implementation_variant": "target_promotion_adaptation",
        "legal_filter": "new non-self-loop triples only; no original KG, Set-B, or global duplicate facts",
        "targets": candidate_audit,
    })
    write_json(output_dir / "selected_poisoning_paths.json", {
        "attack": attack, "implementation_variant": "target_promotion_adaptation",
        "budget_distinct_new_triples_per_target": budget, "attack_seed": seed, "targets": selected_audit,
    })
    report = {
        "experiment": f"{attack} target-promotion adaptation for UMLS rank-11--15 Set B",
        "attack": attack, "implementation_variant": "target_promotion_adaptation",
        "parameters": {
            "budget_distinct_new_triples_per_target": budget, "attack_seed": seed,
            "directadd_lambda_2": 1.0, "directadd_lambda_3": 1.0,
            "directadd_promotion_step": directadd_step, "comattack_pattern": "composition",
        },
        "method_note": (
            "DirectAdd Eq. 3 ranking over direct facts, promotion sign, no semantic/PRA/delta filtering."
            if attack == "DirectAdd" else
            "ComAttack-style TransE composition relation pair and two legal body facts; target-promotion adaptation."
        ),
        "source_files": {
            "targets": str(targets_file), "train": str(train_file), "valid": str(valid_file),
            "test": str(test_file), "surrogate_checkpoint": str(surrogate_checkpoint),
        },
        "source_sha256": {
            "targets": sha256_file(targets_file), "train": sha256_file(train_file),
            "valid": sha256_file(valid_file), "test": sha256_file(test_file),
            "surrogate_checkpoint": sha256_file(surrogate_checkpoint),
        },
        "num_targets_set_B": len(targets),
        "num_targets_with_injections": sum(row["status"] == "injected" for row in summary_rows),
        "num_targets_not_injected": sum(row["status"] != "injected" for row in summary_rows),
        "num_clean_train_triples": len(train), "num_unique_injected_triples": len(globally_injected),
        "num_poisoned_train_triples": len(poisoned),
        "injected_train_overlap": len(globally_injected & train_set),
        "injected_valid_overlap": len(globally_injected & valid_set),
        "injected_test_overlap": len(globally_injected & test_set),
        "injected_set_B_overlap": len(globally_injected & target_set), "targets": summary_rows,
    }
    write_json(output_dir / "poisoning_summary.json", report)
    print("\n=== ATTACK BASELINE DATASET CONSTRUCTION ===")
    for key in ("num_targets_set_B", "num_targets_with_injections", "num_unique_injected_triples",
                "num_clean_train_triples", "num_poisoned_train_triples"):
        print(f"{key}: {report[key]}")
    print(f"Saved: {output_dir}")
    return report
