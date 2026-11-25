#!/usr/bin/env python3
"""Train a native DICE model in both roles and compare filtered tail ranks."""

import argparse
from collections import defaultdict
import contextlib
import csv
import math
import os
import sys
import tempfile
from importlib.metadata import version
from pathlib import Path

HERE = Path(__file__).resolve().parent
os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.path.insert(0, str(HERE.parent))

from Test_Triplet.run_experiment import load_candidates, read_names, rescore_candidates
from Experiment_Setup.construct_poisoned_dataset import (
    construct,
    read_triples,
    sha256,
    write_json,
    write_triples,
)

from dice_models import MODEL_NAMES, train_dice_model


TRAINING = {"embedding_dim": 128, "batch_size": 256, "lr": 0.01}
GRAPH_SETTINGS = {"budget": 2, "delta": 0.7, "seed": 42, "selection_policy": "baseline"}
RESULT_COLUMNS = (
    "Poisoned Model", "Victim Model", "Head", "Relation", "Tail",
    "Clean rank", "Poisoned rank", "Rank change",
)


def reserve_run(details: Path) -> Path:
    details.mkdir(parents=True, exist_ok=True)
    number = 1
    while True:
        run_dir = details / f"run_{number:03d}"
        try:
            run_dir.mkdir()
            return run_dir
        except FileExistsError:
            number += 1


def build_poisoned_graph(inputs, paths, targets, candidates, clean, model, scratch):
    feasibility = scratch / "feasibility"
    feasibility_rows = []
    for index, target in enumerate(targets, 1):
        target_id = f"target_{index:03d}"
        rescored = rescore_candidates(candidates[target], model, clean)
        write_json(feasibility / f"{target_id}.json", rescored)
        path_count = len(rescored["ranked_candidate_paths"])
        head, relation, tail = target
        feasibility_rows.append({
            "target_id": target_id, "head": head, "relation": relation, "tail": tail,
            "num_ranked_paths": path_count, "status": "feasible" if path_count else "infeasible",
        })
    write_json(feasibility / "feasibility_summary.json", {
        "set_B_targets_total": len(targets), "targets": feasibility_rows,
    })
    construction = scratch / "construction"
    report = construct(
        targets_file=inputs / "target_triplet.txt",
        feasibility_dir=feasibility,
        output_dir=construction,
        train_file=paths["clean_graph"],
        valid_file=inputs / "valid.txt",
        test_file=inputs / "test.txt",
        **GRAPH_SETTINGS,
    )
    return read_triples(construction / "poisoned_train.txt", require_unique=True), report


def filtered_tail_rank(model, target, tails):
    h, r, t = target
    scores = model.triple_scores([(h, r, candidate) for candidate in tails])
    if len(scores) != len(tails) or not all(map(math.isfinite, scores)):
        raise ValueError("DICE returned invalid scores for the tail candidates")
    score = scores[tails.index(t)]
    better = sum(candidate > score for candidate in scores)
    tied = sum(candidate == score for candidate in scores)
    return 1 + better + (tied - 1) / 2


def run(epochs: int, model_name: str = "TransE") -> Path:
    if epochs < 1:
        raise ValueError("Epochs must be positive")
    if model_name not in MODEL_NAMES:
        raise ValueError(f"Choose one of {', '.join(MODEL_NAMES)}")

    inputs = HERE / "Input"
    input_names = ("train.txt", "valid.txt", "test.txt", "target_triplet.txt",
                   "entities.txt", "relations.txt", "mass_candidates.json")
    input_hashes = {name: sha256(inputs / name) for name in input_names}
    train, valid, test, targets = (
        read_triples(inputs / name, require_unique=True) for name in input_names[:4]
    )
    target_set = set(targets)
    clean, held_out = set(train), set(valid) | set(test)
    original_facts = clean | held_out
    if clean & held_out or target_set & original_facts:
        raise ValueError("Use disjoint training and held-out facts and unseen target triplets")

    entities, relations = (read_names(inputs / name) for name in input_names[4:6])
    entity_set, relation_set = set(entities), set(relations)
    if any(h not in entity_set or t not in entity_set or r not in relation_set
           for h, r, t in original_facts | target_set):
        raise ValueError("An input fact uses an entity or relation outside the configured vocabulary")
    if ({entity for h, _, t in train for entity in (h, t)} != entity_set
            or {r for _, r, _ in train} != relation_set):
        raise ValueError("DICE train-only indexing requires the complete vocabulary in the clean graph")
    known_tails = defaultdict(set)
    for head, relation, tail in original_facts:
        known_tails[head, relation].add(tail)
    candidates = load_candidates(inputs / "mass_candidates.json", targets)

    output = HERE / "Output"
    details = reserve_run(output / "Details" / model_name)
    run_name = details.name
    paths = {
        "clean_graph": HERE / "Graphs" / "Clean" / model_name / run_name / "train.txt",
        "poisoned_graph": HERE / "Graphs" / "Poisoned" / model_name / run_name / "train.txt",
        "poisoned_model": HERE / "Poisoned" / model_name / run_name,
        "clean_victim": HERE / "Victim" / model_name / run_name / "Clean",
        "poisoned_victim": HERE / "Victim" / model_name / run_name / "Poisoned",
    }
    log = details / "training.log"
    print(f"Training DICE {model_name} for {len(targets)} targets. Training log: {log}", flush=True)
    with log.open("w", encoding="utf-8", buffering=1) as stream:
        with contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream):
            write_triples(paths["clean_graph"], train)
            poisoned_model = train_dice_model(
                paths["clean_graph"].parent, paths["poisoned_model"],
                model_name=model_name, epochs=epochs, seed=43, **TRAINING,
            )
            with tempfile.TemporaryDirectory(dir=details) as temporary:
                poisoned, graph_report = build_poisoned_graph(
                    inputs, paths, targets, candidates, clean, poisoned_model, Path(temporary),
                )
            poisoned_set = set(poisoned)
            additions = poisoned_set - clean
            if (not clean <= poisoned_set or not 1 <= len(additions) <= GRAPH_SETTINGS["budget"] * len(targets)
                    or additions & (held_out | target_set)):
                raise ValueError("Expected a clean graph plus new, budgeted facts and no target or held-out facts")
            write_triples(paths["poisoned_graph"], poisoned)
            clean_victim = train_dice_model(
                paths["clean_graph"].parent, paths["clean_victim"],
                model_name=model_name, epochs=epochs, seed=42, **TRAINING,
            )
            poisoned_victim = train_dice_model(
                paths["poisoned_graph"].parent, paths["poisoned_victim"],
                model_name=model_name, epochs=epochs, seed=42, **TRAINING,
            )
            if (clean_victim.entity_to_id != poisoned_victim.entity_to_id
                    or clean_victim.relation_to_id != poisoned_victim.relation_to_id
                    or set(clean_victim.entity_to_id) != entity_set
                    or set(clean_victim.relation_to_id) != relation_set):
                raise ValueError("The victim runs must have identical DICE indexing and the original vocabulary")
            result_rows = []
            candidate_counts = []
            for target in targets:
                excluded = known_tails[target[:2]]
                tails = [entity for entity in entities if entity not in excluded]
                clean_rank = filtered_tail_rank(clean_victim, target, tails)
                poisoned_rank = filtered_tail_rank(poisoned_victim, target, tails)
                result_rows.append((model_name, model_name, *target, f"{clean_rank:g}", f"{poisoned_rank:g}",
                                    f"{poisoned_rank - clean_rank:g}"))
                candidate_counts.append(len(tails))

    if any(sha256(inputs / name) != digest for name, digest in input_hashes.items()):
        raise ValueError("Input files changed during the experiment")
    write_json(details / "config.json", {
        "dicee_version": version("dicee"),
        "model": model_name,
        "training": {**TRAINING, "epochs": epochs, "poisoned_model_seed": 43, "victim_seed": 42},
        "graph_settings": GRAPH_SETTINGS,
        "input_sha256": input_hashes,
        "artifacts": {key: path.relative_to(HERE).as_posix() for key, path in paths.items()},
        "clean_graph_facts": len(train),
        "poisoned_graph_facts": len(poisoned),
        "target_count": len(targets),
        "graph_mode": "One shared poisoned graph for all targets",
        "added_facts": sorted(additions),
        "target_graph_construction": graph_report["targets"],
        "rank_method": "Filtered tail prediction using original train, valid and test facts; average ties",
        "rank_candidates_after_filtering": candidate_counts,
    })
    results = output / "final_result" / f"{model_name}_results.csv"
    results.parent.mkdir(parents=True, exist_ok=True)
    saved_result = details / results.name
    with saved_result.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(RESULT_COLUMNS)
        writer.writerows(result_rows)
    # Publish only complete results. The copy in Details preserves earlier runs.
    pending = details / "latest.csv.tmp"
    pending.write_bytes(saved_result.read_bytes())
    pending.replace(results)
    print(f"Completed {len(result_rows)} target comparisons. Results: {results}", flush=True)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODEL_NAMES, default="TransE")
    parser.add_argument("--epochs", type=int, default=500, help="DICE training epochs per checkpoint")
    args = parser.parse_args()
    run(args.epochs, args.model)


if __name__ == "__main__":
    main()
