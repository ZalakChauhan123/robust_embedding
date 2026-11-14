#!/usr/bin/env python3
"""UMLS Figure 8: ConvE score distributions for correct, poisoned and noisy facts.

The main figure uses one matching poisoned ConvE model for all three groups.
Paired clean-model scores are retained as a supplementary control. No training
is performed; the source dataset/checkpoint hashes are checked before scoring.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import sys
from collections import Counter
from pathlib import Path

MASS_DIR = Path(__file__).resolve().parents[2]
if str(MASS_DIR) not in sys.path:
    sys.path.insert(0, str(MASS_DIR))

import torch

from Path_Generation_Module.ConvE_Victim import ConvE
from Path_Generation_Module.TransE_Surrogate import read_names
from Experiment_Setup.train_victim_models import verify_datasets
from Evaluate_victim_models import require_triples, verify_training_sources
from RQ4_Semantic_Plot import summarize_scores, histogram_counts, plot_semantic_information

DATA = MASS_DIR / "dataset_uml"
DEFAULT_EXPERIMENT = MASS_DIR / "outputs" / "Experiments" / "UMLS_feasible_rank11_15_seed42"
GROUPS = ("Correct facts", "Poisoned triplets", "Noisy facts")


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def select_fact_samples(injected, correct_pool, *, num_facts=None, seed=42):
    n = len(injected) if num_facts is None else num_facts
    if n < 1 or n > min(len(injected), len(correct_pool)):
        raise ValueError("Requested facts must be positive and fit both the injected and correct pools without replacement")
    if len(set(injected)) != len(injected) or len(set(correct_pool)) != len(correct_pool):
        raise ValueError("Fact pools must contain unique triples")
    correct = random.Random(seed).sample(correct_pool, n)
    poisoned = list(injected) if n == len(injected) else random.Random(seed + 2).sample(injected, n)
    return correct, poisoned


def make_noisy_facts(correct, entities, excluded, *, seed=43):
    """Make one unique, unobserved head/tail corruption per source fact.

    Choose each available corruption direction with equal probability, then
    choose a replacement uniformly within that direction. No known positives,
    injected facts, targets, unchanged triples, duplicate noise or self-loops.
    These are unobserved synthetic negatives, not verified false statements.
    """
    if not entities or len(set(entities)) != len(entities):
        raise ValueError("Entity list must be nonempty and unique")
    rng, seen, result = random.Random(seed), set(excluded), []
    for h, r, t in correct:
        candidates = {
            "head": [(e, r, t) for e in entities if e != h and e != t and (e, r, t) not in seen],
            "tail": [(h, r, e) for e in entities if e != t and e != h and (h, r, e) not in seen],
        }
        available = [direction for direction in ("head", "tail") if candidates[direction]]
        if not available:
            raise ValueError(f"No legal one-entity noise corruption remains for {(h, r, t)}")
        direction = rng.choice(available)
        triple = rng.choice(candidates[direction])
        seen.add(triple)
        result.append({"triple": triple, "source": (h, r, t), "corrupted_field": direction})
    return result


@torch.inference_mode()
def score_triples(model, triples, entity_to_id, relation_to_id, *, batch_size=256):
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    model.eval()
    encoded = []
    for h, r, t in triples:
        if h not in entity_to_id or t not in entity_to_id or r not in relation_to_id:
            raise ValueError(f"Unknown entity or relation while scoring {(h, r, t)}")
        encoded.append((entity_to_id[h], relation_to_id[r], entity_to_id[t]))
    values = []
    for start in range(0, len(encoded), batch_size):
        batch = torch.tensor(encoded[start:start + batch_size], dtype=torch.long)
        logits = model.score_all_tails(batch[:, 0], batch[:, 1])
        values.extend(logits.gather(1, batch[:, 2, None]).squeeze(1).cpu().tolist())
    if len(values) != len(triples) or any(not math.isfinite(value) for value in values):
        raise ValueError("Scoring produced a missing or nonfinite value")
    return values


def export_analysis(rows, output, *, bins, checkpoint_sha256):
    summary = summarize_scores(rows)
    clean_rows = [{**row, "score": row["clean_score"]} for row in rows]
    clean_summary = summarize_scores(clean_rows)
    edges, counts = histogram_counts(rows, bins=bins)
    histogram = [{"bin_left": edges[index], "bin_right": edges[index + 1],
                  **{group: counts[group][index] for group in GROUPS}} for index in range(len(edges) - 1)]
    relation_counts = Counter((row["group"], row["relation"]) for row in rows)
    relation_rows = [{"group": group, "relation": relation, "count": count}
                     for (group, relation), count in sorted(relation_counts.items())]
    for name, values in (("semantic_information_summary", summary),
                         ("semantic_information_clean_summary", clean_summary), ("histogram_counts", histogram)):
        write_csv(output / f"{name}.csv", values)
        write_json(output / f"{name}.json", values)
    write_csv(output / "relation_counts.csv", relation_rows)
    paths = plot_semantic_information(rows, output, bins=bins)
    write_json(output / "figure8_plot_config.json", {
        "bins": bins, "common_bin_edges": edges, "histogram": "overlapping counts, not density or stacked",
        "trend_curves": {"method": "Gaussian KDE", "bandwidth": "Scott's rule",
                         "count_scale": "density * group sample size * shared bin width", "grid_points": 512,
                         "degenerate_fallback": "half-bin-width Gaussian for singleton or near-constant scores",
                         "display": "existing measured-score range; tails outside this range are not renormalized"},
        "score_column": "score", "score_definition": "raw poisoned ConvE tail logit; higher is more plausible",
        "scoring_checkpoint_sha256": checkpoint_sha256,
        "measured_scores_sha256": file_hash(output / "semantic_information_scores.json"),
        "plot_source_sha256": file_hash(Path(__file__).parent / "RQ4_Semantic_Plot.py"),
        "exports": {key: str(path) for key, path in paths.items()},
    })
    return summary


def run(args):
    experiment = args.experiment_dir.resolve()
    output = (args.output_dir or experiment / "RQ4" / "semantic_information").resolve()
    if args.plot_only:
        manifest = read_json(output / "semantic_information_manifest.json")
        for name, expected in manifest["measured_artifacts_sha256"].items():
            if file_hash(output / name) != expected:
                raise ValueError(f"Measured scores changed: {name}")
        rows = read_json(output / "semantic_information_scores.json")
        export_analysis(rows, output, bins=args.bins,
                        checkpoint_sha256=manifest["scoring_checkpoint_sha256"])
        print(f"Replotted saved scores: {output}")
        return
    if min(args.bins, args.batch_size, args.torch_threads) < 1 or min(args.seed, args.victim_seed) < 0:
        raise ValueError("Bins, batch size and threads must be positive; seeds must be nonnegative")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output}; use --plot-only or a new output directory")
    torch.set_num_threads(args.torch_threads)
    method = (args.method_dir or experiment / "baseline").resolve()
    victim = (args.victim_dir or method / f"seed_{args.victim_seed}" / "conve_victim_models").resolve()
    poisoning = method / "poisoning"
    targets_path = experiment / "setup" / "set_B_target_triples.tsv"
    files = {
        "train": DATA / "train.txt", "valid": DATA / "valid.txt", "test": DATA / "test.txt",
        "entities": DATA / "entities.txt", "relations": DATA / "relations.txt", "targets": targets_path,
        "poisoned_train": poisoning / "poisoned_train.txt", "injected": poisoning / "injected_triples.tsv",
        "poisoning_summary": poisoning / "poisoning_summary.json",
        "training_summary": victim / "training_summary.json",
        "clean_checkpoint": victim / "clean_conve.pt", "poisoned_checkpoint": victim / "poisoned_conve.pt",
        "scoring_code": MASS_DIR / "Path_Generation_Module" / "ConvE_Victim.py",
        "validation_code": MASS_DIR / "Experiment_Setup" / "train_victim_models.py",
        "provenance_code": Path(__file__).parent / "Evaluate_victim_models.py",
        "analysis_code": Path(__file__), "plot_code": Path(__file__).parent / "RQ4_Semantic_Plot.py",
    }
    hashes = {str(path.resolve()): file_hash(path) for path in files.values()}
    training = read_json(files["training_summary"])
    if training.get("victim_model") != "ConvE" or training.get("attack_name") != "MaSS":
        raise ValueError("A ConvE checkpoint trained on the matching MaSS poisoning experiment is required")
    verify_training_sources(
        training, files["clean_checkpoint"], files["poisoned_checkpoint"], files["train"],
        files["poisoned_train"], files["injected"], files["valid"], files["test"], files["targets"],
    )
    entities, relations = read_names(files["entities"]), read_names(files["relations"])
    _, _, integrity = verify_datasets(
        files["train"], files["poisoned_train"], files["injected"], files["valid"], files["test"],
        files["targets"], entities, relations,
    )
    poisoned_model, entity_to_id, relation_to_id, model_config = ConvE.load(files["poisoned_checkpoint"])
    clean_model, clean_e2id, clean_r2id, _ = ConvE.load(files["clean_checkpoint"])
    if (entity_to_id != clean_e2id or relation_to_id != clean_r2id
            or entity_to_id != {name: i for i, name in enumerate(entities)}
            or relation_to_id != {name: i for i, name in enumerate(relations)}):
        raise ValueError("Model entity/relation mappings differ from one another or from UMLS")
    injected = require_triples(files["injected"])
    correct_pool = require_triples(files[args.correct_split])
    correct, poisoned = select_fact_samples(injected, correct_pool, num_facts=args.num_facts, seed=args.seed)
    original_known = set().union(*(set(require_triples(files[name])) for name in ("train", "valid", "test")))
    excluded = original_known | set(injected) | set(require_triples(files["targets"]))
    noise = make_noisy_facts(correct, entities, excluded, seed=args.seed + 1)
    groups = {"Correct facts": correct, "Poisoned triplets": poisoned, "Noisy facts": [item["triple"] for item in noise]}
    if len(set(map(tuple, groups["Noisy facts"]))) != len(correct) or set(groups["Noisy facts"]) & excluded:
        raise AssertionError("Noise uniqueness/exclusion check failed")
    if any(set(groups[first]) & set(groups[second]) for first, second in
           (("Correct facts", "Poisoned triplets"), ("Correct facts", "Noisy facts"), ("Poisoned triplets", "Noisy facts"))):
        raise AssertionError("Fact groups overlap")
    rows = []
    for group in GROUPS:
        triples = groups[group]
        values = score_triples(poisoned_model, triples, entity_to_id, relation_to_id, batch_size=args.batch_size)
        clean_values = score_triples(clean_model, triples, entity_to_id, relation_to_id, batch_size=args.batch_size)
        for index, (triple, score, clean_score) in enumerate(zip(triples, values, clean_values)):
            source = noise[index]["source"] if group == "Noisy facts" else triple
            rows.append({
                "group": group, "sample_id": index + 1, "head": triple[0], "relation": triple[1], "tail": triple[2],
                "score": score, "clean_score": clean_score, "score_change": score - clean_score,
                "source_head": source[0], "source_relation": source[1], "source_tail": source[2],
                "corrupted_field": noise[index]["corrupted_field"] if group == "Noisy facts" else "",
            })
    if any(file_hash(Path(path)) != expected for path, expected in hashes.items()):
        raise AssertionError("An analysis input changed during scoring")
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "semantic_information_scores.csv", rows)
    write_json(output / "semantic_information_scores.json", rows)
    checkpoint_hash = file_hash(files["poisoned_checkpoint"])
    summary = export_analysis(rows, output, bins=args.bins, checkpoint_sha256=checkpoint_hash)
    write_json(output / "semantic_information_manifest.json", {
        "dataset": "UMLS", "experiment_dir": str(experiment), "method_dir": str(method),
        "victim_dir": str(victim), "scoring_model": "ConvE trained on matching poisoned graph",
        "scoring_checkpoint_sha256": checkpoint_hash, "model_config": model_config,
        "victim_training_parameters": training["parameters"],
        "poisoning_parameters": read_json(files["poisoning_summary"])["parameters"],
        "sample_seed": args.seed, "noise_seed": args.seed + 1, "correct_split": args.correct_split,
        "facts_per_group": len(correct), "available_injected": len(injected), "available_correct": len(correct_pool),
        "all_injected_facts_used": len(poisoned) == len(injected), "batch_size": args.batch_size,
        "bins": args.bins, "torch_threads": torch.get_num_threads(), "torch_version": torch.__version__,
        "noise_protocol": "Per sampled correct fact, randomly choose a head/tail direction among those with legal replacements, then a uniform legal entity; retain relation; no self-loops, known positives, injections, targets or duplicate noise.",
        "noise_corruption_counts": dict(Counter(item["corrupted_field"] for item in noise)),
        "score_definition": "Raw tail logits from the same poisoned ConvE for all groups; higher means more plausible; no sigmoid, negation, calibration or rescaling.",
        "clean_model_control": "The same sampled triples are also scored with the paired clean checkpoint; clean_score is not plotted in the main figure.",
        "integrity": integrity, "input_sha256": hashes,
        "measured_artifacts_sha256": {name: file_hash(output / name) for name in
                                     ("semantic_information_scores.csv", "semantic_information_scores.json")},
        "limitations": [
            "Paper Figure 8 uses FB15k-237; clean split, group sampling, noise generation and binning were not specified. These are explicit UMLS choices.",
            "Synthetic corruptions are unobserved negatives, not independently established false facts.",
            "Injected triples were seen during poisoned-model training; high scores are not independent evidence of truth.",
            "Correct facts default to held-out test data; all groups have equal counts but their relation distributions are not exactly matched. See relation_counts.csv.",
            "This is one local poisoning/model seed and a descriptive plausibility comparison, not a calibrated truth or detection test.",
        ],
    })
    for row in summary:
        print(f"{row['group']}: n={row['count']}, mean={row['mean']:.4f}, median={row['median']:.4f}")
    print(f"Figure 8 and reproducible scores saved to {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dir", type=Path, default=DEFAULT_EXPERIMENT)
    parser.add_argument("--method-dir", type=Path)
    parser.add_argument("--victim-dir", type=Path)
    parser.add_argument("--victim-seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--correct-split", choices=("test", "valid", "train"), default="test")
    parser.add_argument("--num-facts", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bins", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--torch-threads", type=int, default=4)
    parser.add_argument("--plot-only", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
