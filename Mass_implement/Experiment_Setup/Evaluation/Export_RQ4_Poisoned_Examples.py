#!/usr/bin/env python3
"""Export two saved UMLS MaSS poisoning paths in the layout of Table 5."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import textwrap
from pathlib import Path

MASS_DIR = Path(__file__).resolve().parents[2]
DEFAULT_EXPERIMENT = MASS_DIR / "outputs" / "Experiments" / "UMLS_feasible_rank11_15_seed42"
DEFAULT_OUTPUT = MASS_DIR / "outputs" / "Final Results" / "RQ - 4"
STEM = "RQ-4 -- Example of Poisoned Triplet Generation (Table 5)"


def read_triples(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = [tuple(row) for row in csv.reader(stream, delimiter="\t") if row]
    if any(len(row) != 3 for row in rows) or len(rows) != len(set(rows)):
        raise ValueError(f"Invalid or duplicate triples: {path}")
    return rows


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def triple_text(triple, width):
    return textwrap.fill("(" + ", ".join(triple) + ")", width=width,
                         subsequent_indent=" ", break_long_words=False, break_on_hyphens=False)


def export(args):
    experiment = args.experiment_dir.resolve()
    poisoning = experiment / "baseline" / "poisoning"
    sources = {
        "selected_paths": poisoning / "selected_poisoning_paths.json",
        "injected_triples": poisoning / "injected_triples.tsv",
        "poisoning_summary": poisoning / "poisoning_summary.json",
        "targets": experiment / "setup" / "set_B_target_triples.tsv",
        **{split: MASS_DIR / "dataset_uml" / f"{split}.txt" for split in ("train", "valid", "test")},
    }
    hashes = {str(path): file_hash(path) for path in sources.values()}
    payload = json.loads(sources["selected_paths"].read_text(encoding="utf-8-sig"))
    summary = json.loads(sources["poisoning_summary"].read_text(encoding="utf-8-sig"))
    clean = set(read_triples(sources["train"]))
    held_out = set(read_triples(sources["valid"])) | set(read_triples(sources["test"]))
    injected = set(read_triples(sources["injected_triples"]))
    targets = set(read_triples(sources["targets"]))
    if injected & (clean | held_out | targets):
        raise ValueError("Saved poisoning data overlaps original facts or target triples")
    examples = []
    for item in payload["targets"]:
        selected = item["stealthiness"]["selected_candidate_paths"]
        if item["status"] != "injected" or len(selected) != 1:
            continue
        path = selected[0]
        target = tuple(item["target_triple"])
        first, second = tuple(path["first_triple"]), tuple(path["second_triple"])
        pair = {first, second}
        if (len(pair) != 2 or pair & clean
                or pair != {tuple(edge) for edge in item["attributed_injected_triples"]}
                or pair != {tuple(edge) for edge in item["new_to_global_training"]}):
            continue
        if (target not in targets or target in injected or not pair <= injected
                or first[0] != target[0] or first[2] != second[0] or second[2] != target[2]
                or any(h == t for h, _, t in pair)):
            raise ValueError(f"Invalid selected path at {item['target_id']}")
        examples.append({
            "target_id": item["target_id"], "target_triple": list(target),
            "poisoned_triplets": [list(first), list(second)], "intermediate_entity": first[2],
            "path_rank": path["rank"], "lower_bound_score": path["lower_bound_score"],
            "both_new_to_clean_train": True, "both_present_in_saved_injected_set": True,
        })
        if len(examples) == 2:
            break
    if len(examples) != 2:
        raise ValueError("Need two targets with one complete selected two-new-edge path")
    for path, expected in hashes.items():
        if file_hash(Path(path)) != expected:
            raise ValueError("An input changed during example extraction")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    paths = {extension: output / f"{STEM}.{extension}" for extension in ("png", "pdf", "csv", "json", "md")}
    rows = []
    for example in examples:
        target = example["target_triple"]
        first, second = example["poisoned_triplets"]
        rows.append({"target_id": example["target_id"], "target_head": target[0],
                     "target_relation": target[1], "target_tail": target[2],
                     "first_head": first[0], "first_relation": first[1], "first_tail": first[2],
                     "second_head": second[0], "second_relation": second[1], "second_tail": second[2],
                     "path_rank": example["path_rank"], "both_triplets_newly_injected": True})
    with paths["csv"].open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    seed = summary["parameters"]["seed"]
    budget = summary["parameters"]["budget_distinct_new_triples_per_target"]
    metadata = {
        "dataset": "UMLS", "method": "MaSS baseline", "seed": seed, "budget_per_target": budget,
        "selection": "First two saved targets in target order with exactly one selected two-edge path; both edges newly injected globally.",
        "examples": examples, "input_sha256": hashes,
        "note": "Targets were not directly injected. Examples are generated poisoning facts, not assertions of verified biomedical truth.",
    }
    paths["json"].write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    markdown = ["# Table 5: Examples of MaSS poisoned triplets on UMLS", "",
                "| Target triplet | Poisoned triplets (newly injected two-hop path) |",
                "|---|---|"]
    for example in examples:
        target = "(" + ", ".join(example["target_triple"]) + ")"
        first, second = ["(" + ", ".join(edge) + ")" for edge in example["poisoned_triplets"]]
        markdown.append(f"| `{target}` | `{first}` → `{second}` |")
    markdown += ["", f"Saved MaSS baseline; seed {seed}, budget {budget}. Targets: " +
                 ", ".join(example["target_id"] for example in examples) + ".",
                 "Both displayed supporting triplets per target are in the saved injected set and absent from original training/validation/test data.",
                 "The target triplets themselves were not injected. These examples do not establish factual truth."]
    paths["md"].write_text("\n".join(markdown) + "\n", encoding="utf-8")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(11.8, 4.4))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(0.5, 0.97, "UMLS — Examples of MaSS poisoned triplets (Table 5)",
            ha="center", va="center", fontsize=14, weight="bold")
    ax.text(0.205, 0.82, "Target triplet", ha="center", va="center", fontsize=12, weight="bold")
    ax.text(0.705, 0.82, "Poisoned triplets", ha="center", va="center", fontsize=12, weight="bold")
    for y, example in zip((0.59, 0.28), examples):
        first, second = example["poisoned_triplets"]
        target_label = triple_text(example["target_triple"], 43)
        path_label = triple_text(first, 69) + "\n→\n" + triple_text(second, 69)
        ax.text(0.205, y, target_label, ha="center", va="center", fontsize=10.5,
                fontfamily="DejaVu Sans Mono", linespacing=1.5)
        ax.text(0.705, y, path_label, ha="center", va="center", fontsize=10.5,
                fontfamily="DejaVu Sans Mono", linespacing=1.4)
    for y in (0.89, 0.75, 0.435, 0.105):
        ax.plot((0.01, 0.99), (y, y), color="black", linewidth=1.0 if y in (0.89, 0.105) else 0.6)
    ax.plot((0.41, 0.41), (0.105, 0.89), color="black", linewidth=0.6)
    ax.text(0.01, 0.035,
            f"Seed {seed}; budget {budget}. Both supporting triplets per row were newly injected; targets themselves were not injected.",
            ha="left", va="center", fontsize=8.5)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.97, bottom=0.05)
    try:
        for extension in ("png", "pdf"):
            fig.savefig(paths[extension], dpi=300, bbox_inches="tight")
    finally:
        plt.close(fig)
    print(json.dumps({"target_ids": [item["target_id"] for item in examples],
                      "files": {extension: str(path) for extension, path in paths.items()}}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dir", type=Path, default=DEFAULT_EXPERIMENT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    export(parser.parse_args())


if __name__ == "__main__":
    main()
