"""Evaluate trained UMLS TransE embeddings and plot previously saved loss history.

Run from Path_Generation_Module:
    python TransE_Surrogate.py
    python Model_Evaluation.py
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from TransE_Surrogate import MASS_DIR, Surrogate, read_triples

DATASET_DIR = MASS_DIR / "dataset_uml"
OUTPUT_DIR = MASS_DIR / "outputs" / "Path_Generation"


def plot_training_loss(csv_path: Path, output_path: Path) -> bool:
    """Render the true per-epoch loss CSV produced during training."""
    if not csv_path.is_file():
        print(f"No loss history at {csv_path}; skipping loss plot. Train with updated TransE_Surrogate.py first.")
        return False
    epochs, losses = [], []
    with csv_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            epochs.append(int(row["epoch"]))
            losses.append(float(row["training_loss"]))
    if not epochs:
        print("Loss CSV is empty; skipping plot.")
        return False
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(epochs, losses, linewidth=1.7)
    ax.set(title="UMLS TransE — training loss", xlabel="Epoch", ylabel="Mean margin-ranking loss")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    print(f"Saved loss plot: {output_path}")
    return True


def encode_triples(triples: list[tuple[str, str, str]],
                   entity_to_id: dict[str, int], relation_to_id: dict[str, int]) -> list[tuple[int, int, int]]:
    """Encode string triples using exactly the mappings stored in the checkpoint."""
    return [(entity_to_id[h], relation_to_id[r], entity_to_id[t]) for h, r, t in triples]


def metric_summary(ranks: list[float]) -> dict[str, float | int]:
    if not ranks:
        raise ValueError("Evaluation split has no triples.")
    a = np.asarray(ranks, dtype=np.float64)
    return {
        "count": len(ranks),
        "MRR": float(np.mean(1 / a)),
        "MR": float(np.mean(a)),
        "Median_Rank": float(np.median(a)),
        "Hits@1": float(np.mean(a <= 1)),
        "Hits@3": float(np.mean(a <= 3)),
        "Hits@10": float(np.mean(a <= 10)),
    }


@torch.no_grad()
def evaluate_filtered(surrogate: Surrogate, split: str = "valid") -> dict:
    """Filtered head/tail entity ranking using all known true train/valid/test triples.

    Uses average rank for tied scores. Filters all other known true answers,
    but retains the target answer. Both head and tail predictions are included.
    """
    e2id, r2id = surrogate.entity_to_id, surrogate.relation_to_id
    data = {name: encode_triples(read_triples(DATASET_DIR / f"{name}.txt"), e2id, r2id)
            for name in ("train", "valid", "test")}
    all_true = set(data["train"] + data["valid"] + data["test"])
    tails_by_hr: dict[tuple[int, int], set[int]] = {}
    heads_by_rt: dict[tuple[int, int], set[int]] = {}
    for h, r, t in all_true:
        tails_by_hr.setdefault((h, r), set()).add(t)
        heads_by_rt.setdefault((r, t), set()).add(h)

    model = surrogate.model.eval()
    n = len(e2id)
    candidates = torch.arange(n, dtype=torch.long)
    head_ranks, tail_ranks = [], []
    for h, r, t in data[split]:
        heads = torch.full((n,), h, dtype=torch.long)
        rels = torch.full((n,), r, dtype=torch.long)
        tails = torch.full((n,), t, dtype=torch.long)

        tail_scores = model.score(heads, rels, candidates)
        target_tail_score = tail_scores[t].item()
        other_true_tails = tails_by_hr.get((h, r), set()) - {t}
        if other_true_tails:
            tail_scores[list(other_true_tails)] = -torch.inf
        better = (tail_scores > target_tail_score).sum().item()
        equal = (tail_scores == target_tail_score).sum().item()
        tail_ranks.append(1 + better + (equal - 1) / 2)

        head_scores = model.score(candidates, rels, tails)
        target_head_score = head_scores[h].item()
        other_true_heads = heads_by_rt.get((r, t), set()) - {h}
        if other_true_heads:
            head_scores[list(other_true_heads)] = -torch.inf
        better = (head_scores > target_head_score).sum().item()
        equal = (head_scores == target_head_score).sum().item()
        head_ranks.append(1 + better + (equal - 1) / 2)

    return {
        "dataset": "UMLS",
        "evaluation_split": split,
        "protocol": "filtered head and tail entity ranking; all true triples from train/valid/test filtered",
        "tie_policy": "average rank",
        "num_entities": n,
        "num_relations": len(r2id),
        "num_evaluation_triples": len(data[split]),
        "overall": metric_summary(head_ranks + tail_ranks),
        "head_prediction": metric_summary(head_ranks),
        "tail_prediction": metric_summary(tail_ranks),
    }


def save_metrics(results: dict, output_dir: Path) -> None:
    """Save model_performace.json only."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "model_performace.json"
    json_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Saved metrics: {json_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate UMLS TransE checkpoint and plot training loss")
    parser.add_argument("--checkpoint", type=Path, default=OUTPUT_DIR / "transe_umls.pt")
    parser.add_argument("--split", choices=("train", "valid", "test"), default="valid")
    parser.add_argument("--loss-csv", type=Path, default=OUTPUT_DIR / "training_loss.csv")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()

    plot_training_loss(args.loss_csv, args.output_dir / "training_loss.png")
    surrogate = Surrogate.load(args.checkpoint)
    print(f"Evaluating {args.split} using checkpoint: {args.checkpoint}")
    metrics = evaluate_filtered(surrogate, split=args.split)
    save_metrics(metrics, args.output_dir)


if __name__ == "__main__":
    main()
