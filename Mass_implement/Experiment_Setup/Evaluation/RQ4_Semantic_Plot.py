"""Summarize ConvE logits and plot the UMLS counterpart of MaSS Figure 8.

All three distributions use one shared set of linear histogram edges and count
triplets rather than probability densities. Raw logits retain their measured
scale; the score range in the paper is not imposed on a different trained model.
"""
from __future__ import annotations

import math
from collections.abc import Mapping
from numbers import Integral
from pathlib import Path
from statistics import mean, median, stdev

import numpy as np


GROUP_ORDER = ("Correct facts", "Poisoned triplets", "Noisy facts")
GROUP_COLORS = {"Correct facts": "#83ad70", "Poisoned triplets": "#d4875f",
                "Noisy facts": "#438db4"}


def _group_scores(rows):
    grouped = {group: [] for group in GROUP_ORDER}
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping) or "group" not in row or "score" not in row:
            raise ValueError(f"Score row {index} must contain group and score")
        group = row["group"]
        if not isinstance(group, str) or group not in grouped:
            raise ValueError(f"Score row {index} has an unsupported group: {group!r}")
        value = row["score"]
        if isinstance(value, (bool, np.bool_)):
            raise ValueError(f"Score row {index} score must be finite")
        try:
            score = float(value)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError(f"Score row {index} score must be finite") from error
        if not math.isfinite(score):
            raise ValueError(f"Score row {index} score must be finite")
        grouped[group].append(score)
    missing = [group for group, scores in grouped.items() if not scores]
    if missing:
        raise ValueError(f"Each score group must be nonempty; missing: {missing}")
    return grouped


def _histogram(grouped, bins):
    if isinstance(bins, bool) or not isinstance(bins, Integral) or bins < 1:
        raise ValueError("bins must be a positive integer")
    all_scores = np.asarray([score for scores in grouped.values() for score in scores],
                            dtype=float)
    edges = np.histogram_bin_edges(all_scores, bins=int(bins))
    counts = {group: np.histogram(grouped[group], bins=edges)[0].astype(int).tolist()
              for group in GROUP_ORDER}
    return edges.tolist(), counts


def summarize_scores(rows):
    """Return measured score statistics in correct/poisoned/noisy group order.

    ``std`` is the sample standard deviation (ddof=1), or None for one score.
    Percentiles use linear interpolation between sorted observations.
    """
    grouped = _group_scores(rows)
    summary = []
    for group, scores in grouped.items():
        p05, p95 = np.percentile(scores, [5, 95], method="linear")
        summary.append({
            "group": group, "count": len(scores), "mean": mean(scores),
            "median": median(scores), "std": stdev(scores) if len(scores) > 1 else None,
            "min": min(scores), "max": max(scores),
            "p05": float(p05), "p95": float(p95),
        })
    return summary


def histogram_counts(rows, bins=20):
    """Return shared linear bin edges and per-group counts for Figure 8.

    As in NumPy histograms, bins include their left edge and exclude their right
    edge, except that the last bin includes its right edge. For identical scores,
    NumPy expands the otherwise zero-width range by 0.5 on either side.
    """
    return _histogram(_group_scores(rows), bins)


def _count_density_curves(grouped, edges, points=512):
    """Gaussian KDE scaled to expected count per shared histogram bin.

    Scott's bandwidth is used for nondegenerate samples. A half-bin Gaussian
    is a visual fallback for singleton/constant groups. Gaussian tails outside
    the displayed measured-score range are not renormalized.
    """
    from scipy.stats import gaussian_kde

    width = float(edges[1] - edges[0])
    grid = np.linspace(edges[0], edges[-1], points)
    curves = {}
    for group, scores in grouped.items():
        values = np.asarray(scores, dtype=float)
        deviation = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        minimum_bandwidth = max(width * 1e-6, np.finfo(float).eps * (float(np.max(np.abs(values))) + 1))
        bandwidth = deviation * len(values) ** (-0.2)
        density = None
        if math.isfinite(bandwidth) and bandwidth > minimum_bandwidth:
            try:
                density = gaussian_kde(values, bw_method="scott")(grid)
            except np.linalg.LinAlgError:
                pass
        if density is None:
            bandwidth = max(width / 2, minimum_bandwidth)
            z = (grid - float(np.mean(values))) / bandwidth
            density = np.exp(-0.5 * z ** 2) / (bandwidth * math.sqrt(2 * math.pi))
        curves[group] = density * len(values) * width
    return grid, curves


def plot_semantic_information(rows, output_dir, *, bins=20):
    """Save count histograms with count-scaled KDE curves as PNG/PDF."""
    grouped = _group_scores(rows)
    edges, _ = _histogram(grouped, bins)
    grid, curves = _count_density_curves(grouped, edges)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {extension: output_dir / f"figure8_umls_semantic_information.{extension}"
               for extension in ("png", "pdf")}
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    try:
        for group in GROUP_ORDER:
            ax.hist(grouped[group], bins=edges, color=GROUP_COLORS[group], alpha=0.28,
                    edgecolor="white", linewidth=0.4)
        for group in GROUP_ORDER:
            ax.plot(grid, curves[group], color=GROUP_COLORS[group], linewidth=2.2, label=group)
        ax.set_xlim(edges[0], edges[-1])
        ax.set_xlabel("ConvE prediction score (logit)")
        ax.set_ylabel("Count")
        ax.legend(frameon=False)
        ax.set_axisbelow(True)
        ax.grid(axis="y", alpha=0.2)
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        for path in outputs.values():
            fig.savefig(path, dpi=300, bbox_inches="tight")
    finally:
        plt.close(fig)
    return outputs
