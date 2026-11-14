"""Aggregate measured RQ4 runs and draw the UMLS counterpart of Figure 9.

The surrogate dimensions are categorical positions. MRR is shown with bars on
the left axis and poisoned-victim Hit@10 with an x-marker line on the right.
No metric values or dimensional trend are inferred by this module.
"""
from __future__ import annotations

import math
from numbers import Integral
from pathlib import Path
from statistics import mean, stdev


def _integer(value, name: str, *, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _metric(value, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite number in [0, 1]")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} must be a finite number in [0, 1]") from error
    if not math.isfinite(result) or not 0 <= result <= 1:
        raise ValueError(f"{name} must be a finite number in [0, 1]")
    return result


def aggregate_runs(rows, dimensions, seeds):
    """Require exactly one measured run for every requested dimension/seed.

    Rows require ``surrogate_dim``, ``seed``, ``Hit_at_10``, and ``MRR``.
    Additional fields are ignored. Standard deviations are sample standard
    deviations (ddof=1), or None when a dimension has only one seed.
    """
    dimensions = [_integer(value, "dimension", minimum=1) for value in dimensions]
    seeds = [_integer(value, "seed", minimum=0) for value in seeds]
    if not dimensions or not seeds:
        raise ValueError("dimensions and seeds must both be nonempty")
    if len(dimensions) != len(set(dimensions)) or len(seeds) != len(set(seeds)):
        raise ValueError("dimensions and seeds must not contain duplicates")
    expected = {(dimension, seed) for dimension in dimensions for seed in seeds}
    observed = {}
    required = ("surrogate_dim", "seed", "Hit_at_10", "MRR")
    for index, row in enumerate(rows):
        if any(key not in row for key in required):
            raise ValueError(f"Run row {index} must contain {', '.join(required)}")
        dimension = _integer(row["surrogate_dim"], "surrogate_dim", minimum=1)
        seed = _integer(row["seed"], "seed", minimum=0)
        key = (dimension, seed)
        if key not in expected:
            raise ValueError(f"Unexpected dimension/seed run: {key}")
        if key in observed:
            raise ValueError(f"Duplicate dimension/seed run: {key}")
        observed[key] = {
            "Hit_at_10": _metric(row["Hit_at_10"], "Hit_at_10"),
            "MRR": _metric(row["MRR"], "MRR"),
        }
    missing = expected - observed.keys()
    if missing:
        raise ValueError(f"Incomplete dimension/seed coverage; missing runs: {sorted(missing)}")
    aggregated = []
    for dimension in sorted(dimensions):
        hits = [observed[(dimension, seed)]["Hit_at_10"] for seed in seeds]
        mrr = [observed[(dimension, seed)]["MRR"] for seed in seeds]
        aggregated.append({
            "surrogate_dim": dimension,
            "num_seeds": len(seeds),
            "Hit_at_10_mean": mean(hits),
            "Hit_at_10_std": stdev(hits) if len(seeds) > 1 else None,
            "MRR_mean": mean(mrr),
            "MRR_std": stdev(mrr) if len(seeds) > 1 else None,
        })
    return aggregated


def plot_embedding_sensitivity(aggregated, output_dir, *, victim_dim):
    """Save Figure-9-style PNG/PDF charts from completed aggregate results.

    MRR spans [0,0.20] and Hit@10 spans [0.20,0.70]. Out-of-range means are
    annotated at the axis boundary. Multi-seed runs show sample-standard-deviation
    error bars; a single seed shows no error bars. Values are labelled, with
    a metric legend and axis labels only.
    Returns a dictionary mapping ``png`` and ``pdf`` to their output Paths.
    """
    victim_dim = _integer(victim_dim, "victim_dim", minimum=1)
    rows = list(aggregated)
    if not rows:
        raise ValueError("At least one aggregated dimension is required")
    dimensions = [_integer(row["surrogate_dim"], "surrogate_dim", minimum=1) for row in rows]
    if len(set(dimensions)) != len(dimensions):
        raise ValueError("Aggregated dimensions must not contain duplicates")
    seed_counts = {_integer(row["num_seeds"], "num_seeds", minimum=1) for row in rows}
    if len(seed_counts) != 1:
        raise ValueError("Every dimension must have the same seed count")
    num_seeds = seed_counts.pop()
    rows.sort(key=lambda row: row["surrogate_dim"])
    hits = [_metric(row["Hit_at_10_mean"], "Hit_at_10_mean") for row in rows]
    mrr = [_metric(row["MRR_mean"], "MRR_mean") for row in rows]
    deviations = {}
    for key in ("MRR_std", "Hit_at_10_std"):
        if num_seeds == 1:
            if any(row[key] is not None for row in rows):
                raise ValueError("Single-seed standard deviations must be None")
            deviations[key] = None
        else:
            deviations[key] = [_metric(row[key], key) for row in rows]

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FormatStrFormatter

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {extension: output_dir / f"figure9_umls_embedding_sensitivity.{extension}"
               for extension in ("png", "pdf")}
    fig, left = plt.subplots(figsize=(7.6, 4.9))
    right = left.twinx()
    positions = list(range(len(rows)))
    bars = left.bar(positions, mrr, width=0.55, color="#83ad70", edgecolor="white",
                    yerr=deviations["MRR_std"], capsize=4 if num_seeds > 1 else 0,
                    error_kw={"ecolor": "#315c2a", "linewidth": 1.25}, label="MRR (bar)")
    left.bar_label(bars, labels=[f"{value:.4f}" for value in mrr], padding=5,
                   fontsize=10, color="#315c2a")
    if num_seeds > 1:
        line = right.errorbar(positions, hits, yerr=deviations["Hit_at_10_std"],
                             marker="x", markersize=8, linewidth=2, color="#154fca",
                             capsize=4, label="Hit@10 (line)")
    else:
        line, = right.plot(positions, hits, marker="x", markersize=8,
                           linewidth=2, color="#154fca", label="Hit@10 (line)")
    left.set_ylim(0.0, 0.20)
    left.set_yticks([0.0, 0.05, 0.10, 0.15, 0.20])
    right.set_ylim(0.20, 0.70)
    right.set_yticks([0.20, 0.30, 0.40, 0.50, 0.60, 0.70])
    for axis in (left, right):
        axis.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
        axis.spines["top"].set_visible(False)
    for axis, values in ((left, mrr), (right, hits)):
        lower, upper = axis.get_ylim()
        for position, value in zip(positions, values):
            if value < lower or value > upper:
                below = value < lower
                axis.annotate(
                    f"{value:.2f} ({'below' if below else 'above'} range)",
                    xy=(position, lower if below else upper),
                    xytext=(0, 14 if below else -14), textcoords="offset points",
                    ha="center", va="bottom" if below else "top",
                    fontsize=9, color="#154fca" if axis is right else "#315c2a",
                    bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.9},
                    arrowprops={"arrowstyle": "->", "color": "#154fca" if axis is right else "#315c2a"},
                )
    for position, value in zip(positions, hits):
        if right.get_ylim()[0] <= value <= right.get_ylim()[1]:
            right.annotate(f"{value:.2f}", xy=(position, value), xytext=(0, 10),
                           textcoords="offset points", ha="center", va="bottom",
                           fontsize=10, color="#154fca",
                           bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85, "pad": 1.5})
    left.set_ylabel("MRR (bar)", color="#315c2a")
    right.set_ylabel("Hit@10 (line)", color="#154fca")
    left.tick_params(axis="y", colors="#315c2a")
    right.tick_params(axis="y", colors="#154fca")
    left.set_xticks(positions, [str(row["surrogate_dim"]) for row in rows])
    left.set_xlabel("Surrogate TransE embedding size")
    left.set_axisbelow(True)
    left.grid(axis="y", alpha=0.2)
    fig.legend([bars, line], ["MRR (bar)", "Hit@10 (line)"],
               loc="upper center", bbox_to_anchor=(0.5, 0.99), ncols=2, frameon=False)
    fig.subplots_adjust(left=0.11, right=0.89, bottom=0.14, top=0.88)
    try:
        for extension, path in outputs.items():
            fig.savefig(path, dpi=300, bbox_inches="tight")
    finally:
        plt.close(fig)
    return outputs
