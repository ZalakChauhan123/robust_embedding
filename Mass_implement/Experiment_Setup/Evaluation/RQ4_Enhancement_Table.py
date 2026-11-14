"""Build the UMLS counterpart of MaSS Table 3 from measured ConvE runs.

The paper leaves Table 3's evaluation populations implicit. This local
interpretation uses Set B attack metrics for the type-constraint comparison
and Set A clean utility for the plausibility-constraint comparison, following
the discussion in section 5.5. Both comparisons use the same unenhanced attack
as their baseline; the two enhancements are separate rather than cumulative.
No result or direction of improvement is assumed by this module.
"""
from __future__ import annotations

import csv
import json
import math
from collections.abc import Mapping
from numbers import Integral
from pathlib import Path
from statistics import mean


ATTACKS = ("DirectAdd", "ComAttack")
ENHANCEMENTS = ("none", "type", "plausibility")
METRICS = ("set_B_Hit_at_10", "set_B_MRR", "set_A_Hit_at_10", "set_A_MRR")
TABLE_COLUMNS = (
    "attack", "num_seeds",
    "type_before_Hit_at_10", "type_before_MRR",
    "type_after_Hit_at_10", "type_after_MRR",
    "plausibility_before_Hit_at_10", "plausibility_before_MRR",
    "plausibility_after_Hit_at_10", "plausibility_after_MRR",
)


def _integer(value, name: str, *, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _metric(value, name: str) -> float:
    message = f"{name} must be a finite number in [0, 1]"
    if isinstance(value, bool):
        raise ValueError(message)
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(message) from error
    if not math.isfinite(result) or not 0 <= result <= 1:
        raise ValueError(message)
    return result


def _seeds(seeds) -> list[int]:
    result = [_integer(seed, "seed", minimum=0) for seed in seeds]
    if not result:
        raise ValueError("seeds must be nonempty")
    if len(set(result)) != len(result):
        raise ValueError("seeds must not contain duplicates")
    return result


def build_table3(metrics, seeds):
    """Return two aggregate rows, ordered DirectAdd then ComAttack.

    Require exactly one measured row for each attack, enhancement
    (``none``, ``type``, ``plausibility``), and requested nonnegative seed.
    Each row requires all four ``set_A_*``/``set_B_*`` metrics, representing
    the corresponding poisoned victim's performance. All metrics must be
    finite and in [0, 1]. An optional ``num_targets`` must be a positive
    integer; other additional fields are ignored.

    Type columns use Set B. Plausibility columns use Set A. ``before`` always
    means enhancement ``none``; ``after`` uses only its named enhancement.
    Values are arithmetic means across seeds, without intermediate rounding.
    """
    seeds = _seeds(seeds)
    expected = {(attack, enhancement, seed) for attack in ATTACKS
                for enhancement in ENHANCEMENTS for seed in seeds}
    observed = {}
    required = ("attack", "enhancement", "seed", *METRICS)
    for index, row in enumerate(metrics):
        if not isinstance(row, Mapping) or any(field not in row for field in required):
            raise ValueError(f"Metric row {index} must contain {', '.join(required)}")
        attack, enhancement = row["attack"], row["enhancement"]
        if attack not in ATTACKS or enhancement not in ENHANCEMENTS:
            raise ValueError(f"Unexpected attack/enhancement: {attack!r}, {enhancement!r}")
        seed = _integer(row["seed"], "seed", minimum=0)
        key = (attack, enhancement, seed)
        if key not in expected:
            raise ValueError(f"Unexpected attack/enhancement/seed run: {key}")
        if key in observed:
            raise ValueError(f"Duplicate attack/enhancement/seed run: {key}")
        if "num_targets" in row:
            _integer(row["num_targets"], "num_targets", minimum=1)
        observed[key] = {field: _metric(row[field], field) for field in METRICS}
    missing = expected - observed.keys()
    if missing:
        raise ValueError(f"Incomplete attack/enhancement/seed coverage; missing runs: {sorted(missing)}")

    result = []
    for attack in ATTACKS:
        aggregated = {"attack": attack, "num_seeds": len(seeds)}
        for constraint, population in (("type", "B"), ("plausibility", "A")):
            for period, enhancement in (("before", "none"), ("after", constraint)):
                for metric in ("Hit_at_10", "MRR"):
                    values = [observed[(attack, enhancement, seed)][f"set_{population}_{metric}"]
                              for seed in seeds]
                    aggregated[f"{constraint}_{period}_{metric}"] = mean(values)
        result.append(aggregated)
    return result


def _markdown(rows, seeds) -> str:
    lines = [
        "# UMLS — ConvE victim: enhancement of baseline attacks", "",
        f"Arithmetic means across {len(seeds)} seed(s): {', '.join(map(str, seeds))}.", "",
        "The evaluation populations below are a local interpretation of the discussion in "
        "MaSS section 5.5; the original Table 3 does not explicitly label them. "
        "Before is the unenhanced attack. Each after variant applies only the named constraint.",
        "",
    ]
    for constraint, label in (
        ("type", "Type constraint: Set B attack effectiveness"),
        ("plausibility", "Plausibility constraint: Set A clean utility (stealthiness)"),
    ):
        lines += [f"## {label}", "",
                  "| Attack | Before Hit@10 | Before MRR | After Hit@10 | After MRR |",
                  "|---|---:|---:|---:|---:|"]
        for row in rows:
            values = [row[f"{constraint}_{period}_{metric}"]
                      for period in ("before", "after") for metric in ("Hit_at_10", "MRR")]
            lines.append(f"| {row['attack']} | " + " | ".join(f"{value:.4f}" for value in values) + " |")
        lines.append("")
    lines += [
        "All metrics come from the corresponding poisoned ConvE victim. Higher Set B "
        "metrics indicate stronger target promotion; higher Set A metrics indicate better "
        "preservation of clean prediction performance. An improvement is not assumed.", "",
    ]
    return "\n".join(lines)


def _render(rows, paths):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(13.4, 3.6))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    edges = [0.01, 0.17] + [0.17 + index * 0.1025 for index in range(1, 9)]
    centers = [(left + right) / 2 for left, right in zip(edges, edges[1:])]
    ax.text(0.5, 0.97, "UMLS — ConvE victim", ha="center", va="center", fontsize=15, weight="bold")
    ax.text((edges[1] + edges[5]) / 2, 0.81, "Type constraint: Set B attack",
            ha="center", va="center", fontsize=12)
    ax.text((edges[5] + edges[9]) / 2, 0.81, "Plausibility constraint: Set A clean utility",
            ha="center", va="center", fontsize=12)
    for start in (1, 5):
        for offset, label in ((0, "Before enhanced"), (2, "After enhanced")):
            ax.text((edges[start + offset] + edges[start + offset + 2]) / 2, 0.70,
                    label, ha="center", va="center", fontsize=11)
    ax.text(centers[0], 0.59, "Attack", ha="center", va="center", fontsize=11)
    for index in range(1, 9):
        ax.text(centers[index], 0.59, "Hit@10" if index % 2 else "MRR",
                ha="center", va="center", fontsize=11)
    for y, row in zip((0.44, 0.29), rows):
        ax.text(centers[0], y, row["attack"], ha="center", va="center", fontsize=12)
        for center, field in zip(centers[1:], TABLE_COLUMNS[2:]):
            ax.text(center, y, f"{row[field]:.4f}", ha="center", va="center", fontsize=11)
    for y, left, linewidth in ((0.87, edges[0], 1.1), (0.75, edges[1], 0.6),
                              (0.64, edges[1], 0.6), (0.53, edges[0], 0.8),
                              (0.20, edges[0], 1.1)):
        ax.plot((left, edges[-1]), (y, y), color="black", linewidth=linewidth)
    for index in (1, 3, 5, 7):
        ax.plot((edges[index], edges[index]), (0.20, 0.87 if index in (1, 5) else 0.75),
                color="black", linewidth=0.6)
    ax.text(0.01, 0.11,
            f"Means over {rows[0]['num_seeds']} seed(s). Before = unenhanced; after = the named constraint only.",
            ha="left", va="center", fontsize=9)
    ax.text(0.01, 0.035, "Metric populations are a local interpretation of MaSS section 5.5; all values are measured.",
            ha="left", va="center", fontsize=9)
    fig.subplots_adjust(left=0.025, right=0.975, top=0.96, bottom=0.05)
    try:
        for extension in ("png", "pdf"):
            fig.savefig(paths[extension], dpi=300, bbox_inches="tight")
    finally:
        plt.close(fig)


def export_table3(metrics, output_dir, *, seeds):
    """Validate, aggregate, and export CSV/JSON/Markdown/PNG/PDF Table 3.

    Files are named ``table3_umls.<extension>``. CSV and JSON retain full
    numeric precision; Markdown and rendered table cells show four decimals.
    JSON contains the same two-row array as CSV. Returns
    ``{'rows': aggregate_rows, 'paths': {extension: pathlib.Path, ...}}``.
    Invalid input is rejected before the output directory is created.
    """
    seeds = _seeds(seeds)
    rows = build_table3(metrics, seeds)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {extension: output_dir / f"table3_umls.{extension}"
             for extension in ("csv", "json", "md", "png", "pdf")}
    with paths["csv"].open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=TABLE_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    paths["json"].write_text(json.dumps(rows, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    paths["md"].write_text(_markdown(rows, seeds), encoding="utf-8")
    _render(rows, paths)
    return {"rows": rows, "paths": paths}
