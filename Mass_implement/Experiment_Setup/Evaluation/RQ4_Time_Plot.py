"""Aggregate measured attack timings and draw the UMLS counterpart of Figure 6.

Timing denominators are attempted target triplets, including targets receiving
no injections. Only measured elapsed nanoseconds are converted into chart data.
"""
from __future__ import annotations

import math
from numbers import Integral
from pathlib import Path
from statistics import mean, stdev


def _integer(value, name, minimum=0):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _nonnegative(value, name):
    if isinstance(value, bool):
        raise ValueError(f"{name} must be finite and >= 0")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} must be finite and >= 0") from error
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"{name} must be finite and >= 0")
    return result


def aggregate_timing_runs(runs, budgets, methods, repeats):
    """Require one measured pass per budget/method/repeat and summarize ms/target.

    Repeat indices are zero based. Standard deviation is the sample standard
    deviation of pass means, not dispersion across individual target timings.
    Deterministic attack coverage and injection counts must match across repeats.
    """
    budgets = [_integer(value, "budget", 1) for value in budgets]
    methods = list(methods)
    repeats = _integer(repeats, "repeats", 1)
    if not budgets or not methods:
        raise ValueError("budgets and methods must both be nonempty")
    if any(not isinstance(method, str) or not method.strip() for method in methods):
        raise ValueError("methods must be nonempty strings")
    if len(set(budgets)) != len(budgets) or len(set(methods)) != len(methods):
        raise ValueError("budgets and methods must not contain duplicates")
    expected = {(budget, method, repeat) for budget in budgets for method in methods
                for repeat in range(repeats)}
    required = ("budget", "method", "repeat", "num_targets", "total_elapsed_ns",
                "targets_with_injections", "unique_injected")
    observed = {}
    target_counts = set()
    for index, run in enumerate(runs):
        if any(key not in run for key in required):
            raise ValueError(f"Timing row {index} must contain {', '.join(required)}")
        budget = _integer(run["budget"], "budget", 1)
        repeat = _integer(run["repeat"], "repeat")
        method = run["method"]
        if not isinstance(method, str):
            raise ValueError("method must be a string")
        key = (budget, method, repeat)
        if key not in expected:
            raise ValueError(f"Unexpected budget/method/repeat: {key}")
        if key in observed:
            raise ValueError(f"Duplicate budget/method/repeat: {key}")
        count = _integer(run["num_targets"], "num_targets", 1)
        injected_targets = _integer(run["targets_with_injections"], "targets_with_injections")
        unique_injected = _integer(run["unique_injected"], "unique_injected")
        if injected_targets > count:
            raise ValueError("targets_with_injections cannot exceed num_targets")
        if (injected_targets == 0) != (unique_injected == 0):
            raise ValueError("Injection and target coverage counts are inconsistent")
        elapsed = _nonnegative(run["total_elapsed_ns"], "total_elapsed_ns")
        observed[key] = {
            "num_targets": count,
            "ms_per_target": elapsed / 1_000_000 / count,
            "targets_with_injections": injected_targets,
            "unique_injected": unique_injected,
        }
        target_counts.add(count)
    missing = expected - observed.keys()
    if missing:
        raise ValueError(f"Incomplete timing coverage; missing passes: {sorted(missing)}")
    if len(target_counts) != 1:
        raise ValueError("All timing passes must use the same num_targets")
    summaries = []
    for budget in sorted(budgets):
        for method in methods:
            passes = [observed[(budget, method, repeat)] for repeat in range(repeats)]
            reference = passes[0]
            for result in passes[1:]:
                for key in ("targets_with_injections", "unique_injected"):
                    if result[key] != reference[key]:
                        raise ValueError(f"Deterministic {key} changed across repeats: {budget}, {method}")
            timings = [result["ms_per_target"] for result in passes]
            summaries.append({
                "method": method, "budget": budget, "num_repeats": repeats,
                "num_targets": reference["num_targets"],
                "mean_ms_per_target": mean(timings),
                "std_ms_per_target": stdev(timings) if repeats > 1 else None,
                "min_ms_per_target": min(timings), "max_ms_per_target": max(timings),
                "targets_with_injections": reference["targets_with_injections"],
                "unique_injected": reference["unique_injected"],
            })
    return summaries


def plot_time_efficiency(summary, output_dir):
    """Save a Figure-6-style horizontal grouped chart for one UMLS dataset.

    Error bars are sample standard deviations across repeated pass means.
    Numeric labels display the measured mean milliseconds per attempted target.
    Returns paths for the PNG and PDF exports.
    """
    aliases = {"DirectAdd": "DirectAdd", "ComAttack": "ComAttack", "MaSS": "MaSS", "baseline": "MaSS"}
    order = ("DirectAdd", "ComAttack", "MaSS")
    colors = {"DirectAdd": "#438db4", "ComAttack": "#d4875f", "MaSS": "#83ad70"}
    rows = list(summary)
    if not rows:
        raise ValueError("At least one timing summary is required")
    lookup = {}
    repeat_counts, target_counts = set(), set()
    for row in rows:
        method = aliases.get(row["method"])
        if method is None:
            raise ValueError(f"Unsupported timing method: {row['method']}")
        budget = _integer(row["budget"], "budget", 1)
        key = (budget, method)
        if key in lookup:
            raise ValueError(f"Duplicate timing summary: {key}")
        repeats = _integer(row["num_repeats"], "num_repeats", 1)
        count = _integer(row["num_targets"], "num_targets", 1)
        average = _nonnegative(row["mean_ms_per_target"], "mean_ms_per_target")
        deviation = row["std_ms_per_target"]
        if repeats == 1:
            if deviation is not None:
                raise ValueError("Single-pass standard deviation must be None")
        else:
            deviation = _nonnegative(deviation, "std_ms_per_target")
        lookup[key] = {"mean": average, "std": deviation}
        repeat_counts.add(repeats)
        target_counts.add(count)
    if len(repeat_counts) != 1 or len(target_counts) != 1:
        raise ValueError("All plotted methods must have matching repeat and target counts")
    budgets = sorted({budget for budget, _ in lookup})
    missing = {(budget, method) for budget in budgets for method in order} - lookup.keys()
    if missing:
        raise ValueError(f"Incomplete plotted timing coverage: {sorted(missing)}")
    repeats = repeat_counts.pop()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {extension: output_dir / f"figure6_umls_time_efficiency.{extension}"
               for extension in ("png", "pdf")}
    fig, ax = plt.subplots(figsize=(8.4, max(3.5, 0.72 * len(budgets) + 1.5)))
    width = 0.24
    largest = 0.0
    for index, method in enumerate(order):
        values = [lookup[(budget, method)]["mean"] for budget in budgets]
        deviations = [lookup[(budget, method)]["std"] for budget in budgets] if repeats > 1 else None
        largest = max(largest, max(value + (deviations[i] if deviations else 0)
                                   for i, value in enumerate(values)))
        bars = ax.barh([i + (index - 1) * width for i in range(len(budgets))], values,
                       height=width, xerr=deviations, capsize=2.5 if repeats > 1 else 0,
                       color=colors[method], label=method, edgecolor="white", linewidth=0.5,
                       error_kw={"ecolor": "#444444", "linewidth": 0.8})
        ax.bar_label(bars, labels=[f"{value:.2f}" for value in values], padding=4, fontsize=8)
    ax.set_yticks(range(len(budgets)), [str(budget) for budget in budgets])
    ax.invert_yaxis()
    ax.set_xlim(0, largest * 1.22 if largest else 1)
    ax.set_xlabel("Time per attacked target triplet (ms)")
    ax.set_ylabel("Injection budget per target (new triplets)")
    ax.set_title("UMLS", pad=35)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncols=3, frameon=False)
    ax.set_axisbelow(True)
    ax.grid(axis="x", alpha=0.2)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    try:
        for path in outputs.values():
            fig.savefig(path, dpi=300, bbox_inches="tight")
    finally:
        plt.close(fig)
    return outputs
