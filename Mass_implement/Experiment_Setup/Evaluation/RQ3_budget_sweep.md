# RQ3 type-constraint budget sweep

Run from `Mass_implement`:

```powershell
python Experiment_Setup/Evaluation/Evaluate_RQ3_Budget_Sweep.py
```

The runner generates and evaluates MaSS, DirectAdd, and ComAttack at budgets
`2 4 6 8 10 12`, using the existing `UMLS_feasible_rank11_15_seed42` experiment.
The budget is the maximum number of distinct new triples attributed to each
target. This detection experiment does not require victim retraining.

The chart follows Figure 5: injection budget on the horizontal axis, remaining
ratio from 0.0 to 1.0 on the vertical axis, and three attack bars per budget.
It applies **type constraints only**; mutual-exclusion detection is a separate
RQ3 measurement.

## Options

```powershell
python Experiment_Setup/Evaluation/Evaluate_RQ3_Budget_Sweep.py --budgets 4 6 8 10 12
```

- `--experiment-dir`: source experiment containing the targets and saved MaSS
  feasibility candidates.
- `--output-dir`: destination; defaults to `<experiment>/RQ3/budget_sweep`.
- `--budgets`: requested budgets; the default is `2 4 6 8 10 12`.
- `--seed`: random seed; default `42`.
- `--delta`: MaSS sampling parameter; default `0.7`.
- `--checkpoint`: TransE surrogate checkpoint.
- `--promotion-step`: DirectAdd promotion step; default `0.1`.
- `--evaluate-only`: reuse the generated budget directories and refresh their
  evaluation summaries and chart.

Use even budgets for ComAttack, which selects two-edge composition bodies.
Keep targets, surrogate checkpoint, ontology, seed, delta, and promotion step
fixed when comparing budgets.

## Outputs

Each `x_2`, `x_4`, etc. directory contains attack datasets under
`baseline/poisoning`, `DirectAdd/poisoning`, and `ComAttack/poisoning`, together
with RQ3 summaries, per-target audit information, and `generation.log`.
`baseline` is the MaSS method.

The output root also contains:

- `type_constraint_budget_summary.csv` and `.json`: aggregate counts and ratios.
- `figure5_umls_type_constraints.png` and `.pdf`: the grouped bar chart.
- `sweep_manifest.json`: configuration and provenance for the sweep.

Original experiment outputs are preserved. When the original configuration
matches, the runner checks the regenerated `x=2` injected sets against the
original saved sets as a regression check.

## Interpretation

For each attack and budget:

```text
remaining ratio = (unique injected triples - type-detected triples)
                  / unique injected triples
```

The denominator is the actual number of unique newly injected triples, not
the number of targets multiplied by the requested budget. MaSS can underfill
because paths are sampled and must be selected whole; shared injected triples
are counted once in the final graph. Report the injected counts alongside the
ratios. Chart generation requires zero unknown type-check outcomes.

This extends the existing UMLS implementation. MaSS reuses its cached candidate
pool. ComAttack selects up to `x/2` composition bodies from the first viable
relation pair, ranked by body plausibility. DirectAdd and ComAttack remain the
project's target-promotion adaptations. These UMLS results are a comparison of
the current implementations, not an exact reproduction of the paper's original
datasets or source attack algorithms. Passing the type filter establishes type
compatibility; it does not establish that a fact is true.
