# RQ4: UMLS attack-generation time efficiency

Run from `Mass_implement`:

```powershell
python Experiment_Setup/Evaluation/Evaluate_RQ4_Time_Efficiency.py
```

The runner compares **DirectAdd, ComAttack, and MaSS** at injection budgets
`2 4 6 8 10 12`, using the 100 frozen targets in
`outputs/Experiments/UMLS_feasible_rank11_15_seed42`. The default surrogate is
the existing 64-dimensional checkpoint at
`outputs/Path_Generation/transe_umls.pt`. It performs three timing repeats
with attack seed 42, four Torch threads, and one BLAS thread. No model training
is needed.

## What is timed

The measured unit is one **attempted target triplet**. Its time includes
candidate generation, scoring and ranking, PRA inference and semantic
processing for MaSS, leakage filtering, sampling, budget allocation, and the
global injected-set update.

Surrogate/PRA pretraining, checkpoint and dataset loading, resource setup,
file writing, printing, victim training, and victim evaluation are excluded.
Models, ontology, dictionaries, and clean-data indexes are loaded before
timing. Candidate paths are regenerated for every target, budget, and repeat;
previously saved feasibility or candidate results are not used as shortcuts.

Each run processes all 100 targets in their fixed order. Its reported time is:

```text
milliseconds per attempted target = sum(target generation times in ns)
                                   / 100 / 1,000,000
```

Targets receiving no injections remain in the denominator. Actual injection
counts and target coverage are reported alongside timings. The budget limits
new triples attributed to a target; it is not the timing denominator.

The first three targets are used to warm up each method before measurements.
The method order rotates and budget order alternates between repeats to
reduce order effects. Global injection state resets for each method/budget
run. All methods use the same loaded data and surrogate.

## Options

```powershell
# More timing repeats.
python Experiment_Setup/Evaluation/Evaluate_RQ4_Time_Efficiency.py --repeats 5 --output-dir outputs/Experiments/UMLS_feasible_rank11_15_seed42/RQ4/time_efficiency_5repeats

# Rebuild tables and the figure from saved timing samples.
python Experiment_Setup/Evaluation/Evaluate_RQ4_Time_Efficiency.py --plot-only
```

Other options include `--budgets`, `--seed`, `--checkpoint`, `--torch-threads`,
`--blas-threads`, `--warmup-targets`, `--output-dir`, and `--experiment-dir`.
`--limit` is for short verification runs and requires an explicit
`--output-dir` so that smoke results do not replace the full benchmark.

## Outputs

The default destination is `<experiment>/RQ4/time_efficiency`:

- `timing_samples.csv` and `.json`: individual per-target measurements.
- `timing_runs.csv` and `.json`: totals and means for each complete pass.
- `timing_summary.csv` and `.json`: means and sample standard deviations of
  repeat means for each attack/budget combination.
- `figure6_umls_time_efficiency.png` and `.pdf`: grouped horizontal bars by
  budget, with value labels and repeat-variation error bars.
- `timing_manifest.json`: inputs, code hashes, hardware, thread settings,
  measurement protocol, coverage, and output-hash checks.

The benchmark requires repeated runs of the same attack configuration to
produce identical injected sets. When the checkpoint and configuration match
the saved budget-sweep manifest, it checks all 18 attack/budget combinations
against those reference injection sets; otherwise it explicitly records that
the reference check was skipped.

## Interpretation

Lower milliseconds per attempted target indicate faster generation under
this recorded protocol. Error bars describe variation between repeated
timing passes, not uncertainty in attack success. Compare injection counts
and target coverage before interpreting speed differences.

These are measurements of the current local UMLS implementations. DirectAdd
and ComAttack are this project's target-promotion adaptations. Their runtime
depends on the current Python/Torch implementation, candidate spaces,
hardware, and thread settings. The paper does not explicitly prescribe every
timing boundary, so this protocol states its choices rather than claiming an
identical hardware benchmark. The observed attack ordering must come from
the measured results; it is not forced to match the paper's ordering.
