# RQ4: enhancing DirectAdd and ComAttack on UMLS — Table 3

`Evaluate_RQ4_Enhancing_Attacks.py` evaluates whether the type and plausibility
constraints discussed in MaSS section 5.5 improve the existing UMLS attacks.
It measures both target promotion and clean prediction performance after
retraining ConvE, and exports a Table-3-style comparison.

Run from the workspace root, `C:\my-code\Robust Embedding`:

```powershell
.\.venv\Scripts\python.exe Mass_implement/Experiment_Setup/Evaluation/Evaluate_RQ4_Enhancing_Attacks.py
```

The default experiment is
`Mass_implement/outputs/Experiments/UMLS_feasible_rank11_15_seed42`. Results
are written to its `RQ4/enhancing_attacks` directory. Existing completed
results require `--resume` or `--plot-only`; the runner does not silently
overwrite a different experiment.

## Experimental conditions

Each attack has three independent variants:

| Variant | Candidate eligibility | Purpose |
|---|---|---|
| `none` | Original baseline eligibility rules | Before-enhancement control |
| `type` | Head belongs to the relation's domain and tail belongs to its range | Type constraint only |
| `plausibility` | Frozen clean TransE score meets the global clean-training cutoff | Plausibility constraint only |

These are six conditions per seed: DirectAdd and ComAttack, each with the
three variants. The `plausibility` variant does not also require `type`.
Baseline ranking objectives and the distinct-new-triples budget remain the
same. Constraints are applied to candidates before selection, allowing the
ranker to select a later eligible candidate. For ComAttack, both edges of
the complete two-edge body must pass the chosen constraint. If a relation
pair has no eligible body, the search continues to later relation pairs.

The default budget is at most **two distinct new triples per target**.
Stricter eligibility may reduce the number of injections and the number
of targets receiving injections. Every condition is still evaluated on
all **100 frozen Set B targets** from
`setup/set_B_target_triples.tsv`; targets with no injection are retained.
Set A is the original UMLS test set. Both sets receive Hit@10 and MRR
measurements for every condition.

### Type constraint

Relation domains and ranges come from
`Mass_implement/dataset_uml/Semantic_Constraints/SRSTRE2`. The eligibility
rule uses their marginal sets:

```text
head in Domain(relation) AND tail in Range(relation)
```

It does not require that the exact head–tail pair previously occurred in
the ontology. For ComAttack, applying this rule to both path edges also
constrains the intermediate entity to the first relation's range and the
second relation's domain.

### Plausibility constraint

The surrogate is the existing, frozen 64-dimensional clean TransE model:
`Mass_implement/outputs/Path_Generation/transe_umls.pt`. Plausibility is its
negative L2 distance:

```text
score(h, r, t) = -||embedding(h) + embedding(r) - embedding(t)||2
eligible       = score(h, r, t) >= cutoff
```

Higher scores are more plausible. These scores are neither probabilities
nor ConvE logits. The cutoff is the global quantile of scores on all 5,216
clean training triples, with `--plausibility-quantile 0.5` selecting the
median. The default measured cutoff is **-0.6652185320854187**, recorded in
`plausibility_calibration.json`. The per-triple calibration scores are saved
in `plausibility_calibration_scores.csv`.

This numerical cutoff is a declared UMLS design choice: the paper does
not specify a numerical plausibility threshold for enhancing its baselines.
Calibration uses clean training facts only. Validation facts, test facts,
Set B targets, and victim predictions do not determine the cutoff. UMLS's
46 × 135 × 135 score cube is cached once, using approximately 3.35 MB,
so candidate checks use the same frozen scores throughout.

## Victim training and controls

The default uses seed 42 and a ConvE victim with the following settings:

| Setting | Default |
|---|---:|
| Embedding dimension | 64 |
| Embedding height | 8 |
| Convolution filters / kernel size | 32 / 3 |
| Input / feature / hidden dropout | 0.2 / 0.2 / 0.3 |
| Label smoothing | 0.1 |
| Learning rate | 0.001 |
| Batch size | 256 |
| Training epochs | 500 |
| Torch threads | 4 |
| DirectAdd promotion step | 0.1 |

Each condition has matched clean and poisoned ConvE victims initialized
from identical parameters. Within each seed, the runner also verifies that
the initial state, trained clean control, and its Set A and Set B metrics
are identical across all conditions. The frozen targets, clean data,
surrogate, ontology, architecture, and training settings provide the
comparison controls.

For `none`, the runner can reuse the corresponding original attack's
saved ConvE victims only after checking the training configuration and
data/checkpoint hashes. Reuse provenance is recorded. Enhanced conditions
train fresh clean/poisoned pairs. Pass `--fresh-baselines` to train the
unenhanced pairs again too; use a new output directory for this change.

## Meaning of the main table

The two groups use different evaluation populations:

| Main table group | Before | After | Interpretation of a larger metric |
|---|---|---|---|
| Type constraint: Set B attack | `none` poisoned victim on target triples | `type` poisoned victim on the same targets | Stronger target promotion |
| Plausibility constraint: Set A clean utility | `none` poisoned victim on clean test triples | `plausibility` poisoned victim on the same clean test triples | Better preservation of clean utility |

Both before columns use the **unenhanced poisoned attack**, evaluated on
the relevant population. They do not substitute the unattacked clean
victim. Both after variants apply only their own constraint. All reported
table metrics are arithmetic means across the requested seeds; rendered
cells show four decimals.

This population mapping is an explicit local interpretation of the
paper's discussion: it describes type constraints in terms of attack
performance and plausibility constraints in terms of stealthiness. The
original Table 3 does not explicitly label its dataset, victim, or metric
populations. Its type-before values match the FB15k-237 TransE attack
values in Table 1, while its ComAttack plausibility-before values match
clean utility in Table 2. DirectAdd's plausibility-before values
(`0.42 / 0.29`) instead match the unattacked clean model, differing from
its Table 2 DirectAdd row (`0.42 / 0.25`). This discrepancy remains
unresolved; the UMLS table uses a consistent poisoned-baseline definition.

To expose both sides of any tradeoff, `enhancement_metrics.csv` and `.json`
contain **both Set A and Set B metrics for all six conditions per seed**,
plus the paired clean-victim controls. Injection counts, target coverage,
constraint pass rates, mean surrogate scores, and provenance accompany
those metrics. The per-condition constraint audit records every injected
triple's type eligibility, surrogate score, cutoff, and plausibility
eligibility. The runner checks the active constraint and verifies that
the aggregate metrics agree with the saved per-triple ranks.

## Resuming, replotting, and changing settings

```powershell
# Continue this experiment with its original settings and verified stages.
.\.venv\Scripts\python.exe Mass_implement/Experiment_Setup/Evaluation/Evaluate_RQ4_Enhancing_Attacks.py --resume

# Re-export the completed measured table without training victims again.
.\.venv\Scripts\python.exe Mass_implement/Experiment_Setup/Evaluation/Evaluate_RQ4_Enhancing_Attacks.py --plot-only

# Run additional seeds against the same frozen targets in a new directory.
.\.venv\Scripts\python.exe Mass_implement/Experiment_Setup/Evaluation/Evaluate_RQ4_Enhancing_Attacks.py --seeds 42 43 44 --output-dir Mass_implement/outputs/Experiments/UMLS_feasible_rank11_15_seed42/RQ4/enhancing_attacks_multiseed

# Evaluate a stricter declared cutoff as a separate experiment.
.\.venv\Scripts\python.exe Mass_implement/Experiment_Setup/Evaluation/Evaluate_RQ4_Enhancing_Attacks.py --plausibility-quantile 0.75 --output-dir Mass_implement/outputs/Experiments/UMLS_feasible_rank11_15_seed42/RQ4/enhancing_attacks_q075
```

`--resume` requires the same settings and source hashes. Changes require a
new output directory. `--plot-only` checks the saved metric artifact hashes
and uses the completed manifest's seeds; it cannot turn partial runs into
a complete table. When using a custom output directory, pass the same
`--output-dir` to resume or replot it.

Available options are `--experiment-dir`, `--output-dir`, `--checkpoint`,
`--ontology`, `--seeds`, `--budget`, `--promotion-step`,
`--plausibility-quantile`, `--victim-epochs`, `--victim-dim`,
`--torch-threads`, `--fresh-baselines`, `--resume`, and `--plot-only`.
`--resume` and `--plot-only` are mutually exclusive. Budgets must be even
and at least 2 because ComAttack injects complete two-edge bodies. ConvE
dimensions must be divisible by 8 and at least 24. A higher plausibility
quantile raises the score cutoff and can reduce candidate coverage; the
runner does not relax it silently if no poisoning remains eligible.

## Saved outputs

Under `<experiment>/RQ4/enhancing_attacks`:

- `table3_umls.csv` and `.json`: two aggregate rows, one per attack, with
  the main table's before/after metrics and seed counts.
- `table3_umls.md`, `.png`, and `.pdf`: the labeled Table-3-style comparison.
- `enhancement_metrics.csv` and `.json`: all measured condition/seed rows,
  including Set A, Set B, coverage, constraints, and clean controls. A
  completed default single-seed experiment has six rows.
- `enhancement_config.json`: frozen protocol, source hashes, calibration,
  victim settings, and environment.
- `enhancement_manifest.json`: completed experiment provenance, control
  checks, baseline reference checks, limitations, and metric artifact hashes.
- `plausibility_calibration.json` and `plausibility_calibration_scores.csv`:
  the cutoff and the clean training scores that determined it.

Each `<attack>/<enhancement>/seed_<seed>/` contains:

- `poisoning/`: injected triples, poisoned training data, selected and
  eligible candidate paths, and poisoning summaries.
- `conve_victim_models/`: matched clean/poisoned checkpoints, training
  losses, and training provenance.
- `conve_evaluation/`: Set A and Set B rank comparisons and evaluation
  summaries.
- `constraint_audit.csv` and `.json`: per-injection checks for both
  constraints, regardless of which constraint was active.
- `run.log` and `stage_manifest.json`: stage logs and completion records.

## Interpreting the result

Compare each enhanced condition with its own unenhanced attack on both
populations. Report the measured Hit@10 and MRR changes together with
target coverage and injection counts. Improved clean utility can come
with weaker target promotion; satisfying a plausibility or type filter
alone does not establish a better attack. Fewer injected triples can
also contribute to the observed utility change.

The paper's qualitative finding is that type constraints can harm
DirectAdd's target promotion while plausibility constraints can improve
clean utility. The UMLS experiment tests that observation without forcing
the same direction or numerical values. The default single seed is a
preliminary result; use multiple seeds to assess its stability.

This is a UMLS adaptation using the project's existing target-promotion
versions of DirectAdd and ComAttack, rather than an exact reproduction of
the original attack implementations, datasets, or Table 3 numbers.
