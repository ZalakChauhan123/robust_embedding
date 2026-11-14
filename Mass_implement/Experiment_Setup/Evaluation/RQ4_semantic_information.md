# RQ4: UMLS semantic information and Figure 8

Run from `Mass_implement`:

```powershell
python Experiment_Setup/Evaluation/Evaluate_RQ4_Semantic_Information.py
```

This compares the ConvE prediction scores of **correct facts**, **MaSS
poisoned triplets**, and **synthetic noisy facts**, following the semantic
information analysis in Section 5.5 of the MaSS paper. It produces the UMLS
version of Figure 8 using existing checkpoints; no model training is needed.

## Default experiment

The experiment is
`outputs/Experiments/UMLS_feasible_rank11_15_seed42`, with attack files in
`baseline/poisoning` and victims in
`baseline/seed_42/conve_victim_models`. These are the original baseline's
**162 unique MaSS injections** and their matching poisoned ConvE checkpoint.
They are separate from the newly trained embedding-sensitivity sweep.

The runner checks checkpoint and training-data provenance before scoring.
An alternative attack directory must use victims trained on that directory's
matching poisoned data. Existing attack data and checkpoints are unchanged.

## Three comparison groups

The default groups contain the same number of triples:

- **Poisoned triplets:** all 162 unique baseline injections.
- **Correct facts:** 162 original test facts, sampled without replacement
  using seed 42. These held-out facts were not in the poisoned training set.
- **Noisy facts:** one corruption of each sampled correct fact, using seed
  43. The relation is retained and exactly one entity, head or tail, changes.

Noise generation selects the corruption direction uniformly when both
directions have legal replacements, then selects a legal replacement
uniformly. If one direction has no legal replacement, it uses the other;
it fails if neither direction is possible. Noise is unique, has no
self-loops, and excludes every original train, validation, and test fact,
every injected triple, and every frozen Set B target. The default noise is
random entity corruption without an additional type-constraint requirement.

`--correct-split valid` or `--correct-split train` changes the source of
correct facts. The training option is an explicitly different comparison
because those correct facts were seen during training. `--num-facts N`
selects a smaller equal-sized comparison; N cannot exceed either the
available injected triples or the selected correct split. Omitting it uses
all injections and requires enough correct facts for an equal-sized sample.

## Scores and plot

All 486 default triples are scored by the **same poisoned ConvE model**.
The score is the raw logit for tail t in `score_all_tails(h, r)`: higher
values indicate greater model plausibility. These are neither TransE
distances nor calibrated probabilities. Scores are not shifted, negated,
or forced to match the paper's negative prediction-score range.

The paired clean ConvE scores the same triples as a saved control. Its
scores are reported separately; the main Figure 8 uses poisoned-model
scores throughout.

The figure overlays three Count histograms with 20 shared bins and smooth
Gaussian KDE curves: green for correct facts, orange for poisoned triplets,
and blue for noisy facts. Curves use Scott's bandwidth and are multiplied by
the group size and shared bin width so their vertical scale matches Count.
They are descriptive smoothing estimates, not additional observations.
Singleton or near-constant scores use a half-bin-width Gaussian fallback.
Gaussian tails outside the displayed score range are not renormalized.
Equal group sizes make counts comparable. The horizontal axis follows the
observed score range. Default inference uses batches of 256 and four Torch
threads; models run in evaluation mode with gradients disabled.

## Options and rerunning the figure

```powershell
# A smaller comparison in a separate output directory.
python Experiment_Setup/Evaluation/Evaluate_RQ4_Semantic_Information.py --num-facts 100 --output-dir outputs/Experiments/UMLS_feasible_rank11_15_seed42/RQ4/semantic_information_100

# Replot the saved scores without scoring the models again.
python Experiment_Setup/Evaluation/Evaluate_RQ4_Semantic_Information.py --plot-only

# Replot with a different shared-bin count.
python Experiment_Setup/Evaluation/Evaluate_RQ4_Semantic_Information.py --plot-only --bins 25
```

Available options include `--experiment-dir`, `--method-dir`, `--victim-dir`,
`--victim-seed` (42), `--output-dir`, `--correct-split` (`test`),
`--num-facts`, `--seed` (42), `--bins` (20), `--batch-size` (256),
`--torch-threads` (4), and `--plot-only`. When using a custom output
directory, pass the same `--output-dir` to replot its saved results.

## Outputs

The default destination is `<experiment>/RQ4/semantic_information`:

- `semantic_information_scores.csv` and `.json`: sampled triples and their
  poisoned-model and paired clean-model scores.
- `semantic_information_summary.csv` and `.json`: group statistics for
  poisoned ConvE, the model used in Figure 8.
- `semantic_information_clean_summary.csv` and `.json`: corresponding
  clean-model statistics for the same triples.
- `histogram_counts.csv` and `.json`: shared bin boundaries and group counts.
- `relation_counts.csv`: relation composition of each comparison group.
- `figure8_umls_semantic_information.png` and `.pdf`: the Figure 8 visual.
- `semantic_information_manifest.json`: inputs, provenance, hashes, sampling
  and scoring settings, and output records.
- `figure8_plot_config.json`: settings used to draw the saved figure.

## Interpretation and limits

Poisoned scores closer to correct facts and above random noise support the
paper's qualitative plausibility observation. Report the measured means,
medians, and distributions; overlap or a different ordering must remain
visible rather than being adjusted to reproduce the paper's appearance.

The paper specifies ConvE trained on the poisoned dataset, but does not
fully specify its Figure 8 sample sizes or noise-generation procedure.
The choices above define a reproducible local UMLS protocol.

Unobserved corruptions are synthetic noise, not proven false facts in an
open-world knowledge graph. Injected triples were seen during training,
whereas default correct facts were held out, so high poisoned scores are
not independent proof of truth or semantic stealthiness. The groups also
have different pooled relation distributions; exact relation matching with
distinct test facts is impossible for this baseline because some injected
relations have too few or no test examples. `relation_counts.csv` exposes
that limitation. This is an empirical comparison of model scores, not a
claim that scores are truth labels or calibrated probabilities.
