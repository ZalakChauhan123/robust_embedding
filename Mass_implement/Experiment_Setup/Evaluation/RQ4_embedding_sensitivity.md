# RQ4: surrogate embedding-size sensitivity on UMLS

Run from `Mass_implement`:

```powershell
python Experiment_Setup/Evaluation/Evaluate_RQ4_Embedding_Sensitivity.py
```

The default experiment varies the **TransE attacker surrogate dimension** over
`32 64 128`, using seed `42`. ConvE is the victim and keeps its dimension fixed
at 64. The chart follows Figure 9: **Hit@10 is a line and MRR is a bar series**.
Both metrics are measured on the poisoned ConvE model over all 100 frozen
Set B attack targets. Higher values favor the attacker.

Each dimension follows the complete pipeline:

1. Train a fresh TransE surrogate on clean UMLS training data.
2. Regenerate PRA/semantic candidate paths and TransE rankings using that
   dimension's checkpoint. Previously cached rankings are not reused.
3. Sample paths and construct a leakage-controlled poisoned graph.
4. Train matched clean and poisoned ConvE victims from identical initial
   states, keeping the victim configuration fixed across dimensions.
5. Evaluate filtered tail ranks on the same Set B targets and original Set A
   test triples. Set A and clean-model metrics are saved as controls.

Targets remain fixed across dimensions, including targets receiving no
injections. They are not reselected according to a new surrogate's ranks.
The true-tail filter uses original train/validation/test facts; injected facts
and synthetic target facts are excluded.

## Fixed settings

| Component | Default settings |
|---|---|
| TransE surrogate | 500 epochs, batch 256, learning rate 0.01, margin 1.0 |
| Candidate generation | Supervised PRA, threshold 0.9, top 10 templates, top 10 entities per template |
| MaSS path selection | Delta 0.7, baseline policy, maximum 2 distinct new triples per target |
| ConvE victim | Dimension 64, embedding height 8, 32 filters, kernel size 3 |
| ConvE training | 500 epochs, batch 256, learning rate 0.001, label smoothing 0.1 |
| ConvE dropout | Input 0.2, feature 0.2, hidden 0.3 |
| CPU execution | 4 Torch threads |

The surrogate and victim use their recorded seed. The two ConvE models are
trained afresh at each point and must have matching initialization hashes.
Injection counts can vary with the surrogate dimension; the reports include
actual counts and target coverage.

## Commands

```powershell
# Select dimensions and evaluate multiple seeds.
python Experiment_Setup/Evaluation/Evaluate_RQ4_Embedding_Sensitivity.py --dimensions 32 64 128 --seeds 42 43 44

# Continue a compatible saved experiment.
python Experiment_Setup/Evaluation/Evaluate_RQ4_Embedding_Sensitivity.py --resume

# Rebuild tables and figures from completed evaluations.
python Experiment_Setup/Evaluation/Evaluate_RQ4_Embedding_Sensitivity.py --plot-only
```

Other options include `--experiment-dir`, `--output-dir`, `--surrogate-epochs`,
`--victim-epochs`, `--victim-dim`, and `--torch-threads`. Keep settings other than
surrogate dimension fixed within a sensitivity comparison. A short-epoch run
is useful for verification but should be identified as a smoke test.

Resume verifies saved configuration, input/code hashes, and stage artifacts
before reusing them. Use the same configuration when resuming. Existing
baseline experiment outputs are not overwritten.

## Outputs

The source experiment defaults to:

```text
outputs/Experiments/UMLS_feasible_rank11_15_seed42
```

The destination defaults to `<experiment>/RQ4/embedding_sensitivity`. It contains:

- `sweep_config.json`: experiment settings and source provenance.
- `embedding_sensitivity_runs.csv` and `.json`: per-seed, per-dimension results.
- `embedding_sensitivity_summary.csv` and `.json`: results aggregated by dimension.
- `figure9_umls_embedding_sensitivity.png` and `.pdf`: line/bar chart.

Each point has its own `seed_42/dim_32` directory, with corresponding names for
other seeds and dimensions:

```text
surrogate/
feasibility/
poisoning/
conve_victim_models/
conve_evaluation/
run.log
stage_manifest.json
```

The evaluation CSVs retain individual target ranks so that Hit@10 and MRR can
be independently recalculated. Report aggregate results alongside seed count,
actual injection counts, and clean-performance controls.

## Interpretation limits

This is a UMLS adaptation of the paper's analysis. The default is one seed,
whereas the paper reports 10 attacks. The local experiment has 100 fixed
targets originally selected at TransE ranks 11–15; the paper uses 500 targets
near rank 500. The local ConvE victim dimension is 64 rather than the paper's
128. The paper's Figure 9 uses FB15k-237 and WN18RR, not UMLS.

The plot measures the final poisoned-model attack metrics, not surrogate
training loss or the change in surrogate link-prediction accuracy. The UMLS
experiment must establish which dimension performs best from its own results;
128 is not guaranteed to outperform 32 or 64. Use multiple seeds before making
strong claims about sensitivity or an optimal dimension.
