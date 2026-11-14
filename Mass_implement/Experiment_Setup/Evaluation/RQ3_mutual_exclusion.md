# UMLS version of Figure 7

Run from `Mass_implement` after the RQ3 budget sweep:

```powershell
python Experiment_Setup/Evaluation/Plot_RQ3_Mutual_Exclusion.py
```

This reads the saved injected triplets and RQ3 summaries at budgets
`2 4 6 8 10 12`. It regenerates no attacks and trains no models. It recomputes
overlap flags against clean UMLS training facts and checks that the counts
match the saved summaries. When a sweep manifest is present, it also checks
the recorded training and injected-file hashes.

For each injected `(h, r, t)`, flag the triplet if clean training contains
`(h, r, other_tail)` or `(other_head, r, t)`. Count it once if both directions
apply. The plotted metric is explicitly defined as:

```text
filter ratio = mutual_detected / total_unique_injected
remaining ratio = 1 - filter ratio
```

Only mutual-exclusion flags are included; type constraints and their combined
union do not contribute to this chart. These flags implement the paper's
stated strict overlap heuristic. They are not confirmed contradictions when
a relation validly permits multiple heads or tails.

The default Figure-7-style chart compares MaSS, DirectAdd, and ComAttack at
**x=2** with one dataset series, **UMLS**. This budget is our explicit choice;
Figure 7 in the paper does not specify its budget or give a ratio formula.
A second chart groups the three attacks at every evaluated budget.
Both use a vertical range of **0 to 1** because the measured UMLS ratios
exceed the paper chart's 0 to 0.05 range.

To choose another budget for the attack comparison:

```powershell
python Experiment_Setup/Evaluation/Plot_RQ3_Mutual_Exclusion.py --budget 12
```

Other options: `--sweep-dir`, `--output-dir`, `--budgets`, and `--train`.
The selected `--budget` must be included in `--budgets`.

Outputs are saved under `<sweep-dir>/mutual_exclusion`:

- `figure7_umls_x2.png` and `.pdf` (or the selected budget).
- `figure7_umls_budget_sweep.png` and `.pdf`.
- `mutual_exclusion_budget_summary.csv` and `.json`, including detected and
  remaining counts, both ratios, and directional overlap counts.
- `figure7_manifest.json`, with the rule, denominator, budget, and input hashes.

The UMLS results differ substantially from the paper's bars. Present them as
measurements of the current UMLS attack implementations under the stated
overlap rule, rather than claiming numerical agreement with Figure 7.
