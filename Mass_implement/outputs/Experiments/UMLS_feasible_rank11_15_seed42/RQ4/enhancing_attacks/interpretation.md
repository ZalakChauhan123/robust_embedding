# UMLS Table 3: interpretation of the measured seed-42 experiment

The experiment uses a frozen 64-dimensional TransE surrogate, a matched
64-dimensional ConvE victim, 100 fixed attack targets, and a maximum budget
of two distinct new triplets per target. Type-only and plausibility-only
constraints are independent conditions.

The displayed type block measures target attack performance on Set B.
The plausibility block measures normal prediction utility on the 661 Set A
test triplets. This population assignment is an explicit local interpretation
of the paper's discussion; its Table 3 does not label populations clearly.
Both populations for all six conditions are in `enhancement_metrics.csv`.

Type constraints reduce target attack effectiveness for both local baselines.
Both type-constrained methods still inject 200 triples across all 100 targets,
so their decrease is not caused by a smaller actual injection count.

Plausibility constraints improve Set A Hit@10 and MRR for both methods while
reducing Set B attack performance. The cutoff is the global median of frozen
TransE scores on the 5,216 clean training triples: -0.6652185320854187.
This is a declared UMLS calibration choice, not a numerical cutoff supplied
by the MaSS paper, and it was not selected using enhanced victim outcomes.

## Actual injection coverage

| Attack | Original: triples / targets | Type-only: triples / targets | Plausibility-only: triples / targets |
|---|---:|---:|---:|
| DirectAdd | 200 / 100 | 200 / 100 | 200 / 100 |
| ComAttack | 200 / 100 | 200 / 100 | 130 / 65 |

All 100 targets remain in the attack evaluation, including uninjected ones.
ComAttack's plausibility condition has 35% fewer injected triples and reaches
only 65 targets. Its clean-utility gain therefore cannot be attributed to
plausibility alone; reduced poisoning coverage is a confounding factor.
A matched-count/coverage experiment would be needed to isolate those effects.

These are single-seed measurements of the current UMLS target-promotion
adaptations, rather than an exact numerical reproduction of the original
paper's datasets and source implementations. An enhancement name does not
guarantee improved attack strength, type compatibility, or every stealthiness
measure. Per-triple constraint audits and both clean/attack metrics are saved.
