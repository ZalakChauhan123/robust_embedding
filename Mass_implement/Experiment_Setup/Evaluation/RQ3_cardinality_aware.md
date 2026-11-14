# Handling UMLS overlap without claiming false contradictions

Keep the strict Figure-7-style overlap results as the paper-rule comparison.
Use a separate cardinality-aware report for claims about contradictions.

An injected `(h, r, t)` is a confirmed rule conflict only if:

```text
(another tail exists for (h, r) AND r has documented forward uniqueness)
OR
(another head exists for (r, t) AND r has documented inverse uniqueness)
```

Forward uniqueness means at most one tail for each head and relation.
Inverse uniqueness means at most one head for each relation and tail.
Rules must apply to the dataset's semantic-type nodes, not merely to some
individual-level interpretation. NLM describes the Semantic Network as
relationships between high-level semantic types:
https://www.nlm.nih.gov/research/umls/new_users/online_learning/SEM_001.html

Without a supporting rule, overlap is **inconclusive**, not a confirmed
contradiction. No overlap is likewise not proof of a fact's truth. Observed
multiplicity in clean data is evidence against universal uniqueness. The
absence of observed multiplicity does not establish uniqueness.

## Run all existing budgets

From `Mass_implement`:

```powershell
python Experiment_Setup/Evaluation/RQ3_Cardinality_Aware_Detection.py --sweep-dir outputs/Experiments/UMLS_feasible_rank11_15_seed42/RQ3/budget_sweep
```

This reads the saved injections, without retraining or altering the strict
overlap charts. New outputs are in `<sweep-dir>/cardinality_aware`:

- `cardinality_aware_budget_summary.csv` and `.json` for all budgets.
- `x_2`, `x_4`, etc., each containing per-triplet decisions, relation
  cardinality evidence, and a coverage summary with input hashes.

For a single experiment, the existing `--experiment-dir` option still works.
Other options include `--budgets`, `--output-dir`, `--train`, and `--methods`.

## Documented rules and interpretation

No curated UMLS exclusivity rules were available for this run. `SRSTRE2`
domain/range pairs do not establish cardinality. Consequently rule coverage
is zero and confirmed detection rates are `null` in JSON (blank in CSV),
meaning **not assessable**, rather than a successful 0% detection result.
The raw count of confirmed conflicts is zero because none can be confirmed.

To supply independently verified rules, use `--rules path/to/rules.json`.
The JSON has a `relations` object keyed by exact UMLS relation name. Each rule
may contain boolean `functional` and `inverse_functional` declarations.
Every true declaration requires a nonempty `source` describing its ontology
or manually validated rule. Omitted directions remain unknown. Unknown
relation names and true rules contradicted by clean multiplicity are rejected.
Do not assign true declarations just to improve the plotted filtering rates.

`rule_coverage_pct` reports injected triplets with at least one applicable
documented uniqueness constraint. The report includes a conditional conflict
rate among those assessed triplets. With partial coverage, the rate over all
injections counts only known-rule conflicts; it is not total contradiction
prevalence and should always be accompanied by coverage.

This is an additional UMLS assessment, not a replacement numerical
reproduction of Figure 7. Establishing a useful contradiction detector still
requires validated rules applicable to this data representation, or a
separately specified and validated detection method.
