# Test_Triplet

This experiment trains a fresh TransE attacker surrogate and fresh paired TransE
victims, then compares each target's plausibility on clean and poisoned graphs.
It reuses saved MaSS PRA relation paths and intermediate-entity candidates. It
recomputes their embedding-dependent scores and selects poisoning paths again.
It does not regenerate the PRA templates or search for new intermediate entities.

Run from `C:\my-code\Robust Embedding`:

```powershell
.\.venv\Scripts\python.exe Mass_implement\Test_Triplet\run_experiment.py
```

The final comparison is `Output/Final result.csv`. Its `delta` column is:

```text
delta = clean_plausibility - poisoned_plausibility
```

A negative delta means the target became more plausible to the poisoned victim.
These are raw negative-L2 model scores, not probabilities or biomedical truth.

## Inputs and model roles

- `Input/train.txt`: original clean training graph; trains both the surrogate
  and clean victim. Its order is preserved.
- `Input/valid.txt` and `Input/test.txt`: held out from training; prevent fact
  leakage during poisoning.
- `Input/entities.txt` and `Input/relations.txt`: fixed ID mappings for all models.
- `Input/target_triplet.txt`: headerless TAB-separated targets, one per line.
  Targets must use known names and be absent from all original data splits.
- `Input/mass_candidates.json`: saved MaSS candidate bundle matched by the actual
  target triple, independently of JSON order or original target IDs.
- `Input/poisoned_train.txt`: generated clean-plus-injected graph; trains only
  the poisoned victim. The target is never directly added as a positive fact.

The initial target is `(antibiotic, complicates, reptile)`, original `target_003`.
Its 43 candidates were copied from the existing UMLS feasibility output and
assigned local ID `target_001`. The bundle records the original path and hash.

`Surrogate_Model/TransE_Surrogate.py` and `Victim_Model/TransE_Victim.py` define
independent local architectures and loaders. `run_experiment.py` trains these
classes directly. It does not load an old trained model as an initialization.

Defaults in `config.json` are 128 dimensions, 500 epochs, batch size 256, Adam
learning rate 0.01, margin 1, uniform one-negative head/tail corruption, and L2
entity normalization after each optimizer step. The surrogate seed is 43. Both
victims use seed 42 and are independently trained from identical initial weights.
MaSS uses delta 0.7, budget 2 and its baseline selection policy, with sampling
seed 42 plus each target's zero-based row index. The sampling delta and the final
CSV's score-difference `delta` are different quantities.

## Outputs

- `Graphs/Clean/clean_graph.graphml` and
  `Graphs/Poisoned/poisoned_graph.graphml` contain every mapped entity and every
  training fact, including parallel relations between the same endpoints.
  Edge attributes include `relation` and `injected`.
- `Surrogate_Model/surrogate_transe.pt` contains the new attacker weights.
- `Victim_Model/clean_transe.pt` and `poisoned_transe.pt` contain the paired victims.
- `Output/rescored_paths/` contains fresh candidate scores and the matching
  per-target feasibility summary used by the existing MaSS constructor.
- `Output/selected_poisoning_paths.json` preserves complete selected paths,
  intermediate entities, scores, sampling and per-target attribution.
- `Output/injected_triples.tsv` contains unique new facts only.
- `Output/poisoning_summary.json` and `.csv` describe selection and coverage.
- `Output/training_logs/` contains stage logs, loss curves and training summaries.
- `Output/Final result.csv` contains one row per target with clean plausibility,
  poisoned plausibility, **clean-minus-poisoned delta**, selection status and counts.
- `Output/Final result.json` retains full precision and run settings.
- `Output/experiment_manifest.json` records input/code/environment fingerprints,
  stage hashes, model controls and verification checks.

Budget 2 is a maximum of distinct non-clean facts per target. A path is selected
atomically; existing clean edges need no additional injection. Different targets
can share an injected edge, which appears only once in the poisoned graph. Zero
eligible/selected paths are reported honestly; targets are retained in results.

## Multiple targets and new experiments

Add additional TAB-separated targets to `target_triplet.txt` and provide a
matching candidate entry for each target in `mass_candidates.json`. Entries have
this structure:

```json
{
  "targets": [
    {
      "source_target_id": "original source ID",
      "source_file": "path to the original MaSS target JSON",
      "source_sha256": "SHA256 of that source file",
      "target_triple": ["head", "relation", "tail"],
      "mass_output": {
        "target_triple": ["head", "relation", "tail"],
        "ranked_candidate_paths": []
      }
    }
  ]
}
```

Use complete real MaSS output objects for `mass_output`, including the candidate
paths. The runner also accepts a single unwrapped original MaSS target JSON.
All targets share one combined poisoned graph and poisoned victim; their effects
can interact. This is not a separate poisoned-victim experiment for every target.

To preserve completed runs, copy `config.json` and the source `Input/` snapshots
into a new experiment directory (excluding the generated `poisoned_train.txt`),
then use `--config path/to/new/config.json`. All configured inputs and outputs
resolve relative to that configuration. Update/remove source-snapshot metadata
when deliberately using a different source dataset.

To verify/reuse an unchanged completed or interrupted run:

```powershell
.\.venv\Scripts\python.exe Mass_implement\Test_Triplet\run_experiment.py --resume
```

Resume verifies hashes before reusing completed stages and retrains only an
unfinished stage. Changed inputs, configuration, code or generated artifacts are
rejected rather than silently mixing experiments. Original project datasets and
the final RQ1 CSV are protected from modification.

Run the focused tests:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s Mass_implement\Test_Triplet -p test_experiment.py -v
```
