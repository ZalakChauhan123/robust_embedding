# DICE target-triplet experiment

Use a separate Python 3.12 environment for all three native models. The tested Windows dependencies use DICE's NumPy 1.26.4 pin. DICE 0.3.2 also supports TransE and ComplEx.

```powershell
python -m pip install "dicee==0.3.3" "pyarrow==20.0.0" "scipy==1.15.3"
```

From the repository root, run the selected model with your DICE environment active.

```powershell
python Mass_implement/Test_Triplets_Dice/run_experiment.py --model TransE
python Mass_implement/Test_Triplets_Dice/run_experiment.py --model ComplEx
python Mass_implement/Test_Triplets_Dice/run_experiment.py --model RotatE
```

DICE may download the GPT-2 tokenizer files to its tokenizer cache during graph setup.

The default is 500 epochs per model. Add `--epochs 1` for a quick workflow check. The UMLS splits are snapshots from `Test_Triplet`. The target file contains the first ten triplets from the existing UMLS Set B in source order, including `(antibiotic, complicates, reptile)`. The candidate bundle records the corresponding saved MaSS files and their hashes.

The runner trains the selected native DICE model on the clean training graph, rescores the saved candidates, and uses the existing MaSS graph construction code to select up to two additions per target. It combines the additions into one shared poisoned graph for that model and trains two Victim checkpoints of the same model type, one on each graph, with the same seed and vocabulary. DICE's negative sampling training is used for all three checkpoints.

Path rescoring uses DICE triple scores and a relation composition distance. TransE adds relation vectors, ComplEx multiplies the complex relation vectors element by element, and RotatE composes the phases and compares the corresponding unit rotations. Victim ranks use the native DICE triple logits.

Open the model's CSV in `Output/final_result/` for the latest completed comparison. It contains one row per target in input order. The two ranks are filtered tail ranks over the same entity vocabulary and the same original train, validation, and test facts. Ties use the average rank. Rank change is poisoned rank minus clean rank. A negative number means the target moved closer to rank 1.

Training logs and settings are under `Output/Details/`, grouped by model. The graph exports under `Graphs/` are tab-separated triples. DICE checkpoint directories under `Poisoned/` and `Victim/` contain the model and the vocabulary needed to reload it.
