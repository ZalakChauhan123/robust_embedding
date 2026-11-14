# Fresh attacker surrogates: UMLS RQ1

| Attacker dimension | Attack | Hit@10 | MRR | Injected targets | New triplets |
|---:|---|---:|---:|---:|---:|
| 64 | MaSS | 0.2700 | 0.10614502 | 96 | 166 |
| 64 | DirectAdd | 0.3100 | 0.13750947 | 100 | 200 |
| 64 | ComAttack | 0.3100 | 0.13854933 | 100 | 200 |
| 128 | MaSS | 0.2800 | 0.09461188 | 96 | 159 |
| 128 | DirectAdd | 0.3100 | 0.15890901 | 100 | 200 |
| 128 | ComAttack | 0.3000 | 0.11857567 | 100 | 200 |

Victim: TransE128; clean Hit@10 = 0.30, MRR = 0.11094261788036644. All100 original targets remain in each evaluation.
Both surrogates and paired victims train from scratch for500epochs, Adam lr0.01, batch256, margin1, seed42. Only surrogate dimension varies between the fresh attacker conditions.
MaSS uses supervisedPRA threshold0.9, top10 templates, top10 entities, delta0.7 and baseline selection. All methods have budget2. DirectAdd and ComAttack retain the project's target-promotion adaptations.
Fresh128 surrogate and clean128 victim have identical weights under these matched settings; this condition is clear-box. Results are from one seed and do not establish general transfer or guaranteed improvement.
Targets were originally selected with64-dimensional TransE ranks11–15; they were not reselected. The final RQ1 CSV is unchanged.

Reproduce: `.venv\Scripts\python.exe Mass_implement\Experiment_Setup\Evaluation\Evaluate_RQ1_Fresh_Surrogates.py --resume`

Comparison scope: the previous frozen64 surrogate has no complete saved training configuration; its adjacent loss history ends at epoch300. Both fresh surrogates train for500epochs at learning rate0.01. Fresh-versus-frozen differences therefore include changed training history and regenerated poisoning. Only fresh64 versus fresh128 is the controlled surrogate-dimension comparison. MaSS injects166 versus159 unique triplets, respectively, under the same per-target budget2.
