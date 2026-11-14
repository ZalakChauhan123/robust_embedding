# UMLS — ConvE victim: enhancement of baseline attacks

Arithmetic means across 1 seed(s): 42.

The evaluation populations below are a local interpretation of the discussion in MaSS section 5.5; the original Table 3 does not explicitly label them. Before is the unenhanced attack. Each after variant applies only the named constraint.

## Type constraint: Set B attack effectiveness

| Attack | Before Hit@10 | Before MRR | After Hit@10 | After MRR |
|---|---:|---:|---:|---:|
| DirectAdd | 0.8300 | 0.4019 | 0.4700 | 0.1938 |
| ComAttack | 0.7800 | 0.3129 | 0.2700 | 0.1123 |

## Plausibility constraint: Set A clean utility (stealthiness)

| Attack | Before Hit@10 | Before MRR | After Hit@10 | After MRR |
|---|---:|---:|---:|---:|
| DirectAdd | 0.9879 | 0.8973 | 0.9894 | 0.9114 |
| ComAttack | 0.9924 | 0.9018 | 0.9939 | 0.9190 |

All metrics come from the corresponding poisoned ConvE victim. Higher Set B metrics indicate stronger target promotion; higher Set A metrics indicate better preservation of clean prediction performance. An improvement is not assumed.
