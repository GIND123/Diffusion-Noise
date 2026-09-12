# Results

## Table 1 — Main result (addition, distance-rule encoding)

| configuration | 5d | 6d | 7d | 8d | 10d | 12d | 15d | 20d | H*(50%) |
|---|---|---|---|---|---|---|---|---|---|
| Autoregressive / baseline | 100.0±0 | 21.5±24 | 1.1±1 | 0.1±0 | 0.0±0 | 0.0±0 | 0.0±0 | 0.0±0 | 5 |
| Autoregressive / **ours** | 100.0±0 | 99.4±0 | 80.9±13 | 34.2±30 | 3.0±4 | 0.2±0 | 0.0±0 | 0.0±0 | 7 |
| Masked diffusion / baseline | 100.0±0 | 57.4±32 | 0.2±0 | 0.0±0 | 0.0±0 | 0.0±0 | 0.0±0 | 0.0±0 | 6 |
| Masked diffusion / **ours** | 100.0±0 | 100.0±0 | 97.6±1 | 81.3±5 | 33.1±7 | 7.6±1 | 0.8±0 | 0.0±0 | 8 |

## Table 2 — Positional encoding sweep (place-value ids)

| encoding | AR in-dist | AR H*(50%) | Diffusion in-dist | Diffusion H*(50%) |
|---|---|---|---|---|
| none | 98.8±1 | 6 | 0.0±0 | 0 |
| learned abs. | 100.0±0 | 6 | 100.0±0 | 8 |
| sinusoidal | 100.0±0 | 6 | 98.8±2 | 6 |
| rotary | 100.0±0 | 10 | 100.0±0 | 12 |
| distance rule | 100.0±0 | 7 | 100.0±0 | 8 |

## Table 3 — Component ablation (addition, distance rule)

| segments | random offset | architecture | 6d | 8d | 12d | H*(50%) |
|---|---|---|---|---|---|---|
| no | no | Autoregressive | 0.0 | 0.0 | 0.0 | 0 |
| no | no | Masked diffusion | 72.0 | 52.2 | 2.7 | 8 |
| no | yes | Autoregressive | 0.0 | 0.0 | 0.0 | 0 |
| no | yes | Masked diffusion | 73.0 | 58.7 | 4.8 | 8 |
| yes | no | Autoregressive | 99.0 | 15.7 | 0.0 | 7 |
| yes | no | Masked diffusion | 99.5 | 86.5 | 13.8 | 8 |
| yes | yes | Autoregressive | 100.0 | 27.5 | 0.0 | 7 |
| yes | yes | Masked diffusion | 99.8 | 75.7 | 11.5 | 8 |

## Table 4 — Multiplication (place value is NOT the algorithm)

| configuration | 3d | 4d | 5d | 6d | 7d |
|---|---|---|---|---|---|
| Autoregressive / baseline | 2.2±0 | 0.0±0 | 0.0±0 | 0.0±0 | 0.0±0 |
| Autoregressive / **ours** | 2.7±1 | 0.0±0 | 0.0±0 | 0.0±0 | 0.0±0 |
| Masked diffusion / baseline | 0.5±0 | 0.0±0 | 0.0±0 | 0.0±0 | 0.0±0 |
| Masked diffusion / **ours** | 2.7±1 | 0.0±0 | 0.0±0 | 0.0±0 | 0.0±0 |

## Table 5 — Per-digit accuracy (partial credit)

| configuration | 6d | 8d | 12d | 20d |
|---|---|---|---|---|
| Autoregressive / baseline | 80.3 | 48.3 | 4.1 | 3.4 |
| Autoregressive / **ours** | 99.9 | 81.7 | 38.1 | 19.1 |
| Masked diffusion / baseline | 88.9 | 32.0 | 9.9 | 3.4 |
| Masked diffusion / **ours** | 100.0 | 97.6 | 83.9 | 64.6 |

## Table 6 — Against published methods (addition, 8 digits)

| method | architecture | 6d | 7d | 8d | 95% CI at 8d |
|---|---|---|---|---|---|
| sequential ids (baseline) | Autoregressive | 7.7 | 0.0 | 0.0 | [0.0, 0.0] |
| sequential ids (baseline) | Masked diffusion | 28.3 | 0.3 | 0.0 | [0.0, 0.0] |
| randomized PE (Ruoss 2023) | Autoregressive | 10.0 | 0.0 | 0.0 | [0.0, 0.0] |
| randomized PE (Ruoss 2023) | Masked diffusion | 97.2 | 61.8 | 7.7 | [4.0, 12.0] |
| Abacus embeddings (McLeish 2024) | Autoregressive | 59.2 | 12.0 | 0.8 | [0.0, 1.5] |
| Abacus embeddings (McLeish 2024) | Masked diffusion | 94.5 | 82.3 | 63.8 | [59.5, 68.5] |
| place-value ids (ours) | Autoregressive | 97.3 | 52.0 | 10.5 | [1.5, 26.5] |
| place-value ids (ours) | Masked diffusion | 99.8 | 96.0 | 75.3 | [71.5, 82.0] |

## Table 7 — Does the method need place value, or just alignment?

Parity has a sequential chain but no place value; reverse has positional alignment but no chain. Together with multiplication (no place-local structure at all) these bound where the method applies.

| task | architecture | ids | in-dist | 2x length | H*(50%) |
|---|---|---|---|---|---|
| parity | Autoregressive | sequential | 100.0 | 0.0 | 12 |
| parity | Autoregressive | aligned (ours) | 100.0 | 0.3 | 15 |
| parity | Masked diffusion | sequential | 100.0 | 0.0 | 12 |
| parity | Masked diffusion | aligned (ours) | 100.0 | 0.2 | 15 |
| reverse | Autoregressive | sequential | 100.0 | 0.0 | 15 |
| reverse | Autoregressive | aligned (ours) | 100.0 | 30.7 | 20 |
| reverse | Masked diffusion | sequential | 100.0 | 0.0 | 12 |
| reverse | Masked diffusion | aligned (ours) | 100.0 | 100.0 | 40 |

## Table 8 — Matched compute

| architecture | non-embedding params | train tokens | train FLOPs | inference passes/example |
|---|---|---|---|---|
| Autoregressive | 10.75M | 250M | 1.61e+16 | 23 |
| Masked diffusion | 10.75M | 250M | 1.61e+16 | 16 |

Both arms share one transformer, identical data, optimizer and step count. Diffusion spends T refinement passes at inference where the autoregressive model spends one per emitted token.

## Table 9 — Scaling

| size | architecture | ids | 6d | 8d | 10d |
|---|---|---|---|---|---|
| d=384 | Autoregressive | baseline | 35.2 | 0.2 | 0.0 |
| d=384 | Autoregressive | **ours** | 99.5 | 52.3 | 5.7 |
| d=384 | Masked diffusion | baseline | 55.0 | 0.0 | 0.0 |
| d=384 | Masked diffusion | **ours** | 100.0 | 84.8 | 32.0 |
| d=512 | Autoregressive | baseline | 28.2 | 0.2 | 0.0 |
| d=512 | Autoregressive | **ours** | 99.8 | 71.0 | 4.7 |
| d=512 | Masked diffusion | baseline | 36.0 | 0.0 | 0.0 |
| d=512 | Masked diffusion | **ours** | 100.0 | 72.3 | 21.3 |
| d=768 | Autoregressive | baseline | 34.7 | 0.0 | 0.0 |
| d=768 | Autoregressive | **ours** | 100.0 | 48.5 | 0.3 |
| d=768 | Masked diffusion | baseline | 9.5 | 0.0 | 0.0 |
| d=768 | Masked diffusion | **ours** | 99.7 | 73.8 | 21.8 |

## Table 10 — Published protocol (train ≤20 digits, test to 100)

Matches the regime used by the arithmetic length-generalization literature, so these numbers are comparable to published work rather than only to our own baseline.

| method | architecture | 20d | 25d | 30d | 40d | 50d | 60d | 80d | 100d |
|---|---|---|---|---|---|---|---|---|---|
| sequential ids | Autoregressive | 66 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| sequential ids | Masked diffusion | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| **place-value (ours)** | Autoregressive | 100 | 98 | 68 | 67 | 66 | 65 | 61 | 48 |
| **place-value (ours)** | Masked diffusion | 100 | 100 | 90 | 49 | 8 | 1 | 0 | 0 |
| Abacus (McLeish 2024) | Autoregressive | 85 | 29 | 1 | 0 | 0 | 0 | 0 | 0 |
| Abacus (McLeish 2024) | Masked diffusion | 100 | 80 | 41 | 1 | 0 | 0 | 0 | 0 |
| randomized PE (Ruoss 2023) | Autoregressive | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| randomized PE (Ruoss 2023) | Masked diffusion | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

## Table 11 — Mechanism: bidirectional attention, not iterative refinement

| training | inference passes | 8d | 10d | 12d |
|---|---|---|---|---|
| standard diffusion | T=1 | 76.0 | 32.0 | 10.0 |
| standard diffusion | T=16 | 78.5 | 34.3 | 11.2 |
| one-shot (always fully masked) | T=1 | 86.8 | 42.7 | 14.8 |
| one-shot (always fully masked) | T=16 | 2.2 | 0.7 | 0.3 |

Iterative refinement adds nothing (T=1 ≈ T=16), and a one-shot bidirectional predictor is *better* than iterative diffusion at 1/16 the inference cost. The advantage attributed to masked diffusion on these tasks comes from bidirectional attention.

## Table 12 — Numeric base (is it place value, or decimal?)

| base | architecture | 8d | 10d | 12d |
|---|---|---|---|---|
| base 2 | Autoregressive | 18.5 | 3.2 | 0.8 |
| base 2 | Masked diffusion | 73.5 | 38.8 | 21.5 |
| base 10 | Autoregressive | 39.2 | 0.2 | 0.0 |
| base 10 | Masked diffusion | 78.5 | 34.3 | 11.2 |
| base 16 | Autoregressive | 14.0 | 0.2 | 0.0 |
| base 16 | Masked diffusion | 66.2 | 22.2 | 5.8 |

The method is about place value in general, not decimal digits.

## Figure 7 — where long answers break

Accuracy is highest at the units end and at the most significant end, and lowest in the middle, so errors are not simply a matter of positions beyond the trained range.

## Table 14 — Confidence intervals bootstrapped over test items

Intervals over *instances* rather than seeds, which is the stronger statement when seed variance is high.

| configuration | 8d accuracy | 95% CI |
|---|---|---|
| Autoregressive / baseline | 0.0 | [0.0, 0.0] (n=1500) |
| Masked diffusion / baseline | 0.0 | [0.0, 0.0] (n=1500) |
| Autoregressive / **ours** | 39.7 | [37.2, 42.1] (n=1500) |
| Masked diffusion / **ours** | 80.3 | [78.3, 82.2] (n=1500) |

