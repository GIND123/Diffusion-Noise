# Results

## Table 1 — Main result (addition, distance-rule encoding)

| configuration | 5d | 6d | 7d | 8d | 10d | 12d | 15d | 20d | H*(50%) |
|---|---|---|---|---|---|---|---|---|---|
| Autoregressive / baseline | 100.0±0 | 35.2±28 | 1.5±1 | 0.2±0 | 0.0±0 | 0.0±0 | 0.0±0 | 0.0±0 | 5 |
| Autoregressive / **ours** | 100.0±0 | 99.5±0 | 89.5±9 | 52.3±32 | 5.7±4 | 0.3±0 | 0.0±0 | 0.0±0 | 8 |
| Masked diffusion / baseline | 100.0±0 | 55.0±24 | 0.0±0 | 0.0±0 | 0.0±0 | 0.0±0 | 0.0±0 | 0.0±0 | 6 |
| Masked diffusion / **ours** | 100.0±0 | 100.0±0 | 97.5±0 | 84.8±5 | 32.0±9 | 7.5±2 | 0.5±0 | 0.0±0 | 8 |

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
| Autoregressive / baseline | 84.8 | 47.4 | 6.9 | 3.6 |
| Autoregressive / **ours** | 99.9 | 87.7 | 45.8 | 21.4 |
| Masked diffusion / baseline | 90.0 | 34.4 | 9.1 | 0.4 |
| Masked diffusion / **ours** | 100.0 | 98.0 | 84.1 | 65.2 |

