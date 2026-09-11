# Results

## Table 1 — Main result (addition, distance-rule encoding)

| configuration | 5d | 6d | 7d | 8d | 10d | 12d | 15d | 20d | H*(50%) |
|---|---|---|---|---|---|---|---|---|---|
| Autoregressive / baseline | 100.0±0 | 27.3±12 | 0.3±0 | 0.0±0 | 0.0±0 | 0.0±0 | 0.0±0 | 0.0±0 | 5 |
| Masked diffusion / baseline | 100.0±0 | 12.0±0 | 1.2±0 | 0.0±0 | 0.0±0 | 0.0±0 | 0.0±0 | 0.0±0 | 5 |

## Table 2 — Positional encoding sweep (place-value ids)

| encoding | AR in-dist | AR H*(50%) | Diffusion in-dist | Diffusion H*(50%) |
|---|---|---|---|---|
| none | – | – | – | – |
| learned abs. | – | – | – | – |
| sinusoidal | – | – | – | – |
| rotary | – | – | – | – |
| distance rule | – | – | – | – |

## Table 3 — Component ablation (addition, distance rule)

| segments | random offset | architecture | 6d | 8d | 12d | H*(50%) |
|---|---|---|---|---|---|---|

## Figure 3 — accuracy vs denoising passes

Autoregressive models have no equivalent knob; this is compute spent purely at inference.

## Table 5 — Per-digit accuracy (partial credit)

| configuration | 6d | 8d | 12d | 20d |
|---|---|---|---|---|
| Autoregressive / baseline | 83.4 | 36.8 | 5.5 | 0.0 |
| Masked diffusion / baseline | 81.9 | 50.8 | 13.0 | 3.8 |

