# Roadmap to a top-tier journal submission

Honest gap analysis as of 2026-09-11. Ordered by what would sink the paper
first, not by effort.

**Where we are:** 180 runs. Addition, subtraction, multiplication, sorting
control. 5 positional encodings × 2 architectures × method/baseline, 3 seeds,
component ablations, a length ladder to 20 digits, and a numerical audit.

**Blunt assessment:** the *science* is in reasonable shape. The *comparisons*
are not. Every number so far compares our method against **our own** baseline.
No published method has been run head-to-head. That is the single thing most
likely to get this rejected, because our method is an adaptation of published
autoregressive work and a reviewer will immediately ask how it compares.

---

## Tier 1 — Would cause rejection if missing

### 1.1 Head-to-head against published methods ⚠️ **the critical gap**

We must reproduce and compare, in our own harness, at matched compute:

| Method | Why it must be there |
|---|---|
| **Position coupling** (Cho et al. 2024) | Essentially our method in the causal setting. Without it a reviewer concludes we reinvented it. |
| **Abacus embeddings** (McLeish et al. 2024) | The strongest published autoregressive arithmetic length generalization. It is the SOTA number. |
| **Randomized positional encodings** (Ruoss et al. 2023) | Standard extrapolation baseline. *Launched — grid 2, study E.* |
| **MGDM** (Ye et al., ICLR 2025) | The diffusion-vs-autoregressive reference. Our diffusion arm must be shown to be a fair implementation of it, or better. |
| **Adaptive Order Policies** (arXiv 2606.00295) | Closest concurrent diffusion work. "No comparison to obvious concurrent work" is a rejection. |

Reproducing published *numbers* is not enough — they must run inside our harness
so the comparison is matched on data, parameters, and compute.

### 1.2 Matched-compute accounting ⚠️

The most common reason diffusion-vs-autoregressive papers are rejected. We must
report, in one table: non-embedding parameters (±5%), training FLOPs, tokens
seen, optimizer/schedule, and **measured wall-clock and service units**. Our
diffusion arm uses *T* forward passes at inference while autoregressive uses one
per token — that must be stated explicitly, not buried.

Partly addressed: the denoising-pass sweep is launched (grid 2, study G), which
gives accuracy as a function of inference compute.

### 1.3 Statistical rigor

Currently mean ± standard deviation over 3 seeds. Needed: **paired bootstrap
over test instances with confidence intervals**, and significance tests on the
headline gaps. Given we already found one configuration spanning 3%–68% across
seeds, this matters more here than in a typical paper. Consider 5 seeds for
headline rows.

### 1.4 Scaling

All headline numbers are 10.7M parameters. *Launched — grid 2, study F* covers
10.7M / 25M / 85M. A reviewer will ask whether the effect is a small-model
artifact; without this the answer is "we don't know."

---

## Tier 2 — Strongly expected at this level

### 2.1 Broader benchmarks

| Benchmark | Status | Why |
|---|---|---|
| Addition | ✅ done, to 20 digits | primary |
| Subtraction | 🔄 launched (study H) | second place-local operation |
| Multiplication | ✅ done — **fails** | boundary condition |
| Sorting | ✅ done | order-insensitive control |
| **Parity / copy / reverse** | ❌ | standard length-generalization probes, very cheap |
| **SCAN or PCFG** | ❌ | compositional generalization; moves beyond arithmetic |
| **Countdown / Sudoku / 3-SAT** (MGDM suite) | ❌ | lets us compare on *their* benchmark rather than only ours |

The arithmetic-only framing is the second-biggest risk after missing baselines.
At least one non-arithmetic task is needed to claim the finding is about
bidirectional generation rather than about digits.

### 2.2 Mechanistic analysis — *why* diffusion wins

We have `src/trace.py` but have never run it at scale. Needed:
- **Decoding-order analysis.** Does the diffusion model resolve low-order digits
  first, i.e. discover the carry chain? This is the causal story behind the
  headline number and currently it is asserted, not shown.
- **Failure analysis at 20 digits.** Per-digit accuracy is 68.9% — *which*
  digits fail? Leading digits? Positions past the training range?
- **Attention inspection** on place-value-aligned heads.

This is the difference between "we measured an effect" and "we explained one."

### 2.3 Metrics not yet reported

- Accuracy vs **training tokens** (sample efficiency) — diffusion is known to be
  more data-efficient; worth checking here
- Accuracy vs **wall-clock** and vs **service units**
- Training curves per run (logged, never plotted)
- Calibration / confidence of the denoiser at each pass

---

## Tier 3 — Would strengthen, not required

- Variable-length canvas (diffusion must currently fix output length up front —
  a genuine diffusion-only limitation we side-step rather than solve)
- Block diffusion (BD3-LM) as the autoregressive↔diffusion interpolation
- Remasking / self-correction (ReMDM)
- Larger operand ranges (100+ digits, as the Abacus work reports)
- Non-decimal bases, to test whether the effect is about place value generally

---

## Suggested order

1. **Implement position coupling and Abacus in our harness** — without this the
   contribution is unclear. Days, not weeks; both are positional-id schemes and
   our code already supports custom identifiers.
2. **Add one non-arithmetic task** (parity or SCAN) — cheap, removes the
   "arithmetic-only" objection.
3. **Matched-compute table + bootstrap confidence intervals** — mostly analysis
   over runs we already have.
4. **Decoding-order analysis** — converts a measurement into an explanation.
5. Fold in scaling and denoising-pass results (launched).

---

## Reality check on compute

Every H100 run costs ~5 service units and takes ~2 minutes. A full 90-run grid
is **under 1%** of the 50 kSU allocation. Compute is not the constraint for any
item above — implementation time is.

The one caveat is that H100s require the locally built environment
(`hf_env`, torch 2.5.1+cu121); the cluster module is 2.0.1 and predates the
hardware. See `slurm/launch.sh`.
