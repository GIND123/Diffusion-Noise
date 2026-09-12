# Roadmap to a top-tier journal submission

Honest gap analysis, updated 2026-09-12 after ~290 runs. Ordered by what would sink the paper
first, not by effort.

**Where we are:** ~290 runs. Addition (to 20 digits), subtraction,
multiplication, parity, reverse, sorting control. 5 positional encodings × 2
architectures × method/baseline, 3–5 seeds, component ablations, 3 model sizes,
two published baselines reimplemented in-harness, bootstrap confidence
intervals, a matched-compute table, a numerical audit, and a decoding-order
analysis.

**Blunt assessment:** the comparisons gap is largely closed. Two risks remain.
First, **no diffusion-side published baseline** (MGDM, Adaptive Order Policies)
has been run — a reviewer of a diffusion paper will expect one. Second, and more
interesting, **we can no longer explain our own result**: the obvious mechanism
(better decoding order) was tested and refuted, so the paper currently reports an
effect it cannot account for.

---

## Tier 1 — Would cause rejection if missing

### 1.1 Head-to-head against published methods — ✅ **largely closed**

Randomized PE (Ruoss 2023) and Abacus embeddings (McLeish 2024) are now
reimplemented in our harness and compared at matched compute, with bootstrap
confidence intervals. Our method wins on both architectures and the intervals
against Abacus on diffusion do not overlap.

Still open: **position coupling (Cho et al. 2024)** as a separately implemented
variant, and **MGDM / Adaptive Order Policies** on the diffusion side. Our
place-value scheme is close enough to position coupling that a reviewer may
accept the concession in the README, but MGDM remains a genuine gap if the paper
claims anything about diffusion planning.

<details><summary>Original gap list, for the record</summary>

Reproduce and compare, in our own harness, at matched compute:

| Method | Why it must be there |
|---|---|
| **Position coupling** (Cho et al. 2024) | Essentially our method in the causal setting. Without it a reviewer concludes we reinvented it. |
| **Abacus embeddings** (McLeish et al. 2024) | The strongest published autoregressive arithmetic length generalization. It is the SOTA number. |
| **Randomized positional encodings** (Ruoss et al. 2023) | Standard extrapolation baseline. *Launched — grid 2, study E.* |
| **MGDM** (Ye et al., ICLR 2025) | The diffusion-vs-autoregressive reference. Our diffusion arm must be shown to be a fair implementation of it, or better. |
| **Adaptive Order Policies** (arXiv 2606.00295) | Closest concurrent diffusion work. "No comparison to obvious concurrent work" is a rejection. |

Reproducing published *numbers* is not enough — they must run inside our harness
so the comparison is matched on data, parameters, and compute.

</details>

### 1.2 Matched-compute accounting — ✅ done (Table 8)

The most common reason diffusion-vs-autoregressive papers are rejected. We must
report, in one table: non-embedding parameters (±5%), training FLOPs, tokens
seen, optimizer/schedule, and **measured wall-clock and service units**. Our
diffusion arm uses *T* forward passes at inference while autoregressive uses one
per token — that must be stated explicitly, not buried.

Partly addressed: the denoising-pass sweep is launched (grid 2, study G), which
gives accuracy as a function of inference compute.

### 1.3 Statistical rigor — ✅ mostly closed

Bootstrap 95% confidence intervals over seeds are now reported for the
published-baseline table, and the headline row has 5 seeds. Still worth adding:
**paired bootstrap over test instances** (not just over seeds). Given we already found one configuration spanning 3%–68% across
seeds, this matters more here than in a typical paper. Consider 5 seeds for
headline rows.

### 1.4 Scaling — ✅ done

10.7M / 25M / 85M all run; see Table 9. A reviewer will ask whether the effect is a small-model
artifact; without this the answer is "we don't know."

---

## Tier 2 — Strongly expected at this level

### 2.1 Broader benchmarks

| Benchmark | Status | Why |
|---|---|---|
| Addition | ✅ done, to 20 digits | primary |
| Subtraction | ✅ done | second place-local operation |
| Multiplication | ✅ done — **fails** | boundary condition |
| Sorting | ✅ done | order-insensitive control |
| **Parity** | ✅ done — chain without alignment, barely helps |
| **Reverse** | ✅ done — alignment without chain, **100% at 2× length** |
| **SCAN or PCFG** | ❌ | compositional generalization; moves beyond arithmetic |
| **Countdown / Sudoku / 3-SAT** (MGDM suite) | ❌ | lets us compare on *their* benchmark rather than only ours |

The arithmetic-only framing is the second-biggest risk after missing baselines.
At least one non-arithmetic task is needed to claim the finding is about
bidirectional generation rather than about digits.

### 2.2 Mechanistic analysis — ⚠️ **ran it; the obvious explanation is false**

`src/order_analysis.py` records the denoising pass at which each digit is fixed.
The correlation between significance and reveal order is −0.40 (ours) and −0.32
(baseline): near-identical, and opposite in sign to the carry-chain prediction.
**Decoding order does not explain the diffusion advantage.**

Remaining mechanistic work, now the most valuable open item:
- Ablate *iterative refinement* (vary T at fixed training) against *bidirectional
  attention* to separate the two candidate explanations
- Attention-pattern inspection on place-value-aligned heads
- **Failure analysis at 20 digits.** Per-digit accuracy is 68.9% — *which*
  digits fail? Leading digits? Positions past the training range?

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

## Suggested order from here

1. **Explain the effect.** Decoding order is ruled out. Separate the two
   remaining candidates by ablating iterative refinement (vary T at fixed
   training) against bidirectional attention. Highest scientific value, and the
   paper is weaker without it.
2. **One diffusion-side published baseline** — MGDM is the obvious choice, and
   its repository is public and runs these task families.
3. **Paired bootstrap over test instances**, not only over seeds.
4. **One compositional task** (SCAN or PCFG) to move the claim beyond arithmetic
   and place value.
5. Failure analysis at 20 digits — per-digit accuracy is 68.9%, so *which*
   digits fail is directly answerable from checkpoints we already have.

---

## Reality check on compute

Every H100 run costs ~5 service units and takes ~2 minutes. A full 90-run grid
is **under 1%** of the 50 kSU allocation. Compute is not the constraint for any
item above — implementation time is.

The one caveat is that H100s require the locally built environment
(`hf_env`, torch 2.5.1+cu121); the cluster module is 2.0.1 and predates the
hardware. See `slurm/launch.sh`.
