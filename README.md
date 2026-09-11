# Does the Canvas Limit the Plan?

**Length and hardness generalization in from-scratch masked diffusion planners.**

Target venue: TACL · Constraints: trained from scratch (no pretrained weights),
masked-diffusion method, must beat published baselines, UMD Zaratan HPC.

> Cluster access, quotas, partitions and job mechanics live in the
> [Zaratan README](../README.md). This document is the project: budget,
> methodology, and experiment plan.

---

## 0. Read this first — model size, not thrift, is the constraint

**Don't agonize over SUs.** The scoped plan below costs ~5 kSU of a 50 kSU pool
and you should just run it. For calibration, last semester's entire usage was
**19 jobs / 498 SU (0.5 kSU)** — 480 of which was one 10-hour A100 job. Cost was
never a concern because usage was ~1% of the pool.

What *is* a hard constraint is the ceiling itself. The 50 kSU is enforced by the
scheduler as `GrpTRESMins=billing=3000000`; when it is exhausted, jobs stop
starting (`AssociationJobLimit`). That is not a budgeting preference.

And the thing that actually decides what fits:

| Config | Cost per training run | Equivalent |
|---|---|---|
| **Tiny (6M)** on a `a100_1g.5gb` slice | **~60 SU** | — |
| **Small (85M)** on a full A100 | **~3,500 SU** | **58 Tiny runs** |
| **Medium (303M)** | **~10,000+ SU** | 20% of the pool, for one run |

So the real trade is: *the entire 88-run study at 6M*, or *1.5 Small runs*.
That — not frugality — is why this plan is Tiny-only, and why Small/Medium
headline runs are the specific thing worth requesting an AAC allocation for.

---

## 0b. The arithmetic that forced the rescope

The original protocol budgets **~2,000 A100-hours**. On Zaratan that is:

```
2,000 A100-h × 48 SU/h = 96,000 SU = 96 kSU
```

Against the actual allocation:

| | |
|---|---|
| `msml612-class` pool | **50 kSU total** |
| Shared between | **67 students**, no per-user limits |
| Even-split fair share | **~746 SU** (≈15 A100-h, ≈106 MIG-slice-h) |
| Protocol requirement | **96 kSU = 192% of the entire class's semester** |

**And there is no free GPU tier.** `scavenger` (0 SU, 14-day walltime) contains
only `compute-*` and `bigmem-*` nodes — `GRES=(null)`. Every GPU node sits
exclusively in the paid `gpu-*` partitions. Verified 2026-09-10. So the usual
"run it free on the preemptible queue" escape hatch **does not exist here**.

Three levers, in order of leverage:

1. **Get a real allocation (do this first).** Faculty can apply to the UMD
   Allocations and Advisory Committee: **up to 50 kSU/year with minimal
   justification, up to 550 kSU/year with proper justification.** A TACL paper
   should not be funded from a 67-way-shared teaching pool. *Ask the professor
   to apply in week 1* — everything below gets easier if this lands.
2. **Use MIG slices.** `a100_1g.5gb` costs **7 SU/h vs 48** for a full A100. A
   6M-parameter model on ≤325-token sequences cannot saturate an A100 anyway,
   so a slice delivers roughly **2–3× more completed work per SU** — and the
   slice node exposes **28 of them**, so sweeps run wide in parallel.
3. **Scope down to 6M.** See §2. This is not a compromise on the science:
   MGDM's own headline result is that a **6M** diffusion model beats a **303M**
   autoregressive one.

**Working assumption for this plan: a negotiated ~5 kSU (10% of the class pool),
which must be cleared with the professor.** Everything below is costed to fit
that, with the full protocol staged behind an AAC allocation.

### The insight that makes this survivable

**The headline contribution is nearly free.** Hardness generalization =
train on easy instances, *evaluate* on harder ones. Evaluation is
inference-only: no extra training runs, seconds-to-minutes per split on a 6M
model. Every OOD curve in this paper is a by-product of models you already
trained for the baseline table.

The budget is spent on *training base models and the PE sweep*, not on the
contribution itself.

---

## 1. Compute cost model

| Resource | Rate | What 5 kSU buys |
|---|---|---|
| `a100_1g.5gb` (1/7 A100, 5 GB) | **7 SU/h** | ~714 slice-hours |
| Full A100 (40 GB) | 48 SU/h | ~104 GPU-hours |
| H100 | 144 SU/h | ~35 GPU-hours — **do not use** |
| CPU core (`scavenger`, free) | 0 SU | data generation, preprocessing, CPU eval |

Billing = **max**(1/core, 0.25/GiB, GPU rate) × *actual* walltime. Confirmed
against real `sacct` records (`billing=48` for 1×A100, `billing=7` for a slice).

**Estimated Tiny (6M) cost per training run** — *these are extrapolations from
the protocol's A100-hour figures and MUST be replaced by measured numbers at
Gate 0:*

| Task | Seq len | Est. slice-hours | Est. SU |
|---|---|---|---|
| Star-graph (l=5, d=2) | ~50 | ~8 | ~56 |
| Sorting (len 16) — control | ~35 | ~6 | ~42 |
| Countdown-3 | **37** | ~8 | ~56 |
| 3-SAT 5v | **258** | ~14 | ~98 |
| ~~Sudoku~~ | 164 | ~40 | ~280 ❌ |

Sequence lengths are from MGDM's own `cutoff_len` table — these tasks are
*tiny*, which is what makes the project viable at all.

---

## 2. Feasible scope — what is in and what is cut

| Original protocol | Decision | Why |
|---|---|---|
| 5 tasks incl. Sudoku | **4 tasks, Sudoku dropped** | Most expensive Tiny task (~12 A100-h) *and* saturated at 100% — zero headroom. Cite MGDM's published number instead. |
| Tiny + Small + Medium | **Tiny (6M) only** | Small headline runs are multi-day (~3 kSU *each*); Medium scaling check is ~9.6 kSU alone. Both deferred to an AAC allocation. |
| 5 seeds headline / 3 ablation | **3 seeds / 2 seeds** | Keeps paired-bootstrap CIs meaningful at a third of the cost. |
| PE sweep: 6 PE × 2 attn × 3 tasks × 3 seeds = 108 | **6 × 2 × 1 task × 3 seeds = 36** | Run on star-graph (cheapest). Extend to 3-SAT only if budget survives. |
| 8-row ablation × 5 tasks × 3 seeds = 120 | **Leave-one-out on 2 tasks, 2 seeds ≈ 12** | Keeps the "component X contributes Y points" claim. |
| 317 runs / ~2,000 A100-h | **~90 runs / ~5 kSU** | Fits a negotiated 10% of the pool. |

### Tasks kept

| Task | Role | Hardness dial | Train → OOD test | Why kept |
|---|---|---|---|---|
| **Star-graph** | **Primary showcase** | path length *l*, degree *d* | l=5,d=2 → l=7,10; d=3,5 | Cheapest run; most dramatic AR failure (mechanistically understood Clever-Hans at the junction); **two independent dials** separate "longer" from "harder" |
| **3-SAT** | Main hardness dial | # variables; clause/var ratio | 5v → 7v, 9v, 11v | Cleanest continuous dial; verifiable in ms; phase transition at ratio ≈4.26 gives a second orthogonal dial |
| **Countdown-3** | Comparability | # input numbers | 3 → 4, 5 | Only 37 tokens; MGDM publishes CD-3/4/5, so direct numeric comparison is possible |
| **Sorting** | **Negative control** | length | 16 → 32, 64 | Order-insensitive. If gains appear here equally, the planning framing is wrong. Non-negotiable — cheap and it buys reviewer trust. |

---

## 3. Research questions

**RQ1 (measurement, guaranteed).** Do from-scratch masked diffusion planners
generalize to instances harder than those seen in training, and do they degrade
more gracefully than matched AR models?

**RQ2 (mechanism).** Which bottleneck dominates the collapse?

| | Hypothesis | Prediction if true |
|---|---|---|
| **H1** | **Fixed canvas** — diffusion commits to output length before denoising | Sharp cliff where required trace length exceeds the training canvas; largely restored by variable-length generation |
| **H2** | **Positional encoding** — bidirectional attention has no causal mask to implicitly encode position | Large OOD spread across PEs, small in-distribution spread |
| **H3** | **Fixed step budget** — denoising steps don't scale with hardness | OOD accuracy rises with step count well past in-distribution saturation |

**RQ3 (method).** Can a from-scratch model combining the best answers extend the
hardness envelope beyond MGDM and Adaptive Order Policies?

### ⚠️ H1 is weakly testable as MGDM formats these tasks

MGDM's canvas sizes are **CD-3/4/5 = 37 / 64 / 74** and **3-SAT 5v/7v/9v =
258 / 285 / 325**. That is **+16%** from CD-4→CD-5 and **+26%** across the whole
3-SAT range — *sub*-linear, not the "super-linear growth" the original protocol
assumed. **Neither task meaningfully stresses the canvas.**

Consequences, choose deliberately:

- **Star-graph is the only task with a genuinely strong length dial** (path
  length l=5→10 doubles the output) → make it the primary H1 testbed.
- To test H1 on Countdown properly you would need **full Stream-of-Search
  traces** (thousands of tokens, the actual search process) rather than MGDM's
  compact final expressions. That raises data to tens of GB *and* run cost
  substantially — **out of budget at 5 kSU; revisit under an AAC allocation.**
- Otherwise, report H1 honestly as *tested primarily on star-graph*, and let
  H2 carry the mechanistic weight.

**The sharpest hook is H2**, not H1. Kazemnejad et al. (arXiv:2305.19466)
established that NoPE beats ALiBi/RoPE/APE for length generalization in
*causal* transformers — and the community explanation rests on **the causal
mask** supplying implicit position. Masked diffusion has **no causal mask**, so
there is a principled reason to expect the ranking *not* to transfer. Nobody has
checked. It is also the cheapest axis in the project.

---

## 4. Method

Absorbing-state masked diffusion, MDLM-style (SUBS parameterization, no timestep
conditioning), bidirectional transformer denoiser, **random init throughout** —
no pretrained weights, no pretrained tokenizer. Vocabularies are tens of symbols,
built from the training corpus.

| Component | Addresses | Implementation |
|---|---|---|
| **C1 — Length-adaptive canvas** | H1 | (a) padded canvas + learned `[EOS]`, or (b) block-wise diffusion (BD3-LM): block size *B* interpolates AR (*B*=1) ↔ full diffusion (*B*=L), giving a free ablation axis |
| **C2 — PE sweep** | H2 | NoPE / learned APE / sinusoidal / RoPE / ALiBi / randomized, matched in all else, run under **both** causal and bidirectional attention |
| **C3 — Adaptive step budget** | H3 | Stop when `max_i max_v p(x_i=v) ≥ τ` over masked positions, capped at `T_max`. **Always report accuracy-vs-NFE curves**, never a single cherry-picked *T* |
| **C4 — Hardness curriculum** | — | fixed-easy vs uniform mixture vs easy→hard anneal |

### What to claim, explicitly

> We do not claim novelty for learned decoding order (Mohamud et al. 2026),
> flexible-length masked diffusion (arXiv:2509.01025), or remasking (ReMDM).
> Our contribution is the first systematic study of hardness generalization for
> from-scratch masked diffusion planners, the characterization of positional
> encoding under bidirectional attention, and a combined recipe that extends the
> solvable hardness envelope.

Stating what you are *not* claiming is a strength in journal review.

---

## 5. Baselines

| # | Baseline | Source | Priority |
|---|---|---|---|
| 1 | Matched AR transformer | MGDM `train-sft.sh` | **Must** |
| 2 | Vanilla MDLM (uniform masking) | kuleshov-group/mdlm | **Must** |
| 3 | **MGDM** | HKUNLP/diffusion-vs-ar | **Must** — prior SOTA on these exact tasks |
| 4 | **Adaptive Order Policies** (arXiv:2606.00295) | reimplement | **Must** — closest concurrent work; "no comparison" = rejection |
| 5 | AR + reverse-order | Bachmann & Nagarajan | star-graph only |
| 6 | AR + teacherless | Bachmann & Nagarajan | star-graph only |
| 7 | Confidence-based adaptive inference (arXiv:2502.06768) | inference-only | cheap, add it |
| 8 | Block diffusion *B*-sweep | bd3lms | if budget survives |

### Matched-compute discipline (non-negotiable)

Hold constant and **report in a table**: non-embedding params (±5%), training
FLOPs, tokens seen, optimizer/schedule/warmup/decay, data splits and seeds.
Also report measured **GPU-hours and SU**, which MGDM does not publish.

**Tune the AR baseline as hard as your own model** — run its LR sweep. And set
`--max_new_tokens` above the longest training sequence; getting that wrong
silently cripples AR and is the most common reason diffusion papers get
rejected.

---

## 6. Evaluation

All five tasks have **exactly checkable** answers — no human eval, no
LLM-as-judge, no MAUVE, no BLEU. Say this in the paper; it removes a whole
category of objection.

| Task | Metric | Checker |
|---|---|---|
| Countdown | exact solve rate | evaluate expression vs target |
| 3-SAT | satisfying-assignment rate | substitute into CNF |
| Star-graph | exact path match | compare to ground truth |
| Sorting | exact match | compare to `sorted()` |

**Headline metric — the hardness envelope** `H*(ε)` = largest hardness level at
which a model exceeds accuracy ε. Report `H*(0.5)` and `H*(0.9)`, plus the full
accuracy-vs-hardness curve (the *shape* — graceful decay vs cliff — is what
separates H1 from H2/H3).

> **Reviewer risk:** defining your own metric can read as weak. Mitigate by
> **also reporting raw accuracy on MGDM's own published splits** (CD-4, CD-5,
> 3-SAT 7v/9v) so there are directly comparable numbers in the paper.

Statistics: 3 seeds headline / 2 ablation, mean ± sd (never a single run),
paired bootstrap over test instances with CIs. Decoding hyperparameters either
fixed identically across methods or swept for all — sweeping only for your own
method is the second-most-common rejection reason.

ELBO/NLL only as a **labeled upper bound**, never in the same column as AR NLL.

---

## 7. Experiment plan, costed

| Phase | Runs | Est. SU | Gate |
|---|---|---|---|
| **0. Calibrate + reproduce** | 4 | ~250 | **G1**: reproduce MGDM > AR by a large margin on star-graph + CD-3. **Also: measure real per-run cost and re-cost everything below.** |
| **1. Baselines** (3 methods × 4 tasks × 3 seeds) | 36 | ~2,270 | **G2**: AR *and* baseline diffusion both degrade on OOD splits. If both stay ~100%, dials are too weak. |
| **2. PE sweep** (6 PE × 2 attn × star-graph × 3 seeds) | 36 | ~2,020 | **G3**: OOD spread across PEs > ~2 points. If not, H2 is a (still publishable) negative result — reallocate. |
| **3. Components + AOP baseline** | ~12 | ~760 | **G4**: ≥1 component beats vanilla MDLM by ≥3 points OOD. If not → pivot to the measurement paper. |
| **OOD evaluation, all phases** | — | **~0** | inference-only |
| **Total** | **~88** | **~5,300 SU ≈ 10.6% of the class pool** | |

**Gate 0 is the most important step in this document.** Do not submit Phase 1
until one calibration run has produced a measured SU cost — every number above
is an extrapolation and could be off by 2–3×.

### Deferred until an AAC allocation lands

Small (85M) headline runs · Medium (303M) scaling check · PE sweep extended to
3-SAT · full Stream-of-Search Countdown traces for a real H1 test · Sudoku ·
5 seeds.

---

## 8. Storage plan

Datasets are negligible; **checkpoints are the only real cost**, and at 6M they
are small.

| Item | Size |
|---|---|
| All datasets incl. OOD splits (MGDM bundle + generated) | **< 5 GB** |
| MGDM repo | 146 KB (code only; data is a separate Drive download) |
| Tiny checkpoint — final weights fp32 / bf16 | 24 MB / 12 MB |
| Tiny checkpoint — full resumable (AdamW, ~16 B/param) | ~96 MB |
| **~90 runs, final weights only** | **~2.2 GB** |
| Peak incl. rolling resume checkpoints | **~10 GB** |

**~10 GB against a 300 GB group quota = ~3%.** Comfortably polite. (The original
317-run plan with Small/Medium would have been ~200 GB, or 68% of the shared
quota — another reason the scoped plan is the right call.)

Layout:

```
~                                            code, configs, results CSVs (10 GB, BACKED UP)
/scratch/zt1/project/msml612/user/govind02/  env, caches, runs, checkpoints (300 GB shared)
/afs/shell.umd.edu/project/msml612/          final ckpts + frozen splits w/ checksums (1 TB)
$TMPDIR (node-local, ~11 TB)                 per-job data staging
```

⚠️ **Never build the conda env in `~`** — 5–15 GB and 100k–300k files against a
10 GB / 650k-inode quota. Newest module is PyTorch 2.0.1 (2023), so you will
need your own env. See [Zaratan README §6](../README.md) for the exact exports.

---

## 9. Job template

```bash
#!/bin/bash
#SBATCH -J mdm-stargraph
#SBATCH -t 12:00:00                  # NEVER omit: default is 15 min
#SBATCH -c 8
#SBATCH --mem=32g
#SBATCH --gpus=a100_1g.5gb:1         # 7 SU/h, not 48
#SBATCH -p gpu-a100_1g.5gb
#SBATCH -o logs/%x-%j.out
#SBATCH --array=0-2                  # seeds; use arrays, never 36 hand-submitted jobs

source /etc/profile
SCR=/scratch/zt1/project/msml612/user/$USER
export HF_HOME=$SCR/hf PIP_CACHE_DIR=$SCR/pip_cache TORCH_HOME=$SCR/torch
conda activate $SCR/envs/diffusion

cp -r $SCR/data/stargraph $TMPDIR/            # stage to node-local
python train.py --seed $SLURM_ARRAY_TASK_ID \
                --data $TMPDIR/stargraph \
                --out $SCR/runs/$SLURM_JOB_NAME-$SLURM_ARRAY_TASK_ID \
                --save_total_limit 1
```

Rules: right-size `-t` (the scheduler reserves against *requested* walltime and
will block you and your classmates with `AssociationJobLimit`); use job arrays
for sweeps; `sbalance` before and after every phase.

---

## 10. Risks

| Risk | Impact | Mitigation |
|---|---|---|
| **Classmates drain the shared 50 kSU** | **Fatal** | Not hypothetical — 67 users, no per-user caps, first-come-first-served. Front-load runs (pool was at 0.00 kSU on 2026-09-10). **Get the AAC allocation.** |
| Measured cost ≫ estimate | High | Gate 0 exists for exactly this. Re-cost before Phase 1. |
| H1 untestable on MGDM-format tasks | Medium | Already known (§3). Star-graph carries H1; H2 carries the paper. |
| AOP (2606.00295) already tested OOD | High | **Check in week 1.** If so, narrow to the PE study + canvas analysis. |
| PE spread negligible | Medium | Publish as a negative result — "bidirectional attention is insensitive to PE choice" contradicts a natural prediction and is genuinely interesting. |
| No component beats baseline (G4 fails) | High | Pivot to the Tier-1 measurement paper. **Decide now you are willing to write it** — deciding under pressure in week 8 produces a worse paper. |
| Reviewers demand scale beyond 6M | Medium | Preempt: MGDM's own headline is 6M diffusion > 303M AR. Add the scaling check if an AAC allocation lands. |

---

## 11. Week-1 checklist

1. **Ask the professor to apply for an AAC allocation** (50–550 kSU). Single
   highest-leverage action in this document.
2. **Get sign-off on spending ~10% of the class pool** — 67 people share it.
3. **Read arXiv:2606.00295's experiments section.** Confirm it does not test
   generalization to harder-than-trained instances. Load-bearing assumption.
4. **Agree the tier structure with the professor.** Show them §7's gates. If
   they will only accept a Tier-3 "we beat SOTA" result, you need to know in
   week 1 — it changes the risk calculus and may argue for a safer task.
5. Confirm the ARR 9-month gap does not block TACL submission.
6. Run Gate 0 and replace every estimated SU number in §1 and §7.

---

## 12. References

**Core:** MGDM — Ye et al., ICLR 2025, arXiv:2410.14157,
[HKUNLP/diffusion-vs-ar](https://github.com/HKUNLP/diffusion-vs-ar) (Apache-2.0) ·
MDLM — Sahoo et al., NeurIPS 2024 ·
BD3-LM — Arriola et al., ICLR 2025 · D3PM — Austin et al., NeurIPS 2021

**Order (cite, don't re-claim):** Adaptive Order Policies — arXiv:2606.00295 ·
Kim et al., ICML 2025, arXiv:2502.06768 · arXiv:2512.09106 · arXiv:2510.04525 ·
arXiv:2511.19152 · arXiv:2509.01025

**AR failure:** Bachmann & Nagarajan, ICML 2024, arXiv:2403.06963

**PE / length generalization:** Kazemnejad et al., arXiv:2305.19466 (the causal
reference point) · ALiBi, ICLR 2022 · RoPE, arXiv:2104.09864 · Randomized PE,
ACL 2023 · MDLMPE, arXiv:2608.03769

**Data:** Stream-of-Search, arXiv:2404.03683 ·
[Next-Token-Failures](https://github.com/gregorbachmann/Next-Token-Failures) ·
3-SAT generated by the MGDM repo
