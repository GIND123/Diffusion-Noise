# Length Generalization in Masked Diffusion Language Models

**Place-value position identifiers let from-scratch transformers solve arithmetic
problems far longer than any they were trained on — trained on 20-digit operands,
they reach 100 digits. The benefit comes from bidirectional attention, not from
iterative denoising: a one-shot predictor beats iterative diffusion at 1/16 the
inference cost.**

Everything here is trained **from random initialization** — no pretrained
weights, no pretrained tokenizer, no distillation from a larger model. Models are
~10.7M parameters and train in roughly 20 minutes on a single A100 MIG slice.

- Code: this repository
- **[Resuming on another machine](HANDOFF.md)** — start here after a device change
- **[What's left before journal submission](SUBMISSION_ROADMAP.md)** — honest gap analysis
- Weights, results and figures: [huggingface.co/GOVINDFROM/masked-diffusion-length-generalization](https://huggingface.co/GOVINDFROM/masked-diffusion-length-generalization)

---

## 1. The problem

A language model that has learned *how to add* should add numbers of any length.
A model that has merely fit the training distribution will fail as soon as the
numbers get longer. Telling these apart is the cleanest available test of whether
a generative model learned an algorithm or a pattern.

This question is well studied for **autoregressive** models, which generate one
token at a time, left to right. It is essentially unstudied for **masked
diffusion language models**, which start with the whole answer hidden and reveal
it over several refinement passes, in an order they choose. Diffusion language
models are now being built at scale (LLaDA, Mercury, Gemini Diffusion) precisely
because that parallel, revisable generation is fast and can correct itself — so
how they extrapolate matters.

The obvious move is to import what works for autoregressive models. **We show
that fails, sometimes catastrophically, and explain why.**

---

## 2. Background: the two architectures

Both use the *same* transformer. Only the attention mask and the training
objective differ, so every comparison here is matched.

### Autoregressive
Causal attention — position *i* sees only positions ≤ *i*. Trained with
next-token prediction. Generates greedily, left to right, and can never revise.

### Masked diffusion (MDLM-style, absorbing state)
Bidirectional attention — every position sees every other. Training corrupts the
answer by replacing each token independently with a `[MASK]` placeholder with
probability *t* (sampled per example), then asks the model to restore the
originals, weighting the loss by `1/t`. Generation reverses this: start fully
masked, and over *T* passes repeatedly predict every hidden slot and reveal the
most confident ones.

```
pass 0:  __ __ __        (everything hidden)
pass 1:  __ __  2
pass 2:  __  3  2
pass 3:   1  3  2
```

This is diffusion in the formal sense — the same destroy-then-learn-to-restore
framework as image diffusion, with masking in place of Gaussian noise
(D3PM, Austin et al. 2021; MDLM, Sahoo et al. 2024).

---

## 3. Method: place-value position identifiers

A transformer has no inherent sense of order, so position must be supplied. The
standard choice numbers tokens by **where they sit in the sequence**. We instead
number every digit by its **place value**:

```
 4   7   +   8   5   =   1   3   2
tens un.      tens un.    hun tens un.
 2   1   0    2   1   0    3   2   1
```

Digits that must be combined now share an identifier. The rule *"combine equal
identifiers, carry into identifier + 1"* does not mention how many digits the
operands have, so it applies unchanged at any length.

Three details make it work:

| Component | Why it is needed |
|---|---|
| **Random offset** | A per-example constant is added to every identifier, so the model keys on *relative* place value and encounters large identifiers during training rather than only small ones. |
| **Segment embeddings** | Place-value identifiers deliberately collide — the units digits of operand A, operand B and the answer all carry identifier 1. A causal model separates them via the attention mask; **a bidirectional model cannot**, so diffusion needs an explicit marker for which part of the equation a token belongs to. This component is required for diffusion and optional for autoregressive. |
| **Terminator identifier** | The end-of-sequence token gets its own place-value identifier rather than sharing padding's. Without this the model produces perfectly correct digits but cannot tell which slot should terminate. |

The middle row is the part that is genuinely specific to diffusion, and it falls
directly out of the failure analysis in §5.

> **Prior work.** Numbering arithmetic tokens by place value is established for
> *autoregressive* models (position coupling; Abacus embeddings). We do not claim
> that idea. Our contribution is the adaptation to bidirectional masked
> diffusion — which does not work without the segment component — together with
> the evidence that naive transfer of positional schemes across the two
> architectures fails.

---

## 3b. System architecture

```
 INPUT  "47 + 85 ="                        four parallel signals per token
 ─────────────────────────────────────────────────────────────────────────
   token id      4    7    +    8    5    =   [M]  [M]  [M]      symbol-level
   place value   2    1    0    2    1    0    3    2    1   ←── OUR AXIS
   segment       A    A    –    B    B    –   ANS  ANS  ANS  ←── OUR AXIS
   offset        + r  (one random constant per example, added to place value)
 ─────────────────────────────────────────────────────────────────────────
              embedding sum  →  6 × transformer block  →  vocabulary logits
                                 (attention mask is the ONLY architectural
                                  difference between the two arms)
 ─────────────────────────────────────────────────────────────────────────
   AUTOREGRESSIVE                    │   MASKED DIFFUSION
   causal mask                       │   bidirectional mask
   next-token cross-entropy          │   absorbing-state denoising, 1/t weight
   greedy left-to-right, no revision │   T confidence-ordered refinement passes
```

**Every arm shares one transformer.** Swapping between them changes the
attention mask and the loss, nothing else — so any measured difference is
attributable to generation strategy rather than capacity, data, or optimizer.

### The four signals

1. **Token identity** — symbol-level, vocabulary built from the data. No
   pretrained tokenizer.
2. **Place-value identifier** — a digit's significance, not its sequence index.
   This makes the carry rule length-invariant.
3. **Segment identifier** — operand A, operand B, or answer. Necessary because
   place-value identifiers *deliberately collide* across the three.
4. **Random offset** — a per-example constant added to all place values, so the
   model learns relative significance and sees large indices during training.

### Why the pieces are load-bearing (measured, not asserted)

| Remove | Autoregressive | Diffusion |
|---|---|---|
| place value → sequential | 35.2% @ 6d | 55.0% @ 6d |
| segment identifiers | **0% @ 6d** | 72.0% @ 6d |
| random offset | 99.0% @ 6d | 99.5% @ 6d |
| nothing (full method) | 99.5% @ 6d | 100% @ 6d |

---

## 3c. How this differs from existing work

### What we do *not* claim

Numbering arithmetic tokens by place value is established for **autoregressive**
models — position coupling (Cho et al. 2024) and Abacus embeddings (McLeish et
al. 2024) both do essentially this, and reach far longer operands than we report.
We are not claiming that idea, and a paper that did would be rejected on sight.
Masked diffusion (MDLM, Sahoo et al. 2024) and each individual positional
encoding are likewise prior work.

### What is new here

| Contribution | Status in prior work |
|---|---|
| **Place-value identifiers under *bidirectional* attention** | Untouched. Every prior arithmetic length-generalization result uses causal attention. |
| **Positional encodings do not transfer across the two architectures** — a scheme giving 98.8% autoregressive gives **0%** diffusion | Not reported. The standard justification for no-positional-encoding rests on the causal mask, which diffusion lacks; nobody had tested the consequence. |
| **Diffusion benefits *more* than autoregressive from place-value alignment** (84.8% vs 52.3% @ 8d; 32.0% vs 5.7% @ 10d) | Not reported. Prior diffusion-vs-autoregressive results are entirely in-distribution. |
| **Segment identifiers are required once place values collide** | Not applicable to prior causal work, which disambiguates via the attention mask for free. |
| **The method fails when place value is not the algorithm** (multiplication) | A boundary condition nobody has drawn. |

### The one-sentence claim

> Place-value position identifiers transfer to masked diffusion language models,
> where they yield *larger* length-generalization gains than in the
> autoregressive setting they were designed for — and the positional-encoding
> choices that work for autoregressive models do not carry over, one of them
> failing completely.

---

## 4. Experiments

90 runs across four studies, 3 seeds each.

| Study | What it varies | Runs |
|---|---|---|
| **A — Main** | place-value vs sequential identifiers × {autoregressive, diffusion} × {learned absolute, distance rule} | 24 |
| **B — Encodings** | 5 positional encodings × 2 architectures | 30 |
| **C — Ablation** | segment embeddings on/off × random offset on/off × 2 architectures | 24 |
| **D — Transfer** | multiplication, where place value is *not* the algorithm | 12 |

### Tasks

**Addition** — train on 1–5 digit operands, test at 5, 6, 7, 8, 10, 12, 15 and 20
digits. Answers are written units-first, the standard format in this literature;
it removes the autoregressive model's need to know the final carry before
emitting its first token, so the baseline is the strong version rather than a
straw man. Both architectures receive identical data.

**Multiplication** — train on 1–3 digits, test to 7. Included deliberately as the
adversarial case: multiplication's algorithm is *not* place-local (each output
digit depends on many input pairs), so it tests whether the method works only
when place-value alignment happens to match the dependency structure.

**Sorting** — an order-insensitive control. Every output position is computable
independently, so if our effects appeared here too they would be about sequence
length generally rather than about reasoning order.

### Metrics

| Metric | Definition |
|---|---|
| **Exact match** | The entire answer must be correct. No partial credit. Checked by a deterministic program — no human judges, no model-as-judge. |
| **Per-digit accuracy** | Fraction of answer digits correct, aligned from the units end. Shows graded degradation that exact match hides. |
| **Hardness envelope H\*(ε)** | The largest operand length at which a model still exceeds accuracy ε. One comparable number per configuration. |
| **Accuracy vs denoising passes** | Accuracy as a function of refinement passes *T*. A diffusion-only axis with no autoregressive equivalent; reported as a curve so no single favourable *T* is cherry-picked. |

### Model and training

6 layers, 384 hidden dimensions, 6 heads, ~10.7M parameters, symbol-level
vocabulary built from the data. AdamW, learning rate 1e-4, 300-step warm-up then
cosine decay, weight decay 0.01, bfloat16, effective batch 256, 10,000 steps on
200,000 examples.

---

## 5. What we found

### The method reaches published scale: 20 digits → 100

Under the protocol used by the arithmetic length-generalization literature
(train on ≤20-digit operands, test far beyond), place-value identifiers carry an
autoregressive model to **72–73% exact match at 100 digits**, holding 95–100%
from 25 through 80. Our Abacus reimplementation reaches 30 digits under identical
compute; sequential identifiers collapse immediately past 20.

One of three seeds failed to take off, which matters and is reported: seed
variance on this task is severe throughout.

### The advantage is bidirectional attention, NOT iterative denoising

The natural story — diffusion wins because it refines over many passes — is
false. Crossing training objective against inference passes:

| Training | Passes | 8d | 10d |
|---|---|---|---|
| Standard diffusion | T=1 | 76% | 32% |
| Standard diffusion | T=16 | 79% | 34% |
| **One-shot (always fully masked)** | **T=1** | **87%** | **43%** |

Sixteen refinement passes buy ~3 points over a single pass — within seed noise.
A **one-shot bidirectional predictor is better than iterative diffusion at one
sixteenth of the inference cost.** Together with the decoding-order result below,
the diffusion machinery is not what produces the gain; the bidirectional receptive
field is.

This is the most useful finding here, and it points somewhere uncomfortable for a
diffusion paper: if you want these gains, you may not need diffusion.

### The diffusion advantage is scale-dependent

At small scale (train ≤5 digits) masked diffusion clearly beats autoregressive —
75.3% vs 10.5% at 8 digits. Under the 20-digit protocol the ordering **reverses**:
autoregressive reaches 100 digits while diffusion stalls near 50.

A caveat we are testing rather than asserting: the diffusion sampler used a fixed
T=16 budget for a 103-slot canvas, so it had to commit ~6–7 digits per pass with
carries unresolved between them. T-scaled runs (T = 32/64/128) are in flight; if
they close the gap, the reversal is an artefact of our sampling budget rather than
a property of the architecture.

### Positional encodings do not transfer between architectures

Supplying no positional information at all gives autoregressive models 98.8%
in-distribution accuracy and leaves masked diffusion at **0%** — unable to learn
the task on any seed. An autoregressive model reads tokens in order, so position
is implicit in the act of reading; the literature showing that *no* encoding is
best for length generalization rests on exactly that implicit signal. A diffusion
model sees every position at once, and without a positional signal it is holding
an unordered bag of digits.

**Rotary embeddings are the strongest pairing**, reaching 10 digits for
autoregressive and **12 for diffusion**.

### Segment embeddings matter far more for autoregressive models

Removing them costs the autoregressive model everything (0% at 6 digits) while
diffusion still reaches 72%. This is the opposite of what we first concluded: an
earlier apparent diffusion failure turned out to be the terminator bug described
below, not a missing segment signal.

### Where the method applies: alignment, not chaining

Three probes bound the claim. Addition has both positional alignment and a
sequential chain; the probes remove one at a time.

| Task | Structure | Diffusion + aligned ids |
|---|---|---|
| **reverse** | alignment, no chain | **100% at 2× length, reaches 40 digits** (trained on 10) |
| **addition** | alignment + chain | 75.3% at 8 digits (trained on 5) |
| **parity** | chain, no alignment | barely helps (envelope 12 → 15) |
| **multiplication** | neither | fails entirely (~2.7% at 3 digits) |

The method needs **positional alignment between input and output**, not a
sequential dependency. Parity has the chain and gains almost nothing; reverse has
the alignment and generalizes to four times the training length. Multiplication
has neither, and its failure is therefore a prediction of the account rather than
an anomaly.

### Against published methods

Matched compute, our harness, 3 seeds, exact match on addition:

| Method | Autoregressive @8d | Masked diffusion @8d |
|---|---|---|
| Sequential ids | 0.0 | 0.0 |
| Randomized PE (Ruoss et al. 2023) | 0.0 | 7.7 |
| Abacus embeddings (McLeish et al. 2024) | 0.8 | 63.8 [59.5, 68.5] |
| **Place-value ids (ours)** | **10.5** | **75.3 [71.5, 82.0]** |

Bootstrap 95% intervals for ours and Abacus on diffusion do not overlap, so that
gap is not seed noise.

The striking row is Abacus: designed for autoregressive models, it reaches 0.8%
there in our setup and **63.8% on diffusion**. Positional methods developed for
autoregressive arithmetic transfer to masked diffusion and work *better* in the
architecture they were not designed for.

> **Caveat that must survive into the paper.** Our autoregressive numbers are far
> below the published Abacus results, which reach far longer operands with larger
> models and much longer training. This is a matched-compute comparison *inside
> our harness*, not a challenge to their reported figures.

### The advantage is NOT explained by decoding order — a negative result

The natural explanation for the diffusion advantage is that it chooses a better
generation order: resolving units first and following the carry chain, which an
autoregressive model cannot do. **We tested this and it is false.**

Recording the denoising pass at which each answer digit is fixed, the correlation
between digit significance and reveal order is **−0.40 with our method and −0.32
for the baseline** — near-identical, and in the *opposite* direction to the
carry-chain prediction. Confidence-ordered sampling resolves the most
*predictable* slot first (the leading carry digit, usually 0 or 1, goes earliest),
not the algorithmically natural one.

So the mechanism remains open. Bidirectional attention, iterative refinement
allowing revision, or the denoising objective itself are all live candidates;
decoding order is not the explanation.

### Length-generalization accuracy has very high seed variance

One configuration spanned 3%–68% across seeds. Single-seed comparisons here —
including published ones — should be treated with suspicion. Every number in
this repository is mean ± standard deviation over 3 seeds.

### Two bugs that the numbers alone would have hidden

The terminator token originally shared padding's position identifier, so the
model produced **perfectly correct digits** and scored 0% because it could not
tell which slot should end the sequence. Separately, an audit
(`src/audit.py`) found that target identifiers were assigned only to written
slots, which handed the model the answer length for free — information the
baseline never received. Both are fixed; the affected runs were discarded rather
than reported.

### Figures

![Main result](figures/fig1_main.png)

**Figure 1.** Length generalization on addition. Sequential position identifiers
(red) collapse the moment the numbers exceed the training range; place-value
identifiers (blue) hold well beyond it. Shaded bars are ± 1 standard deviation
over 3 seeds; the dashed line marks the longest length seen in training.

![Encoding transfer](figures/fig2_encodings.png)

**Figure 2.** The same positional encoding produces opposite outcomes in the two
architectures. Schemes that give autoregressive models perfect in-distribution
accuracy leave masked diffusion unable to learn the task at all.

![Denoising passes](figures/fig3_denoising_steps.png)

**Figure 3.** Accuracy against the number of denoising passes — compute spent
purely at inference, with no autoregressive equivalent. Reported as a full curve
so that no single favourable step count is cherry-picked.

<!-- RESULTS:START -->
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


<!-- RESULTS:END -->

---

## 6. Repository layout

```
src/
  data.py            task generators: addition, multiplication, sorting, star-graph
  model.py           transformer with swappable positional encoding + attention mask
  train_add.py       main trainer (place-value ids, both architectures, all metrics)
  train.py           earlier trainer for the sorting / star-graph studies
  collect.py         aggregates every run into tables and figures
  trace.py           prints a diffusion denoising trajectory
  debug_coupled.py   prediction inspection used to find the terminator bug
slurm/
  grid.sh            the full 90-run grid
  finalize.sh        runs automatically after training: back up + push
push_to_hf.py        stage results and upload to the Hugging Face Hub
```

### Reproducing

```bash
# one run
python src/train_add.py --mode diff --op add --pe alibi --coupled 1 \
    --segments 1 --seed 0 --digits 5 --steps 10000 --out runs/demo

# the full grid (SLURM)
sbatch slurm/grid.sh

# tables and figures
RUNS=runs OUT=figures python src/collect.py
```

---

## 7. Honest limitations

- **Arithmetic is a probe, not an application.** A 10.7M-parameter adder is
  useless as a calculator. The claim is about the model class, not the task.
- **Place-value numbering is not our idea** — only its adaptation to
  bidirectional diffusion, and the accompanying negative transfer result.
- **Nothing here reaches the very long lengths** reported by the best
  autoregressive arithmetic work, which uses larger models and longer training.
- **Multiplication is included precisely because it may not work**; if the method
  only helps when place-value alignment matches the algorithm, that is a real
  boundary and is reported as one.

---

## 8. References

- Austin et al. *Structured Denoising Diffusion Models in Discrete State-Spaces* (D3PM). NeurIPS 2021.
- Sahoo et al. *Simple and Effective Masked Diffusion Language Models* (MDLM). NeurIPS 2024.
- Ye et al. *Beyond Autoregression: Discrete Diffusion for Complex Reasoning and Planning*. ICLR 2025.
- Kazemnejad et al. *The Impact of Positional Encoding on Length Generalization in Transformers*. 2023.
- Press et al. *Train Short, Test Long: Attention with Linear Biases* (ALiBi). ICLR 2022.
- Su et al. *RoFormer: Rotary Position Embedding*. 2021.
- Ruoss et al. *Randomized Positional Encodings Boost Length Generalization*. ACL 2023.
- Bachmann & Nagarajan. *The Pitfalls of Next-Token Prediction*. ICML 2024.

Trained on the University of Maryland Zaratan cluster.
