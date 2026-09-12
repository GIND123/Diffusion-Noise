# Handoff — resuming on a different machine

Everything needed to continue is in this repository or on the cluster. This file
is the entry point.

---

## 1. What exists, and where

| Thing | Location | Survives device change? |
|---|---|---|
| All code | this repo | ✅ GitHub |
| Cluster access notes (`Zaratan/README.md`) | kept outside git, moved manually | ⚠️ **copy by hand** |
| Connection script (`Zaratan/zaratan-run.sh`) | kept outside git, moved manually | ⚠️ **copy by hand** |
| Results, figures, checkpoints | [HF: GOVINDFROM/masked-diffusion-length-generalization](https://huggingface.co/GOVINDFROM/masked-diffusion-length-generalization) | ✅ |
| Raw run outputs (228 runs) | Zaratan `~/msml612-backup` and `$SCRATCH/star/runs` | ✅ home is backed up nightly |
| PyTorch env for H100 | Zaratan `$SCRATCH/hf_env` | ✅ lives on the cluster |
| **Credentials** | local `.env` only | ❌ **must be recreated — see below** |

`$SCRATCH` = `/scratch/zt1/project/msml612/user/govind02`

---

## 2. First 10 minutes on the new machine

Copy the whole `Documents/Zaratan/` folder across by hand (it holds `.env`,
`README.md` and `zaratan-run.sh`, none of which are in git), then:

```bash
cd ~/Documents/Zaratan
git clone https://github.com/GIND123/MSML612.git    # if not copied with the folder
chmod 600 .env && chmod +x zaratan-run.sh

# SSH alias (or use the full host directly)
cat >> ~/.ssh/config <<'EOF'
Host zaratan
    HostName login.zaratan.umd.edu
    User govind02
EOF

../zaratan-run.sh 'sbalance'     # approve the Duo push
```

The folder layout the scripts expect:

```
Documents/Zaratan/
  .env                 credentials (chmod 600, never committed)
  README.md            cluster access + allocation notes
  zaratan-run.sh       connection wrapper
  MSML612/             this git repo
```

`.env` must contain `ZARATAN_HOST`, `ZARATAN_USER`, `ZARATAN_PASSWORD`, `HF_TOKEN`.

> **Rotate the Hugging Face token.** The old one was pasted into a chat session,
> so treat it as compromised: revoke it at huggingface.co/settings/tokens and put
> the new one in `../.env` and in `~/.hf_token` on the cluster.

---

## 3. Running anything

```bash
cd $SCRATCH/star
./launch.sh h100      # fastest; needs $SCRATCH/hf_env (torch 2.5.1+cu121)
./launch.sh cpu       # always schedulable, slower
./launch.sh slice     # cheapest per run
./launch.sh a100      # works, but the queue is often days deep
```

**Hardware notes that cost a day to learn:**
- The cluster's `pytorch/2.0.1` module predates H100 (sm_90). H100 jobs must use
  `$SCRATCH/hf_env/bin/python` with `PYTHONPATH` cleared.
- `gpu-v100` is permanently idle but unusable: no PyTorch module builds for that
  node's microarchitecture.
- `gpu-a100` regularly has 150+ jobs queued; estimated starts of several days are
  normal.
- An H100 run is ~2 min and ~5 SU. A 90-run grid is under 1% of the allocation.

Results sync with:
```bash
python3 sync_results.py     # regenerates figures/tables, pushes to GitHub + HF
```

---

## 4. Where the science stands

**Headline (addition, trained ≤5 digits, 3–5 seeds, exact match at 8 digits):**

| Method | Autoregressive | Masked diffusion |
|---|---|---|
| Sequential ids | 0.0 | 0.0 |
| Randomized PE (Ruoss 2023) | 0.0 | 7.7 |
| Abacus (McLeish 2024) | 0.8 | 63.8 [59.5, 68.5] |
| **Place-value ids (ours)** | **10.5** | **75.3 [71.5, 82.0]** |

Plus: the method needs *positional alignment*, not sequential chaining (reverse
100% at 2× length; parity barely helps; multiplication fails). Positional
encodings do not transfer between architectures — no-position gives autoregressive
98.8% and diffusion **0%**, while randomized PE does the reverse.

**The honest weakness:** we cannot explain the effect. The obvious mechanism
(diffusion choosing a carry-friendly decoding order) was tested and **refuted** —
reveal order correlates −0.40 with digit significance under our method and −0.32
under the baseline. See `src/order_analysis.py`.

Full analysis: [`README.md`](README.md) · gaps: [`SUBMISSION_ROADMAP.md`](SUBMISSION_ROADMAP.md)

---

## 5. Staged but never run

These were written and copied to the cluster but **not smoke-tested**, because
the session was interrupted. Validate before spending compute — the `--base`
change touches the tokenizer, and a silent break there corrupts every run.

| Flag | Purpose |
|---|---|
| `--fixed_t 1.0` | **The mechanism experiment.** Trains a one-shot bidirectional denoiser with no iterative refinement, separating "bidirectional attention" from "multiple passes" as the explanation for the diffusion advantage. Highest scientific value of anything remaining. |
| `--base {2,10,16}` | Is the effect about place value in general, or decimal specifically? |
| `--per_instance 1` | Saves per-instance correctness so intervals can bootstrap over test items, not only seeds. |
| per-position accuracy | Already wired into `result.json`; enables failure analysis at 20 digits (per-digit accuracy is 68.9% — *which* digits break?). |
| `src/preflight.py` | Validation gate: data correctness, train/generation signal agreement, and a tiny-set overfit check per configuration. Written in response to two bugs that wasted full grids. Run it before any new grid. |

**Suggested first action on the new machine:**
```bash
../zaratan-run.sh 'cd $SCRATCH/star/src && $SCRATCH/hf_env/bin/python preflight.py'
```
then launch the `--fixed_t` mechanism ablation.

---

## 6. Bugs already found (do not reintroduce)

1. **`[EOS]` shared padding's position id** — model produced perfectly correct
   digits and scored 0%.
2. **Answer-length leak** — target ids were assigned only to written slots, so
   their count revealed the answer length. Found by `src/audit.py`.
3. **Randomized positions redrawn per decode step** — made a published baseline
   score 0% in-distribution.
4. **Abacus train/generation mismatch** — sequential positions in training,
   place-value at generation. Same symptom.

Bugs 3 and 4 would have meant publishing "two published methods fail completely,"
which is why `preflight.py` now exists. `src/audit.py` catches class 1–2.
