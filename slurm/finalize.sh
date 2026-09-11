#!/bin/bash
# Runs automatically after the training array finishes (SLURM dependency), so it
# survives the laptop being closed, the network dropping, or the session dying.
#
#   1. regenerate figures + summary from every run
#   2. back up results to HOME (the only nightly-backed-up tier)
#   3. back up weights to SHELL (1 TB, no 90-day purge unlike scratch)
#   4. push everything to the Hugging Face Hub
#
#SBATCH -J finalize
#SBATCH -t 02:00:00
#SBATCH -c 4
#SBATCH --mem=16g
#SBATCH -p standard
#SBATCH -o logs/finalize-%j.out

source /etc/profile
module load pytorch/2.0.1

USER_SCR=/scratch/zt1/project/msml612/user/$USER
STAR=$USER_SCR/star
export PYTHONPATH=$USER_SCR/pylibs:$PYTHONPATH
# Compute nodes cannot reach HF's Xet CAS endpoint; force the plain LFS path.
export HF_HUB_DISABLE_XET=1

HOME_BK=$HOME/msml612-backup
SHELL_BK=/afs/shell.umd.edu/project/msml612/user-$USER/msml612-weights

echo "=== 1. collect + plot ==="
cd $STAR/src
RUNS=$STAR/runs OUT=$STAR/figures python collect.py || echo "collect failed (continuing)"

echo "=== 2. back up results + figures + code to HOME (backed up nightly) ==="
mkdir -p $HOME_BK/{results,figures,src}
cp -f $STAR/figures/* $HOME_BK/figures/ 2>/dev/null
cp -f $STAR/src/*.py $HOME_BK/src/ 2>/dev/null
for d in $STAR/runs/*/; do
  n=$(basename $d)
  [ -f "$d/result.json" ] && cp -f "$d/result.json" "$HOME_BK/results/$n.json"
done
echo "  results backed up: $(ls $HOME_BK/results | wc -l)"
du -sh $HOME_BK 2>/dev/null

echo "=== 3. back up weights to SHELL (not purged) ==="
if mkdir -p $SHELL_BK 2>/dev/null; then
  for d in $STAR/runs/*/; do
    n=$(basename $d)
    [ -f "$d/model.pt" ] && cp -f "$d/model.pt" "$SHELL_BK/$n.pt" 2>/dev/null
  done
  echo "  weights backed up: $(ls $SHELL_BK 2>/dev/null | wc -l)"
else
  echo "  SHELL not writable from this node - weights remain on scratch"
fi

echo "=== 4. push to Hugging Face ==="
python - <<'PY'
import os, json, glob, pathlib
from huggingface_hub import HfApi

tok = pathlib.Path(os.path.expanduser("~/.hf_token")).read_text().strip()
star = os.path.expanduser(f"/scratch/zt1/project/msml612/user/{os.environ['USER']}/star")
stage = pathlib.Path(os.path.expanduser("~/msml612-backup"))
repo = "GOVINDFROM/masked-diffusion-length-generalization"

summary = pathlib.Path(star, "figures", "summary.md")
table = summary.read_text() if summary.exists() else "_(pending)_"

(stage / "README.md").write_text(f"""---
license: apache-2.0
tags: [masked-diffusion, discrete-diffusion, length-generalization, positional-encoding, from-scratch]
---

# Length Generalization in Masked Diffusion Language Models

From-scratch masked diffusion language models (MDLM-style absorbing state) and
matched autoregressive baselines, trained on multi-digit addition and evaluated
on operand lengths never seen during training.

**No pretrained weights or tokenizers are used anywhere** - every model starts
from random initialization with a symbol-level vocabulary built from the data.

## Method: significance-aligned position ids

Tokens are numbered by place value rather than sequence index, so digits that
must be combined share an id:

```
 4   7   +   8   5   =   1   3   2
 2   1   0   2   1   0   3   2   1
```

"Combine equal ids, carry into id+1" does not depend on operand length, which is
what permits extrapolation. A random per-example offset makes the model key on
relative place value and exposes it to large ids during training.

## Findings

1. **Positional encodings do not transfer between architectures.** No positional
   information at all, and sinusoidal encoding, each give 100% in-distribution
   accuracy for autoregressive models and **0% for masked diffusion** - the
   diffusion model cannot learn the task at all (3 seeds each). Bidirectional
   attention has no implicit ordering to fall back on.
2. **Length-generalization accuracy has very high seed variance** (one
   configuration spanned 3%-68% across seeds), so single-seed comparisons in
   this setting are not trustworthy.

## Results

{table}

## Setup

6 layers, 384 hidden dimension, 6 heads, ~10.7M parameters. AdamW, learning rate
1e-4, cosine schedule, bfloat16, effective batch 256. Trained on 1-5 digit
operands; evaluated at 5, 6, 7, 8, 10, 12 digits. Exact-match accuracy.

Code: https://github.com/GIND123/MSML612
""")

api = HfApi(token=tok)
api.create_repo(repo, repo_type="model", private=True, exist_ok=True)
api.upload_folder(folder_path=str(stage), repo_id=repo, repo_type="model",
                  commit_message="Automated push: results, figures, code")
print("pushed:", f"https://huggingface.co/{repo}")
PY

echo "=== finalize complete ==="
