#!/bin/bash
# Launch the 90-run grid on whichever hardware is free.
#
#   ./launch.sh cpu     CPU nodes, 356 of them, usually schedulable immediately
#   ./launch.sh slice   1/7 A100 (5 GB), 7 SU/h, one node with 28 slices
#   ./launch.sh a100    full A100 (40 GB), 48 SU/h, 76 GPUs but heavily queued
#
# What works and what does not, verified 2026-09-11:
#   cpu / slice / a100  - fine. Zen3 nodes, and the default pytorch module is
#                         built for zen2, so it loads.
#   gpu-h100            - BROKEN. Hardware is free, but the cluster's PyTorch is
#                         2.0.1, which supports up to sm_86; H100 is sm_90.
#                         Needs a newer PyTorch, which pip 23.0 cannot install
#                         (wheel-name normalisation bug).
#   gpu-v100            - BROKEN. Idle and cheap (12 SU/h), but no pytorch module
#                         builds for that node's microarchitecture; both the
#                         zen2 and icelake builds abort on it.
#
# The training code itself is device-agnostic: it picks CUDA when present and
# CPU otherwise, so switching hardware needs no code change at all.
set -euo pipefail
TARGET="${1:-cpu}"
SCR=/scratch/zt1/project/msml612/user/$USER
cd "$SCR/star"

case "$TARGET" in
  h100)
    # Needs the locally built env: the cluster module is PyTorch 2.0.1, which
    # predates H100 (sm_90). hf_env has torch 2.5.1+cu121, verified on-device.
    HW="#SBATCH -p gpu-h100
#SBATCH --gpus=h100:1
#SBATCH -c 8
#SBATCH --mem=48g
#SBATCH -t 01:00:00"
    PYSETUP="module load python/3.10.10
export PYTHONPATH=
PY=\$SCR/hf_env/bin/python"
    THREADS=""
    BATCH="--bs 256 --accum 1"; CONC=12 ;;
  cpu)
    HW="#SBATCH -p standard
#SBATCH -c 64
#SBATCH --mem=48g
#SBATCH -t 06:00:00"
    PYSETUP="module load pytorch/2.0.1
PY=python"
    THREADS="export OMP_NUM_THREADS=64 MKL_NUM_THREADS=64"
    BATCH="--bs 256 --accum 1"; CONC=45 ;;
  slice)
    HW="#SBATCH -p gpu-a100_1g.5gb
#SBATCH --gpus=a100_1g.5gb:1
#SBATCH -c 4
#SBATCH --mem=24g
#SBATCH -t 01:30:00"
    PYSETUP="module load pytorch/2.0.1
PY=python"
    THREADS=""
    BATCH="--bs 64 --accum 4"   # 5 GB card will OOM above this
    CONC=28 ;;
  a100)
    HW="#SBATCH -p gpu-a100
#SBATCH --gpus=a100:1
#SBATCH -c 8
#SBATCH --mem=48g
#SBATCH -t 01:00:00"
    PYSETUP="module load pytorch/2.0.1
PY=python"
    THREADS=""
    BATCH="--bs 256 --accum 1"; CONC=20 ;;
  *) echo "usage: $0 {h100|cpu|slice|a100}" >&2; exit 1 ;;
esac

# python, not sed: the replacements are multi-line and contain pipes
HW="$HW" PYSETUP="$PYSETUP" THREADS="$THREADS" BATCH="$BATCH" CONC="$CONC" \
python3 - "grid_template.sh" "grid_$TARGET.sh" <<'EOF'
import os, sys
t = open(sys.argv[1]).read()
for k in ("HW", "PYSETUP", "THREADS", "BATCH", "CONC"):
    t = t.replace(f"__{k}__", os.environ.get(k, ""))
open(sys.argv[2], "w").write(t)
EOF

J=$(sbatch --parsable "grid_$TARGET.sh")
echo "grid on $TARGET: $J"
sbatch --dependency=afterany:"$J" finalize.sh
squeue -j "$J" -h -o "%T" | sort | uniq -c
