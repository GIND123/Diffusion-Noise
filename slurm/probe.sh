#!/bin/bash
#SBATCH -J probe
#SBATCH -t 04:00:00
#SBATCH -c 4
#SBATCH --mem=24g
#SBATCH --gpus=a100_1g.5gb:1
#SBATCH -p gpu-a100_1g.5gb
#SBATCH -o logs/probe-%A_%a.out
#SBATCH --array=0-5

source /etc/profile
module load pytorch/2.0.1
SCR=/scratch/zt1/project/msml612/user/$USER
cd $SCR/star/src

# task mode lr steps extra
CFG=(
  "sort diff 1e-4 4000  --k 8"            # 0 pipeline probe: order-insensitive
  "sort ar   1e-4 4000  --k 8"            # 1
  "sort diff 3e-4 4000  --k 8"            # 2
  "sort ar   3e-4 4000  --k 8"            # 3
  "star diff 1e-4 20000 --train_path 3"   # 4 long run: does star-graph converge?
  "star ar   1e-4 20000 --train_path 3"   # 5
)
# `read` already puts the remainder of the line into EXTRA; do not re-parse it.
read -r TASK MODE LR STEPS EXTRA <<< "${CFG[$SLURM_ARRAY_TASK_ID]}"
echo "=== probe $TASK $MODE lr=$LR steps=$STEPS $EXTRA ==="

python train.py --task $TASK --mode $MODE --pe ape --seed 0 --lr $LR \
  --steps $STEPS --n_train 200000 --n_eval 200 $EXTRA \
  --d 384 --layers 6 --heads 6 --bs 64 --accum 4 \
  --eval_every $((STEPS/4)) --T 24 \
  --out $SCR/star/runs/probe-$TASK-$MODE-lr$LR
