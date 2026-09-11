#!/bin/bash
#SBATCH -J diag
#SBATCH -t 01:00:00
#SBATCH -c 4
#SBATCH --mem=24g
#SBATCH --gpus=a100_1g.5gb:1
#SBATCH -p gpu-a100_1g.5gb
#SBATCH -o logs/diag-%A_%a.out
#SBATCH --array=0-7

source /etc/profile
module load pytorch/2.0.1
SCR=/scratch/zt1/project/msml612/user/$USER
cd $SCR/star/src

# mode lr path  -- find a configuration that actually learns the interior nodes
CFG=(
  "diff 1e-4 5"   # 0  Bachmann & Nagarajan's specified LR
  "diff 3e-4 5"   # 1
  "diff 1e-4 3"   # 2  shorter path = fewer hops
  "diff 3e-4 3"   # 3
  "ar   1e-4 5"   # 4
  "ar   1e-4 3"   # 5
  "diff 1e-3 3"   # 6  original LR, easier task
  "ar   3e-4 3"   # 7
)
read -r MODE LR PLEN <<< "${CFG[$SLURM_ARRAY_TASK_ID]}"
echo "=== diag $MODE lr=$LR path=$PLEN ==="

python train.py --mode $MODE --pe ape --seed 0 --lr $LR --train_path $PLEN \
  --steps 3000 --n_train 100000 --n_eval 200 \
  --d 384 --layers 6 --heads 6 --bs 64 --accum 4 \
  --eval_every 1500 --T 24 \
  --out $SCR/star/runs/diag-$MODE-lr$LR-p$PLEN
