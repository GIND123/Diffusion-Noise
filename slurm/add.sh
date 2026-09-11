#!/bin/bash
#SBATCH -J add
#SBATCH -t 03:00:00
#SBATCH -c 4
#SBATCH --mem=24g
#SBATCH --gpus=a100_1g.5gb:1
#SBATCH -p gpu-a100_1g.5gb
#SBATCH -o logs/add-%A_%a.out
#SBATCH --array=0-9

source /etc/profile
module load pytorch/2.0.1
SCR=/scratch/zt1/project/msml612/user/$USER
cd $SCR/star/src

MODES=(ar diff)
PES=(nope ape sin rope alibi)
MODE=${MODES[$((SLURM_ARRAY_TASK_ID / 5))]}
PE=${PES[$((SLURM_ARRAY_TASK_ID % 5))]}
echo "=== add $MODE $PE ==="

python train.py --task add --mode $MODE --pe $PE --seed 0 --lr 1e-4 \
  --digits 5 --steps 8000 --n_train 200000 --n_eval 500 \
  --d 384 --layers 6 --heads 6 --bs 128 --accum 2 \
  --eval_every 2000 --T 16 \
  --out $SCR/star/runs/add-$MODE-$PE
