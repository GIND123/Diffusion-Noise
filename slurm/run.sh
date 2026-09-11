#!/bin/bash
#SBATCH -J star
#SBATCH -t 04:00:00
#SBATCH -c 4
#SBATCH --mem=24g
#SBATCH --gpus=a100_1g.5gb:1
#SBATCH -p gpu-a100_1g.5gb
#SBATCH -o logs/%x-%A_%a.out
#SBATCH --array=0-13

source /etc/profile
module load pytorch/2.0.1

SCR=/scratch/zt1/project/msml612/user/$USER
cd $SCR/star/src

# index -> mode pe seed
CFG=(
  "ar   ape   0"   # 0  core: AR baseline
  "ar   ape   1"   # 1
  "ar   ape   2"   # 2
  "diff ape   0"   # 3  core: masked diffusion
  "diff ape   1"   # 4
  "diff ape   2"   # 5
  "ar   nope  0"   # 6  PE sweep (causal) - reproduces Kazemnejad setting
  "ar   sin   0"   # 7
  "ar   rope  0"   # 8
  "ar   alibi 0"   # 9
  "diff nope  0"   # 10 PE sweep (bidirectional) - the novel column
  "diff sin   0"   # 11
  "diff rope  0"   # 12
  "diff alibi 0"   # 13
)

read -r MODE PE SEED <<< "${CFG[$SLURM_ARRAY_TASK_ID]}"
NAME="${MODE}-${PE}-s${SEED}"
echo "=== $NAME on $(hostname) ==="

python train.py --mode $MODE --pe $PE --seed $SEED \
  --steps 9000 --n_train 200000 --n_eval 500 \
  --d 384 --layers 6 --heads 6 --bs 64 --accum 4 --lr 1e-3 \
  --eval_every 3000 --T 24 \
  --out $SCR/star/runs/$NAME
