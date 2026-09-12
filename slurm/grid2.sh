#!/bin/bash
# Second grid: the comparisons a journal reviewer will ask for.
#   E  0-11   published baseline : randomized positional encodings (Ruoss 2023)
#   F 12-47   scaling            : 3 model sizes x 2 architectures x method/baseline
#   G 48-53   denoising-pass sweep (diffusion only; accuracy vs inference compute)
#   H 54-65   subtraction        : second place-local operation
#SBATCH -J grid2
#SBATCH -p gpu-h100
#SBATCH --gpus=h100:1
#SBATCH -c 8
#SBATCH --mem=48g
#SBATCH -t 02:00:00
#SBATCH -o logs/grid2-%A_%a.out
#SBATCH --array=0-65%12

source /etc/profile
SCR=/scratch/zt1/project/msml612/user/$USER
module load python/3.10.10
export PYTHONPATH=
PY=$SCR/hf_env/bin/python
cd $SCR/star/src

I=$SLURM_ARRAY_TASK_ID
SEEDS=(0 1 2)
BASE="--n_train 200000 --n_eval 200 --lr 1e-4 --bs 256 --accum 1 --T 16 --pe alibi"

if [ $I -lt 12 ]; then                                   # E: randomized PE
  J=$I; MODE=$((J / 6)); RP=$(( (J % 6) / 3 )); S=${SEEDS[$((J % 3))]}
  MODES=(ar diff); M=${MODES[$MODE]}
  NAME="E-randpos$RP-$M-s$S"
  ARGS="--op add --coupled 0 --randpos $RP --mode $M --seed $S --digits 5 --steps 15000 --d 384 --layers 6 --heads 6 --eval_every 15000"

elif [ $I -lt 48 ]; then                                 # F: scaling
  J=$((I - 12)); SZ=$((J / 12)); COUP=$(( (J % 12) / 6 )); MODE=$(( (J % 6) / 3 )); S=${SEEDS[$((J % 3))]}
  MODES=(ar diff); M=${MODES[$MODE]}
  DS=(384 512 768); LS=(6 8 12); HS=(6 8 12)
  NAME="F-size${DS[$SZ]}-c$COUP-$M-s$S"
  ARGS="--op add --coupled $COUP --segments 1 --mode $M --seed $S --digits 5 --steps 15000 --d ${DS[$SZ]} --layers ${LS[$SZ]} --heads ${HS[$SZ]} --eval_every 15000"

elif [ $I -lt 54 ]; then                                 # G: denoising passes
  J=$((I - 48)); COUP=$((J / 3)); S=${SEEDS[$((J % 3))]}
  NAME="G-nfe-c$COUP-diff-s$S"
  ARGS="--op add --coupled $COUP --segments 1 --mode diff --seed $S --digits 5 --steps 15000 --d 384 --layers 6 --heads 6 --eval_every 15000 --nfe_sweep 1"

else                                                      # H: subtraction
  J=$((I - 54)); COUP=$((J / 6)); MODE=$(( (J % 6) / 3 )); S=${SEEDS[$((J % 3))]}
  MODES=(ar diff); M=${MODES[$MODE]}
  NAME="H-sub-c$COUP-$M-s$S"
  ARGS="--op sub --coupled $COUP --segments 1 --mode $M --seed $S --digits 5 --steps 15000 --d 384 --layers 6 --heads 6 --eval_every 15000"
fi

echo "=== $NAME ==="
$PY train_add.py $ARGS $BASE --out $SCR/star/runs/$NAME
