#!/bin/bash
# Third grid: the published-baseline comparison and the non-arithmetic probes.
#   I  0-11   Abacus embeddings (McLeish et al. 2024) vs our place-value ids
#   J 12-35   parity  : sequential chain, NO arithmetic, NO place value
#   K 36-59   reverse : positional alignment, NO chain
#   L 60-71   5 seeds for the headline row (statistical power)
#SBATCH -J grid3
#SBATCH -p gpu-h100
#SBATCH --gpus=h100:1
#SBATCH -c 8
#SBATCH --mem=48g
#SBATCH -t 02:00:00
#SBATCH -o logs/grid3-%A_%a.out
#SBATCH --array=0-71%12

source /etc/profile
SCR=/scratch/zt1/project/msml612/user/$USER
module load python/3.10.10
export PYTHONPATH=
PY=$SCR/hf_env/bin/python
cd $SCR/star/src

I=$SLURM_ARRAY_TASK_ID
SEEDS=(0 1 2)
BASE="--n_train 200000 --n_eval 200 --lr 1e-4 --bs 256 --accum 1 --T 16 --pe alibi --steps 15000 --d 384 --layers 6 --heads 6 --eval_every 15000"

if [ $I -lt 12 ]; then                                   # I: Abacus head-to-head
  J=$I; MODE=$((J / 6)); AB=$(( (J % 6) / 3 )); S=${SEEDS[$((J % 3))]}
  MODES=(ar diff); M=${MODES[$MODE]}
  NAME="I-abacus$AB-$M-s$S"
  ARGS="--op add --coupled 1 --segments 1 --abacus $AB --mode $M --seed $S --digits 5"

elif [ $I -lt 36 ]; then                                 # J: parity (chain, no place value)
  J=$((I - 12)); COUP=$((J / 12)); MODE=$(( (J % 12) / 6 )); SEG=$(( (J % 6) / 3 )); S=${SEEDS[$((J % 3))]}
  MODES=(ar diff); M=${MODES[$MODE]}
  NAME="J-parity-c$COUP-seg$SEG-$M-s$S"
  ARGS="--op parity --coupled $COUP --segments $SEG --mode $M --seed $S --digits 10"

elif [ $I -lt 60 ]; then                                 # K: reverse (alignment, no chain)
  J=$((I - 36)); COUP=$((J / 12)); MODE=$(( (J % 12) / 6 )); SEG=$(( (J % 6) / 3 )); S=${SEEDS[$((J % 3))]}
  MODES=(ar diff); M=${MODES[$MODE]}
  NAME="K-reverse-c$COUP-seg$SEG-$M-s$S"
  ARGS="--op reverse --coupled $COUP --segments $SEG --mode $M --seed $S --digits 10"

else                                                      # L: extra seeds, headline row
  J=$((I - 60)); COUP=$((J / 6)); MODE=$(( (J % 6) / 3 )); S=$(( (J % 3) + 3 ))
  MODES=(ar diff); M=${MODES[$MODE]}
  NAME="A-main-c$COUP-$M-alibi-s$S"
  ARGS="--op add --coupled $COUP --segments 1 --mode $M --seed $S --digits 5"
fi

echo "=== $NAME ==="
$PY train_add.py $ARGS $BASE --out $SCR/star/runs/$NAME
