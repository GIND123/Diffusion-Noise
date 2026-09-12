#!/bin/bash
# Grid 4 — the comparisons a top-tier reviewer will demand.
#   M  0-23  SOTA protocol : train <=20 digits, test to 100, 4 methods x 2 arch
#   N 24-35  mechanism     : one-shot vs iterative denoising (the open question)
#   O 36-53  base          : is it place value, or decimal specifically?
#   P 54-65  per-instance  : bootstrap over test items, not just seeds
#SBATCH -J grid4
#SBATCH -p gpu-h100
#SBATCH --gpus=h100:1
#SBATCH -c 8
#SBATCH --mem=64g
#SBATCH -t 06:00:00
#SBATCH -o logs/grid4-%A_%a.out
#SBATCH --array=0-65%12

source /etc/profile
SCR=/scratch/zt1/project/msml612/user/$USER
module load python/3.10.10
export PYTHONPATH=
PY=$SCR/hf_env/bin/python
cd $SCR/star/src

I=$SLURM_ARRAY_TASK_ID
SEEDS=(0 1 2)

if [ $I -lt 24 ]; then                      # M: published protocol, train<=20 -> 100
  J=$I; METH=$((J / 6)); MODE=$(( (J % 6) / 3 )); S=${SEEDS[$((J % 3))]}
  MODES=(ar diff); M=${MODES[$MODE]}
  case $METH in
    0) MA="--coupled 0 --segments 0";                 MN="sequential" ;;
    1) MA="--coupled 1 --segments 1";                 MN="ours" ;;
    2) MA="--coupled 1 --segments 1 --abacus 1";      MN="abacus" ;;
    3) MA="--coupled 0 --segments 0 --randpos 1";     MN="randpos" ;;
  esac
  NAME="M-long-$MN-$M-s$S"
  ARGS="--op add --ladder long --digits 20 --mode $M --seed $S $MA \
        --steps 25000 --n_eval 100 --d 384 --layers 6 --heads 6 --bs 128 --accum 2"

elif [ $I -lt 36 ]; then                    # N: mechanism - refinement vs bidirectionality
  J=$((I - 24)); FT=$((J / 6)); TT=$(( (J % 6) / 3 )); S=${SEEDS[$((J % 3))]}
  FTS=(0.0 1.0); TS=(1 16)
  NAME="N-mech-ft${FTS[$FT]}-T${TS[$TT]}-s$S"
  ARGS="--op add --digits 5 --mode diff --seed $S --coupled 1 --segments 1 \
        --fixed_t ${FTS[$FT]} --T ${TS[$TT]} --nfe_sweep 1 \
        --steps 15000 --n_eval 200 --d 384 --layers 6 --heads 6 --bs 256 --accum 1"

elif [ $I -lt 54 ]; then                    # O: numeric base
  J=$((I - 36)); B=$((J / 6)); MODE=$(( (J % 6) / 3 )); S=${SEEDS[$((J % 3))]}
  BASES=(2 10 16); MODES=(ar diff); M=${MODES[$MODE]}
  NAME="O-base${BASES[$B]}-$M-s$S"
  ARGS="--op add --digits 5 --base ${BASES[$B]} --mode $M --seed $S --coupled 1 --segments 1 \
        --steps 15000 --n_eval 200 --d 384 --layers 6 --heads 6 --bs 256 --accum 1"

else                                         # P: per-instance flags for bootstrap
  J=$((I - 54)); COUP=$((J / 6)); MODE=$(( (J % 6) / 3 )); S=${SEEDS[$((J % 3))]}
  MODES=(ar diff); M=${MODES[$MODE]}
  NAME="P-inst-c$COUP-$M-s$S"
  ARGS="--op add --digits 5 --mode $M --seed $S --coupled $COUP --segments 1 --per_instance 1 \
        --steps 15000 --n_eval 500 --d 384 --layers 6 --heads 6 --bs 256 --accum 1"
fi

echo "=== $NAME ==="
$PY train_add.py $ARGS --lr 1e-4 --pe alibi --eval_every 999999 --out $SCR/star/runs/$NAME
