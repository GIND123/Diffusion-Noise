#!/bin/bash
# Full experiment grid: 90 runs across four studies.
#   A  0-23   main result   : method vs baseline, both architectures, add
#   B 24-53   encoding sweep: 5 positional encodings x 2 architectures, add
#   C 54-77   ablation      : segments on/off x random offset on/off, add
#   D 78-89   transfer      : multiplication, where place value != the algorithm
#SBATCH -J grid
#SBATCH -t 04:00:00
#SBATCH -c 4
#SBATCH --mem=24g
#SBATCH --gpus=a100_1g.5gb:1
#SBATCH -p gpu-a100_1g.5gb
#SBATCH -o logs/grid-%A_%a.out
#SBATCH --array=0-89%28

source /etc/profile
module load pytorch/2.0.1
SCR=/scratch/zt1/project/msml612/user/$USER
cd $SCR/star/src

I=$SLURM_ARRAY_TASK_ID
SEEDS=(0 1 2)
COMMON="--steps 10000 --n_train 200000 --n_eval 500 --lr 1e-4 --d 384 --layers 6 --heads 6 --bs 128 --accum 2 --eval_every 10000 --T 16"

if [ $I -lt 24 ]; then                       # ---- A: main result
  J=$I; COUP=$((J / 12)); MODE=$(( (J % 12) / 6 )); PEI=$(( (J % 6) / 3 )); S=${SEEDS[$((J % 3))]}
  MODES=(ar diff); PES=(ape alibi)
  M=${MODES[$MODE]}; PE=${PES[$PEI]}
  NAME="A-main-c$COUP-$M-$PE-s$S"
  ARGS="--op add --coupled $COUP --segments 1 --mode $M --pe $PE --seed $S --digits 5"

elif [ $I -lt 54 ]; then                     # ---- B: encoding sweep
  J=$((I - 24)); MODE=$((J / 15)); PEI=$(( (J % 15) / 3 )); S=${SEEDS[$((J % 3))]}
  MODES=(ar diff); PES=(nope ape sin rope alibi)
  M=${MODES[$MODE]}; PE=${PES[$PEI]}
  NAME="B-pe-$M-$PE-s$S"
  ARGS="--op add --coupled 1 --segments 1 --mode $M --pe $PE --seed $S --digits 5"

elif [ $I -lt 78 ]; then                     # ---- C: component ablation
  J=$((I - 54)); SEG=$((J / 12)); OFFI=$(( (J % 12) / 6 )); MODE=$(( (J % 6) / 3 )); S=${SEEDS[$((J % 3))]}
  MODES=(ar diff); OFFS=(0 20)
  M=${MODES[$MODE]}; OFF=${OFFS[$OFFI]}
  NAME="C-abl-seg$SEG-off$OFF-$M-s$S"
  ARGS="--op add --coupled 1 --segments $SEG --max_offset $OFF --mode $M --pe alibi --seed $S --digits 5"

else                                          # ---- D: multiplication
  J=$((I - 78)); COUP=$((J / 6)); MODE=$(( (J % 6) / 3 )); S=${SEEDS[$((J % 3))]}
  MODES=(ar diff); M=${MODES[$MODE]}
  NAME="D-mul-c$COUP-$M-s$S"
  ARGS="--op mul --coupled $COUP --segments 1 --mode $M --pe alibi --seed $S --digits 3"
fi

echo "=== $NAME ==="
python train_add.py $ARGS $COMMON --out $SCR/star/runs/$NAME
