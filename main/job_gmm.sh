#!/bin/bash
#SBATCH --job-name=UnbiasedCNF
#SBATCH -e reports/errors_%j
#SBATCH -o reports/output_%j
#SBATCH --gpus-per-node=1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=01:00:00
#SBATCH --chdir=/home/u6dn/s2601026.u6dn/annealing-work

# -----------------------------
# Mode
# -----------------------------
MODE=train

# -----------------------------
# Mode=eval
# -----------------------------
EVAL_PATH=""
NUM_BOOTSTRAP=2000
DT_EVAL=0.1
EPS_EVAL=0.05
NUM_NOISE_EVAL=10
TIMESTAMP_EVAL=""
INTEGRATOR_EVAL=""

# -----------------------------
# Theory
# -----------------------------
N=10

# -----------------------------
# Network
# -----------------------------
NETWORK=mlpgmm
HIDDEN=(64 64)

# -----------------------------
# Integrator
# -----------------------------
INTEGRATOR=unbiasv2
BS=256
DT=0.1
EPS=0.05
NUM_NOISE=10
NUM_CHECKPOINT=20

# -----------------------------
# Training
# -----------------------------
LR=5e-3
LR_MIN=5e-4
T_MAX_SCHEDULER=400
STEPS=1000

# -----------------------------
# Weight & Biases
# -----------------------------
USE_WANDB=1
WANDB_PROJECT=cnf-explore

module load cray-python
source .cnf/bin/activate

CMD=(
  main_gmm.py
  --mode "$MODE"
  --eval-path "$EVAL_PATH"
  --num-bootstrap "$NUM_BOOTSTRAP"
  --dt-eval "$DT_EVAL"
  --eps-eval "$EPS_EVAL"
  --num-noise-eval "$NUM_NOISE_EVAL"
  --timestamp-eval "$TIMESTAMP_EVAL"
  --integrator-eval "$INTEGRATOR_EVAL"
  --n "$N"
  --network "$NETWORK"
  --hidden "${HIDDEN[@]}"
  --integrator "$INTEGRATOR"
  --bs "$BS"
  --dt "$DT"
  --eps "$EPS"
  --num-noise "$NUM_NOISE"
  --num-checkpoint "$NUM_CHECKPOINT"
  --lr "$LR=5e-3"
  --lr-min "$LR_MIN"
  --t-max-scheduler "$T_MAX_SCHEDULER"
  --steps "$STEPS"
  --wandb-project "$WANDB_PROJECT"
)

if [[ "$USE_WANDB" == "1" ]]; then
  CMD+=(--use-wandb)
fi

echo "Launching Neural PT with:"
printf ' %q' "${CMD[@]}"
echo

srun "${CMD[@]}"

