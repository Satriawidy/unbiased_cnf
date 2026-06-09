#!/bin/bash
#SBATCH --job-name=UnbiasedArrayCNF
#SBATCH -e reports/errors_%j
#SBATCH -o reports/output_%j
#SBATCH --gpus=1
$SBATCH --array=0-4
#SBATCH --time=01:00:00
#SBATCH --chdir=/home/u6dn/s2601026.u6dn/annealing-work

# ---------------------------------------------------------
# Array sweep: Integrator, dt, num_checkpoint and num_noise
# ---------------------------------------------------------
INT_VALUES=(unbiasv1 unbiasv2 exact hutch fp)
DTS_VALUES=(0.0025 0.0025 0.01 0.01 0.01)
CHK_VALUES=(100 100 25 25 25)
NOS_VALUES=(1 1 1 16 1)
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"

if (( TASK_ID < 0 || TASK_ID >= ${#INT_VALUES[@]} )); then
  echo "Invalid SLURM_ARRAY_TASK_ID=$TASK_ID for ${#INT_VALUES[@]} L values" >&2
  exit 1
fi

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
TIMESTAMP_EVAL=0
INTEGRATOR_EVAL=""

# -----------------------------
# Theory
# -----------------------------
L=8
M2=-4.0
LAM=6.008

# -----------------------------
# Network
# -----------------------------
NETWORK=phi4analytic
N_KERNEL=21
N_KERNEL_BOND=20
N_BASIS=20
N_BASIS_BOND=20

# -----------------------------
# Integrator
# -----------------------------
INTEGRATOR="${INT_VALUES[$TASK_ID]}"
BS=256
DT="${DTS_VALUES[$TASK_ID]}"
EPS=0.00125
NUM_NOISE="${NOS_VALUES[$TASK_ID]}"
NUM_CHECKPOINT="${CHK_VALUES[$TASK_ID]}"

# -----------------------------
# Training
# -----------------------------
LR=5e-3
LR_MIN=5e-4
T_MAX_SCHEDULER=40
STEPS=100

# -----------------------------
# Weight & Biases
# -----------------------------
USE_WANDB=1
WANDB_PROJECT=cnf-explore

module load cray-python
source .cnf/bin/activate

CMD=(
  python3 main/main_phi.py
  --mode "$MODE"
  --eval-path "$EVAL_PATH"
  --num-bootstrap "$NUM_BOOTSTRAP"
  --dt-eval "$DT_EVAL"
  --eps-eval "$EPS_EVAL"
  --num-noise-eval "$NUM_NOISE_EVAL"
  --timestamp-eval "$TIMESTAMP_EVAL"
  --integrator-eval "$INTEGRATOR_EVAL"
  --L "$L"
  --m2 "$M2"
  --lam "$LAM"
  --network "$NETWORK"
  --n-kernel "$N_KERNEL"
  --n-kernel-bond "$N_KERNEL_BOND"
  --n-basis "$N_BASIS"
  --n-basis-bond "$N_BASIS_BOND"
  --integrator "$INTEGRATOR"
  --bs "$BS"
  --dt "$DT"
  --eps "$EPS"
  --num-noise "$NUM_NOISE"
  --num-checkpoint "$NUM_CHECKPOINT"
  --lr "$LR"
  --lr-min "$LR_MIN"
  --t-max-scheduler "$T_MAX_SCHEDULER"
  --steps "$STEPS"
  --wandb-project "$WANDB_PROJECT"
)

if [[ "$USE_WANDB" == "1" ]]; then
  CMD+=(--use-wandb)
fi

echo "Array task ${TASK_ID}: INT=${INTEGRATOR}"
printf ' %q' "${CMD[@]}"
echo

srun "${CMD[@]}"

