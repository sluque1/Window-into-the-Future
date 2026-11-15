#!/bin/bash
# Method 1 (gradient attribution) for one run, both of its checkpoints.
#
#   sbatch -J m1_A analysis/modality_importance/m1_sbatch.sh A
#   sbatch -J m1_B analysis/modality_importance/m1_sbatch.sh B
#
# One job per run, because the test dataset is built ONCE and shared by that
# run's best and final checkpoints -- same split, same preprocessing, same
# episode order for both sets of weights, which is what makes best-vs-final a
# one-variable comparison. Splitting by checkpoint instead would build the same
# dataset twice and risk two different builds.
#
# Env overrides:
#   N_EPISODES   0 = the whole held-out split (default)
#   BATCH_SIZE   default 8
#   CONTROLS     "weights labels" to also run the two randomisation controls
#   IG           true to also run Integrated Gradients (default true)
#   IG_EPISODES  default 128
#   IG_STEPS     default 32

# SLURM does not expand environment variables in #SBATCH lines, so there is no
# portable way to template this. Pass your account on the command line instead --
#   sbatch --account=<acct> ...   (submit_train.sh forwards $SLURM_ACCOUNT for you)
# -- or uncomment and hard-code it:
##SBATCH --account=<your-slurm-account>
#SBATCH --partition=medium_gpuh200
#SBATCH --gpus=1
#SBATCH --mem=96G
#SBATCH --cpus-per-task=4
#SBATCH --output=analysis/modality_importance/out/slurm/%x_%j.out
#SBATCH --error=analysis/modality_importance/out/slurm/%x_%j.err
#SBATCH --time=03:00:00

set -euo pipefail

# --- CPU-time limit ---------------------------------------------------------
# The cluster has PropagateResourceLimits=ALL and the login shell carries
# `ulimit -t 600`, so without this every job inherits a 600-SECOND CPU-time
# budget no matter what --time says. That is what killed jobs 639899/639900 at
# 8:35 and 6:19 wall clock with "CPU time limit exceeded". Submit with
# --propagate=NONE as well; this raises the soft limit to the hard ceiling
# (1800s) if the limit is inherited anyway.
ulimit -t 1800 2>/dev/null || true
echo "CPU-time limit: soft=$(ulimit -St) hard=$(ulimit -Ht)"

# CPU time accrues PER THREAD, so unbounded intra-op threading burns the budget
# several times faster than wall clock. The GPU does the arithmetic that matters
# here; the CPU only collates batches.
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

RUN="${1:-${RUN:-}}"
if [[ -z "$RUN" ]]; then
    echo "usage: sbatch analysis/modality_importance/m1_sbatch.sh {A|B}" >&2
    exit 2
fi

N_EPISODES="${N_EPISODES:-0}"
BATCH_SIZE="${BATCH_SIZE:-8}"
CONTROLS="${CONTROLS:-weights labels}"
CONTROL_N="${CONTROL_N:-1000}"
IG="${IG:-true}"
IG_EPISODES="${IG_EPISODES:-128}"
IG_STEPS="${IG_STEPS:-32}"
SEED="${SEED:-0}"

echo "Job started: $(date)"
echo "Node: $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

source "${CONDA_BASE:-$HOME/miniconda3}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV:-egovlpv2-pip}"
cd "$(dirname "$(readlink -f "$0")")/../.."
mkdir -p analysis/modality_importance/out/slurm

echo "RUN=$RUN N_EPISODES=$N_EPISODES BATCH_SIZE=$BATCH_SIZE SEED=$SEED"
echo "CONTROLS='$CONTROLS' CONTROL_N=$CONTROL_N IG=$IG IG_EPISODES=$IG_EPISODES IG_STEPS=$IG_STEPS"

IG_FLAG=""
[[ "$IG" == "true" ]] && IG_FLAG="--ig --ig_episodes $IG_EPISODES --ig_steps $IG_STEPS"

# The precision reconciliation now runs INSIDE m1_gradients.py, on the dataset it
# already built -- a separate invocation would spend another ~5 minutes rebuilding
# it. The runs trained and evaluated with bf16=True (training_args.bin), so
# predictions.pkl holds bfloat16-precision logits while this analysis uses float32.
echo ""

# Main pass AND both controls in ONE invocation. The dataset build is ~5 minutes
# and dominates a short job, so it happens once and every pass reuses it -- which
# also guarantees the controls run on byte-identical episodes to the main pass,
# making the difference attributable to the intervention alone.
#
# The controls run on CONTROL_N episodes rather than the full split: they only
# have to show the modality ranking collapses under an untrained model or shuffled
# labels, and that does not need 5007 episodes.
python analysis/modality_importance/m1_gradients.py \
    --run "$RUN" --which both \
    --control "none,${CONTROLS// /,}" --control_n "$CONTROL_N" \
    --n_episodes "$N_EPISODES" --batch_size "$BATCH_SIZE" --seed "$SEED" --skip_existing \
    $IG_FLAG

echo ""
echo "Job finished: $(date)"
