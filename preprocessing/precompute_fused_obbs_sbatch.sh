#!/bin/bash
# SLURM array job: precompute IoU-fused Boxer OBBs, ONE TASK PER PARTICIPANT.
#
# Writes boxer_obbs/<video_id>/fused_obbs_iou.pkl beside each video's CSV, which
# training then reads via FUSION=iou FUSED_CACHE=true (see train_sbatch.sh). Run
# this ONCE: every LOOCV fold reuses the same per-video files, and the cache is
# modality-independent (env and obb share _window_fused_boxes), so changing
# MODALITY never needs a rebuild.
#
# Usage:
#   ./precompute_fused_obbs_sbatch.sh                 # all 9 participants
#   PARTICIPANTS=P03,P08 ./precompute_fused_obbs_sbatch.sh
#   WORKERS=12 ./precompute_fused_obbs_sbatch.sh
#
# Why --workers: a participant is 2,255-4,641 windows at a measured 6.36 s/window,
# i.e. 4.0-8.2 h serially. medium_cpu caps at 6:10:00, so P01/P03/P08/P09 would be
# killed mid-run. The videos within a participant are independent (separate CSVs,
# separate outputs), so 8 workers brings the worst case (P03) to ~1 h.
#
#   participant  windows  serial    with 8 workers
#   P03            4641   8.20 h    ~1.03 h
#   P08            3804   6.72 h    ~0.84 h
#   P09            3787   6.69 h    ~0.84 h
#   P01            3665   6.47 h    ~0.81 h
#   P02/P04       ~2826   4.99 h    ~0.62 h
#   P06            2616   4.62 h    ~0.58 h
#   P05            2357   4.16 h    ~0.52 h
#   P07            2255   3.98 h    ~0.50 h
#
# Resumable at video granularity: a video whose cache exists is skipped, so a
# killed or timed-out task can simply be resubmitted. Writes are atomic (temp
# file + rename), so an interrupted task never leaves a truncated pickle.
# OVERWRITE=true forces a rebuild.

#SBATCH --job-name="fuse_obbs"
# SLURM does not expand environment variables in #SBATCH lines, so there is no
# portable way to template this. Pass your account on the command line instead --
#   sbatch --account=<acct> ...   (submit_train.sh forwards $SLURM_ACCOUNT for you)
# -- or uncomment and hard-code it:
##SBATCH --account=<your-slurm-account>
#SBATCH --partition=medium_cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=out/fuse_obbs_%A_%a.out
#SBATCH --error=err/fuse_obbs_%A_%a.err

set -euo pipefail

PARTICIPANTS="${PARTICIPANTS:-P01,P02,P03,P04,P05,P06,P07,P08,P09}"
WORKERS="${WORKERS:-8}"
SLIM="${SLIM:-false}"
OVERWRITE="${OVERWRITE:-false}"

IFS=',' read -r -a PID_ARRAY <<< "$PARTICIPANTS"
N_PIDS="${#PID_ARRAY[@]}"

# --- submit mode: re-invoke ourselves as an array, one task per participant ---
if [ -z "${SLURM_ARRAY_TASK_ID:-}" ]; then
    mkdir -p out err
    echo "Submitting ${N_PIDS} tasks (one per participant): ${PARTICIPANTS}"
    echo "  ${WORKERS} workers each, partition medium_cpu, 6h limit"
    # --propagate=NONE for the same reason submit_train.sh needs it: the login
    # node's RLIMIT_CPU (soft 10 min) is otherwise inherited and SIGXCPUs the
    # task minutes in. See the comment in submit_train.sh.
    sbatch --propagate=NONE --export=ALL \
           ${SLURM_ACCOUNT:+--account="$SLURM_ACCOUNT"} \
           ${PARTITION:+--partition="$PARTITION"} \
           --array="0-$((N_PIDS - 1))" \
           -J "fuse_obbs" "$0"
    exit 0
fi

# --- array task: one participant ---------------------------------------------
PID="${PID_ARRAY[$SLURM_ARRAY_TASK_ID]}"
echo "Task ${SLURM_ARRAY_TASK_ID}: participant ${PID} on $(hostname), $(date)"
echo "CPUs: ${SLURM_CPUS_PER_TASK:-?}  workers: ${WORKERS}"

source "${CONDA_BASE:-$HOME/miniconda3}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV:-egovlpv2-pip}"

ARGS=(--participants "$PID" --workers "$WORKERS")
[ "$SLIM" = "true" ] && ARGS+=(--slim)
[ "$OVERWRITE" = "true" ] && ARGS+=(--overwrite)

python precompute_fused_obbs.py "${ARGS[@]}"

echo "Task ${SLURM_ARRAY_TASK_ID} (${PID}) finished $(date)"
