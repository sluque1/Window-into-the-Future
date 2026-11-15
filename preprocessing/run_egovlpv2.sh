#!/bin/bash
# SLURM job: extract per-second EgoVLPv2 visual features for HD-EPIC videos.
#
# This is step 1 of preprocessing and it is REQUIRED -- every training run reads
# these features, and nothing else produces them. See preprocessing/README.md.
#
# Usage:
#   EGOVLPV2_REPO=/path/to/EgoVLPv2 \
#   EGOVLPV2_WEIGHTS=/path/to/EgoVLPv2.pth \
#     sbatch --account=<acct> preprocessing/run_egovlpv2.sh
#
#   # one participant at a time (recommended: a full pass is many hours)
#   PARTICIPANTS="P01 P02" ... sbatch ... preprocessing/run_egovlpv2.sh
#
# The script is restartable: extract_features.py skips any video whose
# <video>_features.pt already exists, so a job killed on the wall clock can be
# resubmitted unchanged.

#SBATCH --job-name=egovlpv2_features
#SBATCH --output=out/%x_%j.out
#SBATCH --error=err/%x_%j.err
#SBATCH --partition=gpunew
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64000
#SBATCH --time=24:00:00
# SLURM does not expand environment variables in #SBATCH lines, so there is no
# portable way to template this. Pass your account on the command line instead --
#   sbatch --account=<acct> preprocessing/run_egovlpv2.sh
# -- or uncomment and hard-code it:
##SBATCH --account=<your-slurm-account>

set -euo pipefail

mkdir -p out err

source "${CONDA_BASE:-$HOME/miniconda3}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV:-egovlpv2-pip}"
echo "Python: $(which python) | $(python --version)"

# --- required: where the EgoVLPv2 code and checkpoint live -------------------
# EGOVLPV2_REPO must contain the inner `EgoVLPv2` package, so that
#   from EgoVLPv2.model.model import FrozenInTime
# resolves. For a clone at ~/src/EgoVLPv2 that is usually ~/src/EgoVLPv2/EgoVLPv2.
: "${EGOVLPV2_REPO:?set EGOVLPV2_REPO to your EgoVLPv2 clone}"
: "${EGOVLPV2_WEIGHTS:?set EGOVLPV2_WEIGHTS to EgoVLPv2.pth}"
export EGOVLPV2_REPO EGOVLPV2_WEIGHTS
export PYTHONPATH="$EGOVLPV2_REPO:${PYTHONPATH:-}"

# --- optional: data locations (default to the HDEPIC_ROOT layout) -----------
# HDEPIC_ROOT is the same variable loader.py reads, so training finds the
# features this writes without any further configuration.
export HDEPIC_ROOT="${HDEPIC_ROOT:-/data/video_datasets/HD-EPIC}"

HERE="$(dirname "$(readlink -f "$0")")"
PARTICIPANTS="${PARTICIPANTS:-}"
GPU_BATCH_SIZE="${GPU_BATCH_SIZE:-16}"

echo "Job started: $(date) on $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Participants: ${PARTICIPANTS:-<all>}"

python -u "$HERE/extract_features.py" \
    --gpu_batch_size "$GPU_BATCH_SIZE" \
    ${PARTICIPANTS:+--participant $PARTICIPANTS}

echo "Job finished: $(date)"
# Peak memory, for sizing a resubmission.
sstat -j "${SLURM_JOB_ID}.batch" --format=JobID,MaxRSS,MaxVMSize,Elapsed 2>/dev/null || true
