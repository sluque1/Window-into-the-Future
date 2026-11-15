#!/bin/bash
# One nn_probe.py participant per SLURM task, on CPU nodes.
#
# The probe has no learned parameters and no forward passes -- its cost is the
# loader's per-window OBB fusion plus an N x N cosine similarity, both CPU-bound.
# So this asks for no GPU, and the nine participants run in parallel: ~15 minutes
# wall instead of ~2 hours sequential, and a task that dies costs one participant
# rather than every participant after it.
#
# Usage:
#   ./analysis/nn_probe_sbatch.sh                       # all nine, 800 episodes
#   N_EPISODES=0 ./analysis/nn_probe_sbatch.sh          # every episode (slower)
#   PARTICIPANTS="P05 P07" ./analysis/nn_probe_sbatch.sh
#
# Each task writes analysis/out/nn_probe/<PID>.json; summarise with
# analysis/nn_probe_summary.py.

#SBATCH --job-name=nn_probe
# SLURM does not expand environment variables in #SBATCH lines, so there is no
# portable way to template this. Pass your account on the command line instead --
#   sbatch --account=<acct> ...   (submit_train.sh forwards $SLURM_ACCOUNT for you)
# -- or uncomment and hard-code it:
##SBATCH --account=<your-slurm-account>
#SBATCH --partition=medium_cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=out/%x_%A_%a.out
#SBATCH --error=err/%x_%A_%a.err

set -euo pipefail
# Absolute before any cd: this script re-submits itself.
SELF="$(readlink -f "$0")"

if [ -z "${SLURM_ARRAY_TASK_ID:-}" ]; then
    # Derived from this file's location, never SLURM_SUBMIT_DIR: when sbatch is
    # called from inside an existing allocation, --export=ALL carries the PARENT
    # job's SLURM_SUBMIT_DIR through, pointing wherever that session started.
    REPO="$(dirname "$(dirname "$SELF")")"
    cd "$REPO"
    read -ra PIDS <<< "${PARTICIPANTS:-P01 P02 P03 P04 P05 P06 P07 P08 P09}"
    mkdir -p analysis/out/nn_probe out err
    printf '%s\n' "${PIDS[@]}" > analysis/out/nn_probe/participants.txt
    echo "${#PIDS[@]} participant(s): ${PIDS[*]}"
    echo "n_episodes=${N_EPISODES:-800}  k=${K:-1 10 50}  -> analysis/out/nn_probe/"
    sbatch --array="0-$((${#PIDS[@]} - 1))" \
           --chdir="$REPO" \
           --export=ALL,REPO="$REPO",N_EPISODES="${N_EPISODES:-800}",K="${K:-1 10 50}",SEED="${SEED:-0}" \
           "$SELF"
    exit 0
fi

cd "${REPO:?REPO not exported by the submission side}"
source "${CONDA_BASE:-$HOME/miniconda3}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV:-egovlpv2-pip}"

PID=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" analysis/out/nn_probe/participants.txt)
echo "task $SLURM_ARRAY_TASK_ID: $PID  n_episodes=$N_EPISODES  k=$K  seed=$SEED"

# $K is intentionally unquoted: it is a list of k values for nargs='+'.
# shellcheck disable=SC2086
python -u analysis/nn_probe.py \
    --participants "$PID" \
    --n_episodes "$N_EPISODES" \
    --k $K \
    --seed "$SEED" \
    --out "analysis/out/nn_probe/${PID}.json"
