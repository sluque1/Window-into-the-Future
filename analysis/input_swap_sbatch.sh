#!/bin/bash
# One input_swap_ablation.py fold per SLURM task.
#
# Why not just loop in an interactive session: two attempts at the nine-fold run
# died silently mid-fold (no traceback in a 2>&1 log, session still alive), which
# is what an external kill looks like. One process per fold caps peak memory at a
# single participant, gives each fold an explicit --mem, and means a fold that
# dies costs that fold rather than every fold after it.
#
# Usage:
#   ./analysis/input_swap_sbatch.sh '2026-08-12_08-54-2[4-7]*_te10'      # P06-P09
#   ./analysis/input_swap_sbatch.sh '2026-08-12_08-54-*_te10'            # all nine
#   MIN_OBS=30 ./analysis/input_swap_sbatch.sh '2026-08-13_17-2*_tln'    # full windows only
#
# Each task writes analysis/out/input_swap/<tag>/<PID>.json; merge them with
# analysis/input_swap_merge.py.

#SBATCH --job-name=input_swap
# SLURM does not expand environment variables in #SBATCH lines, so there is no
# portable way to template this. Pass your account on the command line instead --
#   sbatch --account=<acct> ...   (submit_train.sh forwards $SLURM_ACCOUNT for you)
# -- or uncomment and hard-code it:
##SBATCH --account=<your-slurm-account>
#SBATCH --partition=medium_gpua100
#SBATCH --gpus=1
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --output=out/%x_%A_%a.out
#SBATCH --error=err/%x_%A_%a.err

set -euo pipefail
# Resolve to an absolute path BEFORE any cd: this script re-submits itself, and a
# relative "$0" no longer resolves once the cd below has moved us.
SELF="$(readlink -f "$0")"

SNAP=/data/video_datasets/HD-EPIC/snapshots
PATTERN="${1:-}"

# --- Submission side: expand the glob and fire one array task per run ---------
if [ -z "${SLURM_ARRAY_TASK_ID:-}" ]; then
    [ -n "$PATTERN" ] || { echo "usage: $0 '<snapshot glob>'" >&2; exit 1; }
    # Derived from this file's own location, never from SLURM_SUBMIT_DIR. When
    # sbatch is called from inside an existing allocation, --export=ALL carries
    # the PARENT job's SLURM_SUBMIT_DIR into the child, so it points at wherever
    # the interactive session was launched -- not at this repo. That silently
    # sends the relative --output paths and the python call one level too high.
    REPO="$(dirname "$(dirname "$SELF")")"
    cd "$REPO"
    mapfile -t RUNS < <(eval ls -d "$SNAP/$PATTERN" 2>/dev/null)
    [ "${#RUNS[@]}" -gt 0 ] || { echo "no runs matched $PATTERN" >&2; exit 1; }
    # A tag for the output subdirectory, taken from the arm the pattern selects.
    TAG="${TAG:-$(basename "${RUNS[0]}" | sed 's/.*_vlo_//')}"
    mkdir -p "analysis/out/input_swap/$TAG" out err
    printf '%s\n' "${RUNS[@]}" > "analysis/out/input_swap/$TAG/runs.txt"
    echo "${#RUNS[@]} fold(s) -> analysis/out/input_swap/$TAG/"
    printf '  %s\n' "${RUNS[@]##*/}"
    # --chdir pins the job's working directory, which is what the relative
    # --output/--error paths above resolve against; REPO is passed through as
    # well so the task side does not have to infer it from a spooled "$0".
    sbatch --array="0-$((${#RUNS[@]} - 1))" \
           --chdir="$REPO" \
           --export=ALL,REPO="$REPO",TAG="$TAG",MIN_OBS="${MIN_OBS:-0}",WHICH="${WHICH:-best}" \
           "$SELF" "$PATTERN"
    exit 0
fi

# --- Task side: score exactly one fold ----------------------------------------
cd "${REPO:?REPO not exported by the submission side}"
source "${CONDA_BASE:-$HOME/miniconda3}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV:-egovlpv2-pip}"

RUN=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "analysis/out/input_swap/$TAG/runs.txt")
PID=$(echo "$RUN" | grep -o 'loocv_P0[0-9]' | cut -d_ -f2)
echo "task $SLURM_ARRAY_TASK_ID: $PID  min_obs=$MIN_OBS  which=$WHICH"
echo "  $RUN"

python -u analysis/input_swap_ablation.py \
    --run "$RUN" \
    --which "$WHICH" \
    --min_obs_seconds "$MIN_OBS" \
    --out "analysis/out/input_swap/$TAG/${PID}.json"
