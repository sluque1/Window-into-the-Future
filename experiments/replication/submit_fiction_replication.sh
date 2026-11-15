#!/bin/bash
# Replicates the original FIction (thechargedneutron/FIction) architecture and
# hyperparameters as closely as possible on the HD-EPIC v2 pipeline.
#
# Verified item-by-item against the original repo's main.py/loader.py:
#   - linear decoder (transformer_output[0] -> one Linear(768,4096))
#   - no temporal encoding, no spatial encoding
#   - location as 3x30 axis tokens, location_vocab_size=17 (16+1) -- exact match
#   - no extra LayerNorm beyond nn.TransformerEncoderLayer's own internals
#   - no type-key embedding (none constructed under this modality+layout anyway)
#   - bias always on (nn.Linear default, matches original -- "no bias" really
#     means no special bias *init* trick, i.e. --voxel_bias_init zero)
#   - 6 transformer layers, LR 5e-5, HF-default AdamW/linear-scheduler/no-warmup
#   - batch_size=16 (v2's own default; original used 2, but that reads as a
#     hardware constraint of a different setup rather than a design choice --
#     kept at 16 so this run stays comparable to other v2 experiments)
#   - 1 epoch (deliberate deviation from original's hardcoded 3, for a faster run)
#
# The env voxel grid's "highest-confidence-wins, fixture-wins" fill rule is a
# v2-specific adaptation (original's Detic/WHAM boxes arrive pre-fused, with no
# confidence-based tie-break at all) -- expected given the Detic->Boxer swap.
# Fixture OBBs are left OFF: no furniture-injection concept exists in original.
#
# LOOCV over all nine participants: 9 SLURM jobs, each holding one participant
# out as the test set and training on the other eight (no validation split).
set -euo pipefail

# Run from the repository root: submit_train.sh and the modules it snapshots are
# resolved relative to it, and this script lives two levels down.
cd "$(dirname "$(readlink -f "$0")")/../.."
cd "$(dirname "$0")"

PARTICIPANTS=(P01 P02 P03 P04 P05 P06 P07 P08 P09)
# ONLY_PARTICIPANTS, if set, restricts which folds are SUBMITTED -- for resubmitting
# individual folds after a failure without duplicating the ones already queued:
#   ONLY_PARTICIPANTS=P01,P02 ./<this script>
# It must NOT narrow the training set: every fold still trains on the other EIGHT
# participants, so train_list below is always computed from the full PARTICIPANTS
# list while the loop runs over FOLDS.
FOLDS=("${PARTICIPANTS[@]}")
[ -n "${ONLY_PARTICIPANTS:-}" ] && read -ra FOLDS <<< "${ONLY_PARTICIPANTS//,/ }"

# --- Original-paper hyperparameters. train_sbatch.sh's shell defaults diverge
# from train.py's own argparse defaults for LR and NUM_LAYERS (both lower,
# presumably tuned for the ablation-sweep era) -- must override explicitly. ---
export LR=5e-5
export NUM_LAYERS=6
export NUM_EPOCHS=1
export BATCH_SIZE=16

# Evaluate every 2% of the epoch (49 ticks) -- train.py also evaluates once
# before train(), so the curve starts at an untrained step-0 baseline.
#
# 1% (97 ticks) did NOT fit medium_gpuh200's 6h10m cap; 2% does, with 29% to
# spare. Calibrated against a completed 1% probe (P05, measured 4:18:19 vs 4.42h
# predicted): non-eval cost is a fixed 3913s, so halving the ticks more than
# halves the run. Projected at 2%: P03 4.36h (worst) .. P05 2.73h (best).
# The voxel-query decoder is free at eval (P01 207.9s linear vs 209.3s vq), so
# these numbers hold for all three arms.
export EVAL_STEPS="${EVAL_STEPS:-0.02}"
# PARTITION/TIME_LIMIT are overridable from the environment so a one-off can be
# sent elsewhere (e.g. a timing probe on medium_gpuh200, whose cap is 6h10m)
# without editing this file. Defaults are unchanged.
export PARTITION="${PARTITION:-medium_gpuh200}"
export TIME_LIMIT="${TIME_LIMIT:-6:00:00}"

export MODALITY=video,location,env
export OBS_TIME=30
export ANTICIP_TIME=5
export FUTURE_TIME=180
export VOXEL_CENTERS_DIR="${VOXEL_CENTERS_DIR:-$PWD/new_voxel_centers}"
export VAL_PARTICIPANTS=""   # no-val mode (must be set-but-empty)

# Env-grid construction: bug-fixed fusion (grid fusion doubles box scale), the
# precomputed cache already exists for 152/153 videos. No fixture injection (no
# analog in original). No orphan snap (matches original's "unregistered if it
# fits no voxel centroid" behavior).
export FUSION=iou
export FUSED_CACHE=true
export FIXTURE_OBBS=false
export ENV_ORPHAN_FALLBACK=false

# Correct device-clock alignment (bug fix in the legacy trajectory anchor,
# unrelated to architecture fidelity -- every current production run uses this).
export TIME_ANCHOR=gaze
export GAZE_TIME_OFFSET_S=0.1

# Everything else explicitly restated at its default so the run is readable
# from this file alone -- all match original's architecture by construction.
export DECODER=linear
export TIME_ENCODING=none
export POSITION_ENCODING=none
export TOKEN_LAYERNORM=false
export TYPE_EMBEDDINGS=separate
export VOXEL_BIAS_INIT=zero
export READOUT=token0
export LOCATION_TOKENS=axis
export ENV_POSITIONAL_ENCODING=false
export ENV_CLASS_AVERAGING=false
export ENV_FALLBACK_AVERAGING=off
export TRAJECTORY_SIMPLE=false

for test_p in "${FOLDS[@]}"; do
    train_list=$(printf '%s\n' "${PARTICIPANTS[@]}" | grep -v "^${test_p}$" | paste -sd, -)
    TEST_PARTICIPANTS="$test_p" \
    TRAIN_PARTICIPANTS="$train_list" \
        ./submit_train.sh "loocv_${test_p}_fiction_replication${EXTRA_TAG:-}"
done
