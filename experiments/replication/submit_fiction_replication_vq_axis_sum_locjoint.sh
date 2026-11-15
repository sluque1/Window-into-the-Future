#!/bin/bash
# Same as submit_fiction_replication_vq_fourier_locjoint.sh, but with
# VOXEL_QUERY_TYPE=axis_sum instead of fourier. That is the only changed flag.
#
# What it changes, and one thing that follows automatically:
#   The decoder's 4096 voxel queries are built from the shared
#   location_embedding_x/y/z index tables (e_x[i]+e_y[j]+e_z[k]) plus a
#   query_type_embedding marker, instead of a learned Fourier code of the voxel
#   centre.
#
#   POSITION_ENCODING is still left UNSET, but 'auto' now resolves to 'none'
#   rather than 'fourier' (resolve_position_encoding follows voxel_query_type,
#   model.py:55-68). So spatial_W_r / spatial_gain / spatial_pad_embedding are
#   NOT built at all, and trainer.py's FAST_LR (1e-3) param group disappears
#   with them -- this arm trains every parameter at the base LR 5e-5.
#
#   The weight sharing that LOCATION_TOKENS=joint exists for still holds, just
#   through a different vocabulary: joint location tokens go through
#   _voxel_pos_embedding, which under position_encoding='none' IS the index-table
#   sum the queries use. So location tokens and queries again carry the identical
#   positional vector, differing only by their type markers -- and unlike the
#   fourier arm, location_embedding_x/y/z are alive here (they are what both
#   sides read) rather than dead weight.
#
# So this is the axis_sum counterpart of the fourier locjoint arm: same shared-
# vocabulary property, discrete index tables instead of a continuous code.
#
# env is unchanged (MODALITY=video,location,env, ENV_POSITIONAL_ENCODING=false):
# 4096 class-fill voxel tokens carrying class identity only, no position.
#
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

export FUSION=iou
export FUSED_CACHE=true
export FIXTURE_OBBS=false
export ENV_ORPHAN_FALLBACK=false

export TIME_ANCHOR=gaze
export GAZE_TIME_OFFSET_S=0.1

# --- The decoder (same as the vq_fourier arm) ---
export DECODER=voxel_query
export VOXEL_QUERY_TYPE=axis_sum
# POSITION_ENCODING intentionally NOT set here -- see header comment.

# --- The one change this script exists for ---
export LOCATION_TOKENS=joint

# Everything else at its default, restated for readability.
export TOKEN_LAYERNORM=false
export TYPE_EMBEDDINGS=separate
export VOXEL_BIAS_INIT=zero
export READOUT=token0
export ENV_POSITIONAL_ENCODING=false
export ENV_CLASS_AVERAGING=false
export ENV_FALLBACK_AVERAGING=off
export TRAJECTORY_SIMPLE=false

for test_p in "${FOLDS[@]}"; do
    train_list=$(printf '%s\n' "${PARTICIPANTS[@]}" | grep -v "^${test_p}$" | paste -sd, -)
    TEST_PARTICIPANTS="$test_p" \
    TRAIN_PARTICIPANTS="$train_list" \
        ./submit_train.sh "loocv_${test_p}_fiction_replication_vq_axis_sum_locjoint${EXTRA_TAG:-}"
done
