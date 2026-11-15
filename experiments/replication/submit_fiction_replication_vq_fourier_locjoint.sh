#!/bin/bash
# Same as submit_fiction_replication_voxel_query_fourier.sh, but with
# LOCATION_TOKENS=joint instead of axis. One named change.
#
# What joint does here (model.py's loc_joint branch, ~line 1383):
#   'axis' (the vq_fourier runs) feeds location as three per-axis SCALAR streams
#   (90 tokens) through location_embedding_x/y/z -- index tables that share NO
#   parameters with the decoder's Fourier-coded queries. Those runs therefore
#   carried two unrelated spatial vocabularies at once.
#   'joint' feeds one token per second carrying the whole position through
#   _voxel_pos_embedding (90 tokens -> 30), which under position_encoding=fourier
#   IS the shared Fourier code -- the same spatial_W_r / spatial_gain the decoder's
#   voxel queries use. So a location token and a query at the same voxel now carry
#   the identical positional vector by construction, differing only by their type
#   marker. That weight sharing is the point of this arm.
#
# POSITION_ENCODING stays UNSET (-> auto -> fourier, because
# VOXEL_QUERY_TYPE=fourier). That is what puts BOTH the queries and the joint
# location tokens on the shared code; pinning it to 'none' would silently drop
# location back to the index tables AND give the decoder its own private,
# un-gained table.
#
# Two structural consequences, neither a separate choice:
#   * location_type_embedding (768 params) is created -- required, because a joint
#     location token is otherwise byte-identical to a decoder query at that voxel.
#     model.py builds it LAST on purpose, so this arm and its 'axis' twin draw every
#     other parameter from an identical RNG stream (a paired comparison, not two
#     random inits).
#   * location_embedding_x/y/z (39,168 params) lose their last consumer and receive
#     exactly zero gradient: location now reads the Fourier code, the queries read
#     it too, env carries no position (ENV_POSITIONAL_ENCODING=false), and no
#     gaze/obb/trajectory tokens exist here. Harmless dead weight, but it means a
#     checkpoint from this arm is not loadable into an 'axis' one.
#
# env is deliberately left as is (ENV_POSITIONAL_ENCODING=false), so env tokens
# still carry class identity only and no position.
#
# Every other flag is identical to submit_fiction_replication_voxel_query_fourier.sh.
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
export VOXEL_QUERY_TYPE=fourier
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
        ./submit_train.sh "loocv_${test_p}_fiction_replication_vq_fourier_locjoint${EXTRA_TAG:-}"
done
