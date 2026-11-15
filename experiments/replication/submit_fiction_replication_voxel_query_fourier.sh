#!/bin/bash
# Same as submit_fiction_replication.sh, but with the voxel-query decoder
# (--decoder voxel_query --voxel_query_type fourier) instead of the linear
# readout. See the plan for the full derivation; the short version:
#
#   DECODER=voxel_query, VOXEL_QUERY_TYPE=fourier -- the intended change.
#   POSITION_ENCODING left UNSET (defaults to auto) -- REQUIRED, not optional.
#     auto resolves to 'fourier' because VOXEL_QUERY_TYPE=fourier, which is
#     what makes the decoder's queries use the tied, gain-corrected spatial
#     code (model.py:855-862). Pinning POSITION_ENCODING=none here (as the
#     linear-decoder script does, where it's a no-op) would silently build
#     the decoder's OWN private, un-gained Fourier table instead -- the
#     legacy arm CLAUDE.md documents as a broken 0.6:1 query:evidence ratio.
#   This does not touch any input token for MODALITY=video,location,env with
#   LOCATION_TOKENS=axis: env's positional encoding is gated by the separate
#   ENV_POSITIONAL_ENCODING flag (left false, unchanged), and no obb/
#   trajectory/joint-location/joint-gaze tokens exist under this modality set.
#
# Verified by smoke test (--debug_subset 20, P07):
#   Total trainable parameters: 39,804,082 (vs 40,583,424 for the linear
#   decoder -- output_layer's 3.15M removed, ~2.37M decoder + ~1.9K spatial
#   params added).
#   [optimizer] 2 parameter(s) at lr=0.001 (spatial_W_r, spatial_gain)
#   [optimizer] everything else at lr=5e-05
#
# Every other flag is identical to submit_fiction_replication.sh.
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

# Evaluate every 1% of the epoch (~99 evals) instead of the default 10%. train.py
# already evaluates once before train() (train.py:1001), so the curve starts at an
# untrained step-0 baseline and runs to the end.
#
# This needs long_gpuh200, NOT the default medium_gpuh200. Measured on the 10-eval
# runs: an env-modality eval costs 120-244s (4127 tokens/sequence), against 64.5s
# for the video,location,obb runs (313 tokens). At 99 evals that projects to
# 6.8h (P01) - 7.7h (P03), past medium_gpuh200's 6h10m cap. long_gpuh200 is the
# same H200 hardware with a 3-day cap.
export EVAL_STEPS=0.01
# PARTITION/TIME_LIMIT are overridable from the environment so a one-off can be
# sent elsewhere (e.g. a timing probe on medium_gpuh200, whose cap is 6h10m)
# without editing this file. Defaults are unchanged.
export PARTITION="${PARTITION:-long_gpuh200}"
export TIME_LIMIT="${TIME_LIMIT:-12:00:00}"

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

# --- The decoder change ---
export DECODER=voxel_query
export VOXEL_QUERY_TYPE=fourier
# POSITION_ENCODING intentionally NOT set here -- see header comment.

# Everything else at its default, restated for readability.
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
        ./submit_train.sh "loocv_${test_p}_fiction_replication_vq_fourier${EXTRA_TAG:-}"
done
