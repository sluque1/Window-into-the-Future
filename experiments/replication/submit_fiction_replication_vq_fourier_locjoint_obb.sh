#!/bin/bash
# Arm 3: the vq_fourier_locjoint arm with the OBB modality replacing env.
#
# THE CHANGE: MODALITY video,location,env -> video,location,obb.
#   env  = 4096 tokens, one per voxel, each carrying only the class of the OBB
#          whose centroid won that cell (highest confidence takes the slot, so
#          two objects in one voxel means one is discarded outright).
#   obb  = one token per fused box, capped at 250. Each token is
#            _voxel_pos_embedding(centre) + class_embedding(sem_id+1)
#            + volume_embedding(log-spaced bin) + obb_type_embedding
#          (model.py's obb_mapped block). So co-located objects both survive,
#          and box SIZE becomes readable -- signal the voxel grid could only
#          express as "how many cells did this fill".
#   Encoder sequence drops 4158 -> 312 (measured), which is why this arm is much
#   cheaper than the env arms despite identical everything else.
#
# THE SHARED SPATIAL CODE IS PRESERVED, which is the point of keeping fourier:
#   _voxel_pos_embedding under position_encoding=fourier IS the shared code, so
#   an OBB token, a joint location token and a decoder query at the same voxel
#   all carry the identical positional vector, differing only by their type
#   markers. Verified by gradient probe: spatial_W_r/spatial_gain LIVE, and
#   location_embedding_x/y/z dead (nothing reads the index tables any more).
#   POSITION_ENCODING stays UNSET (auto -> fourier via VOXEL_QUERY_TYPE).
#
# FIXTURES: FIXTURE_OBBS=true appends each participant's 30-61 Digital-Twin
#   fixture boxes to the IoU-fused Boxer detections, after dropping detections a
#   fixture already covers (3D IoU > 0.4 AND class cosine > 0.7). Fixtures carry
#   prob=1.0 against Boxer's max 0.898, so the 250-token cap never drops one.
#   Unlike the env arms -- where fixtures are structurally inert because
#   build_env() never receives the box_transform hook -- they genuinely apply here.
#   Side effect: windows with zero boxes go 3.9% -> 0.0%.
#
# THE PRUNE COMES WITH THEM, not as a separate choice: loader.py:333 sets
#   obb_voxel_grid_prune = bool(flag) or fixture_obbs. It drops boxes whose centre
#   lies outside the participant's voxel grid (1.71% of boxes). Necessary because
#   _nearest_voxel_ijk is an unbounded argmin over the 4096 centres, so an
#   un-pruned distant box gets snapped onto a boundary voxel and asserts an object
#   at the edge of the kitchen. Requires VOXEL_CENTERS_DIR (set below).
#
# CLASS EMBEDDINGS from class_embeddings.npz (OWLv2 text vectors), joined by name
#   through data/boxer_class_vocab.json. Relative path on purpose: train_sbatch.sh
#   runs from the snapshot, so this reads the FROZEN copy submit_train.sh placed
#   there, not the live file.
#   CLASS_EMBEDDING_GAIN=true is required for it to do anything: the projected
#   vectors arrive at norm 0.71 against 47.8/27.7/27.5 for the position/volume/
#   marker terms of the same token, so without the gain swapping a box's class
#   rotates its token by under 1.3 degrees. The gain normalises and rescales to
#   sqrt(768)=27.71 and trains at 1e-3 via trainer.FAST_PARAMS. It changes how
#   LOUD classes are, never their positions relative to one another.
#   CLASS_EMBEDDING_TRAINABLE stays false (default): the 1082 semantic rows are a
#   buffer, only the 512->768 class_proj learns.
#
# Everything else is identical to submit_fiction_replication_vq_fourier_locjoint.sh.
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

# Overridable. 5e-5 is the FICTION-replication value (and HuggingFace's own
# TrainingArguments default); the earlier production arms on this branch used 1e-5.
export LR="${LR:-5e-5}"
# NUM_LAYERS is overridable. 6 is the FICTION-replication spec (the original
# hardcodes num_transformer_layers=6).
export NUM_LAYERS="${NUM_LAYERS:-6}"
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

export MODALITY=video,location,obb
export OBS_TIME=30
export ANTICIP_TIME=5
export FUTURE_TIME=180
export VOXEL_CENTERS_DIR="${VOXEL_CENTERS_DIR:-$PWD/new_voxel_centers}"
export VAL_PARTICIPANTS=""   # no-val mode (must be set-but-empty)

export FUSION=iou
export FUSED_CACHE=true
export FIXTURE_OBBS=true
export OBB_VOXEL_GRID_PRUNE=true   # forced by FIXTURE_OBBS anyway; explicit for readability
export MAX_OBB_TOKENS=250
export CLASS_EMBEDDING_INIT=class_embeddings.npz
export CLASS_EMBEDDING_GAIN=true
export ENV_ORPHAN_FALLBACK=false

export TIME_ANCHOR=gaze
export GAZE_TIME_OFFSET_S=0.1

# --- The decoder (same as the vq_fourier arm) ---
export DECODER=voxel_query
export VOXEL_QUERY_TYPE=fourier
# POSITION_ENCODING intentionally NOT set here -- see header comment.

# --- The one change this script exists for ---
export LOCATION_TOKENS=joint

# env-only flags below are now INERT (env is not in MODALITY); kept so the
# diff against the env arm stays readable.
# Overridable: the tesinusoidal arm sets TOKEN_LAYERNORM=true, which puts every
# segment at sqrt(D)=27.7 instead of the raw norms each init convention happened
# to produce (video 6.25 vs obb 61.7). It also matters for the time code, which
# is added AFTER the segment norm.
export TOKEN_LAYERNORM="${TOKEN_LAYERNORM:-false}"
# Overridable. 'separate' = named per-branch markers, added INSIDE the segment
# LayerNorm, and video never gets one. 'unified' = one [5,768] table that adds a
# video row and is applied AFTER the norm. NOTE: unified skips the legacy draws,
# so it SHIFTS THE RNG STREAM -- a unified run is a new arm, not weight-paired
# with a separate one.
export TYPE_EMBEDDINGS="${TYPE_EMBEDDINGS:-separate}"
# Overridable. 'zero' = plain nn.Linear init (the FICTION default). 'prior' sets
# the output bias to log(p/(1-p)) for the measured train-split positive rate --
# the standard fix for the ~1:200 imbalance that BCEWithLogitsLoss leaves
# unaddressed. A single scalar applied uniformly, NOT a per-voxel field.
export VOXEL_BIAS_INIT="${VOXEL_BIAS_INIT:-zero}"
# Overridable. Under DECODER=voxel_query the readout does not feed the output
# (all 4096 voxels are queries), but it still changes the SEQUENCE the decoder
# cross-attends over: 'cls' prepends a dedicated learned token (+768 params,
# one extra key/value), 'token0' does not and lets the first visual token
# double as a content token.
export READOUT="${READOUT:-token0}"
export ENV_POSITIONAL_ENCODING=false
export ENV_CLASS_AVERAGING=false
export ENV_FALLBACK_AVERAGING=off
# Overridable, and METRIC-NEUTRAL for this modality set. The loader builds
# trajectory tokens for EVERY sample regardless of --input_modality
# (loader.py:1051-1067); the model only reads them when 'trajectory' is in the
# modality (model.py:1717), which it is not here -- so the tensors are discarded.
# But the non-simple branch first calls _get_raw_slam() and reconstructs the full
# path with interior shape points, per sample, in both training and evaluation.
# Measured: TRAJECTORY_SIMPLE=true runs evaluate at 35-61s, false at 251-256s on
# the same folds and hardware -- ~5x, entirely wasted work. Setting it true cannot
# change any metric here; it only stops the raw-SLAM read.
export TRAJECTORY_SIMPLE="${TRAJECTORY_SIMPLE:-false}"

for test_p in "${FOLDS[@]}"; do
    train_list=$(printf '%s\n' "${PARTICIPANTS[@]}" | grep -v "^${test_p}$" | paste -sd, -)
    TEST_PARTICIPANTS="$test_p" \
    TRAIN_PARTICIPANTS="$train_list" \
        ./submit_train.sh "loocv_${test_p}_fiction_replication_vq_fourier_locjoint_obb${EXTRA_TAG:-}"
done
