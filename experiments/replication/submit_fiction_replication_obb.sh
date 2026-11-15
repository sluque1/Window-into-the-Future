#!/bin/bash
# ARM 1 (the FICTION replication) with the OBB modality in place of env.
# That is the ONLY architectural change: the linear readout, per-axis (90-token)
# localisation, no time encoding, no spatial encoding, no token LayerNorm and
# six layers are all kept exactly as in submit_fiction_replication.sh.
#
# env -> obb:
#   env  = 4096 tokens, one per voxel, carrying only the class of the OBB whose
#          centroid won that cell (highest confidence takes it, so a second
#          object in the same voxel is discarded).
#   obb  = one token per fused box, capped at 250, each built as
#            loc_x[i] + loc_y[j] + loc_z[k] + class_embedding + volume_embedding
#            + obb_type_embedding
#          POSITION_ENCODING stays 'none' (arm 1's setting), so the position term
#          is the per-axis index-table sum -- the SAME tables the 90 location
#          tokens read. No Fourier code anywhere in this arm.
#   Sequence length drops from 4218 to ~312.
#
# OBB source: precomputed IoU-fused Boxer detections
#   (boxer_obbs/<video_id>/fused_obbs_iou.pkl) PLUS the Digital-Twin fixtures,
#   ~120 detections + ~43 fixtures = ~159 tokens/window, 0.0% empty windows.
#   OBB_VOXEL_GRID_PRUNE is implied by FIXTURE_OBBS (loader.py:333) and is set
#   explicitly here for readability.
#
# CLASS EMBEDDINGS are read from class_embeddings.npz (OWLv2 text vectors) and
#   are NOT trained: CLASS_EMBEDDING_TRAINABLE=false keeps the 1082 semantic rows
#   as a buffer, so only the 512->768 class_proj learns. The relative geometry of
#   the class space is therefore fixed by the npz, as intended.
#   CLASS_EMBEDDING_GAIN=true is kept from the previous OBB arms: it does not
#   move classes relative to one another, it only sets how loud the class term is
#   (normalise, then a learned scalar init at sqrt(768)=27.71). Without it the
#   projected vectors sit at norm 0.71 against 47.8/27.7/27.5 for the position,
#   volume and marker terms of the same token, so swapping a box's class would
#   rotate the token by under 1.3 degrees and the npz would be inert.
#
# WandB project: Option1.
#
# Everything else is identical to submit_fiction_replication.sh.
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

# Env-grid construction: bug-fixed fusion (grid fusion doubles box scale), the
# precomputed cache already exists for 152/153 videos. No fixture injection (no
# analog in original). No orphan snap (matches original's "unregistered if it
# fits no voxel centroid" behavior).
export FUSION=iou
export FUSED_CACHE=true
export FIXTURE_OBBS=true
export OBB_VOXEL_GRID_PRUNE=true   # implied by FIXTURE_OBBS; explicit for readability
export MAX_OBB_TOKENS=250
export CLASS_EMBEDDING_INIT=class_embeddings.npz
export CLASS_EMBEDDING_TRAINABLE=false   # semantic rows frozen: they come from the npz
export CLASS_EMBEDDING_GAIN=true
export ENV_ORPHAN_FALLBACK=false

# Correct device-clock alignment (bug fix in the legacy trajectory anchor,
# unrelated to architecture fidelity -- every current production run uses this).
export WANDB_PROJECT="${WANDB_PROJECT:-Option1}"

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
# METRIC-NEUTRAL here, and a ~5x speedup. The loader builds trajectory tokens for
# EVERY sample regardless of --input_modality (loader.py:1051-1067); the model only
# reads them when 'trajectory' is in the modality (model.py:1717), which it is not.
# The non-simple branch nonetheless calls _get_raw_slam() and reconstructs the full
# path with interior shape points, per sample, in training AND evaluation, then the
# result is discarded. Measured on identical folds/hardware: true -> 35-61s per
# eval, false -> 251-256s. Setting it true cannot change any metric in this arm.
export TRAJECTORY_SIMPLE="${TRAJECTORY_SIMPLE:-true}"

for test_p in "${FOLDS[@]}"; do
    train_list=$(printf '%s\n' "${PARTICIPANTS[@]}" | grep -v "^${test_p}$" | paste -sd, -)
    TEST_PARTICIPANTS="$test_p" \
    TRAIN_PARTICIPANTS="$train_list" \
        ./submit_train.sh "loocv_${test_p}_fiction_replication_obb${EXTRA_TAG:-}"
done
