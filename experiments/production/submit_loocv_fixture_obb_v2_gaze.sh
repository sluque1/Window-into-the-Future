#!/bin/bash
# Reproduces the loocv_P0*_vlo_vq_fourier_factorised_cls_prior_pe_locjoint_
# tlnon_e1_iou_fixt250_typeemb0.5_clsgain_tesinusoidal_mw15 run family (e.g.
# snapshots/2026-09-02_10-09-57_loocv_P06_..._tesinusoidal_mw15) for all nine
# LOOCV folds, with ONE change:
#   MODALITY   video,location,obb  ->  video,location,obb,gaze
# That family is submit_loocv_fixture_obb_v2.sh run with
# TIME_ENCODING=sinusoidal TIME_ENCODING_MAX_WAVELENGTH=15 (an override of
# that script's own 'fourier' default -- see resubmit_missing_loocv_folds.sh,
# which resubmitted the same override for the P01/P02/P03/P05/P09 folds of
# the non-gaze arm). Baked in as the default here, still overridable via env.
# Every other knob below is copied verbatim from submit_loocv_fixture_obb_v2.sh,
# which itself
# reproduces (config recovered from their own snapshots):
#   snapshots/2026-08-26_22-33-48_loocv_P06_vlo_vq_fourier_factorised_cls_prior_pe_locjoint_tlnoff_e1_iou
#   snapshots/2026-08-26_23-02-12_loocv_P04_..._iou
#   snapshots/2026-08-26_23-02-13_loocv_P07_..._iou
#   snapshots/2026-08-26_23-02-15_loocv_P08_..._iou
# plus the token-scale fixes (TOKEN_LAYERNORM, TYPE_EMBEDDINGS=unified,
# CLASS_EMBEDDING_GAIN, TIME_ENCODING) that make it "v2". See that script's own
# header for the full rationale of every flag below; it is not repeated here.
#
# Gaze needs no new code: gaze.py already turns each second of the observation
# window into the voxel holding the most of that second's gaze points (eye
# yaw/pitch + depth_m, through T_device_CPF and the SLAM head pose), fed as
# three per-axis token streams through the SAME location_embedding_x/y/z
# tables person_locations uses (GAZE_TOKENS=axis, the default -- unchanged
# here). Everything gaze.py/loader.py need to build it is already exported
# below exactly as the baseline had it: TIME_ANCHOR=gaze, GAZE_MAX_HZ=0 (keep
# every sample), GAZE_TIME_OFFSET_S=0.1.
#
# Type embedding: no second change needed. TYPE_EMBEDDINGS=unified is already
# ONE [5,768] type_emb table indexed by {video, location, obb, gaze,
# trajectory} (model.py's TYPE_EMB_INDEX), and TYPE_EMBEDDING_SCALE=0.5 scales
# that whole table. Turning on the `gaze` modality makes the model start
# reading row `gaze` of a table that was already built and already scaled at
# 0.5 -- there is no separate gaze-specific scale to set.
#
# Job-name prefix: 'vlo' (video,location,obb) -> 'vlog' (...,gaze), the same
# convention submit_q5_gaze_joint.sh uses for the same reason -- the modality
# set belongs in the tag, since it is the only thing distinguishing these runs
# from the vlo_ baseline (loocv_P0*_vlo_vq_fourier_factorised_cls_prior_
# pe_locjoint_tlnon_e1_iou_fixt250_typeemb0.5_clsgain_te...).
#
# Partition: defaults to medium_gpunew (override with PARTITION=... as usual).
#
# Usage:
#   ./submit_loocv_fixture_obb_v2_gaze.sh                 # all 9 LOOCV folds
#   FOLDS="P07" ./submit_loocv_fixture_obb_v2_gaze.sh      # one fold
#   DEPENDENCY=afterok:<jobid> ./submit_loocv_fixture_obb_v2_gaze.sh

set -euo pipefail

# Run from the repository root: submit_train.sh and the modules it snapshots are
# resolved relative to it, and this script lives two levels down.
cd "$(dirname "$(readlink -f "$0")")/../.."

ALL="P01,P02,P03,P04,P05,P06,P07,P08,P09"
FOLDS="${FOLDS:-P01 P02 P03 P04 P05 P06 P07 P08 P09}"

export PARTITION="${PARTITION:-medium_gpunew}"

# --- recovered from the four reference snapshots; do not change ---
export LR=1e-5
export NUM_LAYERS=2
export MODALITY=video,location,obb,gaze
export ENV_SOURCE=boxer_grid
export VOXEL_CENTERS_DIR="${VOXEL_CENTERS_DIR:-$PWD/new_voxel_centers}"
export MIN_VIDEO_DURATION=0
export REQUIRE_VALID_LOCATION=false
export GT_INTERACTIONS_RANGE=""
export GT_VOXELS_RANGE=""

export TIME_ANCHOR=gaze
export GAZE_MAX_HZ=0
export GAZE_TIME_OFFSET_S=0.1
export GAZE_TOKENS=axis

export FUSION=iou
export FUSED_CACHE=true
export BOXER_CONF_THRESHOLD=-1.0        # -> 0.6 for iou

# Inert on MODALITY=video,location,obb,gaze (no env, no trajectory segment),
# but echoed by the reference runs, so kept verbatim rather than silently
# dropped.
export ENV_ORPHAN_FALLBACK=false
export ENV_POSITIONAL_ENCODING=true
export ENV_CLASS_AVERAGING=false
export ENV_MAX_CLASSES_PER_VOXEL=4
export ENV_FALLBACK_AVERAGING=off
export TRAJECTORY_SIMPLE=true
export TRAJECTORY_EXCLUDE_UNFINISHED=true

export READOUT=cls
export VOXEL_BIAS_INIT=prior
export DECODER=voxel_query
export VOXEL_QUERY_TYPE=fourier
export VOXEL_BIAS=factorised
export VOXEL_BIAS_SCALE=1.0
export VOXEL_BIAS_LR=-1.0
export VOXEL_QUERY_FOURIER_SCALE=10.0
export VOXEL_QUERY_UNTIED=false
export DECODER_HEADS=8
export DECODER_FFN=false
export DECODER_FFN_DIM=3072
export DECODER_RESIDUAL=raw
export DECODER_ITERS=1
export DECODER_DROPOUT=0.0

export POSITION_ENCODING=auto
export POSITION_ENCODING_MODE=replace
export POSITION_ENCODING_GAIN_INIT=48.0
export LOCATION_TOKENS=joint

# NOTE TIME_ENCODING_SCALE is read ONLY by the 'fourier' branch (model.py:581,
# the std of time_W_r's init). Under TIME_ENCODING=sinusoidal it is inert, and
# appears in the echoed config without doing anything.
export TIME_ENCODING_SCALE=10.0
# Defaults to 15 (not the base script's 0.0 -> 2*OBS_TIME=60), to reproduce
# the tesinusoidal_mw15 run family this script targets.
export TIME_ENCODING_MAX_WAVELENGTH="${TIME_ENCODING_MAX_WAVELENGTH:-15}"

export LOSS=bce
export FOCAL_GAMMA=2.0
export FOCAL_ALPHA=0.25
export POS_WEIGHT=-1.0
export WEIGHT_DECAY=0.0
export ENCODER_DROPOUT=0.1
export OBB_TOKEN_DROPOUT=0.0

export NUM_EPOCHS=1
export BATCH_SIZE=16
export EVAL_STEPS=0.01
export OBS_TIME=30
export FUTURE_TIME=180
export ANTICIP_TIME=5
export TRAIN_N=0
export VAL_N=0
export TEST_N=0

# --- the point of v2 ---
export MAX_OBB_TOKENS=250
export OBB_VOXEL_GRID_PRUNE=true
export FIXTURE_OBBS=true
export FIXTURE_IOU_THRESH="${FIXTURE_IOU_THRESH:-0.4}"
export FIXTURE_COS_THRESH="${FIXTURE_COS_THRESH:-0.7}"
export CLASS_EMBEDDING_INIT=class_embeddings.npz
export CLASS_EMBEDDING_TRAINABLE=false
export TOKEN_LAYERNORM=true
export TYPE_EMBEDDINGS=unified
export TYPE_EMBEDDING_SCALE="${TYPE_EMBEDDING_SCALE:-0.5}"
export CLASS_EMBEDDING_GAIN=true
export CLASS_EMBEDDING_GAIN_INIT="${CLASS_EMBEDDING_GAIN_INIT:--1.0}"
export TIME_ENCODING="${TIME_ENCODING:-sinusoidal}"
export TIME_ENCODING_GAIN="${TIME_ENCODING_GAIN:-learned}"
export TIME_ENCODING_GAIN_INIT="${TIME_ENCODING_GAIN_INIT:-8.0}"
export WANDB_PROJECT="${WANDB_PROJECT:-OBB_tokens_fixtures_v2}"

for TEST_PID in $FOLDS; do
    # Train on every participant except the held-out one; no val split, so the
    # test set drives eval and best-model selection -- exactly as the
    # reference runs did.
    export TEST_PARTICIPANTS="$TEST_PID"
    export TRAIN_PARTICIPANTS=$(echo "$ALL" | tr ',' '\n' | grep -v "^${TEST_PID}$" | paste -sd,)
    export VAL_PARTICIPANTS=""

    MW_SUFFIX=""
    if [ "$TIME_ENCODING" = "sinusoidal" ] && \
       [ "$TIME_ENCODING_MAX_WAVELENGTH" != "0.0" ] && \
       [ "$TIME_ENCODING_MAX_WAVELENGTH" != "0" ]; then
        MW_SUFFIX="_mw${TIME_ENCODING_MAX_WAVELENGTH}"
    fi
    # 'vlog' = video,location,obb,gaze -- the vlo_ baseline plus gaze.
    NAME="loocv_${TEST_PID}_vlog_vq_fourier_factorised_cls_prior_pe_locjoint_tlnon_e1_iou_fixt250_typeemb${TYPE_EMBEDDING_SCALE}_clsgain_te${TIME_ENCODING}${MW_SUFFIX}"
    echo "--- $NAME"
    echo "    train=$TRAIN_PARTICIPANTS test=$TEST_PARTICIPANTS partition=$PARTITION"
    echo "    modality=$MODALITY cap=$MAX_OBB_TOKENS prune=$OBB_VOXEL_GRID_PRUNE fixtures=$FIXTURE_OBBS" \
         "iou>$FIXTURE_IOU_THRESH cos>$FIXTURE_COS_THRESH" \
         "ln=$TOKEN_LAYERNORM type_emb=$TYPE_EMBEDDINGS@$TYPE_EMBEDDING_SCALE class_gain=$CLASS_EMBEDDING_GAIN" \
         "time_enc=$TIME_ENCODING/$TIME_ENCODING_GAIN@$TIME_ENCODING_GAIN_INIT"
    ./submit_train.sh "$NAME"
done
