#!/bin/bash
# v2 of the fixture arm: everything submit_loocv_fixture_obb.sh does, plus the
# token-scale fixes measured out of that arm's own checkpoints.
#
# Reference runs this reproduces (config recovered from their own snapshots):
#   snapshots/2026-08-26_22-33-48_loocv_P06_vlo_vq_fourier_factorised_cls_prior_pe_locjoint_tlnoff_e1_iou
#   snapshots/2026-08-26_23-02-12_loocv_P04_..._iou
#   snapshots/2026-08-26_23-02-13_loocv_P07_..._iou
#   snapshots/2026-08-26_23-02-15_loocv_P08_..._iou
# Every knob below is the value those four echoed at startup. They were
# identical to each other in everything but the participant split.
#
# The ONLY differences from those runs:
#   MAX_OBB_TOKENS       512  -> 250
#   OBB_VOXEL_GRID_PRUNE false -> true    (step 3a)
#   FIXTURE_OBBS         false -> true    (steps 2 + 3b + 4)
#   FIXTURE_IOU_THRESH / FIXTURE_COS_THRESH   new, 0.4 / 0.7
#   CLASS_EMBEDDING_INIT  ""   -> class_embeddings.npz
#   TOKEN_LAYERNORM      false -> true
#   TYPE_EMBEDDINGS      separate -> unified   (scale 0.5)
#   CLASS_EMBEDDING_GAIN false -> true
#   TIME_ENCODING        none  -> fourier, gain learned @ 8.0
#   WANDB_PROJECT   OBB_tokens2 -> OBB_tokens_fixtures_v2  (kept apart from the baselines)
#
# Class embeddings
# ----------------
# CLASS_EMBEDDING_INIT seeds the ONE class_embedding table that env, obb and
# trajectory tokens all read (model._class_embed is the single accessor for all
# three) from the OWLv2 text vectors in class_embeddings.npz -- the same file the
# fixture filter uses for its cosine gate, so filtering and embedding now share
# one notion of what a class is.
#
#   1082 of the 1222 rows get a text vector, held frozen in a buffer and mapped
#        into the hidden space by a LEARNED 512->768 projection (+393,216 params),
#        so the space can rotate and rescale but cannot pull two classes apart.
#   140  rows have no vector -- index 0 (empty/pad), the OOV row 1221, and the 138
#        sem_id slots the taxonomy never assigns -- and keep their LEARNED
#        nn.Embedding rows, selected per row by model._class_embed's torch.where.
#   All 20 fixture class rows are on the semantic side (checked).
#
# CLASS_EMBEDDING_TRAINABLE stays false on purpose: making the 1082 semantic rows
# trainable would let six kitchens' worth of gradient pull the rare classes apart
# again, which is the LOOCV failure this initialisation exists to prevent. The
# projection is trainable either way.
#
# Measured structure in the resulting table: cupboard/cabinet 0.929, stove/oven
# 0.784, cup/mug 0.765, against cup/refrigerator 0.092 and mug/stove 0.221.
#
# NOTE this changes the parameter count, so these runs are NOT checkpoint-
# compatible with the four references (17,743,282 params). That is intended.
#
# Threshold defaults, and why
# ---------------------------
# FIXTURE_COS_THRESH=0.7 is the value the BoxerNet reference pipeline uses to
# gate merges on OWLv2 label similarity (iou_fusion.py's module docstring), and
# it lands in the gap the measured embedding neighbourhoods leave: it passes the
# genuine synonyms -- cupboard/cabinet 0.926, stove/cooker 0.885, stove/oven
# 0.769 -- while blocking the merely-related furniture that must NOT absorb a
# detection: cupboard/storage 0.647, cupboard/shelf 0.644, refrigerator/
# dishwasher 0.603.
#
# FIXTURE_IOU_THRESH=0.4 is BoxerEnvBuilder's own iou_threshold, i.e. the overlap
# the pipeline already treats as "these two boxes are the same physical object".
# Reusing it keeps one definition of that. It errs toward keeping boxes: a higher
# threshold discards less, and the filter only ever removes.
#
# Both are exposed, so sweeping them needs no code change:
#   FIXTURE_IOU_THRESH=0.3 FIXTURE_COS_THRESH=0.75 ./submit_loocv_fixture_obb_v2.sh
#
# The `floor` and `surface` fixture classes are excluded entirely
# (fixture_obbs.FIXTURE_DROP_CLASSES): they are room-scale slabs, one `surface`
# per kitchen measuring 14-40 m^3, which as a single token would carry the top
# volume bin and a centre near the middle of the room. 2 meshes per participant.
#
# Measured effect over all 28,777 cached windows (obs=30s, anticip=5s, gaze
# anchor), per window:
#   fused 124.1 -> -2.1 outside the grid -> -3.7 covered by a fixture
#                -> +44.1 fixtures = 162.3, max 317
#   250 truncates 546 windows (1.90%), losing 0.30 tokens/window over the set.
#   Truncation is confidence-ordered and fixtures carry prob=1.0 against a
#   maximum Boxer prob of 0.898, so a fixture is never the box dropped.
#   Worst fold by truncation is P05 (13.9% of its windows), then P09 (3.5%);
#   P03 and P04 never truncate. P05/P09 are train-only in these four folds.
#
# Prerequisites (all already satisfied, checked before writing this script):
#   - the fused IoU cache, boxer_obbs/<video>/fused_obbs_iou.pkl: 152 of 153
#     videos, 28,777 windows. Every fold needs ALL NINE participants' caches,
#     since LOOCV trains on the eight it does not test.
#   - Digital-Twin meshes under
#     /data/video_datasets/HD-EPIC/Digital-Twin/blenders/meshes/<pid>/*.obj
#     (32-63 per participant, 422 total; 404 after floor/surface are excluded,
#     30-61 per participant, all of them with a class embedding).
#   - new_voxel_centers/<pid>_voxel_centers.npy.
# Re-check the assumptions behind the fixtures with:
#   python fixture_obbs.py --verify
#
# Usage:
#   ./submit_loocv_fixture_obb_v2.sh                    # all four folds
#   FOLDS="P07" ./submit_loocv_fixture_obb_v2.sh        # one fold
#   DEPENDENCY=afterok:<jobid> ./submit_loocv_fixture_obb_v2.sh

set -euo pipefail

# Run from the repository root: submit_train.sh and the modules it snapshots are
# resolved relative to it, and this script lives two levels down.
cd "$(dirname "$(readlink -f "$0")")/../.."

ALL="P01,P02,P03,P04,P05,P06,P07,P08,P09"
FOLDS="${FOLDS:-P06 P04 P07 P08}"

# --- recovered from the four reference snapshots; do not change ---
export LR=1e-5
export NUM_LAYERS=2
export MODALITY=video,location,obb
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

# Inert on MODALITY=video,location,obb (no env, no trajectory segment), but
# echoed by the reference runs, so kept verbatim rather than silently dropped.
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
# The sinusoidal knob: the longest wavelength in the geometric ladder, i.e. how
# slow the slowest "hand" is. 0.0 resolves to 2*OBS_TIME = 60. The window is only
# 30 slots, so a wavelength far beyond that cannot complete half a turn across it
# and those dimensions are frozen -- at the literature's 10000, 147 of 384 are
# dead. Measured at T=30 (analysis/time_code_similarity.py): mw 15 gives the
# sharpest resolution (cos<0.5 at gap 3) and the widest dynamic range (1.099);
# mw 60 is smoother (gap 4, range 0.898); >=120 starts throwing capacity away.
export TIME_ENCODING_MAX_WAVELENGTH="${TIME_ENCODING_MAX_WAVELENGTH:-0.0}"
# TIME_ENCODING / TIME_ENCODING_GAIN were none/fixed in the reference runs; they
# are set below, with the other token-scale changes this script exists for.
# TOKEN_LAYERNORM was false in the reference runs; it is set true below, with the
# other token-scale changes this script exists for.

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

# --- the point of this script -----------------------------------------------
export MAX_OBB_TOKENS=250
export OBB_VOXEL_GRID_PRUNE=true
export FIXTURE_OBBS=true
export FIXTURE_IOU_THRESH="${FIXTURE_IOU_THRESH:-0.4}"
export FIXTURE_COS_THRESH="${FIXTURE_COS_THRESH:-0.7}"
# Relative on purpose: the job's cwd is the snapshot directory, and
# submit_train.sh copies class_embeddings.npz into it, so each run reads the
# vectors frozen alongside its own source rather than the live tree's.
export CLASS_EMBEDDING_INIT=class_embeddings.npz
export CLASS_EMBEDDING_TRAINABLE=false
# --- token-scale fixes (the point of v2) ---
# Every segment now leaves its own LayerNorm at sqrt(768)=27.71 instead of
# entering the encoder at video 6.25 / location 55.14 / obb 61.73.
export TOKEN_LAYERNORM=true
# One [5, H] marker table covering VIDEO too (the only modality that had none),
# added AFTER the segment LayerNorm so a marker's norm is an absolute magnitude
# against a normalised token rather than a share of it.
export TYPE_EMBEDDINGS=unified
# 0.5 puts each marker at 13.8 = 44.6% of its token, which is the share the obb
# marker already had in the reference runs (27.01 inside a 61.59 token = 44%).
# Measured cost: within-segment video token cosine 0.206 -> 0.366. Scale 1.0 would
# make markers louder than they have ever been (70.6%) at a cosine of 0.604.
export TYPE_EMBEDDING_SCALE="${TYPE_EMBEDDING_SCALE:-0.5}"
# The OWLv2 class code sat at 0.71 against 47.8 / 27.7 / 27.5 for the position,
# volume and marker terms of the same OBB token, and lr 1e-5 never moved it -- so
# swapping an OBB's class rotated its token by under 1.3 degrees. This normalises
# the projected vectors and scales them by a learned gain (init sqrt(768)),
# leaving WHICH classes are close to which completely unchanged.
export CLASS_EMBEDDING_GAIN=true
export CLASS_EMBEDDING_GAIN_INIT="${CLASS_EMBEDDING_GAIN_INIT:--1.0}"
# Temporal encoding, finally audible. It is added to the per-second modalities
# (video / location / gaze) AFTER the segment LayerNorm and after the marker, so
# its magnitude is set entirely by the gain -- cos^2+sin^2=1 means time_W_r sets
# frequencies and can never set amplitude.
#   GAIN=fixed  pins the code at norm 0.707, i.e. 2.3% of a token. That is why the
#               earlier te10 sweep measured as a no-op; do not use it here.
#   GAIN=learned gives it a trainable scalar in trainer.FAST_PARAMS (lr 1e-3),
#               init 8.0 = ~26% of a 30.8-norm token.
export TIME_ENCODING="${TIME_ENCODING:-fourier}"
export TIME_ENCODING_GAIN="${TIME_ENCODING_GAIN:-learned}"
export TIME_ENCODING_GAIN_INIT="${TIME_ENCODING_GAIN_INIT:-8.0}"
export WANDB_PROJECT="${WANDB_PROJECT:-OBB_tokens_fixtures_v2}"

for TEST_PID in $FOLDS; do
    # Train on every participant except the held-out one; no val split, so the
    # test set drives eval and best-model selection -- exactly as the four
    # reference runs did (they logged "[no-val] ... using the TEST set").
    export TEST_PARTICIPANTS="$TEST_PID"
    export TRAIN_PARTICIPANTS=$(echo "$ALL" | tr ',' '\n' | grep -v "^${TEST_PID}$" | paste -sd,)
    export VAL_PARTICIPANTS=""

    # 'tlnon' replaces the reference arm's 'tlnoff': TOKEN_LAYERNORM is true here.
    # The marker scale is interpolated rather than hardcoded so that overriding
    # TYPE_EMBEDDING_SCALE cannot leave the run name claiming the wrong value.
    # Both the marker scale and the time encoding are interpolated rather than
    # hardcoded, so overriding either cannot leave the run name claiming the wrong
    # value. NOTE the 8 runs submitted 2026-09-01 predate the te$TIME_ENCODING
    # suffix and are named plain '_te'; those are all TIME_ENCODING=fourier.
    # max_wavelength is appended ONLY when sinusoidal AND explicitly overridden,
    # so the four _tesinusoidal runs of 2026-09-01 (which used the 0.0 -> 2T=60
    # default) keep their existing names and stay comparable. Without this a
    # different wavelength would produce a name identical to theirs.
    MW_SUFFIX=""
    if [ "$TIME_ENCODING" = "sinusoidal" ] && \
       [ "$TIME_ENCODING_MAX_WAVELENGTH" != "0.0" ] && \
       [ "$TIME_ENCODING_MAX_WAVELENGTH" != "0" ]; then
        MW_SUFFIX="_mw${TIME_ENCODING_MAX_WAVELENGTH}"
    fi
    NAME="loocv_${TEST_PID}_vlo_vq_fourier_factorised_cls_prior_pe_locjoint_tlnon_e1_iou_fixt250_typeemb${TYPE_EMBEDDING_SCALE}_clsgain_te${TIME_ENCODING}${MW_SUFFIX}"
    echo "--- $NAME"
    echo "    train=$TRAIN_PARTICIPANTS test=$TEST_PARTICIPANTS"
    echo "    cap=$MAX_OBB_TOKENS prune=$OBB_VOXEL_GRID_PRUNE fixtures=$FIXTURE_OBBS" \
         "iou>$FIXTURE_IOU_THRESH cos>$FIXTURE_COS_THRESH" \
         "ln=$TOKEN_LAYERNORM type_emb=$TYPE_EMBEDDINGS@$TYPE_EMBEDDING_SCALE class_gain=$CLASS_EMBEDDING_GAIN" \
         "time_enc=$TIME_ENCODING/$TIME_ENCODING_GAIN@$TIME_ENCODING_GAIN_INIT"
    ./submit_train.sh "$NAME"
done
