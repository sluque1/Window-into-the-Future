#!/bin/bash
# SLURM batch script for FIction-HD-EPIC v2 training.
#
# Do NOT invoke directly with `sbatch`. Submit via submit_train.sh, which
# snapshots the source into a timestamped directory first.
#
# v2 builds episodes on the fly from the runtime OBS_TIME / FUTURE_TIME /
# ANTICIP_TIME, so changing those needs no offline preprocessing.
#
# Default splits: train P01-P06, val P07, test P08-P09.

#SBATCH --job-name="fiction_hd_epic_v2_train"   # Overridden by `sbatch -J <name>`
# SLURM does not expand environment variables in #SBATCH lines, so there is no
# portable way to template this. Pass your account on the command line instead --
#   sbatch --account=<acct> ...   (submit_train.sh forwards $SLURM_ACCOUNT for you)
# -- or uncomment and hard-code it:
##SBATCH --account=<your-slurm-account>
#SBATCH --partition=medium_gpuh200
#SBATCH --gpus=1
#SBATCH --mem=96G
#SBATCH --output=out/%x_%j.out
#SBATCH --error=err/%x_%j.err
#SBATCH --time=06:00:00

set -euo pipefail

echo "Job started: $(date)"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Snapshot dir (cwd): $(pwd)"

source "${CONDA_BASE:-$HOME/miniconda3}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV:-egovlpv2-pip}"
echo "Python: $(which python) | $(python --version)"

EXPT_NAME="${SLURM_JOB_NAME:-fiction_hd_epic_v2_train}"

# --- Iteration / model knobs (override via env when submitting) ---
LR="${LR:-1e-5}"
# HF TrainingArguments schedule. 'linear'/0.0 (both defaults) reproduce every
# run before this existed: LR decays linearly to 0 over the whole run, no
# warmup. Applies identically to the FAST_PARAMS group in trainer.py.
LR_SCHEDULER_TYPE="${LR_SCHEDULER_TYPE:-linear}"
WARMUP_RATIO="${WARMUP_RATIO:-0.0}"
NUM_LAYERS="${NUM_LAYERS:-2}"
MODALITY="${MODALITY:-video,location,env}"
NUM_EPOCHS="${NUM_EPOCHS:-10}"
# Eval + checkpoint cadence as a fraction of total training steps
# (0.05 with 5 epochs = every 0.25 epoch).
EVAL_STEPS="${EVAL_STEPS:-0.1}"
BATCH_SIZE="${BATCH_SIZE:-16}"
# Voxel-grid extent source: boxer_grid (default) or detic_grid (Detic OBBs
# define the extent; env content stays Boxer).
ENV_SOURCE="${ENV_SOURCE:-boxer_grid}"
# If set, use fixed per-participant voxel grids from
# <dir>/<pid>_voxel_centers.npy instead of deriving the grid per-video
# (ENV_SOURCE is then ignored; env content stays Boxer).
VOXEL_CENTERS_DIR="${VOXEL_CENTERS_DIR:-}"
# Skip videos shorter than this many seconds before building any episodes for
# them (cheap durations.txt lookup, done before SLAM/Boxer/grid work).
MIN_VIDEO_DURATION="${MIN_VIDEO_DURATION:-0}"
# Drop episodes with no valid SLAM location second in the observation window.
# Set to true when MODALITY=location alone (all-padding windows -> NaN).
REQUIRE_VALID_LOCATION="${REQUIRE_VALID_LOCATION:-false}"
# Keep only episodes whose prediction window holds a GT count in these inclusive
# "lo,hi" ranges (both must pass). Empty = no filter. Changes the positive rate,
# so a filtered PR-AUC is not comparable with an unfiltered one.
GT_INTERACTIONS_RANGE="${GT_INTERACTIONS_RANGE:-}"
GT_VOXELS_RANGE="${GT_VOXELS_RANGE:-}"

# How the per-window Boxer detections are clustered into fused boxes. Feeds BOTH
# the env grid and the OBB tokens -- they share _window_fused_boxes.
#   grid (default, legacy) -- snap centres to a 0.5m cell, group by (name, cell).
#   iou -- link detections overlapping in 3D whose classes are close in OWLv2
#          space, then chain the links (boxer_env.py + iou_fusion.py).
# 'iou' costs ~6.4 s/window against ~0.04 s and a split holds ~28,800 windows, so
# it REQUIRES the precomputed per-video cache. Build it once with
#   ./precompute_fused_obbs_sbatch.sh
# then set FUSED_CACHE=true. Without it the loader raises rather than spending
# ~51 h fusing in __init__ and dying on the wall clock.
FUSION="${FUSION:-grid}"
FUSED_CACHE="${FUSED_CACHE:-false}"
# Minimum Boxer detection probability before fusion. -1 picks per strategy:
# 0.5 for grid (what every earlier run used) and 0.6 for iou.
BOXER_CONF_THRESHOLD="${BOXER_CONF_THRESHOLD:--1.0}"
# --- Ablation toggles (each independent; see boxer_env.py/model.py/trajectory_tokens.py) ---
ENV_ORPHAN_FALLBACK="${ENV_ORPHAN_FALLBACK:-true}"
TRAJECTORY_SIMPLE="${TRAJECTORY_SIMPLE:-false}"
# Composition of a trajectory token (see model.py's trajectory block):
#   mapper (legacy)  -- trajectory_mapper(raw features) + class + both endpoints.
#   end_distance     -- class + pos(end voxel) + distance bin + trajectory_type,
#                       no MLP and NO time term. Needs TRAJECTORY_SIMPLE=true.
TRAJECTORY_TOKEN="${TRAJECTORY_TOKEN:-mapper}"
# Cap on trajectory tokens per episode. 16 is what every earlier trajectory run
# used (hardcoded in loader.py until this flag existed).
MAX_TRAJECTORY_TOKENS="${MAX_TRAJECTORY_TOKENS:-16}"
# Seed class_embedding from OWLv2 text vectors instead of a random table. Empty =
# legacy. Shared table, so it changes env, obb AND trajectory tokens alike.
CLASS_EMBEDDING_INIT="${CLASS_EMBEDDING_INIT:-}"
CLASS_EMBEDDING_TRAINABLE="${CLASS_EMBEDDING_TRAINABLE:-false}"
# Drop tracks still moving at obs_end rather than clipping them to the window
# (clipping interpolates toward their future endpoint -- see trajectory_tokens.py).
TRAJECTORY_EXCLUDE_UNFINISHED="${TRAJECTORY_EXCLUDE_UNFINISHED:-false}"
ENV_POSITIONAL_ENCODING="${ENV_POSITIONAL_ENCODING:-false}"
ENV_CLASS_AVERAGING="${ENV_CLASS_AVERAGING:-false}"
ENV_MAX_CLASSES_PER_VOXEL="${ENV_MAX_CLASSES_PER_VOXEL:-4}"
# Single-winner pass first, then average in only the boxes it missed:
# off | merge (join the nearest voxel regardless) | empty_only (only empty voxels).
ENV_FALLBACK_AVERAGING="${ENV_FALLBACK_AVERAGING:-off}"
# Gaze modality (add 'gaze' to MODALITY to actually feed it to the model).
GAZE_MAX_HZ="${GAZE_MAX_HZ:-0}"
GAZE_TIME_OFFSET_S="${GAZE_TIME_OFFSET_S:-0.1}"
# Clock that defines video second 0 for the device-clock inputs (location + env):
# trajectory (legacy, ~1s late) | gaze (corrected, same anchor gaze.py uses).
TIME_ANCHOR="${TIME_ANCHOR:-trajectory}"
# Cap on OBB tokens per window for MODALITY containing 'obb' (see
# boxer_env.MAX_OBB_TOKENS for the measured distribution behind the default).
MAX_OBB_TOKENS="${MAX_OBB_TOKENS:-250}"
# --- Voxel-grid prune + Digital-Twin fixture OBBs (fixture_obbs.py) ---
# OBB_VOXEL_GRID_PRUNE: drop fused OBBs whose centre is outside the participant's
#   oriented voxel grid (1.71% of all boxes). boxer_env._nearest_voxel_ijk is an
#   argmin over the 4096 centres, so without this it snaps a distant box onto a
#   boundary voxel instead of rejecting it. Needs VOXEL_CENTERS_DIR.
# FIXTURE_OBBS: also drop each surviving box that a Digital-Twin fixture already
#   covers (3D IoU > thresh AND class cosine > thresh against that same fixture),
#   then append that participant's own fixture boxes. Implies the prune.
OBB_VOXEL_GRID_PRUNE="${OBB_VOXEL_GRID_PRUNE:-false}"
FIXTURE_OBBS="${FIXTURE_OBBS:-false}"
FIXTURE_IOU_THRESH="${FIXTURE_IOU_THRESH:-0.4}"
FIXTURE_COS_THRESH="${FIXTURE_COS_THRESH:-0.7}"
# Which sequence position is decoded to the voxel logits:
# token0 (legacy, the first visual token) | cls (a dedicated learned token).
READOUT="${READOUT:-token0}"
# Output-bias init: zero (legacy nn.Linear init) | prior (log-odds of the train
# split's measured positive rate -- the standard fix for the ~1:200 imbalance).
VOXEL_BIAS_INIT="${VOXEL_BIAS_INIT:-zero}"
# Output decoder: linear (legacy Linear(768->4096) on one token) | voxel_query
# (4096 coordinate queries cross-attend, one shared head -- see decoder.py).
DECODER="${DECODER:-linear}"
VOXEL_QUERY_TYPE="${VOXEL_QUERY_TYPE:-axis_sum}"   # axis_sum | fourier | learned
VOXEL_BIAS="${VOXEL_BIAS:-factorised}"             # factorised (48 params) | full (4096) | mlp
VOXEL_BIAS_MLP_HIDDEN="${VOXEL_BIAS_MLP_HIDDEN:-32,32}"   # VOXEL_BIAS=mlp only
# Spatial arms of the bias (submit_bias_sweep.sh). SCALE multiplies the field's
# deviation from the pooled prior: 0 = a plain scalar prior, 1 = as measured,
# -1 = inverted. LR: -1 keeps the bias in the normal group (where it moves 0.036
# at most over a full run, i.e. it is frozen in practice), 0 freezes it
# explicitly, >0 gives it its own group and makes it learnable for real.
VOXEL_BIAS_SCALE="${VOXEL_BIAS_SCALE:-1.0}"
VOXEL_BIAS_LR="${VOXEL_BIAS_LR:--1.0}"
# One value, or three comma-separated (sx,sy,sz) for a per-axis init -- the
# production grids are not cubic alike (z 13.3-14.6cm everywhere, x 16.8-41.7,
# y 12.6-46.6), so one scale resolves height far more finely than the floor axes.
VOXEL_QUERY_FOURIER_SCALE="${VOXEL_QUERY_FOURIER_SCALE:-10.0}"
# Positional code on the input tokens that carry a full (i,j,k) -- obb, env,
# trajectory endpoints. auto (default) follows VOXEL_QUERY_TYPE so the queries and
# the input tokens share one code; none keeps the index tables; fourier forces it.
POSITION_ENCODING="${POSITION_ENCODING:-auto}"
POSITION_ENCODING_MODE="${POSITION_ENCODING_MODE:-replace}"   # replace | add
POSITION_ENCODING_GAIN_INIT="${POSITION_ENCODING_GAIN_INIT:-48.0}"
VOXEL_QUERY_UNTIED="${VOXEL_QUERY_UNTIED:-false}"
LOSS="${LOSS:-bce}"
FOCAL_GAMMA="${FOCAL_GAMMA:-2.0}"
FOCAL_ALPHA="${FOCAL_ALPHA:-0.25}"
POS_WEIGHT="${POS_WEIGHT:--1.0}"
DECODER_ITERS="${DECODER_ITERS:-1}"
# Location segment layout: axis (legacy, 3 tokens/second = 90) | joint (1/second = 30,
# the whole position through the shared positional code).
LOCATION_TOKENS="${LOCATION_TOKENS:-axis}"
# Regularisation. WEIGHT_DECAY has been 0.0 (HF default) for every run so far.
# ENCODER_DROPOUT 0.1 is PyTorch's default and unchanged; DECODER_DROPOUT 0.0 means
# the 4096 queries attend with none. OBB_TOKEN_DROPOUT hides a share of the ~250
# boxes each train step -- the regulariser aimed at kitchen identification itself.
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0}"
ENCODER_DROPOUT="${ENCODER_DROPOUT:-0.1}"
DECODER_DROPOUT="${DECODER_DROPOUT:-0.0}"
OBB_TOKEN_DROPOUT="${OBB_TOKEN_DROPOUT:-0.0}"
DECODER_HEADS="${DECODER_HEADS:-8}"
DECODER_FFN="${DECODER_FFN:-false}"
DECODER_FFN_DIM="${DECODER_FFN_DIM:-3072}"
# Decoder residual: raw (legacy, h = q + attn_out; the query outweighs the
# retrieved evidence 50:1) | normed (ln_q(q) + attn_out, 25:1) | balanced
# (ln_q(q) + ln_a(attn_out), both at sqrt(D)). See decoder.py.
DECODER_RESIDUAL="${DECODER_RESIDUAL:-raw}"
# Temporal encoding on the per-second tokens (video/location/gaze):
# none (legacy, the 30 seconds are an unordered set) | fourier.
TIME_ENCODING="${TIME_ENCODING:-none}"
TIME_ENCODING_SCALE="${TIME_ENCODING_SCALE:-10.0}"
# --time_encoding sinusoidal only. 0 = 2*OBS_TIME. Not the literature's 10000, which
# is tuned for sequences of thousands and is dead over a 30-slot window (measured
# cosine never falls below 0.567; dynamic range 0.406 against 0.898 at 2T).
TIME_ENCODING_MAX_WAVELENGTH="${TIME_ENCODING_MAX_WAVELENGTH:-0.0}"
# Magnitude of that code: fixed (legacy, norm 0.707 = 2.6% of a location token,
# which is why the te10 sweep was a no-op) | learned (unit-norm x trainable scalar).
TIME_ENCODING_GAIN="${TIME_ENCODING_GAIN:-fixed}"
TIME_ENCODING_GAIN_INIT="${TIME_ENCODING_GAIN_INIT:-8.0}"
# LayerNorm each segment before concatenation. Without it the segments enter the
# encoder at norms 5.9 (video) / 27.7 (location) / 66.9 (obb) / 0.56 (cls), set by
# whether nn.Linear or nn.Embedding happened to build them.
TOKEN_LAYERNORM="${TOKEN_LAYERNORM:-false}"
# Modality markers ("this token arrived via video / location / obb / ...").
#   separate (legacy) -- four named parameters, and NO marker at all for video.
#   unified           -- one [5, H] table covering video too, added AFTER the
#                        segment LayerNorm instead of inside it, so a marker's
#                        norm is an absolute magnitude against a sqrt(768) token.
# TYPE_EMBEDDING_SCALE 1.0 puts each marker at 27.7 (= a LayerNorm'd token);
# 0.02 would put it at 0.55, i.e. 2% of the token.
TYPE_EMBEDDINGS="${TYPE_EMBEDDINGS:-separate}"
TYPE_EMBEDDING_SCALE="${TYPE_EMBEDDING_SCALE:-1.0}"
# Magnitude of the OWLv2 class code. Without it the projected class vectors sit at
# 0.71 against 47.8 (position) / 27.7 (volume) / 27.5 (marker) in the same OBB
# token, and lr 1e-5 never moves them -- so the model can barely tell a kettle
# from a sink. -1 init = sqrt(768) = 27.71. CLASS_EMBEDDING_INIT only.
CLASS_EMBEDDING_GAIN="${CLASS_EMBEDDING_GAIN:-false}"
CLASS_EMBEDDING_GAIN_INIT="${CLASS_EMBEDDING_GAIN_INIT:--1.0}"
# Layout of the gaze stream, the analogue of LOCATION_TOKENS: axis (legacy, three
# per-axis scalar streams, 90 tokens -- and the reason gaze sits outside
# POSITION_ENCODING's scope) | joint (one token/second through
# _voxel_pos_embedding, 30 tokens, so gaze shares the one spatial vocabulary).
GAZE_TOKENS="${GAZE_TOKENS:-axis}"
# WandB project. Default keeps earlier runs together; override for a separate
# line of work (e.g. FIction-HD-EPIC-OBB_tokens).
# Weights & Biases. Set USE_WANDB=false to train without it -- otherwise wandb
# needs an API key on the compute node and will stall or fail without one.
USE_WANDB="${USE_WANDB:-true}"
WANDB_PROJECT="${WANDB_PROJECT:-FIction-HD-EPIC-v2}"

# --- Runtime episode hyperparameters (the point of v2) ---
OBS_TIME="${OBS_TIME:-30}"
FUTURE_TIME="${FUTURE_TIME:-180}"
ANTICIP_TIME="${ANTICIP_TIME:-5}"

# --- Splits + caps ---
TRAIN_PARTICIPANTS="${TRAIN_PARTICIPANTS:-P01,P02,P03,P04,P05,P06}"
# Use ${VAR-default} (not :-) so an explicit empty VAL_PARTICIPANTS="" is honored
# (no-val mode: test set drives eval + best-model selection).
VAL_PARTICIPANTS="${VAL_PARTICIPANTS-P07}"
TEST_PARTICIPANTS="${TEST_PARTICIPANTS:-P08,P09}"
TRAIN_N="${TRAIN_N:-0}"
VAL_N="${VAL_N:-0}"
TEST_N="${TEST_N:-0}"

echo "LR=$LR LR_SCHEDULER_TYPE=$LR_SCHEDULER_TYPE WARMUP_RATIO=$WARMUP_RATIO NUM_LAYERS=$NUM_LAYERS MODALITY=$MODALITY ENV_SOURCE=$ENV_SOURCE VOXEL_CENTERS_DIR=${VOXEL_CENTERS_DIR:-<none>}"
echo "MIN_VIDEO_DURATION=$MIN_VIDEO_DURATION REQUIRE_VALID_LOCATION=$REQUIRE_VALID_LOCATION"
echo "GT_INTERACTIONS_RANGE=${GT_INTERACTIONS_RANGE:-<none>} GT_VOXELS_RANGE=${GT_VOXELS_RANGE:-<none>}"
echo "ENV_ORPHAN_FALLBACK=$ENV_ORPHAN_FALLBACK TRAJECTORY_SIMPLE=$TRAJECTORY_SIMPLE ENV_POSITIONAL_ENCODING=$ENV_POSITIONAL_ENCODING ENV_CLASS_AVERAGING=$ENV_CLASS_AVERAGING ENV_MAX_CLASSES_PER_VOXEL=$ENV_MAX_CLASSES_PER_VOXEL"
echo "TRAJECTORY_EXCLUDE_UNFINISHED=$TRAJECTORY_EXCLUDE_UNFINISHED"
echo "TRAJECTORY_TOKEN=$TRAJECTORY_TOKEN MAX_TRAJECTORY_TOKENS=$MAX_TRAJECTORY_TOKENS"
echo "CLASS_EMBEDDING_INIT=${CLASS_EMBEDDING_INIT:-<none>} CLASS_EMBEDDING_TRAINABLE=$CLASS_EMBEDDING_TRAINABLE"
echo "ENV_FALLBACK_AVERAGING=$ENV_FALLBACK_AVERAGING"
echo "GAZE_MAX_HZ=$GAZE_MAX_HZ GAZE_TIME_OFFSET_S=$GAZE_TIME_OFFSET_S"
echo "TIME_ANCHOR=$TIME_ANCHOR MAX_OBB_TOKENS=$MAX_OBB_TOKENS"
echo "OBB_VOXEL_GRID_PRUNE=$OBB_VOXEL_GRID_PRUNE FIXTURE_OBBS=$FIXTURE_OBBS FIXTURE_IOU_THRESH=$FIXTURE_IOU_THRESH FIXTURE_COS_THRESH=$FIXTURE_COS_THRESH"
echo "FUSION=$FUSION FUSED_CACHE=$FUSED_CACHE BOXER_CONF_THRESHOLD=$BOXER_CONF_THRESHOLD"
echo "READOUT=$READOUT VOXEL_BIAS_INIT=$VOXEL_BIAS_INIT VOXEL_BIAS_SCALE=$VOXEL_BIAS_SCALE VOXEL_BIAS_LR=$VOXEL_BIAS_LR"
echo "TIME_ENCODING=$TIME_ENCODING TIME_ENCODING_SCALE=$TIME_ENCODING_SCALE TIME_ENCODING_MAX_WAVELENGTH=$TIME_ENCODING_MAX_WAVELENGTH"
echo "TIME_ENCODING_GAIN=$TIME_ENCODING_GAIN TIME_ENCODING_GAIN_INIT=$TIME_ENCODING_GAIN_INIT TOKEN_LAYERNORM=$TOKEN_LAYERNORM"
echo "TYPE_EMBEDDINGS=$TYPE_EMBEDDINGS TYPE_EMBEDDING_SCALE=$TYPE_EMBEDDING_SCALE CLASS_EMBEDDING_GAIN=$CLASS_EMBEDDING_GAIN CLASS_EMBEDDING_GAIN_INIT=$CLASS_EMBEDDING_GAIN_INIT"
echo "LOSS=$LOSS FOCAL_GAMMA=$FOCAL_GAMMA FOCAL_ALPHA=$FOCAL_ALPHA POS_WEIGHT=$POS_WEIGHT DECODER_ITERS=$DECODER_ITERS"
echo "DECODER=$DECODER VOXEL_QUERY_TYPE=$VOXEL_QUERY_TYPE VOXEL_BIAS=$VOXEL_BIAS VOXEL_BIAS_MLP_HIDDEN=$VOXEL_BIAS_MLP_HIDDEN FOURIER_SCALE=$VOXEL_QUERY_FOURIER_SCALE DECODER_HEADS=$DECODER_HEADS DECODER_FFN=$DECODER_FFN DECODER_RESIDUAL=$DECODER_RESIDUAL"
echo "POSITION_ENCODING=$POSITION_ENCODING MODE=$POSITION_ENCODING_MODE GAIN_INIT=$POSITION_ENCODING_GAIN_INIT LOCATION_TOKENS=$LOCATION_TOKENS GAZE_TOKENS=$GAZE_TOKENS VOXEL_QUERY_UNTIED=$VOXEL_QUERY_UNTIED"
echo "WEIGHT_DECAY=$WEIGHT_DECAY ENCODER_DROPOUT=$ENCODER_DROPOUT DECODER_DROPOUT=$DECODER_DROPOUT OBB_TOKEN_DROPOUT=$OBB_TOKEN_DROPOUT"
echo "WANDB_PROJECT=$WANDB_PROJECT"
echo "NUM_EPOCHS=$NUM_EPOCHS BATCH_SIZE=$BATCH_SIZE EVAL_STEPS=$EVAL_STEPS"
echo "OBS_TIME=$OBS_TIME FUTURE_TIME=$FUTURE_TIME ANTICIP_TIME=$ANTICIP_TIME"
echo "TRAIN=$TRAIN_PARTICIPANTS VAL=$VAL_PARTICIPANTS TEST=$TEST_PARTICIPANTS"
echo "TRAIN_N=$TRAIN_N VAL_N=$VAL_N TEST_N=$TEST_N"

# ---------------------------------------------------------------------------
# Pre-flight: show where the raw inputs come from
# ---------------------------------------------------------------------------
FEATURES_BASE="/data/video_datasets/HD-EPIC/fiction_hd_epic/intermediate/extracted_rgb_features_EgoVLPv2"
BOXER_BASE="/data/video_datasets/HD-EPIC/boxer_obbs"
SLAM_BASE="/data/video_datasets/HD-EPIC/SLAM-and-Gaze"
ASSOC="/data/video_datasets/HD-EPIC/hd-epic-annotations/scene-and-object-movements/assoc_info.json"
echo ""
echo "--- Inputs ---"
printf "  features : %s (%d .pt)\n" "$FEATURES_BASE" "$(find "$FEATURES_BASE" -name '*_features.pt' 2>/dev/null | wc -l)"
printf "  boxer    : %s (%d videos)\n" "$BOXER_BASE" "$(ls "$BOXER_BASE" 2>/dev/null | wc -l)"
printf "  slam     : %s\n" "$SLAM_BASE"
printf "  gt assoc : %s [%s]\n" "$ASSOC" "$([ -f "$ASSOC" ] && echo OK || echo MISSING)"
echo "  outputs  : $(pwd)/results/${EXPT_NAME}/   WandB: ${WANDB_PROJECT}/${EXPT_NAME}"
echo ""

set +e
# -u (unbuffered stdout) is not cosmetic. Python block-buffers stdout when it is
# a file, so the loader's startup report sits in an 8KB buffer and the .out file
# lags minutes-to-hours behind the job. Job 643521 (2026-08-31, P04 fold) stalled
# in dataset construction and the last flushed line was ~100 minutes stale, which
# made it impossible to see WHERE it stalled. Costs nothing; keep it.
python -u train.py \
    --expt_name              "$EXPT_NAME" \
    --learning_rate          "$LR" \
    --lr_scheduler_type      "$LR_SCHEDULER_TYPE" \
    --warmup_ratio           "$WARMUP_RATIO" \
    --num_transformer_layers "$NUM_LAYERS" \
    --input_modality         "$MODALITY" \
    --env_source             "$ENV_SOURCE" \
    --voxel_centers_dir      "$VOXEL_CENTERS_DIR" \
    --min_video_duration     "$MIN_VIDEO_DURATION" \
    --require_valid_location "$REQUIRE_VALID_LOCATION" \
    --gt_interactions_range  "$GT_INTERACTIONS_RANGE" \
    --gt_voxels_range        "$GT_VOXELS_RANGE" \
    --env_orphan_fallback    "$ENV_ORPHAN_FALLBACK" \
    --trajectory_simple      "$TRAJECTORY_SIMPLE" \
    --trajectory_exclude_unfinished "$TRAJECTORY_EXCLUDE_UNFINISHED" \
    --trajectory_token       "$TRAJECTORY_TOKEN" \
    --max_trajectory_tokens  "$MAX_TRAJECTORY_TOKENS" \
    --class_embedding_init   "$CLASS_EMBEDDING_INIT" \
    --class_embedding_trainable "$CLASS_EMBEDDING_TRAINABLE" \
    --env_positional_encoding "$ENV_POSITIONAL_ENCODING" \
    --env_class_averaging    "$ENV_CLASS_AVERAGING" \
    --env_max_classes_per_voxel "$ENV_MAX_CLASSES_PER_VOXEL" \
    --env_fallback_averaging "$ENV_FALLBACK_AVERAGING" \
    --gaze_max_hz            "$GAZE_MAX_HZ" \
    --gaze_time_offset_s     "$GAZE_TIME_OFFSET_S" \
    --time_anchor            "$TIME_ANCHOR" \
    --fusion                 "$FUSION" \
    --fused_cache            "$FUSED_CACHE" \
    --boxer_conf_threshold   "$BOXER_CONF_THRESHOLD" \
    --max_obb_tokens         "$MAX_OBB_TOKENS" \
    --obb_voxel_grid_prune   "$OBB_VOXEL_GRID_PRUNE" \
    --fixture_obbs           "$FIXTURE_OBBS" \
    --fixture_iou_thresh     "$FIXTURE_IOU_THRESH" \
    --fixture_cos_thresh     "$FIXTURE_COS_THRESH" \
    --readout                "$READOUT" \
    --voxel_bias_init        "$VOXEL_BIAS_INIT" \
    --decoder                "$DECODER" \
    --voxel_query_type       "$VOXEL_QUERY_TYPE" \
    --voxel_bias             "$VOXEL_BIAS" \
    --voxel_bias_mlp_hidden  "$VOXEL_BIAS_MLP_HIDDEN" \
    --voxel_bias_scale       "$VOXEL_BIAS_SCALE" \
    --voxel_bias_lr          "$VOXEL_BIAS_LR" \
    --voxel_query_fourier_scale "$VOXEL_QUERY_FOURIER_SCALE" \
    --position_encoding      "$POSITION_ENCODING" \
    --position_encoding_mode "$POSITION_ENCODING_MODE" \
    --loss                   "$LOSS" \
    --focal_gamma            "$FOCAL_GAMMA" \
    --focal_alpha            "$FOCAL_ALPHA" \
    --pos_weight             "$POS_WEIGHT" \
    --decoder_iters          "$DECODER_ITERS" \
    --voxel_query_untied     "$VOXEL_QUERY_UNTIED" \
    --position_encoding_gain_init "$POSITION_ENCODING_GAIN_INIT" \
    --location_tokens        "$LOCATION_TOKENS" \
    --weight_decay           "$WEIGHT_DECAY" \
    --encoder_dropout        "$ENCODER_DROPOUT" \
    --decoder_dropout        "$DECODER_DROPOUT" \
    --obb_token_dropout      "$OBB_TOKEN_DROPOUT" \
    --decoder_heads          "$DECODER_HEADS" \
    --decoder_ffn            "$DECODER_FFN" \
    --decoder_ffn_dim        "$DECODER_FFN_DIM" \
    --decoder_residual       "$DECODER_RESIDUAL" \
    --time_encoding          "$TIME_ENCODING" \
    --time_encoding_scale    "$TIME_ENCODING_SCALE" \
    --time_encoding_max_wavelength "$TIME_ENCODING_MAX_WAVELENGTH" \
    --time_encoding_gain     "$TIME_ENCODING_GAIN" \
    --time_encoding_gain_init "$TIME_ENCODING_GAIN_INIT" \
    --token_layernorm        "$TOKEN_LAYERNORM" \
    --type_embeddings        "$TYPE_EMBEDDINGS" \
    --type_embedding_scale   "$TYPE_EMBEDDING_SCALE" \
    --class_embedding_gain   "$CLASS_EMBEDDING_GAIN" \
    --class_embedding_gain_init "$CLASS_EMBEDDING_GAIN_INIT" \
    --gaze_tokens            "$GAZE_TOKENS" \
    --batch_size             "$BATCH_SIZE" \
    --num_epochs             "$NUM_EPOCHS" \
    --eval_steps             "$EVAL_STEPS" \
    --observation_time       "$OBS_TIME" \
    --future_time            "$FUTURE_TIME" \
    --anticipation_time      "$ANTICIP_TIME" \
    --train_participants     "$TRAIN_PARTICIPANTS" \
    --val_participants       "$VAL_PARTICIPANTS" \
    --test_participants      "$TEST_PARTICIPANTS" \
    --train_n                "$TRAIN_N" \
    --val_n                  "$VAL_N" \
    --test_n                 "$TEST_N" \
    --use_static_env         true \
    --use_wandb              "$USE_WANDB" \
    --wandb_project          "$WANDB_PROJECT"
PY_RC=$?
set -e

echo "Job finished: $(date) (python exit=$PY_RC)"
exit $PY_RC
