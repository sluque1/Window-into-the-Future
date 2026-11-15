#!/bin/bash
# Snapshot the current source into a timestamped directory and submit the SLURM
# job from that snapshot, so edits to the live tree can't affect a running job.
#
# Usage: ./submit_train.sh <job_suffix>
# Example: ./submit_train.sh future180_obs30
#
# Hyperparameters are passed through the environment and read by train_sbatch.sh.
# Examples:
#   OBS_TIME=30 FUTURE_TIME=60  ./submit_train.sh future60
#   FUTURE_TIME=180 NUM_LAYERS=4 ./submit_train.sh ft180_nl4
#   TRAIN_N=2000 NUM_EPOCHS=2    ./submit_train.sh smoke
#
# PARTITION, if set, overrides sbatch.sh's #SBATCH --partition (e.g.
# PARTITION=medium_gpua100 ./submit_train.sh on_a100).
#
# TIME_LIMIT, if set, overrides sbatch.sh's #SBATCH --time (default 06:00:00).
# It must fit the partition's own cap, which differs: medium_gpuh200 and
# medium_gpua100 allow 6h10m, gpua100 allows 24h10m (sinfo -o "%P %l").
#   PARTITION=gpua100 TIME_LIMIT=24:00:00 ./submit_train.sh long_run

if [ "$1" = "" ]; then
    echo "Usage: $0 <job_suffix>   (hyperparameters via env: OBS_TIME, FUTURE_TIME, ANTICIP_TIME, LR, NUM_LAYERS, MODALITY, BATCH_SIZE, NUM_EPOCHS, TRAIN/VAL/TEST_PARTICIPANTS, TRAIN/VAL/TEST_N)"
    exit 1
fi

export JOB_NAME=$(date '+%Y-%m-%d_%H-%M-%S_')$1
export SNAPSHOT_BASE="${SNAPSHOT_BASE:-/data/video_datasets/HD-EPIC/snapshots}"
SNAPSHOT_DIR="$SNAPSHOT_BASE/$JOB_NAME"
mkdir -p "$SNAPSHOT_DIR/out" "$SNAPSHOT_DIR/err"

cp train.py       "$SNAPSHOT_DIR/"
cp model.py       "$SNAPSHOT_DIR/"
cp decoder.py     "$SNAPSHOT_DIR/"
cp loader.py      "$SNAPSHOT_DIR/"
cp boxer_env.py   "$SNAPSHOT_DIR/"
cp iou_fusion.py         "$SNAPSHOT_DIR/"   # FUSION=iou
cp fixture_obbs.py       "$SNAPSHOT_DIR/"   # FIXTURE_OBBS / OBB_VOXEL_GRID_PRUNE
# fixture_obbs reads new_voxel_centers/ and class_embeddings.npz; the grids are
# passed by absolute path (VOXEL_CENTERS_DIR) but the meshes live under
# /data/video_datasets/HD-EPIC/Digital-Twin, also absolute. Nothing else to copy.
# boxer_env.CLASS_EMBEDDINGS_PATH resolves next to the module, so the snapshot
# needs its own copy or FUSION=iou raises at the first window. 2.7MB.
cp class_embeddings.npz  "$SNAPSHOT_DIR/"
cp trainer.py     "$SNAPSHOT_DIR/"
cp trajectory.py         "$SNAPSHOT_DIR/"
cp trajectory_tokens.py  "$SNAPSHOT_DIR/"
cp gaze.py               "$SNAPSHOT_DIR/"
cp slam_utils.py         "$SNAPSHOT_DIR/"
cp object_class_vocab.py "$SNAPSHOT_DIR/"
cp -r data               "$SNAPSHOT_DIR/"   # object_class_vocab.py's boxer_class_vocab.json
cp train_sbatch.sh "$SNAPSHOT_DIR/sbatch.sh"

cd "$SNAPSHOT_DIR"

# Forward all hyperparameter env vars to the batch job.
#
# --propagate=NONE is NOT optional. The cluster runs PropagateResourceLimits=ALL,
# so sbatch copies the SUBMITTING SHELL's rlimits onto the compute node. Login
# nodes apply /etc/security/limits.d/90-hpc-login-node.conf to
# @grp_hpcclusterbocconiusers: soft cpu 10 min / hard cpu 30 min. A 3-hour
# training run submitted from such a shell inherits RLIMIT_CPU=600s and is killed
# with SIGXCPU at ~12.5 minutes -- episode building alone costs 5:33 of it.
#
# This bit on 2026-08-19 (jobs 631194-631206, all 12 dead at the same wall time)
# and not before ONLY because older interactive sessions predate the limits file
# and still carry cpu=unlimited. So the failure depends on which shell ran sbatch,
# which is exactly the kind of thing that looks like a code or cluster change and
# is neither. Raising `ulimit -t` cannot fix it: the hard cap is 1800s.
# DEPENDENCY, if set, is passed to sbatch --dependency. Needed when a run
# consumes something another job is still producing -- e.g. FUSION=iou runs
# gated on the fused-OBB cache array:
#   DEPENDENCY=afterok:637485 FUSION=iou FUSED_CACHE=true ./submit_train.sh iou_run
# Without it such a run starts immediately and the loader refuses, since the
# cache does not yet cover the split.
# EXCLUDE, if set, is passed to sbatch --exclude to keep a run off named nodes:
#   EXCLUDE=gnode12 ./submit_train.sh retry_off_bad_node
sbatch --propagate=NONE --export=ALL ${PARTITION:+--partition="$PARTITION"} \
       ${SLURM_ACCOUNT:+--account="$SLURM_ACCOUNT"} \
       ${TIME_LIMIT:+--time="$TIME_LIMIT"} \
       ${EXCLUDE:+--exclude="$EXCLUDE"} \
       ${DEPENDENCY:+--dependency="$DEPENDENCY"} -J "$JOB_NAME" sbatch.sh
echo "Submitted snapshot: $SNAPSHOT_DIR"
