"""
One plot grid per participant, one scatter per video_id.
x = anchor timestamp (pred_start = obs_end + anticipation_time), y = PR-AUC,
y in [0,1], x in [0, video_duration].
Red dashed line at 30s (obs boundary) and dotted at video_dur-future_time
(pred boundary), only drawn when they fall within the video's x range.

Reads predictions.pkl/labels.pkl from the best checkpoint of each LOOCV
gaze-modality snapshot (test participant = the held-out one, VAL_PARTICIPANTS
empty so the saved checkpoint's predictions/labels ARE the test-set eval --
see trainer.py's CustomTrainer.evaluate() and train.py's no-val branch), and
rebuilds the same HDEpicInteractionDataset (DS_KWARGS below, matching each
snapshot's actual train_sbatch.sh invocation -- see each snapshot's
out/*.out log) purely to read off video_id / obs_end / feature_path per
episode index; dataset order/content is otherwise untouched.

Usage:
  python analysis/scatter_per_video.py
"""
import glob, json, math, os, pickle, sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from sklearn.metrics import average_precision_score  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from loader import HDEpicInteractionDataset, FPS  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "out")

SNAPSHOT_DIRS = {
    "P04": "/data/video_datasets/HD-EPIC/snapshots/2026-07-30_20-06-14_loocv_P04_video_location_env_gaze",
    "P07": "/data/video_datasets/HD-EPIC/snapshots/2026-07-30_20-06-15_loocv_P07_video_location_env_gaze",
    "P08": "/data/video_datasets/HD-EPIC/snapshots/2026-07-30_20-06-16_loocv_P08_video_location_env_gaze",
}

# Matches these snapshots' actual train_sbatch.sh invocation (see each
# snapshot's out/*.out log): video,location,env,gaze modality, lr1e-5 nl2,
# obs30/future180/anticip5, gaze time-anchor, no orphan fallback.
VOXEL_CENTERS_DIR = os.path.join(os.path.dirname(HERE), "new_voxel_centers")
ANTICIPATION_TIME = 5
FUTURE_TIME = 180
DS_KWARGS = dict(observation_time=30, anticipation_time=ANTICIPATION_TIME,
                 future_time=FUTURE_TIME, static_env=True, env_source="boxer_grid",
                 voxel_centers_dir=VOXEL_CENTERS_DIR,
                 env_orphan_fallback=False, trajectory_simple=False,
                 trajectory_exclude_unfinished=False,
                 env_class_averaging=False, env_max_classes_per_voxel=4,
                 env_fallback_averaging="off",
                 gaze_max_hz=0.0, gaze_time_offset_s=0.1,
                 time_anchor="gaze", verbose=False)


def _best_checkpoint_dir(snapshot_dir):
    run_dirs = glob.glob(os.path.join(snapshot_dir, "results", "*"))
    if len(run_dirs) != 1:
        raise SystemExit(f"Expected exactly one run dir under {snapshot_dir}/results, "
                         f"found {len(run_dirs)}")
    run_dir = run_dirs[0]
    any_state = glob.glob(os.path.join(run_dir, "checkpoint-*", "trainer_state.json"))[0]
    with open(any_state) as f:
        state = json.load(f)
    return os.path.normpath(os.path.join(snapshot_dir, state["best_model_checkpoint"]))


def _video_length_seconds(feature_path, cache):
    if feature_path not in cache:
        feat = torch.load(feature_path, map_location="cpu")
        cache[feature_path] = (int(feat["frame_indices"][-1]) + FPS) / FPS
    return cache[feature_path]


def load_video_data(pid, feat_len_cache):
    """Return {video_id: {'xs': [...], 'ys': [...]}} for pid's scored test
    episodes (scored = both classes present in GT, matching
    ComputeMetricsWrapper -- the same episodes that feed the reported pr_auc)."""
    ckpt_dir = _best_checkpoint_dir(SNAPSHOT_DIRS[pid])
    with open(os.path.join(ckpt_dir, "predictions.pkl"), "rb") as f:
        preds = pickle.load(f)
    with open(os.path.join(ckpt_dir, "labels.pkl"), "rb") as f:
        labels = pickle.load(f)
    logits = preds[0] if isinstance(preds, tuple) else preds
    gt = labels["interaction"]
    preds_sigmoid = 1 / (1 + np.exp(-logits))

    ds = HDEpicInteractionDataset("test", participants=[pid], **DS_KWARGS)
    if len(ds) != gt.shape[0]:
        raise SystemExit(f"{pid}: dataset has {len(ds)} episodes but checkpoint has "
                         f"{gt.shape[0]} -- DS_KWARGS likely don't match this training run")

    vid_data = {}
    for i, datum in enumerate(ds.dataset):
        y_true = gt[i]
        if len(np.unique(y_true)) <= 1:
            continue  # not scored by ComputeMetricsWrapper -- excluded here too
        anchor_t = datum["obs_end"] + ANTICIPATION_TIME
        pr_auc = average_precision_score(y_true, preds_sigmoid[i])
        vid = datum["video_id"]
        vid_data.setdefault(vid, {"xs": [], "ys": [], "feature_path": datum["feature_path"]})
        vid_data[vid]["xs"].append(anchor_t)
        vid_data[vid]["ys"].append(pr_auc)
    return vid_data


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    feat_len_cache = {}

    for pid in sorted(SNAPSHOT_DIRS):
        vid_data = load_video_data(pid, feat_len_cache)
        videos   = sorted(vid_data.keys())
        n        = len(videos)

        ncols = math.ceil(math.sqrt(n))
        nrows = math.ceil(n / ncols)

        fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 2.8))
        axes = np.array(axes).flatten()

        for i, vid in enumerate(videos):
            ax  = axes[i]
            xs  = np.array(vid_data[vid]["xs"])
            ys  = np.array(vid_data[vid]["ys"])
            dur = _video_length_seconds(vid_data[vid]["feature_path"], feat_len_cache)

            ax.scatter(xs, ys, s=6, alpha=0.5, color="steelblue", linewidths=0)
            ax.set_xlim(0, dur)
            ax.set_ylim(0, 0.6)

            if dur > 30:
                ax.axvline(30, color="red", lw=1.0, ls="--")

            pred_line = dur - FUTURE_TIME
            if 0 < pred_line < dur:
                ax.axvline(pred_line, color="red", lw=1.0, ls=":")

            ax.set_title(vid[4:], fontsize=7, pad=2)
            ax.tick_params(labelsize=6)
            ax.grid(alpha=0.25)

        for j in range(n, len(axes)):
            axes[j].set_visible(False)

        fig.text(0.5, -0.01, "Anchor timestamp (s)", ha="center", fontsize=10)
        fig.text(-0.01, 0.5, "PR-AUC", va="center", rotation="vertical", fontsize=10)
        fig.suptitle(f"{pid} — PR-AUC vs anchor timestamp, per video", fontsize=12, y=1.01)
        plt.tight_layout()

        out = os.path.join(OUT_DIR, f"scatter_per_video_{pid}.png")
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"{pid}: {n} videos ({nrows}×{ncols} grid) → {out}")

    print("Done.")


if __name__ == "__main__":
    main()
