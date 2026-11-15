"""
Per-participant scatter: source-video length (x) vs. per-episode PR-AUC (y).

For each LOOCV run (one participant held out as TEST), reads the predictions/
labels dumped alongside the best checkpoint (see trainer.py's `_save`
override), recomputes PR-AUC per test episode (same filter as
train.py's ComputeMetricsWrapper: only episodes with both classes present in
the GT are scored -- these are the episodes actually taken into account when
the reported pr_auc metric was computed), and looks up which video each
episode's observation window was drawn from via the same loader.py used at
train time. Video length = duration of the extracted RGB feature track
(1 sample/sec), i.e. the length of the whole recording the episode's
observation window sits inside -- not the 30s window itself.

Usage:
  python analysis/plot_pr_auc_vs_video_length.py
"""

import argparse
import glob
import json
import os
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pickle  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from loader import HDEpicInteractionDataset, FPS  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DEF_OUT = os.path.join(HERE, "out", "pr_auc_vs_video_length.png")

VOXEL_CENTERS_DIR = os.path.join(os.path.dirname(HERE), "new_voxel_centers")

# One LOOCV snapshot per held-out participant (video,location,env,trajectory
# modality; lr1e5, nl2). Ds kwargs below mirror submit_loocv_lr1e5_nl2_traj.sh.
SNAPSHOT_DIRS = {
    "P01": "/data/video_datasets/HD-EPIC/snapshots/2026-07-21_02-01-38_loocv_P01_newvox_lr1e5_nl2_traj",
    "P02": "/data/video_datasets/HD-EPIC/snapshots/2026-07-21_02-01-39_loocv_P02_newvox_lr1e5_nl2_traj",
    "P03": "/data/video_datasets/HD-EPIC/snapshots/2026-07-21_02-01-39_loocv_P03_newvox_lr1e5_nl2_traj",
    "P04": "/data/video_datasets/HD-EPIC/snapshots/2026-07-21_02-01-40_loocv_P04_newvox_lr1e5_nl2_traj",
    "P05": "/data/video_datasets/HD-EPIC/snapshots/2026-07-21_02-01-40_loocv_P05_newvox_lr1e5_nl2_traj",
    "P06": "/data/video_datasets/HD-EPIC/snapshots/2026-07-21_02-01-41_loocv_P06_newvox_lr1e5_nl2_traj",
    "P07": "/data/video_datasets/HD-EPIC/snapshots/2026-07-21_02-01-41_loocv_P07_newvox_lr1e5_nl2_traj",
    "P08": "/data/video_datasets/HD-EPIC/snapshots/2026-07-21_02-01-46_loocv_P08_newvox_lr1e5_nl2_traj",
    "P09": "/data/video_datasets/HD-EPIC/snapshots/2026-07-21_02-01-46_loocv_P09_newvox_lr1e5_nl2_traj",
}

DS_KWARGS = dict(observation_time=30, anticipation_time=5, future_time=180,
                  static_env=True, env_source="boxer_grid",
                  voxel_centers_dir=VOXEL_CENTERS_DIR)

DOT_COLOR = "#2a78d6"


def _best_checkpoint_dir(snapshot_dir):
    """Resolve the best-checkpoint directory for a snapshot's single training run."""
    run_dirs = glob.glob(os.path.join(snapshot_dir, "results", "*"))
    if len(run_dirs) != 1:
        raise SystemExit(f"Expected exactly one run dir under {snapshot_dir}/results, "
                         f"found {len(run_dirs)}")
    run_dir = run_dirs[0]
    any_state = glob.glob(os.path.join(run_dir, "checkpoint-*", "trainer_state.json"))[0]
    with open(any_state) as f:
        best_rel = json.load(f)["best_model_checkpoint"]
    return os.path.normpath(os.path.join(snapshot_dir, best_rel))


def _load_eval_arrays(ckpt_dir):
    with open(os.path.join(ckpt_dir, "predictions.pkl"), "rb") as f:
        preds = pickle.load(f)
    with open(os.path.join(ckpt_dir, "labels.pkl"), "rb") as f:
        labels = pickle.load(f)
    logits = preds[0] if isinstance(preds, tuple) else preds
    return logits, labels["interaction"]


def _video_length_seconds(feature_path, cache):
    if feature_path not in cache:
        feat = torch.load(feature_path, map_location="cpu")
        cache[feature_path] = (int(feat["frame_indices"][-1]) + FPS) / FPS
    return cache[feature_path]


def _episode_records(pid):
    """List of {'video_id','feature_path','pr_auc'} for every scored test episode
    of pid (scored = has both classes in GT, matching ComputeMetricsWrapper)."""
    snapshot_dir = SNAPSHOT_DIRS[pid]
    ckpt_dir = _best_checkpoint_dir(snapshot_dir)
    logits, gt = _load_eval_arrays(ckpt_dir)

    ds = HDEpicInteractionDataset("test", participants=[pid], verbose=False, **DS_KWARGS)
    if len(ds) != logits.shape[0]:
        raise SystemExit(f"{pid}: dataset has {len(ds)} episodes but checkpoint has "
                         f"{logits.shape[0]} -- ds_kwargs likely don't match the training run")

    preds_sigmoid = 1 / (1 + np.exp(-logits))
    from sklearn.metrics import average_precision_score

    records = []
    for i, datum in enumerate(ds.dataset):
        y_true = gt[i]
        if len(np.unique(y_true)) <= 1:
            continue   # not scored by ComputeMetricsWrapper -- excluded here too
        pr_auc = average_precision_score(y_true, preds_sigmoid[i])
        records.append({"video_id": datum["video_id"],
                        "feature_path": datum["feature_path"], "pr_auc": pr_auc})
    return records


def collect_participant_points(pid, feat_len_cache):
    """Return (video_length_minutes, pr_auc) for every scored test episode of pid."""
    points = []
    for r in _episode_records(pid):
        length_s = _video_length_seconds(r["feature_path"], feat_len_cache)
        points.append((length_s / 60.0, r["pr_auc"]))
    return points


def collect_participant_video_pr_auc(pid):
    """Return {video_id: mean PR-AUC over that video's scored test episodes}.
    Videos with zero scored episodes are absent from the returned dict."""
    per_video = defaultdict(list)
    for r in _episode_records(pid):
        per_video[r["video_id"]].append(r["pr_auc"])
    return {vid: float(np.mean(v)) for vid, v in per_video.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=DEF_OUT)
    args = ap.parse_args()

    participants = sorted(SNAPSHOT_DIRS)
    feat_len_cache = {}
    per_pid_points = {}
    for pid in participants:
        print(f"[{pid}] loading...")
        per_pid_points[pid] = collect_participant_points(pid, feat_len_cache)
        print(f"[{pid}] {len(per_pid_points[pid])} scored episodes")

    fig, axes = plt.subplots(3, 3, figsize=(13, 11), sharex=False, sharey=True)
    for ax, pid in zip(axes.flat, participants):
        pts = per_pid_points[pid]
        xs = np.array([p[0] for p in pts])
        ys = np.array([p[1] for p in pts])

        ax.scatter(xs, ys, s=16, color=DOT_COLOR, alpha=0.55, edgecolors="none", zorder=3)
        ax.set_title(f"{pid}  (n={len(pts)} episodes)", fontsize=10, color="#0b0b0b")
        ax.yaxis.grid(True, linestyle=":", color="#e1e0d9", alpha=0.9, zorder=0)
        ax.set_axisbelow(True)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        for spine in ("left", "bottom"):
            ax.spines[spine].set_color("#c3c2b7")
        ax.tick_params(colors="#898781", labelsize=8)
        ax.set_ylim(-0.03, 1.03)

    for ax in axes[-1, :]:
        ax.set_xlabel("Source video length (min)", fontsize=9, color="#52514e")
    for ax in axes[:, 0]:
        ax.set_ylabel("PR-AUC", fontsize=9, color="#52514e")

    fig.suptitle("Per-episode PR-AUC vs. source video length, by held-out participant (LOOCV)",
                fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=150)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
