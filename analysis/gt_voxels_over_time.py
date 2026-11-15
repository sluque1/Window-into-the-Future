"""
Per-video scatter: start second of the prediction window (x) vs. total number
of GT voxels in that window (y).

For each of the three LOOCV snapshots (video,location,env,gaze modality, see
pr_auc_vs_gt_voxels.py's module docstring for how DS_KWARGS was pinned to
match these runs), rebuilds the held-out participant's test HDEpicInteractionDataset
to get every episode's video_id and pred_start (= obs_end + anticipation_time,
== the anchor time t; loader.py::build_episodes), and reads n_gt_voxels for
that same episode straight from the checkpoint's labels.pkl (dumped
index-aligned to the dataset by trainer.py::CustomTrainer._save -- see
pr_auc_vs_gt_voxels.py for the index-alignment check). No PR-AUC / predictions
involved here, so every episode in the test set is plotted (loader.py already
drops empty-GT episodes, so there is no further "scored" filter to apply).

One PNG per participant, one subplot per source video.

Usage:
  python analysis/gt_voxels_over_time.py
"""

import argparse
import math
import os
import pickle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from pr_auc_vs_gt_voxels import (  # noqa: E402
    SNAPSHOT_DIRS, DS_KWARGS, CATEGORICAL, OUT_DIR, _best_checkpoint_dir,
)
from loader import HDEpicInteractionDataset  # noqa: E402

ANTICIPATION_TIME = DS_KWARGS["anticipation_time"]


def video_records(pid):
    """Return (pred_start[n], n_gt_voxels[n], video_id[n]) for every episode
    in pid's test set (dataset order == labels.pkl row order)."""
    ckpt_dir, _ = _best_checkpoint_dir(SNAPSHOT_DIRS[pid])
    with open(os.path.join(ckpt_dir, "labels.pkl"), "rb") as f:
        labels = pickle.load(f)
    gt = labels["interaction"]

    ds = HDEpicInteractionDataset("test", participants=[pid], **DS_KWARGS)
    if len(ds) != gt.shape[0]:
        raise SystemExit(f"{pid}: dataset has {len(ds)} episodes but checkpoint has "
                         f"{gt.shape[0]} -- DS_KWARGS likely don't match this training run")

    pred_start, n_voxels, video_ids = [], [], []
    for i, datum in enumerate(ds.dataset):
        pred_start.append(datum["obs_end"] + ANTICIPATION_TIME)
        n_voxels.append(int(gt[i].sum()))
        video_ids.append(datum["video_id"])

    print(f"[{pid}] {len(pred_start)} episodes from {len(set(video_ids))} videos")
    return np.array(pred_start), np.array(n_voxels), np.array(video_ids)


def plot_participant_by_video(pid, pred_start, n_voxels, video_ids, out_path, color):
    videos = sorted(set(video_ids.tolist()))
    ncols = math.ceil(math.sqrt(len(videos)))
    nrows = math.ceil(len(videos) / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(3.6 * ncols, 3.2 * nrows),
                             sharex=False, sharey=True, squeeze=False)
    axes_flat = axes.flat

    for ax, vid in zip(axes_flat, videos):
        mask = video_ids == vid
        x, y = pred_start[mask], n_voxels[mask]
        ax.scatter(x, y, s=14, color=color, alpha=0.5, edgecolors="none", zorder=3)

        ax.set_title(f"{vid}  (n={mask.sum()})", fontsize=8.5)
        ax.yaxis.grid(True, linestyle=":", color="#e1e0d9", alpha=0.9, zorder=0)
        ax.set_axisbelow(True)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        for spine in ("left", "bottom"):
            ax.spines[spine].set_color("#c3c2b7")
        ax.tick_params(colors="#898781", labelsize=7)

    for ax in axes_flat:  # hide any unused trailing subplots
        ax.set_visible(False)

    fig.suptitle(f"{pid}: # GT voxels in prediction window vs. window start time, by source video",
                fontsize=13)
    fig.text(0.5, 0.005, "Prediction-window start (s into video)", ha="center", fontsize=10)
    fig.text(0.005, 0.5, "# GT voxels in prediction window", va="center",
            rotation="vertical", fontsize=10)
    fig.tight_layout(rect=[0.015, 0.015, 1, 0.96])

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Wrote {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=OUT_DIR)
    args = ap.parse_args()

    participants = sorted(SNAPSHOT_DIRS)
    for i, pid in enumerate(participants):
        pred_start, n_voxels, video_ids = video_records(pid)
        out_path = os.path.join(args.out_dir, f"gt_voxels_over_time_{pid}.png")
        color = CATEGORICAL[i % len(CATEGORICAL)]
        plot_participant_by_video(pid, pred_start, n_voxels, video_ids, out_path, color)


if __name__ == "__main__":
    main()
