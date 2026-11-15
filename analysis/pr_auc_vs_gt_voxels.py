"""
Correlation between # GT voxels in the prediction window and per-episode PR-AUC.

For each of the three LOOCV snapshots given below (video,location,env,gaze
modality; test participant = the held-out one, VAL_PARTICIPANTS empty so the
saved checkpoint's predictions/labels ARE the test-set eval -- see trainer.py's
CustomTrainer.evaluate() and train.py's no-val branch), reads the best
checkpoint's predictions.pkl / labels.pkl (dumped by trainer.py::CustomTrainer._save)
and, for every episode scored by ComputeMetricsWrapper (both classes present in
the GT, train.py:193), computes:

    n_gt_voxels = number of 1s in that episode's [4096] GT label vector
                  (= voxels containing >=1 GT interaction inside the
                  prediction window, train.py:191/193)
    pr_auc      = average_precision_score(y_true, sigmoid(logits))
                  (identical computation to ComputeMetricsWrapper.compute_metrics)

predictions.pkl/labels.pkl are index-aligned to the exact `test_dataset` each
snapshot's train_sbatch.sh run built (same hyperparameters, TEST_N=0 i.e. no
cap; verified below by asserting len(dataset) == labels.shape[0]), so nothing
here needs to be recomputed or re-filtered beyond ComputeMetricsWrapper's own
scoring rule to match what training's script considered.

To split episodes by source video, the same HDEpicInteractionDataset is
rebuilt per participant with the DS_KWARGS these runs actually used (video,
location, env, gaze modality; env_orphan_fallback=False, time_anchor=gaze --
see each snapshot's out/*.out log) purely to read off `video_id` per episode
index; dataset order/content is otherwise untouched.

Reports Pearson r and Spearman rho (n_gt_voxels vs pr_auc), pooled across all
three participants and per participant, and saves one scatter-grid PNG per
participant (one subplot per source video).

Usage:
  python analysis/pr_auc_vs_gt_voxels.py
"""

import argparse
import glob
import json
import math
import os
import pickle
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.stats import pearsonr, spearmanr  # noqa: E402
from sklearn.metrics import average_precision_score  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from loader import HDEpicInteractionDataset  # noqa: E402

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
DS_KWARGS = dict(observation_time=30, anticipation_time=5, future_time=180,
                 static_env=True, env_source="boxer_grid",
                 voxel_centers_dir=VOXEL_CENTERS_DIR,
                 env_orphan_fallback=False, trajectory_simple=False,
                 trajectory_exclude_unfinished=False,
                 env_class_averaging=False, env_max_classes_per_voxel=4,
                 env_fallback_averaging="off",
                 gaze_max_hz=0.0, gaze_time_offset_s=0.1,
                 time_anchor="gaze", verbose=False)

CATEGORICAL = ["#2a78d6", "#008300", "#eda100"]


def _best_checkpoint_dir(snapshot_dir):
    run_dirs = glob.glob(os.path.join(snapshot_dir, "results", "*"))
    if len(run_dirs) != 1:
        raise SystemExit(f"Expected exactly one run dir under {snapshot_dir}/results, "
                         f"found {len(run_dirs)}")
    run_dir = run_dirs[0]
    any_state = glob.glob(os.path.join(run_dir, "checkpoint-*", "trainer_state.json"))[0]
    with open(any_state) as f:
        state = json.load(f)
    return (os.path.normpath(os.path.join(snapshot_dir, state["best_model_checkpoint"])),
            state["best_metric"])


def episode_records(pid):
    """Return (n_gt_voxels[n_scored], pr_auc[n_scored], video_id[n_scored]) for
    pid's scored test episodes (scored = both classes present in GT, matching
    ComputeMetricsWrapper -- the same episodes that feed the reported pr_auc)."""
    ckpt_dir, best_metric = _best_checkpoint_dir(SNAPSHOT_DIRS[pid])
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

    n_voxels, pr_aucs, video_ids = [], [], []
    for i, datum in enumerate(ds.dataset):
        y_true = gt[i]
        if len(np.unique(y_true)) <= 1:
            continue  # not scored by ComputeMetricsWrapper -- excluded here too
        n_voxels.append(int(y_true.sum()))
        pr_aucs.append(average_precision_score(y_true, preds_sigmoid[i]))
        video_ids.append(datum["video_id"])
    n_voxels = np.array(n_voxels)
    pr_aucs = np.array(pr_aucs)
    video_ids = np.array(video_ids)

    recomputed = pr_aucs.mean() if len(pr_aucs) else float("nan")
    print(f"[{pid}] {len(pr_aucs)} scored episodes from {len(set(video_ids))} videos  "
         f"recomputed mean pr_auc={recomputed:.4f}  best_metric={best_metric:.4f}")
    if abs(recomputed - best_metric) > 1e-4:
        print(f"  WARNING: recomputed mean pr_auc doesn't match best_metric for {pid}")
    return n_voxels, pr_aucs, video_ids


def _safe_corr(fn, v, a):
    if len(v) < 2 or len(np.unique(v)) < 2 or len(np.unique(a)) < 2:
        return float("nan"), float("nan")
    return fn(v, a)


def plot_participant_by_video(pid, n_voxels, pr_aucs, video_ids, out_path):
    """One figure per participant: one scatter subplot per source video."""
    videos = sorted(set(video_ids.tolist()))
    ncols = math.ceil(math.sqrt(len(videos)))
    nrows = math.ceil(len(videos) / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(3.6 * ncols, 3.2 * nrows),
                             sharex=False, sharey=True, squeeze=False)
    axes_flat = axes.flat
    color = CATEGORICAL[sorted(SNAPSHOT_DIRS).index(pid) % len(CATEGORICAL)]

    for ax, vid in zip(axes_flat, videos):
        mask = video_ids == vid
        v, a = n_voxels[mask], pr_aucs[mask]
        ax.scatter(v, a, s=16, color=color, alpha=0.5, edgecolors="none", zorder=3)

        pr, _ = _safe_corr(pearsonr, v, a)
        if len(v) >= 2 and len(np.unique(v)) >= 2:
            coeffs = np.polyfit(v, a, 1)
            xs = np.linspace(v.min(), v.max(), 100)
            ax.plot(xs, np.polyval(coeffs, xs), color="#3a3a3a", linewidth=1.3,
                   linestyle="--", zorder=4)

        r_str = f"{pr:+.2f}" if not math.isnan(pr) else "n/a"
        ax.set_title(f"{vid}  (n={mask.sum()})  r={r_str}", fontsize=8.5)
        ax.yaxis.grid(True, linestyle=":", color="#e1e0d9", alpha=0.9, zorder=0)
        ax.set_axisbelow(True)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        for spine in ("left", "bottom"):
            ax.spines[spine].set_color("#c3c2b7")
        ax.tick_params(colors="#898781", labelsize=7)
        ax.set_ylim(-0.03, 1.03)

    for ax in axes_flat:  # hide any unused trailing subplots
        ax.set_visible(False)

    fig.suptitle(f"{pid}: PR-AUC vs. GT voxel count per episode, by source video",
                fontsize=13)
    fig.text(0.5, 0.005, "# GT voxels in prediction window", ha="center", fontsize=10)
    fig.text(0.005, 0.5, "Episode PR-AUC", va="center", rotation="vertical", fontsize=10)
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
    per_pid = {pid: episode_records(pid) for pid in participants}

    all_voxels = np.concatenate([per_pid[p][0] for p in participants])
    all_pr_aucs = np.concatenate([per_pid[p][1] for p in participants])

    print(f"\nPooled across {len(participants)} participants, n={len(all_voxels)} episodes")
    print(f"  n_gt_voxels: mean={all_voxels.mean():.2f} std={all_voxels.std():.2f} "
         f"min={all_voxels.min()} max={all_voxels.max()}")
    r, p_r = pearsonr(all_voxels, all_pr_aucs)
    rho, p_rho = spearmanr(all_voxels, all_pr_aucs)
    print(f"  Pearson  r   = {r:+.4f}  (p={p_r:.2e})")
    print(f"  Spearman rho = {rho:+.4f}  (p={p_rho:.2e})")

    print("\nPer participant:")
    for pid in participants:
        v, a, _ = per_pid[pid]
        pr, ppr = pearsonr(v, a)
        sr, psr = spearmanr(v, a)
        print(f"  [{pid}] n={len(v):4d}  Pearson r={pr:+.4f} (p={ppr:.2e})  "
             f"Spearman rho={sr:+.4f} (p={psr:.2e})")

    print("\nPer video (one figure per participant):")
    for pid in participants:
        v, a, vids = per_pid[pid]
        out_path = os.path.join(args.out_dir, f"pr_auc_vs_gt_voxels_{pid}.png")
        plot_participant_by_video(pid, v, a, vids, out_path)


if __name__ == "__main__":
    main()
