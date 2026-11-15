"""
Per-participant threshold sweep: what happens to PR-AUC if we drop episodes
whose source video is short?

For each held-out participant (LOOCV, video,location,env,trajectory modality
-- the "_traj" snapshots), reuses the per-episode (video_length, pr_auc)
pairs from plot_pr_auc_vs_video_length.py to build one curve:

    y(x) = mean PR-AUC over episodes whose source video length >= x

x=0 keeps every scored episode, so y(0) equals the model's reported best
pr_auc (sanity-checked against trainer_state.json's best_metric below). As x
increases, episodes drawn from shorter videos are dropped. f is a step
function: it only changes value at x = each distinct video length present
for that participant (the set of "excluded videos" only changes there), and
it is left-continuous (a video of length L is still counted at x == L; it
drops out for x > L) -- so it's drawn with drawstyle='steps-post' anchored
at those breakpoints.

Usage:
  python analysis/plot_pr_auc_exclude_short_videos.py
"""

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from plot_pr_auc_vs_video_length import (  # noqa: E402
    SNAPSHOT_DIRS, collect_participant_points, _best_checkpoint_dir,
)

HERE = os.path.dirname(os.path.abspath(__file__))
DEF_OUT = os.path.join(HERE, "out", "pr_auc_exclude_short_videos.png")

# 8-hue categorical order (validated for adjacent-pair CVD safety); P09 reuses
# slot 1's hue with a dashed line so it's distinguished by style, not color alone.
CATEGORICAL = ["#2a78d6", "#008300", "#e87ba4", "#eda100",
              "#1baf7a", "#eb6834", "#4a3aa7", "#e34948"]


def threshold_curve(points):
    """points: list of (length_min, pr_auc). Returns (x_breakpoints, y_values)."""
    lengths = np.array([p[0] for p in points])
    pr_aucs = np.array([p[1] for p in points])
    breakpoints = [0.0] + sorted(set(lengths.tolist()))
    xs, ys = [], []
    for x in breakpoints:
        keep = lengths >= x
        if not keep.any():
            break
        xs.append(x)
        ys.append(pr_aucs[keep].mean())
    return np.array(xs), np.array(ys)


def best_metric_pr_auc(pid):
    ckpt_dir = _best_checkpoint_dir(SNAPSHOT_DIRS[pid])
    with open(os.path.join(ckpt_dir, "trainer_state.json")) as f:
        return json.load(f)["best_metric"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=DEF_OUT)
    args = ap.parse_args()

    participants = sorted(SNAPSHOT_DIRS)
    feat_len_cache = {}
    curves = {}
    for pid in participants:
        print(f"[{pid}] loading...")
        points = collect_participant_points(pid, feat_len_cache)
        xs, ys = threshold_curve(points)
        curves[pid] = (xs, ys)

        ref = best_metric_pr_auc(pid)
        print(f"[{pid}] y(0)={ys[0]:.4f}  best_metric={ref:.4f}  "
             f"n_episodes={len(points)}  n_video_lengths={len(xs) - 1}")
        if abs(ys[0] - ref) > 1e-4:
            print(f"  WARNING: y(0) doesn't match best_metric for {pid} "
                 f"-- ds_kwargs may not match the training run")

    fig, ax = plt.subplots(figsize=(9, 7))
    for i, pid in enumerate(participants):
        xs, ys = curves[pid]
        color = CATEGORICAL[i % len(CATEGORICAL)]
        linestyle = "--" if i >= len(CATEGORICAL) else "-"
        ax.plot(xs, ys, drawstyle="steps-post", color=color, linestyle=linestyle,
                linewidth=2, label=pid, zorder=3)

    ax.set_xlabel("Excluding videos shorter than X (min)")
    ax.set_ylabel("Mean PR-AUC of remaining episodes")
    ax.set_title("PR-AUC when dropping short source videos, by held-out participant (LOOCV)")
    ax.yaxis.grid(True, linestyle=":", color="#e1e0d9", alpha=0.9, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color("#c3c2b7")
    ax.tick_params(colors="#898781")
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False, title="Participant")

    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
