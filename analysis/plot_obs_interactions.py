"""
Box plot of observation-window GT-interaction counts per episode.

Reads the per-video CSVs written by obs_window_interactions.py and draws two
boxes over all episodes:

  * "All interactions"    : n_total_interactions (every valid-3d start & end)
  * "Start+end same obj"  : n_both_same_object   (complete movements observed)

Individual episodes are overlaid as very translucent jittered dots.

Usage:
  python analysis/plot_obs_interactions.py
  python analysis/plot_obs_interactions.py --csv-dir <dir> --out <path.png>
"""

import argparse
import csv
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DEF_CSV_DIR = os.path.join(HERE, "out", "per_video_obs_interactions")
DEF_OUT = os.path.join(HERE, "out", "obs_interactions_boxplot.png")


def load_counts(csv_dir):
    total, both = [], []
    files = sorted(glob.glob(os.path.join(csv_dir, "*.csv")))
    if not files:
        raise SystemExit(f"No CSVs in {csv_dir} — run obs_window_interactions.py first.")
    for path in files:
        with open(path) as f:
            for row in csv.DictReader(f):
                total.append(int(row["n_total_interactions"]))
                both.append(int(row["n_both_same_object"]))
    return np.asarray(total), np.asarray(both), len(files)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv-dir", default=DEF_CSV_DIR)
    ap.add_argument("--out", default=DEF_OUT)
    args = ap.parse_args()

    total, both, n_files = load_counts(args.csv_dir)
    data = [total, both]
    labels = [f"All interactions\n(n={total.size})",
              f"Start+end same object\n(n={both.size})"]
    colors = ["#3b7dd8", "#d8703b"]
    positions = [1, 2]

    fig, ax = plt.subplots(figsize=(7, 6))

    # Translucent jittered dots (drawn under the boxes).
    rng = np.random.default_rng(0)
    for pos, vals, c in zip(positions, data, colors):
        x = pos + rng.uniform(-0.18, 0.18, size=vals.size)
        ax.scatter(x, vals, s=6, color=c, alpha=0.02, edgecolors="none", zorder=1)

    bp = ax.boxplot(data, positions=positions, widths=0.5, showfliers=False,
                    patch_artist=True, zorder=3,
                    medianprops=dict(color="black", linewidth=2),
                    whiskerprops=dict(color="black"),
                    capprops=dict(color="black"),
                    boxprops=dict(edgecolor="black"))
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.55)

    # Mean markers.
    means = [vals.mean() for vals in data]
    ax.scatter(positions, means, marker="D", color="white", edgecolors="black",
               zorder=4, s=45, label="mean")

    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.set_ylabel("GT interactions per observation window")
    ax.set_title("Interactions in the observation window per episode\n"
                 f"({total.size} episodes, {n_files} videos)")
    ax.yaxis.grid(True, linestyle=":", alpha=0.5)
    ax.set_axisbelow(True)
    ax.legend(loc="upper right", frameon=False)

    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=150)
    print(f"Wrote {args.out}")
    for lab, vals in zip(["All interactions", "Start+end same object"], data):
        print(f"  {lab:22s} mean {vals.mean():.2f}  median {np.median(vals):.0f}  "
              f"p25 {np.percentile(vals,25):.0f}  p75 {np.percentile(vals,75):.0f}  "
              f"max {vals.max()}")


if __name__ == "__main__":
    main()
