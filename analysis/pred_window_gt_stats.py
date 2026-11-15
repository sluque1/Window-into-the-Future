"""
Per-participant distributions of what an episode is asked to predict.

For every episode training builds, the PREDICTION window [t, t + future_time]
carries two different notions of "how much GT is there":

  (a) n_gt_events -- how many GT interactions fall in the window. This is
      ``len(ep['gt_locations'])`` (loader.py :: build_episodes): one entry per
      object-mask instance with a valid ``3d_location``. Several masks can share
      a timestamp, and the same object can be logged repeatedly.

  (b) n_gt_voxels -- how many DISTINCT voxels those interactions land in, i.e.
      ``labels['interaction'].sum()``, the number of positive cells in the 4096
      the model scores. A voxel can hold many interactions, so (b) <= (a) always,
      and the gap is how much the grid collapses.

Both are computed from the loader's own ``build_video_events`` /
``build_episodes``, so the episodes here are exactly the episodes training uses.

Grid
----
``--grid fixed`` (default) uses the per-participant grids in
``new_voxel_centers/<pid>_voxel_centers.npy`` -- what every production
``submit_*.sh`` runs with, and the only mode in which "a voxel" means the same
thing across a participant's videos. Video-level filters are then just "features
exist" + "has episodes".

``--grid per_video`` reproduces ``train_sbatch.sh``'s own default
(``VOXEL_CENTERS_DIR`` unset): a grid per video from Boxer OBBs + SLAM, which
additionally drops the 3 videos that have neither. Slower (reads SLAM).

Validation
----------
``--validate`` checks n_gt_voxels against ``out/<pid>_counts.npz``'s ``k_true``,
which was read out of a trained checkpoint's ``labels.pkl`` -- i.e. against the
labels the model was actually trained on (P04/P07/P08, obs30/anticip5/future180,
fixed grids; see pr_auc_vs_gt_voxels.py).

Usage
-----
  python analysis/pred_window_gt_stats.py --validate
  python analysis/pred_window_gt_stats.py --future_time 60
  python analysis/pred_window_gt_stats.py --grid per_video
  python analysis/pred_window_gt_stats.py --from-csv      # re-plot, no recompute
"""

import argparse
import json
import os
import sys

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from scipy import stats  # noqa: E402
from scipy.stats import gaussian_kde  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from loader import (  # noqa: E402
    FEATURES_BASE,
    SLAM_BASE,
    PARTICIPANT_TO_SPLIT,
    _load_annotations,
    build_episodes,
    build_video_events,
)

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
OUT_DIR = os.path.join(HERE, "out", "pred_window_gt")
VOXEL_CENTERS_DIR = os.path.join(REPO, "new_voxel_centers")

# ---------------------------------------------------------------------------
# Chart tokens (see analysis/pr_auc_vs_gt_voxels.py -- same palette)
# ---------------------------------------------------------------------------
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SERIES = "#2a78d6"      # categorical slot 1 -- the observed distribution
SERIES_FILL = "#9ec5f4"  # sequential step 200 of the same hue
FIT = "#eb6834"          # categorical slot 2 -- the fitted normal
POOLED = "#c3c2b7"       # pooled reference outline

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.edgecolor": AXIS,
    "axes.labelcolor": INK_2,
    "text.color": INK,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "axes.linewidth": 0.8,
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
})

METRICS = {
    "events": dict(
        col="n_gt_events",
        title="GT interactions per prediction window",
        xlabel="GT interactions in the prediction window (mask instances)",
        slug="gt_interactions",
    ),
    "voxels": dict(
        col="n_gt_voxels",
        title="Distinct GT voxels per prediction window",
        xlabel="distinct GT voxels in the prediction window (of 4096)",
        slug="gt_voxels",
    ),
}


# ---------------------------------------------------------------------------
# Episode collection
# ---------------------------------------------------------------------------

def _nearest_voxel(points, centres):
    """Flat voxel index [0,4095] of the nearest centre for each point.

    Same rule as loader.coord_to_voxel, vectorised over all points at once.
    """
    out = np.empty(len(points), dtype=np.int32)
    step = 4096  # keep the (chunk, 4096, 3) temporary small
    for lo in range(0, len(points), step):
        chunk = points[lo:lo + step]
        d = ((chunk[:, None, :] - centres[None, :, :]) ** 2).sum(axis=2)
        out[lo:lo + step] = d.argmin(axis=1)
    return out


def _fixed_centres(pid, cache={}):
    if pid not in cache:
        vc = np.load(os.path.join(VOXEL_CENTERS_DIR, f"{pid}_voxel_centers.npy"))
        cache[pid] = vc.astype(np.float64).reshape(-1, 3)
    return cache[pid]


def _per_video_centres(video_id, pid, boxer, slam_cache={}):
    """train_sbatch.sh's own default: AABB(Boxer OBBs UNION SLAM trajectory)."""
    traj = None
    mapping_path = os.path.join(SLAM_BASE, pid, "SLAM", "multi", "vrs_to_multi_slam.json")
    if os.path.exists(mapping_path):
        if mapping_path not in slam_cache:
            with open(mapping_path) as f:
                slam_cache[mapping_path] = json.load(f)
        mapping = slam_cache[mapping_path]
        multi_id = next((v for k, v in mapping.items() if video_id in k), None)
        if multi_id is not None:
            path = os.path.join(SLAM_BASE, pid, "SLAM", "multi", str(multi_id),
                                "slam", "closed_loop_trajectory.csv")
            if os.path.exists(path):
                import pandas as pd
                df = pd.read_csv(path, usecols=["tx_world_device", "ty_world_device",
                                                "tz_world_device"])
                xyz = df.values.astype(np.float64)
                traj = xyz[np.isfinite(xyz).all(axis=1)]
    vc = boxer.compute_and_cache_voxel_centers(video_id, traj)
    return None if vc is None else vc.reshape(-1, 3)


def collect(obs_t, anticip_t, future_t, grid_mode):
    """One row per episode: (split, pid, video_id, pred_start, n_gt_events, n_gt_voxels)."""
    assoc, mask = _load_annotations()
    boxer = None
    if grid_mode == "per_video":
        from boxer_env import BoxerEnvBuilder
        boxer = BoxerEnvBuilder()

    rows = []
    dropped = []
    for video_id in sorted(assoc):
        pid = video_id.split("-")[0]
        split = PARTICIPANT_TO_SPLIT.get(pid)
        if split is None:
            continue
        if not os.path.exists(os.path.join(FEATURES_BASE, pid, f"{video_id}_features.pt")):
            dropped.append((video_id, "no features"))
            continue

        centres = (_fixed_centres(pid) if grid_mode == "fixed"
                   else _per_video_centres(video_id, pid, boxer))
        if centres is None:
            dropped.append((video_id, "no voxel grid"))
            continue

        events, anchors = build_video_events(assoc[video_id], mask[video_id])
        if not events:
            dropped.append((video_id, "no events"))
            continue
        episodes = build_episodes(events, anchors, obs_t, anticip_t, future_t)
        if not episodes:
            dropped.append((video_id, "no episodes"))
            continue

        order = np.argsort([e["timestamp"] for e in events], kind="stable")
        ts = np.array([events[i]["timestamp"] for i in order])
        xyz = np.array([events[i]["location_3d"] for i in order], dtype=np.float64)
        vox = _nearest_voxel(xyz, centres)

        for ep in episodes:
            pred_start = ep["obs_end"] + anticip_t
            lo = np.searchsorted(ts, pred_start, side="left")
            hi = np.searchsorted(ts, pred_start + future_t, side="right")
            n_events = hi - lo
            if n_events != len(ep["gt_locations"]):
                raise SystemExit(f"{video_id} @ {pred_start}: window select disagrees "
                                 f"with build_episodes ({n_events} vs "
                                 f"{len(ep['gt_locations'])})")
            rows.append((split, pid, video_id, pred_start, int(n_events),
                         int(np.unique(vox[lo:hi]).size)))
    return rows, dropped


# ---------------------------------------------------------------------------
# Normal fit
# ---------------------------------------------------------------------------

def fit_normal(v):
    """MLE normal fit + how badly it fits.

    mu/sigma are ``scipy.stats.norm.fit``'s MLE, i.e. the sample mean and the
    population sd (ddof=0). ``median`` is the SAMPLE median, reported alongside
    because a fitted normal has mean == median by construction: the gap between
    the two is the skew the fit cannot represent.

    Goodness of fit is reported as effect sizes, not a verdict. At n in the
    thousands, Anderson-Darling rejects normality for any real count
    distribution -- ``ad_reject`` is therefore expected to be True almost
    everywhere and says little. ``ks_d`` (max |empirical CDF - fitted CDF|) is
    the number to read: it is a distance in probability units and does not grow
    with n.
    """
    v = np.asarray(v, dtype=float)
    mu, sigma = stats.norm.fit(v)
    ad = stats.anderson(v, dist="norm")
    crit_5pct = float(ad.critical_values[list(ad.significance_level).index(5.0)])
    return dict(
        n=len(v),
        mu=float(mu),
        sigma=float(sigma),
        sd_sample=float(v.std(ddof=1)),
        median=float(np.median(v)),
        skew=float(stats.skew(v)),
        excess_kurtosis=float(stats.kurtosis(v)),
        ks_d=float(stats.kstest(v, "norm", args=(mu, sigma)).statistic),
        ad_stat=float(ad.statistic),
        ad_crit_5pct=crit_5pct,
        ad_reject=bool(ad.statistic > crit_5pct),
        p_below_zero=float(stats.norm.cdf(0, mu, sigma)),
    )


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def _facet_grid(pids):
    fig, axes = plt.subplots(3, 3, figsize=(10.5, 8.4), sharex=True, sharey=True)
    return fig, dict(zip(pids, axes.ravel()))


def _clean(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_color(AXIS)
    ax.spines["bottom"].set_color(AXIS)
    ax.tick_params(length=3, width=0.8)


def plot_hist(by_pid, pooled, fits, meta, path, xmax, bins):
    pids = sorted(by_pid)
    fig, axmap = _facet_grid(pids)
    # Integer-aligned bins of equal integer width. Both metrics are counts, so
    # a fractional bin width (xmax/bins) makes some bins cover two integers and
    # some one, which shows up as a comb the data does not have.
    bw = max(1, int(round(xmax / bins)))
    edges = np.arange(-0.5, xmax + bw, bw)
    # Normalise by the participant's FULL episode count, not by the in-range
    # mass: episodes past xmax are reported in the corner rather than piled into
    # the last bin, which would invent a spike there.
    pooled_h, _ = np.histogram(pooled, bins=edges)
    pooled_h = 100 * pooled_h / len(pooled)

    for pid in pids:
        ax = axmap[pid]
        v = by_pid[pid]
        h, _ = np.histogram(v, bins=edges)
        h = 100 * h / len(v)
        # pooled reference first, so the participant reads on top of it
        ax.stairs(pooled_h, edges, color=POOLED, linewidth=1.0, zorder=1)
        ax.stairs(h, edges, fill=True, color=SERIES_FILL, zorder=2)
        ax.stairs(h, edges, color=SERIES, linewidth=1.0, zorder=3)
        med = float(np.median(v))
        ax.axvline(med, color=INK_2, linewidth=1.0, linestyle="-", zorder=4)
        # fitted normal, on the histogram's own scale: density x bin width x 100
        f = fits[pid]
        xs = np.linspace(-0.5, xmax + 0.5, 400)
        ax.plot(xs, 100 * bw * stats.norm.pdf(xs, f["mu"], f["sigma"]),
                color=FIT, linewidth=1.6, zorder=5)
        ax.set_title(f"{pid}", loc="left", fontsize=10, color=INK, pad=6)
        n_over = int((v > xmax).sum())
        ax.annotate(f"n = {len(v):,}\nmedian {med:.0f}\n"
                    f"N({f['mu']:.1f}, {f['sigma']:.1f}²)\nskew {f['skew']:+.2f}"
                    + (f"\n{n_over} beyond axis" if n_over else ""),
                    xy=(0.97, 0.94), xycoords="axes fraction", ha="right", va="top",
                    fontsize=8, color=INK_2, linespacing=1.4)
        ax.grid(axis="y", color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
        _clean(ax)
        ax.set_xlim(-0.5, xmax + 0.5)

    fig.suptitle(meta["title"], x=0.055, y=0.985, ha="left", fontsize=13, color=INK)
    fig.text(0.055, 0.944, meta["subtitle"], ha="left", fontsize=9, color=INK_2)
    fig.supxlabel(meta["xlabel"], fontsize=9, color=INK_2, y=0.035)
    fig.supylabel("% of that participant's episodes", fontsize=9, color=INK_2, x=0.028)
    # legend: which line is which, since two outlines share a facet
    fig.legend(handles=[
        plt.Line2D([], [], color=SERIES, linewidth=1.4, label="this participant"),
        plt.Line2D([], [], color=FIT, linewidth=1.6, label="fitted normal"),
        plt.Line2D([], [], color=POOLED, linewidth=1.4, label="all participants pooled"),
        plt.Line2D([], [], color=INK_2, linewidth=1.0, label="participant median"),
    ], loc="upper right", bbox_to_anchor=(0.985, 0.995), frameon=False,
        fontsize=8.5, labelcolor=INK_2, ncol=1, handlelength=1.6)
    fig.tight_layout(rect=(0.03, 0.045, 1, 0.925))
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_ridge(by_pid, pooled, fits, meta, path, xmax):
    """Kernel-density ridgeline: one row per participant, one shared x-axis.

    A ridgeline rather than nine overlaid curves: nine categorical hues is past
    the point where adjacent series stay distinguishable, and the row label
    carries identity for free.
    """
    pids = sorted(by_pid)
    fig, ax = plt.subplots(figsize=(9.5, 7.4))
    grid = np.linspace(0, xmax, 512)
    dens = {}
    for pid in pids:
        v = by_pid[pid].astype(float)
        kde = gaussian_kde(v)
        d = kde(grid)
        # reflect the mass the kernel puts below 0 back into the domain: counts
        # are bounded at 0 and the low bins are where these distributions live.
        d = d + kde(-grid)
        dens[pid] = d

    peak = max(d.max() for d in dens.values())
    step = 1.0                       # row pitch
    scale = 1.25 * step / peak       # tallest curve reaches 1.25 rows

    pooled_med = float(np.median(pooled))
    top = (len(pids) - 1) * step + 1.25 * step
    ax.vlines(pooled_med, -0.05, top, color=AXIS, linewidth=0.9, zorder=1)
    ax.text(pooled_med, top + 0.04, f"all-participant median {pooled_med:.0f}",
            ha="center", va="bottom", fontsize=8, color=MUTED)

    # Bottom row nearest the viewer: paint rows front-to-back so a curve is
    # occluded by the row BELOW it, never the row above.
    for row, pid in enumerate(reversed(pids)):     # row 0 = P09 at the bottom
        base = row * step
        z = 2 * (len(pids) - row)
        d = dens[pid] * scale
        ax.fill_between(grid, base, base + d, color=SERIES_FILL, zorder=z,
                        linewidth=0)
        # surface-coloured halo under the outline = the 2px gap between
        # overlapping marks, so adjacent ridges stay separable
        ax.plot(grid, base + d, color=SURFACE, linewidth=3.6, zorder=z + 1)
        ax.plot(grid, base + d, color=SERIES, linewidth=1.4, zorder=z + 2)
        f = fits[pid]
        ax.plot(grid, base + scale * stats.norm.pdf(grid, f["mu"], f["sigma"]),
                color=FIT, linewidth=1.3, zorder=z + 2)
        ax.hlines(base, 0, xmax, color=AXIS, linewidth=0.7, zorder=z + 2)
        med = float(np.median(by_pid[pid]))
        med_h = np.interp(med, grid, d)
        ax.vlines(med, base, base + med_h, color=INK_2, linewidth=1.1, zorder=z + 3)
        ax.text(-0.012 * xmax, base + 0.10, pid, ha="right", va="bottom",
                fontsize=9.5, color=INK)
        ax.text(-0.012 * xmax, base + 0.02, f"n={len(by_pid[pid]):,}  med {med:.0f}",
                ha="right", va="top", fontsize=7.5, color=MUTED)

    ax.set_xlim(0, xmax)
    ax.set_ylim(-0.12, top + 0.22)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    ax.tick_params(length=3, width=0.8)
    ax.grid(axis="x", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.set_xlabel(meta["xlabel"], fontsize=9, color=INK_2)

    n_over = int((pooled > xmax).sum())
    sub = meta["subtitle"] + (f"  ·  {n_over} episodes beyond the axis" if n_over else "")
    fig.suptitle(meta["title"] + " — kernel density", x=0.02, y=0.985,
                 ha="left", fontsize=13, color=INK)
    fig.text(0.02, 0.945, sub, ha="left", fontsize=9, color=INK_2)
    ax.legend(handles=[
        plt.Line2D([], [], color=SERIES, linewidth=1.4, label="kernel density"),
        plt.Line2D([], [], color=FIT, linewidth=1.3, label="fitted normal"),
        plt.Line2D([], [], color=INK_2, linewidth=1.0, label="participant median"),
    ], loc="upper right", frameon=False, fontsize=8.5, labelcolor=INK_2)
    fig.tight_layout(rect=(0.045, 0, 1, 0.925))
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--observation_time", type=float, default=30)
    ap.add_argument("--anticipation_time", type=float, default=5)
    ap.add_argument("--future_time", type=float, default=180)
    ap.add_argument("--grid", choices=("fixed", "per_video"), default="fixed")
    ap.add_argument("--out-dir", default=OUT_DIR)
    ap.add_argument("--from-csv", action="store_true",
                    help="re-plot from the saved per-episode CSV")
    ap.add_argument("--validate", action="store_true",
                    help="check n_gt_voxels against out/<pid>_counts.npz k_true")
    ap.add_argument("--xmax-quantile", type=float, default=0.995)
    ap.add_argument("--bins", type=int, default=40)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    csv_path = os.path.join(args.out_dir, "per_episode.csv")

    if args.from_csv:
        raw = np.genfromtxt(csv_path, delimiter=",", names=True, dtype=None, encoding="utf-8")
        rows = [(r["split"], r["participant_id"], r["video_id"], float(r["pred_start"]),
                 int(r["n_gt_events"]), int(r["n_gt_voxels"])) for r in raw]
        dropped = []
    else:
        rows, dropped = collect(args.observation_time, args.anticipation_time,
                                args.future_time, args.grid)
        with open(csv_path, "w") as f:
            f.write("split,participant_id,video_id,pred_start,n_gt_events,n_gt_voxels\n")
            for r in rows:
                f.write(f"{r[0]},{r[1]},{r[2]},{r[3]:.4f},{r[4]},{r[5]}\n")
        print(f"wrote {csv_path}  ({len(rows):,} episodes)")
        for vid, why in dropped:
            print(f"  dropped {vid}: {why}")

    pids = sorted({r[1] for r in rows})
    ev = {p: np.array([r[4] for r in rows if r[1] == p]) for p in pids}
    vx = {p: np.array([r[5] for r in rows if r[1] == p]) for p in pids}

    if args.validate:
        for pid in ("P04", "P07", "P08"):
            npz = os.path.join(HERE, "out", f"{pid}_counts.npz")
            if not os.path.exists(npz):
                continue
            k_true = np.load(npz)["k_true"].astype(int)
            mine = vx[pid]
            ok = len(k_true) == len(mine) and np.array_equal(k_true, mine)
            print(f"[validate] {pid}: {len(mine)} episodes vs {len(k_true)} in "
                  f"{os.path.basename(npz)} -- "
                  f"{'EXACT MATCH' if ok else 'MISMATCH'}")
            if not ok and len(k_true) == len(mine):
                d = np.abs(k_true - mine)
                print(f"           max |diff| {d.max()}, {(d > 0).sum()} episodes differ")

    # summary table
    lines = ["participant,n_episodes,ev_mean,ev_median,ev_p90,ev_max,"
             "vox_mean,vox_median,vox_p90,vox_max,collapse_ratio"]
    print(f"\n{'pid':5} {'n':>7} | {'GT interactions':^30} | {'distinct GT voxels':^30} | ratio")
    print(f"{'':5} {'':>7} | {'mean':>7}{'med':>7}{'p90':>8}{'max':>8} | "
          f"{'mean':>7}{'med':>7}{'p90':>8}{'max':>8} |")
    for pid in pids:
        e, v = ev[pid], vx[pid]
        ratio = v.mean() / e.mean()
        print(f"{pid:5} {len(e):7,d} | {e.mean():7.1f}{np.median(e):7.0f}"
              f"{np.percentile(e, 90):8.0f}{e.max():8d} | "
              f"{v.mean():7.1f}{np.median(v):7.0f}"
              f"{np.percentile(v, 90):8.0f}{v.max():8d} | {ratio:.2f}")
        lines.append(f"{pid},{len(e)},{e.mean():.2f},{np.median(e):.0f},"
                     f"{np.percentile(e, 90):.0f},{e.max()},{v.mean():.2f},"
                     f"{np.median(v):.0f},{np.percentile(v, 90):.0f},{v.max()},"
                     f"{ratio:.3f}")
    all_e = np.concatenate([ev[p] for p in pids])
    all_v = np.concatenate([vx[p] for p in pids])
    print(f"{'ALL':5} {len(all_e):7,d} | {all_e.mean():7.1f}{np.median(all_e):7.0f}"
          f"{np.percentile(all_e, 90):8.0f}{all_e.max():8d} | "
          f"{all_v.mean():7.1f}{np.median(all_v):7.0f}"
          f"{np.percentile(all_v, 90):8.0f}{all_v.max():8d} | "
          f"{all_v.mean() / all_e.mean():.2f}")
    lines.append(f"ALL,{len(all_e)},{all_e.mean():.2f},{np.median(all_e):.0f},"
                 f"{np.percentile(all_e, 90):.0f},{all_e.max()},{all_v.mean():.2f},"
                 f"{np.median(all_v):.0f},{np.percentile(all_v, 90):.0f},{all_v.max()},"
                 f"{all_v.mean() / all_e.mean():.3f}")
    with open(os.path.join(args.out_dir, "summary.csv"), "w") as f:
        f.write("\n".join(lines) + "\n")

    # ---- normal fits -----------------------------------------------------
    fit_rows = ["metric,participant,n,fit_mean_mu,fit_std_sigma,sample_median,"
                "sample_std_ddof1,skew,excess_kurtosis,ks_d,ad_stat,ad_crit_5pct,"
                "ad_rejects_normal,fitted_mass_below_zero"]
    fits = {}
    for key, data in (("events", ev), ("voxels", vx)):
        per = {p: fit_normal(data[p]) for p in pids}
        per["ALL"] = fit_normal(np.concatenate([data[p] for p in pids]))
        fits[key] = per
        label = METRICS[key]["col"]
        print(f"\nNormal fit -- {label}")
        print(f"{'pid':5} {'n':>7} {'mu':>7} {'sigma':>7} {'median':>7} "
              f"{'mu-med':>7} {'skew':>7} {'exkurt':>7} {'KS D':>6} {'A²':>7} {'norm?':>6}")
        for p in list(pids) + ["ALL"]:
            f = per[p]
            print(f"{p:5} {f['n']:7,d} {f['mu']:7.2f} {f['sigma']:7.2f} "
                  f"{f['median']:7.1f} {f['mu'] - f['median']:7.2f} {f['skew']:7.2f} "
                  f"{f['excess_kurtosis']:7.2f} {f['ks_d']:6.3f} {f['ad_stat']:7.1f} "
                  f"{'no' if f['ad_reject'] else 'yes':>6}")
            fit_rows.append(
                f"{label},{p},{f['n']},{f['mu']:.3f},{f['sigma']:.3f},"
                f"{f['median']:.1f},{f['sd_sample']:.3f},{f['skew']:.3f},"
                f"{f['excess_kurtosis']:.3f},{f['ks_d']:.4f},{f['ad_stat']:.2f},"
                f"{f['ad_crit_5pct']:.3f},{f['ad_reject']},{f['p_below_zero']:.5f}")
    with open(os.path.join(args.out_dir, "normal_fits.csv"), "w") as f:
        f.write("\n".join(fit_rows) + "\n")

    grid_note = ("fixed per-participant grids (new_voxel_centers/)" if args.grid == "fixed"
                 else "per-video grids (train_sbatch.sh default)")
    sub = (f"obs {args.observation_time:.0f}s · anticipation {args.anticipation_time:.0f}s · "
           f"future {args.future_time:.0f}s · {len(rows):,} episodes · {grid_note}")

    paths = []
    for key, data in (("events", ev), ("voxels", vx)):
        meta = dict(METRICS[key], subtitle=sub)
        pooled = np.concatenate([data[p] for p in pids])
        xmax = float(np.ceil(np.quantile(pooled, args.xmax_quantile) / 10) * 10)
        h = os.path.join(args.out_dir, f"{meta['slug']}_hist.png")
        k = os.path.join(args.out_dir, f"{meta['slug']}_kde.png")
        plot_hist(data, pooled, fits[key], meta, h, xmax, args.bins)
        plot_ridge(data, pooled, fits[key], meta, k, xmax)
        paths += [h, k]
    print("\nwrote:")
    for p in paths:
        print(f"  {p}")


if __name__ == "__main__":
    main()
