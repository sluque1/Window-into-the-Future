"""
Environment-token / voxel-grid occupancy analysis (per episode).

This inspects exactly what the model receives as its ENV token stream: the
16x16x16 (=4096) semantic voxel grid built by ``boxer_env.BoxerEnvBuilder`` for
each episode's observation window, using the FIXED per-participant voxel centres
in ``new_voxel_centers/<pid>_voxel_centers.npy`` (i.e. the loader's
``voxel_centers_dir`` mode). It answers three questions, per episode:

  Q1. Voxel occupancy.
      How many of the 4096 voxels are actually occupied (env value > 0) just
      before the grid is handed to the embedding layer?  == (env_grid > 0).sum().

  Q2. 3D boxes: fused vs. populating.
      After confidence filtering + spatial fusion (``_fuse_boxes``), how many
      fused 3D object boxes exist in the window, and how many of those actually
      populate the grid -- i.e. have at least one voxel *centre* falling inside
      the box's oriented bounding box (the exact inside-test the voxeliser uses)?

  Q3. Object classes: present vs. populating.
      How many distinct object classes (Boxer ``sem_id``) are present among the
      fused boxes, and how many distinct classes actually survive into the final
      grid (unique non-zero env values, after highest-confidence-wins voxel
      assignment)?

We reuse the *real* pipeline code (loader + boxer_env) so the numbers never
drift from training. ``static_env=False`` skips the loader's own env precompute;
we redo the per-window fusion here so we can additionally expose the fused-box
and populating-box counts that ``build_env`` does not return.

Output
------
  out/env_voxel_occupancy/per_episode.csv        one row per episode
  out/env_voxel_occupancy/per_participant.csv     per-participant summary
  out/env_voxel_occupancy/voxel_occupancy.png     Q1 figure
  out/env_voxel_occupancy/boxes_fused_vs_pop.png  Q2 figure
  out/env_voxel_occupancy/classes_present_vs_pop.png  Q3 figure
A numeric summary is printed to the console.

Usage
-----
  python analysis/env_voxel_occupancy.py
  python analysis/env_voxel_occupancy.py --participants P07
  python analysis/env_voxel_occupancy.py --splits val \
      --observation_time 30 --anticipation_time 5 --future_time 180
"""

import argparse
import csv
import os
import sys

import numpy as np
from tqdm import tqdm

# Import the real pipeline so this analysis never drifts from training.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from boxer_env import (  # noqa: E402
    GRID_SIZE,
    _fuse_boxes,
    _quat_to_rotmat,
    find_closest_voxel,
)
from loader import (  # noqa: E402
    NUM_VOXELS,
    PARTICIPANT_TO_SPLIT,
    HDEpicInteractionDataset,
)

DEFAULT_VC_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "new_voxel_centers")
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "out", "env_voxel_occupancy")

PER_EPISODE_FIELDS = [
    "participant_id", "video_id", "obs_start", "obs_end",
    "voxels_occupied", "voxels_occupied_frac",
    "n_boxes_fused", "n_boxes_populating", "frac_boxes_populating",
    "n_boxes_fallback_assigned", "n_boxes_populating_total",
    "frac_boxes_populating_total",
    "n_classes_fused", "n_classes_populating",
]


# ---------------------------------------------------------------------------
# Per-window fusion + voxelisation (mirrors boxer_env, but returns extra stats)
# ---------------------------------------------------------------------------

def voxelize_with_stats(fused_boxes, voxel_centers):
    """Voxelise fused boxes into a 4096 grid AND return per-box occupancy stats.

    Reproduces ``boxer_env._voxelize_fused_boxes`` exactly: a primary pass
    (highest-confidence voxel assignment among voxel centres inside each box's
    OBB) followed by a fallback pass that assigns boxes with zero primary hits
    to their single nearest voxel, only into voxels still empty after the
    primary pass.

    Returns:
        grid              (4096,) float64 -- final env values (0 = empty).
        box_voxel_hits    (n_boxes,) int  -- PRIMARY-pass voxel centres inside
                          each box's OBB (0 for a box the fallback later
                          rescues).
        fallback_assigned (n_boxes,) bool -- True where a zero-hit box was
                          successfully written by the fallback pass (i.e. its
                          nearest voxel was still empty when it got there).
    """
    flat = voxel_centers.reshape(-1, 3)                 # (4096, 3)
    grid = np.zeros(NUM_VOXELS, dtype=np.float64)
    conf = np.zeros(NUM_VOXELS, dtype=np.float64)
    hits = np.zeros(len(fused_boxes), dtype=np.int64)
    fallback_assigned = np.zeros(len(fused_boxes), dtype=bool)

    orphan_idxs = []
    for i, box in enumerate(fused_boxes):
        sem_id = box['sem_id']
        R = _quat_to_rotmat(*box['quat'])
        d = flat - box['center']
        p_local = d @ R                                 # R^T @ d
        inside = np.all(np.abs(p_local) <= box['half_extents'], axis=1)
        hits[i] = int(inside.sum())
        if sem_id <= 0:
            continue
        if hits[i] == 0:
            orphan_idxs.append(i)
            continue
        update = inside & (box['prob'] > conf)
        grid[update] = sem_id
        conf[update] = box['prob']

    for i in orphan_idxs:
        box = fused_boxes[i]
        vi, vj, vk = find_closest_voxel(box['center'], voxel_centers)
        flat_idx = (vi * GRID_SIZE + vj) * GRID_SIZE + vk
        if grid[flat_idx] != 0:
            continue    # primary pass already claimed this voxel -- leave it
        if box['prob'] > conf[flat_idx]:
            grid[flat_idx] = box['sem_id']
            conf[flat_idx] = box['prob']
            fallback_assigned[i] = True
    return grid, hits, fallback_assigned


def window_stats(boxer, video_id, obs_start, obs_end, t0_s, voxel_centers):
    """Compute the Q1/Q2/Q3 stats for one observation window.

    Returns a dict of the metric fields, or None if the window has no usable
    Boxer content (empty grid).
    """
    empty_stats = dict(voxels_occupied=0, n_boxes_fused=0, n_boxes_populating=0,
                       n_boxes_fallback_assigned=0, n_classes_fused=0,
                       n_classes_populating=0)

    boxer_df = boxer.load_boxer_csv(video_id)
    if boxer_df is None or voxel_centers is None:
        return empty_stats

    vrs_start = (obs_start + t0_s) * 1e9
    vrs_end = (obs_end + t0_s) * 1e9
    mask = (boxer_df['time_ns'] >= vrs_start) & (boxer_df['time_ns'] <= vrs_end)
    window_df = boxer_df.loc[mask]
    if len(window_df) == 0:
        return empty_stats

    fused = _fuse_boxes(
        window_df, conf_threshold=boxer.conf_threshold,
        min_detections=boxer.min_detections, cluster_distance=boxer.cluster_distance)
    if len(fused) == 0:
        return empty_stats

    grid, hits, fallback_assigned = voxelize_with_stats(fused, voxel_centers)

    fused_sem = np.array([b['sem_id'] for b in fused], dtype=np.int64)
    return dict(
        voxels_occupied=int((grid > 0).sum()),
        n_boxes_fused=int(len(fused)),
        n_boxes_populating=int((hits > 0).sum()),
        n_boxes_fallback_assigned=int(fallback_assigned.sum()),
        n_classes_fused=int(np.unique(fused_sem).size),
        n_classes_populating=int(np.unique(grid[grid > 0]).size),
    )


# ---------------------------------------------------------------------------
# Data collection
# ---------------------------------------------------------------------------

def collect(participants, splits, observation_time, anticipation_time,
            future_time, vc_dir):
    """Build episodes for the requested participants and compute per-episode stats.

    Returns a list of per-episode dicts (PER_EPISODE_FIELDS).
    """
    if participants:
        pids = sorted(set(participants))
    else:
        pids = sorted(p for p, s in PARTICIPANT_TO_SPLIT.items() if s in splits)

    rows = []
    for pid in pids:
        split = PARTICIPANT_TO_SPLIT.get(pid, "train")
        # static_env=False -> skip the loader's env precompute; we redo fusion
        # here to also expose fused/populating box counts.
        ds = HDEpicInteractionDataset(
            split=split, observation_time=observation_time,
            anticipation_time=anticipation_time, future_time=future_time,
            participants=[pid], static_env=False,
            voxel_centers_dir=vc_dir, verbose=True)

        # Cache stats per unique observation window (env depends only on the
        # window, not on the anchor/GT), then expand to per-episode rows.
        cache = {}
        for datum in tqdm(ds.dataset, desc=f"[{pid}] windows", leave=False):
            video_id = datum['video_id']
            obs_start = int(datum['obs_start'])
            obs_end = int(datum['obs_end'])
            key = (video_id, obs_start, obs_end)
            if key not in cache:
                vc = ds.boxer.get_voxel_centers(video_id)
                cache[key] = window_stats(
                    ds.boxer, video_id, obs_start, obs_end,
                    ds._t0[video_id], vc)
            st = cache[key]
            n_populating_total = st["n_boxes_populating"] + st["n_boxes_fallback_assigned"]
            rows.append({
                "participant_id": pid,
                "video_id": video_id,
                "obs_start": obs_start,
                "obs_end": obs_end,
                "voxels_occupied": st["voxels_occupied"],
                "voxels_occupied_frac": st["voxels_occupied"] / NUM_VOXELS,
                "n_boxes_fused": st["n_boxes_fused"],
                "n_boxes_populating": st["n_boxes_populating"],
                "frac_boxes_populating": (st["n_boxes_populating"] / st["n_boxes_fused"]
                                          if st["n_boxes_fused"] else 0.0),
                "n_boxes_fallback_assigned": st["n_boxes_fallback_assigned"],
                "n_boxes_populating_total": n_populating_total,
                "frac_boxes_populating_total": (n_populating_total / st["n_boxes_fused"]
                                                if st["n_boxes_fused"] else 0.0),
                "n_classes_fused": st["n_classes_fused"],
                "n_classes_populating": st["n_classes_populating"],
            })
    return rows


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _fmt(a):
    a = np.asarray(a, dtype=np.float64)
    if a.size == 0:
        return "n/a"
    return (f"mean {a.mean():6.2f}  median {np.median(a):5.1f}  "
            f"min {a.min():5.0f}  max {a.max():5.0f}  std {a.std():5.2f}")


def print_report(rows):
    print("\n" + "=" * 78)
    print(f"ENV VOXEL-GRID OCCUPANCY  --  {len(rows)} episodes")
    print("=" * 78)
    occ = [r["voxels_occupied"] for r in rows]
    fused = [r["n_boxes_fused"] for r in rows]
    pop = [r["n_boxes_populating"] for r in rows]
    cf = [r["n_classes_fused"] for r in rows]
    cp = [r["n_classes_populating"] for r in rows]
    fbp = [r["frac_boxes_populating"] for r in rows]
    fba = [r["n_boxes_fallback_assigned"] for r in rows]
    fbpt = [r["frac_boxes_populating_total"] for r in rows]

    print("\nQ1. Occupied voxels / 4096 (env_grid > 0):")
    print("    OVERALL: " + _fmt(occ))
    print(f"             mean occupancy = {100*np.mean(occ)/NUM_VOXELS:.2f}% of 4096")
    print("    per participant:")
    for pid in sorted({r["participant_id"] for r in rows}):
        pocc = np.array([r["voxels_occupied"] for r in rows
                         if r["participant_id"] == pid])
        print(f"      {pid} (n={pocc.size:>4}): {_fmt(pocc)}  "
              f"[{100*pocc.mean()/NUM_VOXELS:4.1f}% of 4096]")

    print("\nQ2. Fused 3D boxes per episode:")
    print("    " + _fmt(fused))
    print("    ...of which populate >=1 voxel (centre inside OBB, primary pass only):")
    print("    " + _fmt(pop))
    print(f"    fraction of fused boxes that populate a voxel (primary only) = "
          f"{100*np.mean(fbp):.1f}% (per-episode mean)")

    print("\nQ2b. Fallback pass (nearest-voxel assignment for zero-hit boxes):")
    print("    boxes rescued by the fallback pass per episode:")
    print("    " + _fmt(fba))
    print(f"    fraction of fused boxes that populate a voxel (primary + fallback) = "
          f"{100*np.mean(fbpt):.1f}% (per-episode mean, vs {100*np.mean(fbp):.1f}% without fallback)")

    print("\nQ3. Object classes (Boxer sem_id) per episode:")
    print("    present after fusing:   " + _fmt(cf))
    print("    populating the grid :   " + _fmt(cp))

    print("\nPer-participant means:")
    print(f"    {'pid':>4} {'eps':>5} {'occ/4096':>9} {'%':>6} "
          f"{'fused':>6} {'pop':>6} {'fbAssign':>8} {'clsF':>6} {'clsP':>6}")
    pids = sorted({r["participant_id"] for r in rows})
    for pid in pids:
        pr = [r for r in rows if r["participant_id"] == pid]
        print(f"    {pid:>4} {len(pr):>5} "
              f"{np.mean([r['voxels_occupied'] for r in pr]):>9.1f} "
              f"{100*np.mean([r['voxels_occupied_frac'] for r in pr]):>5.1f}% "
              f"{np.mean([r['n_boxes_fused'] for r in pr]):>6.1f} "
              f"{np.mean([r['n_boxes_populating'] for r in pr]):>6.1f} "
              f"{np.mean([r['n_boxes_fallback_assigned'] for r in pr]):>8.1f} "
              f"{np.mean([r['n_classes_fused'] for r in pr]):>6.1f} "
              f"{np.mean([r['n_classes_populating'] for r in pr]):>6.1f}")


def write_csvs(rows, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    ep_path = os.path.join(out_dir, "per_episode.csv")
    with open(ep_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=PER_EPISODE_FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: (round(r[k], 4) if isinstance(r[k], float) else r[k])
                        for k in PER_EPISODE_FIELDS})

    pp_path = os.path.join(out_dir, "per_participant.csv")
    pids = sorted({r["participant_id"] for r in rows})
    with open(pp_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["participant_id", "n_episodes",
                    "mean_voxels_occupied", "mean_occupied_frac",
                    "mean_boxes_fused", "mean_boxes_populating",
                    "mean_frac_boxes_populating",
                    "mean_boxes_fallback_assigned", "mean_boxes_populating_total",
                    "mean_frac_boxes_populating_total",
                    "mean_classes_fused", "mean_classes_populating"])
        for pid in pids:
            pr = [r for r in rows if r["participant_id"] == pid]
            w.writerow([
                pid, len(pr),
                round(np.mean([r["voxels_occupied"] for r in pr]), 3),
                round(np.mean([r["voxels_occupied_frac"] for r in pr]), 5),
                round(np.mean([r["n_boxes_fused"] for r in pr]), 3),
                round(np.mean([r["n_boxes_populating"] for r in pr]), 3),
                round(np.mean([r["frac_boxes_populating"] for r in pr]), 4),
                round(np.mean([r["n_boxes_fallback_assigned"] for r in pr]), 3),
                round(np.mean([r["n_boxes_populating_total"] for r in pr]), 3),
                round(np.mean([r["frac_boxes_populating_total"] for r in pr]), 4),
                round(np.mean([r["n_classes_fused"] for r in pr]), 3),
                round(np.mean([r["n_classes_populating"] for r in pr]), 3),
            ])
    return ep_path, pp_path


# ---------------------------------------------------------------------------
# Visualisation  (palette + rules from the dataviz skill)
# ---------------------------------------------------------------------------

# Validated reference palette (light mode): slot 1 blue, slot 2 aqua.
C_BLUE = "#2a78d6"
C_AQUA = "#1baf7a"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASE = "#c3c2b7"
SURF = "#fcfcfb"


def _style():
    import matplotlib as mpl
    mpl.rcParams.update({
        "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
        "axes.edgecolor": BASE, "axes.linewidth": 1.0,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
        "axes.axisbelow": True,
        "text.color": INK, "axes.labelcolor": INK2, "axes.titlecolor": INK,
        "xtick.color": MUTED, "ytick.color": MUTED,
        "xtick.labelsize": 9, "ytick.labelsize": 9,
        "axes.titlesize": 12, "axes.labelsize": 10,
        "font.family": "sans-serif", "font.size": 10,
        "axes.spines.top": False, "axes.spines.right": False,
        "legend.frameon": False, "legend.fontsize": 9,
    })


def _clean(ax):
    ax.grid(axis="x", visible=False)


def make_figs(rows, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _style()

    pids = sorted({r["participant_id"] for r in rows})
    occ = np.array([r["voxels_occupied"] for r in rows])
    fused = np.array([r["n_boxes_fused"] for r in rows])
    pop = np.array([r["n_boxes_populating"] for r in rows])
    cf = np.array([r["n_classes_fused"] for r in rows])
    cp = np.array([r["n_classes_populating"] for r in rows])

    def by_pid(vals):
        return [vals[[i for i, r in enumerate(rows) if r["participant_id"] == p]]
                for p in pids]

    paths = []

    # ---- Q1: voxel occupancy -------------------------------------------------
    fig, (a0, a1) = plt.subplots(2, 1, figsize=(7.6, 8.8))
    fig.suptitle("Q1  Occupied voxels per episode (of 4096)", fontsize=13,
                 fontweight="bold", x=0.5, y=1.0)

    mx = int(occ.max()) if occ.size else 1
    a0.hist(occ, bins=min(40, max(10, mx)), color=C_BLUE, edgecolor=SURF, linewidth=0.4)
    a0.axvline(occ.mean(), color=INK, linewidth=1.5, linestyle="--")
    a0.text(occ.mean(), a0.get_ylim()[1] * 0.95,
            f"  mean {occ.mean():.1f}  ({100*occ.mean()/NUM_VOXELS:.1f}%)",
            color=INK, fontsize=9, va="top")
    a0.set_xlabel("occupied voxels / 4096")
    a0.set_ylabel("episodes")
    a0.set_title("Distribution across all episodes", fontsize=11, loc="left")
    _clean(a0)

    bp = a1.boxplot(by_pid(occ), positions=range(len(pids)), widths=0.6,
                    patch_artist=True, showfliers=False,
                    medianprops=dict(color=INK, linewidth=1.4),
                    whiskerprops=dict(color=BASE), capprops=dict(color=BASE),
                    boxprops=dict(facecolor=C_BLUE, edgecolor=SURF, linewidth=0.8))
    means = [v.mean() for v in by_pid(occ)]
    a1.scatter(range(len(pids)), means, color=INK, s=18, zorder=3, label="mean")
    a1.set_xticks(range(len(pids)))
    a1.set_xticklabels(pids, rotation=45, ha="right")
    a1.set_ylabel("occupied voxels / 4096")
    a1.set_title("Per participant (box = IQR, dot = mean)", fontsize=11, loc="left")
    a1.legend(loc="upper right")
    _clean(a1)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    p = os.path.join(out_dir, "voxel_occupancy.png")
    fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig); paths.append(p)

    # ---- Q2: fused vs populating boxes --------------------------------------
    fig, (a0, a1) = plt.subplots(2, 1, figsize=(7.6, 8.8))
    fig.suptitle("Q2  Fused 3D boxes vs. boxes that populate a voxel",
                 fontsize=13, fontweight="bold", y=1.0)

    lim = int(max(fused.max(), 1)) + 1
    a0.plot([0, lim], [0, lim], color=BASE, linewidth=1.2, linestyle="--", zorder=1)
    a0.text(lim, lim, " y = x", color=MUTED, fontsize=8, va="bottom", ha="right")
    a0.scatter(fused, pop, s=14, color=C_BLUE, alpha=0.35, edgecolors="none", zorder=2)
    a0.set_xlabel("fused boxes in window")
    a0.set_ylabel("boxes populating >=1 voxel")
    a0.set_title("Per episode (each dot = one episode)", fontsize=11, loc="left")
    a0.set_xlim(0, lim); a0.set_ylim(0, lim)
    _clean(a0)

    x = np.arange(len(pids)); wbar = 0.38
    mf = [v.mean() for v in by_pid(fused)]
    mp = [v.mean() for v in by_pid(pop)]
    b1 = a1.bar(x - wbar/2, mf, wbar, color=C_BLUE, edgecolor=SURF, linewidth=1.0,
                label="fused")
    b2 = a1.bar(x + wbar/2, mp, wbar, color=C_AQUA, edgecolor=SURF, linewidth=1.0,
                label="populating", hatch="///")
    a1.bar_label(b1, fmt="%.1f", padding=2, fontsize=7.5, color=INK2)
    a1.bar_label(b2, fmt="%.1f", padding=2, fontsize=7.5, color=INK2)
    a1.set_xticks(x); a1.set_xticklabels(pids, rotation=45, ha="right")
    a1.set_ylabel("boxes per episode (mean)")
    a1.set_title("Per participant", fontsize=11, loc="left")
    a1.set_ylim(0, max(mf) * 1.20)
    a1.legend(loc="upper center", ncol=2)
    _clean(a1)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    p = os.path.join(out_dir, "boxes_fused_vs_pop.png")
    fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig); paths.append(p)

    # ---- Q3: classes present vs populating ----------------------------------
    fig, (a0, a1) = plt.subplots(2, 1, figsize=(7.6, 8.8))
    fig.suptitle("Q3  Object classes present (fused) vs. populating the grid",
                 fontsize=13, fontweight="bold", y=1.0)

    lim = int(max(cf.max(), 1)) + 1
    # integer-count scatter -> jitter so overlapping episodes are visible
    rng = np.random.default_rng(0)
    jx = cf + rng.uniform(-0.18, 0.18, size=cf.shape)
    jy = cp + rng.uniform(-0.18, 0.18, size=cp.shape)
    a0.plot([0, lim], [0, lim], color=BASE, linewidth=1.2, linestyle="--", zorder=1)
    a0.text(lim, lim, " y = x", color=MUTED, fontsize=8, va="bottom", ha="right")
    a0.scatter(jx, jy, s=14, color=C_BLUE, alpha=0.35, edgecolors="none", zorder=2)
    a0.set_xlabel("classes present (fused)")
    a0.set_ylabel("classes populating grid")
    a0.set_title("Per episode (jittered; each dot = one episode)",
                 fontsize=11, loc="left")
    a0.set_xlim(0, lim); a0.set_ylim(0, lim)
    _clean(a0)

    mcf = [v.mean() for v in by_pid(cf)]
    mcp = [v.mean() for v in by_pid(cp)]
    b1 = a1.bar(x - wbar/2, mcf, wbar, color=C_BLUE, edgecolor=SURF, linewidth=1.0,
                label="present (fused)")
    b2 = a1.bar(x + wbar/2, mcp, wbar, color=C_AQUA, edgecolor=SURF, linewidth=1.0,
                label="populating grid", hatch="///")
    a1.bar_label(b1, fmt="%.1f", padding=2, fontsize=7.5, color=INK2)
    a1.bar_label(b2, fmt="%.1f", padding=2, fontsize=7.5, color=INK2)
    a1.set_xticks(x); a1.set_xticklabels(pids, rotation=45, ha="right")
    a1.set_ylabel("classes per episode (mean)")
    a1.set_title("Per participant", fontsize=11, loc="left")
    a1.set_ylim(0, max(mcf) * 1.20)
    a1.legend(loc="upper center", ncol=2)
    _clean(a1)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    p = os.path.join(out_dir, "classes_present_vs_pop.png")
    fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig); paths.append(p)

    return paths


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--participants", type=str, default="",
                    help="comma list e.g. P01,P07; default = all in --splits")
    ap.add_argument("--splits", type=str, default="train,val,test")
    ap.add_argument("--observation_time", type=int, default=30)
    ap.add_argument("--anticipation_time", type=float, default=5)
    ap.add_argument("--future_time", type=float, default=180)
    ap.add_argument("--voxel_centers_dir", type=str, default=DEFAULT_VC_DIR)
    ap.add_argument("--out-dir", type=str, default=OUT_DIR)
    ap.add_argument("--no-plots", action="store_true")
    ap.add_argument("--from-csv", action="store_true",
                    help="skip computation; reload per_episode.csv and redraw "
                         "figures + report only")
    args = ap.parse_args()

    if args.from_csv:
        ep_path = os.path.join(args.out_dir, "per_episode.csv")
        rows = []
        with open(ep_path) as f:
            for d in csv.DictReader(f):
                rows.append({
                    "participant_id": d["participant_id"],
                    "video_id": d["video_id"],
                    "obs_start": int(d["obs_start"]),
                    "obs_end": int(d["obs_end"]),
                    "voxels_occupied": int(d["voxels_occupied"]),
                    "voxels_occupied_frac": float(d["voxels_occupied_frac"]),
                    "n_boxes_fused": int(d["n_boxes_fused"]),
                    "n_boxes_populating": int(d["n_boxes_populating"]),
                    "frac_boxes_populating": float(d["frac_boxes_populating"]),
                    "n_boxes_fallback_assigned": int(d["n_boxes_fallback_assigned"]),
                    "n_boxes_populating_total": int(d["n_boxes_populating_total"]),
                    "frac_boxes_populating_total": float(d["frac_boxes_populating_total"]),
                    "n_classes_fused": int(d["n_classes_fused"]),
                    "n_classes_populating": int(d["n_classes_populating"]),
                })
        print(f"Loaded {len(rows)} episodes from {ep_path}")
        print_report(rows)
        for fpath in make_figs(rows, args.out_dir):
            print(f"Wrote figure          : {fpath}")
        return

    participants = [p.strip() for p in args.participants.split(",") if p.strip()]
    splits = {s.strip() for s in args.splits.split(",") if s.strip()}

    print("Env voxel-grid occupancy analysis")
    print(f"  voxel_centers_dir = {args.voxel_centers_dir}")
    print(f"  obs={args.observation_time}s anticip={args.anticipation_time}s "
          f"future={args.future_time}s")
    print(f"  participants={participants or sorted(splits)}")

    rows = collect(participants, splits, args.observation_time,
                   args.anticipation_time, args.future_time,
                   args.voxel_centers_dir)
    if not rows:
        print("No episodes produced -- nothing to analyse.")
        return

    print_report(rows)
    ep_path, pp_path = write_csvs(rows, args.out_dir)
    print(f"\nWrote per-episode CSV : {ep_path}")
    print(f"Wrote per-participant : {pp_path}")

    if not args.no_plots:
        figs = make_figs(rows, args.out_dir)
        for f in figs:
            print(f"Wrote figure          : {f}")


if __name__ == "__main__":
    main()
