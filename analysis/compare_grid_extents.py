"""
Compare the per-video voxel-grid *extent* under the two env sources:

  * boxer_grid : AABB( Boxer OBB corners  UNION  SLAM per-second trajectory )
  * detic_grid : AABB( Detic OBB corners  UNION  SLAM per-second trajectory )

The grid extent is the axis-aligned bounding box that the 16^3 voxel grid covers
(see boxer_env._grid_from_points). The ONLY difference between the two is the
object-box source:

  boxer : /data/.../boxer_obbs/<video_id>/boxer_3dbbs.csv          (10 Hz-downsampled)
  detic : /data/.../object_bounding_boxes_certainty/<pid>/<video_id>_object_bbs_obb_newparams.pkl

For faithfulness this reuses the exact extent code the loader/trainer use
(boxer_env + HDEpicInteractionDataset._load_slam), so what we report here is what
training actually builds.

Outputs (written next to this script, under ./out/):
  grid_extents.csv               one row per video: min/max/size per axis, volume,
                                  diagonal, for both sources + their differences.
  grid_extents_overview.png      cross-video summary (volume + per-axis size).
  grid_extents_per_video.pdf     per-video XY & XZ AABB overlays (boxer vs detic).

Usage:
  python analysis/compare_grid_extents.py            # all feature videos
  python analysis/compare_grid_extents.py --limit 20 # first 20 (quick look)
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

# Import the real extent code so we never drift from what training uses.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import boxer_env as be
from boxer_env import BoxerEnvBuilder, load_detic_obb_corners, _obb_corners_vectorized
from loader import HDEpicInteractionDataset, FEATURES_BASE

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
GRID = be.GRID_SIZE  # 16


# ---------------------------------------------------------------------------
# Extent helpers
# ---------------------------------------------------------------------------

def _aabb(points):
    """(min_xyz, max_xyz, size_xyz) for an (N,3) point set, or None if empty.

    Applies the same degenerate-axis guard (>= 1e-3) as boxer_env._grid_from_points
    so the reported extent matches the grid that training actually builds.
    """
    if points is None or len(points) == 0:
        return None
    mn = points.min(axis=0)
    size = np.maximum(points.max(axis=0) - mn, 1e-3)
    mx = mn + size
    return mn, mx, size


def _list_feature_videos():
    """All <video_id> that have an EgoVLPv2 feature file (the model's video set)."""
    vids = []
    for pid in sorted(os.listdir(FEATURES_BASE)):
        pdir = os.path.join(FEATURES_BASE, pid)
        if not os.path.isdir(pdir):
            continue
        for fn in sorted(os.listdir(pdir)):
            if fn.endswith("_features.pt"):
                vids.append(fn[:-len("_features.pt")])
    return sorted(vids)


def collect_extents(limit=None):
    """Compute boxer & detic AABB extents for every feature video.

    Returns a DataFrame with one row per video. Missing-source rows have NaN for
    that source's columns.
    """
    boxer = BoxerEnvBuilder()
    # Bare dataset instance purely to reuse its faithful SLAM loader (no __init__).
    slam_loader = HDEpicInteractionDataset.__new__(HDEpicInteractionDataset)

    video_ids = _list_feature_videos()
    if limit:
        video_ids = video_ids[:limit]

    rows = []
    for i, vid in enumerate(video_ids, 1):
        pid = vid.split('-')[0]
        print(f"[{i}/{len(video_ids)}] {vid}", flush=True)

        # --- SLAM per-second trajectory (finite positions only), shared by both ---
        _, slam_sec = slam_loader._load_slam(pid, vid)
        has_slam = slam_sec is not None
        traj = (slam_sec[np.isfinite(slam_sec).all(axis=1)]
                if has_slam else np.empty((0, 3)))

        # --- Boxer corners (10 Hz-downsampled CSV, exactly as training loads) ---
        boxer_df = boxer.load_boxer_csv(vid)
        has_boxer = boxer_df is not None and len(boxer_df) > 0
        boxer_corners = (_obb_corners_vectorized(boxer_df)
                         if has_boxer else np.empty((0, 3)))

        # --- Detic corners ---
        detic_corners = load_detic_obb_corners(vid)
        has_detic = detic_corners is not None and len(detic_corners) > 0
        if detic_corners is None:
            detic_corners = np.empty((0, 3))

        boxer_aabb = _aabb(np.concatenate([boxer_corners, traj], axis=0))
        detic_aabb = _aabb(np.concatenate([detic_corners, traj], axis=0))

        row = {
            'video_id': vid, 'participant_id': pid,
            'has_boxer': has_boxer, 'has_detic': has_detic, 'has_slam': has_slam,
            'n_boxer_obb': len(boxer_corners) // 8,
            'n_detic_obb': len(detic_corners) // 8,
            'n_traj_pts': len(traj),
        }
        for tag, aabb in (('boxer', boxer_aabb), ('detic', detic_aabb)):
            if aabb is None:
                for k in ('min', 'max', 'size'):
                    for a in 'xyz':
                        row[f'{tag}_{k}_{a}'] = np.nan
                row[f'{tag}_volume'] = np.nan
                row[f'{tag}_diag'] = np.nan
                continue
            mn, mx, size = aabb
            for j, a in enumerate('xyz'):
                row[f'{tag}_min_{a}'] = mn[j]
                row[f'{tag}_max_{a}'] = mx[j]
                row[f'{tag}_size_{a}'] = size[j]
            row[f'{tag}_volume'] = float(np.prod(size))
            row[f'{tag}_diag'] = float(np.linalg.norm(size))

        # --- Differences (detic - boxer), only where both exist ---
        if boxer_aabb is not None and detic_aabb is not None:
            for a in 'xyz':
                row[f'diff_size_{a}'] = row[f'detic_size_{a}'] - row[f'boxer_size_{a}']
            row['diff_volume'] = row['detic_volume'] - row['boxer_volume']
            row['volume_ratio'] = (row['detic_volume'] / row['boxer_volume']
                                   if row['boxer_volume'] > 0 else np.nan)
            # Fraction of each grid's AABB volume NOT covered by the other (IoU-style).
            lo = np.maximum([row[f'boxer_min_{a}'] for a in 'xyz'],
                            [row[f'detic_min_{a}'] for a in 'xyz'])
            hi = np.minimum([row[f'boxer_max_{a}'] for a in 'xyz'],
                            [row[f'detic_max_{a}'] for a in 'xyz'])
            inter = float(np.prod(np.maximum(hi - lo, 0.0)))
            union = row['boxer_volume'] + row['detic_volume'] - inter
            row['aabb_iou'] = inter / union if union > 0 else np.nan
        else:
            for a in 'xyz':
                row[f'diff_size_{a}'] = np.nan
            row['diff_volume'] = np.nan
            row['volume_ratio'] = np.nan
            row['aabb_iou'] = np.nan

        rows.append(row)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_overview(df, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    both = df[df['has_boxer'] & df['has_detic']].reset_index(drop=True)
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    # (0,0) AABB volume per video, boxer vs detic
    ax = axes[0, 0]
    x = np.arange(len(both))
    ax.bar(x - 0.2, both['boxer_volume'], width=0.4, label='boxer', color='#3b7dd8')
    ax.bar(x + 0.2, both['detic_volume'], width=0.4, label='detic', color='#e07b39')
    ax.set_title(f"AABB volume per video (n={len(both)} with both sources)")
    ax.set_ylabel("volume (m^3)"); ax.set_xlabel("video index"); ax.legend()

    # (0,1) volume ratio detic/boxer
    ax = axes[0, 1]
    ax.bar(x, both['volume_ratio'], color='#6aa84f')
    ax.axhline(1.0, color='k', lw=1, ls='--')
    ax.set_title("volume ratio  detic / boxer")
    ax.set_ylabel("ratio"); ax.set_xlabel("video index")

    # (1,0) per-axis size difference (detic - boxer) distribution
    ax = axes[1, 0]
    ax.boxplot([both['diff_size_x'], both['diff_size_y'], both['diff_size_z']],
               tick_labels=['x', 'y', 'z'])
    ax.axhline(0.0, color='k', lw=1, ls='--')
    ax.set_title("per-axis extent difference  (detic - boxer)")
    ax.set_ylabel("meters")

    # (1,1) AABB IoU distribution
    ax = axes[1, 1]
    ax.hist(both['aabb_iou'].dropna(), bins=20, color='#a64d79')
    ax.set_title("AABB IoU (boxer vs detic extent overlap)")
    ax.set_xlabel("IoU"); ax.set_ylabel("# videos")

    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_per_video(df, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.patches import Rectangle

    def rect(ax, mn_a, mn_b, sz_a, sz_b, color, label):
        ax.add_patch(Rectangle((mn_a, mn_b), sz_a, sz_b, fill=False,
                               edgecolor=color, lw=2, label=label))

    with PdfPages(path) as pdf:
        for _, r in df.iterrows():
            if not (r['has_boxer'] or r['has_detic']):
                continue
            fig, (axxy, axxz) = plt.subplots(1, 2, figsize=(13, 5.5))
            for tag, color in (('boxer', '#3b7dd8'), ('detic', '#e07b39')):
                if not r[f'has_{tag}'] and not (tag == 'boxer' and r['has_boxer']):
                    if np.isnan(r[f'{tag}_min_x']):
                        continue
                if np.isnan(r[f'{tag}_min_x']):
                    continue
                # top-down XY
                rect(axxy, r[f'{tag}_min_x'], r[f'{tag}_min_y'],
                     r[f'{tag}_size_x'], r[f'{tag}_size_y'], color, tag)
                # side XZ
                rect(axxz, r[f'{tag}_min_x'], r[f'{tag}_min_z'],
                     r[f'{tag}_size_x'], r[f'{tag}_size_z'], color, tag)
            for ax, (a, b) in ((axxy, ('X', 'Y')), (axxz, ('X', 'Z'))):
                ax.set_xlabel(f"{a} (m)"); ax.set_ylabel(f"{b} (m)")
                ax.autoscale_view(); ax.set_aspect('equal', 'datalim')
                ax.legend(loc='upper right', fontsize=8)
            ratio = r['volume_ratio']
            iou = r['aabb_iou']
            fig.suptitle(f"{r['video_id']}   "
                         f"boxer_vol={r['boxer_volume']:.1f}  detic_vol={r['detic_volume']:.1f}  "
                         f"ratio={ratio:.2f}  IoU={iou:.2f}"
                         if not np.isnan(ratio) else
                         f"{r['video_id']}   (only one source present)")
            axxy.set_title("top-down (X-Y)"); axxz.set_title("side (X-Z)")
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def print_summary(df):
    both = df[df['has_boxer'] & df['has_detic']]
    print("\n================ SUMMARY ================")
    print(f"feature videos            : {len(df)}")
    print(f"  with Boxer OBBs         : {int(df['has_boxer'].sum())}")
    print(f"  with Detic OBBs         : {int(df['has_detic'].sum())}")
    print(f"  with SLAM trajectory    : {int(df['has_slam'].sum())}")
    print(f"  with BOTH obb sources   : {len(both)}")
    if len(both):
        print("\nAABB volume (m^3):")
        print(f"  boxer  mean={both['boxer_volume'].mean():.1f}  median={both['boxer_volume'].median():.1f}")
        print(f"  detic  mean={both['detic_volume'].mean():.1f}  median={both['detic_volume'].median():.1f}")
        print(f"  detic/boxer volume ratio: mean={both['volume_ratio'].mean():.2f} "
              f"median={both['volume_ratio'].median():.2f} "
              f"[{both['volume_ratio'].min():.2f}, {both['volume_ratio'].max():.2f}]")
        print(f"  AABB IoU: mean={both['aabb_iou'].mean():.2f} median={both['aabb_iou'].median():.2f}")
        for a in 'xyz':
            d = both[f'diff_size_{a}']
            print(f"  extent diff {a} (detic-boxer): mean={d.mean():+.2f}m median={d.median():+.2f}m")
    print("=========================================\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--limit', type=int, default=None,
                    help="only process the first N videos (quick look)")
    ap.add_argument('--no-plots', action='store_true')
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    df = collect_extents(limit=args.limit)

    csv_path = os.path.join(OUT_DIR, "grid_extents.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nwrote {csv_path}")

    print_summary(df)

    if not args.no_plots:
        ov = os.path.join(OUT_DIR, "grid_extents_overview.png")
        plot_overview(df, ov)
        print(f"wrote {ov}")
        pv = os.path.join(OUT_DIR, "grid_extents_per_video.pdf")
        plot_per_video(df, pv)
        print(f"wrote {pv}")


if __name__ == "__main__":
    main()