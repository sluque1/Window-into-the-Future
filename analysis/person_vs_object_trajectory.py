"""
Person path length vs. object displacement, per track.

Question this answers
----------------------
For each object movement (track), how far did the *person* walk (SLAM path
length, integrated over the raw ~1kHz trajectory) during the track's
time_segment, versus how far the *object* net-displaced (straight line,
first valid 3d_location -> last)? If the person's path length is close to the
object's straight-line displacement, the person walked in roughly a straight
line and a straight-line proxy for the object's path is already adequate. If
the person's path length is much larger than the displacement, the person
took an indirect route (around furniture, to a different counter and back,
etc.) that a straight line would badly misrepresent -- there the full SLAM
trajectory (offset to the object's endpoints) is the better proxy.

By the triangle inequality, path_length >= displacement always, so every
point lies on/above the y=x diagonal in a (person_distance, displacement)
plot -- the question is how far above, and whether that gap grows in a way
that suggests a threshold.

Data
----
  * assoc_info.json / mask_info.json (object tracks; loading reuses
    analysis/object_movement_stats.py:_load_annotations).
  * SLAM closed_loop_trajectory.csv (person side), loaded RAW (not the
    loader's 1Hz-resampled per_sec positions) because many tracks last under
    a second -- 1Hz would alias short movements. Sampling is ~1kHz, so this
    integrates the true path length inside [time_segment[0], time_segment[1]].

Usage
-----
  python analysis/person_vs_object_trajectory.py
  python analysis/person_vs_object_trajectory.py --save-csv --plot
"""

import argparse
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "out")
REPO_ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, REPO_ROOT)
from object_movement_stats import _load_annotations  # noqa: E402
from slam_utils import load_raw_slam, path_length as person_path_length  # noqa: E402


# --------------------------------------------------------------------------- #
# Per-track extraction: person distance vs. object displacement
# --------------------------------------------------------------------------- #

def build_records():
    # Walk assoc/mask directly (rather than reusing track_records) to get
    # (displacement, duration, time_segment) together per track.
    assoc, mask = _load_annotations()

    records = []
    n_out_of_range = 0
    videos_no_slam = set()

    for video_id, assocs in assoc.items():
        v_mask = mask.get(video_id, {})
        pid = video_id.split('-')[0]
        traj = load_raw_slam(pid, video_id)
        if traj is None:
            videos_no_slam.add(video_id)
            continue
        t0, rel_times, xyz = traj

        for aid, ainfo in assocs.items():
            for tr in ainfo["tracks"]:
                ts = tr["time_segment"]
                t_start, t_end = float(ts[0]), float(ts[1])
                duration = t_end - t_start
                if duration <= 0:
                    continue

                pts = []
                for mid in tr["masks"]:
                    m = v_mask.get(mid)
                    if m is None:
                        continue
                    loc = m.get("3d_location")
                    if loc is None or len(loc) != 3 or any(
                            not isinstance(x, (int, float)) or
                            (isinstance(x, float) and math.isnan(x)) for x in loc):
                        continue
                    pts.append((m["frame_number"], np.asarray(loc, dtype=np.float64)))
                if len(pts) < 2:
                    continue
                pts.sort(key=lambda p: p[0])
                coords = np.stack([p[1] for p in pts])
                displacement = float(np.linalg.norm(coords[-1] - coords[0]))

                pdist = person_path_length(rel_times, xyz, t_start, t_end)
                if pdist is None:
                    n_out_of_range += 1
                    continue

                records.append({
                    "video_id": video_id,
                    "association_id": aid,
                    "track_id": tr["track_id"],
                    "name": ainfo.get("name", ""),
                    "duration": duration,
                    "displacement": displacement,
                    "person_distance": pdist,
                })

    return records, videos_no_slam, n_out_of_range


# --------------------------------------------------------------------------- #
# Breaking-point search: segmented (piecewise-constant-ratio) analysis.
# --------------------------------------------------------------------------- #

def analyse_breakpoint(x, y):
    """Report binned median ratio y/x and a simple 2-regime elbow search.

    x = person_distance, y = object displacement. Since y <= x always
    (triangle inequality), we look at the ratio y/x as a function of x: near
    1.0 means "the person walked in a roughly straight line" (a straight-line
    object proxy is already accurate); low ratio means the person's actual
    path deviated a lot from a straight line (use the real trajectory).
    """
    x = np.asarray(x)
    y = np.asarray(y)
    order = np.argsort(x)
    x, y = x[order], y[order]
    ratio = y / np.maximum(x, 1e-6)

    print("\n" + "-" * 74)
    print("Binned median ratio  (object displacement / person path length)")
    print("-" * 74)
    edges = [0, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, np.inf]
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (x >= lo) & (x < hi)
        n = m.sum()
        if n == 0:
            continue
        med_ratio = np.median(ratio[m])
        hi_s = f"{hi:6.2f}" if math.isfinite(hi) else "  inf "
        print(f"  person_dist in [{lo:5.2f}, {hi_s})  n={n:5d}  "
              f"median ratio={med_ratio:.3f}  median disp={np.median(y[m]):.3f} m")

    # Simple change-point search: threshold t splits x into "short" / "long"
    # walks; report the median ratio on each side for a grid of candidate t,
    # and flag where the drop in ratio is steepest (a natural elbow).
    print("\n" + "-" * 74)
    print("Change-point scan (median ratio below vs. above threshold t)")
    print("-" * 74)
    candidates = np.percentile(x, np.arange(5, 100, 5))
    best_gap, best_t = -1, None
    for t in candidates:
        below = ratio[x < t]
        above = ratio[x >= t]
        if len(below) < 20 or len(above) < 20:
            continue
        gap = np.median(below) - np.median(above)
        marker = ""
        if gap > best_gap:
            best_gap, best_t = gap, t
            marker = "  <-- largest gap so far"
        print(f"  t={t:6.3f} m  n_below={len(below):5d} med_ratio={np.median(below):.3f}"
              f"   n_above={len(above):5d} med_ratio={np.median(above):.3f}"
              f"   gap={gap:.3f}{marker}")

    print(f"\nLargest below/above ratio gap at t ~= {best_t:.3f} m "
          f"(gap={best_gap:.3f})")
    return best_t


# --------------------------------------------------------------------------- #
# Plot
# --------------------------------------------------------------------------- #

def make_scatter(records, out_path, breakpoint_t=None, xmax=None, ymax=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = np.array([r["person_distance"] for r in records])
    y = np.array([r["displacement"] for r in records])

    xlim = xmax if xmax is not None else x.max() * 1.05
    ylim = ymax if ymax is not None else y.max() * 1.05

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))

    for ax, logscale in zip(axes, [False, True]):
        ax.scatter(x, y, s=6, alpha=0.25, color="#3b7dd8", edgecolors="none")
        ref = np.linspace(1e-3 if logscale else 0, max(xlim, ylim), 200)
        ax.plot(ref, ref, color="black", linewidth=1.2, linestyle="--",
                 label="y = x  (straight-line path)")
        if breakpoint_t is not None:
            ax.axvline(breakpoint_t, color="#d8703b", linewidth=1.5,
                        label=f"candidate breakpoint = {breakpoint_t:.2f} m")
        ax.set_xlabel("person path length during track (m)")
        ax.set_ylabel("object displacement, straight line (m)")
        ax.legend(frameon=False, fontsize=9, loc="upper left")
        ax.grid(True, linestyle=":", alpha=0.4)
        ax.set_axisbelow(True)
        if logscale:
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlim(1e-3, xlim)
            ax.set_ylim(1e-3, ylim)
            ax.set_title("log-log")
        else:
            ax.set_xlim(0, xlim)
            ax.set_ylim(0, ylim)
            ax.set_title("linear")

    fig.suptitle(f"Person path length vs. object displacement per track  (n={len(x)})",
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150)
    print(f"\nWrote plot: {out_path}")


# --------------------------------------------------------------------------- #

def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--save-csv", action="store_true")
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--xmax", type=float, default=None,
                    help="cap the person-path-length axis at this many metres (default: auto)")
    ap.add_argument("--ymax", type=float, default=None,
                    help="cap the object-displacement axis at this many metres (default: auto)")
    args = ap.parse_args()

    records, videos_no_slam, n_out_of_range = build_records()

    print("=" * 74)
    print("Person path length vs. object displacement, per track")
    print("=" * 74)
    print(f"  tracks with valid records      : {len(records)}")
    print(f"  videos skipped (no SLAM)       : {len(videos_no_slam)}")
    print(f"  tracks skipped (time out of SLAM range): {n_out_of_range}")

    x = [r["person_distance"] for r in records]
    y = [r["displacement"] for r in records]

    best_t = analyse_breakpoint(x, y)

    if args.save_csv:
        import csv
        os.makedirs(OUT_DIR, exist_ok=True)
        path = os.path.join(OUT_DIR, "person_vs_object_trajectory.csv")
        fields = ["video_id", "association_id", "track_id", "name", "duration",
                  "person_distance", "displacement"]
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(records)
        print(f"\nWrote per-track CSV: {path}  ({len(records)} rows)")

    if args.plot:
        make_scatter(records, os.path.join(OUT_DIR, "person_vs_object_trajectory.png"),
                     breakpoint_t=best_t, xmax=args.xmax, ymax=args.ymax)


if __name__ == "__main__":
    main()
