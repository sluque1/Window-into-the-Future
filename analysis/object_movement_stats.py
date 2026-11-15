"""
How much do objects move per track, in DISTANCE and TIME terms?

Data source (HD-EPIC "Digital Twin: Scene & Object Movements")
--------------------------------------------------------------
Two annotation files, linked by mask id (see the dataset README):

  assoc_info.json : per video, object "associations" -> list of "tracks".
                    Each track is a single movement of an object and has a
                    `time_segment` [start_s, end_s] and a list of `masks`.
  mask_info.json  : per video, per mask -> `frame_number`, `3d_location`
                    [x,y,z] (metres, Aria world frame), `bbox`, `fixture`.

A single object movement (track) is annotated at movement onset and end, so a
track normally carries two masks: the object's 3D world position when it is
picked up and when it is put down.

What this script measures, per track
------------------------------------
  * duration  (TIME, seconds)   = time_segment[1] - time_segment[0].
  * distance  (DISTANCE, metres):
      masks with a valid 3d_location are ordered by frame_number, then
        - displacement = straight-line ||last - first||  (net distance moved)
        - path_length  = sum of consecutive step distances (>= displacement;
                         differs from it only for the rare >2-mask tracks).
  * speed     = displacement / duration (m/s), reported as a bonus.

Validity filter (as requested)
------------------------------
"Exclude any cases in which there is not a valid 3d location for the
interaction." A track is kept for the DISTANCE analysis only if it has >= 2
masks with a valid 3d_location (a movement needs a start and an end point).
Validity uses the loader's own `_has_valid_location` (3 finite numbers), so the
filter matches what training treats as a usable interaction. In practice the
only invalid locations in this data are `null` entries.

The TIME analysis is reported on the same kept set (so the two distributions
describe the same tracks), and additionally over ALL tracks for reference.

Usage
-----
  python analysis/object_movement_stats.py
  python analysis/object_movement_stats.py --save-csv --plot
  python analysis/object_movement_stats.py --distance path   # use path length
"""

import argparse
import json
import math
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "out")

ANNOT_DIR = os.environ.get(
    "ANNOT_DIR",
    os.environ.get("HDEPIC_ROOT", "/data/video_datasets/HD-EPIC")
    + "/hd-epic-annotations/scene-and-object-movements")


def _load_annotations():
    """Load assoc_info.json + mask_info.json (the two movement annotation files)."""
    with open(os.path.join(ANNOT_DIR, "assoc_info.json")) as f:
        assoc = json.load(f)
    with open(os.path.join(ANNOT_DIR, "mask_info.json")) as f:
        mask = json.load(f)
    return assoc, mask


def _has_valid_location(loc):
    """True iff loc is a 3-element list of finite numbers.

    Mirrors loader._has_valid_location so this analysis treats a "valid 3d
    location for the interaction" exactly as the training pipeline does, without
    importing the (torch-heavy) loader module.
    """
    if loc is None:
        return False
    try:
        if len(loc) != 3:
            return False
    except TypeError:
        return False
    for x in loc:
        if not isinstance(x, (int, float)) or (isinstance(x, float) and math.isnan(x)):
            return False
    return True


# --------------------------------------------------------------------------- #
# Per-track extraction
# --------------------------------------------------------------------------- #

def track_records(assoc, mask):
    """One record per track across all videos.

    Returns a list of dicts with:
      video_id, association_id, track_id, name,
      duration, n_masks, n_valid_masks,
      displacement, path_length  (None when < 2 valid masks).
    """
    rows = []
    for video_id, assocs in assoc.items():
        v_mask = mask.get(video_id, {})
        for aid, ainfo in assocs.items():
            for tr in ainfo["tracks"]:
                ts = tr["time_segment"]
                duration = float(ts[1]) - float(ts[0])

                # Valid points ordered along time (frame_number).
                pts = []
                for mid in tr["masks"]:
                    m = v_mask.get(mid)
                    if m is None:
                        continue
                    loc = m.get("3d_location")
                    if _has_valid_location(loc):
                        pts.append((m["frame_number"], np.asarray(loc, dtype=np.float64)))
                pts.sort(key=lambda p: p[0])

                displacement = path_length = None
                if len(pts) >= 2:
                    coords = np.stack([p[1] for p in pts])
                    displacement = float(np.linalg.norm(coords[-1] - coords[0]))
                    steps = np.linalg.norm(np.diff(coords, axis=0), axis=1)
                    path_length = float(steps.sum())

                rows.append({
                    "video_id": video_id,
                    "association_id": aid,
                    "track_id": tr["track_id"],
                    "name": ainfo.get("name", ""),
                    "duration": duration,
                    "n_masks": len(tr["masks"]),
                    "n_valid_masks": len(pts),
                    "displacement": displacement,
                    "path_length": path_length,
                })
    return rows


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #

def describe(name, values, unit=""):
    """Print descriptive statistics for a 1-D array."""
    a = np.asarray(values, dtype=np.float64)
    n = a.size
    u = f" {unit}" if unit else ""
    print(f"\n  {name}  (n={n})")
    if n == 0:
        print("    (empty)")
        return
    pcts = [0, 1, 5, 25, 50, 75, 90, 95, 99, 100]
    qs = np.percentile(a, pcts)
    print(f"    mean {a.mean():.3f}{u}   std {a.std():.3f}{u}   sum {a.sum():.1f}{u}")
    line = "  ".join(f"p{p:<3d}={q:8.3f}" for p, q in zip(pcts, qs))
    print("    " + line.replace("p0  ", "min ").replace("p100", "max "))


def ascii_hist(name, values, bins, log=False):
    """Print an ASCII histogram over the given (right-edge) bin boundaries."""
    a = np.asarray(values, dtype=np.float64)
    if a.size == 0:
        return
    edges = np.asarray([0.0] + list(bins) + [np.inf])
    counts, _ = np.histogram(a, bins=edges)
    total = counts.sum()
    scale = "log " if log else ""
    print(f"\n  {name} distribution ({scale}bins, n={total})")
    peak = counts.max() or 1
    for i, c in enumerate(counts):
        lo, hi = edges[i], edges[i + 1]
        label = f"[{lo:6.2f}, {hi:6.2f})" if math.isfinite(hi) else f"[{lo:6.2f},   inf)"
        bar = "#" * int(round(50 * c / peak))
        print(f"    {label}  {c:6d} ({100.0 * c / total:5.1f}%)  {bar}")


# --------------------------------------------------------------------------- #
# Plotting
# --------------------------------------------------------------------------- #

def make_plot(dist, dur, dist_kind, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dist = np.asarray(dist, dtype=np.float64)
    dur = np.asarray(dur, dtype=np.float64)
    c_dist, c_time = "#3b7dd8", "#d8703b"

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    def _panel(ax, data, color, xlabel, title, logx):
        d = data[data > 0] if logx else data
        if logx:
            bins = np.logspace(np.log10(d.min()), np.log10(d.max()), 40)
            ax.set_xscale("log")
        else:
            bins = np.linspace(d.min(), np.percentile(d, 99), 40)
        ax.hist(d, bins=bins, color=color, alpha=0.8, edgecolor="white", linewidth=0.3)
        ax.axvline(np.median(d), color="black", linestyle="--", linewidth=1.2,
                   label=f"median {np.median(d):.2f}")
        ax.axvline(d.mean(), color="black", linestyle=":", linewidth=1.2,
                   label=f"mean {d.mean():.2f}")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("tracks")
        ax.set_title(title)
        ax.legend(frameon=False, fontsize=9)
        ax.grid(True, linestyle=":", alpha=0.4)
        ax.set_axisbelow(True)

    _panel(axes[0, 0], dist, c_dist, f"{dist_kind} (m)",
           f"Object movement {dist_kind} (linear, clipped at p99)", logx=False)
    _panel(axes[0, 1], dist, c_dist, f"{dist_kind} (m)",
           f"Object movement {dist_kind} (log scale)", logx=True)
    _panel(axes[1, 0], dur, c_time, "duration (s)",
           "Movement duration (linear, clipped at p99)", logx=False)
    _panel(axes[1, 1], dur, c_time, "duration (s)",
           "Movement duration (log scale)", logx=True)

    fig.suptitle(f"HD-EPIC object movement per track  (n={dist.size} valid tracks)",
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150)
    print(f"\nWrote plot: {out_path}")


# --------------------------------------------------------------------------- #

def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--distance", choices=["displacement", "path"], default="displacement",
                    help="distance metric: net displacement (default) or full path length")
    ap.add_argument("--save-csv", action="store_true",
                    help="write per-track values to ./out/object_movement_tracks.csv")
    ap.add_argument("--plot", action="store_true",
                    help="write a histogram figure to ./out/object_movement_stats.png")
    args = ap.parse_args()

    assoc, mask = _load_annotations()
    rows = track_records(assoc, mask)

    n_total = len(rows)
    n_videos = len({r["video_id"] for r in rows})
    valid = [r for r in rows if r["displacement"] is not None]
    excluded = n_total - len(valid)

    dist_key = "displacement" if args.distance == "displacement" else "path_length"
    dist_name = "displacement" if args.distance == "displacement" else "path length"

    print("=" * 74)
    print("HD-EPIC object movement per track")
    print("=" * 74)
    print(f"  videos                         : {n_videos}")
    print(f"  tracks (total)                 : {n_total}")
    print(f"  tracks kept (>=2 valid 3d locs): {len(valid)}")
    print(f"  tracks excluded (no valid move): {excluded} "
          f"({100.0 * excluded / n_total:.1f}%)")

    # Why tracks are excluded, by number of valid 3d locations.
    from collections import Counter
    nvalid_hist = Counter(r["n_valid_masks"] for r in rows if r["displacement"] is None)
    if nvalid_hist:
        breakdown = ", ".join(f"{k} valid loc: {v}" for k, v in sorted(nvalid_hist.items()))
        print(f"    excluded breakdown            : {breakdown}")

    dist = [r[dist_key] for r in valid]
    dur_valid = [r["duration"] for r in valid]
    dur_all = [r["duration"] for r in rows]

    print("\n" + "-" * 74)
    print(f"DISTANCE per track  ({dist_name}, straight-line = displacement)")
    print("-" * 74)
    describe(f"movement {dist_name}", dist, unit="m")
    ascii_hist(f"movement {dist_name}", dist,
               bins=[0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0])

    print("\n" + "-" * 74)
    print("TIME per track  (movement duration)")
    print("-" * 74)
    describe("duration (valid-distance tracks)", dur_valid, unit="s")
    describe("duration (ALL tracks)", dur_all, unit="s")
    ascii_hist("duration (valid-distance tracks)", dur_valid,
               bins=[1, 2, 3, 5, 8, 12, 20, 30, 60, 120])

    # Bonus: movement speed.
    speed = [r[dist_key] / r["duration"] for r in valid if r["duration"] > 0]
    print("\n" + "-" * 74)
    print("SPEED per track  (displacement / duration)  [bonus]")
    print("-" * 74)
    describe("speed", speed, unit="m/s")

    if args.save_csv:
        import csv
        os.makedirs(OUT_DIR, exist_ok=True)
        path = os.path.join(OUT_DIR, "object_movement_tracks.csv")
        fields = ["video_id", "association_id", "track_id", "name", "duration",
                  "n_masks", "n_valid_masks", "displacement", "path_length"]
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)
        print(f"\nWrote per-track CSV: {path}  ({len(rows)} rows)")

    if args.plot:
        make_plot(dist, dur_valid, dist_name,
                  os.path.join(OUT_DIR, "object_movement_stats.png"))


if __name__ == "__main__":
    main()
