"""
Descriptive statistics of how many ground-truth (GT) interactions fall inside the
OBSERVATION window of each training episode.

Episode recap (see loader.py :: build_episodes)
-----------------------------------------------
Every episode is anchored on a GT interaction at time ``t`` (a mask timestamp
with a valid 3d_location). With the default hyperparameters
(observation_time=30, anticipation_time=5, future_time=180):

    prediction window : [t,               t + 180]        -> the GT to predict
    observation window: [t - 5 - 30,      t - 5] = [t-35, t-5]

The loader keeps an episode only if ``obs_end >= 1s`` AND the prediction window
contains >=1 GT interaction. This script asks the complementary question:

    Given the episodes training actually uses, how many *other* GT interactions
    occur inside the OBSERVATION window [obs_start, obs_end]?

"GT interactions" here are exactly the same events that form the prediction GT:
masks with a valid ``3d_location`` (see loader.build_video_events). We report two
notions of "how many":

  * events           : number of (object, location) mask instances in the window
                       (several objects can be logged at the same timestamp).
  * distinct moments : number of unique timestamps in the window.

Faithfulness
------------
Episodes are produced by importing the loader's own
``build_video_events`` / ``build_episodes`` (the same code the data loader runs),
so the observation windows analysed here are the windows training actually builds.
The only per-video filter reproduced is "feature file exists" (the loader's first
skip). The rarer "no voxel grid at all" drop (videos with neither SLAM nor Boxer)
needs the heavy spatial data and is not reproduced; pass --require-grid to apply
it too (slower, needs SLAM/Boxer on disk). Videos skipped this way are reported.

Usage
-----
  python analysis/obs_window_gt_stats.py
  python analysis/obs_window_gt_stats.py --observation_time 30 --anticipation_time 5 --future_time 180
  python analysis/obs_window_gt_stats.py --splits train
  python analysis/obs_window_gt_stats.py --save-csv        # per-episode CSV under ./out/
"""

import argparse
import os
import sys

import numpy as np

# Import the real episode code so we never drift from what training uses.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from loader import (  # noqa: E402
    FEATURES_BASE,
    PARTICIPANT_TO_SPLIT,
    _load_annotations,
    build_episodes,
    build_video_events,
)

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")


# --------------------------------------------------------------------------- #
# Per-episode observation-window counts
# --------------------------------------------------------------------------- #

def episode_obs_counts(events, episodes):
    """For each kept episode, count GT interactions inside [obs_start, obs_end].

    Returns a list of (n_events, n_distinct_moments) tuples, one per episode.
    ``events`` / ``episodes`` come straight from loader.build_video_events /
    loader.build_episodes, so the windows match training exactly.
    """
    if not events:
        return []
    ts = np.fromiter((e["timestamp"] for e in events), dtype=np.float64)
    ts.sort()
    out = []
    for ep in episodes:
        lo, hi = ep["obs_start"], ep["obs_end"]
        # inclusive both ends, mirroring the GT test (pred_start <= t <= pred_end)
        left = np.searchsorted(ts, lo, side="left")
        right = np.searchsorted(ts, hi, side="right")
        window = ts[left:right]
        out.append((int(window.size), int(np.unique(window).size)))
    return out


def collect_split(split, assoc, mask, obs_t, anticip_t, future_t,
                  require_grid, boxer=None, load_slam=None):
    """Return (rows, n_videos_used, n_no_feature, n_no_grid) for one split.

    rows: list of dicts, one per episode, with the per-window counts + context.
    """
    wanted_pids = {p for p, s in PARTICIPANT_TO_SPLIT.items() if s == split}
    video_ids = sorted(v for v in assoc if v.split("-")[0] in wanted_pids)

    rows = []
    n_used = n_no_feature = n_no_grid = 0
    for video_id in video_ids:
        pid = video_id.split("-")[0]
        feature_path = os.path.join(FEATURES_BASE, pid, f"{video_id}_features.pt")
        if not os.path.exists(feature_path):
            n_no_feature += 1
            continue

        if require_grid:
            # Reproduce the loader's "no voxel grid at all" drop.
            t0, slam_sec = load_slam(pid, video_id)
            has_slam = slam_sec is not None
            traj = slam_sec[np.isfinite(slam_sec).all(axis=1)] if has_slam else None
            vc = boxer.compute_and_cache_voxel_centers(video_id, traj)
            if vc is None:
                n_no_grid += 1
                continue

        events, anchors = build_video_events(assoc[video_id], mask[video_id])
        if not events:
            continue
        episodes = build_episodes(events, anchors, obs_t, anticip_t, future_t)
        if not episodes:
            continue

        counts = episode_obs_counts(events, episodes)
        for ep, (n_ev, n_mom) in zip(episodes, counts):
            rows.append({
                "split": split,
                "video_id": video_id,
                "participant_id": pid,
                "obs_start": ep["obs_start"],
                "obs_end": ep["obs_end"],
                "n_pred_gt": len(ep["gt_locations"]),
                "n_obs_events": n_ev,
                "n_obs_moments": n_mom,
            })
        n_used += 1

    return rows, n_used, n_no_feature, n_no_grid


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #

def describe(name, values):
    """Print descriptive statistics for a 1-D array of counts."""
    a = np.asarray(values, dtype=np.float64)
    n = a.size
    print(f"\n  {name}  (n={n} episodes)")
    if n == 0:
        print("    (no episodes)")
        return
    pcts = [0, 25, 50, 75, 90, 95, 99, 100]
    qs = np.percentile(a, pcts)
    print(f"    mean {a.mean():7.3f}   std {a.std():7.3f}   sum {int(a.sum())}")
    print("    " + "  ".join(f"p{p:<3d}={q:6.1f}" for p, q in zip(pcts, qs))
          .replace("p0  ", "min ").replace("p100", "max "))
    zero = int((a == 0).sum())
    print(f"    episodes with 0 : {zero} ({100.0 * zero / n:.1f}%)   "
          f">=1 : {n - zero} ({100.0 * (n - zero) / n:.1f}%)")


def print_histogram(name, values, cap=15):
    """Print a value->count table (values >= cap folded into one bucket)."""
    a = np.asarray(values, dtype=np.int64)
    if a.size == 0:
        return
    print(f"\n  {name} distribution (value : episodes)")
    top = min(cap, int(a.max()))
    for v in range(0, top + 1):
        c = int((a == v).sum())
        bar = "#" * min(60, int(round(60 * c / a.size)))
        print(f"    {v:>3d} : {c:6d}  {bar}")
    if a.max() > cap:
        c = int((a > cap).sum())
        print(f"   >{cap:>2d} : {c:6d}  {'#' * min(60, int(round(60 * c / a.size)))}")


def report(title, rows):
    print("\n" + "=" * 74)
    print(title)
    print("=" * 74)
    if not rows:
        print("  (no episodes)")
        return
    ev = [r["n_obs_events"] for r in rows]
    mom = [r["n_obs_moments"] for r in rows]
    describe("GT interaction events in observation window", ev)
    describe("Distinct interaction moments in observation window", mom)
    print_histogram("Events per observation window", ev)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--observation_time", type=int, default=30)
    ap.add_argument("--anticipation_time", type=float, default=5)
    ap.add_argument("--future_time", type=float, default=180)
    ap.add_argument("--splits", type=str, default="train,val,test")
    ap.add_argument("--require-grid", action="store_true",
                    help="also drop videos with no voxel grid (needs SLAM/Boxer; slower)")
    ap.add_argument("--save-csv", action="store_true",
                    help="write per-episode counts to ./out/obs_window_gt_stats.csv")
    args = ap.parse_args()

    print("Observation-window GT-interaction statistics")
    print(f"  observation_time={args.observation_time}s  "
          f"anticipation_time={args.anticipation_time}s  "
          f"future_time={args.future_time}s")
    print(f"  observation window = [t - {args.anticipation_time + args.observation_time:g}, "
          f"t - {args.anticipation_time:g}]  (t = episode anchor)")

    assoc, mask = _load_annotations()

    boxer = load_slam = None
    if args.require_grid:
        from boxer_env import BoxerEnvBuilder  # noqa: E402
        from loader import HDEpicInteractionDataset  # noqa: E402
        boxer = BoxerEnvBuilder()
        # Borrow the loader's SLAM reader without building a full dataset.
        load_slam = HDEpicInteractionDataset._load_slam.__get__(
            object.__new__(HDEpicInteractionDataset))

    all_rows = []
    for split in [s.strip() for s in args.splits.split(",") if s.strip()]:
        rows, n_used, n_no_feat, n_no_grid = collect_split(
            split, assoc, mask, args.observation_time, args.anticipation_time,
            args.future_time, args.require_grid, boxer, load_slam)
        print(f"\n[{split}] {len(rows)} episodes from {n_used} videos "
              f"({n_no_feat} videos skipped: no feature file"
              + (f", {n_no_grid} skipped: no grid" if args.require_grid else "") + ")")
        report(f"SPLIT: {split}", rows)
        all_rows.extend(rows)

    if len([s for s in args.splits.split(",") if s.strip()]) > 1:
        report("ALL SPLITS COMBINED", all_rows)

    if args.save_csv and all_rows:
        import csv
        os.makedirs(OUT_DIR, exist_ok=True)
        csv_path = os.path.join(OUT_DIR, "obs_window_gt_stats.csv")
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
            w.writeheader()
            w.writerows(all_rows)
        print(f"\nWrote per-episode counts: {csv_path}  ({len(all_rows)} rows)")


if __name__ == "__main__":
    main()
