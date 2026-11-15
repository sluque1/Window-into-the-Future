"""
How many object-movement tracks overlap a 30s observation window?

Informs the choice of `max_trajectory_tokens` in loader.py /
trajectory_tokens.py: counts, for every real episode the training pipeline
builds (same anchors/obs windows as loader.build_episodes), how many valid
2-mask tracks overlap that window -- i.e. how many trajectory tokens
loader.py would try to build before any truncation.

Deliberately does not import loader.py (which pulls in torch) -- reimplements
the small amount of needed annotation-parsing logic in pure Python/numpy, to
stay fast and dependency-light like the other analysis/*.py scripts. The
count only depends on observation_time/anticipation_time (window shape), not
future_time (which only affects GT labels, not which tracks overlap the
observation window), so future_time is not a parameter here.

Usage
-----
  python analysis/trajectory_token_count.py
  python analysis/trajectory_token_count.py --observation-time 30 --anticipation-time 5
  python analysis/trajectory_token_count.py --plot
"""

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "out")
sys.path.insert(0, HERE)
from object_movement_stats import _load_annotations, _has_valid_location  # noqa: E402

FPS = 30


def build_video_tracks(v_assoc, v_mask):
    """Mirrors loader.build_video_tracks: usable (t_start, t_end) tracks."""
    tracks = []
    for info in v_assoc.values():
        for track in info["tracks"]:
            if len(track["masks"]) != 2:
                continue
            pts = []
            for mid in track["masks"]:
                m = v_mask.get(mid)
                if m is None or not _has_valid_location(m.get("3d_location")):
                    continue
                pts.append((m["frame_number"], m["3d_location"]))
            if len(pts) != 2:
                continue
            pts.sort(key=lambda p: p[0])
            t_start, t_end = pts[0][0] / FPS, pts[1][0] / FPS
            if t_end <= t_start:
                continue
            tracks.append((t_start, t_end))
    return tracks


def build_episode_windows(v_assoc, v_mask, observation_time, anticipation_time):
    """Mirrors loader.build_video_events + build_episodes' obs windows.

    Only anchors + obs windows are needed (not GT/future_time), since token
    counts don't depend on future_time.
    """
    anchors = set()
    for info in v_assoc.values():
        for track in info["tracks"]:
            for mid in track["masks"]:
                m = v_mask.get(mid)
                if m is None or not _has_valid_location(m.get("3d_location")):
                    continue
                anchors.add(m["frame_number"] / FPS)

    windows = []
    for anchor_t in sorted(anchors):
        obs_end = max(0.0, anchor_t - anticipation_time)
        if obs_end < 1.0:
            continue
        obs_start = max(0.0, obs_end - observation_time)
        windows.append((obs_start, obs_end))
    return windows


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--observation-time", type=float, default=30.0)
    ap.add_argument("--anticipation-time", type=float, default=5.0)
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()

    assoc, mask = _load_annotations()
    counts = []
    for video_id, v_assoc in assoc.items():
        v_mask = mask.get(video_id, {})
        tracks = build_video_tracks(v_assoc, v_mask)
        if not tracks:
            continue
        windows = build_episode_windows(v_assoc, v_mask,
                                        args.observation_time, args.anticipation_time)
        for obs_start, obs_end in windows:
            n = sum(1 for (t0, t1) in tracks if t0 <= obs_end and t1 >= obs_start)
            counts.append(n)

    counts = np.asarray(counts)

    print("=" * 74)
    print(f"Trajectory-token count per episode  "
          f"(obs={args.observation_time}s, anticip={args.anticipation_time}s)")
    print("=" * 74)
    print(f"  episodes: {len(counts)}")
    pcts = [0, 5, 10, 25, 50, 75, 90, 95, 99, 100]
    qs = np.percentile(counts, pcts)
    print("  percentiles: " + "  ".join(f"p{p}={q:.0f}" for p, q in zip(pcts, qs)))
    print(f"  mean: {counts.mean():.2f}   zero-track episodes: {(counts == 0).mean() * 100:.1f}%")

    print("\n  cap -> % episodes truncated (had more tracks than the cap), "
          "mean tokens lost among those")
    for cap in [1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 24, 32]:
        truncated = counts > cap
        pct = truncated.mean() * 100
        lost = (counts[truncated] - cap).mean() if truncated.any() else 0.0
        print(f"    cap={cap:3d}   truncated={pct:5.1f}%   mean_lost={lost:.2f}")

    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 5))
        bins = np.arange(0, counts.max() + 2) - 0.5
        ax.hist(counts, bins=bins, color="#3b7dd8", edgecolor="white", linewidth=0.3)
        ax.set_xlabel("overlapping tracks per 30s window (before truncation)")
        ax.set_ylabel("episodes")
        ax.set_title(f"Trajectory tokens per episode before truncation  (n={len(counts)})")
        ax.grid(True, axis='y', linestyle=":", alpha=0.4)
        ax.set_axisbelow(True)
        os.makedirs(OUT_DIR, exist_ok=True)
        path = os.path.join(OUT_DIR, "trajectory_token_count.png")
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        print(f"\nWrote plot: {path}")


if __name__ == "__main__":
    main()
