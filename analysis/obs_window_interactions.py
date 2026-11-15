"""
Per-video export of how many GT interactions fall in each episode's OBSERVATION
window, using the track-level start/end definition.

Interaction definition (from the scene-and-object-movements annotations)
------------------------------------------------------------------------
assoc_info.json groups object movements into TRACKS. Every track carries a
``time_segment`` [t_start, t_end] (see mask/assoc files):

    "time_segment": [316.933, 327.333]   ->  start interaction @ 316.933
                                             end   interaction @ 327.333

So each track contributes TWO interactions in time: the object is picked
up/first-seen at ``t_start`` and put down/last-seen at ``t_end``.

Valid-3d restriction (matches the prediction-GT pool)
-----------------------------------------------------
Only interactions with a valid 3d_location are counted, consistent with
loader.build_video_events (which drops masks without a 3d_location). The
3d_location lives on the track's MASKS, not on time_segment, so per track:
  * START counts iff the earliest mask has a valid 3d_location (time = t_start).
  * END   counts iff the track has a distinct latest mask (>=2 masks) whose
    3d_location is valid                                     (time = t_end).
A 1-mask track (single observation, sitting at t_start) therefore contributes a
start interaction only; tracks with no valid-3d mask contribute nothing.

Episodes (unchanged from training, see loader.py)
-------------------------------------------------
Episodes and their observation windows are produced by the loader's own
``build_video_events`` / ``build_episodes`` (masks with a valid 3d_location,
anchor t, obs window = [t - anticip - obs, t - anticip], kept iff obs_end >= 1s
and the prediction window [t, t+future] is non-empty). We do NOT change episode
generation; we only recount what is inside each observation window.

Per episode we report, over the observation window [obs_start, obs_end]:
  * n_start_in_obs   : tracks whose START interaction (t_start) is in the window
  * n_end_in_obs     : tracks whose END   interaction (t_end)   is in the window
  * n_total_interactions = n_start_in_obs + n_end_in_obs
                       (every start and every end counted as one interaction)
  * n_both_same_object : tracks whose START and END are BOTH in the window
                       (a complete object movement fully observed)

Only valid-3d start/end interactions are counted (see restriction above).

Output
------
One CSV per video under ./out/per_video_obs_interactions/<video_id>.csv, one row
per episode. A brief console summary is printed at the end.

Usage
-----
  python analysis/obs_window_interactions.py
  python analysis/obs_window_interactions.py --observation_time 30 --anticipation_time 5 --future_time 180
  python analysis/obs_window_interactions.py --splits train,val,test
"""

import argparse
import csv
import os
import sys

import numpy as np

# Import the real episode code so we never drift from what training uses.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from loader import (  # noqa: E402
    FEATURES_BASE,
    PARTICIPANT_TO_SPLIT,
    _has_valid_location,
    _load_annotations,
    build_episodes,
    build_video_events,
)

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "out", "per_video_obs_interactions")

# CSV columns, in order.
FIELDS = [
    "episode_idx", "participant_id", "video_id",
    "anchor_t", "obs_start", "obs_end",
    "n_pred_gt",
    "n_total_interactions", "n_start_in_obs", "n_end_in_obs",
    "n_both_same_object",
]


def track_segments(v_assoc, v_mask):
    """Return (starts, ends, start_valid, end_valid) arrays for one video's tracks.

    starts[i]/ends[i] are the track's time_segment endpoints. start_valid[i] /
    end_valid[i] mark whether that endpoint has a valid-3d mask (see module
    docstring): start = earliest mask valid; end = a distinct latest mask valid.
    """
    starts, ends, sv, ev = [], [], [], []
    for info in v_assoc.values():
        for tr in info["tracks"]:
            seg = tr.get("time_segment")
            if not seg or len(seg) != 2:
                continue
            ms = sorted((m for m in tr["masks"] if m in v_mask),
                        key=lambda m: v_mask[m]["frame_number"])
            start_ok = bool(ms) and _has_valid_location(v_mask[ms[0]].get("3d_location"))
            end_ok = len(ms) >= 2 and _has_valid_location(v_mask[ms[-1]].get("3d_location"))
            starts.append(float(seg[0]))
            ends.append(float(seg[1]))
            sv.append(start_ok)
            ev.append(end_ok)
    return (np.asarray(starts, dtype=np.float64), np.asarray(ends, dtype=np.float64),
            np.asarray(sv, dtype=bool), np.asarray(ev, dtype=bool))


def episode_rows(video_id, pid, episodes, starts, ends, start_valid, end_valid, anticip):
    """Build one output row per episode with the track-based obs-window counts."""
    rows = []
    for i, ep in enumerate(episodes):
        lo, hi = ep["obs_start"], ep["obs_end"]
        s_in = (starts >= lo) & (starts <= hi) & start_valid
        e_in = (ends >= lo) & (ends <= hi) & end_valid
        n_start = int(s_in.sum())
        n_end = int(e_in.sum())
        rows.append({
            "episode_idx": i,
            "participant_id": pid,
            "video_id": video_id,
            # obs_end == anchor - anticip for every kept episode, so recover anchor
            "anchor_t": round(ep["obs_end"] + anticip, 3),
            "obs_start": round(ep["obs_start"], 3),
            "obs_end": round(ep["obs_end"], 3),
            "n_pred_gt": len(ep["gt_locations"]),
            "n_total_interactions": n_start + n_end,
            "n_start_in_obs": n_start,
            "n_end_in_obs": n_end,
            "n_both_same_object": int((s_in & e_in).sum()),
        })
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--observation_time", type=int, default=30)
    ap.add_argument("--anticipation_time", type=float, default=5)
    ap.add_argument("--future_time", type=float, default=180)
    ap.add_argument("--splits", type=str, default="train,val,test")
    ap.add_argument("--out-dir", type=str, default=OUT_DIR)
    args = ap.parse_args()

    wanted_splits = {s.strip() for s in args.splits.split(",") if s.strip()}
    wanted_pids = {p for p, s in PARTICIPANT_TO_SPLIT.items() if s in wanted_splits}

    print("Per-video observation-window interaction export")
    print(f"  observation_time={args.observation_time}s  "
          f"anticipation_time={args.anticipation_time}s  "
          f"future_time={args.future_time}s")
    print(f"  observation window = [t - {args.anticipation_time + args.observation_time:g}, "
          f"t - {args.anticipation_time:g}]   (t = anchor)")
    print(f"  splits={sorted(wanted_splits)}  out_dir={args.out_dir}")

    assoc, mask = _load_annotations()
    os.makedirs(args.out_dir, exist_ok=True)

    video_ids = sorted(v for v in assoc if v.split("-")[0] in wanted_pids)

    n_videos = n_episodes = n_no_feature = 0
    all_total, all_both = [], []
    for video_id in video_ids:
        pid = video_id.split("-")[0]
        if not os.path.exists(os.path.join(FEATURES_BASE, pid, f"{video_id}_features.pt")):
            n_no_feature += 1
            continue

        events, anchors = build_video_events(assoc[video_id], mask[video_id])
        if not events:
            continue
        episodes = build_episodes(events, anchors, args.observation_time,
                                  args.anticipation_time, args.future_time)
        if not episodes:
            continue

        starts, ends, sv, ev = track_segments(assoc[video_id], mask[video_id])
        rows = episode_rows(video_id, pid, episodes, starts, ends, sv, ev,
                            args.anticipation_time)

        with open(os.path.join(args.out_dir, f"{video_id}.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            w.writeheader()
            w.writerows(rows)

        n_videos += 1
        n_episodes += len(rows)
        all_total.extend(r["n_total_interactions"] for r in rows)
        all_both.extend(r["n_both_same_object"] for r in rows)

    print(f"\nWrote {n_videos} per-video CSVs ({n_episodes} episodes total) "
          f"to {args.out_dir}")
    print(f"  ({n_no_feature} videos skipped: no feature file)")
    if all_total:
        t = np.asarray(all_total)
        b = np.asarray(all_both)
        print("\nAcross all episodes:")
        print(f"  total interactions in obs window : "
              f"mean {t.mean():.2f}  median {np.median(t):.0f}  "
              f"min {t.min()}  max {t.max()}")
        print(f"  start+end of same object         : "
              f"mean {b.mean():.2f}  median {np.median(b):.0f}  "
              f"min {b.min()}  max {b.max()}")


if __name__ == "__main__":
    main()