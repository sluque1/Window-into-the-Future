"""How many FUSED OBBs does a real observation window contain?

Sizes the token cap for the OBB-token env modality (one token per fused 3D
OBB, replacing the 4096 voxel tokens). Uses the exact training pipeline:
``loader.HDEpicInteractionDataset`` builds the real episodes (same anchors,
obs windows, ``t0`` and time anchor), then ``BoxerEnvBuilder._window_fused_boxes``
is re-run per unique window to count the fused boxes -- i.e. exactly how many
OBB tokens the modality would emit before any cap.

Unlike the other analysis/*.py scripts this one DOES import loader.py (and so
torch), because the fusion parameters and the per-video time anchor are what
we are measuring; reimplementing them would risk measuring something else.

Usage
-----
  python analysis/obb_token_count.py --participants P07
  python analysis/obb_token_count.py --participants P01,P02,P03,P04,P05,P06
  python analysis/obb_token_count.py --participants P07 --time_anchor gaze
"""

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from loader import HDEpicInteractionDataset  # noqa: E402

CAPS = (64, 96, 128, 192, 256, 320, 384, 448, 512, 640)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--participants', type=str, default='P07',
                    help='Comma-separated participant ids (or full video ids).')
    ap.add_argument('--observation_time', type=int, default=30)
    ap.add_argument('--anticipation_time', type=float, default=5)
    ap.add_argument('--future_time', type=float, default=180)
    ap.add_argument('--time_anchor', type=str, default='trajectory',
                    choices=('trajectory', 'gaze'))
    ap.add_argument('--out', type=str, default=None,
                    help='Optional .npy path to dump the raw per-window counts.')
    args = ap.parse_args()

    ds = HDEpicInteractionDataset(
        split='probe', participants=args.participants.split(','),
        observation_time=args.observation_time,
        anticipation_time=args.anticipation_time,
        future_time=args.future_time,
        time_anchor=args.time_anchor,
        verbose=True)

    seen = set()
    counts = []
    n_classes = []
    per_video = {}
    for d in ds.dataset:
        key = (d['video_id'], int(d['obs_start']), int(d['obs_end']))
        if key in seen:
            continue
        seen.add(key)
        fused, _ = ds.boxer._window_fused_boxes(
            d['video_id'], key[1], key[2], ds._t0[d['video_id']])
        n = 0 if fused is None else len(fused)
        counts.append(n)
        per_video.setdefault(d['video_id'], []).append(n)
        if fused:
            n_classes.append(len({b['sem_id'] for b in fused}))

    c = np.asarray(counts)
    if c.size == 0:
        print("No windows found.")
        return

    print(f"\n===== FUSED OBBs PER {args.observation_time}s WINDOW "
          f"({args.participants}, anchor={args.time_anchor}) =====")
    print(f"windows          : {len(c)}")
    print(f"mean / median    : {c.mean():.1f} / {np.median(c):.0f}")
    print(f"min / max        : {c.min()} / {c.max()}")
    for p in (50, 75, 90, 95, 99, 99.9, 100):
        print(f"  p{p:<5}         : {np.percentile(c, p):.0f}")
    print(f"empty windows    : {(c == 0).sum()} ({(c == 0).mean() * 100:.1f}%)")

    print("\ncap  | windows fully covered | truncated | mean boxes dropped/window")
    for cap in CAPS:
        print(f"{cap:<4} | {(c <= cap).mean() * 100:19.1f}% | "
              f"{(c > cap).sum():9d} | {np.maximum(c - cap, 0).mean():.2f}")

    if n_classes:
        nc = np.asarray(n_classes)
        print(f"\ndistinct classes/window: mean {nc.mean():.1f} "
              f"median {np.median(nc):.0f} max {nc.max()}")

    print("\n--- per video (n windows / mean / max) ---")
    for vid in sorted(per_video):
        a = np.asarray(per_video[vid])
        print(f"  {vid}: n={len(a):5d} mean={a.mean():7.1f} max={a.max():5d}")

    if args.out:
        np.save(args.out, c)
        print(f"\nraw counts -> {args.out}")


if __name__ == '__main__':
    main()
