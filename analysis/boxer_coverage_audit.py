"""Which Boxer OBB CSVs are missing or TRUNCATED, i.e. need rerunning?

A Boxer CSV is expected to hold detections spanning the whole recording. Some
do not: a handful of videos have no ``boxer_3dbbs.csv`` at all, and others have
one that stops early -- e.g. P05-20240425-173353's covers 90 s of a 626 s video.
Both cases look identical downstream (an all-empty env grid / an all-padding OBB
token segment), and neither is currently reported, so windows past the end of a
truncated CSV silently train on "the kitchen is empty" rather than "no data".

Coverage is measured WITHOUT the per-video time anchor: the CSV's own
(max - min) time span is compared against the video duration from
``Videos/<pid>/durations.txt``. That avoids needing SLAM/t0 and is the right
question anyway -- a CSV that spans 90 s cannot cover a 626 s video no matter
how it is aligned.

Usage
-----
  python analysis/boxer_coverage_audit.py
  python analysis/boxer_coverage_audit.py --min-coverage 0.95
  python analysis/boxer_coverage_audit.py --rerun-list /tmp/boxer_rerun.txt
"""

import argparse
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from boxer_env import BOXER_OBBS_ROOT  # noqa: E402
from loader import FEATURES_BASE, _load_annotations, _load_video_durations  # noqa: E402


def csv_span_seconds(path):
    """(min, max, span) of the CSV's time_ns column, in seconds. None if unusable."""
    try:
        df = pd.read_csv(path, usecols=['time_ns'])
    except Exception:
        return None
    if df.empty:
        return None
    lo = df['time_ns'].min() / 1e9
    hi = df['time_ns'].max() / 1e9
    return lo, hi, hi - lo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--min-coverage', type=float, default=0.95,
                    help='CSV span / video duration below this counts as TRUNCATED.')
    ap.add_argument('--rerun-list', type=str, default=None,
                    help='Write the video ids needing a rerun, one per line.')
    args = ap.parse_args()

    assoc, _ = _load_annotations()
    rows = []
    for video_id in sorted(assoc):
        pid = video_id.split('-')[0]
        has_feat = os.path.exists(
            os.path.join(FEATURES_BASE, pid, f"{video_id}_features.pt"))
        duration = _load_video_durations(pid).get(video_id)
        path = os.path.join(BOXER_OBBS_ROOT, video_id, 'boxer_3dbbs.csv')

        if not os.path.exists(path):
            rows.append((video_id, has_feat, duration, None, None, 'MISSING'))
            continue
        span = csv_span_seconds(path)
        if span is None:
            rows.append((video_id, has_feat, duration, None, None, 'EMPTY'))
            continue
        _, _, sec = span
        if duration is None or duration <= 0:
            rows.append((video_id, has_feat, duration, sec, None, 'NO_DURATION'))
            continue
        cov = sec / duration
        status = 'OK' if cov >= args.min_coverage else 'TRUNCATED'
        rows.append((video_id, has_feat, duration, sec, cov, status))

    bad = [r for r in rows if r[5] not in ('OK',)]
    bad.sort(key=lambda r: (r[5] != 'MISSING', r[4] if r[4] is not None else -1))

    print(f"{len(rows)} annotated videos | "
          f"{sum(r[5] == 'OK' for r in rows)} OK | "
          f"{sum(r[5] == 'MISSING' for r in rows)} MISSING | "
          f"{sum(r[5] == 'TRUNCATED' for r in rows)} TRUNCATED "
          f"(< {args.min_coverage:.0%} of the video) | "
          f"{sum(r[5] in ('EMPTY', 'NO_DURATION') for r in rows)} other")

    print(f"\n{'video_id':<24} {'feats':>5} {'video_s':>9} {'csv_s':>9} "
          f"{'coverage':>9}  status")
    for vid, feat, dur, sec, cov, status in bad:
        print(f"{vid:<24} {'yes' if feat else 'NO':>5} "
              f"{dur if dur else float('nan'):9.1f} "
              f"{sec if sec is not None else float('nan'):9.1f} "
              f"{cov * 100 if cov is not None else float('nan'):8.1f}%  {status}")

    rerun = [r[0] for r in bad if r[5] in ('MISSING', 'TRUNCATED', 'EMPTY')]
    print(f"\n=== {len(rerun)} videos need a Boxer rerun ===")
    for v in rerun:
        print(v)

    if args.rerun_list:
        with open(args.rerun_list, 'w') as fh:
            fh.write('\n'.join(rerun) + '\n')
        print(f"\nwritten -> {args.rerun_list}")


if __name__ == '__main__':
    main()
