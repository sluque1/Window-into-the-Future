"""Precompute IoU-fused Boxer OBBs, one pickle per video, beside its source CSV.

Why this exists
---------------
``loader.HDEpicInteractionDataset.__init__`` fuses every unique observation
window before the first training step, single-threaded. Grid fusion costs
~0.04 s/window, so that is minutes; IoU fusion measured **6.36 s/window** (mean
over 24 windows spanning all nine participants; median 5.82, max 15.2). A
production split holds ~28,800 fusable windows, i.e. **~51 h** inside
``__init__`` -- the job's 6 h wall clock expires before training starts. So the
fusion happens here, once, in parallel, and training just reads the result.

Output
------
``boxer_obbs/<video_id>/fused_obbs_iou.pkl``, a sibling of the
``boxer_3dbbs.csv`` it was fused from, so the derived file travels with its
source. Per video rather than per split because every LOOCV fold then reuses it:
a video is train in one fold and test in another, and a per-split cache would be
rebuilt nine times.

Each file holds ``{'meta': {...}, 'windows': {(video_id, obs_start, obs_end):
[box, ...]}}``. ``meta`` records the fusion settings **and ``t0_s``**, and
``BoxerEnvBuilder._fused_for_video`` checks them on load. That check is the whole
point: window keys are ``int``-truncated video seconds, so a cache built under a
different ``time_anchor`` has the *same keys* holding *different boxes* -- wrong
silently rather than missing.

Windows
-------
Exactly the keys ``loader.py``'s two precompute loops use::

    key = (video_id, int(obs_start), int(obs_end))     # loader.py:539, 551

and built with the loader's own ``build_video_events`` / ``build_episodes``, so
this cannot drift from what training asks for. Note the ``int()`` truncation:
~1.18 episodes share a window, so 30,371 episodes need only 25,680 fusions.

``future_time`` only decides which anchors survive the empty-GT drop, so a cache
built at the largest one is a superset of the shorter ones -- build at 180 and
runs at 60/120 hit it too.

Usage
-----
    python precompute_fused_obbs.py --dry-run              # cost, writes nothing
    python precompute_fused_obbs.py --videos P08-20240614-085000
    python precompute_fused_obbs.py --shard 3 --num-shards 32
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys
import time

import loader as L
from boxer_env import BoxerEnvBuilder
from gaze import gaze_t0, has_gaze_data

# The configuration these caches are built for. Fixed by design: every
# production run uses obs=30 s, anticip=5 s, TIME_ANCHOR=gaze. A run that
# differs fails BoxerEnvBuilder's meta check rather than reading wrong boxes.
OBS_TIME = 30
ANTICIP_TIME = 5.0
FUTURE_TIME = 180.0        # superset, covers shorter future_time runs
TIME_ANCHOR = 'gaze'

MEASURED_S_PER_WINDOW = 6.36     # see the module docstring

CORE_KEYS = ('center', 'quat', 'half_extents', 'sem_id', 'prob')


def video_windows(video_id, assoc, mask, obs_time, anticip_time, future_time):
    """The int-truncated windows training will ask for, deduplicated."""
    events, anchors = L.build_video_events(assoc[video_id], mask[video_id])
    if not events:
        return []
    episodes = L.build_episodes(events, anchors, obs_time, anticip_time, future_time)
    return sorted({(int(e['obs_start']), int(e['obs_end'])) for e in episodes})


def write_video_cache(builder, video_id, windows, t0_s, slim=False):
    """Fuse every window of one video and write it beside that video's CSV.

    Written to a temp file then renamed, so a task killed mid-write leaves the
    previous cache intact rather than a truncated pickle.
    """
    boxer_df = builder.load_boxer_csv(video_id)
    if boxer_df is None:
        return 0, 0, None

    times = boxer_df['time_ns'].to_numpy()
    out, n_boxes = {}, 0
    for obs_start_s, obs_end_s in windows:
        lo = (obs_start_s + t0_s) * 1e9
        hi = (obs_end_s + t0_s) * 1e9
        window_df = boxer_df.loc[(times >= lo) & (times <= hi)]
        fused = builder._fuse(window_df) if len(window_df) else []
        if slim:
            fused = [{k: b[k] for k in CORE_KEYS} for b in fused]
        key = (video_id, round(float(obs_start_s), 3), round(float(obs_end_s), 3))
        out[key] = fused
        n_boxes += len(fused)

    path = builder.fused_path(video_id)
    blob = {'meta': dict(builder.fusion_settings(), t0_s=float(t0_s),
                         slim=bool(slim), video_id=video_id,
                         obs_time=OBS_TIME, anticip_time=ANTICIP_TIME,
                         time_anchor=TIME_ANCHOR),
            'windows': out}
    tmp = f"{path}.tmp{os.getpid()}"
    with open(tmp, 'wb') as fh:
        pickle.dump(blob, fh, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, path)
    return len(out), n_boxes, path


def _fuse_one_video(task):
    """Worker body: fuse one whole video. Videos are independent -- separate
    CSVs, separate output files -- so they parallelise with no shared state.

    Runs in a forked child, which inherits the parent's already-loaded
    annotations, so each worker re-reads only its own video's CSV.
    """
    (video_id, obs_time, anticip_time, future_time,
     gaze_max_hz, gaze_time_offset_s, slim, overwrite) = task
    pid = video_id.split('-')[0]
    builder = BoxerEnvBuilder()
    if builder.has_fused_cache(video_id) and not overwrite:
        return video_id, 'cached', 0, 0, 0.0
    assoc, mask = L._load_annotations()          # module-level cache, inherited
    windows = video_windows(video_id, assoc, mask, obs_time, anticip_time,
                            future_time)
    if not windows:
        return video_id, 'no-episodes', 0, 0, 0.0
    t0 = gaze_t0(pid, video_id, gaze_max_hz, gaze_time_offset_s)
    if t0 is None:
        return video_id, 'no-anchor', 0, 0, 0.0
    t = time.time()
    nw, nb, path = write_video_cache(builder, video_id, windows, float(t0),
                                     slim=slim)
    return video_id, 'written', nw, nb, time.time() - t


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--videos', nargs='*', default=None,
                    help="explicit video ids (default: every video with features "
                         "and a Boxer CSV)")
    ap.add_argument('--participants', default=None,
                    help="comma-separated pids to restrict to, e.g. P01,P02")
    ap.add_argument('--shard', type=int, default=0,
                    help="this task's index, 0-based (SLURM_ARRAY_TASK_ID)")
    ap.add_argument('--num-shards', type=int, default=1)
    ap.add_argument('--obs-time', type=int, default=OBS_TIME)
    ap.add_argument('--anticip-time', type=float, default=ANTICIP_TIME)
    ap.add_argument('--future-time', type=float, default=FUTURE_TIME)
    ap.add_argument('--gaze-max-hz', type=float, default=0.0,
                    help="must match the run's GAZE_MAX_HZ: thinning can move the "
                         "first retained sample, which is the time anchor")
    ap.add_argument('--gaze-time-offset-s', type=float,
                    default=L.GAZE_DEFAULT_TIME_OFFSET_S)
    ap.add_argument('--slim', action='store_true',
                    help="store only center/quat/half_extents/sem_id/prob, dropping "
                         "the name/support/yaw/members audit fields (~40%% smaller)")
    ap.add_argument('--workers', type=int, default=1,
                    help="fuse this many videos in parallel. One participant is "
                         "2,255-4,641 windows, i.e. 4.0-8.2 h serially against a "
                         "6:10 partition cap, so a per-participant job needs this. "
                         "Videos are independent, so it scales ~linearly.")
    ap.add_argument('--overwrite', action='store_true',
                    help="rebuild caches that already exist")
    ap.add_argument('--dry-run', action='store_true',
                    help="report the work and its estimated cost, write nothing")
    args = ap.parse_args()

    builder = BoxerEnvBuilder()      # the class defaults ARE the iou settings
    if builder.fusion != 'iou':
        sys.exit(f"expected BoxerEnvBuilder to default to iou fusion, got "
                 f"{builder.fusion!r}")
    print(f"fusion settings: {builder.fusion_settings()}", flush=True)

    assoc, mask = L._load_annotations()
    if args.videos:
        video_ids = list(args.videos)
    else:
        video_ids = sorted(assoc)
        if args.participants:
            want = set(args.participants.split(','))
            video_ids = [v for v in video_ids if v.split('-')[0] in want]

    # Only videos training will actually read: features present (loader.py skips
    # the rest) and a Boxer CSV to fuse.
    todo, no_gaze = [], []
    for vid in video_ids:
        pid = vid.split('-')[0]
        if not os.path.exists(os.path.join(L.FEATURES_BASE, pid, f"{vid}_features.pt")):
            continue
        if not builder.has_boxer_data(vid):
            continue
        # No gaze CSV means no TIME_ANCHOR=gaze anchor. The three such videos are
        # also the three without SLAM, which loader.py drops for having no voxel
        # grid -- so they are never trained on and need no cache.
        if not has_gaze_data(pid, vid):
            no_gaze.append(vid)
            continue
        todo.append(vid)

    if no_gaze:
        print(f"skipping {len(no_gaze)} video(s) with no gaze CSV (no gaze anchor; "
              f"loader.py drops these anyway): {', '.join(no_gaze)}", flush=True)

    todo = todo[args.shard::args.num_shards]     # strided: interleaves long videos
    print(f"shard {args.shard}/{args.num_shards}: {len(todo)} videos", flush=True)

    if args.dry_run:
        total_w = 0
        for vid in todo:
            w = video_windows(vid, assoc, mask, args.obs_time, args.anticip_time,
                              args.future_time)
            total_w += len(w)
            print(f"  {vid}: {len(w)} windows"
                  f"{'  [cached]' if builder.has_fused_cache(vid) else ''}")
        print(f"\n{total_w} windows -> ~{total_w * MEASURED_S_PER_WINDOW / 3600:.2f} h "
              f"for this shard at the measured {MEASURED_S_PER_WINDOW} s/window")
        return

    tasks = [(vid, args.obs_time, args.anticip_time, args.future_time,
              args.gaze_max_hz, args.gaze_time_offset_s, args.slim, args.overwrite)
             for vid in todo]

    # Longest videos first: with uneven work per video, starting the big ones
    # early keeps the last worker from finishing long after the others.
    if args.workers > 1:
        n_win_by_vid = {vid: len(video_windows(vid, assoc, mask, args.obs_time,
                                               args.anticip_time, args.future_time))
                        for vid in todo}
        tasks.sort(key=lambda t: -n_win_by_vid[t[0]])
        print(f"using {args.workers} workers over {len(tasks)} videos "
              f"({sum(n_win_by_vid.values())} windows, longest first)", flush=True)

    started = time.time()
    n_done = n_win = n_box = n_skip = 0

    def _report(i, res):
        nonlocal n_done, n_win, n_box, n_skip
        vid, status, nw, nb, dt = res
        tag = f"[{i}/{len(tasks)}] {vid}"
        if status == 'written':
            n_done += 1; n_win += nw; n_box += nb
            size = os.path.getsize(BoxerEnvBuilder().fused_path(vid)) / 1e6
            print(f"{tag}: {nw} windows, {nb} boxes, {dt:.0f}s "
                  f"({dt/max(nw,1):.2f}s/window) -> {size:.1f} MB", flush=True)
        else:
            if status == 'cached':
                n_skip += 1
            print(f"{tag}: {status}, skipping", flush=True)

    if args.workers > 1:
        import multiprocessing as mp
        # fork so workers inherit the annotations already loaded above rather
        # than each re-reading assoc_info/mask_info.
        with mp.get_context('fork').Pool(args.workers) as pool:
            for i, res in enumerate(pool.imap_unordered(_fuse_one_video, tasks), 1):
                _report(i, res)
    else:
        for i, task in enumerate(tasks, 1):
            _report(i, _fuse_one_video(task))

    el = time.time() - started
    label = (f"{args.participants}" if args.participants
             else f"shard {args.shard}")
    print(f"\n{label}: {n_done} videos written, {n_skip} already cached, "
          f"{n_win} windows, {n_box} boxes in {el/60:.1f} min wall "
          f"({el/max(n_win,1):.2f}s/window wall, {args.workers} workers)")


if __name__ == '__main__':
    main()
