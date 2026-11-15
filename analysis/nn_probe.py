"""Is the spatial signal in the EgoVLPv2 features, or does the model discard it?

input_swap_ablation.py showed that replacing an episode's 30 seconds of video
with another episode's costs 1.4% of PR-AUC on average across the nine LOOCV
folds (range -5.1% to +5.5%), against 18.5% for the OBB tokens. That is a
statement about the TRAINED MODEL: it says those weights do not use the video.
It cannot distinguish two very different situations --

    the features do not encode where the next interaction will be, or
    they do and this architecture fails to extract it

-- and the two call for opposite responses. This script settles it with no
learned parameters at all. For each test episode it finds the most similar OTHER
episode by raw feature cosine similarity and hands over THAT episode's label map
as the prediction. If raw similarity predicts interaction location, the signal is
in the features and the model is the problem.

Three things make the result interpretable rather than just another number:

  * Neighbours must come from a DIFFERENT RECORDING of the same participant.
    Episodes are anchored a few seconds apart, so an adjacent episode of the same
    video shares nearly all of its labels for purely temporal reasons -- any
    probe allowing that scores brilliantly and measures nothing. The
    same-video-allowed arm is reported too, precisely to show how large that
    artefact is.
  * NN-RANDOM is the control: a random episode from a different recording. It
    carries the same "uses this kitchen's labels" advantage as every NN arm, so
    (NN-video - NN-random) isolates what visual similarity is worth. Comparing
    NN-video against the train-prior instead would confound the two.
  * NN-obb is a POSITIVE control. The swap ablation puts obb at 18.5%, so a probe
    that cannot see signal there is broken, and its silence on video would mean
    nothing. Read NN-video only after checking NN-obb clears NN-random.

Everything is per participant. Per-fold base PR-AUC spans .039 to .311 and the
grids are fitted per kitchen (CLAUDE.md), so pooled means are dominated by the
densest kitchens; a mean row is printed but the per-fold rows are the result.

Usage:
    python analysis/nn_probe.py --participants P01
    python analysis/nn_probe.py --participants P01,P02 --n_episodes 800
"""
import argparse
import glob
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from loader import GRID, HDEpicInteractionDataset, NUM_VOXELS  # noqa: E402

SNAP = os.environ.get("SNAPSHOTS_ROOT", "/data/video_datasets/HD-EPIC/snapshots")
# The te10 arm, whose swap-ablation numbers this probe exists to explain.
RUN_GLOB = "2026-08-12_08-54-*_loocv_{pid}_*_te10"


def ap_per_episode(scores, y):
    """(aps, keep) -- average precision for each episode with >=1 positive and >=1
    negative, plus the boolean mask saying which episodes those were. The mask is
    needed to line the APs up with per-episode similarity when binning below.

    scores is [N, 4096] (one map per episode) or [4096] (one map shared by all).
    Ties are handled as sklearn does, which is essential here: a k=1 neighbour map
    is binary, so 4076 voxels tie at 0 and ~20 tie at 1."""
    y = np.asarray(y, np.float64)
    keep = (y.sum(1) > 0) & (y.sum(1) < y.shape[1])
    y = y[keep]
    n = np.arange(1, y.shape[1] + 1, dtype=np.float64)
    s = scores if scores.ndim == 1 else np.asarray(scores, np.float64)[keep]
    if s.ndim == 1:
        order = np.argsort(-s, kind='stable')
        ss, ys = s[order], y[:, order]
        tp = np.cumsum(ys, axis=1)
        ends = np.r_[np.nonzero(np.diff(ss))[0], len(ss) - 1]
        prec = tp[:, ends] / n[ends]
        rec = tp[:, ends] / y.sum(1, keepdims=True)
        return (prec * np.diff(np.c_[np.zeros(len(y)), rec], axis=1)).sum(1), keep
    out = np.empty(len(y))
    for i in range(len(y)):
        order = np.argsort(-s[i], kind='stable')
        si, yi = s[i][order], y[i][order]
        tp = np.cumsum(yi)
        ends = np.r_[np.nonzero(np.diff(si))[0], len(si) - 1]
        out[i] = ((tp[ends] / n[ends]) * np.diff(np.r_[0.0, tp[ends] / yi.sum()])).sum()
    return out, keep


def per_episode_ap(scores, y):
    """Mean of the above -- the same quantity train.py logs as pr_auc."""
    return float(ap_per_episode(scores, y)[0].mean())


def loader_settings(pid):
    """anticipation_time / future_time / time_anchor are not in config.json, so
    read them off the run's own SLURM log rather than defaulting them -- a
    different future_time silently changes every label."""
    runs = sorted(glob.glob(os.path.join(SNAP, RUN_GLOB.format(pid=pid))))
    if not runs:
        raise SystemExit(f"no te10 run found for {pid}")
    log = sorted(glob.glob(os.path.join(runs[0], 'out', '*.out')))[0]
    head = open(log).read(20000)
    want = {'OBS_TIME': int, 'FUTURE_TIME': float, 'ANTICIP_TIME': float,
            'TIME_ANCHOR': str, 'ENV_ORPHAN_FALLBACK': str,
            'TRAJECTORY_EXCLUDE_UNFINISHED': str, 'ENV_FALLBACK_AVERAGING': str,
            'GAZE_MAX_HZ': float, 'GAZE_TIME_OFFSET_S': float,
            'VOXEL_CENTERS_DIR': str, 'MAX_OBB_TOKENS': int}
    out = {}
    for key, cast in want.items():
        m = re.search(rf'\b{key}=(\S*)', head)
        if m is None:
            raise SystemExit(f"{log} has no {key}=; refusing to guess it")
        out[key] = cast(m.group(1)) if m.group(1) else ('' if cast is str else cast(0))
    return out


def episode_features(dataset, sel):
    """Per-episode representations, all L2-normalised so cosine == dot product.

    video    -- mean of the VALID EgoVLPv2 seconds (padding excluded; a window
                truncated at the recording start would otherwise be pulled
                toward zero and match other truncated windows on that alone).
    obb      -- 4096-d occupancy of the fused boxes' voxels: "what is where".
    location -- 4096-d occupancy of the actor's per-second voxels.
    """
    # __getitem__ returns a mix of torch tensors and numpy arrays depending on the
    # key, so normalise rather than assuming either.
    def arr(x):
        return x.numpy() if hasattr(x, 'numpy') else np.asarray(x)

    n = len(sel)
    video = np.zeros((n, 4096), np.float32)
    obb = np.zeros((n, NUM_VOXELS), np.float32)
    loc = np.zeros((n, NUM_VOXELS), np.float32)
    labels = np.zeros((n, NUM_VOXELS), np.uint8)
    for row, idx in enumerate(sel):
        if row % 500 == 0:
            print(f"    episode {row}/{n}", flush=True)
        d = dataset[int(idx)]
        vm = arr(d['visual_mask']).astype(bool)
        vf = arr(d['visual_features'])
        video[row] = vf[vm].mean(0) if vm.any() else vf.mean(0)
        ov, om = arr(d['obb_voxels']), arr(d['obb_mask']).astype(bool)
        real = om & (ov > 0).all(-1)
        if real.any():
            ijk = ov[real] - 1
            obb[row, ijk[:, 0] * GRID * GRID + ijk[:, 1] * GRID + ijk[:, 2]] = 1.0
        pl = arr(d['person_locations'])
        pm = arr(d['person_location_mask']).astype(bool) & (pl > 0).all(-1)
        if pm.any():
            ijk = pl[pm] - 1
            loc[row, ijk[:, 0] * GRID * GRID + ijk[:, 1] * GRID + ijk[:, 2]] = 1.0
        labels[row] = arr(d['labels']['interaction'])

    def unit(x):
        return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-8)

    return {'video': unit(video), 'obb': unit(obb), 'location': unit(loc)}, labels


def neighbour_maps(feats, labels, video_of, k, allow_same_video, rng, block=512):
    """Mean label map of each episode's k nearest neighbours by cosine similarity.

    Self is always excluded; same-video episodes are excluded unless
    allow_same_video, which exists only to show how big that artefact is.

    Also returns HOW SIMILAR those neighbours actually were, which the lift over
    NN-random cannot tell you. "Nearest" is a rank: if every eligible pair sits at
    roughly the same cosine similarity, the top-k is close to arbitrary and
    NN-<arm> would collapse onto NN-random no matter what the features encode.
    That is a failure of the SEARCH, and reads identically to "the features carry
    no signal" unless the similarities themselves are reported.

    Per episode:
        top1 / topk  -- similarity of the best / mean of the k best neighbours
        median       -- median similarity over all eligible neighbours, i.e. the
                        background level for this representation
        z            -- (top1 - mean_eligible) / std_eligible. Raw cosine is not
                        comparable across arms (a dense EgoVLPv2 embedding and a
                        sparse binary occupancy grid have very different baseline
                        similarity), so z is what to compare between them.
    """
    n = len(labels)
    same = video_of[:, None] == video_of[None, :]
    out = np.zeros((n, labels.shape[1]), np.float32)
    stats = {key: np.full(n, np.nan, np.float64)
             for key in ('top1', 'topk', 'median', 'z')}
    lab = labels.astype(np.float32)
    for start in range(0, n, block):
        stop = min(start + block, n)
        sim = feats[start:stop] @ feats.T                       # [b, n]
        for r in range(stop - start):
            i = start + r
            banned = np.zeros(n, bool)
            banned[i] = True
            if not allow_same_video:
                banned |= same[i]
            eligible = sim[r][~banned]
            row = np.where(banned, -np.inf, sim[r])
            if eligible.size < k:
                out[i] = lab.mean(0)          # no eligible neighbour: fall back
                continue
            top = np.argpartition(-row, k - 1)[:k]
            out[i] = lab[top].mean(0)
            sd = eligible.std()
            stats['top1'][i] = eligible.max()
            stats['topk'][i] = sim[r][top].mean()
            stats['median'][i] = np.median(eligible)
            stats['z'][i] = ((eligible.max() - eligible.mean()) / sd) if sd > 0 else np.nan
    return out, stats


def conditional_maps(primary, secondary, labels, video_of, k, pool, rng,
                     rank_by_secondary, block=512):
    """Two-stage neighbour choice: narrow to the `pool` most PRIMARY-similar
    episodes, then pick k from that pool -- either the most SECONDARY-similar
    (rank_by_secondary=True) or at random.

    This is the redundancy test, and the random variant is the control that makes
    it mean anything. All three single-signal arms give near-identical lifts
    (video +44.3%, obb +43.9%, location +42.5% at k=1), which is what you would
    see if they were three views of ONE underlying variable -- roughly, which part
    of the kitchen the actor is in. Comparing NN-video|obb against plain NN-obb
    cannot separate that: the pool is already obb-similar, so it would score well
    however the k are drawn from it.

        (video|obb) - (random|obb)  =  what video adds GIVEN obb.

    Near zero on both orderings means the signals are duplicates and teaching the
    model to read video buys nothing it does not already get from the boxes.
    """
    n = len(labels)
    same = video_of[:, None] == video_of[None, :]
    out = np.zeros((n, labels.shape[1]), np.float32)
    lab = labels.astype(np.float32)
    for start in range(0, n, block):
        stop = min(start + block, n)
        sim_p = primary[start:stop] @ primary.T
        sim_s = secondary[start:stop] @ secondary.T
        for r in range(stop - start):
            i = start + r
            banned = np.zeros(n, bool)
            banned[i] = True
            banned |= same[i]
            eligible = np.nonzero(~banned)[0]
            if eligible.size < k:
                out[i] = lab.mean(0)
                continue
            m = min(pool, eligible.size)
            cand = eligible[np.argpartition(-sim_p[r][eligible], m - 1)[:m]]
            if rank_by_secondary:
                take = cand[np.argsort(-sim_s[r][cand])[:k]]
            else:
                take = rng.choice(cand, min(k, len(cand)), replace=False)
            out[i] = lab[take].mean(0)
    return out


def random_maps(labels, video_of, k, rng):
    """Control: k random episodes from a DIFFERENT recording. Same access to this
    kitchen's labels as every NN arm, no similarity matching."""
    n = len(labels)
    lab = labels.astype(np.float32)
    out = np.zeros((n, labels.shape[1]), np.float32)
    for i in range(n):
        eligible = np.nonzero(video_of != video_of[i])[0]
        if len(eligible) < k:
            out[i] = lab.mean(0)
            continue
        out[i] = lab[rng.choice(eligible, k, replace=False)].mean(0)
    return out


def run(pid, args):
    st = loader_settings(pid)
    print(f"\n=== {pid} ===", flush=True)
    dataset = HDEpicInteractionDataset(
        'test', participants=[pid],
        observation_time=st['OBS_TIME'],
        anticipation_time=st['ANTICIP_TIME'],
        future_time=st['FUTURE_TIME'],
        static_env=True,
        voxel_centers_dir=st['VOXEL_CENTERS_DIR'] or None,
        max_obb_tokens=st['MAX_OBB_TOKENS'],
        trajectory_simple=True,
        trajectory_exclude_unfinished=st['TRAJECTORY_EXCLUDE_UNFINISHED'] == 'true',
        env_orphan_fallback=st['ENV_ORPHAN_FALLBACK'] == 'true',
        env_fallback_averaging=st['ENV_FALLBACK_AVERAGING'],
        gaze_max_hz=st['GAZE_MAX_HZ'],
        gaze_time_offset_s=st['GAZE_TIME_OFFSET_S'],
        time_anchor=st['TIME_ANCHOR'],
        verbose=False)

    rng = np.random.default_rng(args.seed)
    n_all = len(dataset)
    sel = (np.sort(rng.choice(n_all, args.n_episodes, replace=False))
           if 0 < args.n_episodes < n_all else np.arange(n_all))
    video_of = np.array([dataset.dataset[int(i)]['video_id'] for i in sel])
    print(f"  {len(sel)} of {n_all} episodes, {len(set(video_of))} recordings", flush=True)

    feats, labels = episode_features(dataset, sel)
    chance = per_episode_ap(labels.mean(0) * 0 + 1.0, labels)   # constant map = floor
    oracle = per_episode_ap(labels.mean(0), labels)

    rows = {'chance': chance, 'oracle-map': oracle}
    sim_report, dose = {}, {}
    for k in args.k:
        rnd_aps, kept = ap_per_episode(random_maps(labels, video_of, k, rng), labels)
        rows[f'NN-random k={k}'] = float(rnd_aps.mean())
        for name in ('video', 'obb', 'location'):
            maps, st = neighbour_maps(feats[name], labels, video_of, k, False, rng)
            aps, _ = ap_per_episode(maps, labels)
            rows[f'NN-{name} k={k}'] = float(aps.mean())
            # Dose-response: does the prediction improve when the neighbour is
            # genuinely more similar? Binned on the PAIRED difference against the
            # random control, episode by episode, not on raw AP -- episodes differ
            # enormously in difficulty (an episode with 40 positive voxels scores
            # higher than one with 5 whatever you predict), so raw-AP quartiles
            # would mostly report which difficulty landed in which bin. The paired
            # difference cancels that: both arms predict the SAME episode.
            top1 = st['top1'][kept]
            good = np.isfinite(top1)
            if good.sum() > 40:
                delta = aps[good] - rnd_aps[good]
                q = np.quantile(top1[good], [0.25, 0.5, 0.75])
                bins = np.digitize(top1[good], q)
                dose[f'{name} k={k}'] = [float(delta[bins == b].mean())
                                         for b in range(4)]
            if k == args.k[0]:
                sim_report[name] = {s: float(np.nanmean(st[s]))
                                    for s in ('top1', 'topk', 'median', 'z')}
        maps, _ = neighbour_maps(feats['video'], labels, video_of, k, True, rng)
        rows[f'NN-video k={k} SAME-VIDEO-OK'] = per_episode_ap(maps, labels)

        # --- Redundancy: does either signal add anything GIVEN the other? -----
        # All three combined, and both conditional orderings each against their
        # own within-pool random control.
        combined = np.concatenate([feats['video'], feats['obb'], feats['location']], 1)
        combined /= np.maximum(np.linalg.norm(combined, axis=1, keepdims=True), 1e-8)
        maps, _ = neighbour_maps(combined, labels, video_of, k, False, rng)
        rows[f'NN-combined k={k}'] = per_episode_ap(maps, labels)
        for primary, secondary in (('obb', 'video'), ('video', 'obb')):
            for tag, rank in ((secondary, True), ('random', False)):
                rows[f'NN-{tag}|{primary} k={k}'] = per_episode_ap(
                    conditional_maps(feats[primary], feats[secondary], labels,
                                     video_of, k, args.pool, rng, rank), labels)

    print(f"\n  {'scorer':<28} {'PR-AUC':>8}   vs NN-random")
    for name, value in rows.items():
        ref = rows.get(f"NN-random k={name.split('k=')[-1].split()[0]}") if 'k=' in name else None
        delta = f"{value - ref:+8.4f}" if ref is not None and 'random' not in name else ''
        print(f"  {name:<28} {value:8.4f}   {delta}")

    # HOW SIMILAR the chosen neighbours actually were. top1 close to median means
    # the nearest neighbour is barely nearer than an average one, so "nearest" is
    # nearly arbitrary and the arm cannot be expected to beat NN-random.
    print(f"\n  {'arm':<10} {'top-1 cos':>10} {'median cos':>11} {'top1-median':>12} {'z':>7}")
    for name, s in sim_report.items():
        print(f"  {name:<10} {s['top1']:10.4f} {s['median']:11.4f} "
              f"{s['top1'] - s['median']:12.4f} {s['z']:7.2f}")

    if dose:
        print("\n  (AP - AP of NN-random on the SAME episode), by top-1 similarity")
        print("  quartile. Q1 = least similar neighbour. Rising left-to-right means")
        print("  a more similar neighbour really does predict better.")
        print(f"  {'arm':<14} {'Q1':>8} {'Q2':>8} {'Q3':>8} {'Q4':>8} {'Q4-Q1':>9}")
        for name, v in dose.items():
            print(f"  {name:<14} " + ' '.join(f"{x:+8.4f}" for x in v)
                  + f" {v[3] - v[0]:+9.4f}")

    return {'participant': pid, 'n_episodes': len(sel),
            'n_videos': int(len(set(video_of))), 'scores': rows,
            'similarity': sim_report, 'dose_response': dose}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--participants', default='P01')
    ap.add_argument('--n_episodes', type=int, default=0, help="0 = all")
    ap.add_argument('--k', type=int, nargs='+', default=[1, 10])
    ap.add_argument('--pool', type=int, default=100,
                    help="Pool size for the conditional arms: narrow to this many "
                         "neighbours by the primary signal, then pick k from the "
                         "pool by the secondary one (or at random, the control).")
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', default='')
    args = ap.parse_args()

    results = [run(p, args) for p in args.participants.split(',')]
    if args.out:
        os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
        json.dump(results, open(args.out, 'w'), indent=2)
        print(f"\nWrote {args.out}")


if __name__ == '__main__':
    main()
