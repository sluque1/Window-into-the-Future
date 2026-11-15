"""Which parts of the input actually move the prediction?

The attention probe (voxel_query_attention.py) says where the decoder LOOKS.
That is not the same as what it USES: 42% of attention mass on the obb segment is
consistent with the obb tokens setting the answer, and equally consistent with
them being a uniform backdrop the softmax has to put its mass somewhere on. This
script measures use directly, in the metric that is actually reported.

Method. Hold the checkpoint and the labels fixed and re-run evaluation with one
segment's tensors taken from a DIFFERENT episode of the same participant. If
PR-AUC does not move, that segment was not contributing to the prediction the
metric rewards. Swapping (rather than zeroing or masking) is the point: the model
still sees a real, in-distribution 30-second window of this kitchen, so the only
thing destroyed is the correspondence between the window and the episode being
asked about. A zeroed segment would confound "unused" with "off-distribution".

Because the test split of a LOOCV fold is one participant, "a different episode"
is automatically the same kitchen. The arms therefore separate:

    the kitchen               (preserved by every arm)
    this episode's window     (destroyed by the arm that swaps it)

--mode drop, AND WHY SWAPPING ALONE CANNOT ANSWER EVERYTHING. That first line is a
blind spot, not just a caveat. A same-participant swap hands the model another
window of the SAME kitchen, whose furniture is nearly identical -- so whatever a
segment contributes STATICALLY survives every swap arm, `all` included. Measured on
P08: base 0.3735, swap all 0.3335, and a per-voxel prior built from three OTHER
participants' labels only 0.1910 (analysis/label_prior_transfer.py). Destroying
every episode's worth of input costs 0.0400; the kitchen identity that survives it
is worth 0.1425, 3.6x more, and no swap arm touches it.

`--mode drop` zeroes the segment instead of donating one, which removes the static
part too. Zeroing is legitimate HERE, unlike in the swap design above, because 0 is
already the loader's own pad/empty value for every key: 0 is the pad index for the
voxel/class/volume tables, 0 in a *_mask means padding, and model.forward keeps obb
slot 0 attendable precisely for the 7-9% of real windows that have no boxes at all.
The dropped input is one the model already meets in training, so "unused" is not
being confounded with "off-distribution".

Read the two modes together, per segment S (S = one modality: video, location, obb
or gaze):
    swap S big, drop S big      -> S carries episode-specific information
    swap S ~0,  drop S big      -> S carries the kitchen, statically
    swap S ~0,  drop S ~0       -> S is genuinely unused

NOT EVERY DROP IS EQUALLY IN-DISTRIBUTION, and this decides how far each row can be
trusted. Counts from the P08 te10g8 run's own startup log:

    obb       4.1% of TRAIN windows have no boxes at all      -> safe
    gaze      4.1% of windows produced no gaze tokens;
              2 videos have no gaze CSV at all                -> safe
    location  3 videos have no SLAM -> all-padding locations  -> safe
    video     every video in the split has features, so an
              all-padding video segment is one the model has
              essentially NEVER seen                          -> EXTRAPOLATION

A low `drop video` therefore has two readings that this arm cannot separate: video
is load-bearing, or the model has never been asked to cope without it. The other
three rows do not have that ambiguity. If `drop video` looks decisive, confirm it
by retraining without the modality rather than by trusting this number.

Finally, on what an ablation of a TRAINED model does and does not license. This
measures whether THIS model's output depends on a segment. It says nothing about
whether a model trained from scratch without that segment would do worse -- it might
learn to read the same information somewhere else. Sensitivity, not necessity.

and the ALL arm is the honest ceiling on how well this checkpoint does as a pure
map of the kitchen -- everything it can still get right with no episode-specific
input whatsoever.

Read it against two numbers this script also prints:

  * BASE must reproduce the checkpoint's own logged eval_pr_auc. It is a guard,
    not a result: if it does not match, the loader settings reconstructed here
    disagree with the ones the run trained under and every arm below is noise.
  * STATIC-SELF, the AP of the mean predicted map handed to every episode, is
    what the model gets for free without reading anything. ALL should land near
    it; a large gap either way means the swap is doing something other than
    removing episode information.

Loader settings are read from the snapshot's own SLURM log rather than re-specified
here, because anticipation_time / future_time / time_anchor are NOT stored in
config.json and a mismatch silently rebuilds different episodes -- which looks
exactly like a real effect.

Usage:
    python analysis/input_swap_ablation.py                      # every te10 fold, best ckpt
    python analysis/input_swap_ablation.py --glob '*_te10g8_tln' --which last
    python analysis/input_swap_ablation.py --run <snapshot_dir> --n_episodes 500
"""
import argparse
import glob
import json
import os
import re
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from loader import HDEpicInteractionDataset  # noqa: E402
from model import CustomConfig, CustomTransformerModel  # noqa: E402
from train import torch_default_data_collator  # noqa: E402

SNAP = os.environ.get("SNAPSHOTS_ROOT", "/data/video_datasets/HD-EPIC/snapshots")
DEFAULT_GLOB = "2026-08-12_08-54-*_te10"

# Which batch keys carry each segment. Everything a segment contributes must move
# together -- swapping obb_voxels but not obb_mask would hand the model a box
# count from one episode and box positions from another, which is neither the
# real input nor a clean removal.
SEGMENT_KEYS = {
    'video':    ('visual_features', 'visual_mask'),
    'location': ('person_locations', 'person_location_mask'),
    'obb':      ('obb_voxels', 'obb_volumes', 'obb_classes', 'obb_mask'),
    'gaze':     ('gaze_voxels', 'gaze_mask'),
    'env':      ('env_grid', 'env_grid_mask'),
}


def per_episode_ap(scores, y):
    """Mean average precision over episodes with at least one positive and one
    negative -- the same filter train.py's ComputeMetricsWrapper applies, and the
    same quantity it logs as pr_auc (sklearn calls it average_precision_score).

    Vectorised over episodes; ties are handled the way sklearn does, which matters
    for the shared static map (many voxels share a score) and not at all for the
    model's own continuous logits.
    """
    y = np.asarray(y, np.float64)
    keep = (y.sum(1) > 0) & (y.sum(1) < y.shape[1])
    y = y[keep]
    n = np.arange(1, y.shape[1] + 1, dtype=np.float64)
    if scores.ndim == 1:
        order = np.argsort(-scores, kind='stable')
        s, ys = scores[order], y[:, order]
        tp = np.cumsum(ys, axis=1)
        ends = np.r_[np.nonzero(np.diff(s))[0], len(s) - 1]
        prec = tp[:, ends] / n[ends]
        rec = tp[:, ends] / y.sum(1, keepdims=True)
        return float((prec * np.diff(np.c_[np.zeros(len(y)), rec], axis=1)).sum(1).mean())
    order = np.argsort(-scores[keep], axis=1, kind='stable')
    ys = np.take_along_axis(y, order, axis=1)
    tp = np.cumsum(ys, axis=1)
    return float((((tp / n) * ys).sum(1) / y.sum(1)).mean())


def snapshot_root(checkpoint):
    """Walk up from a checkpoint to the snapshot directory holding out/*.out."""
    d = os.path.abspath(checkpoint)
    for _ in range(6):
        d = os.path.dirname(d)
        if glob.glob(os.path.join(d, 'out', '*.out')):
            return d
    raise SystemExit(f"Cannot find the snapshot root above {checkpoint} "
                     "(looked for out/*.out); pass --loader_args to override")


def loader_settings(run_dir):
    """Recover the loader options the run actually trained under, from its log.

    Only the ones config.json does not carry. Missing keys are fatal rather than
    defaulted: silently reconstructing a different future_time would change every
    label in the test set and still produce a plausible-looking table.
    """
    log = sorted(glob.glob(os.path.join(run_dir, 'out', '*.out')))[0]
    with open(log) as f:
        head = f.read(20000)
    want = {'OBS_TIME': int, 'FUTURE_TIME': float, 'ANTICIP_TIME': float,
            'TIME_ANCHOR': str, 'ENV_ORPHAN_FALLBACK': str,
            'TRAJECTORY_EXCLUDE_UNFINISHED': str, 'ENV_FALLBACK_AVERAGING': str,
            'GAZE_MAX_HZ': float, 'GAZE_TIME_OFFSET_S': float,
            'VOXEL_CENTERS_DIR': str, 'TEST': str,
            # FUSION/fixtures landed 2026-08-26 / 2026-08-31, after this script's
            # original DEFAULT_GLOB run. Without these the dataset silently
            # rebuilds under 'grid' fusion with no fixtures for any run that
            # trained under 'iou' + FIXTURE_OBBS -- which fails the BASE-vs-
            # logged-eval_pr_auc guard below rather than staying quiet.
            'FUSION': str, 'FUSED_CACHE': str, 'BOXER_CONF_THRESHOLD': float,
            'OBB_VOXEL_GRID_PRUNE': str, 'FIXTURE_OBBS': str,
            'FIXTURE_IOU_THRESH': float, 'FIXTURE_COS_THRESH': float}
    out = {}
    for key, cast in want.items():
        m = re.search(rf'\b{key}=(\S*)', head)
        if m is None:
            raise SystemExit(f"{log} has no {key}= line; refusing to guess it")
        out[key] = cast(m.group(1)) if m.group(1) else ('' if cast is str else cast(0))
    return out


def load_model(path):
    """Strict load. A partially-matching state dict leaves randomly initialised
    parameters that still emit a tidy-looking prediction, so anything unexpected
    is fatal rather than warned about."""
    config = CustomConfig.from_pretrained(path)
    weights = os.path.join(path, 'model.safetensors')
    if not os.path.exists(weights):
        raise SystemExit(f"{path} holds no model.safetensors")
    from safetensors.torch import load_file
    model = CustomTransformerModel(config)
    missing, unexpected = model.load_state_dict(load_file(weights), strict=False)
    if missing or unexpected:
        raise SystemExit(f"State dict does not match the model built from "
                         f"{path}/config.json\n  missing: {missing[:8]}\n"
                         f"  unexpected: {unexpected[:8]}")
    return model.eval(), config


def pick_checkpoint(run_dir, which):
    cks = glob.glob(f"{run_dir}/results/*/checkpoint-*")
    if not cks:
        return None, None
    by_step = {int(c.rsplit('-', 1)[1]): c for c in cks}
    last = by_step[max(by_step)]
    state = json.load(open(f"{last}/trainer_state.json"))
    if which == 'last':
        return last, state
    best = state.get('best_model_checkpoint')
    step = int(os.path.basename(best).rsplit('-', 1)[1]) if best else min(by_step)
    if step not in by_step:
        raise SystemExit(f"{run_dir}: best checkpoint {step} is no longer on disk "
                         f"(have {sorted(by_step)}) -- save_total_limit evicted it")
    return by_step[step], state


def logged_pr_auc(state, checkpoint):
    """The eval_pr_auc this checkpoint recorded when it was written, for the guard."""
    step = int(os.path.basename(checkpoint).rsplit('-', 1)[1])
    for entry in state.get('log_history', []):
        if entry.get('step') == step and 'eval_pr_auc' in entry:
            return entry['eval_pr_auc']
    return None


def run_fold(run_dir, args):
    checkpoint, state = pick_checkpoint(run_dir, args.which)
    if checkpoint is None:
        print(f"  {os.path.basename(run_dir)}: no checkpoints, skipping")
        return None
    model, config = load_model(checkpoint)
    settings = loader_settings(snapshot_root(checkpoint))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    active = [m for m in config.input_modality.split(',') if m in SEGMENT_KEYS]
    pid = settings['TEST']
    print(f"\n=== {pid}  {os.path.basename(checkpoint)} "
          f"[{config.input_modality}] ===")

    dataset = HDEpicInteractionDataset(
        'test', participants=pid.split(','),
        observation_time=config.observation_time,
        anticipation_time=settings['ANTICIP_TIME'],
        future_time=settings['FUTURE_TIME'],
        static_env=config.use_static_env,
        voxel_centers_dir=settings['VOXEL_CENTERS_DIR'] or None,
        max_obb_tokens=config.max_obb_tokens,
        trajectory_simple=config.trajectory_simple,
        trajectory_exclude_unfinished=settings['TRAJECTORY_EXCLUDE_UNFINISHED'] == 'true',
        env_orphan_fallback=settings['ENV_ORPHAN_FALLBACK'] == 'true',
        env_class_averaging=config.env_class_averaging,
        env_max_classes_per_voxel=config.env_max_classes_per_voxel,
        env_fallback_averaging=settings['ENV_FALLBACK_AVERAGING'],
        gaze_max_hz=settings['GAZE_MAX_HZ'],
        gaze_time_offset_s=settings['GAZE_TIME_OFFSET_S'],
        time_anchor=settings['TIME_ANCHOR'],
        fusion=settings['FUSION'],
        fused_cache=settings['FUSED_CACHE'] == 'true',
        boxer_conf_threshold=(None if settings['BOXER_CONF_THRESHOLD'] < 0
                              else settings['BOXER_CONF_THRESHOLD']),
        obb_voxel_grid_prune=settings['OBB_VOXEL_GRID_PRUNE'] == 'true',
        fixture_obbs=settings['FIXTURE_OBBS'] == 'true',
        fixture_iou_thresh=settings['FIXTURE_IOU_THRESH'],
        fixture_cos_thresh=settings['FIXTURE_COS_THRESH'],
        verbose=False)

    # --- Observation-window length, and the option to require a full one -------
    # loader._build_episodes clamps obs_start to 0 and keeps anything with
    # obs_end >= 1.0, so an anchor less than (anticipation + observation_time)
    # seconds into a recording yields a TRUNCATED window -- as short as one
    # second. Those episodes dilute every arm in the same direction: swapping the
    # video of an episode that has 4 seconds of video cannot move the prediction
    # much, so they push each segment's score toward 100% (no effect) regardless
    # of whether that segment matters on a real window.
    lengths = np.array([d['obs_end'] - d['obs_start'] for d in dataset.dataset])
    full = float(config.observation_time)
    print(f"  obs window: {100 * (lengths >= full).mean():.1f}% full ({full:.0f}s), "
          f"median {np.median(lengths):.0f}s, "
          f"{100 * (lengths < full / 2).mean():.1f}% under {full / 2:.0f}s, "
          f"min {lengths.min():.0f}s")
    keep = np.nonzero(lengths >= args.min_obs_seconds)[0] if args.min_obs_seconds > 0 \
        else np.arange(len(dataset))
    if args.min_obs_seconds > 0:
        print(f"  --min_obs_seconds {args.min_obs_seconds:g}: keeping {len(keep)} of "
              f"{len(dataset)} episodes ({100 * len(keep) / len(dataset):.1f}%)")
        if len(keep) < 50:
            print("  too few episodes survive the filter; skipping this fold")
            return None

    n = len(keep) if args.n_episodes <= 0 else min(args.n_episodes, len(keep))
    # (label, mode, segments). 'base' is the unperturbed reference; every other
    # entry names its mode, so a --mode both table stays unambiguous.
    segs = active + (['all'] if len(active) > 1 else [])
    plan = [('base', None, [])]
    for mode in (['swap', 'drop'] if args.mode == 'both' else [args.mode]):
        for s in segs:
            plan.append((f"{mode} {s}", mode, active if s == 'all' else [s]))
    arms = [label for label, _, _ in plan]
    rng = np.random.default_rng(args.seed)

    # A SUBSET MUST BE RANDOM, not the first n. Episodes are built video by video,
    # so dataset[:400] is one or two recordings rather than a sample of the
    # participant -- it scores a different quantity from the logged metric and
    # makes the guard below fire for a reason that has nothing to do with the run.
    subset = n < len(keep)
    sel = keep[np.sort(rng.choice(len(keep), n, replace=False))] if subset else keep

    # One fixed permutation of POSITIONS, shared by every arm, so the arms differ
    # only in WHICH keys are donated and not in which episode donated them. A
    # random permutation of n>>1 leaves ~1 fixed point, which is negligible and
    # left alone rather than patched into a derangement (that would be a
    # different, slightly biased map).
    perm = rng.permutation(n)

    logits = {arm: np.empty((n, config.output_dim), np.float32) for arm in arms}
    labels = np.empty((n, config.output_dim), np.uint8)
    n_batches = (n + args.batch_size - 1) // args.batch_size

    for bi, start in enumerate(range(0, n, args.batch_size)):
        idx = list(range(start, min(start + args.batch_size, n)))
        base = torch_default_data_collator([dataset[int(sel[i])] for i in idx])
        donor = torch_default_data_collator(
            [dataset[int(sel[perm[i]])] for i in idx])
        if bi % 25 == 0:
            print(f"    batch {bi + 1}/{n_batches}", flush=True)
        labels[idx] = base['labels']['interaction'].numpy().astype(np.uint8)

        for label, mode, segments in plan:
            batch = dict(base)
            for segment in segments:
                for key in SEGMENT_KEYS[segment]:
                    if key not in base:
                        continue
                    if mode == 'swap':
                        if key in donor:
                            batch[key] = donor[key]
                    else:
                        # 0 is the loader's own pad/empty value for every key in
                        # SEGMENT_KEYS, so zeros_like is the real absent-modality
                        # input rather than a synthetic one. See the module
                        # docstring on why zeroing is safe here but not for swap.
                        batch[key] = torch.zeros_like(base[key])
            batch = {k: (v.to(device) if torch.is_tensor(v) else
                         {sk: sv.to(device) for sk, sv in v.items()}
                         if isinstance(v, dict) else v)
                     for k, v in batch.items()}
            with torch.no_grad():
                logits[label][idx] = model(**batch)['interaction'].float().cpu().numpy()

    base_ap = per_episode_ap(logits['base'].astype(np.float64), labels)
    static = per_episode_ap(logits['base'].astype(np.float64).mean(0), labels)
    ref = logged_pr_auc(state, checkpoint)

    guard = ''
    if ref is not None:
        drift = abs(base_ap - ref)
        # The logged metric is over the WHOLE split, so it is only a strict guard
        # when this run scores the whole split too. Under --min_obs_seconds the
        # episode set is deliberately different and base is EXPECTED to move --
        # that difference is a result, not a fault, so do not flag it as one.
        if subset or args.min_obs_seconds > 0:
            why = ("filtered" if args.min_obs_seconds > 0 else
                   f"{100 * n / len(dataset):.0f}% sample")
            guard = (f"logged {ref:.4f} on all {len(dataset)}, reproduced "
                     f"{base_ap:.4f} on this {why} set (not comparable by design)")
        else:
            guard = (f"logged {ref:.4f}, reproduced {base_ap:.4f} "
                     f"({'OK' if drift < 0.005 else 'MISMATCH -- do not trust the arms'})")
    print(f"  {n} of {len(dataset)} episodes | {guard}")
    print(f"  {'arm':<16} {'PR-AUC':>8} {'delta':>9} {'% of base':>10}")
    print(f"  {'base':<16} {base_ap:8.4f} {'':>9} {'':>10}")
    rows = {'base': base_ap}
    for arm in arms[1:]:
        ap = per_episode_ap(logits[arm].astype(np.float64), labels)
        rows[arm] = ap
        print(f"  {arm:<16} {ap:8.4f} {ap - base_ap:+9.4f} "
              f"{100 * ap / base_ap:9.1f}%")
    print(f"  {'static-self':<16} {static:8.4f} {static - base_ap:+9.4f} "
          f"{100 * static / base_ap:9.1f}%   (mean map, no input at all)")
    rows['static-self'] = static
    return {'participant': pid, 'checkpoint': checkpoint, 'n_episodes': n,
            'logged_pr_auc': ref, 'arms': rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--glob', default=DEFAULT_GLOB,
                    help="Snapshot-directory glob, relative to the snapshots root.")
    ap.add_argument('--run', default='', help="A single snapshot dir; overrides --glob.")
    ap.add_argument('--which', default='best', choices=['best', 'last'],
                    help="Which checkpoint of each run. 'best' is the one "
                         "metric_for_best_model selected, i.e. the peak.")
    ap.add_argument('--n_episodes', type=int, default=0,
                    help="0 = every episode that passes --min_obs_seconds. Any "
                         "smaller number is drawn at RANDOM, never as a prefix: "
                         "episodes are built video by video, so dataset[:n] is one "
                         "recording rather than a sample of the participant.")
    ap.add_argument('--min_obs_seconds', type=float, default=0.0,
                    help="Drop episodes whose observation window is shorter than "
                         "this. loader._build_episodes clamps obs_start to 0 and "
                         "keeps anything with obs_end >= 1.0, so anchors near the "
                         "start of a recording produce windows of a few seconds. "
                         "Those cannot respond to a swapped 30s input and so bias "
                         "every arm toward 'no effect'. Set to the run's "
                         "observation_time (30) for full windows only.")
    ap.add_argument('--mode', default='swap', choices=['swap', 'drop', 'both'],
                    help="swap: donate the segment from another episode of the SAME "
                         "participant -- removes this episode's information but "
                         "keeps the kitchen. drop: zero the segment to its pad/empty "
                         "value -- removes the kitchen too. both: one table with "
                         "every arm, which is what separates a segment that carries "
                         "the static layout from one that carries the episode.")
    ap.add_argument('--batch_size', type=int, default=16)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', default='')
    args = ap.parse_args()

    runs = [args.run] if args.run else sorted(glob.glob(os.path.join(SNAP, args.glob)))
    if not runs:
        raise SystemExit(f"No runs matched {args.glob}")
    print(f"{len(runs)} run(s), {args.which} checkpoint, seed {args.seed}")

    results = [r for r in (run_fold(d, args) for d in runs) if r]
    if not results:
        raise SystemExit("nothing scored")

    every = sorted({a for r in results for a in r['arms']} - {'base', 'static-self'})
    print("\n" + "=" * 72)
    print("SUMMARY -- PR-AUC as a % of that fold's unablated score")
    header = f"{'fold':<6} {'base':>8} " + ' '.join(f"{'swap ' + a:>11}" for a in every)
    print(header + f" {'static-self':>12}")
    print("-" * len(header + '             '))
    for r in results:
        cells = ' '.join(f"{100 * r['arms'][a] / r['arms']['base']:10.1f}%"
                         if a in r['arms'] else f"{'-':>11}" for a in every)
        print(f"{r['participant']:<6} {r['arms']['base']:8.4f} {cells} "
              f"{100 * r['arms']['static-self'] / r['arms']['base']:11.1f}%")
    print("-" * len(header + '             '))
    means = {a: np.mean([100 * r['arms'][a] / r['arms']['base']
                         for r in results if a in r['arms']]) for a in every}
    print(f"{'mean':<6} {np.mean([r['arms']['base'] for r in results]):8.4f} "
          + ' '.join(f"{means[a]:10.1f}%" for a in every)
          + f" {np.mean([100 * r['arms']['static-self'] / r['arms']['base'] for r in results]):11.1f}%")
    print("\nA column near 100% means that segment can be replaced by another "
          "episode's\nwithout the reported metric noticing.")

    if args.out:
        os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
        json.dump(results, open(args.out, 'w'), indent=2)
        print(f"\nWrote {args.out}")


if __name__ == '__main__':
    main()
