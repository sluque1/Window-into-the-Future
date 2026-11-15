"""Does the trained model's output actually depend on the gaze tokens?

The Q5 gaze arm tracks its no-gaze baseline to ~0.004 PR-AUC at every eval point.
That is expected from the setup alone -- identical parameter count means an
identical init, and the runs share seed, data order and eval cadence, so the whole
training trajectory is shared and only 30 tokens differ. Tracking is therefore NOT
evidence either way about whether gaze is used, and the eye cannot separate "gaze
adds nothing" from "gaze never reaches the weights".

This measures it directly, on the trained checkpoint, with dropout off.

METHOD. For one batch of full-length episodes, take the base logits, then re-run
with ONE segment's tensors donated by a different episode of the same participant
-- exactly input_swap_ablation.py's perturbation, but read out as |delta logit|
rather than PR-AUC, so it needs one batch instead of the whole split.

THE COMPARISON IS THE POINT. An absolute |delta| is uninterpretable: the logits
have no natural scale, and a small number could mean "gaze is ignored" or "this
model is insensitive to everything". So gaze is measured beside `location` and
`obb` under the identical perturbation. Read the RELATIVE column:

    gaze ~= 0            -> the weights ignore gaze; the +0.004 is dropout noise
    gaze << obb          -> used, but a minor input (the honest likely answer)
    gaze ~ location      -> used about as much as the actor's own position

Two controls guard the reading. `none` re-runs the base batch unperturbed and must
give exactly 0.0 -- anything else means eval mode is not deterministic and every
row below is noise. `all` perturbs every segment at once and is the ceiling.

Also printed, because it is the most likely explanation if gaze turns out to be
used-but-worthless: how often the gaze voxel is simply the LOCATION voxel. Under
--location_tokens joint the model already holds the actor's position for that same
second, so a gaze token that agrees with it carries no new information -- it is a
duplicate wearing gaze_type_embedding. CLAUDE.md records P08's depth_m as pinned
near 0.27 m (chin distance), which predicts near-total agreement on that fold.

Usage:
    python analysis/gaze_sensitivity.py                       # P08 gaze run, last ckpt
    python analysis/gaze_sensitivity.py --run <snapshot_dir>
    python analysis/gaze_sensitivity.py --batch_size 4        # if memory is tight
"""
import argparse
import glob
import os
import re
import sys

import numpy as np
import torch

SNAP = os.environ.get("SNAPSHOTS_ROOT", "/data/video_datasets/HD-EPIC/snapshots")
DEFAULT_RUN = (f"{SNAP}/2026-08-23_21-41-05_loocv_P08_vlog_vq_fourier_"
               f"factorised_cls_prior_pe_locjoint_gazejoint_e1")

# Identical to input_swap_ablation.SEGMENT_KEYS: everything a segment contributes
# has to move together, or the model gets one episode's box count with another's
# box positions -- neither the real input nor a clean removal.
SEGMENT_KEYS = {
    'video':    ('visual_features', 'visual_mask'),
    'location': ('person_locations', 'person_location_mask'),
    'obb':      ('obb_voxels', 'obb_volumes', 'obb_classes', 'obb_mask'),
    'gaze':     ('gaze_voxels', 'gaze_mask'),
    'env':      ('env_grid', 'env_grid_mask'),
}


def _np(x):
    """loader.py hands some per-episode keys back as torch tensors and others as
    plain ndarrays -- person_locations is built with torch.zeros, gaze_voxels comes
    straight out of gaze.build_episode_gaze_tokens as numpy. Accept either."""
    return x.numpy() if hasattr(x, 'numpy') else np.asarray(x)


def loader_settings(run_dir):
    """The loader options config.json does not carry, read off the run's own log.

    Fatal rather than defaulted: silently reconstructing a different future_time
    changes every label and still prints a plausible-looking table.
    """
    log = sorted(glob.glob(os.path.join(run_dir, 'out', '*.out')))[0]
    with open(log) as f:
        head = f.read(20000)
    want = {'FUTURE_TIME': float, 'ANTICIP_TIME': float, 'TIME_ANCHOR': str,
            'ENV_ORPHAN_FALLBACK': str, 'TRAJECTORY_EXCLUDE_UNFINISHED': str,
            'ENV_FALLBACK_AVERAGING': str, 'GAZE_MAX_HZ': float,
            'GAZE_TIME_OFFSET_S': float, 'VOXEL_CENTERS_DIR': str, 'TEST': str}
    out = {}
    for key, cast in want.items():
        m = re.search(rf'\b{key}=(\S*)', head)
        if m is None:
            raise SystemExit(f"{log} has no {key}= line; refusing to guess it")
        out[key] = cast(m.group(1)) if m.group(1) else ('' if cast is str else cast(0))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', default=DEFAULT_RUN)
    ap.add_argument('--which', default='last', choices=['last', 'best'])
    ap.add_argument('--batch_size', type=int, default=8)
    ap.add_argument('--n_batches', type=int, default=4)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--skip_model', action='store_true',
                    help="Only the gaze-vs-location duplication stats. Skips the "
                         "weight load and every forward pass, so it answers 'why' "
                         "without re-paying for the swap table you already have.")
    args = ap.parse_args()

    run = os.path.abspath(args.run)
    # The run's OWN frozen code, ahead of the repo copy: this checkpoint's weights
    # only mean anything under the model.py that produced them.
    sys.path.insert(0, run)
    from loader import HDEpicInteractionDataset          # noqa: E402
    from model import CustomConfig, CustomTransformerModel  # noqa: E402
    from train import torch_default_data_collator        # noqa: E402
    from safetensors.torch import load_file              # noqa: E402

    cks = glob.glob(f"{run}/results/*/checkpoint-*")
    if not cks:
        raise SystemExit(f"No checkpoints under {run}/results/")
    by_step = {int(c.rsplit('-', 1)[1]): c for c in cks}
    ck = by_step[max(by_step)]
    if args.which == 'best':
        import json
        state = json.load(open(f"{by_step[max(by_step)]}/trainer_state.json"))
        best = state.get('best_model_checkpoint')
        if best:
            step = int(os.path.basename(best).rsplit('-', 1)[1])
            if step not in by_step:
                raise SystemExit(f"best checkpoint {step} was evicted; have {sorted(by_step)}")
            ck = by_step[step]

    cfg = CustomConfig.from_pretrained(ck)   # cheap: json only, no weights
    model = None
    if not args.skip_model:
        model = CustomTransformerModel(cfg)
        # Strict: a partial load leaves randomly initialised weights that still emit
        # a tidy-looking prediction.
        missing, unexpected = model.load_state_dict(
            load_file(f"{ck}/model.safetensors"), strict=False)
        if missing or unexpected:
            raise SystemExit(f"state dict mismatch\n  missing: {missing[:8]}\n"
                             f"  unexpected: {unexpected[:8]}")
        model.eval()   # dropout OFF -- otherwise every row below is dropout noise

    s = loader_settings(run)
    print(f"run   : {os.path.basename(run)}")
    print(f"ckpt  : {os.path.basename(ck)}   modality: {cfg.input_modality}")
    print(f"layout: location_tokens={getattr(cfg, 'location_tokens', 'axis')} "
          f"gaze_tokens={getattr(cfg, 'gaze_tokens', 'axis')} "
          f"position_encoding={getattr(cfg, 'position_encoding', 'n/a')}")

    ds = HDEpicInteractionDataset(
        'test', participants=s['TEST'].split(','),
        observation_time=cfg.observation_time,
        anticipation_time=s['ANTICIP_TIME'], future_time=s['FUTURE_TIME'],
        static_env=cfg.use_static_env,
        voxel_centers_dir=s['VOXEL_CENTERS_DIR'] or None,
        max_obb_tokens=cfg.max_obb_tokens,
        trajectory_simple=cfg.trajectory_simple,
        trajectory_exclude_unfinished=s['TRAJECTORY_EXCLUDE_UNFINISHED'] == 'true',
        env_orphan_fallback=s['ENV_ORPHAN_FALLBACK'] == 'true',
        env_class_averaging=cfg.env_class_averaging,
        env_max_classes_per_voxel=cfg.env_max_classes_per_voxel,
        env_fallback_averaging=s['ENV_FALLBACK_AVERAGING'],
        gaze_max_hz=s['GAZE_MAX_HZ'], gaze_time_offset_s=s['GAZE_TIME_OFFSET_S'],
        time_anchor=s['TIME_ANCHOR'], verbose=False)

    # Full windows only. A 4-second window cannot respond to a swapped 30s input,
    # so short episodes push every arm toward "no effect" regardless of the truth.
    lengths = np.array([d['obs_end'] - d['obs_start'] for d in ds.dataset])
    keep = np.nonzero(lengths >= cfg.observation_time)[0]
    print(f"episodes: {len(keep)} full-length of {len(ds)}  (participant {s['TEST']})")

    rng = np.random.default_rng(args.seed)
    n = min(args.batch_size * args.n_batches, len(keep))
    sel = keep[np.sort(rng.choice(len(keep), n, replace=False))]
    perm = rng.permutation(n)

    # --- Is the gaze token simply a duplicate of the location token? -------------
    # Printed FIRST because it is fast and needs no weights. Under
    # --location_tokens joint the model already holds the actor's position for that
    # same second, so a gaze token agreeing with it carries nothing new -- it is a
    # duplicate wearing gaze_type_embedding. This is the reading to reach for when
    # the swap table below says gaze IS used and PR-AUC still does not move.
    same = tok = 0
    dist = []
    for i in sel:
        d = ds[int(i)]
        g, gm = _np(d['gaze_voxels']), _np(d['gaze_mask']).astype(bool)
        l, lm = _np(d['person_locations']), _np(d['person_location_mask']).astype(bool)
        both = gm & lm
        if not both.any():
            continue
        gg, ll = g[both], l[both]
        tok += int(both.sum())
        same += int((gg == ll).all(axis=1).sum())
        dist.append(np.abs(gg - ll).max(axis=1))
    if tok:
        dist = np.concatenate(dist)
        print(f"gaze vs location voxel, same second, {tok} tokens:")
        print(f"  identical voxel     {100 * same / tok:5.1f}%")
        print(f"  within 1 voxel      {100 * (dist <= 1).mean():5.1f}%")
        print(f"  within 2 voxels     {100 * (dist <= 2).mean():5.1f}%")
        print(f"  median offset       {np.median(dist):.0f} voxels   "
              f"mean {dist.mean():.2f}\n")
    else:
        print("no second has BOTH a real gaze token and a real location token\n")

    if args.skip_model:
        return

    active = [m for m in cfg.input_modality.split(',') if m in SEGMENT_KEYS]
    arms = ['none'] + active + ['all']
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(dev)
    print(f"device: {dev}   {n} episodes, {len(arms)} arms\n")

    absdiff = {a: [] for a in arms}
    base_all = []
    for bi, start in enumerate(range(0, n, args.batch_size)):
        idx = list(range(start, min(start + args.batch_size, n)))
        base = torch_default_data_collator([ds[int(sel[i])] for i in idx])
        donor = torch_default_data_collator([ds[int(sel[perm[i]])] for i in idx])
        print(f"  batch {bi + 1}/{(n + args.batch_size - 1) // args.batch_size}", flush=True)

        def run_batch(batch):
            batch = {k: (v.to(dev) if torch.is_tensor(v) else
                         {sk: sv.to(dev) for sk, sv in v.items()}
                         if isinstance(v, dict) else v)
                     for k, v in batch.items()}
            with torch.no_grad():
                return model(**batch)['interaction'].float().cpu().numpy()

        ref = run_batch(dict(base))
        base_all.append(ref)
        for arm in arms:
            batch = dict(base)
            if arm != 'none':
                for segment in (active if arm == 'all' else [arm]):
                    for key in SEGMENT_KEYS[segment]:
                        if key in base and key in donor:
                            batch[key] = donor[key]
            absdiff[arm].append(np.abs(run_batch(batch) - ref))

    base_all = np.concatenate(base_all)
    print(f"\nbase logits: mean {base_all.mean():+.3f}  std {base_all.std():.3f}\n")

    scores = {a: np.concatenate(v) for a, v in absdiff.items()}
    ref_scale = scores['obb'].mean() if 'obb' in scores else max(
        (v.mean() for a, v in scores.items() if a not in ('none', 'all')), default=1.0)
    print(f"  {'swapped segment':<18} {'mean|dlogit|':>13} {'max|dlogit|':>12} "
          f"{'relative to obb':>16}")
    for arm in arms:
        v = scores[arm]
        rel = '' if arm == 'none' else f"{100 * v.mean() / ref_scale:14.1f}%"
        note = '   <- control, must be 0.000' if arm == 'none' else ''
        print(f"  {arm:<18} {v.mean():13.5f} {v.max():12.5f} {rel:>16}{note}")

    if scores['none'].max() > 0:
        print("\n!! the unperturbed control moved -- eval is not deterministic, "
              "every row above is noise")


if __name__ == '__main__':
    main()