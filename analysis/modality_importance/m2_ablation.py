"""Method 2 -- masking / ablation importance.

Method 1 measured how much the target MOVES for a small nudge to a token. This
measures what happens when information is actually removed, in the metric that is
reported: PR-AUC on the held-out participant. The two answer different questions
and are expected to disagree; the disagreement is the finding, not an error.

    # modality level, both checkpoints of a run, every intervention
    python analysis/modality_importance/m2_ablation.py --run A

    # token level (one token at a time) on a subset
    python analysis/modality_importance/m2_ablation.py --run A --level token --n_episodes 128

Builds on analysis/input_swap_ablation.py, whose reasoning about swap-vs-drop and
about which drops are in-distribution is carried over below, and whose
``per_episode_ap`` is imported rather than reimplemented so this reports the same
quantity train.py logs as ``pr_auc``. Two things that script does not do and this
one must: it has no 'trajectory' entry in SEGMENT_KEYS (so on the
video,location,obb,trajectory runs analysed here it would silently skip a whole
modality), and it works at segment level only.

THE INTERVENTIONS, and why more than one is needed.

  swap   Take the segment's tensors from a DIFFERENT episode of the same
         participant. The model still sees a real, in-distribution 30-second
         window of this kitchen; the only thing destroyed is the correspondence
         between the window and the episode being asked about. This is the least
         distribution-shifting intervention available and it is the primary one.
         Its blind spot: a different window of the SAME kitchen has nearly the
         same furniture, so whatever a segment contributes STATICALLY survives it.

  drop   Zero the segment's tensors. Legitimate here, unlike in most settings,
         because 0 is already the loader's own pad/empty value for every key --
         0 is the pad index for the voxel/class/volume tables and 0 in a *_mask
         means padding. It removes the static part that swap preserves.
         NOT equally safe across modalities (see below).

  mask   Exclude the tokens from attention entirely, via the padding mask the
         encoder and decoder share. This is genuine REMOVAL rather than
         substitution: Method 1 measured a padded token's gradient as exactly 0,
         so a masked token provably cannot influence the output. It is the only
         intervention that works at token level for cls and sep, which have no
         input tensors of their own, and it is the token-level primary.

  mean   Replace the token embeddings with their segment mean over the dataset --
         "a typical token of this kind". This is exactly the baseline Integrated
         Gradients used in Method 1, so it is the arm that makes the two methods
         directly comparable.

HOW IN-DISTRIBUTION IS EACH DROP? This decides how far each row can be trusted,
and the answer differs per modality. From the runs' own startup logs:

    obb         1 of 5007 P08 / 7 of 3311 P04 test windows have NO boxes at all,
                and model.forward keeps obb slot 0 attendable precisely for that
                case                                            -> safe
    location    3 videos in the dataset have no SLAM at all, giving all-padding
                locations; and real episodes reach loc_valid=1/30 -> safe
    trajectory  many episodes have 0 valid tracks already
                (episode 0 of P08: traj_tokens=0/16)             -> safe
    video       every video in both splits has features, and the minimum observed
                is 2-4 of 30 valid seconds, never 0              -> EXTRAPOLATION

So a large `drop video` has two readings this arm cannot separate: video is
load-bearing, or the model has never been asked to cope without it. The other
three rows do not carry that ambiguity. Confirming `drop video` needs a retrained
model, which is out of scope here and is stated as a limitation rather than
guessed at.

READING THE TABLE, per segment S:
    swap S big, drop S big   -> S carries episode-specific information
    swap S ~0,  drop S big   -> S carries the kitchen, statically
    swap S ~0,  drop S ~0    -> S is genuinely unused
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import common  # noqa: E402
import runs as runspec  # noqa: E402
from input_swap_ablation import per_episode_ap  # noqa: E402

# Every tensor a segment contributes must move together: swapping obb_voxels but
# not obb_mask would hand the model a box COUNT from one episode and box POSITIONS
# from another, which is neither the real input nor a clean removal. Extends
# input_swap_ablation.SEGMENT_KEYS with the trajectory entry it lacks.
SEGMENT_KEYS = {
    'video':      ('visual_features', 'visual_mask'),
    'location':   ('person_locations', 'person_location_mask'),
    'obb':        ('obb_voxels', 'obb_volumes', 'obb_classes', 'obb_mask'),
    'trajectory': ('trajectory_tokens', 'trajectory_mask', 'trajectory_classes',
                   'trajectory_start_voxel', 'trajectory_end_voxel',
                   'trajectory_distance'),
    'gaze':       ('gaze_voxels', 'gaze_mask'),
    'env':        ('env_grid', 'env_grid_mask'),
}

INTERVENTIONS = ('swap', 'drop', 'mask', 'mean')


# ---------------------------------------------------------------------------

def per_row_ap(scores, y):
    """[N] average precision, one per row -- the per-row form of the mean that
    input_swap_ablation.per_episode_ap returns.

    Same arithmetic as that function's 2-D branch, without the final .mean(), so
    the token-level loop can score a whole batch of masked variants in one
    vectorised call instead of 27k Python-level ones. Rows with no positive (or
    no negative) get NaN, matching that function's filter rather than silently
    scoring them.
    """
    y = np.asarray(y, np.float64)
    npos = y.sum(1)
    ok = (npos > 0) & (npos < y.shape[1])
    out = np.full(len(y), np.nan)
    if not ok.any():
        return out
    order = np.argsort(-scores[ok], axis=1, kind='stable')
    ys = np.take_along_axis(y[ok], order, axis=1)
    tp = np.cumsum(ys, axis=1)
    n = np.arange(1, ys.shape[1] + 1, dtype=np.float64)
    out[ok] = ((tp / n) * ys).sum(1) / npos[ok]
    return out


def per_row_roc(scores, y):
    """[N] ROC-AUC, one per row. NaN where a row has only one class."""
    y = np.asarray(y, np.float64)
    npos = y.sum(1)
    ok = (npos > 0) & (npos < y.shape[1])
    out = np.full(len(y), np.nan)
    if not ok.any():
        return out
    # ASCENDING: the Mann-Whitney form sums the ranks of the positives with
    # rank 1 = lowest score. Descending computes 1 - AUC, a plausible-looking
    # number caught only by the sklearn cross-check.
    order = np.argsort(scores[ok], axis=1, kind='stable')
    ys = np.take_along_axis(y[ok], order, axis=1)
    ranks = np.arange(1, ys.shape[1] + 1, dtype=np.float64)
    p = npos[ok]
    n = ys.shape[1] - p
    out[ok] = ((ys * ranks).sum(1) - p * (p + 1) / 2) / (p * n)
    return out


def roc_auc(scores, y):
    """Mean per-episode ROC-AUC over episodes with both classes present."""
    y = np.asarray(y, np.float64)
    keep = (y.sum(1) > 0) & (y.sum(1) < y.shape[1])
    y, s = y[keep], scores[keep] if scores.ndim > 1 else np.tile(scores, (keep.sum(), 1))
    # ASCENDING: the Mann-Whitney form below sums the ranks of the positives with
    # rank 1 = lowest score. Sorting descending computes 1 - AUC, which is a
    # plausible-looking number and was caught only by the sklearn cross-check.
    order = np.argsort(s, axis=1, kind='stable')
    ys = np.take_along_axis(y, order, axis=1)
    npos = ys.sum(1)
    nneg = ys.shape[1] - npos
    ranks = np.arange(1, ys.shape[1] + 1, dtype=np.float64)
    sum_rank_pos = (ys * ranks).sum(1)
    return float((((sum_rank_pos - npos * (npos + 1) / 2) / (npos * nneg))).mean())


def apply_input_intervention(batch, segment, mode, donor_batch=None):
    """swap / drop, applied to the batch tensors before the forward pass.

    Returns a shallow copy; the caller's batch is untouched so the baseline and
    every arm see byte-identical inputs apart from the intervention itself.
    """
    out = dict(batch)
    for key in SEGMENT_KEYS[segment]:
        if key not in batch or batch[key] is None:
            continue
        if mode == 'drop':
            # 0 is the loader's own pad/empty value for every one of these keys.
            out[key] = torch.zeros_like(batch[key])
        elif mode == 'swap':
            out[key] = donor_batch[key]
        else:
            raise ValueError(f"apply_input_intervention does not handle {mode!r}")
    return out


def segment_mask_edit(layout, names, device):
    """mask_edit closure that masks out every token of the named segments."""
    extra = np.zeros(layout.seq_len, dtype=bool)
    for n in names:
        extra |= layout.index_mask(n)
    extra_t = torch.as_tensor(extra, device=device)

    def edit(m):
        return m | extra_t.unsqueeze(0)
    return edit


def token_mask_edit(positions, seq_len, device):
    """mask_edit closure that masks out the given sequence positions."""
    extra = torch.zeros(seq_len, dtype=torch.bool, device=device)
    extra[list(positions)] = True

    def edit(m):
        return m | extra.unsqueeze(0)
    return edit


def evaluate_arms(model, dataset, indices, device, batch_size, layout, arms,
                  donors, active, mean_baseline=None, progress_every=20):
    """Every arm, in ONE pass over the data.

    The obvious structure -- a full pass per arm -- calls dataset[i] once per
    episode per arm, which at 21 arms over 5007 episodes is 105k __getitem__ calls
    of feature loading and env lookup. That is CPU work the GPU cannot absorb, and
    it exhausted the cluster's 1800 s CPU-time ceiling partway through the second
    arm. Loading each episode once and running every arm on it cuts the dataset
    access by the number of arms.

    Scores are reduced to per-episode AP / ROC / targets inside the loop rather
    than accumulating logits: 21 arms x 5007 episodes x 4096 voxels of float32
    would be 1.7 GB held for no reason, and per-episode values are what the raw
    output should contain anyway.
    """
    keys = ['baseline'] + [f'{m}:{s}' for s, m in arms]
    acc = {k: {'ap': [], 'roc': [], 'tgt': []} for k in keys}
    t0 = time.time()

    for b0 in range(0, len(indices), batch_size):
        sel = indices[b0:b0 + batch_size]
        batch = common.collate(dataset, sel, device)
        labels = batch.pop('labels')['interaction']
        labn = labels.cpu().numpy()
        donor = None
        if any(m == 'swap' for _, m in arms):
            donor = common.collate(dataset, donors[b0:b0 + len(sel)], device)
            donor.pop('labels', None)

        def record(key, logits):
            lg = logits.float().cpu().numpy()
            acc[key]['ap'].append(per_row_ap(lg, labn))
            acc[key]['roc'].append(per_row_roc(lg, labn))
            t = common.compute_targets(logits, labels)
            acc[key]['tgt'].append(np.stack([t[x].cpu().numpy()
                                             for x in common.TARGETS], 1))

        with torch.no_grad():
            base_logits = model(**batch)['interaction']
        record('baseline', base_logits)
        if b0 == 0:
            # The docstring claims this reports the same quantity train.py logs.
            # Check it on real data rather than asserting it: per_episode_ap is
            # input_swap_ablation's implementation, per_row_ap is the per-row form
            # used here, and their means must agree.
            lg = base_logits.float().cpu().numpy()
            mine = float(np.nanmean(per_row_ap(lg, labn)))
            theirs = per_episode_ap(lg, labn)
            if abs(mine - theirs) > 1e-9:
                raise SystemExit(
                    f"per_row_ap disagrees with input_swap_ablation.per_episode_ap "
                    f"({mine:.10f} vs {theirs:.10f}) -- the metric is not the one "
                    "train.py logs")

        for segment, mode in arms:
            kw, use = {}, batch
            if mode in ('swap', 'drop'):
                targets = active if segment == 'all' else [segment]
                for s in targets:
                    use = apply_input_intervention(use, s, mode, donor)
            elif mode == 'mask':
                kw['mask_edit'] = segment_mask_edit(layout, [segment], device)
            elif mode == 'mean':
                kw['substitute'] = (
                    lambda x, b=mean_baseline, s=segment: _mean_sub(x, b, layout, s))
            with common.Capture(model, retain_grad=False, **kw):
                with torch.no_grad():
                    record(f'{mode}:{segment}', model(**use)['interaction'])

        if progress_every and (b0 // batch_size) % progress_every == 0:
            done = b0 + len(sel)
            rate = done / max(time.time() - t0, 1e-9)
            print(f"    {done:6d}/{len(indices)}  {rate:5.1f} ep/s  "
                  f"eta {(len(indices) - done) / max(rate, 1e-9):5.0f}s", flush=True)

    return {k: {'ap': np.concatenate(acc[k]['ap']),
                'roc': np.concatenate(acc[k]['roc']),
                'tgt': np.concatenate(acc[k]['tgt'])} for k in keys}



def _mean_sub(x, baseline, layout, segment):
    """Replace one segment's rows of the encoder input with the segment mean."""
    idx = torch.as_tensor(layout.index_mask(segment), device=x.device)
    out = x.clone()
    out[idx] = baseline.to(x.dtype).expand_as(x)[idx]
    return out



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True, choices=['A', 'B'])
    ap.add_argument('--which', default='both', choices=['best', 'final', 'both'])
    ap.add_argument('--level', default='modality', choices=['modality', 'token'])
    ap.add_argument('--modes', default='swap,drop,mask,mean')
    ap.add_argument('--n_episodes', type=int, default=0,
                    help="0 = whole split at modality level; token level defaults to 128")
    ap.add_argument('--batch_size', type=int, default=16)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out_dir', default='')
    ap.add_argument('--device', default='')
    args = ap.parse_args()

    common.set_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device(args.device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    whiches = ['best', 'final'] if args.which == 'both' else [args.which]
    modes = [m.strip() for m in args.modes.split(',') if m.strip()]
    for m in modes:
        if m not in INTERVENTIONS:
            raise SystemExit(f"unknown intervention {m!r} (expected {INTERVENTIONS})")

    spec = runspec.RUNS[args.run]
    print(f"=== {spec['label']}  test={spec['test_participants']}  device={device}")
    print(f"    level={args.level}  interventions={modes}")

    _, cfg0 = common.load_checkpoint(runspec.checkpoint_dir(args.run, whiches[0]))
    t0 = time.time()
    dataset = common.build_test_dataset(args.run, cfg0)
    print(f"    dataset: {len(dataset)} episodes, built in {time.time() - t0:.1f}s")

    default_n = len(dataset) if args.level == 'modality' else 128
    n = default_n if args.n_episodes <= 0 else min(args.n_episodes, len(dataset))
    indices = list(range(n))
    # Donor for the swap arm: a fixed derangement, so no episode donates to itself
    # and the pairing is identical for every segment and every checkpoint.
    rng = np.random.default_rng(args.seed)
    donors = (np.arange(n) + 1 + rng.integers(0, max(n - 1, 1), size=n)) % n
    donors = np.where(donors == np.arange(n), (donors + 1) % n, donors).tolist()

    for which in whiches:
        tag = runspec.tag(args.run, which)
        print(f"\n--- {tag}  [{n} episodes]")
        model, config = common.load_checkpoint(runspec.checkpoint_dir(args.run, which), device)
        layout = common.read_layout(model, dataset, device)
        active = [m for m in config.input_modality.split(',') if m in SEGMENT_KEYS]
        missing = [m for m in config.input_modality.split(',') if m not in SEGMENT_KEYS]
        if missing:
            raise SystemExit(f"no SEGMENT_KEYS entry for active modality/ies {missing} -- "
                             "they would be silently skipped")
        print(f"    active modalities: {active}")

        rows, per_ep, tok = None, {}, None
        if args.level == 'modality':
            mean_baseline = None
            if 'mean' in modes:
                from m1_gradients import segment_mean_baseline
                mean_baseline = segment_mean_baseline(model, dataset, layout, device)
            arms = [(s, m) for m in modes for s in active]
            arms += [(s, 'mask') for s in ('cls', 'sep') if 'mask' in modes]
            arms += [('all', m) for m in modes if m in ('swap', 'drop')]

            t0 = time.time()
            res = evaluate_arms(model, dataset, indices, device, args.batch_size,
                                layout, arms, donors, active, mean_baseline)
            print(f"    {len(arms) + 1} arms in one pass, "
                  f"{n / max(time.time() - t0, 1e-9):.1f} ep/s")

            def summarise(key):
                r = res[key]
                return {'pr_auc': float(np.nanmean(r['ap'])),
                        'roc_auc': float(np.nanmean(r['roc'])),
                        **{f'target_{t}': float(r['tgt'][:, i].mean())
                           for i, t in enumerate(common.TARGETS)}}

            base = summarise('baseline')
            print(f"    baseline: PR-AUC {base['pr_auc']:.4f}  "
                  f"ROC-AUC {base['roc_auc']:.4f}")
            print(f"    logged PR-AUC for this checkpoint: "
                  f"{spec['eval_pr_auc'][which]:.4f}  "
                  f"(drift {abs(base['pr_auc'] - spec['eval_pr_auc'][which]):.4f})")
            rows = [{'arm': 'baseline', 'segment': '', 'mode': '', **base}]
            per_ep = {'baseline': res['baseline']['tgt']}
            for segment, mode in arms:
                key = f'{mode}:{segment}'
                mm = summarise(key)
                rows.append({'arm': key, 'segment': segment, 'mode': mode, **mm})
                per_ep[key] = res[key]['tgt']
                print(f"    {mode:5s} {segment:11s} PR-AUC {mm['pr_auc']:.4f} "
                      f"({mm['pr_auc'] - base['pr_auc']:+.4f})  "
                      f"ROC {mm['roc_auc']:.4f} ({mm['roc_auc'] - base['roc_auc']:+.4f})")
            per_ep.update({f'ap__{k}': v['ap'] for k, v in res.items()})
            per_ep.update({f'roc__{k}': v['roc'] for k, v in res.items()})
        else:
            base_logits = base_labels = None
            base = {}
            tok = _token_level(model, dataset, indices, device, args.batch_size,
                               layout, base_logits, base_labels)

        sub = args.level
        out_dir = args.out_dir or os.path.join(common.OUT_BASE, 'm2', tag, sub)
        os.makedirs(out_dir, exist_ok=True)
        if rows is not None:
            np.savez_compressed(os.path.join(out_dir, 'arms.npz'),
                                arms=np.array([r['arm'] for r in rows]),
                                targets=np.array(common.TARGETS),
                                **{f'per_episode__{k}': v for k, v in per_ep.items()})
            with open(os.path.join(out_dir, 'arms.json'), 'w') as f:
                json.dump(rows, f, indent=2)
        else:
            np.savez_compressed(os.path.join(out_dir, 'tokens.npz'), **tok)

        common.write_json(os.path.join(out_dir, 'manifest.json'), {
            'run': args.run, 'which': which, 'tag': tag, 'level': args.level,
            'checkpoint': runspec.checkpoint_dir(args.run, which),
            'n_episodes': n, 'interventions': modes, 'seed': args.seed,
            'baseline': base, 'logged_pr_auc': spec['eval_pr_auc'][which],
            'arms': [f'{m}:{s_}' for s_, m in arms] if args.level == 'modality' else [],
            'layout': layout.as_dict(),
            'dataset_kwargs': runspec.DATASET_KWARGS,
            'environment': common.environment_manifest(),
        })
        print(f"    wrote {out_dir}")
    print("\ndone")



def _token_level(model, dataset, indices, device, batch_size, layout,
                 base_logits, base_labels):
    """Mask ONE token at a time, per episode, and record the change.

    Only VALID tokens are ablated -- a padded slot is already absent, so masking
    it is a no-op by construction and would dilute every summary with zeros.
    Episodes are processed one at a time, with the token variants batched, since
    each episode has its own set of valid positions.
    """
    L = layout.seq_len
    delta_pr = np.full((len(indices), L), np.nan, dtype=np.float32)
    delta_tgt = np.full((len(indices), L), np.nan, dtype=np.float32)
    valid_all = np.zeros((len(indices), L), dtype=bool)
    t0 = time.time()
    for e, ep in enumerate(indices):
        batch1 = common.collate(dataset, [ep], device)
        labels = batch1.pop('labels')['interaction']
        with common.Capture(model, retain_grad=False) as cap:
            with torch.no_grad():
                out = model(**batch1)
        valid = common.valid_matrix(cap.pad_mask)[0]
        valid_all[e] = valid
        base_t = common.compute_targets(out['interaction'], labels)[
            common.PRIMARY_TARGET].item()
        base_ap = float(per_row_ap(out['interaction'].float().cpu().numpy(),
                                   labels.cpu().numpy())[0])
        positions = np.nonzero(valid)[0]
        for c0 in range(0, len(positions), batch_size):
            chunk = positions[c0:c0 + batch_size]
            reps = len(chunk)
            batch = common.collate(dataset, [ep] * reps, device)
            lab = batch.pop('labels')['interaction']
            extra = torch.zeros(reps, L, dtype=torch.bool, device=device)
            for r, p in enumerate(chunk):
                extra[r, int(p)] = True
            with common.Capture(model, retain_grad=False,
                                mask_edit=lambda m, x=extra: m | x):
                with torch.no_grad():
                    o = model(**batch)
            lg = o['interaction']
            t = common.compute_targets(lg, lab)[common.PRIMARY_TARGET].cpu().numpy()
            aps = per_row_ap(lg.float().cpu().numpy(), lab.cpu().numpy())
            for r, p in enumerate(chunk):
                delta_tgt[e, int(p)] = t[r] - base_t
                delta_pr[e, int(p)] = aps[r] - base_ap
        if (e + 1) % 10 == 0:
            print(f"    token level {e + 1}/{len(indices)}  "
                  f"{(e + 1) / max(time.time() - t0, 1e-9):.2f} ep/s", flush=True)
    return {'episode': np.asarray(indices), 'valid': valid_all,
            'delta_pr_auc': delta_pr, 'delta_target': delta_tgt,
            'segments': np.array(layout.names)}


if __name__ == '__main__':
    main()
