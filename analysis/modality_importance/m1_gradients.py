"""Method 1 -- gradient-based token and modality importance.

Computes, for every episode of a run's held-out test split, the gradient of a
scalar target with respect to the token embeddings entering the transformer
encoder, and reduces it to per-token and per-modality scores.

    # smoke (32 episodes, both checkpoints of run A, all sanity checks)
    python analysis/modality_importance/m1_gradients.py --run A --n_episodes 32

    # full split
    python analysis/modality_importance/m1_gradients.py --run A

    # controls
    python analysis/modality_importance/m1_gradients.py --run A --control weights
    python analysis/modality_importance/m1_gradients.py --run A --control labels

    # integrated gradients (separate, primary target only)
    python analysis/modality_importance/m1_gradients.py --run A --ig --ig_episodes 128

What this measures, stated plainly: how much a small change to one token's
embedding would move the target, and (grad x input) the first-order estimate of
how much the target would fall if that token were zeroed. It is a LOCAL, linear
description of a non-linear model. It is not an ablation -- nothing is removed --
and it says nothing about what would happen under a large intervention. Method 2
answers that question and is expected to disagree in places; the disagreement is
information, not an error.

Deliberately NOT done here: parameter-gradient magnitudes. A large gradient on
visual_mapper.weight says the visual projection is far from optimal, not that
video matters to the prediction; those are different quantities and conflating
them is the standard way this analysis goes wrong.
"""
import argparse
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402
import runs as runspec  # noqa: E402


# ---------------------------------------------------------------------------

def batched_gradients(model, dataset, layout, indices, batch_size, device,
                      label_permutation=None, saved_preds=None, n_repro=64,
                      progress_every=20):
    """One forward + one backward per target, per batch.

    Samples never interact inside the model (attention is within-sample,
    LayerNorm is per position), so a single backward on ``target.sum()`` gives the
    correct per-sample gradient in each batch column. That is what makes batching
    exact here rather than an approximation, and it is asserted by the
    --check_batching flag.
    """
    n = len(indices)
    L = layout.seq_len
    S = len(layout.names)
    T = common.TARGETS

    store = {
        'episode': np.asarray(indices, dtype=np.int64),
        'valid': np.zeros((n, L), dtype=bool),
        'x_norm': np.zeros((n, L), dtype=np.float32),
        'target_value': np.zeros((n, len(T)), dtype=np.float32),
        'n_valid': np.zeros((n, S), dtype=np.int32),
    }
    for t in T:
        for s in common.SCORES:
            store[f'total__{t}__{s}'] = np.zeros((n, S), dtype=np.float64)
        # Exact additive split of the SIGNED gxi score into the named parts each
        # token is built from. Answers how much of a modality's contribution flows
        # through parameters it SHARES with the others (pos_code, class) rather
        # than through anything specific to it.
        for c in common.CHANNELS:
            store[f'chan__{t}__{c}'] = np.zeros((n, S), dtype=np.float64)
    # Token-level detail is kept for the PRIMARY target only: [N, 592] per score
    # is 11.9 MB per array at N=5007, and four targets of it would be 140 MB of
    # mostly redundant numbers. The other targets survive at modality level.
    for s in common.SCORES:
        store[f'tokens__{s}'] = np.zeros((n, L), dtype=np.float32)

    diagnostics = {'pad_grad_max': 0.0, 'nonfinite_batches': 0, 'repro_max_abs': None,
                   'component_residual_max': 0.0, 'channel_sum_error_max': 0.0}
    repro_deltas = []
    t0 = time.time()

    for b0 in range(0, n, batch_size):
        sel = indices[b0:b0 + batch_size]
        batch = common.collate(dataset, sel, device)
        labels = batch.pop('labels')['interaction']
        if label_permutation is not None:
            # Control: attribute towards ANOTHER episode's ground truth. The
            # attribution pattern must change; if it does not, the scores are not
            # reading the target at all.
            perm_sel = label_permutation[b0:b0 + batch_size]
            perm_batch = common.collate(dataset, perm_sel, device)
            labels = perm_batch['labels']['interaction']

        with common.Capture(model) as cap:
            out = model(**batch)
            logits = out['interaction']
            x = cap.encoder_input
            valid = common.valid_matrix(cap.pad_mask)          # [B, L]
            targets = common.compute_targets(logits, labels)

            x_det = x.detach()
            store['x_norm'][b0:b0 + len(sel)] = x_det.norm(dim=-1).T.cpu().numpy()
            store['valid'][b0:b0 + len(sel)] = valid

            # The parts each token is summed from. Verified against the captured
            # encoder input every batch, so a change to model.forward's assembly
            # fails loudly here instead of quietly redistributing the channels.
            with torch.no_grad():
                comps = common.token_components(model, batch, layout, x_det)
                diagnostics['component_residual_max'] = max(
                    diagnostics['component_residual_max'],
                    common.verify_components(comps, x_det))

            for k, tname in enumerate(T):
                grad, = torch.autograd.grad(targets[tname].sum(), x,
                                            retain_graph=(k < len(T) - 1))
                if not torch.isfinite(grad).all():
                    diagnostics['nonfinite_batches'] += 1
                sc = common.token_scores(grad, x_det)

                # Sanity check 2: a padded position is masked as a KEY everywhere
                # it is read (encoder self-attention and decoder cross-attention
                # share the one mask), so it cannot influence any output and its
                # gradient must be exactly zero. Not an assumption -- measured.
                pad = ~valid
                if pad.any():
                    diagnostics['pad_grad_max'] = max(
                        diagnostics['pad_grad_max'],
                        float(np.abs(sc['grad_l2'][pad]).max()))

                for s in common.SCORES:
                    tot, nval = common.aggregate(sc[s], valid, layout)
                    store[f'total__{tname}__{s}'][b0:b0 + len(sel)] = tot
                    if k == 0 and s == common.SCORES[0]:
                        store['n_valid'][b0:b0 + len(sel)] = nval
                    if tname == common.PRIMARY_TARGET:
                        store[f'tokens__{s}'][b0:b0 + len(sel)] = sc[s]

                # <grad, part> per channel, aggregated by segment. Because the
                # token is an exact sum of the parts and the dot product is
                # bilinear, these must sum back to the segment's gxi total --
                # checked below rather than trusted.
                chan_sum = np.zeros_like(store[f'total__{tname}__gxi'][b0:b0 + len(sel)])
                for c in common.CHANNELS:
                    per_tok = (grad * comps[c]).sum(dim=-1).T.detach().cpu().numpy()
                    ctot, _ = common.aggregate(per_tok, valid, layout)
                    store[f'chan__{tname}__{c}'][b0:b0 + len(sel)] = ctot
                    chan_sum += ctot
                diagnostics['channel_sum_error_max'] = max(
                    diagnostics['channel_sum_error_max'],
                    float(np.abs(chan_sum - store[f'total__{tname}__gxi'][b0:b0 + len(sel)]).max()))

            store['target_value'][b0:b0 + len(sel)] = np.stack(
                [targets[t].detach().cpu().numpy() for t in T], axis=1)

        # Sanity check 1: reproduce the logits CustomTrainer saved for this
        # checkpoint. Reported as a magnitude, NOT asserted bit-identical --
        # float32 reductions differ between devices and BLAS builds, and chasing
        # that has cost this project time before.
        if saved_preds is not None and b0 < n_repro:
            got = logits.detach().float().cpu().numpy()
            want = saved_preds[np.asarray(sel)]
            repro_deltas.append(np.abs(got - want).max())

        if progress_every and (b0 // batch_size) % progress_every == 0:
            done = b0 + len(sel)
            rate = done / max(time.time() - t0, 1e-9)
            print(f"    {done:6d}/{n}  {rate:6.1f} ep/s  "
                  f"eta {(n - done) / max(rate, 1e-9):6.0f}s", flush=True)

    if repro_deltas:
        diagnostics['repro_max_abs'] = float(np.max(repro_deltas))
    diagnostics['seconds'] = time.time() - t0
    diagnostics['episodes_per_second'] = n / max(diagnostics['seconds'], 1e-9)
    return store, diagnostics


def check_batching(model, dataset, layout, device, indices=(0, 1, 2, 3)):
    """Assert that a batched backward equals per-sample backwards.

    The whole run depends on samples being independent inside the model. If a
    future change introduced any cross-sample term (a batch norm, a pooled
    statistic) every batched number here would be silently wrong, so this is
    checked rather than assumed.
    """
    def one(sel):
        batch = common.collate(dataset, sel, device)
        labels = batch.pop('labels')['interaction']
        with common.Capture(model) as cap:
            out = model(**batch)
            t = common.compute_targets(out['interaction'], labels)[common.PRIMARY_TARGET]
            g, = torch.autograd.grad(t.sum(), cap.encoder_input)
        return g.detach()

    batched = one(list(indices))
    worst = 0.0
    for i, ep in enumerate(indices):
        single = one([ep])[:, 0]
        worst = max(worst, float((batched[:, i] - single).abs().max()))
    return worst


def segment_mean_baseline(model, dataset, layout, device, n_episodes=64, batch_size=8):
    """[L, 1, H] -- each position replaced by its SEGMENT's mean valid token.

    The alternative to the all-zero baseline. Zeroing the whole sequence scales
    the attention logits by roughly alpha^2 along the path (queries and keys
    shrink together), so the straight line from 0 passes through a regime where
    attention is near-uniform and nothing the model saw in training lives. A
    segment-mean baseline stays inside the data distribution: it means "a typical
    token of this kind" rather than "no vector at all", and the attribution then
    answers "what does THIS episode's content add over a typical one".
    """
    acc = torch.zeros(layout.seq_len, model.config.transformer_hidden_size, device=device)
    cnt = torch.zeros(layout.seq_len, 1, device=device)
    n = min(n_episodes, len(dataset))
    for b0 in range(0, n, batch_size):
        batch = common.collate(dataset, list(range(b0, min(b0 + batch_size, n))), device)
        batch.pop('labels', None)
        with common.Capture(model, retain_grad=False) as cap:
            with torch.no_grad():
                model(**batch)
        x = cap.encoder_input.detach()                       # [L, B, H]
        keep = (~cap.pad_mask).T.unsqueeze(-1).float()       # [L, B, 1]
        acc += (x * keep).sum(dim=1)
        cnt += keep.sum(dim=1)
    per_pos = acc / cnt.clamp(min=1)                         # [L, H]
    # Average within each segment so the baseline is "a typical obb token", not
    # "the typical token at slot 137" -- the latter would bake in the confidence
    # ordering of the obb segment and the time structure of video/location.
    out = torch.zeros_like(per_pos)
    for name in layout.names:
        idx = torch.as_tensor(layout.index_mask(name), device=device)
        w = cnt.squeeze(-1)[idx]
        if w.sum() > 0:
            out[idx] = (per_pos[idx] * w.unsqueeze(-1)).sum(0) / w.sum()
    return out.unsqueeze(1)                                  # [L, 1, H]


def integrated_gradients(model, dataset, layout, indices, steps, batch_size, device,
                         baseline=None):
    """IG at the encoder input, midpoint Riemann sum along a straight path.

    ``baseline`` is None (all-zero token embeddings) or a [L, 1, H] tensor, e.g.
    segment_mean_baseline. The zero baseline is the standard choice and makes the
    completeness identity clean, but a zeroed token is not something the model
    ever saw in training -- so it answers "what does the model do as this input
    fades in from nothing", which is a different question from "what if this token
    were a plausible alternative". Method 2 takes up the second question, where
    the choice of baseline is the whole crux.

    Completeness (an arithmetic identity, not a modelling assumption):
        sum over all tokens and dims of IG  ==  T(input) - T(baseline)
    The gap is reported; a large one means too few steps or a bug, and
    m1_ig_diagnose.py is what tells the two apart.
    """
    n = len(indices)
    L = layout.seq_len
    out = {
        'episode': np.asarray(indices, dtype=np.int64),
        'ig_token': np.zeros((n, L), dtype=np.float32),     # summed over the 768 dims
        'valid': np.zeros((n, L), dtype=bool),
        'completeness_sum': np.zeros(n, dtype=np.float64),
        'completeness_delta': np.zeros(n, dtype=np.float64),
    }
    alphas = (np.arange(steps) + 0.5) / steps      # midpoint rule
    t0 = time.time()

    for b0 in range(0, n, batch_size):
        sel = indices[b0:b0 + batch_size]
        batch = common.collate(dataset, sel, device)
        labels = batch.pop('labels')['interaction']

        # The real encoder input, and the target at both ends of the path.
        with common.Capture(model, retain_grad=False) as cap:
            with torch.no_grad():
                o = model(**batch)
            x_real = cap.encoder_input.detach()
            valid = common.valid_matrix(cap.pad_mask)
            t_one = common.compute_targets(o['interaction'], labels)[common.PRIMARY_TARGET]

        base = (torch.zeros_like(x_real) if baseline is None
                else baseline.to(x_real.dtype).expand_as(x_real).contiguous())
        with common.Capture(model, substitute=lambda _x, b=base: b, retain_grad=False) as cap:
            with torch.no_grad():
                o0 = model(**batch)
            t_zero = common.compute_targets(o0['interaction'], labels)[common.PRIMARY_TARGET]

        delta = x_real - base
        acc = torch.zeros_like(x_real)
        for a in alphas:
            point = (base + float(a) * delta).clone().requires_grad_(True)
            with common.Capture(model, substitute=lambda _x, p=point: p) as cap:
                o = model(**batch)
                t = common.compute_targets(o['interaction'], labels)[common.PRIMARY_TARGET]
                g, = torch.autograd.grad(t.sum(), point)
            acc += g.detach()

        ig = delta * acc / steps                               # [L, B, H]
        per_token = ig.sum(dim=-1).T.cpu().numpy()             # [B, L]
        out['ig_token'][b0:b0 + len(sel)] = per_token
        out['valid'][b0:b0 + len(sel)] = valid
        out['completeness_sum'][b0:b0 + len(sel)] = per_token.sum(axis=1)
        out['completeness_delta'][b0:b0 + len(sel)] = (
            (t_one - t_zero).detach().cpu().numpy())
        done = b0 + len(sel)
        print(f"    IG {done:5d}/{n}  {done / max(time.time() - t0, 1e-9):5.2f} ep/s", flush=True)

    gap = np.abs(out['completeness_sum'] - out['completeness_delta'])
    denom = np.maximum(np.abs(out['completeness_delta']), 1e-9)
    out['completeness_rel_error'] = gap / denom
    return out


def _already_done(out_dir, n_episodes, control):
    """True if this pass already produced results covering n_episodes.

    Exists so a job killed partway can be resumed without redoing the passes that
    succeeded -- which matters here because the cluster's inherited 600s CPU-time
    limit killed the first attempt after two of four main passes had completed.
    """
    man = os.path.join(out_dir, 'manifest.json')
    if not (os.path.exists(man) and os.path.exists(os.path.join(out_dir, 'scores.npz'))):
        return False
    import json
    try:
        with open(man) as f:
            d = json.load(f)
    except Exception:
        return False
    return d.get('n_episodes', 0) >= n_episodes and d.get('control') == control


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True, choices=['A', 'B'])
    ap.add_argument('--which', default='both', choices=['best', 'final', 'both'])
    ap.add_argument('--n_episodes', type=int, default=0,
                    help="0 = the whole held-out test split (5007 for A, 3311 for B)")
    ap.add_argument('--batch_size', type=int, default=8)
    ap.add_argument('--control', default='none',
                    help="comma-separated subset of none,weights,labels -- all run against "
                         "ONE dataset build, which is 5 minutes each time it is repeated. "
                         "weights: attribute an UNTRAINED model built from the same config. "
                         "labels: attribute towards another episode's ground truth. Both must "
                         "change the modality ranking substantially or the scores are not "
                         "measuring what they claim to.")
    ap.add_argument('--control_n', type=int, default=0,
                    help="episodes for the CONTROL passes (0 = same as --n_episodes). The "
                         "controls only have to show the ranking collapses, which does not "
                         "need the full split.")
    ap.add_argument('--ig', action='store_true', help="also run Integrated Gradients")
    ap.add_argument('--ig_episodes', type=int, default=128)
    ap.add_argument('--ig_steps', type=int, default=32)
    ap.add_argument('--ig_baseline', default='segment_mean',
                    choices=['zero', 'segment_mean'],
                    help="segment_mean (default) replaces each position with its "
                         "SEGMENT's mean valid token. Measured completeness error at 32 "
                         "steps: 0.0001 against the zero baseline's 0.797 -- the zero "
                         "baseline needs 512 steps to reach 0.016. See m1_ig_diagnose.py.")
    ap.add_argument('--precision_check', type=int, default=64,
                    help="episodes for the fp32-vs-bf16 reconciliation against "
                         "predictions.pkl (0 = skip). Reuses the dataset already built.")
    ap.add_argument('--n_repro', type=int, default=64,
                    help="episodes compared against the checkpoint's saved predictions.pkl")
    ap.add_argument('--skip_existing', action='store_true',
                    help="skip a (checkpoint, control) pass whose scores.npz already "
                         "covers the requested episode count -- for resuming a job that "
                         "died partway without recomputing what already succeeded")
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out_dir', default='')
    ap.add_argument('--device', default='')
    args = ap.parse_args()

    common.set_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device(args.device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    whiches = ['best', 'final'] if args.which == 'both' else [args.which]

    controls = [c.strip() for c in args.control.split(',') if c.strip()]
    for c in controls:
        if c not in ('none', 'weights', 'labels'):
            raise SystemExit(f"unknown control {c!r} (expected none, weights or labels)")

    spec = runspec.RUNS[args.run]
    print(f"=== {spec['label']}  test={spec['test_participants']}  device={device}")
    print(f"    checkpoints: {[runspec.tag(args.run, w) for w in whiches]}")
    print(f"    controls={controls}  seed={args.seed}")

    # The dataset is built ONCE and shared by both checkpoints of the run: same
    # split, same preprocessing, same episode order for every set of weights,
    # which is what makes best-vs-final a one-variable comparison.
    _, cfg0 = common.load_checkpoint(runspec.checkpoint_dir(args.run, whiches[0]))
    t0 = time.time()
    dataset = common.build_test_dataset(args.run, cfg0)
    print(f"    dataset: {len(dataset)} episodes, built in {time.time() - t0:.1f}s")

    # Precision reconciliation, on the dataset just built rather than in a separate
    # process that would spend another ~5 minutes rebuilding it. The runs evaluated
    # under bf16 autocast; this analysis runs in float32 (the dtype the weights are
    # stored in), and this measures what that costs.
    if args.precision_check > 0:
        from check_precision import logits_for
        w0 = whiches[0]
        mdl, _ = common.load_checkpoint(runspec.checkpoint_dir(args.run, w0), device)
        pidx = list(range(min(args.precision_check, len(dataset))))
        ref = common.saved_predictions(args.run, w0)[pidx]
        print(f"\n--- precision reconciliation ({len(pidx)} episodes, {runspec.tag(args.run, w0)})")
        prec = {'n_episodes': len(pidx), 'mean_abs_logit': float(np.abs(ref).mean())}
        for name, dt in (('float32', None), ('bfloat16', torch.bfloat16)):
            try:
                got = logits_for(mdl, dataset, pidx, device, dt)
            except RuntimeError as e:
                print(f"    {name:9s} unavailable on {device}: {str(e)[:90]}")
                continue
            d = np.abs(got - ref)
            from scipy.stats import spearmanr
            rho = float(np.mean([spearmanr(got[i], ref[i]).statistic for i in range(len(pidx))]))
            prec[name] = {'max_abs': float(d.max()), 'mean_abs': float(d.mean()),
                          'relative': float(d.mean() / np.abs(ref).mean()),
                          'spearman_voxel_order': rho}
            print(f"    {name:9s} vs predictions.pkl: max|d| {d.max():.5f}  "
                  f"mean|d| {d.mean():.5f} ({d.mean() / np.abs(ref).mean():.3%})  "
                  f"voxel-order Spearman {rho:.6f}")
        common.write_json(os.path.join(common.OUT_BASE, 'm1', f'precision_{args.run}.json'),
                          {'run': args.run, 'which': w0, **prec,
                           'environment': common.environment_manifest()})
        del mdl

    n_main = len(dataset) if args.n_episodes <= 0 else min(args.n_episodes, len(dataset))
    n_ctrl = min(args.control_n, n_main) if args.control_n > 0 else n_main

    for control in controls:
      n = n_main if control == 'none' else n_ctrl
      indices = list(range(n))
      label_permutation = None
      if control == 'labels':
        rng = np.random.default_rng(args.seed)
        label_permutation = list(rng.permutation(len(dataset))[:n])

      for which in whiches:
        tag = runspec.tag(args.run, which)
        sub_dir = os.path.join(common.OUT_BASE, 'm1', tag,
                               control if control != 'none' else 'main')
        if args.skip_existing and _already_done(sub_dir, n, control):
            print(f"\n--- {tag}  [control={control}] SKIPPED: "
                  f"{sub_dir}/scores.npz already covers {n} episodes")
            continue
        print(f"\n--- {tag}  [control={control}, {n} episodes]")
        path = runspec.checkpoint_dir(args.run, which)
        model, config = common.load_checkpoint(path, device)
        if control == 'weights':
            from model import CustomTransformerModel
            common.set_seed(args.seed + 1)
            model = CustomTransformerModel(config).to(device).eval()
            print("    CONTROL: untrained model built from the same config "
                  "(no voxel-bias prior init applied)")
        assert model.training is False, "model must be in eval mode"

        layout = common.read_layout(model, dataset, device)
        print(f"    L={layout.seq_len}  segments={ {k: layout.n_slots(k) for k in layout.names} }")

        batch_err = check_batching(model, dataset, layout, device)
        print(f"    batched-vs-single backward max|delta| = {batch_err:.3g}")

        saved = None if control != 'none' else common.saved_predictions(args.run, which)
        store, diag = batched_gradients(
            model, dataset, layout, indices, args.batch_size, device,
            label_permutation=label_permutation, saved_preds=saved, n_repro=args.n_repro)
        diag['batched_vs_single_max_abs'] = batch_err

        sub = control if control != 'none' else 'main'
        out_dir = args.out_dir or os.path.join(common.OUT_BASE, 'm1', tag, sub)
        os.makedirs(out_dir, exist_ok=True)
        np.savez_compressed(os.path.join(out_dir, 'scores.npz'),
                            segments=np.array(layout.names),
                            targets=np.array(common.TARGETS),
                            score_names=np.array(common.SCORES),
                            channels=np.array(common.CHANNELS),
                            shared_channels=np.array(common.SHARED_CHANNELS),
                            **store)

        manifest = {
            'run': args.run, 'which': which, 'tag': tag, 'checkpoint': path,
            'control': control, 'seed': args.seed,
            'test_participants': spec['test_participants'],
            'n_episodes': n, 'n_episodes_available': len(dataset),
            'batch_size': args.batch_size,
            'eval_pr_auc_logged': spec['eval_pr_auc'][which],
            'layout': layout.as_dict(),
            'targets': {t: common.TARGET_DOC[t] for t in common.TARGETS},
            'primary_target': common.PRIMARY_TARGET,
            'scores': common.SCORE_DOC,
            'dataset_kwargs': runspec.DATASET_KWARGS,
            'model_config': config.to_dict(),
            'diagnostics': diag,
            'environment': common.environment_manifest(),
        }
        common.write_json(os.path.join(out_dir, 'manifest.json'), manifest)

        print(f"    pad-token max |grad_l2|        = {diag['pad_grad_max']:.3g}   (must be 0)")
        print(f"    channel reconstruction resid   = {diag['component_residual_max']:.3g}")
        print(f"    channel sum vs gxi total       = {diag['channel_sum_error_max']:.3g}")
        if diag['repro_max_abs'] is not None:
            print(f"    predictions.pkl max|delta|     = {diag['repro_max_abs']:.3g}")
        print(f"    {diag['episodes_per_second']:.1f} ep/s, {diag['seconds']:.0f}s total")
        print(f"    wrote {out_dir}/scores.npz")

        if args.ig and control == 'none':
            ig_n = min(args.ig_episodes, n)
            print(f"    integrated gradients: {ig_n} episodes x {args.ig_steps} steps")
            ig_base = (segment_mean_baseline(model, dataset, layout, device)
                       if args.ig_baseline == 'segment_mean' else None)
            ig = integrated_gradients(model, dataset, layout, list(range(ig_n)),
                                      args.ig_steps, max(1, args.batch_size // 2), device,
                                      baseline=ig_base)
            np.savez_compressed(os.path.join(out_dir, 'ig.npz'),
                                segments=np.array(layout.names), steps=args.ig_steps,
                                baseline=np.array(args.ig_baseline), **ig)
            rel = ig['completeness_rel_error']
            print(f"    completeness relative error: median {np.median(rel):.4f}  "
                  f"p90 {np.percentile(rel, 90):.4f}  max {rel.max():.4f}")
            print(f"    wrote {out_dir}/ig.npz")

    print("\ndone")


if __name__ == '__main__':
    main()
