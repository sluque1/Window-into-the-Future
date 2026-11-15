"""Where does the static per-voxel map live -- the 48-parameter bias, or the evidence?

THE QUESTION. input_swap_ablation.py measured static-self at 108.5% of base on
P08: one averaged prediction handed to every episode outscores the model reading
the real episode. Something in this model is a per-kitchen map that PR-AUC mostly
rewards. There are exactly two places it can be, and decoder.forward makes the
split ADDITIVE and therefore exactly decomposable:

    logits[e, v] = head(ln_o(h[e, v]))   +   bias()[v]
                   \\_______evidence_____/       \\__48 params__/

  * bias() is `bias_x[i] + bias_y[j] + bias_z[k]` (decoder.py, --voxel_bias
    factorised), 48 numbers, identical for every episode, added unconditionally in
    EVERY residual mode -- `evidence` included. That last point is why the
    2026-08-19 resevidence runs do NOT rule the bias out: they delete q from h and
    leave this term untouched, so a static map survives them by construction.
  * the head term is NOT "the evidence", and calling it that is a trap this script
    fell into once. Under the default --decoder_residual raw, h = q + attn_out, so
    the head sees the QUERY as well -- and |q| ~ 48 against |attn_out| ~ 1.11, so it
    mostly sees the query. q is a pure function of the voxel index (4096 distinct
    static vectors, far more expressive than the 48-param bias), while attn_out is
    the only genuinely episode-dependent thing in the sum.

THREE CANDIDATES, THEN, NOT TWO -- and the third is the one that needs a separate
measurement, because the additive split cannot separate q from attn_out (they are
summed INSIDE a LayerNorm, so head(ln_o(q + a)) != head(ln_o(q)) + head(ln_o(a))):

    bias()                 48 params, static           -> scored below
    head(ln_o(q))          4096 values, static         -> scored below, weights only
    the rest               episode-dependent           -> what is left over

`decoder.queries()` builds q from weights alone, so the query-only map is closed
form: no dataset, no encoder, no forward pass over episodes. If its AP lands near
static-self, the static map is voxel IDENTITY and the encoder is decoration. If it
lands far below while the mean map stays high, the static part really is coming
through attention -- which is possible even though attn_out varies per episode,
since the kitchen furniture `obb` describes is largely shared across a
participant's windows. The obb swap arm bounds how much of it is NOT shared.

Because the two are summed, mean_e(logits) = bias + mean_e(evidence) exactly, with
no fitting and no approximation. This script scores each side on its own.

WHY AP AND NOT R-SQUARED IS THE ANSWER. Average precision depends only on the
RANKING of a map, so a component can carry most of the variance and still rank
badly, or carry little and rank well. Variance shares are reported because they
are informative, but the number that decides the question is the AP each map
scores on its own, against the same labels, through the same per_episode_ap the
training loop logs as pr_auc -- imported from input_swap_ablation rather than
reimplemented, so the tie handling that matters for a shared map cannot drift.

WHAT THE ANSWERS MEAN.
  bias alone ~= static-self   -> the result is 48 parameters and a prevalence
                                 prior. Encoder, modalities and decoder
                                 composition are all beside the point.
  evidence-mean ~= static-self, bias alone much lower
                              -> the map is retrieved from obb, and the 48
                                 parameters are not the story.
  neither alone reaches it    -> the two are cancelling structure in each other;
                                 read the covariance line.

No forward pass, no dataset, no GPU: predictions.pkl / labels.pkl are already
written next to every checkpoint by trainer.CustomTrainer, and the bias is read
straight out of model.safetensors. Runs in seconds.

Usage:
    python analysis/static_map_decomposition.py --checkpoint <dir>/checkpoint-NNNN
    python analysis/static_map_decomposition.py --run <snapshot_dir>          # last ckpt
"""
import argparse
import glob
import json
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from input_swap_ablation import per_episode_ap, snapshot_root  # noqa: E402

GRID = 16
NUM_VOXELS = GRID ** 3


def load_pkl(path, width=NUM_VOXELS):
    """Load one of the trainer's pickles as an [N, width] array.

    trainer.py documents these as "predictions.pkl -- np.ndarray or tuple of
    np.ndarrays" and "labels.pkl -- dict with 'interaction'", so neither is
    reliably a bare array. Unwrap by SHAPE rather than by position: a tuple's
    first element is not guaranteed to be the voxel logits, and picking the wrong
    one would still produce a full-looking table.
    """
    with open(path, 'rb') as f:
        obj = pickle.load(f)
    if isinstance(obj, dict):
        obj = obj['interaction'] if 'interaction' in obj else next(iter(obj.values()))
    if isinstance(obj, (tuple, list)):
        cands = [a for a in obj if getattr(a, 'ndim', 0) == 2
                 and a.shape[1] == width]
        if len(cands) != 1:
            raise SystemExit(f"{path}: expected exactly one [N, {width}] array in a "
                             f"{type(obj).__name__} of {len(obj)}, found {len(cands)}")
        obj = cands[0]
    arr = np.asarray(obj)
    if arr.ndim != 2 or arr.shape[1] != width:
        raise SystemExit(f"{path}: got shape {arr.shape}, expected [N, {width}]")
    return arr


def axis_fits(m):
    """The factorised fit, and each of its three axes on its own.

    WHY THE PER-AXIS SPLIT MATTERS MORE THAN THE TOTAL. The grids are wall-fitted
    per kitchen (CLAUDE.md: nine manual yaw angles, one per participant), so height
    is the only axis whose meaning is shared across folds -- 'interactions happen at
    counter level' transfers, 'interactions happen at x=11' is one kitchen's floor
    plan. A factorised bias LEARNED on the training participants can only inherit
    the transferable part, so the honest ceiling for --voxel_bias_lr is what `z`
    alone scores here, not what all three axes score fitted to this fold's own map.
    Each axis map is that axis's marginal alone, with the other two flat.

    The fit is closed form: the three axis means, each less 2/3 of the grand mean
    so the grand mean is counted once rather than three times -- the same
    correction decoder.set_voxel_bias applies when it seeds the bias from a target
    field, so 'all three' is exactly the map that flag would install.
    """
    g = m.reshape(GRID, GRID, GRID)
    grand = g.mean()
    mx, my, mz = g.mean((1, 2)), g.mean((0, 2)), g.mean((0, 1))
    ax, ay, az = mx - 2 / 3 * grand, my - 2 / 3 * grand, mz - 2 / 3 * grand
    o = np.ones(GRID)
    return {
        'all three': (ax[:, None, None] + ay[None, :, None] + az[None, None, :]).reshape(-1),
        'x only': (ax[:, None, None] * o[None, :, None] * o[None, None, :]).reshape(-1),
        'y only': (o[:, None, None] * ay[None, :, None] * o[None, None, :]).reshape(-1),
        'z only (height)': (o[:, None, None] * o[None, :, None] * az[None, None, :]).reshape(-1),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', default='')
    ap.add_argument('--run', default='', help="Snapshot dir; uses its last checkpoint.")
    args = ap.parse_args()

    ck = args.checkpoint
    if not ck:
        if not args.run:
            raise SystemExit("pass --checkpoint or --run")
        cks = glob.glob(f"{os.path.abspath(args.run)}/results/*/checkpoint-*")
        if not cks:
            raise SystemExit(f"no checkpoints under {args.run}/results/")
        ck = max(cks, key=lambda c: int(c.rsplit('-', 1)[1]))
    print(f"checkpoint: {ck}\n")

    preds = load_pkl(f"{ck}/predictions.pkl").astype(np.float64)
    labels = load_pkl(f"{ck}/labels.pkl").astype(np.float64)
    if preds.shape[0] != labels.shape[0]:
        raise SystemExit(f"predictions has {preds.shape[0]} episodes, labels "
                         f"{labels.shape[0]} -- these are not the same eval pass")
    print(f"predictions {preds.shape}  range [{preds.min():.3f}, {preds.max():.3f}] "
          f"mean {preds.mean():.3f}")
    # The additive split below is only valid on LOGITS. Sigmoid outputs live in
    # [0,1] and would make bias + evidence a nonsense decomposition, so refuse
    # rather than print a plausible-looking table.
    if preds.min() >= 0.0 and preds.max() <= 1.0:
        raise SystemExit("predictions look like probabilities, not logits -- the "
                         "additive bias/evidence split does not hold; aborting")
    print(f"labels      {labels.shape}  positives/episode {labels.sum(1).mean():.1f}\n")

    # --- the trained bias, rebuilt exactly as decoder.bias() does ----------------
    from safetensors.numpy import load_file
    w = load_file(f"{ck}/model.safetensors")
    kx = [k for k in w if k.endswith('bias_x')]
    if kx:
        pre = kx[0][:-len('bias_x')]
        bx, by, bz = w[pre + 'bias_x'], w[pre + 'bias_y'], w[pre + 'bias_z']
        # voxel_ijk0 = unravel_index(arange(4096), (16,16,16)) -- decoder.py's own
        # construction, the label grid's C order (flat = i*256 + j*16 + k).
        i, j, k = np.unravel_index(np.arange(NUM_VOXELS), (GRID, GRID, GRID))
        bias = (bx[i] + by[j] + bz[k]).astype(np.float64)
        print(f"bias: factorised, 48 params ({pre}bias_x/y/z)")
    else:
        kf = [k for k in w if k.endswith('voxel_bias')]
        if not kf:
            raise SystemExit("no bias_x/y/z and no voxel_bias in the checkpoint")
        bias = w[kf[0]].astype(np.float64)
        print(f"bias: full, {bias.size} params ({kf[0]})")

    mean_map = preds.mean(0)
    evidence = mean_map - bias        # exact: logits = head(...) + bias()

    # --- variance: how static is this model, and who owns the static part? -------
    grand = preds.mean()
    static_var = ((mean_map - grand) ** 2).mean()          # between-voxel
    episode_var = preds.var(0).mean()                      # within-voxel
    print(f"\n=== variance of the predicted logits ===")
    print(f"  static (per-voxel mean)   {static_var:9.4f}   "
          f"{100 * static_var / (static_var + episode_var):5.1f}%")
    print(f"  episode-specific          {episode_var:9.4f}   "
          f"{100 * episode_var / (static_var + episode_var):5.1f}%")

    vb, ve = bias.var(), evidence.var()
    cov = np.cov(bias, evidence)[0, 1]
    print(f"\n=== the static map, split additively ===")
    print(f"  var(bias)                 {vb:9.4f}")
    print(f"  var(head-term mean)       {ve:9.4f}   (q AND attn_out, not evidence alone)")
    print(f"  2*cov                     {2 * cov:9.4f}   "
          f"(negative = the two cancel each other)")
    print(f"  var(mean_map)             {mean_map.var():9.4f}   "
          f"= sum of the three above")
    print(f"  corr(bias, mean_map)      {np.corrcoef(bias, mean_map)[0, 1]:9.4f}")

    # --- the query-only map: what the head emits with attention zeroed -----------
    # Closed form from the weights. decoder.queries() is the same q forward() uses,
    # so this is literally the model's output for an episode whose attention
    # returned nothing -- a pure function of the voxel index. Wrapped because it
    # needs torch and the run's frozen model.py, and the numpy-only table above is
    # still worth printing if that import fails.
    query_map = None
    run_dir = os.path.abspath(args.run) if args.run else snapshot_root(ck)
    try:
        import torch
        sys.path.insert(0, run_dir)
        from model import CustomConfig, CustomTransformerModel
        from safetensors.torch import load_file as load_torch
        cfg = CustomConfig.from_pretrained(ck)
        if getattr(cfg, 'decoder_residual', 'raw') == 'concat':
            # ln_o expects [q; a] at 2*d_model there, so 'q with attention zeroed'
            # is not a thing the head can be handed. Refuse rather than pad zeros
            # and call the result the query's own map.
            raise RuntimeError("decoder_residual='concat': q alone has no head input")
        m = CustomTransformerModel(cfg)
        miss, unexp = m.load_state_dict(load_torch(f"{ck}/model.safetensors"),
                                        strict=False)
        if miss or unexp:
            raise RuntimeError(f"state dict mismatch: {miss[:4]} {unexp[:4]}")
        m.eval()
        d = m.decoder
        with torch.no_grad():
            q = d.queries()                                   # [4096, D], no data
            query_map = (d.head(d.ln_o(q)).squeeze(-1) + d.bias()).double().numpy()
        print(f"\nquery-only map built from weights ({run_dir.split('/')[-1][:40]}...)")
    except Exception as e:
        print(f"\n(query-only map unavailable: {e})")

    # --- the number that decides it: AP of each map ALONE ------------------------
    fits = axis_fits(mean_map)
    rows = [
        ('model (per episode)', preds,           'reproduces the logged pr_auc'),
        ('mean map', mean_map,                   'static-self: one map for every episode'),
        ('bias alone (48 params)', bias,         'the 48-param static term, by itself'),
        ('head-term mean', evidence,             'mean map minus bias = mean of q+attn_out'),
        ('fit: all three axes', fits['all three'],
         'ceiling for ANY 48-param axis-sum map, fitted to THIS fold'),
        ('fit: z only (height)', fits['z only (height)'],
         'the transferable axis -- the honest --voxel_bias_lr ceiling'),
        ('fit: x only', fits['x only'],          'one kitchen floor plan, does not transfer'),
        ('fit: y only', fits['y only'],          'one kitchen floor plan, does not transfer'),
    ]
    if query_map is not None:
        rows.insert(3, ('query alone (no attn)', query_map,
                        'head(ln_o(q)) + bias, weights only'))

    # Scored first, printed second: the % column is relative to the mean map, which
    # is not the first row.
    scored = [(name, per_episode_ap(m, labels), note) for name, m, note in rows]
    ref = dict((n, a) for n, a, _ in scored)['mean map']
    print(f"\n=== per-episode AP of each map on its own ===")
    print(f"  {'map':<24} {'AP':>8} {'% of mean map':>15}")
    for name, a, note in scored:
        print(f"  {name:<24} {a:8.4f} {100 * a / ref:14.1f}%   {note}")


if __name__ == '__main__':
    main()