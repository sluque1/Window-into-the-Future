"""Why is the attribution almost independent of which voxels are the target?

The label-randomisation control failed: shuffling the ground truth across episodes
changed the TARGET substantially (per-episode correlation +0.24 for pos_contrast,
+0.05 for loss) but left the modality attribution nearly untouched (+0.97 and
+0.83). The label-independent `top20` target came out bit-identical, so the
machinery is wired correctly -- the insensitivity is real, not a plumbing bug.

The hypothesis this measures:

    d(logit_v)/dx is nearly the SAME VECTOR for every voxel v.

If it is, then averaging it over any 21 voxels gives nearly the same answer
whichever 21 you pick, and every label-dependent target inherits that. The model's
sensitivity to an input token would then be essentially global -- "how much does
this token move the whole map" -- while WHICH voxel scores high is carried by the
decoder's query and bias, neither of which depends on the input at all. That is
the token-gradient counterpart of the finding already recorded in CLAUDE.md, that
~91% of logit variance is a static per-voxel field.

Measured here as the pairwise cosine similarity between d(logit_v)/dx for
different v, over the flattened [L, H] gradient. Reported separately for pairs of
ground-truth-positive voxels, pairs of random voxels, and mixed pairs -- if the
similarity is high in all three, the target choice cannot matter and no
label-dependent target will fix it.

    python analysis/modality_importance/m1_voxel_alignment.py --run A --n 8
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402
import runs as runspec  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', default='A', choices=['A', 'B'])
    ap.add_argument('--which', default='best', choices=['best', 'final'])
    ap.add_argument('--n', type=int, default=16, help="episodes")
    ap.add_argument('--n_pos', type=int, default=12, help="GT-positive voxels per episode")
    ap.add_argument('--n_rand', type=int, default=12, help="random voxels per episode")
    ap.add_argument('--device', default='')
    args = ap.parse_args()

    device = torch.device(args.device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    common.set_seed(0)
    rng = np.random.default_rng(0)
    model, config = common.load_checkpoint(runspec.checkpoint_dir(args.run, args.which), device)
    dataset = common.build_test_dataset(args.run, config, verbose=False)
    layout = common.read_layout(model, dataset, device)
    print(f"\n{runspec.tag(args.run, args.which)}  device={device}  {args.n} episodes")

    # Cosines are recorded WITH the grid distance between the two voxels, because
    # the raw pos-vs-rand comparison is confounded: interactions cluster (counter
    # height, near hobs and sinks) while random voxels are spread over the whole
    # 16^3 grid, so pos-pos pairs are physically closer on average. Each voxel's
    # query is built from its coordinates, so nearby voxels get similar queries and
    # therefore similar gradients for purely geometric reasons. Without matching on
    # distance, "GT voxels are alike" cannot be told apart from "these voxels are
    # near each other".
    stats = {'pos-pos': [], 'rand-rand': [], 'pos-rand': []}
    dists = {'pos-pos': [], 'rand-rand': [], 'pos-rand': []}
    GRID = 16
    ijk_all = np.stack(np.unravel_index(np.arange(GRID ** 3), (GRID,) * 3), axis=-1)
    seg_share = []
    for ep in range(min(args.n, len(dataset))):
        batch = common.collate(dataset, [ep], device)
        labels = batch.pop('labels')['interaction'][0]
        pos = torch.nonzero(labels, as_tuple=False).flatten().cpu().numpy()
        neg = np.setdiff1d(np.arange(labels.numel()), pos)
        sel_pos = rng.choice(pos, size=min(args.n_pos, len(pos)), replace=False)
        sel_rand = rng.choice(neg, size=min(args.n_rand, len(neg)), replace=False)
        voxels = np.concatenate([sel_pos, sel_rand])
        kind = np.array(['pos'] * len(sel_pos) + ['rand'] * len(sel_rand))

        grads = []
        with common.Capture(model) as cap:
            out = model(**batch)
            logits = out['interaction']
            x = cap.encoder_input
            valid = common.valid_matrix(cap.pad_mask)[0]
            for i, v in enumerate(voxels):
                g, = torch.autograd.grad(logits[0, int(v)], x,
                                         retain_graph=(i < len(voxels) - 1))
                grads.append(g[:, 0].detach().flatten())
        G = torch.stack(grads)                         # [V, L*H]
        G = G / G.norm(dim=1, keepdim=True).clamp(min=1e-12)
        C = (G @ G.T).cpu().numpy()                    # cosine similarity

        coords = ijk_all[voxels]                       # [V, 3] grid indices
        for i in range(len(voxels)):
            for j in range(i + 1, len(voxels)):
                key = ('pos-pos' if kind[i] == kind[j] == 'pos'
                       else 'rand-rand' if kind[i] == kind[j] == 'rand' else 'pos-rand')
                stats[key].append(C[i, j])
                dists[key].append(float(np.linalg.norm(coords[i] - coords[j])))

        # Does the MODALITY split of a single voxel's gradient depend on the voxel?
        # If the vectors are near-parallel the shares must agree too, but shares are
        # what the report actually claims, so they are checked in their own right.
        gl = torch.stack(grads).view(len(voxels), layout.seq_len, -1).norm(dim=-1)
        gl = gl.cpu().numpy() * valid[None, :]
        sh = np.stack([gl[:, layout.index_mask(nm)].sum(1) for nm in layout.names], 1)
        sh = sh / sh.sum(1, keepdims=True)
        seg_share.append(sh)
        print(f"  episode {ep:3d}: {len(pos):3d} GT voxels, "
              f"mean cos(pos,pos)={np.mean([C[i,j] for i in range(len(sel_pos)) for j in range(i+1,len(sel_pos))]):.4f}")

    print(f"\n=== RAW (confounded by distance -- see the mean-distance column)")
    print(f"{'pair type':12s} {'n':>8s} {'mean cos':>10s} {'median':>9s} {'p5':>9s} "
          f"{'min':>9s} {'mean dist':>10s}")
    for k in stats:
        v, d = np.array(stats[k]), np.array(dists[k])
        print(f"{k:12s} {len(v):8d} {v.mean():10.4f} {np.median(v):9.4f} "
              f"{np.percentile(v,5):9.4f} {v.min():9.4f} {d.mean():10.2f}")

    # --- distance-matched comparison -------------------------------------
    # Bin every pair by grid distance, then compare pair types WITHIN a bin. If
    # the pos-pos advantage survives, GT-ness carries signal; if it vanishes, the
    # raw gap was geometry and no contrastive target follows from it.
    edges = np.array([0, 3, 5, 7, 9, 11, 13, 16, 100.0])
    print(f"\n=== DISTANCE-MATCHED (compare ACROSS a row, not down a column)")
    hdr = f"{'grid dist':>12s}"
    for k in stats:
        hdr += f" | {k+' n':>12s} {'cos':>7s}"
    print(hdr)
    binned = {k: np.digitize(np.array(dists[k]), edges) for k in stats}
    for b in range(1, len(edges)):
        lo, hi = edges[b - 1], edges[b]
        row = f"{lo:5.0f}-{hi:<6.0f}"
        any_row = False
        for k in stats:
            sel = binned[k] == b
            n = int(sel.sum())
            if n >= 5:
                row += f" | {n:12d} {np.array(stats[k])[sel].mean():7.4f}"
                any_row = True
            else:
                row += f" | {n:12d} {'-':>7s}"
        if any_row:
            print(row)

    # Fine bins (width 1.0) over the COMMON SUPPORT only. Both halves of the
    # comparison must be restricted to the same bins: pos-pos pairs reach distances
    # rand-rand pairs never do, and averaging pos-pos over its full range against
    # rand-rand over the overlap alone reintroduces exactly the bias the matching
    # exists to remove. Validated on synthetic data where cosine depends ONLY on
    # distance: the raw gap of +0.24 falls to +0.002 here, and to only +0.06 if
    # either refinement is dropped.
    fine = np.arange(0.0, 28.1, 1.0)
    fb = {k: np.digitize(np.array(dists[k]), fine) for k in stats}

    def matched(target_key, other_key, min_n=5):
        bt, bo = fb[target_key], fb[other_key]
        ct, co = np.array(stats[target_key]), np.array(stats[other_key])
        tnum = onum = den = 0.0
        for b in np.unique(bt):
            selt, selo = bt == b, bo == b
            if selt.sum() >= min_n and selo.sum() >= min_n:
                w = float(selt.sum())
                tnum += w * ct[selt].mean()
                onum += w * co[selo].mean()
                den += w
        if den == 0:
            return float('nan'), float('nan'), 0
        return tnum / den, onum / den, int(den)

    print(f"\n  both sides restricted to the SAME distance bins (width 1.0):")
    print(f"  {'comparison':28s} {'pos-pos':>9s} {'other':>9s} {'gap':>9s} {'pairs':>8s}")
    for other in ('rand-rand', 'pos-rand'):
        t, o, n = matched('pos-pos', other)
        print(f"  {'pos-pos vs ' + other:28s} {t:9.4f} {o:9.4f} {t - o:+9.4f} {n:8d}")
    print("\n  gap ~ 0        -> the raw separation was DISTANCE, not GT-ness. No claim")
    print("                    about contrastive targets follows.")
    print("  gap clearly >0 -> GT voxels share something beyond being near each other.")

    sh = np.concatenate(seg_share, 0)
    print(f"\nper-voxel modality SHARE, spread across voxels (all episodes pooled):")
    print(f"  {'segment':12s} {'mean':>9s} {'sd':>9s} {'min':>9s} {'max':>9s}")
    for i, nm in enumerate(layout.names):
        print(f"  {nm:12s} {sh[:,i].mean():9.4f} {sh[:,i].std():9.4f} "
              f"{sh[:,i].min():9.4f} {sh[:,i].max():9.4f}")

    print("\nreading this:")
    print("  the MODALITY SHARE table above is the finding that matters, and it needs no")
    print("     pos/rand contrast: if every voxel gives the same share (small sd), then a")
    print("     share-based statistic cannot depend on WHICH voxels are the target, which")
    print("     is exactly why the label-shuffle control came back unchanged.")
    print("  the cosine table answers a separate question -- whether the model distinguishes")
    print("     voxels at all. cos well below 1.0 says it does. Only the DISTANCE-MATCHED")
    print("     rows license any claim about GT-ness specifically.")


if __name__ == '__main__':
    main()
