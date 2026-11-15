"""What does a per-voxel prior learned from OTHER kitchens score on the held-out one?

THE QUESTION THIS SETTLES. static_map_decomposition.py showed the model's output on
P08 is 87.5% static, that the static map is not voxel identity (query alone: 0.0080,
against chance 0.0064) and not the 48-param bias (var 0.0000), and that a factorised
fit to the model's own mean output scores 0.4201 -- above the model's own 0.3735.
Since P08 was held out, the model must be READING that kitchen's layout from `obb` at
inference rather than remembering it, which is consistent with `obb` being the only
input the swap ablation registers (-0.0401, everything else under 0.008).

That story has a cheap falsifier. If a prior fitted to OTHER participants' ground
truth already scores near the model's number on the held-out kitchen, then the model's
whole result is a transferable occupancy prior obtainable with no video, no gaze and
no encoder -- and every modality experiment in this series was measuring the residue.
If instead the transferred prior scores near the z-only level (~0.04 on P08), the
model's live reading of `obb` is doing work no cross-kitchen prior can replicate, and
that is a positive result rather than a deflationary one.

Nothing is trained here and no model is loaded: labels.pkl next to each fold's
checkpoint is that fold's held-out participant's ground truth, so leave-one-fold-out
over the folds you have is a real transfer test.

THE ARMS.
  model                 the trained model on this fold (reproduces its logged pr_auc)
  full prior (other)    logit of the per-voxel positive rate pooled over the OTHER
                        folds -- 4096 free numbers, the best any transferable
                        per-voxel prior can do given these kitchens
  factorised (other)    ax[i]+ay[j]+az[k] fitted to that -- 48 numbers, exactly the
                        map --voxel_bias_lr could learn and install in decoder.bias()
  full prior (SELF)     the same thing fitted to the TARGET fold's own labels. An
                        oracle, listed to bound the other two: it is what a per-voxel
                        prior would score if kitchen layouts transferred perfectly.
  factorised (SELF)     the oracle's 48-param version.
  chance                a constant map. per_episode_ap ties everything, so this comes
                        out at the prevalence and calibrates every row above it.

WHY logit AND NOT THE RATE. AP reads only the ranking and the logit is monotone in
the rate, so the two are identical for the `full` arms. They are NOT identical once
the map is factorised: the axis fit is a least-squares approximation, and doing it in
logit space is what makes `factorised (other)` the map decoder.bias() would actually
hold -- the bias is additive in the logits, not in the probabilities. Laplace
smoothing (alpha=0.5) keeps empty voxels finite.

ONE LIMIT, STATED PLAINLY. A real --voxel_bias_lr run learns its bias from the eight
TRAINING participants. This script only has the folds whose runs you pass it -- three
others when you pass four. Fewer kitchens means less averaging-out of the
per-kitchen x/y layout, so `factorised (other)` here is if anything OPTIMISTIC about
what a trained bias would transfer.

Usage:
    python analysis/label_prior_transfer.py --glob '2026-08-24_00-30-*_loocv_P0*_vlog_*te10g8_e1'
    python analysis/label_prior_transfer.py --run <dir> --run <dir> ...
"""
import argparse
import glob
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from input_swap_ablation import per_episode_ap            # noqa: E402
from static_map_decomposition import axis_fits, load_pkl  # noqa: E402

SNAP = os.environ.get("SNAPSHOTS_ROOT", "/data/video_datasets/HD-EPIC/snapshots")


def logit_rate(labels, alpha=0.5):
    """Per-voxel log-odds of being a positive, Laplace-smoothed.

    alpha=0.5 (Jeffreys) rather than 0 so a voxel never seen positive is a large
    negative number instead of -inf, which would make the axis fit below undefined.
    """
    n = labels.shape[0]
    p = (labels.sum(0) + alpha) / (n + 2 * alpha)
    return np.log(p / (1 - p))


def last_checkpoint(run):
    cks = glob.glob(f"{run}/results/*/checkpoint-*")
    if not cks:
        return None
    return max(cks, key=lambda c: int(c.rsplit('-', 1)[1]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--glob', default='')
    ap.add_argument('--run', action='append', default=[])
    args = ap.parse_args()

    runs = [os.path.abspath(r) for r in args.run]
    if args.glob:
        runs += sorted(glob.glob(os.path.join(SNAP, args.glob)))
    if len(runs) < 2:
        raise SystemExit("need at least 2 folds to leave one out")

    folds = []
    for run in runs:
        ck = last_checkpoint(run)
        if ck is None:
            print(f"  skipping {os.path.basename(run)}: no checkpoint")
            continue
        m = re.search(r'loocv_(P\d+)_', os.path.basename(run))
        pid = m.group(1) if m else os.path.basename(run)[:12]
        labels = load_pkl(f"{ck}/labels.pkl").astype(np.float64)
        preds = load_pkl(f"{ck}/predictions.pkl").astype(np.float64)
        folds.append({'pid': pid, 'labels': labels, 'preds': preds,
                      'ck': os.path.basename(ck)})
        print(f"  {pid}: {labels.shape[0]} episodes, "
              f"{labels.sum(1).mean():.1f} positives/episode, {os.path.basename(ck)}")
    if len(folds) < 2:
        raise SystemExit("need at least 2 usable folds")

    print(f"\nleave-one-fold-out over {len(folds)} folds "
          f"({', '.join(f['pid'] for f in folds)})")
    hdr = (f"\n{'target':<7} {'model':>8} {'full(oth)':>10} {'fact(oth)':>10} "
           f"{'full(SELF)':>11} {'fact(SELF)':>11} {'chance':>8}")
    print(hdr)
    print('-' * len(hdr.strip()))

    for i, tgt in enumerate(folds):
        y = tgt['labels']
        others = [f['labels'] for j, f in enumerate(folds) if j != i]
        pooled = np.concatenate(others, axis=0)

        full_other = logit_rate(pooled)
        fact_other = axis_fits(full_other)['all three']
        full_self = logit_rate(y)
        fact_self = axis_fits(full_self)['all three']

        row = [
            per_episode_ap(tgt['preds'], y),
            per_episode_ap(full_other, y),
            per_episode_ap(fact_other, y),
            per_episode_ap(full_self, y),
            per_episode_ap(fact_self, y),
            # A constant map: per_episode_ap ties every voxel, so this lands on the
            # prevalence and calibrates everything above it.
            per_episode_ap(np.zeros(y.shape[1]), y),
        ]
        print(f"{tgt['pid']:<7} " + ' '.join(f"{v:>{w}.4f}" for v, w in
                                             zip(row, (8, 10, 10, 11, 11, 8))))

    print("\nfull(oth)/fact(oth) use ONLY the other folds' ground truth -- no video, "
          "no encoder.\nfull(SELF)/fact(SELF) are oracles: fitted to the target's own "
          "labels, so they bound\nwhat any per-voxel prior could reach if layouts "
          "transferred perfectly.")


if __name__ == '__main__':
    main()