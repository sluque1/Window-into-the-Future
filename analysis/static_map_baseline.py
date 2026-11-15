"""Is the model anticipating, or returning a fixed map of the kitchen?

Companion to prediction_variability.py, which asked this of three env-modality
runs using the model's OWN mean prediction as the static map. That map is built
from the test participant's episodes, so it quietly carries information about
the held-out kitchen -- it answers "does the input matter?" but not "could a
prior fitted WITHOUT the test kitchen have done this?".

This script adds the honest baseline. Under LOOCV the nine folds partition the
data exactly (sum of the nine test sets = 35378 episodes; drop any one and the
remainder is that fold's training set -- 31052 for P01, matching its train log),
so fold P's TRAIN-split label marginal is the episode-count-weighted mean of the
other eight folds' test labels. No forward passes, no extra data loading.

Five scorers, all on the per-episode PR-AUC that drives model selection
(train.py compute_metrics: average_precision_score per episode, then mean):

  CHANCE      -- constant score. Per-episode AP degenerates to the episode's own
                 positive rate; the floor.
  TRAIN-PRIOR -- GT voxel marginal over the eight TRAINING participants, handed
                 to every test episode. Sees nothing of the test kitchen, and
                 nothing of the observation window. The baseline that matters:
                 anything the model does not beat here, it did not learn.
  MODEL       -- the real per-episode predictions.
  STATIC-SELF -- the model's own mean prediction over the test episodes. What the
                 model gets for free without reading the input; (MODEL -
                 STATIC-SELF) is what the 30s window is worth.
  ORACLE MAP  -- GT voxel marginal of the TEST participant itself. Upper bound on
                 any fixed map of this kitchen.

Usage:  python analysis/static_map_baseline.py [--glob PATTERN]
"""
import argparse
import glob
import os
import pickle

import numpy as np
from sklearn.metrics import average_precision_score

SNAP = os.environ.get("SNAPSHOTS_ROOT", "/data/video_datasets/HD-EPIC/snapshots")
DEFAULT_GLOB = f"{SNAP}/*_loocv_P0*_vlo_vq_axis_sum_factorised_cls_prior"


def latest_checkpoint(run_dir):
    cks = glob.glob(f"{run_dir}/results/*/checkpoint-*")
    if not cks:
        return None
    return max(cks, key=lambda p: int(p.rsplit("-", 1)[1]))


def load(ck, want_preds=True):
    """Labels (and optionally logits) in dataset order."""
    y = np.asarray(pickle.load(open(f"{ck}/labels.pkl", "rb"))["interaction"], np.uint8)
    if not want_preds:
        return y, None
    lg = pickle.load(open(f"{ck}/predictions.pkl", "rb"))
    if isinstance(lg, tuple):
        lg = lg[0]
    lg = np.asarray(lg, np.float32)
    idx = np.asarray(pickle.load(open(f"{ck}/indexes.pkl", "rb")))
    o = np.argsort(idx)
    return y[o], lg[o]


def per_episode_ap(scores, y):
    """scores: [4096] shared by every episode, or [N, 4096]. Mean AP over the
    episodes that have at least one positive and one negative -- the same filter
    train.py applies."""
    shared = scores.ndim == 1
    out = []
    for i in range(len(y)):
        yi = y[i]
        s = yi.sum()
        if 0 < s < len(yi):
            out.append(average_precision_score(yi, scores if shared else scores[i]))
    return float(np.mean(out)) if out else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default=DEFAULT_GLOB)
    args = ap.parse_args()

    runs = {}
    for d in sorted(glob.glob(args.glob)):
        pid = d.split("_loocv_")[1][:3]
        ck = latest_checkpoint(d)
        if ck:
            runs[pid] = ck
    if len(runs) < 2:
        raise SystemExit(f"need >=2 folds to build a train prior; found {len(runs)}")
    print(f"{len(runs)} folds: {', '.join(sorted(runs))}\n")

    # Pass 1 -- per-fold label marginal and episode count (cheap; no predictions).
    marg, n_ep = {}, {}
    for pid, ck in sorted(runs.items()):
        y, _ = load(ck, want_preds=False)
        marg[pid], n_ep[pid] = y.mean(0), len(y)
        print(f"  {pid}: {len(y)} episodes, positive rate {y.mean():.6f}")
    total = sum(n_ep.values())
    print(f"  pooled: {total} episodes\n")

    # Pass 2 -- score every fold.
    hdr = (f"{'fold':5} {'chance':>8} {'train-prior':>12} {'MODEL':>8} "
           f"{'static-self':>12} {'oracle-map':>11}   {'model-train':>12} {'model-static':>13}")
    print(hdr)
    print("-" * len(hdr))
    rows = []
    for pid, ck in sorted(runs.items()):
        y, lg = load(ck)
        # Train-split marginal: episode-count-weighted mean of the OTHER folds.
        others = [p for p in runs if p != pid]
        w = np.array([n_ep[p] for p in others], np.float64)
        train_map = (np.stack([marg[p] for p in others]) * w[:, None]).sum(0) / w.sum()

        chance = float(np.mean([y[i].mean() for i in range(len(y))
                                if 0 < y[i].sum() < y.shape[1]]))
        prior = per_episode_ap(train_map, y)
        model = per_episode_ap(1 / (1 + np.exp(-lg.astype(np.float64))), y)
        static = per_episode_ap(lg.astype(np.float64).mean(0), y)
        oracle = per_episode_ap(marg[pid], y)
        rows.append((pid, chance, prior, model, static, oracle))
        print(f"{pid:5} {chance:8.4f} {prior:12.4f} {model:8.4f} {static:12.4f} "
              f"{oracle:11.4f}   {model - prior:+12.4f} {model - static:+13.4f}")
        del y, lg

    a = np.array([r[1:] for r in rows])
    m = a.mean(0)
    print("-" * len(hdr))
    print(f"{'mean':5} {m[0]:8.4f} {m[1]:12.4f} {m[2]:8.4f} {m[3]:12.4f} {m[4]:11.4f}"
          f"   {m[2] - m[1]:+12.4f} {m[2] - m[3]:+13.4f}")
    beat = sum(1 for r in rows if r[3] > r[2])
    print(f"\nmodel beats the train-only prior on {beat}/{len(rows)} folds")


if __name__ == "__main__":
    main()
