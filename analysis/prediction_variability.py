"""How much of the model's output actually responds to the input?

Companion to count_analysis.py. That script showed the model's *count* is nearly
constant per participant; this one asks the stronger question -- is the whole
4096-vector nearly constant?

Decomposes each participant's predictions into a static and a dynamic part and
scores three things with the same per-episode PR-AUC used for model selection:

  1. MODEL      -- the real per-episode prediction p_i.
  2. STATIC     -- the model's OWN mean prediction pbar = mean_i p_i, handed to every
                   episode identically. Whatever pbar scores, the model gets for free
                   without reading the input at all. (model - static) is what the
                   30s observation window is worth.
  3. ORACLE MAP -- the GT marginal frequency of each voxel over the participant's
                   episodes, also constant. Upper bound on any fixed map of this
                   kitchen, so (oracle - static) is how much of the kitchen's layout
                   the model failed to infer.

Plus top-K set stability, to see directly whether the model reshuffles its top
voxels between episodes or returns the same ones every time.

Usage:  python analysis/prediction_variability.py
"""
import os
import pickle

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
SNAP = os.environ.get("SNAPSHOTS_ROOT", "/data/video_datasets/HD-EPIC/snapshots")

RUNS = {
    "P04": f"{SNAP}/2026-07-30_20-06-14_loocv_P04_video_location_env_gaze/results/2026-07-31-07-46-57_2026-07-30_20-06-14_loocv_P04_video_location_env_gaze/checkpoint-8020",
    "P07": f"{SNAP}/2026-07-30_20-06-15_loocv_P07_video_location_env_gaze/results/2026-07-31-08-42-40_2026-07-30_20-06-15_loocv_P07_video_location_env_gaze/checkpoint-2043",
    "P08": f"{SNAP}/2026-07-30_20-06-16_loocv_P08_video_location_env_gaze/results/2026-07-31-09-19-23_2026-07-30_20-06-16_loocv_P08_video_location_env_gaze/checkpoint-7596",
}
TOPK = 25
RNG = np.random.default_rng(0)


def load(d):
    lg = pickle.load(open(f"{d}/predictions.pkl", "rb"))
    if isinstance(lg, tuple):
        lg = lg[0]
    y = pickle.load(open(f"{d}/labels.pkl", "rb"))["interaction"]
    idx = pickle.load(open(f"{d}/indexes.pkl", "rb"))
    o = np.argsort(idx)                       # restore dataset (temporal) order
    return np.asarray(lg, np.float64)[o], np.asarray(y, np.float64)[o]


def per_episode_auc(scores, y):
    """scores: [4096] (shared) or [N,4096]. Returns mean (ROC, PR) over episodes."""
    roc, pr = [], []
    shared = scores.ndim == 1
    for i in range(len(y)):
        if 0 < y[i].sum() < 4096:
            s = scores if shared else scores[i]
            roc.append(roc_auc_score(y[i], s))
            pr.append(average_precision_score(y[i], s))
    return float(np.mean(roc)), float(np.mean(pr))


def set_stability(sets, n_pairs=4000):
    """Mean pairwise Jaccard between random pairs of top-K sets."""
    n = len(sets)
    i, j = RNG.integers(0, n, n_pairs), RNG.integers(0, n, n_pairs)
    m = i != j
    return float(np.mean([len(sets[a] & sets[b]) / len(sets[a] | sets[b])
                          for a, b in zip(i[m], j[m])]))


rows = []
for pid, d in RUNS.items():
    logits, y = load(d)
    p = 1.0 / (1.0 + np.exp(-logits))
    n = len(y)

    pbar = p.mean(0)                      # the model's static component
    prior = y.mean(0)                     # oracle map (uses test labels -> ceiling)

    m_roc, m_pr = per_episode_auc(p, y)
    s_roc, s_pr = per_episode_auc(pbar, y)
    o_roc, o_pr = per_episode_auc(prior, y)

    # --- how constant is the output? ---
    # share of total variance in p that survives after removing the per-participant mean
    v_between = p.var(0).mean()                       # variance across episodes, per voxel
    v_across = pbar.var()                             # variance across voxels of the mean
    # correlation of the full prediction vector between random episode pairs
    i, j = RNG.integers(0, n, 4000), RNG.integers(0, n, 4000)
    m = i != j
    zc = (p - p.mean(1, keepdims=True)) / np.maximum(p.std(1, keepdims=True), 1e-12)
    r_pairs = float(np.mean((zc[i[m]] * zc[j[m]]).sum(1) / 4096))

    # --- top-K stability ---
    top = np.argsort(-p, 1)[:, :TOPK]
    sets = [set(r.tolist()) for r in top]
    flat = top.ravel()
    freq = np.bincount(flat, minlength=4096) / n      # fraction of episodes each voxel is in top-K
    gt_sets = [set(np.flatnonzero(y[i]).tolist()) for i in range(n)]

    # how well does the model's static map line up with the true one?
    r_map = float(np.corrcoef(pbar, prior)[0, 1])
    prior_top = set(np.argsort(-prior)[:TOPK].tolist())
    core = set(np.flatnonzero(freq >= 0.9).tolist())  # in top-K for >=90% of episodes

    rows.append(dict(
        pid=pid, n=n,
        m_roc=m_roc, m_pr=m_pr, s_roc=s_roc, s_pr=s_pr, o_roc=o_roc, o_pr=o_pr,
        v_between=v_between, v_across=v_across, r_pairs=r_pairs, r_map=r_map,
        stab=set_stability(sets), gt_stab=set_stability(gt_sets),
        n_distinct=int((freq > 0).sum()), n_core=len(core),
        core_frac=len(core) / TOPK,
        core_in_prior_top=len(core & prior_top),
        top_always=float(np.sort(freq)[::-1][:TOPK].mean()),
    ))

hdr = lambda t: print(f"\n{'=' * 104}\n{t}\n{'=' * 104}")

hdr(f"IS THE OUTPUT CONSTANT?  (top-{TOPK} sets, and the full 4096-vector)")
print("pid   mean pairwise Jaccard      distinct voxels     voxels in top-K      corr between")
print("      of top-K sets (GT's own)   ever in top-K       for >=90% of eps     two random episodes' full vectors")
for r in rows:
    print(f"{r['pid']}   {r['stab']:8.3f}  ({r['gt_stab']:.3f})       {r['n_distinct']:11d}       "
          f"{r['n_core']:8d} / {TOPK:<8d}    {r['r_pairs']:.4f}")

hdr("WHAT IS THE INPUT WORTH?  (per-episode PR-AUC, same metric as model selection)")
print("pid    STATIC (model's own mean)   MODEL (real, per-episode)   gain from input   ORACLE MAP   static/oracle")
for r in rows:
    gain = r["m_pr"] - r["s_pr"]
    print(f"{r['pid']}    {r['s_pr']:23.4f}   {r['m_pr']:23.4f}   {gain:+15.4f}   {r['o_pr']:10.4f}   "
          f"{r['s_pr']/r['o_pr']:12.2f}")

print("\n  same, ROC-AUC:")
print("pid    STATIC    MODEL     gain    ORACLE MAP")
for r in rows:
    print(f"{r['pid']}    {r['s_roc']:.4f}   {r['m_roc']:.4f}   {r['m_roc']-r['s_roc']:+.4f}   {r['o_roc']:.4f}")

hdr("DOES THE STATIC MAP POINT AT THE RIGHT PLACES?")
print("pid   corr(model mean, true GT map)   of the always-on top-K voxels, how many are in the true top-K")
for r in rows:
    print(f"{r['pid']}   {r['r_map']:29.3f}   {r['core_in_prior_top']:6d} of {r['n_core']}")

np.savez(os.path.join(OUT, "prediction_variability.npz"),
         **{f"{r['pid']}_{k}": v for r in rows for k, v in r.items() if isinstance(v, (int, float))})
print(f"\nwrote {os.path.join(OUT, 'prediction_variability.npz')}")
