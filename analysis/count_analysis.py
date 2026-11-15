"""Is the model's error a count error or a placement error?

Reads the saved eval dumps (`predictions.pkl` = raw logits [N,4096],
`labels.pkl['interaction']` = binary [N,4096]) from the best-by-pr_auc checkpoint
of the three LOOCV gaze runs and separates two things that PR-AUC mixes together:

  (A) cardinality -- does the model know HOW MANY voxels will be interacted with?
      Read out as the expected count sum_v sigmoid(logit_v): under BCEWithLogitsLoss
      sigmoid(logit_v) is the model's P(y_v=1), and |G| = sum_v y_v is a sum of
      Bernoullis, so E[|G|] = sum_v sigmoid(logit_v) by linearity of expectation
      (no independence assumption, no threshold).

  (B) placement  -- are its top-ranked voxels the right ones, and if not, how far
      off are they (in voxels and in metres, since voxel size differs per kitchen)?

Also compares against a static per-kitchen prior (the marginal GT frequency of each
voxel over that participant's own episodes) to see how much of the score is just
"this kitchen's hotspots" rather than anything read off the observation window.

Usage:  python analysis/count_analysis.py
"""
import os
import pickle

import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import roc_auc_score, average_precision_score

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
VOX = os.path.join(os.path.dirname(HERE), "new_voxel_centers")
SNAP = os.environ.get("SNAPSHOTS_ROOT", "/data/video_datasets/HD-EPIC/snapshots")

# best-by-pr_auc checkpoint of each run (from its trainer_state.json best_model_checkpoint)
RUNS = {
    "P04": f"{SNAP}/2026-07-30_20-06-14_loocv_P04_video_location_env_gaze/results/2026-07-31-07-46-57_2026-07-30_20-06-14_loocv_P04_video_location_env_gaze/checkpoint-8020",
    "P07": f"{SNAP}/2026-07-30_20-06-15_loocv_P07_video_location_env_gaze/results/2026-07-31-08-42-40_2026-07-30_20-06-15_loocv_P07_video_location_env_gaze/checkpoint-2043",
    "P08": f"{SNAP}/2026-07-30_20-06-16_loocv_P08_video_location_env_gaze/results/2026-07-31-09-19-23_2026-07-30_20-06-16_loocv_P08_video_location_env_gaze/checkpoint-7596",
}

BUDGETS = [1, 5, 10, 20, 50, 100, 200, 500]
THRESHOLDS = [0.01, 0.05, 0.1, 0.2]      # the operating points compute_metrics() uses


def load(d):
    logits = pickle.load(open(f"{d}/predictions.pkl", "rb"))
    if isinstance(logits, tuple):
        logits = logits[0]
    labels = pickle.load(open(f"{d}/labels.pkl", "rb"))["interaction"]
    return np.asarray(logits, np.float64), np.asarray(labels, np.float64)


def mean_auc(scores, y):
    """Mean per-episode (ROC-AUC, PR-AUC) over episodes with both classes present."""
    roc, pr = [], []
    for i in range(len(y)):
        if 0 < y[i].sum() < 4096:
            s = scores[i] if scores.ndim == 2 else scores
            roc.append(roc_auc_score(y[i], s))
            pr.append(average_precision_score(y[i], s))
    return float(np.mean(roc)), float(np.mean(pr))


rows, per_ep = [], {}
for pid, d in RUNS.items():
    logits, y = load(d)
    p = 1.0 / (1.0 + np.exp(-logits))
    n = len(y)
    k_true = y.sum(1)
    k_pred = p.sum(1)
    kk = k_true.astype(int)

    # The 4096 centres form a regular orthogonal 16^3 lattice that is ROTATED about
    # the vertical axis (yaw -24 deg for P04, +35 for P07, +24 for P08), so the
    # axis-aligned span of the centres is NOT the grid's own size. Take the per-axis
    # voxel dimension from the lattice step vectors instead.
    centers = np.load(f"{VOX}/{pid}_voxel_centers.npy").reshape(16, 16, 16, 3)
    steps = np.array([centers[1, 0, 0] - centers[0, 0, 0],
                      centers[0, 1, 0] - centers[0, 0, 0],
                      centers[0, 0, 1] - centers[0, 0, 0]])
    vox_m = np.linalg.norm(steps, axis=1)                        # per-axis voxel size
    centers = centers.reshape(-1, 3)
    coords = np.stack(np.unravel_index(np.arange(4096), (16, 16, 16)), 1).astype(float)

    order = np.argsort(-p, axis=1)
    y_sorted = np.take_along_axis(y, order, axis=1)
    csum = np.cumsum(y_sorted, axis=1)

    # --- (A) cardinality ---------------------------------------------------
    A = np.stack([k_pred, np.ones_like(k_pred)], 1)
    coef, *_ = np.linalg.lstsq(A, k_true, rcond=None)
    card = dict(
        k_mean=k_true.mean(), k_std=k_true.std(), k_min=k_true.min(), k_max=k_true.max(),
        kp_mean=k_pred.mean(), kp_std=k_pred.std(),
        mae_model=np.abs(k_pred - k_true).mean(),
        mae_const=np.abs(np.median(k_true) - k_true).mean(),
        mae_recal=np.abs(A @ coef - k_true).mean(),
        r_p=pearsonr(k_pred, k_true)[0], r_s=spearmanr(k_pred, k_true)[0],
    )

    # --- (B) placement: budget sweep --------------------------------------
    budget = {}
    for K in BUDGETS:
        hits = csum[:, K - 1]
        prec, rec = hits / K, hits / kk
        f1 = np.where(prec + rec > 0, 2 * prec * rec / np.maximum(prec + rec, 1e-12), 0.0)
        budget[K] = (prec.mean(), rec.mean(), f1.mean())
    # oracle budget K=|G| (precision == recall there)
    prec_kgt = csum[np.arange(n), kk - 1] / kk

    # --- (B) placement: threshold sweep -----------------------------------
    thr = {}
    for t in THRESHOLDS:
        flag = p >= t
        nflag = flag.sum(1)
        hits = (flag * y).sum(1)
        prec = np.where(nflag > 0, hits / np.maximum(nflag, 1), 0.0)
        thr[t] = (nflag.mean(), (nflag == 0).mean(), prec.mean(), (hits / kk).mean())

    # --- (B) placement: how far off are the top voxels? -------------------
    # distance from each of the top-10 predicted voxels to the NEAREST GT voxel
    d_vox, d_m = [], []
    for i in range(n):
        g = coords[y[i] > 0]                       # [|G|,3] GT voxel coords
        t10 = coords[order[i, :10]]                # [10,3] top-10 predicted
        dif = t10[:, None, :] - g[None, :, :]      # [10,|G|,3]
        d_vox.append(np.sqrt((dif ** 2).sum(-1)).min(1))
        d_m.append(np.sqrt(((dif * vox_m) ** 2).sum(-1)).min(1))
    d_vox, d_m = np.concatenate(d_vox), np.concatenate(d_m)

    # --- baselines ---------------------------------------------------------
    # static per-kitchen prior: marginal GT frequency of each voxel over this
    # participant's own episodes. Oracle (peeks at the test labels), so it is an
    # UPPER BOUND on what a fixed spatial prior for this kitchen can achieve.
    prior = y.mean(0)
    prior_roc, prior_pr = mean_auc(prior, y)
    model_roc, model_pr = mean_auc(p, y)

    rows.append(dict(
        pid=pid, n=n, vox_m=vox_m, extent=np.ptp(centers, 0),
        n_distinct=int((y.sum(0) > 0).sum()),          # voxels ever used as GT
        base_rate=k_true.mean() / 4096,
        model_roc=model_roc, model_pr=model_pr,
        prior_roc=prior_roc, prior_pr=prior_pr,
        prec_kgt=prec_kgt.mean(),
        d_vox_med=np.median(d_vox), d_m_med=np.median(d_m),
        d_vox_p90=np.percentile(d_vox, 90), d_m_p90=np.percentile(d_m, 90),
        frac_top10_hit=(d_vox == 0).mean(),           # top-10 voxel IS a GT voxel
        frac_top10_adj=(d_vox <= np.sqrt(3)).mean(),  # GT voxel or one of its 26 neighbours
        **card, budget=budget, thr=thr,
    ))
    per_ep[pid] = (k_true, k_pred)
    np.savez(os.path.join(OUT, f"{pid}_counts.npz"),
             k_true=k_true, k_pred=k_pred, prec_kgt=prec_kgt, d_vox=d_vox, d_m=d_m)


def hr(t):
    print(f"\n{'=' * 100}\n{t}\n{'=' * 100}")


hr("SETUP -- the three kitchens are not the same problem")
print("pid      n  grid extent (m)        voxel size (m)        distinct GT voxels  base rate")
for r in rows:
    print(f"{r['pid']}  {r['n']:5d}  {np.array2string(r['extent'], precision=2):22s} "
          f"{np.array2string(r['vox_m'], precision=3):22s} {r['n_distinct']:14d}  {r['base_rate']*100:8.3f}%")

hr("(A) CARDINALITY -- does the model know HOW MANY voxels?")
print("pid     true |G|          predicted count     MAE model  MAE const  MAE recal   Pearson r  Spearman")
for r in rows:
    print(f"{r['pid']}  {r['k_mean']:5.1f} +/- {r['k_std']:4.1f}     {r['kp_mean']:6.1f} +/- {r['kp_std']:4.1f}     "
          f"{r['mae_model']:9.2f}  {r['mae_const']:9.2f}  {r['mae_recal']:9.2f}   {r['r_p']:9.3f}  {r['r_s']:8.3f}")

hr("(B1) BUDGET SWEEP -- take the top-K voxels, how many are right?")
for r in rows:
    print(f"\n{r['pid']}   (true |G| averages {r['k_mean']:.1f})")
    print("   K      precision   recall     F1")
    for K in BUDGETS:
        pr_, rc_, f1_ = r["budget"][K]
        print(f"  {K:4d}   {pr_:9.3f}   {rc_:7.3f}   {f1_:6.3f}")
    print(f"  K=|G|   {r['prec_kgt']:9.3f}   {r['prec_kgt']:7.3f}   {r['prec_kgt']:6.3f}   (oracle count)")

hr("(B2) THRESHOLD SWEEP -- at a fixed probability cut, how many voxels fire?")
print("pid   thresh   mean #flagged   episodes w/ 0 flagged   precision   recall")
for r in rows:
    for t in THRESHOLDS:
        nf, zero, pr_, rc_ = r["thr"][t]
        print(f"{r['pid']}   {t:5.2f}   {nf:13.1f}   {zero*100:20.1f}%   {pr_:9.3f}   {rc_:6.3f}")
    print()

hr("(B3) PLACEMENT -- when a top-10 voxel is wrong, how wrong is it?")
print("pid   top-10 is GT   within 1 voxel   median dist (vox)   median dist (m)   p90 (vox)   p90 (m)")
for r in rows:
    print(f"{r['pid']}   {r['frac_top10_hit']*100:10.1f}%   {r['frac_top10_adj']*100:13.1f}%   "
          f"{r['d_vox_med']:17.2f}   {r['d_m_med']:15.2f}   {r['d_vox_p90']:9.2f}   {r['d_m_p90']:7.2f}")

hr("(C) MODEL vs STATIC KITCHEN PRIOR (prior is an oracle -> upper bound)")
print("pid   model ROC   prior ROC    model PR    prior PR    PR lift over prior   PR lift over chance")
for r in rows:
    print(f"{r['pid']}   {r['model_roc']:9.3f}   {r['prior_roc']:9.3f}   {r['model_pr']:9.3f}   "
          f"{r['prior_pr']:9.3f}   {r['model_pr']/r['prior_pr']:18.2f}x   {r['model_pr']/r['base_rate']:18.1f}x")

# --- plot ------------------------------------------------------------------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SURFACE, INK, MUTED, SERIES = "#fcfcfb", "#0b0b0b", "#898781", "#2a78d6"

lim = max(max(kt.max(), kp.max()) for kt, kp in per_ep.values()) * 1.05
fig, axes = plt.subplots(1, 3, figsize=(13, 4.6), sharex=True, sharey=True, facecolor=SURFACE)
for ax, r in zip(axes, rows):
    kt, kp = per_ep[r["pid"]]
    ax.set_facecolor(SURFACE)
    hb = ax.hexbin(kt, kp, gridsize=45, extent=(0, lim, 0, lim), bins="log",
                   cmap="Blues", mincnt=1, linewidths=0)
    ax.plot([0, lim], [0, lim], color=MUTED, lw=2, ls="--", zorder=3)
    edges = np.linspace(0, lim, 19)
    idx = np.digitize(kt, edges) - 1
    cx, cy = [], []
    for b in range(len(edges) - 1):
        m = idx == b
        if m.sum() >= 20:
            cx.append(0.5 * (edges[b] + edges[b + 1]))
            cy.append(np.median(kp[m]))
    ax.plot(cx, cy, color=SERIES, lw=2, zorder=4)
    ax.set_title(f"{r['pid']}   r={r['r_p']:.2f}   MAE {r['mae_model']:.1f} vs {r['mae_const']:.1f} const",
                 color=INK, fontsize=11, pad=10)
    ax.set_xlabel("true |G|  (GT voxels)", color=MUTED, fontsize=10)
    ax.tick_params(colors=MUTED, labelsize=9)
    for s in ax.spines.values():
        s.set_color(MUTED)
        s.set_linewidth(0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, color=MUTED, alpha=0.18, lw=0.6)
    ax.set_axisbelow(True)
axes[0].set_ylabel(r"predicted count  $\sum_v \sigma(\mathrm{logit}_v)$", color=MUTED, fontsize=10)
fig.suptitle("Predicted voxel count vs ground-truth count, per episode\n"
             "dashed = perfect calibration, blue = binned median of predictions",
             color=INK, fontsize=12.5, y=1.02)
fig.colorbar(hb, ax=axes, label="episodes (log)", fraction=0.02, pad=0.015)
path = os.path.join(OUT, "count_calibration.png")
fig.savefig(path, dpi=160, bbox_inches="tight", facecolor=SURFACE)
print(f"\nwrote {path}")
