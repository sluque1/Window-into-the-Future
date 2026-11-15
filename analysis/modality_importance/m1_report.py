"""Method 1 reporting: aggregation, statistics, plots, and a markdown summary.

Reads whatever m1_gradients.py wrote under out/m1/<tag>/<control>/ and produces
the tables and figures. Reruns are cheap -- everything here is derived from the
saved per-episode arrays, so the interpretation can be revised without touching
the GPU again.

    python analysis/modality_importance/m1_report.py

Two aggregations are reported side by side throughout, because they answer
different questions and rank the modalities differently by construction:

  TOTAL     the summed score over a modality's valid tokens. This is the honest
            "how much does this modality contribute", and it is dominated by
            token count -- obb has up to 512 slots against video's 30.
  PER TOKEN total / (number of valid tokens). "How much does one token of this
            modality carry", which is the right question when comparing a
            512-slot segment with a 1-slot one, and the wrong one if you want to
            know what the model actually leans on.

Neither is more correct. Reporting only one is how this analysis misleads.
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402
import runs as runspec  # noqa: E402

# Colour-blind-safe, one stable colour per segment across every figure.
SEG_COLOUR = {
    'cls':        '#4C6EF5',
    'video':      '#E8590C',
    'sep':        '#868E96',
    'location':   '#2F9E44',
    'obb':        '#9C36B5',
    'trajectory': '#0CA678',
}


# ---------------------------------------------------------------------------

def load(tag, control='main'):
    d = os.path.join(common.OUT_BASE, 'm1', tag, control)
    npz = os.path.join(d, 'scores.npz')
    if not os.path.exists(npz):
        return None
    z = np.load(npz, allow_pickle=False)
    with open(os.path.join(d, 'manifest.json')) as f:
        manifest = json.load(f)
    return {'dir': d, 'z': z, 'manifest': manifest,
            'segments': [str(s) for s in z['segments']],
            'targets': [str(t) for t in z['targets']]}


def derived(rec, target, score):
    """(total, per_token, share) each [N, S]. Padding is already excluded."""
    z = rec['z']
    total = z[f'total__{target}__{score}']
    nval = z['n_valid'].astype(np.float64)
    with np.errstate(invalid='ignore', divide='ignore'):
        per_token = np.where(nval > 0, total / np.maximum(nval, 1), np.nan)
        denom = total.sum(axis=1, keepdims=True)
        share = np.where(np.abs(denom) > 0, total / denom, np.nan)
    return total, per_token, share


def bootstrap_ci(x, n_boot=10000, alpha=0.05, seed=0, chunk=250):
    """Percentile CI of the mean over episodes, per column. x: [N, S].

    Resamples in chunks: the full [n_boot, N, S] gather would be 2.4 GB at
    N=5007, n_boot=10000, S=6, which is a silent OOM on a busy node.
    """
    rng = np.random.default_rng(seed)
    n = x.shape[0]
    means = np.empty((n_boot, x.shape[1]), dtype=np.float64)
    for start in range(0, n_boot, chunk):
        k = min(chunk, n_boot - start)
        idx = rng.integers(0, n, size=(k, n))
        means[start:start + k] = np.nanmean(x[idx], axis=1)
    lo = np.nanpercentile(means, 100 * alpha / 2, axis=0)
    hi = np.nanpercentile(means, 100 * (1 - alpha / 2), axis=0)
    return lo, hi


def kendall_tau(a, b):
    from scipy.stats import kendalltau
    t, p = kendalltau(a, b)
    return float(t), float(p)


def rank_stability(x, n_chunks=10, seed=0):
    """Kendall tau between modality rankings computed on disjoint episode chunks.

    Answers "would a different batch have given a different ranking?", which is
    the question a single mean over 5000 episodes cannot answer.
    """
    from itertools import combinations
    n = x.shape[0]
    if n < n_chunks * 2:
        return None
    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    chunks = np.array_split(order, n_chunks)
    means = np.stack([np.nanmean(x[c], axis=0) for c in chunks])   # [chunks, S]
    taus = [kendall_tau(means[i], means[j])[0] for i, j in combinations(range(n_chunks), 2)]
    return {'mean_tau': float(np.mean(taus)), 'min_tau': float(np.min(taus)),
            'n_pairs': len(taus), 'n_chunks': n_chunks}


# ---------------------------------------------------------------------------

def table_rows(rec, target, score):
    segs = rec['segments']
    total, per_token, share = derived(rec, target, score)
    nval = rec['z']['n_valid']
    xn = rec['z']['x_norm']
    valid = rec['z']['valid']
    lo_s, hi_s = bootstrap_ci(share)
    lo_p, hi_p = bootstrap_ci(per_token)
    rows = []
    for s, name in enumerate(segs):
        idx = _seg_index(rec, name)
        xn_mean = float(np.nanmean(np.where(valid[:, idx], xn[:, idx], np.nan)))
        rows.append({
            'segment': name,
            'slots': int(idx.sum()),
            'valid_tokens_mean': float(nval[:, s].mean()),
            'token_norm_mean': xn_mean,
            'total_mean': float(np.nanmean(total[:, s])),
            'total_median': float(np.nanmedian(total[:, s])),
            'total_sd': float(np.nanstd(total[:, s])),
            'share_mean': float(np.nanmean(share[:, s])),
            'share_ci_lo': float(lo_s[s]), 'share_ci_hi': float(hi_s[s]),
            'per_token_mean': float(np.nanmean(per_token[:, s])),
            'per_token_median': float(np.nanmedian(per_token[:, s])),
            'per_token_sd': float(np.nanstd(per_token[:, s])),
            'per_token_ci_lo': float(lo_p[s]), 'per_token_ci_hi': float(hi_p[s]),
        })
    return rows


def _seg_index(rec, name):
    groups = rec['manifest']['layout']['groups'][name]
    L = rec['manifest']['layout']['seq_len']
    m = np.zeros(L, dtype=bool)
    for s, e in groups:
        m[s:e] = True
    return m


# ---------------------------------------------------------------------------

def fig_modality(records, target, score, out_path):
    """Total share and per-valid-token mean, side by side, across checkpoints."""
    tags = list(records)
    segs = records[tags[0]]['segments']
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    width = 0.8 / len(tags)

    for panel, (ax, kind) in enumerate(zip(axes, ('share', 'per_token'))):
        for ti, tag in enumerate(tags):
            rec = records[tag]
            total, per_token, share = derived(rec, target, score)
            y = share if kind == 'share' else per_token
            m = np.nanmean(y, axis=0)
            lo, hi = bootstrap_ci(y)
            xs = np.arange(len(segs)) + ti * width - 0.4 + width / 2
            ax.bar(xs, m, width * 0.92, label=tag,
                   color=[SEG_COLOUR[s] for s in segs],
                   alpha=0.45 + 0.55 * (ti / max(len(tags) - 1, 1)),
                   edgecolor='black', linewidth=0.4)
            ax.errorbar(xs, m, yerr=[m - lo, hi - m], fmt='none',
                        ecolor='black', elinewidth=0.7, capsize=2)
        ax.set_xticks(np.arange(len(segs)))
        ax.set_xticklabels(segs, rotation=20)
        ax.set_title('share of total (sums to 1 across segments)' if kind == 'share'
                     else 'mean per VALID token')
        ax.grid(axis='y', alpha=0.25, linewidth=0.5)
        if kind == 'per_token':
            ax.set_yscale('log')
    fig.suptitle(f'Method 1 -- {score} attribution of target "{target}"  '
                 f'(bars: {" / ".join(tags)}, left to right; 95% bootstrap CI)', fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def fig_token_profile(records, out_path, score='grad_l2'):
    """Per-position mean, so within-modality structure is visible.

    The video and location panels are a TIME profile: slot 29 is the second
    nearest the anchor (the loader end-aligns both). The obb panel is a rank
    profile: boxes enter sorted by detection confidence, so slot 0 is the most
    confident box, and the tail beyond ~250 is padding on essentially every
    episode.
    """
    tags = list(records)
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.4))
    panels = [('video', 'seconds (29 = nearest the anchor)'),
              ('location', 'seconds (29 = nearest the anchor)'),
              ('obb', 'box slot (0 = highest confidence)'),
              ('trajectory', 'track slot')]
    for ax, (name, xlabel) in zip(axes, panels):
        for tag in tags:
            rec = records[tag]
            if name not in rec['manifest']['layout']['groups']:
                continue
            idx = _seg_index(rec, name)
            tok = rec['z'][f'tokens__{score}'][:, idx]
            valid = rec['z']['valid'][:, idx]
            with np.errstate(invalid='ignore'):
                prof = np.nansum(np.where(valid, tok, np.nan), axis=0) / np.maximum(valid.sum(axis=0), 1)
            ax.plot(prof, linewidth=1.2, label=tag)
        ax.set_title(name, color=SEG_COLOUR.get(name, 'k'))
        ax.set_xlabel(xlabel, fontsize=8)
        ax.grid(alpha=0.25, linewidth=0.5)
    axes[0].set_ylabel(f'mean {score} per valid token')
    axes[-1].legend(fontsize=7)
    fig.suptitle(f'Method 1 -- within-modality {score} profile '
                 f'(primary target "{common.PRIMARY_TARGET}")', fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def fig_signed(records, out_path, target=None):
    """Signed grad x input: which modalities push the target up vs down."""
    target = target or common.PRIMARY_TARGET
    tags = list(records)
    segs = records[tags[0]]['segments']
    fig, ax = plt.subplots(figsize=(8, 4.2))
    width = 0.8 / len(tags)
    for ti, tag in enumerate(tags):
        total, per_token, _ = derived(records[tag], target, 'gxi')
        m = np.nanmean(total, axis=0)
        lo, hi = bootstrap_ci(total)
        xs = np.arange(len(segs)) + ti * width - 0.4 + width / 2
        ax.bar(xs, m, width * 0.92, label=tag, color=[SEG_COLOUR[s] for s in segs],
               alpha=0.45 + 0.55 * (ti / max(len(tags) - 1, 1)),
               edgecolor='black', linewidth=0.4)
        ax.errorbar(xs, m, yerr=[m - lo, hi - m], fmt='none', ecolor='black',
                    elinewidth=0.7, capsize=2)
    ax.axhline(0, color='black', linewidth=0.8)
    ax.set_xticks(np.arange(len(segs)))
    ax.set_xticklabels(segs, rotation=20)
    ax.set_ylabel('summed grad x input (signed)')
    ax.set_title(f'Signed first-order contribution to "{target}"  '
                 f'(positive = pushes the target up)', fontsize=10)
    ax.grid(axis='y', alpha=0.25, linewidth=0.5)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def channel_table(rec, target):
    """[S, C] mean signed contribution per (segment, channel), plus shared-share.

    pos_code is ONE function (model._voxel_pos_embedding) used by location, obb,
    trajectory and the decoder's 4096 queries; class is shared by obb and
    trajectory. This says how much of each modality's signed contribution arrives
    through those shared parameters rather than through anything specific to it --
    the difference between "obb tells the model something" and "obb is another
    place the shared spatial code is read".
    """
    segs, chans = rec['segments'], [str(c) for c in rec['z']['channels']]
    shared = [str(c) for c in rec['z']['shared_channels']]
    M = np.stack([np.nanmean(rec['z'][f'chan__{target}__{c}'], axis=0) for c in chans], axis=1)
    total = np.nanmean(rec['z'][f'total__{target}__gxi'], axis=0)          # [S]
    # Share of the modality's ABSOLUTE channel mass, so cancelling +/- terms do
    # not manufacture a share above 1.
    absmass = np.abs(M).sum(axis=1)
    shared_idx = [chans.index(c) for c in shared]
    with np.errstate(invalid='ignore', divide='ignore'):
        shared_frac = np.where(absmass > 0, np.abs(M[:, shared_idx]).sum(axis=1) / absmass, np.nan)
    return segs, chans, M, total, shared_frac


def fig_channels(records, out_path, target=None):
    """Stacked signed contribution per modality, split by the part it came from."""
    target = target or common.PRIMARY_TARGET
    tags = list(records)
    fig, axes = plt.subplots(1, len(tags), figsize=(5.2 * len(tags), 4.4), squeeze=False)
    cmap = plt.get_cmap('tab10')
    for ax, tag in zip(axes[0], tags):
        segs, chans, M, total, _ = channel_table(records[tag], target)
        pos = np.zeros(len(segs)); neg = np.zeros(len(segs))
        for ci, c in enumerate(chans):
            v = M[:, ci]
            if not np.any(np.abs(v) > 0):
                continue
            base = np.where(v >= 0, pos, neg)
            ax.bar(np.arange(len(segs)), v, 0.7, bottom=base, label=c,
                   color=cmap(ci % 10), edgecolor='black', linewidth=0.3)
            pos = pos + np.maximum(v, 0); neg = neg + np.minimum(v, 0)
        ax.axhline(0, color='black', linewidth=0.8)
        ax.set_xticks(np.arange(len(segs)))
        ax.set_xticklabels(segs, rotation=20)
        ax.set_title(tag, fontsize=9)
        ax.grid(axis='y', alpha=0.25, linewidth=0.5)
    axes[0][0].set_ylabel('signed grad x input')
    axes[0][-1].legend(fontsize=7, ncol=2)
    fig.suptitle(f'Channel decomposition of "{target}" -- which PART of each token '
                 'carries the contribution\n(pos_code and class are SHARED parameters)',
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def fig_scale_confound(records, out_path, target=None, score='grad_l2'):
    """Is the ranking just re-reporting how big the tokens are?

    TOKEN_LAYERNORM=false in both runs, so segments enter the encoder at norms
    spanning 0.56 (cls) to ~62 (obb) purely because of how they were built. If
    per-token attribution tracks that line, the ranking is a scale artefact.
    """
    target = target or common.PRIMARY_TARGET
    fig, ax = plt.subplots(figsize=(6, 4.6))
    for tag, rec in records.items():
        rows = table_rows(rec, target, score)
        x = [r['token_norm_mean'] for r in rows]
        y = [r['per_token_mean'] for r in rows]
        ax.scatter(x, y, s=42, edgecolor='black', linewidth=0.5,
                   c=[SEG_COLOUR[r['segment']] for r in rows], label=tag)
        for r, xi, yi in zip(rows, x, y):
            ax.annotate(r['segment'], (xi, yi), fontsize=7,
                        xytext=(3, 3), textcoords='offset points')
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel('mean token norm ||x||  (how big the token is)')
    ax.set_ylabel(f'mean {score} per valid token')
    ax.set_title('Scale confound check', fontsize=10)
    ax.grid(alpha=0.25, linewidth=0.5)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--control', default='main')
    ap.add_argument('--target', default=common.PRIMARY_TARGET)
    ap.add_argument('--score', default='grad_l2')
    ap.add_argument('--out_dir', default=os.path.join(common.OUT_BASE, 'm1', '_report'))
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    records = {}
    for run, which in runspec.CHECKPOINT_KEYS:
        tag = runspec.tag(run, which)
        rec = load(tag, args.control)
        if rec is not None:
            records[tag] = rec
    if not records:
        raise SystemExit(f"no results found under {common.OUT_BASE}/m1/*/{args.control}/ "
                         "-- run m1_gradients.py first")
    print(f"loaded {len(records)} checkpoint(s): {list(records)}")

    summary = {'control': args.control, 'target': args.target, 'score': args.score,
               'checkpoints': {}, 'comparisons': {}}
    csv_lines = ['tag,target,score,segment,slots,valid_tokens_mean,token_norm_mean,'
                 'total_mean,total_median,total_sd,share_mean,share_ci_lo,share_ci_hi,'
                 'per_token_mean,per_token_median,per_token_sd,per_token_ci_lo,per_token_ci_hi']

    md = ['# Method 1 -- gradient-based modality importance', '',
          f"Target: **{args.target}** -- {common.TARGET_DOC[args.target]}", '',
          f"Score: **{args.score}** -- {common.SCORE_DOC[args.score]}", '']

    for tag, rec in records.items():
        man = rec['manifest']
        diag = man['diagnostics']
        rows = table_rows(rec, args.target, args.score)
        keys = list(rows[0])   # matches the header after tag,target,score
        for r in rows:
            csv_lines.append(','.join([tag, args.target, args.score] +
                                      [str(r[k]) for k in keys]))
        _, _, share = derived(rec, args.target, args.score)
        stab = rank_stability(share)
        # Does the ranking just track token size?
        from scipy.stats import pearsonr, spearmanr
        xn = [r['token_norm_mean'] for r in rows]
        pt = [r['per_token_mean'] for r in rows]
        r_scale = spearmanr(xn, pt)
        summary['checkpoints'][tag] = {
            'checkpoint': man['checkpoint'], 'n_episodes': man['n_episodes'],
            'eval_pr_auc_logged': man['eval_pr_auc_logged'],
            'diagnostics': diag, 'rows': rows,
            'rank_stability': stab,
            'scale_confound_spearman': float(r_scale.statistic),
        }

        md += [f"## {tag}", '',
               f"- checkpoint: `{man['checkpoint']}`",
               f"- episodes: {man['n_episodes']} (held-out {man['test_participants']}), "
               f"logged eval PR-AUC {man['eval_pr_auc_logged']:.4f}",
               f"- padded-token max |grad| = `{diag['pad_grad_max']:.3g}` (must be 0)",
               f"- predictions.pkl max|delta| = "
               f"`{diag['repro_max_abs'] if diag['repro_max_abs'] is not None else 'n/a'}`",
               f"- batched vs single backward max|delta| = `{diag['batched_vs_single_max_abs']:.3g}`",
               f"- ranking stability across {stab['n_chunks'] if stab else '-'} disjoint chunks: "
               f"mean Kendall tau = {stab['mean_tau']:.3f} (min {stab['min_tau']:.3f})"
               if stab else "- ranking stability: too few episodes",
               f"- token-size confound: Spearman(mean ||x||, per-token score) = "
               f"{r_scale.statistic:+.3f}", '',
               '| segment | slots | valid/ep | mean ||x|| | share of total | per valid token |',
               '|---|---|---|---|---|---|']
        for r in rows:
            md.append(f"| {r['segment']} | {r['slots']} | {r['valid_tokens_mean']:.1f} | "
                      f"{r['token_norm_mean']:.2f} | "
                      f"{r['share_mean']:.4f} [{r['share_ci_lo']:.4f}, {r['share_ci_hi']:.4f}] | "
                      f"{r['per_token_mean']:.4g} [{r['per_token_ci_lo']:.4g}, "
                      f"{r['per_token_ci_hi']:.4g}] |")
        md.append('')

    # --- cross-checkpoint agreement ---
    def share_vec(tag, kind='share'):
        _, per_token, share = derived(records[tag], args.target, args.score)
        return np.nanmean(share if kind == 'share' else per_token, axis=0)

    tags = list(records)
    md += ['## Agreement between checkpoints', '',
           '| pair | Kendall tau (share) | Kendall tau (per token) |', '|---|---|---|']
    for i in range(len(tags)):
        for j in range(i + 1, len(tags)):
            a, b = tags[i], tags[j]
            ts = kendall_tau(share_vec(a), share_vec(b))[0]
            tp = kendall_tau(share_vec(a, 'per_token'), share_vec(b, 'per_token'))[0]
            summary['comparisons'][f'{a}|{b}'] = {'tau_share': ts, 'tau_per_token': tp}
            md.append(f"| {a} vs {b} | {ts:+.3f} | {tp:+.3f} |")
    md.append('')

    # --- controls, if they were run ---
    for ctrl in ('weights', 'labels'):
        ctrl_recs = {t: load(t, ctrl) for t in tags}
        ctrl_recs = {k: v for k, v in ctrl_recs.items() if v is not None}
        if not ctrl_recs:
            continue
        md += [f'## Control: {ctrl}', '',
               'The attribution pattern must change substantially. If it does not, the '
               'scores are not measuring what they claim to.', '',
               '| checkpoint | Kendall tau vs main | max |delta share| |', '|---|---|---|']
        for t, rec in ctrl_recs.items():
            _, _, s_ctrl = derived(rec, args.target, args.score)
            base = share_vec(t)
            ctrl_mean = np.nanmean(s_ctrl, axis=0)
            tau = kendall_tau(base, ctrl_mean)[0]
            summary['comparisons'][f'{t}|control_{ctrl}'] = {
                'tau': tau, 'max_abs_delta_share': float(np.nanmax(np.abs(base - ctrl_mean)))}
            md.append(f"| {t} | {tau:+.3f} | {np.nanmax(np.abs(base - ctrl_mean)):.4f} |")
        md.append('')

    # --- channel decomposition ---
    chan_recs = {t: r for t, r in records.items()
                 if f'chan__{args.target}__pos_code' in r['z']}
    if len(chan_recs) < len(records):
        print(f"note: {len(records) - len(chan_recs)} checkpoint(s) predate the channel "
              "decomposition and are omitted from that section; rerun m1_gradients.py "
              "for them to include it")
    if chan_recs:
        md += ['## Channel decomposition of the signed contribution', '',
               'Every token is an exact sum of named parts (token_layernorm=False), and '
               'grad x input is bilinear, so `<grad, token> = sum_parts <grad, part>` '
               'holds exactly. `pos_code` is one function shared by location, obb, '
               'trajectory **and the decoder\'s 4096 queries**; `class` is shared by obb '
               'and trajectory. The "shared" column is the fraction of a modality\'s '
               'absolute channel mass arriving through those shared parameters -- a high '
               'value means the modality is largely another place the shared code is read, '
               'not an independent source of evidence.', '']
        for tag, rec in chan_recs.items():
            segs, chans, M, total, shared_frac = channel_table(rec, args.target)
            live = [c for ci, c in enumerate(chans) if np.any(np.abs(M[:, ci]) > 0)]
            md += [f'### {tag}', '',
                   '| segment | ' + ' | '.join(live) + ' | total | shared |',
                   '|---|' + '---|' * (len(live) + 2)]
            for s, name in enumerate(segs):
                cells = [f"{M[s, chans.index(c)]:+.4g}" for c in live]
                md.append(f"| {name} | " + " | ".join(cells) +
                          f" | {total[s]:+.4g} | "
                          f"{'' if np.isnan(shared_frac[s]) else f'{shared_frac[s]:.1%}'} |")
            md.append('')
            summary['checkpoints'][tag]['channels'] = {
                'names': chans,
                'by_segment': {name: {c: float(M[s, ci]) for ci, c in enumerate(chans)}
                               for s, name in enumerate(segs)},
                'shared_fraction': {name: (None if np.isnan(shared_frac[s]) else float(shared_frac[s]))
                                    for s, name in enumerate(segs)},
            }
        fig_channels(chan_recs, os.path.join(args.out_dir, "channels.png"), args.target)

    # --- integrated gradients, if run ---
    ig_rows = []
    for tag, rec in records.items():
        p = os.path.join(rec['dir'], 'ig.npz')
        if not os.path.exists(p):
            continue
        z = np.load(p, allow_pickle=False)
        rel = z['completeness_rel_error']
        segs = records[tag]['segments']
        tot = []
        for name in segs:
            idx = _seg_index(rec, name)
            tot.append(float(np.nanmean((z['ig_token'][:, idx] * z['valid'][:, idx]).sum(axis=1))))
        ig_rows.append((tag, int(z['steps']), float(np.median(rel)),
                        float(np.percentile(rel, 90)), dict(zip(segs, tot))))
    if ig_rows:
        md += ['## Integrated Gradients (zero baseline)', '',
               'Completeness is an arithmetic identity: the attributions must sum to '
               'T(input) - T(baseline). A large relative error means too few steps or a bug.', '',
               '| checkpoint | steps | completeness rel. err (median / p90) | ' +
               ' | '.join(records[tags[0]]['segments']) + ' |',
               '|---|---|---|' + '---|' * len(records[tags[0]]['segments'])]
        for tag, steps, med, p90, tot in ig_rows:
            md.append(f"| {tag} | {steps} | {med:.4f} / {p90:.4f} | " +
                      " | ".join(f"{tot[s]:+.4g}" for s in records[tags[0]]['segments']) + " |")
        summary['integrated_gradients'] = [
            {'tag': t, 'steps': s, 'completeness_median': m, 'completeness_p90': p,
             'segment_total': d} for t, s, m, p, d in ig_rows]
        md.append('')

    # --- figures ---
    fig_modality(records, args.target, args.score,
                 os.path.join(args.out_dir, f'modality_{args.score}.png'))
    fig_modality(records, args.target, 'abs_gxi',
                 os.path.join(args.out_dir, 'modality_abs_gxi.png'))
    fig_token_profile(records, os.path.join(args.out_dir, 'token_profile.png'), args.score)
    fig_signed(records, os.path.join(args.out_dir, 'signed_gxi.png'), args.target)
    fig_scale_confound(records, os.path.join(args.out_dir, 'scale_confound.png'),
                       args.target, args.score)
    md += ['## Figures', '',
           f'- `modality_{args.score}.png` -- share of total and per-valid-token mean',
           '- `modality_abs_gxi.png` -- the same for |grad x input|',
           '- `token_profile.png` -- within-modality structure (time profile, box rank)',
           '- `signed_gxi.png` -- which modalities push the target up vs down',
           '- `channels.png` -- which PART of each token carries the contribution',
           '- `scale_confound.png` -- per-token score against token norm', '']

    with open(os.path.join(args.out_dir, 'summary.md'), 'w') as f:
        f.write("\n".join(md))
    common.write_json(os.path.join(args.out_dir, 'summary.json'), summary)
    with open(os.path.join(args.out_dir, 'modality_table.csv'), 'w') as f:
        f.write("\n".join(csv_lines))
    print(f"wrote {args.out_dir}/summary.md, summary.json, modality_table.csv and 5 figures")


if __name__ == '__main__':
    main()
