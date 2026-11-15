"""Summarise nn_probe.py's per-participant JSONs.

The number that matters is never an arm's raw PR-AUC -- it is the arm MINUS
NN-random at the same k. Every NN arm predicts with a real episode's label map, so
every one of them beats chance simply by being a plausible map of this kitchen,
whether or not the similarity search did anything. Subtracting the random control
removes that and leaves what the matching signal was worth.

Read NN-video only after checking NN-obb in the same row: obb is worth 18.5% by
the swap ablation, so a participant where NN-obb fails to clear NN-random has an
uninformative probe, and its NN-video value means nothing either way.

Usage:  python analysis/nn_probe_summary.py [analysis/out/nn_probe]
"""
import glob
import json
import os
import sys

import numpy as np

ARMS = ['video', 'obb', 'location']


def main():
    d = sys.argv[1] if len(sys.argv) > 1 else 'analysis/out/nn_probe'
    rows = []
    for path in sorted(glob.glob(os.path.join(d, 'P0*.json'))):
        rows += json.load(open(path))
    if not rows:
        raise SystemExit(f"no P0*.json in {d}")
    rows.sort(key=lambda r: r['participant'])

    ks = sorted({int(name.split('k=')[1].split()[0])
                 for name in rows[0]['scores'] if 'k=' in name})

    # --- The raw ladder: every scorer, every participant, no arithmetic -------
    # Ordered floor-upward so the steps are visible: chance knows nothing;
    # NN-random already gains a lot purely by predicting with SOME real map of
    # this kitchen; the NN-<signal> arms add whatever choosing the neighbour is
    # worth; oracle-map is the best any single fixed map of the kitchen can do;
    # and SAME-VIDEO-OK shows the temporal artefact the exclusion removes.
    order = ['chance']
    for k in ks:
        order += [f'NN-random k={k}'] + [f'NN-{a} k={k}' for a in ARMS]
    order += [f'NN-video k={k} SAME-VIDEO-OK' for k in ks] + ['oracle-map']
    order = [n for n in order if n in rows[0]['scores']]
    print("=== full ladder: PR-AUC of every scorer, per participant ===")
    head = f"{'scorer':<30} " + ' '.join(f"{r['participant']:>8}" for r in rows)
    print(head)
    print('-' * len(head))
    for name in order:
        print(f"{name:<30} " + ' '.join(f"{r['scores'][name]:8.4f}" for r in rows))

    for k in ks:
        rnd = f'NN-random k={k}'
        print(f"\n=== k={k}: PR-AUC, and the lift over NN-random (the control) ===")
        head = (f"{'fold':<6} {'ep':>5} {'vid':>5} {'chance':>7} {'NN-rand':>8} "
                + ' '.join(f"{'NN-' + a:>8}" for a in ARMS)
                + '   ' + ' '.join(f"{'+' + a:>8}" for a in ARMS)
                + f" {'oracle':>8}")
        print(head)
        print('-' * len(head))
        lifts = {a: [] for a in ARMS}
        for r in rows:
            s = r['scores']
            base = s[rnd]
            for a in ARMS:
                lifts[a].append(s[f'NN-{a} k={k}'] - base)
            print(f"{r['participant']:<6} {r['n_episodes']:5d} {r['n_videos']:5d} "
                  f"{s['chance']:7.4f} {base:8.4f} "
                  + ' '.join(f"{s[f'NN-{a} k={k}']:8.4f}" for a in ARMS)
                  + '   ' + ' '.join(f"{s[f'NN-{a} k={k}'] - base:+8.4f}" for a in ARMS)
                  + f" {s['oracle-map']:8.4f}")
        print('-' * len(head))
        print(f"{'mean':<6} {'':>5} {'':>5} {'':>7} {'':>8} " + ' ' * (9 * len(ARMS))
              + '  ' + ' '.join(f"{np.mean(lifts[a]):+8.4f}" for a in ARMS))
        for a in ARMS:
            n_pos = sum(x > 0 for x in lifts[a])
            print(f"  NN-{a:<9} clears the control on {n_pos}/{len(rows)} participants"
                  f"   (min {min(lifts[a]):+.4f}, max {max(lifts[a]):+.4f})")

    # --- Redundancy: does either signal add anything GIVEN the other? -------
    # Each conditional arm is scored against a random draw from ITS OWN pool, not
    # against the unconditional arms. The pool is already similar in the primary
    # signal, so it scores well however the k are drawn from it; only the
    # within-pool difference isolates the secondary signal's marginal value.
    # Also shown: NN-combined against the best single arm. If all three signals
    # are views of one variable, combining them adds nothing over the best of them.
    if all(f'NN-video|obb k={ks[0]}' in r['scores'] for r in rows):
        for k in ks:
            print(f"\n=== k={k}: marginal value of each signal GIVEN the other ===")
            head = (f"{'fold':<6} {'rnd|obb':>9} {'vid|obb':>9} {'+video':>9}   "
                    f"{'rnd|vid':>9} {'obb|vid':>9} {'+obb':>9}   "
                    f"{'combined':>9} {'best-1':>8} {'+comb':>8}")
            print(head); print('-' * len(head))
            gv, go, gc = [], [], []
            for r in rows:
                s = r['scores']
                ro, vo = s[f'NN-random|obb k={k}'], s[f'NN-video|obb k={k}']
                rv, ov = s[f'NN-random|video k={k}'], s[f'NN-obb|video k={k}']
                comb = s[f'NN-combined k={k}']
                best = max(s[f'NN-{a} k={k}'] for a in ARMS)
                gv.append(vo - ro); go.append(ov - rv); gc.append(comb - best)
                print(f"{r['participant']:<6} {ro:9.4f} {vo:9.4f} {vo - ro:+9.4f}   "
                      f"{rv:9.4f} {ov:9.4f} {ov - rv:+9.4f}   "
                      f"{comb:9.4f} {best:8.4f} {comb - best:+8.4f}")
            print('-' * len(head))
            print(f"{'mean':<6} {'':>9} {'':>9} {np.mean(gv):+9.4f}   "
                  f"{'':>9} {'':>9} {np.mean(go):+9.4f}   "
                  f"{'':>9} {'':>8} {np.mean(gc):+8.4f}")
            print(f"  video adds given obb on {sum(x > 0 for x in gv)}/{len(rows)}, "
                  f"obb adds given video on {sum(x > 0 for x in go)}/{len(rows)}, "
                  f"combined beats the best single on {sum(x > 0 for x in gc)}/{len(rows)}")

    # --- Was the "nearest" neighbour actually near? -------------------------
    # A lift over NN-random says the ranking helped. It does not say the top-k
    # were similar in any absolute sense: if every eligible pair sat at the same
    # cosine similarity, "nearest" would be arbitrary. top1 well above median
    # means the search found something; top1 ~ median means it did not, and that
    # arm's flat result says nothing about whether the features encode anything.
    if all('similarity' in r for r in rows):
        print("\n=== how similar the chosen neighbour actually was ===")
        head = (f"{'fold':<6} " + ' '.join(
            f"{a + ' top1/med':>18}" for a in ARMS))
        print(head); print('-' * len(head))
        for r in rows:
            print(f"{r['participant']:<6} " + ' '.join(
                f"{r['similarity'][a]['top1']:8.3f}/{r['similarity'][a]['median']:<9.3f}"
                for a in ARMS))

    # --- Dose-response: does MORE similarity buy MORE accuracy? -------------
    # The decisive check. A positive mean lift with a flat dose-response means
    # the lift is not coming from similarity, and something else is carrying it.
    if all(r.get('dose_response') for r in rows):
        keys = sorted(rows[0]['dose_response'], key=lambda s: (s.split()[0], s))
        print("\n=== (AP - NN-random AP on the same episode) by top-1 similarity")
        print("    quartile: Q4-Q1 > 0 means a nearer neighbour predicts better ===")
        head = f"{'fold':<6} " + ' '.join(f"{k:>16}" for k in keys)
        print(head); print('-' * len(head))
        for r in rows:
            print(f"{r['participant']:<6} " + ' '.join(
                f"{r['dose_response'][k][3] - r['dose_response'][k][0]:+16.4f}"
                for k in keys))
        print('-' * len(head))
        print(f"{'mean':<6} " + ' '.join(
            f"{np.mean([r['dose_response'][k][3] - r['dose_response'][k][0] for r in rows]):+16.4f}"
            for k in keys))
        print(f"{'n>0':<6} " + ' '.join(
            f"{sum(r['dose_response'][k][3] > r['dose_response'][k][0] for r in rows):>13}/{len(rows)}"
            for k in keys))

    # The artefact the same-video exclusion exists to prevent, for the record.
    k0 = ks[0]
    leak = [r['scores'].get(f'NN-video k={k0} SAME-VIDEO-OK') for r in rows]
    if all(x is not None for x in leak):
        clean = [r['scores'][f'NN-video k={k0}'] for r in rows]
        print(f"\nSame-video neighbours allowed (k={k0}): "
              f"{np.mean(leak):.4f} vs {np.mean(clean):.4f} with the exclusion "
              f"-- {np.mean(leak) / max(np.mean(clean), 1e-9):.1f}x. That gap is "
              "label overlap between episodes seconds apart, not visual similarity.")


if __name__ == '__main__':
    main()
