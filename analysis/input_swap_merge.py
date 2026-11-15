"""Merge per-fold input_swap_ablation.py JSONs into one table.

The per-fold JSONs come from analysis/input_swap_sbatch.sh, which runs one fold
per SLURM task. Folds are reported as a PERCENTAGE of each fold's own unablated
score rather than pooled: per-episode PR-AUC is prevalence-sensitive and the
voxel grids are fitted per kitchen (CLAUDE.md), so an absolute mean over folds is
dominated by the densest ones. A ratio within a fold is not.

Each argument is either a directory of per-fold JSONs (what input_swap_sbatch.sh
writes) or a single combined JSON (what one interactive invocation writes).

Usage:
    python analysis/input_swap_merge.py analysis/out/input_swap/<tag>
    python analysis/input_swap_merge.py <dir_or_json_a> <dir_or_json_b>   # two arms
"""
import glob
import json
import os
import sys

import numpy as np

ORDER = ['video', 'location', 'gaze', 'env', 'obb', 'all']


def load(target):
    """Rows from a directory of per-fold JSONs, or from a single combined JSON.

    Both layouts exist: input_swap_sbatch.sh writes one file per fold into a
    directory, while a single interactive invocation over several folds writes
    them all to one file. Each file holds a LIST of fold records either way, so
    the only difference is how many paths to read."""
    paths = ([target] if os.path.isfile(target)
             else sorted(glob.glob(os.path.join(target, '*.json'))))
    rows = [entry for path in paths for entry in json.load(open(path))]
    return sorted(rows, key=lambda r: r['participant'])


def table(rows, label, only=None):
    if only is not None:
        rows = [r for r in rows if r['participant'] in only]
    arms = [a for a in ORDER if any(a in r['arms'] for r in rows)]
    print(f"\n=== {label} -- {len(rows)} fold(s), PR-AUC as % of that fold's base ===")
    head = f"{'fold':<6} {'base':>8} " + ' '.join(f"{'swap ' + a:>12}" for a in arms)
    print(head + f" {'static-self':>12}")
    print('-' * len(head + '             '))
    for r in rows:
        base = r['arms']['base']
        cells = ' '.join(f"{100 * r['arms'][a] / base:11.1f}%" if a in r['arms']
                         else f"{'-':>12}" for a in arms)
        print(f"{r['participant']:<6} {base:8.4f} {cells} "
              f"{100 * r['arms']['static-self'] / base:11.1f}%")
    print('-' * len(head + '             '))
    means = {a: np.mean([100 * r['arms'][a] / r['arms']['base']
                         for r in rows if a in r['arms']]) for a in arms}
    means['static-self'] = np.mean(
        [100 * r['arms']['static-self'] / r['arms']['base'] for r in rows])
    print(f"{'mean':<6} {np.mean([r['arms']['base'] for r in rows]):8.4f} "
          + ' '.join(f"{means[a]:11.1f}%" for a in arms)
          + f" {means['static-self']:11.1f}%")
    return means


def main():
    targets = sys.argv[1:]
    if not targets:
        raise SystemExit(__doc__)
    loaded = {}
    for t in targets:
        rows = load(t)
        if not rows:
            print(f"{t}: no JSONs yet")
            continue
        loaded[os.path.basename(os.path.normpath(t))] = rows
        table(rows, t)
    if len(loaded) > 1:
        # Arms are only comparable on the folds they SHARE. Averaging a nine-fold
        # arm against a four-fold one compares different kitchens, and per-fold
        # PR-AUC varies more between participants (.039 to .311) than any arm
        # moves it -- so the intersection is the only honest mean.
        common = set.intersection(*({r['participant'] for r in rows}
                                    for rows in loaded.values()))
        print(f"\n=== arms side by side, {len(common)} shared fold(s): "
              f"{', '.join(sorted(common))} ===")
        means = {name: table(rows, f"{name} (shared folds only)", only=common)
                 for name, rows in loaded.items()}
        arms = [a for a in ORDER + ['static-self'] if any(a in m for m in means.values())]
        print(f"\n{'':<40} " + ' '.join(f"{a:>12}" for a in arms))
        for name, m in means.items():
            print(f"{name:<40} " + ' '.join(
                f"{m[a]:11.1f}%" if a in m else f"{'-':>12}" for a in arms))


if __name__ == '__main__':
    main()
