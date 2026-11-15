"""Is the model learning a transferable spatial prior, or one kitchen's floor plan? (spec §7b)

Averages the predicted probability map over the test episodes, then marginalises
it over each axis independently: three curves of 16 points. Does the same for the
ground-truth label map, so the two can be read against each other.

How to read it, and why the axes are not equivalent:

  * `z` is clean. The world frame is gravity-aligned, so normalised height means
    the same thing in every kitchen -- floor, counter / interaction band, upper
    cabinets. Structure here is EXPECTED AND HEALTHY: "interactions happen at
    counter level" is exactly the kind of prior that should transfer.

  * `x` and `y` are not. Each participant's grid was manually rotated about z to
    hug that kitchen's footprint, which aligns the axes to walls but does not fix
    WHICH wall is +x -- the eight symmetries of a rectangle remain, and the long
    axis really is inconsistent (4 participants x-long, 5 y-long, measured).
    So sharp structure in x or y is a horizontal layout memorised from the eight
    training kitchens, and it cannot be right for the held-out one.

Worth running on every trained model including the baselines -- it needs no
attention weights and no re-training, only a checkpoint's saved dumps. The
comparison it was written for is --voxel_bias full against factorised: if the
reasoning in spec §4.3 holds, `full` shows visibly more x/y structure.

Usage:
    python analysis/axis_marginals.py results/<run>/checkpoint-1234 [more...] \
        [--labels] [--out analysis/out/axis_marginals]

Multiple checkpoints are overlaid on one plot, which is the point -- the shape of
one curve alone is hard to judge.
"""
import argparse
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

GRID = 16
NUM_VOXELS = GRID ** 3
AXES = ('x (grid axis 0)', 'y (grid axis 1)', 'z (height, axis 2)')
OUT_DEFAULT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           'out', 'axis_marginals')


def load_maps(ckpt_dir):
    """(mean predicted probability map [4096], mean label map [4096], n_episodes).

    predictions.pkl holds raw logits [N, 4096] and labels.pkl a dict with
    'interaction' [N, 4096] -- written per checkpoint by trainer.py::_save.
    """
    for name in ('predictions.pkl', 'labels.pkl'):
        if not os.path.exists(os.path.join(ckpt_dir, name)):
            raise SystemExit(f"{ckpt_dir} has no {name}. Point this at a checkpoint "
                             "directory written by CustomTrainer._save.")
    with open(os.path.join(ckpt_dir, 'predictions.pkl'), 'rb') as f:
        logits = pickle.load(f)
    if isinstance(logits, tuple):
        logits = logits[0]
    with open(os.path.join(ckpt_dir, 'labels.pkl'), 'rb') as f:
        labels = pickle.load(f)['interaction']
    logits = np.asarray(logits, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)
    if logits.shape[1] != NUM_VOXELS:
        raise SystemExit(f"{ckpt_dir}: predictions are {logits.shape}, expected "
                         f"(N, {NUM_VOXELS})")
    probs = 1.0 / (1.0 + np.exp(-logits))
    return probs.mean(axis=0), labels.mean(axis=0), len(logits)


def marginals(flat_map):
    """Three curves of 16 points: the map averaged over the other two axes.

    The reshape is C order over (16,16,16) -- flat = i*256 + j*16 + k, the same
    layout labels['interaction'] and the decoder's queries use.
    """
    cube = np.asarray(flat_map).reshape(GRID, GRID, GRID)
    return [cube.mean(axis=(1, 2)), cube.mean(axis=(0, 2)), cube.mean(axis=(0, 1))]


def structure_score(curve):
    """Coefficient of variation across the 16 bins: 0 = flat, higher = peakier.

    Scale-free on purpose, so a model whose mean probability is 0.004 and one at
    0.05 can be compared on shape alone -- which is the only thing that answers
    "did it memorise a layout?"
    """
    curve = np.asarray(curve, dtype=np.float64)
    return float(curve.std() / curve.mean()) if curve.mean() > 0 else float('nan')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('checkpoints', nargs='+',
                    help="checkpoint directories holding predictions.pkl/labels.pkl")
    ap.add_argument('--labels', action='store_true',
                    help="Overlay the ground-truth marginals from the first checkpoint.")
    ap.add_argument('--out', type=str, default=OUT_DEFAULT)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    rows, label_curves = [], None
    for ckpt in args.checkpoints:
        pred_map, label_map, n = load_maps(ckpt)
        curves = marginals(pred_map)
        if label_curves is None:
            label_curves = marginals(label_map)
        name = os.path.basename(os.path.normpath(ckpt))
        run = os.path.basename(os.path.dirname(os.path.normpath(ckpt)))
        rows.append((f"{run}/{name}", curves, n))
        print(f"{run}/{name}  n={n}")
        for axis_name, curve in zip(AXES, curves):
            print(f"    {axis_name:<22} CV={structure_score(curve):.3f}  "
                  f"min={curve.min():.5f} max={curve.max():.5f} "
                  f"argmax={int(curve.argmax())}")

    print("\nground truth (same test set)")
    for axis_name, curve in zip(AXES, label_curves):
        print(f"    {axis_name:<22} CV={structure_score(curve):.3f}  "
              f"argmax={int(curve.argmax())}")
    print("\nRead z against the GT z curve: tracking it is the prior that should "
          "transfer.\nStructure in x or y is a training-kitchen layout and cannot "
          "transfer -- the yaw\nfixes the axes to walls but not which wall is +x.")

    csv_path = os.path.join(args.out, 'axis_marginals.csv')
    with open(csv_path, 'w') as f:
        f.write("run,axis,bin,mean_value,cv\n")
        for name, curves, _ in rows:
            for axis_name, curve in zip(AXES, curves):
                cv = structure_score(curve)
                for b, value in enumerate(curve):
                    f.write(f"{name},{axis_name},{b},{value:.8f},{cv:.6f}\n")
        for axis_name, curve in zip(AXES, label_curves):
            cv = structure_score(curve)
            for b, value in enumerate(curve):
                f.write(f"ground_truth,{axis_name},{b},{value:.8f},{cv:.6f}\n")

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print(f"\nWrote {csv_path} (matplotlib unavailable, no plot)")
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), sharex=True)
    for ax, axis_name, gt in zip(axes, AXES, label_curves):
        for name, curves, _ in rows:
            curve = curves[AXES.index(axis_name)]
            ax.plot(range(GRID), curve / curve.mean(), marker='o', ms=3, label=name)
        if args.labels:
            ax.plot(range(GRID), gt / gt.mean(), 'k--', lw=2, label='ground truth')
        ax.set_title(axis_name)
        ax.set_xlabel('grid index')
        ax.axhline(1.0, color='0.8', lw=1, zorder=0)
    axes[0].set_ylabel('mean value / its own mean')
    axes[-1].legend(fontsize=7)
    fig.suptitle('Axis marginals -- structure in z should transfer, structure in x/y cannot')
    fig.tight_layout()
    png = os.path.join(args.out, 'axis_marginals.png')
    fig.savefig(png, dpi=150)
    print(f"\nWrote {csv_path}\nWrote {png}")


if __name__ == '__main__':
    main()
