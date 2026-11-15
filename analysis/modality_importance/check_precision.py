"""Why our fp32 logits differ from the run's saved predictions.pkl.

The training jobs ran with ``bf16=True`` (read from training_args.bin), and HF
Trainer applies that autocast to the prediction loop as well as to training. So
predictions.pkl holds bfloat16-precision numbers while this analysis recomputes
in float32. bfloat16 keeps 8 mantissa bits (~3 significant digits) against
float32's 24 (~7), so a 1-2% disagreement on logits of magnitude 7-14 is the
expected consequence, not a reconstruction error.

This script measures that rather than asserting it: it evaluates the same
episodes twice, once in float32 and once under bfloat16 autocast, and compares
BOTH against the saved predictions. If the bf16 pass lands far closer, the
difference is precision. If neither does, the reconstruction really is wrong and
the whole analysis is invalid.

    python analysis/modality_importance/check_precision.py --run A --n 32
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402
import runs as runspec  # noqa: E402


def logits_for(model, dataset, indices, device, dtype=None):
    out = []
    for b0 in range(0, len(indices), 8):
        sel = indices[b0:b0 + 8]
        batch = common.collate(dataset, sel, device)
        batch.pop('labels', None)
        with torch.no_grad():
            if dtype is None:
                o = model(**batch)
            else:
                with torch.autocast(device_type=device.type, dtype=dtype):
                    o = model(**batch)
        out.append(o['interaction'].float().cpu().numpy())
    return np.concatenate(out, axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', default='A', choices=['A', 'B'])
    ap.add_argument('--which', default='best', choices=['best', 'final'])
    ap.add_argument('--n', type=int, default=32)
    ap.add_argument('--device', default='')
    args = ap.parse_args()

    device = torch.device(args.device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    common.set_seed(0)
    path = runspec.checkpoint_dir(args.run, args.which)
    model, config = common.load_checkpoint(path, device)
    dataset = common.build_test_dataset(args.run, config, verbose=False)
    idx = list(range(min(args.n, len(dataset))))
    saved = common.saved_predictions(args.run, args.which)[idx]

    print(f"\n{runspec.tag(args.run, args.which)}  device={device}  {len(idx)} episodes")
    print(f"saved predictions.pkl: range [{saved.min():.3f}, {saved.max():.3f}], "
          f"mean |logit| {np.abs(saved).mean():.3f}")

    rows = []
    got32 = logits_for(model, dataset, idx, device, None)
    rows.append(('float32', got32))
    try:
        got16 = logits_for(model, dataset, idx, device, torch.bfloat16)
        rows.append(('bfloat16 autocast', got16))
    except RuntimeError as e:
        print(f"bfloat16 autocast unavailable on {device}: {e}")

    print(f"\n{'pass':22s} {'max|delta|':>12s} {'mean|delta|':>12s} "
          f"{'rel. to |logit|':>16s} {'corr':>8s}")
    for name, got in rows:
        d = np.abs(got - saved)
        rel = d.mean() / np.abs(saved).mean()
        corr = np.corrcoef(got.ravel(), saved.ravel())[0, 1]
        print(f"{name:22s} {d.max():12.5f} {d.mean():12.5f} {rel:15.4%} {corr:8.6f}")

    if len(rows) == 2:
        a = np.abs(rows[0][1] - saved).mean()
        b = np.abs(rows[1][1] - saved).mean()
        print(f"\nbfloat16 is {a / max(b, 1e-12):.1f}x closer to the saved predictions "
              f"than float32 is.")
        print("If that factor is large, the disagreement is the original run's own "
              "bf16 precision,\nnot a reconstruction error -- the weights and the data "
              "match.")
    # The ranking of voxels is what PR-AUC reads, so check it survived precision.
    from scipy.stats import spearmanr
    rho = np.mean([spearmanr(got32[i], saved[i]).statistic for i in range(len(idx))])
    print(f"\nper-episode Spearman(fp32, saved) over the 4096 voxels: {rho:.6f}")
    print("A value near 1 means the two agree on the voxel ORDERING, i.e. they would "
          "score\nthe same PR-AUC, and the attribution runs on the same model behaviour.")


if __name__ == '__main__':
    main()
