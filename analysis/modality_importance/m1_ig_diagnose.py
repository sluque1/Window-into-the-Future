"""Why does Integrated Gradients fail its completeness check, and is it fixable?

The first full run reported a median relative completeness error of 0.81 with a
zero baseline and 32 steps -- IG accounted for only ~20% of T(input) - T(baseline).
Completeness is exact arithmetic in the limit of infinitely many steps, so there
are exactly two explanations and they need opposite fixes:

  QUADRATURE  the integrand is nasty and 32 midpoint samples miss most of it.
              Then the error falls as the step count rises, roughly as 1/steps
              for a smooth integrand and more slowly for a peaked one.
  BUG         the implementation is wrong. Then the error does NOT fall with
              more steps -- it sits where it is.

This sweeps both the step count and the baseline and prints the table that tells
them apart. It calls m1_gradients.integrated_gradients itself, so it tests the
code path that actually runs, not a reimplementation of it.

Why the baseline is swept too: scaling the whole encoder input by alpha scales the
attention LOGITS by roughly alpha^2, because queries and keys shrink together. The
straight path out of an all-zero input therefore crosses a regime where attention
is near-uniform and the model is far outside anything it saw in training, and any
concentration of the integrand there is exactly what a coarse midpoint rule
misses. A segment-mean baseline stays inside the data distribution.

    python analysis/modality_importance/m1_ig_diagnose.py --run A --n 16
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402
import runs as runspec  # noqa: E402
from m1_gradients import integrated_gradients, segment_mean_baseline  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', default='A', choices=['A', 'B'])
    ap.add_argument('--which', default='best', choices=['best', 'final'])
    ap.add_argument('--n', type=int, default=16, help="episodes per cell")
    ap.add_argument('--steps', default='8,32,128,512')
    ap.add_argument('--baselines', default='zero,segment_mean')
    ap.add_argument('--batch_size', type=int, default=4)
    ap.add_argument('--device', default='')
    args = ap.parse_args()

    device = torch.device(args.device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    common.set_seed(0)
    steps_list = [int(s) for s in args.steps.split(',')]
    baselines = [b.strip() for b in args.baselines.split(',')]

    path = runspec.checkpoint_dir(args.run, args.which)
    model, config = common.load_checkpoint(path, device)
    dataset = common.build_test_dataset(args.run, config, verbose=False)
    layout = common.read_layout(model, dataset, device)
    idx = list(range(min(args.n, len(dataset))))
    print(f"\n{runspec.tag(args.run, args.which)}  device={device}  {len(idx)} episodes")

    base_tensors = {'zero': None}
    if 'segment_mean' in baselines:
        print("building the segment-mean baseline over 64 episodes ...")
        base_tensors['segment_mean'] = segment_mean_baseline(model, dataset, layout, device)

    print(f"\n{'baseline':14s} {'steps':>6s} {'median rel err':>15s} {'p90':>9s} "
          f"{'|T(x)-T(b)|':>13s}")
    results = {}
    for bname in baselines:
        prev = None
        for steps in steps_list:
            out = integrated_gradients(model, dataset, layout, idx, steps,
                                       args.batch_size, device,
                                       baseline=base_tensors[bname])
            rel = out['completeness_rel_error']
            med = float(np.median(rel))
            delta = float(np.abs(out['completeness_delta']).mean())
            trend = '' if prev is None else f"  ({med / prev:.2f}x previous)"
            print(f"{bname:14s} {steps:6d} {med:15.4f} "
                  f"{float(np.percentile(rel, 90)):9.4f} {delta:13.4f}{trend}")
            results[(bname, steps)] = med
            prev = med

    print("\nreading the table:")
    print("  error falling steadily with steps  -> QUADRATURE; raise steps or change baseline")
    print("  error flat across steps            -> BUG in the IG implementation")
    print("  |T(x)-T(b)| near zero              -> the identity is ill-conditioned for that")
    print("                                        baseline: a tiny denominator inflates the")
    print("                                        RELATIVE error even when the absolute gap")
    print("                                        is small, and the ratio is then meaningless")
    for bname in baselines:
        vals = [results[(bname, s)] for s in steps_list]
        verdict = ('falls with steps -> quadrature' if vals[-1] < 0.5 * vals[0]
                   else 'flat -> not a step-count problem')
        print(f"  {bname:14s} {vals[0]:.4f} -> {vals[-1]:.4f}   {verdict}")


if __name__ == '__main__':
    main()
