"""Data-free smoke test: every checkpoint loads strictly and lays tokens out identically.

No dataset, no GPU, ~1 minute on CPU. Run this before anything expensive -- it is
what catches a wrong checkpoint path, a config/weights mismatch, or a layout that
differs between the four sets of weights.

    python analysis/modality_importance/verify_layout.py
"""
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402
import runs as runspec  # noqa: E402


def synthetic_batch(config, batch_size=2, seed=0):
    """Shape-correct tensors with in-range indices. Values are meaningless -- this
    test is about layout and loading, never about numbers."""
    g = torch.Generator().manual_seed(seed)
    T = config.observation_time
    n_obb = config.max_obb_tokens
    n_traj = 16
    return dict(
        index=torch.arange(batch_size),
        visual_features=torch.randn(batch_size, T, config.visual_feature_dim, generator=g),
        visual_mask=torch.ones(batch_size, T, dtype=torch.int32),
        person_locations=torch.randint(1, 17, (batch_size, T, 3), generator=g),
        person_location_mask=torch.ones(batch_size, T, dtype=torch.int32),
        obb_voxels=torch.randint(1, 17, (batch_size, n_obb, 3), generator=g),
        obb_volumes=torch.randint(1, 17, (batch_size, n_obb), generator=g),
        obb_classes=torch.randint(1, 1221, (batch_size, n_obb), generator=g),
        obb_mask=torch.ones(batch_size, n_obb, dtype=torch.bool),
        trajectory_tokens=torch.randn(batch_size, n_traj, config.trajectory_raw_dim, generator=g),
        trajectory_mask=torch.ones(batch_size, n_traj, dtype=torch.int64),
        trajectory_classes=torch.randint(1, 1221, (batch_size, n_traj), generator=g),
        trajectory_start_voxel=torch.randint(1, 17, (batch_size, n_traj, 3), generator=g),
        trajectory_end_voxel=torch.randint(1, 17, (batch_size, n_traj, 3), generator=g),
        trajectory_distance=torch.randint(1, 17, (batch_size, n_traj), generator=g),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=os.path.join(common.OUT_BASE, 'layout.json'))
    args = ap.parse_args()

    common.set_seed(0)
    report, reference = {}, None
    failures = []

    for run, which in runspec.CHECKPOINT_KEYS:
        tag = runspec.tag(run, which)
        path = runspec.checkpoint_dir(run, which)
        print(f"\n=== {tag}\n    {path}")
        model, config = common.load_checkpoint(path)
        print(f"    loaded strictly: {sum(p.numel() for p in model.parameters()):,} parameters")
        print(f"    modality={config.input_modality} readout={config.readout} "
              f"decoder={config.decoder} query={config.voxel_query_type}")

        batch = synthetic_batch(config)
        with torch.no_grad():
            out = model(**batch, need_attention=True)
        layout = common.TokenLayout(out['segments'], out['attention'].shape[-1])
        print(layout.describe())

        # The gradient path must exist in eval mode: this is the check that
        # someone did not leave a torch.no_grad() on the analysis path.
        assert model.training is False, "model is not in eval mode"
        fake_labels = torch.zeros_like(out['interaction'], dtype=torch.long)
        fake_labels[:, :20] = 1
        with common.Capture(model) as cap:
            out2 = model(**batch)
            tgt = common.compute_targets(out2['interaction'], fake_labels)
            grad, = torch.autograd.grad(tgt[common.PRIMARY_TARGET].sum(), cap.encoder_input)
        assert grad.shape == cap.encoder_input.shape, "gradient shape mismatch"
        assert torch.isfinite(grad).all(), "non-finite gradient"
        assert cap.pad_mask is not None and cap.pad_mask.shape == (2, layout.seq_len), \
            f"pad mask shape {None if cap.pad_mask is None else tuple(cap.pad_mask.shape)}"
        print(f"    gradient at encoder input: {tuple(grad.shape)}  "
              f"L2={grad.norm():.4g}   pad_mask {tuple(cap.pad_mask.shape)}")

        report[tag] = {
            'checkpoint': path,
            'layout': layout.as_dict(),
            'n_parameters': sum(p.numel() for p in model.parameters()),
            'input_modality': config.input_modality,
            'readout': config.readout,
            'decoder': config.decoder,
            'observation_time': config.observation_time,
            'max_obb_tokens': config.max_obb_tokens,
        }
        if reference is None:
            reference = (tag, layout.as_dict())
        elif layout.as_dict() != reference[1]:
            failures.append(f"{tag} has a different token layout from {reference[0]}")

    print("\n" + "=" * 72)
    if failures:
        for f in failures:
            print("FAIL:", f)
        raise SystemExit(1)
    print(f"OK -- all {len(report)} checkpoints load strictly and share one token layout")
    for name, start, end in reference[1]['spans']:
        print(f"  [{start:4d}, {end:4d})  {name:12s} n={end - start}")
    common.write_json(args.out, {'environment': common.environment_manifest(),
                                 'checkpoints': report})
    print(f"\nwrote {args.out}")


if __name__ == '__main__':
    main()
