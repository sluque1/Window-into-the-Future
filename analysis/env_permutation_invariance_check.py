"""
Sanity check: does the ORDER of the 4096 env tokens affect the model's output?

Context
-------
The env segment is built by ``class_embedding(env_grid)`` (model.py) and
concatenated into the transformer's input sequence with no positional
encoding added anywhere in the model (video, location, and env segments are
all raw content, no per-slot position vector). A vanilla
``nn.TransformerEncoderLayer`` computes, at every position, an
attention-weighted SUM over all tokens' value vectors -- and a sum's value
does not depend on the order of its terms. So the claim under test is: with
no positional encoding, permuting the env segment before the transformer call
should leave the output logits EXACTLY unchanged (up to float rounding),
proving the model currently has no way to know *which* voxel any given env
token came from.

Method
------
1. Build a randomly-initialised CustomTransformerModel (weights don't matter
   -- this is a structural property of the architecture, not something
   learned) in eval() mode (disables dropout, which would otherwise make two
   forward passes differ for reasons unrelated to token order).
2. Build one synthetic batch matching the shapes CustomTransformerModel.forward
   expects (see model.py docstring): visual_features, visual_mask, env_grid,
   person_locations, person_location_mask.
3. Run the forward pass once as-is, and once with the env_grid's 4096 voxel
   slots permuted by a random permutation (same permutation applied to every
   sample in the batch). Everything else in the batch is untouched.
4. Compare the two logit tensors. If the architecture truly has no positional
   encoding for the env segment, they must be bit-for-bit identical (modulo
   floating point non-determinism, which we tolerate with a tight atol).

Usage
-----
  python analysis/env_permutation_invariance_check.py
  python analysis/env_permutation_invariance_check.py --seed 123 --batch_size 8
"""

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model import CustomConfig, CustomTransformerModel  # noqa: E402
from object_class_vocab import NUM_OBJECT_CLASSES  # noqa: E402


def build_random_batch(batch_size, obs_time, num_voxels, num_object_classes,
                        location_vocab_size, generator):
    """Synthetic batch matching CustomTransformerModel.forward's expected shapes."""
    visual_features = torch.randn(batch_size, obs_time, 4096, generator=generator)
    visual_mask = torch.ones(batch_size, obs_time, dtype=torch.int32)
    env_grid = torch.randint(0, num_object_classes, (batch_size, num_voxels),
                              generator=generator)
    person_locations = torch.randint(0, location_vocab_size, (batch_size, obs_time, 3),
                                      generator=generator)
    person_location_mask = torch.ones(batch_size, obs_time, dtype=torch.int32)
    index = torch.arange(batch_size)
    return dict(
        index=index,
        visual_features=visual_features,
        visual_mask=visual_mask,
        env_grid=env_grid,
        person_locations=person_locations,
        person_location_mask=person_location_mask,
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--obs_time", type=int, default=30)
    ap.add_argument("--num_voxels", type=int, default=4096)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--atol", type=float, default=1e-5,
                    help="Max allowed abs diff between original and permuted logits.")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    generator = torch.Generator().manual_seed(args.seed)

    config = CustomConfig(
        visual_feature_dim=4096,
        transformer_hidden_size=768,
        num_transformer_layers=2,   # depth doesn't matter for this property; keep it cheap
        num_heads=8,
        output_dim=args.num_voxels,
        num_object_classes=NUM_OBJECT_CLASSES,
        observation_time=args.obs_time,
        location_vocab_size=17,
        use_static_env=True,
        input_modality='video,location,env',
    )
    model = CustomTransformerModel(config)
    model.eval()   # disable dropout -- otherwise the two passes would legitimately differ

    batch = build_random_batch(args.batch_size, args.obs_time, args.num_voxels,
                                config.num_object_classes, config.location_vocab_size,
                                generator)

    perm = torch.randperm(args.num_voxels, generator=generator)
    permuted_batch = dict(batch)
    permuted_batch['env_grid'] = batch['env_grid'][:, perm]

    with torch.no_grad():
        logits_orig = model(**batch)['interaction']
        logits_perm = model(**permuted_batch)['interaction']

    max_abs_diff = (logits_orig - logits_perm).abs().max().item()
    identical = torch.allclose(logits_orig, logits_perm, atol=args.atol)

    print("=" * 78)
    print("ENV-SEGMENT PERMUTATION INVARIANCE CHECK")
    print("=" * 78)
    print(f"  batch_size={args.batch_size}  obs_time={args.obs_time}  "
          f"num_voxels={args.num_voxels}  seed={args.seed}")
    print(f"  logits shape           : {tuple(logits_orig.shape)}")
    print(f"  max |orig - permuted|  : {max_abs_diff:.3e}")
    print(f"  allclose (atol={args.atol:g}) : {identical}")
    print()
    if identical:
        print("  -> Logits are UNCHANGED by permuting the env segment's 4096 slots.")
        print("     Confirms: the model has NO way to know which voxel an env token")
        print("     came from -- order carries zero spatial information today.")
    else:
        print("  -> Logits CHANGED when the env segment was permuted.")
        print("     Order does carry information (unexpected given the architecture --")
        print("     re-check for an unnoticed positional signal, or float non-determinism).")
    print("=" * 78)


if __name__ == "__main__":
    main()
