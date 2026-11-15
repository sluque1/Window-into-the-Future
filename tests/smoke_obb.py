"""Regression test: does the OBB embedding actually CARRY its three signals?

Run: python smoke_obb.py [token0|cls]   (pure torch, no dataset needed, ~10s)

The optional argument is the --readout setting (default token0). None of the
checks below depend on a token's absolute position in the sequence -- they test
sensitivity, masking, order-invariance and table sharing -- so all five must
hold under either readout, and the argument exists to prove that rather than to
assume it.

Finite loss proves nothing -- a token that is computed but constant, or wiped
by the padding mask, still trains. These checks test the real properties:

  1. SENSITIVITY   changing each of {voxel, volume, class} independently must
                   change the logits. If one does not, that component is dead.
  2. GRADIENTS     every table the modality claims to use must receive a
                   non-zero gradient, and volume/obb_type must receive one ONLY
                   via this modality.
  3. PAD ISOLATION rewriting the *padding* slots must NOT change the logits.
  4. PERMUTATION   reordering real tokens must NOT change the logits (there is
                   no positional encoding; order must be meaningless).
  5. SHARED TABLES an OBB at voxel v must move when location_embedding_x row v
                   moves -- proving it really uses the shared spatial tables.
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from boxer_env import MAX_OBB_TOKENS  # noqa: E402
from model import CustomConfig, CustomTransformerModel  # noqa: E402
from object_class_vocab import NUM_OBJECT_CLASSES  # noqa: E402

torch.manual_seed(0)
B, N, T = 2, MAX_OBB_TOKENS, 30
N_REAL = 40

READOUT = sys.argv[1] if len(sys.argv) > 1 else 'token0'
print(f"readout = {READOUT}\n")

cfg = CustomConfig(num_object_classes=NUM_OBJECT_CLASSES,
                   input_modality='video,location,obb',
                   max_obb_tokens=N, num_transformer_layers=2,
                   readout=READOUT)
model = CustomTransformerModel(cfg).eval()


def make_batch(seed=0):
    g = torch.Generator().manual_seed(seed)
    mask = torch.zeros(B, N, dtype=torch.bool)
    mask[:, :N_REAL] = True
    voxels = torch.zeros(B, N, 3, dtype=torch.long)
    volumes = torch.zeros(B, N, dtype=torch.long)
    classes = torch.zeros(B, N, dtype=torch.long)
    voxels[:, :N_REAL] = torch.randint(1, 17, (B, N_REAL, 3), generator=g)
    volumes[:, :N_REAL] = torch.randint(1, 17, (B, N_REAL), generator=g)
    classes[:, :N_REAL] = torch.randint(1, 1200, (B, N_REAL), generator=g)
    return dict(
        index=torch.arange(B),
        visual_features=torch.randn(B, T, 4096, generator=g),
        visual_mask=torch.ones(B, T, dtype=torch.int),
        person_locations=torch.randint(0, 17, (B, T, 3), generator=g),
        person_location_mask=torch.ones(B, T, dtype=torch.int),
        obb_voxels=voxels, obb_volumes=volumes,
        obb_classes=classes, obb_mask=mask)


def logits(b):
    with torch.no_grad():
        return model(**b)['interaction']


base = make_batch()
L0 = logits(base)
fails = []

# --- 1. sensitivity ------------------------------------------------------
print("1. SENSITIVITY -- each component must move the logits")
for field, newval in [('obb_voxels', None), ('obb_volumes', None), ('obb_classes', None)]:
    b = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in base.items()}
    if field == 'obb_voxels':
        b[field][:, :N_REAL] = 17 - b[field][:, :N_REAL]        # mirror the grid
    elif field == 'obb_volumes':
        b[field][:, :N_REAL] = 17 - b[field][:, :N_REAL]        # flip size bins
    else:
        b[field][:, :N_REAL] = (b[field][:, :N_REAL] + 500) % 1200 + 1
    d = (logits(b) - L0).abs().max().item()
    ok = d > 1e-6
    print(f"   {field:<14} max|Δlogit| = {d:.6f}   {'OK' if ok else 'DEAD -- not wired!'}")
    if not ok:
        fails.append(f"{field} does not affect output")

# --- 2. gradients --------------------------------------------------------
print("\n2. GRADIENTS -- tables this modality uses must receive non-zero grad")
model.zero_grad()
b = make_batch()
b['labels'] = {'interaction': torch.randint(0, 2, (B, 4096))}
model(**b)['loss'].backward()
for name in ['volume_embedding.weight', 'obb_type_embedding', 'class_embedding.weight',
             'location_embedding_x.weight', 'location_embedding_y.weight',
             'location_embedding_z.weight']:
    p = dict(model.named_parameters())[name]
    gn = 0.0 if p.grad is None else p.grad.abs().sum().item()
    ok = gn > 0
    print(f"   {name:<32} |grad| = {gn:.6f}   {'OK' if ok else 'NO GRADIENT'}")
    if not ok:
        fails.append(f"{name} receives no gradient")

# volume/obb_type must be OBB-only: with no obb in the modality they get nothing
cfg2 = CustomConfig(num_object_classes=NUM_OBJECT_CLASSES,
                    input_modality='video,location', num_transformer_layers=2,
                    readout=READOUT)
m2 = CustomTransformerModel(cfg2)
b2 = make_batch(); b2['labels'] = {'interaction': torch.randint(0, 2, (B, 4096))}
m2(**b2)['loss'].backward()
for name in ['volume_embedding.weight', 'obb_type_embedding']:
    p = dict(m2.named_parameters())[name]
    gn = 0.0 if p.grad is None else p.grad.abs().sum().item()
    ok = gn == 0
    print(f"   {name:<32} |grad| without obb = {gn:.6f}   "
          f"{'OK (obb-only)' if ok else 'LEAKING into other modalities'}")
    if not ok:
        fails.append(f"{name} active without the obb modality")

# --- 3. padding isolation ------------------------------------------------
print("\n3. PAD ISOLATION -- rewriting padded slots must NOT change logits")
b = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in base.items()}
g = torch.Generator().manual_seed(99)
b['obb_voxels'][:, N_REAL:] = torch.randint(1, 17, (B, N - N_REAL, 3), generator=g)
b['obb_volumes'][:, N_REAL:] = torch.randint(1, 17, (B, N - N_REAL), generator=g)
b['obb_classes'][:, N_REAL:] = torch.randint(1, 1200, (B, N - N_REAL), generator=g)
d = (logits(b) - L0).abs().max().item()
ok = d < 1e-5
print(f"   max|Δlogit| = {d:.8f}   {'OK -- padding is masked out' if ok else 'LEAK!'}")
if not ok:
    fails.append("padded slots affect the output")

# --- 4. permutation invariance ------------------------------------------
print("\n4. PERMUTATION -- reordering real tokens must NOT change logits")
b = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in base.items()}
perm = torch.randperm(N_REAL)
b['obb_voxels'][:, :N_REAL] = b['obb_voxels'][:, perm]
b['obb_volumes'][:, :N_REAL] = b['obb_volumes'][:, perm]
b['obb_classes'][:, :N_REAL] = b['obb_classes'][:, perm]
d = (logits(b) - L0).abs().max().item()
ok = d < 1e-4
print(f"   max|Δlogit| = {d:.8f}   {'OK -- order is meaningless' if ok else 'ORDER MATTERS (unexpected)'}")
if not ok:
    fails.append("output depends on OBB token order")

# --- 5. shared spatial tables -------------------------------------------
print("\n5. SHARED TABLES -- OBBs must read location_embedding_x")
v = int(base['obb_voxels'][0, 0, 0])
with torch.no_grad():
    model.location_embedding_x.weight[v] += 10.0
d = (logits(base) - L0).abs().max().item()
with torch.no_grad():
    model.location_embedding_x.weight[v] -= 10.0
ok = d > 1e-6
print(f"   perturbing location_embedding_x[{v}] -> max|Δlogit| = {d:.6f}   "
      f"{'OK -- shared with location/gaze/trajectory' if ok else 'NOT SHARED'}")
if not ok:
    fails.append("OBB tokens do not use the shared location tables")

print("\n" + "=" * 60)
if fails:
    print("FAILURES:")
    for f in fails:
        print(f"  - {f}")
    sys.exit(1)
print("ALL EMBEDDING CHECKS PASSED")
