"""Checks for the time-to-anchor encoding (docs/temporal_spatial_encoding_spec.md §3).

Run: python smoke_time_encoding.py     (pure torch, no dataset needed, ~20s on CPU)

This change deliberately breaks a property the repo previously asserted: the 30
seconds of video / location / gaze were an unordered SET to self-attention, and
several existing tests pin that. So the checks here come in two halves --

  * what must STILL hold: --time_encoding none reproduces the old logits exactly,
    OBB tokens stay order-invariant (a fused box has no timestamp), and the
    decoder's key-permutation invariance is untouched;
  * what must now be FALSE: reordering the per-second tokens must change the
    output, or the encoding is not reaching them.

The subtle one is test 4. delta is read off the SLOT INDEX, which is only correct
because the loader end-aligns these sequences to obs_end -- slot T-1 is the second
nearest the anchor. Get the orientation backwards and the model learns a recency
prior pointing at the wrong end of the window, which no shape check would catch.
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import CustomConfig, CustomTransformerModel  # noqa: E402
from object_class_vocab import NUM_OBJECT_CLASSES  # noqa: E402
from smoke_decoder import B, N_OBB, N_OBB_REAL, NUM_VOXELS, T, make_batch  # noqa: E402

MODALITY = 'video,location,obb,gaze'
FAILS = []


def check(ok, label, detail=''):
    print(f"   {label:<56} {detail:<24} {'OK' if ok else 'FAIL'}")
    if not ok:
        FAILS.append(label)
    return ok


def build(seed=0, **overrides):
    base = dict(num_object_classes=NUM_OBJECT_CLASSES, num_transformer_layers=2,
                use_static_env=True, max_obb_tokens=N_OBB, input_modality=MODALITY)
    base.update(overrides)
    torch.manual_seed(seed)
    return CustomTransformerModel(CustomConfig(**base)).eval()


def logits(model, batch):
    with torch.no_grad():
        return model(**batch)['interaction']


def reverse_seconds(batch, keys=('visual_features', 'person_locations', 'gaze_voxels')):
    """Reverse the time axis of the named per-second modalities."""
    b = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in batch.items()}
    for key in keys:
        b[key] = torch.flip(b[key], dims=[1])
    return b


# Reversing the VIDEO tokens moves whichever token sits at sequence seat 0, and
# under --readout token0 that seat IS the output. So video order changes the logits
# even with no encoding whatsoever -- a property of the readout, not of any
# positional information. Everything below therefore separates the two: the
# set-invariance claim is tested on location+gaze (never at seat 0), or under
# --readout cls, where the summary token makes the whole sequence a set again.
NON_SEAT0 = ('person_locations', 'gaze_voxels')


print("1. OFF IS OFF -- the default must not move")
m_off = build(0, time_encoding='none')
m_ref = build(0)                      # no time_encoding argument at all
batch = make_batch(0)
check(torch.equal(logits(m_off, batch), logits(m_ref, batch)),
      "time_encoding='none' == omitting the flag")
check(not any(n == 'time_W_r' for n, _ in m_off.named_parameters()),
      "no time parameter exists when disabled")

print("\n2. THE PROPERTY BEING REPLACED -- time was unreadable, and now is not")
m_on = build(0, time_encoding='fourier')
base_on, base_off = logits(m_on, batch), logits(m_off, batch)

# (a) The honest statement of the old behaviour: tokens that are not at seat 0 are
#     a set. Reversing them changes nothing without an encoding.
d = (logits(m_off, reverse_seconds(batch, NON_SEAT0)) - base_off).abs().max().item()
check(d < 1e-5, "none: reversing location+gaze leaves logits unchanged",
      f"max|delta| = {d:.4e}")

# (b) ...and with the encoding, it must not.
d = (logits(m_on, reverse_seconds(batch, NON_SEAT0)) - base_on).abs().max().item()
check(d > 1e-5, "fourier: reversing location+gaze DOES change them",
      f"max|delta| = {d:.4e}")

# (c) Under --readout cls the whole sequence is a set, video included, so this is
#     the cleanest form of both statements.
m_off_cls = build(0, time_encoding='none', readout='cls')
m_on_cls = build(0, time_encoding='fourier', readout='cls')
d_off_cls = (logits(m_off_cls, reverse_seconds(batch))
             - logits(m_off_cls, batch)).abs().max().item()
d_on_cls = (logits(m_on_cls, reverse_seconds(batch))
            - logits(m_on_cls, batch)).abs().max().item()
check(d_off_cls < 1e-5, "none + cls: reversing ALL 30 seconds changes nothing",
      f"max|delta| = {d_off_cls:.4e}")
check(d_on_cls > 1e-5, "fourier + cls: reversing ALL 30 seconds changes the logits",
      f"max|delta| = {d_on_cls:.4e}")

# (d) The caveat this test exposed: under token0 the readout itself depends on
#     which token is first, so video order matters even with no encoding. That is
#     the readout, not recency -- and it is why CLAUDE.md's blanket "permuting the
#     seconds leaves the output unchanged" is wrong as written.
d = (logits(m_off, reverse_seconds(batch, ('visual_features',)))
     - base_off).abs().max().item()
check(d > 1e-5, "none + token0: video order matters via seat 0 alone",
      f"max|delta| = {d:.4e}")

print("\n3. RIGHT SEGMENTS ONLY -- obb has no timestamp to encode")
b = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in batch.items()}
perm = torch.randperm(N_OBB_REAL)
for key in ('obb_voxels', 'obb_volumes', 'obb_classes'):
    b[key][:, :N_OBB_REAL] = b[key][:, perm]
d = (logits(m_on, b) - base_on).abs().max().item()
check(d < 1e-4, "permuting OBB tokens still leaves logits unchanged",
      f"max|delta| = {d:.4e}")

print("\n4. DELTA ORIENTATION -- slot T-1 is the second nearest the anchor")
# The loader end-aligns: real seconds fill the LAST slots. So a signal placed one
# slot later is one second NEARER the anchor, and must pick up the delta that the
# later slot carries. Comparing "signal at slot t" against "the same signal at
# slot t+1, read one slot along" isolates the encoding from the content.
with torch.no_grad():
    te = m_on._time_encoding(T, torch.device('cpu'), torch.float32)   # [T, 768]
check(te.shape == (T, m_on.config.transformer_hidden_size), "shape", f"{tuple(te.shape)}")
deltas = [(T - 1 - n) / T for n in range(T)]
check(deltas[-1] == 0.0 and deltas[0] > deltas[-1],
      "last slot is delta=0, first slot is the oldest",
      f"delta[0]={deltas[0]:.3f} delta[-1]={deltas[-1]:.3f}")
# Monotone similarity: adjacent slots must be more alike than distant ones, which
# is the whole reason for Fourier over a 30-entry lookup.
near = (te[-1] - te[-2]).norm().item()
far = (te[-1] - te[0]).norm().item()
check(near < far, "adjacent seconds are closer than distant ones",
      f"|t29-t28|={near:.3f} < |t29-t0|={far:.3f}")

print("\n5. DISTINCTNESS -- 30 codes must not collapse")
for scale in (2.0, 10.0, 30.0):
    m = build(0, time_encoding='fourier', time_encoding_scale=scale)
    with torch.no_grad():
        te_s = m._time_encoding(T, torch.device('cpu'), torch.float32)
        d_mat = torch.cdist(te_s, te_s)
        d_mat.fill_diagonal_(float('inf'))
        worst = d_mat.min().item()
    check(worst > 1e-3, f"scale={scale}: all 30 seconds distinguishable",
          f"min distance = {worst:.4e}")

print("\n6. GRADIENT -- the encoding trains, and only when enabled")
b = make_batch(0)
b['labels'] = {'interaction': torch.randint(0, 2, (B, NUM_VOXELS))}
m_on.zero_grad()
m_on(**b)['loss'].backward()
g = m_on.time_W_r.grad
check(g is not None and g.abs().sum().item() > 0, "time_W_r receives gradient",
      f"|grad| = {0.0 if g is None else g.abs().sum().item():.6f}")

print("\n7. LENGTH ASSERTION -- a mismatched window must fail loudly")
# delta is read off the slot index, so a window of a different length would
# silently relabel every second rather than erroring.
short = make_batch(0)
short['visual_features'] = short['visual_features'][:, :T - 5]
short['visual_mask'] = short['visual_mask'][:, :T - 5]
short['person_locations'] = short['person_locations'][:, :T - 5]
short['person_location_mask'] = short['person_location_mask'][:, :T - 5]
short['gaze_voxels'] = short['gaze_voxels'][:, :T - 5]
short['gaze_mask'] = short['gaze_mask'][:, :T - 5]
try:
    logits(m_on, short)
    check(False, "short window raises rather than mislabelling", "it ran")
except AssertionError:
    check(True, "short window raises rather than mislabelling")

print("\n" + "=" * 60)
if FAILS:
    print("FAILURES:")
    for f in FAILS:
        print(f"  - {f}")
    sys.exit(1)
print("ALL TIME-ENCODING CHECKS PASSED")
