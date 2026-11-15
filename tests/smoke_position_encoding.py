"""Checks for the shared spatial Fourier code (docs/temporal_spatial_encoding_spec.md §2).

Run: python smoke_position_encoding.py    (pure torch, no dataset needed, ~40s on CPU)

The change this covers is a *sharing* change more than an encoding change. Before
it, ``--voxel_query_type fourier`` gave the decoder's 4096 queries a private
``W_r`` while every input token kept the index tables -- two unrelated spatial
vocabularies, so the arm was never the clean "coordinates vs lookup" comparison it
read as. Now the queries call the model's own ``_voxel_pos_embedding``.

Three of these are the ones that would silently train a wrong model:

  * **Test 3** -- the query at voxel v and an input token at voxel v must carry the
    identical code. Two call sites agree on ``(ijk-0.5)/16`` vs ``(ijk0+0.5)/16``
    only by construction; nothing else in the code enforces it, and a mismatch
    trains happily against queries that are half a cell off.
  * **Test 4** -- the Fourier and axis_sum arms must start at the same magnitude.
    cos/sin are bounded, so ``W_r`` can never set amplitude; the legacy ``/sqrt(D)``
    put the query at norm 0.707 against ``|attn_out| ~ 1.11``, which turns
    ``--decoder_residual raw`` from a 50:1 regime into 0.6:1. An unmatched arm
    measures the residual balance, not the encoding.
  * **Test 6** -- pad slots must not read as a real corner voxel. ``(0-0.5)/16``
    lands a third of a cell outside the floor corner, i.e. almost exactly on top of
    voxel 1, and ``model.forward`` deliberately keeps slot 0 attendable on all-pad
    OBB rows where it must mean "no objects here".
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import (CustomConfig, CustomTransformerModel,  # noqa: E402
                   resolve_position_encoding)
from object_class_vocab import NUM_OBJECT_CLASSES  # noqa: E402
from smoke_decoder import B, N_OBB, NUM_VOXELS, make_batch  # noqa: E402

MODALITY = 'video,location,obb'
FAILS = []


def check(ok, label, detail=''):
    print(f"   {label:<58} {detail:<26} {'OK' if ok else 'FAIL'}")
    if not ok:
        FAILS.append(label)
    return ok


def build(seed=0, **overrides):
    base = dict(num_object_classes=NUM_OBJECT_CLASSES, num_transformer_layers=2,
                use_static_env=True, max_obb_tokens=N_OBB, input_modality=MODALITY,
                decoder='voxel_query', readout='cls')
    base.update(overrides)
    torch.manual_seed(seed)
    return CustomTransformerModel(CustomConfig(**base)).eval()


def logits(model, batch):
    with torch.no_grad():
        return model(**batch)['interaction']


def pos(model, ijk):
    with torch.no_grad():
        return model._voxel_pos_embedding(torch.tensor([ijk], dtype=torch.long))


print("=" * 60)
print("SPATIAL POSITION ENCODING")
print("=" * 60)

print("\n1. AUTO RESOLUTION -- the input side follows the query parameterisation")
for vqt, want in (('axis_sum', 'none'), ('fourier', 'fourier'), ('learned', 'none')):
    got = resolve_position_encoding(
        CustomConfig(decoder='voxel_query', voxel_query_type=vqt))
    check(got == want, f"voxel_query_type={vqt} -> {want}", got)
check(resolve_position_encoding(CustomConfig()) == 'none',
      "decoder=linear (the default) -> none")
check(resolve_position_encoding(
    CustomConfig(decoder='voxel_query', voxel_query_type='fourier',
                 position_encoding='none')) == 'none',
      "explicit 'none' overrides auto (reproduces the 2026-08-11 runs)")

print("\n2. OFF IS OFF -- the default arm keeps its exact weights and logits")
m_a, m_a2 = build(voxel_query_type='axis_sum'), build(voxel_query_type='axis_sum')
batch = make_batch()
check(torch.equal(logits(m_a, batch), logits(m_a2, batch)),
      "axis_sum is reproducible at a fixed seed")
names = dict(m_a.named_parameters())
check(not any(n.startswith('spatial_') for n in names),
      "no spatial_* parameter exists under position_encoding='none'",
      f"{len(names)} params")
m_legacy = build(voxel_query_type='fourier', position_encoding='none')
check('decoder.W_r' in dict(m_legacy.named_parameters()),
      "position_encoding='none' keeps the decoder's private W_r")
check('query_type_embedding' not in dict(m_legacy.decoder.named_parameters()),
      "...and adds no marker, so those checkpoints still load")

print("\n3. ONE CODE -- a query and an input token at a voxel must be identical")
m = build(voxel_query_type='fourier')
with torch.no_grad():
    q = m.decoder.queries()
    p = m._voxel_pos_embedding(m.decoder.voxel_ijk1)
    delta = (q - m.decoder.query_type_embedding - p).abs().max().item()
check(delta < 1e-4, "query == input code + query_type_embedding", f"max|d| = {delta:.2e}")
check(m.decoder._pos_embedding[0].__self__ is m,
      "the decoder calls the model's own _voxel_pos_embedding")
check('W_r' not in dict(m.decoder.named_parameters()),
      "no private decoder W_r on the shared path")
check(sum(1 for n, _ in m.named_parameters() if n == 'spatial_W_r') == 1,
      "spatial_W_r appears exactly once in the state_dict")
with torch.no_grad():
    coords_model = (m.decoder.voxel_ijk1.float() - 0.5) / 16
check(torch.allclose(coords_model, m.decoder.voxel_coords),
      "(ijk-0.5)/16 == decoder.voxel_coords, the grids agree")

print("\n4. MAGNITUDE -- the arms must start matched, or raw's balance is the experiment")
with torch.no_grad():
    qa, qf = m_a.decoder.queries().norm(dim=-1).mean(), m.decoder.queries().norm(dim=-1).mean()
    pa = m_a._voxel_pos_embedding(m_a.decoder.voxel_ijk1).norm(dim=-1).mean()
    pf = m._voxel_pos_embedding(m.decoder.voxel_ijk1).norm(dim=-1).mean()
    ql = m_legacy.decoder.queries().norm(dim=-1).mean()
check(abs(qa - qf) / qa < 0.05, "query norm: fourier matches axis_sum",
      f"{qf:.1f} vs {qa:.1f}")
check(abs(pa - pf) / pa < 0.05, "input position norm matches", f"{pf:.1f} vs {pa:.1f}")
check(ql < 1.0, "the legacy arm really was ~78x quieter (why the gain exists)",
      f"legacy {ql:.3f}")

print("\n5. SMOOTHNESS -- adjacent voxels must start closer than distant ones")
for name, mm in (('axis_sum', m_a), ('fourier', m)):
    near = (pos(mm, [8, 8, 8]) - pos(mm, [8, 8, 9])).norm().item()
    far = (pos(mm, [8, 8, 8]) - pos(mm, [2, 14, 3])).norm().item()
    ratio = near / far
    # axis_sum's floor is sqrt(2/6) = 0.577: one differing axis is two independent
    # embedding rows, three differing axes are six. It cannot go below that by
    # construction, which is the discreteness the Fourier code is meant to fix.
    check(ratio < 0.577 if name == 'fourier' else abs(ratio - 0.577) < 0.05,
          f"{name}: near/far ratio", f"{near:.1f}/{far:.1f} = {ratio:.3f}")

print("\n6. PADDING -- an absent box must not read as a floor-corner box")
d_pad = (pos(m, [0, 0, 0]) - pos(m, [1, 1, 1])).norm().item()
check(d_pad > 10.0, "pad code is far from voxel (1,1,1)", f"|d| = {d_pad:.1f}")
b_pad = make_batch()
b_pad['obb_voxels'] = torch.zeros_like(b_pad['obb_voxels'])
b_pad['obb_mask'] = torch.zeros_like(b_pad['obb_mask'])
check(torch.isfinite(logits(m, b_pad)).all(),
      "an all-pad OBB segment stays finite (slot 0 attendable)")

print("\n7. GRADIENTS -- every new parameter trains")
labels = {'interaction': torch.randint(0, 2, (B, NUM_VOXELS)).float()}
mg = build(voxel_query_type='fourier')
mg(**make_batch(), labels=labels)['loss'].backward()
for name in ('spatial_W_r', 'spatial_gain', 'decoder.query_type_embedding'):
    g = dict(mg.named_parameters())[name].grad
    check(g is not None and g.norm() > 0, f"{name} receives gradient",
          f"|g| = {g.norm():.2e}" if g is not None else "None")
# spatial_pad_embedding is the exception, and correctly so: padded OBB slots are
# masked out of attention, so on an ordinary window it has no gradient path at
# all. It earns one only on the all-pad rows where forward() keeps slot 0
# attendable -- 4.0% of train windows on the production data, which is why it is
# worth a parameter but also why the ordinary batch above must NOT reach it.
g_ord = dict(mg.named_parameters())['spatial_pad_embedding'].grad
check(g_ord is None or g_ord.norm() == 0,
      "spatial_pad_embedding is inert while every box is real",
      "no grad" if g_ord is None else f"|g| = {g_ord.norm():.2e}")
mp = build(voxel_query_type='fourier')
b_allpad = make_batch()
b_allpad['obb_voxels'] = torch.zeros_like(b_allpad['obb_voxels'])
b_allpad['obb_mask'] = torch.zeros_like(b_allpad['obb_mask'])
mp(**b_allpad, labels=labels)['loss'].backward()
g_pad = dict(mp.named_parameters())['spatial_pad_embedding'].grad
check(g_pad is not None and g_pad.norm() > 0,
      "...and trains on the all-pad rows it exists for",
      f"|g| = {g_pad.norm():.2e}" if g_pad is not None else "None")

print("\n8. COHERENCE -- refuse the pairing that would mean two vocabularies")
for vqt in ('axis_sum', 'learned'):
    try:
        build(voxel_query_type=vqt, position_encoding='fourier')
        check(False, f"fourier inputs + {vqt} queries refused")
    except ValueError:
        check(True, f"fourier inputs + {vqt} queries refused")
try:
    build(decoder='linear', position_encoding='fourier')
    check(True, "fourier inputs + linear decoder allowed (no queries to clash)")
except ValueError:
    check(False, "fourier inputs + linear decoder allowed (no queries to clash)")

print("\n9. MODE AND PER-AXIS SCALE")
m_add = build(voxel_query_type='fourier', position_encoding_mode='add')
with torch.no_grad():
    p_add = m_add._voxel_pos_embedding(m_add.decoder.voxel_ijk1).norm(dim=-1).mean()
check(p_add > pf, "'add' carries both codes, so it is louder than 'replace'",
      f"{p_add:.1f} vs {pf:.1f}")
with torch.no_grad():
    q_add = m_add.decoder.queries()
    p_add_v = m_add._voxel_pos_embedding(m_add.decoder.voxel_ijk1)
check((q_add - m_add.decoder.query_type_embedding - p_add_v).abs().max() < 1e-4,
      "'add' keeps query and input identical too")
m_ax = build(voxel_query_type='fourier', voxel_query_fourier_scale='4.0,4.0,12.0')
stds = [m_ax.spatial_W_r[:, a].std().item() for a in range(3)]
check(stds[2] > 2 * stds[0], "per-axis scale reaches the right columns",
      f"{[round(s, 1) for s in stds]}")
check(logits(m_ax, batch).shape == (B, NUM_VOXELS), "per-axis arm runs end to end")

print("\n10. CHECKPOINT ROUND TRIP")
import tempfile  # noqa: E402
with tempfile.TemporaryDirectory() as d:
    m.save_pretrained(d)
    back = CustomTransformerModel.from_pretrained(d).eval()
    check(torch.allclose(logits(m, batch), logits(back, batch), atol=1e-5),
          "reloaded model reproduces its logits")
# Cross-arm loading must fail LOUDLY, the same way smoke_decoder pins it for the
# linear/voxel_query pair: load_state_dict is strict, from_pretrained is not and
# would only warn. The arms differ by spatial_W_r/_gain/_pad_embedding and
# decoder.query_type_embedding, so neither direction can be silently absorbed.
for src, dst, label in ((m, m_a, 'fourier -> axis_sum'),
                        (m_a, m, 'axis_sum -> fourier'),
                        (m_legacy, m, 'legacy private-W_r -> shared')):
    try:
        dst.load_state_dict(src.state_dict())
        check(False, f"{label} refused", "it LOADED")
    except RuntimeError:
        check(True, f"{label} refused")

print("\n11. LOCATION TOKENS -- 'joint' folds 3 tokens/second into 1")
m_ax = build(voxel_query_type='axis_sum')
m_jt = build(voxel_query_type='axis_sum', location_tokens='joint')
check('location_type_embedding' not in dict(m_ax.named_parameters()),
      "'axis' creates no marker, so it keeps its exact state_dict")
with torch.no_grad():
    lay_a = {n: e - s for n, s, e in m_ax(**batch, need_attention=True)['segments']}
    lay_j = {n: e - s for n, s, e in m_jt(**batch, need_attention=True)['segments']}
check(lay_a['location'] == 3 * lay_j['location'],
      "location segment 90 -> 30 tokens", f"{lay_a['location']} -> {lay_j['location']}")
with torch.no_grad():
    # The joint token must be the SAME function every other positional token uses --
    # that is the whole reason to merge by summing rather than by a fresh projection.
    d = (m_jt._voxel_pos_embedding(batch['person_locations'])
         - (m_jt.location_embedding_x(batch['person_locations'][:, :, 0])
            + m_jt.location_embedding_y(batch['person_locations'][:, :, 1])
            + m_jt.location_embedding_z(batch['person_locations'][:, :, 2]))).abs().max()
check(d == 0, "joint token is _voxel_pos_embedding, not a private projection",
      f"max|d| = {d:.1e}")
check(torch.isfinite(logits(build(voxel_query_type='fourier',
                                  location_tokens='joint'), batch)).all(),
      "'joint' composes with the Fourier code")
# Under 'axis' the shared time stamp is the ONLY thing binding a second's three
# tokens; under 'joint' the binding is structural and the stamp only carries
# recency. So order-invariance without a time code must hold in both layouts.
b_rev = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in batch.items()}
for key in ('person_locations', 'visual_features'):
    b_rev[key] = torch.flip(b_rev[key], dims=[1])
m_jn = build(voxel_query_type='axis_sum', location_tokens='joint', time_encoding='none')
m_jf = build(voxel_query_type='axis_sum', location_tokens='joint',
             time_encoding='fourier')
check(torch.allclose(logits(m_jn, batch), logits(m_jn, b_rev), atol=1e-5),
      "joint + no time code: the 30 seconds are still an unordered set")
check(not torch.allclose(logits(m_jf, batch), logits(m_jf, b_rev), atol=1e-5),
      "joint + time code: order now matters")
try:
    m_ax.load_state_dict(m_jt.state_dict())
    check(False, "joint -> axis refused", "it LOADED")
except RuntimeError:
    check(True, "joint -> axis refused")
# The consequence that is easy to miss: under position_encoding='fourier', 'joint'
# routes location through _voxel_pos_embedding too, so on the video,location,obb
# modality set NOTHING reads location_embedding_x/y/z any more -- location, obb and
# the queries all carry the Fourier code. Under 'axis' the tables stay live and the
# model runs TWO spatial vocabularies at once (tables for location, Fourier for the
# rest). Assert both, because which one is true is the actual content of this arm.
lab = {'interaction': torch.randint(0, 2, (B, NUM_VOXELS)).float()}
for layout, want_dead in (('axis', False), ('joint', True)):
    mm = build(voxel_query_type='fourier', location_tokens=layout)
    mm(**make_batch(), labels=lab)['loss'].backward()
    g = mm.location_embedding_x.weight.grad
    dead = g is None or g.norm() == 0
    check(dead == want_dead,
          f"'{layout}': index tables {'unused' if want_dead else 'still in use'}",
          f"|g| = {0.0 if g is None else g.norm():.2e}")

print("\n12. UNTIED QUERIES -- the same code, different parameters")
# The cell the 2026-08-11 vq_fourier arm could not reach: Fourier on BOTH sides
# with SEPARATE parameters. That arm was untied AND at norm 0.707 against
# axis_sum's 55.7, so it changed the basis, the tying and the amplitude at once.
# Here tied and untied differ in exactly one thing -- whose tensors -- which is
# what makes "is sharing worth anything?" answerable. PETR (ECCV 2022) builds
# query and key position codes with separate MLPs; DAB-DETR (ICLR 2022) argues
# the opposite, but gets it free because its sinusoids have no parameters.
_tied = build(decoder='voxel_query', voxel_query_type='fourier',
              position_encoding='fourier', voxel_query_untied=False)
_unt = build(decoder='voxel_query', voxel_query_type='fourier',
             position_encoding='fourier', voxel_query_untied=True)
# voxel_ijk1 = voxel_ijk0 + 1, so the query row for ijk1=(3,7,12) is the FLAT
# index of ijk0=(2,6,11). Getting this wrong compares two different voxels and
# the test passes for the wrong reason.
_ijk1 = torch.tensor([[3, 7, 12]])
_flat = 2 * 256 + 6 * 16 + 11
with torch.no_grad():
    _qt = _tied.decoder.queries()[_flat] - _tied.decoder.query_type_embedding
    _it = _tied._voxel_pos_embedding(_ijk1)[0]
    _qu = _unt.decoder.queries()[_flat] - _unt.decoder.query_type_embedding
    _iu = _unt._voxel_pos_embedding(_ijk1)[0]
check((_qt - _it).abs().max() < 1e-4, "tied: query IS the input code",
      f"max|d| = {(_qt - _it).abs().max():.2e}")
check((_qu - _iu).abs().max() > 1.0, "untied: query is NOT the input code",
      f"max|d| = {(_qu - _iu).abs().max():.2e}")
# Amplitude must NOT be the variable -- that was the 2026-08-11 confound.
check(abs(_qu.norm() - _qt.norm()) < 1.0, "untied query norm matches tied",
      f"{_qu.norm():.1f} vs {_qt.norm():.1f}")
_p = dict(_unt.named_parameters())
check('decoder.W_r' in _p and 'decoder.query_gain' in _p,
      "untied owns W_r and query_gain")
check('decoder.W_r' not in dict(_tied.named_parameters()),
      "tied has no private decoder.W_r")
check(set(build(decoder='voxel_query', voxel_query_type='fourier',
                position_encoding='fourier').state_dict())
      == set(_tied.state_dict()), "omitting the flag == untied=False")
# Both codes must keep training: the input side is still the model's.
_unt(**make_batch(), labels={'interaction':
                            torch.randint(0, 2, (B, NUM_VOXELS)).float()}
     )['loss'].backward()
for _n in ('decoder.W_r', 'decoder.query_gain', 'spatial_W_r', 'spatial_gain'):
    _g = dict(_unt.named_parameters())[_n].grad
    check(_g is not None and _g.abs().sum() > 0, f"{_n} trains",
          f"|g| = {_g.abs().mean():.2e}" if _g is not None else "no grad")
# The flag is meaningless for the other two query types -- refuse, don't mislead.
try:
    build(decoder='voxel_query', voxel_query_type='axis_sum',
          position_encoding='none', voxel_query_untied=True)
    check(False, "axis_sum + untied refused")
except ValueError:
    check(True, "axis_sum + untied refused")
for _lbl, _a, _b2 in (("untied -> tied refused", _tied, _unt),
                      ("tied -> untied refused", _unt, _tied)):
    try:
        _a.load_state_dict(_b2.state_dict()); check(False, _lbl)
    except RuntimeError:
        check(True, _lbl)
# The legacy private path must stay bit-compatible: the nine 2026-08-11
# checkpoints load against it and still reproduce.
_lg = build(decoder='voxel_query', voxel_query_type='fourier',
            position_encoding='none')
_sd = _lg.state_dict()
check('decoder.W_r' in _sd and 'decoder.query_gain' not in _sd
      and 'decoder.query_type_embedding' not in _sd,
      "legacy: private W_r, no gain, no marker")
with torch.no_grad():
    _n0 = _lg.decoder.queries()[_flat].norm()
check(abs(_n0 - 0.7071) < 1e-3, "legacy query norm still 0.707", f"{_n0:.4f}")

print("\n" + "=" * 60)
if FAILS:
    print(f"{len(FAILS)} CHECK(S) FAILED:")
    for f in FAILS:
        print(f"  - {f}")
    sys.exit(1)
print("ALL POSITION-ENCODING CHECKS PASSED")
print("=" * 60)
