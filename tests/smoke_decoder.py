"""Regression + correctness tests for the voxel-query decoder (docs/voxel_query_decoder_spec.md §6).

Run: python smoke_decoder.py            (pure torch, no dataset needed, ~30s on CPU)
     python smoke_decoder.py --capture  (re-record the reference logits -- see below)

Test 1 is the one that must never fail. Every experimental result in this repo
was produced by the pre-decoder readout (a single Linear(768->4096) on sequence
seat 0), so `--decoder linear --readout token0 --voxel_bias_init zero` has to
stay bit-identical to it forever. The reference logits in
`smoke_decoder_reference.pt` were captured from the unmodified model before any
of this landed; --capture must NOT be re-run to "fix" a failure. If test 1 goes
red, the default path drifted and that is the bug.

The other ten checks (spec §6) cover the new decoder: shape, key-permutation
invariance, query-permutation equivariance, pad inertness, gradient flow both
ways, NaN safety with a single attendable key, query distinctness, and -- most
important -- that the query grid and the label grid agree on the flattening
order (tests 9 and 11, which cross-check each other).
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import CustomConfig, CustomTransformerModel  # noqa: E402
from object_class_vocab import NUM_OBJECT_CLASSES  # noqa: E402

REFERENCE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              'smoke_decoder_reference.pt')

# Deliberately small: these must run on CPU in under a minute. N_OBB is well
# below boxer_env.MAX_OBB_TOKENS because the cap value is irrelevant here --
# what matters is that real and padded slots both exist.
B, T = 2, 30
N_OBB, N_OBB_REAL = 64, 40
GRID, NUM_VOXELS = 16, 4096

# The two modality sets the reference covers: the historical production default
# (`video,location,env`) and the OBB branch this work is developed on.
REFERENCE_MODALITIES = ('video,location,env', 'video,location,obb')


# Spec test 10: every check below must hold under both readouts. Rather than
# duplicate the suite, main() runs it twice with this switched -- test 1 pins
# itself to token0 explicitly, since that is the setting the reference used.
READOUT = 'token0'


def build_config(**overrides):
    """Config for the smoke tests. Two encoder layers rather than six -- the
    properties under test are architectural, not depth-dependent."""
    base = dict(num_object_classes=NUM_OBJECT_CLASSES,
                num_transformer_layers=2,
                use_static_env=True,
                max_obb_tokens=N_OBB,
                readout=READOUT)
    base.update(overrides)
    return CustomConfig(**base)


def build_model(seed=0, **overrides):
    """Deterministic model. Every modality's submodule is constructed regardless
    of input_modality, so a fixed seed gives byte-identical weights across
    modality settings -- which is what lets one reference cover both. (The two
    decoders are the exception: they are exclusive, and only the selected one is
    built.)"""
    torch.manual_seed(seed)
    return CustomTransformerModel(build_config(**overrides)).eval()


# Every decoder test runs on the OBB branch: L = 30 video + 90 location + 64 obb
# + 2 sep = 186 tokens. The env branch is 4219 and would put a 4096 x 4219
# attention on a CPU for no extra coverage -- the properties under test are
# independent of what is in the sequence.
DECODER_MODALITY = 'video,location,obb'
QUERY_TYPES = ('axis_sum', 'fourier', 'learned')


def build_decoder(seed=0, **overrides):
    """A voxel-query decoder plus synthetic memory, exercised directly.

    Testing the decoder against random memory rather than through the encoder is
    deliberate: the set-like properties (key permutation, pad inertness, NaN
    safety) are properties OF THE DECODER, and driving them through a
    permutation-equivariant encoder would only test the encoder again, slowly.
    """
    length = overrides.pop('length', 64)
    n_pad = overrides.pop('n_pad', 8)
    model = build_model(seed, decoder='voxel_query',
                        input_modality=DECODER_MODALITY, **overrides)
    g = torch.Generator().manual_seed(seed + 1)
    memory = torch.randn(length, B, model.config.transformer_hidden_size, generator=g)
    mask = torch.zeros(B, length, dtype=torch.bool)
    mask[:, length - n_pad:] = True          # a real pad tail, as the loader produces
    return model, model.decoder, memory, mask


def decoder_logits(decoder, memory, mask):
    with torch.no_grad():
        return decoder(memory, mask)[0]


def make_batch(seed=0, n_obb=N_OBB, n_obb_real=N_OBB_REAL):
    """One fixed synthetic batch carrying every modality's tensors at once;
    input_modality decides which are actually read."""
    g = torch.Generator().manual_seed(seed)
    obb_mask = torch.zeros(B, n_obb, dtype=torch.bool)
    obb_mask[:, :n_obb_real] = True
    obb_voxels = torch.zeros(B, n_obb, 3, dtype=torch.long)
    obb_volumes = torch.zeros(B, n_obb, dtype=torch.long)
    obb_classes = torch.zeros(B, n_obb, dtype=torch.long)
    obb_voxels[:, :n_obb_real] = torch.randint(1, 17, (B, n_obb_real, 3), generator=g)
    obb_volumes[:, :n_obb_real] = torch.randint(1, 17, (B, n_obb_real), generator=g)
    obb_classes[:, :n_obb_real] = torch.randint(1, 1200, (B, n_obb_real), generator=g)
    return dict(
        index=torch.arange(B),
        visual_features=torch.randn(B, T, 4096, generator=g),
        visual_mask=torch.ones(B, T, dtype=torch.int),
        env_grid=torch.randint(0, 1200, (B, NUM_VOXELS), generator=g),
        person_locations=torch.randint(0, 17, (B, T, 3), generator=g),
        person_location_mask=torch.ones(B, T, dtype=torch.int),
        gaze_voxels=torch.randint(0, 17, (B, T, 3), generator=g),
        gaze_mask=torch.ones(B, T, dtype=torch.int),
        obb_voxels=obb_voxels, obb_volumes=obb_volumes,
        obb_classes=obb_classes, obb_mask=obb_mask,
    )


def logits_of(model, batch):
    with torch.no_grad():
        return model(**batch)['interaction']


def capture_reference():
    """Record the default-path logits. Run once, against the unmodified model."""
    ref = {}
    for modality in REFERENCE_MODALITIES:
        model = build_model(0, input_modality=modality)
        ref[modality] = logits_of(model, make_batch(0)).clone()
    torch.save(ref, REFERENCE_PATH)
    print(f"Wrote reference logits for {len(ref)} modality sets -> {REFERENCE_PATH}")
    for k, v in ref.items():
        print(f"   {k:<22} shape={tuple(v.shape)} "
              f"mean={v.mean():+.6f} std={v.std():.6f} sum={v.sum():+.6f}")


FAILS = []


def check(ok, label, detail=''):
    """Report one assertion. Collects rather than raises, so a run reports every
    failure at once instead of only the first."""
    print(f"   {label:<58} {detail:<26} {'OK' if ok else 'FAIL'}")
    if not ok:
        FAILS.append(label)
    return ok


# --------------------------------------------------------------------------
# 1. Regression -- the one that must never fail
# --------------------------------------------------------------------------

def test_1_regression():
    """Default flags must reproduce the pre-decoder logits bit for bit.

    Not 'to within tolerance': identical. Any drift means every PR-AUC number
    recorded in this repo is no longer comparable to a run on this branch.
    """
    print("1. REGRESSION -- default flags must be bit-identical to pre-change main")
    ref = torch.load(REFERENCE_PATH)
    for modality in REFERENCE_MODALITIES:
        model = build_model(0, input_modality=modality, readout='token0',
                            voxel_bias_init='zero')
        got = logits_of(model, make_batch(0))
        same = torch.equal(got, ref[modality])
        d = (got - ref[modality]).abs().max().item()
        check(same, f"{modality} bit-identical", f"max|delta| = {d:.3e}")


def test_2_shape():
    """Every decoder/query-type combination returns [B, 4096]."""
    print("\n2. SHAPE -- [B, 4096] for every decoder / query-type combination")
    m = build_model(0, input_modality=DECODER_MODALITY, decoder='linear')
    check(logits_of(m, make_batch(0)).shape == (B, NUM_VOXELS), "linear")
    for qt in QUERY_TYPES:
        m = build_model(0, input_modality=DECODER_MODALITY,
                        decoder='voxel_query', voxel_query_type=qt)
        out = logits_of(m, make_batch(0))
        check(out.shape == (B, NUM_VOXELS) and torch.isfinite(out).all(),
              f"voxel_query / {qt}", f"{tuple(out.shape)}")


def test_3_key_permutation_invariance():
    """Cross-attention is a set operation over K/V: permuting the real memory
    tokens (and the mask with them) must not move a single logit. Catches an
    accidental positional leak."""
    print("\n3. KEY PERMUTATION -- reordering memory tokens must not move logits")
    for qt in QUERY_TYPES:
        _, dec, memory, mask = build_decoder(0, voxel_query_type=qt)
        base = decoder_logits(dec, memory, mask)
        perm = torch.randperm(memory.shape[0])
        d = (decoder_logits(dec, memory[perm], mask[:, perm]) - base).abs().max().item()
        check(d < 1e-4, f"{qt}", f"max|delta| = {d:.2e}")


def test_4_query_permutation_equivariance():
    """Permuting the queries must permute the outputs identically -- proof that
    a voxel's identity comes from its query (and its bias row) and from nothing
    else, e.g. not from an output-row index."""
    print("\n4. QUERY PERMUTATION -- permuting queries permutes logits identically")
    for qt in QUERY_TYPES:
        _, dec, memory, mask = build_decoder(0, voxel_query_type=qt)
        base = decoder_logits(dec, memory, mask)
        perm = torch.randperm(NUM_VOXELS)
        with torch.no_grad():
            # Permute every per-voxel input together: the coordinate buffers the
            # queries are built from AND the bias's index buffers.
            dec.voxel_ijk0.copy_(dec.voxel_ijk0[perm])
            dec.voxel_ijk1.copy_(dec.voxel_ijk1[perm])
            dec.voxel_coords.copy_(dec.voxel_coords[perm])
            if qt == 'learned':
                dec.query_embedding.weight.copy_(dec.query_embedding.weight[perm])
        d = (decoder_logits(dec, memory, mask) - base[:, perm]).abs().max().item()
        check(d < 1e-4, f"{qt}", f"max|delta| = {d:.2e}")


def test_5_pad_inert():
    """Appending garbage keys flagged as padding must change nothing."""
    print("\n5. PAD INERTNESS -- masked keys cannot reach the output")
    for qt in QUERY_TYPES:
        _, dec, memory, mask = build_decoder(0, voxel_query_type=qt)
        base = decoder_logits(dec, memory, mask)
        g = torch.Generator().manual_seed(7)
        junk = torch.randn(32, B, memory.shape[2], generator=g) * 50
        memory2 = torch.cat([memory, junk], dim=0)
        mask2 = torch.cat([mask, torch.ones(B, 32, dtype=torch.bool)], dim=1)
        d = (decoder_logits(dec, memory2, mask2) - base).abs().max().item()
        check(d < 1e-4, f"{qt}", f"max|delta| = {d:.2e}")


def _new_parameter_names(model):
    """Parameters the voxel-query decoder adds, excluding the location tables,
    which it shares with the input modalities rather than owning."""
    return [n for n, _ in model.named_parameters()
            if n.startswith('decoder.') and 'location_embedding' not in n]


def test_6_gradient_flow():
    """Every new parameter must receive gradient -- and, under --decoder linear,
    none of them may even exist, so a `linear` run on this branch is the same
    model the baseline runs used."""
    print("\n6. GRADIENTS -- every new parameter trains; none exist under linear")
    for qt in QUERY_TYPES:
        m = build_model(0, input_modality=DECODER_MODALITY,
                        decoder='voxel_query', voxel_query_type=qt)
        b = make_batch(0)
        b['labels'] = {'interaction': torch.randint(0, 2, (B, NUM_VOXELS))}
        m.zero_grad()
        m(**b)['loss'].backward()
        params = dict(m.named_parameters())
        dead = [n for n in _new_parameter_names(m)
                if params[n].grad is None or params[n].grad.abs().sum().item() == 0]
        check(not dead, f"{qt}: all {len(_new_parameter_names(m))} decoder params train",
              f"dead: {dead if dead else 'none'}")
        # The shared tables must be reached by the queries too, not only by the
        # input tokens -- that sharing is the design claim of axis_sum.
        if qt == 'axis_sum':
            g = params['location_embedding_x.weight'].grad
            check(g is not None and g.abs().sum().item() > 0,
                  "axis_sum: queries reach the shared location tables")

    m = build_model(0, input_modality=DECODER_MODALITY, decoder='linear')
    check(m.decoder is None and not _new_parameter_names(m),
          "linear: no decoder parameters exist at all")
    check(m.output_layer is not None, "linear: output_layer is the readout")
    m_vq = build_model(0, input_modality=DECODER_MODALITY, decoder='voxel_query')
    check(m_vq.output_layer is None,
          "voxel_query: the 3.15M output_layer is gone, not merely unused")


def test_7_single_attendable_key():
    """One attendable key per row must still give finite logits. All-True rows
    make softmax return NaN, which is why the OBB segment keeps slot 0
    attendable; this checks the decoder survives the extreme case."""
    print("\n7. NaN SAFETY -- one attendable key must still produce finite logits")
    for qt in QUERY_TYPES:
        _, dec, memory, mask = build_decoder(0, voxel_query_type=qt)
        mask = torch.ones_like(mask)
        mask[:, 0] = False
        check(bool((~mask).any(dim=1).all()), f"{qt}: every row keeps a key")
        out = decoder_logits(dec, memory, mask)
        check(torch.isfinite(out).all(), f"{qt}: logits finite",
              f"n_nan = {int((~torch.isfinite(out)).sum())}")


def test_8_queries_distinct():
    """4096 queries that are nearly identical would make the model predict one
    global logit. The usual cause is too small a fourier_scale."""
    print("\n8. QUERY DISTINCTNESS -- no two voxels may share a query vector")
    for qt in QUERY_TYPES:
        _, dec, _, _ = build_decoder(0, voxel_query_type=qt)
        with torch.no_grad():
            q = dec.queries()
            d = torch.cdist(q, q)
            d.fill_diagonal_(float('inf'))
            worst = d.min().item()
        check(worst > 1e-3, f"{qt}", f"min off-diagonal distance = {worst:.4e}")


def test_9_query_label_index_agreement():
    """The highest-value test here: do the query grid and the label grid agree
    on which flat index is voxel (i, j, k)?

    A mismatch trains perfectly happily against a permuted target -- the loss
    goes down, nothing errors, and every spatial conclusion drawn afterwards is
    wrong. So this walks the loader's OWN path: build a grid the way boxer_env
    builds one, take the world coordinate of a chosen cell centre, and let
    loader.coord_to_voxel place it, exactly as __getitem__ does when it writes
    labels['interaction'].
    """
    print("\n9. QUERY/LABEL INDEX AGREEMENT -- the flattening must match exactly")
    from loader import coord_to_voxel                       # the label path itself

    _, dec, _, _ = build_decoder(0)

    # A grid built the way boxer_env._grid_from_points builds one: meshgrid over
    # the three axes with indexing='ij', then flattened C-order to (4096, 3).
    axes = [np.linspace(0.5, 15.5, GRID) * step
            for step in (0.17, 0.27, 0.14)]                 # anisotropic, as real ones are
    mesh = np.stack(np.meshgrid(*axes, indexing='ij'), axis=-1)   # (16,16,16,3)
    vc_flat = mesh.reshape(-1, 3)

    for (i, j, k) in [(0, 0, 0), (5, 8, 6), (15, 15, 15), (3, 14, 1), (12, 0, 9)]:
        flat_expected = i * GRID * GRID + j * GRID + k

        # The loader's exact two lines for a ground-truth point.
        gt = np.zeros(NUM_VOXELS, dtype=np.int64)
        gt[coord_to_voxel(mesh[i, j, k], vc_flat)] = 1
        flat_label = int(gt.argmax())

        want = torch.tensor([(i + 0.5) / GRID, (j + 0.5) / GRID, (k + 0.5) / GRID])
        got = dec.voxel_coords[flat_label]
        ijk1 = dec.voxel_ijk1[flat_label]
        ok = (flat_label == flat_expected
              and torch.allclose(got, want, atol=1e-6)
              and torch.equal(ijk1, torch.tensor([i + 1, j + 1, k + 1])))
        check(ok, f"voxel ({i},{j},{k}) -> flat {flat_expected}",
              f"label flat {flat_label}, q {got.tolist()}")

    check(int(gt.sum()) == 1, "one ground-truth point marks exactly one voxel")

    # And against the real production grids, which are rotated -- so this also
    # confirms the rotation does not disturb the index convention.
    real = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        'new_voxel_centers', 'P07_voxel_centers.npy')
    if os.path.exists(real):
        vc = np.load(real).astype(np.float64)
        centres = vc.reshape(GRID, GRID, GRID, 3)
        for (i, j, k) in [(5, 8, 6), (0, 15, 3)]:
            flat_label = coord_to_voxel(centres[i, j, k], vc.reshape(-1, 3))
            check(flat_label == i * GRID * GRID + j * GRID + k,
                  f"P07 real grid: ({i},{j},{k})", f"flat {flat_label}")
    else:
        print(f"   (skipped the real-grid check: {real} not present)")


def test_11_bias_factorisation():
    """Changing bias_z[k] alone must shift exactly the 256 voxels sharing that k,
    all by the same amount, and nothing else.

    This and test 9 must agree about axis order: k is the fastest-varying index,
    so those 256 voxels are the ones with flat % 16 == k. If one of the two
    passes and the other fails, do not adjust a test -- find which side is wrong.
    """
    print("\n11. BIAS FACTORISATION -- bias_z[k] moves exactly its 256 voxels")
    _, dec, memory, mask = build_decoder(0, voxel_bias='factorised')
    base = decoder_logits(dec, memory, mask)
    k, delta = 6, 3.0
    with torch.no_grad():
        dec.bias_z[k] += delta
    diff = (decoder_logits(dec, memory, mask) - base)

    flat = torch.arange(NUM_VOXELS)
    shifted = (flat % GRID) == k                       # the k-plane, per test 9's formula
    check(shifted.sum().item() == GRID * GRID, "the k-plane holds 256 voxels",
          f"{int(shifted.sum())}")
    check(torch.allclose(diff[:, shifted], torch.full_like(diff[:, shifted], delta),
                         atol=1e-4), "those 256 shift by exactly the delta",
          f"max|err| = {(diff[:, shifted] - delta).abs().max().item():.2e}")
    check(diff[:, ~shifted].abs().max().item() < 1e-4, "every other voxel is untouched",
          f"max|delta| = {diff[:, ~shifted].abs().max().item():.2e}")

    # ... and the other two axes address the other two indices, not the same one.
    for axis, name, index_of in ((0, 'bias_x', flat // (GRID * GRID)),
                                 (1, 'bias_y', (flat // GRID) % GRID)):
        _, dec, memory, mask = build_decoder(0, voxel_bias='factorised')
        base = decoder_logits(dec, memory, mask)
        with torch.no_grad():
            getattr(dec, name)[k] += delta
        diff = decoder_logits(dec, memory, mask) - base
        sel = index_of == k
        ok = (torch.allclose(diff[:, sel], torch.full_like(diff[:, sel], delta), atol=1e-4)
              and diff[:, ~sel].abs().max().item() < 1e-4)
        check(ok, f"{name}[{k}] moves exactly its own plane", f"{int(sel.sum())} voxels")

    # 'full' is the ablation arm: one independent entry per voxel.
    _, dec, memory, mask = build_decoder(0, voxel_bias='full')
    base = decoder_logits(dec, memory, mask)
    with torch.no_grad():
        dec.voxel_bias[1234] += delta
    diff = decoder_logits(dec, memory, mask) - base
    moved = (diff.abs() > 1e-4).any(dim=0)
    check(int(moved.sum()) == 1 and bool(moved[1234]),
          "full: one entry moves exactly one voxel", f"{int(moved.sum())} moved")


def test_12_attention_contract():
    """analysis/voxel_query_attention.py reads two things out of the forward pass
    -- the [B, 4096, L] attention and the segment spans -- and both are silent
    failure modes. A span table that is off by the sep tokens attributes mass to
    the wrong modality and still prints a tidy table, so it is checked here
    rather than discovered in a plot.
    """
    print("\n12. ATTENTION CONTRACT -- what the §7 probe reads must be well formed")
    for modality in ('video,location,obb', 'video,location,obb,gaze'):
        m = build_model(0, input_modality=modality, decoder='voxel_query')
        with torch.no_grad():
            out = m(**make_batch(0), need_attention=True)
        attn, spans = out['attention'], out['segments']
        length = attn.shape[2]
        check(attn.shape == (B, NUM_VOXELS, length), f"{modality}: attention shape",
              f"{tuple(attn.shape)}")
        check(torch.allclose(attn.sum(-1), torch.ones(B, NUM_VOXELS), atol=1e-4),
              f"{modality}: each voxel's attention sums to 1")
        # Spans must tile [0, L) exactly: no gap, no overlap, nothing dropped.
        covered, at, ok = [], 0, True
        for name, s, e in spans:
            ok &= (s == at)
            at = e
            covered.append(name)
        check(ok and at == length, f"{modality}: spans tile the sequence exactly",
              f"{at} vs L={length}")
        expected = set(modality.split(',')) | {'sep'}
        if READOUT == 'cls':
            expected.add('cls')
            check(spans[0][0] == 'cls', f"{modality}: cls is seat 0")
        check(set(covered) == expected, f"{modality}: every segment is named",
              f"{sorted(set(covered))}")

    # Padded keys must receive no mass -- the probe would otherwise attribute
    # attention to slots holding nothing.
    _, dec, memory, mask = build_decoder(0)
    with torch.no_grad():
        _, w = dec(memory, mask, need_weights=True)
    check(w[:, :, mask[0]].abs().max().item() < 1e-6, "padded keys get zero mass",
          f"max = {w[:, :, mask[0]].abs().max().item():.2e}")


def test_13_checkpoint_roundtrip():
    """A checkpoint must save, reload strictly, and reproduce its logits.

    This exists because it already failed once: the decoder shares the model's
    location tables, and registering them as decoder submodules put them in the
    state dict twice, which makes save_pretrained refuse to write safetensors
    ("shared tensors ... mismatching the transformers base configuration"). Every
    axis_sum run would have died at its first checkpoint, hours in. The reverse
    direction matters too -- a pre-change checkpoint must NOT load into a
    voxel_query model, since strict=False there would leave a random decoder that
    still produces plausible numbers.
    """
    import shutil
    import tempfile
    print("\n13. CHECKPOINT ROUND TRIP -- save, strict-load, identical logits")
    tmp = tempfile.mkdtemp(prefix='smoke_decoder_')
    try:
        for qt in QUERY_TYPES:
            m = build_model(0, input_modality=DECODER_MODALITY,
                            decoder='voxel_query', voxel_query_type=qt)
            n_shared = sum('location_embedding_x.weight' in k for k in m.state_dict())
            check(n_shared == 1, f"{qt}: shared tables appear once in the state dict",
                  f"{n_shared}")
            path = os.path.join(tmp, qt)
            try:
                m.save_pretrained(path)
            except RuntimeError as e:
                check(False, f"{qt}: save_pretrained", str(e).splitlines()[0][:40])
                continue
            reloaded = CustomTransformerModel.from_pretrained(path).eval()
            batch = make_batch(0)
            check(torch.equal(logits_of(m, batch), logits_of(reloaded, batch)),
                  f"{qt}: reloaded model reproduces its logits")

        m_lin = build_model(0, input_modality=DECODER_MODALITY, decoder='linear')
        m_vq = build_model(0, input_modality=DECODER_MODALITY, decoder='voxel_query')
        try:
            m_vq.load_state_dict(m_lin.state_dict())
            check(False, "pre-change checkpoint refused by a voxel_query model",
                  "it LOADED")
        except RuntimeError:
            check(True, "pre-change checkpoint refused by a voxel_query model")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_prior_init():
    """Stage B: the prior logit must reach whichever bias is in play, and the
    factorised form must SUM to it rather than to three times it."""
    print("\nB. PRIOR INIT -- the class prior reaches the bias, once")
    p = -5.3
    m = build_model(0, input_modality=DECODER_MODALITY, decoder='linear',
                    voxel_bias_init='prior')
    check(m.apply_voxel_bias_init(p), "linear: init applied")
    check(torch.allclose(m.output_layer.bias, torch.full((NUM_VOXELS,), p)),
          "linear: every output bias entry is the prior logit")

    for bias_type in ('factorised', 'full'):
        m = build_model(0, input_modality=DECODER_MODALITY, decoder='voxel_query',
                        voxel_bias=bias_type, voxel_bias_init='prior')
        m.apply_voxel_bias_init(p)
        with torch.no_grad():
            b = m.decoder.bias()
        check(torch.allclose(b, torch.full((NUM_VOXELS,), p), atol=1e-5),
              f"{bias_type}: bias sums to the prior logit", f"{b[0].item():+.4f}")

    m = build_model(0, input_modality=DECODER_MODALITY, decoder='linear',
                    voxel_bias_init='zero')
    before = m.output_layer.bias.clone()
    check(not m.apply_voxel_bias_init(p), "zero: init is a no-op")
    check(torch.equal(m.output_layer.bias, before), "zero: bias untouched")


def test_14_decoder_residual():
    """--decoder_residual must change the query:evidence balance and nothing else.

    The point of the flag is a magnitude claim, so the test measures magnitudes
    rather than just asserting the logits moved: 'raw' keeps the embedding-scale
    query (|q| ~ 55 against |attn_out| ~ 1), 'normed' halves that to sqrt(D), and
    'balanced' puts both sides at sqrt(D). If a future init change makes those
    ratios untrue, the flag has stopped doing what its docstring says.
    """
    print("\n14. DECODER RESIDUAL -- query:evidence balance is what the flag controls")
    ratios = {}
    for mode in ('raw', 'normed', 'balanced'):
        model, dec, memory, mask = build_decoder(0, decoder_residual=mode)
        with torch.no_grad():
            q = dec.queries().unsqueeze(1).expand(-1, memory.shape[1], -1)
            qn = dec.ln_q(q)
            a, _ = dec.attn(qn, dec.ln_m(memory), dec.ln_m(memory),
                            key_padding_mask=mask, need_weights=False)
            lhs = q if mode == 'raw' else qn
            rhs = a if mode != 'balanced' else dec.ln_a(a)
            r = (lhs.norm(dim=-1).mean() / rhs.norm(dim=-1).mean()).item()
        ratios[mode] = r
        out = decoder_logits(dec, memory, mask)
        check(out.shape == (memory.shape[1], NUM_VOXELS) and torch.isfinite(out).all(),
              f"{mode}: finite [B, 4096]", f"query:evidence = {r:.1f}:1")
        check(('ln_a.weight' in dict(dec.named_parameters())) == (mode == 'balanced'),
              f"{mode}: ln_a exists only under 'balanced'")
        dec(memory, mask)[0].sum().backward()   # decoder_logits() is no_grad
        dead = [n for n, p in dec.named_parameters() if p.grad is None or not p.grad.any()]
        check(not dead, f"{mode}: every decoder parameter trains", f"dead: {dead or 'none'}")

    check(ratios['raw'] > 3 * ratios['balanced'],
          "raw is far more query-dominated than balanced",
          f"{ratios['raw']:.1f}:1 vs {ratios['balanced']:.1f}:1")
    check(abs(ratios['balanced'] - 1.0) < 0.35, "balanced really is ~1:1",
          f"{ratios['balanced']:.2f}:1")
    check(ratios['normed'] < ratios['raw'], "normed sits between the two",
          f"{ratios['normed']:.1f}:1")

    # The arms must not be interchangeable. HF's from_pretrained does NOT refuse a
    # raw checkpoint here -- ln_a is simply a missing key, so it warns and inits it
    # fresh. That is safe (LayerNorm's default init is well defined) but it means
    # nothing stops you loading one arm's weights under another's flag, so the
    # invariant worth pinning is that the two then disagree: identical weights plus
    # a different residual is a different model, never a silent no-op.
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        m_raw = build_model(0, decoder='voxel_query', input_modality=DECODER_MODALITY,
                            decoder_residual='raw')
        m_raw.save_pretrained(td)
        reloaded = CustomTransformerModel.from_pretrained(
            td, config=build_config(decoder='voxel_query',
                                    input_modality=DECODER_MODALITY,
                                    decoder_residual='balanced')).eval()
        check('ln_a.weight' in dict(reloaded.decoder.named_parameters()),
              "raw checkpoint loaded as balanced gains a fresh ln_a")
        with torch.no_grad():
            a = m_raw.decoder(memory, mask)[0]
            b = reloaded.decoder(memory, mask)[0]
        check((a - b).abs().max().item() > 1e-3,
              "the two arms disagree on identical weights",
              f"max|delta| = {(a - b).abs().max().item():.3e}")

    # --- The two arms that REMOVE the query rather than rescaling it ---------
    # raw/normed/balanced only change how loud q is, and measured on fold P01 they
    # land together (.0463/.0460/.0495). These two cut it out, so what is asserted
    # here is the FORMULA, not a magnitude: the decisive property of 'evidence' is
    # that q contributes exactly nothing to h, and a ratio test cannot show that
    # (q still reaches the logits through the attention, as the query -- which is
    # the whole point of the arm and would make a naive "change q, see if logits
    # move" test fail for the right reason).
    for mode in ('evidence', 'concat'):
        model, dec, memory, mask = build_decoder(0, decoder_residual=mode)
        d_model = model.config.transformer_hidden_size
        with torch.no_grad():
            q = dec.queries().unsqueeze(1).expand(-1, memory.shape[1], -1)
            qn = dec.ln_q(q)
            a, _ = dec.attn(qn, dec.ln_m(memory), dec.ln_m(memory),
                            key_padding_mask=mask, need_weights=False)
            h = dec.ln_a(a) if mode == 'evidence' else torch.cat([qn, dec.ln_a(a)], dim=-1)
            expected = (dec.head(dec.ln_o(h)).squeeze(-1).transpose(0, 1) + dec.bias())
            actual = dec(memory, mask)[0]
        check(actual.shape == (memory.shape[1], NUM_VOXELS) and torch.isfinite(actual).all(),
              f"{mode}: finite [B, 4096]")
        # This is the assertion that matters for 'evidence': `expected` is written
        # with NO q term, so agreeing with forward() is what proves q was removed.
        check((actual - expected).abs().max().item() < 1e-5,
              f"{mode}: forward matches its stated formula",
              f"max|delta| = {(actual - expected).abs().max().item():.2e}")
        check('ln_a.weight' in dict(dec.named_parameters()),
              f"{mode}: ln_a is built (it is not, under raw/normed)")
        width = dec.head.in_features
        check(width == (d_model * 2 if mode == 'concat' else d_model),
              f"{mode}: head width", f"Linear({width} -> 1)")
        check(dec.ln_o.normalized_shape[0] == width,
              f"{mode}: ln_o matches the head's width")
        dec(memory, mask)[0].sum().backward()
        dead = [n for n, p in dec.named_parameters() if p.grad is None or not p.grad.any()]
        check(not dead, f"{mode}: every decoder parameter trains", f"dead: {dead or 'none'}")

    # The structural consequence, stated as behaviour rather than as a formula.
    # Hand every voxel the SAME attn_out. Under 'evidence' h is then identical for
    # all 4096, so once the per-voxel bias is subtracted every logit must collapse
    # to one number -- there is no other route by which a voxel's identity could
    # reach the output. Under 'raw' the same substitution leaves q in place and the
    # logits stay spread out, which is the contrast being pinned.
    class _ConstAttn(torch.nn.Module):
        """Stands in for the MultiheadAttention and hands every voxel one vector.
        A Module rather than a lambda because nn.Module.__setattr__ refuses to
        register anything else in place of a child module."""

        def __init__(self, const):
            super().__init__()
            self.const = const

        def forward(self, *args, **kwargs):
            return self.const, None

    spreads = {}
    for mode in ('evidence', 'raw'):
        model, dec, memory, mask = build_decoder(0, decoder_residual=mode)
        d_model = model.config.transformer_hidden_size
        const = torch.randn(1, 1, d_model).expand(NUM_VOXELS, memory.shape[1], d_model)
        real_attn = dec.attn
        dec.attn = _ConstAttn(const)
        with torch.no_grad():
            spreads[mode] = (dec(memory, mask)[0] - dec.bias()).std(dim=1).max().item()
        dec.attn = real_attn
    check(spreads['evidence'] < 1e-5,
          "evidence: identical attn_out => identical logits, bias aside",
          f"spread across voxels = {spreads['evidence']:.2e}")
    check(spreads['raw'] > 1e-3,
          "raw: identical attn_out still gives spread (q reaches the output)",
          f"spread across voxels = {spreads['raw']:.3f}")

    # 'concat' splits head.weight into a q half and an evidence half. That split is
    # the readout the arm exists for, so the halves must be separately addressable.
    m_c, dec_c, _, _ = build_decoder(0, decoder_residual='concat')
    dm = m_c.config.transformer_hidden_size
    wq, wa = dec_c.head.weight[0, :dm], dec_c.head.weight[0, dm:]
    check(wq.numel() == dm and wa.numel() == dm,
          "concat: head.weight splits into two readable halves",
          f"|w_q| = {wq.norm():.3f}, |w_attn| = {wa.norm():.3f}")

    # concat + FFN is a width mismatch; it must be refused where the message can
    # still name both flags, not deep inside forward().
    try:
        build_decoder(0, decoder_residual='concat', decoder_ffn=True)
        check(False, "concat + decoder_ffn is refused at construction")
    except ValueError as e:
        check('concat' in str(e) and 'ffn' in str(e).lower(),
              "concat + decoder_ffn is refused at construction", str(e)[:60] + "...")


def test_10_cls_readout():
    """Stage A: the [CLS] token is a real, used, extra seat.

    Three things can go wrong silently and each is checked: the token exists but
    nothing reads it (logits identical to token0), it is created but masked out
    (no gradient), or the padding mask is not widened to match the sequence --
    which would misalign every mask column by one and is invisible in the output
    shape.
    """
    print("\n10. CLS READOUT -- a dedicated seat that is actually read")
    for modality in REFERENCE_MODALITIES:
        m0 = build_model(0, input_modality=modality, readout='token0')
        mc = build_model(0, input_modality=modality, readout='cls')
        batch = make_batch(0)
        l0, lc = logits_of(m0, batch), logits_of(mc, batch)
        check(lc.shape == (B, NUM_VOXELS), f"{modality} shape", f"{tuple(lc.shape)}")
        check((lc - l0).abs().max().item() > 1e-6, f"{modality} cls changes the readout")
        # Every pre-existing parameter must be drawn from the same RNG stream
        # under both readouts, so a cls run differs by the token alone.
        check(torch.equal(m0.output_layer.weight, mc.output_layer.weight),
              f"{modality} shared weights untouched by --readout")

    m = build_model(0, input_modality=REFERENCE_MODALITIES[1], readout='cls')
    b = make_batch(0)
    b['labels'] = {'interaction': torch.randint(0, 2, (B, NUM_VOXELS))}
    m.zero_grad()
    m(**b)['loss'].backward()
    g = m.cls_token.grad
    check(g is not None and g.abs().sum().item() > 0, "cls_token receives gradient",
          f"|grad| = {0.0 if g is None else g.abs().sum().item():.6f}")

    # A masked-out CLS seat would still produce logits and still train; the only
    # way to see it is to confirm the mask grew with the sequence.
    seen = {}
    orig = torch.nn.TransformerEncoder.forward

    def spy(self, src, mask=None, src_key_padding_mask=None, **kw):
        seen['L'] = src.shape[0]
        seen['mask'] = src_key_padding_mask
        return orig(self, src, mask=mask, src_key_padding_mask=src_key_padding_mask, **kw)

    torch.nn.TransformerEncoder.forward = spy
    try:
        logits_of(build_model(0, input_modality='video,location,obb', readout='token0'),
                  make_batch(0))
        l_tok, m_tok = seen['L'], seen['mask']
        logits_of(build_model(0, input_modality='video,location,obb', readout='cls'),
                  make_batch(0))
        l_cls, m_cls = seen['L'], seen['mask']
    finally:
        torch.nn.TransformerEncoder.forward = orig
    check(l_cls == l_tok + 1, "sequence grows by exactly one seat", f"{l_tok} -> {l_cls}")
    check(tuple(m_cls.shape) == (B, l_cls), "padding mask widened with it",
          f"{tuple(m_tok.shape)} -> {tuple(m_cls.shape)}")
    check(not m_cls[:, 0].any(), "cls column is attendable (False)")
    check(torch.equal(m_cls[:, 1:], m_tok), "the rest of the mask is unshifted")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--capture', action='store_true',
                    help="Re-record the reference logits. Only legitimate before the "
                         "decoder work landed; never as a fix for a failing test 1.")
    args = ap.parse_args()
    if args.capture:
        capture_reference()
        sys.exit(0)

    test_1_regression()
    test_10_cls_readout()

    # Spec test 10: the rest of the suite must hold under BOTH readouts, since
    # --readout cls changes the sequence the decoder attends over.
    for readout in ('token0', 'cls'):
        READOUT = readout
        globals()['READOUT'] = readout
        print(f"\n{'-' * 60}\n--readout {readout}\n{'-' * 60}")
        test_2_shape()
        test_3_key_permutation_invariance()
        test_4_query_permutation_equivariance()
        test_5_pad_inert()
        test_6_gradient_flow()
        test_7_single_attendable_key()
        test_8_queries_distinct()
        test_9_query_label_index_agreement()
        test_11_bias_factorisation()
        test_12_attention_contract()
        test_13_checkpoint_roundtrip()
        test_14_decoder_residual()
        test_prior_init()

    print("\n" + "=" * 60)
    if FAILS:
        print("FAILURES:")
        for f in FAILS:
            print(f"  - {f}")
        sys.exit(1)
    print("ALL DECODER CHECKS PASSED")
