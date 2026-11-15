# Implementation spec: time encoding and continuous position encoding

Target repo: `FIction_HD_EPIC_v2`, branch `arch_method`. Companion to
`voxel_query_decoder_spec.md`, which this assumes has landed (`decoder.py`, `--readout`,
`--voxel_bias_init`, `--decoder`).

Two independent changes, in order:

| | Flag | What it adds | Files |
|---|---|---|---|
| **A** | `--time_encoding fourier` | Learned Fourier encoding of time-to-anchor on every per-second token | `model.py` only |
| **B1** | `--position_encoding fourier` | Learned Fourier encoding of the voxel **centre**, shared by the input tokens and the decoder's queries | `model.py`, `decoder.py` |
| **B2** | *(not built)* | Sub-voxel resolution: the same code over the **continuous** position instead of the cell centre | `loader.py`, `boxer_env.py` |

Both default to off. With both off the forward pass must stay **bit-identical** to the
pre-change model — `smoke_decoder.py` test 1 already pins this and its reference tensor
stays valid precisely because the defaults do not move.

**A is implemented. B1 is implemented** (2026-08-15, `smoke_position_encoding.py`), and
split out of B for a reason worth recording: B as originally written bundled two changes
that answer different questions. Sharing one spatial code between queries and input tokens
(B1) is free of new data and testable on its own; recovering sub-voxel position (B2) needs
the loader to stop discarding it and carries the LOPO memorisation risk predicted below.
**B1 does not touch the resolution at all** — it encodes the same cell centre the index
tables encoded, so §2.1's rotation-aware projection is still unbuilt and still required
before B2 means anything.

### What B1 actually changed, and the confound it removed

`--voxel_query_type fourier` already existed, but its queries used a **private** `W_r`
while every input token kept the index tables. The two sides shared no parameters, so the
nine `vq_fourier` runs of 2026-08-11 were never the "coordinates vs lookup" comparison they
read as — they also swapped a shared vocabulary for two unrelated ones.

A second confound sat underneath it. cos/sin are bounded, so `W_r` sets frequencies and can
never set amplitude: the queries were pinned at norm **0.707** against `|attn_out| ~ 1.11`,
where `axis_sum` sits at **55.7**. Under the default `--decoder_residual raw` that is not a
smaller version of the same model, it is a 0.6:1 query:evidence regime against a 50:1 one.
Those runs measured the residual balance at least as much as the encoding.

B1 fixes both: one `spatial_W_r` owned by the model and *called* by the decoder (so the
codes are identical by construction, not by two call sites agreeing on a formula), plus a
learned `spatial_gain` initialised to `sqrt(3*768) = 48.0`, the measured norm of the three
summed index rows it replaces. Measured after the change: query 55.6 vs axis_sum 55.5,
input position 48.0 vs 48.1. Same `FAST_LR` group as `time_gain`, for the same reason.

`--position_encoding` defaults to `auto` (follow `voxel_query_type`); `none` reproduces the
2026-08-11 arm exactly, private `W_r` and all, so those checkpoints still load.

**Scope limit worth stating plainly:** B1 reaches only the tokens carrying a full `(i,j,k)`
— `obb`, `env`, and `trajectory` endpoints. `location` and `gaze` are fed as *three separate
per-axis token streams*, so a token there holds one scalar and not a coordinate; a 3-D
Fourier code has nowhere to go. Covering them means either merging each second's three
tokens into one or defining a per-axis restriction of the code, and neither is built.

---

## 0. Why these two, and why they are not the same kind of change

The model currently has **no positional encoding of any kind** (`CLAUDE.md`, "Model /
training"). Two consequences, one temporal and one spatial:

1. Every per-second modality is an unordered *set* to self-attention. Permuting the 30
   seconds of video, location, or gaze leaves the output unchanged. "What was I looking at
   two seconds ago" is not a question this model can currently ask — which is a strange
   handicap for a task whose entire premise is anticipation. Gaze suffers most: without
   order it is close to a bag of voxels.
2. Every spatial input is snapped to a voxel index, 1–16 per axis. Sub-voxel position —
   roughly 17–27 cm depending on participant and axis — is computed inside
   `loader.coord_to_voxel` and then discarded.

**These two changes carry very different risk under LOPO, and that difference should be
stated before the results arrive.**

Time-to-anchor is **kitchen-agnostic**: "2 seconds ago" means the same thing in every
kitchen, so A adds a recency prior without adding any capacity to identify *which* room
the model is in.

Continuous position is the opposite. Measured on the nine-fold `voxel_query` runs
(2026-08-11): held-out PR-AUC peaks at the **first** evaluation, a quarter of one epoch,
and declines monotonically thereafter while train loss keeps falling; on several folds a
single static heat map — one map, identical for every episode — scores *higher* than the
model (P08: 0.372 static vs 0.320 model). The dominant failure mode is the model
identifying the training kitchen and recalling its hotspots. Sub-voxel geometry makes that
identification **easier**.

> **Prediction, recorded in advance:** A helps or is neutral; B improves training loss and
> may well *worsen* held-out PR-AUC. If B helps on held-out folds, the memorisation account
> above is incomplete and that is the more interesting result.

---

## 1. Change A — time encoding

### 1.1 The quantity

Time **relative to the anchor**, not absolute time in the recording:

```
Δ_n = (T - 1 - n) / T          n = 0 .. T-1, T = observation_time (30)
```

so Δ ∈ [0, 1), with Δ = 0 at the last observed second and Δ → 1 at the oldest. Relative
because it is invariant to where the window sits in the recording, and because recency is
the quantity that should matter.

### 1.2 Why the index gives Δ directly

All three per-second modalities are **end-aligned**, so slot `n` is exactly `T-1-n` seconds
before `obs_end`. Verify, don't assume:

- `location`: `loader.py` walks `t` from `obs_end-1` downwards filling slot 29 first, and
  leaves gaps as padding rather than closing them.
- `gaze`: same construction, same shape.
- `video`: pads at the **front** (`torch.cat([zeros(pad_len), obs_features])`) and packs the
  available features contiguously. That is only equivalent to end-alignment if the features
  are dense at 1 Hz. **Measured: 0 of 3,516 consecutive frame indices on P08's videos differ
  from +1 s.** Assert it in the model rather than trusting it — if a gappy video ever
  appears, the assertion should fire rather than silently mislabelling Δ.

### 1.3 The encoding

LLaVA-4D Eq. 6, one-dimensional:

```python
self.time_W_r = nn.Parameter(torch.empty(d_model // 2, 1))
nn.init.normal_(self.time_W_r, std=config.time_encoding_scale)
...
delta = ((T - 1 - arange(T)) / T).unsqueeze(-1)        # [T, 1] in [0,1)
proj  = delta @ self.time_W_r.T                        # [T, 384]
te    = cat([proj.cos(), proj.sin()], -1) / sqrt(d)    # [T, 768]
```

384 parameters. Fourier rather than `nn.Embedding(30, 768)` for the same reason positional
encoding beats a lookup: Δ = 3 s and Δ = 4 s should be *near* each other, so the model can
learn a smooth recency prior instead of 30 unrelated vectors.

`time_encoding_scale` is the **std of `W_r`'s init**, not a range for Δ. With Δ ∈ [0,1] and
std 10, a typical component sweeps ~10 radians ≈ 1.6 cycles across the window: smooth, with
adjacent seconds close together. Too small and all 30 seconds collapse toward one vector;
too large and adjacent seconds alias. Default 10.0, matching `voxel_query_fourier_scale`.
Ablate {2, 10, 30}.

### 1.4 Where it is added

| Segment | TE? | Why |
|---|---|---|
| video (30) | **yes** | one token per second |
| location (3 × 30) | **yes** | same Δ on all three axis sequences of a given second |
| gaze (3 × 30) | **yes** | same; expected to benefit most |
| obb | **no** | a fused box has no single timestamp — it is an aggregate over the window |
| env | **no** | one token per voxel, not per second |
| trajectory | **no** | already carries sinusoidal endpoint times (`trajectory_tokens.py`); TE would double-encode |
| sep / cls | **no** | not per-second tokens |

Padding slots receive TE like any other, and are masked out of attention regardless.

### 1.5 Flags

```
--time_encoding {none,fourier}     default none
--time_encoding_scale FLOAT        default 10.0
```

Env passthrough `TIME_ENCODING`, `TIME_ENCODING_SCALE` in `train_sbatch.sh`.

---

## 2. Change B — continuous position encoding (B1 built; B2, the continuous part, not yet)

> **Status.** Everything in §2.2–2.5 below is now live *at voxel-centre resolution* via
> `--position_encoding`, sharing one `spatial_W_r` with the decoder's queries — except the
> "Continuous source" column of §2.2's table, which is precisely what remains. Read §2.1 as
> the spec for B2.
>
> One caveat §2.4 raised has been acted on: `--voxel_query_fourier_scale` now accepts three
> comma-separated values for a per-axis init. The anisotropy is not a rounding detail —
> measured over the nine production grids, the z cell is 13.3–14.6 cm in *every* kitchen
> while x runs 16.8–41.7 cm and y 12.6–46.6 cm, so a single scale resolves height several
> times more finely than either floor axis.

### 2.1 The quantity, and the correction the obvious formula needs

The naive normalisation `(x − grid_min_x) / (grid_max_x − grid_min_x)` assumes a
world-axis-aligned box. **The production grids are rotated per participant** — measured
yaws: P01 3.56°, P02 9.10°, P03 −18.36°, P04 −24.33°, P05 23.00°, P06 22.12°, P07 35.15°,
P08 23.62°, P09 0.00°. The world-frame AABB of a rotated box is not the grid, and
normalising by it is neither frame-invariant nor aligned to the label axes.

Project onto the lattice axes instead:

```
x̃_a = ((p − origin) · û_a) / extent_a          a ∈ {0, 1, 2}
```

where `origin`, `û_a`, `extent_a` come from the participant's voxel-centre `.npy`.
`analysis/grid_geometry_audit.py` already recovers exactly this (its rotation-aware lattice
section) — reuse it rather than reimplementing.

Axis order must match the label flattening, `flat = i*256 + j*16 + k`, so that
`PE(centre of voxel (i,j,k))` coincides with the decoder's `fourier` query for that flat
index. That is a testable coherence property.

**Out-of-grid points are clamped to [0,1]**, matching how the current pipeline clamps to
boundary voxels. (For the record: `grid_geometry_audit.py` measured P01 clamping **9.0%** of
its points — 3× the next worst. Representing "outside" honestly is a possible future change;
it is deliberately *not* part of this one, so B changes only resolution, not semantics.)

### 2.2 What it recovers, and where

One shared module, applied to every modality that carries a position:

| Modality | Continuous source | Currently |
|---|---|---|
| location | SLAM world position per second | voxel index 1–16 |
| gaze | gaze point per second | voxel index of densest cell |
| obb | fused box centre | `_nearest_voxel_ijk` centre |
| trajectory | track endpoints (`trajectory_simple`) | start/end voxel index |
| env | the voxel's own centre | index, via `--env_positional_encoding` |

`env` is included as an **opt-in argument**: an env token's position *is* its voxel centre,
so PE there is a smooth alternative to the existing `--env_positional_encoding`, and the two
should be selectable independently so they can be compared.

### 2.3 Added, not substituted

`p_xyz` is **added to** the existing index-embedding sum, not in place of it. The lookup
tables can express non-smooth structure a Fourier basis cannot ("the counter runs along the
k=8 slab"); PE adds continuity and sub-voxel detail. Which wins is genuinely open — ablate
`{index only, index+PE, PE only}`.

### 2.4 Reuse, don't duplicate

`decoder.py` already implements this exact encoding for `voxel_query_type='fourier'`
(`W_r ∈ ℝ^{384×3}`, `cos ‖ sin`, `/√d`). Factor it into one shared
`FourierPositionalEncoding` used by both the decoder's queries and the input tokens — the
same "one table, many modalities" property the repo already maintains for
`location_embedding_x/y/z` and `class_embedding`.

Note the anisotropy caveat carried over from the decoder spec: one scale covers all three
axes, but voxels are not cubic (P01: 0.168 / 0.267 / 0.137 m). `W_r` is learnable and can
compensate; the init cannot.

### 2.5 Flags

```
--position_encoding {none,fourier}          default none
--position_encoding_scale FLOAT             default 10.0
--position_encoding_modalities STR          default "location,gaze,obb,trajectory"
                                            (add "env" to include env tokens)
```

---

## 3. Tests

### 3.1 What must not change

`smoke_decoder.py` test 1: with `--time_encoding none --position_encoding none` (and the
existing decoder defaults) the logits stay bit-identical to `smoke_decoder_reference.pt`.
This is why both flags default to off and why the reference does not need recapturing.

### 3.2 What A deliberately breaks

Temporal permutation invariance is currently an *asserted* property. After A:

| Check | Before | After A |
|---|---|---|
| permuting the 30 video / location / gaze seconds | logits unchanged | **logits must change** |
| permuting OBB tokens (`smoke_obb.py` test 4) | logits unchanged | **still unchanged** |
| permuting decoder memory (`smoke_decoder.py` test 3) | unchanged | **still unchanged** |

The third stays because it permutes the encoder's *output*; cross-attention remains a set
operation over keys regardless of what the encoder did.

New checks in `smoke_time_encoding.py`:

1. **Off is off.** `--time_encoding none` reproduces the current logits exactly, and
   `time_W_r` receives no gradient.
2. **Order now matters.** Reversing the 30 seconds changes the logits under `fourier`.
3. **The right segments.** With TE on, permuting *only* the OBB tokens still leaves logits
   unchanged — TE has not leaked into a segment that has no time.
4. **Δ orientation.** The last slot is Δ=0 and the first is Δ→1, verified against the
   loader's end-alignment: shifting a signal one second later must move its token by exactly
   one slot.
5. **Density assertion fires.** A synthetic gappy `frame_indices` triggers the assertion
   rather than silently mislabelling Δ.
6. **Gradient.** `time_W_r` trains under `fourier`.
7. **Distinctness.** The 30 time codes are pairwise separated (`cdist` off-diagonal min >
   1e-3) — catches a collapsed `time_encoding_scale`, the analogue of decoder test 8.

---

## 4. Evaluation

The nine-fold runs of 2026-08-11 established two things that change how any of this should
be measured:

- **Held-out PR-AUC peaks at the first evaluation** (step ~475–511 of ~19,000) and declines
  thereafter. A 10-epoch schedule measures mostly the decay. Evaluate with
  `NUM_EPOCHS=1 EVAL_STEPS=0.01` — 100 evaluations across one epoch — which both locates the
  true peak (still unmeasured) and costs about a tenth as much.
- **A static per-fold heat map matches or beats the model** on several folds. Report every
  result against that baseline, not against PR-AUC alone; otherwise a change that merely
  sharpens the unconditional prior will read as a change that improves anticipation.

Run on two or three folds spanning the range (P05 worst, P06 mid, P08 best) rather than all
nine, per `voxel_query_decoder_spec.md` §8b.

**A specific check that A is doing what it claims**, independent of PR-AUC:
`analysis/voxel_query_attention.py` writes `by_delta_t.csv`, the video segment's attention
mass by Δt. If TE works, that profile should sharpen toward Δt → 0 relative to the same run
without it. A flat profile with TE on means the model was handed recency and declined to use
it — which is itself worth knowing, and is not visible in a metric.
