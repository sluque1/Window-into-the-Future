# Implementation spec: coordinate-query cross-attention decoder

Target repo: `FIction_HD_EPIC_v2`. Files touched: `model.py`, `train.py`, `trainer.py` (minor),
new `smoke_decoder.py`, new `analysis/voxel_query_attention.py`.

**Do not trust the line numbers below.** They come from an earlier reading of the repo and may
have drifted. Locate the code by symbol name and verify the surrounding logic before editing.

---

## 0. Motivation (keep this in mind, it constrains the design)

Current readout, `model.py` ~189 and ~513-519:

```python
self.output_layer = nn.Linear(config.transformer_hidden_size, config.output_dim)  # 768 -> 4096
...
aggregate_feature = transformer_output[0]     # token 0 only
logits = self.output_layer(aggregate_feature) # [B, 4096]
```

Two problems this change addresses:

1. **The readout token is a content token.** Seat 0 is always the first visual-feature token
   (second 0 of the observation window, EgoVLPv2). It carries specific semantic content *and* is
   asked to serve as the global summary. Nothing was designed for this.
2. **The output layer has one independent 768-vector per voxel.** 3.15M parameters, and voxel
   (5,8,6) shares nothing with voxel (5,8,7). With participant-level splits (one kitchen per
   participant) this is a large capacity sink positioned exactly where memorising training-kitchen
   voxel priors would pay off. It also gives the model no mechanism to route evidence about a
   *region* to the voxels in that region.

The replacement inverts the relationship: the **voxel coordinate becomes the query**, the
observation tokens become keys/values, and a **single shared head** maps each attended vector to
one logit. Prediction at a voxel must then come from evidence retrieved for that location.

Secondary benefit, and arguably the main reason to do this: the attention map is directly
interpretable. For a correctly predicted voxel you can read which observation tokens it drew from
— late video tokens (anticipation) vs. an OBB token for a co-located object (co-occurrence prior).
That is a much sharper probe of the "is FICTION anticipating or just learning co-occurrence"
question than ablation deltas.

---

## 0b. Grid geometry — read this before §4, it constrains the design

The 16³ grid is built **per participant**, from the bounding box (oriented — see (i)) of (all that participant's Boxer OBB
corners ∪ the full SLAM trajectory), then normalised to that box. Splits are participant-level
(LOPO: 8 kitchens train, 1 held out). Three facts follow, and they are not obvious:

**(i) `z` transfers across kitchens. Horizontal *structure* transfers; horizontal *position* does
not.** Two separate facts combine here.

Two frames are in play throughout this document, and they are not the same:

- **World frame** — the Aria/SLAM axes. `z` up; the horizontal directions are whatever SLAM
  initialisation produced, unrelated to the room.
- **Grid frame** — the directions the voxel grid's own `i`, `j`, `k` run along; the box's borders.
  Rotated about `z` relative to the world frame, per participant (see (iv)).

`z` is clean and identical in both frames: the world frame is gravity-aligned, so `z` is up and
normalised height means roughly the same thing everywhere — floor, counter / interaction band,
upper cabinets.

Horizontally, each participant's grid frame has been **manually rotated about `z`** so the box hugs
that kitchen's footprint. Because kitchens are roughly rectangular rooms, hugging one means the
**grid frame's** borders end up running parallel to the counters and walls. (The world frame's
horizontal axes are *not* wall-parallel — that is precisely why the rotation was needed.) This is a
real gain over a world-axis-aligned box: voxel boundaries follow physical structure instead of
cutting diagonally across it, and the occupied region fills the grid more efficiently.

But fitting to shape does not fix *which* wall is `+x`. It reduces a continuous arbitrariness to a
discrete one — the eight symmetries of a rectangle (4 rotations × 2 reflections) — and nothing
selects among them. Two of the three ambiguities are known to be live: horizontal extents are not
consistently ordered (some kitchens are wider than long, so the long axis is not always `x`), and
mirrored layouts (counter on the left vs the right) are a genuine feature of the world that no
rotation removes. With 8 training kitchens each landing in one of 8 orientations, a model cannot
average this out.

Consequence, and this is the operative sentence:

> **Absolute horizontal position does not correspond across kitchens; relative horizontal structure
> does.** Voxel (5,8,·) is not the same place in P01 and P07, but "along the wall" and "across the
> room" mean the same thing in both.

**Both halves of this are now measured, not assumed.** The long axis is inconsistently assigned:
4 participants are x-long, 5 are y-long, with extremes at P03 (2.50× y-long) and P04 (2.57× x-long).
So the 90° ambiguity is live and horizontal position genuinely does not correspond. Conversely the
z-extent spread is small — 2.016 m (P02) → 2.332 m (P04), mean 2.188, i.e. −7.9%/+6.6%, well inside
the ±20% threshold — so normalised height *is* comparable across kitchens and a decoder height prior
is safe on that axis.

Sign consistency remains unmeasured: an occupied-voxel centroid cannot detect a sign flip, since
flipping maps `c → 1−c` and all nine centroids sit near 0.5 (axis0 0.344–0.505, axis1 0.415–0.557).
Settling it needs an asymmetric landmark — sink/hob position, or the trajectory's dominant direction
— not a centroid. Until then, assume signs are arbitrary.

Design accordingly — see the factorised bias in §4.3 and the ablation prediction in §8c.

**(ii) Voxel size differs per participant — but this does NOT drive the PR-AUC spread. Tested and
rejected.** The box is a union over all of that participant's sessions, so participants who recorded
more or roamed further get a larger box, always divided into 16³ cells. Measured range of
`voxel_volume_m3`: 0.00598 (P04) → 0.01388 (P07), a 2.32× spread.

The hypothesis was that larger voxels lower the positive rate, and that PR-AUC's prevalence
sensitivity then produces the observed per-participant spread. **It does not hold**, and the dynamic
ranges settle it before the correlations matter:

| quantity | range | ratio |
|---|---|---|
| `voxel_volume_m3` | 0.00598 → 0.01388 | 2.32× |
| `mean_pos_rate` | 0.00445 → 0.00634 | 1.42× |
| PR-AUC | 0.0195 → 0.2370 | 12.17× |

Prevalence moves 1.4×; PR-AUC moves 12×. Spearman(volume, pos_rate) = −0.317 (p=0.41, right sign
but weak); Spearman(pos_rate, PR-AUC) = +0.117 (≈0); Spearman(volume, PR-AUC) = +0.233 (wrong sign).
Normalising each fold by its own prevalence leaves a 10.64× spread, down from 12.17× — prevalence
absorbs almost none of it.

**Why prevalence is flat, which is the useful insight:** fitting the box per kitchen already
normalises size away. A larger kitchen gets larger voxels, but its GT interaction points spread
proportionally, so `mean_pos_count` lands at 18–26 distinct positive voxels per episode for every
participant. The property that looked like a bug — 16 cells per axis regardless of physical size —
is exactly what makes prevalence comparable across kitchens.

`fill_fraction` (0.504 → 0.772, mean 0.641) does not behave as a confound either:
Spearman(fill, volume) = +0.083, and Spearman(fill, PR-AUC) = −0.433, i.e. the wrong direction.

Consequence for this spec: cross-participant PR-AUC comparison is **not** compromised by grid
geometry. The real threat to the measurement is §10.

**(iii) Voxels are anisotropic.** A kitchen is ~2–3 m tall against 4–8 m wide, with 16 cells on every
axis, so voxels are squat — perhaps 40 cm wide and 18 cm tall. Normalising all three axes to (0,1)
therefore means a different number of metres per unit on each axis. Relevant to the single
`fourier_scale` in §4.2, which is applied to all three.

**(iv) The horizontal rotation is hand-tuned per participant.** The nine yaw angles were chosen
visually, by fitting each grid to its kitchen's footprint. Two consequences outside the scope of
this task but worth recording:

- *Reproducibility — CONFIRMED DEFECT, fix this.* The nine angles exist **nowhere in code**. They
  live only in the `.blend` transforms and, derivatively, in the `.npy` the loader reads. If that
  `.npy` is lost or regenerated differently, no result in this project can be reproduced. Extract the
  nine angles, commit them as a table with provenance, and have the grid construction read them from
  that table. (For the record as measured: P06 22.12°, P05 23.00°, P08 23.62° cluster within 1.5°
  mod 90, and P04 65.67° / P03 71.64° within 6° — probably coincidence at n=9 with arbitrary Aria
  frames, but worth a glance when the angles are formalised.)
- *Protocol caveat.* Under LOPO, the held-out participant's grid was fitted by a human who looked at
  that participant's data. This is not label leakage (no interaction annotations are involved), but
  it is a manual step applied to held-out data and it means the pipeline is not automatic for an
  unseen kitchen. Either frame it as a one-time per-environment calibration step, or replace it with
  an automatic estimator and show the results are unchanged — see §11.

---

## 1. Scope and staging

Implement as **three independently-togglable changes**, each defaulting to current behaviour.
They must be runnable in isolation so their effects don't confound each other.

| Stage | Flag | What it does | Effort |
|---|---|---|---|
| A | `--readout cls` | Dedicated learned `[CLS]` token instead of seat 0 | ~10 lines |
| B | `--voxel_bias_init prior` | Initialise output bias to the class prior logit | ~3 lines |
| C | `--decoder voxel_query` | The cross-attention decoder | the bulk |

Stage C additionally introduces `--voxel_bias {factorised,full}` (default `factorised`) — see §4.3.

**Stage B is a confound control, not an improvement.** The 400:1 imbalance is currently unaddressed
(`BCEWithLogitsLoss()` with no `pos_weight`), and initialising the output bias to
`log(p/(1-p)) ≈ -5.99` is a well-known fix that helps on its own. The new decoder introduces a
`voxel_bias` parameter which will get this init. If you don't also run B on the *current* decoder,
you cannot tell whether a gain came from the architecture or from the bias init. Run B alone first.

Defaults: `--readout token0`, `--voxel_bias_init zero`, `--decoder linear`. With all three at
default the forward pass must be **bit-identical** to current `main`. This mirrors how
`--time_anchor` was handled: old runs stay comparable, production scripts opt in explicitly.

---

## 2. Stage A — `[CLS]` readout

### Changes in `model.py`

Add to `__init__`:

```python
self.readout = config.readout  # "token0" | "cls"
if self.readout == "cls":
    self.cls_token = nn.Parameter(torch.zeros(1, 1, config.transformer_hidden_size))
    nn.init.normal_(self.cls_token, std=0.02)
```

In `forward`, after the segment concatenation and **before** the transformer encoder call, prepend
the CLS token to the sequence and prepend a `False` column to the padding mask.

### Critical detail: check the tensor layout first

`transformer_output[0]` selecting "token 0" implies the encoder is **sequence-first**
(`[L, B, D]`, i.e. `batch_first=False`). Verify this. If sequence-first:

```python
cls = self.cls_token.expand(-1, B, -1)          # [1, B, D]
seq = torch.cat([cls, seq], dim=0)              # [L+1, B, D]
pad_mask = torch.cat([
    torch.zeros(B, 1, dtype=torch.bool, device=pad_mask.device), pad_mask
], dim=1)                                        # [B, L+1], key_padding_mask stays batch-first
```

If it turns out to be batch-first, adjust — but do not silently assume.

### Audit for index-dependent code

Prepending shifts every token index by one. Grep for any code that indexes into the sequence by
position: the `sep` insertion logic, any per-segment slicing used for diagnostics, anything in
`smoke_obb.py` that asserts on positions. Fix or gate each on `self.readout`.

`smoke_obb.py` must still pass under both `--readout` values.

---

## 3. Stage B — voxel bias initialisation

Compute the positive rate once from the training split (fraction of the 4096 voxels that are 1,
averaged over episodes) and log it at startup. Expect ≈ 1/400 ≈ 0.0025 → logit ≈ −5.99. Do not
hardcode 400; the true rate depends on `future_time` **and on which participants are in the split**,
since voxel size varies per participant (§0b.ii). Log the per-participant positive rate too, not
just the pooled one — it is needed for §10 and costs nothing here.

Apply to `self.output_layer.bias` for `--decoder linear`, and to `voxel_bias` for
`--decoder voxel_query`. Note this is a **scalar** init applied uniformly to all 4096 entries —
not a per-voxel empirical prior, which would leak training-kitchen spatial structure and defeat the
point of the whole change.

---

## 4. Stage C — the voxel-query decoder

### 4.1 New module

Add to `model.py` (or a new `decoder.py` if you prefer to keep `model.py` readable):

```python
class VoxelQueryDecoder(nn.Module):
    """Cross-attention decoder: 4096 coordinate queries attend to observation tokens."""
    def __init__(self, config, location_embeddings=None):
        ...
    def forward(self, memory, memory_key_padding_mask, need_weights=False):
        """
        memory: [L, B, D] (sequence-first, matching the encoder output)
        memory_key_padding_mask: [B, L], True = pad
        returns: logits [B, 4096], attn [B, 4096, L] or None
        """
```

### 4.2 Query construction — three variants, selected by `--voxel_query_type`

The grid is 16×16×16 and is normalised **per participant** (§0b). The query set is therefore
identical for every sample and every participant, which is why the queries can be a
`register_buffer` rather than per-sample input.

**But do not read that as "voxel (5,8,6) means the same place in every kitchen."** Per §0b.i, `z`
carries shared meaning across participants and horizontal *structure* does, but absolute horizontal
*position* does not — the grid frames are wall-fitted, yet nothing fixes which wall is `+x`. The
query grid is a fully shared coordinate on one axis and a shared-up-to-eight-symmetries coordinate
on the other two. This is a property of the data, not something the decoder can fix — but it does
dictate the bias factorisation (§4.3), the expected ablation ordering (§8c), and the axis-marginal
diagnostic (§7b).

Base coordinates:

```python
idx = torch.arange(16)
c = (idx + 0.5) / 16.0                                    # (0,1), cell centres
grid = torch.stack(torch.meshgrid(c, c, c, indexing="ij"), -1)  # [16,16,16,3]
self.register_buffer("voxel_coords", grid.reshape(4096, 3))     # [4096, 3]
```

**Verify the flattening order matches the label layout.** The labels are
`labels['interaction'] [4096]` produced by reshaping a 16³ grid. Whatever order `loader.py` /
`boxer_env.py` use (`_nearest_voxel_ijk` and the voxelisation path), the queries must use the
same. Get this wrong and the model trains against a permuted target — it will still learn
something, which is what makes the bug hard to spot. **Add an explicit assertion test for this**
(see §6, test 9).

**Variant `axis_sum` (recommended default).** Reuses the existing shared tables:

```python
q = (self.location_embedding_x(i + 1)
   + self.location_embedding_y(j + 1)
   + self.location_embedding_z(k + 1)
   + self.query_type_embedding)          # [4096, 768]
```

Consistent with the repo's existing design philosophy (OBB tokens, gaze, and trajectory all share
these tables rather than learning private vocabularies), and adds only one new 768-vector. The
`query_type_embedding` marker is needed for the same reason `gaze_type_embedding` and
`obb_type_embedding` are: without it a query would be indistinguishable from an input location
token at the same voxel, and there is no positional encoding to separate them.

Risk to note: this ties the "where the person is" representation to the "where the prediction is"
representation. That coupling is probably beneficial (both are positions in the same grid) but it
is a real design commitment — if `axis_sum` underperforms `fourier`, this coupling is the first
thing to suspect.

**Variant `fourier`.** Learnable Fourier features, LLaVA-4D Eq. 5:

```python
self.W_r = nn.Parameter(torch.empty(config.transformer_hidden_size // 2, 3))
nn.init.normal_(self.W_r, std=config.voxel_query_fourier_scale)
...
proj = self.voxel_coords @ self.W_r.T                    # [4096, 384]
q = torch.cat([proj.cos(), proj.sin()], -1) / math.sqrt(768)   # [4096, 768]
```

`voxel_query_fourier_scale` default 10.0. Sensitive hyperparameter: too small and all 4096 queries
collapse to near-identical vectors (the model degenerates to predicting one global logit); too
large and adjacent voxels alias, losing the smoothness that is the whole point. Ablate {2, 10, 30}.
Only 1152 new parameters, which is the appeal.

Two caveats specific to this dataset. First, one scale is applied to all three axes, but voxels are
anisotropic (§0b.iii) — a unit of normalised `z` is roughly half the metres of a unit of `x`, so the
effective spatial frequency differs per axis. `W_r` is learnable and can compensate, but the *init*
does not. Second, `fourier` mixes all three axes into every frequency component, so unlike
`axis_sum` it cannot treat the cleanly-transferable axis (`z`) differently from the two that are
shared only up to the rectangle symmetries (§0b.i).
Both point the same way — see §8.

**Variant `learned`.** `nn.Embedding(4096, 768)`. 3.15M params — same count as the layer being
removed. Include it as a control: if `learned` matches `axis_sum`, the gain came from the
cross-attention routing, not from the query being coordinate-structured. Useful to know.

### 4.3 The decoder block

Pre-LN, one cross-attention layer, optional FFN:

```python
h = q.unsqueeze(1).expand(-1, B, -1)                  # [4096, B, D]
attn_out, attn_w = self.attn(
    self.ln_q(h), self.ln_m(memory), self.ln_m(memory),
    key_padding_mask=memory_key_padding_mask,
    need_weights=need_weights,
)
h = h + attn_out
if self.use_ffn:
    h = h + self.ffn(self.ln_f(h))
logits = self.head(self.ln_o(h)).squeeze(-1).transpose(0, 1)   # [B, 4096]
logits = logits + self.voxel_bias                              # [4096] broadcast
```

- `self.attn = nn.MultiheadAttention(768, config.decoder_heads, batch_first=False)` — match the
  encoder's layout, don't transpose back and forth.
- `self.head = nn.Linear(768, 1)` — **one head shared across all 4096 voxels.** This is the
  parameter-reduction that matters.
- **`voxel_bias`, selected by `--voxel_bias {factorised,full}`, default `factorised`.** This
  preserves the model's ability to express an unconditional per-voxel prior — which the current
  3.15M layer can express and which is doing real work under 400:1 imbalance — so removing it
  entirely would confound the comparison. But the two forms differ in what they can memorise:

  - `factorised` (default): three vectors of length 16, `bias_x[i] + bias_y[j] + bias_z[k]`,
    **48 parameters**. Each gets `prior_logit / 3` at init so the sum starts at `prior_logit`.
  - `full`: `nn.Parameter(torch.full((4096,), prior_logit))`, **4096 parameters**.

  Rationale for the default: per §0b.i, absolute horizontal position does not correspond across
  kitchens — the grids are wall-aligned but the choice among the eight rectangle symmetries is
  unfixed, so voxel (5,8,·) is not the same place in P01 and P07. A height prior ("interactions
  happen at counter level") is exactly what a factorised bias expresses well and is the part that
  should transfer. A full `[4096]` bias additionally has capacity to memorise a specific horizontal
  layout, which cannot generalise to the held-out participant for the reason just given.

  Note the factorised form is not *only* a height prior — `bias_x` and `bias_y` remain free, and
  since the grids are wall-aligned they can legitimately learn something like "interactions
  concentrate away from the grid edges." That is transferable in a way a full per-voxel map is not.

  Keep `full` as an ablation arm rather than deleting it. If it beats `factorised` on held-out
  participants that contradicts the reasoning above and is worth chasing, not ignoring.
- FFN off by default (`--decoder_ffn` to enable). See §5 for why.

### 4.4 Padding mask

Reuse the **exact same** `key_padding_mask` passed to the encoder — do not rebuild it. Getting a
different mask here is a silent correctness bug.

Note the OBB empty-window handling already in the repo: 7.2% of train windows have no boxes, and
slot 0 is kept attendable to avoid a NaN softmax. That protection applies here too — every row of
`memory_key_padding_mask` must have at least one `False`. Since video tokens are always present
(and CLS under Stage A), this holds, but assert it in the smoke test rather than trusting it.

---

## 5. Compute and memory

Attention matrix is `[B, heads, 4096, L]`.

| Config | L | B=16, fp32, materialised |
|---|---|---|
| `video,obb` | 543 | ~1.1 GB |
| `video,location,obb,gaze` | ~640 | ~1.3 GB |
| `video,env` | 4127 | ~8.7 GB |

**Set `need_weights=False` for training.** `nn.MultiheadAttention` dispatches to
`scaled_dot_product_attention` (flash/mem-efficient) only when weights aren't requested, and then
the matrix is never materialised — memory drops to roughly the size of the output. With
`need_weights=True` it always materialises. Only the diagnostic script should request weights, and
only on a small batch.

**Recommendation: develop and evaluate this on the `obb` branch, not `env`.** 543 tokens vs 4127.
The `env` branch is viable with flash attention but the diagnostic probe will not fit, and the
probe is half the value of doing this at all.

Parameter count, `axis_sum` + no FFN:

```
removed:  output_layer            3,149,824
added:    query_type_embedding          768
          attn in_proj            1,771,776
          attn out_proj             590,592
          3 LayerNorms                4,608
          head                          769
          voxel_bias (factorised)          48
          -----------------------------------
          total added             2,368,561    (net -781,263)
```

With FFN (768→3072→768) it adds 4.72M and the model gets *larger* than before, which undercuts the
"reduce the memorisation sink" argument. Hence off by default. If you enable it, report the
parameter count alongside the metric so the comparison stays honest.

---

## 6. Tests — new `smoke_decoder.py`, modelled on `smoke_obb.py`

1. **Regression.** `--decoder linear --readout token0 --voxel_bias_init zero` produces logits
   bit-identical to current `main` on a fixed seed and fixed batch. Non-negotiable.
2. **Shape.** Output is `[B, 4096]` for all three decoder/query-type combinations.
3. **Key permutation invariance.** Permuting non-pad observation tokens (and the mask with them)
   leaves logits unchanged to within fp tolerance. Cross-attention is a set operation over K/V;
   this confirms no accidental positional leak.
4. **Query permutation equivariance.** Permuting the query rows permutes the output logits
   identically. Proves per-voxel identity comes from the query and nothing else.
5. **Pad tokens are inert.** Append 32 garbage tokens with mask `True`; logits unchanged.
6. **Gradient flow.** Every new parameter receives non-zero grad: query table or `W_r`,
   `query_type_embedding`, both attention projections, `head`, `voxel_bias`. And the inverse:
   when `--decoder linear`, none of the new parameters receive grad, so `linear` runs on this
   branch still match the pre-change baseline.
7. **No NaN with a single attendable token.** Mask everything but one key; output must be finite.
8. **Queries are distinct.** `torch.cdist(q, q)` off-diagonal minimum > 1e-3. Catches a collapsed
   `fourier_scale`.
9. **Query/label index agreement.** Construct a synthetic sample whose only positive label is at a
   known `(i,j,k)`; assert `voxel_coords[flat_index(i,j,k)]` equals `((i+.5)/16, (j+.5)/16,
   (k+.5)/16)` using the same flattening the loader uses. This is the highest-value test in the
   list — a permuted target trains without erroring.
10. **CLS interop.** All of the above pass under both `--readout token0` and `--readout cls`.
11. **Bias factorisation.** Under `--voxel_bias factorised`, changing `bias_z[k]` alone must shift
    the logits of all 256 voxels sharing that `k` by an equal amount and leave every other voxel
    unchanged. Confirms the index arithmetic matches the flattening formula from test 9 — the two
    tests must agree on axis order or one of them is wrong.

---

## 7. Diagnostic probe — new `analysis/voxel_query_attention.py`

The reason to do this change at all. Load a trained `voxel_query` checkpoint, run inference with
`need_weights=True` on a few hundred test episodes, and for each **true-positive** voxel
(prediction above threshold, label 1) record the attention distribution over `L`.

Report:

- **Attention mass by segment**: video / location / gaze / obb / trajectory. Fractions summing to 1.
- **Within the video segment, mass by Δt** (time-to-anchor, 0–30 s). A monotone concentration near
  Δt→0 is evidence of anticipation from recent observation. A flat distribution says the video
  tokens are being used as a bag of scene context.
- **Within the obb segment, whether the top-attended box is spatially co-located with the queried
  voxel.** High co-location = the model is reading "there is a kettle here, kettles get used" —
  a co-occurrence prior. Low co-location with high attention on a *different* region is more
  interesting and harder to explain as co-occurrence.
- Same three, computed on **false positives** and on true negatives, for contrast.

Save the raw `[n_voxels, L]` slices for a handful of episodes so the maps can be plotted per-voxel
later.

### 7b. Axis-marginal spatial prior

Separate, much cheaper, and worth running on every trained model including the baselines. Average
the predicted probability map over the test episodes, then marginalise it over each axis
independently, giving three curves of 16 points each. Also plot the same three marginals of the
ground-truth label map.

Read it against §0b.i:

- **Expected and healthy:** clear structure in `z` (a peak in the counter / interaction band),
  near-flat `x` and `y`.
- **Expected and bad:** sharp structure in `x` or `y`. That is a horizontal layout memorised from
  the 8 training kitchens, and it cannot be right for the held-out one — the yaw is arbitrary.

This is the most direct available test of whether the model is learning something transferable or
fitting training-kitchen geometry, and it needs no attention weights. Run it on `--voxel_bias full`
versus `factorised` in particular; if the reasoning in §4.3 holds, `full` should show visibly more
`x`/`y` structure.

---

## 8. Run plan

All on the `obb` branch, `TIME_ANCHOR=gaze`, same LOOCV protocol as the current production scripts,
same seeds:

| Run | Flags | Question |
|---|---|---|
| R0 | current defaults | baseline |
| R1 | `--voxel_bias_init prior` | how much of any gain is just the bias? |
| R2 | `--readout cls` | is seat 0 a real bottleneck? |
| R3 | `--readout cls --voxel_bias_init prior` | do A and B interact? |
| R4 | `--decoder voxel_query --voxel_query_type axis_sum` (+ cls, prior) | the main result |
| R5 | R4 but `--voxel_query_type fourier` | does continuity help over lookup? |
| R6 | R4 but `--voxel_query_type learned` | was it the queries or the routing? |
| R7 | R4 but `--voxel_bias full` | does per-voxel bias memorise horizontal layout? |

R2 is worth reading carefully even if it does nothing: a null result there says the readout path
isn't the bottleneck and lowers the expected payoff of R4, which is useful before investing in the
`fourier_scale` sweep.

### 8b. Running this under LOPO without 72 training runs

The split is leave-one-participant-out over 9 kitchens. Eight configs × nine folds is 72 runs —
don't. Instead:

1. Run R0–R7 on **two or three fixed folds**, chosen to span the observed positive-rate range (see
   §10 — pick the extremes and one middle).
2. Pick a winner.
3. Run full 9-fold LOPO on **baseline vs winner only**. ~24 + 18 runs rather than 72.

**Compare paired per-fold. Never average PR-AUC across folds.** PR-AUC is prevalence-sensitive and
the positive rate varies by participant, so a 9-fold mean is dominated by whichever kitchens have
the densest labels. Report the distribution of per-fold deltas (same fold, config A vs config B),
and alongside the raw value report lift over that fold's positive rate so the numbers are
commensurable across kitchens.

### 8c. Predicted ordering, stated in advance

Recording this now so the result is a test rather than a post-hoc story. Given §0b.i — `z` transfers
cleanly, horizontal *structure* transfers, horizontal *position* does not:

> `axis_sum` ≥ `fourier` > `learned`, with the gaps widening *because* the split is cross-kitchen.

`axis_sum` keeps the three axes separable, so each can learn at the level that actually transfers:
a strong height prior on `z`, weak marginal structure on `x`/`y`, and no capacity to encode a
specific horizontal layout. `fourier` mixes all three axes into every component, so it cannot treat
the clean axis differently from the ambiguous ones. `learned` has no axis structure at all and 3.1M
parameters pointed straight at memorising training-kitchen layouts. Similarly
`--voxel_bias factorised` ≥ `full`.

Confidence: moderate for `learned` losing, weaker for `axis_sum` vs `fourier`. The manual wall
alignment makes horizontal position *more* meaningful than it would be under arbitrary yaw, so the
gap between the axis-factorised and axis-mixed variants should be smaller than it would have been.

If `learned` or `full` wins, the reasoning above is wrong somewhere and that is the more interesting
result — chase it rather than filing it.

---

## 9. Compatibility notes

- Old checkpoints load unchanged under `--decoder linear`; the new parameters simply don't exist.
  Loading an old checkpoint *into* a `voxel_query` model should fail loudly, not silently
  `strict=False`.
- `config.output_dim` stays 4096 and is still the source of the grid size (`N = round(4096 ** (1/3))`
  — assert it comes out to exactly 16).
- Add the three new flags to `train.py` argparse **and** to the env-var passthrough in
  `submit_train.sh`, following the existing `OBS_TIME` / `TIME_ANCHOR` pattern.
- `submit_train.sh` snapshots source files — if you add `decoder.py`, add it to the snapshot list.

---

## 10. Evaluation protocol — not blocking, but fix it for the §8 runs

The production LOPO scripts (see `experiments/production/`) set
`VAL_PARTICIPANTS=""`, so the test split drives both periodic evaluation and best-model selection.
Every reported per-fold PR-AUC is the maximum over ~40 test evaluations.

**This has been assessed and is NOT the explanation for the per-participant PR-AUC spread. Do not
re-litigate it.** An earlier configuration used 5 checkpoints (one per epoch), where max-over-5
carries far less inflation than max-over-40, and the cross-fold spread was already large. Combined
with §0b.ii ruling out geometry, the spread appears to be a real phenomenon.

What remains is a forward-looking concern about the §8 ablations specifically. Selection noise does
not need to explain the between-*participant* spread in order to contaminate the between-*config*
comparison — it only needs to be large relative to the decoder's effect, which may be 1–2 PR-AUC
points. Two cheap options, either sufficient:

- Set `VAL_PARTICIPANTS` to one of the 8 training participants per fold, by a fixed deterministic
  rule (e.g. next index, wrapping); select the checkpoint on val, report on test.
- Or report at a fixed step chosen a priori, with no selection at all.

One line in the submit script. Decide before launching §8, not after.

---

## 10b. Open leads from the geometry audit (not blocking, but recorded)

- **`n_videos` is the strongest correlate found.** Spearman(n_videos, PR-AUC) = −0.661 (p=0.053),
  Spearman(eps_per_video, PR-AUC) = +0.633 (p=0.067). The two best folds have the fewest videos
  (P07: 9, P08: 12); the two worst have the most (P05: 26, P01: 27). Post-hoc and n=9, so a lead
  rather than a finding — and it is partly entangled with §10, since test-set size varies 2695–5416
  and smaller eval sets inflate a max-over-40 more. Re-examine after 10a.
- **P01 clamps 9.0% of its Boxer/SLAM points to boundary voxels** — 3× the next worst (P05 3.0%),
  15× P09 (0.6%). P01 also has the smallest footprint (2.69 × 4.26 m) and the second-worst PR-AUC.
  Worth checking whether that box is simply too small or mis-rotated.
- **The "3 dropped videos" note in `CLAUDE.md` does not apply to production runs.** With the fitted
  grid a video can never lack a grid; the loader drops zero. Those videos are retained with an empty
  env and padding locations, and 13.5% of P02's / 9.6% of P06's windows have no Boxer boxes at all —
  which `env_pad_mask` reports to the model as "the kitchen is empty" rather than "no data".

---

## 11. Out of scope, but noted

Two upstream changes that this analysis points at. Both live in `loader.py` / `boxer_env.py` and are
explicitly **not** part of this task — recorded here so they are not lost.

- **Fix the discrete symmetry.** The continuous part of the rotation problem is already solved by the
  manual wall alignment (§0b.iv). What remains is the choice among the eight rectangle symmetries,
  and four of those eight are removable by a deterministic rule applied after rotation — no angle
  estimation needed, which makes this a much smaller job than it would have been:
  - *90° swap:* always assign the longer horizontal extent to `x`.
  - *180° sign:* choose the sign of each horizontal axis by a consistent statistic — e.g. put the
    centroid of occupied voxels, or of large fixed appliances, on a fixed side.
  - *Mirror:* genuinely irreducible. Two kitchens can be reflections of each other. Data augmentation
    by random horizontal reflection at train time is the standard answer and is cheap to try.

  If this lands, `--voxel_bias full` and `--voxel_query_type learned` become much more defensible
  and §8c's prediction should be re-tested rather than assumed.

- **Automate the rotation.** Replacing the hand-tuned angles with an estimator (PCA on the SLAM
  trajectory, or modal yaw of Boxer OBB quaternions for large fixed objects, which align to walls)
  would remove the protocol caveat in §0b.iv and make the pipeline automatic for an unseen kitchen.
  The bar is low: it only has to reproduce the manual angles closely enough that results don't move,
  which is directly checkable against the nine existing values.
- **Metric-fixed grid.** A fixed physical extent (e.g. 8 × 8 × 3 m centred on the trajectory
  centroid) instead of a per-participant fitted box would make one voxel the same size everywhere,
  removing the §10 confound at source and letting the model learn priors in metres — such as
  "interactions happen within arm's reach," probably the most useful reachability prior available
  and currently inexpressible.
