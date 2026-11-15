# Architecture and token layout — modality-importance analysis

Written before any analysis code. Everything below is **measured** against the
real checkpoints (loaded strictly, no `strict=False`), not read off the docs.

Verification commands used are at the bottom.

---

## 1. Execution path from `train_sbatch.sh`

`train_sbatch.sh` contains **no model**. It is a SLURM wrapper that turns ~60
environment variables into CLI flags and calls one entry point:

```
submit_train.sh                 # snapshots the source into a timestamped dir, then sbatch
  └── train_sbatch.sh           # env vars -> CLI flags        (renamed sbatch.sh in the snapshot)
        └── python train.py     # THE entry point
              ├── CustomConfig(...)                        model.py:61
              ├── CustomTransformerModel(config)           model.py:356
              ├── HDEpicInteractionDataset(split, ...)     loader.py:251   (train / test; val empty here)
              ├── positive_rate_report -> voxel_prior_logit
              ├── model.apply_voxel_bias_init(...)         model.py:843
              └── CustomTrainer(...).train()               trainer.py  (HF Trainer subclass)
```

Sub-modules reached from `model.py`:

| file | role |
|---|---|
| `decoder.py` | `VoxelQueryDecoder` — the 4096-query cross-attention readout |
| `loader.py` | `HDEpicInteractionDataset.__getitem__` — builds every input tensor |
| `boxer_env.py` | fused 3-D boxes -> OBB tokens (`build_obb_tokens`, `_nearest_voxel_ijk`) |
| `trajectory_tokens.py` | per-track features, end voxel, distance bin |
| `object_class_vocab.py` | class index convention (0 = empty/pad, 1–1220 = Boxer `sem_id`+1, 1221 = OOV) |
| `trainer.py` | `CustomTrainer.evaluate` — saves `predictions.pkl` / `labels.pkl` / `indexes.pkl` |

**Snapshot code is byte-identical to the working tree.** `diff` over
`model.py decoder.py loader.py boxer_env.py train.py trainer.py
trajectory_tokens.py iou_fusion.py object_class_vocab.py gaze.py slam_utils.py`
against both snapshots returns nothing. Analysis can therefore import from the
working tree and still describe the trained model exactly.

---

## 2. The two runs

Both are **LOOCV folds of one experiment**, differing only in which participant
is held out. Identified unambiguously from each snapshot's own
`out/*_<jobid>.out`, which echoes every environment variable.

| | Run A | Run B |
|---|---|---|
| snapshot | `2026-08-27_23-22-29_loocv_P08_..._iou` | `2026-08-27_23-22-26_loocv_P04_..._iou` |
| SLURM job | 639400 | 639397 |
| held-out (test) participant | **P08** | **P04** |
| train participants | P01,P02,P03,P04,P05,P06,P07,P09 | P01,P02,P03,P05,P06,P07,P08,P09 |
| val participants | *(empty)* | *(empty)* |
| train / test episodes | 30371 / **5007** (12 videos) | 32067 / **3311** (18 videos) |
| train positive rate | 0.005067 (prior logit −5.2800) | 0.005332 (prior logit −5.2287) |
| wandb | `OBB_tokens2` / `ibasqwe0` | `OBB_tokens2` / `8z3ubpfm` |

Shared configuration (identical across both):

```
MODALITY=video,location,obb,trajectory      # NO env, NO gaze
LR=1e-5  NUM_LAYERS=2  NUM_EPOCHS=1  BATCH_SIZE=16  EVAL_STEPS=0.01
OBS_TIME=30  FUTURE_TIME=180  ANTICIP_TIME=5
VOXEL_CENTERS_DIR=.../new_voxel_centers     # fixed per-participant 16^3 grids
TIME_ANCHOR=gaze  FUSION=iou  FUSED_CACHE=true  BOXER_CONF_THRESHOLD=-1.0
MAX_OBB_TOKENS=512  MAX_TRAJECTORY_TOKENS=16
READOUT=cls  DECODER=voxel_query  VOXEL_QUERY_TYPE=fourier  VOXEL_BIAS=factorised
VOXEL_BIAS_INIT=prior  VOXEL_BIAS_SCALE=1.0  VOXEL_BIAS_LR=-1.0
POSITION_ENCODING=auto(->fourier)  MODE=replace  GAIN_INIT=48.0
LOCATION_TOKENS=joint  TRAJECTORY_TOKEN=end_distance  TRAJECTORY_SIMPLE=true
CLASS_EMBEDDING_INIT=class_embeddings.npz  CLASS_EMBEDDING_TRAINABLE=false
TIME_ENCODING=fourier  TIME_ENCODING_GAIN=learned  GAIN_INIT=8.0
TOKEN_LAYERNORM=false  LOSS=bce  ENCODER_DROPOUT=0.1  DECODER_DROPOUT=0.0
OBB_TOKEN_DROPOUT=0.0  DECODER_ITERS=1  DECODER_RESIDUAL=raw  DECODER_FFN=false
```

Inert flags (echoed but no effect on this modality set): `ENV_POSITIONAL_ENCODING`,
`ENV_*`, `GAZE_*` — `env` and `gaze` are not in `MODALITY`.

### 2.1 Checkpoints — four sets of weights

`--val_participants` is empty, so `train.py` runs in **no-val mode**: the *test*
split drives periodic eval **and best-model selection**. The "best" checkpoint of
each run was therefore selected on the same episodes the attribution will run
over. This is why both best and final are analysed.

| set | path (under `<snapshot>/results/<ts>/`) | step | PR-AUC | ROC-AUC | eval loss |
|---|---|---|---|---|---|
| **A-best** | `checkpoint-855` | 855 / 1899 (45%) | **0.3941** | 0.9506 | 0.02357 |
| **A-final** | `checkpoint-1881` | 1881 / 1899 (99.0%) | 0.3759 | 0.9502 | 0.02406 |
| **B-best** | `checkpoint-756` | 756 / 2005 (38%) | **0.1417** | 0.9329 | 0.02257 |
| **B-final** | `checkpoint-1995` | 1995 / 2005 (99.5%) | 0.1337 | 0.9256 | 0.02335 |

> **"Final" = last checkpoint written, not the last optimizer step.** HF Trainer's
> `save_total_limit` kept only the best plus the two most recent, and the runs
> ended at steps 1899 / 2005. Steps 1881 and 1995 are the closest to the end that
> exist on disk (99.0% / 99.5% of training). The literal terminal weights were
> never saved and cannot be recovered.

Both curves rise to a plateau by step ~300–600 and then drift down slightly; the
best-vs-final PR-AUC gap is −4.6% (A) and −5.6% (B) relative, so "best" is close
to a noise pick on the plateau. Run B's absolute PR-AUC is 2.8× lower than Run A's
— they are different kitchens with different voxel volumes, so **PR-AUC is not
comparable between the two runs** (per-participant prevalence differs; see
CLAUDE.md's grid-geometry note). Only within-run and rank-level comparisons are
meaningful.

### 2.2 Reproducibility anchor

Each checkpoint directory holds `predictions.pkl` `[N, 4096]` float32,
`labels.pkl`, and `indexes.pkl`. `indexes.pkl` is exactly `0..N-1` in order — the
eval dataloader does not shuffle — so **`predictions[i]` corresponds to
`dataset[i]`**. Re-running the model over a rebuilt test dataset must reproduce
those logits; this is the end-to-end check that the dataset reconstruction, the
config, and the weights are all right, and it will be the first smoke test.

The `fused_obbs_iou.pkl` caches (dated 2026-08-26) that both runs read are
unchanged since, so the OBB tokens are reproducible. Note both runs logged
`[boxer_env] WARNING: fused cache is version 1, code is version 2 -- USING THE
CACHED BOXES AS THEY ARE`; the analysis must reproduce that, i.e. **must not**
rebuild the cache.

---

## 3. Model architecture

`CustomTransformerModel` (`model.py:356`), 17,946,931 parameters, **all
trainable** (`class_semantic` is a buffer, not a frozen parameter).

```
per-modality embedding  ->  [592, B, 768]
        |
        v
  nn.TransformerEncoder            2 layers x nn.TransformerEncoderLayer
    d_model=768, nhead=8, dim_feedforward=2048 (default), dropout=0.1,
    activation=relu, batch_first=False, post-LN (norm_first=False)
    src_key_padding_mask = [B, 592] bool, True = ignore
        |
        v  transformer_output [592, B, 768]
        |
  VoxelQueryDecoder                 ONE cross-attention layer
    query : 4096 voxel coordinates -> fourier code (TIED to the model's
            _voxel_pos_embedding) + query_type_embedding      [4096, B, 768]
    key/value : ln_m(transformer_output)                       [592, B, 768]
    nn.MultiheadAttention(768, 8 heads, dropout=0.0), same padding mask
    residual 'raw':  h = q + attn_out
    head : nn.Linear(768 -> 1) SHARED by all 4096 voxels
    + factorised bias  bias_x[i] + bias_y[j] + bias_z[k]  (48 params)
        |
        v
  logits [B, 4096]  ->  BCEWithLogitsLoss vs labels['interaction'] [B, 4096] {0,1}
```

**Attention paths that can connect a modality to the prediction** (Method 3 needs
this): there is **no decoder self-attention** and **no causal masking**.

1. **Encoder self-attention**, 2 layers × 8 heads, over all 592 tokens (any token
   → any non-pad token).
2. **Decoder cross-attention**, 1 layer × 8 heads, 4096 voxel queries → the 592
   encoder outputs. This is the only path from the sequence to a logit.

So a modality reaches voxel *v* either directly (cross-attn column mass on its
tokens) or indirectly (encoder self-attention mixing it into another token that
the query then reads). Raw cross-attention alone therefore understates a modality
whose information has been mixed into, e.g., the CLS token — which is the case
attention rollout exists for.

`need_attention=True` already exists in `forward` and returns `attention`
`[B, 4096, 592]` (head-averaged) plus `segments`. It disables the
`scaled_dot_product_attention` fast path and materialises `[B, 8, 4096, 592]`, so
it is batch-size limited but needs **no code change** for the decoder. The
**encoder's** attention is not exposed and will need instrumentation (Method 3).

**The prediction target.** Multi-label binary: 4096 independent voxel logits,
mean-reduced `BCEWithLogitsLoss`, ~20.8 positive voxels per episode (≈1:200).
There is no softmax and no "predicted class".

---

## 4. Token layout — verified empirically

`forward` assembles segments in the fixed order `video → location → env → obb →
gaze → trajectory`, inserting one `sep` **between** present segments (never
leading, never trailing), then prepends the CLS token under `--readout cls`.
`model.py:1514–1564` builds a `segment_layout` list for exactly this purpose; the
table below is `outputs['segments']` printed from **Run A's real weights**:

| index range | segment | n | valid mask | notes |
|---|---|---|---|---|
| `[0, 1)` | **cls** | 1 | never padded | learned `cls_token`; the decoder's global summary seat |
| `[1, 31)` | **video** | 30 | `visual_mask` | 1 token/second, end-aligned |
| `[31, 32)` | *sep* | 1 | never padded | shared `sep_token` parameter |
| `[32, 62)` | **location** | 30 | `person_location_mask` | `location_tokens=joint` → 1 token/second (not 90) |
| `[62, 63)` | *sep* | 1 | never padded | |
| `[63, 575)` | **obb** | 512 | `obb_mask` | ~140 real (P08) / ~96 real (P04); rest padding |
| `[575, 576)` | *sep* | 1 | never padded | |
| `[576, 592)` | **trajectory** | 16 | `trajectory_mask` | cap `max_trajectory_tokens=16` |

**Total sequence length L = 592.** Verified: `logits [2, 4096]`,
`attention [2, 4096, 592]`.

`sep` occupies 3 positions but is **one** parameter repeated; it is not a
modality and will be reported as its own row, never folded into a neighbour.

### 4.1 How each segment's token is built

`_ln(...)` is the identity here (`token_layernorm=false`). `time_enc` is the
learned Fourier time-to-anchor code (`time_W_r` [384,1], `time_gain` learned,
init 8.0), added **only** to per-second modalities.

| segment | composition | time code? |
|---|---|---|
| cls | `cls_token` (768 params) | no |
| video | `visual_mapper(visual_features)` — `Linear(4096→768)` on EgoVLPv2 features | **yes** |
| sep | `sep_token` (768 params) | no |
| location | `_voxel_pos_embedding(person_locations)` + `location_type_embedding` | **yes** |
| obb | `_voxel_pos_embedding(obb_voxels)` + `_class_embed(obb_classes)` + `volume_embedding(obb_volumes)` + `obb_type_embedding` | no |
| trajectory | `_voxel_pos_embedding(trajectory_end_voxel)` + `distance_embedding(trajectory_distance)` + `trajectory_type_embedding` + `_class_embed(trajectory_classes)` | no |

Under `position_encoding=fourier, mode=replace`, `_voxel_pos_embedding(ijk)` is
the learned Fourier code `fourier_code(ijk, spatial_W_r, spatial_gain, 16)` — of
norm exactly `spatial_gain` — with pad slots (`any(ijk == 0)`) replaced by
`spatial_pad_embedding`. The **same function** produces the decoder's queries
(tied), so a query and an input token at one voxel carry the identical positional
vector, differing only by their type markers.

`_class_embed` projects the frozen OWLv2 `class_semantic` buffer `[1222, 512]`
through a trainable `class_proj` `Linear(512→768, bias=False)`, falling back to
learned `class_embedding` rows for index 0, OOV, and the 138 unassigned `sem_id`s.
Under `trajectory_token=end_distance` there is **no** `trajectory_mapper` — the
segment is four table lookups wide, and the MLP is not built at all.

### 4.2 Parameter inventory (feeds attribution interpretation)

| module | params | reaches |
|---|---|---|
| `transformer` (2 encoder layers) | 11,027,968 | everything |
| `visual_mapper` | 3,146,496 | video only |
| `decoder` | 2,368,561 | readout |
| `class_embedding` | 938,496 | obb + trajectory (fallback rows only) |
| `class_proj` | 393,216 | obb + trajectory |
| `distance_embedding` | 13,056 | trajectory |
| `volume_embedding` | 13,056 | obb |
| `spatial_W_r` / `spatial_gain` | 1,152 / 1 | location, obb, trajectory, **and the queries** |
| `time_W_r` / `time_gain` | 384 / 1 | video, location |
| `cls_token`, `sep_token`, `obb_type_`, `location_type_`, `trajectory_type_`, `spatial_pad_` | 768 each | as named |
| `location_embedding_x/y/z` | 13,056 × 3 = **39,168** | **nothing — dead** |
| `gaze_type_embedding` | 768 | **nothing — dead** |

Measured by backward pass: `location_embedding_x/y/z` and `gaze_type_embedding`
receive `grad is None` in this exact configuration. This is the dead-table case
CLAUDE.md predicts for `position_encoding=fourier + location_tokens=joint`
without `gaze`. It matters for interpretation: **location's positional content
does not go through the index tables here, it goes through `spatial_W_r`, which
is shared with obb, trajectory and the decoder queries.** No modality-level
conclusion can be drawn from those tables.

### 4.3 Parameter sharing across modalities — and what it does and does not confound

Three parameter groups serve more than one modality, by design:

| parameter | consumers |
|---|---|
| `spatial_W_r` / `spatial_gain` (via `_voxel_pos_embedding`) | location, obb, trajectory, **and the decoder's 4096 queries** |
| `class_embedding` / `class_proj` (via `_class_embed`) | obb, trajectory |
| `time_W_r` / `time_gain` | video, location |

**What this does not confound.** Method 1 differentiates with respect to
*activations at disjoint sequence positions*, not with respect to parameters.
`dT/dx_40` (a location token) is computed holding `x_100` (an obb token) fixed;
two tokens built from the same table are still two different vectors at two
different positions. Weight sharing introduces no coupling into that derivative.
This is also why parameter-gradient magnitudes are excluded from every method
here: a gradient on `spatial_W_r` genuinely does sum four un-separable
contributions, and reporting it as a modality score would be exactly the error
this section exists to prevent.

**What it does affect, and is worth stating with any result.**

1. *The three positional modalities are not independent measurements.* The shared
   Fourier code has norm exactly `spatial_gain` = **47.82** (trained value, Run A),
   which is the dominant part of every positional token — 47.82 of an obb token's
   total length of 62.43. If that encoding is good, location, obb and trajectory
   all benefit together; if it is poor, all three degrade together. A ranking
   between them describes the trained model, but it cannot support "location's
   encoding works and obb's does not".
2. *There is an architectural asymmetry with video.* The decoder's queries pass
   through the same `_voxel_pos_embedding`, so a query at voxel *v* and an input
   token at voxel *v* carry an identical positional vector by construction, and
   their cross-attention is largely a geometric match. Video carries no positional
   code at all and must be read through content. Any "video vs the rest" result
   is a result about a model where the other three have a structural route to the
   readout that video does not.
3. *Ablation-with-retraining would be confounded.* Drop obb and retrain, and
   `spatial_W_r` trains differently, moving location, trajectory and the queries
   too. Inference-time ablation (Method 2's main analysis) is unaffected, because
   the parameters are frozen.

**What can be measured about it.** `token_layernorm=False` makes `_ln` the
identity, so every token is an **exact** sum of named parts (verified:
`max|token − Σ parts| = 0.0`), and grad×input is bilinear, so

```
<grad, token> = <grad, pos_code> + <grad, class> + <grad, volume> + <grad, obb_type>
```

holds exactly (measured residual 1.1e-05 on values of order 10²). Method 1
therefore reports a **channel decomposition** splitting each modality's signed
contribution into the part it came from, with the `pos_code` and `class` columns
flagged as shared. "How much of obb's contribution arrives through parameters it
shares with location and the queries" is a number, not a caveat.

Channels per segment (`common.CHANNELS`):

| segment | channels |
|---|---|
| cls | `cls_token` |
| video | `visual` (= `visual_mapper(features)`) + `time` |
| sep | `sep_token` |
| location | **`pos_code`** + `type_marker` + `time` |
| obb | **`pos_code`** + **`class`** + `volume` + `type_marker` |
| trajectory | **`pos_code`** + **`class`** + `distance` + `type_marker` |

Two exactness checks run every batch and are fatal on failure: the channels must
reconstruct the captured encoder input, and their per-segment dot products must
sum back to that segment's gxi total. If `model.forward`'s token assembly ever
changes, both fail loudly rather than silently redistributing the channels.

**What remains genuinely un-attributable.** The *training-knob* question — change
a flag, attribute the metric change to it — is not clean, and CLAUDE.md already
records why: `--position_encoding fourier` moves location, obb, trajectory and the
queries at once; `--location_tokens joint` completes the shared vocabulary *and*
kills the index tables in one flag; the 2026-08-11 `vq_fourier` arm changed basis,
tying and amplitude together. No post-hoc attribution can unpick that — it needs a
designed one-variable arm. It does not apply to the A-vs-B comparison here, whose
configs are identical and differ only in the held-out participant.

### 4.4 Padding and masks

One mask, `[B, 592]` bool, `True = ignore`, built by concatenating per-segment
masks in the same order, then gaining a `False` column for CLS. The **same
tensor** is handed unchanged to the encoder (`src_key_padding_mask`) and to the
decoder (`key_padding_mask`) — `forward` passes `attention_mask` straight through
to `self.decoder(...)`.

- `visual_mask`, `person_location_mask`: 1 = valid; sequences are **end-aligned**
  to `obs_end`, so slot 29 is the second nearest the anchor.
- `obb_mask`: 1 = real box. **Special case:** when a window has no boxes at all,
  `model.py:1457` forces slot 0 attendable (`obb_pad_mask[all_pad, 0] = False`) to
  stop the softmax returning NaN. That token is an all-zero-index token ("pad
  voxel, pad size, empty class"). Measured: 1 of 5007 P08 test windows and 7 of
  3311 P04 test windows. The analysis must count those as *valid-but-empty*, not
  as real evidence.
- `trajectory_mask`: 1 = real track.
- cls and sep are never masked.

Because padded positions are masked as **keys** everywhere they are read, they
cannot influence any output — so `d(target)/d(pad token) == 0` exactly. That is a
free correctness check, not an assumption.

---

## 5. Consequences for the four methods

1. **One hook point covers everything.** The concatenated encoder input
   `[592, B, 768]` — after CLS prepend, before layer 1 — is a single tensor. A
   `forward_pre_hook` on `model.transformer` with `retain_grad()` captures it and
   its gradient with **zero changes to `model.py`**. Verified: grad shape
   `[592, 2, 768]`. Method 1 and Method 2 both attach here.
2. **The target needs care.** There is no predicted class. CLAUDE.md records that
   ~91% of logit variance is static per voxel on the P08 baseline, arriving
   through the query residual `h = q + attn_out` and the bias — neither of which
   depends on input tokens. Raw-logit attribution therefore measures the small
   episode-driven part on top of a large constant; a contrast target
   (GT-positive logits minus the episode's own mean logit) isolates it.
3. **Token counts are wildly unequal** (1 / 30 / 30 / 512 / 16 + 3 sep), and the
   obb segment is mostly padding. Totals and per-valid-token means will rank
   modalities differently, by construction. Both get reported.
4. **The two runs are different kitchens.** Cross-run comparison must be on
   *rankings and normalised shares*, never absolute scores.
5. **Best was selected on the test split** (no-val mode). Analysing best and
   final separately is the guard against reading a selection artefact as a
   modality conclusion.
6. **Three modalities share their positional parameters with each other and with
   the readout** (§4.3). Token-level attribution is unaffected — positions are
   disjoint — but the three positional modalities are correlated by construction,
   and video is structurally disadvantaged in reaching the queries. Every method
   reports the channel decomposition so the shared part is visible rather than
   assumed away, and no method reports parameter gradients.
7. **The precision the runs were evaluated at is not the precision analysed.**
   `training_args.bin` has `bf16=True`, so `predictions.pkl` holds
   bfloat16-precision logits; this analysis recomputes in float32 (the dtype the
   weights are stored in). Measured on 32 P08 episodes: mean |Δ| 0.0287 against a
   mean |logit| of 8.353 (**0.34%**, the scale bf16's 8 mantissa bits predict),
   correlation 0.999882, and **per-episode Spearman 0.999837** over the 4096
   voxels — the voxel ordering, which is what PR-AUC reads, is unchanged.

---

## 6. Verification commands

```bash
source "$HOME/miniconda3/etc/profile.d/conda.sh" && conda activate egovlpv2-pip
cd /path/to/fiction-hd-epic-v2

# token layout + strict weight load (data-free, CPU, ~20 s)
python analysis/modality_importance/verify_layout.py     # written in Method 1

# run identification
grep -E '^(LR=|MODALITY|TRAIN=|NUM_EPOCHS)' \
  /data/video_datasets/HD-EPIC/snapshots/2026-08-27_23-22-2[69]_loocv_P0[48]_*iou/out/*.out

# snapshot vs working tree
for f in model.py decoder.py loader.py boxer_env.py train.py trainer.py; do
  diff -q "$f" "/data/video_datasets/HD-EPIC/snapshots/2026-08-27_23-22-29_loocv_P08_vlot_vq_fourier_factorised_cls_prior_pe_locjoint_trajend_clsemb_tlnoff_e1_iou/$f"
done
```

Environment: Python 3.9.25, conda env `egovlpv2-pip`, `transformers` 4.40.2,
torch as installed in that env. Login node has **no GPU** (`lnode02`); GPU work
goes through SLURM (`medium_gpuh200` / `medium_gpua100`).
