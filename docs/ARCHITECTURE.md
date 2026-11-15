# FIction HD-EPIC v2

Future interaction prediction in egocentric video, on the **HD-EPIC** dataset.
A transformer encoder watches a 30-second observation window and predicts
*where in 3D* the actor will interact with objects next, as a 16×16×16 voxel
grid (4096 cells) covering the kitchen.

## What makes this "v2"

The earlier `FIction_HD_EPIC` ("v1") baked every episode (observation window +
ground-truth interactions) into per-config JSON files via `generate_episodes.py`.
Changing `future_time` / `observation_time` / `anticipation_time` meant
regenerating all of them.

**v2 builds episodes on the fly in the data loader**, so those three are plain
runtime hyperparameters — sweep them with no offline preprocessing:

```bash
OBS_TIME=30 FUTURE_TIME=60  ./submit_train.sh future60
OBS_TIME=30 FUTURE_TIME=180 ./submit_train.sh future180
```

v2 also uses **raw, self-contained inputs** (no v1 precomputed artifacts):

| Signal | Source |
|---|---|
| Visual | `extracted_rgb_features_EgoVLPv2/<pid>/<video_id>_features.pt` (EgoVLPv2, 1/sec) |
| Location | SLAM `closed_loop_trajectory.csv` via `SLAM/multi/vrs_to_multi_slam.json` |
| Environment | `boxer_obbs/<video_id>/boxer_3dbbs.csv` (Boxer 3D OBBs) |
| Gaze | `SLAM-and-Gaze/<pid>/GAZE_HAND/mps_<video_id>_vrs/eye_gaze/general_eye_gaze.csv` (see `gaze.py`) |
| GT interactions | `scene-and-object-movements/{assoc_info,mask_info}.json` (object 3D world locations) |

Orientation and pose are intentionally **out of scope** for now.

## Data pipeline (`loader.py` + `boxer_env.py`)

For each **participant** a single **16³ voxel grid** is used — one grid shared by
all of that participant's videos, loaded from
`new_voxel_centers/<pid>_voxel_centers.npy` (a `(4096, 3)` array of voxel centres,
flattened C-order so `flat = i*256 + j*16 + k`). GT, actor location, and the
environment are all mapped into this one grid so they are spatially consistent.

> ⚠️ **The grid is per participant only when `--voxel_centers_dir` is set** (env
> `VOXEL_CENTERS_DIR`), which every production `submit_*.sh` does. With the flag
> unset, `loader.py` falls back to a **per-video** grid built by
> `boxer_env._grid_from_points` as the world-axis-aligned AABB of (that video's
> Boxer OBB corners ∪ its SLAM trajectory). Treat that fallback as legacy: all
> current results use the fixed per-participant grids.
>
> **The production grids are not AABBs and are not computed by this repo.** They
> are HD-EPIC Digital-Twin exports, and each carries a **manual rotation about
> `z`** fitting the box to that kitchen's footprint — so the grid axes run along
> the walls and counters rather than along the arbitrary Aria world axes. No code
> path here can produce a rotated box; `_grid_from_points` uses `np.linspace` per
> world axis. `z` is a pure vertical axis in both frames.
>
> **The nine yaw angles exist nowhere in code.** They live only in the Blender
> object transforms and, derivatively, in the `.npy` files. If a `.npy` is lost or
> regenerated differently, the affected results cannot be reproduced.
> `analysis/grid_geometry_audit.py` recovers them by measuring the lattice, and
> `analysis/out/grid_geometry.csv` records them: P01 3.56°, P02 9.10°, P03
> −18.36°, P04 −24.33°, P05 23.00°, P06 22.12°, P07 35.15°, P08 23.62°, P09
> 0.00°. They were chosen **visually**, not estimated, and applied to the held-out
> participant's own data — a per-environment calibration step, not label leakage,
> but it does mean the pipeline is not automatic for an unseen kitchen.
>
> **Voxel size therefore differs per participant.** The box is fitted per kitchen
> and always divided into 16³ cells, so `voxel_volume_m3` spans 0.00598 (P04) to
> 0.01388 (P07), a 2.32× range. Positive rates differ accordingly (0.00445–0.00634
> measured at `future_time=180`), and PR-AUC is prevalence-sensitive — so
> **per-participant PR-AUC values are not directly comparable, and a mean over
> folds is dominated by the densest kitchens.** Compare paired per-fold instead.
> (Fitting per kitchen does keep prevalence roughly stable — 18–26 positive voxels
> per episode everywhere — which is why the audit rules geometry out as the cause
> of the per-fold PR-AUC spread; see `docs/voxel_query_decoder_spec.md` §0b.)
Boxer / SLAM / GT annotations share the per-recording Aria world frame; a
per-video time offset `T0` (= SLAM trajectory start) aligns Boxer device-clock
time and SLAM time to video-seconds (`frame_number/30`).

> ⚠️ **The trajectory-start `T0` is wrong by ~1 s. `--time_anchor gaze` fixes it;
> `trajectory` (the default) keeps the old behaviour for comparability.**
> The SLAM trajectory does not begin when the recording begins — it begins once
> SLAM has initialised, which is a fixed **1.500 s (P01–P03) / 0.966 s
> (P04–P09)** later. Video second 0 actually corresponds to the **first
> eye-gaze timestamp**, which is essentially the recording start. Measured
> directly as `traj_t0 − first_gaze_t0` over three videos per participant:
> 1.4997 s for P01–P03 (one outlier, `P02-20240209-184316`, 3.899 s) and
> 0.949/0.966/0.983 s for P04–P09 (= ±1 sample at 60 Hz).
>
> Consequence under `trajectory`: when the loader builds the window "video
> seconds 100→130", the device time it reads is `traj_t0+100 → traj_t0+130`,
> which is video seconds **~101→131**. Every device-clock input (`person_locations`
> **and** the Boxer env windows) lands ~1 s **later** in the video than the
> window asked for, while the visual features and the GT labels
> (`frame_number/FPS`) are on the true video clock.
>
> Evidence for the corrected anchor: reconstructing HD-EPIC's annotated look-at
> points (`eye-gaze-priming/priming_info.json`) gives ~17° of error using the
> trajectory start and ~2–3° using the gaze start, with the same correction on
> every video tested.
>
> `--time_anchor gaze` (env `TIME_ANCHOR=gaze`) resolves video second 0 **per
> video** as `first gaze timestamp + gaze_time_offset_s`, the same anchor
> `gaze.py` uses, for `person_locations` and the Boxer windows too — so video,
> location, env, gaze and the labels share one clock. Per video rather than a
> per-participant constant because of that one 3.899 s P02 video. Videos with no
> gaze CSV fall back to the trajectory start (both are dropped anyway for having
> no voxel grid). Side effect: video second 0 now precedes the trajectory's first
> sample, so the first 1–2 seconds of each video become location padding instead
> of a stale clamped position. Each loader prints the anchor and the mean/min/max
> shift at startup (`[<split>] time anchor: ...`).
>
> The default stays `trajectory` so runs recorded before this was fixed remain
> comparable; episode timing itself is unaffected either way (anchors come from
> `mask_info`'s `frame_number/FPS`, already the video clock). In practice, the
> current LOOCV production scripts (see
> `experiments/production/`) all set `TIME_ANCHOR=gaze` explicitly —
> `trajectory` remains the CLI default but is no longer what real runs use.

`HDEpicInteractionDataset.__init__(split, observation_time, anticipation_time,
future_time, participants, static_env=True, ...)` — plus ~15 more keyword
options covering the modalities documented below (`time_anchor`,
`max_obb_tokens`, `gaze_max_hz`, `gaze_time_offset_s`, `env_orphan_fallback`,
`env_class_averaging`, `env_fallback_averaging`, `trajectory_simple`,
`max_trajectory_tokens`, etc. — see `loader.py` for the full list):
1. Loads `assoc_info.json` + `mask_info.json` once (module-level cache).
2. For each video in the split, builds events (object masks with a valid
   `3d_location`) and anchors (each mask timestamp = a "current time").
3. Per anchor: `obs_end = anchor − anticipation`, `obs_start = obs_end −
   observation_time`, prediction window `[anchor, anchor + future_time]`. GT =
   object locations inside the prediction window. Empty-GT episodes are dropped.
4. Computes per-video `T0`, voxel grid, and per-second SLAM positions; precomputes
   the per-window env snapshots (so forked DataLoader workers inherit them).

`__getitem__` returns the keys the model expects: `visual_features [obs,4096]`,
`visual_mask`, `env_grid [4096]` (Boxer `sem_id+1`, 0=empty), `person_locations
[obs,3]` (+mask), `labels['interaction'] [4096]` (binary) — plus, always,
`trajectory_tokens/_mask/_classes`, `obb_voxels/_volumes/_classes/_mask`, and
`gaze_voxels/_mask` for the modalities below (unused modalities' tensors are
just ignored by the model), and conditionally `env_grid_mask`
(multi-class env) / `trajectory_start_voxel`/`trajectory_end_voxel`
(`trajectory_simple`).

### Environment construction (Boxer) — the expensive part, and how it's tamed

Boxer emits ~30 OBB detections/sec across ~858 LVIS classes, so a single video's
CSV is hundreds of thousands of rows and a 30s window holds ~10k. Three measures
keep per-window fusion fast (~0.04s):
- **10 Hz temporal downsampling** at CSV load (`BoxerEnvBuilder.boxer_fps`) — keeps
  every persistent object, trims transient detection noise.
- **Vectorised fusion** (`_fuse_boxes`): detections are snapped to a 0.5 m grid and
  aggregated per `(name, cell)` with pandas (confidence-weighted average position/
  scale, max-confidence quaternion + `sem_id`, ≥2-detection support); no O(n²)
  clustering.
- **Per-window env cache** keyed by `(video_id, obs_start, obs_end)`.

### IoU fusion (`--fusion iou`, `iou_fusion.py`) — what every current run uses

> ⚠️ **The three measures above describe `--fusion grid` only.** It is still the
> CLI default, for reproducibility of everything before 2026-08-26, but **no
> current production run uses it** — they all set `FUSION=iou FUSED_CACHE=true`.
> `BoxerEnvBuilder`'s own class default is already `iou`; the entry point
> (`loader.py`, `train.py`) keeps `grid` deliberately, so the two disagree by design.

`_fuse_boxes` has no notion of overlap: it snaps centres to a `cluster_distance`
cell and groups by `(name, cell)`, so one object whose centre jitters across a cell
boundary becomes two boxes, while two distinct objects sharing a cell are merged.
`iou_fusion.fuse_boxes_iou` replaces the clustering step: link detections that
actually overlap in 3D (IoU ≥ `iou_threshold`, default 0.4) **and** whose class
names are close in OWLv2 embedding space (≥ `semantic_threshold`, default 0.6),
then chain the links with union-find, then confidence-weight each group.

Two things it does that `grid` cannot:

- **Semantic gating** via `class_embeddings.npz` — cupboard/cabinet 0.926,
  mug/cup 0.756, chair/armchair 0.615 — where `_fuse_boxes` required exact `name`
  equality. Exact closed-form 3D IoU is used rather than the reference repo's
  sampling, valid because every HD-EPIC Boxer OBB is yaw-only (`qx = qy = 0`).
- **It fixes a scale bug.** `scale_x/y/z` in the CSV is the box's **full** size,
  but `_fuse_boxes` writes it into `half_extents` unchanged, so **every
  grid-fused box is twice too large per axis**. IoU fusion returns true
  half-extents (`legacy_half=True` reproduces the old behaviour). This shifts OBB
  volume bins by ~3.4 bins, so **grid- and IoU-fused runs are not comparable**,
  and the worked example table below is a grid-fusion artefact.

`--boxer_conf_threshold` defaults per strategy: **0.5 for `grid`** (what every
earlier run used) and **0.6 for `iou`**.

#### The fused cache — mandatory, not an optimisation

IoU fusion costs a measured **6.36 s/window** against ~0.04 s, and a split holds
~28,800 windows: ~51 h inside a single-threaded `__init__`, well past the 6 h wall
clock. So it is precomputed:

- `precompute_fused_obbs.py` writes one pickle per video,
  `boxer_obbs/<video_id>/fused_obbs_iou.pkl`, a sibling of the CSV it was fused
  from (~1.0 GB for the full set). Driver: `precompute_fused_obbs_sbatch.sh`, a
  SLURM array, **one task per participant**.
- Windows are built with the loader's own `build_video_events` / `build_episodes`,
  so the cache cannot drift from what training asks for. Fixed build config
  `obs=30, anticip=5, future=180, time_anchor=gaze`; `future_time` at 180 is a
  superset, so runs at 60/120 hit the same cache. Per video rather than per split
  so all nine LOOCV folds reuse it.
- The pickle records the settings it was built under **including `t0_s`**, checked
  on load. That check is the point: window keys are `int`-truncated video seconds,
  so a cache built under a different `--time_anchor` has the *same keys holding
  different boxes* — silently wrong rather than missing.
- `loader.IOU_UNCACHED_WINDOW_LIMIT = 500` caps how many windows may be fused
  inline before `__init__` refuses, quoting the projected hours. It counts only
  videos that *have* a Boxer CSV — counting the three CSV-less ones made them look
  like 707 windows of pending work and aborted every run.
- `submit_train.sh` must copy `class_embeddings.npz` into the snapshot, because
  `boxer_env.CLASS_EMBEDDINGS_PATH` resolves next to the module; without it
  `FUSION=iou` raises at the first window.

#### `FUSION_VERSION` — and the version-1 caches currently on disk

`FUSION_VERSION = 2` guards changes that alter what the fusion emits for identical
settings — a code fix, which recorded settings alone cannot catch. A cache *newer*
than the code is a hard error; a cache *older* is accepted with a loud warning
naming what differs (`FUSION_CHANGELOG`).

> ⚠️ **The caches on disk are version 1 while the code is version 2, deliberately
> and unrebuilt.** v2 breaks ties between equally-frequent class labels
> alphabetically instead of by set iteration order. Because Python randomises
> string hashing per process, and the cache was built **one SLURM array task per
> participant**, v1 resolved tied labels *differently in each kitchen* — measured,
> 1,060 of the 1,317 tied groups that appear in more than one kitchen
> (cabinet/cupboard, cup/mug, kitchen_sink/sink). Since `sem_id` is embedded
> through a learned table, two spellings of one object are unrelated rows, so this
> puts a vocabulary shift on exactly the train/test boundary a LOOCV fold measures.
> Scope: **class ids only, on 3.86% of OBB tokens** (4.8 of 123.9 per window), and
> those are the weakest boxes (median 2 supporting detections against 5 overall);
> geometry, position and volume are bit-identical. Rebuild with
> `OVERWRITE=true ./precompute_fused_obbs_sbatch.sh` (~1 h/participant) — kept as
> is so far because every completed arm used it.

### Digital-Twin fixture OBBs (`fixture_obbs.py`)

The kitchen's *furniture* is re-detected in every window, with jittering geometry
and an unstable class label, competing for OBB token slots with the movable objects
that actually move. The HD-EPIC Digital-Twin gives that furniture once, exactly,
per kitchen — so `--fixture_obbs` replaces redundant furniture detections with the
modelled fixture, and drops detections the voxel grid cannot represent anyway.

**Source**: `/data/video_datasets/HD-EPIC/Digital-Twin/blenders/meshes/<pid>/*.obj`
— 422 meshes over nine kitchens; class from the filename
(`P01_counter.009.obj` → `counter`).

> **The meshes are already in the per-recording Aria world frame** — the same frame
> as `boxer_3dbbs.csv`'s `t*_world_object` and `new_voxel_centers/<pid>.npy`. **No
> transform is applied.** Verified two ways: extents agree (P01 meshes span
> `[-2.40,-4.35,-1.53]..[0.21,-0.52,0.67]` against grid centres
> `[-2.37,-4.11,-1.46]..[0.39,0.04,0.60]`), and semantically the median fused-Boxer
> `sink` lands **0.04 / 0.12 / 0.04 / 0.04 m** from the nearest `sink` fixture on
> P01/P04/P07/P08, `microwave_oven`→`microwave` 0.03–0.10 m, `stove`→`hob`
> 0.04–0.21 m. `python fixture_obbs.py --verify` re-runs the whole check. If either
> stops holding, this module is wrong and must not be used.

**Mesh → OBB**: yaw-only, matching the fused-box convention exactly — z from the
vertex range, xy from a **minimum-area enclosing rectangle** (rotating calipers over
the convex hull). 363 of 422 meshes are exact 8-vertex boxes and the fit recovers
them to 1.7e-15 m. The rest (sinks, bins, appliances) get a circumscribing box; the
6 cylindrical `bin` meshes are over-covered by exactly `4/π`.

**Voxel-grid membership** (`VoxelGrid`): the production grids are rotated exports,
so the test is done in **fractional lattice coordinates** — inside iff
`−0.5 ≤ u ≤ 15.5` on all three axes. Measured over all nine grids the lattice is
exactly regular and orthogonal (max residual 3.6e-7 m, pairwise axis dots < 4e-9),
so "nearest centre within half the pitch" and "inside the oriented box" coincide.
Fractional rather than a metric tolerance because the pitch is anisotropic and
per-kitchen (0.126–0.417 m on the floor axes, 0.132–0.146 m in z). This is a real
test, unlike `boxer_env._nearest_voxel_ijk`, which is an argmin over the 4096
centres and so snaps an arbitrarily distant box onto a boundary voxel.

**Per-window pipeline** (`build_window_boxes`), in this order:

1. **3a — voxel-grid prune** (`--obb_voxel_grid_prune`): drop fused boxes whose
   centre is outside the participant's grid. Measured **1.71%** of all boxes.
2. **3b — fixture-overlap filter**: drop a surviving box iff **some single fixture**
   satisfies **both** 3D IoU > `--fixture_iou_thresh` **and** class-embedding
   cosine > `--fixture_cos_thresh`. Measured **2.99%** of boxes, and it removes
   exactly what it should: cupboard, microwave_oven, sink, cabinet, stove,
   dishwasher, refrigerator, oven — never the movable objects.
3. **Append the fixtures** (30–61 per kitchen).
4. **Then** the cap, in `_obb_tokens_from_fused` — truncation stays in one place.

Defaults `iou_thresh=0.4` (`BoxerEnvBuilder`'s own linking threshold, i.e. one
definition of "same physical object") and `cos_thresh=0.7` (the BoxerNet reference
value, and it lands in the gap the measured neighbourhoods leave: it passes
cupboard/cabinet 0.926, stove/cooker 0.885, stove/oven 0.769 while blocking
cupboard/storage 0.647, cupboard/shelf 0.644, refrigerator/dishwasher 0.603).

`FIXTURE_DROP_CLASSES = {floor, surface}` are excluded **entirely** — no token, no
filtering, no counting. They are room-scale slabs (one `surface` per kitchen at
14–40 m³, i.e. most of the room); as a single token each would carry the top volume
bin, a centre near the middle of the room and no class — ~2 identical dead tokens
in every window.

`FIXTURE_CLASS_ALIAS` maps the Digital-Twin vocabulary (British English, unspaced
compounds, a `top_X` = "top surface of X" convention) onto LVIS names that carry an
OWLv2 vector; 20 of 31 fixture classes have no literal LVIS entry. **An unmapped
fixture still becomes a token and still prunes, but can never satisfy clause (b),
so it never removes anything** — the fail-safe direction. The one judgment call is
`counter → kitchen_table`: LVIS has no "countertop", and `kitchen_table` is what
Boxer itself assigns to counter surfaces. Measured, it removes 6 boxes in the whole
dataset — the worktop slab (0.008–0.011 m³) cannot reach IoU 0.4 against Boxer's
floor-to-worktop base-unit box (0.34–0.77 m³) — so its real effect is which vector
the counter *token* carries, not filtering.

**LOOCV isolation**: fixtures are keyed by each episode's own `participant_id`
(`loader._obb_fixture_ctx`), read from that participant's own meshes. The held-out
kitchen's fixtures never enter a training sample.

**Wiring**: `boxer_env.build_obb_tokens` takes a `box_transform` hook, applied
**before** truncation and **called even when a window has no fused boxes** — which
is why fixture runs have no all-padding windows. Fixtures carry `prob = 1.0`
against a maximum Boxer `prob` of 0.898, so the confidence sort never drops one.

## Model / training (`model.py`, `trainer.py`, `train.py`)

Reused essentially verbatim from v1. `CustomTransformerModel`: linear-projects
visual features, embeds actor location per axis and the env grid per voxel,
concatenates `[video | sep | loc_x,y,z | sep | env]` (a `sep` is inserted only
*between* present segments, none trailing), runs the Transformer encoder
(8 heads, hidden 768), and decodes to 4096 voxel logits
(see *Readout and decoder* below) with `BCEWithLogitsLoss`.
`num_object_classes=1222` (`object_class_vocab.py`: 0=empty/pad, 1–1220=Boxer
LVIS `sem_id ∈ [0,1219]` shifted by +1, 1221=OOV — a real trajectory track
whose name didn't match the vocab; `env` never emits it). `--input_modality`
selects any subset of `{video,location,env,obb,gaze,trajectory}`.

> **Depth: `--num_transformer_layers` defaults to 6, but nothing runs at 6.**
> `train_sbatch.sh` sets `NUM_LAYERS=2` and every production LOOCV run uses 2.
> This file used to say "a 6-layer Transformer encoder" as if describing the model.

By default there is **no positional encoding over the token sequence**, so every
per-second modality (`location`, `gaze`, `video`) is an unordered *set* to
self-attention and recency is not readable. `--time_encoding fourier` adds one
(see *Time encoding* below); `--env_positional_encoding` addresses the analogous
problem for env tokens spatially, and `--position_encoding fourier` replaces the
index tables that flag uses with a continuous code shared with the decoder's
queries (see *Spatial position encoding*).

> Note `--env_positional_encoding` is **inert unless `env` is in
> `--input_modality`**: it only touches `env_per_voxel`, which is computed and then
> never concatenated when the `obb` branch replaces that segment. Several production
> LOOCV runs echo `ENV_POSITIONAL_ENCODING=true` alongside `MODALITY=video,location,obb`,
> where it does nothing. Same for `TRAJECTORY_SIMPLE` without `trajectory`.

> One correction to the way this used to be stated here. Self-attention without
> positional encoding is permutation-**equivariant**, not invariant, so the output
> is only unchanged if the readout is invariant too — and under `--readout token0`
> it is not: seat 0 is the *first video token*, so reversing the video order swaps
> which second is decoded and moves the logits substantially (measured
> `max|Δ| = 3.06` with no encoding at all, `smoke_time_encoding.py` test 2).
> Reversing `location`/`gaze`, which never occupy seat 0, changes nothing
> (`5.96e-07`), and under `--readout cls` reversing *all* 30 seconds changes
> nothing (`7.75e-07`).
>
> So the model could always distinguish one privileged second from the rest, but
> had no ordering over the other 29 — "recency is not readable" was right, "the
> output is unchanged" was not. It is also an unintended argument for
> `--readout cls`: seat 0 was carrying positional weight nothing designed.

### Time encoding (`--time_encoding`)

`fourier` adds a learned Fourier encoding of **time-to-anchor** to every
per-second token: Δ = (T−1−n)/T ∈ [0,1), so Δ=0 is the second nearest the anchor
and Δ→1 the oldest. Relative rather than absolute so it is invariant to where the
window sits in the recording. 384 parameters (`W_r ∈ ℝ^{384×1}`), default `none`.

Fourier rather than `nn.Embedding(30, 768)` for the reason positional encoding
generally beats a lookup: Δ=3 s and Δ=4 s stay close, so the model can learn a
smooth recency prior instead of 30 unrelated vectors. `--time_encoding_scale`
(default 10.0) is the **std of `W_r`'s init**, not a range for Δ; too small and
all 30 seconds collapse to one vector, too large and adjacent seconds alias.
Ablate {2, 10, 30}.

#### Four modes, not two

`--time_encoding` accepts **`none | fourier | sinusoidal | lookup`**.

| mode | params | what it is |
|---|---|---|
| `none` | 0 | default; the 30 seconds are an unordered set |
| `fourier` | 384 (+1 with a learned gain) | frequencies drawn from `N(0, scale)` |
| `sinusoidal` | **0** (a non-persistent `time_inv_freq` buffer) | Vaswani's fixed **geometric ladder** of wavelengths |
| `lookup` | 23,040 | MMT-style learned table, one row per slot |

**`sinusoidal` is not "fourier with fixed weights" — the difference is the
spectrum, and it is the whole point.** `fourier` draws from `N(0, scale)`, which
has **one** characteristic scale, so `scale` trades adjacent-second sharpness
against window-length range and cannot buy both. A geometric ladder resolves
several timescales at once. Measured at T=30 by
`analysis/time_code_similarity.py`, reporting adjacent-slot cosine and the gap
past which every pair is inside the noise floor:

| code | adjacent cos | informative to gap |
|---|---|---|
| `fourier` scale=10 | +0.946 | **6 s** |
| `fourier` scale=2 | +0.998 | 29 s, but resolves nothing finer than gap 18 |
| `sinusoidal` at 2T | +0.941 | **22 s** |

i.e. the same local sharpness as `scale=10` with ~3.7× the range — a cell no value
of `scale` can reach. `fourier` at scale 10 can say "1, 2, 3, 4, 5 seconds apart"
but 10, 20 and 29 apart all look identical to it.

`--time_encoding_max_wavelength` is the **`sinusoidal`-only** knob (it is inert
under `fourier`, which reads `--time_encoding_scale` instead — easy to misread in
an echoed config). `0.0` resolves to `2 × observation_time`, deliberately *not*
the literature's 10000: over 30 slots most of that ladder is constant — 147 of 384
dimensions never move — the cosine never falls below 0.567 and the dynamic range
collapses to 0.406 against 0.898 at 2T. Smaller wavelengths spend the ladder on
finer scales: at 15 the code is sharper (cos<0.5 by gap 3), has the widest dynamic
range (1.099) and stays informative across the whole window.

**`--time_encoding_gain {fixed, learned}` decides whether any of this is audible.**
`cos²+sin²=1` fixes the code's norm whatever `W_r` holds, so the basis sets
frequencies and can never set amplitude. `fixed` (legacy) divides by `sqrt(D)` and
pins the code at norm **0.707** — about **2%** of a LayerNorm'd token, which is why
the earlier `te10` sweep measured as a no-op. `learned` multiplies a unit-norm code
by a trainable scalar (`--time_encoding_gain_init`, default 8.0 ≈ 25–29% of a
token) which `trainer.FAST_PARAMS` puts in its own `lr=1e-3` group — so 8.0 is an
init, not a setting. `lookup` refuses `learned`, deliberately: its magnitude *is*
its weights (`nn.Embedding` inits `N(0,1)`, so a row sits at `sqrt(H)=27.7`), and a
separate scalar would be a redundant reparameterisation.

Applied to `video`, `location` and `gaze` — the three per-second modalities, with
the *same* Δ on all three axis sequences of a given second. **Not** applied to
`obb` (a fused box is an aggregate over the window and has no single timestamp),
`env` (one token per voxel, not per second), or `trajectory` (already carries
sinusoidal endpoint times, see `trajectory_tokens.py`).

Δ is read off the **slot index**, which is correct only because the loader
end-aligns these sequences to `obs_end`: `person_locations`/`gaze_voxels` are
filled from slot T−1 downwards, and `visual_features` is padded at the front with
the available features packed contiguously — equivalent to end-alignment because
the EgoVLPv2 features are dense at 1 Hz (measured: 0 of 3,516 consecutive frame
indices on P08's videos differ from +1 s). `forward` asserts the window length
matches `observation_time` rather than silently relabelling every second.

`smoke_time_encoding.py` covers it: off-is-off, the invariance table above,
OBB order-invariance preserved, Δ orientation, code distinctness across scales,
gradient flow, and the length assertion. Design doc:
`docs/temporal_spatial_encoding_spec.md` (which also specifies the unbuilt
change B, continuous-coordinate position encoding).

### Readout and decoder (`--readout`, `--voxel_bias_init`, `--decoder`)

Three independently-togglable changes, each defaulting to the pre-existing
behaviour, so every earlier run stays reproducible on this branch. Design doc:
`docs/voxel_query_decoder_spec.md`. Defaults: **`--readout token0
--voxel_bias_init zero --decoder linear`**, and with all three at default the
forward pass is *bit-identical* to the pre-change model — `smoke_decoder.py`
test 1 asserts exactly that against logits captured before any of this landed.

**The pre-change readout, stated precisely**, since the doc used to under-specify
it: `transformer_output[0]` — sequence seat 0, which is always the **first visual
token** (second 0 of the observation window, not a `[CLS]`) — through a single
`nn.Linear(768 → 4096)`. No intermediate projection, no pooling, no non-linearity:
3,149,824 parameters, one independent 768-vector per voxel.

| flag | values (default first) | what it does |
|---|---|---|
| `--readout` | `token0` \| `cls` | `cls` prepends a dedicated learned token so the global summary has its own seat instead of doubling as a content token. Adds 768 parameters. |
| `--voxel_bias_init` | `zero` \| `prior` \| `empirical` \| `empirical_factorised` | `prior` sets the output bias to `log(p/(1−p))` for the **measured** train-split positive rate — the standard fix for the ~1:200 imbalance, which `BCEWithLogitsLoss()` (no `pos_weight`) otherwise leaves unaddressed. A scalar applied uniformly. The rate is measured and logged either way, pooled **and per participant**, by `train.positive_rate_report`. See the ⚠️ below on the two `empirical` arms. |
| `--decoder` | `linear` \| `voxel_query` | `voxel_query` replaces the readout entirely (see below). |

> ⚠️ **This document used to say the bias is "never a per-voxel empirical prior".
> That is no longer true, and the exception is deliberate.** `empirical` fills the
> bias with the train split's measured per-voxel log-odds, and
> `empirical_factorised` with its factorised approximation. A per-voxel empirical
> field **is** the training kitchens' spatial layout, which under participant-level
> splits is precisely the thing that must not transfer — so these are not defaults
> and should never become one. They exist to *measure* what handing the model that
> layout does to input-driven learning. Read any run that sets one as an upper
> bound contaminated by layout, not as a result (`model.py`'s `_init_voxel_bias`).
>
> Two companion flags, both undocumented until now:
> `--voxel_bias_scale` interpolates `bias = pooled + scale*(field − pooled)`, so
> `0` is a plain scalar prior, `1` is the field as measured and `-1` is the
> anti-prior; `--voxel_bias_lr` gives the bias its own optimiser group (`-1` keeps
> it in the normal group, `0` freezes it explicitly, `>0` makes it genuinely
> learnable). Worth knowing before interpreting any bias arm: measured drift over
> 27 finished `voxel_query` runs was **at most 0.036**, i.e. the bias is frozen at
> its init in every result so far.

**`--decoder voxel_query`** (`decoder.py`): the voxel coordinate becomes the
**query**, the encoder's output tokens are keys/values, and **one shared
`nn.Linear(768 → 1)`** produces every logit. A voxel's prediction must then come
from evidence retrieved for that location — and the attention map says which
tokens those were, which is a far sharper probe of "anticipation vs co-occurrence
prior" than an ablation delta. Sub-flags:

- `--voxel_query_type axis_sum` (default) builds each query from the **shared**
  `location_embedding_x/y/z` tables plus a `query_type_embedding` marker (same
  rationale as `gaze_type_embedding`/`obb_type_embedding`: with no positional
  encoding, a query would otherwise be identical to an input location token at
  that voxel). `fourier` uses learnable Fourier features (`--voxel_query_fourier_scale`,
  default 10.0 — sensitive; too small collapses all 4096 queries together; accepts
  three comma-separated values for a per-axis init, see *Spatial position encoding*).
  `learned` is a private `4096×768` table: the control arm, same parameter count
  as the layer removed, no coordinate structure.
- `--voxel_bias factorised` (default) is `bias_x[i] + bias_y[j] + bias_z[k]`, **48
  parameters**, against `full`'s flat `[4096]`. Rationale: the grids are
  wall-fitted but nothing fixes which wall is `+x`, so absolute horizontal
  position does not correspond across kitchens while height does; a factorised
  bias expresses the transferable part ("interactions happen at counter level")
  without the capacity to memorise one kitchen's floor plan. `full` is kept as an
  ablation arm.
- `--decoder_ffn` is **off** by default: it adds 4.72M parameters, which would make
  the model *larger* than the readout being replaced. Report the parameter count
  with the metric if you enable it.
- `--decoder_residual raw` (default) is `h = q + attn_out`, the standard pre-LN
  residual. Its usual justification does not hold here: `q` comes from
  `nn.Embedding` tables (N(0,1), three rows summed) while `attn_out` is an
  attention output, so the two sides differ by ~50×. Measured on the trained
  `loocv_P01_vlo_vq_axis_sum_factorised_cls_prior` checkpoint: **|q| = 55.7 vs
  |attn_out| = 1.11**, i.e. the vector reaching `ln_o` points essentially where
  the query points and evidence is a perturbation the head must amplify. It is
  not blocked — 20–39% of logit variance is episode-driven across the folds — so
  this is conditioning, not a wall. `normed` (`ln_q(q) + attn_out`) only halves
  the ratio, since LayerNorm output is still `sqrt(D) ≈ 27.8`; `balanced`
  (`ln_q(q) + ln_a(attn_out)`, +1536 params) puts both sides at `sqrt(D)`.
  `smoke_decoder.py` test 14 asserts the measured ratios, not merely that the
  logits moved.

  > **Five arms, not three, and the first three answer a question that turned out
  > not to be the variable.** Measured on fold P01 `raw` .0463, `normed` .0460 and
  > `balanced` .0495 land on top of each other — turning the query down is
  > measurably *not* what matters. Two further arms remove the query from the
  > output instead, for a different question. On the P08 baseline at its best
  > checkpoint **91% of logit variance is static per voxel** (bit-identical on
  > every episode) and only 9% is episode-driven; of that static part the frozen
  > voxel bias is 0.23%, so essentially all of it arrives through `q`. What is
  > unknown is whether that map lives in the residual or in the retrieved evidence
  > — the kitchen does not move, so a voxel attends to the same boxes every episode
  > and `attn_out` can be near-constant without anything being wired wrong.
  > - `evidence` (`h = ln_a(attn_out)`, +1536) cuts the residual entirely. `q` still
  >   selects what to attend to, but a voxel's logit becomes a function of what it
  >   retrieved rather than of which voxel it is. **If the static map survives this,
  >   it was never in the residual.**
  > - `concat` (`h = [ln_q(q); ln_a(attn_out)]`, head 1536→1, +3840) keeps both but
  >   stops summing them, so the two halves of `head.weight` say how much the model
  >   wanted each — unrecoverable from a sum. Incompatible with `--decoder_ffn`,
  >   which is built at `d_model`; that combination raises at construction.

Parameter count, `axis_sum` + no FFN: −3,149,824 (`output_layer`, which is not
built at all under `voxel_query`) + 2,368,561 = **net −781,263**. A pre-change
checkpoint therefore fails loudly if loaded into a `voxel_query` model, which is
intended — never paper over it with `strict=False`.

Diagnostics: `analysis/voxel_query_attention.py` (attention mass by segment, by
Δt within the video segment, and OBB co-location — needs a trained `voxel_query`
checkpoint) and `analysis/axis_marginals.py` (three 16-point axis marginals of the
mean predicted map vs the label map; runs on any checkpoint's `predictions.pkl`,
baselines included). Read the marginals against the grid geometry above:
structure in `z` is the prior that should transfer, structure in `x`/`y` is a
memorised horizontal layout that cannot.

`smoke_decoder.py` is the regression test for all of this: **15 test groups**
(numbered 1–14, plus the prior-init group printed as `B`), 173 individual
assertions, ~1 min 40 s on CPU, no dataset needed. Tests 1 and 10 run once; the
other 13 run **twice, under both readouts**, since `--readout cls` changes the
sequence the decoder attends over. Group by group: 1 bit-identical regression,
2 shape, 3 key-permutation invariance, 4 query-permutation equivariance, 5 pad
inertness, 6 gradient flow (and that no decoder parameter exists under
`linear`), 7 NaN safety with one attendable key, 8 query distinctness, 9
query/label index agreement, 10 the CLS seat, 11 bias factorisation, 12 the
attention-probe contract, 13 checkpoint round trip, 14 the `--decoder_residual`
magnitude ratios, B the prior init. The two that matter most are 9 and 11:
they independently pin the query grid and the factorised bias to the **label**
flattening order (`flat = i*256 + j*16 + k`) and must agree with each other — a
mismatched query grid trains without error against a permuted target.

### Spatial position encoding (`--position_encoding`)

The index tables are a **48-vector vocabulary for 4096 voxels**: `e_x[i] + e_y[j] +
e_z[k]`. Two costs follow. They are *discrete* — `e_x[4]` and `e_x[5]` are independent
768-vectors with no prior that they are adjacent — and *additive*, so no cell can be
made distinctive without smearing the same signal across its whole row, column and
layer. Both are measurable: `smoke_position_encoding.py` test 5 finds axis_sum's
near/far voxel-distance ratio pinned at **0.570**, which is its structural floor
`sqrt(2/6) = 0.577` (one differing axis is two independent rows, three axes are six).
It has essentially no smoothness available to it.

`--position_encoding fourier` replaces (or under `--position_encoding_mode add`,
supplements) that sum with a learned Fourier code of the voxel's normalised centre.
`W_r ∈ ℝ^{384×3}` mixes all three axes *before* cos/sin, so `cos(w₁x+w₂y+w₃z)` is a
genuinely joint function of position rather than a per-axis sum. Measured ratio: 0.408.

**The point is the sharing, not the basis.** `spatial_W_r` belongs to the model, and
the decoder is handed `_voxel_pos_embedding` *itself* — so a query and an input token
at one voxel carry the identical vector by construction, not because two call sites
agree on a formula. `smoke_position_encoding.py` test 3 pins that, and test 3's
`(ijk−0.5)/16 == decoder.voxel_coords` check pins the two grid conventions to each
other. This is what `--voxel_query_type fourier` did *not* do before: it gave the
queries a private `W_r` while the input tokens kept the tables, so the arm swapped a
shared vocabulary for two unrelated ones at the same time as it swapped the basis.

> ⚠️ **`spatial_gain` is not optional decoration.** cos/sin are bounded, so `W_r` sets
> frequencies and can never set amplitude. The legacy `/sqrt(D)` pinned the query code
> at norm **0.707** against `|attn_out| ≈ 1.11` where `axis_sum` sits at **55.7** —
> under the default `--decoder_residual raw` that is a 0.6:1 query:evidence regime
> against a 50:1 one, so the 2026-08-11 `vq_fourier` runs measured the residual balance
> as much as the encoding. The gain inits at `sqrt(3·768) = 48.0`, the measured norm of
> the three index rows it replaces (verified: query 55.6 vs 55.5, input 48.0 vs 48.1),
> and gets `trainer.FAST_LR` for the same reason `time_gain` does.

| flag | values (default first) | what it does |
|---|---|---|
| `--position_encoding` | `auto` \| `none` \| `fourier` | `auto` follows `--voxel_query_type`, so `fourier` there turns the input side on too. `none` forces the tables and reproduces the 2026-08-11 arm exactly (private `W_r`, no marker), so those checkpoints still load. |
| `--position_encoding_mode` | `replace` \| `add` | `replace`: the position *is* the code, the structural analogue of the axis_sum arm. `add`: tables **+** code, keeping non-smooth structure a Fourier basis cannot express ("the counter runs along the `k=8` slab"). |
| `--position_encoding_gain_init` | `48.0` | Init only — read the trained value out of the checkpoint. |
| `--voxel_query_untied` | `false` \| `true` | `voxel_query_type=fourier` only. `true` gives the decoder its **own** `W_r`, gain and marker instead of the model's, so tied-vs-untied differs in exactly one thing: whose parameters. Creates no parameter when off, so every existing checkpoint still loads. |

> **Why untied exists.** The 2026-08-11 `vq_fourier` arm was untied *and* pinned at
> query norm 0.707 against `axis_sum`'s 55.7, so it changed the basis, the tying and
> the amplitude simultaneously — a 0.6:1 query:evidence regime against a 50:1 one.
> `--voxel_query_untied` reaches the cell that arm could not: Fourier on both sides,
> separate parameters, **matched gain**. Both paths call one `decoder.fourier_code`,
> so the formula cannot drift between them. This is PETR's arrangement (ECCV 2022,
> separate MLPs for query and key position codes) against DAB-DETR's (ICLR 2022,
> query PE in the key's space — free there, since its sinusoids have no parameters).
> `smoke_position_encoding.py` test 12 covers it, including that the legacy private
> path still has no gain and no marker and still sits at 0.707.

**Scope.** Only tokens carrying a full `(i,j,k)`: `obb`, `env`, and `trajectory`
endpoints, plus — depending on their layout flag — `location` and `gaze`.
`location` is excluded **under the default `--location_tokens axis`, and included
under `joint`** (see below). **`gaze` is the same story**: under the default
`--gaze_tokens axis` it is three per-axis scalar streams, so a token holds one
scalar rather than a coordinate and the code cannot apply; under
`--gaze_tokens joint` it becomes one token per second carrying the looked-at
voxel through `_voxel_pos_embedding` (90 tokens → 30), so gaze finally shares
whatever `--position_encoding` selects. Earlier versions of this file said gaze
was excluded outright — that was true only of the `axis` layout.

#### `--location_tokens {axis, joint}`

`axis` (default, legacy) feeds location as three 30-token streams, one per spatial
axis: 90 tokens, each holding a **scalar**. The axes combine only downstream through
attention, and the *only* thing marking `loc_x[n]`, `loc_y[n]`, `loc_z[n]` as the
same instant is the shared time code — which is **2.6%** of a location token under
`--time_encoding_gain fixed` (measured on `loocv_P08_..._te10` checkpoint-475: code
norm 0.707, `location_embedding_x` row norm 27.68). It cannot grow: `cos²+sin²=1`
fixes the code's norm at `sqrt(H/2)` whatever `time_W_r` holds, so `W_r` sets
frequencies and never amplitude.

`joint` feeds **one token per second** carrying the whole position, through the same
`_voxel_pos_embedding` every other positional token uses — so it inherits
`--position_encoding` (index-table sum under `none`, the shared Fourier code under
`fourier`). 90 tokens → 30. Summed rather than projected from a concatenation on
purpose: a private `Linear(2304→768)` would add parameters and break the
one-vocabulary property. Adds a `location_type_embedding` (768 params), required
because a joint location token is otherwise byte-identical to the positional part of
an OBB token, and to a decoder query at that voxel.

**Neither dominates a priori.** `axis` gives attention three independent slots, so a
query for a high voxel can weight `loc_z` and ignore floor position; `joint` cannot
separate the axes when routing. What `joint` buys is that the binding is
*structural* rather than carried by a 2.6% shared component. Covered by
`smoke_position_encoding.py` test 11.

> ⚠️ **`joint` is what actually completes the shared vocabulary, and this is easy to
> miss.** Under `--position_encoding fourier --location_tokens axis` (the arm run on
> 2026-08-15, `..._te10_pe`) the model carries **two** spatial vocabularies at once:
> location reads `location_embedding_x/y/z` while `obb` and the decoder's queries read
> the Fourier code, and the two share no parameters. `joint` routes location through
> `_voxel_pos_embedding` as well, so location, `obb` and the queries all carry the
> identical positional vector, differing only by their type markers.
>
> The flip side, on the `video,location,obb` modality set: the three index tables then
> have **no consumer at all** and receive exactly zero gradient — 39,168 parameters of
> dead weight. Harmless but worth knowing, and config-dependent: adding `gaze` (which
> still reads the tables per-axis) or `--voxel_query_type axis_sum` revives them. Test
> 11 asserts both directions.

**It does not add resolution.** The code encodes the same cell centre the tables did;
`_nearest_voxel_ijk` still snaps the box centre first. Sub-voxel position needs the
loader to stop discarding it (`docs/temporal_spatial_encoding_spec.md` §2.1, B2), and
that is also where the LOPO memorisation risk recorded in that spec lives.

Fourier and axis_sum checkpoints are mutually unloadable in both directions
(`spatial_*` and `decoder.query_type_embedding` differ) — asserted in test 10, and
intended.

### OBB-token modality (`--input_modality ...,obb`)

The per-box alternative to the `env` voxel grid: instead of 4096 tokens (one per
voxel, most empty), **one token per fused 3D OBB**. Same fusion input as `env`
(`_window_fused_boxes`, so the two are directly comparable) — only the
rasterisation changes.

Each token sums three embeddings, each through the table that already owns that
kind of information, so nothing learns a second private vocabulary:

| | source | table |
|---|---|---|
| **where** | grid index of the box centre | `location_embedding_x/y/z` (shared with location/gaze/trajectory) |
| **what** | Boxer `sem_id+1` | `class_embedding` (shared with env/trajectory) |
| **how big** | log-spaced volume bin (1–16) | `volume_embedding` (new) |

plus a modality marker (`obb_type_embedding`, or row 2 of the unified `type_emb`
table — see *Token scale* below), for the same reason gaze has one: with no
positional encoding, an OBB token and an env token at the same voxel with the
same class would otherwise be near-identical.

#### How a box becomes a token, end to end

Traced on a real window (`P07-20240529-131737`, seconds 241–271, 287 fused boxes):

| box | centre (world m) | half-extents | volume | → voxel | volbin | class |
|---|---|---|---|---|---|---|
| toaster | (−0.12, −1.59, −0.44) | (0.30, 0.35, 0.22) | 0.185 m³ | (9, 6, 8) | 9 | 1095 |
| toaster_oven | (−0.15, −1.59, −0.43) | (0.28, 0.33, 0.20) | 0.151 m³ | (9, 6, 8) | 9 | 1096 |
| microwave_oven | (−1.95, −2.71, −0.38) | (0.40, 0.42, 0.28) | 0.378 m³ | (6, 5, 8) | 11 | 687 |
| automatic_washer | (−3.27, −0.63, −1.00) | (0.65, 0.65, 0.88) | 2.993 m³ | (5, 8, 6) | 14 | 1160 |
| paper_towel | (−2.28, −2.27, −0.40) | (0.12, 0.12, 0.26) | 0.032 m³ | (6, 5, 8) | 7 | 757 |

1. **Fuse** — `_fuse_boxes` (unchanged, shared with `env` via `_window_fused_boxes`):
   snap detection centres to a 0.5 m grid, group by `(name, cell)`, require ≥2
   detections, confidence-weight the position/scale, keep the max-confidence `sem_id`.
2. **Centre → voxel** — `_nearest_voxel_ijk` (`boxer_env.py`) picks the nearest of
   the 4096 grid centres; 0–15 per axis, `+1` → **1–16** (0 reserved for padding).
3. **Size → volume bin** — `scale_*` are half-extents, so full volume is
   `8·sx·sy·sz`, then `np.digitize` against 15 log-spaced edges over 0.001–10 m³,
   `+1` → **1–16**. Log-spaced because one scheme has to resolve a 0.03 m³ paper
   towel and a 3 m³ washing machine: bin 7 is 0.027–0.052 m³, bin 14 is 2.68–5.18 m³.
   Linear bins would put ~95% of boxes in the first bucket.
4. **Class** — `sem_id+1`, already the index the env grid uses.
5. **Sum the lookups** (`model.py`, the `obb_mapped` block) — each integer indexes
   a 768-dim table and the four vectors are **added**:
   `loc_x[i] + loc_y[j] + loc_z[k] + class_embedding[c] + volume_embedding[v] + obb_type`.
   Summed, not concatenated, so everything stays in the single 768-dim hidden
   space — which is also what lets those tables be genuinely shared with the
   location/gaze/trajectory modalities instead of each relearning "where" and "what".
6. **Over cap** — sort by `prob`, keep the top `MAX_OBB_TOKENS`; the rest pad.

**What this buys.** toaster and toaster_oven land in the *same* voxel (9,6,8) with
the same volume bin. Under the env grid one of them wins that cell and the other is
**discarded outright** (`_voxelize_fused_boxes`, highest confidence takes the slot);
here both survive as tokens differing only in the class term. Same for
microwave_oven and paper_towel, both in voxel (6,5,8). That is the fragmentation
behind "247 boxes but only ~114 distinct classes per window" — the grid collapsed
it, OBB tokens keep it. The cost is that sub-voxel position is lost, which is the
same resolution the grid had anyway.

`smoke_obb.py` is the regression test for all of this: it checks each of
voxel/volume/class independently moves the logits, every table used receives
gradient, `volume_embedding`/`obb_type_embedding` receive **none** when `obb` is
inactive (so `env` runs on this branch still match `improved_method`), padded
slots never reach the output, token order is irrelevant, and OBB tokens move
with `location_embedding_x` (proving the spatial tables really are shared).

Size is the genuinely new signal — the voxel grid could only express it as "how
many voxels did this fill", and boxes too small to contain a voxel centre didn't
register at all (hence `--env_orphan_fallback`). The centre is snapped to its
voxel rather than kept as raw world xyz **deliberately**: every recording has its
own Aria world frame with an arbitrary origin (measured spans differ by metres
between participants) and the splits are participant-level, so literal
coordinates would not transfer to an unseen kitchen.

**Token counts.** `MAX_OBB_TOKENS = 250` (`boxer_env.py`). The distribution
depends entirely on which fusion produced the boxes, so quote it with the
strategy attached:

| measured over | mean | median | p95 | p99 | max |
|---|---|---|---|---|---|
| `grid` fusion, train P01–P06 *(historic)* | 238.6 | 250 | — | 458 | 585 |
| **`iou` fusion**, all 28,777 cached windows | **124.1** | 125 | 194 | 219 | **276** |
| **`iou` + `--fixture_obbs`** | **162.3** | 163 | 233 | 261 | **317** |

So the old 512 was never reached by any window of any participant once IoU fusion
landed. The current 250 truncates **546 windows (1.90%)** of the fixture-augmented
set, losing 0.30 tokens per window averaged over the whole set. Over-cap windows
keep the highest-confidence boxes, the same rule the voxeliser uses — and fixtures
carry `prob = 1.0` against a maximum Boxer `prob` of 0.898, so **a fixture is never
the box dropped**.

Sequence length for `video,obb` is **281** (30 + sep + 250), and the production
`video,location,obb` with `--location_tokens joint --readout cls` is **313**,
against **4127** for `video,env`.

> ⚠️ `analysis/obb_token_count.py` — which this section used to cite as its source —
> has no `--fusion` or fixture support and cannot reproduce the current numbers.
> The live figures come from the loader's own startup report (see below).

**Empty windows used to matter here.** Under `grid` fusion 7.2% of train / 9.0% of
test windows had no boxes at all (missing or truncated CSV — see the Coverage
warning); under `iou` it is ~3–4% of train. With one token per box that means an
all-padding segment, whose all-True mask makes attention's softmax return NaN. The
model keeps slot 0 attendable on those rows (an all-zero-index token = "pad voxel,
pad size, empty class"), so an `obb`-only ablation is safe. The `env` grid never hit
this because its mask is all-False regardless — which is exactly why it silently
mislabels "no data" as "empty".

**Under `--fixture_obbs` the case is structurally eliminated: 0.0%.** The fixture
hook is called even when a window has zero Boxer boxes, and appends that
participant's 30–61 fixtures unconditionally, so the only surviving route to an
all-padding segment is a video with no voxel grid at all (the 3 dropped videos).
The NaN-safety machinery above is still present but is dead code for fixture runs.

The loader prints all of this at startup, per split, and that is the authoritative
source rather than any number frozen into this file:

```
[train] obb tokens: mean 159.3 median 161 max 250 of 250 slots; 577 windows hit the cap,
        0 (0.0%) have NO boxes at all (all-padding -- fatal as the only modality)
[train] obb pipeline over 26589 windows: fused 121.3/window -> -2.0 outside the voxel grid
        -> -3.7 covered by a fixture -> +44.1 fixtures = 159.6/window before the 250-token cap
```

### Token scale (`--token_layernorm`, `--type_embeddings`, `--class_embedding_*`)

Each segment's magnitude is an accident of which init convention built it, and
`nn.TransformerEncoderLayer` is **post-LN**, so layer 1's attention sees exactly
these raw numbers. Measured on a real checkpoint, on real P07 data:

| segment | norm | | OBB component | norm |
|---|---|---|---|---|
| video (`nn.Linear`, fan_in 4096) | **6.25** | | position | 47.76 |
| location (joint) | **55.14** | | volume | 27.72 |
| obb | **61.73** | | marker | 27.45 |
| | | | **class** (with `--class_embedding_init`) | **0.719** |

**`--token_layernorm`** builds one `nn.LayerNorm` **per segment type** (8 of them:
video, location, env, obb, gaze, trajectory, sep, cls) — not per token — putting
them all at `sqrt(D)` while leaving each segment its own γ/β. `nn.LayerNorm`'s init
draws no random numbers, so the flag leaves every other parameter untouched.

**`--type_embeddings {separate, unified}`.** `separate` (legacy) is four
independently named parameters — `obb_`, `gaze_`, `location_` (joint only) and
`trajectory_type_embedding` (`end_distance` only) — and **video has no marker at
all**. `unified` replaces them with one `[5, 768]` `type_emb` table indexed by
`TYPE_EMB_INDEX = {video, location, obb, gaze, trajectory}`, adding the video slot
that never existed. `env` is deliberately absent: it has never carried a marker and
adding one would change every env run for a collision that does not arise.

Placement differs, and this is the point of the flag:

```python
separate:  self._ln(name, x + marker)    # inside the norm (legacy)
unified:   self._ln(name, x) + marker    # after it
```

`unified` puts the marker *after* so its norm is an absolute magnitude against a
`sqrt(D)` token rather than a share of it — the same discipline the time code
follows. `separate` keeps it inside to hold every pre-existing arm bit-identical,
down to the float32 association order of the sum, which `smoke_decoder` test 1 pins
to exactly 0 and which `(a+b+c) + m` vs `(a+b+c+m)` would otherwise break by ~5e-7.

`--type_embedding_scale` sets the table's `randn` init: **1.0** gives each marker
norm 27.7 (matching the legacy markers, and 71% of the resulting token); 0.5 gives
13.8 (44%, the share the obb marker historically had); 0.02 would give 0.55, i.e.
2% — the same inaudibility described next. Measured, scale 1.0 takes video's
pairwise token cosine from 0.21 to 0.60, scale 0.5 to 0.37.

> `unified` skips the four legacy draws, so it **shifts the RNG stream** and is not
> weight-paired with a `separate` run. It is a new arm, not an ablation.

**`--class_embedding_init`** seeds the shared `class_embedding` from the OWLv2 text
vectors in `class_embeddings.npz`, joined **by name** through
`data/boxer_class_vocab.json` (case-insensitively — the vocab says `cd_player`, the
npz `CD_player`; without that, 11 of 1082 classes silently fall back to a random
row). A partial join raises rather than warns. Of the 1222 rows, **1082 get a text
vector and 140 keep a learned `nn.Embedding` row**: index 0 (empty/pad), the OOV
row 1221, and the 138 `sem_id` slots the taxonomy never assigns — so "empty voxel"
stays something the model can learn rather than a projection of zeros. Under the
default `--class_embedding_trainable false` the semantic table is a **buffer**, so
it travels with `state_dict` and `.to(device)` but no optimiser group can pick it
up; only the learned 512→768 `class_proj` trains.

> ⚠️ **`--class_embedding_init` alone makes the class term inaudible.** The OWLv2
> vectors arrive unit-length and `class_proj`'s fan-in-512 init shrinks them to
> **0.71**, against 47.8 / 27.7 / 27.5 for position / volume / marker in the same
> OBB token — and lr 1e-5 does not move it (0.708 at init, 0.711 after a full
> epoch). Two class vectors then differ by at most 1.44 against a token norm of
> 61.7, so **swapping an OBB's class rotates its token by under 1.3°**. cup and mug
> *are* close (cosine 0.765); that closeness is simply not audible.
>
> **`--class_embedding_gain`** fixes it: `F.normalize(projected) * class_gain`,
> init `sqrt(768) = 27.71`, where the learned rows and the volume/marker terms
> already sit. `normalize` strips amplitude from `class_proj` entirely and hands it
> to the scalar, which is what makes this a real constraint rather than a redundant
> reparameterisation of a bias-free `Linear`. **It does not move classes relative to
> one another — only how loud they all are.** `class_gain` joins
> `trainer.FAST_PARAMS` (lr 1e-3) for the same reason `spatial_gain` and
> `time_gain` do: a lone scalar never travels at 1e-5.

### Gaze modality (`gaze.py`)

One token per second of the observation window. Each gaze sample becomes a 3D
world point (eye yaw/pitch + `depth_m`, through `T_device_CPF` and the SLAM head
pose); the token is the voxel containing the most of that second's points. Fed to
the model split per axis as three 30-token sequences, through the *same*
`location_embedding_x/y/z` tables `person_locations` uses, plus a learned
`gaze_type_embedding` marker (needed because without positional encoding a gaze
token and a location token holding the same index would be identical vectors).

Facts worth carrying:

- **Direction is trustworthy, distance is not.** The eye cameras see which way
  the eyeballs point (accurate to ~2–3°), but `depth_m` is *inferred* from how
  far the two eyes cross inward — over a 6.3 cm baseline, so it degrades fast
  past ~1 m. Against the SLAM semidense point cloud its error is ~0.16–0.26 m
  (about one voxel), biased per-video rather than random. Since the direction is
  fine, the error slides the point **along the line of sight**.

- **P08's `depth_m` is effectively broken** (median **0.27 m**, **98.2%** of
  samples under 0.5 m, vs P04 0.63 m / P07 0.68 m). 0.27 m is chin distance —
  the estimator is pinned near its floor, not merely noisy. So for P08 the gaze
  point sits just in front of the face, and the token largely encodes **head
  position and orientation** rather than a look-at target. Note orientation is
  otherwise out of scope, so gaze is the model's only route to it — and P08 is
  where gaze gained most (25.6 → 27.5 PR-AUC). Whether that gain is "what they
  looked at" or "which way they faced" is **unresolved**; the test is to re-run
  with `depth_m` replaced by a constant.

- **Time anchor:** gaze always uses first-gaze-timestamp + 0.1 s. The rest of the
  pipeline uses it only under `--time_anchor gaze`; the default `trajectory`
  leaves location/env ~1 s off — see the ⚠️ note under *Data pipeline* above.

- Swapping `depth_m` for a ray-cast onto the SLAM semidense cloud would remove
  the dependence on it entirely; it is a drop-in replacement for
  `gaze.gaze_points_world`.

`CustomTrainer` evaluates val + test at the same cadence and saves
`predictions.pkl` / `labels.pkl` / `indexes.pkl` per checkpoint. Best model by
`pr_auc`. Metrics: ROC-AUC, PR-AUC, IoU and Chamfer at several thresholds.

### Splits (participant-level)
Train P01–P06, Val P07, Test P08–P09 (override with
`--train/val/test_participants`). Each participant is a different kitchen, so the
splits test generalisation to unseen environments.

## Running

```bash
# Local smoke test of the loader (one participant):
python loader.py --splits val            # uses the val participants (P07)

# Local training smoke (tiny, no SLURM, no WandB):
python train.py --expt_name smoke --debug_subset 20 --num_epochs 1 \
    --use_wandb false --train_participants P07 --val_participants P07 \
    --test_participants P07

# Decoder / readout smoke tests (pure torch, no dataset, ~1m40s on CPU):
python tests/smoke_decoder.py             # 15 groups, 205 assertions, both readouts
python tests/smoke_obb.py cls            # the OBB checks under either readout
python tests/smoke_time_encoding.py      # --time_encoding
python tests/smoke_position_encoding.py  # --position_encoding, 12 groups, ~40s
python tests/smoke_token_scale.py        # --token_layernorm + the gain param groups

# SLURM run (snapshots source, then sbatch):
FUTURE_TIME=180 ./submit_train.sh ft180

# The voxel-query decoder (spec §8's main arm), on the obb branch:
MODALITY=video,location,obb TIME_ANCHOR=gaze \
  READOUT=cls VOXEL_BIAS_INIT=prior DECODER=voxel_query \
  ./submit_train.sh vq_axis_sum

# IoU fusion + Digital-Twin fixtures -- what current production runs look like:
MODALITY=video,location,obb TIME_ANCHOR=gaze FUSION=iou FUSED_CACHE=true \
  OBB_VOXEL_GRID_PRUNE=true FIXTURE_OBBS=true \
  VOXEL_CENTERS_DIR=$PWD/new_voxel_centers \
  ./submit_train.sh fixt250
```

> ⚠️ **Run the smoke tests without pinning BLAS threads.** `smoke_decoder` test 1
> asserts bit-identity against a stored reference; `OPENBLAS_NUM_THREADS=1` changes
> the reduction order and makes it report `max|delta| ~ 5e-7` on unmodified code.
> Unset, it is exactly `0.000e+00`.

Snapshots land in `/data/video_datasets/HD-EPIC/snapshots/<JOB_NAME>/` — **15
frozen files**, not just the model: `train.py`, `model.py`, `decoder.py`,
`loader.py`, `boxer_env.py`, `iou_fusion.py`, `fixture_obbs.py`,
`class_embeddings.npz`, `trainer.py`, `trajectory.py`, `trajectory_tokens.py`,
`gaze.py`, `slam_utils.py`, `object_class_vocab.py`, `data/`, plus
`train_sbatch.sh → sbatch.sh`. Two are load-bearing: `class_embeddings.npz`
(`FUSION=iou` raises at the first window without it) and `fixture_obbs.py`. Note
`fixture_obbs.py` reads `new_voxel_centers/` and the Digital-Twin meshes by
**absolute** path, so those are *not* snapshotted — the code is frozen, that data
is not. Checkpoints go to `results/<expt_name>/`.

WandB project defaults to `FIction-HD-EPIC-v2`; current LOOCV production scripts
set `OBB_tokens_fixtures` / `OBB_tokens_fixtures_v2`.

Four env vars are read by `submit_train.sh`, only one of which used to be
documented here:

| var | effect |
|---|---|
| `PARTITION` | overrides `#SBATCH --partition` (default `medium_gpuh200`) |
| `TIME_LIMIT` | overrides `#SBATCH --time` (default `06:00:00`); must fit the partition cap — `medium_gpu*` allow 6h10m, `gpua100` 24h10m |
| `EXCLUDE` | `sbatch --exclude`, to keep a run off named nodes |
| `DEPENDENCY` | `sbatch --dependency`; needed when a run consumes something another job is still producing, e.g. gating `FUSION=iou` on the fused-cache array — without it the loader refuses at startup, by design |

`--propagate=NONE` on the `sbatch` line is **not optional**: the cluster runs
`PropagateResourceLimits=ALL`, and login shells carry `RLIMIT_CPU` soft 10 min /
hard 30 min, which killed a whole batch at ~12.5 min wall. Raising `ulimit -t`
cannot fix it — the hard cap is 1800 s.

`train_sbatch.sh` runs **`python -u`**. Block-buffered stdout once left a `.out`
file ~100 minutes stale during a dataset-construction stall, making it impossible
to see where the job hung.

> **`--seed`, `--debug_subset` and `--dataloader_num_workers` are never passed by
> `train_sbatch.sh`.** So every production run to date is **seed 42** with 4
> dataloader workers, and **no result in this repo has a seed-variance estimate**.
> Differences of ~0.02–0.03 PR-AUC cannot currently be told from seed noise.

## Coverage / fallbacks
Of 156 feature videos, 153 have Boxer OBBs, 153 have a SLAM mapping, and 154 have
eye-gaze CSVs (`SLAM-and-Gaze/<pid>/GAZE_HAND/mps_<video_id>_vrs/eye_gaze/
general_eye_gaze.csv`). Videos missing Boxer get an empty env (grid still built
from the SLAM trajectory); missing SLAM gives all-padding locations (grid built
from Boxer). A video with neither is dropped (it has no usable voxel grid) — this
is **3** videos (`P02-20240210-113925`, `P02-20240211-182630`,
`P06-20240510-140459`). Every usable video has gaze — but not because those three
are the gaze-less ones: only **two** of them are (156 − 154 = 2), and
`P06-20240510-140459` does have a gaze CSV. It is dropped for having neither Boxer
nor SLAM. Coverage counts are printed at `__init__`.

**Fused-cache coverage**: 152 of the 153 Boxer videos have a
`fused_obbs_iou.pkl` (28,777 windows). The exception is `P02-20240210-100930`,
which is also one of the four truncated CSVs below — worth knowing, because
`loader` counts a window as "uncached" only if the video *has* a Boxer CSV, so
that video's windows do count against `IOU_UNCACHED_WINDOW_LIMIT`.

**Fixture source**: 422 `.obj` meshes over nine kitchens; after
`FIXTURE_DROP_CLASSES` and the voxel-grid prune, **30–61 survive per participant**
(P01 40, P02 41, P03 43, P04 37, P05 61, P06 43, P07 53, P08 30, P09 56).

> ⚠️ **4 more videos have a Boxer CSV that stops early.** Missing-vs-present is
> not the whole story: `analysis/boxer_coverage_audit.py` compares each CSV's
> time span against the video duration. As of the last rerun, 4 cover <95% of
> theirs: `P03-20240217-131219` (41.6%), `P02-20240210-100930` (81.4%),
> `P02-20240210-220650` (89.4%), `P02-20240209-231412` (94.4%) — a big
> improvement from an earlier state of 17 truncated CSVs (9 under 65%) that
> `boxer/slurm_boxer_hdepic_rerun_array.sh` fixed for all but these 4 (which the
> rerun script now also targets, alongside the 3 SLAM-missing videos above).
> Windows past the end of a truncated CSV get an all-empty env grid, and since
> `env_pad_mask` is all-False (model.py), the model reads that as "I looked, the
> kitchen is empty" rather than "no data". Affects train and test splits alike.
> Re-run `analysis/boxer_coverage_audit.py` after any further rerun to confirm
> these numbers are still current — this section drifts easily.

## Class vocabulary — a gap worth knowing about

`class_embeddings.npz` holds **1220** OWLv2 text vectors, but
`data/boxer_class_vocab.json` — the `sem_id → name` bridge the join goes through —
holds only **1082**. The cause is in the generator: `analysis/build_boxer_class_vocab.py`
**scans the Boxer CSVs**, so a class only gets an entry if some HD-EPIC video
happened to detect it. The 138 gaps are the non-kitchen tail of LVIS (`camel`,
`baboon`, `bagpipe`, `barge`, `army_tank`), and they fall back to learned rows purely
because the bridge is incomplete, not because the embeddings are missing.

**It is currently harmless**: of 3,570,014 fused boxes, **0** index any of the 138.
All 857 classes Boxer actually emits resolve, and the harvested names agree with the
vocab on every shared row.

The canonical 1220-entry source, if it ever needs completing, is
**`<boxer>/owl/lvisplus_classes.csv`** (the Boxer release's LVIS+ class list) — one name per line,
**line index == `sem_id`** (verified against real data: line 34 `handbag`, line 420
`refrigerator`), name set exactly equal to the npz's. Note the ordering is LVIS's
own, which is alphabetical by **WordNet synset**, not by display name — `trash_can`
is `ashcan.n.01` and so sits at 22, between `artichoke` and `avocado`. Only 46 of
1082 entries match a naive alphabetical index, so the missing names cannot be
recovered positionally from the npz alone.

Completing it would also widen `object_class_vocab._Vocab.match`, which does
longest-phrase word-subset matching over *every* vocabulary phrase for trajectory
names — 138 new short phrases (`bead`, `button`, `calf`) could capture names that
previously fell through to OOV. Measure that delta before committing the file.

## Analysis scripts

34 scripts live in `analysis/`; these are the ones that answer recurring questions:

| script | question |
|---|---|
| `voxel_query_attention.py` | where the decoder's attention mass goes, by segment and by Δt |
| `axis_marginals.py` | axis marginals of the predicted vs label map — `z` structure transfers, `x`/`y` is memorised layout |
| `time_code_similarity.py` | how sharp and how far-reaching a time code is; sweeps `--time_encoding_scale` and the sinusoidal wavelength |
| `residual_balance.py` | the `q` vs `attn_out` magnitude ratio, and how much logit variance is static per voxel |
| `static_map_baseline.py` | how much of PR-AUC a per-kitchen occupancy prior alone explains |
| `gaze_sensitivity.py` | whether gaze gains come from the look-at target or from head pose |
| `input_swap_ablation.py` | swaps one modality between episodes to test what the model actually reads |
| `grid_geometry_audit.py` | recovers the nine per-kitchen yaw angles from the lattice |
| `boxer_coverage_audit.py` | which Boxer CSVs stop early |
| `build_boxer_class_vocab.py` | regenerates `data/boxer_class_vocab.json` (see the gap above) |

Most take `--checkpoint <dir>` and refuse to run on a partially initialised model:
they load with `strict=False` only to *report* missing/unexpected keys, and treat
anything non-empty as fatal. That is why every new parameter must be gated behind a
flag that defaults off — otherwise old checkpoints stop being analysable.
