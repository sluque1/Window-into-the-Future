# A Window into the Future

**Predicting *where* in 3D a person will interact with objects next, from
egocentric video.**

<sub>Thesis code, 2026. Internally this project is `FIction HD-EPIC v2`; that
name appears throughout the code, the specs and the run tags.</sub>

The model watches a 30-second observation window, skips a 5-second anticipation
gap, and predicts which cells of a 16×16×16 voxel grid (4096 cells, covering one
kitchen) will contain an object interaction over the following 180 seconds.

It is a transformer encoder over heterogeneous per-second tokens — visual
features, the actor's SLAM location, 3D object boxes, eye gaze — read out by a
**voxel-query decoder**: 4096 coordinate-conditioned queries, one per cell, each
cross-attending the whole input sequence for itself. The output is 4096
independent logits trained with per-voxel BCE, so the task is multi-label, not a
distribution over cells.

This adapts [FIction](https://github.com/thechargedneutron/FIction) (CVPR 2025)
to the [HD-EPIC](https://hd-epic.github.io/) dataset. "v2" refers to building
episodes on the fly in the data loader, so the window geometry
(`observation_time`, `anticipation_time`, `future_time`) is a set of runtime
hyperparameters rather than something baked into preprocessed files.

For the architecture in depth — token layout, environment construction, the
encodings, the fixture OBBs — see [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

---

## Contents

```
train.py model.py decoder.py loader.py trainer.py    core: entry point, encoder,
boxer_env.py iou_fusion.py fixture_obbs.py           voxel-query decoder, dataset,
gaze.py trajectory.py trajectory_tokens.py           modality builders
slam_utils.py object_class_vocab.py
class_embeddings.npz                                 OWLv2 embeddings (required)

submit_train.sh  train_sbatch.sh                     how training is launched

data/boxer_class_vocab.json                          Boxer sem_id -> LVIS name
new_voxel_centers/P0{1..9}_voxel_centers.npy          the nine 16³ voxel grids

preprocessing/      EgoVLPv2 feature extraction + IoU-fused OBB cache
experiments/        the configurations behind the reported results
tests/              seven smoke suites
analysis/           45 scripts, one question each
docs/               architecture and encoding specifications
```

Modules sit flat at the repository root on purpose: `submit_train.sh` snapshots
them by bare filename (see **Training** below).

---

## Install

```bash
conda create -n fiction-hd-epic python=3.9
conda activate fiction-hd-epic

# CUDA 11.8 wheels are not on plain PyPI -- install these first
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu118

pip install -r requirements.txt
```

The pins in `requirements.txt` record the environment the reported results were
produced in. The one that genuinely matters is `transformers==4.40.2`:
`trainer.py` subclasses `Trainer` and relies on its 4.40-era `TrainingArguments`
surface.

Training needs one GPU. The data-free tests run on CPU in seconds.

---

## Data you must obtain

HD-EPIC is not redistributed here. Get it from <https://hd-epic.github.io/> and
point `HDEPIC_ROOT` at it:

```bash
export HDEPIC_ROOT=/path/to/HD-EPIC
```

Five inputs are read, at these subpaths:

| Signal | Path under `$HDEPIC_ROOT` | Used by |
|---|---|---|
| Videos | `Videos/<pid>/*.mp4` | preprocessing only (feature extraction) |
| Actor location | `SLAM-and-Gaze/<pid>/.../closed_loop_trajectory.csv` | `location` modality |
| Eye gaze | `SLAM-and-Gaze/<pid>/GAZE_HAND/mps_<video_id>_vrs/eye_gaze/general_eye_gaze.csv` | `gaze` modality |
| 3D object boxes | `boxer_obbs/<video_id>/boxer_3dbbs.csv` | `env` and `obb` modalities |
| Kitchen fixtures | `Digital-Twin/blenders/meshes/<pid>/*.obj` | `--fixture_obbs true` |
| Ground truth | `hd-epic-annotations/scene-and-object-movements/{assoc_info,mask_info}.json` | the labels |

`HDEPIC_ROOT` moves all of them together. Any single root can also be set on its
own, which is most often useful for the features you generate yourself:

```
HDEPIC_ROOT     FEATURES_BASE   SLAM_BASE      VIDEOS_BASE
ASSOC_PATH      MASK_PATH       BOXER_OBBS_ROOT
DETIC_OBB_ROOT  MESH_ROOT       SNAPSHOTS_ROOT
```

Each defaults to the layout this was developed against, so leaving them unset
reproduces the original runs exactly.

### Already included

- `new_voxel_centers/P0{1..9}_voxel_centers.npy` — one 16³ grid per participant, as `(4096, 3)` voxel centres, C-ordered so `flat = i·256 + j·16 + k`. These are rotated HD-EPIC Digital-Twin exports: the box is fitted to each kitchen's footprint so the grid axes run along the walls rather than the arbitrary Aria world axes. **Nothing in this repository can regenerate them** — the rotation exists only inside the `.npy`.
- `class_embeddings.npz` — OWLv2 text embeddings of the 1220-name LVIS+ vocabulary, L2-normalised. `boxer_env.py` resolves it next to the module, so `--fusion iou` fails at the first window without it.
- `data/boxer_class_vocab.json` — the `sem_id → canonical LVIS name` lookup. Regenerable with `analysis/build_boxer_class_vocab.py`.

---

## Preprocessing

**You must run step 1.** Full detail in
[`preprocessing/README.md`](preprocessing/README.md).

### Step 1 — EgoVLPv2 visual features (required for every run)

Every run reads one 4096-d feature per second of video, and nothing else
produces them.

```bash
git clone https://github.com/facebookresearch/EgoVLPv2.git
export EGOVLPV2_REPO=/path/to/EgoVLPv2/EgoVLPv2     # must contain the inner package
export EGOVLPV2_WEIGHTS=/path/to/EgoVLPv2.pth        # from that repo's downloads

PARTICIPANTS="P01" sbatch --account=<acct> preprocessing/run_egovlpv2.sh
# or: python preprocessing/extract_features.py --participant P01
```

Writes `$FEATURES_BASE/<split>/<pid>/<video_id>_features.pt`, each holding
`{'features': [n_seconds, 4096], 'frame_indices': [...]}`. For second *t*, 16
frames are sampled evenly across that second and pushed through
`FrozenInTime.compute_video`. `<split>` is `train` for P01–P06, `val` for P07,
`test` for P08–P09 — matching `loader.PARTICIPANT_TO_SPLIT`, which is how
training finds them. It is restartable: existing outputs are skipped.

### Step 2 — IoU-fused OBB cache (required for `FUSION=iou`)

`FUSION=iou` links Boxer detections that overlap in 3D *and* whose classes are
close in OWLv2 space. At ~6.4 s per window against ~0.04 s for `grid`, and
~28,800 windows per split, the loader **refuses** to fuse inline and raises
instead. Build it once:

```bash
SLURM_ACCOUNT=<acct> ./preprocessing/precompute_fused_obbs_sbatch.sh
```

~1.0 GB, written beside each Boxer CSV. The production configuration needs this;
`FUSION=grid` (the default) does not.

---

## Training

Training goes through two scripts:

```
./submit_train.sh <tag>
      |
      |  copies the 14 modules + class_embeddings.npz + data/ into
      |  $SNAPSHOT_BASE/<timestamp>_<tag>/  and submits from there,
      |  so later edits cannot affect a running job
      v
train_sbatch.sh            reads 86 environment variables, each with a default
      |
      v
python -u train.py --expt_name ... (87 flags)
```

Every hyperparameter is passed **through the environment**. That is the whole
interface:

```bash
# a quick end-to-end check: 200 train episodes, one epoch, no W&B
TRAIN_N=200 VAL_N=50 TEST_N=50 NUM_EPOCHS=1 USE_WANDB=false ./submit_train.sh smoke

# sweep the prediction horizon -- no preprocessing needed, this is v2's point
FUTURE_TIME=60  ./submit_train.sh future60
FUTURE_TIME=180 ./submit_train.sh future180

# hold out one participant
TRAIN_PARTICIPANTS=P01,P02,P03,P04,P05,P06,P08,P09 TEST_PARTICIPANTS=P07 \
  VAL_PARTICIPANTS="" ./submit_train.sh loocv_P07
```

Cluster-level overrides, also environment variables: `SLURM_ACCOUNT`,
`PARTITION`, `TIME_LIMIT`, `EXCLUDE`, `DEPENDENCY`, `SNAPSHOT_BASE`,
`CONDA_BASE`, `CONDA_ENV`.

> **A tuning note.** `train_sbatch.sh` requests a GPU and 96 GB but no
> `--cpus-per-task`, so SLURM allocates one CPU, while `train.py` defaults to
> `--dataloader_num_workers 4`. PyTorch warns about this on every run. It is
> harmless — the first optimiser step absorbs the worker startup and later steps
> are unaffected — but if you want it clean, either add
> `#SBATCH --cpus-per-task=4` to `train_sbatch.sh` or pass
> `--dataloader_num_workers 1`. The flag is not reachable through the
> environment, so changing it means editing the `python -u train.py` call.
>
> Episode building, not training, dominates a short run: each split takes
> several minutes before the first step, since the loader constructs every
> episode from the raw annotations on the fly.

> SLURM does not expand environment variables inside `#SBATCH` lines, so
> `--account` cannot be templated there. `submit_train.sh` forwards
> `$SLURM_ACCOUNT` on the `sbatch` command line instead; alternatively uncomment
> the `##SBATCH --account=` line in `train_sbatch.sh`.

### Without SLURM

`train.py` is an ordinary script — the SLURM layer only snapshots and schedules:

```bash
python train.py \
    --expt_name my_run \
    --input_modality video,location,obb,gaze \
    --decoder voxel_query --voxel_query_type fourier \
    --voxel_bias factorised --voxel_bias_init prior \
    --readout cls --location_tokens joint \
    --fusion iou --fused_cache true --fixture_obbs true \
    --time_encoding sinusoidal --time_encoding_max_wavelength 15 \
    --token_layernorm true --type_embeddings unified --type_embedding_scale 0.5 \
    --class_embedding_gain true \
    --voxel_centers_dir "$PWD/new_voxel_centers" \
    --num_transformer_layers 2 --learning_rate 1e-5 \
    --batch_size 16 --num_epochs 1 --eval_steps 0.01 \
    --observation_time 30 --anticipation_time 5 --future_time 180 \
    --train_participants P01,P02,P03,P04,P05,P06,P08,P09 \
    --val_participants "" --test_participants P07 \
    --use_wandb false
```

---

## Configuration reference

86 of `train.py`'s 90 flags are wired to an environment variable in
`train_sbatch.sh`, all with defaults (`--use_static_env` is passed as a constant
`true`). Below, **Env** is what you export,
**Flag** is the `train.py` argument, **Default** is `train_sbatch.sh`'s value.
`python train.py --help` carries a longer explanation for each.

Boolean flags accept `true`/`false`.

### Episode geometry — what the task is

| Env | Flag | Default | Meaning |
|---|---|---|---|
| `OBS_TIME` | `--observation_time` | `30` | Observation window length, seconds. One token stream per second. |
| `ANTICIP_TIME` | `--anticipation_time` | `5` | Gap between the window's end and the prediction window's start. `0` makes it a recall task, not anticipation. |
| `FUTURE_TIME` | `--future_time` | `180` | Prediction window length, seconds. Longer windows raise the positive rate and are **not** comparable across settings. |
| `TIME_ANCHOR` | `--time_anchor` | `trajectory` | What second 0 is anchored to: `trajectory`, `gaze`, or `slam`. Production uses `gaze`. |

### Splits and subsampling

| Env | Flag | Default | Meaning |
|---|---|---|---|
| `TRAIN_PARTICIPANTS` | `--train_participants` | `P01,…,P06` | Comma-separated participant IDs. |
| `VAL_PARTICIPANTS` | `--val_participants` | `P07` | Set to `""` for no validation split — the test set then drives eval and best-model selection. |
| `TEST_PARTICIPANTS` | `--test_participants` | `P08,P09` | |
| `TRAIN_N` / `VAL_N` / `TEST_N` | `--train_n` / `--val_n` / `--test_n` | `0` | Cap on episodes per split; `0` = all. Use for smoke runs. |
| `MIN_VIDEO_DURATION` | `--min_video_duration` | `0` | Skip videos shorter than this many seconds, before any episode work. |
| `REQUIRE_VALID_LOCATION` | `--require_valid_location` | `false` | Drop episodes with no valid SLAM second in the window. **Set true when training on `location` alone**, or all-padding windows produce NaN. |
| `GT_INTERACTIONS_RANGE` | `--gt_interactions_range` | `""` | Keep only episodes whose prediction window holds a GT count in this inclusive `lo,hi` range. Changes the positive rate, so a filtered PR-AUC is not comparable with an unfiltered one. |
| `GT_VOXELS_RANGE` | `--gt_voxels_range` | `""` | As above, on GT *voxels*. |

Three flags are not reachable through `train_sbatch.sh` at all — pass them to
`train.py` directly: `--seed` (42), `--debug_subset` (0),
`--dataloader_num_workers` (4).

### Modalities

| Env | Flag | Default | Meaning |
|---|---|---|---|
| `MODALITY` | `--input_modality` | `video,location,env` | Comma-separated subset of `video`, `location`, `env`, `obb`, `gaze`, `trajectory`. Production uses `video,location,obb,gaze`. |
| `LOCATION_TOKENS` | `--location_tokens` | `axis` | `axis` = three per-axis streams (90 tokens); `joint` = one token per second (30). `joint` shares the decoder's spatial vocabulary. |
| `GAZE_TOKENS` | `--gaze_tokens` | `axis` | How gaze enters: `axis` or `joint`. |
| `GAZE_MAX_HZ` | `--gaze_max_hz` | `0` | Downsample gaze samples; `0` keeps every one. |
| `GAZE_TIME_OFFSET_S` | `--gaze_time_offset_s` | `0.1` | Residual offset on the first-gaze-timestamp anchor. |

`env` and `obb` are two ways to encode the same Boxer detections: `env` as 4096
per-voxel semantic tokens, `obb` as one token per fused 3D box. They are
alternatives, not complements.

> `trajectory` (object-movement tracks) is implemented and its flags are
> accepted, but it is not part of the final method and is not documented here.

### Environment / OBB construction

| Env | Flag | Default | Meaning |
|---|---|---|---|
| `FUSION` | `--fusion` | `grid` | How detections are clustered into boxes. `grid`: snap centres to a 0.5 m cell, group by `(name, cell)`. `iou`: link by 3D overlap + OWLv2 class proximity — **requires the precomputed cache**. |
| `FUSED_CACHE` | `--fused_cache` | `false` | Use the precomputed cache. Must be `true` with `FUSION=iou`. |
| `MAX_OBB_TOKENS` | `--max_obb_tokens` | `250` | Slot cap for OBB tokens; windows above it are truncated. |
| `OBB_VOXEL_GRID_PRUNE` | `--obb_voxel_grid_prune` | `false` | Drop boxes the voxel grid cannot represent. |
| `FIXTURE_OBBS` | `--fixture_obbs` | `false` | Replace redundant furniture detections with the exact Digital-Twin fixture. |
| `FIXTURE_IOU_THRESH` | `--fixture_iou_thresh` | `0.4` | 3D IoU above which a detection counts as the fixture. |
| `FIXTURE_COS_THRESH` | `--fixture_cos_thresh` | `0.7` | OWLv2 class cosine above which the labels count as the same thing. |
| `BOXER_CONF_THRESHOLD` | `--boxer_conf_threshold` | `-1.0` | Detection confidence floor; `-1` selects the per-fusion default (0.6 under `iou`). |
| `ENV_SOURCE` | `--env_source` | `boxer_grid` | What defines the grid extent when `VOXEL_CENTERS_DIR` is unset: `boxer_grid` or `detic_grid`. |
| `VOXEL_CENTERS_DIR` | `--voxel_centers_dir` | `""` | Use fixed per-participant grids from `<dir>/<pid>_voxel_centers.npy`. **Set this to `$PWD/new_voxel_centers`** — it overrides `ENV_SOURCE`. |
| `ENV_POSITIONAL_ENCODING` | `--env_positional_encoding` | `false` | Add a positional code to env tokens. |
| `ENV_CLASS_AVERAGING` | `--env_class_averaging` | `false` | Average class embeddings per voxel instead of picking one. |
| `ENV_MAX_CLASSES_PER_VOXEL` | `--env_max_classes_per_voxel` | `4` | Classes retained per voxel. |
| `ENV_FALLBACK_AVERAGING` | `--env_fallback_averaging` | `off` | `off`, `merge`, or `empty_only`. |
| `ENV_ORPHAN_FALLBACK` | `--env_orphan_fallback` | `true` | Keep detections that fall outside the grid by snapping them in. |

### Encoder and token scale

| Env | Flag | Default | Meaning |
|---|---|---|---|
| `NUM_LAYERS` | `--num_transformer_layers` | `2` | Stacked `TransformerEncoderLayer` blocks (hidden 768, 8 heads). |
| `ENCODER_DROPOUT` | `--encoder_dropout` | `0.1` | |
| `OBB_TOKEN_DROPOUT` | `--obb_token_dropout` | `0.0` | Drop whole OBB tokens during training. |
| `TOKEN_LAYERNORM` | `--token_layernorm` | `false` | Per-segment LayerNorm on each base token before the time code is added. Puts every modality on one scale. |
| `TYPE_EMBEDDINGS` | `--type_embeddings` | `separate` | `separate` = a per-modality marker parameter; `unified` = one `[5, 768]` table indexed by modality. |
| `TYPE_EMBEDDING_SCALE` | `--type_embedding_scale` | `1.0` | Scales that whole table. |
| `CLASS_EMBEDDING_INIT` | `--class_embedding_init` | `""` | Initialise object-class embeddings from an `.npz` (`class_embeddings.npz`). |
| `CLASS_EMBEDDING_TRAINABLE` | `--class_embedding_trainable` | `false` | Fine-tune them. |
| `CLASS_EMBEDDING_GAIN` | `--class_embedding_gain` | `false` | Learn a scalar amplitude for the class embedding. |
| `CLASS_EMBEDDING_GAIN_INIT` | `--class_embedding_gain_init` | `-1.0` | `-1` = derive from the data. |
| `READOUT` | `--readout` | `token0` | `token0` decodes sequence seat 0 (the first visual token); `cls` prepends a dedicated learned token. **Under `--decoder voxel_query` the readout seat is not used for the output** — the queries are — but a `cls` token still joins the sequence and is attendable. |

### Spatial and time encoding

| Env | Flag | Default | Meaning |
|---|---|---|---|
| `POSITION_ENCODING` | `--position_encoding` | `auto` | Spatial code on input tokens: `fourier`, `none`, or `auto` (follow `VOXEL_QUERY_TYPE`). |
| `POSITION_ENCODING_MODE` | `--position_encoding_mode` | `replace` | `replace` or `add`. |
| `POSITION_ENCODING_GAIN_INIT` | `--position_encoding_gain_init` | `48.0` | Amplitude of the spatial code. |
| `TIME_ENCODING` | `--time_encoding` | `none` | Time-to-anchor code shared by every per-second modality: `none`, `sinusoidal`, or `fourier`. Without it the 30 seconds are an unordered bag. |
| `TIME_ENCODING_MAX_WAVELENGTH` | `--time_encoding_max_wavelength` | `0.0` | Longest sinusoidal wavelength, seconds. `0` = `2·OBS_TIME`. Production uses `15`. |
| `TIME_ENCODING_SCALE` | `--time_encoding_scale` | `10.0` | Read **only** by the `fourier` branch; inert under `sinusoidal`. |
| `TIME_ENCODING_GAIN` | `--time_encoding_gain` | `fixed` | `fixed` or `learned` amplitude. |
| `TIME_ENCODING_GAIN_INIT` | `--time_encoding_gain_init` | `8.0` | Starting amplitude. |

### Decoder

| Env | Flag | Default | Meaning |
|---|---|---|---|
| `DECODER` | `--decoder` | `linear` | `linear`: one `Linear(768, 4096)` off the readout token. `voxel_query`: 4096 coordinate queries each cross-attend the sequence, with a shared `Linear(d, 1)` head. |
| `VOXEL_QUERY_TYPE` | `--voxel_query_type` | `axis_sum` | How a query encodes its cell: `axis_sum` (shared per-axis index tables), `fourier` (code of the normalised centre), `learned` (a free `[4096, 768]` table). |
| `VOXEL_QUERY_FOURIER_SCALE` | `--voxel_query_fourier_scale` | `10.0` | Fourier frequency spread; accepts per-axis `x,y,z`. |
| `VOXEL_QUERY_UNTIED` | `--voxel_query_untied` | `false` | Give queries their own code instead of sharing the input tokens'. |
| `DECODER_HEADS` | `--decoder_heads` | `8` | |
| `DECODER_ITERS` | `--decoder_iters` | `1` | Cross-attention rounds. |
| `DECODER_RESIDUAL` | `--decoder_residual` | `raw` | How query and attention output combine: `raw` (`q + attn`, where the query norm is ~55 against ~1 for the attention output), `normed`, `balanced`, `concat`, or `evidence` (the query never reaches the head — the arm that tests where the static map lives). |
| `DECODER_FFN` | `--decoder_ffn` | `false` | Add a feed-forward block. Not supported with `concat`. |
| `DECODER_FFN_DIM` | `--decoder_ffn_dim` | `3072` | |
| `DECODER_DROPOUT` | `--decoder_dropout` | `0.0` | |

### Output bias

With a ~1:200 positive rate, where the output bias starts matters more than most
things here.

| Env | Flag | Default | Meaning |
|---|---|---|---|
| `VOXEL_BIAS_INIT` | `--voxel_bias_init` | `zero` | `zero` keeps `nn.Linear`'s init (the model starts at `sigmoid(0)=0.5`). `prior` sets every entry to the train split's `log(p/(1−p))` — the standard imbalance fix. `empirical` / `empirical_factorised` give each voxel its own train log-odds, which hands the model the training kitchens' layout **on purpose**: measurement arms, not defaults. |
| `VOXEL_BIAS` | `--voxel_bias` | `factorised` | Parameterisation: `factorised` (48 params, axis marginals), `full` (4096), `mlp`. |
| `VOXEL_BIAS_MLP_HIDDEN` | `--voxel_bias_mlp_hidden` | `32,32` | `VOXEL_BIAS=mlp` only. |
| `VOXEL_BIAS_SCALE` | `--voxel_bias_scale` | `1.0` | Scales the spatial part around the pooled mean. |
| `VOXEL_BIAS_LR` | `--voxel_bias_lr` | `-1.0` | Separate LR for the bias; `-1` = the global one. Never weight-decayed. |

### Loss

| Env | Flag | Default | Meaning |
|---|---|---|---|
| `LOSS` | `--loss` | `bce` | `bce` = `BCEWithLogitsLoss` over all 4096 voxels. `focal` = `(1−p_t)^γ`-modulated BCE; `γ=0` reduces it to α-weighted BCE exactly. |
| `FOCAL_GAMMA` | `--focal_gamma` | `2.0` | `LOSS=focal` only. |
| `FOCAL_ALPHA` | `--focal_alpha` | `0.25` | `LOSS=focal` only. |
| `POS_WEIGHT` | `--pos_weight` | `-1.0` | `BCEWithLogitsLoss` positive weight; `≤0` disables it. The production runs address the imbalance through `VOXEL_BIAS_INIT=prior` instead. |

### Optimisation and logging

| Env | Flag | Default | Meaning |
|---|---|---|---|
| `LR` | `--learning_rate` | `1e-5` | |
| `LR_SCHEDULER_TYPE` | `--lr_scheduler_type` | `linear` | Any HF schedule name. |
| `WARMUP_RATIO` | `--warmup_ratio` | `0.0` | |
| `WEIGHT_DECAY` | `--weight_decay` | `0.0` | Never applied to the voxel bias or the learned gains. |
| `BATCH_SIZE` | `--batch_size` | `16` | |
| `NUM_EPOCHS` | `--num_epochs` | `10` | |
| `EVAL_STEPS` | `--eval_steps` | `0.1` | Eval + checkpoint cadence as a fraction of total steps. |
| `EXPT_NAME` | `--expt_name` | job name | Output directory under `results/`. |
| `WANDB_PROJECT` | `--wandb_project` | `FIction-HD-EPIC-v2` | |
| `USE_WANDB` | `--use_wandb` | `true` | Set `USE_WANDB=false` to train without Weights & Biases. Leave it on and wandb needs an API key on the compute node, or the run stalls waiting for a login. |

---

## Reproducing the headline configuration

The best configuration is 9-fold leave-one-participant-out with modality
`video,location,obb,gaze` ("vlog"). One command runs all nine folds:

```bash
export HDEPIC_ROOT=/path/to/HD-EPIC
export SLURM_ACCOUNT=<acct>
./experiments/production/submit_loocv_fixture_obb_v2_gaze.sh      # all 9 folds
FOLDS="P07" ./experiments/production/submit_loocv_fixture_obb_v2_gaze.sh   # one fold
```

It requires the fused-OBB cache (preprocessing step 2). The configuration, in
full:

```bash
MODALITY=video,location,obb,gaze   TIME_ANCHOR=gaze
GAZE_TOKENS=axis                   GAZE_MAX_HZ=0        GAZE_TIME_OFFSET_S=0.1
LOCATION_TOKENS=joint
VOXEL_CENTERS_DIR=$PWD/new_voxel_centers

FUSION=iou                         FUSED_CACHE=true     BOXER_CONF_THRESHOLD=-1.0
MAX_OBB_TOKENS=250                 OBB_VOXEL_GRID_PRUNE=true
FIXTURE_OBBS=true                  FIXTURE_IOU_THRESH=0.4   FIXTURE_COS_THRESH=0.7

DECODER=voxel_query                VOXEL_QUERY_TYPE=fourier
VOXEL_QUERY_FOURIER_SCALE=10.0     VOXEL_QUERY_UNTIED=false
DECODER_HEADS=8                    DECODER_ITERS=1      DECODER_RESIDUAL=raw
DECODER_FFN=false                  DECODER_DROPOUT=0.0
READOUT=cls

VOXEL_BIAS=factorised              VOXEL_BIAS_INIT=prior
VOXEL_BIAS_SCALE=1.0               VOXEL_BIAS_LR=-1.0

POSITION_ENCODING=auto             POSITION_ENCODING_MODE=replace
POSITION_ENCODING_GAIN_INIT=48.0
TIME_ENCODING=sinusoidal           TIME_ENCODING_MAX_WAVELENGTH=15
TIME_ENCODING_GAIN=learned         TIME_ENCODING_GAIN_INIT=8.0

TOKEN_LAYERNORM=true               TYPE_EMBEDDINGS=unified
TYPE_EMBEDDING_SCALE=0.5           CLASS_EMBEDDING_GAIN=true
CLASS_EMBEDDING_INIT=class_embeddings.npz

LOSS=bce                           POS_WEIGHT=-1.0      WEIGHT_DECAY=0.0
LR=1e-5                            NUM_LAYERS=2         ENCODER_DROPOUT=0.1
BATCH_SIZE=16                      NUM_EPOCHS=1         EVAL_STEPS=0.01
OBS_TIME=30                        ANTICIP_TIME=5       FUTURE_TIME=180
```

Two things to be aware of when comparing numbers:

- **`NUM_EPOCHS=1`** with `EVAL_STEPS=0.01` — the run evaluates 100 times within a single epoch. This is not an under-trained accident; longer runs were measured and the decoder's performance peaks early and then decays.
- **`VAL_PARTICIPANTS=""`** — there is no validation split, so the held-out test participant drives both evaluation and best-checkpoint selection. That is how the reference runs were done and is reproduced here deliberately, but it means the reported best-checkpoint score is selected on test.

The `vlo` baseline it derives from (`video,location,obb`, no gaze) is
`experiments/production/submit_loocv_fixture_obb_v2.sh`. For the ladder from the
original FIction architecture up to this one, see
[`experiments/README.md`](experiments/README.md).

---

## Outputs and metrics

Checkpoints land in `results/<expt_name>/<timestamp>_<expt_name>/checkpoint-N/`:

| File | Contents |
|---|---|
| `model.safetensors` | weights |
| `config.json` | **every resolved config value** — the record of what the run actually used |
| `predictions.pkl` | raw logits per episode, `[n_episodes, 4096]` |
| `labels.pkl` | the binary GT voxel masks |
| `indexes.pkl` | episode indices, to join predictions back to episodes |
| `trainer_state.json` | loss/metric history and `best_model_checkpoint` |

The model outputs **raw logits**, never probabilities. `BCEWithLogitsLoss`
applies the sigmoid internally during training; evaluation applies it explicitly
in `compute_metrics`.

Metrics, computed per episode and averaged:

| Metric | Definition |
|---|---|
| `pr_auc` | Average precision over the 4096 voxels. **The headline metric** — with a ~1:200 positive rate, ROC-AUC is optimistic. |
| `roc_auc` | Per-episode ROC-AUC. |
| `iou_thresh-{0.01,0.05,0.1,0.2}-topk-5` | IoU after binarising at that probability threshold, forcing at least the top 5 voxels on. |
| `chamfer_dist_thresh-…-topk-5` | Chamfer distance between predicted and GT voxel sets, normalised by the grid diagonal. |
| `best_chamfer_dist` | Best Chamfer across the four threshold settings. |

Episodes whose GT is all-zero or all-one are excluded from the AUCs, since
neither is defined there.

---

## Tests

```bash
python tests/smoke_decoder.py           # ~60 checks, no data needed, seconds
python tests/smoke_position_encoding.py
python tests/smoke_time_encoding.py
python tests/smoke_token_scale.py
```

See [`tests/README.md`](tests/README.md), including the two suites that need the
dataset and a note on the bit-identical regression check.

## Analysis

45 scripts, each answering one question about a trained model or the data —
where a predicted voxel gets its evidence, whether the model is anticipating or
returning a fixed map of the kitchen, which modality carries the prediction.
Indexed one by one in [`analysis/README.md`](analysis/README.md).

---

## Citation and acknowledgements

This code accompanies the thesis **“A Window into the Future”** (2026). If you
use it, please cite the thesis — GitHub's *Cite this repository* box reads
[`CITATION.cff`](CITATION.cff).

It builds directly on three pieces of upstream work. Please cite them from
their own pages rather than from this list:

- **FIction** (CVPR 2025) — the task formulation and architecture this adapts. <https://github.com/thechargedneutron/FIction>
- **HD-EPIC** — the dataset, its Digital-Twin meshes, Boxer 3D detections, SLAM and gaze. <https://hd-epic.github.io/>
- **EgoVLPv2** — the frozen video encoder used for the per-second visual features. <https://github.com/facebookresearch/EgoVLPv2>

Code in this repository is MIT-licensed (see [LICENSE](LICENSE)). The HD-EPIC
dataset, the EgoVLPv2 checkpoint, and the OWLv2-derived embeddings in
`class_embeddings.npz` carry their own terms.
