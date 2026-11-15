# Preprocessing

Two steps turn the HD-EPIC release into something `train.py` can read. The first
is **required for every run**. The second is required only for `FUSION=iou`,
which is what the production configuration uses.

Everything else the model needs — the per-participant voxel grids
(`new_voxel_centers/`), the OWLv2 class embeddings (`class_embeddings.npz`) and
the Boxer class vocabulary (`data/boxer_class_vocab.json`) — already ships in
this repository. In particular the voxel grids **cannot be regenerated** from
anything here: they are rotated HD-EPIC Digital-Twin exports and the per-kitchen
rotation exists only inside the `.npy` files.

---

## Step 1 — EgoVLPv2 visual features (required)

Every training run reads one 4096-d feature vector per second of video. Nothing
else in the pipeline produces them, and the loader raises if they are missing.

### What you need

1. **The EgoVLPv2 code.** Clone it somewhere outside this repo:

   ```bash
   git clone https://github.com/facebookresearch/EgoVLPv2.git
   ```

   `EGOVLPV2_REPO` must point at the directory that *contains* the inner
   `EgoVLPv2` package, so that `from EgoVLPv2.model.model import FrozenInTime`
   resolves. For a clone at `~/src/EgoVLPv2` that is usually
   `~/src/EgoVLPv2/EgoVLPv2`. Check with:

   ```bash
   ls "$EGOVLPV2_REPO/EgoVLPv2/model/model.py"
   ```

2. **The EgoVLPv2 checkpoint** (`EgoVLPv2.pth`), from that repository's own
   download instructions. It is not redistributed here.

3. **HD-EPIC videos** at `$HDEPIC_ROOT/Videos/<pid>/*.mp4`.

### Run it

```bash
export HDEPIC_ROOT=/path/to/HD-EPIC
export EGOVLPV2_REPO=/path/to/EgoVLPv2/EgoVLPv2
export EGOVLPV2_WEIGHTS=/path/to/EgoVLPv2.pth

# one participant per job -- a full pass is many hours
PARTICIPANTS="P01" sbatch --account=<acct> preprocessing/run_egovlpv2.sh
```

Or directly, without SLURM:

```bash
python preprocessing/extract_features.py --participant P01 --gpu_batch_size 16
```

`--weights`, `--video_root` and `--output_root` override the environment if you
need to.

### What it writes

```
$FEATURES_BASE/<split>/<pid>/<video_id>_features.pt
```

where `<split>` is `train` for P01–P06, `val` for P07, `test` for P08–P09 — the
same mapping as `loader.PARTICIPANT_TO_SPLIT`, which is how training finds these
files. `FEATURES_BASE` defaults to
`$HDEPIC_ROOT/fiction_hd_epic/intermediate/extracted_rgb_features_EgoVLPv2`.

Each file is a dict:

```python
{'features': torch.Tensor,      # [n_seconds, 4096], float32
 'frame_indices': list[int]}    # the frame index each second was sampled from
```

One feature per second of video: for second *t*, 16 frames are sampled evenly
across `[t·fps, (t+1)·fps)`, resized to 224×224, ImageNet-normalised, and pushed
through `FrozenInTime.compute_video` (projection_dim 4096).

### Notes

- **It is restartable.** Any video whose `_features.pt` already exists is
  skipped, so a job killed on the wall clock can be resubmitted unchanged.
- Frames are read **sequentially** in 300-second chunks rather than by seeking,
  which is far faster on these long videos and bounds RAM at roughly 2–3 GB per
  chunk.
- Raise `--gpu_batch_size` if the GPU is underused; it only affects the forward
  pass, not the decode.

---

## Step 2 — IoU-fused OBB cache (required for `FUSION=iou`)

Boxer emits ~30 detections per second across ~858 LVIS classes, so one video's
CSV is hundreds of thousands of rows and a 30-second window holds ~10k. The
`iou` fusion mode links detections that overlap in 3D *and* whose classes are
close in OWLv2 embedding space, then chains those links.

That costs about **6.4 s per window** against ~0.04 s for `FUSION=grid`, and a
split holds roughly 28,800 windows. Fusing inline would take ~51 hours inside
`Dataset.__init__`, so **the loader refuses** to do more than 500 uncached
windows and raises instead.

Build the cache once. Note this script **submits itself** — run it directly, do
not `sbatch` it:

```bash
# all nine participants, one array task each
SLURM_ACCOUNT=<acct> ./preprocessing/precompute_fused_obbs_sbatch.sh

# or a subset
SLURM_ACCOUNT=<acct> PARTICIPANTS=P03,P08 ./preprocessing/precompute_fused_obbs_sbatch.sh
```

It writes `fused_obbs_iou.pkl` beside each Boxer CSV, under
`$BOXER_OBBS_ROOT/<video_id>/` — about **1.0 GB** in total.

A participant is 2,255–4,641 windows at ~6.36 s/window, i.e. 4.0–8.2 h serially,
which would exceed a 6 h partition limit for P01/P03/P08/P09. The videos within
a participant are independent, so the array task fans out over `WORKERS=8`
processes by default and the worst case (P03) lands near 1 h.

Then train with:

```bash
FUSION=iou FUSED_CACHE=true ./submit_train.sh my_run
```

If the cache job is still running, gate the training job on it rather than
letting it start and fail:

```bash
DEPENDENCY=afterok:<array_jobid> FUSION=iou FUSED_CACHE=true ./submit_train.sh my_run
```

`FUSION=grid` (the default) needs none of this — it snaps detection centres to a
0.5 m cell and groups by `(name, cell)`. It is the cheaper, lower-fidelity path.
