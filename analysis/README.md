# Analysis

Each script answers one question. They are standalone — none is imported by the
training path — and they are the measurements the thesis's claims rest on.

Run them from anywhere; each puts the repository root on `sys.path` itself:

```bash
python analysis/<script>.py --help      # most take arguments
```

**Outputs** go to `analysis/out/`, which is gitignored: every result here is
regenerable, and some files exceed 10 MB.

**Inputs.** Scripts are marked below with what they need:

| Mark | Needs |
|---|---|
| **D** | HD-EPIC data only (annotations, Boxer CSVs, SLAM) — no trained model |
| **C** | a trained checkpoint, passed as `--checkpoint <snapshot>/results/<run>/checkpoint-N` |
| **S** | the snapshot *tree*, to find runs by name — set `SNAPSHOTS_ROOT` if yours is elsewhere |

> **Three scripts pin specific run names.** `plot_pr_auc_vs_video_length.py`,
> `pr_auc_vs_gt_voxels.py` and `scatter_per_video.py` hold hard-coded snapshot
> directory names from July 2026 (e.g.
> `2026-07-21_02-01-38_loocv_P01_newvox_lr1e5_nl2_traj`). Those are run
> identifiers, not paths, so no environment variable fixes them — edit the `RUNS`
> dict at the top of each to point at runs you have.

---

## 1. Dataset and episode statistics — **D**

What the data looks like before any model touches it. An "episode" is a 30 s
observation window, a 5 s anticipation gap, then a 180 s prediction window (see
`loader.build_episodes`).

| Script | Question |
|---|---|
| `obs_window_gt_stats.py` | How many ground-truth interactions fall inside each episode's *observation* window? The distribution that decides whether the task is anticipation or recall. |
| `pred_window_gt_stats.py` | Per-participant distributions of what an episode is actually asked to predict. |
| `obs_window_interactions.py` | Per-video export of the same observation-window counts, using the track-level start/end definition. One CSV per video. |
| `plot_obs_interactions.py` | Box plot of the observation-window interaction counts per episode. |
| `obb_token_count.py` | How many *fused* OBBs does a real observation window contain? This is what sets `--max_obb_tokens`. |
| `trajectory_token_count.py` | How many object-movement tracks overlap a 30 s observation window? |
| `object_movement_stats.py` | How far do objects move per track, in distance and in time? |
| `person_vs_object_trajectory.py` | Person path length vs. object displacement, per track — are the two signals redundant? |
| `env_voxel_occupancy.py` | Occupancy of the 4096-cell semantic voxel grid the model receives as its `env` token stream, per episode. |
| `plot_activity_gantt.py` | Per-participant Gantt chart of high-level activity timelines, one bar per video. Needs the `high-level/activities` annotations. |

## 2. Grid and input audits — **D**

| Script | Question |
|---|---|
| `grid_geometry_audit.py` | Does the per-participant 16³ grid's geometry explain the per-participant PR-AUC spread? Every kitchen gets 16 cells per axis regardless of its size, so cell volume differs between participants. (Answer, measured: no — voxel size is ruled out.) |
| `compare_grid_extents.py` | How does the grid extent differ between `--env_source boxer_grid` and `detic_grid`? |
| `boxer_coverage_audit.py` | Which Boxer OBB CSVs are missing or truncated, i.e. need rerunning? Run this first if a fold behaves strangely. |
| `build_boxer_class_vocab.py` | Regenerates `data/boxer_class_vocab.json` — the `sem_id → canonical LVIS name` lookup — by scanning every Boxer CSV. One-off; the output already ships. |

## 3. What the model actually reads — **C**

The group that distinguishes "the model uses this input" from "this input is
present".

| Script | Question |
|---|---|
| `voxel_query_attention.py` | Where does a predicted voxel get its evidence? For each true-positive voxel, records attention mass by segment (video / location / gaze / obb / trajectory / sep / cls), mass by Δt within the video segment, and whether the top-attended box sits in the queried voxel. Mass concentrating as Δt→0 is evidence of anticipation; a flat profile says the video tokens are a bag of scene context. |
| `cls_sink_probe.py` | Is the CLS token an attention sink? Under `--decoder voxel_query` the CLS readout seat is never read, so attention is the only way it can matter. |
| `input_swap_ablation.py` | Which parts of the input actually move the prediction? Swaps one modality's tokens between episodes and measures the change. Pair with `input_swap_merge.py` to merge the per-fold JSONs into one table; `input_swap_sbatch.sh` runs it across folds. |
| `gaze_sensitivity.py` | Does the trained model's output actually depend on the gaze tokens? |
| `prediction_variability.py` | How much of the output responds to the input at all, versus being constant across episodes? |
| `env_permutation_invariance_check.py` | Sanity check: does the *order* of the 4096 env tokens change the output? It must not. |
| `nn_probe.py` | Is the spatial signal present in the EgoVLPv2 features, or does the model discard it? A nearest-neighbour probe directly on the features. `nn_probe_summary.py` summarises its per-participant JSONs; `nn_probe_sbatch.sh` runs it per fold. |

## 4. Anticipation or memorisation? — **C**, **S**

The central question: is the model predicting the future, or returning a fixed
map of the kitchen it was trained on?

| Script | Question |
|---|---|
| `static_map_baseline.py` | Is the model anticipating, or returning a fixed map? Compares it against a per-kitchen occupancy prior. |
| `static_map_decomposition.py` | Where does the static per-voxel map live — the 48-parameter factorised bias, or the retrieved evidence? |
| `label_prior_transfer.py` | What does a per-voxel prior learned from *other* kitchens score on the held-out one? The cross-kitchen floor of the metric. |
| `axis_marginals.py` | Is the learned spatial prior transferable, or one kitchen's floor plan? |
| `count_analysis.py` | Is the model's error a *count* error or a *placement* error? |

## 5. Decoder and encoding diagnostics — **C**

| Script | Question |
|---|---|
| `residual_balance.py` | How lopsided is the voxel-query decoder's residual on real data? Under `--decoder_residual raw`, the query norm is ~55 against an attention-output norm of ~1, so the query can dominate what the head sees. |
| `time_code_similarity.py` | How similar are adjacent seconds in the time code, against distant ones? Checks that `--time_encoding` produces a usable ordering rather than noise or a constant. |

## 6. Per-video and per-fold result plots — **C**, **S**

| Script | Question |
|---|---|
| `plot_pr_auc_vs_video_length.py` | Per-participant scatter: source-video length vs. per-episode PR-AUC. *(pins run names)* |
| `plot_pr_auc_exclude_short_videos.py` | Threshold sweep: what happens to PR-AUC if episodes from short videos are dropped? |
| `pr_auc_vs_gt_voxels.py` | Correlation between the number of GT voxels in the prediction window and per-episode PR-AUC. *(pins run names)* |
| `gt_voxels_over_time.py` | Per-video scatter: prediction-window start second vs. total GT voxels in that window. |
| `scatter_per_video.py` | One plot grid per participant, one scatter per `video_id`. *(pins run names)* |

## 7. Run bookkeeping — **S**

| Script | Question |
|---|---|
| `collect_runs.py` | What was every run configured with, and what did it score? Recovers `(participant, tag, best eval PR-AUC, parameter count, ~40 flags)` from the snapshot directories by parsing the flag echo `train_sbatch.sh` writes into each job log. This is the only way to recover a finished run's configuration — the snapshot holds the code, the log holds the flags, nothing else records them. A flag absent from a log is reported as its code default with a trailing `*`, since older runs predate flags added later. Override the scan window with `SNAPSHOT_GLOB` / `SNAPSHOT_SINCE`. |

---

## `modality_importance/` — which modality carries the prediction?

A self-contained sub-study with its own documentation: read
[`ARCHITECTURE.md`](modality_importance/ARCHITECTURE.md) for the token layout it
assumes and [`RESULTS_METHOD1.md`](modality_importance/RESULTS_METHOD1.md) for
the findings.

| Script | Role |
|---|---|
| `runs.py` | Frozen specification of the two trained runs and their four checkpoints the whole sub-study is measured on. Start here. |
| `common.py` | Shared machinery for all methods — loading, token layout, batching. |
| `m1_gradients.py` | **Method 1**: gradient-based token and modality importance. |
| `m1_report.py` | Method 1 reporting — aggregation, statistics, plots, markdown summary. |
| `m2_ablation.py` | **Method 2**: masking / ablation importance. |
| `verify_layout.py` | Data-free check that every checkpoint loads strictly and lays tokens out identically. Run this before trusting any attribution. |
| `m1_voxel_alignment.py` | Why is the attribution almost independent of which voxels are the target? |
| `m1_ig_diagnose.py` | Why does Integrated Gradients fail its completeness check, and is it fixable? |
| `check_precision.py` | Why fp32 logits computed here differ from the run's saved `predictions.pkl`. |
| `m1_sbatch.sh` | SLURM driver for the Method 1 gradient pass. |
