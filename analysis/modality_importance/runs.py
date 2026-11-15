"""Frozen specification of the two trained runs and their four checkpoints.

Everything here was READ OFF each snapshot's own ``out/*_<jobid>.out``, which
echoes every environment variable ``train_sbatch.sh`` turned into a CLI flag.
Nothing is inferred from defaults, because the defaults differ from what these
runs used (``fusion``, ``time_anchor``, ``env_orphan_fallback`` and
``trajectory_simple`` all differ from the CLI defaults).

Two facts that make this file necessary rather than convenient:

  * ``config.json`` stores only what ``CustomConfig`` carries -- the MODEL side.
    Every loader-side option (anticipation_time, future_time, fusion,
    fused_cache, time_anchor, voxel_centers_dir, ...) is absent from it and can
    only come from the run's log. Getting one wrong silently changes the input
    tensors and every attribution computed from them.
  * ``--val_participants`` was empty, so ``train.py`` ran in no-val mode and the
    TEST split drove both periodic eval and best-model selection. The "best"
    checkpoint of each run was therefore selected on the very episodes the
    attribution runs over. That is why both best and final are analysed.

"Final" is the last checkpoint WRITTEN, not the last optimizer step: HF Trainer's
save_total_limit kept the best plus the two most recent, and the runs ended at
steps 1899 (A) and 2005 (B). Steps 1881 and 1995 are 99.0% and 99.5% through.
The literal terminal weights were never saved and cannot be recovered.
"""
import glob
import os

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SNAPSHOT_BASE = os.environ.get("SNAPSHOTS_ROOT", "/data/video_datasets/HD-EPIC/snapshots")

# Loader options shared by both runs. Mirrors train.py's ds_kwargs (train.py:752)
# exactly, with the values echoed in both logs. observation_time and the model-side
# options are NOT here -- they come from each checkpoint's config.json, so a
# mismatch between the two is impossible by construction.
DATASET_KWARGS = dict(
    anticipation_time=5,
    future_time=180,
    static_env=True,
    env_source='boxer_grid',
    voxel_centers_dir=os.path.join(REPO, 'new_voxel_centers'),
    min_video_duration=0.0,
    require_valid_location=False,
    gt_interactions_range='',
    gt_voxels_range='',
    env_orphan_fallback=False,
    trajectory_simple=True,
    trajectory_exclude_unfinished=False,
    max_trajectory_tokens=16,
    env_class_averaging=False,
    env_max_classes_per_voxel=4,
    env_fallback_averaging='off',
    gaze_max_hz=0.0,
    gaze_time_offset_s=0.1,
    time_anchor='gaze',
    fusion='iou',
    fused_cache=True,
    boxer_conf_threshold=None,   # BOXER_CONF_THRESHOLD=-1.0 -> None (train.py:775)
)

RUNS = {
    'A': dict(
        label='Run A',
        snapshot='2026-08-27_23-22-29_loocv_P08_vlot_vq_fourier_factorised_cls_prior_pe_locjoint_trajend_clsemb_tlnoff_e1_iou',
        slurm_job=639400,
        wandb='OBB_tokens2/ibasqwe0',
        test_participants=['P08'],
        train_participants=['P01', 'P02', 'P03', 'P04', 'P05', 'P06', 'P07', 'P09'],
        n_test_episodes=5007,
        max_steps=1899,
        checkpoints={'best': 855, 'final': 1881},
        eval_pr_auc={'best': 0.3941446116814449, 'final': 0.3759},
    ),
    'B': dict(
        label='Run B',
        snapshot='2026-08-27_23-22-26_loocv_P04_vlot_vq_fourier_factorised_cls_prior_pe_locjoint_trajend_clsemb_tlnoff_e1_iou',
        slurm_job=639397,
        wandb='OBB_tokens2/8z3ubpfm',
        test_participants=['P04'],
        train_participants=['P01', 'P02', 'P03', 'P05', 'P06', 'P07', 'P08', 'P09'],
        n_test_episodes=3311,
        max_steps=2005,
        checkpoints={'best': 756, 'final': 1995},
        eval_pr_auc={'best': 0.14172990926491952, 'final': 0.1337},
    ),
}

CHECKPOINT_KEYS = [(r, w) for r in ('A', 'B') for w in ('best', 'final')]


def snapshot_dir(run):
    return os.path.join(SNAPSHOT_BASE, RUNS[run]['snapshot'])


def checkpoint_dir(run, which):
    """Absolute path to one checkpoint directory.

    The results subdirectory is timestamped at job start, so it is globbed rather
    than hardcoded -- but exactly one must match, and a run with two results
    directories would mean the snapshot was reused, which would make "the"
    checkpoint ambiguous. Raise instead of picking one.
    """
    step = RUNS[run]['checkpoints'][which]
    pattern = os.path.join(snapshot_dir(run), 'results', '*', f'checkpoint-{step}')
    hits = sorted(glob.glob(pattern))
    if len(hits) != 1:
        raise SystemExit(
            f"expected exactly one directory matching {pattern}, found {len(hits)}: {hits}")
    return hits[0]


def run_log(run):
    """The .out file whose echoed environment DATASET_KWARGS was transcribed from."""
    hits = sorted(glob.glob(os.path.join(snapshot_dir(run), 'out', '*.out')))
    return hits[0] if hits else None


def tag(run, which):
    """Short stable identifier used in output paths, e.g. 'A_best_P08_ckpt855'."""
    pid = RUNS[run]['test_participants'][0]
    return f"{run}_{which}_{pid}_ckpt{RUNS[run]['checkpoints'][which]}"
