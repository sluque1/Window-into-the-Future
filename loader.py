"""
HD-EPIC future-interaction dataset (v2).

Unlike v1 (FIction_HD_EPIC), episodes are NOT read from pre-generated per-config
JSON files. They are built **on the fly in __init__** from the HD-EPIC
scene-and-object-movements annotations, parameterized by the runtime
``observation_time`` / ``anticipation_time`` / ``future_time``. Sweeping those
hyperparameters therefore needs no offline preprocessing.

Inputs (all raw / self-contained):
  * Visual : extracted_rgb_features_EgoVLPv2/<pid>/<video_id>_features.pt (1/sec)
  * Location: SLAM closed_loop_trajectory.csv (via vrs_to_multi_slam.json)
  * Env    : boxer_obbs/<video_id>/boxer_3dbbs.csv  (see boxer_env.py)
  * GT      : scene-and-object-movements/{assoc_info,mask_info}.json
             (object 3D world locations inside the prediction window)

All three spatial signals (GT, location, env) are mapped into ONE per-video
16x16x16 voxel grid computed from the Boxer OBBs UNION the SLAM trajectory, so
they are spatially consistent.
"""

import json
import math
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from boxer_env import MAX_OBB_TOKENS, BoxerEnvBuilder
from fixture_obbs import (DEFAULT_COS_THRESH as FIXTURE_DEFAULT_COS,
                          DEFAULT_IOU_THRESH as FIXTURE_DEFAULT_IOU,
                          VoxelGrid as FixtureVoxelGrid, build_window_boxes,
                          class_candidate_index, load_fixtures,
                          prune_outside_grid)
from gaze import (DEFAULT_TIME_OFFSET_S as GAZE_DEFAULT_TIME_OFFSET_S, VoxelLattice,
                  build_episode_gaze_tokens, gaze_t0, has_gaze_data, load_gaze_csv)
from object_class_vocab import class_idx_for_name
from slam_utils import load_raw_slam, load_raw_slam_poses
from trajectory_tokens import (build_episode_trajectory_tokens, trajectory_raw_dim,
                                build_episode_trajectory_tokens_simple, trajectory_raw_dim_simple)

# --- Path constants ---
# Every root below defaults to the layout this was developed against and can be
# overridden from the environment, so the same code runs on another machine
# without edits. HDEPIC_ROOT moves all five at once; the individual variables are
# there for the case where one directory sits apart from the rest (most often
# FEATURES_BASE, since the EgoVLPv2 features are the one input you generate
# yourself rather than download -- see preprocessing/README.md).
HDEPIC_ROOT = os.environ.get("HDEPIC_ROOT", "/data/video_datasets/HD-EPIC")

FEATURES_BASE = os.environ.get(
    "FEATURES_BASE",
    f"{HDEPIC_ROOT}/fiction_hd_epic/intermediate/extracted_rgb_features_EgoVLPv2")
SLAM_BASE = os.environ.get("SLAM_BASE", f"{HDEPIC_ROOT}/SLAM-and-Gaze")
ASSOC_PATH = os.environ.get(
    "ASSOC_PATH",
    f"{HDEPIC_ROOT}/hd-epic-annotations/scene-and-object-movements/assoc_info.json")
MASK_PATH = os.environ.get(
    "MASK_PATH",
    f"{HDEPIC_ROOT}/hd-epic-annotations/scene-and-object-movements/mask_info.json")
VIDEOS_BASE = os.environ.get("VIDEOS_BASE", f"{HDEPIC_ROOT}/Videos")

FPS = 30
NUM_VOXELS = 4096
GRID = 16

# How many windows fusion='iou' may fuse inline (i.e. missing from fused_cache)
# before __init__ refuses. 500 * 6.36 s ~= 53 min, already more than a smoke test
# should cost and far less than the hours a full split would take.
IOU_UNCACHED_WINDOW_LIMIT = 500

# Participant -> split. Train P01-P06, Val P07, Test P08-P09 (matches v1).
PARTICIPANT_TO_SPLIT = {
    "P01": "train", "P02": "train", "P03": "train",
    "P04": "train", "P05": "train", "P06": "train",
    "P07": "val",
    "P08": "test", "P09": "test",
}

# Module-level annotation cache so train/val/test instances share one copy (~23MB).
_ASSOC = None
_MASK = None


def _load_annotations():
    global _ASSOC, _MASK
    if _ASSOC is None:
        with open(ASSOC_PATH) as f:
            _ASSOC = json.load(f)
        with open(MASK_PATH) as f:
            _MASK = json.load(f)
    return _ASSOC, _MASK


# Module-level cache of Videos/<pid>/durations.txt, so a min_video_duration
# filter is a cheap dict lookup rather than a per-instance re-parse.
_VIDEO_DURATIONS = {}   # pid -> {video_id: duration_seconds}


def _load_video_durations(pid):
    if pid not in _VIDEO_DURATIONS:
        durations = {}
        path = os.path.join(VIDEOS_BASE, pid, "durations.txt")
        if os.path.exists(path):
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    name, dur = line.split()
                    video_id = name[:-4] if name.endswith(".mp4") else name
                    durations[video_id] = float(dur)
        _VIDEO_DURATIONS[pid] = durations
    return _VIDEO_DURATIONS[pid]


def _has_valid_location(loc):
    """True iff loc is a 3-element list of finite numbers."""
    if loc is None:
        return False
    try:
        if len(loc) != 3:
            return False
    except TypeError:
        return False
    for x in loc:
        if not isinstance(x, (int, float)) or (isinstance(x, float) and math.isnan(x)):
            return False
    return True


def coord_to_voxel(xyz, voxel_centers_flat):
    """Flat index [0,4095] of the voxel centre nearest to world point xyz."""
    dists = np.linalg.norm(voxel_centers_flat - np.asarray(xyz, dtype=np.float64), axis=1)
    return int(np.argmin(dists))


# =====================================================================
# Episode construction (in-memory port of v1's generate_episodes.py)
# =====================================================================

def build_video_events(v_assoc, v_mask):
    """Return (events, anchors) for one video.

    events: list of {'timestamp','location_3d'} for every mask with a valid
            3d_location. Both the GT pool and the anchor set are drawn from these.
    anchors: sorted unique mask timestamps (each a candidate "current time").
    """
    events = []
    for info in v_assoc.values():
        for track in info["tracks"]:
            for mask_id in track["masks"]:
                m = v_mask.get(mask_id)
                if m is None or not _has_valid_location(m.get("3d_location")):
                    continue
                events.append({
                    "timestamp": m["frame_number"] / FPS,
                    "location_3d": m["3d_location"],
                })
    anchors = sorted({e["timestamp"] for e in events})
    return events, anchors


def build_video_tracks(v_assoc, v_mask):
    """Return one dict per usable object-movement track for one video.

    A track is usable for trajectory reconstruction iff it has exactly 2
    masks, both with a valid 3d_location, ordered by frame_number as
    (start, end), and a positive duration -- matching the filter used in
    analysis/object_movement_stats.py and the person-vs-object analysis.
    Tracks with 1 mask (object appears/disappears mid-track) or >=3 masks are
    dropped: there is no well-defined single (start, end) pair for them.
    """
    tracks = []
    for info in v_assoc.values():
        name = info.get("name", "")
        class_idx = class_idx_for_name(name)   # see object_class_vocab.py
        for track in info["tracks"]:
            if len(track["masks"]) != 2:
                continue
            pts = []
            for mask_id in track["masks"]:
                m = v_mask.get(mask_id)
                if m is None or not _has_valid_location(m.get("3d_location")):
                    continue
                pts.append((m["frame_number"], np.asarray(m["3d_location"], dtype=np.float64)))
            if len(pts) != 2:
                continue
            pts.sort(key=lambda p: p[0])
            t_start, t_end = pts[0][0] / FPS, pts[1][0] / FPS
            if t_end <= t_start:
                continue
            tracks.append({
                "name": name,
                "class_idx": class_idx,
                "t_start": t_start,
                "t_end": t_end,
                "start_xyz": pts[0][1],
                "end_xyz": pts[1][1],
            })
    return tracks


def _parse_range(value):
    """(lo, hi) from "lo,hi", a 2-sequence, or None. Inclusive on both ends."""
    if value is None or value == '':
        return None
    if isinstance(value, (list, tuple)):
        parts = list(value)
    else:
        parts = [p.strip() for p in str(value).split(',') if p.strip()]
    if len(parts) != 2:
        raise ValueError(f"range must be 'lo,hi', got {value!r}")
    lo, hi = int(parts[0]), int(parts[1])
    if lo > hi:
        raise ValueError(f"range lo must be <= hi, got {lo},{hi}")
    return (lo, hi)


def build_episodes(events, anchors, observation_time, anticipation_time, future_time):
    """Build episode dicts for one video given the runtime time params.

    Each anchor is a "current time". Observation window precedes it by the
    anticipation gap; the prediction window is [anchor, anchor + future_time].
    GT = all events falling inside the prediction window. Mirrors v1 exactly,
    but with the windows parameterized at call time.
    """
    episodes = []
    for anchor_t in anchors:
        obs_end = max(0.0, anchor_t - anticipation_time)
        if obs_end < 1.0:            # not enough observation history
            continue
        obs_start = max(0.0, obs_end - observation_time)
        pred_start = obs_end + anticipation_time     # == anchor_t
        pred_end = pred_start + future_time
        gt = [e["location_3d"] for e in events
              if pred_start <= e["timestamp"] <= pred_end]
        if not gt:
            continue
        episodes.append({
            "obs_start": obs_start,
            "obs_end": obs_end,
            "gt_locations": gt,
        })
    return episodes


# =====================================================================
# Dataset
# =====================================================================

class HDEpicInteractionDataset(Dataset):
    def __init__(self, split='train', observation_time=30, anticipation_time=5,
                 future_time=180, participants=None, static_env=True,
                 env_source='boxer_grid', voxel_centers_dir=None,
                 max_trajectory_tokens=16, min_video_duration=0.0,
                 require_valid_location=False,
                 gt_interactions_range=None, gt_voxels_range=None,
                 env_orphan_fallback=True, trajectory_simple=False,
                 trajectory_exclude_unfinished=False,
                 env_class_averaging=False, env_max_classes_per_voxel=4,
                 env_fallback_averaging='off',
                 max_obb_tokens=MAX_OBB_TOKENS,
                 gaze_max_hz=0.0,
                 gaze_time_offset_s=GAZE_DEFAULT_TIME_OFFSET_S,
                 time_anchor='trajectory',
                 fusion='grid', fused_cache=None, boxer_conf_threshold=None,
                 obb_voxel_grid_prune=False, fixture_obbs=False,
                 fixture_iou_thresh=FIXTURE_DEFAULT_IOU,
                 fixture_cos_thresh=FIXTURE_DEFAULT_COS,
                 boxer_debug=False, verbose=True):
        self.split = split
        self.observation_time = int(observation_time)
        self.anticipation_time = float(anticipation_time)
        self.future_time = float(future_time)
        # Videos shorter than this (seconds, from Videos/<pid>/durations.txt)
        # are skipped before any SLAM/Boxer/episode work is done for them.
        self.min_video_duration = float(min_video_duration)
        # Drop episodes whose observation window holds no valid SLAM location.
        # Location-only training needs this: with every token padding, the whole
        # attention mask is True and the softmax returns NaN.
        self.require_valid_location = bool(require_valid_location)
        # Keep only episodes whose prediction window holds a GT count inside each
        # given [lo, hi] (inclusive). An episode must pass BOTH to survive. None
        # disables that filter, which is the default, so an unfiltered run is
        # unaffected. See the filter block in __init__ for what this does to the
        # metric -- it is not a free change.
        self.gt_interactions_range = _parse_range(gt_interactions_range)
        self.gt_voxels_range = _parse_range(gt_voxels_range)
        # PROVISIONAL cap on trajectory tokens per episode -- see
        # trajectory_tokens.py's module docstring; the selection policy
        # (currently: most-recent-first) is expected to be revisited.
        self.max_trajectory_tokens = int(max_trajectory_tokens)
        # Simplified trajectory tokens (times + voxel coords only, no interior
        # shape points, no person-path branch check) -- see trajectory_tokens.py.
        self.trajectory_simple = bool(trajectory_simple)
        # Drop object tracks still in motion at obs_end instead of clipping them
        # to the window. Clipping interpolates along the straight line to the
        # track's TRUE end, which lies in the future -- so the clipped token
        # encodes the destination's direction and arrival speed, and the track
        # can be one of the episode's own GT events. See
        # trajectory_tokens._select_overlapping.
        self.trajectory_exclude_unfinished = bool(trajectory_exclude_unfinished)
        # Average the class embeddings of every object overlapping a voxel
        # (up to env_max_classes_per_voxel) instead of keeping only the
        # highest-confidence winner -- see boxer_env.py's build_env_multi.
        self.env_class_averaging = bool(env_class_averaging)
        self.env_max_classes_per_voxel = int(env_max_classes_per_voxel)
        # Single-winner pass first, then average in ONLY the boxes that pass
        # missed -- see boxer_env.py's _voxelize_fused_boxes_fallback_avg.
        #   'off'        -- disabled
        #   'merge'      -- a missed box joins its nearest voxel whether or not
        #                   that voxel already has a winner
        #   'empty_only' -- a missed box is kept only where the voxel is empty
        if env_fallback_averaging not in ('off', 'merge', 'empty_only'):
            raise ValueError(
                f"Unknown env_fallback_averaging: {env_fallback_averaging!r} "
                "(expected 'off', 'merge' or 'empty_only')")
        self.env_fallback_averaging = env_fallback_averaging
        # Both averaging modes hand the model a [4096, K] class grid + validity
        # mask instead of a single class id per voxel.
        self.env_is_multi = self.env_class_averaging or self.env_fallback_averaging != 'off'
        # OBB-token modality: one token per fused 3D box (voxel of its centre +
        # volume bin + class) instead of one token per voxel. See
        # boxer_env.build_obb_tokens and MAX_OBB_TOKENS for the measured
        # distribution the cap is drawn from. Always built, like trajectory and
        # gaze tokens, so the modality can be switched on at the model without
        # rebuilding the dataset.
        self.max_obb_tokens = int(max_obb_tokens)
        # --- OBB voxel-grid prune + Digital-Twin fixtures (fixture_obbs.py) ---
        # obb_voxel_grid_prune: drop fused boxes whose centre lies outside the
        #   participant's oriented voxel grid. Without it boxer_env's
        #   _nearest_voxel_ijk snaps an arbitrarily distant box onto a boundary
        #   voxel, since it is an argmin over the 4096 centres with no notion of
        #   being outside. Measured: 1.71% of all fused boxes are outside.
        # fixture_obbs: additionally drop each surviving box that a Digital-Twin
        #   fixture already represents (3D IoU > fixture_iou_thresh AND class
        #   cosine > fixture_cos_thresh against that same fixture), then append
        #   the participant's fixture boxes themselves. Implies the prune, since
        #   the fixtures are defined in grid terms.
        # Both are off by default, so runs made before this existed reproduce
        # with no flags.
        self.fixture_obbs = bool(fixture_obbs)
        self.obb_voxel_grid_prune = bool(obb_voxel_grid_prune) or self.fixture_obbs
        self.fixture_iou_thresh = float(fixture_iou_thresh)
        self.fixture_cos_thresh = float(fixture_cos_thresh)
        if self.obb_voxel_grid_prune and not voxel_centers_dir:
            raise ValueError(
                "obb_voxel_grid_prune / fixture_obbs need --voxel_centers_dir: "
                "the membership test and the fixture meshes are both defined "
                "against the fixed per-participant grid, and the per-video "
                "fallback grid is a different box per video.")
        # pid -> (FixtureVoxelGrid, fixtures, class-candidate index). Fixtures are
        # static geometry, so this is built once per participant, not per window.
        self._fixture_ctx = {}
        self._fixture_stats = {}
        # Gaze modality (see gaze.py): one token per second of the observation
        # window -- the voxel holding the most mass of a 3D Gaussian fitted to
        # that second's gaze points. Same length and alignment as
        # person_locations, so token i of each refers to the same second.
        self.gaze_max_hz = float(gaze_max_hz)
        self.gaze_time_offset_s = float(gaze_time_offset_s)
        # How the per-window Boxer detections are clustered into fused boxes,
        # feeding BOTH the env grid and the OBB tokens (they share
        # _window_fused_boxes, so this changes the two together).
        #   'grid' -- snap centres to a 0.5 m cell, group by (name, cell). What
        #       every run before 2026-08-26 used; the default so those stay
        #       reproducible with no flags.
        #   'iou'  -- link detections overlapping in 3D and semantically close in
        #       OWLv2 space, then chain the links (see iou_fusion.py).
        # NOTE 'iou' costs ~6.4 s per window against ~0.04 s, measured over 24
        # windows spanning all nine participants, and a production run holds
        # ~28,800 windows -- ~51 h in this single-threaded __init__, well past the
        # 6 h job limit. So 'iou' is only usable with a fused_cache built offline
        # by precompute_fused_iou.sh; without one this raises rather than hanging.
        if fusion not in ('grid', 'iou'):
            raise ValueError(f"Unknown fusion: {fusion!r} (expected 'grid' or 'iou')")
        self.fusion = fusion
        # True reads the per-video pickles precompute_fused_obbs.py writes beside
        # each video's CSV (boxer_obbs/<video_id>/fused_obbs_iou.pkl), loaded one
        # video at a time so a split only holds its own. A dict is used directly.
        # Each file records the settings it was built under and BoxerEnvBuilder
        # raises on a mismatch -- necessary because t0 (set by time_anchor)
        # changes which boxes a window holds without changing its key.
        self.fused_cache = fused_cache or None
        # BoxerEnvBuilder's own conf_threshold default is 0.6, the value the IoU
        # settings were tuned at -- but every 'grid' run to date used 0.5, and
        # _fuse_boxes reads it, so following the class default would silently
        # change legacy runs. Pin it per strategy unless explicitly overridden.
        if boxer_conf_threshold is None:
            boxer_conf_threshold = 0.5 if fusion == 'grid' else 0.6
        self.boxer_conf_threshold = float(boxer_conf_threshold)
        # Which device-clock instant counts as video second 0 for every
        # device-clock input (person_locations AND the Boxer env windows):
        #   'trajectory' -- the SLAM trajectory's first sample (legacy). The
        #       trajectory only starts once SLAM has initialised, a fixed
        #       ~0.95-1.50 s into the recording, so every read lands ~1 s LATER
        #       in the video than the window asked for.
        #   'gaze'       -- first gaze timestamp + gaze_time_offset_s, the
        #       anchor gaze.py already uses and the one validated against
        #       HD-EPIC's annotated look-at points. Puts location, env, gaze,
        #       the visual features and the GT labels on one clock.
        # Falls back to the trajectory start on videos with no gaze CSV.
        if time_anchor not in ('trajectory', 'gaze'):
            raise ValueError(f"Unknown time_anchor: {time_anchor!r} "
                             "(expected 'trajectory' or 'gaze')")
        self.time_anchor = time_anchor
        self.static_env = static_env   # v2 always uses a single per-window env snapshot
        if env_source not in ('boxer_grid', 'detic_grid'):
            raise ValueError(f"Unknown env_source: {env_source!r} "
                             "(expected 'boxer_grid' or 'detic_grid')")
        self.env_source = env_source   # which detector defines the voxel grid extent
        # If set, the voxel grid is NOT derived per-video from OBBs/SLAM. Instead a
        # fixed per-participant grid is loaded from
        # ``<voxel_centers_dir>/<pid>_voxel_centers.npy`` (a (4096,3) array) and
        # shared by all of that participant's videos. GT, actor location and Boxer
        # env content are all mapped into this fixed grid; env_source is ignored.
        self.voxel_centers_dir = voxel_centers_dir or None
        self.verbose = verbose

        self.dataset = []   # one dict per episode
        # fusion/conf_threshold are passed explicitly rather than left to
        # BoxerEnvBuilder's own defaults (which are 'iou'/0.6): the training
        # entry point defaults to the legacy grid path so runs made before
        # 2026-08-26 reproduce without flags. See self.fusion above.
        self.boxer = BoxerEnvBuilder(debug=boxer_debug,
                                     orphan_fallback=env_orphan_fallback,
                                     fusion=self.fusion,
                                     conf_threshold=self.boxer_conf_threshold,
                                     fused_cache=self.fused_cache)

        # Per-video caches (filled in __init__, inherited by forked workers)
        self._t0 = {}                 # video_id -> device-clock seconds at video-second 0
        self._voxel_centers_flat = {} # video_id -> (4096, 3) float64
        self._slam_sec = {}           # video_id -> (n_sec, 3) float64 (nan where missing)
        self._features_cache = {}     # feature_path -> {'features', 'frame_indices'}
        self._env_cache = {}          # (video_id, obs_start, obs_end) -> (4096,) int64
        self._obb_cache = {}          # (video_id, obs_start, obs_end) -> (voxels, volumes, classes, mask)
        self._fixed_vc_by_pid = {}    # pid -> (16,16,16,3) float64 (fixed-grid mode)
        self._tracks = {}             # video_id -> list of usable movement tracks
        self._slam_raw = {}           # video_id -> (rel_times, xyz) or None (lazy, see _get_raw_slam)
        self._gaze_cache = {}         # (video_id, obs_start, obs_end) -> (voxels[N,3], mask[N])
        self._anchor_shift = {}       # video_id -> seconds the trajectory anchor was late by
        self._lattice = {}            # video_id -> gaze.VoxelLattice over that video's grid

        assoc, mask = _load_annotations()

        # Which participants for this split. Entries may be participant ids
        # ("P07") or full video ids ("P07-20240211-123051"), mixed freely --
        # full ids enable video-level splits within a participant.
        if participants is not None:
            wanted_pids = set(participants)
        else:
            wanted_pids = {p for p, s in PARTICIPANT_TO_SPLIT.items() if s == split}

        video_ids = sorted(v for v in assoc
                           if v.split('-')[0] in wanted_pids or v in wanted_pids)

        n_no_grid = 0
        n_no_boxer = 0
        n_no_slam = 0
        n_no_gaze = 0
        n_detic_grid = 0        # videos whose grid extent came from Detic OBBs
        n_detic_fallback = 0    # detic_grid mode but no Detic OBBs -> Boxer grid
        n_too_short = 0         # videos skipped by min_video_duration
        used_videos = 0

        for video_id in tqdm(video_ids, desc=f"[{split}] building episodes", disable=not verbose):
            pid = video_id.split('-')[0]

            # Cheapest possible check first: a durations.txt dict lookup, so
            # short videos never pay for SLAM/Boxer/grid/episode work at all.
            if self.min_video_duration > 0:
                duration = _load_video_durations(pid).get(video_id)
                if duration is not None and duration < self.min_video_duration:
                    n_too_short += 1
                    continue

            feature_path = os.path.join(FEATURES_BASE, pid, f"{video_id}_features.pt")
            if not os.path.exists(feature_path):
                continue

            # --- SLAM trajectory: t0 + per-second world positions ---
            t0, slam_sec = self._load_slam(pid, video_id)
            has_slam = slam_sec is not None
            if not has_slam:
                n_no_slam += 1

            # --- Per-video voxel grid extent ---
            # boxer_grid: AABB(Boxer OBB corners UNION SLAM trajectory).
            # detic_grid: AABB(Detic OBB corners UNION SLAM trajectory); env
            #   *content* is still Boxer. Videos with no Detic OBBs fall back to
            #   the Boxer grid. Either way the chosen grid is cached in the boxer
            #   builder so GT, person, and env content all map into the same grid.
            has_boxer = self.boxer.has_boxer_data(video_id)
            if not has_boxer:
                n_no_boxer += 1
            traj_xyz = slam_sec[np.isfinite(slam_sec).all(axis=1)] if has_slam else None
            if self.voxel_centers_dir is not None:
                # Fixed per-participant grid: same 16^3 centres for every video of
                # this participant, independent of Boxer/Detic/SLAM extent.
                vc = self._load_fixed_voxel_centers(pid)
                self.boxer.set_voxel_centers(video_id, vc)
            elif self.env_source == 'detic_grid' and self.boxer.has_detic_data(video_id):
                vc = self.boxer.compute_and_cache_detic_voxel_centers(video_id, traj_xyz)
                if vc is not None:
                    n_detic_grid += 1
            else:
                if self.env_source == 'detic_grid':
                    n_detic_fallback += 1
                vc = self.boxer.compute_and_cache_voxel_centers(video_id, traj_xyz)
            if vc is None:
                n_no_grid += 1
                continue   # cannot place GT without a grid

            if t0 is None:
                # No SLAM trajectory: person_locations are all padding anyway,
                # but the Boxer env windows still need an anchor, and under the
                # gaze anchor there is one available without SLAM.
                t0 = self._anchor_t0(pid, video_id, None)
            self._t0[video_id] = t0 if t0 is not None else 0.0
            self._voxel_centers_flat[video_id] = vc.reshape(-1, 3)
            if has_slam:
                self._slam_sec[video_id] = slam_sec

            has_gaze = has_gaze_data(pid, video_id)
            if not has_gaze:
                n_no_gaze += 1

            # --- Build episodes for this video (runtime-parameterized) ---
            events, anchors = build_video_events(assoc[video_id], mask[video_id])
            if not events:
                continue
            self._tracks[video_id] = build_video_tracks(assoc[video_id], mask[video_id])
            episodes = build_episodes(events, anchors, self.observation_time,
                                      self.anticipation_time, self.future_time)
            if not episodes:
                continue
            for ep in episodes:
                self.dataset.append({
                    'video_id': video_id,
                    'participant_id': pid,
                    'feature_path': feature_path,
                    'obs_start': ep['obs_start'],
                    'obs_end': ep['obs_end'],
                    'gt_locations': ep['gt_locations'],
                    'has_boxer': has_boxer,
                    'has_slam': has_slam,
                    'has_gaze': has_gaze,
                })
            used_videos += 1

        # Same validity test __getitem__ applies per second (finite SLAM sample
        # inside the window); an episode where no second passes it would be
        # all-padding to the location modality.
        if self.require_valid_location:
            def _any_valid_location(d):
                slam_sec = self._slam_sec.get(d['video_id'])
                if slam_sec is None:
                    return False
                return any(0 <= t < len(slam_sec) and np.isfinite(slam_sec[t]).all()
                           for t in range(int(d['obs_start']), int(d['obs_end'])))
            n_before = len(self.dataset)
            self.dataset = [d for d in self.dataset if _any_valid_location(d)]
            if verbose and n_before != len(self.dataset):
                print(f"[{split}] require_valid_location: dropped "
                      f"{n_before - len(self.dataset)} of {n_before} episodes "
                      f"(no valid location second in the observation window)")

        # --- GT-size filter (gt_interactions_range / gt_voxels_range) ---
        # Keep only episodes whose prediction window holds a GT count inside each
        # given [lo, hi]; an episode must pass BOTH. Applied here, before the env
        # and OBB precompute below, so dropped episodes cost no box fusion.
        #
        # Two things to be deliberate about. It applies to EVERY split, train and
        # test alike -- it is a property of the episode, not of the model. And
        # dropping episodes changes the positive rate, to which PR-AUC is
        # sensitive, so a filtered number is NOT comparable with an unfiltered
        # one; read it against a prior recomputed on the same filtered set
        # (analysis/static_map_baseline.py) rather than against a 180s run.
        if self.gt_interactions_range or self.gt_voxels_range:
            # coord_to_voxel is a 4096-point distance search, and the same event
            # recurs across every overlapping window, so memoise per (video, xyz)
            # rather than paying for it once per episode-occurrence.
            memo = {}

            def _keep(d):
                if self.gt_interactions_range is not None:
                    lo, hi = self.gt_interactions_range
                    if not lo <= len(d['gt_locations']) <= hi:
                        return False
                if self.gt_voxels_range is not None:
                    lo, hi = self.gt_voxels_range
                    vc = self._voxel_centers_flat.get(d['video_id'])
                    if vc is None:
                        return False
                    hit = set()
                    for loc in d['gt_locations']:
                        key = (d['video_id'], loc[0], loc[1], loc[2])
                        flat = memo.get(key)
                        if flat is None:
                            flat = coord_to_voxel(loc, vc)
                            memo[key] = flat
                        hit.add(flat)
                    # Distinct voxels, matching what __getitem__ writes into
                    # labels['interaction'] (gt_voxels[flat] = 1, so duplicates
                    # collapse) -- this must count the same thing the metric does.
                    if not lo <= len(hit) <= hi:
                        return False
                return True

            n_before = len(self.dataset)
            self.dataset = [d for d in self.dataset if _keep(d)]
            if verbose:
                print(f"[{split}] gt filter: kept {len(self.dataset)} of {n_before} "
                      f"episodes ({100.0 * len(self.dataset) / max(n_before, 1):.1f}%) "
                      f"-- interactions in {self.gt_interactions_range}, "
                      f"voxels in {self.gt_voxels_range}")
            if not self.dataset:
                raise ValueError(
                    f"[{split}] the gt filter removed every episode "
                    f"(interactions {self.gt_interactions_range}, "
                    f"voxels {self.gt_voxels_range})")

        # --- IoU fusion is only affordable from a cache ---------------------
        # Both precompute loops below are single-threaded and run before the
        # first training step. Grid fusion is ~0.04 s/window so that is a few
        # minutes; IoU fusion measured 6.36 s/window (mean over 24 windows across
        # all nine participants, median 5.82, max 15.2), which on a full split is
        # tens of hours and overruns the job's wall clock without ever training.
        # Fail here, with the arithmetic, rather than after hours of silence.
        if self.fusion == 'iou':
            all_keys = {(d['video_id'], int(d['obs_start']), int(d['obs_end']))
                        for d in self.dataset}
            # Ask the builder, so this counts hits the same way _window_fused_boxes
            # will -- and so a cache built under the wrong settings raises here,
            # at startup, rather than after the first few windows.
            #
            # Only videos WITH a Boxer CSV can be counted: without one there is
            # nothing to fuse, _window_fused_boxes returns None immediately, and
            # precompute_fused_obbs.py rightly writes no cache. Counting them
            # made the three CSV-less videos (P02-20240210-113925,
            # P02-20240211-182630, P06-20240510-140459 -- no Boxer, SLAM or gaze)
            # look like 707 windows of pending work and aborted every run.
            uncached = [k for k in all_keys
                        if self.boxer.has_boxer_data(k[0])
                        and not self.boxer.has_fused_window(*k)]
            n_no_csv = sum(1 for k in all_keys if not self.boxer.has_boxer_data(k[0]))
            if len(uncached) > IOU_UNCACHED_WINDOW_LIMIT:
                raise RuntimeError(
                    f"[{split}] fusion='iou' with {len(uncached)} of "
                    f"{len(all_keys)} windows missing from fused_cache. At "
                    f"~6.4 s/window that is ~{len(uncached) * 6.36 / 3600:.1f} h "
                    f"of single-threaded work in __init__, before training "
                    f"starts. Build the cache first with "
                    f"./precompute_fused_obbs_sbatch.sh (it must be generated "
                    f"with the same --time_anchor, --observation_time and "
                    f"--anticipation_time as this run) and enable it with "
                    f"--fused_cache true / FUSED_CACHE=true. Raise "
                    f"loader.IOU_UNCACHED_WINDOW_LIMIT to override for a "
                    f"deliberately small run.")
            if uncached and verbose:
                print(f"[{split}] fusion=iou: {len(uncached)} of {len(all_keys)} "
                      f"windows not in the cache, fusing inline "
                      f"(~{len(uncached) * 6.36 / 60:.1f} min)")
            elif verbose:
                print(f"[{split}] fusion=iou: all {len(all_keys) - n_no_csv} "
                      f"fusable windows served from the fused cache"
                      + (f"; {n_no_csv} more are on videos with no Boxer CSV "
                         f"(empty env, no OBB tokens -- same as grid fusion)"
                         if n_no_csv else ""))

        # Pre-compute env snapshots once so forked DataLoader workers inherit them.
        if self.static_env:
            seen = set()
            for d in self.dataset:
                key = (d['video_id'], int(d['obs_start']), int(d['obs_end']))
                if key in seen:
                    continue
                seen.add(key)
                self._env_cache[key] = self._build_env(
                    d['video_id'], int(d['obs_start']), int(d['obs_end']))

        # Same for OBB tokens -- they reuse the very same per-window fusion the
        # env grid does, so precomputing here keeps it to one fusion per window.
        seen = set()
        n_obb_tokens = []
        for d in self.dataset:
            key = (d['video_id'], int(d['obs_start']), int(d['obs_end']))
            if key in seen:
                continue
            seen.add(key)
            tokens = self.boxer.build_obb_tokens(
                d['video_id'], key[1], key[2], self._t0[d['video_id']],
                max_tokens=self.max_obb_tokens,
                box_transform=self._obb_box_transform(d['participant_id']))
            self._obb_cache[key] = tokens
            n_obb_tokens.append(int(tokens[3].sum()))

        # Same for gaze tokens: reading a 60 Hz gaze CSV per worker per epoch
        # would be wasteful, and the token set for a given window is fixed.
        seen = set()
        n_gaze_tokens = []
        for d in self.dataset:
            key = (d['video_id'], int(d['obs_start']), int(d['obs_end']))
            if key in seen:
                continue
            seen.add(key)
            tokens = self._build_gaze(d['participant_id'], d['video_id'],
                                      int(d['obs_start']), int(d['obs_end']))
            self._gaze_cache[key] = tokens
            n_gaze_tokens.append(int(tokens[1].sum()))

        if verbose:
            print(f"[{split}] {len(self.dataset)} episodes from {used_videos} videos "
                  f"(obs={self.observation_time}s anticip={self.anticipation_time}s "
                  f"future={self.future_time}s)")
            shifts = np.asarray([s for s in self._anchor_shift.values()])
            if self.time_anchor == 'gaze' and len(shifts):
                print(f"[{split}] time anchor: GAZE -- video second 0 = first gaze "
                      f"timestamp + {self.gaze_time_offset_s}s, for person_locations "
                      f"AND the Boxer env windows. Device-clock reads moved "
                      f"{shifts.mean():.3f}s earlier on average "
                      f"(min {shifts.min():.3f}s, max {shifts.max():.3f}s); they were "
                      f"fetching content from that much later in the video.")
            else:
                print(f"[{split}] time anchor: TRAJECTORY (legacy) -- device-clock "
                      f"reads land ~1s later in the video than the window asked for")
            print(f"[{split}] coverage: {n_no_boxer} videos without Boxer, "
                  f"{n_no_slam} without SLAM, {n_no_gaze} without gaze, "
                  f"{n_no_grid} dropped (no grid at all)")
            if n_obb_tokens:
                no = np.asarray(n_obb_tokens)
                print(f"[{split}] obb tokens: mean {no.mean():.1f} median "
                      f"{np.median(no):.0f} max {no.max()} of "
                      f"{self.max_obb_tokens} slots; "
                      f"{(no == self.max_obb_tokens).sum()} windows hit the cap, "
                      f"{(no == 0).sum()} ({(no == 0).mean()*100:.1f}%) have NO "
                      f"boxes at all (all-padding -- fatal as the only modality)")
            if self.obb_voxel_grid_prune and self._fixture_stats:
                tot = {k: sum(s[k] for s in self._fixture_stats.values())
                       for k in ('n_in', 'n_grid_pruned', 'n_fixture_filtered',
                                 'n_fixtures_added', 'n_out', 'n_windows')}
                n_win = max(tot['n_windows'], 1)
                print(f"[{split}] obb pipeline over {n_win} windows: "
                      f"fused {tot['n_in']/n_win:.1f}/window "
                      f"-> -{tot['n_grid_pruned']/n_win:.1f} outside the voxel grid "
                      f"-> -{tot['n_fixture_filtered']/n_win:.1f} covered by a fixture "
                      f"-> +{tot['n_fixtures_added']/n_win:.1f} fixtures "
                      f"= {tot['n_out']/n_win:.1f}/window before the "
                      f"{self.max_obb_tokens}-token cap")
                if self.fixture_obbs:
                    per = ", ".join(
                        f"{pid}:{len(self._obb_fixture_ctx(pid)[1])}"
                        for pid in sorted(self._fixture_stats))
                    print(f"[{split}] fixtures per participant (own kitchen only, "
                          f"iou>{self.fixture_iou_thresh} & "
                          f"cos>{self.fixture_cos_thresh}): {per}")
            if n_gaze_tokens:
                ng = np.asarray(n_gaze_tokens)
                print(f"[{split}] gaze: {(ng > 0).mean()*100:.1f}% of windows produced "
                      f"tokens (mean {ng.mean():.1f}/{self.observation_time} "
                      f"seconds covered, max_hz={self.gaze_max_hz or 'all'})")
            if self.env_fallback_averaging != 'off':
                fs = self.boxer.fallback_stats
                n_left = fs.get('n_leftover', 0)
                print(f"[{split}] env fallback averaging ({self.env_fallback_averaging}): "
                      f"{n_left} leftover boxes over {fs.get('n_windows', 0)} windows "
                      f"-- {fs.get('n_leftover_on_empty', 0)} landed on an EMPTY voxel, "
                      f"{fs.get('n_leftover_on_filled', 0)} on a FILLED one, "
                      f"{fs.get('n_leftover_dropped', 0)} dropped by this policy")
                if self.env_fallback_averaging == 'empty_only' and n_left:
                    kept = n_left - fs.get('n_leftover_dropped', 0)
                    print(f"[{split}] NOTE: 'empty_only' kept {kept}/{n_left} "
                          f"({100*kept/n_left:.1f}%) of leftover boxes -- if that is "
                          f"near zero this run is equivalent to the single-winner baseline.")
            if self.min_video_duration > 0:
                print(f"[{split}] {n_too_short} videos skipped for being shorter than "
                      f"{self.min_video_duration}s (min_video_duration)")
            if self.voxel_centers_dir is not None:
                print(f"[{split}] grid extent: fixed per-participant grids from "
                      f"{self.voxel_centers_dir} (env_source ignored)")
                return
            print(f"[{split}] grid extent: env_source={self.env_source}", end="")
            if self.env_source == 'detic_grid':
                print(f" | {n_detic_grid} videos on Detic grid, "
                      f"{n_detic_fallback} fell back to Boxer grid")
            else:
                print(" | all videos on Boxer grid")

    # -- Env grid construction -------------------------------------------------

    def _obb_fixture_ctx(self, pid):
        """(grid, fixtures, candidate-index) for one participant, built once.

        In LOOCV each episode carries its OWN participant id and this is keyed
        by it, so a sample only ever sees its own kitchen's fixtures -- the test
        participant's fixtures never enter a training sample and vice versa.
        The fixtures are read from that participant's own Digital-Twin meshes,
        exactly as its voxel grid already is.
        """
        if pid not in self._fixture_ctx:
            grid = FixtureVoxelGrid.load(self.voxel_centers_dir, pid)
            fixtures = (load_fixtures(pid, self.voxel_centers_dir)
                        if self.fixture_obbs else [])
            cand = (class_candidate_index(fixtures, cos_thresh=self.fixture_cos_thresh)
                    if fixtures else None)
            self._fixture_ctx[pid] = (grid, fixtures, cand)
        return self._fixture_ctx[pid]

    def _obb_box_transform(self, pid):
        """The ``box_transform`` handed to ``build_obb_tokens``, or None.

        Order is fixed by design: 3a voxel-grid prune, then 3b fixture/class
        filter on the survivors, then the fixtures are appended. Boxes dropped
        at either stage are gone and never occupy a token slot. The cap is
        applied afterwards, inside ``_obb_tokens_from_fused``.
        """
        if not self.obb_voxel_grid_prune:
            return None
        grid, fixtures, cand = self._obb_fixture_ctx(pid)
        acc = self._fixture_stats.setdefault(
            pid, dict(n_in=0, n_grid_pruned=0, n_fixture_filtered=0,
                      n_fixtures_added=0, n_out=0, n_windows=0))

        def _transform(fused):
            if fixtures:
                boxes, s = build_window_boxes(
                    fused, grid, fixtures, iou_thresh=self.fixture_iou_thresh,
                    cos_thresh=self.fixture_cos_thresh, candidates=cand)
            else:
                kept = prune_outside_grid(fused, grid)
                boxes = kept
                s = dict(n_in=len(fused), n_grid_pruned=len(fused) - len(kept),
                         n_fixture_filtered=0, n_fixtures_added=0,
                         n_out=len(kept))
            for k in ('n_in', 'n_grid_pruned', 'n_fixture_filtered',
                      'n_fixtures_added', 'n_out'):
                acc[k] += s[k]
            acc['n_windows'] += 1
            return boxes

        return _transform

    def _build_env(self, video_id, obs_start, obs_end):
        """One env snapshot for [obs_start, obs_end), dispatching to the
        single-winner, all-classes-averaging, or fallback-averaging builder.

        Returns:
          env_is_multi=False: int64[4096] -- sem_id+1 per voxel, 0=empty.
          env_is_multi=True:  (int64[4096, K], bool[4096, K]) -- class ids
            (0=pad slot) and validity mask per voxel, K=env_max_classes_per_voxel.
        """
        t0 = self._t0[video_id]
        k = self.env_max_classes_per_voxel
        if self.env_fallback_averaging != 'off':
            grid_multi, mask = self.boxer.build_env_fallback_avg(
                video_id, obs_start, obs_end, t0, k_max=k,
                mode=self.env_fallback_averaging)
        elif self.env_class_averaging:
            grid_multi, mask = self.boxer.build_env_multi(
                video_id, obs_start, obs_end, t0, k_max=k)
        else:
            return self.boxer.build_env(
                video_id, obs_start, obs_end, t0).reshape(-1).astype(np.int64)
        return grid_multi.reshape(-1, k).astype(np.int64), mask.reshape(-1, k)

    # -- Gaze token construction ----------------------------------------------

    def _get_lattice(self, video_id):
        """Cached O(1) world-point -> voxel-index lookup for this video's grid."""
        if video_id not in self._lattice:
            self._lattice[video_id] = VoxelLattice(self._voxel_centers_flat[video_id])
        return self._lattice[video_id]

    def _build_gaze(self, pid, video_id, obs_start, obs_end):
        """Top-N most-looked-at voxels for one observation window (see gaze.py).

        NOTE the time anchor: gaze uses video-second 0 == the first gaze
        timestamp, which is the validated anchor, while ``person_locations`` and
        the Boxer windows above keep using the SLAM trajectory's start. They
        differ by a fixed ~1.0-1.5 s per participant. That inconsistency is
        deliberate and contained: correcting the others would shift every
        observation window and break comparability with the already-completed
        runs, so only the new modality adopts the corrected anchor.

        Returns (int64[observation_time, 3], int64[observation_time]).
        """
        return build_episode_gaze_tokens(
            load_gaze_csv(pid, video_id, self.gaze_max_hz),
            load_raw_slam_poses(pid, video_id),
            gaze_t0(pid, video_id, self.gaze_max_hz, self.gaze_time_offset_s),
            obs_start, obs_end, self._get_lattice(video_id),
            observation_time=self.observation_time)

    # -- Fixed per-participant voxel grid ------------------------------------

    def _load_fixed_voxel_centers(self, pid):
        """Load & cache the fixed (16,16,16,3) grid for a participant.

        Reads ``<voxel_centers_dir>/<pid>_voxel_centers.npy`` (a (4096,3) array,
        flattened in C order matching meshgrid indexing='ij').
        """
        if pid not in self._fixed_vc_by_pid:
            path = os.path.join(self.voxel_centers_dir, f"{pid}_voxel_centers.npy")
            vc = np.load(path).astype(np.float64)
            if vc.shape != (NUM_VOXELS, 3):
                raise ValueError(
                    f"{path}: expected ({NUM_VOXELS}, 3) voxel centres, got {vc.shape}")
            self._fixed_vc_by_pid[pid] = vc.reshape(GRID, GRID, GRID, 3)
        return self._fixed_vc_by_pid[pid]

    # -- SLAM loading --------------------------------------------------------

    def _anchor_t0(self, pid, video_id, traj_t0):
        """Device-clock seconds at video second 0, under the configured anchor.

        See ``time_anchor`` in ``__init__``. Returns ``traj_t0`` unchanged in
        'trajectory' mode, or when 'gaze' mode has no gaze CSV to anchor on.
        """
        if self.time_anchor == 'gaze':
            g0 = gaze_t0(pid, video_id, self.gaze_max_hz, self.gaze_time_offset_s)
            if g0 is not None:
                return g0
        return traj_t0

    def _load_slam(self, pid, video_id):
        """Return (t0_seconds, per_second_xyz [n_sec,3]) or (None, None).

        t0 = device-clock seconds at video second 0, per ``self.time_anchor``.
        Per-second positions are the nearest trajectory sample to each integer
        second; missing seconds are NaN.
        """
        mapping_path = os.path.join(SLAM_BASE, pid, "SLAM", "multi", "vrs_to_multi_slam.json")
        if not os.path.exists(mapping_path):
            return None, None
        with open(mapping_path) as f:
            mapping = json.load(f)
        multi_id = next((v for k, v in mapping.items() if video_id in k), None)
        if multi_id is None:
            return None, None
        traj_path = os.path.join(SLAM_BASE, pid, "SLAM", "multi", str(multi_id),
                                 "slam", "closed_loop_trajectory.csv")
        if not os.path.exists(traj_path):
            return None, None
        try:
            df = pd.read_csv(traj_path, usecols=[
                'tracking_timestamp_us', 'tx_world_device',
                'ty_world_device', 'tz_world_device'])
        except Exception:
            return None, None
        if df.empty:
            return None, None

        times = df['tracking_timestamp_us'].values.astype(np.float64) / 1e6  # seconds
        order = np.argsort(times)
        times = times[order]
        xyz = df[['tx_world_device', 'ty_world_device', 'tz_world_device']].values[order]
        traj_t0 = float(times[0])
        t0 = self._anchor_t0(pid, video_id, traj_t0)
        self._anchor_shift[video_id] = traj_t0 - t0
        rel = times - t0                       # video-seconds along the trajectory
        n_sec = int(math.floor(rel[-1])) + 1
        secs = np.arange(n_sec, dtype=np.float64)
        # nearest trajectory sample to each integer second (rel is sorted)
        pos = np.searchsorted(rel, secs)
        pos = np.clip(pos, 1, len(rel) - 1)
        left = pos - 1
        choose_left = np.abs(rel[left] - secs) <= np.abs(rel[pos] - secs)
        idx = np.where(choose_left, left, pos)
        per_sec = xyz[idx].astype(np.float64)
        # Under the gaze anchor the trajectory starts ~1 s AFTER video second 0,
        # so the first second or two have no sample of their own and the
        # nearest-sample clip above would report a stale position. Mark them
        # padding instead. No-op in 'trajectory' mode, where rel[0] == 0.
        per_sec[secs < rel[0] - 0.5] = np.nan
        return t0, per_sec

    # -- torch Dataset API ---------------------------------------------------

    def __len__(self):
        return len(self.dataset)

    def _get_features(self, path):
        if path not in self._features_cache:
            self._features_cache[path] = torch.load(path, map_location='cpu')
        return self._features_cache[path]

    def _get_raw_slam(self, pid, video_id):
        """Lazy-loaded, cached (rel_times, xyz) raw (~1kHz) SLAM trajectory, or
        (None, None). Separate from self._slam_sec (1Hz, used for
        person_locations) -- trajectory reconstruction needs the raw
        trajectory since many object movements last under a second.
        slam_utils.load_raw_slam already caches per traj file at module
        level, so this per-instance cache mainly saves the mapping-file
        lookup + dict indirection on repeat __getitem__ calls.
        """
        if video_id not in self._slam_raw:
            traj = load_raw_slam(pid, video_id)
            self._slam_raw[video_id] = (None, None) if traj is None else (traj[1], traj[2])
        return self._slam_raw[video_id]

    def __getitem__(self, index):
        datum = self.dataset[index]
        video_id = datum['video_id']
        obs_start = int(datum['obs_start'])
        obs_end = int(datum['obs_end'])
        vc_flat = self._voxel_centers_flat[video_id]
        debug = (index == 0)

        # --- 1. Visual features [obs, 4096] + mask ---
        feat = self._get_features(datum['feature_path'])
        features = feat['features']
        frame_indices = feat['frame_indices']
        feat_dim = features.shape[1]
        n = min(len(features), len(frame_indices))
        selected = [i for i in range(n)
                    if obs_start <= frame_indices[i] // FPS < obs_end]
        obs_features = features[selected] if selected else torch.zeros(0, feat_dim)

        visual_mask = torch.zeros(self.observation_time, dtype=torch.int)
        if obs_features.shape[0] < self.observation_time:
            pad_len = self.observation_time - obs_features.shape[0]
            obs_features = torch.cat([torch.zeros(pad_len, feat_dim), obs_features], dim=0)
            visual_mask[pad_len:] = 1
        else:
            obs_features = obs_features[:self.observation_time]
            visual_mask[:] = 1

        # --- 2. Environment grid (single per-window snapshot) ---
        # env_class_averaging=False: int64[4096] (sem_id+1 per voxel, 0=empty).
        # env_class_averaging=True:  (int64[4096,K], bool[4096,K]) class ids + mask.
        key = (video_id, obs_start, obs_end)
        if key in self._env_cache:
            env_grid = self._env_cache[key]
        else:
            env_grid = self._build_env(video_id, obs_start, obs_end)
        if self.env_is_multi:
            env_grid, env_grid_mask = env_grid

        # --- 3. Actor location [obs, 3] + mask (back-filled, pad at front) ---
        person_locations = torch.zeros(self.observation_time, 3, dtype=torch.int)
        person_location_mask = torch.zeros(self.observation_time, dtype=torch.int)
        slam_sec = self._slam_sec.get(video_id)
        if slam_sec is not None:
            loc_idx = self.observation_time - 1
            for t in range(obs_end - 1, obs_start - 1, -1):
                if loc_idx < 0:
                    break
                if 0 <= t < len(slam_sec) and np.isfinite(slam_sec[t]).all():
                    flat = coord_to_voxel(slam_sec[t], vc_flat)
                    ix, iy, iz = np.unravel_index(flat, (GRID, GRID, GRID))
                    person_locations[loc_idx] = torch.tensor([ix, iy, iz]) + 1
                    person_location_mask[loc_idx] = 1
                loc_idx -= 1

        # --- 4. GT interaction voxels [4096] ---
        gt_voxels = np.zeros(NUM_VOXELS, dtype=np.int64)
        for loc in datum['gt_locations']:
            gt_voxels[coord_to_voxel(loc, vc_flat)] = 1

        # --- 5. Trajectory tokens [max_trajectory_tokens, raw_dim] + mask + classes ---
        # Simplified path (self.trajectory_simple): times + voxel coords only,
        # no raw SLAM needed (trajectory.reconstruct_linear is a plain
        # constant-speed straight line between the track's true endpoints).
        tracks = self._tracks.get(video_id, [])
        if self.trajectory_simple:
            (trajectory_tokens, trajectory_mask, trajectory_classes,
             trajectory_start_voxel, trajectory_end_voxel,
             trajectory_distance) = build_episode_trajectory_tokens_simple(
                tracks, obs_start, obs_end, vc_flat, self.max_trajectory_tokens,
                exclude_unfinished=self.trajectory_exclude_unfinished)
        else:
            rel_times, person_xyz = self._get_raw_slam(datum['participant_id'], video_id)
            trajectory_tokens, trajectory_mask, trajectory_classes = build_episode_trajectory_tokens(
                tracks, obs_start, obs_end, rel_times, person_xyz, self.max_trajectory_tokens,
                exclude_unfinished=self.trajectory_exclude_unfinished)
            trajectory_start_voxel = trajectory_end_voxel = trajectory_distance = None

        # --- 5b. OBB tokens [max_obb_tokens] (voxel + volume bin + class) + mask ---
        # The per-box alternative to the env voxel grid: one token per fused
        # 3D box rather than one per voxel (see boxer_env.build_obb_tokens).
        if key in self._obb_cache:
            obb_voxels, obb_volumes, obb_classes, obb_mask = self._obb_cache[key]
        else:
            obb_voxels, obb_volumes, obb_classes, obb_mask = self.boxer.build_obb_tokens(
                video_id, obs_start, obs_end, self._t0[video_id],
                max_tokens=self.max_obb_tokens,
                box_transform=self._obb_box_transform(datum['participant_id']))

        # --- 6. Gaze tokens [obs, 3] + mask ---
        # One voxel per second: where the actor was looking during that second
        # (see gaze.py). Same shape and end-aligned padding as person_locations,
        # so token i of each refers to the same second. Always built, like
        # trajectory tokens, so the modality can be switched on at the model
        # without touching the loader.
        if key in self._gaze_cache:
            gaze_voxels, gaze_mask = self._gaze_cache[key]
        else:
            gaze_voxels, gaze_mask = self._build_gaze(
                datum['participant_id'], video_id, obs_start, obs_end)

        if debug:
            env_occupied = ((env_grid > 0).any(axis=-1).sum() if self.env_is_multi
                            else (env_grid > 0).sum())
            print(f"[DEBUG s0] video={video_id} obs=[{obs_start},{obs_end}) "
                  f"feat_valid={int(visual_mask.sum())}/{self.observation_time} "
                  f"env_occupied={env_occupied} "
                  f"loc_valid={int(person_location_mask.sum())}/{self.observation_time} "
                  f"gt_voxels={int(gt_voxels.sum())} "
                  f"traj_tokens={int(trajectory_mask.sum())}/{self.max_trajectory_tokens} "
                  f"obb_tokens={int(obb_mask.sum())}/{self.max_obb_tokens} "
                  f"gaze_tokens={int(gaze_mask.sum())}/{self.observation_time}")

        item = {
            'index': index,
            'visual_features': obs_features,
            'visual_mask': visual_mask,
            'env_grid': env_grid,
            'person_locations': person_locations,
            'person_location_mask': person_location_mask,
            'trajectory_tokens': trajectory_tokens,
            'trajectory_mask': trajectory_mask,
            'trajectory_classes': trajectory_classes,
            'obb_voxels': obb_voxels,
            'obb_volumes': obb_volumes,
            'obb_classes': obb_classes,
            'obb_mask': obb_mask,
            'gaze_voxels': gaze_voxels,
            'gaze_mask': gaze_mask,
            'labels': {'interaction': gt_voxels},
        }
        if self.env_is_multi:
            item['env_grid_mask'] = env_grid_mask
        if self.trajectory_simple:
            item['trajectory_start_voxel'] = trajectory_start_voxel
            item['trajectory_end_voxel'] = trajectory_end_voxel
            # Log-spaced bin of the distance travelled inside the window. Emitted
            # unconditionally alongside the endpoints (like every other modality's
            # tensors) so --trajectory_token can be switched at the model without
            # touching the loader; the 'mapper' layout simply ignores it.
            item['trajectory_distance'] = trajectory_distance
        return item


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--observation_time', type=int, default=30)
    ap.add_argument('--anticipation_time', type=float, default=5)
    ap.add_argument('--future_time', type=float, default=180)
    ap.add_argument('--splits', type=str, default='train,val,test')
    ap.add_argument('--env_source', type=str, default='boxer_grid',
                    choices=['boxer_grid', 'detic_grid'])
    ap.add_argument('--voxel_centers_dir', type=str, default='')
    ap.add_argument('--env_orphan_fallback', type=lambda s: s.lower() == 'true', default=True)
    ap.add_argument('--trajectory_simple', type=lambda s: s.lower() == 'true', default=False)
    ap.add_argument('--trajectory_exclude_unfinished', type=lambda s: s.lower() == 'true',
                    default=False)
    ap.add_argument('--env_class_averaging', type=lambda s: s.lower() == 'true', default=False)
    ap.add_argument('--env_max_classes_per_voxel', type=int, default=4)
    ap.add_argument('--env_fallback_averaging', type=str, default='off',
                    choices=['off', 'merge', 'empty_only'])
    ap.add_argument('--gaze_max_hz', type=float, default=0.0)
    ap.add_argument('--gaze_time_offset_s', type=float,
                    default=GAZE_DEFAULT_TIME_OFFSET_S)
    ap.add_argument('--time_anchor', type=str, default='trajectory',
                    choices=['trajectory', 'gaze'])
    args = ap.parse_args()
    for split in args.splits.split(','):
        ds = HDEpicInteractionDataset(
            split, observation_time=args.observation_time,
            anticipation_time=args.anticipation_time, future_time=args.future_time,
            env_source=args.env_source,
            voxel_centers_dir=args.voxel_centers_dir or None,
            env_orphan_fallback=args.env_orphan_fallback,
            trajectory_simple=args.trajectory_simple,
            trajectory_exclude_unfinished=args.trajectory_exclude_unfinished,
            env_class_averaging=args.env_class_averaging,
            env_max_classes_per_voxel=args.env_max_classes_per_voxel,
            env_fallback_averaging=args.env_fallback_averaging,
            gaze_max_hz=args.gaze_max_hz,
            gaze_time_offset_s=args.gaze_time_offset_s,
            time_anchor=args.time_anchor)
        print(f"  {split}: {len(ds)} samples")
        if len(ds) > 0:
            s = ds[0]
            print(f"  visual_features {tuple(s['visual_features'].shape)} "
                  f"env_grid {s['env_grid'].shape} "
                  f"person_locations {tuple(s['person_locations'].shape)} "
                  f"gaze_voxels {s['gaze_voxels'].shape} "
                  f"gt_nonzero {int(s['labels']['interaction'].sum())}")
