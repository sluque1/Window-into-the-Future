"""
Boxer-based environment token construction for HD-EPIC.

Adapted from the EgoExo4D reference (FIction_untouched/boxer_env.py). Creates
16x16x16 semantic voxel grids from Boxer 3D oriented bounding-box detections,
filtered to the observation window of each training sample.

Key differences vs the EgoExo4D version
----------------------------------------
* HD-EPIC Boxer CSVs live at ``boxer_obbs/<video_id>/boxer_3dbbs.csv`` (one dir
  per *video_id*, not per take/participant) and already carry a numeric
  ``sem_id`` (LVIS class id) per row, so no LVIS-name -> id taxonomy lookup is
  needed. The env value written into a voxel is ``sem_id + 1`` (0 = empty).
* Time alignment: Boxer ``time_ns`` is device-clock nanoseconds. The loader
  passes a per-video offset ``t0_s`` (device-clock seconds at video-second 0,
  derived from the SLAM trajectory). Observation seconds are converted to
  device ns via ``(sec + t0_s) * 1e9`` to filter the Boxer rows.
* The per-video voxel grid is computed from the AABB of all Boxer OBB corners
  UNION the SLAM device trajectory (instead of SMPL meshes / Aria trajectory).

Design (unchanged from the reference)
-------------------------------------
* The voxel grid (``voxel_centers``) is computed **once per video** so it is
  stable across every episode of that video.
* Per training sample, only Boxer boxes whose ``time_ns`` falls inside the
  observation window are fused (greedy spatial clustering + confidence-weighted
  averaging) and voxelised into ``env_voxels``.
"""

import logging
import os

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default paths / constants
# ---------------------------------------------------------------------------
# Overridable from the environment; see loader.py's path-constant block for why.
HDEPIC_ROOT = os.environ.get("HDEPIC_ROOT", "/data/video_datasets/HD-EPIC")
BOXER_OBBS_ROOT = os.environ.get("BOXER_OBBS_ROOT", f"{HDEPIC_ROOT}/boxer_obbs")
DETIC_OBB_ROOT = os.environ.get(
    "DETIC_OBB_ROOT",
    f"{HDEPIC_ROOT}/fiction_hd_epic/intermediate/object_bounding_boxes_certainty")

GRID_SIZE = 16
SKIP_OBJECT_NAMES = frozenset({'person', 'hands'})

# OWLv2 text embeddings of the lvisplus vocabulary (1220 names, 512-d, L2-normed),
# written by iou_fusion.build_class_embeddings. Resolved next to this module so a
# snapshot directory picks up its own copy -- submit_train.sh must copy it there
# alongside iou_fusion.py, or "iou" fusion raises at the first window.
CLASS_EMBEDDINGS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                     "class_embeddings.npz")

# Precomputed fused boxes live beside the CSV they were fused from, at
# boxer_obbs/<video_id>/<FUSED_CACHE_NAME>. Written by precompute_fused_obbs.py,
# read when BoxerEnvBuilder is given fused_cache=True.
FUSED_CACHE_NAME = "fused_obbs_iou.pkl"

# Bumped whenever the fusion changes what it emits for identical settings, so
# _fused_for_video's meta check rejects caches built by the older code. The
# recorded settings alone cannot catch that: this kind of change is a code fix,
# not a parameter.
#   1 -- initial IoU fusion
#   2 -- deterministic (alphabetical) tie-break in iou_fusion._fuse_cluster.
#        Version 1 caches picked an arbitrary winner among tied class labels,
#        differently in each interpreter and therefore differently per
#        participant. See the comment at iou_fusion._fuse_cluster.
FUSION_VERSION = 2

# What each bump changed, and how much it mattered -- so a stale-cache warning
# says whether a rebuild is worth it, not merely that versions differ.
FUSION_CHANGELOG = {
    2: ("v2 breaks ties between equally-frequent class labels alphabetically "
        "instead of by set iteration order, which varied per process and so per "
        "participant. Affects ONLY the class id of cross-class merges with tied "
        "member counts: measured 3.86% of OBB tokens (4.8 of 123.9 per window), "
        "and those are the weakest boxes (median 2 detections against 5 "
        "overall). Geometry, position and volume are identical."),
}

# --- OBB-token modality -----------------------------------------------------
# Volume bin edges for the per-OBB size feature (see build_obb_tokens). Fused
# box volumes span more than two orders of magnitude -- measured over
# P01/P03/P04/P07/P08 the median box is 0.015-0.048 m^3 while the p95 is
# 0.38-3.06 m^3 -- so the bins are log-spaced; linear bins would drop ~95% of
# boxes into the first bucket. 15 interior edges -> 16 bins, reported as 1..16
# with 0 reserved for padding, matching the 17-row convention
# location_embedding_x/y/z already uses for voxel indices.
VOLUME_BIN_EDGES = np.logspace(-3, 1, 15)     # m^3, 0.001 .. 10
NUM_VOLUME_BINS = len(VOLUME_BIN_EDGES) + 1   # 16
VOLUME_VOCAB_SIZE = NUM_VOLUME_BINS + 1       # 17 (index 0 = pad)

# Default cap on OBB tokens per observation window, overridable per run with
# --max_obb_tokens / MAX_OBB_TOKENS.
#
# The old default of 512 was measured under 'grid' fusion (train P01-P06 mean
# 238.6, p99 458, max 585). Under the 'iou' fusion every current run uses, the
# same windows are far smaller -- measured over all 28,777 windows in the
# precomputed caches (obs=30s anticip=5s time_anchor=gaze):
#   mean 124.1, median 125, p90 179, p95 194, p99 219, max 276
# so 512 was never reached by any window of any participant.
#
# 250 is the cap for the fixture-augmented pipeline (see fixture_obbs.py). After
# the voxel-grid prune, the fixture-overlap filter and the 30-61 static fixture
# boxes appended per participant, the final per-window count is mean 162.3,
# median 163, p95 233, p99 261, max 317; 250 truncates 546 windows (1.90%),
# losing 0.30 tokens per window averaged over the whole set. Truncation is by
# confidence and fixtures carry prob=1.0 against a maximum Boxer prob of 0.898,
# so only the weakest detections are ever dropped, never a fixture.
MAX_OBB_TOKENS = 250


# =====================================================================
# Helpers
# =====================================================================

def _quat_to_rotmat(qw, qx, qy, qz):
    """Quaternion (w, x, y, z) -> 3x3 rotation matrix (world-from-object)."""
    return np.array([
        [1 - 2*(qy*qy + qz*qz), 2*(qx*qy - qz*qw), 2*(qx*qz + qy*qw)],
        [2*(qx*qy + qz*qw), 1 - 2*(qx*qx + qz*qz), 2*(qy*qz - qx*qw)],
        [2*(qx*qz - qy*qw), 2*(qy*qz + qx*qw), 1 - 2*(qx*qx + qy*qy)],
    ], dtype=np.float64)


def find_closest_voxel(world_pos, voxel_centers):
    """Map a 3D world position to the nearest (i, j, k) grid index."""
    distances = np.linalg.norm(voxel_centers - world_pos, axis=-1)
    return np.unravel_index(np.argmin(distances), distances.shape)


# =====================================================================
# Voxel-grid construction from Boxer OBBs + SLAM trajectory
# =====================================================================

def _obb_aabb_corners(tx, ty, tz, qw, qx, qy, qz, sx, sy, sz):
    """Compute the 8 world-frame corners of an oriented bounding box.

    ``scale_x/y/z`` from the Boxer CSV are treated as half-extents (matching the
    EgoExo4D reference pipeline). Returns (8, 3) array.
    """
    R = _quat_to_rotmat(qw, qx, qy, qz)
    signs = np.array([
        [-1, -1, -1], [-1, -1,  1], [-1,  1, -1], [-1,  1,  1],
        [ 1, -1, -1], [ 1, -1,  1], [ 1,  1, -1], [ 1,  1,  1],
    ], dtype=np.float64)
    local_corners = signs * np.array([sx, sy, sz])   # (8, 3)
    world_corners = local_corners @ R.T + np.array([tx, ty, tz])  # (8, 3)
    return world_corners


def _obb_corners_vectorized(boxer_df):
    """All 8 world-frame corners for every OBB row -> (N*8, 3). Vectorised."""
    t = boxer_df[['tx_world_object', 'ty_world_object', 'tz_world_object']].values.astype(np.float64)
    qw = boxer_df['qw_world_object'].values.astype(np.float64)
    qx = boxer_df['qx_world_object'].values.astype(np.float64)
    qy = boxer_df['qy_world_object'].values.astype(np.float64)
    qz = boxer_df['qz_world_object'].values.astype(np.float64)
    s = boxer_df[['scale_x', 'scale_y', 'scale_z']].values.astype(np.float64)  # (N,3)

    # Rotation matrices (N,3,3), world-from-object.
    R = np.empty((len(boxer_df), 3, 3), dtype=np.float64)
    R[:, 0, 0] = 1 - 2*(qy*qy + qz*qz); R[:, 0, 1] = 2*(qx*qy - qz*qw); R[:, 0, 2] = 2*(qx*qz + qy*qw)
    R[:, 1, 0] = 2*(qx*qy + qz*qw); R[:, 1, 1] = 1 - 2*(qx*qx + qz*qz); R[:, 1, 2] = 2*(qy*qz - qx*qw)
    R[:, 2, 0] = 2*(qx*qz - qy*qw); R[:, 2, 1] = 2*(qy*qz + qx*qw); R[:, 2, 2] = 1 - 2*(qx*qx + qy*qy)

    signs = np.array([
        [-1, -1, -1], [-1, -1, 1], [-1, 1, -1], [-1, 1, 1],
        [1, -1, -1], [1, -1, 1], [1, 1, -1], [1, 1, 1],
    ], dtype=np.float64)  # (8,3)
    local = signs[None, :, :] * s[:, None, :]            # (N,8,3)
    world = np.einsum('nij,nkj->nki', R, local) + t[:, None, :]  # (N,8,3)
    return world.reshape(-1, 3)


def _normalize_traj(trajectory_xyz):
    """Coerce a trajectory argument into an (N, 3) float array (possibly empty)."""
    traj = np.asarray(trajectory_xyz, dtype=np.float64)
    if traj.ndim == 1 and traj.size == 3:
        traj = traj.reshape(1, 3)
    elif traj.size == 0:
        traj = np.empty((0, 3), dtype=np.float64)
    return traj


def _grid_from_points(all_points):
    """Build a 16x16x16 voxel-centre grid covering the AABB of ``all_points``.

    Args:
        all_points: (M, 3) array of world positions defining the spatial extent.

    Returns:
        (16, 16, 16, 3) array of voxel centres.
    """
    if all_points.shape[0] == 0:
        raise ValueError("Cannot build voxel grid: no points to bound")

    min_overall = all_points.min(axis=0)
    max_overall = all_points.max(axis=0)
    # Guard against a degenerate (zero-extent) axis.
    extent = np.maximum(max_overall - min_overall, 1e-3)
    max_overall = min_overall + extent

    voxel_size = extent / GRID_SIZE
    axes = [
        np.linspace(min_overall[d] + voxel_size[d] / 2,
                    max_overall[d] - voxel_size[d] / 2, GRID_SIZE)
        for d in range(3)
    ]
    x_grid, y_grid, z_grid = np.meshgrid(axes[0], axes[1], axes[2], indexing='ij')
    return np.stack([x_grid, y_grid, z_grid], axis=-1)


def compute_voxel_centers_from_boxer(boxer_df, trajectory_xyz):
    """Compute a 16x16x16 voxel-centre grid from Boxer OBBs + a trajectory.

    The spatial extent is the union AABB of all Boxer bounding-box corners and
    the supplied trajectory positions (the per-video SLAM device trajectory).

    Args:
        boxer_df: DataFrame (full-video Boxer CSV) or None.
        trajectory_xyz: (N, 3) array of world positions, or empty.

    Returns:
        (16, 16, 16, 3) array of voxel centres.
    """
    if boxer_df is not None and len(boxer_df) > 0:
        all_corners = _obb_corners_vectorized(boxer_df)
    else:
        all_corners = np.empty((0, 3), dtype=np.float64)

    traj = _normalize_traj(trajectory_xyz)
    all_points = np.concatenate([all_corners, traj], axis=0)
    return _grid_from_points(all_points)


def compute_voxel_centers_from_detic(detic_corners, trajectory_xyz):
    """Compute a 16x16x16 voxel-centre grid from Detic OBBs + a trajectory.

    The spatial extent is the union AABB of all Detic bounding-box corners and
    the supplied trajectory positions (the per-video SLAM device trajectory).
    The environment *content* still comes from Boxer; only the grid extent is
    defined by Detic here.

    Args:
        detic_corners: (M, 3) array of Detic OBB corner world positions, or empty.
        trajectory_xyz: (N, 3) array of world positions, or empty.

    Returns:
        (16, 16, 16, 3) array of voxel centres.
    """
    corners = np.asarray(detic_corners, dtype=np.float64)
    if corners.size == 0:
        corners = np.empty((0, 3), dtype=np.float64)
    traj = _normalize_traj(trajectory_xyz)
    all_points = np.concatenate([corners, traj], axis=0)
    return _grid_from_points(all_points)


def load_detic_obb_corners(video_id):
    """Load the static Detic 3D OBB corners for a video as an (M*8, 3) array.

    Reads ``<DETIC_OBB_ROOT>/<pid>/<video_id>_object_bbs_obb_newparams.pkl``,
    whose ``rrc`` field has shape (M, 3, 8) = M objects x (x,y,z) x 8 corners.
    Returns None if the file is absent.
    """
    pid = video_id.split('-')[0]
    path = os.path.join(DETIC_OBB_ROOT, pid,
                        f"{video_id}_object_bbs_obb_newparams.pkl")
    if not os.path.exists(path):
        return None
    # These pkls were written under a newer numpy; shim numpy._core -> numpy.core.
    import pickle
    import sys
    sys.modules.setdefault('numpy._core', np.core)
    for sub in ['.multiarray', '.numeric', '._multiarray_umath',
                '._exceptions', '.overrides']:
        try:
            sys.modules.setdefault('numpy._core' + sub,
                                   __import__('numpy.core' + sub, fromlist=['_']))
        except Exception:
            pass
    with open(path, 'rb') as fh:
        data = pickle.load(fh)
    rrc = np.asarray(data['rrc'], dtype=np.float64)   # (M, 3, 8)
    if rrc.size == 0:
        return np.empty((0, 3), dtype=np.float64)
    # (M, 3, 8) -> (M, 8, 3) -> (M*8, 3)
    return np.moveaxis(rrc, 1, 2).reshape(-1, 3)


# =====================================================================
# Lightweight 3D-box fusion
# =====================================================================

def _fuse_boxes(df, conf_threshold=0.5, min_detections=2, cluster_distance=0.5):
    """Fuse Boxer CSV rows into a list of static box dicts.

    Detections are filtered by confidence, then clustered by snapping each box
    centre to a ``cluster_distance``-resolution grid and grouping by
    ``(name, snapped cell)``. Within each cluster, position/scale are
    confidence-weighted averaged and the highest-confidence quaternion + sem_id
    are kept. This is a vectorised replacement for the original greedy O(n^2)
    single-linkage clustering, which is far too slow for the dense HD-EPIC
    Boxer CSVs (~10k detections in a 30s window).
    """
    df = df[(df['prob'] >= conf_threshold) & (~df['name'].isin(SKIP_OBJECT_NAMES))]
    if len(df) == 0:
        return []

    pos_cols = ['tx_world_object', 'ty_world_object', 'tz_world_object']
    scale_cols = ['scale_x', 'scale_y', 'scale_z']
    pos = df[pos_cols].values.astype(np.float64)
    scale = df[scale_cols].values.astype(np.float64)
    w = df['prob'].values.astype(np.float64)
    cell = np.floor(pos / cluster_distance).astype(np.int64)

    # Vectorised confidence-weighted aggregation per (name, snapped cell).
    work = pd.DataFrame({
        'name': df['name'].values,
        'cx': cell[:, 0], 'cy': cell[:, 1], 'cz': cell[:, 2],
        'w': w,
        'wx': w * pos[:, 0], 'wy': w * pos[:, 1], 'wz': w * pos[:, 2],
        'wsx': w * scale[:, 0], 'wsy': w * scale[:, 1], 'wsz': w * scale[:, 2],
    })
    g = work.groupby(['name', 'cx', 'cy', 'cz'], sort=False)
    agg = g[['w', 'wx', 'wy', 'wz', 'wsx', 'wsy', 'wsz']].sum()
    cnt = g.size()
    keep = cnt.values >= min_detections
    if not keep.any():
        return []
    agg = agg[keep]
    w_sum = agg['w'].values
    centers = np.stack([agg['wx'].values, agg['wy'].values, agg['wz'].values], axis=1) / w_sum[:, None]
    halves = np.stack([agg['wsx'].values, agg['wsy'].values, agg['wsz'].values], axis=1) / w_sum[:, None]
    prob_mean = w_sum / cnt.values[keep]

    # Highest-confidence row per kept group supplies the quaternion + sem_id.
    work2 = df.assign(_cx=cell[:, 0], _cy=cell[:, 1], _cz=cell[:, 2])
    best = work2.loc[work2.groupby(['name', '_cx', '_cy', '_cz'], sort=False)['prob'].idxmax()]
    best = best[keep]
    quats = best[['qw_world_object', 'qx_world_object',
                  'qy_world_object', 'qz_world_object']].values.astype(np.float64)
    sem_ids = best['sem_id'].values.astype(np.int64) + 1   # +1 reserves 0 for empty

    fused = []
    for i in range(len(centers)):
        fused.append({
            'center': centers[i],
            'quat': quats[i],
            'half_extents': halves[i],
            'sem_id': int(sem_ids[i]),
            'prob': float(prob_mean[i]),
        })
    return fused


def _voxelize_fused_boxes(fused_boxes, voxel_centers, orphan_fallback=True):
    """Place fused boxes into a 16x16x16 semantic grid (highest conf wins).

    Two passes when ``orphan_fallback`` is True:
      1. Primary: mark every voxel whose centre falls inside a box's OBB
         (correctly covers objects that span multiple voxels).
      2. Fallback: a box smaller than a voxel, or off-centre within one, can
         contain no voxel centre and so is invisible to the primary pass. For
         each such orphaned box, assign it to its single nearest voxel --
         but only if that voxel is still empty after the primary pass, so
         nothing the primary pass marked is ever overwritten. Conflicts
         between fallback boxes sharing a nearest voxel are resolved by the
         same highest-confidence rule as the primary pass.

    When ``orphan_fallback`` is False (the traditional method), orphaned
    boxes are simply dropped -- only the primary pass runs.
    """
    grid = np.zeros((GRID_SIZE, GRID_SIZE, GRID_SIZE), dtype=np.float64)
    conf = np.zeros((GRID_SIZE, GRID_SIZE, GRID_SIZE), dtype=np.float64)
    flat_centers = voxel_centers.reshape(-1, 3)  # (4096, 3)

    orphans = []
    for box in fused_boxes:
        sem_id = box['sem_id']
        if sem_id <= 0:
            continue
        R = _quat_to_rotmat(*box['quat'])
        d = flat_centers - box['center']
        p_local = d @ R                           # R^T @ d
        inside = np.all(np.abs(p_local) <= box['half_extents'], axis=1)
        inside_3d = inside.reshape(GRID_SIZE, GRID_SIZE, GRID_SIZE)
        if not inside_3d.any():
            if orphan_fallback:
                orphans.append(box)
            continue
        update = inside_3d & (box['prob'] > conf)
        grid[update] = sem_id
        conf[update] = box['prob']

    for box in orphans:
        i, j, k = find_closest_voxel(box['center'], voxel_centers)
        if grid[i, j, k] != 0:
            continue   # already marked by the primary pass -- leave it as is
        if box['prob'] > conf[i, j, k]:
            grid[i, j, k] = box['sem_id']
            conf[i, j, k] = box['prob']
    return grid


def _voxelize_fused_boxes_multi(fused_boxes, voxel_centers, k_max=4, orphan_fallback=True):
    """Like ``_voxelize_fused_boxes``, but keeps every class overlapping a
    voxel (up to ``k_max``) instead of only the highest-confidence winner.

    For each voxel, every box whose OBB contains the voxel's centre is a
    "hit"; hits are sorted by confidence and the top ``k_max`` are kept
    (ties/overflow broken by confidence, same rule as the single-class
    path). Voxels with no hits stay all-zero/all-empty -- callers should
    treat those as "no object" (the model embeds them via class id 0), not
    average them with anything.

    Returns:
      grid_multi: (16,16,16,k_max) int64 -- sem_id+1 per slot, 0 = pad slot.
      mask:       (16,16,16,k_max) bool  -- True where the slot holds a real hit.
    """
    flat_centers = voxel_centers.reshape(-1, 3)  # (4096, 3)
    hits = {}   # (i,j,k) -> list of (prob, sem_id)

    def _add_hit(i, j, k, prob, sem_id):
        hits.setdefault((i, j, k), []).append((prob, sem_id))

    for box in fused_boxes:
        sem_id = box['sem_id']
        if sem_id <= 0:
            continue
        R = _quat_to_rotmat(*box['quat'])
        d = flat_centers - box['center']
        p_local = d @ R
        inside = np.all(np.abs(p_local) <= box['half_extents'], axis=1)
        inside_3d = inside.reshape(GRID_SIZE, GRID_SIZE, GRID_SIZE)
        idxs = np.argwhere(inside_3d)
        if idxs.size == 0:
            if orphan_fallback:
                i, j, k = find_closest_voxel(box['center'], voxel_centers)
                _add_hit(i, j, k, box['prob'], sem_id)
            continue
        for i, j, k in idxs:
            _add_hit(int(i), int(j), int(k), box['prob'], sem_id)

    grid_multi = np.zeros((GRID_SIZE, GRID_SIZE, GRID_SIZE, k_max), dtype=np.int64)
    mask = np.zeros((GRID_SIZE, GRID_SIZE, GRID_SIZE, k_max), dtype=bool)
    for (i, j, k), cell_hits in hits.items():
        cell_hits.sort(key=lambda h: -h[0])
        for slot, (prob, sem_id) in enumerate(cell_hits[:k_max]):
            grid_multi[i, j, k, slot] = sem_id
            mask[i, j, k, slot] = True
    return grid_multi, mask


def _voxelize_fused_boxes_fallback_avg(fused_boxes, voxel_centers, k_max=4, mode='merge'):
    """Single-winner primary pass, then class AVERAGING applied only to the
    boxes that pass missed.

    Stage 1 is exactly ``_voxelize_fused_boxes`` with ``orphan_fallback=False``:
    every voxel whose centre falls inside a box's OBB takes that box's class,
    highest confidence winning; voxels no OBB covers stay empty.

    Stage 2 handles the "leftover" boxes -- those too small or too off-centre to
    contain ANY voxel centre, so stage 1 never saw them. Each is assigned to the
    single voxel nearest its centre (the same rule ``_voxelize_fused_boxes``'s
    orphan fallback uses), but its class is kept ALONGSIDE whatever is already
    there instead of being dropped or overwriting. Two mutually exclusive
    policies:

      'merge'
          The leftover's class joins the voxel's slot list whether or not stage
          1 filled it. A voxel won by 'cupboard' whose nearest leftover box is a
          'cup' ends up averaging both class embeddings.
      'empty_only'
          The leftover's class is kept only where stage 1 left the voxel empty;
          filled voxels keep their single winner untouched. Several leftovers
          landing on the same empty voxel are averaged together.

          NOTE: measured on P04/P08 (24 windows, ~5k fused boxes), 98-100% of
          leftover boxes land on an ALREADY-FILLED voxel, so this policy is
          expected to reproduce the single-winner grid almost exactly. The
          returned ``stats`` reports the split per run so that is visible in the
          logs rather than assumed.

    Slot order per voxel is the stage-1 winner first (the primary, highest-
    confidence signal), then leftovers by descending confidence, truncated to
    ``k_max``. Averaging itself happens in model.py, which consumes exactly the
    (grid, mask) pair ``_voxelize_fused_boxes_multi`` produces.

    Returns:
      grid_multi: (16,16,16,k_max) int64 -- sem_id+1 per slot, 0 = pad slot.
      mask:       (16,16,16,k_max) bool  -- True where the slot holds a real class.
      stats:      dict -- n_boxes, n_leftover, n_leftover_on_empty,
                  n_leftover_on_filled, n_leftover_dropped.
    """
    if mode not in ('merge', 'empty_only'):
        raise ValueError(f"Unknown fallback-averaging mode: {mode!r} "
                         "(expected 'merge' or 'empty_only')")

    grid = np.zeros((GRID_SIZE, GRID_SIZE, GRID_SIZE), dtype=np.int64)
    conf = np.zeros((GRID_SIZE, GRID_SIZE, GRID_SIZE), dtype=np.float64)
    flat_centers = voxel_centers.reshape(-1, 3)   # (4096, 3)

    # -- Stage 1: single winner per voxel, leftovers set aside ----------------
    leftovers = []
    for box in fused_boxes:
        sem_id = box['sem_id']
        if sem_id <= 0:
            continue
        R = _quat_to_rotmat(*box['quat'])
        p_local = (flat_centers - box['center']) @ R      # R^T @ d
        inside = np.all(np.abs(p_local) <= box['half_extents'], axis=1)
        inside_3d = inside.reshape(GRID_SIZE, GRID_SIZE, GRID_SIZE)
        if not inside_3d.any():
            leftovers.append(box)
            continue
        update = inside_3d & (box['prob'] > conf)
        grid[update] = sem_id
        conf[update] = box['prob']

    # -- Stage 2: leftovers contribute an extra class to their nearest voxel --
    extra = {}   # (i,j,k) -> list of (prob, sem_id)
    stats = dict(n_boxes=len(fused_boxes), n_leftover=len(leftovers),
                 n_leftover_on_empty=0, n_leftover_on_filled=0,
                 n_leftover_dropped=0)
    for box in leftovers:
        i, j, k = find_closest_voxel(box['center'], voxel_centers)
        occupied = grid[i, j, k] != 0
        stats['n_leftover_on_filled' if occupied else 'n_leftover_on_empty'] += 1
        if occupied and mode == 'empty_only':
            stats['n_leftover_dropped'] += 1
            continue
        extra.setdefault((i, j, k), []).append((box['prob'], box['sem_id']))

    # -- Assemble the multi-class grid ---------------------------------------
    grid_multi = np.zeros((GRID_SIZE, GRID_SIZE, GRID_SIZE, k_max), dtype=np.int64)
    mask = np.zeros((GRID_SIZE, GRID_SIZE, GRID_SIZE, k_max), dtype=bool)
    occupied_idx = np.argwhere(grid != 0)
    for i, j, k in occupied_idx:
        grid_multi[i, j, k, 0] = grid[i, j, k]
        mask[i, j, k, 0] = True
    for (i, j, k), hits in extra.items():
        slot = 1 if grid[i, j, k] != 0 else 0
        for prob, sem_id in sorted(hits, key=lambda h: -h[0]):
            if slot >= k_max:
                break
            grid_multi[i, j, k, slot] = sem_id
            mask[i, j, k, slot] = True
            slot += 1
    return grid_multi, mask, stats


# =====================================================================
# OBB tokens (one token per fused box, instead of one per voxel)
# =====================================================================

def _nearest_voxel_ijk(points, voxel_centers):
    """Nearest grid index for each of ``points``. Vectorised over all points.

    Args:
        points: (N, 3) world positions.
        voxel_centers: (16, 16, 16, 3) grid centres.

    Returns:
        (N, 3) int64 array of (i, j, k), each 0..15.
    """
    flat = voxel_centers.reshape(-1, 3)                       # (4096, 3)
    d2 = ((points[:, None, :] - flat[None, :, :]) ** 2).sum(-1)   # (N, 4096)
    flat_idx = d2.argmin(axis=1)                              # (N,)
    return np.stack(np.unravel_index(flat_idx, (GRID_SIZE,) * 3), axis=-1)


def _obb_tokens_from_fused(fused_boxes, voxel_centers, max_tokens=MAX_OBB_TOKENS):
    """Turn fused boxes into fixed-width per-OBB token arrays.

    One token per fused box, carrying the three pieces of information a box
    has: WHERE it is (the grid index of its centre, so a position means the
    same learned thing here as it does for person_locations / gaze /
    trajectory endpoints), HOW BIG it is (log-spaced volume bin), and WHAT it
    is (Boxer sem_id+1, the same vocabulary the env voxel grid uses).

    The centre is snapped to its containing voxel rather than kept as a raw
    world coordinate deliberately: each recording has its own Aria world frame
    with an arbitrary origin (measured spans differ by metres between
    participants), and the splits are participant-level, so literal world xyz
    would not transfer to an unseen kitchen. Grid indices are comparable
    across kitchens by construction.

    When more than ``max_tokens`` boxes are present the highest-confidence
    ones are kept -- the same rule the voxeliser uses to resolve conflicts.

    Returns:
        voxels:  (max_tokens, 3) int64 -- 1..16 per axis, 0 = padding slot.
        volumes: (max_tokens,) int64   -- 1..16 volume bin, 0 = padding slot.
        classes: (max_tokens,) int64   -- sem_id+1, 0 = padding slot.
        mask:    (max_tokens,) bool    -- True where the slot is a real box.
    """
    voxels = np.zeros((max_tokens, 3), dtype=np.int64)
    volumes = np.zeros(max_tokens, dtype=np.int64)
    classes = np.zeros(max_tokens, dtype=np.int64)
    mask = np.zeros(max_tokens, dtype=bool)

    boxes = [b for b in (fused_boxes or []) if b['sem_id'] > 0]
    if not boxes:
        return voxels, volumes, classes, mask

    # Highest confidence first, then truncate -- see the docstring.
    boxes.sort(key=lambda b: -b['prob'])
    boxes = boxes[:max_tokens]
    n = len(boxes)

    centers = np.stack([b['center'] for b in boxes])            # (n, 3)
    halves = np.stack([b['half_extents'] for b in boxes])       # (n, 3)

    # scale_x/y/z are half-extents (see _obb_aabb_corners), so the full-box
    # volume is (2*sx)(2*sy)(2*sz) = 8*sx*sy*sz.
    vol = 8.0 * np.prod(np.abs(halves), axis=1)                 # (n,) m^3

    voxels[:n] = _nearest_voxel_ijk(centers, voxel_centers) + 1   # 0 stays pad
    volumes[:n] = np.digitize(vol, VOLUME_BIN_EDGES) + 1          # 1..16
    classes[:n] = [b['sem_id'] for b in boxes]
    mask[:n] = True
    return voxels, volumes, classes, mask


# =====================================================================
# Main builder class
# =====================================================================

class BoxerEnvBuilder:
    """Builds per-sample 16x16x16 semantic voxel grids from Boxer OBBs.

    Caches per-video Boxer CSVs and voxel centres. The voxel grid is computed
    once per video (Boxer corners UNION SLAM trajectory) and shared across all
    of that video's episodes.
    """

    def __init__(self, boxer_root=BOXER_OBBS_ROOT, min_detections=2,
                 conf_threshold=0.6, cluster_distance=0.5, boxer_fps=10.0,
                 orphan_fallback=True, debug=False,
                 fusion="iou", iou_threshold=0.4, semantic_threshold=0.6,
                 class_embeddings=CLASS_EMBEDDINGS_PATH, nms_iou=None,
                 legacy_half=False, fused_cache=None):
        self.boxer_root = boxer_root
        self.min_detections = min_detections
        self.conf_threshold = conf_threshold
        self.cluster_distance = cluster_distance
        # -- clustering strategy -------------------------------------------
        # "iou" (default): iou_fusion.fuse_boxes_iou -- link detections that
        #   actually overlap in 3D (>= iou_threshold) and whose class names are
        #   close in OWLv2 embedding space (>= semantic_threshold), then chain
        #   the links. ~600x slower per window than "grid", so a dataloader
        #   wants a precomputed fused_cache (see precompute_fused_windows).
        # "grid": the original _fuse_boxes -- snap centres to a cluster_distance
        #   cell and group by (name, cell). Fast, but splits one object whose
        #   centre jitters across a cell boundary and merges distinct objects
        #   sharing a cell. This is what every run before 2026-08-26 used.
        # cluster_distance is unused when fusion == "iou".
        self.fusion = fusion
        self.iou_threshold = iou_threshold
        self.semantic_threshold = semantic_threshold
        # Path to class_embeddings.npz (name -> unit vector), or an already-loaded
        # dict. Enabled by default so synonyms merge (cabinet/cupboard 0.926,
        # mug/cup 0.756, chair/armchair 0.615, against stove/sink 0.386); set to
        # None to require identical class names instead. Loaded once, lazily --
        # see _get_class_embeddings.
        self.class_embeddings = class_embeddings
        self._class_embeddings_cache = None
        # Post-fusion NMS. Inert whenever nms_iou > iou_threshold, since those
        # pairs were already merged during clustering.
        self.nms_iou = nms_iou
        # scale_x/y/z in the CSV is the box's FULL size (bb3_max - bb3_min), but
        # _fuse_boxes passes it through as a half-extent, so every box comes out
        # twice its true size per axis. False (default) uses the correct
        # convention; True reproduces the legacy oversized boxes. Flipping this
        # changes env occupancy and shifts OBB-token volume bins by ~3.4 bins, so
        # it is not comparable with a run made under the other setting.
        self.legacy_half = legacy_half
        # Precomputed fusion results, consulted before doing any work. Either an
        # explicit {(video_id, obs_start, obs_end): fused} dict, or True to read
        # the per-video pickles precompute_fused_obbs.py writes alongside each
        # video's CSV (boxer_obbs/<video_id>/FUSED_CACHE_NAME). The per-video
        # files are loaded one at a time on first use, so a split only holds its
        # own videos in memory -- the full set is ~1.0 GB.
        self.use_fused_files = fused_cache is True
        self.fused_cache = None if fused_cache is True else fused_cache
        self._fused_by_video = {}       # video_id -> {key: fused} or None
        self._warned_stale_fusion = False   # see _warn_stale_fusion
        # Whether boxes too small/off-centre to contain any voxel centre get
        # assigned to their nearest voxel (see _voxelize_fused_boxes) or are
        # simply dropped (the traditional method).
        self.orphan_fallback = orphan_fallback
        # Temporal downsampling: Boxer emits ~30 OBB detections/sec, so a full
        # video is hundreds of thousands of rows. We keep at most one timestamp
        # every (1 / boxer_fps) seconds (all object rows at the kept timestamps),
        # which thins per-window fusion input ~3x at 10 Hz while preserving every
        # persistent object. Set boxer_fps<=0 to disable downsampling.
        self.boxer_fps = boxer_fps
        self.debug = debug

        self._boxer_csv_cache = {}
        self._voxel_centers_cache = {}
        # Accumulated leftover-box placement counts from build_env_fallback_avg,
        # so a run can report how often the fallback actually changed anything.
        self.fallback_stats = {}

    # -- per-video caches ----------------------------------------------------

    def csv_path(self, video_id):
        return os.path.join(self.boxer_root, video_id, "boxer_3dbbs.csv")

    def fused_path(self, video_id):
        """Where this video's precomputed fused boxes live -- a sibling of its
        source CSV, so the derived file travels with the data it came from."""
        return os.path.join(self.boxer_root, video_id, FUSED_CACHE_NAME)

    def has_fused_cache(self, video_id):
        return os.path.exists(self.fused_path(video_id))

    def has_fused_window(self, video_id, obs_start_s, obs_end_s):
        """Whether this window is already fused, by the same lookup
        ``_window_fused_boxes`` uses. Loading a cache here also surfaces a
        settings mismatch at startup rather than mid-run."""
        key = (video_id, round(float(obs_start_s), 3), round(float(obs_end_s), 3))
        if self.fused_cache is not None and key in self.fused_cache:
            return True
        if self.use_fused_files:
            per_video = self._fused_for_video(video_id)
            return per_video is not None and key in per_video
        return False

    def _fused_for_video(self, video_id):
        """This video's {key: fused} mapping from disk, loaded at most once.

        The pickle carries the settings it was built under; they are checked
        against this builder's, because the window key alone cannot detect a
        mismatch. t0 in particular (set by time_anchor) decides which CSV rows a
        window covers, so a cache built under a different anchor has the SAME
        keys and DIFFERENT boxes -- silently wrong rather than missing.
        """
        if video_id in self._fused_by_video:
            return self._fused_by_video[video_id]

        path = self.fused_path(video_id)
        if not os.path.exists(path):
            self._fused_by_video[video_id] = None
            return None

        import pickle
        with open(path, 'rb') as fh:
            blob = pickle.load(fh)
        meta = blob.get('meta', {})
        # A key absent from meta is ignored (older caches predate it) EXCEPT
        # fusion_version: its absence is itself the signal that this file came
        # from version 1, so it must default rather than be skipped.
        live = self.fusion_settings()
        mismatch = {k: (meta.get(k), v) for k, v in live.items()
                    if k in meta and meta[k] != v}
        # Version handling is deliberately asymmetric.
        #   cache NEWER than the code -> hard error. The code cannot know what a
        #     future version emits, so this means the checkout is behind.
        #   cache OLDER than the code -> accepted, with a loud warning naming
        #     what differs. Rebuilding is correct but not always worth it: see
        #     FUSION_CHANGELOG for the measured impact of each bump.
        cached_version = meta.get('fusion_version', 1)
        live_version = live['fusion_version']
        if cached_version > live_version:
            mismatch['fusion_version'] = (cached_version, live_version)
        elif cached_version < live_version:
            self._warn_stale_fusion(cached_version, live_version)
        if mismatch:
            raise ValueError(
                f"{path} was built under settings this run does not match: "
                + "; ".join(f"{k}: cache={c!r} run={r!r}"
                            for k, (c, r) in sorted(mismatch.items()))
                + ". Rebuild it with precompute_fused_obbs.py, or change the run "
                  "to match. (t0 depends on time_anchor, so a mismatched cache "
                  "returns boxes from the wrong part of the video.)")
        self._fused_by_video[video_id] = blob.get('windows', {})
        return self._fused_by_video[video_id]

    def _warn_stale_fusion(self, cached_version, live_version):
        """Announce, once per builder, that the cache predates the fusion code.

        Once rather than per video: a split holds up to 152 of them. Printed as
        well as logged so it lands in the job's .out and the run is on record as
        having used older boxes.
        """
        if self._warned_stale_fusion:
            return
        self._warned_stale_fusion = True
        notes = [FUSION_CHANGELOG[v] for v in range(cached_version + 1, live_version + 1)
                 if v in FUSION_CHANGELOG]
        # print, not logger: the loader reports its startup summary the same way,
        # and this must land in the job's .out beside it. Emitting both duplicates
        # it in every log.
        print(f"[boxer_env] WARNING: fused cache is version {cached_version}, code "
              f"is version {live_version} -- USING THE CACHED BOXES AS THEY ARE.",
              flush=True)
        for n in notes:
            print(f"[boxer_env]   {n}", flush=True)
        print("[boxer_env]   Rebuild with "
              "`OVERWRITE=true ./precompute_fused_obbs_sbatch.sh` to change this.",
              flush=True)

    def fusion_settings(self):
        """The settings that determine what a fused box contains, recorded in
        each cache file and checked on load."""
        return dict(fusion_version=FUSION_VERSION,
                    fusion=self.fusion,
                    conf_threshold=self.conf_threshold,
                    min_detections=self.min_detections,
                    boxer_fps=self.boxer_fps,
                    cluster_distance=(self.cluster_distance
                                      if self.fusion == 'grid' else None),
                    iou_threshold=(self.iou_threshold
                                   if self.fusion == 'iou' else None),
                    semantic_threshold=(self.semantic_threshold
                                        if self.fusion == 'iou' else None),
                    nms_iou=(self.nms_iou if self.fusion == 'iou' else None),
                    legacy_half=(self.legacy_half
                                 if self.fusion == 'iou' else None),
                    class_embeddings=(os.path.basename(str(self.class_embeddings))
                                      if self.fusion == 'iou' and self.class_embeddings
                                      else None))

    def has_boxer_data(self, video_id):
        return os.path.exists(self.csv_path(video_id))

    def load_boxer_csv(self, video_id):
        """Load and cache the full-video Boxer CSV (or None)."""
        if video_id not in self._boxer_csv_cache:
            path = self.csv_path(video_id)
            if not os.path.exists(path):
                self._boxer_csv_cache[video_id] = None
            else:
                try:
                    df = pd.read_csv(path)
                    if df.empty or 'time_ns' not in df.columns:
                        self._boxer_csv_cache[video_id] = None
                    else:
                        df.sort_values('time_ns', inplace=True)
                        df.reset_index(drop=True, inplace=True)
                        df = self._downsample_time(df)
                        self._boxer_csv_cache[video_id] = df
                        if self.debug:
                            logger.info("[%s] boxer rows after %.0fHz downsample: %d",
                                        video_id, self.boxer_fps, len(df))
                except Exception as exc:
                    logger.warning("Failed to read Boxer CSV for %s: %s", video_id, exc)
                    self._boxer_csv_cache[video_id] = None
        return self._boxer_csv_cache[video_id]

    def _downsample_time(self, df):
        """Keep all object rows only at timestamps spaced >= 1/boxer_fps apart.

        Boxer detections at one frame share a ``time_ns``. We keep a subset of
        the unique timestamps (every ~1/boxer_fps seconds) and all rows at those
        timestamps, so each kept frame retains its full set of objects.
        """
        if self.boxer_fps is None or self.boxer_fps <= 0:
            return df
        times = df['time_ns'].values
        uniq = np.unique(times)                       # sorted unique timestamps
        step_ns = 1e9 / self.boxer_fps
        kept = [uniq[0]]
        last = uniq[0]
        for t in uniq[1:]:
            if t - last >= step_ns:
                kept.append(t)
                last = t
        kept = np.asarray(kept)
        out = df[np.isin(times, kept)].reset_index(drop=True)
        return out

    def compute_and_cache_voxel_centers(self, video_id, trajectory_xyz):
        """Compute (once) the per-video voxel grid from Boxer OBBs + trajectory.

        Returns the (16,16,16,3) centres, or None if neither Boxer nor a
        trajectory is available.
        """
        if video_id in self._voxel_centers_cache:
            return self._voxel_centers_cache[video_id]

        boxer_df = self.load_boxer_csv(video_id)
        traj = np.asarray(trajectory_xyz, dtype=np.float64) if trajectory_xyz is not None \
            else np.empty((0, 3))
        if (boxer_df is None or len(boxer_df) == 0) and traj.size == 0:
            self._voxel_centers_cache[video_id] = None
            return None

        vc = compute_voxel_centers_from_boxer(boxer_df, traj)
        self._voxel_centers_cache[video_id] = vc
        return vc

    def has_detic_data(self, video_id):
        pid = video_id.split('-')[0]
        path = os.path.join(DETIC_OBB_ROOT, pid,
                            f"{video_id}_object_bbs_obb_newparams.pkl")
        return os.path.exists(path)

    def compute_and_cache_detic_voxel_centers(self, video_id, trajectory_xyz):
        """Compute (once) the per-video voxel grid from Detic OBBs + trajectory.

        The environment *content* still comes from Boxer (``build_env`` reads the
        Boxer CSV); only the grid *extent* is defined by the Detic OBBs here.
        Returns the (16,16,16,3) centres, or None if neither Detic OBBs nor a
        trajectory is available.
        """
        if video_id in self._voxel_centers_cache:
            return self._voxel_centers_cache[video_id]

        detic_corners = load_detic_obb_corners(video_id)
        traj = np.asarray(trajectory_xyz, dtype=np.float64) if trajectory_xyz is not None \
            else np.empty((0, 3))
        no_detic = detic_corners is None or len(detic_corners) == 0
        if no_detic and traj.size == 0:
            self._voxel_centers_cache[video_id] = None
            return None

        vc = compute_voxel_centers_from_detic(detic_corners, traj)
        self._voxel_centers_cache[video_id] = vc
        return vc

    def set_voxel_centers(self, video_id, voxel_centers):
        """Override the per-video voxel grid with externally supplied centres.

        Used when the grid is defined by a fixed, pre-computed per-participant
        grid rather than derived from Boxer/Detic OBBs. Boxer content is still
        voxelised into these centres by ``build_env``.
        """
        self._voxel_centers_cache[video_id] = np.asarray(voxel_centers, dtype=np.float64)

    def get_voxel_centers(self, video_id):
        return self._voxel_centers_cache.get(video_id)

    # -- per-sample env construction -----------------------------------------

    def _window_fused_boxes(self, video_id, obs_start_s, obs_end_s, t0_s):
        """Fused boxes (see ``_fuse_boxes``) for one observation window, or
        None if there is no Boxer data / voxel grid / boxes in the window.
        Shared by ``build_env`` and ``build_env_multi`` so both voxelization
        strategies see identical fused input.
        """
        voxel_centers = self.get_voxel_centers(video_id)
        if voxel_centers is None:
            return None, voxel_centers

        # Consulted BEFORE load_boxer_csv: a cache hit needs no CSV at all, and
        # those are 9-64MB each to read and parse. Checking after would make a
        # fully-cached run still parse every video. Keys are rounded so float
        # noise in obs_start cannot miss one.
        key = (video_id, round(float(obs_start_s), 3), round(float(obs_end_s), 3))
        if self.fused_cache is not None and key in self.fused_cache:
            return (self.fused_cache[key] or None), voxel_centers
        if self.use_fused_files:
            per_video = self._fused_for_video(video_id)
            if per_video is not None and key in per_video:
                return (per_video[key] or None), voxel_centers

        boxer_df = self.load_boxer_csv(video_id)
        if boxer_df is None:
            return None, voxel_centers

        vrs_start = (obs_start_s + t0_s) * 1e9
        vrs_end = (obs_end_s + t0_s) * 1e9
        mask = (boxer_df['time_ns'] >= vrs_start) & (boxer_df['time_ns'] <= vrs_end)
        window_df = boxer_df.loc[mask]
        if self.debug:
            logger.info("[%s] obs=[%.1f,%.1f]s -> %d/%d boxes in window",
                        video_id, obs_start_s, obs_end_s, len(window_df), len(boxer_df))
        if len(window_df) == 0:
            return None, voxel_centers

        fused = self._fuse(window_df)
        return (fused or None), voxel_centers

    def precompute_fused_windows(self, windows, t0_by_video, out_path=None,
                                 progress_every=200):
        """Fuse a list of windows up front and return the {key: fused} mapping.

        IoU fusion costs ~5-8 s per 30 s window against ~0.01 s for grid
        snapping, which is fine once and ruinous inside a dataloader. Run this
        offline over every unique window the training set uses, pickle the
        result, and hand it back through ``fused_cache``.

        Args:
            windows: iterable of (video_id, obs_start_s, obs_end_s).
            t0_by_video: {video_id: t0_s} -- the device-clock second of
                video-second 0, exactly as the loader defines it (TIME_ANCHOR).
            out_path: optional path to pickle the mapping to.

        Returns:
            {(video_id, round(obs_start, 3), round(obs_end, 3)): fused list}
        """
        import pickle
        import time

        cache = {}
        started = time.time()
        for i, (video_id, obs_start_s, obs_end_s) in enumerate(windows):
            boxer_df = self.load_boxer_csv(video_id)
            if boxer_df is None:
                continue
            t0_s = t0_by_video[video_id]
            vrs_start = (obs_start_s + t0_s) * 1e9
            vrs_end = (obs_end_s + t0_s) * 1e9
            mask = ((boxer_df['time_ns'] >= vrs_start)
                    & (boxer_df['time_ns'] <= vrs_end))
            window_df = boxer_df.loc[mask]
            key = (video_id, round(float(obs_start_s), 3),
                   round(float(obs_end_s), 3))
            cache[key] = self._fuse(window_df) if len(window_df) else []
            if progress_every and (i + 1) % progress_every == 0:
                rate = (i + 1) / (time.time() - started)
                logger.info("fused %d windows (%.2f/s)", i + 1, rate)

        if out_path:
            with open(out_path, 'wb') as fh:
                pickle.dump(cache, fh)
            logger.info("wrote %d fused windows to %s", len(cache), out_path)
        return cache

    def _fuse(self, window_df):
        """Cluster + average one window's rows, by whichever strategy is set.

        Both strategies return the same list of dicts (center / quat /
        half_extents / sem_id / prob), so everything downstream -- voxelisation,
        OBB tokens -- is unaffected by the choice. The IoU path adds `name`,
        `support`, `members` and `yaw`, which the consumers ignore.
        """
        if self.fusion == "grid":
            return _fuse_boxes(
                window_df, conf_threshold=self.conf_threshold,
                min_detections=self.min_detections,
                cluster_distance=self.cluster_distance)

        if self.fusion != "iou":
            raise ValueError(f"unknown fusion strategy {self.fusion!r}; "
                             f"expected 'grid' or 'iou'")

        # Imported lazily so a run using grid fusion never pays for it, and so
        # boxer_env stays importable if iou_fusion.py is absent.
        from iou_fusion import fuse_boxes_iou
        return fuse_boxes_iou(
            window_df, conf_threshold=self.conf_threshold,
            iou_threshold=self.iou_threshold,
            min_detections=self.min_detections,
            semantic_threshold=self.semantic_threshold,
            embeddings=self._get_class_embeddings(),
            nms_iou=self.nms_iou,
            legacy_half=self.legacy_half)

    def _get_class_embeddings(self):
        """The {class name -> unit vector} mapping, loaded at most once.

        ``fuse_boxes_iou`` accepts a path and would re-read the 2.7MB .npz on
        every window, so the path is resolved to a dict here and cached. A
        configured path that does not exist raises rather than quietly falling
        back to exact-name matching -- that fallback silently changes which
        boxes merge, and the file is easy to forget when copying a snapshot.
        """
        if self.class_embeddings is None:
            return None
        if self._class_embeddings_cache is None:
            emb = self.class_embeddings
            if isinstance(emb, (str, bytes, os.PathLike)):
                if not os.path.exists(emb):
                    raise FileNotFoundError(
                        f"class_embeddings not found at {emb!r}. Copy "
                        f"class_embeddings.npz next to boxer_env.py, point "
                        f"class_embeddings at it, or pass class_embeddings=None "
                        f"to merge only identical class names.")
                from iou_fusion import load_class_embeddings
                emb = load_class_embeddings(emb)
            self._class_embeddings_cache = emb
        return self._class_embeddings_cache

    def build_env(self, video_id, obs_start_s, obs_end_s, t0_s):
        """Build a 16x16x16 env_voxels grid for one observation window
        (single winning class per voxel, highest confidence wins).

        Args:
            video_id:    str.
            obs_start_s: float — observation start in video-seconds.
            obs_end_s:   float — observation end in video-seconds.
            t0_s:        float — device-clock seconds at video-second 0
                         (so device_ns = (video_sec + t0_s) * 1e9).

        Returns:
            (16,16,16) float64 array of (sem_id+1) semantic ids; 0 = empty.
        """
        empty = np.zeros((GRID_SIZE, GRID_SIZE, GRID_SIZE), dtype=np.float64)
        fused, voxel_centers = self._window_fused_boxes(video_id, obs_start_s, obs_end_s, t0_s)
        if fused is None:
            return empty

        grid = _voxelize_fused_boxes(fused, voxel_centers, orphan_fallback=self.orphan_fallback)
        if self.debug:
            logger.info("[%s] env grid: %d/4096 voxels occupied",
                        video_id, int(np.count_nonzero(grid)))
        return grid

    def build_env_multi(self, video_id, obs_start_s, obs_end_s, t0_s, k_max=4):
        """Like ``build_env``, but keeps every class overlapping a voxel (up
        to ``k_max``) instead of only the highest-confidence winner -- used
        by the env-class-averaging modality (see model.py).

        Returns:
          grid_multi: (16,16,16,k_max) int64 -- sem_id+1 per slot, 0 = pad.
          mask:       (16,16,16,k_max) bool  -- True where the slot is a real hit.
        """
        empty_grid = np.zeros((GRID_SIZE, GRID_SIZE, GRID_SIZE, k_max), dtype=np.int64)
        empty_mask = np.zeros((GRID_SIZE, GRID_SIZE, GRID_SIZE, k_max), dtype=bool)
        fused, voxel_centers = self._window_fused_boxes(video_id, obs_start_s, obs_end_s, t0_s)
        if fused is None:
            return empty_grid, empty_mask

        grid_multi, mask = _voxelize_fused_boxes_multi(
            fused, voxel_centers, k_max=k_max, orphan_fallback=self.orphan_fallback)
        if self.debug:
            logger.info("[%s] env grid (multi): %d/4096 voxels occupied",
                        video_id, int(mask.any(axis=-1).sum()))
        return grid_multi, mask

    def build_obb_tokens(self, video_id, obs_start_s, obs_end_s, t0_s,
                         max_tokens=MAX_OBB_TOKENS, box_transform=None):
        """One token per fused OBB for an observation window.

        The OBB-token alternative to ``build_env``: identical fusion input
        (``_window_fused_boxes``), but instead of rasterising the boxes into
        4096 voxel tokens each box becomes a single token. Measured over the
        real episode set that is ~124 tokens per window under 'iou' fusion, or
        ~162 once fixture_obbs appends the Digital-Twin fixtures, against 4096
        -- see MAX_OBB_TOKENS for the distribution behind the cap. (The ~240
        this used to quote was the 'grid' fusion figure.)

        Windows with no Boxer data at all come back all-padding; callers that
        feed this as the only modality must handle that, since a fully
        masked segment makes attention's softmax return NaN. That was 7-9% of
        episodes under 'grid' fusion, ~3-4% under 'iou', and is 0% whenever
        ``box_transform`` appends fixtures -- they are unconditional, so the
        all-padding case then only survives on a video with no voxel grid.

        ``box_transform``, if given, is called as ``f(fused_or_empty_list)`` and
        must return the box list to tokenise. It is the hook the fixture
        pipeline uses (loader.py) to prune boxes outside the voxel grid, drop
        the ones a Digital-Twin fixture already covers, and append the fixtures
        themselves. Applied BEFORE truncation, so the cap always sees the final
        set and truncation stays in one place (``_obb_tokens_from_fused``). It
        is called even when there are no fused boxes, so a window with no Boxer
        data still receives that participant's fixtures.

        Returns:
            (voxels[max_tokens,3], volumes[max_tokens], classes[max_tokens],
             mask[max_tokens]) -- see ``_obb_tokens_from_fused``.
        """
        fused, voxel_centers = self._window_fused_boxes(
            video_id, obs_start_s, obs_end_s, t0_s)
        if voxel_centers is None:
            return (np.zeros((max_tokens, 3), dtype=np.int64),
                    np.zeros(max_tokens, dtype=np.int64),
                    np.zeros(max_tokens, dtype=np.int64),
                    np.zeros(max_tokens, dtype=bool))

        if box_transform is not None:
            fused = box_transform(fused or [])
        tokens = _obb_tokens_from_fused(fused, voxel_centers, max_tokens=max_tokens)
        if self.debug:
            logger.info("[%s] obb tokens: %d/%d slots used",
                        video_id, int(tokens[3].sum()), max_tokens)
        return tokens

    def build_env_fallback_avg(self, video_id, obs_start_s, obs_end_s, t0_s,
                               k_max=4, mode='merge'):
        """Like ``build_env``, but the boxes the single-winner pass MISSED get
        averaged in rather than dropped -- see
        ``_voxelize_fused_boxes_fallback_avg`` for the two policies.

        Leftover-placement counts are accumulated into ``self.fallback_stats``
        so the loader can report how often the fallback actually fires (which,
        for mode='empty_only', is expected to be almost never).

        Returns:
          grid_multi: (16,16,16,k_max) int64 -- sem_id+1 per slot, 0 = pad.
          mask:       (16,16,16,k_max) bool  -- True where the slot is real.
        """
        empty_grid = np.zeros((GRID_SIZE, GRID_SIZE, GRID_SIZE, k_max), dtype=np.int64)
        empty_mask = np.zeros((GRID_SIZE, GRID_SIZE, GRID_SIZE, k_max), dtype=bool)
        fused, voxel_centers = self._window_fused_boxes(video_id, obs_start_s, obs_end_s, t0_s)
        if fused is None:
            return empty_grid, empty_mask

        grid_multi, mask, stats = _voxelize_fused_boxes_fallback_avg(
            fused, voxel_centers, k_max=k_max, mode=mode)
        for key, value in stats.items():
            self.fallback_stats[key] = self.fallback_stats.get(key, 0) + value
        self.fallback_stats['n_windows'] = self.fallback_stats.get('n_windows', 0) + 1
        if self.debug:
            logger.info("[%s] env grid (fallback-avg, mode=%s): %d/4096 occupied, "
                        "%d leftover boxes (%d on empty, %d on filled)",
                        video_id, mode, int(mask.any(axis=-1).sum()),
                        stats['n_leftover'], stats['n_leftover_on_empty'],
                        stats['n_leftover_on_filled'])
        return grid_multi, mask
