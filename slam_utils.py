"""
Raw (un-resampled, ~1kHz) SLAM trajectory loading and querying.

loader.py's `_load_slam` resamples to 1 Hz for the model's `person_locations`
input. Object-trajectory reconstruction (trajectory.py) needs finer
resolution because many object movements last under a second, which 1 Hz
would alias.
"""

import json
import os

import numpy as np
import pandas as pd

# Overridable from the environment; see loader.py's path-constant block for why.
# loader.py and gaze.py hold their own copy of this root -- the env var is what
# keeps all three pointing at the same directory.
SLAM_BASE = os.environ.get(
    "SLAM_BASE",
    os.environ.get("HDEPIC_ROOT", "/data/video_datasets/HD-EPIC") + "/SLAM-and-Gaze")

_traj_cache = {}       # traj_path -> (t0, rel_times [n], xyz [n,3]) or None
_pose_cache = {}        # traj_path -> (device_times [n], xyz [n,3], quat_xyzw [n,4]) or None
_mapping_cache = {}     # pid -> {video_filename_key: multi_id}


def _load_mapping(pid):
    if pid not in _mapping_cache:
        path = os.path.join(SLAM_BASE, pid, "SLAM", "multi", "vrs_to_multi_slam.json")
        if os.path.exists(path):
            with open(path) as f:
                _mapping_cache[pid] = json.load(f)
        else:
            _mapping_cache[pid] = {}
    return _mapping_cache[pid]


def load_raw_slam(pid, video_id):
    """Return (t0, rel_times, xyz) for the video's SLAM trajectory, or None.

    t0: device-clock seconds at trajectory start (== video second 0), matching
    loader._load_slam's convention.
    rel_times: seconds since t0 (== video-seconds), sorted ascending.
    xyz: [n,3] world positions (Aria world frame), same rows as rel_times.
    """
    mapping = _load_mapping(pid)
    multi_id = next((v for k, v in mapping.items() if video_id in k), None)
    if multi_id is None:
        return None
    traj_path = os.path.join(SLAM_BASE, pid, "SLAM", "multi", str(multi_id),
                              "slam", "closed_loop_trajectory.csv")
    if traj_path in _traj_cache:
        return _traj_cache[traj_path]
    if not os.path.exists(traj_path):
        _traj_cache[traj_path] = None
        return None
    try:
        df = pd.read_csv(traj_path, usecols=[
            'tracking_timestamp_us', 'tx_world_device',
            'ty_world_device', 'tz_world_device'])
    except Exception:
        _traj_cache[traj_path] = None
        return None
    if df.empty:
        _traj_cache[traj_path] = None
        return None

    times = df['tracking_timestamp_us'].values.astype(np.float64) / 1e6
    order = np.argsort(times)
    times = times[order]
    xyz = df[['tx_world_device', 'ty_world_device', 'tz_world_device']].values[order]
    t0 = float(times[0])
    rel = times - t0
    result = (t0, rel, xyz)
    _traj_cache[traj_path] = result
    return result


def load_raw_slam_poses(pid, video_id):
    """Return (device_times, xyz, quat_xyzw) for the video's SLAM trajectory, or None.

    Same trajectory file as ``load_raw_slam``, but keeps the ORIENTATION too and
    leaves timestamps in raw device-clock seconds rather than re-zeroing them.
    Gaze needs both: the gaze CSV is timestamped on the same device clock, and
    turning a gaze direction into a world ray requires the full head pose, not
    just its position (see gaze.py).

    device_times: seconds on the device clock, sorted ascending.
    xyz:          [n,3] world positions (Aria world frame).
    quat_xyzw:    [n,4] world-from-device rotations as (qx, qy, qz, qw).
    """
    mapping = _load_mapping(pid)
    multi_id = next((v for k, v in mapping.items() if video_id in k), None)
    if multi_id is None:
        return None
    traj_path = os.path.join(SLAM_BASE, pid, "SLAM", "multi", str(multi_id),
                              "slam", "closed_loop_trajectory.csv")
    if traj_path in _pose_cache:
        return _pose_cache[traj_path]
    if not os.path.exists(traj_path):
        _pose_cache[traj_path] = None
        return None
    try:
        df = pd.read_csv(traj_path, usecols=[
            'tracking_timestamp_us', 'tx_world_device', 'ty_world_device',
            'tz_world_device', 'qx_world_device', 'qy_world_device',
            'qz_world_device', 'qw_world_device'])
    except Exception:
        _pose_cache[traj_path] = None
        return None
    if df.empty:
        _pose_cache[traj_path] = None
        return None

    times = df['tracking_timestamp_us'].values.astype(np.float64) / 1e6
    order = np.argsort(times)
    result = (
        times[order],
        df[['tx_world_device', 'ty_world_device',
            'tz_world_device']].values[order].astype(np.float64),
        df[['qx_world_device', 'qy_world_device', 'qz_world_device',
            'qw_world_device']].values[order].astype(np.float64),
    )
    _pose_cache[traj_path] = result
    return result


def interp_position(rel_times, xyz, t):
    """Position at time t, linearly interpolated and clamped to the trajectory's range."""
    t = min(max(t, rel_times[0]), rel_times[-1])
    idx = np.searchsorted(rel_times, t)
    idx = np.clip(idx, 1, len(rel_times) - 1)
    t0, t1 = rel_times[idx - 1], rel_times[idx]
    if t1 == t0:
        return xyz[idx]
    frac = (t - t0) / (t1 - t0)
    return xyz[idx - 1] + frac * (xyz[idx] - xyz[idx - 1])


def sample_positions(rel_times, xyz, times):
    """Interpolated positions at an array of query times (each clamped to range)."""
    return np.stack([interp_position(rel_times, xyz, t) for t in times])


def path_length(rel_times, xyz, t_start, t_end):
    """Path length (m) of the raw trajectory restricted to [t_start, t_end].

    Interpolates exactly at the two endpoints for sub-sample accuracy, then
    sums consecutive-sample distances in between. Returns None if the window
    falls entirely outside the trajectory's covered range.
    """
    if t_end < rel_times[0] or t_start > rel_times[-1]:
        return None
    t_start_c = max(t_start, rel_times[0])
    t_end_c = min(t_end, rel_times[-1])
    if t_end_c <= t_start_c:
        return 0.0

    lo = np.searchsorted(rel_times, t_start_c, side='left')
    hi = np.searchsorted(rel_times, t_end_c, side='right')
    inner = xyz[lo:hi]

    pts = ([interp_position(rel_times, xyz, t_start_c)] + list(inner) +
           [interp_position(rel_times, xyz, t_end_c)])
    coords = np.stack(pts)
    steps = np.linalg.norm(np.diff(coords, axis=0), axis=1)
    return float(steps.sum())
