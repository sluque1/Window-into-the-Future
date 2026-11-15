"""
Gaze modality: which voxels the actor looked at most during the observation window.

Pipeline per episode
--------------------
One token PER SECOND of the observation window, so gaze is a temporal sequence
the same shape and alignment as ``person_locations`` -- token i of each refers
to the same second, which is what lets the model relate where the actor was to
where they were looking. For each second:

1. Take the gaze samples that fall inside that one-second bucket.
2. Turn each into a 3D world point (eye angles + focus distance + head pose).
3. Emit the voxel containing the most of those points.

Step 3 is a plain count: assign each point to its voxel, tally, take the
largest. Ties go to the lowest flat voxel index, so the token is a deterministic
function of the gaze data.

The token is then fed to the model split per axis, as three 30-long sequences
(90 tokens), which is precisely how ``person_locations`` is fed in -- same three
embedding tables, same end-aligned padding. See model.py.

A second with no gaze samples (or no head pose) is a padding token, exactly as
an unknown actor location is.

Only the coordinate is passed on: this is deliberately a positional modality
answering "where were they looking at second i", not "how hard".

Where the numbers come from
---------------------------
``SLAM-and-Gaze/<pid>/GAZE_HAND/mps_<video_id>_vrs/eye_gaze/general_eye_gaze.csv``
(10 Hz for P01-P03, 60 Hz for P04-P09; 154 of 156 videos have one) supplies per
sample:
  * ``left_yaw_rads_cpf`` / ``right_yaw_rads_cpf`` / ``pitch_rads_cpf`` -- the
    direction the eyes point, in the Central Pupil Frame (CPF). Validated
    accurate to ~2-3 deg against the dataset's own annotated look-at points.
  * ``depth_m`` -- how far along that direction the eyes are focused, inferred
    from vergence (the 6.3 cm between the eyes, cf. the CSV's own
    ``tx_left_eye_cpf`` = +0.0315 / ``tx_right_eye_cpf`` = -0.0315). Measured
    against the SLAM semidense point cloud, its error is ~0.16-0.26 m -- about
    one voxel -- with a sign that varies by video. This is the modality's main
    weakness and worth stating plainly: a one-voxel radial error can move the
    winning voxel of a second, and within a single video the error appears to be
    biased rather than random, so it will not simply average out. Casting the ray
    onto the semidense cloud instead would remove the dependence on depth_m
    altogether, and is a drop-in replacement for ``gaze_points_world``.

``SLAM/multi/<id>/slam/closed_loop_trajectory.csv`` supplies the head pose
(``T_world_device``) at the same device-clock timestamps.

Time alignment
--------------
Video second 0 is the FIRST GAZE TIMESTAMP, not the trajectory's first sample:
the closed-loop trajectory only begins once SLAM has initialised, a fixed
1.500 s (P01-P03) / 0.966 s (P04-P09) after the recording starts. Fitting the
offset that best reproduces the annotated priming look-at points lands on
first-gaze + 0.05-0.10 s on every video tested (residual 1.3-3.5 deg, versus
9-22 deg using the trajectory start).

On top of that there is a small residual, ``DEFAULT_TIME_OFFSET_S`` = +0.1 s.
Sweeping it against the same annotated look-at points gives (median 3D error,
metres, using each event's true distance so only the timing varies):

    offset      +0.000  +0.050  +0.075  +0.100  +0.150
    P01 (10Hz)   0.060   0.046   0.054   0.063   0.083
    P03 (10Hz)   0.060   0.074   0.076   0.088   0.118
    P04 (60Hz)   0.168   0.149   0.084   0.047   0.046
    P08 (60Hz)   0.238   0.061   0.050   0.064   0.080
    P09 (60Hz)   0.239   0.134   0.079   0.067   0.081

+0.1 s puts every video tested at or below 0.09 m; leaving it at zero costs the
60 Hz participants up to 0.24 m, i.e. roughly two voxels. The 10 Hz group's
curve is flat near its optimum because nearest-sample matching is already
+-50 ms for them, which is why they tolerate the same constant.

Caveat worth keeping in mind: this residual is fitted against HD-EPIC's own
priming annotation, so part of it may be that annotation's frame-to-time
convention rather than a physical latency. That is an argument FOR applying it
rather than against -- the interaction labels this model is trained on come from
the same annotation family (``mask_info.json``'s ``frame_number / FPS``), so
matching their convention is what keeps gaze and the targets on one clock.
Exposed as ``gaze_time_offset_s`` throughout, so it can be set back to 0.

This module is deliberately the ONLY consumer of the corrected anchor --
loader.py's ``person_locations`` and Boxer windowing keep anchoring on the
trajectory start, so runs stay comparable with everything recorded before this
was discovered. Internally nothing is converted anyway: sample selection and
pose lookup both happen in raw device time, and the video window is mapped INTO
device time once, at the top of ``build_episode_gaze_tokens``.
"""

import os

import numpy as np
import pandas as pd

from slam_utils import load_raw_slam_poses

# Overridable from the environment; see loader.py's path-constant block for why.
# loader.py and slam_utils.py hold their own copy of this root -- the env var is
# what keeps all three pointing at the same directory.
SLAM_BASE = os.environ.get(
    "SLAM_BASE",
    os.environ.get("HDEPIC_ROOT", "/data/video_datasets/HD-EPIC") + "/SLAM-and-Gaze")
GRID = 16

# Residual offset added to the first-gaze-timestamp anchor -- see the module
# docstring for the sweep this came from and the caveat attached to it.
DEFAULT_TIME_OFFSET_S = 0.1

# Eye (CPF) -> head (device) frame for Aria Gen1, from projectaria_tools'
# DeviceCadExtrinsics (DeviceCalibration.get_transform_device_cpf). A fixed
# property of the glasses: the rotation is identical for both frame sizes and
# the translation differs by ~2 mm, far below one voxel. Hard-coded so the
# training environment needs no projectaria_tools dependency.
T_DEVICE_CPF = np.array([
    [-0.03186014, -0.99862953, -0.04152095,  0.00495296],
    [ 0.7933534 ,  0.        , -0.60876135, -0.04914837],
    [ 0.60792706, -0.05233601,  0.79226614, -0.04918546],
    [ 0.        ,  0.        ,  0.        ,  1.        ],
], dtype=np.float64)

_gaze_cache = {}   # csv path -> (device_times, yaw, pitch, depth) or None


# =====================================================================
# Loading
# =====================================================================

def gaze_csv_path(pid, video_id):
    return os.path.join(SLAM_BASE, pid, "GAZE_HAND", f"mps_{video_id}_vrs",
                        "eye_gaze", "general_eye_gaze.csv")


def has_gaze_data(pid, video_id):
    return os.path.exists(gaze_csv_path(pid, video_id))


def load_gaze_csv(pid, video_id, max_hz=0.0):
    """Load and cache one video's gaze samples, or None.

    ``max_hz`` optionally thins the 60 Hz participants (P04-P09) toward the
    10 Hz the others were recorded at. It defaults to OFF: the token is an argmax
    over one second's points, so more points is strictly a better estimate of
    which voxel was actually being looked at, and the cost of keeping them all is
    a few dozen O(1) lattice lookups. Retained as a knob only for testing whether
    the two recording rates behave differently.

    Returns (device_times [n], yaw [n], pitch [n], depth [n]); device_times are
    raw device-clock seconds, matching closed_loop_trajectory.csv.
    """
    path = gaze_csv_path(pid, video_id)
    key = (path, float(max_hz))
    if key in _gaze_cache:
        return _gaze_cache[key]
    if not os.path.exists(path):
        _gaze_cache[key] = None
        return None
    try:
        df = pd.read_csv(path, usecols=[
            'tracking_timestamp_us', 'left_yaw_rads_cpf',
            'right_yaw_rads_cpf', 'pitch_rads_cpf', 'depth_m'])
    except Exception:
        _gaze_cache[key] = None
        return None
    if df.empty:
        _gaze_cache[key] = None
        return None

    times = df['tracking_timestamp_us'].values.astype(np.float64) / 1e6
    order = np.argsort(times)
    times = times[order]
    # The combined gaze direction is the mean of the two eyes' yaws. (The
    # vergence-weighted combination projectaria_tools computes differs by
    # ~4e-4 rad, i.e. 0.02 deg -- far below the ~2-3 deg the direction itself
    # is accurate to, and far below one voxel at any reachable distance.)
    yaw = 0.5 * (df['left_yaw_rads_cpf'].values[order]
                 + df['right_yaw_rads_cpf'].values[order])
    pitch = df['pitch_rads_cpf'].values[order].astype(np.float64)
    depth = df['depth_m'].values[order].astype(np.float64)

    keep = _thin_to_rate(times, max_hz)
    result = (times[keep], yaw[keep], pitch[keep], depth[keep])
    _gaze_cache[key] = result
    return result


def _thin_to_rate(times, max_hz):
    """Indices of a subset of ``times`` spaced at least 1/max_hz apart."""
    if max_hz is None or max_hz <= 0 or len(times) == 0:
        return np.arange(len(times))
    step = 1.0 / max_hz
    keep = [0]
    last = times[0]
    for i in range(1, len(times)):
        if times[i] - last >= step:
            keep.append(i)
            last = times[i]
    return np.asarray(keep, dtype=np.int64)


def gaze_t0(pid, video_id, max_hz=0.0, time_offset_s=DEFAULT_TIME_OFFSET_S):
    """Device-clock seconds at video second 0, for the gaze modality.

    That is the first gaze timestamp plus ``time_offset_s`` -- see the module
    docstring for both the ~1.0-1.5 s difference from loader.py's
    trajectory-based ``t0`` and the small residual offset on top. Returns None
    when the video has no gaze CSV.

    NOTE ``max_hz`` must match whatever is passed to ``load_gaze_csv``, since
    thinning can move the first retained sample.
    """
    gz = load_gaze_csv(pid, video_id, max_hz)
    return None if gz is None else float(gz[0][0]) + float(time_offset_s)


# =====================================================================
# Gaze -> world points
# =====================================================================

def _quat_xyzw_to_rotmat(q):
    """(qx, qy, qz, qw) -> 3x3 rotation matrix, vectorised over leading dims."""
    qx, qy, qz, qw = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    R = np.empty(q.shape[:-1] + (3, 3), dtype=np.float64)
    R[..., 0, 0] = 1 - 2*(qy*qy + qz*qz)
    R[..., 0, 1] = 2*(qx*qy - qz*qw)
    R[..., 0, 2] = 2*(qx*qz + qy*qw)
    R[..., 1, 0] = 2*(qx*qy + qz*qw)
    R[..., 1, 1] = 1 - 2*(qx*qx + qz*qz)
    R[..., 1, 2] = 2*(qy*qz - qx*qw)
    R[..., 2, 0] = 2*(qx*qz - qy*qw)
    R[..., 2, 1] = 2*(qy*qz + qx*qw)
    R[..., 2, 2] = 1 - 2*(qx*qx + qy*qy)
    return R


def gaze_points_cpf(yaw, pitch, depth):
    """Focus points in the eye (CPF) frame -- (n, 3).

    ``depth`` is the Euclidean distance to the focus point, so the direction is
    normalised before scaling (matching projectaria_tools'
    ``get_eyegaze_point_at_depth``).
    """
    d = np.stack([np.tan(yaw), np.tan(pitch), np.ones_like(yaw)], axis=-1)
    d /= np.linalg.norm(d, axis=-1, keepdims=True)
    return d * np.asarray(depth, dtype=np.float64)[..., None]


def gaze_points_world(device_times, yaw, pitch, depth, pose_times, pose_xyz, pose_quat):
    """World-frame gaze focus points -- (n, 3).

    Each sample is placed with the head pose nearest its own timestamp (poses
    run at ~1 kHz, so nearest-sample is sub-millisecond and interpolating the
    rotation would not move the point measurably). Samples falling outside the
    trajectory's covered time range are dropped.

    Returns (points [m,3], keep [m]) where ``keep`` indexes back into the input
    samples, so callers can tell how many were usable.
    """
    if len(device_times) == 0 or len(pose_times) == 0:
        return np.empty((0, 3)), np.empty((0,), dtype=np.int64)

    in_range = (device_times >= pose_times[0]) & (device_times <= pose_times[-1])
    keep = np.nonzero(in_range)[0]
    if len(keep) == 0:
        return np.empty((0, 3)), keep

    t = device_times[keep]
    idx = np.searchsorted(pose_times, t)
    idx = np.clip(idx, 1, len(pose_times) - 1)
    left = idx - 1
    idx = np.where(np.abs(pose_times[left] - t) <= np.abs(pose_times[idx] - t), left, idx)

    p_cpf = gaze_points_cpf(yaw[keep], pitch[keep], depth[keep])           # (m,3)
    p_dev = p_cpf @ T_DEVICE_CPF[:3, :3].T + T_DEVICE_CPF[:3, 3]           # (m,3)
    R = _quat_xyzw_to_rotmat(pose_quat[idx])                                # (m,3,3)
    p_world = np.einsum('nij,nj->ni', R, p_dev) + pose_xyz[idx]
    return p_world, keep


# =====================================================================
# World points -> voxel density -> tokens
# =====================================================================

class VoxelLattice:
    """O(1) world-point -> (i,j,k) lookup for a regular voxel grid.

    Both grid flavours this repo builds are regular lattices: the Boxer/Detic
    grids are axis-aligned (boxer_env._grid_from_points), and the fixed
    per-participant grids in new_voxel_centers/ are the same lattice rotated in
    XY. So the index is just the centre offset expressed in the lattice basis
    and rounded -- no need to scan all 4096 centres per point, which matters
    when a window holds hundreds of gaze samples.

    Falls back to brute-force nearest-centre if the grid turns out not to be a
    regular lattice.
    """

    def __init__(self, vc_flat, grid=GRID):
        self.vc_flat = np.asarray(vc_flat, dtype=np.float64)
        self.grid = grid
        g = self.vc_flat.reshape(grid, grid, grid, 3)
        steps = np.stack([g[1, 0, 0] - g[0, 0, 0],
                          g[0, 1, 0] - g[0, 0, 0],
                          g[0, 0, 1] - g[0, 0, 0]])          # (3,3), rows = axis steps
        self.origin = g[0, 0, 0]
        try:
            self.inv_steps = np.linalg.inv(steps.T)           # world offset -> index
        except np.linalg.LinAlgError:
            self.inv_steps = None
        self.is_lattice = self.inv_steps is not None and self._verify(g, steps)

    def _verify(self, g, steps, tol=1e-4):
        """Check a few far-flung centres actually sit where the lattice predicts."""
        probes = [(0, 0, 0), (self.grid - 1, 0, 0), (0, self.grid - 1, 0),
                  (0, 0, self.grid - 1), (self.grid - 1,) * 3]
        for i, j, k in probes:
            predicted = self.origin + np.array([i, j, k], dtype=np.float64) @ steps
            if np.linalg.norm(predicted - g[i, j, k]) > tol:
                return False
        return True

    def index(self, points):
        """(n, 3) world points -> (n, 3) int64 (i, j, k), clipped to the grid."""
        points = np.atleast_2d(np.asarray(points, dtype=np.float64))
        if not self.is_lattice:
            d = points[:, None, :] - self.vc_flat[None, :, :]
            flat = np.argmin(np.einsum('nvd,nvd->nv', d, d), axis=1)
            return np.stack(np.unravel_index(flat, (self.grid,) * 3), axis=-1)
        idx = np.rint((points - self.origin) @ self.inv_steps.T)
        return np.clip(idx, 0, self.grid - 1).astype(np.int64)


def densest_voxel(points, lattice):
    """The voxel containing the most of ``points``.

    A plain tally: every point is assigned to its voxel and the fullest voxel
    wins. Ties break towards the lowest flat voxel index (numpy's argmax), so
    the result is a deterministic function of the gaze data.

    Returns (i, j, k) int64, or None if there are no points.
    """
    if len(points) == 0:
        return None
    ijk = lattice.index(points)
    flat = np.ravel_multi_index((ijk[:, 0], ijk[:, 1], ijk[:, 2]),
                                (lattice.grid,) * 3)
    winner = int(np.bincount(flat, minlength=lattice.grid ** 3).argmax())
    return np.asarray(np.unravel_index(winner, (lattice.grid,) * 3), dtype=np.int64)


def build_episode_gaze_tokens(gaze, poses, t0_gaze, obs_start, obs_end, lattice,
                              observation_time=30):
    """One gaze token per second of the observation window.

    gaze:  (device_times, yaw, pitch, depth) from load_gaze_csv, or None.
    poses: (pose_times, pose_xyz, pose_quat) from slam_utils.load_raw_slam_poses,
           or None.
    t0_gaze: device-clock seconds at video second 0 (see module docstring).
    obs_start, obs_end: the episode's observation window, in video-seconds.

    Token i covers video-second ``obs_end - observation_time + i``, i.e. the
    sequence is aligned to the END of the window and padded at the FRONT --
    deliberately identical to loader.py's ``person_locations`` back-fill, so
    token i of both modalities describes the same second.

    Returns:
      voxels: int64[observation_time, 3] -- 1..16 per axis, matching the
              person_locations convention; 0 at padding slots.
      mask:   int64[observation_time]    -- 1 = real token, 0 = padding.
    """
    voxels = np.zeros((observation_time, 3), dtype=np.int64)
    mask = np.zeros((observation_time,), dtype=np.int64)
    if gaze is None or poses is None or t0_gaze is None:
        return voxels, mask

    device_times, yaw, pitch, depth = gaze
    idx = observation_time - 1
    for second in range(int(obs_end) - 1, int(obs_start) - 1, -1):
        if idx < 0:
            break
        # The one place video-seconds meet device time: bucket the second here,
        # then stay in device time for the pose lookup.
        lo, hi = second + t0_gaze, second + 1 + t0_gaze
        sel = np.nonzero((device_times >= lo) & (device_times < hi))[0]
        if len(sel):
            points, _ = gaze_points_world(
                device_times[sel], yaw[sel], pitch[sel], depth[sel],
                poses[0], poses[1], poses[2])
            ijk = densest_voxel(points, lattice)
            if ijk is not None:
                voxels[idx] = ijk + 1   # +1: 0 is reserved for pad/unknown
                mask[idx] = 1
        idx -= 1
    return voxels, mask
