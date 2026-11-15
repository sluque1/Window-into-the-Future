"""
Per-track object trajectory reconstruction.

Rule (decided from analysis/person_vs_object_trajectory.py)
-------------------------------------------------------------
Let `displacement` = ||end_xyz - start_xyz|| (straight line between a
track's two annotated endpoints) and `person_dist` = the camera-wearer's
SLAM path length over the track's time_segment.

  * person_dist > displacement -> the person's actual path is longer than a
    direct line between the object's endpoints, i.e. they took some indirect
    route (around furniture, to another counter and back). Use the person's
    path SHAPE as the object's trajectory, offset so it passes exactly
    through the annotated start/end points.
  * person_dist <= displacement -> the person's path is no more informative
    than a direct line (typically short reaches/slides where SLAM barely
    moves, or object motion outpaces a near-stationary head). Use a straight
    line.

The analysis found no single fixed distance threshold (the person/object
distance ratio decays smoothly across the observed range rather than
showing one elbow), so the decision is made per track, relative to that
track's own displacement, rather than against a fixed cutoff.

Offset blending
----------------
The person's head trajectory is not the object's trajectory -- it is offset
by wherever the object is held relative to the head. `blend_to_endpoints`
removes that offset by linearly interpolating it to zero at both ends, so
the returned path has the person's path *shape* but passes exactly through
`start_xyz` at `t_start` and `end_xyz` at `t_end`.
"""

import numpy as np

from slam_utils import interp_position, path_length, sample_positions


def reconstruct_trajectory(start_xyz, t_start, end_xyz, t_end,
                            rel_times, person_xyz, sample_times):
    """Object 3D positions at `sample_times`, for a track whose end is ALREADY
    KNOWN (t_end has already happened by the time this is used, i.e. no
    future leakage in start_xyz/end_xyz/t_start/t_end themselves).

    start_xyz, end_xyz: (3,) arrays, the track's annotated endpoints.
    t_start, t_end: the track's time_segment (video-seconds), t_start < t_end.
    rel_times, person_xyz: the video's raw SLAM trajectory, as returned by
        slam_utils.load_raw_slam (or None if the video has no SLAM).
    sample_times: 1D array-like of query times. Expected to lie within
        [t_start, t_end]; values outside are clamped.

    Returns: (len(sample_times), 3) float64 array.

    IMPORTANT: do not call this for a track still in flight as of the
    observation window's end (t_end > obs_end) -- clipping `sample_times`
    alone is not enough to prevent leakage, because the branch decision
    (straight line vs. person path) and the offset blend both depend on the
    true end_xyz/t_end, i.e. the object's future placement. Use
    `reconstruct_open_ended_trajectory` for that case instead.
    """
    start_xyz = np.asarray(start_xyz, dtype=np.float64)
    end_xyz = np.asarray(end_xyz, dtype=np.float64)
    sample_times = np.asarray(sample_times, dtype=np.float64)

    if t_end <= t_start:
        return np.tile(start_xyz, (len(sample_times), 1))

    alpha = np.clip((sample_times - t_start) / (t_end - t_start), 0.0, 1.0)[:, None]
    straight_line = start_xyz[None, :] * (1 - alpha) + end_xyz[None, :] * alpha

    if rel_times is None:
        return straight_line

    person_dist = path_length(rel_times, person_xyz, t_start, t_end)
    displacement = float(np.linalg.norm(end_xyz - start_xyz))
    if person_dist is None or person_dist <= displacement:
        return straight_line

    person_at_samples = sample_positions(rel_times, person_xyz, sample_times)
    person_at_start = interp_position(rel_times, person_xyz, t_start)
    person_at_end = interp_position(rel_times, person_xyz, t_end)
    offset_start = start_xyz - person_at_start
    offset_end = end_xyz - person_at_end
    offset = offset_start[None, :] * (1 - alpha) + offset_end[None, :] * alpha

    return person_at_samples + offset


def reconstruct_linear(start_xyz, t_start, end_xyz, t_end, sample_times):
    """Object 3D positions at `sample_times`, assuming constant-speed
    straight-line motion between the track's two TRUE annotated endpoints.

    Used by the simplified trajectory-token path (trajectory_tokens.py's
    ``build_track_token_simple``): no person-path SLAM proxy, no branch
    decision -- just linear interpolation between (start_xyz, t_start) and
    (end_xyz, t_end). `sample_times` may lie outside [t_start, t_end]; the
    interpolation parameter is clamped to [0, 1] so those clamp to the
    nearer endpoint.

    Returns: (len(sample_times), 3) float64 array.
    """
    start_xyz = np.asarray(start_xyz, dtype=np.float64)
    end_xyz = np.asarray(end_xyz, dtype=np.float64)
    sample_times = np.asarray(sample_times, dtype=np.float64)

    if t_end <= t_start:
        return np.tile(start_xyz, (len(sample_times), 1))

    alpha = np.clip((sample_times - t_start) / (t_end - t_start), 0.0, 1.0)[:, None]
    return start_xyz[None, :] * (1 - alpha) + end_xyz[None, :] * alpha


def reconstruct_open_ended_trajectory(start_xyz, t_start, rel_times, person_xyz, sample_times):
    """Object 3D positions at `sample_times`, for a track that is still IN
    FLIGHT as of the querying observation window's end (the true end_xyz/
    t_end have not happened yet, or lie in the future relative to "now").

    Only `start_xyz`/`t_start` (the pickup, already observed) are known.
    There is no future endpoint to blend toward and no displacement to
    compare against a person-path length, so the branch rule from
    `reconstruct_trajectory` does not apply here -- the person's actual,
    causally-known path since pickup is the only grounded proxy available.
    It is anchored rigidly to `start_xyz` (a single, constant offset carried
    forward from t_start), with no correction toward an unknown destination.

    sample_times: 1D array-like, expected to lie within [t_start, obs_end]
        (the caller passes the querying window's obs_end as the latest
        sample time, and truncates the track's "end" fields to obs_end
        elsewhere -- see trajectory_tokens.py).

    Falls back to a constant position at start_xyz when there is no SLAM for
    this video (object assumed to stay where it was picked up).

    Returns: (len(sample_times), 3) float64 array.
    """
    start_xyz = np.asarray(start_xyz, dtype=np.float64)
    sample_times = np.asarray(sample_times, dtype=np.float64)

    if rel_times is None or len(sample_times) == 0:
        return np.tile(start_xyz, (len(sample_times), 1))

    person_at_samples = sample_positions(rel_times, person_xyz, sample_times)
    person_at_start = interp_position(rel_times, person_xyz, t_start)
    offset = start_xyz - person_at_start
    return person_at_samples + offset[None, :]
