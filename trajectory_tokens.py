"""
Per-episode trajectory-token construction.

Turns a video's usable object-movement tracks (loader.build_video_tracks)
into a fixed-size set of raw feature vectors for one episode's observation
window [obs_start, obs_end]. The learned MLP that maps these raw vectors
into the transformer's hidden space lives in model.py (`trajectory_mapper`);
this module only produces the (un-learned) numeric features.

Token layout (raw, un-embedded)
--------------------------------
    [ sinusoid(t_start - obs_end)   : 2*NUM_TIME_FREQS
    | start_xyz                     : 3
    | interior_traj_points          : K_INTERIOR * 3
    | sinusoid(t_end_used - obs_end): 2*NUM_TIME_FREQS
    | end_xyz_used                  : 3
    | is_ongoing                    : 1 ]

`object_class` is NOT part of this raw vector -- it is looked up through the
model's `class_embedding` table (the same one env tokens use, see
object_class_vocab.py) and added to the mapped token in model.py, rather than
being concatenated as a raw feature here. `build_episode_trajectory_tokens`
returns a parallel `classes` array (one class_embedding row index per token,
see object_class_vocab.py's index convention) alongside `tokens`/`mask`.

Track selection, clipping, and the leakage rule
-------------------------------------------------
A track is included in an episode if its time_segment overlaps
[obs_start, obs_end] at all. Only what falls inside the window is ever
exposed to the model: both `t_start`/`start_xyz` and `t_end`/`end_xyz` in the
token are clipped to [obs_start, obs_end] (`visible_start`, `visible_end`),
even when the track's true span extends further in either direction. Two
cases for how the clipped positions are computed:

  * t_end <= obs_end (the movement already concluded by "now"): both true
    endpoints are already-observed facts (not future information, even if
    t_start < obs_start), so the full track anchors `trajectory.
    reconstruct_trajectory`'s branch decision and offset blend. The
    returned path is then queried only at times within
    [visible_start, visible_end] -- i.e. a track that started before the
    window has its pre-window portion computed (for the correct shape) but
    not exposed; the token's "start" is where the object was AT obs_start,
    not the true pickup location/time. is_ongoing = 0.
  * t_end > obs_end (still in flight as the window closes): using the true
    end_xyz/t_end would leak the object's future placement, so
    `trajectory.reconstruct_open_ended_trajectory` is used instead, anchored
    at the true pickup (start_xyz, t_start) and queried only up to
    visible_end = obs_end. The token's "end" fields describe the LAST
    OBSERVED state (position at obs_end), not the true final placement.
    is_ongoing = 1.

Either way, interior shape points are also resampled within
[visible_start, visible_end], not the track's true full span -- the model
only ever sees the portion of the trajectory that occurred during the 30s
observation window.

How many tokens per episode
-----------------------------
PROVISIONAL: capped at `max_tokens`, keeping the `max_tokens` tracks with
the most recent observed end time (t_end_used). This selection policy is a
placeholder -- see the loader.py/model.py TODOs; the intended next step is
to decide this deliberately (e.g. by studying how many overlapping tracks a
typical window has) rather than leaving this default in place.
"""

import numpy as np

from trajectory import reconstruct_trajectory, reconstruct_open_ended_trajectory, reconstruct_linear

NUM_TIME_FREQS = 6
MIN_PERIOD = 2.0      # seconds -- shortest wavelength in the sinusoidal time encoding
MAX_PERIOD = 256.0    # seconds -- longest wavelength
K_INTERIOR = 6         # interior trajectory-shape points per token (excludes both endpoints)
GRID = 16               # voxel-grid resolution per axis, matches loader.py's GRID

# --- Distance travelled, for the 'end_distance' token layout (see model.py) ---
# The straight-line length of the VISIBLE span, ||pos(visible_end) - pos(visible_start)||
# in world metres. Binned rather than fed as a raw scalar for the same reason OBB
# size is (boxer_env.VOLUME_BIN_EDGES): an integer index through an nn.Embedding
# stays in the one shared 768-dim space, where a bare float would need its own
# projection and its own scale.
#
# Log-spaced because the range is three orders of magnitude. Measured over all
# 16,509 usable tracks (analysis-side script, full-track displacement; the clipped
# span the model actually bins is always <= this):
#   min 0.0013  p5 0.024  p25 0.077  p50 0.224  p75 0.693  p95 2.111  max 5.702
# geomspace(0.01, 5.0, 15) leaves NO empty bin and puts at most 11.1% of tracks in
# any one bin; linear bins would have put ~75% of them in the first two.
DISTANCE_BIN_EDGES = np.geomspace(0.01, 5.0, 15)
NUM_DISTANCE_BINS = len(DISTANCE_BIN_EDGES) + 1        # 16 occupied indices, 1..16
DISTANCE_VOCAB_SIZE = NUM_DISTANCE_BINS + 1            # +1 for index 0 = padding


def distance_bin(d):
    """1..16 log-spaced bin index for a displacement in metres (0 is reserved for pad).

    Mirrors boxer_env's volume binning exactly: np.digitize against 15 edges gives
    0..15, then +1 lifts it clear of the padding index.
    """
    return int(np.digitize(float(d), DISTANCE_BIN_EDGES)) + 1


def trajectory_raw_dim(k_interior=K_INTERIOR, num_time_freqs=NUM_TIME_FREQS):
    """Dimensionality of one raw (un-embedded) trajectory-token feature vector."""
    return 2 * (2 * num_time_freqs) + 3 + k_interior * 3 + 3 + 1


def trajectory_raw_dim_simple(num_time_freqs=NUM_TIME_FREQS):
    """Dimensionality of one raw feature vector in the simplified (--trajectory_simple)
    path: just the two sinusoidal time encodings -- start/end voxel coordinates are
    embedded separately (see model.py) rather than concatenated here, so the model
    can share weights with the location modality's per-axis embedding tables."""
    return 2 * (2 * num_time_freqs)


def sinusoidal_time_encoding(t, num_freqs=NUM_TIME_FREQS,
                              min_period=MIN_PERIOD, max_period=MAX_PERIOD):
    """[sin(2*pi*t/T_i), cos(2*pi*t/T_i)] for num_freqs log-spaced periods T_i.

    t is typically (track time - obs_end), i.e. negative for past events.
    Bounded in [-1, 1] regardless of |t|, unlike a raw scalar.
    """
    periods = np.geomspace(min_period, max_period, num_freqs)
    angles = 2.0 * np.pi * t / periods
    return np.concatenate([np.sin(angles), np.cos(angles)]).astype(np.float32)


def _interior_sample_times(t_start, t_end, k):
    """k times strictly between t_start and t_end (excludes both endpoints)."""
    return np.linspace(t_start, t_end, k + 2, dtype=np.float64)[1:-1]


def build_track_token(track, obs_start, obs_end, rel_times, person_xyz,
                       k_interior=K_INTERIOR, num_time_freqs=NUM_TIME_FREQS):
    """Raw feature vector + selection metadata for one track in one episode.

    track: dict from loader.build_video_tracks (name, t_start, t_end,
        start_xyz, end_xyz).
    obs_start, obs_end: the episode's observation window (video-seconds).
        Both the track's start and end are clipped to this window (see the
        module docstring) -- only what happened during the 30s window is
        exposed, regardless of how much earlier/later the track itself runs.
    rel_times, person_xyz: the video's raw SLAM trajectory (slam_utils.
        load_raw_slam), or (None, None) if the video has no SLAM.

    Returns (feature: float32[trajectory_raw_dim()], t_end_used: float), or
    None if the track's visible span within the window is empty (can happen
    when a track only touches the window boundary).
    """
    t_start, t_end = track["t_start"], track["t_end"]
    start_xyz, end_xyz = track["start_xyz"], track["end_xyz"]
    is_ongoing = t_end > obs_end

    visible_start = max(t_start, obs_start)
    visible_end = min(t_end, obs_end)
    if visible_end <= visible_start:
        return None

    query_times = np.concatenate([
        [visible_start],
        _interior_sample_times(visible_start, visible_end, k_interior),
        [visible_end],
    ])
    if is_ongoing:
        positions = reconstruct_open_ended_trajectory(
            start_xyz, t_start, rel_times, person_xyz, query_times)
    else:
        positions = reconstruct_trajectory(
            start_xyz, t_start, end_xyz, t_end, rel_times, person_xyz, query_times)

    start_xyz_used, end_xyz_used = positions[0], positions[-1]
    interior = positions[1:-1]

    feature = np.concatenate([
        sinusoidal_time_encoding(visible_start - obs_end, num_time_freqs),
        np.asarray(start_xyz_used, dtype=np.float32),
        interior.reshape(-1).astype(np.float32),
        sinusoidal_time_encoding(visible_end - obs_end, num_time_freqs),
        np.asarray(end_xyz_used, dtype=np.float32),
        np.array([1.0 if is_ongoing else 0.0], dtype=np.float32),
    ])
    return feature, visible_end


def _select_overlapping(tracks, obs_start, obs_end, exclude_unfinished=False):
    """Tracks visible inside the observation window.

    Default: any track overlapping [obs_start, obs_end] at all, INCLUDING one
    still in motion at obs_end. Such a track is then clipped to the window by
    interpolating along the straight line between its true annotated start and
    its true annotated END -- and that end lies in the future. The clipped
    boundary position therefore points straight at the destination, at a speed
    that says when it arrives. Worse, the prediction window starts only
    anticipation_time after obs_end, so an unfinished track can be one of the
    episode's own GT events.

    ``exclude_unfinished=True`` drops those tracks outright (keeping
    ``t_end <= obs_end``) rather than clipping them, so nothing derived from a
    post-window endpoint reaches the model. Tracks that merely START before
    obs_start are unaffected: clipping those interpolates between two past
    points.
    """
    if exclude_unfinished:
        return [tr for tr in tracks
                if obs_start <= tr["t_end"] <= obs_end]
    return [tr for tr in tracks if tr["t_start"] <= obs_end and tr["t_end"] >= obs_start]


def build_episode_trajectory_tokens(tracks, obs_start, obs_end, rel_times, person_xyz,
                                     max_tokens, k_interior=K_INTERIOR,
                                     num_time_freqs=NUM_TIME_FREQS,
                                     exclude_unfinished=False):
    """Fixed-size (tokens, mask, classes) for one episode.

    tracks: this video's full track list (loader.build_video_tracks output;
        each track carries a `class_idx` -- see object_class_vocab.py).
    max_tokens: PROVISIONAL cap -- see module docstring.
    exclude_unfinished: drop tracks still in motion at obs_end -- see
        _select_overlapping.

    Returns:
      tokens:  float32[max_tokens, trajectory_raw_dim()] (zero-padded)
      mask:    int64[max_tokens]  (1 = real token, 0 = padding)
      classes: int64[max_tokens]  (class_embedding row index; 0 at padding slots)
    """
    dim = trajectory_raw_dim(k_interior, num_time_freqs)
    overlapping = _select_overlapping(tracks, obs_start, obs_end, exclude_unfinished)

    built = [(tr, build_track_token(tr, obs_start, obs_end, rel_times, person_xyz, k_interior, num_time_freqs))
             for tr in overlapping]
    built = [(tr, ft) for tr, ft in built if ft is not None]
    built.sort(key=lambda item: item[1][1], reverse=True)   # most-recent visible_end first
    kept = built[:max_tokens]

    tokens = np.zeros((max_tokens, dim), dtype=np.float32)
    mask = np.zeros((max_tokens,), dtype=np.int64)
    classes = np.zeros((max_tokens,), dtype=np.int64)
    for i, (tr, (feature, _)) in enumerate(kept):
        tokens[i] = feature
        mask[i] = 1
        classes[i] = tr["class_idx"]
    return tokens, mask, classes


# =====================================================================
# Simplified path (--trajectory_simple): times + voxel coords only, no
# interior shape points, no person-path branch check, no is_ongoing flag.
# Both endpoints always use the track's TRUE annotated (start_xyz, t_start,
# end_xyz, t_end) via trajectory.reconstruct_linear (constant-speed straight
# line). The true endpoint is never exposed VERBATIM when it lies outside the
# window -- but do not read that as "no future information leaks": the clipped
# position at obs_end is interpolated ALONG THE LINE TO that endpoint, so it
# encodes the destination's direction and the arrival speed, and the track may
# itself be one of the episode's GT events. Use exclude_unfinished=True (loader
# ``trajectory_exclude_unfinished`` / --trajectory_exclude_unfinished) to drop
# tracks that finish after obs_end instead of clipping them.
# =====================================================================

def _nearest_voxel_ijk(xyz, vc_flat, grid=GRID):
    """(i, j, k) grid index of the voxel centre nearest to world point xyz."""
    dists = np.linalg.norm(vc_flat - np.asarray(xyz, dtype=np.float64), axis=1)
    flat_idx = int(np.argmin(dists))
    return np.unravel_index(flat_idx, (grid, grid, grid))


def build_track_token_simple(track, obs_start, obs_end, vc_flat, num_time_freqs=NUM_TIME_FREQS):
    """Simplified per-track feature: sinusoidal start/end times only --
    start/end voxel coordinates are returned separately (not concatenated
    into `feature`) so the model can embed them via the location modality's
    shared per-axis embedding tables (see model.py).

    Returns (feature: float32[trajectory_raw_dim_simple()],
             start_ijk: int64[3], end_ijk: int64[3], visible_end: float,
             dist_bin: int), or None if the track's visible span within the
    window is empty.

    `dist_bin` is the log-spaced bin of the straight-line distance the object
    travelled WITHIN the window, measured on the interpolated positions before
    they are snapped to voxels -- so it keeps the sub-voxel resolution the two
    ijk indices throw away, and a movement too small to cross a cell boundary is
    still visible as a nonzero distance.
    """
    t_start, t_end = track["t_start"], track["t_end"]
    start_xyz, end_xyz = track["start_xyz"], track["end_xyz"]

    visible_start = max(t_start, obs_start)
    visible_end = min(t_end, obs_end)
    if visible_end <= visible_start:
        return None

    positions = reconstruct_linear(start_xyz, t_start, end_xyz, t_end,
                                    [visible_start, visible_end])
    start_ijk = np.asarray(_nearest_voxel_ijk(positions[0], vc_flat), dtype=np.int64)
    end_ijk = np.asarray(_nearest_voxel_ijk(positions[1], vc_flat), dtype=np.int64)
    dist_bin = distance_bin(np.linalg.norm(positions[1] - positions[0]))

    feature = np.concatenate([
        sinusoidal_time_encoding(visible_start - obs_end, num_time_freqs),
        sinusoidal_time_encoding(visible_end - obs_end, num_time_freqs),
    ])
    return feature, start_ijk, end_ijk, visible_end, dist_bin


def build_episode_trajectory_tokens_simple(tracks, obs_start, obs_end, vc_flat,
                                            max_tokens, num_time_freqs=NUM_TIME_FREQS,
                                            exclude_unfinished=False):
    """Fixed-size (tokens, mask, classes, start_voxel, end_voxel) for one
    episode, simplified path -- see build_track_token_simple.

    exclude_unfinished: drop tracks still in motion at obs_end -- see
    _select_overlapping.

    Returns:
      tokens:      float32[max_tokens, trajectory_raw_dim_simple()] (zero-padded)
      mask:        int64[max_tokens]  (1 = real token, 0 = padding)
      classes:     int64[max_tokens]  (class_embedding row index; 0 at padding slots)
      start_voxel: int64[max_tokens, 3]  (1-16 per axis, same convention as
                   loader.py's person_locations; 0 at padding slots)
      end_voxel:   int64[max_tokens, 3]  (same convention)
      distance:    int64[max_tokens]  (1-16 log-spaced distance bin, 0 at padding
                   slots -- see distance_bin)
    """
    dim = trajectory_raw_dim_simple(num_time_freqs)
    overlapping = _select_overlapping(tracks, obs_start, obs_end, exclude_unfinished)

    built = [(tr, build_track_token_simple(tr, obs_start, obs_end, vc_flat, num_time_freqs))
             for tr in overlapping]
    built = [(tr, ft) for tr, ft in built if ft is not None]
    built.sort(key=lambda item: item[1][3], reverse=True)   # most-recent visible_end first
    kept = built[:max_tokens]

    tokens = np.zeros((max_tokens, dim), dtype=np.float32)
    mask = np.zeros((max_tokens,), dtype=np.int64)
    classes = np.zeros((max_tokens,), dtype=np.int64)
    start_voxel = np.zeros((max_tokens, 3), dtype=np.int64)
    end_voxel = np.zeros((max_tokens, 3), dtype=np.int64)
    distance = np.zeros((max_tokens,), dtype=np.int64)
    for i, (tr, (feature, start_ijk, end_ijk, _, dist_bin)) in enumerate(kept):
        tokens[i] = feature
        mask[i] = 1
        classes[i] = tr["class_idx"]
        start_voxel[i] = start_ijk + 1   # +1: match location_vocab_size convention (0=pad)
        end_voxel[i] = end_ijk + 1
        distance[i] = dist_bin
    return tokens, mask, classes, start_voxel, end_voxel, distance
