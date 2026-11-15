"""
Does the per-participant 16^3 voxel grid's geometry explain the per-participant
PR-AUC spread under LOPO?

Every participant gets 16 cells per axis regardless of how big their kitchen is,
so voxel size differs per participant. A bigger voxel collapses more distinct GT
interaction points into the same cell, which lowers the per-voxel positive rate
-- and PR-AUC is prevalence-sensitive. This script measures, per participant:
voxel volume, GT positive rate, how tightly the grid hugs the kitchen
(fill_fraction), and the held-out PR-AUC, then correlates them.

--------------------------------------------------------------------------
WHERE THE GRID ACTUALLY COMES FROM (read this before trusting any number)
--------------------------------------------------------------------------
The production runs set VOXEL_CENTERS_DIR=new_voxel_centers, which sends
loader.py down ``_load_fixed_voxel_centers`` -- a *pre-baked* (4096, 3) .npy per
participant. That path never calls ``boxer_env._grid_from_points``, so for the
runs this study is about the grid is **not** "AABB(Boxer OBB corners UNION SLAM
trajectory)". That construction is only the fallback used when
``voxel_centers_dir`` is None, and it is strictly axis-aligned (a plain
``np.linspace`` per world axis -- it cannot produce a rotated box).

The real provenance, traced through the v1 tree:

  /data/video_datasets/HD-EPIC/Digital-Twin/blenders/Voxels_corrected/<pid>_voxels.blend
      -> collection "Voxels_Cupboards_16_obj"  (P03: "Voxels_Objects_16_obj")
      -> FIction_HD_EPIC/additional/extract_voxel_centers_blend_v2.py
         (driven by run_extract_voxel_centers.sh / run_extract_voxel_centers_local.py)
      -> new_voxel_centers/<pid>_voxel_centers.npy

i.e. the grid is HD-EPIC's own Digital-Twin voxel grid, hand-fitted to each
kitchen in Blender. The yaw is baked into the Blender object transforms and
therefore into the .npy; **no rotation angle is written down anywhere in this
codebase**. This script recovers the nine angles by measuring the lattice
(see ``lattice_geometry``) -- that measurement is the only record of them.

Consequence for the geometry maths: the 4096 centres are a regular orthogonal
lattice ROTATED about the vertical axis, so the axis-aligned span of the centres
is NOT the grid's size. Extents must come from the lattice step vectors. (The
same point is made in analysis/count_analysis.py.)

--------------------------------------------------------------------------
What it prints
--------------------------------------------------------------------------
  TABLE 1  per-participant geometry + label prevalence + fill_fraction
           -> analysis/out/grid_geometry.csv
  TABLE 2  held-out PR-AUC per participant, joined onto Table 1, recomputed
           from the saved predictions.pkl/labels.pkl (not read off a log)
  CORRELATIONS  Spearman + text scatters
  TABLE 3a z-extent spread across kitchens
  TABLE 3b rotation provenance, long-axis consistency, occupancy-centroid signs

Read-only. It imports from loader.py / boxer_env.py / gaze.py / slam_utils.py /
analysis/boxer_coverage_audit.py and never writes to them. No SLURM, no GPU.

Usage:
  python analysis/grid_geometry_audit.py
  python analysis/grid_geometry_audit.py --list-families-only
  python analysis/grid_geometry_audit.py --family run1_vle_aligned
  python analysis/grid_geometry_audit.py --geometry-only   # skip episode building
"""

import argparse
import json
import os
import pickle
import re
import sys
from collections import defaultdict

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

from loader import (HDEpicInteractionDataset, FEATURES_BASE, GRID, NUM_VOXELS,  # noqa: E402
                    coord_to_voxel, _load_video_durations)
from gaze import VoxelLattice  # noqa: E402
from boxer_env import BOXER_OBBS_ROOT  # noqa: E402
import boxer_coverage_audit as bca  # noqa: E402

OUT_DIR = os.path.join(HERE, "out")
CSV_PATH = os.path.join(OUT_DIR, "grid_geometry.csv")
VOXEL_CENTERS_DIR = os.path.join(ROOT, "new_voxel_centers")
SNAPSHOTS_DIR = os.environ.get("SNAPSHOTS_ROOT", "/data/video_datasets/HD-EPIC/snapshots")

PARTICIPANTS = [f"P{i:02d}" for i in range(1, 10)]
SEED = 0
N_SAMPLE_EPISODES = 300

# Settings of the LOPO production runs this study is about:
# submit_video_location_env_eval25pct.sh -- OBS_TIME=30, ANTICIP_TIME=5,
# FUTURE_TIME=180, TIME_ANCHOR=gaze, VOXEL_CENTERS_DIR=new_voxel_centers.
# (submit_video_location_env.sh, the run1 predecessor, is identical apart from
# eval cadence. FUTURE_TIME=60/120 variants exist as separate sweeps; 180 is the
# one the run1/run2 "vle_aligned" LOPO folds used, so 180 it is.)
OBS_TIME, ANTICIP_TIME, FUTURE_TIME = 30, 5, 180

# The 9-fold LOPO family joined by default: one job per held-out participant,
# no val split, so its test set is exactly the held-out participant.
DEFAULT_FAMILY = "run2_vle_aligned"

# Loader kwargs that are pure caching, not semantics. static_env=False skips the
# per-window env-grid precompute in __init__; __getitem__ would then build the
# same grid lazily. This script never calls __getitem__ (it reads
# ds.dataset[i]['gt_locations'] directly), so the env values are never used and
# turning the precompute off only saves time. Grid centres and GT labels are
# byte-identical either way.
STATIC_ENV = False


# =====================================================================
# Lattice geometry (rotation-aware)
# =====================================================================

def lattice_geometry(vc_flat):
    """Geometry of one participant's 16^3 grid, from its lattice step vectors.

    The centres form a regular orthogonal lattice that is rotated about the
    vertical axis, so per-world-axis min/max (or np.unique per column) would
    report the AABB of a tilted box, not the box. The three step vectors
    g[1,0,0]-g[0,0,0] etc. are the box's own axes.

    Returns a dict with the per-lattice-axis extents/edges, the recovered yaw,
    and the residuals that justify calling it an orthogonal lattice at all.
    """
    g = np.asarray(vc_flat, dtype=np.float64).reshape(GRID, GRID, GRID, 3)
    steps = np.stack([g[1, 0, 0] - g[0, 0, 0],
                      g[0, 1, 0] - g[0, 0, 0],
                      g[0, 0, 1] - g[0, 0, 0]])          # (3,3), rows = axis steps
    edges = np.linalg.norm(steps, axis=1)                # voxel edge per axis
    units = steps / edges[:, None]

    # Residual 1: is it really affine? Predict every centre from the steps.
    idx = np.stack(np.meshgrid(*[np.arange(GRID)] * 3, indexing="ij"), axis=-1)
    predicted = g[0, 0, 0] + idx.reshape(-1, 3) @ steps
    affine_err = float(np.abs(predicted - g.reshape(-1, 3)).max())

    # Residual 2: are the axes mutually perpendicular?
    gram = units @ units.T
    orth_err = float(np.abs(gram - np.eye(3)).max())

    # Residual 3: is the rotation a pure yaw (third axis vertical)?
    z_tilt_deg = float(np.degrees(np.arccos(np.clip(abs(units[2, 2]), -1, 1))))

    # Yaw of lattice axis 0 in the world XY plane. A lattice axis' *sign* is
    # arbitrary (the exporter could have walked the cubes either way), so fold
    # by 180 into (-90, 90].
    yaw = float(np.degrees(np.arctan2(units[0, 1], units[0, 0])))
    yaw = (yaw + 90.0) % 180.0 - 90.0
    if yaw == -90.0:
        yaw = 90.0
    # Footprint orientation invariant to which axis is called "x": fold by 90.
    yaw_mod90 = float(yaw % 90.0)

    return {
        "steps": steps,
        "origin": g[0, 0, 0],
        "edges": edges,
        "extents": edges * GRID,
        "rotation_deg": yaw,
        "rotation_mod90_deg": yaw_mod90,
        "affine_err_m": affine_err,
        "orth_err": orth_err,
        "z_tilt_deg": z_tilt_deg,
    }


def load_participant_grid(pid):
    path = os.path.join(VOXEL_CENTERS_DIR, f"{pid}_voxel_centers.npy")
    vc = np.load(path).astype(np.float64)
    if vc.shape != (NUM_VOXELS, 3):
        raise ValueError(f"{path}: expected ({NUM_VOXELS}, 3), got {vc.shape}")
    return vc


# =====================================================================
# Occupancy (fill_fraction) and label prevalence
# =====================================================================

def flat_index(ijk):
    return np.ravel_multi_index((ijk[:, 0], ijk[:, 1], ijk[:, 2]), (GRID,) * 3)


def continuous_index(points, geom):
    """(n,3) world points -> (n,3) real-valued lattice coordinates, unclipped.

    A point is inside the box iff every coordinate is within [-0.5, GRID-0.5]
    (centres sit at integer coordinates, so the box edge is half a voxel out).
    """
    inv = np.linalg.inv(geom["steps"].T)
    return (np.asarray(points, dtype=np.float64) - geom["origin"]) @ inv.T


def occupancy_stats(ds, geom, lattice, used_videos):
    """Which of the 4096 voxels ever hold a Boxer OBB centre or a SLAM sample.

    Boxer centres come from ds.boxer.load_boxer_csv (the same 10 Hz-downsampled
    frame the env builder fuses, already cached by __init__, so this is free);
    SLAM comes from the per-second positions the loader already built.

    Two flavours of "occupied":
      * clipped   -- production semantics. boxer_env.find_closest_voxel /
                     loader.coord_to_voxel take the NEAREST centre, so a point
                     outside the box still lands in a boundary voxel.
      * strict    -- only points actually inside the box count. The gap between
                     the two says how much data the grid fails to contain.
    """
    occ_clip = np.zeros(NUM_VOXELS, dtype=bool)
    occ_strict = np.zeros(NUM_VOXELS, dtype=bool)
    boxer_occ_clip = np.zeros(NUM_VOXELS, dtype=bool)
    n_pts = n_outside = 0
    n_boxer_videos = n_slam_videos = 0

    for vid in used_videos:
        chunks = []
        df = ds.boxer.load_boxer_csv(vid)
        if df is not None and len(df):
            centres = df[["tx_world_object", "ty_world_object",
                          "tz_world_object"]].to_numpy(dtype=np.float64)
            centres = centres[np.isfinite(centres).all(axis=1)]
            if len(centres):
                n_boxer_videos += 1
                ijk = lattice.index(centres)
                boxer_occ_clip[flat_index(ijk)] = True
                chunks.append(centres)

        slam_sec = ds._slam_sec.get(vid)
        if slam_sec is not None and len(slam_sec):
            traj = np.asarray(slam_sec, dtype=np.float64)
            traj = traj[np.isfinite(traj).all(axis=1)]
            if len(traj):
                n_slam_videos += 1
                chunks.append(traj)

        if not chunks:
            continue
        pts = np.concatenate(chunks, axis=0)
        ijk = lattice.index(pts)
        occ_clip[flat_index(ijk)] = True

        cont = continuous_index(pts, geom)
        inside = ((cont >= -0.5) & (cont <= GRID - 0.5)).all(axis=1)
        if inside.any():
            occ_strict[flat_index(ijk[inside])] = True
        n_pts += len(pts)
        n_outside += int((~inside).sum())

    # Centroid of Boxer-occupied voxels, per horizontal lattice axis, as a
    # fraction of the grid. Unweighted over distinct occupied voxels, so it
    # answers "where in the box does the kitchen sit", not "where is the mass".
    occ_idx = np.argwhere(boxer_occ_clip.reshape(GRID, GRID, GRID))
    if len(occ_idx):
        centroid = occ_idx.mean(axis=0) / (GRID - 1.0)
    else:
        centroid = np.full(3, np.nan)

    return {
        "fill_fraction": float(occ_clip.mean()),
        "fill_fraction_strict": float(occ_strict.mean()),
        "fill_fraction_boxer": float(boxer_occ_clip.mean()),
        "frac_points_outside_grid": float(n_outside / n_pts) if n_pts else float("nan"),
        "n_occupancy_points": int(n_pts),
        "n_boxer_videos": n_boxer_videos,
        "n_slam_videos": n_slam_videos,
        "occ_centroid_axis0": float(centroid[0]),
        "occ_centroid_axis1": float(centroid[1]),
        "occ_centroid_axis2": float(centroid[2]),
    }


def label_prevalence_stats(ds, lattice, rng):
    """Mean positive rate / positive count over a random episode subsample.

    Reproduces loader.__getitem__'s label block exactly: gt_voxels[coord_to_voxel(
    loc)] = 1 for each GT location, so the number of positives is the number of
    DISTINCT voxels the episode's GT points fall in. Uses the lattice for the
    lookup instead of the 4096-way scan (verified equivalent in verify_lattice).
    """
    n = len(ds.dataset)
    if n == 0:
        return {"n_episodes_total": 0, "n_episodes_sampled": 0,
                "mean_pos_rate": float("nan"), "mean_pos_count": float("nan")}
    n_sample = min(N_SAMPLE_EPISODES, n)
    idxs = rng.choice(n, size=n_sample, replace=False)

    counts = []
    for i in idxs:
        locs = ds.dataset[int(i)]["gt_locations"]
        if not len(locs):
            counts.append(0)
            continue
        ijk = lattice.index(np.asarray(locs, dtype=np.float64))
        counts.append(len(np.unique(flat_index(ijk))))
    counts = np.asarray(counts, dtype=np.float64)
    return {
        "n_episodes_total": n,
        "n_episodes_sampled": n_sample,
        "mean_pos_rate": float((counts / NUM_VOXELS).mean()),
        "mean_pos_count": float(counts.mean()),
    }


def verify_lattice(ds, lattice, vc_flat, rng, n_probe=200):
    """Sanity-check the fast lattice lookup against loader.coord_to_voxel.

    They must agree, otherwise every prevalence number below is measuring a
    different grid from the one the model trained on.
    """
    pool = [loc for d in ds.dataset[:400] for loc in d["gt_locations"]]
    if not pool:
        return None
    sel = rng.choice(len(pool), size=min(n_probe, len(pool)), replace=False)
    pts = np.asarray([pool[int(i)] for i in sel], dtype=np.float64)
    fast = flat_index(lattice.index(pts))
    slow = np.array([coord_to_voxel(p, vc_flat) for p in pts])
    return int((fast != slow).sum()), len(pts)


# =====================================================================
# Data-quality notes (dropped videos / truncated Boxer CSVs)
# =====================================================================

def list_feature_videos(pid):
    pdir = os.path.join(FEATURES_BASE, pid)
    if not os.path.isdir(pdir):
        return []
    return sorted(fn[: -len("_features.pt")] for fn in os.listdir(pdir)
                  if fn.endswith("_features.pt"))


def data_quality_notes(pid, used_videos):
    """Re-derived here rather than copied from CLAUDE.md, whose coverage section
    is explicitly flagged as drifting."""
    feat_videos = set(list_feature_videos(pid))
    dropped = sorted(feat_videos - set(used_videos))

    durations = _load_video_durations(pid)
    truncated = []
    for vid in used_videos:
        path = os.path.join(BOXER_OBBS_ROOT, vid, "boxer_3dbbs.csv")
        if not os.path.exists(path):
            truncated.append((vid, "no Boxer CSV"))
            continue
        span = bca.csv_span_seconds(path)
        if span is None:
            truncated.append((vid, "empty Boxer CSV"))
            continue
        dur = durations.get(vid)
        if not dur or dur <= 0:
            continue
        cov = span[2] / dur
        if cov < 0.95:
            truncated.append((vid, f"Boxer CSV covers {cov * 100:.0f}%"))

    parts = []
    if dropped:
        parts.append(f"{len(dropped)}/{len(feat_videos)} feature video(s) contributed "
                     f"no episodes: {','.join(dropped)}")
    if truncated:
        parts.append(f"{len(truncated)} contributing video(s) truncated/missing in Boxer: "
                     + ",".join(f"{v}({s})" for v, s in truncated))
    return " | ".join(parts)


# =====================================================================
# Part 1 driver
# =====================================================================

def build_participant_dataset(pid):
    return HDEpicInteractionDataset(
        split=f"audit-{pid}",       # cosmetic: participants= overrides split selection
        observation_time=OBS_TIME, anticipation_time=ANTICIP_TIME,
        future_time=FUTURE_TIME, participants=[pid], static_env=STATIC_ENV,
        voxel_centers_dir=VOXEL_CENTERS_DIR, time_anchor="gaze", verbose=True,
    )


def collect_geometry_table(geometry_only=False):
    rng = np.random.default_rng(SEED)
    rows = []
    for pid in PARTICIPANTS:
        print(f"\n--- {pid} " + "-" * 60, flush=True)
        vc_flat = load_participant_grid(pid)
        geom = lattice_geometry(vc_flat)
        extents, edges = geom["extents"], geom["edges"]
        print(f"[{pid}] grid: extent=({extents[0]:.2f}, {extents[1]:.2f}, {extents[2]:.2f}) m  "
              f"yaw={geom['rotation_deg']:+.2f} deg  "
              f"voxel_vol={float(np.prod(edges)):.6f} m^3", flush=True)
        if geom["affine_err_m"] > 1e-3 or geom["orth_err"] > 1e-3:
            print(f"[{pid}] WARNING: centres are not a clean orthogonal lattice "
                  f"(affine residual {geom['affine_err_m']:.2e} m, orthogonality "
                  f"{geom['orth_err']:.2e}) -- extents below are unreliable.")
        if geom["z_tilt_deg"] > 0.5:
            print(f"[{pid}] WARNING: third lattice axis is {geom['z_tilt_deg']:.2f} deg "
                  f"off vertical -- the rotation is not a pure yaw.")

        row = {
            "participant": pid,
            "extent_x_m": extents[0], "extent_y_m": extents[1], "extent_z_m": extents[2],
            "rotation_deg": geom["rotation_deg"],
            "rotation_mod90_deg": geom["rotation_mod90_deg"],
            "voxel_edge_x_m": edges[0], "voxel_edge_y_m": edges[1], "voxel_edge_z_m": edges[2],
            "voxel_volume_m3": float(np.prod(edges)),
            "affine_err_m": geom["affine_err_m"],
            "orth_err": geom["orth_err"],
            "z_tilt_deg": geom["z_tilt_deg"],
        }

        if geometry_only:
            rows.append(row)
            continue

        print(f"[{pid}] building episodes (obs={OBS_TIME}s anticip={ANTICIP_TIME}s "
              f"future={FUTURE_TIME}s, TIME_ANCHOR=gaze, fixed grid)...", flush=True)
        ds = build_participant_dataset(pid)
        used_videos = sorted({d["video_id"] for d in ds.dataset})
        row["n_videos"] = len(used_videos)
        if not used_videos:
            print(f"[{pid}] WARNING: no video produced episodes.")
            row["data_notes"] = data_quality_notes(pid, used_videos)
            rows.append(row)
            continue

        lattice = VoxelLattice(vc_flat)
        if not lattice.is_lattice:
            print(f"[{pid}] NOTE: VoxelLattice fell back to brute force "
                  f"(grid not recognised as regular) -- slower but still correct.")
        mism = verify_lattice(ds, lattice, vc_flat, rng)
        if mism is not None:
            n_bad, n_tot = mism
            print(f"[{pid}] lattice lookup vs loader.coord_to_voxel: "
                  f"{n_bad}/{n_tot} disagreements"
                  + ("  <-- INVESTIGATE" if n_bad else " (exact match)"))

        stats = label_prevalence_stats(ds, lattice, rng)
        print(f"[{pid}] computing occupancy over {len(used_videos)} video(s)...", flush=True)
        occ = occupancy_stats(ds, geom, lattice, used_videos)

        row.update(stats)
        row.update(occ)
        row["n_episodes"] = stats["n_episodes_sampled"]
        row["data_notes"] = data_quality_notes(pid, used_videos)
        rows.append(row)

        print(f"[{pid}] n_videos={row['n_videos']} "
              f"mean_pos_rate={row['mean_pos_rate']:.5f} "
              f"mean_pos_count={row['mean_pos_count']:.2f} "
              f"(n={stats['n_episodes_sampled']}/{stats['n_episodes_total']} episodes)  "
              f"fill_fraction={occ['fill_fraction']:.3f} "
              f"(strict {occ['fill_fraction_strict']:.3f}, "
              f"{occ['frac_points_outside_grid'] * 100:.1f}% of points outside the box)",
              flush=True)
        if row["data_notes"]:
            print(f"[{pid}] NOTE: {row['data_notes']}", flush=True)

    df = pd.DataFrame(rows)
    front = ["participant", "n_videos", "extent_x_m", "extent_y_m", "extent_z_m",
             "rotation_deg", "voxel_edge_x_m", "voxel_edge_y_m", "voxel_edge_z_m",
             "voxel_volume_m3", "mean_pos_rate", "mean_pos_count", "n_episodes",
             "fill_fraction"]
    cols = [c for c in front if c in df.columns] + [c for c in df.columns if c not in front]
    return df[cols]


# =====================================================================
# Part 2: LOPO PR-AUC join
# =====================================================================

_FAMILY_RE = re.compile(r"^\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}_loocv_(P0[1-9])_(.+)$")


def find_loocv_families(snapshots_dir):
    """family -> {participant: snapshot_dir}. Name-based shortlist only; the
    actual held-out participant is confirmed from the job log, see fold_test_participants."""
    families = defaultdict(dict)
    if not os.path.isdir(snapshots_dir):
        return families
    for name in sorted(os.listdir(snapshots_dir)):
        m = _FAMILY_RE.match(name)
        if m:
            pid, family = m.groups()
            families[family][pid] = os.path.join(snapshots_dir, name)
    return families


def fold_test_participants(snapshot_dir):
    """The participants this job actually tested on, read from the job's own
    stdout ("TRAIN=... VAL=... TEST=..." printed by sbatch.sh).

    Deliberately NOT inferred from the directory name: the name is a label the
    submitter chose, the log line is what the run did. The frozen sbatch.sh in
    the snapshot only records the fallback default, not the exported value, so
    the log is the sole authority.
    """
    out_dir = os.path.join(snapshot_dir, "out")
    if not os.path.isdir(out_dir):
        return None, "no out/ dir"
    for fn in sorted(os.listdir(out_dir)):
        path = os.path.join(out_dir, fn)
        try:
            with open(path, "r", errors="replace") as f:
                for line in f:
                    m = re.search(r"\bTEST=(\S*)", line)
                    if m:
                        test = [p for p in m.group(1).split(",") if p]
                        return test, None
        except OSError:
            continue
    return None, "no 'TEST=' line in out/"


def best_checkpoint_dir(run_dir):
    """The checkpoint HF Trainer selected as best (metric_for_best_model='pr_auc'
    in train.py), resolved to a directory inside this snapshot."""
    ckpts = sorted((d for d in os.listdir(run_dir) if d.startswith("checkpoint-")),
                   key=lambda d: int(d.split("-")[-1]))
    if not ckpts:
        return None, None, "no checkpoint-* dir"
    chosen, recorded = ckpts[-1], None
    for name in reversed(ckpts):
        state_path = os.path.join(run_dir, name, "trainer_state.json")
        if not os.path.exists(state_path):
            continue
        with open(state_path) as f:
            state = json.load(f)
        best_path = state.get("best_model_checkpoint")
        recorded = state.get("best_metric")
        if best_path:
            cand = os.path.basename(best_path.rstrip("/"))
            if cand in ckpts:
                chosen = cand
        break
    return os.path.join(run_dir, chosen), recorded, None


def pr_auc_from_artifacts(ckpt_dir):
    """Mean per-episode average precision, recomputed exactly as train.py's
    compute_metrics does: sigmoid the logits, average_precision_score per row,
    skipping rows with a single class, then nanmean."""
    needed = ["predictions.pkl", "labels.pkl", "indexes.pkl"]
    missing = [f for f in needed if not os.path.exists(os.path.join(ckpt_dir, f))]
    if missing:
        return None, None, f"missing {missing}"
    with open(os.path.join(ckpt_dir, "predictions.pkl"), "rb") as f:
        preds = pickle.load(f)
    with open(os.path.join(ckpt_dir, "labels.pkl"), "rb") as f:
        labels = pickle.load(f)
    logits = preds[0] if isinstance(preds, tuple) else preds
    y = labels["interaction"] if isinstance(labels, dict) else labels
    probs = 1.0 / (1.0 + np.exp(-logits.astype(np.float64)))

    from sklearn.metrics import average_precision_score
    aps = [average_precision_score(y[i], probs[i])
           for i in range(probs.shape[0]) if len(np.unique(y[i])) > 1]
    if not aps:
        return None, int(probs.shape[0]), "no episode had both classes"
    return float(np.nanmean(aps)), int(probs.shape[0]), None


def collect_pr_auc_join(geometry_df, family):
    families = find_loocv_families(SNAPSHOTS_DIR)
    print(f"Searched {SNAPSHOTS_DIR} for '<timestamp>_loocv_<PID>_<family>' snapshots "
          f"(each holds results/<run>/checkpoint-*/predictions.pkl|labels.pkl|indexes.pkl).")
    if not families:
        print("Found none -- emitting the geometry table alone.")
        return None
    print(f"Found {len(families)} candidate LOPO families "
          f"(participant coverage out of 9):")
    for fam, pid_map in sorted(families.items(), key=lambda kv: (-len(kv[1]), kv[0])):
        mark = "   <-- USING THIS ONE" if fam == family else ""
        print(f"  {len(pid_map)}/9  {fam}{mark}")

    if family not in families:
        print(f"\n'{family}' is not among them -- emitting the geometry table alone.")
        return None

    pid_map = families[family]
    rows = []
    print(f"\nPer-fold PR-AUC for '{family}' (recomputed from the saved artifacts; "
          f"'logged' is the trainer's own best_metric, shown as a cross-check):")
    for pid in PARTICIPANTS:
        if pid not in pid_map:
            rows.append({"participant": pid, "pr_auc": np.nan,
                         "pr_auc_note": "no snapshot for this fold"})
            print(f"  [{pid}] no snapshot")
            continue
        snap = pid_map[pid]
        test_ps, err = fold_test_participants(snap)
        if err or not test_ps:
            rows.append({"participant": pid, "pr_auc": np.nan,
                         "pr_auc_note": f"cannot confirm fold ({err})"})
            print(f"  [{pid}] SKIPPED: cannot confirm held-out participant ({err})")
            continue
        if test_ps != [pid]:
            rows.append({"participant": pid, "pr_auc": np.nan,
                         "pr_auc_note": f"log says TEST={','.join(test_ps)}"})
            print(f"  [{pid}] SKIPPED: dir name says {pid} but the job log says "
                  f"TEST={','.join(test_ps)} -- not joining an ambiguous fold")
            continue

        results_dir = os.path.join(snap, "results")
        run_dirs = ([os.path.join(results_dir, d) for d in sorted(os.listdir(results_dir))
                     if os.path.isdir(os.path.join(results_dir, d))]
                    if os.path.isdir(results_dir) else [])
        if len(run_dirs) != 1:
            note = "no results/ run dir" if not run_dirs else f"{len(run_dirs)} run dirs"
            rows.append({"participant": pid, "pr_auc": np.nan, "pr_auc_note": note})
            print(f"  [{pid}] SKIPPED: {note}")
            continue

        ckpt, recorded, err = best_checkpoint_dir(run_dirs[0])
        if err:
            rows.append({"participant": pid, "pr_auc": np.nan, "pr_auc_note": err})
            print(f"  [{pid}] SKIPPED: {err}")
            continue
        pr_auc, n_eval, err = pr_auc_from_artifacts(ckpt)
        rows.append({
            "participant": pid, "pr_auc": pr_auc if pr_auc is not None else np.nan,
            "pr_auc_logged": recorded, "pr_auc_n_eval_episodes": n_eval,
            "pr_auc_checkpoint": os.path.basename(ckpt), "pr_auc_note": err or "",
        })
        if pr_auc is None:
            print(f"  [{pid}] SKIPPED: {err}")
        else:
            rec = f"{recorded:.5f}" if isinstance(recorded, (int, float)) else "n/a"
            print(f"  [{pid}] TEST={pid}  {os.path.basename(ckpt):>18}  "
                  f"PR-AUC={pr_auc:.5f}  (logged {rec}, {n_eval} eval episodes)")
    return geometry_df.merge(pd.DataFrame(rows), on="participant", how="left")


def check_episode_counts(df):
    """Cross-check: the fold's eval-set size should equal the number of episodes
    this script built for that participant. A mismatch means the run used a
    different loader config, so its PR-AUC is not comparable to these numbers."""
    if "pr_auc_n_eval_episodes" not in df.columns or "n_episodes_total" not in df.columns:
        return
    print("\nFold identification cross-check (eval-set size vs episodes built here "
          f"at obs={OBS_TIME}/anticip={ANTICIP_TIME}/future={FUTURE_TIME}):")
    for _, r in df.iterrows():
        built, evald = r.get("n_episodes_total"), r.get("pr_auc_n_eval_episodes")
        if pd.isna(built) or pd.isna(evald):
            continue
        ok = "match" if int(built) == int(evald) else "MISMATCH -- configs differ"
        print(f"  [{r['participant']}] built {int(built):>5}   eval {int(evald):>5}   {ok}")


# =====================================================================
# Part 3: z-extent spread; rotation provenance / symmetry
# =====================================================================

PROVENANCE = """\
Where the nine rotation angles live
-----------------------------------
They are NOT hardcoded anywhere in this codebase, and they are not computed by
it either. Traced by search:

  * loader.py takes the grid from `voxel_centers_dir` (production sets
    VOXEL_CENTERS_DIR=new_voxel_centers) via `_load_fixed_voxel_centers`, which
    just np.loads `new_voxel_centers/<pid>_voxel_centers.npy`. No rotation code.
  * boxer_env._grid_from_points -- the "AABB of Boxer OBB corners UNION SLAM
    trajectory" construction -- is axis-aligned by construction (np.linspace per
    world axis) and is only reached when voxel_centers_dir is None. It is NOT
    what the production runs used.
  * The .npy files were exported from HD-EPIC's Digital Twin:
      /data/video_datasets/HD-EPIC/Digital-Twin/blenders/Voxels_corrected/<pid>_voxels.blend
      collection "Voxels_Cupboards_16_obj" (P03: "Voxels_Objects_16_obj")
      -> FIction_HD_EPIC/additional/extract_voxel_centers_blend_v2.py
         (run_extract_voxel_centers.sh, or run_extract_voxel_centers_local.py)
    The yaw is part of the Blender object transforms, so it is baked into the
    .npy and never written down as a number.
  * An older, superseded generator, FIction_HD_EPIC/additional/compute_voxel_centers.py,
    hardcodes per-participant min/max bounds "(from Blender output)" but builds
    an AXIS-ALIGNED grid -- it did not produce new_voxel_centers/.
  * The only place any angle appears as text is a comment in
    analysis/count_analysis.py (yaw -24 P04, +35 P07, +24 P08) -- three of nine,
    rounded, in a comment.

So: reproducing the grid requires the .blend files, and the angles themselves
exist only as measurements off the .npy -- which is what the table below is.\
"""


def report_z_extent_spread(df):
    z = df["extent_z_m"].dropna()
    if z.empty:
        print("No usable extent_z_m values.")
        return
    mean, mn, mx = float(z.mean()), float(z.min()), float(z.max())
    lo_pct, hi_pct = (mn - mean) / mean * 100, (mx - mean) / mean * 100
    worst = max(abs(lo_pct), abs(hi_pct))
    print(f"extent_z_m over {len(z)} participants: "
          f"min={mn:.3f}  max={mx:.3f}  mean={mean:.3f}  "
          f"(min {lo_pct:+.1f}%, max {hi_pct:+.1f}% of the mean; max/min = {mx / mn:.2f}x)")
    if worst > 20:
        print(f"  FLAG: spread reaches {worst:.1f}% of the mean, beyond the ~+/-20% "
              f"bar -- a normalised height index does NOT mean the same physical "
              f"height across kitchens, so a decoder prior over z is not comparable "
              f"between participants.")
    else:
        print(f"  Within ~+/-20% of the mean (worst {worst:.1f}%) -- normalised height "
              f"is roughly comparable across kitchens.")


def report_rotation_and_symmetry(df):
    print(PROVENANCE)

    print("\n(a) Recovered yaw per participant (measured off the lattice step vectors):")
    print(f"  {'pid':<6}{'rotation_deg':>14}{'mod 90':>10}   {'residuals (affine m / orth / z-tilt deg)':<44}")
    for _, r in df.iterrows():
        print(f"  {r['participant']:<6}{r['rotation_deg']:>+14.2f}{r['rotation_mod90_deg']:>10.2f}   "
              f"{r.get('affine_err_m', float('nan')):.1e} / "
              f"{r.get('orth_err', float('nan')):.1e} / "
              f"{r.get('z_tilt_deg', float('nan')):.3f}")

    print("\n(b) Which horizontal axis is the long one:")
    n_x = n_y = 0
    print(f"  {'pid':<6}{'extent_x_m':>12}{'extent_y_m':>12}{'longer':>9}{'ratio':>8}")
    for _, r in df.iterrows():
        ex, ey = r["extent_x_m"], r["extent_y_m"]
        longer = "x" if ex >= ey else "y"
        n_x += longer == "x"
        n_y += longer == "y"
        print(f"  {r['participant']:<6}{ex:>12.3f}{ey:>12.3f}{longer:>9}{max(ex, ey) / min(ex, ey):>8.2f}")
    print(f"  -> {n_x} participant(s) x-long, {n_y} y-long.")
    if n_x and n_y:
        print(f"     INCONSISTENT: 'axis 0' is the kitchen's long axis for {n_x} "
              f"participants and its short axis for {n_y}. A model with no positional "
              f"encoding cannot tell these apart, and any prior that assumes 'x is "
              f"the counter run' is right for only {max(n_x, n_y)} of {n_x + n_y}.")
    else:
        print("     CONSISTENT: the long axis is the same lattice axis for all participants.")

    if "occ_centroid_axis0" not in df.columns:
        print("\n(c) Occupancy centroids unavailable (ran with --geometry-only).")
        return
    print("\n(c) Centroid of Boxer-occupied voxels, as a fraction of the grid "
          "(0 = axis start, 1 = axis end):")
    print(f"  {'pid':<6}{'axis0 (x)':>12}{'axis1 (y)':>12}{'axis2 (z)':>12}")
    for _, r in df.iterrows():
        print(f"  {r['participant']:<6}{r['occ_centroid_axis0']:>12.3f}"
              f"{r['occ_centroid_axis1']:>12.3f}{r['occ_centroid_axis2']:>12.3f}")
    for axis, name in ((0, "axis0"), (1, "axis1")):
        vals = df[f"occ_centroid_axis{axis}"].dropna().to_numpy()
        if len(vals) < 2:
            continue
        print(f"  {name}: min={vals.min():.3f} max={vals.max():.3f} "
              f"mean={vals.mean():.3f} sd={vals.std(ddof=1):.3f} "
              f"(spread {vals.max() - vals.min():.3f})")
    print("  A tight cluster near a common value would mean the exporter used a "
          "consistent axis orientation; values scattered across [0,1] mean the sign "
          "of each horizontal axis is arbitrary between kitchens, so voxel index i "
          "does not denote the same side of the room for two participants.")


# =====================================================================
# Correlations
# =====================================================================

def spearman(x, y):
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    n = int(mask.sum())
    if n < 3:
        return None, None, n
    try:
        from scipy.stats import spearmanr
        rho, p = spearmanr(x, y)
        return float(rho), float(p), n
    except ImportError:
        rx, ry = pd.Series(x).rank().to_numpy(), pd.Series(y).rank().to_numpy()
        return float(np.corrcoef(rx, ry)[0, 1]), None, n


def report_correlation(df, xcol, ycol, expectation):
    if xcol not in df.columns or ycol not in df.columns:
        print(f"\n{xcol} vs {ycol}: unavailable.")
        return
    rho, p, n = spearman(df[xcol], df[ycol])
    print(f"\nSpearman({xcol}, {ycol}) = " +
          (f"{rho:+.3f}" if rho is not None else "n/a") +
          (f"  (p={p:.4f}, n={n})" if p is not None else f"  (n={n})") +
          f"   [expected: {expectation}]")
    if rho is None:
        print(f"  Not enough finite pairs (n={n}, need >=3).")
        return
    sub = df[["participant", xcol, ycol]].dropna().sort_values(xcol)
    print(f"  {'pid':<6}{xcol:>22}{ycol:>22}")
    for _, r in sub.iterrows():
        print(f"  {r['participant']:<6}{r[xcol]:>22.6g}{r[ycol]:>22.6g}")


# =====================================================================
# main
# =====================================================================

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--family", default=DEFAULT_FAMILY,
                    help="LOPO snapshot family to join PR-AUC from (default: %(default)s)")
    ap.add_argument("--list-families-only", action="store_true",
                    help="List candidate LOPO families under snapshots/ and exit.")
    ap.add_argument("--geometry-only", action="store_true",
                    help="Skip episode building (no prevalence / fill_fraction).")
    args = ap.parse_args()

    if args.list_families_only:
        families = find_loocv_families(SNAPSHOTS_DIR)
        print(f"{len(families)} candidate LOPO families under {SNAPSHOTS_DIR}:")
        for fam, pid_map in sorted(families.items(), key=lambda kv: (-len(kv[1]), kv[0])):
            print(f"  {len(pid_map)}/9  {fam}  {sorted(pid_map)}")
        return

    os.makedirs(OUT_DIR, exist_ok=True)

    print("=" * 78)
    print("GRID GEOMETRY AUDIT")
    print(f"  grid source : {VOXEL_CENTERS_DIR} (pre-baked per-participant 16^3 centres)")
    print(f"  episodes    : obs={OBS_TIME}s anticip={ANTICIP_TIME}s future={FUTURE_TIME}s, "
          f"TIME_ANCHOR=gaze")
    print(f"  subsample   : {N_SAMPLE_EPISODES} episodes/participant, seed={SEED}")
    print("=" * 78)

    df = collect_geometry_table(geometry_only=args.geometry_only)

    print("\n" + "=" * 78)
    print("TABLE 1: grid geometry + label prevalence + fill_fraction")
    print("=" * 78)
    show = [c for c in ["participant", "n_videos", "extent_x_m", "extent_y_m", "extent_z_m",
                        "rotation_deg", "voxel_edge_x_m", "voxel_edge_y_m", "voxel_edge_z_m",
                        "voxel_volume_m3", "mean_pos_rate", "mean_pos_count", "n_episodes",
                        "fill_fraction"] if c in df.columns]
    with pd.option_context("display.width", 200, "display.max_columns", None):
        print(df[show].to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    if "fill_fraction_strict" in df.columns:
        print("\nfill_fraction detail (strict = only points genuinely inside the box):")
        print(df[["participant", "fill_fraction", "fill_fraction_strict",
                  "fill_fraction_boxer", "frac_points_outside_grid",
                  "n_occupancy_points"]].to_string(index=False,
                                                   float_format=lambda v: f"{v:.4f}"))
        ff = df["fill_fraction"].dropna()
        if len(ff):
            print(f"  spread: min={ff.min():.3f} ({df.loc[ff.idxmin(), 'participant']})  "
                  f"max={ff.max():.3f} ({df.loc[ff.idxmax(), 'participant']})  "
                  f"mean={ff.mean():.3f}  sd={ff.std(ddof=1):.3f}")
    if "data_notes" in df.columns:
        notes = df[df["data_notes"].fillna("").astype(bool)]
        if len(notes):
            print("\nData-quality notes (these rows are kept, not excluded):")
            for _, r in notes.iterrows():
                print(f"  [{r['participant']}] {r['data_notes']}")

    print("\n" + "=" * 78)
    print("TABLE 2: LOPO held-out PR-AUC join")
    print("=" * 78)
    joined = collect_pr_auc_join(df, args.family) if not args.geometry_only else None
    out_df = joined if joined is not None else df
    out_df.to_csv(CSV_PATH, index=False)

    if joined is not None:
        check_episode_counts(joined)
        cols = [c for c in ["participant", "voxel_volume_m3", "fill_fraction",
                            "mean_pos_rate", "mean_pos_count", "pr_auc", "pr_auc_note"]
                if c in joined.columns]
        print("\nJoined table:")
        print(joined[cols].to_string(index=False, float_format=lambda v: f"{v:.5f}"))

    print("\n" + "=" * 78)
    print("CORRELATIONS (Spearman across the 9 participants)")
    print("=" * 78)
    report_correlation(df, "voxel_volume_m3", "mean_pos_rate", "strongly negative")
    report_correlation(df, "fill_fraction", "mean_pos_rate", "positive if fit drives prevalence")
    report_correlation(df, "fill_fraction", "voxel_volume_m3", "negative if a poor fit inflates voxels")
    if joined is not None:
        report_correlation(joined, "mean_pos_rate", "pr_auc", "positive")
        report_correlation(joined, "voxel_volume_m3", "pr_auc", "negative if geometry drives PR-AUC")
        report_correlation(joined, "fill_fraction", "pr_auc", "positive if fit drives PR-AUC")
    else:
        print("\nNo PR-AUC join -- skipping every correlation involving pr_auc.")

    print("\n" + "=" * 78)
    print("TABLE 3a: z-extent spread across kitchens")
    print("=" * 78)
    report_z_extent_spread(df)

    print("\n" + "=" * 78)
    print("TABLE 3b: rotation provenance and residual symmetry")
    print("=" * 78)
    report_rotation_and_symmetry(df)

    print(f"\nwrote {CSV_PATH}")


if __name__ == "__main__":
    main()
