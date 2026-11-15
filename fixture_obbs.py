"""Static fixture OBBs from the HD-EPIC Digital-Twin meshes, plus the two
filters that use them.

Why this exists
---------------
The fused Boxer boxes are per-window detections: the kitchen's *furniture*
(counters, cupboards, the hob, the sink) is re-detected in every window, with
jittering geometry and an unstable class label, and it competes for OBB token
slots with the movable objects that actually move. The Digital-Twin meshes give
the same furniture once, exactly, per kitchen. So:

  * replace redundant furniture detections with the modelled fixture, and
  * drop detections that fall outside the participant's voxel grid entirely,
    which the token encoder cannot represent anyway.

Frame
-----
**The meshes are already in the per-recording Aria world frame** -- the same
frame as ``boxer_3dbbs.csv``'s ``t*_world_object`` and as
``new_voxel_centers/<pid>_voxel_centers.npy``. No transform is applied. Verified
two ways before this module was written:

  * extent: P01's mesh vertices span [-2.40,-4.35,-1.53]..[0.21,-0.52,0.67]
    against a grid-centre span of [-2.37,-4.11,-1.46]..[0.39,0.04,0.60].
  * co-location by semantics, across four participants: the median fused-Boxer
    centre of ``sink`` lands 0.04 m (P01), 0.12 m (P04), 0.04 m (P07), 0.04 m
    (P08) from the nearest ``sink`` fixture; ``microwave_oven`` 0.03/0.03/0.10 m
    from ``microwave``; ``stove`` 0.04/0.21/-/0.10 m from ``hob``.

If either check stops holding (new meshes, a re-exported grid), this module is
wrong and must not be used -- re-run ``python fixture_obbs.py --verify``.

Box convention
--------------
Identical to what ``iou_fusion`` emits and ``iou_fusion.box_iou3d`` consumes:
``(cx, cy, cz, hx, hy, hz, yaw)`` -- upright (yaw-only about z), **half**
extents. Every HD-EPIC Boxer OBB is yaw-only, and the fixture meshes are too
(363 of 422 are exact 8-vertex boxes; the rest are upright detailed shapes), so
a yaw-only fit is the matching convention rather than a lossy approximation.
The xy footprint is fitted with a minimum-area enclosing rectangle (rotating
calipers over the convex hull), which recovers an 8-vertex box exactly.
"""

from __future__ import annotations

import glob
import os
import re

import numpy as np

from iou_fusion import box_iou3d, load_class_embeddings
from object_class_vocab import OOV_CLASS_IDX, _get_vocab

# Overridable from the environment; see loader.py's path-constant block for why.
HDEPIC_ROOT = os.environ.get("HDEPIC_ROOT", "/data/video_datasets/HD-EPIC")
MESH_ROOT = os.environ.get("MESH_ROOT", f"{HDEPIC_ROOT}/Digital-Twin/blenders/meshes")
# Same value boxer_env.py resolves, repeated here rather than imported: it is used
# only by the __main__ diagnostic below, and importing boxer_env for it would add
# a module dependency the library path does not need.
BOXER_OBBS_ROOT = os.environ.get("BOXER_OBBS_ROOT", f"{HDEPIC_ROOT}/boxer_obbs")
CLASS_EMBEDDINGS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                     "class_embeddings.npz")

GRID_SIZE = 16

# Defaults for the fixture-overlap filter (see the module docstring of
# submit_loocv_fixture_obb.sh for the justification, and --report for the
# measured effect of each).
DEFAULT_IOU_THRESH = 0.4
DEFAULT_COS_THRESH = 0.7

# Fixture classes excluded entirely -- no token, no filtering, no counting.
#
# Both are room-scale slabs standing in for the kitchen's shell rather than for
# an object in it. Measured: one `surface` per participant at 14.0-40.1 m^3 (the
# whole room), and a `floor` alongside it. As a single OBB token each would carry
# the top volume bin, a centre 0.22-0.51 m from the mesh's own centroid, and no
# class -- ~2 tokens per window per kitchen that say nothing about where an
# interaction will happen. They are dropped before the voxel-grid test, so they
# never appear in any per-stage count either.
FIXTURE_DROP_CLASSES = frozenset({'floor', 'surface'})

# Digital-Twin fixture class -> the LVIS/OWLv2 name that carries its embedding.
#
# The mesh names are the Digital-Twin's own vocabulary (British-English, unspaced
# compounds, and a `top_X` convention meaning "the top surface of X"); the
# embedding table in class_embeddings.npz is LVIS-plus. 20 of the 31 fixture
# classes have no literal entry there, so they are aliased here.
#
# A fixture class NOT in this map keeps its geometry -- it still becomes a token
# and still prunes on the voxel grid -- but has no embedding, so clause (b) of
# the overlap filter can never hold and it never discards a fused box. That is
# the conservative direction: an unmapped fixture adds, it never removes.
FIXTURE_CLASS_ALIAS = {
    # literal matches, listed for completeness
    'basket': 'basket', 'cupboard': 'cupboard', 'dishwasher': 'dishwasher',
    'drawer': 'drawer', 'hook': 'hook', 'oven': 'oven', 'radiator': 'radiator',
    'shelf': 'shelf', 'sink': 'sink', 'storage': 'storage', 'table': 'table',
    # unspaced compounds / British-English synonyms
    'bin': 'trash_can',
    'boiler': 'water_heater',
    'extractor_fan': 'fan',
    'extractorfan': 'fan',
    'freezer': 'refrigerator',
    'fridge': 'refrigerator',
    'fridgefreezer': 'refrigerator',
    'hob': 'stove',
    'microwave': 'microwave_oven',
    'washingmachine': 'automatic_washer',
    'windowsill': 'window',
    # `top_X` = the top surface of X; same semantics for filtering purposes
    'top_cupboard': 'cupboard',
    'top_drawer': 'drawer',
    'top_freezer': 'refrigerator',
    'top_fridge': 'refrigerator',
    'top_microwave': 'microwave_oven',
    'top_storage': 'storage',
    # -- judgment calls, flagged --------------------------------------------
    # LVIS has no "countertop"/"worktop". `kitchen_table` is what Boxer itself
    # assigns to counter surfaces in this dataset (7,912 boxes in a 31-video
    # sample, the most common furniture label after cupboard/stove/sink), so
    # aliasing here is what lets a counter fixture absorb those detections.
    # This is the one alias with large consequences -- `counter` is the most
    # numerous fixture class. Drop the entry to disable it.
    'counter': 'kitchen_table',
    # `floor` and `surface` are not listed: they are dropped outright by
    # FIXTURE_DROP_CLASSES above, so they never need an embedding.
}


# =====================================================================
# Yaw-only OBB fitting
# =====================================================================

def _read_obj_vertices(path):
    """The 'v x y z' lines of a Wavefront OBJ as an (N, 3) float array."""
    out = []
    with open(path) as fh:
        for line in fh:
            if line.startswith('v '):
                out.append([float(x) for x in line.split()[1:4]])
    return np.asarray(out, dtype=np.float64)


def _convex_hull_2d(pts):
    """Andrew's monotone chain hull, counter-clockwise."""
    pts = np.unique(np.round(pts, 9), axis=0)
    pts = pts[np.lexsort((pts[:, 1], pts[:, 0]))]
    if len(pts) <= 2:
        return pts

    def _half(seq):
        out = []
        for p in seq:
            while len(out) >= 2:
                a, b = out[-2], out[-1]
                if (b[0]-a[0])*(p[1]-a[1]) - (b[1]-a[1])*(p[0]-a[0]) <= 0:
                    out.pop()
                else:
                    break
            out.append(p)
        return out

    lower, upper = _half(pts), _half(pts[::-1])
    return np.asarray(lower[:-1] + upper[:-1])


def _min_area_rect(pts):
    """Minimum-area enclosing rectangle of 2D points -> (cx, cy, hx, hy, yaw).

    Rotating calipers: the minimum-area rectangle is flush with a hull edge, so
    trying every edge direction is exact. Recovers an axis-aligned or rotated
    8-vertex box exactly (verified in --verify).
    """
    hull = _convex_hull_2d(pts)
    if len(hull) < 3:
        lo, hi = pts.min(0), pts.max(0)
        c = (lo + hi) / 2.0
        return c[0], c[1], (hi[0]-lo[0])/2.0, (hi[1]-lo[1])/2.0, 0.0

    best = None
    for i in range(len(hull)):
        edge = hull[(i + 1) % len(hull)] - hull[i]
        n = float(np.hypot(edge[0], edge[1]))
        if n < 1e-12:
            continue
        c, s = edge[0] / n, edge[1] / n
        rot = np.array([[c, s], [-s, c]])          # world -> edge frame
        q = pts @ rot.T
        lo, hi = q.min(0), q.max(0)
        area = (hi[0] - lo[0]) * (hi[1] - lo[1])
        if best is None or area < best[0]:
            ctr = rot.T @ ((lo + hi) / 2.0)
            best = (area, ctr[0], ctr[1],
                    (hi[0]-lo[0])/2.0, (hi[1]-lo[1])/2.0, float(np.arctan2(s, c)))

    _, cx, cy, hx, hy, yaw = best
    # A rectangle is invariant under a half turn, so fold yaw into (-pi/2, pi/2]
    # WITHOUT swapping the extents (a quarter turn would swap them; pi does not).
    yaw = (yaw + np.pi / 2.0) % np.pi - np.pi / 2.0
    return cx, cy, hx, hy, yaw


def mesh_to_obb(path):
    """Yaw-only OBB of one mesh as (cx, cy, cz, hx, hy, hz, yaw)."""
    v = _read_obj_vertices(path)
    if len(v) == 0:
        return None
    cx, cy, hx, hy, yaw = _min_area_rect(v[:, :2])
    z_lo, z_hi = float(v[:, 2].min()), float(v[:, 2].max())
    return np.array([cx, cy, (z_lo + z_hi) / 2.0, hx, hy,
                     (z_hi - z_lo) / 2.0, yaw], dtype=np.float64)


def fixture_class_from_filename(path, pid):
    """'…/P01_counter.009.obj' -> 'counter'.

    The class is the token between the participant prefix and the numeric
    suffix; a mesh with no numeric suffix ('P01_sink.obj') is handled too.
    """
    stem = os.path.basename(path)
    stem = stem[:-4] if stem.endswith('.obj') else stem
    if not stem.startswith(pid + '_'):
        return None
    stem = stem[len(pid) + 1:]
    return re.split(r'\.\d+$', stem)[0].split('.')[0]


# =====================================================================
# Voxel-grid membership
# =====================================================================

class VoxelGrid:
    """The participant's 16x16x16 grid as an oriented box, with a membership test.

    The production grids are Digital-Twin exports carrying a manual yaw about z,
    so they are NOT axis-aligned in the Aria world frame. Measured over all nine
    files the lattice is exactly regular and orthogonal (max residual 3.6e-7 m,
    float32 noise; every pairwise axis dot product < 4e-9), so the grid is fully
    described by an origin plus three orthogonal step vectors.

    Membership rule
    ---------------
    A point is INSIDE iff, expressed in fractional lattice coordinates,
    ``-0.5 <= u <= GRID_SIZE-0.5`` on all three axes.

    That is simultaneously
      * "the nearest voxel centre is within half the pitch on every axis", and
      * "the point lies inside the grid's oriented bounding box"
    -- the two formulations coincide exactly here because the lattice is regular
    and orthogonal, so no choice between them is needed. Stated in fractional
    coordinates because the pitch is anisotropic and differs per participant
    (measured 0.126-0.417 m along the floor axes, 0.134-0.146 m in z), so a
    single scalar distance tolerance would be wrong on every kitchen.

    This is a real test, unlike ``boxer_env._nearest_voxel_ijk``, which returns
    the argmin over the 4096 centres and therefore snaps an arbitrarily distant
    point onto a boundary voxel with no notion of being outside.
    """

    def __init__(self, centers):
        c = np.asarray(centers, dtype=np.float64).reshape(GRID_SIZE, GRID_SIZE,
                                                          GRID_SIZE, 3)
        self.centers = c
        self.origin = c[0, 0, 0]
        steps = np.stack([
            (c[1:, :, :, :] - c[:-1, :, :, :]).reshape(-1, 3).mean(0),
            (c[:, 1:, :, :] - c[:, :-1, :, :]).reshape(-1, 3).mean(0),
            (c[:, :, 1:, :] - c[:, :, :-1, :]).reshape(-1, 3).mean(0),
        ])                                            # (3, 3), row a = step along axis a
        self.steps = steps
        self.pitch = np.linalg.norm(steps, axis=1)    # (3,) metres per index step
        # Fractional coordinates: u = M @ (p - origin), with M the pseudo-inverse
        # of the step matrix. Orthogonal steps make this a plain scaled rotation.
        self._to_frac = np.linalg.pinv(steps.T)       # (3, 3)

    @classmethod
    def load(cls, voxel_centers_dir, pid):
        path = os.path.join(voxel_centers_dir, f"{pid}_voxel_centers.npy")
        return cls(np.load(path))

    def fractional(self, points):
        """(N, 3) world points -> (N, 3) fractional lattice coordinates."""
        p = np.atleast_2d(np.asarray(points, dtype=np.float64))
        return (p - self.origin) @ self._to_frac.T

    def contains(self, points):
        """(N,) bool -- see the class docstring for the rule."""
        u = self.fractional(points)
        return np.all((u >= -0.5) & (u <= GRID_SIZE - 0.5), axis=1)


# =====================================================================
# Fixtures per participant
# =====================================================================

_FIXTURE_CACHE = {}       # (pid, voxel_centers_dir) -> list[dict]
_EMB_CACHE = None


def _embeddings():
    """{lowercased LVIS name -> unit vector}. The table stores original LVIS
    casing ('CD_player'), the Boxer CSV and boxer_class_vocab.json lowercase, so
    both sides are lowercased on lookup."""
    global _EMB_CACHE
    if _EMB_CACHE is None:
        raw = load_class_embeddings(CLASS_EMBEDDINGS_PATH)
        _EMB_CACHE = {k.lower(): v for k, v in raw.items()}
    return _EMB_CACHE


def _class_index(lvis_name):
    """class_embedding row for an LVIS name: sem_id+1, or OOV for an unmapped
    fixture class (the same convention env/obb/trajectory tokens already use)."""
    if lvis_name is None:
        return OOV_CLASS_IDX
    sem_id = _get_vocab().match(lvis_name)
    return sem_id + 1 if sem_id is not None else OOV_CLASS_IDX


def load_fixtures(pid, voxel_centers_dir, mesh_root=MESH_ROOT, verbose=False):
    """Every fixture OBB of one participant that survives the voxel-grid prune.

    Static geometry, so this is computed once per participant and cached at
    module level for the process. Returns a list of dicts shaped like a fused
    box, so downstream code can treat the two uniformly::

        {'box': (7,) float64 (cx,cy,cz,hx,hy,hz,yaw),
         'center': (3,), 'half_extents': (3,), 'yaw': float,
         'name': 'counter',            # Digital-Twin class
         'lvis': 'kitchen_table',      # aliased LVIS name, or None
         'sem_id': int,                # class_embedding row (sem_id+1 / OOV)
         'prob': 1.0,
         'fixture': True}
    """
    key = (pid, os.path.abspath(voxel_centers_dir))
    if key in _FIXTURE_CACHE:
        return _FIXTURE_CACHE[key]

    grid = VoxelGrid.load(voxel_centers_dir, pid)
    paths = sorted(glob.glob(os.path.join(mesh_root, pid, "*.obj")))
    kept, dropped, excluded = [], 0, 0
    for path in paths:
        cls = fixture_class_from_filename(path, pid)
        if cls is None:
            continue
        if cls in FIXTURE_DROP_CLASSES:
            excluded += 1
            continue
        box = mesh_to_obb(path)
        if box is None:
            continue
        if not bool(grid.contains(box[:3])[0]):
            dropped += 1
            continue
        lvis = FIXTURE_CLASS_ALIAS.get(cls)
        kept.append({
            'box': box,
            'center': box[:3].copy(),
            'half_extents': box[3:6].copy(),
            'yaw': float(box[6]),
            'name': cls,
            'lvis': lvis,
            'sem_id': _class_index(lvis),
            'prob': 1.0,
            'fixture': True,
        })

    if verbose:
        print(f"[fixtures] {pid}: {len(kept)} of {len(paths)} meshes kept "
              f"({excluded} excluded by class {sorted(FIXTURE_DROP_CLASSES)}, "
              f"{dropped} centres outside the voxel grid)", flush=True)
    _FIXTURE_CACHE[key] = kept
    return kept


# =====================================================================
# The two filters
# =====================================================================

def _fused_to_box(b):
    """A cached fused box -> the (cx,cy,cz,hx,hy,hz,yaw) box_iou3d wants.

    ``iou_fusion`` stores a true ``yaw`` alongside the quaternion; fall back to
    deriving it from the quaternion's z component for any box that lacks one.
    """
    yaw = b.get('yaw')
    if yaw is None:
        qw, _, _, qz = b['quat']
        yaw = float(np.arctan2(2.0 * qw * qz, 1.0 - 2.0 * qz * qz))
    c, h = np.asarray(b['center'], float), np.asarray(b['half_extents'], float)
    return np.array([c[0], c[1], c[2], h[0], h[1], h[2], float(yaw)])


def prune_outside_grid(fused, grid):
    """3a -- drop fused boxes whose centre is outside the participant's grid."""
    if not fused:
        return []
    inside = grid.contains(np.stack([np.asarray(b['center'], float) for b in fused]))
    return [b for b, ok in zip(fused, inside) if ok]


def class_candidate_index(fixtures, cos_thresh=DEFAULT_COS_THRESH):
    """{boxer class name -> [fixture boxes similar enough to filter it]}.

    Clause (b) depends only on the two class names, never on geometry, so it is
    resolved once per participant instead of once per box. That matters: the
    loader runs this over ~26k windows x ~124 boxes, and the cosine gate rejects
    the overwhelming majority of (box, fixture) pairs -- the fused caches are
    dominated by movable objects (bottle, cup, bowl, knife) that resemble no
    fixture -- so pre-resolving it removes almost every IoU evaluation.

    Built lazily per class name, since a split touches far fewer Boxer names
    than the 1,220 in the table.
    """
    emb = _embeddings()
    fx = [(f, emb.get((f['lvis'] or '').lower()), _fused_to_box(f))
          for f in fixtures]
    fx = [(f, v, box) for f, v, box in fx if v is not None]
    cache = {}

    def candidates(name):
        key = str(name).lower()
        if key not in cache:
            vec = emb.get(key)
            cache[key] = ([] if vec is None else
                          [box for _, v, box in fx
                           if float(np.dot(vec, v)) > cos_thresh])
        return cache[key]

    return candidates


def filter_by_fixtures(fused, fixtures, iou_thresh=DEFAULT_IOU_THRESH,
                       cos_thresh=DEFAULT_COS_THRESH, boxer_name_key='name',
                       candidates=None):
    """3b -- drop a fused box iff SOME single fixture satisfies both clauses.

    Both clauses must hold **for the same fixture**: 3D IoU above ``iou_thresh``
    AND cosine similarity of the two class embeddings above ``cos_thresh``. A
    box whose class has no embedding, or a fixture whose class has no alias, can
    never satisfy (b), so it is kept -- unmapped classes never remove anything.

    Pass ``candidates`` (from ``class_candidate_index``) to reuse the class gate
    across windows.
    """
    if not fused or not fixtures:
        return list(fused), 0
    if candidates is None:
        candidates = class_candidate_index(fixtures, cos_thresh=cos_thresh)

    kept, n_dropped = [], 0
    for b in fused:
        cand = candidates(b.get(boxer_name_key, ''))
        if not cand:
            kept.append(b)
            continue
        box = _fused_to_box(b)
        if any(box_iou3d(box, fbox) > iou_thresh for fbox in cand):
            n_dropped += 1
        else:
            kept.append(b)
    return kept, n_dropped


def build_window_boxes(fused, grid, fixtures, iou_thresh=DEFAULT_IOU_THRESH,
                       cos_thresh=DEFAULT_COS_THRESH, candidates=None):
    """The full step-4 pipeline for one window.

        fused -> 3a voxel-grid prune -> 3b fixture/class filter
              -> append the (already pruned) fixtures

    The cap is NOT applied here: ``boxer_env._obb_tokens_from_fused`` applies it,
    so that truncation stays in one place. Fixtures carry ``prob = 1.0`` so the
    cap's confidence sort keeps them ahead of any detection.

    Returns ``(boxes, stats)``.
    """
    n_in = len(fused or [])
    after_grid = prune_outside_grid(fused or [], grid)
    after_fx, n_fx_dropped = filter_by_fixtures(
        after_grid, fixtures, iou_thresh=iou_thresh, cos_thresh=cos_thresh,
        candidates=candidates)
    boxes = after_fx + list(fixtures)
    return boxes, {
        'n_in': n_in,
        'n_after_grid': len(after_grid),
        'n_grid_pruned': n_in - len(after_grid),
        'n_after_fixture_filter': len(after_fx),
        'n_fixture_filtered': n_fx_dropped,
        'n_fixtures_added': len(fixtures),
        'n_out': len(boxes),
    }


# =====================================================================
# Self-check / reporting entry point
# =====================================================================

def _verify(voxel_centers_dir):
    """Assert the facts this module depends on. Run after any data change."""
    import pickle

    print("=== lattice regularity + orthogonality ===")
    for pid in [f"P0{i}" for i in range(1, 10)]:
        g = VoxelGrid.load(voxel_centers_dir, pid)
        c = g.centers
        res = max(
            np.abs((c[1:, :, :, :] - c[:-1, :, :, :]).reshape(-1, 3) - g.steps[0]).max(),
            np.abs((c[:, 1:, :, :] - c[:, :-1, :, :]).reshape(-1, 3) - g.steps[1]).max(),
            np.abs((c[:, :, 1:, :] - c[:, :, :-1, :]).reshape(-1, 3) - g.steps[2]).max())
        dots = [abs(float(np.dot(g.steps[a], g.steps[b]))) for a, b in ((0, 1), (0, 2), (1, 2))]
        u = g.fractional(c.reshape(-1, 3))
        idx = np.stack(np.meshgrid(*[np.arange(GRID_SIZE)] * 3, indexing='ij'), -1).reshape(-1, 3)
        rt = np.abs(u - idx).max()
        assert res < 1e-5 and max(dots) < 1e-6 and rt < 1e-4, pid
        assert g.contains(c.reshape(-1, 3)).all(), f"{pid}: own centres not inside"
        print(f"  {pid}: pitch={np.round(g.pitch, 4)} residual={res:.1e} "
              f"max|dot|={max(dots):.1e} round-trip={rt:.1e}  OK")

    print("\n=== 8-vertex box recovery (fit must be exact) ===")
    worst = 0.0
    for pid in [f"P0{i}" for i in range(1, 10)]:
        for path in sorted(glob.glob(os.path.join(MESH_ROOT, pid, "*.obj"))):
            v = _read_obj_vertices(path)
            if len(v) != 8:
                continue
            box = mesh_to_obb(path)
            c, s = np.cos(box[6]), np.sin(box[6])
            rot = np.array([[c, s], [-s, c]])
            q = (v[:, :2] - box[:2]) @ rot.T
            err = max(abs(np.abs(q[:, 0]).max() - box[3]),
                      abs(np.abs(q[:, 1]).max() - box[4]))
            worst = max(worst, err)
    print(f"  worst half-extent error over all 8-vertex meshes: {worst:.2e} m")
    assert worst < 1e-6

    print("\n=== fixture inventory (dropped classes + voxel-grid prune) ===")
    for pid in [f"P0{i}" for i in range(1, 10)]:
        n_all = len(glob.glob(os.path.join(MESH_ROOT, pid, "*.obj")))
        fx = load_fixtures(pid, voxel_centers_dir)
        n_emb = sum(1 for f in fx if f['lvis'] and f['lvis'].lower() in _embeddings())
        print(f"  {pid}: {len(fx)}/{n_all} kept, {n_emb} with an embedding "
              f"(can filter), {len(fx) - n_emb} geometry-only")

    print("\n=== frame co-location (fused Boxer vs fixture, same participant) ===")
    pairs = {'sink': 'sink', 'microwave_oven': 'microwave', 'stove': 'hob',
             'dishwasher': 'dishwasher', 'cupboard': 'cupboard'}
    for pid in ["P01", "P04", "P06", "P07", "P08"]:
        caches = sorted(glob.glob(
            os.path.join(BOXER_OBBS_ROOT, f"{pid}-*", "fused_obbs_iou.pkl")))
        if not caches:
            continue
        blob = pickle.load(open(caches[0], 'rb'))
        seen = {}
        for w in blob['windows'].values():
            for b in (w or []):
                seen.setdefault(b['name'], []).append(b['center'])
        fx = load_fixtures(pid, voxel_centers_dir)
        out = []
        for bname, fname in pairs.items():
            cand = [f for f in fx if f['name'] == fname]
            if bname not in seen or not cand:
                continue
            bc = np.median(np.stack(seen[bname]), axis=0)
            d = min(float(np.linalg.norm(f['center'] - bc)) for f in cand)
            out.append(f"{bname}->{fname} {d:.2f}m")
        print(f"  {pid}: " + "  ".join(out))


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--voxel_centers_dir',
                    default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                         'new_voxel_centers'))
    ap.add_argument('--verify', action='store_true')
    args = ap.parse_args()
    if args.verify:
        _verify(args.voxel_centers_dir)
    else:
        for pid in [f"P0{i}" for i in range(1, 10)]:
            load_fixtures(pid, args.voxel_centers_dir, verbose=True)
