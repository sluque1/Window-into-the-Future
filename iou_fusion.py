"""IoU-based fusion of Boxer OBBs, as a drop-in alternative to ``_fuse_boxes``.

``boxer_env._fuse_boxes`` clusters by snapping centres to a ``cluster_distance``
grid and grouping by ``(name, cell)``. That has no notion of overlap, so one
object whose per-frame centre jitters across a cell boundary becomes two boxes
(the "doormat"/"stove" duplicates), while two distinct objects inside one cell
are merged. This module replaces the clustering step with the approach the
BoxerNet repo uses (``utils/fuse_3d_boxes.py``): link detections that actually
overlap, chain the links, and average each connected group.

Pipeline (per observation window, any length):

  1. filter    drop prob < conf_threshold and the SKIP_OBJECT_NAMES classes
  2. pair      for each class, find detection pairs with 3D IoU >= iou_threshold
  3. chain     union-find over those pairs -> connected components ("chaining":
               A-B and B-C put A, B and C in one group even if A and C do not
               overlap)
  4. fuse      confidence-weighted average of centre, size and probability,
               with 90-degree size alignment and pi-periodic yaw averaging
  5. filter    drop groups supported by fewer than min_detections detections
  6. nms       optionally drop fused boxes that still overlap heavily

Two deliberate differences from the BoxerNet implementation:

*Exact IoU instead of sampled.* Every Boxer OBB in HD-EPIC is yaw-only
(``qx = qy = 0`` in all 329,644 rows of P08 -- BoxerNet emits upright boxes with
a tanh-limited yaw), so the 3D IoU factorises exactly into a rotated-rectangle
intersection in xy times an interval overlap in z. The repo samples 8^3 points
per box because it must handle arbitrary rotations; here the closed form is both
exact and faster.

*Class equality instead of text embeddings.* The repo gates merges on the cosine
similarity of OWLv2 label embeddings (>= 0.7, which blocked 46% of candidate
merges on a 5-minute slice). Requiring the class name to match is stricter, needs
no 881MB checkpoint, and matches what ``_fuse_boxes`` already assumes by grouping
on ``name``. It will not merge a "cabinet" detection with a "cupboard" one; pass
``aliases`` if you want specific classes treated as equivalent.

Scale convention
----------------
``scale_x/y/z`` in boxer_3dbbs.csv is the box's FULL size: the CSV writer stores
``bb3_diagonal = bb3_max - bb3_min`` (boxer/utils/file_io.py), and the medians
agree -- refrigerator 0.64 x 0.66 x 2.11 m, mug 0.10 x 0.10 x 0.10 m. Note that
``boxer_env._fuse_boxes`` puts those values into ``half_extents`` unchanged,
which makes every voxelised box twice too large per axis. This module treats
them as full sizes and returns true half-extents; pass ``legacy_half=True`` to
reproduce the existing behaviour instead.

Usage as a library:

    from iou_fusion import fuse_boxes_iou
    fused = fuse_boxes_iou(window_df, conf_threshold=0.5, iou_threshold=0.3,
                           min_detections=2)

Usage inside the model (leaves boxer_env.py untouched):

    from iou_fusion import BoxerEnvBuilderIoU
    builder = BoxerEnvBuilderIoU(debug=..., orphan_fallback=...,
                                 iou_threshold=0.3)

Usage from the shell:

    python iou_fusion.py --video P08-20240614-085000 --start 13.6 --end 43.6 \
        --t0 92.442354 --compare
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from boxer_env import SKIP_OBJECT_NAMES, BoxerEnvBuilder, _fuse_boxes

POS_COLS = ["tx_world_object", "ty_world_object", "tz_world_object"]
SCALE_COLS = ["scale_x", "scale_y", "scale_z"]


# =====================================================================
# Exact IoU for yaw-only (upright) boxes
# =====================================================================

def _rect_corners(cx, cy, hx, hy, yaw):
    """The 4 corners of a yaw-rotated rectangle, counter-clockwise."""
    c, s = np.cos(yaw), np.sin(yaw)
    local = np.array([[-hx, -hy], [hx, -hy], [hx, hy], [-hx, hy]])
    return np.stack([cx + c * local[:, 0] - s * local[:, 1],
                     cy + s * local[:, 0] + c * local[:, 1]], axis=1)


def _cross2(u, v):
    """2D scalar cross product (np.cross on 2-vectors is deprecated in numpy 2)."""
    return u[0] * v[1] - u[1] * v[0]


def _clip_convex(subject, clipper):
    """Sutherland-Hodgman: intersection polygon of two convex polygons."""
    out = subject
    n = len(clipper)
    for i in range(n):
        if len(out) == 0:
            return out
        a, b = clipper[i], clipper[(i + 1) % n]
        edge = b - a
        # inside = left of the directed edge a->b (clipper is counter-clockwise)
        prev = out[-1]
        prev_in = _cross2(edge, prev - a) >= 0
        new = []
        for cur in out:
            cur_in = _cross2(edge, cur - a) >= 0
            if cur_in:
                if not prev_in:
                    new.append(_edge_intersect(prev, cur, a, b))
                new.append(cur)
            elif prev_in:
                new.append(_edge_intersect(prev, cur, a, b))
            prev, prev_in = cur, cur_in
        out = np.asarray(new) if new else np.empty((0, 2))
    return out


def _edge_intersect(p, q, a, b):
    """Intersection of segment p->q with the infinite line a->b."""
    r, s = q - p, b - a
    denom = _cross2(r, s)
    if abs(denom) < 1e-12:
        return p
    t = _cross2(a - p, s) / denom
    return p + t * r


def _poly_area(poly):
    if len(poly) < 3:
        return 0.0
    x, y = poly[:, 0], poly[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def box_iou3d(a, b):
    """Exact 3D IoU of two yaw-only boxes.

    Each box is (cx, cy, cz, hx, hy, hz, yaw) with h = half-extents. Because both
    boxes are upright, the intersection volume is (xy overlap area) x (z overlap
    length) -- no sampling needed.
    """
    z_lo = max(a[2] - a[5], b[2] - b[5])
    z_hi = min(a[2] + a[5], b[2] + b[5])
    dz = z_hi - z_lo
    if dz <= 0:
        return 0.0
    inter_xy = _poly_area(_clip_convex(_rect_corners(a[0], a[1], a[3], a[4], a[6]),
                                       _rect_corners(b[0], b[1], b[3], b[4], b[6])))
    if inter_xy <= 0:
        return 0.0
    inter = inter_xy * dz
    va = 8.0 * a[3] * a[4] * a[5]
    vb = 8.0 * b[3] * b[4] * b[5]
    union = va + vb - inter
    return float(inter / union) if union > 0 else 0.0


# =====================================================================
# Clustering
# =====================================================================

class _UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def load_class_embeddings(path):
    """Load the {class name -> unit vector} cache written by build_class_embeddings."""
    z = np.load(path, allow_pickle=False)
    names = [str(n) for n in z["names"]]
    return {n: v for n, v in zip(names, z["emb"])}


def build_class_embeddings(names, out_path, boxer_repo=None):
    """Embed every class name once with OWLv2's text encoder and cache to .npz.

    Needs torch and boxer/ckpts/owlv2-base-patch16-ensemble.pt, so run it from an
    environment that has them (the fusion itself then needs only numpy). The
    taxonomy is fixed, so this is a one-off: ~600 names take a few seconds.
    """
    import sys
    if boxer_repo:
        sys.path.insert(0, str(boxer_repo))
    from owl.clip_tokenizer import TextEmbedder

    names = sorted(set(names))
    emb = TextEmbedder().forward(names).cpu().numpy().astype(np.float32)  # L2-normed
    np.savez(out_path, names=np.array(names), emb=emb)
    print(f"wrote {len(names)} class embeddings ({emb.shape[1]}-d) to {out_path}")
    return {n: v for n, v in zip(names, emb)}


def _name_groups(names, emb, threshold):
    """Partition class names into sets that are allowed to merge.

    Returns (groups, compat) where ``compat[i, j]`` says whether names i and j
    may be linked (cosine similarity >= threshold) and ``groups`` are connected
    components of that relation. Similarity is not transitive, so the components
    are only a coarse bucket that keeps the pairwise search local -- every
    individual pair is still checked against ``compat`` before it becomes an
    edge, exactly as the BoxerNet fuser does.

    Names missing from the embedding cache fall back to exact-match only.
    """
    k = len(names)
    known = [n for n in names if n in emb]
    compat = np.eye(k, dtype=bool)
    if known:
        idx = [i for i, n in enumerate(names) if n in emb]
        mat = np.stack([emb[names[i]] for i in idx])
        sim = mat @ mat.T                       # rows are unit vectors
        sub = sim >= threshold
        for a, i in enumerate(idx):
            for b, j in enumerate(idx):
                if sub[a, b]:
                    compat[i, j] = True

    uf = _UnionFind(k)
    for i in range(k):
        for j in range(i + 1, k):
            if compat[i, j]:
                uf.union(i, j)
    groups = {}
    for i in range(k):
        groups.setdefault(uf.find(i), []).append(i)
    return list(groups.values()), compat


def _candidate_pairs(centers, radii):
    """Index pairs whose bounding spheres touch -- a cheap superset of overlap.

    Avoids an O(n^2) IoU evaluation: two boxes can only intersect if their centres
    are closer than the sum of their bounding-sphere radii. Uses a KD-tree when
    scipy is available and falls back to the dense test otherwise (fine for the
    few hundred detections one class contributes to a 30 s window).
    """
    n = len(centers)
    if n < 2:
        return []
    r_max = float(radii.max())
    try:
        from scipy.spatial import cKDTree
        tree = cKDTree(centers)
        pairs = tree.query_pairs(2.0 * r_max, output_type="ndarray")
    except ImportError:
        ii, jj = np.triu_indices(n, k=1)
        pairs = np.stack([ii, jj], axis=1)
    if len(pairs) == 0:
        return []
    d = np.linalg.norm(centers[pairs[:, 0]] - centers[pairs[:, 1]], axis=1)
    keep = d <= radii[pairs[:, 0]] + radii[pairs[:, 1]]
    return pairs[keep]


# =====================================================================
# Fusing one cluster
# =====================================================================

def _align_r90(sizes, yaws, weights):
    """Undo 90-degree ambiguity before averaging.

    The same object can be reported as (0.3 x 0.6) at yaw 0 or (0.6 x 0.3) at
    yaw 90 degrees. Averaging those directly gives a square box at 45 degrees.
    Each detection is compared with the highest-weight one and, if its yaw is
    nearer to perpendicular than parallel, its x/y sizes are swapped and its yaw
    rotated by 90 degrees.
    """
    ref = int(np.argmax(weights))
    out_sizes, out_yaws = sizes.copy(), yaws.copy()
    delta = np.abs(np.angle(np.exp(1j * (yaws - yaws[ref]))))
    swap = (delta > np.pi / 4) & (delta < 3 * np.pi / 4)
    out_sizes[swap] = out_sizes[swap][:, [1, 0, 2]]
    out_yaws[swap] = out_yaws[swap] + np.pi / 2
    return out_sizes, out_yaws


def _mean_yaw(yaws, weights):
    """Weighted circular mean with pi periodicity (a box rotated 180 deg is the
    same box), via the doubled-angle trick."""
    z = np.sum(weights * np.exp(2j * yaws))
    return float(np.angle(z) / 2.0)


def _robust_weights(sizes, centers, probs):
    """Confidence weights, down-weighting members that disagree with the group.

    Deviation from the weighted median of centre and size is scaled by the
    median absolute deviation; anything beyond ~3 MADs is strongly suppressed.
    Mirrors the intent of the repo's ``robust`` weighting without copying it.
    """
    if len(probs) <= 2:
        return probs
    ref_c, ref_s = np.median(centers, axis=0), np.median(sizes, axis=0)
    dev = np.concatenate([centers - ref_c, sizes - ref_s], axis=1)
    scale = np.median(np.abs(dev), axis=0) + 1e-6
    r = np.max(np.abs(dev) / scale, axis=1)
    return probs / (1.0 + (r / 3.0) ** 2)


def _fuse_cluster(rows, weighting):
    """Confidence-weighted average of one group of detections -> one box dict."""
    probs = rows["prob"].to_numpy(np.float64)
    centers = rows[POS_COLS].to_numpy(np.float64)
    sizes = rows["_size"].to_list()
    sizes = np.asarray(sizes, dtype=np.float64)
    yaws = rows["_yaw"].to_numpy(np.float64)

    w = probs.copy()
    sizes_a, yaws_a = _align_r90(sizes, yaws, w)
    if weighting == "robust":
        w = _robust_weights(sizes_a, centers, probs)
    w = w / (w.sum() + 1e-12)

    center = (centers * w[:, None]).sum(axis=0)
    size = (sizes_a * w[:, None]).sum(axis=0)
    yaw = _mean_yaw(yaws_a, w)
    prob = float((probs * w).sum())

    names = rows["name"].to_list()
    # Most common label, ties broken alphabetically. sorted() is not cosmetic:
    # iterating a set uses Python's per-process randomised string hashing, so
    # max(set(names), ...) returns an ARBITRARY winner among tied labels and a
    # different one in each interpreter. Measured over the 152 cache files built
    # without it, 1060 of the 1317 tied groups seen in more than one kitchen
    # resolved differently per kitchen -- cabinet/cupboard, cup/mug,
    # kitchen_sink/sink -- and since model.py embeds sem_id through a learned
    # nn.Embedding (not these class embeddings), two spellings of one object are
    # unrelated rows there. That put a vocabulary shift on exactly the
    # train/test boundary a LOOCV split measures.
    name = max(sorted(set(names)), key=names.count)
    sem_id = int(rows.loc[rows["name"] == name, "sem_id"].mode().iloc[0])

    return dict(
        center=center,
        quat=np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)]),  # (w,x,y,z)
        half_extents=size / 2.0,
        sem_id=sem_id + 1,          # +1 reserves 0 for empty, as boxer_env does
        prob=prob,
        name=name,
        support=len(rows),
        yaw=yaw,
        # what the members were labelled, so cross-class merges are auditable
        members={n: names.count(n) for n in sorted(set(names))},
    )


# =====================================================================
# Public API
# =====================================================================

def fuse_boxes_iou(df, conf_threshold=0.5, iou_threshold=0.3, min_detections=2,
                   weighting="conf", nms_iou=None, aliases=None,
                   embeddings=None, semantic_threshold=0.6,
                   legacy_half=False, skip_names=SKIP_OBJECT_NAMES):
    """Fuse one window's Boxer rows by 3D IoU. Returns the same dicts as
    ``boxer_env._fuse_boxes`` (center / quat / half_extents / sem_id / prob),
    plus ``name``, ``support`` and ``yaw``.

    Args:
        df: Boxer CSV rows for the window (any length).
        conf_threshold: drop detections below this probability.
        iou_threshold: minimum 3D IoU for two detections to be linked.
        min_detections: discard groups with fewer members.
        weighting: "conf" (weight = probability) or "robust" (also down-weight
            members that disagree with their group).
        nms_iou: if set, drop the lower-confidence of any two fused boxes of the
            same class overlapping by more than this.
        aliases: optional {name: canonical_name} to treat classes as equivalent
            (e.g. {"cupboard": "cabinet"}). Applied before embeddings, and mostly
            unnecessary once embeddings are supplied.
        embeddings: {class name -> unit vector} or a path to the .npz written by
            build_class_embeddings. When given, two classes may merge if their
            cosine similarity reaches semantic_threshold, so "chair" and
            "armchair" fuse without an explicit alias. When None, only identical
            class names merge.
        semantic_threshold: cosine similarity required between two class names.
            Measured on the lvisplus vocabulary: unrelated pairs stay below 0.39
            (stove/sink 0.386, chair/banana 0.134) while related ones run 0.61
            upward (chair/armchair 0.615, kettle/teakettle 0.612, mug/cup 0.756,
            cabinet/cupboard 0.926). The repo's 0.7 default would reject
            chair/armchair, hence 0.6 here. Values as low as 0.5 stay safe in
            practice because a merge also requires 3D overlap.
        legacy_half: treat scale_* as half-extents, reproducing the existing
            (oversized) convention in boxer_env._fuse_boxes.
    """
    df = df[(df["prob"] >= conf_threshold) & (~df["name"].isin(skip_names))]
    if len(df) == 0:
        return []

    df = df.copy()
    if aliases:
        df["name"] = df["name"].map(lambda n: aliases.get(n, n))
    scale = df[SCALE_COLS].to_numpy(np.float64)
    full = scale * 2.0 if legacy_half else scale     # full size of each box
    df["_size"] = list(full)
    # qx = qy = 0 for every Boxer OBB, so the rotation is a pure yaw about +z.
    df["_yaw"] = 2.0 * np.arctan2(df["qz_world_object"].to_numpy(np.float64),
                                  df["qw_world_object"].to_numpy(np.float64))

    # Which classes may merge with which. Without embeddings this is the identity
    # (exact name match); with them it is the OWLv2 cosine-similarity relation.
    names = sorted(df["name"].unique())
    name_idx = {n: i for i, n in enumerate(names)}
    if embeddings is None:
        groups_of_names = [[i] for i in range(len(names))]
        compat = np.eye(len(names), dtype=bool)
    else:
        if isinstance(embeddings, (str, bytes)) or hasattr(embeddings, "__fspath__"):
            embeddings = load_class_embeddings(embeddings)
        groups_of_names, compat = _name_groups(names, embeddings, semantic_threshold)

    df["_ni"] = df["name"].map(name_idx).to_numpy()

    fused = []
    for gi in groups_of_names:
        grp = df[df["_ni"].isin(gi)].reset_index(drop=True)
        n = len(grp)
        if n < min_detections:
            continue
        centers = grp[POS_COLS].to_numpy(np.float64)
        halves = np.stack(grp["_size"].to_numpy()) / 2.0
        radii = np.linalg.norm(halves, axis=1)
        boxes = np.concatenate(
            [centers, halves, grp["_yaw"].to_numpy(np.float64)[:, None]], axis=1)
        ni = grp["_ni"].to_numpy()

        uf = _UnionFind(n)
        for i, j in _candidate_pairs(centers, radii):
            i, j = int(i), int(j)
            # a pair needs BOTH a compatible class and enough overlap
            if not compat[ni[i], ni[j]]:
                continue
            if box_iou3d(boxes[i], boxes[j]) >= iou_threshold:
                uf.union(i, j)

        groups = {}
        for i in range(n):
            groups.setdefault(uf.find(i), []).append(i)
        for members in groups.values():
            if len(members) < min_detections:
                continue
            fused.append(_fuse_cluster(grp.iloc[members], weighting))

    if nms_iou is not None and fused:
        fused = _nms(fused, nms_iou)
    return fused


def _nms(fused, nms_iou):
    """Keep the best of any two same-class fused boxes that still overlap."""
    order = sorted(range(len(fused)), key=lambda i: -fused[i]["prob"])
    kept, kept_boxes = [], []
    for i in order:
        f = fused[i]
        box = np.concatenate([f["center"], f["half_extents"], [f["yaw"]]])
        if any(f["name"] == g["name"] and box_iou3d(box, b) > nms_iou
               for g, b in zip(kept, kept_boxes)):
            continue
        kept.append(f)
        kept_boxes.append(box)
    return kept


class BoxerEnvBuilderIoU(BoxerEnvBuilder):
    """``BoxerEnvBuilder`` with IoU fusion in place of grid snapping.

    Everything else -- CSV loading, the 10 Hz downsample, the time mask, the
    voxelisation and the OBB tokens -- is inherited unchanged, so swapping the
    class at the loader's construction site is the only edit needed.
    """

    def __init__(self, *args, iou_threshold=0.3, weighting="conf", nms_iou=None,
                 aliases=None, legacy_half=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.iou_threshold = iou_threshold
        self.weighting = weighting
        self.nms_iou = nms_iou
        self.aliases = aliases
        self.legacy_half = legacy_half

    def _window_fused_boxes(self, video_id, obs_start_s, obs_end_s, t0_s):
        boxer_df = self.load_boxer_csv(video_id)
        voxel_centers = self.get_voxel_centers(video_id)
        if boxer_df is None or voxel_centers is None:
            return None, voxel_centers

        vrs_start = (obs_start_s + t0_s) * 1e9
        vrs_end = (obs_end_s + t0_s) * 1e9
        mask = (boxer_df["time_ns"] >= vrs_start) & (boxer_df["time_ns"] <= vrs_end)
        window_df = boxer_df.loc[mask]
        if len(window_df) == 0:
            return None, voxel_centers

        fused = fuse_boxes_iou(
            window_df, conf_threshold=self.conf_threshold,
            iou_threshold=self.iou_threshold,
            min_detections=self.min_detections, weighting=self.weighting,
            nms_iou=self.nms_iou, aliases=self.aliases,
            legacy_half=self.legacy_half)
        return (fused or None), voxel_centers


# =====================================================================
# CLI
# =====================================================================

def main():
    import argparse
    import os
    import time

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--video", default="P08-20240614-085000")
    ap.add_argument("--root", default=os.path.dirname(os.path.abspath(__file__)))
    ap.add_argument("--start", type=float, default=13.6, help="window start (video s)")
    ap.add_argument("--end", type=float, default=43.6, help="window end (video s)")
    ap.add_argument("--t0", type=float, default=None,
                    help="device-clock second of video-second 0 (TIME_ANCHOR); "
                         "default: first gaze timestamp + 0.1")
    ap.add_argument("--iou", type=float, default=0.3)
    ap.add_argument("--conf", type=float, default=0.5)
    ap.add_argument("--min-detections", type=int, default=2)
    ap.add_argument("--weighting", default="conf", choices=["conf", "robust"])
    ap.add_argument("--nms-iou", type=float, default=None)
    ap.add_argument("--legacy-half", action="store_true")
    ap.add_argument("--embeddings", default=None,
                    help="class-embedding .npz (see --build-embeddings); enables "
                         "similarity-based class matching")
    ap.add_argument("--semantic-threshold", type=float, default=0.6)
    ap.add_argument("--build-embeddings", metavar="OUT.npz", default=None,
                    help="embed the vocabulary once and exit (needs torch + the "
                         "OWLv2 checkpoint; run from the boxer venv)")
    ap.add_argument("--vocabulary", default=None,
                    help="class list for --build-embeddings (default: the video's "
                         "own names; prefer boxer/owl/lvisplus_classes.csv)")
    ap.add_argument("--boxer-repo", default=None,
                    help="path to the boxer checkout, for --build-embeddings")
    ap.add_argument("--audit", metavar="OUT.csv", default=None,
                    help="write every class pair the threshold would allow to "
                         "merge, most similar first, and exit")
    ap.add_argument("--compare", action="store_true",
                    help="also run the model's grid-snap fusion and compare")
    ap.add_argument("--out", default=None, help="write fused boxes to this CSV")
    args = ap.parse_args()

    builder = BoxerEnvBuilder(boxer_root=args.root)

    if args.build_embeddings:
        if args.vocabulary:
            vocab = [l.strip() for l in open(args.vocabulary) if l.strip()]
        else:
            vocab = pd.read_csv(os.path.join(args.root, args.video,
                                             "boxer_3dbbs.csv"),
                                usecols=["name"])["name"].unique().tolist()
        build_class_embeddings(vocab, args.build_embeddings,
                               boxer_repo=args.boxer_repo)
        return

    if args.audit:
        if not args.embeddings:
            raise SystemExit("--audit needs --embeddings")
        emb = load_class_embeddings(args.embeddings)
        names = sorted(emb)
        mat = np.stack([emb[n] for n in names])
        sim = mat @ mat.T
        ii, jj = np.triu_indices(len(names), k=1)
        keep = sim[ii, jj] >= args.semantic_threshold
        out = pd.DataFrame({
            "class_a": [names[i] for i in ii[keep]],
            "class_b": [names[j] for j in jj[keep]],
            "similarity": sim[ii, jj][keep],
        }).sort_values("similarity", ascending=False)
        out.to_csv(args.audit, index=False)
        print(f"{len(out)} of {len(ii)} class pairs reach "
              f"{args.semantic_threshold}; wrote {args.audit}")
        print(out.head(15).to_string(index=False))
        return

    df = builder.load_boxer_csv(args.video)
    if df is None:
        raise SystemExit(f"no Boxer CSV for {args.video} under {args.root}")

    t0 = args.t0
    if t0 is None:
        gaze = pd.read_csv(os.path.join(args.root, args.video, "general_eye_gaze.csv"),
                           usecols=["tracking_timestamp_us"])
        t0 = float(gaze["tracking_timestamp_us"].iloc[0]) / 1e6 + 0.1
    lo, hi = (args.start + t0) * 1e9, (args.end + t0) * 1e9
    w = df[(df["time_ns"] >= lo) & (df["time_ns"] <= hi)]
    print(f"{args.video}  window [{args.start}, {args.end}] s  t0={t0:.6f}")
    print(f"  rows in window: {len(w)} over {w['time_ns'].nunique()} timestamps")

    t = time.time()
    fused = fuse_boxes_iou(w, conf_threshold=args.conf, iou_threshold=args.iou,
                           min_detections=args.min_detections,
                           weighting=args.weighting, nms_iou=args.nms_iou,
                           embeddings=args.embeddings,
                           semantic_threshold=args.semantic_threshold,
                           legacy_half=args.legacy_half)
    dt = time.time() - t
    print(f"  IoU fusion  : {len(fused)} boxes, {len({f['name'] for f in fused})} "
          f"classes  ({dt:.2f} s)")

    if args.compare:
        t = time.time()
        grid = _fuse_boxes(w, conf_threshold=args.conf,
                           min_detections=args.min_detections,
                           cluster_distance=builder.cluster_distance)
        print(f"  grid fusion : {len(grid)} boxes  ({time.time() - t:.2f} s)")
        by_name = {}
        for f in fused:
            by_name[f["name"]] = by_name.get(f["name"], 0) + 1
        dupes = {n: c for n, c in by_name.items() if c > 1}
        print(f"  classes with >1 IoU-fused box: {len(dupes)} of {len(by_name)}")

    if args.out:
        pd.DataFrame([{
            "name": f["name"], "sem_id": f["sem_id"] - 1,
            "tx_world_object": f["center"][0], "ty_world_object": f["center"][1],
            "tz_world_object": f["center"][2],
            "qw_world_object": f["quat"][0], "qx_world_object": f["quat"][1],
            "qy_world_object": f["quat"][2], "qz_world_object": f["quat"][3],
            # written as full sizes, matching the input CSV convention
            "scale_x": f["half_extents"][0] * 2, "scale_y": f["half_extents"][1] * 2,
            "scale_z": f["half_extents"][2] * 2,
            "prob": f["prob"], "support": f["support"],
        } for f in fused]).sort_values("prob", ascending=False).to_csv(args.out,
                                                                      index=False)
        print(f"  wrote {args.out}")


if __name__ == "__main__":
    main()
