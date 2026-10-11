#!/usr/bin/env python3
# meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
# Source-available, not Open Source. See LICENSE at the repository root.
# No GPL-licensed source was consulted.
"""
costbound.py - the G-COST lower bound (docs/17 §12 row AUT-06, docs/15 §F).

Per geometry this computes a LOWER BOUND on the cells of ANY octree mesh
that holds the R-WIN wall level on the whole wall, keeps 2:1 shells out to
the base grid and carries 8 wall layers, and reports whether G-COST's 1.5x
(median cells of the full system <= 1.5 x median B0-template cells on the
geometries both pass, evaluate.CELLS_RATIO_MAX) is attainable at all.
Nothing here changes a rule, a gate or a constant: where 1.5x is not
attainable the report says so and the decision (D-8) is the user's.

CB-1 lattice: n_a = round((hi_a - lo_a)/b), s_a = (hi_a - lo_a)/n_a; a
level-l cell i_a in [0, n_a 2^l) spans [lo_a + i_a h_a, lo_a + (i_a+1) h_a]
with h_a = s_a / 2**l, computed in that order.  CB-2 the touched set S_L:
the level-L cells whose CLOSED box meets a CLOSED triangle - the
separating-axis test of Akenine-Möller 2001, where an axis separates only
on a strict gap and touching is not separated; the candidates per triangle
are the cells of its bbox.  CB-3 the least balanced tree: R_{L-1} = the
parents of S_L, then level by level R_{l-1} = parents(R_l) together with
the parents of R_l's face neighbours (SPEC-LIT (92.21), the least octree
refinement in which every S_L cell is a level-L leaf and every
face-adjacent leaf pair differs by at most one level; Sundar et al. 2008).
CB-4 the leaves: base cells not in R_0, and children of R_{l-1} not in R_l.
CB-5 solid: the parity of +z ray crossings at the leaf centre, the
projected triangle taken under Pineda's top-left rule, the plane z
STRICTLY above (SPEC-LIT (92.23)).  CB-6 the wall faces: face-adjacent
leaf pairs with one fluid and one solid leaf, counted once, a 2:1 coarse
face four faces (92.19).  CB-7 bound = fluid + n_layers x wall_faces.
CB-8 a campaign row bounds when fluid + blc8 x layer_cells <= its n_cells.
CB-9 attainable: median(bound)/median(B0) <= evaluate.CELLS_RATIO_MAX on a
set's both-pass rows.

    python tools/autonomy/costbound.py --selftest
    python tools/autonomy/costbound.py --report [--jobs N] [--ids A,B]
                                     [--work DIR] [--out-dir DIR]
    python tools/autonomy/costbound.py --id GEOMETRY_ID --set tuning

Two sets: "tuning" (system aml/tuning_rules_L5.json.gz against
baseline/tuning_b0-template.json.gz, manifest split.load("tuning",
"costbound"); the TREE has no 420-row full-system tuning campaign, the L5
rules campaign is the tuning re-measure) and "seed1-posthoc"
(evaluate/eval_full.json.gz against evaluate/eval_b0-template.json.gz,
manifest split.load("test", split.EVALUATE)), labelled post-hoc on the
spent seed-1 test split (evaluated 2026-09-26); not a claim.

Cited (all DOIs checked on Crossref 2026-10-11): Akenine-Möller, T. (2001),
"Fast 3D Triangle-Box Overlap Testing", Journal of Graphics Tools 6(1),
29-33, DOI 10.1080/10867651.2001.10487535 (CB-2); Sundar, H., Sampath,
R. S. & Biros, G. (2008), "Bottom-Up Construction and 2:1 Balance
Refinement of Linear Octrees in Parallel", SIAM J. Sci. Comput. 30(5),
2675-2708, DOI 10.1137/070681727 (CB-3); Pineda, J. (1988), "A parallel
algorithm for polygon rasterization", ACM SIGGRAPH Computer Graphics
22(4), 17-20, DOI 10.1145/378456.378457 (CB-5's top-left rule); SPEC-LIT
§92.2 (92.3), §92.9 (92.17)-(92.21), §92.10 (92.23) (the mesher's own
lattice, balance and centre test); docs/15 §D.3, §F G-COST.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib
import json
import os
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "corpus"))
import schema  # noqa: E402
import split  # noqa: E402
import features  # noqa: E402
import rules  # noqa: E402
import preflight  # noqa: E402
import evaluate  # noqa: E402

N_LAYERS = 8
OUT_DIR = os.path.join(HERE, "costbound")
SCHEMA_ID = "autonomy-costbound/1"
REPAIR_TIMEOUT_S = 300


class CostBoundError(Exception):
    """A refused input, or a surface stl_repair leaves open (SURFACE-OPEN)."""


def _level(level) -> int:
    """level as an int, refusing a negative or a non-int (names the value)."""
    if isinstance(level, bool) or not isinstance(level, int) or level < 0:
        raise CostBoundError("level: %r is negative or not an int" % (level,))
    return int(level)


def lattice(extent: list, base_size: float) -> dict:
    """CB-1: the base lattice of extent E at base size b (n, s, lo, n_base)."""
    b = float(base_size)
    if not b > 0.0:
        raise CostBoundError("base_size: %r must be > 0" % (base_size,))
    e = [float(v) for v in extent]
    if len(e) != 6:
        raise CostBoundError("extent: %r is not [x0,x1,y0,y1,z0,z1]" % (extent,))
    lo, hi = (e[0], e[2], e[4]), (e[1], e[3], e[5])
    for a in range(3):
        if not hi[a] > lo[a]:
            raise CostBoundError("extent: hi %r <= lo %r on axis %d"
                                 % (hi[a], lo[a], a))
    n = [int(round((hi[a] - lo[a]) / b)) for a in range(3)]
    if min(n) < 1:
        raise CostBoundError("base_size: %r gives %r base cells < 1"
                             % (base_size, n))
    s = [(hi[a] - lo[a]) / n[a] for a in range(3)]
    return {"n": n, "s": s, "lo": list(lo), "n_base": n[0] * n[1] * n[2]}


def _dims(lat: dict, level: int) -> tuple:
    """The level's cell counts per axis (n_a 2^l)."""
    f = 1 << level
    return (lat["n"][0] * f, lat["n"][1] * f, lat["n"][2] * f)


def _unpack(p: int, nl: tuple) -> tuple:
    """(i, j, k) of a cell packed as (i ny + j) nz + k."""
    i, r = divmod(p, nl[1] * nl[2])
    j, k = divmod(r, nl[2])
    return i, j, k


def _pack(i: int, j: int, k: int, nl: tuple) -> int:
    return (i * nl[1] + j) * nl[2] + k


def _parent_key(p: int, nl: tuple) -> int:
    """The parent's packed key one level down."""
    i, j, k = _unpack(p, nl)
    return ((i >> 1) * (nl[1] >> 1) + (j >> 1)) * (nl[2] >> 1) + (k >> 1)


def _unpack_vec(p, nl: tuple):
    """(i, j, k) arrays of packed keys."""
    i, r = np.divmod(p, nl[1] * nl[2])
    j, k = np.divmod(r, nl[2])
    return i, j, k


def _neighbours(i: int, j: int, k: int, nl: tuple) -> list:
    """The cell's 6 face neighbours inside the domain, packed."""
    out = []
    if i > 0:
        out.append(_pack(i - 1, j, k, nl))
    if i + 1 < nl[0]:
        out.append(_pack(i + 1, j, k, nl))
    if j > 0:
        out.append(_pack(i, j - 1, k, nl))
    if j + 1 < nl[1]:
        out.append(_pack(i, j + 1, k, nl))
    if k > 0:
        out.append(_pack(i, j, k - 1, nl))
    if k + 1 < nl[2]:
        out.append(_pack(i, j, k + 1, nl))
    return out


def _sat(a, b, c, n, lo, h, ii, jj, kk) -> np.ndarray:
    """The non-box axes of Akenine-Möller's test over one candidate block.

    The three box axes are the candidate range itself; this tests the
    triangle normal and the 9 edge cross products, an axis separating only
    on a strict gap (touching is not separated).  A cross axis of zero
    length is skipped.
    """
    cmin = (lo[0] + ii * h[0], lo[1] + jj * h[1], lo[2] + kk * h[2])
    ctr = (cmin[0] + 0.5 * h[0], cmin[1] + 0.5 * h[1], cmin[2] + 0.5 * h[2])
    axes = [n]
    for e in (b - a, c - b, a - c):
        for ax in ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)):
            w = np.cross(e, ax)
            if np.any(w):
                axes.append(w)
    sep = np.zeros(ii.shape, dtype=bool)
    for v in axes:
        t0 = min(float(a @ v), float(b @ v), float(c @ v))
        t1 = max(float(a @ v), float(b @ v), float(c @ v))
        r = 0.5 * float(abs(v[0]) * h[0] + abs(v[1]) * h[1] + abs(v[2]) * h[2])
        p0 = ctr[0] * v[0] + ctr[1] * v[1] + ctr[2] * v[2]
        sep |= (t0 > p0 + r) | (t1 < p0 - r)
        if sep.all():
            break
    return ~sep


def touched_cells(P, T, lat: dict, level: int) -> set:
    """CB-2: the level-l cells whose CLOSED box meets a CLOSED triangle."""
    level = _level(level)
    Pt = np.asarray(P, dtype=np.float64)
    Tt = np.asarray(T, dtype=np.int64)
    lo = np.array(lat["lo"], dtype=np.float64)
    h = np.array(lat["s"], dtype=np.float64) / float(2 ** level)
    nl = np.array([n << level for n in lat["n"]], dtype=np.int64)
    out = np.empty(0, dtype=np.int64)
    for ia, ib, ic in Tt:
        a, b, c = Pt[ia], Pt[ib], Pt[ic]
        n = np.cross(b - a, c - a)
        if not np.any(n):        # a zero-area triangle is skipped
            continue
        tmin = np.minimum(np.minimum(a, b), c)
        tmax = np.maximum(np.maximum(a, b), c)
        i0 = np.ceil((tmin - lo) / h).astype(np.int64) - 1
        i1 = np.floor((tmax - lo) / h).astype(np.int64) + 1
        i0 = np.maximum(i0, np.zeros(3, dtype=np.int64))
        i1 = np.minimum(i1, nl)
        if np.any(i0 >= i1):
            continue
        ii, jj, kk = np.meshgrid(np.arange(i0[0], i1[0]),
                                 np.arange(i0[1], i1[1]),
                                 np.arange(i0[2], i1[2]), indexing="ij")
        keep = _sat(a, b, c, n, lo, h, ii, jj, kk)
        if keep.any():
            cells = ii[keep] * (nl[1] * nl[2]) + jj[keep] * nl[2] + kk[keep]
            out = np.unique(np.concatenate((out, cells)))
    return set(int(v) for v in out)


def split_sets(touched: set, lat: dict, level: int) -> list:
    """CB-3: [R_0, ..., R_{L-1}], the least balanced tree around `touched`.

    R_{L-1} = parents of S_L; then R_{l-1} = parents(R_l) together with the
    parents of R_l's face neighbours at level l inside the domain.  L = 0
    means no split.  Every S_L cell stays a level-L leaf and every
    face-adjacent leaf pair differs by at most one level.
    """
    level = _level(level)
    if level == 0:
        return []
    R = [None] * level
    R[level - 1] = set(_parent_key(c, _dims(lat, level)) for c in touched)
    for l in range(level - 1, 0, -1):
        nl = _dims(lat, l)
        nxt = set()
        for c in R[l]:
            nxt.add(_parent_key(c, nl))
            for m in _neighbours(*_unpack(c, nl), nl):
                nxt.add(_parent_key(m, nl))
        R[l - 1] = nxt
    return R


def leaf_arrays(R: list, lat: dict, level: int) -> tuple:
    """CB-4: (levels, centres, keys) of the leaves of the tree R.

    Level-0 leaves are the base cells not in R_0; level-l leaves (l >= 1)
    are the children of R_{l-1} cells not in R_l.  keys are packed ints at
    each leaf's own level, centres are CB-1 midpoints, float64.
    """
    level = _level(level)
    Rs = list(R) + [set()] * (level + 1 - len(R))
    keys, levs = [], []
    for l in range(level + 1):
        if l == 0:
            allk = np.arange(lat["n_base"], dtype=np.int64)
            ex = np.array(sorted(Rs[0]), dtype=np.int64)
            keep = allk[~np.isin(allk, ex)] if ex.size else allk
        else:
            pk = np.array(sorted(Rs[l - 1]), dtype=np.int64)
            pi, pj, pkk = _unpack_vec(pk, _dims(lat, l - 1))
            nl = _dims(lat, l)
            base = ((2 * pi) * nl[1] + 2 * pj) * nl[2] + 2 * pkk
            offs = np.array([di * nl[1] * nl[2] + dj * nl[2] + dk
                             for di in (0, 1) for dj in (0, 1)
                             for dk in (0, 1)], dtype=np.int64)
            ch = (base[:, None] + offs[None, :]).reshape(-1)
            ex = np.array(sorted(Rs[l]), dtype=np.int64)
            keep = ch[~np.isin(ch, ex)] if ex.size else ch
        keys.append(np.ascontiguousarray(keep, dtype=np.int64))
        levs.append(np.full(keep.size, l, dtype=np.int64))
    K = np.concatenate(keys)
    L = np.concatenate(levs)
    lo = np.array(lat["lo"], dtype=np.float64)
    C = np.empty((K.size, 3), dtype=np.float64)
    for l in range(level + 1):
        m = L == l
        if not m.any():
            continue
        i, j, k = _unpack_vec(K[m], _dims(lat, l))
        hl = np.array(lat["s"], dtype=np.float64) / float(2 ** l)
        C[m, 0] = lo[0] + (i + 0.5) * hl[0]
        C[m, 1] = lo[1] + (j + 0.5) * hl[1]
        C[m, 2] = lo[2] + (k + 0.5) * hl[2]
    return L, C, K


def _tl_edge(p, q) -> bool:
    """Top edge (a.y == b.y and b.x < a.x) or left edge (b.y < a.y)."""
    return (p[1] == q[1] and q[0] < p[0]) or (q[1] < p[1])


def inside(P, T, points) -> np.ndarray:
    """CB-5: bool per point, the parity of +z ray crossings (Pineda 1988).

    A triangle is crossed when its xy projection has non-zero signed area,
    the projection contains the point under the top-left rule (the
    projected triangle oriented counter-clockwise; a point on an edge
    counts only on a top or a left edge, so a centre on a shared projected
    edge or vertex is counted once per surface sheet), and the triangle's
    plane at (x, y) has z STRICTLY greater than the point's z.
    """
    Pt = np.asarray(P, dtype=np.float64)
    Tt = np.asarray(T, dtype=np.int64)
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    order = np.argsort(pts[:, 0], kind="stable")
    xs = pts[order, 0]
    ys = pts[order, 1]
    zs = pts[order, 2]
    solid = np.zeros(pts.shape[0], dtype=bool)
    for ia, ib, ic in Tt:
        a, b, c = Pt[ia], Pt[ib], Pt[ic]
        ar2 = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
        if ar2 == 0.0:
            continue
        if ar2 < 0.0:
            b, c, ar2 = c, b, -ar2
        s0 = int(np.searchsorted(xs, min(a[0], b[0], c[0]), side="left"))
        s1 = int(np.searchsorted(xs, max(a[0], b[0], c[0]), side="right"))
        if s0 >= s1:
            continue
        ys_sl = ys[s0:s1]
        m = (ys_sl >= min(a[1], b[1], c[1])) & (ys_sl <= max(a[1], b[1], c[1]))
        if not m.any():
            continue
        off = np.nonzero(m)[0]
        px = xs[s0 + off]
        py = ys_sl[off]
        e0 = (b[0] - a[0]) * (py - a[1]) - (b[1] - a[1]) * (px - a[0])
        e1 = (c[0] - b[0]) * (py - b[1]) - (c[1] - b[1]) * (px - b[0])
        e2 = (a[0] - c[0]) * (py - c[1]) - (a[1] - c[1]) * (px - c[0])
        take = (e0 > 0.0) | ((e0 == 0.0) & _tl_edge(a, b))
        take &= (e1 > 0.0) | ((e1 == 0.0) & _tl_edge(b, c))
        take &= (e2 > 0.0) | ((e2 == 0.0) & _tl_edge(c, a))
        zpl = (e1 * a[2] + e2 * b[2] + e0 * c[2]) / ar2
        hit = take & (zpl > zs[s0 + off])
        if hit.any():
            solid[order[s0 + off[hit]]] ^= True
    return solid


def _wall_faces(keys, levels, solid, lat: dict) -> int:
    """CB-6: face-adjacent leaf pairs one fluid one solid, counted once.

    Across a 2:1 interface the coarse face is four faces, one per finer
    leaf (SPEC-LIT (92.19)): a same-level pair is taken from the lower key
    alone, a cross-level pair from the finer side alone.  Domain-boundary
    faces never count.
    """
    per = {}
    for l in range(int(levels.max()) + 1):
        m = levels == l
        if m.any():
            o = np.argsort(keys[m], kind="stable")
            per[l] = (keys[m][o], solid[m][o])
    faces = 0
    for l in sorted(per):
        ks, so = per[l]
        nl = _dims(lat, l)
        i, j, k = _unpack_vec(ks, nl)
        for axis, st in enumerate((nl[1] * nl[2], nl[2], 1)):
            idx = (i, j, k)[axis]
            for sgn in (-1, 1):
                ok = (idx + sgn >= 0) & (idx + sgn < nl[axis])
                if not ok.any():
                    continue
                q = ks + sgn * st
                pos = np.searchsorted(ks, q)
                pos[pos >= ks.size] = 0
                same = ok & (ks[pos] == q)
                faces += int(np.count_nonzero(same & (ks < q)
                                              & (so != so[pos])))
                if l == 0 or not (ok & ~same).any():
                    continue
                qi, qj, qk = _unpack_vec(q, nl)
                pl = _dims(lat, l - 1)
                pk = ((qi >> 1) * pl[1] + (qj >> 1)) * pl[2] + (qk >> 1)
                kp, sp = per.get(l - 1, (None, None))
                if kp is None or kp.size == 0:
                    continue
                pp = np.searchsorted(kp, pk)
                pp[pp >= kp.size] = 0
                hit = ok & ~same & (kp[pp] == pk)
                faces += int(np.count_nonzero(hit & (so != sp[pp])))
    return faces


def bound(P, T, extent, base_size, level, n_layers=N_LAYERS) -> dict:
    """CB-7: the cell lower bound of one geometry, as a dict of ints."""
    level = _level(level)
    lat = lattice(extent, base_size)
    touched = touched_cells(P, T, lat, level)
    R = split_sets(touched, lat, level)
    levels, centres, keys = leaf_arrays(R, lat, level)
    solid = inside(P, T, centres)
    fluid = int(np.count_nonzero(~solid))
    wall_faces = _wall_faces(keys, levels, solid, lat)
    layer_cells = n_layers * wall_faces
    return {"n_base": lat["n_base"], "touched": len(touched),
            "splits": sum(len(r) for r in R), "leaves": int(levels.size),
            "fluid": fluid, "wall_faces": wall_faces,
            "layer_cells": layer_cells, "bound": fluid + layer_cells}


def row_inputs(bundle: dict, geometry_id: str) -> dict:
    """{"extent", "base_size", "wall_level"} from a bundle's attempt-1 records.

    In list order the LAST edit to /domain/extent and to /domain/base_size
    (R-DOM, then R-PLANE if it edits), and the R-WIN record with verdict
    apply, its input named wall_level.  Missing any of them refuses.
    """
    extent = base_size = wall_level = None
    for r in bundle.get("records", []):
        if r.get("attempt") != 1 or r.get("geometry_id") != geometry_id:
            continue
        rec = r.get("record") or {}
        for e in rec.get("edits", []):
            p = e.get("pointer")
            if p == "/domain/extent":
                extent = e.get("to")
            elif p == "/domain/base_size":
                base_size = e.get("to")
        if rec.get("rule_id") == "R-WIN" and rec.get("verdict") == "apply":
            for i in rec.get("inputs", []):
                if i.get("name") == "wall_level":
                    wall_level = i.get("value")
    if wall_level is None:
        raise CostBoundError("%s: no R-WIN wall level" % geometry_id)
    if extent is None:
        raise CostBoundError("%s: no /domain/extent edit" % geometry_id)
    if base_size is None:
        raise CostBoundError("%s: no /domain/base_size edit" % geometry_id)
    return {"extent": [float(v) for v in extent],
            "base_size": float(base_size), "wall_level": int(wall_level)}


def surface_of(mrow: dict, work_dir: str) -> str:
    """Regenerate + stl_repair, as campaign.observe does; the surface path."""
    gid = mrow["geometry_id"]
    raw_dir = os.path.join(work_dir, "stl", "raw")
    mod = importlib.import_module("inject" if mrow["family"] == "G"
                                  else rules.GEN_OF[mrow["family"]])
    os.makedirs(raw_dir, exist_ok=True)
    raw = mod.write_row(mrow, raw_dir)
    out = os.path.join(work_dir, "stl", gid + ".stl")
    rep = os.path.join(work_dir, "stl", gid + "_repair.json")
    proc = subprocess.run(
        [sys.executable, preflight.STL_REPAIR, raw, "--out", out,
         "--json", rep], timeout=REPAIR_TIMEOUT_S, capture_output=True,
        text=True, encoding="utf-8", errors="replace")
    if proc.returncode != 0 or not os.path.isfile(rep):
        raise CostBoundError("%s: stl_repair failed (exit %s)"
                             % (gid, proc.returncode))
    with open(rep, encoding="utf-8") as f:
        report = json.load(f)
    before, after = report["before"], report["after"]
    o = report.get("orientation") or {}
    if bool(before["closed"]) and not o.get("reoriented_triangles", 0) \
            and not o.get("flipped_components", 0):
        shutil.copyfile(raw, out)
    elif not bool(after["closed"]):
        raise CostBoundError("%s: SURFACE-OPEN (%d open edge(s) after repair)"
                             % (gid, after.get("open_edges", -1)))
    return out


_SETS = {
    "tuning": {
        "system": os.path.join("aml", "tuning_rules_L5.json.gz"),
        "b0": os.path.join("baseline", "tuning_b0-template.json.gz"),
        "manifest": ("tuning", "costbound"),
        "label": "the L5 rules campaign, the tuning re-measure",
    },
    "seed1-posthoc": {
        "system": os.path.join("evaluate", "eval_full.json.gz"),
        "b0": os.path.join("evaluate", "eval_b0-template.json.gz"),
        "manifest": ("test", split.EVALUATE),
        "label": "post-hoc on the spent seed-1 test split (evaluated "
                 "2026-09-26); not a claim",
    },
}


def _load_bundle(rel: str) -> dict:
    with gzip.open(os.path.join(HERE, rel), "rt", encoding="utf-8") as f:
        return json.load(f)


def _sha256(rel: str) -> str:
    h = hashlib.sha256()
    with open(os.path.join(HERE, rel), "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def _one_row(job, work_dir: str):
    """One geometry: regenerate the surface, bound it, compare (CB-8)."""
    name, gid, mrow, ri, blc8, n_cells, b0_cells = job
    stl = surface_of(mrow, work_dir)
    surf = features.read_surface(stl)
    b = bound(surf["P"], surf["T"], ri["extent"], ri["base_size"],
              ri["wall_level"])
    blc = 0 if blc8 is None else blc8
    bound_row = b["fluid"] + blc * b["layer_cells"]
    return name, {
        "geometry_id": gid, "family": mrow["family"],
        "wall_level": ri["wall_level"], "base_size": ri["base_size"],
        "extent": list(ri["extent"]),
        "n_base": b["n_base"], "touched": b["touched"], "splits": b["splits"],
        "leaves": b["leaves"], "fluid": b["fluid"],
        "wall_faces": b["wall_faces"], "layer_cells": b["layer_cells"],
        "bound": b["bound"], "blc8": blc, "bound_row": bound_row,
        "n_cells": n_cells, "b0_cells": b0_cells,
        "bound_over_b0": b["bound"] / b0_cells,
        "bounds": bool(bound_row <= n_cells)}


def _both_pass(gs: dict, gb: dict, want) -> list:
    """Exactly evaluate.g_cost's filter, then the --ids restriction."""
    return sorted(gid for gid in gs
                  if gid in gb and gs[gid]["failure"] is False
                  and gb[gid]["failure"] is False
                  and gs[gid]["n_cells"] is not None
                  and gb[gid]["n_cells"] is not None
                  and (want is None or gid in want))


def report(sets=("tuning", "seed1-posthoc"), jobs=6, ids=None,
           work_dir=None) -> dict:
    """CB-9: bound every both-pass row of each set; the COSTBOUND data."""
    want = None
    if ids is not None:
        want = set(str(s) for s in ids)
        if not want:
            raise CostBoundError("ids: an empty id list")
    tmp = None
    if work_dir is None:
        tmp = tempfile.mkdtemp(prefix="costbound-")
        work_dir = tmp
    os.makedirs(work_dir, exist_ok=True)
    data = {"schema": SCHEMA_ID, "n_layers": N_LAYERS,
            "cells_ratio_max": evaluate.CELLS_RATIO_MAX, "bundles": {},
            "sets": {}}
    prep = []
    seen = set()
    for name in sets:
        cfg = _SETS[name]
        system = _load_bundle(cfg["system"])
        b0 = _load_bundle(cfg["b0"])
        gs = dict((g["geometry_id"], g) for g in system["geometries"])
        gb = dict((g["geometry_id"], g) for g in b0["geometries"])
        data["bundles"][name] = {"system": _sha256(cfg["system"]),
                                 "b0": _sha256(cfg["b0"])}
        data["sets"][name] = {"label": cfg["label"]}
        mrows = dict((r["geometry_id"], r)
                     for r in split.load(cfg["manifest"][0],
                                         cfg["manifest"][1]))
        for gid in _both_pass(gs, gb, want):
            seen.add(gid)
            if gid not in mrows:
                raise CostBoundError("%s: no manifest row" % gid)
            prep.append((name, gid, mrows[gid], row_inputs(system, gid),
                         gs[gid].get("blc8_a_priori"), gs[gid]["n_cells"],
                         gb[gid]["n_cells"]))
    if want is not None:
        missing = sorted(want.difference(seen))
        if missing:
            raise CostBoundError("not a both-pass row of any selected set: %s"
                                 % ", ".join(missing))
    rows = {}
    done = 0
    with ProcessPoolExecutor(max_workers=max(1, int(jobs))) as ex:
        futs = [ex.submit(_one_row, j, work_dir) for j in prep]
        for fu in as_completed(futs):
            name, row = fu.result()
            rows.setdefault(name, {})[row["geometry_id"]] = row
            done += 1
            print("[costbound] %d/%d (%d%%) %s: bound_row %d vs n_cells %d"
                  % (done, len(prep), 100 * done // max(len(prep), 1),
                     row["geometry_id"], row["bound_row"], row["n_cells"]),
                  flush=True)
    for name in sets:
        rl = [rows.get(name, {}).get(g) for g in sorted(rows.get(name, {}))]
        rl = [r for r in rl if r is not None]
        data["sets"][name].update(_set_stats(rl))
    data["verdict"] = dict((n, "attainable" if data["sets"][n]["attainable"]
                            else "not-attainable") for n in sets)
    if tmp is not None:
        shutil.rmtree(tmp, ignore_errors=True)
    return data


def _set_stats(rows: list) -> dict:
    """CB-9: the medians and attainability over one set's rows."""
    if not rows:
        return {"rows": [], "n_rows": 0, "n_bounds": 0, "median_bound": None,
                "median_fluid": None, "median_b0": None, "median_cells": None,
                "ratio": None, "ratio_octree": None, "attainable": False}
    mb = statistics.median([r["bound"] for r in rows])
    mf = statistics.median([r["fluid"] for r in rows])
    m0 = statistics.median([r["b0_cells"] for r in rows])
    mc = statistics.median([r["n_cells"] for r in rows])
    ratio = (mb / m0) if m0 else None
    return {"rows": rows, "n_rows": len(rows),
            "n_bounds": sum(1 for r in rows if r["bounds"]),
            "median_bound": mb, "median_fluid": mf, "median_b0": m0,
            "median_cells": mc, "ratio": ratio,
            "ratio_octree": (mf / m0) if m0 else None,
            "attainable": bool(ratio is not None
                               and ratio <= evaluate.CELLS_RATIO_MAX)}


def _fmt(v) -> str:
    return "%g" % (v,)


def _markdown(data: dict) -> str:
    """COSTBOUND.md: one table per set plus the measured result lines."""
    out = ["# COSTBOUND - the G-COST lower bound (AUT-06)", "",
           "Per geometry the fewest cells of ANY octree mesh that holds the",
           "R-WIN wall level on the whole wall with 2:1 shells out to the",
           "base grid and %d layers (CB-1..CB-7); a row bounds when fluid +"
           % data["n_layers"],
           "blc8 x layer_cells <= its n_cells (CB-8); attainable =",
           "median(bound)/median(B0) <= %s (CB-9), G-COST's own constant"
           % _fmt(evaluate.CELLS_RATIO_MAX),
           "read from evaluate.py.  This changes no rule, no gate and no",
           "constant; where it is not attainable the decision (D-8) is the",
           "user's.", ""]
    for name in sorted(data["sets"]):
        st = data["sets"][name]
        out.append("## %s" % name)
        out.append("")
        out.append(st["label"])
        out.append("")
        out.append("| geometry | family | wall level | n_base | touched | "
                   "splits | leaves | fluid | wall_faces | layer_cells | "
                   "bound | blc8 | bound_row | n_cells | b0_cells | "
                   "bound/b0 | bounds |")
        out.append("|" + "---|" * 17)
        for r in st["rows"]:
            out.append("| %s | %s | %d | %d | %d | %d | %d | %d | %d | %d "
                       "| %d | %g | %d | %d | %d | %.2f | %s |"
                       % (r["geometry_id"], r["family"], r["wall_level"],
                          r["n_base"], r["touched"], r["splits"],
                          r["leaves"], r["fluid"], r["wall_faces"],
                          r["layer_cells"], r["bound"], r["blc8"],
                          r["bound_row"], r["n_cells"], r["b0_cells"],
                          r["bound_over_b0"],
                          "yes" if r["bounds"] else "NO"))
        out.append("")
        out.append("%s: bound <= actual on %d/%d rows"
                   % (name, st["n_bounds"], st["n_rows"]))
        if st["ratio"] is not None:
            out.append("G-COST %sx attainable on %s: %s (median bound %s / "
                       "median B0 %s = %.1f)"
                       % (_fmt(evaluate.CELLS_RATIO_MAX), name,
                          "yes" if st["attainable"] else "no",
                          _fmt(st["median_bound"]), _fmt(st["median_b0"]),
                          st["ratio"]))
            if not st["attainable"]:
                out.append("D-8: the gate goes to the user; the constant is "
                           "not touched.")
        else:
            out.append("G-COST %sx attainable on %s: no rows"
                       % (_fmt(evaluate.CELLS_RATIO_MAX), name))
        out.append("")
    return "\n".join(out) + "\n"


def _write(out_dir: str, data: dict) -> str:
    """COSTBOUND.json (indent 1, sorted, newline at end) and COSTBOUND.md."""
    os.makedirs(out_dir, exist_ok=True)
    p = os.path.join(out_dir, "COSTBOUND.json")
    with open(p, "w", encoding="utf-8", newline="\n") as f:
        json.dump(data, f, indent=1, sort_keys=True)
        f.write("\n")
    with open(os.path.join(out_dir, "COSTBOUND.md"), "w", encoding="utf-8",
              newline="\n") as f:
        f.write(_markdown(data))
    return p


def _print_summary(data: dict, out_dir: str, seconds: float) -> None:
    print("wrote %s and %s (%.1f s wall)"
          % (os.path.join(out_dir, "COSTBOUND.json"),
             os.path.join(out_dir, "COSTBOUND.md"), seconds))
    for name in sorted(data["sets"]):
        st = data["sets"][name]
        print("%s: bound <= actual on %d/%d rows"
              % (name, st["n_bounds"], st["n_rows"]))
        if st["ratio"] is not None:
            print("G-COST %sx attainable on %s: %s (median bound %s / "
                  "median B0 %s = %.1f, median fluid %s, median cells %s, "
                  "ratio_octree %s)"
                  % (_fmt(evaluate.CELLS_RATIO_MAX), name,
                     "yes" if st["attainable"] else "no",
                     _fmt(st["median_bound"]), _fmt(st["median_b0"]),
                     st["ratio"], _fmt(st["median_fluid"]),
                     _fmt(st["median_cells"]), _fmt(st["ratio_octree"])))
        else:
            print("G-COST %sx attainable on %s: no rows"
                  % (_fmt(evaluate.CELLS_RATIO_MAX), name))


# --- the selftest ------------------------------------------------------------

_FIXTURES = (
    ("F0", [0, 8, 0, 8, 0, 8], 1.0, (2, 2, 2, 6, 6, 6), 0,
     dict(n_base=512, touched=208, splits=0, leaves=512, fluid=448,
          wall_faces=96, layer_cells=768, bound=1216)),
    ("F1", [0, 8, 0, 8, 0, 8], 1.0, (2, 2, 2, 6, 6, 6), 1,
     dict(n_base=512, touched=784, splits=208, leaves=1968, fluid=1512,
          wall_faces=384, layer_cells=3072, bound=4584)),
    ("F2", [0, 8, 0, 8, 0, 8], 1.0, (2, 2, 2, 6, 6, 6), 2,
     dict(n_base=512, touched=3088, splits=992, leaves=7456, fluid=4928,
          wall_faces=1536, layer_cells=12288, bound=17216)),
    ("F3", [0, 10, 0, 8, 0, 8], 1.0, (3, 2, 3, 5, 6, 5), 3,
     dict(n_base=640, touched=5136, splits=1728, leaves=12736, fluid=8240,
          wall_faces=2560, layer_cells=20480, bound=28720)),
    ("F4", [0, 8, 0, 8, 0, 8], 1.0, (2.25, 2.5, 2.75, 5.75, 5.5, 5.25), 3,
     dict(n_base=512, touched=6864, splits=2480, leaves=17872, fluid=11376,
          wall_faces=3424, layer_cells=27392, bound=38768)),
    ("F5", [0, 4, 0, 2, 0, 2], 0.5, (1, 0.5, 0.5, 3, 1.5, 1.5), 2,
     dict(n_base=128, touched=1296, splits=432, leaves=3152, fluid=2296,
          wall_faces=640, layer_cells=5120, bound=7416)),
)


def _box_tris(box, m=1):
    """A box as 12 m^2 triangles, wound outward: each face an m x m grid of
    squares, each square split along the diagonal through its
    lexicographically smallest corner.  Returns (P, T)."""
    x0, y0, z0, x1, y1, z1 = box
    bnds = ((x0, x1), (y0, y1), (z0, z1))
    tri = []
    for axis in range(3):
        u_ax, w_ax = [o for o in range(3) if o != axis]
        for val in bnds[axis]:
            n_out = [0.0, 0.0, 0.0]
            n_out[axis] = 1.0 if val == bnds[axis][1] else -1.0
            for iu in range(m):
                for iw in range(m):
                    us = (bnds[u_ax][0] + (bnds[u_ax][1] - bnds[u_ax][0])
                          * iu / m, bnds[u_ax][0]
                          + (bnds[u_ax][1] - bnds[u_ax][0]) * (iu + 1) / m)
                    ws = (bnds[w_ax][0] + (bnds[w_ax][1] - bnds[w_ax][0])
                          * iw / m, bnds[w_ax][0]
                          + (bnds[w_ax][1] - bnds[w_ax][0]) * (iw + 1) / m)
                    cyc = []
                    for u in (us[0], us[1]):
                        for w in (ws[0], ws[1]):
                            p = [0.0, 0.0, 0.0]
                            p[axis] = float(val)
                            p[u_ax] = u
                            p[w_ax] = w
                            cyc.append(p)
                    cyc = [cyc[0], cyc[1], cyc[3], cyc[2]]
                    e1 = [cyc[1][d] - cyc[0][d] for d in range(3)]
                    e2 = [cyc[2][d] - cyc[0][d] for d in range(3)]
                    cr = (e1[1] * e2[2] - e1[2] * e2[1],
                          e1[2] * e2[0] - e1[0] * e2[2],
                          e1[0] * e2[1] - e1[1] * e2[0])
                    if sum(cr[d] * n_out[d] for d in range(3)) < 0.0:
                        cyc = [cyc[0], cyc[3], cyc[2], cyc[1]]
                    tri.append((cyc[0], cyc[1], cyc[2]))
                    tri.append((cyc[0], cyc[2], cyc[3]))
    Q = np.array([v for t in tri for v in t], dtype=np.float64)
    Pu, inv = np.unique(Q, axis=0, return_inverse=True)
    return Pu, np.ascontiguousarray(inv, dtype=np.int64).reshape(-1, 3)


def _ancestor_splits(touched: set, lat: dict, level: int) -> int:
    """CB-3's ancestor sets alone (the neighbour term dropped): splits."""
    cur = set(_parent_key(c, _dims(lat, level)) for c in touched)
    tot = len(cur)
    for l in range(level - 1, 0, -1):
        cur = set(_parent_key(c, _dims(lat, l)) for c in cur)
        tot += len(cur)
    return tot


def selftest() -> int:
    """Seven groups; [ok] per group, then SELFTEST PASS (exit 0)."""
    for fid, ext, b, box, lev, want in _FIXTURES:
        P, T = _box_tris(box)
        got = bound(P, T, ext, b, lev)
        assert got == want, "%s: %r != %r" % (fid, got, want)
    print("[ok] fixtures: F0-F5 every key equal (ints, exact)")
    f4 = _FIXTURES[4]
    Pf, Tf = _box_tris(f4[3])
    lat4 = lattice(f4[1], f4[2])
    S4 = touched_cells(Pf, Tf, lat4, f4[4])
    anc = _ancestor_splits(S4, lat4, f4[4])
    full = sum(len(r) for r in split_sets(S4, lat4, f4[4]))
    assert (anc, full) == (2072, 2480), "F4 splits %d/%d" % (anc, full)
    print("[ok] F4 balance: the ancestor sets alone 2072, CB-3 2480")
    P3, T3 = _box_tris(_FIXTURES[1][3], m=4)
    assert T3.shape[0] == 192, "grid box has %d triangles" % T3.shape[0]
    got = bound(P3, T3, _FIXTURES[1][1], _FIXTURES[1][2], _FIXTURES[1][4])
    assert got == _FIXTURES[1][5], "F1 on the grid box: %r" % got
    print("[ok] F1 on the 192-triangle box: the same row")
    lat1 = lattice(_FIXTURES[1][1], _FIXTURES[1][2])
    nl = _dims(lat1, 1)
    h = [s / 2 for s in lat1["s"]]
    ii, jj, kk = np.meshgrid(np.arange(nl[0]), np.arange(nl[1]),
                             np.arange(nl[2]), indexing="ij")
    pts = np.stack([lat1["lo"][0] + (ii + 0.5) * h[0],
                    lat1["lo"][1] + (jj + 0.5) * h[1],
                    lat1["lo"][2] + (kk + 0.5) * h[2]],
                   axis=-1).reshape(-1, 3)
    P1, T1 = _box_tris(_FIXTURES[1][3])
    n_solid = int(inside(P1, T1, pts).sum())
    assert pts.shape[0] == 4096 and n_solid == 512, "solid %d" % n_solid
    print("[ok] inside: 512 of the 4096 level-1 centres solid "
          "(indices 4..11 on every axis, the face diagonals included)")
    P0, T0 = _box_tris((2, 2, 2, 6, 6, 6))
    lat0 = lattice([0, 8, 0, 8, 0, 8], 1.0)
    bad = [lambda: lattice([0, 1, 0, 1, 1, 0], 1.0),
           lambda: lattice([0, 1, 0, 1, 0, 1], 0.0),
           lambda: touched_cells(P0, T0, lat0, -1),
           lambda: leaf_arrays([], lat0, -1),
           lambda: bound(P0, T0, [0, 1, 0, 1, 1, 0], 1.0, 0)]
    for fn in bad:
        try:
            fn()
        except CostBoundError:
            continue
        raise AssertionError("a bad input was not refused")
    print("[ok] refusals: level -1, base_size 0, extent hi <= lo")
    return _selftest_rows()


def _selftest_rows() -> int:
    """Groups (6) and (7): the recorded E-1-004 inputs and the two runs."""
    ri = row_inputs(_load_bundle(_SETS["tuning"]["system"]), "E-1-004")
    want = {"extent": [-1.0507, 1.9513, -1.2008, 1.0507, -0.9006, 0.9006],
            "base_size": 0.1501, "wall_level": 4}
    assert ri == want, "row_inputs E-1-004: %r" % ri
    assert lattice(ri["extent"], ri["base_size"])["n_base"] == 3600
    print("[ok] row_inputs: E-1-004 wall level 4, base 0.1501, n_base 3600")
    dirs = [tempfile.mkdtemp(prefix="cb-st-" + t + "-")
            for t in ("a", "b", "w1", "w2")]
    d1, d2, w1, w2 = dirs
    try:
        args = dict(ids=["B-1-009", "E-1-004"], sets=("tuning",), jobs=2)
        p1 = _write(d1, report(work_dir=w1, **args))
        p2 = _write(d2, report(work_dir=w2, **args))
        with open(p1, "rb") as f:
            b1 = f.read()
        with open(p2, "rb") as f:
            b2 = f.read()
        assert b1 == b2, "the two report runs differ"
        r1 = json.loads(b1.decode("utf-8"))
        rows = r1["sets"]["tuning"]["rows"]
        assert len(rows) == 2 and all(r["bounds"] for r in rows), rows
        print("[ok] report: byte-identical twice, both rows bound")
    finally:
        for d in dirs:
            shutil.rmtree(d, ignore_errors=True)
    print("SELFTEST PASS")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="the G-COST lower bound (AUT-06)")
    ap.add_argument("--selftest", action="store_true",
                    help="the seven selftest groups")
    ap.add_argument("--report", action="store_true",
                    help="write costbound/COSTBOUND.json and COSTBOUND.md")
    ap.add_argument("--jobs", type=int, default=6,
                    help="report workers (default 6)")
    ap.add_argument("--ids", help="comma-separated geometry ids")
    ap.add_argument("--work", help="the surface work directory")
    ap.add_argument("--out-dir",
                    help="the report out dir (default %s)" % OUT_DIR)
    ap.add_argument("--id", help="print one geometry's row JSON")
    ap.add_argument("--set", default="tuning", choices=sorted(_SETS),
                    help="the set of --id (default tuning)")
    a = ap.parse_args(argv)
    try:
        if a.selftest:
            return selftest()
        if a.id:
            data = report(sets=(a.set,), ids=[a.id], work_dir=a.work)
            print(json.dumps(data["sets"][a.set]["rows"][0], indent=1,
                             sort_keys=True))
            return 0
        if a.report:
            ids = None
            if a.ids:
                ids = [s.strip() for s in a.ids.split(",") if s.strip()]
                if not a.out_dir:
                    raise CostBoundError("--ids: give --out-dir (the default "
                                         "out dir is refused)")
            out_dir = a.out_dir or OUT_DIR
            t0 = time.time()
            data = report(sets=("tuning", "seed1-posthoc"), jobs=a.jobs,
                          ids=ids, work_dir=a.work)
            _write(out_dir, data)
            _print_summary(data, out_dir, time.time() - t0)
            tuning_rows = data["sets"].get("tuning", {}).get("rows", [])
            return 0 if all(r["bounds"] for r in tuning_rows) else 1
    except CostBoundError as e:
        print("COSTBOUND REFUSED: %s" % e, file=sys.stderr)
        return 2
    ap.error("choose --selftest, --report or --id")


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    sys.exit(main())
