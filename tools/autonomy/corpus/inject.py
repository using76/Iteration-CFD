#!/usr/bin/env python3
# meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
# Source-available, not Open Source. See LICENSE at the repository root.
# No GPL-licensed source was consulted.
"""inject - docs/15 §E family G: 120 defect geometries, docs/15 §F G-CORPUS.

Six defect classes, 20 each at every seed, injected into FRESH parent
geometries of families A, B, D, E and F at index 1000 + index - outside the
A-F corpus index range, so no G row shares its geometry with any corpus row:

    hole           a triangle disk whose removal opens a k-edge loop, k in 3..64
    flip           a disk patch of m in 1..32 triangles written corners-reversed
    duplicate      d in 1..8 facets copied verbatim right after themselves
    signed_zero    z in 1..8 corner occurrences planted with -0.0 on a zero axis
    near_duplicate z in 1..8 corners moved by 0.1-0.4 of the weld tolerance
    t_junction     j in 1..8 edges split on one triangle side only

Every row carries the exact counts the injection plants ("expect"), from the
measured formulas above, and build_bytes verifies them before writing: once
against a bit-exact in-memory re-read of the written bytes (reread_counts,
numpy only) and once against the recomputed expect.  Nothing is repaired,
dropped or flipped to make a check pass.  The strata are A PRIORI classes -
what stl_repair's own repertoire can close on paper (a tolerance weld the
easy ones, a local fill the medium ones, nothing the hard ones); gate.py's
G-CORPUS run REPORTS what stl_repair actually closed, it does not gate it.

    python tools/autonomy/corpus/inject.py --seed 1 --n 120 --out DIR
    python tools/autonomy/corpus/inject.py --selftest
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
import re
import shutil
import sys
import tempfile

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (_HERE, os.path.dirname(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import meshkit  # noqa: E402
import schema  # noqa: E402
import stl_io  # noqa: E402

FAMILY = "G"
GENERATOR = "corpus/inject.py"
SALT = 16
PARENTS = {"A": "gen_wing", "B": "gen_lathe", "D": "gen_bluff",
           "E": "gen_gap", "F": "gen_thin"}
PARENT_FAMILIES = ("A", "B", "D", "E", "F")
DEFECTS = ("hole", "flip", "duplicate", "signed_zero", "near_duplicate",
           "t_junction")
PARENT_OFFSET = 1000
WELD_REL = 1e-6
MAX_HOLE_EDGES = 32
COUNT_RANGE = {"hole": (3, 64), "flip": (1, 32), "duplicate": (1, 8),
               "signed_zero": (1, 8), "near_duplicate": (1, 8),
               "t_junction": (1, 8)}
FRAC_RANGE = (0.1, 0.4)
EXPECT_KEYS = ("open_edges", "non_manifold_edges", "triangles_in",
               "points_bit_exact", "weld_merged", "negzero", "per_hole")


# --- the site machinery -----------------------------------------------------


def edge_map(T) -> dict:
    """Every undirected edge (min, max) -> the triangles using it (2 on a
    closed parent)."""
    emap = {}
    for ti in range(len(T)):
        a, b, c = int(T[ti][0]), int(T[ti][1]), int(T[ti][2])
        for u, v in ((a, b), (b, c), (c, a)):
            emap.setdefault((u, v) if u < v else (v, u), []).append(ti)
    return emap


def adjacency(emap: dict, nP: int) -> list:
    """adj[p] = the points sharing an edge with p."""
    adj = [set() for _ in range(nP)]
    for a, b in emap:
        adj[a].add(b)
        adj[b].add(a)
    return adj


def grow_disk(T, n_tri, rng, emap):
    for _attempt in range(64):
        t0 = int(rng.integers(len(T)))
        region, in_reg, verts = [t0], {t0}, set(int(v) for v in T[t0])
        while len(region) < n_tri:
            cand = set()
            for ti in region:
                for k in range(3):
                    a, b = int(T[ti][k]), int(T[ti][(k + 1) % 3])
                    for o in emap[(min(a, b), max(a, b))]:
                        if o in in_reg:
                            continue
                        third = [int(v) for v in T[o] if int(v) not in (a, b)]
                        if len(third) == 1 and third[0] not in verts:
                            cand.add(o)
            if not cand:
                break
            o = sorted(cand)[int(rng.integers(len(cand)))]
            region.append(o)
            in_reg.add(o)
            verts.update(int(v) for v in T[o])
        if len(region) == n_tri:
            return region
    raise RuntimeError("grow_disk: no disk of %d triangles after 64 seeds"
                       % n_tri)


def _inject_hole(P, T, count, rng):
    """Remove the triangles of a (count - 2)-triangle disk: one open loop of
    `count` edges, everything else untouched."""
    emap = edge_map(T)
    region = grow_disk(T, count - 2, rng, emap)
    keep = np.ones(len(T), dtype=bool)
    keep[np.array(region, dtype=np.int64)] = False
    C = P[T[keep]]
    expect = {"open_edges": int(count), "non_manifold_edges": 0,
              "triangles_in": int(len(T) - (count - 2)),
              "points_bit_exact": int(len(P)), "weld_merged": 0,
              "negzero": 0, "per_hole": [int(count)]}
    return C, expect, {"region": region}


def _inject_flip(P, T, count, rng):
    """Write a `count`-triangle disk patch corners-reversed in place: every
    boundary edge of the patch is then used twice in the same direction."""
    emap = edge_map(T)
    region = grow_disk(T, count, rng, emap)
    C = P[T].copy()
    for t in region:
        C[t] = C[t][[0, 2, 1]]
    expect = {"open_edges": 0, "non_manifold_edges": int(count + 2),
              "triangles_in": int(len(T)), "points_bit_exact": int(len(P)),
              "weld_merged": 0, "negzero": 0, "per_hole": []}
    return C, expect, {"region": region}


def _inject_duplicate(P, T, count, rng):
    """Copy `count` vertex-disjoint facets verbatim, right after themselves:
    each copy's 3 edges are then used 3 times."""
    order = rng.permutation(len(T))
    used, picks = set(), []
    for e in order:
        t = int(e)
        pts = set(int(v) for v in T[t])
        if pts & used:
            continue
        used |= pts
        picks.append(t)
        if len(picks) == count:
            break
    if len(picks) < count:
        raise RuntimeError("duplicate: only %d of %d sites" % (len(picks), count))
    pick_set = set(picks)
    blocks = []
    for ti in range(len(T)):
        blocks.append(P[T[ti]])
        if ti in pick_set:
            blocks.append(P[T[ti]])
    C = np.array(blocks)
    expect = {"open_edges": 0, "non_manifold_edges": int(3 * count),
              "triangles_in": int(len(T) + count),
              "points_bit_exact": int(len(P)), "weld_merged": 0,
              "negzero": 0, "per_hole": []}
    return C, expect, {"picks": picks}


def _inject_signed_zero(P, T, count, rng):
    """Plant -0.0 on the first zero axis of `count` corner occurrences: the
    bit-exact weld splits each point in two (4 open edges, one 4-edge loop
    per site) and the tolerance weld merges them back.  Taken triangles are
    also pairwise vertex-DISJOINT, so no two sites' 4-edge loops share a
    vertex - that keeps the loop walk's "single outgoing edge" true (two
    loops sharing a corner would be indistinguishable there)."""
    adj = adjacency(edge_map(T), len(P))
    order = rng.permutation(3 * len(T))
    used_t, used_p, frozen, sites = set(), set(), set(), []
    for e in order:
        ti, j = divmod(int(e), 3)
        p = int(T[ti][j])
        pts = frozenset(int(v) for v in T[ti])
        if (ti in used_t or p in used_p or (adj[p] & used_p)
                or (pts & frozen)):
            continue
        zeros = [ax for ax in range(3) if P[p, ax] == 0.0]
        if not zeros:
            continue
        used_t.add(ti)
        used_p.add(p)
        frozen |= pts
        sites.append((ti, j, p, zeros[0]))
        if len(sites) == count:
            break
    if len(sites) < count:
        raise RuntimeError("signed_zero: only %d of %d sites" % (len(sites), count))
    C = P[T].copy()
    for ti, j, _p, ax in sites:
        C[ti, j, ax] = -0.0
    expect = {"open_edges": int(4 * count), "non_manifold_edges": 0,
              "triangles_in": int(len(T)),
              "points_bit_exact": int(len(P) + count), "weld_merged": int(count),
              "negzero": int(count), "per_hole": []}
    return C, expect, {"sites": sites}


def _inject_near_duplicate(P, T, count, frac, rng):
    """Move `count` corner occurrences by frac * tol along their first axis
    that has room inside the bbox: a new bit-pattern the tolerance weld
    merges back (max_move = frac * tol, never over half the tolerance).
    Taken triangles are pairwise vertex-disjoint, as in signed_zero."""
    adj = adjacency(edge_map(T), len(P))
    lo, hi = P.min(axis=0), P.max(axis=0)
    diag = float(np.linalg.norm(hi - lo))
    tol = WELD_REL * diag
    order = rng.permutation(3 * len(T))
    used_t, used_p, frozen, sites = set(), set(), set(), []
    for e in order:
        ti, j = divmod(int(e), 3)
        p = int(T[ti][j])
        pts = frozenset(int(v) for v in T[ti])
        if (ti in used_t or p in used_p or (adj[p] & used_p)
                or (pts & frozen)):
            continue
        ax = -1
        for cand in range(3):
            if lo[cand] < P[p, cand] and P[p, cand] + frac * tol < hi[cand]:
                ax = cand
                break
        if ax < 0:
            continue
        used_t.add(ti)
        used_p.add(p)
        frozen |= pts
        sites.append((ti, j, p, ax))
        if len(sites) == count:
            break
    if len(sites) < count:
        raise RuntimeError("near_duplicate: only %d of %d sites" % (len(sites), count))
    C = P[T].copy()
    for ti, j, p, ax in sites:
        C[ti, j, ax] = P[p, ax] + frac * tol
    expect = {"open_edges": int(4 * count), "non_manifold_edges": 0,
              "triangles_in": int(len(T)),
              "points_bit_exact": int(len(P) + count), "weld_merged": int(count),
              "negzero": 0, "per_hole": []}
    return C, expect, {"sites": sites}


def _inject_t_junction(P, T, count, rng):
    """Split the shared edge of two triangles on ONE side only: t1 becomes
    (u, M, w), (M, v, w) with M the edge midpoint, t2 keeps the whole edge -
    a 3-edge loop per site that only a T-junction-aware repair could close."""
    nP, nT = len(P), len(T)
    emap = edge_map(T)
    diag = float(np.linalg.norm(P.max(axis=0) - P.min(axis=0)))
    tol = WELD_REL * diag
    keys = sorted(emap)
    order = rng.permutation(len(keys))
    used, sites = set(), []
    for e in order:
        a, b = keys[int(e)]
        tris = sorted(emap[(a, b)])
        t1, t2 = tris[0], tris[1]
        pts = set(int(v) for v in T[t1]) | set(int(v) for v in T[t2])
        if pts & used:
            continue
        M = stl_io.canonical(0.5 * (P[a] + P[b]))
        if float(np.linalg.norm(P - M, axis=1).min()) <= 10.0 * tol:
            continue
        used |= pts
        tk = (int(T[t1][0]), int(T[t1][1]), int(T[t1][2]))
        if (tk[0], tk[1]) in ((a, b), (b, a)):
            u, v, w = tk
        elif (tk[1], tk[2]) in ((a, b), (b, a)):
            u, v, w = tk[1], tk[2], tk[0]
        else:
            u, v, w = tk[2], tk[0], tk[1]
        sites.append((t1, u, v, w, M))
        if len(sites) == count:
            break
    if len(sites) < count:
        raise RuntimeError("t_junction: only %d of %d sites" % (len(sites), count))
    by_t1 = {s[0]: s for s in sites}
    blocks = []
    for ti in range(nT):
        if ti in by_t1:
            _t1, u, v, w, M = by_t1[ti]
            blocks.append(np.array([P[u], M, P[w]]))
            blocks.append(np.array([M, P[v], P[w]]))
        else:
            blocks.append(P[T[ti]])
    C = np.array(blocks)
    expect = {"open_edges": int(3 * count), "non_manifold_edges": 0,
              "triangles_in": int(nT + count), "points_bit_exact": int(nP + count),
              "weld_merged": 0, "negzero": 0, "per_hole": [3] * int(count)}
    return C, expect, {"sites": sites}


# --- the corner writer and the bit-exact re-read oracle ---------------------


def _corner_normals(C):
    """stl_io.unit_normals on the corner array: canonical in, cross, unit,
    canonical out; a zero-area triangle is refused by name."""
    Cc = stl_io.canonical(C)
    a, b, c = Cc[:, 0], Cc[:, 1], Cc[:, 2]
    n = np.cross(b - a, c - a)
    norm = np.linalg.norm(n, axis=1)
    for i, z in enumerate(norm):
        if z == 0.0:
            raise ValueError("degenerate triangle %d: zero area" % i)
    return stl_io.canonical(n / norm[:, None])


def corner_bytes(C, solid: str) -> bytes:
    """The SAME text as stl_io.stl_bytes, but the vertex values are printed
    AS GIVEN (no canonical) - that is how a planted -0.0 reaches the file.
    The normals go through stl_io.canonical, so the parent's bytes are
    identical to stl_io.stl_bytes(P, T, solid) on a clean parent."""
    C = np.asarray(C, dtype=np.float64)
    if C.ndim != 3 or C.shape[1:] != (3, 3):
        raise ValueError("corners: expected (nT, 3, 3), got %r" % (C.shape,))
    if not np.isfinite(C).all():
        raise ValueError("corners: non-finite coordinate")
    N = _corner_normals(C)
    parts = ["solid %s\n" % solid]
    for i in range(len(C)):
        parts.append("facet normal %.9e %.9e %.9e\n"
                     % (N[i, 0], N[i, 1], N[i, 2]))
        parts.append("outer loop\n")
        for j in range(3):
            p = C[i, j]
            parts.append("vertex %.9e %.9e %.9e\n" % (p[0], p[1], p[2]))
        parts.append("endloop\n")
        parts.append("endfacet\n")
    parts.append("endsolid %s\n" % solid)
    return "".join(parts).encode("ascii")


def reread_counts(data: bytes) -> dict:
    """The in-memory oracle: parse the vertex lines back, weld bit-exactly
    (numpy only, independent of stl_repair), count.  per_hole walks the open
    DIRECTED edges - the bit-exact state BEFORE any tolerance weld."""
    verts, n_facets = [], 0
    for ln in data.split(b"\n"):
        if ln.startswith(b"vertex "):
            parts = ln.split()
            verts.append([float(parts[1]), float(parts[2]), float(parts[3])])
        elif ln.startswith(b"facet "):
            n_facets += 1
    corners = np.array(verts, dtype=np.float64).reshape(-1, 3)
    keys = np.ascontiguousarray(corners).view(np.uint64).reshape(-1, 3)
    uniq, inv = np.unique(keys, axis=0, return_inverse=True)
    n_unique = int(len(uniq))
    Tb = np.asarray(inv, dtype=np.int64).reshape(-1, 3)
    rep = stl_io.edge_report(Tb, n_unique)
    a = Tb[:, [0, 1, 2]].reshape(-1)
    b = Tb[:, [1, 2, 0]].reshape(-1)
    key = np.minimum(a, b) * np.int64(n_unique) + np.maximum(a, b)
    _uk, inv2, counts = np.unique(key, return_inverse=True, return_counts=True)
    is_open = np.asarray(counts)[np.asarray(inv2)] == 1
    open_idx = np.nonzero(is_open)[0]
    out_map = {}
    for e in open_idx:
        out_map.setdefault(int(a[e]), []).append(int(e))
    used, loops = set(), []
    for e0 in open_idx:
        e0 = int(e0)
        if e0 in used:
            continue
        used.add(e0)
        start, cur, n = int(a[e0]), int(b[e0]), 1
        while cur != start:
            outs = out_map.get(cur, [])
            if len(outs) != 1:
                raise RuntimeError("reread_counts: boundary vertex with %d "
                                   "outgoing open edges" % len(outs))
            ne = int(outs[0])
            used.add(ne)
            cur = int(b[ne])
            n += 1
        loops.append(n)
    return {"open_edges": rep["open"],
            "non_manifold_edges": rep["non_manifold"] + rep["same_direction"],
            "triangles_in": n_facets, "points_bit_exact": n_unique,
            "negzero": int(data.count(b"-0.")), "per_hole": sorted(loops)}


def _oracle_diffs(got, exp, defect: str, count: int) -> list:
    """Every key where the re-read disagrees with expect.  For signed_zero
    and near_duplicate the bit-exact walk still sees the [4]*count loops a
    planted corner makes; the tolerance weld closes them BEFORE stl_repair's
    hole stage (measured: "signed zeros and near-duplicates are welded
    closed"), which is why expect.per_hole is empty for those two - the
    gate checks per_hole against stl_repair's post-weld hole report."""
    bad = []
    for k in ("open_edges", "non_manifold_edges", "triangles_in",
              "points_bit_exact", "negzero"):
        if got[k] != exp[k]:
            bad.append("%s %r != expect %r" % (k, got[k], exp[k]))
    want = exp["per_hole"]
    if defect in ("signed_zero", "near_duplicate"):
        want = [4] * int(count)
    if got["per_hole"] != want:
        bad.append("per_hole %r != expect %r" % (got["per_hole"], want))
    return bad


# --- params: sampling, validation, rows -------------------------------------


def parent_module(params):
    """The parent generator module, loaded by name (docs/15 §E)."""
    return importlib.import_module(PARENTS[params["parent_family"]])


def _run_defect(defect, P, T, count, rng, frac=None):
    if defect == "hole":
        return _inject_hole(P, T, count, rng)
    if defect == "flip":
        return _inject_flip(P, T, count, rng)
    if defect == "duplicate":
        return _inject_duplicate(P, T, count, rng)
    if defect == "signed_zero":
        return _inject_signed_zero(P, T, count, rng)
    if defect == "near_duplicate":
        return _inject_near_duplicate(P, T, count, frac, rng)
    return _inject_t_junction(P, T, count, rng)


def sample_params(seed, index) -> dict:
    """The only draw is meshkit.uniforms(SALT, seed, index); the injection
    itself draws from default_rng([SALT, site_seed]) so build_bytes can
    replay it exactly."""
    seed, index = int(seed), int(index)
    U = meshkit.uniforms(SALT, seed, index)
    defect = DEFECTS[index % 6]
    parent_family = PARENT_FAMILIES[(index // 6) % 5]
    parent_seed = seed
    parent_index = PARENT_OFFSET + index
    pgen = importlib.import_module(PARENTS[parent_family])
    params = {"defect": defect, "parent_family": parent_family,
              "parent_seed": parent_seed, "parent_index": parent_index,
              "parent_id": pgen.geometry_id(parent_seed, parent_index),
              "count": meshkit.ik(U, 0, *COUNT_RANGE[defect]),
              "site_seed": meshkit.ik(U, 2, 0, 2 ** 31 - 1)}
    if defect == "near_duplicate":
        params["frac"] = meshkit.fl(meshkit.lin(U, 1, 0.1, 0.4), 4)
    pp = pgen.sample_params(parent_seed, parent_index)
    P, T = pgen.build(pp)
    params["parent_sha256"] = hashlib.sha256(
        stl_io.stl_bytes(P, T, pgen.SOLID)).hexdigest()
    rng = np.random.default_rng([SALT, params["site_seed"]])
    _C, expect, _info = _run_defect(defect, P, T, params["count"], rng,
                                    params.get("frac"))
    params["expect"] = expect
    return params


_HEX64 = re.compile(r"^[0-9a-f]{64}$")


def _is_int(v) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def validate_params(params) -> None:
    """Refuse anything the sampler could not have written, by name."""
    if not isinstance(params, dict):
        raise ValueError("params: expected a dict, got %r"
                         % type(params).__name__)
    defect = params.get("defect")
    if defect not in DEFECTS:
        raise ValueError("defect: %r is not one of hole, flip, duplicate, "
                         "signed_zero, near_duplicate, t_junction" % (defect,))
    keys = ["defect", "parent_family", "parent_seed", "parent_index",
            "parent_id", "parent_sha256", "count", "site_seed", "expect"]
    if defect == "near_duplicate":
        keys.append("frac")
    for k in sorted(keys):
        if k not in params:
            raise ValueError("%s: missing key for %s" % (k, defect))
    for k in sorted(params):
        if k not in keys:
            raise ValueError("%s: unknown key for %s" % (k, defect))
    family = params["parent_family"]
    if family not in PARENT_FAMILIES:
        raise ValueError("parent_family: %r is not one of A, B, D, E, F"
                         % (family,))
    if not _is_int(params["parent_seed"]) or params["parent_seed"] < 0:
        raise ValueError("parent_seed: %r is not an int >= 0"
                         % (params["parent_seed"],))
    pi = params["parent_index"]
    if not _is_int(pi) or pi < PARENT_OFFSET:
        raise ValueError("parent_index: %s is inside the A-F corpus range (< %d)"
                         % (pi, PARENT_OFFSET))
    if params["parent_id"] != parent_module(params).geometry_id(
            params["parent_seed"], pi):
        raise ValueError("parent_id: %r != geometry_id(%s, %s)"
                         % (params["parent_id"], params["parent_seed"], pi))
    sha = params["parent_sha256"]
    if not isinstance(sha, str) or not _HEX64.match(sha):
        raise ValueError("parent_sha256: %r is not 64 lowercase hex" % (sha,))
    ss = params["site_seed"]
    if not _is_int(ss) or not 0 <= ss <= 2 ** 31 - 1:
        raise ValueError("site_seed: %r is not an int in [0, 2^31 - 1]" % (ss,))
    lo, hi = COUNT_RANGE[defect]
    c = params["count"]
    if not _is_int(c) or not lo <= c <= hi:
        raise ValueError("count: %s outside [%d, %d] for %s" % (c, lo, hi, defect))
    if defect == "near_duplicate":
        f = params["frac"]
        if (not isinstance(f, float) or not math.isfinite(f)
                or not FRAC_RANGE[0] <= f <= FRAC_RANGE[1]):
            raise ValueError("frac: %r outside [0.1, 0.4]" % (f,))
    exp = params["expect"]
    if not isinstance(exp, dict) or sorted(exp) != sorted(EXPECT_KEYS):
        raise ValueError("expect: keys %r != %r"
                         % (sorted(exp) if isinstance(exp, dict) else exp,
                            sorted(EXPECT_KEYS)))
    for k in EXPECT_KEYS:
        v = exp[k]
        if k == "per_hole":
            if not isinstance(v, list) or not all(_is_int(x) for x in v):
                raise ValueError("expect.per_hole: %r is not a list of ints"
                                 % (v,))
        elif not _is_int(v):
            raise ValueError("expect.%s: %r is not an int" % (k, v))


def build_bytes(params) -> bytes:
    """Regenerate the parent, replay the injection, and refuse to return
    anything whose re-read counts disagree with expect.  Never repairs,
    drops or flips anything to make a check pass."""
    validate_params(params)
    pgen = parent_module(params)
    pp = pgen.sample_params(params["parent_seed"], params["parent_index"])
    P, T = pgen.build(pp)
    sha = hashlib.sha256(stl_io.stl_bytes(P, T, pgen.SOLID)).hexdigest()
    if sha != params["parent_sha256"]:
        raise ValueError("parent_sha256: %s != the regenerated parent %s"
                         % (params["parent_sha256"], sha))
    rng = np.random.default_rng([SALT, params["site_seed"]])
    C, recomputed, _info = _run_defect(params["defect"], P, T, params["count"],
                                       rng, params.get("frac"))
    data = corner_bytes(C, pgen.SOLID)
    diffs = _oracle_diffs(reread_counts(data), params["expect"],
                          params["defect"], params["count"])
    if diffs:
        raise RuntimeError("oracle: " + "; ".join(diffs))
    if params["expect"] != recomputed:
        raise ValueError("expect: %s %r != recomputed %r"
                         % (params["defect"], params["expect"], recomputed))
    return data


def expected(params) -> dict:
    validate_params(params)
    return params["expect"]


def n_bodies(params) -> int:
    pgen = parent_module(params)
    if hasattr(pgen, "n_bodies"):
        pp = pgen.sample_params(params["parent_seed"], params["parent_index"])
        return int(pgen.n_bodies(pp))
    return 1


def l_ref(params) -> float:
    pgen = parent_module(params)
    pp = pgen.sample_params(params["parent_seed"], params["parent_index"])
    return float(pgen.l_ref(pp))


def flow(params) -> dict:
    """The parent's flow: family G inherits it (the defect does not change
    the reference quantities)."""
    pgen = parent_module(params)
    pp = pgen.sample_params(params["parent_seed"], params["parent_index"])
    return pgen.flow(pp)


def stratum(params) -> str:
    """A PRIORI classes only - what stl_repair's repertoire can close on
    paper; the gate REPORTS what it actually closed."""
    d, c = params["defect"], params["count"]
    if d in ("signed_zero", "near_duplicate"):
        return "easy"
    if d == "hole":
        return "medium" if c <= MAX_HOLE_EDGES else "hard"
    if d == "flip":
        return "medium" if c == 1 else "hard"
    if d == "t_junction":
        return "medium"
    return "hard"


def geometry_id(seed, index) -> str:
    return "G-%d-%03d" % (seed, index)


def make_row(seed, index):
    """One params dict, one defect STL, one autonomy-manifest/1 row (no
    split key - split.py adds it)."""
    seed, index = int(seed), int(index)
    params = sample_params(seed, index)
    data = build_bytes(params)
    row = {"schema": "autonomy-manifest/1",
           "geometry_id": geometry_id(seed, index), "family": FAMILY,
           "generator": GENERATOR, "params": params, "seed": seed,
           "stratum": stratum(params), "commensurate": False,
           "stl_sha256": hashlib.sha256(data).hexdigest(),
           "flow": flow(params)}
    return row, data


def write_row(row: dict, out_dir: str) -> str:
    """Rebuild from params alone, refuse a sha drift."""
    data = build_bytes(row["params"])
    path = os.path.join(out_dir, row["geometry_id"] + ".stl")
    with open(path, "wb") as f:
        f.write(data)
    with open(path, "rb") as f:
        got = hashlib.sha256(f.read()).hexdigest()
    if got != row["stl_sha256"]:
        raise ValueError("%s: rebuilt sha256 %s != row %s"
                         % (row["geometry_id"], got, row["stl_sha256"]))
    return path


def generate(seed, n, out_dir: str) -> list:
    """n defect STLs plus manifest_G.jsonl (the row machinery of
    meshkit.row_generate, with bytes from build_bytes)."""
    if os.path.isdir(out_dir):
        if os.listdir(out_dir):
            raise ValueError("out: %s is not empty" % out_dir)
    elif os.path.exists(out_dir):
        raise ValueError("out: %s exists and is not a directory" % out_dir)
    else:
        os.makedirs(out_dir)
    rows = []
    for index in range(n):
        row, data = make_row(seed, index)
        with open(os.path.join(out_dir, row["geometry_id"] + ".stl"), "wb") as f:
            f.write(data)
        rows.append(row)
    man = os.path.join(out_dir, "manifest_%s.jsonl" % FAMILY)
    with open(man, "wb") as f:
        for row in rows:
            f.write((json.dumps(row, sort_keys=True, separators=(",", ":"),
                                ensure_ascii=True) + "\n").encode("ascii"))
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="inject",
                                 description=__doc__.splitlines()[0])
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args(argv)
    if args.selftest:
        return _selftest()
    if args.seed is None or args.n is None or not args.out:
        ap.error("--seed, --n and --out are required (or --selftest)")
    rows = generate(args.seed, args.n, args.out)
    print("wrote %d STL(s) and manifest_%s.jsonl to %s"
          % (len(rows), FAMILY, args.out))
    return 0


# --- the selftest -----------------------------------------------------------


def _disk_props(T, region):
    """A grown region must be a disk: every edge used at most twice within
    it, ONE boundary loop of len(region) + 2 edges, no interior vertex."""
    use, verts, out = {}, set(), {}
    for t in region:
        for k in range(3):
            a, b = int(T[t][k]), int(T[t][(k + 1) % 3])
            verts.add(a)
            verts.add(b)
            e = (a, b) if a < b else (b, a)
            use[e] = use.get(e, 0) + 1
    assert all(c in (1, 2) for c in use.values()), "an edge used 3+ times"
    for t in region:
        for k in range(3):
            a, b = int(T[t][k]), int(T[t][(k + 1) % 3])
            if use[(a, b) if a < b else (b, a)] == 1:
                out.setdefault(a, []).append(b)
    assert all(len(v) == 1 for v in out.values()), "a branching boundary vertex"
    start = min(out)
    loop, seen, cur = [start], {start}, out[start][0]
    while cur != start:
        assert cur not in seen, "the boundary is not one simple loop"
        seen.add(cur)
        loop.append(cur)
        cur = out[cur][0]
    n_bnd = sum(1 for c in use.values() if c == 1)
    assert len(loop) == n_bnd == len(region) + 2, (len(loop), n_bnd, len(region))
    assert verts == seen, "the region has an interior vertex"
    return len(loop)


def _selftest() -> int:
    holder = {}

    def twelve():
        if "rows" not in holder:
            holder["rows"] = [make_row(7, i) for i in range(12)]
        return holder["rows"]

    def group_writer():
        notes = []
        for fam in PARENT_FAMILIES:
            pgen = importlib.import_module(PARENTS[fam])
            pp = pgen.sample_params(7, PARENT_OFFSET)
            P, T = pgen.build(pp)
            assert corner_bytes(P[T], pgen.SOLID) == \
                stl_io.stl_bytes(P, T, pgen.SOLID), \
                "%s: corner_bytes != stl_io.stl_bytes" % fam
            notes.append(fam)
        return "corner_bytes == stl_io.stl_bytes on one clean parent of %s" \
            % ", ".join(notes)

    def group_sampler():
        defect_n = {d: 0 for d in DEFECTS}
        fam_n = {f: 0 for f in PARENT_FAMILIES}
        pair_n, idx = {}, set()
        strata_n = {"easy": 0, "medium": 0, "hard": 0}
        for i in range(120):
            params = sample_params(7, i)
            d, f = params["defect"], params["parent_family"]
            defect_n[d] += 1
            fam_n[f] += 1
            pair_n[(d, f)] = pair_n.get((d, f), 0) + 1
            idx.add(params["parent_index"])
            lo, hi = COUNT_RANGE[d]
            assert lo <= params["count"] <= hi, (d, params["count"])
            strata_n[stratum(params)] += 1
        assert all(v == 20 for v in defect_n.values()), defect_n
        assert all(v == 24 for v in fam_n.values()), fam_n
        assert all(v == 4 for v in pair_n.values()), \
            [kv for kv in pair_n.items() if kv[1] != 4]
        assert len(idx) == 120, len(idx)
        print("  [inject] G strata at seed 7 n 120: easy %d, medium %d, "
              "hard %d" % (strata_n["easy"], strata_n["medium"],
                           strata_n["hard"]))
        return "6 defects x 20, 5 parent families x 24, 20 pairs x 4, 120 " \
            "distinct parent_index"

    def group_sites():
        seen = 0
        for row, _data in twelve():
            params = row["params"]
            d = params["defect"]
            pgen = parent_module(params)
            pp = pgen.sample_params(params["parent_seed"],
                                    params["parent_index"])
            P, T = pgen.build(pp)
            rng = np.random.default_rng([SALT, params["site_seed"]])
            _C, _e, info = _run_defect(d, P, T, params["count"], rng,
                                       params.get("frac"))
            if d in ("hole", "flip"):
                loop = _disk_props(T, info["region"])
                want = params["count"] if d == "flip" else params["count"] - 2
                assert loop == want + 2, (d, loop, want)
            elif d == "duplicate":
                sets = [set(int(v) for v in T[t]) for t in info["picks"]]
                union = set().union(*sets)
                assert len(union) == 3 * len(sets), \
                    "duplicate picks are not vertex-disjoint"
            elif d in ("signed_zero", "near_duplicate"):
                adj = adjacency(edge_map(T), len(P))
                tsets = [set(int(v) for v in T[s[0]])
                         for s in info["sites"]]
                for i, ts in enumerate(tsets):
                    assert not any(ts & u for u in tsets[:i]), \
                        "two sites' triangles share a vertex"
                for i, (ti, j, p, ax) in enumerate(info["sites"]):
                    earlier = {s[2] for s in info["sites"][:i]}
                    assert not (adj[p] & earlier), "sites are adjacent"
                    earlier_t = {s[0] for s in info["sites"][:i]}
                    assert ti not in earlier_t, "two sites in one triangle"
                    if d == "near_duplicate":
                        lo, hi = P.min(axis=0), P.max(axis=0)
                        tol = WELD_REL * float(np.linalg.norm(hi - lo))
                        room = lo[ax] < P[p, ax] and P[p, ax] + params["frac"] * tol < hi[ax]
                        assert room, "the moved axis is not strictly inside the bbox"
            else:
                diag = float(np.linalg.norm(P.max(axis=0) - P.min(axis=0)))
                tol = WELD_REL * diag
                for _t1, _u, _v, _w, M in info["sites"]:
                    near = float(np.linalg.norm(P - M, axis=1).min())
                    assert near > 10.0 * tol, "a midpoint is not 10 tol away"
            seen += 1
        # regression: near_duplicate rows whose sites shared a triangle
        # vertex while the vertex-disjoint guard was never updated - their
        # 4-edge loops met at a corner and the re-read walk refused them
        for sd, ix in ((2, 76), (4, 46)):
            row, _data = make_row(sd, ix)
            assert row["params"]["defect"] == "near_duplicate", row["geometry_id"]
        return "site rules hold on 2 rows per defect (%d rows); G-2-076 and " \
            "G-4-046 (near_duplicate, once vertex-sharing) build" % seen

    def group_oracle():
        n_checked = 0
        for row, data in twelve():
            params = row["params"]
            got = reread_counts(data)
            exp = params["expect"]
            for k in ("open_edges", "non_manifold_edges", "triangles_in",
                      "points_bit_exact", "negzero"):
                assert got[k] == exp[k], (row["geometry_id"], k, got[k], exp[k])
            d, c = params["defect"], params["count"]
            if d in ("signed_zero", "near_duplicate"):
                assert exp["per_hole"] == [] and got["per_hole"] == [4] * c, \
                    (row["geometry_id"], got["per_hole"])
            else:
                assert got["per_hole"] == exp["per_hole"], \
                    (row["geometry_id"], got["per_hole"], exp["per_hole"])
            n_checked += 1
        row, data = twelve()[0]
        lines = data.split(b"\n")
        doctored = b"\n".join(lines[:-9] + lines[-2:])
        got2 = reread_counts(doctored)
        diffs = _oracle_diffs(got2, row["params"]["expect"],
                              row["params"]["defect"], row["params"]["count"])
        assert diffs, "the doctored re-read passed the oracle comparison"
        return "reread == expect on %d rows; a dropped facet fails the " \
            "comparison (%s)" % (n_checked, diffs[0].split(":")[0])

    def group_determinism():
        rows = [make_row(7, i) for i in (0, 3)]
        for row, data in rows:
            assert build_bytes(row["params"]) == data, \
                "%s: rebuild differs" % row["geometry_id"]
            assert build_bytes(json.loads(json.dumps(row["params"]))) == data, \
                "%s: json round-trip differs" % row["geometry_id"]
        meshkit._poison_global_rng()
        for row, data in rows:
            assert build_bytes(row["params"]) == data, \
                "%s: poisoned rebuild differs" % row["geometry_id"]
        d = tempfile.mkdtemp(prefix="inj_det_")
        try:
            generate(7, 3, d)
            with open(os.path.join(d, geometry_id(7, 0) + ".stl"), "rb") as f:
                assert f.read() == rows[0][1], "alone != in batch"
        finally:
            shutil.rmtree(d, ignore_errors=True)
        bad = []
        for name in ("inject.py", "split.py"):
            path = os.path.join(_HERE, name)
            if not os.path.isfile(path):
                continue
            with open(path, encoding="utf-8") as f:
                src = f.read()
            for mt in re.finditer(r"np\.random\.\w+|import\x20random", src):
                if mt.group(0) != "np.random.default_rng":
                    bad.append("%s: %s" % (name, mt.group(0)))
        assert not bad, bad
        return "2 rows rebuild bit-identically, alone, in batch, json and " \
            "poisoned; default_rng only"

    def group_rows():
        n = 0
        for row, _data in twelve():
            gid = row["geometry_id"]
            assert "split" not in row, "%s carries split" % gid
            assert row["family"] == FAMILY and row["commensurate"] is False, gid
            for s in ("tuning", "test"):
                errs = schema.errors(dict(row, split=s), "ManifestRow")
                assert not errs, (gid, errs[:1])
            pgen = parent_module(row["params"])
            pp = pgen.sample_params(row["params"]["parent_seed"],
                                    row["params"]["parent_index"])
            assert row["flow"] == pgen.flow(pp), "%s: flow != parent flow" % gid
            assert row["params"]["parent_index"] >= PARENT_OFFSET, gid
            n += 1
        return "%d rows are ManifestRows with split added, G, not " \
            "commensurate, the parent's flow" % n

    def group_refusals():
        hole, flip = sample_params(7, 0), sample_params(7, 1)
        dup, near = sample_params(7, 2), sample_params(7, 4)
        sz = sample_params(7, 3)
        k = 0

        def refuses(params, frag, fn=None):
            nonlocal k
            try:
                (fn or validate_params)(params)
            except (ValueError, RuntimeError) as e:
                assert frag in str(e), "wrong refusal: %s" % e
                k += 1
                return
            raise AssertionError("no refusal naming %r" % frag)

        refuses(dict(hole, defect="zzz"), "defect: 'zzz' is not one of")
        refuses(dict(hole, count=2), "count: 2 outside [3, 64] for hole")
        refuses(dict(hole, count=65), "count: 65 outside [3, 64] for hole")
        refuses(dict(flip, count=0), "count: 0 outside [1, 32] for flip")
        refuses(dict(dup, count=9), "count: 9 outside [1, 8] for duplicate")
        refuses(dict(near, frac=0.5), "frac: 0.5 outside [0.1, 0.4]")
        refuses(dict(hole, parent_index=5), "parent_index: 5 is inside")
        refuses(dict(hole, parent_family="C"), "parent_family: 'C'")
        refuses(dict(hole, parent_id="G-7-999"), "parent_id:")
        refuses(dict(hole, count=True), "count:")
        bad = dict(hole)
        bad["zzz"] = 1
        refuses(bad, "zzz: unknown key for hole")
        short = dict(hole)
        del short["count"]
        refuses(short, "count: missing key for hole")
        edited = json.loads(json.dumps(sz))
        edited["expect"]["per_hole"] = [1]
        refuses(edited, "expect: signed_zero", build_bytes)
        wrong_parent = json.loads(json.dumps(hole))
        wrong_parent["parent_sha256"] = "0" * 64
        refuses(wrong_parent, "parent_sha256:", build_bytes)
        return "%d refusals by name" % k

    groups = [("[ok] writer", group_writer),
              ("[ok] sampler", group_sampler),
              ("[ok] sites", group_sites),
              ("[ok] oracle", group_oracle),
              ("[ok] determinism", group_determinism),
              ("[ok] rows", group_rows),
              ("[ok] refusals", group_refusals)]
    for name, fn in groups:
        try:
            note = fn()
        except AssertionError as e:
            print("SELFTEST FAIL: %s: %s" % (name, e))
            return 1
        print("%s: %s" % (name, note))
    print("SELFTEST PASS")
    return 0


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    sys.exit(main())
