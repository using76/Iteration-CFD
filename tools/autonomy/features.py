#!/usr/bin/env python3
# meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
# Source-available, not Open Source. See LICENSE at the repository root.
# No GPL-licensed source was consulted.
"""features.py - the autonomy-fingerprint/1 of one closed STL (docs/15 section C).

Reads ASCII (multi-solid) or binary STL through tools/geom/stl_repair.py's
parser, welds bit-exactly (no tolerance, no repair), refuses an open,
non-manifold, mis-oriented, inward or degenerate surface by name, and
computes: per-patch area in first-appearance order, total area and volume,
bbox, sharp-edge length at the feature angle (the automesher's own dihedral
test), curvature-radius p5/p50/p95 from a per-vertex osculating-paraboloid
fit, inner thickness and outer gap by shrinking tangent balls, the
axis-aligned planar area fraction, and an origin-free lattice
commensurability. The result validates against the fixed AM-1 Fingerprint
schema. docs/15 section G AM-4 is the analytic gate; run it:

    python tools/autonomy/features.py STL --id ID [--feature-angle DEG] [--diag]
    python tools/autonomy/features.py --corpus A|B --seed S --n N
    python tools/autonomy/features.py --selftest
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import math
import os
import sys
import tempfile
import time
import shutil

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
_CORPUS = os.path.join(HERE, "corpus")
_GEOM = os.path.join(os.path.dirname(os.path.dirname(HERE)), "tools", "geom")
for _p in (_GEOM, _CORPUS, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import schema  # noqa: E402
import stl_io  # noqa: E402
import stl_repair  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402

SCHEMA_ID = "autonomy-fingerprint/1"
FEATURE_ANGLE_DEG = 30.0   # refinement.feature_angle_deg default, rust/src/automesher/mod.rs d_feature_angle
COORD_TOL = 1e-9           # x bbox diagonal: axis-planar test and plane clustering
FLAT_KAPPA = 1e-6          # kappa_max * diagonal below this is flat (no curvature radius)
BALL_CAP = 0.125           # initial tangent-ball radius = BALL_CAP * diagonal
BALL_EPS = 1e-9            # a point is inside the ball when |q - c| < r * (1 - BALL_EPS)
BALL_MAX_ITER = 100
FACING_DOT = -0.5          # a limiting point faces the sample when n_p . n_q < FACING_DOT
LATTICE_LEVELS = 6         # the octree cap (knobs.json /refinement/max_level max 6)
LATTICE_TOL = 1e-6         # |d/s - round(d/s)| <= LATTICE_TOL


# --- reading and welding (docs/15 section G AM-4, C2) -----------------------

def _f(v):
    """A Python float with no -0.0 (the schema and json.dumps must see one)."""
    return float(v) + 0.0


def _n_or_none(v):
    return None if v is None else _f(v)


def read_surface(path: str) -> dict:
    """Read one STL file and weld it bit-exactly; refuse what does not read."""
    try:
        with open(path, "rb") as f:
            data = f.read()
    except OSError:
        raise ValueError("read: %s: no such file" % path)
    stem = os.path.splitext(os.path.basename(path))[0]
    err = io.StringIO()
    try:
        with contextlib.redirect_stderr(err):
            tris, _fmt = stl_repair.read_stl(path, stem)
    except SystemExit:
        raise ValueError("read: %s" % err.getvalue().strip())
    names, idx = [], {}
    tp = np.empty(len(tris), dtype=np.int64)
    for i, (nm, _tri) in enumerate(tris):
        if nm not in idx:
            idx[nm] = len(names)
            names.append(nm)
        tp[i] = idx[nm]
    X = stl_io.canonical(np.array([v for (_n, tri) in tris for v in tri],
                                  dtype=np.float64))
    P, inv = np.unique(X, axis=0, return_inverse=True)
    T = np.asarray(inv, dtype=np.int64).reshape(-1, 3)
    return {"P": np.ascontiguousarray(P, dtype=np.float64),
            "T": np.ascontiguousarray(T, dtype=np.int64),
            "tri_patch": tp, "patch_names": names, "data": data}


def topology(P, T) -> dict:
    """Areas, unit normals and the edge table; refuse a bad surface by name.

    Order of refusal: too_small, degenerate, non_manifold, open, orientation.
    """
    P = np.asarray(P, dtype=np.float64)
    T = np.asarray(T, dtype=np.int64)
    nP, nT = len(P), len(T)
    if nT < 4:
        raise ValueError("surface/too_small: %d triangle(s), need >= 4" % nT)
    rep = ((T[:, 0] == T[:, 1]) | (T[:, 1] == T[:, 2]) | (T[:, 0] == T[:, 2]))
    cr = np.cross(P[T[:, 1]] - P[T[:, 0]], P[T[:, 2]] - P[T[:, 0]])
    area2 = np.linalg.norm(cr, axis=1)
    bad = rep | (area2 == 0.0)
    if bad.any():
        i = int(np.nonzero(bad)[0][0])
        raise ValueError("surface/degenerate: triangle %d has zero area" % i)
    dir_e = T[:, [0, 1, 1, 2, 2, 0]].reshape(-1, 2)
    lo = np.minimum(dir_e[:, 0], dir_e[:, 1])
    hi = np.maximum(dir_e[:, 0], dir_e[:, 1])
    key = lo * np.int64(nP) + hi
    _uniq, inv, counts = np.unique(key, return_inverse=True, return_counts=True)
    nm = int(np.count_nonzero(counts >= 3))
    if nm:
        raise ValueError("surface/non_manifold: %d edge(s) with 3+ triangles" % nm)
    op = int(np.count_nonzero(counts == 1))
    if op:
        raise ValueError("surface/open: %d edge(s) with one triangle" % op)
    order = np.argsort(key, kind="stable")
    eid = inv[order]
    first = np.empty(len(eid), dtype=bool)
    first[0] = True
    first[1:] = eid[1:] != eid[:-1]
    pos = np.nonzero(first)[0]
    t1 = (order[pos] // 3).astype(np.int64)
    t2 = (order[pos + 1] // 3).astype(np.int64)
    d0 = dir_e[order[pos], 0] < dir_e[order[pos], 1]
    d1 = dir_e[order[pos + 1], 0] < dir_e[order[pos + 1], 1]
    sd = int(np.count_nonzero(d0 == d1))
    if sd:
        raise ValueError("surface/orientation: %d edge(s) used twice in the same direction" % sd)
    nE = len(pos)
    E = np.empty((nE, 2), dtype=np.int64)
    E[:, 0] = _uniq // nP
    E[:, 1] = _uniq % nP
    A = 0.5 * area2
    n = cr / area2[:, None]
    return {"A": A, "n": stl_io.canonical(n), "E": E, "t1": t1, "t2": t2}


# --- sharp edges (SPEC-LIT (92.34), the automesher's own test) --------------

def sharp_mask(topo, feature_angle_deg) -> np.ndarray:
    """(nE,) bool: degrees(arccos(n_t1 . n_t2)) > feature_angle_deg."""
    dot = np.einsum("ij,ij->i", topo["n"][topo["t1"]], topo["n"][topo["t2"]])
    phi = np.degrees(np.arccos(np.clip(dot, -1.0, 1.0)))
    return phi > float(feature_angle_deg)


def weighted_quantile(v, w, q) -> float:
    """The q-quantile of v under weights w (stable argsort, cumulative walk)."""
    v = np.asarray(v, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    o = np.argsort(v, kind="stable")
    cw = np.cumsum(w[o])
    i = int(np.searchsorted(cw, q * cw[-1], side="left"))
    return float(v[o][min(i, len(v) - 1)])


# --- curvature radius (do Carmo 1976, section 3-3) --------------------------

def _vertex_fields(P, T, topo):
    """(VA, vn): per-vertex area and area-weighted unit normal."""
    nP = len(P)
    flat = T.reshape(-1)
    VA = np.zeros(nP, dtype=np.float64)
    np.add.at(VA, flat, np.repeat(topo["A"] / 3.0, 3))
    vn = np.zeros((nP, 3), dtype=np.float64)
    np.add.at(vn, flat, np.repeat(topo["n"] * topo["A"][:, None], 3, axis=0))
    ln = np.linalg.norm(vn, axis=1)
    return VA, vn / ln[:, None]


def _bbox_diag(P) -> float:
    return float(np.linalg.norm(P.max(axis=0) - P.min(axis=0)))


def curvature_radii(P, T, topo, sharp):
    """(p5, p50, p95, n_curved) of 1/kappa_max over fitted, non-flat vertices.

    Fitted: >= 3 ring neighbours, no sharp edge, cond(ATA) < 1e10. Curved:
    fitted and kappa_max * diag > FLAT_KAPPA. Weights are the vertex areas.
    """
    nP = len(P)
    diag = _bbox_diag(P)
    VA, vn = _vertex_fields(P, T, topo)
    ref = np.where((np.abs(vn[:, 0]) < 0.9)[:, None],
                   np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]))
    u = np.cross(vn, ref)
    u /= np.linalg.norm(u, axis=1)[:, None]
    v = np.cross(vn, u)
    I = np.concatenate([topo["E"][:, 0], topo["E"][:, 1]])
    J = np.concatenate([topo["E"][:, 1], topo["E"][:, 0]])
    d = P[J] - P[I]
    x = np.einsum("ij,ij->i", d, u[I])
    y = np.einsum("ij,ij->i", d, v[I])
    z = np.einsum("ij,ij->i", d, vn[I])
    w = np.stack([x * x, x * y, y * y], axis=1)
    ATA = np.zeros((nP, 3, 3), dtype=np.float64)
    ATb = np.zeros((nP, 3), dtype=np.float64)
    cnt = np.zeros(nP, dtype=np.int64)
    np.add.at(ATA, I, w[:, :, None] * w[:, None, :])
    np.add.at(ATb, I, w * z[:, None])
    np.add.at(cnt, I, 1)
    sharp_v = np.zeros(nP, dtype=bool)
    if sharp.any():
        sharp_v[topo["E"][sharp].reshape(-1)] = True
    cand = (cnt >= 3) & ~sharp_v
    p5 = p50 = p95 = None
    n_curved = 0
    if cand.any():
        idx_c = np.nonzero(cand)[0]
        sv = np.linalg.svd(ATA[idx_c], compute_uv=False)
        ok = sv[:, 0] / sv[:, -1] < 1e10
        idx = idx_c[ok]
        if len(idx):
            sol = np.linalg.solve(ATA[idx], ATb[idx][..., None])[..., 0]
            a, b, c = sol[:, 0], sol[:, 1], sol[:, 2]
            kmax = np.abs(a + c) + np.sqrt((a - c) ** 2 + b ** 2)
            curved = kmax * diag > FLAT_KAPPA
            if curved.any():
                r = 1.0 / kmax[curved]
                wgt = VA[idx[curved]]
                p5 = weighted_quantile(r, wgt, 0.05)
                p50 = weighted_quantile(r, wgt, 0.5)
                p95 = weighted_quantile(r, wgt, 0.95)
                n_curved = int(len(r))
    return _n_or_none(p5), _n_or_none(p50), _n_or_none(p95), n_curved


# --- inner thickness and outer gap (Ma, Bae & Choi 2012) --------------------

def _ball_inputs(P, T, topo, sharp):
    """(S, N, Q, QN): samples and the full point set with their normals."""
    cent = (P[T[:, 0]] + P[T[:, 1]] + P[T[:, 2]]) / 3.0
    VA, vn = _vertex_fields(P, T, topo)
    sharp_v = np.zeros(len(P), dtype=bool)
    if sharp.any():
        sharp_v[topo["E"][sharp].reshape(-1)] = True
    vtx = np.nonzero(~sharp_v)[0]
    S = np.concatenate([cent, P[vtx]])
    N = np.concatenate([topo["n"], vn[vtx]])
    Q = np.concatenate([cent, P])
    QN = np.concatenate([topo["n"], vn])
    return S, N, Q, QN


def _run_balls(S, N, Q, QN, tree, sign, diag) -> dict:
    """Shrink one tangent ball per sample until no point of Q is inside it."""
    nS = len(S)
    m = sign * N
    r = np.full(nS, BALL_CAP * diag, dtype=np.float64)
    lim = np.full(nS, -1, dtype=np.int64)
    active = np.ones(nS, dtype=bool)
    iterations = 0
    for _ in range(BALL_MAX_ITER):
        idx = np.nonzero(active)[0]
        if len(idx) == 0:
            break
        iterations += 1
        c = S[idx] + r[idx][:, None] * m[idx]
        dd, j = tree.query(c)
        inside = dd < r[idx] * (1.0 - BALL_EPS)
        ins = idx[inside]
        jb = j[inside]
        qv = Q[jb] - S[ins]
        r[ins] = np.einsum("ij,ij->i", qv, qv) / (2.0 * np.einsum("ij,ij->i", qv, m[ins]))
        lim[ins] = jb
        active[idx[~inside]] = False
    unconverged = int(np.count_nonzero(active))
    limited = (lim >= 0) & ~active
    facing = np.zeros(nS, dtype=bool)
    if limited.any():
        li = np.nonzero(limited)[0]
        facing[li] = np.einsum("ij,ij->i", N[li], QN[lim[li]]) < FACING_DOT
    return {"diam": 2.0 * r, "limited": limited, "facing": facing,
            "iterations": iterations, "unconverged": unconverged}


def tangent_balls(P, T, topo, sharp, sign: float) -> dict:
    """One shrinking tangent ball per sample, sign -1 inward / +1 outward."""
    S, N, Q, QN = _ball_inputs(P, T, topo, sharp)
    tree = cKDTree(Q)
    return _run_balls(S, N, Q, QN, tree, float(sign), _bbox_diag(P))


# --- planar fraction and lattice commensurability (C6) ----------------------

def planar_and_lattice(P, T):
    """(planar_frac, commensurate, lattice_base_size_m), origin-free."""
    P = np.asarray(P, dtype=np.float64)
    T = np.asarray(T, dtype=np.int64)
    cr = np.cross(P[T[:, 1]] - P[T[:, 0]], P[T[:, 2]] - P[T[:, 0]])
    A = 0.5 * np.linalg.norm(cr, axis=1)
    lo, hi = P.min(axis=0), P.max(axis=0)
    tol = COORD_TOL * _bbox_diag(P)
    planar = np.zeros(len(T), dtype=bool)
    ds = []
    for a in range(3):
        ca = P[T[:, [0, 1, 2]], a]
        pl = (ca.max(1) - ca.min(1)) <= tol
        planar |= pl
        if pl.any():
            vals = np.sort(ca[pl, 0])
            clusters = [float(vals[0])]
            for v in vals[1:]:
                if float(v) > clusters[-1] + tol:
                    clusters.append(float(v))
            ds.extend(clusters[k] - clusters[0] for k in range(1, len(clusters)))
    frac = float(np.sum(A[planar]) / np.sum(A))
    if frac < 1.0:
        return frac, False, None
    if not ds:
        return frac, False, None
    dmin = min(ds)
    floor = float(np.max(hi - lo)) / 2 ** LATTICE_LEVELS
    m = 1
    darr = np.asarray(ds, dtype=np.float64)
    while dmin / m >= floor * (1.0 - 1e-12):
        s = dmin / m
        q = darr / s
        if float(np.max(np.abs(q - np.round(q)))) <= LATTICE_TOL:
            return frac, True, float(s)
        m += 1
    return frac, False, None


# --- compute and fingerprint (C7) -------------------------------------------

def compute(surf: dict, feature_angle_deg: float = FEATURE_ANGLE_DEG):
    """The Fingerprint's geometric fields plus the diagnostics dict."""
    fa = float(feature_angle_deg)
    if not math.isfinite(fa) or not (0.0 < fa < 180.0):
        raise ValueError("feature_angle_deg: %r outside (0, 180)" % (feature_angle_deg,))
    P, T = surf["P"], surf["T"]
    topo = topology(P, T)
    vol = stl_io.signed_volume(P, T)
    if vol <= 0.0:
        raise ValueError("surface/volume: signed volume %r <= 0 (wound inward)" % vol)
    sharp = sharp_mask(topo, fa)
    sel = P[topo["E"][sharp, 1]] - P[topo["E"][sharp, 0]]
    sel_len = float(np.linalg.norm(sel, axis=1).sum()) if len(sel) else 0.0
    p5, p50, p95, n_curved = curvature_radii(P, T, topo, sharp)
    frac, comm, base = planar_and_lattice(P, T)
    S, N, Q, QN = _ball_inputs(P, T, topo, sharp)
    tree = cKDTree(Q)
    diag_d = _bbox_diag(P)
    bi = _run_balls(S, N, Q, QN, tree, -1.0, diag_d)
    bo = _run_balls(S, N, Q, QN, tree, 1.0, diag_d)
    inner = float(bi["diam"][bi["facing"]].max()) if bi["facing"].any() else None
    outer = float(bo["diam"][bo["facing"]].min()) if bo["facing"].any() else None
    lo, hi = P.min(axis=0), P.max(axis=0)
    tri_patch = surf["tri_patch"]
    patches = [{"name": nm, "area_m2": _f(topo["A"][tri_patch == k].sum())}
               for k, nm in enumerate(surf["patch_names"])]
    fields = {
        "patches": patches,
        "area_m2": _f(topo["A"].sum()),
        "volume_m3": _f(vol),
        "bbox": [_f(lo[0]), _f(hi[0]), _f(lo[1]), _f(hi[1]), _f(lo[2]), _f(hi[2])],
        "sharp_edge_length_m": _f(sel_len),
        "curvature_radius_p5_m": _n_or_none(p5),
        "curvature_radius_p50_m": _n_or_none(p50),
        "curvature_radius_p95_m": _n_or_none(p95),
        "inner_thickness_m": _n_or_none(inner),
        "outer_gap_m": _n_or_none(outer),
        "planar_frac": _f(frac),
        "commensurate": bool(comm),
        "lattice_base_size_m": _n_or_none(base),
    }
    diag = {
        "n_vertices": int(len(P)), "n_samples": int(len(S)), "n_curved": int(n_curved),
        "n_sharp_edges": int(np.count_nonzero(sharp)),
        "n_inner_limited": int(np.count_nonzero(bi["limited"])),
        "n_inner_facing": int(np.count_nonzero(bi["facing"])),
        "n_outer_limited": int(np.count_nonzero(bo["limited"])),
        "n_outer_facing": int(np.count_nonzero(bo["facing"])),
        "n_unconverged": int(bi["unconverged"] + bo["unconverged"]),
        "ball_iterations_max": int(max(bi["iterations"], bo["iterations"])),
        "seconds": 0.0,
    }
    return fields, diag


def fingerprint(path: str, geometry_id: str,
                feature_angle_deg: float = FEATURE_ANGLE_DEG):
    """(fp, diag): the autonomy-fingerprint/1 of one STL file, schema-checked."""
    t0 = time.perf_counter()
    surf = read_surface(path)
    fields, diag = compute(surf, feature_angle_deg)
    diag["seconds"] = time.perf_counter() - t0
    fp = {"schema": SCHEMA_ID, "geometry_id": geometry_id,
          "stl_sha256": hashlib.sha256(surf["data"]).hexdigest(),
          "n_triangles": int(len(surf["T"])),
          "feature_angle_deg": _f(feature_angle_deg)}
    fp.update(fields)
    schema.validate(fp, "Fingerprint")
    return fp, diag


# --- CLI (C8) ---------------------------------------------------------------

def _corpus(fam: str, seed: int, n: int) -> int:
    import gen_wing
    import gen_lathe
    gen = gen_wing if fam == "A" else gen_lathe
    tmp = tempfile.mkdtemp(prefix="features-corpus-")
    valid = sh = cu = th = gp = cm = 0
    smax, sid = 0.0, "-"
    try:
        for i in range(n):
            row, data = gen.make_row(seed, i)
            gid = row["geometry_id"]
            pth = os.path.join(tmp, gid + ".stl")
            with open(pth, "wb") as f:
                f.write(data)
            try:
                fp, dg = fingerprint(pth, gid)
            except ValueError:
                continue
            valid += 1
            s = dg["seconds"] / (fp["n_triangles"] / 1e4)
            if s > smax:
                smax, sid = s, gid
            sh += 1 if fp["sharp_edge_length_m"] > 0.0 else 0
            cu += 1 if fp["curvature_radius_p50_m"] is not None else 0
            th += 1 if fp["inner_thickness_m"] is not None else 0
            gp += 1 if fp["outer_gap_m"] is not None else 0
            cm += 1 if fp["commensurate"] else 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("features corpus %s seed %d n %d: valid %d/%d, max s/10k %.3f (%s), "
          "sharp>0 %d, curvature %d, thickness %d, gap %d, commensurate %d"
          % (fam, seed, n, valid, n, smax, sid, sh, cu, th, gp, cm))
    return 0 if (valid == n and smax < 2.0) else 1


# --- the selftest (C8) ------------------------------------------------------

class _Fail(Exception):
    """A group failed: the message already carries the group name."""


def _icosphere(R, sub, center):
    """A subdivided icosahedron projected to radius R; outward by volume."""
    t = (1.0 + math.sqrt(5.0)) / 2.0
    pts = [(-1.0, t, 0.0), (1.0, t, 0.0), (-1.0, -t, 0.0), (1.0, -t, 0.0),
           (0.0, -1.0, t), (0.0, 1.0, t), (0.0, -1.0, -t), (0.0, 1.0, -t),
           (t, 0.0, -1.0), (t, 0.0, 1.0), (-t, 0.0, -1.0), (-t, 0.0, 1.0)]
    pts = [np.asarray(p, dtype=np.float64) / np.linalg.norm(p) for p in pts]
    F = np.array([[0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
                  [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
                  [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
                  [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1]],
                 dtype=np.int64)
    for _ in range(sub):
        mid = {}
        nf = []

        def g(i, j):
            k = (i, j) if i < j else (j, i)
            if k not in mid:
                p = pts[i] + pts[j]
                pts.append(p / np.linalg.norm(p))
                mid[k] = len(pts) - 1
            return mid[k]

        for a, b, c in F:
            ab, bc, ca = g(a, b), g(b, c), g(c, a)
            nf.extend([[a, ab, ca], [ab, b, bc], [ca, bc, c], [ab, bc, ca]])
        F = np.array(nf, dtype=np.int64)
    P = np.asarray(pts, dtype=np.float64) * float(R) \
        + np.asarray(center, dtype=np.float64)
    T = F
    if stl_io.signed_volume(P, T) < 0.0:
        T = np.ascontiguousarray(T[:, [0, 2, 1]])
    return P, T


def _uvsphere(R, nth, nph, center):
    """A UV sphere: pole (0,0,R), nph-1 rings, pole (0,0,-R); outward by volume."""
    rows = [[0.0, 0.0, float(R)]]
    for k in range(1, nph):
        ph = math.pi * k / nph
        for i in range(nth):
            th = 2.0 * math.pi * i / nth
            rows.append([R * math.sin(ph) * math.cos(th),
                         R * math.sin(ph) * math.sin(th),
                         R * math.cos(ph)])
    rows.append([0.0, 0.0, -float(R)])
    P = np.asarray(rows, dtype=np.float64)
    south = len(P) - 1
    F = []
    for i in range(nth):
        F.append([0, 1 + (i + 1) % nth, 1 + i])
    for k in range(nph - 2):
        b = 1 + k * nth
        c = b + nth
        for i in range(nth):
            j = (i + 1) % nth
            F.append([b + i, c + j, c + i])
            F.append([b + i, b + j, c + j])
    bl = 1 + (nph - 2) * nth
    for i in range(nth):
        F.append([south, bl + i, bl + (i + 1) % nth])
    T = np.asarray(F, dtype=np.int64)
    if stl_io.signed_volume(P, T) < 0.0:
        T = np.ascontiguousarray(T[:, [0, 2, 1]])
    return P + np.asarray(center, dtype=np.float64), T


def _close(a, b, rtol) -> bool:
    if a is None or b is None:
        return a is None and b is None
    return abs(a - b) <= rtol * max(abs(a), abs(b), 1.0)


_LENK = ("sharp_edge_length_m", "curvature_radius_p5_m", "curvature_radius_p50_m",
         "curvature_radius_p95_m", "inner_thickness_m", "outer_gap_m",
         "lattice_base_size_m")


def _g1(tmp, st):
    """Sphere curvature: p50 within 2 % of R; sharp 0, planar 0, balls None."""
    devs = []
    for tag, R, P, T in (("icosphere sub4 R0.5", 0.5, *_icosphere(0.5, 4, (0.0, 0.0, 0.0))),
                         ("icosphere sub4 R2.0", 2.0, *_icosphere(2.0, 4, (0.0, 0.0, 0.0))),
                         ("uv 64x32 R0.5", 0.5, *_uvsphere(0.5, 64, 32, (0.0, 0.0, 0.0)))):
        pth = os.path.join(tmp, "g1.stl")
        stl_io.write_stl(pth, P, T, "sph")
        fp, dg = fingerprint(pth, "g1-sphere")
        st["fps"].append(("sphere", fp, dg))
        p50 = fp["curvature_radius_p50_m"]
        dev = p50 / R - 1.0
        assert abs(dev) <= 0.02, "p50 %r vs R %r" % (p50, R)
        assert fp["curvature_radius_p5_m"] <= p50 <= fp["curvature_radius_p95_m"]
        assert fp["sharp_edge_length_m"] == 0.0, fp["sharp_edge_length_m"]
        assert fp["planar_frac"] == 0.0
        assert fp["commensurate"] is False and fp["lattice_base_size_m"] is None
        assert fp["inner_thickness_m"] is None and fp["outer_gap_m"] is None
        devs.append("%+.3f%%" % (100.0 * dev))
    return ("[ok] sphere curvature: p50 dev %s (<= 2%%), sharp 0.0, planar 0.0, "
            "commensurate False, inner/outer None" % ", ".join(devs))


def _g2(tmp, st):
    """Cube sharp edges: within 0.5 % of 12L; curvature all None."""
    from sensitivity import box_mesh
    devs = []
    for tag, lo, hi, nd in (("CUBE 1-2 nd8", (1, 1, 1), (2, 2, 2), 8),
                            ("unit nd1", (0, 0, 0), (1, 1, 1), 1),
                            ("box0.3 nd4", (0, 0, 0), (0.3, 0.3, 0.3), 4)):
        P, T = box_mesh(lo, hi, nd)
        pth = os.path.join(tmp, "g2.stl")
        stl_io.write_stl(pth, P, T, "box")
        fp, dg = fingerprint(pth, "g2-box")
        st["fps"].append(("box", fp, dg))
        L = float(hi[0]) - float(lo[0])
        dev = abs(fp["sharp_edge_length_m"] / (12.0 * L) - 1.0)
        assert dev <= 0.005, "sharp %r vs 12L %r" % (fp["sharp_edge_length_m"], 12.0 * L)
        assert (fp["curvature_radius_p5_m"] is None
                and fp["curvature_radius_p50_m"] is None
                and fp["curvature_radius_p95_m"] is None)
        assert dg["n_sharp_edges"] == 12 * nd, dg["n_sharp_edges"]
        assert fp["planar_frac"] == 1.0
        devs.append("%s %.1e" % (tag.split()[0], dev))
        if tag.startswith("box0.3"):
            st["mesh"]["box03"] = (P, T)
        if tag.startswith("CUBE"):
            st["mesh"]["cube"] = (P, T)
    return "[ok] cube sharp edges: dev %s (<= 0.5%%), curvature None, n_sharp 12*nd" % ", ".join(devs)


def _g3(tmp, st):
    """Two-sphere gap within 2 %; written as one multi-solid file."""
    cases = [("ico", 0.1), ("ico", 0.03), ("ico", 0.01), ("uv", 0.1)]
    devs = []
    for i, (kind, g) in enumerate(cases):
        if kind == "ico":
            pa = _icosphere(0.5, 4, (0.0, 0.0, 0.0))
            pb = _icosphere(0.5, 4, (0.0, 1.0 + g, 0.0))
        else:
            pa = _uvsphere(0.5, 64, 32, (0.0, 0.0, 0.0))
            pb = _uvsphere(0.5, 64, 32, (0.0, 0.0, 1.0 + g))
        pth = os.path.join(tmp, "g3-%s%s.stl" % (kind, g))
        with open(pth, "wb") as f:
            f.write(stl_io.stl_bytes(pa[0], pa[1], "a"))
            f.write(stl_io.stl_bytes(pb[0], pb[1], "b"))
        fp, dg = fingerprint(pth, "g3-pair")
        st["fps"].append(("sphere", fp, dg))
        gap = fp["outer_gap_m"]
        assert abs(gap / g - 1.0) <= 0.02, "gap %r vs g %r" % (gap, g)
        assert fp["inner_thickness_m"] is None
        devs.append("%s%s %+.3f%%" % (kind, g, 100.0 * (gap / g - 1.0)))
        if kind == "ico" and g == 0.1:
            st["mesh"]["pair"] = (pa, pb)
            st["paths"]["pair"] = pth
            st["fp_pair"] = fp
    return "[ok] two-sphere gap: dev %s (<= 2%%), inner None" % ", ".join(devs)


def _g4(tmp, st):
    """NACA0012 thickness within 1 % of 0.12*c_root, both wings."""
    import gen_wing
    row, _data = gen_wing.make_row(1, 0)
    base = dict(row["params"], m=0, p=0, tt=12, code="0012", taper=1.0,
                sweep_deg=0.0, twist_deg=0.0)
    devs = []
    for tag, over in (("untapered", {}), ("taper0.5 sweep20", dict(taper=0.5, sweep_deg=20.0))):
        prm = dict(base, **over)
        P, T = gen_wing.build(prm)
        pth = os.path.join(tmp, "g4.stl")
        stl_io.write_stl(pth, P, T, "wing")
        fp, dg = fingerprint(pth, "g4-wing")
        st["fps"].append(("wing", fp, dg))
        th = fp["inner_thickness_m"]
        want = 0.12 * prm["c_root"]
        assert th is not None and abs(th / want - 1.0) <= 0.01, "%r vs %r" % (th, want)
        assert fp["curvature_radius_p5_m"] is not None
        devs.append("%s %+.3f%%" % (tag.split()[0], 100.0 * (th / want - 1.0)))
        if not over:
            st["mesh"]["wing"] = (P, T)
            st["wing_params"] = prm
    return "[ok] NACA0012 thickness: dev %s (<= 1%%), p5 not None" % ", ".join(devs)


def _g5(tmp, st):
    """Planar fraction: 1.0 boxes, 0.0 spheres; the wing's equals the
    independently recomputed axis-planar share (gen_wing's blunt-TE base
    strip at x = c_root is axis-planar too, so the planar set is the two tip
    caps plus that strip, not the caps alone)."""
    n_box = n_sph = 0
    for kind, fp, _dg in st["fps"]:
        if kind == "box":
            assert fp["planar_frac"] == 1.0, fp["planar_frac"]
            n_box += 1
        elif kind == "sphere":
            assert fp["planar_frac"] == 0.0, fp["planar_frac"]
            n_sph += 1
    P, T = st["mesh"]["wing"]
    A = 0.5 * np.linalg.norm(np.cross(P[T[:, 1]] - P[T[:, 0]],
                                      P[T[:, 2]] - P[T[:, 0]]), axis=1)
    tol = COORD_TOL * _bbox_diag(P)
    planar = np.zeros(len(T), dtype=bool)
    for a in range(3):
        ca = P[T[:, [0, 1, 2]], a]
        planar |= (ca.max(1) - ca.min(1)) <= tol
    share = float(A[planar].sum() / A.sum())
    ys = st["wing_params"]["span"] / 2.0
    Y = P[T][:, :, 1]
    tip = np.all((Y == ys) | (Y == -ys), axis=1)
    assert planar[tip].all(), "a tip-cap triangle is not axis-planar"
    strip = planar & ~tip
    fp = [f for k, f, _ in st["fps"] if k == "wing"][0]
    assert abs(fp["planar_frac"] - share) <= 1e-9, (fp["planar_frac"], share)
    return ("[ok] planar fraction: %d boxes 1.0, %d spheres 0.0, wing %.9f = "
            "planar share %.9f (tip %.9f + TE strip %.9f) to 1e-9"
            % (n_box, n_sph, fp["planar_frac"], share,
               float(A[tip].sum() / A.sum()), float(A[strip].sum() / A.sum())))


def _two_cubes(off):
    from sensitivity import box_mesh
    P1, T1 = box_mesh((0.0, 0.0, 0.0), (1.0, 1.0, 1.0), 4)
    P2, T2 = box_mesh((off, 0.0, 0.0), (1.0 + off, 1.0, 1.0), 4)
    return np.concatenate([P1, P2]), np.concatenate([T1, T2 + len(P1)])


def _g6(tmp, st):
    """Commensurability: the seven C6 cases with exactly the stated verdicts."""
    from sensitivity import box_mesh
    icoP, icoT = _icosphere(0.5, 4, (0.0, 0.0, 0.0))
    cases = [("BOX-c", box_mesh((0.0, -0.25, -0.25), (1.0, 0.25, 0.25), 8), True, 0.5),
             ("BOX-n", box_mesh((0.0137, -0.2113, -0.1709), (0.9137, 0.2291, 0.1893), 16),
              False, None),
             ("CUBE", box_mesh((1.0, 1.0, 1.0), (2.0, 2.0, 2.0), 8), True, 1.0),
             ("cube-offset", box_mesh((0.0137, 0.0137, 0.0137), (1.0137, 1.0137, 1.0137), 4),
              True, 1.0),
             ("two-cubes-1.5", _two_cubes(1.5), True, 0.5),
             ("two-cubes-1.3137", _two_cubes(1.3137), False, None),
             ("icosphere", (icoP, icoT), False, None)]
    got = []
    for tag, (P, T), want_c, want_s in cases:
        pth = os.path.join(tmp, "g6.stl")
        stl_io.write_stl(pth, P, T, "shape")
        fp, dg = fingerprint(pth, "g6-" + tag)
        st["fps"].append(("comm", fp, dg))
        assert fp["commensurate"] is want_c, (tag, fp["commensurate"])
        if want_s is None:
            assert fp["lattice_base_size_m"] is None, tag
            got.append("%s False/-" % tag)
        else:
            assert abs(fp["lattice_base_size_m"] - want_s) <= 1e-12, \
                (tag, fp["lattice_base_size_m"])
            got.append("%s True/%g" % (tag, want_s))
        if tag == "BOX-c":
            st["mesh"]["boxc"] = (P, T)
    return "[ok] commensurability: %s" % ", ".join(got)


def _g7(tmp, st):
    """Patches in first-appearance order summing to area_m2; box totals exact."""
    fp = st["fp_pair"]
    assert [p["name"] for p in fp["patches"]] == ["a", "b"], fp["patches"]
    surf = read_surface(st["paths"]["pair"])
    topo = topology(surf["P"], surf["T"])
    tot = 0.0
    for kx, patch in enumerate(fp["patches"]):
        own = float(topo["A"][surf["tri_patch"] == kx].sum())
        assert abs(patch["area_m2"] / own - 1.0) <= 1e-12, (patch, own)
        tot += own
    assert abs(fp["area_m2"] / tot - 1.0) <= 1e-12, (fp["area_m2"], tot)
    P, T = st["mesh"]["box03"]
    fpb = [f for k, f, _ in st["fps"] if k == "box"][-1]
    assert abs(fpb["volume_m3"] / 0.027 - 1.0) <= 1e-12, fpb["volume_m3"]
    assert fpb["bbox"] == [0.0, 0.3, 0.0, 0.3, 0.0, 0.3], fpb["bbox"]
    assert abs(fpb["area_m2"] / 0.54 - 1.0) <= 1e-12, fpb["area_m2"]
    return ("[ok] patches and totals: [a, b] in order, each = its own sum, "
            "sum = area_m2 (1e-12 rel); box0.3 volume 0.027, bbox exact, area 0.54")


_QUANT = ("curvature_radius_p5_m", "curvature_radius_p50_m", "curvature_radius_p95_m")


def _cmp_translate(base, new, shift):
    # The curvature quantiles are order statistics over a dense set: a
    # translation perturbs the fit's last bits and the crossing vertex jumps
    # by a neighbour gap (measured: p5 2.2e-8, p50 4.5e-6, p95 1.5e-5 rel on
    # the wing), so they get 1e-4 while every other field keeps 1e-6.
    for k in _LENK + ("area_m2", "volume_m3", "planar_frac"):
        assert _close(base[k], new[k], 1e-4 if k in _QUANT else 1e-6), (k, base[k], new[k])
    for pb, pn in zip(base["patches"], new["patches"]):
        assert pb["name"] == pn["name"] and _close(pb["area_m2"], pn["area_m2"], 1e-6)
    for i in range(6):
        assert _close(base["bbox"][i] + shift[i // 2], new["bbox"][i], 1e-6), i
    assert base["commensurate"] == new["commensurate"]
    assert base["n_triangles"] == new["n_triangles"]


def _cmp_scale(base, new):
    for k in _LENK:
        wb = None if base[k] is None else 2.0 * base[k]
        assert _close(wb, new[k], 1e-4 if k in _QUANT else 1e-6), (k, base[k], new[k])
    for pb, pn in zip(base["patches"], new["patches"]):
        assert pb["name"] == pn["name"] and _close(4.0 * pb["area_m2"], pn["area_m2"], 1e-6)
    assert _close(4.0 * base["area_m2"], new["area_m2"], 1e-6)
    assert _close(8.0 * base["volume_m3"], new["volume_m3"], 1e-6)
    for i in range(6):
        assert _close(2.0 * base["bbox"][i], new["bbox"][i], 1e-6), i
    # planar_frac is an area ratio of the read-back surface: x2 of the
    # %.9e-rounded coordinates is not the %.9e of x2, so compare at 1e-9.
    assert _close(base["planar_frac"], new["planar_frac"], 1e-9), \
        (base["planar_frac"], new["planar_frac"])
    assert base["commensurate"] == new["commensurate"]


def _cmp_perm(base, new):
    for k in _LENK + ("area_m2", "volume_m3", "planar_frac"):
        assert _close(base[k], new[k], 1e-9), (k, base[k], new[k])
    assert _close(base["curvature_radius_p5_m"], new["curvature_radius_p5_m"], 1e-9)

    assert base["bbox"] == new["bbox"]
    assert base["patches"] == new["patches"] or all(
        pb["name"] == pn["name"] and _close(pb["area_m2"], pn["area_m2"], 1e-9)
        for pb, pn in zip(base["patches"], new["patches"]))
    assert base["commensurate"] == new["commensurate"]
    assert base["lattice_base_size_m"] == new["lattice_base_size_m"]
    assert base["n_triangles"] == new["n_triangles"]


def _g8(tmp, st):
    """Invariance: translate, x2 scale, permute, and byte-identical reruns."""
    shift = (0.37, -1.1, 2.3)
    n = 0
    for tag in ("ico", "wing", "boxc"):
        if tag == "ico":
            P, T = _icosphere(0.5, 4, (0.0, 0.0, 0.0))
        else:
            P, T = st["mesh"][tag]
        pth = os.path.join(tmp, "g8-base.stl")
        stl_io.write_stl(pth, P, T, "shape")
        base, _ = fingerprint(pth, "g8-base")
        Pt, Tt = P + np.asarray(shift, dtype=np.float64), T
        stl_io.write_stl(os.path.join(tmp, "g8-t.stl"), Pt, Tt, "shape")
        _cmp_translate(base, fingerprint(os.path.join(tmp, "g8-t.stl"), "g8-t")[0], shift)
        stl_io.write_stl(os.path.join(tmp, "g8-s.stl"), 2.0 * P, T, "shape")
        _cmp_scale(base, fingerprint(os.path.join(tmp, "g8-s.stl"), "g8-s")[0])
        perm = np.random.default_rng(3).permutation(len(T))
        stl_io.write_stl(os.path.join(tmp, "g8-p.stl"), P, np.ascontiguousarray(T[perm]), "shape")
        _cmp_perm(base, fingerprint(os.path.join(tmp, "g8-p.stl"), "g8-p")[0])
        f1 = json.dumps(fingerprint(pth, "g8-base")[0], sort_keys=True)
        f2 = json.dumps(fingerprint(pth, "g8-base")[0], sort_keys=True)
        assert f1.encode("utf-8") == f2.encode("utf-8")
        n += 1
    return ("[ok] invariance: translate/x2/permute/double-run equal on ico, "
            "wing, BOX-c (%d shapes, 1e-6/1e-6/1e-9 rel, curvature quantiles 1e-4, rerun byte-identical)" % n)


def _g9(tmp, st):
    """Refusals: each ValueError carries its leading name."""
    from sensitivity import box_mesh
    icoP, icoT = _icosphere(0.5, 4, (0.0, 0.0, 0.0))

    def wr(name, P, T, solid="shape"):
        pth = os.path.join(tmp, name)
        stl_io.write_stl(pth, P, T, solid)
        return pth

    def refu(pth, name, fa=FEATURE_ANGLE_DEG, gid="g9"):
        try:
            fingerprint(pth, gid, fa)
        except ValueError as e:
            assert str(e).startswith(name), "%s: %s" % (name, e)
            return 1
        raise AssertionError("%s: no ValueError" % name)

    k = 0
    k += refu(wr("g9-open.stl", icoP, np.delete(icoT, 5, axis=0)), "surface/open")
    flip = icoT.copy()
    flip[7] = flip[7, [0, 2, 1]]
    k += refu(wr("g9-ori.stl", icoP, flip), "surface/orientation")
    k += refu(wr("g9-in.stl", icoP, icoT[:, ::-1]), "surface/volume")
    k += refu(wr("g9-nm.stl", icoP, np.concatenate([icoT, icoT[:1]])),
              "surface/non_manifold")
    deg = (b"solid deg\nfacet normal 0 0 1\nouter loop\nvertex 0 0 0\n"
           b"vertex 1 0 0\nvertex 0 0 0\nendloop\nendfacet\n"
           b"facet normal 0 0 1\nouter loop\nvertex 0 0 0\nvertex 1 0 0\n"
           b"vertex 1 1 0\nendloop\nendfacet\n"
           b"facet normal 0 0 1\nouter loop\nvertex 1 1 0\nvertex 1 0 0\n"
           b"vertex 0 0 0\nendloop\nendfacet\n"
           b"facet normal 0 1 0\nouter loop\nvertex 0 0 0\nvertex 1 1 0\n"
           b"vertex 0 1 0\nendloop\nendfacet\nendsolid deg\n")
    pth = os.path.join(tmp, "g9-deg.stl")
    with open(pth, "wb") as f:
        f.write(deg)
    k += refu(pth, "surface/degenerate")
    P3, T3 = box_mesh((0.0, 0.0, 0.0), (1.0, 1.0, 1.0), 1)
    k += refu(wr("g9-small.stl", P3, T3[:3]), "surface/too_small")
    good = wr("g9-good.stl", icoP, icoT)
    k += refu(good, "feature_angle_deg", fa=0.0)
    k += refu(good, "feature_angle_deg", fa=180.0)
    bad = os.path.join(tmp, "g9-text.stl")
    with open(bad, "w", encoding="ascii") as f:
        f.write("hello world\n")
    k += refu(bad, "read")
    k += refu(os.path.join(tmp, "g9-absent.stl"), "read")
    try:
        fingerprint(good, "bad id!")
    except ValueError as e:
        assert str(e).startswith("Fingerprint:") and "geometry_id" in str(e), str(e)
        k += 1
    else:
        raise AssertionError("geometry_id: no SchemaError")
    return "[ok] refusals: %d by name" % k


def _g10(tmp, st):
    """Every fingerprint so far plus 12 corpus files: errors == []."""
    import gen_wing
    import gen_lathe
    n = 0
    for _kind, fp, _dg in st["fps"]:
        assert schema.errors(fp, "Fingerprint") == [], schema.errors(fp, "Fingerprint")[:1]
        n += 1
    for gen in (gen_wing, gen_lathe):
        for i in range(6):
            row, data = gen.make_row(1, i)
            pth = os.path.join(tmp, "g10-%s.stl" % row["geometry_id"])
            with open(pth, "wb") as f:
                f.write(data)
            fp, dg = fingerprint(pth, row["geometry_id"])
            assert schema.errors(fp, "Fingerprint") == []
            st["fps"].append(("corpus", fp, dg))
            n += 1
    return "[ok] schema: %d fingerprints, errors == [] on every one" % n


def _g11(tmp, st):
    """Speed: seconds per 10k triangles under 2.0 on the 20480-triangle
    icosphere and the 12 corpus files, max printed. The per-10k rate of a
    12-triangle box is its fixed overhead times 833, so the small shapes
    of the other groups are not timed here."""
    P, T = _icosphere(0.5, 5, (0.0, 0.0, 0.0))
    pth = os.path.join(tmp, "g11.stl")
    stl_io.write_stl(pth, P, T, "sph")
    fp, dg = fingerprint(pth, "g11-ico5")
    st["fps"].append(("sphere", fp, dg))
    timed = [(k, f, d) for k, f, d in st["fps"]
             if k == "corpus" or f["geometry_id"] == "g11-ico5"]
    assert len(timed) == 13, len(timed)
    worst, who = 0.0, "-"
    for _kind, f, d in timed:
        s = d["seconds"] / (f["n_triangles"] / 1e4)
        if s > worst:
            worst, who = s, f["geometry_id"]
        assert s < 2.0, (f["geometry_id"], s)
    assert fp["n_triangles"] == 20480, fp["n_triangles"]
    return "[ok] speed: max s/10k %.3f at %s over %d timed fingerprints (all < 2.0)" \
        % (worst, who, len(timed))


def _selftest() -> int:
    tmp = tempfile.mkdtemp(prefix="features-selftest-")
    st = {"fps": [], "mesh": {}, "paths": {}}
    groups = (("[1] sphere curvature", _g1), ("[2] cube sharp edges", _g2),
              ("[3] two-sphere gap", _g3), ("[4] NACA0012 thickness", _g4),
              ("[5] planar fraction", _g5), ("[6] commensurability", _g6),
              ("[7] patches and totals", _g7), ("[8] invariance", _g8),
              ("[9] refusals", _g9), ("[10] schema", _g10), ("[11] speed", _g11))
    oks = []
    try:
        for name, fn in groups:
            try:
                oks.append(fn(tmp, st))
            except _Fail:
                raise
            except (AssertionError, ValueError, OSError, KeyError,
                    IndexError, TypeError) as e:
                raise _Fail("%s: %r" % (name, e))
    except _Fail as e:
        print("SELFTEST FAIL: %s" % (e,))
        return 1
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    for line in oks:
        print(line)
    print("SELFTEST PASS")
    return 0


def main(argv) -> int:
    """CLI: one fingerprint, a corpus family, or the selftest. 0/1/2."""
    ap = argparse.ArgumentParser(prog="features", add_help=True,
                                 description="the autonomy-fingerprint/1 of a closed STL")
    ap.add_argument("stl", nargs="?", help="the STL file to fingerprint")
    ap.add_argument("--id", help="geometry_id for the fingerprint")
    ap.add_argument("--feature-angle", type=float, default=FEATURE_ANGLE_DEG,
                    help="the dihedral threshold in degrees (default 30)")
    ap.add_argument("--diag", action="store_true", help="also print the diagnostics")
    ap.add_argument("--corpus", choices=["A", "B"], help="fingerprint a corpus family")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return _selftest()
    if a.corpus:
        return _corpus(a.corpus, a.seed, a.n)
    if not a.stl or not a.id:
        ap.print_usage(sys.stderr)
        return 2
    try:
        fp, dg = fingerprint(a.stl, a.id, a.feature_angle)
    except ValueError as e:
        sys.stderr.write("features: %s\n" % (e,))
        return 1
    print(json.dumps(fp, sort_keys=True))
    if a.diag:
        print("diag: %s" % json.dumps(dg, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    sys.exit(main(sys.argv[1:]))
