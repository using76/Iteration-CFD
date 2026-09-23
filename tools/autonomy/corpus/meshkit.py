#!/usr/bin/env python3
# meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
# Source-available, not Open Source. See LICENSE at the repository root.
# No GPL-licensed source was consulted.
"""meshkit - the shared builders and row machinery of corpus families D, E, F.

Everything docs/15 §E's AM-6 generators share, numpy and stdlib only plus
stl_io/schema: twelve uniform draws per (salt, seed, index) as the ONLY
randomness; axis-aligned grid boxes, lofts and prisms of counter-clockwise
rings with fan caps (odd axis permutations emitted with swapped corners),
UV spheres with exact poles, 64-gon circles with exact quarter points, and
a union of bodies that refuses overlapping ones.  Every quad is split along
the diagonal through its lexicographically smallest corner, so two opposite
parallel faces with the same grid are mirror-triangulated - that is what
makes features.py's thickness and gap samples face each other.  Surfaces
are wound outward by construction and refused, never repaired, when
stl_io.check_closed does not pass.  lattice_spacing restates the
fingerprint's lattice rule (docs/15 §C) on analytic plane coordinates.

    python tools/autonomy/corpus/meshkit.py --selftest
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
import re
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (_HERE, os.path.dirname(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import stl_io  # noqa: E402
import schema  # noqa: E402


def uniforms(salt: int, seed: int, index: int) -> np.ndarray:
    """Twelve draws in [0, 1): the only randomness any generator may use."""
    return np.random.default_rng([int(salt), int(seed), int(index)]).uniform(
        0.0, 1.0, size=12)


def lin(U: np.ndarray, k: int, lo: float, hi: float) -> float:
    """Draw k mapped linearly onto [lo, hi]."""
    return lo + (hi - lo) * float(U[k])


def ik(U: np.ndarray, k: int, lo: int, hi: int) -> int:
    """Draw k mapped onto an integer in [lo, hi]."""
    return lo + min(hi - lo, int(math.floor(float(U[k]) * (hi - lo + 1))))


def fl(v: float, d: int) -> float:
    """Round to d decimals and kill any -0.0."""
    return round(float(v), d) + 0.0


def divs(length: float, step: float) -> int:
    """The number of pieces of a straight edge of `length` at `step`."""
    return max(1, int(math.ceil(float(length) / float(step) - 1e-9)))


def axis_coords(lo: float, hi: float, n: int) -> np.ndarray:
    """n cells from lo to hi; the end coordinates are exact."""
    n = int(n)
    if n < 1:
        raise ValueError("axis_coords: n %d < 1" % n)
    lo, hi = float(lo), float(hi)
    c = lo + (hi - lo) * np.arange(n + 1) / n
    c[0] = lo
    c[-1] = hi
    return c


def _split_quad(pts: list, q: tuple) -> tuple:
    """The diagonal rule: split along the diagonal through the corner whose
    world point is lexicographically smallest as the (x, y, z) tuple, so two
    opposite parallel faces with the same grid pick the same world diagonal
    and features.py's thickness and gap samples face each other."""
    m = min(range(4), key=lambda i: tuple(pts[q[i]]))
    if m in (0, 2):
        return ((q[0], q[1], q[2]), (q[0], q[2], q[3]))
    return ((q[0], q[1], q[3]), (q[1], q[2], q[3]))


def grid_box(xs, ys, zs):
    """An axis-aligned box on the given per-axis coordinate arrays.

    Every corner coordinate comes from one per-axis array, so a point shared
    by two faces is bit-identical; surface nodes are keyed by their integer
    (i, j, k) index into (xs, ys, zs) and created at first appearance in the
    face table's order.  Each quad is split by the diagonal rule.  Refused,
    never repaired, when stl_io.check_closed does not pass.
    """
    xs = np.asarray(xs, dtype=np.float64)
    ys = np.asarray(ys, dtype=np.float64)
    zs = np.asarray(zs, dtype=np.float64)
    ax = (xs, ys, zs)
    for name, c in (("x", xs), ("y", ys), ("z", zs)):
        if len(c) < 2 or not bool(np.all(np.diff(c) > 0.0)):
            raise ValueError("grid_box: %s coordinates not strictly increasing"
                             % name)
    faces = ((0, 0, 2, 1), (0, 1, 1, 2), (1, 0, 0, 2),
             (1, 1, 2, 0), (2, 0, 1, 0), (2, 1, 0, 1))
    pts, index, tris = [], {}, []

    def node(axis: int, side: int, ua: int, va: int, a: int, b: int) -> int:
        # the (i, j, k) index triple with the last position written as -1,
        # so every face names a shared node by the same triple
        idx = [0, 0, 0]
        idx[axis] = 0 if side == 0 else -1
        idx[ua] = a if a < len(ax[ua]) - 1 else -1
        idx[va] = b if b < len(ax[va]) - 1 else -1
        key = tuple(idx)
        if key not in index:
            p = [0.0, 0.0, 0.0]
            p[axis] = float(ax[axis][0] if side == 0 else ax[axis][-1])
            p[ua] = float(ax[ua][a])
            p[va] = float(ax[va][b])
            index[key] = len(pts)
            pts.append(p)
        return index[key]

    for axis, side, ua, va in faces:
        for a in range(len(ax[ua]) - 1):
            for b in range(len(ax[va]) - 1):
                q = (node(axis, side, ua, va, a, b),
                     node(axis, side, ua, va, a + 1, b),
                     node(axis, side, ua, va, a + 1, b + 1),
                     node(axis, side, ua, va, a, b + 1))
                tris.extend(_split_quad(pts, q))
    P = stl_io.canonical(pts)
    T = np.asarray(tris, dtype=np.int64)
    stl_io.check_closed(P, T)
    return P, T


def line(a: tuple, b: tuple, n: int) -> list:
    """n pieces from a towards b, excluding b (the next piece carries it)."""
    n = int(n)
    if n < 1:
        raise ValueError("line: n %d < 1" % n)
    a = (float(a[0]), float(a[1]))
    b = (float(b[0]), float(b[1]))
    return [a] + [(a[0] + (b[0] - a[0]) * k / n,
                   a[1] + (b[1] - a[1]) * k / n) for k in range(1, n)]


def arc(start: tuple, c: tuple, r: float, deg0: float, deg1: float,
        n: int) -> list:
    """n pieces of a circular arc from deg0 to deg1 about c, excluding the
    end; `start` is passed exactly by the caller, never from cos/sin."""
    n = int(n)
    if n < 1:
        raise ValueError("arc: n %d < 1" % n)
    c = (float(c[0]), float(c[1]))
    r = float(r)
    out = [(float(start[0]), float(start[1]))]
    for k in range(1, n):
        t = math.radians(deg0 + (deg1 - deg0) * k / n)
        out.append((c[0] + r * math.cos(t), c[1] + r * math.sin(t)))
    return out


def circle(cu: float, cv: float, r: float, n: int) -> list:
    """A counter-clockwise n-gon; the quarter points are exact."""
    n = int(n)
    if n < 4 or n % 4 != 0:
        raise ValueError("circle: n %d is not a multiple of 4 (>= 4)" % n)
    cu, cv, r = float(cu), float(cv), float(r)
    exact = {0: (1.0, 0.0), n // 4: (0.0, 1.0),
             n // 2: (-1.0, 0.0), 3 * n // 4: (0.0, -1.0)}
    ring = []
    for j in range(n):
        d = exact.get(j)
        if d is None:
            th = 2.0 * math.pi * j / n
            d = (math.cos(th), math.sin(th))
        ring.append((cu + r * d[0], cv + r * d[1]))
    return ring


def loft(rings: list, ws: list, axes: tuple, centres: tuple):
    """Loft equal-length counter-clockwise rings through increasing stations.

    A point (u, v) at station w lands on p[axes[0]] = u, p[axes[1]] = v,
    p[axes[2]] = w; centres are the (u, v) of the two flat cap fans.  The
    winding is derived for a CCW ring in a right-handed (u, v, w): the side
    quad (a, b, c, d) is outward, the bottom cap (cb, i1, i) and the top cap
    (ct, L+i, L+i1) are outward.  An odd permutation of (0, 1, 2) is a
    reflection, so every triangle is emitted as (t0, t2, t1) - construction,
    not repair.  Each cap triangle's 2D signed area must be positive, which
    refuses a centre that does not see an edge.
    """
    rings = [[(float(u), float(v)) for (u, v) in ring] for ring in rings]
    ws = [float(w) for w in ws]
    if len(rings) < 2 or len(ws) != len(rings):
        raise ValueError("loft: %d ring(s) and %d station(s)" %
                         (len(rings), len(ws)))
    if any(ws[k + 1] <= ws[k] for k in range(len(ws) - 1)):
        raise ValueError("ws: not strictly increasing")
    if tuple(sorted(int(a) for a in axes)) != (0, 1, 2):
        raise ValueError("axes: %r is not a permutation of (0, 1, 2)"
                         % (tuple(axes),))
    m = len(rings[0])
    if m < 3:
        raise ValueError("loft: ring 0 has %d point(s), need >= 3" % m)
    for k, ring in enumerate(rings):
        if len(ring) != m:
            raise ValueError("loft: ring %d has %d point(s) != %d"
                             % (k, len(ring), m))
        for i in range(m):
            if ring[i] == ring[(i + 1) % m]:
                raise ValueError("ring %d: repeated point" % k)
        area2 = 0.0
        for i in range(m):
            u0, v0 = ring[i]
            u1, v1 = ring[(i + 1) % m]
            area2 += u0 * v1 - u1 * v0
        if not area2 > 0.0:
            raise ValueError("ring %d: not counter-clockwise" % k)
    n_st = len(rings)
    cb = (float(centres[0][0]), float(centres[0][1]))
    ct = (float(centres[1][0]), float(centres[1][1]))
    pts = []

    def put(u: float, v: float, w: float) -> None:
        p = [0.0, 0.0, 0.0]
        p[axes[0]] = u
        p[axes[1]] = v
        p[axes[2]] = w
        pts.append(p)

    for k, ring in enumerate(rings):
        for (u, v) in ring:
            put(u, v, ws[k])
    put(cb[0], cb[1], ws[0])
    put(ct[0], ct[1], ws[-1])
    n_stm = n_st * m
    tris = []
    odd = tuple(axes) in ((0, 2, 1), (1, 0, 2), (2, 1, 0))

    def emit(t: tuple) -> None:
        tris.append((t[0], t[2], t[1]) if odd else t)

    for k in range(n_st - 1):
        for i in range(m):
            i1 = (i + 1) % m
            for t in _split_quad(pts, (k * m + i, k * m + i1,
                                       (k + 1) * m + i1, (k + 1) * m + i)):
                emit(t)
    L = (n_st - 1) * m
    for i in range(m):
        i1 = (i + 1) % m
        (u0, v0), (u1, v1) = rings[0][i], rings[0][i1]
        sa = 0.5 * ((u0 - cb[0]) * (v1 - cb[1]) - (u1 - cb[0]) * (v0 - cb[1]))
        if not sa > 0.0:
            raise ValueError("cap bottom: centre does not see edge %d" % i)
        emit((n_stm, i1, i))
        (u0, v0), (u1, v1) = rings[-1][i], rings[-1][i1]
        sa = 0.5 * ((u0 - ct[0]) * (v1 - ct[1]) - (u1 - ct[0]) * (v0 - ct[1]))
        if not sa > 0.0:
            raise ValueError("cap top: centre does not see edge %d" % i)
        emit((n_stm + 1, L + i, L + i1))
    P = stl_io.canonical(pts)
    T = np.asarray(tris, dtype=np.int64)
    stl_io.check_closed(P, T)
    return P, T


def prism(ring: list, ws: list, axes: tuple, centre: tuple):
    """A straight extrusion of one ring: the same ring at every station."""
    return loft([list(ring) for _ in ws], ws, axes, (centre, centre))


def uv_sphere(R: float, centre: tuple, axis: int, n_theta: int,
              n_phi: int):
    """A UV sphere about `centre`, poles on `axis`, both poles ONE vertex.

    Local frame (a1, a2, a3) = (1,2,0)/(2,0,1)/(0,1,2) for axis 0/1/2
    (right-handed); the two pole coordinates are exact, not from sin/cos.
    Winding outward by construction and refused, never repaired, when
    stl_io.check_closed does not pass.
    """
    frames = {0: (1, 2, 0), 1: (2, 0, 1), 2: (0, 1, 2)}
    if axis not in frames:
        raise ValueError("uv_sphere: axis %r not in (0, 1, 2)" % (axis,))
    a1, a2, a3 = frames[axis]
    n_theta, n_phi = int(n_theta), int(n_phi)
    if n_theta < 8 or n_phi < 4:
        raise ValueError("uv_sphere: n_theta %d, n_phi %d too small"
                         % (n_theta, n_phi))
    R = float(R)
    c = [float(v) for v in centre]
    pts = []
    p = list(c)
    p[a3] += R
    pts.append(p)
    for k in range(1, n_phi):
        phi = math.pi * k / n_phi
        for i in range(n_theta):
            th = 2.0 * math.pi * i / n_theta
            p = list(c)
            p[a1] = c[a1] + R * math.sin(phi) * math.cos(th)
            p[a2] = c[a2] + R * math.sin(phi) * math.sin(th)
            p[a3] = c[a3] + R * math.cos(phi)
            pts.append(p)
    p = list(c)
    p[a3] -= R
    pts.append(p)
    S = len(pts) - 1
    bl = 1 + (n_phi - 2) * n_theta
    tris = []
    for i in range(n_theta):
        i1 = (i + 1) % n_theta
        tris.append((0, 1 + i, 1 + i1))
    for k in range(n_phi - 2):
        b = 1 + k * n_theta
        cc = b + n_theta
        for i in range(n_theta):
            i1 = (i + 1) % n_theta
            tris.extend(_split_quad(pts, (b + i, cc + i, cc + i1, b + i1)))
    for i in range(n_theta):
        i1 = (i + 1) % n_theta
        tris.append((S, bl + i1, bl + i))
    P = stl_io.canonical(pts)
    T = np.asarray(tris, dtype=np.int64)
    stl_io.check_closed(P, T)
    return P, T


def combine(bodies: list):
    """A disjoint union: concatenation with index offsets, nothing welded.

    Every pair of bodies must be strictly separated along at least one axis
    (touching is refused too); the union must then be closed by edge report,
    have Euler characteristic 2 per body and positive volume - refused by
    name, never repaired, otherwise.
    """
    if not bodies:
        raise ValueError("combine: no bodies")
    boxes = [(np.asarray(P, dtype=np.float64).min(axis=0),
              np.asarray(P, dtype=np.float64).max(axis=0)) for P, _T in bodies]
    for i in range(len(bodies)):
        for j in range(i + 1, len(bodies)):
            (alo, ahi), (blo, bhi) = boxes[i], boxes[j]
            if not any(float(ahi[a]) < float(blo[a]) or
                       float(bhi[a]) < float(alo[a]) for a in range(3)):
                raise ValueError("combine: bodies %d and %d overlap" % (i, j))
    off = 0
    ts = []
    ps = []
    for P, T in bodies:
        ps.append(np.asarray(P, dtype=np.float64))
        ts.append(np.asarray(T, dtype=np.int64) + off)
        off += len(P)
    P = stl_io.canonical(np.concatenate(ps, axis=0))
    T = np.concatenate(ts, axis=0)
    er = stl_io.edge_report(T, len(P))
    if er["open"] or er["non_manifold"] or er["same_direction"]:
        raise RuntimeError("combine: edge_report %s" % er)
    chi = stl_io.euler_characteristic(P, T)
    if chi != 2 * len(bodies):
        raise RuntimeError("combine: euler characteristic %d != %d"
                           % (chi, 2 * len(bodies)))
    vol = stl_io.signed_volume(P, T)
    if not vol > 0.0:
        raise RuntimeError("combine: signed volume %.6g <= 0" % vol)
    return P, T


def lattice_spacing(planes, levels: int = 6, tol: float = 1e-6):
    """docs/15 §C's commensurability on analytic plane coordinates: the
    largest cubic spacing, at least the largest axis extent over 2**levels,
    on which every axis-aligned plane coordinate lies.  None when no such
    spacing exists (features.planar_and_lattice restates this on the mesh).
    """
    ds, ext = [], 0.0
    for coords in planes:
        c = sorted(set(float(v) for v in coords))
        ext = max(ext, c[-1] - c[0])
        ds.extend(v - c[0] for v in c[1:])
    if not ds:
        return None
    dmin = min(ds)
    floor = ext / 2 ** levels
    m = 1
    while dmin / m >= floor * (1.0 - 1e-12):
        s = dmin / m
        if max(abs(d / s - round(d / s)) for d in ds) <= tol:
            return s
        m += 1
    return None


def _poison_global_rng() -> None:
    importlib.import_module("numpy").random.seed(999)
    importlib.import_module("random").seed(999)


def row_make(gen, seed: int, index: int):
    """gen_wing.make_row, shared: one params dict, one STL, one row."""
    params = gen.sample_params(seed, index)
    P, T = gen.build(params)
    data = stl_io.stl_bytes(P, T, gen.SOLID)
    comm = (gen.expected_features(params)["commensurate"]
            if hasattr(gen, "expected_features") else False)
    row = {"schema": "autonomy-manifest/1",
           "geometry_id": gen.geometry_id(seed, index), "family": gen.FAMILY,
           "generator": gen.GENERATOR, "params": params, "seed": int(seed),
           "stratum": gen.stratum(params), "commensurate": bool(comm),
           "stl_sha256": hashlib.sha256(data).hexdigest(),
           "flow": gen.flow(params)}
    return row, data


def row_write(gen, row: dict, out_dir: str) -> str:
    """gen_wing.write_row, shared: rebuild from params, refuse a sha drift."""
    P, T = gen.build(row["params"])
    path = os.path.join(out_dir, row["geometry_id"] + ".stl")
    with open(path, "wb") as f:
        f.write(stl_io.stl_bytes(P, T, gen.SOLID))
    with open(path, "rb") as f:
        got = hashlib.sha256(f.read()).hexdigest()
    if got != row["stl_sha256"]:
        raise ValueError("%s: rebuilt sha256 %s != row %s"
                         % (row["geometry_id"], got, row["stl_sha256"]))
    return path


def row_generate(gen, seed: int, n: int, out_dir: str) -> list:
    """gen_wing.generate, shared: n STLs plus manifest_<FAMILY>.jsonl."""
    if os.path.isdir(out_dir):
        if os.listdir(out_dir):
            raise ValueError("out: %s is not empty" % out_dir)
    elif os.path.exists(out_dir):
        raise ValueError("out: %s exists and is not a directory" % out_dir)
    else:
        os.makedirs(out_dir)
    rows = []
    for index in range(n):
        row, data = row_make(gen, seed, index)
        with open(os.path.join(out_dir, row["geometry_id"] + ".stl"), "wb") as f:
            f.write(data)
        rows.append(row)
    man = os.path.join(out_dir, "manifest_%s.jsonl" % gen.FAMILY)
    with open(man, "wb") as f:
        for row in rows:
            f.write((json.dumps(row, sort_keys=True, separators=(",", ":"),
                                ensure_ascii=True) + "\n").encode("ascii"))
    return rows


def row_cli(gen, argv, selftest) -> int:
    """gen_wing.main, shared: --seed/--n/--out or --selftest."""
    ap = argparse.ArgumentParser(
        prog=os.path.splitext(os.path.basename(gen.__file__))[0],
        description=gen.__doc__.splitlines()[0])
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args(argv)
    if args.selftest:
        return selftest()
    if args.seed is None or args.n is None or not args.out:
        ap.error("--seed, --n and --out are required (or --selftest)")
    rows = row_generate(gen, args.seed, args.n, args.out)
    print("wrote %d STL(s) and manifest_%s.jsonl to %s"
          % (len(rows), gen.FAMILY, args.out))
    return 0


def check_determinism(gen, seed: int, indices: tuple) -> str:
    """R9 as assertions: rebuilds are bit-identical, alone and poisoned, the
    JSON round-trip of params rebuilds the same sha, and the sources use no
    randomness beyond np.random.default_rng."""
    rows, datas = {}, {}
    for i in indices:
        rows[i], datas[i] = gen.make_row(seed, i)
    for i in indices:
        gid = gen.geometry_id(seed, i)
        P, T = gen.build(rows[i]["params"])
        if stl_io.stl_bytes(P, T, gen.SOLID) != datas[i]:
            raise AssertionError("determinism: rebuild of %s differs" % gid)
        P2, T2 = gen.build(json.loads(json.dumps(rows[i]["params"])))
        if stl_io.stl_bytes(P2, T2, gen.SOLID) != datas[i]:
            raise AssertionError("determinism: json round-trip of %s differs"
                                 % gid)
    _poison_global_rng()
    for i in indices:
        gid = gen.geometry_id(seed, i)
        P, T = gen.build(rows[i]["params"])
        if stl_io.stl_bytes(P, T, gen.SOLID) != datas[i]:
            raise AssertionError("determinism: poisoned rebuild of %s differs"
                                 % gid)
    i0 = indices[0]
    row_s, data_s = gen.make_row(seed, i0)
    if data_s != datas[i0] or (json.dumps(row_s, sort_keys=True) !=
                               json.dumps(rows[i0], sort_keys=True)):
        raise AssertionError("determinism: %s built alone differs from the "
                             "batch" % gen.geometry_id(seed, i0))
    for path in (gen.__file__, __file__):
        with open(path, encoding="utf-8") as f:
            src = f.read()
        for mt in re.finditer(r"np\.random\.\w+|import\x20random", src):
            if mt.group(0) != "np.random.default_rng":
                raise AssertionError("determinism: %s uses %s"
                                     % (os.path.basename(path), mt.group(0)))
    return "%d row(s) rebuild bit-identically, alone and poisoned" % len(indices)


def check_rows(gen, seed: int, indices: tuple) -> str:
    """R10 as assertions: every row is a ManifestRow once split is added and
    carries expected_features's commensurate verdict."""
    for i in indices:
        row, _data = gen.make_row(seed, i)
        if "split" in row:
            raise AssertionError("rows: %s carries split" % row["geometry_id"])
        errs = []
        for s in ("tuning", "test"):
            errs += schema.errors(dict(row, split=s), "ManifestRow")
        if errs:
            raise AssertionError("rows: %s: %s" % (row["geometry_id"], errs[0]))
        if hasattr(gen, "expected_features"):
            exp = gen.expected_features(row["params"])["commensurate"]
            if row["commensurate"] != exp:
                raise AssertionError("rows: %s commensurate %s != expected %s"
                                     % (row["geometry_id"], row["commensurate"],
                                        exp))
    return "%d row(s) validate as ManifestRow with split added" % len(indices)


# --- the selftest ---------------------------------------------------------


def _shoelace2(ring: list) -> float:
    a2 = 0.0
    for i in range(len(ring)):
        u0, v0 = ring[i]
        u1, v1 = ring[(i + 1) % len(ring)]
        a2 += u0 * v1 - u1 * v0
    return 0.5 * a2


def _l_ring(A: float, B: float, t: float, step: float) -> list:
    return (line((0.0, 0.0), (A, 0.0), divs(A, step))
            + line((A, 0.0), (A, t), divs(t, step))
            + line((A, t), (t, t), divs(A - t, step))
            + line((t, t), (t, B), divs(B - t, step))
            + line((t, B), (0.0, B), divs(t, step))
            + line((0.0, B), (0.0, 0.0), divs(B, step)))


def _selftest() -> int:
    def group_grid_box():
        xs = axis_coords(0.0, 2.0, 4)
        ys = axis_coords(0.0, 1.0, 2)
        zs = axis_coords(0.0, 1.0, 2)
        P, T = grid_box(xs, ys, zs)
        nx, ny, nz = 4, 2, 2
        nP = (nx + 1) * (ny + 1) * (nz + 1) - (nx - 1) * (ny - 1) * (nz - 1)
        nT = 4 * (nx * ny + ny * nz + nx * nz)
        assert len(P) == nP, "nP %d != %d" % (len(P), nP)
        assert len(T) == nT, "nT %d != %d" % (len(T), nT)
        vol = stl_io.signed_volume(P, T)
        assert abs(vol - 2.0) <= 1e-12, "volume %r" % vol
        return ("4x2x2 grid: closed, nP %d (independent count), nT %d, "
                "volume 2.0" % (len(P), len(T)))

    def group_diagonal():
        xs = axis_coords(0.0, 2.0, 4)
        ys = axis_coords(0.0, 1.0, 2)
        zs = axis_coords(0.0, 1.0, 2)
        P, T = grid_box(xs, ys, zs)
        for axis in range(3):
            others = [a for a in range(3) if a != axis]
            sets = []
            for side in (0, 1):
                w = (xs if axis == 0 else ys if axis == 1 else zs)
                w = float(w[0] if side == 0 else w[-1])
                cs = []
                for t in T:
                    if all(P[t[k], axis] == w for k in range(3)):
                        cs.append((float(P[t][:, others[0]].mean()),
                                   float(P[t][:, others[1]].mean())))
                sets.append(sorted(cs))
            assert sets[0] == sets[1] and sets[0], \
                "axis %d: opposite faces are not mirror-triangulated" % axis
        return "opposite faces of the grid box share one centroid set per axis"

    def group_loft():
        ring = _l_ring(1.0, 0.6, 0.1, 0.025)
        cen = (0.05, 0.05)
        P1, T1 = prism(ring, [0.0, 0.5], (0, 2, 1), cen)
        P2, T2 = prism(ring, [0.0, 0.5], (0, 1, 2), cen)
        for P, T, ax in ((P1, T1, "(0,2,1)"), (P2, T2, "(0,1,2)")):
            assert abs(stl_io.signed_volume(P, T) - 0.075) <= 1e-12, \
                "L prism %s volume %r" % (ax, stl_io.signed_volume(P, T))
        hex1 = [(2.0, 0.0), (1.0, 1.7320508), (-1.0, 1.7320508), (-2.0, 0.0),
                (-1.0, -1.7320508), (1.0, -1.7320508)]
        hex2 = [(0.5 * u, 0.5 * v) for (u, v) in hex1]
        h = 0.8
        a1 = _shoelace2(hex1)
        vc = h * (a1 + a1 / 4.0 + math.sqrt(a1 * a1 / 4.0)) / 3.0
        P3, T3 = loft([hex1, hex2], [0.0, h], (0, 1, 2), ((0.0, 0.0),) * 2)
        got = stl_io.signed_volume(P3, T3)
        assert abs(got - vc) <= 1e-12, "hex frustum %r != %r" % (got, vc)
        return ("L ring under (0,2,1) and (0,1,2): volume 0.075 to 1e-12; "
                "tapered hexagon frustum exact to 1e-12")

    def group_sphere():
        out = []
        for axis in (0, 1, 2):
            c = [0.1, -0.2, 0.3]
            R = 0.3
            P, T = uv_sphere(R, c, axis, 64, 32)
            a3 = {0: (1, 2, 0), 1: (2, 0, 1), 2: (0, 1, 2)}[axis][2]
            assert P[0, a3] == c[a3] + R, "north pole inexact"
            assert P[-1, a3] == c[a3] - R, "south pole inexact"
            vc = 4.0 / 3.0 * math.pi * R ** 3
            dev = abs(stl_io.signed_volume(P, T) / vc - 1.0)
            assert dev < 0.005, "axis %d volume dev %r" % (axis, dev)
            out.append("axis %d %.3f %%" % (axis, 100.0 * dev))
        return "axes 0,1,2 closed, poles exact, volume dev %s (0.401 %%)" % (
            ", ".join(out))

    def group_circle():
        ring = circle(0.25, -0.5, 1.5, 64)
        assert len(ring) == 64
        assert ring[0] == (1.75, -0.5)
        assert ring[16] == (0.25, 1.0)
        assert ring[32] == (-1.25, -0.5)
        assert ring[48] == (0.25, -2.0)
        assert _shoelace2(ring) > 0.0, "circle not counter-clockwise"
        return "quarter points exact, shoelace %.3f > 0 (CCW)" % _shoelace2(ring)

    def group_combine():
        b1 = grid_box(axis_coords(0.0, 1.0, 2), axis_coords(0.0, 1.0, 2),
                      axis_coords(0.0, 1.0, 2))
        b2 = grid_box(axis_coords(2.0, 3.0, 2), axis_coords(0.0, 1.0, 2),
                      axis_coords(0.0, 1.0, 2))
        P, T = combine([b1, b2])
        er = stl_io.edge_report(T, len(P))
        assert not (er["open"] or er["non_manifold"] or er["same_direction"])
        assert stl_io.euler_characteristic(P, T) == 4
        assert abs(stl_io.signed_volume(P, T) - 2.0) <= 1e-12
        b3 = grid_box(axis_coords(0.5, 1.5, 2), axis_coords(0.0, 1.0, 2),
                      axis_coords(0.0, 1.0, 2))
        try:
            combine([b1, b3])
            raise AssertionError("overlapping boxes were accepted")
        except ValueError as e:
            assert "combine: bodies 0 and 1 overlap" in str(e), str(e)
        return "two boxes: edge report 0, euler 4, volume 2.0; overlap refused"

    def group_lattice():
        cases = [
            ("box-c", [[0, 1], [-0.25, 0.25], [-0.25, 0.25]], 0.5),
            ("box-n", [[0.0137, 0.9137], [-0.2113, 0.2291], [-0.1709, 0.1893]],
             None),
            ("cube", [[1, 2], [1, 2], [1, 2]], 1.0),
            ("cube-offset", [[0.0137, 1.0137], [0.0137, 1.0137],
                             [0.0137, 1.0137]], 1.0),
            ("two-cubes-1.5", [[0, 1, 1.5, 2.5], [0, 1], [0, 1]], 0.5),
            ("two-cubes-1.3137", [[0, 1, 1.3137, 2.3137], [0, 1], [0, 1]],
             None),
            ("thin-slab", [[0.0, 0.0103], [0, 1], [0, 1]], None),
        ]
        for name, planes, want in cases:
            got = lattice_spacing(planes)
            if want is None:
                assert got is None, "%s: %r is not None" % (name, got)
            else:
                assert got is not None and abs(got - want) <= 1e-12, \
                    "%s: %r != %r" % (name, got, want)
        return "7 of 7 agree with features.py's verdicts"

    def group_refusals():
        def refuses(fn, frag):
            try:
                fn()
            except ValueError as e:
                assert frag in str(e), "wrong refusal: %s" % e
                return str(e)
            raise AssertionError("no ValueError naming %r" % frag)

        bad = [
            ("grid x not increasing",
             lambda: grid_box([0.0, 1.0, 1.0], [0.0, 1.0], [0.0, 1.0]),
             "grid_box: x coordinates not strictly increasing"),
            ("clockwise ring",
             lambda: loft([[(0.0, 0.0), (0.0, 1.0), (1.0, 0.0)],
                           [(0.0, 0.0), (0.0, 1.0), (1.0, 0.0)]], [0.0, 1.0],
                          (0, 1, 2), ((0.2, 0.2),) * 2),
             "not counter-clockwise"),
            ("repeated ring point",
             lambda: loft([[(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (1.0, 1.0)],
                           [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (1.0, 1.0)]],
                          [0.0, 1.0], (0, 1, 2), ((0.4, 0.4),) * 2),
             "ring 0: repeated point"),
            ("centre does not see an edge",
             lambda: prism(_l_ring(1.0, 0.6, 0.1, 0.025), [0.0, 0.5],
                           (0, 1, 2), (0.5, 0.3)),
             "cap bottom: centre does not see edge"),
            ("bad axes tuple",
             lambda: prism([(0.0, 0.0), (1.0, 0.0), (1.0, 1.0),
                            (0.0, 1.0)], [0.0, 1.0], (0, 0, 1), (0.4, 0.4)),
             "axes: (0, 0, 1) is not a permutation of (0, 1, 2)"),
            ("overlapping bodies",
             lambda: combine([grid_box(axis_coords(0.0, 1.0, 1),
                                       axis_coords(0.0, 1.0, 1),
                                       axis_coords(0.0, 1.0, 1))] * 2),
             "combine: bodies 0 and 1 overlap"),
            ("circle n not a multiple of 4",
             lambda: circle(0.0, 0.0, 1.0, 30), "circle: n 30"),
        ]
        for _name, fn, frag in bad:
            refuses(fn, frag)
        return "%d by name" % len(bad)

    groups = [("[ok] grid_box", group_grid_box),
              ("[ok] diagonal rule", group_diagonal),
              ("[ok] loft", group_loft),
              ("[ok] uv_sphere", group_sphere),
              ("[ok] circle", group_circle),
              ("[ok] combine", group_combine),
              ("[ok] lattice", group_lattice),
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


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="meshkit",
                                 description=__doc__.splitlines()[0])
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args(argv)
    if args.selftest:
        return _selftest()
    ap.error("--selftest is required (meshkit is a library)")
    return 2


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    sys.exit(main())
