#!/usr/bin/env python3
# meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
# Source-available, not Open Source. See LICENSE at the repository root.
# No GPL-licensed source was consulted.

"""The F1 promo video's post-processing: three npz for scene.py + numbers.json.

Exports one snapshot of the promo GPU solve (tools/promo/solve.py's out) as
the three npz inputs tools/promo/scene.py imports - cp.npz (Cp over the
sim-surface sample points), streamlines.npz (speed-coloured polylines) and
slice.npz (the y = 0 mid-plane speed grid) - plus numbers.json with Cd, Cl,
the Cp range, the streamline and slice stats, the three files' hashes and a
forces.csv cross-check.

Cp: the solver's p is kinematic (m2/s2); q_kin = 0.5*U**2 and p_ref is the
area-weighted mean of the owner-cell kinematic p over the inlet patch
faces, so a wall face's Cp = (p_owner - p_ref)/q_kin, and each sim point
(the triangle centroid of its patch STL) carries the Cp of the nearest
wall-face centre of the SAME mesh patch.

Interpolation: the velocity at an arbitrary point is the inverse-distance-
squared mean over the K_NEAR = 8 nearest cell centres (exactly the nearest
centre's U when it lies closer than 1e-12), and a point counts as fluid iff
min_j d_j / h_ij <= FLUID_RATIO over the K_FLUID = 64 nearest centres with
the cell size h = V**(1/3), the volume from the divergence theorem; points
that already pass over their first K_NEAR neighbours are fluid, and only
the failing points are re-queried at K_FLUID (one batched query), so a
point next to a refinement-level step is not marked solid just because the
finer neighbours crowd its coarse container out of the K_NEAR list.

Seeding: three groups marched from each seed in both directions with the
midpoint rule on the unit direction - 160 lines across the front wing, 4 x
48 around the wheel clusters, 2 x 60 at the rear-wing tips - dropped when
they leave the march box or the fluid; 300..600 lines are kept.

The mesh and geometry derive from a CC BY 4.0 model ("F1 2026 concept" by
Qvist_Designs, via Sketchfab), so NOTHING made from them is written inside
the repository: every output goes under --out, which must lie outside the
repository, with the model's LICENSE.txt copied beside it.

--selftest runs T1-T8 on the CPU.
"""

import argparse
import contextlib
import hashlib
import io
import json
import math
import os
import re
import shutil
import sys
import tempfile
import time

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tunnel_mesh import refuse, read_stl_bin  # noqa: E402
from solve import (np_nums, read_boundary, read_points, read_ints, read_faces,  # noqa: E402
                   PolyMesh, read_foam_scalar, wheel_clusters, patch_pressure_force,
                   CAR, WHEEL_ORDER, REPO, write_poly_mesh,
                   _tiny_points, _tiny_faces, _tiny_owner)

TOOL = "tools/promo/post.py"
VERSION = "promo-post/1"
K_NEAR = 8             # cell centres per velocity interpolation
K_FLUID = 64           # cell centres per fluid test
FLUID_RATIO = 0.9      # fluid iff min_j d_j / h_ij <= this
N_MAX = 1000           # march steps per direction
UP_GAP = 0.3           # seeds this far ahead of the nose
END_GAP = 5.0          # march box ends this far behind the tail
Y_HALF = 3.0           # march box |y| half width
Z_TOP = 3.0            # march box ceiling
U_MIN_FRAC = 1e-3      # stall speed as a fraction of U
LINES_MIN = 300
LINES_MAX = 600
SLICE_Y0 = 0.0         # the mid-plane
SLICE_BACK = 1.0       # slice starts this far ahead of the nose
SLICE_AFT = 4.0        # and runs this far behind the tail
SLICE_ZTOP = 1.6
XCHECK_LIMIT = 1e-9
CHUNK = 100000         # slice query chunk

_P_REF_RULE = "area-weighted mean of the owner-cell kinematic p over the inlet patch"
_GROUP_NAMES = {0: "front_wing", 1: "wheels", 2: "rear_wing_tips"}


def _posix(path):
    return path.replace(os.sep, "/")


def prog(stage, t0):
    print("[promo-post] " + stage + " %.1fs" % (time.time() - t0))
    sys.stdout.flush()


# ---------------------------------------------------------------- field reader

def read_foam_vector(path, n_expected):
    """internalField of one volVectorField -> (n_expected, 3) float64.

    Only the internalField list is parsed: the boundaryField that follows
    also contains lists, and the first line that is just ")" closes the
    internalField list."""
    text = open(path, encoding="utf-8", errors="replace").read()
    i = text.index("internalField")
    seg = text[i:]
    u = seg.find("uniform")
    v = seg.find("nonuniform")
    if v != -1 and (u == -1 or v < u):
        m = re.search(r"nonuniform\s+List<vector>\s+(\d+)", seg)
        if not m:
            refuse("PP-FIELD", path + ": unparseable nonuniform internalField")
        cnt = int(m.group(1))
        a = seg.index("(", m.end())
        b = seg.index("\n)", a)
        vals = np_nums(seg[a + 1:b].replace("(", " ").replace(")", " "), np.float64)
        if cnt != n_expected or vals.size != 3 * cnt:
            refuse("PP-FIELD", path + ": " + str(cnt) + " declared, " +
                   str(vals.size // 3) + " parsed, " + str(n_expected) + " cells")
        return vals.reshape(cnt, 3)
    if u != -1:
        rest = seg[u + len("uniform"):].strip().rstrip(";").strip()
        m = re.match(r"\(\s*(\S+)\s+(\S+)\s+(\S+)\s*\)", rest)
        if not m:
            refuse("PP-FIELD", path + ": unparseable uniform internalField")
        one = np.array([float(m.group(1)), float(m.group(2)), float(m.group(3))])
        return np.tile(one, (max(n_expected, 1), 1))
    refuse("PP-FIELD", path + ": internalField is neither uniform nor nonuniform")


# ---------------------------------------------------------------- cells

def cell_geometry(mesh, owner, neighbour):
    """C2 -> (Sf, xf, |Sf|, C, V, h, nc): face area vectors and centres, cell
    centres (area-weighted over all faces of the cell, both sides), volumes
    from the divergence theorem, sizes h = V**(1/3); all np.bincount."""
    n_faces = int(mesh.arity.shape[0])
    n_internal = int(neighbour.shape[0])
    sf, xf = mesh.face_sf_centres(0, n_faces)
    mag_sf = np.linalg.norm(sf, axis=1)
    nc = int(owner.max()) + 1
    flux = (xf * sf).sum(axis=1)
    vol = np.bincount(owner, weights=flux, minlength=nc)
    vol -= np.bincount(neighbour, weights=flux[:n_internal], minlength=nc)
    vol /= 3.0
    wsum = np.bincount(owner, weights=mag_sf, minlength=nc)
    wsum = wsum + np.bincount(neighbour, weights=mag_sf[:n_internal], minlength=nc)
    cen = np.empty((nc, 3))
    for ax in range(3):
        comp = np.bincount(owner, weights=mag_sf * xf[:, ax], minlength=nc)
        comp = comp + np.bincount(neighbour, weights=mag_sf[:n_internal] * xf[:n_internal, ax],
                                  minlength=nc)
        cen[:, ax] = comp / wsum
    return sf, xf, mag_sf, cen, vol, np.cbrt(vol), nc


def make_vel(cen, h, u_cells, nc):
    """C2: vel(P) -> (U (n,3), fluid (n,)) - inverse-distance-squared mean of
    the K_NEAR nearest cell centres; fluid iff min_j d_j/h_ij <= FLUID_RATIO
    over the min(K_FLUID, nc) nearest centres. Points passing over the first
    K_NEAR neighbours are fluid; only the failing points are re-queried at
    K_FLUID (one batched query), so a refinement-level step cannot crowd the
    containing coarse cell out of the fluid test."""
    k = min(K_NEAR, int(nc))
    kf = min(K_FLUID, int(nc))
    tree = cKDTree(cen)

    def vel(P):
        d, idx = tree.query(P, k=k)
        if k == 1:
            d = d[:, None]
            idx = idx[:, None]
        fluid = np.min(d / h[idx], axis=1) <= FLUID_RATIO
        if kf > k:
            bad = np.flatnonzero(~fluid)
            if bad.size:
                df, idxf = tree.query(P[bad], k=kf)
                if kf == 1:
                    df = df[:, None]
                    idxf = idxf[:, None]
                fluid[bad] = np.min(df / h[idxf], axis=1) <= FLUID_RATIO
        near = d[:, 0] < 1e-12
        safe = np.where(near[:, None], 1.0, d)
        w = 1.0 / (safe * safe)
        out = (w[:, :, None] * u_cells[idx]).sum(axis=1) / w.sum(axis=1)[:, None]
        if bool(near.any()):
            out[near] = u_cells[idx[near, 0]]
        return out, fluid

    return vel


# ---------------------------------------------------------------- Cp

def inlet_p_ref(owner, p, mag_sf, boundary):
    """C3: area-weighted mean of p[owner[f]] over the inlet patch faces."""
    for name, nf, st in boundary:
        if name == "inlet":
            pf = owner[st:st + nf]
            w = mag_sf[st:st + nf]
            return float((w * p[pf]).sum() / w.sum())
    refuse("PP-SOLVE", "boundary has no inlet patch")


def patch_face_cp(owner, p, p_ref, q_kin, start, count):
    """C3: (p[owner] - p_ref)/q_kin over one wall patch's face range."""
    return (p[owner[start:start + count]] - p_ref) / q_kin


def cp_nearest(sim_pts, face_centres, face_cp):
    """C3 for one patch: each sim point takes the Cp of the nearest wall-face
    centre of the same mesh patch -> (values, dist)."""
    tree = cKDTree(face_centres)
    d, i = tree.query(sim_pts, k=1)
    return face_cp[i], d


# ---------------------------------------------------------------- streamlines

def build_seeds(x_up, fw_lo, fw_hi, rw_lo, rw_hi, wheels_ccr):
    """C4: the three seed groups in order -> (seeds (n,3), groups (n,) int8).
    wheels_ccr is one (cx, cy, R) per wheel in WHEEL_ORDER."""
    seeds = []
    groups = []
    for j in range(20):
        y = fw_lo[1] + (j + 0.5) / 20.0 * (fw_hi[1] - fw_lo[1])
        for k in range(8):
            z = fw_lo[2] + (k + 0.5) / 8.0 * (fw_hi[2] - fw_lo[2])
            seeds.append((x_up, y, z))
            groups.append(0)
    for cx, cy, r in wheels_ccr:
        for a in np.linspace(-0.9, 0.9, 8):
            for i in range(6):
                b = 0.15 + 0.35 * i
                seeds.append((cx - 1.3 * r, cy + float(a) * r, b * r))
                groups.append(1)
    for y_tip in (rw_hi[1], rw_lo[1]):
        for dy in (-0.10, -0.06, -0.02, 0.02, 0.06, 0.10):
            for k in range(10):
                z = rw_lo[2] + (k + 0.5) / 10.0 * (rw_hi[2] - rw_lo[2])
                seeds.append((rw_lo[0] - 0.2, y_tip + dy, z))
                groups.append(2)
    return (np.array(seeds, dtype=np.float64),
            np.array(groups, dtype=np.int8))


def _march(sub, u0s, vel, ds, n_max, s, x_up, x_end, y_half, z_top, u_min):
    """One direction for a fixed seed set: all active lines advance together
    (one batched vel call per stage). Returns per seed (points (c,3), speeds)
    with c possibly 0."""
    m = sub.shape[0]
    X = sub.copy()
    Ux = u0s.copy()
    store = np.zeros((m, int(n_max), 3))
    spds = np.zeros((m, int(n_max)))
    cnt = np.zeros(m, dtype=np.int64)
    alive = np.ones(m, dtype=bool)
    out = [None] * m
    for _ in range(int(n_max)):
        ia = np.flatnonzero(alive)
        if ia.size == 0:
            break
        Xa = X[ia]
        Ua = Ux[ia]
        spx = np.linalg.norm(Ua, axis=1)
        dead = spx <= 0.0
        dv = np.where(dead[:, None], 0.0, Ua / np.where(spx > 0.0, spx, 1.0)[:, None])
        M = Xa + 0.5 * ds * s * dv
        Um, flm = vel(M)
        spm = np.linalg.norm(Um, axis=1)
        dead |= (~flm) | (spm < u_min)
        dm = np.zeros_like(Um)
        pos = spm > 0.0
        dm[pos] = Um[pos] / spm[pos, None]
        Y = Xa + ds * s * dm
        Uy, fly = vel(Y)
        spy = np.linalg.norm(Uy, axis=1)
        acc = ((~dead) & fly & (spy >= u_min) & (Y[:, 0] >= x_up) & (Y[:, 0] <= x_end) &
               (np.abs(Y[:, 1]) <= y_half) & (Y[:, 2] >= 0.0) & (Y[:, 2] <= z_top))
        ik = ia[acc]
        kk = cnt[ik]
        store[ik, kk] = Y[acc]
        spds[ik, kk] = spy[acc]
        cnt[ik] = kk + 1
        X[ik] = Y[acc]
        Ux[ik] = Uy[acc]
        for row in ia[~acc]:
            c = int(cnt[row])
            out[row] = (store[row, :c].copy(), spds[row, :c].copy())
            alive[row] = False
    for row in np.flatnonzero(alive):
        c = int(cnt[row])
        out[row] = (store[row, :c].copy(), spds[row, :c].copy())
    return out


def trace(seeds, vel, ds, n_max, x_up, x_end, y_half, z_top, u_min):
    """C4: per seed None (dropped) or (points (m,3), speed (m,)); the line is
    the backward points reversed, then the seed, then the forward points."""
    seeds = np.ascontiguousarray(seeds, dtype=np.float64)
    n = seeds.shape[0]
    out = [None] * n
    if n == 0:
        return out
    u0, fl0 = vel(seeds)
    spd0 = np.linalg.norm(u0, axis=1)
    ok = (fl0 & (spd0 >= u_min) & (seeds[:, 0] >= x_up) & (seeds[:, 0] <= x_end) &
          (np.abs(seeds[:, 1]) <= y_half) & (seeds[:, 2] >= 0.0) & (seeds[:, 2] <= z_top))
    if not bool(ok.any()):
        return out
    idx = np.flatnonzero(ok)
    sub = seeds[idx]
    u0s = u0[idx]
    fwd = _march(sub, u0s, vel, ds, n_max, 1.0, x_up, x_end, y_half, z_top, u_min)
    bwd = _march(sub, u0s, vel, ds, n_max, -1.0, x_up, x_end, y_half, z_top, u_min)
    for row, j in enumerate(idx):
        bp, bs = bwd[row]
        fp, fs = fwd[row]
        if bp.shape[0] + fp.shape[0] == 0:
            continue
        out[j] = (np.concatenate([bp[::-1], sub[row:row + 1], fp]),
                  np.concatenate([bs[::-1], spd0[j:j + 1], fs]))
    return out


# ---------------------------------------------------------------- slice

def slice_axes(x0, x1, z1, dx):
    """C5 -> (x (nx,), z (nz,)): x from x0 on the dx grid, z offset half a dx."""
    nx = int(math.floor((x1 - x0) / dx + 1e-9)) + 1
    nz = int(math.floor((z1 - 0.5 * dx) / dx + 1e-9)) + 1
    return x0 + dx * np.arange(nx, dtype=np.float64), 0.5 * dx + dx * np.arange(nz, dtype=np.float64)


def slice_speed(vel, xs, zs, y0, chunk=CHUNK):
    """C5: speed[k, i] = |vel((x_i, y0, z_k))|, NaN where not fluid; the
    grid is queried in chunks."""
    gx, gz = np.meshgrid(xs, zs)                       # (nz, nx)
    P = np.stack([gx.ravel(), np.full(gx.size, float(y0)), gz.ravel()], axis=1)
    out = np.full(P.shape[0], np.nan)
    for a in range(0, P.shape[0], chunk):
        b = min(a + chunk, P.shape[0])
        U, fl = vel(P[a:b])
        out[a:b] = np.where(fl, np.linalg.norm(U, axis=1), np.nan)
    return out.reshape(zs.size, xs.size)


# ---------------------------------------------------------------- npz

def _npz_key(z, key, path):
    if key not in z.files:
        refuse("PP-NPZ", path + ": missing key " + key)
    return z[key]


def _npz_float(a):
    return np.issubdtype(a.dtype, np.floating)


def check_npz(kind, path):
    """C6: re-open one npz and check every row of the contract table; any
    failure refuses PP-NPZ."""
    try:
        z = np.load(path)
    except Exception as exc:
        refuse("PP-NPZ", path + ": cannot load npz (" + repr(exc) + ")")
    if kind == "surface":
        pts = _npz_key(z, "points", path)
        vals = _npz_key(z, "values", path)
        patch = _npz_key(z, "patch", path)
        dist = _npz_key(z, "dist", path)
        if pts.ndim != 2 or pts.shape[1] != 3 or not _npz_float(pts):
            refuse("PP-NPZ", path + ": points must be (n,3) float")
        n = pts.shape[0]
        if vals.shape != (n,) or not _npz_float(vals):
            refuse("PP-NPZ", path + ": values must be (n,) float matching points")
        if not bool(np.isfinite(vals).all()):
            refuse("PP-NPZ", path + ": values must be finite")
        if patch.shape != (n,) or patch.dtype != np.int8:
            refuse("PP-NPZ", path + ": patch must be (n,) int8")
        if dist.shape != (n,) or dist.dtype != np.float32:
            refuse("PP-NPZ", path + ": dist must be (n,) float32")
    elif kind == "streamlines":
        pts = _npz_key(z, "points", path)
        spd = _npz_key(z, "speed", path)
        off = _npz_key(z, "offsets", path)
        grp = _npz_key(z, "group", path)
        sds = _npz_key(z, "seeds", path)
        if pts.ndim != 2 or pts.shape[1] != 3 or not _npz_float(pts):
            refuse("PP-NPZ", path + ": points must be (m,3) float")
        m = pts.shape[0]
        if spd.shape != (m,) or not _npz_float(spd):
            refuse("PP-NPZ", path + ": speed must be (m,) float matching points")
        if not bool(np.isfinite(spd).all()):
            refuse("PP-NPZ", path + ": speed must be finite")
        if off.ndim != 1 or not np.issubdtype(off.dtype, np.integer):
            refuse("PP-NPZ", path + ": offsets must be a 1-D integer array")
        if off.size < 1 or int(off[0]) != 0 or int(off[-1]) != m:
            refuse("PP-NPZ", path + ": offsets must start at 0 and end at " + str(m))
        if off.size >= 2 and not bool((np.diff(off.astype(np.int64)) >= 2).all()):
            refuse("PP-NPZ", path + ": every polyline needs >= 2 points")
        k = off.size - 1
        if grp.shape != (k,) or grp.dtype != np.int8:
            refuse("PP-NPZ", path + ": group must be (k,) int8")
        if sds.shape != (k, 3) or not _npz_float(sds):
            refuse("PP-NPZ", path + ": seeds must be (k,3) float")
    elif kind == "slice":
        xs = _npz_key(z, "x", path)
        zs = _npz_key(z, "z", path)
        y0 = _npz_key(z, "y0", path)
        spd = _npz_key(z, "speed", path)
        for nm, a in (("x", xs), ("z", zs)):
            if a.ndim != 1 or a.size < 2 or not _npz_float(a):
                refuse("PP-NPZ", path + ": " + nm + " must be (n>=2,) float")
            if not bool((np.diff(a) > 0.0).all()):
                refuse("PP-NPZ", path + ": " + nm + " must be strictly increasing")
        if y0.ndim != 0 or not _npz_float(y0):
            refuse("PP-NPZ", path + ": y0 must be a 0-d float")
        if spd.shape != (zs.size, xs.size) or not _npz_float(spd):
            refuse("PP-NPZ", path + ": speed must be (nz,nx) float")
    else:
        refuse("PP-NPZ", path + ": unknown npz kind " + kind)


# ---------------------------------------------------------------- validation

_SNAP_RE = re.compile(r"^[0-9]+(\.[0-9]+)?$")
_POLY_FILES = ("boundary", "faces", "owner", "neighbour", "points")


def _inside_repo(path):
    rp = os.path.normcase(os.path.abspath(REPO))
    pp = os.path.normcase(os.path.abspath(path))
    return pp == rp or pp.startswith(rp + os.sep)


def _validate_solve(solve_dir, time_arg):
    """PP-SOLVE gates -> (snapshot dir, snapshot time)."""
    for rel in ["solve.json", "forces.csv", "LICENSE.txt"] + \
               ["case/constant/polyMesh/" + f for f in _POLY_FILES]:
        if not os.path.isfile(os.path.join(solve_dir, rel)):
            refuse("PP-SOLVE", solve_dir + " lacks " + rel)
    case = os.path.join(solve_dir, "case")
    snaps = [nm for nm in os.listdir(case)
             if _SNAP_RE.match(nm) and os.path.isdir(os.path.join(case, nm))]
    if not snaps:
        refuse("PP-SOLVE", solve_dir + " has no snapshot dir under case")
    if time_arg is None:
        nm = None
        for cand in snaps:
            if float(cand) > 0.0 and (nm is None or float(cand) > float(nm)):
                nm = cand
        if nm is None:
            refuse("PP-SOLVE", solve_dir + " has no snapshot dir named > 0")
    else:
        nm = None
        for cand in snaps:
            if abs(float(cand) - float(time_arg)) <= 1e-9:
                nm = cand
                break
        if nm is None:
            refuse("PP-SOLVE", "--time " + repr(float(time_arg)) +
                   " names no snapshot dir (have " + ", ".join(sorted(snaps)) + ")")
    snap = os.path.join(case, nm)
    for f in ("U", "p"):
        if not os.path.isfile(os.path.join(snap, f)):
            refuse("PP-SOLVE", snap + " lacks " + f)
    return snap, float(nm)


def _validate_geom(geom_dir):
    """PP-GEOM gates: the five patch STLs and sim_geom.json, with each STL's
    triangle count equal to its sim_geom.json patches entry."""
    sgj = os.path.join(geom_dir, "sim_geom.json")
    if not os.path.isfile(sgj):
        refuse("PP-GEOM", sgj + ": missing")
    sg = json.load(open(sgj, encoding="utf-8"))
    tris = {ent["name"]: int(ent["tris"]) for ent in sg.get("patches", [])}
    for name in CAR:
        stl = os.path.join(geom_dir, name + ".stl")
        if not os.path.isfile(stl):
            refuse("PP-GEOM", stl + ": missing patch STL")
        n = (os.path.getsize(stl) - 84) // 50
        if name in tris and tris[name] != n:
            refuse("PP-GEOM", stl + ": " + str(n) + " triangles, sim_geom.json says " +
                   str(tris[name]))


# ---------------------------------------------------------------- export

def cmd_export(args):
    t0 = time.time()
    out_dir = os.path.abspath(args.out)
    if _inside_repo(out_dir):
        refuse("PP-OUT", out_dir + " resolves inside the repository " + REPO)
    if os.path.isdir(out_dir) and os.listdir(out_dir):
        refuse("PP-OUT", out_dir + " exists and is not empty")
    solve_dir = os.path.abspath(args.solve)
    snap, snap_t = _validate_solve(solve_dir, args.time)
    geom_dir = os.path.abspath(args.geom)
    _validate_geom(geom_dir)

    pmd = os.path.join(solve_dir, "case", "constant", "polyMesh")
    points = read_points(os.path.join(pmd, "points"))
    arity, ptr, flat = read_faces(os.path.join(pmd, "faces"))
    owner = read_ints(os.path.join(pmd, "owner"))
    neighbour = read_ints(os.path.join(pmd, "neighbour"))
    boundary = read_boundary(os.path.join(pmd, "boundary"))
    mesh = PolyMesh(points, arity, ptr, flat)
    prog("mesh read", t0)

    sf, xf, mag_sf, cen, vol, h, nc = cell_geometry(mesh, owner, neighbour)
    prog("cells", t0)

    sj = json.load(open(os.path.join(solve_dir, "solve.json"), encoding="utf-8"))
    U = float(sj["params"]["U"])
    rho = float(sj["params"]["rho"])
    a_ref = float(sj["params"]["A_ref"])
    q = float(sj["params"]["q"])
    u_cells = read_foam_vector(os.path.join(snap, "U"), nc)
    p = read_foam_scalar(os.path.join(snap, "p"), nc)
    if p.shape[0] != nc:
        refuse("PP-FIELD", os.path.join(snap, "p") + ": " + str(p.shape[0]) +
               " values for " + str(nc) + " cells")
    prog("fields", t0)

    vel = make_vel(cen, h, u_cells, nc)

    # cp -----------------------------------------------------------------
    q_kin = 0.5 * U * U
    p_ref = inlet_p_ref(owner, p, mag_sf, boundary)
    rng = {name: (st, nf) for name, nf, st in boundary}
    tris_map = {}
    cp_pts, cp_vals, cp_patch, cp_dist = [], [], [], []
    for gi, name in enumerate(CAR):
        tris_map[name] = read_stl_bin(os.path.join(geom_dir, name + ".stl"))
        st, nf = rng[name]
        fc = xf[st:st + nf]
        fcp = patch_face_cp(owner, p, p_ref, q_kin, st, nf)
        sim = tris_map[name].mean(axis=1)
        v, d = cp_nearest(sim, fc, fcp)
        cp_pts.append(sim)
        cp_vals.append(v)
        cp_patch.append(np.full(sim.shape[0], gi, dtype=np.int8))
        cp_dist.append(d.astype(np.float32))
    cp_pts = np.concatenate(cp_pts)
    cp_vals = np.concatenate(cp_vals)
    cp_patch = np.concatenate(cp_patch)
    cp_dist = np.concatenate(cp_dist)
    prog("cp", t0)

    # streamlines ----------------------------------------------------------
    verts = np.concatenate([tris_map[n].reshape(-1, 3) for n in CAR])
    car_lo_x = float(verts[:, 0].min())
    car_hi_x = float(verts[:, 0].max())
    fw_lo = tris_map["front_wing"].reshape(-1, 3).min(axis=0)
    fw_hi = tris_map["front_wing"].reshape(-1, 3).max(axis=0)
    rw_lo = tris_map["rear_wing"].reshape(-1, 3).min(axis=0)
    rw_hi = tris_map["rear_wing"].reshape(-1, 3).max(axis=0)
    clusters, _x_split = wheel_clusters(geom_dir, U)
    wheels_ccr = [(clusters[w]["cx"], clusters[w]["cy"], clusters[w]["R"])
                  for w in WHEEL_ORDER]
    x_up = car_lo_x - UP_GAP
    x_end = car_hi_x + END_GAP
    u_min = U_MIN_FRAC * U
    seeds, groups = build_seeds(x_up, fw_lo, fw_hi, rw_lo, rw_hi, wheels_ccr)
    lines = trace(seeds, vel, args.ds, N_MAX, x_up, x_end, Y_HALF, Z_TOP, u_min)
    kept = [j for j, r in enumerate(lines) if r is not None]
    if not (LINES_MIN <= len(kept) <= LINES_MAX):
        refuse("PP-LINES", str(len(kept)) + " kept lines outside [" +
               str(LINES_MIN) + ", " + str(LINES_MAX) + "]")
    sl_pts = np.concatenate([lines[j][0] for j in kept])
    sl_spd = np.concatenate([lines[j][1] for j in kept])
    offs = np.zeros(len(kept) + 1, dtype=np.int64)
    offs[1:] = np.cumsum([lines[j][0].shape[0] for j in kept])
    sl_grp = groups[np.array(kept, dtype=np.int64)]
    sl_seeds = seeds[np.array(kept, dtype=np.int64)]
    prog("streamlines", t0)

    # slice ----------------------------------------------------------------
    sl_x0 = car_lo_x - SLICE_BACK
    sl_x1 = car_hi_x + SLICE_AFT
    xs, zs = slice_axes(sl_x0, sl_x1, SLICE_ZTOP, args.slice_dx)
    spd2d = slice_speed(vel, xs, zs, SLICE_Y0)
    prog("slice", t0)

    # write ----------------------------------------------------------------
    os.makedirs(out_dir, exist_ok=True)
    lic_dst = os.path.join(out_dir, "LICENSE.txt")
    shutil.copyfile(os.path.join(solve_dir, "LICENSE.txt"), lic_dst)
    p_cp = os.path.join(out_dir, "cp.npz")
    np.savez(p_cp, points=cp_pts, values=cp_vals, patch=cp_patch, dist=cp_dist)
    p_sl = os.path.join(out_dir, "streamlines.npz")
    np.savez(p_sl, points=sl_pts, speed=sl_spd, offsets=offs, group=sl_grp,
             seeds=sl_seeds)
    p_sli = os.path.join(out_dir, "slice.npz")
    np.savez(p_sli, x=xs, z=zs, y0=np.float64(SLICE_Y0), speed=spd2d)
    check_npz("surface", p_cp)
    check_npz("streamlines", p_sl)
    check_npz("slice", p_sli)

    # numbers --------------------------------------------------------------
    f_tool = np.zeros(3)
    for name in CAR:
        st, nf = rng[name]
        f_tool = f_tool + patch_pressure_force(mesh, owner, p, st, nf)
    f_csv = None
    with open(os.path.join(solve_dir, "forces.csv"), encoding="utf-8") as fh:
        rows = [ln.strip().split(",") for ln in fh if ln.strip()]
    for r in rows[1:]:
        if abs(float(r[0]) - snap_t) <= 1e-9:
            f_csv = np.array([float(r[2]), float(r[3]), float(r[4])])
            break
    if f_csv is None:
        rel = None
        xpass = False
    else:
        rel = float(np.linalg.norm(f_tool - f_csv) /
                    max(float(np.linalg.norm(f_csv)), 1.0))
        xpass = bool(rel <= XCHECK_LIMIT)

    per_patch = {}
    for gi, name in enumerate(CAR):
        sel = cp_patch == gi
        v = cp_vals[sel]
        per_patch[name] = {"n": int(sel.sum()), "min": float(v.min()),
                           "max": float(v.max()), "mean": float(v.mean())}
    fluid = np.isfinite(spd2d)
    lengths = [float(np.linalg.norm(np.diff(lines[j][0], axis=0), axis=1).sum())
               for j in kept]
    ki = np.array(kept, dtype=np.int64)
    groups_stat = {g: {"seeds": int((groups == gi).sum()),
                       "lines": int((groups[ki] == gi).sum())}
                   for gi, g in _GROUP_NAMES.items()}

    files = {}
    for nm, pth in (("cp.npz", p_cp), ("streamlines.npz", p_sl), ("slice.npz", p_sli)):
        blob = open(pth, "rb").read()
        files[nm] = {"bytes": len(blob), "sha256": hashlib.sha256(blob).hexdigest()}

    numbers = {
        "tool": TOOL,
        "version": VERSION,
        "solve_dir": _posix(solve_dir),
        "geom_dir": _posix(geom_dir),
        "out_dir": _posix(out_dir),
        "attribution": sj.get("attribution"),
        "time": float(snap_t),
        "seconds": time.time() - t0,
        "params": {"U": U, "rho": rho, "A_ref": a_ref, "q": q, "q_kin": q_kin,
                   "p_ref": p_ref, "p_ref_rule": _P_REF_RULE},
        "forces": {kk: sj["final_forces"][kk] for kk in
                   ("Cd", "Cl", "CdA", "ClA", "drag_N", "downforce_N")},
        "balance": sj["balance"],
        "window": {kk: sj["window"][kk] for kk in
                   ("Cd_mean", "Cd_min", "Cd_max", "Cl_mean", "Cl_min", "Cl_max")},
        "class": sj["class"],
        "reason_id": sj["reason_id"],
        "xcheck": {"F_tool": [float(v) for v in f_tool],
                   "F_csv": None if f_csv is None else [float(v) for v in f_csv],
                   "rel": rel, "limit": XCHECK_LIMIT, "pass": xpass},
        "cp": {"n": int(cp_vals.size), "min": float(cp_vals.min()),
               "max": float(cp_vals.max()), "mean": float(cp_vals.mean()),
               "p01": float(np.percentile(cp_vals, 1)),
               "p99": float(np.percentile(cp_vals, 99)),
               "dist_max": float(cp_dist.max()), "dist_mean": float(cp_dist.mean()),
               "per_patch": per_patch},
        "streamlines": {"n_seeds": int(seeds.shape[0]), "n_lines": len(kept),
                        "n_dropped": int(seeds.shape[0] - len(kept)),
                        "n_points": int(sl_pts.shape[0]), "ds": float(args.ds),
                        "groups": groups_stat,
                        "speed_min": float(sl_spd.min()),
                        "speed_max": float(sl_spd.max()),
                        "speed_p99": float(np.percentile(sl_spd, 99)),
                        "length_mean_m": float(np.mean(lengths))},
        "slice": {"y0": float(SLICE_Y0), "x0": float(sl_x0), "x1": float(sl_x1),
                  "z0": float(zs[0]), "z1": float(zs[-1]), "dx": float(args.slice_dx),
                  "nx": int(xs.size), "nz": int(zs.size),
                  "fluid_frac": float(fluid.sum() / fluid.size),
                  "speed_min": float(spd2d[fluid].min()),
                  "speed_max": float(spd2d[fluid].max())},
        "files": files,
        "caveats": [
            "castellated mesh: snap moved no point and no boundary layer was grown "
            "(tools/promo/MESH-FULL.md); y+ is far outside the wall-function range, "
            "so Cd and Cl are illustrative, not engineering values",
            "pseudo-transient march; class " + str(sj["class"]) +
            " by the house stopping rule (tools/promo/SOLVE.md)",
        ],
        "license_copied": os.path.isfile(lic_dst),
    }
    with open(os.path.join(out_dir, "numbers.json"), "w", encoding="utf-8") as fh:
        json.dump(numbers, fh, indent=1)
    prog("written", t0)


# ---------------------------------------------------------------- selftest

def _fixture_mesh(scale_x=1.0):
    pts = np.array(_tiny_points(), dtype=np.float64)
    pts[:, 0] *= scale_x
    faces = _tiny_faces()
    ars = np.array([len(f) for f in faces], dtype=np.int64)
    ptr = np.concatenate([np.zeros(1, dtype=np.int64), np.cumsum(ars)])
    flat = np.array([i for f in faces for i in f], dtype=np.int64)
    owner = np.array(_tiny_owner(), dtype=np.int64)
    return PolyMesh(pts, ars, ptr, flat), owner, np.array([1], dtype=np.int64)


def _t1():
    mesh, owner, nb = _fixture_mesh()
    sf, xf, mag, cen, vol, h, nc = cell_geometry(mesh, owner, nb)
    assert nc == 2
    assert np.allclose(cen, [[0.5, 0.5, 0.5], [1.5, 0.5, 0.5]], atol=1e-12, rtol=0)
    assert np.allclose(vol, [1.0, 1.0], atol=1e-12, rtol=0)
    assert np.allclose(h, [1.0, 1.0], atol=1e-12, rtol=0)
    mesh2, owner2, nb2 = _fixture_mesh(2.0)
    sf2, xf2, mag2, cen2, vol2, h2, nc2 = cell_geometry(mesh2, owner2, nb2)
    assert np.allclose(cen2, [[1.0, 0.5, 0.5], [3.0, 0.5, 0.5]], atol=1e-12, rtol=0)
    assert np.allclose(vol2, [2.0, 2.0], atol=1e-12, rtol=0)
    assert np.allclose(h2, [1.2599210498948732] * 2, atol=1e-12, rtol=0)
    print("[ok] T1 cell geometry (centres, volumes, h)")


def _t2():
    mesh, owner, nb = _fixture_mesh()
    sf, xf, mag, cen, vol, h, nc = cell_geometry(mesh, owner, nb)
    u_cells = np.array([[1.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
    vel = make_vel(cen, h, u_cells, nc)
    u, fl = vel(np.array([[1.0, 0.5, 0.5], [0.75, 0.5, 0.5],
                          [0.5, 0.5, 0.5], [-1.0, 0.5, 0.5]]))
    assert list(fl) == [True, True, True, False]
    assert np.allclose(u[0], [2.0, 0.0, 0.0], atol=1e-12, rtol=0)
    assert np.allclose(u[1], [1.2, 0.0, 0.0], rtol=1e-12, atol=0)
    assert np.array_equal(u[2], np.array([1.0, 0.0, 0.0]))

    # T2b: 12 fine centres (h 0.25) nearer than the containing coarse cell -
    # the 8-nearest fluid test fails P, the K_FLUID = 64 test must pass it.
    cen_b = np.array([(0.0, 0.0, 0.0)] +
                     [(0.625, a, b)
                      for a in (-0.375, -0.125, 0.125, 0.375)
                      for b in (-0.375, -0.125, 0.125, 0.375)])
    h_b = np.array([1.0] + [0.25] * 16)
    u_b = np.array([(1.0, 0.0, 0.0)] + [(2.0, 0.0, 0.0)] * 16)
    vel_b = make_vel(cen_b, h_b, u_b, 17)
    u, fl = vel_b(np.array([[0.45, 0.0, 0.0], [2.0, 0.0, 0.0]]))
    assert list(fl) == [True, False], fl
    assert np.allclose(u[0], [2.0, 0.0, 0.0], rtol=1e-12, atol=0), u[0]
    print("[ok] T2 velocity interpolation and fluid test (K_FLUID re-query)")


def _t3():
    def uni(P):
        return np.tile(np.array([1.0, 0.0, 0.0]), (P.shape[0], 1)), \
            np.ones(P.shape[0], dtype=bool)

    pts, spd = trace(np.array([[0.5, 0.0, 0.5]]), uni, 0.125, 1000,
                     0.0, 1.0, 10.0, 10.0, 1e-3)[0]
    assert pts.shape == (9, 3)
    assert np.array_equal(pts[:, 0], np.arange(9) * 0.125)
    assert np.allclose(pts[:, 1], 0.0, atol=1e-15)
    assert np.allclose(pts[:, 2], 0.5, atol=1e-15)
    assert np.array_equal(spd, np.ones(9))

    def rot(P):
        return np.stack([-P[:, 1], P[:, 0], np.zeros(P.shape[0])], axis=1), \
            np.ones(P.shape[0], dtype=bool)

    pts, spd = trace(np.array([[1.0, 0.0, 0.0]]), rot, 0.01, 100,
                     -10.0, 10.0, 10.0, 10.0, 0.0)[0]
    assert pts.shape == (201, 3)
    assert np.allclose(pts[100], [1.0, 0.0, 0.0], atol=1e-15, rtol=0)
    assert np.allclose(pts[200], [0.5402988597588151, 0.8414732717923807, 0.0],
                       atol=1e-10, rtol=0)
    assert np.allclose(pts[0], [0.5402988597588151, -0.8414732717923807, 0.0],
                       atol=1e-10, rtol=0)

    def piece(P):
        u = np.where((P[:, 0] < 0.3)[:, None], np.array([1.0, 0.0, 0.0]),
                     np.zeros(3))
        return u, np.ones(P.shape[0], dtype=bool)

    pts, spd = trace(np.array([[0.0, 0.0, 0.5]]), piece, 0.125, 1000,
                     0.0, 10.0, 10.0, 10.0, 1e-3)[0]
    assert np.allclose(pts[:, 0], [0.0, 0.125, 0.25], atol=1e-15, rtol=0)

    def nofl(P):
        return np.tile(np.array([1.0, 0.0, 0.0]), (P.shape[0], 1)), \
            np.zeros(P.shape[0], dtype=bool)

    assert trace(np.array([[0.5, 0.0, 0.5]]), nofl, 0.125, 1000,
                 0.0, 1.0, 10.0, 10.0, 1e-3)[0] is None
    print("[ok] T3 streamline tracer")


def _t4():
    wheels = [(1.0, 0.7, 0.4)] * 4
    seeds, groups = build_seeds(-0.3,
                                np.array([0.0, -0.9, 0.03]), np.array([0.9, 0.9, 0.33]),
                                np.array([4.6, -0.5, 0.4]), np.array([5.4, 0.5, 1.0]),
                                wheels)
    assert seeds.shape == (472, 3)
    assert np.array_equal(groups, np.array([0] * 160 + [1] * 192 + [2] * 120,
                                           dtype=np.int8))
    exp = {0: (-0.3, -0.855, 0.04875), 159: (-0.3, 0.855, 0.31125),
           160: (0.48, 0.34, 0.06), 207: (0.48, 1.06, 0.76),
           352: (4.4, 0.4, 0.43), 471: (4.4, -0.4, 0.97)}
    for j, e in exp.items():
        assert np.allclose(seeds[j], e, atol=1e-12, rtol=0), (j, seeds[j], e)
    print("[ok] T4 seed groups")


def _t5():
    mesh, owner, nb = _fixture_mesh()
    boundary = [("inlet", 1, 1), ("body", 3, 2), ("side", 6, 5)]
    p = np.array([1.0, 3.0])
    U = 2.0
    q_kin = 0.5 * U * U
    sf, xf, mag, cen, vol, h, nc = cell_geometry(mesh, owner, nb)
    p_ref = inlet_p_ref(owner, p, mag, boundary)
    assert abs(p_ref - 1.0) <= 1e-12
    cp_body = patch_face_cp(owner, p, p_ref, q_kin, 2, 3)
    assert np.allclose(cp_body, [1.0, 0.0, 1.0], atol=1e-12, rtol=0)
    sim = np.array([[2.1, 0.5, 0.5], [0.4, -0.05, 0.5], [1.6, 0.0, 0.6]])
    vals, dist = cp_nearest(sim, xf[2:5], cp_body)
    assert np.allclose(vals, [1.0, 0.0, 1.0], atol=1e-12, rtol=0)
    assert np.allclose(dist, [0.1, 0.1118033988749895, 0.1414213562373095],
                       rtol=1e-12, atol=0)
    print("[ok] T5 p_ref and Cp mapping")


def _t6():
    xs, zs = slice_axes(-1.0, 0.0, 0.05, 0.01)
    assert xs.size == 101 and abs(xs[0] + 1.0) <= 1e-12 and abs(xs[-1] - 0.0) <= 1e-12
    assert zs.size == 5
    assert np.allclose(zs, [0.005, 0.015, 0.025, 0.035, 0.045], atol=1e-12, rtol=0)
    print("[ok] T6 slice axes")


def _export_args(pe):
    pe.add_argument("--solve", required=True)
    pe.add_argument("--geom", required=True)
    pe.add_argument("--out", required=True)
    pe.add_argument("--time", type=float, default=None)
    pe.add_argument("--ds", type=float, default=0.02)
    pe.add_argument("--slice-dx", type=float, default=0.01)
    return pe


def _export_parser():
    return _export_args(argparse.ArgumentParser(prog=TOOL))


def _expect_refuse(argv, code):
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            cmd_export(_export_parser().parse_args(argv))
    except SystemExit as e:
        assert int(e.code) == 2, (code, e.code, buf.getvalue())
        assert code in buf.getvalue(), (code, buf.getvalue())
        return
    raise AssertionError("no refusal, wanted " + code)


def _npz_expect_refuse(path, kind):
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            check_npz(kind, path)
    except SystemExit as e:
        assert int(e.code) == 2, (e.code, buf.getvalue())
        assert "PP-NPZ" in buf.getvalue(), buf.getvalue()
        return
    raise AssertionError("no PP-NPZ refusal for " + path)


def _t7():
    d = tempfile.mkdtemp(prefix="pp-t7-")
    pts = np.arange(15, dtype=np.float64).reshape(5, 3)
    spd = np.linspace(1.0, 2.0, 5)
    grp = np.array([0, 1], dtype=np.int8)
    sds = np.zeros((2, 3))
    ok = os.path.join(d, "ok.npz")
    np.savez(ok, points=pts, speed=spd, offsets=np.array([0, 2, 5], dtype=np.int64),
             group=grp, seeds=sds)
    check_npz("streamlines", ok)
    bad = os.path.join(d, "bad_off.npz")
    np.savez(bad, points=pts, speed=spd,
             offsets=np.array([0, 1, 5], dtype=np.int64), group=grp, seeds=sds)
    _npz_expect_refuse(bad, "streamlines")
    bad = os.path.join(d, "bad_slice.npz")
    np.savez(bad, x=np.array([0.0, 0.5, 0.2]), z=np.array([0.0, 1.0]),
             y0=np.float64(0.0), speed=np.zeros((2, 3)))
    _npz_expect_refuse(bad, "slice")
    bad = os.path.join(d, "bad_surf.npz")
    np.savez(bad, points=pts, values=np.linspace(0.0, 1.0, 4),
             patch=np.zeros(4, dtype=np.int8), dist=np.zeros(4, dtype=np.float32))
    _npz_expect_refuse(bad, "surface")
    print("[ok] T7 npz contract check")


def _t8():
    d = tempfile.mkdtemp(prefix="pp-t8-")
    _expect_refuse(["--solve", os.path.join(d, "nope"), "--geom", d,
                    "--out", os.path.join(d, "o1")], "PP-SOLVE")
    _expect_refuse(["--solve", d, "--geom", d,
                    "--out", os.path.join(REPO, "_pp_out_probe")], "PP-OUT")
    sdir = os.path.join(d, "solve")
    write_poly_mesh(os.path.join(sdir, "case"), _tiny_points(), _tiny_faces(),
                    _tiny_owner(), [1], [("body", 2, 1), ("side", 8, 3)])
    with open(os.path.join(sdir, "solve.json"), "w", encoding="utf-8") as fh:
        json.dump({"note": "fixture"}, fh)
    with open(os.path.join(sdir, "forces.csv"), "w", encoding="utf-8") as fh:
        fh.write("time,step,Fx,Fy,Fz,Cd_p,Cl_p,D_front_N,D_rear_N,front_share\n")
    with open(os.path.join(sdir, "LICENSE.txt"), "w", encoding="utf-8") as fh:
        fh.write("fixture\n")
    snap = os.path.join(sdir, "case", "0.5")
    os.makedirs(snap, exist_ok=True)
    for f in ("U", "p"):
        with open(os.path.join(snap, f), "w", encoding="utf-8") as fh:
            fh.write("FoamFile\n{\n}\ninternalField uniform 0;\n")
    gdir = os.path.join(d, "geom")
    os.makedirs(gdir, exist_ok=True)
    with open(os.path.join(gdir, "sim_geom.json"), "w", encoding="utf-8") as fh:
        json.dump({"patches": []}, fh)
    for name in ("body", "rear_wing", "wheels", "floor"):
        with open(os.path.join(gdir, name + ".stl"), "wb") as fh:
            fh.write(b"\0" * 84)
    _expect_refuse(["--solve", sdir, "--geom", gdir,
                    "--out", os.path.join(d, "o3")], "PP-GEOM")
    print("[ok] T8 refusals")


def selftest():
    for fn in (_t1, _t2, _t3, _t4, _t5, _t6, _t7, _t8):
        fn()
    print("SELFTEST PASS 8/8")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(prog=TOOL)
    ap.add_argument("--selftest", action="store_true")
    sub = ap.add_subparsers(dest="cmd")
    _export_args(sub.add_parser("export"))
    args = ap.parse_args(argv)
    if args.selftest:
        return selftest()
    if args.cmd == "export":
        return cmd_export(args)
    ap.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
