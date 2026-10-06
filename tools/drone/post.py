#!/usr/bin/env python3
# meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
# Source-available, not Open Source. See LICENSE at the repository root.
# No GPL-licensed source was consulted.

"""post.py - turn the drone's forward-flight solve into the render inputs.

Two subcommands:

export  one solve directory (polyMesh + the requested time's U and p, the
        4 wall STLs) -> the three npz files the promo scene code already
        reads, in the promo npz contracts, plus numbers.json: surface.npz
        (Cp over the sim-surface triangle centroids, each carrying the Cp
        of the nearest wall-face centre of the same mesh patch),
        streamlines.npz (speed polylines marched both directions from the
        rotor seed rings and a rake) and slice.npz (the mid-rotor speed
        grid). Gates G-LINES, G-SLICE, G-FREESTREAM, G-CP.
walls   the review's before/after polyMesh pair -> walls_before.npz /
        walls_after.npz (the snapped wall triangles of each label, the real
        faces whose area over the STL area is exactly the review's
        area_ratio) plus walls.json. Gates W-COUNT, W-AREA, W-GROW.

The drone flow runs along -x and the march box reaches below z = 0, so the
tracer wraps the promo tracer's frame (x' = -x, z' = z - zlo) through
trace_box instead of re-writing the march: promo_post.trace is called with
the mapped seeds and a mapped velocity, and every returned point is mapped
back. Promo modules are reached only by importlib (promo_post).

The drone geometry is the BSD-3-Clause PX4 x500 model ("(c) 2022 Rudis
Laboratories / PX4 Autopilot for Drones"), so NOTHING is written inside the
repository: every output goes under --out, which must lie outside it, with
the model's LICENSE.txt, LICENSE and rotors.json copied beside them. The
flow values are demonstration numbers from an unsteady, unvalidated run; no
validated coefficients.

--selftest runs P1-P8 on the CPU.
"""

import argparse
import copy
import hashlib
import importlib.util
import json
import math
import os
import shutil
import sys
import tempfile
import time

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TOOL = "tools/drone/post.py"
VERSION = "drone-post/1"
CREDIT = "(c) 2022 Rudis Laboratories / PX4 Autopilot for Drones, BSD-3-Clause"
HONESTY = "demonstration numbers from an unsteady, unvalidated run; no validated coefficients"
WALLS = ["body", "arms", "motors", "skids"]          # patch index = position in this list
BOX = (-0.9, -0.4, -0.45, 0.5, 0.4, 0.35)            # march box xlo, ylo, zlo, xhi, yhi, zhi (m); needs ylo == -yhi
DS = 0.005                                           # march step (m)
N_MAX = 600                                          # steps per direction
U_MIN_FRAC = 1e-3                                    # stall speed / |U_inf|
RING_R = (0.5, 0.75, 0.95)                           # seed ring radii / rotor radius
RING_N = 12                                          # seeds per ring
RING_DZ = 0.02                                       # seeds this far above the disk centre (m)
RAKE_X = 0.45
RAKE_Y = (-0.12, 0.12, 9)                            # np.linspace(lo, hi, n)
RAKE_Z = (-0.22, 0.04, 6)
SLICE_DX = 0.004
SLICE_X = (-0.7, 0.35)                               # the drone and its near wake; the freestream column i = nx-1 (x = 0.35) is still upstream of the drone's x <= 0.21
SLICE_Z = (-0.28, 0.18)                              # 116 x 263 grid
CP_RANGE = (-0.8, 1.0)                               # display range for the bars (data p1/p99 = -0.72/0.92)
SPEED_RANGE = (5.0, 18.0)                            # m/s, display range (slice p1/p99 = 6.4/16.8)
LINES_MIN_FRAC = 0.9
SLICE_FLUID_MIN = 0.95
FREESTREAM_TOL = 0.05
CP_DIST_MAX = 0.05                                   # m, max sim-point to wall-face-centre distance
AREA_TOL = 1e-9
P_REF_RULE = "area-weighted mean of the owner-cell kinematic p over the inlet patch"
POLY_FILES = ("boundary", "faces", "owner", "neighbour", "points")


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _load_post():
    here = os.path.dirname(os.path.abspath(__file__))
    return _load("promo_post", os.path.normpath(os.path.join(here, "..", "promo", "post.py")))


PROMO_POST = _load_post()


def refuse(code, text):
    print("refused: " + code + ": " + text, flush=True)
    raise SystemExit(2)


def _posix(path):
    return str(path).replace(os.sep, "/")


def prog(stage, t0, timings):
    secs = time.time() - t0
    timings[stage] = float(secs)
    print("[drone-post] " + stage + " %.1f s" % secs, flush=True)


def _sha256(path):
    hh = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            hh.update(chunk)
    return hh.hexdigest()


def _file_entry(path):
    return {"bytes": int(os.path.getsize(path)), "sha256": _sha256(path)}


# ---------------------------------------------------------------- pure core

def trace_box(seeds, vel, ds, n_max, box, u_min):
    """Per seed None or (points (m,3), speed (m,)) in the ORIGINAL frame.

    The promo tracer marches x' = -x with z' measured from the box floor,
    so every query and every returned point goes through the frame map
    (x, z) -> (-x, z - zlo): the mapped velocity is (U * F, fluid) of
    vel(P) where P = P' * F with z = z' + zlo, and the march box in the
    promo frame is x_up = -xhi, x_end = -xlo, |y| <= yhi, 0 <= z' <=
    zhi - zlo. The line order stays promo's (upstream end first).
    DP-BOX unless ylo == -yhi and lo < hi on every axis."""
    xlo, ylo, zlo, xhi, yhi, zhi = (float(v) for v in box)
    if not (ylo == -yhi and xlo < xhi and ylo < yhi and zlo < zhi):
        refuse("DP-BOX", "box needs ylo == -yhi and lo < hi on every axis: " + str(tuple(box)))
    F = np.array([-1.0, 1.0, 1.0])
    seeds = np.asarray(seeds, dtype=np.float64).reshape(-1, 3)
    s2 = seeds * F
    s2[:, 2] -= zlo

    def vel2(P):
        Q = np.asarray(P, dtype=np.float64) * F
        Q[:, 2] += zlo
        U, fl = vel(Q)
        return U * F, fl

    out = PROMO_POST.trace(s2, vel2, ds, n_max, x_up=-xhi, x_end=-xlo,
                           y_half=yhi, z_top=zhi - zlo, u_min=u_min)
    for line in out:
        if line is None:
            continue
        pts = line[0]
        pts[:, 0] = -pts[:, 0]
        pts[:, 2] += zlo
    return out


def rotor_seeds(rotors):
    """(seeds (4*36, 3) float64, rotor (n,) int8): for rotor i (list order),
    ring j (RING_R[j]) and k in range(RING_N), row (i*len(RING_R) + j)*RING_N + k
    is centre + (f*R cos a, f*R sin a, RING_DZ), a = 2 pi k / RING_N."""
    rows = []
    which = []
    for i, rot in enumerate(rotors):
        c = np.asarray(rot["centre_m"], dtype=np.float64)
        rad = float(rot["radius_m"])
        for frac in RING_R:
            for k in range(RING_N):
                a = 2.0 * math.pi * k / RING_N
                rows.append(c + np.array([frac * rad * math.cos(a),
                                          frac * rad * math.sin(a),
                                          RING_DZ]))
                which.append(i)
    return (np.array(rows, dtype=np.float64),
            np.array(which, dtype=np.int8))


def rake_seeds():
    """(54, 3): (RAKE_X, y, z) for y in linspace(*RAKE_Y) (outer) for
    z in linspace(*RAKE_Z) (inner)."""
    ys = np.linspace(*RAKE_Y)
    zs = np.linspace(*RAKE_Z)
    rows = [[float(RAKE_X), float(y), float(z)] for y in ys for z in zs]
    return np.array(rows, dtype=np.float64)


def fan_tris(arity, ptr, flat, start, count):
    """(t, 3) int64 GLOBAL point indices: face f -> (v0, v_i, v_i+1) for
    i = 1 .. arity-2, faces in order."""
    ar = np.asarray(arity[start:start + count], dtype=np.int64)
    per = np.maximum(ar - 2, 0)
    total = int(per.sum())
    if total == 0:
        return np.zeros((0, 3), dtype=np.int64)
    face_id = np.repeat(np.arange(count, dtype=np.int64), per)
    first = np.repeat(np.cumsum(per) - per, per)
    q = np.arange(total, dtype=np.int64) - first + 1
    fstart = np.asarray(ptr[start:start + count], dtype=np.int64)[face_id]
    v0 = np.asarray(flat)[fstart]
    vi = np.asarray(flat)[fstart + q]
    vj = np.asarray(flat)[fstart + q + 1]
    return np.stack([v0, vi, vj], axis=1).astype(np.int64)


def wall_patch(mesh, start, count):
    """(V (n,3) float64, F (t,3) int64, area float): the used points only,
    renumbered in ascending global index order (np.unique), F = fan_tris
    remapped, area = the sum of |Sf| from mesh.face_sf_centres."""
    fan = fan_tris(mesh.arity, mesh.ptr, mesh.flat, start, count)
    uniq, inv = np.unique(fan.reshape(-1), return_inverse=True)
    V = mesh.points[uniq]
    F = inv.reshape(-1, 3).astype(np.int64)
    sf, _c = mesh.face_sf_centres(start, count)
    area = float(np.linalg.norm(sf, axis=1).sum())
    return V, F, area


def judge_export(ex):
    """G-LINES/G-SLICE/G-FREESTREAM/G-CP over numbers.json without gate."""
    sl = ex.get("streamlines") or {}
    ns = sl.get("n_seeds") or {}
    nk = sl.get("n_kept") or {}

    def lines_ok(group):
        want = ns.get(group)
        got = nk.get(group)
        return (isinstance(want, (int, float)) and isinstance(got, (int, float))
                and got >= LINES_MIN_FRAC * want)

    sz = ex.get("slice") or {}
    fs = sz.get("freestream") or {}
    cp = ex.get("cp") or {}
    checks = [
        ("G-LINES", lines_ok("rotor") and lines_ok("rake")),
        ("G-SLICE", (sz.get("fluid_frac") is not None
                     and sz["fluid_frac"] >= SLICE_FLUID_MIN)),
        ("G-FREESTREAM", (fs.get("rel_err") is not None
                          and fs["rel_err"] <= FREESTREAM_TOL)),
        ("G-CP", (cp.get("dist_max") is not None
                  and cp["dist_max"] <= CP_DIST_MAX)),
    ]
    failed = [cid for cid, ok in checks if not ok]
    return {"pass": not failed, "failed": failed}


def judge_walls(wj):
    """W-COUNT/W-AREA/W-GROW over walls.json without gate."""
    count_ok = True
    area_ok = True
    for label in ("before", "after"):
        blk = wj.get(label) or {}
        nf = blk.get("n_faces") or {}
        if not all((nf.get(w) or 0) > 0 for w in WALLS):
            count_ok = False
        if not (blk.get("n_tris") or 0) > 0:
            count_ok = False
        if not (blk.get("max_abs_diff") is not None
                and blk["max_abs_diff"] <= AREA_TOL):
            area_ok = False
    target = wj.get("target") or {}
    metric = target.get("metric") or ""
    grow_ok = False
    if metric.startswith("area_ratio."):
        w = metric.split(".", 1)[1]
        b = (wj.get("before") or {}).get("area_ratio") or {}
        a = (wj.get("after") or {}).get("area_ratio") or {}
        grow_ok = (b.get(w) is not None and a.get(w) is not None
                   and a[w] > b[w])
    checks = [("W-COUNT", count_ok), ("W-AREA", area_ok), ("W-GROW", grow_ok)]
    failed = [cid for cid, ok in checks if not ok]
    return {"pass": not failed, "failed": failed}


# ---------------------------------------------------------------- checks

def _check_out(out):
    """DP-OUT: outside the repository (the drone geometry and everything
    made from it is never written inside it)."""
    try:
        rel = os.path.relpath(out, REPO)
    except ValueError:
        rel = ".."
    if rel != ".." and not rel.startswith(".." + os.sep):
        refuse("DP-OUT", _posix(out) + ": --out resolves inside the repository")


def _need_input(path, what=""):
    if not os.path.exists(path):
        refuse("DP-INPUT", _posix(path) + ": missing " + what)


def _check_poly_dir(pm_dir):
    missing = [f for f in POLY_FILES if not os.path.isfile(os.path.join(pm_dir, f))]
    if missing:
        refuse("DP-INPUT", _posix(pm_dir) + ": missing " + ", ".join(missing))


def _check_walls_stls(sim):
    for w in WALLS:
        _need_input(os.path.join(sim, w + ".stl"), "--sim/" + w + ".stl")


def _pick_time(solve, time_sel):
    case = os.path.join(solve, "case")
    if time_sel == "latest":
        names = []
        for nm in os.listdir(case):
            p = os.path.join(case, nm)
            if not os.path.isdir(p):
                continue
            try:
                fv = float(nm)
            except ValueError:
                continue
            if fv != 0.0:
                names.append((fv, nm))
        if not names:
            refuse("DP-INPUT", _posix(case) + ": no time directories")
        time_sel = max(names)[1]
    tdir = os.path.join(case, time_sel)
    if not os.path.isdir(tdir):
        refuse("DP-INPUT", _posix(tdir) + ": no such time directory")
    for f in ("U", "p"):
        _need_input(os.path.join(tdir, f), "time dir/" + f)
    return tdir, time_sel


def _patch_range(boundary, name):
    for b in boundary:
        if b[0] == name:
            return int(b[1]), int(b[2])
    refuse("DP-INPUT", "boundary has no " + name + " patch")


# ---------------------------------------------------------------- export

def cmd_export(args):
    t_all = time.time()
    timings = {}
    _check_out(args.out)
    t0 = time.time()
    _need_input(args.solve, "--solve (not a directory)" if not os.path.isdir(args.solve) else "--solve")
    _need_input(os.path.join(args.solve, "write.json"), "write.json")
    _need_input(os.path.join(args.solve, "rotors.json"), "rotors.json")
    pm_dir = os.path.join(args.solve, "case", "constant", "polyMesh")
    _check_poly_dir(pm_dir)
    tdir, time_name = _pick_time(args.solve, args.time)
    _check_walls_stls(args.sim)
    prog("inputs", t0, timings)
    os.makedirs(args.out, exist_ok=True)

    t0 = time.time()
    boundary = PROMO_POST.read_boundary(os.path.join(pm_dir, "boundary"))
    points = PROMO_POST.read_points(os.path.join(pm_dir, "points"))
    owner = PROMO_POST.read_ints(os.path.join(pm_dir, "owner"))
    neighbour = PROMO_POST.read_ints(os.path.join(pm_dir, "neighbour"))
    arity, ptr, flat = PROMO_POST.read_faces(os.path.join(pm_dir, "faces"))
    mesh = PROMO_POST.PolyMesh(points, arity, ptr, flat)
    Sf, xf, mag_sf, cen, vol, h, nc = PROMO_POST.cell_geometry(mesh, owner, neighbour)
    prog("mesh " + str(int(nc)) + " cells", t0, timings)

    t0 = time.time()
    U = PROMO_POST.read_foam_vector(os.path.join(tdir, "U"), nc)
    p = PROMO_POST.read_foam_scalar(os.path.join(tdir, "p"), nc)
    with open(os.path.join(args.solve, "write.json"), "r", encoding="utf-8") as f:
        write_json = json.load(f)
    U_inf = np.asarray(write_json["params"]["U_vec"], dtype=np.float64)
    u_mag = float(np.linalg.norm(U_inf))
    q_kin = 0.5 * u_mag * u_mag
    p_ref = PROMO_POST.inlet_p_ref(owner, p, mag_sf, boundary)
    prog("fields", t0, timings)

    t0 = time.time()
    surf_pts = []
    surf_val = []
    surf_patch = []
    surf_dist = []
    per_patch = {}
    for pi, w in enumerate(WALLS):
        nf, st = _patch_range(boundary, w)
        tris = PROMO_POST.read_stl_bin(os.path.join(args.sim, w + ".stl"))
        sim_pts = np.ascontiguousarray(tris.mean(axis=1))
        face_centres = mesh.face_centres(st, nf)
        face_cp = PROMO_POST.patch_face_cp(owner, p, p_ref, q_kin, st, nf)
        vals, dist = PROMO_POST.cp_nearest(sim_pts, face_centres, face_cp)
        surf_pts.append(sim_pts)
        surf_val.append(vals)
        surf_patch.append(np.full(sim_pts.shape[0], pi, dtype=np.int8))
        surf_dist.append(dist)
        per_patch[w] = {"n": int(sim_pts.shape[0]), "min": float(vals.min()),
                        "max": float(vals.max()), "dist_mean": float(dist.mean()),
                        "dist_max": float(dist.max())}
    spts = np.concatenate(surf_pts).astype(np.float64)
    svals = np.concatenate(surf_val).astype(np.float64)
    spatch = np.concatenate(surf_patch).astype(np.int8)
    sdist = np.concatenate(surf_dist).astype(np.float32)
    surface_path = os.path.join(args.out, "surface.npz")
    np.savez(surface_path, points=spts, values=svals, patch=spatch, dist=sdist)
    prog("surface " + str(int(spts.shape[0])) + " points", t0, timings)

    t0 = time.time()
    vel = PROMO_POST.make_vel(cen, h, U, nc)
    with open(os.path.join(args.solve, "rotors.json"), "r", encoding="utf-8") as f:
        rotors = json.load(f)["rotors"]
    r_seeds, r_rot = rotor_seeds(rotors)
    rk = rake_seeds()
    all_seeds = np.concatenate([r_seeds, rk]).astype(np.float64)
    group_all = np.concatenate([np.zeros(r_seeds.shape[0], np.int8),
                                np.ones(rk.shape[0], np.int8)])
    rotor_all = np.concatenate([r_rot, np.full(rk.shape[0], -1, np.int8)])
    lines = trace_box(all_seeds, vel, DS, N_MAX, BOX, U_MIN_FRAC * u_mag)
    kept_pts = []
    kept_spd = []
    offsets = [0]
    grp = []
    sd = []
    rt = []
    n_seeds = {"rotor": int(r_seeds.shape[0]), "rake": int(rk.shape[0])}
    n_kept = {"rotor": 0, "rake": 0}
    for j, line in enumerate(lines):
        if line is None:
            continue
        pts, spd = line
        if pts.shape[0] < 2:
            continue
        kept_pts.append(pts)
        kept_spd.append(spd)
        offsets.append(offsets[-1] + int(pts.shape[0]))
        grp.append(int(group_all[j]))
        sd.append(all_seeds[j])
        rt.append(int(rotor_all[j]))
        n_kept["rotor" if group_all[j] == 0 else "rake"] += 1
    sl_points = (np.concatenate(kept_pts) if kept_pts else np.zeros((0, 3))).astype(np.float64)
    sl_speed = (np.concatenate(kept_spd) if kept_spd else np.zeros(0)).astype(np.float64)
    stream_path = os.path.join(args.out, "streamlines.npz")
    np.savez(stream_path, points=sl_points, speed=sl_speed,
             offsets=np.asarray(offsets, dtype=np.int64),
             group=np.asarray(grp, dtype=np.int8),
             seeds=np.asarray(sd, dtype=np.float64),
             rotor=np.asarray(rt, dtype=np.int8))
    prog("streamlines " + str(len(offsets) - 1) + " lines", t0, timings)

    t0 = time.time()
    y0 = float(rotors[0]["centre_m"][1])
    nx = int(math.floor((SLICE_X[1] - SLICE_X[0]) / SLICE_DX + 1e-9)) + 1
    nz = int(math.floor((SLICE_Z[1] - SLICE_Z[0]) / SLICE_DX + 1e-9)) + 1
    xg = SLICE_X[0] + SLICE_DX * np.arange(nx, dtype=np.float64)
    zg = SLICE_Z[0] + SLICE_DX * np.arange(nz, dtype=np.float64)
    spd = PROMO_POST.slice_speed(vel, xg, zg, y0)
    slice_path = os.path.join(args.out, "slice.npz")
    np.savez(slice_path, x=xg, z=zg, y0=np.float64(y0), speed=spd.astype(np.float64))
    prog("slice " + str(nz) + "x" + str(nx), t0, timings)

    t0 = time.time()
    PROMO_POST.check_npz("surface", surface_path)
    PROMO_POST.check_npz("streamlines", stream_path)
    PROMO_POST.check_npz("slice", slice_path)
    for name in ("LICENSE.txt", "LICENSE", "rotors.json"):
        src = os.path.join(args.solve, name)
        if os.path.isfile(src):
            shutil.copyfile(src, os.path.join(args.out, name))
    prog("npz contract + licence copies", t0, timings)

    t0 = time.time()
    finite = spd[np.isfinite(spd)]
    col = spd[:, nx - 1]
    col_fin = col[np.isfinite(col)]
    median = float(np.median(col_fin)) if col_fin.size else float("nan")
    rel_err = abs(median - u_mag) / u_mag
    ex = {
        "tool": TOOL,
        "version": VERSION,
        "credit": CREDIT,
        "solve_dir": _posix(args.solve),
        "time": time_name,
        "U_inf": [float(v) for v in U_inf],
        "q_kin": q_kin,
        "p_ref": p_ref,
        "p_ref_rule": P_REF_RULE,
        "cp": {
            "min": float(svals.min()), "max": float(svals.max()),
            "p1": float(np.percentile(svals, 1)), "p99": float(np.percentile(svals, 99)),
            "per_patch": per_patch,
            "dist_max": float(sdist.max()),
        },
        "display": {"cp_range": [float(CP_RANGE[0]), float(CP_RANGE[1])],
                    "speed_range": [float(SPEED_RANGE[0]), float(SPEED_RANGE[1])]},
        "streamlines": {"n_seeds": n_seeds, "n_kept": n_kept,
                        "n_points": int(sl_points.shape[0])},
        "slice": {
            "y0": y0, "nx": int(nx), "nz": int(nz),
            "fluid_frac": float(finite.size / float(spd.size)),
            "p1": float(np.percentile(finite, 1)), "p99": float(np.percentile(finite, 99)),
            "max": float(finite.max()),
            "freestream": {
                "column": "i = nx-1 (x = SLICE_X[1], upstream)",
                "median": median,
                "rel_err": float(rel_err),
            },
        },
        "files": {os.path.basename(q): _file_entry(q)
                  for q in (surface_path, stream_path, slice_path)},
        "timing_s": timings,
        "honesty": HONESTY,
    }
    ex["timing_s"]["total"] = float(time.time() - t_all)
    gate = judge_export(ex)
    ex["gate"] = gate
    with open(os.path.join(args.out, "numbers.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(ex, f, indent=1)
    prog("numbers.json", t0, timings)
    if gate["pass"]:
        print("DRONE-POST GATE PASS", flush=True)
        return 0
    print("DRONE-POST GATE FAIL " + ",".join(gate["failed"]), flush=True)
    return 1


# ---------------------------------------------------------------- walls

def _stl_area(path):
    tris = PROMO_POST.read_stl_bin(path)
    n = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    return float(0.5 * np.linalg.norm(n, axis=1).sum())


def cmd_walls(args):
    t_all = time.time()
    timings = {}
    _check_out(args.out)
    t0 = time.time()
    _need_input(args.review, "--review (not a directory)" if not os.path.isdir(args.review) else "--review")
    ba_path = os.path.join(args.review, "before_after.json")
    _need_input(ba_path, "before_after.json")
    with open(ba_path, "r", encoding="utf-8") as f:
        ba = json.load(f)
    mesh_dirs = {}
    for label in ("before", "after"):
        md = (ba.get(label) or {}).get("mesh_dir")
        if not md:
            refuse("DP-INPUT", "before_after.json: " + label + " has no mesh_dir")
        pm = os.path.join(md, "case", "constant", "polyMesh")
        _check_poly_dir(pm)
        mesh_dirs[label] = (md, pm)
    _check_walls_stls(args.sim)
    prog("inputs", t0, timings)
    os.makedirs(args.out, exist_ok=True)

    blocks = {}
    files = {}
    for label in ("before", "after"):
        t0 = time.time()
        md, pm = mesh_dirs[label]
        points = PROMO_POST.read_points(os.path.join(pm, "points"))
        boundary = PROMO_POST.read_boundary(os.path.join(pm, "boundary"))
        arity, ptr, flat = PROMO_POST.read_faces(os.path.join(pm, "faces"))
        mesh = PROMO_POST.PolyMesh(points, arity, ptr, flat)
        Vs = []
        Fs = []
        patches = []
        voff = 0
        n_faces = {}
        n_tris = 0
        area_ratio = {}
        for pi, w in enumerate(WALLS):
            nf, st = _patch_range(boundary, w)
            V, F, area = wall_patch(mesh, st, nf)
            Vs.append(V.astype(np.float64))
            Fs.append((F + voff).astype(np.int64))
            patches.append(np.full(F.shape[0], pi, dtype=np.int8))
            voff += int(V.shape[0])
            n_faces[w] = int(nf)
            n_tris += int(F.shape[0])
            area_ratio[w] = area / _stl_area(os.path.join(args.sim, w + ".stl"))
        npz_path = os.path.join(args.out, "walls_" + label + ".npz")
        np.savez(npz_path, V=np.concatenate(Vs).astype(np.float64),
                 F=np.concatenate(Fs).astype(np.int64),
                 patch=np.concatenate(patches).astype(np.int8))
        files[os.path.basename(npz_path)] = _file_entry(npz_path)
        review_ar = {}
        for row in ba.get("rows", []):
            m = row.get("metric", "")
            if m.startswith("area_ratio."):
                review_ar[m.split(".", 1)[1]] = row.get(label)
        review_ar = {w: review_ar[w] for w in WALLS if w in review_ar}
        diffs = [abs(area_ratio[w] - review_ar[w]) for w in WALLS if w in review_ar]
        blocks[label] = {
            "mesh_dir": _posix(md),
            "n_faces": n_faces,
            "n_tris": int(n_tris),
            "area_ratio": {w: float(area_ratio[w]) for w in WALLS},
            "area_ratio_review": {w: (float(review_ar[w]) if w in review_ar else None)
                                  for w in WALLS},
            "max_abs_diff": float(max(diffs)) if len(diffs) == len(WALLS) else None,
        }
        prog(label + " walls " + str(n_tris) + " tris", t0, timings)

    report = {
        "tool": TOOL,
        "version": VERSION,
        "credit": CREDIT,
        "review": _posix(args.review),
        "remedy_say": (ba.get("remedy") or {}).get("say"),
        "verdict": ba.get("verdict"),
        "target": ba.get("target"),
        "before": blocks["before"],
        "after": blocks["after"],
        "files": files,
        "honesty": HONESTY,
    }
    gate = judge_walls(report)
    report["gate"] = gate
    with open(os.path.join(args.out, "walls.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(report, f, indent=1)
    report["timing_s"] = dict(timings)
    report["timing_s"]["total"] = float(time.time() - t_all)
    if gate["pass"]:
        print("DRONE-WALLS GATE PASS", flush=True)
        return 0
    print("DRONE-WALLS GATE FAIL " + ",".join(gate["failed"]), flush=True)
    return 1


# ---------------------------------------------------------------- selftest

def _vel_const(vec):
    v = np.asarray(vec, dtype=np.float64)

    def vel(P):
        P = np.asarray(P, dtype=np.float64)
        return np.tile(v, (P.shape[0], 1)), np.ones(P.shape[0], dtype=bool)
    return vel


def _t1():
    lines = trace_box(np.array([[0.0, 0.0, 0.0]]), _vel_const((-15.0, 0.0, 0.0)),
                      0.1, 100, (-1.05, -0.5, -0.5, 1.05, 0.5, 0.5), 0.015)
    assert len(lines) == 1 and lines[0] is not None, "expected one kept line"
    pts, spd = lines[0]
    assert pts.shape == (21, 3), "shape " + str(pts.shape)
    assert np.allclose(pts[0], (1.0, 0.0, 0.0), atol=1e-9), str(pts[0])
    assert np.allclose(pts[-1], (-1.0, 0.0, 0.0), atol=1e-9), str(pts[-1])
    assert np.allclose(spd, 15.0, atol=1e-12), "speeds"
    assert float(np.max(np.abs(pts[:, 1]))) <= 1e-12
    assert float(np.max(np.abs(pts[:, 2]))) <= 1e-12


def _t2():
    lines = trace_box(np.zeros((1, 3)), _vel_const((0.0, 0.0, -5.0)),
                      0.1, 100, (-0.5, -0.5, -0.55, 0.5, 0.5, 0.55), 1e-9)
    pts, _spd = lines[0]
    assert pts.shape == (11, 3), "shape " + str(pts.shape)
    assert np.allclose(pts[0], (0.0, 0.0, 0.5), atol=1e-9), str(pts[0])
    assert np.allclose(pts[-1], (0.0, 0.0, -0.5), atol=1e-9), str(pts[-1])
    try:
        trace_box(np.zeros((1, 3)), _vel_const((0.0, 0.0, -5.0)),
                  0.1, 10, (-1.0, -0.4, -1.0, 1.0, 0.5, 1.0), 1e-9)
        raise AssertionError("DP-BOX not raised")
    except SystemExit as e:
        assert e.code == 2, str(e.code)


def _t3():
    rot = [{"centre_m": [0.174, -0.174, 0.051162], "radius_m": 0.146769}]
    seeds, rotor = rotor_seeds(rot)
    assert seeds.shape == (36, 3), str(seeds.shape)
    assert rotor.dtype == np.int8 and bool((rotor == 0).all())
    assert np.allclose(seeds[0], (0.174 + 0.5 * 0.146769, -0.174, 0.071162), atol=1e-12), str(seeds[0])
    assert np.allclose(seeds[15], (0.174, -0.174 + 0.75 * 0.146769, 0.071162), atol=1e-12), str(seeds[15])
    seeds4, rotor4 = rotor_seeds(rot * 4)
    assert seeds4.shape == (144, 3), str(seeds4.shape)
    assert np.allclose(seeds4[36], seeds[0], atol=1e-12)
    assert bool((rotor4[:36] == 0).all()) and bool((rotor4[36:72] == 1).all())


def _t4():
    rk = rake_seeds()
    assert rk.shape == (54, 3), str(rk.shape)
    assert np.allclose(rk[0], (0.45, -0.12, -0.22), atol=1e-12), str(rk[0])
    assert np.allclose(rk[5], (0.45, -0.12, 0.04), atol=1e-12), str(rk[5])
    assert np.allclose(rk[6], (0.45, -0.09, -0.22), atol=1e-12), str(rk[6])


def _t5():
    arity = np.array([4, 3, 5], np.int64)
    ptr = np.array([0, 4, 7, 12], np.int64)
    flat = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], np.int64)
    F = fan_tris(arity, ptr, flat, 0, 3)
    assert np.array_equal(F, [[0, 1, 2], [0, 2, 3], [4, 5, 6],
                              [7, 8, 9], [7, 9, 10], [7, 10, 11]]), str(F.tolist())
    F1 = fan_tris(arity, ptr, flat, 1, 1)
    assert np.array_equal(F1, [[4, 5, 6]]), str(F1.tolist())


def _t6():
    pts = np.array([(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0), (2, 0, 0), (9, 9, 9)], float)
    m = PROMO_POST.PolyMesh(pts, np.array([4, 3], np.int64), np.array([0, 4, 7], np.int64),
                            np.array([0, 1, 2, 3, 1, 4, 2], np.int64))
    V, F, area = wall_patch(m, 0, 2)
    assert np.allclose(V, pts[:5]), str(V.tolist())
    assert np.array_equal(F, [[0, 1, 2], [0, 2, 3], [1, 4, 2]]), str(F.tolist())
    assert abs(area - 1.5) <= 1e-12, str(area)
    V, F, area = wall_patch(m, 1, 1)
    assert np.allclose(V, [(1, 0, 0), (1, 1, 0), (2, 0, 0)]), str(V.tolist())
    assert np.array_equal(F, [[0, 2, 1]]), str(F.tolist())
    assert abs(area - 0.5) <= 1e-12, str(area)


def _passing_export():
    return {
        "streamlines": {"n_seeds": {"rotor": 144, "rake": 54},
                        "n_kept": {"rotor": 144, "rake": 54},
                        "n_points": 100},
        "slice": {"fluid_frac": 0.999, "freestream": {"rel_err": 0.005}},
        "cp": {"dist_max": 0.028},
    }


def _passing_walls():
    def blk(ratio_arms):
        return {"mesh_dir": "m", "n_faces": {w: 10 for w in WALLS}, "n_tris": 100,
                "area_ratio": {w: 0.5 for w in WALLS},
                "area_ratio_review": {w: 0.5 for w in WALLS},
                "max_abs_diff": 0.0} | {"area_ratio": {**{w: 0.5 for w in WALLS}, "arms": ratio_arms}}
    return {
        "before": blk(0.5),
        "after": blk(0.6),
        "target": {"metric": "area_ratio.arms", "before": 0.5, "after": 0.6, "improved": True},
    }


def _t7():
    base = _passing_export()
    good = judge_export(base)
    assert good == {"pass": True, "failed": []}, str(good)
    breaks = {
        "G-LINES": lambda d: d["streamlines"]["n_kept"].__setitem__("rotor", 0),
        "G-SLICE": lambda d: d["slice"].__setitem__("fluid_frac", 0.5),
        "G-FREESTREAM": lambda d: d["slice"]["freestream"].__setitem__("rel_err", 0.5),
        "G-CP": lambda d: d["cp"].__setitem__("dist_max", 0.5),
    }
    assert list(breaks) == ["G-LINES", "G-SLICE", "G-FREESTREAM", "G-CP"]
    for gid, brk in breaks.items():
        d = copy.deepcopy(base)
        brk(d)
        r = judge_export(d)
        assert r == {"pass": False, "failed": [gid]}, gid + ": " + str(r)
    base_w = _passing_walls()
    good = judge_walls(base_w)
    assert good == {"pass": True, "failed": []}, str(good)
    breaks_w = {
        "W-COUNT": lambda d: d["after"]["n_faces"].__setitem__("arms", 0),
        "W-AREA": lambda d: d["before"].__setitem__("max_abs_diff", 1e-3),
        "W-GROW": lambda d: d["after"]["area_ratio"].__setitem__("arms", 0.4),
    }
    assert list(breaks_w) == ["W-COUNT", "W-AREA", "W-GROW"]
    for wid, brk in breaks_w.items():
        d = copy.deepcopy(base_w)
        brk(d)
        r = judge_walls(d)
        assert r == {"pass": False, "failed": [wid]}, wid + ": " + str(r)


def _t8():
    tmp = tempfile.mkdtemp(prefix="drone-post-t8-")
    try:
        out_in = os.path.normpath(os.path.join(REPO, "tools", "drone", "_x"))
        try:
            main(["export", "--solve", tmp, "--sim", tmp, "--out", out_in])
            raise AssertionError("DP-OUT not raised")
        except SystemExit as e:
            assert e.code == 2, str(e.code)
        assert not os.path.exists(out_in) or not os.listdir(out_in), "DP-OUT wrote something"
        out2 = os.path.join(tmp, "o2")
        try:
            main(["export", "--solve", os.path.join(tmp, "missing"), "--sim", tmp, "--out", out2])
            raise AssertionError("DP-INPUT not raised")
        except SystemExit as e:
            assert e.code == 2, str(e.code)
        assert not os.path.exists(out2) or not os.listdir(out2)
        out3 = os.path.join(tmp, "o3")
        try:
            main(["walls", "--review", tmp, "--sim", tmp, "--out", out3])
            raise AssertionError("DP-INPUT not raised")
        except SystemExit as e:
            assert e.code == 2, str(e.code)
        assert not os.path.exists(out3) or not os.listdir(out3)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def selftest():
    """R11: P1-P8, CPU only; temp files only under tempfile.mkdtemp()."""
    tests = [("P1", _t1), ("P2", _t2), ("P3", _t3), ("P4", _t4),
             ("P5", _t5), ("P6", _t6), ("P7", _t7), ("P8", _t8)]
    npass = 0
    for name, fn in tests:
        try:
            fn()
            print("[ok] " + name, flush=True)
            npass += 1
        except Exception as exc:
            print("[FAIL] " + name + ": " + str(exc), flush=True)
    if npass == len(tests):
        print("SELFTEST PASS " + str(npass) + "/" + str(len(tests)), flush=True)
        return 0
    print("SELFTEST FAIL " + str(npass) + "/" + str(len(tests)), flush=True)
    return 1


def main(argv=None):
    ap = argparse.ArgumentParser(prog="post.py", description=__doc__.splitlines()[0])
    ap.add_argument("--selftest", action="store_true")
    sub = ap.add_subparsers(dest="cmd")
    ep = sub.add_parser("export")
    ep.add_argument("--solve", required=True)
    ep.add_argument("--sim", required=True)
    ep.add_argument("--out", required=True)
    ep.add_argument("--time", default="latest")
    wp = sub.add_parser("walls")
    wp.add_argument("--review", required=True)
    wp.add_argument("--sim", required=True)
    wp.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    if args.selftest:
        return selftest()
    if args.cmd == "export":
        return cmd_export(args)
    if args.cmd == "walls":
        return cmd_walls(args)
    ap.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
