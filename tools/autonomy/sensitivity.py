#!/usr/bin/env python3
# meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
# Source-available, not Open Source. See LICENSE at the repository root.
# No GPL-licensed source was consulted.
"""sensitivity.py - docs/15 §F's G-PILOT: the one-at-a-time knob sweep that
decides, before any optimiser code exists, whether tuning can unpin the snap.

It builds the 12 tuning geometries (five family-A wings, five family-B lathes
at seed 1, and two boxes - one on the octree lattice, one off it), derives the
L4 template P0 from each geometry's own l_ref, builds 24 whitelisted jobs per
geometry plus the 3-run 8-layer commensurate-cube check at the R-WIN t1, runs
them through ofgpu-automesher (at most 6 concurrent processes), scores every
run with score.py, and computes the PASS/KILL verdict from the rows alone.

    python tools/autonomy/sensitivity.py --selftest
    python tools/autonomy/sensitivity.py --list
    python tools/autonomy/sensitivity.py --pilot --out DIR --work DIR [--jobs N]
                                         [--timeout S] [--only ID,ID]
                                         [--groups g,g] [--binary P]
    python tools/autonomy/sensitivity.py --report DIR
"""

from __future__ import annotations

import argparse
import concurrent.futures
import copy
import hashlib
import json
import math
import os
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
import threading
import time

import numpy as np
import psutil

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, "corpus"))

import schema  # noqa: E402
import score  # noqa: E402
import stl_io  # noqa: E402
import gen_wing  # noqa: E402
import gen_lathe  # noqa: E402

REPO = score.REPO
BINARY_DEFAULT = score.BINARY_DEFAULT
MAX_JOBS = 6                      # the house cap on concurrent automesher processes
NU = 1.5e-5                       # m^2/s, the corpus viscosity (gen_wing.NU)
WALL_LEVEL = 4                    # the pilot template's wall level (docs/15 §B's L4 probes)
FEATURE_ANGLE_DEG = 30.0          # refinement.feature_angle_deg default, mod.rs d_feature_angle
CELL_FRAC = 0.5                   # layers.cell_frac default (mod.rs d_cell_frac); NOT a knob (§I-5)
G5_RATIO = 0.05                   # quality.min_thickness_ratio default (92.51); NOT a knob
DEFAULTS = {"/snap/iterations": 30, "/snap/tolerance": 1e-3, "/snap/smoothing_passes": 3,
            "/snap/smoothing": 0.5, "/snap/undo_limit": 4, "/snap/feature_tolerance": 0.5,
            "/layers/growth": 1.3, "/layers/normal_passes": 3}   # mod.rs 273-299, 363-377
ROW_SCHEMA = "autonomy-pilot/1"
RUN_KEY_RE = re.compile(r"^[A-Za-z0-9._-]{1,64}$")
L4_GROUPS = ("baseline", "l4", "layers")
L4_KNOBS = ("wall_level", "band_distance", "feature_level", "feature_tolerance",
            "snap_smoothing_passes", "growth")
EXT_KNOBS = ("iterations", "tolerance", "snap_smoothing", "undo_limit", "feature_off")


# --- the geometries (C2) -----------------------------------------------------

def box_mesh(lo, hi, nd):
    """An axis-aligned box triangulated on a nd-per-axis grid.

    Every corner coordinate is taken from one per-axis array (so a point
    shared by two faces is bit-identical), each quad is two triangles wound
    outward, and the result is deduped with np.unique and refused, never
    repaired, when stl_io.check_closed does not accept it.
    """
    lo = [float(v) for v in lo]
    hi = [float(v) for v in hi]
    ax = []
    for a in range(3):
        c = lo[a] + (hi[a] - lo[a]) * np.arange(nd + 1) / nd
        c[0] = lo[a]
        c[-1] = hi[a]
        ax.append(c)
    # (axis, side 0=lo 1=hi, u_axis, v_axis) with cross(u_hat, v_hat) outward
    faces = ((0, 0, 2, 1), (0, 1, 1, 2), (1, 0, 0, 2),
             (1, 1, 2, 0), (2, 0, 1, 0), (2, 1, 0, 1))
    pts = []
    expect = []
    quads = []
    for axis, side, ua, va in faces:
        w = ax[axis][0] if side == 0 else ax[axis][-1]
        outward = -1.0 if side == 0 else 1.0
        grid = {}
        for i in range(nd + 1):
            for j in range(nd + 1):
                p = [0.0, 0.0, 0.0]
                p[axis] = w
                p[ua] = ax[ua][i]
                p[va] = ax[va][j]
                grid[(i, j)] = len(pts)
                pts.append(p)
        for i in range(nd):
            for j in range(nd):
                quads.append((grid[(i, j)], grid[(i + 1, j)],
                              grid[(i + 1, j + 1)], grid[(i, j + 1)]))
        expect.extend([(outward, axis)] * (2 * nd * nd))
    pts = np.asarray(pts, dtype=np.float64)
    tri = []
    for a, b, c, d in quads:
        tri.append((a, b, c))
        tri.append((a, c, d))
    tri = np.asarray(tri, dtype=np.int64)
    exp = np.zeros((len(tri), 3), dtype=np.float64)
    for t, (outward, axis) in enumerate(expect):
        exp[t, axis] = outward
    n = np.cross(pts[tri[:, 1]] - pts[tri[:, 0]], pts[tri[:, 2]] - pts[tri[:, 0]])
    flip = np.einsum("ij,ij->i", n, exp) < 0.0
    if flip.any():
        tri[flip] = tri[flip][:, [0, 2, 1]]
    used = pts[tri.reshape(-1)]
    P, inv = np.unique(used, axis=0, return_inverse=True)
    T = np.asarray(inv, dtype=np.int64).reshape(-1, 3)
    stl_io.check_closed(P, T)
    return P, T


def feature_edge_count(P, T, angle_deg=FEATURE_ANGLE_DEG):
    """Undirected manifold edges whose two face normals differ by more than
    angle_deg. Decided from the STL's own dihedral angles, never from a
    summary (the summary's count drops to 0 when the attraction is off)."""
    P = np.asarray(P, dtype=np.float64)
    T = np.asarray(T, dtype=np.int64)
    nP = len(P)
    a, b, c = P[T[:, 0]], P[T[:, 1]], P[T[:, 2]]
    n = np.cross(b - a, c - a)
    ln = np.linalg.norm(n, axis=1)
    ln[ln == 0.0] = 1.0
    n = n / ln[:, None]
    edges = T[:, [[0, 1], [1, 2], [2, 0]]].reshape(-1, 2)
    en = np.repeat(n, 3, axis=0)
    lo = edges.min(axis=1).astype(np.int64)
    hi = edges.max(axis=1).astype(np.int64)
    uk, inv, cnt = np.unique(lo * np.int64(nP) + hi,
                             return_inverse=True, return_counts=True)
    acc = np.zeros((len(uk), 3), dtype=np.float64)
    np.add.at(acc, inv, en)
    sel = cnt == 2
    dot = (np.einsum("ij,ij->i", acc[sel], acc[sel]) - 2.0) / 2.0
    return int(np.count_nonzero(dot < math.cos(math.radians(angle_deg))))


def patch_areas(g):
    """{solid: total triangle area} from the geometry's own mesh."""
    P = np.asarray(g["P"], dtype=np.float64)
    T = np.asarray(g["T"], dtype=np.int64)
    a, b, c = P[T[:, 0]], P[T[:, 1]], P[T[:, 2]]
    ar = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
    return {g["solid"]: float(ar.sum())}


def _box_geom(geometry_id, lo, hi, nd, stratum, l_ref_m, note):
    """A box geometry dict with the §D.3-window a priori Re_L = 2e+04 flow."""
    P, T = box_mesh(lo, hi, nd)
    return {"geometry_id": geometry_id, "family": "box", "stratum": stratum,
            "solid": "body", "P": P, "T": T, "l_ref_m": l_ref_m,
            "flow": {"u_ref_m_s": 2.0e4 * NU / l_ref_m, "l_ref_m": l_ref_m,
                     "nu_m2_s": NU, "note": note}}


_GEOMS = None


def pilot_geometries():
    """The 12 G-PILOT tuning geometries (docs/15 §F), in this fixed order."""
    global _GEOMS
    if _GEOMS is not None:
        return list(_GEOMS)
    geoms = []
    for i in (0, 1, 2, 3, 12):
        row, _data = gen_wing.make_row(1, i)
        P, T = gen_wing.build(row["params"])
        geoms.append({"geometry_id": row["geometry_id"], "family": gen_wing.FAMILY,
                      "stratum": row["stratum"], "solid": gen_wing.SOLID,
                      "P": np.asarray(P, dtype=np.float64),
                      "T": np.asarray(T, dtype=np.int64),
                      "l_ref_m": gen_wing.l_ref(row["params"]), "flow": row["flow"]})
    for i in (0, 1, 2, 3, 4):
        row, _data = gen_lathe.make_row(1, i)
        P, T = gen_lathe.build(row["params"])
        geoms.append({"geometry_id": row["geometry_id"], "family": gen_lathe.FAMILY,
                      "stratum": row["stratum"], "solid": gen_lathe.SOLID,
                      "P": np.asarray(P, dtype=np.float64),
                      "T": np.asarray(T, dtype=np.int64),
                      "l_ref_m": gen_lathe.l_ref(row["params"]), "flow": row["flow"]})
    geoms.append(_box_geom("BOX-c", (0.0, -0.25, -0.25), (1.0, 0.25, 0.25), 16,
                           "commensurate", 1.0,
                           "a priori target Re_L = 2e+04 (docs/15 §D.3 window)"))
    geoms.append(_box_geom("BOX-n", (0.0137, -0.2113, -0.1709),
                           (0.9137, 0.2291, 0.1893), 16, "straddling", 0.9,
                           "a priori target Re_L = 2e+04 (docs/15 §D.3 window)"))
    _GEOMS = geoms
    return list(geoms)


def cube_geometry():
    """The 8-layer commensurate-cube check's geometry (group cube)."""
    return _box_geom("CUBE", (1.0, 1.0, 1.0), (2.0, 2.0, 2.0), 8, "commensurate",
                     1.0, "a priori target Re_L = 2e+04 (docs/15 §D.3 worked example)")


def write_geometry(g, stl_dir):
    """Write <stl_dir>/<geometry_id>.stl; return the absolute path, forward slashes."""
    path = os.path.join(os.path.abspath(stl_dir), g["geometry_id"] + ".stl")
    stl_io.write_stl(path, g["P"], g["T"], g["solid"])
    return path.replace(os.sep, "/")


def commensurate(g, cfg, level):
    """True when all six face coordinates sit on base_size / 2**level of cfg."""
    ext = cfg["domain"]["extent"]
    cell = cfg["domain"]["base_size"] / 2 ** level
    P = np.asarray(g["P"], dtype=np.float64)
    lo, hi = P.min(axis=0), P.max(axis=0)
    for a in range(3):
        for v in (float(lo[a]), float(hi[a])):
            q = (v - ext[2 * a]) / cell
            if abs(q - round(q)) > 1e-9:
                return False
    return True


# --- the pilot template P0 (C3) ----------------------------------------------

def template_config(g, stl_path, case_dir, name):
    """docs/15 §B's L4 recipe, generalised by L = l_ref_m: the sweep starts
    from the measured failure. No key beyond input/domain/refinement/output."""
    P = np.asarray(g["P"], dtype=np.float64)
    L = g["l_ref_m"]
    b = round(0.5 * L, 4) + 0.0
    lo, hi = P.min(axis=0), P.max(axis=0)
    k = [int(math.floor((lo[0] - 3 * L) / b)), int(math.ceil((hi[0] + 6 * L) / b)),
         int(math.floor((lo[1] - 2.5 * L) / b)), int(math.ceil((hi[1] + 2.5 * L) / b)),
         int(math.floor((lo[2] - 2.5 * L) / b)), int(math.ceil((hi[2] + 2.5 * L) / b))]
    return {"input": {"surfaces": [{"path": stl_path}]},
            "domain": {"extent": [round(ki * b, 10) + 0.0 for ki in k],
                       "base_size": b},
            "refinement": {"levels": [{"patch": g["solid"],
                                       "bands": [{"distance": round(0.1 * L, 6),
                                                  "level": WALL_LEVEL},
                                                 {"distance": round(0.5 * L, 6),
                                                  "level": WALL_LEVEL - 1},
                                                 {"distance": round(1.5 * L, 6),
                                                  "level": WALL_LEVEL - 2}],
                                       "feature_level": WALL_LEVEL}],
                           "max_level": WALL_LEVEL},
            "output": {"case_dir": case_dir, "name": name}}


# --- the a priori helpers (C5, pure) -----------------------------------------

def t1_floor(t1, sig=4):
    """The first-layer thickness rounded DOWN to sig significant digits, so
    schema.yplus_a_priori stays <= 1 (the raw t1 can give 1.0000000000000002)."""
    k = sig - 1 - math.floor(math.log10(t1))
    return math.floor(t1 * 10 ** k) / 10 ** k


def stack_sum(g, n=8):
    """1 + g + ... + g^(n-1) - the §D.3 stack limiter's T / t1 ratio."""
    return sum(g ** i for i in range(n))


def window(g, n=8):
    """The §D.3 h/t1 window (stack_sum / cell_frac, 3 / G5_RATIO), or None."""
    lo = stack_sum(g, n) / CELL_FRAC
    hi = 3 / G5_RATIO
    return None if lo > hi else (lo, hi)


def r_win(t1, base_size, n=8, max_level=6):
    """The pilot's own copy of §D.3's R-WIN: the coarsest level whose h
    satisfies both the G5 early check and the stack limiter, then the largest
    growth whose 8-layer stack still fits cell_frac * h. rules.py owns this
    later; docs/15 §D.3 is the source either way."""
    level = None
    h = None
    for lvl in range(max_level + 1):
        h = base_size / 2 ** lvl
        if 3 * t1 / h >= G5_RATIO and n * t1 <= CELL_FRAC * h:
            level = lvl
            break
    if level is None:
        raise ValueError("R-WIN: the window is empty at max_level %d for "
                         "t1=%g m, base_size=%g m" % (max_level, t1, base_size))
    g_max = None
    for k in range(1000, 2001):
        gk = k / 1000
        if t1 * stack_sum(gk, n) <= CELL_FRAC * h:
            g_max = gk
    return {"level": level, "h": h, "ratio": h / t1, "g_max": g_max,
            "T": t1 * stack_sum(g_max, n)}


# --- pointer edits -----------------------------------------------------------

def _vslug(value):
    """The run-key slug of a knob value: -1 -> m1, +1 -> p1, 0.25 -> 0.25."""
    return str(value).replace("-", "m").replace("+", "p")


def _pointer_get(cfg, pointer):
    cur = cfg
    for seg in pointer.strip("/").split("/"):
        cur = cur[int(seg)] if isinstance(cur, list) else cur[seg]
    return cur


def _pointer_set(cfg, pointer, value):
    segs = pointer.strip("/").split("/")
    cur = cfg
    for seg in segs[:-1]:
        if isinstance(cur, list):
            cur = cur[int(seg)]
            continue
        nxt = cur.get(seg)
        if nxt is None:   # the template has no /snap or /layers block yet
            nxt = {}
            cur[seg] = nxt
        cur = nxt
    last = segs[-1]
    if isinstance(cur, list):
        cur[int(last)] = value
    else:
        cur[last] = value


def _edit(cfg, pointer, to, edits):
    """Apply one pointer edit to cfg and record {pointer, from, to}; `from`
    is the template's value, or the DEFAULTS value when the key is absent."""
    try:
        frm = _pointer_get(cfg, pointer)
    except (KeyError, IndexError, TypeError):
        frm = DEFAULTS.get(pointer)
    _pointer_set(cfg, pointer, to)
    edits.append({"pointer": pointer, "from": frm, "to": to})


def _check_edit(pointer, to, knobs):
    """Raise, naming the refusal's message, when the whitelist refuses."""
    ref = schema.check_edit(pointer, to, knobs)
    if ref is not None:
        raise ValueError("pilot_jobs: %s" % ref["message"])


# --- the jobs (C4) -----------------------------------------------------------

def _editor_wall_level(d):
    def f(c, e):
        bands = c["refinement"]["levels"][0]["bands"]
        for i, bd in enumerate(bands):
            _edit(c, "/refinement/levels/0/bands/%d/level" % i,
                  max(0, bd["level"] + d), e)
        _edit(c, "/refinement/levels/0/feature_level", WALL_LEVEL + d, e)
        _edit(c, "/refinement/max_level", WALL_LEVEL + d, e)
    return f


def _editor_band_distance(factor):
    def f(c, e):
        bands = c["refinement"]["levels"][0]["bands"]
        for i, bd in enumerate(bands):
            _edit(c, "/refinement/levels/0/bands/%d/distance" % i,
                  round(bd["distance"] * factor, 6), e)
    return f


def _editor_feature_level(offset):
    def f(c, e):
        fl = WALL_LEVEL + offset
        _edit(c, "/refinement/levels/0/feature_level", fl, e)
        _edit(c, "/refinement/max_level", max(WALL_LEVEL, fl), e)
    return f


def _editor_pointer(pointer, value):
    def f(c, e):
        _edit(c, pointer, value, e)
    return f


def _editor_layers(solid, t1, growth):
    def f(c, e):
        _edit(c, "/layers/patches", [solid], e)
        _edit(c, "/layers/n", 8, e)
        _edit(c, "/layers/first_thickness", t1, e)
        _edit(c, "/layers/growth", growth, e)
    return f


def _editor_feature_off(c, e):
    _edit(c, "/refinement/levels/0/feature_level", 0, e)


def _geom_plan(g):
    """The 24 (group, knob, value, editor) rows for one tuning geometry."""
    t1 = t1_floor(schema.a_priori_wall(g["flow"])["t1_a_priori_m"])
    solid = g["solid"]
    return [
        ("baseline", "base", 0, lambda c, e: None),
        ("l4", "wall_level", -1, _editor_wall_level(-1)),
        ("l4", "wall_level", 1, _editor_wall_level(1)),
        ("l4", "band_distance", 0.5, _editor_band_distance(0.5)),
        ("l4", "band_distance", 2.0, _editor_band_distance(2.0)),
        ("l4", "feature_level", 1, _editor_feature_level(1)),
        ("l4", "feature_level", 2, _editor_feature_level(2)),
        ("l4", "feature_tolerance", 0.0, _editor_pointer("/snap/feature_tolerance", 0.0)),
        ("l4", "feature_tolerance", 0.25, _editor_pointer("/snap/feature_tolerance", 0.25)),
        ("l4", "snap_smoothing_passes", 0, _editor_pointer("/snap/smoothing_passes", 0)),
        ("l4", "snap_smoothing_passes", 1, _editor_pointer("/snap/smoothing_passes", 1)),
        ("l4", "snap_smoothing_passes", 2, _editor_pointer("/snap/smoothing_passes", 2)),
        ("layers", "growth", 1.1, _editor_layers(solid, t1, 1.1)),
        ("layers", "growth", 1.2, _editor_layers(solid, t1, 1.2)),
        ("layers", "growth", 1.3, _editor_layers(solid, t1, 1.3)),
        ("ext", "iterations", 60, _editor_pointer("/snap/iterations", 60)),
        ("ext", "iterations", 200, _editor_pointer("/snap/iterations", 200)),
        ("ext", "tolerance", 0.0001, _editor_pointer("/snap/tolerance", 0.0001)),
        ("ext", "tolerance", 0.01, _editor_pointer("/snap/tolerance", 0.01)),
        ("ext", "snap_smoothing", 0.0, _editor_pointer("/snap/smoothing", 0.0)),
        ("ext", "snap_smoothing", 1.0, _editor_pointer("/snap/smoothing", 1.0)),
        ("ext", "undo_limit", 0, _editor_pointer("/snap/undo_limit", 0)),
        ("ext", "undo_limit", 10, _editor_pointer("/snap/undo_limit", 10)),
        ("ext", "feature_off", 0, _editor_feature_off),
    ]


def _finish_job(gid, group, knob, value, cfg, edits, work_dir, knobs):
    """Check every edit against the whitelist, set output, return the job."""
    for e in edits:
        _check_edit(e["pointer"], e["to"], knobs)
    run_key = "%s.%s.%s" % (gid, knob, _vslug(value))
    if not RUN_KEY_RE.match(run_key):
        raise ValueError("pilot_jobs: run key %r misses %s" % (run_key, RUN_KEY_RE.pattern))
    case_dir = os.path.join(os.path.abspath(work_dir), run_key).replace(os.sep, "/")
    cfg["output"] = {"case_dir": case_dir, "name": run_key}
    return {"run_key": run_key, "geometry_id": gid, "group": group, "knob": knob,
            "value": value, "edits": edits, "config": cfg, "case_dir": case_dir,
            "config_path": case_dir + "/" + run_key + ".json"}


def pilot_jobs(geoms, stl_dir, work_dir):
    """The G-PILOT job list: 24 jobs per tuning geometry, the 3 cube jobs for
    CUBE. Builds configs only; nothing is written until run_jobs."""
    knobs = schema.load_knobs()
    jobs = []
    for g in geoms:
        gid = g["geometry_id"]
        if gid == "CUBE":
            jobs.extend(_cube_jobs(g, stl_dir, work_dir, knobs))
            continue
        stl_path = os.path.join(os.path.abspath(stl_dir), gid + ".stl").replace(os.sep, "/")
        p0 = template_config(g, stl_path, None, None)
        for group, knob, value, editor in _geom_plan(g):
            cfg = copy.deepcopy(p0)
            edits = []
            editor(cfg, edits)
            jobs.append(_finish_job(gid, group, knob, value, cfg, edits,
                                    work_dir, knobs))
    return jobs


def _cube_jobs(g, stl_dir, work_dir, knobs):
    """The three 8-layer commensurate-cube jobs at the R-WIN t1 (group cube).

    R-PLANE (SPEC-LIT §92.15.5): the faces lie on the cell planes, so the
    feature attraction and snap smoothing are OFF. Not the P0 template."""
    t1 = t1_floor(schema.a_priori_wall(g["flow"])["t1_a_priori_m"])
    w = r_win(t1, 1.0)
    stl_path = os.path.join(os.path.abspath(stl_dir), "CUBE.stl").replace(os.sep, "/")

    def mk(_growth):
        return {"input": {"surfaces": [{"path": stl_path}]},
                "domain": {"extent": [0.0, 4.0, 0.0, 4.0, 0.0, 4.0],
                           "base_size": 1.0},
                "refinement": {"levels": [{"patch": g["solid"],
                                           "bands": [{"distance": 2 * w["h"],
                                                      "level": w["level"]}]}],
                               "max_level": w["level"]}}

    plan = [("rwin", w["g_max"]), ("g", 1.2), ("rwin_np0", w["g_max"])]
    jobs = []
    for knob, value in plan:
        cfg = mk(value)
        edits = []   # snap and layers are added by the edits, so `from` is the default
        for pointer, to in (("/snap/feature_tolerance", 0.0),
                            ("/snap/smoothing_passes", 0),
                            ("/layers/patches", [g["solid"]]),
                            ("/layers/n", 8),
                            ("/layers/first_thickness", t1),
                            ("/layers/growth", value)):
            _edit(cfg, pointer, to, edits)
        if knob == "rwin_np0":
            _edit(cfg, "/layers/normal_passes", 0, edits)
        jobs.append(_finish_job(g["geometry_id"], "cube", knob, value, cfg,
                                edits, work_dir, knobs))
    return jobs


# --- the runner (C6) and the row (C7) ----------------------------------------

def _sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _dash(v):
    return "-" if v is None else str(v)


def _fmt3(v):
    return "-" if v is None else "%.3f" % v


def _row_outcome(raw):
    """The row's trimmed outcome: the C7 keys, patches renamed to layers."""
    return {"verdict": raw["verdict"], "failure_class": raw["failure_class"],
            "failure": raw["failure"], "strict_failure": raw["strict_failure"],
            "flags": raw["flags"], "n_cells": raw["n_cells"],
            "blc8_a_priori": raw["blc8_a_priori"],
            "blc_full_a_priori": raw["blc_full_a_priori"],
            "refusal_line": raw["refusal_line"],
            "layers": [{"name": p["name"], "n_layers": p["n_layers"],
                        "dropped": p["dropped"], "layer_class": p["layer_class"],
                        "full_area_frac": p["full_area_frac"],
                        "t1_requested_m": p["t1_requested_m"],
                        "t1_min_m": p["t1_min_m"], "delivered": p["delivered"]}
                       for p in raw["patches"]]}


def _row_snap(summary, raw_outcome, ratios, cfg):
    """The row's snap block: summary stages[snap] with the three ratios from
    the outcome (recomputed from the stage when scoring failed), plus the
    whole stage dict without `seconds`, for the identity check."""
    if summary is None:
        return None
    st = next((s for s in summary["stages"] if s.get("stage") == "snap"), None)
    if st is None:
        return None
    if ratios is not None:
        pf, p99, mx = ratios
    else:
        nb = st["n_boundary_points"]
        pf = st["n_pinned"] / nb if nb else None
        hf = cfg["domain"]["base_size"] / 2 ** cfg["refinement"]["max_level"]
        p99, mx = st["p99_residual"] / hf, st["max_residual"] / hf
    return {"n_boundary_points": st["n_boundary_points"], "n_pinned": st["n_pinned"],
            "pinned_frac": pf, "p99_over_hf": p99, "max_over_hf": mx,
            "converged": st["converged"], "iterations": st["iterations"],
            "max_step": st["max_step"], "n_scaled_back": st["n_scaled_back"],
            "n_abandoned": st["n_abandoned"],
            "n_feature_edges": st["n_feature_edges"],
            "n_feature_corners": st["n_feature_corners"],
            "n_snapped_to_edge": st["n_snapped_to_edge"],
            "n_snapped_to_corner": st["n_snapped_to_corner"],
            "stage": {k: v for k, v in st.items() if k != "seconds"}}


def _run_one(j, g, binary, bin_sha, timeout_s, areas, fedges):
    """One job: write the config, run the binary, score, drop the polyMesh."""
    cfg = j["config"]
    case_dir = j["case_dir"]
    os.makedirs(case_dir, exist_ok=True)
    with open(j["config_path"], "w", encoding="utf-8", newline="\n") as f:
        json.dump(cfg, f, indent=1)
    so_path = os.path.join(case_dir, "stdout.txt")
    se_path = os.path.join(case_dir, "stderr.txt")
    t0 = time.perf_counter()
    timed_out = False
    peak = 0
    with open(so_path, "wb") as fo, open(se_path, "wb") as fe:
        proc = subprocess.Popen([binary, j["config_path"], "-runId", j["run_key"]],
                                cwd=case_dir, stdout=fo, stderr=fe)
        try:
            watcher = psutil.Process(proc.pid)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            watcher = None   # gone (or closed to us) before it opened: no peak read
        while proc.poll() is None:
            if watcher is not None:
                try:
                    mi = watcher.memory_info()
                    peak = max(peak, getattr(mi, "peak_wset", None) or mi.rss)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
            if time.perf_counter() - t0 > timeout_s:
                proc.kill()
                proc.wait()
                timed_out = True
                break
            time.sleep(0.2)
    seconds = time.perf_counter() - t0
    exit_code = proc.returncode
    stdout = open(so_path, "r", encoding="utf-8", errors="replace").read()
    stderr = open(se_path, "r", encoding="utf-8", errors="replace").read()
    sum_path = os.path.join(case_dir, j["run_key"] + "_summary.json")
    summary = None
    if exit_code == 0 and os.path.isfile(sum_path):
        with open(sum_path, encoding="utf-8") as f:
            summary = json.load(f)
    ratios = None
    raw_outcome = None
    outcome = None
    content_sha = None
    score_error = None
    try:
        res = score.score_run(exit_code=exit_code, stdout=stdout, stderr=stderr,
                              summary=summary, config=cfg,
                              patch_areas_m2=areas, flow=g["flow"],
                              timed_out=timed_out, wall_seconds=seconds,
                              case_dir=case_dir)
        raw_outcome = res["outcome"]
        ratios = (raw_outcome["pinned_frac"], raw_outcome["p99_over_hf"],
                  raw_outcome["max_over_hf"])
        outcome = _row_outcome(raw_outcome)
        content_sha = res["content_sha256"]
    except score.ScoreParseError as e:
        score_error = str(e)
    pm = os.path.join(case_dir, "constant", "polyMesh")
    if os.path.isdir(pm):
        shutil.rmtree(pm)
    ml = cfg["refinement"]["max_level"]
    return {"schema": ROW_SCHEMA, "run_key": j["run_key"],
            "geometry_id": j["geometry_id"], "family": g["family"],
            "stratum": g["stratum"], "feature_bearing": bool(fedges > 0),
            "feature_edges": int(fedges), "group": j["group"], "knob": j["knob"],
            "value": j["value"], "edits": j["edits"],
            "config_sha": schema.canonical_sha256(cfg), "max_level": ml,
            "exit_code": exit_code, "timed_out": timed_out, "seconds": seconds,
            "peak_rss_mib": peak / 2 ** 20,
            "snap": _row_snap(summary, raw_outcome, ratios, cfg),
            "outcome": outcome, "score_error": score_error,
            "content_sha256": content_sha, "binary_sha256": bin_sha}


def run_jobs(jobs, out_dir, geoms, binary, n_jobs, timeout_s, log=print):
    """Run the jobs resume-aware; return only the NEW rows appended this call.

    Rows live in <out_dir>/pilot_rows.jsonl, one JSON object per line. A job
    whose run key is already there with the same config sha is skipped; the
    same run key with a DIFFERENT sha is a ValueError naming the key. Jobs
    run in order of -(max_level), then run key - the expensive ones first."""
    if not (isinstance(n_jobs, int) and not isinstance(n_jobs, bool)
            and 1 <= n_jobs <= MAX_JOBS):
        raise ValueError("run_jobs: n_jobs %r outside 1..%d" % (n_jobs, MAX_JOBS))
    gidx = {g["geometry_id"]: g for g in geoms}
    os.makedirs(out_dir, exist_ok=True)
    rows_path = os.path.join(out_dir, "pilot_rows.jsonl")
    have = {}
    if os.path.isfile(rows_path):
        with open(rows_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    r = json.loads(line)
                    have[r["run_key"]] = r.get("config_sha")
    todo = []
    for j in jobs:
        sha = schema.canonical_sha256(j["config"])
        if j["run_key"] in have:
            if have[j["run_key"]] != sha:
                raise ValueError("stale row for %s: recorded config sha %s, "
                                 "the job now builds %s"
                                 % (j["run_key"], have[j["run_key"]], sha))
            continue
        todo.append(j)
    todo.sort(key=lambda j: (-j["config"]["refinement"]["max_level"], j["run_key"]))
    bin_sha = _sha256_file(binary)
    areas = {gid: patch_areas(g) for gid, g in gidx.items()}
    fedges = {gid: feature_edge_count(g["P"], g["T"]) for gid, g in gidx.items()}
    lock = threading.Lock()
    state = {"n": 0}
    rows = []

    def one(j):
        row = _run_one(j, gidx[j["geometry_id"]], binary, bin_sha, timeout_s,
                       areas[j["geometry_id"]], fedges[j["geometry_id"]])
        with lock:
            state["n"] += 1
            rows.append(row)
            with open(rows_path, "a", encoding="utf-8", newline="\n") as f:
                f.write(json.dumps(row, sort_keys=True, separators=(",", ":"),
                                   ensure_ascii=True) + "\n")
                f.flush()
            oc = row["outcome"]
            log("[%d/%d] %s exit %s pinned %s cells %s %.1f s peak %.0f MiB"
                % (state["n"], len(todo), row["run_key"], _dash(row["exit_code"]),
                   _fmt3(row["snap"]["pinned_frac"] if row["snap"] else None),
                   _dash(oc["n_cells"] if oc else None),
                   row["seconds"], row["peak_rss_mib"]))
        return row

    if todo:
        with concurrent.futures.ThreadPoolExecutor(max_workers=n_jobs) as ex:
            for _ in list(ex.map(one, todo)):
                pass
    return rows


# --- the verdict (C8) --------------------------------------------------------

_KEYS = None


def expected_run_keys():
    """The run keys the FULL pilot must produce (12 x 24 + 3 = 291)."""
    global _KEYS
    if _KEYS is None:
        tmp = os.path.join(tempfile.gettempdir(), "sensitivity-expected")
        jobs = pilot_jobs(pilot_geometries() + [cube_geometry()], tmp, tmp)
        _KEYS = {j["run_key"] for j in jobs}
    return set(_KEYS)


def read_rows(out_dir):
    """The rows of out_dir/pilot_rows.jsonl, in file order."""
    rows = []
    with open(os.path.join(out_dir, "pilot_rows.jsonl"), encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _ok_row(r, pmax):
    s = r["snap"]
    return (r["exit_code"] == 0 and not r["timed_out"] and s is not None
            and s.get("pinned_frac") is not None and s["pinned_frac"] <= pmax)


def g_pilot(rows, gates):
    """The G-PILOT verdict computed from the rows alone. Pure: the numbers
    are measured from rows, never typed (docs/15 §F, §I-2)."""
    F = sorted({r["geometry_id"] for r in rows
                if r["group"] in L4_GROUPS and r["feature_bearing"]})
    n = len(F)
    t = math.ceil(n / 3)
    pmax = gates["pinned_frac_max"]
    inF = set(F)

    def per_knob(knobs, extra):
        out = {}
        for K in knobs:
            out[K] = {r["geometry_id"] for r in rows if r["knob"] == K
                      and r["geometry_id"] in inF and _ok_row(r, pmax) and extra(r)}
        return out

    passed = per_knob(L4_KNOBS, lambda r: True)
    att = per_knob(L4_KNOBS, lambda r: r["snap"]["n_feature_edges"] > 0)
    f3 = per_knob(L4_KNOBS, lambda r: r["snap"]["p99_over_hf"] is not None
                  and r["snap"]["p99_over_hf"] <= gates["p99_residual_over_hf_max"]
                  and r["snap"]["max_over_hf"] is not None
                  and r["snap"]["max_over_hf"] <= gates["max_residual_over_hf_max"])
    extended = per_knob(EXT_KNOBS, lambda r: True)
    baseline_ok = {r["geometry_id"] for r in rows if r["knob"] == "base"
                   and r["geometry_id"] in inF and _ok_row(r, pmax)}
    best = max(L4_KNOBS, key=lambda K: len(passed[K]))
    best_att = max(L4_KNOBS, key=lambda K: len(att[K]))
    best_f3 = max(L4_KNOBS, key=lambda K: len(f3[K]))
    best_ext = max(EXT_KNOBS, key=lambda K: len(extended[K]))
    snap_verdict = "PASS" if max(len(passed[K]) for K in L4_KNOBS) >= t else "KILL"
    cube_rows = [r for r in rows if r["geometry_id"] == "CUBE" and r["knob"] == "rwin"]
    if not cube_rows:
        cube = "MISSING"
    else:
        cr = cube_rows[0]
        oc = cr["outcome"]
        lay = oc["layers"][0] if oc and oc.get("layers") else None
        cube = "DELIVERED" if (lay is not None and lay["delivered"] is True
                               and oc["blc8_a_priori"] == 1.0) else "REFUSED"
    expected = expected_run_keys()
    got = {r["run_key"] for r in rows}
    present = len(got & expected)
    complete = expected <= got
    if not complete:
        overall = "INCOMPLETE (%d of %d runs)" % (present, len(expected))
    elif snap_verdict == "KILL":
        overall = "KILL (return to the user before AM-13/AM-14, docs/15 §F)"
    elif cube != "DELIVERED":
        overall = "CUBE REFUSED (G-BLC-0 returns to the user)"
    else:
        overall = "PASS"

    def kv(d, knobs):
        return ", ".join("%s %d" % (K, len(d[K])) for K in knobs)

    def cube_parts():
        cr = cube_rows[0] if cube_rows else None
        if cr is None:
            return "no CUBE.rwin.* row among the rows"
        oc = cr["outcome"]
        lay = (oc["layers"][0] if oc and oc.get("layers") else {})
        main = ("%s: n_layers %s, dropped %s, full_area_frac %s, t1 %s m, "
                "growth %s, level %s, blc8 %s, blc_full %s"
                % (cr["run_key"], _dash(lay.get("n_layers")),
                   _dash(lay.get("dropped")), _dash(lay.get("full_area_frac")),
                   _dash(lay.get("t1_requested_m")), _dash(cr.get("value")),
                   _dash(cr.get("max_level")),
                   _dash(oc["blc8_a_priori"] if oc else None),
                   _dash(oc["blc_full_a_priori"] if oc else None)))
        others = []
        for r in sorted((r for r in rows if r["geometry_id"] == "CUBE"
                         and r["knob"] in ("g", "rwin_np0")),
                        key=lambda r: r["run_key"]):
            l0 = (r["outcome"]["layers"][0]
                  if r["outcome"] and r["outcome"].get("layers") else {})
            others.append("%s %s/%s" % (r["run_key"], _dash(l0.get("n_layers")),
                                        _dash(l0.get("dropped"))))
        return main + ("; also " + ", ".join(others) if others else "")

    lines = [
        "G-PILOT snap: %s - best L4 knob %s brings pinned <= 5 %% on %d of %d "
        "feature-bearing geometries (needs ceil(%d/3) = %d); per knob: %s"
        % (snap_verdict, best, len(passed[best]), n, n, t, kv(passed, L4_KNOBS)),
        "G-PILOT snap, attraction on (n_feature_edges > 0): best %s %d of %d; "
        "per knob: %s" % (best_att, len(att[best_att]), n, kv(att, L4_KNOBS)),
        "G-PILOT snap, F3 clean (pinned <= 5 %%, p99/h_f <= 0.1, max/h_f <= 0.5): "
        "best %s %d of %d; per knob: %s"
        % (best_f3, len(f3[best_f3]), n, kv(f3, L4_KNOBS)),
        "G-PILOT extended (whitelisted snap knobs, reported only): best %s %d of "
        "%d; per knob: %s" % (best_ext, len(extended[best_ext]), n,
                              kv(extended, EXT_KNOBS)),
        "G-PILOT baseline: pinned <= 5 %% on %d of %d before any knob moves"
        % (len(baseline_ok), n),
        "G-PILOT cube: %s - %s" % (cube, cube_parts()),
        "G-PILOT: %s" % overall,
    ]
    return {"n_feature": n, "needed": t, "snap_verdict": snap_verdict, "best": best,
            "passed": {K: len(passed[K]) for K in L4_KNOBS},
            "attraction_on": {K: len(att[K]) for K in L4_KNOBS},
            "f3_clean": {K: len(f3[K]) for K in L4_KNOBS},
            "extended": {K: len(extended[K]) for K in EXT_KNOBS},
            "baseline_ok": len(baseline_ok), "cube": cube, "complete": complete,
            "n_present": present, "n_expected": len(expected), "overall": overall,
            "lines": lines}


# --- the report (C9) ---------------------------------------------------------

L4_PAIRS = [("base", 0), ("wall_level", -1), ("wall_level", 1),
            ("band_distance", 0.5), ("band_distance", 2.0),
            ("feature_level", 1), ("feature_level", 2),
            ("feature_tolerance", 0.0), ("feature_tolerance", 0.25),
            ("snap_smoothing_passes", 0), ("snap_smoothing_passes", 1),
            ("snap_smoothing_passes", 2)]
GROWTH_VALUES = (1.1, 1.2, 1.3)
GROWTH_PAIRS = [("growth", v) for v in GROWTH_VALUES]
EXT_PAIRS = [("iterations", 60), ("iterations", 200), ("tolerance", 0.0001),
             ("tolerance", 0.01), ("snap_smoothing", 0.0), ("snap_smoothing", 1.0),
             ("undo_limit", 0), ("undo_limit", 10), ("feature_off", 0)]
COST_ROWS = [("L3 (wall_level -1)", "wall_level", -1), ("L4 (base)", "base", 0),
             ("L5 (wall_level +1)", "wall_level", 1),
             ("L4 + feature L5", "feature_level", 1),
             ("L4 + feature L6", "feature_level", 2)]


def _git_head():
    try:
        p = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO,
                           capture_output=True, text=True, timeout=30)
        return (p.stdout or "").strip() or "unknown"
    except OSError:
        return "unknown"


def _med(vs):
    return statistics.median(vs) if vs else None


def _pair_rows(rows, knob, value, inF=None):
    return [r for r in rows if r["knob"] == knob and r["value"] == value
            and (inF is None or r["geometry_id"] in inF)]


def _cell_pin(r):
    if r is None:
        return "-"
    if r["timed_out"]:
        return "T"
    if r["exit_code"] != 0:
        return "x%d" % r["exit_code"]
    s = r["snap"]
    if s is None or s.get("pinned_frac") is None:
        return "-"
    return "%.3f" % s["pinned_frac"]


def render_report(out_dir):
    """Rebuild pilot_report.json and G-PILOT.md from the rows alone."""
    rows = read_rows(out_dir)
    gates = schema.load_gates()
    gp = g_pilot(rows, gates)
    idx = {}
    for r in rows:
        idx.setdefault((r["geometry_id"], r["knob"], r["value"]), r)
    inF = {r["geometry_id"] for r in rows
           if r["group"] in L4_GROUPS and r["feature_bearing"]}
    bin_sha = rows[0]["binary_sha256"] if rows else "-"
    head = _git_head()
    total_s = sum(r["seconds"] or 0.0 for r in rows)
    exits = {}
    for r in rows:
        exits[r["exit_code"]] = exits.get(r["exit_code"], 0) + 1
    md = ["# G-PILOT — AM-5 pilot", "",
          "binary sha256: `%s`" % bin_sha, "git head: `%s`" % head,
          "rows: %d; total seconds: %.1f" % (len(rows), total_s),
          "exits by code: " + (", ".join("exit %s: %d" % (k, v)
                                         for k, v in sorted(exits.items(),
                                                            key=lambda kv: str(kv[0])))
                               or "none"), ""]
    md.extend(gp["lines"])
    md.append("")
    # -- section 2: pinned fraction tables
    md.append("## Pinned fraction, L4 knobs")
    md.append("")
    pairs = [p for p in L4_PAIRS if any(r["knob"] == p[0] and r["value"] == p[1]
                                        and r["group"] in ("baseline", "l4")
                                        for r in rows)]
    gids = sorted({r["geometry_id"] for r in rows
                   if r["group"] in ("baseline", "l4")})
    md.append("| geometry | family | stratum | feature edges | "
              + " | ".join("%s %s" % (k, _vslug(v)) for k, v in pairs) + " |")
    md.append("|" + "---|" * (4 + len(pairs)))
    for gid in gids:
        r0 = idx.get((gid, "base", 0)) or \
            next((r for r in rows if r["geometry_id"] == gid), None)
        fe = _dash(r0 and r0.get("feature_edges"))
        cells = [_cell_pin(idx.get((gid, k, v))) for k, v in pairs]
        md.append("| %s | %s | %s | %s | %s |"
                  % (gid, _dash(r0 and r0.get("family")),
                     _dash(r0 and r0.get("stratum")), fe,
                     " | ".join(cells)))
    md.append("")
    md.append("## Pinned fraction, extended knobs (reported only)")
    md.append("")
    epairs = [p for p in EXT_PAIRS if any(r["knob"] == p[0] and r["value"] == p[1]
                                          and r["group"] == "ext" for r in rows)]
    md.append("| geometry | " + " | ".join("%s %s" % (k, _vslug(v))
                                           for k, v in epairs) + " |")
    md.append("|" + "---|" * (1 + len(epairs)))
    for gid in gids:
        cells = [_cell_pin(idx.get((gid, k, v))) for k, v in epairs]
        md.append("| %s | %s |" % (gid, " | ".join(cells)))
    md.append("")
    # -- section 3: per knob-value
    md.append("## Per knob-value")
    md.append("")
    base_cells = {}
    for gid in gids:
        r0 = idx.get((gid, "base", 0))
        if r0 and r0["outcome"] and r0["outcome"]["n_cells"]:
            base_cells[gid] = r0["outcome"]["n_cells"]
    md.append("| knob | value | n | median pinned | ok | F3 clean | median cells / base | median s |")
    md.append("|---|---|---|---|---|---|---|---|")
    per_knob = []
    for knob, value in L4_PAIRS + GROWTH_PAIRS + EXT_PAIRS:
        rs = _pair_rows(rows, knob, value, inF)
        if not rs:
            continue
        pins = [r["snap"]["pinned_frac"] for r in rs
                if r["snap"] and r["snap"]["pinned_frac"] is not None]
        okc = sum(1 for r in rs if _ok_row(r, gates["pinned_frac_max"]))
        f3c = sum(1 for r in rs if _ok_row(r, gates["pinned_frac_max"])
                  and r["snap"]["p99_over_hf"] is not None
                  and r["snap"]["p99_over_hf"] <= gates["p99_residual_over_hf_max"]
                  and r["snap"]["max_over_hf"] is not None
                  and r["snap"]["max_over_hf"] <= gates["max_residual_over_hf_max"])
        rats = [r["outcome"]["n_cells"] / base_cells[r["geometry_id"]]
                for r in rs if r["outcome"] and r["outcome"]["n_cells"]
                and r["geometry_id"] in base_cells]
        secs = [r["seconds"] for r in rs if r["seconds"] is not None]
        rowd = {"knob": knob, "value": value, "n": len(rs),
                "median_pinned_frac": _med(pins), "ok": okc, "f3_clean": f3c,
                "median_cells_ratio": _med(rats), "median_seconds": _med(secs)}
        per_knob.append(rowd)
        md.append("| %s | %s | %d | %s | %d | %d | %s | %s |"
                  % (knob, _vslug(value), len(rs), _fmt3(rowd["median_pinned_frac"]),
                     okc, f3c, _fmt3(rowd["median_cells_ratio"]),
                     _fmt3(rowd["median_seconds"])))
    md.append("")
    # -- section 4: cost per level
    md.append("## Cost per level")
    md.append("")
    mem = psutil.virtual_memory().total
    cost = []
    md.append("| level | family | n | s median/max | cells median/max | peak MiB "
              "median/max | cap |")
    md.append("|---|---|---|---|---|---|---|")
    for label, knob, value in COST_ROWS:
        for fam in ("A", "B", "box", "all"):
            rs = _pair_rows(rows, knob, value)
            if fam != "all":
                rs = [r for r in rs if r["family"] == fam]
            if not rs:
                continue
            secs = [r["seconds"] for r in rs if r["seconds"] is not None]
            cells = [r["outcome"]["n_cells"] for r in rs
                     if r["outcome"] and r["outcome"]["n_cells"]]
            pks = [r["peak_rss_mib"] for r in rs if r["peak_rss_mib"] is not None]
            pmax = max(pks) if pks else 0.0
            cap = (min(12, math.floor((mem - 4 * 2 ** 30) / (pmax * 2 ** 20)))
                   if pmax > 0 else None)
            cost.append({"level": label, "family": fam, "n": len(rs),
                         "seconds_median": _med(secs),
                         "seconds_max": max(secs) if secs else None,
                         "cells_median": _med(cells),
                         "cells_max": int(max(cells)) if cells else None,
                         "peak_mib_median": _med(pks), "peak_mib_max": pmax,
                         "cap_at_this_ram": cap})
            md.append("| %s | %s | %d | %s / %s | %s / %s | %s / %.0f | %s |"
                      % (label, fam, len(rs), _fmt3(cost[-1]["seconds_median"]),
                         _fmt3(cost[-1]["seconds_max"]),
                         _dash(cost[-1]["cells_median"]),
                         _dash(cost[-1]["cells_max"]),
                         _fmt3(cost[-1]["peak_mib_median"]), pmax, _dash(cap)))
    md.append("")
    md.append("cap = min(12, floor((total RAM %.1f GiB - 4 GiB) / max peak)) - "
              "today's machine's total RAM." % (mem / 2 ** 30))
    md.append("")
    # -- section 5: layer sweep and the snap identity check
    md.append("## Layer sweep (growth, n = 8, t1 from R-YP)")
    md.append("")
    md.append("| geometry | growth | exit | failure class | n_layers | layer class "
              "| full area frac | blc8 |")
    md.append("|---|---|---|---|---|---|---|---|")
    layer_rows = []
    ident = {"k": 0, "n": 0}
    for gid in gids:
        r0 = idx.get((gid, "base", 0))
        for gv in GROWTH_VALUES:
            r = idx.get((gid, "growth", gv))
            if r is None:
                continue
            oc = r["outcome"]
            l0 = (oc["layers"][0] if oc and oc.get("layers") else {})
            layer_rows.append({"geometry_id": gid, "growth": gv,
                               "exit_code": r["exit_code"],
                               "failure_class": oc["failure_class"] if oc else None,
                               "n_layers": l0.get("n_layers"),
                               "layer_class": l0.get("layer_class"),
                               "full_area_frac": l0.get("full_area_frac"),
                               "blc8_a_priori": oc["blc8_a_priori"] if oc else None})
            md.append("| %s | %s | %s | %s | %s | %s | %s | %s |"
                      % (gid, _vslug(gv), _dash(r["exit_code"]),
                         _dash(layer_rows[-1]["failure_class"]),
                         _dash(l0.get("n_layers")), _dash(l0.get("layer_class")),
                         _dash(l0.get("full_area_frac")),
                         _dash(layer_rows[-1]["blc8_a_priori"])))
            if (r0 and r0["exit_code"] == 0 and r["exit_code"] == 0
                    and r0["snap"] and r["snap"]):
                ident["n"] += 1
                if r["snap"]["stage"] == r0["snap"]["stage"]:
                    ident["k"] += 1
    md.append("")
    md.append("Snap identity (the growth run's snap stage equal to its base "
              "run's, `seconds` excluded): %d of %d identical." % (ident["k"], ident["n"]))
    md.append("")
    # -- section 6: snap diagnosis, read-only
    md.append("## Snap diagnosis (read-only)")
    md.append("")
    md.append("| geometry | converged | iterations | max step | scaled back | "
              "abandoned | pinned / boundary | snapped to edge / feature edges |")
    md.append("|---|---|---|---|---|---|---|---|")
    for gid in gids:
        s = (idx.get((gid, "base", 0)) or {}).get("snap")
        if not s:
            md.append("| %s | - | - | - | - | - | - | - |" % gid)
            continue
        md.append("| %s | %s | %s | %s | %s | %s | %s / %s | %s / %s |"
                  % (gid, _dash(s.get("converged")), _dash(s.get("iterations")),
                     _dash(s.get("max_step")), _dash(s.get("n_scaled_back")),
                     _dash(s.get("n_abandoned")), _dash(s.get("n_pinned")),
                     _dash(s.get("n_boundary_points")),
                     _dash(s.get("n_snapped_to_edge")),
                     _dash(s.get("n_feature_edges"))))
    md.append("")
    # -- section 7: the cube rows in full
    md.append("## Cube check")
    md.append("")
    cube_rows = [r for r in rows if r["geometry_id"] == "CUBE"]
    if not cube_rows:
        md.append("No cube rows in this out directory.")
    for r in sorted(cube_rows, key=lambda r: r["run_key"]):
        md.append("### %s" % r["run_key"])
        md.append("")
        md.append("```json")
        md.append(json.dumps(r, sort_keys=True, indent=1, ensure_ascii=True))
        md.append("```")
        md.append("")
    report = {"g_pilot": gp, "per_knob": per_knob, "cost": cost,
              "layers": layer_rows, "identity": ident, "binary_sha256": bin_sha,
              "git_head": head, "n_rows": len(rows)}
    with open(os.path.join(out_dir, "pilot_report.json"), "w",
              encoding="utf-8", newline="\n") as f:
        json.dump(report, f, sort_keys=True, indent=1)
        f.write("\n")
    with open(os.path.join(out_dir, "G-PILOT.md"), "w",
              encoding="utf-8", newline="\n") as f:
        f.write("\n".join(md) + "\n")
    return report


# --- the CLI (C10) -----------------------------------------------------------

def _cmd_list():
    """Job counts per group, no run."""
    geoms = pilot_geometries() + [cube_geometry()]
    tmp = os.path.join(tempfile.gettempdir(), "sensitivity-list")
    jobs = pilot_jobs(geoms, tmp, tmp)
    counts = {}
    for j in jobs:
        counts[j["group"]] = counts.get(j["group"], 0) + 1
    assert len({j["run_key"] for j in jobs}) == len(jobs), "run keys not unique"
    print("jobs: %d = %d for the 12 tuning geometries (%s) + %d cube"
          % (len(jobs), len(jobs) - counts.get("cube", 0),
             ", ".join("%s %d" % (g, counts[g]) for g in
                       ("baseline", "l4", "layers", "ext")),
             counts.get("cube", 0)))
    return 0


def _cmd_pilot(a):
    if not a.out or not a.work:
        print("--pilot needs --out DIR and --work DIR")
        return 2
    if not (1 <= a.jobs <= MAX_JOBS):
        print("--jobs must be 1..%d" % MAX_JOBS)
        return 2
    if not os.path.isfile(a.binary):
        print("binary missing: %s" % a.binary)
        return 1
    geoms = pilot_geometries() + [cube_geometry()]
    if a.only:
        keep = {s.strip() for s in a.only.split(",") if s.strip()}
        geoms = [g for g in geoms if g["geometry_id"] in keep]
    stl_dir = os.path.join(os.path.abspath(a.work), "stl")
    os.makedirs(stl_dir, exist_ok=True)
    for g in geoms:
        write_geometry(g, stl_dir)
    jobs = pilot_jobs(geoms, stl_dir, a.work)
    if a.groups:
        keepg = {s.strip() for s in a.groups.split(",") if s.strip()}
        jobs = [j for j in jobs if j["group"] in keepg]
    done = run_jobs(jobs, a.out, geoms, a.binary, a.jobs, a.timeout)
    print("[ok] pilot: %d new rows this invocation" % len(done))
    rep = render_report(a.out)
    for line in rep["g_pilot"]["lines"]:
        print(line)
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="G-PILOT (docs/15 §F): the one-at-a-time L4 knob sweep")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--report", metavar="DIR")
    ap.add_argument("--out", metavar="DIR")
    ap.add_argument("--work", metavar="DIR")
    ap.add_argument("--jobs", type=int, default=MAX_JOBS)
    ap.add_argument("--timeout", type=float, default=1800.0)
    ap.add_argument("--only", metavar="ID,ID")
    ap.add_argument("--groups", metavar="g,g")
    ap.add_argument("--binary", default=BINARY_DEFAULT)
    a = ap.parse_args(argv)
    if a.selftest:
        return _selftest()
    if a.list:
        return _cmd_list()
    if a.report:
        render_report(a.report)
        print("[ok] report: %s" % os.path.join(a.report, "G-PILOT.md"))
        return 0
    if a.pilot:
        return _cmd_pilot(a)
    ap.error("one of --selftest, --list, --pilot, --report is required")


# --- the selftest (R1-R9) ----------------------------------------------------

FORBIDDEN_KEYS = ("quality", "cell_frac", "medial_frac", "min_thickness",
                  "permissive")   # never in a config this file writes (§C, §I-5)
GATES_TEST = {"pinned_frac_max": 0.05, "p99_residual_over_hf_max": 0.1,
              "max_residual_over_hf_max": 0.5}


def _leaves(cfg, prefix=""):
    """Leaf pointers of a config; scalar lists (extent, patches) are leaves."""
    out = {}
    if isinstance(cfg, dict):
        for k, v in cfg.items():
            out.update(_leaves(v, prefix + "/" + k))
    elif isinstance(cfg, list) and any(isinstance(v, (dict, list)) for v in cfg):
        for i, v in enumerate(cfg):
            out.update(_leaves(v, prefix + "/%d" % i))
    else:
        out[prefix] = cfg
    return out


def _chk_geoms():
    geoms = pilot_geometries()
    ids = [g["geometry_id"] for g in geoms]
    want = ["A-1-000", "A-1-001", "A-1-002", "A-1-003", "A-1-012",
            "B-1-000", "B-1-001", "B-1-002", "B-1-003", "B-1-004", "BOX-c", "BOX-n"]
    assert ids == want, ids
    assert [g["family"] for g in geoms] == ["A"] * 5 + ["B"] * 5 + ["box"] * 2
    tris = {g["geometry_id"]: len(g["T"]) for g in geoms}
    assert tris["BOX-c"] == 3072 and tris["BOX-n"] == 3072, tris
    edges = {g["geometry_id"]: feature_edge_count(g["P"], g["T"]) for g in geoms}
    by = {g["geometry_id"]: g for g in geoms}
    c_ok = [commensurate(by["BOX-c"], template_config(by["BOX-c"], "x.stl", None, None), L)
            for L in range(7)]
    n_ok = [commensurate(by["BOX-n"], template_config(by["BOX-n"], "x.stl", None, None), L)
            for L in range(7)]
    assert c_ok == [False] + [True] * 6, c_ok
    assert not any(n_ok), n_ok
    strata = [g["stratum"] for g in geoms]
    return ("12 ids A/B seed 1 + BOX-c, BOX-n; triangles %s; feature edges %s; "
            "BOX-c commensurate at levels 1..6 not 0, BOX-n at none of 0..6; "
            "strata %s" % (sorted({tris[i] for i in want[:10]}) +
                           [tris["BOX-c"], tris["BOX-n"]], edges, strata))


def _chk_box():
    P16, T16 = box_mesh((0.0, -0.25, -0.25), (1.0, 0.25, 0.25), 16)
    stl_io.check_closed(P16, T16)   # raises unless closed AND outward (volume > 0)
    P4, T4 = box_mesh((0.0, 0.0, 0.0), (1.0, 1.0, 1.0), 4)
    stl_io.check_closed(P4, T4)
    e16 = feature_edge_count(P16, T16)
    e4 = feature_edge_count(P4, T4)
    assert (len(P16), len(T16)) == (1538, 3072), (len(P16), len(T16))
    assert e16 == 192, e16
    assert e4 == 48, e4
    wing_min = min(feature_edge_count(g["P"], g["T"]) for g in pilot_geometries()
                   if g["family"] == "A")
    assert wing_min > 0, wing_min
    return ("closed and outward by check_closed; 1538 points, 3072 triangles; "
            "feature edges nd16 %d, nd4 %d; wing minimum %d > 0"
            % (e16, e4, wing_min))


def _chk_template():
    by = {g["geometry_id"]: g for g in pilot_geometries()}
    for gid in ("A-1-012", "BOX-c", "BOX-n"):
        g = by[gid]
        cfg = template_config(g, "stl.stl", "case", gid)
        assert set(cfg) == {"input", "domain", "refinement", "output"}, set(cfg)
        for bad in ("quality", "snap", "layers", "castellation"):
            assert bad not in cfg, bad
        b = cfg["domain"]["base_size"]
        for v in cfg["domain"]["extent"]:
            assert abs(v / b - round(v / b)) <= 1e-9, (gid, v, b)
        lv = cfg["refinement"]["levels"][0]
        L = g["l_ref_m"]
        assert [bd["distance"] for bd in lv["bands"]] == \
            [round(0.1 * L, 6), round(0.5 * L, 6), round(1.5 * L, 6)], lv["bands"]
        assert [bd["level"] for bd in lv["bands"]] == [4, 3, 2]
        assert lv["feature_level"] == WALL_LEVEL
        assert cfg["refinement"]["max_level"] == WALL_LEVEL
        assert lv["patch"] == g["solid"]
    return ("keys exactly input/domain/refinement/output; extents integer "
            "multiples of base_size; bands 0.1/0.5/1.5 L at levels 4/3/2; no "
            "quality, snap, layers or castellation key")


def _chk_jobs():
    tmp = os.path.join(tempfile.gettempdir(), "sensitivity-selftest-jobs")
    geoms = pilot_geometries()
    jobs = pilot_jobs(geoms, tmp, tmp)
    assert len(jobs) == 288, len(jobs)
    counts = {}
    for j in jobs:
        counts[j["group"]] = counts.get(j["group"], 0) + 1
    assert counts == {"baseline": 12, "l4": 132, "layers": 36, "ext": 108}, counts
    knobs = schema.load_knobs()
    seen = set()
    for j in jobs:
        assert RUN_KEY_RE.match(j["run_key"]), j["run_key"]
        assert j["run_key"] not in seen, j["run_key"]
        seen.add(j["run_key"])
        for e in j["edits"]:
            ref = schema.check_edit(e["pointer"], e["to"], knobs)
            assert ref is None, (j["run_key"], ref and ref["message"])
    by = {g["geometry_id"]: g for g in geoms}
    for gid, g in by.items():
        stl_path = os.path.join(tmp, gid + ".stl").replace(os.sep, "/")
        p0 = _leaves({k: v for k, v in
                      template_config(g, stl_path, None, None).items()
                      if k != "output"})
        gj = [j for j in jobs if j["geometry_id"] == gid]
        assert len(gj) == 24, (gid, len(gj))
        shas = set()
        for j in gj:
            lv = _leaves({k: v for k, v in j["config"].items() if k != "output"})
            diff = {p for p in set(lv) | set(p0) if lv.get(p) != p0.get(p)}
            eps = {e["pointer"] for e in j["edits"]}
            assert diff == eps, (j["run_key"], sorted(diff ^ eps))
            shas.add(schema.canonical_sha256(j["config"]))
            for p in _leaves(j["config"]):
                for bad in FORBIDDEN_KEYS:
                    assert bad not in p.split("/"), (j["run_key"], p)
        assert len(shas) == 24, (gid, len(shas))
    cj = pilot_jobs([cube_geometry()], tmp, tmp)
    assert [j["run_key"] for j in cj] == ["CUBE.rwin.1.27", "CUBE.g.1.2",
                                          "CUBE.rwin_np0.1.27"], cj
    for j in cj:
        for e in j["edits"]:
            assert e["from"] != e["to"], (j["run_key"], e)
            assert schema.check_edit(e["pointer"], e["to"], knobs) is None, e
    return ("288 jobs (baseline 12, l4 132, layers 36, ext 108), 24 per "
            "geometry; every edit passes the whitelist; every config differs "
            "from the template only at its edits' pointers plus output; run "
            "keys unique and matching the pattern; 24 distinct config shas "
            "per geometry; no forbidden pointer in any config")


def _chk_rwin():
    assert t1_floor(7.29699e-4) == 7.296e-4, t1_floor(7.29699e-4)
    flow = cube_geometry()["flow"]
    t1 = t1_floor(schema.a_priori_wall(flow)["t1_a_priori_m"])
    assert t1 == 7.296e-4, t1
    yp = schema.yplus_a_priori(t1, flow)
    assert yp <= 1.0, yp
    w = r_win(t1, 1.0)
    assert w["level"] == 5 and w["h"] == 1.0 / 32, w
    assert w["g_max"] == 1.27, w
    assert w["T"] <= CELL_FRAC * w["h"], w
    assert t1 * stack_sum(1.271) > CELL_FRAC * w["h"]
    w10, w12, w13, w14 = window(1.0), window(1.2), window(1.3), window(1.4)
    assert round(w10[0], 1) == 16.0 and w10[1] == 60, w10
    assert round(w12[0], 1) == 33.0 and round(w13[0], 1) == 47.7, (w12, w13)
    assert w14 is None, w14
    return ("t1_floor(7.29699e-4) = 7.296e-4 with y+ %.6f <= 1; r_win: level %d, "
            "g %.3f, T %.6f <= 0.5 h, and 1.271 over it; window(g): %.1f/%.1f/"
            "%.1f/empty at 1/1.2/1.3/1.4"
            % (yp, w["level"], w["g_max"], w["T"], w10[0], w12[0], w13[0]))


def _vrow(gid, knob="wall_level", value=1, exit_code=0, timed_out=False,
          pinned=0.01, nfe=12, p99=0.01, mx=0.05, feature=True, group="l4"):
    """A synthetic row for the verdict checks."""
    return {"run_key": "%s.%s.%s" % (gid, knob, _vslug(value)),
            "geometry_id": gid, "family": "A", "stratum": "hard",
            "feature_bearing": feature, "feature_edges": nfe, "group": group,
            "knob": knob, "value": value, "exit_code": exit_code,
            "timed_out": timed_out, "score_error": None,
            "snap": {"pinned_frac": pinned, "n_feature_edges": nfe,
                     "p99_over_hf": p99, "max_over_hf": mx},
            "outcome": None}


def _chk_verdict():
    gates = GATES_TEST
    rows1 = [_vrow("W%02d" % i, pinned=(0.01 if i < 4 else 0.40))
             for i in range(10)]
    g1 = g_pilot(rows1, gates)
    assert g1["snap_verdict"] == "PASS" and g1["needed"] == 4, g1
    assert g1["best"] == "wall_level" and g1["passed"]["wall_level"] == 4, g1
    rows2 = [_vrow("W%02d" % i, pinned=(0.01 if i < 3 else 0.40))
             for i in range(10)]
    g2 = g_pilot(rows2, gates)
    assert g2["snap_verdict"] == "KILL" and g2["passed"]["wall_level"] == 3, g2
    rows3 = [_vrow("W%02d" % i, pinned=(0.01 if i < 7 else 0.40),
                   exit_code=(1 if 3 <= i < 7 else 0)) for i in range(10)]
    g3 = g_pilot(rows3, gates)
    assert g3["passed"]["wall_level"] == 3 and g3["snap_verdict"] == "KILL", g3
    rows4 = [_vrow("W%02d" % i, pinned=(0.01 if i < 7 else 0.40),
                   timed_out=(3 <= i < 7)) for i in range(10)]
    g4 = g_pilot(rows4, gates)
    assert g4["passed"]["wall_level"] == 3 and g4["snap_verdict"] == "KILL", g4
    rows5 = [_vrow("W%02d" % i, knob="base", value=0, group="baseline")
             for i in range(10)]
    g5 = g_pilot(rows5, gates)
    assert max(g5["passed"].values()) == 0 and g5["snap_verdict"] == "KILL", g5
    assert g5["baseline_ok"] == 10, g5
    rows6 = [_vrow("W%02d" % i, pinned=0.01, feature=False) for i in range(10)]
    g6 = g_pilot(rows6, gates)
    assert g6["n_feature"] == 0, g6
    assert max(g6["passed"].values()) == 0, g6   # a non-feature geometry never counts
    assert g6["needed"] == 0 and g6["snap_verdict"] == "PASS", g6  # vacuous at n = 0
    rows7 = ([_vrow("W%02d" % i, pinned=0.01, nfe=12) for i in range(4)]
             + [_vrow("W%02d" % i, pinned=0.01, nfe=0) for i in range(4, 8)])
    g7 = g_pilot(rows7, gates)
    assert g7["passed"]["wall_level"] == 8, g7
    assert g7["attraction_on"]["wall_level"] == 4, g7
    assert g7["f3_clean"]["wall_level"] == 8, g7   # F3 clean does not need the attraction
    rows8 = ([_vrow("W%02d" % i, pinned=0.01, p99=0.2) for i in range(4)]
             + [_vrow("W%02d" % i, pinned=0.01, mx=0.6) for i in range(4, 7)]
             + [_vrow("W%02d" % i, pinned=0.01) for i in range(7, 10)])
    g8 = g_pilot(rows8, gates)
    assert g8["passed"]["wall_level"] == 10, g8
    assert g8["attraction_on"]["wall_level"] == 10, g8
    assert g8["f3_clean"]["wall_level"] == 3, g8
    assert g8["overall"].startswith("INCOMPLETE"), g8["overall"]
    return ("6 cases: 4 of 10 PASS, 3 of 10 KILL, exit!=0 and timed-out never "
            "count, base-only and non-feature never count, attraction_on "
            "counts only n_feature_edges > 0, f3_clean adds the residual "
            "bounds")


def _live_jobs(work, stl_dir):
    """The R7 live pair: the small cube with layers n 3 and without layers."""
    P, T = box_mesh((1.0, 1.0, 1.0), (2.5, 2.5, 2.5), 1)
    g = {"geometry_id": "LIVE", "family": "box", "stratum": "commensurate",
         "solid": "body", "P": P, "T": T, "l_ref_m": 1.0,
         "flow": {"u_ref_m_s": 0.3, "l_ref_m": 1.0, "nu_m2_s": NU,
                  "note": "the R7 live check (small cube)"}}
    stl_path = os.path.join(os.path.abspath(stl_dir), "LIVE.stl").replace(os.sep, "/")
    jobs = []
    for rk, with_layers in (("live.with_layers", True), ("live.no_layers", False)):
        cfg = {"input": {"surfaces": [{"path": stl_path}]},
               "domain": {"extent": [0.0, 4.0, 0.0, 4.0, 0.0, 4.0],
                          "base_size": 1.0},
               "refinement": {"levels": [{"patch": "body",
                                          "bands": [{"distance": 0.5, "level": 1}]}],
                              "max_level": 1},
               "snap": {"feature_tolerance": 0.0, "smoothing_passes": 0}}
        edits = [{"pointer": "/snap/feature_tolerance", "from": 0.5, "to": 0.0},
                 {"pointer": "/snap/smoothing_passes", "from": 3, "to": 0}]
        if with_layers:
            cfg["layers"] = {"patches": ["body"], "n": 3, "first_thickness": 0.02}
            edits.append({"pointer": "/layers/patches", "from": None, "to": ["body"]})
            edits.append({"pointer": "/layers/n", "from": None, "to": 3})
            edits.append({"pointer": "/layers/first_thickness", "from": None,
                          "to": 0.02})
        case_dir = os.path.join(os.path.abspath(work), rk).replace(os.sep, "/")
        cfg["output"] = {"case_dir": case_dir, "name": rk}
        jobs.append({"run_key": rk, "geometry_id": "LIVE", "group": "live",
                     "knob": "layers" if with_layers else "base",
                     "value": 1 if with_layers else 0, "edits": edits,
                     "config": cfg, "case_dir": case_dir,
                     "config_path": case_dir + "/" + rk + ".json"})
    return g, jobs


def _chk_live(out, work, stl_dir):
    if not os.path.isfile(BINARY_DEFAULT):
        raise AssertionError("binary missing: %s" % BINARY_DEFAULT)
    os.makedirs(stl_dir, exist_ok=True)
    g, jobs = _live_jobs(work, stl_dir)
    write_geometry(g, stl_dir)
    t0 = time.perf_counter()
    rows = run_jobs(jobs, out, [g], BINARY_DEFAULT, 2, 60.0, log=lambda s: None)
    wall = time.perf_counter() - t0
    assert len(rows) == 2, len(rows)
    by = {r["run_key"]: r for r in rows}
    wl, nl = by["live.with_layers"], by["live.no_layers"]
    cds = {j["run_key"]: j["case_dir"] for j in jobs}
    for r in rows:
        assert r["exit_code"] == 0 and not r["timed_out"], (r["run_key"], r["exit_code"])
        assert r["peak_rss_mib"] > 0, r["peak_rss_mib"]
        assert not os.path.isdir(os.path.join(cds[r["run_key"]], "constant",
                                              "polyMesh"))
    lay = wl["outcome"]["layers"][0]
    assert lay["n_layers"] == 3, lay
    summ = {}
    for rk in ("live.with_layers", "live.no_layers"):
        path = os.path.join(cds[rk], rk + "_summary.json")
        with open(path, encoding="utf-8") as f:
            summ[rk] = json.load(f)
    st = next(s for s in summ["live.with_layers"]["stages"]
              if s["stage"] == "layers")
    assert st["patches"][0]["n_faces"] == 54, st["patches"][0]
    assert st["n_layer_cells"] == 162, st["n_layer_cells"]
    assert st["patches"][0]["n_layers"] == 3, st["patches"][0]
    assert wl["snap"]["stage"] == nl["snap"]["stage"], "snap stages differ"
    cfgs = {j["run_key"]: j["config"] for j in jobs}
    for rk in ("live.with_layers", "live.no_layers"):
        for pointer, dv in DEFAULTS.items():
            node = summ[rk]["config"]
            for seg in pointer.strip("/").split("/"):
                node = node[seg]
            try:
                want = _pointer_get(cfgs[rk], pointer)
            except (KeyError, IndexError, TypeError):
                want = dv
            assert node == want, (rk, pointer, node, want)
    return ("%d runs exit 0 in %.1f s; layer row n_layers 3, n_faces 54, %d "
            "layer cells; snap stages identical; peak %.0f/%.0f MiB; polyMesh "
            "removed; echoed defaults match"
            % (len(rows), wall, st["n_layer_cells"], wl["peak_rss_mib"],
               nl["peak_rss_mib"]))


def _chk_resume(out, work, stl_dir):
    g, jobs = _live_jobs(work, stl_dir)
    rows = run_jobs(jobs, out, [g], BINARY_DEFAULT, 2, 60.0, log=lambda s: None)
    assert rows == [], rows
    j0 = copy.deepcopy(jobs[0])
    j0["config"]["refinement"]["levels"][0]["bands"][0]["distance"] = 0.4
    try:
        run_jobs([j0], out, [g], BINARY_DEFAULT, 2, 60.0, log=lambda s: None)
    except ValueError as e:
        assert "stale row for live.with_layers" in str(e), str(e)
    else:
        raise AssertionError("a changed config under a used run key did not raise")
    return ("0 jobs rerun into the same out dir (resume by config sha); a "
            "changed config under the same run key raises ValueError naming "
            "the key")


def _chk_report(out):
    rep = render_report(out)
    mdp = os.path.join(out, "G-PILOT.md")
    jsp = os.path.join(out, "pilot_report.json")
    assert os.path.isfile(mdp) and os.path.isfile(jsp), (mdp, jsp)
    md = open(mdp, encoding="utf-8").read()
    assert "G-PILOT" in md, "no G-PILOT in the markdown"
    assert "INCOMPLETE" in md, "no INCOMPLETE in the markdown"
    assert rep["g_pilot"]["overall"].startswith("INCOMPLETE"), rep["g_pilot"]["overall"]
    assert rep["n_rows"] == 2, rep["n_rows"]
    return ("G-PILOT.md + pilot_report.json from the 2 live rows; overall "
            "%r" % rep["g_pilot"]["overall"])


def _chk_fast_exit(out, work):
    """A child that exits before psutil opens it: the row is written, not raised."""
    real = subprocess.Popen

    class _Gone(real):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            self.wait()
    g, jobs = _live_jobs(work, os.path.join(work, "stl"))
    j = copy.deepcopy(jobs[1])
    j["run_key"] = "live.fast_exit"
    j["case_dir"] = os.path.join(os.path.abspath(work), "live.fast_exit").replace(os.sep, "/")
    j["config_path"] = j["case_dir"] + "/live.fast_exit.json"
    subprocess.Popen = _Gone
    try:
        row = _run_one(j, g, sys.executable, "-", 60.0, patch_areas(g), 12)
    finally:
        subprocess.Popen = real
    assert row["exit_code"] is not None and not row["timed_out"]         and row["peak_rss_mib"] == 0.0, row
    return "a child gone before the watcher opens gives exit %d, peak 0 MiB, no raise" % row["exit_code"]


def _selftest():
    """R1-R9 and the fast-exit check as ten [ok] groups; temp dirs under tempfile, removed in finally."""
    root = tempfile.mkdtemp(prefix="sensitivity-selftest-")
    out = os.path.join(root, "out")
    work = os.path.join(root, "work")
    stl_dir = os.path.join(work, "stl")
    passed = []
    failed = []

    def check(name, fn):
        try:
            msg = fn()
        except Exception as e:  # a failed check is a FAIL, never a pass
            failed.append(name)
            print("SELFTEST FAIL: %s: %s: %s" % (name, type(e).__name__, e))
            return
        passed.append(name)
        print("[ok] %s: %s" % (name, msg))

    try:
        check("geometries", lambda: _chk_geoms())
        check("box", lambda: _chk_box())
        check("template", _chk_template)
        check("jobs", _chk_jobs)
        check("r_win", _chk_rwin)
        check("verdict", _chk_verdict)
        check("live", lambda: _chk_live(out, work, stl_dir))
        check("resume", lambda: _chk_resume(out, work, stl_dir))
        check("report", lambda: _chk_report(out))
        check("fast exit", lambda: _chk_fast_exit(out, work))
    finally:
        shutil.rmtree(root, ignore_errors=True)
    if failed or len(passed) < 10:
        print("SELFTEST FAIL: %d of 10 checks passed (%s failed)"
              % (len(passed), ", ".join(failed) or "none"))
        return 1
    print("SELFTEST PASS")
    return 0


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    try:
        sys.exit(main())
    except SystemExit:
        raise
    except Exception as e:
        print("ERROR: %s: %s" % (type(e).__name__, e))
        sys.exit(1)
