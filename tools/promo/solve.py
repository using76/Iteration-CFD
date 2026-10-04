#!/usr/bin/env python3
# meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
# Source-available, not Open Source. See LICENSE at the repository root.
# No GPL-licensed source was consulted.

"""The F1 promo video's moving-ground RANS case: write, run, post.

Writes the 250 km/h case on the full promo tunnel mesh and drives the pinned
GPU solver ``ofgpu-lowmach`` (rust/target/gpu-pin/ofgpu-lowmach.exe, built
from feat/core-2 90510fc, sha256-checked) as a pseudo-transient march to a
steady state: steady SIMPLE diverges in its first outer iteration on this
mesh (M 3.9e6 at a body-adjacent cell at step 0, laminar and kOmegaSST
alike, from rest or from uniform U), while ddtSchemes Euler + PIMPLE stays
bounded, so the steady answer is the end state of a transient march judged
by the house stopping rule (solver numerics untouched - only case inputs
and run flags). The four wheels rotate through ``movingWallVelocity``
per-face values written into 0/U (v = Omega x (xf - c) per face); the
ground moves at the free-stream speed. The mesh and geometry derive from a
CC BY 4.0 model ("F1 2026 concept" by Qvist_Designs, via Sketchfab), so
NOTHING made from them is written inside the repository: every output goes
under --out, which must lie outside the repository, with the model's
LICENSE.txt copied beside it.

Subcommands: write (case files, CPU only), run (write + GPU preflight +
launch + post), post (re-derive Cd/Cl/front-rear balance/residuals from an
existing out). --selftest runs T1-T9 on the CPU.
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
import subprocess
import sys
import tempfile
import threading
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tunnel_mesh import refuse, read_stl_bin  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TOOL = "tools/promo/solve.py"
VERSION = "promo-solve/1"
RHO = 1.2041          # the solver's printed rho_ref
T_K = 293.15
INTENSITY = 0.005
NUT_RATIO = 10.0
CAR = ["body", "front_wing", "rear_wing", "wheels", "floor"]
MIN_FREE_MIB = 11000
BIN = "rust/target/gpu-pin/ofgpu-lowmach.exe"
BIN_SHA256 = "50471caaa54125e0c2eee4fb34eebdfc2da44727d2b818a2b96bb4234c02103c"
PATCHES = ["inlet", "outlet", "side_ymin", "side_ymax", "zMin", "top",
           "body", "front_wing", "rear_wing", "wheels", "floor", "ground"]
SLIP = ["side_ymin", "side_ymax", "top", "zMin"]
PLAIN_WALLS = ["body", "front_wing", "rear_wing", "floor"]
WHEEL_ORDER = ["front_left", "front_right", "rear_left", "rear_right"]
# the house stopping rule, tools/cad/gates.json "stop_rule", verbatim
# (the values below; that file lives on the core/gui branches, not here)
STOP_RULE = {"residual_decades": 4, "cont_err_max": 1e-06,
             "window_iters": 200, "dp_rel_change_max": 1e-05,
             "cd_rel_change_max": 1e-05}
XCHECK_LIMIT = 1e-4

_NUM = r"-?(?:nan|inf|\d+(?:\.\d+)?(?:e[+-]\d+)?)"
RE_ITER = re.compile(
    r"^iter +(\d+)  \|U\| res (" + _NUM + r")  \|p\| res (" + _NUM +
    r")  contErr (" + _NUM + r")  T \[(" + _NUM + r"), (" + _NUM +
    r")\] K  rho \[(" + _NUM + r"), (" + _NUM + r")\] kg/m3  p0 (" + _NUM +
    r") Pa  dp0/dt (" + _NUM + r") Pa/s  M max (" + _NUM + r") \(cell (\d+)\) mean (" +
    _NUM + r")(  \*\*\* NaN/Inf \*\*\*)?$")
RE_FORCE = re.compile(
    r"^  (\w+): area (\S+) m2 \| F_visc = \((\S+) (\S+) (\S+)\) N "
    r".*?\| F_pres = \((\S+) (\S+) (\S+)\) N")
RE_DONE = re.compile(r"^done in (\S+) s$")
RE_END = re.compile(r"^run ended: (budget|diverged|refused|error) \| (.*) \| exit code (\d+)$")
RE_SNAP = re.compile(r"[0-9]+(?:\.[0-9]+)?")

FIXTURE_T4 = "[ofgpu] -permissive: max Mach number 1.009 at cell 971629 (step 0) is above 0.3; SPEC-LIT §25's premise M << 1 no longer holds (at M = 0.3 rho/rho0 = 0.956 already, a 4.4 % density change) and the acoustic filtering is unjustified - continuing because -permissive was given (SPEC-LIT §93.6)\niter      0  |U| res 0.0203496  |p| res 3.553e-02  contErr 0.00046231  T [293.15, 293.15] K  rho [1.2041, 1.2041] kg/m3  p0 101325 Pa  dp0/dt 0 Pa/s  M max 1.00879 (cell 971629) mean 0.202352\niter      1  |U| res 0.014021  |p| res 5.718e-02  contErr 0.000549338  T [293.15, 293.15] K  rho [1.2041, 1.2041] kg/m3  p0 101325 Pa  dp0/dt 0 Pa/s  M max 0.604784 (cell 62536) mean 0.20235\niter     19  |U| res 0.000921024  |p| res 1.246e-03  contErr 2.50188e-05  T [293.15, 293.15] K  rho [1.2041, 1.2041] kg/m3  p0 101325 Pa  dp0/dt 0 Pa/s  M max 0.391677 (cell 615044) mean 0.202361\ndone in 108.58 s\n  body: area 19.825 m2 | F_visc = (2.7138 0.00389234 -0.205464) N [wall function (rho u_tau^2, u_tau = Cmu^1/4 sqrt(k_P))] | F_pres = (2072.05 -5.97582 1856.73) N | axis = (0.997145 0.00143018 -0.0754946) (DEFAULT: this patch's own mean traction direction - no thermostat direction and no cyclic pair) | drag along axis: viscous 2.72157 N, pressure 1925.95 N, total 1928.67 N\n  front_wing: area 4.5611 m2 | F_visc = (0.624381 0.000510913 0.0211063) N [wall function (rho u_tau^2, u_tau = Cmu^1/4 sqrt(k_P))] | F_pres = (570.337 -2.91981 -1515.38) N | axis = (0.999429 0.000817804 0.0337842) (DEFAULT: this patch's own mean traction direction - no thermostat direction and no cyclic pair) | drag along axis: viscous 0.624738 N, pressure 518.813 N, total 519.438 N\n  rear_wing: area 2.63293 m2 | F_visc = (0.31891 0.000865539 -0.0504377) N [wall function (rho u_tau^2, u_tau = Cmu^1/4 sqrt(k_P))] | F_pres = (346.484 8.65936 -1694.68) N | axis = (0.98772 0.00268072 -0.156214) (DEFAULT: this patch's own mean traction direction - no thermostat direction and no cyclic pair) | drag along axis: viscous 0.322875 N, pressure 606.985 N, total 607.308 N\n  wheels: area 8.32422 m2 | F_visc = (0.464589 -0.000106945 -0.264938) N [wall function (rho u_tau^2, u_tau = Cmu^1/4 sqrt(k_P))] | F_pres = (3288.18 -0.0994423 770.276) N | axis = (0.868679 -0.000199963 -0.495375) (DEFAULT: this patch's own mean traction direction - no thermostat direction and no cyclic pair) | drag along axis: viscous 0.534822 N, pressure 2474.79 N, total 2475.33 N\n  floor: area 5.03436 m2 | F_visc = (0.86405 -0.000463814 -0.0234324) N [wall function (rho u_tau^2, u_tau = Cmu^1/4 sqrt(k_P))] | F_pres = (115.784 10.7844 -3853.97) N | axis = (0.999632 -0.000536593 -0.0271093) (DEFAULT: this patch's own mean traction direction - no thermostat direction and no cyclic pair) | drag along axis: viscous 0.864368 N, pressure 220.214 N, total 221.079 N\n  ground: area 1439.85 m2 | F_visc = (36.8767 0.00199016 0) N [wall function (rho u_tau^2, u_tau = Cmu^1/4 sqrt(k_P))] | F_pres = (0 0 -37305.9) N | axis = (1 5.3968e-05 0) (DEFAULT: this patch's own mean traction direction - no thermostat direction and no cyclic pair) | drag along axis: viscous 36.8767 N, pressure 0 N, total 36.8767 N\ntotal: F_visc = (41.8624 0.0066882 -0.523166) N | F_pres = (6392.83 10.4487 -41742.9) N | F = (6434.69 10.4553 -41743.4) N | rho_ref = 1.2041 kg/m3 | nu = 1.5e-05 m2/s\nrun ended: budget | endTime 0.01 s reached in 20 steps | exit code 0\n"


def fnum(x):
    """repr-precision number formatting."""
    return repr(float(x))


# ---------------------------------------------------------------- readers

def np_nums(text, dtype=np.float64):
    """Whitespace-separated numbers -> 1-D array. np.fromstring degrades to
    minutes on a 350 MB string, so parse in newline-aligned ~8 MB chunks."""
    out = []
    k, n = 0, len(text)
    step = 8 << 20
    while k < n:
        e = text.find("\n", k + step)
        e = n if e == -1 else e + 1
        v = np.fromstring(text[k:e], dtype=dtype, sep=" ")
        if v.size:
            out.append(v)
        k = e
    return np.concatenate(out) if out else np.zeros(0, dtype=dtype)


def _list_slice(text):
    """The body of the first ( ... ) list."""
    i = text.index("\n(") + 2
    j = text.index("\n)", i)
    return text[i:j]


def read_boundary(path):
    """[(name, nFaces, startFace)] in file order."""
    text = open(path, encoding="utf-8", errors="replace").read()
    body = _list_slice(text)
    out = []
    for m in re.finditer(r"(\w+)\s*\n\s*\{(.*?)\}", body, re.S):
        name = m.group(1)
        blk = m.group(2)
        nf = re.search(r"nFaces\s+(\d+)", blk)
        sf = re.search(r"startFace\s+(\d+)", blk)
        if not (nf and sf):
            refuse("PS-MESH", "boundary: patch " + name + " lacks nFaces/startFace")
        out.append((name, int(nf.group(1)), int(sf.group(1))))
    if not out:
        refuse("PS-MESH", path + ": no patch list found")
    return out


def read_points(path):
    text = open(path, encoding="utf-8", errors="replace").read()
    body = _list_slice(text)
    vals = np_nums(body.replace("(", " ").replace(")", " "), np.float64)
    pts = vals.reshape(-1, 3)
    if pts.shape[0] == 0:
        refuse("PS-MESH", path + ": empty point list")
    return pts


def read_ints(path):
    text = open(path, encoding="utf-8", errors="replace").read()
    return np_nums(_list_slice(text), np.int64)


def _faces_slow(body):
    """Exotic meshes (arity >= 10): per-face regex, correctness over speed."""
    ars = []
    chunks = []
    for m in re.finditer(rb"(\d+)\s*\(([^)]*)\)", body):
        ars.append(int(m.group(1)))
        chunks.append(np_nums(m.group(2).decode("ascii"), np.int64))
    if not ars:
        refuse("PS-MESH", "faces: no face list found")
    arity = np.array(ars, dtype=np.int64)
    flat = np.concatenate(chunks)
    if flat.size != int(arity.sum()):
        refuse("PS-MESH", "faces: index count " + str(flat.size) +
               " does not match the face arities")
    ptr = np.concatenate([np.zeros(1, np.int64), np.cumsum(arity, dtype=np.int64)])
    return arity, ptr, flat


def read_faces(path):
    """(arity (nf,), ptr (nf+1,), flat point indices), vectorised over the
    whole file - no per-face Python loop on the common path."""
    blob = open(path, "rb").read()
    i = blob.index(b"\n(") + 2
    j = blob.index(b"\n)", i)
    body = blob[i:j]
    arr = np.frombuffer(body, dtype=np.uint8)
    ipos = np.flatnonzero(arr == 40)
    if ipos.size == 0:
        return _faces_slow(body)
    arity = arr[ipos - 1].astype(np.int64) - 48
    prev = arr[np.maximum(ipos - 2, 0)]
    if not (np.all((arity >= 1) & (arity <= 9)) and
            not np.any((prev >= 48) & (prev <= 57))):
        return _faces_slow(body)
    n = arity.size
    clean = body.replace(b"(", b" ").replace(b")", b" ").decode("ascii")
    flat = np_nums(clean, np.int64)
    if np.all(arity == arity[0]):
        m = int(arity[0])
        if flat.size == n * (m + 1):
            grid = flat.reshape(n, m + 1)
            if np.all(grid[:, 0] == m):
                idx = np.ascontiguousarray(grid[:, 1:]).reshape(-1)
                ptr = np.arange(0, n * m + 1, m, dtype=np.int64)
                return arity, ptr, idx
    total = flat.size
    if total == int((arity + 1).sum()):
        starts = np.zeros(n, dtype=np.int64)
        np.cumsum(arity[:-1] + 1, out=starts[1:])
        row = np.repeat(np.arange(n, dtype=np.int64), arity + 1)
        inrow = np.arange(total, dtype=np.int64) - np.repeat(starts, arity + 1)
        idx = flat[inrow != 0]
        ptr = np.concatenate([np.zeros(1, np.int64), np.cumsum(arity, dtype=np.int64)])
        return arity, ptr, idx
    return _faces_slow(body)


class PolyMesh:
    """points + CSR faces; per-face centres and area vectors (C8)."""

    def __init__(self, points, arity, ptr, flat):
        self.points = points
        self.arity = arity
        self.ptr = ptr
        self.flat = flat

    def face_centres(self, start, count):
        if count == 0:
            return np.zeros((0, 3))
        lo = int(self.ptr[start])
        hi = int(self.ptr[start + count])
        coords = self.points[self.flat[lo:hi]]
        lp = self.ptr[start:start + count + 1] - lo
        return np.add.reduceat(coords, lp[:-1], axis=0) / self.arity[start:start + count, None]

    def face_sf_centres(self, start, count):
        """(Sf (n,3), centres (n,3)); Sf = 0.5 sum (Pi - c) x (Pi+1 - c)."""
        if count == 0:
            return np.zeros((0, 3)), np.zeros((0, 3))
        lo = int(self.ptr[start])
        hi = int(self.ptr[start + count])
        coords = self.points[self.flat[lo:hi]]
        cnt = self.arity[start:start + count]
        lp = self.ptr[start:start + count + 1] - lo
        centres = np.add.reduceat(coords, lp[:-1], axis=0) / cnt[:, None]
        rep = np.repeat(lp[:-1], cnt)
        local = np.arange(coords.shape[0], dtype=np.int64) - rep
        nxt = rep + (local + 1) % np.repeat(cnt, cnt)
        cc = np.repeat(centres, cnt, axis=0)
        cross = np.cross(coords - cc, coords[nxt] - cc)
        sf = 0.5 * np.add.reduceat(cross, lp[:-1], axis=0)
        return sf, centres


def read_foam_scalar(path, n_expected):
    """internalField of one volScalarField -> (n_expected,) float64."""
    text = open(path, encoding="utf-8", errors="replace").read()
    i = text.index("internalField")
    seg = text[i:]
    u = seg.find("uniform")
    v = seg.find("nonuniform")
    if v != -1 and (u == -1 or v < u):
        m = re.search(r"nonuniform\s+List<scalar>\s+(\d+)", seg)
        if not m:
            refuse("PS-LOG", path + ": unparseable nonuniform internalField")
        cnt = int(m.group(1))
        a = seg.index("(", m.end())
        b = seg.index(")", a)
        vals = np_nums(seg[a + 1:b], np.float64)
        if vals.size != cnt:
            refuse("PS-LOG", path + ": " + str(vals.size) + " values for " + str(cnt))
        return vals
    if u != -1:
        rest = seg[u + len("uniform"):].strip().rstrip(";").strip()
        tok = rest.split()
        if not tok:
            refuse("PS-LOG", path + ": uniform internalField without a value")
        return np.full(max(n_expected, 1), float(tok[0]))
    refuse("PS-LOG", path + ": internalField is neither uniform nor nonuniform")


# ---------------------------------------------------------------- wheels

def wheel_clusters(geom_dir, U):
    """C6: the four wheel clusters from <geom>/wheels.stl, or PS-GEOM.
    Returns (clusters dict name -> {cx, cy, cz, R, omega_y, n_tris}, x_split)."""
    stl = os.path.join(geom_dir, "wheels.stl")
    sgj = os.path.join(geom_dir, "sim_geom.json")
    if not os.path.isfile(stl) or not os.path.isfile(sgj):
        refuse("PS-GEOM", geom_dir + ": needs wheels.stl and sim_geom.json")
    tris = read_stl_bin(stl)
    if tris.shape[0] == 0:
        refuse("PS-GEOM", stl + ": no triangles")
    x_split = (float(tris[:, :, 0].min()) + float(tris[:, :, 0].max())) / 2.0
    cent = tris.mean(axis=1)
    is_front = cent[:, 0] < x_split
    is_left = cent[:, 1] >= 0.0
    clusters = {}
    counts = {}
    for name in WHEEL_ORDER:
        front = name.startswith("front")
        left = name.endswith("left")
        sel = ((is_front if front else ~is_front) &
               (is_left if left else ~is_left))
        if not bool(sel.any()):
            refuse("PS-GEOM", stl + ": cluster " + name + " is empty (need exactly 4)")
        pts = tris[sel].reshape(-1, 3)
        lo = pts.min(axis=0)
        hi = pts.max(axis=0)
        c = (lo + hi) / 2.0
        r = (float(hi[2]) - float(lo[2])) / 2.0
        counts[name] = int(sel.sum())
        clusters[name] = {"cx": float(c[0]), "cy": float(c[1]), "cz": float(c[2]),
                          "R": r, "omega_y": -U / r, "n_tris": counts[name]}
    return clusters, x_split


def axles_from(clusters, x_split):
    front_x = (clusters["front_left"]["cx"] + clusters["front_right"]["cx"]) / 2.0
    rear_x = (clusters["rear_left"]["cx"] + clusters["rear_right"]["cx"]) / 2.0
    return {"x_split": float(x_split), "front_x": float(front_x),
            "rear_x": float(rear_x), "wheelbase": float(rear_x - front_x)}


def wheel_face_velocity(centres, x_split, clusters):
    """C6: per-face rigid-body velocity v = Omega x (xf - c); the cluster of
    a face comes from its own centre (front iff xf_x < x_split, left iff
    xf_y >= 0). The solver removes each value's normal component."""
    out = np.zeros((centres.shape[0], 3))
    is_front = centres[:, 0] < x_split
    is_left = centres[:, 1] >= 0.0
    for name in WHEEL_ORDER:
        cl = clusters[name]
        sel = ((is_front if name.startswith("front") else ~is_front) &
               (is_left if name.endswith("left") else ~is_left))
        if not bool(sel.any()):
            continue
        c = np.array([cl["cx"], cl["cy"], cl["cz"]])
        r = centres[sel] - c
        oy = cl["omega_y"]
        out[sel, 0] = oy * r[:, 2]
        out[sel, 2] = -oy * r[:, 0]
    return out


# ---------------------------------------------------------------- forces

def patch_pressure_force(mesh, owner, p, start, count):
    """C8: RHO * sum_f p[owner[f]] * Sf[f] over one patch's face range."""
    sf, _ = mesh.face_sf_centres(start, count)
    pf = owner[start:start + count]
    return RHO * (p[pf][:, None] * sf).sum(axis=0)


def balance(face_f, face_x, front_x, rear_x):
    """C9: axle loads from face force vectors at face centres, moments about
    the rear axle at ground level."""
    wb = rear_x - front_x
    d_front = (float(np.sum(-face_f[:, 2] * (rear_x - face_x[:, 0]))) -
               float(np.sum(face_f[:, 0] * face_x[:, 2]))) / wb
    d_total = float(-np.sum(face_f[:, 2]))
    d_rear = d_total - d_front
    share = (d_front / d_total) if d_total != 0 else None
    return {"D_front_N": d_front, "D_rear_N": d_rear, "front_share": share}


# ---------------------------------------------------------------- log parsing

def _f(x):
    return float(x)


def parse_log(text):
    """C10: iter lines, the last force block, done in, the end line and the
    -permissive lines."""
    iters = []
    forces = {}
    solver_seconds = None
    end = None
    permissive = 0
    permissive_non_mach = 0
    for line in text.splitlines():
        m = RE_ITER.match(line)
        if m:
            iters.append({"step": int(m.group(1)), "U_res": _f(m.group(2)),
                          "p_res": _f(m.group(3)), "cont_err": _f(m.group(4)),
                          "T_lo": _f(m.group(5)), "T_hi": _f(m.group(6)),
                          "rho_lo": _f(m.group(7)), "rho_hi": _f(m.group(8)),
                          "p0": _f(m.group(9)), "dp0_dt": _f(m.group(10)),
                          "M": _f(m.group(11)), "cell": int(m.group(12)),
                          "mean": _f(m.group(13)), "nan": m.group(14) is not None})
            continue
        m = RE_FORCE.match(line)
        if m:
            forces[m.group(1)] = {"area": _f(m.group(2)),
                                  "F_visc": [_f(m.group(3)), _f(m.group(4)), _f(m.group(5))],
                                  "F_pres": [_f(m.group(6)), _f(m.group(7)), _f(m.group(8))]}
            continue
        m = RE_DONE.match(line)
        if m:
            solver_seconds = _f(m.group(1))
            continue
        if line.startswith("[ofgpu] -permissive:"):
            permissive += 1
            if "max Mach number" not in line:
                permissive_non_mach += 1
            continue
        if line.startswith("run ended: "):
            m = RE_END.match(line)
            if m:
                end = (m.group(1), m.group(2), int(m.group(3)))
    return {"iters": iters, "forces": forces, "solver_seconds": solver_seconds,
            "end": end, "permissive": permissive,
            "permissive_non_mach": permissive_non_mach}


def classify(parsed, steps, cd, cl):
    """C11 in order. steps/cd/cl are the aligned per-snapshot series."""
    out = {"class": None, "reason_id": None, "criteria": {}, "failed": []}
    end = parsed["end"]
    if end is None:
        out["class"] = "refused"
        out["reason_id"] = "PS-LOG"
        return out
    word = end[0]
    if word == "diverged":
        out["class"] = "diverged"
        out["reason_id"] = "PS-DIVERGED"
        return out
    if word == "refused":
        out["class"] = "refused"
        out["reason_id"] = "PS-REFUSED"
        return out
    if word == "error":
        out["class"] = "refused"
        out["reason_id"] = "PS-ERROR"
        return out
    if parsed["permissive_non_mach"] > 0:
        out["class"] = "refused"
        out["reason_id"] = "PS-PERMISSIVE"
        return out
    if word != "budget":
        out["class"] = "refused"
        out["reason_id"] = "PS-LOG"
        return out
    iters = parsed["iters"]
    lim = STOP_RULE

    def decades(a, b):
        if a > 0 and b > 0:
            return math.log10(a / b)
        if b == 0 and a > 0:
            return float("inf")
        return float("-inf")

    if iters:
        u_dec = decades(iters[0]["U_res"], iters[-1]["U_res"])
        p_dec = decades(iters[0]["p_res"], iters[-1]["p_res"])
        cont = abs(iters[-1]["cont_err"])
        n_steps = iters[-1]["step"] + 1
    else:
        u_dec = p_dec = None
        cont = None
        n_steps = 0
    crit = {
        "U_decades": {"value": u_dec, "limit": float(lim["residual_decades"]),
                      "pass": u_dec is not None and u_dec >= lim["residual_decades"]},
        "p_decades": {"value": p_dec, "limit": float(lim["residual_decades"]),
                      "pass": p_dec is not None and p_dec >= lim["residual_decades"]},
        "cont_err": {"value": cont, "limit": float(lim["cont_err_max"]),
                     "pass": cont is not None and cont <= lim["cont_err_max"]},
    }
    wsel = [i for i, s in enumerate(steps) if s >= n_steps - lim["window_iters"]]
    for name, series in (("Cl_rel_change", cl), ("Cd_rel_change", cd)):
        ws = [series[i] for i in wsel]
        val = None
        ok = False
        if len(ws) >= 2:
            last = ws[-1]
            if last == 0:
                val = 0.0 if all(w == 0 for w in ws) else float("inf")
            else:
                val = max(abs(w - last) for w in ws) / abs(last)
            ok = val < lim["cd_rel_change_max"]
        crit[name] = {"value": val,
                      "limit": float(lim["cd_rel_change_max"]), "pass": ok}
    out["criteria"] = crit
    order = ["U_decades", "p_decades", "cont_err", "Cl_rel_change", "Cd_rel_change"]
    out["failed"] = [nm for nm in order if not crit[nm]["pass"]]
    if out["failed"]:
        out["class"] = "unsteady"
        out["reason_id"] = "PS-UNSTEADY"
    else:
        out["class"] = "steady"
        out["reason_id"] = None
    return out


# ---------------------------------------------------------------- case writer

def foam_header(name, cls, loc):
    return ("/*--------------------------------*- C++ -*"
            "----------------------------------*\\\n"
            "| ofgpu  --  GPU-native finite volume CFD"
            "                                     |\n"
            "\\*--------------------------------------------"
            "-------------------------------*/\n"
            "FoamFile\n{\n    version     2.0;\n    format      ascii;\n"
            "    class       " + cls + ";\n    location    \"" + loc + "\";\n"
            "    object      " + name + ";\n}\n"
            "// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *"
            " * * * //\n\n")


def _patch_block(patch, entries):
    out = "    " + patch + "\n    {\n"
    for k, v in entries:
        out += "        " + k.ljust(16) + v + ";\n"
    out += "    }\n"
    return out


def _slip():
    return [("type", "slip")]


def boundary_entries(names, U, k_in, omega_in, wheel_value):
    """field -> patch -> [(key, value)] per C4."""
    uvec = "uniform (" + fnum(U) + " 0 0)"
    fields = {"U": {}, "p": {}, "T": {}, "k": {}, "omega": {}, "nut": {}}
    for nm in names:
        if nm == "inlet":
            fields["U"][nm] = [("type", "fixedValue"), ("value", uvec)]
            fields["p"][nm] = [("type", "zeroGradient")]
            fields["T"][nm] = [("type", "fixedValue"), ("value", "uniform " + fnum(T_K))]
            fields["k"][nm] = [("type", "fixedValue"), ("value", "uniform " + fnum(k_in))]
            fields["omega"][nm] = [("type", "fixedValue"), ("value", "uniform " + fnum(omega_in))]
            fields["nut"][nm] = [("type", "calculated"), ("value", "uniform 0")]
        elif nm == "outlet":
            fields["U"][nm] = [("type", "inletOutlet"), ("inletValue", "uniform (0 0 0)"),
                               ("value", uvec)]
            fields["p"][nm] = [("type", "fixedValue"), ("value", "uniform 0")]
            fields["T"][nm] = [("type", "zeroGradient")]
            fields["k"][nm] = [("type", "zeroGradient")]
            fields["omega"][nm] = [("type", "zeroGradient")]
            fields["nut"][nm] = [("type", "calculated"), ("value", "uniform 0")]
        elif nm in SLIP:
            for f in fields:
                fields[f][nm] = _slip()
        elif nm == "ground":
            fields["U"][nm] = [("type", "movingWallVelocity"), ("value", uvec)]
            fields["p"][nm] = [("type", "zeroGradient")]
            fields["T"][nm] = [("type", "zeroGradient")]
            fields["k"][nm] = [("type", "kqRWallFunction"), ("value", "uniform " + fnum(k_in))]
            fields["omega"][nm] = [("type", "omegaWallFunction"),
                                   ("value", "uniform " + fnum(omega_in))]
            fields["nut"][nm] = [("type", "nutkWallFunction"), ("value", "uniform 0")]
        elif nm == "wheels":
            fields["U"][nm] = [("type", "movingWallVelocity"), ("value", wheel_value)]
            fields["p"][nm] = [("type", "zeroGradient")]
            fields["T"][nm] = [("type", "zeroGradient")]
            fields["k"][nm] = [("type", "kqRWallFunction"), ("value", "uniform " + fnum(k_in))]
            fields["omega"][nm] = [("type", "omegaWallFunction"),
                                   ("value", "uniform " + fnum(omega_in))]
            fields["nut"][nm] = [("type", "nutkWallFunction"), ("value", "uniform 0")]
        else:
            fields["U"][nm] = [("type", "noSlip")]
            fields["p"][nm] = [("type", "zeroGradient")]
            fields["T"][nm] = [("type", "zeroGradient")]
            fields["k"][nm] = [("type", "kqRWallFunction"), ("value", "uniform " + fnum(k_in))]
            fields["omega"][nm] = [("type", "omegaWallFunction"),
                                   ("value", "uniform " + fnum(omega_in))]
            fields["nut"][nm] = [("type", "nutkWallFunction"), ("value", "uniform 0")]
    return fields


def _write_text(path, text):
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)


def write_field(case0, name, cls, dims, internal, names, bf):
    text = foam_header(name, cls, "0")
    text += "dimensions      " + dims + ";\n"
    text += "internalField   " + internal + ";\n\n"
    text += "boundaryField\n{\n"
    for nm in names:
        text += _patch_block(nm, bf[nm])
    text += "}\n\n"
    _write_text(os.path.join(case0, name), text)


def solve_command(bin_abs, end_time, dt, write_interval):
    return [bin_abs, "case", "-endTime", fnum(end_time), "-deltaT", fnum(dt),
            "-writeInterval", fnum(write_interval), "-check", "1", "-permissive"]


def bin_path():
    return os.path.normpath(os.path.join(REPO, BIN))


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def binary_info():
    p = bin_path()
    if not os.path.isfile(p):
        return p, None
    return p, sha256_file(p)


def cmd_write(args):
    mesh = os.path.abspath(args.mesh)
    geom = os.path.abspath(args.geom)
    out = os.path.abspath(args.out)
    end_time = float(args.end_time)
    dt = float(args.dt)
    write_interval = float(args.write_interval)
    try:
        rel = os.path.relpath(out, REPO)
    except ValueError:
        rel = ".." + os.sep + ".."
    if rel != ".." and not rel.startswith(".." + os.sep):
        refuse("PS-OUT", out + ": --out resolves inside the repository")
    if os.path.isdir(out):
        if os.listdir(out):
            refuse("PS-OUT", out + ": exists and is not empty")
    elif os.path.exists(out):
        refuse("PS-OUT", out + ": exists and is not a directory")
    pm_dir = os.path.join(mesh, "case", "constant", "polyMesh")
    five = ["boundary", "faces", "owner", "neighbour", "points"]
    missing = [f for f in five if not os.path.isfile(os.path.join(pm_dir, f))]
    if missing:
        refuse("PS-MESH", pm_dir + ": missing " + ", ".join(missing))
    for f in ("tunnel_mesh.json", "LICENSE.txt"):
        if not os.path.isfile(os.path.join(mesh, f)):
            refuse("PS-MESH", mesh + ": missing " + f)
    boundary = read_boundary(os.path.join(pm_dir, "boundary"))
    names = [b[0] for b in boundary]
    if sorted(names) != sorted(PATCHES):
        refuse("PS-MESH", "boundary patches " + ",".join(names) +
               " are not exactly the 12 expected")
    tm = json.load(open(os.path.join(mesh, "tunnel_mesh.json"), encoding="utf-8"))
    speed_kmh = float(tm["params"]["speed_kmh"])
    nu = float(tm["params"]["nu"])
    U = speed_kmh / 3.6
    sg = json.load(open(os.path.join(geom, "sim_geom.json"), encoding="utf-8"))
    A_ref = float(sg["sim_surface"]["frontal_area_m2"])
    k_in = 1.5 * (INTENSITY * U) ** 2
    omega_in = k_in / (NUT_RATIO * nu)
    q = 0.5 * RHO * U * U
    clusters, x_split = wheel_clusters(geom, U)
    axles = axles_from(clusters, x_split)

    t0 = time.monotonic()
    points = read_points(os.path.join(pm_dir, "points"))
    arity, ptr, flat = read_faces(os.path.join(pm_dir, "faces"))
    pm = PolyMesh(points, arity, ptr, flat)
    bd = {b[0]: (b[1], b[2]) for b in boundary}
    wcnt, wstart = bd["wheels"]
    centres = pm.face_centres(wstart, wcnt)
    vel = wheel_face_velocity(centres, x_split, clusters)
    lines = [("(" + fnum(v[0]) + " " + fnum(v[1]) + " " + fnum(v[2]) + ")") for v in vel]
    wheel_value = ("nonuniform List<vector> " + str(wcnt) + "\n    (\n" +
                   "\n".join(lines) + "\n    )")
    print("[promo-solve] mesh parsed in " + "%.1f" % (time.monotonic() - t0) +
          " s (" + str(points.shape[0]) + " points, " + str(arity.size) + " faces)", flush=True)

    case = os.path.join(out, "case")
    os.makedirs(os.path.join(case, "constant", "polyMesh"), exist_ok=True)
    os.makedirs(os.path.join(case, "system"), exist_ok=True)
    os.makedirs(os.path.join(case, "0"), exist_ok=True)
    sizes = {}
    for f in five:
        shutil.copyfile(os.path.join(pm_dir, f), os.path.join(case, "constant", "polyMesh", f))
        sizes[f] = os.path.getsize(os.path.join(pm_dir, f))
    shutil.copyfile(os.path.join(mesh, "LICENSE.txt"), os.path.join(out, "LICENSE.txt"))

    _write_text(os.path.join(case, "constant", "physicalProperties"),
                foam_header("physicalProperties", "dictionary", "constant") +
                "viscosityModel  constant;\n\n"
                "nu              [0 2 -1 0 0 0 0] " + fnum(nu) + ";\n\n")
    _write_text(os.path.join(case, "constant", "momentumTransport"),
                foam_header("momentumTransport", "dictionary", "constant") +
                "simulationType  RAS;\n\nRAS\n{\n    model       kOmegaSST;\n"
                "    turbulence  on;\n    printCoeffs  on;\n}\n\n")
    _write_text(os.path.join(case, "system", "controlDict"),
                foam_header("controlDict", "dictionary", "system") +
                "application     foamRun;\n"
                "startFrom       startTime;\n"
                "startTime       0;\n"
                "stopAt          endTime;\n"
                "endTime         1;\n"
                "deltaT          1;\n"
                "writeControl    timeStep;\n"
                "writeInterval   1;\n"
                "writeFormat     ascii;\n"
                "writePrecision  8;\n\n")
    _write_text(os.path.join(case, "system", "fvSchemes"),
                foam_header("fvSchemes", "dictionary", "system") +
                "ddtSchemes { default Euler; }\n"
                "gradSchemes { default Gauss linear; }\n"
                "divSchemes { default none; div(phi,U) bounded Gauss linearUpwind grad(U);"
                " div(phi,T) bounded Gauss upwind; div(phi,k) bounded Gauss upwind;"
                " div(phi,omega) bounded Gauss upwind; }\n"
                "laplacianSchemes { default Gauss linear corrected; }\n"
                "interpolationSchemes { default linear; }\n"
                "snGradSchemes { default corrected; }\n\n")
    _write_text(os.path.join(case, "system", "fvSolution"),
                foam_header("fvSolution", "dictionary", "system") +
                "solvers\n{\n"
                "    p { solver PCG; preconditioner DIC; tolerance 1e-07; relTol 0.01;"
                " maxIter 1000; }\n"
                "    U { solver PBiCGStab; preconditioner diagonal; tolerance 1e-08;"
                " relTol 0.1; maxIter 200; }\n"
                "    T { solver PBiCGStab; preconditioner diagonal; tolerance 1e-08;"
                " relTol 0.1; maxIter 200; }\n"
                "    k { solver PBiCGStab; preconditioner diagonal; tolerance 1e-08;"
                " relTol 0.1; maxIter 200; }\n"
                "    omega { solver PBiCGStab; preconditioner diagonal; tolerance 1e-08;"
                " relTol 0.1; maxIter 200; }\n"
                "}\n"
                "PIMPLE { nCorrectors 2; nNonOrthogonalCorrectors 1; }\n"
                "relaxationFactors { fields { p 0.3; } equations"
                " { U 0.7; T 0.7; k 0.7; omega 0.7; } }\n\n")

    bf = boundary_entries(names, U, k_in, omega_in, wheel_value)
    write_field(os.path.join(case, "0"), "U", "volVectorField", "[0 1 -1 0 0 0 0]",
                "uniform (" + fnum(U) + " 0 0)", names, bf["U"])
    write_field(os.path.join(case, "0"), "p", "volScalarField", "[0 2 -2 0 0 0 0]",
                "uniform 0", names, bf["p"])
    write_field(os.path.join(case, "0"), "T", "volScalarField", "[0 0 0 1 0 0 0]",
                "uniform " + fnum(T_K), names, bf["T"])
    write_field(os.path.join(case, "0"), "k", "volScalarField", "[0 2 -2 0 0 0 0]",
                "uniform " + fnum(k_in), names, bf["k"])
    write_field(os.path.join(case, "0"), "omega", "volScalarField", "[0 0 -1 0 0 0 0]",
                "uniform " + fnum(omega_in), names, bf["omega"])
    write_field(os.path.join(case, "0"), "nut", "volScalarField", "[0 2 -1 0 0 0 0]",
                "uniform 0", names, bf["nut"])

    bpath, bsha = binary_info()
    wj = {
        "tool": TOOL,
        "version": VERSION,
        "mesh_dir": mesh,
        "geom_dir": geom,
        "out_dir": out,
        "attribution": tm.get("attribution"),
        "params": {"speed_kmh": speed_kmh, "U": U, "nu": nu, "rho": RHO,
                   "A_ref": A_ref, "q": q, "dt": dt, "end_time": end_time,
                   "write_interval": write_interval, "intensity": INTENSITY,
                   "nut_ratio": NUT_RATIO, "k_in": k_in, "omega_in": omega_in},
        "wheels": [dict([("name", nm)] + list(clusters[nm].items())) for nm in WHEEL_ORDER],
        "axles": axles,
        "binary": {"path": bpath, "sha256": bsha},
        "command": solve_command(bpath, end_time, dt, write_interval),
        "sizes": sizes,
        "license_copied": True,
    }
    with open(os.path.join(out, "write.json"), "w", encoding="utf-8", newline="\n") as fh:
        json.dump(wj, fh, ensure_ascii=False, indent=1)
        fh.write("\n")
    print("[promo-solve] case written: " + out + " (" +
          "%.1f" % (time.monotonic() - t0) + " s total, U = " + fnum(U) +
          " m/s, k_in = " + fnum(k_in) + ", omega_in = " + fnum(omega_in) + ")", flush=True)


# ---------------------------------------------------------------- launcher

def nvidia_query():
    """(query text, compute-apps text) or (None, None) when nvidia-smi fails."""
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total,memory.used,utilization.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=120)
        if r.returncode != 0 or not r.stdout.strip():
            return None, None
        a = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=120)
        return r.stdout.strip(), (a.stdout.strip() if a.returncode == 0 else "")
    except (OSError, subprocess.TimeoutExpired):
        return None, None


def gpu_preflight():
    """PS-GPU: (before_text, apps_text, shared)."""
    before, apps = nvidia_query()
    if before is None:
        refuse("PS-GPU", "nvidia-smi query failed")
    first = before.splitlines()[0]
    parts = [p.strip() for p in first.split(",")]
    if len(parts) < 3:
        refuse("PS-GPU", "nvidia-smi output has no total,used,utilization: " + first)
    total = float(parts[0])
    used = float(parts[1])
    util = float(parts[2])
    if total - used < MIN_FREE_MIB:
        refuse("PS-GPU", "only " + fnum(total - used) + " MiB free (" +
               fnum(total) + " - " + fnum(used) + "), need " + str(MIN_FREE_MIB))
    shared = bool(apps) or util > 0
    return before, apps, shared


def launch(out, cmd, dt, n_steps_total, timeout):
    """C13: run cmd with cwd <out>, stream into solve.log + console, one
    progress line every 20 steps; timeout > 0 kills the child (rc 124)."""
    log_path = os.path.join(out, "solve.log")
    t0 = time.monotonic()
    last_step = -1
    last_m = float("nan")
    with open(log_path, "w", encoding="utf-8", newline="\n") as log:
        proc = subprocess.Popen(cmd, cwd=out, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, encoding="utf-8",
                                errors="replace", bufsize=1)
        killed = False
        watchdog = None
        if timeout > 0:
            def _bite():
                nonlocal killed
                killed = True
                proc.kill()
            watchdog = threading.Timer(timeout, _bite)
            watchdog.daemon = True
            watchdog.start()
        try:
            for line in proc.stdout:
                log.write(line)
                log.flush()
                try:
                    sys.stdout.write(line)
                    sys.stdout.flush()
                except Exception:
                    pass
                m = RE_ITER.match(line)
                if m:
                    step = int(m.group(1))
                    last_step = step
                    last_m = float(m.group(11))
                    if (step + 1) % 20 == 0:
                        elapsed = time.monotonic() - t0
                        s = step + 1
                        eta = (elapsed / s * (n_steps_total - s)
                               if 0 < s < n_steps_total else 0.0)
                        print("[promo-solve] step " + str(s) + "/" + str(n_steps_total) +
                              "  t=" + fnum((step + 1) * dt) + " s  elapsed " +
                              "%.1f" % elapsed + " s  ETA " + "%.1f" % eta +
                              " s  M max " + "%.6g" % last_m, flush=True)
            rc = proc.wait(timeout=(timeout if timeout > 0 else None))
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
            rc = 124
        finally:
            if watchdog is not None:
                watchdog.cancel()
        if killed:
            rc = 124
    return rc, time.monotonic() - t0


def cmd_run(args):
    out = os.path.abspath(args.out)
    cmd_write(args)
    wj_path = os.path.join(out, "write.json")
    wj = json.load(open(wj_path, encoding="utf-8"))
    params = wj["params"]
    bpath, bsha = binary_info()
    if bsha is None:
        refuse("PS-BIN", bpath + ": the pinned solver binary is missing")
    if bsha != BIN_SHA256:
        refuse("PS-BIN", bpath + ": sha256 " + bsha + " does not match the pin " + BIN_SHA256)
    before, apps, shared = gpu_preflight()
    cmd = wj["command"]
    n_total = int(round(float(params["end_time"]) / float(params["dt"])))
    print("[promo-solve] launching " + os.path.basename(bpath) +
          " (shared GPU: " + str(shared) + ")", flush=True)
    rc, wall = launch(out, cmd, float(params["dt"]), n_total, float(args.timeout))
    after, _ = nvidia_query()
    with open(os.path.join(out, "run.json"), "w", encoding="utf-8", newline="\n") as fh:
        json.dump({"returncode": rc, "wall_seconds": wall,
                   "gpu": {"before": before, "after": after, "shared": shared}},
                  fh, ensure_ascii=False, indent=1)
        fh.write("\n")
    cmd_post(argparse.Namespace(out=out))


# ---------------------------------------------------------------- post

def cmd_post(args):
    out = os.path.abspath(args.out)
    wj_path = os.path.join(out, "write.json")
    if not os.path.isdir(out) or not os.path.isfile(wj_path):
        refuse("PS-OUT", out + ": post needs the directory to exist with write.json")
    wj = json.load(open(wj_path, encoding="utf-8"))
    params = wj["params"]
    axles = wj["axles"]
    A_ref = float(params["A_ref"])
    q = float(params["q"])
    dt = float(params["dt"])
    log_path = os.path.join(out, "solve.log")
    text = ""
    if os.path.isfile(log_path):
        text = open(log_path, encoding="utf-8", errors="replace").read()
    parsed = parse_log(text)

    pm_dir = os.path.join(out, "case", "constant", "polyMesh")
    boundary = read_boundary(os.path.join(pm_dir, "boundary"))
    bd = {b[0]: (b[1], b[2]) for b in boundary}
    points = read_points(os.path.join(pm_dir, "points"))
    arity, ptr, flat = read_faces(os.path.join(pm_dir, "faces"))
    pm = PolyMesh(points, arity, ptr, flat)
    owner = read_ints(os.path.join(pm_dir, "owner"))
    n_cells = int(owner.max()) + 1 if owner.size else 0

    case_dir = os.path.join(out, "case")
    snaps = []
    for nm in os.listdir(case_dir):
        full = os.path.join(case_dir, nm)
        if os.path.isdir(full) and RE_SNAP.fullmatch(nm) and float(nm) > 0:
            if os.path.isfile(os.path.join(full, "p")):
                snaps.append((float(nm), nm))
    snaps.sort()

    rows = []
    for t, nm in snaps:
        pvals = read_foam_scalar(os.path.join(case_dir, nm, "p"), n_cells)
        fv = np.zeros(3)
        fl = []
        xl = []
        per = {}
        for name in CAR:
            if name not in bd:
                continue
            cnt, st = bd[name]
            sf, cen = pm.face_sf_centres(st, cnt)
            fvec = RHO * pvals[owner[st:st + cnt]][:, None] * sf
            per[name] = [float(x) for x in fvec.sum(axis=0)]
            fv = fv + fvec.sum(axis=0)
            fl.append(fvec)
            xl.append(cen)
        qa = q * A_ref
        bal = {"D_front_N": None, "D_rear_N": None, "front_share": None}
        if fl:
            bal = balance(np.concatenate(fl), np.concatenate(xl),
                          float(axles["front_x"]), float(axles["rear_x"]))
        rows.append({"time": t, "step": int(round(t / dt)), "F": fv,
                     "Cd_p": (float(fv[0]) / qa if qa != 0 else None),
                     "Cl_p": (float(fv[2]) / qa if qa != 0 else None),
                     "per": per, "balance": bal})

    with open(os.path.join(out, "residuals.csv"), "w", encoding="utf-8", newline="\n") as fh:
        fh.write("step,U_res,p_res,cont_err,M_max,M_cell\n")
        for it in parsed["iters"]:
            fh.write(",".join([str(it["step"]), fnum(it["U_res"]), fnum(it["p_res"]),
                               fnum(it["cont_err"]), fnum(it["M"]), str(it["cell"])]) + "\n")
    with open(os.path.join(out, "forces.csv"), "w", encoding="utf-8", newline="\n") as fh:
        fh.write("time,step,Fx,Fy,Fz,Cd_p,Cl_p,D_front_N,D_rear_N,front_share\n")
        for r in rows:
            sh = r["balance"]["front_share"]
            fh.write(",".join([fnum(r["time"]), str(r["step"]), fnum(r["F"][0]),
                               fnum(r["F"][1]), fnum(r["F"][2]), fnum(r["Cd_p"]),
                               fnum(r["Cl_p"]), fnum(r["balance"]["D_front_N"]),
                               fnum(r["balance"]["D_rear_N"]),
                               "" if sh is None else fnum(sh)]) + "\n")

    cls = classify(parsed, [r["step"] for r in rows],
                   [r["Cd_p"] for r in rows], [r["Cl_p"] for r in rows])

    n_steps = (parsed["iters"][-1]["step"] + 1) if parsed["iters"] else 0
    wsel = [r for r in rows if r["step"] >= n_steps - STOP_RULE["window_iters"]]
    times = [r["time"] for r in wsel]
    cdw = [r["Cd_p"] for r in wsel]
    clw = [r["Cl_p"] for r in wsel]

    def stats(v):
        if not v:
            return {"mean": None, "min": None, "max": None}
        a = np.array([x for x in v if x is not None])
        if a.size == 0:
            return {"mean": None, "min": None, "max": None}
        return {"mean": float(a.mean()), "min": float(a.min()), "max": float(a.max())}

    cds = stats(cdw)
    cls_ = stats(clw)
    window = {"times": times, "Cd_p": cdw, "Cl_p": clw,
              "front_share": [r["balance"]["front_share"] for r in wsel],
              "Cd_mean": cds["mean"], "Cd_min": cds["min"], "Cd_max": cds["max"],
              "Cl_mean": cls_["mean"], "Cl_min": cls_["min"], "Cl_max": cls_["max"]}

    xrel = {}
    if rows:
        last = rows[-1]
        for name in CAR:
            if name in last["per"] and name in parsed["forces"]:
                ft = np.array(last["per"][name])
                fs = np.array(parsed["forces"][name]["F_pres"])
                xrel[name] = float(np.linalg.norm(ft - fs) /
                                   max(float(np.linalg.norm(fs)), 1.0))
    max_rel = max(xrel.values()) if xrel else None

    per_out = {}
    cv = np.zeros(3)
    cp = np.zeros(3)
    for nm, blk in parsed["forces"].items():
        per_out[nm] = {"F_visc": blk["F_visc"], "F_pres": blk["F_pres"]}
    for name in CAR:
        if name in parsed["forces"]:
            cv = cv + np.array(parsed["forces"][name]["F_visc"])
            cp = cp + np.array(parsed["forces"][name]["F_pres"])
    fvec = cv + cp
    qa = q * A_ref
    final = {"per_patch": per_out,
             "car": {"F_visc": [float(x) for x in cv], "F_pres": [float(x) for x in cp],
                     "F": [float(x) for x in fvec]},
             "drag_N": float(fvec[0]), "downforce_N": float(-fvec[2]),
             "Cd": (float(fvec[0]) / qa if qa != 0 else None),
             "Cl": (float(fvec[2]) / qa if qa != 0 else None)}
    final["CdA"] = (final["Cd"] * A_ref if final["Cd"] is not None else None)
    final["ClA"] = (final["Cl"] * A_ref if final["Cl"] is not None else None)

    runj = {}
    runj_path = os.path.join(out, "run.json")
    if os.path.isfile(runj_path):
        runj = json.load(open(runj_path, encoding="utf-8"))
    endw, enddetail, endcode = (parsed["end"] if parsed["end"] else (None, None, None))
    run = {"returncode": runj.get("returncode", endcode),
           "wall_seconds": runj.get("wall_seconds"),
           "solver_seconds": parsed["solver_seconds"],
           "end": {"word": endw, "detail": enddetail, "exit_code": endcode},
           "n_steps": n_steps,
           "permissive_lines": parsed["permissive"],
           "permissive_non_mach": parsed["permissive_non_mach"]}
    gpu = runj.get("gpu") or {"before": None, "after": None, "shared": None}

    its = parsed["iters"]
    if its:
        mx = max(its, key=lambda d: d["M"] if d["M"] == d["M"] else float("-inf"))
        residuals = {"first": its[0]["U_res"], "last": its[-1]["U_res"], "n": len(its)}
        mach = {"last": its[-1]["M"], "last_cell": its[-1]["cell"],
                "max": mx["M"], "max_step": mx["step"]}
    else:
        residuals = {"first": None, "last": None, "n": 0}
        mach = {"last": None, "last_cell": None, "max": None, "max_step": None}

    binfo = dict(wj.get("binary") or {"path": None, "sha256": None})
    if not binfo.get("sha256"):
        bp, bs = binary_info()
        if bs:
            binfo = {"path": binfo.get("path") or bp, "sha256": bs}
    bpath, _ = binary_info()
    command = wj.get("command") or solve_command(
        bpath, float(params["end_time"]), dt, float(params["write_interval"]))
    lastbal = rows[-1]["balance"] if rows else {"D_front_N": None, "D_rear_N": None,
                                                "front_share": None}
    sol = {
        "tool": wj.get("tool", TOOL),
        "version": wj.get("version", VERSION),
        "mesh_dir": wj.get("mesh_dir"),
        "geom_dir": wj.get("geom_dir"),
        "out_dir": wj.get("out_dir", out),
        "attribution": wj.get("attribution"),
        "binary": binfo,
        "command": command,
        "params": params,
        "wheels": wj.get("wheels"),
        "axles": axles,
        "gpu": gpu,
        "run": run,
        "residuals": residuals,
        "mach": mach,
        "final_forces": final,
        "balance": lastbal,
        "window": window,
        "criteria": cls["criteria"],
        "failed": cls["failed"],
        "class": cls["class"],
        "reason_id": cls["reason_id"],
        "xcheck": {"per_patch_rel": xrel, "max_rel": max_rel,
                   "limit": XCHECK_LIMIT,
                   "pass": (max_rel is not None and max_rel <= XCHECK_LIMIT)},
        "license_copied": bool(wj.get("license_copied")),
    }
    with open(os.path.join(out, "solve.json"), "w", encoding="utf-8", newline="\n") as fh:
        json.dump(sol, fh, ensure_ascii=False, indent=1)
        fh.write("\n")
    print("[promo-solve] post: class " + str(sol["class"]) +
          " (" + str(sol["reason_id"]) + "), Cd " + fnum(final["Cd"]) + ", Cl " +
          fnum(final["Cl"]) + ", snapshots " + str(len(rows)), flush=True)


# ---------------------------------------------------------------- selftest

def _box_tris(lo, hi):
    lo = np.asarray(lo, dtype=np.float64)
    hi = np.asarray(hi, dtype=np.float64)
    v = np.array([lo,
                  [hi[0], lo[1], lo[2]],
                  [hi[0], hi[1], lo[2]],
                  [lo[0], hi[1], lo[2]],
                  [lo[0], lo[1], hi[2]],
                  [hi[0], lo[1], hi[2]],
                  hi,
                  [lo[0], hi[1], hi[2]]])
    quads = [(0, 1, 2, 3), (4, 5, 6, 7), (0, 1, 5, 4),
             (3, 2, 6, 7), (1, 2, 6, 5), (0, 3, 7, 4)]
    tris = []
    for a, b, c, d in quads:
        tris.append((v[a], v[b], v[c]))
        tris.append((v[a], v[c], v[d]))
    return np.array(tris)


def _write_stl(path, tris):
    tris = np.asarray(tris, dtype=np.float64)
    n = tris.shape[0]
    normals = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    ln = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = normals / np.where(ln == 0, 1.0, ln)
    rec = np.zeros((n, 12), dtype="<f4")
    rec[:, 0:3] = normals
    rec[:, 3:6] = tris[:, 0]
    rec[:, 6:9] = tris[:, 1]
    rec[:, 9:12] = tris[:, 2]
    with open(path, "wb") as fh:
        fh.write(b"\0" * 80)
        fh.write(np.uint32(n).tobytes())
        for i in range(n):
            fh.write(rec[i].tobytes())
            fh.write(b"\0\0")


def _tiny_points():
    p = []
    for x in (0.0, 1.0, 2.0):
        for y in (0.0, 1.0):
            for z in (0.0, 1.0):
                p.append((x, y, z))
    return p


def _tiny_faces():
    return ([(4, 6, 7, 5), (0, 1, 3, 2), (8, 10, 11, 9), (0, 4, 5, 1), (4, 8, 9, 5),
             (2, 3, 7, 6), (6, 7, 11, 10), (0, 2, 6, 4), (4, 6, 10, 8),
             (1, 5, 7, 3), (5, 9, 11, 7)])


def _tiny_owner():
    return [0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]


def write_poly_mesh(case_dir, points, faces, owner, neighbour, patches,
                    flip_face=None):
    pmd = os.path.join(case_dir, "constant", "polyMesh")
    os.makedirs(pmd, exist_ok=True)
    pts = list(points)
    fcs = [list(f) for f in faces]
    if flip_face is not None:
        fcs[flip_face] = list(reversed(fcs[flip_face]))

    def lst(cls, obj, count, rows):
        return ("FoamFile\n{\n    version     2.0;\n    format      ascii;\n"
                "    class       " + cls + ";\n    object      " + obj + ";\n}\n"
                "// * * //\n\n" + str(count) + "\n(\n" + rows + ")\n\n")

    _write_text(os.path.join(pmd, "points"), lst("vectorField", "points", len(pts),
                "".join("(" + " ".join(fnum(c) for c in p) + ")\n" for p in pts)))
    _write_text(os.path.join(pmd, "faces"), lst("faceList", "faces", len(fcs),
                "".join(str(len(f)) + "(" + " ".join(str(i) for i in f) + ")\n"
                        for f in fcs)))
    _write_text(os.path.join(pmd, "owner"), lst("labelList", "owner", len(owner),
                "".join(str(o) + "\n" for o in owner)))
    _write_text(os.path.join(pmd, "neighbour"), lst("labelList", "neighbour",
                len(neighbour), "".join(str(o) + "\n" for o in neighbour)))
    txt = ("FoamFile\n{\n    version     2.0;\n    format      ascii;\n"
           "    class       polyBoundaryMesh;\n    object      boundary;\n}\n"
           "// * * //\n\n" + str(len(patches)) + "\n(\n")
    for name, nf, sf in patches:
        txt += ("    " + name + "\n    {\n        type            wall;\n"
                "        nFaces          " + str(nf) + ";\n"
                "        startFace       " + str(sf) + ";\n    }\n")
    txt += ")\n\n"
    _write_text(os.path.join(pmd, "boundary"), txt)


def load_tiny(case_dir):
    pmd = os.path.join(case_dir, "constant", "polyMesh")
    pts = read_points(os.path.join(pmd, "points"))
    arity, ptr, flat = read_faces(os.path.join(pmd, "faces"))
    owner = read_ints(os.path.join(pmd, "owner"))
    return PolyMesh(pts, arity, ptr, flat), owner


def _t1():
    U = 250.0 / 3.6
    boxes = {"front_left": ((0.0, 0.5, 0.0), (1.0, 1.0, 0.4)),
             "front_right": ((0.0, -1.0, 0.0), (1.0, -0.5, 0.4)),
             "rear_left": ((3.0, 0.5, 0.0), (4.0, 1.0, 0.4)),
             "rear_right": ((3.0, -1.0, 0.0), (4.0, -0.5, 0.4))}
    with tempfile.TemporaryDirectory() as td:
        tris = np.concatenate([_box_tris(lo, hi) for lo, hi in boxes.values()])
        _write_stl(os.path.join(td, "wheels.stl"), tris)
        _write_text(os.path.join(td, "sim_geom.json"), "{}\n")
        clusters, x_split = wheel_clusters(td, U)
        assert abs(x_split - 2.0) <= 1e-12, x_split
        zf = float(np.float32(0.4))
        for name, (lo, hi) in boxes.items():
            cl = clusters[name]
            assert abs(cl["cx"] - (lo[0] + hi[0]) / 2.0) <= 1e-12
            assert abs(cl["cy"] - (lo[1] + hi[1]) / 2.0) <= 1e-12
            assert abs(cl["cz"] - zf / 2.0) <= 1e-12
            assert abs(cl["R"] - zf / 2.0) <= 1e-12
            assert abs(cl["omega_y"] + U / (zf / 2.0)) <= 1e-9 * abs(cl["omega_y"])
            assert cl["n_tris"] == 12
        ax = axles_from(clusters, x_split)
        assert abs(ax["front_x"] - 0.5) <= 1e-12
        assert abs(ax["rear_x"] - 3.5) <= 1e-12
        assert abs(ax["wheelbase"] - 3.0) <= 1e-12
        # 3 clusters -> PS-GEOM
        with tempfile.TemporaryDirectory() as td3:
            tris3 = np.concatenate([_box_tris(*boxes[nm]) for nm in
                                    ("front_left", "front_right", "rear_left")])
            _write_stl(os.path.join(td3, "wheels.stl"), tris3)
            _write_text(os.path.join(td3, "sim_geom.json"), "{}\n")
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    wheel_clusters(td3, U)
            except SystemExit as e:
                assert e.code == 2, e.code
            else:
                raise AssertionError("3-cluster STL did not refuse PS-GEOM")


def _t2():
    U = 69.44444444444444
    cl = {"front_left": {"cx": 1.0, "cy": 0.7, "cz": 0.36, "R": 0.36,
                         "omega_y": -U / 0.36, "n_tris": 1}}
    for nm in ("front_right", "rear_left", "rear_right"):
        cl[nm] = dict(cl["front_left"])
    centres = np.array([[1.0, 0.7, 0.0], [1.0, 0.7, 0.72], [1.36, 0.7, 0.36]])
    v = wheel_face_velocity(centres, 2.0, cl)
    exp = np.array([[U, 0.0, 0.0], [-U, 0.0, 0.0], [0.0, 0.0, U]])
    assert np.allclose(v, exp, rtol=1e-9, atol=0.0), v


def _t3():
    U = 250.0 / 3.6
    nu = 1.5e-5
    k_in = 1.5 * (INTENSITY * U) ** 2
    omega_in = k_in / (NUT_RATIO * nu)
    assert abs(k_in / 0.18084490740740738 - 1.0) <= 1e-12, k_in
    assert abs(omega_in / 1205.6327160493825 - 1.0) <= 1e-12, omega_in


def _t4():
    p = np.array([1.0, 3.0])
    with tempfile.TemporaryDirectory() as td:
        case = os.path.join(td, "case")
        write_poly_mesh(case, _tiny_points(), _tiny_faces(), _tiny_owner(), [1],
                        [("body", 2, 1), ("side", 8, 3)])
        mesh, owner = load_tiny(case)
        f_body = patch_pressure_force(mesh, owner, p, 1, 2)
        assert np.allclose(f_body, [2.0 * RHO, 0.0, 0.0], rtol=0.0, atol=1e-12), f_body
        sf1, _ = mesh.face_sf_centres(1, 1)
        with tempfile.TemporaryDirectory() as td2:
            case2 = os.path.join(td2, "case")
            write_poly_mesh(case2, _tiny_points(), _tiny_faces(), _tiny_owner(), [1],
                            [("body", 2, 1), ("side", 8, 3)], flip_face=1)
            mesh2, _ = load_tiny(case2)
            sf2, _ = mesh2.face_sf_centres(1, 1)
            assert np.allclose(sf2, -sf1, rtol=0.0, atol=1e-12), (sf1, sf2)


def _t5():
    fx, rx = 1.2439, 4.6439
    b = balance(np.array([[0.0, 0.0, -1000.0]]), np.array([[1.2439, 0.0, 0.0]]), fx, rx)
    assert abs(b["D_front_N"] - 1000.0) <= 1e-9, b
    assert abs(b["D_rear_N"] - 0.0) <= 1e-9
    b = balance(np.array([[0.0, 0.0, -1000.0]]), np.array([[4.6439, 0.0, 0.0]]), fx, rx)
    assert abs(b["D_front_N"] - 0.0) <= 1e-9 and abs(b["D_rear_N"] - 1000.0) <= 1e-9
    b = balance(np.array([[0.0, 0.0, -1000.0]]), np.array([[2.9439, 0.0, 0.0]]), fx, rx)
    assert abs(b["D_front_N"] - 500.0) <= 1e-9 and abs(b["D_rear_N"] - 500.0) <= 1e-9
    assert abs(b["front_share"] - 0.5) <= 1e-12
    ff = np.array([[0.0, 0.0, -1000.0], [100.0, 0.0, 0.0]])
    fxx = np.array([[2.9439, 0.0, 0.0], [2.9439, 0.0, 1.0]])
    b = balance(ff, fxx, fx, rx)
    assert abs(b["D_front_N"] - 470.5882352941176) <= 1e-6, b
    assert abs(b["D_rear_N"] - (1000.0 - 470.5882352941176)) <= 1e-6, b


def _car_sums(parsed):
    cv = np.zeros(3)
    cp = np.zeros(3)
    for name in CAR:
        if name in parsed["forces"]:
            cv = cv + np.array(parsed["forces"][name]["F_visc"])
            cp = cp + np.array(parsed["forces"][name]["F_pres"])
    return cv, cp


def _t6():
    parsed = parse_log(FIXTURE_T4)
    assert [it["step"] for it in parsed["iters"]] == [0, 1, 19]
    it = parsed["iters"][-1]
    assert abs(it["U_res"] / 0.000921024 - 1.0) <= 1e-12
    assert abs(it["p_res"] / 0.001246 - 1.0) <= 1e-12
    assert abs(it["cont_err"] / 2.50188e-05 - 1.0) <= 1e-12
    assert abs(it["M"] / 0.391677 - 1.0) <= 1e-12 and it["cell"] == 615044
    assert abs(parsed["solver_seconds"] - 108.58) <= 1e-12
    cv, cp = _car_sums(parsed)
    assert np.allclose(cv, [4.98573, 0.004698033, -0.5231658], rtol=1e-9, atol=0.0)
    assert np.allclose(cp, [6392.835, 10.4486877, -4437.024], rtol=1e-9, atol=0.0)
    U = 250.0 / 3.6
    q = 0.5 * RHO * U * U
    assert abs(q / 2903.4047067901233 - 1.0) <= 1e-12
    Cd = (cv[0] + cp[0]) / (q * 1.4563)
    Cl = (cv[2] + cp[2]) / (q * 1.4563)
    assert abs(Cd / 1.513120937340805 - 1.0) <= 1e-9, Cd
    assert abs(Cl / -1.0495051065629546 - 1.0) <= 1e-9, Cl
    word, detail, code = parsed["end"]
    assert word == "budget" and code == 0
    assert detail == "endTime 0.01 s reached in 20 steps"
    assert parsed["permissive"] == 1 and parsed["permissive_non_mach"] == 0


def _t7():
    parsed = parse_log(FIXTURE_T4)
    steps = [0, 1, 19]
    res = classify(parsed, steps, [1.5, 1.5, 1.5], [-1.0, -1.0, -1.0])
    assert res["class"] == "unsteady" and res["reason_id"] == "PS-UNSTEADY", res
    assert res["failed"] == ["U_decades", "p_decades", "cont_err"], res["failed"]
    u_dec = math.log10(parsed["iters"][0]["U_res"] / parsed["iters"][-1]["U_res"])
    assert abs(res["criteria"]["U_decades"]["value"] - u_dec) <= 1e-9 * max(1.0, abs(u_dec))
    res2 = classify(parsed, steps, [1.5, 1.6, 1.5], [-1.0, -1.0, -1.0])
    assert "Cd_rel_change" in res2["failed"]
    assert abs(res2["criteria"]["Cd_rel_change"]["value"] - 0.06666666666666667) \
        <= 1e-12, res2["criteria"]["Cd_rel_change"]
    end_old = "run ended: budget | endTime 0.01 s reached in 20 steps | exit code 0"
    text_div = FIXTURE_T4.replace(end_old, "run ended: diverged | x | exit code 2")
    res3 = classify(parse_log(text_div), steps, [1.5], [-1.0])
    assert res3["class"] == "diverged" and res3["reason_id"] == "PS-DIVERGED"
    text_gamg = FIXTURE_T4 + "[ofgpu] -permissive: solver GAMG was requested\n"
    res4 = classify(parse_log(text_gamg), steps, [1.5], [-1.0])
    assert res4["class"] == "refused" and res4["reason_id"] == "PS-PERMISSIVE"
    text_noend = FIXTURE_T4.replace(end_old + "\n", "")
    res5 = classify(parse_log(text_noend), steps, [1.5], [-1.0])
    assert res5["class"] == "refused" and res5["reason_id"] == "PS-LOG"


def _refused(fn):
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            fn()
    except SystemExit as e:
        assert e.code == 2, e.code
        assert "refused:" in buf.getvalue(), buf.getvalue()
        return buf.getvalue()
    raise AssertionError("did not refuse")


def _t8():
    with tempfile.TemporaryDirectory() as td:
        _refused(lambda: cmd_post(argparse.Namespace(out=os.path.join(td, "nope")))) \
            .index("PS-OUT")
        out_in_repo = os.path.join(REPO, "_solve_selftest_out_should_not_exist")
        _refused(lambda: cmd_write(argparse.Namespace(
            mesh=td, geom=td, out=out_in_repo, end_time=0.5, dt=0.0005,
            write_interval=0.025))).index("PS-OUT")
        assert not os.path.exists(out_in_repo)
        mesh = os.path.join(td, "mesh")
        pmd = os.path.join(mesh, "case", "constant", "polyMesh")
        os.makedirs(pmd)
        for f in ("boundary", "faces", "owner", "neighbour", "points"):
            _write_text(os.path.join(pmd, f), "1\n(\n0)\n\n")
        _write_text(os.path.join(mesh, "tunnel_mesh.json"), "{}\n")
        out = os.path.join(td, "out")
        _refused(lambda: cmd_write(argparse.Namespace(
            mesh=mesh, geom=mesh, out=out, end_time=0.5, dt=0.0005,
            write_interval=0.025))).index("PS-MESH")


SOLVE_KEYS = {"tool", "version", "mesh_dir", "geom_dir", "out_dir", "attribution",
              "binary", "command", "params", "wheels", "axles", "gpu", "run",
              "residuals", "mach", "final_forces", "balance", "window", "criteria",
              "failed", "class", "reason_id", "xcheck", "license_copied"}


def _t9():
    U = 250.0 / 3.6
    with tempfile.TemporaryDirectory() as td:
        out = os.path.join(td, "out")
        case = os.path.join(out, "case")
        write_poly_mesh(case, _tiny_points(), _tiny_faces(), _tiny_owner(), [1],
                        [("body", 2, 1), ("side", 8, 3)])
        os.makedirs(os.path.join(case, "0.01"))
        _write_text(os.path.join(case, "0.01", "p"),
                    "FoamFile\n{\n    version     2.0;\n    format      ascii;\n"
                    "    class       volScalarField;\n    object      p;\n}\n"
                    "dimensions      [0 2 -2 0 0 0 0];\n"
                    "internalField   nonuniform List<scalar> 2\n(\n1\n3\n)\n;\n\n"
                    "boundaryField\n{\n}\n\n")
        _write_text(os.path.join(out, "solve.log"), FIXTURE_T4)
        k_in = 1.5 * (INTENSITY * U) ** 2
        wj = {"tool": TOOL, "version": VERSION, "mesh_dir": "m", "geom_dir": "g",
              "out_dir": out, "attribution": "test",
              "params": {"speed_kmh": 250.0, "U": U, "nu": 1.5e-5, "rho": RHO,
                         "A_ref": 1.4563, "q": 0.5 * RHO * U * U, "dt": 0.0005,
                         "end_time": 0.5, "write_interval": 0.025,
                         "intensity": INTENSITY, "nut_ratio": NUT_RATIO,
                         "k_in": k_in, "omega_in": k_in / (NUT_RATIO * 1.5e-5)},
              "wheels": [{"name": nm, "cx": 0.0, "cy": 0.0, "cz": 0.0, "R": 0.36,
                          "omega_y": -U / 0.36, "n_tris": 1} for nm in WHEEL_ORDER],
              "axles": {"x_split": 2.9439, "front_x": 1.2439, "rear_x": 4.6439,
                        "wheelbase": 3.4},
              "binary": {"path": "b", "sha256": "s"},
              "command": ["b"],
              "license_copied": True}
        _write_text(os.path.join(out, "write.json"),
                    json.dumps(wj, ensure_ascii=False, indent=1) + "\n")
        with contextlib.redirect_stdout(io.StringIO()):
            cmd_post(argparse.Namespace(out=out))
        sol = json.load(open(os.path.join(out, "solve.json"), encoding="utf-8"))
        assert set(sol.keys()) == SOLVE_KEYS, sorted(set(sol.keys()) ^ SOLVE_KEYS)
        res = open(os.path.join(out, "residuals.csv"), encoding="utf-8").read().splitlines()
        assert len(res) == 4 and res[0] == "step,U_res,p_res,cont_err,M_max,M_cell", res
        fcs = open(os.path.join(out, "forces.csv"), encoding="utf-8").read().splitlines()
        assert len(fcs) == 2, fcs
        assert fcs[0] == ("time,step,Fx,Fy,Fz,Cd_p,Cl_p,D_front_N,D_rear_N,"
                          "front_share")
        row = fcs[1].split(",")
        assert row[0] == "0.01" and row[1] == "20", row
        assert abs(float(row[2]) - 2.0 * RHO) <= 1e-9, row
        assert sol["class"] == "unsteady"
        assert sol["run"]["n_steps"] == 20
        assert sol["final_forces"]["per_patch"]["ground"]["F_pres"] == \
            [0.0, 0.0, -37305.9]


def selftest():
    t0 = time.monotonic()
    tests = [("wheels", _t1), ("wheel velocity", _t2), ("inflow turbulence", _t3),
             ("force", _t4), ("balance", _t5), ("parse", _t6), ("classify", _t7),
             ("refusals", _t8), ("json", _t9)]
    n_ok = 0
    for i, (name, fn) in enumerate(tests, 1):
        try:
            fn()
        except SystemExit as e:
            print("[fail] T%d %s: unexpected refuse (exit %s)" % (i, name, e.code),
                  flush=True)
        except Exception as e:
            print("[fail] T%d %s: %r" % (i, name, e), flush=True)
        else:
            print("[ok] T%d %s" % (i, name), flush=True)
            n_ok += 1
    print("SELFTEST %s %d/%d (%.1f s)" % ("PASS" if n_ok == len(tests) else "FAIL",
                                          n_ok, len(tests),
                                          time.monotonic() - t0), flush=True)
    return 0 if n_ok == len(tests) else 1


def main(argv=None):
    ap = argparse.ArgumentParser(prog="solve.py", description=__doc__.splitlines()[0])
    ap.add_argument("--selftest", action="store_true")
    sub = ap.add_subparsers(dest="cmd")
    for name in ("write", "run", "post"):
        sp = sub.add_parser(name)
        sp.add_argument("--mesh")
        sp.add_argument("--geom")
        sp.add_argument("--out", required=True)
        if name != "post":
            sp.add_argument("--end-time", type=float, default=0.5)
            sp.add_argument("--dt", type=float, default=0.0005)
            sp.add_argument("--write-interval", type=float, default=0.025)
        if name == "run":
            sp.add_argument("--timeout", type=float, default=0.0)
    args = ap.parse_args(argv)
    if args.selftest:
        return selftest()
    if args.cmd == "write":
        cmd_write(args)
    elif args.cmd == "run":
        cmd_run(args)
    elif args.cmd == "post":
        cmd_post(args)
    else:
        ap.print_help()
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
