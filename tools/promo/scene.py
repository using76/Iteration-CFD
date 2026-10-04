#!/usr/bin/env python3
# meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
# Source-available, not Open Source. See LICENSE at the repository root.
# No GPL-licensed source was consulted.

"""A reusable Blender studio scene for the F1 2026 promo render model.

The scene holds the imported car with the materials car_paint and car_field,
a reflective studio floor, HDRI-free studio lights (a dark world plus five
area lights), the 50 mm cameras cam_hero, cam_side and cam_top_rear aimed at
the car centre, and the turntable (an empty named turntable carrying
cam_turntable through one full orbit over --turntable-frames frames). Three
field importers bring CFD results onto it and the script renders 1920x1080
stills that pass their own gate:

- apply_surface_field maps a Cp sample onto the car vertices by the
  nearest sim point (mathutils KD-tree) and paints the car with car_field;
- add_streamlines sweeps the speed polylines into open tube meshes;
- add_slice builds the mid-plane speed slice as quads with NaN holes.

The importers read these .npz formats:

| file | key | shape / type | meaning |
|---|---|---|---|
| surface field | points | (n,3) float | sim-surface sample points (face centres), metres, same frame as the placed car |
| surface field | values | (n,) float | Cp at those points |
| streamlines | points | (m,3) float | all polyline points, concatenated |
| streamlines | speed | (m,) float | |U| at those points, m/s |
| streamlines | offsets | (k+1,) int | polyline j = points[offsets[j]:offsets[j+1]]; offsets[0] == 0, offsets[-1] == m, every polyline has >= 2 points (strictly increasing by >= 2) |
| slice | x | (nx,) float | strictly increasing, nx >= 2 |
| slice | z | (nz,) float | strictly increasing, nz >= 2 |
| slice | y0 | () float | the plane y = y0 |
| slice | speed | (nz,nx) float | |U|, NaN where not fluid |

With --synthetic the script derives synthetic test fields from
sim_surface.stl (Cp as the cosine of the triangle normal against +x,
Gaussian-bump streamline tubes over the car, a speed slice carved open over
the body), writes them as .npz under <out>/synthetic and loads them back
through the same importers.

Run (Blender 5.1, headless):

blender -b --factory-startup --python-exit-code 1 --python tools/promo/scene.py -- --glb GLB --sim-geom DIR --out DIR [--synthetic | --cp NPZ --streamlines NPZ --slice NPZ]

CC BY 4.0 rule: the render model and every file made from it (renders, npz,
.blend) are never committed to the repository; they stay under --out, the
model's LICENSE.txt is copied beside the outputs, and the attribution line
goes in the video credits.
"""

import argparse
import json
import math
import os
import shutil
import struct
import sys
import tempfile
import time

import numpy as np
import bpy
from mathutils import Matrix, Vector, kdtree

TOOL = "tools/promo/scene.py"
NAN_RGBA = (0.18, 0.18, 0.18, 1.0)
CP_RAMP = [
    (0.0, 0.02, 0.10, 0.60),
    (0.25, 0.10, 0.45, 0.95),
    (0.5, 0.90, 0.90, 0.90),
    (0.75, 0.95, 0.40, 0.08),
    (1.0, 0.60, 0.02, 0.02),
]
SPEED_RAMP = [
    (0.0, 0.05, 0.03, 0.30),
    (0.25, 0.05, 0.30, 0.75),
    (0.5, 0.05, 0.70, 0.55),
    (0.75, 0.85, 0.80, 0.10),
    (1.0, 0.95, 0.20, 0.05),
]
U_SYNTH = 250.0 / 3.6
TT_DIR = (-0.70, -0.62, 0.28)
TT_DIST = 8.5
STILLS_TABLE = {
    "hero": {"car": "field_if_cp", "streamlines": False, "slice": False},
    "side": {"car": "paint", "streamlines": True, "slice": False},
    "top_rear": {"car": "paint", "streamlines": True, "slice": True},
}
MATERIALS = {}


class SceneRefusal(Exception):
    def __init__(self, code, message):
        self.code = code
        self.message = message
        super().__init__(code + ": " + message)


def say(msg):
    print("[scene] " + msg, flush=True)


def read_stl_bin(path):
    """Binary STL -> (n, 3, 3) float64; the size must be exactly 84 + 50 n."""
    size = os.path.getsize(path)
    with open(path, "rb") as f:
        head = f.read(84)
        if len(head) < 84:
            raise SceneRefusal("SC-STL", "file shorter than the 84-byte STL header: " + path)
        (n,) = struct.unpack("<I", head[80:84])
        if size != 84 + 50 * n:
            raise SceneRefusal("SC-STL", "size " + str(size) + " != 84 + 50 * " + str(n) + " (" + path + ")")
        raw = f.read(n * 50)
    if n == 0:
        return np.zeros((0, 3, 3), dtype=np.float64)
    tris = np.frombuffer(raw, dtype=np.uint8).reshape(n, 50)[:, 12:48].copy().view("<f4")
    return tris.reshape(n, 3, 3).astype(np.float64)


def synthetic_cp(tris):
    """Centroids and cos-of-angle-to-x Cp of a triangle soup (float64)."""
    a, b, c = tris[:, 0], tris[:, 1], tris[:, 2]
    centroids = tris.mean(axis=1)
    nrm = np.cross(b - a, c - a)
    with np.errstate(invalid="ignore", divide="ignore"):
        cp = nrm[:, 0] / np.linalg.norm(nrm, axis=1)
    return centroids, cp


def ramp_colors(values, lo, hi, ramp):
    """Map values through a piecewise-linear colour ramp -> (n, 4) float32."""
    if hi <= lo:
        raise SceneRefusal("SC-RANGE", "hi " + str(hi) + " <= lo " + str(lo))
    vals = np.asarray(values, dtype=np.float64).ravel()
    out = np.empty((vals.shape[0], 4), dtype=np.float32)
    out[:] = NAN_RGBA
    finite = np.isfinite(vals)
    t = np.clip((vals[finite] - lo) / (hi - lo), 0.0, 1.0)
    xs = np.array([p[0] for p in ramp], dtype=np.float64)
    for ch in range(3):
        ys = np.array([p[1 + ch] for p in ramp], dtype=np.float64)
        out[finite, ch] = np.interp(t, xs, ys)
    out[:, 3] = 1.0
    return out


def nearest_values(query, points, values):
    """Nearest sample per query row through one balanced mathutils KD-tree."""
    kd = kdtree.KDTree(len(points))
    for i in range(len(points)):
        kd.insert(points[i], i)
    kd.balance()
    n = len(query)
    vals = np.empty(n, dtype=np.float64)
    dist = np.empty(n, dtype=np.float64)
    idx = np.empty(n, dtype=np.int64)
    for j in range(n):
        co, i, d = kd.find(query[j])
        vals[j] = values[i]
        dist[j] = d
        idx[j] = i
    return vals, dist, idx


def tube_mesh(points, offsets, radius, sides):
    """The TUBE RULE: one ring of `sides` vertices per polyline point, quads
    between consecutive points of the same polyline, no end caps; src[v] is
    the global polyline-point index the vertex belongs to."""
    points = np.asarray(points, dtype=np.float64)
    offsets = [int(v) for v in offsets]
    m = points.shape[0]
    n_faces = sum(offsets[j + 1] - offsets[j] - 1 for j in range(len(offsets) - 1)) * sides
    V = np.empty((m * sides, 3), dtype=np.float64)
    F = np.empty((n_faces, 4), dtype=np.int64)
    src = np.empty(m * sides, dtype=np.int64)
    up = np.array([0.0, 0.0, 1.0])
    alt = np.array([0.0, 1.0, 0.0])
    fi = 0
    for j in range(len(offsets) - 1):
        s0, s1 = offsets[j], offsets[j + 1]
        p = points[s0:s1]
        mm = s1 - s0
        tang = np.empty((mm, 3), dtype=np.float64)
        tang[0] = p[1] - p[0]
        tang[mm - 1] = p[mm - 1] - p[mm - 2]
        if mm > 2:
            tang[1:mm - 1] = p[2:mm] - p[0:mm - 2]
        for i in range(mm):
            t = tang[i] / np.linalg.norm(tang[i])
            a = up if abs(float(np.dot(t, up))) < 0.9 else alt
            n1 = np.cross(t, a)
            n1 = n1 / np.linalg.norm(n1)
            n2 = np.cross(t, n1)
            g = s0 + i
            for s in range(sides):
                ang = 2.0 * math.pi * s / sides
                V[g * sides + s] = p[i] + radius * (math.cos(ang) * n1 + math.sin(ang) * n2)
                src[g * sides + s] = g
            if i + 1 < mm:
                gn = g + 1
                for s in range(sides):
                    F[fi] = (g * sides + s, g * sides + (s + 1) % sides,
                             gn * sides + (s + 1) % sides, gn * sides + s)
                    fi += 1
    return V, F, src


def slice_mesh(x, z, y0, speed):
    """Grid slice at y = y0; one quad per cell whose four corner speeds are
    finite (NaN keeps the hole open). Vertex k*nx + i = (x[i], y0, z[k])."""
    x = np.asarray(x, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    speed = np.asarray(speed, dtype=np.float64)
    nx, nz = x.shape[0], z.shape[0]
    V = np.empty((nx * nz, 3), dtype=np.float64)
    for k in range(nz):
        V[k * nx:(k + 1) * nx, 0] = x
        V[k * nx:(k + 1) * nx, 1] = y0
        V[k * nx:(k + 1) * nx, 2] = z[k]
    faces = []
    for k in range(nz - 1):
        for i in range(nx - 1):
            corners = (speed[k, i], speed[k, i + 1], speed[k + 1, i + 1], speed[k + 1, i])
            if all(np.isfinite(v) for v in corners):
                faces.append((k * nx + i, k * nx + i + 1, (k + 1) * nx + i + 1, (k + 1) * nx + i))
    F = np.array(faces, dtype=np.int64).reshape(-1, 4)
    return V, F


def turntable_angle(frame, n_frames, start=1):
    return 2.0 * math.pi * (frame - start) / n_frames


def _require(path):
    if not os.path.isfile(path):
        raise SceneRefusal("SC-INPUT", "npz not found: " + path)


def load_surface_field(path):
    """Surface field npz -> (points (n,3), values (n,)), validated."""
    _require(path)
    with np.load(path) as z:
        for k in ("points", "values"):
            if k not in z.files:
                raise SceneRefusal("SC-FIELD", "surface field npz missing key: " + k)
        points = np.asarray(z["points"], dtype=np.float64)
        values = np.asarray(z["values"], dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise SceneRefusal("SC-FIELD", "points shape " + str(points.shape) + " != (n, 3)")
    if values.shape != (points.shape[0],):
        raise SceneRefusal("SC-FIELD", "values shape " + str(values.shape) + " != (" + str(points.shape[0]) + ",)")
    return points, values


def load_streamlines(path):
    """Streamlines npz -> (points (m,3), speed (m,), offsets (k+1,))."""
    _require(path)
    with np.load(path) as z:
        for k in ("points", "speed", "offsets"):
            if k not in z.files:
                raise SceneRefusal("SC-FIELD", "streamlines npz missing key: " + k)
        points = np.asarray(z["points"], dtype=np.float64)
        speed = np.asarray(z["speed"], dtype=np.float64)
        offsets = np.asarray(z["offsets"])
    if points.ndim != 2 or points.shape[1] != 3:
        raise SceneRefusal("SC-FIELD", "points shape " + str(points.shape) + " != (m, 3)")
    m = points.shape[0]
    if speed.shape != (m,):
        raise SceneRefusal("SC-FIELD", "speed shape " + str(speed.shape) + " != (" + str(m) + ",)")
    if offsets.ndim != 1 or not np.issubdtype(offsets.dtype, np.integer):
        raise SceneRefusal("SC-FIELD", "offsets must be a 1-D integer array, got shape " + str(offsets.shape))
    offsets = offsets.astype(np.int64)
    if offsets.size < 2 or int(offsets[0]) != 0 or int(offsets[-1]) != m:
        raise SceneRefusal("SC-FIELD", "offsets must start at 0 and end at " + str(m) + ", got " + str(offsets[[0, -1]].tolist()))
    steps = np.diff(offsets)
    if not np.all(steps >= 2):
        raise SceneRefusal("SC-FIELD", "every polyline needs >= 2 points; smallest offsets step is " + str(int(steps.min())))
    return points, speed, offsets


def load_slice(path):
    """Slice npz -> (x (nx,), z (nz,), y0 float, speed (nz,nx))."""
    _require(path)
    with np.load(path) as z:
        for k in ("x", "z", "y0", "speed"):
            if k not in z.files:
                raise SceneRefusal("SC-FIELD", "slice npz missing key: " + k)
        x = np.asarray(z["x"], dtype=np.float64)
        zz = np.asarray(z["z"], dtype=np.float64)
        y0 = np.asarray(z["y0"], dtype=np.float64)
        speed = np.asarray(z["speed"], dtype=np.float64)
    if x.ndim != 1 or x.size < 2:
        raise SceneRefusal("SC-FIELD", "x shape " + str(x.shape) + " != (nx,) with nx >= 2")
    if zz.ndim != 1 or zz.size < 2:
        raise SceneRefusal("SC-FIELD", "z shape " + str(zz.shape) + " != (nz,) with nz >= 2")
    if not np.all(np.diff(x) > 0):
        raise SceneRefusal("SC-FIELD", "x must be strictly increasing")
    if not np.all(np.diff(zz) > 0):
        raise SceneRefusal("SC-FIELD", "z must be strictly increasing")
    if y0.shape != ():
        raise SceneRefusal("SC-FIELD", "y0 shape " + str(y0.shape) + " != ()")
    if speed.shape != (zz.size, x.size):
        raise SceneRefusal("SC-FIELD", "speed shape " + str(speed.shape) + " != (nz, nx) = " + str((zz.size, x.size)))
    return x, zz, float(y0), speed


def synthetic_fields(sim_stl, out_dir):
    """Write cp.npz, streamlines.npz, slice.npz with the synthetic test fields."""
    os.makedirs(out_dir, exist_ok=True)
    centroids, cp = synthetic_cp(read_stl_bin(sim_stl))
    cp_path = os.path.join(out_dir, "cp.npz")
    np.savez(cp_path, points=centroids, values=cp)

    xs = np.linspace(-3.0, 12.0, 151)
    g = np.exp(-((xs - 2.7) / 1.2) ** 2)
    pts = []
    speeds = []
    offsets = [0]
    for z0 in np.linspace(0.15, 1.2, 6):
        for y0 in np.linspace(-1.0, 1.0, 8):
            y = y0 * (1.0 + 0.25 * g)
            z = z0 + 0.3 * g
            speed = U_SYNTH * (1.0 + 0.3 * g * (z0 - 0.6))
            pts.append(np.stack([xs, y, z], axis=1))
            speeds.append(speed)
            offsets.append(offsets[-1] + xs.shape[0])
    sl_path = os.path.join(out_dir, "streamlines.npz")
    np.savez(sl_path, points=np.concatenate(pts, axis=0), speed=np.concatenate(speeds, axis=0),
             offsets=np.array(offsets, dtype=np.int64))

    xs2 = np.linspace(-3.0, 12.0, 301)
    zs2 = np.linspace(0.0, 3.0, 61)
    X, Z = np.meshgrid(xs2, zs2)
    h = np.where(X >= 0.0, np.exp(-np.maximum(X - 5.394, 0.0) / 4.0), 0.0)
    speed2 = U_SYNTH * (1.0 - 0.6 * np.exp(-((Z - 0.5) / 0.6) ** 2) * h)
    speed2 = np.where((X > -0.025) & (X < 5.375) & (Z < 0.975), np.nan, speed2)
    slc_path = os.path.join(out_dir, "slice.npz")
    np.savez(slc_path, x=xs2, z=zs2, y0=np.array(0.0), speed=speed2)
    return {"cp": cp_path, "streamlines": sl_path, "slice": slc_path}


def build_arg_parser():
    ap = argparse.ArgumentParser(prog=TOOL, description="Blender studio scene for the F1 promo.")
    ap.add_argument("--glb")
    ap.add_argument("--sim-geom", dest="sim_geom")
    ap.add_argument("--out", dest="out")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--cp")
    ap.add_argument("--streamlines")
    ap.add_argument("--slice")
    ap.add_argument("--cp-range", dest="cp_range", nargs=2, type=float, default=[-1.0, 1.0])
    ap.add_argument("--speed-range", dest="speed_range", nargs=2, type=float, default=None)
    ap.add_argument("--samples", type=int, default=32)
    ap.add_argument("--device", choices=["CPU", "GPU"], default="CPU")
    ap.add_argument("--turntable-frames", dest="turntable_frames", type=int, default=240)
    ap.add_argument("--tube-radius", dest="tube_radius", type=float, default=0.012)
    ap.add_argument("--tube-sides", dest="tube_sides", type=int, default=8)
    ap.add_argument("--stills", default="hero,side,top_rear")
    ap.add_argument("--save-blend", dest="save_blend", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    return ap


def validate_args(args):
    if not args.selftest:
        if not args.glb:
            raise SceneRefusal("SC-INPUT", "--glb is required")
        if not args.sim_geom:
            raise SceneRefusal("SC-INPUT", "--sim-geom is required")
        if not args.out:
            raise SceneRefusal("SC-INPUT", "--out is required")
    if args.glb and not os.path.isfile(args.glb):
        raise SceneRefusal("SC-INPUT", "glb not found: " + args.glb)
    if args.sim_geom:
        if not os.path.isdir(args.sim_geom):
            raise SceneRefusal("SC-INPUT", "sim-geom dir not found: " + args.sim_geom)
        for fname in ("sim_geom.json", "LICENSE.txt", "sim_surface.stl"):
            p = os.path.join(args.sim_geom, fname)
            if not os.path.isfile(p):
                raise SceneRefusal("SC-INPUT", "missing file in --sim-geom: " + p)
    for p in (args.cp, args.streamlines, args.slice):
        if p and not os.path.isfile(p):
            raise SceneRefusal("SC-INPUT", "npz not found: " + p)
    for nm in [s.strip() for s in args.stills.split(",") if s.strip()]:
        if nm not in STILLS_TABLE:
            raise SceneRefusal("SC-INPUT", "unknown still name: " + nm)
    if args.out:
        os.makedirs(args.out, exist_ok=True)


def _t1():
    vals = [-1.0, 0.0, 1.0, 0.5, 2.0, -0.75, float("nan")]
    got = ramp_colors(vals, -1.0, 1.0, CP_RAMP)
    want = [
        (0.02, 0.10, 0.60), (0.90, 0.90, 0.90), (0.60, 0.02, 0.02),
        (0.95, 0.40, 0.08), (0.60, 0.02, 0.02), (0.06, 0.275, 0.775),
        NAN_RGBA,
    ]
    assert got.dtype == np.float32 and got.shape == (7, 4), (str(got.dtype), str(got.shape))
    for row, w in zip(got, want):
        for ch in range(3):
            assert abs(float(row[ch]) - w[ch]) <= 1e-6, str(row.tolist()) + " vs " + str(w)
        assert abs(float(row[3]) - 1.0) <= 1e-6, "alpha"
    try:
        ramp_colors([0.0], 1.0, 1.0, CP_RAMP)
    except SceneRefusal as e:
        assert e.code == "SC-RANGE", e.code
        return
    raise AssertionError("ramp_colors with hi == lo did not refuse")


def _newell(v):
    nrm = np.zeros(3)
    for i in range(len(v)):
        a = v[i]
        b = v[(i + 1) % len(v)]
        nrm[0] += (a[1] - b[1]) * (a[2] + b[2])
        nrm[1] += (a[2] - b[2]) * (a[0] + b[0])
        nrm[2] += (a[0] - b[0]) * (a[1] + b[1])
    return nrm


def _t2():
    pts = np.array([(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (2.0, 0.0, 0.0)], dtype=np.float64)
    V, F, src = tube_mesh(pts, [0, 3], 0.1, 4)
    assert V.shape == (12, 3) and F.shape == (8, 4), (str(V.shape), str(F.shape))
    assert src.tolist() == [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2], src.tolist()
    ring = [(0.0, -0.1, 0.0), (0.0, 0.0, -0.1), (0.0, 0.1, 0.0), (0.0, 0.0, 0.1)]
    for s, w in enumerate(ring):
        assert float(np.max(np.abs(V[s] - np.array(w)))) <= 1e-12, V[s].tolist()
    for f in F:
        verts = V[f]
        cen = verts.mean(axis=0)
        radial = cen - np.array([cen[0], 0.0, 0.0])
        assert float(np.dot(_newell(verts), radial)) > 0.0, "face not outward"
    pts2 = np.array([(0, 0, 0), (1, 0, 0), (5, 5, 5), (5, 5, 6), (5, 5, 7)], dtype=np.float64)
    V, F, src = tube_mesh(pts2, [0, 2, 5], 0.1, 6)
    assert V.shape == (30, 3) and F.shape == (18, 4), (str(V.shape), str(F.shape))
    for f in F:
        ids = set(f.tolist())
        assert ids <= set(range(12)) or ids <= set(range(12, 30)), "face spans two polylines"
    assert float(np.max(np.abs(V[12] - np.array([4.9, 5.0, 5.0])))) <= 1e-12, V[12].tolist()


def _t3():
    x = np.array([0.0, 1.0, 2.0, 3.0])
    z = np.array([0.0, 1.0, 2.0])
    spd = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, np.nan]], dtype=np.float64)
    V, F = slice_mesh(x, z, 0.5, spd)
    assert V.shape == (12, 3) and F.shape == (5, 4), (str(V.shape), str(F.shape))
    for k in range(3):
        for i in range(4):
            want = np.array([x[i], 0.5, z[k]])
            assert float(np.max(np.abs(V[k * 4 + i] - want))) <= 1e-12, V[k * 4 + i].tolist()
    assert F[0].tolist() == [0, 1, 5, 4], F[0].tolist()
    quads = set(tuple(r) for r in F.tolist())
    assert (1, 2, 6, 5) in quads and (5, 6, 10, 9) in quads, "kept quads"
    assert (6, 7, 11, 10) not in quads, "NaN quad must be dropped"


def _t4():
    q = [(0.1, 0, 0), (0.9, 0.05, 0), (0.1, 0.8, 0), (5, 4, 5)]
    p = [(0, 0, 0), (1, 0, 0), (0, 1, 0)]
    vals, dist, idx = nearest_values(q, p, [10.0, 20.0, 30.0])
    assert vals.tolist() == [10.0, 20.0, 30.0, 20.0], vals.tolist()
    assert idx.tolist() == [0, 1, 2, 1], idx.tolist()
    assert abs(float(dist[0]) - 0.1) <= 1e-6, str(float(dist[0]))


def _t5():
    reset_scene()
    empty, cam = add_turntable((1.0, 2.0, 0.5), (-0.70, -0.62, 0.28), 8.5, 240)
    sc = bpy.context.scene
    assert sc.frame_end == 240, sc.frame_end
    for f, want in ((1, 0.0), (61, math.pi / 2), (121, math.pi), (240, 2.0 * math.pi * 239.0 / 240.0)):
        sc.frame_set(f)
        got = float(empty.rotation_euler.z)
        assert abs(got - want) <= 1e-6, (f, got, want)
    sc.frame_set(61)
    off = Vector((-0.70, -0.62, 0.28)).normalized() * 8.5
    want_cam = Vector((1.0, 2.0, 0.5)) + Matrix.Rotation(math.pi / 2.0, 4, "Z") @ off
    d = float((cam.matrix_world.translation - want_cam).length)
    assert d <= 1e-4, str(d)


def _t6():
    tris = np.array([
        [(0, 0, 0), (0, 1, 0), (0, 0, 1)],
        [(0, 0, 0), (0, 0, 1), (0, 1, 0)],
        [(0, 0, 0), (1, 0, 0), (0, 1, 0)],
    ], dtype=np.float64)
    cen, cp = synthetic_cp(tris)
    assert float(np.max(np.abs(cp - np.array([1.0, -1.0, 0.0])))) <= 1e-12, cp.tolist()
    assert float(np.max(np.abs(cen[0] - np.array([0.0, 1.0 / 3.0, 1.0 / 3.0])))) <= 1e-12, cen[0].tolist()
    tmp = tempfile.mkdtemp()
    payload = bytes(80) + struct.pack("<I", 2)
    for tri in tris[:2]:
        payload = payload + np.zeros(3, dtype="<f4").tobytes() + tri.astype("<f4").tobytes() + bytes(2)
    good = os.path.join(tmp, "good.stl")
    with open(good, "wb") as f:
        f.write(payload)
    short = os.path.join(tmp, "short.stl")
    with open(short, "wb") as f:
        f.write(payload + bytes(1))
    try:
        read_stl_bin(short)
    except SceneRefusal as e:
        assert e.code == "SC-STL", e.code
    else:
        raise AssertionError("short STL did not refuse")
    back = read_stl_bin(good)
    assert back.shape == (2, 3, 3), back.shape
    assert float(np.max(np.abs(back - tris[:2]))) <= 1e-6, "coordinates"


def _t7():
    tmp = tempfile.mkdtemp()
    p = os.path.join(tmp, "no_offsets.npz")
    np.savez(p, points=np.zeros((3, 3)), speed=np.zeros(3))
    try:
        load_streamlines(p)
    except SceneRefusal as e:
        assert e.code == "SC-FIELD", e.code
    else:
        raise AssertionError("missing offsets did not refuse")
    p = os.path.join(tmp, "one_point.npz")
    np.savez(p, points=np.zeros((3, 3)), speed=np.zeros(3), offsets=np.array([0, 1, 3], dtype=np.int64))
    try:
        load_streamlines(p)
    except SceneRefusal as e:
        assert e.code == "SC-FIELD", e.code
    else:
        raise AssertionError("1-point polyline did not refuse")
    p = os.path.join(tmp, "short_values.npz")
    np.savez(p, points=np.zeros((4, 3)), values=np.zeros(3))
    try:
        load_surface_field(p)
    except SceneRefusal as e:
        assert e.code == "SC-FIELD", e.code
    else:
        raise AssertionError("short values did not refuse")
    p = os.path.join(tmp, "transposed.npz")
    np.savez(p, x=np.array([0.0, 1.0, 2.0]), z=np.array([0.0, 1.0]), y0=np.array(0.0), speed=np.zeros((3, 2)))
    try:
        load_slice(p)
    except SceneRefusal as e:
        assert e.code == "SC-FIELD", e.code
    else:
        raise AssertionError("transposed slice speed did not refuse")
    p = os.path.join(tmp, "ok.npz")
    pts = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 1.0], [2.0, 0.0, 0.0], [3.0, 0.0, 1.0]])
    spd = np.array([1.0, 2.0, 3.0, 4.0])
    off = np.array([0, 2, 4], dtype=np.int64)
    np.savez(p, points=pts, speed=spd, offsets=off)
    p2, s2, o2 = load_streamlines(p)
    assert np.array_equal(p2, pts) and np.array_equal(s2, spd) and np.array_equal(o2, off), "round trip"


SELFTESTS = [("T1", _t1), ("T2", _t2), ("T3", _t3), ("T4", _t4), ("T5", _t5), ("T6", _t6), ("T7", _t7)]


def selftest():
    results = []
    for name, fn in SELFTESTS:
        try:
            fn()
            print("[ok] " + name, flush=True)
            results.append(True)
        except Exception as exc:
            print("[FAIL] " + name + ": " + repr(exc), flush=True)
            results.append(False)
    good = sum(1 for r in results if r)
    verdict = "PASS" if good == len(results) else "FAIL"
    print("SELFTEST " + verdict + " " + str(good) + "/7", flush=True)
    return 0 if good == len(results) else 1


def reset_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)


def import_render_model(glb, placement):
    """Import the GLB, join it, apply transforms, scale to metres, place it."""
    known = {o.name for o in bpy.data.objects}
    bpy.ops.import_scene.gltf(filepath=glb)
    meshes = [o for o in bpy.data.objects if o.type == "MESH" and o.name not in known]
    if not meshes:
        raise SceneRefusal("SC-INPUT", "glTF import produced no mesh objects: " + glb)
    bpy.ops.object.select_all(action="DESELECT")
    for o in meshes:
        o.select_set(True)
    bpy.context.view_layer.objects.active = meshes[0]
    bpy.ops.object.parent_clear(type="CLEAR_KEEP_TRANSFORM")
    bpy.ops.object.join()
    obj = bpy.context.view_layer.objects.active
    obj.name = "car"
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    s = float(placement["scale"])
    t = [float(v) for v in placement["translate_m"]]
    obj.data.transform(Matrix.Diagonal((s, s, s, 1.0)))
    obj.data.transform(Matrix.Translation(Vector(t)))
    return obj


def car_center(obj):
    """Centre and bounds of the vertex array (bound_box is stale after data.transform)."""
    me = obj.data
    n = len(me.vertices)
    co = np.empty(n * 3, dtype=np.float64)
    me.vertices.foreach_get("co", co)
    co = co.reshape(n, 3)
    lo = co.min(axis=0)
    hi = co.max(axis=0)
    return (lo + hi) / 2.0, lo, hi, co


def _principled(mat):
    return mat.node_tree.nodes["Principled BSDF"]


def _emissive_field(name, attr_name, strength, roughness):
    m = bpy.data.materials.new(name)
    b = _principled(m)
    attr = m.node_tree.nodes.new("ShaderNodeAttribute")
    attr.attribute_name = attr_name
    out = attr.outputs["Color"]
    m.node_tree.links.new(out, b.inputs["Base Color"])
    m.node_tree.links.new(out, b.inputs["Emission Color"])
    b.inputs["Emission Strength"].default_value = strength
    b.inputs["Roughness"].default_value = roughness
    return m


def build_materials():
    MATERIALS.clear()
    m = bpy.data.materials.new("car_paint")
    b = _principled(m)
    b.inputs["Base Color"].default_value = (0.30, 0.008, 0.008, 1.0)
    b.inputs["Metallic"].default_value = 0.5
    b.inputs["Roughness"].default_value = 0.35
    b.inputs["Coat Weight"].default_value = 1.0
    b.inputs["Coat Roughness"].default_value = 0.03
    MATERIALS["car_paint"] = m

    m = bpy.data.materials.new("car_field")
    b = _principled(m)
    attr = m.node_tree.nodes.new("ShaderNodeAttribute")
    attr.attribute_name = "Cp_rgb"
    m.node_tree.links.new(attr.outputs["Color"], b.inputs["Base Color"])
    b.inputs["Metallic"].default_value = 0.0
    b.inputs["Roughness"].default_value = 0.35
    b.inputs["Coat Weight"].default_value = 0.6
    b.inputs["Coat Roughness"].default_value = 0.05
    MATERIALS["car_field"] = m

    m = bpy.data.materials.new("studio_floor")
    b = _principled(m)
    b.inputs["Base Color"].default_value = (0.03, 0.03, 0.035, 1.0)
    b.inputs["Roughness"].default_value = 0.2
    MATERIALS["studio_floor"] = m

    MATERIALS["streamline_tube"] = _emissive_field("streamline_tube", "speed_rgb", 0.5, 0.4)
    MATERIALS["slice_speed"] = _emissive_field("slice_speed", "speed_rgb", 0.8, 0.5)
    return MATERIALS


def _aim_at(obj, target):
    obj.rotation_euler = (Vector(target) - obj.location).to_track_quat("-Z", "Y").to_euler()


def add_ground(material):
    me = bpy.data.meshes.new("ground")
    corners = [(-100.0, -100.0, 0.0), (100.0, -100.0, 0.0), (100.0, 100.0, 0.0), (-100.0, 100.0, 0.0)]
    me.from_pydata(corners, [], [(0, 1, 2, 3)])
    me.update()
    obj = bpy.data.objects.new("ground", me)
    bpy.context.scene.collection.objects.link(obj)
    me.materials.append(material)
    return obj


def add_world():
    w = bpy.data.worlds.new("studio_world")
    bpy.context.scene.world = w
    bg = w.node_tree.nodes["Background"]
    bg.inputs[0].default_value = (0.015, 0.015, 0.018, 1.0)
    bg.inputs[1].default_value = 1.0
    return w


def add_lights(center):
    table = [
        ("top", 7.0, 3.0, 900.0, (2.7, 0.0, 6.0)),
        ("key", 3.0, 3.0, 600.0, (-3.0, -5.0, 3.5)),
        ("fill_left", 6.0, 1.0, 250.0, (2.7, -7.0, 1.5)),
        ("fill_right", 6.0, 1.0, 250.0, (2.7, 7.0, 1.5)),
        ("rim", 2.0, 3.0, 500.0, (10.0, 2.0, 2.5)),
    ]
    for name, sx, sy, energy, loc in table:
        ld = bpy.data.lights.new(name, type="AREA")
        ld.shape = "RECTANGLE"
        ld.size = sx
        ld.size_y = sy
        ld.energy = energy
        obj = bpy.data.objects.new(name, ld)
        bpy.context.scene.collection.objects.link(obj)
        obj.location = loc
        _aim_at(obj, center)


def add_cameras(center):
    table = [
        ("cam_hero", (-0.70, -0.62, 0.28), 7.8),
        ("cam_side", (0.0, -1.0, 0.06), 9.4),
        ("cam_top_rear", (0.62, 0.42, 0.66), 11.0),
    ]
    for name, direction, distance in table:
        cd = bpy.data.cameras.new(name)
        cd.lens = 50.0
        obj = bpy.data.objects.new(name, cd)
        bpy.context.scene.collection.objects.link(obj)
        obj.location = Vector(center) + Vector(direction).normalized() * distance
        _aim_at(obj, center)


def add_turntable(center, direction, distance, n_frames):
    """The turntable: an empty keyed 0..2pi over n_frames + 1 frames, LINEAR,
    carrying cam_turntable at local offset normalize(direction) * distance."""
    empty = bpy.data.objects.new("turntable", None)
    bpy.context.scene.collection.objects.link(empty)
    empty.location = Vector(center)
    cd = bpy.data.cameras.new("cam_turntable")
    cd.lens = 50.0
    cam = bpy.data.objects.new("cam_turntable", cd)
    bpy.context.scene.collection.objects.link(cam)
    offset = Vector(direction).normalized() * distance
    cam.location = offset
    cam.rotation_euler = (-offset).to_track_quat("-Z", "Y").to_euler()
    cam.parent = empty
    empty.rotation_euler = (0.0, 0.0, 0.0)
    empty.keyframe_insert(data_path="rotation_euler", index=2, frame=1)
    empty.rotation_euler = (0.0, 0.0, 2.0 * math.pi)
    empty.keyframe_insert(data_path="rotation_euler", index=2, frame=n_frames + 1)
    empty.rotation_euler = (0.0, 0.0, 0.0)
    act = empty.animation_data.action
    cb = act.layers[0].strips[0].channelbag(act.slots[0])
    for fc in cb.fcurves:
        for kp in fc.keyframe_points:
            kp.interpolation = "LINEAR"
    sc = bpy.context.scene
    sc.frame_start = 1
    sc.frame_end = n_frames
    return empty, cam


def configure_render(samples, device):
    sc = bpy.context.scene
    sc.render.engine = "CYCLES"
    cycles = sc.cycles
    prefs = bpy.context.preferences.addons["cycles"].preferences
    if device == "GPU":
        prefs.compute_device_type = "OPTIX"
        prefs.get_devices()
        for dev in prefs.devices:
            if dev.type == "OPTIX":
                dev.use = True
        cycles.device = "GPU"
    else:
        prefs.compute_device_type = "NONE"
        cycles.device = "CPU"
    cycles.samples = int(samples)
    cycles.use_adaptive_sampling = True
    cycles.use_denoising = True
    cycles.denoiser = "OPENIMAGEDENOISE"
    cycles.max_bounces = 6
    sc.render.resolution_x = 1920
    sc.render.resolution_y = 1080
    sc.render.resolution_percentage = 100
    sc.render.image_settings.file_format = "PNG"
    sc.render.image_settings.color_mode = "RGB"
    sc.render.image_settings.color_depth = "8"
    sc.view_settings.view_transform = "AgX"


def apply_surface_field(car, points, values, lo, hi):
    """Map Cp onto the car vertices through the nearest sim point (KD-tree):
    point attribute Cp (FLOAT) and colour attribute Cp_rgb (FLOAT_COLOR)."""
    me = car.data
    n = len(me.vertices)
    co = np.empty(n * 3, dtype=np.float64)
    me.vertices.foreach_get("co", co)
    co = co.reshape(n, 3)
    vals, dist, _ = nearest_values(co, points, values)
    rgba = ramp_colors(vals, lo, hi, CP_RAMP)
    a = me.attributes.new("Cp", "FLOAT", "POINT")
    a.data.foreach_set("value", vals.astype(np.float32))
    c = me.color_attributes.new("Cp_rgb", "FLOAT_COLOR", "POINT")
    c.data.foreach_set("color", rgba.astype(np.float32).ravel())
    me.update()
    return {"n": int(n), "max_dist": float(dist.max()), "mean_dist": float(dist.mean())}


def set_car_material(car, which):
    if which not in ("paint", "field"):
        raise SceneRefusal("SC-INPUT", "unknown car material: " + which)
    me = car.data
    me.materials.clear()
    me.materials.append(MATERIALS["car_field" if which == "field" else "car_paint"])
    idx = np.zeros(len(me.polygons), dtype=np.int32)
    me.polygons.foreach_set("material_index", idx)
    me.update()


def _mesh_from(name, V, F):
    me = bpy.data.meshes.new(name)
    me.from_pydata(V.tolist(), [], F.tolist())
    me.update()
    return me


def _add_speed_attributes(me, values, lo, hi):
    a = me.attributes.new("speed", "FLOAT", "POINT")
    a.data.foreach_set("value", values.astype(np.float32))
    rgba = ramp_colors(values, lo, hi, SPEED_RAMP)
    c = me.color_attributes.new("speed_rgb", "FLOAT_COLOR", "POINT")
    c.data.foreach_set("color", rgba.astype(np.float32).ravel())
    me.update()


def add_streamlines(points, speed, offsets, lo, hi, radius, sides):
    """Sweep the speed polylines into tube meshes coloured by SPEED_RAMP."""
    V, F, src = tube_mesh(points, offsets, radius, sides)
    me = _mesh_from("streamlines", V, F)
    _add_speed_attributes(me, speed[src], lo, hi)
    obj = bpy.data.objects.new("streamlines", me)
    bpy.context.scene.collection.objects.link(obj)
    me.materials.append(MATERIALS["streamline_tube"])
    return obj


def add_slice(x, z, y0, speed, lo, hi):
    """Build the mid-plane speed slice; NaN vertices keep their hole open."""
    V, F = slice_mesh(x, z, y0, speed)
    me = _mesh_from("slice", V, F)
    _add_speed_attributes(me, speed.reshape(-1), lo, hi)
    obj = bpy.data.objects.new("slice", me)
    bpy.context.scene.collection.objects.link(obj)
    me.materials.append(MATERIALS["slice_speed"])
    return obj


def show_for(name, have_cp):
    row = STILLS_TABLE[name]
    car = "paint"
    if row["car"] == "field_if_cp" and have_cp:
        car = "field"
    return {"car": car, "streamlines": row["streamlines"], "slice": row["slice"]}


def render_still(name, path, show):
    """Render one still through cam_<name> per the STILLS table; wall seconds."""
    sc = bpy.context.scene
    t0 = time.time()
    sc.camera = bpy.data.objects["cam_" + name]
    set_car_material(bpy.data.objects["car"], show["car"])
    for key in ("streamlines", "slice"):
        obj = bpy.data.objects.get(key)
        if obj is not None:
            obj.hide_render = not show[key]
    sc.render.filepath = path
    bpy.ops.render.render(write_still=True)
    return time.time() - t0


def image_stats(path):
    """width, height and the population std of the pixel luminance of a PNG."""
    img = bpy.data.images.load(path)
    w, h = img.size
    channels = img.channels
    n = w * h
    px = np.empty(n * channels, dtype=np.float32)
    img.pixels.foreach_get(px)
    bpy.data.images.remove(img)
    px = px.reshape(n, channels)[:, :3].astype(np.float64)
    lum = 0.2126 * px[:, 0] + 0.7152 * px[:, 1] + 0.0722 * px[:, 2]
    return {"width": int(w), "height": int(h), "lum_std": float(lum.std())}


def _check_names(ctx):
    need = ["car", "ground", "top", "key", "fill_left", "fill_right", "rim",
            "cam_hero", "cam_side", "cam_top_rear", "cam_turntable", "turntable"]
    if ctx["sl_data"] is not None:
        need.append("streamlines")
    if ctx["slc_data"] is not None:
        need.append("slice")
    missing = [nm for nm in need if bpy.data.objects.get(nm) is None]
    if missing:
        return False, "missing " + ", ".join(missing)
    return True, str(len(need)) + " objects present"


def _check_turntable(ctx):
    empty = bpy.data.objects.get("turntable")
    cam = bpy.data.objects.get("cam_turntable")
    n = ctx["args"].turntable_frames
    sc = bpy.context.scene
    bad = []
    if sc.frame_start != 1:
        bad.append("frame_start " + str(sc.frame_start) + " != 1")
    if sc.frame_end != n:
        bad.append("frame_end " + str(sc.frame_end) + " != " + str(n))
    center = Vector(ctx["center"])
    offset = Vector(TT_DIR).normalized() * TT_DIST
    for f in (1, 1 + n // 4, 1 + n // 2):
        sc.frame_set(f)
        want = turntable_angle(f, n)
        got = float(empty.rotation_euler.z)
        if abs(got - want) > 1e-6:
            bad.append("frame " + str(f) + " rotation " + str(got) + " != " + str(want))
        want_cam = center + Matrix.Rotation(turntable_angle(f, n), 4, "Z") @ offset
        d = float((cam.matrix_world.translation - want_cam).length)
        if d > 1e-4:
            bad.append("frame " + str(f) + " cam off by " + str(d) + " m")
    sc.frame_set(1)
    if bad:
        return False, "; ".join(bad)
    return True, "frames 1/" + str(1 + n // 4) + "/" + str(1 + n // 2) + " exact"


def _check_cp_map(ctx):
    car = ctx["car_obj"]
    me = car.data
    attr = me.attributes.get("Cp")
    nv = len(me.vertices)
    if attr is None:
        return False, "attribute Cp missing"
    if len(attr.data) != nv:
        return False, "attribute Cp length " + str(len(attr.data)) + " != " + str(nv)
    stored = np.empty(nv, dtype=np.float32)
    attr.data.foreach_get("value", stored)
    points, values = ctx["cp"]
    co = ctx["car_co"]
    rng = np.random.default_rng(0)
    sample = rng.choice(nv, 500, replace=False)
    bad = 0
    for vi in sample:
        d = np.sqrt(np.sum((points - co[vi]) ** 2, axis=1))
        cands = np.nonzero(d <= d.min() + 1e-6)[0]
        v32 = stored[vi]
        if not any(np.float32(values[j]) == v32 for j in cands):
            bad += 1
    if bad:
        return False, str(bad) + " of 500 sampled vertices have no matching nearest sim point"
    return True, "500/500 vertices match their brute-force nearest sim point"


def _check_streamlines(ctx):
    pts, spd, offs = ctx["sl_data"]
    sides = ctx["args"].tube_sides
    obj = bpy.data.objects["streamlines"]
    nv = len(obj.data.vertices)
    nf = len(obj.data.polygons)
    attr = obj.data.attributes.get("speed")
    if attr is None:
        return False, "speed attribute missing"
    got = np.empty(nv, dtype=np.float32)
    attr.data.foreach_get("value", got)
    bad = []
    if nv != len(pts) * sides:
        bad.append("vertices " + str(nv) + " != " + str(len(pts) * sides))
    if nf != int(np.sum(np.diff(offs) - 1)) * sides:
        bad.append("faces " + str(nf) + " != " + str(int(np.sum(np.diff(offs) - 1)) * sides))
    for label, want in (("min", float(spd.min())), ("max", float(spd.max()))):
        gv = float(got.min()) if label == "min" else float(got.max())
        if abs(gv - want) > 1e-4 * max(abs(want), 1e-30):
            bad.append("speed " + label + " " + str(gv) + " != " + str(want))
    if bad:
        return False, "; ".join(bad)
    return True, str(nv) + " vertices, " + str(nf) + " faces, speed attribute ok"


def _check_slice(ctx):
    x, z, _y0, spd = ctx["slc_data"]
    obj = bpy.data.objects["slice"]
    nx, nz = x.shape[0], z.shape[0]
    nv = len(obj.data.vertices)
    nf = len(obj.data.polygons)
    fin = np.isfinite(spd)
    quads = fin[:-1, :-1] & fin[1:, :-1] & fin[1:, 1:] & fin[:-1, 1:]
    nf_want = int(quads.sum())
    bad = []
    if nv != nx * nz:
        bad.append("vertices " + str(nv) + " != " + str(nx * nz))
    if nf != nf_want:
        bad.append("faces " + str(nf) + " != " + str(nf_want))
    if bad:
        return False, "; ".join(bad)
    return True, str(nv) + " vertices, " + str(nf) + " faces"


def _check_renders(ctx):
    bad = []
    for nm, info in ctx["stills"].items():
        if not os.path.isfile(info["path"]):
            bad.append(nm + " png missing")
            continue
        if info["width"] != 1920 or info["height"] != 1080:
            bad.append(nm + " size " + str(info["width"]) + "x" + str(info["height"]))
        if info["seconds"] > 120:
            bad.append(nm + " took " + format(info["seconds"], ".1f") + " s")
        if not info["lum_std"] > 0.02:
            bad.append(nm + " lum_std " + format(info["lum_std"], ".4f"))
    if bad:
        return False, "; ".join(bad)
    return True, str(len(ctx["stills"])) + " stills ok"


def _check_license(ctx):
    p = os.path.join(ctx["args"].out, "LICENSE.txt")
    if not os.path.isfile(p):
        return False, p + " missing"
    if not (isinstance(ctx["attribution"], str) and ctx["attribution"]):
        return False, "attribution is not a non-empty string"
    return True, "LICENSE.txt copied, attribution present"


def checks(ctx):
    ck = {}
    ck["names"] = _check_names(ctx)
    ck["turntable"] = _check_turntable(ctx)
    if ctx["cp"] is not None:
        ck["cp_map"] = _check_cp_map(ctx)
    if ctx["sl_data"] is not None:
        ck["streamlines"] = _check_streamlines(ctx)
    if ctx["slc_data"] is not None:
        ck["slice"] = _check_slice(ctx)
    ck["renders"] = _check_renders(ctx)
    ck["license"] = _check_license(ctx)
    return ck


def judge(ck):
    reasons = [k + ": " + detail for k, (ok, detail) in ck.items() if not ok]
    return (len(reasons) == 0), reasons


def run(args):
    t_total = time.time()
    validate_args(args)
    with open(os.path.join(args.sim_geom, "sim_geom.json"), "r", encoding="utf-8") as f:
        sg = json.load(f)
    for k in ("placement", "source"):
        if k not in sg:
            raise SceneRefusal("SC-INPUT", "sim_geom.json missing key: " + k)
    placement = sg["placement"]
    for k in ("scale", "translate_m"):
        if k not in placement:
            raise SceneRefusal("SC-INPUT", "sim_geom.json placement missing key: " + k)
    attribution = str(sg["source"].get("attribution", ""))
    shutil.copyfile(os.path.join(args.sim_geom, "LICENSE.txt"), os.path.join(args.out, "LICENSE.txt"))
    say("inputs ok, attribution: " + attribution)

    t0 = time.time()
    reset_scene()
    car = import_render_model(args.glb, placement)
    center, bbox_lo, bbox_hi, car_co = car_center(car)
    say("car: " + str(len(car.data.vertices)) + " vertices, " + str(len(car.data.polygons)) + " polygons")
    build_materials()
    add_ground(MATERIALS["studio_floor"])
    add_world()
    add_lights(center)
    add_cameras(center)
    add_turntable(center, TT_DIR, TT_DIST, args.turntable_frames)
    configure_render(args.samples, args.device)
    t_build = time.time() - t0

    t0 = time.time()
    paths = {"cp": args.cp, "streamlines": args.streamlines, "slice": args.slice}
    if args.synthetic:
        synth_dir = os.path.join(args.out, "synthetic")
        paths = synthetic_fields(os.path.join(args.sim_geom, "sim_surface.stl"), synth_dir)
        say("synthetic fields written under " + synth_dir)
    cp_data = load_surface_field(paths["cp"]) if paths["cp"] else None
    sl_data = load_streamlines(paths["streamlines"]) if paths["streamlines"] else None
    slc_data = load_slice(paths["slice"]) if paths["slice"] else None

    if args.speed_range:
        spd_lo, spd_hi = float(args.speed_range[0]), float(args.speed_range[1])
    else:
        pools = []
        if sl_data is not None:
            pools.append(sl_data[1])
        if slc_data is not None:
            pools.append(slc_data[3])
        if pools:
            allv = np.concatenate([p[np.isfinite(p)] for p in pools])
            spd_lo, spd_hi = float(allv.min()), float(allv.max())
        else:
            spd_lo, spd_hi = None, None
    say("speed range: " + str(spd_lo) + " .. " + str(spd_hi))

    cp_map = None
    if cp_data is not None:
        cp_map = apply_surface_field(car, cp_data[0], cp_data[1], args.cp_range[0], args.cp_range[1])
        say("cp mapped onto the car: max_dist " + format(cp_map["max_dist"], ".4f") + " m")
    sl_report = None
    if sl_data is not None:
        sl_obj = add_streamlines(sl_data[0], sl_data[1], sl_data[2], spd_lo, spd_hi, args.tube_radius, args.tube_sides)
        sl_report = {
            "n_lines": int(len(sl_data[2]) - 1),
            "n_points": int(len(sl_data[0])),
            "n_vertices": int(len(sl_obj.data.vertices)),
            "n_faces": int(len(sl_obj.data.polygons)),
            "speed_min": float(sl_data[1].min()),
            "speed_max": float(sl_data[1].max()),
        }
        say("streamlines: " + str(sl_report["n_lines"]) + " lines, " + str(sl_report["n_vertices"]) + " vertices")
    slc_report = None
    if slc_data is not None:
        slc_obj = add_slice(slc_data[0], slc_data[1], slc_data[2], slc_data[3], spd_lo, spd_hi)
        slc_report = {
            "nx": int(len(slc_data[0])),
            "nz": int(len(slc_data[1])),
            "n_vertices": int(len(slc_obj.data.vertices)),
            "n_faces": int(len(slc_obj.data.polygons)),
            "speed_min": float(np.nanmin(slc_data[3])),
            "speed_max": float(np.nanmax(slc_data[3])),
        }
        say("slice: " + str(slc_report["nx"]) + " x " + str(slc_report["nz"]) + ", " + str(slc_report["n_faces"]) + " faces")
    t_fields = time.time() - t0

    t0 = time.time()
    stills_report = {}
    names = [s.strip() for s in args.stills.split(",") if s.strip()]
    for nm in names:
        path = os.path.join(args.out, nm + ".png")
        show = show_for(nm, cp_data is not None)
        secs = render_still(nm, path, show)
        st = image_stats(path)
        stills_report[nm] = {"path": path, "seconds": float(secs), "width": st["width"],
                             "height": st["height"], "lum_std": st["lum_std"]}
        say("still " + nm + ": " + str(st["width"]) + "x" + str(st["height"])
            + " in " + format(secs, ".1f") + " s, lum_std " + format(st["lum_std"], ".4f"))
    t_renders = time.time() - t0

    ctx = {"args": args, "car_obj": car, "car_co": car_co, "center": center,
           "cp": cp_data, "cp_map": cp_map, "sl_data": sl_data, "slc_data": slc_data,
           "stills": stills_report, "attribution": attribution}
    ck = checks(ctx)
    ok, reasons = judge(ck)
    report = {
        "tool": TOOL,
        "glb": args.glb,
        "sim_geom": args.sim_geom,
        "attribution": attribution,
        "params": {k: getattr(args, k) for k in sorted(vars(args))},
        "car": {
            "n_vertices": int(len(car.data.vertices)),
            "n_polygons": int(len(car.data.polygons)),
            "center": [float(v) for v in center],
            "bbox_lo": [float(v) for v in bbox_lo],
            "bbox_hi": [float(v) for v in bbox_hi],
        },
        "cp": None if cp_data is None else {
            "n_points": int(len(cp_data[0])),
            "min": float(cp_data[1].min()),
            "max": float(cp_data[1].max()),
            "mean": float(cp_data[1].mean()),
            "map": cp_map,
        },
        "streamlines": sl_report,
        "slice": slc_report,
        "speed_range": None if spd_lo is None else [float(spd_lo), float(spd_hi)],
        "stills": stills_report,
        "timing_s": {"build": float(t_build), "fields": float(t_fields),
                     "renders": float(t_renders), "total": float(time.time() - t_total)},
        "gate": {"checks": {k: bool(good) for k, (good, _d) in ck.items()},
                 "pass": bool(ok), "reasons": reasons},
    }
    with open(os.path.join(args.out, "scene.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=1)
    if args.save_blend:
        bpy.ops.wm.save_as_mainfile(filepath=os.path.join(args.out, "scene.blend"))
        say("scene.blend saved")
    say("report: " + os.path.join(args.out, "scene.json"))
    if ok:
        print("GATE PASS", flush=True)
        return 0
    print("GATE FAIL " + "; ".join(reasons), flush=True)
    return 1


def main():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    args = build_arg_parser().parse_args(argv)
    try:
        if args.selftest:
            return selftest()
        return run(args)
    except SceneRefusal as e:
        print("refused: " + e.code + ": " + e.message, flush=True)
        return 2


if __name__ == "__main__":
    sys.exit(main())
