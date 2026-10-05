#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-or-later
# meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
# This Blender Python API script is free software: you can redistribute it
# and/or modify it under the terms of the GNU General Public License as
# published by the Free Software Foundation, either version 2 of the License,
# or (at your option) any later version. It is distributed WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE. See tools/promo/COPYING for the licence text and
# tools/drone/LICENSE-NOTICE.md for why this file is not under the
# repository's Prosperity licence (a separate program run by Blender,
# aggregated with meteor-cfd, not linked into it).
# render.py - render the drone showreel plates inside Blender.
# No GPL-licensed source was consulted.

"""Render the drone showreel plates with Blender Cycles.

Five subcommands (argv after "--"):

stills     the six SHOTS 1920x1080 stills: hero clean and Cp, the rotor
           downwash with the trimmed streamlines, the speed slice cropped
           to the drone with the drone whole, and the review's before/after
           walls over the ghost drone
turntable  --kind clean|cp: turntable_<kind>/frame_NNNN.png, resumable per
           frame like the promo turntable
reveal     the streamline reveal through cam_downwash (reveal/frame_NNNN.png,
           resumable): every line grows along the flow from the most
           upstream (+x) first point
multiverse 36 visual variants of the drone (arm stretch, prop scale, motor
           cant, a camera pod) rendered as RGBA layers for compositing plus
           a colour-mask silhouette layer and two 1920x1080 grid plates -
           the variants are visual only, NOT solved, no flow is computed
check      judges an --out directory into render.json and prints
           DRONE-RENDER GATE PASS / GATE FAIL

It imports ../promo/render.py through importlib (promo_render) and uses its
loaded scene module object (R.S) - never a second copy, because S.MATERIALS
is module state - so the bars, the reveal material, the framing, the bar
check and the GPU record are exactly the promo code. The drone geometry is
the BSD-3-Clause PX4 x500 model ("(c) 2022 Rudis Laboratories / PX4
Autopilot for Drones"), so NOTHING is written inside the repository: every
output goes under --out, which must lie outside it, and the model's
LICENSE.txt / LICENSE are copied beside them. The flow values come from a
demonstration run: an unsteady, unvalidated solve; no validated
coefficients.

--selftest runs T1-T10 inside Blender (no rendering).
"""

import argparse
import colorsys
import contextlib
import io
import json
import math
import os
import shutil
import sys
import tempfile
import time

import numpy as np
import bpy
from mathutils import Matrix, Vector

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _load(name, path):
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


R = _load("promo_render", os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "promo", "render.py")))
S = R.S                    # one scene module: promo render.py's own loaded scene.py

TOOL = "tools/drone/render.py"
VERSION = "drone-render/1"
CREDIT = "(c) 2022 Rudis Laboratories / PX4 Autopilot for Drones, BSD-3-Clause"
HONESTY = "demonstration numbers from an unsteady, unvalidated run; no validated coefficients"
TARGET = (0.0, 0.0, -0.05)
N_FRAMES = 240              # turntable
REVEAL_FRAMES = 150         # R.reveal_front's REVEAL_GROW = 120 grows, 121..150 hold
STREAM_RADIUS = 0.0012
STREAM_SIDES = 8
TRIM_X = 0.3                # rake lines keep only the x <= TRIM_X half (m)
RAKE_KEEP_Y = 0.061         # m, |seed y| <= this keeps a rake line
CP_EMIT = 0.15
WIRE_T = 0.0005
GHOST_ALPHA = 0.25
CP_MAP_MAX = 0.05           # m
CP_MAP_MEAN_MAX = 0.005     # m
LIT = 0.15                  # a pixel is lit iff max(R,G,B) >= LIT
VARIANT_RES = 768
GRID = (9, 4)               # cols, rows of the 1920x1080 grid plates
ARMS = (1.0, 0.8, 1.25)
PROPS = (1.0, 0.85, 1.2)
CANTS = (0.0, 12.0)
PODS = (False, True)
ARM_PARTS = ("frame_CarbonFiber", "frame_Metal")
ARM_R0 = 0.07               # m: vertices with |xy| <= ARM_R0 never move
CANT_Z0 = 0.034             # m: the cant pivot height (motor mount)
POD_CENTRE = (0.02, 0.0, -0.095)
POD_HALF = (0.11, 0.045, 0.04)
MASK_MIN_PX = 50
ALPHA_FRAC = (0.02, 0.6)
CLAIM = "visual variants for compositing only - not solved, no flow computed"

MATERIALS_DRONE = {          # name: (base RGBA, metallic, roughness)
    "carbon": ((0.035, 0.038, 0.042, 1.0), 0.0, 0.42),
    "metal":  ((0.32, 0.34, 0.36, 1.0), 1.0, 0.32),
    "rubber": ((0.012, 0.012, 0.013, 1.0), 0.0, 0.8),
    "bell":   ((0.05, 0.06, 0.07, 1.0), 1.0, 0.28),
    "prop":   ((0.02, 0.02, 0.022, 1.0), 0.0, 0.5),
    "pcb":    ((0.08, 0.09, 0.10, 1.0), 0.3, 0.4),
    "pod":    ((0.06, 0.065, 0.07, 1.0), 0.3, 0.4),
    "walls_steel": ((0.55, 0.58, 0.62, 1.0), 0.6, 0.35),
}
WORLD_RGB = (0.004, 0.006, 0.009)
TEAL = (0.08, 0.72, 0.65)
LIGHTS = [  # name, AREA size (m), energy (W), location, colour - each aimed at TARGET
    ("key", 1.2, 60.0, (0.9, -1.2, 1.4), (1.0, 0.97, 0.93)),
    ("fill", 1.5, 15.0, (-1.2, -0.8, 0.4), (0.75, 0.85, 1.0)),
    ("rim", 0.6, 50.0, (-1.0, 1.0, 0.8), TEAL),
    ("top", 2.0, 30.0, (0.0, 0.0, 2.0), (0.9, 0.95, 1.0)),
]
CAMERAS = {  # name: (target, direction, distance); lens 50 mm, aimed with S._aim_at
    "cam_hero": (TARGET, (0.72, -0.60, 0.35), 1.8),
    "cam_downwash": ((-0.25, 0.0, -0.04), (0.25, -0.90, 0.40), 2.3),
    "cam_slice": ((-0.10, -0.174, -0.05), (0.20, -0.95, 0.24), 2.0),
    "cam_walls": (TARGET, (0.60, -0.62, 0.50), 2.1),
    "cam_variant": ((0.0, 0.0, -0.06), (0.62, -0.62, 0.48), 1.45),
}
TT = (TARGET, (0.72, -0.60, 0.30), 1.85)      # S.add_turntable(centre, direction, distance, N_FRAMES)
SHOTS = {   # drone: clean | cp | ghost; vt: view transform; bar: None | "cp" | "speed"
    "hero_clean": {"camera": "cam_hero", "drone": "clean", "streamlines": False, "slice": False, "walls": None, "vt": "AgX", "bar": None},
    "hero_cp":    {"camera": "cam_hero", "drone": "cp", "streamlines": False, "slice": False, "walls": None, "vt": "Standard", "bar": "cp"},
    "downwash":   {"camera": "cam_downwash", "drone": "clean", "streamlines": True, "slice": False, "walls": None, "vt": "AgX", "bar": None},
    "slice":      {"camera": "cam_slice", "drone": "clean", "streamlines": False, "slice": True, "walls": None, "vt": "Standard", "bar": "speed"},
    "before":     {"camera": "cam_walls", "drone": "ghost", "streamlines": False, "slice": False, "walls": "before", "vt": "AgX", "bar": None},
    "after":      {"camera": "cam_walls", "drone": "ghost", "streamlines": False, "slice": False, "walls": "after", "vt": "AgX", "bar": None},
}
CHECK_IDS = ["C-STILLS", "C-BARS", "C-CLIP", "C-FRAMING", "C-CPMAP", "C-TT-CLEAN", "C-TT-CP",
             "C-REVEAL", "C-WALLS", "C-VARIANTS", "C-GRID", "C-LICENSE"]

LOOK_LIGHT_SCALE = {"Standard": 0.45, "AgX": 1.0}    # AgX at full brightness; only the Standard (Cp) look dims for the clip gate
LIGHT_BASE = {}              # light object name -> build-time energy, recorded in build_drone_scene

CLEAN_MAT = {}               # drone mesh object name -> its clean material


def say(msg):
    print("[drone-render] " + msg, flush=True)


# ---------------------------------------------------------------- pure core

def part_material(name):
    """The MATERIALS_DRONE key for one x500 part name (first match wins)."""
    if name.startswith("prop_"):
        return "prop"
    if name.startswith("motor_bell"):
        return "bell"
    if name.startswith("motor_base") or name == "frame_Metal":
        return "metal"
    if "Rubber" in name or "Foam" in name:
        return "rubber"
    if name in ("frame_CarbonFiber", "frame_LandingPlastic"):
        return "carbon"
    return "pcb"


def motor_index(name):
    """The rotor index i of motor_base_<i> / motor_bell_<i>_... / prop_<i>_...; else None."""
    for pre in ("motor_base_", "motor_bell_", "prop_"):
        if name.startswith(pre):
            rest = name[len(pre):]
            digits = ""
            for ch in rest:
                if ch.isdigit():
                    digits += ch
                else:
                    break
            if digits:
                return int(digits)
    return None


def variant_table(n=36):
    """The n visual variants (DR-N unless 1 <= n <= 36): arm scale, prop
    scale, cant degrees and pod flag from the index, never solved."""
    if not (1 <= int(n) <= 36):
        raise S.SceneRefusal("DR-N", "--n " + str(n) + " outside 1..36")
    rows = []
    for k in range(int(n)):
        rows.append({"index": k, "arm_scale": ARMS[k % 3],
                     "prop_scale": PROPS[(k // 3) % 3],
                     "cant_deg": CANTS[(k // 9) % 2],
                     "pod": PODS[(k // 18) % 2],
                     "baseline": k == 0, "solved": False})
    return rows


def radial_stretch(co, s, r0=ARM_R0):
    """(n,3) -> (n,3): r = |(x, y)|; where r > r0 the xy radius becomes
    r0 + (r - r0) * s along the same direction; z and r <= r0 untouched."""
    co = np.asarray(co, dtype=np.float64)
    out = co.copy()
    r = np.hypot(co[:, 0], co[:, 1])
    sel = r > r0
    if bool(sel.any()):
        f = (r0 + (r[sel] - r0) * float(s)) / r[sel]
        out[sel, 0] = co[sel, 0] * f
        out[sel, 1] = co[sel, 1] * f
    return out


def scale_about(co, c, k):
    """(n,3) -> (n,3): (x, y) scaled by k about c's xy; z untouched."""
    co = np.asarray(co, dtype=np.float64)
    out = co.copy()
    out[:, 0] = float(c[0]) + (co[:, 0] - float(c[0])) * float(k)
    out[:, 1] = float(c[1]) + (co[:, 1] - float(c[1])) * float(k)
    return out


def cant_rotate(co, c, deg, z0=CANT_Z0):
    """(n,3) -> (n,3): rotate about the axis through (c_x, c_y, z0) along
    t = (-sin phi, cos phi, 0), phi = atan2(c_y, c_x), by +deg right-handed
    about t (+deg tilts the top outward). Rodrigues formula."""
    co = np.asarray(co, dtype=np.float64)
    cx, cy = float(c[0]), float(c[1])
    phi = math.atan2(cy, cx)
    t = np.array([-math.sin(phi), math.cos(phi), 0.0])
    th = math.radians(float(deg))
    o = np.array([cx, cy, float(z0)])
    v = co - o
    tv = v @ t
    cross = np.cross(t, v)
    vr = (v * math.cos(th) + cross * math.sin(th)
          + np.outer(tv, t) * (1.0 - math.cos(th)))
    return o + vr


def compose_grid(images, cols, W, H):
    """Tile the TOP-DOWN (row 0 = top) images into one (H, W, 4) float32
    grid: rows = ceil(n/cols), nearest-neighbour shrink centred per tile,
    tile (row k//cols, col k%cols) from the top-left."""
    n = len(images)
    rows = int(math.ceil(n / float(cols)))
    out = np.zeros((int(H), int(W), 4), dtype=np.float32)
    tw = int(W) // cols
    th = int(H) // rows
    for k, img in enumerate(images):
        img = np.asarray(img, dtype=np.float32)
        ih, iw = img.shape[0], img.shape[1]
        row = k // cols
        col = k % cols
        scale = min(tw / float(iw), th / float(ih))
        sw = max(1, int(math.floor(iw * scale)))
        sh = max(1, int(math.floor(ih * scale)))
        x0 = col * tw + (tw - sw) // 2
        y0 = row * th + (th - sh) // 2
        sr = np.floor((np.arange(sh) + 0.5) * ih / sh).astype(np.int64)
        scc = np.floor((np.arange(sw) + 0.5) * iw / sw).astype(np.int64)
        out[y0:y0 + sh, x0:x0 + sw] = img[np.ix_(sr, scc)]
    return out


def trim_lines(points, speed, offsets, group, seeds, trim_x=TRIM_X, keep_y=RAKE_KEEP_Y):
    """(P, U, O, kept): the picture lines. Rotor lines (group 0) keep the
    points from the seed on (the seed is exactly one of the polyline points,
    so take the nearest); rake lines (group 1) survive only when
    abs(seed y) <= keep_y and then keep the points from the first point with
    x <= trim_x on - the upstream half of every line is noise. Lines with
    < 2 kept points are dropped. kept = the original line indices in order,
    O restarts at 0."""
    points = np.asarray(points, dtype=np.float64)
    speed = np.asarray(speed, dtype=np.float64)
    offsets = np.asarray(offsets, dtype=np.int64)
    group = np.asarray(group)
    seeds = np.asarray(seeds, dtype=np.float64)
    Ps = []
    Us = []
    O = [0]
    kept = []
    for j in range(len(offsets) - 1):
        s0, s1 = int(offsets[j]), int(offsets[j + 1])
        pts = points[s0:s1]
        if int(group[j]) == 0:
            st = int(np.argmin(np.linalg.norm(pts - seeds[j], axis=1)))
            idx = np.arange(st, pts.shape[0])
        else:
            if abs(float(seeds[j][1])) > float(keep_y):
                continue
            hits = np.nonzero(pts[:, 0] <= float(trim_x))[0]
            if hits.size == 0:
                continue
            idx = np.arange(int(hits[0]), pts.shape[0])
        if idx.size < 2:
            continue
        Ps.append(pts[idx])
        Us.append(speed[s0:s1][idx])
        O.append(O[-1] + int(idx.size))
        kept.append(j)
    P = np.concatenate(Ps) if Ps else np.zeros((0, 3), dtype=np.float64)
    U = np.concatenate(Us) if Us else np.zeros(0, dtype=np.float64)
    return P, U, np.asarray(O, dtype=np.int64), kept


def drone_reveal_times(points, speed, offsets, u_inf):
    """R.reveal_times of the mapped lines (x' = -x): the front starts at the
    most upstream (+x) first point."""
    F = np.array([-1.0, 1.0, 1.0])
    return R.reveal_times(np.asarray(points, dtype=np.float64) * F,
                          speed, offsets, u_ref=float(u_inf))


def judge(ck):
    """{"pass", "failed"}: the CHECK_IDS in order that are False or missing."""
    failed = [cid for cid in CHECK_IDS if not ck.get(cid)]
    return {"pass": not failed, "failed": failed}


# ---------------------------------------------------------------- refusals

def _check_out(out):
    a = os.path.normcase(os.path.abspath(out))
    r = os.path.normcase(REPO)
    if a == r or a.startswith(r + os.sep):
        raise S.SceneRefusal("DR-OUT", "--out " + str(out) + " resolves inside the repository (" + REPO + ")")


def _require_arg(value, name):
    if value is None or value == "":
        raise S.SceneRefusal("DR-INPUT", "--" + name + " is required")


def _require_file(path, what):
    if not os.path.isfile(path):
        raise S.SceneRefusal("DR-INPUT", what + " not found: " + str(path))


def _posix(path):
    return str(path).replace(os.sep, "/")


def _load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def parse_shots(s):
    names = [t.strip() for t in str(s).split(",") if t.strip()]
    if not names:
        raise S.SceneRefusal("DR-SHOT", "--shots is empty")
    for nm in names:
        if nm not in SHOTS:
            raise S.SceneRefusal("DR-SHOT", "unknown shot: " + nm)
    return names


def validate_common(args, post=True):
    _require_arg(args.blend, "blend")
    _require_arg(args.out, "out")
    _check_out(args.out)
    _require_file(args.blend, "--blend")
    if post:
        _require_arg(args.post, "post")
        _require_file(os.path.join(args.post, "numbers.json"), "--post/numbers.json")
        _require_file(os.path.join(args.post, "LICENSE.txt"), "--post/LICENSE.txt")
        _require_file(os.path.join(args.post, "LICENSE"), "--post/LICENSE")


def validate_stills(args):
    _require_arg(args.blend, "blend")
    _require_arg(args.out, "out")
    _check_out(args.out)
    shots = parse_shots(args.shots)
    _require_file(args.blend, "--blend")
    _require_arg(args.post, "post")
    _require_file(os.path.join(args.post, "numbers.json"), "--post/numbers.json")
    _require_file(os.path.join(args.post, "LICENSE.txt"), "--post/LICENSE.txt")
    _require_file(os.path.join(args.post, "LICENSE"), "--post/LICENSE")
    need = {
        "cp": any(SHOTS[n]["drone"] == "cp" for n in shots),
        "stream": any(SHOTS[n]["streamlines"] for n in shots),
        "slice": any(SHOTS[n]["slice"] for n in shots),
        "walls": any(SHOTS[n]["walls"] for n in shots),
    }
    if need["cp"]:
        _require_file(os.path.join(args.post, "surface.npz"), "--post/surface.npz")
    if need["stream"]:
        _require_file(os.path.join(args.post, "streamlines.npz"), "--post/streamlines.npz")
    if need["slice"]:
        _require_file(os.path.join(args.post, "slice.npz"), "--post/slice.npz")
    if need["walls"]:
        for fn in ("walls_before.npz", "walls_after.npz", "walls.json"):
            _require_file(os.path.join(args.post, fn), "--post/" + fn)
    return shots


def validate_turntable(args):
    if args.kind not in ("clean", "cp"):
        raise S.SceneRefusal("DR-KIND", "unknown --kind: " + str(args.kind))
    validate_common(args)
    if args.kind == "cp":
        _require_file(os.path.join(args.post, "surface.npz"), "--post/surface.npz")


def validate_reveal(args):
    validate_common(args)
    _require_file(os.path.join(args.post, "streamlines.npz"), "--post/streamlines.npz")


def validate_multiverse(args):
    _require_arg(args.blend, "blend")
    _require_arg(args.rotors, "rotors")
    _require_arg(args.out, "out")
    _check_out(args.out)
    variant_table(args.n)
    _require_file(args.blend, "--blend")
    _require_file(args.rotors, "--rotors")


# ---------------------------------------------------------------- materials

def _drone_material(name):
    m = bpy.data.materials.get(name)
    if m is not None:
        return m
    base, metallic, rough = MATERIALS_DRONE[name]
    m = bpy.data.materials.new(name)
    b = S._principled(m)
    b.inputs["Base Color"].default_value = base
    b.inputs["Metallic"].default_value = metallic
    b.inputs["Roughness"].default_value = rough
    return m


def _cp_material():
    m = bpy.data.materials.get("drone_cp")
    if m is not None:
        return m
    m = bpy.data.materials.new("drone_cp")
    nt = m.node_tree
    b = S._principled(m)
    attr = nt.nodes.new("ShaderNodeAttribute")
    attr.attribute_name = "Cp_rgb"
    nt.links.new(attr.outputs["Color"], b.inputs["Base Color"])
    nt.links.new(attr.outputs["Color"], b.inputs["Emission Color"])
    b.inputs["Emission Strength"].default_value = CP_EMIT
    b.inputs["Roughness"].default_value = 0.45
    b.inputs["Metallic"].default_value = 0.0
    return m


def _ghost_material():
    m = bpy.data.materials.get("ghost")
    if m is not None:
        return m
    m = bpy.data.materials.new("ghost")
    b = S._principled(m)
    b.inputs["Base Color"].default_value = (0.03, 0.035, 0.04, 1.0)
    b.inputs["Alpha"].default_value = GHOST_ALPHA
    b.inputs["Roughness"].default_value = 0.5
    return m


def _walls_wire_material():
    m = bpy.data.materials.get("walls_wire")
    if m is not None:
        return m
    m = bpy.data.materials.new("walls_wire")
    nt = m.node_tree
    em = nt.nodes.new("ShaderNodeEmission")
    em.inputs["Color"].default_value = (TEAL[0], TEAL[1], TEAL[2], 1.0)
    em.inputs["Strength"].default_value = 2.0
    nt.links.new(em.outputs["Emission"], nt.nodes["Material Output"].inputs["Surface"])
    return m


def _sil_material(k, n):
    name = "sil_%02d" % k
    m = bpy.data.materials.get(name)
    if m is not None:
        return m
    m = bpy.data.materials.new(name)
    nt = m.node_tree
    em = nt.nodes.new("ShaderNodeEmission")
    r, g, b = colorsys.hsv_to_rgb(0.50 - 0.42 * k / max(n - 1, 1), 0.75, 0.9)
    em.inputs["Color"].default_value = (r, g, b, 1.0)
    em.inputs["Strength"].default_value = 1.0
    nt.links.new(em.outputs["Emission"], nt.nodes["Material Output"].inputs["Surface"])
    return m


# ---------------------------------------------------------------- scene

def _drone_mesh_objects():
    return [o for o in bpy.data.objects if o.type == "MESH" and o.name in CLEAN_MAT]


def _obj_co(o):
    n = len(o.data.vertices)
    co = np.empty(n * 3, dtype=np.float64)
    o.data.vertices.foreach_get("co", co)
    return co.reshape(n, 3)


def build_drone_scene(blend, device, samples):
    """Append the x500 .blend, bake transforms into single-user meshes
    (local == world), drop non-mesh objects, and build the studio: world,
    lights, cameras, turntable, render config."""
    t0 = time.time()
    CLEAN_MAT.clear()
    S.reset_scene()
    with bpy.data.libraries.load(blend, link=False) as (src, dst):
        dst.objects = src.objects
    for ob in list(bpy.data.objects):
        if ob.type != "MESH":
            bpy.data.objects.remove(ob)
            continue
        mw = ob.matrix_world.copy()
        ob.data = ob.data.copy()
        ob.data.transform(mw)
        ob.parent = None
        ob.matrix_world = Matrix.Identity(4)
    for ob in bpy.data.objects:
        try:
            bpy.context.scene.collection.objects.link(ob)
        except RuntimeError:
            pass
    for ob in bpy.data.objects:
        if ob.type != "MESH":
            continue
        mat = _drone_material(part_material(ob.name))
        ob.data.materials.clear()
        ob.data.materials.append(mat)
        CLEAN_MAT[ob.name] = mat
    S.build_materials()
    w = bpy.data.worlds.new("drone_world")
    bpy.context.scene.world = w
    bg = w.node_tree.nodes["Background"]
    bg.inputs[0].default_value = (WORLD_RGB[0], WORLD_RGB[1], WORLD_RGB[2], 1.0)
    bg.inputs[1].default_value = 1.0
    for name, size, energy, loc, col in LIGHTS:
        ld = bpy.data.lights.new(name, type="AREA")
        ld.shape = "SQUARE"
        ld.size = size
        ld.energy = energy
        ld.color = col
        ob = bpy.data.objects.new(name, ld)
        bpy.context.scene.collection.objects.link(ob)
        ob.location = loc
        S._aim_at(ob, TARGET)
    LIGHT_BASE.clear()
    for ob in bpy.data.objects:
        if ob.type == "LIGHT":
            LIGHT_BASE[ob.name] = float(ob.data.energy)
    for name, (target, direction, dist) in CAMERAS.items():
        cd = bpy.data.cameras.new(name)
        cd.lens = 50.0
        ob = bpy.data.objects.new(name, cd)
        bpy.context.scene.collection.objects.link(ob)
        ob.location = Vector(target) + Vector(direction).normalized() * dist
        S._aim_at(ob, target)
    S.add_turntable(*TT, N_FRAMES)
    S.configure_render(samples, device)
    R.ensure_gpu(device)
    bpy.context.scene.render.use_persistent_data = True
    meshes = _drone_mesh_objects()
    nverts = sum(len(o.data.vertices) for o in meshes)
    say("scene built in %.1f s: %d objects, %d mesh objects, %d vertices"
        % (time.time() - t0, len(bpy.data.objects), len(meshes), nverts))
    return meshes


def apply_cp_field(surface_path, lo, hi):
    """Cp state: apply_surface_field once per non-prop object, then swap its
    material to drone_cp; props keep their clean material."""
    pts, vals = S.load_surface_field(surface_path)
    t0 = time.time()
    cp_mat = _cp_material()
    agg_n = 0
    agg_max = 0.0
    dsum = 0.0
    for ob in _drone_mesh_objects():
        if ob.name.startswith("prop_"):
            continue
        r = S.apply_surface_field(ob, pts, vals, lo, hi)
        ob.data.materials.clear()
        ob.data.materials.append(cp_mat)
        agg_n += int(r["n"])
        agg_max = max(agg_max, float(r["max_dist"]))
        dsum += float(r["mean_dist"]) * int(r["n"])
    cp_map = {"n": agg_n, "max_dist": float(agg_max),
              "mean_dist": float(dsum / agg_n) if agg_n else None}
    say("field cp: %d vertices mapped in %.1f s, max_dist %.4f m"
        % (agg_n, time.time() - t0, agg_max))
    return cp_map


def build_walls_objects(post_dir):
    """walls_<label> (steel) + walls_<label>_wire (teal wireframe) per label,
    both hidden; the state setter shows one pair at a time."""
    made = {}
    for label in ("before", "after"):
        path = os.path.join(post_dir, "walls_" + label + ".npz")
        with np.load(path) as z:
            V = np.asarray(z["V"], dtype=np.float64)
            F = np.asarray(z["F"], dtype=np.int64)
        me = bpy.data.meshes.new("walls_" + label)
        me.from_pydata(V.tolist(), [], F.tolist())
        me.update()
        me.materials.append(_drone_material("walls_steel"))
        obj = bpy.data.objects.new("walls_" + label, me)
        bpy.context.scene.collection.objects.link(obj)
        obj.hide_render = True
        me2 = me.copy()
        me2.materials.clear()
        me2.materials.append(_walls_wire_material())
        obj2 = bpy.data.objects.new("walls_" + label + "_wire", me2)
        bpy.context.scene.collection.objects.link(obj2)
        mod = obj2.modifiers.new("wire", "WIREFRAME")
        mod.thickness = WIRE_T
        obj2.hide_render = True
        made[label] = (obj, obj2)
        say("walls_" + label + ": %d tris" % len(F))
    return made


def apply_drone_state(mode, walls_label=None):
    """Materials + hide_render for one shot state: clean | cp | ghost. The
    slice shot keeps the drone whole and clean - parts behind the opaque
    slice plane are hidden by the plane itself."""
    cp_mat = _cp_material() if mode == "cp" else None
    ghost = _ghost_material() if mode == "ghost" else None
    for o in _drone_mesh_objects():
        if mode == "cp":
            mat = CLEAN_MAT[o.name] if o.name.startswith("prop_") else cp_mat
        elif mode == "ghost":
            mat = ghost
        else:
            mat = CLEAN_MAT[o.name]
        o.data.materials.clear()
        o.data.materials.append(mat)
    for lb in ("before", "after"):
        solid = bpy.data.objects.get("walls_" + lb)
        if solid is None:
            continue
        wire = bpy.data.objects.get("walls_" + lb + "_wire")
        show = (mode == "ghost" and lb == walls_label)
        solid.hide_render = not show
        wire.hide_render = not show


def apply_light_scale(vt):
    """Every light's energy = LIGHT_BASE * LOOK_LIGHT_SCALE[vt], always from
    the recorded base (never cumulative) - the promo LOOK rule."""
    scale = LOOK_LIGHT_SCALE[vt]
    for name, base in LIGHT_BASE.items():
        obj = bpy.data.objects.get(name)
        if obj is not None and obj.type == "LIGHT":
            obj.data.energy = base * scale


def drone_corners():
    """The 8 world-bbox corners of the drone mesh objects (local == world)."""
    lo = np.array([np.inf, np.inf, np.inf])
    hi = -lo.copy()
    for o in _drone_mesh_objects():
        co = _obj_co(o)
        lo = np.minimum(lo, co.min(axis=0))
        hi = np.maximum(hi, co.max(axis=0))
    return [(float(px), float(py), float(pz))
            for px in (float(lo[0]), float(hi[0]))
            for py in (float(lo[1]), float(hi[1]))
            for pz in (float(lo[2]), float(hi[2]))]


def add_pod():
    """The multiverse camera pod: a 48x24 UV sphere whose MESH DATA holds the
    world coordinates (scaled to POD_HALF, then translated by POD_CENTRE in
    the data, object location left at 0) - so _obj_co(pod) is world."""
    bpy.ops.mesh.primitive_uv_sphere_add(segments=48, ring_count=24, radius=1.0,
                                         location=(0.0, 0.0, 0.0))
    obj = bpy.context.active_object
    obj.name = "pod"
    obj.data.transform(Matrix.Translation(Vector(POD_CENTRE))
                       @ Matrix.Diagonal((POD_HALF[0], POD_HALF[1], POD_HALF[2], 1.0)))
    obj.data.materials.clear()
    obj.data.materials.append(_drone_material("pod"))
    obj.hide_render = True
    CLEAN_MAT[obj.name] = _drone_material("pod")
    return obj


def set_materials(mat_or_map):
    """One material for every drone object, or {name: material}."""
    for o in _drone_mesh_objects():
        mat = mat_or_map.get(o.name) if isinstance(mat_or_map, dict) else mat_or_map
        o.data.materials.clear()
        o.data.materials.append(mat)


# ---------------------------------------------------------------- runners

def _ranges_from(numbers):
    disp = numbers.get("display") or {}
    cp = disp.get("cp_range") or [CP_RANGE[0], CP_RANGE[1]]
    sp = disp.get("speed_range") or [SPEED_RANGE[0], SPEED_RANGE[1]]
    return [float(cp[0]), float(cp[1])], [float(sp[0]), float(sp[1])]


def _copy_licences(post_dir, out):
    for fn in ("LICENSE.txt", "LICENSE"):
        src = os.path.join(post_dir, fn)
        if os.path.isfile(src):
            shutil.copyfile(src, os.path.join(out, fn))


def credits_text(blender_version):
    return "\n".join([
        "Drone model: PX4 x500 assembly, " + CREDIT,
        "Licence: BSD-3-Clause (see LICENSE beside this file). No endorsement by the copyright holders is implied.",
        "Changes: materials replaced; Cp and speed from a demonstration CFD run mapped onto it; "
        "multiverse variants are geometric edits for visuals only, not solved.",
        "Rendered with Blender " + blender_version + " (Cycles).",
    ]) + "\n"


def run_stills(args):
    t_total = time.time()
    shots = validate_stills(args)
    os.makedirs(args.out, exist_ok=True)
    gpu_start = R.gpu_record()
    numbers = _load_json(os.path.join(args.post, "numbers.json"))
    (cp_lo, cp_hi), (sp_lo, sp_hi) = _ranges_from(numbers)
    walls_block = None
    wj_path = os.path.join(args.post, "walls.json")
    if os.path.isfile(wj_path):
        wj = _load_json(wj_path)
        walls_block = {"before": wj.get("before"), "after": wj.get("after"),
                       "target": wj.get("target")}
    build_drone_scene(args.blend, args.device, args.samples)
    if any(SHOTS[n]["bar"] == "cp" for n in shots):
        R.build_bar(bpy.data.objects["cam_hero"], "cp", cp_lo, cp_hi)
    if any(SHOTS[n]["bar"] == "speed" for n in shots):
        R.build_bar(bpy.data.objects["cam_slice"], "speed", sp_lo, sp_hi)
    R.set_bar_state(None)
    n_lines = None
    n_lines_all = None
    if any(SHOTS[n]["streamlines"] for n in shots):
        t0 = time.time()
        pts, spd, offs = S.load_streamlines(os.path.join(args.post, "streamlines.npz"))
        with np.load(os.path.join(args.post, "streamlines.npz")) as z:
            grp = np.asarray(z["group"], dtype=np.int8)
            seeds = np.asarray(z["seeds"], dtype=np.float64)
        n_lines_all = int(len(offs) - 1)
        pts, spd, offs, _kept = trim_lines(pts, spd, offs, grp, seeds)
        n_lines = int(len(offs) - 1)
        S.add_streamlines(pts, spd, offs, sp_lo, sp_hi, STREAM_RADIUS, STREAM_SIDES)
        say("field streamlines: %d of %d lines in %.1f s" % (n_lines, n_lines_all, time.time() - t0))
    slice_xz = None
    if any(SHOTS[n]["slice"] for n in shots):
        t0 = time.time()
        x, z, y0, spd = S.load_slice(os.path.join(args.post, "slice.npz"))
        slc = S.add_slice(x, z, y0, spd, sp_lo, sp_hi)
        slc.visible_shadow = False
        slice_xz = (x, z)
        say("field slice: %d x %d in %.1f s" % (len(x), len(z), time.time() - t0))
    cp_map = None
    if any(SHOTS[n]["drone"] == "cp" for n in shots):
        cp_map = apply_cp_field(os.path.join(args.post, "surface.npz"), cp_lo, cp_hi)
    if any(SHOTS[n]["walls"] for n in shots):
        build_walls_objects(args.post)
    corners = drone_corners()
    sc = bpy.context.scene
    shot_reports = {}
    for name in shots:
        row = SHOTS[name]
        apply_drone_state(row["drone"], row["walls"])
        for key, obj_name in (("streamlines", "streamlines"), ("slice", "slice")):
            obj = bpy.data.objects.get(obj_name)
            if obj is not None:
                obj.hide_render = not row[key]
        R.set_bar_state(("bar_cp_cam_hero" if row["bar"] == "cp" else
                         "bar_speed_cam_slice" if row["bar"] == "speed" else None))
        R.set_view_transform(row["vt"])
        apply_light_scale(row["vt"])
        sc.camera = bpy.data.objects[row["camera"]]
        if name == "slice" and slice_xz is not None:
            x, z = slice_xz
            y0v = float(y0)
            pts_f = [(float(x[0]), y0v, float(z[0])), (float(x[0]), y0v, float(z[-1])),
                     (float(x[-1]), y0v, float(z[0])), (float(x[-1]), y0v, float(z[-1]))]
        else:
            pts_f = corners
        u_max = R.PLATE[0] if row["bar"] else 1.0 - R.FRAME_MARGIN
        framing = R.framing_entry(R.frame_uv(bpy.data.objects[row["camera"]], pts_f), u_max)
        path = os.path.join(args.out, name + ".png")
        secs = R.do_render(path)
        vt = str(sc.view_settings.view_transform)
        bc = None
        if row["bar"] == "cp":
            bc = R.bar_check(path, cp_lo, cp_hi, S.CP_RAMP)
        elif row["bar"] == "speed":
            bc = R.bar_check(path, sp_lo, sp_hi, S.SPEED_RAMP)
        clip = R.clip_fraction(path)
        shot_reports[name] = {
            "png": name + ".png",
            "camera": row["camera"],
            "vt": vt,
            "bar": row["bar"],
            "bar_check": bc,
            "clip": clip,
            "framing": framing,
            "camera_distance": float(CAMERAS[row["camera"]][2]),
            "seconds": float(secs),
        }
        say("still %s in %.1f s, view %s, bar %s, clip %.4f, framing %s, dist %.2f m"
            % (name, secs, vt, "max_diff %.4f" % bc["max_diff"] if bc else "none",
               clip, "pass" if framing["pass"] else "FAIL", CAMERAS[row["camera"]][2]))
    gpu_end = R.gpu_record()
    report = {
        "tool": TOOL,
        "version": VERSION,
        "credit": CREDIT,
        "honesty": HONESTY,
        "blender": bpy.app.version_string,
        "device": args.device,
        "samples": int(args.samples),
        "ranges": {"cp": [cp_lo, cp_hi], "speed": [sp_lo, sp_hi]},
        "n_lines": n_lines,
        "n_lines_all": n_lines_all,
        "cp_map": cp_map,
        "walls": walls_block,
        "shots": shot_reports,
        "gpu_start": gpu_start,
        "gpu_end": gpu_end,
        "seconds_total": float(time.time() - t_total),
    }
    with open(os.path.join(args.out, "stills.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=1)
    with open(os.path.join(args.out, "CREDITS.txt"), "w", encoding="utf-8", newline="\n") as f:
        f.write(credits_text(bpy.app.version_string))
    _copy_licences(args.post, args.out)
    say("stills.json written under " + args.out)
    return 0


def run_turntable(args):
    t_total = time.time()
    validate_turntable(args)
    kind = args.kind
    A, B = R.parse_frames(args.frames, N_FRAMES)
    os.makedirs(args.out, exist_ok=True)
    gpu_start = R.gpu_record()
    numbers = _load_json(os.path.join(args.post, "numbers.json"))
    (cp_lo, cp_hi), _sp = _ranges_from(numbers)
    build_drone_scene(args.blend, args.device, args.samples)
    if kind == "cp":
        apply_cp_field(os.path.join(args.post, "surface.npz"), cp_lo, cp_hi)
        R.build_bar(bpy.data.objects["cam_turntable"], "cp", cp_lo, cp_hi)
    apply_drone_state("cp" if kind == "cp" else "clean")
    vt = "Standard" if kind == "cp" else "AgX"
    R.set_view_transform(vt)
    apply_light_scale(vt)
    R.set_bar_state("bar_cp_cam_turntable" if kind == "cp" else None)
    sc = bpy.context.scene
    cam_obj = bpy.data.objects["cam_turntable"]
    sc.camera = cam_obj
    corners = drone_corners()
    u_max = R.PLATE[0] if kind == "cp" else 1.0 - R.FRAME_MARGIN
    d = os.path.join(args.out, "turntable_" + kind)
    os.makedirs(d, exist_ok=True)
    framing_frames = set(range(A, B + 1, 60))
    framing = []
    n = B - A + 1
    rendered = 0
    skipped = 0
    spent = 0.0
    for i, f in enumerate(range(A, B + 1)):
        if f in framing_frames:
            sc.frame_set(f)
            entry = R.framing_entry(R.frame_uv(cam_obj, corners), u_max)
            framing.append({"frame": int(f), **entry})
        path = os.path.join(d, "frame_%04d.png" % f)
        if os.path.isfile(path) and os.path.getsize(path) > 0:
            skipped += 1
            say("turntable " + kind + " frame " + str(f) + " (" + str(i + 1) + "/" + str(n) + ") skipped")
            continue
        sc.frame_set(f)
        t0 = time.time()
        sc.render.filepath = path
        bpy.ops.render.render(write_still=True)
        secs = time.time() - t0
        rendered += 1
        spent += secs
        eta = (B - f) * (spent / rendered) / 60.0
        say("turntable " + kind + " frame " + str(f) + " (" + str(i + 1) + "/" + str(n) + ") "
            + format(secs, ".1f") + " s, eta " + format(eta, ".1f") + " min")
    gpu_end = R.gpu_record()
    report = {
        "tool": TOOL,
        "version": VERSION,
        "credit": CREDIT,
        "blender": bpy.app.version_string,
        "kind": kind,
        "frames": [A, B],
        "rendered": rendered,
        "skipped": skipped,
        "view_transform": vt,
        "cp_range": [cp_lo, cp_hi] if kind == "cp" else None,
        "framing": framing,
        "samples": int(args.samples),
        "device": args.device,
        "seconds_total": float(time.time() - t_total),
        "seconds_per_rendered_frame": (spent / rendered) if rendered else None,
        "gpu_start": gpu_start,
        "gpu_end": gpu_end,
    }
    with open(os.path.join(args.out, "turntable_" + kind + ".json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=1)
    say("turntable_" + kind + ".json written: " + str(rendered) + " rendered, " + str(skipped) + " skipped")
    return 0


def run_reveal(args):
    t_total = time.time()
    validate_reveal(args)
    A, B = R.parse_frames(args.frames, REVEAL_FRAMES)
    os.makedirs(args.out, exist_ok=True)
    gpu_start = R.gpu_record()
    numbers = _load_json(os.path.join(args.post, "numbers.json"))
    _cp, (sp_lo, sp_hi) = _ranges_from(numbers)
    u_inf = float(np.linalg.norm(np.asarray(numbers["U_inf"], dtype=np.float64)))
    build_drone_scene(args.blend, args.device, args.samples)
    apply_drone_state("clean")
    R.set_view_transform("AgX")
    apply_light_scale("AgX")
    R.set_bar_state(None)
    sc = bpy.context.scene
    sc.camera = bpy.data.objects["cam_downwash"]
    pts, spd, offs = S.load_streamlines(os.path.join(args.post, "streamlines.npz"))
    with np.load(os.path.join(args.post, "streamlines.npz")) as z:
        grp = np.asarray(z["group"], dtype=np.int8)
        seeds = np.asarray(z["seeds"], dtype=np.float64)
    n_lines_all = int(len(offs) - 1)
    pts, spd, offs, _kept = trim_lines(pts, spd, offs, grp, seeds)
    n_lines = int(len(offs) - 1)
    tarr = drone_reveal_times(pts, spd, offs, u_inf)
    t_end = float(tarr.max())
    x0 = min(float(pts[int(o), 0]) for o in offs[:-1])
    tubes = S.add_streamlines(pts, spd, offs, sp_lo, sp_hi, STREAM_RADIUS, STREAM_SIDES)
    mat = R._streamline_reveal_material()
    tubes.data.materials.clear()
    tubes.data.materials.append(mat)
    rt = tubes.data.attributes.new("reveal_t", "FLOAT", "POINT")
    rt.data.foreach_set("value", np.repeat(tarr.astype(np.float32), STREAM_SIDES))
    tubes.data.update()
    sc.cycles.transparent_max_bounces = 64
    tv = mat.node_tree.nodes["reveal_T"]
    say("reveal: %d of %d lines, t_end %.4f s, x0 %.3f m" % (n_lines, n_lines_all, t_end, x0))
    d = os.path.join(args.out, "reveal")
    os.makedirs(d, exist_ok=True)
    n = B - A + 1
    rendered = 0
    skipped = 0
    spent = 0.0
    for i, f in enumerate(range(A, B + 1)):
        front = R.reveal_front(f, t_end)
        path = os.path.join(d, "frame_%04d.png" % f)
        if os.path.isfile(path) and os.path.getsize(path) > 0:
            skipped += 1
            say("reveal frame " + str(f) + " (" + str(i + 1) + "/" + str(n) + ") skipped")
            continue
        tv.outputs["Value"].default_value = front
        secs = R.do_render(path)
        rendered += 1
        spent += secs
        eta = (B - f) * (spent / rendered) / 60.0
        say("reveal frame " + str(f) + " (" + str(i + 1) + "/" + str(n) + ") " + format(secs, ".1f")
            + " s, T " + format(front, ".3f") + " s, eta " + format(eta, ".1f") + " min")
    gpu_end = R.gpu_record()
    report = {
        "tool": TOOL,
        "version": VERSION,
        "credit": CREDIT,
        "blender": bpy.app.version_string,
        "frames": [A, B],
        "rendered": rendered,
        "skipped": skipped,
        "n_lines": n_lines,
        "n_lines_all": n_lines_all,
        "t_end": t_end,
        "x0": x0,
        "u_inf": u_inf,
        "samples": int(args.samples),
        "device": args.device,
        "seconds_total": float(time.time() - t_total),
        "seconds_per_rendered_frame": (spent / rendered) if rendered else None,
        "gpu_start": gpu_start,
        "gpu_end": gpu_end,
    }
    with open(os.path.join(args.out, "reveal.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=1)
    say("reveal.json written: " + str(rendered) + " rendered, " + str(skipped) + " skipped")
    return 0


def _load_rgba(path):
    """(h, w, ch) float32, stored values (Non-Color), row 0 = bottom."""
    img = bpy.data.images.load(path)
    img.colorspace_settings.name = "Non-Color"
    w, h = img.size
    ch = img.channels
    px = np.empty(w * h * ch, dtype=np.float32)
    img.pixels.foreach_get(px)
    bpy.data.images.remove(img)
    return px.reshape(h, w, ch)


def _load_topdown(path):
    a = _load_rgba(path)
    if a.shape[2] == 3:
        a = np.concatenate([a, np.ones(a.shape[:2] + (1,), np.float32)], axis=2)
    return np.ascontiguousarray(a[::-1]).astype(np.float32)


def _write_grid_image(images_topdown, path):
    out = compose_grid(images_topdown, GRID[0], 1920, 1080)
    img = bpy.data.images.new(os.path.basename(path), 1920, 1080, alpha=True)
    img.colorspace_settings.name = "Non-Color"
    img.pixels.foreach_set(np.ascontiguousarray(out[::-1], dtype=np.float32).reshape(-1))
    img.filepath_raw = path
    img.file_format = "PNG"
    img.save()
    bpy.data.images.remove(img)


def run_multiverse(args):
    t_total = time.time()
    validate_multiverse(args)
    os.makedirs(args.out, exist_ok=True)
    gpu_start = R.gpu_record()
    n = int(args.n)
    rows = variant_table(n)
    with open(args.rotors, "r", encoding="utf-8") as f:
        rotors = json.load(f)["rotors"]
    centres = [np.asarray(r["centre_m"], dtype=np.float64) for r in rotors]
    build_drone_scene(args.blend, args.device, args.samples)
    R.set_view_transform("AgX")
    apply_light_scale("AgX")
    R.set_bar_state(None)
    pod = add_pod()
    pod.hide_render = True
    sc = bpy.context.scene
    sc.render.film_transparent = True
    sc.render.image_settings.color_mode = "RGBA"
    sc.render.resolution_x = VARIANT_RES
    sc.render.resolution_y = VARIANT_RES
    sc.camera = bpy.data.objects["cam_variant"]
    meshes = _drone_mesh_objects()
    original = {o.name: _obj_co(o) for o in meshes}
    u_max = 1.0 - R.FRAME_MARGIN
    d = os.path.join(args.out, "multiverse")
    os.makedirs(d, exist_ok=True)
    cleans = []
    sils = []
    vreports = []
    for row in rows:
        t0 = time.time()
        k = row["index"]
        for o in meshes:
            name = o.name
            co = original[name].copy()
            i = motor_index(name)
            c = centres[i] if i is not None and i < len(centres) else None
            if name.startswith("prop_") and c is not None:
                co = scale_about(co, c, row["prop_scale"])
            if c is not None:
                co = cant_rotate(co, c, row["cant_deg"])
                delta = (radial_stretch(c[None, :], row["arm_scale"]) - c)[0]
                co[:, 0] += delta[0]
                co[:, 1] += delta[1]
            elif name in ARM_PARTS:
                co = radial_stretch(co, row["arm_scale"])
            o.data.vertices.foreach_set("co", np.ascontiguousarray(co).reshape(-1))
            o.data.update()
        pod.hide_render = not row["pod"]
        lo = np.array([np.inf] * 3)
        hi = -lo.copy()
        for o in meshes:
            if o is pod and pod.hide_render:
                continue
            co = _obj_co(o)
            lo = np.minimum(lo, co.min(axis=0))
            hi = np.maximum(hi, co.max(axis=0))
        corners = [(float(px), float(py), float(pz))
                   for px in (float(lo[0]), float(hi[0]))
                   for py in (float(lo[1]), float(hi[1]))
                   for pz in (float(lo[2]), float(hi[2]))]
        framing = R.framing_entry(R.frame_uv(bpy.data.objects["cam_variant"], corners), u_max)
        clean_name = "variant_%02d_clean.png" % k
        sil_name = "variant_%02d_sil.png" % k
        secs = 0.0
        t1 = time.time()
        R.do_render(os.path.join(d, clean_name))
        secs += time.time() - t1
        a_clean = _load_rgba(os.path.join(d, clean_name))
        alpha_clean = float(np.mean(a_clean[:, :, 3] > 0.5)) if a_clean.shape[2] == 4 else None
        t1 = time.time()
        R.set_view_transform("Standard")
        apply_light_scale("Standard")
        sil = _sil_material(k, n)
        set_materials(sil)
        R.do_render(os.path.join(d, sil_name))
        a_sil = _load_rgba(os.path.join(d, sil_name))
        alpha_sil = float(np.mean(a_sil[:, :, 3] > 0.5)) if a_sil.shape[2] == 4 else None
        set_materials({nm: CLEAN_MAT[nm] for nm in CLEAN_MAT})
        R.set_view_transform("AgX")
        apply_light_scale("AgX")
        secs += time.time() - t1
        cleans.append(_load_topdown(os.path.join(d, clean_name)))
        sils.append(_load_topdown(os.path.join(d, sil_name)))
        vreports.append(dict(row) | {
            "clean": clean_name,
            "sil": sil_name,
            "alpha_frac_clean": alpha_clean,
            "alpha_frac_sil": alpha_sil,
            "framing": framing,
        })
        say("variant %02d in %.1f s, alpha %.3f/%.3f, framing %s, dist %.2f m"
            % (k, secs, alpha_clean, alpha_sil,
               "pass" if framing["pass"] else "FAIL", CAMERAS["cam_variant"][2]))
    grid_clean = "grid_clean.png"
    grid_sil = "grid_sil.png"
    _write_grid_image(cleans, os.path.join(d, grid_clean))
    _write_grid_image(sils, os.path.join(d, grid_sil))
    gpu_end = R.gpu_record()
    for o in meshes:
        o.data.vertices.foreach_set("co", np.ascontiguousarray(original[o.name]).reshape(-1))
        o.data.update()
    report = {
        "tool": TOOL,
        "version": VERSION,
        "credit": CREDIT,
        "claim": CLAIM,
        "solved": False,
        "n": n,
        "layers_note": "each variant PNG is an RGBA layer for a parallax tunnel composited later",
        "variants": vreports,
        "grid": {"cols": GRID[0], "rows": int(math.ceil(n / float(GRID[0]))),
                 "clean": grid_clean, "sil": grid_sil},
        "blender": bpy.app.version_string,
        "samples": int(args.samples),
        "device": args.device,
        "seconds": float(time.time() - t_total),
        "gpu_start": gpu_start,
        "gpu_end": gpu_end,
    }
    with open(os.path.join(d, "variants.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=1)
    say("variants.json written: %d variants, grids %s" % (n, grid_clean + " + " + grid_sil))
    return 0


# ---------------------------------------------------------------- check

def _tt_check(out, kind, vt_want, F, cp_range):
    d = os.path.join(out, "turntable_" + kind)
    problems = []
    if not os.path.isdir(d):
        return False, "dir missing: " + d
    expected = set("frame_%04d.png" % i for i in range(1, F + 1))
    actual = set(os.listdir(d))
    extra = sorted(actual - expected)
    missing = sorted(expected - actual)
    if extra:
        problems.append("extra " + ", ".join(extra[:5]))
    if missing:
        problems.append("missing " + str(len(missing)) + " e.g. " + missing[0])
    for i in range(1, F + 1):
        p = os.path.join(d, "frame_%04d.png" % i)
        if os.path.isfile(p):
            w, h = R.png_size(p)
            if (w, h) != (1920, 1080):
                problems.append("frame " + str(i) + " size " + str(w) + "x" + str(h))
    jpath = os.path.join(out, "turntable_" + kind + ".json")
    if not os.path.isfile(jpath):
        problems.append("turntable_" + kind + ".json missing")
    else:
        j = _load_json(jpath)
        if j.get("view_transform") != vt_want:
            problems.append("json view " + str(j.get("view_transform")) + " != " + vt_want)
        for entry in (j.get("framing") or []):
            if not entry.get("pass"):
                problems.append("framing frame " + str(entry.get("frame")) + " failed")
        if cp_range is not None:
            for i in (1, F):
                p = os.path.join(d, "frame_%04d.png" % i)
                if os.path.isfile(p):
                    diff = R.bar_check(p, cp_range[0], cp_range[1], S.CP_RAMP)["max_diff"]
                    if diff > R.BAR_TOL:
                        problems.append("frame " + str(i) + " bar max_diff " + format(diff, ".4f"))
    if problems:
        return False, "; ".join(problems)
    return True, "exactly " + str(F) + " frames 1920x1080, view " + vt_want + ", framing pass"


def run_check(args):
    _require_arg(args.out, "out")
    _check_out(args.out)
    out = args.out
    if not os.path.isdir(out):
        raise S.SceneRefusal("DR-INPUT", "out dir not found: " + out)
    spath = os.path.join(out, "stills.json")
    if not os.path.isfile(spath):
        raise S.SceneRefusal("DR-INPUT", "stills.json not found under " + out)
    stills = _load_json(spath)
    shots = stills.get("shots", {})
    ranges = stills.get("ranges") or {}
    F = int(args.frames_expected)
    RF = int(args.reveal_expected)
    NV = int(args.variants_expected)
    checks = {}

    def record(name, ok, detail):
        checks[name] = {"pass": bool(ok), "detail": detail}

    problems = []
    for name in SHOTS:
        p = os.path.join(out, name + ".png")
        if not os.path.isfile(p):
            problems.append(name + " png missing")
            continue
        w, h = R.png_size(p)
        if (w, h) != (1920, 1080):
            problems.append(name + " size " + str(w) + "x" + str(h))
    record("C-STILLS", not problems,
           "; ".join(problems) if problems else str(len(SHOTS)) + "/" + str(len(SHOTS)) + " stills 1920x1080")

    bad = []
    cp_range = ranges.get("cp")
    sp_range = ranges.get("speed")
    if cp_range is None:
        bad.append("cp_range missing from stills.json")
    else:
        p = os.path.join(out, "hero_cp.png")
        if os.path.isfile(p):
            diff = R.bar_check(p, cp_range[0], cp_range[1], S.CP_RAMP)["max_diff"]
            if diff > R.BAR_TOL:
                bad.append("hero_cp bar max_diff " + format(diff, ".4f") + " > " + str(R.BAR_TOL))
        else:
            bad.append("hero_cp png missing")
    if sp_range is None:
        bad.append("speed_range missing from stills.json")
    else:
        p = os.path.join(out, "slice.png")
        if os.path.isfile(p):
            diff = R.bar_check(p, sp_range[0], sp_range[1], S.SPEED_RAMP)["max_diff"]
            if diff > R.BAR_TOL:
                bad.append("slice bar max_diff " + format(diff, ".4f") + " > " + str(R.BAR_TOL))
        else:
            bad.append("slice png missing")
    record("C-BARS", not bad, "; ".join(bad) if bad else "hero_cp + slice bars within " + str(R.BAR_TOL))

    bad = []
    for name in SHOTS:
        p = os.path.join(out, name + ".png")
        if not os.path.isfile(p):
            bad.append(name + " png missing")
            continue
        clip = R.clip_fraction(p)
        if clip > R.CLIP_MAX:
            bad.append(name + " clip " + format(clip, ".4f") + " > " + str(R.CLIP_MAX))
    record("C-CLIP", not bad, "; ".join(bad) if bad else "every still clip <= " + str(R.CLIP_MAX))

    bad = []
    for name in SHOTS:
        fr = (shots.get(name) or {}).get("framing")
        if not fr or not fr.get("pass"):
            bad.append(name + " framing missing/failed")
    record("C-FRAMING", not bad, "; ".join(bad) if bad else "every still framing passes")

    cp_map = stills.get("cp_map")
    bad = []
    if not cp_map:
        bad.append("cp_map missing (no cp shot rendered)")
    else:
        if float(cp_map.get("max_dist") or 1e9) > CP_MAP_MAX:
            bad.append("cp_map max_dist " + format(cp_map["max_dist"], ".4f") + " > " + str(CP_MAP_MAX))
        md = cp_map.get("mean_dist")
        if md is None or float(md) > CP_MAP_MEAN_MAX:
            bad.append("cp_map mean_dist " + format(md, ".4f") + " > " + str(CP_MAP_MEAN_MAX))
    record("C-CPMAP", not bad, "; ".join(bad) if bad else
           "max_dist " + format(cp_map["max_dist"], ".4f") + ", mean_dist " + format(cp_map["mean_dist"], ".4f"))

    ok, detail = _tt_check(out, "clean", "AgX", F, None)
    record("C-TT-CLEAN", ok, detail)
    ok, detail = _tt_check(out, "cp", "Standard", F, cp_range)
    record("C-TT-CP", ok, detail)

    bad = []
    d = os.path.join(out, "reveal")
    expected = set("frame_%04d.png" % i for i in range(1, RF + 1))
    actual = set(os.listdir(d)) if os.path.isdir(d) else set()
    if expected - actual:
        bad.append("missing " + str(len(expected - actual)) + " reveal frames")
    if actual - expected:
        bad.append("extra " + ", ".join(sorted(actual - expected)[:5]))
    lit_first = lit_last = None
    if not bad:
        for i in range(1, RF + 1):
            p = os.path.join(d, "frame_%04d.png" % i)
            w, h = R.png_size(p)
            if (w, h) != (1920, 1080):
                bad.append("frame " + str(i) + " size " + str(w) + "x" + str(h))
        p1 = os.path.join(d, "frame_%04d.png" % 1)
        p2 = os.path.join(d, "frame_%04d.png" % RF)
        lit_first = float(np.mean(R.load_px(p1).max(axis=2) >= LIT))
        lit_last = float(np.mean(R.load_px(p2).max(axis=2) >= LIT))
        if lit_last < lit_first:
            bad.append("lit fraction fell " + format(lit_first, ".4f") + " -> " + format(lit_last, ".4f"))
        if RF == R.REVEAL_FRAMES and not lit_last > lit_first + 0.002:
            bad.append("full reveal lit growth " + format(lit_last - lit_first, ".4f") + " <= 0.002")
    record("C-REVEAL", not bad, "; ".join(bad) if bad else
           "exactly " + str(RF) + " frames, lit " + format(lit_first, ".4f") + " -> " + format(lit_last, ".4f"))

    bad = []
    p1 = os.path.join(out, "before.png")
    p2 = os.path.join(out, "after.png")
    if not (os.path.isfile(p1) and os.path.isfile(p2)):
        bad.append("before/after png missing")
    else:
        a = R.load_px(p1)
        b = R.load_px(p2)
        if a.shape != b.shape:
            bad.append("before/after size mismatch")
        else:
            mad = float(np.mean(np.abs(a - b)))
            if not mad > 0.002:
                bad.append("mean |a-b| " + format(mad, ".4f") + " <= 0.002")
    record("C-WALLS", not bad, "; ".join(bad) if bad else "before/after differ")

    bad = []
    vpath = os.path.join(out, "multiverse", "variants.json")
    if not os.path.isfile(vpath):
        bad.append("variants.json missing")
    else:
        v = _load_json(vpath)
        rows = v.get("variants") or []
        if v.get("n") != NV:
            bad.append("n " + str(v.get("n")) + " != " + str(NV))
        if v.get("claim") != CLAIM:
            bad.append("claim mismatch")
        if v.get("solved") is not False:
            bad.append("top-level solved not false")
        if len(rows) != NV:
            bad.append("rows " + str(len(rows)) + " != " + str(NV))
        masks = []
        for row in rows:
            if row.get("solved") is not False:
                bad.append("variant " + str(row.get("index")) + " solved not false")
            for key, af_key in (("clean", "alpha_frac_clean"), ("sil", "alpha_frac_sil")):
                p = os.path.join(out, "multiverse", row.get(key, ""))
                if not os.path.isfile(p):
                    bad.append(str(row.get("index")) + " " + key + " png missing")
                    continue
                w, h = R.png_size(p)
                if (w, h) != (VARIANT_RES, VARIANT_RES):
                    bad.append(str(row.get("index")) + " " + key + " size " + str(w) + "x" + str(h))
                af = row.get(af_key)
                if af is None or not (ALPHA_FRAC[0] <= af <= ALPHA_FRAC[1]):
                    bad.append(str(row.get("index")) + " " + af_key + " " + format(af, ".4f")
                               + " outside " + str(ALPHA_FRAC))
            fr = row.get("framing")
            if not fr or not fr.get("pass"):
                bad.append("variant " + str(row.get("index")) + " framing failed")
            p = os.path.join(out, "multiverse", row.get("sil", ""))
            if os.path.isfile(p):
                masks.append(_load_rgba(p)[:, :, 3] > 0.5)
        worst = None
        for i in range(len(masks)):
            for j in range(i + 1, len(masks)):
                c = int(np.count_nonzero(masks[i] ^ masks[j]))
                if worst is None or c < worst:
                    worst = c
        if worst is not None and worst < MASK_MIN_PX:
            bad.append("closest sil masks differ in only " + str(worst) + " px < " + str(MASK_MIN_PX))
    record("C-VARIANTS", not bad, "; ".join(bad) if bad else
           str(NV) + " variants, alphas in " + str(ALPHA_FRAC) + ", masks >= " + str(MASK_MIN_PX) + " px apart")

    bad = []
    for fn in ("grid_clean.png", "grid_sil.png"):
        p = os.path.join(out, "multiverse", fn)
        if not os.path.isfile(p):
            bad.append(fn + " missing")
            continue
        w, h = R.png_size(p)
        if (w, h) != (1920, 1080):
            bad.append(fn + " size " + str(w) + "x" + str(h))
    record("C-GRID", not bad, "; ".join(bad) if bad else "both grid plates 1920x1080")

    bad = []
    cred = None
    cpath = os.path.join(out, "CREDITS.txt")
    for fn in ("LICENSE.txt", "LICENSE", "CREDITS.txt"):
        if not os.path.isfile(os.path.join(out, fn)):
            bad.append(fn + " missing")
    if os.path.isfile(cpath):
        cred = open(cpath, "r", encoding="utf-8").read()
        for needle in (CREDIT, "No endorsement", "not solved"):
            if needle not in cred:
                bad.append("CREDITS.txt lacks " + repr(needle))
    else:
        bad.append("CREDITS.txt unreadable")
    record("C-LICENSE", not bad, "; ".join(bad) if bad else "licence trio present, credits carry the model line")

    gate = judge({cid: checks[cid]["pass"] for cid in CHECK_IDS if cid in checks})
    report = {
        "tool": TOOL,
        "version": VERSION,
        "out": _posix(out),
        "frames_expected": F,
        "reveal_expected": RF,
        "variants_expected": NV,
        "checks": checks,
        "gate": gate,
    }
    with open(os.path.join(out, "render.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=1)
    if gate["pass"]:
        print("DRONE-RENDER GATE PASS", flush=True)
        return 0
    print("DRONE-RENDER GATE FAIL " + ",".join(gate["failed"]), flush=True)
    return 1


# ---------------------------------------------------------------- selftest

def _t1():
    rows = variant_table(36)
    assert len(rows) == 36
    tuples = [(r["arm_scale"], r["prop_scale"], r["cant_deg"], r["pod"]) for r in rows]
    assert len(set(tuples)) == 36, "tuples not distinct"
    assert tuples[0] == (1.0, 1.0, 0.0, False) and rows[0]["baseline"] is True
    assert tuples[4] == (0.8, 0.85, 0.0, False)
    assert tuples[35] == (1.25, 1.2, 12.0, True)
    assert all(r["solved"] is False for r in rows)
    for bad_n in (37, 0):
        try:
            variant_table(bad_n)
            raise AssertionError("DR-N not raised for " + str(bad_n))
        except S.SceneRefusal as e:
            assert e.code == "DR-N", e.code


def _t2():
    a = radial_stretch(np.array([[0.246, 0.0, 0.01]]), 1.25)
    assert np.allclose(a, [[0.29, 0.0, 0.01]], atol=1e-12), str(a)
    b = radial_stretch(np.array([[0.05, 0.0, 0.0]]), 1.25)
    assert np.allclose(b, [[0.05, 0.0, 0.0]], atol=1e-12), str(b)
    c = radial_stretch(np.array([[0.174, 0.174, 0.02]]), 0.8)
    r = math.hypot(0.174, 0.174)
    r2 = ARM_R0 + (r - ARM_R0) * 0.8
    want = np.array([[0.174 * r2 / r, 0.174 * r2 / r, 0.02]])
    assert np.allclose(c, want, atol=1e-12), str(c) + " vs " + str(want)


def _t3():
    a = scale_about(np.array([[0.2, -0.1, 0.05]]), (0.174, -0.174, 0.0), 2.0)
    assert np.allclose(a, [[0.226, -0.026, 0.05]], atol=1e-12), str(a)
    b = cant_rotate(np.array([[1.0, 0.0, 1.0]]), (1.0, 0.0, 0.0), 90.0, z0=0.0)
    assert np.allclose(b, [[2.0, 0.0, 0.0]], atol=1e-12), str(b)
    c = cant_rotate(np.array([[1.0, 0.0, 1.0]]), (1.0, 0.0, 0.0), 0.0, z0=0.0)
    assert np.allclose(c, [[1.0, 0.0, 1.0]], atol=1e-12), str(c)


def _t4():
    imgs = []
    for k in range(4):
        im = np.empty((8, 8, 4), dtype=np.float32)
        im[:, :, 0] = im[:, :, 1] = im[:, :, 2] = (k + 1) / 10.0
        im[:, :, 3] = 1.0
        imgs.append(im)
    out = compose_grid(imgs, 2, 20, 20)
    assert out.shape == (20, 20, 4)
    for k, (r0, c0) in enumerate(((0, 0), (0, 10), (10, 0), (10, 10))):
        tile = out[r0:r0 + 10, c0:c0 + 10]
        assert np.allclose(tile[:, :, 0], (k + 1) / 10.0, atol=1e-6), str(k)
        assert np.allclose(tile[:, :, 3], 1.0, atol=1e-6)
    out2 = compose_grid(imgs, 2, 24, 20)
    assert float(np.min(out2[0:10, 0, 3])) == 0.0, "column 0 not empty"
    assert np.allclose(out2[0:10, 1:11, 0], 0.1, atol=1e-6)


def _t5():
    pts = np.array([
        [0.5, 0.0, 0.0], [0.4, 0.0, 0.0], [0.3, 0.0, 0.0], [0.2, 0.0, 0.0],
        [0.45, 0.03, 0.0], [0.35, 0.03, 0.0], [0.25, 0.03, 0.0], [0.15, 0.03, 0.0],
        [0.45, 0.12, 0.0], [0.2, 0.12, 0.0],
        [0.5, 0.5, 0.0], [0.4, 0.5, 0.0]], dtype=np.float64)
    speed = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0])
    offsets = np.array([0, 4, 8, 10, 12], dtype=np.int64)
    group = np.array([0, 1, 1, 0], dtype=np.int8)
    seeds = np.array([[0.4, 0.0, 0.0], [0.45, 0.03, 0.0],
                      [0.45, 0.12, 0.0], [0.4, 0.5, 0.0]], dtype=np.float64)
    P, U, O, kept = trim_lines(pts, speed, offsets, group, seeds)
    assert kept == [0, 1], str(kept)
    assert O.tolist() == [0, 3, 5], str(O.tolist())
    assert np.allclose(P[:, 0], [0.4, 0.3, 0.2, 0.25, 0.15], atol=1e-12), str(P[:, 0].tolist())
    assert np.allclose(U, [2.0, 3.0, 4.0, 7.0, 8.0], atol=1e-12), str(U.tolist())


def _t6():
    want = {"prop_2_cw": "prop", "motor_bell_1_Bellside": "bell",
            "motor_base_3": "metal", "frame_Metal": "metal",
            "frame_FMURubber": "rubber", "frame_LandingFoam": "rubber",
            "frame_CarbonFiber": "carbon", "frame_LandingPlastic": "carbon",
            "frame_FMUK66": "pcb", "frame_RailsAntennaHolder": "pcb"}
    for name, mat in want.items():
        got = part_material(name)
        assert got == mat, name + ": " + got + " != " + mat


def _t7():
    want = {"motor_base_0": 0, "motor_bell_3_BellHead.001": 3, "prop_1_ccw": 1,
            "frame_Metal": None}
    for name, i in want.items():
        got = motor_index(name)
        assert got == i, name + ": " + str(got) + " != " + str(i)


def _t8():
    def refuse(argv):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = main(list(argv))
        return rc, buf.getvalue()

    tmp = tempfile.mkdtemp(prefix="drone-render-t8-")
    try:
        out_in = os.path.normpath(os.path.join(REPO, "tools", "drone", "_x"))
        rc, so = refuse(["stills", "--blend", "Z:/definitely-missing.blend",
                         "--post", tmp, "--out", out_in])
        assert rc == 2, str(rc)
        assert "refused: DR-OUT:" in so, so
        assert not os.path.exists(out_in) or not os.listdir(out_in), "DR-OUT wrote something"
        out2 = os.path.join(tmp, "o2")
        rc, so = refuse(["stills", "--blend", "Z:/definitely-missing.blend",
                         "--post", tmp, "--out", out2])
        assert rc == 2 and "refused: DR-INPUT:" in so, (rc, so)
        rc, so = refuse(["turntable", "--kind", "smoke", "--blend", "Z:/definitely-missing.blend",
                         "--post", tmp, "--out", out2])
        assert rc == 2 and "refused: DR-KIND:" in so, (rc, so)
        rc, so = refuse(["stills", "--blend", "Z:/definitely-missing.blend",
                         "--post", tmp, "--out", out2, "--shots", "hero,nope"])
        assert rc == 2 and "refused: DR-SHOT:" in so, (rc, so)
        rc, so = refuse(["multiverse", "--blend", "Z:/definitely-missing.blend",
                         "--rotors", "Z:/definitely-missing.json", "--out", out2, "--n", "0"])
        assert rc == 2 and "refused: DR-N:" in so, (rc, so)
        assert not os.path.exists(out2) or not os.listdir(out2), "refusal wrote something"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _t9():
    good = judge({cid: True for cid in CHECK_IDS})
    assert good == {"pass": True, "failed": []}, str(good)
    ck = {cid: True for cid in CHECK_IDS}
    ck["C-WALLS"] = False
    del ck["C-GRID"]
    bad = judge(ck)
    assert bad == {"pass": False, "failed": ["C-WALLS", "C-GRID"]}, str(bad)


def _t10():
    pts = np.array([[0.5, 0.0, 0.0], [0.4, 0.0, 0.0], [0.3, 0.0, 0.0],
                    [0.3, 0.0, 0.0], [0.2, 0.0, 0.0]], dtype=np.float64)
    speed = np.array([10.0, 10.0, 10.0, 5.0, 5.0], dtype=np.float64)
    offsets = np.array([0, 3, 5], dtype=np.int64)
    t = drone_reveal_times(pts, speed, offsets, 15.0)
    want = np.array([0.0, 0.01, 0.02, 0.0133333, 0.0333333])
    assert np.allclose(t, want, atol=1e-6), str(t.tolist())


def selftest():
    """R11: T1-T10 inside Blender, no rendering; temp files only under
    tempfile.mkdtemp()."""
    tests = [("T1", _t1), ("T2", _t2), ("T3", _t3), ("T4", _t4), ("T5", _t5),
             ("T6", _t6), ("T7", _t7), ("T8", _t8), ("T9", _t9), ("T10", _t10)]
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
    if argv is None:
        argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    if argv and argv[0] == "--selftest":
        return selftest()
    ap = argparse.ArgumentParser(prog="render.py", description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd")
    sp = sub.add_parser("stills")
    sp.add_argument("--blend", required=True)
    sp.add_argument("--post", required=True)
    sp.add_argument("--out", required=True)
    sp.add_argument("--device", default="GPU", choices=["GPU", "CPU"])
    sp.add_argument("--samples", type=int, default=128)
    sp.add_argument("--shots", default="hero_clean,hero_cp,downwash,slice,before,after")
    tp = sub.add_parser("turntable")
    tp.add_argument("--kind", required=True)
    tp.add_argument("--blend", required=True)
    tp.add_argument("--post", required=True)
    tp.add_argument("--out", required=True)
    tp.add_argument("--frames", default="1-" + str(N_FRAMES))
    tp.add_argument("--device", default="GPU", choices=["GPU", "CPU"])
    tp.add_argument("--samples", type=int, default=96)
    rp = sub.add_parser("reveal")
    rp.add_argument("--blend", required=True)
    rp.add_argument("--post", required=True)
    rp.add_argument("--out", required=True)
    rp.add_argument("--frames", default="1-" + str(REVEAL_FRAMES))
    rp.add_argument("--device", default="GPU", choices=["GPU", "CPU"])
    rp.add_argument("--samples", type=int, default=96)
    mp = sub.add_parser("multiverse")
    mp.add_argument("--blend", required=True)
    mp.add_argument("--rotors", required=True)
    mp.add_argument("--out", required=True)
    mp.add_argument("--n", type=int, default=36)
    mp.add_argument("--device", default="GPU", choices=["GPU", "CPU"])
    mp.add_argument("--samples", type=int, default=64)
    kp = sub.add_parser("check")
    kp.add_argument("--out", required=True)
    kp.add_argument("--frames-expected", type=int, default=240)
    kp.add_argument("--reveal-expected", type=int, default=150)
    kp.add_argument("--variants-expected", type=int, default=36)
    args = ap.parse_args(argv)
    try:
        if args.cmd == "stills":
            return run_stills(args)
        if args.cmd == "turntable":
            return run_turntable(args)
        if args.cmd == "reveal":
            return run_reveal(args)
        if args.cmd == "multiverse":
            return run_multiverse(args)
        if args.cmd == "check":
            return run_check(args)
    except S.SceneRefusal as e:
        print("refused: " + e.code + ": " + e.message, flush=True)
        return 2
    ap.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
