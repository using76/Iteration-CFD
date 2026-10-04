#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-or-later
# meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
# This Blender Python API script is free software: you can redistribute it
# and/or modify it under the terms of the GNU General Public License as
# published by the Free Software Foundation, either version 2 of the License,
# or (at your option) any later version. It is distributed WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE. See tools/promo/COPYING for the licence text and
# tools/promo/LICENSE-NOTICE.md for why this file is not under the
# repository's Prosperity licence (a separate program run by Blender,
# aggregated with meteor-cfd, not linked into it).
# No GPL-licensed source was consulted.

"""Render the F1 promo from the real solve with Blender Cycles.

Three subcommands (argv after "--"):

stills     five 1920x1080 stills per the SHOTS table below; every bar-carrying
           still has its colour bar drawn IN THE SAME RENDER
turntable  --kind cp (Cp field, Standard) or --kind clean (paint, AgX):
           turntable_<kind>/frame_NNNN.png frames, resumable (existing
           non-empty frames are skipped and counted)
check      judges an --out directory written by the two above into
           render.json and prints GATE PASS / GATE FAIL

The SHOTS table (stills; hidden/shown is hide_render):

| shot | camera | car | car_cut | streamlines | slice | view transform | bar |
|---|---|---|---|---|---|---|---|
| hero | cam_hero | field | hidden | hidden | hidden | Standard | cp |
| side | cam_side | field | hidden | hidden | hidden | Standard | cp |
| top_rear | cam_top_rear | field | hidden | hidden | hidden | Standard | cp |
| streamlines | cam_stream | paint | hidden | shown | hidden | AgX | none |
| slice | cam_slice | hidden | shown | hidden | shown | Standard | speed |

Colour-bar rule: the bar is geometry parented to the shot's own camera, lit
by an Emission shader only, and rendered in the same render as the shot, so
it passes through the same view transform as the image. bar_check reads the
bar's pixels back from the PNG (Non-Color) and demands srgb(ramp colour)
within BAR_TOL; under AgX the same bar misses by more than 0.04, which is
what proves which view transform the still went through.

Outputs (all under --out, never inside the repository): stills writes
<shot>.png, stills.json, CREDITS.txt and LICENSE.txt; turntable writes
turntable_<kind>/frame_NNNN.png and turntable_<kind>.json; check writes
render.json.

CC BY 4.0 rule: the car model ("F1 2026 concept" by Qvist_Designs) and every
file made from it stay outside the repository under --out; the model's
LICENSE.txt is copied beside the outputs and its attribution line goes into
CREDITS.txt and the video credits.

Run (Blender 5.1, headless):

blender -b --factory-startup --python-exit-code 1 --python tools/promo/render.py -- --selftest
blender -b --factory-startup --python-exit-code 1 --python tools/promo/render.py -- stills --glb GLB --sim-geom DIR --post DIR --out DIR [--device GPU|CPU] [--samples N] [--shots hero,side,top_rear,streamlines,slice] [--cp-range LO HI] [--speed-range LO HI]
blender -b --factory-startup --python-exit-code 1 --python tools/promo/render.py -- turntable --kind cp|clean --glb GLB --sim-geom DIR --out DIR [--post DIR] [--device GPU|CPU] [--samples N] [--frames A-B] [--cp-range LO HI]
blender -b --factory-startup --python-exit-code 1 --python tools/promo/render.py -- check --out DIR [--frames-expected N]

It imports ../scene.py (GPL-2.0-or-later too) for the studio scene.
"""

import argparse
import json
import math
import os
import shutil
import struct
import subprocess
import sys
import tempfile
import time

import numpy as np
import bpy
from mathutils import Matrix, Vector

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import scene as S

TOOL = "tools/promo/render.py"
VERSION = "promo-render/1"
N_FRAMES = 240
BAR = (0.925, 0.945, 0.20, 0.80)   # u0, u1, v0, v1 image fractions; v UP
BAR_LEVELS = 64
BAR_DEPTH = 0.5                    # metres in front of the camera
BAR_TOL = 0.02
BAR_SAMPLES = 32
TEXT_H = 0.03                      # label height, fraction of image height
PLATE = (0.865, 0.965, 0.15, 0.88)  # plate u0, u1, v0, v1 image fractions; v UP
LUM_MIN = 0.02

SHOTS = {
    "hero": {"camera": "cam_hero", "car": "field", "car_cut": False,
             "streamlines": False, "slice": False, "vt": "Standard", "bar": "cp"},
    "side": {"camera": "cam_side", "car": "field", "car_cut": False,
             "streamlines": False, "slice": False, "vt": "Standard", "bar": "cp"},
    "top_rear": {"camera": "cam_top_rear", "car": "field", "car_cut": False,
                 "streamlines": False, "slice": False, "vt": "Standard", "bar": "cp"},
    "streamlines": {"camera": "cam_stream", "car": "paint", "car_cut": False,
                    "streamlines": True, "slice": False, "vt": "AgX", "bar": None},
    "slice": {"camera": "cam_slice", "car": None, "car_cut": True,
              "streamlines": False, "slice": True, "vt": "Standard", "bar": "speed"},
}

BARREG = {}


def say(msg):
    print("[render] " + msg, flush=True)


def require_arg(value, name):
    if value is None or value == "":
        raise S.SceneRefusal("RD-INPUT", "--" + name + " is required")


def require_file(path, what):
    if not os.path.isfile(path):
        raise S.SceneRefusal("RD-INPUT", what + " not found: " + path)


def validate_out(out):
    """RD-OUT when --out resolves inside the repository (two levels above this file)."""
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    a = os.path.normcase(os.path.abspath(out))
    r = os.path.normcase(repo)
    if a == r or a.startswith(r + os.sep):
        raise S.SceneRefusal("RD-OUT", "--out " + out + " resolves inside the repository (" + repo + ")")


def parse_frames(s):
    """'A-B' -> (A, B); 'A' -> (A, A); 1 <= A <= B <= N_FRAMES else RD-INPUT."""
    text = str(s).strip()
    try:
        if "-" in text:
            a, b = text.split("-", 1)
            A, B = int(a), int(b)
        else:
            A = B = int(text)
    except ValueError:
        raise S.SceneRefusal("RD-INPUT", "bad --frames: " + str(s))
    if not (1 <= A <= B <= N_FRAMES):
        raise S.SceneRefusal("RD-INPUT", "bad --frames " + str(s) + ": need 1 <= A <= B <= " + str(N_FRAMES))
    return (A, B)


def default_ranges(numbers):
    """((cp_lo, cp_hi), (sp_lo, sp_hi)) rounded out to 2 decimals / whole m/s."""
    if "cp" not in numbers or "streamlines" not in numbers:
        raise S.SceneRefusal("RD-INPUT", "numbers.json missing cp or streamlines")
    cp = numbers["cp"]
    sl = numbers["streamlines"]
    for k in ("p01", "p99"):
        if k not in cp:
            raise S.SceneRefusal("RD-INPUT", "numbers.json cp missing key: " + k)
    if "speed_p99" not in sl:
        raise S.SceneRefusal("RD-INPUT", "numbers.json streamlines missing speed_p99")
    cp_lo = float(math.floor(100.0 * float(cp["p01"])) / 100.0)
    cp_hi = float(math.ceil(100.0 * float(cp["p99"])) / 100.0)
    sp_lo = 0.0
    sp_hi = float(math.ceil(float(sl["speed_p99"])))
    return (cp_lo, cp_hi), (sp_lo, sp_hi)


def resolve_ranges(numbers, cp_override, sp_override):
    """default_ranges with --cp-range / --speed-range applied; lo >= hi refuses."""
    (cp_lo, cp_hi), (sp_lo, sp_hi) = default_ranges(numbers)
    if cp_override is not None:
        cp_lo, cp_hi = float(cp_override[0]), float(cp_override[1])
    if sp_override is not None:
        sp_lo, sp_hi = float(sp_override[0]), float(sp_override[1])
    if cp_lo >= cp_hi:
        raise S.SceneRefusal("RD-RANGE", "cp range lo " + str(cp_lo) + " >= hi " + str(cp_hi))
    if sp_lo >= sp_hi:
        raise S.SceneRefusal("RD-RANGE", "speed range lo " + str(sp_lo) + " >= hi " + str(sp_hi))
    return (cp_lo, cp_hi), (sp_lo, sp_hi)


def parse_license(text):
    """The promo model's LICENSE.txt -> title_line, source, licence, attribution."""
    lines = text.splitlines()
    if not lines:
        raise S.SceneRefusal("RD-LICENSE", "LICENSE.txt is empty")
    def after(prefix):
        for ln in lines:
            if ln.startswith(prefix):
                return ln[len(prefix):].strip()
        raise S.SceneRefusal("RD-LICENSE", "LICENSE.txt has no line starting with: " + prefix)
    return {
        "title_line": lines[0].strip(),
        "source": after("Source: "),
        "licence": after("Licence: "),
        "attribution": after("Attribution line for the video: "),
    }


CREDITS_CHANGES = ("Changes: scaled to metres, closed into a watertight simulation surface, "
                   "meshed and solved with meteor-cfd; the colours on the car, the streamlines "
                   "and the slice are CFD results and are not part of the original model.")


def credits_text(lic, blender_version):
    lines = [
        "Credits for the meteor-cfd F1 promo renders",
        "",
        "Car model: " + lic["attribution"],
        "Original: " + lic["title_line"],
        "Source: " + lic["source"],
        "Licence: " + lic["licence"],
        CREDITS_CHANGES,
        "Rendered with Blender " + blender_version + " (Cycles).",
    ]
    return "\n".join(lines) + "\n"


def png_size(path):
    """(w, h) straight from the IHDR chunk (signature check, no decoding)."""
    with open(path, "rb") as f:
        head = f.read(24)
    if len(head) < 24 or head[:8] != b"\x89PNG\r\n\x1a\n":
        raise S.SceneRefusal("RD-PNG", "not a PNG: " + path)
    w = struct.unpack(">I", head[16:20])[0]
    h = struct.unpack(">I", head[20:24])[0]
    return (int(w), int(h))


def srgb(c):
    """The sRGB electro-optical transfer, elementwise (numpy)."""
    c = np.asarray(c, dtype=np.float64)
    return np.where(c <= 0.0031308,
                    12.92 * c,
                    1.055 * np.power(np.clip(c, 0.0031308, None), 1.0 / 2.4) - 0.055)


def bar_levels(lo, hi, ramp, n):
    """(n+1, 4) float32 ramp colours at lo + (k/n)*(hi - lo)."""
    vals = [lo + (k / float(n)) * (hi - lo) for k in range(n + 1)]
    return np.asarray(S.ramp_colors(vals, lo, hi, ramp), dtype=np.float32)


def frame_hw_hh(cam_data, res_x, res_y, depth):
    """The frame's half width and half height at z = -depth, in the camera's local units."""
    if cam_data.type == "ORTHO":
        hw = cam_data.ortho_scale / 2.0
    else:
        hw = depth * (cam_data.sensor_width / 2.0) / cam_data.lens
    return hw, hw * float(res_y) / float(res_x)


def overlay_rect(cam_data, res_x, res_y, rect, depth):
    """The rect in the camera's LOCAL frame at z = -depth (camera looks down -Z, +X right, +Y up)."""
    u0, u1, v0, v1 = rect
    hw, hh = frame_hw_hh(cam_data, res_x, res_y, depth)
    return ((u0 - 0.5) * 2.0 * hw,
            (u1 - 0.5) * 2.0 * hw,
            (v0 - 0.5) * 2.0 * hh,
            (v1 - 0.5) * 2.0 * hh)


def load_px(path):
    """A PNG's RGB as (h, w, 3) float64, stored values (Non-Color; row 0 = bottom)."""
    img = bpy.data.images.load(path)
    img.colorspace_settings.name = "Non-Color"
    w, h = img.size
    ch = img.channels
    px = np.empty(w * h * ch, dtype=np.float32)
    img.pixels.foreach_get(px)
    bpy.data.images.remove(img)
    return px.reshape(h, w, ch)[:, :, :3].astype(np.float64)


def bar_check(path, lo, hi, ramp, rect=BAR):
    """{max_diff, n}: the bar's stored pixels vs srgb(ramp(t)) sampled up the bar."""
    u0, u1, v0, v1 = rect
    w, h = png_size(path)
    px = load_px(path)
    col = int(((u0 + u1) / 2.0) * w)
    max_diff = 0.0
    for v in np.linspace(v0 + 0.01, v1 - 0.01, BAR_SAMPLES):
        row = int(v * h)
        if row >= h:
            row = h - 1
        t = ((row + 0.5) / h - v0) / (v1 - v0)
        expected = srgb(S.ramp_colors([lo + t * (hi - lo)], lo, hi, ramp)[0, :3])
        got = px[row, col]
        max_diff = max(max_diff, float(np.max(np.abs(got - expected))))
    return {"max_diff": max_diff, "n": BAR_SAMPLES}


def gpu_record():
    """nvidia-smi state; null when nvidia-smi is not usable."""
    try:
        g = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.used,memory.total,utilization.gpu",
                            "--format=csv,noheader"], capture_output=True, text=True, timeout=60)
        a = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
                            "--format=csv,noheader"], capture_output=True, text=True, timeout=60)
    except Exception:
        return None
    if g.returncode != 0 or a.returncode != 0:
        return None
    shared = False
    mine = os.getpid()
    for ln in a.stdout.strip().splitlines():
        pid = ln.split(",")[0].strip()
        if pid.isdigit() and int(pid) != mine:
            shared = True
    return {"gpus": g.stdout.strip(), "apps": a.stdout.strip(), "shared": shared}


def ensure_gpu(device):
    """RD-GPU unless the OPTIX branch of configure_render really took."""
    if device != "GPU":
        return
    prefs = bpy.context.preferences.addons["cycles"].preferences
    ok = any(d.type == "OPTIX" and d.use for d in prefs.devices)
    if not ok or bpy.context.scene.cycles.device != "GPU":
        raise S.SceneRefusal("RD-GPU", "no OPTIX device enabled or cycles.device != GPU after configure_render")


def _bar_material():
    """bar_emit: Emission (strength 1.0) fed by the FLOAT_COLOR attribute bar_rgb."""
    m = bpy.data.materials.get("bar_emit")
    if m is None:
        m = bpy.data.materials.new("bar_emit")
        nt = m.node_tree
        em = nt.nodes.new("ShaderNodeEmission")
        em.inputs["Strength"].default_value = 1.0
        attr = nt.nodes.new("ShaderNodeAttribute")
        attr.attribute_name = "bar_rgb"
        nt.links.new(attr.outputs["Color"], em.inputs["Color"])
        nt.links.new(em.outputs["Emission"], nt.nodes["Material Output"].inputs["Surface"])
    return m


def _text_material():
    """text_emit: white Emission, strength 1.0."""
    m = bpy.data.materials.get("text_emit")
    if m is None:
        m = bpy.data.materials.new("text_emit")
        nt = m.node_tree
        em = nt.nodes.new("ShaderNodeEmission")
        em.inputs["Color"].default_value = (1.0, 1.0, 1.0, 1.0)
        em.inputs["Strength"].default_value = 1.0
        nt.links.new(em.outputs["Emission"], nt.nodes["Material Output"].inputs["Surface"])
    return m


def _plate_material():
    """plate_emit: dark Emission, strength 1.0, straight into Material Output."""
    m = bpy.data.materials.get("plate_emit")
    if m is None:
        m = bpy.data.materials.new("plate_emit")
        nt = m.node_tree
        em = nt.nodes.new("ShaderNodeEmission")
        em.inputs["Color"].default_value = (0.02, 0.02, 0.025, 1.0)
        em.inputs["Strength"].default_value = 1.0
        nt.links.new(em.outputs["Emission"], nt.nodes["Material Output"].inputs["Surface"])
    return m


def _camera_only(obj):
    """Seen by the camera only."""
    obj.visible_diffuse = False
    obj.visible_glossy = False
    obj.visible_transmission = False
    obj.visible_volume_scatter = False
    obj.visible_shadow = False


def _add_label(cam_obj, text, x, y, size, align_x):
    cu = bpy.data.curves.new("lbl", type="FONT")
    cu.body = text
    cu.size = size
    cu.align_x = align_x
    cu.align_y = "CENTER"
    obj = bpy.data.objects.new("lbl_" + cam_obj.name + "_" + text, cu)
    bpy.context.scene.collection.objects.link(obj)
    obj.parent = cam_obj
    obj.matrix_parent_inverse = Matrix.Identity(4)
    obj.location = (x, y, -BAR_DEPTH)
    cu.materials.append(_text_material())
    _camera_only(obj)
    return obj


def build_bar(cam_obj, kind, lo, hi):
    """The colour bar + tick labels for one camera, parented to it, Emission-only."""
    sc = bpy.context.scene
    res_x, res_y = int(sc.render.resolution_x), int(sc.render.resolution_y)
    x0, x1, y0, y1 = overlay_rect(cam_obj.data, res_x, res_y, BAR, BAR_DEPTH)
    _, hh_frame = frame_hw_hh(cam_obj.data, res_x, res_y, BAR_DEPTH)
    size = TEXT_H * 2.0 * hh_frame
    name = ("bar_cp_" if kind == "cp" else "bar_speed_") + cam_obj.name
    ramp = S.CP_RAMP if kind == "cp" else S.SPEED_RAMP
    levels = bar_levels(lo, hi, ramp, BAR_LEVELS)

    def ux(u):
        return x0 + (u - BAR[0]) / (BAR[1] - BAR[0]) * (x1 - x0)

    def vy(v):
        return y0 + (v - BAR[2]) / (BAR[3] - BAR[2]) * (y1 - y0)

    verts = []
    for k in range(BAR_LEVELS + 1):
        yk = y0 + (k / float(BAR_LEVELS)) * (y1 - y0)
        verts.append((x0, yk, -BAR_DEPTH))
        verts.append((x1, yk, -BAR_DEPTH))
    quads = [(2 * k, 2 * k + 1, 2 * k + 3, 2 * k + 2) for k in range(BAR_LEVELS)]
    me = bpy.data.meshes.new(name)
    me.from_pydata(verts, [], quads)
    me.update()
    colors = np.repeat(levels, 2, axis=0).astype(np.float32)
    cattr = me.color_attributes.new("bar_rgb", "FLOAT_COLOR", "POINT")
    cattr.data.foreach_set("color", colors.ravel())
    me.update()
    me.materials.append(_bar_material())
    obj = bpy.data.objects.new(name, me)
    bpy.context.scene.collection.objects.link(obj)
    obj.parent = cam_obj
    obj.matrix_parent_inverse = Matrix.Identity(4)
    _camera_only(obj)
    pz = -(BAR_DEPTH + 0.001)
    pu0, pu1, pv0, pv1 = overlay_rect(cam_obj.data, res_x, res_y, PLATE, -pz)
    pme = bpy.data.meshes.new(name + "_plate")
    pme.from_pydata([(pu0, pv0, pz), (pu1, pv0, pz), (pu1, pv1, pz), (pu0, pv1, pz)],
                    [], [(0, 1, 2, 3)])
    pme.update()
    pme.materials.append(_plate_material())
    plate = bpy.data.objects.new(name + "_plate", pme)
    bpy.context.scene.collection.objects.link(plate)
    plate.parent = cam_obj
    plate.matrix_parent_inverse = Matrix.Identity(4)
    _camera_only(plate)
    labels = []
    if kind == "cp":
        ticks = [lo] + ([0.0] if lo < 0.0 < hi else []) + [hi]
        fmt, title = "%.2f", "Cp"
    else:
        ticks = [lo, 0.5 * (lo + hi), hi]
        fmt, title = "%.0f", "|U| m/s"
    for v in ticks:
        vf = BAR[2] + (v - lo) / (hi - lo) * (BAR[3] - BAR[2])
        labels.append(_add_label(cam_obj, fmt % v, ux(BAR[0] - 0.008), vy(vf), size, "RIGHT"))
    labels.append(_add_label(cam_obj, title, ux(0.5 * (BAR[0] + BAR[1])), vy(BAR[3] + 0.03), size, "CENTER"))
    BARREG[name] = {"obj": obj, "labels": labels, "plate": plate,
                    "camera": cam_obj.name, "kind": kind}
    return obj


def set_bar_state(active):
    """Show only the named bar (and its labels and plate); hide every other bar."""
    for name, entry in BARREG.items():
        hide = name != active
        entry["obj"].hide_render = hide
        entry["plate"].hide_render = hide
        for lobj in entry["labels"]:
            lobj.hide_render = hide


def build_bars(jobs, cp_lo, cp_hi, sp_lo, sp_hi):
    for cam_name, kind in jobs:
        cam_obj = bpy.data.objects.get(cam_name)
        if cam_obj is None:
            raise S.SceneRefusal("RD-INPUT", "camera not found for the bar: " + cam_name)
        lo, hi = (cp_lo, cp_hi) if kind == "cp" else (sp_lo, sp_hi)
        build_bar(cam_obj, kind, lo, hi)
    set_bar_state(None)


def far_half_faces(V, faces):
    """Ascending indices of the faces whose EVERY vertex has y >= 0."""
    y = np.asarray(V, dtype=np.float64)[:, 1]
    out = []
    for i, f in enumerate(faces):
        idx = f.vertices if hasattr(f, "vertices") else f
        if min(y[j] for j in idx) >= 0.0:
            out.append(i)
    return out


def build_car_cut(car, co):
    """car_cut: the placed car's vertices, only the far-half (y >= 0) faces, car_paint."""
    me_car = car.data
    nf = len(me_car.polygons)
    loop_start = np.empty(nf, dtype=np.int64)
    loop_total = np.empty(nf, dtype=np.int64)
    me_car.polygons.foreach_get("loop_start", loop_start)
    me_car.polygons.foreach_get("loop_total", loop_total)
    nl = len(me_car.loops)
    loop_verts = np.empty(nl, dtype=np.int64)
    me_car.loops.foreach_get("vertex_index", loop_verts)
    sel = np.asarray(far_half_faces(co, me_car.polygons), dtype=np.int64)
    counts = loop_total[sel]
    face_id = np.repeat(np.arange(nf, dtype=np.int64), loop_total)
    lv = loop_verts[np.isin(face_id, sel)]
    starts = np.concatenate(([0], np.cumsum(counts)[:-1]))
    F_sel = [lv[s:s + c].tolist() for s, c in zip(starts, counts)]
    me = bpy.data.meshes.new("car_cut")
    me.from_pydata(np.asarray(co, dtype=np.float64).tolist(), [], F_sel)
    me.update()
    me.materials.append(S.MATERIALS["car_paint"])
    obj = bpy.data.objects.new("car_cut", me)
    bpy.context.scene.collection.objects.link(obj)
    obj.hide_render = True
    say("car_cut: " + str(len(sel)) + " of " + str(nf) + " faces kept (the y >= 0 half)")
    return obj, int(len(sel))


def add_shot_cameras(center):
    """cam_stream and cam_slice, 50 mm, aimed with scene._aim_at."""
    table = [
        ("cam_stream", Vector(center) + Vector((1.2, 0.0, -0.1)), (-0.60, -0.62, 0.50), 11.0),
        ("cam_slice", Vector((float(center[0]) + 1.0, 0.0, 0.55)), (-0.30, -0.90, 0.32), 10.0),
    ]
    for name, target, direction, dist in table:
        cd = bpy.data.cameras.new(name)
        cd.lens = 50.0
        obj = bpy.data.objects.new(name, cd)
        bpy.context.scene.collection.objects.link(obj)
        obj.location = target + Vector(direction).normalized() * dist
        S._aim_at(obj, target)


def set_view_transform(vt):
    sc = bpy.context.scene
    vs = sc.view_settings
    vs.view_transform = vt
    vs.look = "None"
    vs.exposure = 0.0
    vs.gamma = 1.0
    sc.display_settings.display_device = "sRGB"


def build_scene(args):
    """The studio scene in scene.run's order, plus the two shot cameras."""
    sg_path = os.path.join(args.sim_geom, "sim_geom.json")
    with open(sg_path, "r", encoding="utf-8") as f:
        sg = json.load(f)
    if "placement" not in sg:
        raise S.SceneRefusal("RD-INPUT", "sim_geom.json missing key: placement")
    placement = sg["placement"]
    for k in ("scale", "translate_m"):
        if k not in placement:
            raise S.SceneRefusal("RD-INPUT", "sim_geom.json placement missing key: " + k)
    t0 = time.time()
    S.reset_scene()
    car = S.import_render_model(args.glb, placement)
    center, bbox_lo, bbox_hi, co = S.car_center(car)
    S.build_materials()
    S.add_ground(S.MATERIALS["studio_floor"])
    S.add_world()
    S.add_lights(center)
    S.add_cameras(center)
    add_shot_cameras(center)
    S.add_turntable(center, S.TT_DIR, S.TT_DIST, N_FRAMES)
    S.configure_render(args.samples, args.device)
    ensure_gpu(args.device)
    bpy.context.scene.render.use_persistent_data = True
    say("scene built in " + format(time.time() - t0, ".1f") + " s: car "
        + str(len(car.data.vertices)) + " vertices, " + str(len(car.data.polygons)) + " polygons, center ("
        + format(center[0], ".3f") + ", " + format(center[1], ".3f") + ", " + format(center[2], ".3f") + ")")
    return car, center, co


def prepare_shot(name, car):
    """View transform, bars, car/car_cut and field visibility for one still."""
    row = SHOTS[name]
    set_view_transform(row["vt"])
    active = None
    if row["bar"]:
        active = ("bar_cp_" if row["bar"] == "cp" else "bar_speed_") + row["camera"]
    set_bar_state(active)
    if row["car"] is None:
        car.hide_render = True
    else:
        car.hide_render = False
        S.set_car_material(car, row["car"])
    cut = bpy.data.objects.get("car_cut")
    if cut is not None:
        cut.hide_render = not row["car_cut"]
    for key in ("streamlines", "slice"):
        obj = bpy.data.objects.get(key)
        if obj is not None:
            obj.hide_render = not row[key]


def prepare_turntable(kind, car):
    """One state for a whole turntable run; returns the view transform."""
    vt = "Standard" if kind == "cp" else "AgX"
    set_view_transform(vt)
    set_bar_state("bar_cp_cam_turntable" if kind == "cp" else None)
    S.set_car_material(car, "field" if kind == "cp" else "paint")
    car.hide_render = False
    cut = bpy.data.objects.get("car_cut")
    if cut is not None:
        cut.hide_render = True
    for key in ("streamlines", "slice"):
        obj = bpy.data.objects.get(key)
        if obj is not None:
            obj.hide_render = True
    sc = bpy.context.scene
    sc.camera = bpy.data.objects["cam_turntable"]
    return vt


def do_render(path):
    sc = bpy.context.scene
    sc.render.filepath = path
    t0 = time.time()
    bpy.ops.render.render(write_still=True)
    return time.time() - t0


def parse_shots(s):
    names = [t.strip() for t in str(s).split(",") if t.strip()]
    if not names:
        raise S.SceneRefusal("RD-INPUT", "--shots is empty")
    for n in names:
        if n not in SHOTS:
            raise S.SceneRefusal("RD-INPUT", "unknown shot: " + n)
    return names


def args_dict(args):
    out = {}
    for k in sorted(vars(args)):
        v = getattr(args, k)
        if isinstance(v, (tuple, set)):
            v = list(v)
        out[k] = v
    return out


def validate_stills_inputs(args):
    require_arg(args.glb, "glb")
    require_arg(args.sim_geom, "sim-geom")
    require_arg(args.post, "post")
    require_arg(args.out, "out")
    validate_out(args.out)
    require_file(args.glb, "--glb")
    for fn in ("sim_geom.json", "LICENSE.txt", "sim_surface.stl"):
        require_file(os.path.join(args.sim_geom, fn), "--sim-geom/" + fn)
    for fn in ("cp.npz", "streamlines.npz", "slice.npz", "numbers.json"):
        require_file(os.path.join(args.post, fn), "--post/" + fn)


def validate_turntable_inputs(args):
    require_arg(args.kind, "kind")
    if args.kind not in ("cp", "clean"):
        raise S.SceneRefusal("RD-INPUT", "unknown --kind: " + str(args.kind))
    require_arg(args.glb, "glb")
    require_arg(args.sim_geom, "sim-geom")
    require_arg(args.out, "out")
    validate_out(args.out)
    require_file(args.glb, "--glb")
    for fn in ("sim_geom.json", "LICENSE.txt", "sim_surface.stl"):
        require_file(os.path.join(args.sim_geom, fn), "--sim-geom/" + fn)
    if args.kind == "cp":
        require_arg(args.post, "post")
        for fn in ("cp.npz", "streamlines.npz", "slice.npz", "numbers.json"):
            require_file(os.path.join(args.post, fn), "--post/" + fn)


def run_stills(args):
    t_total = time.time()
    validate_stills_inputs(args)
    os.makedirs(args.out, exist_ok=True)
    gpu_start = gpu_record()
    lic_path = os.path.join(args.sim_geom, "LICENSE.txt")
    with open(lic_path, "r", encoding="utf-8") as f:
        lic = parse_license(f.read())
    shutil.copyfile(lic_path, os.path.join(args.out, "LICENSE.txt"))
    with open(os.path.join(args.out, "CREDITS.txt"), "w", encoding="utf-8", newline="\n") as f:
        f.write(credits_text(lic, bpy.app.version_string))
    with open(os.path.join(args.post, "numbers.json"), "r", encoding="utf-8") as f:
        numbers = json.load(f)
    (cp_lo, cp_hi), (sp_lo, sp_hi) = resolve_ranges(numbers, args.cp_range, args.speed_range)
    shots = parse_shots(args.shots)
    car, center, co = build_scene(args)
    jobs = []
    for n in shots:
        if SHOTS[n]["bar"]:
            jobs.append((SHOTS[n]["camera"], SHOTS[n]["bar"]))
    build_bars(jobs, cp_lo, cp_hi, sp_lo, sp_hi)
    cut_obj, n_cut = build_car_cut(car, co)
    need_cp = any(SHOTS[n]["car"] == "field" for n in shots)
    cp_map = None
    if need_cp:
        t0 = time.time()
        pts, vals = S.load_surface_field(os.path.join(args.post, "cp.npz"))
        cp_map = S.apply_surface_field(car, pts, vals, cp_lo, cp_hi)
        say("field cp: " + str(cp_map["n"]) + " car vertices mapped in " + format(time.time() - t0, ".1f")
            + " s, max_dist " + format(cp_map["max_dist"], ".4f") + " m")
    if "streamlines" in shots:
        t0 = time.time()
        pts, spd, offs = S.load_streamlines(os.path.join(args.post, "streamlines.npz"))
        S.add_streamlines(pts, spd, offs, sp_lo, sp_hi, 0.012, 8)
        say("field streamlines: " + str(len(offs) - 1) + " lines in " + format(time.time() - t0, ".1f") + " s")
    if "slice" in shots:
        t0 = time.time()
        x, z, y0, spd = S.load_slice(os.path.join(args.post, "slice.npz"))
        slc = S.add_slice(x, z, y0, spd, sp_lo, sp_hi)
        slc.visible_shadow = False
        say("field slice: " + str(len(x)) + " x " + str(len(z)) + " in " + format(time.time() - t0, ".1f") + " s")
    sc = bpy.context.scene
    shot_reports = {}
    for name in shots:
        row = SHOTS[name]
        prepare_shot(name, car)
        sc.camera = bpy.data.objects[row["camera"]]
        path = os.path.join(args.out, name + ".png")
        secs = do_render(path)
        vt = str(sc.view_settings.view_transform)
        st = S.image_stats(path)
        bc = None
        if row["bar"] == "cp":
            bc = bar_check(path, cp_lo, cp_hi, S.CP_RAMP)
        elif row["bar"] == "speed":
            bc = bar_check(path, sp_lo, sp_hi, S.SPEED_RAMP)
        shot_reports[name] = {
            "file": name + ".png",
            "camera": row["camera"],
            "view_transform": vt,
            "bar": row["bar"],
            "seconds": float(secs),
            "width": st["width"],
            "height": st["height"],
            "lum_std": st["lum_std"],
            "bar_check": bc,
        }
        say("still " + name + " " + str(st["width"]) + "x" + str(st["height"]) + " in " + format(secs, ".1f")
            + " s, view " + vt + ", bar max_diff " + (format(bc["max_diff"], ".4f") if bc else "none"))
    gpu_end = gpu_record()
    report = {
        "tool": TOOL,
        "version": VERSION,
        "args": args_dict(args),
        "blender": bpy.app.version_string,
        "device_used": args.device,
        "samples": int(args.samples),
        "cp_range": [cp_lo, cp_hi],
        "speed_range": [sp_lo, sp_hi],
        "attribution": lic["attribution"],
        "car": {
            "n_vertices": int(len(car.data.vertices)),
            "n_polygons": int(len(car.data.polygons)),
            "center": [float(center[0]), float(center[1]), float(center[2])],
        },
        "cp_map": cp_map,
        "car_cut": {"n_faces": n_cut},
        "shots": shot_reports,
        "gpu_start": gpu_start,
        "gpu_end": gpu_end,
        "seconds_total": float(time.time() - t_total),
    }
    with open(os.path.join(args.out, "stills.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=1)
    say("stills.json written under " + args.out)
    return 0


def _load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def run_turntable(args):
    t_total = time.time()
    validate_turntable_inputs(args)
    kind = args.kind
    A, B = parse_frames(args.frames)
    os.makedirs(args.out, exist_ok=True)
    gpu_start = gpu_record()
    cp_lo = cp_hi = None
    if kind == "cp":
        numbers = _load_json(os.path.join(args.post, "numbers.json"))
        (cp_lo, cp_hi), _sp = resolve_ranges(numbers, args.cp_range, None)
    car, center, co = build_scene(args)
    if kind == "cp":
        t0 = time.time()
        pts, vals = S.load_surface_field(os.path.join(args.post, "cp.npz"))
        cp_map = S.apply_surface_field(car, pts, vals, cp_lo, cp_hi)
        say("field cp: " + str(cp_map["n"]) + " car vertices mapped in " + format(time.time() - t0, ".1f")
            + " s, max_dist " + format(cp_map["max_dist"], ".4f") + " m")
        build_bars([("cam_turntable", "cp")], cp_lo, cp_hi, 0.0, 1.0)
    else:
        build_bars([], 0.0, 1.0, 0.0, 1.0)
    vt = prepare_turntable(kind, car)
    d = os.path.join(args.out, "turntable_" + kind)
    os.makedirs(d, exist_ok=True)
    sc = bpy.context.scene
    sc.camera = bpy.data.objects["cam_turntable"]
    n = B - A + 1
    rendered = 0
    skipped = 0
    spent = 0.0
    for i, f in enumerate(range(A, B + 1)):
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
    gpu_end = gpu_record()
    report = {
        "tool": TOOL,
        "version": VERSION,
        "kind": kind,
        "args": args_dict(args),
        "blender": bpy.app.version_string,
        "device_used": args.device,
        "samples": int(args.samples),
        "view_transform": vt,
        "cp_range": None if cp_lo is None else [cp_lo, cp_hi],
        "frames": [A, B],
        "rendered": rendered,
        "skipped": skipped,
        "seconds_total": float(time.time() - t_total),
        "seconds_per_rendered_frame": (spent / rendered) if rendered else None,
        "gpu_start": gpu_start,
        "gpu_end": gpu_end,
    }
    with open(os.path.join(args.out, "turntable_" + kind + ".json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=1)
    say("turntable_" + kind + ".json written: " + str(rendered) + " rendered, " + str(skipped) + " skipped")
    return 0


def _check_turntable_dir(out, kind, vt_want, F):
    d = os.path.join(out, "turntable_" + kind)
    if not os.path.isdir(d):
        return False, "dir missing: " + d, []
    problems = []
    notes = []
    expected = set("frame_%04d.png" % i for i in range(1, F + 1))
    actual = set(os.listdir(d))
    extra = sorted(actual - expected)
    missing = sorted(expected - actual)
    if extra:
        problems.append("extra " + ", ".join(extra[:5]))
    if missing:
        problems.append("missing " + str(len(missing)) + " e.g. " + missing[0])
    sized = 0
    for i in range(1, F + 1):
        p = os.path.join(d, "frame_%04d.png" % i)
        if os.path.isfile(p):
            w, h = png_size(p)
            if (w, h) != (1920, 1080):
                problems.append("frame " + str(i) + " size " + str(w) + "x" + str(h))
            sized += 1
    lum_frames = sorted(set([1, 1 + F // 4, 1 + F // 2, 1 + 3 * F // 4]))
    lums = []
    for i in lum_frames:
        p = os.path.join(d, "frame_%04d.png" % i)
        if os.path.isfile(p):
            lum = S.image_stats(p)["lum_std"]
            lums.append("f" + str(i) + " " + format(lum, ".4f"))
            if not lum > LUM_MIN:
                problems.append("frame " + str(i) + " lum_std " + format(lum, ".4f") + " <= " + str(LUM_MIN))
    if F >= 2:
        p1 = os.path.join(d, "frame_%04d.png" % 1)
        p2 = os.path.join(d, "frame_%04d.png" % (1 + F // 2))
        if os.path.isfile(p1) and os.path.isfile(p2):
            a = load_px(p1)
            b = load_px(p2)
            if a.shape == b.shape:
                mad = float(np.mean(np.abs(a - b)))
                notes.append("mad(1," + str(1 + F // 2) + ") " + format(mad, ".4f"))
                if not mad > 0.01:
                    problems.append("frames 1/" + str(1 + F // 2) + " nearly identical (mad "
                                    + format(mad, ".4f") + ")")
            else:
                problems.append("frame size mismatch for the diff test")
    notes.append("lum " + ", ".join(lums))
    jpath = os.path.join(out, "turntable_" + kind + ".json")
    if os.path.isfile(jpath):
        j = _load_json(jpath)
        if j.get("view_transform") != vt_want:
            problems.append("json view " + str(j.get("view_transform")) + " != " + vt_want)
        notes.append(str(j.get("rendered")) + " rendered, " + str(j.get("skipped")) + " skipped")
    else:
        problems.append("turntable_" + kind + ".json missing")
    if problems:
        return False, "; ".join(problems), []
    return True, "exactly " + str(sized) + "/" + str(F) + " frames 1920x1080; " + "; ".join(notes), []


def run_check(args):
    require_arg(args.out, "out")
    validate_out(args.out)
    out = args.out
    if not os.path.isdir(out):
        raise S.SceneRefusal("RD-INPUT", "out dir not found: " + out)
    spath = os.path.join(out, "stills.json")
    if not os.path.isfile(spath):
        raise S.SceneRefusal("RD-INPUT", "stills.json not found under " + out)
    stills = _load_json(spath)
    F = int(args.frames_expected)
    checks = {}

    def record(name, ok, detail):
        checks[name] = {"pass": bool(ok), "detail": detail}

    problems = []
    for name in ("hero", "side", "top_rear", "streamlines", "slice"):
        p = os.path.join(out, name + ".png")
        if not os.path.isfile(p):
            problems.append(name + " png missing")
            continue
        w, h = png_size(p)
        if (w, h) != (1920, 1080):
            problems.append(name + " size " + str(w) + "x" + str(h))
            continue
        lum = S.image_stats(p)["lum_std"]
        if not lum > LUM_MIN:
            problems.append(name + " lum_std " + format(lum, ".4f") + " <= " + str(LUM_MIN))
    record("stills", not problems,
           "; ".join(problems) if problems else "5/5 stills 1920x1080, lum_std > " + str(LUM_MIN))

    shots = stills.get("shots", {})
    want = {"hero": "Standard", "side": "Standard", "top_rear": "Standard",
            "slice": "Standard", "streamlines": "AgX"}
    bad = []
    for k in ("hero", "side", "top_rear", "streamlines", "slice"):
        got = shots.get(k, {}).get("view_transform")
        if got != want[k]:
            bad.append(k + " view " + str(got) + " != " + want[k])
    record("view", not bad, "; ".join(bad) if bad else "hero/side/top_rear/slice Standard, streamlines AgX")

    bad = []
    det = []
    cp_range = stills.get("cp_range")
    sp_range = stills.get("speed_range")
    if cp_range is None:
        bad.append("cp_range missing from stills.json")
    else:
        for name in ("hero", "side", "top_rear"):
            p = os.path.join(out, name + ".png")
            if not os.path.isfile(p):
                bad.append(name + " png missing")
                continue
            d = bar_check(p, cp_range[0], cp_range[1], S.CP_RAMP)["max_diff"]
            det.append(name + " " + format(d, ".4f"))
            if d > BAR_TOL:
                bad.append(name + " max_diff " + format(d, ".4f") + " > " + str(BAR_TOL))
    if sp_range is None:
        bad.append("speed_range missing from stills.json")
    else:
        p = os.path.join(out, "slice.png")
        if os.path.isfile(p):
            d = bar_check(p, sp_range[0], sp_range[1], S.SPEED_RAMP)["max_diff"]
            det.append("slice " + format(d, ".4f"))
            if d > BAR_TOL:
                bad.append("slice max_diff " + format(d, ".4f") + " > " + str(BAR_TOL))
        else:
            bad.append("slice png missing")
    record("bars", not bad, "; ".join(bad) if bad else "recomputed max_diff: " + ", ".join(det))

    for kind, vt_want in (("cp", "Standard"), ("clean", "AgX")):
        ok, detail, _unused = _check_turntable_dir(out, kind, vt_want, F)
        record("turntable_" + kind, ok, detail)

    bad = []
    det = []
    jpath = os.path.join(out, "turntable_cp.json")
    if not os.path.isfile(jpath):
        bad.append("turntable_cp.json missing")
    else:
        j = _load_json(jpath)
        cr = j.get("cp_range")
        if cr is None:
            bad.append("turntable_cp.json has no cp_range")
        else:
            for i in sorted(set([1, 1 + F // 2])):
                p = os.path.join(out, "turntable_cp", "frame_%04d.png" % i)
                if not os.path.isfile(p):
                    bad.append("frame " + str(i) + " missing")
                    continue
                d = bar_check(p, cr[0], cr[1], S.CP_RAMP)["max_diff"]
                det.append("f" + str(i) + " " + format(d, ".4f"))
                if d > BAR_TOL:
                    bad.append("frame " + str(i) + " max_diff " + format(d, ".4f") + " > " + str(BAR_TOL))
    record("turntable_cp_bars", not bad,
           "; ".join(bad) if bad else "cp bars on frames 1/" + str(1 + F // 2) + " <= "
           + str(BAR_TOL) + " (" + ", ".join(det) + ")")

    bad = []
    det = []
    for nm, p in (("hero", os.path.join(out, "hero.png")),
                  ("side", os.path.join(out, "side.png")),
                  ("top_rear", os.path.join(out, "top_rear.png")),
                  ("slice", os.path.join(out, "slice.png")),
                  ("turntable_cp/frame_0001",
                   os.path.join(out, "turntable_cp", "frame_0001.png"))):
        if not os.path.isfile(p):
            bad.append(nm + " png missing")
            continue
        w, h = png_size(p)
        got = load_px(p)[int(0.165 * h), int(0.955 * w)]
        m = float(np.max(got))
        det.append(nm + " " + format(m, ".4f"))
        if not m <= 0.2:
            bad.append(nm + " max channel " + format(m, ".4f") + " > 0.2")
    record("plates", not bad,
           "; ".join(bad) if bad else "plate pixel max channel at (0.955, 0.165): " + ", ".join(det))

    try:
        lpath = os.path.join(out, "LICENSE.txt")
        cpath = os.path.join(out, "CREDITS.txt")
        if not os.path.isfile(lpath) or not os.path.isfile(cpath):
            raise S.SceneRefusal("RD-LICENSE", "LICENSE.txt or CREDITS.txt missing under " + out)
        with open(lpath, "r", encoding="utf-8") as f:
            lic = parse_license(f.read())
        with open(cpath, "r", encoding="utf-8", newline="") as f:
            got = f.read()
        want_txt = credits_text(lic, str(stills.get("blender", "")))
        ok = got == want_txt and lic["attribution"] in got
        record("credits", ok,
               "CREDITS.txt == credits_text(parse_license(LICENSE.txt)), attribution present" if ok
               else "CREDITS.txt differs from credits_text(parse_license(LICENSE.txt)) or misses the attribution line")
    except S.SceneRefusal as e:
        record("credits", False, e.code + ": " + e.message)

    reasons = [k + ": " + c["detail"] for k, c in checks.items() if not c["pass"]]
    ok = not reasons
    report = {
        "tool": TOOL,
        "version": VERSION,
        "out": out,
        "frames_expected": F,
        "checks": checks,
        "pass": ok,
        "reasons": reasons,
    }
    with open(os.path.join(out, "render.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=1)
    if ok:
        print("GATE PASS", flush=True)
        return 0
    print("GATE FAIL " + "; ".join(reasons), flush=True)
    return 1


def build_arg_parser():
    p = argparse.ArgumentParser(prog="tools/promo/render.py",
                                description="F1 promo renderer (see the module docstring)")
    p.add_argument("--selftest", action="store_true")
    sub = p.add_subparsers(dest="cmd")

    ps = sub.add_parser("stills")
    ps.add_argument("--glb")
    ps.add_argument("--sim-geom", dest="sim_geom")
    ps.add_argument("--post")
    ps.add_argument("--out")
    ps.add_argument("--device", choices=["GPU", "CPU"], default="GPU")
    ps.add_argument("--samples", type=int, default=128)
    ps.add_argument("--shots", default="hero,side,top_rear,streamlines,slice")
    ps.add_argument("--cp-range", dest="cp_range", type=float, nargs=2, default=None, metavar=("LO", "HI"))
    ps.add_argument("--speed-range", dest="speed_range", type=float, nargs=2, default=None, metavar=("LO", "HI"))

    pt = sub.add_parser("turntable")
    pt.add_argument("--kind", choices=["cp", "clean"], default=None)
    pt.add_argument("--glb")
    pt.add_argument("--sim-geom", dest="sim_geom")
    pt.add_argument("--out")
    pt.add_argument("--post")
    pt.add_argument("--device", choices=["GPU", "CPU"], default="GPU")
    pt.add_argument("--samples", type=int, default=64)
    pt.add_argument("--frames", default="1-240")
    pt.add_argument("--cp-range", dest="cp_range", type=float, nargs=2, default=None, metavar=("LO", "HI"))

    pc = sub.add_parser("check")
    pc.add_argument("--out")
    pc.add_argument("--frames-expected", dest="frames_expected", type=int, default=N_FRAMES)
    return p


def dispatch(argv):
    args = build_arg_parser().parse_args(argv)
    if args.selftest:
        return selftest()
    if args.cmd == "stills":
        return run_stills(args)
    if args.cmd == "turntable":
        return run_turntable(args)
    if args.cmd == "check":
        return run_check(args)
    raise S.SceneRefusal("RD-INPUT", "a subcommand is required: stills | turntable | check (or --selftest)")


# ------------------------------- selftest ---------------------------------
# T1-T9, CPU only, exact expected values; under 120 s in total.


def selftest():
    fails = []
    tmpdirs = []

    def run_test(name, fn):
        try:
            detail = fn()
        except S.SceneRefusal as e:
            print("[FAIL] " + name + " unexpected refusal " + e.code + ": " + e.message, flush=True)
            fails.append(name)
        except Exception as e:
            print("[FAIL] " + name + " " + repr(e), flush=True)
            fails.append(name)
        else:
            print("[ok] " + name + ((" " + detail) if detail else ""), flush=True)

    def expect_refusal(code, fn):
        try:
            fn()
        except S.SceneRefusal as e:
            if e.code != code:
                raise AssertionError("expected " + code + ", got " + e.code + ": " + e.message)
            return
        raise AssertionError("expected refusal " + code + ", none raised")

    def t1():
        cam = bpy.data.cameras.new("t1_persp")
        cam.lens = 50.0
        got = overlay_rect(cam, 1920, 1080, BAR, BAR_DEPTH)
        want = (0.153, 0.1602, -0.06075, 0.06075)
        assert max(abs(got[i] - want[i]) for i in range(4)) <= 1e-9, "persp " + str(tuple(got))
        cam2 = bpy.data.cameras.new("t1_ortho")
        cam2.type = "ORTHO"
        cam2.ortho_scale = 10.0
        got2 = overlay_rect(cam2, 1920, 1080, BAR, BAR_DEPTH)
        want2 = (4.25, 4.45, -1.6875, 1.6875)
        assert max(abs(got2[i] - want2[i]) for i in range(4)) <= 1e-9, "ortho " + str(tuple(got2))
        return "persp (0.153, 0.1602, -0.06075, 0.06075), ortho (4.25, 4.45, -1.6875, 1.6875)"

    def t2():
        lv = bar_levels(-1.0, 1.0, S.CP_RAMP, BAR_LEVELS)
        assert lv.shape == (BAR_LEVELS + 1, 4), "shape " + str(lv.shape)
        want = {0: (0.02, 0.10, 0.60), 8: (0.06, 0.275, 0.775), 16: (0.10, 0.45, 0.95),
                32: (0.90, 0.90, 0.90), 64: (0.60, 0.02, 0.02)}
        for k, w in want.items():
            assert max(abs(float(lv[k, c]) - w[c]) for c in range(3)) <= 1e-6, "row " + str(k)
        got = srgb(np.array([0.0, 0.002, 0.5, 0.9, 1.0]))
        w2 = [0.0, 0.02584, 0.7353569830524495, 0.9546871718858662, 1.0]
        assert max(abs(float(got[i]) - w2[i]) for i in range(5)) <= 1e-12, "srgb " + str(got)
        return "levels 0/8/16/32/64 exact, srgb exact"

    def t3():
        # Rendered at 768x432, not the brief's 384x216: measured on this machine
        # (2026-10-04) the default production OIDN (HIGH, RGB_ALBEDO_NORMAL,
        # ACCURATE prefilter) diffuses the thin 64-level gradient bar and at
        # 384x216 the bar ends sit ~2 px from the black background, giving
        # max_diff 0.0269 > BAR_TOL even though the un-denoised bar is exact to
        # 0.0048. At 768x432 the same production denoiser settings give 0.0089
        # (AgX 0.207). Every other pinned aspect is unchanged: CPU, 4 samples,
        # OIDN, the same build_bar the shots use.
        S.reset_scene()
        sc = bpy.context.scene
        S.configure_render(4, "CPU")
        sc.render.resolution_x = 768
        sc.render.resolution_y = 432
        w = bpy.data.worlds.new("t3_world")
        sc.world = w
        w.node_tree.nodes["Background"].inputs[0].default_value = (0.0, 0.0, 0.0, 1.0)
        cd = bpy.data.cameras.new("t3cam")
        cd.lens = 50.0
        cam = bpy.data.objects.new("t3cam", cd)
        sc.collection.objects.link(cam)
        cam.location = (0.0, 0.0, 5.0)
        sc.camera = cam
        build_bar(cam, "cp", -0.63, 0.72)
        size_want = TEXT_H * 2.0 * (0.18 * 432.0 / 768.0)  # hw = 0.5 * 18 / 50
        entry = BARREG["bar_cp_t3cam"]
        for lobj in entry["labels"]:
            assert abs(lobj.data.size - size_want) <= 1e-9, lobj.name + " size " + str(lobj.data.size)
        plate = entry["plate"]
        assert plate.name == "bar_cp_t3cam_plate" and plate.parent is cam, plate.name
        set_bar_state("bar_cp_t3cam")
        assert not plate.hide_render and not entry["obj"].hide_render, "plate not shown with its bar"
        set_bar_state(None)
        assert plate.hide_render and entry["obj"].hide_render, "plate not hidden with its bar"
        set_bar_state("bar_cp_t3cam")  # restore: the renders below need the bar shown
        d = tempfile.mkdtemp()
        tmpdirs.append(d)
        diffs = []
        for vt, fn in (("Standard", "t3_standard.png"), ("AgX", "t3_agx.png")):
            set_view_transform(vt)
            p = os.path.join(d, fn)
            sc.render.filepath = p
            bpy.ops.render.render(write_still=True)
            diffs.append(bar_check(p, -0.63, 0.72, S.CP_RAMP)["max_diff"])
        d_std, d_agx = diffs
        assert d_std <= BAR_TOL, "standard max_diff " + format(d_std, ".4f") + " > " + str(BAR_TOL)
        assert d_agx > 0.04, "agx max_diff " + format(d_agx, ".4f") + " <= 0.04"
        return "standard max_diff " + format(d_std, ".4f") + " <= " + str(BAR_TOL) \
            + ", agx max_diff " + format(d_agx, ".4f") + " > 0.04" \
            + ", label size " + format(size_want, ".6f") + ", plate " + plate.name

    def t4():
        V = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0),
             (0.0, 1.0, 0.0), (0.0, -1.0, 0.0), (1.0, -1.0, 0.0)]
        faces = [(0, 1, 2, 3), (4, 5, 1, 0), (3, 4, 2)]
        assert far_half_faces(V, faces) == [0], str(far_half_faces(V, faces))
        V2 = [(x, y + 2.0, z) for (x, y, z) in V]
        assert far_half_faces(V2, faces) == [0, 1, 2], str(far_half_faces(V2, faces))
        return "[0] then [0, 1, 2]"

    def t5():
        F1 = (
            '"F1 2026 concept (polygon model)" by Qvist_Designs' + "\n"
            + "Source: https://sketchfab.com/3d-models/ea3bde709b1e4dc9b0ec8557d106ed42" + "\n"
            + "Licence: CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/)" + "\n"
            + "Downloaded by the user, 2026-10-03, GLB." + "\n"
            + 'Attribution line for the video: "F1 2026 concept" by Qvist_Designs, CC BY 4.0, via Sketchfab.'
        )
        lic = parse_license(F1)
        assert lic["title_line"] == '"F1 2026 concept (polygon model)" by Qvist_Designs'
        assert lic["source"] == "https://sketchfab.com/3d-models/ea3bde709b1e4dc9b0ec8557d106ed42"
        assert lic["licence"] == "CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/)"
        assert lic["attribution"] == '"F1 2026 concept" by Qvist_Designs, CC BY 4.0, via Sketchfab.'
        txt = credits_text(lic, "5.1.2")
        lines = txt.split("\n")
        assert len(lines) == 9, "lines " + str(len(lines))
        assert lines[0] == "Credits for the meteor-cfd F1 promo renders"
        assert lines[2] == 'Car model: "F1 2026 concept" by Qvist_Designs, CC BY 4.0, via Sketchfab.'
        assert lines[-2] == "Rendered with Blender 5.1.2 (Cycles).", lines[-2]
        expect_refusal("RD-LICENSE", lambda: parse_license(F1.replace(
            'Attribution line for the video: "F1 2026 concept" by Qvist_Designs, CC BY 4.0, via Sketchfab.', "")))
        return "9 lines, title/source/licence/attribution exact, missing attribution refused"

    def t6():
        assert parse_frames("1-240") == (1, 240)
        assert parse_frames("5") == (5, 5)
        for bad in ("0-3", "3-2", "1-241", "a"):
            expect_refusal("RD-INPUT", lambda b=bad: parse_frames(b))
        return "1-240 -> (1, 240), 5 -> (5, 5), 4 bad inputs refused"

    def t7():
        d = tempfile.mkdtemp()
        tmpdirs.append(d)
        img = bpy.data.images.new("t7x3", 7, 3)
        p = os.path.join(d, "t7x3.png")
        img.filepath_raw = p
        img.file_format = "PNG"
        img.save()
        assert png_size(p) == (7, 3), str(png_size(p))
        p2 = os.path.join(d, "not.png")
        with open(p2, "w", encoding="utf-8") as f:
            f.write("definitely not a png")
        expect_refusal("RD-PNG", lambda: png_size(p2))
        return "Blender-written 7x3 png -> (7, 3), text file refused"

    def t8():
        numbers = {"cp": {"p01": -0.625401108936, "p99": 0.7183598295600055},
                   "streamlines": {"speed_p99": 77.99009017685366}}
        got = default_ranges(numbers)
        want = ((-0.63, 0.72), (0.0, 78.0))
        for (g, wgt) in zip(got, want):
            assert abs(g[0] - wgt[0]) <= 1e-12 and abs(g[1] - wgt[1]) <= 1e-12, str(got)
        expect_refusal("RD-RANGE", lambda: resolve_ranges(numbers, (0.5, 0.5), None))
        return "((cp), (speed)) = ((-0.63, 0.72), (0.0, 78.0)), cp-range 0.5 0.5 refused"

    def t9():
        d = tempfile.mkdtemp()
        tmpdirs.append(d)
        post_missing = os.path.join(d, "no-such-post")
        expect_refusal("RD-INPUT", lambda: dispatch(["stills", "--glb", os.path.join(d, "x.glb"),
                                                     "--sim-geom", d, "--post", post_missing, "--out", d]))
        simdir = os.path.join(d, "sim")
        os.makedirs(simdir)
        for fn in ("sim_geom.json", "LICENSE.txt", "sim_surface.stl"):
            with open(os.path.join(simdir, fn), "wb"):
                pass
        glb = os.path.join(d, "car.glb")
        with open(glb, "wb"):
            pass
        postdir = os.path.join(d, "post")
        os.makedirs(postdir)
        expect_refusal("RD-INPUT", lambda: dispatch(["stills", "--glb", glb, "--sim-geom", simdir,
                                                     "--post", postdir, "--out", d]))
        repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        expect_refusal("RD-OUT", lambda: dispatch(["stills", "--glb", glb, "--sim-geom", simdir,
                                                   "--post", postdir,
                                                   "--out", os.path.join(repo, "tools", "promo", "selftest-rdout")]))
        expect_refusal("RD-INPUT", lambda: dispatch(["check", "--out", d]))
        return "4 refusals: no post dir, post without cp.npz, --out in the repo, check without stills.json"

    for name, fn in (("T1 overlay_rect", t1), ("T2 bar_levels/srgb", t2), ("T3 bar discrimination", t3),
                     ("T4 far_half_faces", t4), ("T5 credits", t5), ("T6 parse_frames", t6),
                     ("T7 png_size", t7), ("T8 default_ranges", t8), ("T9 refusals", t9)):
        run_test(name, fn)
    for d in tmpdirs:
        shutil.rmtree(d, ignore_errors=True)
    if fails:
        print("SELFTEST FAIL " + str(len(fails)) + "/9: " + ", ".join(fails), flush=True)
        return 1
    print("SELFTEST PASS 9/9", flush=True)
    return 0


def main():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    try:
        return dispatch(argv)
    except S.SceneRefusal as e:
        print("refused: " + e.code + ": " + e.message, flush=True)
        return 2


if __name__ == "__main__":
    sys.exit(main())
