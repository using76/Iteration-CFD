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
# No GPL-licensed source was consulted.

"""Turn the BSD-3-Clause PX4 x500 assembly into one drone sim surface.

Headless Blender recipe (run with --factory-startup, Cycles on the CPU only):

  1. glTF import (25 meshes, PX4 FLU body frame, metres, Z-up); clear
     parents keeping transforms; apply transforms; the sim surface stays in
     the GLB frame (scale 1, translate 0) so it lines up with the render
     model.
  2. The four prop meshes are measured and removed: each blade set becomes
     one actuator disk in rotors.json - centre on the motor-bell axis,
     normal +z (the PX4 joint axis), radius and blade thickness measured
     from the prop mesh, hub radius from the bell head, spin from the
     ccw/cw name suffix. A rotor is not a wall, so the disks are not part
     of the sim surface and the blades are render-only.
  3. The 21 airframe meshes are joined (every face carrying an integer
     src attribute naming its part), welded at --weld and the remaining
     boundary loops filled (holes_fill, all sides); triangulate.
  4. Reference metrics of the welded airframe: the parts divergence volume
     (reported, not gated - it leaves out the sealed tube bores and motor
     interiors), the y-z raster frontal area, and the outer-skin volume of
     an OpenVDB voxel remesh at --ref-voxel, checked converged against
     --ref-voxel2.
  5. Sim surface: REMESH VOXEL at --voxel, floater removal, COLLAPSE
     decimation to --target-tris, dissolve_degenerate, an edge-collapse
     pass below 1e-6 m so no edge survives under the house STL readers'
     1e-6 x diagonal weld, triangulate.
  6. Four patches (body, arms, motors, skids) per sim triangle from the
     nearest welded airframe part and region; written as binary STLs in
     metres and gated on the written files (exact float32 merge AND house
     weld), with rotors.json and sim_geom.json beside them.

The voxel defaults to 0.6 mm: the render model's outer-skin volume
converges by 0.3-0.4 mm (0.14 % apart) and that converged value is the
gate reference, while at 0.7 mm and coarser the remesh bridges the ~0.6 mm
slits between parts and overshoots the reference by more than 3 %; at
0.6 mm the overshoot is +1.7 %, inside the 3 % volume gate.

The input GLB (PX4 x500 airframe, (c) 2022 Rudis Laboratories / PX4
Autopilot for Drones, BSD-3-Clause) and every mesh made from it stay
OUTSIDE the repository; the script copies the model's LICENSE.txt and
LICENSE beside the outputs and credits the copyright holders in
rotors.json and sim_geom.json. No endorsement by the copyright holders is
implied.
"""

import argparse
import importlib.util
import json
import math
import os
import re
import shutil
import struct
import sys
import tempfile
import time
import traceback
import types

import bmesh
import bpy
import numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree

PATCH_ORDER = ("body", "arms", "motors", "skids")
Z_SKID = -0.05
R_ARM = 0.085
# An arm triangle must ALSO lie near an arm diagonal (the x500 arms run to the
# four motors at (+-0.174, +-0.174), i.e. 45 deg modulo 90): plates like the
# forward payload rail (~15 deg off +x) and the landing-gear mounts are not
# arms - they fall to body.
ARM_HALF_ANGLE_DEG = 20.0
# The house STL weld is 1e-6 x the sim bbox diagonal (~6.4e-7 m here); collapse
# every edge below 1e-6 m - edges only, never a vertex weld - so the written
# surface survives both the exact float32 merge and the house weld.
SHORT_EDGE_M = 1e-6
CREDIT = "(c) 2022 Rudis Laboratories / PX4 Autopilot for Drones, BSD-3-Clause"


def load_promo():
    """R2: reuse the promo helpers by import; NEVER import sim_geom by name."""
    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.normpath(os.path.join(here, "..", "promo", "sim_geom.py"))
    spec = importlib.util.spec_from_file_location("promo_sim_geom", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


PROMO = load_promo()


def stage(text):
    sys.stdout.write("[sim_geom] " + text + "\n")
    sys.stdout.flush()


def parse_args(argv):
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    ap = argparse.ArgumentParser(prog="sim_geom.py", description=__doc__.splitlines()[0])
    ap.add_argument("--glb", required=True, help="input GLB (metres, outside the repository)")
    ap.add_argument("--out", required=True, help="output directory, created if absent")
    ap.add_argument("--voxel", type=float, default=0.0006, help="VOXEL remesh size, in m")
    ap.add_argument("--ref-voxel", dest="ref_voxel", type=float, default=0.0003,
                    help="outer-skin reference remesh size, in m")
    ap.add_argument("--ref-voxel2", dest="ref_voxel2", type=float, default=0.0004,
                    help="convergence-check reference remesh size, in m")
    ap.add_argument("--target-tris", dest="target_tris", type=int, default=800000,
                    help="COLLAPSE decimate target on triangles; 0 disables decimation")
    ap.add_argument("--weld", type=float, default=1e-6, help="remove_doubles distance, in m")
    ap.add_argument("--floater", type=float, default=1e-8, help="component volume floor, in m^3")
    ap.add_argument("--raster", type=float, default=0.0005, help="frontal-area pixel size, in m")
    ap.add_argument("--no-preview", dest="no_preview", action="store_true")
    ap.add_argument("--samples", type=int, default=8, help="Cycles preview samples")
    return ap.parse_args(argv)


def refuse(code, message):
    print(code + ": " + message, file=sys.stderr)
    sys.stderr.flush()
    sys.exit(2)


def import_glb(path):
    """Import the GLB; clear parents keeping transforms; apply transforms.

    Returns (airframe objects, prop objects). The meshes stay in the GLB
    frame: no scaling, no placement transform.
    """
    known = {o.name for o in bpy.data.objects}
    bpy.ops.import_scene.gltf(filepath=path)
    meshes = [o for o in bpy.data.objects if o.type == "MESH" and o.name not in known]
    if not meshes:
        refuse("SG-INPUT", "glTF import produced no mesh objects: " + path)
    bpy.ops.object.select_all(action="DESELECT")
    for o in meshes:
        o.select_set(True)
    bpy.context.view_layer.objects.active = meshes[0]
    bpy.ops.object.parent_clear(type="CLEAR_KEEP_TRANSFORM")
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    props = []
    airframe = []
    for o in meshes:
        if re.fullmatch("prop_[0-9]+_(ccw|cw)", o.name):
            props.append(o)
        else:
            airframe.append(o)
    return airframe, props


def _world_verts(obj):
    V = PROMO.mesh_vertices(obj.data)
    M = np.asarray(obj.matrix_world, dtype=np.float64)
    return V @ M[:3, :3].T + M[:3, 3]


def measure_rotors(props, bells):
    """R5: one actuator-disk dict per rotor, in index order 0..3."""
    found = {}
    for o in props:
        m = re.fullmatch("prop_([0-9]+)_(ccw|cw)", o.name)
        if m is None:
            refuse("SG-ROTORS", "prop object with unparseable name: " + o.name)
        i = int(m.group(1))
        if i in found:
            refuse("SG-ROTORS", "duplicate prop index " + str(i))
        found[i] = (o, m.group(2))
    if sorted(found) != [0, 1, 2, 3]:
        refuse("SG-ROTORS", "prop indices " + str(sorted(found)) + " != 0..3")
    if sorted(bells) != [0, 1, 2, 3]:
        refuse("SG-ROTORS", "bell-head indices " + str(sorted(bells)) + " != 0..3")
    rotors = []
    for i in range(4):
        prop, spin = found[i]
        bell = bells[i]
        bv = _world_verts(bell)
        ax = 0.5 * (float(bv[:, 0].min()) + float(bv[:, 0].max()))
        ay = 0.5 * (float(bv[:, 1].min()) + float(bv[:, 1].max()))
        pv = _world_verts(prop)
        d = np.hypot(pv[:, 0] - ax, pv[:, 1] - ay)
        radius = float(d.max())
        outer = pv[d >= 0.5 * radius]
        zc = 0.5 * (float(outer[:, 2].min()) + float(outer[:, 2].max()))
        thick = float(outer[:, 2].max() - outer[:, 2].min())
        hub = float(np.hypot(bv[:, 0] - ax, bv[:, 1] - ay).max())
        rotors.append({"index": i, "name": prop.name,
                       "centre_m": [ax, ay, zc], "normal": [0.0, 0.0, 1.0],
                       "radius_m": radius, "hub_radius_m": hub, "thickness_m": thick,
                       "spin": spin, "spin_sign": 1 if spin == "ccw" else -1})
    return rotors


def _set_src(obj, idx):
    """Integer face attribute src = the part's index in the sorted parts list."""
    me = obj.data
    attr = me.attributes.get("src")
    if attr is None:
        attr = me.attributes.new("src", "INT", "FACE")
    n = len(me.polygons)
    try:
        attr.data.foreach_set("value", np.full(n, idx, dtype=np.int32))
    except Exception:
        for item in attr.data:
            item.value = idx


def _join(objs, name):
    bpy.ops.object.select_all(action="DESELECT")
    for o in objs:
        o.select_set(True)
    bpy.context.view_layer.objects.active = objs[0]
    bpy.ops.object.join()
    obj = bpy.context.view_layer.objects.active
    obj.name = name
    obj.data.name = name
    return obj


def _remesh_volume(src_obj, voxel):
    """Enclosed volume of a VOXEL remesh of a copy of src_obj, as is.

    bmesh calc_volume(signed=True) on the remeshed mesh without
    triangulating or converting millions of faces to numpy (memory).
    """
    tmp = src_obj.copy()
    tmp.data = src_obj.data.copy()
    tmp.name = "ref_remesh"
    bpy.context.scene.collection.objects.link(tmp)
    bpy.ops.object.select_all(action="DESELECT")
    tmp.select_set(True)
    bpy.context.view_layer.objects.active = tmp
    mod = tmp.modifiers.new("remesh", "REMESH")
    mod.mode = "VOXEL"
    mod.voxel_size = voxel
    mod.adaptivity = 0.0
    PROMO._apply_modifier(tmp, mod.name)
    bm = bmesh.new()
    bm.from_mesh(tmp.data)
    vol = abs(bm.calc_volume(signed=True))
    bm.free()
    me_data = tmp.data
    bpy.data.objects.remove(tmp)
    bpy.data.meshes.remove(me_data)
    return float(vol)


def _collapse_short_edges(bm, limit, max_passes=8):
    """Collapse every edge shorter than limit; edges only, never a vertex weld."""
    collapsed = 0
    for _ in range(max_passes):
        short = [e for e in bm.edges if e.calc_length() < limit]
        if not short:
            break
        collapsed += len(short)
        bmesh.ops.collapse(bm, edges=short)
    return collapsed


def write_stl(path, V, T, name):
    """Binary STL, float32, metres; the header names the patch and never
    starts with solid (the house binary reader would take that for ASCII)."""
    T = np.asarray(T, dtype=np.int64)
    P = V[T]
    n = np.cross(P[:, 1] - P[:, 0], P[:, 2] - P[:, 0])
    ln = np.linalg.norm(n, axis=1)
    ln[ln == 0.0] = 1.0
    n = (n / ln[:, None]).astype(np.float32)
    rec = np.zeros(len(T), dtype=np.dtype([("n", "<f4", (3,)), ("v", "<f4", (3, 3)), ("attr", "<u2")]))
    rec["n"] = n
    rec["v"] = P.astype(np.float32)
    header = "drone sim_geom.py patch=" + name + " units=metres PX4 x500 BSD-3-Clause"
    with open(path, "wb") as f:
        f.write(header[:80].encode("ascii", "replace").ljust(80, b" "))
        f.write(struct.pack("<I", len(T)))
        f.write(rec.tobytes())


def render_preview(render_objs, sim_obj, patch_of_face, rotors, out_png, samples):
    """R13: left = the full render model grey, right = the sim surface with one
    flat colour per patch plus the four translucent rotor disks; same camera."""
    sc = bpy.context.scene
    sc.render.engine = "CYCLES"
    sc.cycles.device = "CPU"
    try:
        prefs = bpy.context.preferences.addons["cycles"].preferences
        prefs.compute_device_type = "NONE"
    except Exception:
        pass
    sc.cycles.samples = max(1, int(samples))
    sc.render.resolution_x = 960
    sc.render.resolution_y = 540
    sc.render.resolution_percentage = 100
    sc.render.image_settings.file_format = "PNG"
    try:
        sc.view_settings.view_transform = "Standard"
    except Exception:
        pass

    world = bpy.data.worlds[0] if len(bpy.data.worlds) else bpy.data.worlds.new("World")
    sc.world = world
    world.use_nodes = True
    bg = world.node_tree.nodes.get("Background")
    if bg is not None:
        bg.inputs[0].default_value = (1.0, 1.0, 1.0, 1.0)
        bg.inputs[1].default_value = 1.0
    sun_data = bpy.data.lights.new("dronegeom_sun", "SUN")
    sun_data.energy = 3.0
    sun = bpy.data.objects.new("dronegeom_sun", sun_data)
    sc.collection.objects.link(sun)
    sun.rotation_euler = (math.radians(55.0), 0.0, math.radians(35.0))

    bb = [render_objs[0].matrix_world @ Vector(c) for c in render_objs[0].bound_box]
    mn = Vector((min(p.x for p in bb), min(p.y for p in bb), min(p.z for p in bb)))
    mx = Vector((max(p.x for p in bb), max(p.y for p in bb), max(p.z for p in bb)))
    center = (mn + mx) * 0.5
    cam_data = bpy.data.cameras.new("dronegeom_cam")
    cam_data.lens = 50.0
    cam = bpy.data.objects.new("dronegeom_cam", cam_data)
    sc.collection.objects.link(cam)
    direction = Vector((0.62, 0.60, 0.50)).normalized()
    cam.location = center + direction * 1.4
    cam.rotation_euler = (center - cam.location).to_track_quat("-Z", "Y").to_euler()
    sc.camera = cam

    colors = {"body": (0.62, 0.63, 0.66, 1.0), "arms": (0.16, 0.32, 0.85, 1.0),
              "motors": (0.85, 0.16, 0.12, 1.0), "skids": (0.20, 0.62, 0.28, 1.0)}
    grey = bpy.data.materials.new("dronegeom_grey")
    grey.use_nodes = True
    gb = grey.node_tree.nodes.get("Principled BSDF")
    if gb is not None:
        gb.inputs["Base Color"].default_value = (0.55, 0.55, 0.56, 1.0)
        gb.inputs["Roughness"].default_value = 0.9
    disk_mat = bpy.data.materials.new("dronegeom_disk")
    disk_mat.use_nodes = True
    db = disk_mat.node_tree.nodes.get("Principled BSDF")
    if db is not None:
        db.inputs["Base Color"].default_value = (1.0, 0.45, 0.10, 1.0)
        db.inputs["Roughness"].default_value = 0.6
        db.inputs["Alpha"].default_value = 0.30
    disks = []
    for i, r in enumerate(rotors):
        bpy.ops.mesh.primitive_cylinder_add(vertices=64, radius=r["radius_m"],
                                            depth=r["thickness_m"], location=r["centre_m"])
        disk = bpy.context.active_object
        disk.name = "rotor_disk_" + str(i)
        disk.data.materials.append(disk_mat)
        disk.visible_shadow = False
        disk.hide_render = True
        disks.append(disk)

    mats = []
    for name in PATCH_ORDER:
        m = bpy.data.materials.new("dronegeom_" + name)
        m.use_nodes = True
        nb = m.node_tree.nodes.get("Principled BSDF")
        if nb is not None:
            nb.inputs["Base Color"].default_value = colors[name]
            nb.inputs["Roughness"].default_value = 0.9
        mats.append(m)

    for o in render_objs:
        o.data.materials.clear()
        o.data.materials.append(grey)
    me_s = sim_obj.data
    me_s.materials.clear()
    for m in mats:
        me_s.materials.append(m)
    try:
        me_s.polygons.foreach_set("material_index", np.asarray(patch_of_face, dtype=np.int32))
    except Exception:
        me_s.polygons.foreach_set("material_index", np.asarray(patch_of_face).tolist())

    tmpdir = tempfile.mkdtemp(prefix="dronegeom_prev_")
    arrays = []
    for tag, left in (("left", True), ("right", False)):
        for o in render_objs:
            o.hide_render = not left
        sim_obj.hide_render = left
        for d in disks:
            d.hide_render = left
        p = os.path.join(tmpdir, tag + ".png")
        sc.render.filepath = p
        bpy.ops.render.render(write_still=True)
        img = bpy.data.images.load(p)
        w, h = img.size
        buf = np.empty(w * h * 4, dtype=np.float32)
        img.pixels.foreach_get(buf)
        arrays.append(buf.reshape(h, w, 4))
        bpy.data.images.remove(img)
    comp = np.ascontiguousarray(np.concatenate(arrays, axis=1))
    out = bpy.data.images.new("dronegeom_preview", width=comp.shape[1], height=comp.shape[0], alpha=True)
    out.pixels.foreach_set(comp.ravel())
    out.filepath_raw = out_png
    out.file_format = "PNG"
    out.save()


def main(argv):
    t0 = time.perf_counter()
    args = parse_args(argv)
    os.makedirs(args.out, exist_ok=True)
    timing = {}

    if not os.path.isfile(args.glb):
        refuse("SG-INPUT", "GLB not found: " + args.glb)
    try:
        bpy.ops.object.select_all(action="SELECT")
        bpy.ops.object.delete()
    except Exception:
        pass

    stage("importing GLB")
    t = time.perf_counter()
    airframe, props = import_glb(args.glb)
    n_objects = len(airframe) + len(props)
    n_tris_raw = int(sum(len(o.data.polygons) for o in airframe))
    timing["import"] = time.perf_counter() - t

    stage("rotors")
    t = time.perf_counter()
    bells = {}
    for o in airframe:
        m = re.match("motor_bell_[0-9]+_BellHead", o.name)
        if m is not None:
            i = int(re.match("motor_bell_([0-9]+)_BellHead", o.name).group(1))
            if i in bells:
                refuse("SG-ROTORS", "two BellHead objects for rotor " + str(i))
            bells[i] = o
    rotors = measure_rotors(props, bells)
    rot_doc = {
        "tool": "tools/drone/sim_geom.py",
        "source": {"glb": args.glb, "credit": CREDIT},
        "frame": "PX4 FLU body frame of x500_assembled.glb: +x forward, +y left, +z up; metres; origin = GLB origin",
        "model": "actuator_disk",
        "spin_convention": "ccw/cw as seen from +normal (from above); spin_sign +1 = right-handed rotation about +normal",
        "rotors": rotors,
    }
    with open(os.path.join(args.out, "rotors.json"), "w", encoding="utf-8") as f:
        json.dump(rot_doc, f, indent=1)
    timing["rotors"] = time.perf_counter() - t

    stage("weld + fill")
    t = time.perf_counter()
    parts = sorted(o.name for o in airframe)
    order = {n: i for i, n in enumerate(parts)}
    missing = []
    if "frame_CarbonFiber" not in order:
        missing.append("frame_CarbonFiber")
    if "frame_Metal" not in order:
        missing.append("frame_Metal")
    if not any(n.startswith("frame_Landing") for n in parts):
        missing.append("frame_Landing*")
    if not any(n.startswith("motor_") for n in parts):
        missing.append("motor_*")
    if missing:
        refuse("SG-PARTS", "airframe is missing: " + ", ".join(missing))
    for o in airframe:
        _set_src(o, order[o.name])
    render_props = _join(props, "render_props")
    render_props.hide_render = True
    obj = _join(airframe, "airframe")
    me = obj.data
    bm = bmesh.new()
    bm.from_mesh(me)
    open_raw, nm_raw = PROMO.edge_counts(bm)
    bmesh.ops.remove_doubles(bm, verts=bm.verts[:], dist=args.weld)
    open_weld, nm_weld = PROMO.edge_counts(bm)
    boundary = [e for e in bm.edges if len(e.link_faces) == 1]
    bmesh.ops.holes_fill(bm, edges=boundary, sides=0)
    open_filled, _nm_filled = PROMO.edge_counts(bm)
    bmesh.ops.triangulate(bm, faces=bm.faces[:])
    bm.normal_update()
    # Hole-fill faces carry the default src (0 = frame_CarbonFiber); their
    # triangles then route through the region rules like CarbonFiber faces.
    src_layer = bm.faces.layers.int.get("src")
    if src_layer is None:
        refuse("SG-INPUT", "joined airframe lost its src face attribute")
    src_arr = np.fromiter((f[src_layer] for f in bm.faces), dtype=np.int64, count=len(bm.faces))
    # Write the welded, filled, triangulated surface back to the airframe mesh
    # (src face attribute kept): the reference remeshes and the preview must
    # see the closed hull, not the raw unwelded join.
    bm.to_mesh(me)
    me.update()
    V0, T_ref = PROMO.bm_to_arrays(bm)
    mn0 = V0.min(axis=0)
    mx0 = V0.max(axis=0)
    length = float(mx0[0] - mn0[0])
    width = float(mx0[1] - mn0[1])
    height = float(mx0[2] - mn0[2])
    if not (0.30 <= length <= 0.60):
        refuse("SG-UNITS", "airframe x extent " + format(length, ".4f") + " m outside 0.30..0.60 m")
    vol_parts = PROMO.signed_volume(V0, T_ref)
    labels0 = PROMO.components(V0, T_ref)
    ncomp_ref = int(labels0.max()) + 1
    del labels0
    timing["weld_fill"] = time.perf_counter() - t

    stage("reference metrics")
    t = time.perf_counter()
    px = args.raster
    pad_px = 8
    grid = (float(mn0[1] - pad_px * px), float(mn0[2] - pad_px * px),
            int(math.ceil((mx0[1] - mn0[1]) / px)) + 2 * pad_px,
            int(math.ceil((mx0[2] - mn0[2]) / px)) + 2 * pad_px)
    PROMO.set_raster(px, grid)
    fa_ref = PROMO.frontal_area(V0, T_ref, px, grid)
    vol_outer = _remesh_volume(obj, args.ref_voxel)
    vol_outer2 = _remesh_volume(obj, args.ref_voxel2)
    timing["reference"] = time.perf_counter() - t

    stage("remesh")
    t = time.perf_counter()
    bm_sim = bm.copy()
    sim_me = bpy.data.meshes.new("sim_surface")
    bm_sim.to_mesh(sim_me)
    bm_sim.free()
    sim_ob = bpy.data.objects.new("sim_surface", sim_me)
    bpy.context.scene.collection.objects.link(sim_ob)
    bpy.ops.object.select_all(action="DESELECT")
    sim_ob.select_set(True)
    bpy.context.view_layer.objects.active = sim_ob
    mod = sim_ob.modifiers.new("remesh", "REMESH")
    mod.mode = "VOXEL"
    mod.voxel_size = args.voxel
    mod.adaptivity = 0.0
    PROMO._apply_modifier(sim_ob, mod.name)
    bms = bmesh.new()
    bms.from_mesh(sim_me)
    bmesh.ops.triangulate(bms, faces=bms.faces[:])
    tris_remesh = len(bms.faces)
    Vs, Ts = PROMO.bm_to_arrays(bms)
    lab = PROMO.components(Vs, Ts)
    vtri = PROMO.tri_volumes(Vs, Ts)
    comp_vol = np.bincount(lab, weights=vtri)
    small = np.flatnonzero(comp_vol < args.floater)
    drop = np.isin(lab, small)
    floaters_dropped = int(len(small))
    floater_tris = int(drop.sum())
    if floaters_dropped:
        bms.faces.ensure_lookup_table()
        drop_faces = [bms.faces[i] for i in np.flatnonzero(drop).tolist()]
        bmesh.ops.delete(bms, geom=drop_faces, context="FACES")
    # Always write the triangulated bmesh back, so the decimation ratio below
    # is taken on TRIANGLES whether or not floaters were dropped.
    bms.to_mesh(sim_me)
    bms.free()
    del Vs, Ts, lab, vtri, comp_vol, drop
    timing["remesh"] = time.perf_counter() - t

    stage("decimate")
    t = time.perf_counter()
    n_cur = len(sim_me.polygons)
    if args.target_tris > 0 and n_cur > args.target_tris:
        mod2 = sim_ob.modifiers.new("decimate", "DECIMATE")
        mod2.decimate_type = "COLLAPSE"
        mod2.ratio = args.target_tris / n_cur
        mod2.use_collapse_triangulate = True
        PROMO._apply_modifier(sim_ob, mod2.name)
    bms = bmesh.new()
    bms.from_mesh(sim_me)
    bmesh.ops.dissolve_degenerate(bms, dist=1e-7, edges=bms.edges[:])
    short_collapsed = _collapse_short_edges(bms, SHORT_EDGE_M)
    bmesh.ops.triangulate(bms, faces=bms.faces[:])
    bms.normal_update()
    tris_decimated = len(bms.faces)
    V_s, T_s = PROMO.bm_to_arrays(bms)
    bms.to_mesh(sim_me)
    sim_me.update()
    bms.free()
    timing["decimate"] = time.perf_counter() - t

    stage("patches")
    t = time.perf_counter()
    tri_c = V_s[T_s].mean(axis=1)
    bvh = BVHTree.FromBMesh(bm)
    find_nearest = bvh.find_nearest
    centroids = tri_c.tolist()
    src_of = np.empty(len(T_s), dtype=np.int64)
    for i in range(len(T_s)):
        hit = find_nearest(centroids[i])
        src_of[i] = src_arr[hit[2]] if hit[2] is not None else 0
    is_motor = np.array([n.startswith("motor_") for n in parts])[src_of]
    is_landing = np.array([n.startswith("frame_Landing") for n in parts])[src_of]
    is_carbon = np.array([n == "frame_CarbonFiber" for n in parts])[src_of]
    is_metal = np.array([n == "frame_Metal" for n in parts])[src_of]
    zc = tri_c[:, 2]
    rc = np.hypot(tri_c[:, 0], tri_c[:, 1])
    idx = {n: i for i, n in enumerate(PATCH_ORDER)}
    # Priority motors > skids > arms > body: assign lowest priority first, the
    # later writes overwrite.
    patch = np.full(len(T_s), idx["body"], dtype=np.int64)
    phi = np.degrees(np.arctan2(tri_c[:, 1], tri_c[:, 0])) % 90.0
    near_diagonal = np.abs(phi - 45.0) <= ARM_HALF_ANGLE_DEG
    patch[(is_carbon | is_metal) & (rc >= R_ARM) & near_diagonal] = idx["arms"]
    patch[is_landing | (is_carbon & (zc < Z_SKID))] = idx["skids"]
    patch[is_motor] = idx["motors"]
    tri_a = 0.5 * np.linalg.norm(
        np.cross(V_s[T_s[:, 1]] - V_s[T_s[:, 0]], V_s[T_s[:, 2]] - V_s[T_s[:, 0]]), axis=1)
    counts = np.bincount(patch, minlength=4)
    areas = np.bincount(patch, weights=tri_a, minlength=4)
    timing["patches"] = time.perf_counter() - t

    stage("write + read-back check")
    t = time.perf_counter()
    for name in PATCH_ORDER:
        sel = patch == idx[name]
        write_stl(os.path.join(args.out, name + ".stl"), V_s, T_s[sel], name)
    write_stl(os.path.join(args.out, "sim_surface.stl"), V_s, T_s, "sim_surface")
    for lic in ("LICENSE.txt", "LICENSE"):
        src_lic = os.path.join(os.path.dirname(args.glb), lic)
        if os.path.isfile(src_lic):
            shutil.copyfile(src_lic, os.path.join(args.out, lic))
    PROMO.set_raster(px, grid)
    paths = [os.path.join(args.out, n + ".stl") for n in PATCH_ORDER]
    uni = PROMO.check_union(paths)
    lab_s = PROMO.components(uni["V"], uni["T"])
    ncomp_sim = int(lab_s.max()) + 1 if len(lab_s) else 0
    del lab_s, uni["V"], uni["T"]
    timing["write_check"] = time.perf_counter() - t

    if args.no_preview:
        timing["preview"] = 0.0
    else:
        stage("preview")
        tp = time.perf_counter()
        try:
            render_preview([obj, render_props], sim_ob, patch, rotors,
                           os.path.join(args.out, "preview.png"), args.samples)
        except Exception:
            traceback.print_exc()
        timing["preview"] = time.perf_counter() - tp

    timing["total"] = time.perf_counter() - t0

    reasons = []
    house_bad = []
    if uni["house_open_edges"]:
        house_bad.append("house_open=" + str(uni["house_open_edges"]))
    if uni["house_nonmanifold_edges"]:
        house_bad.append("house_nonmanifold=" + str(uni["house_nonmanifold_edges"]))
    if uni["min_edge_m"] < uni["house_weld_m"]:
        house_bad.append("min_edge=" + format(uni["min_edge_m"], ".4e")
                         + " < house_weld=" + format(uni["house_weld_m"], ".4e"))
    watertight = (uni["open_edges"] == 0 and uni["nonmanifold_edges"] == 0
                  and uni["degenerate_tris"] == 0 and not house_bad)
    if not watertight:
        reasons.append("watertight open=" + str(uni["open_edges"])
                       + " nonmanifold=" + str(uni["nonmanifold_edges"])
                       + " degenerate=" + str(uni["degenerate_tris"])
                       + ("; " + ", ".join(house_bad) if house_bad else ""))
    ref_conv = vol_outer / vol_outer2 - 1.0
    if abs(ref_conv) > 0.005:
        reasons.append("ref_conv=" + format(ref_conv, ".4f"))
    volume_rel = uni["volume_m3"] / vol_outer - 1.0
    if abs(volume_rel) > 0.03:
        reasons.append("volume_rel=" + format(volume_rel, ".4f"))
    frontal_rel = uni["frontal_area_m2"] / fa_ref - 1.0
    if abs(frontal_rel) > 0.03:
        reasons.append("frontal_rel=" + format(frontal_rel, ".4f"))
    tris_ok = 200000 <= uni["tris"] <= 1500000
    if not tris_ok:
        reasons.append("tris=" + str(uni["tris"]))
    patches_ok = bool(np.all(counts >= 1)) and int(counts.sum()) == len(T_s) == uni["tris"]
    if not patches_ok:
        reasons.append("patches counts=" + str(counts.tolist()) + " sum=" + str(int(counts.sum()))
                       + " sim=" + str(len(T_s)))
    radii = [r["radius_m"] for r in rotors]
    rotors_ok = (len(rotors) == 4
                 and [r["spin"] for r in rotors] == ["ccw", "ccw", "cw", "cw"]
                 and (max(radii) - min(radii)) <= 1e-4)
    if not rotors_ok:
        reasons.append("rotors spins=" + str([r["spin"] for r in rotors])
                       + " radii=" + str([format(x, ".6f") for x in radii]))
    png = os.path.join(args.out, "preview.png")
    preview_ok = True if args.no_preview else (os.path.isfile(png) and os.path.getsize(png) > 10240)
    if not preview_ok:
        reasons.append("preview missing or small")
    gate_pass = not reasons

    result = {
        "tool": "tools/drone/sim_geom.py",
        "source": {"glb": args.glb, "bytes": os.path.getsize(args.glb), "credit": CREDIT},
        "params": {"weld_m": args.weld, "voxel_m": args.voxel, "ref_voxel_m": args.ref_voxel,
                   "ref_voxel2_m": args.ref_voxel2, "target_tris": args.target_tris,
                   "floater_m3": args.floater, "raster_m": args.raster, "short_edge_m": SHORT_EDGE_M},
        "placement": {"scale": 1.0, "translate_m": [0.0, 0.0, 0.0],
                      "length_m": length, "width_m": width, "height_m": height},
        "render_model": {"objects": n_objects, "parts": parts, "props": len(props),
                         "tris": n_tris_raw, "open_edges_raw": open_raw,
                         "nonmanifold_raw": nm_raw, "open_edges_welded": open_weld,
                         "nonmanifold_welded": nm_weld, "open_edges_filled": open_filled,
                         "components": ncomp_ref, "volume_parts_m3": vol_parts,
                         "volume_outer_m3": vol_outer, "volume_outer2_m3": vol_outer2,
                         "frontal_area_m2": fa_ref},
        "sim_surface": {"tris_remesh": int(tris_remesh), "floaters_dropped": floaters_dropped,
                        "floater_tris_dropped": floater_tris, "tris_decimated": int(tris_decimated),
                        "short_edges_collapsed": int(short_collapsed), "tris": int(len(T_s)),
                        "components": ncomp_sim, "open_edges": uni["open_edges"],
                        "nonmanifold_edges": uni["nonmanifold_edges"],
                        "degenerate_tris": uni["degenerate_tris"],
                        "min_edge_m": uni["min_edge_m"], "house_weld_m": uni["house_weld_m"],
                        "house_open_edges": uni["house_open_edges"],
                        "house_nonmanifold_edges": uni["house_nonmanifold_edges"],
                        "volume_m3": uni["volume_m3"], "frontal_area_m2": uni["frontal_area_m2"]},
        "patch_rule": {"z_skid": Z_SKID, "r_arm": R_ARM,
                       "arm_half_angle_deg": ARM_HALF_ANGLE_DEG},
        "patches": [{"name": n, "file": n + ".stl", "tris": int(counts[idx[n]]),
                     "area_m2": float(areas[idx[n]])} for n in PATCH_ORDER],
        "rotors_file": "rotors.json",
        "gate": {"watertight": watertight, "ref_conv": ref_conv, "volume_rel": volume_rel,
                 "frontal_rel": frontal_rel, "tris_ok": tris_ok, "patches_ok": patches_ok,
                 "rotors_ok": rotors_ok, "preview_ok": preview_ok, "pass": gate_pass,
                 "reasons": reasons},
        "timing_s": {k: float(timing[k]) for k in ("import", "rotors", "weld_fill", "reference",
                                                   "remesh", "decimate", "patches", "write_check",
                                                   "preview", "total")},
    }
    with open(os.path.join(args.out, "sim_geom.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=1)
    print("GATE PASS" if gate_pass else "GATE FAIL " + "; ".join(reasons))
    return 0 if gate_pass else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
