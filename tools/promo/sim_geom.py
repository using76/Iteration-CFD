#!/usr/bin/env python3
# meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
# Source-available, not Open Source. See LICENSE at the repository root.
# No GPL-licensed source was consulted.

"""Turn the CC BY 4.0 "F1 2026 concept" render model into one sim surface.

Headless Blender recipe (run with --factory-startup, Cycles on the CPU only):

  1. glTF import; join the chunk meshes; clear parents keeping transforms;
     apply transforms; scale x0.001 from millimetres to metres.
  2. Weld vertices at --weld (default 1e-6 m = the measured 1e-3 mm probe)
     and fill the remaining boundary loops (holes_fill, all sides).
  3. Place: nose (min x) at x = 0, centred in y, lowest point at
     z = -contact, so the car points along +x with the tyres 3 mm below the
     ground plane z = 0.
  4. Reference metrics on the placed render model: the volume by the signed
     divergence theorem V = sum a.(b x c)/6 and the frontal area by a pixel
     centre raster of the y-z projection (the same grid is reused for the
     sim surface).
  5. Sim surface: REMESH modifier, mode VOXEL, at --voxel; drop every
     component with signed volume < --floater; DECIMATE COLLAPSE to
     --target-tris; dissolve_degenerate; bisect at z = 0 (clear_inner);
     edge collapses below SHORT_EDGE_M so no edge survives under the
     1e-6 x diagonal weld the house STL readers apply (R12b); cap the contact
     loops planar at z = 0; triangulate.
  6. Five patches (body, front_wing, rear_wing, wheels, floor) assigned per
     sim triangle by nearest render-model component and by region; written
     as binary STLs in metres, read back and gated on the exact float32
     merge AND the house weld, which may not produce an open or a
     non-manifold edge either (R12b).

The voxel resolution defaults to 3 mm: at 3 mm every front-wing and
rear-wing element stays a separate closed section, while at 6 mm the rear
flap merges with its main plane across the slot.

The input GLB ("F1 2026 concept" by Qvist_Designs, CC BY 4.0, via
Sketchfab) and every mesh made from it stay OUTSIDE the repository; the
script copies the model's LICENSE.txt beside the outputs and records the
"Attribution line for the video" text in sim_geom.json, as CC BY 4.0
requires.
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
import traceback

import bmesh
import bpy
import numpy as np
from mathutils import Matrix, Vector
from mathutils.bvhtree import BVHTree

PATCH_ORDER = ("body", "front_wing", "rear_wing", "wheels", "floor")
Z_REAR_LOW = 0.40
Z_FLOOR = 0.10
# R12b: run 1 measured sim edges down to 2.4e-7 m while the house STL weld
# (1e-6 x the 5.82 m diagonal) sits at 5.8e-6 m; collapse every edge below
# 6e-6 m — edges only, so neighbouring sheets are never welded together.
SHORT_EDGE_M = 6e-6

_RASTER = (0.002, (0.0, 0.0, 1, 1))
_IMPORT_INFO = {"objects": 0}


def parse_args(argv):
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    ap = argparse.ArgumentParser(prog="sim_geom.py", description=__doc__.splitlines()[0])
    ap.add_argument("--glb", required=True, help="input GLB (millimetres, outside the repository)")
    ap.add_argument("--out", required=True, help="output directory, created if absent")
    ap.add_argument("--voxel", type=float, default=0.003, help="VOXEL remesh size in m")
    ap.add_argument("--target-tris", dest="target_tris", type=int, default=1500000,
                    help="COLLAPSE decimate target; 0 disables decimation")
    ap.add_argument("--contact", type=float, default=0.003, help="tyre bottoms below z=0, in m")
    ap.add_argument("--weld", type=float, default=1e-6, help="remove_doubles distance, in m")
    ap.add_argument("--floater", type=float, default=1e-5, help="component volume floor, in m^3")
    ap.add_argument("--raster", type=float, default=0.002, help="frontal-area pixel size, in m")
    ap.add_argument("--no-preview", dest="no_preview", action="store_true")
    ap.add_argument("--samples", type=int, default=8, help="Cycles preview samples")
    return ap.parse_args(argv)


def refuse(code, message):
    sys.stderr.write(f"{code}: {message}\n")
    sys.stderr.flush()
    sys.exit(2)


def import_glb(path):
    """Import the GLB, join its mesh objects, apply transforms, scale to metres."""
    known = {o.name for o in bpy.data.objects}
    bpy.ops.import_scene.gltf(filepath=path)
    meshes = [o for o in bpy.data.objects if o.type == "MESH" and o.name not in known]
    if not meshes:
        refuse("SG-INPUT", f"glTF import produced no mesh objects: {path}")
    _IMPORT_INFO["objects"] = len(meshes)
    bpy.ops.object.select_all(action="DESELECT")
    for o in meshes:
        o.select_set(True)
    bpy.context.view_layer.objects.active = meshes[0]
    bpy.ops.object.parent_clear(type="CLEAR_KEEP_TRANSFORM")
    bpy.ops.object.join()
    obj = bpy.context.view_layer.objects.active
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    obj.data.transform(Matrix.Diagonal((0.001, 0.001, 0.001, 1.0)))
    return obj


def _poly_vertices(me, count):
    for dt in (np.int32, np.int64):
        try:
            arr = np.empty(count, dtype=dt)
            me.polygons.foreach_get("vertices", arr)
            return arr.astype(np.int64, copy=False)
        except Exception:
            continue
    flat = []
    for p in me.polygons:
        flat.extend(p.vertices)
    return np.array(flat, dtype=np.int64)


def mesh_vertices(me):
    """(n, 3) float64 vertex coordinates of a mesh (mesh-level foreach)."""
    V = np.empty(len(me.vertices) * 3, dtype=np.float64)
    me.vertices.foreach_get("co", V)
    return V.reshape(-1, 3)


def bm_to_arrays(bm):
    """(V, T) float64 / int64 arrays of a triangulated bmesh, face order kept."""
    bm.verts.ensure_lookup_table()
    me = bpy.data.meshes.new(".simgeom_arrays")
    bm.to_mesh(me)
    V = mesh_vertices(me)
    T = _poly_vertices(me, len(me.polygons) * 3).reshape(-1, 3)
    bpy.data.meshes.remove(me)
    return V, T


def edge_counts(bm):
    """(open, nonmanifold) edge counts; open = 1 face, nonmanifold = >2 faces."""
    open_e = nm = 0
    for e in bm.edges:
        k = len(e.link_faces)
        if k == 1:
            open_e += 1
        elif k > 2:
            nm += 1
    return open_e, nm


def edge_stats(V, T):
    """(shortest unique edge length, open, nonmanifold); self-loops are ignored."""
    if len(T) == 0:
        return 0.0, 0, 0
    E = np.concatenate([T[:, [0, 1]], T[:, [1, 2]], T[:, [2, 0]]])
    E.sort(axis=1)
    nv = int(T.max()) + 1
    key = E[:, 0].astype(np.int64) * nv + E[:, 1]
    uk, cnt = np.unique(key[E[:, 0] != E[:, 1]], return_counts=True)
    L = np.linalg.norm(V[uk // nv] - V[uk % nv], axis=1)
    return float(L.min()), int((cnt == 1).sum()), int((cnt > 2).sum())


def _cell_keys(V, weld):
    """int64 grid keys; diag/weld <= 1e6 < 2^21 keeps the three-key pack exact."""
    q = np.floor((V - V.min(axis=0)) / weld).astype(np.int64)
    P = np.int64(1) << np.int64(21)
    return (q[:, 0] * P + q[:, 1]) * P + q[:, 2], P


def weld_clusters(V, weld):
    """Union-find parent over a numpy grid hash: vertices closer than `weld`
    share a cluster (27-cell neighbourhood, no scipy)."""
    n = len(V)
    keys, P = _cell_keys(V, weld)
    uk, inv = np.unique(keys, return_inverse=True)
    inv = np.asarray(inv).ravel()
    order = np.argsort(inv, kind="stable")
    counts = np.bincount(inv, minlength=len(uk))
    start = np.concatenate(([0], np.cumsum(counts)))
    offs = np.array([(dx, dy, dz) for dx in (-1, 0, 1) for dy in (-1, 0, 1)
                     for dz in (-1, 0, 1)], dtype=np.int64)
    steps = (offs[:, 0] * P + offs[:, 1]) * P + offs[:, 2]
    found_a, found_b = [], []
    for step in steps.tolist():
        pos = np.searchsorted(uk, uk + step)
        pos_c = np.minimum(pos, len(uk) - 1)
        hit = uk[pos_c] == uk + step
        src = np.flatnonzero(hit)
        if not len(src):
            continue
        nb = pos_c[hit]
        cn = counts[src].astype(np.int64)
        nn = counts[nb].astype(np.int64)
        tot = cn * nn
        cum = np.cumsum(tot)
        if cum[-1] == 0:
            continue
        cum0 = np.concatenate(([0], cum))
        base = 0
        s = 0
        while s < len(src):
            e = int(np.searchsorted(cum, base + 2000000, side="right"))
            e = min(max(e, s + 1), len(src))
            lo = int(cum0[s])
            hi = int(cum0[e])
            k = np.repeat(np.arange(s, e), tot[s:e])
            ar = np.arange(lo, hi) - cum0[k]
            vi = order[start[src[k]] + ar // nn[k]]
            vj = order[start[nb[k]] + ar % nn[k]]
            d = V[vi] - V[vj]
            near = (np.einsum("ij,ij->i", d, d) < weld * weld) & (vi != vj)
            if np.any(near):
                found_a.append(vi[near])
                found_b.append(vj[near])
            base = int(cum0[e])
            s = e
    parent = np.arange(n, dtype=np.int64)
    if found_a:
        a = np.concatenate(found_a)
        b = np.concatenate(found_b)
        while True:
            ra = parent[a]
            rb = parent[b]
            lo = np.minimum(ra, rb)
            np.minimum.at(parent, ra, lo)
            np.minimum.at(parent, rb, lo)
            while True:
                pp = parent[parent]
                if np.array_equal(pp, parent):
                    break
                parent = pp
            if not np.any(parent[a] != parent[b]):
                break
    return parent


def signed_volume(V, T):
    """V = sum a.(b x c)/6 over the triangles (outward winding -> positive)."""
    a = V[T[:, 0]]
    b = V[T[:, 1]]
    c = V[T[:, 2]]
    return float(np.einsum("ij,ij->", a, np.cross(b, c)) / 6.0)


def tri_volumes(V, T):
    a = V[T[:, 0]]
    b = V[T[:, 1]]
    c = V[T[:, 2]]
    return np.einsum("ij,ij->i", a, np.cross(b, c)) / 6.0


def components(V, T):
    """Connected-component label per triangle, by shared (undirected) edges."""
    m = len(T)
    if m == 0:
        return np.zeros(0, dtype=np.int64)
    E = np.concatenate([T[:, [0, 1]], T[:, [1, 2]], T[:, [2, 0]]])
    E.sort(axis=1)
    tri = np.tile(np.arange(m, dtype=np.int64), 3)
    order = np.lexsort((E[:, 1], E[:, 0]))
    Eo = E[order]
    to = tri[order]
    dup = np.all(Eo[1:] == Eo[:-1], axis=1)
    a = to[:-1][dup]
    b = to[1:][dup]
    del E, tri, order, Eo, to, dup
    parent = np.arange(m, dtype=np.int64)
    while True:
        ra = parent[a]
        rb = parent[b]
        lo = np.minimum(ra, rb)
        np.minimum.at(parent, ra, lo)
        np.minimum.at(parent, rb, lo)
        while True:
            pp = parent[parent]
            if np.array_equal(pp, parent):
                break
            parent = pp
        if not np.any(parent[a] != parent[b]):
            break
    roots = np.unique(parent)
    inv = np.zeros(m, dtype=np.int64)
    inv[roots] = np.arange(len(roots))
    return inv[parent]


def set_raster(px, grid):
    global _RASTER
    _RASTER = (px, grid)


def frontal_area(V, T, px, grid):
    """Projected y-z area: pixel centres inside any triangle's projection."""
    y0, z0, ny, nz = grid
    cov = np.zeros(ny * nz, dtype=bool)
    if len(T) == 0:
        return 0.0
    chunk = 32768
    for s in range(0, len(T), chunk):
        P = V[T[s:s + chunk]]
        y = P[:, :, 1]
        z = P[:, :, 2]
        iy0 = np.ceil((y.min(axis=1) - y0) / px - 0.5)
        iy1 = np.floor((y.max(axis=1) - y0) / px - 0.5)
        iz0 = np.ceil((z.min(axis=1) - z0) / px - 0.5)
        iz1 = np.floor((z.max(axis=1) - z0) / px - 0.5)
        iy0 = np.clip(iy0, 0, ny).astype(np.int64)
        iy1 = np.clip(iy1, -1, ny - 1).astype(np.int64)
        iz0 = np.clip(iz0, 0, nz).astype(np.int64)
        iz1 = np.clip(iz1, -1, nz - 1).astype(np.int64)
        h = np.maximum(iy1 - iy0 + 1, 0)
        w = np.maximum(iz1 - iz0 + 1, 0)
        cnt = h * w
        total = int(cnt.sum())
        if total == 0:
            continue
        tri_ix = np.repeat(np.arange(len(cnt), dtype=np.int64), cnt)
        starts = np.cumsum(cnt) - cnt
        offs = np.arange(total, dtype=np.int64) - np.repeat(starts, cnt)
        wi = w[tri_ix]
        iy = iy0[tri_ix] + offs // wi
        iz = iz0[tri_ix] + offs % wi
        yj = y0 + (iy + 0.5) * px
        zj = z0 + (iz + 0.5) * px
        y1 = y[tri_ix, 0]
        y2 = y[tri_ix, 1]
        y3 = y[tri_ix, 2]
        z1 = z[tri_ix, 0]
        z2 = z[tri_ix, 1]
        z3 = z[tri_ix, 2]
        den = (z2 - z3) * (y1 - y3) + (y3 - y2) * (z1 - z3)
        good = np.abs(den) > 0.0
        u1 = np.zeros(total, dtype=np.float64)
        u2 = np.zeros(total, dtype=np.float64)
        g = np.flatnonzero(good)
        d = den[g]
        u1[g] = ((z2[g] - z3[g]) * (yj[g] - y3[g]) + (y3[g] - y2[g]) * (zj[g] - z3[g])) / d
        u2[g] = ((z3[g] - z1[g]) * (yj[g] - y3[g]) + (y1[g] - y3[g]) * (zj[g] - z3[g])) / d
        u3 = 1.0 - u1 - u2
        inside = good & (u1 >= 0.0) & (u2 >= 0.0) & (u3 >= 0.0)
        cov[iy[inside] * nz + iz[inside]] = True
    return float(cov.sum()) * px * px


def write_stl(path, V, T, name):
    """Binary STL, float32, header text names the patch."""
    T = np.asarray(T, dtype=np.int64)
    P = V[T]
    n = np.cross(P[:, 1] - P[:, 0], P[:, 2] - P[:, 0])
    ln = np.linalg.norm(n, axis=1)
    ln[ln == 0.0] = 1.0
    n = (n / ln[:, None]).astype(np.float32)
    rec = np.zeros(len(T), dtype=np.dtype([("n", "<f4", (3,)), ("v", "<f4", (3, 3)), ("attr", "<u2")]))
    rec["n"] = n
    rec["v"] = P.astype(np.float32)
    header = f"sim_geom.py patch={name} units=metres F1 2026 concept by Qvist_Designs CC-BY-4.0"
    with open(path, "wb") as f:
        f.write(header[:80].encode("ascii", "replace").ljust(80, b" "))
        f.write(struct.pack("<I", len(T)))
        f.write(rec.tobytes())


def read_stl(path):
    """Binary STL -> (m, 3, 3) float32 triangles."""
    with open(path, "rb") as f:
        f.read(80)
        (n,) = struct.unpack("<I", f.read(4))
        raw = np.frombuffer(f.read(n * 50), dtype=np.uint8)
    return raw.reshape(n, 50)[:, 12:48].copy().view("<f4").reshape(n, 3, 3)


def check_union(paths):
    """R12 + R12b: read the patch STLs back; measure the exact float32 merge and
    the house weld (1e-6 x diagonal) that readers outside the mesher apply."""
    tris = np.concatenate([read_stl(p) for p in paths]) if paths else np.zeros((0, 3, 3), np.float32)
    flat = np.ascontiguousarray(tris.reshape(-1, 3))
    view = flat.view([("x", "<f4"), ("y", "<f4"), ("z", "<f4")]).ravel()
    uniq, inv = np.unique(view, return_inverse=True)
    V = np.stack([uniq["x"], uniq["y"], uniq["z"]], axis=1).astype(np.float64)
    T = np.asarray(inv).ravel().astype(np.int64).reshape(-1, 3)
    min_edge, open_e, nm_e = edge_stats(V, T)
    a = V[T[:, 0]]
    b = V[T[:, 1]]
    c = V[T[:, 2]]
    area = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
    eq = (T[:, 0] == T[:, 1]) | (T[:, 1] == T[:, 2]) | (T[:, 2] == T[:, 0])
    degenerate = int((eq | (area < 1e-12)).sum())
    house_weld = 1e-6 * float(np.linalg.norm(V.max(axis=0) - V.min(axis=0)))
    Th = weld_clusters(V, house_weld)[T]
    _h_min, house_open, house_nm = edge_stats(V, Th)
    px, grid = _RASTER
    return {
        "tris": int(len(T)),
        "open_edges": open_e,
        "nonmanifold_edges": nm_e,
        "degenerate_tris": degenerate,
        "min_edge_m": min_edge,
        "house_weld_m": house_weld,
        "house_open_edges": house_open,
        "house_nonmanifold_edges": house_nm,
        "volume_m3": signed_volume(V, T),
        "frontal_area_m2": frontal_area(V, T, px, grid),
        "V": V,
        "T": T,
    }


def parse_attribution(text):
    for line in text.splitlines():
        if "attribution line for the video" in line.lower():
            return line.partition(":")[2].strip()
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    return lines[0] if lines else ""


def _apply_modifier(ob, name):
    with bpy.context.temp_override(object=ob, active_object=ob,
                                   selected_objects=[ob], selected_editable_objects=[ob]):
        bpy.ops.object.modifier_apply(modifier=name)


def _group_loops(bm, edges):
    """Boundary edges -> connected loops (shared vertices)."""
    vmap = {}
    for e in edges:
        for v in e.verts:
            vmap.setdefault(v, []).append(e)
    seen = set()
    loops = []
    for e0 in edges:
        if e0 in seen:
            continue
        seen.add(e0)
        stack = [e0]
        comp = []
        while stack:
            e = stack.pop()
            comp.append(e)
            for v in e.verts:
                for e2 in vmap[v]:
                    if e2 not in seen:
                        seen.add(e2)
                        stack.append(e2)
        loops.append(comp)
    return loops


def render_preview(render_obj, sim_obj, patch_of_face, out_png, samples):
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
    sun_data = bpy.data.lights.new("simgeom_sun", "SUN")
    sun_data.energy = 3.0
    sun = bpy.data.objects.new("simgeom_sun", sun_data)
    sc.collection.objects.link(sun)
    sun.rotation_euler = (math.radians(55.0), 0.0, math.radians(35.0))

    bb = [render_obj.matrix_world @ Vector(c) for c in render_obj.bound_box]
    mn = Vector((min(p.x for p in bb), min(p.y for p in bb), min(p.z for p in bb)))
    mx = Vector((max(p.x for p in bb), max(p.y for p in bb), max(p.z for p in bb)))
    center = (mn + mx) * 0.5
    cam_data = bpy.data.cameras.new("simgeom_cam")
    cam_data.lens = 50.0
    cam = bpy.data.objects.new("simgeom_cam", cam_data)
    sc.collection.objects.link(cam)
    direction = Vector((-0.62, -0.60, 0.50)).normalized()
    # 9.0 m filled only ~55 % of the panel width (run 1); 6.8 m reached ~78 %
    # but clipped the sim rear wing at the frame edge; 7.4 m keeps the car
    # above ~70 % without clipping, same 3/4 front-left view.
    cam.location = center + direction * 7.4
    cam.rotation_euler = (center - cam.location).to_track_quat("-Z", "Y").to_euler()
    sc.camera = cam

    colors = {
        "body": (0.62, 0.63, 0.66, 1.0),
        "front_wing": (0.85, 0.16, 0.12, 1.0),
        "rear_wing": (0.16, 0.32, 0.85, 1.0),
        "wheels": (0.07, 0.07, 0.08, 1.0),
        "floor": (0.20, 0.62, 0.28, 1.0),
    }
    grey = bpy.data.materials.new("simgeom_grey")
    grey.use_nodes = True
    gb = grey.node_tree.nodes.get("Principled BSDF")
    if gb is not None:
        gb.inputs["Base Color"].default_value = (0.55, 0.55, 0.56, 1.0)
        gb.inputs["Roughness"].default_value = 0.9
    mats = []
    for name in PATCH_ORDER:
        m = bpy.data.materials.new("simgeom_" + name)
        m.use_nodes = True
        nb = m.node_tree.nodes.get("Principled BSDF")
        if nb is not None:
            nb.inputs["Base Color"].default_value = colors[name]
            nb.inputs["Roughness"].default_value = 0.9
        mats.append(m)

    me_r = render_obj.data
    me_r.materials.clear()
    me_r.materials.append(grey)
    me_s = sim_obj.data
    me_s.materials.clear()
    for m in mats:
        me_s.materials.append(m)
    try:
        me_s.polygons.foreach_set("material_index", np.asarray(patch_of_face, dtype=np.int32))
    except Exception:
        me_s.polygons.foreach_set("material_index", np.asarray(patch_of_face).tolist())

    tmpdir = tempfile.mkdtemp(prefix="simgeom_prev_")
    arrays = []
    # R14: left panel = the placed render model in grey, right = the sim surface.
    for tag, hide_sim in (("left", True), ("right", False)):
        render_obj.hide_render = not hide_sim
        sim_obj.hide_render = hide_sim
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
    out = bpy.data.images.new("simgeom_preview", width=comp.shape[1], height=comp.shape[0], alpha=True)
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
        refuse("SG-INPUT", f"GLB not found: {args.glb}")
    try:
        bpy.ops.object.select_all(action="SELECT")
        bpy.ops.object.delete()
    except Exception:
        pass

    sys.stdout.write("[sim_geom] importing GLB\n")
    sys.stdout.flush()
    t = time.perf_counter()
    obj = import_glb(args.glb)
    me = obj.data
    n_objects = _IMPORT_INFO["objects"]
    n_tris_raw = len(me.polygons)
    timing["import"] = time.perf_counter() - t

    sys.stdout.write("[sim_geom] weld + fill\n")
    sys.stdout.flush()
    t = time.perf_counter()
    bm = bmesh.new()
    bm.from_mesh(me)
    open_raw, nm_raw = edge_counts(bm)
    bmesh.ops.remove_doubles(bm, verts=bm.verts[:], dist=args.weld)
    open_weld, nm_weld = edge_counts(bm)
    boundary = [e for e in bm.edges if len(e.link_faces) == 1]
    bmesh.ops.holes_fill(bm, edges=boundary, sides=0)
    open_filled, _nm_filled = edge_counts(bm)
    bmesh.ops.triangulate(bm, faces=bm.faces[:])
    bm.normal_update()
    bm.to_mesh(me)
    me.update()
    V0 = mesh_vertices(me)
    T_ref = _poly_vertices(me, len(me.polygons) * 3).reshape(-1, 3)
    mn0 = V0.min(axis=0)
    mx0 = V0.max(axis=0)
    length = float(mx0[0] - mn0[0])
    width = float(mx0[1] - mn0[1])
    height = float(mx0[2] - mn0[2])
    if not (4.5 <= length <= 6.0):
        refuse("SG-UNITS", f"placed length {length:.3f} m outside 4.5..6.0 m")
    tr = Vector((-float(mn0[0]), -0.5 * float(mn0[1] + mx0[1]), -float(mn0[2]) - args.contact))
    bmesh.ops.translate(bm, vec=tr, verts=bm.verts[:])
    timing["weld_fill"] = time.perf_counter() - t

    sys.stdout.write("[sim_geom] reference metrics\n")
    sys.stdout.flush()
    t = time.perf_counter()
    V_ref = V0 + np.array((tr.x, tr.y, tr.z), dtype=np.float64)
    vol_ref = signed_volume(V_ref, T_ref)
    px = args.raster
    mnv = V_ref.min(axis=0)
    mxv = V_ref.max(axis=0)
    pad_px = 8
    ny = int(np.ceil((mxv[1] - mnv[1]) / px)) + 2 * pad_px
    nz = int(np.ceil((mxv[2] - mnv[2]) / px)) + 2 * pad_px
    grid = (float(mnv[1] - pad_px * px), float(mnv[2] - pad_px * px), ny, nz)
    set_raster(px, grid)
    fa_ref = frontal_area(V_ref, T_ref, px, grid)
    labels = components(V_ref, T_ref)
    ncomp_ref = int(labels.max()) + 1
    tz = np.minimum(np.minimum(V_ref[T_ref[:, 0], 2], V_ref[T_ref[:, 1], 2]), V_ref[T_ref[:, 2], 2])
    comp_minz = np.full(ncomp_ref, np.inf)
    np.minimum.at(comp_minz, labels, tz)
    model_minz = float(tz.min())
    low = np.flatnonzero(comp_minz <= model_minz + 0.001)
    if len(low) != 4:
        refuse("SG-WHEELS", f"expected 4 wheel components at the model's lowest point, found {len(low)}")
    tc = V_ref[T_ref].mean(axis=1)
    comp_n = np.bincount(labels, minlength=ncomp_ref)
    comp_cx = np.bincount(labels, weights=tc[:, 0]) / np.maximum(comp_n, 1)
    order2 = np.argsort(comp_cx[low])
    front2 = low[order2[:2]]
    rear2 = low[order2[2:]]
    fmask = np.isin(labels, front2)
    rmask = np.isin(labels, rear2)
    ft = T_ref[fmask].ravel()
    rt = T_ref[rmask].ravel()
    x_fw = float(V_ref[ft, 0].min())
    x_rc = float(comp_cx[rear2].mean())
    z_rt = float(V_ref[rt, 2].max())
    x_rr = float(V_ref[rt, 0].max())
    wheel_set = set(np.flatnonzero(np.isin(labels, low)).tolist())
    timing["reference"] = time.perf_counter() - t

    sys.stdout.write("[sim_geom] remesh\n")
    sys.stdout.flush()
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
    _apply_modifier(sim_ob, mod.name)
    bms = bmesh.new()
    bms.from_mesh(sim_me)
    bmesh.ops.triangulate(bms, faces=bms.faces[:])
    tris_remesh = len(bms.faces)
    Vs, Ts = bm_to_arrays(bms)
    lab = components(Vs, Ts)
    vtri = tri_volumes(Vs, Ts)
    comp_vol = np.bincount(lab, weights=vtri)
    small = np.flatnonzero(comp_vol < args.floater)
    drop = np.isin(lab, small)
    floaters_dropped = int(len(small))
    floater_tris = int(drop.sum())
    if floaters_dropped:
        bms.faces.ensure_lookup_table()
        drop_faces = [bms.faces[i] for i in np.flatnonzero(drop).tolist()]
        bmesh.ops.delete(bms, geom=drop_faces, context="FACES")
        bms.to_mesh(sim_me)
    bms.free()
    del Vs, Ts, lab, vtri, comp_vol, drop
    timing["remesh"] = time.perf_counter() - t

    sys.stdout.write("[sim_geom] decimate\n")
    sys.stdout.flush()
    t = time.perf_counter()
    if args.target_tris > 0:
        n_cur = len(sim_me.polygons)
        if n_cur > args.target_tris:
            mod2 = sim_ob.modifiers.new("decimate", "DECIMATE")
            mod2.decimate_type = "COLLAPSE"
            mod2.ratio = args.target_tris / n_cur
            mod2.use_collapse_triangulate = True
            _apply_modifier(sim_ob, mod2.name)
    bms = bmesh.new()
    bms.from_mesh(sim_me)
    bmesh.ops.dissolve_degenerate(bms, dist=1e-7, edges=bms.edges[:])
    bmesh.ops.triangulate(bms, faces=bms.faces[:])
    tris_decimated = len(bms.faces)
    bms.to_mesh(sim_me)
    bms.free()
    timing["decimate"] = time.perf_counter() - t

    sys.stdout.write("[sim_geom] ground contact\n")
    sys.stdout.flush()
    t = time.perf_counter()
    bmc = bmesh.new()
    bmc.from_mesh(sim_me)
    geom = bmc.verts[:] + bmc.edges[:] + bmc.faces[:]
    bmesh.ops.bisect_plane(bmc, geom=geom, plane_co=Vector((0.0, 0.0, 0.0)),
                           plane_no=Vector((0.0, 0.0, 1.0)), clear_inner=True, dist=1e-4)
    bmc.edges.ensure_lookup_table()
    bedges = [e for e in bmc.edges if len(e.link_faces) == 1]
    loops = _group_loops(bmc, bedges)
    if len(loops) != 4:
        refuse("SG-CONTACT", f"expected 4 contact loops at z=0, found {len(loops)}")
    capverts = set()
    for loop in loops:
        for e in loop:
            capverts.update(e.verts)
    for v in capverts:
        v.co.z = 0.0
    # R12b: collapse every edge below SHORT_EDGE_M before capping, so the written
    # surface survives the house weld as well as the exact float32 merge — an
    # edge collapse touches only that edge, never welds two sheets together.
    for _ in range(8):
        short = [e for e in bmc.edges if e.calc_length() < SHORT_EDGE_M]
        if not short:
            break
        bmesh.ops.collapse(bmc, edges=short)
    bmc.edges.ensure_lookup_table()
    bedges = [e for e in bmc.edges if len(e.link_faces) == 1]
    loops = _group_loops(bmc, bedges)
    if len(loops) != 4:
        refuse("SG-CONTACT", f"expected 4 contact loops after the short-edge weld, found {len(loops)}")
    bmc.verts.ensure_lookup_table()
    cap_idx = np.array(sorted({v.index for loop in loops for e in loop for v in e.verts}),
                       dtype=np.int64)
    for loop in loops:
        bmesh.ops.holes_fill(bmc, edges=loop, sides=0)
    bmesh.ops.triangulate(bmc, faces=bmc.faces[:])
    bmc.normal_update()
    bmc.verts.ensure_lookup_table()
    V_s, T_s = bm_to_arrays(bmc)
    min_z = float(V_s[T_s].reshape(-1, 3)[:, 2].min())
    lab_s = components(V_s, T_s)
    ncomp_sim = int(lab_s.max()) + 1 if len(lab_s) else 0
    del lab_s
    timing["contact"] = time.perf_counter() - t

    sys.stdout.write("[sim_geom] patches\n")
    sys.stdout.flush()
    t = time.perf_counter()
    tri_c = V_s[T_s].mean(axis=1)
    xs = tri_c[:, 0]
    zs = tri_c[:, 2]
    is_cap = np.isin(T_s, cap_idx).all(axis=1)
    bvh = BVHTree.FromBMesh(bm)
    find_nearest = bvh.find_nearest
    wheel_or_cap = is_cap.copy()
    centroids = tri_c.tolist()
    for i in range(len(T_s)):
        if wheel_or_cap[i]:
            continue
        hit = find_nearest(centroids[i])
        if hit[2] is not None and hit[2] in wheel_set:
            wheel_or_cap[i] = True
    idx = {n: i for i, n in enumerate(PATCH_ORDER)}
    patch = np.full(len(T_s), idx["body"], dtype=np.int64)
    patch[zs <= Z_FLOOR] = idx["floor"]
    patch[((xs > x_rc) & (zs > z_rt)) | ((xs > x_rr) & (zs > Z_REAR_LOW))] = idx["rear_wing"]
    patch[xs < x_fw] = idx["front_wing"]
    patch[wheel_or_cap] = idx["wheels"]
    tri_a = 0.5 * np.linalg.norm(
        np.cross(V_s[T_s[:, 1]] - V_s[T_s[:, 0]], V_s[T_s[:, 2]] - V_s[T_s[:, 0]]), axis=1)
    counts = np.bincount(patch, minlength=5)
    areas = np.bincount(patch, weights=tri_a, minlength=5)
    timing["patches"] = time.perf_counter() - t

    sys.stdout.write("[sim_geom] write + read-back check\n")
    sys.stdout.flush()
    t = time.perf_counter()
    for name in PATCH_ORDER:
        sel = patch == idx[name]
        write_stl(os.path.join(args.out, name + ".stl"), V_s, T_s[sel], name)
    write_stl(os.path.join(args.out, "sim_surface.stl"), V_s, T_s, "sim_surface")
    attribution = ""
    lic_src = os.path.join(os.path.dirname(args.glb), "LICENSE.txt")
    if os.path.isfile(lic_src):
        shutil.copyfile(lic_src, os.path.join(args.out, "LICENSE.txt"))
        with open(lic_src, "r", encoding="utf-8", errors="replace") as f:
            attribution = parse_attribution(f.read())
    paths = [os.path.join(args.out, n + ".stl") for n in PATCH_ORDER]
    uni = check_union(paths)
    timing["write_check"] = time.perf_counter() - t

    if not args.no_preview:
        sys.stdout.write("[sim_geom] preview\n")
        sys.stdout.flush()
        tp = time.perf_counter()
        try:
            bm.to_mesh(me)
            me.update()
            bmc.to_mesh(sim_me)
            sim_me.update()
            render_preview(obj, sim_ob, patch, os.path.join(args.out, "preview.png"), args.samples)
        except Exception:
            traceback.print_exc()
        timing["preview"] = time.perf_counter() - tp
    else:
        timing["preview"] = 0.0

    timing["total"] = time.perf_counter() - t0

    reasons = []
    house_bad = []
    if uni["house_open_edges"]:
        house_bad.append(f"house_open={uni['house_open_edges']}")
    if uni["house_nonmanifold_edges"]:
        house_bad.append(f"house_nonmanifold={uni['house_nonmanifold_edges']}")
    if uni["min_edge_m"] < uni["house_weld_m"]:
        house_bad.append(f"min_edge={uni['min_edge_m']:.4e} < house_weld={uni['house_weld_m']:.4e}")
    watertight = (uni["open_edges"] == 0 and uni["nonmanifold_edges"] == 0
                  and uni["degenerate_tris"] == 0 and not house_bad)
    if not watertight:
        reasons.append(f"watertight open={uni['open_edges']} nonmanifold={uni['nonmanifold_edges']}"
                       f" degenerate={uni['degenerate_tris']}"
                       + ("; " + ", ".join(house_bad) if house_bad else ""))
    volume_rel = uni["volume_m3"] / vol_ref - 1.0
    if abs(volume_rel) > 0.02:
        reasons.append(f"volume_rel={volume_rel:.4f}")
    frontal_rel = uni["frontal_area_m2"] / fa_ref - 1.0
    if abs(frontal_rel) > 0.02:
        reasons.append(f"frontal_rel={frontal_rel:.4f}")
    tris_ok = 500000 <= uni["tris"] <= 2000000
    if not tris_ok:
        reasons.append(f"tris={uni['tris']}")
    patches_ok = bool(np.all(counts >= 1)) and int(counts.sum()) == len(T_s) == uni["tris"]
    if not patches_ok:
        reasons.append(f"patches counts={counts.tolist()} sum={int(counts.sum())} sim={len(T_s)}")
    gate_pass = not reasons

    result = {
        "tool": "tools/promo/sim_geom.py",
        "source": {"glb": args.glb, "bytes": os.path.getsize(args.glb), "attribution": attribution},
        "params": {"weld_m": args.weld, "voxel_m": args.voxel, "target_tris": args.target_tris,
                   "contact_depth_m": args.contact, "floater_m3": args.floater, "raster_m": args.raster},
        "placement": {"scale": 0.001, "translate_m": [float(tr.x), float(tr.y), float(tr.z)],
                      "length_m": length, "width_m": width, "height_m": height},
        "render_model": {"objects": n_objects, "tris": int(n_tris_raw), "open_edges_raw": open_raw,
                         "nonmanifold_raw": nm_raw, "open_edges_welded": open_weld,
                         "nonmanifold_welded": nm_weld, "open_edges_filled": open_filled,
                         "components": ncomp_ref, "wheel_components": 4, "volume_m3": vol_ref,
                         "frontal_area_m2": fa_ref},
        "sim_surface": {"tris_remesh": int(tris_remesh), "floaters_dropped": floaters_dropped,
                        "floater_tris_dropped": floater_tris, "tris_decimated": int(tris_decimated),
                        "contact_loops": 4, "tris": int(len(T_s)), "components": ncomp_sim,
                        "open_edges": uni["open_edges"], "nonmanifold_edges": uni["nonmanifold_edges"],
                        "degenerate_tris": uni["degenerate_tris"], "min_z_m": min_z,
                        "min_edge_m": uni["min_edge_m"], "house_weld_m": uni["house_weld_m"],
                        "house_open_edges": uni["house_open_edges"],
                        "house_nonmanifold_edges": uni["house_nonmanifold_edges"],
                        "volume_m3": uni["volume_m3"], "frontal_area_m2": uni["frontal_area_m2"]},
        "patch_rule": {"x_fw": x_fw, "x_rc": x_rc, "z_rt": z_rt, "x_rr": x_rr,
                       "z_rear_low": Z_REAR_LOW, "z_floor": Z_FLOOR},
        "patches": [{"name": n, "file": n + ".stl", "tris": int(counts[idx[n]]),
                     "area_m2": float(areas[idx[n]])} for n in PATCH_ORDER],
        "gate": {"watertight": watertight, "volume_rel": volume_rel, "frontal_rel": frontal_rel,
                 "tris_ok": tris_ok, "patches_ok": patches_ok, "pass": gate_pass, "reasons": reasons},
        "timing_s": {k: float(timing[k]) for k in ("import", "weld_fill", "reference", "remesh",
                                                   "decimate", "contact", "patches", "write_check",
                                                   "preview", "total")},
    }
    with open(os.path.join(args.out, "sim_geom.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=1)
    print("GATE PASS" if gate_pass else "GATE FAIL " + "; ".join(reasons))
    return 0 if gate_pass else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
