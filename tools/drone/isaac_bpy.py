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
# isaac_bpy.py - export the x500 .usdc inside Blender.
# No GPL-licensed source was consulted.

"""Blender side of DRONE-ISAAC: the x500 USD asset.

Run by tools/drone/isaac_export.py as a separate program:

  blender -b <geom>/x500_assembled.blend --python tools/drone/isaac_bpy.py --
      usd --out <file.usdc> --manifest <json>

usd runs on the already-opened x500_assembled.blend: one usd_export of all
objects with materials, plus the world-space bbox of every MESH object's
evaluated vertices in the manifest.

(The flow .vdb is no longer written here: tools/drone/isaac_vdb.py writes
it in plain Python - so that file stays under the repository's Prosperity
licence - with Isaac Sim's own openvdb module, whose file format 224 Isaac
Sim 6.0 reads; Blender's bundled module writes 225, which Isaac refuses.)

The drone model: (c) 2022 Rudis Laboratories / PX4 Autopilot for Drones,
BSD-3-Clause.
"""

import json
import os
import sys

import bpy


def _die(msg):
    print("[isaac_bpy] ERROR: " + str(msg), flush=True)
    sys.exit(1)


def _flags(argv):
    out = {}
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == "--":
            i += 1
            continue
        if a.startswith("--"):
            key = a[2:]
            if i + 1 < len(argv) and not argv[i + 1].startswith("--"):
                out[key] = argv[i + 1]
                i += 2
            else:
                out[key] = True
                i += 1
        else:
            _die("unexpected argument: " + a)
    return out


def _need(flags, mode, keys):
    for k in keys:
        if k not in flags:
            _die(mode + " needs --" + k)


def mode_usd(argv):
    flags = _flags(argv)
    _need(flags, "usd", ("out", "manifest"))
    out = flags["out"]
    d = os.path.dirname(out)
    if d:
        os.makedirs(d, exist_ok=True)
    bpy.ops.wm.usd_export(filepath=out, selected_objects_only=False,
                          export_materials=True, export_cameras=False,
                          export_lights=False)
    meshes = [o for o in bpy.data.objects if o.type == "MESH"]
    lo_pt = [float("inf")] * 3
    hi_pt = [float("-inf")] * 3
    deps = bpy.context.evaluated_depsgraph_get()
    for o in meshes:
        oe = o.evaluated_get(deps)
        me = oe.to_mesh()
        mw = oe.matrix_world
        for v in me.vertices:
            w = mw @ v.co
            for i in range(3):
                if w[i] < lo_pt[i]:
                    lo_pt[i] = float(w[i])
                if w[i] > hi_pt[i]:
                    hi_pt[i] = float(w[i])
        oe.to_mesh_clear()
    man = {"file": out.replace(os.sep, "/"), "bytes": int(os.path.getsize(out)),
           "mesh_objects": len(meshes), "bbox_min": lo_pt, "bbox_max": hi_pt}
    with open(flags["manifest"], "w", encoding="utf-8") as fh:
        json.dump(man, fh, indent=2)
    print("[isaac_bpy] usd wrote " + out + " (" + str(len(meshes)) + " meshes)",
          flush=True)


def main():
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    else:
        argv = []
    if not argv:
        _die("usage: isaac_bpy.py -- usd ...")
    mode = argv[0]
    rest = argv[1:]
    if mode == "usd":
        mode_usd(rest)
    else:
        _die("unknown mode: " + mode)


try:
    main()
except SystemExit:
    raise
except Exception as exc:
    _die(type(exc).__name__ + ": " + str(exc))
