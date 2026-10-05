<!-- meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
     No GPL-licensed source was consulted. -->

# Licences in tools/drone

## Which file is under which licence

| File | Imports bpy | Licence |
|---|---|---|
| `sim_geom.py` | yes (bpy, bmesh, mathutils) | GPL-2.0-or-later |
| `tunnel_mesh.py` | no | Prosperity Public License 3.0.0 |
| `LICENSE-NOTICE.md` | - | Prosperity Public License 3.0.0 |

## Why `sim_geom.py` is GPL-2.0-or-later

`sim_geom.py` imports Blender's Python API (`bpy`, and the `bmesh` and
`mathutils` modules shipped inside Blender), and it also imports the helper
functions of `../promo/sim_geom.py`, which is itself GPL-2.0-or-later, so it
must be GPL-compatible too. Blender is GPL software; the Blender Foundation
treats a script that uses its API as a work that runs inside Blender's
process, so it must be GPL-compatible — the reasons are spelled out in
`../promo/LICENSE-NOTICE.md`, whose argument this file follows. The licence
text is `../promo/COPYING` (the GNU GPL version 2).

## Why this does not make meteor-cfd GPL

`sim_geom.py` is a separate program run by `blender.exe -b --python`. The Rust
engine, the GUI and the rest of the tools never import, link into or load it.
They meet it only through files on disk (binary STL, JSON, PNG). That is "mere
aggregation" in the sense of GPL-2.0 section 2, as in `../promo/LICENSE-NOTICE.md`:
the GPL covers this script and nothing else in the repository. Iterations Co.,
Ltd. wrote the script and holds its copyright, so the licence is outbound only.

## What the GPL does not cover

The files the script writes (`body.stl`, `arms.stl`, `motors.stl`, `skids.stl`,
`sim_surface.stl`, `rotors.json`, `sim_geom.json`, `preview.png`) are its
output — data, not a work based on the program. The PX4 x500 model is a
separate work: "(c) 2022 Rudis Laboratories / PX4 Autopilot for Drones",
BSD-3-Clause. The GLB and every mesh made from it stay outside the repository,
the model's `LICENSE.txt` and `LICENSE` are copied beside every output, the
copyright holders are credited in the video, and no endorsement by them is
implied.

## The rule for new scripts here

The same rule as `tools/promo`: a new `tools/drone` script that imports `bpy`,
`bmesh` or `mathutils` gets the GPL header of `sim_geom.py` verbatim and a row
in the table above; a script that does not import them keeps the house header
and must never import a GPL script — run it as a separate Blender process
instead.
