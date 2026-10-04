<!-- meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
     No GPL-licensed source was consulted. -->

# Licences in tools/promo

## Which file is under which licence

| File | Imports bpy | Licence |
|---|---|---|
| `scene.py` | yes (bpy, mathutils) | GPL-2.0-or-later |
| `sim_geom.py` | yes (bpy, bmesh, mathutils) | GPL-2.0-or-later |
| `tunnel_mesh.py` | no | Prosperity Public License 3.0.0 (the repository's LICENSE) |
| `solve.py` | no | Prosperity Public License 3.0.0 (the repository's LICENSE) |
| `post.py` | no | Prosperity Public License 3.0.0 (the repository's LICENSE) |
| `COPYING` | - | the GNU GPL version 2 text (verbatim, from Blender 5.1's license/spdx/GPL-2.0-or-later.txt; line endings LF) |
| `LICENSE-NOTICE.md` | - | Prosperity Public License 3.0.0 |

## Why the two Blender scripts are GPL

Blender is GPL software. Its source is GPL-2.0-or-later, as its own copyright.txt
states; the Blender 5.1 binary is released under GPL-3.0-or-later because some of
its dependencies are Apache-2.0. The Blender Foundation treats a Python script
that uses Blender's Python API (`bpy`, and the `bmesh` and `mathutils` modules
shipped inside Blender) as a work that runs inside Blender's process, so it must
be GPL-compatible. These two scripts are therefore licensed GPL-2.0-or-later;
"or later" makes them usable with the GPL-3.0-or-later Blender binary. This was
decided by recommendation (2026-10-04), not by a legal opinion.

## Why this does not make meteor-cfd GPL

The two scripts are separate programs run by `blender.exe -b --python`. The Rust
engine, the GUI and `tunnel_mesh.py` do not import, link into or load them. They
talk to the rest of meteor-cfd only through files on disk (binary STL, `.npz`,
JSON, PNG). That is "mere aggregation" in the sense of GPL-2.0 section 2: the
GPL covers the two scripts and nothing else in the repository.

## What the GPL does not cover

The files the scripts write (sim surface STLs, renders, `.npz`, `.blend`, JSON)
are their output. GPL-2.0 section 0 covers output only when its contents are a
work based on the program, and these are data. The F1 render model is a separate
CC BY 4.0 work ("F1 2026 concept" by Qvist_Designs, via Sketchfab); it and
everything made from it stay outside the repository, and its attribution goes in
the video credits. The GPL changes nothing about that.

## Copyright and provenance

Iterations Co., Ltd. wrote both scripts and holds their copyright, so it may also
licence its own code to others on other terms; the GPL grant is to recipients.
No GPL-licensed source was consulted in writing them: Blender's source was never
read, only its Python API was called. See `rust/PROVENANCE.md` and
`tools/promo/COPYING`.

## The rule for new scripts here

A new `tools/promo` script that imports `bpy`, `bmesh` or `mathutils` gets the
GPL header of `scene.py` verbatim and a row in the table above. A script that
does not import them keeps the house header and must never `import` a GPL
script; run it as a separate Blender process instead.
