# tools/isaac/probe.py - the generic Isaac visibility probe (VIO-10)

Headless Isaac Sim probe that decides, from rendered pixels alone, whether
a volume prim is actually visible in a scene: SPEC-LIT 117.10.

## What it does

Opens `--scene` (a .usda) through SimulationApp headless, lights it
(DomeLight 1000 + DistantLight 3000), finds the volume prim (a UsdVol.Volume,
or a Mesh flagged `bool primvars:isVolume = 1`), frames a box around it and
renders, per view, frame `a` with the volume shown and - first view only -
frames `b`, `c` with it hidden, grabbed back to back.  Then

    changed(a, b) = #{ p : d(p) > 4 }     d(p) = largest abs RGB diff / 255
    shows(a, b) <=> changed(a, b)/N_px > 0.0005 and max_p d(p) > 4   (117.15)
    visible    <=> shows(a, b) and changed(a, b) > 2 changed(b, c)   (117.16)

b/c is the re-render noise: a and b are separate accumulations whose speckle
is not the volume.  The verdict goes to probe.json and to the one-line
summary `ISAAC-PROBE <json>` on stdout; the stage is never saved.

## Running it (supervisor only)

Only the supervisor runs this, with Isaac's own Python, never this
repository's tooling:

    C:/iss/env/Scripts/python.exe tools/isaac/probe.py \
        --scene SCENE.usda --out-dir DIR [--profile auto|index|pathtrace] \
        [--renderer NAME] [--volume PRIMPATH] [--box X0 Y0 Z0 X1 Y1 Z1] \
        [--vfov-deg 60.0] [--width 1280] [--height 720] [--frames 3] \
        [--warmup 120] [--max-seconds 540.0]

## Profiles and renderer settings

`--profile auto` reads the scene text: any `primvars:isVolume = 1` (or
`= true`) means pathtrace, else a `def Volume "` means index, else auto
fails.  `renderer_for` maps a profile to its renderer and carb settings:

- index     -> RaytracedLighting, no extra settings
- pathtrace -> PathTracing with
               /rtx/raytracing/indexdirect/enabled = false
               /rtx/index/compositeEnabled = false
               /rtx/pathtracing/ptvol/enabled = true
               /rtx/pathtracing/nonUniformVolumes = true

## Framing (117.17)

With r the box half-diagonal and theta_v the vertical field of view, each
eye sits at R = 1.1 r / sin(theta_v / 2) from the box centre, in the
direction front34 (1, -1, 0.6) / side (0, -1, 0) / back34 (-1, 1, 0.6)
(Z-up, converted for Y-up stages), clipping [0.01 R, 10 (R + r)], and the
camera gets focalLength = V / (2 tan(theta_v / 2)) with V = 20.955 *
height / width (horizontalAperture 20.955).  `--box` overrides the volume
prim's world bound; explicit views (the drone caller) skip the framing and
keep the USD default lens.

## probe.json and exit codes

Keys: tool, version, isaac_version, profile, renderer, scene, stage_loaded,
mesh_prims, volume_prims, volume_path, box, frames, diff,
noise, volume_visible, wall_s, timed_out, error, plus caller extras.  Exit
0 normal, 1 on an exception, 3 on the --max-seconds watchdog (which still
writes probe.json and prints the summary line).

## Selftest

`--selftest` runs P1-P10 in plain Python (stdlib + numpy, no Isaac):

    python tools/isaac/probe.py --selftest      # SELFTEST PASS 10/10

## Beside it

- tools/drone/isaac_probe.py is a thin caller of this module: it only wires
  the drone scene's /World/Flow staging and its three camera triples.
- tools/isaac/usda_check.py (VIO-09) is the scene checker: run it on a
  .usda before spending an Isaac run on it.
