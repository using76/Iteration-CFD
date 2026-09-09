# Iterations-CFD — Technical Guidebook

How to actually run a GPU-resident finite-volume CFD solver: installation, a
first case, meshing, choosing a solver, reading convergence, looking at results,
and the causes of the errors you will meet.

What this document does not cover: the derivation of each discretisation and its
check against the source paper is in [`rust/SPEC-LIT.md`](../rust/SPEC-LIT.md);
what was read to write each file is in
[`rust/PROVENANCE.md`](../rust/PROVENANCE.md). This document sits on top of those
and talks only about **use**.

한국어: [`GUIDEBOOK.md`](GUIDEBOOK.md)

---

## Contents

1. [What it is, and what it is not](#1-what-it-is-and-what-it-is-not)
2. [Installing](#2-installing)
3. [Five minutes: a first solve](#3-five-minutes-a-first-solve)
4. [The shape of a case](#4-the-shape-of-a-case)
5. [Making a mesh](#5-making-a-mesh)
6. [Choosing a solver — where people go wrong](#6-choosing-a-solver--where-people-go-wrong)
7. [Running it, and reading convergence](#7-running-it-and-reading-convergence)
8. [Looking at results](#8-looking-at-results)
9. [The Studio](#9-the-studio)
10. [MCP — handing the solvers to your own assistant](#10-mcp--handing-the-solvers-to-your-own-assistant)
11. [Verification and grounds](#11-verification-and-grounds)
12. [What it cannot do](#12-what-it-cannot-do)
13. [Troubleshooting](#13-troubleshooting)
14. [Licence](#14-licence)

---

## 1. What it is, and what it is not

**The whole time-integration loop stays on the GPU.** Once the mesh and the
fields are uploaded, no iteration allocates device memory or brings a field back
to the host. That is the only reason it is fast, and it is also the reason for
the constraint: **one card, one process**. There is no MPI and no multi-GPU.

The host is Rust 1.85, the kernels are CUDA C++, double precision by default
(single via the `single` feature), and the dependencies are `cudarc` and
`thiserror` — that is all (AMGX is optional).

**What it is for**: incompressible and low-Mach flow. RANS/LES/hybrid turbulence,
buoyant plumes, variable-density low-Mach, two-phase VOF, conjugate heat
transfer, surface-to-surface radiation, Lagrangian sprays, and ventilation and
data-centre airflow with fan curves and porous jumps.

**What it is not**: a compressible or transonic solver. There is no finite-rate
chemistry. It publishes no multi-GPU scaling numbers. The full list is in
[§12](#12-what-it-cannot-do).

And one more thing — **the comparison is not yet sufficient, and help with it is
very welcome.** Verification uses manufactured solutions, analytic solutions and
published benchmarks. What has not been closed is not hidden: `ofgpu-validate`
names it on every run. If you know of a measurement or a case worth holding this
against, please say so ([§11](#11-verification-and-grounds)).

---

## 2. Installing

### Requirements

| | |
|---|---|
| GPU | one NVIDIA GPU with CUDA 13 support |
| OS | Windows 10/11 x64 |
| Runtime | Visual C++ 2015-2022 redistributable |
| (building from source) | Rust 1.85+, Visual Studio 2022 C++ workload, CUDA Toolkit 13.x |

CUDA 13 is linked **statically**, so an installed build needs no CUDA Toolkit.
A driver is enough.

### The installer

Take `iterations-*.exe` from the releases and run it. Tick the PATH option and
`ofgpu-*` runs from any prompt.

It installs the 16 solver binaries, the case definitions (`cases/*.jsonc`), the
race-car sample, `docs/` (this guide plus SPEC-LIT, PROVENANCE and the schema),
and the licence documents.

### From source

```powershell
cd rust
cargo build --release
```

`build.rs` sets up the MSVC environment through `vcvars64.bat` and compiles every
`.cu` to **CUBIN**, not PTX — precompiled for the architecture of the machine
that built it, with no runtime JIT.

### Check the install

```powershell
ofgpu-probe
```

It reports the device, the scalar precision, and whether the device result is
**bitwise identical** to the host result. You want `max |gpu - cpu| = 0.000e0`
and `PASS`. If that fails, nothing below it can be trusted.

---

## 3. Five minutes: a first solve

The race-car sample is the fastest route: one geometry and two commands.

```powershell
cd cases
.\racecar.cmd
```

What the script does:

```powershell
# 1. Carve the body out of a 1 m cube tunnel, cut-cell.   minutes (all cores)
ofgpu-generate-mesh big racecar_case 128 -stl car=racecar.stl -cutcell

# 2. Solve the momentum too.                              1-2 min (GPU)
ofgpu-lowmach racecar_case -iters 3000 -check 250 -output foam
```

Step 1's inside/outside classification — two million cells each tested
against the STL, the ones the surface crosses cut open — now runs on every
core. Measured on 32 cores, a 96-cube carve of this car went from 559 s to
48 s, about 11.6x; the 128-cube this sample builds takes minutes. Step 2 is
the GPU-resident loop and takes minutes.

The generator also writes `p` and `T` into `0/` itself, so the copy step the
recipe used to carry is gone. Which driver solves what is
[§6](#6-choosing-a-solver--where-people-go-wrong) — in short, the
turbulence-only drivers solve two turbulence equations on a frozen `U`.

When it finishes, `racecar_case/0/` holds `U`, `p`, `T`, `k`, `epsilon`,
`omega` and `nut` in OpenFOAM ASCII. Open it in the Studio's 3D viewer, or
read it straight into ParaView.

Details: [`cases/racecar.md`](../cases/racecar.md).

---

## 4. The shape of a case

There are two ways to hand over a case.

### (a) A JSONC case — one file

Mesh, properties, boundary conditions and numerics in a single JSON file that
allows comments and trailing commas. The schema is generated into
[`docs/schema/case-1.json`](schema/); examples are
[`docs/case-example.json`](case-example.json) and `cases/*.jsonc`.

Drivers that read it: `ofgpu-lowmach`, `ofgpu-k-epsilon`, `ofgpu-cht`,
`ofgpu-datacentre`, `ofgpu-decompose`.

Results go to `<stem>_jsonc/<time>/`.

### (b) An OpenFOAM case directory

```
racecar_case/
  0/                       initial and boundary fields (U, p, T, k, epsilon, omega, nut, ...)
  constant/
    polyMesh/              points, faces, owner, neighbour, boundary
    physicalProperties     viscosityModel, nu
    momentumTransport      simulationType, RAS { model kEpsilon | kOmega | ... }
    g                      gravity vector, if the case has one
  system/
    controlDict            startTime, endTime, deltaT, writeControl
    fvSchemes              ddt, div, grad, laplacian, snGrad
    fvSolution             solvers, relaxation, correctors
```

It reads and writes the OpenFOAM ASCII format. **That is interoperability, not
derivation** — ofgpu links against no part of OpenFOAM and contains none of its
source. A file format is not a work.

### Worth knowing

- `model` in `momentumTransport` picks the turbulence model. The mesh generator
  writes `kEpsilon`; to run a k-omega driver you must change it.
- An unrecognised setting is **refused, not silently substituted** (SPEC-LIT
  §13.4). Pass `-permissive` to fall back to documented defaults — it prints
  what it replaced with what.
- If `constant/g` is present, gravity really enters the equations. Do not add one
  to change the viewer's up axis. Without it the viewer reads up from the normal
  of a `bottomWall`, `floor` or `ground` patch.

---

## 5. Making a mesh

```powershell
ofgpu-generate-mesh <preset> <outputDir> [nx ny nz] [-stl [name=]path]...
                    [-cutcell [-s N] [-thetaMin X]]
                    [-extent xlo xhi ylo yhi zlo zhi] [-grading x|y|z=r]...
                    [-wallModel standard|spalding|rough|lowRe [-Ks x [-Cs y]]]
                    [-cyclic x|y|z] [-permissive]
```

Presets: `channel`, `cavity`, `step`, `big`, `plume`, `room`, `damBreak`. `big`
is a 1 m cube tunnel and takes a **single** cell count (`n³`).

What comes out is a complete, ready-to-run case: `constant/polyMesh`,
`constant/physicalProperties`, `constant/momentumTransport`,
`system/{controlDict,fvSchemes,fvSolution}`, and a `0/` with `U`, `k`, `epsilon`,
`omega` and `nut` — plus `p` and `T` when the preset is `channel`, `cavity`,
`step` or `big`, or one of the buoyant pair (`plume`/`room`), on the block and
the cut-cell path alike. `damBreak` is the one exception: the two-phase path
puts `alpha.water` and `p_rgh` in `0/` instead.

### A block the size of your site — `-extent`, `-grading`

```powershell
ofgpu-generate-mesh big site 320 260 40 -extent -1240 400 -800 500 3.5 200 -grading z=6 -stl site=site.stl -cutcell
```

When a preset's fixed extent (`big` is a unit cube) cannot hold a real site,
`-extent xlo xhi ylo yhi zlo zhi` (metres) sets the block's six faces directly
and `-grading x|y|z=r` sets the cell growth along one axis (last cell divided
by first cell) — `r > 1` puts the smallest cell at the axis's low end (the
ground when the axis is z). Both flags apply to the plain path, to `-stl`
castellation and to `-cutcell` alike. The block's six patches keep the
preset's names and types (for `big`: `inlet` at xMin, `outlet` at xMax,
`bottomWall`, `topWall`, `backWall` at yMin, `frontWall` at yMax) — rename or
retype them in `constant/polyMesh/boundary` and `0/` afterwards. `plume`,
`room` and `damBreak` refuse `-extent`: their openings (and, for damBreak,
the water column) are placed from the preset's own extents.

### Putting a geometry in

```powershell
ofgpu-generate-mesh big case 128 -stl car=racecar.stl -cutcell
```

`-stl [name=]path` adds the geometry; `-cutcell` cuts the cells the surface
crosses rather than stair-stepping them. The surface becomes a wall patch under
the name you gave.

Cut-cell classification (SPEC-LIT §23.2) needs a **closed manifold**. Two parts
meeting face-to-face are rejected as `non-manifold edge(s)`, so make them
**overlap** instead.

`-s N` is the sub-sampling per cell and `-thetaMin X` the merge volume-fraction
threshold; the defaults are `s = 16` and `theta_min = 0.2`. It reports:

```
[cutcell] 128 x 128 x 128 block, s = 16, theta_min = 0.2:
          3183 solid, 2087745 fluid, 6224 cut (917 merged) -> 2093052 cells
[cutcell] new wall patch car: 5850 face(s)
```

`solid` are lattice sites inside the body with no cell left, `cut` are cells the
surface crossed, `merged` are cells too small to keep and folded into a neighbour.

### Wall-treatment presets

`-wallModel` fills the `nut`/`k`/`epsilon`/`omega` (and `T`, when the energy
equation is solved) boundary types on every wall of a case as **one consistent
row** — so that picking them field by field cannot produce a contradictory mix.

| Preset | `nut` | `k` | `epsilon`/`omega` |
|---|---|---|---|
| `standard` (default) | `nutkWallFunction` | `kqRWallFunction` | `epsilonWallFunction`/`omegaWallFunction` |
| `spalding` | `nutUWallFunction` | `kqRWallFunction` | as above |
| `rough` | `nutkRoughWallFunction` (needs `-Ks`) | `kqRWallFunction` | as above |
| `lowRe` | `nutLowReWallFunction` | `kLowReWallFunction` | `epsilon`: `fixedValue` (no `value`, so 0) / `omega`: `zeroGradient` |

The full table, including how it collapses for LES, is in
[`cases/README.md`](../cases/README.md).

### From a STEP geometry to a mesh, and to Fluent

CAD hands you a STEP file; the solver eats a mesh. Two steps sit between them —
Gmsh builds the mesh, and this repository's converter brings it in. This section
is the full manual for those two steps. If you want only the summary — the
schema, the 10-stage pipeline table, the checkers —
[`tools/mesh/README.md`](../tools/mesh/README.md) carries it; this section
carries the narrative. The worked example is one ammonia release site: from a
single STEP file to about 3.6 M tetrahedra, with the timings below measured on
that run.

#### What you need

- **One STEP file.** Its units are whatever `scale` says they are. The tool
  never reads the file's unit declaration, it just multiplies — `0.001` for a
  mm STEP, `1.0` if it is already in metres.
- **Python 3.13** with `gmsh`, `numpy`, `scipy`, `pymeshlab`. What
  `step_mesh.py` imports directly is gmsh and numpy; the surface-repair route
  ([`repairs`](#writing-the-config)) additionally needs pymeshlab. The numbers
  in this document come from a gmsh 4.14.1 environment.
- **The built converter** `rust\target\release\ofgpu-convert-mesh.exe` — built
  by `cd rust; cargo build --release` ([§2](#2-installing)). The Studio-side
  tool looks at the `OFGPU_BIN_DIR` environment variable first, then at the
  same build trees.

The fastest way to confirm both are in place — it builds its own tiny STEP,
meshes it four ways in seconds, and, when the converter is built, checks the
polyMesh and a Fluent mesh too:

```powershell
python tools\mesh\selftest.py
```

#### Writing the config

The setup is a single JSON file. The full schema is in
[`tools/mesh/README.md`](../tools/mesh/README.md); unknown keys are refused by
name, missing keys take their defaults, and only `step`, `out_dir` and
`domain_box` are required. The whole ammonia site is this one file
([`tools/mesh/examples/nh3_site.json`](../tools/mesh/examples/nh3_site.json)):

```json
{
  "step": "C:/Users/sdd32/Desktop/암모니아누출/cfd_optimized_defeatured_recommended.step",
  "scale": 0.001,
  "out_dir": "C:/Users/sdd32/Desktop/암모니아누출/mesh",
  "name": "nh3_site",
  "fluid": {"tag": 1},
  "domain_box": [-1250.0, -1250.0, -7.5, 1250.0, 1250.0, 200.0],
  "outer_tol": 0.05,
  "solids": {"sink_m": 1.5, "fuse": false, "exclude_tags": []},
  "repairs": [
    {"tag": 33, "method": "resample", "cell_m": 1.5, "target_faces": 6000, "lift_z": 3.05}
  ],
  "trim": {"below_z": 3.05},
  "sea_z": 3.05,
  "points": {
    "tank_shell":    [-916.9, 349.8],
    "ESDV1":         [-910.2, 329.3],
    "skid":          [-869.5, 233.8],
    "pipe_mid":      [-434.8, 116.9],
    "ESDV2":         [-23.2,    9.4],
    "ship_manifold": [  18.5,  -7.5]
  },
  "pool_radius_m": 26.0,
  "roof_patches": {"nh3_source": 306},
  "sizes": {
    "min": 1.5, "max": 40.0, "pool": 2.0, "box": 4.0, "growth_from": 2.5,
    "near_struct": 4.0, "far_struct": 12.0, "near_radius": 400.0,
    "size_mult": 1.0, "roof_boxes": [2.5, 5.0, 10.0]
  },
  "mesh": {"algo2d": 6, "algo3d": 1, "optimize_passes": 5, "threads": 32},
  "post": {
    "flat_tets": true, "flat_threshold": 1e-7, "seam_merge_m": 0.02,
    "sliver_edge_m": 0.6, "sliver_vol_m3": 0.2, "thin_push_m": 0.0
  },
  "classification": {"wall_prefix": "wall_", "big_roof_is_ground_m2": 2000.0}
}
```

What changes when you change a key:

| Key | What happens |
|---|---|
| `step` | The STEP to read. Another site starts by changing this line alone |
| `scale` | Multiplied into the STEP coordinates (`Geometry.OCCScaling`, applied BEFORE the import). Get it wrong and the geometry is a thousand times too big or small, and the cut refuses on the expected-mass check |
| `out_dir` | Where the `.msh`/`.vtk`/`_summary.json` land, and `work/` with them |
| `name` | The prefix of every output and checkpoint file |
| `fluid` | Which imported solid is the flow domain. When you do not know the tags, `{"largest": true}` |
| `domain_box` | The domain box. Five of its six faces become the `top`/`west`/`east`/`south`/`north` patches. Grow it and the far field fills with that much more cell at `sizes.max` |
| `outer_tol` | Tolerance of the outer-face tests. When an outer patch comes out empty and the run refuses, raise this first |
| `solids.sink_m` | Stretches every building base this far downwards, about its roof (removes the hairline air layer a floating base leaves above the terrain). But keep the bases off the trim plane: 2.0 puts them at exactly 3.0 and 1,814 faces reach below the trim, 1.5 puts none there |
| `solids.fuse` | `true` fuses the solids before the cut — merging the coincident faces and hairline slits of touching or overlapping neighbours, at the price of a heavier boolean |
| `solids.exclude_tags` | Solids left out of the cut entirely (removed from the model) |
| `repairs[].tag` | The solid to replace with a repair — for self-intersecting solids, which on this site means the ship's hull |
| `repairs[].method` | Only `"resample"` exists: a 2-D mesh at 3 m → pymeshlab uniform resampling on a `cell_m` grid → quadric decimation (the coarsest of the `target_faces`/10000/16000 ladder that is watertight, manifold and free of self-intersections) → an OCC solid of planar triangles. Cached as `work/repaired_<tag>.brep`; pass `"brep"` to reuse a cache |
| `repairs[].cell_m` | The resampling grid. Coarser blunts the hull and yields fewer faces |
| `repairs[].target_faces` | The first target of the decimation ladder. Lower is a cruder proxy |
| `repairs[].lift_z` | How far the repaired solid is raised afterwards. This STEP floats the ship at the STEP's z = 0 while the sea plane is +3, hence 3.05; 0 keeps the solid's own height |
| `trim.below_z` | Everything below this plane is cut away with a box; `null` means no trim. This site uses 3.05 because the real sea surface is at z = +3: five centimetres above the existing face, the boolean does not have to cut along a face that is already there, and the plane the cut creates is what becomes `wall_sea_surface` |
| `sea_z` | Flat faces within ±0.06 of this height are classified `wall_sea_surface`. Keep it at the trim height |
| `points` | Name → (x, y). Each point gets its ground height found by an isInside scan from z = −1 in 0.25 m steps, and with it a pool disc, a ±40 m box, a ±150 m box and a growth field. Renaming a point makes the checkpoint refuse — ground heights and pools belong to it, so a rename is a rerun from the start |
| `pool_radius_m` | The radius of the pool disc imprinted at each point. A pool assumes the ground is planar there |
| `roof_patches` | Name → solid tag. That solid's flat roof becomes its own patch, and three refinement boxes settle around it (roof ±60 m, ±250 m, and a −700 m downwind arm) |
| `sizes.min` | `Mesh.MeshSizeMin` — the floor: no edge shorter than this anywhere |
| `sizes.max` | `Mesh.MeshSizeMax` and every field's outside size — the far-field cell size |
| `sizes.pool` | The size inside each point's ±40 m, ground +10 m box (the pool and the first metres above it) |
| `sizes.box` | The size inside each point's ±150 m, +30 m box (the refinement box of the brief) |
| `sizes.growth_from` | The size at a point, held for 40 m and grown to `max` over 700 m |
| `sizes.near_struct` | The size near structures within `near_radius` of a point (grown to `max` over 80 m) |
| `sizes.far_struct` | The size near the rest of the structures (over 120 m) |
| `sizes.near_radius` | The near/far split, measured from the point's (x, y) |
| `sizes.size_mult` | Above 1, multiplied into every size (min, max, every box's VIn, every threshold, Thickness) — a coarse all-over mesh for a solver-robustness reproducer |
| `sizes.roof_boxes` | The three sizes of the boxes around a roof patch [near, mid, downwind] |
| `mesh.algo2d` | The 2-D algorithm (6 = Frontal-Delaunay). Faces left with overlapping triangles are remeshed automatically with 5 (Delaunay), then 1 (MeshAdapt) |
| `mesh.algo3d` | **Leave it at 1 (Delaunay).** HXT (10) hits gmsh's unfinished Steiner-point path during boundary recovery on this class of geometry and kills the whole process. When another algorithm ends with zero tetrahedra the tool retries with Delaunay itself |
| `mesh.optimize_passes` | Passes of gmsh's own optimiser (edge/face swaps + smoothing), threshold 0.5 |
| `mesh.threads` | Threads for the 2-D and 3-D passes |
| `post.flat_tets` | The flat-tet removal stage after the 3-D pass. Delaunay leaves zero-volume tets on planar boundaries; keep it on |
| `post.flat_threshold` | A tet is flat when \|V\| < this × (longest edge from node 0)³ |
| `post.seam_merge_m` | Node pairs closer than this inside a flat tet are merged everywhere (the seam treatment) |
| `post.sliver_edge_m`, `post.sliver_vol_m3` | Edges shorter than `sliver_edge_m` inside a tet of less than `sliver_vol_m3` are collapsed to kill slivers |
| `post.thin_push_m` | Above 0, pushes nodes of thin tets (gamma < 0.02) by up to this. For thin wedges left between the hull and the sea plane (e.g. 0.2) |
| `classification.wall_prefix` | The prefix of the four `wall_*` groups |
| `classification.big_roof_is_ground_m2` | A flat roof on a slab bigger than 100 × 100 m whose area exceeds this is terrain (`wall_ground_land`), not a building |

The numbers the schema does not name — the points' boxes at ±40/±150 m, the roof
boxes at ±60/±250 m plus the −700 m downwind arm, the growth distances 40/700 m,
the structure-field distances 0–80/120 m, the ground scan at −1..60 m in 0.25 m,
the 0.5 m hull slack, the big roof's 100 × 100 m shape test — are the reference
script's hard-coded values, ported unchanged. That field layout is what this
site's mesh was tuned with.

#### Running it

First a scaled-down pass:

```powershell
python tools\mesh\step_mesh.py step.json --dry-run
```

It stops after the import and the cut and prints volumes, masses, surface
counts and the ground heights under the points (also written to
`work/dry_run_summary.json`). Minutes on this site; most geometry refusals end
here.

Then the real run, visible in its own console:

```powershell
.\tools\mesh\run_step_mesh.cmd step.json
```

That .cmd runs step_mesh.py in the current console — the stage banners print as
they happen — copies every line to `<out_dir>\work\run.log` as well, and passes
step_mesh.py's exit code straight through.

For iterating on sizes there is the checkpoint:

```powershell
python tools\mesh\step_mesh.py step.json --stop-after-checkpoint   # cut + pools
python tools\mesh\step_mesh.py step.json --from-checkpoint --tag coarse
```

`--stop-after-checkpoint` writes the checkpoint (`work/<name>_pools.brep` +
`.json`) and stops. `--from-checkpoint` starts from it — reads the
metre-space checkpoint, drops loose surfaces, takes the ground heights and the
pool outcomes from the `.json` — so a repeat that only changes `sizes` or
`post` skips the cut and the pools (about 15 minutes on this site) and re-meshes
from the trim in minutes. It refuses when the config's `points` differ from the
checkpoint's — the ground heights and the pools belong to it. `--tag NAME`
suffixes the outputs to `<name>_<NAME>.msh` and so on, keeping runs apart.

The run is ten stages, each announcing itself with a banner shaped like
`========== [ 7/10] trim  (elapsed ...) ==========` and the elapsed seconds;
every line is prefixed with the elapsed seconds too:

1. **import** — the STEP in at `scale` (`Geometry.OCCScaling`), the fluid solid
   found by tag or as the largest, its mass reported
2. **cut** — per-solid repairs, the sink stretch, the optional fuse, one boolean
   cut, sealed pockets dropped and listed
3. **ground heights** — per point, an isInside scan from z = −1 in 0.25 m steps
4. **classification** — every boundary face into exactly one patch (reported
   before the pools and again after the trim)
5. **pool discs** — one disc per point at its ground height; when one fails the
   run restores the post-cut checkpoint and continues without that pool
6. **checkpoint** — writes `work/<name>_pools.brep` + `.json`
   (`--stop-after-checkpoint` ends here)
7. **trim** — unless the config's trim is null, cuts everything below
   `below_z` with a box and reports the new flat faces' count and area, and how
   many faces reach below the trim plane
8. **groups + fields** — physical groups and the size fields (point boxes,
   growth thresholds, near/far structure fields, roof boxes), combined with a Min
9. **mesh** — `generate(1)`, coincident-curve families unified, `generate(2)`,
   faces with overlapping triangles remeshed (5, then 1), `generate(3)`, the
   gmsh optimiser
10. **post + write** — flat-tet removal with quality before and after, the three
    output files

Measured on this site:

| Stage | Time |
|---|---|
| import | 12 s |
| ship repair (repairs) | 60 s |
| cut | 3 min |
| pool discs (six points) | 9–11 min |
| trim | 1.5 min |
| 2-D meshing | 20 s |
| 3-D meshing | 3 min |
| post + write | 2 min |
| **Total (about 3.6 M tets)** | **about 23 min** |

The outputs land in `out_dir`:

```
mesh/
  nh3_site.msh              Gmsh 4.1 ASCII, physical groups = the patches
  nh3_site.vtk              binary, for viewing only
  nh3_site_summary.json     counts, groups, quality, timings, the whole config
  work/
    nh3_site_cut.brep       just after the cut (the pool-failure restore point)
    nh3_site_pools.brep     the checkpoint (+ .json) — where --from-checkpoint starts
    nh3_site_trimmed.brep   just after the trim
    repaired_33.brep        the ship-repair cache
    run.log                 every line, when run through run_step_mesh.cmd
```

`_summary.json` holds the gmsh version, the per-stage `timings_s`, the fluid
tag and mass, the solid bounding boxes, the repair replacements, the dropped
pockets, the ground height per point, the pool outcomes (`imprinted` / `not
imprinted` / `failed`), each patch's surface count and area (`groups`), the mass
the trim removed, the faces remeshed for overlaps, the triangle and tetrahedron
counts, the quality (before and after the flat-tet stage), the flat-tet notes,
the file sizes in bytes, and the whole config.

Three exit codes: **0** success (including `--dry-run` and
`--stop-after-checkpoint`), **1** refusal — an unknown config key, a missing
fluid tag, a cut mass that disagrees, an empty outer patch, a classification
that fails to cover the boundary — with a message that names the problem, and
**3** the 3-D pass came out empty — in which case the surface mesh is saved to
`work/surface_only.msh` and the summary to `work/failed_summary.json`, so the
PLC error's point can be inspected.

#### Reading the result

The patch names come from the classification: `top`/`west`/`east`/`south`/
`north` (five faces of the domain box), `wall_sea_surface` (the flat faces at
`sea_z`), `wall_ship_hull` (within 0.5 m of a repaired solid's bbox),
`wall_buildings` (faces inside a solid's bbox), `wall_ground_land` (everything
else, plus the flat roofs of very large slabs), `pool_<point name>` (the
disc-sized pieces at a point's ground height), and one patch per `roof_patches`
name (`nh3_source` here). `_summary.json`'s `groups` has each patch's surface
count and area — that is where you see a patch that came out empty or
suspiciously wide.

Quality prints twice, after the 3-D pass and again after the flat-tet stage,
in two measures:

```
  minSICN min 0.2999  p1 0.xxx  p5 0.xxx  p50 0.xxx  <0.1: 0  <0: 0
```

`minSICN` is the minimum signed inverse condition number: 1 is a regular
tetrahedron, 0 is collapsed, negative is inverted. `p1`/`p5`/`p50` are
percentiles, `<0.1` the count of tets below 0.1, `<0` the count of negative
volumes. **A `<0` above zero means the mesh must not reach a solver.** (The
selftest's little box lands near 0.2999.) `gamma` prints as a line of the same
shape alongside.

The post notes, in `_summary.json`'s `flat_tets_notes`, record how many flat
tets were found and removed by re-triangulating the boundary under them, how
many seam node pairs were merged and how many tets and triangles collapsed with
them, how many sliver edges were collapsed and how many collapses were reverted
(one that would invert a tet is refused), and how many thin tets were pushed.

#### To the solver

```powershell
ofgpu-convert-mesh mesh\nh3_site.msh nh3_case -type pool_tank_shell=wall
```

writes the `constant/polyMesh` of a case directory from one .msh (format (b)
above). A patch's type follows the prefix of its name — starting with `wall`
gives `wall`, `empty` gives `empty`, `symmetry` gives `symmetry` (all
case-insensitive), anything else keeps `patch`. The NAME stays exactly as Gmsh
wrote it; only the type changes. The `pool_*` patches sit on the ground, so for
the solver they are walls — wall treatment is picked from the TYPE, so a pool
left as a plain `patch` would never see a wall function. `-type name=type`,
repeatable per patch, wins over the prefix convention; a value that means
nothing is refused with the accepted list (`patch`, `wall`, `mappedWall`,
`empty`, `symmetry`, `symmetryPlane`, `wedge`, `cyclic`, `cyclicAMI`,
`cyclicSlip`, `processor`, `processorCyclic`).

Two more refusals worth knowing: a target directory that already holds a
`constant/polyMesh` is refused rather than overwritten (it may be the only copy
some pre-processing chain produced), and `-fluent` output covers tetrahedral
meshes only — a quadrilateral face, or a cell with a face count other than
four, is refused by name.

**Check the mesh with a one-iteration run first.**

```powershell
ofgpu-buoyant nh3_case -iters 1
```

Before it touches any field, the loader prints the mesh statistics — the
`mesh: <cells> cells, ...` line, the patch table (name, type, face count),
`volume: total ..., min ..., max ...`, `non-orthogonality: max ... deg, mean
... deg`, `face closure`, `lduAddressing: upper-triangular`. Those lines are
this mesh's report card. A freshly converted case has no `0/`, so what follows
is:

```
error: no time directory with initial fields found in nh3_case
```

**That refusal is the expected, healthy outcome** — the mesh loaded and its
structure checked out. If the volume or the non-orthogonality worries you, see
the troubleshooting table below; if it reads well, fill in `0/`,
`constant/` and `system/`. The shape of a case is [§4](#4-the-shape-of-a-case),
the field and dictionary details are in
[`cases/README.md`](../cases/README.md), and which driver solves what is
[§6](#6-choosing-a-solver--where-people-go-wrong) — this site is a buoyant
case, so it goes to `ofgpu-buoyant`.

#### To Fluent

```powershell
ofgpu-convert-mesh mesh\nh3_site.msh nh3_case -fluent nh3_site_fluent.msh
```

`-fluent` writes the SAME fixed-up mesh as an ASCII mesh ANSYS Fluent reads. The
format is Appendix B "Mesh File Format", B.3.7 "Faces" of the ANSYS FLUENT 12.0
User's Guide (public mirror:
[afs.enea.it/project/neptunius/docs/fluent/html/ug/node1471.htm](http://afs.enea.it/project/neptunius/docs/fluent/html/ug/node1471.htm)).
The orientation rule is that document's too — "if you curl the fingers of your
right hand in the order of the nodes, your thumb will point toward c1" — and
every index is hexadecimal and 1-based.

Zone names follow the patch names. The defaults take the polyMesh type first (a
`wall` stays `wall`, `symmetry` stays `symmetry`), then the names this project
gives its inlets — `east`, a name starting with `inlet`, one ending with
`_source` become `velocity-inlet`, and everything else `pressure-outlet`.
Fluent re-zones after reading anyway, so a default only has to be a sane
starting point; `-fluentType name=zone` sets one outright and wins over the
defaults, and a zone outside `wall`, `velocity-inlet`, `pressure-inlet`,
`pressure-outlet`, `outflow`, `symmetry`, `interior` is refused. The zone ids
run 1 for the fluid cell zone, 2 for the interior face zone, and the patches
from 10 up in order.

In Fluent: File > Read > Mesh. Coordinates are treated as metres, and after
reading run Mesh > Check.

**A path caveat.** Fluent's Cortex fails on a folder path with Korean
characters in it — this site's working folder was one. Put the mesh file Fluent
will read on an ASCII-only path.

If the counterpart wants CGNS instead, the alternative is writing it straight
from the Gmsh step with `gmsh.write('x.cgns')` and opening it with File >
Import > CGNS.

**Checking it independently.** The file the converter wrote can be checked
without Fluent:

```powershell
python tools\mesh\tester_mesh.py 0.5
ofgpu-convert-mesh tester_tet.msh case -fluent tester_ours.msh
python tools\mesh\fluent_check.py tester_ours.msh tester_tet_geometry.json
```

`tester_mesh.py` builds a small known case (a hexahedral box with a tetrahedron
cut out of its middle) and `fluent_check.py` parses any Fluent ASCII mesh from
scratch and checks that the header counts match the bodies, that every cell
closes and has positive volume, that the total volume equals box minus
tetrahedron, that boundary faces have one cell and their normals point out of
the domain, which c0/c1 orientation convention the file actually follows
(measured, not assumed), and that every zone lies on the plane or solid it is
named after. On 2026-09-09 this tester was also compared against OpenFOAM's
foamMeshToFluent: both files hold the same cells, nodes, topology and volume,
and differ only in the orientation convention (the counterpart writes the
inverse of the manual's rule).

#### In the Studio

Both steps are also registered as Studio default tools. The server reads them
out of `gui/server/tools.defaults.json` when it starts, so nothing needs to be
created by hand.

- **`mesh_from_step`** — inputs: `config` (the path of the config JSON,
  required) and `extra` (optional extra CLI tokens, e.g. `--from-checkpoint`).
  It runs the same command as the console — `python <repo>\tools\mesh\step_mesh.py
  <config> <extra>`.
- **`mesh_to_fluent`** — inputs: `msh`, `case`, `fluent` (required) and
  `types` (optional `-type name=type` tokens). The same command again —
  `<OFGPU_BIN_DIR>\ofgpu-convert-mesh <msh> <case> -fluent <fluent> <types>`.

An optional field left empty contributes no token at all, and a value holding
whitespace becomes one argument per word. `<repo>` resolves to the repository
root, and `<OFGPU_BIN_DIR>` finds the binary the way the solvers are found
(`OFGPU_BIN_DIR` → `rust/target/release` → `rust/target/debug`).

Running goes through `custom_tool_run` and asks for approval like every other
tool. A user tool of the same name, registered with `custom_tool_create`,
**wins**: the merge puts the user's tools first and a default only fills a name
the user has not taken. Defaults are never written into the user's file
(`gui/config/custom-tools.json`).

One thing to know: command tools have a 60-second time cap. A full site run
(about 23 minutes) does not fit inside it — that belongs in a console through
`run_step_mesh.cmd` — and the Studio tools are for short stages such as
`--dry-run` and the conversions.

#### Troubleshooting — the failures met on this site

| Symptom | Cause | What to do |
|---|---|---|
| Choosing `algo3d: 10` (HXT) kills the whole process without warning | gmsh's unfinished Steiner-point path, hit during boundary recovery. It is a kill, not an exception, so nothing can catch it | Leave `algo3d: 1` (Delaunay). When another algorithm ends with zero tets the tool retries with Delaunay itself |
| Netgen's optimiser dies with an access violation | A defect of this gmsh build (4.14.1) | Netgen is not offered at all; use gmsh's own optimiser (`optimize_passes`) |
| `No elements in volume` — the 3-D pass drops every tet | Nodes were merged BEFORE the 3-D pass | Never merge before the 3-D pass. The seam merge is the post stage's `seam_merge_m`, after it (the working route, found 2026-09-08) |
| A self-intersecting solid (the ship's hull) yields no volume | The STEP itself carries faces that cross each other | `repairs` with `resample` — replace it with a closed proxy resampled on a `cell_m` grid. Reuse the cache (`work/repaired_<tag>.brep`) via `brep` to skip the repair |
| A building floats above the terrain | The STEP's base sits above the ground under it | Bury it with `solids.sink_m`. But keep bases off the trim plane — 2.0 lands them at exactly 3.0 and 1,814 faces reach below the trim, 1.5 lands none |
| Hairline gaps between buildings | Neighbouring solids stand a few centimetres apart in the STEP | Do not fatten the solids. Where neighbours touch or overlap, `solids.fuse: true` |
| The 3-D pass fails on overlapping facets (exit 3) | The price of fattening a solid to close a gap — two walls now occupy the same place | Undo the fatten. Bases are buried with `sink_m`, touching solids are merged with `fuse` |
| The sea surface is not where the brief says | The brief is wrong | Look at the geometry, not the brief — find the flat face with the right area. This site's sea is at z = +3 (the apparent sheet at z = 0 is a duplicate shell); trim/sea go five centimetres above it, at 3.05 |
| A pool comes out `not imprinted` | The ground at that point is not planar — the disc came back embedded in a slope | Pool discs assume planar ground. The run continues without the pool and the summary says so; move the point to flat ground if the pool matters |
| The GPU solver is sensitive to thin cells | Thin wedges between the hull and the sea plane, a roof and a slab | The loader's `volume: ... min ...` and `non-orthogonality: max ... deg` lines are the evidence — see the one-iteration check above. Try `post.thin_push_m` (e.g. 0.2). The solver-side work on this is a separate tranche |

---

## 6. Choosing a solver — where people go wrong

**This is the most important section in this document.**

A driver named after a turbulence model does not necessarily solve the flow.
`ofgpu-k-epsilon`, `ofgpu-k-omega` and `ofgpu-sa` solve the **turbulence
equations only**. They leave the velocity field **frozen**.

Run an external-aerodynamics case with one of those and:

- the streamlines come out straight
- the contour comes out one flat colour
- no `U` and no `p` are written

You are not looking at flow around the body. You are looking at the **initial
field**.

### What solves what

| Driver | Solves | Case format |
|---|---|---|
| `ofgpu-k-epsilon` | k, ε — **U frozen** | directory / JSONC |
| `ofgpu-k-omega` | k, ω — **U frozen** | directory |
| `ofgpu-sa` | ν̃ — **U frozen** | directory |
| `ofgpu-lowmach` | **U, p**, T, turbulence | directory / JSONC |
| `ofgpu-buoyant` | **U, p**, T, buoyancy | directory |
| `ofgpu-plume` | **U, p**, T, buoyant plume | directory |
| `ofgpu-vof` | **U, p**, α — two-phase free surface | directory |
| `ofgpu-cht` | multi-region conduction + conjugate natural convection | JSONC |
| `ofgpu-datacentre` | data-centre room (fans, tiles, humidity, metrics) | JSONC |

If you want a velocity field, **pick one of the bold ones.**

### Which model goes to which driver

The code decides which binary a model name means. `driver_for` in
`rust/src/bin/common/mod.rs` routes them exactly like this:

| Model | Driver |
|---|---|
| `kEpsilon`, `RealizableKE`, `RNGkEpsilon` | `ofgpu-k-epsilon` |
| `LaunderSharmaKE` | `ofgpu-buoyant` or `ofgpu-lowmach` |
| `kOmega` | `ofgpu-k-omega` |
| `kOmegaSST` | `ofgpu-buoyant` or `ofgpu-lowmach` |
| `kOmegaSSTLM`, `kOmegaSSTGamma` | `ofgpu-buoyant` or `ofgpu-lowmach` |

The last row is not `ofgpu-k-omega` for a reason: both models are SST with
extra equations bolted on, reachable through `build_coupled`, and running one
in `ofgpu-k-omega` would solve a transitional case fully turbulent from the
leading edge — the plausible converged wrong answer SPEC-LIT §13.4 exists to
stop.

`kOmegaSST` sits in the same row for a different reason: SST needs a wall
distance (SPEC-LIT §6.6). `KOmegaSst::new` takes it as a required argument,
`build_coupled` computes it before constructing the model, and
`ofgpu-k-omega` never computes one — so SST is reachable only through
`ofgpu-buoyant` and `ofgpu-lowmach`.

### When the turbulence-only drivers are the right tool

When you want the turbulence quantities on a prescribed velocity field:
validating the model itself, comparing wall functions, studying the turbulence
response to a given field. `plume.jsonc` is exactly that — two equations
converging on a frozen `U`.

### What a momentum driver needs

`ofgpu-lowmach` solves `U` and `p` together, and the low-Mach loop wants a `T`.
The mesh generator writes all three into `0/` itself — for `channel`, `cavity`,
`step`, `big` and the buoyant pair `plume`/`room`, on the cut-cell path too —
so a freshly generated case runs as it stands. The `k` and `epsilon`/`omega`
that a RANS model reads are filled in the same way.

Without them — a hand-written case missing a field, or a generated `damBreak`
whose `0/` holds only `alpha.water` and `p_rgh` — it refuses:

```
error: cases/racecar_case\0 has no p field;
       ofgpu-lowmach with kOmega solves U, p, T, k, omega
```

---

## 7. Running it, and reading convergence

### Common options

```
-iters N          iteration cap (it stops earlier if it converges)
-fixedIters N     exactly N, no convergence test
-check N          print residuals every N
-write NAME       result directory name
-noWrite          write nothing
-output LIST      result formats: foam, vtu, nvdb, vdb, usda
-permissive       downgrade unsupported-setting errors to warnings and
                  substitute a documented default
```

`-output` is a list of **formats**, not of fields. `-output "U,p"` is rejected.

### Reading the residual line

From `ofgpu-lowmach`:

```
iter    500  |U| res 0.0928755  |p| res 1.351e-01  contErr 3.39691e-05
             T [293.15, 293.15] K  rho [1.2041, 1.2041] kg/m3  p0 101325 Pa
```

- `|U| res`, `|p| res` — normalised residuals. In a steady run, falling is
  converging.
- `contErr` — the continuity error. **If this does not fall, the pressure solve
  has failed** and the result cannot be trusted however pretty the residuals are.
- The `T` and `rho` ranges are there so a physically impossible field is visible
  at a glance. `T [inf, -inf]` means the field is empty or broken.

A turbulence-only driver prints this instead:

```
   250  omega res 7.457e-09 (0)  k res 1.799e-03 (1)  max dk/k 9.999e-04
```

The bracket is how many cells hit a limiter that iteration; `max dk/k` is the
largest relative change.

### When it diverges

**It stops.** It does not quietly keep going. It names the iteration and says
what went non-finite:

```
[ofgpu-lowmach] a field went non-finite at step 0 - stopping
iter 0 ... T [inf, -inf] K ... *** NaN/Inf ***
error: solution diverged (NaN/Inf)
```

### The mesh warning

```
[ofgpu] mesh warning: maximum non-orthogonality 79.7 deg;
        the explicit correction will dominate and the solution may need
        extra non-orthogonal correctors
```

Normal on a cut-cell mesh — cut cells make large non-orthogonality. If
convergence suffers, raise the non-orthogonal corrector count in `fvSolution`.

---

## 8. Looking at results

### Output formats

| `-output` | What |
|---|---|
| `foam` | OpenFOAM ASCII — ParaView and `foamToVTK` read it directly |
| `vtu` | VTK unstructured grid |
| `nvdb` / `vdb` | NanoVDB / OpenVDB volume |
| `usda` | USD ASCII |

### The Studio 3D viewer

Slices, arbitrary planes, iso-surfaces, streamlines and vector glyphs, drawn on
the spot. WebGPU where the browser has it, quietly WebGL2 where it does not.

**It works on cut-cell meshes too.** An external-aerodynamics mesh is a block
with the body carved out of it, which the exact lattice detector reads as
unstructured. The viewer takes a per-axis histogram of the cell centres, tells
lattice sites (spikes) from cut centroids (noise), and **recovers the original
block**, marking the sites inside the body as holes:

```
structured block recovered from the cut-cell mesh:
128 x 128 x 128, 4,100 site(s) inside the body
```

A hole contributes nothing to interpolation. Probe inside the body and you get
**no data** rather than zero; a streamline stops at the surface.

---

## 9. The Studio

A desktop workbench: file explorer, editor, terminal, live residual chart, 3D
viewer, and an AI panel that drives all of it.

Ask in words and it reads the case file, finds what has to change, **shows you
the diff, and writes only after you approve**. Start a solver and the residuals
are drawn from the moment it takes the GPU.

It is currently an **open beta**, reachable with an email address alone. It
supports English and Korean, switchable in settings (English by default).

---

## 10. MCP — handing the solvers to your own assistant

The Studio is one client. The solvers also speak the Model Context Protocol, so
Claude Code, Claude Desktop, or anything else that speaks MCP can mesh a geometry
and solve a case on your GPU directly.

```powershell
claude mcp add ofgpu -- node mcp\server.mjs
```

Six tools (`ofgpu_probe`, `ofgpu_list_cases`, `ofgpu_generate_mesh`,
`ofgpu_solve`, `ofgpu_validate`, `ofgpu_read_case_file`), four of them read-only.
No shell; every path is resolved against one workspace directory and refused
outside it; there is no delete tool at all.

Full guide: [`mcp/README.md`](../mcp/README.md).

---

## 11. Verification and grounds

Three documents hold each other up.

- **[`rust/SPEC-LIT.md`](../rust/SPEC-LIT.md)** — every discretisation specified
  with its source citation. That document is the authority, not the code.
- **[`rust/PROVENANCE.md`](../rust/PROVENANCE.md)** — what was read to write each
  file.
- **`ofgpu-validate`** — manufactured solutions, analytic solutions, published
  benchmarks.

```powershell
ofgpu-validate
```

Every run prints how many checks passed and then **names the gates that miss and
the gates that are open**. That list is not maintained by hand: it is generated
from a registry a gate enters at the very point where it reports its own verdict
(SPEC-LIT §69). Printing the verdict and registering it are the same call, so a
new gate cannot fall off the list.

The current counts and the MISSES/OPEN list live in the Status section of
[`README.md`](../README.md), and those numbers are what that tree actually
produced when run.

### Every check, one by one — what kind of verification it is, and what it was compared against

The table below takes every section `ofgpu-validate` prints (the `=== … ===` headings) and
records three things for it: **what kind of verification** the section is, **what its verdict
was compared against** (a paper's analytic solution, published benchmark numbers, a
correlation, public data, or a measurement this repository recorded), and the **address** of
that material (a DOI or URL; for older literature whose bibliographic record carries no DOI,
the citation alone with "no DOI"). The full bibliography, including which sources were not
read, is the "References" section of [`README.md`](../README.en.md); the citation behind each
equation is in [`rust/SPEC-LIT.md`](../rust/SPEC-LIT.md).

The kinds of verification are five. **Identity** (a closed form the discretisation must
satisfy exactly — no external material, SPEC-LIT itself is the ground), **reference
transcription** (agreement between `ofgpu::reference`, an independent host transcription of
SPEC-LIT §3 written as scatter loops, and the device's gathers), **manufactured/analytic
solution** (the observed convergence order of an MMS, the error against a paper's closed
form), **published benchmark/correlation/data** (a comparison against published numbers,
correlations or public data sets — the only external evidence this project claims), and
**recorded replay** (a verdict re-taken on numbers measured earlier and recorded in
`docs/07-lowmach-solver.md` rather than on this run — 45 of the 833). No item is compared
with the output of another CFD code (SPEC-LIT §0 rule 4).

| `ofgpu-validate` section | What kind of check | What it was compared against (the material used) | Address |
|---|---|---|---|
| `3-D graded block`, `3-D sheared block (non-orthogonal)`, `2-D block with empty front and back` — mesh identities | Identities: cell closure \|ΣS_f\|/V^(2/3), volume = analytic volume, every face separates its two cell centres, interpolation weights ∈ [0,1], upper-triangular lduAddressing (SPEC-LIT §10) | The definitions of SPEC-LIT §2–§3 themselves (no external material). Their sources: Jasak (1996) PhD thesis ch. 3; Moukalled, Mangani & Darwish (2016) | Jasak: <http://hdl.handle.net/10044/1/8335>; Moukalled et al.: Springer, DOI 10.1007/978-3-319-16874-6 |
| same sections — explicit operators | Identity + reference transcription: `fvc::grad` of a linear field = analytic, divergence of a uniform flux = 0, Laplacian order (SPEC-LIT §10 "Gradient", "Divergence", "Laplacian order") | Analytic identities (linear field, uniform flux) and `ofgpu::reference` (the host transcription of SPEC-LIT §3); the discrete Gauss theorem after Jasak (1996) ch. 3 | as above |
| same sections — skew correction | Reference transcription: the §74.4 correction on skewed faces, device gather against host scatter, and that the setting changes the answer | `ofgpu::reference`; the correction after Jasak (1996) §3.3 and Ferziger & Perić (2002) | as above; Ferziger & Perić: Springer, DOI 10.1007/978-3-642-56026-2 |
| same sections — implicit assembly | Identities: diagonal, off-diagonals, relaxation, boundary folding and Amul against the reference, three convection schemes (SPEC-LIT §10) | `ofgpu::reference`; relaxation and boundary folding after Patankar (1980) ch. 4–6; the face interpolation of Rhie & Chow (1983) | Patankar: Hemisphere, ISBN 0-89116-522-3 (no DOI); Rhie & Chow: *AIAA J.* 21, 1525, DOI 10.2514/3.8284 |
| `3-D block with 2:1 refinement interfaces`, `the adapt: refine, coarsen, and what a rebuild costs` | Identities: closure and assembly on a mesh with 2:1 interfaces, integrals preserved through refine/coarsen, the cost of a rebuild (SPEC-LIT §82–84) | SPEC-LIT §82–84 themselves (integral preservation is exact by construction); interface treatment after Jasak (1996) and Moukalled et al. (2016) | as above |
| `linear solvers` | Analytic/direct comparison: the Krylov solve against a dense direct solve (Gaussian elimination), the cuFFT direct Poisson solve against the iterative solve of the same matrix (SPEC-LIT §10 "Solver", "FFT Poisson") | The direct solve is the reference (no external material). The methods' sources: Hestenes & Stiefel (1952) CG, van der Vorst (1992) BiCGStab, Saad (2003), Swarztrauber (1977) FFT Poisson | Hestenes & Stiefel: DOI 10.6028/jres.049.044; van der Vorst: DOI 10.1137/0913035; Saad: DOI 10.1137/1.9780898718003; Swarztrauber: *SIAM Rev.* 19, 490, DOI 10.1137/1019071 |
| `method of manufactured solutions, -lap(psi) = f` | Manufactured solution: one refinement on 3-D graded / 3-D sheared / 2-D empty, reporting the observed order log₂(e_coarse/e_fine) | The MMS methodology: Roache (1998) *Verification and Validation in Computational Science and Engineering*, Hermosa; Roache (2002) *J. Fluids Eng.* 124, 4 | Roache 1998: ISBN 0-913478-08-3 (no DOI); Roache 2002: DOI 10.1115/1.1436090 |
| `buoyancy` | Identities: the arithmetic of b = g(T_ref/T − 1), the hydrostatic balance, the sign of buoyancy (SPEC-LIT §9, §10 "Hydrostatic", "Buoyancy sign") | SPEC-LIT §9 itself. The formulation's sources: Rehm & Baum (1978) *J. Res. NBS* 83, 297; Spiegel & Veronis (1960) for the ΔT/T ≪ 1 condition | Rehm & Baum: NIST J. Res. archive <https://nvlpubs.nist.gov/nistpubs/jres/>; Spiegel & Veronis: *Astrophys. J.* 131, 442 (no DOI) |
| `buoyancy production, sources, species, phi I/O` | Identities: the sign of G_b in stable/unstable stratification, a volumetric heat source raises the enthalpy by exactly P, species sum = 1 and within [0,1], a written-and-reread `phi` gives the same first pressure residual bit for bit (SPEC-LIT §17, §22) | The definition of G_b: Rodi (1987) *J. Geophys. Res.* 92, 5305; Henkes, van der Vlugt & Hoogendoorn (1991) *IJHMT* 34, 377. The rest are SPEC-LIT §22's own identities | Rodi: DOI 10.1029/JC092iC05p05305; Henkes et al.: DOI 10.1016/0017-9310(91)90258-G |
| `volume of fluid (SPEC-LIT 20, the 22 rows)` | Analytic + conservation: 20.1 a translating interface does not smear, 20.2 Zalesak's rotating slotted disc, 20.3 mass flux consistent with the advected density, 20.4 the curvature of a circle and its Laplace pressure jump, 20.5 two stratified fluids at rest (p_rgh) | Zalesak (1979) *J. Comput. Phys.* 31, 335 (the disc problem); Brackbill, Kothe & Zemach (1992) *J. Comput. Phys.* 100, 335 (CSF curvature, Laplace jump); Hirt & Nichols (1981) *J. Comput. Phys.* 39, 201 (VOF); Ubbink (1997) and Rusche (2002) PhD theses (interface compression) | Zalesak: DOI 10.1016/0021-9991(79)90051-2; Brackbill et al.: DOI 10.1016/0021-9991(92)90240-Y; Hirt & Nichols: DOI 10.1016/0021-9991(81)90145-5; Ubbink, Rusche: Imperial College theses (no DOI) |
| `msh hex closure, cut-cell closure (SPEC-LIT 23, 24)` | Identities: a Gmsh 4.1 hexahedron read through the real `parse_msh` closes, the cut faces of cells carved from an STL close by construction | The public file-format specifications: the Gmsh MSH 4.1 format document, the STL format. Closure itself is SPEC-LIT §23–24's definition | Gmsh MSH: <https://gmsh.info/doc/texinfo/gmsh.html#MSH-file-format> |
| `the low-Mach reference pressure (SPEC-LIT 25)` | Analytic: a heater of power P in a sealed box raises the pressure at exactly dp₀/dt = (γ−1)P/V, and p₀ does not move in an open domain | The low-Mach formulation: Rehm & Baum (1978); Majda & Sethian (1985) *Combust. Sci. Technol.* 42, 185; NIST *FDS Technical Reference Guide* (SP 1018-1, public domain) | Rehm & Baum: as above; FDS Technical Reference Guide: <https://pages.nist.gov/fds-smv/manuals.html> |
| `wall treatment: Ks -> 0, the thermal wall function (SPEC-LIT 29)` | Identities: the rough-wall law reproduces the smooth wall at K_s = 0 to round-off (both wall-function families), Jayatilleke's P(Pr/Pr_t = 1) = 0, the one-cell conductance equals the analytic heat flux | Jayatilleke (1969) *Prog. Heat Mass Transfer* 1, 193 (the thermal wall function's P); Cebeci & Bradshaw (1977) (rough-wall constants, Nikuradse's sand-grain data); Spalding (1961) *J. Appl. Mech.* 28, 455 | Jayatilleke: Pergamon (no DOI); Cebeci & Bradshaw: Hemisphere (no DOI); Spalding: DOI 10.1115/1.3641728 |
| `Werner-Wengle, coupled-solver turbulence selection (SPEC-LIT 30)` | Identities: the two branches of the WW wall model are continuous, a manufactured τ_w is inverted to round-off; `kOmegaSST` in a buoyant driver really builds SST | Werner & Wengle (1991) *8th Symp. Turbulent Shear Flows* (the power-law wall model); Menter (1994) *AIAA J.* 32, 1598 and Menter, Kuntz & Langtry (2003) (SST) | Werner & Wengle: Springer *Turbulent Shear Flows 8*, DOI 10.1007/978-3-642-77674-8_12; Menter 1994: DOI 10.2514/3.12149 |
| `periodic domains: cyclic-pair invariants (SPEC-LIT 31.1)` | Identities: the geometric matching of a cyclic pair | SPEC-LIT §31.1's own definition (no external material) | — |
| `the thermal wall-function gate, redesigned (SPEC-LIT 32)` | Identity + correlation + live + replay: the `flux_to_grad` identity; Dittus–Boelter and Gnielinski within each other's ±20–25 % (Re 1.6e4, Pr 0.71); the realised friction factor against the force balance and laminar Poiseuille f·Re (live); the wall-function verdict replayed — §32.4's **three OPEN verdicts** live here | Dittus & Boelter (1930, reprinted 1985) Nu correlation; Gnielinski (1976) *Int. Chem. Eng.* 16, 359 (±10 % — all three verdicts are held against this correlation); Petukhov (1970) smooth-pipe f; the record: `docs/07-lowmach-solver.md` §1.1, case `cases/channelPeriodicFluxWF.jsonc` | Dittus–Boelter reprint: *Int. Commun. Heat Mass Transfer* 12 (1985) 3, DOI 10.1016/0735-1933(85)90003-X; Gnielinski: no DOI; Petukhov: *Adv. Heat Transfer* 6, 503, DOI 10.1016/S0065-2717(08)70153-9 |
| `Launder-Sharma low-Re k-epsilon: damping functions (SPEC-LIT 33.3)` | Identities: the Re_t → ∞ / 0 limits of f_μ and f₂, monotonicity, reduction to the standard model | Launder & Sharma (1974) *Lett. Heat Mass Transfer* 1, 131; the standard coefficients of Launder & Spalding (1974) *CMAME* 3, 269; the low-Re review of Patel, Rodi & Scheuerer (1985) *AIAA J.* 23, 1308 | Launder & Sharma: DOI 10.1016/0094-4548(74)90150-7; Launder & Spalding: DOI 10.1016/0045-7825(74)90029-2; Patel et al.: DOI 10.2514/3.9086 |
| `resolved leg mesh resolution, replayed (SPEC-LIT 33.2/34)` | Recorded replay: first-cell y⁺ = 0.00174 on the resolved mesh, 192 of 400 cells below y⁺ 20 | This repository's record: `docs/07-lowmach-solver.md` §1.1, case `cases/channelPeriodicFluxLowRe.jsonc`, verdict from `ofgpu::models::mesh_resolution_report` | in-repository |
| `the bulk-temperature thermostat (SPEC-LIT 35)` | Identity (live): sign and steady offset of the proportional controller | SPEC-LIT §35.1's proportional law itself (no external material) | — |
| `thermostat weighting: the decisive experiment, replayed (SPEC-LIT 35.3.2)` | Recorded replay: mass-flux weighting widens (T_w − T_b), lowers Nu, and more so on the resolved mesh — three predicted statements against four measurements | This repository's record: `docs/07-lowmach-solver.md` §1.1 (two meshes × two weightings) | in-repository |
| `bounded convection on momentum: the isolation, replayed (SPEC-LIT 3.1/32.5.5)` | Recorded replay: the `bounded` prefix and the convection scheme's order separated — dropping `bounded` closes the drag balance | This repository's record: `docs/07-lowmach-solver.md` §1.1 (four combinations on the resolved leg, three on the wall-function leg) | in-repository |
| `Kays-Crawford turbulent Prandtl number (SPEC-LIT 37.1/37.2)` + the experiment replay | Correlation arithmetic: 2·Pr_t∞ = 1.70 at Pe_t → 0 (inside Kays's 1.5–1.9 for air), 0.85 at Pe_t → ∞; replay: Nu falls on both legs and more on the resolved mesh | Kays (1994) *ASME J. Heat Transfer* 116, 284 (the Pr_t correlation and its rise towards the wall); the record in `docs/07-lowmach-solver.md` §1.1 | Kays: DOI 10.1115/1.2911398 |
| `realizable and RNG k-epsilon (SPEC-LIT 40, 41)` | Identities + live: realizability ⟨u_a u_a⟩ ≥ 0 with the GPU's C_μ (Shih's threshold λ_max k/ε = 3.7037 evaluated, not quoted), the closed forms the coefficients imply, the asymptotic Sk/ε of the homogeneous-shear ODE, the failure of a constant C_μ under strong strain | Shih, Liou, Shabbir, Yang & Zhu (1995) *Comput. Fluids* 24, 227 — read as NASA TM-106721; Yakhot et al. (1992) *Phys. Fluids A* 4, 1510 — read as ICASE 91-65; realizability after Reynolds AGARD-755 (1987) and Lumley (1978) | Shih et al.: <https://ntrs.nasa.gov/citations/19950005029>; Yakhot et al.: <https://ntrs.nasa.gov/citations/19910021152>; Lumley: *Adv. Appl. Mech.* 18, 123, DOI 10.1016/S0065-2156(08)70266-7 |
| `the output block, and fp16 voxels (SPEC-LIT 44, 45)` | Execution check: the `output` block resolved and refused where it must be, the writers actually writing, measuring and deleting files | SPEC-LIT §44–45 themselves; the format specifications: OpenFOAM ASCII fields, VTK, OpenVDB/NanoVDB (public formats) | — |
| `conjugate heat transfer (SPEC-LIT 46, 47, 48)` | Analytic + conservation + benchmark: a two-layer slab with contact resistance, the two free limits, the transient interface temperature, conservation; Gate 5 Kaminski & Prakash (1986) **MISSES** (−7.11 %), Gate 6 Qu & Mudawar (2002), Gate 7 Flageul et al. (2015) not run | Analytic solutions: Carslaw & Jaeger (1959) ch. I; contact conductance: Cooper, Mikic & Yovanovich (1969); partitioned stability: Meng et al. (2017), Henshaw & Chand (2009), Verstraete & Scholl (2016), Giles (1997); Gate 5's comparison values come **not** from Kaminski & Prakash (paywalled, not read) but from the **secondary source** Belazizia et al. (2012) | Kaminski & Prakash: DOI 10.1016/0017-9310(86)90017-7; Cooper et al.: DOI 10.1016/0017-9310(69)90011-8; Meng et al.: DOI 10.1016/j.jcp.2017.04.052; Henshaw & Chand: DOI 10.1016/j.jcp.2009.02.007; Verstraete & Scholl: DOI 10.1016/j.ijheatmasstransfer.2016.05.041; Giles: DOI 10.1002/(SICI)1097-0363(19970830)25:4<421::AID-FLD557>3.0.CO;2-J; Belazizia et al.: *Adv. Theor. Appl. Mech.* 5 (2012), no DOI |
| `the conjugate fluid/solid interface (SPEC-LIT 59, 60)` | Published benchmark: Gate 59-A de Vahl Davis (1983) square-cavity natural convection, Nu = 2.243 at Ra = 10⁴ (0.6 %); 59-B an exact identity; Gate 5 conjugate natural convection (above) | de Vahl Davis (1983) *Int. J. Numer. Methods Fluids* 3, 249 — the benchmark Nu of its Table I | de Vahl Davis: DOI 10.1002/fld.1650030305 |
| within the same section — §79 forced convection, Gate 6 | Published benchmark: the thermal resistances of Qu & Mudawar's (2002) micro-channel heat sink — the authors' Fig. 4(b)/4(c) **digitised** (Disclosure 1); the inlet/outlet resistance measurements of Kawano et al. (1998) | Qu & Mudawar (2002) *IJHMT* 45, 3973 (read in full from the authors' public copy); Kawano, Minakami, Iwasaki & Ishizuka (1998) *ASME HTD-361-3* 173 | Qu & Mudawar: DOI 10.1016/S0017-9310(02)00101-1; Kawano et al.: no DOI |
| `surface-to-surface radiation (SPEC-LIT 49, 50, 51)` | Analytic: view factors C-11 and C-14 (unobstructed), Shapiro's FACET obstructed configuration F₁₂ = 0.115621, closure at scale; grey infinite parallel plates, concentric bodies, the three-surface enclosure with a re-radiating wall, radiative equilibrium; determinism | Howell's catalog of configuration factors, entries C-11 and C-14; Shapiro (1983) FACET UCID-19887; Walton (2002) NISTIR 6925 (the obstructed integration); the radiosity closed forms of Modest (2013) ch. 5 and Hottel & Sarofim (1967) ch. 3, 5 | Howell: <https://www.thermalradiation.net/>; Shapiro: DOI 10.2172/5607653; Walton: <https://nvlpubs.nist.gov/nistpubs/Legacy/IR/nistir6925.pdf>; Modest: Academic Press, ISBN 978-0-12-386944-9 (no DOI) |
| `fan curves, porous jumps, psychrometrics, metrics (SPEC-LIT 52, 53, 54, 55)` | Closed forms + public data: the fan operating point (exact); **Gate 52-B compares directly against NIST FDS's `fan_test`/`qfan_test` inputs and published CSVs (public-domain data)**; series resistances and the tile flow split of a porous jump; ASHRAE's thirteen moist-air coefficients and the IAPWS boiling point; the RCI/RTI/SHI identities and the one external number that was reachable | The FDS verification set `Verification/HVAC/fan_test.fds`, `qfan_test.fds` and their CSVs; AMCA 210 / ASHRAE 51 (what a fan curve is); ASHRAE Handbook—Fundamentals (2021) ch. 1 (Hyland–Wexler coefficients), Gatley et al. (2008) dry-air molar mass; IAPWS; Ward (1964) Darcy–Forchheimer; Karki & Patankar (2006) tiles; Herrlin (2005, 2008) RCI/RTI; Sharma, Bash & Patel (2002) SHI/RHI | FDS HVAC cases: <https://github.com/firemodels/fds/tree/master/Verification/HVAC>; Gatley et al.: DOI 10.1080/10789669.2008.10391032; IAPWS: <http://www.iapws.org/>; Ward: DOI 10.1061/JYCEAJ.0001096; Karki & Patankar: DOI 10.1016/j.buildenv.2005.03.005; Herrlin 2005: <https://www.semanticscholar.org/paper/99b942df4aa448a1e06f77d36b48d5d52a40c6e0>; Sharma et al.: DOI 10.2514/6.2002-3091 |
| `Spalart-Allmaras, DES97/DDES/IDDES (SPEC-LIT 56, 57, 58)` | Closed forms + published numbers: the TMR's table of far-field ν_t/ν (six digits), the DES length-scale identities, an experiment run live; the TMR flat plate (§56.11) and the periodic hill of Fröhlich et al. (2005) (§57.12) are **not run** | The NASA/TMBWG Turbulence Modeling Resource page for SA (government documentation, quoted to the printed digit); Allmaras, Johnson & Spalart (2012) ICCFD7-1902; Spalart & Allmaras (1994); DES97 Spalart et al. (1997); DDES Spalart et al. (2006) *TCFD* 20, 181; for IDDES the open restatements Herr et al. (2023) arXiv:2301.07223 and Savino et al. (2026) arXiv:2603.08875 in place of Shur et al. (2008) (paywalled, not read) | TMR SA: <https://tmbwg.github.io/turbmodels/spalart.html>; Allmaras et al.: <https://www.iccfd.org/iccfd7/assets/pdf/papers/ICCFD7-1902_paper.pdf>; Spalart et al. 2006: DOI 10.1007/s00162-006-0015-0; Herr et al.: <https://arxiv.org/abs/2301.07223>; Savino et al.: <https://arxiv.org/abs/2603.08875> |
| `gamma-Re_theta transition (SPEC-LIT 88, 89)`, `the 2015 gamma transition model (SPEC-LIT 90)` | Closed forms + analytic: the TMR's expanded form of the correlation against the paper's nested one (1e-12), the Blasius momentum thickness ∫f'(1−f') = 0.664, max(Re_V)/Re_θ = 2.193, the λ_B reference −0.003434; Gates 88-T/90-T reproduce the T3A free-stream decay (Tu 3.3 %) and leave the onset location **OPEN** | Langtry & Menter (2009) *AIAA J.* 47, 2894; Menter, Smirnov, Liu & Avancha (2015) *FTC* 95, 583 (paywalled, not read — every digit from the TMR page); the NASA/TMBWG TMR page for Menter-γ-2015; the Blasius solution integrated by this repository itself | Langtry & Menter: DOI 10.2514/1.42362; Menter et al. 2015: DOI 10.1007/s10494-015-9622-4; TMR γ-2015: <https://tmbwg.github.io/turbmodels/menter_gamma_3eqn.html>; TMR index: <https://tmbwg.github.io/turbmodels/> |
| `Lagrangian parcels (SPEC-LIT 66)` | Analytic + determinism: 66-A terminal velocity against the analytic force balance at four time steps, 66-B a ballistic parcel's cell and straight-line position, 66-C two runs and a CUDA-graph replay bit-identical | Equation of motion: Maxey & Riley (1983), Crowe, Sommerfeld & Tsuji (1998); drag: Schiller & Naumann (1933, as compiled by Clift, Grace & Weber 1978); the parcel model: Dukowicz (1980) | Maxey & Riley: DOI 10.1063/1.864230; Dukowicz: DOI 10.1016/0021-9991(80)90087-X; Clift et al.: Academic Press, ISBN 0-12-176950-X (no DOI) |
| `the parcel sort and gather-shaped deposition (SPEC-LIT 67)` | Determinism: the sort and per-cell CSR canonicalisation change nothing, and a replay is bit-identical | PSI-CELL: Crowe, Sharma & Stock (1977); sort/scan algorithms: Satish, Harris & Garland (2009), Merrill & Grimshaw (2009), Blelloch (1990), Hillis & Steele (1986) — papers read, no implementation opened | Crowe et al.: DOI 10.1115/1.3448756; Satish et al.: DOI 10.1109/IPDPS.2009.5161005; Hillis & Steele: DOI 10.1145/7902.7903; Elghobashi (1994) coupling map: DOI 10.1007/BF00936835 |
| `two-way coupling of the dispersed phase (SPEC-LIT 68)` | Identities + experiment: 68-A what the parcels took from the gas is what the gas is given, momentum and energy, to round-off; 68-B no parcels = bit-unchanged; 68-C Theobald's (1981) ~90 hose streams **MISSES** (61.29 % of the measured range with the gas at rest; the drag-free bracket 198.65 %) | Theobald (1981) *Fire Safety J.* 4, 1–13 (nozzle design and hose-stream range experiments); the sensible-heat half of Ranz & Marshall's (1952) Nu correlation | Theobald: *Fire Safety J.* 4 (1981) 1 (no DOI in the bibliography); Ranz & Marshall: *Chem. Eng. Prog.* 48, 141 and 173 (no DOI) |
| `droplet heating and evaporation (SPEC-LIT 76)` | Analytic + correlation: the d² law (closed form), the settling temperature against the crate's own balance (round-off) and against ASHRAE's wet bulb, parcel mass conservation | Spalding (1953, 1963) B_M and Stefan flow; Godsave (1953); Abramzon & Sirignano (1989) B_T; Ranz & Marshall (1952); properties: Watson (1943), Marrero & Mason (1972), NIST Chemistry WebBook SRD 69; the review of Sazhin (2006) | Abramzon & Sirignano: DOI 10.1016/0017-9310(89)90043-4; Marrero & Mason: DOI 10.1063/1.3253094; NIST WebBook: <https://webbook.nist.gov/chemistry/>; Sazhin: DOI 10.1016/j.pecs.2005.11.001 |
| `the vapour into the gas (SPEC-LIT 77)` | Identities + correlation: 77-A mass lost = mass given, 77-B the energy ledger across the phase change, 77-C both halves of the divergence, 77-D ASHRAE's adiabatic-saturation temperature (spraying into a sealed adiabatic box) | The moist-air relations of the ASHRAE Handbook—Fundamentals (2021) ch. 1 (adiabatic saturation, wet bulb); the Lewis (1922) relation; the divergence term of Rehm & Baum (1978) | ASHRAE Handbook: <https://www.ashrae.org/technical-resources/ashrae-handbook> (paid); Lewis: *Trans. ASME* 44, 325 (no DOI) |
| `droplet-wall impact (SPEC-LIT 78)` | Identities + published criteria: 78-A the regime boundaries against the closed-form inverse of the published criterion at 10⁻¹² either side, 78-B the wall mass ledger bit by bit, 78-C a run that meets no wall bit-unchanged, 78-D the two published splash criteria disagree with each other (We × 4.78) — **OPEN** | Mundo, Sommerfeld & Tropea (1995) *IJMF* 21, 151 (K = Oh·Re^1.25, K_crit = 57.7, the default — the experimental data itself was not transcribed); Bai & Gosman (1995) SAE 950283 (the alternative threshold); the review of Yarin (2006); IAPWS R1-76 surface tension | Mundo et al.: DOI 10.1016/0301-9322(94)00069-V; Bai & Gosman: DOI 10.4271/950283; Yarin: DOI 10.1146/annurev.fluid.38.050304.092144; IAPWS R1-76: <http://www.iapws.org/> |
| §38.9 non-Newtonian channel (Gate 1 Herschel–Bulkley plane Poiseuille live, Gate 2 Buckingham–Reiner) | Analytic: the closed-form profile §38.9 derives, on two meshes and four power-law indices; Buckingham–Reiner equal to the integral of its own Bingham profile (the coefficients 1, −4/3, +1/3 verified by quadrature); monotone approach as Papanastasiou's m rises | Herschel & Bulkley (1926) *Kolloid-Z.* 39, 291; Papanastasiou (1987) *J. Rheol.* 31, 385 (the regularisation); Buckingham–Reiner after Chhabra & Richardson (2008) *Non-Newtonian Flow and Applied Rheology*, 2nd ed. | Herschel & Bulkley: DOI 10.1007/BF01432034; Papanastasiou: DOI 10.1122/1.549926; Chhabra & Richardson: Butterworth-Heinemann, ISBN 978-0-7506-8532-0 (no DOI) |
| §39.7 contact angle (Jurin, Hoffman, Tanner, Šikalo) | Analytic: the sign of Jurin's height h = 2σcosθ/(ρgR) (θ > 90° depresses, 90° gives zero), Hoffman's master curve through the Jiang–Oh–Slattery correlation, Tanner's R ∼ t^(1/10), the dynamic angles of Šikalo et al. | Jurin (1718) capillary rise / Washburn (1921) *Phys. Rev.* 17, 273; Hoffman (1975) *JCIS* 50, 228; Jiang, Oh & Slattery (1979) *JCIS* 69, 74; Tanner (1979) *J. Phys. D* 12, 1473; Šikalo et al. (2005) *Phys. Fluids* 17, 062103 | Washburn: DOI 10.1103/PhysRev.17.273; Hoffman: DOI 10.1016/0021-9797(75)90225-8; Jiang et al.: DOI 10.1016/0021-9797(79)90081-X; Tanner: DOI 10.1088/0022-3727/12/9/009; Šikalo et al.: DOI 10.1063/1.1928828 |
| `cargo test --release --bin ofgpu-validate -- --ignored` — Ghia lid-driven cavity, Re 100/400 | Published benchmark (`#[ignore]`d, minutes): 17 centreline points of u and v against Ghia, Ghia & Shin's (1982) Tables I and II within 2 % (with the paper's erratum at Re 400) | Ghia, Ghia & Shin (1982) *J. Comput. Phys.* 48, 387 — Tables I (u) and II (v) | Ghia et al.: DOI 10.1016/0021-9991(82)90058-4 |

**What the gates that miss and the gates that are open were compared against.** These are
the eight names at the end of every `ofgpu-validate` run.

| Gate | Verdict | Compared against | Address |
|---|---|---|---|
| §60.5 Gate 5 — Kaminski & Prakash (1986) conjugate natural convection | MISSES (−7.11 % at Kr = 0.1) | The primary paper is paywalled and was not read; the comparison is the table of the secondary source Belazizia et al. (2012) | Kaminski & Prakash: DOI 10.1016/0017-9310(86)90017-7; Belazizia et al.: no DOI |
| §68.12 Gate 68-C — Theobald (1981) hose streams | MISSES (61.29 %) | Theobald's (1981) ~90 hose-stream experiments, *Fire Safety J.* 4, 1–13 | no DOI (citation only) |
| §32.4 verdicts 1 and 2 — plane-channel Nu, three legs | OPEN | Gnielinski's (1976) correlation at ±10 % (a correlation, not a measurement), Petukhov's (1970) f; the record in `docs/07-lowmach-solver.md` §1.1 | Petukhov: DOI 10.1016/S0065-2717(08)70153-9; Gnielinski: no DOI |
| §88 Gate 88-T, §90 Gate 90-T — transition onset | OPEN | No measured onset Re_x for the T3A flat plate could be found to close the comparison; the free-stream decay matches the TMR's Tu 3.3 % | <https://tmbwg.github.io/turbmodels/> |
| 78-D — the two splash criteria disagree | OPEN | The thresholds of Mundo et al. (1995) and Bai & Gosman (1995) differ by We × 4.78 for the same droplet | DOI 10.1016/0301-9322(94)00069-V; DOI 10.4271/950283 |

The table was made from the `ofgpu-validate` output of 2026-09-09 (833/833, 37 sections)
and the citations of [`README.md`](../README.en.md) "References" and
[`rust/SPEC-LIT.md`](../rust/SPEC-LIT.md). Sources that were paywalled and not read (Kaminski
& Prakash, Menter et al. 2015, Shur et al. 2008) and the places where a secondary source
stood in are stated in the table as they are.

**A distinction that matters**: "everything `ofgpu-validate` runs passes" is not
the statement "this project reproduces every published benchmark it compares
against". Do not confuse the two. The gates that miss are printed, by name, with
their numbers.

And **no GPL source was consulted.** Every file declares it in its header, and it
is enforced by a test rather than by prose (`provenance_audit`).

---

## 12. What it cannot do

- **No MPI, no multi-GPU.** Decomposition, halos and distributed
  PCG/PBiCGStab are implemented and gated, but this runs in one process on one
  card. It links no communication library and publishes no strong-scaling
  numbers.
- **No compressible or transonic flow.** The density-weighted time derivative is
  used in VOF, but the pressure equation is incompressible.
- **No finite-rate (Arrhenius) chemistry** — no stiff ODE integrator, no
  reaction mechanisms.
- **Radiation is surface-to-surface only.** Grey diffuse exchange across a
  *transparent* medium; nothing absorbs, emits or scatters inside a volume.
- **Surface-to-surface radiation, species transport and Lagrangian sprays have
  no case format.** All three are specified and gated as a library API, but no
  driver binary reads them from a case file.
- **Adaptive refinement is wired to no solver.**
- **No tetrahedral-to-polyhedral dual-mesh conversion.** Reading a tet mesh
  and solving on it work (Gmsh MSH 4.1, and the face-based model already
  treats a read tet mesh as a general polyhedral mesh). What is missing is the
  `polyDualMesh`-style operation that merges the tetrahedra around each node
  into one polyhedron, and it only becomes a converter once the work starts at
  revisiting SPEC-LIT's assumption that every face is planar and every cell
  convex, and at deciding the boundary treatment of the dual — a boundary
  node's dual cell is open.
- **AMGX is off by default**, and the selector reports it explicitly as
  "unavailable" when it is off.
- **No claim that the DES family reproduces published separated-flow
  statistics.**

The two gates that miss, and the open ones, are named with their numbers in
[`README.md`](../README.md).

---

## 13. Troubleshooting

In roughly the order you will meet them.

### `non-manifold edge(s)` — cut-cell rejects the STL

Two parts are **sharing a face**. Cut-cell classification needs a closed manifold
and cannot tell inside from outside across a shared face. Make the parts
**overlap**. A part that bites slightly into its neighbour is always safer than
one that meets it exactly.

### `has no p field` — the momentum driver refuses

You no longer meet this on a generated case — the generator writes `0/p` and
`0/T` itself (`channel`, `cavity`, `step`, `big`, `plume`, `room`). The ways
left to meet it: a hand-written case missing the field, a momentum driver
pointed at a generated `damBreak` whose `0/` holds only `alpha.water` and
`p_rgh`, or a field file deleted or moved. Put the missing field in `0/`. See
[§6](#6-choosing-a-solver--where-people-go-wrong).

### Straight streamlines and a one-colour contour

You ran a turbulence-only driver. The velocity field is frozen and you are
looking at the initial field. Re-run with a momentum driver such as
`ofgpu-lowmach`.

### `NO_STRUCTURED_GRID` — the viewer refuses to slice

The block behind the cut-cell mesh was not recovered. Current versions do the
histogram recovery, so meeting this now means either (a) the mesh really is not
block-based, or (b) it is holier than the recovery will accept (more than ~15 %
of the sites). The second is a deliberate refusal: reading a mesh that merely
resembles a lattice as one would put cells in the wrong place.

### The viewer draws the case on its side

The case has no `constant/g` and its floor patch is not named `bottomWall`,
`floor` or `ground`. Rename the patch, or add a `constant/g` if the case really
needs gravity. **Do not add `g` just to rotate the viewer** — gravity enters the
equations and changes the solution.

### `-output "U,p"` is rejected

`-output` takes formats. Choose from `foam`, `vtu`, `nvdb`, `vdb`, `usda`.

### Results land in `0/` rather than `1/`

`startTime`, `endTime` and `writeControl` in `system/controlDict` decide that. A
steady driver writes the final state only, into whichever time directory those
settings name. Keep a copy of the initial fields elsewhere if you need them.

### Refused for an unsupported setting

Intended (SPEC-LIT §13.4): there is no silent substitution. Pass `-permissive` to
take documented defaults; it prints what it replaced with what.

### The non-orthogonality warning

Normal on a cut-cell mesh. If convergence suffers, raise the non-orthogonal
corrector count in `fvSolution`.

---

## 14. Licence

**Prosperity Public License 3.0.0, plus a licensor interpretation clause.**

Free: personal study and hobby use, educational institutions, universities and
the research institutes attached to them, government bodies, public safety,
health and environmental organisations, and charities.

Any other commercial use needs a paid licence after a **30-day trial** — counted
per company, not per person.

**Research is judged by its purpose, not by who owns the institute.** Research
whose purpose is firefighting, medicine, health, safety, disaster response or
environmental protection is free even when a government-funded research institute
carries it out. Technology development aimed at a specific product or at
transfer to industry — electric vehicles, rail and propulsion, aircraft engines,
**anything subject to a technology-fee levy** — is commercial use, government
institute or not.

Full text: [`LICENSE`](../LICENSE). Explanation:
[`LICENSING.md`](../LICENSING.md).

---

## Further reading

| Document | What |
|---|---|
| [`README.en.md`](../README.en.md) | Overview, status numbers, what it can and cannot do, bibliography |
| [`rust/SPEC-LIT.md`](../rust/SPEC-LIT.md) | Every discretisation, specified and cited |
| [`rust/PROVENANCE.md`](../rust/PROVENANCE.md) | Per-file provenance |
| [`cases/README.md`](../cases/README.md) | Case formats, wall-model presets, per-case records |
| [`cases/racecar.md`](../cases/racecar.md) | The race-car sample in detail |
| [`mcp/README.md`](../mcp/README.md) | The MCP server |
| [`docs/07-lowmach-solver.md`](07-lowmach-solver.md) | Design and diagnostics of the low-Mach solver |
| [`docs/01-model-catalog.md`](01-model-catalog.md) | Model catalogue |

---

Owned by Iterations Inc. · Collaboration and contribution: Meteor Simulation Inc.
· Contact: simul@msimul.com
