# ofgpu-automesher

## Purpose

`ofgpu-automesher` is this project's own mesher, specified in `rust/SPEC-LIT.md`
§92. It exists because the ammonia-terminal site - 466 solids over 2.5 km x
2.5 km x 200 m, meshed as Gmsh tetrahedra for §92.1 - came back not merely
inaccurate but *unsolvable*: sealed pockets under buildings (disjoint cell
regions with no path to an outlet) made the pressure equation singular, and
the run diverged inside 50 iterations. A mesher that cannot promise its mesh
is solvable is not worth running, so this one measures its own output against
the §92.3 gate before it will emit anything.

Tranche 1 implements the **hex-dominant path** (octree refinement,
castellation, snapping, layers) and is complete: one command takes a config
file to a `constant/polyMesh` a solver can be pointed at. The tetrahedral path
is §92.4 and the polyhedral path is §92.5; both are specified there and neither
is implemented.

## Usage

```text
ofgpu-automesher <config.json> [-stopAfter STAGE] [-tag NAME] [-check [<caseDir>]]
                               [-dryRun] [-schema]
```

- `ofgpu-automesher <config.json>` - the meshing path, SPEC-LIT §92.14. Reads
  and validates the config, loads and merges `input.surfaces[]`, requires a
  closed surface, runs the §92.2 stage-0 domain check, then runs the four
  stages of (92.55) in order - `octree`, `castellate`, `snap`, `layers` -
  printing a banner **before** each one and its elapsed seconds after, so a
  stage that takes forty minutes has said which stage it is. On success it
  writes `<case_dir>/constant/polyMesh` and `<case_dir>/<name>_summary.json`
  and exits 0. On a gate failure it prints §92.3's refusal - the gate, the cell
  ids, their centroids, the measured values - and exits 1 **having written
  neither file** (§92.14.4).
- `-stopAfter STAGE` - stop after `octree`, `castellate`, `snap` or `layers`
  and write what that stage returned. `features` is accepted and means `snap`:
  §92.12 folded §92.2's stage 5 into stage 4's own loop, so there is no point
  in the pipeline between them at which a mesh exists (§92.14.1). A stopped run
  is not a way to get an ungated mesh - every stage gates its own output before
  returning it.
- `-tag NAME` - give this run its own output: the case directory becomes
  `<output.case_dir>_<NAME>` and the mesh name `<output.name>_<NAME>`, so two
  variants of one config do not overwrite each other's `constant/polyMesh`.
  This is `tools/mesh/step_mesh.py --tag`'s convention.
- `-check [<caseDir>]` - §92.14.5: run the §92.3 gate on a polyMesh that
  ALREADY exists, with this config's `quality` thresholds, and write nothing.
  The directory defaults to the config's own (tag-adjusted) `output.case_dir`;
  a directory named after the flag overrides it. Every gate passed: the
  measured summary, exit 0. Any gate failed: the refusal, exit 1. This is how a
  mesh from this mesher, from `ofgpu-convert-mesh`, or from anywhere else is
  judged before a solver is pointed at it.
- `-dryRun` - everything up to and including the surface summary, then exit 0
  without attempting the meshing stages. What a config check wants before
  queuing a long run.
- `-schema` - the JSON Schema of the config on stdout, exit 0, generated from
  the same types that parse it. No config is read and every other argument is
  ignored.

### The files in this directory

| File | What it is |
|---|---|
| `run_automesher.cmd` | Runs the mesher in a visible console with every line also copied to `<case_dir>/work/run.log`. The long runs go through this. |
| `sections.py` | Draws xy / yz / zx sections of a written polyMesh - the actual face polygons, not a block sketch. How a mesh is looked at before a solver is pointed at it. |
| `examples/box_sphere.json` | The smallest config that exercises the whole pipeline: a 1 m sphere in a 10 m box. Seconds to run; the config to try a change against. |
| `examples/make_box_sphere_stl.py` | Writes `box_sphere.stl` - a generator rather than a checked-in binary asset. |
| `examples/nh3_site.json` | The ammonia-terminal site of §92.1, the mesher's own yardstick. |

A first run, end to end:

```bat
cargo build --release --bin ofgpu-automesher
cd tools\automesher\examples
python make_box_sphere_stl.py box_sphere.stl
..\..\..\rust\target\release\ofgpu-automesher box_sphere.json
..\..\..\rust\target\release\ofgpu-automesher box_sphere.json -check
python ..\sections.py box_sphere_case 5 5 5 sections
```

### From Claude Studio

`gui/server/tools.defaults.json` ships the mesher as the tool `automesh`
(`config`, plus an optional `extra` string of CLI tokens such as
`-stopAfter snap -tag try2`), beside `mesh_from_step` and `mesh_to_fluent`. It
resolves the binary the way the solver runs do, through `<OFGPU_BIN_DIR>`.

## The config schema

JSONC (comments and trailing commas allowed, unknown keys refused by name).
The worked examples are [`examples/box_sphere.json`](examples/box_sphere.json) -
a sphere in a box, seconds to run - and
[`examples/nh3_site.json`](examples/nh3_site.json) - the ammonia-terminal site.
Both carry comments on every block. Every key of
`AutomeshConfig` (`rust/src/automesher/mod.rs`):

| Key | Type | Default | What it does |
|---|---|---|---|
| `$schema` | string | - | Accepted and ignored; points at the `-schema` output. |
| `input.surfaces` | array | *(required)* | The STL files the mesher casts against (stages 3-4). |
| `input.surfaces[].path` | string | *(required)* | Path to one STL file. |
| `input.surfaces[].name` | string | `null` | Replaces that file's own solid (patch) names wholesale. |
| `domain.extent` | number[6] | *(required)* | `[xlo, xhi, ylo, yhi, zlo, zhi]` of the background block, metres. |
| `domain.base_size` | number | *(required)* | Target background cell size, metres; must be > 0. |
| `domain.grading` | number[3] | `[1, 1, 1]` | Per-axis one-sided expansion, last cell / first cell. |
| `refinement.levels` | array | `[]` | Per-patch distance bands; empty means no band-driven refinement. |
| `refinement.levels[].patch` | string | *(required)* | Patch (STL solid) name the bands apply to. |
| `refinement.levels[].bands` | array | *(required)* | `[{distance, level}, ...]` - within `distance` of the patch, at least `level` (eq. 92.1). |
| `refinement.levels[].feature_level` | integer | `0` | A leaf within its own longest edge of one of this patch's feature edges (eq. 92.34) refines to this level (eq. 92.37); 0 is no feature refinement. |
| `refinement.feature_angle_deg` | number | `30.0` | Dihedral angle past which a triangulation edge is a feature edge (eq. 92.2). |
| `refinement.max_level` | integer | `2` | The level cap `l(c)` is min'd with (eq. 92.1); 6 is the octree cap §74.2 states and `mesh::refined::build` enforces. |
| `castellation.keep_region` | string | `"largest"` | Which connected component of the fluid survives eq. (92.4): `"largest"` or `"seed"`. |
| `castellation.seed_point` | number[3] | `null` | The keep point `"seed"` needs; required then, ignored otherwise. |
| `castellation.min_faces` | integer | `4` | A kept cell with fewer faces than this is dropped - a hole in the addressing, not a control volume. |
| `snap.iterations` | integer | `30` | Snapping iterations, the `k` of eq. (92.5). |
| `snap.tolerance` | number | `1e-3` | The dead band and the convergence test of eq. (92.28), as a FRACTION of `domain.base_size`: a point within `tolerance * base_size` of the surface is on it, and the loop stops when no point moves further. |
| `snap.smoothing_passes` | integer | `3` | Laplacian passes over the displacement field, eq. (92.6). |
| `snap.smoothing` | number | `0.5` | The smoothing relaxation weight, in `[0, 1]`. |
| `snap.undo_limit` | integer | `4` | Halvings of a gate-breaking displacement before it is zeroed and the point pinned - eq. (92.7)'s undo. |
| `snap.max_area_ratio` | number | `4.0` | Eq. (92.32): a wall patch carrying more than this many times its own surface area is geometry the cells never resolved, and the run refuses rather than collapsing the cell that reached it. Must be >= 1. |
| `snap.feature_tolerance` | number | `0.5` | Eq. (92.38): a boundary point whose surface target lies within this fraction of `domain.base_size` of a feature edge is snapped onto the edge instead, and onto the corner it claims (92.39). Zero turns the attraction off. Must be >= 0. |
| `layers.patches` | string[] | `[]` | The patches layers are added to; empty means none. |
| `layers.n` | integer | `0` | Number of layers, eq. (92.9)'s `n`. Zero: no layers. |
| `layers.first_thickness` | number | `0.05` | The first layer's thickness, metres. |
| `layers.growth` | number | `1.3` | The expansion from one layer to the next; must be > 0. |
| `layers.min_thickness` | number | `0.1` | Total thickness below which the layers are dropped. |
| `layers.medial_frac` | number | `0.5` | Fraction of the local medial-axis distance below which the layers are dropped (eq. 92.10). |
| `layers.cell_frac` | number | `0.5` | Eq. (92.45): the fraction of the LOCAL CELL SIZE `h_i` the whole stack may take, where `h_i` is the shortest edge of the layer faces carrying point `i` - on a snapped mesh that is the CUT cell's edge, not the octree leaf's. This is usually the limiter that binds, so read the achieved first layer in metres off the run's `layers: patch ...` line rather than assuming `first_thickness` was delivered. |
| `layers.normal_passes` | integer | `3` | Eq. (92.41): smoothing passes over the point normals. Zero leaves them the area-weighted average of (92.40), which is what keeps a flat wall's prisms exact. |
| `layers.smoothing` | number | `0.5` | Eqs. (92.41) and (92.46)'s `w`. |
| `layers.smoothing_passes` | integer | `4` | Eq. (92.46): passes of the displacement into the interior. |
| `layers.retreat_limit` | integer | `4` | Eq. (92.47): how many times the thickness is halved - on the EXTRUDED mesh - before the patch loses its layers by name. |
| `quality.max_closure` | number | `1e-10` | G2 (92.12): `\|sum s Sf\| / V^(2/3)` stays under this - `mesh::geometry::CLOSURE_LIMIT`, the crate's own. |
| `quality.max_non_orth_deg` | number | `70.0` | G4 (92.13): max internal-face non-orthogonality, degrees. |
| `quality.report_non_orth_deg` | number | `60.0` | G4: faces past this are counted and reported, not refused. |
| `quality.min_thickness_ratio` | number | `0.05` | G5 (92.14): `tau_c = 3 V_c / A_max^(3/2)` stays at or above this. |
| `quality.max_cond` | number | `1e4` | G6 (92.15): `cond(T_c)` stays under this. |
| `output.case_dir` | string | *(required)* | The case directory the mesh is written to (`<case_dir>/constant/polyMesh`). |
| `output.name` | string | *(required)* | The mesh's name within the case. |
| `output.patch_names` | object | `{}` | Eq. (92.56): rename the final mesh's patches on the way out, `{"xMin": "west", ...}` - the octree's own six names and the STL's solid names become the names the case's boundary conditions are written against. A key that names no patch, a value the polyMesh reader would refuse, and two patches that would collide are all refused before anything is written. |

## The pipeline

SPEC-LIT §92.2's stages, and where each one lives. All of them are built;
§92.14 is the driver that runs them in order.

0. **The background block** (`automesher::octree::Background`) - a
   `blockgen::BlockSpec` over `domain.extent` with `base_size` as the target
   cell size and `domain.grading` per axis. The domain and the surface must
   hold each other with one `base_size` of margin, or the run refuses before
   any work is done, naming the axis and both numbers. A surface that spans
   PAST the domain on both sides of an axis - the site-solid setup - is the
   legitimate other case and is said in the log, not refused.
1. **Octree refinement** (`octree::refine_to_surface`, §92.9) - per-cell level
   from the distance bands (92.1), the feature edges (92.37) and the level cap
   (92.1). `-stopAfter octree` writes the leaf mesh itself: the six domain
   patches, no wall.
2. **2:1 balance** (`Octree::balance_2to1`, §74.2's fixed point, eq. 92.3) -
   inside stage 1, on §74's face conventions: the coarse cell owns four split
   faces, each a real polygon over real points.
3. **Castellation** (`castellate::castellate`, §92.10) - parity classification
   (92.23), the connected-component keep-set (92.4), the pinch rule (92.25),
   and the walls (92.26) named after the STL solid nearest each exposed face.
   The dropped components are reported by count and by bounding box: a mesher
   that silently deletes 900 cells under a building has told you nothing.
4. **Snapping** (`snap::snap`, §92.11) - boundary points to the closest surface
   point (92.5), the displacement field smoothed (92.6), and any displacement
   that breaks the §92.3 gate halved up to `snap.undo_limit` times and then
   abandoned (92.7). This is the stage that buys back the geometry
   castellation's staircase lost.
5. **Feature snapping** (§92.12) - **inside stage 4's loop**, not after it:
   (92.38) pulls a point whose surface target is near a feature edge onto the
   edge, and (92.39) onto a corner that claims it. `-stopAfter features` is
   therefore `-stopAfter snap`; there is no mesh between them.
6. **Layers** (`layers::add_layers`, §92.13) - the shrink, the extrusion, and
   the retreat ladder (92.47).

   **`layers.patches` is supported on a wall that CASTELLATES ONTO THE CELL
   PLANES.** On a snapped wall it is attempted and usually given up: stage 4
   always runs before stage 6, so the wall face the layer stage turns into an
   internal face carries whatever non-orthogonality the snap left on it, and
   §92.13's table measures a snapped sphere at 74-80 degrees against G4's 70 -
   at every surface resolution from 128 to 8192 triangles, at octree level 3,
   and at every thickness tried. A snapped axis-aligned box is not safe
   either: its convex edges tangle the stack (37 folded cells at one offset,
   an outright negative volume at another). When that happens the patch loses
   its layers BY NAME, the run returns the snapped mesh, and the summary says
   which patch and why - it does not refuse with a list of faces you cannot
   act on. Moving that line needs either a wall-face/owner-centre alignment
   step in stage 4 or a G4 rule of its own for a newly internalised wall face;
   both change numbers §92.3 fixes and neither is written.

## What a run writes

Two files, and only after every stage has returned (§92.14.4 - a refused run
writes nothing at all):

- `<case_dir>/constant/polyMesh` - `points`, `faces`, `owner`, `neighbour`,
  `boundary`, written by `io::polymesh::write_poly_mesh_raw`. **Real points**:
  the face lists are the actual polygons of the actual cells, not the synthetic
  quads the cut-cell writer emits, so `sections.py` can draw them and any
  polyMesh reader can read them.
- `<case_dir>/<name>_summary.json` - (92.57): the tool, the config path, the
  stage the run stopped after (or `null`), the surface's counts and bounding
  box, one row per stage with its wall-clock seconds and its own counts, the
  final mesh's counts and patch list, the §92.3 report gate by gate, the total
  seconds, and **the config as it parsed**, defaults filled in. A mesh on disk
  with no record of what made it is a mesh nobody can reproduce, and the
  per-stage times are the only input a plan for the next run has.

The patch names in `boundary` are the ones `output.patch_names` asked for
(92.56), applied inside the mesher just before the write - not by a script
afterwards, because a script that rewrites `boundary` is a second reader of the
one file the solver may not be allowed to guess at.

## The quality gate

SPEC-LIT §92.3's seven checks - the precondition every stage must hold, and
the report `-check` prints. The mesher **refuses rather than ship a bad mesh**:
a stage that cannot keep the mesh inside these ends the run, naming the gate,
the subject it failed on (a cell, or a face for G4 and G7) and its id, its
centroid to six figures, the measured value and the threshold - the per-gate
form §92.3 fixes, `tau = 0.011538, need >= 0.05` and not `value = ...`. The
header counts every failing subject even though only ten are listed.

| Gate | Check | Threshold | Config knob |
|---|---|---|---|
| G1 | positive volume, `V_c > 0` (92.11) | every cell, no knob | - |
| G2 | closure, `\|sum s Sf\| / V_c^(2/3)` (92.12) | `< 1e-10` | `quality.max_closure` |
| G3 | one cell region (`cell_regions`) | exactly 1, no knob | - |
| G4 | non-orthogonality, internal faces (92.13) | max `< 70 deg`; faces past 60 deg reported | `quality.max_non_orth_deg`, `quality.report_non_orth_deg` |
| G5 | thickness, `tau_c = 3 V_c / A_max^(3/2)` (92.14), `A_max` the largest PLANAR FACE GROUP (faces whose outward normals agree to within 5 deg, summed) so a split 2:1 face measures as the one face it is | `>= 0.05` | `quality.min_thickness_ratio` |
| G6 | gradient conditioning, `cond(T_c)` (92.15) | `< 1e4` | `quality.max_cond` |
| G7 | addressing (owner < neighbour, upper-triangular order, no duplicate faces) | pass/fail, no knob | - |

## Troubleshooting

The mesher's failures are refusals with a named cause, so start from the exact
message. The ones that actually happen:

| What you see | What it is | What to do |
|---|---|---|
| `domain.extent: on z the surface neither sits inside the domain with one base_size of margin nor spans past it on both sides` | Stage 0. The surface ENDS inside the domain, leaving an edge with no solid behind it for castellation to cut against. | Either pull `domain.extent` in so the surface spans past it on that axis (the site setup), or push it out so the surface clears it by `base_size` (the wind-tunnel setup). |
| `surface ... open edges` from `require_closed` | The STL is not closed, and (92.23)'s parity classification through a hole is a coin toss. | Repair the STL. `tools/mesh/step_mesh.py`'s geometry stage or any STL repair will do; the mesher will not guess. |
| `castellate: no cell survived the walk` | The keep-set (92.4) is empty: usually `castellation.seed_point` is inside the solid, or outside the domain. | Put the seed in open fluid - for a site, high and upwind, e.g. `[-1200, 0, 150]`. |
| a large `removed.off_region` and dropped boxes under buildings | Working as intended: those are the sealed pockets §92.1 exists for. | Read the boxes in the log. If one of them is a region you wanted, the seed is on the wrong side of a wall. |
| `snap: patch "X" carries N times its own surface area` (92.32) | The cells never resolved that geometry, and snapping would collapse the one cell that reached it. | Raise `refinement.max_level` or add a tighter distance band for that patch. Raising `snap.max_area_ratio` hides the problem instead of fixing it. |
| `layers: patch "X": ...` with `n_layers: 0` and a reason | The retreat ladder (92.47) gave up. On a SNAPPED wall this is the expected outcome - see the pipeline note above. | Read the reason. The run continued and the mesh is the snapped one; if you need the layers, the wall has to castellate onto the cell planes. |
| `layers: patch "X": 3 layer(s) ... first layer 4.0e-03 m of 4.0e-01 requested` | (92.45)'s cell-size limiter bound, not `first_thickness`. | This is the number to quote, not `first_thickness`. Refine the wall if you need a thicker stack. |
| gate G4 fails at 70-80 degrees on a snapped wall | (92.13). The snap traded a staircase for a slanted face. | Refine that patch, or relax `quality.max_non_orth_deg` **only** if the solver settings can carry it (`uncorrected` Laplacian, `nNonOrthogonalCorrectors 0`) - and say so in the case. |
| the run is silent for a long time | It is not: the banner of §92.14.1 is printed before each stage. If nothing new has appeared for an hour, the named stage is the one to look at. | `refinement.max_level` and the base grid together set the leaf count; halving `base_size` is 8x the work. Run `-dryRun` first and read the plan line. |

Sizing, from the site run: the plan line prints the base grid and the finest
cell before any work is done, and `-stopAfter octree` is the cheap way to learn
the leaf count a config will produce before paying for castellation.

## Licence note

Nothing GPL-licensed was consulted for §92 or for this unit: not
`snappyHexMesh`, not cfMesh, not TetGen, not CGAL's GPL modules, and not the
octree families p4est (GPL-2.0-or-later), libsc (LGPL-2.1) or t8code (GPL-2.0).
The specification's documentation sources are named exactly in
`rust/PROVENANCE.md`: CFD Direct's *OpenFOAM v12 User Guide* §5.5 ((c)
2015-2025 CFD Direct Ltd) and ESI-OpenCFD's *OpenFOAM User Guide* meshing
chapter ((c) OpenCFD Ltd, CC BY-NC-ND 4.0), read for the castellation /
snapping / layer-addition stage list, the distance-band shape and the
30-degree feature-angle default, with no text, figure or table reproduced;
the papers §92 cites
(Schneiders 1996; Ito, Shih & Soni 2009; Marechal 2009; Owen, Staten &
Sorensen 2011; Isaac, Burstedde & Ghattas 2012; Freitag & Ollivier-Gooch 1997;
Garimella & Shephard 2000), and Ericson's *Real-Time Collision Detection*
(a textbook, read as a textbook). The seven gate checks are this project's
own, stated in §92.3.
