# tools/mesh — STEP → Gmsh mesh, configuration-driven

`step_mesh.py` turns one STEP file into a solver-ready tetrahedral mesh from one
JSON config: geometry (which solid is the fluid, which are cut away, which are
replaced by a repaired proxy), sizing (pool, refinement boxes, near/far
structures), post-processing (flat-tet removal). It is the ammonia site mesh
pipeline — `nh3_site_mesh.py` plus the ship repair (route E) and the trim cut —
ported stage for stage, with every number that script hard-coded made a config
key. [`examples/nh3_site.json`](examples/nh3_site.json) is that site as a
config.

```
python tools/mesh/step_mesh.py <config.json> [--from-checkpoint] [--stop-after-checkpoint]
                                              [--tag NAME] [--dry-run]
```

writes `<out_dir>/<name>[_TAG].msh` (Gmsh 4.1 ASCII, physical groups = the
patches), `<name>[_TAG].vtk` (binary, for viewing) and
`<name>[_TAG]_summary.json` (counts, groups, quality, timings, the config),
and keeps its checkpoints in `<out_dir>/work/`. Then one command takes the mesh
to the solver and to Fluent:

```
rust/target/release/ofgpu-convert-mesh <out_dir>/<name>.msh <caseDir> -fluent <name>_fluent.msh
```

Every patch whose name starts with `wall` becomes type `wall` on conversion;
`-fluentType <patch>=<zone>` overrides a zone. Sealed cell pockets (whole regions
cut off from the rest of the mesh, the kind a mesher leaves under buildings) are
dropped by default and the `regions:` line says so; `-keepRegions` converts every
region instead.

On Windows, `run_step_mesh.cmd <config.json> [flags]` runs the tool in the
current console — the stage banners print as they happen — and copies every
line to `<out_dir>/work/run.log`.

`mesh_case.sh <case.json> [--from-checkpoint] [--no-geometry] [--no-fluent]` is
the whole chain for one case: `run_step_mesh.cmd`, the fluid solid of the meshed
domain exported as STEP (`<case dir>/geometry/fluid_<name>.step`, and
`_full.step` before the trim) as soon as the trim has run, then the converter
with every point's inlet patch typed `velocity-inlet` — `case/constant/polyMesh`,
`<name>_fluent.msh`, `convert.log` and `mesh.exit` in the case directory. Launch
it in its own console from Bash. `examples/pool_three_inlets.json` is a case with
three sources in one geometry (a disc, the ring around it, another disc) whose
patches are named after the points (`classification.pool_prefix` `""`);
`examples/pool_ring_case.json` the single-source form. The steady air test that
judges a mesh is `solver_test/` (`test_cases.sh <iters> <write> <caseDir>...`).

> The site playbook — what broke a 2.5 km site mesh and the recipe that fixed it (far-solid
> convex hulls, the gap rule, the thickness gate), the review gates and the diagnosis scripts —
> is `docs/08-site-mesh-playbook.md`; the step-by-step procedure for a session is the
> `site-mesh` skill (`.claude/skills/site-mesh/SKILL.md`). Diagnosis scripts live in `diag/`,
> the six-iteration steady air test in `solver_test/`, a full recipe in `examples/pool_ring_case.json`.

## The pipeline (10 stages, checkpointed)

| # | stage | what happens |
|---|-------|--------------|
| 1 | import | STEP in at `scale` (`Geometry.OCCScaling`, so `0.001` for mm STEP); the fluid solid found by tag or as the largest, its mass reported |
| 2 | cut | per-solid repairs, the sink stretch, optional fuse, one boolean cut, sealed pockets dropped and listed, near-touching solid pairs recorded (see `solids.touch_warn_m`) |
| 3 | ground heights | per point, an `isInside` scan from z = −1 in 0.25 m steps |
| 4 | classification | every boundary face into exactly one patch (see below) |
| 5 | pool discs | one disc per point at its ground height, imprinted one at a time; a failure restores the post-cut checkpoint and the run continues without that pool |
| 6 | checkpoint | `work/<name>_pools.brep` + `.json` (points, pools, solid bboxes, hull bboxes); `--stop-after-checkpoint` exits here |
| 7 | trim | optional: everything below `trim.below_z` cut away with a box, on the checkpoint model |
| 8 | groups + fields | physical groups; size fields (Min of the per-point boxes, growth thresholds, near/far structure distance fields, roof boxes) |
| 9 | mesh | generate(1); coincident-curve families unified; generate(2); faces with overlapping triangles remeshed (algorithms 5, then 1); generate(3); the gmsh optimiser |
| 10 | post + write | flat-tet removal, quality before/after, the three output files |

`--from-checkpoint` starts at stage 6's result (import at scale 1.0, drop loose
surfaces, ground heights and pools come from the `.json`), so a tweak of the
sizes or the post settings re-meshes in one run without redoing the boolean
work. `--dry-run` stops after the cut and prints volumes, masses, surface
counts and the ground heights. A 3-D mesh that comes out empty writes
`work/surface_only.msh` + `work/failed_summary.json` and exits **3**; every
refusal (config, geometry, classification) exits **1** with a named message.

The narrative manual for this workflow — every config key explained, the
ammonia site as a worked example with per-stage timings, the solver and Fluent
legs, the Studio tools, and the table of failures met on that site — is
[`docs/GUIDEBOOK.md`](../../docs/GUIDEBOOK.md) §5 "STEP 형상에서 격자로, 그리고
Fluent로" (English: [`docs/GUIDEBOOK.en.md`](../../docs/GUIDEBOOK.en.md) §5).
This file stays the terse reference.

## Patches

`top` `west` `east` `south` `north` (the `domain_box` faces, tolerance
`outer_tol`), `wall_sea_surface` (flat faces at `sea_z`), `wall_ship_hull`
(within 0.5 m of a repaired solid's bbox), `wall_buildings` (faces inside a
solid's bbox), `wall_ground_land` (the rest, plus the flat roofs of slabs
bigger than 100 × 100 m and `big_roof_is_ground_m2`), `pool_<point>` (flat
disc-sized pieces at a point's ground height — a point given as
`{"x", "y", "r", "r_inner"}` carries its own radius, and `r_inner > 0` makes
it a ring, so an inner disc and an outer ring at one centre are two patches
whose pieces are told apart by whether they fit inside the inner square), and one patch per
`roof_patches` name (a solid's flat roof). `wall_prefix` renames the four
`wall_*` groups; `pool_prefix` (default `pool_`) is what goes in front of a
point's name for its inlet patch — `""` names the patch after the point itself,
so points `inlet1`, `inlet2`, `inlet3` give patches `inlet1`… and sides
`wall_inlet1`… (the converter's Fluent default already makes a name that
starts with `inlet` a velocity-inlet).

## Config schema

Unknown keys are refused by name; missing keys take these defaults. `step`,
`out_dir` and `domain_box` are required.

```jsonc
{
  "step":       "path/to/geometry.step",  // required
  "scale":      0.001,                    // Geometry.OCCScaling, applied BEFORE the import (mm -> m)
  "out_dir":    "path",                   // required; work files go to out_dir/work
  "name":       "site",                   // prefixes every output and checkpoint file
  "fluid":      {"tag": 1},               // the fluid solid's tag, or {"largest": true}
  "domain_box": [xmin, ymin, zmin, xmax, ymax, zmax],  // required; the outer faces are classified against it
  "outer_tol":  0.05,                     // tolerance of the top/west/east/south/north tests
  "solids": {
    "base_below_sea_m": 0.0, // > 0: every base ends at least this far under sea_z (a solid over
                            // water would else leave a slab of fluid under its sunk base)
    "sink_m": 2.0,          // stretch every other solid this far below its base, about its
                            // roof (buildings must not float above the terrain)
    "fuse": false,          // fuse the solids before the cut (merges touching/overlapping ones)
    "exclude_tags": [],     // solids left out of the cut entirely (removed from the model)
    "touch_warn_m": 0.05,   // model points 1 mm..this far apart (0 disables) are logged as
                            // near-touching pairs and stored in the summary's "near_touching":
                            // razor-thin fluid gaps the mesher may treat as duplicate points
    "hull_beyond_m": 0,     // > 0: every solid whose bbox centre is farther than this from the
                            // nearest point is replaced by the prism of its footprint's convex
                            // hull (its own z range), pushed out by hull_pad_m, so neighbouring
                            // far buildings overlap and the cut merges their slits; repaired and
                            // excluded solids are left alone (see "Far-solid smear" below)
    "hull_pad_m": 1.0,
    "hull_snap_m": 0.05,    // prism corners of different solids closer than this become one
                            // point, so neighbours share their vertical edges exactly (two
                            // edges millimetres apart are "duplicate points" to the mesher and
                            // the 3-D boundary recovery fails on them); use with "fuse": true
    "hull_box_snap_m": 0,   // > 0: hull-prism corners closer than this to a domain x/y side
                            // (after the shrink) move 1 m outside it, so a wall meets the side
                            // squarely instead of leaving a wedge of fluid narrowing to nothing
    "hull_box_inset_m": 0,  // > 0: those corners move this far INSIDE the side instead, so no far
                            // wall touches an outlet at all - a wall-outlet junction, even a square
                            // one, was where the steady solution broke next
    "boolean_tol_m": 0       // > 0: fuse and cut run as fuzzy booleans (Geometry.ToleranceBoolean)
                            // merging entities closer than this; OCC's fuzzy fuse of hundreds of
                            // overlapping prisms failed at 5 cm on the site, snapping did not
  },
  "repairs": [                            // per solid, applied before the cut
    {"tag": 33, "method": "resample", "cell_m": 1.5, "target_faces": 6000,
     "lift_z": 3.05, "brep": "optional path to reuse"}
  ],
  "trim":  {"below_z": 3.05,              // or null: cut everything below this plane away
            "shrink_xy_m": 0},            // > 0: a margin cut off every x/y side after the z trim,
                                          // the outer patches moving inward with the box - the
                                          // terrain STEP's own edge is a staircase of centimetre
                                          // steps and hull prisms cross the sides at shallow angles
  "sea_z": 3.05,                          // flat faces at this height (± 0.06) -> wall_sea_surface
  "points": {"tank_shell": [-916.9, 349.8],   // refinement/pool points; ground found by isInside
             "qcdc_inner": {"x": 18.5, "y": -7.5, "r": 10.5, "h": 0.5},    // its own radius; h > 0
             "qcdc_ring":  {"x": 18.5, "y": -7.5, "r": 22.1, "r_inner": 10.5, "h": 0.5}},
                                          // raises the pool: the circle is pulled up h into a
                                          // cylinder cut from the fluid, its top is the pool
                                          // patch and its side the wall patch wall_pool_<name>;
                                          // r_inner > 0 splits the top into a disc and a ring
  "pool_radius_m": 26.0,                  // the radius of every [x, y] point
  "roof_patches": {"nh3_source": 306},    // solid tag whose flat roof becomes its own patch,
                                          // with the 2.5/5/10 m refinement boxes around and
                                          // downwind (-x) of it
  "regions": {                            // solids kept as their own volumes (see "Regions")
    "solids": [ {"tag": 2, "name": "building", "kind": "solid",
                 "material": "concrete", "outer": "building_outer"} ],
    "interface_names": true               // false: no 2-D group for the shared faces
  },
  "sizes": {
    "min": 1.5,            // Mesh.MeshSizeMin
    "max": 40.0,           // Mesh.MeshSizeMax and every field's VOut
    "pool": 2.0,           // the ±40 m, +10 m box at each point (the pool)
    "box": 4.0,            // the ±150 m, +30 m box at each point (the brief's box)
    "growth_from": 2.5,    // size at a point, growing to "max" over 40..700 m
    "near_struct": 4,      // size within structures nearer than near_radius to a point
    "far_struct": 12,      // size near the rest of the structures
    "near_radius": 400,    // the near/far split, from the point's (x, y)
    "gap_ratio": 0,         // >0: the local-feature-size rule h <= g/gap_ratio. After the surface
                            // mesh, every triangle measures g, the distance along its inward
                            // normal to the nearest triangle facing it (a slot between two walls,
                            // a wall over the ground); where g/gap_ratio is below the local size
                            // the spot is refined (Box fields on a 10 x 10 x 5 m grid) and the
                            // surface is meshed again. Slots narrower than gap_ratio x min cannot
                            // be meshed without slivers: they are listed in the summary's
                            // "gap_unresolvable" - smear or remove that geometry (solids.hull_*)
    "gap_min_m": 0,         // the floor of the gap refinement far from the points (0 = min);
                            // without it the gap rule tripled a site mesh to 24 M tets
    "plume": {"downwind_m": 0, "upwind_m": 0, "half_width_m": 0, "height_m": 0,
              "size": 0, "thickness_m": 60},
                            // a box of this size at every point: downwind_m towards -x
                            // (the wind blows that way), upwind_m towards +x, +-half_width_m,
                            // from 1 m under the ground to height_m; off while size is 0
    "size_mult": 1.0,      // >1 coarsens every size (a solver-robustness reproducer)
    "regions": {"building": {"size": 3.0, "reach_m": 20.0}},   // per declared region (see "Regions")
    "roof_boxes": [2.5, 5.0, 10.0]   // the three roof-patch box sizes (near, mid, downwind)
  },
  "mesh":  {"algo2d": 6, "algo3d": 1, "optimize_passes": 5, "threads": 32,
            "relocate_passes": 0,   // passes of gmsh's Relocate3D after the optimiser: interior
                                    // nodes move to the quality-optimal spot, boundary nodes and
                                    // connectivity stay (5 lifts the p1/p5 quality markedly)
            "smoothing": 1},        // Laplacian smoothing steps of the surface meshes (gmsh default 1)
  "post":  {
    "flat_tets": true,       // the flat-tet stage after the 3-D pass
    "flat_threshold": 1e-7,  // a tet is flat when |V| < this × (longest edge from node 0)³
    "seam_merge_m": 0.02,    // node pairs closer than this in a flat tet are merged everywhere
    "sliver_edge_m": 0.6,    // edges shorter than this inside a tet under sliver_vol_m3 collapse
    "sliver_vol_m3": 0.2,
    "sliver_rel": 0.0,       // >0 switches the sliver collapse to relative thresholds: a tet is
                             // thin when |V| < sliver_rel × (mean edge)³ (a regular tet has
                             // |V| = 0.11785 × mean edge³, so 0.01 selects tets flatter than
                             // about 8 % of regular) and sliver_edge_m / sliver_vol_m3 go unused
    "sliver_edge_rel": 0.25, // in that mode, an edge is collapsible below this × the tet's mean edge
    "thin_push_m": 0.0,      // >0: push a node out of thin tets (gamma < 0.02) by up to this
    "min_thickness": 0.0,    // >0: the thickness gate tau = 3V/A_max^1.5 (regular tet 1.24, the
                             // site's slivers 0.01): every tet below it has the node opposite its
                             // largest face pushed along the normal by the height the gate asks,
                             // guarded against inverting any tet around; nodes inside a pool's
                             // refinement box never move; repeated repair_rounds times, then the
                             // survivors are counted (notes "thickness: ...") and the worst ten
                             // listed in the summary's "thickness_gate" with their positions
    "repair_rounds": 3,
    "improve_below": 0,     // > 0: every tet under this SICN has its nodes tried at a few
                            // positions (towards the neighbours' centroid, away from the
                            // opposite face) and keeps the one that raises the minimum SICN of
                            // the tets around the node; interior nodes move freely, a boundary
                            // node slides only inside one plane of one patch, the rest stay
    "improve_rounds": 3
  },
  "classification": {"wall_prefix": "wall_", "pool_prefix": "pool_", "big_roof_is_ground_m2": 2000}
}
```

Near-touching solids: two solids whose vertices stand 1 mm..`solids.touch_warn_m` apart
leave a razor-thin fluid gap that the mesher may treat as duplicate points, and a 3-D
boundary-recovery failure usually sits at one of the pairs. The cut stage logs each pair
(`near-touching solids: d 0.0198 m at (1058.99, 1163.57, 5.00) solids [11, 338]`, at most
30 lines, sorted by distance) and ends with the count plus that hint; the full list lands in
the summary's `near_touching`, and an empty 3-D mesh names it again before writing
`work/surface_only.msh`. Diagnostic only — nothing is moved or fused; exclude one solid of
the pair with `solids.exclude_tags` to remove one side.

Far-solid smear (`solids.hull_beyond_m` > 0): on the ammonia site the steady solution first
blew up 1.2 km from the release, in a 0.2–0.4 m slot between the walls of two neighbouring
blocks — a slot narrower than the 20 m far-field cell, which the mesher can only fill with
slivers — and the luck-dependent 3-D failures sat at 2–5 cm corner gaps of the same kind. Far
from every point a building's outline does not matter, so each far solid becomes the prism of
its footprint's convex hull pushed out by `hull_pad_m` (1 m closes every slit up to 2 m wide;
the prisms overlap and the cut merges them). Where two buildings shared a wall line the padded
corners land millimetres apart, so corners of different prisms within `hull_snap_m` are snapped
onto one point before the prisms are built and `fuse: true` then merges the shared edges. The
log says how many solids were replaced, how many corners were snapped, how many were kept (near
a point) or skipped (degenerate outline); the summary carries `solids_hulled` and
`hull_corners_snapped`.

Wedges: the gap pass also finds every boundary edge whose fluid-side dihedral is under
30 degrees (a basin slope meeting the water plane, a wall grazing a domain side): the log
names the places and the smallest angle, the summary keeps them as `wedges`, and the
triangles on both sides are refined - the cells along such an edge are thin however they
are sized, so the geometry there has to change (fill the pit, pull the wall back, cut the
domain short of it).

The absolute sliver thresholds (`sliver_edge_m`, `sliver_vol_m3`) are for meshes whose
smallest cells are metres; the relative ones (`sliver_rel` above 0, collapsing edges below
`sliver_edge_rel` × the mean edge) follow the locally refined cell size. On a pool refined
to 0.35–0.55 m the absolute keys count every fine tet as thin and the collapse loop makes
the quality worse; that is what the relative mode is for.

A repair is the reference's route E: the solid's surface meshed at 3 m,
pymeshlab-uniform-resampled on a `cell_m` grid, quadric-decimated (the
coarsest of `target_faces` / 10000 / 16000 that is watertight, manifold and
free of self-intersections), rebuilt as an OCC solid of planar triangles, and
cached as `work/repaired_<tag>.brep` (already in metres; pass `"brep"` to reuse
one). `lift_z` raises it afterwards — the ammonia STEP floats the ship at the
STEP's z = 0 while the sea plane is z = +3.05. Repairs need `pymeshlab`.

Numbers the reference hard-codes and the schema does not name (the field boxes'
extents — ±40/±150 m around the points, ±60/±250 m and the −700 m downwind arm
around a roof patch, the growth distances 40/700 m, the structure-field
distances 0–80/120 m, the ground scan's −1..60 m at 0.25 m, the 0.5 m hull
slack, the 2000 m² big-roof area's 100 × 100 m shape test) are ported
unchanged; they are the field layout the site mesh was tuned with.

## Regions

A `regions.solids` entry `{"tag", "name", "kind": "solid", "material",
"outer"}` keeps one imported solid as its own volume: never cut, sunk or
hulled — after the cut the fluid is `occ.fragment`ed with it, so every face
they share is one OCC surface, meshed once (conformal interface). Each volume
gets its 3-D group (`fluid`, `<name>`); the shared faces get one 2-D group
`fluid_to_<name>` (`<a>_to_<b>` between two declared solids, config order);
the solid's remaining faces get `<name>_outer` (`outer` renames it).
`interface_names: false` writes no 2-D group for the shared faces —
`ofgpu-convert-mesh -keepRegions` then yields one merged mesh with the
interface internal; the summary still reports it. Every declared solid gets a
Distance/Threshold field over all of its faces (`sizes.regions.<name>`:
`size` default `sizes.near_struct`, `reach_m` default 80 m). Refused by name
with regions: `trim` (keeps only the largest volume), `repairs`,
`post.flat_tets` (rebuilds the fluid's mesh alone), `sizes.gap_ratio` (reads
the fluid alone), a declared tag that is also excluded, repaired or a
`roof_patches` solid, a tag that is the fluid's or not in the STEP, a `name`
colliding with `fluid`, a fixed patch, a point or a roof patch; the fluid
must outweigh every declared solid, and a solid partly outside the fluid is
refused (the fragment must leave one piece per solid). The summary carries
`regions.<name>` (tag, kind, material, outer, mass_m3, centroid,
shared_with_fluid, groups, tetrahedra), `interfaces`, `fluid_tetrahedra` and
`groups.fluid_to_<name>`; the checkpoint carries `regions`, and
`--from-checkpoint` re-identifies each region by mass and centroid. M4's
`regions_from_msh.py` is the consumer of the names this writes.

## The self-test

```
python tools/mesh/selftest.py [--keep]
```

builds its own tiny STEP (a 100 × 60 × 40 box minus a 20 × 20 × 8 building on a
z = 0 floor), writes a config for it (one point, one roof patch) and runs the
tool four ways — `--dry-run`, `--stop-after-checkpoint`, the full run, and
`--from-checkpoint` — asserting each time: exit 0; the patches `top`, `west`,
`east`, `south`, `north`, `wall_ground_land`, `wall_buildings`, `roof`,
`pool_yard` exist; tets > 0; no negative volumes in the summary. The full run
also asserts `near_touching == []` (the tiny STEP has no near-touching pair),
and the `--from-checkpoint` path reruns the post stage with `sliver_rel: 0.01,
sliver_edge_rel: 0.25` and prints the flat-tet notes. With the building
declared as a region, the tool runs three more times (dry-run, full,
from-checkpoint) plus one with `interface_names: false`, and four refusals
are checked; a `--keep` run leaves `out_regions/selftest.msh` for M4. When
`rust/target/release/ofgpu-convert-mesh.exe` is built, the mesh is also
converted to a case's `polyMesh` and to a Fluent mesh. Seconds, no STEP input
needed.

## Regions: one polyMesh per volume

M3's `.msh` becomes a region layout — one complete, standalone polyMesh per
named volume, every shared face a boundary face of BOTH regions in a patch
pair whose k-th faces coincide, and a `regions.json` naming it all:

```
python tools/mesh/regions_from_msh.py <mesh.msh> <outDir> [--material R=N]... [--fluid NAME] [--overwrite]
python tools/mesh/regions_check.py <outDir>/regions.json
```

The layout on disk:

```
outDir/
  regions.json
  fluid/polyMesh/{points,faces,owner,neighbour,boundary}
  building/polyMesh/{...}
```

`regions.json` is docs/10-fsi-solid-mesh-plan.md §C: `version` 1, `units`,
`regions` (`name`, `kind`, relative `polyMesh`, and `material` on a solid),
`interfaces` (`regions`, `patches`, `faces`, `tolerance`), `source` (tool,
version, geometry basename, config flags — no paths). The region order puts
the fluid first (the volume named by `--fluid`, default `fluid`), then every
other volume by ascending physical tag; `--fluid none` makes every region
solid. A shared face becomes the patch pair `<a>_to_<b>` / `<b>_to_<a>` in
identical order — the k-th face of one is the reversed k-th face of the
other — so the solver can pair by index. Patch types follow convert_mesh.rs's
convention: a case-insensitive `wall`/`empty`/`symmetry` prefix, else
`patch`; an interface patch is always `patch`. Boundary faces no physical
surface covers become `defaultFaces` (warned). The five files of every
polyMesh are byte-identical to what `ofgpu-convert-mesh` writes for the same
single-volume mesh — `polymesh_write.py` is the Rust writer's byte-for-byte
twin (same banner, `%.17g` points, face order, boundary padding).

Refused by name, writing nothing: a volume without exactly one physical name,
a mesh with no `$PhysicalNames` section, a face shared by three cells, a
volume name carrying `_to_`, `--fluent` (the Fluent writer is tet-only and
single cell zone), an existing `regions.json` without `--overwrite`,
`--fluid`/`--material` naming no volume. `regions_check.py` refuses a layout
that breaks R1 (a region complete and standalone, positive pyramid volumes),
R2 (the paired faces coincident within the manifest tolerance, reported as
centroid / area / normal worsts), R3 (the pair names and types) or R6 (the
§C keys, relative paths, five files): exit 0, or 1 with one line per
violation, or 2 on usage. `regions_selftest.py` builds six gmsh meshes (tet
and transfinite hex), runs the tool and the checker on them and demands byte
equality with the Rust converter:
`python tools/mesh/regions_selftest.py [--keep DIR] [--msh PATH]`.

The face dedup is pure Python (a dict keyed by each face's sorted vertex
tuple), built for benchmark-sized meshes: a 10 M-tet site mesh takes minutes
and gigabytes.

## Benchmark recipes (tools/mesh/examples/*.py)

Five recipes mesh the benchmark geometries of the FSI / solid-mesh plan
(docs/10-fsi-solid-mesh-plan.md §I M5) as hex region layouts and hand the
`.msh` to M4's converter:

```
python tools/mesh/examples/turek_hron.py     [--level 1|2|3] [--out DIR] [--dz 0.02]
python tools/mesh/examples/bimetal_strip.py  [--level 1|2|3] [--out DIR]
python tools/mesh/examples/thick_cylinder.py [--level 1|2|3] [--out DIR]
python tools/mesh/examples/cantilever.py     [--level 1|2|3] [--out DIR]
python tools/mesh/examples/selftest_recipes.py [--keep] [--full]
```

`turek_hron.py` is the Turek & Hron (2006) FSI benchmark: the 2.5 × 0.41
channel, the cylinder r = 0.05 at (0.2, 0.2) and the elastic flap
[0.2, 0.6] × [0.19, 0.21] minus the disk, a 2-D OCC fragment extruded one
cell with recombine (`dz` 0.02, forces are F / dz). Measured on the reference
machine (gmsh seconds / M4 seconds in the sidecar's `timings_s`):

| level | h_near | h_far | fluid cells | flap cells | cylinder faces | seconds |
|---|---|---|---|---|---|---|
| 1 | 0.005   | 0.04 | 6188  | 298  | 60  | 0.2 gmsh, 10 M4 |
| 2 | 0.0025  | 0.02 | 25145 | 1146 | 118 | 1.0 gmsh, 38 M4 |
| 3 | 0.00125 | 0.01 | 98220 | 4524 | measured by `--full` | 6.8 gmsh, 146 M4 |

Patches: `fluid_to_flap` / `flap_to_fluid` (the interface, the wetted flap
surface), `empty_front`, `empty_back`, `inlet`, `outlet`, `wall_top`,
`wall_bottom`, `cylinder`, `flap_fixed` (the clamp, the flap's face on the
cylinder). Level 1 lands in `cases/turekHron/mesh` — the path the two case
skeletons `cases/turekHron/fsi1.jsonc` and `cfd1.jsonc` name — and levels 2
and 3 default to `mesh_L2` / `mesh_L3` beside it. Every recipe writes a
sidecar `<out>/<recipe>.json`: `recipe`, `level`, `gmsh`, `units`,
`timings_s`, the per-region `cells` / `patches` / `elements`, the
`interfaces` with the pairing worsts, plus per recipe — Turek-Hron: `dz`,
`h_near`, `h_far`, `flap_length_m` (0.35101, the free length after the disk
subtraction), `point_A`, `flap_cells_across`, `cylinder_faces`,
`interface_faces`; bimetal: `L`, `w`, `h1`, `h2`, `n` and `zones` — the two
S9 `bounds` boxes (`lower` z 0..h1, `upper` z h1..2h1) with their measured
cell counts. S9 assigns materials per closed box on the cell centroid, so no
cell-index list is written anywhere; `nz` is even so the bond plane is a
mesh plane.

The three solids are transfinite hex, one solid region each (`--fluid none`):
`bimetal_strip.py` (`strip`, 50·5·8 = 2000 cells at level 1, patches
`fixed_end`, `free_end`, `bottom`, `top`, `side_y0`, `side_y1`),
`thick_cylinder.py` (`cylinder`, the quarter annulus r 0.05 → 0.10, z 0..0.02,
nr·nt·nz = 192 at level 1, patches `inner`, `outer`, `zmin`, `zmax`,
`symmetry_x0`, `symmetry_y0` — typed `symmetry` by the name convention) and
`cantilever.py` (`beam`, 40·4·4 = 640 cells at level 1, the cantilever
patches, the sidecar carrying `I`, `A`, `beta1_L` and the two closed forms
the S7/S8 gates check against). Their default `--out cases/<recipe>_L<level>`
is git-ignored, and so is everything under `cases/turekHron/mesh*` — the
layouts are regenerated, never cloned.

`python tools/mesh/examples/selftest_recipes.py` runs every recipe at level 1
plus Turek-Hron level 2 under `tempfile.mkdtemp` and asserts the M5 table;
`--full` adds level 3 and the three solids at level 2.

## Known limits

- **3-D Delaunay only (`algo3d: 1`).** HXT (10) is faster but on this class of
  geometry it hits gmsh's unfinished Steiner-point path during boundary
  recovery and kills the whole process — no exception, so nothing can catch it.
  Netgen's optimiser dies with an access violation on this gmsh build (4.14.1)
  and is not offered.
- **Thin cells.** The solver's pressure equation is sensitive to them; the
  flat-tet stage exists because Delaunay leaves zero-volume tets on planar
  boundaries and slivers along seams. Raise `post.thin_push_m` (e.g. 0.2) when
  a site still produces thin wedges between a hull and the sea plane.
- **Seam merging happens after the 3-D pass.** Merging duplicate nodes before
  it makes gmsh drop every tet ("No elements in volume"); the reference found
  this on 2026-09-08 and the flat-tet stage's `seam_merge_m` is the working
  route.
- **Pool discs assume a planar ground** at the point. A disc over sloping
  ground comes back embedded, is reported `not imprinted` in the summary, and
  the run continues without that pool.
- STEP units are whatever `scale` says they are; the tool never inspects the
  file's unit declaration, it just multiplies.

## Checking a Fluent mesh independently

`tester_mesh.py [size]` builds a small known case - a hexahedral fluid box with a tetrahedral
solid cut out of its middle - and `fluent_check.py <mesh.msh> [tester_tet_geometry.json]`
parses any Fluent ASCII mesh from scratch, rebuilds every cell from its faces and checks that
each cell closes and has positive volume, that the total volume is the box minus the
tetrahedron, that boundary faces have one cell and point outwards, which c0/c1 orientation
the file follows (ANSYS B.3.7: the right-hand normal points toward c1), and that every zone
lies on the plane or solid it is named after.

```
python tools/mesh/tester_mesh.py 0.5
ofgpu-convert-mesh tester_tet.msh case -fluent tester_ours.msh
python tools/mesh/fluent_check.py tester_ours.msh tester_tet_geometry.json
```

On 2026-09-09 this tester was also run through OpenFOAM's foamMeshToFluent (the official
opencfd/openfoam-default:2312 Docker image, used only as a tool): both files hold the same
9,488 cells, nodes and face-cell topology and the same volume 238.396667 m^3; they differed
only in the orientation convention. On 2026-09-10 Fluent 2022 R2's own mesh check settled
it: the file written to the manual's sentence (thumb toward c1) came back with every face
left-handed and every one of its 7.8 M cells negative, and the file with every face's node
order reversed (thumb toward c0, boundary normals into the cell - foamMeshToFluent's
convention) was accepted. The converter has written the reversed order since; `fluent_check.py`
still measures and reports whichever convention a file follows.

## A caveat on `solids.sink_m` and `trim.below_z`

Keep the sunk building bases clear of the trim plane. The ammonia site's buildings stand at
z = 5 and its sea plane is z = 3.05: `sink_m: 2.0` puts every base at exactly 3.0, five
centimetres under the trim, and the boolean leaves a 5 cm strip of wall under the sea face
(the run reports it as `faces reaching below: N`, 1,814 there); `sink_m: 1.5` keeps the bases at
3.5 m, still 1.5 m below the lowest terrain a building floats over (4.7 m), and the count drops
to zero. Sink enough to bury every floating base, not so much that a base lands on the trim.

A solid standing over water is the exception: its sunk base (3.5 m) still floats 0.45 m above
the sea plane, and that slab of fluid under it is where the three-inlet mesh kept a cell the
thickness gate could not repair (tau 0.024 at the north-east corner, a fused hull prism straddling
the shore). `base_below_sea_m: 1.0` pushes every base to at least 2.05 m — inside the ground on
land, and cut away by the trim over water, so the prism wall meets the sea face at a right angle.
