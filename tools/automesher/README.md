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
castellation, snapping, layers). The tetrahedral path is §92.4 and the
polyhedral path is §92.5; both are specified there and neither is implemented.

## Usage

```text
ofgpu-automesher <config.json> [-schema] [-check <caseDir>] [-dryRun]
```

- `ofgpu-automesher <config.json>` - the meshing path. Reads and validates the
  config, loads and merges `input.surfaces[]`, requires a closed surface,
  runs the §92.2 stage-0 domain check, prints the surface summary and the
  plan, and then - unit 1 ships the skeleton, unit 2 builds stage 1 - refuses:
  `not implemented: stage 1 (octree refinement) - SPEC-LIT §92.2; tranche 1 unit 2`,
  exit status 1. Nothing is written.
- `ofgpu-automesher <config.json> -check <caseDir>` - **fully working.** Reads
  `<caseDir>/constant/polyMesh` and runs the §92.3 quality gate (G1-G7) with
  the config's `quality` thresholds. Every gate passed: the measured summary,
  exit 0. Any gate failed: the refusal naming the gate, the cell ids, their
  centroids and the measured values, exit 1. This is how a mesh that already
  exists - from this mesher, a converter, or anywhere else - is judged before
  the solvers are pointed at it.
- `ofgpu-automesher <config.json> -dryRun` - everything up to and including
  the surface summary, then exit 0 without attempting the meshing stages.
  What a config check wants before queuing a long run.
- `ofgpu-automesher -schema` - the JSON Schema of the config on stdout, exit
  0, generated from the same types that parse it. No config is read and every
  other argument is ignored.

## The config schema

JSONC (comments and trailing commas allowed, unknown keys refused by name).
The worked example is [`examples/nh3_site.json`](examples/nh3_site.json) - the
ammonia-terminal site, with comments on every block. Every key of
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
| `refinement.feature_angle_deg` | number | `30.0` | Dihedral angle past which a triangulation edge is a feature edge (eq. 92.2). |
| `refinement.max_level` | integer | `2` | The level cap `l(c)` is min'd with (eq. 92.1); 6 is the §74.2 octree cap. |
| `castellation.keep_region` | string | `"largest"` | Which connected component of the fluid survives eq. (92.4): `"largest"` or `"seed"`. |
| `castellation.seed_point` | number[3] | `null` | The keep point `"seed"` needs; required then, ignored otherwise. |
| `castellation.min_faces` | integer | `4` | A kept cell with fewer faces than this is dropped - a hole in the addressing, not a control volume. |
| `snap.iterations` | integer | `30` | Snapping iterations, the `k` of eq. (92.5). |
| `snap.tolerance` | number | `1e-3` | Points closer than this to the surface are considered on it (and the convergence test, in units of `base_size`). |
| `snap.smoothing_passes` | integer | `3` | Laplacian passes over the displacement field, eq. (92.6). |
| `snap.smoothing` | number | `0.5` | The smoothing relaxation weight, in `[0, 1]`. |
| `snap.undo_limit` | integer | `4` | Halvings of a gate-breaking displacement before it is zeroed and the point pinned - eq. (92.7)'s undo. |
| `layers.patches` | string[] | `[]` | The patches layers are added to; empty means none. |
| `layers.n` | integer | `0` | Number of layers, eq. (92.9)'s `n`. Zero: no layers. |
| `layers.first_thickness` | number | `0.05` | The first layer's thickness, metres. |
| `layers.growth` | number | `1.3` | The expansion from one layer to the next; must be > 0. |
| `layers.min_thickness` | number | `0.1` | Total thickness below which the layers are dropped. |
| `layers.medial_frac` | number | `0.5` | Fraction of the local medial-axis distance below which the layers are dropped (eq. 92.10). |
| `quality.max_closure` | number | `1e-9` | G2 (92.12): `\|sum s Sf\| / V^(2/3)` stays under this. |
| `quality.max_non_orth_deg` | number | `70.0` | G4 (92.13): max internal-face non-orthogonality, degrees. |
| `quality.report_non_orth_deg` | number | `60.0` | G4: faces past this are counted and reported, not refused. |
| `quality.min_thickness_ratio` | number | `0.05` | G5 (92.14): `tau_c = 3 V_c / A_max^(3/2)` stays at or above this. |
| `quality.max_cond` | number | `1e4` | G6 (92.15): `cond(T_c)` stays under this. |
| `output.case_dir` | string | *(required)* | The case directory the mesh is written to. |
| `output.name` | string | *(required)* | The mesh's name within the case. |

## The pipeline

SPEC-LIT §92.2's seven stages, and which tranche-1 unit implements each.
Unit 1 is this unit: the binary skeleton, the config tree, and the §92.3 gate.
Units 2-7 are the stages themselves.

0. **The background block** - a `blockgen::BlockSpec` over `domain.extent`
   with `base_size` as the target cell size and `domain.grading` per axis;
   the domain-and-surface containment check, then the plan. Unit 1: the check
   and the plan run here, and `blockgen` (§23.4) already builds the block.
1. **Octree refinement** - per-cell level from distance bands, feature edges
   and explicit regions, capped at `refinement.max_level` (eq. 92.1). **Unit
   2 - the stage the skeleton currently refuses at.**
2. **2:1 balance** - the level fixed point of §74.2 (eq. 92.3), on §74's face
   conventions. **Unit 3** (reusing `mesh::refined::balance_2to1`).
3. **Castellation** - parity classification (§23.3), keep the connected
   component eq. (92.4) names, drop the sealed pockets by name and count.
   **Unit 4.**
4. **Snapping** - boundary points to the closest surface point (92.5), the
   displacement field smoothed (92.6), gate-breaking displacements undone
   (92.7). **Unit 5.**
5. **Feature snapping** - points near a feature edge projected onto the edge,
   corners pinned (eq. 92.8), before stage 4 and pinned through it.
   **Unit 6.**
6. **Layers** - inward prismatic extrusion (92.9)-(92.10) with the medial-axis
   limit, gate-checked per patch. **Unit 7.**

## The quality gate

SPEC-LIT §92.3's seven checks - the precondition every stage must hold, and
the report `-check` prints. The mesher **refuses rather than ship a bad mesh**:
a stage that cannot keep the mesh inside these ends the run, naming the gate,
the cell ids, their centroids to six figures, the measured values and the
thresholds.

| Gate | Check | Threshold | Config knob |
|---|---|---|---|
| G1 | positive volume, `V_c > 0` (92.11) | every cell, no knob | - |
| G2 | closure, `\|sum s Sf\| / V_c^(2/3)` (92.12) | `< 1e-9` | `quality.max_closure` |
| G3 | one cell region (`cell_regions`) | exactly 1, no knob | - |
| G4 | non-orthogonality, internal faces (92.13) | max `< 70 deg`; faces past 60 deg reported | `quality.max_non_orth_deg`, `quality.report_non_orth_deg` |
| G5 | thickness, `tau_c = 3 V_c / A_max^(3/2)` (92.14) | `>= 0.05` | `quality.min_thickness_ratio` |
| G6 | gradient conditioning, `cond(T_c)` (92.15) | `< 1e4` | `quality.max_cond` |
| G7 | addressing (owner < neighbour, upper-triangular order, no duplicate faces) | pass/fail, no knob | - |

## Licence note

Nothing GPL-licensed was consulted for §92 or for this unit: not
`snappyHexMesh`, not cfMesh, not TetGen, not CGAL's GPL modules. The
specification's sources are the **OpenFOAM User Guide** (documentation, for
the castellation / snapping / layer-addition stage list), the papers §92 cites
(Schneiders 1996; Ito, Shih & Soni 2009; Marechal 2009; Owen, Staten &
Sorensen 2011; Isaac, Burstedde & Ghattas 2012; Freitag & Ollivier-Gooch 1997;
Garimella & Shephard 2000), and Ericson's *Real-Time Collision Detection*
(a textbook, read as a textbook). The seven gate checks are this project's
own, stated in §92.3.
