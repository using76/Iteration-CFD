# tools/geom — the geometry tool's read side

`geom_tool.py` reads a geometry file on gmsh's OpenCASCADE kernel, lists what is
in it and writes it back in another format. It is the M1 half of Stream M of the
FSI / solid-mesh plan (docs/10); `edit`, the M2 half, lives in `geom_edit.py`,
which `geom_tool.py` imports for the `edit` subcommand.

```
python tools/geom/geom_tool.py info   <file> [--json <out.json>] [--scale S]
python tools/geom/geom_tool.py export <file> --out <path> [--scale S] [--stl-size M]
python tools/geom/geom_tool.py repair <file.stl> [--out <path.stl>] [--json <out.json>] [--weld REL] [--max-hole-edges N] [--ascii | --binary]
```

| extension | read by | default `--scale` (file units → metres) | export |
|---|---|---|---|
| `.step`, `.stp` | `occ.importShapes` | `0.001` (STEP files are millimetres) | yes — exact mm, via a BREP re-import at scale 1000 |
| `.brep` | `occ.importShapes` | `1.0` (raw numbers, metres) | yes — as is |
| `.iges`, `.igs` | `occ.importShapes` | `0.001` | refused (surfaces only) |
| `.xao` | `gmsh.merge` (+ `occ.dilate` when scale ≠ 1) | `1.0` | yes — one 3-D physical group per solid |
| `.stl` | `gmsh.merge` (discrete) | `1.0` | yes — binary surface mesh, no sidecar |
| anything else | refused by name, exit 2 | | refused by name, exit 2 |

`info` prints a text table — one line per solid: `tag, name, material, volume
[m^3], centroid [m], faces, closed` — and, with `--json`, the same document to
a file (never stdout): `solids` sorted by tag, `surfaces` (IGES only), `discrete`
(STL only: `n_triangles`, closedness by the every-edge-in-exactly-two-triangles
rule, and the divergence-theorem volume `V = Σ a·(b×c)/6`, null while the
surface is open), plus `duplicates_removed` (`{surfaces_before, surfaces_after}`
for STEP/BREP/IGES — a STEP round trip duplicates a shared face and
`removeAllDuplicates` repairs it, and the report says so) and the IGES `note`.
Every number is metres, m³ or m².

## The sidecar and the name rule

A STEP carries neither names nor materials and OCC renumbers tags on the way
back in, so the names travel in a `<stem>.geom.json` sidecar beside the file:
`{version, tool, units, scale, file, solids: [{tag, name, material, volume,
centroid}], ops}`. On import the records are matched to the model's solids by
centroid and volume first (1e-6 relative to the model diagonal) and by tag only
as the fallback — a moved solid keeps its name through a round trip, and a
sidecar whose tags are swapped still names every solid right. Without a
sidecar: the XAO physical group's name, then the OCC label's last path
component (unless it is OCC's own `Open CASCADE ...` default), then
`solid_<tag>`. `export` and `edit` measure the model in metres and write the
output's sidecar before the export replaces the model; `edit` appends its ops
to `ops`.

## Refusals

Exit 2, by name, before anything is written: a missing file; an unsupported
extension; `--scale` ≤ 0; an `--out` equal to the input; an `--out` this tool
does not write (`.iges` — IGES imports as surfaces only — or any
non-exportable extension); an input with no entities. A gmsh exception (an
unreadable or empty file) is exit 1 as
`geom_tool: <format> import failed: <gmsh message>`.

## The self-test

```
python tools/geom/selftest.py [--keep]
```

builds its own inputs with gmsh in a scratch dir (the two-box STEP, a one-box
STEP, a one-triangle STL), runs the tool as a subprocess and asserts the
contract: two solids of 1.0 and 2.0 m³ to 1e-9; the STEP/BREP/XAO round trips
(12 → 11 and 11 → 11 surfaces, names carried by the sidecar); the sidecar name
rules (swapped tags, moved centroids); IGES surfaces only (11 faces, 15 m²);
the STL read-back (a closed cube of volume 1.0 to 1e-6, an open triangle
null); the refusals. The repair tests build their own STLs the same way —
the perturbed cube (every corner copy shifted by up to 3.8e-8), the holed
cubes (one triangle and one whole face missing), the flipped and inward
cubes, the two-solid and fin cubes — and run on the tracked
`cases/racecar.stl`. Seconds, no input files needed.

## Known limits

IGES imports as surfaces only and is never written. An STL is a discrete
triangulation: no solids, no sidecar, `volume` null while open, `--scale`
refused (the mesh is not an OCC shape; it is read in its own units), and the
`centroid` of a `discrete` entry is a surface centroid, not a volume centroid.
gmsh's own STEP writer does not carry entity names — the sidecar is the store,
and `.xao` is the one written format that carries them natively.

## edit

```
python tools/geom/geom_tool.py edit <file> --ops ops.json --out <path> [--scale S]
```

applies named operations on gmsh's OpenCASCADE kernel and writes the result by
extension (`.step`/`.brep`/`.xao`) through the same export and sidecar as
`export`, with one stdout line per applied op. The output sidecar's `solids`
are the edited model in metres; its `ops` is the input sidecar's history plus
one entry per applied op — `{"op": <the op object as given>, "in": [[tag,
name], ...], "out": [[tag, name], ...]}` — cumulative across edits, no
timestamps, no absolute paths.

| op | keys |
|---|---|
| `rename` | `solid`, `name` |
| `set_material` | `solid`, `material` (a non-empty string, or null to clear) |
| `delete` | `solids` |
| `translate` | `solids`, `by` (3 numbers) |
| `rotate` | `solids`, `point`, `axis` (non-zero), `angle_deg` |
| `scale` | `solids`, `point`, `factor` or `factors` (3 non-zero numbers) |
| `mirror` | `solids`, `plane` `[a, b, c, d]`: the plane `a x + b y + c z + d = 0` |
| `fuse` / `cut` / `intersect` | `object`, `tools`, optional `name` |
| `fragment` | `object`, `tools` |
| `box` | `name`, `origin`, `size` (> 0), optional `material` |
| `cylinder` | `name`, `origin`, `axis` (non-zero), `radius` > 0, optional `material` |
| `sphere` | `name`, `centre`, `radius` > 0, optional `material` |

A solid is referenced by its name or its current tag; `solids`/`object`/`tools`
are non-empty, repeat no solid, and `object` and `tools` are disjoint. Lengths
are in the model's units after import (metres), `angle_deg` is degrees. Names
match `^[A-Za-z_][A-Za-z0-9_]*$` and stay unique.

The naming rule (docs/10's "say the rule"): within one run names ride gmsh's
`out_map`, because a boolean's products have new centroids and volumes no
matcher could recognise — a `fragment` names each piece by its contributors
(two overlapping boxes `a`, `b` give `a`, `a_b`, `b`), a `fuse`/`cut`/
`intersect` takes the op's `name` or the first object's name, and a name given
to n > 1 outputs becomes `name_1 … name_n`, every piece suffixed. Across files
— OCC's renumbering on the way back in — names follow the sidecar
centroid/volume rule of the section above. Bystanders keep name and material
through booleans: OCC renumbers the operands, not the neighbours.

Refusals, exit 2 before anything is written: an unknown op or key, a missing
key, a wrong type (the message names `ops[k].<key>`; a stranger key gets a
"did you mean" hint); a reference that resolves to nothing (the message lists
the names that exist); `--out` equal to the input (`same file`); an input or
output of `.stl`/`.iges`/`.igs` (they import as surfaces only). Exit 1 is a
failure while applying or writing: a boolean with no output, a result with no
solid, a lost bystander tag, a gmsh exception. A `cut` consumes a disjoint
tool all the same. The millimetre STEP export is exact: the model goes out
through a BREP and back in at `Geometry.OCCScaling = 1000` — the old
`occ.dilate` dance re-approximated curved faces (a sphere lost 4.4e-4 of its
volume).

## repair

`repair` (stl_repair.py, also run standalone as
`python tools/geom/stl_repair.py <file.stl> ...`) repairs an STL to watertight
WHERE IT CAN BE and reports what it could not, writing a NEW file — the Rust
reader's weld stays bit-exact (*DESIGN*, SPEC-LIT §23.1), so an epsilon weld
must live outside the files that reader is pointed at. The order: the
bit-exact weld is measured first (`before` — what
`Surface::require_closed` would print for the input today), then duplicate
corners are welded at `--weld` × the bounding-box diagonal (default 1e-6, the
house convention; `0` = bit-exact only), degenerate triangles are dropped,
duplicate facets are removed (an exact copy always; a coincident opposite
pair whole when it is a zero-thickness sheet, otherwise the one of the two
that runs against the surrounding surface), a vertex lying on another
triangle's open edge within the weld tolerance splits that edge (a
T-junction), each vertex-connected component
is oriented patch by patch across manifold edges and every closed shell is
flipped outward, a boundary loop of up to
`--max-hole-edges` edges is filled — three edges with one triangle, four or
more by ear clipping in the plane of the loop's Newell normal (Meisters
1975) — and the result is written with normals recomputed from the winding.

Flags: `--out PATH` (without it the tool only reports), `--json OUT` (the
report, never stdout), `--weld REL`, `--max-hole-edges N` (default 32),
`--ascii | --binary` (default: the input's format; binary refuses a
multi-`solid` file). Report keys: `file, format_in, out, format_out, units,
bbox, diagonal, triangles_in, triangles_out, degenerate_dropped, duplicates
{exact_removed, opposite_removed}, t_junctions
{split_edges, inserted_vertices, triangles_added, passes}, weld
{tol_rel, tol_abs, points_raw, points_bit_exact, points_welded, merged,
max_move}, orientation {reoriented_triangles, left_alone, left_alone_reason,
reseeded_patches, flipped_components}, holes
{max_hole_edges, filled, filled_triangles, unfilled[], per_hole[]}, stages[]
(input, weld, dedup, tjunction, orient, fill: open and non-manifold edges after each), before
{}, after {open_edges, non_manifold_edges, closed, volume}, n_components,
components[]
{index, patch, n_triangles, open_edges, non_manifold_edges, closed, volume,
flipped}, patches`. Stdout carries eight lines (the summary, the weld, the
duplicates, the T-junctions, the orientation, the holes, the after-state, the written file),
one line per stage, plus one line per
open component when the result is not closed. Exit 0 whenever a report was
produced — an unclosed result is reported, not an error; refusals (missing
file, a non-`.stl` extension, `--out` equal to the input, `--weld < 0`,
`--max-hole-edges < 3`, `--binary` on a multi-patch file) exit 2 before
anything is written; a file that is neither binary nor ASCII STL, or carries
a non-finite coordinate, exits 1.

Limits: the weld tolerance is a geometry EDIT — the tool prints `max_move`,
the largest distance any corner moved, and the user decides; near pairs can
CHAIN (`a-b` and `b-c` merge `a` and `c` at up to twice the tolerance), and
the representative is always a coordinate the file already had. A loop
longer than `--max-hole-edges`, a loop that is not simple in its own
plane (ear clipping finds no ear), and a loop whose fill would add a
triangle with zero area at float64 — the same exact test that drops a
degenerate triangle on the way IN — are named in `holes.unfilled` and
never filled; the zero-area refusal (`filling it would add N zero-area
triangle(s)`) skips the loop whole rather than snapping, because filling
it would have written a triangle this tool's own reader drops, so the
file would not have the open-edge count the run reported. Every closed
shell is oriented outward — a cavity's shell is flipped outward too,
and no cavity is detected. Non-manifold edges are reported, never cut. The
licence rule: the tool imports numpy (BSD-3) and scipy (BSD-3) only; gmsh
(GPL-2.0-or-later) stays a separate program used by `info`/`export`/`edit`
whose files are exchanged and whose source is never read; pymeshlab (GPL-3)
is not used; no mesh-repair implementation of any licence was read.

A fill is applied only if it leaves every one of the new triangles'
undirected edges used at most twice: the uses are counted once before the
walk, a running tally follows the accepted fills, an ear-clipped loop's
whole candidate fan — its `m − 2` triangles, the edges they share with
each other counted once per triangle — is tallied before anything is
committed, and a loop whose fill would push an edge to three uses is
refused whole and reported in `holes.unfilled` as
`filling it would make N edge(s) non-manifold` instead of being filled.
Every loop the walk closes and every walk it abandons is one entry of
`holes.per_hole`, in walk order — `{"edges", "outcome": "filled" |
"skipped", "triangles", "reason"}` — whose `triangles` sum is
`holes.filled_triangles` and whose `skipped` entries mirror
`holes.unfilled` one for one.
After the repair the defects are recounted, and a result carrying MORE
non-manifold edges than the input refuses the write — exit 3,
`the fill raised non_manifold_edges from A to B - this is a bug, report
it` — before anything is written.

Duplicates (AUT-02a) are found after the weld, so `before` and the
weld numbers still count them: triangles with the same three welded
corners form a group, the first of each winding is kept and every later
copy is an exact duplicate (`duplicates.exact_removed`); a group left with
one triangle of each winding is a coincident opposite pair, removed whole
when every one of its edges has an even number of other users (a
zero-thickness sheet) and otherwise reduced to the one that runs opposite
to the other users on more edges (`duplicates.opposite_removed`; a tie
keeps the earlier one).  Duplicates are not counted in
`degenerate_dropped`.

T-junctions (AUT-02b) are resolved after the duplicates and before the
orientation pass.  Only open edges (one use) and boundary vertices (an
endpoint of an open edge) take part, so a closed surface is untouched.
A vertex `m` splits the open edge `(a, b)` of triangle `(a, b, c)` when it
is none of the three corners, its parameter `s = (m − a)·(b − a) / |b − a|²`
lies strictly between 0 and 1, its distance to `a + s (b − a)` is at most
the weld's absolute tolerance (`weld.tol_abs`), and it is farther than that
from `a` and from `b`.  The hits, sorted by `s`, replace the triangle in
place by the fan `(q_i, q_i+1, c)` from `a` to `b`, which keeps its winding
and its patch; a triangle splits one edge per pass, and a fan is not
made when one of its triangles has an exactly-zero cross product, when
one of its new spokes `(q_i, c)` is already an edge, because that fan
would lay a triangle over one already there — the three-triangle fan
around a collinear sliver `A, B, C` keeps its open loop for the
zero-area fill refusal instead — or when a sub-edge `(q_i, q_i+1)`
already has two uses (counted on the pass's input plus the fans planned
before it, the rule the hole fill applies, so no edge ends with three);
passes repeat until none splits (at most 16).  `t_junctions` counts the split edges, the vertex
insertions, the triangles added and the passes that split.  On the 14
tuning t_junction rows of family G every row closes, with area and
signed volume equal to the regenerated parent's to 1.3e-13 relative and
no hole filled; before AUT-02b, 5 of them closed only because the
three-edge fill laid a sliver triangle over the split.

The orientation pass (AUT-02a) walks PATCHES: a patch is a maximal set of
triangles joined by two-triangle edges whose two uses run opposite in the
input, so its winding is already consistent and every two-triangle edge
between two patches runs the same way in the input.  Each component is
seeded at its largest-area patch, which is never flipped; a patch reached
from the walk counts its same-way (S) and opposite (O) edges to the
patches already decided and is flipped whole when S > O, so a
corners-reversed disk of any size is re-oriented across its boundary
edges and the non-manifold count can only fall.  The pass counts two
things separately.  `left_alone` is the number of triangles in reached
patches it did not flip although S > 0 (S <= O), named in
`left_alone_reason`; a triangle that was merely never reached is not left
alone.
`reseeded_patches` is the number of extra seeds the orientation walk
needed because a component's remainder was not reached from the previous
seed; a re-seeded patch is oriented consistently within itself but not
necessarily with the patch before it.

Worked example, re-measured 2026-10-14 after AUT-02b on the F1 body that
an earlier session could not close (`c42-f1.stl`, 237,482 triangles):
before 13,854 open / 810 non-manifold edges; 1,874 degenerate triangles
dropped, 0 points welded; 9 exact duplicates and 28 opposite-coincident
triangles removed; 31 T-junction edges split at 32 vertex insertions
(32 triangles added, one pass); the orientation pass reorients 95
triangles, leaves none alone and takes 240 extra seeds
(`reseeded_patches`); the stages run input 13,854 open / 810
non-manifold, weld 13,854 / 810, dedup 13,859 / 773, tjunction 13,783 /
773, orient 13,783 / 677 and fill 6,978 / 677; the fill closes 491 loops
with 5,823 triangles — 33 of them three-edge, 458 ear-clipped — and names
1,272 loops in `holes.unfilled`; exit 0.  (After AUT-02a, 2026-10-13, the
same file ended at 7,119 open edges, 18 of its three-edge loops refused
as zero-area fills; before AUT-02a, 2026-09-20, at 7,138 open / 787
non-manifold: 15 triangles reoriented and 55 flips refused.)  The file is
still not closed, with 6,978 open edges: 87 loops longer than
`--max-hole-edges` 32 (4,364 edges), 1,135 walks that hit a boundary
vertex without exactly one unused outgoing open edge (1,958 edges), 43
loops whose clipped fan would make an edge non-manifold (542 edges) and
7 loops with no ear (114 edges) — the four families' edges sum to 6,978,
and no zero-area loop is left — so `ofgpu-generate-mesh` still refuses it
without `-permissive`.  Re-reading the file the run wrote reports 0
degenerate triangles dropped, because the fill never writes a triangle
the reader would drop, and 0 T-junctions.

`--max-hole-edges` is a trade, not a bug report: on the same
`c42-f1.stl`, `python tools/geom/geom_tool.py repair <file>.stl
--max-hole-edges 256 --out ... --json ...` (the default stays 32) fills
566 loops with 9,173 triangles and takes the open-edge total from 6,978
to 3,478 — the 87 loops longer than the default limit go to 0, at the
cost of 3,350 more triangles written — while the refusal families grow
because longer loops are harder to clip without pushing an edge to three
uses: the non-manifold family rises from 43 loops / 542 edges to 50 / 894
and the no-ear family from 7 / 114 to 12 / 626, and the non-manifold edge
count itself is unchanged at 677.  A bigger limit closes more of the surface at
the cost of more ear-clipped triangles and more loops that fail the
manifold and ear tests, and the default stays 32.
