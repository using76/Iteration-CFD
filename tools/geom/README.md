# tools/geom — the geometry tool's read side

`geom_tool.py` reads a geometry file on gmsh's OpenCASCADE kernel, lists what is
in it and writes it back in another format. It is the M1 half of Stream M of the
FSI / solid-mesh plan (docs/10): `edit` (transforms and booleans, an ops
history in the sidecar) is the next unit and lands in this same file.

```
python tools/geom/geom_tool.py info   <file> [--json <out.json>] [--scale S]
python tools/geom/geom_tool.py export <file> --out <path> [--scale S] [--stl-size M]
```

| extension | read by | default `--scale` (file units → metres) | export |
|---|---|---|---|
| `.step`, `.stp` | `occ.importShapes` | `0.001` (STEP files are millimetres) | yes — the mm dilate dance |
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
`solid_<tag>`. `export` measures the model in metres and writes the output's
sidecar before the STEP dilate dance; `edit` will append to `ops`.

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
null); the refusals. Seconds, no input files needed.

## Known limits

IGES imports as surfaces only and is never written. An STL is a discrete
triangulation: no solids, no sidecar, `volume` null while open, `--scale`
refused (the mesh is not an OCC shape; it is read in its own units), and the
`centroid` of a `discrete` entry is a surface centroid, not a volume centroid.
gmsh's own STEP writer does not carry entity names — the sidecar is the store,
and `.xao` is the one written format that carries them natively.
