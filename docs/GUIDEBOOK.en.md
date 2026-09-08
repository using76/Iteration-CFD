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

The race-car sample is the fastest route: one geometry and three commands.

```powershell
cd cases
.\racecar.cmd
```

What the script does:

```powershell
# 1. Carve the body out of a 1 m cube tunnel, cut-cell.   20-60 min (CPU)
ofgpu-generate-mesh big racecar_case 128 -stl car=racecar.stl -cutcell

# 2. Add the pressure and temperature fields.             instant
copy racecar.fields\p racecar_case\0\p
copy racecar.fields\T racecar_case\0\T

# 3. Solve the momentum too.                              1-2 min (GPU)
ofgpu-lowmach racecar_case -iters 3000 -check 250 -output foam
```

Step 1 is slow because the **classification runs on the CPU**: two million cells
each tested inside-or-outside against the STL, and the ones the surface crosses
cut open. Step 3 is the GPU-resident loop and takes minutes.

Why step 2 exists is [§6](#6-choosing-a-solver--where-people-go-wrong). In short:
the mesh generator writes the fields the **turbulence-only** drivers read, and
not the `p` and `T` a momentum driver needs.

When it finishes, `racecar_case/0/` holds `U`, `p`, `T`, `k`, `omega`, `nut` and
`rho` in OpenFOAM ASCII. Open it in the Studio's 3D viewer, or read it straight
into ParaView.

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
                    [-wallModel standard|spalding|rough|lowRe [-Ks x [-Cs y]]]
                    [-cyclic x|y|z] [-permissive]
```

Presets: `channel`, `cavity`, `step`, `big`, `plume`, `room`, `damBreak`. `big`
is a 1 m cube tunnel and takes a **single** cell count (`n³`).

What comes out is a complete, ready-to-run case: `constant/polyMesh`,
`constant/physicalProperties`, `constant/momentumTransport`,
`system/{controlDict,fvSchemes,fvSolution}`, and a `0/` with `U`, `k`, `epsilon`,
`omega` and `nut`.

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
| `lowRe` | `nutLowReWallFunction` | `kLowReWallFunction` | `zeroGradient` |

The full table, including how it collapses for LES, is in
[`cases/README.md`](../cases/README.md).

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

### When the turbulence-only drivers are the right tool

When you want the turbulence quantities on a prescribed velocity field:
validating the model itself, comparing wall functions, studying the turbulence
response to a given field. `plume.jsonc` is exactly that — two equations
converging on a frozen `U`.

### What a momentum driver needs

`ofgpu-lowmach` solves `U` and `p` together, and the low-Mach loop wants a `T`.
The mesh generator writes neither, so you supply them. The race-car sample ships
two uniform fields in `cases/racecar.fields/` whose entire content is their
boundary conditions (`p = 0` fixed at the outlet, zero-gradient elsewhere,
isothermal 293.15 K).

Without them it refuses:

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

The mesh generator writes only the fields the turbulence-only drivers read. Add
`0/p`, and `0/T` for the low-Mach loop. `cases/racecar.fields/` in the race-car
sample is the worked example. See
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
settings name. Keep a copy of the initial fields elsewhere if you need them —
that is why `racecar.fields/` exists in the sample.

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
