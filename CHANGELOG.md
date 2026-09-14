# Changelog

All notable changes to meteor-cfd (`ofgpu`). The project is source-available under the
Prosperity Public License 3.0.0; see `LICENSE` and `LICENSING.md`.

## 2.0.0 — 2026-09-15

188 commits since 0.1.0. The engine grew a second physics, a mesh toolchain, a workbench, and — the
change that matters most — a habit of measuring itself on the card it runs on rather than trusting its
own unit tests.

Measured on the release commit, one NVIDIA GeForce RTX 5070 Ti (16 GB, driver 596.49, sm_120, f64):

| check | result |
|---|---|
| `ofgpu-validate` | **880 / 880 checks passed**, 835 live + 45 replayed, 35.7 min, exit 0 |
| library test suite | **1924 passed, 0 failed**, 8 ignored |
| GUI server / shared | 322 / 50 passed |
| device vs host | `ofgpu-probe`: `max \|gpu − cpu\| = 0.000e0`, bitwise identical |
| determinism | every run repeated; peak stress, iteration counts and residual lines bitwise identical |

### Added

**A solid that deforms.** Quasi-static thermo-elasticity on the same polyhedral mesh as the flow:
displacement solved segregated on the existing scalar LDU with Anderson acceleration, the traction
condition solved at the face, stress recovered as a tensor field with von Mises, principal stresses and
the hydrostatic/deviatoric split, and two bonded materials joined by a series coefficient rather than an
interpolated one. Gated on closed forms — free expansion, the linear-displacement patch test, a thick
cylinder driven through the conduction solve (≤ 1 % on three meshes with its grid-convergence index
beside it), and Timoshenko's 1925 bimetallic strip to 0.6 %.

**A conjugate run that tells the truth about convergence.** One residual per region on the §8.4
normalisation, and `converged` is printed only when every region met the tolerance — a union run whose
soft region sat four decades above the global number used to report success.

**Verification that measures its own order.** Observed order of accuracy and grid-convergence index
(Roache; Celik; Eça & Hoekstra), an uncertainty attached to every multi-mesh gate, and a gate that
carries no uncertainty now fails to register.

**A mesh per region, and a layout that names the pairs.** An imported polyMesh per region, a
`regions.json` manifest with conformal interface patch pairs whose k-th faces coincide, and a Python
toolchain that produces it: a geometry tool on OpenCASCADE (list, rename, transform, boolean, export),
multi-region meshing from one STEP, a multi-volume `.msh` split into standalone region meshes, and
recipes for the Turek–Hron benchmark and three solid gate bodies.

**Our own mesher.** `ofgpu-automesher`: octree, castellation, snapping, feature capture and layers,
with a quality gate a mesh cannot leave without passing, and a driver that runs the four stages behind
one command.

**A workbench.** A desktop-class window (viewport, mesh and post-processing, run monitor, case editor,
geometry tab) and an AI assistant that drives both the solver and the window through a typed tool
registry with an approval round trip — 35 tools, per-tool auto/ask/never policies, sessions on disk.

**Real-point output.** VTU with shared points and polyhedral cells instead of synthesised per-face
quads, point data beside cell data, and tensor fields.

### Changed

- The site-mesh toolchain became a documented recipe: one command from a STEP to polyMesh, Fluent and a
  fluid STEP, with a plume refinement box, node relocation smoothing and a worst-cell repair pass.
- `ofgpu-cht` gained a `mechanics` block, a `stress` run mode, an `output` block and a shipped schema.

### Known limits, named

This project states what it cannot do rather than letting a user discover it.

- **A bending-dominated slender body is refused.** The segregated displacement loop stalls above a
  slenderness of about 5 — measured on three meshes, at two Poisson ratios, with and without Aitken and
  Anderson acceleration, and with the implicit coefficient enlarged. The block-coupled matrix is named as
  the route and is the next section to be written. Compact bodies are what release 2 stands on.
- **A memory step at 4.72 M cells.** The k-ε model's device footprint jumps by about 6.4 GB between
  4.72 M and 4.80 M cells, deterministically, cutting the practical ceiling on a 16 GB card from roughly
  14 M cells to 6 M. Diagnosed, not yet fixed.
- **`solver GAMG;` runs PBiCGStab.** There is no multigrid in the default build and the substitution is
  currently silent. A refusal is planned.
- **Two gates miss and six are open**, all against published literature and all pre-existing; none is in
  the solid work. `ofgpu-validate` names each one.
- **The reference-data store does not exist yet.** Published numbers are literals in the gate binary;
  several source files describe a `reference/` directory that is not in the repository.
- Single GPU. The decomposition path is proved bitwise at 2 and 4 parts but runs the parts in sequence on
  one device.
- f64 only in practice: the `single` feature exists and nothing builds it.

### Provenance

No GPL-, LGPL- or AGPL-licensed source was consulted for any of it. Every one of the 205 source files
under `rust/` carries that statement and a row in `PROVENANCE.md`, and two tests enforce both directions.
The specification the code is written against (`rust/SPEC-LIT.md`, 28,443 lines, 87 sections) cites its
sources by DOI and a citation audit fails the build when a citation names a section that does not exist.

## 0.1.0 — 2026-09-04

First tagged state.
