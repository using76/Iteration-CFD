# Licensing audit — what can and cannot ship

**Goal: ship meteor-cfd under a licence 주식회사 메테오시뮬레이션 controls.**

> This audit was written while the target was the MIT licence. The target has
> since changed to the Meteor Simulation Source-Available License 1.0 — source
> public, education free, research and commercial paid. **The analysis below is
> unchanged and remains the reason the rewrite happened**: GPL-derived code
> cannot be relicensed under MIT *or* under a proprietary licence. Removing it
> was necessary either way, and it is what makes the current licence possible
> at all.

This document is the honest accounting of where the current code came from, so
the reimplementation has a definite scope rather than a vague sense of unease.

---

## 1. The problem, stated plainly

The numerical core of ofgpu was **transcribed from OpenFOAM**, which is
**GPL-3.0**. Not "inspired by" — transcribed. The evidence is in this
repository's own words:

- `gpu/SPEC.md` opens with *"Authoritative transcription of the OpenFOAM-12
  semantics this project reproduces. Every formula below was read out of
  `upstream/OpenFOAM-Foundation-12/src`."*
- Each model header says *"A one-for-one transcription of OpenFOAM-12
  `src/MomentumTransportModels/.../kEpsilon`."*
- The porting instructions given to every implementer said *"The C++ under
  `gpu/` is the specification... TRANSCRIBE IT."*

Two things follow, and they are worth stating because both are commonly
misunderstood:

1. **Changing the results does not help.** Copyright attaches to the
   expression, not to the numbers it produces. Perturbing a summation order so
   the output stops being bit-identical would obscure the provenance, not
   remove it.
2. **The mathematics is not the problem.** The k-epsilon equations are Launder
   & Spalding (1974). Coefficients like `Cmu = 0.09` are facts. Anyone may
   implement a published model. What cannot be copied is OpenFOAM's *expression*
   of it — their code, their structure, and the implementation choices that are
   theirs rather than the literature's.

So the fix is not obfuscation. It is reimplementation from sources that permit
it, with the provenance documented.

---

## 2. Dependency licences — all clear

**Re-verified against `rust/Cargo.lock` as it stands, not against memory.**
27 packages in the lock file (the crate itself plus 26 dependencies, direct
and transitive), every licence read from the crate's own `Cargo.toml` in the
local registry:

| Dependency (from `Cargo.lock`) | Licence | MIT-compatible |
|---|---|---|
| cudarc | MIT OR Apache-2.0 | yes |
| serde, serde_core, serde_derive, serde_derive_internals, serde_json, serde_path_to_error | MIT OR Apache-2.0 | yes |
| schemars, schemars_derive | MIT | yes |
| jsonc-parser | MIT | yes |
| thiserror, thiserror-impl, anyhow | MIT OR Apache-2.0 | yes |
| proc-macro2, quote, syn, cfg-if, dyn-clone, itoa, ref-cast, ref-cast-impl, windows-link | MIT OR Apache-2.0 | yes |
| zmij | MIT | yes |
| libloading | ISC | yes |
| memchr | Unlicense OR MIT | yes |
| unicode-ident | (MIT OR Apache-2.0) AND Unicode-3.0 | yes |
| AMGX (NVIDIA) | BSD-3-Clause | yes — optional, `amgx` feature, off by default |
| CUDA toolkit, cuFFT, cuBLAS | NVIDIA EULA, linking permitted | yes — same as any CUDA program |
| **OpenFOAM** | **GPL-3.0** | **no** |

**No GPL, LGPL or AGPL dependency appears anywhere in the graph, direct or
transitive.** The whole set is MIT / Apache-2.0 / ISC / Unlicense /
Unicode-3.0 / BSD-3-Clause. There is no release blocker here.

The one entry that IS a problem is OpenFOAM, and it was never linked against
— it was copied from. That is worse, and it was fixed: the derived numerical
core is gone (§"What phase 0 actually removed"), and every one of the 105
source files in `rust/` now carries the line *"No GPL-licensed source was
consulted."*

---

## 3. File-by-file status

### 3a. Derived from OpenFOAM — must be reimplemented

The numerical core. ~24,500 lines carry an explicit OpenFOAM provenance note.

| Area | Files |
|---|---|
| Specification | `gpu/SPEC.md`, `rust/PORT.md`, `rust/BUOYANT.md` |
| Reference C++ | all of `gpu/` |
| Discretisation | `rust/cuda/{fv,ldu,field}.cu`, `rust/src/{fv,ldu_ops,field_ops}.rs` |
| Linear solvers | `rust/cuda/solver.cu`, `rust/src/solver.rs` |
| Turbulence | `rust/cuda/{turbulence,wallfunctions}.cu`, `rust/src/{turbulence,wallfunctions}.rs`, `rust/src/models/*` |
| Momentum / SIMPLE | `rust/cuda/{momentum,simple}.cu`, `rust/src/{momentum,simple}.rs` |
| Mesh geometry | `rust/src/mesh/geometry.rs` |
| CPU reference | `rust/src/reference.rs` |
| Field BC mapping | `rust/src/field_setup.rs`, `rust/src/field.rs` |

### 3b. Clean — original work, keeps its authorship

These were designed here, not transcribed. They are the parts worth keeping.

| Area | Files | Why it is clean |
|---|---|---|
| GPU plumbing | `rust/src/device.rs`, `error.rs`, `build.rs` | cudarc wrapper, CUDA-graph capture, MSVC/nvcc build glue — no OpenFOAM analogue |
| Pressure backends | `rust/src/pressure/*`, `rust/cuda/pressure.cu` | The backend trait, the measuring selector, the cuFFT direct Poisson solver. OpenFOAM has none of these |
| Potential flow | `rust/src/potential_flow.rs` | Original — solving a Laplace problem to seed a conservative flux |
| Mesh generation | `rust/src/blockgen.rs` | Original generator. It *writes* OpenFOAM's file format, which is interoperability, not derivation |
| Tooling | `tools/*.py`, `rust/src/bin/{graph_bench,dispatch_bench,probe}.rs` | Volume renderer, report builder, benchmarks |
| Data structures | `rust/src/mesh.rs` (the types), `rust/src/ldu.rs` (CSR export) | The cell→face CSR and the LDU→CSR permutation are our design |

### 3c. Format interoperability — fine to keep, worth an audit

`rust/src/io/*` reads and writes OpenFOAM's ASCII case format. **File formats
are not copyrightable**, and interoperability is a legitimate purpose; the
parser here was written fresh rather than ported from OpenFOAM's `ISstream`.
Keep it, but re-read the headers and delete any comment that cites an OpenFOAM
source file as the specification — the format itself, observed from data files,
is the specification.

### 3d. Must not appear in the published repository

| Path | Why |
|---|---|
| `upstream/OpenFOAM-Foundation-12` | GPL-3.0 source |
| `upstream/OpenFOAM-v2606` | GPL-3.0 source |
| `upstream/_caseClashRecovered/` | recovered GPL-3.0 source files |
| `gpu/` | the transcribed C++ |

Shipping any of these inside an MIT-licensed repository would relicense GPL
code, which is exactly the thing to avoid. Keep them in a separate,
clearly-GPL working tree for reference, or delete them from the published one.

### 3e. The documentation

`docs/01`–`docs/03` describe OpenFOAM: what components exist, how they classify
for GPU porting, and their equations. Facts about a work are not the work, and
a catalogue of names and a portability analysis are original commentary.

`docs/04-porting-roadmap.md` was the one to be careful with: it contained
equations whose stated provenance was *"transcribed from the original source"*.
Most of that content is published mathematics and would be identical if taken
from the papers — but the citation must point at the paper, not at
`upstream/.../kEpsilon.C`. **DONE, by removal rather than re-sourcing.** That
file is no longer in this tree (see §"What phase 0 actually removed" below,
and `docs/README.md`); `rust/SPEC-LIT.md`, in which every equation is cited to
a paper or textbook, is the reimplementation specification instead.

---

## 4. What a defensible reimplementation looks like

A strict clean room means implementers who have never seen the original,
working from a specification written by people who have. That is not achievable
retroactively here — this project read the OpenFOAM sources. What *is*
achievable, and is standard industry practice:

1. **Specify from the literature.** Every equation cited to a paper or
   textbook, verified against it. No OpenFOAM file paths anywhere in the spec.
2. **Implement without the source in reach.** Implementers work only from the
   literature spec, with `upstream/` deleted or off-limits.
3. **Do not reproduce OpenFOAM-specific choices.** Where OpenFOAM does
   something the literature does not prescribe — its particular bounding
   heuristics, its wall-cell area-weighted blending, the exact ordering inside
   `setValues` — either derive an equivalent from first principles or design
   our own, and say which.
4. **Validate against physics, not against OpenFOAM.** Method of manufactured
   solutions, analytical solutions, and published benchmarks — Ghia et al.
   (1982) lid-driven cavity, Moser–Kim–Mansour (1999) channel DNS, ERCOFTAC
   cases. Never "matches OpenFOAM to 1e-13", which is evidence of the wrong
   thing.
5. **Document provenance per item**, so the record exists before anyone asks.

This substantially reduces risk and is how permissively-licensed
reimplementations are normally done. It is not a courtroom guarantee. **If the
commercial stakes are material, have a lawyer review the plan and the result.**
I am not one.

---

## 5. Primary sources for the reimplementation

The mathematics is all published. Nothing needed here is unique to OpenFOAM.

| Component | Source |
|---|---|
| Finite-volume discretisation, unstructured | Jasak, *Error Analysis and Estimation for the Finite Volume Method*, PhD thesis, Imperial College (1996) |
| The same, textbook treatment | Moukalled, Mangani & Darwish, *The Finite Volume Method in CFD*, Springer (2016) |
| | Ferziger & Perić, *Computational Methods for Fluid Dynamics* |
| SIMPLE, staggered/collocated | Patankar, *Numerical Heat Transfer and Fluid Flow* (1980) |
| SIMPLEC | Van Doormaal & Raithby, *Numer. Heat Transfer* 7 (1984) |
| PISO | Issa, *J. Comput. Phys.* 62 (1986) |
| Rhie–Chow interpolation | Rhie & Chow, *AIAA J.* 21 (1983) |
| k-epsilon | Launder & Spalding, *Comput. Methods Appl. Mech. Eng.* 3 (1974) |
| k-omega | Wilcox, *Turbulence Modeling for CFD*, DCW Industries |
| k-omega SST | Menter, *AIAA J.* 32 (1994); Menter, Kuntz & Langtry (2003) for the 2003 revision |
| Spalart–Allmaras | Spalart & Allmaras, *La Recherche Aérospatiale* 1 (1994) |
| Realizable k-epsilon | Shih et al., *Computers & Fluids* 24 (1995) |
| RNG k-epsilon | Yakhot & Orszag, *J. Sci. Comput.* 1 (1986) |
| Wall functions | Launder & Spalding (1974); Spalding, *J. Appl. Mech.* 28 (1961) |
| Smagorinsky | Smagorinsky, *Mon. Weather Rev.* 91 (1963) |
| WALE | Nicoud & Ducros, *Flow Turbul. Combust.* 62 (1999) |
| Deardorff | Deardorff, *Boundary-Layer Meteorol.* 18 (1980) |
| TVD/NVD limiters | Sweby, *SIAM J. Numer. Anal.* 21 (1984); Leonard, *CMAME* 88 (1991) |
| VOF with interface compression | Weller, OpenCFD Technical Report (2008) — check its licence separately |
| | or Hirt & Nichols, *J. Comput. Phys.* 39 (1981) for plain VOF |
| CSF surface tension | Brackbill, Kothe & Zemach, *J. Comput. Phys.* 100 (1992) |
| Krylov solvers | Saad, *Iterative Methods for Sparse Linear Systems*, 2nd ed. (2003) |
| BiCGStab | van der Vorst, *SIAM J. Sci. Stat. Comput.* 13 (1992) |
| Algebraic multigrid | Stüben, *J. Comput. Appl. Math.* 128 (2001) |
| FFT Poisson solvers | Swarztrauber, *SIAM Review* 19 (1977) |
| Low-Mach formulation | Rehm & Baum, *J. Res. NBS* 83 (1978); the FDS Technical Reference Guide |

Permissively-licensed code that may be read directly (verify each before use):

| Project | Licence | Useful for |
|---|---|---|
| SU2 | LGPL-2.1 — **linking only, do not copy source** | — |
| Nek5000 | BSD-3-Clause | spectral element, GPU residency patterns |
| PyFR | BSD-3-Clause | high-order GPU methods |
| AMGX | BSD-3-Clause | already a dependency |
| Ginkgo | BSD-3-Clause | GPU linear algebra |
| deal.II | LGPL | reference only |
| FDS | US Government work, public domain | low-Mach, FFT pressure solve, fire-specific models |

**FDS is public domain** (a US NIST work) and is the closest match to the fire
problem being solved here. That makes it the single most valuable permissive
reference available for this project.

---

## 6. Plan and progress

| Phase | Work | Gate | Status |
|---|---|---|---|
| 0 | This audit. Remove every GPL file from the tree. | the MIT tree contains no GPL file | **done** |
| 1 | Write `rust/SPEC-LIT.md`: every equation cited to a paper, verified against it. | no OpenFOAM path appears anywhere in it | **done** |
| 2 | Reimplement §3a from `SPEC-LIT.md` only, with no GPL source in reach. | builds and passes §10 validation | **done** |
| 3 | Rebuild validation on MMS, analytical solutions and published benchmarks. | second-order convergence shown without reference to any other code | **done** |
| 4 | `LICENSE` = MIT. Per-file provenance. `NOTICE` for BSD dependencies. | ready to publish | **done** — see `rust/PROVENANCE.md` |

### What phases 2 and 3 produced

13 Rust modules and 8 CUDA units, written from `SPEC-LIT.md` and the papers it
cites. Each carries a header naming its sources and the line *"No GPL-licensed
source was consulted."*

```
ofgpu-validate     168 / 168 checks passed
cargo test         321 tests passed, 0 failed
```

Observed order of convergence, method of manufactured solutions: **2.10** on a
3-D graded mesh, **1.91** sheared, **2.07** in 2-D with empty patches.

Lid-driven cavity against Ghia, Ghia & Shin (1982), 80x80: worst centreline
difference **0.0088** at Re = 100 and **0.0067** at Re = 400.

Not one test compares against another CFD code.

**The rewrite found a real bug the old code did not have a test for.** The
kernels compared the mesh's `b_kind` — which holds `PatchKind`, where
`Empty = 2` — against `BcKind::Empty = 5`. Two enums, both with a variant
called `Empty`, numbered differently. It compiles, it runs, and a 2-D case
silently integrates flux through its own front and back planes. The check that
caught it ("an empty patch carries no flux") exists because the validation was
rebuilt from the specification rather than carried over, and it is the one
check that isolates the rule: with the usual equal-and-opposite boundary values
the two planes cancel whether they were skipped or not, so every other check
passes either way.

### What phase 0 actually removed

| Path | Size | Why |
|---|---|---|
| `upstream/OpenFOAM-Foundation-12` | 249 MB | GPL-3.0 source |
| `upstream/OpenFOAM-v2606` | 318 MB | GPL-3.0 source |
| `upstream/_caseClashRecovered/` | — | recovered GPL-3.0 source files |
| `gpu/` | 45 MB | the transcribed C++, including `gpu/SPEC.md` |
| `rust/{PORT,BUOYANT}.md` | — | porting instructions that named OpenFOAM files as the specification |
| the derived Rust and CUDA numerics | ~24,500 lines | listed in §3a |
| `docs/04-porting-roadmap.md` | 619 KB | equations cited to GPL files by line number, including OpenFOAM-specific implementation choices the literature does not prescribe |

Also done in phase 0:

- **The `FoamFile` ASCII banner was replaced** in 338 case files and in both
  generators. The old banner was verbatim text reading *"OpenFOAM: The Open
  Source CFD Toolbox"*, which is both copied text and a false claim about who
  wrote the file. The replacement states what the format is and that ofgpu is
  independent.
- **OpenFOAM-as-authority comments were rewritten.** A comment saying *"this is
  what OpenFOAM does"* about a numerical choice now cites the paper, or says
  plainly that the choice is ours. Comments about the **case file format** were
  kept: they are accurate, they are necessary to explain the reader and writer,
  and a file format is not a work of authorship.
- **`Cargo.toml` licence changed** from `GPL-3.0-or-later` to `MIT`.
- **`docs/01` and `docs/02` had their `upstream/` path prefixes rewritten** to
  name the distribution instead, so a citation no longer implies this
  repository ships a copy of the source.

### What phase 2 keeps

§3b stands as it is — it was ours already, and it is a larger fraction of the
value than the line counts suggest: the CUDA-graph residency, the pressure
backend selector and the cuFFT direct solver have no OpenFOAM equivalent at all.

`rust/src/field_setup.rs` was rewritten from `SPEC-LIT.md` §4 rather than
reimplemented by an agent, because it is the boundary between the case reader
and the device field representation, and that representation is our own design.

### The reference that *is* permitted

`reference/fds` holds the NIST Fire Dynamics Simulator. Its licence states
plainly that *"software developed by NIST employees is not subject to copyright
protection within the United States"*. It may be read and adapted; where it is,
the file says so and acknowledges NIST. It is the closest permissive match to
the buoyant-plume problem this project solves, and it implements the Rehm & Baum
low-Mach formulation directly.

It is **not** part of what ofgpu distributes — it is a separate clone, and
`.gitignore` excludes it.
