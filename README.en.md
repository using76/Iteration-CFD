# meteor-cfd

**A GPU-resident finite volume CFD solver**

Iterations Co., Ltd. · in collaboration with Meteo Simulation Co., Ltd. · Rust host, CUDA kernels · [한국어](README.md)

---

## Overview

meteor-cfd is an unstructured finite volume CFD solver designed so that the entire time-integration loop stays on the GPU. Once the mesh and fields are uploaded, no device allocation and no field transfer to the host occur inside the loop. It is **single-GPU only**, and what it is for is incompressible and low-Mach flow — RANS/LES/hybrid turbulence, buoyant plumes, variable-density low-Mach flow, two-phase VOF, conjugate heat transfer, surface-to-surface radiation, Lagrangian sprays, and ventilation and data-centre airflow with fan curves and porous jumps. The numerical core is implemented directly from published literature, and every formulation is specified in [`rust/SPEC-LIT.md`](rust/SPEC-LIT.md) with a citation to its original paper. Validation uses the method of manufactured solutions, analytical solutions and published benchmarks. **The comparison is not yet sufficient, and help with it is very welcome.** A Rust 1.85 host with CUDA C++ kernels; double precision by default, single via the `single` feature (a supported configuration; its section-by-section measurements, open defects, speed and memory are in SPEC-LIT §112.4–112.6); NVIDIA GPUs. The direct dependencies are seven crates — `cudarc`, `jsonc-parser`, `schemars`, `serde`, `serde_json`, `serde_path_to_error`, `thiserror` (`rust/Cargo.toml`, `[dependencies]`; `anyhow` joins for the build script) — and the resolved graph is 27 packages (this crate plus 26), each named in [`NOTICE`](NOTICE) with the terms it ships under; AMGX stays optional.

---

## Licence

| Use | Terms |
|---|---|
| Personal study, hobby and amateur work | **Free** |
| Educational institutions — teaching, coursework, lab work | **Free** |
| Universities and their institutes and laboratories | **Free** |
| Government institutions | **Free** |
| Public safety, public health and environmental protection organisations | **Free** |
| Charitable organisations | **Free** |
| Research whose purpose is public — fire, medicine, safety, disaster, environment | **Free**, including at a government-funded institute |
| **Product or technology-transfer R&D at a research institute** | **Thirty-day trial**, then a commercial licence |
| Any other commercial use | **Thirty-day trial**, then a commercial licence |

**"Public" describes the purpose of the work, not the owner of the body doing
it.** Research aimed at fire and rescue, medicine, public health, safety,
disaster response or environmental protection is free wherever it is done,
including at a government-funded research institute. Research and development
aimed at a particular product or at technology to be transferred to industry —
electric vehicles, rail and propulsion, aircraft engines and the like — is
commercial use even when a government-funded institute performs it. One test
applies: **if the work carries a technology fee, or is done for a company or
under a programme that hands the result to one, it needs a licence** after the
thirty-day trial. Universities and schools, and the institutes and laboratories
that belong to them, are educational institutions and stay free.

**The thirty-day trial is per company, not per person.** If you use this for
work, your company gets one trial period covering all personnel. CFD is not
software anyone buys without running it on their own cases, their own meshes
and their own hardware first, and the trial exists for exactly that.

Contributing feedback, changes or additions back under a standardised public
licence — Blue Oak 1.0.0, Apache 2.0, MIT or two-clause BSD — does not count as
commercial use.

If your situation is unclear, please ask. Where the substance of the use is not
commercial, a free or reduced licence will be considered.

**Licensing enquiries: simul@msimul.com**

The copyright holder and licensor is 주식회사 이터레이션즈 (Iterations Co., Ltd.);
주식회사 메테오시뮬레이션 (Meteo Simulation Co., Ltd.) is a collaborator on and
contributor to the work. The full
text is in [`LICENSE`](LICENSE), what a commercial licence covers is in
[`LICENSING.md`](LICENSING.md), and third-party notices are in
[`NOTICE`](NOTICE).

> Prosperity is a noncommercial licence, not an open-source one. It is not
> OSI-approved and it restricts commercial use, so this project does not call
> itself open source.

---

## Status

**Each line below is a dated measurement taken on the machine of record** — an NVIDIA GeForce RTX 5070 Ti (sm_120), CUDA 13.3, double precision — and names the date it was measured and where it is recorded.

```
cargo test --release   1,774 passed, 0 failed, 6 ignored   (all targets, 18 suites; 2026-09-08)
                       1,645 passed, 0 failed, 4 ignored   (the lib crate alone; 2026-09-08)
```

<!-- validate-run:status:en begin -->
```
ofgpu-validate (f64)   1634 / 1708 checks passed in 95 sections (2026-10-20, one unfiltered run, 3,533 s; SPEC-LIT §110.5)
                       77 sections hold, 16 miss, 2 have no rows, 0 NOT RUN
                       then 35 gates registered MISSES and 37 OPEN, named below
ofgpu-validate (f32)   1431 / 1655 checks passed in 95 sections (2026-10-20, 5 runs, 5,087 s; SPEC-LIT §110.5)
                       49 sections hold, 43 miss, 2 have no rows, 1 NOT RUN
                       then 42 gates registered MISSES and 37 OPEN, named below
```
<!-- validate-run:status:en end -->

That list is not maintained by hand: it is **generated** from the registry each gate enters at the point it reports its own verdict (SPEC-LIT §69). Printing a verdict and registering one are the same call, so every registered verdict is named on every run and one more could not fail to appear. The block above is generated too: `python tools/validate_record.py check README.en.md` holds its numbers equal to the record of one full run in `docs/validate-run/`, the run SPEC-LIT §110.5 records section by section. **Every row the run counts and does not name as failing passes, and the failing rows sit in the sections the guidebook's §11 table marks. That is a different statement from "this project reproduces every published benchmark it compares against", and the two must not be confused.**

---

## Technical guidebook

Installing, a first case, meshing, choosing a solver, reading convergence,
looking at results, and the causes of the errors you will meet — the one
document about **use** is [`docs/GUIDEBOOK.en.md`](docs/GUIDEBOOK.en.md). In
Korean: [`docs/GUIDEBOOK.md`](docs/GUIDEBOOK.md).

If you read one section, read §6, Choosing a solver: `ofgpu-k-epsilon`,
`ofgpu-k-omega` and `ofgpu-sa` solve **the turbulence equations only, on a
frozen velocity field**.

---

## Build and run

Requirements: Rust 1.85 or newer, Visual Studio 2022 (C++ workload), CUDA Toolkit 13.x. `build.rs` sets up the MSVC environment through `vcvars64.bat` and compiles every `.cu` to CUBIN rather than PTX.

```powershell
cd rust
cargo build --release
cargo run --release --bin ofgpu-generate-mesh -- channel ..\cases\channel 200 120 1
cargo run --release --bin ofgpu-k-epsilon     -- ..\cases\channel -iters 4000 -check 400
```

The whole validation suite is `cargo run --release --bin ofgpu-validate`. The other twenty-one executables (`ofgpu-lowmach`, `ofgpu-compressible`, `ofgpu-vof`, `ofgpu-cht`, `ofgpu-datacentre`, `ofgpu-decompose`, the benchmarks), the case file format and the command-line options are in the **user guide**. Cases are read and written either as a single JSONC file or as an OpenFOAM ASCII case directory — the latter for interoperability with existing tools such as ParaView and `foamToVTK`; meteor-cfd links against no part of OpenFOAM and contains none of its source.

---

### Sample case — race-car external aerodynamics

One geometry and two commands live in [`cases/`](cases/). No mesh and no
results: together they are over a gigabyte, and producing them is what this
program is for.

```powershell
cd cases
.\racecar.cmd     # a few minutes of meshing (all cores, in parallel) + about 40 s of solving (GPU)
```

Details in [`cases/racecar.md`](cases/racecar.md).

### MCP — hand the solvers to an assistant

Claude Code, Claude Desktop, or anything else that speaks the Model Context
Protocol can run these solvers directly. One file, no dependencies.

```powershell
claude mcp add ofgpu -- node mcp\server.mjs
```

The guide, the tool list, and what the server refuses to do:
[`mcp/README.md`](mcp/README.md).

---

## What it can do

| Area | Supported |
|---|---|
| Discretisation | Gauss linear, upwind, linearUpwind, cubic, QUICK, Gamma, blended; six TVD limiters; Green–Gauss and least-squares gradients with cell- and face-limiters (Barth–Jespersen, Venkatakrishnan); over-relaxed non-orthogonal correction; steadyState, Euler, BDF2, local time stepping |
| Pressure–velocity | SIMPLE, SIMPLEC, PISO, PIMPLE. Rhie–Chow interpolation, with body forces treated at the face rather than interpolated from cell values |
| Turbulence | RANS: standard, realizable and RNG k-ε, Wilcox k-ω, Menter SST, Launder–Sharma low-Re, Spalart-Allmaras (four variants). LES: Smagorinsky, WALE, Deardorff. Hybrid: DES97, DDES, IDDES. Transition: k-ω SST-LM, k-ω SST-γ (Menter 2015, Galilean invariant). Wall treatment: standard and continuous wall functions, `lowRe` integration, the Jayatilleke thermal wall function, roughness |
| Multiphase and transport | VOF (interface compression, Zalesak FCT bounding, CSF surface tension, static, hysteresis and dynamic contact angles), multicomponent species, six generalised-Newtonian viscosity models, Darcy–Forchheimer porosity, non-Boussinesq buoyancy |
| Conjugate heat transfer | Solid regions, contact resistance, harmonic-mean interface conductivity, a fluid region with up to one inlet and one outlet |
| Ventilation and data centre | Fan performance curves (AMCA 210 corrections), porous jumps, moist air (Hyland–Wexler), RCI, RTI, SHI and RHI metrics |
| Lagrangian parcels | SoA pool, Schiller–Naumann drag, two-way coupling, evaporation and vapour coupling, the Bai-Gosman wall-impact map — **as a library API only** |
| Linear solvers | PBiCGStab, PCG; Jacobi, multi-colour DIC and DILU; a cuFFT direct Poisson backend and AMGX (optional feature); backend chosen automatically on applicability, accuracy and measured time |
| Mesh and I/O | Block meshes with grading, castellated STL carving, cut cells, Gmsh v4.1, multiple cyclic pairs, `empty`/`symmetry` constraints, 2:1 adaptive refinement (wired to no solver); JSONC cases with a generated schema, OpenFOAM ASCII, VTU, NanoVDB/OpenVDB, USD, double-precision restart |

- **From a STEP geometry to a mesh, and to Fluent.** A STEP file from CAD becomes a Gmsh mesh with `tools/mesh/step_mesh.py` (one config JSON), and `ofgpu-convert-mesh` writes that .msh into a case's `constant/polyMesh` and an ANSYS Fluent mesh in one go. The same two steps sit in the Studio as the `mesh_from_step` and `mesh_to_fluent` tools — [Guidebook §5](docs/GUIDEBOOK.en.md#5-making-a-mesh).

## What it cannot do

- **No MPI and no multi-GPU.** Decomposition, halos, decomposition-invariant reductions and distributed PCG/PBiCGStab are all implemented and gated (§71–§73), but they run in one process on one card. No communication library is linked and no strong-scaling number is published.
- **Compressible flow is new, and several of its gates miss.** `ofgpu-compressible` solves the all-speed pressure-based equations (SPEC-LIT §122-§127). Gates CMP-LM, CMP-COU and CMP-OBL hold (CMP-OBL without meeting its field stop). CMP-SOD's PISO row, CMP-AC, CMP-NOZ, CMP-RING and CMP-RAE miss, and `ofgpu-validate` names each one (§127.6). `cases/sodShockTube.jsonc` and `cases/nozzle` are its two shipped cases.
- **No finite-rate (Arrhenius) chemistry** — no stiff ODE integrator, no Jacobian, no reaction mechanism.
- **Crank–Nicolson cannot be used with an under-relaxed equation** — it reports the reason as an error rather than silently falling back to Euler.
- **Radiation here is surface-to-surface only.** §49/§50 solve grey diffuse exchange across a *transparent* medium; nothing in the volume absorbs, emits or scatters. The `radiationModel` selector recognises `viewFactor` (and `s2s`, the native spelling of the same model) and nothing else — any other value is an **unrecognised setting**, refused with the recognised set beside it rather than substituted for (§13.4). A `constant/radiationProperties` that exists and names no `radiationModel` is refused for the same reason: there is no default for the entry, and a reader that guessed would answer a question the case never asked.
- **Surface-to-surface radiation, species transport and Lagrangian sprays have no case format.** All three are specified and gated as library APIs, but no driver binary reads an enclosure, a species set or a spray out of a case file (§50.12, §13.4.2).
- **The one preset for which `ofgpu-generate-mesh` writes neither `p` nor `T` is `damBreak`.** Every other preset — `channel`, `cavity`, `step`, `big`, and the buoyant pair `plume`/`room` — gets both written into `0/` on both generation paths, block and cut cell, so a momentum driver runs a freshly generated case as it stands. `damBreak` is the two-phase path and its `0/` holds `alpha.water` and `p_rgh`; pointing `ofgpu-lowmach` at it still earns `has no p field`. The cut-cell path still writes no `nuTilda` for `ofgpu-sa` — that field is added by the block path only.
- **Parcels do not create splash children, there is no film transport, and parcels do not absorb radiation** (§78.11, §68.13).
- **Adaptive refinement is wired to no solver** — face fluxes are not transferred and there is no post-adaptation pressure projection.
- **No tetrahedral-to-polyhedral dual-mesh conversion.** Reading a tet mesh and solving on it work — the Gmsh MSH 4.1 reader (`io/msh.rs`) accepts Tet, Hex, Prism and Pyramid, and the mesh model is face-based, so a tet mesh is already handled as a general polyhedral mesh once read. What does not exist is the dual-mesh operation that merges the tetrahedra around each node into one polyhedron — what Fluent's polyhedral conversion and OpenFOAM's `polyDualMesh` do to cut cell counts and improve gradient reconstruction on tet meshes; no `polyDualMesh`, `dual_mesh` or `dualMesh` symbol exists in the tree. It is not a small addition: SPEC-LIT already assumes every face planar and every cell convex ("assumes every face planar"), and the faces of a dual mesh are in general neither, so the work starts at revisiting that assumption, and at deciding the boundary treatment of the dual — a boundary node's dual cell is open — before it is a converter.
- **AMGX is off by default**, and with it off the selector still reports AMGX explicitly as "unavailable".
- **It is not claimed that the DES family reproduces a published separated-flow statistic**, and Spalart-Allmaras's TMR flat-plate gate is not run (§57.12, §56.11).

**And the gates that miss or are open.** `ofgpu-validate` names every one on every run. The table is the f64 run's list as it printed it, MISSES first and then OPEN; the f32 column is the single build's verdict for the same registration. `tools/validate_record.py` generates it from `docs/validate-run/`. A gate name that repeats was registered once per leg.

<!-- validate-run:gates:en begin -->
| # | Gate | f64 | f32 | Compared against: the verdict |
|---|---|---|---|---|
| 1 | SPEC-LIT S60.5 Gate 5 | MISSES | MISSES | Kaminski & Prakash (1986), conjugate natural convection in a square enclosure - through the open-access SECONDARY table of Belazizia et al. (2012), the primary being paywalled and never read - bar 3 % at every conductivity ratio: worst disagreement 7.11 %, and it is at the SMALLEST conductivity ratio - the CONDUCTION-DOMINATED end - shrinking to 0.07 % at Kr = 10; reference-limited at Kr <= 1, SPEC-LIT 115 (O1, D-VAL-4) |
| 2 | SPEC-LIT S68.12 Gate 68-C | MISSES | MISSES | Theobald (1981), 90 hose streams, maximum throw, at the shape of the FDS Validation Guide's own metric for this quantity - +-10 % bias, 30 % scatter: with the gas at rest this solver throws the stream 61.2858 % of the measured distance on average |
| 3 | Gate 95-A cantilever | MISSES | MISSES | Timoshenko & Goodier ch. 3, plane strain, block-coupled (SPEC-LIT 109.6), four meshes r = 2, orders on the finest pair, the tip study on the finest three: 5:1: converged 3/4, p_u 1.42, p_sigma 1.46 (n_y 16 -> 32), finest tip 9.76e-5 |
| 4 | Gate 95-A cantilever | MISSES | MISSES | Timoshenko & Goodier ch. 3, plane strain, block-coupled (SPEC-LIT 109.6), four meshes r = 2, orders on the finest pair, the tip study on the finest three: 10:1: converged 3/4, p_u 1.90, p_sigma 1.45 (n_y 16 -> 32), finest tip 1.52e-4 |
| 5 | Gate 95-A cantilever | MISSES | MISSES | Timoshenko & Goodier ch. 3, plane strain, block-coupled (SPEC-LIT 109.6), four meshes r = 2, orders on the finest pair, the tip study on the finest three: 5:1 GMRES(50): converged 4/4, p_u 1.42, p_sigma 1.46 (n_y 16 -> 32), finest tip 9.83e-5 |
| 6 | Gate 95-A cantilever | MISSES | MISSES | Timoshenko & Goodier ch. 3, plane strain, block-coupled (SPEC-LIT 109.6), four meshes r = 2, orders on the finest pair, the tip study on the finest three: 10:1 GMRES(50): converged 4/4, p_u 1.75, p_sigma 1.45 (n_y 16 -> 32), finest tip 1.67e-4 |
| 7 | SPEC-LIT S110.3 Gate 110-B backward-facing step reattachment (Driver & Seegmiller 1985) | MISSES | MISSES | Driver & Seegmiller (1985) x_r/H = 6.26 +- 0.10, reference/driver_seegmiller_1985/reattachment.txt: study not formed: the dx = 0.1H level stopped - run ended: refused \| ofgpu-lowmach: max Mach number 3.615 at cell 56 (step 1232); the dx = 0.05H level stopped - run ended: refused \| ofgpu-lowmach: max Mach number 1.920 at cell 917604 (step 30); on the finest level that converged, dx = 0.2H, x_r/H = 5.637 against 6.26, E = -0.623 (-9.9% of D), outside the datum's own +-0.10 at that level, with no u_num |
| 8 | SPEC-LIT S110.3 Gate 110-B backward-facing step reattachment (Driver & Seegmiller 1985) | MISSES | MISSES | Driver & Seegmiller (1985) x_r/H = 6.26 +- 0.10, reference/driver_seegmiller_1985/reattachment.txt: class R: the single build's dx = 0.2H run left no record - run ended: refused \| ofgpu-lowmach: max Mach number 10255882.000 at cell 57149 (step 0) |
| 9 | SPEC-LIT 146.2 Gate 78-D measured leg, mundo | MISSES | MISSES | Riboux & Gordillo, Phys. Rev. Lett. 113 (2014) 024507 (arXiv:1401.6943v3): the splash threshold speeds of Fig. 4(c), eight liquids on smooth dry glass in air, band +-20 % per row: 0 of 8 liquids inside the band (146.6), 6 needed; 0 of 17 rows inside (146.5), e from -0.734 to -0.214 |
| 10 | SPEC-LIT 146.2 Gate 78-D measured leg, baiGosman | MISSES | MISSES | Riboux & Gordillo, Phys. Rev. Lett. 113 (2014) 024507 (arXiv:1401.6943v3): the splash threshold speeds of Fig. 4(c), eight liquids on smooth dry glass in air, band +-20 % per row: 1 of 8 liquids inside the band (146.6), 6 needed; 3 of 17 rows inside (146.5), e from -0.396 to +0.690 |
| 11 | SPEC-LIT 144.4 Gate 88-T leg 4, kOmegaSSTLM, transition onset Re_x against Roach & Brierley | MISSES | MISSES | Roach & Brierley's measured C_f(Re_x), the ERCOFTAC T3A key (SPEC-LIT 144.6): kOmegaSSTLM onset Re_x (least C_f) against (144.4a): E = -35.88 % at Re_x 8.8932e4 on the finest mesh, band 15 %, u_val 6.53 % (u_num 6.53 %, u_D 0 %) |
| 12 | SPEC-LIT 144.4 Gate 88-T leg 4, kOmegaSSTLM, laminar C_f against Blasius | MISSES | MISSES | Blasius' 0.664/sqrt(Re_x) (SPEC-LIT 88.7) over the part of the plate the key measures laminar: kOmegaSSTLM worst laminar C_f against Blasius (144.4c): E = +16.29 % at Re_x 7.5000e4 on the finest mesh, band 10 %, u_val 0.62 % (u_num 0.62 %, u_D 0 %) |
| 13 | SPEC-LIT 144.4 Gate 90-T leg 4, kOmegaSSTGamma, laminar C_f against Blasius | MISSES | MISSES | Blasius' 0.664/sqrt(Re_x) (SPEC-LIT 88.7) over the part of the plate the key measures laminar: kOmegaSSTGamma worst laminar C_f against Blasius (144.4c): E = +15.78 % at Re_x 1.0000e5 on the finest mesh, band 10 %, u_val 1.46 % (u_num 1.46 %, u_D 0 %) |
| 14 | SPEC-LIT 143.5 Gate 131-P turbulent pipe, kOmegaSST, Re_tau 576.69, the core defect | MISSES | MISSES | Nikuradse's core defect (U_axis - U_b)/u_tau = 4.07 (Schlichting NACA TM 1218 eq 16.22): kOmegaSST at Re_tau 576.69, defect against 4.07: E = -11.21 % on the finest level (n = 160), bar 10 %, u_val 0.01 % |
| 15 | SPEC-LIT 143.5 Gate 131-P turbulent pipe, kOmegaSST, Re_tau 2358, the log law | MISSES | MISSES | the log law u+ = 2.5 ln y+ + 5.5 over 30 <= y+ <= 0.2 Re_tau (Schlichting NACA TM 1218 eq 16.14): kOmegaSST at Re_tau 2358, log-law deviation: E = 1.230 in u+ on the finest level (n = 160), bar 1.0 in u+, u_val 0.173 |
| 16 | SPEC-LIT 143.5 Gate 131-P turbulent pipe, kOmegaSST, Re_tau 2358, the core defect | MISSES | MISSES | Nikuradse's core defect (U_axis - U_b)/u_tau = 4.07 (Schlichting NACA TM 1218 eq 16.22): kOmegaSST at Re_tau 2358, defect against 4.07: E = -18.39 % on the finest level (n = 160), bar 10 %, u_val 0.04 % |
| 17 | SPEC-LIT 127.6 Gate CMP-SOD Sod shock tube | MISSES | MISSES | the exact Riemann solution of Sod's (1978) shock tube, (127.5)-(127.7b), evaluated live (CMP-07): PISO, 800 cells: worst plateau rho*_L 45.98 % (bar 1 %), shock not found inside the tube (bar 0.5 % of the length); L1(rho) order 0.015 (bar >= 0.6) |
| 18 | SPEC-LIT 127.6 Gate CMP-AC acoustic pulse speed | MISSES | MISSES | linear acoustics: c_0 = sqrt(gamma R_s T_0), (122.4): the solver's default coupling (nOuterCorrectors 1, nCorrectors 2), 400 cells: the pulse travels at 1.1895 c_0 (bar: within 1 %) |
| 19 | SPEC-LIT 127.6 Gate CMP-NOZ quasi-1-D nozzle | MISSES | MISSES | the quasi-1-D keys (127.23)-(127.25), SPEC-LIT §127.2's isentropic and shocked nozzle relations evaluated live by compressible::exact (CMP-06): Gate CMP-NOZ subsonic: M(x) within 0.5 % of the key's M(xi), all 200 cells (SPEC-LIT 127.6): 6.0014e-3 against 0.005; Gate CMP-NOZ supersonic: M(x) within 0.5 % of the key's M(xi), all 200 cells (SPEC-LIT 127.6): no field, the march stopped on a bound; Gate CMP-NOZ supersonic: exit p within 0.5 % of the key at the last cell centre (SPEC-LIT 127.6): no field, the march stopped on a bound; Gate CMP-NOZ shocked: shock position within 1 % of the length, 200 cells (SPEC-LIT 127.6): no field, the march stopped on a bound; Gate CMP-NOZ shocked: p_t2/p_t1 within 1 % of (127.25)'s, the last cell (SPEC-LIT 127.6): no field, the march stopped on a bound; Gate CMP-NOZ shocked: exit p within 0.5 % of the key at the last cell centre (SPEC-LIT 127.6): no field, the march stopped on a bound |
| 20 | SPEC-LIT 127.6 Gate CMP-RING Ringleb flow | MISSES | - | the isentropic Ringleb field of (127.8)-(127.11a), SPEC-LIT section 127.3's smooth answer key evaluated live by compressible::exact (CMP-07): the L2 entropy error reads E = 1.0304e-2 / 1.4008e-2 / 1.9372e-2 (finest first, 96 x 24 / 48 x 12 / 24 x 6); pair orders 0.469 (h/2 vs h), 0.443 (h/4 vs h/2), fitted order 0.456 against the bar 1.5; iterations 10000 / 10000 / 10000 |
| 21 | SPEC-LIT 127.6 Gate CMP-RAE RAE 2822 | MISSES | MISSES | RAE 2822 Case 6 (the corrected conditions M 0.729, alpha 2.31 degrees, Re 6.5e6) of Cook, McDonald & Firmin, AGARD AR-138 (1979), as the NPARC archive reproduces it - the answer key rae2822-cp-case6: the smoke's residual did not fall to a tenth of its first chunk's; f64 L0 stopped on a bound at step 5968 (the first out-of-bounds cell 6312); f64 L1 stopped on a bound at step 5454 (the first out-of-bounds cell 21072); f64 L2 stopped on a bound at step 5376 (the first out-of-bounds cell 75232); single L1 stopped on a bound at step 210 (the first out-of-bounds cell 151) |
| 22 | Gate 116-P | MISSES | MISSES | Hagen-Poiseuille, 8 mu L Q / (pi R^4) (116.27): Hagen-Poiseuille on the L1 pipe at Re 100: f64 +15.49 %, single stopped (diverged), the solve 78.6 s, the single-build mesh refused (the known A28) |
| 23 | SPEC-LIT 144.7 Gate 88-T leg 3, kOmegaSSTLM, T3B laminar C_f against Blasius | MISSES | MISSES | Blasius' 0.664/sqrt(Re_x) (SPEC-LIT 88.7) over the part of the plate the key measures laminar: kOmegaSSTLM worst laminar C_f against Blasius (144.4c): E = +58.30 % at Re_x 3.9360e4 on the finest mesh, band 10 %, u_val 8.97 % (u_num 8.97 %, u_D 0 %) |
| 24 | SPEC-LIT 144.7 the T3A- inflow fit against ERCOFTAC's measured free-stream Tu(x) | MISSES | MISSES | the ERCOFTAC key's measured free-stream Tu(x), its TUFS column: fitted decay misses the measured Tu(x) by 3.63 % at Re_x 7.7200e5 (band 3 %); the run uses this inflow anyway (an input check, SPEC-LIT 144.7) |
| 25 | SPEC-LIT 144.7 Gate 88-T leg 3, kOmegaSSTLM, T3A- turbulent C_f against Roach & Brierley | MISSES | MISSES | Roach & Brierley's measured C_f(Re_x), the ERCOFTAC T3A- key (ercoftac-t3am, SPEC-LIT 144.7): kOmegaSSTLM worst turbulent C_f against the station's (144.4b): E = +132.12 % at Re_x 2.0220e6 on the finest mesh, band 10 %, u_val 12.81 % (u_num 12.65 %, u_D 2 %) |
| 26 | SPEC-LIT 144.7 Gate 88-T leg 3, kOmegaSSTLM, T3A- laminar C_f against Blasius | MISSES | MISSES | Blasius' 0.664/sqrt(Re_x) (SPEC-LIT 88.7) over the part of the plate the key measures laminar: kOmegaSSTLM worst laminar C_f against Blasius (144.4c): E = +27.55 % at Re_x 1.2097e6 on the finest mesh, band 10 %, u_val 0.63 % (u_num 0.63 %, u_D 0 %) |
| 27 | SPEC-LIT 157.5 Gate DST-LEEDS Leeds fan-stirred cornflour bomb | MISSES | MISSES | Bradley, Chen & Swithenbank (1989) via Huang et al. (2020) Fig. 9, key huang2020-leeds-st-b: 5 of the gate's rows leave the bands of §157.5 |
| 28 | SPEC-LIT 157.6 Gate DST-VES closed vessels: thin-flame spheres and the REMBE closed stage | MISSES | MISSES | Huang, Bloching & Lipatnikov (2022) Fig. 10, key huang2022-rembe-p-closed: 14 of the gate's rows leave the bands of §157.6 |
| 29 | SPEC-LIT 133.12 Gate CHR-13 1-D freely propagating CH4-air flame, BFER phi 1.0 | MISSES | MISSES | the plan's S_L = U_in - dx_f/dt within 10 % of Konnov et al. 2018 (closed, so that comparison is NOT RUN); a level that stops before its end time gives no S_L at all: 3 of 3 levels stopped before 6 ms on the reactor's refusal (100 µm at 0.0006 s, 50 µm at 0.0006 s, 25 µm at 0.0006 s), so no S_L was measured |
| 30 | SPEC-LIT 134.8 Gate CHR-16 Sandia Flame D, EDC + DRM19, k-epsilon C1 1.6, REDUCED | MISSES | MISSES | the plan's REDUCED gate: 300 iterations, finite fields, element and mass budgets within 1e-6: 40 of 300 steps: chemistry: EDC refused 1 cell(s); the first is cell 244 with status 8 (BELOW_NEGATIVE_ATOL) (SPEC-LIT §134.8); the worst inventory budget is C at 2.354e-1 against 1e-6 |
| 31 | SPEC-LIT 136.7 Gate CHR-SLAB the isothermal grey slab | MISSES | MISSES | the exact slab 1 - 2 E_3(tau), (135.6) of SPEC-LIT 135.2: S8 N 256 tau 0.1: e_256 +2.1713e-2 outside 2 % on both walls, S8's own angular limit (136.13) e_lim +2.2191e-2 above 2 % |
| 32 | SPEC-LIT 136.8 Gate CHR-CUBE scattering slab and cube | MISSES | MISSES | (136.6), the cube's wall-centre flux by the exact angular integral, SPEC-LIT 136.3: S8 kL 0.1 24^3: e -3.1005e-2 outside 3 %, by the step scheme's spatial error |
| 33 | SPEC-LIT 159.11 Gate DST-B-LAM Phase B laminar dust flame: the fine-particle limit and the size order | MISSES | MISSES | (i) \|S_L,B(1 um)/S_L,gas - 1\| within 2 % and (ii) S_L(d_{j+1}) <= S_L(d_j) along d = 5, 10, 20, 40 um - the gate's own bands; no published measurement: 1 of the gate's rows leave the bands of 159.11 |
| 34 | SPEC-LIT 159.12 Gate DST-B-VES Phase B cloud in a closed 20 L sphere: the AICC end pressure and the thin-flame half time | MISSES | MISSES | the cloud's host AICC (159.21) and the thin-flame half time (159.30) with S_u(p) of the 1-D Phase B tube: 7 of the gate's rows leave the bands of 159.12 |
| 35 | SPEC-LIT 147.2 Gate 68-C coupled leg: Theobald test 3 through ofgpu-lowmach, replayed | MISSES | MISSES | Theobald (1981) test 3's measured maximum throw, 24.4 m, at the shape of the FDS Validation Guide's own metric for this quantity - +-10 % bias, 30 % scatter: 3 of the gate's rows leave the bars of 68.12 |
| 36 | SPEC-LIT S32.4 verdict 2 (Reynolds analogy), wall-function leg | OPEN | OPEN | Gnielinski (1976) evaluated at this leg's own MEASURED wall friction factor, +-10 % band: WF Nu is +34.4% of Gnielinski at the MEASURED f and +15.2% at the viscous form of it - outside the band on both |
| 37 | SPEC-LIT S32.4 verdict 2 (Reynolds analogy), wall-function leg, at PrtModel KaysCrawford (opt-in, S37) | OPEN | OPEN | Gnielinski (1976) evaluated at this leg's own MEASURED wall friction factor, +-10 % band, on the PrtModel KaysCrawford record: WF Nu is +34.0% of Gnielinski at the MEASURED f, at PrtModel KaysCrawford (opt-in, S37) - outside the band |
| 38 | SPEC-LIT S32.4 verdict 1 (absolute prediction), resolved leg | OPEN | OPEN | Gnielinski (1976) evaluated at the Petukhov smooth-PIPE f, +-10 % band: resolved leg Nu is +11.9% of it - outside the band by 1.9 points, against an energy-balance uncertainty of 0.00000% which is now round-off, so the gate does NOT close and the miss is DECISIVE with nothing left to hide any of it behind (S32.4's UNDECIDED clause does not apply) |
| 39 | SPEC-LIT S32.4 verdict 2 (Reynolds analogy), resolved leg | OPEN | OPEN | Gnielinski (1976) evaluated at this leg's own MEASURED wall friction factor, +-10 % band: resolved leg Nu is +14.9% of it at the MEASURED f = 2.3832e-02 - outside the band |
| 40 | SPEC-LIT S88 Gate 88-T | OPEN | OPEN | the ERCOFTAC T3A flat plate's measured transition, on the NASA/TMBWG 2D T3A rig (U = 69.44 m/s, Re/m = 2.00e5, Tu = 3.300 % at the leading edge): the model's own onset criterion fires at Re_x = 85251.4, where the local Tu has decayed to 2.24405 % and Re_theta~ = 234.21 - but no published digit-level onset Re_x for T3A was found to hold it against, so the comparison is not closed |
| 41 | SPEC-LIT S90 Gate 90-T | OPEN | OPEN | the ERCOFTAC T3A flat plate's measured transition, on the NASA/TMBWG 2D T3A rig (U = 69.44 m/s, Re/m = 2.00e5, Tu = 3.300 % at the leading edge): the 2015 correlation's onset switch fires at Re_x = 22826.1, where the local Tu_L has decayed to 54.579 % and Re_thetac reads 100 - but no published digit-level onset Re_x for T3A was found to hold it against, so the comparison is not closed |
| 42 | 78-D | OPEN | OPEN | Bai & Gosman SAE 950283 (1995) against Mundo, Sommerfeld & Tropea, Int. J. Multiphase Flow 21 (1995) 151: the two published splash criteria disagree by a factor of 4.78 in Weber number for the same 100 um water droplet - Mundo splashes it at 8.99 m/s (We 111), Bai-Gosman at 19.65 m/s (We 531) |
| 43 | SPEC-LIT 105.14 Gate 105-C Turek-Hron | OPEN | OPEN | Turek & Hron (2006), CFD1/CFD2 level 6+0 and CFD3 level 4+0: the Turek-Hron meshes are not generated, so no comparison was run |
| 44 | SPEC-LIT S110.2 Gate 110-A channel DNS (Moser, Kim & Mansour 1999) | OPEN | OPEN | Moser, Kim & Mansour (1999) DNS mean-velocity profiles, reference/mkm99/chan{180,395,590}.means, u_D = 0 (none stated): answer key missing: mkm99-chan180, mkm99-chan395, mkm99-chan590 - absent from reference/ or not its manifest digest (SPEC-LIT 110.1); no comparison was run |
| 45 | SPEC-LIT S110.4 Gate 110-C buoyant plume centreline (McCaffrey 1979) | OPEN | OPEN | McCaffrey (1979) NBSIR 79-1910 Table 1 plume-region correlations at the declared Q, reference/mccaffrey_1979/centreline_table1.txt: P3: the finest centreline dT decays as z^(-2.130) over the six stations, outside -5/3 +- 0.25 - the correlation's exponent, so the gate is left open (S110.1) |
| 46 | SPEC-LIT S110.4 Gate 110-C buoyant plume centreline (McCaffrey 1979) | OPEN | OPEN | McCaffrey (1979) NBSIR 79-1910 Table 1 plume-region correlations at the declared Q, reference/mccaffrey_1979/centreline_table1.txt: P1 worst +212.6% outside, P2 worst +76.1% undecided against the +-15% band at u_val/D = 1.422 - on the predicted high side but BEYOND what 20-40 % radiative loss accounts for (dT +16% to +41%, w +8% to +19%): radiation alone does not explain this miss (S110.4) |
| 47 | SPEC-LIT 144.3 T3A leg 1 in 2-D, kOmegaSSTLM, against the TMR's 3.300 % | OPEN | OPEN | the NASA/TMBWG T3A page's stated leading-edge Tu, 3.300 % (SPEC-LIT 88.10): Tu at the leading edge 3.35332 % on the finest mesh, 1.616 % from 3.300 % (bar 0.5 %); the 1-D closed form of the same model is 1.607 % from it |
| 48 | SPEC-LIT 144.3 T3A leg 1 in 2-D, kOmegaSSTGamma, against the TMR's 3.300 % | OPEN | OPEN | the NASA/TMBWG T3A page's stated leading-edge Tu, 3.300 % (SPEC-LIT 88.10): Tu at the leading edge 3.35332 % on the finest mesh, 1.616 % from 3.300 % (bar 0.5 %); the 1-D closed form of the same model is 1.607 % from it |
| 49 | SPEC-LIT 144.4 Gate 88-T leg 4, kOmegaSSTLM, turbulent C_f against Roach & Brierley | OPEN | OPEN | Roach & Brierley's measured C_f(Re_x), the ERCOFTAC T3A key (SPEC-LIT 144.6): undecided, within u_val of the band edge (SPEC-LIT 32.4, 110.1): kOmegaSSTLM worst turbulent C_f against the station's (144.4b): E = -7.07 % at Re_x 3.0930e5 on the finest mesh, band 10 %, u_val 5.22 % (u_num 4.82 %, u_D 2 %) |
| 50 | SPEC-LIT 144.4 Gate 90-T leg 4, kOmegaSSTGamma, transition onset Re_x against Roach & Brierley | OPEN | OPEN | Roach & Brierley's measured C_f(Re_x), the ERCOFTAC T3A key (SPEC-LIT 144.6): undecided, within u_val of the band edge (SPEC-LIT 32.4, 110.1): kOmegaSSTGamma onset Re_x (least C_f) against (144.4a): E = +12.72 % at Re_x 1.5633e5 on the finest mesh, band 15 %, u_val 6.99 % (u_num 6.99 %, u_D 0 %) |
| 51 | SPEC-LIT 144.5 Gate 132-Z ZPG plate, SpalartAllmaras, C_f against Karman-Schoenherr | OPEN | OPEN | the Karman-Schoenherr C_f(Re_theta) as the NASA TMR tabulates it (a correlation, SPEC-LIT 144.5): undecided, within u_val of the bar (SPEC-LIT 32.4, 110.1): SpalartAllmaras worst C_f against K-S over Re_theta 4000..13000: E = -4.455% at Re_theta 4000 on the finest mesh, bar 3 %, u_val 6.344% |
| 52 | SPEC-LIT 144.5 Gate 132-Z ZPG plate, kOmegaSST, C_f against Karman-Schoenherr | OPEN | OPEN | the Karman-Schoenherr C_f(Re_theta) as the NASA TMR tabulates it (a correlation, SPEC-LIT 144.5): outside the bar, against a correlation (SPEC-LIT 144.5): kOmegaSST worst C_f against K-S over Re_theta 4000..13000: E = -4.310% at Re_theta 4000 on the finest mesh, bar 3 %, u_val 0.626% |
| 53 | SPEC-LIT 143.5 Gate 131-P turbulent pipe, kOmegaSST, Re_tau 576.69, f against Prandtl | OPEN | OPEN | Prandtl's smooth-pipe law at the input Re_tau (Nikuradse's measured pipes, Schlichting NACA TM 1218 eq 16.26): undecided, within u_val of the band edge (SPEC-LIT 32.4, 110.1): kOmegaSST at Re_tau 576.69, f against Prandtl: E = +5.18 % on the finest level (n = 160), bar 5 %, u_val 0.66 % |
| 54 | SPEC-LIT 143.5 Gate 131-P turbulent pipe, kOmegaSST, Re_tau 576.69, the log law | OPEN | OPEN | the log law u+ = 2.5 ln y+ + 5.5 over 30 <= y+ <= 0.2 Re_tau (Schlichting NACA TM 1218 eq 16.14): undecided, within u_val of the band edge (SPEC-LIT 32.4, 110.1): kOmegaSST at Re_tau 576.69, log-law deviation: E = 1.177 in u+ on the finest level (n = 160), bar 1.0 in u+, u_val 0.237 |
| 55 | SPEC-LIT 144.7 Gate 88-T leg 3, kOmegaSSTLM, T3B onset Re_x against Roach & Brierley | OPEN | OPEN | Roach & Brierley's measured C_f(Re_x), the ERCOFTAC T3B key (ercoftac-t3b, SPEC-LIT 144.7): undecided, within u_val of the band edge (SPEC-LIT 32.4, 110.1): kOmegaSSTLM onset Re_x (the first grid minimum (144.7e), the least face being the trailing edge) against (144.4a): E = -11.50 % at Re_x 4.9938e4 on the finest mesh, band 15 %, u_val 28.55 % (u_num 28.55 %, u_D 0 %) |
| 56 | SPEC-LIT 144.7 Gate 90-T leg 3, kOmegaSSTGamma, T3B onset Re_x against Roach & Brierley | OPEN | OPEN | Roach & Brierley's measured C_f(Re_x), the ERCOFTAC T3B key (ercoftac-t3b, SPEC-LIT 144.7): no kOmegaSSTGamma T3B run recorded yet - run the record_t3fit_* tests of SPEC-LIT 144.7 |
| 57 | SPEC-LIT 144.7 Gate 90-T leg 3, kOmegaSSTGamma, T3B turbulent C_f against Roach & Brierley | OPEN | OPEN | Roach & Brierley's measured C_f(Re_x), the ERCOFTAC T3B key (ercoftac-t3b, SPEC-LIT 144.7): no kOmegaSSTGamma T3B run recorded yet - run the record_t3fit_* tests of SPEC-LIT 144.7 |
| 58 | SPEC-LIT 144.7 Gate 90-T leg 3, kOmegaSSTGamma, T3B laminar C_f against Blasius | OPEN | OPEN | Roach & Brierley's measured C_f(Re_x), the ERCOFTAC T3B key (ercoftac-t3b, SPEC-LIT 144.7): no kOmegaSSTGamma T3B run recorded yet - run the record_t3fit_* tests of SPEC-LIT 144.7 |
| 59 | SPEC-LIT 144.7 Gate 88-T leg 3, kOmegaSSTLM, T3A- onset Re_x against Roach & Brierley | OPEN | OPEN | Roach & Brierley's measured C_f(Re_x), the ERCOFTAC T3A- key (ercoftac-t3am, SPEC-LIT 144.7): undecided, within u_val of the band edge (SPEC-LIT 32.4, 110.1): kOmegaSSTLM onset Re_x (least C_f) against (144.4a): E = -16.13 % at Re_x 1.2106e6 on the finest mesh, band 15 %, u_val 5.68 % (u_num 5.68 %, u_D 0 %) |
| 60 | SPEC-LIT 144.7 Gate 90-T leg 3, kOmegaSSTGamma, T3A- onset Re_x against Roach & Brierley | OPEN | OPEN | Roach & Brierley's measured C_f(Re_x), the ERCOFTAC T3A- key (ercoftac-t3am, SPEC-LIT 144.7): no kOmegaSSTGamma T3A- run recorded yet - run the record_t3fit_* tests of SPEC-LIT 144.7 |
| 61 | SPEC-LIT 144.7 Gate 90-T leg 3, kOmegaSSTGamma, T3A- turbulent C_f against Roach & Brierley | OPEN | OPEN | Roach & Brierley's measured C_f(Re_x), the ERCOFTAC T3A- key (ercoftac-t3am, SPEC-LIT 144.7): no kOmegaSSTGamma T3A- run recorded yet - run the record_t3fit_* tests of SPEC-LIT 144.7 |
| 62 | SPEC-LIT 144.7 Gate 90-T leg 3, kOmegaSSTGamma, T3A- laminar C_f against Blasius | OPEN | OPEN | Roach & Brierley's measured C_f(Re_x), the ERCOFTAC T3A- key (ercoftac-t3am, SPEC-LIT 144.7): no kOmegaSSTGamma T3A- run recorded yet - run the record_t3fit_* tests of SPEC-LIT 144.7 |
| 63 | SPEC-LIT 144.7 Gate 88-T leg 3, kOmegaSSTLM, T3A onset Re_x against Roach & Brierley | OPEN | OPEN | Roach & Brierley's measured C_f(Re_x), the ERCOFTAC T3A key (ercoftac-t3a, SPEC-LIT 144.6): no kOmegaSSTLM T3A run recorded yet - run the record_t3fit_* tests of SPEC-LIT 144.7 |
| 64 | SPEC-LIT 144.7 Gate 88-T leg 3, kOmegaSSTLM, T3A turbulent C_f against Roach & Brierley | OPEN | OPEN | Roach & Brierley's measured C_f(Re_x), the ERCOFTAC T3A key (ercoftac-t3a, SPEC-LIT 144.6): no kOmegaSSTLM T3A run recorded yet - run the record_t3fit_* tests of SPEC-LIT 144.7 |
| 65 | SPEC-LIT 144.7 Gate 88-T leg 3, kOmegaSSTLM, T3A laminar C_f against Blasius | OPEN | OPEN | Roach & Brierley's measured C_f(Re_x), the ERCOFTAC T3A key (ercoftac-t3a, SPEC-LIT 144.6): no kOmegaSSTLM T3A run recorded yet - run the record_t3fit_* tests of SPEC-LIT 144.7 |
| 66 | SPEC-LIT 144.7 Gate 90-T leg 3, kOmegaSSTGamma, T3A onset Re_x against Roach & Brierley | OPEN | OPEN | Roach & Brierley's measured C_f(Re_x), the ERCOFTAC T3A key (ercoftac-t3a, SPEC-LIT 144.6): no kOmegaSSTGamma T3A run recorded yet - run the record_t3fit_* tests of SPEC-LIT 144.7 |
| 67 | SPEC-LIT 144.7 Gate 90-T leg 3, kOmegaSSTGamma, T3A turbulent C_f against Roach & Brierley | OPEN | OPEN | Roach & Brierley's measured C_f(Re_x), the ERCOFTAC T3A key (ercoftac-t3a, SPEC-LIT 144.6): no kOmegaSSTGamma T3A run recorded yet - run the record_t3fit_* tests of SPEC-LIT 144.7 |
| 68 | SPEC-LIT 144.7 Gate 90-T leg 3, kOmegaSSTGamma, T3A laminar C_f against Blasius | OPEN | OPEN | Roach & Brierley's measured C_f(Re_x), the ERCOFTAC T3A key (ercoftac-t3a, SPEC-LIT 144.6): no kOmegaSSTGamma T3A run recorded yet - run the record_t3fit_* tests of SPEC-LIT 144.7 |
| 69 | SPEC-LIT 144.7 Gate 88-T leg 3, kOmegaSSTLM, onset spread Re_x(T3A-)/Re_x(T3B) against Roach & Brierley | OPEN | OPEN | Roach & Brierley's T3 series, the keys' own (144.4a) onsets and their ratio: undecided, within u_val of the band edge (SPEC-LIT 32.4, 110.1): the onset spread (144.7d): S_sim = 2.4242e1 against S_key = 2.5578e1, E = -5.22 %, band 15 %, u_val 32.97 % (u_num from the two onsets' studies) |
| 70 | SPEC-LIT 144.7 Gate 90-T leg 3, kOmegaSSTGamma, onset ordering T3B < T3A < T3A- against Roach & Brierley | OPEN | OPEN | Roach & Brierley's T3 series, the keys' own (144.4a) onsets and their ratio: missing a T3B fine f64 run, a T3A- fine f64 run |
| 71 | SPEC-LIT 144.7 Gate 90-T leg 3, kOmegaSSTGamma, onset spread Re_x(T3A-)/Re_x(T3B) against Roach & Brierley | OPEN | OPEN | Roach & Brierley's T3 series, the keys' own (144.4a) onsets and their ratio: missing a T3B fine f64 run, a T3A- fine f64 run |
| 72 | SPEC-LIT S110.4 Gate 110-C leg 2 radiating plume centreline (McCaffrey 1979) | OPEN | OPEN | McCaffrey (1979) NBSIR 79-1910 Table 1 plume-region correlations at the declared Q, reference/mccaffrey_1979/centreline_table1.txt; the run's radiant fraction beside the report's 20-40 %: undecided: P1/P2 worst 1.539 is within u_val/D = 2.678 of the 15% edge (S32.4); the run's radiant fraction chi_r = 0.077 against McCaffrey's 0.20-0.40 |
| 73 | Gate 95-A cantilever | - | MISSES | Timoshenko & Goodier ch. 3, plane strain, block-coupled (SPEC-LIT 109.6), four meshes r = 2, orders on the finest pair, the tip study on the finest three: 5:1 GMRES(50): converged 0/4, p_u 0.59, p_sigma 4.61 (n_y 16 -> 32), finest tip 9.71e-1 |
| 74 | Gate 95-A cantilever | - | MISSES | Timoshenko & Goodier ch. 3, plane strain, block-coupled (SPEC-LIT 109.6), four meshes r = 2, orders on the finest pair, the tip study on the finest three: 10:1 GMRES(50): converged 0/4, p_u -53.62, p_sigma -inf (n_y 16 -> 32), finest tip 1.37e16 |
| 75 | SPEC-LIT 98.6 Gate 98-B radiating slab | - | MISSES | the steady balance of a slab radiating to a large surround, (S98.8), SPEC-LIT 98.6: T_b rel 6.56e-7, residual 0.00e0, worst Newton ratio 1.092e4 against 2C = 1.725e-3, 5 ratios |
| 76 | SPEC-LIT 98.9 Gate 98-C enclosure with a running flow | - | MISSES | the enclosure's own power balance, (S98.9), (S98.10), parallel_plate_flux and concentric_flux, SPEC-LIT 98.9: (a) power balance 3.39e-6 against 1e-10; (b) (S98.10) worst 5.77e-2 against 1e-9; (b) cell source rel 9.08e-8 against 1e-12; (b) closed form rel 5.02e-4 against 1e-10 over 100 faces; (c) not every radiationRelaxation converged within 4000 iterations |
| 77 | SPEC-LIT 100.4 Gate 100-B published property tables | - | MISSES | Ho, Powell and Liley (1972), J. Phys. Chem. Ref. Data 1, 279, p. 394: silicon, recommended values: silicon: knots to the bit false, worst 1.55 % between them against 5 % |
| 78 | SPEC-LIT 100.4 Gate 100-B published property tables | - | MISSES | McBride, Zehe and Gordon, NASA/TP-2002-211556 (2002), eq. (1), Appendix D and Table B1: cp/R: worst \|Cp(298.15) - Table B1\| 3.51e-4 J/(K mol) against 5e-4, worst jump at 1000 K 3.41e-7 against 1e-8 |
| 79 | SPEC-LIT 100.15 Gate 100-C Brinkman plane Poiseuille | - | MISSES | Brinkman's fully developed plane Poiseuille with viscous heating, (S100.15) and (S100.16), SPEC-LIT 100.15: both walls held, ny = 8: converged false, balance 5.60e-6; both walls held, ny = 16: converged false, balance 1.68e-4; both walls held, ny = 32: converged false, balance 1.86e-4; top adiabatic, ny = 8: converged false, balance 3.15e-5; top adiabatic, ny = 16: converged false, balance 1.53e-4; top adiabatic, ny = 32: converged false, balance 1.50e-4 |
| 80 | SPEC-LIT 158.7 Gate DST-REMBE: the vented REMBE explosion, REDUCED PLACEHOLDER smoke on the castellated vessel, key J.3 NOT RUN | - | MISSES | Huang, Bloching & Lipatnikov (2022) Fig. 11, key huang2022-rembe-p-vented: 2 row(s) of the REDUCED PLACEHOLDER smoke leave their bars (never compared with H22) |
<!-- validate-run:gates:en end -->

What kind of verification each section is, **what it was compared against** (a paper, benchmark numbers, a correlation, public data, or the repository's own record), and that material's address (DOI/URL) are tabulated in [`docs/GUIDEBOOK.en.md`](docs/GUIDEBOOK.en.md) §11, "Every check, one by one".

A verdict is **`OPEN`** when the comparison cannot decide: it is held against a correlation rather than a measurement (§32.4's plane channel against Gnielinski 1976), two published criteria disagree with **each other** (78-D, by a factor of 4.78 in Weber number), no measured value exists to close it (Gates 88-T and 90-T on T3A), a band edge lies inside the validation uncertainty `u_val`, or the run it needs is not recorded or its answer key is not in the checkout (Gate 110-A without `reference/mkm99/`, SPEC-LIT §110.5).

---

## Where the detail went

| Document | Contents |
|---|---|
| **User guide** (separate page) | Building, the case file, the settings contract, running, output, what it cannot do |
| **Technical guidebook** (separate page) | Discretisation, boundary conditions, pressure–velocity, turbulence, low-Mach, surface-to-surface radiation, validation, GPU residency, mesh adaptation and performance |
| [`rust/SPEC-LIT.md`](rust/SPEC-LIT.md) | The numerical specification, 141 sections (numbered §0 to §161, with gaps), with a citation for every formulation — both guides are drawn *from* it and neither replaces it |
| [`rust/PROVENANCE.md`](rust/PROVENANCE.md) · [`LICENSING.md`](LICENSING.md) · [`NOTICE`](NOTICE) | Per-file provenance and design decisions, the licence audit, third-party notices |
| [`cases/README.md`](cases/README.md) · [`docs/README.md`](docs/README.md) | Test case geometries, and the index to `docs/` — the model catalogue, GPU portability, the I/O redesign and the JSONC schema, and `ofgpu-lowmach`'s low-Mach formulation and wall-heat gate record |

---

## References

Sources for the numerical methods and models. Section numbers refer to SPEC-LIT, and each entry carries only the bibliographic detail SPEC-LIT actually prints — where a title, issue number or page range is missing it is because SPEC-LIT cited the work without one. Sources that were not read (paywalled, unreachable, or deliberately left closed) and the verdicts `ofgpu-validate` prints by name are marked on their own lines.

### Finite volume discretisation
- Jasak, H. (1996). *Error Analysis and Estimation for the Finite Volume Method with Applications to Fluid Flows.* PhD thesis, Imperial College London. `http://hdl.handle.net/10044/1/8335` — §2, §3, §74, §82
- Moukalled, F., Mangani, L., & Darwish, M. (2016). *The Finite Volume Method in Computational Fluid Dynamics.* Springer. — §2, §3, §11, §74, §82
- Ferziger, J. H., & Perić, M. *Computational Methods for Fluid Dynamics*, 3rd ed. Springer (2002). — §2.4, §3.3, §11.1, §11.5, §74, §82
- Patankar, S. V. (1980). *Numerical Heat Transfer and Fluid Flow.* Hemisphere. ISBN 0-89116-522-3. — §3.4, §5.2, §18, §46, §50, §56, §68
### Convection schemes
- Warming, R. F., & Beam, R. M. (1976). *AIAA Journal*, 14, 1241–1249. — §11.2
- Leonard, B. P. (1979). *Computer Methods in Applied Mechanics and Engineering*, 19, 59–98. — §11.3
- Leonard, B. P. (1991). *Computer Methods in Applied Mechanics and Engineering*, 88, 17–74. — §7 (the NVD framework)
- Khosla, P. K., & Rubin, S. G. (1974). *Computers & Fluids*, 2, 207–209. — §11.1
- Jasak, H., Weller, H. G., & Gosman, A. D. (1999). *International Journal for Numerical Methods in Fluids*, 31, 431–449. — §11.6
- Sweby, P. K. (1984). *SIAM Journal on Numerical Analysis*, 21, 995–1011. — §7 (the TVD framework)
- van Leer, B. (1977). *Journal of Computational Physics*, 23. — §7 (the limiter)
- van Leer, B. (1979). — §7 (the MUSCL limiter)
- van Albada, G. D., van Leer, B., & Roberts, W. W. (1982). *Astronomy and Astrophysics*, 108. — §7
- Roe, P. L. (1986). — §7 (minmod and Superbee)
- Darwish, M., & Moukalled, F. (2003). *International Journal of Heat and Mass Transfer*, 46, 599–611. — §7 (the gradient ratio on an unstructured mesh)
### Gradients and limiters
- Barth, T. J., & Jespersen, D. C. (1989). The design and application of upwind schemes on unstructured meshes. *27th Aerospace Sciences Meeting*, AIAA 89-0366. DOI `10.2514/6.1989-366` — §12.2, §75
- Venkatakrishnan, V. (1993). *AIAA Paper 93-0880.* — §12.2 (the smooth variant)
### Time integration
- Crank, J., & Nicolson, P. (1947). *Proceedings of the Cambridge Philosophical Society*, 43, 50–67. — §13.1
### Pressure–velocity coupling
- Patankar, S. V., & Spalding, D. B. (1972). — §5.2 (SIMPLE)
- Van Doormaal, J. P., & Raithby, G. D. (1984). — §5.3 (SIMPLEC)
- Issa, R. I. (1986). *Journal of Computational Physics*, 62, 40–65. — §5.4, §14
- Rhie, C. M., & Chow, W. L. (1983). *AIAA Journal*, 21, 1525–1532. — §5.1, §52
### Turbulence — RANS
- Launder, B. E., & Spalding, D. B. (1974). *Computer Methods in Applied Mechanics and Engineering*, 3, 269–289. — §6.1, §6.4
- Wilcox, D. C. *Turbulence Modeling for CFD.* DCW Industries. — §6.2 (the 1988 form); §5.4 there is the source of the Favre-averaged dilatation terms
- Menter, F. R. (1994). *AIAA Journal*, 32, 1598–1605. — §6.3
- Menter, F. R., Kuntz, M., & Langtry, R. (2003). *Turbulence, Heat and Mass Transfer*, 4. — §6.3 (the 2003 revision)
- Launder, B. E., & Sharma, B. I. (1974). *Letters in Heat and Mass Transfer*, 1, 131–138. — §33
- Patel, V. C., Rodi, W., & Scheuerer, G. (1985). *AIAA Journal*, 23, 1308. — §33 (the review of the low-Reynolds-number family)
- Shih, T.-H., Liou, W. W., Shabbir, A., Yang, Z., & Zhu, J. (1995). *Computers & Fluids*, 24, 227–238. Read as **NASA TM-106721 / ICOMP-94-21 (August 1994)**, `https://ntrs.nasa.gov/citations/19950005029`, a US government work in the public domain; **the journal version is paywalled and was not read**. — §40
- Yakhot, V., Orszag, S. A., Thangam, S., Gatski, T. B., & Speziale, C. G. (1992). *Physics of Fluids A*, 4, 1510–1520. Read as **ICASE Report 91-65 / NASA CR-187611 (1991)**, `https://ntrs.nasa.gov/citations/19910021152`, US government-sponsored, public domain via NTRS. — §41
- Yakhot, V., & Orszag, S. A. (1986). *Journal of Scientific Computing*, 1, 3–51. — §41 (the original renormalisation-group derivation)
- Reynolds, W. C. AGARD Report 755 (1987). — §40 (the realizability constraints — positivity of the normal stresses, the Schwarz inequality — that the variable `C_mu` is constructed to satisfy)
- Lumley, J. L. (1978). *Advances in Applied Mechanics*, 18, 123–176. — §40 (realizability as a modelling principle)
- Pope, S. B. *Turbulent Flows* (2000), §10.4. — §40
- Spalart, P. R., & Allmaras, S. R. *AIAA Paper* 92-0439 (1992); also *La Recherche Aérospatiale*, 1 (1994), 5–21. — §56 (the original)
- Allmaras, S. R., Johnson, F. T., & Spalart, P. R. (2012). Modifications and Clarifications for the Implementation of the Spalart-Allmaras Turbulence Model. *ICCFD7-1902.* `https://www.iccfd.org/iccfd7/assets/pdf/papers/ICCFD7-1902_paper.pdf` — a freely distributed conference paper, **the copy actually read**, and the implementation reference. — §56
- NASA / Turbulence Modeling Benchmarking Working Group. *Turbulence Modeling Resource — The Spalart-Allmaras Turbulence Model.* `https://tmbwg.github.io/turbmodels/spalart.html` — US government-authored DOCUMENTATION, not source; quoted to the printed digit. — §56
- Rumsey, C. L., & Spalart, P. R. (2009). *AIAA Journal*, 47, 982–993. — §56 (why the free-stream `nu~/nu` matters)
- Menter, F. R., Smirnov, P. E., Liu, T., & Avancha, R. (2015). *Flow, Turbulence and Combustion*, 95, 583–619. DOI 10.1007/s10494-015-9622-4. **Paywalled and NOT read**; every digit comes from the NASA/TMBWG Turbulence Modeling Resource page below. — §90
- NASA / Turbulence Modeling Benchmarking Working Group. *Turbulence Modeling Resource — SST-2003-Menter-Gamma-2015.* `https://tmbwg.github.io/turbmodels/menter_gamma_3eqn.html` — US government-authored DOCUMENTATION, not source; fetched and read 2026-09-05, quoted to the printed digit, including the constant it prints in no equation (§90.3). — §90
### Turbulence — LES
- Smagorinsky, J. (1963). *Monthly Weather Review*, 91, 99–164. — §6.5
- Deardorff, J. W. (1970). *Journal of Fluid Mechanics*, 41, 453–480. — §16.1
- Deardorff, J. W. (1980). *Boundary-Layer Meteorology*, 18, 495–527. — §6.5 (the model FDS uses)
- Nicoud, F., & Ducros, F. (1999). *Flow, Turbulence and Combustion*, 62, 183–200. — §6.5 (WALE)
- van Driest, E. R. (1956). *Journal of the Aeronautical Sciences*, 23, 1007–1011. — §16.4
- Scotti, A., Meneveau, C., & Lilly, D. K. (1993). *Physics of Fluids A*, 5, 2306–2308. — §16.3
### Turbulence — hybrid RANS-LES
- Spalart, P. R., Jou, W.-H., Strelets, M., & Allmaras, S. R. (1997). Comments on the feasibility of LES for wings, and on a hybrid RANS/LES approach. In *Advances in DNS/LES*, Greyden Press, 137–147. — §57 (DES97)
- Shur, M., Spalart, P. R., Strelets, M., & Travin, A. (1999). *Engineering Turbulence Modelling and Experiments 4*, 669–678. — §57 (the `C_DES = 0.65` calibration on the SA background, at `Delta = h_max`)
- Strelets, M. *AIAA Paper* 2001-0879. — §57 (SST-DES, the `k`-equation dissipation form)
- Spalart, P. R., Deck, S., Shur, M., Squires, K. D., Strelets, M., & Travin, A. (2006). *Theoretical and Computational Fluid Dynamics*, 20, 181–195. — §57 (DDES: `r_d`, `f_d`, and the grid-induced separation they fix)
- Shur, M., Spalart, P. R., Strelets, M., & Travin, A. (2008). *International Journal of Heat and Fluid Flow*, 29, 1638–1649 — IDDES. **Paywalled and NOT read**; §57's IDDES equations come from the two open-access restatements below. — §57
- Gritskevich, M. S., Garbaruk, A. V., Schütze, J., & Menter, F. R. (2012). *Flow, Turbulence and Combustion*, 88, 431–449 — the SST-background recalibration. **Paywalled and NOT read**: `C_dt1 = 20`, `c_t = 1.87` and `c_l = 5.0` are carried from a design note's reading of it, defaulted, printed in the banner, and **not independently verified**. — §57
- Herr, F., Radespiel, R., & Probst, A. (2023). Improved Delayed Detached Eddy Simulation with Reynolds-Stress Background Modeling. *arXiv:2301.07223v2*; published in *Computers & Fluids*, 265 (2023) 106014. **Appendix A is a complete restatement of the IDDES formulation** and is where (57.9)–(57.16) come from, equation by equation. Open access, fetched and read in full. — §57
- Savino, A., Griffin, K., Lee, S., Vijayakumar, G., Wu, S., & Sprague, M. (2026). Improving boundary-layer separation prediction by an IDDES turbulence model using a pressure-gradient sensor. *arXiv:2603.08875*, arXiv non-exclusive distribution licence. **Section 2 states SST-IDDES** and is where `C_DES1 = 0.78`, `C_DES2 = 0.61`, `C_w = 0.15` and the simplified filter width (57.18) come from. Open access, read in full. — §57
- Nikitin, N. V., Nicoud, F., Wasistho, B., Squires, K. D., & Spalart, P. R. (2000). *Physics of Fluids*, 12, 1629–1632. — §57 (the log-layer mismatch `f_e` exists to remove)
- Spalart, P. R. (2009). *Annual Review of Fluid Mechanics*, 41, 181–202. — §57 (the review)
- Fröhlich, J., Mellen, C. P., Rodi, W., Temmerman, L., & Leschziner, M. A. (2005). *Journal of Fluid Mechanics*, 526, 19–66. — §57.12, the periodic-hill gate at `Re_b = 10 595`: **named and NOT run**
### Wall treatment
- Spalding, D. B. (1961). *Journal of Applied Mechanics*, 28, 455–458. — §6.4, §15.1
- Cebeci, T., & Bradshaw, P. *Momentum Transfer in Boundary Layers*, Hemisphere (1977). — §15.3 (rough-wall boundary layers; Nikuradse's sand-grain data underlies the constants)
- Jayatilleke, C. L. V. (1969). *Progress in Heat and Mass Transfer*, 1, 193–330. — §29.3 (the sublayer-resistance correction to the thermal log law)
- Werner, H., & Wengle, H. (1991). Large-eddy simulation of turbulent flow over and around a cube in a plate channel. *8th Symposium on Turbulent Shear Flows.* — §30.1
- Tucker, P. G. (1998). *Applied Mathematical Modelling*, 22, 293–305. — §6.6 (the Poisson wall-distance approach)
- Dittus, F. W., & Boelter, L. M. K. (1930). *University of California Publications in Engineering*, 2, 443, reprinted in *International Communications in Heat and Mass Transfer*, 12 (1985) 3. — §32.3. Conventionally quoted at ±20–25 %.
- Gnielinski, V. (1976). *International Chemical Engineering*, 16, 359. — §32.3, ±10 %. **OPEN × 3.** §32.4's three channel verdicts are held against this correlation and none closes at the shipped default: verdict 1 (absolute prediction, resolved leg, at the Petukhov smooth-pipe `f`), and verdict 2 (Reynolds analogy, at each leg's own MEASURED `f`) on both the wall-function leg and the resolved leg. Every such statement must name which `f` Gnielinski was evaluated at — §32.3's own rule.
- Kays, W. M. (1994). *ASME Journal of Heat Transfer*, 116, 284–295. — §32.5, §37 (that `Pr_t` rises towards a wall: a named hypothesis with a mechanism and a direction for the §32.4 verdicts, and nothing here has measured it)
### Buoyancy
- Rehm, R. G., & Baum, H. R. (1978). The equations of motion for thermally driven, buoyant flows. *Journal of Research of the National Bureau of Standards*, 83, 297–308. — §9, §25, §77
- Majda, A., & Sethian, J. (1985). *Combustion Science and Technology*, 42, 185. — §25
- Spiegel, E. A., & Veronis, G. (1960). *Astrophysical Journal*, 131, 442. — §9 (the `ΔT/T << 1` requirement, which a strongly heated plume does not meet)
- Rodi, W. (1987). *Journal of Geophysical Research*, 92, 5305–5328. — §17
- Henkes, R. A. W. M., van der Vlugt, F. F., & Hoogendoorn, C. J. (1991). *International Journal of Heat and Mass Transfer*, 34, 377–388. — §17
### Conjugate heat transfer
- Carslaw, H. S., & Jaeger, J. C. *Conduction of Heat in Solids*, 2nd ed., Oxford University Press (1959), ch. I. ISBN 0-19-853368-3. — §46 (the anisotropic solid, and the affine transformation that reduces `div(K grad T)` to `lap T`)
- Aavatsmark, I. (2002). An introduction to multipoint flux approximations for quadrilateral grids. *Computational Geosciences*, 6, 405–432. DOI `10.1023/A:1021291114475` — §46.4 (the rigorous full-tensor treatment, and therefore the reason §46.4 refuses rather than approximating)
- Lipnikov, K., Shashkov, M., Svyatskiy, D., & Vassilevski, Yu. (2007). *Journal of Computational Physics*, 227, 492–512. DOI `10.1016/j.jcp.2007.08.008` — §46.4 (the nonlinear monotone alternative, named in the same refusal)
- Yovanovich, M. M. (2005). *IEEE Transactions on Components and Packaging Technologies*, 28, 182–206. DOI `10.1109/TCAPT.2005.848483` — §46.3 (the layered-stack conductivities the Wiener pair homogenises), §47.12 (the review, and the gas-gap and elastic regimes §47.12 omits)
- Giles, M. B. (1997). *International Journal for Numerical Methods in Fluids*, 25, 421–436. DOI `10.1002/(SICI)1097-0363(19970830)25:4<421::AID-FLD557>3.0.CO;2-J` — §47 (the Godunov–Ryabenkii normal-mode analysis behind the classical "Dirichlet on the fluid, Neumann on the solid" rule)
- Meng, F., Banks, J. W., Henshaw, W. D., & Schwendeman, D. W. (2017). A stable and accurate partitioned algorithm for conjugate heat transfer. *Journal of Computational Physics*, 344, 51–85. DOI `10.1016/j.jcp.2017.04.052` — §47.7, **Theorem 1**: the amplification factor that is the reason Dirichlet–Neumann partitioning is not implemented here
- Henshaw, W. D., & Chand, K. K. (2009). *Journal of Computational Physics*, 228, 3708–3741. DOI `10.1016/j.jcp.2009.02.007` — §47 (Robin coefficients can always be chosen so the sub-time-step iteration converges)
- Verstraete, T., & Scholl, S. (2016). *International Journal of Heat and Mass Transfer*, 101, 852–869. DOI `10.1016/j.ijheatmasstransfer.2016.05.041` — §47 (the numerical Biot number, and FFTB's instability above `Bi = 1`)
- Gander, M. J. (2006). Optimized Schwarz methods. *SIAM Journal on Numerical Analysis*, 44, 699–731. DOI `10.1137/S0036142903425409` — §47 (the physical series conductance is the zeroth-order optimised-Schwarz weight; the optimal weight is a non-local operator)
- Cooper, M. G., Mikic, B. B., & Yovanovich, M. M. (1969). Thermal contact conductance. *International Journal of Heat and Mass Transfer*, 12, 279–300. DOI `10.1016/0017-9310(69)90011-8` — §47.12 (the plastic-deformation contact conductance correlation)
- de Vahl Davis, G. (1983). Natural convection of air in a square cavity: a bench mark numerical solution. *International Journal for Numerical Methods in Fluids*, 3, 249–264. DOI `10.1002/fld.1650030305` — §59.8, the fluid-only anchor run first, because a conjugate answer built on an unvalidated buoyant solver measures nothing. **The primary is paywalled**; its four numbers are quoted from Qi et al., *Nanoscale Research Letters*, 8 (2013) 56, DOI `10.1186/1556-276X-8-56`, Table 3 (open access).
- Kaminski, D. A., & Prakash, C. (1986). *International Journal of Heat and Mass Transfer*, 29(12), 1979–1988. DOI `10.1016/0017-9310(86)90017-7` — §47.12's Gate 5, the configuration §60.5 runs. **Paywalled; no open-access copy was found and the primary table was never read**, so no title is asserted for it here either. **MISSES — Gate 5.**
- Belazizia, A., Benissaad, S., & Abboudi, S. (2012). Effect of wall conductivity on conjugate natural convection in a square enclosure with finite vertical wall thickness. *Advanced Theoretical and Applied Mechanics*, 5, no. 4, 179–190. Open access at `m-hikari.com/atam/atam2012/atam1-4-2012/` — an independent published solution of the Kaminski–Prakash configuration, itself validated against it. **The SECONDARY source Gate 5 actually compares against.** — §60.5
- Qu, W., & Mudawar, I. (2002). Analysis of three-dimensional heat transfer in micro-channel heat sinks. *International Journal of Heat and Mass Transfer*, 45, 3973–3985. DOI `10.1016/S0017-9310(02)00101-1` — §47.12's Gate 6, the semiconductor gate. Read in full from the authors' own public copy. **§79.12 runs it and it PASSES**: both substrate temperatures fall inside the experimental uncertainty the paper draws. — §79
- Kawano, K., Minakami, K., Iwasaki, H., & Ishizuka, M. Micro channel heat exchanger for cooling electrical equipment. *ASME HTD-361-3/PID-3* (1998) 173–180. — the inlet and outlet thermal-resistance measurements Gate 6 is held against. **NOT OBTAINED** (an ASME conference volume; no copy found), so the comparison is a **digitisation of Qu & Mudawar's Fig. 4**, which §79.12's Disclosure 1 states. — §79
### Radiation
- McGrattan, K., Hostikka, S., McDermott, R., Floyd, J., Weinschenk, C., Overholt, K., Vanella, M., et al. *Fire Dynamics Simulator Technical Reference Guide*, NIST SP 1018-1. NIST, US public domain; read locally from `reference/fds/Manuals/` with `reference/fds/LICENSE.md` read verbatim. **No FDS source code was read for these sections.** — §25, §66, §68, §76
- Modest, M. F. *Radiative Heat Transfer*, 3rd ed., Academic Press (2013), ch. 5. — §50
- Hottel, H. C., & Sarofim, A. F. *Radiative Transfer*, McGraw-Hill (1967), ch. 3, 5. — §50 (the net-radiation exchange method; the method of images, named in §50.9's refusal)
- Walton, G. N. *Calculation of Obstructed View Factors by Adaptive Integration.* NISTIR 6925, National Institute of Standards and Technology, November 2002. `https://nvlpubs.nist.gov/nistpubs/Legacy/IR/nistir6925.pdf` — US Government, public domain. §49 (the double area integral and its dot-product form, the obstruction-elimination tests, the row-sum figure of merit, and the `BB104` benchmark)
- Shapiro, A. B. *FACET — A Radiation View Factor Computer Code for Axisymmetric, 2D Planar and 3D Geometries with Shadowing.* UCID-19887, Lawrence Livermore National Laboratory, 1983. DOI `10.2172/5607653` — US DOE, public domain. §49.8 (the shadowed configuration `F_12 = 0.115621`)
- Howell, J. R. *A Catalog of Radiation Heat Transfer Configuration Factors*, 3rd ed. `https://www.thermalradiation.net/` — entries **C-11** (identical parallel directly-opposed rectangles) and **C-14** (two rectangles of equal length sharing an edge at 90°), both tracing to Hottel (1931) and Hamilton & Morgan (1952). §49.8 (the two analytic view-factor gates)
- Gebhart, B. (1961). *International Journal of Heat and Mass Transfer*, 3(4), 341–346. DOI `10.1016/0017-9310(61)90048-5` — §50.2, the absorption-factor alternative: **named and not used**
- Balaji, C., & Venkateshan, S. P. *International Journal of Heat and Fluid Flow*, 14(3) (1993) 260–267, DOI `10.1016/0142-727X(93)90057-T`, and 15(3) (1994) 249–251, DOI `10.1016/0142-727X(94)90046-9`; Akiyama, M., & Chong, Q. P. *Numerical Heat Transfer A*, 32(4) (1997) 419–433, DOI `10.1080/10407789708913899` — the coupled convection-plus-surface-radiation cavity gate. — §50.12, **NOT run**: the tabulated `Nu_conv`/`Nu_rad` are paywalled and the fluid side has no case format for a radiating enclosure.
### Rheology and the contact angle
- Ostwald, W. (1925). *Kolloid-Zeitschrift*, 36, 99–117; de Waele, A. (1923). — §38 (the power law)
- Cross, M. M. (1965). *Journal of Colloid Science*, 20, 417–437. — §38
- Carreau, P. J. (1972). *Transactions of the Society of Rheology*, 16, 99–127; Yasuda, K., Armstrong, R. C., & Cohen, R. E. (1981). *Rheologica Acta*, 20, 163–178. — §38 (one formula serves both)
- Herschel, W. H., & Bulkley, R. (1926). *Kolloid-Zeitschrift*, 39, 291–300. — §38
- Casson, N. In Mill (ed.), *Rheology of Disperse Systems*, Pergamon (1959), 84–104. — §38
- Papanastasiou, T. C. (1987). *Journal of Rheology*, 31, 385–404. — §38.3 (the regularisation, in the **product** form)
- Bercovier, M., & Engelman, M. (1980). *Journal of Computational Physics*, 36, 313–326. — §38.3 (the alternative regularisation)
- Frigaard, I. A., & Nouar, C. (2005). *Journal of Non-Newtonian Fluid Mechanics*, 127, 1–26. — §38.3 (what regularisation costs)
- Bird, R. B., Armstrong, R. C., & Hassager, O. *Dynamics of Polymeric Liquids*, vol. 1, 2nd ed., Wiley (1987). — §38 (the family)
- Chhabra, R. P., & Richardson, J. F. *Non-Newtonian Flow and Applied Rheology*, 2nd ed. (2008). — §38.9 (Buckingham–Reiner)
- Young, T. (1805). *Philosophical Transactions of the Royal Society*, 95, 65–87. — §39 (the equilibrium angle)
- Huh, C., & Scriven, L. E. (1971). *Journal of Colloid and Interface Science*, 35, 85–101. — §39 (the moving contact-line singularity)
- Voinov, O. V. (1976). *Fluid Dynamics*, 11, 714–721; Cox, R. G. (1986). *Journal of Fluid Mechanics*, 168, 169–194. — §39.4 (the asymptotic matching)
- Hoffman, R. L. (1975). *Journal of Colloid and Interface Science*, 50, 228–241. — §39.4 (the master curve)
- Jiang, T.-S., Oh, S.-G., & Slattery, J. C. (1979). *Journal of Colloid and Interface Science*, 69, 74–77. — §39.4 (the explicit correlation used here; Kistler's fit is **deliberately absent**, its four constants coming from a book chapter this project has not read)
- Afkhami, S., Zaleski, S., & Bussmann, M. (2009). *Journal of Computational Physics*, 228, 5370–5389 — the mesh-dependent (numerical-slip) angle. §39.8, **named and deliberately not implemented** until the gate that would show it works exists
- Sui, Y., Ding, H., & Spelt, P. D. M. (2014). *Annual Review of Fluid Mechanics*, 46, 97–119. — §39 (the review)
- Washburn, E. W. (1921). *Physical Review*, 17, 273–283. — §39.7 (capillary rise)
### Lagrangian particles
- Dukowicz, J. K. A particle-fluid numerical model for liquid sprays. *Journal of Computational Physics*, 35 (1980) 229–253. DOI `10.1016/0021-9991(80)90087-X` — §66 (the discrete droplet model: the parcel, and the real-valued weight `n_p`)
- Crowe, C. T., Sharma, M. P., & Stock, D. E. The particle-source-in-cell (PSI-CELL) model for gas-droplet flows. *Journal of Fluids Engineering*, 99 (1977) 325. DOI `10.1115/1.3448756` — §67, §68 (the per-cell sum every coupled source is)
- Crowe, C., Sommerfeld, M., & Tsuji, Y. *Multiphase Flows with Droplets and Particles*, CRC Press (1998). ISBN 0-8493-9469-4 — §66.2 (the equation of motion, and the regime argument for which of its terms survive)
- Maxey, M. R., & Riley, J. J. *Physics of Fluids*, 26 (1983) 883. DOI `10.1063/1.864230` — §66, §68 (the equation of motion: the added-mass coefficient, and the drag term §68 returns to the gas)
- Schiller, L., & Naumann, A. *Zeitschrift des Vereines Deutscher Ingenieure*, 77 (1933) 318, in the form compiled by Clift, R., Grace, J. R., & Weber, M. E. *Bubbles, Drops, and Particles*, Academic Press (1978). ISBN 0-12-176950-X — §66.3 (the drag correlation)
- Macpherson, G. B., Nordin, N., & Weller, H. G. *Communications in Numerical Methods in Engineering*, 25 (2009) 263. DOI `10.1002/cnm.1128` — §66.6, barycentric tracking: the **paper** was read and it is **not implemented** — the one case the face-crossing walk cannot do. Its OpenFOAM implementation is GPL-3.0 and was **not** opened.
- Elghobashi, S. On predicting particle-laden turbulent flows. *Applied Scientific Research*, 52 (1994) 309. DOI `10.1007/BF00936835` — §67, §68 (the coupling map: below `alpha_p ~ 1e-6` one-way coupling suffices, `1e-6`–`1e-3` needs §68, above `1e-3` collisions matter and are not here)
- Satish, N., Harris, M., & Garland, M. Designing efficient sorting algorithms for manycore GPUs. *IEEE IPDPS 2009.* DOI `10.1109/IPDPS.2009.5161005` — the **paper** was read; no implementation of it was opened. §67.4 (the three-phase radix pass)
- Merrill, D., & Grimshaw, A. *Parallel scan for stream architectures.* University of Virginia Technical Report CS2009-14. — §67.2 (the reduce-then-scan decomposition)
- Blelloch, G. E. *Prefix sums and their applications.* CMU-CS-90-190 (1990). — §67.2 (the exclusive scan and its work-efficiency argument)
- Hillis, W. D., & Steele, G. L., Jr. *Communications of the ACM*, 29(12) (1986) 1170. DOI `10.1145/7902.7903` — §67.2 (the in-block scan network, chosen for a property that is not speed)
- Steele, G. L., Jr., Lea, D., & Flood, C. H. Fast splittable pseudorandom number generators. *OOPSLA 2014*, ACM SIGPLAN Notices, 49(10) 453. DOI `10.1145/2660193.2660195` — §66.9 (the SplitMix64 finalising mix, used as a **bijection** and not as a generator)
- Ranz, W. E., & Marshall, W. R. Evaporation from drops. *Chemical Engineering Progress*, 48 (1952) 141–146 (Part I) and 173–180 (Part II). — §68.5 (the sensible-heat half of `Nu_0 = 2 + 0.6 Re^(1/2) Pr^(1/3)`), §76 (the mass-transfer half, and the 56 suspended-droplet experiments §76.12's first gate measures)
- Spalding, D. B. The combustion of liquid fuels. *4th Symposium (International) on Combustion* (1953) 847–864; and *Convective Mass Transfer: An Introduction*, Edward Arnold (1963). — §76.6 (`B_M`, and the Stefan-flow rate)
- Godsave, G. A. E. Studies of the combustion of drops in a fuel spray. *4th Symposium (International) on Combustion* (1953) 818–830. — §76.9 (the heat-limited rate at the boiling point)
- Abramzon, B., & Sirignano, W. A. Droplet vaporization model for spray combustion calculations. *International Journal of Heat and Mass Transfer*, 32 (1989) 1605–1618. DOI `10.1016/0017-9310(89)90043-4` — §76.6 (`B_T = (1 + B_M)^φ − 1`, the default)
- Sazhin, S. S. Advanced models of fuel droplet heating and evaporation. *Progress in Energy and Combustion Science*, 32 (2006) 162–214. DOI `10.1016/j.pecs.2005.11.001` — §76
- Watson, K. M. Thermodynamics of the liquid state. *Industrial & Engineering Chemistry*, 35 (1943) 398–406. — §76.4 (`h_v(T)`)
- Marrero, T. R., & Mason, E. A. Gaseous diffusion coefficients. *Journal of Physical and Chemical Reference Data*, 1 (1972) 3–118. DOI `10.1063/1.3253094` — §76.4
- Lewis, W. K. The evaporation of a liquid into a gas. *Transactions of the ASME*, 44 (1922) 325–340. — §76.13
- NIST Chemistry WebBook, SRD 69. US government, public domain. — §76.4 (the water-vapour specific heat and the critical constants)
- Theobald, R. C. The effect of nozzle design on the stability and performance of turbulent water jets. *Fire Safety Journal*, 4 (1981) 1–13. — §68.12, about 90 hose-stream experiments. **MISSES — Gate 68-C**, with the gas held at rest.
- Bai, C., & Gosman, A. D. *SAE 950283* (1995). — §78.1, §78.4 (the impact regime map, and the alternative splash threshold)
- Mundo, C., Sommerfeld, M., & Tropea, C. *International Journal of Multiphase Flow*, 21 (1995) 151. — §78.4 (`K = Oh Re^1.25`, `K_crit = 57.7`, the default). The experimental data itself was **not transcribed**, which is why Gate 78-D is open. **OPEN — Gate 78-D**: the two published splash criteria disagree by a measured factor in Weber number for the same droplet.
- Yarin, A. L. *Annual Review of Fluid Mechanics*, 38 (2006) 159. — §78.1 (the review that explains why neither threshold is better than approximately right)
- IAPWS R1-76 (2014). — §78.2 (`sigma = B tau^mu (1 + b tau)`, implemented verbatim and gated against the release's own table)
### Multiphase flow
- Hirt, C. W., & Nichols, B. D. *Journal of Computational Physics*, 39 (1981) 201–225. — §20.1
- Zalesak, S. T. *Journal of Computational Physics*, 31 (1979) 335–362. — §20.2 (and §22's rotating-slotted-disc boundedness check)
- Brackbill, J. U., Kothe, D. B., & Zemach, C. *Journal of Computational Physics*, 100 (1992) 335–354. — §20.4, §87 (the continuum-surface-force regularisation)
- Ubbink, O. PhD thesis, Imperial College London (1997). — §20.1 (the interface-compressed finite-volume form on unstructured meshes)
- Rusche, H. PhD thesis, Imperial College London (2002). — §20.1 (the same)
### Linear solvers
- Saad, Y. *Iterative Methods for Sparse Linear Systems*, 2nd ed. SIAM (2003). DOI `10.1137/1.9780898718003` — §8, §21 (§6.7 PCG, §7.4.2 BiCGStab, §12.4 multicolour ILU, ch. 14 block Jacobi and additive Schwarz)
- van der Vorst, H. A. Bi-CGSTAB: A Fast and Smoothly Converging Variant of Bi-CG for the Solution of Nonsymmetric Linear Systems. *SIAM Journal on Scientific and Statistical Computing*, 13(2), 631–644 (1992). DOI `10.1137/0913035` — §8.1
- Hestenes, M. R., & Stiefel, E. Methods of conjugate gradients for solving linear systems. *Journal of Research of the National Bureau of Standards*, 49(6), 409 (1952). DOI `10.6028/jres.049.044` — §8.2. A US Government work, public domain.
- Swarztrauber, P. N. *SIAM Review*, 19 (1977) 490–501. — §8.5
- Stüben, K. *Journal of Computational and Applied Mathematics*, 128 (2001) 281–309; Ruge & Stüben (1987). — §8.3. Provided here by AMGX (BSD-3-Clause), **not reimplemented**.
### Porous media
- Ward, J. C. Turbulent flow in porous media. *Journal of the Hydraulics Division, ASCE*, 90(5) (1964) 1–12. DOI `10.1061/JYCEAJ.0001096` — §18, §53 (the same Darcy–Forchheimer law integrated through a slab instead of over a cell)
- Idelchik, I. E. *Handbook of Hydraulic Resistance*, 4th ed., Begell House (2007). ISBN 978-1-56700-251-5, Diagrams 8-1 to 8-6 — perforated plates and screens, the source of `K(sigma)`. **Not opened for §53**; the thin-plate form used is the one published in the open literature, and §53.7 gates it against its own limits. — §53
- Karki, K. C., Radmehr, A., & Patankar, S. V. Use of computational fluid dynamics for calculating flow rates through perforated tiles in raised-floor data centers. *HVAC&R Research*, 9(2) (2003) 153–166. DOI `10.1080/10789669.2003.10391062` — §53.8, the per-tile flow-rate gate: **NOT run** — the paper was not reachable from this environment.
- Karki, K. C., & Patankar, S. V. Airflow distribution through perforated tiles in raised-floor data centers. *Building and Environment*, 41(6) (2006) 734–744. DOI `10.1016/j.buildenv.2005.03.005` — §53
### Ventilation, psychrometrics and data-centre metrics
- AMCA 210 / ASHRAE 51, *Laboratory Methods of Testing Fans for Certified Aerodynamic Performance Rating.* — §52 (what a manufacturer's curve **is** — a static-pressure rise against volumetric flow at a stated density and shaft speed — and therefore why §52.5 carries a density and a speed correction rather than treating the table as absolute)
- NIST, *Fire Dynamics Simulator* 6 verification suite, `Verification/HVAC/fan_test.fds` and `qfan_test.fds` with their published `.csv` reference values — US Government public domain. **The case files and their reference numbers** are the external cross-check of §52.12 Gate 52-B. `Source/hvac.f90` was read for the DISCIPLINE only: that a fan curve is scaled by `rho/rho_curve` at every evaluation, and that its tabulated branch resolves the operating point by a bisection with a data-dependent trip count, which is correct for a CPU code and uncapturable here (§52.7).
- Buzbee, B. L., Dorr, F. W., George, J. A., & Golub, G. H. The direct solution of the discrete Poisson equation on irregular regions. *SIAM Journal on Numerical Analysis*, 8(4) (1971) 722–736. DOI `10.1137/0708066` — §52.9, the capacitance-matrix path: **named and refused**
- ASHRAE. *ASHRAE Handbook—Fundamentals*, Chapter 1, "Psychrometrics", ASHRAE (2021). — §54.2 (whose equation numbering is used), §54.8 (Table 2, the external comparison, at 101.325 kPa)
- Hyland, R. W., & Wexler, A. Formulations for the thermodynamic properties of the saturated phases of H2O from 173.15 K to 473.15 K. *ASHRAE Transactions*, 89(2A) (1983) 500–519. — §54.2 (the `C1`–`C13` coefficients), §76.4 (reused rather than re-fitted)
- Herrmann, S., Kretzschmar, H.-J., & Gatley, D. P. Thermodynamic properties of real moist air, dry air, steam, water, and ice (RP-1485). *HVAC&R Research*, 15(5) (2009) 961–986. DOI `10.1080/10789669.2009.10390874` — §54.3, **named and not implemented**; the enhancement factor it carries is what makes the ideal relations 0.44 % low in `W_s` at 25 °C, which §54.8 prints rather than tolerates.
- Gatley, D. P., Herrmann, S., & Kretzschmar, H.-J. A twenty-first century molar mass for dry air. *HVAC&R Research*, 14(5) (2008) 655–662. DOI `10.1080/10789669.2008.10391032` — §54 (where `M_a = 28.966 g/mol`, and hence `eps = 0.621945`, come from)
- Herrlin, M. K. Rack cooling effectiveness in data centers and telecom central offices: the Rack Cooling Index (RCI). *ASHRAE Transactions*, 111(2) (2005) 725–731. *(ASHRAE Transactions of this vintage carries no DOI; stable record `https://www.semanticscholar.org/paper/99b942df4aa448a1e06f77d36b48d5d52a40c6e0`.)* — §55.1
- Herrlin, M. K. Airflow and cooling performance of data centers: two performance metrics. *ASHRAE Transactions*, 114(2) (2008) 182–187. *(No DOI; same caveat.)* — §55.2 (RTI)
- Sharma, R. K., Bash, C. E., & Patel, C. D. Dimensionless parameters for evaluation of thermal design and performance of large-scale data centers. *AIAA 2002-3091* (2002). DOI `10.2514/6.2002-3091` — §55.3 (SHI and RHI)
- ASHRAE Technical Committee 9.9. *Thermal Guidelines for Data Processing Environments*, 5th ed., ASHRAE (2021). ISBN 978-1-947192-90-4 — §55.1 (the Class A1–A4 **recommended**, 18–27 °C, and **allowable** envelopes RCI is measured against)
- Wibron, E., Ljung, A.-L., & Lundström, T. S. *Energies*, 12(8) (2019) 1473. DOI `10.3390/en12081473` — **CC-BY-4.0, licence verified live through the Crossref REST API**, but **the full text was not reachable from this environment**, so §55.8's six-configuration ranking gate is **NOT run** and only the one relation the abstract states is gated. — §55.8
### Validation data
- Ghia, U., Ghia, K. N., & Shin, C. T. *Journal of Computational Physics*, 48 (1982) 387–411. — the lid-driven cavity's tabulated centreline profiles
- Moser, R. D., Kim, J., & Mansour, N. N. *Physics of Fluids*, 11 (1999) 943. — DNS channel profiles at `Re_tau` 180 / 395 / 590, and the sublayer `k+ ≈ C_v (y+)^2` with `C_v ≈ 0.07` of §15.2
- Driver, D. M., & Seegmiller, H. L. *AIAA Journal*, 23 (1985) 163–171. — the backward-facing step's reattachment length, `x_r/h = 6.26 ± 0.10`. Named in §22 and **NOT run** by §41's section.
- McCaffrey, B. J. **NBS TN 910** (1979). — the buoyant plume's centreline temperature and velocity decay correlations, `ΔT ~ z^{−5/3}` in the plume region
- Martin, J. C., & Moyce, W. J. *Philosophical Transactions of the Royal Society A*, 244 (1952) 312. — the dam break's surge-front position against time

---

## Contact

**simul@msimul.com** · Iterations Co., Ltd. / 주식회사 이터레이션즈
In collaboration with Meteo Simulation Co., Ltd. / 주식회사 메테오시뮬레이션

Teaching and academic research are free. Industrial R&D, research institutes outside
an educational institution, contract work and commercial use require a licence — see
sections 2 and 3 of [`LICENSE`](LICENSE). If the boundary is unclear, please ask.
