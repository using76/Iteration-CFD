# meteor-cfd

**A GPU-resident finite volume CFD solver**

Meteo Simulation Co., Ltd. · Rust host, CUDA kernels

[한국어](README.md)

---

## Overview

meteor-cfd is an unstructured finite volume CFD solver designed so that the
entire time-integration loop stays on the GPU. Once the mesh and fields are
uploaded, no device allocation and no field transfer to the host occur inside
the loop.

The numerical core is implemented directly from published literature. Every
formulation is specified in [`rust/SPEC-LIT.md`](rust/SPEC-LIT.md) with a
citation to its original paper. Validation uses the method of manufactured
solutions, analytical solutions and published benchmarks only — never a
comparison against another CFD code.

| | |
|---|---|
| Languages | Rust 1.85 (host), CUDA C++ (kernels) |
| Precision | Double by default; single via the `single` feature |
| Target | NVIDIA GPUs |
| Dependencies | cudarc, thiserror (AMGX optional) |
| Validation | 905 unit tests across all targets (814 in the lib), 314 `ofgpu-validate` checks |

---

## Licence

**The source is published, but this is not Open Source.**

| Use | Terms |
|---|---|
| Teaching — courses, coursework, lab classes, self-study | **Free** |
| **Academic research** — universities, schools, and their research institutes and laboratories | **Free** |
| Publishing the results of that research — papers, theses, talks | **Free**, with acknowledgement |
| Industrial R&D | Commercial licence |
| Government or national research institutes not part of an educational institution | Commercial licence |
| Consultancy, contract research, testing services | Commercial licence |
| Design, certification or operation of a real product, plant or system | Commercial licence |

Research funded by a company remains within the free grant **provided the
results are published openly**. Only sponsored work whose results are
confidential, or in which the sponsor holds exclusive rights, requires a
commercial licence.

If you publish results obtained with meteor-cfd, acknowledge it. This is a
condition of the grant, not a request:

```
meteor-cfd, Meteo Simulation Co., Ltd., https://github.com/using76/meteor-cfd
```

For cases where the boundary is unclear — a national laboratory, a hospital
research group, a non-profit, a university spin-out — please ask. Where the use
is not commercial in substance, no-cost or reduced-cost terms are available.

**Licence enquiries: simul@msimul.com**

The governing licence is the **Meteor Simulation Source-Available License,
Version 1.1**; the licensor is 주식회사 메테오시뮬레이션 (Meteo Simulation Co.,
Ltd.). On 23 August 2036 the version published as of that date becomes available
under the Apache License 2.0. See [`LICENSE`](LICENSE) for the full text and
[`NOTICE`](NOTICE) for third-party notices.

> This licence borrows the structure of the Business Source License 1.1 but is
> not BUSL. BUSL grants all non-production use free of charge, which would
> include a company's internal R&D. Here the free grant is limited to teaching
> and academic research.

---

## Capabilities

### Discretisation

| | |
|---|---|
| Convection | Gauss linear, upwind, linearUpwind, cubic, QUICK, Gamma, blended |
| TVD limiters | minmod, van Leer, van Albada, Superbee, MUSCL, Sweby-φ |
| Gradients | Green–Gauss, least squares, cell-limited and face-limited (Barth–Jespersen, Venkatakrishnan) |
| Surface-normal gradient | uncorrected, corrected, limited α |
| Time integration | steadyState, Euler, BDF2 (variable time step), local time stepping |
| Diffusion | Over-relaxed non-orthogonal correction with corrector iterations |

### Pressure–velocity coupling

SIMPLE, SIMPLEC, PISO and PIMPLE. Rhie–Chow interpolation, with body forces
applied on faces rather than interpolated from cell values.

### Turbulence

| | |
|---|---|
| RANS | Standard k-ε, Wilcox k-ω, Menter k-ω SST, Launder-Sharma low-Re k-ε (SPEC-LIT §33 — the only model `wallTreatment lowRe` is valid under; its damping functions integrate through the viscous sublayer, checked against the analytic `Re_t` limits and against a live channel's own `u+`/`y+` law of the wall), **realizable k-ε** (Shih et al., SPEC-LIT §40 — `C_mu` a field, so the Boussinesq normal stress cannot go negative; gated on that directly rather than on a channel, and on the homogeneous-shear fixed point its own coefficients imply) and **RNG k-ε** (Yakhot & Orszag, SPEC-LIT §41 — the `R` term absorbed into a per-cell `C_e2*`, diffusivities `α(ν + ν_t)` rather than `ν + ν_t/σ`) |
| LES | Smagorinsky, WALE, Deardorff |
| LES filter widths | Cube root of volume, maximum edge length, Scotti anisotropy correction, van Driest damping |
| LES wall model | Werner–Wengle (1991) — an analytically invertible power law integrated over the first cell's wall-parallel average speed, no Newton iteration; the `standard`/`spalding` presets both collapse to this one model under LES, `lowRe` gives `nu_t,w = 0`, and `rough` is refused by name (§13.4) since no rough LES wall model exists yet |
| Wall functions | nutk, nutU (inverse Spalding law), nutLowRe, rough walls (Cebeci–Bradshaw), epsilon, omega, kqR, kLowRe |
| Turbulence selection in the coupled solvers (`ofgpu-buoyant`, `ofgpu-fire`) | `ofgpu-buoyant`: the `CoupledTurbulence` trait dispatches on the case's own `RAS { model ...; }`/`simulationType`, exactly as the standalone drivers do — k-ε, k-ω, k-ω SST (wall distance computed automatically) and LES (Smagorinsky/WALE/Deardorff, with §16's filter widths and van Driest damping) all construct the ACTUAL model asked for, and buoyancy production `G_b` is wired into the right equation for each (§17, §30.2). `ofgpu-fire`: still k-ε only — its combustion mixing-time closure and thermal wall function need `epsilon` directly, so any other model is refused by name, a §13.4 error, not a silent substitution |
| Wall-model presets (`wallTreatment`) | `standard`/`spalding`/`rough`/`lowRe` — one setting expands to a CONSISTENT row of per-field patch types (nut/k/epsilon/omega, and T when the energy equation is solved) at case-build time; a hand-mixed row across families is refused by name, `-permissive` substitutes the row implied by the `nut` choice (SPEC-LIT §29.1). `lowRe` additionally requires a turbulence model with near-wall validity — `LaunderSharmaKE` (SPEC-LIT §33) is the one model on that menu, `kEpsilon`/`kOmega`/`kOmegaSST`/`realizableKE`/`RNGkEpsilon` still are not, so `lowRe` under any of the latter three is refused by name rather than left to diverge (SPEC-LIT §32) |
| Thermal wall function | Jayatilleke's sublayer-resistance correction to the thermal log law (`thermalWallFunction`, alias `compressible::alphatJayatillekeWallFunction`) — every preset row applies it to `T` on walls except `lowRe`, which leaves the resolved sublayer's own molecular resistance alone (SPEC-LIT §29.3); wired into `ofgpu-fire`'s energy equation. Validated against Dittus-Boelter/Gnielinski on a fixed-heat-flux periodic PLANE CHANNEL (SPEC-LIT §32/§34): the **wall-function leg CLOSES** — Gnielinski at Petukhov's smooth-pipe `f` −5.9% (±10%), Dittus-Boelter −12.9% (±20–25%) — and the **resolved `lowRe` leg does NOT** (+11.9% Gnielinski, +4.0% Dittus-Boelter). At each leg's own MEASURED wall friction factor the Reynolds-analogy verdict closes on neither (+34.3%, +14.9%). Numbers are the record at the SHIPPED DEFAULT `PrtModel constant`, rerun after SPEC-LIT §26.1. Selecting SPEC-LIT §37's Kays-Crawford variable `Pr_t` (opt-in, one token, nothing tuned) moves the resolved leg from +11.9% to **+4.3%** and closes the absolute-prediction verdict on BOTH legs, while moving the wall-function control only −0.06% of `Nu`; the Reynolds-analogy verdict on the wall-function leg is untouched at +34.0%, being a friction finding rather than a thermal one. **SPEC-LIT §26.1 closed the energy imbalance this gate carried as an uncertainty on every one of those numbers**: §25.1's divergence constraint was implemented without its conduction term `div(k_eff grad T)`, so the resolved leg's steady bookkeeping was short by +3.11% (+3.35% under Kays-Crawford); it is now +0.000089%, and the resolved leg's Kays-Crawford pass no longer needs an error bar quoted beside it. Full account, including what it does NOT establish, in `docs/07-fire-solver.md` §1.1 |
| Conjugate heat transfer (SPEC-LIT §46/§47) | **Solid regions, contact resistance and the fluid/solid interface.** The interface is a Robin triple on BOTH sides with the series conductance `h_G = (1/C_A + R_c + 1/C_B)^-1` (§47.2): flux continuity AND the contact-resistance jump hold **exactly at every iterate**, not at convergence, because ONE kernel writes both sides from one `h_G` and one `|Sf|`. It needs **no new matrix code** — `lduAmul` and `lduAddBoundaryContributions` key on `bNbrCell >= 0`, not on "cyclic", so a conformal interface in a concatenated fluid+solid cell numbering is already solved implicitly; a conjugate interface IS a cyclic couple with a zero transform (§47.3/§47.4). Two CUDA lines and one new kernel pair were all it took. Solid side: `(rho_s c_s) dT/dt = div(K_s grad T) + q'''`, Patankar's **harmonic** multi-material face conductivity, and mesh-axis-diagonal anisotropic `K` (§46) — a full tensor on a skewed mesh is **refused by name** with the measured residual, MPFA and the two ways out, never approximated by its diagonal. Dirichlet-Neumann partitioning is **never implemented**: Meng et al. (2017) Theorem 1 gives its amplification as `K_R/K_L`, which fails exactly at air/plastic. Gates, all live: the two-layer slab `q = dT/(L1/k1 + Rc + L2/k2)` to `4e-14` after ONE assembly and ONE solve; flux continuity on the FIRST unconverged iterate to `1.8e-16`; conservation `2.0e-16`; `k_solid -> 0` contributes **bitwise nothing** (= `fixedFluxTemperature`, `q = 0`) and `k_solid -> infinity` reproduces the `fixedValue`/`thermalWallFunction` wall; the two half-spaces' interface sits at the effusivity-weighted mean and is constant in time to `1.1e-11` of `dT`. Reached from a case file by `ofgpu-cht` — `cases/dieStack.cht.jsonc`, a 100 W die through three contact resistances to a cold plate, junction temperature **649.7118 K** against its own closed form |
| Wall distance | Poisson method (Tucker 1998) |
| Buoyancy production | G_b term (Rodi 1987, Henkes et al. 1991) |

### Multiphase and transport

| | |
|---|---|
| VOF | Interface compression, Zalesak FCT bounding, sub-cycling, CSF surface tension, p_rgh formulation |
| Scalar transport | Temperature, multi-component species (sum-to-one enforced) |
| Source terms | Volumetric heat release, momentum sources, Darcy–Forchheimer porous drag |
| Buoyancy | Non-Boussinesq density ratio `b = g(T_ref/T − 1)` |

### Linear solvers

| | |
|---|---|
| Krylov | PBiCGStab, PCG |
| Preconditioners | None, Jacobi, multi-colour DIC, multi-colour DILU |
| Pressure backends | Iterative, cuFFT direct solve, AMGX (optional feature) |
| Backend selection | Applicability filter → accuracy verification → measured timing |

### Mesh generation

| | |
|---|---|
| Block mesh | Per-case structured grids — grading, boundary patches and the `0/` fields written together |
| STL obstacles | Binary and ASCII STL carve the block into a castellated mesh — closure validation (refused with the open-edge count; `-permissive` proceeds on parity voting), column-parity classification with a 3-axis majority vote, new wall patches receive the existing wall boundary conditions (`-stl [name=]path`, Aftosmis et al. 1998; Barill et al. 2018) |
| Cut cells | The next stage past castellation — intersected cells keep reduced volume/area fractions and a closure-defined cut face instead of being removed (supersampling, default 16³, SPEC-LIT §24); slivers below `theta_min` merge into the fluid neighbour sharing the largest shared face |
| Gmsh `.msh` | v4.1 tetrahedron/hexahedron/prism/pyramid elements, `$PhysicalNames` patches — read from the published format specification |
| Cyclic patches | Any number of opposite block-face pairs coupled instead of left as boundaries (SPEC-LIT §31.1, generalised to multiple pairs by §34.2 — a plane channel periodic in two directions, or a fully periodic box in three, can be declared today) — `ofgpu-generate-mesh -cyclic x\|y\|z` (repeatable) or a JSONC `mesh.cyclic` array, each entry naming both sides and the transform (`translate` only; `rotate` is refused by name). Face matching by nearest translated centroid, checked against two invariants per pair — every face matches exactly once, and `Sf_a == -Sf_b` after the transform to a stated tolerance — either failing names the patch pair and the worst offending face rather than building a mesh that silently fails to conserve; an axis named by two pairs, or a pair sharing a slot with a constraint patch (below), is refused by name rather than silently resolved |
| Constraint patches (`empty`/`symmetry`) | `PatchKind::Empty` and `PatchKind::Symmetry` (SPEC-LIT §4's BC triple, the operator branches and vector reflection every driver already carries) newly nameable from JSONC (`"kind": "empty"`/`"symmetry"`, SPEC-LIT §34.1) — a case that needs a genuinely 2-D domain no longer has to be written in the OpenFOAM case-directory format only. A CONSTRAINT, not a boundary condition: a rule of either kind may not also set a per-field BC (refused by name), and `empty` is refused on any axis with more than one cell (naming the slot and its actual cell count) |

### Fire physics (low-Mach variable density, combustion, radiation)

`ofgpu-fire` wires SPEC-LIT §25–28 together into one solver.

| | |
|---|---|
| Low-Mach formulation | `p = p0(t) + p~(x,t)` split, the divergence constraint, `p0(t)` integration in a sealed or open compartment (Rehm & Baum 1978) |
| Energy equation | Sensible enthalpy, `k_eff = k + rho cp nu_t/Prt`, fixed-flux and fixed-temperature wall conditions, the Jayatilleke thermal wall function on `thermalWallFunction` walls (SPEC-LIT §29.3) |
| Combustion | Mixing-controlled single-step EDM (Magnussen & Hjertager 1977, **the default**) — `Y_F`/`Y_O2`/`Y_P` transport, a fuel-depletion clip, fuel mass consumed and heat released agreeing exactly (to round-off). **And the serial two-step mixing-controlled scheme** (McGrattan, McDermott & Floyd, ISFEH10 2022 — SPEC-LIT §42, selected by `scheme serialTwoStep`): the SAME mixing-controlled rate applied twice **serially** inside one time step, so the oxygen step 1 left over oxidises the CO step 1 made. No Arrhenius rate, no Jacobian, no ODE integrator, no stiffness. One extra transported species `Y_I`, and `Y_CO = f_CO Y_I` written out |
| Local extinction | The FDS `EXTINCTION 1` critical-flame-temperature predicate (SPEC-LIT §43, `extinctionModel oxygen`) — a piecewise-linear limiting oxygen index against cell temperature, a free-burn cut-off and an auto-ignition rule. Defaults to `none`, so every recorded result is unmoved |
| Radiation | Gray P1 approximation (Modest ch. 15), Marshak wall condition, a `chi_r` radiant-fraction floor — **and gray fvDOM** (Modest ch. 16; Fiveland 1984; Truelove 1987 — SPEC-LIT §36): the same RTE along 24 level-symmetric S4 ordinates, `radiationModel` selects between them. Measured on `cases/burnerPlume.jsonc` (32,768 cells, 1,200 steps, RTX 5070 Ti): radiated fraction 14.97% (P1) vs 13.79% (fvDOM), wall time 19.22 s vs 121.5 s |
| Validation gates | Sealed-box `dp0/dt` ramp (analytic), exact burner heat release, radiative equilibrium, cut-cell closure, msh hex closure — all permanent `ofgpu-validate` checks |
| Field output & restart | `-output foam,vtu,nvdb,vdb,usda` and `-writeInterval` write `U`, `p`, `T`, the turbulence closure and any species fields the same way `ofgpu-buoyant`/`ofgpu-vof` do; `-restartWrite N`/`-restartFrom FILE` checkpoint and resume — `p0`, `dp0dt` and the species mass fractions are carried across the restart, not only `U`/`p`/`T`, because a low-Mach run's thermodynamic state is more than those three fields. 40 steps continuous vs. 20+restart+20 agree on the first post-restart pressure residual, `p0` and total enthalpy |
| Volumetric sources | `sources[]` (JSONC) or `constant/fvSources` (OpenFOAM case directories) register a source on the momentum equation — a uniform body force over the whole domain, the one a periodic (cyclic-patch) case needs since it has no inlet to prescribe a mass flow from |

### Case input formats and restart

| | |
|---|---|
| JSONC case | One JSON file (comments and trailing commas allowed) naming mesh, physics, boundaries, numerics, sources and the fire block — the schema is generated by `schemars` from the same types the reader uses, so the two cannot disagree |
| Restart (`.mcr`) | Full double precision, `phi` included, refused on a mesh-hash mismatch, versioned |
| Visualisation/interchange output | VTU (appended binary, polyhedra preserved), NanoVDB/OpenVDB (`.vdb`/`.nvdb`, `fp32` or `fp16` voxels), a USD (`.usda`) scene referencing them |
| The case file drives them (SPEC-LIT §44) | `output.visualisation` (`format`, `interval`, `fields`, `precision`, `usdScene`), `output.exact` (`format`, `interval`) and `output.restart` (`interval`, `keep`) are read by every driver that reads a JSONC case. `fields` selects and orders and refuses a name the run does not have, listing what it does; `precision` is `fp16`/`fp32` on the two volume writers and an error anywhere else; `keep` retains the N most recent checkpoints and deletes older ones — only ones **this run wrote**, never anything else in the directory |

---

## GPU residency

Once the mesh and fields are uploaded, inside the time loop:

- No `cudaMalloc` — every buffer is allocated once at construction.
- No `cudaMemcpy` of field data — fields, fluxes and matrices all stay on the
  device.
- The Krylov solvers' control scalars (α, β, ω, ρ, residuals) also live in
  device memory. Single-thread kernels update them and the axpy kernels
  dereference the device pointers directly.

Exactly two scalars ever reach the host:

| | Size | When | Disable |
|---|---|---|---|
| Linear-solver convergence flag | 4 B | Every `checkInterval` iterations | `-fixedIters N` |
| Residual log | 3 × 8 B | On completing an equation | `-fixedIters N` |

With `-fixedIters` there is no host transfer at all, and only in that state can
a whole time step be captured as a CUDA graph.

Every operation that would accumulate into a diagonal by scattering over faces
is instead a **gather** over a cell→face CSR. No double-precision atomics are
needed, the summation order is fixed, and results are bitwise reproducible.

---

## Case settings: honoured or refused

An unsupported setting is never quietly replaced with something else. Every
entry in a case file is treated in one of three ways:

| State | Behaviour |
|---|---|
| Supported | Applied as written |
| Recognised but not implemented | Error naming the setting and the alternatives |
| Not recognised | Error naming the setting |

```
error: divSchemes/div(phi,k): "Gauss totalGarbage" is not supported by ofgpu;
       available: Gauss linear, Gauss upwind, Gauss linearUpwind [grad],
       Gauss cubic, Gauss QUICK, Gauss QUICKUnlimited, Gauss Gamma <0.1..0.5>,
       Gauss blended <0..1>, Gauss linearUpwindBlended <0..1>,
       Gauss limitedLinear <1..2>, Gauss vanLeer, Gauss vanAlbada,
       Gauss Minmod, Gauss SuperBee, Gauss MUSCL
  (run with -permissive to substitute Gauss upwind and continue)
```

`-permissive` is the only exception, and it prints what it substituted every
time.

Changing only the discretisation scheme on one case produces a different result
in each instance:

| `divSchemes` entry | `0/k` hash |
|---|---|
| `Gauss upwind` | `dec2a499fd69` |
| `Gauss linear` | `4c774d8fd354` |
| `Gauss vanLeer` | `e3315377c41a` |
| `Gauss linearUpwind grad(U)` | `b9ce961dad61` |
| `Gauss QUICK` | `05413b401b03` |
| `Gauss totalGarbageScheme` | Error, non-zero exit |

The same three-way rule also covers whole SETTINGS COMBINATIONS, not only one
entry at a time: a transient run (`run.endTime > 0` and `ddt` not
`steadyState`) naming the steady `SIMPLE` algorithm, or a steady run naming a
transient `PISO`/`PIMPLE`, is refused by name rather than executed with
under-relaxation fighting a steady state the case does not have —
`cases/burnerPlume.jsonc` reached `Inf` around step 20 exactly this way, with
`endTime`, `ddt` and the algorithm dictionary each individually valid and
nothing having warned:

```
error: numerics/algorithm: "SIMPLE (ddt "Euler", endTime 6)" is a steady
       algorithm on a transient case (endTime > 0 and ddt is not steadyState)
  available for a transient run: PISO, PIMPLE
  (run with -permissive to substitute PIMPLE with one outer corrector and continue)
```

**A block of the case format that no driver implements is not exempt either**
(SPEC-LIT §13.4.2). `ofgpu-fire` used to print a one-line "nobody reads this"
note for the `output` block and continue; `ofgpu-k-epsilon`, which reads the
same format, printed nothing — a note is per driver, and drivers drift. One
shared refusal now covers all of them:

| Setting | Treatment |
|---|---|
| the whole `output` block | **implemented** — SPEC-LIT §44. It was refused until then, and the reason is worth keeping: `visualisation.fields`, `visualisation.precision` and `restart.keep` had no implementation anywhere in the crate, so wiring the two that did exist (`format`, `interval`) and dropping those three in silence would have been §13.4.1's defect manufactured inside its own fix. §44 built all three first. What is still refused is the *combination*: a case that carries the block **and** a command line naming `-output`/`-writeInterval`/`-restartWrite` says the same thing twice, and that is an error naming both rather than a silent winner |
| `run.adjustTimeStep: true`, `run.maxCo` | **refused** — no driver that reads a JSONC case adjusts its own step. `ofgpu-vof` is the one adaptive loop in this crate (an OpenFOAM directory's `controlDict` `adjustTimeStep` + `maxCo` + `maxDeltaT`, or `-maxCo`) and it now actually reads all three |
| `controlDict/adjustTimeStep yes` (OpenFOAM) | **refused** in `read_control_dict`, once, for every driver that goes through it |
| `physics.gravity` / `constant/g` under `ofgpu-k-epsilon` or `ofgpu-k-omega` | **refused** — both models have had a `set_buoyancy` all along and nothing called it; §17's `G_b` needs a temperature field and these two drivers read none. The error names `ofgpu-plume`, `ofgpu-buoyant` and `ofgpu-fire` |

`-permissive` prints what it substituted for each of these and continues.

---

## Validation

```
cargo test        946 passed, 0 failed, 2 ignored (lib)
                  1057 passed, 0 failed, 4 ignored (every target — including the
                  per-binary CLI-parsing suites and SPEC-LIT §13.4.1's per-driver
                  "two runs must differ" pair tests)
ofgpu-validate    449 / 449 checks passed (401 computed live, 48 replayed from recorded measurements)
```

**SPEC-LIT §13.4.1's standing requirement**: two short runs of a driver,
differing in exactly one setting of the case file and nothing else, must write
DIFFERENT output. If they are bit-identical the setting is inert. All six
drivers now carry such a pair test — `ofgpu-fire` 13 settings, `ofgpu-buoyant`
17, `ofgpu-vof` 15, `ofgpu-plume` 11, `ofgpu-k-epsilon` and `ofgpu-k-omega` 11
each — each running the driver's own `parse` + `run` and comparing every field
file written.

### Order of convergence — method of manufactured solutions

`−∇²ψ = f`, mesh spacing halved.

| Mesh | Coarse L2 | Fine L2 | Observed order |
|---|---|---|---|
| 3-D graded (10³ → 20³) | 7.943 × 10⁻³ | 1.857 × 10⁻³ | **2.10** |
| 3-D sheared (8³ → 16³) | 4.350 × 10⁻³ | 1.154 × 10⁻³ | **1.91** |
| 2-D with empty patches (16² → 32²) | 4.075 × 10⁻³ | 9.711 × 10⁻⁴ | **2.07** |

### Published benchmarks

**Lid-driven cavity** — Ghia, Ghia & Shin (1982) Tables I and II, 80 × 80.

| Re | SIMPLE iterations | Momentum residual | max \|Δu\| | max \|Δv\| |
|---|---|---|---|---|
| 100 | 3,000 | 1.011 × 10⁻⁴ | 0.0046 | 0.0088 |
| 400 | 6,000 | 7.382 × 10⁻⁴ | 0.0067 | 0.0057 |

The Table II entry at Re = 400, x = 0.9063 (−0.23827) is excluded from the
comparison as a typographical error in the paper. It breaks the monotone run in
the paper's own table — from −0.22847 at x = 0.9453 to the minimum −0.44993 at
x = 0.8594 — and Nilsson & Wallin (2022) §5.2 exclude it for the same reason.
The tabulated data is kept unedited and the excluded station is marked in the
output.

### VOF

**Dam break** — 6,000 cells, 0.25 s, 1,250 time steps, 118 s wall clock.

```
phase volume 1.256250e-05 → 1.256250e-05    (relative change 1.35 × 10⁻¹⁶)
alpha in [-4.163e-17, 1]
```

| Check | Result |
|---|---|
| Zalesak rotating slotted disc: min α ≥ 0 | 1.7 × 10⁻¹⁸ |
| Zalesak rotating slotted disc: phase volume conserved | 3.9 × 10⁻¹² |
| Static drop, Laplace pressure σ/R | 4.888 against 5.000 (2.2 %) |
| Sealed stratified tank stays at rest | max \|U\| = 5.5 × 10⁻¹¹ m/s (√gH = 3.13) |

The last of these is the decisive test of the p_rgh formulation.

### Buoyancy production, sources, species

| Check | Result |
|---|---|
| G_b sign — negative in stable stratification (dT/dz > 0) | Correct |
| G_b sign — positive above a heat source (dT/dz < 0) | Correct |
| G_b magnitude | 1.6 × 10⁻¹⁴ |
| Heat source injects exactly its rated power | 2.3 × 10⁻¹⁶ |
| Species mass fractions sum to 1 | 0.0 |

### Machine-precision checks

| Check | Error |
|---|---|
| Matrix assembly (diagonal, upper, lower, source, boundary coefficients) against an independent CPU implementation | ~2 × 10⁻¹⁶ |
| Under-relaxation, boundary folding, matrix–vector product | ~3 × 10⁻¹⁶ |
| PCG / PBiCGStab against a dense direct solve | 2.8 × 10⁻¹⁵ / 1.1 × 10⁻¹⁵ |
| cuFFT direct Poisson solve against the iterative solve of the same matrix | 1.4 × 10⁻¹⁵ |
| Hydrostatic balance | 6.6 × 10⁻¹⁵ |

The CPU reference is written deliberately as scatter loops, where the device
code gathers. Two implementations of different structure agreeing is what makes
the comparison meaningful.

### Wall treatment (SPEC-LIT §29)

Two permanent `ofgpu-validate` gates for the `wallTreatment` presets and the
Jayatilleke thermal wall function:

| Check | Result |
|---|---|
| `Ks → 0` reproduces the smooth `nutk` wall function everywhere | 0 (round-off) |
| `Ks → 0` reproduces the smooth `nutU` wall function everywhere | 0 (round-off) |
| `P(Pr/Pr_t = 1) = 0` exactly | 0 (round-off) |
| At `Pr = Pr_t`, `T+ == Pr_t · u+` everywhere | 1.3 × 10⁻¹⁶ |
| The `thermalWallFunction` Robin triple encodes exactly the analytic Jayatilleke flux (the one-cell conductance identity) | 0 (round-off) |
| Werner-Wengle: both branches agree at the branch point, and each branch's own closed form reproduces a manufactured `tau_w` to round-off | 0 (round-off) |
| Coupled-solver selection: `kOmegaSST` via `ofgpu-buoyant`'s `build_coupled`, on a buoyant case, yields a different `nut` FNV hash than `kEpsilon` on the identical case | hashes differ (decisive) |
| Thermal wall-function gate, Nusselt verdict (replayed measurement) — `cases/channelPeriodicFluxWF.jsonc`'s own numbers, against Gnielinski at the Petukhov pipe `f` / Dittus-Boelter | −5.9% / −12.9% (inside both ±10% / ±20–25% bands — **closes**) |
| Resolved-leg mesh resolution (replayed measurement) — `cases/channelPeriodicFluxLowRe.jsonc`'s worst wall-adjacent y+ and cells-below-y+-20 count | y+ = 0.00179, 192/400 cells (both requirements met) |
| Resolved-leg Nusselt verdict (replayed measurement) — same case, same two correlations | +11.9% / +4.0% (inside the DB band, outside Gnielinski's — **does not close**, and since SPEC-LIT §26.1 the leg carries ±0.0001% of energy-balance uncertainty, so the miss is decisive) |
| Thermostat weighting, the decisive experiment (replayed) — four runs, `"weighting"` the only token changed | `massFlux` lowers `Nu` and widens `T_w − T_b` on both legs, and moves the resolved mesh 2.7× more than the wall-function one (−3.72% vs −1.38%, re-measured after §26.1) |
| Bounded convection on momentum, the isolation experiment (replayed) — seven runs over `div(phi,U)` ∈ {`Gauss upwind`, `Gauss linearUpwind grad(U)`} × {plain, `bounded`} | dropping `bounded` closes the kinematic drag balance on both legs (−3.787% → −0.000%, −0.112% → −0.005%); the scheme's ORDER is worth < 0.3% of `Nu`. **Re-run after §26.1 the same token leaves +0.000% on the resolved leg** — the dilatation those −3.787% were integrating against was itself the artefact of an incomplete `Q`. §3.1's rule is unchanged |
| Thermostat sign and steady-state offset — source when cold, sink when hot, matches the closed form `target + Q·tau/rho_cp` | 0 (round-off) |

These establish that the rough-wall law collapses to the existing smooth one
at `Ks = 0` — the case that never mentions roughness — that the thermal wall
function's own algebra is internally exact, and that a case naming
`kOmegaSST` in a coupled solver actually gets it. They do **not** by
themselves establish that a coarse wall-function mesh agrees with an
independent published correlation on the wall heat flux of a real flow — that
claim needed a live run, and SPEC-LIT §32's redesigned gate is what makes it
checkable at all: three earlier fixed-wall-temperature attempts (ratios
0.095, 0.381, 0.107) compared two runs that, it turns out, had solved
different problems (a fixed `T_w` lets the bulk temperature float, so two
meshes with different near-wall conductances settle at different ΔT).

**Validated, and rebuilt as a genuine 2-D plane channel per SPEC-LIT §34 —
then rerun three times as three separate defects in the COMPARISON were
found and removed.** Fixing the SAME wall heat flux `q_w` on both meshes —
letting each predict its own ΔT — and comparing the result as a Nusselt
number against Dittus & Boelter (1930) and Gnielinski (1976) is the gate.
The gate used to run on a 3-D duct only because JSONC could not say `empty`;
now that it can (§34.1), `cases/channelPeriodicFluxWF.jsonc` is
streamwise-cyclic, `empty` front/back, hot walls top and bottom, and nothing
else. The three defects, each fixed and each followed by a full rerun of
both legs, were: a UNIFORM domain heat sink where the compensation should be
mass-flux weighted (§35.3); a friction factor INFERRED from the body force
rather than MEASURED at the wall (§32.5); and — the one that moved the
verdict — a driver that read **none** of the case's own `numerics` block, so
the momentum equation ran `bounded Gauss upwind` on two cases asking for
`Gauss linearUpwind grad(U)` (§13.4.1, §32.5.5); and — the one that closed the last open
item — §25.1's low-Mach divergence constraint implemented without its
conduction term `div(k_eff grad T)`, which was the whole of the resolved leg's
+3.11% energy imbalance (§26.1). The current numbers, both legs at 40 000
iterations on the settings the case files actually name:

| | wall-function leg | resolved `lowRe` leg |
|---|---|---|
| y+ (wall-adjacent) | 56.88 / 57.77 / 58.57 | 0.00179 (192 of 400 cells below y+ 20) |
| `T_w` (diagnosed) / `T_b` (mixed-mean) | 317.497 K / 293.251 K | 314.549 K / 292.773 K |
| `U_b` / Re | 5.39407 m/s / 28 768 | 4.93682 m/s / 26 330 |
| **Nu (measured)** | **64.4894** | **71.6830** |
| Gnielinski at Petukhov's smooth-pipe `f` (ABSOLUTE PREDICTION) | −5.9% — **inside ±10%** | +11.9% — **outside** |
| Dittus-Boelter | −12.9% — inside ±20–25% | +4.0% — inside |
| Gnielinski at this leg's own MEASURED `f` (REYNOLDS ANALOGY) | +34.3% — outside | +14.9% — outside |
| energy balance (thermostat power vs measured wall heat) | +0.0174% | **+0.000089%** |
| kinematic force balance (§32.5.2) | −0.005% | −0.000% |
| `contErr` floor | 2.0×10⁻⁸ | **6.7×10⁻¹⁴** |

`D_h = 2H = 0.08 m` is the only hydraulic diameter on the table: for a
genuine plane channel the heated and wetted perimeters COINCIDE, so there is
no convention to choose. `ofgpu-validate` replays both legs' measurements on
every run, permanently.

**The verdict, stated once.** At the SHIPPED DEFAULT (`PrtModel constant`) the
wall-function leg CLOSES under §32.4's absolute-prediction verdict and the
resolved leg does NOT — and that miss is now decisive in a way it has never
been, because the ±3.1% energy-balance uncertainty that used to be quoted
beside it is gone: the balance closes to 0.0001%, so there is no bookkeeping
gap left to hide any part of the 11.9% behind. Under the REYNOLDS-ANALOGY
verdict, taken at the friction factor each leg's own wall measures, the gate
closes on NEITHER leg; the "+6.4% / +6.8%, both legs pass" once published here
rested on friction factors inferred from the body force, which measurement
showed to be 8–25% wrong.

Selecting SPEC-LIT §37's Kays-Crawford `Pr_t` on both legs — opt-in, one
token, nothing tuned — closes the absolute-prediction verdict on BOTH:
**−7.3%** and **+4.3%**, with Dittus-Boelter at −14.1% and −3.1%. On the
resolved leg the Reynolds-analogy verdict closes too, at +7.7%, which it could
not be said to do before §26.1: its ±3.35% band then straddled the edge, and
now there is no band.

**What the remainder implicates.** Removing the `bounded` token from the
momentum equation closed the resolved leg's kinematic drag imbalance
outright, from −3.787% to −0.000%, and a seven-run isolation over
`div(phi,U)` shows why: the scheme's ORDER is worth less than 0.3% of `Nu`,
while `bounded` alone carries the whole imbalance. SPEC-LIT §3.1 records the
rule that came out of it — a driver may not default a momentum equation to
the bounded form, because §25.1 makes `div u` a prescribed physical quantity
in a low-Mach flow, not a convergence error to be subtracted away. **Two later corrections to that paragraph, both from measurement.** First,
§26.1 showed the −3.787% it reports to be unreproducible on the fixed solver:
the dilatation §3.1's correction was integrating against was itself the
artefact of an incomplete `Q`, and with `Q` complete the same `bounded` run
closes the drag balance to +0.000%. §3.1's rule is unchanged — a fire plume's
expansion is real and the correction still eats it — but the channel is no
longer the case that demonstrates it. Second, the resolved leg's +3.11% energy
imbalance, which that paragraph left as an open item needing code, is what
§26.1 went after and closed: the whole of it was §25.1's conduction term,
missing from the divergence constraint. Two candidate fixes were run and both
are refutations — dropping the energy equation's bounded correction closes the
balance and gives `Nu` = 128.5, and subtracting the part §25.1 prescribes
diverges the case to 605 K.

What is left is the constant `Pr_t = 0.85` reaching a first cell at
y+ = 0.0019 (Kays 1994 reports `Pr_t` rising to ~1.5–1.9 in the sublayer), and
that one HAS been measured: SPEC-LIT §37's Kays-Crawford model moves the
resolved leg to +4.3% and the wall-function control by −0.06%, the asymmetry
the hypothesis predicted.

**The velocity field checks out, on both legs.** `LaunderSharmaKE` (SPEC-LIT
§33) checks out on every front now available: its damping-function limits are
exact (`ofgpu-validate`), it reproduces the viscous sublayer `u+ = y+` to
under 1% and the log law within 1% on a clean periodic channel, and its own
resolved leg converges its velocity field to round-off (`|U|` residual
`4×10⁻¹²`). `U_b/u_tau`, formed from each rerun's own printed `tau_w` and
`rho(T_b)`, is **18.3** on the resolved leg and **20.1** on the wall-function
one (viscous `tau_w`; 21.6 at that leg's `rho u_tau²` form) — the resolved
leg sits closer to the 15–17 a fully developed plane channel gives, as it did
before, but at the corrected numerics **neither leg is inside that range any
more** (they were 17.35 and 19.23 at the substituted `bounded Gauss upwind`).
That is reported, not explained. The duct-corner hypothesis an
earlier round left untested is CONFIRMED: removing the corners fixed the
velocity collapse. The energy equation's earlier undamped DRIFT is fixed too
— SPEC-LIT §35 diagnosed it as a pure-Neumann null space (every thermal
boundary in a streamwise-periodic closed domain is Neumann, so the steady
temperature equation is singular up to an additive constant) and replaced the
old fixed `-heaterPower` sink with a proportional controller on the domain's
own volume-mean `T`. The regression that diagnosis needed passes decisively:
the resolved leg run from T0 = 293.15 K and from T0 = 400 K converges to the
IDENTICAL state, every printed digit.

See `docs/07-fire-solver.md` §1.1 for the full numbers in order — the
superseded duct-era attempts, the law-of-the-wall table, the SPEC-LIT §35
diagnosis and fix, the friction-factor measurement, and the §13.4.1 rerun
that is the current statement.

---

## Performance

Measured on an NVIDIA GeForce RTX 5070 Ti (70 SMs, 896 GB/s), double precision.

One outer iteration is two complete transport equations: assembly,
under-relaxation, wall-function constraints, and a Krylov solve.

| Mesh | k-ε (ms/iter) | k-ω (ms/iter) | Mcell-iter/s | Device memory |
|---|---|---|---|---|
| 80 k | 1.187 | 1.150 | 67 / 70 | 1.4 GB |
| 500 k | 3.427 | 3.400 | 146 / 147 | 1.9 GB |
| 2 M | 13.346 | 13.337 | 150 / 150 | 4.0 GB |

Kernel launch overhead dominates on small meshes; above 500 k cells the solver
is memory-bandwidth bound.

### CUDA graphs

24,000 cells, 200 outer iterations.

| Mode | ms/iter | Mcell-iter/s |
|---|---|---|
| Adaptive (4-byte flag per iteration) | 1.323 | 18.1 |
| Fixed iterations, per-launch | 1.191 | 20.1 |
| Fixed iterations, **CUDA graph** | **0.377** | **63.7** |

A **3.16×** improvement, for a one-off capture and instantiation cost of
0.46 ms. The result is bitwise identical to the per-launch path across all
24,000 cells: a graph removes launch overhead without changing execution order.

### Pressure backend selection

Measured by the selector on an 82,320-cell plume mesh:

```
uniform cartesian    (98, 42, 20), h = (0.1494, 0.1486, 0.15)
separable bcs        true    symmetric  true    constant coefficient  true

  PBiCGStab   applicable    51.13 ms   (reference)      residual 7.19e-12
  cuFFT       applicable     2.05 ms   agrees to 8.0e-11
  AMGX        unavailable              feature 'amgx' not enabled

chosen: cuFFT   —  25.0x
```

With transfers disabled, cuFFT drops to 0.86 ms. The two solutions differ by
1.5 × 10⁻¹⁴ relative, which is what an exact inverse of the same matrix looks
like.

This gain holds only for a constant-coefficient Poisson equation. SIMPLE's
pressure equation carries a per-cell coefficient `rAUf`, and the selector
correctly rejects cuFFT for it.

### Cost of runtime dispatch

| Granularity | Cost |
|---|---|
| Virtual call per kernel launch | Below the noise floor |
| Virtual call per element | 1.75 – 1.80× |

Every runtime choice in this solver — SIMPLE/PISO, turbulence model, solver
backend — sits at launch granularity, so no per-combination build is required.

---

## Building

Requires Rust 1.85 or later, Visual Studio 2022 (C++ workload), and CUDA
Toolkit 13.x.

```powershell
cd rust
cargo build --release
cargo test  --release
```

`build.rs` runs `vcvars64.bat` to establish the MSVC environment and compiles
each `.cu` to **CUBIN** rather than PTX: when the toolkit is newer than the
version the driver reports, PTX fails with
`CUDA_ERROR_UNSUPPORTED_PTX_VERSION`.

`-Xcompiler=/Zc:preprocessor` is required because the CUDA 13 CCCL headers
raise `fatal error C1189` under the traditional MSVC preprocessor.

### Executables

| Executable | Purpose |
|---|---|
| `ofgpu-validate` | Numerical validation (314 checks) |
| `ofgpu-bench` | Throughput and memory benchmarks |
| `ofgpu-graph-bench` | CUDA graph against per-launch execution |
| `ofgpu-dispatch-bench` | Runtime dispatch cost |
| `ofgpu-probe` | Device properties |
| `ofgpu-generate-mesh` | Case generation |
| `ofgpu-k-epsilon`, `ofgpu-k-omega` | Turbulence models, standalone |
| `ofgpu-plume`, `ofgpu-buoyant` | Buoyant plume |
| `ofgpu-vof` | Two-phase VOF |
| `ofgpu-fire` | Low-Mach combustion and radiation (SPEC-LIT §25–28) |

---

## Running

```powershell
cd rust
cargo run --release --bin ofgpu-generate-mesh -- channel  ..\cases\channel  200 120 1
cargo run --release --bin ofgpu-k-epsilon     -- ..\cases\channel -iters 4000 -check 400
cargo run --release --bin ofgpu-generate-mesh -- damBreak ..\cases\damBreak  60 100 1
cargo run --release --bin ofgpu-generate-mesh -- plume    ..\cases\plumeCol  60 40 30 -stl column=column.stl
cargo run --release --bin ofgpu-vof           -- ..\cases\damBreak -endTime 0.25 -surge
cargo run --release --bin ofgpu-fire          -- ..\cases\burnerPlume.jsonc -combustion -radiation -endTime 6.0 -deltaT 0.005
cargo run --release --bin ofgpu-validate
```

| Option | Meaning |
|---|---|
| `-iters N` | Outer iterations (defaults to `endTime` in `controlDict`) |
| `-fixedIters N` | Run the linear solver exactly N times and never read the residual — no host transfer |
| `-check N` | Convergence check every N iterations |
| `-write NAME` | Time directory to write results into |
| `-noWrite` | Do not write results |
| `-permissive` | Downgrade unsupported-setting errors to warnings and report the substitution |

Generatable cases: `channel`, `cavity`, `step`, `big`, `plume`, `damBreak`.
`-stl [name=]path` (repeatable) carves any case's block mesh against a
triangulated surface — see [cases/README.md](cases/README.md).

The turbulence model is chosen by `RAS { model ...; }` or `simulationType LES;`
in `constant/momentumTransport`. Naming a model that is not implemented
produces an error listing those that are.

Cases are read and written in the OpenFOAM ASCII format — `constant/polyMesh`,
`0/`, `constant/physicalProperties`, `constant/momentumTransport`,
`system/{fvSolution, fvSchemes, controlDict}`. This is for interoperability
with existing pre- and post-processing tools such as ParaView and `foamToVTK`.
meteor-cfd links against no part of OpenFOAM and contains none of its source.
Convert binary-format cases to ASCII before use.

---

## Documentation

| File | Contents |
|---|---|
| [`rust/SPEC-LIT.md`](rust/SPEC-LIT.md) | The numerical specification, with a citation for every formulation |
| [`rust/PROVENANCE.md`](rust/PROVENANCE.md) | Per-file provenance and design decisions |
| [`LICENSING.md`](LICENSING.md) | Licensing audit |
| `docs/01-model-catalog.md` | Catalogue of CFD components (1,823 entries) |
| `docs/02-gpu-portability.md` | GPU portability classification |
| `docs/03-esi-vs-foundation.md` | Differences between upstream distributions |
| `cases/README.md` | Test case geometries |

---

## Limitations

- **No MPI or multi-GPU support.** Single GPU only.
- **AMGX is provided behind the `amgx` Cargo feature and is off by default.**
  NVIDIA's Windows support is limited and the newest verified toolkit is CUDA
  12.2, against 13.3 on the development machine. With the feature off, the
  backend selector still reports AMGX explicitly as unavailable.
- **Crank–Nicolson is implemented but unreachable from an under-relaxed
  equation.** The θ weighting and the implicit under-relaxation want the same
  slot in the assembly, and the relaxation must see the unweighted diagonal.
  Rather than silently falling back to Euler, the solver reports the reason.
- **No compressible or transonic capability.** Density-weighted time derivatives
  are implemented and used by VOF, but the pressure equation is incompressible.
- ~~**Combustion is mixing-controlled single-step (EDM) only.**~~ **Partly
  fixed (SPEC-LIT §42/§43).** The serial two-step mixing-controlled scheme and
  a local-extinction predicate are implemented, and predict CO, incomplete
  combustion and where the flame goes out. There is still **no finite-rate
  (Arrhenius) mechanism** — no Westbrook–Dryer, no Jones–Lindstedt, no stiff
  ODE integrator. §42 also keeps the molar mass `W̄` and the specific heat
  `c_p` CONSTANT, so density and thermal expansion are still computed with
  air's values even once CO2 and CO are distinguished. ~~**Radiation is gray P1 only.**~~ **Fixed (SPEC-LIT
  §36).** `fvDOM` (finite-volume discrete ordinates, 24-ordinate level-symmetric
  S4) is implemented and selected by `radiationModel`; both models share one
  `EnergySources` registration and the same `chi_r` floor. Radiation is still
  GRAY on both — no spectral (WSGG) band model, and the absorption coefficient
  does not vary with local soot/CO2/H2O.
- ~~Only one cyclic pair.~~ **Fixed (SPEC-LIT §34.2).** `BlockSpec::cyclic`
  is a list now, and a JSONC case's `mesh.cyclic` accepts any number of
  pairs (one per axis) — a plane channel periodic in two directions, or a
  fully periodic box in three, can be declared today. See "Cyclic patches"
  above.

---

## References

Sources for the numerical methods and models. Section numbers refer to
[`rust/SPEC-LIT.md`](rust/SPEC-LIT.md).

### Finite volume discretisation

- Jasak, H. (1996). *Error Analysis and Estimation for the Finite Volume Method
  with Applications to Fluid Flows.* PhD thesis, Imperial College London. — §2, §3
- Moukalled, F., Mangani, L., & Darwish, M. (2016). *The Finite Volume Method in
  Computational Fluid Dynamics.* Springer. — §2, §3, §11
- Ferziger, J. H., & Perić, M. *Computational Methods for Fluid Dynamics.*
  Springer. — §2.4, §3.3, §11.1, §11.5
- Patankar, S. V. (1980). *Numerical Heat Transfer and Fluid Flow.* Hemisphere.
  — §3.4, §5.2, §18

### Convection schemes

- Warming, R. F., & Beam, R. M. (1976). Upwind second-order difference schemes
  and applications in aerodynamic flows. *AIAA Journal*, 14(9), 1241–1249. — §11.2
- Leonard, B. P. (1979). A stable and accurate convective modelling procedure
  based on quadratic upstream interpolation. *Computer Methods in Applied
  Mechanics and Engineering*, 19(1), 59–98. — §11.3
- Khosla, P. K., & Rubin, S. G. (1974). A diagonally dominant second-order
  accurate implicit scheme. *Computers & Fluids*, 2(2), 207–209. — §11.1
- Jasak, H., Weller, H. G., & Gosman, A. D. (1999). High resolution NVD
  differencing scheme for arbitrarily unstructured meshes. *International Journal
  for Numerical Methods in Fluids*, 31(2), 431–449. — §11.6
- Sweby, P. K. (1984). High resolution schemes using flux limiters for hyperbolic
  conservation laws. *SIAM Journal on Numerical Analysis*, 21(5), 995–1011. — §7
- van Leer, B. (1977). Towards the ultimate conservative difference scheme IV.
  A new approach to numerical convection. *Journal of Computational Physics*,
  23(3), 276–299. — §7
- van Leer, B. (1979). Towards the ultimate conservative difference scheme V.
  A second-order sequel to Godunov's method. *Journal of Computational Physics*,
  32(1), 101–136. — §7
- van Albada, G. D., van Leer, B., & Roberts, W. W. (1982). A comparative study
  of computational methods in cosmic gas dynamics. *Astronomy and Astrophysics*,
  108, 76–84. — §7
- Roe, P. L. (1986). Characteristic-based schemes for the Euler equations.
  *Annual Review of Fluid Mechanics*, 18, 337–365. — §7
- Darwish, M., & Moukalled, F. (2003). TVD schemes for unstructured grids.
  *International Journal of Heat and Mass Transfer*, 46(4), 599–611. — §7

### Gradients and limiters

- Barth, T. J., & Jespersen, D. C. (1989). The design and application of upwind
  schemes on unstructured meshes. *AIAA Paper 89-0366.* — §12.2
- Venkatakrishnan, V. (1993). On the accuracy of limiters and convergence to
  steady state solutions. *AIAA Paper 93-0880.* — §12.2

### Time integration

- Crank, J., & Nicolson, P. (1947). A practical method for numerical evaluation
  of solutions of partial differential equations of the heat-conduction type.
  *Mathematical Proceedings of the Cambridge Philosophical Society*, 43(1),
  50–67. — §13.1

### Pressure–velocity coupling

- Patankar, S. V., & Spalding, D. B. (1972). A calculation procedure for heat,
  mass and momentum transfer in three-dimensional parabolic flows.
  *International Journal of Heat and Mass Transfer*, 15(10), 1787–1806. — §5.2
- Van Doormaal, J. P., & Raithby, G. D. (1984). Enhancements of the SIMPLE method
  for predicting incompressible fluid flows. *Numerical Heat Transfer*, 7(2),
  147–163. — §5.3
- Issa, R. I. (1986). Solution of the implicitly discretised fluid flow equations
  by operator-splitting. *Journal of Computational Physics*, 62(1), 40–65.
  — §5.4, §14
- Rhie, C. M., & Chow, W. L. (1983). Numerical study of the turbulent flow past
  an airfoil with trailing edge separation. *AIAA Journal*, 21(11), 1525–1532.
  — §5.1

### Turbulence — RANS

- Launder, B. E., & Spalding, D. B. (1974). The numerical computation of
  turbulent flows. *Computer Methods in Applied Mechanics and Engineering*, 3(2),
  269–289. — §6.1, §6.4
- Wilcox, D. C. *Turbulence Modeling for CFD.* DCW Industries. — §6.2
- Menter, F. R. (1994). Two-equation eddy-viscosity turbulence models for
  engineering applications. *AIAA Journal*, 32(8), 1598–1605. — §6.3
- Menter, F. R., Kuntz, M., & Langtry, R. (2003). Ten years of industrial
  experience with the SST turbulence model. *Turbulence, Heat and Mass Transfer*,
  4, 625–632. — §6.3
- Launder, B. E., & Sharma, B. I. (1974). Application of the energy-dissipation
  model of turbulence to the calculation of flow near a spinning disc. *Letters
  in Heat and Mass Transfer*, 1(2), 131–138. — §33
- Patel, V. C., Rodi, W., & Scheuerer, G. (1985). Turbulence models for
  near-wall and low Reynolds number flows: a review. *AIAA Journal*, 23(9),
  1308–1319. — §33
- Shih, T.-H., Liou, W. W., Shabbir, A., Yang, Z., & Zhu, J. (1995). A new
  k-epsilon eddy viscosity model for high Reynolds number turbulent flows.
  *Computers & Fluids*, 24(3), 227–238. Read as **NASA TM-106721 /
  ICOMP-94-21** (1994), a US government work in the public domain; the journal
  version is paywalled and was not read. — §40
- Yakhot, V., Orszag, S. A., Thangam, S., Gatski, T. B., & Speziale, C. G.
  (1992). Development of turbulence models for shear flows by a double
  expansion technique. *Physics of Fluids A*, 4(7), 1510–1520. Read as
  **ICASE Report 91-65 / NASA CR-187611** (1991). — §41
- Yakhot, V., & Orszag, S. A. (1986). Renormalization group analysis of
  turbulence. I. Basic theory. *Journal of Scientific Computing*, 1(1), 3–51.
  — §41
- Reynolds, W. C. (1987). *Fundamentals of turbulence for turbulence modeling
  and simulation.* AGARD Report No. 755. — §40 (the realizability
  constraints the variable `C_mu` is constructed to satisfy)

### Turbulence — LES

- Smagorinsky, J. (1963). General circulation experiments with the primitive
  equations. *Monthly Weather Review*, 91(3), 99–164. — §6.5
- Deardorff, J. W. (1970). A numerical study of three-dimensional turbulent
  channel flow at large Reynolds numbers. *Journal of Fluid Mechanics*, 41(2),
  453–480. — §16.1
- Deardorff, J. W. (1980). Stratocumulus-capped mixed layers derived from a
  three-dimensional model. *Boundary-Layer Meteorology*, 18(4), 495–527. — §6.5
- Nicoud, F., & Ducros, F. (1999). Subgrid-scale stress modelling based on the
  square of the velocity gradient tensor. *Flow, Turbulence and Combustion*,
  62(3), 183–200. — §6.5
- van Driest, E. R. (1956). On turbulent flow near a wall. *Journal of the
  Aeronautical Sciences*, 23(11), 1007–1011. — §16.4
- Scotti, A., Meneveau, C., & Lilly, D. K. (1993). Generalized Smagorinsky model
  for anisotropic grids. *Physics of Fluids A*, 5(9), 2306–2308. — §16.3

### Wall treatment

- Spalding, D. B. (1961). A single formula for the law of the wall. *Journal of
  Applied Mechanics*, 28(3), 455–458. — §6.4, §15.1
- Cebeci, T., & Bradshaw, P. (1977). *Momentum Transfer in Boundary Layers.*
  Hemisphere. — §15.3
- Jayatilleke, C. L. V. (1969). The influence of Prandtl number and surface
  roughness on the resistance of the laminar sub-layer to momentum and heat
  transfer. *Progress in Heat and Mass Transfer*, 1, 193–330. — §29.3
- Werner, H., & Wengle, H. (1991). Large-eddy simulation of turbulent flow
  over and around a cube in a plate channel. *8th Symposium on Turbulent
  Shear Flows.* — §30.1
- Tucker, P. G. (1998). Assessment of geometric multilevel convergence robustness
  and a wall distance method for flows with multiple internal boundaries.
  *Applied Mathematical Modelling*, 22(4–5), 293–305. — §6.6
- Dittus, F. W., & Boelter, L. M. K. (1930). Heat transfer in automobile
  radiators of the tubular type. *University of California Publications in
  Engineering*, 2, 443–461. (reprinted: *International Communications in
  Heat and Mass Transfer*, 12(1), 3–22, 1985.) — §32.3
- Gnielinski, V. (1976). New equations for heat and mass transfer in
  turbulent pipe and channel flow. *International Chemical Engineering*,
  16(2), 359–368. — §32.3

### Buoyancy

- Rehm, R. G., & Baum, H. R. (1978). The equations of motion for thermally driven
  buoyant flows. *Journal of Research of the National Bureau of Standards*,
  83(3), 297–308. — §9
- Spiegel, E. A., & Veronis, G. (1960). On the Boussinesq approximation for a
  compressible fluid. *Astrophysical Journal*, 131, 442–447. — §9
- Rodi, W. (1987). Examples of calculation methods for flow and mixing in
  stratified fluids. *Journal of Geophysical Research*, 92(C5), 5305–5328. — §17
- Henkes, R. A. W. M., van der Vlugt, F. F., & Hoogendoorn, C. J. (1991).
  Natural-convection flow in a square cavity calculated with low-Reynolds-number
  turbulence models. *International Journal of Heat and Mass Transfer*, 34(2),
  377–388. — §17

### Multiphase flow

- Hirt, C. W., & Nichols, B. D. (1981). Volume of fluid (VOF) method for the
  dynamics of free boundaries. *Journal of Computational Physics*, 39(1),
  201–225. — §20.1
- Zalesak, S. T. (1979). Fully multidimensional flux-corrected transport
  algorithms for fluids. *Journal of Computational Physics*, 31(3), 335–362.
  — §20.2
- Brackbill, J. U., Kothe, D. B., & Zemach, C. (1992). A continuum method for
  modeling surface tension. *Journal of Computational Physics*, 100(2), 335–354.
  — §20.4
- Ubbink, O. (1997). *Numerical Prediction of Two Fluid Systems with Sharp
  Interfaces.* PhD thesis, Imperial College London. — §20.1
- Rusche, H. (2002). *Computational Fluid Dynamics of Dispersed Two-Phase Flows
  at High Phase Fractions.* PhD thesis, Imperial College London. — §20.1

### Linear solvers

- Saad, Y. (2003). *Iterative Methods for Sparse Linear Systems*, 2nd ed. SIAM.
  — §8, §21
- van der Vorst, H. A. (1992). Bi-CGSTAB: a fast and smoothly converging variant
  of Bi-CG for the solution of nonsymmetric linear systems. *SIAM Journal on
  Scientific and Statistical Computing*, 13(2), 631–644. — §8.1
- Hestenes, M. R., & Stiefel, E. (1952). Methods of conjugate gradients for
  solving linear systems. *Journal of Research of the National Bureau of
  Standards*, 49(6), 409–436. — §8.2
- Swarztrauber, P. N. (1977). The methods of cyclic reduction, Fourier analysis
  and the FACR algorithm for the discrete solution of Poisson's equation on a
  rectangle. *SIAM Review*, 19(3), 490–501. — §8.5
- Stüben, K. (2001). A review of algebraic multigrid. *Journal of Computational
  and Applied Mathematics*, 128(1–2), 281–309. — §8.3

### Porous media

- Ward, J. C. (1964). Turbulent flow in porous media. *Journal of the Hydraulics
  Division, ASCE*, 90(5), 1–12. — §18

### Validation data

- Ghia, U., Ghia, K. N., & Shin, C. T. (1982). High-Re solutions for
  incompressible flow using the Navier–Stokes equations and a multigrid method.
  *Journal of Computational Physics*, 48(3), 387–411.
- Moser, R. D., Kim, J., & Mansour, N. N. (1999). Direct numerical simulation of
  turbulent channel flow up to Re_τ = 590. *Physics of Fluids*, 11(4), 943–945.
- Driver, D. M., & Seegmiller, H. L. (1985). Features of a reattaching turbulent
  shear layer in divergent channel flow. *AIAA Journal*, 23(2), 163–171.
- McCaffrey, B. J. (1979). *Purely Buoyant Diffusion Flames: Some Experimental
  Results.* NBSIR 79-1910, National Bureau of Standards.
- Martin, J. C., & Moyce, W. J. (1952). An experimental study of the collapse of
  liquid columns on a rigid horizontal plane. *Philosophical Transactions of the
  Royal Society A*, 244(882), 312–324.
- Nilsson, A., & Wallin, S. (2022). *Lid Driven Cavity Flow Using Finite
  Difference and Radial Basis Function Methods.* Uppsala University report 22015.

---

## Contact

**simul@msimul.com**

Meteo Simulation Co., Ltd. / 주식회사 메테오시뮬레이션

Teaching and academic research are free. Industrial R&D, research institutes
outside an educational institution, contract work and commercial use require a
licence — see sections 2 and 3 of [`LICENSE`](LICENSE). If the boundary is
unclear for your case, please ask.
