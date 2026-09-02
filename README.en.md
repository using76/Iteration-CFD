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
| Validation | 1,657 unit tests across all targets (1,509 in the lib), 813 `ofgpu-validate` checks |

---

## Licence

**Prosperity Public License 3.0.0** — source-available, not open source.
Noncommercial use is free; commercial use gets a thirty-day trial and then
needs a commercial licence.

| Use | Terms |
|---|---|
| Personal study, hobby and amateur work | **Free** |
| Educational institutions — teaching, coursework, lab work | **Free** |
| Universities and their institutes and laboratories | **Free** |
| Public research organisations and government institutions | **Free**, whatever the funding |
| Public safety, public health and environmental protection organisations | **Free** |
| Charitable organisations | **Free** |
| Any other commercial use | **Thirty-day trial**, then a commercial licence |

What decides the free tier is **the kind of organisation using the software,
not who pays for it**. A public research organisation or a government
institution is inside the free tier *regardless of the source of its funding* —
which is a change from the previous licence.

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

The licensor is 주식회사 메테오시뮬레이션 (Meteo Simulation Co., Ltd.). The full
text is in [`LICENSE`](LICENSE), what a commercial licence covers is in
[`LICENSING.md`](LICENSING.md), and third-party notices are in
[`NOTICE`](NOTICE).

> Prosperity is a noncommercial licence, not an open-source one. It is not
> OSI-approved and it restricts commercial use, so this project does not call
> itself open source.

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
| Spalart-Allmaras (SPEC-LIT §56) | **One transport equation**, for a working variable `nu~` that is not the eddy viscosity — the cheapest closure in this tree, one linear solve per outer iteration where k-ε needs two. Read from Allmaras, Johnson & Spalart's ICCFD7-1902 implementation paper, with `noft2`/`ft2` and Allmaras' **negative continuation** (`noft2-neg`/`ft2-neg`) selectable by `variant`. Its wall condition is exact and needs **no new `BcKind`**: `nu~ = 0` at a no-slip wall is a plain Dirichlet condition with no `y+`-dependent blending. Verified against the model's own published definition rather than a drag coefficient, because a flat-plate `C_d` can be right for the wrong reason: §56.4's log-layer identity is an *exact* property of the model and holds to round-off, `r = g = f_w = 1` there exactly, the closed-form bound on `c_n1` is derived here and checked, the SA-neg passivity identity is bitwise, and NASA's Turbulence Modeling Resource `nu_t/nu = 0.210438` and `1.294234` at the two ends of the recommended far-field range are reproduced to the printed digit. **The TMR flat-plate `C_d` gate is NOT run** and §56.11 says why — the case is compressible at `M = 0.2`, its grid family is curvilinear CGNS/Plot3D where `blockgen` builds axis-aligned blocks, and the tabulated values live in data files rather than on the page. Nothing here claims SA predicts separation, transition or a pressure-gradient boundary layer correctly. Reached by `ofgpu-sa` |
| LES | Smagorinsky, WALE, Deardorff |
| LES filter widths | Cube root of volume, maximum edge length, Scotti anisotropy correction, van Driest damping |
| LES wall model | Werner–Wengle (1991) — an analytically invertible power law integrated over the first cell's wall-parallel average speed, no Newton iteration; the `standard`/`spalding` presets both collapse to this one model under LES, `lowRe` gives `nu_t,w = 0`, and `rough` is refused by name (§13.4) since no rough LES wall model exists yet |
| Hybrid RANS-LES (SPEC-LIT §57) | **DES97, DDES and IDDES**, on the Spalart-Allmaras background and on k-ω SST. The shielding is **bitwise**, and derived here rather than argued: in an equilibrium log layer `r_d >= 1`, f64 `tanh` saturates to exactly `1.0` past `19.0615475`, which `(8 r_d)^3` clears for every `r_d > 0.33391`, so `f_d == 0.0` exactly and `dtil` returns `d` bit for bit — which is also why the SST background reproduces itself bitwise and the default is unmoved BY CONSTRUCTION. Grid-induced separation, the defect DDES exists to fix, is reproduced **live on two meshes differing only in streamwise cell count**: DES97 puts 2,048 of 5,632 attached cells into LES mode with a destruction amplification of 5.66, while DDES and IDDES leave `dtil == d` bitwise. IDDES's `h_wn` is the wall-normal cell height the crate did not have; `\|grad y\| = 1` holds to `3.2e-3` within five wall-adjacent cell heights and to only `0.495` over the whole block, but (57.19) normalises, so only the direction is load-bearing and that is exact to `2.0e-13` — the claim is recorded in the form the measurement supports. **Not claimed: that this solver reproduces a published separated-flow statistic.** There is no low-dissipation convection blending, no time-averaging seam and no synthetic-turbulence inlet, so the periodic hill is not run and IDDES's WMLES branch is exercised as closed forms and a length-scale field rather than as a simulation (§57.12). `ofgpu-validate` prints that distinction on every run |
| Wall functions | nutk, nutU (inverse Spalding law), nutLowRe, rough walls (Cebeci–Bradshaw), epsilon, omega, kqR, kLowRe |
| Turbulence selection in the coupled solvers (`ofgpu-buoyant`, `ofgpu-fire`) | `ofgpu-buoyant`: the `CoupledTurbulence` trait dispatches on the case's own `RAS { model ...; }`/`simulationType`, exactly as the standalone drivers do — k-ε, k-ω, k-ω SST (wall distance computed automatically) and LES (Smagorinsky/WALE/Deardorff, with §16's filter widths and van Driest damping) all construct the ACTUAL model asked for, and buoyancy production `G_b` is wired into the right equation for each (§17, §30.2). `ofgpu-fire`: still k-ε only — its combustion mixing-time closure and thermal wall function need `epsilon` directly, so any other model is refused by name, a §13.4 error, not a silent substitution |
| Wall-model presets (`wallTreatment`) | `standard`/`spalding`/`rough`/`lowRe` — one setting expands to a CONSISTENT row of per-field patch types (nut/k/epsilon/omega, and T when the energy equation is solved) at case-build time; a hand-mixed row across families is refused by name, `-permissive` substitutes the row implied by the `nut` choice (SPEC-LIT §29.1). `lowRe` additionally requires a turbulence model with near-wall validity — `LaunderSharmaKE` (SPEC-LIT §33) is the one model on that menu, `kEpsilon`/`kOmega`/`kOmegaSST`/`realizableKE`/`RNGkEpsilon` still are not, so `lowRe` under any of the latter three is refused by name rather than left to diverge (SPEC-LIT §32) |
| Thermal wall function | Jayatilleke's sublayer-resistance correction to the thermal log law (`thermalWallFunction`, alias `compressible::alphatJayatillekeWallFunction`) — every preset row applies it to `T` on walls except `lowRe`, which leaves the resolved sublayer's own molecular resistance alone (SPEC-LIT §29.3); wired into `ofgpu-fire`'s energy equation. Validated against Dittus-Boelter/Gnielinski on a fixed-heat-flux periodic PLANE CHANNEL (SPEC-LIT §32/§34): the **wall-function leg CLOSES** — Gnielinski at Petukhov's smooth-pipe `f` −5.9% (±10%), Dittus-Boelter −12.9% (±20–25%) — and the **resolved `lowRe` leg does NOT** (+11.9% Gnielinski, +4.0% Dittus-Boelter). At each leg's own MEASURED wall friction factor the Reynolds-analogy verdict closes on neither (+34.3%, +14.9%). Numbers are the record at the SHIPPED DEFAULT `PrtModel constant`, rerun after SPEC-LIT §26.1. Selecting SPEC-LIT §37's Kays-Crawford variable `Pr_t` (opt-in, one token, nothing tuned) moves the resolved leg from +11.9% to **+4.3%** and closes the absolute-prediction verdict on BOTH legs, while moving the wall-function control only −0.06% of `Nu`; the Reynolds-analogy verdict on the wall-function leg is untouched at +34.0%, being a friction finding rather than a thermal one. **SPEC-LIT §26.1 closed the energy imbalance this gate carried as an uncertainty on every one of those numbers**: §25.1's divergence constraint was implemented without its conduction term `div(k_eff grad T)`, so the resolved leg's steady bookkeeping was short by +3.11% (+3.35% under Kays-Crawford); it is now +0.000089%, and the resolved leg's Kays-Crawford pass no longer needs an error bar quoted beside it. Full account, including what it does NOT establish, in `docs/07-fire-solver.md` §1.1 |
| Conjugate heat transfer (SPEC-LIT §46/§47/§59/§60) | **Solid regions, contact resistance, the fluid/solid interface, and the fluid on the far side of it.** The interface is a Robin triple on BOTH sides with the series conductance `h_G = (1/C_A + R_c + 1/C_B)^-1` (§47.2): flux continuity AND the contact-resistance jump hold **exactly at every iterate**, not at convergence, because ONE kernel writes both sides from one `h_G` and one `\|Sf\|`. It needs **no new matrix code** — `lduAmul` and `lduAddBoundaryContributions` key on `bNbrCell >= 0`, not on "cyclic", so a conformal interface in a concatenated fluid+solid cell numbering is already solved implicitly; a conjugate interface IS a cyclic couple with a zero transform (§47.3/§47.4). Two CUDA lines and one new kernel pair were all it took. Solid side: `(rho_s c_s) dT/dt = div(K_s grad T) + q'''`, Patankar's **harmonic** multi-material face conductivity, and mesh-axis-diagonal anisotropic `K` (§46) — a full tensor on a skewed mesh is **refused by name** with the measured residual, MPFA and the two ways out, never approximated by its diagonal. Dirichlet-Neumann partitioning is **never implemented**: Meng et al. (2017) Theorem 1 gives its amplification as `K_R/K_L`, which fails exactly at air/plastic. Gates, all live: the two-layer slab `q = dT/(L1/k1 + Rc + L2/k2)` to `4e-14` after ONE assembly and ONE solve; flux continuity on the FIRST unconverged iterate to `1.8e-16`; conservation `2.0e-16`; `k_solid -> 0` contributes **bitwise nothing** (= `fixedFluxTemperature`, `q = 0`) and `k_solid -> infinity` reproduces the `fixedValue`/`thermalWallFunction` wall; the two half-spaces' interface sits at the effusivity-weighted mean and is constant in time to `1.1e-11` of `dT`. Reached from a case file by `ofgpu-cht` — `cases/dieStack.cht.jsonc`, a 100 W die through three contact resistances to a cold plate, junction temperature **649.7118 K** against its own closed form. **The fluid side (SPEC-LIT §59/§60).** `Energy` now runs over the concatenated fluid+solid thermal mesh, opt-in through `attach_conjugate`: the retarget is four elementwise blends plus §47.2's interface kernel, and the diff on `energy.rs` REMOVES exactly four lines, each the last statement of a function turned into a guarded call. The convective term vanishes in the solid **exactly**, because `Energy` masks `phi_conv` itself rather than trusting the driver — and it needs TWO masks, not one, because the conductance blend keeps a fluid interface face whose `C_A` IS `k_eff Δ` while the convective mask drops it. The default is proved **bitwise identical** on a full five-iteration run — fields, `T_b` and all six matrix arrays, not one coefficient — and `ofgpu-validate` re-runs that comparison live. Gates that pass, all live: de Vahl Davis (1983) at **+0.06 / +0.34 / +0.59 %** on 40²/60²/80², the two walls carrying the same heat to `1.1e-7`; the analytic conduction limit at `Kr = 0.1, 1, 10` to `2.5e-9`, `4.4e-8` and `7.1e-8`; interface conservation `1.5e-16`. **Its published gate — Kaminski & Prakash (1986) — runs live and MISSES**; see “Gates that miss” below. Reached by `ofgpu-cht` — `cases/kaminskiPrakash.cht.jsonc`. **Forced convection (SPEC-LIT §79).** §60.2's fluid region was a CLOSED CAVITY - every non-`empty` patch a no-slip wall - which is why §47.12's Gate 6 was UNREACHABLE rather than refused. A fluid patch may now say `"kind": "inlet"` (carrying `U` and a `fixedValue` `T`) or `"kind": "outlet"` (taking `inletOutlet`), exactly one of each or neither, and **neither is the old closed cavity in every bit** - the same branch, the same triples, no potential-flow solve, no value-fraction update. The four pieces §60.6 named are all there: §4's triple switched by the sign of the face flux, §31's `laplacian(Phi) = 0` flux-establishment pass (`max_c |sum_f phi_f|` = `3.1e-14 m^3/s` against a `1.7e-8` inlet flux), the outflow treatment that is `momFluxIsPrescribed` returning FALSE at the outlet so the pressure equation owns its flux, and a global mass balance that is reported rather than assumed (`2.5e-13` relative). `buoyancy` becomes OPTIONAL once there is an inlet, and its absence is a MODEL - constant density and exactly zero body force, which is the right one for liquid water and is what §25's ideal-gas `rho = p0/(R_s T)` is not. **`inletOutlet` is `zeroGradient` in every BIT while the flow leaves**, asserted in the cells and in the evaluated face values; and its `inletValue` is an entry the §13.4.1 contract cannot see, because whether it is read is decided by the FLOW - so the run reports how many outlet faces the flow re-entered through, and the pair test is written in both halves. A micro-channel unit cell is a box with a hole in it, so it is NINE regions and twelve couples; eight of the twelve are same-material and are NOT an approximation - at `R_c = 0` the series conductance IS the internal-face coefficient, proved and measured. **Gate 6 (Qu & Mudawar 2002) now RUNS and HOLDS**: `R_t,in = 0.0929` and `R_t,out = 0.2351 C cm^2/W` against Kawano et al.'s `0.116 [0.080, 0.152]` and `0.222 [0.156, 0.288]`, both INSIDE the measured error bars, over four meshes to `259 350` cells. Two disclosures go with it and are printed on every run: Kawano et al. (1998) was NOT obtained and the reference is Qu & Mudawar's Fig. 4 DIGITISED, and the paper states only one of the four water properties so the other three were chosen at the inlet temperature - which §79.12 quantifies as the largest single uncertainty in the gate. Reached by `ofgpu-cht` - `cases/quMudawar.cht.jsonc` |
| Wall distance | Poisson method (Tucker 1998) |
| Buoyancy production | G_b term (Rodi 1987, Henkes et al. 1991) |

### Multiphase and transport

| | |
|---|---|
| VOF | Interface compression, Zalesak FCT bounding, sub-cycling, CSF surface tension, p_rgh formulation |
| Scalar transport | Temperature, multi-component species (sum-to-one enforced) |
| Source terms | Volumetric heat release, momentum sources, Darcy–Forchheimer porous drag |
| Buoyancy | Non-Boussinesq density ratio `b = g(T_ref/T − 1)` |
| Generalised-Newtonian viscosity (SPEC-LIT §38) | `nu` becomes a **field**: power law, Cross, Bird-Carreau (`a = 2`) and Carreau-Yasuda (general `a`) from one formula, Herschel-Bulkley and Casson. The strain-rate invariant `gdot = sqrt(2 D:D)` is the `turbStrainRateMag` §6 already computed and nothing ever called; this is its first caller, and the whole chain stays gather-shaped and atomic-free. The two yield-stress models are Papanastasiou-regularised in the **product** form, the one that stays bounded for `n < 1` as well. Plane Poiseuille converges at order **2.08 / 2.03 / 2.00 / 1.97** against a closed form the spec derives; the Newtonian reduction is `4.5e-14` and every model's own reduction is constant in `gdot` to round-off; Buckingham-Reiner's `1, -4/3, +1/3` bracket is verified against the numerical integral of the profile it is the closed form of (`9.4e-11`) rather than quoted. Selected by `physics.fluid.rheology.model` (JSONC) or `viscosityModel` (`constant/physicalProperties`); an unrecognised name errors listing all six, and a coefficient the named model does not use errors naming it. **The drivers that hold `U` frozen refuse a non-Newtonian model by name** — `blockgen` had written `viscosityModel constant;` into every generated case since that generator existed and nothing ever read it, the sixth instance of that contract defect, and reading it without the refusal would have created the seventh while fixing the sixth |
| Contact angle (SPEC-LIT §39) | Static, hysteresis and dynamic on the VOF wall, replacing §20's `nHatf = 0` — which was already a contact angle, of ninety degrees, chosen because it adds no unstated physics. Jiang-Oh-Slattery and Cox-Voinov for the dynamic angle; **Kistler is deliberately absent**, its four constants coming from a book chapter this project has not read. `theta = 90` is bit-identical to no model and `theta = 45` is not, so neither test is vacuous, and the `cos(pi/2) != 0` trap is guarded twice with the test **measuring** the premise (`6.12e-17`) rather than asserting it. Gated on the geometry (`bNHatf = magSf cos theta` at 0/45/90/135/180 degrees, sign checked at both ends), on the `alpha` fixed-gradient triple, on both correlations returning `theta_e` EXACTLY at `Ca = 0`, and on Jurin's height as a closed form — `theta_e > 90` must give depression. **Not claimed**: that a live capillary-rise or drop-impact run reproduces a published `theta_d(t)`. Tanner's law, Sikalo et al.'s drop impact and the two-resolution displacement experiment are not run, and Afkhami, Zaleski & Bussmann's mesh-dependent correction is deliberately withheld until the gate that would show it works exists (§39.8) |
| Lagrangian parcels (SPEC-LIT §66) | SoA pool, exponential drag update (Schiller–Naumann), face-crossing walk over the cell→face CSR, deterministic injection. Parcels feel the gas; what the gas feels back is §68's, and what a droplet does to itself is §76's. **No driver reads a spray from a case file yet**, and §13.4.2 forbids adding a `parcels` block before the driver that would read it, so the pool is a library API and `ofgpu-validate`'s gates are what drive it |
| Parcel deposition gather (SPEC-LIT §67) | Radix sort on the `(cell, uid)` total order, a device exclusive scan, the per-cell parcel CSR, and a one-thread-per-cell gather. **No f64 atomics**: the scatter is transposed into a gather so the result is bitwise reproducible |
| Two-way coupling (SPEC-LIT §68) | The drag impulse the parcel integrator applied, handed back to the gas through §18's source registries — momentum (kinematic, explicit or Patankar-split) and sensible heat, with `physics heating` for the droplet side. **Conservative to round-off by construction**, and bitwise inert when no parcel has been injected. Droplet radiation and wall splash are refused by name; the mass coupling §76 changed the reason for is **built at §77** |
| Droplet heating and evaporation (SPEC-LIT §76) | `physics evaporating`: Spalding's Stefan-flow mass rate with Abramzon–Sirignano's `B_T` (Ranz–Marshall and FDS's `B_T = B_M` also selectable), Watson's `h_v(T)`, Marrero & Mason's `D(T)`, the 1/3-rule film state, and Godsave's heat-limited branch at the boiling point. Two saturation curves — a general Clausius–Clapeyron and §54's own Hyland–Wexler, which is refused for anything but water. The mass is derived from the energy the exponential integrator applied, so the droplet's own budget closes in f64; the accumulator is the difference of the step's two endpoint masses, so conservation is an identity. Gated on the `d²` law (`1.8e-6`), on the wet-bulb temperature (`5.7e-13 K` against this crate's own balance, **`0.76 K` against ASHRAE's**, and the gap is the Lewis number), and on the parcel's mass (`9.1e-16`). **The vapour stays on the parcel here** — §77 is what carries it into the gas |
| The vapour into the gas (SPEC-LIT §77) | `coupling/mass evaporation`: the mass, the enthalpy it carries and the volume it makes, through the seams the solvers already had — §61.2's whole-field explicit source on `Y_v`, §18's energy registry, and the one `Energy::target_divergence` seam §25.3 says the pressure loop changes in. **The mass identity is `6.3e-16`** and the energy ledger closes to `1.3e-12`, which is the droplet's own budget and not round-off. Two findings worth the row: there is **no second latent-heat sink** — (76.10)'s budget already puts every joule the phase change consumed inside the convective heat §68 deposits, so depositing `q_lat` again would count it twice, and what the gas is actually owed is the arriving mass's sensible enthalpy, 12 % of the latent heat; and **the ENERGY half of the divergence arrives through `Q` unaided while the VOLUME half does not**, which is §26.1's omission with the halves swapped and is why it is a named argument on its own method. A mist is a **net contraction**, by `h_v/(cp T) ~ 8`. Gated on ASHRAE's adiabatic saturation — 40 °C air at 12 % rh sprayed to saturation lands at **19.179 °C against ASHRAE's 19.296 °C**, and the wet bulb, which the relation says is an invariant of the process, drifts by the same 0.117 K. **A run with no dispersed phase is bitwise what it was, by construction**: `energyTargetDivergence` is untouched and the old call path is the old call path — and now measured one seam further on, with an empty pool's `+0.0` deposits pushed through a real `Y_v` solve and a real `(div u)_target` and compared with `to_bits()` |
| Droplet-wall impact (SPEC-LIT §78) | `wallInteraction stick` and `weber`: the four-regime impingement map of **Bai & Gosman (1995)** — adhere, rebound, spread, splash — selected by the impact Weber number, with **Mundo, Sommerfeld & Tropea (1995)**'s `K = Oh Re^1.25 > 57.7` as the default splash criterion and Bai & Gosman's own `We_c = A La^(-0.18)` as the alternative. Surface tension is a constant or **IAPWS R1-76**'s correlation at the droplet's own temperature, gated against the release's own table to **0.0045 mN/m**. `K` is never formed: the decision is taken in its fourth power, `We² Re > K_crit⁴`, which is the same test — `Oh Re = √We` exactly — with no `pow` and no `sqrt`, and is therefore bit-stable across compute capabilities where the criterion's own form is not (§38.6). **Nothing vanishes at a wall, and the statement is about bits**: the pool reclaims no slot, so a deposit is `remove` plus exactly two statements (`u ← 0`, one flag bit) and the mass on the wall is *bitwise* the mass that arrived; the ledger is a partition of the pool — `live` / `deposited` / `gone` — not an accumulator, and there is no atomic in it. Gated at **0 defects** on every regime boundary at `1 ± 10⁻⁸` of its closed-form inverse, on a 16-point sweep through all four regimes, and on the ledger; a run that meets no wall is **bitwise unmoved** by the whole map. **Splash is detected and NOT enacted** — the parent deposits whole, `n_splash` counts it, and the wall deposit is published as the upper bound that makes it. **Film transport is refused by name** (§78.11) with what each missing piece would need. **Gate 78-D is OPEN**: the two published splash criteria disagree by a factor of **4.78** in Weber number for the same 100 µm droplet, and neither is measured against an experiment here |

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
| Radiation | Gray P1 approximation (Modest ch. 15), Marshak wall condition, a `chi_r` radiant-fraction floor — **and gray fvDOM** (Modest ch. 16; Fiveland 1984; Truelove 1987 — SPEC-LIT §36): the same RTE along 24 level-symmetric S4 ordinates, `radiationModel` selects between them, and **gray is still what a case gets when it says nothing**. A third model, `viewFactor`, is refused by name here — `ofgpu-fire`'s medium is participating, so P1 or fvDOM is the physically right choice, and surface-to-surface radiation needs an enclosure definition this driver has no flag for (§50.12). Measured on `cases/burnerPlume.jsonc` (32,768 cells, 1,200 steps, RTX 5070 Ti, soot off, run back to back twice, §65.7): radiated fraction **14.97 %** (P1) against **13.79 %** (fvDOM), wall time **22.93 / 22.60 s** against **141.12 / 141.16 s**, peak device memory 172 MiB against 236 MiB |
| Soot (SPEC-LIT §61) | A transported mass fraction `rho Y_s` — `sootModel none` (the default), `prescribedYield` or `laminarSmokePoint`. The smoke-point model is Lautenberger, de Ris, Dembsey, Barnett & Baum (2005): a formation rate shaped on a cubic in mixture fraction between `Z_L` and `Z_H`, anchored on a measured laminar smoke-point height, with Magnussen-Hjertager oxidation and an availability clip. The cubic is checked back against the four conditions it was **solved from** (`2.4e-16` worst); propane's `Z_st = 0.0600725` to `1.3e-9`; the published peak rate `omega_sf,P = 0.45699 kg/(m³ s)` to `8.7e-8`; the prescribed-yield mass balance to `1e-14`; host closed forms against the device kernels to `1e-14`. Moss-Brookes and the sectional family are refused by name **with the reason**. §61's whole point is §62: `rho Y_s` reaches every WSGG band's `kappa` and therefore `T`. **Gate 61-A — the one number a published soot measurement can be held against — MISSES totally**; see “Gates that miss” below |
| Spectral radiation — WSGG (SPEC-LIT §62) | `spectralModel gray` (default) / `grayBanded` / `wsgg`, for **both** P1 and fvDOM, through the one `EnergySources` registration §36 already used. `wsgg` is Bordbar, Węcel & Hyppänen (2014): four gray gases plus a transparent window, `kappa_j` built per cell per band from the local `X_H2O`, `X_CO2` and soot, `a_j` from the local `T`. `grayBanded` (one band, `a_1 = 1`, the case's own `a`) is **bitwise identical to `gray` through the whole driver**, checked on every byte of every file two runs write — so “the default is unmoved” is a measurement a case can repeat, not an assertion. **§64 and §65 solve the banded slab EXACTLY, band by band**, which is the only gate that measures the banded *answer* rather than an identity or another model, and both overturned something they were built to confirm. fvDOM's angular error is closed form — `E_2^S4(tau) = (1/4pi) sum_m w_m exp(-tau/\|mu_m\|)`, verified at `E_2^S4(0) - 1 = 3.3e-16` — which splits the measured error into an angular half needing no run and a spatial residue that does; **P1 has no such split**. fvDOM solves the transparent window exactly (`1.7e-15`) where banded P1 must floor it and pays `-0.039 %` with hot walls and **`+10.1 %`** with a hot gas; on the band P1 is worst at, fvDOM is better by up to **7.8×**. **What it costs, measured (§65.7)**, same case, same two passes: P1 + WSGG **93.64 / 94.01 s** (**4.12×** gray P1) at 172 MiB; **fvDOM + WSGG 527.06 / 536.42 s — 23.36× — at 332 MiB**. The bands cost 4.12× on P1 and 3.77× on fvDOM, on opposite sides of their own arithmetic. `updateInterval: 4` recovers **6.7×** of the fvDOM factor for **0.12 points** of radiated fraction. The physics beside the price: the spectral model roughly **triples** the radiated fraction on this case (14.97 → 47.28 % on P1, 13.79 → 43.76 % on fvDOM) where switching the angular method moves it by 1.2 points gray and 3.5 banded — a factor of nine larger effect. **Its emissivity gate MISSES**; see “Gates that miss” below |
| Open radiative boundary (SPEC-LIT §63) | `openBoundary zeroGradient` (the default) or `coldSurroundings` with an `ambientT`. A zero-gradient `G` (P1) or `I_m` (fvDOM) on an open face says that whatever leaves comes straight back: **an open-sided fire domain with zero-gradient radiation boundaries is a perfectly reflecting enclosure**. It is also **singular** for WSGG's transparent band — under P1 that band is a pure-Neumann Laplace problem — and both models go non-finite on a WSGG open-domain fire without this condition, which is why §63 exists. An all-wall enclosure is **bitwise unmoved** by the setting, there being no open face for it to touch, and the refusal names both conditions and says what the default IS. Measured on the gray legs of `cases/burnerPlume.jsonc`: the radiated fraction goes 14.97 → 33.89 % (P1) and 13.79 → 25.07 % (fvDOM) once the radiation can leave |
| Validation gates | Sealed-box `dp0/dt` ramp (analytic), exact burner heat release, radiative equilibrium, cut-cell closure, msh hex closure, the two-step scheme's derived stoichiometry, the soot cubic against the four conditions it was solved from, the WSGG coefficient set's own invariants and §64/§65's exact banded slab — all permanent `ofgpu-validate` checks. Four fire-side gates **miss** — §42.8b Gate 2, §62.12 Gate 1-E, §61.8 Gate 61-A and §62.12 Gate 4 — and the summary **generates** the list it names them in from a registry each enters at the point it reports (SPEC-LIT §69), so they cannot go missing from it and a fifth could not be added without appearing; see “Gates that miss” below |
| Field output & restart | `-output foam,vtu,nvdb,vdb,usda` and `-writeInterval` write `U`, `p`, `T`, the turbulence closure and any species fields the same way `ofgpu-buoyant`/`ofgpu-vof` do; `-restartWrite N`/`-restartFrom FILE` checkpoint and resume — `p0`, `dp0dt` and the species mass fractions are carried across the restart, not only `U`/`p`/`T`, because a low-Mach run's thermodynamic state is more than those three fields. 40 steps continuous vs. 20+restart+20 agree on the first post-restart pressure residual, `p0` and total enthalpy |
| Volumetric sources | `sources[]` (JSONC) or `constant/fvSources` (OpenFOAM case directories) register a source on the momentum equation — a uniform body force over the whole domain, the one a periodic (cyclic-patch) case needs since it has no inlet to prescribe a mass flow from |

### Ventilation, enclosure radiation and data-centre airflow

`ofgpu-datacentre` wires SPEC-LIT §52–§55 together into a room solver —
`cases/coldAisle.dc.jsonc`. Surface-to-surface radiation (§49–§51) belongs with
them physically but is **not** part of that driver, or of any other: it is
specified, gated and pair-tested as a library, and the last row says so.

| | |
|---|---|
| Fan boundary (SPEC-LIT §52) | A manufacturer's pressure–flow curve as a **Robin triple**. The exact operator is rank-1 and dense; lumping it onto three per-face arrays imposes identically the flow rate the dense operator does, to `1.7e-14`. Quadratic, tabulated and constant-flow curves, with AMCA 210's density and speed corrections rather than the table treated as absolute. `dp = dp_max[1 - (Q/Q_max)^2]` is **even in `Q`**, so a reversed fan pushes harder the more it is pushed back — `Q` ran 3.0, −4.6, −33, −90, −1692 over five iterations before `Q\|Q\|` replaced it, which is identical on the forward branch. The Woodbury/capacitance-matrix path is refused by name. Gates: the closed-form operating point to `1e-10` relative, evaluated from the formula rather than quoted; NIST's public-domain FDS HVAC decks `fan_test`/`qfan_test` to `1.1e-6` and `2.7e-4` relative (their **input files and published CSVs only** — no FDS source read); `S = 0` bitwise against `fixedValue` and `S = 1e12` against a prescribed flow; and the whole `cases/coldAisle.dc.jsonc` network against its own hand-solved closed form, **within 2 % on both openings**, with whole-boundary continuity at `4.2e-10` |
| Porous jump (SPEC-LIT §53) | §18's Darcy–Forchheimer law (Ward 1964) integrated through a slab instead of over a cell — a resistive **face**, internal or on a boundary: three arrays divided by one number on the internal side, §52's triple on the boundary side. Perforated tiles and screens by open-area ratio, on the thin-plate `K(sigma)` published in the open literature and gated against its own limits and against the two values the design note quotes, **one of which it contradicts**. Reverse flow is modelled rather than clipped, and what the model gets wrong is printed in the report |
| Humidity and psychrometrics (SPEC-LIT §54) | Humidity ratio as one more transported species, Hyland–Wexler saturation pressure, and moist-air buoyancy through the **virtual temperature** — so `Y_v = 0` is bitwise the dry default. Gates: the thirteen `C1`–`C13` coefficients against the formula from an independent host transcription, which is the gate that matters most because everything else is downstream of `p_ws`; ASHRAE Handbook—Fundamentals (2021) Ch. 1 Table 2 at 0.5 % absolute, with the **0.44 % `W_s` residual attributed to the missing enhancement factor and PRINTED** rather than quietly tolerated; and IAPWS at the boiling point (`p_ws(100 °C) = 101 418.7 Pa`), whose reference is *not* ASHRAE, which is what makes it worth having. Herrmann, Kretzschmar & Gatley's real-gas formulation is **named and not implemented** |
| Data-centre metrics (SPEC-LIT §55) | RCI (Herrlin 2005), RTI (Herrlin 2008), SHI and RHI (Sharma, Bash & Patel 2002), measured against ASHRAE TC 9.9's Class A1–A4 recommended and allowable envelopes, reduced on the device. Every identity is checked against its formula rather than a stored number — `SHI + RHI = 1`, `RCI = 100 %` inside the band, `0 %` at the allowable limit, linearity in a uniform offset. **The six-configuration ranking gate is NOT run**: Wibron, Ljung & Lundström (2019) is CC-BY-4.0 with its licence verified live through the Crossref REST API, but its full text was not reachable from this environment, so what is gated is the one quantitative relation the abstract states — halving the supply flow doubles RTI, 40 % → 80 % — and the omission is printed on every `ofgpu-validate` run. Public data-centre CFD validation data is thin and mostly behind publisher walls; that is recorded rather than glossed |
| Surface-to-surface radiation (SPEC-LIT §49/§50/§51) | Deterministic view factors and enclosure radiosity for a **non-participating** medium. The whole model is one rewritten Robin triple, because a transparent medium contributes nothing volumetric. Monte Carlo is refused on **accuracy**, not on reproducibility — MCRT *can* be made bitwise reproducible with a counter-based RNG, and the answer to that counter-argument is NISTIR 6925 Table 2: `2.7e-4` at a million samples per pair, where deterministic integration reaches `4e-6` at 18,525 points. RT cores are refused too, as a reproducibility hole with a switch on it. The design note's double-area-integral method **misses the shared-edge gate by 40 %** and converges like `nq^-0.54`, so a single line integral was built instead — `6.6e-6`. Gates, all closed forms or identities, none replayed: Howell C-11 `0.1998248957` and C-14 `0.2000437761`, each **evaluated from the formula** so a transcription error fails rather than agrees; Shapiro/FACET's obstructed `F_12 = 0.11562061` with reciprocity checked at `A_3/A_1 = 0.25` exactly; NISTIR 6925's `BB104` construction at 120 surfaces — quadrature `6.6e-6` in 0.014 s, reciprocity exactly `0`, closure `<= 1e-12`, `min G >= 0`; grid against linear scan **bitwise**, and two builds of the same geometry bitwise. **Configured, not yet run from a case**: the dictionary, the refusals and §51's ten pair tests exist and the gates drive the library API, but **no driver binary reads an enclosure out of a case directory** (§50.12), and `radiationModel viewFactor` is refused by name in the JSONC fire block for the same reason. The coupled cavity gate (Balaji & Venkateshan; Akiyama & Chong) is not run for that reason **and** a paywall, and `ofgpu-validate` prints the omission on every run |

---

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

Since SPEC-LIT §70 the matrix–vector product, the implicit under-relaxation and
the value pinning walk a **merged row map ordered by the global face id** rather
than two lists ordered by the local one. That is groundwork for multi-GPU and
nothing else: cutting a mesh turns an internal face into a boundary face on both
sides, which moves its term between the two lists, and floating-point addition
is not associative — so `A·psi` would move in its bits under decomposition
before any communication existed. On an undecomposed mesh the merged list is
bit-for-bit the two old lists concatenated, so **nothing moved**: `ofgpu-validate`
prints identical output apart from two wall-clock figures, and every field file
the shipped cases write is byte-identical.

SPEC-LIT §71 then cuts the mesh for real. A Hilbert space-filling-curve
partitioner, a one-cell-deep ghost layer, and an exchange that is one gather
kernel plus one device-to-device copy — no unpack kernel, and **no
floating-point operation anywhere in it**, so a ghost cell holds bit for bit
what its owner holds. A cut face's metrics are *copied* from the whole mesh
rather than recomputed, which is what keeps the coefficients identical: two
parts deriving a shared face's `Sf` from their own point lists traverse it in
opposite windings and agree to round-off, not to the bit. `ofgpu-decompose
<case> -method all` cuts a shipped case into 2, 3 and 4 parts by each of three
partitioners, runs them in one process and compares every cell:

| Case | Cells | 3 partitioners × 3 part counts |
|---|---|---|
| `cases/channelPeriodicWF.jsonc` (cyclic) | 192 | **9/9 bitwise identical** |
| `cases/channel` | 24,000 | **9/9** |
| `cases/burnerPlume.jsonc` | 32,768 | **9/9** |
| `cases/plume` | 82,320 | **9/9** |

SPEC-LIT §72 removes the next obstacle: the **cross-part reduction**. Every dot
product, residual, volume mean and patch total in the crate is a sum over a set
that a partition splits differently every time, and floating-point addition is
not associative. The cheap answer — run each part's existing reduction and
*gather* the `P` partials into the one-block kernel this project already owns —
is implemented, and it is **not** enough: it is reproducible run to run for a
fixed cut and it moved in 20 of 108 relabelled decompositions on the solver's
own output, and 75 of 108 on an adversarial field. What is enough is an
accumulator in which addition *is* associative. Every term is split exactly
into four 30-bit integer limbs against one **global** anchor, the limbs are
summed as `i64`, and the result is converted back once — so the answer is a
function of the multiset of terms and of nothing else. Over the four cases
above, at 2, 3 and 4 parts, three partitioners, **and every relabelling of every
cut — 108 decompositions — the exact sum and dot product are bitwise identical,
every time.** It costs a measured **four to five times** the plain reduction, and
because of that number **nothing calls it yet**; §72.6 and §72.7 say so rather
than implying otherwise.

SPEC-LIT §73 puts the two together and runs a real solve: **PCG and PBiCGStab
over a decomposed mesh**, to a tolerance rather than for a fixed count, with one
halo exchange per matrix product and none anywhere else — every other step of
both recurrences is elementwise on a cell's own values. `ofgpu-decompose` gates
that too, and the criterion is stricter than §71's, because the convergence test
now reads a residual the exact accumulator did not move: the field must be
bit-identical **and the solve must stop on the same iteration**.

| Case | Cells | PCG / PBiCGStab, whole mesh | Decomposed |
|---|---|---|---|
| `cases/channelPeriodicWF.jsonc` (cyclic) | 192 | 29 / 20 iterations | **18/18 bitwise identical** |
| `cases/channel` | 24,000 | 124 / 79 | **18/18** |
| `cases/burnerPlume.jsonc` | 32,768 | 97 / 62 | **18/18** |
| `cases/plume` | 82,320 | 125 / 78 | **18/18** |
| `cases/gb_800000` | 800,000 | 286 / 174 | **6/6** |

**78 of 78**, at 2, 3 and 4 parts under three partitioners, every cell and every
iteration count. Then every one of those cuts again under **every rotation of
its part labels** — the same cells in the same groups owned by a different rank,
which is the permutation a rank-indexed reduction would fail while passing
everything else: **156 of 156**, bits and iteration count.

The preconditioner is where the cost is, and §73.5 measures it rather than
arguing it. A **diagonal** preconditioner is elementwise, and a part's diagonal
is bitwise the whole mesh's, so it is partition-invariant *for free*: its
iteration count is the same integer at every part count on every case — 1.00×,
not approximately. **DIC and DILU are not.** The factorisation is sequential
across colours and the sequence crosses cuts, so what runs is each part's own
submatrix with the couplings across every cut dropped — **block Jacobi, not
restricted additive Schwarz**, because the halo is read by the matrix product
and by nothing in the preconditioner, so there is no overlap to restrict. Its
cost at 16 parts, over the ten DIC/DILU rows of the five cases above: **0.91× to
1.84× the whole-mesh iteration count** — nine of the ten are a penalty and those
nine run 1.03× to 1.84×, and the tenth is the speed-up described below.
It is cheapest on the largest mesh, where the whole-mesh DIC is
worth 2.0× over Jacobi and the block-local one still buys 1.95×. Two results
contradict the obvious expectation and both are reported: the degradation is a
**step function of which direction the cut crosses**, not a smooth function of
the part count (`cases/channel` holds at 62 iterations through 4 parts and jumps
to 101 at 8), and it is **not monotone** — on the 800,000-cell mesh the
block-local DILU takes *fewer* iterations at 8 parts than on the whole mesh,
because BiCGStab's count is not a monotone functional of preconditioner quality.
This project's own test asserted that it was, and had to be corrected.

**Still one process and one GPU, and no strong-scaling number is published.**
There is no MPI and no NCCL: this machine has one device, and the installed CUDA
13.3 toolkit contains no NCCL at all — NVIDIA ships it for Linux only. Running
`P` parts as `P` processes on one card would measure context switching, so that
number is refused rather than invented. What §73.6 publishes instead is
measured: the per-cell cost of an iteration (0.84 ns at 800,000 cells), the
collective count per iteration (one exchange and two reductions for PCG, two and
four for PBiCGStab), and the cells-per-GPU at which communication overtakes
arithmetic for a *named* range of collective latencies. It also reports where
the communication volume **stops improving** — at 8 parts on two of the five
cases — and diagnoses that it is the **partitioner**, not the solver: the
Hilbert index normalises each axis to its own extent, so on a 2-D mesh it sorts
by the `z = 0` slice of a three-dimensional curve and fragments, and on a mesh
four cells deep it spends a top-level bit on the thin direction. On
`cases/channel` at 8 parts the plain linear cut is measured 14× better. That is
named in §73.7 and **not fixed here**.

The sixteen assembly kernels of §70.5 are still not partition-invariant, so a
decomposed run is handed a matrix assembled on the whole mesh; they are refused
by name in §71.7 along with the FFT pressure backend, decomposed I/O and
Lagrangian parcels, §72.7 adds the decomposed enclosure and the cut fan patch,
and §73.7 adds the pipelined Krylov variants (on the strength of Cools 2019's
accuracy result), the exact per-colour distributed factorisation, and
communication/computation overlap. **Nothing in the crate calls any of it yet**,
so no default can move. **METIS 5.2.x is Apache-2.0** (verified from its
`LICENSE`) and is deliberately not linked: §71.2 gives the three reasons, the
deciding one being that its output depends on its build, which would make the
partition a property of the linked library rather than of the mesh.

---

### A mesh with 2:1 refinement interfaces — and the norm that decides it

Adaptive mesh refinement is **not wired into any solver** — §75 below is what
exists and what it costs. SPEC-LIT §74 does the part that has to come first and can be settled on its own: build a mesh
that is *born* with 2:1 refinement interfaces, put the existing operators on it,
and measure whether they are still second order across one. If they are not, no
adaptation machinery would fix it.

On a face-based polyMesh a 2:1 interface is **not** the problem the AMR
literature says it is. A hanging node is a difficulty for node-based
discretisations; here a coarse cell's face onto a finer neighbour is simply
*four faces where there was one*, the coarse cell becomes a polyhedron with up
to 24 faces, and every operator already loops over "the faces of this cell"
without caring how many there are. **There is no flux register and no
refluxing** — the Berger–Colella apparatus exists because block-structured AMR
keeps a separate coarse-level flux that then disagrees with the sum of the fine
ones, and in an LDU/polyMesh formulation there is one flux per face picked up
with opposite signs by the two cells' gathers. Conservation is exact by
construction, and the four sub-areas sum to the parent area to the last bit.

What a 2:1 hex interface *does* carry is measured, not derived on paper:
**25.239401820678103°** of non-orthogonality, an owner weight of `1/3` instead
of `1/2`, `|k| = 0.4714`, and a relative skewness of **0.1421** — every one of
them asserted to `1e-12` by a test that builds the mesh, and re-measured by
`ofgpu-validate` on a different one.

**The gate, and the correction to it.** The brief was "an MMS convergence study
on a statically refined mesh must recover second order". Measured in a
volume-weighted L2 norm, **the unmodified code already passes** — plain
`corrected` reaches observed order 1.985 across a 2:1 interface. That is not a
success, it is a defective gate: a refinement interface is a two-dimensional set
inside a three-dimensional mesh, so L2 gives a local defect there only its own
share of the norm. Measured in L-infinity on the same four meshes (`N` = 8, 16,
24, 32 with the middle block one level finer, so the interface has faces, edges
**and** corners), the same run stalls dead — observed order **−0.070** on the
finest pair, the pointwise error at `N = 32` no better than at `N = 24`.

| snGrad / gradient | L2 orders | L∞ orders |
|---|---|---|
| `uncorrected` / Gauss | 1.183 1.035 0.994 | 0.515 0.713 0.774 |
| `corrected` / Gauss | 2.047 1.995 **1.985** | 2.198 1.126 **−0.070** |
| `skewCorrected` / Gauss | 2.046 2.001 1.993 | 1.712 1.861 **0.342** |
| `skewCorrected` / Gauss + skew-corrected gradient | 2.083 2.031 **2.017** | 1.955 1.941 **1.940** |
| `corrected` / `leastSquares` | 2.062 2.025 **2.014** | 1.914 1.928 **1.931** |
| `skewCorrected` / `leastSquares` | 2.045 2.013 2.005 | 1.873 1.894 1.904 |

**The load-bearing piece is the gradient, not the snGrad scheme.** Green–Gauss
places a face value where the face plane cuts the line `P–N`, not at the face
centroid; on the fine side of a 2:1 interface that error does not cancel and
does not shrink with `h`. On the gate mesh a Green–Gauss gradient of a **linear**
field is off by **30.17 %**, at any resolution, and that error is fed straight
into the non-orthogonal correction. Only the two treatments whose gradient is
linear-exact clear 1.9 in L-infinity.

Both routes to that are shipped and both are measured. `snGradSchemes
skewCorrected` (new — SPEC-LIT §2.5 and §74.4) adds the face-centroid skewness
term to the diffusive flux and to the matching flux read-back, in the same
multiplication order, so the two agree to the last bit; it is worth **27 % of
the L2 error and 51 % of the L-infinity error** at fixed order. The gradient
half is a deferred Picard iteration with a measured contraction of 0.221, whose
fixed point reproduces a linear field to `2.3e-15`. And the cheapest complete fix
for that half is a scheme **this crate already shipped**: `leastSquares`
differences cell centres and never forms a face value, so it is exact for a
linear field on any mesh, and `corrected`/`leastSquares` reaches L-infinity order
1.931 with no new code at all. §74.6 records that rather than burying it,
because it makes most of the new gradient machinery optional.

**Defaults do not move, by construction.** `skewCorrected` is not the default
and no shipped case names it, so the extra kernel is never launched. Beyond that,
the skewness vector is **exactly** `Vec3::ZERO` on a mesh whose faces are
unskewed, so even a case that does name it on a uniform mesh gets `corrected`
bit for bit — asserted with `assert_eq!` on the whole source vector, and with
`assert_ne!` on a refined one so the setting is not a no-op everywhere. That
second guarantee needed a **floor** that a measurement forced: the skewness
vector is a difference of two nearly-equal *computed* positions, so where the
true value is zero the computed one has no significant digits — a uniform box
came out at `6.9e-18` on some faces and at exactly zero on others depending on
its cell dimensions, and this project's own first version of the test passed on
one box and failed on another. Below `1e-9·|d|` the vector is zeroed; real
skewness at a 2:1 interface is `0.1421`, eight orders above that.

**The cuFFT direct Poisson path is excluded from a refined mesh by name, and the
exclusion is measured** rather than asserted: the detector the backend chooser
actually runs is run on one, and what it says is recorded —
`cell volumes are not uniform (cell 146 is 2.441406e-4, cell 0 is 1.953125e-3)`
— together with a check that the *unrefined* base grid of the same box still
passes, so the refusal is about the refinement and not about the generator. No
capacitance-matrix repair is offered: that trick needs O(1) modified rows and a
refined block is O(N^(2/3)) of them.

**What §74 does not claim.** Nothing in §74 adapts, so its own measurements say
nothing about restriction, prolongation or a post-adapt projection; §75 below is
where the mesh starts changing. The skewness correction is measured on the Poisson
equation only — not on convection, not on the pressure–velocity coupling, not on
a turbulence model. A skewed *cyclic* couple is not corrected, and that is
recorded as a limitation rather than detected. **p4est is GPL-2.0-or-later, not
BSD-2** as the brief assumed; its licence was checked before anything was
opened, and neither it nor libsc, t8code or OpenFOAM's refinement code was read.

---

### The adapt: refine, coarsen, and what a rebuild really costs

SPEC-LIT §75 makes the mesh change. A criterion marks cells, a plan turns marks
into a new leaf set under 2:1 balance, the addressing is rebuilt from one sort
and two binary searches, and the fields are moved across conservatively. **No
solver calls any of it and no case file can reach it** — what is delivered is
the adapt as an operation, with its gates, and the README says that here rather
than in a footnote.

The mesh state is a **linear octree**: a base grid with the leaves of an octree
over each base cell, stored as a leaf set and nothing else. There is no free
list. Every adapt re-sorts the whole set, so the cell numbering is a function of
the leaf set and not of the order the adapt visited things in — which is what
makes an adapt reproducible run to run. On any mesh the static generator of §74
can also express, the two emitters produce **identical bits**, asserted on the
points, the face lists, and every geometric array.

**The gates, and what measuring them found.**

| gate | result |
|---|---|
| `sum V`, `sum ρV`, `sum ρφV` across a refine | **exactly 0** drift |
| the same across a coarsen | **exactly 0** drift |
| no new extremum, either way | holds |
| refine → coarsen round trip | `6.5e-16` on `φ`, `2.4e-16` on `ρ` |

Exactly zero, not merely below `1e-14`: the integrals are summed exactly and the
transferred masses are the same floating-point numbers, redistributed.

**The round trip the brief asked for cannot fail for a scheme that conserves.**
Restriction is the exact left inverse of prolongation into a complete family, so
refine-then-coarsen returns the field to round-off *by construction* rather than
to the interpolation error. The direction that actually loses information is the
other one, and it is the one worth measuring:

| prolongation | coarsen-then-refine, observed order |
|---|---|
| piecewise-constant | **0.993, 0.998** |
| limited-linear (Barth–Jespersen) | **2.200, 2.115** |

**The conservative rescale the design note prescribed is singular, and is not
needed.** `λ = ρφV / Σ ρ φ̂ V` divides by zero for any field whose
volume-weighted mean over the parent is zero — a velocity component in a
recirculation, `p_rgh` itself. Recentring the reconstruction on the
volume-weighted centroid of the children makes the conserved sum telescope
exactly, is never singular, and preserves the reconstructed gradient. A test
refines a cell where `φ` is exactly zero, requires every new value finite and
the mass and energy conserved, then **measures the rescale's denominator on the
same data** and requires it to have vanished.

**The rebuild needs no prefix scan.** The design note specified "two binary
searches plus an exclusive scan". There is no scan: a `lower_bound` over a
sorted array already *is* the exclusive prefix sum of the per-cell counts, so
`cf_offset[c] = lower_bound(owner, c) + lower_bound(nbrKey, c)` exactly — and
that identity is asserted cell by cell against a direct count, on a generated
mesh, a 2:1 mesh and one that has actually been adapted. The rebuilt CSR equals
what the mesh builder makes, element for element, on host and on device. Nine
kernels, every one a gather, no `atomicAdd` of any width.

**And the finding that reorders the work.** A captured CUDA graph bakes every
kernel argument, so an adapt invalidates it; the design note called that the
blunt version of the AMR problem and proposed an invasive redesign to avoid the
recapture. Measured on this machine:

| cells | one step /ms | host mesh rebuild /ms | device transfer /ms | graph recapture /ms | adapt every N | N if only the recapture |
|---|---|---|---|---|---|---|
| 512 | 0.262 | 2.204 | 0.040 | 0.084 | 444 | **17** |
| 4 096 | 0.271 | 12.944 | 0.052 | 0.082 | 2 414 | **16** |
| 13 824 | 0.288 | 30.076 | 0.057 | 0.083 | 5 244 | **15** |

**Capture and instantiate cost 0.083 ms and do not grow with the mesh** — a
graph's cost is in its node count — so recapturing every 16 steps costs 2 % of
the run and every 50 steps costs 0.6 %. The recapture-avoidance redesign is not
needed for this reason, and is not implemented, and that is why. What an adapt
actually costs is the **host** mesh-and-geometry rebuild: 362× the recapture at
13 824 cells and growing linearly while the recapture stays flat. At 10⁶ cells
that is of order two seconds against a step of a millisecond or two, and no
cadence makes it affordable. The binding constraint on adaptive refinement here
is the **host rebuild**, not the graph.

### The rebuild on the device — and the half §75 named wrongly

SPEC-LIT §75.8 went one step further and named the culprit inside the rebuild:
`mesh/geometry.rs::compute`, "1396 host lines of polygon and pyramid
decomposition". SPEC-LIT §82 ports that sweep to the device, gates it on
**bitwise identity with the host sweep** — on five fixtures and on a mesh an
adapt actually produced — and then measures the rebuild again. **The naming was
wrong.** At 13 824 cells:

| piece of a 30.7 ms rebuild | /ms | share |
|---|---|---|
| the emitter — `Forest::build`'s face grouping | 16.5 | **54 %** |
| the geometry sweep — `mesh/geometry.rs::compute` | 9.4 | 31 % |
| the cell → face CSR and the plan's own bookkeeping | 4.8 | 15 % |

So a geometry sweep costing *nothing* could only have made the rebuild 1.44×
faster. The port delivers what it can: the sweep runs **5.2× faster** on the
device, but as a drop-in returning a `HostMesh` it is 2.2×, because downloading
sixteen arrays is half of what the drop-in costs. End to end the adapt cadence
improves by 17 % — N goes from 5 415 to **4 479** at 13 824 cells — and that is
not enough. **The new bottleneck is the emitter**, and §82.9 states what a
device version of it would have to reproduce, including the point numbering,
which is the hard half.

A number this section got wrong in its own draft and corrected: the unported
host prologue was quoted at 1.9 ms and argued about. It is a *residual* — one
timing minus two others — and across runs of the same code it moves between
0.12 and 0.78 ms. It is small, and the table now prints it with a note saying
it is a bound rather than a measurement.

One thing the port established that was not asked for: **`-fmad=false` is
load-bearing, and the first argument for it was measured in the wrong place.**
nvcc contracts `a*b + c` into a single rounding and rustc does not, so the build
system compiles `cuda/meshgeom.cu` twice — with the contraction off, which ships,
and with it on, which nothing links against and one test runs. The contracted
build misses the host's bits on **8 to 15 of the sixteen geometry arrays on
every fixture, uniform box included**. An earlier version of that check
*simulated* a contraction on the host, found nothing on any box mesh, and
concluded the flag was buying nothing; it had probed the wrong expression.

### The mesh that never comes home

§82 left one sentence of work behind it: *a `HostMesh` is the wrong
destination*. SPEC-LIT §83 builds the destination. `adapt::plan_resident`
rebuilds the mesh, computes its geometry on the device and hands back a
`GpuMesh` — no host geometry array is assembled and none is uploaded back.

Two things came out of it that were not in the brief.

**The download was not the only round trip, and not the larger one.** An adapt
that keeps its mesh also pays `GpuMesh::upload` — sixteen arrays in the other
direction — which §82 measured separately and left out of its table because
`plan` returned a `HostMesh` and there was nothing to compare it against.
Priced to the same place, *the new mesh on the device*:

| cells | host + upload /ms | device drop-in + upload /ms | **resident /ms** | N host | N dev | **N res** |
|---|---|---|---|---|---|---|
| 64 | 0.307 | 0.682 | 0.420 | 93 | 176 | 119 |
| 216 | 0.927 | 1.239 | 0.851 | 217 | 282 | 201 |
| 512 | 2.467 | 2.111 | 1.644 | 510 | 440 | 348 |
| 4 096 | 12.879 | 11.494 | 9.313 | 2 453 | 2 192 | 1 780 |
| 13 824 | 30.140 | 24.728 | **21.894** | 5 345 | 4 389 | **3 889** |

N falls from 5 345 to **3 889** at 13 824 cells — 27 %, against the 17 % §82
reported for the drop-in — and **it is still not enough**. `Forest::build`'s
emit loop is 15.4 ms of a 29.1 ms rebuild, 53 %, and no amount of geometry work
reaches it. §82.9 is still the specification for closing that, and nothing here
touched it.

**And "never round-trip the geometry" is one array short of achievable.** The
array is not `total_volume`; it is the conservative prolongation weight
`w_qp = V_q / Σ V` that makes the transfer conserve. `plan` builds it on the
host out of the new mesh's cell volumes, and that fold is in ascending cell id,
so a tree reduction on the device would be a different answer. `v` comes home;
the other fifteen do not. §83.9 is the specification for the kernel that would
remove even that, and it is not written.

§82 could not say where the device sweep overtakes the host, because 512 cells
was the smallest mesh it measured and the device was still losing there. With
64 and 216 added the answer turns out to be three answers: **the kernels cross
between 216 and 512 cells, the drop-in between 512 and 4 096, and the resident
route between 64 and 216** — three fixed costs, being four kernel launches,
sixteen synchronous device-to-host copies, and none.

The gate is §82's, one constructor further along: a mesh built on the device
must *reach* the device as the same mesh — sixteen arrays by their bits, plus
§70's row map, `total_volume`, and the transfer map's weights — on five
fixtures and on a mesh an adapt produced, at every size in the sweep.

One thing this section changed about how the project measures. §75.8's and
§82.5's tables time each quantity **once**, and on this machine that is not a
measurement: four runs put one step at 0.535, 0.262, 0.482 and 0.390 ms, and
the unmodified §82 binary measured the same 4 096-cell rebuild at **13.4 ms in
one run and 68.5 ms in another**, same code, same machine, an hour apart. The adapt section now reports the **minimum of three or four
calls**, which is the one estimator interference cannot bias upward, and the
evidence that it was needed is that the per-step time is now monotone in mesh
size — 0.227 / 0.241 / 0.254 / 0.265 / 0.283 ms — which it must be and never
was.

**What is not claimed.** The face flux is not transferred at all — neither the
area-weighted split of a parent face nor a divergence-free filling of a refined
parent's new interior faces — and that is the largest single gap between this
and a solver that can adapt. There is no post-adapt pressure projection. Only
scalar fields are transferred. **The emitter is still on the host, and is now
the larger half of a rebuild.** No solver was switched to the device sweep: the
bits are identical, so switching is safe, but a setup sweep runs once and the
change would be motion without a measurement. The multi-colour
preconditioner is not rebuilt, and an adapt raises the maximum cell degree from
6 to 24. `restart.rs`, the polyMesh writer and `bin/probe.rs` are untouched and
would all need work before an adapt could happen mid-run. The Jasak–Gosman
residual estimator and the Pope LES resolution index are **refused by name**,
with the Löhner indicator as the alternative in both cases.

### The emitter on the device: a point numbering that is one scan

§82 measured that the emitter, not the geometry sweep, was 54 % of a mesh
rebuild, and §82.9 wrote the specification for porting it — splitting it
honestly into an easy half and a hard one. SPEC-LIT §84 is that port.
`cuda/meshemit.cu` builds the voxel ownership map, groups the faces, numbers
the points, and writes the internal faces already in `(owner, neighbour)` order
and the boundary faces patch by patch, on the device.

**The hard half was the point numbering, and the trick is that the sequential
loop is a number.** The host numbers a grid point the first time its
cell-major, axis-major, minus-then-plus traversal touches it — which is a total
order on `(cell, axis, slot, corner)`, so it is an integer, the *touch rank*.
The host's point id is then the position of a site's smallest touch rank among
all sites' smallest touch ranks: a minimum per grid point, and one exclusive
scan. Both are pure functions of the leaf set, so there is nothing left for the
hardware to order.

The minimum is **gathered, not scattered**. An `atomicMin` over touches would
give the same answer — integer minimum does not care what order it arrives in —
but every rectangle this emitter emits is a whole face of some leaf, so every
point it touches has an incident voxel belonging to a leaf that touches it.
Read the eight voxels around a grid point, replay those leaves' own touches,
take the minimum. There is no atomic of any width in the file.

Two smaller observations did the rest. `(owner, neighbour)` keys are **unique** —
two boxes adjacent on one axis overlap on the other two — so the host's sort of
the internal face list is exactly "each cell's faces to higher-numbered
neighbours, ascending", which a cell can gather itself: no sort. And the
boundary flag array is patch-major, so **one** scan gives both the patch starts
and the within-patch order.

§82.9 offered permission to produce a *different but equivalent* point
numbering. It was declined: `io::polymesh` writes the point list to disk, and
§75.2's cross-check against the static generator compares it element for
element. The gate is bitwise on `points` and on every face's point list.

| cells | emit (host) /ms | **emit (device) /ms** | speed-up | rebuild, resident /ms | **rebuild, device-emitted /ms** | N res | **N emit** |
|---|---|---|---|---|---|---|---|
| 64 | 0.088 | 0.400 | 0.2× | 0.448 | 0.771 | 126 | 197 |
| 216 | 0.399 | 0.481 | 0.8× | 0.884 | 0.958 | 198 | 212 |
| 512 | 1.080 | 0.979 | 1.1× | 1.696 | 1.191 | 348 | 252 |
| 4 096 | 6.426 | 0.802 | 8.0× | 9.254 | 3.833 | 1 732 | 730 |
| 13 824 | **15.249** | **1.751** | **8.7×** | 22.563 | **7.081** | 4 039 | **1 287** |

The emitter is **8.7× faster** and the whole rebuild **3.2×**, priced to the
same place — the new mesh already on the device. N falls from 4 039 to
**1 287**, or from 5 388 against where §82 started.

**Is that enough? No, and by a factor of forty.** One step is 0.281 ms; an
adapt every 30 steps would cost 84 % of the run. But the bottleneck has moved
again and this time it is in four roughly equal pieces, none of them the
emitter: of the 7.08 ms, the device emitter and its download are 1.75,
**`build_cell_face_maps` on the host is 1.95**, the resident geometry route is
1.81, and `plan`'s own bookkeeping is 1.58 (a residual of four timings, not a
measurement — at 512 cells it comes out negative, which is what a residual
looks like when its inputs are noisier than the thing being differenced out).
The interesting one is the second: the cell → face CSR is now a larger piece of
a rebuild than the emitter that feeds it, and **a device version of it already
exists and is already gated** — `adaptCellFaceCsr` and `adaptBoundaryCsr`,
which `ofgpu-validate` checks element for element on every run. They are
unwired because the CSR used to be 6 % of a rebuild. It is 27 % now.

Four things this section found that were not in its brief. **A gate in the
repository caught a dependency this section had got backwards.** §75.9's
`no_time_loop_reaches_the_adapt` walks every source file and fails if anything
outside `adapt` names `crate::adapt`; the first draft had the mesh module
reading `adapt::VOXEL_LIMIT` and its tests importing `Forest`. The fix was not
to add two names to the allow-list — the test's own comment warns against that
— but to turn the arrow round: the voxel limit now travels with the call, and
the two gates that need a `Forest` live in `adapt`'s test module. **The fixtures
could not be §82's.** Three of that gate's five meshes — a graded block with a
cyclic axis, a box with every point displaced, a box with a fifth vertex on
every face — are not leaf sets at all, and the forest emitter cannot produce
them; the gate here uses the two that carry over plus six chosen to reach
branches the *device* emitter has and the host one does not. **The emitter's
cost below 512 cells is four host round trips, not arithmetic** — 0.40 / 0.48 /
0.98 ms on the three smallest meshes, which is the gap check, two grouping
diagnostics and the nine counts. And **`best_of` does not remove the machine**: the same code
measured the same 13 824-cell device emission at **0.727, 1.439 and 1.751 ms**
in three runs within forty minutes, every one of them the minimum of four
calls. That is a factor of 2.4 that survives `best_of` entirely. The table
above is the third of them — the run the committed binary produced, and the
slowest — so the honest reading of the 8.7× is "between eight and twenty".

### The flame that could not fire: a gate that missed for the wrong reason

SPEC-LIT §61.8's **Gate 61-A** holds the `laminarSmokePoint` model's predicted
post-flame soot yield against Tewarson's measured `0.024 kg/kg` for propane. It
missed **totally** — `0.000` against `0.024` — because **0 of 32 768 cells** on
the demonstration burner reached the model's own `1375 K` formation threshold.
§61.7 had predicted that collapse before §61 was written, and §61.8's diagnosis
was: *the model works and the mesh is too cold for it.*

**The mesh was not the reason.** SPEC-LIT §85 measures what was, and it is not
in the soot model at all. `Species` is handed the case's `TurbulenceControls`
whole, and `ScalarTransport` under-relaxes with that struct's `k_relax` — **the
turbulence kinetic energy's factor**. So a case naming `relaxation { k: 0.5 }`
for its k-epsilon model was under-relaxing `Y_F`, `Y_O2`, `Y_P` and `Y_s` by
the same 0.5. In a non-iterative transient splitting there is no outer
iteration for a relaxed equation to converge in, so that factor multiplies the
rate at which fuel can enter the domain's books at all.

The cleanest measurement removes every other term at once: the reaction
switched off, the domain isothermal at 293.15 K to the printed digit, nothing
crossing an outlet, so the only thing that can happen to the fuel is that it
accumulates. Against the **2.709 g** the burner admits in 6 s:

| `relaxation.k` | resident fuel after 6 s | fraction of what entered |
|---|---|---|
| 0.25 | 0.8186 g | **30.2 %** |
| 0.5 (what the case shipped) | 1.5687 g | **57.9 %** |
| 1.0 | 3.0562 g | 112.8 % |

Proportional to the factor. **42 % of the propane that crossed the burner never
appeared anywhere**, which is also what the fire's own fuel budget had been
saying without naming it: 20.97 kW in, 7.45 kW burnt, 6.7e-7 kW out, 0.029 g
resident, and 13.5 kW simply gone.

The species now read `relaxation."Y"` **by name**, falling back to `k`'s factor
so every published number is bitwise unmoved by construction, and the run
prints which of the two it got. A per-species key (`Y_F`, `Y_O2`, …) is refused
by name, because there is one factor for the set.

**With that out of the way, refinement does what §61.8 said it would.** The
sweep that had reported zero at every mesh was measured while 42 % of the fuel
was being deleted; re-run on a resolved 2.95 kW burner:

| cells | peak T | cells > 1375 K | predicted yield |
|---|---|---|---|
| 32³ | 1350.98 K | **0** | `0` |
| 48³ | 1379.01 K | **24** | ≈ `0` |
| 64³ | **1394.60 K** | **296** | **`0.0124 kg/kg`** |

`0.0124` against the measured `0.024` is a factor of **1.94** — inside the
factor of two Gate 61-A asks for.

**It is still not reported as a pass, and that is the point of the section.**
The leg that reaches the model's window exports **4.56 kW** of unburnt fuel
against a **2.95 kW** supply, so the yield is read off cells partly made of
fuel that entered nowhere. The species equation is
`ddt(psi) + div(phi, psi) - laplacian = 0` with a *volumetric* flux — the
constant-density transport equation — where a fire needs `d(rho Y)/dt +
div(rho u Y)`. The two available convection schemes bracket the truth from
opposite sides and neither closes it: **−17 %** conservative, **+162 %** under
the bounded default. A number inside the target's factor of two, produced by a
run that manufactures its own reactant, is not a pass, and the next unit of
work here is a species equation in `rho Y` rather than another case file.

Radiation is **exonerated** by measurement along the way, all three legs on
the same species convection so that only the radiation moves: peak T is
816.92 K as the case ships, 1015.50 K with the radiant-fraction floor at zero,
and 996.35 K with radiation removed entirely. The most that touching radiation
can buy is **+198.6 K** on a flame still **359 K** short of the threshold. And the gate case's own previous header, which claimed 95.86 % and
1012.0 K at the command it documented, **does not reproduce**: the same command
gives 46.67 % and 743.1 K, and §85.11 says so rather than quietly substituting
a different pair.

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
cargo test        1586 passed, 0 failed, 4 ignored (lib)
                  1734 passed, 0 failed, 6 ignored (every target — including the
                  per-binary CLI-parsing suites and SPEC-LIT §13.4.1's per-driver
                  "two runs must differ" pair tests)
ofgpu-validate    861 / 861 checks passed (813 computed live, 48 replayed from recorded
                  measurements), then a GENERATED list of the 6 gates that miss and the
                  4 verdicts that are open (SPEC-LIT §69)
```

**SPEC-LIT §13.4.1's standing requirement**: two short runs of a driver,
differing in exactly one setting of the case file and nothing else, must write
DIFFERENT output. If they are bit-identical the setting is inert. **Seven
drivers now carry such a pair test in their own binary** — `ofgpu-fire` 41
settings, `ofgpu-buoyant` 21, `ofgpu-vof` 18, `ofgpu-k-epsilon` 18,
`ofgpu-plume` and `ofgpu-k-omega` 11 each, `ofgpu-sa` 8 — each running the
driver's own `parse` + `run` and comparing every field file written. Every
capability added since carries the same test on its own **case document**, in
the module that reads it: §51.2 for the enclosure dictionary, §55.6 for the
data-centre case (23 settings), §58.4 for Spalart-Allmaras and the hybrids,
§60.4 for a conjugate fluid/solid case, and §61.5 / §62.11 for soot and the
spectral model. Two pairs in the whole suite are **inverted** and required to
come back identical rather than different — `spectralModel` absent against
`"gray"`, and `"gray"` against `"grayBanded"` — and they live outside the
differ-list, because that list asserts the opposite of what they claim.

### Gates that miss

Every check `ofgpu-validate` runs passes; that is what 813 / 813 means. It is
a different statement from "every published benchmark this project compares
itself against is reproduced", and the two are not allowed to be confused
here. The gates below are **comparisons against published measurements that
this solver does not reproduce**. `ofgpu-validate`'s summary **generates** the
list of them from a registry each gate enters at the point it reports
(SPEC-LIT §69): printing a verdict and registering one are the same call, so
all six are named on every run and a seventh could not be added without
appearing there. Two of the run's own rows assert exactly that — that nothing
printed a verdict outside the registry, and that every registered gate is
named in the list. They are repeated here so that reading the binary's output
is not the only way to find them out.

That mechanism replaced a hand-written sentence, and the sentence was wrong
three ways at once: it named four of the six, left the fifth (§42.8b) as an
aside inside a parenthetical about replayed measurements, omitted the sixth
(§68.12's Gate 68-C) altogether, and asserted that §62.12 Gate 4's verdict was
"noted in the soot/WSGG block above" when **no line the binary printed
mentioned that gate at all**. Two earlier passes fixed the sentence by editing
it, which is what produced the third defect; this one deleted the sentence.
**The table below is still maintained by hand against what the binary prints,
and nothing yet compares the two files** — SPEC-LIT §69.9 names that as the
next step rather than leaving it implied.

| Gate | Verdict |
|---|---|
| **SPEC-LIT §60.5 Gate 5 — Kaminski & Prakash (1986)**, conjugate natural convection in a square enclosure. **Run live.** | **MISSES its 3 % bar at the conduction-dominated end.** The live 40² run at `Ra = 10⁴` reads `-7.11 %`, `-2.77 %`, `-0.07 %` at `Kr = 0.1, 1, 10` — worst at the SMALLEST conductivity ratio, shrinking to nothing at the largest. §60.5's mesh-converged sweep (eighteen runs on 40²/60²/80², every 60→80 change under 0.38 %) puts it at `-7.12 %`, `-3.00 %`, `-0.48 %` at `Ra = 10⁴` and `-7.79 %`, `-4.32 %`, `-0.81 %` at `Ra = 10⁵`. **The primary table was never read** — the paper is paywalled, and ScienceDirect, Scholar, Semantic Scholar, OpenAlex, Unpaywall, CORE, arXiv and two institutional repositories were all tried. The comparison is against Belazizia et al. (2012), open access, same configuration, **labelled a SECONDARY source** in the spec, in the case file and in the output. The disagreement tracks how much of the answer is conduction — 2 % of the series resistance at `Kr = 10`, 71 % at `Kr = 0.1` — and the secondary table's own `Ra = 500` column sits 3–7 % **above** the analytic conduction limit at a Rayleigh number whose fluid-layer value is `O(100)`, which is not physically possible. Gate 59-B reproduces that limit to `1e-8`. So the reference numbers appear to carry an offset where the miss is. **Nothing was tuned toward them** |
| **SPEC-LIT §62.12 Gate 1-E — the WSGG total emissivity against RADCAL** (Grosshandler, NIST TN 1402, US public domain, compiled **unmodified** from `reference/fds/Source/rcal.f90` behind `tools/radcal_emissivity/`). **Run live.** | **MISSES its ±10 % bar at 58 of 108 points**, mean `\|d eps/eps\|` **11.4353 %**, worst **30.5234 %** at `M_r = 2`, `T = 400 K`, `p_a L = 0.03 atm.m`. Bordbar's own table could not be obtained, which is why the reference is RADCAL. **The shape is the finding**: the bias is a monotone ladder with exactly one sign change, `+20.84 %` at 400 K, `+14.11 %`, `+6.64 %`, `-1.31 %`, `-7.46 %`, `-12.28 %` at 2400 K, crossing zero near Bordbar's own `T_ref = 1200 K`. This is **not** evidence that Bordbar's set is wrong — RADCAL is a narrow-band model on NASA SP-3080 band data, Bordbar's is a fit to line-by-line HITEMP-2010, and at 2400 K both extrapolate. **Neither model is truth and the verdict line says so.** What it *is* evidence of is that the disagreement is structured rather than scattered, so a fire's smoke layer and its flame are the two places the choice of coefficient set moves the answer most |
| **SPEC-LIT §61.8 Gate 61-A — the predicted post-flame soot yield** against Tewarson's measured one. A 1,200-step fire, so **not run inside `ofgpu-validate`**; its verdict is registered and printed there anyway (SPEC-LIT §69). | **MISSES totally: 0.000 kg/kg against a measured 0.024 for propane.** Not a small miss, and diagnosed: **0 of 32,768 cells** on `cases/burnerPlume.jsonc` sit above the model's own 1375 K formation threshold, so the burner mesh is too cold for the model to fire at all and the `laminarSmokePoint` run is bit-identical to having no soot. §61.7 predicted exactly this before any code was written. **The model is wired and the mesh is cold, which are different sentences**: the five smoke-point pair rows, run on a duct that *is* hot enough, pass. The `prescribedYield` leg returns `0.024` against `0.024` and is **labelled an IDENTITY on the line that prints it**, because it is handed the answer |
| **SPEC-LIT §62.12 Gate 4 — the NIST 37 cm propane burner's radiative fraction** (Sung et al., NIST TN 2162r1, 2021: 0.23 / 0.30 / 0.33 at 20 / 34 / 50 kW). A multi-minute fire per heat release rate, so **not run inside `ofgpu-validate`**; its verdict is registered and printed there anyway (SPEC-LIT §69). | **MISSES.** `cases/nistBurner37cm.jsonc` never reaches a state in which a radiative fraction is a meaningful quantity: its combustion efficiency comes out at **226 %**, meaning the domain is consuming an accumulated fuel inventory rather than burning what enters, and §62.12's own gate text named `~95 %` efficiency as the precondition before any of this ran. **Gate 6 of the same family used to be unrunnable for a capability reason — Qu & Mudawar's forced-convection micro-channel had no inlet to name — and SPEC-LIT §79 lifted that; it now runs live and HOLDS (§79.12)** |

Two more comparisons miss. Both are in the generated list on every run —
§42.8b as its entry `[1]` and §68.12's Gate 68-C as its `[6]` — and both are
recorded here in the same voice rather than omitted.

**SPEC-LIT §42.8b**, the NIST Reduced Scale Enclosure compartment sweep, misses
and is replayed among the 48. Above 200 kW the predicted ceiling CO is low by a
factor of up to 20, where ISFEH10's own published statistic for this model on
this experiment is a bias factor of 1.08. The diagnosis comes from the runs
rather than from the model, and it is the **ventilation** and not the
chemistry: the combustion efficiency is 15–58 %, so most of the fuel leaves
the compartment unburnt, and the doorway admits roughly a tenth of the air a
400 kW fire in that room draws. Steckler, Quintiere & Rinkinen (1982), the
doorway-flow gate, is the prerequisite this miss names, and it is still not
run.

**SPEC-LIT §68.12 Gate 68-C**, Theobald's ~90 hose streams, misses **with the
gas held at rest**. Its verdict is printed in the run's parcel block and, since
SPEC-LIT §69, in the summary's generated list beside the other five; before §69
it was in the parcel block only, and the summary line did not know it existed.
The stream is thrown **61.29 %** of the measured
distance on average, against a `±10 %` bias and `30 %` scatter bar, and the
number beside it says why rather than leaving it to be inferred: the vacuum
bracket — the same launch with no drag at all — is **198.65 %** of the
measurement, so between 61 % and 199 % of the throw is decided by what the
**air** does, and with the air held still there is nothing left to decide it
with. Re-running the same 90 launches in a uniform co-flow shows that about
**6 m/s** of entrained air brings the mean to within 2.3 % of the measurement
and *tightens* the scatter from 0.359 to 0.271 — which says the still-air
scatter is one missing mechanism seen ninety times, not ninety independent
modelling errors. Having that number is worth more than a pass.

**Four further verdicts are `OPEN` rather than missing**, and the generated
list carries them as a second, separately counted group. Three are SPEC-LIT
§32.4's plane-channel comparisons against **Gnielinski (1976)**, which is a
correlation and not a measurement. The wall-function leg is `+34.4 %` and
`+15.2 %` of it at its own measured friction factor; the resolved leg is
`+11.9 %` at the Petukhov pipe `f` and `+14.9 %` at its own — all outside the
±10 % band. They are not counted among the six because a band not met against
a fitted correlation, with a one-token opt-in (§37's Kays-Crawford `Pr_t`)
that moves the resolved leg to `+4.3 %` and inside, is a different finding
from a published measurement this solver does not reproduce. They were absent
from the summary line before §69 for the same reason Gate 68-C was.

The fourth is **SPEC-LIT §78.10's Gate 78-D**, and it is open for a different
reason: the two *published* splash criteria disagree with **each other**. For a
100 µm water droplet at 20 °C, Mundo, Sommerfeld & Tropea (1995) put the
splash threshold at `8.99 m/s` (`We 111`) and Bai & Gosman (1995) at
`19.65 m/s` (`We 531`) — a factor of **4.78** in Weber number — and Bai &
Gosman's own roughness range moves theirs by a further factor of about two.
SAE 950283 is paywalled and was not obtained, so Bai & Gosman's constants
here are the ones their citing literature quotes — which is a direct reason
every threshold in the map is a control with a default rather than a constant
in the source. And neither criterion is measured against an experiment here:
Mundo's own deposition/splashing boundary is a figure rather than a table and
nothing in this repository transcribes it, so holding the kernel against the criterion (which
gate 78-A does, to `10⁻⁸`) is a transcription check and calling it a
validation would be exactly the confusion §69 exists to prevent. The splash
boundary of this map is known to within a factor of five and the run says so.

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

**That ratio is a property of the case, not of the feature, and it falls as the
mesh grows.** A graph buys back CPU submission time, which is a cost per
*launch* and is independent of how much work each launch does — so the
ratio is governed by launches per unit of GPU work, not by cells. Re-measured
on an RTX 5070 Ti, CUDA 13.3, 3 sweeps, 300 outer iterations, every row still
bitwise identical:

| cells | per-launch ms/iter | graph ms/iter | ratio |
|---|---|---|---|
| 1,500 | 1.229 | 0.332 | **3.70×** |
| 24,000 | 1.258 | 0.412 | **3.06×** |
| 240,000 | 2.851 | 2.265 | **1.26×** |
| 800,000 | 9.101 | 8.794 | **1.03×** |

Holding the mesh at 24,000 cells and raising the launch count instead (1, 3,
10, 30 solver sweeps) moves the ratio the other way: 2.72×, 2.96×,
3.27×, 3.41×. Above roughly half a million cells this solver is
memory-bandwidth bound and the graph is worth a few per cent. It is worth most
where an iteration is many small kernels — which is what the fire, VOF and
multi-model paths are.

### Every module is gated for capture, or refused by name

The bitwise claim above is now checked **per module** rather than asserted once.
`SPEC-LIT` §81 derives the population from the tree — every module
that launches a kernel or declares a per-iteration entry — so a module
added later is gated, excused by name, or it fails the tests. Fifty modules:

| | |
|---|---|
| gated: capture, replay 3×, compare **bitwise** | **36** |
| refused, with the reason and the alternative | **10** |
| outside the iteration | 1 |
| **ungated** — `pressure/mod.rs`, `pressure/fft.rs`, `simple.rs` | **3** |

Writing the gates found six modules that could not be captured at all. Three
were the same defect — a *diagnostic counter* reduced on the device and
read back to the host once per iteration — in WSGG, combustion and
surface-to-surface radiation; each is now behind a flag that defaults to on, so
every existing run reports exactly what it reported before, and each has a
companion query so a `0` meaning "not counted" is distinguishable from a `0`
meaning "nothing happened".

The other three are refused, and the refusals are **executed** rather than
written down:

* **VOF** — `Vof::step` downloads the alpha Courant number and derives the
  MULES sub-cycle count from it, so the trip count is data-dependent and lives
  on the host. A graph would record whatever count the capture saw and replay
  it for ever. The alternative is a prescribed `nAlphaSubCycles`; not
  implemented;
* **fvDOM** — the ordinate sweep carries each ordinate's boundary intensity
  to the next *through the host*. P-1 is gated and is the alternative;
* **`PCG` and `DIC`** — `solve` verifies symmetry before running conjugate
  gradients or an incomplete Cholesky, and that check ends in a read-back.
  Choosing `PCG` in `fvSolution` costs that equation its CUDA graph, however
  the solve is otherwise configured — the fixed-iteration mode removes the
  *other* read-back, not this one. This was documented in the code and had
  never been tested.

AMR is refused for a different reason: it reallocates every cell-sized buffer,
and a captured graph holds the old pointers.

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
| `ofgpu-validate` | Numerical validation (813 checks) |
| `ofgpu-bench` | Throughput and memory benchmarks |
| `ofgpu-graph-bench` | CUDA graph against per-launch execution |
| `ofgpu-dispatch-bench` | Runtime dispatch cost |
| `ofgpu-probe` | Device properties |
| `ofgpu-decompose` | Cut a case into parts, run them in one process, and report whether any bit moved (SPEC-LIT §71); reduce over every relabelling of the cut and report whether the reduction moved (§72); then solve it with distributed PCG and PBiCGStab — under each partitioner and under every relabelling of each cut — and report whether the field or the iteration count moved, with the block-local DIC/DILU iteration ladder and the per-iteration cost (§73) |
| `ofgpu-generate-mesh` | Case generation |
| `ofgpu-k-epsilon`, `ofgpu-k-omega` | Turbulence models, standalone |
| `ofgpu-sa` | Spalart-Allmaras and the DES97/DDES/IDDES family, standalone (SPEC-LIT §56–§58) |
| `ofgpu-plume`, `ofgpu-buoyant` | Buoyant plume |
| `ofgpu-vof` | Two-phase VOF |
| `ofgpu-fire` | Low-Mach combustion, soot and radiation (SPEC-LIT §25–28, §42/§43, §61–63) |
| `ofgpu-cht` | Conjugate heat transfer — solid regions, contact resistance, and a fluid on the far side of the interface (SPEC-LIT §46/§47/§59/§60) |
| `ofgpu-datacentre` | Room airflow with fan curves, porous jumps, psychrometrics and the rack metrics (SPEC-LIT §52–55) |

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
cargo run --release --bin ofgpu-cht           -- ..\cases\dieStack.cht.jsonc
cargo run --release --bin ofgpu-datacentre    -- ..\cases\coldAisle.dc.jsonc
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
| `docs/05-io-redesign.md` | The case input/output redesign — the JSONC case format and its schema |
| `docs/06-mesh-oss-2024.md` | Survey of open-source meshing |
| `docs/07-fire-solver.md` | `ofgpu-fire`'s formulation and its validation gates — the full record of the wall heat-transfer gate, the soot and WSGG gates, and the measured cost of spectral radiation |
| `docs/schema/case-1.json` | The JSONC case schema, generated by `schemars` from the reader's own types |
| `cases/README.md` | Test case geometries |

---

## Limitations

- **No MPI or multi-GPU support.** Single GPU only. The decomposition, the
  halo, the partition-invariant reduction and a distributed PCG/PBiCGStab all
  exist and are gated (SPEC-LIT §71–§73), but they run `P` parts in one
  process on one card; no communication library is linked and no
  strong-scaling number is published.
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
- ~~**Combustion is mixing-controlled single-step (EDM) only.**~~ **Restated
  (SPEC-LIT §42/§43).** The serial two-step mixing-controlled scheme and the FDS
  critical-flame-temperature extinction predicate are implemented, so CO,
  incomplete combustion and where the flame goes out are predicted rather than
  absent. What is still missing is a **finite-rate (Arrhenius) mechanism** —
  no Westbrook–Dryer, no Jones–Lindstedt, no stiff ODE integrator, no
  Jacobian. §42 also keeps the molar mass `W̄` and the specific heat `c_p`
  CONSTANT, so density and thermal expansion are still computed with air's
  values even once CO2 and CO are distinguished, and `dh1`'s exact split needs
  a heat-of-formation table that was not read. The compartment gate this
  scheme was built for (§42.8b, the NIST Reduced Scale Enclosure sweep)
  **misses**, and the diagnosis is the ventilation rather than the chemistry.
- ~~**Radiation is gray P1 only.**~~ **Fixed (SPEC-LIT §36, §61–§65).** `fvDOM`
  (24-ordinate level-symmetric S4) is selected by `radiationModel`; **WSGG**
  (Bordbar, Węcel & Hyppänen 2014 — four gray gases plus a transparent window)
  is selected by `spectralModel`, for both models, with `kappa_j` built per
  cell per band from the local `X_H2O`, `X_CO2` and **soot** (§61); and §63's
  `coldSurroundings` gives an open domain somewhere for the radiation to go.
  Three things a reader should know before switching any of it on. **(a) It is
  expensive, and the number is measured** (§65.7, `cases/burnerPlume.jsonc`,
  32,768 cells, 1,200 steps, RTX 5070 Ti, two passes): WSGG costs **4.12×** on
  P1 and **3.77×** on fvDOM, and **fvDOM + WSGG is 23.36× gray P1** —
  527.06 / 536.42 s against 22.93 / 22.60 s — at 332 MiB against 172 MiB.
  Measured storage is **+4,608 B/cell** for fvDOM + WSGG, so 10⁶ cells needs
  **9.3 GB, not the 960 MB** §62.10 predicted as arithmetic; the practical
  ceiling on a 16 GB card is about 1.6 × 10⁶ cells, and the clock gets there
  first. **fvDOM + WSGG at `updateInterval: 1` is impractical above roughly
  3 × 10⁵ cells**; `updateInterval: 4` recovers 6.7× of that for 0.12 points
  of radiated fraction. **(b) The emissivity gate misses.** §62.12's Gate 1-E
  is outside its ±10 % bar at **58 of 108 points**, mean 11.44 %, worst
  30.52 %, with a **monotone temperature bias** — `+20.8 %` at 400 K to
  `-12.3 %` at 2400 K — against RADCAL. Bordbar's own table could not be
  obtained; neither model is truth, and the structured shape means the choice
  of coefficient set moves the answer most in a fire's smoke layer and its
  flame. **(c) The soot yield gate misses totally**, and on the shipped burner
  the reason is the mesh, not the model: `laminarSmokePoint` predicts
  **0.000 kg/kg** against Tewarson's measured 0.024 for propane because **0 of
  32,768 cells** on `cases/burnerPlume.jsonc` reach the model's own 1375 K
  formation threshold, so that run is bit-identical to having no soot at all.
  The smoke-point pair tests, on a duct hot enough to fire the model, pass.
- **Conjugate heat transfer's published gate misses, and its primary reference
  was never read (SPEC-LIT §60.5).** Kaminski & Prakash (1986) is paywalled;
  nine routes to an open copy were tried — ScienceDirect, Scholar, Semantic
  Scholar, OpenAlex, Unpaywall, CORE, arXiv and two institutional
  repositories — and none worked, so the comparison
  is against Belazizia et al. (2012) — open access, same configuration —
  labelled a SECONDARY source everywhere it appears. Gate 5 misses its 3 % bar
  at the conduction-dominated end — `-7.11 %` at `Kr = 0.1` on the live 40² run,
  shrinking to `-0.07 %` at `Kr = 10`, and `-7.12 / -3.00 / -0.48 %` across the
  mesh-converged sweep. The secondary table's own
  `Ra = 500` column sits above the analytic conduction limit, which is not
  physically possible, and this solver reproduces that limit to `1e-8`;
  nothing was tuned toward the reference.
- **The conjugate fluid region has exactly one inlet and one outlet, or
  neither (SPEC-LIT §79.2).** §60.2's closed cavity was lifted by §79 — a
  fluid patch may say `"kind": "inlet"` or `"outlet"`, and Gate 6 (Qu &
  Mudawar 2002) runs. What is still refused: a *second* opening, which would
  need a pressure level of its own to decide how the flow splits and a
  flux-establishment solve with more than one Dirichlet reference; a velocity
  profile at the inlet (uniform only); a pressure-driven opening; and a
  transient forced case (§59.6, unchanged). The outflow condition on `T` is
  `dT/dx = 0` while the flow leaves, where Qu & Mudawar's own (11) is
  `d2T/dx2 = 0`, and §79.12 measures what that costs rather than glossing it.
- **Lagrangian parcels evaporate and the gas now gets the vapour, but no
  driver reads a spray (SPEC-LIT §77.12).** §76 computes what leaves each
  droplet and §77 hands the gas all three of it — the mass into `Y_v`, the
  enthalpy the mass carries into the energy registry, and the volume it makes
  into the divergence constraint. What is still missing is a *driver*:
  §13.4.2 forbids adding a `parcels` block before something reads it, so the
  couplings are a library API driven by `ofgpu-validate`'s gates and
  `ofgpu-fire` still runs without a dispersed phase. Two model gaps are named
  rather than hidden: §25's gas has **one molar mass**, so adding water vapour
  does not make the mixture lighter (§54 measures that at 0.85 % in density at
  saturation), and §26 gives **every gas one `cp`**, so the vapour's enthalpy
  is booked at dry air's 1005 J/(kg·K) — which is what the 0.117 K of gate
  77-D is made of. **§78 built the wall interaction §68.13 refused**, so parcels
  now stick, rebound, spread or splash by the Bai-Gosman map and the mass that
  lands is accounted for bitwise — but **splash children are still not
  emitted** (the parent deposits whole and `n_splash` publishes the upper bound
  that makes), and there is still **no film transport** (§78.11): a deposited
  droplet stays where it landed, with no thickness, no momentum equation, no
  dripping and no re-entrainment, so the wetted-wall boundary conditions and
  the FM/NIST suppression law both remain out of reach. Parcels also still
  do not absorb radiation (§68.13) — water mist is a
  radiation shield and that is most of its value in suppression. Also absent:
  buoyancy and added-mass reaction on the gas, the momentum the vapour carries
  off, deposition into more than one cell along a crossing, and sub-grid
  dispersion of the coupled source. The hose-stream gate (§68.12, Theobald
  1981) **misses with the gas held at rest**, and the measurement says why:
  61–199 % of the throw is decided by the entrained air.
- **Surface-to-surface radiation and Lagrangian parcels are library
  capabilities with no case format.** Both are fully specified, gated and
  pair-tested, but **no driver binary reads an enclosure or a spray out of a
  case file** — §13.4.2 forbids adding a block before the driver that would
  read it, so `radiationModel viewFactor` is refused by name in the JSONC fire
  block and there is no `parcels` block at all. §50.12 and §68.13 record both
  boundaries as the next step.
- **No published separated-flow statistic is reproduced by the DES family, and
  the Spalart-Allmaras flat-plate gate is not run.** §57.12 and §56.11 name
  what stands in the way — no low-dissipation convection blending, no
  time-averaging seam, no synthetic-turbulence inlet, no curvilinear grids,
  and a TMR case that is compressible at `M = 0.2`. What is delivered is a
  correct implementation of published models, verified against their own
  definitions; `ofgpu-validate` prints that distinction on every run.
- ~~Only one cyclic pair.~~ **Fixed (SPEC-LIT §34.2).** `BlockSpec::cyclic`
  is a list now, and a JSONC case's `mesh.cyclic` accepts any number of
  pairs (one per axis) — a plane channel periodic in two directions, or a
  fully periodic box in three, can be declared today. See "Cyclic patches"
  above.

---

## References

Sources for the numerical methods and models. Section numbers refer to
[`rust/SPEC-LIT.md`](rust/SPEC-LIT.md). Each entry carries the bibliographic
detail SPEC-LIT itself records and no more: where a title, issue number or page
range is missing it is because the source was cited there without one, and §0
forbids supplying it from memory. Sources that were **not read** — paywalled,
unreachable, or deliberately left closed — say so on their own line.

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
- Lumley, J. L. (1978). *Advances in Applied Mechanics*, 18, 123–176. — §40
  (realizability as a modelling principle)
- Spalart, P. R., & Allmaras, S. R. (1992). *AIAA Paper 92-0439*; also
  *La Recherche Aérospatiale*, 1 (1994), 5–21. — §56 (the original)
- Allmaras, S. R., Johnson, F. T., & Spalart, P. R. (2012). Modifications and
  clarifications for the implementation of the Spalart-Allmaras turbulence
  model. *ICCFD7-1902.* A freely distributed conference paper — the copy
  actually read, and the implementation reference. — §56
- NASA / Turbulence Modeling Benchmarking Working Group. *Turbulence Modeling
  Resource — The Spalart-Allmaras Turbulence Model.* US government-authored
  DOCUMENTATION, not source; quoted here to the printed digit. — §56
- Rumsey, C. L., & Spalart, P. R. (2009). *AIAA Journal*, 47, 982–993. — §56
  (why the free-stream `nu~/nu` matters)

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

### Turbulence — hybrid RANS-LES

- Spalart, P. R., Jou, W.-H., Strelets, M., & Allmaras, S. R. (1997). Comments
  on the feasibility of LES for wings, and on a hybrid RANS/LES approach. In
  *Advances in DNS/LES*, Greyden Press, 137–147. — §57 (DES97)
- Shur, M., Spalart, P. R., Strelets, M., & Travin, A. (1999). *Engineering
  Turbulence Modelling and Experiments 4*, 669–678. — §57 (the
  `C_DES = 0.65` calibration on the SA background)
- Strelets, M. (2001). *AIAA Paper 2001-0879.* — §57 (SST-DES, the
  `k`-equation dissipation form)
- Spalart, P. R., Deck, S., Shur, M. L., Squires, K. D., Strelets, M. Kh., &
  Travin, A. (2006). *Theoretical and Computational Fluid Dynamics*, 20,
  181–195. — §57 (DDES, `r_d`, `f_d`, and the grid-induced separation they
  fix)
- Shur, M. L., Spalart, P. R., Strelets, M. Kh., & Travin, A. K. (2008).
  *International Journal of Heat and Fluid Flow*, 29, 1638–1649 — IDDES.
  **Paywalled and NOT read**, which is why §57's IDDES equations come from the
  two open-access restatements below. — §57
- Gritskevich, M. S., Garbaruk, A. V., Schütze, J., & Menter, F. R. (2012).
  *Flow, Turbulence and Combustion*, 88, 431–449 — the SST-background
  recalibration. **Paywalled and NOT read.** — §57
- Herr, F., Radespiel, R., & Probst, A. (2023). Improved delayed detached eddy
  simulation with Reynolds-stress background modeling. *arXiv:2301.07223v2*;
  published in *Computers & Fluids*, 265, 106014. **Appendix A is a complete
  restatement of the IDDES formulation**, and is where §57's IDDES equations
  come from, equation by equation. — §57
- Savino, A., Griffin, K., Lee, S., Vijayakumar, G., Wu, S., & Sprague, M.
  (2026). Improving boundary-layer separation prediction by an IDDES turbulence
  model using a pressure-gradient sensor. *arXiv:2603.08875.* Section 2 states
  SST-IDDES, and is where `C_DES1 = 0.78`, `C_DES2 = 0.61`, `C_w = 0.15` and
  the simplified filter width come from. — §57
- Nikitin, N. V., Nicoud, F., Wasistho, B., Squires, K. D., & Spalart, P. R.
  (2000). *Physics of Fluids*, 12, 1629–1632. — §57 (the log-layer mismatch
  `f_e` exists to remove)
- Spalart, P. R. (2009). *Annual Review of Fluid Mechanics*, 41, 181–202.
  — §57 (the review)
- Fröhlich, J., Mellen, C. P., Rodi, W., Temmerman, L., & Leschziner, M. A.
  (2005). *Journal of Fluid Mechanics*, 526, 19–66. — §57.12 (the
  periodic-hill gate, named and **NOT run**)

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

### Conjugate heat transfer

- Carslaw, H. S., & Jaeger, J. C. (1959). *Conduction of Heat in Solids*, 2nd
  ed., Oxford University Press, ch. I. — §46 (the anisotropic solid, and the
  affine transformation that reduces `div(K grad T)` to `lap T`)
- Aavatsmark, I. (2002). An introduction to multipoint flux approximations for
  quadrilateral grids. *Computational Geosciences*, 6, 405–432. — §46.4
  (the rigorous full-tensor treatment, and therefore the reason a full tensor
  on a skewed mesh is refused rather than approximated by its diagonal)
- Lipnikov, K., Shashkov, M., Svyatskiy, D., & Vassilevski, Yu. (2007).
  *Journal of Computational Physics*, 227, 492–512. — §46.4 (the nonlinear
  monotone alternative, named in the same refusal)
- Giles, M. B. (1997). *International Journal for Numerical Methods in Fluids*,
  25, 421–436. — §47 (the Godunov–Ryabenkii normal-mode analysis behind the
  classical "Dirichlet on the fluid, Neumann on the solid" rule)
- Meng, F., Banks, J. W., Henshaw, W. D., & Schwendeman, D. W. (2017). A stable
  and accurate partitioned algorithm for conjugate heat transfer. *Journal of
  Computational Physics*, 344, 51–85. — §47.7 (**Theorem 1**, the
  amplification factor `K_R/K_L` that is the reason Dirichlet–Neumann
  partitioning is not implemented here)
- Henshaw, W. D., & Chand, K. K. (2009). *Journal of Computational Physics*,
  228, 3708–3741. — §47 (Robin coefficients can always be chosen so the
  sub-time-step iteration converges)
- Verstraete, T., & Scholl, S. (2016). *International Journal of Heat and Mass
  Transfer*, 101, 852–869. — §47 (the numerical Biot number, and FFTB's
  instability above `Bi = 1`)
- Gander, M. J. (2006). Optimized Schwarz methods. *SIAM Journal on Numerical
  Analysis*, 44, 699–731. — §47 (the physical series conductance)
- de Vahl Davis, G. (1983). Natural convection of air in a square cavity: a
  bench mark numerical solution. *International Journal for Numerical Methods in
  Fluids*, 3, 249–264. — §59.8, the fluid-only anchor run first, because a
  conjugate answer built on an unvalidated buoyant solver measures nothing. Its
  four numbers are quoted here from Qi et al. (2013), *Nanoscale Research
  Letters*, 8, 56, Table 3 (open access), the primary being paywalled.
- Kaminski, D. A., & Prakash, C. (1986). *International Journal of Heat and
  Mass Transfer*, 29(12), 1979–1988. **Paywalled; no open-access copy was
  found and the primary table was never read**, so no title is asserted for it
  here either. — §60.5
- Belazizia, A., Benissaad, S., & Abboudi, S. (2012). Effect of wall
  conductivity on conjugate natural convection in a square enclosure with finite
  vertical wall thickness. *Advanced Theoretical and Applied Mechanics*, 5(4),
  179–190. Open access; an independent published solution of the
  Kaminski–Prakash configuration, itself validated against it. **The SECONDARY
  source Gate 5 actually compares against.** — §60.5
- Qu, W., & Mudawar, I. (2002). *International Journal of Heat and Mass
  Transfer*, 45, 3973–3985. — §47.12's Gate 6, **run live at §79.12** on
  `cases/quMudawar.cht.jsonc`, and it HOLDS: both substrate temperatures
  inside the experimental uncertainty the paper plots. Read in full from the
  authors' own copy.
- Kawano, K., Minakami, K., Iwasaki, H., & Ishizuka, M. (1998). *ASME
  HTD-361-3/PID-3*, 173–180. — the measured inlet and outlet thermal
  resistances Gate 6 is held against. **NOT obtained** (an ASME conference
  volume; no copy found), so the comparison is against Qu & Mudawar's own
  Fig. 4, **digitised**, and §79.12's Disclosure 1 says so.

### Combustion, soot and radiation

- Magnussen, B. F., & Hjertager, B. H. (1977). *Proceedings of the Combustion
  Institute*, 16, 719–729. — §27, §61.3
- McGrattan, K., McDermott, R., & Floyd, J. E. (2022). A simple two-step reaction
  scheme for soot and CO. *Proceedings of the Tenth International Seminar on Fire
  and Explosion Hazards (ISFEH10)*, Oslo, 23–27 May 2022. A NIST work, US public
  domain; fetched and read in full, and its Eqs. (1)–(5) are the model
  implemented. — §42
- McGrattan, K., Hostikka, S., McDermott, R., Floyd, J., Weinschenk, C., &
  Overholt, K. *Fire Dynamics Simulator Technical Reference Guide*, NIST SP 1018
  (NIST, US public domain; read locally from `reference/fds/Manuals/` — **no FDS
  source code was read**). — §25, §42, §43, §66
- Beyler, C. (2016). Flammability limits of premixed and diffusion flames. In
  *SFPE Handbook of Fire Protection Engineering*, 5th ed. The
  critical-flame-temperature and auto-ignition values, as quoted by two
  independent NIST sources both read here. — §43
- Morehart, J. H., Zukoski, E. E., & Kubota, T. (1991). NIST-GCR-90-585. — §43
  (the self-extinction bracket, as quoted by the FDS Technical Reference Guide)
- Lautenberger, C. W., de Ris, J. L., Dembsey, N. A., Barnett, J. R., & Baum,
  H. R. (2005). A simplified model for soot formation and oxidation in CFD
  simulation of non-premixed hydrocarbon flames. *Fire Safety Journal*, 40(2),
  141–176. — §61 (the laminar-smoke-point model, and every constant in it)
- Kent, J. H., & Honnery, D. (1990). *Combustion and Flame*, 79, 287. — §61
  (the measured formation-rate map the smoke-point polynomials are shaped to)
- Tewarson, A. *SFPE Handbook*, ch. 36, Table A.40, as quoted in the FDS
  Validation Guide (NIST, public domain). — §61.8 (the measured post-flame
  yield Gate 61-A misses)
- Modest, M. F. (2013). *Radiative Heat Transfer*, 3rd ed., Academic Press,
  ch. 5, 11, 15–17. — §28, §36, §50, §62, §65
- Fiveland, W. A. (1984). Discrete-ordinates solutions of the radiative
  transport equation for rectangular enclosures. *Journal of Heat Transfer*,
  106, 699. — §36, §65
- Truelove, J. S. (1987). Discrete-ordinate solutions of the radiation
  transport equation. *Journal of Heat Transfer*, 109, 1048. — §36, §65
- Hottel, H. C., & Sarofim, A. F. (1967). *Radiative Transfer*, McGraw-Hill.
  — §50 (the net-radiation exchange method), §62 (the weighted-sum
  construction itself)
- Bordbar, M. H., Węcel, G., & Hyppänen, T. (2014). A line by line based
  weighted sum of gray gases model for inhomogeneous CO2–H2O mixture in
  oxy-fired combustion. *Combustion and Flame*, 161(9), 2435–2445. — §62 (the
  coefficient set implemented). Its own tabulated emissivities could not be
  obtained, which is why Gate 1-E measures against RADCAL instead.
- Grosshandler, W. (1993). *RADCAL: A Narrow-Band Model for Radiation
  Calculations in a Combustion Environment*, NIST Technical Note 1402. US
  public domain; NIST's own implementation ships at `reference/fds/Source/rcal.f90` and
  `tools/radcal_emissivity/` compiles it **unmodified** behind a standalone
  driver. — §62.12 (the reference Gate 1-E measures the total emissivity
  against)
- Sung, Chen, Bundy, Fernandez & Hamins (2021). NIST Technical Note 2162r1 —
  the 37 cm gas burner's measured radiative fractions, 0.23 / 0.30 / 0.33 at
  20 / 34 / 50 kW. — §62.13 (Gate 4, which **misses**)
- Steckler, Quintiere & Rinkinen (1982) — the compartment doorway-flow
  measurements. Named in SPEC-LIT §42.8b as the prerequisite the Reduced Scale
  Enclosure miss points to, and **NOT run**; the paper itself has not been read
  here, so no report number or title is asserted for it.
- Walton, G. N. (2002). *Calculation of Obstructed View Factors by Adaptive
  Integration.* NISTIR 6925, National Institute of Standards and Technology. US
  Government, public domain. — §49 (the area integral and its dot-product form,
  the obstruction-elimination tests, the row-sum figure of merit, and the
  `BB104` benchmark)
- Shapiro, A. B. (1983). *FACET — A Radiation View Factor Computer Code for
  Axisymmetric, 2D Planar and 3D Geometries with Shadowing.* UCID-19887, Lawrence
  Livermore National Laboratory. US DOE, public domain. — §49.8 (the shadowed
  configuration `F_12 = 0.115621`)
- Howell, J. R. *A Catalog of Radiation Heat Transfer Configuration Factors*,
  3rd ed. Entries **C-11** and **C-14**, both tracing to Hottel (1931) and
  Hamilton & Morgan (1952). — §49.8 (the two analytic view-factor gates)
- Gebhart, B. (1961). *International Journal of Heat and Mass Transfer*, 3(4),
  341–346. — §50.2, the absorption-factor alternative, **named and not used**
- Balaji & Venkateshan (1993, 1994); Akiyama & Chong (1997) — the coupled
  convection-plus-surface-radiation cavity gate. — §50.12, **NOT run**: the
  tabulated `Nu_conv`/`Nu_rad` are paywalled and the fluid side has no case
  format for a radiating enclosure.

### Rheology and the contact angle

- Ostwald, W. (1925). *Kolloid-Zeitschrift*, 36, 99–117; de Waele, A. (1923).
  — §38 (the power law)
- Cross, M. M. (1965). *Journal of Colloid Science*, 20, 417–437. — §38
- Carreau, P. J. (1972). *Transactions of the Society of Rheology*, 16,
  99–127; Yasuda, K., Armstrong, R. C., & Cohen, R. E. (1981). *Rheologica
  Acta*, 20, 163–178. — §38 (one formula serves both)
- Herschel, W. H., & Bulkley, R. (1926). *Kolloid-Zeitschrift*, 39, 291–300.
  — §38
- Casson, N. (1959). In Mill, C. C. (ed.), *Rheology of Disperse Systems*,
  Pergamon, 84–104. — §38
- Papanastasiou, T. C. (1987). *Journal of Rheology*, 31, 385–404. — §38.3
  (the regularisation, in the **product** form)
- Bercovier, M., & Engelman, M. (1980). *Journal of Computational Physics*, 36,
  313–326. — §38.3 (the alternative regularisation)
- Frigaard, I. A., & Nouar, C. (2005). *Journal of Non-Newtonian Fluid
  Mechanics*, 127, 1–26. — §38.3 (what regularisation costs)
- Bird, R. B., Armstrong, R. C., & Hassager, O. (1987). *Dynamics of Polymeric
  Liquids*, vol. 1, 2nd ed., Wiley. — §38 (the family)
- Chhabra, R. P., & Richardson, J. F. (2008). *Non-Newtonian Flow and Applied
  Rheology*, 2nd ed. — §38.9 (Buckingham–Reiner)
- Young, T. (1805). *Philosophical Transactions of the Royal Society*, 95,
  65–87. — §39 (the equilibrium angle)
- Huh, C., & Scriven, L. E. (1971). *Journal of Colloid and Interface Science*,
  35, 85–101. — §39 (the moving contact-line singularity)
- Voinov, O. V. (1976). *Fluid Dynamics*, 11, 714–721; Cox, R. G. (1986).
  *Journal of Fluid Mechanics*, 168, 169–194. — §39.4 (the asymptotic matching)
- Hoffman, R. L. (1975). *Journal of Colloid and Interface Science*, 50,
  228–241. — §39.4 (the master curve)
- Jiang, T.-S., Oh, S.-G., & Slattery, J. C. (1979). *Journal of Colloid and
  Interface Science*, 69, 74–77. — §39.4 (the explicit correlation used
  here; Kistler's fit is **deliberately absent**, its four constants coming from
  a book chapter this project has not read)
- Afkhami, S., Zaleski, S., & Bussmann, M. (2009). *Journal of Computational
  Physics*, 228, 5370–5389 — the mesh-dependent (numerical-slip) angle.
  — §39.8, **named and deliberately not implemented** until the gate that would
  show it works exists
- Sui, Y., Ding, H., & Spelt, P. D. M. (2014). *Annual Review of Fluid
  Mechanics*, 46, 97–119. — §39 (the review)
- Washburn, E. W. (1921). *Physical Review*, 17, 273–283. — §39.7 (capillary
  rise)

### Lagrangian particles

- Dukowicz, J. K. (1980). A particle-fluid numerical model for liquid sprays.
  *Journal of Computational Physics*, 35, 229–253. — §66 (the discrete
  droplet model: the parcel, and the real-valued weight `n_p`)
- Crowe, C. T., Sharma, M. P., & Stock, D. E. (1977). The
  particle-source-in-cell (PSI-CELL) model for gas-droplet flows. *Journal of
  Fluids Engineering*, 99, 325. — §67, §68 (the per-cell sum every coupled
  source is)
- Crowe, C., Sommerfeld, M., & Tsuji, Y. (1998). *Multiphase Flows with Droplets
  and Particles*, CRC Press. — §66.2 (the equation of motion, and which of its
  terms survive)
- Maxey, M. R., & Riley, J. J. (1983). *Physics of Fluids*, 26, 883. — §66,
  §68 (the equation of motion: the added-mass coefficient, and the drag term
  §68 returns to the gas)
- Schiller, L., & Naumann, A. (1933). *Zeitschrift des Vereines Deutscher
  Ingenieure*, 77, 318, in the form compiled by Clift, R., Grace, J. R., &
  Weber, M. E. (1978). *Bubbles, Drops, and Particles*, Academic Press. — §66.3
  (the drag correlation)
- Macpherson, G. B., Nordin, N., & Weller, H. G. (2009). *Communications in
  Numerical Methods in Engineering*, 25, 263. — §66.6, barycentric tracking:
  the **paper** was read and it is **not implemented** — the one case the
  face-crossing walk cannot do. Its OpenFOAM implementation is GPL-3.0 and was
  **not** opened
- Elghobashi, S. (1994). On predicting particle-laden turbulent flows. *Applied
  Scientific Research*, 52, 309. — §67, §68 (the coupling map: at which
  `alpha_p` two-way coupling begins to matter, and where collisions do)
- Satish, N., Harris, M., & Garland, M. (2009). Designing efficient sorting
  algorithms for manycore GPUs. *IEEE IPDPS 2009.* The **paper** was read; no
  implementation of it was opened. — §67.4 (the three-phase radix pass)
- Merrill, D., & Grimshaw, A. *Parallel scan for stream architectures.*
  University of Virginia Technical Report CS2009-14. — §67.2
- Blelloch, G. E. (1990). *Prefix sums and their applications.* CMU-CS-90-190.
  — §67.2 (the exclusive scan and its work-efficiency argument)
- Hillis, W. D., & Steele, G. L., Jr. (1986). *Communications of the ACM*,
  29(12), 1170. — §67.2 (the in-block scan network, chosen for a property that
  is not speed)
- Steele, G. L., Jr., Lea, D., & Flood, C. H. (2014). Fast splittable
  pseudorandom number generators. *OOPSLA 2014*, ACM SIGPLAN Notices, 49(10),
  453. — §66.9 (the SplitMix64 finalising mix, used as a **bijection** and not
  as a generator)
- Ranz, W. E., & Marshall, W. R. (1952). Evaporation from drops. *Chemical
  Engineering Progress*, 48, 141 and 173. — §68.5 (the sensible-heat half of
  `Nu = 2 + 0.6 Re^(1/2) Pr^(1/3)`) and §76 (the mass-transfer half, and the
  `d(D²)/dt` metric its 56 experiments are correlated on)
- Spalding, D. B. (1953). The combustion of liquid fuels. *4th Symposium
  (International) on Combustion*, 847–864. — §76.6 (the mass transfer number
  `B_M` and the Stefan-flow rate)
- Godsave, G. A. E. (1953). Studies of the combustion of drops in a fuel spray.
  *4th Symposium (International) on Combustion*, 818–830. — §76.9 (the
  heat-limited rate at the boiling point)
- Abramzon, B., & Sirignano, W. A. (1989). Droplet vaporization model for spray
  combustion calculations. *International Journal of Heat and Mass Transfer*,
  32, 1605–1618. — §76.6 (`B_T = (1 + B_M)^φ − 1`, the default; their
  film-thickness corrections are **refused by name**)
- Sazhin, S. S. (2006). Advanced models of fuel droplet heating and
  evaporation. *Progress in Energy and Combustion Science*, 32, 162–214.
  — §76 (the survey; its effective-conductivity droplet interior is **named
  and not implemented**)
- Watson, K. M. (1943). Thermodynamics of the liquid state. *Industrial &
  Engineering Chemistry*, 35, 398–406. — §76.4 (`h_v(T)`)
- Marrero, T. R., & Mason, E. A. (1972). Gaseous diffusion coefficients.
  *Journal of Physical and Chemical Reference Data*, 1, 3–118. — §76.4
  (`D(H₂O–air) = 1.87e-10 T^2.072/p`, and hence the Lewis number §76.13's
  wet-bulb gap is made of)
- Lewis, W. K. (1922). The evaporation of a liquid into a gas. *Transactions of
  the ASME*, 44, 325–340. — §76.13 (the psychrometric ratio, and why a
  droplet's balance is not ASHRAE's wet bulb)
- Theobald, R. C. (1981). The effect of nozzle design on the stability and
  performance of turbulent water jets. *Fire Safety Journal*, 4, 1–13.
  — §68.12, Gate 68-C, which **misses with the gas held at rest**
- Bai, C., & Gosman, A. D. (1995). Development of methodology for spray
  impingement simulation. *SAE Technical Paper 950283.* — §78.1, §78.4 (the
  four-regime map, its Weber-number boundaries, and `We_c = A La^(−0.18)`)
- Mundo, C., Sommerfeld, M., & Tropea, C. (1995). Droplet–wall collisions:
  experimental studies of the deformation and breakup process. *International
  Journal of Multiphase Flow*, 21(2), 151–173. — §78.4 (`K = Oh Re^1.25`,
  `K_crit = 57.7`, the DEFAULT criterion); their splashing data themselves are
  **not transcribed**, which is what Gate 78-D is open about
- Yarin, A. L. (2006). Drop impact dynamics: splashing, spreading, receding,
  bouncing… *Annual Review of Fluid Mechanics*, 38, 159–192. — §78.1 (why every
  threshold in the map is a control and not a constant)
- IAPWS (2014). *Revised Release on Surface Tension of Ordinary Water
  Substance*, R1-76. — §78.2 (`σ = B τ^μ (1 + b τ)`, implemented and gated
  against the release's own table)

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
  Division, ASCE*, 90(5), 1–12. — §18, §53 (the same law integrated through a
  slab instead of over a cell)
- Idelchik, I. E. (2007). *Handbook of Hydraulic Resistance*, 4th ed., Begell
  House, Diagrams 8-1 to 8-6 — perforated plates and screens, the source of
  `K(sigma)`. **Not opened for §53**; the thin-plate form used is the one
  published in the open literature, and §53.7 gates it against its own limits.
  — §53
- Karki, K. C., Radmehr, A., & Patankar, S. V. (2003). Use of computational fluid
  dynamics for calculating flow rates through perforated tiles in raised-floor
  data centers. *HVAC&R Research*, 9(2), 153–166. — §53.8 (the per-tile
  flow-rate gate, **NOT run** — the paper was not reachable from this
  environment)
- Karki, K. C., & Patankar, S. V. (2006). Airflow distribution through perforated
  tiles in raised-floor data centers. *Building and Environment*, 41(6),
  734–744. — §53

### Ventilation, psychrometrics and data-centre metrics

- AMCA 210 / ASHRAE 51, *Laboratory Methods of Testing Fans for Certified
  Aerodynamic Performance Rating.* — §52 (what a manufacturer's curve **is**,
  and therefore why §52.5 carries a density and a speed correction rather than
  treating the table as absolute)
- NIST, *Fire Dynamics Simulator* verification suite, `Verification/HVAC/
  fan_test.fds` and `qfan_test.fds` with their published `.csv` reference values
  — US Government public domain. **The input files and their published results
  only**; no FDS source was read for the fan model. — §52.12 Gate 52-B
- Buzbee, B. L., Dorr, F. W., George, J. A., & Golub, G. H. (1971). The direct
  solution of the discrete Poisson equation on irregular regions. *SIAM Journal
  on Numerical Analysis*, 8(4), 722–736. — §52.9 (the capacitance-matrix path,
  **named and refused**)
- ASHRAE (2021). *ASHRAE Handbook—Fundamentals*, Chapter 1, "Psychrometrics."
  — §54.2 (whose equation numbering is used), §54.8 (Table 2, the external
  comparison)
- Hyland, R. W., & Wexler, A. (1983). Formulations for the thermodynamic
  properties of the saturated phases of H2O from 173.15 K to 473.15 K. *ASHRAE
  Transactions*, 89(2A), 500–519. — §54.2 (the `C1`–`C13` coefficients)
- Herrmann, S., Kretzschmar, H.-J., & Gatley, D. P. (2009). Thermodynamic
  properties of real moist air, dry air, steam, water, and ice (RP-1485).
  *HVAC&R Research*, 15(5), 961–986. — §54.3, **named and not implemented**;
  the enhancement factor it carries is what makes the ideal relations 0.44 % low
  in `W_s` at 25 °C, which §54.8 prints rather than tolerates
- Gatley, D. P., Herrmann, S., & Kretzschmar, H.-J. (2008). A twenty-first
  century molar mass for dry air. *HVAC&R Research*, 14(5), 655–662. — §54
  (where `M_a = 28.966 g/mol`, and hence `eps = 0.621945`, come from)
- Herrlin, M. K. (2005). Rack cooling effectiveness in data centers and telecom
  central offices: the Rack Cooling Index (RCI). *ASHRAE Transactions*, 111(2),
  725–731. — §55.1
- Herrlin, M. K. (2008). Airflow and cooling performance of data centers: two
  performance metrics. *ASHRAE Transactions*, 114(2), 182–187. — §55.2 (RTI)
- Sharma, R. K., Bash, C. E., & Patel, C. D. (2002). Dimensionless parameters for
  evaluation of thermal design and performance of large-scale data centers.
  *AIAA 2002-3091.* — §55.3 (SHI and RHI)
- ASHRAE Technical Committee 9.9 (2021). *Thermal Guidelines for Data Processing
  Environments*, 5th ed. — §55.1 (the Class A1–A4 recommended and allowable
  envelopes RCI is measured against)
- Wibron, E., Ljung, A.-L., & Lundström, T. S. (2019). *Energies*, 12(8), 1473. **CC-BY-4.0, licence verified live through the
  Crossref REST API**, but the full text was not reachable from this environment,
  so §55.8's six-configuration ranking gate is **NOT run** and only the one
  relation the abstract states is gated. — §55.8

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
