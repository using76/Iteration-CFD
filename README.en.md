# meteor-cfd

**A GPU-resident finite volume CFD solver**

Meteo Simulation Co., Ltd. · Rust host, CUDA kernels · [한국어](README.md)

---

## Overview

meteor-cfd is an unstructured finite volume CFD solver designed so that the entire time-integration loop stays on the GPU. Once the mesh and fields are uploaded, no device allocation and no field transfer to the host occur inside the loop. It is **single-GPU only**, and what it is for is incompressible and low-Mach flow — RANS/LES/hybrid turbulence, buoyant plumes, variable-density low-Mach flow, two-phase VOF, conjugate heat transfer, surface-to-surface radiation, Lagrangian sprays, and ventilation and data-centre airflow with fan curves and porous jumps. The numerical core is implemented directly from published literature, and every formulation is specified in [`rust/SPEC-LIT.md`](rust/SPEC-LIT.md) with a citation to its original paper. Validation uses the method of manufactured solutions, analytical solutions and published benchmarks only — **never a comparison against another CFD code.** A Rust 1.85 host with CUDA C++ kernels; double precision by default, single via the `single` feature; NVIDIA GPUs; cudarc and thiserror are the only dependencies, with AMGX optional.

---

## Licence

| Use | Terms |
|---|---|
| Personal study, hobby and amateur work | **Free** |
| Educational institutions — teaching, coursework, lab work | **Free** |
| Universities and their institutes and laboratories | **Free** |
| Public research organisations and government institutions | **Free**, whatever the funding |
| Public safety, public health and environmental protection organisations | **Free** |
| Charitable organisations | **Free** |
| Any other commercial use | **Thirty-day trial**, then a commercial licence |

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

## Status

**This is the output of actually running it on this working tree on 2026-09-03**, on an NVIDIA GeForce RTX 5070 Ti (sm_120), CUDA 13.3, double precision.

```
cargo test --release   1,853 passed, 0 failed, 6 ignored   (all targets, 18 suites)
                       1,699 passed, 0 failed, 4 ignored   (the lib crate alone)

ofgpu-validate         901 / 901 checks passed
                       853 computed live, 48 replayed from recorded measurements
                       then a list naming the 2 gates whose verdict is MISSES and the 5 OPEN
```

That list is not maintained by hand: it is **generated** from the registry each gate enters at the point it reports its own verdict (SPEC-LIT §69). Printing a verdict and registering one are the same call, so all seven are named on every run and an eighth could not fail to appear. **Everything `ofgpu-validate` runs passes. That is a different statement from "this project reproduces every published benchmark it compares against", and the two must not be confused.**

---

## Build and run

Requirements: Rust 1.85 or newer, Visual Studio 2022 (C++ workload), CUDA Toolkit 13.x. `build.rs` sets up the MSVC environment through `vcvars64.bat` and compiles every `.cu` to CUBIN rather than PTX.

```powershell
cd rust
cargo build --release
cargo run --release --bin ofgpu-generate-mesh -- channel ..\cases\channel 200 120 1
cargo run --release --bin ofgpu-k-epsilon     -- ..\cases\channel -iters 4000 -check 400
```

The whole validation suite is `cargo run --release --bin ofgpu-validate`. The other fifteen executables (`ofgpu-lowmach`, `ofgpu-vof`, `ofgpu-cht`, `ofgpu-datacentre`, `ofgpu-decompose`, the benchmarks), the case file format and the command-line options are in the **user guide**. Cases are read and written either as a single JSONC file or as an OpenFOAM ASCII case directory — the latter for interoperability with existing tools such as ParaView and `foamToVTK`; meteor-cfd links against no part of OpenFOAM and contains none of its source.

---

## What it can do

| Area | Supported |
|---|---|
| Discretisation | Gauss linear, upwind, linearUpwind, cubic, QUICK, Gamma, blended; six TVD limiters; Green–Gauss and least-squares gradients with cell- and face-limiters (Barth–Jespersen, Venkatakrishnan); over-relaxed non-orthogonal correction; steadyState, Euler, BDF2, local time stepping |
| Pressure–velocity | SIMPLE, SIMPLEC, PISO, PIMPLE. Rhie–Chow interpolation, with body forces treated at the face rather than interpolated from cell values |
| Turbulence | RANS: standard, realizable and RNG k-ε, Wilcox k-ω, Menter SST, Launder–Sharma low-Re, Spalart-Allmaras (four variants). LES: Smagorinsky, WALE, Deardorff. Hybrid: DES97, DDES, IDDES. Transition: k-ω SST-LM. Wall treatment: standard and continuous wall functions, `lowRe` integration, the Jayatilleke thermal wall function, roughness |
| Multiphase and transport | VOF (interface compression, Zalesak FCT bounding, CSF surface tension, static, hysteresis and dynamic contact angles), multicomponent species, six generalised-Newtonian viscosity models, Darcy–Forchheimer porosity, non-Boussinesq buoyancy |
| Conjugate heat transfer | Solid regions, contact resistance, harmonic-mean interface conductivity, a fluid region with up to one inlet and one outlet |
| Ventilation and data centre | Fan performance curves (AMCA 210 corrections), porous jumps, moist air (Hyland–Wexler), RCI, RTI, SHI and RHI metrics |
| Lagrangian parcels | SoA pool, Schiller–Naumann drag, two-way coupling, evaporation and vapour coupling, the Bai-Gosman wall-impact map — **as a library API only** |
| Linear solvers | PBiCGStab, PCG; Jacobi, multi-colour DIC and DILU; a cuFFT direct Poisson backend and AMGX (optional feature); backend chosen automatically on applicability, accuracy and measured time |
| Mesh and I/O | Block meshes with grading, castellated STL carving, cut cells, Gmsh v4.1, multiple cyclic pairs, `empty`/`symmetry` constraints, 2:1 adaptive refinement (wired to no solver); JSONC cases with a generated schema, OpenFOAM ASCII, VTU, NanoVDB/OpenVDB, USD, double-precision restart |

## What it cannot do

- **No MPI and no multi-GPU.** Decomposition, halos, decomposition-invariant reductions and distributed PCG/PBiCGStab are all implemented and gated (§71–§73), but they run in one process on one card. No communication library is linked and no strong-scaling number is published.
- **No compressible or transonic flow.** The density-weighted time derivative is used by VOF, but the pressure equation is incompressible.
- **No finite-rate (Arrhenius) chemistry** — no stiff ODE integrator, no Jacobian, no reaction mechanism.
- **Crank–Nicolson cannot be used with an under-relaxed equation** — it reports the reason as an error rather than silently falling back to Euler.
- **No combustion, no soot, and no participating-medium radiation.** There is no reaction model, no soot equation, and no P1 or fvDOM solver in this engine. Both of those model names are still RECOGNISED by the `radiationModel` selector and resolved by nothing (§13.4), and a case naming `physics.fire`, `physics.fire.combustion`, `physics.fire.radiation`, `physics.fire.soot` or `initial.Y_F` is refused **by name**, with what it selected and what is here instead — never read and dropped. Radiation here means §49/§50's surface-to-surface exchange across a *transparent* medium.
- **Surface-to-surface radiation, species transport and Lagrangian sprays have no case format.** All three are specified and gated as library APIs, but no driver binary reads an enclosure, a species set or a spray out of a case file (§50.12, §13.4.2).
- **Parcels do not create splash children, there is no film transport, and parcels do not absorb radiation** (§78.11, §68.13).
- **Adaptive refinement is wired to no solver** — face fluxes are not transferred and there is no post-adaptation pressure projection.
- **AMGX is off by default**, and with it off the selector still reports AMGX explicitly as "unavailable".
- **It is not claimed that the DES family reproduces a published separated-flow statistic**, and Spalart-Allmaras's TMR flat-plate gate is not run (§57.12, §56.11).

**And two gates that miss.** `ofgpu-validate` names them on every run; these are its own names for them.

| Gate | Verdict |
|---|---|
| §60.5 Gate 5 — conjugate natural convection in a square enclosure (Kaminski & Prakash 1986) | **MISSES** its 3 % bar at the conduction-dominated end: −7.11 % at `Kr = 0.1`, −0.07 % at `Kr = 10`. The primary reference is paywalled and was never read, so the comparison is against Belazizia et al. (2012), a **secondary source** |
| §68.12 Gate 68-C — Theobald's (1981) 90 hose streams | **MISSES with the gas held at rest**: the throws average **61.29 %** of the measured range, while a vacuum bracket with no drag reaches 198.65 %, so what decides the throw is entrained air |

**Five more verdicts are `OPEN`** and are printed as a second group of the same list. Three hold §32.4's plane channel against a **correlation rather than a measurement** (Gnielinski 1976); the fourth, `78-D`, is open because the two published splash criteria disagree with **each other** by a factor of 4.78 in Weber number; and the fifth, §88.10 Gate 88-T, because no measured onset `Re_x` for the T3A flat plate could be found to close the comparison against.

---

## Where the detail went

| Document | Contents |
|---|---|
| **User guide** (separate page) | Building, the case file, the settings contract, running, output, what it cannot do |
| **Technical guidebook** (separate page) | Discretisation, boundary conditions, pressure–velocity, turbulence, low-Mach, surface-to-surface radiation, validation, GPU residency, mesh adaptation and performance |
| [`rust/SPEC-LIT.md`](rust/SPEC-LIT.md) | The numerical specification, 79 sections, with a citation for every formulation — both guides are drawn *from* it and neither replaces it |
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
- McGrattan, K., Hostikka, S., McDermott, R., Floyd, J., Weinschenk, C., Overholt, K., Vanella, M., et al. *Fire Dynamics Simulator Technical Reference Guide*, NIST SP 1018-1. NIST, US public domain; read locally from `reference/fds/Manuals/` with `reference/fds/LICENSE.md` read verbatim. **No FDS source code was read for these sections.** — §25, §42, §43, §66, §68, §76
- Modest, M. F. *Radiative Heat Transfer*, 3rd ed., Academic Press (2013), ch. 5. — §50
- Hottel, H. C., & Sarofim, A. F. *Radiative Transfer*, McGraw-Hill (1967), ch. 3, 5. — §50 (the net-radiation exchange method; the method of images, named in §50.9's refusal), §62 (the weighted-sum construction itself)
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

**simul@msimul.com** · Meteo Simulation Co., Ltd. / 주식회사 메테오시뮬레이션

Teaching and academic research are free. Industrial R&D, research institutes outside
an educational institution, contract work and commercial use require a licence — see
sections 2 and 3 of [`LICENSE`](LICENSE). If the boundary is unclear, please ask.
