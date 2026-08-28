# Provenance — where every file came from

meteor-cfd is source-available under the Meteor Simulation Source-Available
License 1.0 (see ../LICENSE). This file is the per-file record that
[`../LICENSING.md`](../LICENSING.md) §6 phase 4 calls for: for each source
file, what it was written from.

Three categories, and nothing falls outside them:

- **literature** — implemented from the papers named in the file's own header,
  via [`SPEC-LIT.md`](SPEC-LIT.md), which carries the citations.
- **original** — designed here. No external source, permissive or otherwise.
- **format** — reads or writes the OpenFOAM ASCII case format. A file format is
  not a work of authorship, and interoperability is the whole purpose.
- **carried over** — moved across from this project's own earlier C++ when the
  crate became Rust, rather than rewritten. See the note below; this matters,
  and it is why the category exists.

**No file in this tree was written from GPL-licensed source.** The numerical
core that was is gone; see `LICENSING.md` for what was removed and why.

## What "carried over" means, and why it is not a hole in the audit

The crate was Rust-with-CUDA before the relicensing rewrite, and it had been
ported from an earlier C++ version of the same project. When the GPL-derived
numerics were deleted and rewritten, three groups of files were **not**
rewritten, because nothing in them was derived from anyone else's code:

| Carried over | What it does |
|---|---|
| `src/io/*` | reads and writes the ASCII case format |
| `src/blockgen.rs` | generates structured meshes and writes them in that format |
| `src/bin/*` | argument parsing, case loading, reporting loops |

Their headers used to say *"Transcribed from `gpu/common/io/foam_io.cu`"*, and
one said the deleted C++ was *"the specification"* — wording that is accurate
about the mechanics, useless to a reader now that the path is gone, and
alarming in exactly the way the numerics turned out to deserve. They now state
the real chain: the C++ they came from was itself written from the case format
as it appears in data files, not from any CFD code's source.

The distinction that matters: the deleted C++ had its **numerics** transcribed
from OpenFOAM. Its **I/O layer and mesh generator** were not — they were
written from data files, which is why they are here and the numerics are not.
This paragraph exists because an audit of this repository flagged the
contradiction between the old headers and this document. The headers were
wrong; the classification was right.

---

## Numerical core — literature

Each of these carries its own citation header. The papers are listed in
`SPEC-LIT.md`; the section column says which part of it specifies the file.

| File | SPEC-LIT | Primary sources |
|---|---|---|
| `src/mesh/geometry.rs` | §2 | Jasak (1996) §3.2, §3.3.1, §3.4.2; Moukalled et al. (2016) §6.4, §8.6.4; Ferziger & Perić §8.6 |
| `cuda/fv.cu`, `src/fv.rs` | §3, §7 | Jasak (1996) ch. 3; Moukalled et al. ch. 8, 11, 12, §15.4; Patankar (1980) ch. 4–6; Ferziger & Perić §6.3.2; Sweby (1984); van Leer (1977, 1979); van Albada et al. (1982); Roe (1986); Darwish & Moukalled (2003) |
| `cuda/ldu.cu`, `src/ldu_ops.rs` | §1, §3, §4, §5.2 | Jasak (1996) ch. 3; Patankar (1980) §4.2–4.9; Moukalled et al. ch. 8; Saad (2003) §3.4 |
| `cuda/field.cu`, `src/field_ops.rs` | §4 | the Robin boundary form; the representation itself is *DESIGN* |
| `cuda/solver.cu`, `src/solver.rs` | §8 | Saad (2003) §6.7, §7.4.2, ch. 10, §12.4; van der Vorst (1992); Hestenes & Stiefel (1952) |
| `cuda/turbulence.cu`, `src/turbulence.rs`, `src/models/*` | §6.1–6.3, §17 | Launder & Spalding (1974); Wilcox, *Turbulence Modeling for CFD*; Menter (1994); Rodi (1987) and Henkes, van der Vlugt & Hoogendoorn (1991) for the buoyancy production `G_b` and its `C_3` |
| `cuda/sst.cu`, `src/models/k_omega_sst.rs`, `src/models/k_omega_sst/kernels.rs` | §6.3, §6.6 | Menter (1994); Menter, Kuntz & Langtry (2003) — the **2003 revision** is what is implemented, and the file says which two terms differ; Bradshaw, Ferriss & Atwell (1967) for `tau = a_1 k`; Patankar (1980) §4.2 for the cross-diffusion linearisation |
| `src/models/launder_sharma.rs`, the §33 kernels in `cuda/turbulence.cu` (`turbNutLaunderSharma`, `turbLsSqrtPositive`, `turbLsDTerm`, `turbLsGradGradUMagSqr`, `turbLsETerm`, `turbLsEpsilonSources`) | §33 | Launder & Sharma, *Letters in Heat and Mass Transfer* 1 (1974) 131–138; Patel, Rodi & Scheuerer, *AIAA J.* 23 (1985) 1308 (background on the low-Re family). Coefficients are §6.1's own, reused unchanged; the `grad(grad U)` route (Gauss gradient of the already-computed velocity gradient, boundary faces zero-gradient extrapolated) is *DESIGN*, as SPEC-LIT §33.1 itself marks it |
| `src/walldistance.rs`, `turbWallDistance` in `cuda/turbulence.cu` | §6.6 | Tucker, *Applied Mathematical Modelling* 22 (1998) 293–305 — one Poisson solve, no search |
| `cuda/les.cu`, `src/les.rs`, `src/models/les.rs` | §6.5, §16 | Smagorinsky (1963); Nicoud & Ducros (1999); Deardorff (1970, 1980); Scotti, Meneveau & Lilly (1993); van Driest (1956). **The Deardorff kernel follows the algebraic form used by FDS (NIST, public domain — see `reference/fds`, `Source/velo.f90`, and the FDS Technical Reference Guide), used with thanks and acknowledged in the file header.** Its unstructured-mesh test filter is ours. |
| `cuda/wallfunctions.cu`, `src/wallfunctions.rs` | §6.4, §15.1–§15.3, §15.5, §29.2, §29.3, §30.1 | Launder & Spalding (1974); Spalding, *J. Appl. Mech.* 28 (1961) 455; Kader, *Int. J. Heat Mass Transfer* 24 (1981) 1541 (the exponential blend); Menter & Esch (2001) and Popovac & Hanjalić, *Flow Turbul. Combust.* 78 (2007) 177 (blending precedent); Cebeci & Bradshaw, *Momentum Transfer in Boundary Layers*, Hemisphere (1977) — the rough-wall downshift `dB(Ks+, Cs)`; Jayatilleke, *Prog. Heat Mass Transfer* 1 (1969) 193–330 — the thermal sublayer-resistance correction `P(Pr/Pr_t)`; **Werner & Wengle, "Large-eddy simulation of turbulent flow over and around a cube in a plate channel", 8th Symp. Turb. Shear Flows (1991)** — the LES wall model `tau_w_werner_wengle`/`nut_wall_werner_wengle`, integrated-and-inverted over the first cell rather than solved by Newton iteration |
| `src/energy.rs` (thermal wall wiring only — the low-Mach formulation itself is cited separately below) | §29.3 | Jayatilleke (1969), as above; the law and its device kernel live in `src/wallfunctions.rs`/`cuda/wallfunctions.cu`, this file owns only which faces get it and when the Robin triple is rewritten |
| `src/wallfunctions.rs` (`dittus_boelter_nu`, `gnielinski_nu`, `gnielinski_f`), `src/bin/validate.rs` (`check_nu_correlations`, `check_thermal_wall_function_gate_verdict_replay`) | §32.3, §32.4 | Dittus & Boelter, *Univ. Calif. Publ. Eng.* 2 (1930) 443 (reprinted in *Int. Commun. Heat Mass Transfer* 12 (1985) 3); Gnielinski, *Int. Chem. Eng.* 16 (1976) 359 — the two independent, published turbulent-pipe-flow Nusselt-number correlations SPEC-LIT §32's redesigned thermal-wall-function gate compares a live measurement against, in place of comparing two runs of this code against each other |
| `src/models/registry.rs`, `src/models/coupled.rs` | §6, §6.3, §6.5, §6.6, §13.4, §16, §17, §30.2 | Launder & Spalding (1974); Wilcox; Menter (1994, 2003); Smagorinsky (1963); Nicoud & Ducros (1999); Deardorff (1980); Rodi (1987) and Henkes, van der Vlugt & Hoogendoorn (1991) for `G_b`. `registry.rs` *dispatches* to `kOmegaSST` and to the three LES closures rather than naming them in a refusal, and refuses what is still missing; `coupled.rs`'s `CoupledTurbulence` is what lets `src/bin/buoyant.rs` (a coupled solver, not a standalone RAS/LES driver) reach that same dispatch instead of constructing `KEpsilon` directly regardless of what the case asked for — `src/bin/fire.rs` does not use this trait yet and refuses `kOmegaSST`/LES by name (§13.4), its combustion mixing-time closure needing `epsilon` directly being the stated reason; see the *DESIGN* table for the trait itself |
| `src/field.rs`, `src/field_setup.rs` (boundary conditions) | §4, §13.4, §15.2, §15.5, §15.6 | Launder & Spalding (1974) for `k = 3/2 (I\|U\|)²` and `ε = C_mu^{3/4} k^{3/2}/L`; Wilcox for the ω form |
| `cuda/momentum.cu`, `src/momentum.rs` | §5.1, §9 | Rhie & Chow (1983); Moukalled et al. §15.6; Ferziger & Perić §7.5 |
| `cuda/simple.cu`, `src/simple.rs` | §5.2–5.4, §14 | Patankar & Spalding (1972); Patankar (1980) ch. 6; Van Doormaal & Raithby (1984); Issa (1986); Ferziger & Perić §7.4 |
| `src/scalar_transport.rs` | §3, §18 | the same assembly with an effective diffusivity, plus the volumetric sources of §18 |
| `cuda/sources.cu`, `src/sources.rs` | §3.4, §18 | Patankar (1980) §4.2; Ward (1964) for Darcy–Forchheimer |
| `cuda/species.cu`, `src/species.rs` | §19 | the `N-1` formulation, boundedness and sum-to-one |
| `src/reference.rs` | §3, §10 | the CPU mirror, written as scatter loops so its structure differs from the device gather |
| `src/potential_flow.rs` | — | Ferziger & Perić §7.1 |
| `cuda/vof.cu`, `src/vof.rs` | §20, §22 | Hirt & Nichols (1981); Ubbink (1997); Rusche (2002); Zalesak (1979); Brackbill, Kothe & Zemach (1992); Rhie & Chow (1983); Issa (1986); Ferziger & Perić §7.5 |

| `src/timescheme.rs`, `cuda/timescheme.cu` | §3.3, §13 | Crank & Nicolson (1947); Ferziger & Perić §6.3 — the theta method, BDF2 with variable dt, local time stepping |
| `src/precon.rs`, `cuda/precon.cu` | §21 | Saad (2003) §12.4 — multi-colour DIC / DILU |
| `src/walldistance.rs` | §6.6 | Tucker, *Appl. Math. Modelling* 22 (1998) 293 |
| `src/models/k_omega_sst.rs`, `cuda/sst.cu` | §6.3 | Menter, *AIAA J.* 32 (1994) 1598; Menter, Kuntz & Langtry (2003) |
| `src/les.rs`, `src/models/les.rs`, `cuda/les.cu` | §6.5, §16 | Smagorinsky (1963); Nicoud & Ducros (1999); Deardorff (1970, 1980); Scotti, Meneveau & Lilly (1993); van Driest (1956) |
| `src/vof.rs`, `cuda/vof.cu` | §20 | Hirt & Nichols (1981); Zalesak (1979); Brackbill, Kothe & Zemach (1992); Ubbink (1997) and Rusche (2002) theses |
| `src/species.rs`, `cuda/species.cu` | §19 | advection–diffusion with a turbulent Schmidt number |
| `src/sources.rs`, `cuda/sources.cu` | §18 | Patankar (1980) §4.2; Ward, *J. Hydraul. Div. ASCE* 90 (1964) 1 |
| `src/io/schemes.rs` | §11, §12, §13.4 | Warming & Beam (1976); Leonard (1979); Khosla & Rubin (1974); Jasak, Weller & Gosman (1999); Barth & Jespersen (1989); Venkatakrishnan (1993) |
| `src/surface/mod.rs`, `src/surface/stl.rs` | §23.1–§23.2 | The STL format (3D Systems, 1987 — a de facto public specification); Aftosmis, Berger & Melton, *AIAA J.* 36(6) (1998) 952 for the castellation context. Normals are recomputed from the winding; stored normals are never used |
| `src/surface/classify.rs` | §23.3 | Column-parity ray casting with simulation-of-simplicity jitter and a 3-axis majority vote; Barill, Dickson, Schmidt, Levin & Jacobson, *ACM TOG* 37(4) (2018) — the exact solid-angle winding number as the arbiter for cells the vote cannot settle |
| `write_carved_case` and the carver in `src/blockgen.rs` | §23.4–§23.5 | Aftosmis, Berger & Melton (1998), the “castellate” stage only; the FDS precedent (NIST, public domain) for stair-step obstructions |

## Fire physics — literature

`ofgpu-fire`'s low-Mach variable-density solver: SPEC-LIT sections 25-28.

| File | SPEC-LIT | Primary sources |
|---|---|---|
| `src/energy.rs` | §25, §26 | Rehm & Baum, *J. Res. Natl. Bur. Stand.* 83 (1978) 297-308 (the `p = p0(t) + p~` split, the divergence constraint and `p0` evolution); Majda & Sethian, *Combust. Sci. Technol.* 42 (1985) 185 (low-Mach acoustic filtering); the FDS Technical Reference Guide (McGrattan et al., NIST SP 1018, public domain — `reference/fds` read and adapted for the SHAPE of the divergence constraint and the sealed/open `p0` bookkeeping, acknowledged per SPEC-LIT §0; no FDS code copied); Patankar (1980) §4.2 for the `Su`/`Sp` linearisation the `EnergySources` registry hands to `fvm_su`/`fvm_sp` |
| `src/combustion.rs`, `cuda/combustion.cu` | §27 | Magnussen & Hjertager, *Proc. Combust. Inst.* 16 (1977) 719-729 (the eddy-dissipation model, `omega_F = C_EDM rho (eps/k) min(Y_F, Y_O2/s)`); Poinsot & Veynante, *Theoretical and Numerical Combustion*, for background; the FDS Technical Reference Guide (public domain) read for acknowledgement that FDS's own mixing-controlled default rests on the same idea — no FDS machinery used; Patankar (1980) §4.2 for the fuel sink's implicit linearisation |
| `src/radiation.rs`, `cuda/radiation.cu` | §28 | Modest, *Radiative Heat Transfer*, 3rd ed., ch. 15 (the P1/differential approximation and its Marshak boundary condition); Patankar (1980) §4.2 for the `T^4 ~= 4 T0^3 T - 3 T0^4` implicit emission linearisation |

## New I/O formats and machinery — format / literature

| File | What it is |
|---|---|
| `src/io/msh.rs` | Gmsh `.msh` 4.1 reader (tet/hex/prism/pyramid cells, `$PhysicalNames` patches). The FORMAT is published (Gmsh manual) and not copyrightable; `reference/pyfr`'s BSD-3 `pyfr/readers/gmsh.py` was the legal cross-check for blocked `$Nodes`, non-contiguous tags and physical names — no Gmsh source was read |
| `src/surface/stl.rs`, `src/surface/obj.rs` | STL (3D Systems, 1987, a de facto public spec) and Wavefront OBJ (also a published, uncopyrightable format) readers, for surface intake (SPEC-LIT §23) |
| `src/surface/classify.rs` | Column-parity ray-casting inside/outside classification (SPEC-LIT §23.3); Barill, Dickson, Schmidt, Levin & Jacobson, *ACM TOG* 37(4) (2018), the exact solid-angle winding number as tie-break |
| `src/surface/cutcell.rs`, the cut-cell section of `src/blockgen.rs` | Embedded-boundary volume/area fractions by supersampling (SPEC-LIT §24) — Aftosmis, Berger & Melton, *AIAA J.* 36(6) (1998) 952 for the castellation/cut-cell context; the cut face's closure-defined area vector, the merge rule and the sample count are *DESIGN* (§24.3, §24.5) |
| `src/restart.rs` | The `.mcr` restart file: full-precision `phi`, mesh-hash guarded, versioned. Format and layout are *DESIGN* — original, no external spec |
| `src/io/case_json.rs` | The JSONC case reader/lowering (`docs/05-io-redesign.md` §4.1). `schemars`-generated JSON Schema so the schema and the reader cannot disagree; the patch-major, ordered-array layout is prior art from `reference/pyfr`'s `[soln-bcs-<patch>]` INI convention (BSD-3), read as a case-format precedent, not code |
| `src/io/vtu.rs` | Appended-binary VTU/VTK writer; `reference/pyfr`'s BSD-3 `pyfr/writers/vtk/` was the cross-check for the encoding and offsets |
| `src/io/vdb.rs`, `src/io/nvdb.rs` | OpenVDB-compatible and NanoVDB volume writers. The NanoVDB layout is read from the Apache-2.0 PNanoVDB/NanoVDB headers (AcademySoftwareFoundation GitHub) |
| `src/io/usda.rs` | USD ASCII (`.usda`) scene writer referencing the VDB output — the format is Pixar's published USD spec |

## Ours by design

Marked *DESIGN* in `SPEC-LIT.md` where the literature does not prescribe the
choice. These are documented as our decisions in the code, not attributed to
anyone.

| Thing | Where | SPEC-LIT |
|---|---|---|
| The single Robin triple `(fr, ref_value, ref_grad)` for every BC | `src/field.rs`, `src/field_setup.rs` | §4 |
| Vertex welding by exact bit-equality only — no epsilon weld, because an epsilon is a silent geometry edit | `src/surface/mod.rs`, `src/surface/stl.rs` | §23.1 |
| Faces carved against solid cells are `wall` type and receive the same wall boundary conditions blockgen already writes, so a carved case runs unmodified | `src/blockgen.rs` | §23.4 |
| Gather over a cell→face CSR instead of scatter over faces | `src/mesh/topology.rs` and every kernel | §1 |
| Continuous blending across `y+_lam`, and the wall-adjacent-cell treatment | `src/wallfunctions.rs` | §6.4 |
| Bounding of `k` and `epsilon` | `src/turbulence.rs` | §6.1 |
| `NO_WALL = 1e10` m in a domain with no wall in it, rather than a singular all-Neumann Poisson solve — which is also the physically right answer, since `F_1 → 0` and the van Driest length grows without bound | `src/walldistance.rs` | §6.6 |
| Reading the wall faces from the mesh's `PatchKind::Wall` rather than from any one field's `boundaryField`, since the wall distance is a property of the geometry | `src/walldistance.rs` | §6.6 |
| Cell extents measured as `2·max_f |Cf − C|` over the cell's own faces — exact for an axis-aligned hexahedron, and biased *down* (so `nu_t` down) on a skewed one, because the mesh does not retain the points a cell was built from | `cuda/les.cu` | §16.2, §16.3 |
| `y+` for van Driest damping from the LOCAL wall-normal shear, with the wall normal taken from `grad y` — no nearest-wall search, and van Driest's own reading of `A+` | `cuda/les.cu`, `src/les.rs` | §16.4 |
| Applying van Driest damping only where `y+ > 0`, so that a quiescent field or a wall-free box keeps its geometric filter width instead of having it annihilated | `cuda/les.cu` | §16.4 |
| Filter-width smoothing: ratio 1.15, two sweeps, off by default, raising rather than lowering — the same sweep as §13.2's local time step | `src/les.rs`, `cuda/les.cu` | §16.5 |
| The unstructured-mesh test filter for Deardorff — a face-neighbour gather that reduces to FDS's `(1, 2, 1)/4` kernel in one dimension, because an unstructured mesh has no diagonal neighbours | `cuda/les.cu` | §6.5 |
| `C_s = 0.168`, Lilly's inertial-range value, rather than a wall-tuned fit — with the reasoning, and the dictionary entry that overrides it, in the code | `src/models/les.rs` | §6.5 |
| `RasModel::Les` as a variant of the model enum, so a RANS-only driver refuses an LES case by name instead of reading it as laminar | `src/models/registry.rs` | §13.4 |
| Residual normalisation | `src/solver.rs` | §8.4 |
| Density-ratio buoyancy `b = g(T_ref/T − 1)` | `src/momentum.rs` | §9 |
| The over-relaxed `0.05·|d|` floor on the delta coefficient | `src/mesh/geometry.rs` | §2.4 |
| The limited-snGrad expression `min(1, α·\|orth\|/(\|corr\|+ε))` | `src/io/schemes.rs` | §12.3 |
| The LTS smoothing sweep and damping | `src/timescheme.rs` | §13.2 |
| `-permissive` and the loud-failure contract | `src/io/contract.rs` | §13.4 |
| Each field owning its own wall-function decision | `src/wallfunctions.rs` | §15.5 |
| The `wallTreatment` preset table (`standard`/`spalding`/`rough`/`lowRe`) and its precedence (explicit per-field type > per-patch override > case default) — one setting expands to per-field `BcKind`s at case-build time; the per-face kernel architecture of §4 never sees a preset, only what it expands to | `src/io/case.rs`, `src/blockgen.rs`, `src/field_setup.rs` | §29.1 |
| The consistency contract: the four (five, with `T`) per-field types on one wall patch must belong to one preset row — a mixed row is a §13.4 error naming the patch, the offending pair and the consistent completions; `-permissive` substitutes the row implied by the `nut` choice and says so | `src/field_setup.rs`, `src/io/contract.rs` | §29.1, §13.4 |
| `validate_low_re_wall_treatment`: a `lowRe` wall treatment is only valid under a turbulence model that itself integrates through the viscous sublayer — `LaunderSharmaKE` (§33) is the one model on that menu today, `kEpsilon`/`kOmega`/`kOmegaSST` still are not, so resolving `lowRe` under any of the latter three is a §13.4 error naming the menu and the alternative (`standard`); `-permissive` substitutes `standard` and says so. SPEC-LIT §32's own second finding — `cases/channelPeriodicFluxLowRe.jsonc` still diverged after its under-resolved side walls were corrected, because standard k-epsilon has no near-wall damping at all, not because the mesh was wrong | `src/io/case.rs`, `src/io/case_json.rs` | §32, §13.4 |
| The `grad(grad U)` route for the `E` term — the Gauss gradient of the ALREADY-COMPUTED cell velocity gradient (`RasCore::update_flow_derived`'s own `grad_u`), rather than a dedicated second-derivative stencil, boundary faces zero-gradient-extrapolated because `grad U` carries no boundary field of its own; SPEC-LIT §33.1 marks this *DESIGN* itself and states its cost (one more CSR gather, roughly three times a vector-gradient pass, paid once per outer iteration) | `src/turbulence.rs` (`ls_grad_grad_u_mag_sqr`), `cuda/turbulence.cu` (`turbLsGradGradUMagSqr`) | §33.1 |
| `MeshResolutionReport` — the §33.2 mesh check (worst wall-adjacent y+, cells globally below y+ 20) as a pure function of already-downloaded `k`/wall-distance/wall-face-owner arrays rather than a GPU kernel, so a driver can call it once at set-up without a new device dependency; counted GLOBALLY rather than per wall-normal column, a stated approximation (this crate carries no wall-normal-column topology) | `src/models/launder_sharma.rs` (`mesh_resolution_report`) | §33.2 |
| Reusing [`u_plus`]'s own exponential (Kader) blend for the thermal law `T+`, rather than the root-sum-square blend `epsilon`/`omega` use — the log branch is stated in terms of `u+` itself (§29.3), so sharing the blend makes the `Pr = Pr_t` identity exact rather than approximate | `src/wallfunctions.rs` | §29.3 |
| The thermal wall function rewrites the Robin triple as the `fr = 0` (fixedGradient) degenerate case, with `ref_grad` chosen so the total flux is exactly the analytic Jayatilleke `q_w` whatever `k_eff` the energy equation computed that face — the same lagged-coefficient convention every other wall quantity in this crate already runs at | `src/wallfunctions.rs`, `src/energy.rs` | §29.3 |
| The `C_3` default for buoyancy production | `src/turbulence.rs` | §17 |
| The inert-species choice for the sum-to-one constraint | `src/species.rs` | §19 |
| Geometric cell-set selection for sources | `src/sources.rs` | §18 |
| Zalesak limiter iteration count | `src/vof.rs` | §20.2 |
| `-permissive`, and the one-line-per-setting warning it prints | `src/io/contract.rs` | §13.4 |
| **Always** blending the two branches of the law of the wall — there is no `blended` switch, because there is no case in which the discontinuous form is wanted | `src/wallfunctions.rs`, `src/io/case.rs` | §6.4 |
| Evaluating the boundary conditions that read another field (`turbulentIntensity…`, `turbulentMixingLength…`, `totalPressure`, `pressureInletOutletVelocity`) once at setup, and refreshing only their value fraction from the flux thereafter | `src/field_setup.rs` | §4 |
| Defaulting `C_3` to the Henkes form, and spelling the override `C3Buoyancy` so it cannot collide with §6.1's dilatation `C_3` | `src/turbulence.rs`, `src/bin/buoyant.rs` | §17 |
| The `CoupledTurbulence` trait (`initialise`/`correct`/`nut`/`name`/`output_fields[_mut]`) - kept to exactly what a coupled driver's outer loop, writer seam and `.mcr` checkpoint need, and no more; a `dyn` dispatch is acceptable here (one virtual call per OUTER iteration) where it would not be in a standalone driver, because which model runs is a runtime fact read out of the case rather than known at the call site | `src/models/coupled.rs` | §30.2 |
| The LES wall-model preset mapping under `simulationType LES` - `standard`/`spalding` both read as `wernerWengleWallFunction`, `lowRe` as `nutLowReWallFunction` (`nu_t,w = 0`), and `rough` is a §13.4 error naming the two rather than an alias (a rough LES wall model is future work) | `src/io/case.rs` (`WallTreatment::les_nut_type`) | §30.1 |
| `ScalarBc::ThermalWallFunction { value }` - an explicit, per-patch `T_w` for `thermalWallFunction` on `T`, alongside the case-level `wallTreatment` auto-completion (which has no wall temperature of its own to write, only the neighbour cell's) - the same "explicit wall-function variant carries its own value" shape `TurbBc` already gives `nut`/`k`/`epsilon`/`omega` | `src/io/case_json.rs` | §29.1, §29.3 |
| The integrated wall-heat-flux report in `ofgpu-fire` - reading the SAME `(fr, ref_value, ref_grad)` Robin triple the energy matrix was assembled from, generically over every `wall`-kind patch, rather than re-deriving Jayatilleke or a plain Dirichlet flux externally per BC type | `src/bin/fire.rs` | §29.3, §30.3 |
| Keeping the STABLE branch of `G_b` in the `k` equation only, unless the case asks for it in `epsilon` as well | `cuda/turbulence.cu` | §17 |
| Selecting a source's cells geometrically — box, sphere, explicit list — and the `constant/fvSources` dictionary that expresses it | `src/sources.rs` | §18 |
| Refusing a source whose selection catches no cells | `src/sources.rs` | §18, §13.4 |
| Choosing the inert species by the largest volume-weighted mean when the case names none | `src/species.rs` | §19 |
| Writing `phi` at round-trip precision rather than at `writePrecision`, because its value carries a discrete constraint and not just a number | `src/io/fields.rs` | §5.1, §22 |
| Three iterations of the Zalesak limiter, and the GLOBAL bounds `[0, 1]` rather than the local-extremum ones | `cuda/vof.cu` | §20.2 |
| `eps = 1e-8/L` in the interface-normal normalisation, with `L` the cube root of the mean cell volume | `src/vof.rs` | §20.1 |
| A ninety-degree contact angle at every non-cyclic boundary — the one choice that adds no unstated physics, since §20 specifies no contact-angle model | `cuda/vof.cu` | §20.4 |
| Forming the compression term as `alpha_f(1 - alpha_f)` from the interpolated face value rather than interpolating the cell product, which would switch compression off exactly at a sharp interface | `cuda/vof.cu` | §20.1 |
| The plain interpolated cell gradient as the face interface normal, rather than that gradient with its normal component replaced by `snGrad(alpha)` — measured, and the replacement is seven times worse | `cuda/vof.cu` | §20.4 |
| Leaving the momentum predictor switchable, and taking `|U|` as the limiter sensor for `div(rhoPhi, U)` | `src/vof.rs` | §20.3, §5.4 |
| `rho_f` by linear interpolation of cell `rho`, `rho_infinity` from `TRef` at `p0` | `src/energy.rs` | §25.3 |
| `chi_r` radiant-fraction floor default (0.35, FDS practice) and the `LES` mixing-frequency substitute `C_EDM'` defaulted to the same `4.0` as the RANS constant until a case overrides it | `src/combustion.rs` | §27 |
| Combustion applied as its own operator-split pass over `Species`, not fused into its transport matrix — an EDM rate is a LOCAL field, not the uniform value every other `SourceSet` entry is | `src/combustion.rs` | §27, §18 |
| The fuel sink's backward-Euler closed form on the 0-D reaction ODE, positive for any `dt` | `src/combustion.rs` | §3.4, §27 |
| `s×s` face supersample lattice default `s = 16`, and `theta_min = 0.2` for cut-cell merging | `src/surface/cutcell.rs` | §24.2, §24.5 |
| Cut face centroid as the mean of interface sample midpoints, and its patch from the nearest surface triangle | `src/surface/cutcell.rs` | §24.3 |
| `ofgpu-fire`'s `Y_O2`/`Y_P` boundary conditions derived from patch kind (ambient `inletOutlet` on `open`, `fixedValue 0` on a fuel `inlet`) rather than named per case — only `Y_F` is a case-supplied field | `src/io/case_json.rs`, `src/bin/fire.rs` | §27, §19 |
| Species transport and combustion/radiation's segregated lag: read at the END of the PREVIOUS unit of work, registered before the CURRENT iteration's target-divergence/energy assembly — the same one-iteration lag `nu_t` and `T` already run at | `src/bin/fire.rs` | §25, §26, §27, §28 |
| The `.mcr` restart format itself (header, mesh hash, versioning, full-`f64` `phi`) | `src/restart.rs` | — |
| The cyclic-pair format itself — naming both sides of a pair plus the transform (`translate` only; `rotate` is a named §13.4 refusal, not a silent subset), rather than OpenFOAM's own separate `neighbourPatch`/`transform` dictionary entries on each side independently | `src/blockgen.rs` (`-cyclic x\|y\|z`, `BlockSpec::set_cyclic_axis`), `src/io/case_json.rs` (`mesh.cyclic[]`) | §31.1 |
| Face matching by nearest translated centroid, and the two invariants that make a mismatched pair loud instead of silently unconserving — every face matches exactly once (a bijection), and `Sf_a == -Sf_b` after the transform to a stated tolerance | `src/blockgen.rs` | §31.1 |
| The transient/algorithm contract — a case's `run.endTime` and `numerics.ddt`/`numerics.algorithm.kind` are checked as ONE combined setting (`is_transient_run`, `check_transient_algorithm_contract`) rather than three independently-valid ones, because that is exactly how `cases/burnerPlume.jsonc` reached step 20 as `Inf` with nothing having warned | `src/io/case.rs` | §31.3 |
| A `sources[]` `momentumSource` entry in JSONC — the one case a PERIODIC domain needs (no inlet to prescribe a mass flow from) reusing `crate::sources::SourceTerm::BodyForce`/`CellSelector::All` verbatim, rather than a second JSON copy of the whole `constant/fvSources` box/sphere/six-term-kind surface that format already has | `src/io/case_json.rs` (`JsonSource`) | §18, §31.1 |

## GPU plumbing and tooling — original

No external source. These have no counterpart in any CFD code we are aware of,
and they are a larger share of the value than their line counts suggest.

| File | What it is |
|---|---|
| `src/device.rs` | The cudarc wrapper: context, dedicated non-blocking stream, `DevBuf`, `KernelSet`, CUDA graph capture |
| `src/error.rs` | The error type |
| `src/types.rs` | `Vec3` / `Tensor`, `#[repr(C)]` mirrors of the device structs, with a layout test |
| `build.rs` | The MSVC/nvcc build glue: `vcvars64.bat` capture, CUBIN emission, `/Zc:preprocessor` |
| `src/mesh.rs`, `src/mesh/topology.rs` | The mesh types and the LDU→CSR inversion |
| `src/ldu.rs` | The matrix type and the LDU→CSR permutation for AMGX/cuDSS |
| `src/field.rs` | The device field types and `BcKind` |
| `src/pressure/mod.rs` | The `PressureBackend` trait and the measuring selector — hard filter, then accuracy check, then measured timing |
| `src/pressure/fft.rs`, `cuda/pressure.cu` | The cuFFT direct Poisson solver. Method: Swarztrauber, *SIAM Review* 19 (1977) 490; Press et al., *Numerical Recipes* §19.4 |
| `src/pressure/cartesian.rs` | Detecting whether a mesh is a uniform Cartesian block, which is what makes the FFT path applicable |
| `src/pressure/amgx.rs` | The AMGX backend. AMGX itself is BSD-3-Clause; see `../NOTICE` |
| `src/blockgen.rs` | The structured mesh generator |
| `src/bin/*` | Drivers and benchmarks |
| `src/bin/vof.rs` | The two-phase driver and its surge-front report |
| `../tools/*.py` | The isometric volume renderer and the HTML report builder |

## Case format interoperability — format

Written from the format as it appears in data files, so that existing pre- and
post-processing tools (ParaView, `foamToVTK`) work with ofgpu output. ofgpu
links against no part of OpenFOAM and contains no OpenFOAM source.

| File | What it reads or writes |
|---|---|
| `src/io/tokenizer.rs` | The ASCII tokeniser |
| `src/io/dict.rs` | Dictionaries, `#include`, `$var` lookup, `Switch`, dimensioned scalars |
| `src/io/polymesh.rs` | `constant/polyMesh`: points, faces, owner, neighbour, boundary |
| `src/io/fields.rs` | `volScalarField` / `volVectorField` files and their patch entries |
| `src/io/case.rs` | `system/fvSolution`, `system/fvSchemes`, `system/controlDict`, `constant/*` |
| `src/io/regex.rs` | The pattern matcher for `boundaryField` and `solvers` keys. Written from scratch — a backtracking matcher over the subset the format uses, anchored at both ends. No regex crate is a dependency |
| `src/io/contract.rs` | The unsupported-setting contract of SPEC-LIT §13.4 |
| `src/io/regex.rs` | The POSIX ERE matcher a quoted dictionary key needs. Written from IEEE Std 1003.1 §9.4, Kleene (1956) and Thompson (1968) |
| `src/io/contract.rs` | SPEC-LIT §13.4: a setting this solver cannot honour fails loudly, and `-permissive` is the one escape hatch. Ours by design |

The `FoamFile` banner these write is ofgpu's own, and states plainly that the
file is written by ofgpu in that format. It is not the upstream banner.

## Permissive references

| Source | Licence | Used for |
|---|---|---|
| `../reference/fds` | US Government work (NIST), public domain | The low-Mach formulation of Rehm & Baum (1978), and fire-specific models. Any file that adapts from it says so and acknowledges NIST |
| AMGX | BSD-3-Clause | Linked as the algebraic-multigrid backend |
| cudarc | MIT OR Apache-2.0 | Cargo dependency |

## Validation

`SPEC-LIT.md` §10 and §22 are the list, and the rule is absolute: **no test
compares against another CFD code.** Method of manufactured solutions,
analytical solutions, and published benchmark data — Ghia, Ghia & Shin (1982);
Moser, Kim & Mansour (1999); Driver & Seegmiller (1985); McCaffrey (1979);
Martin & Moyce (1952).

One note on the last of those. `ofgpu-vof -surge` and the `#[ignore]`d
`a_dam_break_surge_front_stays_under_the_ritter_bound` in `src/vof.rs` print
the surge front in Martin & Moyce's dimensionless variables so that a reader
holding the paper can lay one on the other. **Their table is deliberately not
transcribed into this repository**, because nobody working on it has the paper
to transcribe from, and a table of numbers attributed to a 1952 experiment and
actually recalled would be worse than no table. What those tests *assert* is
analytic: exact boundedness of `alpha`, exact conservation of the phase volume,
and a surge front that never outruns the Ritter (1892) shallow-water
characteristic speed `2 sqrt(g h0)`.

Agreement with another solver to 1e-13 is not evidence that this code is
correct. It is evidence of something else entirely, and it is exactly what this
project spent a phase removing.
