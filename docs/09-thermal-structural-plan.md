# Heat transfer and thermal deformation in `ofgpu` — a staged plan, §93 onwards

**Status:** adopted 2026-09-11 (Opus 5 synthesis of five independent Opus 5 audits, reviewed by Fable 5.1; §H added on review) (fluid-side heat transfer;
conduction and the conjugate interface; thermal deformation; numerical machinery; what an engineer
needs to defend a number). Nothing in this document has been implemented; no file in the repository
was changed to produce it.

**Licence rule observed.** No GPL/LGPL/AGPL source was consulted, opened, or reasoned from in
preparing this plan. Every formulation below comes from the published literature, from this
repository's own SPEC-LIT, or is marked *DESIGN* under §0 rule 3. Where permissively licensed prior
art is named it is named as **documentation** with its licence, and no code from it was read.

**Verification.** Every claim about the present state of the tree below was re-checked against the
tree, not carried over from the audits; where an audit was wrong, the correction is in §A.4. Every
DOI in §E was resolved against `api.crossref.org` in this session; the ones that could not be are
listed as unverified and nothing rests on them.

---

## A. Judging the five audits

### A.1 What is real (re-verified in the tree)

| Claim | Verified how |
|---|---|
| No solid mechanics anywhere | `grep -rniE "young\|poisson.?s? ratio\|von ?mises\|lame\|hooke"` over `rust/src` returns only Young's *contact angle* (`contact_angle.rs`), Young–Goldstein–Block (`marangoni.rs`) and the mesher's point displacement |
| No tensor field can be written | `src/io/output_types.rs:30` — `FieldValues` has exactly `Scalar` and `Vector`; `src/io/vtu.rs` writes `CellData` only (the one `PointData` block, line 463, is the `.vtp` particle writer) |
| Radiation is unreachable from every driver | `RadiationConfig` appears only in `radiation.rs`, `s2s.rs`, `field*.rs`, `xref.rs`; no driver constructs it; the `from_case` reader's only callers are its own tests |
| No participating media | no `src/participating.rs` exists, although SPEC-LIT line 13764 (§68.13) names it as a file that "still takes a constant `Gamma`" |
| Conjugate fluid is hard-coded laminar | `src/cht/flow.rs:947-949` — "Laminar: `nut` is zero on both meshes and never written" |
| A conjugate fluid case must be steady | `src/io/case_cht.rs:1199` refuses `has_fluid && !steady` by name |
| A conjugate region is a graded box | `ChtRegionMesh` → `build_region_mesh` (line 1309) → `blockgen::build_mesh` (line 1391); `deny_unknown_fields` leaves no other spelling |
| Gate 46-B does not exist | `grep -rn "46-B" --include=*.rs` returns nothing, though §46.7 specifies it |
| One global residual norm over a multi-region union | `solver::device_norm_factor` (line 1149) reduces `x_ref = mean(psi)` over `a.n_cells`, and `ConjugateHeat` solves one matrix over the concatenated mesh |
| Every property is a constant | `GasProperties` (energy.rs:276) is constant `cp`, `k`, `W`, `gamma`; no Sutherland, no JANAF, no `cp(T)` anywhere; `ChtMaterial` is `{rho, c, kappa}` |
| No viscous dissipation, no Mach diagnostic | `grep -i "viscous dissipation"` → one comment in a droplet test; `grep -i mach` → prose only |
| `ofgpu-cht` writes a CSV and nothing else | `src/bin/cht.rs:374 write_csv`; no `output` block, no VTU |
| The `psychro` refusal names an unreachable alternative | `psychro.rs:295` offers "a coil-surface saturated boundary condition"; `DcCase` (case_dc.rs:298-307) carries only `sc_t`, `barometric_pressure`, `virtual_temperature` |
| The machinery a stress solve needs exists | `fv::fvm_laplacian` + `_non_orth_correction` + `_skew_correction`, `fvc_grad_vector_scheme` (fv.rs:1951), `fvm_su/sp`, `Momentum::fill_component` (momentum.rs:948), `GpuLduMatrix`, PCG + multi-colour DIC, `ThermalRegion::cells()` as a contiguous range (cht.rs:433) |

### A.2 Duplicates, merged

* **Radiation is a library no case can reach** — lens 1 gap 2 and lens 5 gap 4 are the same gap; lens 2 gap 9 (*a conjugate interface cannot also radiate*) is its other half. Merged into one item and one section (§98); §47.2's cell-source route is the mechanism for the conjugate half, and §47.10 has to be amended, not extended.
* **Turbulent conjugate / Gate 7 unrun** — lens 2 gap 3 and lens 5 gap 5. One item, §100.
* **Constant properties** — lens 1 gap 4 (gas), lens 2 gap 7 (solid `k_s(T)` and the source), lens 5 gap 2 (both, plus `E`, `nu`, `alpha`). One item, §99: a `Property` that is a value *or* a curve, one carrier for all of them.
* **No observed order, no reported uncertainty** — lens 2 gap 5 (Gate 46-B and the interface order) and lens 5 gap 7 (GCI / ASME V&V 20). One item, §94: they are the same instrument at two ends.
* **Transient** — lens 2 gap 4 (no transient conjugate, Euler hard-coded), lens 2 gap 6 (no per-region clock), lens 4 gap 6 (no step controller). One staged item, §102.
* **Phase change** — lens 1 gap 9 (enthalpy–porosity in the fluid) and lens 2 gap 11 (in the solid) are one model with two hosts. Deferred together (§A.3).
* **Buoyancy validation** — lens 1 gap 3 (no buoyant wall treatment, no turbulent-Ra gate), lens 1 gap 5 (no plume gate), lens 5 gap 8 (de Vahl Davis stops at Ra 1e5). One item, §101: the model and the gates that would measure it.
* **The container for a second unknown** — lens 4 gap 4 (no region-restricted equation) and lens 3 gaps 3–4 (no tensor field, no mechanical material) are the same Stage 0. Lens 4 found the load-bearing half: without a restriction, a displacement equation must be assembled over fluid cells and pinned, which fills the Krylov space with identity rows and makes the residual meaningless.

### A.3 Dropped, or demoted out of the plan (with the reason)

1. **Mesh motion / deforming solids** (lens 3 gap 7, lens 2 gap 12). Both authors said it is out of proportion, and they are right. It is not a gap in this area; it is a solver-wide programme starting at the space conservation law. It survives here only as a **named refusal with a measurement** inside §95 (`max|u| / min cell size`, refused above a stated threshold), which is the honest form.
2. **Local thermal non-equilibrium porous media** (lens 1 gap 10, rated `nice` by its own author). A rack- and filter-bank feature for the data-centre product, not part of heat transfer or thermal deformation. Dropped from the stages.
3. **Coil / effectiveness-NTU and wall condensation** (lens 1 gaps 7, 8). Real, and the second exposes a genuine defect (a refusal naming an alternative no case can express — §A.5). But both are data-centre product work, not thermal deformation. Deferred; the *defect* is fixed cheaply in Stage 0 by deleting the unreachable clause or making it reachable.
4. **Participating-media radiation** (lens 1 gap 1, rated `must`, L). Demoted out of this plan's stages. It is the right model for the fire/plume driver and it is genuinely absent — but it serves `ofgpu-plume`, not the thermo-mechanical goal, and it is a quarter of work whose first customer is a case (`cases/plume.jsonc`) that has no published gate either. The right order is: gate the plume first (§101), then decide whether P1 closes the gap the gate measures. Recorded as the largest single deferred item.
5. **Block-coupled matrix as a *must*** (lens 4 gap 1). Demoted to a **refusal by name** in §95 and a measurement in Gate 95-F. See §A.4.
6. **AMG with a near-null space** (lens 4 gap 2). Real, and the mesh-independence evidence in §73.5 is this repository's own. But release 1's elasticity matrix is constant, assembled once and preconditioned once, so the setup cost amortises over the whole run; whether DIC-PCG is adequate is a *measurement*, not an assumption, and Gate 95-F produces it. Deferred behind that measurement.
7. **Non-conformal (AMI/mortar) interfaces** (lens 2 gap 2, L). Real and correctly ranked by its lens — but it becomes binding only after mesh import, and the current refusal (`ThermalMesh::couple` refuses a face-count mismatch and an unmatched centroid, by name) is honest in the meantime. Deferred to after Stage 2, with the design (a host-built CSR of overlap weights, gather-shaped on the device) already fixed by §47.4.
8. **Distributed CHT** (lens 2 gap 10, lens 4 gap 8). Latent: nothing in the tree calls `distsolve` from a physics driver. Deferred — except the cheap half, which is a **refusal now** for a decomposition that cuts a conjugate interface, because §47.2's conservation proof has a stated precondition ("ONE launch writing both sides") that a cut violates and nothing says so today.
9. **Newmark / HHT / generalized-alpha** (lens 4 gap 7). Correctly rated `should` by its lens and not needed by a quasi-static release. It becomes a named refusal in §95: a `ddtScheme` on a displacement equation is an error, naming inertia as the thing that is not modelled.

### A.4 Where the lenses disagree, and how I ruled

1. **Segregated FV versus a block matrix for elasticity.** Lens 3 recommends segregated FV on the existing scalar LDU and says block coupling should be *refused by name*, because a 3×3 coefficient per face breaks §1's storage exactly as MPFA breaks §46.4. Lens 4 makes the block matrix its first `must`, on Cardiff et al. (2016)'s measured order-of-magnitude reduction in outer iterations.
   **Ruling: lens 3, with lens 4's objection converted into the gate.** Release 1 is segregated, because it reuses an operator stack that is already gated, adds no second storage format and no second solver path, and can ship in a quarter. Lens 4's cost is real but it is a *speed* argument, and this repository's rule is that a speed argument is measured, not asserted (§46.4 is the precedent: an asserted estimate had to be corrected). Gate 95-F therefore prints the predicted contraction `1/(2(1-nu))` beside the observed outer-iteration count at `nu` = 0.2/0.3/0.45. If the measurement says the segregated loop is unaffordable at engineering Poisson ratios, the block matrix is the next section and lens 4's design is the one to build. Until then it is refused by name.
2. **How much does the first stress release need?** Lens 3 scopes release 1 at a quarter including NAFEMS LE1/LE10/LE11. Lens 2 and lens 4 imply that is not reachable, because the case format can only build graded boxes and because there is no region-restricted equation.
   **Ruling: both are right about different halves, and the resolution is cheap.** `src/bin/validate.rs` already imports `polymesh::{read_poly_mesh, build_host_mesh}` and uses them for a shear fixture, so a **gate** can run on a mesh the §92 automesher wrote *today*, with no case-format work. What cannot be done today is a **user** running their own part. So: the gate suite is Stage 1 (no mesh-import dependency), and the case-format import is Stage 2 (user reach). No single lens saw this; lens 3's gate plan would otherwise have been blocked behind lens 2's gap 1 for no reason.
3. **Is radiation wiring or participating media the higher priority?** Lens 1 ranks participating media first (L, `must`) and the wiring second; lens 5 ranks the wiring `must` and never mentions participating media.
   **Ruling: lens 5.** Wiring a fully gated model into a driver is weeks; building a new RTE solver is a quarter, and the case that most needs it (the plume) has no gate to measure the improvement against. Value per week is not close.
4. **Is the global residual norm a defect or a nuance?** Only lens 4 raised it.
   **Ruling: it is a defect, and it is the cheapest item on this list.** It is §13.4's own failure mode — a plausible wrong answer, silently produced — living inside the convergence test rather than in a reader, and it becomes unavoidable the moment a displacement unknown (metres, ~1e-6) shares a union with a temperature (kelvin, ~1e2). Promoted into Stage 0.
5. **Lens 5's die-stack property numbers are wrong.** Lens 5 argues from "a constant 148/120/30" for `cases/dieStack.cht.jsonc`. The case actually carries `"kappa": [120.0, 120.0, 30.0]` for the die — a §46.3 Wiener pair for a layered stack, not bulk silicon — and 148 is `cases/quMudawar.cht.jsonc`'s silicon. Lens 4 flagged its own version of this number as unverified and was right to. **The argument survives the correction** (k_Si falls by roughly a third from 300 K to 650 K, Glassbrenner & Slack 1964), but the gate must quote the case, not the folklore.

### A.5 In-repo defects observed while synthesising (not gaps, and not fixed here)

1. SPEC-LIT §68.13 (line 13764) names `participating.rs` as an existing file that "still takes a constant `Gamma`". No such file exists. §80's citation audit checks section numbers, not file names, so nothing caught it.
2. `psychro::refuse_condensation` (psychro.rs:295) names "a coil-surface saturated boundary condition" as an available alternative. `DcCase` is `deny_unknown_fields` and carries no such entry. A refusal that redirects to something no case can say is the same defect class §13.4.1's pair tests exist to catch.
3. `cases/dieStack.cht.jsonc`'s header says the fluid side is one "no case format reaches yet" and that `ofgpu-cht` "refuses a `"kind": "fluid"` region by name". `case_cht.rs:614` accepts `"fluid"`, and two shipped cases use it. Stale prose in a shipped case.
4. §46.7 specifies Gate 46-B ("must converge at second order") and it is implemented nowhere — which means nothing in §46/§47/§59 measures an observed order of accuracy at all.

---

## B. The ranked gap list

Rank is for the stated goal: **what heat transfer still lacks, and giving the solver thermal
deformation.** Size: S under a week, M a few weeks, L a quarter.

| # | Gap | Size | Why here |
|---|---|---|---|
| 1 | **No thermo-elastic solid.** No displacement, no strain, no stress; no `E`, `nu`, `alpha` anywhere in the tree or in any case schema. §46 produces `T` in a solid and the chain stops. | L | The headline, and it is the whole of half the stated goal. `cases/dieStack.cht.jsonc` computes a 350 K rise across a solder joint whose failure mode is thermal fatigue and cannot say whether the joint yields. Every electronics, weld, furnace, pressure-vessel and turbine question ends in a stress, and the export to a second code is exactly where this project's provenance and gate culture stop applying. |
| 2 | **No container for a second unknown, and no tensor a run can write.** One equation over the union is all §47.4 can express; the residual norm is one number over the union; `FieldValues` has two arms and the `.vtu` writer has no `PointData`. | M | Stage 0 for everything above it. Without the restriction a displacement equation must be assembled over fluid cells and pinned, filling the Krylov space with identity rows. Without the per-region norm, a 200:1 conductivity union reports "converged" set by the stiff region while the soft one is 1e-4 of its own scale. Without a tensor arm and point data, a stress solve cannot emit von Mises or a deformed shape — the two things a user acts on. |
| 3 | **Nothing measures an observed order of accuracy, and no gate reports a discretisation uncertainty.** Gate 46-B is specified and unwritten; §60.5/§79.12 read three meshes by hand and quote a percentage change. | S | The house's own rule turned on itself, and the cheapest item on the list. It is also the precondition for believing any stress number: FV stress is one order below FV displacement, so a stress gate without a measured order is not evidence. It converts Gate 5's "MISSES by 7.12 %" from an argument into a measurement. |
| 4 | **A solid face can only be held at a temperature or a flux.** No `h (T - T_inf)`, no `eps sigma (T^4 - T_env^4)`; §49/§50's gated radiation is reachable from no driver and no case; a conjugate face cannot also radiate. | M | This is how every electronics and enclosure case is actually posed — you do not mesh the room. Today `dieStack` must hold a cold plate at exactly 300 K and everything else is adiabatic. Meanwhile the single largest piece of finished, gated physics in the repository (eight radiation gates) cannot be applied to one flow: §51.1 says so in its own words. |
| 5 | **A conjugate region can only be a graded box.** `ChtRegionMesh` → `blockgen`; both shipped conjugate cases are box stacks. `ThermalMesh::build` is already mesh-agnostic. | M | The library is ready and the door is missing. Every geometry this module is named for — a wire-bonded die, a finned sink, a motor slot, a coolant passage — is not nine boxes, so the only conjugate answers the solver can produce are ones whose 1-D closed form is already known. Good verification posture, complete engineering dead end. |
| 6 | **Every property and every source is a constant.** No `k(T)`, `cp(T)`, `mu(T)`; no `alpha(T)`, `E(T)`; one scalar `q'''` per region with no space or time variation; no `fvm_sp` for a temperature-dependent sink. | M | The repository has measured this itself: §79.12's Disclosure 2 calls the property choice the largest single uncertainty in the one gate this area passes. Over 300–650 K silicon's conductivity falls by about a third. A thermal-stress number with a constant `alpha` and no `T_ref` discipline is not a number anyone signs. A uniform constant source also removes the leakage-power feedback that thermal-runaway analysis exists to find, and forbids a duty cycle. |
| 7 | **Every conjugate fluid case is laminar.** `nut` is allocated zero and never written; §47.6's wall-function conductance is implemented on both host and device and reached by no case; Gate 7 (Flageul) is still unrun. | M | Essentially all industrial conjugate heat transfer is turbulent. Gate 6 passes at Re ~ 140 in a micro-channel; a heat sink under a fan is at Re ~ 1e4 and cannot be posed at all. It is also the largest *unmeasured* claim in the area — §59.9 declines to claim anything about a turbulent conjugate interface. |
| 8 | **No transient conjugate path, no step controller, no per-region clock.** A conjugate fluid case must be steady; the solid-only transient is hard-coded Euler with no `ddtScheme`; §13.3's variable-dt BDF2 coefficients are implemented and nothing varies the step. | L | The questions users bring to a conjugate solver are mostly transient (time to 125 °C after a fan fails; a duty cycle; a wall assembly's response), and thermal *fatigue* — the reason to want gap 1 — is a duty cycle by definition. A fixed dt over a start-up spike and a soak is either unresolved or unaffordable; without a per-region clock a copper spreader costs 1e3–1e5 fluid steps per solid time constant. |
| 9 | **Natural convection is gated only to Ra 1e5 and only laminar**, while `ofgpu-buoyant` and `ofgpu-datacentre` run rooms at Ra 1e10–1e12 on a wall function whose velocity scale is the shear one, with nothing saying so; and the plume §10/§22 name a gate for is ungated. | M | The two drivers a room, enclosure or fire engineer reaches for have no buoyant thermal gate at all, and §32's careful two-verdict machinery has no buoyant counterpart. A shear log law on a buoyancy-driven wall is wrong by tens of percent at high Ra, and today the run will not even report that it has left the regime its wall model was gated in. |
| 10 | **Viscous dissipation is absent from the energy equation and nothing refuses it.** | S | §38 ships generalised-Newtonian rheology and §79 ships a liquid micro-channel — the two regimes where self-heating *is* the design question, because viscosity falls as the fluid heats. Three lines of kernel on operators that already exist, and a closed-form gate. |
| 11 | **The gate culture stops at `ofgpu-validate`: a case run leaves no defensible record.** `ofgpu-cht` prints a banner and optionally a cell-centre CSV — no VTU, no `output` block, no report. | M | "How a run is reported so an engineer can defend the number" is the framing, and today the answer is terminal scrollback. §69.9 names this itself. The hard half — a registry a verdict cannot escape — is already built; it is pointed only at the validation binary. |
| 12 | **The cheap refusals the culture demands and does not have** (bundle, S each): no Mach number is computed or refused by `ofgpu-lowmach`, whose specification's first line states the regime; nothing refuses a decomposition that cuts a conjugate interface, though §47.2's conservation proof has a precondition a cut violates; `psychro::refuse_condensation` redirects to a condition no case can express. | S | Each is one reduction or one contract call, and each is a place where this project's own discipline — never substitute silently, never name an alternative that does not exist — is not yet applied to itself. |

---

## C. The staged plan

Section numbers run from **§93**, the first vacant address (§0 rule 6: numbers are addresses, not an
ordering, and a number is allocated only when the section is written). Each stage ends in something
measured and registered through `Checks::report(GateReport{..})` so §69's three relations hold.

### Stage 0 — What must be true before any of it (4 weeks)

Nothing in stages 1–6 is safe until the container can hold a second unknown, the convergence test
can tell two regions apart, and a gate can say how accurate it is.

#### §93 — An equation that lives on a region: the restricted assembly, the per-region residual, and the tensor a run can write

* **Modules.** `src/cht.rs` (a `RegionView` built from the `cell_offset/n_cells`,
  `internal_face_offset/n_internal_faces`, `boundary_face_offset/n_boundary_faces` that
  `ThermalRegion` *already stores and nothing reads*); `src/solver.rs` (`device_norm_factor` gains a
  per-block variant; `check_convergence` gains the per-block test); `src/io/output_types.rs` (a
  `Tensor` arm on `FieldValues`); `src/io/vtu.rs` (a `PointData` path, the cell-to-point
  interpolation, and the 6→9 component expansion VTK wants); `src/reference.rs` (the scatter-shaped
  mirror of any new operator).
* **Equations owned.** None new. This section owns a *restriction*: for region `r`,
  `A_r psi_r = b_r` assembled over `cells(r)` and `internal_faces(r)` only, with faces on
  `PatchKind::Interface` entering as boundary faces of `r`. And the §8.4 normalisation restated per
  block: `norm_r = sum_{P in r} |A psi - A x_ref,r| + sum_{P in r} |b - A x_ref,r| + eps`,
  `x_ref,r = mean_r(psi)`.
* **Gate 93-A (bitwise, DESIGN).** The restricted assembly of region `r` on a two-region
  concatenated mesh is **bit-for-bit** the assembly of that region alone as a single-region mesh:
  every `diag`, `upper`, `lower`, `source` entry equal in every bit, and the converged field equal in
  every bit over a multi-iteration run. This is §59.9's own proof shape ("bitwise unmoved"), which is
  why it is the right acceptance test.
* **Gate 93-B (published, two-material slab).** Silicon on mould compound, k ratio 200:1, against the
  1-D series solution (Carslaw & Jaeger ch. I — the same closed form §47.12 Gate 3 already uses).
  Requirement: with a per-region tolerance of 1e-8, the low-k layer's cell temperatures match the
  closed form to 1e-6 relative. **And the gap is measured, not asserted:** the same case is run with
  the global norm at the same nominal tolerance and the low-k region's own residual is printed beside
  it. The predicted result is that the global-norm run reports converged at 1e-6 while the soft
  region sits near 1e-4 of its own scale; if the measurement says otherwise, this finding is
  withdrawn in the section text.
* **Refusal.** A run over a union may not print `converged` unless **every** region's own residual
  met the tolerance; a case that names one `residualControl` for a union gets the per-region
  tolerance derived and disclosed by name in §13.4.2's start-up block, listing each region and the
  ratio of its row scale to the largest.
* **Also in this section** (ranked gap 12): `GasState` computes `M = |u| / sqrt(gamma R_s T)` (it
  already holds every term) and `ofgpu-lowmach` prints max and volume-mean `M` in its banner and
  every write line, refusing above a stated threshold or warning by name under `-permissive`;
  `decompose.rs` refuses a partition that cuts a `PatchKind::Interface` face, naming §47.2
  consequence 2 as the precondition it would violate; `psychro::refuse_condensation` stops naming a
  boundary condition no case can express.

#### §94 — Observed order and reported uncertainty

* **Modules.** New `src/vv.rs` (pure host arithmetic — observed order, Richardson extrapolation,
  GCI); an `uncertainty` field on `GateReport` in `src/bin/validate.rs`; Gate 46-B itself in
  `src/cht/tests.rs`.
* **Equations owned.** Observed order from three solutions at ratio `r`:
  `p = ln((f_3 - f_2)/(f_2 - f_1)) / ln r` (Roache 1994; Celik et al. 2008 in the fixed-point form
  for non-integer `r`); Richardson value `f_ext = f_1 + (f_1 - f_2)/(r^p - 1)`;
  `GCI_fine = Fs |f_1 - f_ext| / |f_1|` with `Fs = 1.25`; the least-squares estimator of Eça &
  Hoekstra (2014) when `p` is outside the asymptotic range or the sequence is oscillatory; and the
  validation comparison `E = S - D`, `u_val = sqrt(u_num^2 + u_input^2 + u_D^2)` (ASME V&V 20's
  framing, as restated in the open literature).
* **Gate 94-A = Gate 46-B, as §46.7 already specifies it.** A manufactured solution for
  `div(K grad T) = -f` on the affine-reduced anisotropic problem at `k_x : k_y : k_z = 1 : 10 : 100`,
  three meshes at `r = 2`. Requirement: observed `p` in `[1.9, 2.1]`.
* **Gate 94-B (the order nobody has measured).** The same manufactured solution across a conjugate
  interface with a 100:1 conductivity jump and a non-zero `R_c`, on a mesh sheared so the interface
  sits at 20° to the face normal. §47.3 suppresses the non-orthogonal correction on interface faces
  outright, so the operator there is formally first order and the 5° pairing refusal
  (`PairingTolerances::non_orth = 3.8e-3`) is a number chosen by argument. Requirement: `p` is
  **reported**; the gate fails if `p < 0.9` or if it is not reported. This is the measurement that
  tells us what accepting a 4.9° interface costs, and it must exist before Stage 2 lets real meshes
  in.
* **Gate 94-C (retrofit).** §60.5's and §79.12's existing three-mesh sequences are fed through
  `vv.rs` and re-reported: Gate 5's `-7.12 %` becomes `E = -7.12 % against u_val = ± X %`.
* **Refusal.** A `GateReport` from a multi-mesh gate that carries no `uncertainty` fails to register;
  a single-mesh gate must declare itself as such by name. This extends §69's own rule — the run
  cannot print a verdict it did not register — to "cannot print a verdict whose confidence it did not
  measure".

---

### Stage 1 — Thermal deformation, release 1 (12 weeks) — THE HEADLINE

Detailed in §D. Two sections.

#### §95 — The thermo-elastic solid

* **Modules.** New `src/solid/{mod.rs, displacement.rs, bc.rs, thermal.rs, stress.rs}`;
  `cuda/solid.cu` (`solidDivSigmaExp`, `solidThermalSource`, `solidBoundaryTraction`, `solidStress`,
  `solidVonMises`, `solidPrincipal`); the scatter-shaped mirrors in `src/reference.rs`; one new row
  in `rust/PROVENANCE.md`.
* **Equations owned.**
  * quasi-static equilibrium `div(sigma) + rho_s g = 0`;
  * small strain `eps = 1/2 (grad u + grad u^T)`;
  * Duhamel–Neumann `sigma = 2 mu eps + lambda tr(eps) I - (3 lambda + 2 mu) alpha (T - T_ref) I`,
    with `mu = E/(2(1+nu))`, `lambda = E nu/((1+nu)(1-2nu))`;
  * the segregated FV split (Demirdžić & Muzaferija 1994/1995; Jasak & Weller 2000):
    `sum_f (2 mu + lambda) grad(u).Sf` implicit,
    `sum_f [mu grad(u)^T + lambda tr(grad u) I - (mu + lambda) grad(u)].Sf` deferred,
    `- sum_f (3 lambda + 2 mu) alpha (T_f - T_ref) Sf` the thermal load (Demirdžić & Martinović 1993);
  * traction continuity at a multi-material face instead of interpolating a discontinuous
    `(3 lambda + 2 mu) alpha` (Tuković, Ivanković & Karač 2013) — the mechanical partner of §46.2;
  * von Mises `sqrt(3/2 s:s)`; principal stresses as the roots of the characteristic cubic.
* **Gates.** 95-A cantilever (order + closed form), 95-B free expansion (round-off), 95-C patch test
  (exact), 95-D thick cylinder with a radial temperature gradient (the gate that chains §46's
  conduction to the stress), 95-E bimetallic strip (Timoshenko 1925, DOI verified), 95-F the outer
  loop's measured contraction, 95-G NAFEMS LE1/LE10/LE11 **only when the primary is in hand**. Full
  specification in §D.4.
* **Refusals.** §D.5.

#### §96 — What a thermo-elastic case says, the refusal list, and the pair tests

* **Modules.** `src/io/case_cht.rs` (a `mechanics` block per solid region, and a `stress` run mode),
  `docs/schema/case-1.json`, `src/solid/mod.rs` (`MechanicalMaterial::validate`, the Lamé
  conversion), and the `output` block finally reaching `ofgpu-cht` through `common::output_plan` /
  `build_writers`.
* **Gate 96-A (pair tests, §13.4.1 shape).** Ten pairs, byte-identical but for one entry, each
  **required** to move the answer and to fail by name if it does not: `alpha`, `E`, `nu`, `T_ref`, a
  traction value, a fixed-displacement value, a symmetry plane versus a free surface, the mechanical
  solver tolerance, the outer-loop relaxation, and the multi-material interface treatment.
* **Refusal.** `E <= 0`; `nu` outside `(-1, 0.5)`; `alpha < 0`; `alpha` without `T_ref`; a mechanics
  block on a fluid region; a `ddtScheme` on a displacement equation; and the near-incompressible
  refusal of §D.5, with the number in it.

---

### Stage 2 — Reach: a real mesh, and a face that exchanges heat with something not meshed (6 weeks)

#### §97 — The imported region

* **Modules.** `src/io/case_cht.rs` (`ChtRegionMesh` becomes an untagged enum
  `Block{..} | PolyMesh{path} | Msh{path}`, with `build_region_mesh` dispatching);
  `src/io/polymesh.rs` and `src/io/msh.rs` unchanged — they already produce a `HostMesh`, and
  `ThermalMesh::build` already takes `RegionInput{name, kind, mesh: &HostMesh}`.
* **Scope, deliberately small.** One region = one mesh, patches by name. Zone splitting of a single
  imported mesh into regions, and a faceZone-derived interface, are **refused by name** here and are
  the next section when they are built. This keeps Stage 2 at weeks: a bimetallic strip, an LE11
  taper and a real heat-sink fin are each one mesh per region, and §47.4's centroid-hash pairing
  already couples two separately imported conformal meshes.
* **Gate 97-A.** A block case rebuilt as an imported polyMesh of the same cells gives a field
  **bit-for-bit** identical to the block-generated run. `cases/dieStack.cht.jsonc` is the subject; the
  closed form it already matches to 1e-8 is the second check.
* **Refusal.** A zone-split request; a non-conformal pairing (already refused, now with the AMI
  design named); a mesh whose region is not a contiguous ascending cell range after concatenation.

#### §98 — A face that exchanges heat with something not meshed

* **Modules.** `src/io/case_cht.rs` and `src/energy.rs` (two new `T` conditions as §4 Robin triples);
  `src/cht.rs` (the cell-source route for a radiating conjugate face, §47.2 consequence 3);
  `src/radiation.rs` + `src/s2s.rs` (the driver-facing construction that `RadiationConfig::from_case`
  has been waiting for); `src/cht/flow.rs` and `src/bin/buoyant.rs` (the `S2s::update` call in the
  outer loop, and its stated cadence); `src/solver.rs::matrix_is_symmetric` (§48.3's paired-coefficient
  check was extended *for* this case: with radiation on one side of an interface the two coefficients
  stop being equal, so it should now fire and select an asymmetric solver).
* **Equations owned.**
  * convection to an unmeshed ambient, `-k dT/dn = h (T_b - T_inf)`, as `(fr, refValue, refGrad)`;
  * grey radiation to a surround, `q = eps sigma (T_b^4 - T_env^4)`, Newton-linearised as
    `h_r = eps sigma (T_b^2 + T_env^2)(T_b + T_env)`, with the linearisation residual reported;
  * `h_total = h_conv + h_r`, which is what a datasheet quotes;
  * the radiating conjugate face: the net radiative flux enters the **cell** source as `fvm_su` with
    `q_r |Sf| / V_P`, never `refGrad` (which `snGrad` weights by `(1-fr)` and therefore
    under-delivers — §47.11 already has a test for that trap);
  * `h` from Churchill & Chu (1975) when the case names the correlation instead of a number, with a
    refusal when the case leaves the correlation's stated range.
* **Gate 98-A (closed form).** The analytic straight fin, `q = sqrt(h P k A) theta_b tanh(mL)`
  (Carslaw & Jaeger), within 0.5 % on the finest of three meshes with the §94 GCI beside it.
* **Gate 98-B (closed form).** A slab radiating to a large surround: the steady balance
  `eps sigma (T_b^4 - T_env^4) = k (T_i - T_b)/d` reproduced to 1e-10, and the linearisation shown to
  converge quadratically in the outer loop.
* **Gate 98-C (coupled, on a live flow).** A heated wall in an enclosure with a running flow: the
  split of its heat into convective and radiative parts against the grey two-surface closed forms
  **already coded and gated** in `src/s2s.rs` (`parallel_plate_flux`, concentric), and the enclosure
  power balance `sum_i A_i q_r,i = 0` to 1e-10 end to end. This is the gate §50.12 says has never
  existed, and it is what makes `radiationRelaxation` a measured setting rather than an untested one.
* **Refusal.** Specular reflection, non-grey bands and a non-zero `absorptionCoefficient` stay refused
  by name (§50.9) — and §47.10's sentence that a conjugate face cannot radiate is **amended**, not
  extended, because this section builds the route it names.

---

### Stage 3 — Properties that are functions, and sources that vary (4 weeks)

#### §99 — `k(T)`, `cp(T)`, `mu(T)`, `alpha(T)`, `E(T)`, and `q'''(x, t, T)`

* **Modules.** New `src/properties.rs` (`Property = Constant | Table | Polynomial`, with a device
  evaluator); `src/energy.rs` (`GasProperties` becomes a transport model; `update_k_eff` and
  `set_thermal_wall` read it per face); `src/cht.rs` (`Conduction::build` splits into a setup part and
  a per-iteration rebuild; `assemble` gains the `fvm_sp` call §46.1's own term table already
  specifies; `run_case` gains a real outer loop with a convergence criterion and a refusal when it
  stalls); `src/io/case_cht.rs` (a scalar *or* a curve in the same entry; a zone map or box list for a
  source; an optional time table).
* **Equations owned.** Sutherland `mu(T) = mu_ref (T/T_ref)^{3/2} (T_ref + S)/(T + S)`; the
  NASA/JANAF 7-coefficient `cp(T)/R`; the harmonic face conductivity of (S46.2) evaluated on the
  current iterate; Patankar's source linearisation `S = S_C + S_P T`, `S_P <= 0`, which is exactly the
  sign convention `fvm_sp` already encodes; the Kirchhoff transform `psi = int k(T) dT`, under which
  steady conduction with `k(T)` has the same linear solution the constant-`k` case has.
* **Gate 99-A (exact).** The Kirchhoff-transform slab with `k(T)` linear in `T`: the transformed
  solution reproduced to 1e-12 relative, the untransformed one to the discretisation error §94
  reports.
* **Gate 99-B (published table).** `k_Si(T)` over 300–600 K against Glassbrenner & Slack (1964) within
  the source's own stated uncertainty; `mu_air(T)` and `k_air(T)` against Kadoya, Matsunaga &
  Nagashima (1985) over 250–1000 K within 1 %.
* **Gate 99-C (closed form, viscous dissipation).** `Phi = 2 mu_eff S:S - (2/3) mu_eff (div u)^2`
  registered into §18's registry as an unconditional heat source (`Phi >= 0`, so it needs no
  splitting), with the RAS statement of which part is resolved and which is the `rho epsilon` the
  turbulence model already carries, so the two are not double-counted. Gated against Brinkman (1951):
  plane Poiseuille flow at a stated Brinkman number has a closed-form temperature profile, and the
  adiabatic-wall case falls out of the same form. Amends §26.1's term list.
* **Gate 99-D (retrofit, honest).** Gate 6 (Qu & Mudawar) re-run with `mu(T)` live, and the movement
  reported — the repository has already estimated what a constant water viscosity costs in §79.12's
  Disclosure 2, and this replaces the estimate with a measurement.
* **Refusal.** Evaluating a table or a polynomial outside its stated range is an error naming the
  range — the precedent is already in the tree (`models/transition.rs` floors `Re_theta` so a fitted
  polynomial can never be evaluated outside its fit). A `q'''(T)` whose `S_P > 0` is refused by name
  (it would break diagonal dominance).

---

### Stage 4 — The two regimes with no gate (6 weeks)

#### §100 — The turbulent conjugate interface

* **Modules.** `src/cht/flow.rs` (instantiate a model from `crate::turbulence`, concatenate `nut` onto
  the thermal mesh through (S59.3)'s existing mask, and delete the zero allocations at 947–949);
  `src/cht.rs::ConjugateInterfaces::update` and `cuda/cht.cu` (select `C_A` per face by the interface
  face's own `nut` patch type, per §15.5, instead of the static `b_cond`); `src/io/case_cht.rs` (a
  `turbulence` block under `numerics.flow`, and inlet `k`/`epsilon` or an intensity);
  `src/energy.rs::attach_conjugate` (narrow §59.6's refusal to the cases the new branch does not
  honour).
* **Equations owned.** None new: §47.6's `C_A = rho c_p u_tau / T+` with Jayatilleke's `T+` is already
  implemented in `wallfunctions.rs:1703` and `cuda/wallfunctions.cu:1054` and gated in both limits by
  §47.12's Gate 2. This section is the selection and the wiring, plus the statement of which `C_A` a
  face takes and why.
* **Gate 100-A = Gate 7, finally run.** Flageul, Benhamadouche, Lamballais & Laurence (2015), DNS of a
  channel at `Re_tau = 150`, `Pr = 0.71`, four thermal wall treatments; open access at
  hal.science/hal-01321586v1. Verdict in §32.4's discipline — a RANS model against a DNS is a band
  statement, not an agreement claim, and the section must say which quantities are predicted and which
  are handed in. §47.12's own warning stands: do not gate a RANS model on the temperature-variance
  dissipation discontinuity.
* **Gate 100-B (second leg, milder).** Tiselj et al. (2001), wall-temperature-fluctuation statistics
  for the same configuration class.
* **Refusal.** A wall-function fluid side whose `y+` leaves the range §29.1's preset was gated in; a
  conjugate interface whose two sides disagree about whether the fluid face is resolved or modelled.

#### §101 — The buoyant wall, and the Rayleigh numbers the drivers actually run at

* **Modules.** `src/wallfunctions.rs` and `cuda/wallfunctions.cu` (a buoyant velocity scale
  `u_* = (g beta q_w nu / (rho c_p))^{1/4}` beside `y_plus_of`/`t_plus`); §29.1's `wallTreatment`
  preset table and its consistency contract; `src/bin/validate.rs`; `cases/ampofoCavity.jsonc`.
* **Equations owned.** Hölling & Herwig's (2005) asymptotic near-wall temperature law on the buoyant
  scale, and the Nusselt correlations Balaji, Hölling & Herwig (2007) build from it; Churchill & Chu
  (1975) as the correlation band, used the way Gnielinski is used in §32.4.
* **Gate 101-A (cheap, first).** de Vahl Davis (1983) at **Ra = 1e6** — the fourth row of the table
  Gate 59-A already quotes from and stops short of (`Nu = 8.800`), within 1 % with the §94 GCI.
* **Gate 101-B (the measurement).** Ampofo & Karayiannis (2003), 0.75 m air-filled square cavity at
  `Ra = 1.58e9`: local and mean Nusselt number and wall heat flux, band verdict against the
  experiment's own stated accuracy. Betts & Bokhari (2000)'s tall cavity is the second leg.
* **Gate 101-C (the plume, with its prediction stated in advance).** McCaffrey's three-regime
  centreline correlations for excess temperature and velocity in `z/Q^{2/5}`, against a reconfigured
  `cases/plume.jsonc` with a declared heat release rate. **The section must state before the run that
  a non-radiating solver is expected to sit high**, because McCaffrey's flames are real fires and
  radiative loss is 20–40 % of `Q` — which is what makes this a measurement rather than a tuning
  exercise, and what would make the case for participating media if it is ever built. The report
  identifier must be confirmed against the NIST catalogue first (§F.4).
* **Refusal (the important half).** A per-face `Gr_y / Re_y^2` is computed from data already on the
  device, and a wall face whose buoyancy-to-shear ratio is large while a shear wall function is
  selected is **named** — refused, or warned by name under `-permissive`. Today the substitution is
  silent, which is the exact defect class this project keeps removing.

---

### Stage 5 — Time (8 weeks)

#### §102 — Time in a conjugate run: the scheme a case can name, the per-region clock, the controller, and the duty cycle a stress solve rides

* **Modules.** `src/io/case_cht.rs` (a `ddtScheme` entry; narrowing the `has_fluid && !steady`
  refusal); `src/cht.rs::run_case` (take the scheme through `TimeState::coeffs` instead of hard-coding
  `DdtCoeffs{1/dt, -1/dt, 0}`); `src/energy.rs` (the `ConjugateEnergy` blend gains one more per-cell
  mask, following (S59.3) exactly, so the time weight is per region); `src/cht/flow.rs` (the transient
  outer loop, the old-time levels, the sub-cycling and the coupling interval); `src/timescheme.rs` (a
  `StepController` beside `Lts`), fed by one new `device_sum_mag` over the predictor difference in
  `src/solver.rs`.
* **Equations owned.** The variable-dt BDF2 already implemented in `timescheme.rs` (nothing to add);
  the BDF2 local-error estimate against the same-order explicit predictor, which needs no state beyond
  the `f0`/`f00` a field already carries (Kay, Gresho, Griffiths & Silvester 2010); the PI /
  digital-filter step controller (Söderlind 2003); the quasi-steady solid as `DdtCoeffs::ZERO` **per
  region** (§46.1's own control flag, made local); Errera & Chemin's (2013) coupling-coefficient
  stability condition when the two regions advance on different clocks; and Verstraete & Scholl's
  (2016) numerical Biot number printed per interface as the diagnostic.
* **Gate 102-A (MMS, no published data needed).** A manufactured `T(x,t)` across a two-material
  interface, `dt` refined at fixed `h`: observed `p -> 1.0 ± 0.1` for `euler` and `2.0 ± 0.1` for
  `backward`. This is the time-order measurement nothing in §46/§47/§59 makes today — Gate 3 is
  constant in time, so a first-order scheme reproduces it exactly as well as a second-order one.
* **Gate 102-B (controller).** A manufactured transient reached to a fixed error tolerance in fewer
  steps than a fixed `dt` needs **at the same error**; plus the second leg, that the controller does
  not change the answer when the step it selects is constant.
* **Gate 102-C (the per-region clock).** A transient whose solid is stepped at N times the fluid step
  agrees with the fully coupled run to a stated tolerance, with the numerical Biot number printed per
  interface and the refusal firing above the stated bound.
* **Gate 102-D (the deliverable).** A duty cycle on `cases/dieStack`: the thermal transient drives a
  sequence of quasi-static §95 solves, and the stress range over the cycle is reported. This is what
  fatigue work asks for, and it is the first time the two halves of this plan are used together.
* **Refusal.** `crankNicolson` on a coupled interface until it is gated; a coupling interval whose
  numerical Biot number exceeds the stability bound; a transient conjugate case whose solid and fluid
  clocks differ without the per-region scheme named.

---

### Stage 6 — The record (3 weeks)

#### §103 — A gate registry a driver can print into, and the report a reviewer reads

* **Modules.** The `GateReport` registry lifted out of `src/bin/validate.rs` into the library so a
  driver can register and print through the same call §69 requires; new `src/report.rs` generating the
  artefact; called from `src/bin/cht.rs`, `src/bin/datacentre.rs`, `src/bin/lowmach.rs`.
* **What the artefact carries.** The case **as lowered** (an echo of what ran, not of what was typed);
  the mesh quality report the driver already computes; the boundary and interface heat-flow table with
  its closure residual; the solver settings and converged per-**region** residuals (§93); the §94
  uncertainty where a mesh sequence exists; the software version and commit; and the gate verdicts for
  the models this case actually used, pulled from the registry rather than re-typed.
* **Gate 103-A (DESIGN, §69's own shape).** A transcript audit: a number in the report that no part of
  the run produced must be impossible to write, and a model the case used whose gate is not registered
  must make the report say so by name. No external source — §0 rule 3.

---

## D. Thermal deformation, release 1 — the headline, in full

### D.1 What it solves

Quasi-static, small-strain, isotropic, linear **thermo-elasticity** on the solid regions of the
existing `cht::ThermalMesh`, one-way from §46/§47's converged temperature field.

```
equilibrium    div(sigma) + rho_s g = 0
kinematics     eps = 1/2 (grad u + grad u^T)
constitutive   sigma = 2 mu eps + lambda tr(eps) I - (3 lambda + 2 mu) alpha (T - T_ref) I
               mu = E / (2(1+nu)),   lambda = E nu / ((1+nu)(1-2nu))
```

The finite-strain Green–Lagrange form `E = 1/2(grad u + grad u^T + grad u^T . grad u)` is **not**
proposed and is refused by name.

### D.2 On what mesh

The **same** mesh, the same cell array, the same addressing. `ThermalRegion::cells()` is already a
contiguous ascending range per region, so the solid's temperature is already per-cell on the cells a
displacement equation is assembled over: no interpolation, no second mesh, no transfer layer, no
second validation culture. The restriction to the solid regions is §93's `RegionView`.

For the gate suite, meshes come from `blockgen` where a box will do and from §92's automesher (written
as a polyMesh, read by `polymesh::{read_poly_mesh, build_host_mesh}`, which `src/bin/validate.rs`
already imports) where a cantilever, a thick-walled cylinder or a taper will not. **The gate suite
therefore does not wait for Stage 2.**

### D.3 With what discretisation, and coupled how

Cell-centred finite volume on the polyhedral mesh, **not** finite element on a second one. Integrate
equilibrium over cell P, apply Gauss, and split the surface stress (Demirdžić & Muzaferija 1995;
Jasak & Weller 2000):

```
sum_f [ (2 mu + lambda) grad(u) . Sf ]_implicit
  + sum_f [ mu grad(u)^T + lambda tr(grad u) I - (mu + lambda) grad(u) ] . Sf      (deferred)
  - sum_f (3 lambda + 2 mu) alpha (T_f - T_ref) Sf                                  (thermal load)
  + rho_s g V_P  =  0
```

* The first term **is** `fv::fvm_laplacian` with `gammaMagSf = (2 mu + lambda)|Sf|`, plus §2.4's
  over-relaxed non-orthogonal correction and §2.5's skewness correction — implemented and gated.
* The second is a surface integral of a tensor `fv::fvc_grad_vector_scheme` already produces: one new
  gather kernel and its scatter-shaped mirror in `src/reference.rs`.
* The third is a surface integral of a scalar against `Sf` — structurally the pressure-gradient term
  `src/momentum.rs` already assembles.
* Three components assemble into **one** `GpuLduMatrix` and are solved one at a time, exactly as
  `Momentum::fill_component` (momentum.rs:948) already does; the inter-component coupling is entirely
  on the right-hand side, so the outer loop is Picard, not SIMPLE, and there is no Rhie–Chow analogue.
* Because `mu` and `lambda` are constant on a static mesh, **the matrix is assembled once and the
  preconditioner built once**, for every outer iteration and every load step. `src/solver.rs` already
  separates those calls. This is the single largest lever and it is free here.
* Per component the operator is `-div((2 mu + lambda) grad u_i)`: symmetric, SPD once any displacement
  is fixed, so `solve_pcg` (§8.2) with multi-colour DIC (§21), `device_norm_factor` (§8.4, now per
  region) and §73's distributed Krylov all apply **unchanged**.
* The outer loop is a fixed point whose contraction is
  `|deferred|/|implicit| = (mu + lambda)/(2 mu + lambda) = 1/(2(1 - nu))`, with Aitken delta-squared
  dynamic under-relaxation on the displacement increment. That closed form is **DESIGN, derived here,
  and must be measured** (Gate 95-F) — §46.4 records what happened the last time this project asserted
  such an estimate instead of measuring it.
* **Multi-material.** `(3 lambda + 2 mu) alpha` is discontinuous across a bond line, and linearly
  interpolating a discontinuous coefficient invents a traction that does not exist — the identical
  failure §46.2 documents for linearly interpolated `k`, where the error is a factor `(1+r)^2/(4r)`
  that does **not** vanish under refinement. Traction continuity at the face (Tuković, Ivanković &
  Karač 2013) is what release 1 implements, and Gate 95-E's second leg shows the naive form failing to
  converge.
* **Coupling: one-way and sequential.** §46/§47 converge; then the displacement loop reads `T`.
  Two-way (the `-T_0 (3 lambda + 2 mu) alpha d(tr eps)/dt` term in §46.1's energy equation) is **named
  and not implemented**, with the coupling parameter
  `delta = (3 lambda + 2 mu)^2 alpha^2 T_0 / ((lambda + 2 mu) rho_s c_s)` **computed and printed**
  (order 1e-2 for steel), so that "negligible" is a measurement and not an assertion.

### D.4 Proving what — the gate suite

| Gate | Case | Reference value | Source | Tolerance |
|---|---|---|---|---|
| **95-A** | End-loaded cantilever, three meshes at `r = 2` | analytic `sigma_xx`, `sigma_xy`, tip deflection | Timoshenko & Goodier, *Theory of Elasticity*, 3rd ed. | observed `p >= 1.9` on displacement and `>= 0.9` on cell-centre stress, **both reported** via §94 — FV stress is one order below displacement and the section says so before the run |
| **95-B** | Unrestrained block, uniform `Delta T` | `eps = alpha Delta T I`, `sigma = 0` | closed form | `sigma <= 1e-12` relative to `(3 lambda + 2 mu) alpha Delta T`; `eps_ii` to 1e-14. One line, and it catches the commonest sign error |
| **95-C** | Linear-displacement patch test on a sheared polyhedral mesh | exact reproduction | standard FV/FE verification | `max\|u - u_exact\| <= 1e-12 \|u\|`. This is what the non-orthogonal correction has to earn |
| **95-D** | Thick-walled cylinder, steady radial temperature gradient, **chained through §46's conduction solve** | analytic `sigma_r`, `sigma_theta`, `sigma_z` | Timoshenko & Goodier; Boley & Weiner ch. 9 | `<= 1 %` on the finest of three meshes with the §94 GCI beside it. The only gate that tests the **coupling** rather than the elasticity |
| **95-E** | Bimetallic strip, two materials, uniform `Delta T` | closed-form curvature | Timoshenko, *JOSA* 11 (1925) 233 — **DOI 10.1364/JOSA.11.000233 verified** | `<= 2 %`; plus the second leg in which interpolating `(3 lambda + 2 mu) alpha` linearly is shown **not** to converge under refinement |
| **95-F** | Outer-loop contraction at `nu = 0.2 / 0.3 / 0.45` | predicted `1/(2(1-nu))` = 0.625 / 0.714 / 0.909 | derived here — **DESIGN** | the predicted and observed outer-iteration counts printed side by side. The gate is that the prediction is **measured**; §D.5's refusal threshold is then set from the measurement, not from the algebra |
| **95-G** | NAFEMS LE1 elliptic membrane; LE10 thick plate; LE11 cylinder/taper/sphere under `T = sqrt(x^2+y^2) + z` | `sigma_yy(D) = 92.7 MPa`; `sigma_yy(D) = -5.38 MPa`; `sigma_zz(A) = -105 MPa` | *The Standard NAFEMS Benchmarks*, ref. P18 Rev. 3 (1990) — **paywalled, not obtained** | `<= 3 %` on the finest mesh. **Not written until the primary is in hand.** The targets above are the ones the secondary literature universally restates; the full geometry, material and boundary-condition specification certainly cannot be taken from a restatement. If the primary cannot be obtained, the gate carries §60.5's Gate 5 disclosure in the same words |

Release 1 is **fully gated without NAFEMS**: 95-A through 95-F need only closed forms and a
manufactured solution. LE11 is the external leg and the sales line; it is not the load-bearing one.

Plus §96's ten pair tests, of which the load-bearing one is: two cases differing only in `alpha` must
give different displacement fields, failing by name if they do not.

### D.5 What release 1 refuses, by name

1. **Near-incompressible solids.** `nu` above the threshold Gate 95-F measures (provisionally 0.45,
   where the contraction is 0.909 and six decades cost ~145 outer iterations; 0.49 costs ~684). The
   error names the number reached and names block coupling as the route that survives it. A rubber
   gasket or a sealant is ordinary in the enclosures this solver already meshes, and a
   near-incompressible solid entered without a refusal does not fail — it stalls at a residual the
   user reads as converged.
2. **Block-coupled FV** (Cardiff, Tuković, Jasak & Ivanković 2016): a 3×3 coefficient per face breaks
   §1's one-entry-per-face LDU storage exactly as MPFA breaks it in §46.4, and it is refused for the
   same reason and named the same way.
3. **Finite strain / large rotation** (Cardiff, Karač & Ivanković 2014).
4. **Plasticity** — named as the first thing after release 1 (Demirdžić & Martinović 1993: a J2 return
   map on the same assembly).
5. **Contact and friction** — needs a search, an active set and a non-symmetric, non-LDU system.
6. **Fracture.**
7. **Inertia.** A `ddtScheme` on a displacement equation is an error naming the Newmark / HHT /
   generalized-alpha family as what is not built, and naming controlled algorithmic damping
   (`rho_infinity`) as what BDF2 would not give.
8. **Anisotropic (orthotropic) stiffness on a non-aligned mesh** — measured and refused in exactly
   §46.4's shape, with an alignment residual per face, not asserted.
9. **Two-way thermoelastic coupling** — `delta` computed and printed.
10. **Displacement reaching the fluid.** `max|u| / min cell size` is computed over the solid and the
    run errors out above a stated threshold rather than silently reporting a fluid solution on a mesh
    that should have moved. The space conservation law (Demirdžić & Perić 1988) and a mesh-motion
    solver are named as what a two-way route would need, and are explicitly **not** an extension of
    §75's adapt, which changes topology and would silently give the wrong temporal term.

### D.6 What it outputs

`u` (vector, on **points** as well as cells, so a deformed shape can be warped), `sigma` (6
independent components through §93's `Tensor` arm, expanded to VTK's 9 at write time), von Mises
`sqrt(3/2 s:s)`, the three principal stresses (characteristic cubic — trigonometric closed form, or a
Jacobi rotation for conditioning), the hydrostatic/deviatoric split, and `|u|` for clearance and
interference. These are the numbers a user acts on: von Mises against a published proof stress, max
principal against a brittle allowable, `|u|` against a gap.

---

## E. Sources the plan rests on

All DOIs below were resolved against `https://api.crossref.org/works/<doi>` in this session unless the
row says otherwise.

### E.1 Thermal deformation

| Source | What it gives | Licence / ID | DOI verified |
|---|---|---|---|
| Demirdžić & Muzaferija, *IJNME* 37 (1994) 3751–3766 | FV stress analysis in complex domains — the segregated cell-centred formulation | paper | **yes** 10.1002/nme.1620372110 |
| Demirdžić & Muzaferija, *CMAME* 125 (1995) 235–255 | Coupled fluid flow, heat transfer and stress on unstructured meshes; the multi-region container | paper | **yes** 10.1016/0045-7825(95)00800-G |
| Jasak & Weller, *IJNME* 48 (2000) 267–287 | The `(2 mu + lambda)` implicit split and its convergence behaviour | paper | **yes** 10.1002/(SICI)1097-0207(20000520)48:2<267::AID-NME884>3.0.CO;2-Q (Crossref date part is **20000520**, not the 20000530 that circulates) |
| Demirdžić & Martinović, *CMAME* 109 (1993) 331–349 | The thermal-strain source in FV form; the J2 return map named for release 2 | paper | **yes** 10.1016/0045-7825(93)90085-C |
| Tuković, Ivanković & Karač, *IJNME* 93 (2013) 400–419 | Multi-material FV stress: traction continuity instead of interpolating stiffness | paper | **yes** 10.1002/nme.4390 (Crossref issued 2012 online) |
| Cardiff & Demirdžić, *ACME* 28 (2021) 3721–3780 | The thirty-year review; segregated convergence degradation as `nu -> 0.5`; boundary stress recovery | review | **yes** 10.1007/s11831-020-09523-0 |
| Cardiff, Tuković, Jasak & Ivanković, *Comput. Struct.* 175 (2016) 100–122 | Block-coupled FV elasticity — **named in the refusal**, not implemented | paper | **yes** 10.1016/j.compstruc.2016.07.004 |
| Cardiff, Karač & Ivanković, *CMAME* 268 (2014) 318–335 | Large-strain FV — named in the refusal | paper | **yes** 10.1016/j.cma.2013.09.008 |
| Demirdžić, Muzaferija & Perić, *IJNME* 40 (1997) 1893–1908 | Published FV solutions of structural benchmarks; multigrid on the outer loop | paper | **yes** 10.1002/(SICI)1097-0207(19970530)40:10<1893::AID-NME146>3.0.CO;2-L |
| Timoshenko, *JOSA* 11 (1925) 233 | Bimetallic-strip curvature — Gate 95-E's closed form | paper | **yes** 10.1364/JOSA.11.000233 |
| Demirdžić & Perić, *IJNMF* 8 (1988) 1037–1050 | Space conservation law — named in the mesh-motion refusal | paper | **yes** 10.1002/fld.1650080906 |
| Timoshenko & Goodier, *Theory of Elasticity*, 3rd ed., McGraw-Hill (1970) | Cantilever; thick cylinder with a radial temperature gradient; stress invariants | book, no DOI | n/a — ISBN not asserted |
| Boley & Weiner, *Theory of Thermal Stresses*, Wiley (1960)/Dover (1997) | Duhamel–Neumann; the thermoelastic coupling parameter; the thick-cylinder thermal solution | book, no DOI | n/a |
| *The Standard NAFEMS Benchmarks*, ref. P18 Rev. 3 (1990) | LE1, LE10, LE11 — Gate 95-G | NAFEMS publication, **paywalled** | no DOI exists; **targets not read from the primary** |

### E.2 Verification, coupling and time

| Source | What it gives | Licence / ID | DOI verified |
|---|---|---|---|
| Roache, *J. Fluids Eng.* 124 (2002) 4–10 | Method of manufactured solutions | paper | **yes** 10.1115/1.1436090 |
| Roache, *J. Fluids Eng.* 116 (1994) 405–413 | Uniform reporting of grid refinement; the GCI | paper | **yes** 10.1115/1.2910291 |
| Celik, Ghia, Roache, Freitas, Coleman & Raad, *J. Fluids Eng.* 130 (2008) 078001 | The three-mesh observed-order + GCI procedure in the form a reviewer expects | paper | **yes** 10.1115/1.2960953 |
| Eça & Hoekstra, *JCP* 262 (2014) 104–130 | The least-squares estimator for non-asymptotic and oscillatory sequences | paper | **yes** 10.1016/j.jcp.2014.01.006 |
| Meng, Banks, Henshaw & Schwendeman, *JCP* 344 (2017) 51–85 | Already §47's Theorem 1; the transient partitioned analysis and its MMS setup | paper | **yes** 10.1016/j.jcp.2017.04.052 |
| Errera & Chemin, *JCP* 245 (2013) 431–455 | Quasi-steady solid; stability when regions advance on different clocks | paper | **yes** 10.1016/j.jcp.2013.03.004 |
| Verstraete & Scholl, *IJHMT* 101 (2016) 852–869 | The numerical Biot number, printed per interface | paper | **yes** 10.1016/j.ijheatmasstransfer.2016.05.041 |
| Küttler & Wall, *Comput. Mech.* 43 (2008) 61–72 | Aitken delta-squared dynamic relaxation of a partitioned fixed point | paper | **yes** 10.1007/s00466-008-0255-5 |
| Kay, Gresho, Griffiths & Silvester, *SISC* 32 (2010) 111–128 | The BDF2 predictor–corrector local-error estimate and the step formula | paper | **yes** 10.1137/080728032 |
| Söderlind, *ACM TOMS* 29 (2003) 1–26 | The PI / digital-filter step controller | paper | **yes** 10.1145/641876.641877 |
| Duchaine, Corpron, Pons, Moureau, Nicoud & Poinsot, *IJHFF* 30 (2009) 1129–1141 | A coupled strategy with two solvers on different time scales; the exchange frequency | paper | **yes** 10.1016/j.ijheatfluidflow.2009.07.004 |
| Arioli, *Numer. Math.* 97 (2004) 1–24 | Why a globally normalised algebraic residual is not a statement about discretisation error | paper | carried from lens 4; **not re-verified here** |
| Carslaw & Jaeger, *Conduction of Heat in Solids*, 2nd ed., OUP (1959) | The affine reduction for Gate 46-B; the fin closed form; the Kirchhoff transform; Stefan | book, ISBN 0-19-853368-3 — **already an in-repo source** | n/a |

### E.3 Heat transfer

| Source | What it gives | Licence / ID | DOI verified |
|---|---|---|---|
| Flageul, Benhamadouche, Lamballais & Laurence, *IJHFF* 55 (2015) 34–44 | Gate 7: DNS of conjugate channel flow, four thermal wall treatments | paper, open access at hal.science | **yes** 10.1016/j.ijheatfluidflow.2015.07.009 |
| Tiselj, Bergant, Mavko, Bajsić & Hetsroni, *JHT* 123 (2001) 849–857 | Second, milder DNS leg | paper | **yes** 10.1115/1.1389060 |
| Ampofo & Karayiannis, *IJHMT* 46 (2003) 3551–3572 | Turbulent natural convection, `Ra = 1.58e9` — Gate 101-B | paper (paywalled; values not read) | **yes** 10.1016/S0017-9310(03)00147-9 |
| Betts & Bokhari, *IJHFF* 21 (2000) 675–683 | Tall-cavity turbulent natural convection, second leg | paper (paywalled) | **yes** 10.1016/S0142-727X(00)00033-3 |
| de Vahl Davis, *IJNMF* 3 (1983) 249–264 | The benchmark table Gate 59-A already uses — and its `Ra = 1e6` row, which is unrun | paper, already an in-repo source | **yes** 10.1002/fld.1650030305 |
| Churchill & Chu, *IJHMT* 18 (1975) 1323–1329 | `Nu(Ra, Pr)` band for the convective BC and the buoyant wall | paper | **yes** 10.1016/0017-9310(75)90243-4 |
| Hölling & Herwig, *JFM* 541 (2005) 383–397 | Asymptotic near-wall analysis for turbulent natural convection | paper | **yes** 10.1017/S0022112005006300 (note `...006300`, not `...006178`) |
| Balaji, Hölling & Herwig, *JHT* 129 (2007) 1100–1105 | Nusselt correlations from that analysis | paper | **yes** 10.1115/1.2737485 (Crossref issued 2006) |
| Glassbrenner & Slack, *Phys. Rev.* 134 (1964) A1058 | `k_Si(T)` — Gate 99-B | paper | **yes** 10.1103/PhysRev.134.A1058 |
| Okada & Tokumaru, *JAP* 56 (1984) 314 | `alpha_Si(T)` | paper | **yes** 10.1063/1.333965 |
| Sutherland, *Phil. Mag.* 36 (1893) 507–531 | The two-constant viscosity law | paper | **yes** 10.1080/14786449308620508 |
| Kadoya, Matsunaga & Nagashima, *JPCRD* 14 (1985) 947–970 | Reference correlation for dry air `mu(T)`, `k(T)` — Gate 99-B | paper | **yes** 10.1063/1.555744 |
| Brinkman, *Appl. Sci. Res.* A2 (1951) 120–124 | Heat effects in capillary flow — Gate 99-C's closed form | paper | **yes** 10.1007/BF00411976 |
| Cooper, Mikić & Yovanovich, *IJHMT* 12 (1969) 279–300 | The plastic contact correlation already implemented as `cmy_contact_conductance` | paper, already an in-repo source | **yes** 10.1016/0017-9310(69)90011-8 |
| Mikić, *IJHMT* 17 (1974) 205–214 | The elastic / elastoplastic regimes the plastic correlation does not cover | paper | **yes** 10.1016/0017-9310(74)90082-9 |
| Yovanovich, *IEEE TCPT* 28 (2005) 182–206 | The review covering gap conduction and the elastic regime; `h_c(P)` | paper, already an in-repo source | **yes** 10.1109/TCAPT.2005.848483 |
| Patankar, *Numerical Heat Transfer and Fluid Flow*, Hemisphere (1980) | Source linearisation `S = S_C + S_P T`, `S_P <= 0` | book, ISBN 0-89116-522-3 — **already an in-repo source** | n/a |
| Modest, *Radiative Heat Transfer*, 3rd ed. | The grey two-surface closed forms already coded; radiation coupled to convection and conduction | book, ISBN 978-0-12-386944-9 — already in §50 | not re-checked |
| McBride, Zehe & Gordon, NASA/TP-2002-211556 | The 7-coefficient `cp(T)` polynomial form | **US Government, public domain**, no DOI | n/a |
| McCaffrey, *Purely Buoyant Diffusion Flames*, NBS (1979) | The three-regime plume centreline correlations — Gate 101-C | **US Government, public domain**, no DOI | identifier **unconfirmed** — see §F.4 |

### E.4 Named as documentation only (no code read; licence stated per §0 rule 2)

* **PyFR (BSD-3, Imperial College)**, already vendored at `reference/pyfr` — `writers/vtk/` is the
  blessed cross-check for the appended-binary VTU encoding when §93 adds `PointData`.
* **MFEM (BSD-3, LLNL)** and **Kratos Multiphysics (BSD-3)** — published documentation only, named in
  the deferred AMG-near-null-space and region-decomposition discussions. Nothing read, nothing used.
* **FDS Technical Reference Guide (NIST, US Government public domain)**, vendored at `reference/fds` —
  cited as documentation for the same published plume correlations under §0 rule 3, if Gate 101-C
  needs a second statement of them.

---

## F. Risks, and the cheapest experiment that finds out early

| # | Stage | What could make it fail | Cheapest early experiment |
|---|---|---|---|
| 1 | 1 (§95) | **The segregated outer loop is too slow at engineering Poisson ratios.** `1/(2(1-nu))` is derived here, not quoted; at `nu = 0.45` it predicts ~145 outer iterations for six decades and at 0.49 ~684. If the real contraction is worse (non-orthogonality and the deferred traction BC both feed it), release 1 ships a module nobody can afford to run. | **Two days, before any kernel.** Write the displacement operator host-side only, in `src/reference.rs`'s scatter-shaped style, on a 20×20×20 block with a fixed face and a uniform `Delta T`, and count outer iterations at `nu = 0.2/0.3/0.45` with and without Aitken. If the observed contraction tracks `1/(2(1-nu))`, Stage 1 proceeds as written; if it is materially worse, lens 4's block matrix moves ahead of §95 and the stage plan changes before anything is built on the GPU. |
| 2 | 1 (§95) | **Boundary stress extrapolation.** FV produces cell and face values; NAFEMS targets are at boundary *points*. A solver that gets displacement right and boundary stress wrong is exactly the one a user would trust and should not. | Gate 95-A's stress order, measured on the cantilever **before** LE10 is attempted: if cell-centre stress converges at `p >= 0.9` but the extrapolated boundary point does not, the extrapolation is specified and gated in its own right — a paragraph in §95, not a surprise in 95-G. |
| 3 | 0 (§93) | **The per-region residual changes existing gate results.** Nine gates in this area currently pass against a global norm; tightening the test could turn a PASS into a MISS, and it would be right to. | Run §93's per-region norm in **report-only** mode first across the whole existing gate suite: print both numbers, change no verdict. One afternoon, and it tells you exactly which gates were passing on the stiff region's back. |
| 4 | 4 (§101) | **The McCaffrey identifier.** §10/§22 cite "NBS TN 910 (1979)"; the report is more commonly catalogued as NBSIR 79-1910. A gate name written on a wrong identifier is a provenance defect in a repository whose whole claim is provenance. | One NIST catalogue lookup, before the gate name is written. Cheap, and it must precede §101's section text. |
| 5 | 1 (§95) | **NAFEMS P18 cannot be obtained.** The three target values circulate; the geometry, material and boundary-condition specification do not. | Decide now, not later: release 1's verdict rests on 95-A..F, which need nothing paywalled. If P18 arrives, 95-G is added; if not, it carries §60.5's Gate 5 disclosure in the same words. **Nothing in the stage plan blocks on it.** |
| 6 | 2 (§98) | **The radiation/energy outer loop does not converge** where `h_rad/h_conv` is large — §50.12 records that `radiationRelaxation` exists only because the Picard lag has no convergence proof, and "has never been exercised against a case that actually needs it". | Gate 98-C run as a sweep in `h_rad/h_conv` over three decades, counting outer iterations. It is a small enclosure case; a day of machine time answers whether the lag needs relaxation, and where. |
| 7 | 2 (§97) | **Imported meshes fail §47.4's pairing tolerances** — the 5° interface non-orthogonality refusal was chosen by argument, and a real automesher mesh may simply not pass it. | **Gate 94-B is the experiment, and it is already in Stage 0.** Measure the observed order on a deliberately skewed interface before Stage 2 lets real meshes in; that measurement sets the threshold instead of the argument. |
| 8 | 3 (§99) | **The per-iteration conductivity rebuild costs more than it is worth**, and the outer loop `run_case` does not have today may not converge for a strongly nonlinear `k(T)`. | Gate 99-A's Kirchhoff slab with `k(T)` varying by 5× across the slab: count outer iterations, and measure the rebuild against the solve. If the rebuild dominates, cache the face conductances and rebuild on a cadence — a decision the measurement makes. |
| 9 | 5 (§102) | **The transient conjugate coupling is unstable** at the `(rho c)` ratio of O(1e3) §59.6 refuses on, once the two regions are on different clocks. | Verstraete & Scholl's numerical Biot number, computed and printed **per interface on the existing steady cases** before any transient code is written. It costs one reduction, and it says in advance which of the shipped cases would be stable. |
| 10 | all | **Scope drift into the deferred list.** Participating media, AMI, block coupling, plasticity and phase change are each individually defensible and collectively a second year. | The refusal list is the control: every deferred item is refused **by name, in code**, at the stage that would otherwise be tempted to start it. A refusal that names the paper is cheaper than a half-built model, and it is this repository's own habit. |

---

## G. What this plan does not do

It does not add participating-media radiation, non-conformal interfaces, a block-coupled matrix, an
AMG the project owns, plasticity, phase change, porous-media thermal non-equilibrium, a cooling-coil
model, wall condensation, mesh motion, or a distributed conjugate run. Each is named above with the
reason and, where it is a physics capability rather than an optimisation, with the refusal that should
exist in the meantime. Four of them (participating media, AMI, block coupling, plasticity) are the
natural §104+ once the measurements in Gates 94-B, 95-F and 101-C say which of them the evidence
actually asks for.


---

## H. Review additions (Fable 5.1, 2026-09-11)

### H.1 A second track: the mesh that moves — §104–§106

§A.3 item 1 demotes mesh motion as "a solver-wide programme starting at the space conservation
law", and for the thermo-mechanical goal that ruling stands. But the same programme is the base of
three things the product has been asked for since — a moving mesh, an overset (chimera) mesh, and a
mesh fit for VOF with a floating body — and of the two-way route §D.5 item 10 refuses. It is
therefore planned here as **Track B**, to start after Stage 1 has shipped, in this order:

| § | unit | what it owns | size | gate |
|---|---|---|---|---|
| **§104** | **Arbitrary Lagrangian–Eulerian motion and the space conservation law** | point motion (prescribed rigid-body, and a Laplacian / inverse-distance smoothing solve on `fv::fvm_laplacian`); the swept-volume mesh flux `phi_mesh` per face computed with the same time scheme as the transported quantity (Demirdžić & Perić 1988, DOI 10.1002/fld.1650080906; Thomas & Lombard 1979, DOI 10.2514/3.61273); cell volumes advanced by the swept volumes so `V^{n+1} - V^n = dt · Σ_f phi_mesh,f` holds to round-off (the space conservation law); `phi - phi_mesh` in every convective term; the moving-wall velocity condition; the per-step geometry recompute on the device that §82/§83 already own | M | **104-A** uniform flow on a mesh in prescribed rigid motion stays uniform to round-off (the SCL test); **104-B** a translating/expanding box with an exact solution, time order measured through §94; **104-C** the oscillating-cylinder Strouhal lock-in band (Williamson & Roshko 1988, DOI 10.1016/S0889-9746(88)90058-8) |
| **§105** | **A mesh fit for VOF** | an adapt criterion (§75) on `|grad alpha|` that tracks the interface with a hysteresis band; a `freeSurfaceBand` refinement box in the automesher (§92); the wetted-wall layer treatment; and, only if the measurement asks for it, a geometric (PLIC) reconstruction beside §20's algebraic compression | S–M | **105-A** §20's existing dam-break gate re-run with the interface-tracking adapt, the interface thickness in cells reported at each level; **105-B** the Martin & Moyce front position at half the uniform-mesh cost |
| **§106** | **Overset (chimera)** | hole cutting against the body mesh; donor search through §92's octree / §66–§67's mesh walk; interpolation stencils (inverse distance, then gradient-corrected) written as off-diagonal CSR entries through §48's coupled-entry path; the pressure-equation flux correction for the non-conservative interpolation; orphan handling refused by name (Steger, Dougherty & Benek 1983, AIAA 83-1944; Chesshire & Henshaw 1990, DOI 10.1016/0021-9991(90)90196-8; Benek, Buning & Steger 1985, AIAA 85-1523) | L | **106-A** a static overset of two boxes reproduces the single-mesh Poiseuille solution to discretisation error; **106-B** a body moving through a background mesh (needs §104) conserves mass to a stated bound, the deficit printed; **106-C** the cylinder-in-crossflow drag band |

§104 also unblocks §D.5 item 10: the displacement of a solid region moving the fluid mesh is one
call into §104's smoother, and the refusal there becomes a capability.

### H.2 How it is built

Coding is done directly by Opus 5 agents (not the GLM wrapper), one unit per agent, on the worktree
`Iteration-CFD-solver` (branch `feat/thermal-structural`, from `main`), each unit ending with the
house gates green: `cargo build --release`, the unit's own `cargo test --release`, and the two
audits `xref` (§80) and `provenance_audit`. The order is the plan's own: the risk-1 experiment
first (a host-side prototype of the displacement outer loop, so Stage 1's discretisation is chosen
by a measurement), then §93, §94, §95, §96, then verification. Track B follows Stage 1.
