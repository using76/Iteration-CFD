# DUST stream — INSIGHTS (design decisions and why)

Regenerated 2026-10-10 by the Opus planner, papers only (RESEARCH.md §A). Equation ids `D.n` refer to RESEARCH.md.
Each decision says what it is, why it follows from first principles and the open papers, how a gate checks it,
and what would make us change it.

---

## 0. The clean room, and what the regeneration did

**Binding rule (the user, verbatim):** "인사이트만 얻고 완전히 새로운 코드로 만들어줘 ... 수식만 논문에서보고 같게 구현해줘"
— take only insights, write entirely new code, take the equations only from the papers and implement them
identically.

How the stream keeps it:

1. **Equations come from the open papers only** (H20, H22, TMR, BKM24, ISL22, MZG02), or are derived from them in
   RESEARCH.md with the derivation written out. A paper that is not open is used only through an open paper that
   restates it, and is listed as not open (RESEARCH §L).
2. **This regeneration opened no FSCDustFoam code**, no file of its Zenodo archive, and no OpenFOAM source. Only the
   Zenodo record page was read, for the citation (RESEARCH §A.3). The insights (a)-(f) below were re-derived from the
   papers and from first principles; where the lost spec held a measured fact this file could not re-verify
   (insight (a)'s "0.5-0.67 S_L"), it says so and shows the mechanism instead.
3. **The coder never sees FSCDustFoam or any code-derived text.** GLM briefs quote SPEC-LIT §151-§160 (written by
   DST-00 from RESEARCH.md) and the unit's gate. Nothing else.
4. **Every PROVENANCE row a DUST unit creates carries the user-mandated sentence:** *"The specification was informed
   by a review of FSCDustFoam (Huang & Lipatnikov, GPL) and no code was copied."* It refers to the original planning
   pass; the regenerated specification is papers-only, and the row may add that. The house header line *"No
   GPL-licensed source was consulted."* stays true for every new source file, because the coder consults none.
5. **Licence position:** the Zenodo record says CC BY 4.0, but the solver is an OpenFOAM (GPL v3) derivative. The
   stream treats it as GPL, the stricter reading and the user's.

---

## (a) Regenerate Ψ from its defining integral; never use a published fit as is

**What.** t_r in the laminar-limit source Q (D.18) is set by `t_r = Ψ² κ_u / S_L²` (D.25). The papers define Ψ by a
constraint, not by a formula: with D_t = U_t = 0, the stationary 1-D solution of the c~ equation must propagate at
exactly S_L (H20 p. 3-4). The stream solves that eigenproblem (D.26) on the host in f64 at case setup, for the
case's own closure, and checks the converged solution against the integral identity
`Ψ² = ∫_0^1 g w / P dc` (D.27). A tabulated grid over (σ, Θ/T_u, n) is generated for documentation and review,
never as the solver's input.

**Why a fit fails.**
- Ψ is a functional of the whole state curve along the flame: T(c), ρ(c) and the diffusivity law ρκ(c). The
  published two-argument form Ψ(T_b/T_u, Θ/T_u) is exact only for one particular closure (D.24 with one κ law). The
  papers state neither κ nor its temperature law (RESEARCH §K item 12).
- The sensitivity is large. With the REMBE state (σ = 7.04, Θ/T_u = 73.3), a Ψ built for ρκ = const (n = 1) run in a
  solver whose κ law is n = 0.5 propagates the laminar limit at **0.64 S_L**; n = 0 gives 0.41 S_L, n = 1.5 gives
  1.56 S_L (RESEARCH D.27, planner prototype). Sutherland μ with a constant Pr is n ≈ 1.7. The lost spec recorded
  that the published fit, evaluated in the equation as written, propagates at **about 0.5-0.67 S_L**. That could not
  be re-measured here (the fit is in Huang et al. 2016, not open), but the mechanism above reproduces exactly that
  range for a one-step mismatch in the κ law.
- Even the classical large-activation-energy (ZFK) integral `Ψ² ≈ 2∫ g w dc` (D.28) is 13 % fast in speed at the
  Zel'dovich numbers of these cases (Ze ≈ 9): Λ/Λ_ZFK = 0.78. An asymptotic shortcut is not good enough either.
- One-temperature thermo (insight (b)) changes T(c) and ρ(c) again (W_U ≠ W_B, c_p,U ≠ c_p,B). Only a Ψ computed
  for the solver's own curves satisfies the definition.

**Why it survives turbulent validation, and why it still matters.** In turbulent cells Q is divided by
`1 + D_t/κ_b` (D.18), about 70 for REMBE (D_t,∞ ≈ 0.03 m²/s, κ_b ≈ 4e−4 m²/s at 1778 K), so the turbulent U_t term carries the
flame and a wrong Ψ barely moves turbulent results. It does matter where turbulence is weak: kernel initiation after
ignition, near walls, quiescent closed vessels (the laminar Dahoe-type tests), and the low-turbulence fresh mixture
outside a vent. A model whose laminar limit is wrong by a factor of two is wrong exactly where users test it first.

**Gate.** DST-02 (the eigen-solver: N-convergence, the identity D.27, c_cut insensitivity, Λ/Λ_ZFK → 1 as Ze grows)
and DST-04 (an independent method: the device FV solver's 1-D laminar flame propagates at S_L within 1 % with the
regenerated Ψ, and at (Λ_ZFK/Λ)^{1/2} S_L within 1 % with the ZFK value — the two numerical methods agree both
ways).

**Would change if** an open source states the κ law the published fit was built for. Even then the stream keeps
the setup-time solve, because it costs milliseconds and is exact for whatever closure the case selects.

---

## (b) One energy equation, one temperature; the closed-vessel check is AICC P_max/p_0 = 8.57

**What.** The equivalent premixed gas is two pseudo-species, unburned U and burned B, each an exact NASA-7 linear
combination of its component species (D.8). `c~` is the mass fraction of B. The mixture has one enthalpy
`h(T, c~) = (1−c~) h_U(T) + c~ h_B(T)`, one temperature T and one energy equation (D.12), with W(c~) in the gas law
(D.9). There is no separate unburned-temperature field and no second energy equation.

**Why.**
- The progress variable already fixes the local mean thermochemical state: in the BML picture (D.15) the mixture is
  an intermittent sequence of fully unburned and fully burned gas, so (c~, h) determine T. A second enthalpy for the
  unburned gas adds a second equation for one energy budget, and two equations for one budget can drift apart.
- The unburned temperature is still needed, for S_L(T_u, p) (D.14, option) and for compression. In a closed vessel
  it follows algebraically from isentropic compression of the fresh gas, `T_u = T_u0 (p/p_0)^{(γ−1)/γ}` (D.52), using
  the sealed p_0(t) the low-Mach driver already integrates (§25.2). No transport equation is needed.
- The published implementation evidently carries both: H22 Table 3 lists boundary conditions for a `Tu` field and a
  `b` field. That is all that was inferred; the choice here was made on the grounds above.
- Two pseudo-species put Phase A on the CHEMRAD machinery that tranche 17 builds anyway: NASA-7 thermo (CHR-01),
  species transport with a source (CHR-09), the c_p field and the heat release (CHR-10a/b), W_mix in the density and
  the molar-change term in the sealed constraint (CHR-11). The FSC model becomes "the source of Y_B", not a solver.
- Conservation becomes checkable. A sealed adiabatic box that burns completely must end at the adiabatic isochoric
  complete-combustion state (AICC, D.11) of the same thermo, an identity no tuning can fake.

**The number.** For corn starch C6H7.88O4.98 at φ_corr = 0.62, T_0 = 273 K, complete combustion to CO2/H2O/O2/N2,
dust counted as gas moles (the equivalent-gas convention H20 Table 1's W_u = 32.8 g/mol requires), O2:N2 = 1:3.76:
**P_max/p_0 = 8.57** (stream value). The planner's independent hand check gives 8.597 (+0.3 %) with GRI-Mech 3.0
NASA-7 products and H22 Table 1 fuel data. The conventions matter and are fixed in §152: 1:3.71 (as H20 printed)
gives 8.66 (+1.0 %), and counting the solid dust as having no gas volume gives 8.81 (+2.8 %). The isobaric state at
the same inputs is T_b = 1778 K, σ = 7.04 (REMBE; H22 states no σ).

**Two thermo modes, both one-temperature.** *Formula mode* (REMBE, the default): U and B built from the formula,
D.1-D.8. *Prescribed mode* (Leeds): constant c_p, W_u = 32.8 and W_b = 29.7 g/mol and T_b = 1500 K (measured) as
H20 Table 1 states, so that σ = 5.06 as published. The constant-c_p prescribed mode is also the one in which the
thin-flame relation D.51 is exact (insight (h)).

**Gates.** DST-01 (AICC 8.57 ± 1 %, the heat-of-reaction identity D.6 to 0.1 %), DST-06 (a sealed box in
`ofgpu-lowmach -sealed` ends at the host AICC of its own thermo within 0.5 %, error first order in Δt), DST-10 (the
same identity in `ofgpu-compressible`).

**Would change if** a measured T_b-driven case (radiative loss, incomplete burnout) is wanted in formula mode. That
is a heat-loss model, named and not built, not a second temperature.

---

## (c) A one-domain opening vent; no separate outside domain

**What.** The mesh includes the outside domain from t = 0. The vent is a face set M of that one mesh, selected by
geometry, which behaves as a wall on both sides until the area-weighted pressure difference across it reaches
p_stat, and from the next step on is an ordinary interior face set (irreversible, D.57).

**Why.**
- H22 ran the closed stage, stopped at 0.1 bar, and mapped the fields onto a second, larger mesh for venting
  (§3.6). A mesh-to-mesh map is an interpolation. It does not conserve mass and energy exactly, it must re-seed the
  turbulence, the time levels and the flame clock, and it needs its own code path. One domain has none of these
  problems: mass, momentum and energy are conserved through the opening by construction.
- The outside gas is quiescent ambient air during the closed stage. Computing it costs little, and nothing in it
  changes until the membrane opens.
- A membrane burst is a Riemann problem. That gives an exact open reference at once (Toro's solution, already in the
  suite through CMP-07): a 1-D tube with an opening membrane must reproduce the Sod shock tube.
- The opening is a discrete event at a step in either approach. Here it is decided by the solver's own pressure
  field, not by a person stopping a run.

**Why compressible.** A vent at 0.6 bar overpressure drives a near-sonic jet (choking starts near Δp/p ≈ 0.89), so
the vented cases run on `ofgpu-compressible` (CMP-15) with its total-pressure outer boundaries (CMP-16). The closed
vessels, whose flames are three orders of magnitude slower than sound and whose pressure is uniform to M², run on
`ofgpu-lowmach -sealed` (§25.2).

**Not modelled (named):** membrane inertia and an opening time, partial or petal opening, and the BC names that
describe active baffles. They are refused by name in BC-01's catalogue, pointing to the vent block.

**Gate.** DST-09 (closed membrane: zero flux, bitwise, each side evolves exactly as with a wall there; open at t = 0:
the Sod result bitwise equal to the same run without a membrane, and Sod against Toro to CMP-19's tolerances; the
trigger fires at the first step with Δp ≥ p_stat). DST-11 (REMBE vented).

---

## (d) A flame-Courant time-step limit

**What.** Besides the flow CFL, every step satisfies
`Co_f = Δt (|u~| + ρ_u U_t/ρ‾) / Δx ≤ Co_f,max` (default 0.5, D.39). The laminar-limit source Q is linearised
implicitly (D.38). The driver refuses, by name, a fixed Δt whose predicted Co_f is too large, and reports the
measured maximum at every check.

**Why.**
- The term ρ_u U_t |∇c~| is explicit and hyperbolic: it moves the c~ front relative to the fluid at
  `w = ρ_u U_t/ρ‾`, which grows from U_t on the fresh side to σ U_t on the burned side (7 U_t for REMBE). A step can
  pass the flow CFL in the nearly stagnant burned core while the front crosses several cells. That is exactly the
  CFL condition for the front's own speed.
- Q is stiff (rate constants of order 1e6 s⁻¹ in the burned gas), but it is linear in (1 − c~). The Patankar split
  `S_U = ρK`, `S_P = −ρK` keeps c~ ≤ 1 for any Δt, so Q needs no step limit.
- The low-Mach driver captures its step into a CUDA graph (§81) with a fixed Δt. A setup-time check plus a reported
  maximum keeps capture intact. Adaptive stepping is used only where a driver already has it.
- H20 ran at a maximum Courant number of 0.1, which is conservative. The gate shows the propagation speed does not
  depend on Co_f below the bound.

**Gate.** DST-04 (planar front speed independent of Δt within 0.5 % for Co_f = 0.1 and 0.4; a run at Co_f > 1 is
refused by name).

---

## (e) f32-safe Arrhenius evaluation (single precision is IMPORTANT)

**What.** The device never forms `1/t_r` and `exp(−Θ/T)` separately. The host forms the scaled rate constant
`K_b = exp(−Θ/T_b)/t_r` once in f64, and the device evaluates only `K_b exp(θ_b (1 − T_b/T))` (D.37). Below
`ln(FLT_MIN) + 24 ln 2` the exponent returns exactly zero, and the kernel counts the cells it zeroed. The scope is
recorded as a subsection of §118, the tranche's single mixed-precision section.

**Why.**
- The magnitudes are extreme. For REMBE, t_r ≈ 3e−11 s, so 1/t_r ≈ 3e10 s⁻¹, while exp(−Θ/T_u) = 1.9e−32. The
  product is about 6e−22 s⁻¹ at the fresh side and 1e5-1e6 s⁻¹ at the burned side. In f32 the fresh-side exponential
  drops into the subnormal range below about 227 K, where flush-to-zero behaviour differs between builds and flags,
  and a difference that only exists in f32 is the worst kind to debug.
- After scaling, K_b is of the order of the inverse reaction time, and the exponent is bounded: at most 0 below T_b,
  of order one where the reaction actually happens, at most θ_b ≈ 11 under compression. f32 then carries about 7
  digits where the source matters and an exact zero where it does not.
- Precision rules of docs/17 §5.1 apply to every DUST unit. Both builds compile. The f64 build stays bitwise unless a
  DUST option is selected. There is no new single-ignore attribute. Every gate row has a precision class (P, A, D, R).
  The Ψ eigen-solve, the AICC and thin-flame references, and every oracle are host f64 in both builds. Reductions
  (burned volume, ∫Q dV for p_0, flame radius integrals) accumulate in `ofacc`.

**Gate.** DST-03 (over T ∈ [150, 3500] K and c~ ∈ [0, 1] the f32 kernel matches the f64 host twin to the R-class
tolerance wherever the f64 value is representable, returns exactly 0 below the threshold, writes no subnormal (bit
check), and its zero-counter equals the host's prediction). Every later gate has an f32 row in its own class.

---

## (f) DST-20: a local flame-age clock, as an OPTION, default OFF

**What.** An optional transported age concentration α of the burned gas (D.69): every burned element ages at rate
one, newly burned gas inherits the local age, and `t_fd(x, t) = α / max(c~, c_a)` replaces the global clock in D.19-D.20
where the option is on.

**Why an option.** The papers' t_fd is one global clock counted from ignition. That is right for one kernel, and it
is what H20 and H22 validated. It is wrong, or at least undefined, for two ignition points, a flame that passes into
a connected vessel, a restart, and arguably for the fresh mixture outside a vent that the venting jet ignites. A
transported age gives each kernel its own clock without any bookkeeping of kernels.

**Why OFF by default.** Turning it on changes the numerics of a validated path, and whether the outside fireball
should restart its turbulent flame development is a physics question the papers do not settle. That is the user's
call per case (docs/17 §5.3 item 4), not a default.

**Why it is safe to add.** With one ignition, the uniform clock `a = t − t_ig` is an exact solution of D.69 (D.70),
and because α and c~ share one linear discretisation, the discrete solution has that property to round-off. So:
OFF is bitwise the global clock path; ON with one ignition equals the global clock to round-off (an R-class row);
ON with two separated kernels gives each its own clock until they merge.

---

## (g) ρ_u in the propagation term is the current unburned density

**What.** In ρ_u U_t |∇c~| (D.17) and in the BML-type relations, ρ_u is the **local unburned density at the current
pressure**, `ρ_u(p) = ρ_u0 (p/p_0)^{1/γ_u}` (D.52), not the initial value.

**Why.** The mass burning rate per unit flame area is ρ_u S. In a closed vessel the fresh gas is compressed by the
pressure rise. At the AICC pressure ratio 8.57 its density has risen by a factor of 4.6 (γ = 1.4). A constant ρ_u
would under-burn the late stage by that factor and would break the thin-flame identity D.54 and the cube-root law
D.55. In the published validations the pressure rise was small (Leeds: negligible, H20 §3.2; REMBE closed stage:
≤ 0.1 bar), which is why the papers write ρ_u as a constant. The 20-L and 1-m³ sphere gates go to full pressure.

**Gate.** DST-08's thin-flame rows (they fail by tens of percent with a constant ρ_u).

---

## (h) The closed-vessel reference is the thin-flame model, used where it is exact

**What.** The spherical closed-vessel gate (DST-08) runs the TFC operator with a prescribed burning velocity
U_t ≡ S_u(p) and a negligible D_t, in the constant-c_p, constant-W prescribed thermo, and compares p(t) against the
thin-flame equation D.54 integrated on the host with the same S_u(p).

**Why.** For ideal gases with one constant γ and one W, the internal energy of the whole charge is `pV/(γ−1)` plus
the chemical energy released, so `p − p_0` is exactly proportional to the burned mass (D.51 is exact, not an
approximation). The isentropic compression of the fresh gas (D.52) is exact for an adiabatic thin flame. D.53-D.54
then follow exactly (BKM24 Eq. 6.10, CC BY). The comparison tests the coupling: spherical geometry, the sealed p_0(t),
compression of the fresh gas, ρ_u(p), and the AICC end state. It does not test laminar chemistry, which DST-04
covers separately. Prescribing U_t avoids resolving a 0.08 mm laminar flame in a 0.17 m vessel. The resolved
laminar-limit flame needs about 200 k cells and 4e5 steps there, far over the 10-minute rule. The cube-root law
(D.55) then follows: K_st must be the same in 20 L and 1 m³ (Dahoe et al. 1996: valid while δ/R < 1 %).

---

## (i) REDUCED gates on statistically symmetric 1-D meshes

**What.** The Leeds bomb, the laminar spheres and the REMBE closed stage run on a 1-D spherical-sector mesh: a thin
pyramid whose four planar side faces pass through the origin with slip conditions. Face areas then grow as r², and
cell volumes are exact for the polyhedra. The full 3-D runs are recorded runs (R), deferred.

**Why.**
- Leeds: central spark, statistically homogeneous isotropic turbulence held stationary by the fans. The mean flame is
  statistically spherical, which is how H20 verified its own code (§3.1).
- Laminar spheres: exactly spherical.
- REMBE closed stage: the pressure depends only on the burned mass and the vessel volume (D.51). The burning rate
  depends on the shape only once the walls truncate the kernel. At 0.1 bar overpressure the burned-mass fraction is
  about 0.5 % and the kernel radius about 0.46 m. For any plausible cylinder of 11.5 m³ (L/D from 1 to 3, inner
  radius 0.85-1.2 m) that is still clear of the walls, so an equal-volume sphere (R = 1.40 m) carries the same
  p(t) up to rupture. The vented stage is truly 3-D and stays 3-D (DST-11).
- All of these verify in seconds to minutes, inside the 10-minute rule with sizes reduced and tolerances not.

---

## (j) ρ-weighted k-ω SST, opt-in; SST-sust for the fan-stirred bomb

**What.** A conservative ρ-weighted SST-2003 (D.40-D.47) with the dilatation terms of the production (D.42), selected
by the DUST cases. With ρ ≡ 1 its matrices are bitwise the existing kinematic SST, and every existing case is
untouched. The sustaining terms of SST-sust (D.48) are an option, used for the Leeds bomb.

**Why.**
- H22 found k-ω and SST best for the vented stage (§4), and the user chose SST.
- Across a flame front ∇·u is large (thermal expansion by a factor of σ ≈ 5-7). The terms −(2/3)ρk∇·u and
  −(2/3)μ_t(∇·u)² change k in the brush, and the conservative ρ form is required for ρ-weighted transport in a flow
  whose density changes by σ. COMP did this for SA (CMP-25) and listed ρ-weighted SST as backlog, which DUST now
  takes up.
- Leeds: H20 used k-ε with the sustaining sources ρε_0 and ρε_0²/k_0 to keep the fan turbulence stationary. SST-sust
  does the same for k-ω (TMR), with k_amb = k_0 and ω_amb = ε_0/(β* k_0). The FSC model sees only D_t,∞, u' and L.
  With the same frozen upstream state those are the same inputs, so the different turbulence model does not change
  what the Leeds gate tests.
- Opt-in keeps docs/17 §5.3 item 4: no existing numerics path changes.

---

## (k) Pr_t, κ and the other unstated or inconsistent inputs are case inputs, stated once

The papers disagree or are silent on several inputs (RESEARCH §K): Pr_t (0.4 vs 0.28), p_0 for Leeds (110 000 Pa),
the air ratio (3.71 vs 3.76), the starch a5, κ and its law, σ_t of the ignition source, the t_fd origin, and the REMBE
geometry. The stream fixes each one in SPEC-LIT §151 with its reason, uses each case's own published value where
one exists, and never tunes any of them to pass a gate. A gate that needs an input no open source gives is **named
and not run** (REMBE vented, until the vessel geometry is obtained; the Phase B flame against a real starch, until
open devolatilisation kinetics are found).

Two of the published validations were partly calibrated, and the gates say so. H22 "guessed and then adjusted" u' and
L to match the closed stage, and tuned t_flacc = 0.15 s for the vented peak. With those published inputs, the
closed-stage row is a reproduction row (our implementation of the same model reproduces the measured curve within
the published model's own deviation plus the reading uncertainty), not an independent prediction. The Leeds data
(H20) used untuned A = 0.4 and a fitted Pr_t; the slope dS_t,b/du' is the least-tuned quantity, and it is gated.

---

## (l) Phase B: Euler–Lagrange reacting dust, built on the parcel stack

**What.** Parcels carry moisture, volatiles, char and ash. They heat up (D.61 with Ranz–Marshall, D.62, and the Bird
correction, D.63), release moisture (§76), devolatilise by a single first-order rate (D.64) into a named volatile gas
species, and burn char at a kinetic–diffusion-limited rate (D.65). The gas phase burns the volatiles with the
CHEMRAD finite-rate machinery. Radiation optionally couples to P1 (D.67). Two-way coupling uses §68/§77 and CHR-26.

**Why.** The equivalent gas assumes the dust is fine and volatile enough that particle heat-up and devolatilisation
are fast against the flame (H22 §1, citing the FLACS-DustEx premise). It fails for coarse particles, low-volatile
dusts, strongly non-uniform or settling clouds, and near flammability limits. Phase B covers those cases with the
parcel machinery tranche 17 already finishes (F32-12 f64 positions, CHR-25 case input, CHR-26 parcels in
`ofgpu-lowmach`).

**Honest limit.** No open source for corn starch devolatilisation kinetics (A, E_a) was found in this pass. Phase B
therefore gates what is exact: the single-particle closed forms (D.66), the conservation identities, the AICC end
state, and the fine-particle limit, in which Phase B must approach the Phase A behaviour of the same volatile gas. The
comparison with Dahoe's measured S_L is **named and not run** until open kinetics are found.

**Not planned (named):** non-uniform φ (a mixture fraction beside c~). The Zenodo record's title says
"premixed/partially-premixed", but the open papers describe only uniform φ. Also not planned: metal dusts,
agglomeration and fragmentation, and dust lifting from layers (secondary explosions).

---

## (m) No MPI, no multi-GPU

The user deferred MS-MPI and a second GPU. DUST plans none of it. The DUST drivers refuse decomposition by name, as
the compressible, chemistry and radiation drivers do (docs/17 A16), so `src/dust/` is outside PAR-09's allocation
audit, and PAR-14's refusal list names the dust model.
