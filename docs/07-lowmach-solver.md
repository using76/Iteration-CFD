# The low-Mach solver — `ofgpu-lowmach`

**meteor-cfd — SPEC-LIT sections 25 and 26, wired into one driver**

주식회사 메테오시뮬레이션 · 2026-08-28

This note states what `ofgpu-lowmach` actually solves, the assumptions it makes
(stated, not hidden), and the wall-heat-transfer gate record that stands behind
it. It is a companion to [`rust/SPEC-LIT.md`](../rust/SPEC-LIT.md) §25/§26, not
a replacement for it — read that document for the derivations and citations;
this one is the map of how the pieces fit together in the running driver.

> **What is in here, and what is not.** §1 is the formulation and §1.1 is the
> wall-heat-transfer gate record, and they are the whole of this document.
> `bin/validate.rs` alone names it sixteen times, and `xref.rs` resolves every
> one of those citations against the headings below rather than excusing them
> (`rust/SPEC-LIT.md` §80.3), so a heading number here is an address and not a
> label. Every measurement below was taken through the low-Mach loop on a
> general plane channel, and §1.1's last subsection re-measures the whole gate
> table under `ofgpu-lowmach` at 40 000 iterations on both legs.

---

## 1. The formulation

> **Which binary this section and §1.1 are about.** Everything below —
> the low-Mach formulation, §29.3's wall heat transfer, and the whole gate
> record of §1.1 — is the driver `ofgpu-lowmach`, and every command line
> below names it. Some of these measurements were taken while the driver
> carried another name; the identity of the loop across that rename was
> checked rather than assumed — the two binaries printed identical residual,
> bulk-state, wall-flux, thermostat, budget and friction lines on
> `cases/channelPeriodicFluxWF.jsonc`, and §1.1's table below was re-measured
> under the new name at the full 40 000 iterations on both legs.

A strongly heated flow is buoyancy-driven and low-Mach with density ratios of
3–4 (1173 K against a 293 K ambient), so the Boussinesq approximation does
not apply (SPEC-LIT §9) and acoustics have to be filtered out of the pressure
field explicitly rather than solved for and discarded. The formulation is
Rehm & Baum (1978), read from `reference/fds` for the shape of the
bookkeeping and rebuilt entirely out of this crate's own, already-validated
finite-volume operators (SPEC-LIT §3, §13).

The pressure splits into a spatially uniform thermodynamic part and a
hydrodynamic perturbation the momentum equation actually sees:

```
p(x, t) = p0(t) + p~(x, t),        p~ << p0
rho      = p0 / (R_s T)                                  (SPEC-LIT §25)
```

Continuity combined with the energy equation gives a divergence CONSTRAINT
rather than the usual `div(u) = 0`:

```
div(u) = Q / (rho cp T) - (1/(gamma p0)) dp0/dt          (§25.1)
Q      = q'''_c + div(k_eff grad T) - div(q_r)           (§25.1, §26)
```

`Q = 0` in a sealed, unheated box recovers `div(u) = 0` exactly — the
incompressible limit is a special case of this equation, not a different
code path. SIMPLE/PISO change in exactly ONE place for this: the pressure
equation's source gains the target divergence
(`Simple::correct_outer_low_mach`, §25.3), and every convective flux becomes
a MASS flux `rho_f phi` rather than a volumetric one.

`p0` itself evolves by integrating the constraint over the domain:

```
sealed:  dp0/dt = (gamma / V) * ((gamma - 1)/gamma * integral(Q) dV)     (§25.2)
open:    p0 = const, dp0/dt = 0
```

The energy equation is temperature-form sensible enthalpy:

```
rho cp [dT/dt + div(u T) - T div(u)] = div(k_eff grad T) + q'''_c - div(q_r) + dp0/dt
k_eff = k + rho cp nu_t / Prt                                            (§26)
```

A model with a volumetric heat term never touches this equation's assembly
directly — it registers through `EnergySources`
(`register_explicit`/`register_implicit_sink`), the same registry a plain
volumetric heater uses (§18). The energy module does not know it exists.

### 1.1 Wall heat transfer (§29.3)

`T` supports plain fixed-flux and fixed-temperature walls (the generic §4
Robin triple every scalar in this crate uses, with `flux_to_grad` doing the
one conversion a fixed-flux wall needs), plus a `thermalWallFunction` patch
type that applies Jayatilleke's (1969) sublayer-resistance correction to the
thermal log law. A wall-function mesh's first cell sits in the log layer, so
a plain molecular-only Robin condition there overpredicts the wall's thermal
resistance for the same reason `nut` needs a wall model at all — an earlier
revision of this driver noted the gap and left it unfixed; `thermalWallFunction`
is that fix.

`set_thermal_wall` wires the faces `T`'s own patch type names (SPEC-LIT
§15.5's rule, extended to `T`) onto the same near-wall machinery
`crate::wallfunctions` already gives `nut`/`epsilon`/`omega`, and
`Energy::correct` refreshes the Robin triple every outer iteration, right
after `k_eff` is updated and before the matrix is assembled. Selection
follows the `wallTreatment` preset table of SPEC-LIT §29.1: every row
applies `thermalWallFunction` to `T` on walls except `lowRe`, which pins the
molecular resistance a resolved (`y+ ~ 1`) mesh already provides and needs no
correction for. A case that never asks for a wall model keeps `T`'s wall
exactly as before — adiabatic (`zeroGradient`) by default, or whatever
fixed-flux/fixed-temperature value it names.

Two permanent `ofgpu-validate` gates stand behind the correction itself:
`P(Pr/Pr_t = 1) = 0` exactly (the correction vanishes when the fluid and
turbulent Prandtl numbers coincide, whatever the log-law argument is), and
the Robin triple `thermalWallFunction` writes encodes EXACTLY the analytic
Jayatilleke flux `rho cp u_tau (T_w - T_P)/T+` for a one-cell energy
balance, to round-off. What these do not by themselves establish is that a
coarse wall-function mesh and a resolved low-Re mesh agree on the wall heat
flux of one physical flow — the full claim SPEC-LIT §29.3's own table asks
for needs a converged 3-D turbulent-channel run at two mesh resolutions.

**The gate: CLOSED for the wall-function leg (SPEC-LIT §32's redesigned
gate).** Three reruns of a fixed-wall-temperature version of this gate (kept
below, collapsed, for the record) produced ratios of 0.095, 0.381 and 0.107
without converging on anything meaningful. SPEC-LIT §32.1 names the reason in
one line, and it explains all three at once: **a fixed wall temperature lets
the bulk temperature float**. In a periodic domain the bulk temperature
drifts until the wall heat balances whatever sink holds the energy budget, so
two meshes that predict different near-wall conductances settle at different
ΔT — the third rerun's own trace found driving temperature differences of
about 50 K on the wall-function mesh and 3 K on the resolved mesh — and
comparing `q_w = h·ΔT` between them compares two products in which BOTH
factors differ. **The two runs solved different problems**, at every
resolution tried; no ratio between them could have meant anything.

SPEC-LIT §32.2 redesigns the gate around that diagnosis instead of tuning it:
impose the SAME `q_w` on both meshes (`fixedFluxTemperature`, the generic §4
Robin triple's existing `flux_to_grad` translation — the identical condition
a resolved mesh already used, extended to the wall-function mesh too), let
each mesh predict its OWN ΔT, and compare the result as a dimensionless
Nusselt number against two independent, published pipe-flow correlations —
Dittus & Boelter (1930) and Gnielinski (1976) — rather than against another
run of this code. That is the same shape every other validation in this
project has (SPEC-LIT §10, §22).

> **Superseded below.** Everything from here down to *"The gate, rerun on the
> settings the cases actually ask for"* — the last subsection of §1.1 — was
> measured with a UNIFORM thermostat sink, judged against a friction factor
> INFERRED from the case's own body force, and produced by a driver that did
> not read the case's own `numerics` block at all (a SPEC-LIT §13.4
> violation: the momentum equation ran `bounded Gauss upwind` on two cases
> asking for `Gauss linearUpwind grad(U)`). All three were defects in the
> comparison, all three are now fixed, and both legs have been rerun after
> each. The trace is kept in full, and in order, because the history of this
> gate is the instructive part of it; the current numbers and the current
> verdict are at the end.

**Rebuilt per SPEC-LIT §34 as a genuine 2-D plane channel.** §34.1 gave JSONC
an `empty` patch kind; the original version of this gate was a 3-D DUCT
(`wallSide.*` real walls on 4 spanwise cells) only because JSONC could not
say `empty` yet. `cases/channelPeriodicFluxWF.jsonc` is now
streamwise-cyclic, `empty` front/back, hot walls top and bottom, and
nothing else — `standard` wallTreatment, y+ target 30–60 — and was rerun,
not carried over: it converged to a bit-identical fixed point
(`\|U\|` residual `1.4×10⁻¹⁰`, `T` unchanged in its last four decimal
places from iteration 5 000 through 40 000):

| Quantity | Value |
|---|---|
| `q_w` (imposed, both hot walls) | 500 W/m² |
| y+ (min / mean / max) | 56.80 / 57.69 / 58.50 |
| `T_w` (diagnosed by the thermal wall function) | 316.861 K |
| `T_b` (mixed-mean) | 292.92 K |
| `ΔT = T_w − T_b` | 23.941 K |
| `U_b` | 5.3696 m/s |
| `rho(T_b)` | 1.20504 kg/m³ |
| `k_thermal = rho·cp·nu / Pr` | 0.025611 W/(m·K) |

For a genuine plane channel the heated-perimeter and wetted-perimeter
conventions COINCIDE — both walls are hot, there is no third or fourth wall
to argue about — so `D_h = 2H = 0.08 m` (`H` the full 0.04 m gap) is the only
number on the table, not a choice between two:

| `D_h` | Re | Nu (measured) | Dittus-Boelter | Gnielinski |
|---|---|---|---|---|
| **0.08 m (= 2H)** | **28 638** | **65.24** | 73.75 (**−11.5%**) | 68.33 (**−4.5%**) |

Both inside their own stated band (Gnielinski ±10%, Dittus-Boelter ±20–25%):
**inside both bands**. SPEC-LIT §32.4's own verdict rule — "the gate closes
when both meshes sit inside the correlation band" — is met for the
wall-function leg, on the corrected geometry.

<details>
<summary>Superseded: this leg's Reynolds-analogy verdict, taken at an inferred friction factor (kept for the record)</summary>

*Which `f`, added later.* That Gnielinski number is evaluated at Petukhov's
smooth-PIPE friction factor, so it is an ABSOLUTE-PREDICTION verdict in the
sense SPEC-LIT §32.4 now requires every verdict to declare. This leg passes
the REYNOLDS-ANALOGY verdict as well — Gnielinski at the `f` this leg itself
realises, 0.02162, gives Nu_Gn = 61.30, so +6.4% — see the two-verdict table
further down, and §32.5.3 for why the friction factor quoted there is still
an inference from the body force rather than a measurement at the wall.

</details>

**No side-wall-drag caveat needed this time.** The force balance gives
`u_tau = sqrt(g_x·H/2) = 0.2793 m/s` (`g_x` the 3.9 m/s² body force and `H`
the 0.04 m gap — written `f` in an earlier revision of this line, renamed
here because `f` is the friction factor everywhere else in this section,
not because the number changed), `Re_tau = 372`, and `U_b/u_tau = 19.23`
against the 15–17 a fully developed plane channel gives — close enough to be
unremarkable (an integrated log-law estimate at this same `Re_tau`, `(1/κ)
ln(Re_tau) + B − 1/κ`, gives ≈17.6; standard k-epsilon's coarse
wall-function treatment on only 6 wall-normal cells accounts for the
remaining few percent), and there is no longer a third or fourth wall left
to blame for anything.

`ofgpu-validate` replays this rebuilt measurement permanently
(`check_thermal_wall_function_gate_verdict_replay`, SPEC-LIT §32/§34) — it
feeds the numbers above through `dittus_boelter_nu`/`gnielinski_nu_at_f` and
asserts the bands under BOTH of §32.4's verdicts, each named, so a future
regression in the thermal wall function, `Energy`, or the SIMPLE loop is
caught on every commit, not only on the next multi-second re-run someone
remembers to do by hand.

That check is one of six families in `ofgpu-validate` that judge a RECORDED
measurement rather than something the binary computed on the spot (the others
are the resolved leg's mesh resolution, its gate verdict, the
thermostat-weighting experiment, the bounded-convection isolation, and — since
SPEC-LIT §37 — the Kays-Crawford `Pr_t` experiment). The summary line splits
them out — today `699 computed live, 48 replayed from recorded measurements`
— so the headline `N/N checks passed` means one thing only. Everything those
six then DO with their frozen inputs — the correlations, the friction-factor
conversions, the band arithmetic — is computed live.

### The resolved leg — rebuilt as a plane channel (SPEC-LIT §34), OPEN for a
third and different reason

<details>
<summary>Superseded: the 3-D duct version of this case (kept for the record
— the corner hypothesis this history ends on is what SPEC-LIT §34's rebuild
below tests, and confirms)</summary>

**Superseded trace (standard k-epsilon, before `LaunderSharmaKE` existed).**
`cases/channelPeriodicFluxLowRe.jsonc` did **not** converge — `k` ran away to
336 m²/s² at the hot wall (12–18 m/s of turbulent fluctuation superimposed on
a 1–2 m/s mean flow). Traced in three steps, all reproducible:

1. Switching the model to `kOmega` made it WORSE (`k` → 942 m²/s²) — the
   model FAMILY was not the cause.
2. The case is a DUCT: its side walls have only 4 spanwise cells and are not
   resolved, yet the case-wide `lowRe` preset told the solver to apply NO
   wall model there. A per-patch `"treatment": "standard"` override on the
   side walls (SPEC-LIT §29.1 route b) brought them to heel — side-wall `k`
   fell from 4125-y+ nonsense to 5.85 m²/s².
3. Even then the HOT walls still blew up (`k` = 160 m²/s² at y+ 1.4–6.4).
   That was the real cause: standard k-epsilon is a HIGH-REYNOLDS-NUMBER
   model with no near-wall damping function, invalid below y+ ~ 30 whatever
   the mesh does.

`crate::io::case::validate_low_re_wall_treatment` catches this before the
solver ever runs: `wallTreatment lowRe` under a model with no low-Reynolds
validity is a SPEC-LIT §13.4 error naming the model and the menu of ones
that qualify; `-permissive` substitutes `standard` and says so.

**`LaunderSharmaKE` has since landed (SPEC-LIT §33), and it is on that menu
— but this specific (duct) case still did not close the gate.** Two separate
questions, checked separately, with two different answers:

**1. Is the model itself right? Yes — checked on a clean channel, not on
this duct.** A wall-resolved, streamwise-periodic, genuinely TWO-DIMENSIONAL
channel (`empty` front/back — no side walls to confound anything), 8×150×1
cells, `y` graded two-sided (`expansion 20`) to a first cell at y+ ≈ 0.46,
driven by a uniform body force to Re_tau ≈ 437 (`u_tau` = 0.9113 m/s from
the force balance `tau_w = rho f H`, `nu` = 2.0838×10⁻³ m²/s, `H` = 1 m —
these are abstract units chosen only to land Re_tau usefully high, not a
physical duct), run through `ofgpu-lowmach` with `LaunderSharmaKE`/`lowRe`,
heater off, walls adiabatic (no energy complication needed for a pure
velocity check):

| Region | Check | Result |
|---|---|---|
| Viscous sublayer, y+ < 5 | `u+ = y+` | reproduced to **< 1%** (worst deviation 0.79% at y+ = 4.41; 0.002% at y+ = 0.46) |
| Log layer, y+ ≈ 30–35 | `u+ = (1/κ) ln(E y+)`, κ=0.41, E=9.8 | within **1%** (−0.8% at y+=30.6, +0.01% at y+=35.1) |
| Outer layer, y+ ≈ 90–300 | departs from the log law | grows to +6% by y+≈300, then falls back approaching the centreline — the textbook wake departure, not a defect |

`u_tau` computed independently two ways — from the domain-wide force balance,
and from the first cell's own molecular wall shear (`nu_t,w = 0` exact under
`lowRe`, so `tau_w = rho nu U_1/y_1`) — agree to 4 significant figures
(0.9113 m/s both ways), which is itself a strong internal consistency check.
The run took ~627 s for 40 000 iterations and had not fully converged even
then (`|U|` residual plateaus around 5×10⁻² — the periodic pressure
equation's own null space, SPEC-LIT §31.1, not the turbulence quantities
still moving); the profile shape above is unambiguous regardless. **This is
the law-of-the-wall check SPEC-LIT §33.3 asks for, and the damping functions
pass it.**

**2. Does `cases/channelPeriodicFluxLowRe.jsonc` itself (the duct) converge
to a usable state? No.** Run as shipped (`wallSide.*` overridden to
`standard`, hot walls `lowRe`, `-heaterPower -3.2` — the closed-form
`-2 q_w A_wall` compensating sink SPEC-LIT §32.2 derives, identical to the
wall-function leg's own), 20 000 iterations:

| Quantity | Value |
|---|---|
| `y+` at the hot walls (min / mean / max) | 0.00102 / 0.00181 / 0.00261 — far inside the y+ < 1 requirement |
| `y+` at the side walls (`standard`, min / mean / max) | 1.31 / 43.2 / 65.7 — in the wall-function leg's own 30–60 regime |
| cells globally at y+ < 20 (approximate — hot-wall distance only, not a true nearest-wall Poisson field) | 524 / 1600 (33%), comfortably past SPEC-LIT §33.2's "at least 10" |
| `U_b` | **0.243 m/s** |
| `T_b`, `T_w` at 20 000 iterations | 294.9 K, 306.1 K — still rising, not converged |
| `Nu` from this (non-converged) snapshot | ≈ 140 — **2.8× the wall-function leg's 50.41**, outside both correlation bands |

The mesh itself is not the problem: a **laminar** solve of the identical
mesh and boundary conditions gives `U_b` = 14.76 m/s, matching the
hand-derived square-duct laminar solution (`U_b = 2fH²/(f_D Re) ≈ 14.6 m/s`,
Shah & London's `f_D Re = 56.91`) almost exactly. The wall-function twin,
`channelPeriodicFluxWF.jsonc`, gets `U_b` = 3.51 m/s on the identical body
force. So `LaunderSharmaKE` on this duct is suppressing the flow to roughly
1/60th of the laminar answer and 1/14th of the wall-function answer — not a
divergence (no NaN, no runaway `k`; the model's own damping is doing
something), but not a usable flow either.

Three follow-up runs narrow down what it is not:

* **Not the wall-row mismatch.** All-`lowRe` (removing the `standard`
  override on the side walls) gives `U_b` = 1.54 m/s — still collapsed, and
  the side-wall y+ balloons to 52–239 (the OLD "unresolved `lowRe`"
  symptom returns, as expected, since those 4 spanwise cells still are not
  resolved) — a different secondary failure, same underlying collapse.
* **Not the `lowRe` Dirichlet condition on its own.** All-`standard` (every
  wall, including the hot ones, using the SAME row `channelPeriodicFluxWF`
  uses — the only remaining difference from that working case is the
  finer mesh) gives `U_b` = 0.0119 m/s — **worse**, ruling out "`k = 0`,
  `epsilon_tilde = 0` at the wall" as the specific cause, since removing it
  does not help.
* **Not the grading's aspect ratio.** Repeating the shipped configuration
  with `expansion: 20` instead of `200` (a far less extreme near-wall cell)
  gives `U_b` = 0.2433 m/s — indistinguishable from the original.

What DID distinguish this case from the clean channel that passed the law
of the wall above was the one thing every variant here shared and the clean
channel did not: **a third pair of real walls** (the duct's side walls,
`z`, present as an actual boundary rather than `empty`) using `LaunderSharmaKE`
at the same time — the leading, untested lead was SPEC-LIT §33.1's own
DESIGN note: the `E` term's `grad(grad U)` needs a boundary extrapolation of
`grad U`, which carries no boundary field of its own, and a DUCT CORNER cell
(where the hot-wall boundary layer and the side-wall boundary layer meet,
with no `empty` direction to fall back on) is exactly where that
extrapolation has the least to work with. SPEC-LIT §34's rebuild below
removes that corner entirely and tests the lead directly.

</details>

**SPEC-LIT §34 rebuild: a genuine 2-D plane channel.** `empty` in JSONC
(§34.1) means the corner hypothesis above can finally be tested rather than
argued from a law-of-the-wall run on a *different* case. `cases/
channelPeriodicFluxLowRe.jsonc` is now streamwise-cyclic, `empty` front/back,
hot walls top and bottom — 8×50×1 cells, `y` graded two-sided (`expansion
200`) toward both hot walls — the SAME q_w = 500 W/m² and the SAME
closed-form `-2 q_w A_wall = -3.2 W` sink as the wall-function twin, so the
two differ only in mesh and treatment. Rerun, not carried over:

**1. The velocity collapse is FIXED — the corner hypothesis is confirmed.**
`\|U\|` converges to round-off (`2.3×10⁻¹²` at 40 000 iterations, was
plateauing at `5×10⁻²` on the duct) and stays there for the rest of the run.
Sanity, checked before anything else (SPEC-LIT §34's own order):

| Check | Result |
|---|---|
| `y+` at the hot walls (min / mean / max) | 0.00175 / 0.00175 / 0.00175 — far inside y+ < 1 |
| Cells globally at y+ < 20 (`ofgpu-lowmach`'s own §33.2 report, a true Poisson wall distance this time, not the duct's hot-wall-only approximation) | 192 / 400 (48%), comfortably past "at least 10" |
| `U_b` | 4.846 m/s — BELOW the laminar closed form (plane Poiseuille, `U_b = f H²/(12 nu)` = 34.67 m/s) by a factor of 7.2, and below the wall-function twin's 5.37 m/s, both sane |
| `U_b / u_tau` | **17.35**, against the 15–17 a fully developed plane channel gives — the target the duct version could not reach (12.6) even approximately, now met almost exactly |

Both mandatory sanity gates pass, and pass more convincingly than the
wall-function leg's own 19.23 does — the resolved model's velocity profile
is, if anything, the more faithful of the two.

**2. The energy equation's drift, diagnosed as SPEC-LIT §35 and fixed with a
bulk-temperature thermostat.** The finding below (kept in full, collapsed,
for the record) turned out to have a one-line cause: a closed,
streamwise-periodic domain whose every thermal boundary is Neumann (fixed
flux walls, cyclic streamwise, `empty` front/back) has a steady temperature
equation that is pure Neumann and singular up to an additive constant —
exactly the null space §8.5 already zeroes for a pure-Neumann pressure
Poisson problem, read for `T` instead of `p`. Two initial temperatures
(293.15 K, 400 K) on the SAME case converged to different bulk states
(291.96 K / 396.37 K, `Nu` 74.0 / 86.0) — the level simply kept whatever it
was given, and the "undamped drift" below was that same free constant being
carried, slowly, through the low-Mach `rho(T)` coupling.

SPEC-LIT §35.1's fix is a proportional controller on the domain's own
VOLUME-mean `T` (not the mixed-mean `Nu` uses): `q_thermostat = -rho_cp
(T_mean - T_target)/tau`, uniform, registered as a `sources[]` entry
(`crate::sources::Thermostat`) exactly like the heater it replaces, but
recomputed from the CURRENT `T` field every outer iteration instead of
fixed once at start-up. It removes the null direction without imposing a
value at any point of the boundary. Both channel cases now carry
`{"type": "thermostat", "target": 293.15, "tau": 0.02}` in place of the old
`-heaterPower -3.2`, targeting this case's own `TRef` so the
already-recorded wall-function measurement above is REPRODUCED rather than
moved.

**Correction, added later.** An earlier revision of this paragraph — and of
SPEC-LIT §35.1 itself — went on to claim that "the profile stays entirely
`Energy`'s own prediction, only the offset is pinned". That is FALSE, and
SPEC-LIT §35.3 now derives why. A UNIFORM volumetric sink is the slug-flow
limit of the compensating source a streamwise-periodic constant-flux duct
actually calls for, which is proportional to the LOCAL streamwise mass flux
`rho u . e_hat` (Kays & Crawford ch. 9; Patankar, Liu & Sparrow 1977). Against
the correct distribution the uniform form removes too much heat where
`u . e_hat < U_b` — the near-wall layer — and too little in the core, which
shrinks `(T_w - T_b)` and biases `Nu` HIGH. Every number in this subsection
was measured with the uniform form and stands exactly as recorded; the
weighted form (`"weighting": "massFlux"`) is an explicit opt-in and the
default is still `uniform`, precisely so that these numbers stay
reproducible bit for bit. **Nothing below has been rerun with it.**
*(Both legs HAVE since been rerun with it, and both case files now name it
explicitly — the last subsection of §1.1 has the experiment and what it
found. The paragraph above stands as written because the numbers below are
still the uniform-form numbers it describes.)*

**The regression this whole diagnosis needed** (SPEC-LIT §35.2): rerunning
`channelPeriodicFluxLowRe.jsonc` from T0 = 293.15 K and T0 = 400 K, 40 000
iterations each, now converges to the IDENTICAL state from either start —
`T_mean` = 293.574 K, `T_b` = 292.817 K, `U_b` = 4.84388 m/s, thermostat
power = −3.28977 W, every one of these to the last printed digit. The drift
itself is gone, not merely slower: `T` is bit-identical (`T` ∈ [290.421,
313.879] K) from iteration 5 000 through 39 999, where the pre-thermostat
run was still climbing linearly at 150 000.

**Energy balance** (SPEC-LIT §35.2's own check): the thermostat's
integrated power should equal the wall heat input to round-off. On the
wall-function leg it does — power −3.2 W against 3.2 W measured, difference
2.8×10⁻⁷ W. On the resolved leg it does not, quite: power −3.28977 W
against 3.2 W measured, a 2.8% gap. Traced, not shrugged off: `contErr`
(the SIMPLE loop's own continuity residual) plateaus at 9.2×10⁻⁸ on this
mesh — identical from either initial temperature, so a property of the
converged fixed point, not a still-moving transient — and tightening `p`'s
`relTol` from 0.01 to 10⁻⁴ **diverges** the run at iteration 3 317 (NaN):
this two-sided-graded mesh (`expansion: 200`) needs the looser tolerance to
stay stable, and the residual it leaves feeds into the bounded convection
scheme's own `div(phi)` correction term. Reported as a real, small,
mesh-conditioning-limited imbalance, not tuned away.

**The gate, rerun at the same bulk state for the first time:**

| Mesh | `D_h` | Re | `T_w` | `T_b` | Nu (measured) | Dittus-Boelter | Gnielinski |
|---|---|---|---|---|---|---|---|
| (a) wall-function, `standard` | 0.08 m | 28 638 | 317.253 K | 293.283 K | **65.24** | 73.75 (**−11.5%**) | 68.33 (**−4.5%**) |
| (b) resolved, `LaunderSharmaKE`/`lowRe` | 0.08 m | 25 834 | 314.087 K | 292.817 K | **73.40** | 67.92 (**+8.1%**) | 63.10 (**+16.3%**) |

Two-mesh ratio `Nu_b / Nu_a` = **1.125**.

**Verdict: the wall-function leg CLOSES on both correlations (unchanged
from the rebuild above). The resolved leg closes on Dittus-Boelter's wider
±20–25% band but sits 6.3 points outside Gnielinski's tighter ±10% band —
the gate does NOT close on both legs, under SPEC-LIT §32.4's own rule.**
This is a categorically smaller and different question than the one the
thermostat answers: with the domain-mean bookkeeping fixed and both legs
now genuinely comparable at the same, stable, reproducible bulk state, a
±16% single-correlation miss most plausibly implicates Launder-Sharma's own
near-wall THERMAL prediction, which nothing in this project has
independently validated the way SPEC-LIT §33.3's law-of-the-wall check
validated its MOMENTUM prediction — not another energy-accounting defect,
and not the +31%/+41%-and-still-drifting failure this section used to
report.

<details>
<summary>Superseded: the two-verdict tables and the two-mesh decomposition, both taken at a friction factor INFERRED from the body force (kept for the record — the measurement that retired them is in the last subsection of §1.1)</summary>

**Which `f` that verdict was judged at — added later, and it changes what
the verdict means.** Every Gnielinski number in the table above was
evaluated at Petukhov's SMOOTH-PIPE friction factor
`f = (0.79 ln Re − 1.64)^-2`. Gnielinski is explicitly a function of `f`,
and the case is a plane CHANNEL, not a pipe: parallel plates run a
measurably higher `f` than a pipe at the same `Re_Dh` (Jones, *ASME J.
Fluids Eng.* 98 (1976) 173). Supplying each leg's own `f` instead gives a
different — and weaker, and clearly labelled — verdict:

| Leg | `f` realised | vs Petukhov pipe `f` | Nu_Gn at pipe `f` | Nu_Gn at realised `f` |
|---|---|---|---|---|
| (a) wall function | 0.02162 | **−9.6%** | 68.33 (**−4.5%**) | 61.30 (**+6.4%**) |
| (b) resolved | 0.02653 | **+8.2%** | 63.10 (**+16.3%**) | 68.72 (**+6.8%**) |

SPEC-LIT §32.4 now requires every band statement to name the `f` behind it,
because these are two different claims:

* **Absolute prediction** (pipe `f`): from `Re` alone, is the heat transfer
  right? **The resolved leg fails this, by +16.3% against ±10%. The gate
  remains OPEN, exactly as stated above. Nothing below changes that.**
* **Reynolds analogy** (realised `f`): given the momentum this model
  actually transports, does it transport heat consistently with it?
  **Both legs pass, at +6.4% and +6.8%.** This is the weaker claim — it is
  handed one of the two quantities instead of predicting it — and it may
  never be quoted as an absolute-prediction pass.

**Every friction number above is an INFERENCE, not a measurement.** It comes
from the steady force balance on the case's own `bodyForce` (3.9 m/s² per
unit mass, `rho_bar` from the recorded thermostat power through SPEC-LIT
§35.1's law), i.e. from a case input plus a recorded `U_b` — not from the
wall. SPEC-LIT §32.5.1 now specifies the direct measurement (the wall-face
viscous traction, in whichever of two forms is correct for each leg's own
wall treatment), `ofgpu-lowmach` implements it and prints both `f` estimates
side by side with their disagreement, and `ofgpu-validate` checks the
measurement live against plane Poiseuille's `f Re = 96` and against the
force balance itself. **Neither channel case has been rerun with it.**

*What the rerun should expect, so a gap is not misread.* On the RESOLVED leg
the viscous form is literally the discrete momentum sink the matrix assembled
— same `mu_eff`, same `deltaCoeffs` — so at convergence it must reproduce the
force balance, and any gap is a statement about convergence and nothing else.
On the WALL-FUNCTION leg the reported form is the wall function's own
`rho u_tau²` with `u_tau = C_mu^{1/4} sqrt(k_P)`, which is NOT the discrete
sink (that is the viscous form evaluated with the wall function's `nu_t,w`),
and the two coincide only in local equilibrium. A gap there is a finding
about how far from equilibrium the wall-adjacent cell is. `ofgpu-lowmach` prints
both forms on every wall patch precisely so that gap can be attributed
instead of guessed — SPEC-LIT §32.5.2.

A 400-iteration smoke run of a scratchpad COPY of the wall-function case
(deliberately far from converged — `|U|` residual 1.2×10⁻³ against the
1.4×10⁻¹⁰ of the recorded measurement, `U_b` 5.164 against 5.370) confirms
only that the code path executes and that the disagreement flag fires: it
reported the two friction factors 26% apart. **No number from it is quoted
anywhere as a measurement, and none should be** — it is a smoke check, not
the gate.

**What the decomposition implicates — and what it displaces.** The two legs
realise friction factors 22.7% apart at the SAME body force, because they
predict `U_b` = 4.844 and 5.370 m/s and `f ~ 1/U_b²` at fixed forcing.
Gnielinski evaluated at those two friction factors predicts a two-mesh
Nusselt ratio of **1.121**; the measured ratio is **1.125**. So the 1.125
ratio — which this section attributes above to Launder-Sharma's near-wall
thermal prediction, and which the paragraph below offers the thermostat's
distribution defect as a second candidate for — is accounted for almost
entirely by the two meshes disagreeing about MOMENTUM. If that survives a
rerun, the open question stops being "is the low-Re model's thermal sublayer
right?" and becomes "why does the resolved mesh predict a 10% lower bulk
velocity at the same forcing?", which is SPEC-LIT §33.3's territory. It is a
decomposition through a correlation, resting on an inferred `f` and no
rerun, so it is a hypothesis with a number behind it — not a finding, and
the attributions above and below stand as written until a rerun says
otherwise.

**A second candidate, added later and NOT yet measured.** The thermostat
distribution defect corrected above is concentrated in the near-wall
velocity deficit, so it is carried in FULL by the resolved leg's 50-cell
`expansion: 200` mesh and largely absent from the wall-function leg's 6
cells — which makes it a candidate contributor to the 1.125 two-mesh ratio
alongside Launder-Sharma's own thermal prediction. SPEC-LIT §35.3 specifies
the weighted form that would separate the two, and it is implemented and
unit-tested, but neither leg has been rerun with it. The attribution above
stands as written until something does. `cases/channelPeriodicFluxLowRe.jsonc` and
`channelPeriodicFluxWF.jsonc` both carry this update in their own headers.

</details>

<details>
<summary>Superseded: the undamped-drift finding SPEC-LIT §35 diagnosed and
fixed above (kept for the record — the two-initial-temperature experiment
that opens this subsection is what settled it)</summary>

**2. But the ENERGY equation does not reach a steady state — a new,
distinct, and non-obvious problem.** `T_b`/`T_w` (and, coupled to them
through the low-Mach `rho(T)` in the steady continuity equation, the
mixed-mean state generally) drift upward for as long as the run is
extended, at a rate that does **not** decay:

| Iterations | `T_b` | `T_w` | `ΔT` | `U_b` | Nu | Nu vs Dittus-Boelter | Nu vs Gnielinski |
|---|---|---|---|---|---|---|---|
| 15 000 | 298.26 K | 319.53 K | 21.27 K | 4.8462 m/s | 74.76 | +10.0% | **+18.4%** |
| 40 000 | 321.11 K | 343.19 K | 22.08 K | 4.8460 m/s | 77.55 | +14.1% | **+22.9%** |
| 150 000 | 427.17 K | 452.72 K | 25.55 K | 4.8453 m/s | 89.15 | **+31.2%** | **+41.2%** |

`U_b` (hence `Re` = 25 842–25 846) is fully converged and bit-stable across
this entire span — only the temperature LEVEL and, more slowly, `ΔT` itself
are still moving. The domain-average `T` rises by a near-perfectly constant
≈0.038 K per 100 iterations from 20 000 through the full 150 000 iterations
run (checked every 10 000 throughout; every 10 000-iteration increment falls
in 0.377–0.384 K, with no sign of decay across an order of magnitude in
iteration count) — a LINEAR, UNDAMPED drift, not a slowly-settling
transient. By 150 000 iterations `Nu` has drifted from inside
Dittus-Boelter's band to **+31% of it and +41% of Gnielinski's — outside
BOTH bands, decisively** — so this is not a case of "the first estimate was
close, later ones moved slightly"; extending the run makes the mismatch
worse, monotonically, with no indication it will stop. The mesh's own y+ and
cells-below-y+-20 counts, by contrast, are bit-identical at 40 000 and
150 000 (0.00174716 and 192/400 both times) — confirming the drift is
confined to the energy equation, not a symptom of the turbulence field
itself still evolving.

Four checks rule out the obvious suspects, none of which is the cause:

* **Not the turbulence model.** Substituting standard `kEpsilon` for
  `LaunderSharmaKE` on the IDENTICAL fine mesh (`-permissive`, since
  `kEpsilon`/`lowRe` is itself refused) reproduces the same drift almost
  exactly (≈0.89 K per 2 500 iterations, non-decaying) — this is not a
  `LaunderSharmaKE`-specific defect.
* **Not the grading severity.** `expansion: 20` instead of `200` (a far
  milder near-wall cell, y+ ≈ 0.087 instead of 0.0017) still drifts at a
  comparable rate — not simply a symptom of an extreme aspect ratio.
* **Not the T-equation's linear-solve tolerance.** Tightening `T`'s
  PBiCGStab from `relTol 0.01` to exact (`relTol 0, tolerance 1e-12`)
  reproduces the SAME trajectory to the last printed digit — the drift is
  not a leaking iterative solve.
* **Not the open/sealed domain classification.** `-sealed` changes nothing,
  because the wall heat input and the compensating sink are EXACTLY equal
  by construction (`-heaterPower` is the closed form `-2 q_w A_wall`), so
  `dp0/dt = (gamma-1) P_net / V = 0` under either classification — neither
  domain type is tracking or pinning the mean temperature here.

What DOES distinguish the two meshes is resolution alone: the coarse,
uniform wall-function mesh (48 cells) reaches a bit-identical fixed point
(unchanged to the last printed decimal from iteration 5 000 through 40 000,
confirmed out to the same 40 000 the resolved mesh was checked against), and
the fine, wall-normal-graded resolved mesh (400 cells) does not, regardless
of which model or grading severity is on it. The most likely explanation:
`q_w` and the volumetric sink are both exactly T-INDEPENDENT (a Neumann flux
and a fixed wattage, not a Newtonian or radiative loss with any restoring
term), buoyancy is off (`gravity: [0,0,0]`), and T is Dirichlet nowhere in
this domain (cyclic in x, `empty` in z, flux-only in y) — so the domain-mean
temperature has no term in the discrete energy balance that opposes a
uniform shift, and the resulting near-neutral mode is thereafter carried by
the (small, but real) `rho(T)` coupling into `U` through the steady
continuity equation. Why the coarse mesh's SIMPLE iteration nonetheless
lands on an exact fixed point while the fine mesh's does not is not
resolved by this round of work; it is reported as a genuinely new, distinct
finding, not chased to a fix.

**Verdict: the gate does NOT close on both legs, but for a different
reason than either previous round.** The wall-function leg CLOSES on the
rebuilt geometry (above). The resolved leg's velocity field is now correct
by every check available — the duct-corner hypothesis is CONFIRMED, not
merely plausible — but its Nu is not a settled number, and extending the
run does not bring it toward either correlation: it starts (15 000
iterations) already outside Gnielinski's band, and by 150 000 iterations —
a 10x extension, with `U_b`/`Re`/the mesh's own y+ all unchanged to the
displayed precision throughout — it sits at **+31% of Dittus-Boelter and
+41% of Gnielinski, outside BOTH bands, decisively**, without a
turbulence-model, mesh-grading, or solver-tolerance explanation (all four
ruled out above). This is a genuinely new, previously-invisible limitation
— a slow, apparently undamped domain-mean-temperature drift specific to a
periodic, T-Dirichlet-free, low-Mach energy equation on a finely
wall-normal-graded mesh — and it is reported rather than tuned away:
`cases/channelPeriodicFluxLowRe.jsonc` is kept exactly as run above, its
header stating this diagnosis.

</details>

<details>
<summary>Superseded: the fixed-wall-temperature gate (three reruns, four
attempts total — kept for the record; see the one-line diagnosis above)</summary>

**The gate, rerun on a periodic domain (Task C).** Both earlier attempts
(kept below for the record) named the same remaining cause: an inlet-driven
duct only ever APPROACHES the fully-developed state Jayatilleke's log law
assumes, however long it is made. SPEC-LIT §31.1 closed the gap that kept
that fix out of reach — a cyclic-patch pair a JSONC case can actually name
(`mesh.cyclic`) — and this rerun uses it: `cases/channelPeriodicWF.jsonc` /
`cases/retired/channelPeriodicLowRe.jsonc` are streamwise-cyclic instead of
inlet/outlet, so every cross-section is the SAME cross-section, not an
increasingly good approximation to one. Two things a periodic domain needs
that a developing duct does not, both closed by this task rather than left
as a further-out "next step":

* **A momentum source, in place of the inlet.** A cyclic domain has no
  boundary left to prescribe a mass flow from, so JSONC gained a `sources[]`
  array (SPEC-LIT §18/§31.1's `JsonSource`, today exactly one variant:
  `momentumSource`) reusing the SAME `crate::sources::SourceTerm::BodyForce`/
  `CellSelector::All` the OpenFOAM `constant/fvSources` route already had — a
  uniform body force per unit mass over the whole domain, calibrated (not
  derived) to `3.9 m/s²` for a ~3 m/s bulk speed at this duct's cross-section.
* **A compensating heat sink, in place of the outlet.** SPEC-LIT §31's own
  energy-balance argument: a cyclic pair contributes zero NET flux by
  construction, so a steady state needs the wall's heat input canceled by
  something else in the domain, or the domain simply heats toward `Tw`
  everywhere (the boring zero-flux equilibrium) and never reaches a steady
  state at all. `ofgpu-lowmach -heaterPower`, wired for a positive heat release,
  is reused NEGATIVE as that something — a uniform domain-wide sink, also
  calibrated by watching the run settle rather than solved for in closed
  form.

Both cases keep the earlier attempts' cross-section (0.04 m × 0.04 m),
`Tw = 373.15 K` hot walls and adiabatic sides unchanged; only the streamwise
direction and how the flow and the heat balance are driven are different.
Run:

```powershell
cargo run --release --bin ofgpu-lowmach -- ..\cases\channelPeriodicWF.jsonc    -iters 3000  -check 3000 -heaterPower -6
# The resolved leg's command is kept only as a record of what was run. The case
# is now cases\retired\channelPeriodicLowRe.jsonc and DOES NOT RUN: a later rule
# (SPEC-LIT 33) refuses `lowRe` together with `kEpsilon`. See
# cases\retired\README.md. The live resolved leg is channelPeriodicFluxLowRe.jsonc.
# cargo run --release --bin ofgpu-lowmach -- ..\cases\retired\channelPeriodicLowRe.jsonc -iters 40000 -check 5000 -heaterPower -60
```

| Mesh | Wall-normal cells | `T` on the hot walls | Measured y+ (min/mean/max) | Wall time |
|---|---|---|---|---|
| (a) `channelPeriodicWF.jsonc` | 6, uniform | `thermalWallFunction`, `standard` preset | 40.25 / 41.73 / 43.41 | 45.2 s (3 000 iters, `\|U\|` residual `1.3e-10`, fully converged) |
| (b) `channelPeriodicLowRe.jsonc` | 50, two-sided graded (`expansion: 200`) | plain `fixedValue`, `lowRe` preset | 0.302 / 0.310 / 0.318 | 479.6 s (40 000 iters, `\|U\|` residual plateaus at `8.1e-6`, still declining very slowly — see below) |

Both land inside their intended range, mesh (a) more centrally than either
earlier attempt (40–43 against a 30–60 target) and mesh (b) closer to a true
`y+ ≈ 0` limit than either earlier mesh reached (0.31 against the second
attempt's 0.89). Mesh (a) is fully converged; mesh (b)'s `\|U\|` residual
plateaus around `1e-5` rather than continuing to machine zero, and its
domain MINIMUM temperature is still drifting down slowly at 40 000 iterations
(312.6 K, was 315.3 K at 35 000) even though the wall-adjacent MAXIMUM has
been flat to three figures since iteration 20 000 (369.78 → 369.92 K) — the
quantity this gate reports, the wall-adjacent heat flux, has stopped moving;
the domain core's slow cooling has not, for a reason given below.

`ofgpu-lowmach`'s own end-of-run report integrates `k_eff_wall · snGrad(T)` over
every wall face, off the SAME Robin triple the energy matrix was assembled
from, exactly as the two earlier attempts. Measured:

| Mesh | Total wall heat input | Mean flux (over all 4 wall patches, incl. the two adiabatic sides) |
|---|---|---|
| (a) standard / `thermalWallFunction`, y+ ≈ 41.7 | **6.00 W** | 468.8 W/m² |
| (b) `lowRe` / plain `fixedValue`, y+ ≈ 0.31 | **55.83 W** | 4361.8 W/m² |

Ratio (a)/(b) = **0.107**. **Honestly: this is WORSE than the second
attempt's 0.381**, not an improvement — removing the "not fully developed"
confound did not bring the two meshes closer together. This gate stays
OPEN, and the reason is not "the wall function is even worse than thought" by
itself; it is a confound this rerun's OWN calibration introduced and could
not remove:

1. **The two meshes are compared at very different driving ΔT, and that is
   not a free choice.** Mesh (a)'s hot walls run at 308–323 K (ΔT to `Tw` ≈
   50 K); mesh (b)'s run at 313–370 K (ΔT to `Tw` ≈ 3 K at the wall-adjacent
   cell) — a resolved `y+ ≈ 0.3` sublayer conducts so much more effectively
   than a `y+ ≈ 42` wall-function cell that the SAME `-6 W` sink leaves mesh
   (b) within a few kelvin of `Tw`, and a sink large enough to pull mesh
   (b)'s near-wall ΔT up to mesh (a)'s ~50 K was tried (`-heaterPower -900`)
   and produces an UNPHYSICAL domain core — 160 K and still falling after
   6 000 iterations, because the sink is uniform per unit VOLUME while the
   heat it has to balance enters almost entirely through the thin near-wall
   cells the fine mesh resolves. That is a real limitation of "one uniform
   domain-wide sink" as a periodic energy closure, not a bug in this run;
   the standard fix in the literature (a spatially-weighted sink, or the
   classical `T = θ(y,z) + β·x` mean-temperature-gradient decomposition for
   a constant-flux wall) is a different, larger piece of work than this task
   asked for.
2. **Because flux scales with ΔT, the raw ratio above is confounded by point
   1, not a clean like-for-like comparison.** Normalising by each mesh's own
   wall-adjacent ΔT gives an effective heat-transfer coefficient
   `h = flux / (Tw - T_wall-adjacent)`: `h_a ≈ 937.5 W/m² / 50.0 K ≈ 18.8
   W/(m² K)` against `h_b ≈ 8724 W/m² / 3.23 K ≈ 2699 W/(m² K)` — a ratio of
   about **0.007**, roughly two orders of magnitude, not the ~9x the raw
   flux ratio alone suggests. Neither number is "the" answer; they bracket
   it, and the true, matched-ΔT comparison this gate's own table asks for is
   still not established by either.
3. **The direction of the surprise is real and worth stating plainly.**
   Removing the entrance-length confound did not narrow the gap — by the
   ΔT-normalised reading it widened it sharply. One honest reading is that
   the second attempt's 0.381 was flattered by BOTH meshes still being
   partially entrance-dominated in a way that happened to bring their
   fluxes closer together, not by the wall function actually performing
   better in that geometry; a fully-developed comparison, once the ΔT
   confound above is also removed, may show a larger discrepancy than either
   earlier number suggested. This is a hypothesis, not a finding this run
   itself sorts out — the sorting-out needs the spatially-weighted sink (or
   equivalent) named in point 1, which is out of this task's scope.

None of this is grounds to tune `Tw`, the body force or the sink until the
ratio looks better — SPEC-LIT §0's rules forbid it, and points 1–3 above are
exactly the kind of honest complication those rules exist to surface rather
than hide. What this rerun DOES establish: the periodic-pair mechanism
itself works end to end (mesh, momentum source, heat sink, steady state,
`ofgpu-validate`'s new cyclic-pair invariants — see below) and both meshes
now sit in a genuinely developed state with no entrance length to argue
about; what it does NOT establish, still, is a same-ΔT wall-heat-flux
comparison between a wall-function mesh and a resolved one. That comparison
— a spatially-weighted (or decomposed) periodic energy closure, run at a
matched near-wall ΔT — is the concrete next step this rerun leaves, not a
vaguer "try periodic" the way the second attempt's own next step was.

SPEC-LIT §31.1's own periodic-pair invariants (bijection, `Sf_a == -Sf_b`)
are now a permanent `ofgpu-validate` gate (`check_cyclic_pair`) — cheap
enough to run every time (a few dozen faces, no GPU), unlike the two-mesh
wall-heat comparison above, which stays a driver-level rerun.

<details>
<summary>The second attempt (graded, non-periodic 2.0 m duct; superseded by the periodic rerun above)</summary>

**The gate, rerun with a graded mesh (Task G).** The first attempt at this
gate (kept below for the record) found the JSONC Cartesian mesh could not
grade toward a wall at all — `blockgen`'s own `GradedAxis` (`expansion`,
`two_sided`) existed and was already exercised by `blockgen`'s own cases
(`src/blockgen.rs`'s `case_block_spec`, e.g. `CaseKind::Channel`'s
`b.y.expansion = 20.0; b.y.two_sided = true;`), but `io/case_json.rs` only
ever wrote a uniform axis. `mesh.grading` (a per-axis, optional JSON object
lowered onto `GradedAxis` exactly the way `blockgen`'s own cases use it — see
`cases/README.md`'s JSONC section and the commented example in
`docs/case-example.json`) closes that gap. Two things changed for this
rerun, both in `cases/channelThermalWF.jsonc` / `cases/retired/channelThermalLowRe.jsonc`:

* **The duct is five times longer** — 2.0 m rather than 0.4 m (`L/Dh = 50`
  rather than `10`, `Dh = 0.04 m` the square duct's hydraulic diameter),
  chosen because JSONC still has no cyclic-patch pair (`blockgen`'s own
  `case_block_spec` comment: a streamwise cyclic "need[s] a coupled patch
  pair" this format does not build) — a longer duct is the honest substitute
  for a periodic one that was available without adding cyclic-patch support
  to the JSONC reader, which this task did not ask for.
* **Mesh (b) is now GRADED, not merely fine.** `mesh.grading: { "y": {
  "expansion": 200.0, "twoSided": true } }` on 50 wall-normal cells (first
  cell ≈ 2×10⁻⁵ m, centre cell ≈ 0.0034 m) replaces the first attempt's 250
  UNIFORM cells — resolving the sublayer at the wall without forcing the
  same fine spacing on the channel core, which is exactly the confound the
  first attempt's cause #2 named. Mesh (a) stays uniform (6 wall-normal
  cells, coarsened from the first attempt's 8) — a wall-FUNCTION mesh does
  not need to resolve the sublayer, so SPEC-LIT §29.1's `standard` row is
  asked to do its actual job on a deliberately coarse near-wall cell, not a
  graded one.

Both run to a quasi-steady state through `ofgpu-lowmach`, `-iters 6000` (mesh
(a) reaches `|U|` residual `1.26e-15` well before that) / `-iters 3000` for
mesh (b), which reaches continuity residual `2.3e-14` and holds `T`/`rho`
unchanged to six significant figures from iteration 1500 onward — `|U|`
residual is still declining slowly at `5.0e-6` at iteration 3000, a
consequence of the graded mesh's near-wall cell aspect ratio (≈500:1)
raising the momentum system's condition number, not of the reported
quantities (T, rho, the wall-heat integral) still moving. The two cases
differ in exactly one place beyond mesh (b)'s grading: which row of
SPEC-LIT §29.1's table the hot walls use —

| Mesh | Wall-normal cells | `T` on the hot walls | Measured y+ (min/mean/max) |
|---|---|---|---|
| (a) `channelThermalWF.jsonc` | 6, uniform | `thermalWallFunction`, `standard` preset | 21.95 / 37.90 / 40.29 |
| (b) `channelThermalLowRe.jsonc` | 50, two-sided graded (`expansion: 200`) | plain `fixedValue`, `lowRe` preset | 0.43 / 0.89 / 1.02 |

Both land in their intended range for the first time in this gate's history:
(a) squarely inside the target y+ 30–60 (asked for by SPEC-LIT §29.1's
`standard` row), (b) squarely inside y+ ~ 1 (a resolved sublayer). `ofgpu-lowmach`'s
own end-of-run report (`=== integrated wall heat flux (SPEC-LIT S29.3) ===`)
integrates `k_eff_wall · snGrad(T)` over every wall face, straight off the
SAME `(fr, ref_value, ref_grad)` Robin triple the energy matrix was
assembled from — so it needs no re-derivation of Jayatilleke, and treats
both wall types through one formula. Measured:

| Mesh | Total wall heat input | Mean flux (over all 4 wall patches, incl. the two adiabatic sides) |
|---|---|---|
| (a) standard / `thermalWallFunction`, y+ ≈ 37.9 | **157.2 W** | 491 W/m² |
| (b) `lowRe` / plain `fixedValue`, y+ ≈ 0.89 | **412.3 W** | 1288 W/m² |

Ratio (a)/(b) = **0.381** — up from the first attempt's **0.095**, a four-fold
improvement from properly targeting both meshes' y+ and giving the resolved
mesh a real graded near-wall cell instead of uniform over-resolution.
**Honestly: 0.381 is still far from 1, and nothing here was tuned to push it
closer** (`T_w`, `U_ref`, `Pr`/`Pr_t` are unchanged from the first attempt).
This gate stays OPEN. What the rerun's own numbers make visible now that the
y+ targeting and the mesh-resolution confound of the first attempt are
largely closed:

1. **Mesh (a) is still, on its own terms, an outer-layer-under-resolved
   mesh.** Six wall-normal cells is enough to LAND the first cell at a
   sensible y+ (that only needs one number, the first cell's height) but not
   enough to represent the k-ε profile across the rest of a 0.04 m channel —
   `standard`'s wall function supplies the near-wall BOUNDARY CONDITION, it
   does not substitute for resolving the flow the condition is applied to.
   Mesh (b), by contrast, now has 50 cells doing that job. The two meshes no
   longer share the SAME confound the first attempt reported (mesh (a)'s
   wall-normal cell count was its entire outer resolution AND its wall
   placement, conflated) — but they are still not outer-layer-equivalent
   meshes, only near-wall-equivalent ones, and that asymmetry is still large
   enough to matter at this cell count.
2. **A wall FUNCTION is a model, not a boundary condition that reproduces a
   resolved sublayer's flux exactly by construction.** Jayatilleke's
   correction (like the log law it sits on) is an equilibrium, locally
   self-similar closure; published validations of the log-law family against
   resolved LES/DNS wall heat transfer typically report tens-of-percent
   agreement even in FAVOURABLE conditions (fully developed, no streamwise
   pressure gradient to speak of, moderate Reynolds number) — a factor of
   ~2.6 (`1/0.381`) is outside that band, but "exactly 1" was never a
   realistic target for this particular closure; SPEC-LIT §29.3's own table
   asks for "a stated tolerance," not identity, and this run is the first
   honest measurement of what that tolerance actually is for this solver.
3. **`L/Dh = 50` is longer, not asymptotically long.** A uniform-inlet duct
   at `Re_Dh ≈ 8000` is conventionally taken to need on the order of `10`–`60`
   hydraulic diameters to be considered developed depending on the
   correlation and what "developed" is measured against (mean velocity vs.
   turbulence statistics vs. thermal field, the slowest of the three); `50`
   sits inside that range, not comfortably past it, and the thermal field in
   particular develops more slowly than the momentum field for `Pr ≈ 1` in
   a duct with a suddenly-imposed wall temperature at `x = 0`. A truly
   periodic (cyclic-patch) domain — still not something the JSONC reader
   builds (`blockgen`'s own comment on `CaseKind::Channel`) — would remove
   this cause entirely rather than merely shrink it; that is the concrete
   next step, not a vaguer "run it longer."
4. **The two meshes are not identical apart from `y`.** `z` (the adiabatic
   side walls, 4 uniform cells both cases) and the streamwise direction (200
   uniform cells both cases) are held fixed, but mesh (b)'s much smaller
   near-wall cell volumes shift its local Courant-like behaviour and its
   discrete k-ε production/dissipation balance at the wall relative to mesh
   (a)'s in ways this comparison does not isolate from the wall-treatment
   difference itself.

None of this is grounds to adjust `T_w`, `U_ref`, the mesh, or the wall
coefficients until the ratio looks better — that would be tuning the gate to
pass, which SPEC-LIT §0's own rules forbid. The honest reading is: the two
closed-form identities (`P(Pr/Pr_t=1)=0`, the one-cell conductance identity)
that `ofgpu-validate` holds to round-off are proven; SPEC-LIT §29's grading
gap that blocked a meaningful end-to-end run is closed (`mesh.grading`,
bit-identical to the pre-grading mesh when absent — see
`io::case_json::tests::a_case_without_grading_lowers_to_the_same_mesh_as_before`);
and the END-TO-END claim that a wall-function mesh and a resolved mesh agree
on real wall heat transfer to a stated tolerance is STILL not established —
0.381 is a real, four-fold improvement over 0.095, and still not a pass. A
periodic (cyclic-patch) duct at this or a higher Reynolds number is the
concrete next step named above, not attempted here — building cyclic-patch
support into the JSONC reader is its own piece of work, out of this task's
scope. (It has since been built — see the periodic rerun above this
collapsed block, which is the "concrete next step" this paragraph named.)

<details>
<summary>The first attempt (superseded by the rerun above, kept for the record)</summary>

The original 0.4 m duct, BOTH meshes uniform (JSONC could not grade at all):

| Mesh | Wall-normal cells | `T` on the hot walls | Measured y+ (min/mean/max) |
|---|---|---|---|
| (a) `channelThermalWF.jsonc` | 8 | `thermalWallFunction`, `standard` preset | 16.7 / 26.5 / 30.0 |
| (b) `channelThermalLowRe.jsonc` | 250 | plain `fixedValue`, `lowRe` preset | 3.5 / 4.2 / 5.1 |

| Mesh | Total wall heat input | Mean flux |
|---|---|---|
| (a) standard / `thermalWallFunction`, y+ ≈ 26.5 | **40.7 W** | 636 W/m² |
| (b) `lowRe` / plain `fixedValue`, y+ ≈ 4.2 | **430.4 W** | 6725 W/m² |

Ratio (a)/(b) = **0.095**. The causes named at the time: neither mesh landed
on its target y+ from a short, under-developed duct; the uniform-only JSONC
mesh forced mesh (a)'s entire outer-layer resolution to be its 8-cell
near-wall placement, confounding "does the wall function correct for an
under-resolved sublayer" with "does an 8-cell channel resolve the outer flow
at all"; and a short, thermally-entrance-dominated duct is exactly where an
equilibrium wall function is weakest. The rerun above closes the grading gap
and lengthens the duct; the ratio improved four-fold but the gate is still
open, for the (different, narrower) reasons listed above it.

</details>

</details>

</details>

### The gate, rerun with both comparison defects removed (SPEC-LIT §32.5.3, §35.3)

> **Superseded by the last subsection of §1.1.** Every run in this subsection
> was produced by a driver that ignored the case's own `numerics` block — the
> §13.4 defect this subsection itself reports at the end, unfixed at the time.
> Its momentum equation ran `bounded Gauss upwind` where both cases ask for
> `Gauss linearUpwind grad(U)`, and the `bounded` half of that substitution
> turns out to be worth the whole of the drag imbalance reported below. Kept
> in full: the thermostat-weighting experiment and the friction measurement it
> records are both still valid results, and the reruns that supersede its
> verdict reproduce every number in it to five significant figures when the
> substituted entry is put back by hand.

Everything above was judged against a comparison that had two defects in it,
found separately and fixed separately:

1. **The thermostat's compensating sink was UNIFORM.** A streamwise-periodic
   duct at constant wall flux calls for a sink proportional to the LOCAL
   streamwise mass flux `rho u . e_hat` (Kays & Crawford ch. 9; Patankar, Liu
   & Sparrow 1977) — SPEC-LIT §35.3. A uniform sink gets the total
   right and the distribution wrong: it over-cools the near-wall fluid,
   shrinks `(T_w − T_b)`, and biases `Nu` HIGH.
2. **The friction factor was an INFERENCE, not a measurement.** Both
   Reynolds-analogy verdicts above were taken at an `f` derived from the
   case's own `bodyForce` plus a recorded `U_b`, on the assumption that the
   force balance closes — SPEC-LIT §32.5.

Six runs settle it: each case at the `uniform` default (the control) and at
`"weighting": "massFlux"` (which both case files now carry), 40 000 iterations
each, plus an isothermal control of each mesh. Nothing above is deleted.

**The control, first, because without it the comparison means nothing.** Rerun
at the `uniform` default, both legs reproduce every number this section
already recorded — to the last printed digit:

| At the `uniform` default | recorded above | rerun |
|---|---|---|
| wall function: `T_b`, `T_w`, `U_b`, `Nu`, thermostat power | 293.283 K, 317.253 K, 5.3696 m/s, 65.24, −3.2 W | identical |
| resolved: `T_b`, `T_w`, `U_b`, `Nu`, thermostat power | 292.817 K, 314.087 K, 4.84388 m/s, 73.40, −3.28977 W | identical |
| resolved: worst y+, cells at y+ < 20 | 0.00174585, 192/400 | identical |

**The decisive experiment: the same mesh, the same case, one token changed.**

| Resolved leg (`8x50x1`, `expansion: 200`) | `uniform` | `massFlux` | change |
|---|---|---|---|
| `T_w` | 314.087 K | 314.909 K | +0.822 K |
| `T_b` | 292.817 K | 292.759 K | −0.058 K |
| `T_w − T_b` | 21.2703 K | 22.1503 K | **+4.14 %** |
| **`Nu`** | **73.4006** | **70.4707** | **−3.99 %** |
| `U_b` | 4.84388 m/s | 4.8357 m/s | −0.17 % |

**The diagnosis is CONFIRMED, in sign and in mechanism, and it is not the
whole story.** `Nu` moved DOWN and `(T_w − T_b)` moved UP, which is exactly
what §35.3.2 predicted before the run. The size is 4.0 % of `Nu`: it
takes the resolved leg's Gnielinski miss from +16.3 % to +11.8 %, about a
third of a 6.3-point excess, and leaves 1.8 points still outside the band.
The same change on the wall-function leg moves `Nu` by only −1.41 %
(65.2386 → 64.3168) — 2.8 times less, which is the other half of
§35.3.2's prediction, since the bias lives in the near-wall velocity
deficit and one mesh resolves that deficit while the other hides it inside a
wall function. The two-mesh ratio falls from **1.125 to 1.096**, so this
mechanism alone accounts for 24 % of the two-mesh disagreement — measured,
not argued.

**The gate at the end of THIS subsection** (both cases at `massFlux`, which is what they ship — but measured by the driver that still ignored their `numerics` block). These are NOT the current numbers: the §13.4.1 rerun below moves them to `Nu` 64.5257 and 72.9988. Kept because the comparison this subsection makes is between its own two columns:

| Mesh | `D_h` | Re | `T_w` | `T_b` | `Nu` | Dittus-Boelter | Gnielinski at the PIPE `f` | Gnielinski at the MEASURED `f` |
|---|---|---|---|---|---|---|---|---|
| (a) wall function, `standard` | 0.08 m | 28 622 | 317.567 K | 293.256 K | **64.32** | 73.72 (**−12.8 %**) | 68.30 (**−5.8 %**) | 48.07 (**+33.8 %**) |
| (b) resolved, `LaunderSharmaKE`/`lowRe` | 0.08 m | 25 790 | 314.909 K | 292.759 K | **70.47** | 67.83 (**+3.9 %**) | 63.02 (**+11.8 %**) | 61.18 (**+15.2 %**) |

**Energy balance, and what it does to those bands.** SPEC-LIT §32.4 now
requires a leg's energy-balance gap to be quoted as an uncertainty on its own
`Nu`, because `Nu` is built from the imposed `q_w` and the measured
`(T_w − T_b)`, and a domain whose steady bookkeeping does not close did not
settle at the temperature field `q_w` alone would have produced:

| Leg | thermostat power | wall heat | gap | uncertainty carried onto `Nu` |
|---|---|---|---|---|
| wall function | −3.20335 W | 3.2 W | **+0.105 %** | ±0.1 % — immaterial |
| resolved | −3.30425 W | 3.2 W | **+3.26 %** | **±3.3 %** |

The resolved leg's gap was 2.8 % at the `uniform` default and is 3.26 % at
`massFlux`: the weighting did not close it and made it very slightly worse.

**The friction factor, measured at the wall for the first time.** `ofgpu-lowmach`
now sums the wall-face traction directly (SPEC-LIT §32.5.1) and compares
it against the body force in the KINEMATIC units this crate's momentum
equation is actually written in (§32.5.2's correction — `momentum.rs`
and `simple.rs` contain no density at all, so a comparison made in newtons
carries a systematic `rho_wall/rho_bar` error):

| Leg | `f` INFERRED (what was published) | `f` MEASURED | error of the inference | force balance, kinematic |
|---|---|---|---|---|
| wall function | 0.02162 | 0.017247 (`rho u_tau^2`) / 0.019960 (viscous) | +25 % / +8 % | **−0.113 %** (**+0.001 %** at `uniform`) |
| resolved | 0.02653 | 0.023870 | +11 % | **−3.787 %** (−3.461 % at `uniform`) |

Two results come out of that table, and they point in opposite directions.
The wall-function leg's viscous traction reproduces the body force to five
significant figures — the identity §32.5.1 asserted, now confirmed on a
real turbulent flow rather than only on a manufactured Poiseuille field. The
resolved leg misses it by 3.8 %, on a run whose `|U|` residual is
`2.8 x 10^-12`, which is NOT the convergence statement §32.5.2 expected a
gap there to be.

**A control run says what that 3.8 % is not.** Both cases were rerun with the
heat removed and nothing else changed — hot walls to `zeroGradient`,
thermostat deleted, same mesh, same grading, same models, same body force, so
`rho` is uniform and the low-Mach dilatation is identically zero:

| Isothermal control | body force | measured viscous drag | disagreement |
|---|---|---|---|
| resolved, `expansion: 200`, 50 cells | 6.010850e-4 N | 6.010850e-4 N | **−0.00 %** |
| wall function, 6 cells | 6.010850e-4 N | 6.010893e-4 N | **+0.0007 %** |

So the mesh, the grading severity, the low-Reynolds model and the wall-normal
resolution are all cleared: with a constant density the identical resolved
mesh balances its own momentum exactly. The imbalance appears only when the
energy equation is coupled in, and it appears together with an energy
imbalance of the same size. Across all six runs the two track the continuity
residual the run settles at, monotonically — `1e-19` and `−0.00 %` on
the isothermal control, `6.6e-16` and `+0.001 %`/`2.8e-7 W` on the
wall-function leg at `uniform`, `9.2e-8` and `−3.46 %`/`+2.81 %` on the
resolved leg. SPEC-LIT §32.5.3 names the mechanism that would produce
exactly that pairing (both equations carry §3.1's `bounded` convection
correction, whose domain integral is `−sum_c field_c (div phi)_c V_c`, zero
only when `div phi` is) and records that it has NOT been tested by switching
that correction off. Until it is, both imbalances are reported as measured.

**The verdict.** Judged as SPEC-LIT §32.4 now requires — every band
statement naming the `f` behind it, and carrying its leg's own energy-balance
gap as an uncertainty on `Nu` — **the gate CLOSES on the wall-function leg
under the absolute-prediction verdict (Gnielinski at Petukhov's smooth-pipe
`f = 0.023911`, −5.8 % against ±10 %, on a leg whose energy balance
closes to 0.105 %) and under Dittus-Boelter (−12.8 % against
±20–25 %); it does NOT close on the resolved leg, which passes
Dittus-Boelter (+3.9 %) but sits at +11.8 % against Gnielinski at that same
pipe `f` with a ±3.3 % energy-balance uncertainty on that number — 1.8
points outside a band whose edge lies inside its own uncertainty, so OPEN and,
for the first time in this gate's history, not decisively so; and the
Reynolds-analogy verdict, Gnielinski at the `f` each leg's own wall measures,
closes on NEITHER leg (+33.8 % and +15.2 %), having been reported above as
closing on both only because the `f` behind it was an inference that was
8–25 % high.**

**What closed what, and what is still open.** The mass-flux weighting closed
about a third of the resolved leg's Gnielinski excess and a quarter of the
two-mesh disagreement. The friction measurement did not close anything — it
OPENED something, by withdrawing a verdict that had been reported as passing.

**What the remaining discrepancy implicates**, now that the thermostat's
profile distortion and the friction-factor mismatch are both off the list:

* **The energy and momentum bookkeeping on the resolved leg, together.** A
  3.3 % energy imbalance and a 3.8 % momentum imbalance, on the same runs, on
  a mesh that closes both exactly when the heat is switched off. That is one
  defect with two symptoms, it is worth 3.3 % of the 11.8 %, and §32.5.3
  names a mechanism for it that is one experiment away from being tested.
* **The measured `f` is BELOW the smooth-pipe correlation on both legs**
  (−2.7 % resolved, −16.5 % wall function in the viscous form), where
  Jones (1976) says parallel plates should run ABOVE it. Under-predicted wall
  friction is now an open finding of this gate in its own right — and a
  MOMENTUM finding, SPEC-LIT §33.3's territory rather than §29.3's.
* **The wall function's own two `tau_w` forms disagree by 13.6 %** on leg (a)
  (`rho u_tau^2` = 0.074737 Pa against a viscous 0.086491 Pa). §32.5.2
  says a gap there measures how far the wall-adjacent cell is from local
  equilibrium; this is the first time it has been quantified.
* **Launder-Sharma's near-wall THERMAL prediction**, which nothing in this
  project has validated the way §33.3's law-of-the-wall check validated its
  momentum. Still a live candidate for what is left, but no longer the only
  one and no longer the leading one.

**A SPEC-LIT §13.4 defect found while judging this gate, reported and not
fixed here — FIXED SINCE, and both legs rerun: see the last subsection of
§1.1, which supersedes every number in this one.** `ofgpu-lowmach` builds its `MomentumControls` from
`MomentumControls::default()` and overrides only `nu`, `steady`, `delta_t`
and `ddt`; it never reads `numerics.div["div(phi,U)"]` or
`numerics.relaxation.U` from the case at all. Both channel cases ask for
`Gauss linearUpwind grad(U)` and `U: 0.5`, and both get `Gauss upwind` and
`0.7`. Demonstrated rather than inferred: two 500-iteration runs of the
wall-function case differing only in `div(phi,U)` print BIT-IDENTICAL residual
and bulk-state lines, and so do two differing only in `relaxation.U` (0.5
against 0.9). Under §13.4 a named scheme may not be silently substituted,
so this is a defect — and one that touches the very velocity field this
gate measures, since the momentum convection scheme is running first order
where the case asked for second. It is not fixed in this round because fixing
it moves every number `ofgpu-lowmach` has ever recorded, on every case, and that
is its own job with its own reruns. It is named here because "the resolved leg
under-predicts `f`" and "the momentum equation is running a more diffusive
scheme than the case asked for" have to be weighed together. *(Both have now
been weighed. The diffusive scheme was not the important half of the
substitution: the important half was the `bounded` prefix that came with the
default, which is what the drag imbalance above turned out to be — see the
last subsection of §1.1.)*

`ofgpu-validate` replays all of it: the wall-function leg's verdict, the
resolved leg's verdict and mesh resolution, and — new — the
uniform-vs-`massFlux` experiment itself, which asserts the three things
§35.3.2 predicted before the runs (Nu falls on both legs, `(T_w − T_b)`
widens on both, and the shift is larger on the resolved mesh). The two
verdicts that no longer hold are now NOTES rather than assertions, printed in
full with the numbers that retired them.

### The gate, rerun on the settings the cases actually ask for (SPEC-LIT §13.4.1, §32.5.5)

Every number above — every number this gate has ever produced — was measured
by a driver that did not read the case's own `numerics` block. The §13.4
defect the subsection above reports as "found while judging this gate,
reported and not fixed here" has since been fixed (SPEC-LIT §13.4.1;
`lowmach_controls` in `rust/src/bin/lowmach.rs`), and both legs have been rerun as
shipped, 40 000 iterations each. `ofgpu-lowmach` now prints, at start-up, the
scheme, relaxation, solver and non-orthogonal-corrector count every equation
will actually use, so a log says what a run did without the reader having to
trust that the case file was honoured.

> **The same audit was then run across the other five drivers, and it found a
> fifth instance.** `ofgpu-plume` built its temperature equation's controls
> from a copy of the TURBULENCE controls and overrode only `solvers/T` and
> `relaxationFactors/equations/T`, so `T` was assembled with `div(phi,k)`'s
> scheme and `bounded` flag and with `gradSchemes/default` — one equation's
> entry read for another, the same line `read_simple_controls` in
> `src/bin/buoyant.rs` records as instance 3. Two 12-iteration runs of that
> driver on a generated plume, differing only in `div(phi,T)`, wrote a
> **bit-identical** `T` file; turning `div(phi,k)` instead moved `T`'s
> mass-weighted mean from 409.334 to 511.336. It is fixed, and every driver
> now carries the "two runs must differ" pair test this gate's own defect
> produced. **Nothing this repository publishes moved**: `ofgpu-plume` on
> `cases/plume`, `ofgpu-k-epsilon` on `cases/channel`, `ofgpu-k-omega` on
> `cases/channelKW`, `ofgpu-buoyant` on `cases/plumeB` and `ofgpu-vof` on a
> generated `damBreak` all wrote bit-identical time directories across the
> fix, because every case in this tree writes the entries that were confused
> with each other as the same value — which is exactly how the defect
> survived five times. SPEC-LIT §13.4.1 and §13.4.3 carry the full account.

**The control, first.** The setting the old driver silently substituted was
`MomentumControls::default()`, whose convection entry is `bounded Gauss
upwind` — not the `Gauss upwind` the subsection above names, because that
default's `bounded_convection` field is `true`. Rerunning each case as
shipped with `div(phi,U)` set back to exactly that string, and nothing else
changed, reproduces the published record:

| `div(phi,U)` = `bounded Gauss upwind` | published above | control rerun |
|---|---|---|
| resolved: `T_b`, `T_w`, `U_b`, `Nu`, thermostat power, drag balance | 292.759 K, 314.909 K, 4.83570 m/s, 70.4707, −3.30425 W, −3.787 % | 292.759 K, 314.909 K, 4.83570 m/s, **70.4709**, −3.30423 W, **−3.787 %** |
| wall function: `T_b`, `T_w`, `U_b`, `Nu`, thermostat power, drag balance | 293.256 K, 317.567 K, 5.36659 m/s, 64.3168, −3.20335 W, −0.113 % | 293.256 K, 317.568 K, 5.36687 m/s, **64.3136**, −3.20335 W, **−0.112 %** |

That is agreement to five significant figures on runs whose relaxation (0.5,
against the old defaults' 0.7 on both `U` and `T`) and linear-solver
tolerances are now the case's own rather than the defaults — so those
settings, though they were also being ignored, are worth less than `1e-4` of
the converged answer here, and **the whole of the change below is the
convection entry**.

**The gate as it now stands** (both cases exactly as shipped, at the default `PrtModel constant`; the opt-in Kays-Crawford variant is the last subsection):

| Mesh | `D_h` | Re | `T_w` | `T_b` | ΔT | `Nu` | Dittus-Boelter | Gnielinski at the PIPE `f` | Gnielinski at the MEASURED `f` |
|---|---|---|---|---|---|---|---|---|---|
| (a) wall function, `standard` | 0.08 m | 28 785 | 317.483 K | 293.251 K | 24.2318 K | **64.526** | 74.057 (**−12.9 %**) | 68.598 (**−5.9 %**) | 47.996 (**+34.4 %**) |
| (b) resolved, `LaunderSharmaKE`/`lowRe` | 0.08 m | 26 288 | 314.186 K | 292.800 K | 21.3862 K | **72.999** | 68.872 (**+6.0 %**) | 63.959 (**+14.1 %**) | 62.599 (**+16.6 %**) |

**What moved, and what it was worth:**

| | leg (a) before → after | leg (b) before → after |
|---|---|---|
| Re | 28 622 → 28 785 (+0.57 %) | 25 790 → **26 288 (+1.93 %)** |
| `U_b` | 5.36659 → 5.39720 m/s | 4.83570 → **4.92909 m/s** |
| `T_w` | 317.567 → 317.483 K | 314.909 → **314.186 K** |
| `T_b` | 293.256 → 293.251 K | 292.759 → 292.800 K |
| ΔT | 24.3109 → 24.2318 K (−0.33 %) | 22.1503 → **21.3862 K (−3.45 %)** |
| `Nu` | 64.3168 → 64.5257 (+0.32 %) | 70.4707 → **72.9988 (+3.59 %)** |
| `f` measured | 0.017247 → 0.017129 (`rho u_tau^2`); 0.019960 → 0.019760 (viscous) | 0.023870 → 0.023936 (+0.28 %) |
| **kinematic drag balance** | −0.113 % → **−0.005 %** | **−3.787 % → −0.000 %** |
| **energy balance** | +0.105 % → +0.106 % | **+3.26 % → +3.11 %** |
| `contErr` floor | 2.8e−8 → 2.9e−8 | 1.10e−7 → 1.10e−7 |
| mesh resolution | y+ mean 57.66 → 57.78 | worst y+ 0.00174051 → 0.00185363; 192/400 cells at y+ < 20, unchanged |
| Gnielinski at the pipe `f` | −5.8 % → −5.9 % | **+11.8 % → +14.1 %** |
| two-mesh ratio `Nu_b/Nu_a` | 1.096 → **1.131** | |

Only the bold rows are large enough to matter. Leg (a) moves by less than
0.7 % in everything and its verdicts are unchanged in substance. Leg (b)
moves materially in three places: its bulk velocity and Reynolds number rise
by 2 %, its driving ΔT falls by 3.5 % and its Nusselt number rises by 3.6 %
— which takes it **further from** Gnielinski, not closer — and its kinematic
drag balance closes completely.

#### What happened to the two imbalances — and what one token was worth

The drag imbalance is **gone**, and the mechanism SPEC-LIT §32.5.3 named as
the suspect is confirmed by isolating it. Four runs of the resolved leg,
identical in every byte but the `div(phi,U)` entry:

| `div(phi,U)` on the resolved leg | `Nu` | ΔT | `U_b` | **drag balance** | **energy balance** | `contErr` |
|---|---|---|---|---|---|---|
| `bounded Gauss upwind` (what the driver substituted) | 70.4709 | 22.1502 K | 4.83570 | **−3.787 %** | +3.257 % | 1.10205e−07 |
| `bounded Gauss linearUpwind grad(U)` | 70.5193 | 22.1351 K | 4.83723 | **−3.788 %** | +3.255 % | 1.10205e−07 |
| `Gauss upwind` | 72.9508 | 21.4002 K | 4.92755 | **+0.000 %** | +3.116 % | 1.10101e−07 |
| `Gauss linearUpwind grad(U)` (as shipped) | 72.9988 | 21.3862 K | 4.92909 | **−0.000 %** | +3.114 % | 1.10100e−07 |

and the same isolation on the wall-function leg:

| `div(phi,U)` on the wall-function leg | `Nu` | **drag balance** | **energy balance** | `contErr` |
|---|---|---|---|---|
| `bounded Gauss upwind` | 64.3136 | **−0.112 %** | +0.1047 % | 2.80e−08 |
| `Gauss upwind` | 64.3815 | **+0.002 %** | +0.1048 % | 2.81e−08 |
| `Gauss linearUpwind grad(U)` (as shipped) | 64.5257 | **−0.005 %** | +0.1062 % | 2.90e−08 |

Three things follow, and the first is not what this rerun was expected to
find.

**1. It was never the scheme's ORDER. It was the `bounded` token.** Going
from first-order `Gauss upwind` to second-order `Gauss linearUpwind grad(U)`
— the whole premise of "first-order numerical diffusion changes the velocity
profile, hence the wall shear, hence Nu" — is worth **+0.07 % of `Nu`** on
the resolved leg and **+0.22 %** on the wall-function leg, and nothing at all
to either imbalance. Dropping `bounded`, which is what the case files
actually ask for (neither names the prefix), is worth **+3.5 % of `Nu`** on
the resolved leg and **the entire 3.79 % drag imbalance**. Both legs, same
conclusion, and the two effects are cleanly separable because all four
combinations were run.

**2. Why `bounded` was wrong here.** SPEC-LIT §3.1's bounded form subtracts
`V_P (div u)_P` from the diagonal, on the argument that a part-converged flux
is not exactly solenoidal and the spurious source it injects should be
removed. In an incompressible solver that is a numerical correction and it
vanishes at convergence. **In this low-Mach solver it does not vanish**,
because `div u` is not an error here: §25.1 makes it a prescribed constraint,
`div(u) = Q/(rho cp T)`, and the pressure equation solves for exactly that
(`Simple::assemble_pressure`'s `target_div`). The correction therefore removes
a real, converged, O(dilatation) amount of streamwise momentum that never
reaches the wall, which is the shape of the imbalance measured, and it is
absent the moment the case's own unbounded entry is honoured. That much the
isolation establishes on its own.

**What the mechanism does NOT yet explain is the SIZE, and the hand estimate
that suggests it might is reported here because it fails.** Using
`rho cp T = p0 cp / R_s = 3.551e5 J/m³` (constant, by the ideal-gas law at
fixed `p0`), the correction's domain integral is
`Σ_c U_c (div phi)_c V_c = (1/(rho cp T)) Σ_c U_c Q_c V_c`, whose two terms
are +3.2 W of wall heat and −3.2–3.3 W of thermostat, each weighted by the
velocity where it acts. Taking the wall-adjacent velocity and the mass-flux-
weighted mean gives −9.5 % of the body force on the resolved leg against a
measured −3.79 %, and −3.1 % on the wall-function leg against a measured
−0.112 % — the right sign twice, a factor of 2.5 out on one leg and a factor
of 28 out on the other. So the total heat is not what sets the size: the
LOCAL distribution of `div u` is, and that is set by `div(k_eff grad T)`
(§25.1's `Q`), not by the domain-integrated wattage a hand estimate can
reach. The estimate is recorded, with its failure, rather than quietly
dropped, because it is the obvious first thing a reader will try.

**SPEC-LIT §32.5.3's reading of its own table — "both imbalances track the
continuity residual, monotonically" — is retired**: `contErr` is unchanged to
three significant figures across all four resolved-leg runs above while the
drag imbalance switches between −3.79 % and 0.000 % on the `bounded` token
alone. The correlation was real; the causal reading of it was not.

**3. The energy imbalance is NOT the same defect, and it did not go with the
momentum one.** It moves from +3.26 % to +3.11 % — 0.14 points, on a change
that closed 3.79 points of momentum imbalance — so "one defect with two
symptoms", which the subsection above proposed, is refuted by measurement.
The energy equation carries a bounded correction too, but `Energy::assemble`
applies it **unconditionally** (SPEC-LIT §26: with a nonzero target
divergence it is physics there, not stabilisation) and on the MASS flux
rather than the volumetric one, so no case setting can switch it off and this
rerun could not test it. That test is the next experiment and it needs code,
not a case file: instrument `fvm_div_bounded_correction`'s own domain
integral on the energy equation and compare it against the 0.0996 W the
balance is short.

#### The verdict

> **Still current, and now conditional on one setting.** Everything in this
> subsection is the record at the SHIPPED DEFAULT, `PrtModel constant`, and
> both cases are byte-for-byte the ones that produced it. The candidate this
> verdict ends by naming has since been implemented and measured (SPEC-LIT
> §37), and under `PrtModel KaysCrawford` the absolute-prediction verdict
> closes on BOTH legs - see the last subsection of §1.1 for the four runs and
> what they do and do not establish.

**The gate does NOT close.** Leg (a), the wall-function mesh, still passes the
ABSOLUTE-PREDICTION verdict — Gnielinski at Petukhov's smooth-pipe
`f = 0.023878`, `Nu` 64.526 against 68.598, **−5.9 %**, inside ±10 %, with a
±0.11 % energy-balance uncertainty far narrower than that margin — and passes
Dittus-Boelter at −12.9 %; leg (b), the resolved mesh, has moved **further
out**, from +11.8 % to **+14.1 %** against Gnielinski at its own pipe
`f = 0.024416`, and its ±3.1 % energy-balance uncertainty (`Nu` ∈ [70.7,
75.3], i.e. +10.6 % to +17.7 % of Gnielinski) **no longer reaches the band
edge**, so a miss the subsection above could report as "outside, but by less
than its own uncertainty, and therefore not decisive" is now decisive; and
the REYNOLDS-ANALOGY verdict, Gnielinski at the `f` each leg's own wall
measures, still closes on NEITHER leg (+34.4 % and +16.6 %).

**What closed what.** The §13.4 fix closed the resolved leg's kinematic drag
balance outright (−3.787 % → −0.000 %) and confirmed §32.5.3's suspected
mechanism by isolating it to one token. It did not close the Nusselt gate; it
moved leg (b) 2.3 points further away from it.

**What the remaining discrepancy implicates.** The uniform-sink distortion,
the friction-factor inference, the §13.4 scheme violation and now the
momentum bounded-convection correction are all off the list of suspects. What
is left is narrower and more specific than anything this gate has been able
to say before:

* **The resolved leg now gets the MOMENTUM very nearly right and the HEAT
  14 % too high.** Its measured `f = 0.023936` is **−2.0 %** of Petukhov's
  pipe `f` (Jones 1976 says a plane channel should sit slightly above a pipe,
  so this is a small under-prediction, no longer a large one), and its drag
  balance closes exactly. A leg that transports very nearly the right
  momentum and 14 % too much heat is making a **thermal** statement, and for
  the first time in this gate's history there is nothing left on the momentum
  side to blame for it.
* **The leading named candidate is now the constant turbulent Prandtl
  number, not Launder-Sharma's damping alone.** `k_eff = k + rho cp nu_t/Pr_t`
  (§26) uses a single `Pr_t = 0.85` everywhere, including a first cell at
  y+ = 0.0019. Kays, *ASME J. Heat Transfer* 116 (1994) 284–295, reviews the
  evidence that `Pr_t` RISES towards a wall — of order 1.5–1.9 for air in the
  sublayer — so a constant 0.85 over-predicts near-wall turbulent heat
  transport, narrows `(T_w − T_b)` and raises `Nu`. That is the right sign
  for a +14 % miss, and it is carried in full by the resolved mesh and hardly
  at all by the wall-function mesh, whose first cell sits at y+ 58 where 0.85
  is appropriate and whose wall heat goes through Jayatilleke's own thermal
  law instead. **A hypothesis with a mechanism and a direction, not a
  finding** — nothing here has measured it.
* **The energy imbalance, +3.11 %, alone and unexplained.** It survived the
  thermostat weighting, the friction measurement and the momentum fix
  essentially unchanged. It is now a single anomaly rather than half of a
  pair, and it stays on this leg's `Nu` as a stated uncertainty.
* **The two-mesh disagreement is still a MOMENTUM story.** The two meshes
  predict `U_b` = 5.397 and 4.929 m/s at the same body force and measure
  viscous-form friction factors 21.1 % apart; Gnielinski at that pair
  predicts a two-mesh `Nu` ratio of **1.119** against a measured **1.131**,
  so about 91 % of the two legs' Nusselt disagreement is accounted for by
  their momentum disagreement — the same fraction as before, on a larger
  ratio. Why the resolved mesh predicts a 9 % lower bulk velocity at the same
  forcing is SPEC-LIT §33.3's territory and remains open.

`ofgpu-validate` replays all of it: both legs' verdicts at the new numbers,
the resolved leg's mesh resolution, the `uniform`-versus-`massFlux` weighting
experiment re-measured at the corrected numerics, and — new —
`check_bounded_convection_experiment_replay`, which asserts the three
statements the isolation above establishes (dropping `bounded` closes the
drag balance on both legs, the scheme's ORDER is worth less than 0.3 % of
`Nu` on either, and the energy imbalance moves with neither).

### The gate, with the named candidate finally measured: Kays-Crawford `Pr_t` (SPEC-LIT §37)

> **Superseded in its NUMBERS by the subsection after this one (SPEC-LIT
> §26.1), and in nothing else.** Every run below carries the +3.11 %/+3.35 %
> energy imbalance this section reports as "untouched and still the open
> anomaly"; that anomaly has since been traced to §25.1's `Q` being
> implemented without its conduction term, and closed. All four runs were
> repeated on the fixed solver: `Nu` falls ~1.8 % on the resolved leg and
> ~0.06 % on the wall-function one, every conclusion this section draws about
> `Pr_t` survives unchanged, and the ±3.35 % error bar this section has to
> quote beside its own verdict becomes ±0.0001 %. Read this section for what
> §37 establishes and the next one for the numbers that stand.

The subsection above ends by naming ONE remaining candidate, with a mechanism
and a direction and nothing measured: `k_eff = k + rho cp nu_t/Pr_t` uses a
single `Pr_t = 0.85` everywhere, including a first cell at y+ = 0.0019, where
Kays (*ASME J. Heat Transfer* 116 (1994) 284-295) puts `Pr_t` at 1.5-1.9. A
`Pr_t` too LOW makes `alpha_t = nu_t/Pr_t` too LARGE, moves too much heat,
narrows `(T_w − T_b)` and biases `Nu` HIGH — the right sign for a +14.1 %
miss, carried in full by the mesh that resolves the sublayer and hardly at all
by the one that does not.

SPEC-LIT §37 specifies the model that tests it — Kays-Crawford's
`Pr_t(Pe_t)`, `Pe_t = (nu_t/nu) Pr`, with the published constants `C = 0.3`
and `Pr_t_inf = 0.85` and **nothing fitted**: `Pr_t_inf` is the case's own
existing `Prt` entry, re-read as the free-stream asymptote, and the sublayer
value `2 Pr_t_inf = 1.70` is a DERIVED limit of the formula, not a second
number anyone chose. §37.2 derives both limits and the two numerical branches
the formula needs; `ofgpu-validate` checks them live.

**The default is unchanged, deliberately.** `PrtModel` defaults to
`constant`, both shipped channel cases are byte-for-byte what they were, and
every number in every subsection above still stands as the shipped-default
record. The four runs below were made on scratchpad copies with one token
added, `"PrtModel": "KaysCrawford"`, and nothing else.

**Does `Pr_t` reach the Jayatilleke wall function? Yes — and §37.3 keeps
`Pr_t_inf` there, by derivation.** `Pr_t` enters §29.3's thermal law twice,
as `T+ = Pr_t (u+ + P(Pr/Pr_t))`, and both are the LOG-LAYER value: `P` is an
integrated sublayer resistance obtained once under an assumed near-wall `Pr_t`
profile, and the log branch it is added to is the fully turbulent one. That
value is exactly what Kays-Crawford's own `Pe_t -> infinity` limit returns
(§37.2), so feeding a local sublayer `Pr_t` in would count the same physics
twice with a number `P` was never calibrated against. The wall function is
therefore untouched — a substantive decision, stated and announced by the
driver at start-up, not an omission.

*Two qualifications, because the point of leg (a) is that it is a CONTROL and
a control has to be described exactly.* First, **this gate does not run a
`thermalWallFunction` face at all**: both channel cases name
`fixedFluxTemperature` on their hot walls explicitly (§15.5's rule — an
explicit per-field type beats the `wallTreatment` preset), so §29.3's law
enters leg (a) only as the POSTPROCESSING diagnosis of `T_w` that `Nu` is
then built from. Second, `k_eff,wall` does change on leg (a): the imposed
flux does not (`ref_grad = q_w/k_eff,wall`, so the product is `q_w` whatever
`k_eff,wall` is — the flux is exactly 500 W/m² in all four runs below), but
the boundary VALUE of `T` that same triple extrapolates does, and the
wall-face density read off it enters the `T_w` diagnosis. Leg (a) is a
near-control with two identified channels, not an inert one, and both are
quantified below.

**A reporting defect this experiment found, and the measurement it
discarded.** `ofgpu-lowmach` recomputed `k_eff,wall` on the host, for its own
wall-heat integral, with the constant `Pr_t` unconditionally. Under
`KaysCrawford` the wall face's own `Pe_t` is O(2), so the true `k_eff,wall`
is ~16 % smaller — and the first wall-function run duly REPORTED 580.5 W/m²
of wall heat and `T_w` = 321.148 K on a wall imposing exactly 500 W/m². The
solve was never wrong; the report was, by exactly the ratio of the two
`Pr_t`. Fixed (SPEC-LIT §37.3), and **all four runs below were then made
again on the fixed binary**, which is why the first wall-function
Kays-Crawford numbers appear nowhere in this table.

**The four runs.** `cases/channelPeriodicFluxLowRe.jsonc` and
`channelPeriodicFluxWF.jsonc`, 40 000 iterations each, `PrtModel` the only
token that differs. The `constant` legs are the CONTROL and reproduce this
section's own published record to every printed digit — `Nu` 72.9988 and
64.5257, `U_b` 4.92909 and 5.39720 m/s, drag balance −0.000 % and −0.005 %,
`y+` 0.00185363 and 57.7793 — so this is a controlled comparison and not two
different states. All four reach a fixed point rather than merely a small
residual: `T` is unchanged in every printed digit from iteration 30 000
through 39 999 on all four runs, at `|U|` residuals of 2.1×10⁻⁸ (both
wall-function runs) and 4.1×10⁻¹² / 1.3×10⁻¹¹ (resolved, `constant` /
`KaysCrawford`), 165-194 s each on an RTX 5070 Ti:

| | (a) wall function, `constant` | (a) wall function, **`KaysCrawford`** | (b) resolved, `constant` | (b) resolved, **`KaysCrawford`** |
|---|---|---|---|---|
| `T_w` | 317.483 K | 317.828 K | 314.186 K | 315.692 K |
| `T_P` (first cell) | 297.093 K | 297.204 K | 313.978 K | 315.482 K |
| `T_b` (mixed-mean) | 293.251 K | 293.241 K | 292.800 K | 292.748 K |
| `ΔT = T_w − T_b` | 24.2318 K | 24.5874 K (**+1.47 %**) | 21.3862 K | 22.9439 K (**+7.28 %**) |
| `U_b` | 5.39720 m/s | 5.39738 m/s (+0.003 %) | 4.92909 m/s | 4.92984 m/s (+0.015 %) |
| Re | 28 785 | 28 786 | 26 288 | 26 293 |
| **`Nu`** | **64.5257** | **63.5900 (−1.45 %)** | **72.9988** | **68.0305 (−6.81 %)** |
| `q_w` (imposed, measured) | 500 W/m² | 500 W/m² | 500 W/m² | 500 W/m² |
| **kinematic drag balance** | −0.005 % | −0.005 % | −0.000 % | +0.000 % |
| **energy balance** | +0.106 % | +0.110 % | +3.11 % | +3.35 % |
| `f` measured | 0.017129 (`rho u_tau²`) / 0.019760 (viscous) | 0.016933 / 0.019534 | 0.023936 (viscous) | 0.023811 |
| `nu_t/nu` across the channel | [16.86, 28.65] | [16.86, 28.65] | [3.9e−7, 35.52] | [3.9e−7, 35.52] |
| **`Pr_t` in use** | 0.85 | **[0.8748, 0.8917]**, volume-mean 0.8817, wall-adjacent cells [0.8760, 0.8802] | 0.85 | **[0.8701, 1.7000]**, volume-mean 0.9042, **wall-adjacent cells 1.7000** |
| worst `y+` / cells at y+ < 20 | 57.7793 mean | 57.7794 mean | 0.00185363, 192/400 | 0.00185378, 192/400 |

**The `Pr_t` range is the whole mechanism, measured.** On the resolved mesh
`Pr_t` runs from 0.870 in the core to **exactly 1.7000 in the wall-adjacent
cells** — the `Pe_t -> 0` limit §37.2 derives, reached because
`LaunderSharmaKE` pins `nu_t/nu` to 3.9×10⁻⁷ there. On the wall-function mesh
it never leaves [0.875, 0.892]: six wall-normal cells, the nearest of them at
y+ 58, none of them anywhere near the sublayer where `Pr_t` departs from
0.85. That is the asymmetry the hypothesis rested on, and it is now a
measurement rather than an argument.

#### The verdicts

| Leg, model | Gnielinski at the PIPE `f` (absolute prediction) | Gnielinski at the MEASURED `f` (Reynolds analogy) | Dittus-Boelter |
|---|---|---|---|
| (a) wall function, `constant` | 68.598 → **−5.9 %** ✔ | 47.996 → +34.4 % ✘ | 74.057 → −12.9 % ✔ |
| (a) wall function, `KaysCrawford` | 68.600 → **−7.3 %** ✔ | 47.412 → +34.1 % ✘ | 74.059 → −14.1 % ✔ |
| (b) resolved, `constant` | 63.959 → **+14.1 %** ✘ | 62.599 → +16.6 % ✘ | 68.872 → +6.0 % ✔ |
| (b) resolved, `KaysCrawford` | 63.966 → **+6.4 %** ✔ | 62.253 → **+9.3 %** ✔ | 68.881 → −1.2 % ✔ |

Carrying each leg's own energy-balance gap as an uncertainty on its `Nu`, as
SPEC-LIT §32.4 requires:

* Leg (b) at `KaysCrawford`, ±3.35 %: `Nu` ∈ [65.75, 70.31], i.e. **+2.8 % to
  +9.9 %** of Gnielinski at the pipe `f` — **inside ±10 % across the whole
  band**, where at `constant` the same construction gave +10.6 % to +17.7 %,
  entirely outside it.
* Leg (b)'s Reynolds-analogy pass is the weaker of the two and does NOT
  survive its own uncertainty: [+5.6 %, +13.0 %] straddles the band edge. It
  closes at the measured value and is reported as closing at the measured
  value only.
* Leg (a), ±0.11 %, is immaterial either way.

**The verdict. Under the shipped default (`PrtModel constant`) the gate
stands exactly where the subsection above left it: leg (a) passes the
absolute-prediction verdict at −5.9 %, leg (b) fails it at +14.1 %, and the
gate does NOT close. Under `PrtModel KaysCrawford` on both legs the
ABSOLUTE-PREDICTION verdict CLOSES ON BOTH LEGS for the first time in this
gate's history — −7.3 % and +6.4 % against ±10 %, with Dittus-Boelter at
−14.1 % and −1.2 % against ±20-25 % — and the REYNOLDS-ANALOGY verdict still
fails on leg (a), at +34.1 %, exactly as it did before: that is a
wall-function FRICTION finding (the `rho u_tau²` form sits 13.6 % below the
viscous form on the same faces, §32.5.2's own out-of-equilibrium measure) and
§37 does not touch it and did not move it.**

**What the experiment establishes, and what it does not.**

1. **The mechanism is confirmed in sign, in mesh-dependence, and in
   magnitude-ordering.** `Nu` fell on both legs, `(T_w − T_b)` widened on
   both, and the shift is **4.7 times larger on the resolved mesh**
   (−6.81 % against −1.45 %) — which is the asymmetry the hypothesis
   predicted before the runs, for the reason the `Pr_t` row above now
   measures.
2. **The magnitude was over-predicted, and that is reported rather than
   rounded.** The prediction going in was that leg (b)'s `Nu` would fall by
   roughly 10-14 %; it fell by **6.8 %**, about half. The reason is visible
   in the same table: `Pr_t` reaches 1.7 exactly where `nu_t/nu` is 4×10⁻⁷,
   so `alpha_t` there is negligible against the molecular `k` either way. The
   correction bites in the BUFFER layer, where `nu_t/nu` is O(1) and `Pr_t`
   is 0.98-1.44 — a smaller lever than the sublayer number suggests. A
   correction that had needed to be 14 % to close the band would have been a
   worse result than one that only needed to be 7 %.
3. **Nothing was tuned.** `C = 0.3` and `Pr_t_inf = 0.85` are the published
   constants; `Pr_t_inf` is the value this project has used since §26 was
   written. There is no free parameter in this model, and no case setting
   exposes one.
4. **Leg (a) is a near-control, and its 1.45 % decomposes exactly.**
   `ΔT` rose 0.3556 K, of which **+0.234 K is `(T_w − T_P)`** — the wall-face
   density, read off a boundary value the (`fr = 0`) fixed-flux triple
   extrapolates through `k_eff,wall`, which `Pr_t` changes — and **+0.121 K
   is `(T_P − T_b)`**, the interior profile responding to a core `Pr_t` that
   rose 3.7 % (0.85 → 0.882 volume-mean). As fractions of the 24.2318 K
   driving difference that is +0.97 % and +0.50 %, and their sum is the whole
   of the +1.47 %. Both are the small effects §37.3's scoping argument
   predicts; neither is the sublayer effect leg (b) carries, and the imposed
   500 W/m² does not move at all.
5. **The energy imbalance is untouched and remains the open anomaly.**
   +3.11 % → +3.35 % on leg (b), +0.106 % → +0.110 % on leg (a). §37 is a
   thermal-diffusivity model; it has nothing to say about a domain whose
   steady bookkeeping does not close, and it did not accidentally say
   anything.
6. **The "two-mesh disagreement is a momentum story" reading does not survive
   this, and is qualified here.** The measured two-mesh ratio `Nu_b/Nu_a`
   falls from **1.131 to 1.070**. Gnielinski evaluated at each leg's own
   VISCOUS-form measured `f` — the momentum-implied ratio the subsection
   above used — gives 1.119 at `constant` (so ~91 % of the excess, as
   reported) and 1.127 at `KaysCrawford`, which the measured 1.070 is now
   **below**. Applying the same thermal correction to both legs moves the two
   meshes past each other relative to their momentum, because only one of
   them resolves the layer the correction acts in. The two-mesh ratio was
   never §32.4's verdict rule — each leg is judged against the correlations,
   not against the other — but the decomposition claim built on it is now
   bounded by a measurement.
7. **What is still open.** Leg (a)'s Reynolds-analogy miss (+34 %), which is
   about how far its wall-adjacent cell is from local equilibrium and not
   about heat; leg (b)'s +3.35 % energy imbalance; and the fact that the
   absolute-prediction verdict closes on both legs only when a model that is
   NOT the default is selected on both. The honest one-line summary is that
   §37 removes the last named thermal suspect and closes the verdict it was
   named to close, on a leg whose momentum was already right — not that the
   gate now closes by default.

`ofgpu-validate` carries both halves: the correlation itself LIVE (both
limits, both branches, the monotone bound, the agreement with the literature
form and the point at which that form loses its digits), and this four-run
experiment as a replay asserting the three statements §37 predicted before it
— `Nu` falls on both legs, `(T_w − T_b)` widens on both, and the shift is
several times larger on the resolved mesh.

### The gate, with the energy imbalance finally CLOSED (SPEC-LIT §26.1)

Every subsection above ends by naming the same leftover: leg (b)'s energy
balance is short by 3.11 % (3.35 % under `KaysCrawford`), it survived the
thermostat weighting, the friction measurement, the §13.4 numerics fix and
§37's `Pr_t` model, and §32.4 required it to be carried as an uncertainty on
that leg's `Nu`. §32.5.5 specified the experiment that would localise it and a
later round ran it: instrument the energy equation's own bounded-convection
correction and compare its domain integral against the shortfall. It agreed to
five significant figures on both legs — the whole gap sits in one term.

**That localisation was right and the mechanism proposed with it was wrong, so
this subsection begins with two measured refutations.** SPEC-LIT §26.1 has the
derivation; here is what it means for the gate.

#### What the defect actually was

The energy equation is assembled on the MASS flux, so summing it over a closed
domain leaves an identity — `ddt` is zero, convection telescopes, the laplacian
telescopes to the wall heat, the sources integrate to the thermostat power:

```
−cp Σ_c T_c (Σ_f ±rho_f phi_f)_c  =  q_wall + P_thermostat
```

The correction's domain integral IS the balance gap. But split the mass-flux
divergence at the cell density and the part that §25.1 PRESCRIBES —
`rho_P V_P (∇·u)_target` — contributes **exactly zero**, because the ideal gas
at fixed `p0` makes `cp rho T = γ p0/(γ−1)` a constant, so that half is a
constant times a telescoping sum. Measured on leg (b): **−2.06e-13 W** of a
−0.0996313 W total. The whole gap is the OTHER half, the discrete `u·∇rho`,
and that half is nonzero only because `∇·(rho u)` was not the zero steady
continuity requires — which it was not, because §25.1's `Q` was implemented
without its **conduction** term `∇·(k_eff ∇T)`. On these two cases the omitted
term is the entire 3.2 W the walls put in.

The omission was documented (`src/energy.rs`, DESIGN choice 2) and its cost was
accounted only against §25.2's `p0` ramp. It has two consumers, not one: the
same `Q` builds `(∇·u)_target`, which the pressure equation solves for.

#### The two obvious fixes, both run, both refutations

Each was run twice — once at the incomplete `Q` where the question arose, and
again at the corrected one — on `cases/channelPeriodicFluxLowRe.jsonc`,
40 000 iterations, nothing else changed. The true answer is `Nu` 71.68 with
ΔT 21.78 K.

| candidate | at the incomplete `Q` | at the CORRECTED `Q` |
|---|---|---|
| **drop the correction** — the conservative `div(rho cp phi, T)` | balance closes (−1.63e-06 W) and the answer is destroyed: `Nu` **128.5**, ΔT 12.17 K, on a converged run | worse, and cleanly so: `Nu` **7092**, ΔT **0.2207 K**, `T` ∈ [293.483, 293.675] K — the field is isothermal to 0.19 K with 500 W/m² going into both walls |
| **subtract only the non-prescribed part** — the literal transfer of §3.1's momentum rule | **diverges**: `T` → 605 K, thermostat → −2420 W, gap −2417 W, `contErr` 1.3e-04 | `T` ≡ **293.15 K exactly**, thermostat **2.6e-10 W**, gap **+3.2 W** — the whole wall input disappears into the correction and the controller does nothing |

The first one's second column is the cleaner demonstration. `rho cp T` being
constant makes `∇·(rho cp u T) ≡ (γ/(γ−1)) p0 ∇·u`, so the conservative form of
a temperature equation for an ideal gas at fixed `p0` carries no information
about `T` at all; at the incomplete `Q` the fictitious dilatation was itself
breaking that degeneracy, which is why the run still transported *some* heat
and landed at 128.5. With `Q` right the degeneracy is exact and the equation
stops seeing the temperature.

The second one's own integral is `−∫Q dV`, so removing it does not remove
zero — it puts `∫Q dV` into the identity above with the wrong sign, and the
fixed point moves to where `P_thermostat = 0`.

So the correction stays, unconditionally, and for a better reason than §26 gave
it. What changed is `Q`.

#### The fix, and what it did to the balance

`Q` = the §18 registry PLUS `∇·(k_eff ∇T)`, in both places §25 uses `Q`. The
conduction term is the explicit divergence of exactly the face flux
`fvm_laplacian` assembles implicitly, so nothing new is discretised and its
domain integral telescopes to the boundary heat exactly.

| both cases as shipped, `PrtModel constant` | (b) resolved before | (b) resolved AFTER | (a) wall function before | (a) wall function AFTER |
|---|---|---|---|---|
| thermostat power | −3.29963 W | **−3.20000 W** | −3.20340 W | **−3.20056 W** |
| **energy balance** | **+3.11 %** | **+0.000089 %** | **+0.106 %** | **+0.0174 %** |
| the correction's own domain integral | −0.0996313 W | **+8.85245e-08 W** | −0.00339937 W | **−0.000556869 W** |
| its PRESCRIBED half | −2.06005e-13 W | +2.04339e-13 W | +1.96645e-13 W | +2.89829e-13 W |
| `contErr` floor | 1.10100e-07 | **6.7253e-14** | 2.89888e-08 | **1.99200e-08** |
| kinematic drag balance | −0.000 % | −0.000 % | −0.005 % | −0.005 % |
| wall time, 40 000 iterations | 164 s | **385 s** | 170 s | **219 s** |

The correction falls **1126×** on leg (b) and **6.1×** on leg (a), and leg (b)'s
continuity residual falls **seven orders of magnitude**. That last row retires
one more thing this gate had on record: leg (b)'s `contErr` floor of 1.1e-07,
and the fact that tightening `p`'s `relTol` from 0.01 to 1e-4 diverges the run
at iteration 3317, were both read as properties of the graded mesh. The floor
was the missing term.

The extra wall time is two separate charges and neither is the three added
kernels. Most of it is the pressure solve earning those seven orders of
magnitude (leg (b) alone: 164 s → 302 s). The rest is the coefficient prologue
— `rho cp`, `k_eff`, the §29.3 wall triple, §32.2's fixed-flux rewrite —
running twice per outer iteration, once for the target divergence and once
inside `Energy::correct`. It has to run for the target divergence, because
reading `k_eff` off whatever the previous `correct` left is wrong at a
RESTART: `ofgpu-lowmach`'s own restart gate caught exactly that, the first
post-restart pressure residual missing the continuous run's step-21 residual by
4.8 % against a 0.1 % tolerance. The duplicate is ~1.2 ms per outer iteration
on BOTH legs, which on 48 and 400 cells is kernel-launch overhead and nothing
else; on a 32 768-cell buoyant case the whole of §26.1 is worth
18.96 s → 19.22 s over 1 200 steps, +1.4 %.

Leg (b) is a fixed point, not a slow drift: at 80 000 iterations every reported
quantity is identical to the 40 000-iteration run in every printed digit — `Nu`
71.683, `T_b` 292.773 K, `U_b` 4.93682 m/s, thermostat −3.2 W, gap
−2.84e-06 W. Its `|U|` residual now plateaus at ~1e-07 rather than 4e-12: `T`
still enters the target divergence at one outer iteration of lag (it is only
updated inside `Energy::correct`), so the loop `T → (∇·u)_target → u → T`
leaves a small limit cycle in the RESIDUAL. The STATE does not move, and that
is reported rather than smoothed over.

#### The gate, before and after, on all four runs

Both cases exactly as shipped for the `constant` column; `KaysCrawford` on
scratchpad copies with one token added, as §37 did it. 40 000 iterations each.

| | (a) WF `constant` | (a) WF `KaysCrawford` | (b) resolved `constant` | (b) resolved `KaysCrawford` |
|---|---|---|---|---|
| `T_w` before → after | 317.483 → **317.497 K** | 317.828 → **317.842 K** | 314.186 → **314.549 K** | 315.692 → **316.079 K** |
| `T_b` | 293.251 → **293.251 K** | 293.241 → **293.241 K** | 292.800 → **292.773 K** | 292.748 → **292.718 K** |
| ΔT | 24.2318 → **24.2454 K** | 24.5874 → **24.6019 K** | 21.3862 → **21.7767 K** | 22.9439 → **23.3605 K** |
| `U_b` | 5.39720 → **5.39407** | 5.39738 → **5.39426** | 4.92909 → **4.93682** | 4.92984 → **4.93761** |
| Re | 28 785 → **28 768** | 28 786 → **28 769** | 26 288 → **26 330** | 26 293 → **26 334** |
| **`Nu`** | 64.5257 → **64.4894** (−0.06 %) | 63.5900 → **63.5527** (−0.06 %) | 72.9988 → **71.6830** (−1.80 %) | 68.0305 → **66.8107** (−1.79 %) |
| `f` measured (viscous on (b)) | 0.017129 → **0.017140** | 0.016933 → **0.016944** | 0.023936 → **0.023832** | 0.023811 → **0.023705** |
| **energy balance** | +0.106 % → **+0.0174 %** | +0.110 % → **+0.0185 %** | +3.11 % → **+0.000089 %** | +3.35 % → **+0.000094 %** |
| **kinematic drag balance** | −0.005 % → −0.005 % | −0.005 % → −0.005 % | −0.000 % → −0.000 % | +0.000 % → +0.000 % |
| `Pr_t` in use | 0.85 | [0.874934, 0.891895], vol-mean 0.881819 | 0.85 | [0.871299, **1.7000**], vol-mean 0.906154, wall cells 1.7000 |
| worst `y+` / cells at y+ < 20 | 57.779 → **57.766** mean | 57.779 → **57.766** mean | 0.00185363 → **0.00179449**, 192/400 | 0.00185378 → **0.00179449**, 192/400 |

**The verdicts:**

| Leg, model | Gnielinski at the PIPE `f` (absolute prediction) | Gnielinski at the MEASURED `f` (Reynolds analogy) | Dittus-Boelter |
|---|---|---|---|
| (a) WF, `constant` | −5.9 % → **−5.9 %** ✔ | +34.4 % → +34.3 % ✘ | −12.9 % → −12.9 % ✔ |
| (a) WF, `KaysCrawford` | −7.3 % → **−7.3 %** ✔ | +34.1 % → +34.0 % ✘ | −14.1 % → −14.1 % ✔ |
| (b) resolved, `constant` | +14.1 % → **+11.9 %** ✘ | +16.6 % → +14.9 % ✘ | +6.0 % → +4.0 % ✔ |
| (b) resolved, `KaysCrawford` | +6.4 % → **+4.3 %** ✔ | +9.3 % → **+7.7 %** ✔ | −1.2 % → −3.1 % ✔ |

**THE GATE DOES NOT RE-OPEN. It closes harder.** Under `PrtModel KaysCrawford`
the absolute-prediction verdict closes on both legs, as it did before, and leg
(b) moves from +6.4 % to **+4.3 %** — deeper inside the band, not out of it.
Under the shipped default `PrtModel constant` leg (b) is still outside at
+11.9 %, as it was at +14.1 %, so the default-configuration verdict is
unchanged in kind and 2.2 points better in degree.

**What changed most is not the numbers but what may be claimed about them.**
§32.4 requires every band statement to carry the leg's own energy-balance gap
as an uncertainty on `Nu`. That uncertainty was ±3.35 % on leg (b) at
`KaysCrawford`, which put the pass at "+2.8 % to +9.9 %, inside across the
whole band" — a pass that needed its own error bar quoted to be believed. It is
now **±0.000094 %**. The same applies to the verdict this gate could NOT
previously claim: leg (b)'s Reynolds-analogy pass at `KaysCrawford` was
+9.3 % with a ±3.35 % band of [+5.6 %, +13.0 %] that straddled the edge, and
§37's own write-up said so ("does NOT survive its own uncertainty ... reported
as closing at the measured value only"). It is now **+7.7 % with no band at
all**, so it closes outright. Two verdicts, both leg (b): one strengthened, one
promoted from conditional to unconditional.

**What did NOT move.** Leg (a)'s Reynolds-analogy miss is +34.0/34.3 %, exactly
where it was — that is a wall-function FRICTION finding (the `rho u_tau²` form
sits 13.6 % below the viscous form on the same faces) and §26.1 is an energy
term. Both legs' kinematic drag balances are unchanged to their printed
precision. The `Pr_t` ranges are unchanged, and leg (b) still reaches the
derived `Pe_t → 0` limit of exactly 1.7000 in its wall-adjacent cells. The
resolved mesh's cell count at `y+ < 20` is 192/400 as it has been through every
change this gate has seen.

**And the two-mesh ratio.** `Nu_b/Nu_a` moves 1.1313 → **1.1115** at
`constant` and 1.0698 → **1.0513** at `KaysCrawford`. §37's qualification of
§32.5.5's momentum decomposition stands and gets slightly sharper: at
`KaysCrawford` the measured ratio is further below the 1.127 the two legs' own
viscous-form friction factors imply, so applying a sublayer thermal correction
to a mesh that resolves the sublayer and one that does not still moves the two
meshes past each other relative to their momentum.

#### What is left open

* Leg (a)'s Reynolds-analogy miss, +34 %. Untouched, and a friction statement.
* The absolute-prediction verdict closes on both legs only when `PrtModel
  KaysCrawford` — not the default — is selected on both. §26.1 does not change
  that; it removes the error bar that used to be quoted beside it.
* At the shipped default, leg (b) is +11.9 % of Gnielinski at the pipe `f`. It
  now carries **no** energy-balance uncertainty, so that miss is decisive in a
  way it has never been: there is no longer a bookkeeping gap wide enough to
  hide any part of it.
* Leg (b)'s `|U|` residual floor of ~1e-07, which the one-iteration lag on the
  conduction term introduces. The state is a fixed point to every printed digit
  at 80 000 iterations; the residual is not.

#### Re-measured under `ofgpu-lowmach`

The table above was taken under the driver's earlier name. Both legs were run
again, 40 000 iterations each, on the cases exactly as shipped, through
`ofgpu-lowmach` — the driver every command line in §1 and §1.1 now names. **Every solved quantity above reproduces to the printed
digit**:

| | (a) WF `constant` | (b) resolved `constant` |
|---|---|---|
| `T_w` | 317.497 K | 314.549 K |
| `T_b` | 293.251 K | 292.773 K |
| ΔT | 24.2454 K | 21.7767 K |
| `U_b` | 5.39407 m/s | 4.93682 m/s |
| Re | 28 768 | 26 330 |
| **`Nu`** | **64.4894** | **71.683** |
| `f` measured | 0.0171402 | 0.0238317 |
| thermostat vs wall heat | −3.20056 W vs 3.2 W (+0.0174 %) | −3.2 W vs 3.2 W (−2.84e-06 W) |
| the correction's own domain integral | −0.000556869 W | +8.85303e-08 W |
| `contErr` | 1.99200e-08 | 6.72602e-14 |
| kinematic drag balance | −0.005 % | −0.000 % |
| `y+` | 56.877 / 57.766 / 58.574 | worst 0.00179449, 192/400 below y+ 20 |
| Kays-Crawford `Pr_t` had it been asked for | [0.874935, 0.891896], vol-mean 0.88182 | [0.871299, 1.7000], vol-mean 0.906154 |
| verdicts (absolute / Reynolds / D-B) | −5.9 % ✔ / +34.3 % ✘ / −12.9 % ✔ | +11.9 % ✘ / +14.9 % ✘ / +4.0 % ✔ |

The one number that moves is §26.1's *prescribed half* of the correction —
9.78e-14 W against the recorded 2.90e-13 W on leg (a), −2.54e-14 W against
2.04e-13 W on leg (b). Both are the "exactly zero" that row exists to report,
and the digit that changed is the last one of a quantity whose whole claim is
that it is nought.

**The wall-time row was NOT re-measured**, and is left as the dated figure it
already was: another process held the GPU throughout this rerun, and identical
5 000-iteration runs of the same case came back anywhere between 30.6 s and
208.6 s. A wall time taken under that is not a measurement of anything. It is
also why no claim is made here about the two binaries' relative speed — they
execute the same code path on these cases and the machine could not resolve
the difference.

A shorter check stands behind the table: the two binaries were run back to
back on `cases/channelPeriodicFluxWF.jsonc`, `channelPeriodicWF.jsonc` and
`channelThermalWF.jsonc` and their console output diffed line by line. The only
differences are the driver's own name, the two species lines `ofgpu-lowmach`
does not print, and the wall-clock. Every residual,
temperature, density, flux, budget and friction figure is character-identical.

---
