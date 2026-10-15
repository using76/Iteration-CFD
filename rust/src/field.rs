// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.

//! `volScalarField` / `volVectorField` / `surfaceScalarField`, device resident.
//!
//! Provenance: original. The single Robin boundary representation below is
//! this project's own design, specified in SPEC-LIT.md section 4. The second
//! old time level (`f00`) is SPEC-LIT.md section 13.3. No GPL-licensed source
//! was consulted.
//!
//! Boundary conditions are stored in ONE universal form - the Robin (mixed)
//! form that every scalar boundary condition in this solver reduces to
//! (SPEC-LIT.md section 4; the representation is our design):
//!
//! ```text
//! psi_b = fr*refValue + (1 - fr)*(psi_c + refGrad/deltaCoeffs)
//! ```
//!
//! with `fr = 1` giving fixedValue, `fr = 0, g = 0` zeroGradient (and symmetry
//! or slip for a scalar), `fr = 0, g != 0` fixedGradient, and anything between
//! mixed or inletOutlet.
//!
//! That single representation is what makes the matrix coefficients
//! branch-free, which matters a lot on a GPU:
//!
//! ```text
//! valueInternalCoeffs    =  1 - fr
//! valueBoundaryCoeffs    =  fr*refValue + (1 - fr)*refGrad/deltaCoeffs
//! gradientInternalCoeffs = -fr*deltaCoeffs
//! gradientBoundaryCoeffs =  fr*deltaCoeffs*refValue + (1 - fr)*refGrad
//! ```
//!
//! A wall function is then nothing more than a kernel that rewrites `fr`,
//! `ref_value` and `ref_grad` on the faces it owns. No virtual dispatch, no
//! host involvement.
//!
//! # Names, and what happens to one this solver does not have
//!
//! [`BcKind::from_name`] is the *only* place a boundary-condition name is
//! interpreted. It follows SPEC-LIT.md §13.4: a name it implements is used, a
//! name it recognises but does not implement is an error naming the
//! alternatives, and a name it has never heard of is an error naming the
//! setting. It used to turn everything it did not know into `Calculated`,
//! which evaluates as a hard Dirichlet at whatever the file's `value` held -
//! so `turbulentIntensityKineticEnergyInlet`, `totalPressure` and the literal
//! string `garbageBC` all ran to completion and all gave the same answer.
//! Every public type name's status - implemented, alias, planned, owned by
//! another stream, or refused permanently - is the catalogue of SPEC-LIT
//! §120, held to the code by `field::tests::every_catalogue_name_has_a_status`.
//!
//! Extended from:
//!   ofgpu `SPEC-LIT.md` §4 (the triple), §13.4 (the rule above), §15.2 and
//!     §15.5 (which patches get a wall function, and `nutLowRe`)
//!   ofgpu `SPEC-LIT.md` §15.1, §15.3 and §29.2 - the `nutU`/`nutk` rough-wall
//!     variants below; the physics is `crate::wallfunctions`' and
//!     `crate::field_setup::NutRoughness`'s, this file only names them
//!   ofgpu `SPEC-LIT.md` §29.3 - `ThermalWallFunction` below, and the
//!     `compressible::alphatJayatillekeWallFunction` alias `from_name`
//!     accepts for it; the law itself is `crate::wallfunctions`' and
//!     `crate::energy`'s, this file only names the condition and the alias
//!   ofgpu `SPEC-LIT.md` §30.1 - `WernerWengleWallFunction` below, the LES
//!     wall model selected under `simulationType LES;`; the law itself is
//!     `crate::wallfunctions`' ([`crate::wallfunctions::WernerWengleData`]),
//!     this file only names the condition
//!   ofgpu `SPEC-LIT.md` §120.8 - the thermal family: the three `Mixed`-kind
//!     names and `turbulentHeatFluxTemperature` below, and the two
//!     `alphat`-only aliases
//! No GPL-licensed source was consulted.

use crate::device::{DevBuf, Gpu};
use crate::error::Result;
use crate::io::contract::{unsupported, unsupported_note};
use crate::mesh::GpuMesh;
use crate::{Acc, Label, Scalar, Vec3};

/// Which rule regenerates `(fr, ref_value, ref_grad)` each outer iteration.
///
/// Stored per boundary face as an `i32` so the kernels can switch on it.
#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BcKind {
    FixedValue = 0,
    ZeroGradient = 1,
    FixedGradient = 2,
    Mixed = 3,
    /// Value is written by the model, never solved for.
    Calculated = 4,
    /// 2-D front/back: contributes nothing anywhere.
    Empty = 5,
    /// Scalars: zeroGradient. Vectors: reflect the normal component.
    Symmetry = 6,
    /// Coupled; handled inside `amul`.
    Cyclic = 7,
    /// `fr = 1` where the face flux is inward, else 0.
    ///
    /// 8 to 12 are the FLUX-SWITCHED block: every one of them is Dirichlet on
    /// inflow and zero-gradient on outflow, and they differ only in where the
    /// inflow value comes from. `fldInletOutletFraction` in `cuda/field.cu`
    /// tests the whole range, so the block must stay contiguous.
    InletOutlet = 8,

    /// `k = 3/2 (I |U|)^2` on inflow (Launder & Spalding's definition of the
    /// turbulence intensity `I`); zero-gradient on outflow.
    TurbulentIntensityKineticEnergyInlet = 9,

    /// `epsilon = C_mu^{3/4} k^{3/2} / L` on inflow; zero-gradient on outflow.
    TurbulentMixingLengthDissipationRateInlet = 10,

    /// `omega = k^{1/2} / (C_mu^{1/4} L)` on inflow; zero-gradient on outflow.
    TurbulentMixingLengthFrequencyInlet = 11,

    /// Velocity at an open boundary next to a prescribed pressure: the flux
    /// sets the normal component on inflow, zero-gradient on outflow.
    PressureInletOutletVelocity = 12,

    /// Pressure whose surface-normal gradient is whatever makes the boundary
    /// flux come out as prescribed - a fixed-gradient condition whose gradient
    /// the pressure equation writes.
    FixedFluxPressure = 13,

    /// `p = p0 - |U|^2/2` on inflow, `p = p0` on outflow (kinematic pressure).
    TotalPressure = 14,

    /// The wall's own velocity, with the wall-normal component removed.
    MovingWallVelocity = 15,

    /// `U = -n Q/A`: a uniform normal velocity carrying the prescribed
    /// volumetric flow rate.
    FlowRateInletVelocity = 16,

    /// `nu_t = 0` at the wall: the mesh resolves the viscous sublayer and NO
    /// wall function is wanted (SPEC-LIT §15.2). This is 22 for history; the
    /// point of the name is that it is not a wall function at all, and
    /// [`Self::is_nut_wall_function`] says so.
    NutkWallFunction = 20,
    NutUWallFunction = 21,
    NutLowReWallFunction = 22,
    EpsilonWallFunction = 23,
    OmegaWallFunction = 24,
    /// zeroGradient wearing a wall-function name.
    KqRWallFunction = 25,
    KLowReWallFunction = 26,

    /// The Jayatilleke thermal wall function on temperature - SPEC-LIT §29.3.
    ///
    /// Rewrites the fixed-T Robin triple to encode the sublayer-resistance
    /// conductance `rho cp u_tau (T_w - T_P)/T+` instead of the molecular-only
    /// conductance a plain `fixedValue` implies. `T_w` is the file's `value`
    /// entry, exactly as the other wall-function kinds above read theirs.
    ThermalWallFunction = 27,

    /// `nutkWallFunction` with the Cebeci & Bradshaw roughness downshift of
    /// SPEC-LIT §15.3/§29.2: `y+` still comes from `k`, but `E` is replaced by
    /// `E_eff = E exp(-kappa dB(Ks+, Cs))` wherever it appears. `Ks = 0`
    /// collapses `dB` to zero and this reproduces [`Self::NutkWallFunction`]
    /// to round-off - the §22 gate.
    NutkRoughWallFunction = 28,

    /// `nutUWallFunction` with the same downshift, except `Ks+ = Cs Ks
    /// u_tau/nu` is now part of the unknown the §15.1 Newton solves for: it
    /// is recomputed from the current `u_tau` iterate every step, exactly
    /// like `u+` itself (SPEC-LIT §29.2). `Ks = 0` reproduces
    /// [`Self::NutUWallFunction`] to round-off.
    NutURoughWallFunction = 29,

    /// Werner & Wengle (1991) - SPEC-LIT §30.1, the LES wall model. Unlike
    /// every RAS wall function above, `nu_t,w` here comes from the
    /// analytically-invertible power law integrated over the first cell,
    /// fed by the wall-parallel CELL-AVERAGE velocity rather than by `k` -
    /// an LES has no `k` to read. Selected under `simulationType LES;` by
    /// the `standard`/`spalding` rows of §29.1's table (§30.1 remaps them
    /// both to this one wall model); `rough` has no LES wall model yet and
    /// is a §13.4 error naming this and `nutLowReWallFunction`.
    WernerWengleWallFunction = 30,

    /// A fixed wall heat FLUX, `q` (W/m^2) - SPEC-LIT §32.2's redesigned
    /// thermal-wall gate. `q` is read from the field file's own `q` entry
    /// (stored in `ref_value`, exactly as `ThermalWallFunction` stores its
    /// `T_w` there) and never rewritten, so it survives being read again
    /// every outer iteration; `ref_grad` is rewritten every iteration to
    /// `crate::energy::flux_to_grad(q, k_eff_wall)` with the CURRENT
    /// `k_eff_wall` `crate::energy::Energy::update_k_eff` just computed at
    /// that face - see [`Self::is_fixed_flux_temperature`] and
    /// `crate::energy::Energy::update_fixed_flux`.
    ///
    /// This is deliberately the SAME condition on a `lowRe` (resolved) wall
    /// and a wall-function wall: SPEC-LIT §32.2's own point is that
    /// `flux_to_grad(q, k_eff)` is exact WHATEVER `k_eff` is (a `fr = 0`
    /// Robin condition delivers exactly the flux it is given, independent of
    /// the conductivity used to construct it - the ratio cancels exactly
    /// against the same `k_eff` the matrix assembly multiplies by) - a
    /// SEPARATE Jayatilleke-corrected BcKind would add machinery this
    /// condition does not need for the FLUX itself. Where the Jayatilleke
    /// correction still matters is diagnosing the WALL TEMPERATURE a fixed
    /// flux produces, which is a postprocessing read of `crate::wallfunctions`'
    /// existing pure functions (`y_plus_of`/`t_plus`/`u_tau_of`), not a
    /// second device kernel - see SPEC-LIT §32.2 and
    /// `crate::wallfunctions`'s module doc on the fixed-q form "falling out
    /// of the same function".
    FixedFluxTemperature = 31,

    /// The contact angle on `alpha` at a wall - SPEC-LIT §39.3.
    ///
    /// A plain FIXED-GRADIENT condition in §4's triple: `fr = 0`,
    /// `refValue` carries `cos(theta0)` (the equilibrium angle, in the form
    /// every consumer wants it), and `refGrad` is rewritten every outer
    /// iteration to `|grad(alpha)_P| cos(theta)` by
    /// `vofAlphaContactAngleGrad` - exactly as
    /// [`Self::FixedFluxTemperature`] rewrites its own from `q`.
    ///
    /// **No new device branch.** `cuda/field.cu` consults `bcKind` for
    /// `calculated`, `cyclic` and vector `symmetry` only, and this is none of
    /// those, so it is evaluated by the same `fldMixed` every other condition
    /// is. The discriminant exists so the READER can tell which faces the
    /// model owns and what angle each was given; until the model writes
    /// `refGrad` it degenerates to zero-gradient, which is what a wall
    /// carried before §39 and what a wall function does too.
    ///
    /// `constantAlphaContactAngle` and `dynamicAlphaContactAngle` both map
    /// here; which one a patch named is carried by
    /// [`crate::contact_angle::ContactAnglePatch`], read out of the same
    /// entry.
    ContactAngle = 32,

    /// The conjugate fluid/solid interface - SPEC-LIT S47.
    ///
    /// A Robin triple like every other condition in S4, with `fr = h_G/C_A`,
    /// `refValue = T_Q` (the cell across the interface, read through the
    /// mesh's `b_nbr_cell`) and `refGrad = 0`, rewritten every outer
    /// iteration by [`crate::cht::ConjugateInterfaces::update`]. That one
    /// kernel writes BOTH sides of a pair from one `h_G` and one `|Sf|`, which
    /// is what makes the two fluxes cancel bitwise (S47.2 consequence 2).
    ///
    /// **No new device branch**, for the same reason
    /// [`Self::ContactAngle`] needs none: the discriminant is outside every
    /// range `cuda/field.cu` consults, so `fldCorrectBcScalar` evaluates it
    /// with the same `fldMixed` as everything else, and
    /// `fldMixed(fr, T_Q, 0, T_P, Delta) = T_P - fr (T_P - T_Q)` is exactly
    /// (S47.7)'s face value - the only form that can carry the contact
    /// resistance's temperature JUMP. The cyclic branch, which interpolates
    /// `psi[nbr]` geometrically, cannot.
    ///
    /// **This condition CONTAINS the thermal wall function.** SPEC-LIT S47.6:
    /// on a wall-function fluid mesh the cell-to-face conductance is
    /// `rho c_p u_tau/T+`, not `k_eff Delta`, so a face cannot be both
    /// `thermalWallFunction` and `coupledTemperature` - they rewrite the same
    /// triple. Which conductance law a given interface face uses is selected
    /// from the `nut` patch type on that face, S15.5's rule extended once
    /// more.
    CoupledTemperature = 33,

    /// The surface-to-surface radiating wall - SPEC-LIT S50.3.
    ///
    /// A Robin triple like every other condition in S4, rewritten every
    /// outer iteration by [`crate::s2s::S2s::update`] from (S50.12):
    ///
    /// ```text
    /// h        = 4 eps sigma T0^3
    /// fr       = h/(h + k_eff Delta_b)
    /// refValue = (3/4) T0 + H_b/(4 sigma T0^3)
    /// refGrad  = q_ext/k_eff
    /// ```
    ///
    /// The local emission is IMPLICIT through `fr` (Patankar's `Sp <= 0`
    /// rule, satisfied unconditionally because `T` is absolute); the
    /// incoming irradiation `H_b` is lagged into `refValue`.
    ///
    /// **The emissivity does not appear in `refValue`.** That is not
    /// bookkeeping luck: it is what makes `eps -> 0` collapse **bitwise**
    /// onto [`Self::FixedFluxTemperature`] - `fr` exactly `0.0` and
    /// `refGrad` exactly `q_ext/k_eff` - which is a S22-style "reproduces
    /// the simpler model" gate obtained for free. Choosing `refGrad = 0`
    /// instead of `q_ext/k_eff` would put `eps` back into `refValue` and
    /// destroy it.
    ///
    /// **No new device branch**, for the same reason [`Self::ContactAngle`]
    /// and [`Self::CoupledTemperature`] need none: the discriminant is
    /// outside every range `cuda/field.cu` consults, so `fldCorrectBcScalar`
    /// evaluates it with the same `fldMixed` as everything else. Until
    /// `crate::s2s` writes the triple it degenerates to zero-gradient -
    /// adiabatic, the safe default.
    ///
    /// `q_ext` and `eps` live in the S2S module's own buffers rather than in
    /// `ref_value` (where [`Self::FixedFluxTemperature`] keeps its `q`),
    /// because the stamp OVERWRITES `ref_value` every update.
    S2sWall = 34,

    /// The fan / blower patch on the PRESSURE - SPEC-LIT S52.
    ///
    /// A Robin triple like every other condition in S4, rewritten once per
    /// pressure corrector by [`crate::fan::FlowDevices::update`] from
    /// (S52.10)/(S52.11):
    ///
    /// ```text
    /// fr       = 1/(1 + S SIGMA_D)
    /// refValue = c + S Phi ,   c = p_a - sigma F* - S Q*
    /// refGrad  = 0
    /// ```
    ///
    /// `S = -dF/dQ_dev >= 0` is the fan-curve slope at the current operating
    /// point, `SIGMA_D = SUM_f rAU_f a_f Delta_f` and
    /// `Phi = SUM_f phi_HbyA,f` are two patch reductions, and `sigma` is the
    /// direction. **One triple for the whole patch** - the fan sets one
    /// pressure, not a profile, which is (S52.7)'s own content.
    ///
    /// **This is a symmetric rank-1 downdate, lumped.** S52.2 shows the
    /// patch-integral coupling is `A = diag(D) - kappa d d^T`, exactly
    /// symmetric, and S52.3 shows that lumping it onto the diagonal
    /// preserving the row sum gives exactly the triple above. Symmetry
    /// survives, the M-matrix property survives (the diagonal gains
    /// `fr D_f >= 0`), and no new storage or matrix kernel is needed.
    ///
    /// **`S = 0` is `fixedValue`, BITWISE.** `S SIGMA_D` is exactly `0.0`, so
    /// `fr = 1.0/1.0 = 1.0` and `refValue = c + 0.0*Phi = c`. A flat curve
    /// reproduces the existing condition bit for bit, which is the
    /// regression gate for the whole section (S52.4).
    ///
    /// **No new device branch**, for the same reason [`Self::ContactAngle`]
    /// and [`Self::S2sWall`] need none: the discriminant is outside every
    /// range `cuda/field.cu` consults, so `fldCorrectBcScalar` evaluates it
    /// with the same `fldMixed` as everything else. Until `crate::fan` writes
    /// the triple it degenerates to zero-gradient.
    FanPressure = 35,

    /// A porous jump against a prescribed plenum pressure - SPEC-LIT S53.3.
    ///
    /// The SAME algebra as [`Self::FanPressure`] with `S SIGMA_D -> R D_f`:
    ///
    /// ```text
    /// fr       = 1/(1 + R D_f)
    /// refValue = p_plenum
    /// refGrad  = 0
    /// ```
    ///
    /// `R = (nu t_m/alpha + C2 t_m |phi_f|/(2 a_f))/a_f >= 0` is the sheet's
    /// resistance, evaluated from the previous iterate (a Picard
    /// linearisation of the Forchheimer half). `R = 0` gives `fr = 1.0`
    /// exactly - a plain `fixedValue` at the plenum pressure, bitwise - and
    /// `R -> infinity` gives `fr -> 0` with `refGrad = 0`, which is a wall.
    ///
    /// The INTERNAL form of the same jump is not a `BcKind` at all: it is a
    /// per-face division of `rAU_f`, `rAU_f|Sf|` and `phi_HbyA,f` by the same
    /// `(1 + R D_f)`, with `fvm_laplacian` called unmodified (S53.2).
    ///
    /// **No new device branch**, for the same reason as above.
    PorousJumpPressure = 36,

    /// The tensorial mixed condition of SPEC-LIT 120.3 - `directionMixed`,
    /// `partialSlip`, `fixedNormalSlip` and `fixedShearStress` all ride it:
    ///
    /// ```text
    /// U_b = F.r + (I - F).(U_P + g/Delta_b)                    (120.3a)
    /// ```
    ///
    /// with `F` a symmetric value-fraction tensor, `r` a reference value and
    /// `g` a reference gradient, held per face in the 12-entry
    /// [`GpuVectorField::dm`] block `F_xx F_xy F_xz F_yy F_yz F_zz r_x r_y
    /// r_z g_x g_y g_z`. A per-component scalar `fr` cannot carry `F`, so
    /// what the matrix sees is (120.3h): `fr = f_t = (tr F - n.F.n)/2`,
    /// `refValue = r`, `refGrad = g` - the TANGENTIAL part of (120.3a)
    /// enters the matrix implicitly, through the one triple of §4, and the
    /// boundary kernel writes the face value (120.3a) into `bf` only.
    /// `F = I` is fixedValue, `F = 0` is fixedGradient, `F = n n` fixes the
    /// normal component and leaves the tangential one free. The host twin of
    /// the device evaluation is
    /// [`direction_mixed_value`](crate::field::direction_mixed_value).
    DirectionMixed = 37,

    /// `outletInlet`, the reverse switch of `inletOutlet` - a fixed value
    /// where the flow LEAVES, zero-gradient where it comes back in:
    ///
    /// ```text
    /// fr = 1 if phi_b > 0 (outflow), fr = 0 otherwise                    (120.4a)
    /// refValue = outletValue, refGrad = 0
    /// ```
    ///
    /// Its own kind, OUTSIDE the flux-switched block `[8, 12]` (whose kinds
    /// are all Dirichlet on INflow), so `fldInletOutletFraction` and its host
    /// twin `field_setup::update_inlet_outlet` test it by its own value with
    /// the strict `phi_b > 0` of (120.4a). A face with exactly zero flux is
    /// zero-gradient under both conditions.
    OutletInlet = 38,

    /// `advective`, Orlanski's (1976) radiation outlet with the phase speed
    /// GIVEN rather than estimated: the face relaxes toward a far value along
    /// the outward normal at the wave speed `w` (120.4b), and `field::advective`
    /// rewrites the §4 triple from (120.4d) - a Robin face like any other, so
    /// no evaluation branch of its own. The wave speed and the rate are FACE
    /// -FIELD inputs, not properties of the kind: CMP-17 passes its own
    /// `w = u_n + c` and `K = sigma c (1 - M_n^2)/L` of (126.6) to the same
    /// kernels. `field_setup` refuses the name unless the caller sets
    /// [`crate::field_setup::BcInputs::advective`], the promise that
    /// `field::advective` refreshes the condition every step.
    Advective = 39,

    /// A wedge side (SPEC-LIT 119.2), the rotational self-couple BC-04:
    /// the cell across the face is the cell itself turned about the axis,
    /// so on a vector the face value is the chord average
    ///
    /// ```text
    /// U_b = (U_P + R(sigma theta) U_P)/2 = U_P + w                       (119.2c)
    /// ```
    ///
    /// reached with NO new evaluation branch - `cuda/field.cu`'s
    /// `fldWedgeRefGrad` rewrites `refGrad = Delta_b w` (119.2d) from the
    /// current cell value before the section-4 evaluation, and the face
    /// value comes out of the one expression. A scalar is zero-gradient,
    /// which the seeded triple already is. The flux is prescribed by the
    /// velocity condition, `phi_b = U_b . Sf` (119.2e) - the swirl through
    /// the side, which is not zero. The mesh has the last word, as for
    /// `empty` and `cyclic`: a patch it types `wedge` carries this kind on
    /// every field, and `type wedge` on a patch it does not is refused by
    /// name. The host twin of the rewrite is
    /// [`wedge_ref_grad`](crate::field::wedge_ref_grad).
    Wedge = 40,

    /// `pressureInletVelocity`, the first of the five pressure-driven
    /// velocity inlets of SPEC-LIT 120.5: on a patch whose PRESSURE is
    /// given, the velocity is whatever the mass flux through the face makes
    /// it (Ferziger & Peric 2002). All five are a section-4 triple with
    /// `refGrad = 0`, rewritten once per outer iteration from the flux the
    /// last corrector left, by `fldPressureInletValue` (host twin
    /// [`pressure_inlet_values`]) before the boundary evaluation:
    ///
    /// ```text
    /// U_b = S_f phi_b/(S_f . S_f),  fr = 1                        (120.5e)
    /// ```
    ///
    /// `momFluxIsPrescribed` answers FALSE for kinds `[41, 45]` whatever
    /// their `fr`, so the pressure correction reaches the face. A caller
    /// opts in with [`crate::field_setup::BcInputs::pressure_inlet`]; every
    /// other setup refuses the name (SPEC-LIT 120.5).
    PressureInletVelocity = 41,

    /// `pressureInletUniformVelocity` (SPEC-LIT 120.5, (120.5f)): the
    /// patch-uniform form - `U_b = (S_f/|S_f|) PHI_p/A_p`, the patch sums
    /// `PHI_p = SUM phi_f` and `A_p = SUM |S_f|` run on one device thread
    /// per patch head, in `ofacc` face order.
    PressureInletUniformVelocity = 42,

    /// `pressureDirectedInletVelocity` (SPEC-LIT 120.5, (120.5g)): the flux
    /// enters along `inletDirection`, `U_b = d_f phi_b/(d_f . S_f)`, with
    /// the normalised `d_f` per face in the field's `dm` block's first
    /// three slots.
    PressureDirectedInletVelocity = 43,

    /// `pressureDirectedInletOutletVelocity` (SPEC-LIT 120.5, (120.5h)):
    /// (120.5g) where `phi_b < 0` (`fr = 1`), zeroGradient where
    /// `phi_b >= 0` (`fr = 0`) - switched by `fldInletOutletFraction` on
    /// the same strict `phi_b < 0` as the flux-switched block `[8, 12]`.
    PressureDirectedInletOutletVelocity = 44,

    /// `pressureInletOutletParSlipVelocity` (SPEC-LIT 120.5, (120.5i)): the
    /// parallel-slip inflow - the normal part of (120.5e) plus the
    /// tangential part of the cell value, `U_b = S_f phi_b/(S_f . S_f) +
    /// U_P - S_f (S_f . U_P)/(S_f . S_f)` where `phi_b < 0`, zeroGradient
    /// where `phi_b >= 0`.
    PressureInletOutletParSlipVelocity = 45,

    /// `fixedMean` (SPEC-LIT 120.6, (120.6a)): the cell profile with one
    /// constant `s` added, chosen so the area-weighted patch mean is the
    /// target `meanValue` - `refValue = psi_P + s`, `s = (meanValue A -
    /// M)/A` from the patch sums `A = SUM |S_f|`, `M = SUM |S_f| psi_P`.
    /// One rule per patch, rewritten once per outer iteration by
    /// `fldPatchMeanScalar`/`fldPatchMeanVector` wherever the caller opted
    /// in with [`crate::field_setup::BcInputs::patch_means`]; host twin
    /// [`crate::field::patch_mean::patch_mean_values_scalar`].
    FixedMean = 46,

    /// `fixedMeanOutletInlet` (SPEC-LIT 120.6, (120.6b)): (120.6a)'s mean
    /// held where the flow LEAVES - `fr = 1` where `phi_f > 0`, zeroGradient
    /// where it comes back in - so the inflow faces carry `psi_b = psi_P`
    /// exactly and the patch mean is still `meanValue` whenever any face is
    /// an outflow.
    FixedMeanOutletInlet = 47,

    /// `fixedFluxExtrapolatedPressure` (SPEC-LIT 120.6, (120.6f)): a
    /// fixed-gradient pressure whose gradient is the cell's own `grad p`
    /// carried onto the face - `refGrad_f = S_f . (grad p)_P/|S_f|`, exact
    /// for a linear `p`. Scalar only; rewritten once per outer iteration by
    /// `fldExtrapolatedSnGrad` from the momentum predictor's gradient.
    FixedFluxExtrapolatedPressure = 48,

    /// `outletMappedUniformInlet` (SPEC-LIT 120.6, (120.6c)): one uniform
    /// value on every face - `fraction` times the flux-weighted mean over
    /// the outflow faces of the patch `outletPatch` (its area mean where the
    /// patch has no outflow), plus `offset`. Fixed point of a loop duct,
    /// (120.6d): `U = offset/(1 - fraction)`.
    OutletMappedUniformInlet = 49,

    /// `matchedFlowRateOutletVelocity` (SPEC-LIT 120.6, (120.6e)): the cell
    /// velocity shifted along the face normal by one constant `c`, chosen so
    /// the outlet carries out exactly the flow rate entering
    /// `inletPatch` - `refValue = U_P + c S_f/|S_f|`, `c = (Q - F)/A`.
    /// Vector only.
    MatchedFlowRateOutletVelocity = 50,

    /// `nutkAtmRoughWallFunction` (SPEC-LIT 120.9, (120.9g)): the `k`-based
    /// wall value of a ground whose roughness LENGTH `z0` the Richards &
    /// Hoxey (1993) profile is written in - `u* = C_mu^{1/4} sqrt(k_P)`,
    /// `nu_t,w = max(0, u* kappa y / ln(1 + y/z0) - nu)`, so the wall shear
    /// is `u*^2` wherever the cell sits on the profile. A wall-function kind:
    /// the triple degenerates to the cell value until `wfNutWall` writes it.
    NutkAtmRoughWallFunction = 51,

    /// `nutUBlendedWallFunction` (SPEC-LIT 120.9, (120.9h)): the `nutU`
    /// family with `u_tau` the binomial blend `(u_v^n + u_l^n)^{1/n}` of the
    /// viscous-sublayer and log-law friction velocities, `n` from the patch
    /// entry (4 when absent). A wall-function kind, velocity-based.
    NutUBlendedWallFunction = 52,
}

/// The flux-switched block, `[FLUX_SWITCHED_FIRST, FLUX_SWITCHED_LAST]`.
///
/// Every kind in it is Dirichlet where the face flux is inward and
/// zero-gradient where it is outward; they differ only in what the inflow
/// value is. `fldInletOutletFraction` in `cuda/field.cu` tests the range, and
/// `field_ops::bc_kind_values_match_the_device` pins these two numbers to the
/// `OFGPU_BC_INLET_OUTLET_*` macros there.
pub const FLUX_SWITCHED_FIRST: Label = BcKind::InletOutlet as Label;
pub const FLUX_SWITCHED_LAST: Label = BcKind::PressureInletOutletVelocity as Label;

/// The five pressure-driven velocity inlets, `[PRESSURE_INLET_FIRST,
/// PRESSURE_INLET_LAST]` (SPEC-LIT 120.5). `momFluxIsPrescribed` in
/// `cuda/momentum.cu` answers FALSE for every kind in the range whatever its
/// `fr`, and `fldPressureInletValue` in `cuda/field.cu` rewrites their
/// `ref_value` from the flux; `bc_kind_values_match_the_device` pins both
/// numbers to the `OFGPU_BC_PRESSURE_INLET_*` macros there.
pub const PRESSURE_INLET_FIRST: Label = BcKind::PressureInletVelocity as Label;
pub const PRESSURE_INLET_LAST: Label = BcKind::PressureInletOutletParSlipVelocity as Label;

/// Every condition [`BcKind::from_name`] accepts, for the diagnostic a
/// rejected name gets. Kept next to the match so the two cannot drift.
pub const IMPLEMENTED_BC_NAMES: &[&str] = &[
    "fixedValue",
    "noSlip",
    "zeroGradient",
    "fixedGradient",
    "uniformFixedGradient",
    "mixed",
    "freestream",
    "freestreamVelocity",
    "freestreamPressure",
    "calculated",
    "empty",
    "symmetry",
    "symmetryPlane",
    "slip",
    "wedge",
    "cyclic",
    "cyclicSlip",
    "processor",
    "cyclicAMI",
    "inletOutlet",
    "turbulentIntensityKineticEnergyInlet",
    "turbulentMixingLengthDissipationRateInlet",
    "turbulentMixingLengthFrequencyInlet",
    "pressureInletOutletVelocity",
    "fixedFluxPressure",
    "totalPressure",
    "movingWallVelocity",
    "flowRateInletVelocity",
    "nutkWallFunction",
    "nutUWallFunction",
    "nutLowReWallFunction",
    "epsilonWallFunction",
    "omegaWallFunction",
    "kqRWallFunction",
    "kLowReWallFunction",
    "thermalWallFunction",
    "nutkRoughWallFunction",
    "nutURoughWallFunction",
    "wernerWengleWallFunction",
    "fixedFluxTemperature",
    "constantAlphaContactAngle",
    "dynamicAlphaContactAngle",
    "coupledTemperature",
    "thermalContactResistance",
    "greyDiffusiveRadiationViewFactor",
    "s2sWall",
    "fanPressure",
    "fan",
    "porousJumpPressure",
    "porousBafflePressure",
    // SPEC-LIT 120.1: the Function1 conditions of the time-varying family.
    "uniformFixedValue",
    "uniformInletOutlet",
    "uniformTotalPressure",
    "fixedProfile",
    // SPEC-LIT 120.3: the wall and slip family. `extrapolatedCalculated` is
    // zeroGradient; the four tensorial names ride `BcKind::DirectionMixed`;
    // the two moving walls ride `FixedValue`.
    "extrapolatedCalculated",
    "directionMixed",
    "rotatingWallVelocity",
    "translatingWallVelocity",
    "partialSlip",
    "fixedNormalSlip",
    "fixedShearStress",
    // SPEC-LIT 120.4: the two outlets. `outletInlet` is (120.4a)'s reverse
    // flux switch; `advective` is (120.4b)-(120.4d)'s Robin outlet, reached
    // only where a caller refreshes it every step (BcInputs::advective).
    "outletInlet",
    "advective",
    // SPEC-LIT 120.5: the velocity-inlet family. The three fixed-value
    // profiles ride `BcKind::FixedValue` with a per-face value the setup
    // computes; the five pressure-driven kinds rewrite their triple from the
    // flux every outer iteration, where the caller opts in.
    "surfaceNormalFixedValue",
    "cylindricalInletVelocity",
    "swirlFlowRateInletVelocity",
    "pressureInletVelocity",
    "pressureInletUniformVelocity",
    "pressureDirectedInletVelocity",
    "pressureDirectedInletOutletVelocity",
    "pressureInletOutletParSlipVelocity",
    // SPEC-LIT 120.6: the Outlets II family - four patch-mean rewrites and
    // an extrapolated pressure gradient, refreshed every outer iteration
    // where the caller opts in with `BcInputs::patch_means`.
    "fixedMean",
    "fixedMeanOutletInlet",
    "fixedFluxExtrapolatedPressure",
    "outletMappedUniformInlet",
    "matchedFlowRateOutletVelocity",
    // SPEC-LIT 120.7: the buoyant pressure family - a static pressure at a
    // patch, written in the solved `p_rgh` from the caller's `PrghInputs`.
    "prghPressure",
    "prghTotalPressure",
    "prghTotalHydrostaticPressure",
    "uniformDensityHydrostaticPressure",
    // SPEC-LIT 120.8: the thermal family - three Robin triples `Energy`
    // rewrites from the face's current conductance every outer iteration,
    // and `turbulentHeatFluxTemperature`, which rides `fixedFluxTemperature`
    // as it is.
    "externalWallHeatFluxTemperature",
    "turbulentHeatFluxTemperature",
    "lumpedMassTemperature",
    "totalFlowRateAdvectiveDiffusive",
    // SPEC-LIT 120.9: the atmospheric family. The four Richards & Hoxey
    // inlet profiles ride `BcKind::FixedValue` with a per-face value the
    // setup computes; the two wall functions are wall-function kinds that
    // `wfNutWall` writes.
    "atmBoundaryLayerInletVelocity",
    "atmBoundaryLayerInletK",
    "atmBoundaryLayerInletEpsilon",
    "atmBoundaryLayerInletOmega",
    "nutkAtmRoughWallFunction",
    "nutUBlendedWallFunction",
    // SPEC-LIT 120.10: the recycled inflow. All three ride
    // `BcKind::FixedValue`; their refValue is rewritten by `fldMapped`
    // every outer iteration (BcInputs::mapped) or by `fldTurbulentInlet`
    // every time step (BcInputs::turbulent_inlet).
    "mapped",
    "mappedFixedValue",
    "turbulentInlet",
    // SPEC-LIT 120.12: the synthetic LES inflow. Both ride
    // `BcKind::FixedValue` on a vector field; their refValue is rewritten by
    // `fldDigitalFilterInlet` / `fldDfsemInlet` every time step
    // (BcInputs::turbulent_inlet).
    "turbulentDigitalFilterInlet",
    "turbulentDFSEMInlet",
    // SPEC-LIT 120.11: the inflow database. A `FixedValue` whose refValue
    // the `TimedBcs` of the `*_timed` setup rewrites after each clock step.
    "timeVaryingMappedFixedValue",
    // SPEC-LIT 120.2: the two cyclic jumps. Both ride `BcKind::Cyclic` on a
    // pressure: the pair's per-face jump array, not the triple, carries the
    // jump, and `field_setup` builds it on a cyclic pair and refuses the
    // names anywhere else.
    "fixedJump",
    "uniformJump",
    // SPEC-LIT 120.14: the VOF family. Each rides a plain triple kind -
    // `Mixed` on the phase fraction and on `p_rgh`, `FixedValue` on the
    // velocity - whose triple `Vof::refresh_vof_family` rewrites from the
    // current phase fraction once per time step (BcInputs::vof_family).
    "variableHeightFlowRate",
    "variableHeightFlowRateInletVelocity",
    "outletPhaseMeanVelocity",
    "phaseHydrostaticPressure",
];

/// The host twin of the device evaluation of (120.3a) - SPEC-LIT 120.3.
///
/// `dm` is the 12-entry per-face block
/// `[F_xx F_xy F_xz F_yy F_yz F_zz r_x r_y r_z g_x g_y g_z]`, `u_p` the
/// wall-adjacent cell value and `delta` the boundary delta coefficient of
/// SPEC-LIT 2.4. The SAME operation order the kernel in `cuda/field.cu`
/// runs: `q = U_P + g/Delta_b` (the `g/Delta_b` term 0 where `Delta_b = 0`),
/// `a = F.r`, `b = F.q`, `U_b = a + (q - b)`, each matrix-vector product
/// summed `F_x0 v_0 + F_x1 v_1 + F_x2 v_2` left to right. The device is
/// compiled with FMA contraction on, so host and device agree to round-off,
/// not bitwise - SPEC-LIT 120.3 records that.
#[inline]
pub fn direction_mixed_value(dm: &[Scalar], u_p: Vec3, delta: Scalar) -> Vec3 {
    debug_assert!(dm.len() >= 12, "a DirectionMixed block has 12 entries");

    let fxx = dm[0];
    let fxy = dm[1];
    let fxz = dm[2];
    let fyy = dm[3];
    let fyz = dm[4];
    let fzz = dm[5];
    let (rx, ry, rz) = (dm[6], dm[7], dm[8]);
    let (gx, gy, gz) = (dm[9], dm[10], dm[11]);

    // q = U_P + g/Delta_b, the guard of section 4 on a degenerate face.
    let (qx, qy, qz) = if delta != 0.0 {
        (u_p.x + gx / delta, u_p.y + gy / delta, u_p.z + gz / delta)
    } else {
        (u_p.x, u_p.y, u_p.z)
    };

    // a = F.r and b = F.q, each row summed left to right.
    let a = Vec3::new(
        fxx * rx + fxy * ry + fxz * rz,
        fxy * rx + fyy * ry + fyz * rz,
        fxz * rx + fyz * ry + fzz * rz,
    );
    let b = Vec3::new(
        fxx * qx + fxy * qy + fxz * qz,
        fxy * qx + fyy * qy + fyz * qz,
        fxz * qx + fyz * qy + fzz * qz,
    );

    Vec3::new(a.x + (qx - b.x), a.y + (qy - b.y), a.z + (qz - b.z))
}

/// The host twin of `fldWedgeRefGrad` - SPEC-LIT 119.2, (119.2c)-(119.2d).
///
/// The wedge condition is a rotational self-couple: the cell across the
/// face is the cell itself turned about the axis by `sigma theta`
/// (`sigma = +1` on the front patch, `-1` on the back), so the chord
/// average is `U_b = (U_P + R(sigma theta) U_P)/2 = U_P + w` and the
/// condition is the section-4 triple `fr = 0, refValue = 0,
/// refGrad = Delta_b w` with
///
/// ```text
/// w = ( sigma sin(theta) (a x U_P) - vers(theta) (U_P - a (a . U_P)) ) / 2
/// ```
///
/// `vers_theta` is `vers(theta) = 1 - cos(theta) = 2 sin^2(theta/2)`. The
/// SAME operation order the kernel runs: `d = a . U_P` summed x, y, z left
/// to right; `c = a x U_P`; `q = U_P - a d`;
/// `w = (sigma sin(theta) c - vers(theta) q)/2`; `refGrad = Delta_b w`.
/// `a`, `sin(theta)` and `vers(theta)` are formed in f64 from 119.1's f64
/// wedge and cast once (SPEC-LIT 118.2). The device contracts FMA, so host
/// and device agree to round-off, not bitwise.
#[inline]
pub fn wedge_ref_grad(
    u_p: Vec3,
    axis: Vec3,
    sin_theta: Scalar,
    vers_theta: Scalar,
    sigma: Scalar,
    delta: Scalar,
) -> Vec3 {
    let d = axis.x * u_p.x + axis.y * u_p.y + axis.z * u_p.z;
    let c = Vec3::new(
        axis.y * u_p.z - axis.z * u_p.y,
        axis.z * u_p.x - axis.x * u_p.z,
        axis.x * u_p.y - axis.y * u_p.x,
    );
    let q = Vec3::new(
        u_p.x - axis.x * d,
        u_p.y - axis.y * d,
        u_p.z - axis.z * d,
    );
    let s = sigma * sin_theta;
    let w = Vec3::new(
        (s * c.x - vers_theta * q.x) / 2.0,
        (s * c.y - vers_theta * q.y) / 2.0,
        (s * c.z - vers_theta * q.z) / 2.0,
    );
    Vec3::new(delta * w.x, delta * w.y, delta * w.z)
}

/// The host twin of `fldPressureInletValue` - SPEC-LIT 120.5,
/// (120.5e)-(120.5i). Indexed by boundary face; `u_p[i]` is the value of
/// the cell next to face `i`. Writes `ref_value[i]` only where
/// `kinds[i]` is one of the five pressure-driven kinds `[41, 45]`; every
/// other face is untouched, so the whole boundary can be swept.
///
/// The SAME operation order the kernel runs: `S.S` summed `x*x + y*y +
/// z*z` left to right, the flux divided in first and the quotient scaled
/// onto the vector, (120.5f)'s patch sums run face by face in face order
/// in `Acc` (f64, both builds) and `PHI_p/A_p` rounded once. The device
/// contracts FMA, so host and device agree to round-off, not bitwise.
///
/// # Panics
/// Only through the slice bounds: `ref_value`, `b_sf`, `b_mag_sf`,
/// `phi_b`, `u_p` and `kinds` must each cover the boundary, and `dm` the
/// 12-entry block of every face the kinds 43/44 read.
#[allow(clippy::too_many_arguments)]
pub fn pressure_inlet_values(
    kinds: &[Label],
    b_patch: &[Label],
    phi_b: &[Scalar],
    b_sf: &[Vec3],
    b_mag_sf: &[Scalar],
    dm: &[Scalar],
    u_p: &[Vec3],
    ref_value: &mut [Vec3],
) {
    let n = ref_value.len();
    for i in 0..n {
        let kind = kinds[i];
        if !(PRESSURE_INLET_FIRST..=PRESSURE_INLET_LAST).contains(&kind) {
            continue;
        }
        let sf = b_sf[i];
        let phi = phi_b[i];
        // The patch-uniform form (120.5f): one thread per patch head -
        // a face whose predecessor is not the same patch carrying the
        // same kind - sums the patch's phi and |S_f| in face order, then
        // a second loop writes every face of the patch. Other kind-42
        // faces are written by their head, never by themselves.
        if kind == BcKind::PressureInletUniformVelocity as Label {
            let head = i == 0
                || b_patch[i - 1] != b_patch[i]
                || kinds[i - 1] != BcKind::PressureInletUniformVelocity as Label;
            if !head {
                continue;
            }
            let mut acc_phi: Acc = 0.0;
            let mut acc_a: Acc = 0.0;
            let mut j = i;
            while j < n
                && b_patch[j] == b_patch[i]
                && kinds[j] == BcKind::PressureInletUniformVelocity as Label
            {
                acc_phi += phi_b[j] as Acc;
                acc_a += b_mag_sf[j] as Acc;
                j += 1;
            }
            let ubar = if acc_a != 0.0 {
                (acc_phi / acc_a) as Scalar
            } else {
                0.0 as Scalar
            };
            for f in i..j {
                let m = b_mag_sf[f];
                let k = if m != 0.0 { ubar / m } else { 0.0 };
                ref_value[f] = Vec3::new(b_sf[f].x * k, b_sf[f].y * k, b_sf[f].z * k);
            }
            continue;
        }
        let rv = match kind {
            k if k == BcKind::PressureInletVelocity as Label => {
                let ss = sf.x * sf.x + sf.y * sf.y + sf.z * sf.z;
                if ss != 0.0 {
                    let q = phi / ss;
                    Vec3::new(sf.x * q, sf.y * q, sf.z * q)
                } else {
                    Vec3::ZERO
                }
            }
            k if k == BcKind::PressureDirectedInletVelocity as Label
                || k == BcKind::PressureDirectedInletOutletVelocity as Label =>
            {
                let d = Vec3::new(dm[12 * i], dm[12 * i + 1], dm[12 * i + 2]);
                let ds = d.x * sf.x + d.y * sf.y + d.z * sf.z;
                if ds != 0.0 {
                    let q = phi / ds;
                    Vec3::new(d.x * q, d.y * q, d.z * q)
                } else {
                    Vec3::ZERO
                }
            }
            k if k == BcKind::PressureInletOutletParSlipVelocity as Label => {
                let ss = sf.x * sf.x + sf.y * sf.y + sf.z * sf.z;
                let d = sf.x * u_p[i].x + sf.y * u_p[i].y + sf.z * u_p[i].z;
                if ss != 0.0 {
                    let q = phi / ss;
                    let t = d / ss;
                    Vec3::new(
                        sf.x * q + u_p[i].x - sf.x * t,
                        sf.y * q + u_p[i].y - sf.y * t,
                        sf.z * q + u_p[i].z - sf.z * t,
                    )
                } else {
                    Vec3::ZERO
                }
            }
            _ => Vec3::ZERO,
        };
        // 44 and 45 hold their Dirichlet form only where the flux comes
        // IN (`phi_b < 0`, strict); the outflow half is zero-gradient,
        // whose refValue the section-4 evaluation never reads - the same
        // strict test `fldInletOutletFraction` switches the fraction by.
        let inflow = phi < 0.0;
        let switched = kind == BcKind::PressureDirectedInletOutletVelocity as Label
            || kind == BcKind::PressureInletOutletParSlipVelocity as Label;
        ref_value[i] = if switched && !inflow { Vec3::ZERO } else { rv };
    }
}

impl BcKind {
    /// Map an OpenFOAM boundary-condition type string, per SPEC-LIT §13.4.
    ///
    /// A name this solver does not implement is an **error** naming the
    /// setting and listing what is available. `-permissive` downgrades that to
    /// a warning and substitutes `calculated` - a Dirichlet at whatever the
    /// file's `value` entry held - which is what this function used to do
    /// silently for every name it had never heard of.
    ///
    /// `patch` is only for the diagnostic; a user with forty patches needs to
    /// be told which one is wrong.
    pub fn from_name(name: &str, field: &str, patch: &str) -> Result<Self> {
        let k = match name {
            "fixedValue" | "noSlip" => Self::FixedValue,
            // SPEC-LIT 120.1: the Function1 conditions ride as the condition
            // they degenerate to; `field_setup` reads their entry and refuses
            // a time-varying one where the clock does not advance.
            "uniformFixedValue" | "fixedProfile" => Self::FixedValue,
            "zeroGradient" => Self::ZeroGradient,
            "fixedGradient" | "uniformFixedGradient" => Self::FixedGradient,
            "mixed" | "freestream" | "freestreamVelocity" => Self::Mixed,
            // Its own condition now, and only for a field that says so: a
            // value written by a model and never solved for.
            "calculated" => Self::Calculated,
            "empty" => Self::Empty,
            // SPEC-LIT 119.2: `wedge` is its own condition, the rotational
            // self-couple BC-04 - accepted on any patch by NAME, but a patch
            // the mesh does not type `wedge` refuses the entry in
            // `field_setup` (the mesh has the last word, as for `empty`).
            "symmetry" | "symmetryPlane" | "slip" => Self::Symmetry,
            "wedge" => Self::Wedge,
            "cyclic" | "cyclicSlip" | "processor" => Self::Cyclic,
            // SPEC-LIT 120.13: a static arbitrary mesh interface. The polyMesh
            // reader imprints its two patches into one conformal pair
            // (SPEC-LIT 120.18), so on the field it IS the conformal cyclic;
            // a moving mesh refuses it by name (`HostMesh::refuse_a_moving_cyclic_ami`).
            "cyclicAMI" => Self::Cyclic,
            "inletOutlet" | "freestreamPressure" | "uniformInletOutlet" => Self::InletOutlet,

            "turbulentIntensityKineticEnergyInlet" => {
                Self::TurbulentIntensityKineticEnergyInlet
            }
            "turbulentMixingLengthDissipationRateInlet" => {
                Self::TurbulentMixingLengthDissipationRateInlet
            }
            "turbulentMixingLengthFrequencyInlet" => Self::TurbulentMixingLengthFrequencyInlet,
            "pressureInletOutletVelocity" => Self::PressureInletOutletVelocity,
            "fixedFluxPressure" => Self::FixedFluxPressure,
            "totalPressure" | "uniformTotalPressure" => Self::TotalPressure,
            "movingWallVelocity" => Self::MovingWallVelocity,
            "flowRateInletVelocity" => Self::FlowRateInletVelocity,

            // SPEC-LIT 120.3: the wall and slip family. `extrapolatedCalculated`
            // IS zeroGradient on any field. The four tensorial names all carry
            // (120.3a) - the per-face F, r and g they each imply are read and
            // packed into `GpuVectorField::dm` by `field_setup`, which also
            // refuses the names where the entry they are defined by is
            // missing. The two moving walls are a fixedValue at the wall
            // velocity; `rotatingWallVelocity`'s wall-normal component is
            // removed by the setup that builds it, not by the kind.
            "extrapolatedCalculated" => Self::ZeroGradient,
            "directionMixed" | "partialSlip" | "fixedNormalSlip" | "fixedShearStress" => {
                Self::DirectionMixed
            }
            "rotatingWallVelocity" | "translatingWallVelocity" => Self::FixedValue,

            // SPEC-LIT 120.5: the velocity-inlet family. The three
            // fixed-value profiles ARE a fixedValue - `field_setup` computes
            // each face's value once from the case's own profile data
            // ((120.5a)-(120.5c)) and seeds `fr = 1`. The five
            // pressure-driven kinds keep kinds of their own: their triple is
            // rewritten from the flux every outer iteration by
            // `fldPressureInletValue`, and `momFluxIsPrescribed` answers
            // FALSE for the whole range `[41, 45]` so the pressure
            // correction reaches the face.
            "surfaceNormalFixedValue"
            | "cylindricalInletVelocity"
            | "swirlFlowRateInletVelocity" => Self::FixedValue,
            "pressureInletVelocity" => Self::PressureInletVelocity,
            "pressureInletUniformVelocity" => Self::PressureInletUniformVelocity,
            "pressureDirectedInletVelocity" => Self::PressureDirectedInletVelocity,
            "pressureDirectedInletOutletVelocity" => Self::PressureDirectedInletOutletVelocity,
            "pressureInletOutletParSlipVelocity" => Self::PressureInletOutletParSlipVelocity,

            // SPEC-LIT 120.6: the Outlets II family. Four of them hold a
            // PATCH INTEGRAL - a mean, a mean carried from another patch, or
            // a flow rate - and one extrapolates the cell's pressure
            // gradient onto the face; all five are a section-4 triple with
            // no new evaluation branch, rewritten once per outer iteration
            // by the fldPatchMean*/fldExtrapolatedSnGrad kernels wherever
            // the caller opted in with `BcInputs::patch_means`.
            "fixedMean" => Self::FixedMean,
            "fixedMeanOutletInlet" => Self::FixedMeanOutletInlet,
            "fixedFluxExtrapolatedPressure" => Self::FixedFluxExtrapolatedPressure,
            "outletMappedUniformInlet" => Self::OutletMappedUniformInlet,
            "matchedFlowRateOutletVelocity" => Self::MatchedFlowRateOutletVelocity,

            // SPEC-LIT 120.7: the buoyant pressure family. A static pressure
            // given at a patch, written in the SOLVED `p_rgh` by the setup's
            // 120.7 block - the two plain names ARE a fixedValue, and the two
            // total forms ride `TotalPressure`, as `totalPressure` does.
            "prghPressure" | "uniformDensityHydrostaticPressure" => Self::FixedValue,
            "prghTotalPressure" | "prghTotalHydrostaticPressure" => Self::TotalPressure,

            // SPEC-LIT 120.14: the VOF family. A section-4 triple with no new
            // evaluation branch: the phase fraction's clamp and the
            // phase-weighted hydrostatic pressure are `Mixed`, the two
            // velocities a `FixedValue`; `field_setup` refuses them unless
            // the caller promises the per-step refresh.
            "variableHeightFlowRate" | "phaseHydrostaticPressure" => Self::Mixed,
            "variableHeightFlowRateInletVelocity" | "outletPhaseMeanVelocity" => Self::FixedValue,

            // SPEC-LIT 120.3: OpenFOAM's Spalding spelling of the nut-U wall
            // function. `nutUWallFunction` already applies Spalding's single
            // formula (SPEC-LIT 15.1), so the name is an alias, printed once -
            // the same treatment `compressible::alphatJayatillekeWallFunction`
            // below gets.
            "nutUSpaldingWallFunction" => {
                crate::io::contract::warn_once(
                    &format!("{field}: nutUSpaldingWallFunction"),
                    &format!(
                        "{field}: boundaryField/{patch}/type `nutUSpaldingWallFunction` \
                         (OpenFOAM's Spalding spelling; `nutUWallFunction` already \
                         applies Spalding's law, SPEC-LIT 15.1) mapped to \
                         `nutUWallFunction`"
                    ),
                );
                Self::NutUWallFunction
            }

            "nutkWallFunction" => Self::NutkWallFunction,
            "nutUWallFunction" => Self::NutUWallFunction,

            // SPEC-LIT 15.3/29.2: the Cebeci & Bradshaw roughness downshift,
            // reading Ks (sand-grain height) and Cs (roughness constant) from
            // the patch entry - see `field_setup::NutRoughness`.
            "nutkRoughWallFunction" => Self::NutkRoughWallFunction,
            "nutURoughWallFunction" => Self::NutURoughWallFunction,

            // SPEC-LIT 120.9: the atmospheric family. The ground's wall
            // function is written in the roughness LENGTH `z0` of the
            // Richards & Hoxey profile ((120.9g)), not in a sand-grain
            // height, so it is a kind of its own; the blended `nutU` reads
            // its exponent `n` ((120.9h)). The four inlet profiles ARE a
            // fixedValue - `field_setup` computes each face's value once
            // from (120.9a)-(120.9e).
            "nutkAtmRoughWallFunction" => Self::NutkAtmRoughWallFunction,
            "nutUBlendedWallFunction" => Self::NutUBlendedWallFunction,
            "atmBoundaryLayerInletVelocity"
            | "atmBoundaryLayerInletK"
            | "atmBoundaryLayerInletEpsilon"
            | "atmBoundaryLayerInletOmega" => Self::FixedValue,
            // SPEC-LIT 120.10: the recycled inflow is a fixedValue whose
            // refValue a refresh kernel rewrites - `fldMapped` from the
            // offset plane's cells, `fldTurbulentInlet` from the seeded
            // fluctuation - so all three names are this kind; `field_setup`
            // builds the rule and refuses where no caller refreshes it.
            "mapped" | "mappedFixedValue" | "turbulentInlet" => Self::FixedValue,
            // SPEC-LIT 120.12: the synthetic LES inflow, a fixedValue whose
            // refValue the digital filter or the eddies rewrite every time
            // step; `field_setup` builds the rule and refuses it where no
            // caller draws it.
            "turbulentDigitalFilterInlet" | "turbulentDFSEMInlet" => Self::FixedValue,
            // SPEC-LIT 120.11: a fixedValue read from
            // `constant/boundaryData/<patch>`, interpolated in space at setup
            // and in time on the device; `field_setup` builds the rule and
            // refuses it on a setup path that does not advance the clock.
            "timeVaryingMappedFixedValue" => Self::FixedValue,

            // SPEC-LIT 120.2: a jump across a cyclic pair is a coupled face
            // whose far cell is seen at psi_N + j_f. The face is the cyclic
            // couple it always was, so the kind is `Cyclic`; `field_setup`
            // writes the pair's jump array and refuses both names on a patch
            // that is not one half of a cyclic pair.
            //
            // ONLY on the pressure: the jump source enters the PRESSURE
            // assembly (`Simple`'s, through `ldu_ops::add_jump_source`) and
            // no other equation's, so on any other field the jump would be
            // read by the face value and the gradient but never by the
            // matrix - a condition half applied (SPEC-LIT 13.4.1).
            "fixedJump" | "uniformJump" => {
                if !is_pressure_field(field) {
                    return unsupported(
                        &format!("{field}: boundaryField/{patch}/type"),
                        name,
                        &["cyclic"],
                        "cyclic - but note that a cyclic jump belongs on the \
                         PRESSURE field `p`, whose assembly is the only one that \
                         adds the jump source (SPEC-LIT 120.2); on any other \
                         field the jump would reach the face value and never \
                         the matrix",
                        Self::Cyclic,
                    );
                }
                Self::Cyclic
            }
            "nutLowReWallFunction" => Self::NutLowReWallFunction,
            "epsilonWallFunction" => Self::EpsilonWallFunction,
            "omegaWallFunction" => Self::OmegaWallFunction,
            "kqRWallFunction" => Self::KqRWallFunction,
            "kLowReWallFunction" => Self::KLowReWallFunction,
            "thermalWallFunction" => Self::ThermalWallFunction,

            // SPEC-LIT 30.1: the LES wall model. Nothing to alias here - this
            // is a meteor-cfd name, not an OpenFOAM one (OpenFOAM's LES wall
            // treatment is spelled through `nutUWallFunction` with a
            // `WernerWengle` sub-model this solver does not read that way);
            // reached only under `simulationType LES;`.
            "wernerWengleWallFunction" => Self::WernerWengleWallFunction,

            // SPEC-LIT 32.2: a fixed wall heat flux, on either a wall-
            // function or a resolved (lowRe) mesh - see the variant's own
            // doc for why one condition serves both.
            "fixedFluxTemperature" => Self::FixedFluxTemperature,

            // SPEC-LIT 39.3: the contact angle on `alpha`. Both spellings map
            // to one BcKind because both ARE one condition in S4's triple -
            // fixedGradient with a rewritten refGrad. What differs is how
            // theta is computed, which `contact_angle::ContactAnglePatch`
            // reads out of the same entry, and which is why the two names are
            // kept distinct here rather than aliased into one: a patch that
            // says `constantAlphaContactAngle` and then writes `thetaA` is
            // refused by name (39.6), and it can only be refused if the
            // reader knows which of the two the case wrote.
            //
            // ONLY on a phase fraction. Nothing but `crate::vof` rewrites the
            // `ref_grad` this condition is defined by, so on any other field
            // it would be zero-gradient wearing a contact angle's name - a
            // setting the case can express and the solver silently ignores,
            // which is the S13.4 defect this project keeps finding. It is an
            // error naming the field it belongs on instead.
            "constantAlphaContactAngle" | "dynamicAlphaContactAngle" => {
                if !field.starts_with("alpha") {
                    return unsupported(
                        &format!("{field}: boundaryField/{patch}/type"),
                        name,
                        &["zeroGradient", "fixedGradient", "fixedValue"],
                        "zeroGradient - but note that a contact angle belongs on the                          phase fraction `alpha.<phase>`, which is the only field                          SPEC-LIT 39 rewrites the gradient of; on any other field                          nothing would ever compute the angle",
                        Self::ZeroGradient,
                    );
                }
                Self::ContactAngle
            }

            // SPEC-LIT 47.9: the conjugate interface. Both spellings are ONE
            // condition - `thermalContactResistance` is `coupledTemperature`
            // with a non-zero `Rc`, and the resistance is read from the
            // interface entry, not from the name - but both are kept in the
            // table so a case that writes either is understood without a
            // substitution warning.
            //
            // ONLY on a temperature. Nothing but `crate::cht` rewrites the
            // triple this condition is defined by, so on any other field it
            // would be zeroGradient wearing a conjugate interface's name -
            // the S13.4 defect this project keeps finding. Refused naming the
            // field it belongs on, exactly as `constantAlphaContactAngle` on
            // a non-`alpha` field is.
            "coupledTemperature" | "thermalContactResistance" => {
                if !is_temperature_field(field) {
                    return unsupported(
                        &format!("{field}: boundaryField/{patch}/type"),
                        name,
                        &["zeroGradient", "fixedValue", "fixedFluxTemperature"],
                        "zeroGradient - but note that a conjugate interface belongs on \
                         the TEMPERATURE field `T`, which is the only field SPEC-LIT 47 \
                         rewrites the Robin triple of; on any other field nothing would \
                         ever compute the interface conductance",
                        Self::ZeroGradient,
                    );
                }
                Self::CoupledTemperature
            }

            // SPEC-LIT 47.9: OpenFOAM's own spelling of the same condition.
            // Accepted as an alias rather than rejected, and the substitution
            // is printed once - the same treatment
            // `compressible::alphatJayatillekeWallFunction` gets below.
            "compressible::turbulentTemperatureCoupledBaffleMixed" => {
                if !is_temperature_field(field) {
                    return unsupported(
                        &format!("{field}: boundaryField/{patch}/type"),
                        name,
                        &["zeroGradient", "fixedValue", "fixedFluxTemperature"],
                        "zeroGradient - a conjugate interface belongs on the TEMPERATURE \
                         field",
                        Self::ZeroGradient,
                    );
                }
                crate::io::contract::warn_once(
                    &format!("{field}: compressible::turbulentTemperatureCoupledBaffleMixed"),
                    &format!(
                        "{field}: boundaryField/{patch}/type \
                         `compressible::turbulentTemperatureCoupledBaffleMixed` (OpenFOAM's \
                         spelling of the conjugate interface) mapped to `coupledTemperature`"
                    ),
                );
                Self::CoupledTemperature
            }

            // SPEC-LIT 50.8: this name asks for S47's conjugate coupling AND
            // S50's radiative exchange ON THE SAME FACE, and those two
            // conditions rewrite the same three numbers - exactly as S47.6
            // says `thermalWallFunction` and `coupledTemperature` do.
            // Surface-to-surface view factors now EXIST (SPEC-LIT S49/S50),
            // so the refusal names both conditions rather than only the
            // conjugate one; what is still missing is a face that is both at
            // once. S47.10 used to say the view factors were "a search
            // problem and tier D"; S49.2 is that search made deterministic,
            // and this message is the part of S47.10 that had to change.
            "compressible::turbulentTemperatureRadCoupledMixed" => {
                return unsupported_note(
                    &format!("{field}: boundaryField/{patch}/type"),
                    name,
                    &[
                        "coupledTemperature",
                        "thermalContactResistance",
                        "greyDiffusiveRadiationViewFactor",
                    ],
                    "a face carries the conjugate interface OR the radiating wall, never \
                     both - they rewrite the same (fr, refValue, refGrad) (SPEC-LIT 50.8)",
                    "coupledTemperature - the conjugate coupling WITHOUT the radiative \
                     exchange term",
                    Self::CoupledTemperature,
                );
            }

            // SPEC-LIT 50.8: the surface-to-surface radiating wall. Both the
            // OpenFOAM spelling and the native one map to one BcKind.
            //
            // ONLY on a temperature. Nothing but `crate::s2s` rewrites the
            // triple this condition is defined by, so on any other field it
            // would be zeroGradient wearing a radiating wall's name - the
            // S13.4 defect this project keeps finding. Refused naming the
            // field it belongs on, exactly as `coupledTemperature` on a
            // non-temperature field is.
            "greyDiffusiveRadiationViewFactor" | "s2sWall" => {
                if !is_temperature_field(field) {
                    return unsupported(
                        &format!("{field}: boundaryField/{patch}/type"),
                        name,
                        &["zeroGradient", "fixedValue", "fixedFluxTemperature"],
                        "zeroGradient - but note that a surface-to-surface radiating wall \
                         belongs on the TEMPERATURE field `T`, which is the only field \
                         SPEC-LIT 50 rewrites the Robin triple of; on any other field \
                         nothing would ever compute the irradiation",
                        Self::ZeroGradient,
                    );
                }
                Self::S2sWall
            }

            // SPEC-LIT 52.5: the fan / blower patch. Both spellings are ONE
            // condition - the curve is read from the fan entry, not from the
            // name - but both are kept in the table so a case that writes
            // either is understood without a substitution warning.
            //
            // ONLY on the pressure. Nothing but `crate::fan` rewrites the
            // triple this condition is defined by, so on any other field it
            // would be zeroGradient wearing a fan's name - the S13.4.1
            // defect this project keeps finding. Refused naming the field it
            // belongs on, exactly as `coupledTemperature` on a
            // non-temperature field is.
            "fanPressure" | "fan" => {
                if !is_pressure_field(field) {
                    return unsupported(
                        &format!("{field}: boundaryField/{patch}/type"),
                        name,
                        &["zeroGradient", "fixedValue", "pressureInletOutletVelocity"],
                        "zeroGradient - but note that a fan curve belongs on the \
                         PRESSURE field `p`, which is the only field SPEC-LIT 52 \
                         rewrites the Robin triple of; on any other field nothing \
                         would ever compute the operating point. The VELOCITY side \
                         of a fan patch needs no new condition either, but it is \
                         `zeroGradient` and NOT `pressureInletOutletVelocity`: this \
                         solver seeds kind 12 at fr = 1 and never refreshes it from \
                         the flux, so `momFluxIsPrescribed` would pin an inflow face \
                         at zero and the fan could move no air (SPEC-LIT S52.10)",
                        Self::ZeroGradient,
                    );
                }
                Self::FanPressure
            }

            // SPEC-LIT 53.3: the porous jump against a plenum. Same rule, and
            // `porousBafflePressure` is accepted as an alias for the name
            // OpenFOAM users will reach for - the condition here is a jump on
            // the pressure-flux relation, and S53.5 says in as many words
            // that INSERTING the baffle topology is refused rather than
            // silently answered with an internal face.
            "porousJumpPressure" | "porousBafflePressure" => {
                if !is_pressure_field(field) {
                    return unsupported(
                        &format!("{field}: boundaryField/{patch}/type"),
                        name,
                        &["zeroGradient", "fixedValue"],
                        "zeroGradient - but note that a porous jump belongs on the \
                         PRESSURE field `p`, which is the only field SPEC-LIT 53 \
                         rewrites the Robin triple of; on any other field nothing \
                         would ever compute the resistance",
                        Self::ZeroGradient,
                    );
                }
                Self::PorousJumpPressure
            }

            // SPEC-LIT 120.8: the thermal family. Three of the names ride
            // `Mixed` - their triple is rewritten every outer iteration from
            // the face's CURRENT conductance by `crate::energy::Energy`'s
            // thermal-wall kernel, and `field_setup` refuses them unless the
            // caller's `BcInputs::thermal` promises that wiring.
            // `turbulentHeatFluxTemperature` rides §32.2's
            // `FixedFluxTemperature` as it is: its flux goes through the
            // EFFECTIVE conductivity, which is exactly what that kernel
            // already writes - only the `refValue = q` seed differs, which
            // `field_setup`'s 120.8 block computes.
            "externalWallHeatFluxTemperature"
            | "lumpedMassTemperature"
            | "totalFlowRateAdvectiveDiffusive" => Self::Mixed,
            "turbulentHeatFluxTemperature" => Self::FixedFluxTemperature,

            // SPEC-LIT 29.3: OpenFOAM spells the Jayatilleke thermal wall
            // function on `alphat`, a field this solver does not carry (its
            // energy equation applies the correction directly to `T`
            // instead). Accepted as an alias rather than rejected, but the
            // substitution is printed once, per SPEC-LIT 29.3's own wording -
            // "the reader accepts ... and says what it mapped it to".
            "compressible::alphatJayatillekeWallFunction" => {
                crate::io::contract::warn_once(
                    &format!("{field}: compressible::alphatJayatillekeWallFunction"),
                    &format!(
                        "{field}: boundaryField/{patch}/type \
                         `compressible::alphatJayatillekeWallFunction` (OpenFOAM's alphat \
                         wall function; this solver has no alphat field) mapped to \
                         `thermalWallFunction`"
                    ),
                );
                Self::ThermalWallFunction
            }

            // SPEC-LIT 120.8: the two alphat aliases. This solver carries no
            // `alphat` field - `k_eff,b` already holds `rho cp nu_t,b/Pr_t`
            // with `nu_t,b` from `nut`'s own wall function (S26) - so on a
            // field named `alphat` the names are read as `calculated`, with
            // the mapping printed once; on any other field they are refused,
            // naming `alphat`.
            "alphatWallFunction" | "compressible::alphatWallFunction" => {
                if field == "alphat" {
                    crate::io::contract::warn_once(
                        &format!("{field}: {name}"),
                        &format!(
                            "{field}: boundaryField/{patch}/type `{name}` (OpenFOAM's \
                             alphat wall function; this solver has no alphat field) \
                             mapped to `calculated`"
                        ),
                    );
                    Self::Calculated
                } else {
                    return unsupported(
                        &format!("{field}: boundaryField/{patch}/type"),
                        name,
                        &["calculated"],
                        "calculated - but note that this name belongs on the `alphat` \
                         field, which this solver does not carry: its energy equation \
                         applies the turbulent thermal diffusivity to `k_eff` directly \
                         (SPEC-LIT 120.8)",
                        Self::Calculated,
                    );
                }
            }

            // SPEC-LIT 120.4: its own condition now - (120.4a)'s reverse
            // switch, fixed on OUTflow, refused until setup by nothing here
            // (the `outletValue` entry is `field_setup`'s to demand).
            "outletInlet" => Self::OutletInlet,
            // SPEC-LIT 120.4: (120.4b)-(120.4d)'s Robin outlet. Accepted as a
            // kind here; the SETUP paths refuse it unless the caller sets
            // `BcInputs::advective` - the promise that `field::advective`
            // refreshes the condition every step - so no driver reaches it
            // today and the condition is reached through the API.
            "advective" => Self::Advective,

            // SPEC-LIT §120: the permanent refusals. Each of these names is
            // catalogue row `refused`: it cannot be made meaningful by any
            // unit of the completion plan, so instead of the generic menu it
            // gets a note saying WHY, and what to write instead. The arms sit
            // after every named arm and before the generic one;
            // `waveTransmissive` is deliberately NOT caught by the `wave`
            // prefix - it is an absorbing outlet, CMP-17's to implement, and
            // falls through to the generic refusal.
            name if name.starts_with("coded") || name == "#codeStream" => {
                return unsupported_note(
                    &format!("{field}: boundaryField/{patch}/type"),
                    name,
                    &["fixedValue", "fixedGradient", "mixed"],
                    "a coded condition compiles C++ written into the case and loads it at run time, and this solver loads no user code; write the condition the code computes as fixedValue, fixedGradient or mixed (SPEC-LIT §120)",
                    "calculated (a fixed value at whatever the file's `value` entry held)",
                    Self::Calculated,
                );
            }

            "cyclicACMI" => {
                return unsupported_note(
                    &format!("{field}: boundaryField/{patch}/type"),
                    name,
                    &["cyclic"],
                    "cyclicACMI couples two patches that only partly overlap and turns the uncovered part into a wall face by face; this solver couples conformal faces only, and a static, fully overlapping interface is cyclicAMI, which the polyMesh reader imprints into a conformal cyclic (SPEC-LIT 120.13)",
                    "calculated (a fixed value at whatever the file's `value` entry held)",
                    Self::Calculated,
                );
            }

            "cyclicRepeatAMI" => {
                return unsupported_note(
                    &format!("{field}: boundaryField/{patch}/type"),
                    name,
                    &["cyclic"],
                    "cyclicRepeatAMI maps a patch onto copies of another one repeated around a sector, which needs interpolation weights and a repeat transform this solver does not have; it couples conformal faces only (SPEC-LIT §120)",
                    "calculated (a fixed value at whatever the file's `value` entry held)",
                    Self::Calculated,
                );
            }

            name if name.starts_with("nonConformal") => {
                return unsupported_note(
                    &format!("{field}: boundaryField/{patch}/type"),
                    name,
                    &["cyclic"],
                    "the non-conformal coupled patch family intersects its two sides while the case runs, moving meshes included; this solver couples conformal faces only, and a static non-conformal interface is cyclicAMI, which the polyMesh reader imprints into a conformal cyclic (SPEC-LIT 120.13)",
                    "calculated (a fixed value at whatever the file's `value` entry held)",
                    Self::Calculated,
                );
            }

            // The v2-f model's wall functions: the model registry refuses the
            // model itself (SPEC-LIT §13.4.4), so no field this solver solves
            // can ask for them.
            "v2WallFunction" | "fWallFunction" => {
                return unsupported_note(
                    &format!("{field}: boundaryField/{patch}/type"),
                    name,
                    &[],
                    "this wall function belongs to the v2-f turbulence model, which this solver does not have (the model registry refuses v2f by name, SPEC-LIT §13.4.4), so no field it solves needs it (SPEC-LIT §120)",
                    "calculated (a fixed value at whatever the file's `value` entry held)",
                    Self::Calculated,
                );
            }

            name if name.starts_with("wave") && name != "waveTransmissive" => {
                return unsupported_note(
                    &format!("{field}: boundaryField/{patch}/type"),
                    name,
                    &[],
                    "the wave conditions generate or absorb free-surface waves through a wave model, and this solver has no wave model (SPEC-LIT §120)",
                    "calculated (a fixed value at whatever the file's `value` entry held)",
                    Self::Calculated,
                );
            }

            // SPEC-LIT 120.15 (CHR-29): the two species conditions are
            // lowered by the species set itself, `Species::setup_member`,
            // which rewrites them out of a member's raw field before this
            // reader sees it. Reaching here means the field is not a solved
            // member of a species set - a case without species, or a field
            // outside the set.
            "semiPermeableBaffleMassFraction" | "specieTransfer" => {
                return unsupported_note(
                    &format!("{field}: boundaryField/{patch}/type"),
                    name,
                    &[],
                    &format!(
                        "{name} is a species boundary condition (SPEC-LIT 120.15): it is lowered \
                         only on a solved member of a species set, and \"{field}\" is not one - \
                         a case without species, or a field outside the set"
                    ),
                    "zeroGradient (an impermeable wall)",
                    Self::ZeroGradient,
                );
            }

            "activeBaffleVelocity" => {
                return unsupported_note(
                    &format!("{field}: boundaryField/{patch}/type"),
                    name,
                    &[],
                    "activeBaffleVelocity opens a baffle during the run by turning its wall faces into coupled ones, a change of mesh topology this solver never makes while a case runs (SPEC-LIT §120)",
                    "calculated (a fixed value at whatever the file's `value` entry held)",
                    Self::Calculated,
                );
            }

            other => {
                return unsupported(
                    &format!("{field}: boundaryField/{patch}/type"),
                    other,
                    IMPLEMENTED_BC_NAMES,
                    "calculated (a fixed value at whatever the file's `value` entry held)",
                    Self::Calculated,
                );
            }
        };
        Ok(k)
    }

    /// True for the conditions whose value fraction is regenerated from the
    /// sign of the face flux every outer iteration.
    #[inline]
    pub fn is_flux_switched(self) -> bool {
        let v = self as Label;
        (FLUX_SWITCHED_FIRST..=FLUX_SWITCHED_LAST).contains(&v)
    }

    /// True for the two conditions that constrain the wall-adjacent CELL
    /// rather than just the face value - SPEC-LIT §15.5.
    ///
    /// Asked of `epsilon`'s or `omega`'s OWN patch type, never of another
    /// field's.
    #[inline]
    pub fn constrains_wall_cell(self) -> bool {
        matches!(self, Self::EpsilonWallFunction | Self::OmegaWallFunction)
    }

    /// True where `nu_t` gets a wall-function value - SPEC-LIT §15.2 and
    /// §15.5.
    ///
    /// **`nutLowReWallFunction` is deliberately false.** It declares that the
    /// mesh resolves the viscous sublayer, so `nu_t = 0` at the wall and the
    /// molecular viscosity alone carries the shear; applying a wall function
    /// there adds turbulent viscosity the mesh is already resolving and
    /// overpredicts the wall shear stress. Asked of `nut`'s own patch type,
    /// never of `epsilon`'s.
    #[inline]
    pub fn is_nut_wall_function(self) -> bool {
        matches!(
            self,
            Self::NutkWallFunction
                | Self::NutUWallFunction
                | Self::NutkRoughWallFunction
                | Self::NutURoughWallFunction
                | Self::NutkAtmRoughWallFunction
                | Self::NutUBlendedWallFunction
        )
    }

    /// True where `nu_t` is pinned to zero at the wall (SPEC-LIT §15.2:
    /// "`nu_t,w = 0`. That is the whole model, and the point of it.").
    #[inline]
    pub fn is_nut_low_re(self) -> bool {
        matches!(self, Self::NutLowReWallFunction)
    }

    /// True for the `nutU` family: `y+` comes from the local velocity via the
    /// §15.1 Newton solve for `u_tau`, rather than from `k`. Asked of `nut`'s
    /// own patch type.
    #[inline]
    pub fn is_nut_velocity_based(self) -> bool {
        matches!(
            self,
            Self::NutUWallFunction | Self::NutURoughWallFunction | Self::NutUBlendedWallFunction
        )
    }

    /// True where the face carries the Cebeci & Bradshaw roughness downshift
    /// of SPEC-LIT §15.3/§29.2 and therefore needs a `Ks`/`Cs` patch entry.
    #[inline]
    pub fn is_nut_rough(self) -> bool {
        matches!(self, Self::NutkRoughWallFunction | Self::NutURoughWallFunction)
    }

    /// True where the Jayatilleke thermal wall function (SPEC-LIT §29.3)
    /// rewrites temperature's Robin triple. Asked of the temperature field's
    /// OWN patch type, the same discipline as [`Self::is_nut_wall_function`]
    /// - SPEC-LIT §15.5's rule extends unchanged to a fifth field.
    #[inline]
    pub fn is_thermal_wall_function(self) -> bool {
        matches!(self, Self::ThermalWallFunction)
    }

    /// True where a fixed wall heat flux (SPEC-LIT §32.2) rewrites `ref_grad`
    /// every outer iteration. Asked of `T`'s OWN patch type, the same
    /// discipline [`Self::is_thermal_wall_function`] uses - this is not a
    /// wall-function-only condition (see [`Self::FixedFluxTemperature`]'s own
    /// doc), so it is checked independently of it.
    #[inline]
    pub fn is_fixed_flux_temperature(self) -> bool {
        matches!(self, Self::FixedFluxTemperature)
    }

    /// True where the Werner-Wengle LES wall model (SPEC-LIT §30.1) owns the
    /// face. Asked of `nut`'s OWN patch type, never of another field's -
    /// SPEC-LIT §15.5's rule, same as [`Self::is_nut_wall_function`].
    ///
    /// Deliberately NOT part of [`Self::is_nut_wall_function`]: that family
    /// feeds [`crate::wallfunctions::WallData`], the RAS `nutk`/`nutU`
    /// machinery keyed on `k` or a Newton solve for `u_tau` - neither of
    /// which Werner-Wengle uses. Mixing the two would route a WW face
    /// through the wrong kernel.
    #[inline]
    pub fn is_werner_wengle_wall_function(self) -> bool {
        matches!(self, Self::WernerWengleWallFunction)
    }

    /// True where SPEC-LIT §39's contact angle owns the face and rewrites
    /// `alpha`'s `ref_grad` every outer iteration. Asked of `alpha`'s OWN
    /// patch type, the same discipline [`Self::is_fixed_flux_temperature`]
    /// uses.
    #[inline]
    pub fn is_contact_angle(self) -> bool {
        matches!(self, Self::ContactAngle)
    }

    /// True where SPEC-LIT S47's conjugate interface owns the face and
    /// rewrites `T`'s Robin triple every outer iteration. Asked of `T`'s OWN
    /// patch type, the same discipline [`Self::is_thermal_wall_function`]
    /// uses - and note that the two are mutually exclusive by construction
    /// (S47.6): a face carries one or the other, never both, because they
    /// rewrite the same three numbers.
    #[inline]
    pub fn is_coupled_temperature(self) -> bool {
        matches!(self, Self::CoupledTemperature)
    }

    /// True where SPEC-LIT S52's fan curve owns the face and rewrites `p`'s
    /// Robin triple once per pressure corrector. Asked of `p`'s OWN patch
    /// type, the same S15.5 discipline [`Self::is_coupled_temperature`] uses.
    #[inline]
    pub fn is_fan_pressure(self) -> bool {
        matches!(self, Self::FanPressure)
    }

    /// True where SPEC-LIT S53.3's porous jump owns the face. Asked of `p`'s
    /// OWN patch type - and note that a face carries one of this and
    /// [`Self::is_fan_pressure`], never both, because they rewrite the same
    /// three numbers.
    #[inline]
    pub fn is_porous_jump_pressure(self) -> bool {
        matches!(self, Self::PorousJumpPressure)
    }

    /// True where SPEC-LIT S50's surface-to-surface radiating wall owns the
    /// face and rewrites `T`'s Robin triple every outer iteration. Asked of
    /// `T`'s OWN patch type, the same S15.5 discipline
    /// [`Self::is_coupled_temperature`] uses - and note the two are mutually
    /// exclusive by construction (S50.8), because they rewrite the same
    /// three numbers.
    #[inline]
    pub fn is_s2s_wall(self) -> bool {
        matches!(self, Self::S2sWall)
    }
}

/// Is `field` the temperature? SPEC-LIT S47.9's "on a field that is not a
/// temperature" refusal needs to know, and nothing else does - which is why
/// it is a free function next to the match rather than a method.
///
/// `T` is this solver's temperature; `T.<region>` is what a multi-region case
/// writes per region (S47.4), so the prefix is accepted with a separator after
/// it. `Tref`, `Twall` and friends are NOT temperatures in the sense that
/// matters here - they are parameters - so a bare `starts_with("T")` would be
/// wrong.
/// Is `field` the pressure? SPEC-LIT S52.5's "on a field that is not the
/// pressure" refusal needs to know, and it is the exact analogue of
/// [`is_temperature_field`] - `p` is this solver's kinematic pressure and
/// `p_rgh` / `p.<region>` are the spellings a case may write, while `pRef`
/// and `p0` are PARAMETERS and must not match.
pub(crate) fn is_pressure_field(field: &str) -> bool {
    field == "p"
        || field == "p_rgh"
        || field
            .strip_prefix('p')
            .is_some_and(|rest| rest.starts_with('.') || rest.starts_with('_'))
}

fn is_temperature_field(field: &str) -> bool {
    field == "T"
        || field
            .strip_prefix('T')
            .is_some_and(|rest| rest.starts_with('.') || rest.starts_with('_'))
}

/// A cell field plus its boundary state.
pub struct GpuScalarField {
    pub name: String,
    pub n_cells: usize,
    pub n_boundary_faces: usize,

    /// `[n_cells]` internal field
    pub f: DevBuf<Scalar>,
    /// `[n_cells]` previous time level, `psi^{n-1}`
    pub f0: DevBuf<Scalar>,
    /// `[n_cells]` the level before that, `psi^{n-2}`.
    ///
    /// BDF2 needs it (`SPEC-LIT` §3.3, §13.3) and a field carrying only one
    /// old level cannot support the scheme whatever the kernel can compute -
    /// which is precisely why `fv::fvm_ddt_bdf2` sat here unreachable. The
    /// rotation is `psi^{n-2} <- psi^{n-1} <- psi`, in that order, once per
    /// time step: [`crate::field_ops::store_old_time`].
    pub f00: DevBuf<Scalar>,
    /// `[n_bf]` evaluated boundary values
    pub bf: DevBuf<Scalar>,

    /// `[n_bf]` valueFraction
    pub fr: DevBuf<Scalar>,
    pub ref_value: DevBuf<Scalar>,
    pub ref_grad: DevBuf<Scalar>,
    /// `[n_bf]` `BcKind` as `i32`
    pub bc_kind: DevBuf<Label>,
    /// SPEC-LIT 120.6: the Outlets II rules the setup built for this field,
    /// one per patch carrying one of the five names - the kind, the patch's
    /// face range, the other patch's face range (kinds 49, 50), the target
    /// or offset, and the fraction. `update_patch_means_scalar` launches one
    /// kernel per rule once per outer iteration; a field with no rule
    /// launches nothing.
    pub mean_rules: Vec<patch_mean::PatchMeanRule>,
    /// SPEC-LIT 120.10: the `mapped`/`mappedFixedValue` patches, each with
    /// its sampled cells on the device - `update_mapped_scalar` launches one
    /// kernel per rule once per outer iteration; none, nothing.
    pub mapped: Vec<mapped::GpuMapped>,
    /// SPEC-LIT 120.10: the `turbulentInlet` patches, each with its
    /// reference and step counter on the device - advanced once per time
    /// step by `advance_turbulent_inlet_scalar`; none, nothing.
    pub turbulent: Vec<mapped::GpuTurbulentInlet>,
    /// SPEC-LIT 120.2: `[n_bf]` the cyclic jump `j_f` - the amount the far
    /// cell of a coupled face is offset by as this side sees it,
    /// `psi_N + j_f` (120.2b). Zero on every face that is not one half of a
    /// fixedJump / uniformJump / fan / porousBafflePressure pair, and on
    /// every face of a field with none, so a plain cyclic face reads exactly
    /// what it always read. A separate array rather than the coupled face's
    /// unused `ref_value`, because drivers and fixtures write the triple
    /// wholesale and a stray value there would become a jump.
    pub jump: DevBuf<Scalar>,
    /// SPEC-LIT 120.2: the field's jump pairs on the device - `Some` exactly
    /// when the field carries one, which is when a pressure assembly adds
    /// `ldu_ops::add_jump_source`; its flux laws (fan, porousBafflePressure)
    /// are re-evaluated from the face flux by `refresh` once per corrector.
    pub cyclic_jump: Option<crate::function1::DeviceCyclicJumps>,
    /// SPEC-LIT 120.14: the VOF-family patches (`variableHeightFlowRate`,
    /// `phaseHydrostaticPressure`), rewritten on the host from the current
    /// phase fraction once per time step by `Vof::refresh_vof_family`;
    /// none, nothing.
    pub vof_rules: Vec<vof_family::VofFamilyRule>,
}

impl GpuScalarField {
    /// All zeros, with every face marked from the mesh's `PatchKind` rather
    /// than left at `0` - which would be `FixedValue` with `refValue = 0`, a
    /// silent Dirichlet condition on every patch.
    pub fn zeros(gpu: &Gpu, m: &GpuMesh, name: &str) -> Result<Self> {
        let kinds: Vec<Label> = kinds_from_patches(m);
        Ok(Self {
            name: name.to_string(),
            n_cells: m.n_cells,
            n_boundary_faces: m.n_boundary_faces,
            f: gpu.zeros(m.n_cells)?,
            f0: gpu.zeros(m.n_cells)?,
            f00: gpu.zeros(m.n_cells)?,
            bf: gpu.zeros(m.n_boundary_faces)?,
            fr: gpu.zeros(m.n_boundary_faces)?,
            ref_value: gpu.zeros(m.n_boundary_faces)?,
            ref_grad: gpu.zeros(m.n_boundary_faces)?,
            bc_kind: gpu.upload(&kinds)?,
            mean_rules: Vec::new(),
            mapped: Vec::new(),
            turbulent: Vec::new(),
            jump: gpu.zeros(m.n_boundary_faces)?,
            cyclic_jump: None,
            vof_rules: Vec::new(),
        })
    }

    /// Does any boundary face of this field carry a Dirichlet-like
    /// condition (`fr > 0`)?
    ///
    /// The same test [`crate::simple::Simple`]/[`crate::vof`] already run on
    /// `p` to decide whether the pressure Poisson problem is pinned
    /// (SPEC-LIT §8.5's null space), generalised so SPEC-LIT §35.1's check -
    /// "no Dirichlet `T` anywhere" - can reuse it instead of a third private
    /// copy. `Empty`/`Cyclic` faces are excluded: neither constrains a
    /// value, whatever `fr` a lowering happened to leave on them.
    pub fn has_a_dirichlet(&self, gpu: &Gpu) -> Result<bool> {
        if self.n_boundary_faces == 0 {
            return Ok(false);
        }
        let fr = gpu.download(&self.fr)?;
        let kinds = gpu.download(&self.bc_kind)?;
        for i in 0..self.n_boundary_faces {
            let k = kinds[i];
            if k == BcKind::Empty as Label || k == BcKind::Cyclic as Label {
                continue;
            }
            if fr[i] > 0.0 {
                return Ok(true);
            }
        }
        Ok(false)
    }
}

pub struct GpuVectorField {
    pub name: String,
    pub n_cells: usize,
    pub n_boundary_faces: usize,

    pub f: DevBuf<Vec3>,
    /// `psi^{n-1}`
    pub f0: DevBuf<Vec3>,
    /// `psi^{n-2}` - see [`GpuScalarField::f00`].
    pub f00: DevBuf<Vec3>,
    pub bf: DevBuf<Vec3>,

    pub fr: DevBuf<Scalar>,
    pub ref_value: DevBuf<Vec3>,
    pub ref_grad: DevBuf<Vec3>,
    pub bc_kind: DevBuf<Label>,
    /// `[12 n_bf]` the DirectionMixed block of SPEC-LIT 120.3,
    /// `F_xx F_xy F_xz F_yy F_yz F_zz r_x r_y r_z g_x g_y g_z` per boundary
    /// face - `(120.3a)`'s `F`, `r` and `g`, and `(120.3h)`'s triple source:
    /// the matrix reads `fr = f_t`, `refValue = r`, `refGrad = g`. Zero
    /// everywhere a face does not carry one of the four tensorial names
    /// (see [`BcKind::DirectionMixed`]).
    pub dm: DevBuf<Scalar>,
    /// SPEC-LIT 120.6: the Outlets II rules the setup built for this field,
    /// as on [`GpuScalarField::mean_rules`] - `update_patch_means_vector`
    /// launches one kernel per rule once per outer iteration.
    pub mean_rules: Vec<patch_mean::PatchMeanRule>,
    /// SPEC-LIT 120.10: as [`GpuScalarField::mapped`], refreshed by
    /// `update_mapped_vector`.
    pub mapped: Vec<mapped::GpuMapped>,
    /// SPEC-LIT 120.10: as [`GpuScalarField::turbulent`], advanced by
    /// `advance_turbulent_inlet_vector`.
    pub turbulent: Vec<mapped::GpuTurbulentInlet>,
    /// SPEC-LIT 120.12: the synthetic LES inflow patches, each with its
    /// parameters and its state on the device, drawn once per TIME step by
    /// `advance_synthetic_inlet_vector`; none, nothing.
    pub synthetic: Vec<synthetic::GpuSynthetic>,
    /// SPEC-LIT 120.14: as [`GpuScalarField::vof_rules`], for
    /// `variableHeightFlowRateInletVelocity` and `outletPhaseMeanVelocity`.
    pub vof_rules: Vec<vof_family::VofFamilyRule>,
}

impl GpuVectorField {
    pub fn zeros(gpu: &Gpu, m: &GpuMesh, name: &str) -> Result<Self> {
        let kinds: Vec<Label> = kinds_from_patches(m);
        Ok(Self {
            name: name.to_string(),
            n_cells: m.n_cells,
            n_boundary_faces: m.n_boundary_faces,
            f: gpu.zeros(m.n_cells)?,
            f0: gpu.zeros(m.n_cells)?,
            f00: gpu.zeros(m.n_cells)?,
            bf: gpu.zeros(m.n_boundary_faces)?,
            fr: gpu.zeros(m.n_boundary_faces)?,
            ref_value: gpu.zeros(m.n_boundary_faces)?,
            ref_grad: gpu.zeros(m.n_boundary_faces)?,
            bc_kind: gpu.upload(&kinds)?,
            dm: gpu.zeros(12 * m.n_boundary_faces)?,
            mean_rules: Vec::new(),
            mapped: Vec::new(),
            turbulent: Vec::new(),
            synthetic: Vec::new(),
            vof_rules: Vec::new(),
        })
    }
}

/// A face field: the volumetric flux `phi`, and interpolated diffusivities.
pub struct GpuSurfaceScalarField {
    pub name: String,
    pub n_internal_faces: usize,
    pub n_boundary_faces: usize,
    pub f: DevBuf<Scalar>,
    pub bf: DevBuf<Scalar>,
}

impl GpuSurfaceScalarField {
    pub fn zeros(gpu: &Gpu, m: &GpuMesh, name: &str) -> Result<Self> {
        Ok(Self {
            name: name.to_string(),
            n_internal_faces: m.n_internal_faces,
            n_boundary_faces: m.n_boundary_faces,
            f: gpu.zeros(m.n_internal_faces)?,
            bf: gpu.zeros(m.n_boundary_faces)?,
        })
    }
}

/// Seed `bc_kind` from the mesh topology. The field file overwrites this for
/// patches it names; the mesh has the last word on `empty` and `cyclic`,
/// because a field file cannot put a fixedValue on an empty patch and have it
/// mean anything.
fn kinds_from_patches(m: &GpuMesh) -> Vec<Label> {
    use crate::mesh::PatchKind;

    let mut kinds = vec![BcKind::ZeroGradient as Label; m.n_boundary_faces];
    for p in &m.patches {
        let k = match p.kind {
            PatchKind::Empty => BcKind::Empty,
            PatchKind::Cyclic | PatchKind::Processor => BcKind::Cyclic,
            // SPEC-LIT 119.2: a wedge patch carries the rotational
            // self-couple, `BcKind::Wedge`, on every field.
            PatchKind::Wedge => BcKind::Wedge,
            PatchKind::Symmetry => BcKind::Symmetry,
            // SPEC-LIT S47.4: an `Interface` patch is topologically coupled
            // but is NOT seeded `Cyclic` - see `topology_override` in
            // `field_setup.rs` for the reasoning. It stays zero-gradient
            // (adiabatic, the safe default) until `crate::cht` claims it.
            _ => BcKind::ZeroGradient,
        } as Label;
        for i in 0..p.size {
            kinds[p.start + i] = k;
        }
    }
    kinds
}

/// The `advective` outlet of SPEC-LIT 120.4: the host twin of (120.4d), the
/// per-face wave-speed state, and the Robin rewrite - see
/// [`advective::advective_triple`] and [`advective::AdvectiveScalar`].
pub mod advective;

/// The Outlets II conditions of SPEC-LIT 120.6: the [`PatchMeanRule`] the
/// setup holds per patch, and the host twins of the `fldPatchMeanScalar`,
/// `fldPatchMeanVector` and `fldExtrapolatedSnGrad` kernels - see
/// [`patch_mean::patch_mean_values_scalar`],
/// [`patch_mean::patch_mean_values_vector`] and
/// [`patch_mean::extrapolated_sn_grad`].
pub mod patch_mean;

/// The recycled inflow of SPEC-LIT 120.10: the [`mapped::MappedRule`] and
/// [`mapped::TurbulentInletRule`] the setup holds per patch, and the host
/// twins of the `fldMapped` and `fldTurbulentInlet` kernels - see
/// [`mapped::mapped_values`], [`mapped::turbulent_seed`] and
/// [`mapped::turbulent_advance`].
pub mod mapped;

/// The VOF family of SPEC-LIT 120.14: the [`vof_family::VofFamilyRule`] the
/// setup holds per patch, and the four host rewrites `Vof` runs once per
/// time step - see [`vof_family::variable_height_alpha`],
/// [`vof_family::variable_height_inlet_velocity`],
/// [`vof_family::outlet_phase_mean_velocity`] and
/// [`vof_family::phase_hydrostatic_pressure`].
pub mod vof_family;

/// The cyclic jumps of SPEC-LIT 120.2: the [`cyclic_jump::CyclicJumpRule`]
/// a fan or porousBafflePressure pair carries, and the host twin of the
/// `f1CyclicJump` kernel - see [`cyclic_jump::CyclicJumpRule::jumps`].
pub mod cyclic_jump;

/// The inflow database of SPEC-LIT 120.11: `timeVaryingMappedFixedValue`'s
/// planar Delaunay triangulation, barycentric face weights and per-time face
/// slabs - see [`tvmapped::build_rule`] and [`tvmapped::TvMappedRule`].
pub mod tvmapped;

/// The synthetic LES inflow of SPEC-LIT 120.12: the digital filter and the
/// divergence-free eddies, their rules, their host twins of the
/// `fldDigitalFilterInlet` / `fldDfsemInlet` kernels and their device state -
/// see [`synthetic::df_values`] and [`synthetic::dfsem_values`].
pub mod synthetic;

/// The boundary-condition catalogue of SPEC-LIT §120, test-only; see
/// [`catalogue::CATALOGUE`]. Held to `from_name` by the three §120 tests of
/// [`mod@tests`].
#[cfg(test)]
mod catalogue;

/// That catalogue as the JSON the studio reads (docs/17 BC-22), test-only;
/// see [`catalogue_json::emit_bc_catalogue`].
#[cfg(test)]
mod catalogue_json;

#[cfg(test)]
mod tests {
    use super::*;

    fn gpu() -> Option<Gpu> {
        Gpu::new(0).ok()
    }

    /// SPEC-LIT §39.3: the contact angle is a condition on the PHASE
    /// FRACTION, and nothing but `crate::vof` rewrites the `ref_grad` it is
    /// defined by. On any other field it would be zero-gradient wearing a
    /// contact angle's name - a setting the case can express and the solver
    /// silently ignores - so it is an error naming the field it belongs on.
    #[test]
    fn a_contact_angle_belongs_on_a_phase_fraction_and_nowhere_else() {
        crate::io::contract::reset_warnings();
        for name in ["constantAlphaContactAngle", "dynamicAlphaContactAngle"] {
            assert_eq!(
                BcKind::from_name(name, "alpha.water", "walls").expect("alpha accepts it"),
                BcKind::ContactAngle
            );
            let e = BcKind::from_name(name, "T", "walls")
                .expect_err("a contact angle on T must be refused");
            let msg = e.to_string();
            assert!(msg.contains("alpha"), "the message must name the field: {msg}");
            assert!(msg.contains(name), "{msg}");
        }
    }

    /// And the discriminant is outside both device-consulted ranges, so
    /// `cuda/field.cu` evaluates it with the same `fldMixed` as everything
    /// else - SPEC-LIT §39.3's "no new device branch".
    #[test]
    fn the_contact_angle_kind_needs_no_device_branch() {
        let v = BcKind::ContactAngle as Label;
        assert_eq!(v, 32);
        assert_ne!(v, BcKind::Calculated as Label);
        assert_ne!(v, BcKind::Cyclic as Label);
        assert_ne!(v, BcKind::Symmetry as Label);
        assert!(!(FLUX_SWITCHED_FIRST..=FLUX_SWITCHED_LAST).contains(&v));
        assert!(BcKind::ContactAngle.is_contact_angle());
        assert!(!BcKind::ZeroGradient.is_contact_angle());
    }

    fn boxed_gpu_mesh(gpu: &Gpu) -> GpuMesh {
        let (mut m, points, faces) =
            crate::mesh::topology::tests::box_mesh([3, 3, 3], Vec3::new(0.1, 0.1, 0.1));
        m.compute_geometry(&points, &faces).expect("geometry");
        m.build_cell_face_maps();
        GpuMesh::upload(gpu, &m).expect("upload")
    }

    /// SPEC-LIT §35.1 reuses this to decide "no Dirichlet T anywhere" -
    /// every boundary is `ZeroGradient`/`Empty`/`Cyclic` after
    /// [`GpuScalarField::zeros`], so a fresh field on a wall-only mesh has
    /// no Dirichlet face at all.
    #[test]
    fn a_fresh_field_on_an_all_neumann_mesh_has_no_dirichlet() {
        let Some(gpu) = gpu() else { return };
        let m = boxed_gpu_mesh(&gpu);
        let f = GpuScalarField::zeros(&gpu, &m, "T").expect("field");
        assert!(!f.has_a_dirichlet(&gpu).expect("has_a_dirichlet"));
    }

    /// One `fixedValue` face (the same `fr > 0` convention
    /// `Simple::pressure_has_a_dirichlet` reads on `p`) is enough.
    #[test]
    fn one_fixed_value_face_is_a_dirichlet() {
        let Some(gpu) = gpu() else { return };
        let m = boxed_gpu_mesh(&gpu);
        let mut f = GpuScalarField::zeros(&gpu, &m, "T").expect("field");

        let mut kinds = gpu.download(&f.bc_kind).expect("kinds");
        let mut fr = gpu.download(&f.fr).expect("fr");
        kinds[0] = BcKind::FixedValue as Label;
        fr[0] = 1.0;
        gpu.write(&mut f.bc_kind, &kinds).expect("write kinds");
        gpu.write(&mut f.fr, &fr).expect("write fr");

        assert!(f.has_a_dirichlet(&gpu).expect("has_a_dirichlet"));
    }

    /// SPEC-LIT 120.13 (and §13.4.4's audit): `cyclicAMI` was refused by
    /// name, because pairing a NON-conformal interface by position is not
    /// what it names. The polyMesh reader now imprints the pair into a
    /// conformal one (SPEC-LIT 120.18), so the field name IS the conformal
    /// cyclic, accepted with no warning and no `-permissive`.
    #[test]
    fn cyclic_ami_is_the_conformal_cyclic_its_imprint_makes() {
        let _guard = crate::io::contract::permissive_test_guard();
        crate::io::contract::set_permissive(false);
        crate::io::contract::reset_warnings();

        assert!(matches!(
            BcKind::from_name("cyclicAMI", "U", "inlet").expect("accepted"),
            BcKind::Cyclic
        ));
        assert!(IMPLEMENTED_BC_NAMES.contains(&"cyclicAMI"));
        assert!(
            !crate::io::contract::warned("U: boundaryField/inlet/type"),
            "an implemented name is not a substitution"
        );
    }

    /// SPEC-LIT 120.4: the two outlets are conditions of their OWN, on any
    /// field - `outletInlet` the reverse flux switch of (120.4a), `advective`
    /// the Robin outlet of (120.4b)-(120.4d). Neither is in the
    /// flux-switched block, whose kinds are all Dirichlet on INflow.
    #[test]
    fn outlet_inlet_and_advective_are_their_own_kinds() {
        for field in ["psi", "T", "U", "p", "alpha.water"] {
            assert_eq!(
                BcKind::from_name("outletInlet", field, "outlet").expect("outletInlet"),
                BcKind::OutletInlet
            );
            assert_eq!(
                BcKind::from_name("advective", field, "outlet").expect("advective"),
                BcKind::Advective
            );
        }
        assert_eq!(BcKind::OutletInlet as Label, 38);
        assert_eq!(BcKind::Advective as Label, 39);
        assert!(!(FLUX_SWITCHED_FIRST..=FLUX_SWITCHED_LAST)
            .contains(&(BcKind::OutletInlet as Label)));
        assert!(!(FLUX_SWITCHED_FIRST..=FLUX_SWITCHED_LAST)
            .contains(&(BcKind::Advective as Label)));
    }

    // ==================================================================
    // SPEC-LIT §120 - the boundary-condition catalogue
    // ==================================================================

    /// The concrete names a refused entry is probed with: its `probes` when
    /// it names a prefix family, the name itself otherwise.
    fn refusal_probes(e: &catalogue::Entry) -> Vec<&'static str> {
        if e.probes.is_empty() {
            vec![e.name]
        } else {
            e.probes.to_vec()
        }
    }

    /// SPEC-LIT §120: every catalogue entry must behave exactly as its status
    /// says, and every unit the catalogue points at must exist in the
    /// completion plan.
    #[test]
    fn every_catalogue_name_has_a_status() {
        let _guard = crate::io::contract::permissive_test_guard();
        crate::io::contract::set_permissive(false);
        crate::io::contract::reset_warnings();

        let mut names: Vec<&str> = catalogue::CATALOGUE.iter().map(|e| e.name).collect();
        let n = names.len();
        assert_eq!(n, 130, "the catalogue holds every public name");
        names.sort_unstable();
        names.dedup();
        assert_eq!(names.len(), n, "catalogue names must be unique");

        let mut counts = [0usize; 5];
        for e in catalogue::CATALOGUE {
            counts[match e.status {
                catalogue::Status::Implemented => 0,
                catalogue::Status::Alias { .. } => 1,
                catalogue::Status::Planned { .. } => 2,
                catalogue::Status::Owned { .. } => 3,
                catalogue::Status::Refused { .. } => 4,
            }] += 1;
        }
        assert_eq!(counts, [102, 5, 3, 10, 10], "the per-status counts");
        println!(
            "  [S120] {} names: {} implemented, {} alias, {} planned, {} owned, {} refused",
            n, counts[0], counts[1], counts[2], counts[3], counts[4]
        );

        // The Implemented set IS IMPLEMENTED_BC_NAMES, in both directions.
        let mut implemented: Vec<&str> = catalogue::CATALOGUE
            .iter()
            .filter(|e| matches!(e.status, catalogue::Status::Implemented))
            .map(|e| e.name)
            .collect();
        implemented.sort_unstable();
        let mut menu: Vec<&str> = IMPLEMENTED_BC_NAMES.to_vec();
        menu.sort_unstable();
        let catalogue_only: Vec<&str> =
            implemented.iter().filter(|x| !menu.contains(x)).copied().collect();
        let menu_only: Vec<&str> =
            menu.iter().filter(|x| !implemented.contains(x)).copied().collect();
        assert!(
            catalogue_only.is_empty() && menu_only.is_empty(),
            "the Implemented set and IMPLEMENTED_BC_NAMES differ - \
             catalogue-only {catalogue_only:?}, menu-only {menu_only:?}"
        );

        // Strict mode: each status behaves as its row says.
        for e in catalogue::CATALOGUE {
            match e.status {
                catalogue::Status::Implemented => {
                    let k = BcKind::from_name(e.name, e.field, "p0")
                        .unwrap_or_else(|err| panic!("`{}` must be accepted on {}: {err}", e.name, e.field));
                    assert!(
                        e.name == "calculated" || k != BcKind::Calculated,
                        "`{}` must not degrade to Calculated",
                        e.name
                    );
                }
                catalogue::Status::Alias { to } => {
                    assert!(
                        IMPLEMENTED_BC_NAMES.contains(&to),
                        "the alias target `{to}` must be an implemented name"
                    );
                    assert!(
                        !IMPLEMENTED_BC_NAMES.contains(&e.name),
                        "an alias is accepted with a mapping and is not itself in the menu: `{}`",
                        e.name
                    );
                    let a = BcKind::from_name(e.name, e.field, "p0")
                        .unwrap_or_else(|err| panic!("the alias `{}` must be accepted: {err}", e.name));
                    let b = BcKind::from_name(to, e.field, "p0")
                        .unwrap_or_else(|err| panic!("the target `{to}` must be accepted: {err}"));
                    assert_eq!(a, b, "the alias must land on its target's kind");
                }
                catalogue::Status::Planned { unit } | catalogue::Status::Owned { unit } => {
                    let err = BcKind::from_name(e.name, e.field, "p0").err().unwrap_or_else(|| {
                        panic!("`{}` is `{unit}`'s to implement and must be refused today", e.name)
                    });
                    assert!(
                        err.to_string().contains(e.name),
                        "the refusal must name `{}`: {err}",
                        e.name
                    );
                }
                // Held by `the_permanent_refusals_name_their_reason`.
                catalogue::Status::Refused { .. } => {}
            }
        }

        // Unit-id shapes, and every id a row for in the completion plan.
        fn unit_is(unit: &str, prefix: &str) -> bool {
            let Some(rest) = unit.strip_prefix(prefix) else { return false };
            let b = rest.as_bytes();
            (b.len() == 2 && b.iter().all(u8::is_ascii_digit))
                || (b.len() == 3
                    && b[..2].iter().all(u8::is_ascii_digit)
                    && b[2].is_ascii_lowercase())
        }
        let plan_path =
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../docs/17-completion-plan.md");
        let plan = std::fs::read_to_string(&plan_path)
            .unwrap_or_else(|err| panic!("docs/17-completion-plan.md must be readable: {err}"));
        let mut ids: Vec<&str> = Vec::new();
        for e in catalogue::CATALOGUE {
            match e.status {
                catalogue::Status::Planned { unit } => {
                    assert!(unit_is(unit, "BC-"), "`{unit}` must be a BC-NN unit id");
                    ids.push(unit);
                }
                catalogue::Status::Owned { unit } => {
                    assert!(
                        unit_is(unit, "CMP-") || unit_is(unit, "CHR-"),
                        "`{unit}` must be a CMP-NN or CHR-NN unit id"
                    );
                    ids.push(unit);
                }
                _ => {}
            }
            ids.extend(e.extended_by.iter().copied());
        }
        ids.sort_unstable();
        ids.dedup();
        for id in ids {
            let row = format!("| **{id}** |");
            assert!(
                plan.contains(&row),
                "docs/17-completion-plan.md must hold the `{row}` row"
            );
        }

        // Structural rules: `extended_by` only on an implemented name,
        // `probes` only on a refused prefix family.
        for e in catalogue::CATALOGUE {
            if !matches!(e.status, catalogue::Status::Implemented) {
                assert!(
                    e.extended_by.is_empty(),
                    "`{}`: only an implemented name is extended later",
                    e.name
                );
            }
            let prefix_refused =
                matches!(e.status, catalogue::Status::Refused { .. }) && e.name.ends_with('*');
            if !prefix_refused {
                assert!(
                    e.probes.is_empty(),
                    "`{}`: only a refused prefix family carries probes",
                    e.name
                );
            }
        }

        // `waveTransmissive` is OUTSIDE the `wave*` refusal - it is CMP-17's
        // absorbing outlet, and today it reaches the generic refusal.
        let wt = catalogue::CATALOGUE
            .iter()
            .find(|e| e.name == "waveTransmissive")
            .expect("waveTransmissive is in the catalogue");
        assert!(
            matches!(wt.status, catalogue::Status::Owned { unit: "CMP-17" }),
            "waveTransmissive is CMP-17's to implement"
        );
        let err = BcKind::from_name("waveTransmissive", wt.field, "p0")
            .expect_err("waveTransmissive must be refused today");
        let msg = err.to_string();
        assert!(
            !msg.contains("no wave model"),
            "waveTransmissive must not be caught by the wave* refusal: {msg}"
        );
        assert!(
            msg.contains(IMPLEMENTED_BC_NAMES.join(", ").as_str()),
            "waveTransmissive must reach the generic refusal and its menu: {msg}"
        );

        crate::io::contract::set_permissive(false);
    }

    /// SPEC-LIT §120: every permanent refusal is refused BY NAME WITH ITS
    /// REASON - strict mode prints the note carrying the entry's key and
    /// never the whole menu, and `-permissive` downgrades it to a warning
    /// plus the documented fallback.
    #[test]
    fn the_permanent_refusals_name_their_reason() {
        let _guard = crate::io::contract::permissive_test_guard();
        crate::io::contract::set_permissive(false);
        crate::io::contract::reset_warnings();

        let menu = IMPLEMENTED_BC_NAMES.join(", ");
        let refused: Vec<&catalogue::Entry> = catalogue::CATALOGUE
            .iter()
            .filter(|e| matches!(e.status, catalogue::Status::Refused { .. }))
            .collect();
        assert_eq!(refused.len(), 10);

        let mut probes = 0usize;
        for e in &refused {
            let catalogue::Status::Refused { key } = e.status else {
                unreachable!("filtered above")
            };
            for probe in refusal_probes(e) {
                probes += 1;
                let err = BcKind::from_name(probe, e.field, "p0")
                    .expect_err(&format!("`{probe}` must be refused"));
                let msg = err.to_string();
                assert!(msg.contains(probe), "{msg}");
                assert!(
                    msg.contains(key),
                    "`{probe}`: the note must say why ({key}): {msg}"
                );
                assert!(msg.contains("note:"), "the refusal carries a note: {msg}");
                assert!(msg.contains("p0"), "the refusal must name the patch: {msg}");
                assert!(
                    !msg.contains(menu.as_str()),
                    "a permanent refusal names its reason, not the whole menu: {msg}"
                );
            }
        }
        assert_eq!(probes, 14);

        crate::io::contract::set_permissive(true);
        for e in &refused {
            // Every new arm documents `calculated` as its fallback. The one
            // refusal the catalogue did not add - the rad-coupled name,
            // refused by its own arm since SPEC-LIT §50.8 - keeps that arm's
            // own documented fallback: the conjugate coupling WITHOUT the
            // radiative exchange.
            let expected = if e.name == "compressible::turbulentTemperatureRadCoupledMixed" {
                BcKind::CoupledTemperature
            } else {
                BcKind::Calculated
            };
            for probe in refusal_probes(e) {
                assert_eq!(
                    BcKind::from_name(probe, e.field, "p0").expect("permissive"),
                    expected,
                    "`{probe}` under -permissive"
                );
            }
            assert!(
                crate::io::contract::warned(&format!("{}: boundaryField/p0/type", e.field)),
                "`{}` must be announced under -permissive",
                e.name
            );
        }
        crate::io::contract::set_permissive(false);
    }

    /// SPEC-LIT §120: the table in the document is the catalogue, row for
    /// row, in the catalogue's order.
    #[test]
    fn the_spec_lit_120_table_is_the_catalogue() {
        let spec_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("SPEC-LIT.md");
        let spec = std::fs::read_to_string(&spec_path)
            .unwrap_or_else(|err| panic!("SPEC-LIT.md must be readable: {err}"));

        let mut section: Vec<&str> = Vec::new();
        let mut in_section = false;
        for line in spec.lines() {
            if !in_section {
                in_section = line.starts_with("## 120.");
                continue;
            }
            if line.starts_with("## ") {
                break;
            }
            section.push(line);
        }
        assert!(in_section, "SPEC-LIT.md must hold the §120 heading");

        const HEADER: &str = "| family | name | status | unit or target | extended by | note |";
        let header = section
            .iter()
            .position(|l| *l == HEADER)
            .expect("the §120 table header line");

        let mut rows: Vec<Vec<&str>> = Vec::new();
        for line in &section[header + 1..] {
            if !line.starts_with('|') {
                break;
            }
            let cells: Vec<&str> = line.split('|').map(str::trim).collect();
            if cells.len() > 2 && cells[2].starts_with('`') {
                rows.push(cells);
            }
        }
        assert_eq!(rows.len(), 130, "the §120 table must hold every catalogue row");

        for (i, cells) in rows.iter().enumerate() {
            let e = &catalogue::CATALOGUE[i];
            let row = format!("§120 table row {} (`{}`)", i + 1, e.name);
            assert_eq!(cells[1], e.family, "{row}: family");
            assert_eq!(cells[2].trim_matches('`'), e.name, "{row}: name");
            let word = match e.status {
                catalogue::Status::Implemented => "implemented",
                catalogue::Status::Alias { .. } => "alias",
                catalogue::Status::Planned { .. } => "planned",
                catalogue::Status::Owned { .. } => "owned",
                catalogue::Status::Refused { .. } => "refused",
            };
            assert_eq!(cells[3], word, "{row}: status");
            let target = match e.status {
                catalogue::Status::Alias { to } => format!("`{to}`"),
                catalogue::Status::Planned { unit } | catalogue::Status::Owned { unit } => {
                    unit.to_string()
                }
                catalogue::Status::Implemented | catalogue::Status::Refused { .. } => {
                    "\u{2014}".to_string()
                }
            };
            assert_eq!(cells[4], target, "{row}: unit or target");
            let extended = if e.extended_by.is_empty() {
                "\u{2014}".to_string()
            } else {
                e.extended_by.join(", ")
            };
            assert_eq!(cells[5], extended, "{row}: extended by");
        }
    }
}
