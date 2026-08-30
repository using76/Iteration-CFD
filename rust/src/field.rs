// meteor-cfd - Copyright (c) 2026 주식회사 메테오시뮬레이션 (Meteo Simulation Co., Ltd.)
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
//! No GPL-licensed source was consulted.

use crate::device::{DevBuf, Gpu};
use crate::error::Result;
use crate::io::contract::{unsupported, unsupported_note};
use crate::mesh::GpuMesh;
use crate::{Label, Scalar, Vec3};

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

/// Every condition [`BcKind::from_name`] accepts, for the diagnostic a
/// rejected name gets. Kept next to the match so the two cannot drift.
pub const IMPLEMENTED_BC_NAMES: &[&str] = &[
    "fixedValue",
    "uniformFixedValue",
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
    "cyclicAMI",
    "cyclicSlip",
    "processor",
    "inletOutlet",
    "outletInlet",
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
];

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
            "fixedValue" | "noSlip" | "uniformFixedValue" => Self::FixedValue,
            "zeroGradient" => Self::ZeroGradient,
            "fixedGradient" | "uniformFixedGradient" => Self::FixedGradient,
            "mixed" | "freestream" | "freestreamVelocity" => Self::Mixed,
            // Its own condition now, and only for a field that says so: a
            // value written by a model and never solved for.
            "calculated" => Self::Calculated,
            "empty" => Self::Empty,
            "symmetry" | "symmetryPlane" | "slip" | "wedge" => Self::Symmetry,
            "cyclic" | "cyclicAMI" | "cyclicSlip" | "processor" => Self::Cyclic,
            "inletOutlet" | "outletInlet" | "freestreamPressure" => Self::InletOutlet,

            "turbulentIntensityKineticEnergyInlet" => {
                Self::TurbulentIntensityKineticEnergyInlet
            }
            "turbulentMixingLengthDissipationRateInlet" => {
                Self::TurbulentMixingLengthDissipationRateInlet
            }
            "turbulentMixingLengthFrequencyInlet" => Self::TurbulentMixingLengthFrequencyInlet,
            "pressureInletOutletVelocity" => Self::PressureInletOutletVelocity,
            "fixedFluxPressure" => Self::FixedFluxPressure,
            "totalPressure" => Self::TotalPressure,
            "movingWallVelocity" => Self::MovingWallVelocity,
            "flowRateInletVelocity" => Self::FlowRateInletVelocity,

            "nutkWallFunction" => Self::NutkWallFunction,
            "nutUWallFunction" => Self::NutUWallFunction,

            // SPEC-LIT 15.3/29.2: the Cebeci & Bradshaw roughness downshift,
            // reading Ks (sand-grain height) and Cs (roughness constant) from
            // the patch entry - see `field_setup::NutRoughness`.
            "nutkRoughWallFunction" => Self::NutkRoughWallFunction,
            "nutURoughWallFunction" => Self::NutURoughWallFunction,

            // The atmospheric (Monin-Obukhov) rough wall function is a
            // different profile, not a discarded parameter - it is an error
            // naming the two non-atmospheric rough functions this solver
            // does implement, not a silent substitution.
            "nutkAtmRoughWallFunction" => {
                return unsupported(
                    &format!("{field}: boundaryField/{patch}/type"),
                    name,
                    &["nutkRoughWallFunction", "nutURoughWallFunction"],
                    "the non-atmospheric rough wall function of the same family (the \
                     Monin-Obukhov atmospheric profile is not implemented)",
                    Self::NutkRoughWallFunction,
                );
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

            // SPEC-LIT 47.9: an external heat-transfer-coefficient/ambient
            // wall. Not implemented; the two conditions that ARE are named.
            "externalWallHeatFluxTemperature" => {
                return unsupported(
                    &format!("{field}: boundaryField/{patch}/type"),
                    name,
                    &["fixedFluxTemperature", "coupledTemperature"],
                    "fixedFluxTemperature (a prescribed wall heat flux) or \
                     coupledTemperature (a real conjugate solid region)",
                    Self::FixedFluxTemperature,
                );
            }

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
        matches!(self, Self::NutUWallFunction | Self::NutURoughWallFunction)
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
fn is_pressure_field(field: &str) -> bool {
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
}
