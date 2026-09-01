// meteor-cfd - Copyright (c) 2026 주식회사 메테오시뮬레이션 (Meteo Simulation Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.

//! Droplet heating and evaporation - the PARCEL side of SPEC-LIT S76.
//!
//! This module is the host mirror of the evaporation arithmetic in
//! `cuda/parcels.cu`: the liquid property set, the two saturation curves, the
//! three blowing corrections, the boiling temperature at the ambient
//! pressure, and the two closed forms every gate in S76.12 is measured
//! against - the steady ("wet-bulb") droplet temperature and the `d^2`-law
//! slope. Nothing here runs in the time loop. It is what a case is built
//! from, what a report prints, and what the gates compare the device against.
//!
//! **The gas is read, never written.** A parcel takes `T_g` and `Y_v` from
//! its own cell and returns nothing: the vapour and the latent heat are
//! accumulated per parcel and left there. That is S76's scope boundary, and
//! it buys one property outright - every parcel in a cell sees the same
//! frozen gas state, so the result cannot depend on the order the parcels in
//! that cell were visited. It also costs one: a closed box does not get
//! wetter as its droplets evaporate, so a run can be driven past saturation
//! and this module will not notice. S76.14 says so by name.
//!
//! Written from:
//!   W. E. Ranz, W. R. Marshall, *Evaporation from drops*, Chem. Eng. Prog.
//!     48 (1952) 141-146 (Part I) and 173-180 (Part II) - `Nu = 2 + 0.6
//!     Re^(1/2) Pr^(1/3)`, `Sh = 2 + 0.6 Re^(1/2) Sc^(1/3)`, and the 56
//!     suspended-droplet experiments the `d^2` metric of S76.12 comes from
//!   D. B. Spalding, *The combustion of liquid fuels*, 4th Symposium
//!     (International) on Combustion (1953) 847-864, and *Convective Mass
//!     Transfer: An Introduction*, Edward Arnold (1963) - the mass transfer
//!     number `B_M` and the Stefan-flow rate `mdot = pi d rho D Sh ln(1+B_M)`
//!   G. A. E. Godsave, *Studies of the combustion of drops in a fuel spray*,
//!     4th Symposium (International) on Combustion (1953) 818-830 - the
//!     heat-limited rate the boiling branch of (76.9) uses
//!   B. Abramzon, W. A. Sirignano, *Droplet vaporization model for spray
//!     combustion calculations*, Int. J. Heat Mass Transfer 32 (1989) 1605,
//!     DOI `10.1016/0017-9310(89)90043-4` - the heat transfer number
//!     `B_T = (1 + B_M)^phi - 1` with `phi = (c_pv/c_pg)(Sh/Nu)/Le`, which is
//!     the DEFAULT here and the reason it is closed form rather than the
//!     fixed-point iteration the design note warned about
//!   S. S. Sazhin, *Advanced models of fuel droplet heating and evaporation*,
//!     Prog. Energy Combust. Sci. 32 (2006) 162, DOI
//!     `10.1016/j.pecs.2005.11.001` - the modern re-derivation, and the
//!     survey that says which of the successors are worth the flops
//!   K. M. Watson, *Thermodynamics of the liquid state*, Ind. Eng. Chem. 35
//!     (1943) 398-406 - the `((T_c - T)/(T_c - T_b))^0.38` latent-heat
//!     correlation, which is the temperature dependence that matters most
//!   T. R. Marrero, E. A. Mason, *Gaseous diffusion coefficients*, J. Phys.
//!     Chem. Ref. Data 1 (1972) 3-118, DOI `10.1063/1.3253094` - the
//!     recommended `D(H2O-air) = 1.87e-10 T^2.072 / p[atm]` m2/s, valid
//!     282-450 K, which sets both the default diffusivity and its exponent
//!   R. W. Hyland, A. Wexler, *ASHRAE Transactions* 89(2A) (1983) 500-519 -
//!     the saturation-pressure polynomial, already in this crate as S54's
//!     [`crate::psychro::p_ws`] and reused verbatim here
//!   W. K. Lewis, *The evaporation of a liquid into a gas*, Trans. ASME 44
//!     (1922) 325-340 - the relation between the heat and mass transfer
//!     coefficients whose failure at `Le != 1` is exactly the gap S76.13's
//!     wet-bulb gate measures
//!   NIST Chemistry WebBook, SRD 69 (US-Government public domain) - the
//!     water-vapour specific heat and the critical constants
//!   K. McGrattan, S. Hostikka, R. McDermott, J. Floyd, M. Vanella et al.,
//!     *Fire Dynamics Simulator Technical Reference Guide*, NIST SP 1018-1
//!     (NIST, US-Government public domain; `reference/fds/LICENSE.md` read
//!     verbatim) - chapter "Lagrangian Particles" and appendix "Development
//!     of an Implicit Solution for Droplet Evaporation". Its `B_T = B_M`
//!     simplification is [`MassTransfer::Spalding`] here, offered and NOT
//!     the default, with (76.6) saying why
//!   ofgpu `SPEC-LIT.md` S76 - the section this module implements; S66 (the
//!     pool and the sub-step), S68 (the accumulators this adds two more to),
//!     S54 (the psychrometrics the wet-bulb gate is held against), S13.4
//! No GPL-licensed source was consulted, and in particular OpenFOAM's
//! `src/lagrangian` tree - which contains the obvious reference
//! implementation of a droplet evaporation model - was not opened.

use crate::error::{Error, Result};
use crate::io::contract;
use crate::Scalar;

#[cfg(test)]
mod tests;

/// The molar gas constant, J/(mol K). CODATA 2018, exact by the 2019 SI
/// redefinition.
pub const R_UNIVERSAL: Scalar = 8.314_462_618_153_24;

/// Molar mass of dry air, kg/mol - Gatley, Herrmann & Kretzschmar (2008),
/// the same number [`crate::psychro`] carries, so the two modules cannot
/// disagree about what air is.
pub const W_DRY_AIR: Scalar = 28.966e-3;

/// Watson's exponent (Watson 1943).
pub const WATSON_EXPONENT: Scalar = 0.38;

// ==========================================================================
//  SPEC-LIT (76.3): the saturation curve
// ==========================================================================

/// How `p_sat(T_p)` is evaluated - SPEC-LIT (76.3).
///
/// The two are not two accuracies of one thing. [`Self::ClausiusClapeyron`]
/// is **general**: it needs only `(h_v, W_v, T_b, p_b)`, so it works for any
/// liquid whose boiling point and latent heat are known, which is what a fuel
/// spray needs. [`Self::HylandWexler`] is **water only** and is the ASHRAE
/// polynomial S54 already carries; it is what a psychrometric comparison has
/// to be made against, because a data-centre or fire user checks the number
/// against a chart drawn from that same polynomial.
///
/// The gap between them is not small and it is not hidden: S76.12 measures
/// it, and the wet-bulb gate runs on both legs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SaturationCurve {
    /// `p_sat(T) = p_b exp[-(h_v(T) W_v/R)(1/T - 1/T_b)]`, anchored so that
    /// `p_sat(T_b) = p_b` exactly. `h_v(T)` is Watson's, so this is NOT the
    /// constant-latent-heat integral: the temperature dependence is inside
    /// the exponent as well as in front of it.
    ClausiusClapeyron,
    /// Hyland & Wexler (1983) as published by ASHRAE, via
    /// [`crate::psychro::p_ws`]. Water only, and
    /// [`EvaporationControls::validate`] refuses it for any liquid whose
    /// molar mass is not water's.
    #[default]
    HylandWexler,
}

impl SaturationCurve {
    pub const NAMES: &'static [&'static str] = &["clausiusClapeyron", "hylandWexler"];

    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Self::ClausiusClapeyron => "clausiusClapeyron",
            Self::HylandWexler => "hylandWexler",
        }
    }

    /// The `OFP_SAT_*` code `cuda/parcels.cu` switches on.
    #[must_use]
    pub fn code(self) -> i32 {
        match self {
            Self::ClausiusClapeyron => 0,
            Self::HylandWexler => 1,
        }
    }

    pub fn from_name(s: &str) -> Result<Self> {
        match s {
            "clausiusClapeyron" | "ClausiusClapeyron" | "clapeyron" => {
                Ok(Self::ClausiusClapeyron)
            }
            "hylandWexler" | "HylandWexler" | "ashrae" => Ok(Self::HylandWexler),
            "antoine" | "wagner" | "buck" | "iapws" => contract::unsupported_note(
                "parcels/saturationCurve",
                s,
                Self::NAMES,
                "each of these needs a per-liquid coefficient set this crate does not \
                 carry, and none of them is more accurate than hylandWexler over the \
                 range a water spray lives in (SPEC-LIT S76.3)",
                "hylandWexler",
                Self::HylandWexler,
            ),
            other => contract::unsupported(
                "parcels/saturationCurve",
                other,
                Self::NAMES,
                "hylandWexler",
                Self::HylandWexler,
            ),
        }
    }
}

// ==========================================================================
//  SPEC-LIT (76.6): the blowing correction
// ==========================================================================

/// Which correction the Stefan flow through the film gets - SPEC-LIT (76.6).
///
/// All three share one mass rate and one conductance; they differ only in
/// which transfer number the *heat* correction is taken at. The design note's
/// survey is answered here by implementing the whole ladder rather than one
/// rung of it, because the rungs differ by more than round-off and a case
/// should be able to say which one it means.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum MassTransfer {
    /// Ranz & Marshall (1952) as published, with no blowing correction at
    /// all: `mdot = -pi d rho_f D_f Sh_0 (Y_s - Y_g)`, `Nu = Nu_0`. The
    /// low-mass-flux limit, correct as `B_M -> 0`, and the historical form
    /// the 56 experiments were correlated with. Kept because it is the only
    /// leg whose `d^2` slope is linear in `(Y_s - Y_g)` rather than in
    /// `ln(1 + B_M)`, which makes it an independent check on the property
    /// chain rather than a second evaluation of the same expression.
    RanzMarshall,
    /// Spalding's Stefan-flow rate with `B_T = B_M`:
    /// `mdot = -pi d rho_f D_f Sh_0 ln(1 + B_M)`, `Nu = Nu_0 ln(1+B_M)/B_M`.
    /// This is FDS's form. It is exact at `Le = 1` and it is cheaper by one
    /// `pow`; for water in air `Le ~ 0.86`, so it is not exact here.
    Spalding,
    /// Abramzon & Sirignano (1989): the mass rate is Spalding's and the heat
    /// correction is taken at `B_T = (1 + B_M)^phi - 1` with
    /// `phi = (c_pv/c_pg)(Sh_0/Nu_0)/Le_f`. Closed form - no fixed point, no
    /// data-dependent trip count, nothing a captured graph cannot hold.
    /// **The default**, because the `phi != 1` it carries is exactly the
    /// Lewis-number correction the wet-bulb gate of S76.13 is about.
    #[default]
    AbramzonSirignano,
}

impl MassTransfer {
    pub const NAMES: &'static [&'static str] =
        &["ranzMarshall", "spalding", "abramzonSirignano"];

    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Self::RanzMarshall => "ranzMarshall",
            Self::Spalding => "spalding",
            Self::AbramzonSirignano => "abramzonSirignano",
        }
    }

    /// The `OFP_MT_*` code `cuda/parcels.cu` switches on.
    #[must_use]
    pub fn code(self) -> i32 {
        match self {
            Self::RanzMarshall => 0,
            Self::Spalding => 1,
            Self::AbramzonSirignano => 2,
        }
    }

    pub fn from_name(s: &str) -> Result<Self> {
        match s {
            "ranzMarshall" | "RanzMarshall" | "none" => Ok(Self::RanzMarshall),
            "spalding" | "Spalding" | "fds" => Ok(Self::Spalding),
            "abramzonSirignano" | "AbramzonSirignano" | "sirignano" => {
                Ok(Self::AbramzonSirignano)
            }
            "abramzonSirignanoFilm" | "filmThickness" => contract::unsupported_note(
                "parcels/massTransfer",
                s,
                Self::NAMES,
                "Abramzon & Sirignano's film-thickness corrections Nu* and Sh* need a \
                 bounded fixed point on F(B) that this section does not have; the B_T \
                 relation IS implemented and is what abramzonSirignano selects \
                 (SPEC-LIT S76.6)",
                "abramzonSirignano",
                Self::AbramzonSirignano,
            ),
            other => contract::unsupported(
                "parcels/massTransfer",
                other,
                Self::NAMES,
                "abramzonSirignano",
                Self::AbramzonSirignano,
            ),
        }
    }
}

// ==========================================================================
//  SPEC-LIT (76.2): the liquid
// ==========================================================================

/// Everything about the evaporating liquid and its vapour that S76 reads.
///
/// Nine numbers, and every one of them is read by the kernel:
/// `tests::every_liquid_property_moves_the_answer` perturbs each in turn and
/// fails if the evaporation rate does not move, which is the check that
/// catches a property plumbed in and then ignored.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LiquidProperties {
    /// Molar mass of the VAPOUR, kg/mol.
    pub w_vapour: Scalar,
    /// Boiling temperature at [`Self::p_boil`], K.
    pub t_boil: Scalar,
    /// The pressure [`Self::t_boil`] is the boiling point at, Pa. Also the
    /// pressure [`Self::d_vapour`] was measured at.
    pub p_boil: Scalar,
    /// Latent heat of vaporisation at [`Self::t_boil`], J/kg.
    pub h_v_boil: Scalar,
    /// Critical temperature, K - Watson's correlation needs it, and it is
    /// also the only thing that stops `h_v` going negative.
    pub t_crit: Scalar,
    /// Binary diffusivity of the vapour in the carrier gas at
    /// ([`Self::t_ref_d`], [`Self::p_boil`]), m2/s.
    pub d_vapour: Scalar,
    /// The temperature [`Self::d_vapour`] was measured at, K.
    pub t_ref_d: Scalar,
    /// The exponent in `D ~ T^n` at fixed pressure.
    pub d_exponent: Scalar,
    /// Specific heat of the VAPOUR at constant pressure, J/(kg K). Read only
    /// by [`MassTransfer::AbramzonSirignano`] and by the boiling branch.
    pub cp_vapour: Scalar,
}

impl LiquidProperties {
    /// Water, from the sources named on the module.
    ///
    /// `d_vapour` and `d_exponent` are Marrero & Mason's recommended
    /// correlation `D = 1.87e-10 T^2.072/p[atm]` evaluated at 298.15 K, which
    /// is `2.50536e-5 m2/s` - and NOT the `2.42e-5` that also circulates. The
    /// two differ by 3.6 %, that difference lands directly on the Lewis
    /// number, and the Lewis number is what S76.13's wet-bulb gap is made of.
    /// Quoting the source rather than the round number is the whole point.
    #[must_use]
    pub fn water() -> Self {
        Self {
            w_vapour: 18.015_268e-3,
            t_boil: 373.15,
            p_boil: 101_325.0,
            h_v_boil: 2.257e6,
            t_crit: 647.096,
            d_vapour: 2.505_362_314_4e-5,
            t_ref_d: 298.15,
            d_exponent: 2.072,
            cp_vapour: 1880.0,
        }
    }

    /// Benzene, the second fluid of Ranz & Marshall's Table 4 - the one that
    /// catches a hard-coded water property. Not used by any default; it
    /// exists so a test can prove the property set is a set.
    #[must_use]
    pub fn benzene() -> Self {
        Self {
            w_vapour: 78.11e-3,
            t_boil: 353.25,
            p_boil: 101_325.0,
            h_v_boil: 393.8e3,
            t_crit: 562.02,
            d_vapour: 0.88e-5,
            t_ref_d: 298.15,
            d_exponent: 1.75,
            cp_vapour: 1050.0,
        }
    }

    fn validate(&self) -> Result<()> {
        let bad = |what: &str, v: Scalar| -> Result<()> {
            Err(Error::Config(format!(
                "parcels/liquid: {what} is {v}; SPEC-LIT S76.2 requires it finite and \
                 positive"
            )))
        };
        for (what, v) in [
            ("wVapour", self.w_vapour),
            ("tBoil", self.t_boil),
            ("pBoil", self.p_boil),
            ("hvBoil", self.h_v_boil),
            ("tCrit", self.t_crit),
            ("dVapour", self.d_vapour),
            ("tRefD", self.t_ref_d),
            ("cpVapour", self.cp_vapour),
        ] {
            if !(v > 0.0) || !v.is_finite() {
                return bad(what, v);
            }
        }
        if !self.d_exponent.is_finite() {
            return bad("dExponent", self.d_exponent);
        }
        if self.t_boil >= self.t_crit {
            return Err(Error::Config(format!(
                "parcels/liquid: tBoil {} is not below tCrit {}. Watson's correlation \
                 (76.4) would return a zero or negative latent heat, and a droplet whose \
                 h_v is zero is divided by it (SPEC-LIT S76.2)",
                self.t_boil, self.t_crit
            )));
        }
        Ok(())
    }
}

// ==========================================================================
//  The settings block
// ==========================================================================

/// Everything a case can say about evaporation - SPEC-LIT (76.2).
///
/// A separate struct from [`crate::parcels::ParcelControls`] and not fields
/// on it, for one reason worth stating: none of it is read unless the physics
/// is [`crate::parcels::ParcelPhysics::Evaporating`], and a reader of a case
/// should be able to see that at a glance rather than infer it from a
/// doc comment on each of a dozen fields.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EvaporationControls {
    pub saturation: SaturationCurve,
    pub transfer: MassTransfer,
    pub liquid: LiquidProperties,
    /// Molar mass of the carrier gas, kg/mol. Enters the mole-to-mass
    /// conversion of (76.5) and nothing else.
    pub w_carrier: Scalar,
    /// Thermodynamic pressure, Pa. NOT the solver's `p` or `p_rgh`: the
    /// low-Mach split means the pressure that sets a boiling point is the
    /// background one, and taking it from the momentum equation's pressure
    /// would make a droplet's boiling point depend on the dynamic head
    /// (SPEC-LIT S76.2).
    pub p_ambient: Scalar,
    /// The largest fraction of `d^2` one sub-step may remove (76.10). A
    /// bound, not a convergence test: it sets the sub-step COUNT, which is
    /// capped by `ParcelControls::max_substeps` exactly as the motion CFL is.
    pub cfl: Scalar,
}

impl Default for EvaporationControls {
    fn default() -> Self {
        Self {
            saturation: SaturationCurve::default(),
            transfer: MassTransfer::default(),
            liquid: LiquidProperties::water(),
            w_carrier: W_DRY_AIR,
            p_ambient: 101_325.0,
            cfl: 0.1,
        }
    }
}

impl EvaporationControls {
    /// Everything that can be wrong with these numbers, named.
    pub fn validate(&self) -> Result<()> {
        self.liquid.validate()?;
        if !(self.w_carrier > 0.0) || !self.w_carrier.is_finite() {
            return Err(Error::Config(format!(
                "parcels/evaporation: wCarrier is {}; it must be finite and positive",
                self.w_carrier
            )));
        }
        if !(self.p_ambient > 0.0) || !self.p_ambient.is_finite() {
            return Err(Error::Config(format!(
                "parcels/evaporation: pAmbient is {}; it must be finite and positive",
                self.p_ambient
            )));
        }
        if !(self.cfl > 0.0) || !(self.cfl <= 1.0) {
            return Err(Error::Config(format!(
                "parcels/evaporation: cfl is {}; (76.10) needs it in (0, 1]. Above 1 a \
                 single sub-step could take more than the whole droplet",
                self.cfl
            )));
        }
        // S13.4: hylandWexler IS water. Offering it for a liquid that is not
        // water would return a plausible number from the wrong curve, which
        // is exactly the failure this project keeps finding.
        if self.saturation == SaturationCurve::HylandWexler
            && (self.liquid.w_vapour - LiquidProperties::water().w_vapour).abs()
                > 1e-2 * LiquidProperties::water().w_vapour
        {
            return contract::unsupported_note(
                "parcels/saturationCurve",
                "hylandWexler",
                &["clausiusClapeyron"],
                "the Hyland-Wexler polynomial is a fit to WATER, and this case's vapour \
                 has a molar mass that is not water's. Evaluating it for another liquid \
                 returns a confident number from the wrong curve; clausiusClapeyron is \
                 the general one and takes its shape from this liquid's own h_v, W_v and \
                 T_b (SPEC-LIT S76.3)",
                "clausiusClapeyron",
                (),
            );
        }
        // The boiling point at the ambient pressure has to exist, and the
        // root find is here - on the host, at setup, where a failure can be
        // named - and not in a kernel, which has no way to say so (S76.7).
        boiling_temperature(self.saturation, &self.liquid, self.p_ambient)?;
        Ok(())
    }

    /// One line for the startup banner - SPEC-LIT S13.4.2.
    #[must_use]
    pub fn describe(&self) -> String {
        let l = &self.liquid;
        format!(
            "evaporation: saturation={} transfer={} pAmbient={} wVapour={} wCarrier={} \
             tBoil={} hvBoil={} tCrit={} dVapour={}@{}K^{} cpVapour={} cfl={} \
             (SPEC-LIT S76)",
            self.saturation.name(),
            self.transfer.name(),
            self.p_ambient,
            l.w_vapour,
            self.w_carrier,
            l.t_boil,
            l.h_v_boil,
            l.t_crit,
            l.d_vapour,
            l.t_ref_d,
            l.d_exponent,
            l.cp_vapour,
            self.cfl,
        )
    }

    /// The boiling temperature at [`Self::p_ambient`], K - what the kernel is
    /// handed and what (76.9)'s branch tests against.
    pub fn boiling_temperature(&self) -> Result<Scalar> {
        boiling_temperature(self.saturation, &self.liquid, self.p_ambient)
    }
}

// ==========================================================================
//  SPEC-LIT (76.4): the properties, and their temperature dependence
// ==========================================================================

/// Watson's (1943) latent heat, J/kg - (76.4).
///
/// `h_v(T) = h_v(T_b) [ (T_c - T)/(T_c - T_b) ]^0.38`, and zero at and above
/// the critical temperature rather than the negative number the power would
/// otherwise return.
#[must_use]
pub fn latent_heat(l: &LiquidProperties, t: Scalar) -> Scalar {
    if t >= l.t_crit {
        return 0.0;
    }
    l.h_v_boil * ((l.t_crit - t) / (l.t_crit - l.t_boil)).powf(WATSON_EXPONENT)
}

/// `dh_v/dT`, J/(kg K) - the exact derivative of [`latent_heat`], which the
/// device linearisation of (76.8) needs and which is one line rather than a
/// finite difference for the usual reason.
#[must_use]
pub fn latent_heat_derivative(l: &LiquidProperties, t: Scalar) -> Scalar {
    if t >= l.t_crit {
        return 0.0;
    }
    -WATSON_EXPONENT * latent_heat(l, t) / (l.t_crit - t)
}

/// Binary diffusivity at the film temperature and the ambient pressure,
/// m2/s - (76.4). `D = D_ref (T_f/T_ref)^n (p_ref/p)`.
#[must_use]
pub fn diffusivity(l: &LiquidProperties, p_ambient: Scalar, t_film: Scalar) -> Scalar {
    l.d_vapour * (t_film / l.t_ref_d).powf(l.d_exponent) * (l.p_boil / p_ambient)
}

/// Saturation vapour pressure, Pa - (76.3).
#[must_use]
pub fn p_sat(curve: SaturationCurve, l: &LiquidProperties, t: Scalar) -> Scalar {
    match curve {
        SaturationCurve::HylandWexler => crate::psychro::p_ws(t),
        SaturationCurve::ClausiusClapeyron => {
            let hv = latent_heat(l, t);
            l.p_boil * (-(hv * l.w_vapour / R_UNIVERSAL) * (1.0 / t - 1.0 / l.t_boil)).exp()
        }
    }
}

/// `dp_sat/dT`, Pa/K - the exact derivative of [`p_sat`], carrying the
/// `dh_v/dT` term that the constant-latent-heat form drops.
#[must_use]
pub fn p_sat_derivative(curve: SaturationCurve, l: &LiquidProperties, t: Scalar) -> Scalar {
    match curve {
        SaturationCurve::HylandWexler => crate::psychro::p_ws(t) * hyland_wexler_dlog(t),
        SaturationCurve::ClausiusClapeyron => {
            let hv = latent_heat(l, t);
            let dhv = latent_heat_derivative(l, t);
            let dlog =
                (l.w_vapour / R_UNIVERSAL) * (hv / (t * t) - dhv * (1.0 / t - 1.0 / l.t_boil));
            p_sat(curve, l, t) * dlog
        }
    }
}

/// `d(ln p_ws)/dT` for the Hyland-Wexler polynomial - the analytic derivative
/// of the exponent S54's [`crate::psychro::p_ws`] evaluates.
///
/// The two coefficient sets are repeated here rather than exported from S54,
/// because S54's are private to it and because the DEVICE needs the value and
/// the slope in one translation unit. The duplication cannot drift silently:
/// `tests::the_hyland_wexler_slope_is_the_derivative_of_psychro` checks this
/// against a central difference of `psychro::p_ws` itself.
fn hyland_wexler_dlog(t: Scalar) -> Scalar {
    const C_ICE: [Scalar; 7] = [
        -5.674_535_9e3,
        6.392_524_7,
        -9.677_843e-3,
        6.221_570_1e-7,
        2.074_782_5e-9,
        -9.484_024e-13,
        4.163_501_9,
    ];
    const C_LIQ: [Scalar; 6] = [
        -5.800_220_6e3,
        1.391_499_3,
        -4.864_023_9e-2,
        4.176_476_8e-5,
        -1.445_209_3e-8,
        6.545_967_3,
    ];
    if t < 273.15 {
        let c = &C_ICE;
        -c[0] / (t * t)
            + c[2]
            + 2.0 * c[3] * t
            + 3.0 * c[4] * t * t
            + 4.0 * c[5] * t * t * t
            + c[6] / t
    } else {
        let c = &C_LIQ;
        -c[0] / (t * t) + c[2] + 2.0 * c[3] * t + 3.0 * c[4] * t * t + c[5] / t
    }
}

/// Equilibrium vapour MASS fraction at the droplet surface, from the mole
/// fraction - (76.5).
///
/// `Y = X W_v / (X W_v + (1 - X) W_c)`. Written as this ratio rather than as
/// `X/(X(1 - W_c/W_v) + W_c/W_v)` because the second form loses a digit when
/// `W_c/W_v` is far from one, and benzene in air makes it 0.37.
#[must_use]
pub fn y_surface(w_vapour: Scalar, w_carrier: Scalar, x: Scalar) -> Scalar {
    let den = w_carrier + x * (w_vapour - w_carrier);
    x * w_vapour / den
}

/// The boiling temperature at `p_ambient`, K - the root of
/// `p_sat(T) = p_ambient`, by bisection.
///
/// **On the host, at setup, and nowhere else** - the same rule S54.5 states
/// for the wet-bulb solve. A bisection has a data-dependent trip count, which
/// is warp divergence and a launch geometry a captured graph cannot express;
/// and a root find that fails needs to say so, which a kernel cannot do.
/// [`crate::parcels::Parcels::new`] calls this once and hands the answer to
/// the kernel as a scalar.
pub fn boiling_temperature(
    curve: SaturationCurve,
    l: &LiquidProperties,
    p_ambient: Scalar,
) -> Result<Scalar> {
    // The bracket is SCANNED for and not assumed, and that is a finding
    // rather than caution. `clausiusClapeyron` with Watson's `h_v` is **not**
    // monotone up to the critical point: the exponent carries `h_v(T)`, which
    // Watson sends to zero at `T_c`, so `p_sat(T_c) = p_b` again and the
    // curve peaks in between - for water at 549 K and 1.74 MPa. A
    // bisection told to bracket on `[T_lo, T_c]` therefore fails for every
    // ambient pressure above one atmosphere, which is how this was found: the
    // test asking for the boiling point at 2 bar refused. The scan takes the
    // FIRST sign change, which is the physical boiling point on the rising
    // limb; the second root, on the falling limb, is an artefact of the
    // correlation and must never be returned.
    let f = |t: Scalar| p_sat(curve, l, t) - p_ambient;
    let t_lo: Scalar = (0.25 * l.t_crit).max(180.0);
    let t_hi: Scalar = l.t_crit * (1.0 - 1e-9);
    const SCAN: usize = 4096;
    let mut lo = t_lo;
    let mut hi = t_hi;
    let mut bracketed = false;
    if f(t_lo) < 0.0 {
        let mut prev = t_lo;
        for i in 1..=SCAN {
            let t = t_lo + (t_hi - t_lo) * (i as Scalar) / (SCAN as Scalar);
            if f(t) >= 0.0 {
                lo = prev;
                hi = t;
                bracketed = true;
                break;
            }
            prev = t;
        }
    }
    if !bracketed {
        return Err(Error::Config(format!(
            "parcels/evaporation: no boiling temperature exists for pAmbient = \
             {p_ambient} Pa on the {} curve between {t_lo} K and {t_hi} K. (76.9) needs \
             it: it is the temperature the droplet is capped at, and without it the \
             surface mole fraction can exceed one and the Spalding number is not \
             defined (SPEC-LIT S76.9)",
            curve.name()
        )));
    }
    // 200 bisections is far past what f64 can hold apart. A fixed count,
    // because the answer is wanted to the last bit and a tolerance test
    // would be a second thing to justify.
    for _ in 0..200 {
        let mid = 0.5 * (lo + hi);
        if mid <= lo || mid >= hi {
            break;
        }
        if f(mid) < 0.0 {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    Ok(0.5 * (lo + hi))
}

// ==========================================================================
//  SPEC-LIT (76.7): the single-droplet rate, in closed form
// ==========================================================================

/// The gas state a droplet sees - the three cell values plus the constant
/// transport properties [`crate::parcels::ParcelControls`] carries.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GasState {
    pub t: Scalar,
    /// Vapour mass fraction of the evaporating species, kg/kg of gas.
    pub y_vapour: Scalar,
    pub rho: Scalar,
    pub mu: Scalar,
    pub k: Scalar,
    pub cp: Scalar,
    /// `|u_p - u|`, m/s.
    pub u_rel: Scalar,
}

/// Everything (76.7) computes for one droplet at one instant. The host mirror
/// of `ofpEvapRate` in `cuda/parcels.cu`; the two are held together by
/// `tests::the_host_and_device_rates_agree`, which runs one sub-step on the
/// device and compares.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DropletRate {
    /// `dm_p/dt` for ONE droplet, kg/s. Negative for evaporation.
    pub mdot: Scalar,
    /// The blowing-corrected conductance `a = pi d k_g Nu`, W/K, so that the
    /// convective heat into the droplet is `a (T_g - T_p)`.
    pub conductance: Scalar,
    /// `h_v(T_p)`, J/kg.
    pub h_v: Scalar,
    /// Spalding mass transfer number. Infinite on the boiling branch, which
    /// is exactly why that branch exists.
    pub b_m: Scalar,
    /// Spalding heat transfer number - equal to `b_m` for
    /// [`MassTransfer::Spalding`], unused by [`MassTransfer::RanzMarshall`].
    pub b_t: Scalar,
    /// `d(-mdot h_v)/dT_p`, W/K, clamped at zero. The linearisation (76.8)
    /// relaxes against; it changes how fast the droplet reaches its steady
    /// temperature and, by construction, NOT what that temperature is.
    pub d_cooling_d_t: Scalar,
    /// Film Lewis number `k/(rho_f c_pg D_f)` - reported because it is what
    /// the wet-bulb gap of S76.13 is made of.
    pub lewis: Scalar,
    /// Surface vapour mass fraction.
    pub y_surface: Scalar,
    /// True when the droplet is at or above the boiling temperature and the
    /// heat-limited branch of (76.9) was taken.
    pub boiling: bool,
}

/// SPEC-LIT (76.7): the whole single-droplet closure, evaluated once.
///
/// `t_boil_p` is the boiling temperature at the ambient pressure - the number
/// [`boiling_temperature`] returns, passed in rather than recomputed so that
/// this function is a pure arithmetic mirror of the kernel and contains no
/// root find.
#[must_use]
pub fn droplet_rate(
    ctrl: &EvaporationControls,
    t_boil_p: Scalar,
    d: Scalar,
    t_p: Scalar,
    gas: &GasState,
) -> DropletRate {
    let l = &ctrl.liquid;
    let pi = std::f64::consts::PI as Scalar;
    let t_p = t_p.min(t_boil_p);

    // (76.4): the film state. The 1/3 rule, and the film density from the
    // ideal gas at constant pressure - so a hot gas is correctly thinner at
    // the surface than in the cell.
    let t_f = t_p + (gas.t - t_p) / 3.0;
    let rho_f = gas.rho * gas.t / t_f;
    let d_f = diffusivity(l, ctrl.p_ambient, t_f);
    let h_v = latent_heat(l, t_p);

    let re = gas.rho * gas.u_rel * d / gas.mu;
    let pr = gas.mu * gas.cp / gas.k;
    let sc = gas.mu / (rho_f * d_f);
    let sh0 = 2.0 + 0.6 * re.sqrt() * sc.cbrt();
    let nu0 = 2.0 + 0.6 * re.sqrt() * pr.cbrt();
    let lewis = gas.k / (rho_f * gas.cp * d_f);

    // (76.9): the boiling branch, taken on `T_p >= T_b(p)` ALONE and not on
    // the gas being hotter as well. At `T_b` the surface is pure vapour, so
    // `Y_s -> 1` and the diffusion-limited `B_M` diverges: the sub-cooled
    // branch is not merely less accurate there, it is a division by zero, and
    // it must not be reachable at `T_b` for ANY gas state. `ofpEvapRate` in
    // `cuda/parcels.cu` takes exactly this condition, and
    // `the_device_closure_is_the_host_closure` is what holds the two to it.
    if t_p >= t_boil_p {
        // Godsave-Spalding: the rate is set by the heat that reaches the
        // surface, not by how fast the vapour can diffuse away - which is
        // the physically right closure exactly where the diffusion-limited
        // one diverges.
        let b_t = l.cp_vapour * (gas.t - t_p) / h_v;
        let f_b = blowing_factor(b_t);
        let cond = pi * d * gas.k * nu0 * f_b;
        return DropletRate {
            mdot: -cond * (gas.t - t_p) / h_v,
            conductance: cond,
            h_v,
            b_m: Scalar::INFINITY,
            b_t,
            d_cooling_d_t: 0.0,
            lewis,
            y_surface: 1.0,
            boiling: true,
        };
    }

    // (76.5): the surface state.
    let ps = p_sat(ctrl.saturation, l, t_p);
    let x = (ps / ctrl.p_ambient).clamp(0.0, 1.0);
    let y_s = y_surface(l.w_vapour, ctrl.w_carrier, x);
    let b_m = (y_s - gas.y_vapour) / (1.0 - y_s);

    let (mdot, f_heat, b_t) = match ctrl.transfer {
        MassTransfer::RanzMarshall => (
            -pi * d * rho_f * d_f * sh0 * (y_s - gas.y_vapour),
            1.0,
            b_m,
        ),
        MassTransfer::Spalding => (
            -pi * d * rho_f * d_f * sh0 * b_m.ln_1p(),
            blowing_factor(b_m),
            b_m,
        ),
        MassTransfer::AbramzonSirignano => {
            let phi = (l.cp_vapour / gas.cp) * (sh0 / nu0) / lewis;
            let bt = (1.0 + b_m).powf(phi) - 1.0;
            (
                -pi * d * rho_f * d_f * sh0 * b_m.ln_1p(),
                blowing_factor(bt),
                bt,
            )
        }
    };
    let cond = pi * d * gas.k * nu0 * f_heat;

    // (76.8): the derivative of the evaporative cooling power, holding the
    // film properties frozen. Its only job is to make the relaxation stiff
    // enough to be stable; the fixed point it relaxes to is where the
    // RESIDUAL vanishes, and the residual does not contain it - so an
    // approximate derivative changes the path and not the destination.
    let dps = p_sat_derivative(ctrl.saturation, l, t_p);
    let dx = dps / ctrl.p_ambient;
    let den = ctrl.w_carrier + x * (l.w_vapour - ctrl.w_carrier);
    let dy = l.w_vapour * ctrl.w_carrier / (den * den) * dx;
    let dhv = latent_heat_derivative(l, t_p);
    let d_cooling = match ctrl.transfer {
        MassTransfer::RanzMarshall => {
            pi * d * rho_f * d_f * sh0 * (h_v * dy + (y_s - gas.y_vapour) * dhv)
        }
        _ => {
            // d ln(1 + B_M)/dT_p = (dY_s/dT_p)/(1 - Y_s), which is exact and
            // is the one place the algebra collapses: the (1 - Y_s)^2 of
            // dB_M/dY_s cancels the (1 - Y_s) of 1/(1 + B_M).
            let dlnb = dy / (1.0 - y_s);
            pi * d * rho_f * d_f * sh0 * (h_v * dlnb + b_m.ln_1p() * dhv)
        }
    };

    DropletRate {
        mdot,
        conductance: cond,
        h_v,
        b_m,
        b_t,
        d_cooling_d_t: d_cooling.max(0.0),
        lewis,
        y_surface: y_s,
        boiling: false,
    }
}

/// `F(B) = ln(1 + B)/B`, with the series that makes it exact at `B = 0`
/// rather than `0/0`.
#[must_use]
pub fn blowing_factor(b: Scalar) -> Scalar {
    if b.abs() > 1e-6 {
        b.ln_1p() / b
    } else {
        1.0 - b / 2.0 + b * b / 3.0
    }
}

// ==========================================================================
//  SPEC-LIT (76.11): the two closed forms the gates are measured against
// ==========================================================================

/// The steady droplet temperature, K - the root of
/// `a(T_g - T_p) + mdot h_v = 0`, by bisection up to `T_b(p)`.
///
/// This is S76's analogue of (66.4)'s [`crate::parcels::terminal_velocity`]:
/// the ANALYTIC statement of what the kernel's relaxation must converge to,
/// solved a different way - a bracketed root find here, an exponential
/// relaxation there - so that agreement between them is evidence and not
/// tautology.
///
/// It is also the number S76.13's gate holds against
/// [`crate::psychro::t_wb`], and the two are NOT the same quantity: this one
/// carries the Lewis number and the psychrometric one does not. The gap IS
/// the gate.
pub fn steady_temperature(
    ctrl: &EvaporationControls,
    t_boil_p: Scalar,
    d: Scalar,
    gas: &GasState,
) -> Result<Scalar> {
    let residual = |t_p: Scalar| -> Scalar {
        let r = droplet_rate(ctrl, t_boil_p, d, t_p, gas);
        r.conductance * (gas.t - t_p) + r.mdot * r.h_v
    };
    // The residual is monotone decreasing in T_p: the heating term falls and
    // the cooling term rises. So it is bracketed by anything cold enough and
    // the boiling point.
    // The top of the bracket is a nanokelvin BELOW the boiling point, and
    // deliberately: at `T_b` exactly, (76.9)'s branch makes the residual
    // `a(T_g - T_b) + mdot h_v` identically zero in exact arithmetic, so in
    // f64 its sign is one ulp of noise and the `residual(hi) > 0` test below
    // would decide "it boils" at random. A nanokelvin down, the sub-cooled
    // branch is what answers, and its answer there is unambiguous: `F(B_T)`
    // has collapsed and `ln(1 + B_M)` has not, so the residual is very
    // negative. That is also the content of S76.9's claim that a sub-cooled
    // droplet never reaches `T_b` from below.
    let mut lo: Scalar = (gas.t - 250.0).max(150.0).min(t_boil_p - 1e-3);
    let mut hi: Scalar = t_boil_p - 1e-9;
    if !(residual(lo) > 0.0) {
        return Err(Error::Config(format!(
            "parcels/evaporation: the steady droplet temperature is not bracketed below \
             {lo} K for T_g = {} K, Y_v = {} (SPEC-LIT S76.11)",
            gas.t, gas.y_vapour
        )));
    }
    if residual(hi) > 0.0 {
        // The balance cannot be met below boiling: the droplet boils, and
        // (76.9)'s branch is the answer.
        return Ok(t_boil_p);
    }
    for _ in 0..200 {
        let mid = 0.5 * (lo + hi);
        if mid <= lo || mid >= hi {
            break;
        }
        if residual(mid) > 0.0 {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    Ok(0.5 * (lo + hi))
}

/// The `d^2`-law slope `K = -d(d^2)/dt`, m2/s, at a given droplet
/// temperature - (76.11).
///
/// From `dm/dt = rho_l (pi/2) d^2 (dd/dt)` and `d(d^2)/dt = 2 d (dd/dt)`,
///
/// ```text
///   K = -4 mdot / (pi rho_l d)
/// ```
///
/// which for the Spalding rate at `Re -> 0` is exactly the textbook
/// `K = 8 rho_f D_f ln(1 + B_M)/rho_l`, for the Ranz-Marshall rate is
/// `K = 8 rho_f D_f (Y_s - Y_g)/rho_l`, and on the boiling branch is
/// Godsave's `K = 8 (k_g/c_pv) ln(1 + B_T)/rho_l`. It is INDEPENDENT of `d`
/// in all three, which is the whole content of the law and is what makes
/// `d^2(t)` a straight line.
#[must_use]
pub fn d2_law_slope(
    ctrl: &EvaporationControls,
    t_boil_p: Scalar,
    rho_liquid: Scalar,
    d: Scalar,
    t_p: Scalar,
    gas: &GasState,
) -> Scalar {
    let r = droplet_rate(ctrl, t_boil_p, d, t_p, gas);
    -4.0 * r.mdot / (std::f64::consts::PI as Scalar * rho_liquid * d)
}
