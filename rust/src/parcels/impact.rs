// meteor-cfd - Copyright (c) 2026 주식회사 메테오시뮬레이션 (Meteo Simulation Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.

//! Droplet-wall impact - the regime map of SPEC-LIT S78.
//!
//! This module is the host mirror of the impact arithmetic in
//! `cuda/parcels.cu`: the dimensionless groups an impact is classified by,
//! the two published splash criteria, the surface tension every one of them
//! divides by, and the map itself. Nothing here runs in the time loop. It is
//! what a case is built from, what a report prints, and what S78.9's gates
//! measure the device against.
//!
//! **The map is a classification, not a closure.** It answers one question -
//! which of the four outcomes this impact is - and the parcel-side ACTION
//! that follows is S78.5's, in the walk. Three of the four outcomes share
//! one action (the droplet's mass stays on the wall), which is why S78.6's
//! ledger can be an exact partition of the pool rather than a sum over
//! regimes.
//!
//! Written from:
//!   C. Bai, A. D. Gosman, *Development of Methodology for Spray Impingement
//!     Simulation*, SAE Technical Paper 950283 (1995), DOI `10.4271/950283` -
//!     the four-regime dry-wall map (adhesion / rebound / spread / splash),
//!     its Weber-number boundaries, and the splash threshold
//!     `We_c = A La^(-0.18)` whose `A` is roughness-dependent
//!   C. Mundo, M. Sommerfeld, C. Tropea, *Droplet-wall collisions:
//!     Experimental studies of the deformation and breakup process*, Int. J.
//!     Multiphase Flow 21 (1995) 151-173, DOI `10.1016/0301-9322(94)00069-V`
//!     - the deposition/splashing parameter `K = Oh Re^1.25` and its measured
//!     threshold `K = 57.7`
//!   A. L. Yarin, *Drop impact dynamics: splashing, spreading, receding,
//!     bouncing...*, Annu. Rev. Fluid Mech. 38 (2006) 159-192, DOI
//!     `10.1146/annurev.fluid.38.050304.092144` - the review that says how
//!     approximately right any of these correlations is, and why the
//!     thresholds are controls here rather than constants
//!   IAPWS, *Revised Release on Surface Tension of Ordinary Water
//!     Substance*, R1-76 (2014) - `sigma = B tau^mu (1 + b tau)` with
//!     `B = 235.8 mN/m`, `b = -0.625`, `mu = 1.256`, `tau = 1 - T/T_c`, a
//!     freely published international standard, whose own table S78.9's
//!     first gate is measured against
//!   NIST Chemistry WebBook, SRD 69 (US Government, public domain) - the
//!     liquid viscosity of water that sets the default `muLiquid`
//!
//! No GPL-licensed source was consulted.

use crate::error::{Error, Result};
use crate::io::contract;
use crate::Scalar;

#[cfg(test)]
mod tests;

/// The critical temperature of water, K - IAPWS R1-76's `T_c`, and the same
/// number [`crate::parcels::LiquidProperties::water`] carries.
pub const T_CRITICAL_WATER: Scalar = 647.096;

// ==========================================================================
//  Surface tension
// ==========================================================================

/// SPEC-LIT (78.2): where the `sigma` in the Weber number comes from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SurfaceTension {
    /// The single number [`WallImpactControls::sigma`], whatever the droplet
    /// temperature is. The default, because it is the only choice that is
    /// right for a liquid that is not water.
    #[default]
    Constant,
    /// IAPWS R1-76's correlation, evaluated at the droplet's own
    /// temperature. **Water only** - selecting it is the case's statement
    /// that the parcels are water, and nothing here can check that.
    IapwsR176,
}

impl SurfaceTension {
    pub const NAMES: &'static [&'static str] = &["constant", "iapwsR1-76"];

    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Self::Constant => "constant",
            Self::IapwsR176 => "iapwsR1-76",
        }
    }

    pub(crate) fn code(self) -> i32 {
        match self {
            Self::Constant => 0,
            Self::IapwsR176 => 1,
        }
    }

    pub fn from_name(s: &str) -> Result<Self> {
        match s {
            "constant" => Ok(Self::Constant),
            "iapwsR1-76" | "iapws" | "iapwsR176" => Ok(Self::IapwsR176),
            other => contract::unsupported(
                "parcels/surfaceTension",
                other,
                Self::NAMES,
                "constant",
                Self::Constant,
            ),
        }
    }
}

/// SPEC-LIT (78.2): surface tension, N/m.
///
/// ```text
///   constant     sigma = sigma_0                                     (78.2a)
///   iapwsR1-76   sigma = B tau^mu (1 + b tau),   tau = 1 - T/T_c     (78.2b)
///                B = 0.2358 N/m, b = -0.625, mu = 1.256, T_c = 647.096 K
/// ```
///
/// (78.2b) is IAPWS R1-76 verbatim. It is **clamped at zero at and above
/// `T_c`** rather than returning a negative surface tension or a NaN from
/// `pow(negative, 1.256)`: a droplet above the critical temperature is not a
/// droplet, and the Weber number of one is `+inf`, which the map reads as
/// "splash" - the physically right answer, reached without a branch anywhere
/// else.
#[must_use]
pub fn surface_tension(model: SurfaceTension, sigma_0: Scalar, t: Scalar) -> Scalar {
    match model {
        SurfaceTension::Constant => sigma_0,
        SurfaceTension::IapwsR176 => {
            let tau = 1.0 - t / T_CRITICAL_WATER;
            if tau <= 0.0 {
                return 0.0;
            }
            0.2358 * tau.powf(1.256) * (1.0 - 0.625 * tau)
        }
    }
}

// ==========================================================================
//  The splash criterion
// ==========================================================================

/// SPEC-LIT (78.4): which published threshold separates spread from splash.
///
/// The two do not agree, and S78.10 measures by how much rather than
/// choosing one and calling it settled.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SplashCriterion {
    /// Mundo, Sommerfeld & Tropea (1995): `K = Oh Re^1.25 > K_crit`, with
    /// `K_crit = 57.7` measured. The default, for two reasons: it is the
    /// threshold the design note prescribes, and its decision form is
    /// polynomial (78.4b), so it is bit-stable where the other is not.
    #[default]
    Mundo,
    /// Bai & Gosman (1995): `We > A La^(-0.18)`, with `A` the roughness
    /// parameter [`WallImpactControls::splash_a`].
    BaiGosman,
}

impl SplashCriterion {
    pub const NAMES: &'static [&'static str] = &["mundo", "baiGosman"];

    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Self::Mundo => "mundo",
            Self::BaiGosman => "baiGosman",
        }
    }

    pub(crate) fn code(self) -> i32 {
        match self {
            Self::Mundo => 0,
            Self::BaiGosman => 1,
        }
    }

    pub fn from_name(s: &str) -> Result<Self> {
        match s {
            "mundo" => Ok(Self::Mundo),
            "baiGosman" | "bai" => Ok(Self::BaiGosman),
            other => contract::unsupported(
                "parcels/splashCriterion",
                other,
                Self::NAMES,
                "mundo",
                Self::Mundo,
            ),
        }
    }
}

// ==========================================================================
//  The four outcomes
// ==========================================================================

/// SPEC-LIT (78.5): what an impact IS. The action that follows is the walk's.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum WallRegime {
    /// `We < weStick`. The droplet adheres. Surface tension wins.
    #[default]
    Stick,
    /// `weStick <= We < weSpread`. The droplet bounces off the air layer it
    /// could not push out of the way.
    Rebound,
    /// `weSpread <= We`, below the splash threshold. The droplet spreads into
    /// a lamella that stays on the wall.
    Spread,
    /// Above the splash threshold. The lamella breaks up and part of the mass
    /// leaves the wall again as secondary droplets - **which this section
    /// detects and does not emit** (SPEC-LIT 78.7).
    Splash,
}

impl WallRegime {
    pub const NAMES: &'static [&'static str] = &["stick", "rebound", "spread", "splash"];

    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Self::Stick => "stick",
            Self::Rebound => "rebound",
            Self::Spread => "spread",
            Self::Splash => "splash",
        }
    }

    #[must_use]
    pub fn code(self) -> i32 {
        match self {
            Self::Stick => 0,
            Self::Rebound => 1,
            Self::Spread => 2,
            Self::Splash => 3,
        }
    }

    /// True for the three outcomes that leave the droplet's mass on the wall.
    ///
    /// `Splash` is among them **here and not in the papers**: the parent's
    /// mass is deposited whole because no child parcels are emitted (SPEC-LIT 78.7),
    /// which makes the deposit an upper bound and is why `n_splash` is a
    /// reported counter rather than an internal detail.
    #[must_use]
    pub fn deposits(self) -> bool {
        !matches!(self, Self::Rebound)
    }
}

// ==========================================================================
//  The dimensionless groups
// ==========================================================================

/// SPEC-LIT (78.3): the groups an impact is classified by, and the
/// fourth-power form the splash decision is actually taken in.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ImpactNumbers {
    /// `We = rho_l d u_n^2 / sigma` - inertia against surface tension.
    pub we: Scalar,
    /// `Re = rho_l u_n d / mu_l` - the LIQUID Reynolds number, not the gas
    /// one the drag law uses.
    pub re: Scalar,
    /// `Oh = mu_l / sqrt(rho_l sigma d) = sqrt(We)/Re`.
    pub oh: Scalar,
    /// `La = rho_l sigma d / mu_l^2 = 1/Oh^2`, the group Bai & Gosman's
    /// threshold is written in.
    pub la: Scalar,
    /// `K = Oh Re^1.25`, Mundo's parameter. Reported; the decision uses
    /// [`Self::k4`], which is the same number to the fourth power and is
    /// formed without `pow`.
    pub k: Scalar,
    /// `K^4 = We^2 Re`. An identity, and the whole of (78.4b).
    pub k4: Scalar,
    /// The surface tension the groups were formed with, N/m.
    pub sigma: Scalar,
}

/// SPEC-LIT (78.3): form the groups for one impact.
///
/// `u_n` is the **magnitude of the wall-normal component** of the impact
/// velocity, m/s. The tangential component does not enter either published
/// threshold in this map, which S78.12 records as a limitation rather than
/// leaving it to be inferred from the signature.
///
/// A zero `sigma` gives `We = +inf`, which the map reads as a splash. That is
/// deliberate and is the only place a degenerate property reaches the
/// classification: [`WallImpactControls::validate`] refuses a non-positive
/// `sigma_0` at setup, and (78.2b) can only reach zero at and above the
/// critical temperature, where "this is not a droplet any more" is the right
/// answer.
#[must_use]
pub fn impact_numbers(
    rho_liquid: Scalar,
    mu_liquid: Scalar,
    sigma: Scalar,
    d: Scalar,
    u_n: Scalar,
) -> ImpactNumbers {
    let we = rho_liquid * d * u_n * u_n / sigma;
    let re = rho_liquid * u_n * d / mu_liquid;
    let oh = mu_liquid / (rho_liquid * sigma * d).sqrt();
    let la = rho_liquid * sigma * d / (mu_liquid * mu_liquid);
    ImpactNumbers {
        we,
        re,
        oh,
        la,
        k: oh * re.powf(1.25),
        // K^4 = (Oh Re)^4 Re = We^2 Re, because Oh Re = sqrt(We). Formed
        // from `we` and `re` and not from `oh`, so that the decision needs
        // neither `pow` nor `sqrt` and is therefore bit-stable across
        // compute capabilities where (78.4a) is not (S38.6).
        k4: we * we * re,
        sigma,
    }
}

// ==========================================================================
//  What a case can say
// ==========================================================================

/// SPEC-LIT S78.8: everything the impact map reads.
///
/// Written out field by field wherever it is built, per S13.4.1(b).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WallImpactControls {
    /// Which published threshold separates spread from splash.
    pub splash: SplashCriterion,
    /// Where `sigma` comes from.
    pub tension: SurfaceTension,
    /// Surface tension, N/m. Read by [`SurfaceTension::Constant`], and
    /// validated always.
    ///
    /// Default: IAPWS R1-76 at 293.15 K, whose own table gives 72.74 mN/m.
    pub sigma: Scalar,
    /// LIQUID dynamic viscosity, Pa s. The gas viscosity is
    /// [`crate::parcels::ParcelControls::mu_gas`]; the two differ by three
    /// orders of magnitude, and `Oh`, `Re` and `La` here all take this one.
    ///
    /// Default: water at 293.15 K and 0.1 MPa, 1.0016 mPa s (NIST Chemistry
    /// WebBook).
    pub mu_liquid: Scalar,
    /// Below this impact Weber number the droplet adheres. Bai & Gosman's
    /// dry-wall map puts it at 2.
    pub we_stick: Scalar,
    /// At and above this impact Weber number a rebound becomes a spread. Bai
    /// & Gosman's dry-wall map puts it at 20.
    pub we_spread: Scalar,
    /// Mundo's threshold on `K = Oh Re^1.25`, measured at 57.7.
    pub k_crit: Scalar,
    /// Bai & Gosman's `A` in `We_c = A La^(-0.18)`. Roughness-dependent: they
    /// report a range over the surfaces they tested, and 2630 is the smooth
    /// end of it.
    pub splash_a: Scalar,
}

impl Default for WallImpactControls {
    fn default() -> Self {
        Self {
            splash: SplashCriterion::Mundo,
            tension: SurfaceTension::Constant,
            sigma: 0.072_74,
            mu_liquid: 1.0016e-3,
            we_stick: 2.0,
            we_spread: 20.0,
            k_crit: 57.7,
            splash_a: 2630.0,
        }
    }
}

impl WallImpactControls {
    /// Everything that can be wrong with these numbers, named.
    ///
    /// Called by [`crate::parcels::ParcelControls::validate`] **always**, and
    /// not only when the map is selected, for the reason S66.11 already gives
    /// for the three thermal properties: a number that is nonsense when it is
    /// read is nonsense when it is written, and the error is more use at
    /// setup than three steps into a run.
    pub fn validate(&self) -> Result<()> {
        let bad = |what: &str, v: Scalar| {
            Err(Error::Config(format!(
                "parcels: {what} is {v}; SPEC-LIT S78.8 requires it finite and positive"
            )))
        };
        if !(self.sigma > 0.0) || !self.sigma.is_finite() {
            return bad("sigma", self.sigma);
        }
        if !(self.mu_liquid > 0.0) || !self.mu_liquid.is_finite() {
            return bad("muLiquid", self.mu_liquid);
        }
        if !(self.k_crit > 0.0) || !self.k_crit.is_finite() {
            return bad("kCrit", self.k_crit);
        }
        if !(self.splash_a > 0.0) || !self.splash_a.is_finite() {
            return bad("splashA", self.splash_a);
        }
        if !(self.we_stick >= 0.0) || !self.we_stick.is_finite() {
            return Err(Error::Config(format!(
                "parcels: weStick is {}; SPEC-LIT S78.8 requires it finite and >= 0",
                self.we_stick
            )));
        }
        if !(self.we_spread >= self.we_stick) || !self.we_spread.is_finite() {
            return Err(Error::Config(format!(
                "parcels: weSpread {} is below weStick {}, which would make the rebound \
                 band empty and the map a different map; SPEC-LIT S78.4 orders them",
                self.we_spread, self.we_stick
            )));
        }
        Ok(())
    }

    /// One line for the startup banner - SPEC-LIT S13.4.2.
    #[must_use]
    pub fn describe(&self) -> String {
        format!(
            "parcels/impact: splash={} tension={} sigma={} muLiquid={} weStick={} \
             weSpread={} {} (SPEC-LIT S78)",
            self.splash.name(),
            self.tension.name(),
            self.sigma,
            self.mu_liquid,
            self.we_stick,
            self.we_spread,
            match self.splash {
                SplashCriterion::Mundo => format!("kCrit={}", self.k_crit),
                SplashCriterion::BaiGosman => format!("splashA={}", self.splash_a),
            },
        )
    }

    /// SPEC-LIT (78.4a): the Weber number Bai & Gosman's threshold sits at,
    /// for a given Laplace number - `We_c = A La^(-0.18)`.
    #[must_use]
    pub fn bai_gosman_we_c(&self, la: Scalar) -> Scalar {
        self.splash_a * la.powf(-0.18)
    }

    /// SPEC-LIT (78.4): is this impact above the splash threshold?
    ///
    /// The Mundo branch is the fourth-power comparison `We^2 Re > K_crit^4`,
    /// which is `K > K_crit` exactly - both sides are non-negative and
    /// `x -> x^4` is strictly increasing there - and needs neither `pow` nor
    /// `sqrt`.
    #[must_use]
    pub fn splashing(&self, n: &ImpactNumbers) -> bool {
        match self.splash {
            SplashCriterion::Mundo => {
                let kc = self.k_crit * self.k_crit;
                n.k4 > kc * kc
            }
            SplashCriterion::BaiGosman => n.we > self.bai_gosman_we_c(n.la),
        }
    }

    /// SPEC-LIT (78.5): the map.
    ///
    /// ```text
    ///   splashing        -> Splash
    ///   We <  weStick    -> Stick
    ///   We <  weSpread   -> Rebound
    ///   otherwise        -> Spread
    /// ```
    ///
    /// The splash test is taken FIRST, which is what makes the map total: on
    /// a liquid viscous enough for `We_c` to fall below `weSpread` the spread
    /// band is empty rather than overlapping, and no ordering of the
    /// thresholds can produce two answers.
    #[must_use]
    pub fn regime(&self, n: &ImpactNumbers) -> WallRegime {
        if self.splashing(n) {
            WallRegime::Splash
        } else if n.we < self.we_stick {
            WallRegime::Stick
        } else if n.we < self.we_spread {
            WallRegime::Rebound
        } else {
            WallRegime::Spread
        }
    }

    /// The whole classification, from the physical state of one impact.
    ///
    /// `t_droplet` is read only by [`SurfaceTension::IapwsR176`].
    #[must_use]
    pub fn classify(
        &self,
        rho_liquid: Scalar,
        d: Scalar,
        t_droplet: Scalar,
        u_n: Scalar,
    ) -> (ImpactNumbers, WallRegime) {
        let sigma = surface_tension(self.tension, self.sigma, t_droplet);
        let n = impact_numbers(rho_liquid, self.mu_liquid, sigma, d, u_n);
        let r = self.regime(&n);
        (n, r)
    }

    /// SPEC-LIT (78.4c): the LOWEST normal impact speed at which the map
    /// returns `to`, for a droplet of diameter `d` - the inverse of the map,
    /// in closed form, and what S78.9's boundary gate is measured against.
    ///
    /// The map is monotone in `u_n` (asserted, over 4000 speeds), so this is
    /// also the boundary between `to` and the regime below it.
    ///
    /// `We = rho_l d u_n^2/sigma` inverts to `u_n = sqrt(We sigma/(rho_l d))`.
    /// Mundo's threshold inverts too, because `K^4 = We^2 Re` is a single
    /// monotone power of `u_n`:
    ///
    /// ```text
    ///   K^4 = (rho_l d/sigma)^2 (rho_l d/mu_l) u_n^5  =>  u_n = (K_c^4/C)^(1/5)
    /// ```
    ///
    /// `Stick` has no lower boundary and returns `None`.
    #[must_use]
    pub fn boundary_speed(
        &self,
        rho_liquid: Scalar,
        d: Scalar,
        t_droplet: Scalar,
        to: WallRegime,
    ) -> Option<Scalar> {
        let sigma = surface_tension(self.tension, self.sigma, t_droplet);
        let u_of_we = |we: Scalar| (we * sigma / (rho_liquid * d)).sqrt();
        match to {
            WallRegime::Stick => None,
            WallRegime::Rebound => Some(u_of_we(self.we_stick)),
            WallRegime::Spread => Some(u_of_we(self.we_spread)),
            WallRegime::Splash => match self.splash {
                SplashCriterion::Mundo => {
                    let a = rho_liquid * d / sigma;
                    let c = a * a * (rho_liquid * d / self.mu_liquid);
                    let kc = self.k_crit * self.k_crit;
                    Some((kc * kc / c).powf(0.2))
                }
                SplashCriterion::BaiGosman => {
                    // `La` does not depend on `u_n`, so `We_c` is a constant
                    // here and the inversion is the same one.
                    let la = rho_liquid * sigma * d / (self.mu_liquid * self.mu_liquid);
                    Some(u_of_we(self.bai_gosman_we_c(la)))
                }
            },
        }
    }
}
