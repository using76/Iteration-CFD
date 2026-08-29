// meteor-cfd - Copyright (c) 2026 주식회사 메테오시뮬레이션 (Meteo Simulation Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.

//! Generalised-Newtonian viscosity - SPEC-LIT §38.
//!
//! Written from:
//!   W. Ostwald, *Kolloid-Z.* 36 (1925) 99-117; A. de Waele (1923) - the
//!     power law
//!   M. M. Cross, *J. Colloid Sci.* 20 (1965) 417-437
//!   P. J. Carreau, *Trans. Soc. Rheol.* 16 (1972) 99-127
//!   K. Yasuda, R. C. Armstrong, R. E. Cohen, *Rheol. Acta* 20 (1981) 163-178
//!   W. H. Herschel, R. Bulkley, *Kolloid-Z.* 39 (1926) 291-300
//!   N. Casson, in C. C. Mill (ed.), *Rheology of Disperse Systems*, Pergamon
//!     (1959) 84-104
//!   T. C. Papanastasiou, *J. Rheol.* 31 (1987) 385-404 - the regularisation
//!   M. Bercovier, M. Engelman, *J. Comput. Phys.* 36 (1980) 313-326 - the
//!     alternative regularisation, named here and not implemented
//!   I. A. Frigaard, C. Nouar, *J. Non-Newtonian Fluid Mech.* 127 (2005) 1-26
//!     - what regularisation costs, and why no finite `m` recovers a true plug
//!   R. B. Bird, R. C. Armstrong, O. Hassager, *Dynamics of Polymeric
//!     Liquids*, vol. 1, 2nd ed., Wiley (1987) - the family
//!   R. P. Chhabra, J. F. Richardson, *Non-Newtonian Flow and Applied
//!     Rheology*, 2nd ed. (2008) - Buckingham-Reiner
//!   ofgpu `SPEC-LIT.md` §38 (all of it), §13.4 (what happens to a setting
//!     this solver does not have), §5 (the momentum equation this feeds)
//! No GPL-licensed source was consulted.
//!
//! # What this module owns
//!
//! One function of one scalar: given the strain-rate magnitude `gdot`, what
//! is the apparent viscosity? Everything else - the gradient, the invariant,
//! the face interpolation, the laplacian - already existed.
//!
//! # The invariant is not computed here
//!
//! `gdot = sqrt(2 D:D)` is exactly what `turbStrainRateMag`
//! (`cuda/turbulence.cu`) has computed since the turbulence models were
//! written, and it had no caller: the RAS models take their production from
//! [`crate::fv::turbulence_production`] instead. §38.1 makes this module its
//! first user, and [`RheologyKernels`] deliberately loads that kernel out of
//! the turbulence module rather than shipping a second copy of the same six
//! lines of arithmetic. Two implementations of one invariant is how the two
//! drift.
//!
//! # Units: dynamic in, kinematic out
//!
//! SPEC-LIT §5's momentum equation is KINEMATIC - `nu` is m²/s and there is
//! no density in it anywhere. Every closure below is fitted in DYNAMIC units.
//! Both facts are true at once and the conversion cannot be guessed, so
//! §38.4 makes the rule explicit: **a case writes the literature's dynamic
//! numbers and states its own `rho`**, and this module divides once, on the
//! host, before anything reaches the device. `rho` is REQUIRED for every
//! non-Newtonian model and refused by name if absent, because `K = 0.35`
//! means two viscosities a thousand apart depending on which unit was meant.
//! That is the §13.4 defect this project keeps finding, in its purest form.
//!
//! # The default does not move
//!
//! [`RheologyModel::Newtonian`] is the default and is not merely *close* to
//! the pre-§38 momentum equation, it is bitwise it: no kernel in this file is
//! launched, `nu_lam` stays at the uniform `ctrl.nu` it was filled with, and
//! `nu_eff[i] = nu_lam[i] + nu_t[i]` is the same IEEE-754 addition
//! `nu_eff[i] = nu_t[i] + nu` was. `the_uniform_buffer_is_the_scalar_bitwise`
//! measures it rather than asserting it.

use cudarc::driver::{CudaFunction, PushKernelArg};

use crate::device::{cfg_for, DevBuf, Gpu, KernelSet};
use crate::error::{Error, Result};
use crate::io::contract::unsupported;
use crate::io::dict::FoamDict;
use crate::{Label, Scalar, Tensor, Vec3};

// ==========================================================================
//  The model
// ==========================================================================

/// Which closure supplies the laminar viscosity - SPEC-LIT §38.2.
///
/// The discriminants are the `OFRHEO_*` codes in `cuda/rheology.cu` and
/// [`model_codes_match_the_device`] pins the two together, the same
/// discipline `bc_kind_values_match_the_device` applies to `BcKind`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum RheologyModel {
    /// `mu = mu`, i.e. the case's own `nu`. The DEFAULT, deliberately: every
    /// measurement this project has recorded was made with a single case-wide
    /// `nu`, and a default that changed would move all of them at once.
    #[default]
    Newtonian = 0,
    /// `mu = K gdot^(n-1)` - Ostwald / de Waele.
    PowerLaw = 1,
    /// `mu = mu_inf + (mu_0 - mu_inf)/(1 + (lambda gdot)^a)` - Cross.
    Cross = 2,
    /// `mu = mu_inf + (mu_0 - mu_inf)[1 + (lambda gdot)^a]^((n-1)/a)`.
    /// `a = 2` is Bird-Carreau proper; a general `a` is Carreau-Yasuda, and
    /// one formula serves both because they ARE one formula.
    BirdCarreau = 3,
    /// `tau = tau_0 + K gdot^n`, regularised by Papanastasiou (§38.3).
    HerschelBulkley = 4,
    /// `sqrt(tau) = sqrt(tau_0) + sqrt(mu_c gdot)`, regularised the same way.
    Casson = 5,
}

impl RheologyModel {
    /// The spelling a case file uses, and what [`Self::parse`] prints.
    pub fn name(self) -> &'static str {
        match self {
            Self::Newtonian => "Newtonian",
            Self::PowerLaw => "powerLaw",
            Self::Cross => "CrossPowerLaw",
            Self::BirdCarreau => "BirdCarreau",
            Self::HerschelBulkley => "HerschelBulkley",
            Self::Casson => "Casson",
        }
    }

    /// Every spelling a case may name, for a §13.4 menu.
    pub const NAMES: [&'static str; 6] = [
        "Newtonian",
        "powerLaw",
        "CrossPowerLaw",
        "BirdCarreau",
        "HerschelBulkley",
        "Casson",
    ];

    /// SPEC-LIT §13.4: a recognised spelling selects the model; anything else
    /// is an error that NAMES the alternatives (`-permissive` substitutes
    /// `Newtonian`, the default, and says so).
    ///
    /// `constant` is accepted as an alias for `Newtonian` because that is the
    /// word an OpenFOAM `physicalProperties` writes, and this crate's own
    /// `blockgen` has been writing `viscosityModel constant;` into every
    /// generated case since it was written - a setting the reader did not
    /// look at at all until §38.
    pub fn parse(setting: &str, value: &str) -> Result<Self> {
        match value {
            "Newtonian" | "newtonian" | "constant" => Ok(Self::Newtonian),
            "powerLaw" | "PowerLaw" => Ok(Self::PowerLaw),
            "CrossPowerLaw" | "crossPowerLaw" | "Cross" | "cross" => Ok(Self::Cross),
            "BirdCarreau" | "birdCarreau" | "CarreauYasuda" | "carreauYasuda" => {
                Ok(Self::BirdCarreau)
            }
            "HerschelBulkley" | "herschelBulkley" => Ok(Self::HerschelBulkley),
            "Casson" | "casson" => Ok(Self::Casson),
            other => unsupported(
                setting,
                other,
                &Self::NAMES,
                "Newtonian (the case's own single nu)",
                Self::Newtonian,
            ),
        }
    }

    /// True for the two ideal-viscoplastic models, which are singular at
    /// `gdot = 0` and therefore need §38.3's regularisation and its `m`.
    #[inline]
    pub fn is_yield_stress(self) -> bool {
        matches!(self, Self::HerschelBulkley | Self::Casson)
    }

    /// Every coefficient key this model reads, for §38.7's "refuses a
    /// coefficient it does not use". [`COMMON_KEYS`] are legal for all of
    /// them and are not repeated here.
    pub fn coefficient_keys(self) -> &'static [&'static str] {
        match self {
            Self::Newtonian => &[],
            Self::PowerLaw => &["K", "n"],
            Self::Cross => &["mu0", "muInf", "lambda", "a"],
            Self::BirdCarreau => &["mu0", "muInf", "lambda", "n", "a"],
            Self::HerschelBulkley => &["tau0", "K", "n", "m"],
            Self::Casson => &["tau0", "muC", "m"],
        }
    }
}

/// The keys every model accepts: the density §38.4 requires, the clip, the
/// floor and the fixed-point relaxation.
pub const COMMON_KEYS: [&str; 5] = ["rho", "muMin", "muMax", "gammaDotFloor", "relax"];

// ==========================================================================
//  Coefficients
// ==========================================================================

/// The fluid, in the literature's DYNAMIC units, plus §38.3's two numerical
/// parameters - SPEC-LIT §38.2, §38.3, §38.4.
///
/// Stored as written so the run banner can print what the case said; the
/// kinematic values the device gets come out of [`Self::kinematic`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RheologyCoeffs {
    pub model: RheologyModel,

    /// Density [kg/m³]. REQUIRED for every non-Newtonian model - §38.4.
    pub rho: Scalar,

    /// Zero-shear viscosity [Pa s]. Carries `mu_c` for [`RheologyModel::Casson`].
    pub mu0: Scalar,
    /// Infinite-shear viscosity [Pa s].
    pub mu_inf: Scalar,
    /// Consistency [Pa s^n].
    pub k: Scalar,
    /// Power-law index [-].
    pub n: Scalar,
    /// Time constant [s].
    pub lambda: Scalar,
    /// Cross's exponent, or Carreau-Yasuda's `a` [-]. One slot, because it is
    /// one role.
    pub a: Scalar,
    /// Yield stress [Pa].
    pub tau0: Scalar,

    /// Papanastasiou's regularisation parameter [s] - §38.3. A NUMERICAL
    /// parameter, not a fluid property, and printed as one.
    pub m_reg: Scalar,

    /// Clip on the apparent viscosity [Pa s]. The defaults, `0` and `+inf`,
    /// are an EXACT no-op: `max(0, mu) = mu` and `min(inf, mu) = mu` for any
    /// finite non-negative `mu`, to the last bit.
    pub mu_min: Scalar,
    pub mu_max: Scalar,

    /// *DESIGN*, §38.3. `gdot` is floored here before anything divides by it
    /// or raises it to a power, so a uniform field on the first iteration
    /// (`gdot = 0` exactly) gives a finite viscosity rather than `0^(n-1)`.
    pub gdot_floor: Scalar,

    /// *DESIGN*, §38.5(iv). Elementwise relaxation of the viscosity fixed
    /// point. `1` is no relaxation and is bitwise `nu = mu(gdot)`.
    pub relax: Scalar,
}

impl Default for RheologyCoeffs {
    /// Newtonian, and every other field inert. This is what a case that says
    /// nothing means, and under it no kernel in this module ever launches.
    fn default() -> Self {
        Self {
            model: RheologyModel::Newtonian,
            rho: 1.0,
            mu0: 0.0,
            mu_inf: 0.0,
            k: 0.0,
            n: 1.0,
            lambda: 0.0,
            a: 2.0,
            tau0: 0.0,
            m_reg: 0.0,
            mu_min: 0.0,
            mu_max: Scalar::INFINITY,
            gdot_floor: DEFAULT_GDOT_FLOOR,
            relax: 1.0,
        }
    }
}

/// *DESIGN*, §38.3: four decades below any shear rate a case is about, and
/// printed in the banner so it is a stated number rather than folklore.
pub const DEFAULT_GDOT_FLOOR: Scalar = 1e-6;

/// The same coefficients divided by `rho`, which is what the kernel and
/// [`apparent_viscosity`] work in - §38.4.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct KinematicCoeffs {
    pub model: RheologyModel,
    /// `mu_0/rho`, or `mu_c/rho` for Casson [m²/s].
    pub nu0: Scalar,
    pub nu_inf: Scalar,
    /// `K/rho` [m² s^(n-2)].
    pub k: Scalar,
    pub n: Scalar,
    pub lambda: Scalar,
    pub a: Scalar,
    /// `tau_0/rho` [m²/s²].
    pub t0: Scalar,
    pub m_reg: Scalar,
    pub gdot_floor: Scalar,
    pub nu_min: Scalar,
    pub nu_max: Scalar,
    pub relax: Scalar,
}

impl RheologyCoeffs {
    /// Divide by `rho` once, here, so nothing downstream has to know a
    /// density exists - §38.4.
    pub fn kinematic(&self) -> KinematicCoeffs {
        let r = self.rho;
        KinematicCoeffs {
            model: self.model,
            nu0: self.mu0 / r,
            nu_inf: self.mu_inf / r,
            k: self.k / r,
            n: self.n,
            lambda: self.lambda,
            a: self.a,
            t0: self.tau0 / r,
            m_reg: self.m_reg,
            gdot_floor: self.gdot_floor,
            nu_min: self.mu_min / r,
            // inf/rho is inf, which is the exact no-op the default wants.
            nu_max: self.mu_max / r,
            relax: self.relax,
        }
    }

    /// True when this is the pre-§38 momentum equation and nothing in this
    /// module should run at all.
    #[inline]
    pub fn is_newtonian(&self) -> bool {
        self.model == RheologyModel::Newtonian
    }

    /// Everything §38.7 requires of the numbers, checked at read time rather
    /// than at the first NaN.
    ///
    /// `setting` names where the case wrote the block, so the message points
    /// at `physics.fluid.rheology` for a JSONC case and at
    /// `constant/physicalProperties/rheology` for an OpenFOAM one.
    pub fn validate(&self, setting: &str) -> Result<()> {
        if self.is_newtonian() {
            return Ok(());
        }
        let bad = |what: &str, v: Scalar, want: &str| -> Error {
            Error::Config(format!(
                "{setting}: {what} = {v} for viscosityModel {}; SPEC-LIT 38.7 \
                 requires {want}",
                self.model.name()
            ))
        };

        if !(self.rho > 0.0) || !self.rho.is_finite() {
            return Err(bad("rho", self.rho, "a positive finite density (38.4)"));
        }
        if !(self.gdot_floor > 0.0) || !self.gdot_floor.is_finite() {
            return Err(bad(
                "gammaDotFloor",
                self.gdot_floor,
                "a positive finite floor (38.3)",
            ));
        }
        if !(self.relax > 0.0 && self.relax <= 1.0) {
            return Err(bad("relax", self.relax, "0 < relax <= 1 (38.5 iv)"));
        }
        if !(self.mu_min >= 0.0) || !(self.mu_max >= self.mu_min) || self.mu_max <= 0.0 {
            return Err(Error::Config(format!(
                "{setting}: muMin = {} and muMax = {}; SPEC-LIT 38.7 requires \
                 0 <= muMin <= muMax with muMax > 0",
                self.mu_min, self.mu_max
            )));
        }

        match self.model {
            RheologyModel::Newtonian => {}
            RheologyModel::PowerLaw => {
                if !(self.k > 0.0) {
                    return Err(bad("K", self.k, "a positive consistency"));
                }
                if !(self.n > 0.0) {
                    return Err(bad("n", self.n, "a positive power-law index"));
                }
            }
            RheologyModel::Cross => {
                if !(self.mu0 > 0.0) {
                    return Err(bad("mu0", self.mu0, "a positive zero-shear viscosity"));
                }
                if !(self.mu_inf >= 0.0) {
                    return Err(bad("muInf", self.mu_inf, "muInf >= 0"));
                }
                if !(self.lambda >= 0.0) {
                    return Err(bad("lambda", self.lambda, "lambda >= 0"));
                }
                if !(self.a > 0.0) {
                    return Err(bad("a", self.a, "a positive exponent"));
                }
            }
            RheologyModel::BirdCarreau => {
                if !(self.mu0 > 0.0) {
                    return Err(bad("mu0", self.mu0, "a positive zero-shear viscosity"));
                }
                if !(self.mu_inf >= 0.0) {
                    return Err(bad("muInf", self.mu_inf, "muInf >= 0"));
                }
                if !(self.lambda >= 0.0) {
                    return Err(bad("lambda", self.lambda, "lambda >= 0"));
                }
                if !(self.n > 0.0) {
                    return Err(bad("n", self.n, "a positive power-law index"));
                }
                if !(self.a > 0.0) {
                    return Err(bad("a", self.a, "a positive Yasuda exponent"));
                }
            }
            RheologyModel::HerschelBulkley => {
                if !(self.tau0 >= 0.0) {
                    return Err(bad("tau0", self.tau0, "tau0 >= 0"));
                }
                if !(self.k > 0.0) {
                    return Err(bad("K", self.k, "a positive consistency"));
                }
                if !(self.n > 0.0) {
                    return Err(bad("n", self.n, "a positive power-law index"));
                }
                if !(self.m_reg > 0.0) || !self.m_reg.is_finite() {
                    return Err(bad(
                        "m",
                        self.m_reg,
                        "a positive finite Papanastasiou parameter (38.3)",
                    ));
                }
            }
            RheologyModel::Casson => {
                if !(self.tau0 >= 0.0) {
                    return Err(bad("tau0", self.tau0, "tau0 >= 0"));
                }
                if !(self.mu0 > 0.0) {
                    return Err(bad("muC", self.mu0, "a positive plastic viscosity"));
                }
                if !(self.m_reg > 0.0) || !self.m_reg.is_finite() {
                    return Err(bad(
                        "m",
                        self.m_reg,
                        "a positive finite Papanastasiou parameter (38.3)",
                    ));
                }
            }
        }
        Ok(())
    }

    /// A one-line summary for the run banner - §38.3 requires `m`, `muMin`
    /// and `muMax` printed, and there is no reason the rest should not be.
    pub fn describe(&self) -> String {
        if self.is_newtonian() {
            return "Newtonian (SPEC-LIT 5)".to_string();
        }
        let mut s = format!("{} (SPEC-LIT 38), rho {}", self.model.name(), self.rho);
        for (key, v) in self.stated_coefficients() {
            s.push_str(&format!(", {key} {v}"));
        }
        s.push_str(&format!(
            ", muMin {} muMax {}, gammaDotFloor {}, relax {}",
            self.mu_min, self.mu_max, self.gdot_floor, self.relax
        ));
        s
    }

    /// The coefficients THIS model reads, paired with their values, in the
    /// order [`RheologyModel::coefficient_keys`] lists them.
    fn stated_coefficients(&self) -> Vec<(&'static str, Scalar)> {
        self.model
            .coefficient_keys()
            .iter()
            .map(|k| (*k, self.coefficient(k)))
            .collect()
    }

    fn coefficient(&self, key: &str) -> Scalar {
        match key {
            "K" => self.k,
            "n" => self.n,
            "mu0" => self.mu0,
            "muC" => self.mu0,
            "muInf" => self.mu_inf,
            "lambda" => self.lambda,
            "a" => self.a,
            "tau0" => self.tau0,
            "m" => self.m_reg,
            _ => Scalar::NAN,
        }
    }

    /// Read `viscosityModel` and its `rheology { ... }` block out of an
    /// OpenFOAM `constant/physicalProperties`, under §13.4's contract.
    ///
    /// `viscosityModel` has been written into every case `blockgen` generates
    /// since that file existed, and was read by NOTHING before §38. That is
    /// the sixth instance of the defect this project keeps finding - a
    /// setting a case can express and the solver silently ignores - and it is
    /// closed here: `constant` is Newtonian, the five model names select
    /// their model, and anything else is an error naming all six.
    pub fn from_dict(d: &FoamDict, where_: &str) -> Result<Self> {
        let mut c = Self::default();

        let Some(raw) = d.get("viscosityModel") else {
            return Ok(c);
        };
        let name = raw.split_whitespace().next().unwrap_or("");
        c.model = RheologyModel::parse(&format!("{where_}: viscosityModel"), name)?;
        if c.is_newtonian() {
            // Still refuse a coefficient block that names a model's numbers
            // with no model selected: those numbers would be read by nothing.
            if d.dict_exists("rheology") {
                return Err(Error::Config(format!(
                    "{where_}: a `rheology` block is present but viscosityModel \
                     is `{name}`, so none of its coefficients would be read \
                     (SPEC-LIT 13.4). Name one of: {}",
                    RheologyModel::NAMES[1..].join(", ")
                )));
            }
            return Ok(c);
        }

        let block = format!("{where_}: rheology");
        if !d.dict_exists("rheology") {
            return Err(Error::Config(format!(
                "{block}: viscosityModel {} needs a `rheology` block giving \
                 rho and {} (SPEC-LIT 38.4, 38.7)",
                c.model.name(),
                c.model.coefficient_keys().join(", ")
            )));
        }

        let present = d.sub_keys("rheology");
        c.read_keys(&present, &block, |k| d.get(&format!("rheology/{k}")).map(str::to_string))?;
        c.validate(&block)?;
        Ok(c)
    }

    /// The half of [`Self::from_dict`] that both case formats share: check
    /// every key present is one this model reads, then read the ones it does.
    ///
    /// `fetch` returns the raw token for a key, or `None` when the case did
    /// not write it. Keeping it a closure is what lets the JSONC reader,
    /// whose values arrive as `Option<f64>` out of serde, walk exactly the
    /// same §13.4 checks as the OpenFOAM one rather than a second copy of
    /// them.
    pub fn read_keys<F>(&mut self, present: &[String], block: &str, fetch: F) -> Result<()>
    where
        F: Fn(&str) -> Option<String>,
    {
        let mine = self.model.coefficient_keys();

        // §38.7: a coefficient the named model does not use is an ERROR, not
        // a number quietly dropped. `powerLaw` with a `tau0` is a user who
        // believes they configured a yield stress.
        for k in present {
            let k = k.as_str();
            if mine.contains(&k) || COMMON_KEYS.contains(&k) {
                continue;
            }
            let mut menu: Vec<&str> = mine.to_vec();
            menu.extend_from_slice(&COMMON_KEYS);
            // `unsupported` returns Err unless `-permissive` is on, in which
            // case it warns once and hands back the fallback - and the
            // fallback here is "carry on and read the keys that ARE this
            // model's", so the `?` must NOT return early.
            unsupported::<()>(
                &format!("{block}/{k}"),
                k,
                &menu,
                "nothing - the entry is read by no model and is ignored",
                (),
            )?;
        }

        let num = |key: &str| -> Result<Option<Scalar>> {
            match fetch(key) {
                None => Ok(None),
                Some(raw) => match crate::io::dict::last_number(&raw) {
                    Some(v) => Ok(Some(v as Scalar)),
                    None => Err(Error::Config(format!(
                        "{block}/{key}: \"{raw}\" is not a number"
                    ))),
                },
            }
        };

        let required = |key: &str, v: Option<Scalar>, what: &str| -> Result<Scalar> {
            v.ok_or_else(|| {
                Error::Config(format!(
                    "{block}/{key} is missing; viscosityModel {} needs it ({what})",
                    self.model.name()
                ))
            })
        };

        // §38.4: the density is not optional and is not defaulted. It is the
        // whole difference between Pa s and m^2/s.
        self.rho = required(
            "rho",
            num("rho")?,
            "SPEC-LIT 38.4 - every coefficient below is DYNAMIC and is divided by it",
        )?;

        if let Some(v) = num("muMin")? {
            self.mu_min = v;
        }
        if let Some(v) = num("muMax")? {
            self.mu_max = v;
        }
        if let Some(v) = num("gammaDotFloor")? {
            self.gdot_floor = v;
        }
        if let Some(v) = num("relax")? {
            self.relax = v;
        }

        match self.model {
            RheologyModel::Newtonian => {}
            RheologyModel::PowerLaw => {
                self.k = required("K", num("K")?, "consistency, Pa s^n")?;
                self.n = required("n", num("n")?, "power-law index")?;
            }
            RheologyModel::Cross => {
                self.mu0 = required("mu0", num("mu0")?, "zero-shear viscosity, Pa s")?;
                self.mu_inf = num("muInf")?.unwrap_or(0.0);
                self.lambda = required("lambda", num("lambda")?, "time constant, s")?;
                self.a = required("a", num("a")?, "Cross's exponent")?;
            }
            RheologyModel::BirdCarreau => {
                self.mu0 = required("mu0", num("mu0")?, "zero-shear viscosity, Pa s")?;
                self.mu_inf = num("muInf")?.unwrap_or(0.0);
                self.lambda = required("lambda", num("lambda")?, "time constant, s")?;
                self.n = required("n", num("n")?, "power-law index")?;
                // a = 2 is Bird-Carreau proper; naming `a` makes it
                // Carreau-Yasuda. The default is printed in the banner, so it
                // is a stated choice rather than a hidden one.
                self.a = num("a")?.unwrap_or(2.0);
            }
            RheologyModel::HerschelBulkley => {
                self.tau0 = required("tau0", num("tau0")?, "yield stress, Pa")?;
                self.k = required("K", num("K")?, "consistency, Pa s^n")?;
                self.n = required("n", num("n")?, "power-law index")?;
                self.m_reg =
                    required("m", num("m")?, "Papanastasiou's regularisation, s - SPEC-LIT 38.3")?;
            }
            RheologyModel::Casson => {
                self.tau0 = required("tau0", num("tau0")?, "yield stress, Pa")?;
                self.mu0 = required("muC", num("muC")?, "Casson plastic viscosity, Pa s")?;
                self.m_reg =
                    required("m", num("m")?, "Papanastasiou's regularisation, s - SPEC-LIT 38.3")?;
            }
        }
        Ok(())
    }
}

// ==========================================================================
//  The closures, on the host - SPEC-LIT §38.2, §38.3
// ==========================================================================

/// The apparent KINEMATIC viscosity, the host twin of `rheoNu` in
/// `cuda/rheology.cu`.
///
/// Every line here has a line-for-line counterpart there; `the_device_agrees_
/// with_the_host` measures the two against each other over eight decades of
/// `gdot` and both regularisation branches. Two implementations of one
/// formula is how the two drift, so they are written to be read side by side.
pub fn apparent_viscosity(c: &KinematicCoeffs, gdot_raw: Scalar) -> Scalar {
    if c.model == RheologyModel::Newtonian {
        return c.nu0;
    }

    // §38.3's floor, applied before anything divides by `g` or raises it to a
    // power. `gdot = 0` exactly is not a corner case: it is the first
    // iteration of every run started from a uniform field.
    let g = gdot_raw.max(c.gdot_floor);

    let nu = match c.model {
        RheologyModel::Newtonian => c.nu0,

        // mu = K gdot^(n-1)
        RheologyModel::PowerLaw => c.k * g.powf(c.n - 1.0),

        // mu = mu_inf + (mu_0 - mu_inf)/(1 + (lambda gdot)^a)
        RheologyModel::Cross => {
            let d = 1.0 + (c.lambda * g).powf(c.a);
            c.nu_inf + (c.nu0 - c.nu_inf) / d
        }

        // mu = mu_inf + (mu_0 - mu_inf)[1 + (lambda gdot)^a]^((n-1)/a)
        RheologyModel::BirdCarreau => {
            let b = 1.0 + (c.lambda * g).powf(c.a);
            c.nu_inf + (c.nu0 - c.nu_inf) * b.powf((c.n - 1.0) / c.a)
        }

        // Papanastasiou in the PRODUCT form (§38.3) - the sum form
        // regularises the yield term alone and still diverges through
        // K gdot^(n-1) for n < 1.
        RheologyModel::HerschelBulkley => {
            let e = 1.0 - (-c.m_reg * g).exp();
            e * (c.t0 + c.k * g.powf(c.n)) / g
        }

        // ( sqrt(nu_c) + sqrt(t0) sqrt((1 - exp(-m gdot))/gdot) )^2
        RheologyModel::Casson => {
            let e = 1.0 - (-c.m_reg * g).exp();
            let r = c.nu0.sqrt() + c.t0.sqrt() * (e / g).sqrt();
            r * r
        }
    };

    nu.max(c.nu_min).min(c.nu_max)
}

/// The IDEAL Herschel-Bulkley apparent viscosity, `tau_0/gdot + K gdot^(n-1)`,
/// with no regularisation at all - §38.2(4).
///
/// Singular at `gdot = 0` by construction. Present so §38.8's "converges to
/// the ideal law" test has something to converge TO that is not a second copy
/// of the regularised formula, and so a reader can see exactly what the
/// regularisation is an approximation OF.
pub fn ideal_herschel_bulkley(t0: Scalar, k: Scalar, n: Scalar, gdot: Scalar) -> Scalar {
    t0 / gdot + k * gdot.powf(n - 1.0)
}

/// The NAIVE Papanastasiou form, regularising the yield term alone:
/// `K gdot^(n-1) + (tau_0/gdot)(1 - exp(-m gdot))`.
///
/// **Not used by anything.** It is here so §38.8 can MEASURE that it diverges
/// as `gdot -> 0` for `n < 1` while the product form does not - the trap
/// §38.3 names, made checkable rather than left as a warning in prose.
pub fn naive_papanastasiou(
    t0: Scalar,
    k: Scalar,
    n: Scalar,
    m: Scalar,
    gdot: Scalar,
) -> Scalar {
    k * gdot.powf(n - 1.0) + (t0 / gdot) * (1.0 - (-m * gdot).exp())
}

/// Buckingham-Reiner: the volumetric flow rate of an IDEAL Bingham plastic
/// in a circular pipe - SPEC-LIT §38.9 Gate 2.
///
/// ```text
/// Q = (pi R^4 dP)/(8 mu_p L) [ 1 - (4/3) xi + (1/3) xi^4 ] ,  xi = tau_0/tau_w
/// tau_w = dP R/(2 L)
/// ```
///
/// Chhabra & Richardson, *Non-Newtonian Flow and Applied Rheology*, 2nd ed.
/// (2008). Valid only for `tau_w > tau_0`; returns `0` otherwise, which is
/// the physics (below the wall yield stress nothing moves) and not a guard.
pub fn buckingham_reiner_q(
    radius: Scalar,
    dp_dl: Scalar,
    mu_p: Scalar,
    tau0: Scalar,
) -> Scalar {
    let tau_w = dp_dl * radius / 2.0;
    if !(tau_w > tau0) {
        return 0.0;
    }
    let xi = tau0 / tau_w;
    let pref =
        std::f64::consts::PI as Scalar * radius.powi(4) * dp_dl / (8.0 * mu_p);
    pref * (1.0 - (4.0 / 3.0) * xi + (1.0 / 3.0) * xi.powi(4))
}

/// The fully developed Herschel-Bulkley velocity profile between plates at
/// `y = 0` and `y = 2h`, driven by a uniform body force `g_x` per unit mass -
/// SPEC-LIT §38.9 Gate 1.
///
/// Derived in §38.9 from `tau(Y) = g_x Y` with `Y = |y - h|`; `t0` and `k`
/// are KINEMATIC (`tau_0/rho`, `K/rho`), because that is what §5's momentum
/// equation is written in.
///
/// `t0 = 0, n = 1, k = nu` reduces to `g_x (h² - Y²)/(2 nu)`, the parabola
/// §32.5 already uses, and `the_hb_profile_reduces_to_the_parabola` checks it.
pub fn herschel_bulkley_channel_u(
    y: Scalar,
    h: Scalar,
    g_x: Scalar,
    t0: Scalar,
    k: Scalar,
    n: Scalar,
) -> Scalar {
    let big_y = (y - h).abs();
    let y0 = if g_x > 0.0 { t0 / g_x } else { 0.0 };
    let p = (n + 1.0) / n;
    let pref = (n / (n + 1.0)) * (g_x / k).powf(1.0 / n);
    let plug = (h - y0).max(0.0).powf(p);
    if big_y <= y0 {
        pref * plug
    } else {
        pref * (plug - (big_y - y0).powf(p))
    }
}

// ==========================================================================
//  Kernels
// ==========================================================================

/// The two kernels of `cuda/rheology.cu`, plus `turbStrainRateMag` borrowed
/// from `cuda/turbulence.cu`.
///
/// The invariant kernel is deliberately NOT re-implemented here: §38.1's
/// whole point is that `sqrt(2 symm(grad U):symm(grad U))` already existed
/// and had no caller.
pub struct RheologyKernels {
    apparent: CudaFunction,
    strain_rate_boundary: CudaFunction,
    /// `turbStrainRateMag`, out of the turbulence module - §38.1.
    strain_rate: CudaFunction,
}

impl RheologyKernels {
    pub fn new(gpu: &Gpu) -> Result<Self> {
        let r = KernelSet::new(gpu, crate::kernels::RHEOLOGY)?;
        let t = KernelSet::new(gpu, crate::kernels::TURBULENCE)?;
        Ok(Self {
            apparent: r.func("rheoApparentViscosity")?,
            strain_rate_boundary: r.func("rheoStrainRateBoundary")?,
            strain_rate: t.func("turbStrainRateMag")?,
        })
    }
}

/// `gdot = sqrt(2 symm(grad U) : symm(grad U))` per cell - §38.1.
///
/// A thin wrapper over the turbulence module's own kernel, so that the one
/// invariant has one implementation.
pub fn strain_rate_mag(
    gpu: &Gpu,
    kern: &RheologyKernels,
    out: &mut DevBuf<Scalar>,
    grad_u: &DevBuf<Tensor>,
    n: usize,
) -> Result<()> {
    if n == 0 {
        return Ok(());
    }
    let nl = n as Label;
    let f = kern.strain_rate.clone();
    unsafe {
        gpu.stream()
            .launch_builder(&f)
            .arg(out)
            .arg(grad_u)
            .arg(&nl)
            .launch(cfg_for(n))?;
    }
    Ok(())
}

/// `nu <- (1 - w) nu + w mu(gdot)/rho`, elementwise - §38.2, §38.5(iv).
///
/// Runs over cells or over boundary faces alike: it reads one array and
/// writes one, and knows nothing about which.
pub fn apparent_viscosity_field(
    gpu: &Gpu,
    kern: &RheologyKernels,
    nu: &mut DevBuf<Scalar>,
    gdot: &DevBuf<Scalar>,
    c: &KinematicCoeffs,
    count: usize,
) -> Result<()> {
    if count == 0 {
        return Ok(());
    }
    let nl = count as Label;
    let model = c.model as Label;
    let f = kern.apparent.clone();
    unsafe {
        gpu.stream()
            .launch_builder(&f)
            .arg(nu)
            .arg(gdot)
            .arg(&model)
            .arg(&c.nu0)
            .arg(&c.nu_inf)
            .arg(&c.k)
            .arg(&c.n)
            .arg(&c.lambda)
            .arg(&c.a)
            .arg(&c.t0)
            .arg(&c.m_reg)
            .arg(&c.gdot_floor)
            .arg(&c.nu_min)
            .arg(&c.nu_max)
            .arg(&c.relax)
            .arg(&nl)
            .launch(cfg_for(count))?;
    }
    Ok(())
}

/// `gdot_b = |(I - n n)·Delta_b (U_b - U_P)|` on boundary faces - §38.5(iii).
///
/// `gdot_cell` is the cell field the cyclic branch falls back to.
#[allow(clippy::too_many_arguments)]
pub fn strain_rate_boundary(
    gpu: &Gpu,
    kern: &RheologyKernels,
    gdot_b: &mut DevBuf<Scalar>,
    u: &DevBuf<Vec3>,
    bu: &DevBuf<Vec3>,
    gdot_cell: &DevBuf<Scalar>,
    m: &crate::mesh::GpuMesh,
) -> Result<()> {
    let nbf = m.n_boundary_faces;
    if nbf == 0 {
        return Ok(());
    }
    let nl = nbf as Label;
    let cyclic = crate::mesh::PatchKind::Cyclic as Label;
    let f = kern.strain_rate_boundary.clone();
    unsafe {
        gpu.stream()
            .launch_builder(&f)
            .arg(gdot_b)
            .arg(u)
            .arg(bu)
            .arg(gdot_cell)
            .arg(&m.b_sf)
            .arg(&m.b_mag_sf)
            .arg(&m.b_delta_coeffs)
            .arg(&m.b_face_cells)
            .arg(&m.b_kind)
            .arg(&cyclic)
            .arg(&nl)
            .launch(cfg_for(nbf))?;
    }
    Ok(())
}

// ==========================================================================
//  Tests
// ==========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn kin(model: RheologyModel) -> KinematicCoeffs {
        KinematicCoeffs {
            model,
            nu0: 1.0,
            nu_inf: 0.0,
            k: 1.0,
            n: 1.0,
            lambda: 0.0,
            a: 2.0,
            t0: 0.0,
            m_reg: 0.0,
            gdot_floor: DEFAULT_GDOT_FLOOR,
            nu_min: 0.0,
            nu_max: Scalar::INFINITY,
            relax: 1.0,
        }
    }

    /// SPEC-LIT §38.2: `n = 1, K = mu` is the Newtonian fluid, for EVERY
    /// `gdot`. If this fails the exponent convention is wrong.
    #[test]
    fn the_power_law_at_n_one_is_newtonian() {
        let mu: Scalar = 3.7e-4;
        let c = KinematicCoeffs { k: mu, n: 1.0, ..kin(RheologyModel::PowerLaw) };
        for g in [0.0 as Scalar, 1e-8, 1e-3, 1.0, 17.0, 1e4, 1e8] {
            let v = apparent_viscosity(&c, g);
            assert!(
                (v - mu).abs() <= 1e-15 * mu,
                "powerLaw n=1 gave {v} at gdot = {g}, not {mu}"
            );
        }
    }

    /// Each of the four remaining models has a parameter setting that IS the
    /// Newtonian fluid. All four must return a constant.
    #[test]
    fn every_model_has_a_newtonian_reduction() {
        let mu: Scalar = 1.7e-3;
        let cases: Vec<(&str, KinematicCoeffs)> = vec![
            (
                "Cross with mu0 = muInf",
                KinematicCoeffs {
                    nu0: mu,
                    nu_inf: mu,
                    lambda: 0.4,
                    a: 0.7,
                    ..kin(RheologyModel::Cross)
                },
            ),
            (
                "BirdCarreau with n = 1",
                KinematicCoeffs {
                    nu0: mu,
                    nu_inf: 0.0,
                    lambda: 8.2,
                    n: 1.0,
                    a: 2.0,
                    ..kin(RheologyModel::BirdCarreau)
                },
            ),
            (
                "HerschelBulkley with tau0 = 0, n = 1 and m gdot >> 1",
                KinematicCoeffs {
                    t0: 0.0,
                    k: mu,
                    n: 1.0,
                    m_reg: 1e12,
                    ..kin(RheologyModel::HerschelBulkley)
                },
            ),
            (
                "Casson with tau0 = 0",
                KinematicCoeffs {
                    t0: 0.0,
                    nu0: mu,
                    m_reg: 1e3,
                    ..kin(RheologyModel::Casson)
                },
            ),
        ];

        for (what, c) in cases {
            for g in [1e-4 as Scalar, 1e-1, 1.0, 1e2, 1e5] {
                let v = apparent_viscosity(&c, g);
                assert!(
                    (v - mu).abs() <= 1e-9 * mu,
                    "{what}: gave {v} at gdot = {g}, not {mu}"
                );
            }
        }
    }

    /// Shear thinning is the whole point of three of the five. `mu` must fall
    /// strictly with `gdot` over twelve decades.
    #[test]
    fn shear_thinning_models_are_monotone() {
        let cases: Vec<(&str, KinematicCoeffs)> = vec![
            (
                "powerLaw n = 0.6",
                KinematicCoeffs { k: 0.35, n: 0.6, ..kin(RheologyModel::PowerLaw) },
            ),
            (
                "Cross",
                KinematicCoeffs {
                    nu0: 1.6e-4,
                    nu_inf: 3.5e-6,
                    lambda: 1.0,
                    a: 0.7,
                    ..kin(RheologyModel::Cross)
                },
            ),
            (
                "BirdCarreau",
                KinematicCoeffs {
                    nu0: 1.6e-4,
                    nu_inf: 3.5e-6,
                    lambda: 8.2,
                    n: 0.21,
                    a: 2.0,
                    ..kin(RheologyModel::BirdCarreau)
                },
            ),
        ];

        for (what, c) in cases {
            let mut prev = apparent_viscosity(&c, 1e-6);
            let mut g: Scalar = 1e-6;
            while g < 1e6 {
                g *= 2.0;
                let v = apparent_viscosity(&c, g);
                assert!(v < prev, "{what} is not monotone at gdot = {g}: {v} >= {prev}");
                prev = v;
            }
        }
    }

    /// The two plateaux are what distinguish Cross and Bird-Carreau from a
    /// bare power law, and a case reading `mu0` expects to get it.
    #[test]
    fn cross_and_carreau_reach_both_plateaux() {
        let (n0, ninf): (Scalar, Scalar) = (1.6e-4, 3.5e-6);
        for model in [RheologyModel::Cross, RheologyModel::BirdCarreau] {
            let c = KinematicCoeffs {
                nu0: n0,
                nu_inf: ninf,
                lambda: 8.2,
                n: 0.2128,
                a: if model == RheologyModel::Cross { 0.64 } else { 2.0 },
                // The plateaux are LIMITS, so the floor has to be well below
                // where they are being read or the floor is what is measured.
                gdot_floor: 1e-300,
                ..kin(model)
            };
            let lo = apparent_viscosity(&c, 1e-40);
            let hi = apparent_viscosity(&c, 1e40);
            assert!(
                (lo - n0).abs() <= 1e-9 * n0,
                "{:?}: gdot -> 0 gave {lo}, not mu0 = {n0}",
                model
            );
            assert!(
                (hi - ninf).abs() <= 1e-9 * ninf,
                "{:?}: gdot -> inf gave {hi}, not muInf = {ninf}",
                model
            );
        }
    }

    /// SPEC-LIT §38.3: the regularisation exists so `mu` is FINITE at
    /// `gdot = 0`, and the limit it reaches is stated, not incidental.
    #[test]
    fn the_regularisation_is_bounded_and_reaches_its_stated_limit() {
        let (t0, k, n, m): (Scalar, Scalar, Scalar, Scalar) = (2.0, 0.35, 0.6, 1000.0);

        let hb = KinematicCoeffs {
            t0,
            k,
            n,
            m_reg: m,
            gdot_floor: 1e-300,
            ..kin(RheologyModel::HerschelBulkley)
        };
        for g in [0.0 as Scalar, 1e-300, 1e-12, 1.0, 1e12, 1e300] {
            let v = apparent_viscosity(&hb, g);
            assert!(v.is_finite() && v >= 0.0, "HB gave {v} at gdot = {g}");
        }
        // gdot -> 0: mu -> m tau_0.
        let lim = apparent_viscosity(&hb, 1e-14);
        assert!(
            (lim - m * t0).abs() <= 1e-6 * m * t0,
            "regularised HB at gdot -> 0 gave {lim}, not m tau0 = {}",
            m * t0
        );

        let nu_c: Scalar = 3.5e-6;
        let cas = KinematicCoeffs {
            t0,
            nu0: nu_c,
            m_reg: m,
            gdot_floor: 1e-300,
            ..kin(RheologyModel::Casson)
        };
        for g in [0.0 as Scalar, 1e-300, 1e-12, 1.0, 1e12, 1e300] {
            let v = apparent_viscosity(&cas, g);
            assert!(v.is_finite() && v >= 0.0, "Casson gave {v} at gdot = {g}");
        }
        let want = (nu_c.sqrt() + (m * t0).sqrt()).powi(2);
        let got = apparent_viscosity(&cas, 1e-14);
        assert!(
            (got - want).abs() <= 1e-6 * want,
            "regularised Casson at gdot -> 0 gave {got}, not {want}"
        );
    }

    /// The evidence the regularisation is doing its job: as `m` rises the
    /// regularised law must approach the IDEAL Herschel-Bulkley law, and the
    /// error must fall MONOTONICALLY. A single tolerance could be tuned; a
    /// trend cannot.
    #[test]
    fn raising_m_converges_to_the_ideal_law_monotonically() {
        let (t0, k, n): (Scalar, Scalar, Scalar) = (2.0, 0.35, 0.6);
        let gdot: Scalar = 1e-3;
        let ideal = ideal_herschel_bulkley(t0, k, n, gdot);

        // The regularised law differs from the ideal one by exactly
        // `exp(-m gdot)` in relative terms, so the sweep has to keep
        // `m gdot` inside the range where that is representable: at
        // `m gdot = 100` it is already 4e-44 and the comparison measures
        // rounding rather than the model.
        let mut prev = Scalar::INFINITY;
        for m in [1e2 as Scalar, 1e3, 1e4, 3e4] {
            let c = KinematicCoeffs {
                t0,
                k,
                n,
                m_reg: m,
                ..kin(RheologyModel::HerschelBulkley)
            };
            let err = (apparent_viscosity(&c, gdot) - ideal).abs() / ideal;
            assert!(
                err < prev,
                "m = {m} gave relative error {err}, not below the previous {prev}"
            );
            prev = err;
        }
        assert!(prev < 1e-10, "m = 3e4 still {prev} from the ideal law");
    }

    /// SPEC-LIT §38.3's named trap, MEASURED rather than warned about: the
    /// naive Papanastasiou form regularises the yield term alone and still
    /// diverges through `K gdot^(n-1)` for `n < 1`. The product form does not.
    #[test]
    fn the_naive_regularisation_still_diverges_and_the_product_form_does_not() {
        let (t0, k, n, m): (Scalar, Scalar, Scalar, Scalar) = (2.0, 0.35, 0.6, 1000.0);
        let c = KinematicCoeffs {
            t0,
            k,
            n,
            m_reg: m,
            gdot_floor: 1e-300,
            ..kin(RheologyModel::HerschelBulkley)
        };

        let mut naive_prev: Scalar = 0.0;
        for g in [1e-10 as Scalar, 1e-20, 1e-30, 1e-40] {
            let naive = naive_papanastasiou(t0, k, n, m, g);
            assert!(
                naive > naive_prev,
                "the naive form did not grow as gdot fell to {g}"
            );
            naive_prev = naive;
        }
        assert!(
            naive_prev > 1e10 * m * t0,
            "the naive form is supposed to diverge; it only reached {naive_prev}"
        );

        // The product form, at the same four points, stays at its own limit.
        for g in [1e-10 as Scalar, 1e-20, 1e-30, 1e-40] {
            let v = apparent_viscosity(&c, g);
            assert!(
                v <= 1.001 * m * t0,
                "the product form reached {v} at gdot = {g}, above its m tau0 = {} limit",
                m * t0
            );
        }
    }

    /// The clip is applied to every model, and its DEFAULT is an exact no-op:
    /// `max(0, x) = x` and `min(inf, x) = x` to the last bit.
    #[test]
    fn the_default_clip_is_bitwise_a_no_op() {
        let c = KinematicCoeffs { k: 0.35, n: 0.6, ..kin(RheologyModel::PowerLaw) };
        for g in [1e-3 as Scalar, 1.0, 1e3] {
            let raw = c.k * g.max(c.gdot_floor).powf(c.n - 1.0);
            assert_eq!(
                apparent_viscosity(&c, g).to_bits(),
                raw.to_bits(),
                "the default clip moved a bit at gdot = {g}"
            );
        }
        // And a clip that BITES, bites.
        let capped = KinematicCoeffs { nu_max: 1.0, ..c };
        assert_eq!(apparent_viscosity(&capped, 1e-6), 1.0);
    }

    /// SPEC-LIT §38.9 Gate 1's reductions, which are what make the closed
    /// form trustworthy without a table to copy.
    #[test]
    fn the_hb_profile_reduces_to_the_parabola() {
        let (h, g_x, nu): (Scalar, Scalar, Scalar) = (0.02, 3.9, 1.5e-5);
        for i in 0..=40 {
            let y = 2.0 * h * (i as Scalar) / 40.0;
            let got = herschel_bulkley_channel_u(y, h, g_x, 0.0, nu, 1.0);
            let big_y = (y - h).abs();
            let want = g_x * (h * h - big_y * big_y) / (2.0 * nu);
            assert!(
                (got - want).abs() <= 1e-10 * want.abs().max(1.0),
                "at y = {y} the HB profile gave {got}, the parabola {want}"
            );
        }
    }

    /// And the power-law reduction, the other half of §38.9 Gate 1's
    /// cross-check: `tau0 = 0` must give
    /// `u = (n/(n+1))(G/k)^(1/n)[h^p - Y^p]`, `p = (n+1)/n`.
    #[test]
    fn the_hb_profile_reduces_to_the_power_law() {
        let (h, g_x, k): (Scalar, Scalar, Scalar) = (0.02, 3.9, 2.0e-5);
        for n in [0.4 as Scalar, 0.7, 1.0, 1.4] {
            let p = (n + 1.0) / n;
            let pref = (n / (n + 1.0)) * (g_x / k).powf(1.0 / n);
            for i in 0..=20 {
                let y = 2.0 * h * (i as Scalar) / 20.0;
                let big_y = (y - h).abs();
                let want = pref * (h.powf(p) - big_y.powf(p));
                let got = herschel_bulkley_channel_u(y, h, g_x, 0.0, k, n);
                assert!(
                    (got - want).abs() <= 1e-10 * want.abs().max(1.0),
                    "n = {n}, y = {y}: {got} against {want}"
                );
            }
        }
    }

    /// The plug is where the yield stress shows: inside `|Y| < y0` the
    /// profile is FLAT, and its edge is at `y0 = t0/G` exactly.
    #[test]
    fn the_hb_profile_has_a_flat_plug_of_the_derived_width() {
        let (h, g_x, t0, k, n): (Scalar, Scalar, Scalar, Scalar, Scalar) =
            (0.02, 3.9, 0.02, 2.0e-5, 1.0);
        let y0 = t0 / g_x;
        assert!(y0 < h, "the test case must actually yield");

        let centre = herschel_bulkley_channel_u(h, h, g_x, t0, k, n);
        for f in [0.0 as Scalar, 0.25, 0.5, 0.75, 0.999] {
            let u = herschel_bulkley_channel_u(h + f * y0, h, g_x, t0, k, n);
            assert!(
                (u - centre).abs() <= 1e-12 * centre,
                "the plug is not flat at Y = {} y0: {u} against {centre}",
                f
            );
        }
        // Just outside it, the profile has started to fall.
        let outside = herschel_bulkley_channel_u(h + 1.5 * y0, h, g_x, t0, k, n);
        assert!(outside < centre, "the profile did not fall outside the plug");
        // And the walls are no-slip.
        for y in [0.0 as Scalar, 2.0 * h] {
            let u = herschel_bulkley_channel_u(y, h, g_x, t0, k, n);
            assert!(u.abs() <= 1e-12 * centre, "u({y}) = {u}, not zero");
        }
    }

    /// SPEC-LIT §38.9 Gate 2: Buckingham-Reiner, checked against the NUMERICAL
    /// integral of the Bingham profile it is the closed form of, so the three
    /// bracket coefficients `1, -4/3, +1/3` are verified here rather than
    /// trusted to a recollection of a table.
    ///
    /// ```text
    /// tau(r) = (dP/dL) r/2 ,  yielded for r > r0 = 2 tau0/(dP/dL)
    /// du/dr  = -(tau - tau0)/mu_p          r > r0
    /// Q      = int_0^R 2 pi r u(r) dr
    /// ```
    #[test]
    fn buckingham_reiner_matches_the_integral_of_its_own_profile() {
        let (radius, mu_p): (Scalar, Scalar) = (0.01, 0.05);

        for xi in [0.1 as Scalar, 0.3, 0.5, 0.7, 0.9] {
            // Pick dP/dL first, then the tau0 that realises this xi.
            let dp_dl: Scalar = 4000.0;
            let tau_w = dp_dl * radius / 2.0;
            let tau0 = xi * tau_w;

            let q_closed = buckingham_reiner_q(radius, dp_dl, mu_p, tau0);

            // u(r) by integrating du/dr inward from the wall, then Q by the
            // trapezium rule on 2 pi r u(r). 400 000 points is enough for six
            // digits and this test is not in a hot loop.
            let steps = 400_000usize;
            let dr = radius / steps as Scalar;
            let r0 = 2.0 * tau0 / dp_dl;
            let u_of = |r: Scalar| -> Scalar {
                // u(r) = int_r^R (tau(s) - tau0)/mu_p ds over the yielded part
                let lo = r.max(r0);
                if lo >= radius {
                    return 0.0;
                }
                // int_lo^R ((dP/dL) s/2 - tau0)/mu_p ds
                let a = (dp_dl / 4.0) * (radius * radius - lo * lo);
                let b = tau0 * (radius - lo);
                (a - b) / mu_p
            };
            let mut q: Scalar = 0.0;
            for i in 0..steps {
                let r1 = i as Scalar * dr;
                let r2 = r1 + dr;
                let f1 = 2.0 * std::f64::consts::PI as Scalar * r1 * u_of(r1);
                let f2 = 2.0 * std::f64::consts::PI as Scalar * r2 * u_of(r2);
                q += 0.5 * (f1 + f2) * dr;
            }

            let rel = (q_closed - q).abs() / q;
            assert!(
                rel < 1e-6,
                "xi = {xi}: Buckingham-Reiner gives {q_closed}, the integral of \
                 its own profile {q}, relative difference {rel}"
            );
        }
    }

    /// And the `tau0 -> 0` collapse to Hagen-Poiseuille, which is the one
    /// number in Gate 2 that needs no integral at all.
    #[test]
    fn buckingham_reiner_collapses_to_hagen_poiseuille() {
        let (radius, mu_p, dp_dl): (Scalar, Scalar, Scalar) = (0.01, 0.05, 4000.0);
        let q = buckingham_reiner_q(radius, dp_dl, mu_p, 0.0);
        let hp = std::f64::consts::PI as Scalar * radius.powi(4) * dp_dl / (8.0 * mu_p);
        assert!((q - hp).abs() <= 1e-14 * hp, "{q} against {hp}");
        // Above the wall shear stress nothing flows.
        let tau_w = dp_dl * radius / 2.0;
        assert_eq!(buckingham_reiner_q(radius, dp_dl, mu_p, 1.01 * tau_w), 0.0);
    }

    /// SPEC-LIT §38.7. An unrecognised `viscosityModel` is an error naming
    /// all six spellings, not a silent fall back to Newtonian.
    #[test]
    fn an_unrecognised_model_is_a_13_4_error_naming_the_alternatives() {
        assert_eq!(RheologyModel::parse("x", "constant").unwrap(), RheologyModel::Newtonian);
        assert_eq!(RheologyModel::parse("x", "powerLaw").unwrap(), RheologyModel::PowerLaw);
        assert_eq!(RheologyModel::parse("x", "Casson").unwrap(), RheologyModel::Casson);

        let e = RheologyModel::parse("constant/physicalProperties: viscosityModel", "Bingham")
            .expect_err("an unrecognised model must be refused");
        let msg = e.to_string();
        for want in RheologyModel::NAMES {
            assert!(msg.contains(want), "the message does not name {want}: {msg}");
        }
        assert!(msg.contains("viscosityModel"), "{msg}");
    }

    /// SPEC-LIT §38.4: the density is what makes a dynamic coefficient
    /// kinematic, so a missing one is refused BY NAME rather than defaulted.
    #[test]
    fn a_missing_density_is_refused_by_name() {
        let src = "viscosityModel  powerLaw;\n\nrheology\n{\n    K 0.35;\n    n 0.6;\n}\n";
        let d = FoamDict::parse(src, "physicalProperties").expect("parses");
        let e = RheologyCoeffs::from_dict(&d, "constant/physicalProperties")
            .expect_err("a rheology block with no rho must be refused");
        let msg = e.to_string();
        assert!(msg.contains("rho"), "{msg}");
        assert!(msg.contains("38.4"), "the message must point at the section: {msg}");
    }

    /// SPEC-LIT §38.7: a coefficient the named model does not read is an
    /// ERROR. `powerLaw` with a `tau0` is a user who believes they have
    /// configured a yield stress, and silently dropping it is the exact
    /// defect this project keeps finding.
    #[test]
    fn a_coefficient_the_model_does_not_use_is_refused_by_name() {
        crate::io::contract::reset_warnings();
        let src = "viscosityModel  powerLaw;\n\nrheology\n{\n    rho 1000;\n    \
                   K 0.35;\n    n 0.6;\n    tau0 2;\n}\n";
        let d = FoamDict::parse(src, "physicalProperties").expect("parses");
        let e = RheologyCoeffs::from_dict(&d, "constant/physicalProperties")
            .expect_err("powerLaw does not read tau0");
        let msg = e.to_string();
        assert!(msg.contains("tau0"), "{msg}");
        assert!(msg.contains("K") && msg.contains("n"), "the menu is missing: {msg}");
    }

    /// And a `rheology` block with no model to read it is refused too - the
    /// mirror-image mistake, and just as silent if it were allowed.
    #[test]
    fn a_rheology_block_with_no_model_is_refused() {
        let src = "viscosityModel  constant;\n\nrheology\n{\n    rho 1000;\n    K 0.35;\n}\n";
        let d = FoamDict::parse(src, "physicalProperties").expect("parses");
        let e = RheologyCoeffs::from_dict(&d, "constant/physicalProperties")
            .expect_err("coefficients with no model must be refused");
        assert!(e.to_string().contains("rheology"), "{}", e);
    }

    /// A whole `physicalProperties` the way a case writes it, read end to
    /// end, and the kinematic conversion §38.4 specifies checked on the way
    /// out.
    #[test]
    fn a_power_law_block_reads_and_converts() {
        let src = "viscosityModel  powerLaw;\n\nnu [0 2 -1 0 0 0 0] 1e-05;\n\n\
                   rheology\n{\n    rho 1000;\n    K 0.35;\n    n 0.6;\n    \
                   muMax 10;\n    relax 0.5;\n}\n";
        let d = FoamDict::parse(src, "physicalProperties").expect("parses");
        let c = RheologyCoeffs::from_dict(&d, "constant/physicalProperties").expect("reads");
        assert_eq!(c.model, RheologyModel::PowerLaw);
        assert_eq!(c.rho, 1000.0);
        assert_eq!(c.k, 0.35);
        assert_eq!(c.n, 0.6);
        assert_eq!(c.relax, 0.5);

        let k = c.kinematic();
        assert_eq!(k.k, 0.35 / 1000.0);
        assert_eq!(k.nu_max, 10.0 / 1000.0);
        // And the banner says all of it.
        let s = c.describe();
        for want in ["powerLaw", "rho 1000", "K 0.35", "n 0.6", "muMax 10", "relax 0.5"] {
            assert!(s.contains(want), "the banner is missing {want}: {s}");
        }
    }

    /// A case that says nothing is Newtonian, and says so.
    #[test]
    fn no_viscosity_model_at_all_is_newtonian() {
        let d = FoamDict::parse("nu [0 2 -1 0 0 0 0] 1e-05;\n", "physicalProperties")
            .expect("parses");
        let c = RheologyCoeffs::from_dict(&d, "constant/physicalProperties").expect("reads");
        assert!(c.is_newtonian());
        assert_eq!(c.describe(), "Newtonian (SPEC-LIT 5)");
    }

    /// Every §38.7 range check, exercised. A bad number is refused at read
    /// time and not at the first NaN four thousand iterations in.
    #[test]
    fn the_parameter_ranges_are_checked_at_read_time() {
        let base = RheologyCoeffs {
            model: RheologyModel::HerschelBulkley,
            rho: 1000.0,
            tau0: 2.0,
            k: 0.35,
            n: 0.6,
            m_reg: 1000.0,
            ..RheologyCoeffs::default()
        };
        base.validate("x").expect("the base case is valid");

        let bad: Vec<(&str, RheologyCoeffs)> = vec![
            ("rho = 0", RheologyCoeffs { rho: 0.0, ..base }),
            ("n = 0", RheologyCoeffs { n: 0.0, ..base }),
            ("K = 0", RheologyCoeffs { k: 0.0, ..base }),
            ("tau0 < 0", RheologyCoeffs { tau0: -1.0, ..base }),
            ("m = 0", RheologyCoeffs { m_reg: 0.0, ..base }),
            ("relax = 0", RheologyCoeffs { relax: 0.0, ..base }),
            ("relax > 1", RheologyCoeffs { relax: 1.5, ..base }),
            ("floor = 0", RheologyCoeffs { gdot_floor: 0.0, ..base }),
            ("muMin > muMax", RheologyCoeffs { mu_min: 2.0, mu_max: 1.0, ..base }),
        ];
        for (what, c) in bad {
            assert!(c.validate("x").is_err(), "{what} was accepted");
        }
    }

    /// A Newtonian case never validates anything, because none of it is read.
    #[test]
    fn a_newtonian_case_validates_whatever_the_other_fields_hold() {
        let c = RheologyCoeffs { rho: -1.0, n: 0.0, ..RheologyCoeffs::default() };
        c.validate("x").expect("Newtonian reads none of it");
    }

    // ----------------------------------------------------------------------
    //  The device twin - SPEC-LIT §38.8
    // ----------------------------------------------------------------------

    fn gpu() -> Option<Gpu> {
        Gpu::new(0).ok()
    }

    /// `cuda/rheology.cu` hard-codes the model codes. If the enum moves, the
    /// kernel silently evaluates a different closure - the same failure mode
    /// `bc_kind_values_match_the_device` exists to stop for `BcKind`.
    #[test]
    fn model_codes_match_the_device() {
        assert_eq!(RheologyModel::Newtonian as Label, 0);
        assert_eq!(RheologyModel::PowerLaw as Label, 1);
        assert_eq!(RheologyModel::Cross as Label, 2);
        assert_eq!(RheologyModel::BirdCarreau as Label, 3);
        assert_eq!(RheologyModel::HerschelBulkley as Label, 4);
        assert_eq!(RheologyModel::Casson as Label, 5);
    }

    /// A representative parameter set per model, exercising both
    /// regularisation branches and both plateaux.
    fn every_model() -> Vec<(&'static str, KinematicCoeffs)> {
        vec![
            (
                "powerLaw",
                KinematicCoeffs { k: 3.5e-4, n: 0.6, ..kin(RheologyModel::PowerLaw) },
            ),
            (
                "Cross",
                KinematicCoeffs {
                    nu0: 1.6e-4,
                    nu_inf: 3.5e-6,
                    lambda: 1.7,
                    a: 0.7,
                    ..kin(RheologyModel::Cross)
                },
            ),
            (
                "BirdCarreau",
                KinematicCoeffs {
                    nu0: 1.6e-4,
                    nu_inf: 3.5e-6,
                    lambda: 8.2,
                    n: 0.2128,
                    a: 2.0,
                    ..kin(RheologyModel::BirdCarreau)
                },
            ),
            (
                "CarreauYasuda (a != 2)",
                KinematicCoeffs {
                    nu0: 1.6e-4,
                    nu_inf: 3.5e-6,
                    lambda: 8.2,
                    n: 0.2128,
                    a: 0.64,
                    ..kin(RheologyModel::BirdCarreau)
                },
            ),
            (
                "HerschelBulkley",
                KinematicCoeffs {
                    t0: 2.0e-3,
                    k: 3.5e-4,
                    n: 0.6,
                    m_reg: 1000.0,
                    ..kin(RheologyModel::HerschelBulkley)
                },
            ),
            (
                "Casson",
                KinematicCoeffs {
                    t0: 5.0e-6,
                    nu0: 3.5e-6,
                    m_reg: 1000.0,
                    ..kin(RheologyModel::Casson)
                },
            ),
        ]
    }

    /// SPEC-LIT §38.8's device twin. `rheoApparentViscosity` and
    /// [`apparent_viscosity`] are two implementations of one formula, and two
    /// implementations of one formula is how the two drift. Eight decades of
    /// `gdot`, `gdot = 0` included, every model, both regularisation
    /// branches.
    #[test]
    fn the_device_agrees_with_the_host() {
        let Some(g) = gpu() else { return };
        let kern = RheologyKernels::new(&g).expect("the rheology module loads");

        // `gdot = 0` first, because that is the one every run starts at.
        let mut gdot: Vec<Scalar> = vec![0.0];
        let mut x: Scalar = 1e-8;
        while x < 1e9 {
            gdot.push(x);
            x *= 3.0;
        }
        let n = gdot.len();

        for (what, c) in every_model() {
            let d_gdot = g.upload(&gdot).expect("upload");
            let mut d_nu = g.zeros::<Scalar>(n).expect("alloc");
            apparent_viscosity_field(&g, &kern, &mut d_nu, &d_gdot, &c, n).expect("launch");
            g.sync().expect("sync");
            let got = g.download(&d_nu).expect("download");

            for (i, gd) in gdot.iter().enumerate() {
                let want = apparent_viscosity(&c, *gd);
                let rel = (got[i] - want).abs() / want.abs().max(Scalar::MIN_POSITIVE);
                assert!(
                    rel < 1e-12,
                    "{what}: at gdot = {gd} the device gave {} and the host {want} \
                     (relative {rel})",
                    got[i]
                );
            }
        }
    }

    /// SPEC-LIT §38.5(iv): `w = 1` is bitwise `nu = mu(gdot)` - no extra
    /// rounding from a relaxation that is not relaxing - and `w < 1` is the
    /// stated convex combination against whatever `nu` already held.
    #[test]
    fn the_relaxation_is_the_stated_convex_combination() {
        let Some(g) = gpu() else { return };
        let kern = RheologyKernels::new(&g).expect("the rheology module loads");

        let gdot: Vec<Scalar> = vec![0.0, 1e-3, 1.0, 17.0, 1e4];
        let n = gdot.len();
        let c = KinematicCoeffs { k: 3.5e-4, n: 0.6, ..kin(RheologyModel::PowerLaw) };
        let old: Vec<Scalar> = vec![7.0e-5; n];

        // w = 1: the previous value is not read AT ALL, so two runs from
        // wildly different starting fields must agree BITWISE. Comparing
        // against the host instead would only measure how close nvcc's `pow`
        // is to Rust's, which is a different question (and is what
        // `the_device_agrees_with_the_host` asks).
        let d_gdot = g.upload(&gdot).expect("upload");
        let mut d_nu = g.upload(&old).expect("upload");
        apparent_viscosity_field(&g, &kern, &mut d_nu, &d_gdot, &c, n).expect("launch");
        g.sync().expect("sync");
        let unrelaxed = g.download(&d_nu).expect("download");

        let mut d_nu2 = g.upload(&vec![1.0e10 as Scalar; n]).expect("upload");
        apparent_viscosity_field(&g, &kern, &mut d_nu2, &d_gdot, &c, n).expect("launch");
        g.sync().expect("sync");
        let from_junk = g.download(&d_nu2).expect("download");

        for (i, gd) in gdot.iter().enumerate() {
            assert_eq!(
                unrelaxed[i].to_bits(),
                from_junk[i].to_bits(),
                "w = 1 read the previous value at gdot = {gd}"
            );
            let want = apparent_viscosity(&c, *gd);
            let rel = (unrelaxed[i] - want).abs() / want;
            assert!(rel < 1e-12, "at gdot = {gd}: {} against {want}", unrelaxed[i]);
        }

        // w = 0.3: exactly (1 - w) old + w fresh.
        let w: Scalar = 0.3;
        let cw = KinematicCoeffs { relax: w, ..c };
        let mut d_nu = g.upload(&old).expect("upload");
        apparent_viscosity_field(&g, &kern, &mut d_nu, &d_gdot, &cw, n).expect("launch");
        g.sync().expect("sync");
        let relaxed = g.download(&d_nu).expect("download");
        for (i, gd) in gdot.iter().enumerate() {
            let want = (1.0 - w) * old[i] + w * apparent_viscosity(&c, *gd);
            let rel = (relaxed[i] - want).abs() / want;
            assert!(rel < 1e-12, "at gdot = {gd}: {} against {want}", relaxed[i]);
            // And it actually relaxed: the answer is not the unrelaxed one.
            assert!(
                relaxed[i] != unrelaxed[i],
                "relax = {w} left the value unchanged at gdot = {gd}"
            );
        }
    }

    /// SPEC-LIT §38.5(iii). On a linear shear field `u = (G y, 0, 0)` with a
    /// no-slip wall at `y = 0`, the boundary strain rate is EXACTLY `G`:
    /// `Delta_b = 2/dy` and `U_P = G dy/2`, so `Delta_b |U_b - U_P| = G` with
    /// no discretisation error at all. That makes this an analytic check
    /// rather than a convergence one.
    ///
    /// The same field also checks the projector: on the `xMin`/`xMax` patches
    /// the two-point difference is entirely NORMAL, so its tangential part -
    /// and therefore `gdot_b` - must be zero.
    #[test]
    fn the_wall_strain_rate_is_the_analytic_shear_rate() {
        use crate::blockgen::{build_mesh, BlockSpec, GradedAxis};

        let Some(g) = gpu() else { return };
        let kern = RheologyKernels::new(&g).expect("the rheology module loads");

        let (ny, height): (usize, Scalar) = (8, 0.04);
        let spec = BlockSpec {
            x: GradedAxis { lo: 0.0, hi: 0.08, n: 4, ..GradedAxis::default() },
            y: GradedAxis { lo: 0.0, hi: height, n: ny, ..GradedAxis::default() },
            z: GradedAxis { lo: 0.0, hi: 0.04, n: 1, ..GradedAxis::default() },
            ..BlockSpec::default()
        };
        let hm = build_mesh(&spec).expect("build the block");
        let m = crate::mesh::GpuMesh::upload(&g, &hm).expect("upload the mesh");

        // Couette: `u = (G y, 0, 0)` everywhere, cells AND boundary faces,
        // so the two-point difference at a face is the exact derivative of
        // the field rather than the derivative plus a jump the field never
        // had. `Delta_b = 2/dy` and `U_b - U_P = G dy/2` at a y wall, so
        // `gdot_b = G` with NO discretisation error.
        let shear: Scalar = 3.5;
        let u: Vec<Vec3> = (0..hm.n_cells)
            .map(|c| Vec3::new(shear * hm.c[c].y, 0.0, 0.0))
            .collect();
        let bu: Vec<Vec3> = (0..hm.n_boundary_faces)
            .map(|i| Vec3::new(shear * hm.b_cf[i].y, 0.0, 0.0))
            .collect();

        let d_u = g.upload(&u).expect("upload");
        let d_bu = g.upload(&bu).expect("upload");
        // The cell field is only read by the cyclic branch, which this block
        // has none of; fill it with a value that would be obvious if it leaked.
        let d_cell = g.upload(&vec![-1.0 as Scalar; hm.n_cells.max(1)]).expect("upload");
        let mut d_gdot = g.zeros::<Scalar>(hm.n_boundary_faces.max(1)).expect("alloc");

        strain_rate_boundary(&g, &kern, &mut d_gdot, &d_u, &d_bu, &d_cell, &m).expect("launch");
        g.sync().expect("sync");
        let got = g.download(&d_gdot).expect("download");

        let mut walls = 0usize;
        let mut normals = 0usize;
        for i in 0..hm.n_boundary_faces {
            let sf = hm.b_sf[i];
            let mag = hm.b_mag_sf[i];
            let ny_hat = sf.y / mag;
            if ny_hat.abs() > 0.99 {
                // A y wall: the shear is entirely tangential and is exactly G.
                walls += 1;
                assert!(
                    (got[i] - shear).abs() <= 1e-10 * shear,
                    "face {i} on a y wall gave gdot_b = {}, not the analytic {shear}",
                    got[i]
                );
            } else if (sf.x / mag).abs() > 0.99 {
                // An x patch: the difference is entirely NORMAL, so the
                // projector must annihilate it.
                normals += 1;
                assert!(
                    got[i].abs() <= 1e-12 * shear,
                    "face {i} on an x patch gave gdot_b = {}, not zero - the \
                     tangential projector is not removing the normal part",
                    got[i]
                );
            }
        }
        assert!(walls > 0 && normals > 0, "the block produced no faces to check");
    }
}
