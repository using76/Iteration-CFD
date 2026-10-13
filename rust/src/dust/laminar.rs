// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.
// Provenance: see PROVENANCE.md. No GPL-licensed source was consulted.

//! The laminar burning velocity of the corn-starch dust: the H22 fit
//! S_L(phi) and its optional Gülder temperature and pressure factors.
//! Host only and `f64` everywhere.
//!
//! Written from:
//!   SPEC-LIT §152.8 - the fit (152.13) and the factor form (152.14), with
//!     its default off
//!   C. Huang, M. Bloching & A. N. Lipatnikov, J. Loss Prev. Process Ind. 75
//!     (2022) 104707, DOI 10.1016/j.jlp.2021.104707 - eq. 3, the Gülder-form
//!     fit to the Dahoe et al. 2002 data, with its four coefficients
//! No GPL-licensed source was consulted.

use crate::error::Error;

/// The temperature and pressure factors of (152.14). The default is
/// [`SlFactors::Off`], which is exactly the H22 fit (SPEC-LIT §152.8).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum SlFactors {
    /// (152.13) alone: S_L(phi, T_u, p) = S_L(phi) bit for bit.
    Off,
    /// The Gülder factors (152.14): (T_u/t_ref)^alpha (p/p_ref)^beta. Non-zero
    /// alpha, beta are a case input, named in the case and refused by every
    /// gate (SPEC-LIT §152.8).
    Gulder {
        /// The temperature exponent.
        alpha: f64,
        /// The pressure exponent.
        beta: f64,
        /// K, the reference temperature.
        t_ref: f64,
        /// Pa, the reference pressure.
        p_ref: f64,
    },
}

/// S_L(phi) of the dust: the four coefficients of (152.13) and the factors
/// option (SPEC-LIT §152.8).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LaminarSpeed {
    /// m/s, the prefactor W (0.2145 for H22).
    pub w: f64,
    /// The phi exponent eta (-0.2774).
    pub eta: f64,
    /// The gaussian curvature xi, 39.1832.
    pub xi: f64,
    /// The peak's phi, sigma_phi in (152.13) (0.6).
    pub sigma_phi: f64,
    /// The temperature and pressure factors, default off.
    pub factors: SlFactors,
}

impl LaminarSpeed {
    /// The H22 fit to the Dahoe et al. 2002 data: W = 0.2145 m/s,
    /// eta = -0.2774, xi = 39.1832, sigma_phi = 0.6, factors off
    /// (SPEC-LIT §152.8).
    pub fn huang2022() -> LaminarSpeed {
        LaminarSpeed {
            w: 0.2145,
            eta: -0.2774,
            xi: 39.1832,
            sigma_phi: 0.6,
            factors: SlFactors::Off,
        }
    }

    /// S_L(phi) in m/s, (152.13): W phi^eta exp[-xi (phi - sigma_phi)^2].
    /// Refused: a phi that is not a finite number above zero.
    pub fn s_l_phi(&self, phi: f64) -> crate::Result<f64> {
        if !phi.is_finite() || phi <= 0.0 {
            return Err(Error::Config(format!(
                "the equivalence ratio phi = {phi} is not a finite number above zero; \
                 only lean or stoichiometric charges burn, 0 < phi <= 1 (SPEC-LIT §152.8)"
            )));
        }
        let shift = phi - self.sigma_phi;
        Ok(self.w * phi.powf(self.eta) * (-(self.xi * shift * shift)).exp())
    }

    /// S_L(phi, T_u, p) in m/s, (152.14). With the factors off this is
    /// [`Self::s_l_phi`] bit for bit. With the Gülder factors it multiplies
    /// (T_u/t_ref)^alpha (p/p_ref)^beta. Refused: a state temperature or
    /// pressure that is not a finite positive value; a non-finite alpha or
    /// beta, or a reference temperature or pressure that is not a finite
    /// positive value.
    pub fn s_l(&self, phi: f64, t_u: f64, p: f64) -> crate::Result<f64> {
        for (nm, v) in [("t_u", t_u), ("p", p)] {
            if !v.is_finite() || v <= 0.0 {
                return Err(Error::Config(format!(
                    "the state S_L is evaluated at needs {nm} = {v}, which is not a \
                     finite positive value (SPEC-LIT §152.8)"
                )));
            }
        }
        match self.factors {
            SlFactors::Off => self.s_l_phi(phi),
            SlFactors::Gulder { alpha, beta, t_ref, p_ref } => {
                if !alpha.is_finite() || !beta.is_finite() {
                    return Err(Error::Config(format!(
                        "the Gülder factors alpha = {alpha}, beta = {beta} are not both \
                         finite (SPEC-LIT §152.8)"
                    )));
                }
                for (nm, v) in [("t_ref", t_ref), ("p_ref", p_ref)] {
                    if !v.is_finite() || v <= 0.0 {
                        return Err(Error::Config(format!(
                            "the Gülder factor of (152.14) needs {nm} = {v}, which is not \
                             a finite positive value (SPEC-LIT §152.8)"
                        )));
                    }
                }
                Ok(self.s_l_phi(phi)? * (t_u / t_ref).powf(alpha)
                    * (p / p_ref).powf(beta))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use crate as ofgpu;

    include!("../../tests/shared/dust_laminar_cases.rs");
}
