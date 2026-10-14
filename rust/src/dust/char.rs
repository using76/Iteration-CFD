// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.
// Provenance: see PROVENANCE.md. No GPL-licensed source was consulted.

//! The char surface reaction of a Phase B parcel - SPEC-LIT §159.2 and
//! §159.8 (docs/17 DST-13). `C(s) + O2 -> CO2` at the kinetic-diffusion
//! limited rate (159.5): the diffusion rate R_diff = C_0 T_m^(3/4)/d and
//! the kinetic rate R_kin = A exp(-E_a/(R0 T_g)) combine harmonically. The
//! particle shrinks at constant density, and over a segment with T_p, T_g
//! and p_O2 frozen the shrinkage has the closed form (159.10), so the d^2
//! law and the linear shrinking of (159.6) are its two limits. The heat
//! of the reaction is split by the retention factor h_s: h_s of it heats
//! the particle and (1 - h_s) goes to the gas.
//!
//! Like `particle`, this is the host twin of `cuda/dustparticle.cu`: `f64`
//! in both builds, and the operations in the kernel's order. The char mass
//! is rounded to [`crate::Scalar`] where it is stored, so the carbon given
//! is the difference of two stored words and the ledger telescopes.
//!
//! Written from:
//!   SPEC-LIT §159.2 - (159.5) the char rate and the heat retention h_s
//!   SPEC-LIT §159.3 - (159.6) the d^2 law and the linear shrinking
//!   SPEC-LIT §159.8 - the exact shrinkage (159.10), the burnout time and
//!     the element ratios (159.11), derived there
//!   A. Islas, A. Rodriguez Fernandez, C. Betegon, E. Martinez-Paneda,
//!     A. Pandal, Computational assessment of biomass dust explosions in the
//!     20L sphere, arXiv:2207.13149 (2022) - eqs. (20)-(23)
//!   IUPAC standard atomic weights (conventional values): C 12.011,
//!     O 15.999 g/mol
//! The specification was informed by a review of FSCDustFoam (Huang &
//! Lipatnikov, GPL) and no code was copied.
//! No GPL-licensed source was consulted.

use crate::dust::particle::R0;
use crate::error::{Error, Result};
use std::f64::consts::PI;

/// Carbon, kg/mol (IUPAC conventional atomic weight).
pub const W_C: f64 = 12.011e-3;
/// O2, kg/mol (2 x 15.999 g/mol).
pub const W_O2: f64 = 31.998e-3;
/// CO2, kg/mol: W_C + W_O2, so that C + O2 = CO2 holds by mass.
pub const W_CO2: f64 = W_C + W_O2;

fn refuse(who: &str, msg: String) -> Error {
    Error::Refused(format!("dust::char::{who}: {msg}"))
}

/// What a case states for the char reaction (159.5).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CharInputs {
    /// C_0 of R_diff, kg/(m s Pa K^0.75).
    pub c0: f64,
    /// A of R_kin, kg/(m2 s Pa).
    pub a: f64,
    /// E_a of R_kin, J/mol.
    pub e_a: f64,
    /// The heat released per kg of carbon burnt, J/kg (ISL22: 32.9 MJ/kg).
    pub h_c: f64,
    /// The retention factor h_s in [0, 1] (ISL22: 0.3).
    pub h_s: f64,
}

impl CharInputs {
    /// No char reaction: the char is inert, as DST-12 left it.
    #[must_use]
    pub fn inert() -> Self {
        Self { c0: 0.0, a: 0.0, e_a: 0.0, h_c: 0.0, h_s: 0.0 }
    }
}

/// [`CharInputs`] checked, with what the kernel takes: theta = E_a/R0, the
/// gas share 1 - h_s and the element ratios of (159.11).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CharProps {
    pub c0: f64,
    pub a: f64,
    pub theta: f64,
    pub h_c: f64,
    pub h_s: f64,
    pub gas_share: f64,
    pub o2_per_c: f64,
    pub co2_per_c: f64,
}

impl CharProps {
    pub fn new(i: &CharInputs) -> Result<Self> {
        for (name, v) in [("c0", i.c0), ("a", i.a), ("e_a", i.e_a)] {
            if !(v.is_finite() && v >= 0.0) {
                return Err(refuse("CharProps::new", format!("{name} = {v} must be finite and >= 0")));
            }
        }
        if !i.h_c.is_finite() {
            return Err(refuse("CharProps::new", format!("h_c = {} must be finite", i.h_c)));
        }
        if !(i.h_s >= 0.0 && i.h_s <= 1.0) {
            return Err(refuse("CharProps::new", format!("h_s = {} must lie in [0, 1]", i.h_s)));
        }
        Ok(Self {
            c0: i.c0,
            a: i.a,
            theta: i.e_a / R0,
            h_c: i.h_c,
            h_s: i.h_s,
            gas_share: 1.0 - i.h_s,
            o2_per_c: W_O2 / W_C,
            co2_per_c: W_CO2 / W_C,
        })
    }
}

/// C' = R_diff d = C_0 [0.5 (T_g + T_p)]^(3/4) of (159.5), kg/(m s Pa).
#[must_use]
pub fn diff_coef(c: &CharProps, t_g: f64, t_p: f64) -> f64 {
    c.c0 * (0.5 * (t_g + t_p)).powf(0.75)
}

/// R_kin = A exp(-E_a/(R0 T_g)) of (159.5), kg/(m2 s Pa).
#[must_use]
pub fn r_kin(c: &CharProps, t_g: f64) -> f64 {
    c.a * (-c.theta / t_g).exp()
}

/// The mass loss rate -dm_p/dt of (159.5), kg/s, at diameter `d`.
#[must_use]
pub fn rate(c: &CharProps, d: f64, p_o2: f64, t_g: f64, t_p: f64) -> f64 {
    let rd = diff_coef(c, t_g, t_p) / d;
    let rk = r_kin(c, t_g);
    if !(rd + rk > 0.0) {
        return 0.0;
    }
    PI * d * d * p_o2 * (rd * rk / (rd + rk))
}

/// (159.10): the shrinkage Delta = d_0 - d(t) after `t` seconds at frozen
/// C', R_kin and p_O2, at constant density rho_p:
/// `Delta = 2 K t/(B + sqrt(B^2 - 2 R_kin K t))`, K = 2 p_O2 C' R_kin/rho_p,
/// B = C' + R_kin d_0. Valid up to the burnout time of the caller.
#[must_use]
pub fn shrink(cd: f64, rk: f64, p_o2: f64, rho_p: f64, d0: f64, t: f64) -> f64 {
    let kk = 2.0 * p_o2 * cd * rk / rho_p;
    let bb = cd + rk * d0;
    2.0 * kk * t / (bb + (bb * bb - 2.0 * rk * kk * t).max(0.0).sqrt())
}

/// What one char segment does to the solid (§159.8), before the energy.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Burn {
    /// The segment's length, s; 0 for a remnant that leaves at once.
    pub tseg: f64,
    /// The char is gone at the segment's end.
    pub out: bool,
    /// The char mass after the segment, as the state stores it.
    pub m_c: f64,
    /// The diameter after the segment, at constant density.
    pub d: f64,
    /// The carbon burnt: the char before less the stored char after.
    pub dm: f64,
}

/// One char segment of at most `rem` seconds for a particle of char `m_c`
/// and ash `m_a` (no moisture or volatile left) at diameter `d0` and
/// temperature `t_p`, in gas at `t_g` with O2 partial pressure `p_o2`.
/// `None` when nothing burns: no char, no oxygen, or a rate that is zero.
#[allow(clippy::too_many_arguments)]
#[must_use]
pub fn burn(c: &CharProps, m_c: f64, m_a: f64, d0: f64, t_p: f64, t_g: f64, p_o2: f64, rem: f64) -> Option<Burn> {
    if !(m_c > 0.0 && p_o2 > 0.0) {
        return None;
    }
    let m0 = m_c + m_a;
    let rho = 6.0 * m0 / (PI * d0 * d0 * d0);
    let cd = diff_coef(c, t_g, t_p);
    let rk = r_kin(c, t_g);
    let kk = 2.0 * p_o2 * cd * rk / rho;
    if !(kk > 0.0) {
        return None;
    }
    let d_end = d0 * (m_a / m0).cbrt();
    let dmax = d0 - d_end;
    let t_out = dmax * (cd + rk * 0.5 * (d0 + d_end)) / kk;
    if !(t_out > 0.0) {
        // A remnant too small to shrink the diameter leaves at once.
        return Some(Burn { tseg: 0.0, out: true, m_c: 0.0, d: d_end, dm: m_c });
    }
    if t_out <= rem {
        return Some(Burn { tseg: t_out, out: true, m_c: 0.0, d: d_end, dm: m_c });
    }
    let sh = shrink(cd, rk, p_o2, rho, d0, rem);
    let d1 = d0 - sh;
    let dm_raw = m0 * sh * (d0 * d0 + d0 * d1 + d1 * d1) / (d0 * d0 * d0);
    #[allow(clippy::unnecessary_cast)]
    let mc1 = (m_c - dm_raw).max(0.0) as crate::Scalar as f64;
    let d = d0 * ((mc1 + m_a) / m0).cbrt();
    Some(Burn { tseg: rem, out: false, m_c: mc1, d, dm: m_c - mc1 })
}

#[cfg(test)]
#[path = "tests_char.rs"]
mod tests;
