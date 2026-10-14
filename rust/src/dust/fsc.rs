// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.
// Provenance: see PROVENANCE.md. No GPL-licensed source was consulted.

//! The FSC and TFC closures of the dust flame - SPEC-LIT §154 (docs/17
//! DST-03): the developing turbulent diffusivity and burning velocity, the
//! intermediately steady burning velocity, the Lagrangian time scale, their
//! k-omega mapping, the extended FSC, the TFC limit, and the laminar-limit
//! rate constant in its f32-safe form with the zero rule. Everything here
//! is the host twin of `cuda/dust.cu`: `f64` in both builds, never
//! [`crate::Scalar`], and the same operations in the same order as the
//! kernels, so the gate between them is four units in the last place.
//!
//! Written from:
//!   SPEC-LIT §154.1 - the laminar-limit source Q (154.1)
//!   SPEC-LIT §154.2 - D_t (154.2), U_t (154.3), and the series of the
//!     bracket below x = 0.5
//!   SPEC-LIT §154.3 - U_t,ISP and Da (154.4), its derived identity, and
//!     tau_L (154.5)
//!   SPEC-LIT §154.4 - (154.6) and its k-omega mapping
//!   SPEC-LIT §154.5 - the extended FSC (154.7)
//!   SPEC-LIT §154.6 - the TFC limit (154.9)
//!   SPEC-LIT §154.7 - the f32-safe rate constant (154.10), the zero rule
//!     and the K_b range; §118.11 records why K_b is formed once in f64
//!   SPEC-LIT §154.8 - the implicit split (154.11)
//!   C. Huang, A. N. Lipatnikov & K. Nessvi, J. Loss Prev. Process Ind. 67
//!     (2020) 104237, DOI 10.1016/j.jlp.2020.104237 - eqs. (3)-(7), (15)-(17)
//!   C. Huang, M. Bloching & A. N. Lipatnikov, J. Loss Prev. Process Ind. 75
//!     (2022) 104707, DOI 10.1016/j.jlp.2021.104707 - eqs. (7)-(11)
//! The specification was informed by a review of FSCDustFoam (Huang &
//! Lipatnikov, GPL) and no code was copied.
//! No GPL-licensed source was consulted.

use crate::error::{Error, Result};

/// Theta of (154.1), K: H20 §2.1, H22 §3.4 (SPEC-LIT §151.5).
pub const THETA: f64 = 20000.0;
/// A of (154.4), "the sole constant of the FSC model" (H20 §2.1).
pub const A_FSC: f64 = 0.4;
/// Zimont's TFC constant: a named, non-default input (SPEC-LIT §151.5).
pub const A_TFC_ZIMONT: f64 = 0.52;
/// C_mu and C_d of (154.6) (H20 §3.3).
pub const C_MU: f64 = 0.09;
pub const C_D: f64 = 1.0;
/// beta* of the k-omega mapping, equal to C_mu (SPEC-LIT §154.4).
pub const BETA_STAR: f64 = 0.09;
/// Below this x the bracket of (154.3) is summed as its series (§154.2).
pub const SERIES_SWITCH: f64 = 0.5;
/// The K_b range the setup admits, 1/s (§154.7).
pub const K_B_MIN: f64 = 1.0e-3;
pub const K_B_MAX: f64 = 1.0e12;
/// FLT_MIN = 2^-126 in both builds: a K below it is written as 0 (D2).
pub const K_FLOOR: f64 = f32::MIN_POSITIVE as f64;

/// The zero rule's exponent bound, ln(FLT_MIN) + 24 ln 2 = -102 ln 2
/// (= -70.70; E < FLT_MIN 2^24 = 1.97e-31), the same in both builds.
pub fn zero_exponent() -> f64 {
    -102.0 * std::f64::consts::LN_2
}

/// A refusal in the one house format.
fn refuse(who: &str, msg: String) -> Error {
    Error::Refused(format!("dust::fsc::{who}: {msg}"))
}

/// `combustion.model`: FSC (the default) or its TFC limit (154.9).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CombustionModel {
    Fsc,
    Tfc,
}

impl CombustionModel {
    /// The kernel's `model` argument.
    pub fn code(self) -> i32 {
        match self {
            CombustionModel::Fsc => 0,
            CombustionModel::Tfc => 1,
        }
    }
}

/// What a case states for the closures (SPEC-LIT §151.5).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FscInputs {
    pub model: CombustionModel,
    /// A of (154.4); [`A_FSC`] unless the case names another.
    pub a: f64,
    /// The laminar burning velocity S_L of the unburned mixture, m/s.
    pub s_l: f64,
    /// The molecular heat diffusivity of the unburned gas kappa_u, m^2/s.
    pub kappa_u: f64,
    /// The turbulent Prandtl number Pr_t.
    pub pr_t: f64,
    /// The reaction time t_r of (153.2), s.
    pub t_r: f64,
    /// The burned temperature T_b that sets theta_b = Theta/T_b, K.
    pub t_b: f64,
    /// The molecular heat diffusivity of the burned gas kappa_b, m^2/s.
    pub kappa_b: f64,
    /// t_flacc of (154.7), s; `None` is the plain FSC.
    pub t_flacc: Option<f64>,
}

/// The closures' constants, formed once on the host in f64 (§154.7,
/// §118.11) and handed to the kernels as they are.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FscConstants {
    pub model: CombustionModel,
    pub a: f64,
    /// tau_f = delta_L/S_L = kappa_u/(S_L S_L), s (154.4).
    pub tau_f: f64,
    pub pr_t: f64,
    pub c_d: f64,
    pub beta_star: f64,
    /// K_b = exp(-theta_b)/t_r, 1/s (154.10).
    pub k_b: f64,
    /// theta_b = Theta/T_b.
    pub theta_b: f64,
    pub t_b: f64,
    pub kappa_b: f64,
    /// t_flacc, s; 0 is off (the kernel's convention).
    pub t_flacc: f64,
    /// [`zero_exponent`].
    pub e_zero: f64,
}

/// A positive, finite input, or a refusal naming it.
fn positive(who: &str, name: &str, v: f64) -> Result<f64> {
    if v.is_finite() && v > 0.0 {
        Ok(v)
    } else {
        Err(refuse(who, format!("{name} = {v} must be positive and finite")))
    }
}

impl FscConstants {
    /// Forms the constants, refusing by name an input that is not positive
    /// and finite, a K_b outside [[`K_B_MIN`], [`K_B_MAX`]] (§154.7), and
    /// t_flacc with TFC, which has no clock (§154.6).
    pub fn new(inp: &FscInputs) -> Result<FscConstants> {
        const W: &str = "FscConstants::new";
        let a = positive(W, "combustion.A", inp.a)?;
        let s_l = positive(W, "S_L", inp.s_l)?;
        let kappa_u = positive(W, "kappa_u", inp.kappa_u)?;
        let pr_t = positive(W, "Pr_t", inp.pr_t)?;
        let t_r = positive(W, "t_r", inp.t_r)?;
        let t_b = positive(W, "T_b", inp.t_b)?;
        let kappa_b = positive(W, "kappa_b", inp.kappa_b)?;
        let t_flacc = match (inp.model, inp.t_flacc) {
            (_, None) => 0.0,
            (CombustionModel::Tfc, Some(_)) => {
                return Err(refuse(
                    W,
                    "combustion.tFlacc is refused with combustion.model TFC: TFC has no \
                     clock (SPEC-LIT §154.6)"
                        .into(),
                ))
            }
            (CombustionModel::Fsc, Some(t)) => positive(W, "combustion.tFlacc", t)?,
        };
        let theta_b = THETA / t_b;
        let k_b = (-theta_b).exp() / t_r;
        if !(K_B_MIN..=K_B_MAX).contains(&k_b) {
            return Err(refuse(
                W,
                format!(
                    "K_b = exp(-Theta/T_b)/t_r = {k_b:e} 1/s is outside [{K_B_MIN:e}, \
                     {K_B_MAX:e}] (SPEC-LIT §154.7): the f32-safe rate cannot hold it"
                ),
            ));
        }
        Ok(FscConstants {
            model: inp.model,
            a,
            tau_f: kappa_u / (s_l * s_l),
            pr_t,
            c_d: C_D,
            beta_star: BETA_STAR,
            k_b,
            theta_b,
            t_b,
            kappa_b,
            t_flacc,
            e_zero: zero_exponent(),
        })
    }
}

/// One cell's turbulence and temperature: T (K), k (m^2/s^2), omega (1/s)
/// and the SST's nu_t (m^2/s) of (156.4).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FscState {
    pub t: f64,
    pub k: f64,
    pub omega: f64,
    pub nu_t: f64,
}

/// One cell's closures, as the kernel writes them.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FscOut {
    pub d_t: f64,
    pub u_t: f64,
    pub u_t_isp: f64,
    pub tau_l: f64,
    /// K = K_b E/(1 + D_t/kappa_b) of (154.11), 1/s; 0 under TFC.
    pub k_rate: f64,
    /// The zero rule wrote K as 0 (FSC only); the kernel counts the cell.
    pub zeroed: bool,
}

/// (154.3)'s bracket by its series below [`SERIES_SWITCH`]:
/// sum_{j>=1} (-1)^{j+1} x^j/(j+1)!, until a term is below 1e-17 of the
/// sum, at most 25 terms (§154.2).
pub fn bracket_series(x: f64) -> f64 {
    let mut term = 0.5 * x;
    let mut sum = term;
    for j in 2..=25 {
        term *= -x / (j + 1) as f64;
        sum += term;
        if term.abs() < 1.0e-17 * sum.abs() {
            break;
        }
    }
    sum
}

/// (154.3)'s bracket by its definition, 1 - 1/x + exp(-x)/x. At x = +inf
/// (tau_L = 0, no nu_t) it is exactly 1: developed at once.
pub fn bracket_closed(x: f64) -> f64 {
    (1.0 - 1.0 / x) + (-x).exp() / x
}

/// (154.3)'s bracket T1(x) for x >= 0, as the kernel evaluates it.
pub fn bracket(x: f64) -> f64 {
    if x < SERIES_SWITCH {
        bracket_series(x)
    } else {
        bracket_closed(x)
    }
}

/// (154.10)'s bounded exponent E(T) = exp(theta_b (1 - T_b/T)) with the
/// zero rule: `None` when the exponent is below `e_zero` (or not a number).
pub fn arrhenius(c: &FscConstants, t: f64) -> Option<f64> {
    let arg = c.theta_b * (1.0 - c.t_b / t);
    if arg >= c.e_zero {
        Some(arg.exp())
    } else {
        None
    }
}

/// The flame development time t_fd = max(t - t_0, 0), counted from the end
/// of the ignition source t_0 (§154.2, §151.7 item 7).
pub fn development_time(t: f64, t_0: f64) -> f64 {
    (t - t_0).max(0.0)
}

/// The closures of one cell at the global clock (t_fd, t) - the kernel
/// `dustFsc`, operation for operation.
pub fn evaluate(c: &FscConstants, s: &FscState, t_fd: f64, t_ign: f64) -> FscOut {
    let (mut dinf, mut isp, mut tl) = (0.0f64, 0.0f64, 0.0f64);
    if s.k > 0.0 && s.omega > 0.0 {
        let u2 = (2.0 * s.k) / 3.0;
        let up = u2.sqrt();
        let l = (c.c_d * s.k.sqrt()) / (c.beta_star * s.omega);
        dinf = s.nu_t / c.pr_t;
        tl = dinf / u2;
        let da = (l / up) / c.tau_f;
        isp = (c.a * up) * da.sqrt().sqrt();
    }
    let (mut d, mut u) = (dinf, isp);
    let mut k_rate = 0.0;
    let mut zeroed = false;
    if c.model == CombustionModel::Fsc {
        let x = if t_fd > 0.0 { t_fd / tl } else { 0.0 };
        d = dinf * -(-x).exp_m1();
        u = isp * bracket(x).sqrt();
        if c.t_flacc > 0.0 && t_ign >= c.t_flacc {
            u *= (t_ign / c.t_flacc).sqrt();
        }
        match arrhenius(c, s.t) {
            None => zeroed = true,
            Some(e) => {
                k_rate = (c.k_b * e) / (1.0 + d / c.kappa_b);
                if k_rate < K_FLOOR {
                    k_rate = 0.0;
                    zeroed = true;
                }
            }
        }
    }
    FscOut { d_t: d, u_t: u, u_t_isp: isp, tau_l: tl, k_rate, zeroed }
}

/// The derived identity of §154.3,
/// U_t,ISP = A u'^{3/4} S_L^{1/2} kappa_u^{-1/4} L^{1/4}: the check on
/// (154.4), never the kernel's form.
pub fn u_t_isp_identity(a: f64, u_prime: f64, s_l: f64, kappa_u: f64, l: f64) -> f64 {
    a * (u_prime.sqrt() * u_prime.sqrt().sqrt())
        * s_l.sqrt()
        * l.sqrt().sqrt()
        / kappa_u.sqrt().sqrt()
}

/// (154.11): S_U = rho K and S_P = -rho K, the source S_U + S_P c~.
pub fn implicit_split(rho: f64, k_rate: f64) -> (f64, f64) {
    let s_u = rho * k_rate;
    // `0 - s_u`, not `-s_u`: a K of +0 gives S_P = +0, never -0.
    (s_u, 0.0 - s_u)
}

/// (154.1) as Q = rho K (1 - c~): a host diagnostic only; the device never
/// multiplies (1 - c~) (§154.8).
pub fn q_source(rho: f64, k_rate: f64, c: f64) -> f64 {
    rho * k_rate * (1.0 - c)
}

#[cfg(test)]
#[path = "tests_fsc.rs"]
mod tests;
