// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.
// Provenance: see PROVENANCE.md. No GPL-licensed source was consulted.

//! The laminar-limit eigenproblem of the dust flame: the reaction time
//! scale's `t_r = Psi^2 kappa_u / S_L^2` with Psi regenerated from its
//! definition, never read from a fit. A host `f64` two-sided shooting
//! solver finds the eigenvalue Lambda = 1/Psi^2 of the stationary planar
//! flame for the case's own state curve - the published closure or the
//! adiabatic line - and its diffusivity exponent n; the integral identity
//! checks the march and the ZFK asymptote is a diagnostic. Host only and
//! `f64` everywhere, so it is identical in both builds.
//!
//! Written from:
//!   SPEC-LIT §153 - the published closure (153.1), the reaction time
//!     (153.2), the defining eigenproblem and its ends (153.3), the
//!     integral identity (153.4), the ZFK asymptote (153.5) and the
//!     numerical method of §153.5
//!   C. Huang, A. N. Lipatnikov & K. Nessvi, J. Loss Prev. Process Ind. 67
//!     (2020) 104237, DOI 10.1016/j.jlp.2020.104237 - eq. (7) and its text
//!     p. 3-4
//!
//! No GPL-licensed source was consulted.

use crate::dust::EquivalentGas;
use crate::error::Error;

/// The flame's state curve along the progress variable (SPEC-LIT §153.1):
/// `t(c) = T(c)/T_u` and `r(c) = rho(c)/rho_u`. Either the published
/// closure (153.1) or the case's own adiabatic line of (152.8) and (152.9).
pub trait FlameCurve {
    /// (t(c), r(c)), c in [0, 1].
    fn state(&self, c: f64) -> crate::Result<(f64, f64)>;
}

/// The published closure (153.1), "the simplest form of the ideal gas state
/// equation": `t = 1 + (sigma - 1) c` and `r = 1/t`. Exact only for one
/// constant W; kept as the published-form option.
#[derive(Clone, Copy, Debug)]
pub struct PublishedCurve {
    /// rho_u/rho_b of the closure, a finite number above 1.
    pub sigma: f64,
}

impl PublishedCurve {
    /// Refuses a non-finite sigma, and a sigma at or below 1 that would make
    /// the flame not expand (SPEC-LIT §153.1).
    pub fn new(sigma: f64) -> crate::Result<PublishedCurve> {
        if !sigma.is_finite() || sigma <= 1.0 {
            return Err(Error::Config(format!(
                "the density ratio sigma = {sigma} is not a finite number > 1 \
                 (SPEC-LIT §153.1)"
            )));
        }
        Ok(PublishedCurve { sigma })
    }
}

impl FlameCurve for PublishedCurve {
    fn state(&self, c: f64) -> crate::Result<(f64, f64)> {
        let t = 1.0 + (self.sigma - 1.0) * c;
        Ok((t, 1.0 / t))
    }
}

/// The case's own adiabatic line (SPEC-LIT §153.1, the stream's default):
/// `T(c)` solves `gas.h(T, c) = gas.h_u(T_u)` ((152.8)) and
/// `r(c) = (W(c)/W_u) (T_u/T(c))` ((152.9) at fixed p).
pub struct AdiabaticCurve<'a> {
    gas: &'a EquivalentGas,
    t_u: f64,
    h0: f64,
    t_b: f64,
}

impl<'a> AdiabaticCurve<'a> {
    /// Refuses a non-finite or non-positive `t_u`, then takes `h0` from the
    /// unburned gas and `t_b` from the isobaric burned state (152.10).
    pub fn new(gas: &'a EquivalentGas, t_u: f64) -> crate::Result<AdiabaticCurve<'a>> {
        if !t_u.is_finite() || t_u <= 0.0 {
            return Err(Error::Config(format!(
                "the unburned temperature t_u = {t_u} K is not a finite positive \
                 value (SPEC-LIT §153.1)"
            )));
        }
        let h0 = gas.h_u(t_u)?;
        let t_b = gas.isobaric_burned(t_u)?.t_b;
        Ok(AdiabaticCurve { gas, t_u, h0, t_b })
    }

    /// K, the burned temperature the line ends at.
    pub fn t_b(&self) -> f64 {
        self.t_b
    }
}

impl FlameCurve for AdiabaticCurve<'_> {
    fn state(&self, c: f64) -> crate::Result<(f64, f64)> {
        if c == 0.0 {
            return Ok((1.0, 1.0));
        }
        let t = self.enthalpy_t(c)?;
        let r = (self.gas.w(c)? / self.gas.w_u()) * (self.t_u / t);
        Ok((t / self.t_u, r))
    }
}

impl AdiabaticCurve<'_> {
    /// The T on [t_u, t_b] with `gas.h(T, c) = h0` ((152.8)): Newton on the
    /// exact derivative `gas.cp`, started from `T = t_u + c (t_b - t_u)` and
    /// clamped to the interval, until |dT| <= 1e-13 T in 50 steps at most;
    /// whatever Newton misses, a bisection on the same interval settles.
    fn enthalpy_t(&self, c: f64) -> crate::Result<f64> {
        let mut t = (self.t_u + c * (self.t_b - self.t_u)).clamp(self.t_u, self.t_b);
        let mut converged = false;
        for _ in 0..50 {
            let f = self.gas.h(t, c)? - self.h0;
            let df = self.gas.cp(t, c)?;
            if !(df > 0.0) || !df.is_finite() {
                break;
            }
            let dt = f / df;
            t = (t - dt).clamp(self.t_u, self.t_b);
            if dt.abs() <= 1e-13 * t {
                converged = true;
                break;
            }
        }
        if converged {
            Ok(t)
        } else {
            self.bisect_t(c)
        }
    }

    /// The same root by bisection on [t_u, t_b], run until the midpoint
    /// equals an end (SPEC-LIT §153.1).
    fn bisect_t(&self, c: f64) -> crate::Result<f64> {
        let mut lo = self.t_u;
        let mut hi = self.t_b;
        loop {
            let mid = 0.5 * (lo + hi);
            if mid <= lo || mid >= hi {
                return Ok(0.5 * (lo + hi));
            }
            if self.gas.h(mid, c)? - self.h0 < 0.0 {
                lo = mid;
            } else {
                hi = mid;
            }
        }
    }
}

/// The march's knobs (SPEC-LIT §153.5). Each leg takes `n_steps/2` RK4
/// steps, so `n_steps` is a multiple of 4.
#[derive(Clone, Copy, Debug)]
pub struct PsiOptions {
    /// RK4 steps over both legs together (40 000).
    pub n_steps: usize,
    /// The c the cold march starts at, on the preheat solution P = c
    /// (1e-10). The eigenvalue must not depend on it below 1e-6.
    pub c_cut: f64,
    /// The s = -ln(1 - c) the hot march starts from, on the saddle manifold
    /// (36.0, 1 - c = 2.3e-16).
    pub s_max: f64,
}

impl Default for PsiOptions {
    fn default() -> PsiOptions {
        PsiOptions { n_steps: 40_000, c_cut: 1e-10, s_max: 36.0 }
    }
}

/// The eigenproblem (153.3) for one state curve, activation ratio and
/// diffusivity exponent: `(g c')' - c' + Lambda w(c) = 0` with
/// `w = r (1 - c) exp(-theta_u/t)`, `g = t^(n-1)`, `c(-inf) = 0`,
/// `c(+inf) = 1`.
pub struct PsiProblem<'a> {
    /// The case's own state curve t(c), r(c).
    pub curve: &'a dyn FlameCurve,
    /// Theta/T_u, the activation temperature over the unburned temperature.
    pub theta_u: f64,
    /// The exponent of kappa prop T^n at fixed p (n = 1 makes rho kappa
    /// constant; Sutherland mu with constant Pr gives n ~ 1.7).
    pub n: f64,
}

/// The converged eigen-solution and its diagnostics (SPEC-LIT §153.4-§153.5).
#[derive(Clone, Copy, Debug)]
pub struct PsiSolution {
    /// The eigenvalue Lambda = kappa_u/(S_L^2 t_r) = 1/Psi^2.
    pub lambda: f64,
    /// 1/sqrt(Lambda), the square root the published fit is of.
    pub psi: f64,
    /// 1/Lambda, the t_r factor kappa_u multiplies in (153.2).
    pub psi2: f64,
    /// Lambda (integral g w/P dc) - 1 of (153.4) on the march's own nodes.
    pub identity_residual: f64,
    /// The ZFK asymptote (153.5), 1/(2 integral g w dc).
    pub lambda_zfk: f64,
    /// Lambda/Lambda_ZFK, below 1 and up to it as Ze grows.
    pub ratio_zfk: f64,
    /// The Zel'dovich number Ze = (theta_u/t(1)) (1 - 1/t(1)).
    pub ze: f64,
    /// k = r(1) exp(-theta_u/t(1)), the hot-end slope of w.
    pub k: f64,
    /// g(1) = t(1)^(n-1).
    pub g1: f64,
    /// a > 0 of the saddle quadratic a^2 + a = Lambda k g(1) at Lambda.
    pub seed_slope: f64,
    /// a_m = P/(1 - c) measured on the converged hot leg at 1 - c = 1e-8.
    pub end_slope: f64,
    /// F(Lambda) = P_cold(1/2) - P_hot(1/2) at the converged Lambda.
    pub mismatch: f64,
    /// The bisections the bracket needed, 200 at most.
    pub bisections: usize,
}

/// One leg's RK4 march at a trial Lambda (SPEC-LIT §153.5): the P it
/// carries across c = 1/2, whether a stage argument or result left (0, inf),
/// and the leg's own nodes (the march variable, P) when they are wanted.
struct Leg {
    p_mid: f64,
    failed: bool,
    nodes: Vec<(f64, f64)>,
}

impl Leg {
    /// A leg that died on its first stage.
    fn failed() -> Leg {
        Leg { p_mid: f64::NAN, failed: true, nodes: Vec::new() }
    }
}

/// A stage argument inside (0, inf), or None where the leg fails (§153.5).
fn stage(q: f64) -> Option<f64> {
    if q > 0.0 && q.is_finite() {
        Some(q)
    } else {
        None
    }
}

impl PsiProblem<'_> {
    /// g(c) w(c) of (153.3) at one c: `r(c) omc exp(-theta_u/t(c))
    /// t(c)^(n-1)`, with `omc = 1 - c` passed in formed, so the hot leg can
    /// hand over omega = e^(-s) itself and never forms 1 - (1 - omega).
    fn gw(&self, c: f64, omc: f64) -> crate::Result<f64> {
        let (t, r) = self.curve.state(c)?;
        Ok(r * omc * (-self.theta_u / t).exp() * t.powf(self.n - 1.0))
    }

    /// (k, g(1)) of the hot end (§153.2): `k = r(1) exp(-theta_u/t(1))` and
    /// `g(1) = t(1)^(n-1)`, both independent of Lambda.
    fn k_g1(&self) -> crate::Result<(f64, f64)> {
        let (t1, r1) = self.curve.state(1.0)?;
        Ok((r1 * (-self.theta_u / t1).exp(), t1.powf(self.n - 1.0)))
    }

    /// a > 0 of the saddle quadratic `a^2 + a = Lambda k g(1)` (§153.2),
    /// the hot march's seed slope at this Lambda.
    fn seed_slope(&self, lam: f64, k: f64, g1: f64) -> f64 {
        (-1.0 + (1.0 + 4.0 * lam * k * g1).sqrt()) / 2.0
    }

    /// dP/du of the cold leg at u = ln c: `c (1 - Lambda g w/P)`, with
    /// `c = e^u` (§153.5). None once the stage argument P has left (0, inf).
    fn cold_rhs(&self, u: f64, p: f64, lam: f64) -> crate::Result<Option<f64>> {
        if !(p > 0.0) || !p.is_finite() {
            return Ok(None);
        }
        let c = u.exp();
        let gw = self.gw(c, 1.0 - c)?;
        Ok(Some(c * (1.0 - lam * gw / p)))
    }

    /// dP/ds of the hot leg at s = -ln(1 - c): `omega (1 - Lambda g w/P)`
    /// with `omega = e^(-s)`, omega itself passed to g w as (1 - c)
    /// (§153.5). None once the stage argument P has left (0, inf).
    fn hot_rhs(&self, s: f64, p: f64, lam: f64) -> crate::Result<Option<f64>> {
        if !(p > 0.0) || !p.is_finite() {
            return Ok(None);
        }
        let om = (-s).exp();
        let gw = self.gw(1.0 - om, om)?;
        Ok(Some(om * (1.0 - lam * gw / p)))
    }

    /// The cold leg: classical RK4 forward in u = ln c from ln c_cut, on
    /// P = c_cut, to ln 1/2 in m = n_steps/2 steps of h, at nodes
    /// u_i = u0 + i h, never accumulated (§153.5).
    fn cold_leg(&self, lam: f64, opt: &PsiOptions, want: bool) -> crate::Result<Leg> {
        let m = opt.n_steps / 2;
        let u0 = opt.c_cut.ln();
        let h = (0.5f64.ln() - u0) / m as f64;
        let mut p = opt.c_cut;
        let mut nodes = Vec::new();
        if want {
            nodes.reserve(m + 1);
            nodes.push((u0, p));
        }
        for i in 0..m {
            let u = u0 + i as f64 * h;
            let Some(k1) = self.cold_rhs(u, p, lam)? else { return Ok(Leg::failed()) };
            let Some(p2) = stage(p + 0.5 * h * k1) else { return Ok(Leg::failed()) };
            let Some(k2) = self.cold_rhs(u + 0.5 * h, p2, lam)? else {
                return Ok(Leg::failed());
            };
            let Some(p3) = stage(p + 0.5 * h * k2) else { return Ok(Leg::failed()) };
            let Some(k3) = self.cold_rhs(u + 0.5 * h, p3, lam)? else {
                return Ok(Leg::failed());
            };
            let Some(p4) = stage(p + h * k3) else { return Ok(Leg::failed()) };
            let Some(k4) = self.cold_rhs(u + h, p4, lam)? else {
                return Ok(Leg::failed());
            };
            p += h * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0;
            if !(p > 0.0) || !p.is_finite() {
                return Ok(Leg::failed());
            }
            if want {
                nodes.push((u0 + (i + 1) as f64 * h, p));
            }
        }
        Ok(Leg { p_mid: p, failed: false, nodes })
    }

    /// The hot leg: classical RK4 backward in s = -ln(1 - c) from s_max,
    /// seeded on the saddle manifold P = a e^(-s_max) with a from the
    /// quadratic at this Lambda, down to s = ln 2 in m = n_steps/2 steps
    /// of negative h, at nodes s_i = s_max + i h, never accumulated
    /// (§153.5).
    fn hot_leg(
        &self,
        lam: f64,
        k: f64,
        g1: f64,
        opt: &PsiOptions,
        want: bool,
    ) -> crate::Result<Leg> {
        let m = opt.n_steps / 2;
        let s0 = opt.s_max;
        let h = (2.0f64.ln() - s0) / m as f64;
        let a = self.seed_slope(lam, k, g1);
        let mut p = a * (-s0).exp();
        let mut nodes = Vec::new();
        if want {
            nodes.reserve(m + 1);
            nodes.push((s0, p));
        }
        for i in 0..m {
            let s = s0 + i as f64 * h;
            let Some(k1) = self.hot_rhs(s, p, lam)? else { return Ok(Leg::failed()) };
            let Some(p2) = stage(p + 0.5 * h * k1) else { return Ok(Leg::failed()) };
            let Some(k2) = self.hot_rhs(s + 0.5 * h, p2, lam)? else {
                return Ok(Leg::failed());
            };
            let Some(p3) = stage(p + 0.5 * h * k2) else { return Ok(Leg::failed()) };
            let Some(k3) = self.hot_rhs(s + 0.5 * h, p3, lam)? else {
                return Ok(Leg::failed());
            };
            let Some(p4) = stage(p + h * k3) else { return Ok(Leg::failed()) };
            let Some(k4) = self.hot_rhs(s + h, p4, lam)? else {
                return Ok(Leg::failed());
            };
            p += h * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0;
            if !(p > 0.0) || !p.is_finite() {
                return Ok(Leg::failed());
            }
            if want {
                nodes.push((s0 + (i + 1) as f64 * h, p));
            }
        }
        Ok(Leg { p_mid: p, failed: false, nodes })
    }

    /// The mismatch F(Lambda) = P_cold(1/2) - P_hot(1/2) the eigenvalue is
    /// the root of (§153.5). A failed cold leg is -inf and a failed hot leg
    /// is +inf, the directions the bracketing needs.
    fn mismatch(&self, lam: f64, opt: &PsiOptions) -> crate::Result<f64> {
        let (k, g1) = self.k_g1()?;
        let cold = self.cold_leg(lam, opt, false)?;
        if cold.failed {
            return Ok(f64::NEG_INFINITY);
        }
        let hot = self.hot_leg(lam, k, g1, opt, false)?;
        if hot.failed {
            return Ok(f64::INFINITY);
        }
        Ok(cold.p_mid - hot.p_mid)
    }

    /// The eigenvalue by bisection on ln Lambda (§153.5): the bracket
    /// [1e-6, 1e6] exp(theta_u/t(1)) is checked for a sign change of F,
    /// then halved geometrically until hi/lo - 1 <= 1e-15, the midpoint
    /// stops moving, or F(m) is exactly zero, 200 bisections at most.
    fn bisect_lambda(&self, opt: &PsiOptions) -> crate::Result<(f64, usize)> {
        let (t1, _) = self.curve.state(1.0)?;
        let mut lo = 1e-6 * (self.theta_u / t1).exp();
        let mut hi = 1e6 * (self.theta_u / t1).exp();
        if !lo.is_finite() || !hi.is_finite() {
            return Err(Error::Config(format!(
                "the bracket [1e-6, 1e6] exp(theta_u/t(1)) = [{lo:+e}, {hi:+e}] \
                 is not finite, so the eigenvalue cannot be bracketed \
                 (SPEC-LIT §153.5)"
            )));
        }
        let flo = self.mismatch(lo, opt)?;
        let fhi = self.mismatch(hi, opt)?;
        if !(flo > 0.0 && fhi < 0.0) {
            return Err(Error::Config(format!(
                "the mismatch F of §153.5 does not change sign across the \
                 bracket [1e-6, 1e6] exp(theta_u/t(1)): F(lo) = {flo:+e} and \
                 F(hi) = {fhi:+e}, so no eigenvalue is inside it \
                 (SPEC-LIT §153.5)"
            )));
        }
        let mut bisections = 0usize;
        let mut zero = None;
        while hi / lo - 1.0 > 1e-15 && bisections < 200 {
            let m = lo.sqrt() * hi.sqrt();
            if m <= lo || m >= hi {
                break;
            }
            let fm = self.mismatch(m, opt)?;
            bisections += 1;
            if fm == 0.0 {
                zero = Some(m);
                break;
            }
            if fm > 0.0 {
                lo = m;
            } else {
                hi = m;
            }
        }
        let lam = zero.unwrap_or_else(|| lo.sqrt() * hi.sqrt());
        Ok((lam, bisections))
    }
}

/// Composite Simpson over a leg's own nodes, M = len - 1 intervals of even
/// count: `|h|/3 (f0 + fM + 4 sum odd + 2 sum even interior)`, with the
/// spacing h read off the first and last node (§153.5).
fn simpson(nodes: &[(f64, f64)], f: impl Fn(&(f64, f64)) -> crate::Result<f64>) -> crate::Result<f64> {
    let last = nodes.len() - 1;
    let h = (nodes[last].0 - nodes[0].0) / last as f64;
    let mut sum = f(&nodes[0])? + f(&nodes[last])?;
    for i in 1..last {
        let w = if i % 2 == 1 { 4.0 } else { 2.0 };
        sum += w * f(&nodes[i])?;
    }
    Ok(h.abs() / 3.0 * sum)
}

impl PsiProblem<'_> {
    /// The integral identity (153.4): composite Simpson on each converged
    /// leg's own nodes, integrands `c g w/P` and `omega g w/P`, plus the
    /// analytic tail `(k g(1)/a) e^(-s_max)` beyond s_max; the residual is
    /// `Lambda (I_cold + I_hot + tail) - 1`.
    fn identity_residual(
        &self,
        cold: &Leg,
        hot: &Leg,
        lam: f64,
        k: f64,
        g1: f64,
        opt: &PsiOptions,
    ) -> crate::Result<f64> {
        let i_cold = simpson(cold.nodes.as_slice(), |&(u, p)| {
            let c = u.exp();
            Ok(c * self.gw(c, 1.0 - c)? / p)
        })?;
        let i_hot = simpson(hot.nodes.as_slice(), |&(s, p)| {
            let om = (-s).exp();
            Ok(om * self.gw(1.0 - om, om)? / p)
        })?;
        let a = self.seed_slope(lam, k, g1);
        let tail = (k * g1 / a) * (-opt.s_max).exp();
        Ok(lam * (i_cold + i_hot + tail) - 1.0)
    }

    /// The ZFK asymptote (153.5): `Lambda_ZFK = 1/(2(J_cold + J_hot))` by
    /// composite Simpson on its own 20 000 intervals per leg, u from
    /// ln c_cut to ln 1/2 and s from ln 2 to s_max.
    fn lambda_zfk(&self, opt: &PsiOptions) -> crate::Result<f64> {
        const N_ZFK: usize = 20_000;
        let mut j = 0.0;
        let u0 = opt.c_cut.ln();
        let hu = (0.5f64.ln() - u0) / N_ZFK as f64;
        let mut sum = 0.0;
        for i in 0..=N_ZFK {
            let c = (u0 + i as f64 * hu).exp();
            let f = c * self.gw(c, 1.0 - c)?;
            sum += match i {
                0 | N_ZFK => f,
                i if i % 2 == 1 => 4.0 * f,
                _ => 2.0 * f,
            };
        }
        j += hu.abs() / 3.0 * sum;
        let s0 = 2.0f64.ln();
        let hs = (opt.s_max - s0) / N_ZFK as f64;
        let mut sum = 0.0;
        for i in 0..=N_ZFK {
            let om = (-(s0 + i as f64 * hs)).exp();
            let f = om * self.gw(1.0 - om, om)?;
            sum += match i {
                0 | N_ZFK => f,
                i if i % 2 == 1 => 4.0 * f,
                _ => 2.0 * f,
            };
        }
        j += hs.abs() / 3.0 * sum;
        Ok(1.0 / (2.0 * j))
    }
}

impl<'a> PsiProblem<'a> {
    /// The eigenvalue of (153.3) by the two-sided shooting of §153.5, with
    /// the identity (153.4), the ZFK asymptote (153.5) and the saddle
    /// diagnostics of the converged march. Everything is host `f64`.
    pub fn eigen(&self, opt: &PsiOptions) -> crate::Result<PsiSolution> {
        if !self.theta_u.is_finite() || self.theta_u <= 0.0 {
            return Err(Error::Config(format!(
                "the activation ratio theta_u = {} is not a finite positive \
                 value (SPEC-LIT §153.5)",
                self.theta_u
            )));
        }
        if !self.n.is_finite() {
            return Err(Error::Config(format!(
                "the diffusivity exponent n = {} is not finite (SPEC-LIT §153.5)",
                self.n
            )));
        }
        if opt.n_steps < 8 || opt.n_steps % 4 != 0 {
            return Err(Error::Config(format!(
                "n_steps = {} is not 8 or more and a multiple of 4; each leg \
                 marches n_steps/2 RK4 steps and Simpson needs an even count \
                 (SPEC-LIT §153.5)",
                opt.n_steps
            )));
        }
        if !(opt.c_cut > 0.0) || opt.c_cut > 1e-6 {
            return Err(Error::Config(format!(
                "c_cut = {:+e} is not in (0, 1e-6] (SPEC-LIT §153.5)", opt.c_cut
            )));
        }
        if !opt.s_max.is_finite() || opt.s_max < 20.0 {
            return Err(Error::Config(format!(
                "s_max = {} is not a finite number of 20 or more (SPEC-LIT §153.5)",
                opt.s_max
            )));
        }

        let (k, g1) = self.k_g1()?;
        let (lam, bisections) = self.bisect_lambda(opt)?;
        let cold = self.cold_leg(lam, opt, true)?;
        let hot = self.hot_leg(lam, k, g1, opt, true)?;
        if cold.failed || hot.failed {
            return Err(Error::Config(format!(
                "the converged march at lambda = {lam:+e} has a failed leg, \
                 which the bracketed bisection of §153.5 excludes \
                 (SPEC-LIT §153.5)"
            )));
        }
        let identity_residual = self.identity_residual(&cold, &hot, lam, k, g1, opt)?;
        let lambda_zfk = self.lambda_zfk(opt)?;
        let (t1, _) = self.curve.state(1.0)?;
        let a = self.seed_slope(lam, k, g1);
        let s_star = 8.0 * 10.0f64.ln();
        let mut best = hot.nodes[0];
        for nd in &hot.nodes {
            if (nd.0 - s_star).abs() < (best.0 - s_star).abs() {
                best = *nd;
            }
        }
        Ok(PsiSolution {
            lambda: lam,
            psi: 1.0 / lam.sqrt(),
            psi2: 1.0 / lam,
            identity_residual,
            lambda_zfk,
            ratio_zfk: lam / lambda_zfk,
            ze: (self.theta_u / t1) * (1.0 - 1.0 / t1),
            k,
            g1,
            seed_slope: a,
            end_slope: best.1 / (-best.0).exp(),
            mismatch: self.mismatch(lam, opt)?,
            bisections,
        })
    }
}

/// One row of the documentation table over the published closure (§153.5),
/// for review only, never the setup's input.
#[derive(Clone, Copy, Debug)]
pub struct PsiRow {
    /// rho_u/rho_b of (153.1).
    pub sigma: f64,
    /// Theta/T_u.
    pub theta_u: f64,
    /// The diffusivity exponent.
    pub n: f64,
    /// The eigen-solution's Psi = 1/sqrt(Lambda).
    pub psi: f64,
    /// Lambda/Lambda_ZFK of (153.5).
    pub ratio_zfk: f64,
    /// The Zel'dovich number.
    pub ze: f64,
    /// (153.4) residual, minus 1.
    pub identity_residual: f64,
}

/// The review table over the published closure (153.1), sigma outer,
/// theta_u middle, n inner (§153.5). For review only.
pub fn documentation_table(
    sigmas: &[f64],
    theta_us: &[f64],
    ns: &[f64],
    opt: &PsiOptions,
) -> crate::Result<Vec<PsiRow>> {
    let mut rows = Vec::new();
    for &sigma in sigmas {
        let curve = PublishedCurve::new(sigma)?;
        for &theta_u in theta_us {
            for &n in ns {
                let problem = PsiProblem { curve: &curve, theta_u, n };
                let s = problem.eigen(opt)?;
                rows.push(PsiRow {
                    sigma,
                    theta_u,
                    n,
                    psi: s.psi,
                    ratio_zfk: s.ratio_zfk,
                    ze: s.ze,
                    identity_residual: s.identity_residual,
                });
            }
        }
    }
    Ok(rows)
}

/// The table as a Markdown string: the header and separator lines, then one
/// line per row.
pub fn format_table(rows: &[PsiRow]) -> String {
    let mut out = String::from(
        "| σ | Θ/T_u | n | Ψ | Λ/Λ_ZFK | Ze | (153.4) − 1 |\n\
         |---|---|---|---|---|---|---|\n",
    );
    for r in rows {
        out.push_str(&format!(
            "| {} | {} | {} | {:e} | {} | {} | {:e} |\n",
            r.sigma, r.theta_u, r.n, r.psi, r.ratio_zfk, r.ze, r.identity_residual
        ));
    }
    out
}

/// The reaction time scale (153.2): `t_r = Psi^2 kappa_u / S_L^2`. Refuses
/// any input that is not a finite positive number.
pub fn reaction_time(psi: f64, kappa_u: f64, s_l: f64) -> crate::Result<f64> {
    for (nm, v) in [("psi", psi), ("kappa_u", kappa_u), ("s_l", s_l)] {
        if !v.is_finite() || v <= 0.0 {
            return Err(Error::Config(format!(
                "the reaction-time input {nm} = {v:+e} is not a finite positive \
                 value (SPEC-LIT §153.2)"
            )));
        }
    }
    Ok(psi * psi * kappa_u / (s_l * s_l))
}

/// The speed a user-supplied Psi_used propagates the laminar limit at,
/// `S_prop/S_L = Psi_true/Psi_used` (§153.5). A diagnostic: it refuses no
/// Psi_used for being far from Psi_true, only an input that is not a finite
/// positive number.
pub fn implied_speed_ratio(psi_true: f64, psi_used: f64) -> crate::Result<f64> {
    for (nm, v) in [("psi_true", psi_true), ("psi_used", psi_used)] {
        if !v.is_finite() || v <= 0.0 {
            return Err(Error::Config(format!(
                "the speed-ratio input {nm} = {v:+e} is not a finite positive \
                 value (SPEC-LIT §153.5)"
            )));
        }
    }
    Ok(psi_true / psi_used)
}

#[cfg(test)]
#[path = "tests_psi.rs"]
mod tests;
