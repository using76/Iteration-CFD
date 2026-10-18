// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.
// Provenance: see PROVENANCE.md. No GPL-licensed source was consulted.

//! The host thin-flame vessel of Gate DST-VES (SPEC-LIT §157.6): the 0-D
//! closed-adiabatic-vessel relations of SPEC-LIT §157.2 integrated in time
//! with the SAME `S_u(p)` and `gamma_u` the 3-D run carries. Host `f64` in
//! BOTH builds - never [`crate::Scalar`] - so the two builds read one trace.
//!
//! Written from:
//!   SPEC-LIT (157.2) - the burned mass fraction `x_b = (p - p0)/(pe - p0)`
//!   SPEC-LIT (157.3) - the flame radius, BKM24 equation 6.10
//!   SPEC-LIT (157.4) - the thin-flame pressure equation `dp/dt`
//!   SPEC-LIT (157.5) - `(dp/dt)_max` and `K_st`, the cube-root law
//!   SPEC-LIT §157.9 (157.9)-(157.10) - the test burning law, and the
//!     end-step and half-time quadrature
//!   SPEC-LIT §155.7 (155.10) - the isentropic compression
//!     `T_u(p)/T_u0 = (p/p0)^((g-1)/g)` and the density ratio the (157.4)
//!     factor `d` carries
//!   Bechtold, Krishnan & Matalon, J. Fluid Mech. 998 (2024) A59,
//!     DOI 10.1017/jfm.2024.919 (CC BY 4.0), equation 6.10
//!
//! The specification was informed by a review of FSCDustFoam (Huang &
//! Lipatnikov, GPL) and no code was copied.
//! No GPL-licensed source was consulted.

use crate::chem::thermo::R_GAS;
use crate::dust::thermo::EquivalentGas;
use crate::error::{Error, Result};

/// The prescribed burning law of Gate DST-VES (SPEC-LIT §157.6 (i)): a test
/// power law in the compressed unburned state,
/// `S_u(p) = s0 (T_u(p)/T_u0)^alpha (p/p0)^beta` with the temperature ratio
/// of (155.10) - a test input, not a correlation claim.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BurningLaw {
    /// m/s, the burning velocity at the initial state.
    pub s0: f64,
    /// The exponent of the unburned-temperature ratio.
    pub alpha: f64,
    /// The exponent of the pressure ratio.
    pub beta: f64,
}

impl BurningLaw {
    /// The constant law: `alpha = beta = 0`.
    pub fn constant(s0: f64) -> Self {
        Self { s0, alpha: 0.0, beta: 0.0 }
    }

    /// Every input a run reads, each refusal naming the input: `s0` finite
    /// and above zero, `alpha` and `beta` finite.
    pub fn validate(&self) -> Result<()> {
        const W: &str = "dust::vessel0d::BurningLaw::validate";
        if !(self.s0.is_finite() && self.s0 > 0.0) {
            return Err(Error::Refused(format!(
                "{W}: s0 = {} is not a finite positive burning velocity",
                self.s0
            )));
        }
        if !self.alpha.is_finite() {
            return Err(Error::Refused(format!(
                "{W}: alpha = {} is not a finite exponent",
                self.alpha
            )));
        }
        if !self.beta.is_finite() {
            return Err(Error::Refused(format!(
                "{W}: beta = {} is not a finite exponent",
                self.beta
            )));
        }
        Ok(())
    }

    /// `S_u(p) = s0 (T_u(p)/T_u0)^alpha (p/p0)^beta` with
    /// `T_u(p)/T_u0 = (p/p0)^((g-1)/g)` of (155.10). EXACT operation order:
    /// the two ratios are formed once each and the three factors multiply
    /// left to right, so the trace and the 3-D run read one value.
    pub fn s_u(&self, p: f64, p0: f64, gamma_u: f64) -> f64 {
        let pr = p / p0;
        let tr = pr.powf((gamma_u - 1.0) / gamma_u);
        self.s0 * tr.powf(self.alpha) * pr.powf(self.beta)
    }
}

/// Which density the propagation reads (SPEC-LIT §157.6): the (155.10)
/// compressed unburned gas of the run, or the initial one - the negative
/// control of INSIGHTS (g), which exists to MISS `(dp/dt)_max`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum UnburnedDensity {
    /// `rho_u0 (p/p0)^(1/gamma_u)`, the run's own (155.10) density.
    #[default]
    Compressed,
    /// `rho_u0`, held at the initial state.
    Initial,
}

/// The closed adiabatic vessel of (157.2)-(157.5): radius `R`, the initial
/// and AICC-end pressures and the unburned `gamma_u` the compression reads.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vessel {
    /// m, the vessel radius.
    pub radius: f64,
    /// Pa, the initial pressure `p0`.
    pub p0: f64,
    /// Pa, the AICC end pressure `p_e`.
    pub pe: f64,
    /// The unburned `gamma_u` at `T_u0`, constant along the compression.
    pub gamma_u: f64,
}

impl Vessel {
    /// The vessel, every input validated by name: finite, `radius` and `p0`
    /// above zero, `pe` above `p0`, `gamma_u` above 1.
    pub fn new(radius: f64, p0: f64, pe: f64, gamma_u: f64) -> Result<Vessel> {
        const W: &str = "dust::vessel0d::Vessel::new";
        let bad = |name: &str, v: f64, why: &str| {
            Error::Refused(format!("{W}: {name} = {v} is {why}"))
        };
        for (name, v) in
            [("radius", radius), ("p0", p0), ("pe", pe), ("gamma_u", gamma_u)]
        {
            if !v.is_finite() {
                return Err(bad(name, v, "not a finite number"));
            }
        }
        if radius <= 0.0 {
            return Err(bad("radius", radius, "not a positive radius"));
        }
        if p0 <= 0.0 {
            return Err(bad("p0", p0, "not a positive pressure"));
        }
        if pe <= p0 {
            return Err(bad("pe", pe, "not above p0"));
        }
        if gamma_u <= 1.0 {
            return Err(bad("gamma_u", gamma_u, "not above 1"));
        }
        Ok(Vessel { radius, p0, pe, gamma_u })
    }

    /// The vessel from the case's own thermo (SPEC-LIT §157.6): `pe = p0 *
    /// aicc(t_u0).p_ratio` (§152.11) and `gamma_u = cp_u/(cp_u - R0/W_u)`
    /// at `t_u0` - the SAME formula as `DustSetup::new` (src/dust/source.rs).
    pub fn from_gas(
        gas: &EquivalentGas,
        t_u0: f64,
        p0: f64,
        radius: f64,
    ) -> Result<Vessel> {
        const W: &str = "dust::vessel0d::Vessel::from_gas";
        let aicc = gas
            .aicc(t_u0)
            .map_err(|e| Error::Refused(format!("{W}: the AICC state: {e}")))?;
        let cp_u = gas
            .cp_u(t_u0)
            .map_err(|e| Error::Refused(format!("{W}: the unburned cp: {e}")))?;
        let gamma_u = cp_u / (cp_u - R_GAS / gas.w_u());
        Vessel::new(radius, p0, p0 * aicc.p_ratio, gamma_u)
    }

    /// m3, `4 pi R^3 / 3`.
    pub fn volume(&self) -> f64 {
        4.0 * std::f64::consts::PI * self.radius.powi(3) / 3.0
    }

    /// (157.2): the burned mass fraction `x_b = (p - p0)/(pe - p0)`.
    pub fn burned_fraction(&self, p: f64) -> f64 {
        (p - self.p0) / (self.pe - self.p0)
    }

    /// The (157.3) bracket `1 - ((pe - p)/(pe - p0)) (p0/p)^(1/gamma_u)`,
    /// floored at 0 and NOT capped: above `pe` it grows past 1, which is
    /// what makes a RK4 stage that stepped over the end smooth.
    pub fn bracket(&self, p: f64) -> f64 {
        let b = 1.0
            - ((self.pe - p) / (self.pe - self.p0))
                * (self.p0 / p).powf(1.0 / self.gamma_u);
        b.max(0.0)
    }

    /// (157.3), BKM24 equation 6.10: the flame radius `R bracket^(1/3)`.
    pub fn flame_radius(&self, p: f64) -> f64 {
        self.radius * self.bracket(p).cbrt()
    }

    /// (157.4): the thin-flame `dp/dt`,
    /// `3 (pe - p0)/R bracket^(2/3) d S_u(p)` with `d = (p/p0)^(1/gamma_u)`
    /// the (155.10) density ratio ([`UnburnedDensity::Compressed`]) or 1
    /// (the negative control, [`UnburnedDensity::Initial`]).
    pub fn rate(&self, law: &BurningLaw, mode: UnburnedDensity, p: f64) -> f64 {
        let d = match mode {
            UnburnedDensity::Compressed => (p / self.p0).powf(1.0 / self.gamma_u),
            UnburnedDensity::Initial => 1.0,
        };
        3.0 * (self.pe - self.p0) / self.radius
            * self.bracket(p).powf(2.0 / 3.0)
            * d
            * law.s_u(p, self.p0, self.gamma_u)
    }

    /// (157.5): `(dp/dt)_max = 3 (pe-p0)/R (pe/p0)^(1/g) S_u(pe)` - for an
    /// `S_u` rising with `p` the rate is largest at the end, where the
    /// bracket reaches 1.
    pub fn dpdt_max_closed(&self, law: &BurningLaw) -> f64 {
        3.0 * (self.pe - self.p0) / self.radius
            * (self.pe / self.p0).powf(1.0 / self.gamma_u)
            * law.s_u(self.pe, self.p0, self.gamma_u)
    }

    /// (157.5): `K_st = 3 (4pi/3)^(1/3) (pe-p0) (pe/p0)^(1/g) S_u(pe)` -
    /// the volume-independent form, which is what makes the cube-root law
    /// exact for a thin flame.
    pub fn k_st_closed(&self, law: &BurningLaw) -> f64 {
        3.0 * (4.0 * std::f64::consts::PI / 3.0).cbrt()
            * (self.pe - self.p0)
            * (self.pe / self.p0).powf(1.0 / self.gamma_u)
            * law.s_u(self.pe, self.p0, self.gamma_u)
    }

    /// The cubic root of the same law: `K_st = (dp/dt)_max V^(1/3)`.
    pub fn k_st(&self, dpdt_max: f64) -> f64 {
        dpdt_max * self.volume().cbrt()
    }
}

/// The trace of one integration: time, pressure and `dp/dt` at every point,
/// the pressure rising from `p_start` to `pe`.
#[derive(Debug, Clone, Default)]
pub struct ThinFlameTrace {
    pub t: Vec<f64>,
    pub p: Vec<f64>,
    pub dpdt: Vec<f64>,
}

/// Classical RK4 in time at the fixed step `h`, from `(t_start, p_start)`
/// to `pe` (SPEC-LIT §157.6): the state is `p` alone, `dp/dt` the (157.4)
/// rate at the current pressure. The final point lands ON `pe`, its time
/// the composite Simpson quadrature of the last partial step. Refusals, by
/// name: `p_start` outside `(p0, pe)`, `h` not a finite positive time, a
/// law that fails [`BurningLaw::validate`], and 10_000_000 steps taken
/// without reaching `pe`.
pub fn integrate(
    v: &Vessel,
    law: &BurningLaw,
    mode: UnburnedDensity,
    t_start: f64,
    p_start: f64,
    h: f64,
) -> Result<ThinFlameTrace> {
    const W: &str = "dust::vessel0d::integrate";
    law.validate()?;
    if !(p_start > v.p0 && p_start < v.pe) {
        return Err(Error::Refused(format!(
            "{W}: p_start = {p_start} is not between p0 = {} and pe = {}",
            v.p0, v.pe
        )));
    }
    if !(h.is_finite() && h > 0.0) {
        return Err(Error::Refused(format!(
            "{W}: h = {h} is not a finite positive time step"
        )));
    }
    let r = |p: f64| v.rate(law, mode, p);
    let (mut t, mut p) = (t_start, p_start);
    let mut tr = ThinFlameTrace::default();
    tr.t.push(t);
    tr.p.push(p);
    tr.dpdt.push(r(p));
    for _ in 0..10_000_000 {
        let k1 = r(p);
        let k2 = r(p + h / 2.0 * k1);
        let k3 = r(p + h / 2.0 * k2);
        let k4 = r(p + h * k3);
        let p_new = p + h / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4);
        if p_new >= v.pe {
            tr.t.push(t + simpson_time(v, law, mode, p, v.pe));
            tr.p.push(v.pe);
            tr.dpdt.push(r(v.pe));
            return Ok(tr);
        }
        t += h;
        p = p_new;
        tr.t.push(t);
        tr.p.push(p);
        tr.dpdt.push(r(p));
    }
    Err(Error::Refused(format!(
        "{W}: steps = 10000000 taken without reaching pe = {} from p_start = \
         {p_start} at h = {h}; the step is too small for the run",
        v.pe
    )))
}

/// The three numbers Gate DST-VES holds the run against: the time to
/// `(p0 + pe)/2`, the end time and pressure, and the largest `dp/dt` the
/// trace carried.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ThinFlameMetrics {
    /// s, the time to the half pressure.
    pub t_half: f64,
    /// s, the time of the trace's last point.
    pub t_end: f64,
    /// Pa, the pressure of the trace's last point.
    pub p_end: f64,
    /// Pa/s, the largest `dp/dt` of the trace.
    pub dpdt_max: f64,
}

/// The metrics of one trace: `t_half` the time the trace's own points
/// bracket, closed with the Simpson quadrature from the last point below
/// `(p0 + pe)/2`; `t_end` and `p_end` its last point; `dpdt_max` the
/// largest `dp/dt` it carried. Refusals, by name: an empty `tr`, and a
/// trace whose first pressure is already at or past the half pressure.
pub fn metrics(
    v: &Vessel,
    law: &BurningLaw,
    mode: UnburnedDensity,
    tr: &ThinFlameTrace,
) -> Result<ThinFlameMetrics> {
    const W: &str = "dust::vessel0d::metrics";
    if tr.p.is_empty() {
        return Err(Error::Refused(format!("{W}: tr is an empty trace")));
    }
    let p_half = (v.p0 + v.pe) / 2.0;
    if tr.p[0] >= p_half {
        return Err(Error::Refused(format!(
            "{W}: tr.p[0] = {} is not below the half pressure p_half = {p_half}",
            tr.p[0]
        )));
    }
    let mut k = 0;
    for (i, &p) in tr.p.iter().enumerate() {
        if p < p_half {
            k = i;
        }
    }
    Ok(ThinFlameMetrics {
        t_half: tr.t[k] + simpson_time(v, law, mode, tr.p[k], p_half),
        t_end: tr.t[tr.t.len() - 1],
        p_end: tr.p[tr.p.len() - 1],
        dpdt_max: tr.dpdt.iter().copied().fold(f64::NEG_INFINITY, f64::max),
    })
}

/// The time `1/rate` integrates over, `[pa, pb]` - composite Simpson at 64
/// intervals, the quadrature the last partial step and `t_half` close with.
pub fn simpson_time(v: &Vessel, law: &BurningLaw, mode: UnburnedDensity, pa: f64, pb: f64) -> f64 {
    const N: usize = 64;
    let h = (pb - pa) / N as f64;
    let f = |p: f64| 1.0 / v.rate(law, mode, p);
    let mut s = f(pa) + f(pb);
    for i in 1..N {
        let w = if i % 2 == 1 { 4.0 } else { 2.0 };
        s += w * f(pa + i as f64 * h);
    }
    s * h / 3.0
}

#[cfg(test)]
#[path = "tests_vessel0d.rs"]
mod tests;
