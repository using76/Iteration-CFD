// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.
// Provenance: see PROVENANCE.md. No GPL-licensed source was consulted.

//! The constant-pressure adiabatic zero-dimensional reactor (SPEC-LIT
//! §132.4): one cell whose state is `y = (Y_1..Y_K, T)`, at constant `p`
//! and constant `h`, advanced by equations (132.7)-(132.9) - SAND89-8009
//! Chapter I's isobaric example ("DTDT = -SUM/(RHO*CPB)") written out -
//! with the Rosenbrock integrator of [`crate::chem::rosenbrock`] as the
//! stiff stepper. Adiabatic and isobaric, the ODE conserves `h` exactly,
//! so an advanced state lands on the `(h, p)` equilibrium of
//! [`crate::chem::equilibrium`]. Host only, and f64 in every build, so the
//! module behaves identically in the default build and in
//! `--features single`.
//!
//! No GPL-licensed source was consulted.

use crate::chem::kinetics::production_rates;
use crate::chem::mechanism::Mechanism;
use crate::chem::rosenbrock::{integrate_observed, Method, RosOptions, RosStats};
use crate::chem::thermo::R_GAS;
use crate::error::Error;

/// The tail of every §132.8 refusal.
const TAG: &str = "(SPEC-LIT §132.8)";

/// The controls of one [`ConstPressureReactor::advance`] call (SPEC-LIT
/// §132.4 and §132.8's R7-R11).
#[derive(Clone, Debug, PartialEq)]
pub struct ReactorOptions {
    /// Which coefficient set runs; [`Method::Rodas3`] by default.
    pub method: Method,
    /// The relative tolerance handed to the integrator.
    pub rtol: f64,
    /// The absolute tolerance of every mass fraction.
    pub atol_y: f64,
    /// The absolute tolerance of the temperature, K.
    pub atol_t: f64,
    /// The initial step, s.
    pub h0: f64,
    /// The upper bound of the step, s; +inf means unbounded.
    pub h_max: f64,
    /// The most step attempts one call may make.
    pub max_steps: usize,
    /// The temperature rise that defines ignition, K (R11).
    pub ignition_rise: f64,
}

impl Default for ReactorOptions {
    fn default() -> Self {
        ReactorOptions {
            method: Method::Rodas3,
            rtol: 1e-8,
            atol_y: 1e-14,
            atol_t: 1e-8,
            h0: 1e-10,
            h_max: f64::INFINITY,
            max_steps: 1_000_000,
            ignition_rise: 400.0,
        }
    }
}

/// What one [`ConstPressureReactor::advance`] returns (SPEC-LIT §132.4).
#[derive(Clone, Debug, PartialEq)]
pub struct ReactorReport {
    /// The final mass fractions, in `mech.species()` order.
    pub y: Vec<f64>,
    /// The final temperature, K.
    pub temperature: f64,
    /// R11's interpolated time at which `T` first reached
    /// `temperature0 + ignition_rise`, or `None` when it never did.
    pub ignition_delay: Option<f64>,
    /// R10's count of mass fractions clipped from `[-atol_y, 0)` to 0.
    pub clipped: usize,
    /// The integrator's counters for the call.
    pub stats: RosStats,
}

/// The one-cell constant-pressure adiabatic reactor of SPEC-LIT §132.4 at
/// the fixed pressure `p`.
#[derive(Clone, Debug)]
pub struct ConstPressureReactor<'a> {
    mech: &'a Mechanism,
    p: f64,
}

impl<'a> ConstPressureReactor<'a> {
    /// Builds the reactor at `p` (R7).
    pub fn new(mech: &'a Mechanism, p: f64) -> crate::Result<Self> {
        if !p.is_finite() || p <= 0.0 {
            return Err(Error::Config(format!(
                "reactor pressure p = {p} Pa {}",
                TAG
            )));
        }
        Ok(ConstPressureReactor { mech, p })
    }

    /// The fixed pressure, Pa.
    pub fn pressure(&self) -> f64 {
        self.p
    }

    /// (132.7)-(132.9) at the state `y = (Y_1..Y_K, T)`, with
    /// `K = mech.species().len()`: the mass fractions' rates from (130.1)'s
    /// molar production and the temperature's rate from the enthalpy
    /// balance, with the density from the ideal-gas law (132.9) and the
    /// clip of §130.7 - `Yp_k = max(y_k, 0)` feeds every evaluation, and
    /// the stored `y` is not touched. Thermo and kinetics errors pass
    /// through unchanged, which is how a `T` outside the thermo range is
    /// refused (§132.7 R5).
    pub fn rhs(&self, y: &[f64], dydt: &mut [f64]) -> crate::Result<()> {
        let k = self.mech.species().len();
        if y.len() != k + 1 || dydt.len() != k + 1 {
            return Err(Error::Config(format!(
                "the reactor state has {} components, the mechanism needs {} {}",
                y.len(),
                k + 1,
                TAG
            )));
        }
        let t = y[k];
        if !t.is_finite() || t <= 0.0 {
            return Err(Error::Config(format!(
                "reactor temperature T = {t} K {}",
                TAG
            )));
        }
        let sp = self.mech.thermo().species();
        let yp: Vec<f64> = y[..k].iter().map(|v| v.max(0.0)).collect();
        // S = sum_k Yp_k / W_k; rho = p / (R T S) is (132.9) with
        // W_mix = 1/S.
        let s: f64 = yp
            .iter()
            .zip(sp)
            .map(|(yk, sk)| yk / sk.molar_mass)
            .sum();
        if s.is_nan() || s <= 0.0 {
            return Err(Error::Config(format!("no species is present {}", TAG)));
        }
        let rho = self.p / (R_GAS * t * s);
        let c: Vec<f64> = yp
            .iter()
            .zip(sp)
            .map(|(yk, sk)| rho * yk / sk.molar_mass)
            .collect();
        let omega = production_rates(self.mech, t, &c)?;
        let cp = self.mech.thermo().cp_mass(t, &yp)?;
        for l in 0..k {
            dydt[l] = sp[l].molar_mass * omega[l] / rho;
        }
        let mut sum = 0.0;
        for l in 0..k {
            if omega[l] != 0.0 {
                sum += sp[l].h(t)? * omega[l];
            }
        }
        dydt[k] = -sum / (rho * cp);
        Ok(())
    }

    /// Integrates from time 0 to `duration` from the state
    /// `(y0, temperature0)`: `y0` holds `K` mass fractions (R9's checks)
    /// and the run advances by (132.7)-(132.9) with R10's per-step
    /// refusal-or-clip and R11's ignition delay (SPEC-LIT §132.4). The
    /// integrator's own refusals (§132.7 R5) and `rhs`'s errors pass
    /// through unchanged.
    pub fn advance(
        &self,
        y0: &[f64],
        temperature0: f64,
        duration: f64,
        opts: &ReactorOptions,
    ) -> crate::Result<ReactorReport> {
        let k = self.mech.species().len();
        if y0.len() != k {
            return Err(Error::Config(format!(
                "the reactor was given {} mass fractions, the mechanism has {} species {}",
                y0.len(),
                k,
                TAG
            )));
        }
        for (name, &v) in self.mech.species().iter().zip(y0) {
            if !v.is_finite() || v < 0.0 {
                return Err(Error::Config(format!(
                    "species {name}: the initial mass fraction {v} {}",
                    TAG
                )));
            }
        }
        if !temperature0.is_finite() || temperature0 <= 0.0 {
            return Err(Error::Config(format!(
                "initial temperature T0 = {temperature0} K {}",
                TAG
            )));
        }
        if !duration.is_finite() || duration < 0.0 {
            return Err(Error::Config(format!("duration = {duration} s {}", TAG)));
        }
        if !opts.ignition_rise.is_finite() || opts.ignition_rise <= 0.0 {
            return Err(Error::Config(format!(
                "ignition_rise = {} K {}",
                opts.ignition_rise, TAG
            )));
        }
        if duration == 0.0 {
            return Ok(ReactorReport {
                y: y0.to_vec(),
                temperature: temperature0,
                ignition_delay: None,
                clipped: 0,
                stats: RosStats::default(),
            });
        }
        let mut y: Vec<f64> = y0.to_vec();
        y.push(temperature0);
        let mut atol = vec![opts.atol_y; k];
        atol.push(opts.atol_t);
        let ros = RosOptions {
            method: opts.method,
            rtol: opts.rtol,
            atol,
            h0: opts.h0,
            h_min: 0.0,
            h_max: opts.h_max,
            max_steps: opts.max_steps,
        };
        let names = self.mech.species();
        // R11: T* and the previous accepted point, starting at
        // (0.0, temperature0).
        let t_star = temperature0 + opts.ignition_rise;
        let mut clipped = 0usize;
        let mut t_a = 0.0f64;
        let mut temp_a = temperature0;
        let mut delay: Option<f64> = None;
        let mut observe = |t: f64, yy: &mut [f64]| -> crate::Result<()> {
            for l in 0..k {
                if yy[l] < -opts.atol_y {
                    return Err(Error::Config(format!(
                        "species {}: Y = {:e} is below -atol_y = {:e} at t = {:e} s {}",
                        names[l], yy[l], opts.atol_y, t, TAG
                    )));
                }
                if yy[l] < 0.0 {
                    yy[l] = 0.0;
                    clipped += 1;
                }
            }
            // R11: the first crossing, linearly interpolated; T_a < T*
            // always holds while `delay` is still `None`.
            if delay.is_none() && yy[k] >= t_star {
                delay = Some(t_a + (t_star - temp_a) * (t - t_a) / (yy[k] - temp_a));
            }
            t_a = t;
            temp_a = yy[k];
            Ok(())
        };
        let mut f = |yy: &[f64], dydt: &mut [f64]| -> crate::Result<()> { self.rhs(yy, dydt) };
        let stats = integrate_observed(&mut f, &mut y, 0.0, duration, &ros, &mut observe)?;
        let temperature = y[k];
        y.pop();
        Ok(ReactorReport {
            y,
            temperature,
            ignition_delay: delay,
            clipped,
            stats,
        })
    }
}
