// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.
// Provenance: see PROVENANCE.md. No GPL-licensed source was consulted.

//! The flame-Courant number of the dust flame - SPEC-LIT §155.6 (docs/17
//! DST-04). The explicit propagation term `rho_u U_t |grad c~|` moves the
//! c~ front relative to the fluid at `w = rho_u U_t / rho`, which grows to
//! sigma U_t in the burned gas, so every step also satisfies, per cell,
//!
//! ```text
//! Co_f = dt (|u~| + rho_u U_t / rho) / dx  <=  Co_f,max ,   dx = V^{1/3} ,  Co_f,max = 0.5   (155.9)
//! ```
//!
//! A fixed dt is checked once against the PREDICTED number, built from
//! `sigma U_t,ISP` of the initial field (t_fd -> infinity bounds U_t by
//! U_t,ISP), and refused by name when it exceeds the bound. The measured
//! number is the kernel `dustFlameCourant` ([`crate::dust::device::
//! flame_courant`]); [`flame_courant_host`] is its host twin.
//!
//! Written from:
//!   SPEC-LIT §155.6 - (155.9), the prediction and the refusal
//!   R. Courant, K. Friedrichs & H. Lewy, Math. Ann. 100 (1928) 32-74,
//!     DOI 10.1007/BF01448839 - the CFL condition
//! The specification was informed by a review of FSCDustFoam (Huang &
//! Lipatnikov, GPL) and no code was copied.
//! No GPL-licensed source was consulted.

use crate::error::{Error, Result};

/// The default bound Co_f,max of (155.9).
pub const CO_F_MAX: f64 = 0.5;

/// The PREDICTED flame-Courant number of a fixed step (155.9):
/// `dt (u_max + sigma U_t,ISP,max) / dx_min`, the burned-gas propagation
/// speed sigma U_t,ISP bounding rho_u U_t/rho everywhere.
pub fn predicted(dt: f64, dx_min: f64, u_max: f64, sigma: f64, u_t_isp_max: f64) -> f64 {
    dt * (u_max + sigma * u_t_isp_max) / dx_min
}

/// Refuses by name a fixed `deltaT` whose predicted Co_f exceeds `co_max`
/// (SPEC-LIT §155.6); returns the predicted value otherwise.
pub fn check_fixed_step(dt: f64, predicted_co: f64, co_max: f64) -> Result<f64> {
    if !(co_max.is_finite() && co_max > 0.0) {
        return Err(Error::Refused(format!(
            "dust::courant: combustion.maxFlameCourant = {co_max} must be positive and finite \
             (SPEC-LIT §155.6)"
        )));
    }
    if !(predicted_co.is_finite() && predicted_co <= co_max) {
        return Err(Error::Refused(format!(
            "dust::courant: deltaT = {dt:e} s gives a predicted flame-Courant number \
             Co_f = {predicted_co:.4} (from sigma U_t,ISP of the initial field), above \
             combustion.maxFlameCourant = {co_max} (SPEC-LIT §155.6, (155.9)); lower deltaT"
        )));
    }
    Ok(predicted_co)
}

/// The host twin of `dustFlameCourant` for one cell: `|u|`, rho, U_t and V
/// as the device holds them, widened to f64; the same operations in order.
pub fn flame_courant_host(u: [f64; 3], rho: f64, u_t: f64, v: f64, rho_u: f64, dt: f64) -> f64 {
    let mag = ((u[0] * u[0] + u[1] * u[1]) + u[2] * u[2]).sqrt();
    let w = (rho_u * u_t) / rho;
    (dt * (mag + w)) / v.cbrt()
}
