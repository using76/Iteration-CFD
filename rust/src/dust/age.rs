// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.
// Provenance: see PROVENANCE.md. No GPL-licensed source was consulted.

//! The local flame-age clock of the dust flame as an option of the c~
//! equation - SPEC-LIT §160 (docs/17 DST-20; RESEARCH D.69-D.70). A
//! transported age concentration alpha of the burned gas,
//!
//! ```text
//! d(rho alpha)/dt + div(rho u~ alpha) = div[rho (kappa + D_t) grad alpha]
//!                                     + rho c~ + a wdot_c ,   a = alpha/max(c~, c_a)
//! t_fd(x, t) := a   (replaces the global clock in (154.2)-(154.3))   (160.1)
//! ```
//!
//! selected by `combustion.flameAge: local`; the default `global` is the
//! papers' clock of (154.2) and adds nothing to a step. The uniform
//! solution a = t - t_ig of §160.2 is what the discretisation is held
//! against, and the clock arithmetic of (160.3)-(160.6) lives here:
//! [`clock_increment`] and [`clock_weights`] form the step's increment and
//! the source weights, [`predict_cell`] and [`sources_cell`] are the host
//! twins of the device kernels `dustAgePredict`/`dustAgeSources`
//! ([`crate::dust::device`], f64 operations in the kernels' order), and
//! [`AgeState`] is the state
//! [`crate::dust::progress::ProgressEquation`] drives.
//!
//! Written from:
//!   SPEC-LIT §160.1 - the age equation (160.1) and its floor c_a
//!   SPEC-LIT §160.2 - the uniform-clock solution and its leading-edge
//!     caveat
//!   SPEC-LIT §160.5 - the clock's origin (D-20a), the increment (160.3),
//!     the predicted level (160.4) and the sources (160.5)-(160.6)
//! The specification was informed by a review of FSCDustFoam (Huang &
//! Lipatnikov, GPL) and no code was copied.
//! No GPL-licensed source was consulted.

use crate::device::DevBuf;
use crate::field::GpuScalarField;
use crate::solver::SolverPerformance;
use crate::Scalar;

/// The floor c_a of (160.1): the age of a cell is alpha/max(c~, c_a), so a
/// cell still to burn holds age 0 instead of a division by zero.
pub const C_A: f64 = 1.0e-6;

/// `combustion.flameAge`: the papers' global clock (the default, the value
/// `dustFsc` is given) or the transported local clock (160.1). Turning it
/// on changes the numerics of a validated path, so it is the user's call
/// per case, never a default (SPEC-LIT §160.1, docs/17 DD8).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum FlameAge {
    #[default]
    Global,
    Local,
}

impl FlameAge {
    /// The case's string: `"global"` or `"local"`, else `None` - any other
    /// string is refused by name (SPEC-LIT §160.5).
    pub fn parse(s: &str) -> Option<FlameAge> {
        match s {
            "global" => Some(FlameAge::Global),
            "local" => Some(FlameAge::Local),
            _ => None,
        }
    }
}

/// (160.3): the global clock's increment over the step, `t_fd^g(t_new) -
/// t_fd^g(t_old)` with t_fd^g = max(t - t_origin, 0) of (154.2). t_old is
/// the previous step's t_new; the first step's t_old is t_new - dt, dt the
/// c~ equation's step.
pub fn clock_increment(t_old: f64, t_new: f64, t_origin: f64) -> f64 {
    crate::dust::fsc::development_time(t_new, t_origin)
        - crate::dust::fsc::development_time(t_old, t_origin)
}

/// (160.5)'s weights (w0, w00) of the scheme's coefficients (a_n, a_0,
/// a_00): the clock source S_alpha = w0 rho^n c~^n + w00 rho^{n-1}
/// c~^{n-1} keeps the uniform clock exact under any scheme of §13
/// (SPEC-LIT §160.5).
pub fn clock_weights(c: crate::timescheme::DdtCoeffs, delta: f64, delta_prev: f64) -> (f64, f64) {
    (-f64::from(c.a_0) * delta, -f64::from(c.a_00) * (delta + delta_prev))
}

/// (160.4) per cell - the host twin of the kernel `dustAgePredict`, the
/// kernel's operations in its order: the PREDICTED level alpha* = alpha^n +
/// Delta_n c~^n and the closures' age a* = alpha*/max(c~^n, c_a), each
/// rounded once at its store.
pub fn predict_cell(alpha0: Scalar, c0: Scalar, delta: f64, c_a: f64) -> (Scalar, Scalar) {
    let v = f64::from(alpha0) + delta * f64::from(c0);
    let s = v as Scalar;
    let a = f64::from(s) / f64::from(c0).max(c_a);
    (s, a as Scalar)
}

/// (160.5)-(160.6) per cell - the host twin of the kernel `dustAgeSources`,
/// the kernel's operations in its order: the clock source S_alpha = w0
/// rho^n c~^n + w00 rho^{n-1} c~^{n-1}, and the inheritance coefficient
/// S_P,a = (S_P' c~^+ - S_U)/max(c~, c_a) with c~^+ = max(c~, 0), whose
/// V-fold `fvm_sp` puts on the alpha diagonal.
#[allow(clippy::too_many_arguments)]
pub fn sources_cell(
    su: Scalar,
    sp: Scalar,
    c: Scalar,
    rho0: Scalar,
    c0: Scalar,
    rho00: Scalar,
    c00: Scalar,
    w0: f64,
    w00: f64,
    c_a: f64,
) -> (Scalar, Scalar) {
    let sua =
        w0 * (f64::from(rho0) * f64::from(c0)) + w00 * (f64::from(rho00) * f64::from(c00));
    let ci = f64::from(c);
    let spa = (f64::from(sp) * ci.max(0.0) - f64::from(su)) / ci.max(c_a);
    (sua as Scalar, spa as Scalar)
}

/// (160.1)'s a = alpha/max(c~, c_a): the local age a cell holds, the t_fd
/// that replaces the global clock.
pub fn local_age(alpha: f64, c: f64, c_a: f64) -> f64 {
    alpha / c.max(c_a)
}

/// The age option's state, owned by
/// [`crate::dust::progress::ProgressEquation`] once the local clock is on
/// ([`FlameAge::Local`]): the alpha field with c~'s patch kinds at zero
/// values (so a fixedValue c~ patch gives alpha 0: inflow enters with age
/// 0), the closures' age a* of (160.4) held over the step's outer
/// iterations as |grad c~| is, the source pair of (160.5)-(160.6), and the
/// clock bookkeeping of (160.3). The fields are `pub(crate)` so
/// `progress.rs` can drive them.
pub struct AgeState {
    pub(crate) alpha: GpuScalarField,
    /// a* of (160.4), `[n_cells]`.
    pub(crate) closure_age: DevBuf<Scalar>,
    /// The clock source of (160.5), `[n_cells]`.
    pub(crate) sua: DevBuf<Scalar>,
    /// The inheritance coefficient of (160.6), `[n_cells]`.
    pub(crate) spa: DevBuf<Scalar>,
    /// The floor c_a of (160.1), positive and finite.
    pub(crate) c_a: f64,
    /// The clock's origin t_0 (D-20a): the end of the ignition ramp, or 0.
    pub(crate) t_origin: f64,
    /// The previous step's t_new; `None` before the first age step.
    pub(crate) t_prev: Option<f64>,
    /// Delta_n of (160.3).
    pub(crate) delta: f64,
    /// Delta_{n-1} of (160.5).
    pub(crate) delta_prev: f64,
    /// `begin_step` set it; [`crate::dust::progress::ProgressEquation::begin_age_step`]
    /// consumes it.
    pub(crate) pending: bool,
    /// The last alpha solve's report.
    pub(crate) last: Option<SolverPerformance>,
}

#[cfg(test)]
#[path = "tests_age.rs"]
mod tests;
