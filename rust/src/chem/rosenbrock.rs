// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.
// Provenance: see PROVENANCE.md. No GPL-licensed source was consulted.

//! Stiff integration by Rosenbrock methods with embedded error control
//! (SPEC-LIT §132): the linearly implicit step of **Sandu, Verwer, Blom,
//! Spee, Carmichael & Potra**, "Benchmarking stiff ODE solvers for
//! atmospheric chemistry problems II: Rosenbrock solvers", *Atmos. Environ.*
//! 31 (1997) 3459-3472, DOI 10.1016/s1352-2310(97)83212-8 (written in
//! lower case, which a DOI ignores, so the §80 audit does not read it as
//! a section) — formula (4) of
//! p. 3461 is the step (132.1)-(132.2) and their section 3.3 is the step-size
//! control of (132.5)-(132.6), read from the open CWI copy — with the RODAS3
//! and ROS3 coefficient sets of (132.3)-(132.4) and §132.3's one-sided
//! finite-difference Jacobian. Host only, and f64 in every build: no
//! `Scalar`, no device types, so the module behaves identically in the
//! default build and in `--features single`.
//!
//! ROS4 is named and not built (SPEC-LIT §132.2): Sandu et al. tested it but
//! did not print its coefficients, and its primaries were not read, so its
//! coefficients wait for a read copy of Hairer & Wanner Table 7.2 or of its
//! primaries; a case naming "ros4" is refused as [`crate::Error::Refused`].
//!
//! No GPL-licensed source was consulted.

use crate::error::Error;

/// One coefficient set of (132.1)-(132.2). Unused entries are 0.
/// `alpha`/`gamma_ij` are strictly lower-triangular: row `i` holds stage
/// `i`'s `alpha_ij` and `gamma_ij` in columns `j < i` (0-based rows against
/// the 1-based stages of the formula).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Coefficients {
    /// The printed name of the set, "RODAS3" or "ROS3".
    pub name: &'static str,
    /// The stage count `s` of (132.1).
    pub stages: usize,
    /// The diagonal `gamma_ii = gamma` of every stage.
    pub gamma: f64,
    /// `alpha_ij` of (132.1), strictly lower-triangular.
    pub alpha: [[f64; 4]; 4],
    /// `gamma_ij` of (132.1), strictly lower-triangular.
    pub gamma_ij: [[f64; 4]; 4],
    /// The weights of the order-`order` formula `y_{n+1}`.
    pub b: [f64; 4],
    /// The weights of the embedded order-`embedded_order` formula
    /// `yhat_{n+1}`.
    pub b_hat: [f64; 4],
    /// The order of the `b` formula.
    pub order: u32,
    /// The order of the `b_hat` formula, `p_hat` of (132.6).
    pub embedded_order: u32,
}

/// RODAS3 of (132.3): Sandu et al. p. 3463's "a stiffly accurate, embedded
/// pair of order 3(2)", `s = 4`, both formulas L-stable. Fractions are
/// written as quotients so each printed value stays recognisable.
pub const RODAS3: Coefficients = Coefficients {
    name: "RODAS3",
    stages: 4,
    gamma: 1.0 / 2.0,
    alpha: [
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [3.0 / 4.0, -1.0 / 4.0, 1.0 / 2.0, 0.0],
    ],
    gamma_ij: [
        [0.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [-1.0 / 4.0, -1.0 / 4.0, 0.0, 0.0],
        [1.0 / 12.0, 1.0 / 12.0, -2.0 / 3.0, 0.0],
    ],
    b: [5.0 / 6.0, -1.0 / 6.0, -1.0 / 6.0, 1.0 / 2.0],
    b_hat: [3.0 / 4.0, -1.0 / 4.0, 1.0 / 2.0, 0.0],
    order: 3,
    embedded_order: 2,
};

/// ROS3 of (132.4): Sandu et al. p. 3464's "an embedded pair of order 3(2)",
/// `s = 3`, the third-order formula L-stable. The printed digits verbatim.
pub const ROS3: Coefficients = Coefficients {
    name: "ROS3",
    stages: 3,
    gamma: 0.43586652150845899941601945119356,
    alpha: [
        [0.0, 0.0, 0.0, 0.0],
        [0.43586652150845899941601945119356, 0.0, 0.0, 0.0],
        [0.43586652150845899941601945119356, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
    ],
    gamma_ij: [
        [0.0, 0.0, 0.0, 0.0],
        [-0.19294655696029095575009695436041, 0.0, 0.0, 0.0],
        [0.0, 1.74927148125794685173529749738960, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
    ],
    b: [
        -0.75457412385404315829818998646589,
        1.94100407061964420292840123379419,
        -0.18642994676560104463021124732829,
        0.0,
    ],
    b_hat: [
        -1.53358745784149585370766523913002,
        2.81745131148625772213931745457622,
        -0.28386385364476186843165221544619,
        0.0,
    ],
    order: 3,
    embedded_order: 2,
};

/// The built coefficient sets of SPEC-LIT §132.2: [`Method::Rodas3`] (the
/// default: stiffly accurate, L-stable at both orders) and
/// [`Method::Ros3`]. `Method::Ros4` is named and not built, so a case
/// naming it is refused (R6).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Method {
    /// RODAS3 of (132.3), the default.
    #[default]
    Rodas3,
    /// ROS3 of (132.4).
    Ros3,
}

impl Method {
    /// Parses the exact, lower-case name: "rodas3" or "ros3". "ros4" is
    /// `Error::Refused`, saying its coefficients wait for a read copy of
    /// Hairer & Wanner Table 7.2 or of its primaries (SPEC-LIT §132.2).
    /// Any other name is `Error::Refused` naming the name and the
    /// recognised set "rodas3, ros3" (R6).
    pub fn from_name(name: &str) -> crate::Result<Method> {
        match name {
            "rodas3" => Ok(Method::Rodas3),
            "ros3" => Ok(Method::Ros3),
            "ros4" => Err(Error::Refused(format!(
                "method \"ros4\": its coefficients wait for a read copy of \
                 Hairer & Wanner Table 7.2 or of its primaries (Shampine \
                 1982; Kaps & Rentrop 1979), so it is named and not built \
                 (SPEC-LIT §132.7)"
            ))),
            other => Err(Error::Refused(format!(
                "method \"{}\": the recognised methods are \"rodas3, ros3\" \
                 (SPEC-LIT §132.7)",
                other
            ))),
        }
    }

    /// The exact, lower-case name: "rodas3" or "ros3".
    pub fn name(self) -> &'static str {
        match self {
            Method::Rodas3 => "rodas3",
            Method::Ros3 => "ros3",
        }
    }

    /// The set's [`Coefficients`], `&RODAS3` or `&ROS3`.
    pub fn coefficients(self) -> &'static Coefficients {
        match self {
            Method::Rodas3 => &RODAS3,
            Method::Ros3 => &ROS3,
        }
    }
}

/// The controls of one [`integrate`] call (SPEC-LIT §132.3 and R5).
#[derive(Clone, Debug, PartialEq)]
pub struct RosOptions {
    /// Which coefficient set runs; [`Method::Rodas3`] by default.
    pub method: Method,
    /// The relative tolerance of (132.5).
    pub rtol: f64,
    /// The absolute tolerance of (132.5), one component per unknown.
    pub atol: Vec<f64>,
    /// The initial step, clamped into `[h_min, h_max]` and then to
    /// `t1 - t0` (R3).
    pub h0: f64,
    /// The lower bound of `h`; a proposed `h` below it after a rejection is
    /// refused, never forced (§132.3).
    pub h_min: f64,
    /// The upper bound of `h`.
    pub h_max: f64,
    /// The most step attempts one call may make.
    pub max_steps: usize,
}

impl RosOptions {
    /// `method` [`Method::Rodas3`], `atol = vec![atol; n]`, `h0` 1e-6,
    /// `h_min` 0.0, `h_max` infinity, `max_steps` 1_000_000.
    pub fn new(n: usize, rtol: f64, atol: f64) -> RosOptions {
        RosOptions {
            method: Method::Rodas3,
            rtol,
            atol: vec![atol; n],
            h0: 1e-6,
            h_min: 0.0,
            h_max: f64::INFINITY,
            max_steps: 1_000_000,
        }
    }
}

/// Counters of one [`integrate`] call. `h_next`: the step the controller
/// proposed after the last accepted step.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct RosStats {
    /// Steps accepted (`Err < 1`, both formulas finite).
    pub accepted: usize,
    /// Steps rejected, `Err >= 1` or not finite.
    pub rejected: usize,
    /// Rejections under R4, also counted in `rejected`.
    pub nonfinite: usize,
    /// Every call of `f`: the Jacobian's `f0` and its `n` columns, every
    /// stage.
    pub f_evals: usize,
    /// Jacobians built (R2: one per accepted point).
    pub jac_evals: usize,
    /// Factorisations of `I - gamma h J` (R1: one per step attempt).
    pub lu_count: usize,
    /// The step the controller proposed after the last accepted step.
    pub h_next: f64,
}

/// R1: in-place LU with partial pivoting (the largest `abs` in the column,
/// the first one on a tie), of the row-major `n x n` `a`. `piv[k]` is the
/// row swapped into `k` at step `k`. Returns false (singular) when a pivot
/// is exactly 0 or not finite.
pub fn lu_factor(a: &mut [f64], n: usize, piv: &mut [usize]) -> bool {
    debug_assert!(a.len() >= n * n && piv.len() >= n);
    for k in 0..n {
        let mut p = k;
        let mut best = a[k * n + k].abs();
        for i in k + 1..n {
            let v = a[i * n + k].abs();
            if v > best {
                best = v;
                p = i;
            }
        }
        if !best.is_finite() || best == 0.0 {
            return false;
        }
        piv[k] = p;
        if p != k {
            for j in 0..n {
                a.swap(k * n + j, p * n + j);
            }
        }
        for i in k + 1..n {
            a[i * n + k] /= a[k * n + k];
            let m = a[i * n + k];
            if m != 0.0 {
                for j in k + 1..n {
                    a[i * n + j] -= m * a[k * n + j];
                }
            }
        }
    }
    true
}

/// Solves `A x = b` with the factors [`lu_factor`] left in `a`, applying
/// the same row swaps to `b` and overwriting it with `x`.
pub fn lu_solve(a: &[f64], n: usize, piv: &[usize], b: &mut [f64]) {
    debug_assert!(a.len() >= n * n && piv.len() >= n && b.len() >= n);
    for k in 0..n {
        b.swap(k, piv[k]);
    }
    for i in 0..n {
        let mut s = b[i];
        for j in 0..i {
            s -= a[i * n + j] * b[j];
        }
        b[i] = s;
    }
    for i in (0..n).rev() {
        let mut s = b[i];
        for j in i + 1..n {
            s -= a[i * n + j] * b[j];
        }
        b[i] = s / a[i * n + i];
    }
}

/// §132.3's one-sided finite-difference Jacobian of `f` at `y`: column `j`
/// is `(f(y + delta_j e_j) - f0) / delta_j`, with
/// `delta_j = sqrt(f64::EPSILON) max(abs(y_j), atol_j)`. `f0` must be
/// `f(y)`, so the whole matrix costs `n` calls of `f`.
pub fn fd_jacobian<F>(f: &mut F, y: &[f64], f0: &[f64], atol: &[f64], jac: &mut [f64]) -> crate::Result<()>
where
    F: FnMut(&[f64], &mut [f64]) -> crate::Result<()>,
{
    let n = y.len();
    if atol.len() != n {
        return Err(Error::Config(format!(
            "fd_jacobian: atol has {} components, y has {} (SPEC-LIT §132.7)",
            atol.len(),
            n
        )));
    }
    if jac.len() < n * n {
        return Err(Error::Config(format!(
            "fd_jacobian: jac has {} entries, y has {} (needs {}) (SPEC-LIT §132.7)",
            jac.len(),
            n,
            n * n
        )));
    }
    let mut y_pert = y.to_vec();
    let mut f_pert = vec![0.0; n];
    for j in 0..n {
        let delta = f64::EPSILON.sqrt() * y[j].abs().max(atol[j]);
        y_pert.copy_from_slice(y);
        y_pert[j] = y[j] + delta;
        f(&y_pert, &mut f_pert)?;
        for i in 0..n {
            jac[i * n + j] = (f_pert[i] - f0[i]) / delta;
        }
    }
    Ok(())
}

/// (132.5): the weighted RMS error measure,
/// `sqrt((1/m) sum_k (est_k / (atol_k + rtol abs(y_new_k)))^2)`.
pub fn error_norm(y_new: &[f64], est: &[f64], rtol: f64, atol: &[f64]) -> f64 {
    let mut sum = 0.0;
    for l in 0..y_new.len() {
        let e = est[l] / (atol[l] + rtol * y_new[l].abs());
        sum += e * e;
    }
    (sum / y_new.len() as f64).sqrt()
}

/// One step (132.1)-(132.2) of size `h` from `y` with the Jacobian `jac`
/// (row-major, `jac[i * n + j] = d f_i / d y_j`): factors `I - gamma h J`
/// once (R1), then one solve per stage. Writes `y_new = y_{n+1}` and
/// `est = y_{n+1} - yhat_{n+1}`. A singular `I - gamma h J` is
/// `Error::Config`, naming `h`.
pub fn ros_step<F>(
    c: &Coefficients,
    f: &mut F,
    y: &[f64],
    h: f64,
    jac: &[f64],
    y_new: &mut [f64],
    est: &mut [f64],
) -> crate::Result<()>
where
    F: FnMut(&[f64], &mut [f64]) -> crate::Result<()>,
{
    let n = y.len();
    let mut lu = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            lu[i * n + j] = -c.gamma * h * jac[i * n + j];
        }
        lu[i * n + i] += 1.0;
    }
    let mut piv = vec![0usize; n];
    if !lu_factor(&mut lu, n, &mut piv) {
        return Err(Error::Config(format!(
            "ros_step: I - gamma h J is singular at h = {:e} (a pivot is 0 \
             or not finite) (SPEC-LIT §132.7)",
            h
        )));
    }
    let s = c.stages;
    let mut k: Vec<Vec<f64>> = (0..s).map(|_| vec![0.0; n]).collect();
    let mut v = vec![0.0; n];
    let mut rhs = vec![0.0; n];
    let mut arg = vec![0.0; n];
    for i in 0..s {
        // The stage argument y_n + sum_{j<i} alpha_ij k_j.
        arg.copy_from_slice(y);
        for j in 0..i {
            let a_ij = c.alpha[i][j];
            if a_ij != 0.0 {
                for l in 0..n {
                    arg[l] += a_ij * k[j][l];
                }
            }
        }
        f(&arg, &mut rhs)?;
        // The right-hand side h f(arg) + h J sum_{j<i} gamma_ij k_j.
        for l in 0..n {
            v[l] = 0.0;
        }
        for j in 0..i {
            let g_ij = c.gamma_ij[i][j];
            if g_ij != 0.0 {
                for l in 0..n {
                    v[l] += g_ij * k[j][l];
                }
            }
        }
        for l in 0..n {
            let mut jv = 0.0;
            for m in 0..n {
                jv += jac[l * n + m] * v[m];
            }
            rhs[l] = h * rhs[l] + h * jv;
        }
        // (I - gamma h J) k_i = rhs, then back-substitution in place.
        lu_solve(&lu, n, &piv, &mut rhs);
        k[i].copy_from_slice(&rhs);
    }
    // (132.2): y_{n+1} = y_n + sum b_i k_i, est = y_{n+1} - yhat_{n+1}
    // = sum (b_i - bhat_i) k_i.
    for l in 0..n {
        let mut sy = 0.0;
        let mut se = 0.0;
        for i in 0..s {
            sy += c.b[i] * k[i][l];
            se += (c.b[i] - c.b_hat[i]) * k[i][l];
        }
        y_new[l] = y[l] + sy;
        est[l] = se;
    }
    Ok(())
}

/// Integrates the autonomous system `y' = f(y)` from `t0` to `t1` in place,
/// by the step (132.1)-(132.2) with the error control of (132.5)-(132.6)
/// and the design choices R1-R6 of SPEC-LIT §132.7: R1's pivoted LU, R2's
/// Jacobian once per accepted point, R3's shortened last step, R4's
/// non-finite rejection, R5's refusals and R6's names. `f`'s own error
/// passes through unchanged. Returns the call's counters.
pub fn integrate<F>(
    f: &mut F,
    y: &mut [f64],
    t0: f64,
    t1: f64,
    opts: &RosOptions,
) -> crate::Result<RosStats>
where
    F: FnMut(&[f64], &mut [f64]) -> crate::Result<()>,
{
    integrate_observed(f, y, t0, t1, opts, &mut |_, _| Ok(()))
}

/// As [`integrate`], and after every ACCEPTED step (t already advanced, y
/// already = y_{n+1}) calls `observe(t, y)`. The observer may change y (the
/// next step starts from the changed y; the Jacobian is rebuilt there
/// anyway by R2) or return `Err`, which ends the call with that `Err`
/// unchanged.
pub fn integrate_observed<F, O>(
    f: &mut F,
    y: &mut [f64],
    t0: f64,
    t1: f64,
    opts: &RosOptions,
    observe: &mut O,
) -> crate::Result<RosStats>
where
    F: FnMut(&[f64], &mut [f64]) -> crate::Result<()>,
    O: FnMut(f64, &mut [f64]) -> crate::Result<()>,
{
    const TAG: &str = "(SPEC-LIT §132.7)";
    let n = y.len();
    // R5's validation, each refusal naming the value.
    if !opts.h0.is_finite() || opts.h0 <= 0.0 {
        return Err(Error::Config(format!(
            "h0 = {:e} must be finite and above 0 {}",
            opts.h0, TAG
        )));
    }
    // `RosOptions::new`'s default `h_max` is infinity, the unbounded cap,
    // so +inf is accepted here; anything else must be finite and above 0.
    if !(opts.h_max.is_finite() && opts.h_max > 0.0) && opts.h_max != f64::INFINITY {
        return Err(Error::Config(format!(
            "h_max = {:e} must be finite and above 0 {}",
            opts.h_max, TAG
        )));
    }
    if !opts.h_min.is_finite() || opts.h_min < 0.0 {
        return Err(Error::Config(format!(
            "h_min = {:e} must be finite and at least 0 {}",
            opts.h_min, TAG
        )));
    }
    if opts.h_min > opts.h_max {
        return Err(Error::Config(format!(
            "h_min = {:e} exceeds h_max = {:e} {}",
            opts.h_min, opts.h_max, TAG
        )));
    }
    if !opts.rtol.is_finite() || opts.rtol < 0.0 {
        return Err(Error::Config(format!(
            "rtol = {:e} must be finite and at least 0 {}",
            opts.rtol, TAG
        )));
    }
    if opts.atol.len() != n {
        return Err(Error::Config(format!(
            "atol has {} components, y has {} {}",
            opts.atol.len(),
            n,
            TAG
        )));
    }
    for (l, &a) in opts.atol.iter().enumerate() {
        if !a.is_finite() || a <= 0.0 {
            return Err(Error::Config(format!(
                "atol[{}] = {:e} must be finite and above 0 {}",
                l, a, TAG
            )));
        }
    }
    if t1 < t0 {
        return Err(Error::Config(format!(
            "t1 = {:e} is below t0 = {:e} {}",
            t1, t0, TAG
        )));
    }
    if opts.max_steps == 0 {
        return Err(Error::Config(format!("max_steps = 0 {}", TAG)));
    }
    let mut stats = RosStats::default();
    // R5: t1 = t0 returns at once, with zero steps.
    if t1 == t0 {
        return Ok(stats);
    }
    let c = opts.method.coefficients();
    let mut t = t0;
    // R3: h0 is clamped into [h_min, h_max] and then to t1 - t0.
    let mut h = opts.h0.clamp(opts.h_min, opts.h_max).min(t1 - t0);
    let mut fac_max = 10.0f64;
    let mut accepted_any = false;
    let mut attempts = 0usize;
    let inv_p1 = 1.0 / (c.embedded_order as f64 + 1.0);
    let mut have_jac = false;
    let mut jac = vec![0.0; n * n];
    let mut f0 = vec![0.0; n];
    let mut y_new = vec![0.0; n];
    let mut est = vec![0.0; n];
    while t < t1 {
        // R3: the end point. The shortened last step is exempt from h_min.
        let last = t + h >= t1;
        if last {
            h = t1 - t;
        }
        // R2: the Jacobian at y_n, once per accepted point; a retry after a
        // rejection reuses it.
        if !have_jac {
            f(y, &mut f0)?;
            stats.f_evals += 1;
            fd_jacobian(f, y, &f0, &opts.atol, &mut jac)?;
            stats.f_evals += n;
            stats.jac_evals += 1;
            have_jac = true;
        }
        attempts += 1;
        if attempts > opts.max_steps {
            return Err(Error::Config(format!(
                "step attempt {} exceeds max_steps = {} at t = {:e} {}",
                attempts, opts.max_steps, t, TAG
            )));
        }
        ros_step(c, f, y, h, &jac, &mut y_new, &mut est)?;
        stats.f_evals += c.stages;
        stats.lu_count += 1;
        let err = error_norm(&y_new, &est, opts.rtol, &opts.atol);
        let finite = err.is_finite() && y_new.iter().all(|v| v.is_finite());
        // (132.6)'s factor, capped by fac_max: 10 after an accepted step, 1
        // after a rejection; `Err = 0` gives the cap through the division.
        let fac = fac_max.min((0.9 / err.powf(inv_p1)).max(0.1));
        if finite && err < 1.0 {
            // Accept. On the last step t is set to t1 exactly (R3).
            t = if last { t1 } else { t + h };
            y.copy_from_slice(&y_new);
            stats.accepted += 1;
            accepted_any = true;
            fac_max = 10.0;
            h = (h * fac).min(opts.h_max);
            stats.h_next = h;
            have_jac = false;
            observe(t, y)?;
        } else {
            // Reject. R4: a non-finite step also counts in `nonfinite` and
            // always divides h by 10.
            stats.rejected += 1;
            fac_max = 1.0;
            if !finite {
                stats.nonfinite += 1;
                h /= 10.0;
            } else if accepted_any {
                h *= fac;
            } else {
                // A rejection of the very first step divides h by 10
                // (Sandu et al. p. 3463).
                h /= 10.0;
            }
            if h < opts.h_min {
                return Err(Error::Config(format!(
                    "h = {:e} fell below h_min = {:e} at t = {:e} after a \
                     rejected step {}",
                    h, opts.h_min, t, TAG
                )));
            }
        }
    }
    Ok(stats)
}
