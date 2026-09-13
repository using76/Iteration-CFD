// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.

//! Verification and validation arithmetic - what a mesh refinement sequence
//! says about its own discretisation error, and what a comparison against a
//! published datum is worth. SPEC-LIT §94. Host only: no `Gpu`, no kernel,
//! no launch of any kind.
//!
//! Written from:
//!   P. J. Roache, *J. Fluids Eng.* 116 (1994) 405-413, DOI `10.1115/1.2910291`
//!     - the Grid Convergence Index: `GCI_fine` as a safety-factored,
//!     grid-refined measure of the discretisation uncertainty, and the reason
//!     a reported order travels with a reported uncertainty
//!   I. B. Celik, U. Ghia, P. J. Roache, C. J. Freitas, H. Coleman, P. Raad,
//!     *J. Fluids Eng.* 130 (2008) 078001, DOI `10.1115/1.2960953` - the
//!     reporting procedure this module's three-level half transcribes:
//!     (94.1) the typical cell size, (94.2) the ratios and differences,
//!     (94.3) the observed order with unequal refinement ratios solved as a
//!     fixed point, (94.4) the Richardson extrapolation, (94.5) `GCI_fine`
//!     with `F_s = 1.25`
//!   L. Eça, M. Hoekstra, *J. Comput. Phys.* 262 (2014) 104-130, DOI
//!     `10.1016/j.jcp.2014.01.006` - (94.6)-(94.9): the least-squares
//!     power-series and fixed-exponent fits, the standard deviation of a fit,
//!     the data-range parameter, and the uncertainty with `F_s = 1.25` or 3,
//!     which is what makes an oscillatory or scattered sequence reportable
//!     at all
//!   H. W. Coleman, F. Stern, "Uncertainties and CFD Code Validation",
//!     *J. Fluids Eng.* 119 (1997) 795-803 - (94.10) the comparison error
//!     `E = S - D` and the validation uncertainty `u_val` as the
//!     root-sum-square of the numerical, input and experimental
//!     uncertainties, the form ASME V&V 20-2009 adopted. No DOI is quoted
//!     for this paper: it is not in the programme's DOI ledger
//!   P. J. Roache, *J. Fluids Eng.* 124 (2002) 4-10, DOI `10.1115/1.1436090`
//!     - manufactured solutions as code verification, the reason the §94
//!     gates are MMS
//!   ofgpu `SPEC-LIT.md` §46.7, §60.5, §69, §79.12, §94
//!
//! No verification toolkit of any licence was consulted; the procedures are
//! the papers' own, transcribed. No GPL-licensed source was consulted.

use crate::{Error, Result, Scalar};

/// One mesh level: its typical cell size (94.1) and the gate's value on it.
/// Level 1 of a sequence is the FINEST mesh; `h` must grow along a sequence.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Level {
    pub h: Scalar,
    pub value: Scalar,
}

/// The arithmetic is `f64` whatever `Scalar` is, so the crate's `single`
/// feature cannot degrade an estimate (the conversion `mms_error` also makes).
fn w(x: Scalar) -> f64 {
    f64::from(x)
}

/// ...and back.
fn sc(x: f64) -> Scalar {
    x as Scalar
}

/// `Scalar` to `f64`, refusing what the papers' procedure cannot read.
fn finite(x: Scalar, what: &str) -> Result<f64> {
    let v = w(x);
    if v.is_finite() {
        Ok(v)
    } else {
        Err(Error::Config(format!("{what}: not finite ({v})")))
    }
}

/// (94.1) - the typical cell size of a uniform mesh: the cube root of the
/// volume per cell in 3-D, the square root of the area per cell in 2-D.
/// `dim` is 2 or 3; anything else is refused by name.
pub fn h_of(total_volume: Scalar, n_cells: usize, dim: usize) -> Result<Scalar> {
    match dim {
        2 | 3 => {}
        other => {
            return Err(Error::Config(format!(
                "h_of: dim must be 2 or 3, got {other}"
            )))
        }
    }
    if n_cells == 0 {
        return Err(Error::Config("h_of: n_cells is zero".to_string()));
    }
    let v = finite(total_volume, "h_of: total_volume")?;
    if v <= 0.0 {
        return Err(Error::Config(format!(
            "h_of: total_volume is {v}, not positive"
        )));
    }
    let per_cell = v / n_cells as f64;
    Ok(sc(if dim == 3 { per_cell.cbrt() } else { per_cell.sqrt() }))
}

/// (94.2)'s classification of the finest triplet, by the sign and the size of
/// `R = eps21/eps32` (Celik et al. 2008, §2): `0 < R < 1` is monotone
/// convergence, `R > 1` monotone divergence, `R < 0` oscillatory - converging
/// when `|R| < 1`, diverging when `|R| > 1`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Behaviour {
    /// `0 < R < 1`: differences shrink at a steady rate.
    Monotone,
    /// `R > 1`: differences grow monotonically - the mesh sequence is moving
    /// away from the answer.
    MonotoneDiverging,
    /// `R < 0`, `|R| < 1`: alternating signs, shrinking.
    Oscillatory,
    /// `R < 0`, `|R| > 1`: alternating signs, growing.
    OscillatoryDiverging,
}

impl Behaviour {
    /// Lower-case words for [`GridStudy::one_line`].
    fn words(self) -> &'static str {
        match self {
            Behaviour::Monotone => "monotone",
            Behaviour::MonotoneDiverging => "monotone diverging",
            Behaviour::Oscillatory => "oscillatory",
            Behaviour::OscillatoryDiverging => "oscillatory diverging",
        }
    }
}

/// (94.2)-(94.5) on the finest three levels. `p`, `phi_ext` and `gci_fine`
/// are `Some` only for [`Behaviour::Monotone`] - for any other behaviour the
/// order "cannot be established" (Eça & Hoekstra 2014, their section 2.4.2)
/// and the uncertainty comes from the least-squares fits of (94.6)-(94.9)
/// instead.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Triplet {
    pub r21: Scalar,
    pub r32: Scalar,
    pub eps21: Scalar,
    pub eps32: Scalar,
    /// `R = eps21/eps32` of (94.2).
    pub ratio: Scalar,
    pub behaviour: Behaviour,
    pub p: Option<Scalar>,
    pub phi_ext: Option<Scalar>,
    /// `e_a = |(phi_1 - phi_2)/phi_1|` - defined whatever `R` is.
    pub e_a: Scalar,
    pub gci_fine: Option<Scalar>,
    /// Fixed-point iterates (94.3) took; 0 when there was no order to find.
    pub iterations: usize,
}

/// (94.2)-(94.5), Celik et al. (2008) §2 steps 1-5, `levels[0]` finest.
///
/// Refuses, by name ([`Error::Config`]): `h` not strictly increasing fine to
/// coarse, `eps21 == 0`, a non-finite `h` or value, `eps32 == 0` (the ratio
/// would not be finite), a `phi_1` of zero (`e_a` would divide by it), or the
/// fixed point of (94.3) not converging in 200 iterations.
pub fn observed_order(levels: &[Level; 3]) -> Result<Triplet> {
    let h = [
        finite(levels[0].h, "observed_order: level 1 h")?,
        finite(levels[1].h, "observed_order: level 2 h")?,
        finite(levels[2].h, "observed_order: level 3 h")?,
    ];
    let phi = [
        finite(levels[0].value, "observed_order: level 1 value")?,
        finite(levels[1].value, "observed_order: level 2 value")?,
        finite(levels[2].value, "observed_order: level 3 value")?,
    ];
    if !(h[0] < h[1] && h[1] < h[2]) {
        return Err(Error::Config(format!(
            "observed_order: h is not strictly increasing fine to coarse: {}, {}, {}",
            h[0], h[1], h[2]
        )));
    }
    let eps21 = phi[0] - phi[1];
    let eps32 = phi[1] - phi[2];
    if eps21 == 0.0 {
        return Err(Error::Config(
            "observed_order: eps21 is zero - the two finest levels carry the same value \
             (94.2)"
                .to_string(),
        ));
    }
    if !eps32.is_finite() || eps32 == 0.0 {
        return Err(Error::Config(
            "observed_order: eps32 is zero or not finite - R = eps21/eps32 cannot be formed \
             (94.2)"
                .to_string(),
        ));
    }
    let r21 = h[1] / h[0];
    let r32 = h[2] / h[1];
    let ratio = eps21 / eps32;
    let behaviour = if ratio > 0.0 {
        if ratio < 1.0 {
            Behaviour::Monotone
        } else {
            Behaviour::MonotoneDiverging
        }
    } else if ratio.abs() < 1.0 {
        Behaviour::Oscillatory
    } else {
        Behaviour::OscillatoryDiverging
    };
    let e_a = if phi[0] != 0.0 {
        ((phi[0] - phi[1]) / phi[0]).abs()
    } else {
        return Err(Error::Config(
            "observed_order: phi_1 is zero - e_a of (94.5) would divide by it".to_string(),
        ));
    };
    let mut p = None;
    let mut phi_ext = None;
    let mut gci = None;
    let mut iterations = 0usize;
    if behaviour == Behaviour::Monotone {
        // (94.3) as a fixed point. The start is the q = 0 form, which is
        // exact when r21 == r32; `s` is sign(eps32/eps21), +1 on the monotone
        // path this branch is (the signs of eps21 and eps32 agree then).
        let sg = if eps32 / eps21 > 0.0 { 1.0 } else { -1.0 };
        let a = (eps32 / eps21).abs().ln().abs();
        let ln_r21 = r21.ln();
        let mut pk = a / ln_r21;
        let mut converged = false;
        for k in 0..200 {
            let q = ((r21.powf(pk) - sg) / (r32.powf(pk) - sg)).ln();
            let pn = (a + q) / ln_r21;
            iterations = k + 1;
            converged = (pn - pk).abs() < 1e-12;
            pk = pn;
            if converged {
                break;
            }
        }
        if !converged || !pk.is_finite() {
            return Err(Error::Config(format!(
                "observed_order: the fixed point of (94.3) did not converge in 200 iterations \
                 (last p = {pk})"
            )));
        }
        // (94.4) and (94.5), F_s = 1.25 - Celik's three-level value, the only
        // kind this crate makes (SPEC-LIT 94.1).
        let rp = r21.powf(pk);
        p = Some(sc(pk));
        phi_ext = Some(sc((rp * phi[0] - phi[1]) / (rp - 1.0)));
        gci = Some(sc(1.25 * e_a / (rp - 1.0)));
    }
    Ok(Triplet {
        r21: sc(r21),
        r32: sc(r32),
        eps21: sc(eps21),
        eps32: sc(eps32),
        ratio: sc(ratio),
        behaviour,
        p,
        phi_ext,
        e_a: sc(e_a),
        gci_fine: gci,
        iterations,
    })
}

/// (94.4) for a given `p`: the Richardson estimate of the exact value from
/// the finest two levels, `phi_ext = (r21^p phi_1 - phi_2) / (r21^p - 1)`.
pub fn richardson_extrapolate(fine: Level, medium: Level, p: Scalar) -> Scalar {
    let rp = (w(medium.h) / w(fine.h)).powf(w(p));
    sc((rp * w(fine.value) - w(medium.value)) / (rp - 1.0))
}

/// (94.5) for a given `p` and safety factor:
/// `GCI_fine = F_s |(phi_1 - phi_2)/phi_1| / (r21^p - 1)`.
pub fn gci_fine(fine: Level, medium: Level, p: Scalar, fs: Scalar) -> Scalar {
    let rp = (w(medium.h) / w(fine.h)).powf(w(p));
    let e_a = ((w(fine.value) - w(medium.value)) / w(fine.value)).abs();
    sc(w(fs) * e_a / (rp - 1.0))
}

/// Which of (94.6a)'s error forms a least-squares fit used. `PowerSeries` is
/// `eps_RE = alpha h^p` with `p` fitted (the outer search of Appendix B);
/// the other three fix the exponent, which is how an oscillatory or a
/// scattered sequence, or an observed order outside `[0.5, 2]`, still gets
/// an uncertainty (Eça & Hoekstra 2014, their section 2.4.2 and Appendix A).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Estimator {
    PowerSeries,
    First,
    Second,
    FirstSecond,
}

impl Estimator {
    /// (94.7)'s `n_par`, counting `phi_0`: 3 for `PowerSeries`
    /// (`phi_0, alpha, p`) and `FirstSecond` (`phi_0, alpha_1, alpha_2`),
    /// 2 for `First` and `Second`. A fit on `n_g` levels has no residual
    /// freedom left when `n_g - n_par <= 0` and is not attempted.
    fn n_par(self) -> usize {
        match self {
            Estimator::First | Estimator::Second => 2,
            Estimator::PowerSeries | Estimator::FirstSecond => 3,
        }
    }

    /// Lower-case words for [`GridStudy::one_line`].
    fn words(self) -> &'static str {
        match self {
            Estimator::PowerSeries => "power series",
            Estimator::First => "first order",
            Estimator::Second => "second order",
            Estimator::FirstSecond => "first+second order",
        }
    }
}

/// One least-squares fit of (94.6): `phi_i ~ phi_0 + eps(h_i)`. `alpha[1]` is
/// used by [`Estimator::FirstSecond`] only; `p` by [`Estimator::PowerSeries`]
/// only. `fitted` is the fit evaluated at every level given, same order.
#[derive(Debug, Clone, PartialEq)]
pub struct Fit {
    pub estimator: Estimator,
    pub weighted: bool,
    pub phi0: Scalar,
    pub alpha: [Scalar; 2],
    pub p: Option<Scalar>,
    /// (94.7): the standard deviation of the fit, over `n_g - n_par` degrees
    /// of freedom. Exact-zero on the three-level `PowerSeries` path, where
    /// the fit IS the triplet solution (94.3)-(94.4).
    pub sigma: Scalar,
    pub fitted: Vec<Scalar>,
}

/// (94.6c): `w_i = 1` unweighted, or `w_i = (1/h_i)/sum_j (1/h_j)` weighted;
/// `nw_i` is what (94.7)'s sum weights with - 1, or `n_g w_i`.
fn weights(levels: &[Level], weighted: bool) -> (Vec<f64>, Vec<f64>) {
    if !weighted {
        let n = levels.len();
        (vec![1.0; n], vec![1.0; n])
    } else {
        let inv: Vec<f64> = levels.iter().map(|l| 1.0 / w(l.h)).collect();
        let sum: f64 = inv.iter().sum();
        let nw_factor = levels.len() as f64;
        let wts: Vec<f64> = inv.iter().map(|x| x / sum).collect();
        let nws: Vec<f64> = wts.iter().map(|x| nw_factor * x).collect();
        (wts, nws)
    }
}

/// The design row `[1, eps-feature(s)]` of (94.6b) for `est` at `h`; `p` is
/// read only by `PowerSeries`. A `FirstSecond` row has three entries, the
/// rest two.
fn design(est: Estimator, h: f64, p: f64) -> [f64; 3] {
    match est {
        Estimator::PowerSeries => [1.0, h.powf(p), 0.0],
        Estimator::First => [1.0, h, 0.0],
        Estimator::Second => [1.0, h * h, 0.0],
        Estimator::FirstSecond => [1.0, h, h * h],
    }
}

fn n_cols(est: Estimator) -> usize {
    match est {
        Estimator::FirstSecond => 3,
        _ => 2,
    }
}

/// Gaussian elimination with partial pivoting, `n <= 3` - the weighted
/// normal equations of (94.6b) are at most 3x3. `None` when singular.
fn gauss_small(a: &mut [[f64; 3]; 3], b: &mut [f64; 3], n: usize) -> Option<[f64; 3]> {
    for col in 0..n {
        let piv = (col..n).fold(col, |best, r| {
            if a[r][col].abs() > a[best][col].abs() {
                r
            } else {
                best
            }
        });
        if a[piv][col] == 0.0 {
            return None;
        }
        a.swap(col, piv);
        b.swap(col, piv);
        for r in col + 1..n {
            let fct = a[r][col] / a[col][col];
            for cc in col..n {
                a[r][cc] -= fct * a[col][cc];
            }
            b[r] -= fct * b[col];
        }
    }
    let mut x = [0.0; 3];
    for r in (0..n).rev() {
        let mut acc = b[r];
        for cc in r + 1..n {
            acc -= a[r][cc] * x[cc];
        }
        x[r] = acc / a[r][r];
    }
    Some(x)
}

/// The raw product of one weighted least-squares solve: the coefficients, the
/// fit at every level, (94.6b)'s weighted SSR `S_X`, and the same sum with
/// (94.7)'s `nw_i` weights.
struct FitRaw {
    phi0: f64,
    alpha: [f64; 2],
    fitted: Vec<f64>,
    s_w: f64,
    s_nw: f64,
}

/// One solve of (94.6b) for `est` (at exponent `p`, read by `PowerSeries`
/// only). `None` when the normal equations are singular.
fn raw_fit(levels: &[Level], est: Estimator, p: f64, weighted: bool) -> Option<FitRaw> {
    let (wt, nw) = weights(levels, weighted);
    let n = n_cols(est);
    let mut a = [[0.0f64; 3]; 3];
    let mut b = [0.0f64; 3];
    for (i, l) in levels.iter().enumerate() {
        let dsn = design(est, w(l.h), p);
        for j in 0..n {
            b[j] += wt[i] * dsn[j] * w(l.value);
            for k in 0..n {
                a[j][k] += wt[i] * dsn[j] * dsn[k];
            }
        }
    }
    let coef = gauss_small(&mut a, &mut b, n)?;
    let mut fitted = Vec::with_capacity(levels.len());
    let mut s_w = 0.0;
    let mut s_nw = 0.0;
    for (i, l) in levels.iter().enumerate() {
        let dsn = design(est, w(l.h), p);
        let mut v = 0.0;
        for j in 0..n {
            v += coef[j] * dsn[j];
        }
        let r = w(l.value) - v;
        s_w += wt[i] * r * r;
        s_nw += nw[i] * r * r;
        fitted.push(v);
    }
    Some(FitRaw {
        phi0: coef[0],
        alpha: [coef[1], coef[2]],
        fitted,
        s_w,
        s_nw,
    })
}

/// (94.7) with `nw_i` from (94.6c). The caller has refused `n_g - n_par <= 0`.
fn sigma_of(s_nw: f64, n_g: usize, n_par: usize) -> f64 {
    (s_nw / (n_g - n_par) as f64).sqrt()
}

/// The MAGNITUDE of the chosen fit's error term (94.6a) at `h` - the
/// `|eps(phi_1)|` that (94.9) and `eps_fine` are stated in.
fn eps_term(fit: &Fit, h: f64) -> f64 {
    match fit.estimator {
        Estimator::PowerSeries => {
            (w(fit.alpha[0]) * h.powf(w(fit.p.expect("power series carries p")))).abs()
        }
        Estimator::First => (w(fit.alpha[0]) * h).abs(),
        Estimator::Second => (w(fit.alpha[0]) * h * h).abs(),
        Estimator::FirstSecond => (w(fit.alpha[0]) * h + w(fit.alpha[1]) * h * h).abs(),
    }
}

/// Golden-section search for the minimum of `f` on `[lo, hi]`, `iters`
/// iterations, returning the final bracket midpoint. DESIGN: this is the
/// outer minimisation of `S_RE` over `p` in (94.6b) - for fixed `p` the fit
/// is linear in `(phi_0, alpha)`, so the inner problem is one 2x2 weighted
/// normal-equations solve, and the search reaches the minimiser Eça &
/// Hoekstra reach through `dS/dp = 0` (their Appendix B.2) without ever
/// dividing by a vanishing derivative.
fn golden(f: impl Fn(f64) -> f64, lo: f64, hi: f64, iters: usize) -> f64 {
    let invphi = (5.0_f64.sqrt() - 1.0) / 2.0;
    let (mut a, mut b) = (lo, hi);
    let mut c = b - invphi * (b - a);
    let mut d = a + invphi * (b - a);
    let mut fc = f(c);
    let mut fd = f(d);
    for _ in 0..iters {
        if fc < fd {
            b = d;
            d = c;
            fd = fc;
            c = b - invphi * (b - a);
            fc = f(c);
        } else {
            a = c;
            c = d;
            fc = fd;
            d = a + invphi * (b - a);
            fd = f(d);
        }
    }
    (a + b) / 2.0
}

/// (94.6)-(94.7) with [`Estimator::PowerSeries`].
///
/// On exactly three levels the fit is the exact triplet solution - `p` from
/// [`observed_order`], `phi_0 = phi_ext` of (94.4), `alpha = (phi_1 -
/// phi_ext)/h_1^p`, `sigma = 0`, `weighted` recorded but unused - and a
/// non-monotone triplet is refused by name. On four or more levels `p` comes
/// from the golden section over `[0.05, 8]` (see [`golden`]). Fewer than
/// three levels is refused by name.
pub fn fit_power_series(levels: &[Level], weighted: bool) -> Result<Fit> {
    if levels.len() < 3 {
        return Err(Error::Config(format!(
            "fit_power_series: {} levels; three are the minimum (SPEC-LIT 94.1)",
            levels.len()
        )));
    }
    if levels.len() == 3 {
        let trip = observed_order(&[levels[0], levels[1], levels[2]])?;
        if trip.behaviour != Behaviour::Monotone {
            return Err(Error::Config(format!(
                "fit_power_series: the finest triplet is {:?} - (94.3) has no order to fit",
                trip.behaviour
            )));
        }
        let p = w(trip.p.expect("monotone triplet carries p"));
        let phi0 = w(trip.phi_ext.expect("monotone triplet carries phi_ext"));
        let a0 = (w(levels[0].value) - phi0) / w(levels[0].h).powf(p);
        let fitted: Vec<Scalar> = levels.iter().map(|l| sc(phi0 + a0 * w(l.h).powf(p))).collect();
        return Ok(Fit {
            estimator: Estimator::PowerSeries,
            weighted,
            phi0: sc(phi0),
            alpha: [sc(a0), sc(0.0)],
            p: Some(sc(p)),
            sigma: sc(0.0),
            fitted,
        });
    }
    let objective = |p: f64| -> f64 {
        raw_fit(levels, Estimator::PowerSeries, p, weighted)
            .map(|r| r.s_w)
            .unwrap_or(f64::INFINITY)
    };
    let p = golden(objective, 0.05, 8.0, 200);
    let raw = raw_fit(levels, Estimator::PowerSeries, p, weighted).ok_or_else(|| {
        Error::Config("fit_power_series: singular normal equations at the fitted p".to_string())
    })?;
    let sigma = sigma_of(raw.s_nw, levels.len(), Estimator::PowerSeries.n_par());
    Ok(Fit {
        estimator: Estimator::PowerSeries,
        weighted,
        phi0: sc(raw.phi0),
        alpha: [sc(raw.alpha[0]), sc(0.0)],
        p: Some(sc(p)),
        sigma: sc(sigma),
        fitted: raw.fitted.into_iter().map(sc).collect(),
    })
}

/// (94.6)-(94.7) with a fixed-exponent estimator - [`Estimator::First`],
/// [`Estimator::Second`] or [`Estimator::FirstSecond`]. Refuses
/// [`Estimator::PowerSeries`] (that is [`fit_power_series`]'s job) and a fit
/// with `n_g - n_par <= 0`, both by name.
pub fn fit_fixed(levels: &[Level], estimator: Estimator, weighted: bool) -> Result<Fit> {
    if estimator == Estimator::PowerSeries {
        return Err(Error::Config(
            "fit_fixed: PowerSeries has a fitted exponent - call fit_power_series".to_string(),
        ));
    }
    let n = levels.len();
    let n_par = estimator.n_par();
    if n <= n_par {
        return Err(Error::Config(format!(
            "fit_fixed: {n} levels leave no residual freedom for {estimator:?} \
             (n_g - n_par = {n} - {n_par} <= 0)"
        )));
    }
    let raw = raw_fit(levels, estimator, 0.0, weighted)
        .ok_or_else(|| Error::Config("fit_fixed: singular normal equations".to_string()))?;
    let sigma = sigma_of(raw.s_nw, n, n_par);
    Ok(Fit {
        estimator,
        weighted,
        phi0: sc(raw.phi0),
        alpha: [sc(raw.alpha[0]), sc(raw.alpha[1])],
        p: None,
        sigma: sc(sigma),
        fitted: raw.fitted.into_iter().map(sc).collect(),
    })
}

/// What a multi-level gate reports: the whole of Eça & Hoekstra's Appendix A,
/// evaluated at the finest level. `behaviour` and `triplet` are always the
/// finest THREE levels' (94.2)-(94.5); `estimator`/`weighted`/`phi_ext`/
/// `sigma` name the chosen fit; `p` is the observed order - the triplet's
/// whenever the triplet is monotone (reported even when a fixed-exponent
/// estimator carries the uncertainty), `None` otherwise; `eps_fine` is
/// `|eps(phi_1)|` of the chosen fit; `fs` by (94.8); `u_fine` by (94.9).
#[derive(Debug, Clone, PartialEq)]
pub struct GridStudy {
    pub n_levels: usize,
    pub behaviour: Behaviour,
    pub estimator: Estimator,
    pub weighted: bool,
    pub p: Option<Scalar>,
    /// The chosen fit's `phi_0` - the Richardson estimate on the
    /// `PowerSeries` path.
    pub phi_ext: Scalar,
    pub eps_fine: Scalar,
    pub sigma: Scalar,
    pub delta_phi: Scalar,
    pub fs: Scalar,
    pub u_fine: Scalar,
    pub triplet: Triplet,
}

/// The fixed-exponent candidates of Appendix A step 1, in the tie-breaking
/// order (unweighted before weighted, `eps_1` before `eps_2` before
/// `eps_12`); `eps_12` only when its degrees of freedom are positive.
fn fixed_candidates(levels: &[Level], with_twelve: bool) -> Result<Vec<Fit>> {
    let n = levels.len();
    let mut fits = Vec::new();
    for weighted in [false, true] {
        fits.push(fit_fixed(levels, Estimator::First, weighted)?);
        fits.push(fit_fixed(levels, Estimator::Second, weighted)?);
        if with_twelve && n > Estimator::FirstSecond.n_par() {
            fits.push(fit_fixed(levels, Estimator::FirstSecond, weighted)?);
        }
    }
    Ok(fits)
}

/// Smallest sigma; exact ties keep the earlier fit (the candidates' order).
fn smallest(fits: Vec<Fit>) -> Fit {
    fits.into_iter()
        .reduce(|a, b| if w(b.sigma) < w(a.sigma) { b } else { a })
        .expect("at least one candidate fit")
}

/// The whole of Appendix A on `levels`, `levels[0]` finest, `n_g >= 3`
/// (refused below). `triplet` is always the finest three levels'
/// (94.2)-(94.5); `phi_ext` is the chosen fit's `phi_0`; `eps_fine` is
/// `|eps(phi_1)|` of the chosen fit; `fs` by (94.8); `u_fine` by (94.9a)
/// when `sigma < delta_phi`, else (94.9b).
pub fn grid_study(levels: &[Level]) -> Result<GridStudy> {
    let n = levels.len();
    if n < 3 {
        return Err(Error::Config(format!(
            "grid_study: {} levels; three are the minimum (SPEC-LIT 94.1)",
            n
        )));
    }
    for (i, l) in levels.iter().enumerate() {
        finite(l.h, &format!("grid_study: level {} h", i + 1))?;
        finite(l.value, &format!("grid_study: level {} value", i + 1))?;
    }
    let trip = observed_order(&[levels[0], levels[1], levels[2]])?;
    // Appendix A step 1: the error estimate.
    let (chosen, study_p) = if trip.behaviour != Behaviour::Monotone {
        // p "cannot be established": the fixed-exponent fits carry it.
        let best = smallest(fixed_candidates(levels, true)?);
        (best, None)
    } else {
        let re_u = fit_power_series(levels, false)?;
        let re_w = fit_power_series(levels, true)?;
        let in_range = |fit: &Fit| matches!(fit.p, Some(p) if 0.5 <= w(p) && w(p) <= 2.0);
        let (u_in, w_in) = (in_range(&re_u), in_range(&re_w));
        if u_in || w_in {
            let best = match (u_in, w_in) {
                (true, true) => {
                    if w(re_u.sigma) <= w(re_w.sigma) { re_u } else { re_w }
                }
                (true, false) => re_u,
                (false, true) => re_w,
                (false, false) => unreachable!("a fit is in range"),
            };
            let p = best.p.expect("in-range power-series fit carries p");
            (best, Some(p))
        } else {
            // Both RE fits out of range: p* from the better of them, then
            // four fixed fits when p* > 2, six when p* < 0.5.
            let pstar = if w(re_u.sigma) <= w(re_w.sigma) {
                w(re_u.p.expect("power-series fit carries p"))
            } else {
                w(re_w.p.expect("power-series fit carries p"))
            };
            let best = smallest(fixed_candidates(levels, pstar < 0.5)?);
            (best, Some(sc(pstar)))
        }
    };
    // Step 2: the data-range parameter (94.8), over ALL levels.
    let mut lo = f64::INFINITY;
    let mut hi = f64::NEG_INFINITY;
    for l in levels {
        let v = w(l.value);
        lo = lo.min(v);
        hi = hi.max(v);
    }
    let delta_phi = (hi - lo) / (n - 1) as f64;
    // Step 3: the safety factor (94.8) - 1.25 only on the reliable path.
    let fs = if chosen.estimator == Estimator::PowerSeries
        && matches!(chosen.p, Some(p) if 0.5 <= w(p) && w(p) < 2.1)
        && w(chosen.sigma) < delta_phi
    {
        1.25
    } else {
        3.0
    };
    // Step 4: the uncertainty (94.9a)/(94.9b), at the FINEST level.
    let ef = eps_term(&chosen, w(levels[0].h));
    let df = (w(levels[0].value) - w(chosen.fitted[0])).abs();
    let sg = w(chosen.sigma);
    let u = if sg < delta_phi {
        fs * ef + sg + df
    } else {
        3.0 * (sg / delta_phi) * (ef + sg + df)
    };
    Ok(GridStudy {
        n_levels: n,
        behaviour: trip.behaviour,
        estimator: chosen.estimator,
        weighted: chosen.weighted,
        p: study_p,
        phi_ext: chosen.phi0,
        eps_fine: sc(ef),
        sigma: chosen.sigma,
        delta_phi: sc(delta_phi),
        fs: sc(fs),
        u_fine: sc(u),
        triplet: trip,
    })
}

impl GridStudy {
    /// One line for a note or the summary, e.g.
    /// `3 levels, monotone, p = 2.145, phi_ext = 1.521650e0, U_fine = 3.747e-3 (Fs 3, second order, unweighted)`
    /// - `p = n/a` when `None`; behaviour and estimator in words.
    pub fn one_line(&self) -> String {
        let p = match self.p {
            Some(p) => format!("{:.3}", w(p)),
            None => "n/a".to_string(),
        };
        format!(
            "{} levels, {}, p = {}, phi_ext = {:.6e}, U_fine = {:.3e} (Fs {}, {}, {})",
            self.n_levels,
            self.behaviour.words(),
            p,
            w(self.phi_ext),
            w(self.u_fine),
            w(self.fs),
            self.estimator.words(),
            if self.weighted { "weighted" } else { "unweighted" },
        )
    }
}

/// (94.10) - Coleman & Stern's validation metric: the comparison error
/// `E = S - D` and the validation uncertainty `u_val` as the root-sum-square
/// of the numerical, input and datum uncertainties; `u_num` is (94.9)'s
/// `U(phi_1)`, the study's `u_fine`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Validation {
    /// The simulation's finest-level value.
    pub s: Scalar,
    /// The reference datum.
    pub d: Scalar,
    pub e: Scalar,
    pub u_num: Scalar,
    pub u_input: Scalar,
    pub u_d: Scalar,
    pub u_val: Scalar,
}

/// (94.10): `E = S - D`, `u_val = sqrt(u_num^2 + u_input^2 + u_D^2)`.
pub fn validation(s: Scalar, d: Scalar, u_num: Scalar, u_input: Scalar, u_d: Scalar) -> Validation {
    let e = w(s) - w(d);
    let u_val = (w(u_num) * w(u_num) + w(u_input) * w(u_input) + w(u_d) * w(u_d)).sqrt();
    Validation {
        s,
        d,
        e: sc(e),
        u_num,
        u_input,
        u_d,
        u_val: sc(u_val),
    }
}

impl Validation {
    /// e.g. `Nu at Kr = 1: E = S - D = -4.710e-2 +/- 3.747e-3 (-3.00 % +/- 0.24 % of D; u_num 3.747e-3, u_input 0, u_D 0): |E| > u_val, the disagreement is not the mesh`
    /// - the last clause is `|E| <= u_val, the comparison cannot see a
    /// modelling error at this resolution` in the other case.
    pub fn one_line(&self, what: &str) -> String {
        let num = |x: Scalar| -> String {
            let v = w(x);
            if v == 0.0 {
                "0".to_string()
            } else {
                format!("{v:.3e}")
            }
        };
        let clause = if w(self.e).abs() <= w(self.u_val) {
            "|E| <= u_val, the comparison cannot see a modelling error at this resolution"
        } else {
            "|E| > u_val, the disagreement is not the mesh"
        };
        let pct = if w(self.d) != 0.0 {
            format!(
                "({:.2} % +/- {:.2} % of D; ",
                100.0 * w(self.e) / w(self.d),
                100.0 * w(self.u_val) / w(self.d)
            )
        } else {
            "(- % +/- - % of D; ".to_string()
        };
        format!(
            "{}: E = S - D = {:.3e} +/- {:.3e} {}u_num {}, u_input {}, u_D {}): {}",
            what,
            w(self.e),
            w(self.u_val),
            pct,
            num(self.u_num),
            num(self.u_input),
            num(self.u_d),
            clause,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lv(h: f64, v: f64) -> Level {
        Level {
            h: sc(h),
            value: sc(v),
        }
    }

    fn f64_of(x: Scalar) -> f64 {
        w(x)
    }

    /// **(94.1)-(94.9), the exact path.** `phi = 1 + 0.3 h^1.9` at
    /// `h = 0.25, 0.5, 1`: the fixed point of (94.3) converges in one iterate
    /// because `r21 = r32`, and the whole chain comes back exactly.
    #[test]
    fn an_exact_power_sequence_gives_its_order_back() {
        let hs = [0.25_f64, 0.5, 1.0];
        let phi = |h: f64| 1.0 + 0.3 * h.powf(1.9);
        let levels = [lv(hs[0], phi(hs[0])), lv(hs[1], phi(hs[1])), lv(hs[2], phi(hs[2]))];
        let t = observed_order(&levels).expect("exact sequence");
        assert_eq!(t.behaviour, Behaviour::Monotone);
        assert!((f64_of(t.p.unwrap()) - 1.9).abs() < 1e-12, "p = {:?}", t.p);
        assert!((f64_of(t.phi_ext.unwrap()) - 1.0).abs() < 1e-12);
        assert_eq!(t.iterations, 1, "equal ratios: q = 0, one iterate");
        let e_a = ((phi(hs[0]) - phi(hs[1])) / phi(hs[0])).abs();
        let want = 1.25 * e_a / (2.0_f64.powf(1.9) - 1.0);
        assert!((f64_of(t.gci_fine.unwrap()) - want).abs() < 1e-12);
        assert!((f64_of(t.gci_fine.unwrap()) - 2.6355e-2).abs() < 1e-6);
        let study = grid_study(&levels).expect("study");
        assert_eq!(study.estimator, Estimator::PowerSeries);
        assert!(!study.weighted);
        assert_eq!(f64_of(study.fs), 1.25);
        assert_eq!(f64_of(study.sigma), 0.0);
        let want_u = 1.25 * 0.3 * 0.25_f64.powf(1.9);
        assert!((f64_of(study.u_fine) - want_u).abs() < 1e-10, "u = {}", study.u_fine);
    }

    /// **(94.3) with unequal ratios.** `h = 1/2, 1/1.5, 1` so `r21 = 4/3`,
    /// `r32 = 3/2`: no closed form, the fixed point iterates (about 22
    /// times) and still recovers the exact order.
    #[test]
    fn unequal_refinement_ratios_recover_a_non_integer_order() {
        let hs = [0.5_f64, 1.0 / 1.5, 1.0];
        let phi = |h: f64| 1.0 + 0.3 * h.powf(1.7);
        let levels = [lv(hs[0], phi(hs[0])), lv(hs[1], phi(hs[1])), lv(hs[2], phi(hs[2]))];
        let t = observed_order(&levels).expect("exact sequence");
        assert_eq!(t.behaviour, Behaviour::Monotone);
        assert!((f64_of(t.r21) - 4.0 / 3.0).abs() < 1e-15);
        assert!((f64_of(t.r32) - 1.5).abs() < 1e-15);
        assert!((f64_of(t.p.unwrap()) - 1.7).abs() < 1e-10, "p = {:?}", t.p);
        assert!((f64_of(t.phi_ext.unwrap()) - 1.0).abs() < 1e-10);
        assert!(t.iterations > 10 && t.iterations < 40, "iters = {}", t.iterations);
    }

    /// **§60.5's Kr = 0.1 row.** The tabulated 40/60/80 `Nu` sequence is
    /// oscillatory at the table's own precision (`R = -0.1429`), so `p` is
    /// `None` and the fixed-exponent fits carry the uncertainty - and a
    /// finite `u_fine` comes out of them (Appendix A's third bullet).
    #[test]
    fn an_oscillatory_triplet_has_no_order_and_a_safety_factor_of_three() {
        let hs = [1.0 / 80.0, 1.0 / 60.0, 1.0 / 40.0];
        let phi = [0.38080_f64, 0.38079, 0.38086];
        let levels = [lv(hs[0], phi[0]), lv(hs[1], phi[1]), lv(hs[2], phi[2])];
        let t = observed_order(&levels).expect("readable triplet");
        assert_eq!(t.behaviour, Behaviour::Oscillatory);
        assert_eq!(t.p, None);
        assert!((f64_of(t.ratio) - -0.1429).abs() < 1e-4);
        let study = grid_study(&levels).expect("study");
        assert_eq!(study.p, None);
        assert_eq!(study.estimator, Estimator::Second);
        assert!(!study.weighted);
        assert_eq!(f64_of(study.fs), 3.0);
        assert!((f64_of(study.phi_ext) - 0.380766).abs() < 1e-5);
        let u = f64_of(study.u_fine);
        assert!((5e-5..3e-4).contains(&u), "u_fine = {u}");
        assert!((u - 9.94e-5).abs() < 5e-6, "u_fine = {u}");
    }

    /// **§60.5's Kr = 1 and Kr = 10 rows.** `p = 2.1454` on the Kr = 1
    /// triplet falls OUTSIDE Appendix A's `[0.5, 2]`, so the four
    /// fixed-exponent fits carry the uncertainty (`F_s = 3`) while the order
    /// is still reported; Kr = 10's `p = 1.9659` is inside and keeps the
    /// power series with `F_s = 1.25`.
    #[test]
    fn the_kaminski_prakash_table_has_its_order() {
        let hs = [1.0 / 80.0, 1.0 / 60.0, 1.0 / 40.0];
        let build =
            |phi: [f64; 3]| [lv(hs[0], phi[0]), lv(hs[1], phi[1]), lv(hs[2], phi[2])];
        // Kr = 1.
        let t = observed_order(&build([1.52290, 1.52382, 1.52659])).expect("triplet");
        assert!((f64_of(t.p.unwrap()) - 2.1454).abs() < 1e-3);
        assert!((f64_of(t.phi_ext.unwrap()) - 1.52182).abs() < 1e-4);
        let study = grid_study(&build([1.52290, 1.52382, 1.52659])).expect("study");
        assert_eq!(study.estimator, Estimator::Second);
        assert!(!study.weighted);
        assert!((f64_of(study.p.unwrap()) - 2.1454).abs() < 1e-3, "p still reported");
        assert!((f64_of(study.phi_ext) - 1.52165).abs() < 1e-4);
        assert_eq!(f64_of(study.fs), 3.0);
        assert!((f64_of(study.u_fine) - 3.747e-3).abs() < 5e-5);
        // Kr = 10.
        let t10 = observed_order(&build([2.26916, 2.27158, 2.27841])).expect("triplet");
        assert!((f64_of(t10.p.unwrap()) - 1.9659).abs() < 1e-3);
        let s10 = grid_study(&build([2.26916, 2.27158, 2.27841])).expect("study");
        assert_eq!(s10.estimator, Estimator::PowerSeries);
        assert!((f64_of(s10.phi_ext) - 2.26598).abs() < 1e-4);
        assert_eq!(f64_of(s10.fs), 1.25);
        assert!((f64_of(s10.u_fine) - 3.978e-3).abs() < 2e-5);
    }

    /// **§79.12's four levels, Eça & Hoekstra's least squares on all four
    /// and Celik's triplet on the finest three.** The two DISAGREE on
    /// `R_t,in`'s order (`p ~ 0.53` against `1.28`) and agree on `R_t,out`
    /// (`1.90` against `1.93`) - which is the sentence the section records.
    #[test]
    fn the_qu_mudawar_levels_fit_and_the_finest_triplet_disagree_on_r_in() {
        let hs: Vec<f64> = [259_350.0_f64, 115_200.0, 48_600.0, 14_400.0]
            .iter()
            .map(|n| n.powf(-1.0 / 3.0))
            .collect();
        let levels = |phi: [f64; 4]| {
            [
                lv(hs[0], phi[0]),
                lv(hs[1], phi[1]),
                lv(hs[2], phi[2]),
                lv(hs[3], phi[3]),
            ]
        };
        // R_t,out: a clean power series on all four levels.
        let out = grid_study(&levels([0.23507, 0.23490, 0.23459, 0.23374])).expect("study");
        assert_eq!(out.estimator, Estimator::PowerSeries);
        let p = f64_of(out.p.expect("monotone"));
        assert!((1.89..=1.90).contains(&p), "p = {p}");
        assert!((f64_of(out.phi_ext) - 0.235325).abs() < 2e-5);
        assert!(f64_of(out.sigma) < f64_of(out.delta_phi), "sigma < delta_phi");
        assert_eq!(f64_of(out.fs), 1.25);
        assert!(
            (3.1e-4..3.3e-4).contains(&f64_of(out.u_fine)),
            "u = {}",
            out.u_fine
        );
        // R_t,in: the same, on a much steeper sequence.
        let r_in = grid_study(&levels([0.09289, 0.09239, 0.09163, 0.09050])).expect("study");
        assert_eq!(r_in.estimator, Estimator::PowerSeries);
        let pin = f64_of(r_in.p.expect("monotone"));
        assert!((0.50..=0.60).contains(&pin), "p = {pin}");
        assert!(
            (0.0961..0.0966).contains(&f64_of(r_in.phi_ext)),
            "phi_ext = {}",
            r_in.phi_ext
        );
        assert_eq!(f64_of(r_in.fs), 1.25);
        assert!(
            (4.0e-3..4.7e-3).contains(&f64_of(r_in.u_fine)),
            "u = {}",
            r_in.u_fine
        );
        // The finest triplet alone (Celik), both quantities.
        let tin = observed_order(&[
            lv(hs[0], 0.09289),
            lv(hs[1], 0.09239),
            lv(hs[2], 0.09163),
        ])
        .expect("triplet");
        assert!((f64_of(tin.p.unwrap()) - 1.277).abs() < 1e-3);
        assert!((f64_of(tin.phi_ext.unwrap()) - 0.09410).abs() < 1e-5);
        let tout = observed_order(&[
            lv(hs[0], 0.23507),
            lv(hs[1], 0.23490),
            lv(hs[2], 0.23459),
        ])
        .expect("triplet");
        assert!((f64_of(tout.p.unwrap()) - 1.927).abs() < 1e-3);
        assert!((f64_of(tout.phi_ext.unwrap()) - 0.23532).abs() < 1e-5);
    }

    /// **(94.6b) on an exact series.** `phi = 2 + 0.7 h^1.5` at four levels:
    /// the golden section lands on the exponent and the normal equations on
    /// the rest, weighted and unweighted alike.
    #[test]
    fn the_least_squares_fit_recovers_an_exact_power_series() {
        let hs = [1.0 / 8.0, 0.25, 0.5, 1.0];
        let phi = |h: f64| 2.0 + 0.7 * h.powf(1.5);
        let levels: Vec<Level> = hs.iter().map(|h| lv(*h, phi(*h))).collect();
        for weighted in [false, true] {
            let fit = fit_power_series(&levels, weighted).expect("fit");
            assert_eq!(fit.estimator, Estimator::PowerSeries);
            assert!((f64_of(fit.p.unwrap()) - 1.5).abs() < 1e-8, "p = {:?}", fit.p);
            assert!((f64_of(fit.phi0) - 2.0).abs() < 1e-10);
            assert!(f64_of(fit.sigma) < 1e-12, "sigma = {}", fit.sigma);
        }
    }

    /// **(94.9b).** The four-level exact series with `+-0.2` scatter has an
    /// oscillatory finest triplet (`R = -0.613`), so `p` is `None`, the six
    /// fixed-exponent fits are tried (winner `First`, unweighted; runner-up
    /// `Second` weighted), and sigma EXCEEDS the data range - which triples
    /// the uncertainty through (94.9b).
    #[test]
    fn scatter_larger_than_the_data_range_triples_the_uncertainty() {
        let hs = [1.0_f64 / 8.0, 0.25, 0.5, 1.0];
        let phi: [f64; 4] = [2.0 + 0.7 * hs[0].powf(1.5) + 0.2,
                   2.0 + 0.7 * hs[1].powf(1.5) - 0.2,
                   2.0 + 0.7 * hs[2].powf(1.5) + 0.2,
                   2.0 + 0.7 * hs[3].powf(1.5) - 0.2];
        let levels: Vec<Level> =
            hs.iter().zip(phi).map(|(h, v)| lv(*h, v)).collect();
        let t = observed_order(&[levels[0], levels[1], levels[2]]).expect("triplet");
        assert_eq!(t.behaviour, Behaviour::Oscillatory);
        assert!((f64_of(t.ratio) - -0.613).abs() < 1e-3);
        let study = grid_study(&levels).expect("study");
        assert_eq!(study.p, None, "oscillatory: no order");
        assert_eq!(study.estimator, Estimator::First);
        assert!(!study.weighted);
        assert!((f64_of(study.sigma) - 0.2446).abs() < 1e-3);
        // The runner-up, checked directly.
        let second_w = fit_fixed(&levels, Estimator::Second, true).expect("fit");
        assert!((f64_of(second_w.sigma) - 0.2457).abs() < 1e-3);
        assert!(f64_of(study.sigma) >= f64_of(study.delta_phi), "sigma >= delta_phi");
        assert!((f64_of(study.delta_phi) - 0.2042).abs() < 1e-3);
        assert_eq!(f64_of(study.fs), 3.0);
        let u = f64_of(study.u_fine);
        assert!((1.4..1.8).contains(&u), "u = {u}");
        assert!((u - 1.5945).abs() < 1e-3, "u = {u}");
        // (94.9a) on the same fit would give ~0.57, what the band excludes.
        let f1 = fit_fixed(&levels, Estimator::First, false).expect("fit");
        let ef = f64_of(f1.alpha[0]) * hs[0];
        let df = (phi[0] - f64_of(f1.fitted[0])).abs();
        let a_branch = f64_of(study.fs) * ef + f64_of(study.sigma) + df;
        assert!((0.5..0.65).contains(&a_branch), "94.9a would give {a_branch}");
    }

    /// **(94.10).** The comparison error is exactly the difference and the
    /// validation uncertainty exactly the root-sum-square; with no input or
    /// datum uncertainty declared, `u_val` is `u_num`.
    #[test]
    fn the_validation_metric_is_the_difference_and_the_root_sum_square() {
        let (s, d, u_num, u_input, u_d) = (1.0_f64, 2.0, 1.5, 0.5, 0.25);
        let v = validation(sc(s), sc(d), sc(u_num), sc(u_input), sc(u_d));
        assert_eq!(f64_of(v.e), s - d);
        let want = (u_num * u_num + u_input * u_input + u_d * u_d).sqrt();
        assert!((f64_of(v.u_val) - want).abs() < 1e-15);
        assert_eq!(f64_of(v.u_val), want);
        let zero = validation(sc(s), sc(d), sc(u_num), sc(0.0), sc(0.0));
        assert_eq!(f64_of(zero.u_val), u_num);
        let line = v.one_line("Nu at Kr = 1");
        assert!(line.contains("E = S - D = -1.000e0 +/- 1.601e0"), "{line}");
        assert!(line.contains("cannot see a modelling error"), "{line}");
        let far = validation(sc(1.0), sc(2.0), sc(0.1), sc(0.0), sc(0.0));
        let line_far = far.one_line("Nu at Kr = 1");
        assert!(line_far.contains("the disagreement is not the mesh"), "{line_far}");
        let close = validation(sc(2.0), sc(2.001), sc(0.1), sc(0.0), sc(0.0));
        assert!(close.one_line("x").contains("cannot see a modelling error"));
    }

    /// **Refusals, by name.** Every sequence the procedure cannot read is
    /// refused as `Error::Config` naming what is wrong with it.
    #[test]
    fn the_estimators_refuse_a_sequence_they_cannot_read() {
        // h not strictly increasing fine -> coarse.
        let flat = [lv(0.5, 1.0), lv(0.5, 2.0), lv(1.0, 3.0)];
        assert!(matches!(observed_order(&flat), Err(Error::Config(_))));
        // eps21 == 0.
        let same = [lv(0.25, 1.0), lv(0.5, 1.0), lv(1.0, 2.0)];
        assert!(matches!(observed_order(&same), Err(Error::Config(_))));
        // A non-finite value.
        let inf = [lv(0.25, f64::INFINITY), lv(0.5, 1.0), lv(1.0, 2.0)];
        assert!(matches!(observed_order(&inf), Err(Error::Config(_))));
        // Fewer than three levels.
        let two = [lv(0.25, 1.0), lv(0.5, 2.0)];
        assert!(matches!(grid_study(&two), Err(Error::Config(_))));
        assert!(matches!(fit_power_series(&two, false), Err(Error::Config(_))));
        // A non-monotone three-level triplet has no power series to fit.
        let osc = [
            lv(1.0 / 80.0, 0.38080),
            lv(1.0 / 60.0, 0.38079),
            lv(1.0 / 40.0, 0.38086),
        ];
        assert!(matches!(fit_power_series(&osc, false), Err(Error::Config(_))));
        // fit_fixed refuses PowerSeries, and a fit with no residual freedom.
        let three = [lv(0.25, 1.0), lv(0.5, 2.0), lv(1.0, 3.0)];
        assert!(matches!(
            fit_fixed(&three, Estimator::PowerSeries, false),
            Err(Error::Config(_))
        ));
        assert!(matches!(
            fit_fixed(&three, Estimator::FirstSecond, false),
            Err(Error::Config(_))
        ));
        // h_of refuses dim not 2 or 3.
        assert!(matches!(h_of(sc(1.0), 8, 1), Err(Error::Config(_))));
        assert!(matches!(h_of(sc(1.0), 8, 4), Err(Error::Config(_))));
    }
}
