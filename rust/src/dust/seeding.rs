// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.
// Provenance: see PROVENANCE.md. No GPL-licensed source was consulted.

//! The seeding of a uniform dust cloud onto a mesh (SPEC-LIT §159.9): the
//! two incomplete gamma functions, the Rosin-Rammler band diameters they
//! give ((159.12), the mass-weighted mean diameter of band j), and the
//! mass-weighted mean of a seeded set.
//!
//! Every number here is `f64` in BOTH builds - the cloud lowers on the host
//! only, and the two builds must agree on its diameters to the last bit.
//!
//! Written from:
//!   ofgpu SPEC-LIT.md §159.9 - the bands (159.12), the mean (159.13) and
//!     the series/continued-fraction pair (159.15) with their derivations
//!   DLMF chapter 8: section 8.2 and eqs. 8.7.1 and 8.9.2
//!     (https://dlmf.nist.gov/8, open) - the definitions, the series,
//!     and the continued fraction whose even contraction is used
//!   W. J. Lentz, Appl. Opt. 15 (1976) 668-671; I. J. Thompson and
//!     A. R. Barnett, J. Comput. Phys. 64 (1986) 490-509 - the modified
//!     Lentz evaluation of the continued fraction
//!
//! The specification was informed by a review of FSCDustFoam (Huang & Lipatnikov, GPL) and no code was copied.
//! No GPL-licensed source was consulted.

use crate::error::{Error, Result};

const TAG: &str = "(SPEC-LIT §159.9)";

/// The one house refusal of this file: a key, a sentence, the section.
fn refused(key: &str, msg: &str) -> Error {
    Error::Refused(format!("{key}: {msg} {TAG}"))
}

/// The lower incomplete gamma function gamma(a, x), a > 1, x >= 0 - the
/// series of (159.15) (DLMF 8.7.1 with Gamma(a)/Gamma(a+k+1) written as the
/// product): `gamma(a,x) = exp(a ln x - x) S`, `S = sum t_k`, `t_0 = 1/a`,
/// `t_k = t_{k-1} x / (a + k)`. Every term is positive, so the sum carries
/// no cancellation; it stops after the first term at or below 1e-17 of the
/// running sum (cap 10000 terms). `x == 0` is 0.
#[must_use]
pub fn lower_gamma(a: f64, x: f64) -> f64 {
    if x == 0.0 {
        return 0.0;
    }
    let mut t = 1.0 / a;
    let mut s = t;
    for k in 1..10000 {
        t *= x / (a + k as f64);
        s += t;
        if t <= 1.0e-17 * s {
            break;
        }
    }
    (a * x.ln() - x).exp() * s
}

/// The upper incomplete gamma function Gamma(a, x), a > 1, x > 0 - the
/// continued fraction of (159.15), the even contraction of DLMF 8.9.2:
/// `Gamma(a,x) = exp(a ln x - x) / (b_0 + a_1/(b_1 + a_2/(b_2 + ...)))`,
/// `b_k = x + 2k + 1 - a`, `a_k = -k (k - a)` for k >= 1. Evaluated by the
/// modified Lentz algorithm, which carries the RECIPROCAL `f` of the
/// fraction: `f = 1/b_0`, then each step `D = b_k + a_k D`, `C = b_k +
/// a_k / C` (a zero denominator is replaced by `f64::MIN_POSITIVE`), `D =
/// 1/D`, `f *= C D`, stopping when `|C D - 1| <= 1e-16` (cap 10000 steps).
/// `x == +inf` is 0, Gamma(a, inf).
#[must_use]
pub fn upper_gamma(a: f64, x: f64) -> f64 {
    if x.is_infinite() {
        return 0.0;
    }
    let tiny = f64::MIN_POSITIVE;
    let b_0 = x + 1.0 - a;
    let mut f = 1.0 / b_0;
    let mut c = 1.0 / tiny;
    let mut d = f;
    for k in 1..=10000 {
        let kf = k as f64;
        let b_k = x + 2.0 * kf + 1.0 - a;
        let a_k = -kf * (kf - a);
        d = b_k + a_k * d;
        if d.abs() < tiny {
            d = tiny;
        }
        c = b_k + a_k / c;
        if c.abs() < tiny {
            c = tiny;
        }
        d = 1.0 / d;
        let delta = c * d;
        f *= delta;
        if (delta - 1.0).abs() <= 1.0e-16 {
            break;
        }
    }
    (a * x.ln() - x).exp() * f
}

/// Gamma(a) = gamma(a, a+1) + Gamma(a, a+1) - (159.15), the two halves each
/// taken in its convergent form at the switch point x = a + 1.
#[must_use]
pub fn gamma(a: f64) -> f64 {
    lower_gamma(a, a + 1.0) + upper_gamma(a, a + 1.0)
}

/// The `K` Rosin-Rammler band diameters of (159.12) (SPEC-LIT §159.9): with
/// (147.1)'s volume (mass) fraction `Q(d) = 1 - exp(-(d/d_bar)^n)` and `x =
/// (d/d_bar)^n`, band j holds the quantiles (j-1)/K to j/K - x from
/// `X_{j-1}` to `X_j` with `X_j = -ln(1 - j/K)`, `X_0 = 0`, `X_K = +inf` -
/// and its mass-weighted mean diameter is
///
///   `d_j = K d_bar [gamma(a, X_j) - gamma(a, X_{j-1})]`,  `a = 1 + 1/n`,
///
/// each band taking the form of (159.15) that does not cancel: a
/// `gamma(a, .)` difference when both ends are below `a + 1`, a
/// `Gamma(a, .)` difference when both are at or above it, and
/// `(Gamma(a) - gamma(a, X_{j-1})) - Gamma(a, X_j)` across the switch. The
/// bands sum to `K d_bar gamma(a, inf) = K d_bar Gamma(a)`, so (1/K) sum d_j
/// is the mass-weighted mean (159.13) for ANY K. Refused: a `d_bar` or `n`
/// that is not finite and above zero, `k == 0`, or a band diameter that is
/// not finite and above zero.
pub fn rosin_rammler_bands(d_bar: f64, n: f64, k: u32) -> Result<Vec<f64>> {
    if !(d_bar.is_finite() && d_bar > 0.0) {
        return Err(refused(
            "rosinRammler.d",
            &format!("{d_bar} is not a finite mean diameter above zero"),
        ));
    }
    if !(n.is_finite() && n > 0.0) {
        return Err(refused(
            "rosinRammler.n",
            &format!("{n} is not a finite exponent above zero"),
        ));
    }
    if k == 0 {
        return Err(refused(
            "parcelsPerCell",
            "0; a cell seeded with no parcels carries none of the cloud's mass",
        ));
    }
    let a = 1.0 + 1.0 / n;
    let switch = a + 1.0;
    let g = gamma(a);
    let kf = k as f64;
    let mut out = Vec::with_capacity(k as usize);
    let mut x0 = 0.0_f64;
    for j in 1..=k {
        let jf = j as f64;
        let x1 = if j == k { f64::INFINITY } else { -(-(jf / kf)).ln_1p() };
        let share = if x1 < switch {
            lower_gamma(a, x1) - lower_gamma(a, x0)
        } else if x0 >= switch {
            upper_gamma(a, x0) - upper_gamma(a, x1)
        } else {
            (g - lower_gamma(a, x0)) - upper_gamma(a, x1)
        };
        let d = kf * d_bar * share;
        if !(d.is_finite() && d > 0.0) {
            return Err(refused(
                "rosinRammler.d",
                &format!("band {j} of {k} is {d}, not a finite diameter above zero"),
            ));
        }
        out.push(d);
        x0 = x1;
    }
    Ok(out)
}

/// The mass-weighted mean of the paired values: `sum(m d) / sum(m)`. An
/// empty pair of slices divides by zero - the caller seeds at least one
/// parcel, so this never happens in the lowering.
#[must_use]
pub fn mass_weighted_mean(mass: &[f64], d: &[f64]) -> f64 {
    let mut md = 0.0;
    let mut m = 0.0;
    for (mi, di) in mass.iter().zip(d.iter()) {
        md += mi * di;
        m += mi;
    }
    md / m
}
