// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.

//! A property that is a function of temperature - §100. One evaluator,
//! four forms: the constant (S100.1), the piecewise-linear table (S100.2),
//! a generalised power series in `T/T_s` over contiguous pieces (S100.3),
//! and Sutherland's law (S100.4); every form but the constant has a range,
//! and an evaluation outside it is refused, never extrapolated (§100.1).
//! Host arithmetic only: nothing here launches a kernel.
//!
//! Written from:
//!   B. J. McBride, M. J. Zehe, S. Gordon, *NASA Glenn Coefficients for
//!     Calculating Thermodynamic Properties of Individual Species*,
//!     NASA/TP-2002-211556 (2002), eq. (1) - the seven-term `cp/R` a
//!     two-piece (S100.3) carries; a US-Government work, public domain
//!   K. Kadoya, N. Matsunaga, A. Nagashima, *J. Phys. Chem. Ref. Data* 14
//!     (1985) 947-970, DOI 10.1063/1.555744, eqs. (3a) and (5a) - the same
//!     series, with a scale `T*` and a factor
//!   W. Sutherland, *Phil. Mag.* 36 (1893) 507-531, DOI
//!     10.1080/14786449308620508 - (S100.4) in its textbook form; the paper
//!     was not read, and no constant of it is carried here: a case states
//!     all three
//!   ofgpu `SPEC-LIT.md` §13.4, §100
//!
//! No GPL-licensed source was consulted.

use crate::error::{Error, Result};
use crate::Scalar;

/// One piece of a (S100.3) series: its closed range, K, and one
/// coefficient per exponent.
#[derive(Debug, Clone, PartialEq)]
pub struct Piece {
    pub lo: Scalar,
    pub hi: Scalar,
    pub coefficients: Vec<Scalar>,
}

/// A property as a function of temperature - §100.1. Build every form but
/// the constant through its constructor, which validates it.
#[derive(Debug, Clone, PartialEq)]
pub enum Property {
    /// (S100.1): the number a case wrote - no range, the same bits at
    /// every temperature.
    Constant(Scalar),
    /// (S100.2): knots `t` strictly increasing, one value each.
    Table { t: Vec<Scalar>, v: Vec<Scalar> },
    /// (S100.3): `F sum_j c_j (T/T_s)^e_j` on each piece.
    Polynomial {
        exponents: Vec<Scalar>,
        pieces: Vec<Piece>,
        scale: Scalar,
        factor: Scalar,
    },
    /// (S100.4): `value (T/t_ref)^(3/2) (t_ref + s)/(T + s)` on `[lo, hi]`.
    Sutherland {
        value: Scalar,
        t_ref: Scalar,
        s: Scalar,
        lo: Scalar,
        hi: Scalar,
    },
}

/// A malformed curve, refused naming its setting (§100.5).
fn refuse(setting: &str, why: String) -> Error {
    Error::Config(format!("{setting}: {why} (SPEC-LIT 100.5)"))
}

/// §100.1's range rule: ascending, finite, from a positive temperature.
fn check_range(setting: &str, lo: Scalar, hi: Scalar) -> Result<()> {
    if !(lo > 0.0) || !lo.is_finite() || !hi.is_finite() || !(hi > lo) {
        return Err(refuse(
            setting,
            format!(
                "the range [{lo}, {hi}] K has to be ascending, finite, and start at a \
                 positive (absolute) temperature"
            ),
        ));
    }
    Ok(())
}

impl Property {
    /// (S100.2), validated: at least two knots, temperatures strictly
    /// increasing from a positive one, every number finite.
    pub fn table(setting: &str, knots: &[(Scalar, Scalar)]) -> Result<Property> {
        if knots.len() < 2 {
            return Err(refuse(
                setting,
                format!("a table needs at least two knots, it has {}", knots.len()),
            ));
        }
        for &(t, v) in knots {
            if !t.is_finite() || !v.is_finite() {
                return Err(refuse(setting, format!("the knot ({t}, {v}) is not finite")));
            }
        }
        for w in knots.windows(2) {
            if !(w[1].0 > w[0].0) {
                return Err(refuse(
                    setting,
                    format!(
                        "the knots' temperatures have to increase strictly, and {} K is \
                         followed by {} K",
                        w[0].0, w[1].0
                    ),
                ));
            }
        }
        check_range(setting, knots[0].0, knots[knots.len() - 1].0)?;
        Ok(Property::Table {
            t: knots.iter().map(|k| k.0).collect(),
            v: knots.iter().map(|k| k.1).collect(),
        })
    }

    /// (S100.3), validated: at least one exponent and one piece, each
    /// piece's coefficient count the exponent count, every number finite,
    /// each range as §100.1 says, the pieces contiguous (`hi` of one IS
    /// `lo` of the next), `scale > 0` and `factor` non-zero.
    pub fn polynomial(
        setting: &str,
        exponents: &[Scalar],
        pieces: &[Piece],
        scale: Scalar,
        factor: Scalar,
    ) -> Result<Property> {
        if exponents.is_empty() || exponents.iter().any(|e| !e.is_finite()) {
            return Err(refuse(
                setting,
                "a polynomial needs at least one exponent, every one finite".to_string(),
            ));
        }
        if pieces.is_empty() {
            return Err(refuse(setting, "a polynomial needs at least one piece".to_string()));
        }
        for (k, p) in pieces.iter().enumerate() {
            if p.coefficients.len() != exponents.len() {
                return Err(refuse(
                    setting,
                    format!(
                        "piece {k} has {} coefficients for {} exponents",
                        p.coefficients.len(),
                        exponents.len()
                    ),
                ));
            }
            if p.coefficients.iter().any(|c| !c.is_finite()) {
                return Err(refuse(setting, format!("piece {k} has a coefficient that is not finite")));
            }
            check_range(setting, p.lo, p.hi)?;
            if k > 0 && p.lo != pieces[k - 1].hi {
                return Err(refuse(
                    setting,
                    format!(
                        "piece {k} starts at {} K but piece {} ends at {} K: the pieces have \
                         to be contiguous",
                        p.lo,
                        k - 1,
                        pieces[k - 1].hi
                    ),
                ));
            }
        }
        if !(scale > 0.0) || !scale.is_finite() {
            return Err(refuse(setting, format!("scale = {scale}: it has to be finite and positive")));
        }
        if factor == 0.0 || !factor.is_finite() {
            return Err(refuse(setting, format!("factor = {factor}: it has to be finite and non-zero")));
        }
        Ok(Property::Polynomial {
            exponents: exponents.to_vec(),
            pieces: pieces.to_vec(),
            scale,
            factor,
        })
    }

    /// (S100.4), validated: `value > 0`, `t_ref > 0`, `s >= 0`, all finite,
    /// and the range as §100.1 says.
    pub fn sutherland(
        setting: &str,
        value: Scalar,
        t_ref: Scalar,
        s: Scalar,
        lo: Scalar,
        hi: Scalar,
    ) -> Result<Property> {
        for (what, x, ok) in [
            ("value", value, value > 0.0),
            ("TRef", t_ref, t_ref > 0.0),
            ("S", s, s >= 0.0),
        ] {
            if !ok || !x.is_finite() {
                return Err(refuse(
                    setting,
                    format!(
                        "{what} = {x}: Sutherland's law needs value > 0, TRef > 0 and S >= 0, \
                         all finite"
                    ),
                ));
            }
        }
        check_range(setting, lo, hi)?;
        Ok(Property::Sutherland { value, t_ref, s, lo, hi })
    }

    /// `true` for (S100.1): the form that takes the path it took before
    /// §100 - no range, no evaluation.
    pub fn is_constant(&self) -> bool {
        matches!(self, Property::Constant(_))
    }

    /// The closed range `[lo, hi]`, K, an evaluation must lie in; `None` for
    /// a constant.
    pub fn range(&self) -> Option<(Scalar, Scalar)> {
        match self {
            Property::Constant(_) => None,
            Property::Table { t, .. } => Some((t[0], t[t.len() - 1])),
            Property::Polynomial { pieces, .. } => Some((pieces[0].lo, pieces[pieces.len() - 1].hi)),
            Property::Sutherland { lo, hi, .. } => Some((*lo, *hi)),
        }
    }

    /// The temperatures §100.3 validates a curve's consumer at: every knot of
    /// a table; both ends and fifteen evenly spaced interior points of any
    /// other curve; none for a constant.
    pub fn samples(&self) -> Vec<Scalar> {
        if let Property::Table { t, .. } = self {
            return t.clone();
        }
        let Some((lo, hi)) = self.range() else {
            return Vec::new();
        };
        (0..=16)
            .map(|i| if i == 16 { hi } else { lo + (hi - lo) * (i as Scalar) / 16.0 })
            .collect()
    }

    /// One phrase for a banner or a refusal: the form, its size, its range.
    pub fn describe(&self) -> String {
        match self {
            Property::Constant(v) => format!("the constant {v}"),
            Property::Table { t, .. } => {
                format!("a table of {} knots on [{}, {}] K", t.len(), t[0], t[t.len() - 1])
            }
            Property::Polynomial { exponents, pieces, .. } => format!(
                "a {}-term polynomial in {} piece(s) on [{}, {}] K",
                exponents.len(),
                pieces.len(),
                pieces[0].lo,
                pieces[pieces.len() - 1].hi
            ),
            Property::Sutherland { lo, hi, .. } => format!("Sutherland's law on [{lo}, {hi}] K"),
        }
    }

    /// The property at `t`, K. A constant returns its number whatever `t`
    /// is; every other form refuses a `t` outside its range, naming
    /// `setting`, `t` and the range - a curve is not extrapolated (§100.1).
    pub fn value(&self, setting: &str, t: Scalar) -> Result<Scalar> {
        if let Some((lo, hi)) = self.range() {
            if !(t >= lo && t <= hi) {
                return Err(Error::Config(format!(
                    "{setting}: T = {t} K lies outside the curve's range [{lo}, {hi}] K; a \
                     curve is not extrapolated (SPEC-LIT 100.1)"
                )));
            }
        }
        Ok(match self {
            Property::Constant(v) => *v,
            Property::Table { t: ts, v } => table_at(ts, v, t),
            Property::Polynomial { exponents, pieces, scale, factor } => {
                // A `t` on a shared end belongs to the LOWER piece (§100.1).
                let p = pieces.iter().find(|p| t <= p.hi).unwrap_or(&pieces[pieces.len() - 1]);
                let x = t / *scale;
                let mut sum = 0.0 as Scalar;
                for (e, c) in exponents.iter().zip(&p.coefficients) {
                    sum += *c * power(x, *e);
                }
                *factor * sum
            }
            Property::Sutherland { value, t_ref, s, .. } => {
                *value * (t / *t_ref).powf(1.5) * (*t_ref + *s) / (t + *s)
            }
        })
    }
}

/// (S100.2) at `t`, which [`Property::value`] has already put inside
/// `[ts[0], ts[n-1]]`. The segment is the one whose LEFT end is the last
/// knot at or below `t`, and the last knot returns its own value, so every
/// knot returns its own value to the bit.
fn table_at(ts: &[Scalar], v: &[Scalar], t: Scalar) -> Scalar {
    let k = ts.partition_point(|&x| x <= t);
    if k >= ts.len() {
        return v[ts.len() - 1];
    }
    let i = k.saturating_sub(1);
    v[i] + (t - ts[i]) / (ts[k] - ts[i]) * (v[k] - v[i])
}

/// `x^e`: repeated multiplication (`powi`) for an integer exponent of
/// magnitude at most 16, `powf` for any other - §100.1's choice, stated so
/// that a device twin can make the same one.
fn power(x: Scalar, e: Scalar) -> Scalar {
    if e.fract() == 0.0 && e.abs() <= 16.0 {
        x.powi(e as i32)
    } else {
        x.powf(e)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_table_returns_each_knot_to_the_bit_and_is_linear_between() {
        let eps = Scalar::EPSILON;
        let knots: [(Scalar, Scalar); 3] = [(300.0, 148.0), (400.0, 98.9), (600.0, 61.9)];
        let p = Property::table("k", &knots).unwrap();
        for &(t, v) in &knots {
            assert_eq!(p.value("k", t).unwrap(), v);
        }
        let mid = p.value("k", 350.0).unwrap();
        assert!((mid - 0.5 * (148.0 + 98.9)).abs() <= 8.0 * eps * 148.0, "mid = {mid}");
        let q = p.value("k", 500.0).unwrap();
        assert!((q - 0.5 * (98.9 + 61.9)).abs() <= 8.0 * eps * 148.0, "q = {q}");
        assert_eq!(p.range(), Some((300.0, 600.0)));
    }

    #[test]
    fn an_evaluation_outside_the_range_is_refused_naming_the_range() {
        let setting = "regions/die/material/kappa";
        let table = Property::table(setting, &[(300.0, 148.0), (600.0, 61.9)]).unwrap();
        for t in [299.0, 601.0] {
            let e = table.value(setting, t).unwrap_err().to_string();
            assert!(e.contains(setting), "{e}");
            assert!(e.contains(&format!("T = {t} K")), "{e}");
            assert!(e.contains("[300, 600] K"), "{e}");
            assert!(e.contains("SPEC-LIT 100.1"), "{e}");
        }
        let poly = Property::polynomial(
            "x",
            &[0.0],
            &[Piece { lo: 200.0, hi: 1000.0, coefficients: vec![1.0] }],
            1.0,
            1.0,
        )
        .unwrap();
        let e = poly.value("x", 1000.5).unwrap_err().to_string();
        assert!(e.contains("[200, 1000] K"), "{e}");
        let suth = Property::sutherland("m", 1.8e-5, 300.0, 110.0, 200.0, 1500.0).unwrap();
        let e = suth.value("m", 150.0).unwrap_err().to_string();
        assert!(e.contains("[200, 1500] K"), "{e}");
        assert_eq!(Property::Constant(148.0).value("x", -5.0).unwrap(), 148.0);
    }

    #[test]
    fn a_malformed_curve_is_refused_by_name() {
        let two = || Piece { lo: 200.0, hi: 400.0, coefficients: vec![1.0] };
        let cases: Vec<(Result<Property>, &str)> = vec![
            (Property::table("x", &[(300.0, 1.0)]), "at least two knots"),
            (Property::table("x", &[(300.0, 1.0), (300.0, 2.0)]), "increase strictly"),
            (Property::table("x", &[(0.0, 1.0), (10.0, 2.0)]), "positive (absolute) temperature"),
            (Property::table("x", &[(300.0, Scalar::NAN), (400.0, 2.0)]), "is not finite"),
            (
                Property::polynomial("x", &[0.0, 1.0], &[two()], 1.0, 1.0),
                "coefficients for 2 exponents",
            ),
            (
                Property::polynomial(
                    "x",
                    &[0.0],
                    &[two(), Piece { lo: 450.0, hi: 600.0, coefficients: vec![1.0] }],
                    1.0,
                    1.0,
                ),
                "contiguous",
            ),
            (Property::polynomial("x", &[0.0], &[], 1.0, 1.0), "at least one piece"),
            (Property::polynomial("x", &[0.0], &[two()], 0.0, 1.0), "scale = 0"),
            (Property::polynomial("x", &[0.0], &[two()], 1.0, 0.0), "factor = 0"),
            (Property::sutherland("x", 1.8e-5, 300.0, -1.0, 200.0, 1500.0), "S = -1"),
            (Property::sutherland("x", 1.8e-5, 300.0, 110.0, 500.0, 400.0), "ascending"),
        ];
        for (result, phrase) in cases {
            let e = result.unwrap_err().to_string();
            assert!(e.contains('x'), "{e}");
            assert!(e.contains(phrase), "{e}");
            assert!(e.contains("SPEC-LIT 100.5"), "{e}");
        }
    }

    #[test]
    fn a_polynomial_is_its_series_piece_by_piece_and_a_shared_end_is_the_lower_piece() {
        let eps = Scalar::EPSILON;
        let e: [Scalar; 4] = [-1.0, 0.0, 0.5, 2.0];
        let lo_c = vec![3.0, 2.0, -0.5, 0.01];
        let hi_c = vec![1.0, 5.0, 0.25, 0.002];
        let p = Property::polynomial(
            "c",
            &e,
            &[
                Piece { lo: 200.0, hi: 1000.0, coefficients: lo_c.clone() },
                Piece { lo: 1000.0, hi: 6000.0, coefficients: hi_c.clone() },
            ],
            100.0,
            2.0,
        )
        .unwrap();
        let f = |c: &[Scalar], t: Scalar| {
            let x = t / 100.0;
            2.0 * (c[0] / x + c[1] + c[2] * x.sqrt() + c[3] * x * x)
        };
        for t in [200.0, 500.0, 1000.0] {
            let got = p.value("c", t).unwrap();
            assert!(
                (got / f(&lo_c, t) - 1.0).abs() <= 16.0 * eps,
                "t = {t}: {got} vs {}",
                f(&lo_c, t)
            );
        }
        for t in [1000.5, 3000.0, 6000.0] {
            let got = p.value("c", t).unwrap();
            assert!(
                (got / f(&hi_c, t) - 1.0).abs() <= 16.0 * eps,
                "t = {t}: {got} vs {}",
                f(&hi_c, t)
            );
        }
        assert!(
            (f(&lo_c, 1000.0) / f(&hi_c, 1000.0) - 1.0).abs() > 0.1,
            "lo = {} hi = {}",
            f(&lo_c, 1000.0),
            f(&hi_c, 1000.0)
        );
    }

    #[test]
    fn sutherland_is_its_value_at_tref_and_the_square_root_of_t_when_s_is_zero() {
        let eps = Scalar::EPSILON;
        let p = Property::sutherland("m", 1.8e-5, 300.0, 110.0, 200.0, 1500.0).unwrap();
        let at_ref = p.value("m", 300.0).unwrap();
        assert!((at_ref / 1.8e-5 - 1.0).abs() <= 4.0 * eps, "at_ref = {at_ref}");
        let q = Property::sutherland("m", 1.8e-5, 300.0, 0.0, 200.0, 1500.0).unwrap();
        let ratio = q.value("m", 1200.0).unwrap() / q.value("m", 300.0).unwrap();
        assert!((ratio - 2.0).abs() <= 16.0 * eps, "ratio = {ratio}");
        assert!(p.value("m", 600.0).unwrap() > p.value("m", 300.0).unwrap());
    }

    #[test]
    fn a_constant_is_its_number_at_every_temperature_and_has_no_range() {
        let c = Property::Constant(148.0);
        for t in [1.0, 300.0, 1.0e6, -5.0] {
            assert_eq!(c.value("k", t).unwrap(), 148.0);
        }
        assert_eq!(c.range(), None);
        assert!(c.is_constant());
        assert!(c.samples().is_empty());
        assert!(c.describe().contains("148"));
        assert!(!Property::table("k", &[(300.0, 1.0), (400.0, 2.0)]).unwrap().is_constant());
    }

    #[test]
    fn samples_cover_every_knot_and_both_ends() {
        let t = Property::table("k", &[(300.0, 1.48), (350.0, 1.19), (400.0, 0.989)]).unwrap();
        assert_eq!(t.samples(), vec![300.0, 350.0, 400.0]);
        let p = Property::polynomial(
            "c",
            &[0.0, 1.0],
            &[Piece { lo: 250.0, hi: 450.0, coefficients: vec![1.0e-5, 1.0e-8] }],
            1.0,
            1.0,
        )
        .unwrap();
        let s = p.samples();
        assert_eq!(s.len(), 17, "s = {s:?}");
        assert_eq!(s[0], 250.0);
        assert_eq!(s[16], 450.0);
        assert!(s.windows(2).all(|w| w[1] > w[0]), "s = {s:?}");
        for x in &s {
            assert!(p.value("c", *x).is_ok(), "x = {x}");
        }
        assert!(p.describe().contains("[250, 450] K"), "{}", p.describe());
    }
}
