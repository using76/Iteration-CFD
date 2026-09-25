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
//! The host evaluator, and its device twin (§100.9) in `cuda/properties.cu`.
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
use crate::{Label, Scalar};
use cudarc::driver::{CudaFunction, PushKernelArg};
use crate::device::{cfg_for, DevBuf, Gpu, KernelSet};

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

    /// `dp/dT` at `t` - (S100.12)'s `S_P` when the property is a source.
    /// The segment and the piece are [`Self::value`]'s; at a table's last
    /// knot it is the last segment's. A constant's is zero. A `t` outside
    /// the range is refused exactly as [`Self::value`] refuses it.
    pub fn slope(&self, setting: &str, t: Scalar) -> Result<Scalar> {
        self.value(setting, t)?;
        Ok(match self {
            Property::Constant(_) => 0.0,
            Property::Table { t: ts, v } => {
                let n = ts.len();
                let k = ts.partition_point(|&x| x <= t).min(n - 1);
                let i = k.max(1) - 1;
                (v[i + 1] - v[i]) / (ts[i + 1] - ts[i])
            }
            Property::Polynomial { exponents, pieces, scale, factor } => {
                let p = pieces.iter().find(|p| t <= p.hi).unwrap_or(&pieces[pieces.len() - 1]);
                let x = t / *scale;
                let mut sum = 0.0 as Scalar;
                for (e, c) in exponents.iter().zip(&p.coefficients) {
                    if *e != 0.0 {
                        sum += *c * *e * power(x, *e - 1.0);
                    }
                }
                *factor * sum / *scale
            }
            Property::Sutherland { value, t_ref, s, .. } => {
                let p = *value * (t / *t_ref).powf(1.5) * (*t_ref + *s) / (t + *s);
                p * (1.5 / t - 1.0 / (t + *s))
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

// ==========================================================================
//  §100.9  The device twin
// ==========================================================================

/// §100.9: the three kernels of `cuda/properties.cu`.
pub struct PropertyKernels {
    table: CudaFunction,
    polynomial: CudaFunction,
    sutherland: CudaFunction,
}

impl PropertyKernels {
    pub fn new(gpu: &Gpu) -> Result<Self> {
        let k = KernelSet::new(gpu, crate::kernels::PROPERTIES)?;
        Ok(Self {
            table: k.func("propertyTable")?,
            polynomial: k.func("propertyPolynomial")?,
            sutherland: k.func("propertySutherland")?,
        })
    }
}

/// A curve's coefficients on the device, in the kernel's layout.
enum DeviceForm {
    Table { t: DevBuf<Scalar>, v: DevBuf<Scalar>, n: Label },
    Polynomial {
        exponents: DevBuf<Scalar>,
        lo: DevBuf<Scalar>,
        hi: DevBuf<Scalar>,
        /// `[n_pieces][n_terms]`, row-major.
        coefficients: DevBuf<Scalar>,
        n_terms: Label,
        n_pieces: Label,
        scale: Scalar,
        factor: Scalar,
    },
    Sutherland { value: Scalar, t_ref: Scalar, s: Scalar },
}

/// §100.9: a curve uploaded once, evaluated elementwise on the device. An
/// evaluation outside the range writes NaN and raises the flag, which the
/// consumer reads between two solves and refuses on (§100.9).
pub struct DeviceProperty {
    form: DeviceForm,
    range: (Scalar, Scalar),
    /// `[1]`: 0 until an evaluation leaves the range, 1 after.
    flag: DevBuf<Scalar>,
}

impl DeviceProperty {
    /// Upload `p`. A constant has no device form - its consumer takes the
    /// constant path - and is refused naming `setting`.
    pub fn upload(gpu: &Gpu, setting: &str, p: &Property) -> Result<Self> {
        let Some(range) = p.range() else {
            return Err(Error::Config(format!(
                "{setting}: a constant has no device form - its consumer takes the \
                 constant path it took before §100 (SPEC-LIT 100.9)"
            )));
        };
        let form = match p {
            Property::Constant(_) => unreachable!("a constant has no range"),
            Property::Table { t, v } => DeviceForm::Table {
                t: gpu.upload(t)?,
                v: gpu.upload(v)?,
                n: t.len() as Label,
            },
            Property::Polynomial { exponents, pieces, scale, factor } => {
                let lo: Vec<Scalar> = pieces.iter().map(|q| q.lo).collect();
                let hi: Vec<Scalar> = pieces.iter().map(|q| q.hi).collect();
                let c: Vec<Scalar> =
                    pieces.iter().flat_map(|q| q.coefficients.iter().copied()).collect();
                DeviceForm::Polynomial {
                    exponents: gpu.upload(exponents)?,
                    lo: gpu.upload(&lo)?,
                    hi: gpu.upload(&hi)?,
                    coefficients: gpu.upload(&c)?,
                    n_terms: exponents.len() as Label,
                    n_pieces: pieces.len() as Label,
                    scale: *scale,
                    factor: *factor,
                }
            }
            Property::Sutherland { value, t_ref, s, .. } => {
                DeviceForm::Sutherland { value: *value, t_ref: *t_ref, s: *s }
            }
        };
        Ok(Self { form, range, flag: gpu.upload(&[0.0 as Scalar])? })
    }

    /// The closed range `[lo, hi]`, K.
    pub fn range(&self) -> (Scalar, Scalar) {
        self.range
    }

    /// `dst[i] = p(t[i])` for `i < n` (§100.9).
    pub fn evaluate(
        &mut self,
        gpu: &Gpu,
        k: &PropertyKernels,
        dst: &mut DevBuf<Scalar>,
        t: &DevBuf<Scalar>,
        n: usize,
    ) -> Result<()> {
        if n == 0 {
            return Ok(());
        }
        let nl = n as Label;
        let (lo, hi) = self.range;
        let Self { form, flag, .. } = self;
        match form {
            DeviceForm::Table { t: ts, v, n: nk } => unsafe {
                gpu.stream()
                    .launch_builder(&k.table)
                    .arg(&mut *dst)
                    .arg(t)
                    .arg(&nl)
                    .arg(&*ts)
                    .arg(&*v)
                    .arg(&*nk)
                    .arg(&mut *flag)
                    .launch(cfg_for(n))?;
            },
            DeviceForm::Polynomial { exponents, lo: plo, hi: phi, coefficients, n_terms, n_pieces, scale, factor } => unsafe {
                gpu.stream()
                    .launch_builder(&k.polynomial)
                    .arg(&mut *dst)
                    .arg(t)
                    .arg(&nl)
                    .arg(&*exponents)
                    .arg(&*n_terms)
                    .arg(&*plo)
                    .arg(&*phi)
                    .arg(&*coefficients)
                    .arg(&*n_pieces)
                    .arg(&*scale)
                    .arg(&*factor)
                    .arg(&mut *flag)
                    .launch(cfg_for(n))?;
            },
            DeviceForm::Sutherland { value, t_ref, s } => unsafe {
                gpu.stream()
                    .launch_builder(&k.sutherland)
                    .arg(&mut *dst)
                    .arg(t)
                    .arg(&nl)
                    .arg(&*value)
                    .arg(&*t_ref)
                    .arg(&*s)
                    .arg(&lo)
                    .arg(&hi)
                    .arg(&mut *flag)
                    .launch(cfg_for(n))?;
            },
        }
        Ok(())
    }

    /// Has any evaluation since the last [`Self::clear_flag`] left the range?
    /// A read-back: between two solves, never inside a captured region.
    pub fn left_range(&self, gpu: &Gpu) -> Result<bool> {
        Ok(gpu.download(&self.flag)?[0] != 0.0)
    }

    /// Lower the flag.
    pub fn clear_flag(&mut self, gpu: &Gpu) -> Result<()> {
        gpu.write(&mut self.flag, &[0.0 as Scalar])
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

    /// SPEC-LIT §100.12: a table's slope is its segment's, the last knot's
    /// the last segment's; a constant's is zero.
    #[test]
    fn the_slope_of_a_table_is_its_segment_s_and_the_last_knot_takes_the_last_segment() {
        let knots: [(Scalar, Scalar); 3] = [(300.0, 148.0), (400.0, 98.9), (600.0, 61.9)];
        let p = Property::table("k", &knots).unwrap();
        let s0: Scalar = (98.9 - 148.0) / (400.0 - 300.0);
        let s1: Scalar = (61.9 - 98.9) / (600.0 - 400.0);
        assert_eq!(p.slope("k", 300.0).unwrap(), s0);
        assert_eq!(p.slope("k", 350.0).unwrap(), s0);
        assert_eq!(p.slope("k", 400.0).unwrap(), s1);
        assert_eq!(p.slope("k", 600.0).unwrap(), s1);
        assert_eq!(Property::Constant(3.0).slope("k", 1.0e4).unwrap(), 0.0);
    }

    /// SPEC-LIT §100.12: a series' and Sutherland's slope is their
    /// derivative, against a central difference.
    #[test]
    #[cfg_attr(feature = "single", ignore = "fails at f32: SPEC-LIT 112.3")]
    fn the_slope_of_a_series_and_of_sutherland_is_their_derivative() {
        let piece = Piece { lo: 200.0, hi: 1000.0, coefficients: vec![3.0e3, 1.0, -2.0e-3, 4.0e-7] };
        let poly = Property::polynomial("cp", &[-1.0, 0.0, 1.0, 2.5], &[piece], 100.0, 1.5).unwrap();
        let suth = Property::sutherland("mu", 1.8e-5, 300.0, 110.4, 200.0, 1000.0).unwrap();
        for p in [&poly, &suth] {
            for t in [250.0 as Scalar, 400.0, 777.0] {
                let h: Scalar = 1.0e-3;
                let fd = (p.value("x", t + h).unwrap() - p.value("x", t - h).unwrap()) / (2.0 * h);
                let s = p.slope("x", t).unwrap();
                println!("{} at {t} K: slope {s:e}, central difference {fd:e}", p.describe());
                assert!((s - fd).abs() <= 1e-7 * s.abs(), "{} at {t} K: {s:e} against {fd:e}", p.describe());
            }
        }
    }

    /// SPEC-LIT §100.12: outside the range the slope is refused as the value is.
    #[test]
    fn a_slope_outside_the_range_is_refused_as_the_value_is() {
        let setting = "regions/die/source";
        let p = Property::table(setting, &[(300.0, 2.0), (600.0, 1.0)]).unwrap();
        for t in [299.0 as Scalar, 601.0] {
            let e = p.slope(setting, t).unwrap_err().to_string();
            assert_eq!(e, p.value(setting, t).unwrap_err().to_string());
            assert!(e.contains(setting), "{e}");
        }
    }

    /// The process's device, if the box has one: every GPU test returns
    /// early rather than failing.
    fn gpu() -> Option<Gpu> {
        Gpu::new(0).ok()
    }

    /// The four curves of §100.9's twin gate, each with the name a failure
    /// prints.
    fn twin_curves() -> Vec<(&'static str, Property)> {
        let table = Property::table(
            "table",
            &[(300.0, 148.0), (350.0, 119.0), (400.0, 98.9), (500.0, 76.2), (600.0, 61.9)],
        )
        .unwrap();
        let nasa = Property::polynomial(
            "nasa",
            &[-2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0],
            &[
                Piece { lo: 200.0, hi: 1000.0, coefficients: vec![2.2e4, -3.8e2, 6.1, -1.1e-2, 1.9e-5, -1.3e-8, 3.6e-12] },
                Piece { lo: 1000.0, hi: 6000.0, coefficients: vec![5.9e5, -2.2e3, 6.6, -6.1e-4, 1.5e-7, -1.9e-11, 1.0e-15] },
            ],
            1.0,
            296.8,
        )
        .unwrap();
        let kadoya = Property::polynomial(
            "kadoya",
            &[1.0, 0.5, 0.0, -1.0, -2.0, -3.0, -4.0],
            &[Piece { lo: 250.0, hi: 1000.0, coefficients: vec![0.128, -0.005, 2.0, -0.4, 0.07, -0.006, 0.0002] }],
            132.5,
            25.9,
        )
        .unwrap();
        let suth = Property::sutherland("sutherland", 1.716e-5, 273.15, 110.4, 200.0, 1000.0).unwrap();
        vec![("table", table), ("nasa", nasa), ("kadoya", kadoya), ("sutherland", suth)]
    }

    /// The device twin (SPEC-LIT 100.9) against the host: four curves, at
    /// both range ends, every knot and piece end, and 4096 random
    /// temperatures, all to `1e-12` relative.
    #[test]
    #[cfg_attr(feature = "single", ignore = "fails at f32: SPEC-LIT 112.3")]
    fn the_device_evaluator_agrees_with_the_host_on_random_temperatures() {
        let Some(g) = gpu() else { return };
        let k = PropertyKernels::new(&g).expect("kernels");
        for (name, p) in twin_curves() {
            let (lo, hi) = p.range().expect("a curve has a range");
            let mut ts: Vec<Scalar> = vec![lo, hi];
            match &p {
                Property::Table { t, .. } => ts.extend_from_slice(t),
                Property::Polynomial { pieces, .. } => {
                    for q in pieces {
                        ts.push(q.lo);
                        ts.push(q.hi);
                    }
                }
                _ => {}
            }
            let mut x: u64 = 0x9E3779B97F4A7C15;
            for _ in 0..4096 {
                x ^= x << 13;
                x ^= x >> 7;
                x ^= x << 17;
                let u = (x >> 11) as Scalar / (1u64 << 53) as Scalar;
                ts.push(lo + (hi - lo) * u);
            }
            let n = ts.len();
            let dt = g.upload(&ts).expect("upload");
            let mut dv = g.zeros::<Scalar>(n).expect("alloc");
            let mut d = DeviceProperty::upload(&g, name, &p).expect("upload");
            d.evaluate(&g, &k, &mut dv, &dt, n).expect("evaluate");
            g.sync().expect("sync");
            let got = g.download(&dv).expect("download");
            let mut worst = 0.0 as Scalar;
            for (i, t) in ts.iter().enumerate() {
                let want = p.value(name, *t).unwrap();
                let rel = (got[i] - want).abs() / want.abs().max(Scalar::MIN_POSITIVE);
                worst = worst.max(rel);
                assert!(
                    rel <= 1e-12,
                    "{name}: at T = {t} K the device gave {} and the host {want} \
                     (relative {rel})",
                    got[i]
                );
            }
            println!("  {name}: worst relative difference {worst:e}");
            assert!(!d.left_range(&g).unwrap(), "{name}: nothing left the range");
        }
    }

    /// Outside the range the device writes NaN and raises the flag (SPEC-LIT
    /// 100.9); `clear_flag` lowers it again, and a constant is refused by
    /// name.
    #[test]
    fn an_evaluation_outside_the_range_writes_nan_and_raises_the_flag() {
        let Some(g) = gpu() else { return };
        let k = PropertyKernels::new(&g).expect("kernels");
        for (name, p) in twin_curves() {
            let (lo, hi) = p.range().expect("a curve has a range");
            let ts: Vec<Scalar> = vec![lo - 1.0, 0.5 * (lo + hi), hi + 1.0];
            let dt = g.upload(&ts).expect("upload");
            let mut dv = g.zeros::<Scalar>(3).expect("alloc");
            let mut d = DeviceProperty::upload(&g, name, &p).expect("upload");
            d.evaluate(&g, &k, &mut dv, &dt, 3).expect("evaluate");
            g.sync().expect("sync");
            let got = g.download(&dv).expect("download");
            assert!(got[0].is_nan(), "{name}: below the range the device writes NaN");
            assert!(got[2].is_nan(), "{name}: above the range the device writes NaN");
            assert!(got[1].is_finite(), "{name}: inside the range the value is finite");
            assert!(d.left_range(&g).unwrap(), "{name}: the flag is up");
            d.clear_flag(&g).expect("clear");
            assert!(!d.left_range(&g).unwrap(), "{name}: the flag is down again");
        }
        let Err(e) = DeviceProperty::upload(&g, "regions/x/fluid/kappa", &Property::Constant(0.026))
        else {
            panic!("a constant has no device form, SPEC-LIT 100.9 refuses it by name");
        };
        let m = format!("{e}");
        for what in ["regions/x/fluid/kappa", "constant", "SPEC-LIT 100.9"] {
            assert!(m.contains(what), "'{what}' is not in the refusal: {m}");
        }
    }

    /// SPEC-LIT 81.7's row for `src/properties.rs`: the evaluation captures
    /// and replays bitwise, its result feeding the next iteration's
    /// temperatures.
    #[test]
    fn the_property_evaluation_replays_bitwise() {
        let Some(g) = gpu() else { return };
        let p = twin_curves().into_iter().find(|(n, _)| *n == "table").unwrap().1;
        let n = 256;
        let k = PropertyKernels::new(&g).expect("kernels");
        let fk = crate::field_ops::FieldKernels::new(&g).expect("field kernels");
        let t0: Vec<Scalar> = (0..n).map(|i| 300.0 + 200.0 * i as Scalar / n as Scalar).collect();
        let report = crate::capture::capture_replays_bitwise(
            &g,
            "the device evaluator (SPEC-LIT 100.9)",
            || {
                let d = DeviceProperty::upload(&g, "table", &p)?;
                Ok((d, g.upload(&t0)?, g.zeros::<Scalar>(n)?))
            },
            |(d, t, v)| {
                d.evaluate(&g, &k, v, t, n)?;
                // The result, scaled to at most 0.015 K, feeds the next
                // iteration's temperatures, so a replay that skipped the
                // evaluation cannot pass.
                crate::field_ops::scale_field(&g, &fk, v, 1.0e-4, n)?;
                crate::field_ops::add_field(&g, &fk, t, v, n)
            },
            |(d, t, v)| {
                Ok(vec![
                    ("t", g.download(t)?),
                    ("v", g.download(v)?),
                    ("flag", g.download(&d.flag)?),
                ])
            },
        )
        .expect("SPEC-LIT 81.7: the device evaluator must capture and replay bitwise");
        println!("  property evaluation: {report}");
    }
}
