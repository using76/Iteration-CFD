// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.
// Provenance: see PROVENANCE.md. No GPL-licensed source was consulted.

//! SPEC-LIT 120.12 - the synthetic LES inflow: `turbulentDigitalFilterInlet`
//! and `turbulentDFSEMInlet`, a mean velocity plus a fluctuation with a
//! prescribed Reynolds stress `R`, drawn once per TIME step.
//!
//! Both are a §4 FIXED VALUE (`fr = 1`, `refGrad = 0`) on a vector field
//! whose `refValue` a kernel rewrites - `fldDigitalFilterInlet` or
//! `fldDfsemInlet`, then the one-thread `fldSyntheticStep`, which moves the
//! rule's device state `[n, X]` (the step and the convected distance) on, so
//! a captured graph advances it (SPEC-LIT 81). The functions here are their
//! host twins, in the same operation order and in f64; each face value
//! rounds once into `Scalar`. The device contracts FMAs and its `exp` is not
//! the host's, so the two agree to round-off, not bitwise.
//!
//! The digital filter is Klein, Sadiki & Janicka (2003),
//! doi:10.1016/S0021-9991(03)00090-1 - the Gaussian filter (120.12c) applied
//! to a random lattice in the two directions of the inlet plane and, through
//! Taylor's hypothesis, in time (120.12d) - followed by the Cholesky
//! transformation of Lund, Wu & Squires (1998), doi:10.1006/jcph.1998.5882
//! (120.12e). The divergence-free eddies follow the idea of Poletto, Craft &
//! Revell (2013), doi:10.1007/s10494-013-9488-2 - a sum of eddies, each a
//! radial shape times `r x alpha`, divergence-free by construction, with the
//! intensities from the Reynolds stress - but the spherical shape (120.12g),
//! its constant (120.12i) and the intensity rule (120.12j) are DESIGN,
//! derived in SPEC-LIT 120.12. The draw (120.12a) chains the SplitMix64
//! finaliser of Steele, Lea & Flood (2014), doi:10.1145/2660193.2660195.
//!
//! No GPL-licensed source was consulted.

// `!(x > 0.0)` is the NaN-refusing test on purpose, and the index loops
// keep the host twins in the kernels' component order.
#![allow(clippy::neg_cmp_op_on_partial_ord, clippy::needless_range_loop)]

use std::collections::BTreeMap;
use std::f64::consts::PI;

use crate::device::{DevBuf, Gpu};
use crate::error::{Error, Result};
use crate::types::DVec3;
use crate::{Label, Scalar, Vec3};

/// The largest filter half-width `N` of (120.12c) in any direction.
pub const DF_MAX_HALF: usize = 128;
/// The largest number of taps `(2N_t+1)(2N_y+1)(2N_z+1)` of one component's
/// (120.12d) sum, per face.
pub const DF_MAX_TAPS: usize = 1 << 22;
/// The largest `nEddy` of (120.12g).
pub const DFSEM_MAX_EDDIES: usize = 1 << 20;
/// (120.12i)'s `G = (4 pi / 3)(128 / 15015) = 512 pi / 45045`.
pub const DFSEM_G: f64 = 512.0 * PI / 45045.0;
/// (120.12a)'s starting word.
const KEY0: u64 = 0x243F_6A88_85A3_08D3;
/// `sqrt(3)`: (120.12a)'s `eps` has unit variance.
const SQRT_3: f64 = 1.7320508075688772;

/// The SplitMix64 finaliser of `z + 0x9E3779B97F4A7C15`.
#[must_use]
pub fn mix(z: u64) -> u64 {
    let mut z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// (120.12a): `u` in `[0, 1)` for the key `(seed, tag, a, b, c, d)` - the
/// finaliser chained over the six words, its top 53 bits.
#[must_use]
pub fn unit(seed: u64, tag: u64, a: u64, b: u64, c: u64, d: u64) -> f64 {
    let mut h = mix(seed ^ KEY0);
    h = mix(h ^ tag);
    h = mix(h ^ a);
    h = mix(h ^ b);
    h = mix(h ^ c);
    h = mix(h ^ d);
    (h >> 11) as f64 * (1.0 / 9_007_199_254_740_992.0)
}

/// (120.12a): `sqrt(3) (2u - 1)`, zero mean and unit variance.
#[must_use]
pub fn eps(seed: u64, tag: u64, a: u64, b: u64, c: u64, d: u64) -> f64 {
    SQRT_3 * (2.0 * unit(seed, tag, a, b, c, d) - 1.0)
}

/// A symmetric tensor `(xx xy xz yy yz zz)`.
pub type Sym6 = [f64; 6];

/// The lower Cholesky factor `(l00 l10 l11 l20 l21 l22)` of a positive
/// SEMI-definite `m` - (120.12e) and (120.12j). A zero pivot whose column is
/// zero (to `1e-14 tr m`) gives a zero column; `None` when `m` is not
/// positive semi-definite or not finite.
#[must_use]
pub fn cholesky(m: Sym6) -> Option<Sym6> {
    if m.iter().any(|x| !x.is_finite()) {
        return None;
    }
    let tol = 1e-14 * (m[0].abs() + m[3].abs() + m[5].abs()).max(f64::MIN_POSITIVE);
    let pivot = |d: f64| -> Option<f64> {
        if d > tol {
            Some(d.sqrt())
        } else if d.abs() <= tol {
            Some(0.0)
        } else {
            None
        }
    };
    let below = |num: f64, piv: f64| -> Option<f64> {
        if piv > 0.0 {
            Some(num / piv)
        } else if num.abs() <= tol {
            Some(0.0)
        } else {
            None
        }
    };
    let l00 = pivot(m[0])?;
    let l10 = below(m[1], l00)?;
    let l20 = below(m[2], l00)?;
    let l11 = pivot(m[3] - l10 * l10)?;
    let l21 = below(m[4] - l10 * l20, l11)?;
    let l22 = pivot(m[5] - l20 * l20 - l21 * l21)?;
    Some([l00, l10, l11, l20, l21, l22])
}

/// `l v` for a lower factor of [`cholesky`] - (120.12e)'s `a u'` and
/// (120.12j)'s `C eps`.
#[must_use]
pub fn lower_times(l: &Sym6, v: [f64; 3]) -> [f64; 3] {
    [
        l[0] * v[0],
        l[1] * v[0] + l[2] * v[1],
        l[3] * v[0] + l[4] * v[1] + l[5] * v[2],
    ]
}

/// (120.12c): the normalised Gaussian filter of `n = L / Delta` cells,
/// `2N + 1` coefficients `b_{-N..N}`, `N = ceil(2n)`. `n = 0` (or not
/// positive) is the identity, `[1]`.
#[must_use]
pub fn filter(n: f64) -> Vec<f64> {
    if !(n > 0.0) || !n.is_finite() {
        return vec![1.0];
    }
    let half = half_width(n);
    let raw: Vec<f64> = (-(half as i64)..=half as i64)
        .map(|k| (-(PI * (k * k) as f64) / (2.0 * n * n)).exp())
        .collect();
    let mut s = 0.0f64;
    for b in &raw {
        s += b * b;
    }
    let norm = s.sqrt();
    raw.iter().map(|b| b / norm).collect()
}

/// (120.12c)'s `N = ceil(2n)`, 0 for `n` not positive.
#[must_use]
pub fn half_width(n: f64) -> usize {
    if !(n > 0.0) || !n.is_finite() {
        0
    } else {
        (2.0 * n).ceil() as usize
    }
}

/// (120.12b): the patch frame - the inward normal `nin`, the in-plane axes
/// `e_y`, `e_z`, the origin `x0`, and each face's `(eta, zeta)` and offset
/// `x_f - x0`.
#[derive(Clone, Debug, PartialEq)]
pub struct PatchFrame {
    pub nin: [f64; 3],
    pub ey: [f64; 3],
    pub ez: [f64; 3],
    pub x0: [f64; 3],
    pub rel: Vec<[f64; 3]>,
    pub eta: Vec<f64>,
    pub zeta: Vec<f64>,
}

fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn d3(v: DVec3) -> [f64; 3] {
    [v.x, v.y, v.z]
}

/// A refusal of the synthetic inflow, naming the setting and SPEC-LIT 120.12.
fn refused(field: &str, setting: &str, why: &str) -> Error {
    Error::Field {
        field: field.to_string(),
        msg: format!("{setting}: {why} (SPEC-LIT 120.12)"),
    }
}

/// (120.12b) from the f64 face centres, unit outward normals and areas. A
/// patch with no area, or whose faces leave its mean plane by more than
/// `1e-6` of its diagonal, is refused by name.
pub fn patch_frame(
    centres: &[DVec3],
    normals: &[DVec3],
    areas: &[f64],
    field: &str,
    base: &str,
) -> Result<PatchFrame> {
    let n = centres.len();
    if n == 0 {
        return Err(refused(field, base, "the patch has no faces"));
    }
    let mut m = [0.0f64; 3];
    let mut x0 = [0.0f64; 3];
    for i in 0..n {
        let (nv, a, c) = (d3(normals[i]), areas.get(i).copied().unwrap_or(0.0), d3(centres[i]));
        for k in 0..3 {
            m[k] += a * nv[k];
            x0[k] += c[k];
        }
    }
    let mm = dot(m, m).sqrt();
    if !(mm > 0.0) {
        return Err(refused(field, base, "the patch has no net area, so it has no inflow normal"));
    }
    let nin = [-m[0] / mm, -m[1] / mm, -m[2] / mm];
    for v in &mut x0 {
        *v /= n as f64;
    }
    let mut big = 0usize;
    for k in 1..3 {
        if nin[k].abs() > nin[big].abs() {
            big = k;
        }
    }
    let first = if big == 0 { 1 } else { 0 };
    let mut a = [0.0f64; 3];
    a[first] = 1.0;
    let an = dot(a, nin);
    let mut ey = [a[0] - an * nin[0], a[1] - an * nin[1], a[2] - an * nin[2]];
    let el = dot(ey, ey).sqrt();
    for v in &mut ey {
        *v /= el;
    }
    let ez = cross(nin, ey);
    let mut rel = Vec::with_capacity(n);
    let (mut eta, mut zeta, mut s) = (Vec::with_capacity(n), Vec::with_capacity(n), Vec::with_capacity(n));
    for c in centres {
        let c = d3(*c);
        let r = [c[0] - x0[0], c[1] - x0[1], c[2] - x0[2]];
        eta.push(dot(r, ey));
        zeta.push(dot(r, ez));
        s.push(dot(r, nin));
        rel.push(r);
    }
    let span = |v: &[f64]| v.iter().cloned().fold(f64::NEG_INFINITY, f64::max) - v.iter().cloned().fold(f64::INFINITY, f64::min);
    let diag = (span(&eta).powi(2) + span(&zeta).powi(2)).sqrt();
    let off = s.iter().fold(0.0f64, |acc, x| acc.max(x.abs()));
    if off > 1e-6 * diag.max(f64::MIN_POSITIVE) {
        return Err(refused(
            field,
            base,
            &format!("the patch is not planar - a face leaves its mean plane by {off:e}, more than 1e-6 of its diagonal {diag:e}"),
        ));
    }
    Ok(PatchFrame { nin, ey, ez, x0, rel, eta, zeta })
}

/// `n` reals from an entry: `(a b c ...)`, optionally after `uniform` or
/// `constant`; a single value may stand without parentheses.
fn reals(entries: &BTreeMap<String, String>, key: &str, n: usize, field: &str, base: &str) -> Result<Option<Vec<f64>>> {
    let Some(raw) = entries.get(key) else {
        return Ok(None);
    };
    let setting = format!("{base}/{key}");
    let mut t = raw.trim();
    for p in ["uniform", "constant"] {
        if let Some(rest) = t.strip_prefix(p) {
            t = rest.trim();
        }
    }
    let t = t.trim_end_matches(';').trim();
    let inner = t.strip_prefix('(').and_then(|x| x.strip_suffix(')')).unwrap_or(t);
    let v: Option<Vec<f64>> = inner.split_whitespace().map(|w| w.parse::<f64>().ok()).collect();
    match v {
        Some(v) if v.len() == n && v.iter().all(|x| x.is_finite()) => Ok(Some(v)),
        _ => Err(refused(field, &setting, &format!("`{}` is not {n} finite constant numbers", raw.trim()))),
    }
}

fn required(entries: &BTreeMap<String, String>, key: &str, n: usize, field: &str, base: &str, name: &str) -> Result<Vec<f64>> {
    reals(entries, key, n, field, base)?
        .ok_or_else(|| refused(field, &format!("{base}/{key}"), &format!("`{name}` needs a `{key}` entry and the patch has none")))
}

fn seed_of(entries: &BTreeMap<String, String>, field: &str, base: &str) -> Result<u64> {
    match entries.get("seed") {
        None => Ok(0),
        Some(v) => v.trim().trim_end_matches(';').trim().parse::<u64>().map_err(|_| {
            refused(field, &format!("{base}/seed"), &format!("`{}` is not a non-negative integer", v.trim()))
        }),
    }
}

/// `R`, `UMean` and `Ubulk` (default `UMean . nin`, which must be positive).
fn common(
    entries: &BTreeMap<String, String>,
    frame: &PatchFrame,
    field: &str,
    base: &str,
    name: &str,
) -> Result<(Sym6, [f64; 3], f64)> {
    let r = required(entries, "R", 6, field, base, name)?;
    let um = required(entries, "UMean", 3, field, base, name)?;
    let umean = [um[0], um[1], um[2]];
    let ubulk = match reals(entries, "Ubulk", 1, field, base)? {
        Some(v) => v[0],
        None => dot(umean, frame.nin),
    };
    if !(ubulk > 0.0) {
        return Err(refused(
            field,
            &format!("{base}/Ubulk"),
            &format!("the bulk velocity {ubulk:e} is not positive - the fluctuation is carried INTO the domain at Ubulk"),
        ));
    }
    Ok(([r[0], r[1], r[2], r[3], r[4], r[5]], umean, ubulk))
}

// ==========================================================================
//  (120.12c)-(120.12f): the digital filter
// ==========================================================================

/// One `turbulentDigitalFilterInlet` patch (SPEC-LIT 120.12), built at setup.
#[derive(Clone, Debug, PartialEq)]
pub struct DigitalFilterRule {
    pub start: usize,
    pub size: usize,
    /// `[size]` each face's lattice node `(j, k)` in `(e_y, e_z)`.
    pub nodes: Vec<[i64; 2]>,
    /// (120.12e)'s `a`, the lower Cholesky factor of `R`.
    pub chol: Sym6,
    pub umean: [f64; 3],
    pub ubulk: f64,
    /// `L[c][0]`, the streamwise length of component `c` - the time filter.
    pub lx: [f64; 3],
    /// (120.12c)'s coefficients of component `c` along `e_y` and `e_z`.
    pub by: [Vec<f64>; 3],
    pub bz: [Vec<f64>; 3],
    pub seed: u64,
}

/// (120.12c)-(120.12d)'s time filters for a step of `dt`: `n_t = L_x /
/// (Ubulk dt)`. Refused by name when a half-width exceeds [`DF_MAX_HALF`] or a
/// component's taps exceed [`DF_MAX_TAPS`].
pub fn df_time_filters(rule: &DigitalFilterRule, dt: f64) -> Result<[Vec<f64>; 3]> {
    if !(dt > 0.0) || !dt.is_finite() {
        return Err(Error::Config(format!(
            "turbulentDigitalFilterInlet: the time step {dt:e} is not positive (SPEC-LIT 120.12)"
        )));
    }
    let mut out: [Vec<f64>; 3] = [Vec::new(), Vec::new(), Vec::new()];
    for c in 0..3 {
        let n = rule.lx[c] / (rule.ubulk * dt);
        let half = half_width(n);
        let taps = (2 * half + 1) * rule.by[c].len() * rule.bz[c].len();
        if half > DF_MAX_HALF || taps > DF_MAX_TAPS {
            return Err(Error::Config(format!(
                "turbulentDigitalFilterInlet: component {c}'s time filter needs N_t = {half} \
                 ({taps} taps a face) at dt {dt:e}, over the limits {DF_MAX_HALF} and {DF_MAX_TAPS} - \
                 raise dt or shorten L (SPEC-LIT 120.12)"
            )));
        }
        out[c] = filter(n);
    }
    Ok(out)
}

/// (120.12d)-(120.12e) at step `n`, lattice node `(j, k)`: `a u'`, the
/// fluctuation without the mean, in f64 - the host twin of one thread of
/// `fldDigitalFilterInlet`.
#[must_use]
pub fn df_fluctuation(rule: &DigitalFilterRule, bt: &[Vec<f64>; 3], n: i64, j: i64, k: i64) -> [f64; 3] {
    let mut up = [0.0f64; 3];
    for c in 0..3 {
        let (t, y, z) = (&bt[c], &rule.by[c], &rule.bz[c]);
        let (ht, hy, hz) = ((t.len() / 2) as i64, (y.len() / 2) as i64, (z.len() / 2) as i64);
        let mut sum = 0.0f64;
        for (ia, ba) in t.iter().enumerate() {
            let m = n + ia as i64 - ht;
            let mut sa = 0.0f64;
            for (ib, bb) in y.iter().enumerate() {
                let jj = j + ib as i64 - hy;
                let mut sb = 0.0f64;
                for (id, bd) in z.iter().enumerate() {
                    let kk = k + id as i64 - hz;
                    sb += bd * eps(rule.seed, 1, m as u64, jj as u64, kk as u64, c as u64);
                }
                sa += bb * sb;
            }
            sum += ba * sa;
        }
        up[c] = sum;
    }
    lower_times(&rule.chol, up)
}

/// One `fldDigitalFilterInlet` launch at step `n` with step `dt`: every face
/// of the rule gets `UMean + a u'` rounded once, into `ref_value`
/// (`[n_bf]`).
pub fn df_values(rule: &DigitalFilterRule, n: i64, dt: f64, ref_value: &mut [Vec3]) -> Result<()> {
    let bt = df_time_filters(rule, dt)?;
    for i in 0..rule.size {
        let [j, k] = rule.nodes[i];
        let u = df_fluctuation(rule, &bt, n, j, k);
        ref_value[rule.start + i] = Vec3::new(
            (rule.umean[0] + u[0]) as Scalar,
            (rule.umean[1] + u[1]) as Scalar,
            (rule.umean[2] + u[2]) as Scalar,
        );
    }
    Ok(())
}

/// The device parameter block of a digital-filter rule (f64): `[0..6)` the
/// factor, `[6..9)` UMean, `[9]` Ubulk, `[10..13)` L_x per component,
/// `[13..16)` / `[16..19)` the half-widths N_y / N_z, `[19..22)` /
/// `[22..25)` the offsets of the e_y / e_z tables, then the tables.
#[must_use]
pub fn df_params(rule: &DigitalFilterRule) -> Vec<f64> {
    let mut p = vec![0.0f64; 25];
    p[..6].copy_from_slice(&rule.chol);
    p[6..9].copy_from_slice(&rule.umean);
    p[9] = rule.ubulk;
    p[10..13].copy_from_slice(&rule.lx);
    for c in 0..3 {
        p[13 + c] = (rule.by[c].len() / 2) as f64;
        p[16 + c] = (rule.bz[c].len() / 2) as f64;
    }
    for c in 0..3 {
        p[19 + c] = p.len() as f64;
        let t = rule.by[c].clone();
        p.extend(t);
    }
    for c in 0..3 {
        p[22 + c] = p.len() as f64;
        let t = rule.bz[c].clone();
        p.extend(t);
    }
    p
}

/// SPEC-LIT 120.12: build a digital-filter rule from the patch entries.
/// Returns the rule and the seed face values, `UMean` - the first time
/// step's `dt` sets the time filter, so nothing is drawn before it.
pub fn build_digital_filter(
    entries: &BTreeMap<String, String>,
    frame: &PatchFrame,
    start: usize,
    field: &str,
    base: &str,
) -> Result<(DigitalFilterRule, Vec<Vec3>)> {
    let name = "turbulentDigitalFilterInlet";
    let (r, umean, ubulk) = common(entries, frame, field, base, name)?;
    let chol = cholesky(r).ok_or_else(|| {
        refused(field, &format!("{base}/R"), "`R` is not positive semi-definite, so it is no Reynolds stress")
    })?;
    let l = required(entries, "L", 9, field, base, name)?;
    if l.iter().any(|x| *x < 0.0) {
        return Err(refused(field, &format!("{base}/L"), "a length of `L` is negative"));
    }
    let dv = required(entries, "planeDivisions", 2, field, base, name)?;
    let div = |x: f64| -> Option<i64> { (x >= 1.0 && x.fract() == 0.0 && x <= 1e6).then_some(x as i64) };
    let (Some(ny), Some(nz)) = (div(dv[0]), div(dv[1])) else {
        return Err(refused(field, &format!("{base}/planeDivisions"), "`planeDivisions` must be two integers from 1 to 1e6"));
    };
    let lo_hi = |v: &[f64]| (v.iter().cloned().fold(f64::INFINITY, f64::min), v.iter().cloned().fold(f64::NEG_INFINITY, f64::max));
    let (ylo, yhi) = lo_hi(&frame.eta);
    let (zlo, zhi) = lo_hi(&frame.zeta);
    let step = |lo: f64, hi: f64, n: i64, what: &str| -> Result<f64> {
        if n == 1 {
            return Ok(0.0);
        }
        let d = (hi - lo) / (n - 1) as f64;
        if d > 0.0 {
            Ok(d)
        } else {
            Err(refused(field, &format!("{base}/planeDivisions"), &format!("{n} divisions along {what}, but the patch has no extent there")))
        }
    };
    let dy = step(ylo, yhi, ny, "e_y")?;
    let dz = step(zlo, zhi, nz, "e_z")?;
    let node = |v: f64, lo: f64, d: f64, n: i64| -> i64 {
        if d > 0.0 {
            (((v - lo) / d).round() as i64).clamp(0, n - 1)
        } else {
            0
        }
    };
    let nodes: Vec<[i64; 2]> = (0..frame.eta.len())
        .map(|i| [node(frame.eta[i], ylo, dy, ny), node(frame.zeta[i], zlo, dz, nz)])
        .collect();
    let mut by: [Vec<f64>; 3] = [Vec::new(), Vec::new(), Vec::new()];
    let mut bz: [Vec<f64>; 3] = [Vec::new(), Vec::new(), Vec::new()];
    for c in 0..3 {
        let ny_c = if dy > 0.0 { l[3 * c + 1] / dy } else { 0.0 };
        let nz_c = if dz > 0.0 { l[3 * c + 2] / dz } else { 0.0 };
        if half_width(ny_c) > DF_MAX_HALF || half_width(nz_c) > DF_MAX_HALF {
            return Err(refused(
                field,
                &format!("{base}/L"),
                &format!("component {c}'s filter is wider than {DF_MAX_HALF} lattice cells - shorten L or use fewer planeDivisions"),
            ));
        }
        by[c] = filter(ny_c);
        bz[c] = filter(nz_c);
    }
    let seed = seed_of(entries, field, base)?;
    let rule = DigitalFilterRule {
        start,
        size: frame.eta.len(),
        nodes,
        chol,
        umean,
        ubulk,
        lx: [l[0], l[3], l[6]],
        by,
        bz,
        seed,
    };
    let v = Vec3::new(umean[0] as Scalar, umean[1] as Scalar, umean[2] as Scalar);
    Ok((rule, vec![v; frame.eta.len()]))
}

// ==========================================================================
//  (120.12g)-(120.12k): the divergence-free eddies
// ==========================================================================

/// One `turbulentDFSEMInlet` patch (SPEC-LIT 120.12), built at setup.
#[derive(Clone, Debug, PartialEq)]
pub struct DfsemRule {
    pub start: usize,
    pub size: usize,
    /// `[size]` `x_f - x0`.
    pub rel: Vec<[f64; 3]>,
    /// (120.12j)'s `C`, the lower Cholesky factor of `A`.
    pub calpha: Sym6,
    pub umean: [f64; 3],
    pub ubulk: f64,
    pub nin: [f64; 3],
    pub ey: [f64; 3],
    pub ez: [f64; 3],
    pub sigma: f64,
    /// (120.12k)'s box across the plane: `eta in [eta_lo, eta_lo + eta_w)`.
    pub eta_lo: f64,
    pub eta_w: f64,
    pub zeta_lo: f64,
    pub zeta_w: f64,
    /// (120.12g)'s `sqrt(V_B / (N sigma^3))`.
    pub scale: f64,
    pub n_eddy: usize,
    pub seed: u64,
}

/// One eddy of (120.12k): its centre relative to `x0`, and `alpha`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Eddy {
    pub pos: [f64; 3],
    pub alpha: [f64; 3],
}

/// (120.12k): eddy `k` after the convected distance `x`.
#[must_use]
pub fn dfsem_eddy(rule: &DfsemRule, k: usize, x: f64) -> Eddy {
    let sg = rule.sigma;
    let two = 2.0 * sg;
    let kk = k as u64;
    let s0 = -sg + two * unit(rule.seed, 2, kk, 0, 0, 0);
    let w = s0 + sg + x;
    let g = (w / two).floor();
    let s = -sg + (w - two * g);
    let gg = g as i64 as u64;
    let eta = rule.eta_lo + rule.eta_w * unit(rule.seed, 2, kk, gg, 1, 0);
    let zeta = rule.zeta_lo + rule.zeta_w * unit(rule.seed, 2, kk, gg, 2, 0);
    let e = [
        eps(rule.seed, 2, kk, gg, 3, 0),
        eps(rule.seed, 2, kk, gg, 4, 0),
        eps(rule.seed, 2, kk, gg, 5, 0),
    ];
    let alpha = lower_times(&rule.calpha, e);
    let mut pos = [0.0f64; 3];
    for c in 0..3 {
        pos[c] = s * rule.nin[c] + eta * rule.ey[c] + zeta * rule.ez[c];
    }
    Eddy { pos, alpha }
}

/// Every eddy at the convected distance `x`.
#[must_use]
pub fn dfsem_eddies(rule: &DfsemRule, x: f64) -> Vec<Eddy> {
    (0..rule.n_eddy).map(|k| dfsem_eddy(rule, k, x)).collect()
}

/// (120.12g): `u'` at the point `p` (relative to `x0`), in f64.
#[must_use]
pub fn dfsem_velocity(rule: &DfsemRule, eddies: &[Eddy], p: [f64; 3]) -> [f64; 3] {
    let mut u = [0.0f64; 3];
    for e in eddies {
        let d = [
            (p[0] - e.pos[0]) / rule.sigma,
            (p[1] - e.pos[1]) / rule.sigma,
            (p[2] - e.pos[2]) / rule.sigma,
        ];
        let q = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
        if q < 1.0 {
            let f = (1.0 - q) * (1.0 - q);
            let c = cross(d, e.alpha);
            for k in 0..3 {
                u[k] += f * c[k];
            }
        }
    }
    [rule.scale * u[0], rule.scale * u[1], rule.scale * u[2]]
}

/// The analytic gradient `g[i][m] = d u'_i / d x_m` of (120.12g) at `p` -
/// (120.12h)'s two terms, `2 F'(q) d_m (d x alpha)_i / sigma` and
/// `F eps_iml alpha_l / sigma`.
#[must_use]
pub fn dfsem_gradient(rule: &DfsemRule, eddies: &[Eddy], p: [f64; 3]) -> [[f64; 3]; 3] {
    let mut g = [[0.0f64; 3]; 3];
    let sg = rule.sigma;
    for e in eddies {
        let d = [(p[0] - e.pos[0]) / sg, (p[1] - e.pos[1]) / sg, (p[2] - e.pos[2]) / sg];
        let q = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
        if q >= 1.0 {
            continue;
        }
        let f = (1.0 - q) * (1.0 - q);
        let fp = -2.0 * (1.0 - q);
        let c = cross(d, e.alpha);
        let a = e.alpha;
        let m = [[0.0, a[2], -a[1]], [-a[2], 0.0, a[0]], [a[1], -a[0], 0.0]];
        for i in 0..3 {
            for mm in 0..3 {
                g[i][mm] += (2.0 * fp * d[mm] * c[i] + f * m[i][mm]) / sg;
            }
        }
    }
    for row in &mut g {
        for v in row.iter_mut() {
            *v *= rule.scale;
        }
    }
    g
}

/// One `fldDfsemInlet` launch at the convected distance `x`: every face gets
/// `UMean + u'(x_f)` rounded once, into `ref_value` (`[n_bf]`).
pub fn dfsem_values(rule: &DfsemRule, x: f64, ref_value: &mut [Vec3]) {
    let eddies = dfsem_eddies(rule, x);
    for i in 0..rule.size {
        let u = dfsem_velocity(rule, &eddies, rule.rel[i]);
        ref_value[rule.start + i] = Vec3::new(
            (rule.umean[0] + u[0]) as Scalar,
            (rule.umean[1] + u[1]) as Scalar,
            (rule.umean[2] + u[2]) as Scalar,
        );
    }
}

/// (120.12k)'s `X^(n) = fma(Ubulk, dt, X^(n-1))`, as `fldSyntheticStep`.
#[must_use]
pub fn convect(x: f64, ubulk: f64, dt: f64) -> f64 {
    ubulk.mul_add(dt, x)
}

/// The device parameter block of a DFSEM rule (f64): `[0..6)` C, `[6..9)`
/// UMean, `[9]` Ubulk, `[10..13)` nin, `[13..16)` e_y, `[16..19)` e_z,
/// `[19]` sigma, `[20..24)` eta_lo, eta_w, zeta_lo, zeta_w, `[24]` scale,
/// `[25]` nEddy.
#[must_use]
pub fn dfsem_params(rule: &DfsemRule) -> Vec<f64> {
    let mut p = vec![0.0f64; 26];
    p[..6].copy_from_slice(&rule.calpha);
    p[6..9].copy_from_slice(&rule.umean);
    p[9] = rule.ubulk;
    p[10..13].copy_from_slice(&rule.nin);
    p[13..16].copy_from_slice(&rule.ey);
    p[16..19].copy_from_slice(&rule.ez);
    p[19] = rule.sigma;
    p[20] = rule.eta_lo;
    p[21] = rule.eta_w;
    p[22] = rule.zeta_lo;
    p[23] = rule.zeta_w;
    p[24] = rule.scale;
    p[25] = rule.n_eddy as f64;
    p
}

/// SPEC-LIT 120.12: build a DFSEM rule from the patch entries. Returns the
/// rule and the seed face values, (120.12g) at `X = 0`.
pub fn build_dfsem(
    entries: &BTreeMap<String, String>,
    frame: &PatchFrame,
    start: usize,
    field: &str,
    base: &str,
) -> Result<(DfsemRule, Vec<Vec3>)> {
    let name = "turbulentDFSEMInlet";
    let (r, umean, ubulk) = common(entries, frame, field, base, name)?;
    let sigma = required(entries, "sigma", 1, field, base, name)?[0];
    if !(sigma > 0.0) {
        return Err(refused(field, &format!("{base}/sigma"), "the eddy radius `sigma` is not positive"));
    }
    // (120.12j): A = (tr R / 2 I - R) / G.
    let h = 0.5 * (r[0] + r[3] + r[5]);
    let a = [
        (h - r[0]) / DFSEM_G,
        -r[1] / DFSEM_G,
        -r[2] / DFSEM_G,
        (h - r[3]) / DFSEM_G,
        -r[4] / DFSEM_G,
        (h - r[5]) / DFSEM_G,
    ];
    let calpha = cholesky(a).ok_or_else(|| {
        refused(
            field,
            &format!("{base}/R"),
            "`R` is not realisable by divergence-free eddies - every principal stress must be at most \
             the sum of the other two (120.12j)",
        )
    })?;
    let lo_hi = |v: &[f64]| (v.iter().cloned().fold(f64::INFINITY, f64::min), v.iter().cloned().fold(f64::NEG_INFINITY, f64::max));
    let (ylo, yhi) = lo_hi(&frame.eta);
    let (zlo, zhi) = lo_hi(&frame.zeta);
    let eta_w = (yhi - ylo) + 2.0 * sigma;
    let zeta_w = (zhi - zlo) + 2.0 * sigma;
    let vb = 2.0 * sigma * eta_w * zeta_w;
    let n_eddy = match reals(entries, "nEddy", 1, field, base)? {
        Some(v) => {
            let x = v[0];
            if !(x >= 1.0 && x.fract() == 0.0) {
                return Err(refused(field, &format!("{base}/nEddy"), "`nEddy` must be a positive integer"));
            }
            x as usize
        }
        None => ((vb / (sigma * sigma * sigma)).ceil() as usize).max(1),
    };
    if n_eddy > DFSEM_MAX_EDDIES {
        return Err(refused(
            field,
            &format!("{base}/nEddy"),
            &format!("{n_eddy} eddies is over the limit {DFSEM_MAX_EDDIES} - raise sigma or set nEddy"),
        ));
    }
    let seed = seed_of(entries, field, base)?;
    let rule = DfsemRule {
        start,
        size: frame.rel.len(),
        rel: frame.rel.clone(),
        calpha,
        umean,
        ubulk,
        nin: frame.nin,
        ey: frame.ey,
        ez: frame.ez,
        sigma,
        eta_lo: ylo - sigma,
        eta_w,
        zeta_lo: zlo - sigma,
        zeta_w,
        scale: (vb / (n_eddy as f64 * sigma * sigma * sigma)).sqrt(),
        n_eddy,
        seed,
    };
    let mut seedv = vec![Vec3::new(0.0, 0.0, 0.0); start + rule.size];
    dfsem_values(&rule, 0.0, &mut seedv);
    Ok((rule, seedv[start..].to_vec()))
}

// ==========================================================================
//  The device side
// ==========================================================================

/// The two synthetic inflows.
#[derive(Clone, Debug, PartialEq)]
pub enum SyntheticRule {
    DigitalFilter(DigitalFilterRule),
    Dfsem(DfsemRule),
}

impl SyntheticRule {
    #[must_use]
    pub fn start(&self) -> usize {
        match self {
            Self::DigitalFilter(r) => r.start,
            Self::Dfsem(r) => r.start,
        }
    }
    #[must_use]
    pub fn size(&self) -> usize {
        match self {
            Self::DigitalFilter(r) => r.size,
            Self::Dfsem(r) => r.size,
        }
    }
}

/// A [`SyntheticRule`] on the device: its parameter block, its per-face
/// table (lattice nodes or face offsets) and its state `[n, X]`, both 0
/// after the seed.
pub struct GpuSynthetic {
    pub rule: SyntheticRule,
    /// [`df_params`] or [`dfsem_params`].
    pub params: DevBuf<f64>,
    /// `[2 size]` the digital filter's nodes; one dummy entry for DFSEM.
    pub nodes: DevBuf<Label>,
    /// `[3 size]` DFSEM's face offsets; one dummy entry for the filter.
    pub rel: DevBuf<f64>,
    /// `[2]` the last step drawn and the convected distance.
    pub state: DevBuf<f64>,
}

/// Upload the setup's synthetic rules with their state at 0.
pub fn upload_synthetic(gpu: &Gpu, rules: Vec<SyntheticRule>) -> Result<Vec<GpuSynthetic>> {
    let mut out = Vec::with_capacity(rules.len());
    for rule in rules {
        let (params, nodes, rel) = match &rule {
            SyntheticRule::DigitalFilter(r) => {
                let mut nd: Vec<Label> = Vec::with_capacity(2 * r.size);
                for [j, k] in &r.nodes {
                    nd.push(*j as Label);
                    nd.push(*k as Label);
                }
                if nd.is_empty() {
                    nd.push(0);
                }
                (df_params(r), nd, vec![0.0f64])
            }
            SyntheticRule::Dfsem(r) => {
                let mut rl: Vec<f64> = r.rel.iter().flat_map(|v| *v).collect();
                if rl.is_empty() {
                    rl.push(0.0);
                }
                (dfsem_params(r), vec![0 as Label], rl)
            }
        };
        out.push(GpuSynthetic {
            params: gpu.upload(&params)?,
            nodes: gpu.upload(&nodes)?,
            rel: gpu.upload(&rel)?,
            state: gpu.upload(&[0.0f64, 0.0])?,
            rule,
        });
    }
    Ok(out)
}
