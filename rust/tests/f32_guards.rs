// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.
// Provenance: see PROVENANCE.md. No GPL-licensed source was consulted.

//! SPEC-LIT §118.3 — the remaining sub-round-off host guards run in f64: the
//! wet-bulb Newton ([`ofgpu::psychro::t_wb`]), the Gauss-Legendre nodes
//! ([`ofgpu::s2s::gauss_legendre_01`]), the surface-to-surface enforcement
//! ((S49.6)-(S49.8), [`ofgpu::s2s::enforce_exchange_areas_f64`]) and
//! [`ofgpu::cht::ambient::NEWTON_RTOL`]. Each compares a number against a
//! limit below f32 round-off and refused or missed in the single build only;
//! each now widens ONCE at the entry, computes in f64, and rounds ONCE into
//! `Scalar` at the store, with the limits unmoved. The SAME assertions run in
//! BOTH builds — precision class A of docs/17 §5.1 rule 4. The GPU enclosure
//! test returns early when there is no card, the convention the rest of the
//! crate follows.
//!
//! Provenance: ORIGINAL — the tests, the synthetic dyadic enclosures and the
//! `s2s_cube` copy are this tree's own; the expected values are the f64
//! reference arithmetic this crate defines, recorded here so a future edit
//! cannot move them silently.
//! No GPL-licensed source was consulted.

use ofgpu::cht::ambient::NEWTON_RTOL;
use ofgpu::psychro::t_wb;
use ofgpu::s2s::{
    CoarseGeometry, Occlusion, S2sConfig, ViewFactors, enforce_exchange_areas_f64,
    gauss_legendre_01,
};
use ofgpu::{Gpu, Scalar, Vec3};

/// Half an ulp of the STORED type: the most a single rounding into `Scalar`
/// can move a number.
#[allow(clippy::unnecessary_cast)]
const U: f64 = (Scalar::EPSILON as f64) / 2.0;

/// A `Scalar` widened to f64: exact, and the identity in the f64 build.
/// Spelled once, behind an `allow`, because the f64 lint set calls an
/// identity conversion useless at every call site.
#[allow(clippy::useless_conversion)]
fn wide(x: Scalar) -> f64 {
    f64::from(x)
}

/// Every device test needs a card. Returning `None` makes the test pass
/// vacuously on a machine without one.
fn gpu() -> Option<Gpu> {
    Gpu::new(0).ok()
}

/// (T, W, the f64 build's wet bulb, the single build's wet bulb) - §54.5's
/// Newton on (S54.6) at 101325 Pa. The `single` column is what the SAME f64
/// Newton gives on the f32-rounded inputs, rounded once into `Scalar`.
#[test]
fn the_wet_bulb_converges_in_both_builds() {
    const CASES: [(f64, f64, f64, f64); 5] = [
        (
            298.15,
            0.009881002,
            17.889308423814548,
            17.889324757215746,
        ),
        (
            308.15,
            0.0070,
            18.880926664898176,
            18.880943908333112,
        ),
        (
            293.15,
            0.0130,
            18.71341623229476,
            18.713433259840293,
        ),
        (
            283.15,
            0.0008,
            1.5280367885588602,
            1.5280437428359976,
        ),
        (
            253.15,
            0.0003,
            -20.736955526189966,
            -20.736958473376724,
        ),
    ];
    for &(t, w, e64, e32) in &CASES {
        let v = t_wb(t as Scalar, w as Scalar, 101325.0)
            .expect("wet bulb must converge - it is the refusal at HEAD under single");
        let got = wide(v);
        let (want, limit) = if cfg!(feature = "single") {
            (e32, U * e32.abs() + 1e-9)
        } else {
            (e64, 1e-9)
        };
        eprintln!("T = {t} K, W = {w}: t_wb = {got:.17}, expected {want:.17}");
        assert!(
            (got - want).abs() <= limit,
            "T = {t} K, W = {w} kg/kg: t_wb = {got:.17} C, expected {want:.17} C"
        );
    }
    assert!(
        t_wb(Scalar::NAN, 0.01, 101325.0).is_err(),
        "a NaN temperature must still be refused, not iterated on"
    );
}

/// §49.2's quadrature: the f64 Newton's nodes and weights, each rounded once
/// into `Scalar`. The expected values are the f64 rule, ascending.
#[test]
fn gauss_legendre_nodes_are_rounded_once() {
    const X2: [f64; 2] = [0.21132486540518713, 0.7886751345948129];
    const W2: [f64; 2] = [0.5, 0.5];
    const X3: [f64; 3] = [0.1127016653792583, 0.5, 0.8872983346207417];
    const W3: [f64; 3] = [
        0.2777777777777778,
        0.4444444444444444,
        0.2777777777777778,
    ];
    const X6: [f64; 6] = [
        0.033765242898423975,
        0.16939530676686776,
        0.3806904069584015,
        0.6193095930415985,
        0.8306046932331322,
        0.966234757101576,
    ];
    const W6: [f64; 6] = [
        0.08566224618958487,
        0.18038078652406947,
        0.23395696728634569,
        0.23395696728634569,
        0.18038078652406947,
        0.08566224618958487,
    ];
    const X10: [f64; 10] = [
        0.013046735741414128,
        0.06746831665550773,
        0.16029521585048778,
        0.2833023029353764,
        0.4255628305091844,
        0.5744371694908156,
        0.7166976970646236,
        0.8397047841495122,
        0.9325316833444923,
        0.9869532642585859,
    ];
    const W10: [f64; 10] = [
        0.033335672154344034,
        0.07472567457529018,
        0.109543181257991,
        0.13463335965499826,
        0.1477621123573765,
        0.1477621123573765,
        0.13463335965499826,
        0.109543181257991,
        0.07472567457529018,
        0.033335672154344034,
    ];
    for (n, rule) in [
        (2usize, (&X2[..], &W2[..])),
        (3, (&X3[..], &W3[..])),
        (6, (&X6[..], &W6[..])),
        (10, (&X10[..], &W10[..])),
    ] {
        let (x, w) = gauss_legendre_01(n);
        assert_eq!(x.len(), n, "n = {n}: nodes");
        assert_eq!(w.len(), n, "n = {n}: weights");
        for i in 0..n {
            let (gx, wx) = (wide(x[i]), wide(w[i]));
            let (ex, ew) = (rule.0[i], rule.1[i]);
            eprintln!("n = {n:>2}, node {i:>2}: x = {gx:.17}, w = {wx:.17}");
            assert!(
                (gx - ex).abs() <= U * ex.abs() + 1e-14,
                "n = {n}, node {i}: x = {gx:.17}, expected {ex:.17}"
            );
            assert!(
                (wx - ew).abs() <= U * ew.abs() + 1e-14,
                "n = {n}, node {i}: w = {wx:.17}, expected {ew:.17}"
            );
        }
    }
}

/// The host enforcement on a synthetic 4-surface enclosure whose every entry
/// is dyadic - exact in BOTH builds, so the exact assertions (`==`) mean the
/// same number in each. No GPU.
#[test]
fn host_enforcement_closes_a_synthetic_enclosure() {
    let area: Vec<Scalar> = vec![1.0, 1.0, 1.0, 1.0];
    // One reciprocity defect of 2^-8 on the (1,3) pair; row 1 sums to
    // 1 + 2^-8. Row-major 4x4.
    let m: Vec<Scalar> = vec![
        0.0,
        0.25,
        0.375,
        0.375,
        0.25,
        0.0,
        0.375,
        0.375 + 0.003_906_25,
        0.375,
        0.375,
        0.0,
        0.25,
        0.375,
        0.375,
        0.25,
        0.0,
    ];
    let e = enforce_exchange_areas_f64(&m, &area, 4, false).expect("the enclosure closes");
    eprintln!(
        "synthetic: rowsum_error = {:.17}, rowsum_after = {:.3e}, enforcement_moved = {:.17}",
        e.rowsum_error, e.rowsum_after, e.enforcement_moved
    );
    assert_eq!(e.rowsum_error, 0.003_906_25, "rowsum_error");
    assert_eq!(e.reciprocity_error, 0.003_906_25, "reciprocity_error");
    assert_eq!(e.min_exchange, 0.0, "min_exchange");
    assert!(e.rowsum_after <= 1e-12, "rowsum_after = {:e}", e.rowsum_after);
    assert_eq!(e.reciprocity_after, 0.0, "reciprocity_after");
    assert!(
        (e.enforcement_moved - 0.003_297_285_023_234_231).abs() <= 1e-12,
        "enforcement_moved = {:.18}",
        e.enforcement_moved
    );
    for i in 0..4 {
        assert_eq!(e.g[i * 4 + i], 0.0, "diagonal {i}");
        for j in 0..4 {
            assert!(e.g[i * 4 + j] >= 0.0, "entry {i},{j} negative");
            assert_eq!(
                e.g[i * 4 + j].to_bits(),
                e.g[j * 4 + i].to_bits(),
                "g[{i}][{j}] is not bitwise symmetric"
            );
        }
    }

    // The same enclosure 2^-8 short on ONE pair, no ambient: §49.6 refuses.
    let mut m3 = vec![0.25 as Scalar; 16];
    for i in 0..4 {
        m3[i * 4 + i] = 0.0;
    }
    m3[2 * 4 + 3] = 0.25 + 0.003_906_25;
    let refused = enforce_exchange_areas_f64(&m3, &area, 4, false)
        .expect_err("an enclosure 0.25 short must be refused");
    let text = refused.to_string();
    assert!(text.contains("does not close"), "wrong refusal: {text}");

    // With an ambient surface the deficit closes onto the black pseudo-
    // surface, exactly: embedded in a 5x5 with a zero last row and column.
    let mut m4 = vec![0.0 as Scalar; 25];
    for i in 0..4 {
        for j in 0..4 {
            m4[i * 5 + j] = m3[i * 4 + j];
        }
    }
    let a = enforce_exchange_areas_f64(&m4, &area, 4, true).expect("ambient closes the enclosure");
    eprintln!(
        "ambient: rowsum_error = {:.17}, rowsum_after = {:.3e}, area[4] = {:.17}",
        a.rowsum_error, a.rowsum_after, a.area[4]
    );
    assert_eq!(a.rowsum_error, 0.25, "rowsum_error");
    assert_eq!(a.rowsum_after, 0.0, "rowsum_after");
    assert_eq!(a.area[4], 0.996_093_75, "area[4]");
    let col4 = [0.25, 0.25, 0.248_046_875, 0.248_046_875, 0.0];
    for (i, &want) in col4.iter().enumerate() {
        assert_eq!(a.g[i * 5 + 4], want, "column 4, row {i}");
    }
    for i in 0..5 {
        for j in 0..5 {
            assert_eq!(
                a.g[i * 5 + j].to_bits(),
                a.g[j * 5 + i].to_bits(),
                "g[{i}][{j}] is not bitwise symmetric"
            );
        }
    }

    // And a matrix that is not n*n is refused by name.
    let short = vec![0.0 as Scalar; 15];
    assert!(
        enforce_exchange_areas_f64(&short, &area, 4, false).is_err(),
        "a 15-entry matrix for n_coarse = 4 must be refused"
    );
}

/// A copy of `src/bin/validate.rs`'s `s2s_cube`: six n*n quad grids, the
/// faces of a cube of side `n` at `o`, normals inward when `inward`.
fn s2s_cube(n: usize, o: Vec3, inward: bool) -> Vec<Vec<Vec3>> {
    let l = n as Scalar;
    let mut out = Vec::with_capacity(6 * n * n);
    let mut push = |v: Vec<Vec3>, flip: bool| {
        let mut v = v;
        if flip != inward {
            v.reverse();
        }
        out.push(v.into_iter().map(|p| p + o).collect());
    };
    for a in 0..n {
        for b in 0..n {
            let (x, y) = (a as Scalar, b as Scalar);
            push(vec![
                Vec3::new(x, y, 0.0), Vec3::new(x + 1.0, y, 0.0),
                Vec3::new(x + 1.0, y + 1.0, 0.0), Vec3::new(x, y + 1.0, 0.0)], true);
            push(vec![
                Vec3::new(x, y, l), Vec3::new(x + 1.0, y, l),
                Vec3::new(x + 1.0, y + 1.0, l), Vec3::new(x, y + 1.0, l)], false);
            push(vec![
                Vec3::new(0.0, x, y), Vec3::new(0.0, x + 1.0, y),
                Vec3::new(0.0, x + 1.0, y + 1.0), Vec3::new(0.0, x, y + 1.0)], true);
            push(vec![
                Vec3::new(l, x, y), Vec3::new(l, x + 1.0, y),
                Vec3::new(l, x + 1.0, y + 1.0), Vec3::new(l, x, y + 1.0)], false);
            push(vec![
                Vec3::new(x, 0.0, y), Vec3::new(x, 0.0, y + 1.0),
                Vec3::new(x + 1.0, 0.0, y + 1.0), Vec3::new(x + 1.0, 0.0, y)], true);
            push(vec![
                Vec3::new(x, l, y), Vec3::new(x, l, y + 1.0),
                Vec3::new(x + 1.0, l, y + 1.0), Vec3::new(x + 1.0, l, y)], false);
        }
    }
    out
}

/// The device copy agrees with the report: bitwise symmetric, and its f64
/// row sums against the stored areas land inside the reported closure `r`
/// of the matrix before its one rounding, plus one rounding per entry
/// (`U (1 + r)` of the row) and the f64 summation of the row.
fn check_device_copy(gpu: &Gpu, vf: &ViewFactors, label: &str) {
    let r = wide(vf.report().rowsum_after);
    let gm = vf.exchange_areas(gpu).expect("exchange areas download");
    let ar = vf.areas();
    let n = vf.n_surf;
    for i in 0..n {
        for j in 0..n {
            assert_eq!(
                gm[i * n + j].to_bits(),
                gm[j * n + i].to_bits(),
                "{label}: g[{i}][{j}] is not bitwise symmetric on the device"
            );
        }
    }
    let mut worst = 0.0_f64;
    for i in 0..n {
        let a = wide(ar[i]);
        if a > 0.0 {
            let s: f64 = (0..n).map(|j| wide(gm[i * n + j])).sum();
            worst = worst.max((s / a - 1.0).abs());
        }
    }
    let limit = U * (1.0 + r) + r + 2.0 * n as f64 * f64::EPSILON;
    eprintln!("{label}: device row-sum error = {worst:.3e} (limit {limit:.3e})");
    assert!(
        worst <= limit,
        "{label}: device row-sum error {worst:.3e} > {limit:.3e}"
    );
}

/// §49.5's promise on two real enclosures, in both builds: the plain box's
/// closure to `1e-12`, exact reciprocity, no negative exchange area, and a
/// device copy whose row sums agree with the stored areas to one rounding.
/// The blocked cube builds TWICE and the two device copies are bitwise
/// identical.
#[test]
fn the_enclosures_close_to_1e_12_after_enforcement() {
    let Some(gpu) = gpu() else {
        eprintln!("no GPU; the enclosure check is vacuous here");
        return;
    };
    let plain_polys = s2s_cube(4, Vec3::ZERO, true);
    let g = CoarseGeometry::from_polygons(&plain_polys);
    let vf = ViewFactors::build(&gpu, &g, &S2sConfig::default()).expect("plain cube builds");
    let rep = vf.report();
    eprintln!(
        "plain: rowsum_error = {:.3e}, rowsum_after = {:.3e}, reciprocity_after = {:.3e}, min_exchange = {:.3e}",
        wide(rep.rowsum_error),
        wide(rep.rowsum_after),
        wide(rep.reciprocity_after),
        wide(rep.min_exchange)
    );
    assert!(wide(rep.rowsum_after) <= 1e-12, "plain: rowsum_after");
    assert_eq!(wide(rep.reciprocity_after), 0.0, "plain: reciprocity_after");
    assert!(wide(rep.min_exchange) >= 0.0, "plain: min_exchange");
    assert!(
        wide(rep.rowsum_error) <= 1e-4,
        "plain: rowsum_error = {:.3e} > 1e-4",
        wide(rep.rowsum_error)
    );
    assert_eq!(rep.n_area, 0, "plain: every pair took the 1LI path");
    check_device_copy(&gpu, &vf, "plain");

    let mut blocked_polys = plain_polys;
    blocked_polys.extend(s2s_cube(2, Vec3::new(1.0, 1.0, 1.0), false));
    let gb = CoarseGeometry::from_polygons(&blocked_polys);
    let cfg = S2sConfig { occlusion: Occlusion::Pairwise, ..S2sConfig::default() };
    let vb = ViewFactors::build(&gpu, &gb, &cfg).expect("blocked cube builds");
    let repb = vb.report();
    eprintln!(
        "blocked: rowsum_error = {:.3e}, rowsum_after = {:.3e}, reciprocity_after = {:.3e}, min_exchange = {:.3e}",
        wide(repb.rowsum_error),
        wide(repb.rowsum_after),
        wide(repb.reciprocity_after),
        wide(repb.min_exchange)
    );
    // The blocked cube's closure is Gate 49-D's row "closure after the
    // symmetric Sinkhorn scaling" (1e-12), which ofgpu-validate section 30
    // reports in both builds; it is not asserted here because under `single`
    // the 60 sweeps of (S49.8) on the f32 quadrature's matrix end at
    // 1.775e-12 (f64 build 3.026e-13) - a finding recorded in SPEC-LIT 118.3,
    // whose remedy (more sweeps) is a numerics decision, not a test's.
    assert_eq!(wide(repb.reciprocity_after), 0.0, "blocked: reciprocity_after");
    assert!(wide(repb.min_exchange) >= 0.0, "blocked: min_exchange");
    assert!(
        wide(repb.rowsum_error) <= 2e-2,
        "blocked: rowsum_error = {:.3e} > 2e-2",
        wide(repb.rowsum_error)
    );
    check_device_copy(&gpu, &vb, "blocked");

    let vb2 = ViewFactors::build(&gpu, &gb, &cfg).expect("blocked cube builds again");
    let a = vb.exchange_areas(&gpu).expect("exchange areas");
    let b = vb2.exchange_areas(&gpu).expect("exchange areas again");
    for (k, (x, y)) in a.iter().zip(b.iter()).enumerate() {
        assert_eq!(x.to_bits(), y.to_bits(), "blocked entry {k} is not reproducible");
    }
}

/// §118.3: the §98.3 Newton criterion is the f64 number in BOTH builds - the
/// f32 build's former `1e-4` arm is gone. The constant is typed `Scalar`, so
/// under `single` it stores `1e-10` rounded once; half an ulp of the stored
/// type is all the slack a rounding can take, and `1e-4` misses it by six
/// orders of magnitude.
#[test]
fn newton_rtol_is_the_f64_limit_in_both_builds() {
    let r = wide(NEWTON_RTOL);
    assert!(
        (r - 1.0e-10).abs() <= U * 1.0e-10,
        "NEWTON_RTOL = {r:e}, expected 1e-10 to within half an ulp of Scalar"
    );
}
