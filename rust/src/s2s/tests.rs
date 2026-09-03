// meteor-cfd - Copyright (c) 2026 주식회사 메테오시뮬레이션 (Meteo Simulation Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.
// Provenance: see PROVENANCE.md. No GPL-licensed source was consulted.

//! SPEC-LIT S49.7, S49.8, S50.10, S50.11 and S51.2 - what must hold for
//! surface-to-surface radiation, and the analytic gates.
//!
//! Every gate here is a closed form or an identity the code checks against
//! itself. Nothing is compared against another CFD code. The two Howell
//! configuration factors are **evaluated from their published formulae**
//! rather than quoted as constants, so a transcription error in the formula
//! shows up as a failure and not as agreement.
//!
//! No GPL-licensed source was consulted.

use super::*;
use crate::field::BcKind;
use crate::io::dict::FoamDict;

fn gpu() -> Option<Gpu> {
    Gpu::new(0).ok()
}

/// The gates of SPEC-LIT S49.8 are bare rectangles in space, so the enclosure
/// is genuinely open and a closure surface is the honest declaration of that
/// (S49.6). Occlusion off unless a gate needs it.
fn open_cfg() -> S2sConfig {
    S2sConfig {
        emissivity: 0.9,
        occlusion: Occlusion::None,
        ambient_temperature: Some(300.0),
        ..Default::default()
    }
}

// ==========================================================================
//  The gate geometries
// ==========================================================================

/// Two identical unit squares, parallel, directly opposed, unit separation -
/// Howell **C-11**. Normals face each other. With `plate`, the Shapiro
/// configuration's back-to-back 0.5 x 0.5 plates at 3/4 of the separation
/// (FACET UCID-19887 Fig. 12).
fn opposed_squares(plate: bool) -> Vec<Vec<Vec3>> {
    let s = 0.5 as Scalar;
    let mut v = vec![
        vec![
            Vec3::new(-s, -s, 0.0),
            Vec3::new(s, -s, 0.0),
            Vec3::new(s, s, 0.0),
            Vec3::new(-s, s, 0.0),
        ],
        vec![
            Vec3::new(-s, -s, 1.0),
            Vec3::new(-s, s, 1.0),
            Vec3::new(s, s, 1.0),
            Vec3::new(s, -s, 1.0),
        ],
    ];
    if plate {
        let q = 0.25 as Scalar;
        v.push(vec![
            Vec3::new(-q, -q, 0.75),
            Vec3::new(-q, q, 0.75),
            Vec3::new(q, q, 0.75),
            Vec3::new(q, -q, 0.75),
        ]);
        v.push(vec![
            Vec3::new(-q, -q, 0.75),
            Vec3::new(q, -q, 0.75),
            Vec3::new(q, q, 0.75),
            Vec3::new(-q, q, 0.75),
        ]);
    }
    v
}

/// Two unit squares of common edge length 1 meeting at 90 degrees - Howell
/// **C-14**, the near-field gate.
fn perpendicular_squares() -> Vec<Vec<Vec3>> {
    vec![
        vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(1.0, 1.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
        ],
        vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
            Vec3::new(1.0, 0.0, 1.0),
            Vec3::new(1.0, 0.0, 0.0),
        ],
    ]
}

/// A cube of side `n` built from `6n^2` unit squares, all normals pointing
/// INTO the cavity (`inward`) or out of it. Offset by `o`.
///
/// This is NISTIR 6925's `BB104` construction: a small cube of unit squares
/// (outward) inside a large one (inward). `HostMesh` is not involved - these
/// are radiating surfaces, not control volumes.
fn cube_of_squares(n: usize, o: Vec3, inward: bool) -> Vec<Vec<Vec3>> {
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
            let (x, y, z) = (a as Scalar, b as Scalar, 0.0 as Scalar);
            // z = 0, +z is inward
            push(
                vec![
                    Vec3::new(x, y, z),
                    Vec3::new(x + 1.0, y, z),
                    Vec3::new(x + 1.0, y + 1.0, z),
                    Vec3::new(x, y + 1.0, z),
                ],
                true,
            );
            // z = l, -z is inward
            push(
                vec![
                    Vec3::new(x, y, l),
                    Vec3::new(x + 1.0, y, l),
                    Vec3::new(x + 1.0, y + 1.0, l),
                    Vec3::new(x, y + 1.0, l),
                ],
                false,
            );
            // x = 0, +x is inward
            push(
                vec![
                    Vec3::new(0.0, x, y),
                    Vec3::new(0.0, x + 1.0, y),
                    Vec3::new(0.0, x + 1.0, y + 1.0),
                    Vec3::new(0.0, x, y + 1.0),
                ],
                true,
            );
            push(
                vec![
                    Vec3::new(l, x, y),
                    Vec3::new(l, x + 1.0, y),
                    Vec3::new(l, x + 1.0, y + 1.0),
                    Vec3::new(l, x, y + 1.0),
                ],
                false,
            );
            // y = 0, +y is inward
            push(
                vec![
                    Vec3::new(x, 0.0, y),
                    Vec3::new(x, 0.0, y + 1.0),
                    Vec3::new(x + 1.0, 0.0, y + 1.0),
                    Vec3::new(x + 1.0, 0.0, y),
                ],
                true,
            );
            push(
                vec![
                    Vec3::new(x, l, y),
                    Vec3::new(x, l, y + 1.0),
                    Vec3::new(x + 1.0, l, y + 1.0),
                    Vec3::new(x + 1.0, l, y),
                ],
                false,
            );
        }
    }
    out
}

// ==========================================================================
//  S49.2  The quadrature itself
// ==========================================================================

/// An `n`-point Gauss-Legendre rule integrates every polynomial up to degree
/// `2n-1` exactly. Generated by Newton iteration rather than transcribed, so
/// this is the check that the generator is right.
#[test]
fn gauss_legendre_is_exact_to_degree_2n_minus_1() {
    for &n in &NQ_TABLE {
        let (x, w) = gauss_legendre_01(n);
        assert_eq!(x.len(), n);
        // Weights sum to the interval length.
        let sw: Scalar = w.iter().sum();
        assert!((sw - 1.0).abs() < 1e-14, "n={n}: weights sum to {sw}");
        for d in 0..=(2 * n - 1) {
            let got: Scalar = x.iter().zip(&w).map(|(&xi, &wi)| wi * xi.powi(d as i32)).sum();
            let want = 1.0 / (d as Scalar + 1.0);
            assert!(
                (got - want).abs() < 1e-13 * want.max(1.0),
                "n={n}, degree {d}: got {got}, want {want}"
            );
        }
        // Nodes strictly inside (0,1), ascending - the property the Duffy
        // map's `u` factor and the 1LI outer loop both rely on.
        for i in 0..n {
            assert!(x[i] > 0.0 && x[i] < 1.0, "n={n}: node {i} = {}", x[i]);
            if i > 0 {
                assert!(x[i] > x[i - 1], "n={n}: nodes not ascending");
            }
        }
    }
}

/// SPEC-LIT S49.8 Gate 49-A's closed form, evaluated here and compared with
/// NISTIR 6925's published `0.19982490`. If the transcription of the formula
/// is wrong, this fails rather than the gate silently agreeing with a wrong
/// reference.
#[test]
fn howell_c11_reproduces_the_published_value() {
    let f = howell_c11(1.0, 1.0);
    assert!(
        (f - 0.1998248957).abs() < 1e-9,
        "C-11 at X=Y=1 evaluates to {f}, not NISTIR 6925's 0.19982490"
    );
    // A far pair sees almost nothing; a near one sees a lot. Monotone in the
    // aspect ratio, which is the only structural property worth pinning.
    assert!(howell_c11(0.1, 0.1) < f);
    assert!(howell_c11(10.0, 10.0) > f);
}

/// Gate 49-B's closed form (Hottel / Hamilton & Morgan).
#[test]
fn howell_c14_reproduces_the_published_value() {
    let f = howell_c14(1.0, 1.0);
    assert!(
        (f - 0.2000437761).abs() < 1e-9,
        "C-14 at H=W=1 evaluates to {f}, not 0.20004378"
    );
}

/// SPEC-LIT S49.8 Gate 49-C: the three internal consistency checks that make
/// the Shapiro benchmark a strong gate rather than one number. `A_3/A_1` is
/// exactly `0.25`, so reciprocity relates the published pairs exactly.
#[test]
fn the_shapiro_published_values_are_internally_consistent() {
    let (f31, f13) = (0.33681717 as Scalar, 0.084204294 as Scalar);
    let (f42, f24) = (0.79445272 as Scalar, 0.19861318 as Scalar);
    let f12_star = 0.19982490 as Scalar;
    let f12 = 0.11562061 as Scalar;
    assert!((f31 * 0.25 - f13).abs() < 2e-9, "F13 != F31 A3/A1");
    assert!((f42 * 0.25 - f24).abs() < 2e-9, "F24 != F42 A4/A2");
    assert!((f12_star - f13 - f12).abs() < 2e-8, "F12 != F*12 - F13");
    // And the unobstructed value is C-11's own.
    assert!((howell_c11(1.0, 1.0) - f12_star).abs() < 1e-8);
}

// ==========================================================================
//  S49.8  The analytic view-factor gates
// ==========================================================================

/// **Gate 49-A.** Two opposed unit squares at unit separation.
/// SPEC-LIT S49.7 asks for `< 1e-6`; measured `4.7e-16`, because the pair is
/// unobstructed and mutually in front, so it takes the 1LI contour path whose
/// inner integral is closed-form.
#[test]
fn gate_49a_two_opposed_unit_squares() {
    let Some(gpu) = gpu() else { return };
    let g = CoarseGeometry::from_polygons(&opposed_squares(false));
    assert!(g.blockers().is_empty(), "two opposed squares cannot obstruct anything");

    let vf = ViewFactors::build(&gpu, &g, &open_cfg()).expect("view factors");
    let n = vf.n_surf;
    let f = vf.view_factors(&gpu).expect("F");
    let want = howell_c11(1.0, 1.0);
    let err = (f[1] - want).abs();
    assert!(err < 1e-6, "C-11: F12 = {}, want {want}, err {err:.3e}", f[1]);
    assert!(err < 1e-12, "C-11 regressed: err {err:.3e} (measured 4.7e-16)");

    // Reciprocity: A1 = A2 = 1, so F21 must equal F12 - and after (S49.7) it
    // is the same number, bit for bit.
    assert_eq!(f[1].to_bits(), f[n].to_bits(), "F12 != F21 bitwise");
    // The pair went through the contour path, which is the claim the
    // tolerance rests on.
    assert_eq!(vf.report().n_line, 2);
    assert_eq!(vf.report().n_area, 0);
}

/// **Gate 49-B, the canary.** Two unit squares sharing an edge at 90 degrees.
///
/// This is the configuration that decided the whole method. Gauss-Legendre
/// 2AI - the note's own recommendation - was MEASURED at `0.2803` against the
/// closed form `0.20004` (40% error) and converging like `nq^-0.5`: `0.2803`
/// at `nq = 6`, `0.2610` at `nq = 10`. The near-field gate is unreachable
/// that way, exactly as NISTIR 6925 Figs. 9-10 predict. 1LI with the
/// Mitalas-Stephenson closed-form inner integral reaches `6.6e-6`.
#[test]
fn gate_49b_two_squares_sharing_an_edge() {
    let Some(gpu) = gpu() else { return };
    let g = CoarseGeometry::from_polygons(&perpendicular_squares());
    assert!(g.blockers().is_empty());

    let vf = ViewFactors::build(&gpu, &g, &open_cfg()).expect("view factors");
    let f = vf.view_factors(&gpu).expect("F");
    let want = howell_c14(1.0, 1.0);
    let err = (f[1] - want).abs();
    assert!(err < 1e-5, "C-14: F12 = {}, want {want}, err {err:.3e}", f[1]);
    assert_eq!(vf.report().n_line, 2, "the near-field pair must take 1LI");
}

/// SPEC-LIT S49.7: raising `nq` moves both analytic gates toward the closed
/// form and never away. This is also §51.2's `viewFactorQuadrature` pair
/// test - two configurations identical but for that one entry, REQUIRED to
/// produce different `F`.
#[test]
fn raising_the_quadrature_order_improves_both_gates_monotonically() {
    let Some(gpu) = gpu() else { return };
    let c11 = CoarseGeometry::from_polygons(&opposed_squares(false));
    let c14 = CoarseGeometry::from_polygons(&perpendicular_squares());
    let (w11, w14) = (howell_c11(1.0, 1.0), howell_c14(1.0, 1.0));

    let mut prev14 = Scalar::INFINITY;
    let mut f_at = Vec::new();
    for &nq in &[2usize, 3, 4, 6, 8, 10] {
        let cfg = S2sConfig { quadrature: nq, ..open_cfg() };
        let a = ViewFactors::build(&gpu, &c11, &cfg).expect("c11");
        let b = ViewFactors::build(&gpu, &c14, &cfg).expect("c14");
        let e11 = (a.view_factors(&gpu).expect("f")[1] - w11).abs();
        let fb = b.view_factors(&gpu).expect("f")[1];
        let e14 = (fb - w14).abs();
        assert!(
            e14 < prev14,
            "C-14 error is not monotone in nq: {e14:.3e} at nq={nq} is not below {prev14:.3e}"
        );
        prev14 = e14;
        f_at.push(fb);
        // The far-field gate is essentially exact from nq = 4 on; only pin
        // that it never gets WORSE than its own coarse value.
        assert!(e11 <= 3.0e-4, "C-11 at nq={nq}: err {e11:.3e}");
    }

    // S13.4.1: the entry must CHANGE the answer, or it is a setting the
    // solver ignores.
    assert!(
        (f_at[0] - f_at[5]).abs() > 1e-6,
        "viewFactorQuadrature 2 and 10 produced the same F ({}); the entry is \
         being ignored",
        f_at[0]
    );
}

/// **Gate 49-C.** The Shapiro configuration: the obstructed `F_12`, and the
/// four published unobstructed factors around it.
///
/// The obstructed number is the one NISTIR 6925 warns about: `b_ij` is a
/// discontinuous integrand, Gaussian quadrature loses its spectral
/// convergence there, and the pair has nowhere to go but the area form.
/// Measured `6.8e-4`; NISTIR's own 2AI-with-blockage reaches `1.1e-4` only at
/// 40 000 uniform samples per surface.
#[test]
#[allow(clippy::too_many_lines)]
fn gate_49c_the_shapiro_obstructed_configuration() {
    let Some(gpu) = gpu() else { return };
    let g = CoarseGeometry::from_polygons(&opposed_squares(true));
    assert_eq!(
        g.blockers(),
        vec![2, 3],
        "only the two plates can obstruct; the enclosing squares cannot"
    );

    let cfg = S2sConfig { occlusion: Occlusion::Pairwise, ..open_cfg() };
    let vf = ViewFactors::build(&gpu, &g, &cfg).expect("view factors");
    let n = vf.n_surf;
    let f = vf.view_factors(&gpu).expect("F");

    // The four unobstructed pairs, which take the 1LI path.
    let cases: [(usize, usize, Scalar, Scalar); 4] = [
        (0, 2, 0.084204294, 1e-8),
        (2, 0, 0.33681717, 1e-8),
        (3, 1, 0.79445272, 1e-8),
        (1, 3, 0.19861318, 1e-8),
    ];
    for (i, j, want, tol) in cases {
        let got = f[i * n + j];
        assert!(
            (got - want).abs() < tol,
            "F{}{} = {got:.9}, want {want}, err {:.3e}",
            i + 1,
            j + 1,
            (got - want).abs()
        );
    }

    // The gate: the obstructed pair.
    let f12 = f[n];
    let err = (f12 - 0.11562061).abs();
    assert!(
        err < 1.0e-3,
        "Shapiro F12 = {f12:.9}, want 0.11562061, err {err:.3e}"
    );
    assert!(err > 1.0e-5, "F12 got BETTER than the recorded 6.8e-4 - re-record the number");

    // The plate blocks: without occlusion the same geometry gives the
    // UNOBSTRUCTED C-11 value. This is S51.2's `occlusion` pair test.
    let open = ViewFactors::build(&gpu, &g, &open_cfg()).expect("no occlusion");
    let f_open = open.view_factors(&gpu).expect("F")[open.n_surf];
    assert!(
        (f_open - howell_c11(1.0, 1.0)).abs() < 1e-9,
        "occlusion none: F12 = {f_open}, want the unobstructed 0.19982490"
    );
    assert!(
        (f_open - f12).abs() > 0.08,
        "`occlusion` changed nothing: {f_open} vs {f12}"
    );

    // Two back-to-back coincident plates exchange NOTHING - r lies in their
    // common plane, so cos(theta) is zero. Found by measurement: the contour
    // form has no cos clamp and returned a large value here until
    // `s2sRelativeSide` was taught to recognise the coplanar case.
    assert_eq!(f[2 * n + 3], 0.0, "coplanar plates 3 and 4 must exchange nothing");
    assert_eq!(f[3 * n + 2], 0.0);
}

/// SPEC-LIT S49.7: the Shapiro obstructed factor must improve with `nq`, even
/// though it converges far more slowly than an unobstructed one.
#[test]
fn the_obstructed_factor_improves_with_the_quadrature_order() {
    let Some(gpu) = gpu() else { return };
    let g = CoarseGeometry::from_polygons(&opposed_squares(true));
    let mut errs = Vec::new();
    for &nq in &[3usize, 6, 10] {
        let cfg = S2sConfig {
            occlusion: Occlusion::PerPoint,
            quadrature: nq,
            ..open_cfg()
        };
        let vf = ViewFactors::build(&gpu, &g, &cfg).expect("vf");
        let f = vf.view_factors(&gpu).expect("F");
        errs.push((f[vf.n_surf] - 0.11562061).abs());
    }
    assert!(
        errs[2] < errs[0],
        "obstructed F12 did not improve from nq=3 ({:.3e}) to nq=10 ({:.3e})",
        errs[0],
        errs[2]
    );
}

// ==========================================================================
//  S49.5, S49.6, S49.7  Enforcement, closure, determinism
// ==========================================================================

/// **Gate 49-D**, at the scale the always-run suite can afford: a closed cube
/// of unit squares with a smaller cube of them inside it, the `BB104`
/// construction. The enclosing cube's faces cannot obstruct anything - only
/// the inner cube's can - which is what exercises the blocker-set
/// elimination.
#[test]
#[allow(clippy::too_many_lines)]
fn gate_49d_closure_at_scale_with_an_internal_blocker() {
    let Some(gpu) = gpu() else { return };
    let mut polys = cube_of_squares(4, Vec3::ZERO, true);
    let n_outer = polys.len();
    polys.extend(cube_of_squares(2, Vec3::new(1.0, 1.0, 1.0), false));

    let g = CoarseGeometry::from_polygons(&polys);
    assert_eq!(g.n, 96 + 24);
    let blockers = g.blockers();
    assert_eq!(
        blockers.len(),
        24,
        "only the inner cube can obstruct; got {} blockers",
        blockers.len()
    );
    assert!(
        blockers.iter().all(|&b| b >= n_outer),
        "an enclosing-cube face was flagged as a blocker"
    );

    // The three error sources are MEASURED SEPARATELY before anything is
    // gated, because "the row sum misses by X" is useless without knowing
    // which of them X came from.

    // (1) The quadrature alone, on the same 96-face enclosure with nothing
    //     in it. Measured 6.6e-6, every pair on the 1LI contour path, 0.014 s
    //     - two orders below NISTIR 6925's View3D figure of 1e-3.
    let plain = CoarseGeometry::from_polygons(&cube_of_squares(4, Vec3::ZERO, true));
    let vp = ViewFactors::build(&gpu, &plain, &S2sConfig::default()).expect("plain");
    println!("gate 49-D plain box:  {}", vp.report().describe());
    assert!(
        vp.report().rowsum_error < 1e-4,
        "the QUADRATURE alone misses closure by {:.3e}",
        vp.report().rowsum_error
    );
    assert_eq!(vp.report().n_area, 0, "a convex enclosure has no area-form pairs");

    // (2) Turning occlusion OFF on a geometry that needs it is CAUGHT rather
    //     than silently wrong: an outer wall then sees the far wall AND the
    //     blocker in front of it, and its row sums to 1.415 instead of 1.
    //     S49.6's closure refusal is what notices.
    let e = match ViewFactors::build(
        &gpu,
        &g,
        &S2sConfig { occlusion: Occlusion::None, ..S2sConfig::default() },
    ) {
        Ok(_) => panic!("`occlusion none` on a blocked enclosure was accepted"),
        Err(e) => e.to_string(),
    };
    assert!(e.contains("does not close"), "{e}");

    // (3) Level 1: five rays per pair, escalating to per-point where they
    //     disagree. Measured 8.8e-3 - and the whole of it is the OCCLUSION's,
    //     because (1) put the quadrature at 6.6e-6 on the same geometry.
    //     Level 1's all-or-nothing decision on a partly-shadowed pair is item
    //     3 of the design note's own "what I am least sure about"; this is
    //     the number it did not have.
    let cfg = S2sConfig { occlusion: Occlusion::Pairwise, ..S2sConfig::default() };
    let pw = *ViewFactors::build(&gpu, &g, &cfg).expect("pairwise").report();
    println!("gate 49-D pairwise:   {}", pw.describe());
    assert!(
        pw.rowsum_error < 2e-2,
        "Level-1 visibility misses closure by {:.3e}",
        pw.rowsum_error
    );
    assert!(pw.min_exchange >= 0.0, "negative exchange area {:.3e}", pw.min_exchange);
    assert_eq!(pw.reciprocity_after, 0.0, "(S49.7) is an elementwise average");
    // 20 sweeps left this at 1.4e-6 - the scaling converges at about a
    // factor of two per sweep on a matrix whose blocked and coplanar pairs
    // put many exact zeros in G, not the factor of ten a convex enclosure
    // suggests. SINKHORN_SWEEPS is 60 because of this measurement.
    assert!(pw.rowsum_after <= 1e-12, "closure after Sinkhorn is {:.3e}", pw.rowsum_after);

    // (4) **Level 2 is NOT uniformly better, and this is the measurement that
    //     says so.** `perPoint` distrusts the five-ray "visible" verdict, so
    //     every pair that could be blocked goes to the AREA form - and on a
    //     box, the adjacent-wall pairs are the C-14 configuration, where the
    //     area form is 40% wrong (SPEC-LIT 49.2b). Closure degrades from 8.8e-3 to
    //     0.16, past S49.6's threshold, and the model REFUSES rather than
    //     shipping that F. The design note assumed Level 2 was strictly more
    //     accurate than Level 1. On this geometry it is strictly worse.
    let pp = ViewFactors::build(
        &gpu,
        &g,
        &S2sConfig { occlusion: Occlusion::PerPoint, ..S2sConfig::default() },
    );
    let m = match pp {
        Ok(v) => panic!(
            "`occlusion perPoint` was expected to lose closure here; it reported \
             {:.3e}. If the area path has been improved, re-record this.",
            v.report().rowsum_error
        ),
        Err(e) => e.to_string(),
    };
    assert!(m.contains("does not close"), "{m}");
    assert!(
        m.contains("occlusion"),
        "the refusal must name the occlusion setting as a cause: {m}"
    );
}

/// SPEC-LIT S49.7: (S49.7) then (S49.8) leaves `G` symmetric to EXACTLY zero,
/// so the two enforcement steps do not fight; and the closure it produces is
/// exact.
#[test]
fn symmetrising_then_scaling_leaves_reciprocity_exactly_zero() {
    let Some(gpu) = gpu() else { return };
    let g = CoarseGeometry::from_polygons(&cube_of_squares(3, Vec3::ZERO, true));
    let vf = ViewFactors::build(&gpu, &g, &S2sConfig::default()).expect("vf");
    let r = *vf.report();
    assert_eq!(r.reciprocity_after, 0.0);
    assert!(r.rowsum_after <= 1e-12, "{:.3e}", r.rowsum_after);
    assert!(r.min_exchange >= 0.0);

    // And bit for bit, not merely to a tolerance.
    let n = vf.n_surf;
    let ge = vf.exchange_areas(&gpu).expect("G");
    for i in 0..n {
        for j in 0..n {
            assert_eq!(
                ge[i * n + j].to_bits(),
                ge[j * n + i].to_bits(),
                "G[{i},{j}] != G[{j},{i}] bitwise"
            );
        }
    }
}

/// SPEC-LIT S49.7: two builds on the same geometry are BITWISE identical.
/// The whole method exists to make this true.
#[test]
fn two_builds_are_bitwise_identical() {
    let Some(gpu) = gpu() else { return };
    let g = CoarseGeometry::from_polygons(&opposed_squares(true));
    let cfg = S2sConfig { occlusion: Occlusion::Pairwise, ..open_cfg() };
    let a = ViewFactors::build(&gpu, &g, &cfg).expect("a");
    let b = ViewFactors::build(&gpu, &g, &cfg).expect("b");
    let (ga, gb) = (a.exchange_areas(&gpu).expect("ga"), b.exchange_areas(&gpu).expect("gb"));
    assert_eq!(ga.len(), gb.len());
    for (k, (x, y)) in ga.iter().zip(&gb).enumerate() {
        assert_eq!(x.to_bits(), y.to_bits(), "G[{k}] differs between two builds");
    }
}

/// SPEC-LIT S49.7: **the grid is an accelerator, not a truth.**
///
/// The same geometry, the same blockers, the same kernel - once walking the
/// uniform grid and once scanning every blocker triangle linearly. Any-hit is
/// a boolean OR, so the two must agree EXACTLY. If they ever do not, the grid
/// is missing a triangle and every view factor downstream is quietly wrong.
///
/// Run on the DEVICE, not against a host transcription of the walker, so it
/// tests the code that actually decides `b_ij`.
#[test]
fn the_uniform_grid_agrees_with_the_linear_scan_bitwise() {
    let Some(gpu) = gpu() else { return };
    let mut blocked_cube = cube_of_squares(3, Vec3::ZERO, true);
    blocked_cube.extend(cube_of_squares(1, Vec3::new(1.0, 1.0, 1.0), false));

    for (name, polys, cfg) in [
        ("shapiro", opposed_squares(true), S2sConfig { occlusion: Occlusion::PerPoint, ..open_cfg() }),
        ("blocked cube", blocked_cube, S2sConfig { occlusion: Occlusion::Pairwise, ..S2sConfig::default() }),
    ] {
        let g = CoarseGeometry::from_polygons(&polys);
        let blockers = g.blockers();
        assert!(!blockers.is_empty(), "{name}: the fixture must have blockers");
        let grid = BlockerGrid::build(&g, &blockers);
        assert!(grid.nx > 1, "{name}: the grid must have more than one cell");

        let with = ViewFactors::build_with_options(&gpu, &g, &cfg, true).expect("grid");
        let without = ViewFactors::build_with_options(&gpu, &g, &cfg, false).expect("scan");
        let (a, b) = (
            with.exchange_areas(&gpu).expect("a"),
            without.exchange_areas(&gpu).expect("b"),
        );
        for (k, (x, y)) in a.iter().zip(&b).enumerate() {
            assert_eq!(
                x.to_bits(),
                y.to_bits(),
                "{name}: G[{k}] differs between the grid ({x}) and the linear \
                 scan ({y}) - the grid is missing a blocker"
            );
        }
    }
}

/// And the counting sort that builds the grid puts every triangle in every
/// cell its bounding box overlaps - the invariant the walk relies on.
#[test]
fn the_counting_sort_places_every_triangle_in_every_cell_it_touches() {
    let mut polys = cube_of_squares(3, Vec3::ZERO, true);
    polys.extend(cube_of_squares(1, Vec3::new(1.0, 1.0, 1.0), false));
    let g = CoarseGeometry::from_polygons(&polys);
    let grid = BlockerGrid::build(&g, &g.blockers());
    let (nx, ny, nz) = (grid.nx, grid.ny, grid.nz);
    assert!(nx > 0 && ny > 0 && nz > 0);

    let idx = |i: Label, j: Label, k: Label| ((k * ny + j) * nx + i) as usize;
    for t in 0..grid.v0.len() {
        let lo = grid.v0[t].cmpt_min(grid.v1[t]).cmpt_min(grid.v2[t]) - grid.lo;
        let hi = grid.v0[t].cmpt_max(grid.v1[t]).cmpt_max(grid.v2[t]) - grid.lo;
        let c = |v: Scalar, e: Scalar, n: Label| -> Label {
            ((v * e) as i64).clamp(0, i64::from(n - 1)) as Label
        };
        for k in c(lo.z, grid.inv.z, nz)..=c(hi.z, grid.inv.z, nz) {
            for j in c(lo.y, grid.inv.y, ny)..=c(hi.y, grid.inv.y, ny) {
                for i in c(lo.x, grid.inv.x, nx)..=c(hi.x, grid.inv.x, nx) {
                    let cell = idx(i, j, k);
                    let (a, b) = (
                        grid.cell_offset[cell] as usize,
                        grid.cell_offset[cell + 1] as usize,
                    );
                    assert!(
                        grid.cell_tri[a..b].contains(&(t as Label)),
                        "triangle {t} is missing from cell ({i},{j},{k})"
                    );
                }
            }
        }
    }

    // The brute-force oracle and the intersection test agree that a ray
    // through the inner cube is blocked and one past it is not.
    let hit = grid.any_hit_brute(
        Vec3::new(1.5, 1.5, 0.0),
        Vec3::new(0.0, 0.0, 3.0),
        -1,
        -1,
    );
    assert!(hit, "a ray straight through the inner cube must be blocked");
    let miss = grid.any_hit_brute(
        Vec3::new(0.2, 0.2, 0.0),
        Vec3::new(0.0, 0.0, 3.0),
        -1,
        -1,
    );
    assert!(!miss, "a ray past the inner cube must not be blocked");
}

/// SPEC-LIT S49.4 Level 0: the convexity proof is run, not assumed.
#[test]
fn the_blocker_proof_finds_a_plate_and_clears_a_convex_box() {
    let box_only = CoarseGeometry::from_polygons(&cube_of_squares(2, Vec3::ZERO, true));
    assert!(
        box_only.blockers().is_empty(),
        "a convex box with nothing in it has no blockers"
    );

    let mut with_plate = cube_of_squares(2, Vec3::ZERO, true);
    let n_wall = with_plate.len();
    with_plate.push(vec![
        Vec3::new(0.5, 0.5, 1.0),
        Vec3::new(1.5, 0.5, 1.0),
        Vec3::new(1.5, 1.5, 1.0),
        Vec3::new(0.5, 1.5, 1.0),
    ]);
    let g = CoarseGeometry::from_polygons(&with_plate);
    assert_eq!(
        g.blockers(),
        vec![n_wall],
        "the plate and nothing but the plate"
    );
}

/// SPEC-LIT S49.6: an enclosure the case CLAIMED was closed, and is not, is
/// refused - naming the deficit, the worst surface, and both ways out.
/// Sinkhorn would otherwise smear a geometric error into a fictitious `F`.
#[test]
fn an_unclosed_enclosure_with_no_ambient_is_refused() {
    let Some(gpu) = gpu() else { return };
    let g = CoarseGeometry::from_polygons(&opposed_squares(false));
    let cfg = S2sConfig { ambient_temperature: None, ..open_cfg() };
    let m = match ViewFactors::build(&gpu, &g, &cfg) {
        Ok(_) => panic!("an unclosed enclosure was accepted"),
        Err(e) => e.to_string(),
    };
    for want in ["does not close", "ambientTemperature", "S49.6", "fictitious"] {
        assert!(m.contains(want), "the refusal must mention `{want}`: {m}");
    }
}

/// Two coplanar surfaces exchange nothing - `r` lies in their common plane,
/// so `cos(theta)` is zero. A whole agglomerated wall is made of such pairs,
/// so getting it wrong is not an edge case.
#[test]
fn coplanar_surfaces_exchange_nothing() {
    let Some(gpu) = gpu() else { return };
    let side_by_side = vec![
        vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(1.0, 1.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
        ],
        vec![
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(2.0, 0.0, 0.0),
            Vec3::new(2.0, 1.0, 0.0),
            Vec3::new(1.0, 1.0, 0.0),
        ],
    ];
    let g = CoarseGeometry::from_polygons(&side_by_side);
    let vf = ViewFactors::build(&gpu, &g, &open_cfg()).expect("vf");
    let n = vf.n_surf;
    let f = vf.view_factors(&gpu).expect("F");
    assert_eq!(f[1], 0.0, "coplanar F12 = {}", f[1]);
    assert_eq!(f[n], 0.0);
}

/// A hand-written `F` goes through the same diagnostics a computed one does,
/// and a non-reciprocal one is reported as such rather than accepted.
#[test]
fn a_hand_written_matrix_is_checked_the_same_way() {
    let Some(gpu) = gpu() else { return };
    // Concentric: F11 = 0, F12 = 1; F21 = A1/A2, F22 = 1 - A1/A2.
    let (a1, a2) = (1.0 as Scalar, 4.0 as Scalar);
    let r = a1 / a2;
    let f = vec![0.0, 1.0, r, 1.0 - r];
    let vf = ViewFactors::from_view_factors(&gpu, &f, &[a1, a2]).expect("vf");
    assert!(vf.report().rowsum_error < 1e-15);
    assert!(vf.report().reciprocity_error < 1e-15);

    let bad = vec![0.0, 1.0, 0.5 * r, 1.0 - r];
    let vb = ViewFactors::from_view_factors(&gpu, &bad, &[a1, a2]).expect("vb");
    assert!(
        vb.report().reciprocity_error > 0.1,
        "a non-reciprocal hand-written F was not reported"
    );
}

// ==========================================================================
//  S50.11  The radiosity gates
// ==========================================================================

/// **Gate 50-A.** Two infinite parallel grey plates:
/// `q = sigma(T1^4 - T2^4)/(1/e1 + 1/e2 - 1)`.
#[test]
fn gate_50a_infinite_parallel_grey_plates() {
    let Some(gpu) = gpu() else { return };
    let (t1, t2) = (800.0 as Scalar, 400.0 as Scalar);
    let f = vec![0.0, 1.0, 1.0, 0.0];
    let vf = ViewFactors::from_view_factors(&gpu, &f, &[1.0, 1.0]).expect("vf");

    for &(e1, e2) in &[(0.9, 0.9), (0.5, 0.5), (0.1, 0.1), (0.9, 0.1)] {
        let eb = vec![SIGMA_SB * t1.powi(4), SIGMA_SB * t2.powi(4)];
        let st = solve_radiosity(&gpu, &vf, &eb, &[e1, e2], 0).expect("solve");
        let want = parallel_plate_flux(t1, t2, e1, e2);
        let err = (st.q[0] - want).abs() / want.abs();
        assert!(
            err < 1e-10,
            "eps ({e1}, {e2}): q1 = {}, want {want}, rel err {err:.3e} after \
             {} sweeps",
            st.q[0],
            st.sweeps
        );
        assert!(
            (st.q[0] + st.q[1]).abs() < 1e-9 * want.abs(),
            "power balance: {} + {}",
            st.q[0],
            st.q[1]
        );
        assert!(st.residual < 1e-12, "residual {:.3e}", st.residual);
    }
}

/// **Gate 50-B.** Concentric grey bodies -
/// `q1 = sigma(T1^4 - T2^4)/[1/e1 + (A1/A2)(1/e2 - 1)]`. The better test,
/// because unequal areas exercise reciprocity.
///
/// It also verifies (S50.8)'s sweep count at `eps = 0.1`, where it is 263.
#[test]
fn gate_50b_concentric_grey_bodies() {
    let Some(gpu) = gpu() else { return };
    let (t1, t2) = (900.0 as Scalar, 350.0 as Scalar);
    for &ratio in &[0.25 as Scalar, 1.0] {
        let (a1, a2) = (1.0 as Scalar, 1.0 / ratio);
        let f = vec![0.0, 1.0, ratio, 1.0 - ratio];
        let vf = ViewFactors::from_view_factors(&gpu, &f, &[a1, a2]).expect("vf");
        for &e in &[0.1 as Scalar, 0.5, 0.9] {
            let eb = vec![SIGMA_SB * t1.powi(4), SIGMA_SB * t2.powi(4)];
            let st = solve_radiosity(&gpu, &vf, &eb, &[e, e], 0).expect("solve");
            let want = concentric_flux(t1, t2, e, e, ratio);
            let err = (st.q[0] - want).abs() / want.abs();
            assert!(
                err < 1e-10,
                "A1/A2 = {ratio}, eps = {e}: q1 = {}, want {want}, rel {err:.3e}, \
                 {} sweeps, residual {:.3e}",
                st.q[0],
                st.sweeps,
                st.residual
            );
            // Power, not flux, is what balances when the areas differ.
            assert!(
                st.net_power.abs() < 1e-9 * (a1 * want).abs(),
                "power balance {} at eps {e}",
                st.net_power
            );
            if e == 0.1 {
                assert_eq!(st.sweeps, 263, "(S50.8) at eps_min = 0.1");
                assert!(st.residual < 1e-12, "residual {:.3e} at 263 sweeps", st.residual);
            }
        }
    }
}

/// **Gate 50-C.** A three-surface enclosure with one RE-RADIATING wall
/// (`q_R = 0`), for which the series-parallel resistance network gives a
/// closed form with no symmetry to hide an error in.
///
/// Two equal parallel plates `1` and `2` of area `A`, connected by a
/// re-radiating surface `R`; `F12 = F21 = f`, `F1R = F2R = 1 - f`.
#[test]
#[allow(clippy::too_many_lines)]
fn gate_50c_three_surface_enclosure_with_a_reradiating_wall() {
    let Some(gpu) = gpu() else { return };
    let (t1, t2) = (1000.0 as Scalar, 400.0 as Scalar);
    let (e1, e2) = (0.7 as Scalar, 0.4 as Scalar);
    let a = 1.0 as Scalar;
    let f12 = 0.2 as Scalar;
    let f1r = 1.0 - f12;
    // The re-radiating surface's own area follows from reciprocity:
    // A_R F_R1 = A F_1R, and F_R1 = F_R2 = 1/2 by symmetry.
    let ar = 2.0 * a * f1r;
    let frx = a * f1r / ar; // = 0.5

    let f = vec![
        0.0, f12, f1r,
        f12, 0.0, f1r,
        frx, frx, 0.0,
    ];
    let vf = ViewFactors::from_view_factors(&gpu, &f, &[a, a, ar]).expect("vf");
    assert!(vf.report().rowsum_error < 1e-15);
    assert!(vf.report().reciprocity_error < 1e-14);

    // A re-radiating surface is adiabatic: eps -> 1 with its own temperature
    // free. Modelled here by iterating its E_b to the fixed point q_R = 0,
    // which is what the network's series-parallel reduction assumes.
    let eb1 = SIGMA_SB * t1.powi(4);
    let eb2 = SIGMA_SB * t2.powi(4);
    let mut ebr = 0.5 * (eb1 + eb2);
    let mut st = None;
    for _ in 0..200 {
        let s = solve_radiosity(&gpu, &vf, &[eb1, eb2, ebr], &[e1, e2, 1.0], 0).expect("solve");
        // q_R = eps(E_bR - H_R) = 0  <=>  E_bR = H_R.
        ebr = s.h[2];
        st = Some(s);
    }
    let st = st.expect("solved");
    assert!(st.q[2].abs() < 1e-8 * st.q[0].abs(), "q_R = {} is not zero", st.q[2]);

    // The resistance network: surface resistances (1-e)/(eA) at each end, and
    // in the middle 1/(A F12) in parallel with 1/(A F1R) + 1/(A_R F_R2).
    // PARALLEL branches add CONDUCTANCES, not resistances: the direct
    // path's conductance is `A F12`, and the path through the
    // re-radiating surface is two resistances in SERIES. The first draft
    // of this line added `1/(A F12)` instead and predicted 26139 W/m^2
    // against the solver's 15368 - a 41% gap that was the TEST's error,
    // not the solver's.
    let r_par = 1.0 / (a * f12 + 1.0 / (1.0 / (a * f1r) + 1.0 / (ar * frx)));
    let r_tot = (1.0 - e1) / (e1 * a) + r_par + (1.0 - e2) / (e2 * a);
    let want = (eb1 - eb2) / r_tot / a;
    let err = (st.q[0] - want).abs() / want.abs();
    assert!(
        err < 1e-8,
        "three-surface: q1 = {}, network says {want}, rel {err:.3e}",
        st.q[0]
    );
    assert!(
        st.net_power.abs() < 1e-8 * (a * want).abs(),
        "power balance {}",
        st.net_power
    );
}

/// SPEC-LIT S50.10: `SUM_i A_i q_i = 0` in a closed enclosure at ANY
/// temperatures - the model's own conservation statement, on a real
/// enclosure whose `F` came from the quadrature rather than from a formula.
#[test]
fn power_balances_in_a_computed_enclosure_at_arbitrary_temperatures() {
    let Some(gpu) = gpu() else { return };
    let g = CoarseGeometry::from_polygons(&cube_of_squares(3, Vec3::ZERO, true));
    let vf = ViewFactors::build(&gpu, &g, &S2sConfig::default()).expect("vf");
    let n = vf.n_surf;
    // A deliberately lumpy temperature field: nothing symmetric to cancel.
    let eb: Vec<Scalar> = (0..n)
        .map(|i| {
            let t = 300.0 + 40.0 * ((i * 37) % 23) as Scalar;
            SIGMA_SB * t.powi(4)
        })
        .collect();
    let eps: Vec<Scalar> = (0..n).map(|i| 0.25 + 0.7 * ((i % 5) as Scalar / 4.0)).collect();
    let st = solve_radiosity(&gpu, &vf, &eb, &eps, 0).expect("solve");
    let gross: Scalar = st.q.iter().zip(vf.areas()).map(|(q, a)| (q * a).abs()).sum();
    assert!(
        st.net_power.abs() < 1e-11 * gross,
        "net power {} against gross {gross} (residual {:.3e})",
        st.net_power,
        st.residual
    );
}

/// SPEC-LIT S50.2's table is a claim about (S50.8), and this checks the
/// formula rather than a code path.
#[test]
fn the_sweep_count_matches_the_published_table() {
    for &(eps, want) in &[
        (0.95 as Scalar, 10usize),
        (0.90, 12),
        (0.80, 18),
        (0.50, 40),
        (0.30, 78),
        (0.10, 263),
        (0.05, 539),
    ] {
        let got = radiosity_sweeps(eps, 1e-12);
        assert_eq!(got, want, "eps_min = {eps}: (S50.8) gives {got}, table says {want}");
    }
    // A black enclosure converges in one sweep: J = E_b exactly.
    assert_eq!(radiosity_sweeps(1.0, 1e-12), 1);
    // And S50.2's refusal threshold really does exceed 1300 sweeps.
    assert!(radiosity_sweeps(EPS_MIN_SUPPORTED, 1e-12) > 1300);
}

// ==========================================================================
//  S50.3, S50.4  The Robin triple
// ==========================================================================

/// SPEC-LIT S50.10: `fr` in `[0,1)` for every input, swept rather than
/// argued. This is strictly better behaved than the Marshak triple a
/// participating-medium model needs, which needed a sign argument to land in
/// range.
#[test]
fn the_triple_lands_in_range_for_every_input() {
    for &eps in &[0.0 as Scalar, 0.01, 0.3, 0.7, 1.0] {
        for &t0 in &[100.0 as Scalar, 300.0, 1200.0, 3000.0] {
            for &k in &[1e-4 as Scalar, 0.026, 1.0, 400.0] {
                for &d in &[1e-3 as Scalar, 1.0, 1e3, 1e6] {
                    let (fr, rv, rg) = s2s_triple(eps, t0, 0.0, 0.0, k, d);
                    assert!(
                        (0.0..1.0).contains(&fr),
                        "fr = {fr} at eps {eps}, T0 {t0}, k {k}, delta {d}"
                    );
                    assert!(rv.is_finite() && rg.is_finite());
                }
            }
        }
    }
}

/// SPEC-LIT S50.4 check 2 and S50.10: at `eps = 0` the triple is **bitwise**
/// `fixedFluxTemperature`. That is the whole reason `refGrad = q_ext/k_eff`
/// was chosen over `refGrad = 0` - with `refGrad = 0` the emissivity would
/// reappear in `refValue` and the collapse would be approximate.
#[test]
fn a_zero_emissivity_wall_is_bitwise_the_fixed_flux_condition() {
    for &q in &[0.0 as Scalar, -250.0, 1e4] {
        for &k in &[0.026 as Scalar, 45.0] {
            let (fr, _rv, rg) = s2s_triple(0.0, 350.0, 1234.0, q, k, 200.0);
            assert_eq!(fr.to_bits(), (0.0 as Scalar).to_bits(), "fr is not exactly zero");
            assert_eq!(
                rg.to_bits(),
                crate::energy::flux_to_grad(q, k).to_bits(),
                "refGrad is not bitwise flux_to_grad(q, k_eff)"
            );
        }
    }
}

/// SPEC-LIT S50.10: the emissivity does not reach `refValue` at all.
#[test]
fn the_emissivity_does_not_reach_the_reference_value() {
    let (_, a, _) = s2s_triple(0.1, 420.0, 3000.0, 0.0, 1.0, 100.0);
    let (_, b, _) = s2s_triple(0.9, 420.0, 3000.0, 0.0, 1.0, 100.0);
    assert_eq!(a.to_bits(), b.to_bits(), "refValue moved with eps: {a} vs {b}");
    // ...but `fr` does, which is what carries the emission into the matrix.
    let (fa, _, _) = s2s_triple(0.1, 420.0, 3000.0, 0.0, 1.0, 100.0);
    let (fb, _, _) = s2s_triple(0.9, 420.0, 3000.0, 0.0, 1.0, 100.0);
    assert!(fb > fa * 5.0, "fr did not scale with eps: {fa} vs {fb}");
}

/// SPEC-LIT S50.4 check 3: refining the mesh does NOT lose the radiation.
/// The quantity that reaches the matrix is `fr Delta_b -> h/k_eff`, a finite
/// radiative conductance - unlike a participating medium's Marshak triple,
/// which degenerates.
#[test]
fn refinement_keeps_a_finite_radiative_conductance() {
    let (eps, t0, k) = (0.85 as Scalar, 500.0 as Scalar, 0.04 as Scalar);
    let h = 4.0 * eps * SIGMA_SB * t0.powi(3);
    let want = h / k;
    let mut prev = Scalar::INFINITY;
    for d in [1e2 as Scalar, 1e3, 1e4, 1e5, 1e6] {
        let (fr, _, _) = s2s_triple(eps, t0, 0.0, 0.0, k, d);
        let cond = fr * d;
        let err = (want - cond) / want;
        assert!(err < prev, "fr*Delta is not converging: {err:.3e} after {prev:.3e}");
        prev = err;
        // The approach is exactly first order in 1/Delta_b, and the constant
        // is `h/k` itself:  fr Delta = h/(h/Delta + k), so the relative gap is
        // (h/k)/(Delta + h/k). Asserting the RATE rather than a bare
        // tolerance is what makes this a statement about the formula.
        let exact = (h / k) / (d + h / k);
        assert!(
            (err - exact).abs() < 1e-12 * exact,
            "Delta = {d}: gap {err:.6e}, first-order prediction {exact:.6e}"
        );
    }
    // Measured 6.0e-4 at Delta_b = 1e6 with h = 24.1 W/m^2K and k = 0.04 -
    // a FINITE radiative conductance, which is the whole point (a Marshak
    // triple degenerates to zero-gradient in the same limit).
    assert!(prev < 1e-3, "fr*Delta_b -> h/k_eff only to {prev:.3e}");
    assert!(
        (h / k - 602.0).abs() < 1.0,
        "the fixture's own h/k moved; re-record the 6.0e-4"
    );
}

/// SPEC-LIT S50.4 check 4: a black isothermal enclosure at `T_inf` has
/// `T0 = T_inf` as an exact fixed point of the triple.
#[test]
fn radiative_equilibrium_is_a_fixed_point_of_the_triple() {
    let t_inf = 640.0 as Scalar;
    let h_b = SIGMA_SB * t_inf.powi(4);
    let (_, rv, rg) = s2s_triple(1.0, t_inf, h_b, 0.0, 0.03, 500.0);
    assert!((rv - t_inf).abs() < 1e-11 * t_inf, "refValue = {rv}, want {t_inf}");
    assert_eq!(rg, 0.0);

    // And it is a CONTRACTION: start off the fixed point and the map pulls
    // back toward it.
    let mut t = 900.0 as Scalar;
    for _ in 0..200 {
        let (_, rv, _) = s2s_triple(1.0, t, h_b, 0.0, 0.03, 500.0);
        t = rv;
    }
    assert!((t - t_inf).abs() < 1e-8 * t_inf, "iterate settled at {t}, want {t_inf}");
}

// ==========================================================================
//  S50.5  Coarse and fine
// ==========================================================================

/// A small closed box mesh with every wall radiating - the fixture the
/// mesh-side tests need.
fn box_rig(gpu: &Gpu, n: [usize; 3], eps: Scalar) -> (crate::mesh::HostMesh, GpuMesh, Vec<Vec3>, Vec<Vec<Label>>, RadiantFaces) {
    use crate::blockgen::{BlockSpec, GradedAxis};
    let axis = |lo: Scalar, hi: Scalar, n: usize| GradedAxis {
        lo,
        hi,
        n,
        expansion: 1.0,
        two_sided: false,
    };
    let spec = BlockSpec {
        x: axis(0.0, 0.1, n[0]),
        y: axis(0.0, 0.1, n[1]),
        z: axis(0.0, 0.1, n[2]),
        ..BlockSpec::default()
    };
    let raw = crate::blockgen::raw_mesh(&spec).expect("raw");
    let mut hm = crate::io::polymesh::build_host_mesh(&raw).expect("host mesh");
    hm.build_cell_face_maps();
    let gm = GpuMesh::upload(gpu, &hm).expect("upload");
    let sel = RadiantFaces {
        radiating: vec![true; hm.n_boundary_faces],
        emissivity: vec![eps; hm.n_boundary_faces],
        q_ext: vec![0.0; hm.n_boundary_faces],
    };
    (hm, gm, raw.points.clone(), raw.faces.clone(), sel)
}

/// SPEC-LIT S49.3's central claim, made concrete: the enclosure built from a
/// MESH closes. The mesh's `Sf` points out of the fluid; the radiating normal
/// faces the cavity, and `SurfaceGeometry::build` reverses the winding to get
/// it. Without that every view factor is zero and the model computes nothing.
#[test]
fn a_box_mesh_becomes_a_closed_enclosure() {
    let Some(gpu) = gpu() else { return };
    let (hm, _gm, points, faces, sel) = box_rig(&gpu, [3, 3, 3], 0.8);
    let surf = SurfaceGeometry::build(&hm, &points, &faces, &sel).expect("surface");
    assert_eq!(surf.n, 6 * 9);
    // Planar faces, so the triangulated area is |Sf| to round-off.
    assert!(surf.area_defect(&hm) < 1e-14, "area defect {}", surf.area_defect(&hm));

    // The normals point INTO the box: every centroid-to-box-centre vector has
    // a positive projection on the radiating normal.
    let mid = Vec3::new(0.05, 0.05, 0.05);
    for s in 0..surf.n {
        let d = (mid - surf.centroid[s]).dot(surf.normal[s]);
        assert!(d > 0.0, "face {s} radiates away from the cavity ({d})");
    }

    let cl = Clustering::identity(surf.n);
    let cg = CoarseGeometry::build(&surf, &cl);
    assert!(cg.blockers().is_empty(), "a box has no internal blockers");
    let vf = ViewFactors::build(&gpu, &cg, &S2sConfig::default()).expect("vf");
    assert!(
        vf.report().rowsum_error < 1e-4,
        "a closed box does not close: {:.3e}",
        vf.report().rowsum_error
    );
}

/// SPEC-LIT S50.10: the coarse `E_b` is the area-weighted mean of
/// `sigma T^4`, **not** `sigma <T>^4`. Averaging `T` first understates a
/// non-isothermal cluster's emission by Jensen's inequality, and the test
/// measures the gap so a regression cannot pass.
#[test]
fn the_coarse_emissive_power_averages_sigma_t4_and_not_t() {
    let Some(gpu) = gpu() else { return };
    let (hm, gm, points, faces, sel) = box_rig(&gpu, [2, 2, 2], 0.9);
    let cfg = S2sConfig { agglomerate: 4, ..S2sConfig::default() };
    let mut s2s = S2s::new(&gpu, &gm, &hm, &points, &faces, &sel, cfg).expect("s2s");
    assert!(s2s.clustering().n_coarse < s2s.n_fine(), "agglomerate 4 did nothing");

    // A hot half and a cold half on every cluster.
    let mut t = GpuScalarField::zeros(&gpu, &gm, "T").expect("T");
    let nbf = hm.n_boundary_faces;
    let tb: Vec<Scalar> = (0..nbf)
        .map(|bf| if bf % 2 == 0 { 300.0 } else { 900.0 })
        .collect();
    gpu.write(&mut t.bf, &tb).expect("bf");
    let k_eff: DevBuf<Scalar> = gpu.upload(&vec![0.03 as Scalar; nbf]).expect("k");
    s2s.update(&gpu, &mut t, &k_eff).expect("update");

    let (_, _, _) = s2s.state(&gpu).expect("state");
    let eb = s2s.coarse_emissive_power(&gpu).expect("eb");
    let cl = s2s.clustering();
    for c in 0..cl.n_coarse {
        let members: Vec<usize> = cl.member[cl.offset[c] as usize..cl.offset[c + 1] as usize]
            .iter()
            .map(|&m| m as usize)
            .collect();
        if members.len() < 2 {
            continue;
        }
        let mut a = 0.0 as Scalar;
        let mut p = 0.0 as Scalar;
        let mut tt = 0.0 as Scalar;
        for &m in &members {
            let bf = s2s.b_face_of(m) as usize;
            let s = hm.b_mag_sf[bf];
            a += s;
            p += s * SIGMA_SB * tb[bf].powi(4);
            tt += s * tb[bf];
        }
        let want = p / a;
        let wrong = SIGMA_SB * (tt / a).powi(4);
        assert!(
            (eb[c] - want).abs() < 1e-9 * want,
            "cluster {c}: E_b = {}, area-weighted sigma T^4 = {want}",
            eb[c]
        );
        assert!(
            (want - wrong).abs() > 0.3 * want,
            "the two averages are indistinguishable on this fixture ({want} vs \
             {wrong}); the test proves nothing"
        );
        return;
    }
    panic!("no cluster had more than one member");
}

/// SPEC-LIT S50.10: at `agglomerate 1` the coarse gather followed by the
/// broadcast is the identity, bitwise.
#[test]
fn the_coarse_fine_round_trip_is_the_identity_at_agglomerate_one() {
    let Some(gpu) = gpu() else { return };
    let (hm, gm, points, faces, sel) = box_rig(&gpu, [2, 2, 2], 0.6);
    let mut s2s =
        S2s::new(&gpu, &gm, &hm, &points, &faces, &sel, S2sConfig::default()).expect("s2s");
    assert_eq!(s2s.clustering().n_coarse, s2s.n_fine());

    let mut t = GpuScalarField::zeros(&gpu, &gm, "T").expect("T");
    let nbf = hm.n_boundary_faces;
    let tb: Vec<Scalar> = (0..nbf).map(|bf| 300.0 + bf as Scalar).collect();
    gpu.write(&mut t.bf, &tb).expect("bf");
    let k_eff: DevBuf<Scalar> = gpu.upload(&vec![0.03 as Scalar; nbf]).expect("k");
    s2s.update(&gpu, &mut t, &k_eff).expect("update");

    let (_, h_coarse, _) = s2s.state(&gpu).expect("state");
    let h_fine = s2s.irradiation_fine(&gpu).expect("hf");
    for s in 0..s2s.n_fine() {
        let c = s2s.clustering().cluster_of[s] as usize;
        assert_eq!(
            h_fine[s].to_bits(),
            h_coarse[c].to_bits(),
            "slot {s} is not a bitwise copy of its cluster"
        );
    }
}

/// SPEC-LIT S50.10: two `update` calls on the same state produce bitwise
/// identical triples.
#[test]
fn two_updates_write_bitwise_identical_triples() {
    let Some(gpu) = gpu() else { return };
    let (hm, gm, points, faces, sel) = box_rig(&gpu, [2, 2, 2], 0.55);
    let mut s2s =
        S2s::new(&gpu, &gm, &hm, &points, &faces, &sel, S2sConfig::default()).expect("s2s");
    let nbf = hm.n_boundary_faces;
    let k_eff: DevBuf<Scalar> = gpu.upload(&vec![0.03 as Scalar; nbf]).expect("k");

    let run = |s2s: &mut S2s| {
        let mut t = GpuScalarField::zeros(&gpu, &gm, "T").expect("T");
        let tb: Vec<Scalar> = (0..nbf).map(|bf| 320.0 + (bf % 7) as Scalar * 40.0).collect();
        gpu.write(&mut t.bf, &tb).expect("bf");
        s2s.update(&gpu, &mut t, &k_eff).expect("update");
        (
            gpu.download(&t.fr).expect("fr"),
            gpu.download(&t.ref_value).expect("rv"),
            gpu.download(&t.ref_grad).expect("rg"),
        )
    };
    let a = run(&mut s2s);
    let b = run(&mut s2s);
    for k in 0..nbf {
        assert_eq!(a.0[k].to_bits(), b.0[k].to_bits(), "fr[{k}]");
        assert_eq!(a.1[k].to_bits(), b.1[k].to_bits(), "refValue[{k}]");
        assert_eq!(a.2[k].to_bits(), b.2[k].to_bits(), "refGrad[{k}]");
    }
}

/// The stamp writes exactly (S50.12), which is what makes `s2s_triple` a
/// statement about the kernel and not a second implementation.
#[test]
fn the_stamp_writes_the_s50_12_triple() {
    let Some(gpu) = gpu() else { return };
    let eps = 0.72 as Scalar;
    let (hm, gm, points, faces, mut sel) = box_rig(&gpu, [2, 2, 2], eps);
    let nbf = hm.n_boundary_faces;
    for bf in 0..nbf {
        sel.q_ext[bf] = 100.0 * (bf % 3) as Scalar;
    }
    let mut s2s =
        S2s::new(&gpu, &gm, &hm, &points, &faces, &sel, S2sConfig::default()).expect("s2s");

    let mut t = GpuScalarField::zeros(&gpu, &gm, "T").expect("T");
    let tb: Vec<Scalar> = (0..nbf).map(|bf| 350.0 + (bf % 5) as Scalar * 30.0).collect();
    gpu.write(&mut t.bf, &tb).expect("bf");
    let kw = 0.031 as Scalar;
    let k_eff: DevBuf<Scalar> = gpu.upload(&vec![kw; nbf]).expect("k");
    s2s.update(&gpu, &mut t, &k_eff).expect("update");

    let fr = gpu.download(&t.fr).expect("fr");
    let rv = gpu.download(&t.ref_value).expect("rv");
    let rg = gpu.download(&t.ref_grad).expect("rg");
    let hf = s2s.irradiation_fine(&gpu).expect("hf");
    let delta = gpu.download(&gm.b_delta_coeffs).expect("delta");

    let mut worst: Scalar = 0.0;
    for s in 0..s2s.n_fine() {
        let bf = s2s.b_face_of(s) as usize;
        let (f, v, g) = s2s_triple(eps, tb[bf], hf[s], sel.q_ext[bf], kw, delta[bf]);
        worst = worst
            .max((fr[bf] - f).abs())
            .max((rv[bf] - v).abs() / v.abs())
            .max((rg[bf] - g).abs() / g.abs().max(1.0));
    }
    assert!(worst < 1e-14, "the stamp departs from (S50.12) by {worst:.3e}");
}

/// SPEC-LIT S50.10, and the §13.4.1 pair test for `q`: two identical rigs
/// differing only in the external flux must produce different `refGrad`.
#[test]
fn the_external_flux_reaches_the_boundary_condition() {
    let Some(gpu) = gpu() else { return };
    let leg = |q: Scalar| -> Vec<Scalar> {
        let (hm, gm, points, faces, mut sel) = box_rig(&gpu, [2, 2, 2], 0.8);
        for e in sel.q_ext.iter_mut() {
            *e = q;
        }
        let mut s2s =
            S2s::new(&gpu, &gm, &hm, &points, &faces, &sel, S2sConfig::default()).expect("s2s");
        let mut t = GpuScalarField::zeros(&gpu, &gm, "T").expect("T");
        let nbf = hm.n_boundary_faces;
        gpu.write(&mut t.bf, &vec![400.0 as Scalar; nbf]).expect("bf");
        let k: DevBuf<Scalar> = gpu.upload(&vec![0.03 as Scalar; nbf]).expect("k");
        s2s.update(&gpu, &mut t, &k).expect("update");
        gpu.download(&t.ref_grad).expect("rg")
    };
    let a = leg(0.0);
    let b = leg(500.0);
    let moved = a.iter().zip(&b).map(|(x, y)| (x - y).abs()).fold(0.0 as Scalar, Scalar::max);
    assert!(
        moved > 1.0,
        "`q` = 0 and `q` = 500 W/m^2 gave the same refGrad; the external flux \
         is being ignored"
    );
    assert!((b[0] - 500.0 / 0.03).abs() < 1e-9 * (500.0 / 0.03), "refGrad != q/k_eff");
}

// ==========================================================================
//  S51  The dictionary, its refusals, and the pair tests
// ==========================================================================

fn dict(body: &str) -> FoamDict {
    FoamDict::parse(body, "<memory>").expect("dict")
}

#[test]
fn the_dictionary_reads_every_entry_s51_1_names() {
    let d = dict(
        "radiationModel viewFactor;\n\
         emissivity 0.83;\n\
         viewFactorQuadrature 6;\n\
         occlusion perPoint;\n\
         agglomerate 9;\n\
         maxClusterAngle 35;\n\
         ambientTemperature 293.15;\n\
         radiationRelaxation 0.4;\n\
         radiositySweeps 25;\n",
    );
    let c = S2sConfig::from_dict(&d).expect("config");
    assert_eq!(c.emissivity, 0.83);
    assert_eq!(c.quadrature, 6);
    assert_eq!(c.occlusion, Occlusion::PerPoint);
    assert_eq!(c.agglomerate, 9);
    assert_eq!(c.max_cluster_angle_deg, 35.0);
    assert_eq!(c.ambient_temperature, Some(293.15));
    assert_eq!(c.relaxation, 0.4);
    assert_eq!(c.sweeps, 25);
}

/// SPEC-LIT S13.4: every entry that can be wrong is refused BY NAME, with the
/// alternatives listed. A silent substitution produces a plausible wrong
/// answer, which is worse than no answer.
#[test]
fn every_bad_entry_is_refused_by_name() {
    crate::io::contract::reset_warnings();
    let cases: [(&str, &[&str]); 9] = [
        ("radiationModel viewFactor;\n", &["emissivity", "no honest default"]),
        ("emissivity 1.4;\n", &["emissivity", "[0, 1]"]),
        ("emissivity 0.9;\nviewFactorQuadrature 11;\n", &["viewFactorQuadrature", "S49.2"]),
        ("emissivity 0.9;\nocclusion sometimes;\n", &["occlusion", "pairwise", "perPoint"]),
        ("emissivity 0.9;\nagglomerate 0;\n", &["agglomerate", "at least 1"]),
        ("emissivity 0.9;\nmaxClusterAngle 120;\n", &["maxClusterAngle", "(0, 90)"]),
        ("emissivity 0.9;\nambientTemperature -5;\n", &["ambientTemperature", "ABSOLUTE"]),
        ("emissivity 0.9;\nradiationRelaxation 1.5;\n", &["radiationRelaxation", "(0, 1]"]),
        (
            "emissivity 0.9;\nabsorptionCoefficient 0.5;\n",
            &["NON-PARTICIPATING", "P1", "fvDOM"],
        ),
    ];
    for (body, wants) in cases {
        let e = S2sConfig::from_dict(&dict(body)).expect_err(&format!("must refuse: {body}"));
        let m = e.to_string();
        for w in wants {
            assert!(m.contains(w), "the refusal for `{body}` must name `{w}`: {m}");
        }
    }
}

/// SPEC-LIT S50.2: a nearly-perfect reflector is refused rather than
/// truncated to a wrong answer, and the message names the sweep count, the
/// arithmetic and the way out.
#[test]
fn a_nearly_specular_emissivity_is_refused_naming_the_sweep_count() {
    let Some(gpu) = gpu() else { return };
    let (hm, gm, points, faces, sel) = box_rig(&gpu, [2, 2, 2], 0.005);
    let m = match S2s::new(&gpu, &gm, &hm, &points, &faces, &sel, S2sConfig::default()) {
        Ok(_) => panic!("a nearly-specular emissivity was accepted"),
        Err(e) => e.to_string(),
    };
    for want in ["SPECULAR", "radiositySweeps", "S50.9", "sweeps"] {
        assert!(m.contains(want), "the refusal must name `{want}`: {m}");
    }
    // ...and `radiositySweeps` really is the documented way out.
    let cfg = S2sConfig { sweeps: 4000, ..S2sConfig::default() };
    S2s::new(&gpu, &gm, &hm, &points, &faces, &sel, cfg).expect("explicit sweeps must be honoured");
}

/// SPEC-LIT S50.6: `N_c` above the cap is refused before anything is
/// allocated, naming the arithmetic and the agglomeration level.
#[test]
fn an_oversized_enclosure_is_refused_with_the_arithmetic() {
    let Some(gpu) = gpu() else { return };
    let n = MAX_COARSE_FACES + 1;
    let m = match ViewFactors::from_view_factors(&gpu, &[], &vec![1.0 as Scalar; n]) {
        Ok(_) => panic!("an oversized enclosure was accepted"),
        Err(e) => e.to_string(),
    };
    for want in ["coarse radiating faces", "agglomerate", "GB"] {
        assert!(m.contains(want), "the refusal must name `{want}`: {m}");
    }
}

/// SPEC-LIT S51.2's pair tests, driven from CASE DOCUMENTS that differ in
/// exactly one entry and are REQUIRED to produce different output.
///
/// Six instances of "a case could say it and the solver ignored it" have been
/// found in this project; this is what stops the seventh.
#[test]
#[allow(clippy::too_many_lines)]
fn the_s13_4_1_pair_tests() {
    let Some(gpu) = gpu() else { return };
    let base = "radiationModel viewFactor;\nemissivity 0.5;\n";

    // ---- emissivity: J, q and fr must all move ------------------------
    {
        let g = CoarseGeometry::from_polygons(&cube_of_squares(2, Vec3::ZERO, true));
        let vf = ViewFactors::build(&gpu, &g, &S2sConfig::default()).expect("vf");
        let n = vf.n_surf;
        let eb: Vec<Scalar> = (0..n)
            .map(|i| SIGMA_SB * (300.0 + 200.0 * (i % 3) as Scalar).powi(4))
            .collect();
        let leg = |body: &str| {
            let c = S2sConfig::from_dict(&dict(body)).expect("cfg");
            solve_radiosity(&gpu, &vf, &eb, &vec![c.emissivity; n], 0).expect("solve")
        };
        let lo = leg("radiationModel viewFactor;\nemissivity 0.2;\n");
        let hi = leg("radiationModel viewFactor;\nemissivity 0.8;\n");
        assert!(
            (lo.j[0] - hi.j[0]).abs() > 1.0,
            "`emissivity` 0.2 and 0.8 gave the same radiosity"
        );
        assert!(
            (lo.q[0] - hi.q[0]).abs() > 1.0,
            "`emissivity` 0.2 and 0.8 gave the same net flux"
        );
        let (fa, _, _) = s2s_triple(0.2, 500.0, 0.0, 0.0, 0.03, 300.0);
        let (fb, _, _) = s2s_triple(0.8, 500.0, 0.0, 0.0, 0.03, 300.0);
        assert!((fa - fb).abs() > 1e-6, "`emissivity` did not move fr");
    }

    // ---- ambientTemperature: the irradiation and the net flux move ----
    {
        let g = CoarseGeometry::from_polygons(&opposed_squares(false));
        let leg = |body: &str| {
            let c = S2sConfig::from_dict(&dict(body)).expect("cfg");
            let vf = ViewFactors::build(&gpu, &g, &c).expect("vf");
            let n = vf.n_surf;
            let t_amb = c.ambient_temperature.expect("ambient");
            let mut eb = vec![SIGMA_SB * (600.0 as Scalar).powi(4); n];
            eb[n - 1] = SIGMA_SB * t_amb.powi(4);
            let mut eps = vec![c.emissivity; n];
            eps[n - 1] = 1.0;
            solve_radiosity(&gpu, &vf, &eb, &eps, 0).expect("solve")
        };
        let cold = leg(&format!("{base}ambientTemperature 300;\n"));
        let hot = leg(&format!("{base}ambientTemperature 600;\n"));
        assert!(
            (cold.h[0] - hot.h[0]).abs() > 1.0,
            "`ambientTemperature` 300 and 600 gave the same irradiation"
        );
        assert!(
            (cold.q[0] - hot.q[0]).abs() > 1.0,
            "`ambientTemperature` did not move the net flux"
        );
    }

    // ---- radiositySweeps: J must move when the count is cut to one ----
    {
        let g = CoarseGeometry::from_polygons(&cube_of_squares(2, Vec3::ZERO, true));
        let vf = ViewFactors::build(&gpu, &g, &S2sConfig::default()).expect("vf");
        let n = vf.n_surf;
        let eb: Vec<Scalar> = (0..n)
            .map(|i| SIGMA_SB * (300.0 + 300.0 * (i % 2) as Scalar).powi(4))
            .collect();
        let one = S2sConfig::from_dict(&dict(&format!("{base}radiositySweeps 1;\n"))).expect("c");
        let many = S2sConfig::from_dict(&dict(base)).expect("c");
        let a = solve_radiosity(&gpu, &vf, &eb, &vec![0.5; n], one.sweeps).expect("a");
        let b = solve_radiosity(&gpu, &vf, &eb, &vec![0.5; n], many.sweeps).expect("b");
        assert!(a.sweeps != b.sweeps);
        assert!(
            (a.j[0] - b.j[0]).abs() > 1.0,
            "`radiositySweeps` 1 and {} gave the same radiosity",
            b.sweeps
        );
        assert!(a.residual > b.residual, "one sweep is not worse than {}", b.sweeps);
    }

    // ---- agglomerate: N_c and F must move -----------------------------
    {
        let (hm, gm, points, faces, sel) = box_rig(&gpu, [4, 4, 4], 0.6);
        let leg = |body: &str| {
            let c = S2sConfig::from_dict(&dict(body)).expect("cfg");
            let s = S2s::new(&gpu, &gm, &hm, &points, &faces, &sel, c).expect("s2s");
            // The LARGEST view factor in the matrix: a scalar summary that is
            // not zero by symmetry the way `F[0][1]` is (two faces of the
            // same wall are coplanar and exchange nothing, whatever the
            // agglomeration - which is what the first draft of this line
            // compared, and it read 0 against 0).
            let f = s.view_factors().view_factors(&gpu).expect("F");
            (
                s.clustering().n_coarse,
                f.iter().fold(0.0 as Scalar, |m, &v| m.max(v)),
            )
        };
        let one = leg(&format!("{base}agglomerate 1;\n"));
        let four = leg(&format!("{base}agglomerate 4;\n"));
        assert!(
            four.0 < one.0,
            "`agglomerate` 4 produced {} coarse faces, the same as 1",
            four.0
        );
        assert!(
            (one.1 - four.1).abs() > 1e-6,
            "`agglomerate` did not change F (max F {} vs {})",
            one.1,
            four.1
        );
    }

    // ---- maxClusterAngle: the cluster count must move -----------------
    //
    // A BOX cannot test this: every face of a patch is coplanar with every
    // other, so the normals agree to zero degrees and no angle limit ever
    // bites. The first draft of this leg used one and read 6 clusters against
    // 6 - the test would have passed a solver that ignored the entry
    // entirely, which is exactly the defect S13.4.1 exists to catch. The
    // fixture is a twelve-sided prism's side wall: one patch, adjacent faces
    // 30 degrees apart, which is what a curved radiating surface looks like.
    {
        let strip = prism_strip(12);
        let hm = one_patch_mesh(strip.n);
        let leg = |body: &str| {
            let c = S2sConfig::from_dict(&dict(body)).expect("cfg");
            Clustering::agglomerate(&strip, &hm, c.agglomerate, c.max_cluster_angle_deg).n_coarse
        };
        let tight = leg(&format!("{base}agglomerate 64;\nmaxClusterAngle 20;\n"));
        let loose = leg(&format!("{base}agglomerate 64;\nmaxClusterAngle 89;\n"));
        assert_eq!(
            tight, strip.n,
            "at 20 degrees no two 30-degree-apart faces may merge, so every \
             face is its own cluster"
        );
        assert!(
            loose < tight,
            "`maxClusterAngle` 20 and 89 both gave {tight} clusters; the entry \
             is being ignored"
        );
    }

    // ---- radiationRelaxation: the irradiation after one update -------
    {
        let leg = |body: &str| {
            let (hm, gm, points, faces, sel) = box_rig(&gpu, [2, 2, 2], 0.6);
            let c = S2sConfig::from_dict(&dict(body)).expect("cfg");
            let mut s = S2s::new(&gpu, &gm, &hm, &points, &faces, &sel, c).expect("s2s");
            let mut t = GpuScalarField::zeros(&gpu, &gm, "T").expect("T");
            let nbf = hm.n_boundary_faces;
            let tb: Vec<Scalar> = (0..nbf).map(|b| 300.0 + 400.0 * (b % 2) as Scalar).collect();
            gpu.write(&mut t.bf, &tb).expect("bf");
            let k: DevBuf<Scalar> = gpu.upload(&vec![0.03 as Scalar; nbf]).expect("k");
            s.update(&gpu, &mut t, &k).expect("update");
            s.irradiation_fine(&gpu).expect("hf")
        };
        let full = leg(&format!("{base}radiationRelaxation 1.0;\n"));
        let slow = leg(&format!("{base}radiationRelaxation 0.3;\n"));
        let moved = full
            .iter()
            .zip(&slow)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0 as Scalar, Scalar::max);
        assert!(
            moved > 1.0,
            "`radiationRelaxation` 1.0 and 0.3 gave the same irradiation"
        );
    }

    // ---- radiationModel: a different model is CONSTRUCTED -------------
    {
        use crate::radiation::RadiationModel;
        assert_eq!(RadiationModel::from_name("P1").expect("P1"), RadiationModel::P1);
        assert_eq!(
            RadiationModel::from_name("viewFactor").expect("viewFactor"),
            RadiationModel::ViewFactor
        );
        assert_ne!(RadiationModel::P1, RadiationModel::ViewFactor);
    }
}

// ==========================================================================
//  S50.8  The boundary condition's name
// ==========================================================================

/// SPEC-LIT S50.8: the radiating wall belongs on the TEMPERATURE field, and
/// on any other field it would be zero-gradient wearing its name - the S13.4
/// defect this project keeps finding.
#[test]
fn a_radiating_wall_belongs_on_a_temperature_and_nowhere_else() {
    crate::io::contract::reset_warnings();
    for name in ["greyDiffusiveRadiationViewFactor", "s2sWall"] {
        assert_eq!(
            BcKind::from_name(name, "T", "lid").expect("T accepts it"),
            BcKind::S2sWall
        );
        assert_eq!(
            BcKind::from_name(name, "T.air", "lid").expect("T.<region> accepts it"),
            BcKind::S2sWall
        );
        let e = BcKind::from_name(name, "U", "lid").expect_err("must be refused on U");
        let m = e.to_string();
        assert!(m.contains("TEMPERATURE"), "{m}");
        assert!(m.contains(name), "{m}");
    }
}

/// And the discriminant is outside every range `cuda/field.cu` consults, so
/// `fldCorrectBcScalar` evaluates it with the same `fldMixed` as everything
/// else - SPEC-LIT S50.8's "no new device branch".
#[test]
fn the_s2s_wall_kind_needs_no_device_branch() {
    let v = BcKind::S2sWall as Label;
    assert_eq!(v, 34);
    assert_ne!(v, BcKind::Calculated as Label);
    assert_ne!(v, BcKind::Cyclic as Label);
    assert_ne!(v, BcKind::Symmetry as Label);
    assert!(!(crate::field::FLUX_SWITCHED_FIRST..=crate::field::FLUX_SWITCHED_LAST).contains(&v));
    assert!(BcKind::S2sWall.is_s2s_wall());
    assert!(!BcKind::CoupledTemperature.is_s2s_wall());
    // S50.8: a face carries the conjugate interface or the radiating wall,
    // never both, because they rewrite the same three numbers.
    assert!(!BcKind::S2sWall.is_coupled_temperature());
    assert!(!BcKind::S2sWall.is_thermal_wall_function());
    assert!(!BcKind::S2sWall.is_fixed_flux_temperature());
}

/// SPEC-LIT S50.8: the OpenFOAM name that asks for BOTH the conjugate
/// coupling and the radiative exchange on one face is still refused, and the
/// refusal now names the radiating wall as one of the two conditions that
/// exist.
#[test]
fn the_combined_conjugate_and_radiative_name_names_both_conditions() {
    crate::io::contract::reset_warnings();
    let e = BcKind::from_name(
        "compressible::turbulentTemperatureRadCoupledMixed",
        "T",
        "interface",
    )
    .expect_err("must be refused");
    let m = e.to_string();
    assert!(m.contains("greyDiffusiveRadiationViewFactor"), "{m}");
    assert!(m.contains("coupledTemperature"), "{m}");
    assert!(m.contains("never"), "the message must say a face carries one or the other: {m}");
}

// ==========================================================================
//  S49.3  The fan must be the one the finite-volume geometry uses
// ==========================================================================

/// SPEC-LIT S49.3: the triangulation is the fan about the VERTEX AVERAGE,
/// which is `mesh::geometry::face_geometry`'s own decomposition. On a warped
/// face a different fan would make the radiating area disagree with
/// `b_mag_sf`, and that shows up later as a reciprocity residual nobody can
/// explain.
#[test]
fn the_fan_reproduces_the_finite_volume_area_vector() {
    // A deliberately WARPED quadrilateral: no plane contains all four points.
    let verts = [
        Vec3::new(0.0, 0.0, 0.0),
        Vec3::new(1.0, 0.0, 0.1),
        Vec3::new(1.0, 1.0, 0.0),
        Vec3::new(0.0, 1.0, 0.25),
    ];
    let tris = fan(&verts);
    assert_eq!(tris.len(), 4);
    let sf: Vec3 = tris.iter().fold(Vec3::ZERO, |a, t| a + t.n * (t.two_a * 0.5));

    // The same sum `face_geometry` forms: the triangle area vectors about the
    // vertex average.
    let mut x_avg = Vec3::ZERO;
    for &v in &verts {
        x_avg += v;
    }
    x_avg = x_avg / 4.0;
    let mut want = Vec3::ZERO;
    for i in 0..4 {
        let a = verts[i];
        let b = verts[(i + 1) % 4];
        want += (a - x_avg).cross(b - x_avg) * 0.5;
    }
    assert!((sf - want).mag() < 1e-15, "fan Sf = {sf:?}, face_geometry {want:?}");

    // On a WARPED face the triangulated area is strictly larger than |Sf| -
    // this is the gap `SurfaceGeometry::area_defect` reports rather than
    // hides.
    let area: Scalar = tris.iter().map(|t| t.two_a * 0.5).sum();
    assert!(area > sf.mag(), "a warped face's triangulated area must exceed |Sf|");
    assert!((area - sf.mag()) / area > 1e-4, "the fixture is not warped enough to test this");
}

/// SPEC-LIT S49.4: the intersection test is watertight - a ray aimed exactly
/// at the shared edge of two fan triangles hits one of them, never neither.
/// Moller-Trumbore would let it leak through, and the leak would be
/// bitwise-reproducible, which is worse.
#[test]
fn the_intersection_test_does_not_leak_through_a_shared_edge() {
    let x_avg = Vec3::new(0.5, 0.5, 1.0);
    let quad = [
        Vec3::new(0.0, 0.0, 1.0),
        Vec3::new(1.0, 0.0, 1.0),
        Vec3::new(1.0, 1.0, 1.0),
        Vec3::new(0.0, 1.0, 1.0),
    ];
    let tris = fan(&quad);
    // Fire along each fan spoke - exactly at the shared edge of two adjacent
    // sub-triangles - and along the diagonal from a corner to x_avg.
    let mut n_rays = 0usize;
    for k in 0..=40 {
        let s = k as Scalar / 40.0;
        for corner in quad {
            let target = x_avg + (corner - x_avg) * s;
            let org = Vec3::new(target.x, target.y, 0.0);
            let dir = Vec3::new(0.0, 0.0, 2.0);
            let hit = tris.iter().any(|t| {
                tri_hit(org, dir, t.p0, t.p0 + t.e1, t.p0 + t.e1 + t.e2, 1e-9, 1.0)
            });
            assert!(hit, "a ray at the fan spoke ({}, {}) leaked", target.x, target.y);
            n_rays += 1;
        }
    }
    assert!(n_rays > 100);
}

// ==========================================================================
//  S51.1  The dictionary, read from a case directory
// ==========================================================================

fn case_with(body: &str, tag: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!("ofgpu_s2s_{tag}"));
    let c = dir.join("constant");
    std::fs::create_dir_all(&c).expect("mkdir");
    std::fs::write(c.join("radiationProperties"), body).expect("write");
    dir
}

/// The whole path a real case takes: `constant/radiationProperties` on disk,
/// through `RadiationModel::from_name`'s S13.4 gate, to the model that gets
/// constructed. This is S51.2's `radiationModel` pair test at the level a
/// user actually writes - two files differing in one word.
///
/// SPEC-LIT §50.2/§50.3 is the other half of it, and it is why the two
/// files do NOT both build something. A surface-to-surface enclosure is
/// not a third participating model: it has no volumetric equation, no
/// incident-radiation field and no energy-source registration, and it
/// shares not one entry with the two models that do. So the `P1` file is
/// refused BY NAME by this reader rather than quietly handed an enclosure
/// - the S13.4 substitution this project exists to stop.
#[test]
#[allow(clippy::too_many_lines)]
fn the_radiation_model_selector_reaches_the_case_directory() {
    use crate::radiation::{RadiationConfig, RadiationModel};

    let common = "absorptionCoefficient 0.1;\n";
    let p1 = case_with(&format!("radiationModel P1;\n{common}"), "p1");
    let vf = case_with(
        "radiationModel viewFactor;\nemissivity 0.83;\nagglomerate 3;\n",
        "vf",
    );

    // The participating model is recognised and is not resolved here.
    let e = RadiationConfig::from_case(&p1).expect_err("P1 is not this reader's model");
    let m = e.to_string();
    assert!(m.contains("P1"), "the refusal must name what was asked for: {m}");
    assert!(m.contains("PARTICIPATING"), "{m}");

    let b = RadiationConfig::from_case(&vf).expect("viewFactor");
    assert_eq!(b.model(), RadiationModel::ViewFactor);
    match b {
        RadiationConfig::S2s(c) => {
            assert_eq!(c.emissivity, 0.83);
            assert_eq!(c.agglomerate, 3);
        }
    }

    // And the same file read through the module's own entry point.
    let c = config_from_case(&vf).expect("config_from_case");
    assert_eq!(c.emissivity, 0.83);

    // A viewFactor case that ALSO sets an absorption coefficient is refused:
    // there is no medium for it to describe (S50.9).
    let both = case_with(
        "radiationModel viewFactor;\nemissivity 0.8;\nabsorptionCoefficient 0.2;\n",
        "both",
    );
    let e = RadiationConfig::from_case(&both).expect_err("must be refused");
    assert!(e.to_string().contains("NON-PARTICIPATING"), "{e}");

    // A viewFactor case with NO emissivity is refused - it is the one entry
    // with no honest default.
    let bare = case_with("radiationModel viewFactor;\n", "bare");
    let e = RadiationConfig::from_case(&bare).expect_err("must be refused");
    assert!(e.to_string().contains("emissivity"), "{e}");

    for d in [p1, vf, both, bare] {
        let _ = std::fs::remove_dir_all(d);
    }
}

// ==========================================================================
//  A curved radiating surface, for the one thing a box cannot test
// ==========================================================================

/// The side wall of a regular `n`-gonal prism, as `n` quads in ONE patch,
/// sharing vertical edges, adjacent normals `360/n` degrees apart.
///
/// A box's patch is flat, so every face of it agrees with every other to zero
/// degrees and `maxClusterAngle` can never bite. This is the smallest fixture
/// on which it can, and it is also what a curved radiating surface - a duct,
/// a cylindrical shield - actually looks like.
fn prism_strip(n: usize) -> SurfaceGeometry {
    let mut g = SurfaceGeometry { vtx_offset: vec![0], ..Default::default() };
    let tau = std::f64::consts::TAU as Scalar;
    for k in 0..n {
        let (a0, a1) = (tau * k as Scalar / n as Scalar, tau * (k + 1) as Scalar / n as Scalar);
        // Wound so the fan normal points OUTWARD from the axis.
        let verts = [
            Vec3::new(a0.cos(), a0.sin(), 0.0),
            Vec3::new(a1.cos(), a1.sin(), 0.0),
            Vec3::new(a1.cos(), a1.sin(), 1.0),
            Vec3::new(a0.cos(), a0.sin(), 1.0),
        ];
        let tris = fan(&verts);
        let mut sf = Vec3::ZERO;
        let mut area: Scalar = 0.0;
        for t in &tris {
            sf += t.n * (t.two_a * 0.5);
            area += t.two_a * 0.5;
        }
        for v in verts {
            g.vtx.push(v);
        }
        g.vtx_offset.push(g.vtx.len() as Label);
        g.b_face.push(k as Label);
        g.centroid.push(verts.iter().fold(Vec3::ZERO, |a, &v| a + v) / 4.0);
        g.normal.push(sf / sf.mag());
        g.area.push(area);
        g.emissivity.push(0.8);
        g.q_ext.push(0.0);
    }
    g.n = n;
    g
}

/// The minimum `HostMesh` [`Clustering::agglomerate`] reads: one patch, `n`
/// boundary faces.
fn one_patch_mesh(n: usize) -> crate::mesh::HostMesh {
    crate::mesh::HostMesh {
        n_boundary_faces: n,
        b_patch: vec![0; n],
        ..Default::default()
    }
}

/// The fixture itself is a claim - adjacent faces 30 degrees apart - and a
/// claim in a test fixture is worth checking.
#[test]
fn the_prism_strip_really_bends_thirty_degrees_per_face() {
    let g = prism_strip(12);
    assert_eq!(g.n, 12);
    let want = (std::f64::consts::PI as Scalar / 6.0).cos(); // cos 30
    for k in 0..12 {
        let d = g.normal[k].dot(g.normal[(k + 1) % 12]);
        assert!((d - want).abs() < 1e-12, "face {k}/{}: cos = {d}", (k + 1) % 12);
        // Outward from the axis.
        assert!(g.normal[k].dot(g.centroid[k] - Vec3::new(0.0, 0.0, 0.5)) > 0.0);
    }
    // And a merge at 20 degrees really is impossible while one at 89 is not.
    let hm = one_patch_mesh(12);
    assert_eq!(Clustering::agglomerate(&g, &hm, 64, 20.0).n_coarse, 12);
    assert!(Clustering::agglomerate(&g, &hm, 64, 89.0).n_coarse < 12);
    // agglomerate 1 is the identity whatever the angle.
    assert_eq!(Clustering::agglomerate(&g, &hm, 1, 89.0).n_coarse, 12);
}

// ==========================================================================
//  S50.11 Gate 50-D, and the two S49.7 rows the gates above only imply
// ==========================================================================

/// **Gate 50-D.** Radiative equilibrium through the ACTUAL kernels rather
/// than through the formula: a closed black box, every wall at `T_inf`.
///
/// The whole chain runs - the coarse gather of `sigma T^4`, the radiosity
/// solve, the broadcast, the stamp - and the triple that comes out must have
/// `refValue = T_inf` and `refGrad = 0`, because at equilibrium the wall
/// neither gains nor loses. It is the one check that catches a sign error or
/// a mis-plumbed buffer anywhere between `t.bf` and `t.ref_value`.
#[test]
fn gate_50d_radiative_equilibrium_through_the_kernels() {
    let Some(gpu) = gpu() else { return };
    let t_inf = 640.0 as Scalar;
    let (hm, gm, points, faces, sel) = box_rig(&gpu, [3, 3, 3], 1.0);
    let mut s2s =
        S2s::new(&gpu, &gm, &hm, &points, &faces, &sel, S2sConfig::default()).expect("s2s");

    let nbf = hm.n_boundary_faces;
    let mut t = GpuScalarField::zeros(&gpu, &gm, "T").expect("T");
    gpu.write(&mut t.bf, &vec![t_inf; nbf]).expect("bf");
    let k_eff: DevBuf<Scalar> = gpu.upload(&vec![0.03 as Scalar; nbf]).expect("k");
    s2s.update(&gpu, &mut t, &k_eff).expect("update");

    // The irradiation on a black isothermal enclosure is sigma T^4.
    let want_h = SIGMA_SB * t_inf.powi(4);
    let hf = s2s.irradiation_fine(&gpu).expect("hf");
    let worst_h = hf[..s2s.n_fine()]
        .iter()
        .map(|h| (h - want_h).abs() / want_h)
        .fold(0.0 as Scalar, Scalar::max);
    assert!(worst_h < 1e-9, "H = sigma T^4 only to {worst_h:.3e}");

    // ...so the triple's reference value is T_inf itself.
    let rv = gpu.download(&t.ref_value).expect("rv");
    let rg = gpu.download(&t.ref_grad).expect("rg");
    let mut worst_t: Scalar = 0.0;
    for s in 0..s2s.n_fine() {
        let bf = s2s.b_face_of(s) as usize;
        worst_t = worst_t.max((rv[bf] - t_inf).abs() / t_inf);
        assert_eq!(rg[bf], 0.0, "an equilibrium wall needs no gradient");
    }
    assert!(worst_t < 1e-9, "refValue = T_inf only to {worst_t:.3e}");

    // And nothing is exchanged: the net flux vanishes on every surface.
    let (_, _, q) = s2s.state(&gpu).expect("state");
    let scale = SIGMA_SB * t_inf.powi(4);
    let worst_q = q.iter().map(|v| v.abs() / scale).fold(0.0 as Scalar, Scalar::max);
    assert!(worst_q < 1e-9, "q_r = {worst_q:.3e} of sigma T^4 at equilibrium");

    // The power balance has to be scaled by `sigma T^4 * A`, NOT by the GROSS
    // exchanged power the other tests use: at equilibrium the gross power is
    // itself zero, so a relative test against it compares two numbers that
    // are both round-off and demands one be a billionth of the other. The
    // first draft did exactly that and failed on a correct answer.
    let a_total: Scalar = s2s.view_factors().areas().iter().sum();
    assert!(
        s2s.report().net_power.abs() < 1e-9 * scale * a_total,
        "net power {} against the {} W an isothermal enclosure radiates",
        s2s.report().net_power,
        scale * a_total
    );
    assert!(
        s2s.report().gross_power < 1e-9 * scale * a_total,
        "gross power {} - at equilibrium NOTHING is exchanged, not just nothing net",
        s2s.report().gross_power
    );
}

/// SPEC-LIT S49.7's two remaining rows, on the cleanest fixture for them.
///
/// **The `s` bucket is symmetric.** `s_ij` is built from `|C_i - C_j|` and
/// `R_i + R_j`, both symmetric, so `nq(i,j) == nq(j,i)` and the two halves of
/// a pair are integrated at the same order. That is not directly observable
/// from outside, but its consequence is: the reciprocity defect BEFORE
/// symmetrisation would be first-order in the quadrature error if the orders
/// disagreed, and it is instead at round-off.
///
/// **`F_ii = 0` exactly.** The diagonal is not computed, not stored and
/// zeroed - integrating it would produce a `1/r^4` singularity for a quantity
/// that is zero for any planar element.
#[test]
fn the_diagonal_is_exactly_zero_and_the_raw_quadrature_is_already_reciprocal() {
    let Some(gpu) = gpu() else { return };
    let g = CoarseGeometry::from_polygons(&cube_of_squares(4, Vec3::ZERO, true));
    let vf = ViewFactors::build(&gpu, &g, &S2sConfig::default()).expect("vf");
    let n = vf.n_surf;

    assert!(
        vf.report().reciprocity_error < 1e-14,
        "the RAW quadrature's reciprocity defect is {:.3e}; if the two halves \
         of a pair were integrated at different orders it would be much larger",
        vf.report().reciprocity_error
    );

    let f = vf.view_factors(&gpu).expect("F");
    for i in 0..n {
        assert_eq!(f[i * n + i], 0.0, "F[{i}][{i}] is not exactly zero");
    }
}

// ----------------------------------------------------------------------
//  SPEC-LIT 81.7: the CUDA-graph capture gate
// ----------------------------------------------------------------------

/// SPEC-LIT 81: one surface-to-surface exchange captures and replays
/// bitwise.
///
/// S2S is the interesting case: the view factors are built ONCE, outside any
/// iteration, and what runs per iteration is the radiosity solve over the
/// coarse clusters plus the scatter back to the fine faces. That is all
/// device work, and this proves it.
#[test]
fn the_s2s_exchange_replays_bitwise() {
    let Some(gpu) = gpu() else { return };
    let (hm, gm, points, faces, sel) = box_rig(&gpu, [2, 2, 2], 0.9);
    let nbf = hm.n_boundary_faces;
    let cfg = S2sConfig { agglomerate: 4, ..S2sConfig::default() };
    let k_eff: DevBuf<Scalar> = gpu.upload(&vec![0.03 as Scalar; nbf]).expect("k");
    let tb: Vec<Scalar> = (0..nbf)
        .map(|bf| if bf % 2 == 0 { 300.0 } else { 900.0 })
        .collect();

    let report = crate::capture::capture_replays_bitwise(
        &gpu,
        "surface-to-surface exchange (SPEC-LIT 49/50)",
        || {
            let mut s = S2s::new(&gpu, &gm, &hm, &points, &faces, &sel, cfg)?;
            // The only host round-trip in the exchange is SPEC-LIT 50.10's
            // report, computed by downloading four cluster arrays and
            // reducing them on the CPU - see S2s::set_measure_report.
            s.set_measure_report(false);
            let mut t = GpuScalarField::zeros(&gpu, &gm, "T")?;
            gpu.write(&mut t.bf, &tb)?;
            Ok((s, t))
        },
        |(s, t): &mut (S2s, GpuScalarField)| s.update(&gpu, t, &k_eff),
        |(s, t): &(S2s, GpuScalarField)| {
            let (a, b, c) = s.state(&gpu)?;
            Ok(vec![
                ("T boundary", gpu.download(&t.bf)?),
                ("radiosity", a),
                ("irradiation", b),
                ("net flux", c),
            ])
        },
    )
    .expect("SPEC-LIT 81.7: the S2S exchange must capture and replay bitwise");
    println!("  S2S: {report}");
}
