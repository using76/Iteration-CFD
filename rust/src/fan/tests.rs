// meteor-cfd - Copyright (c) 2026 주식회사 메테오시뮬레이션 (Meteo Simulation Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.
//
// Provenance: ORIGINAL - the tests for SPEC-LIT S52 and S53. Every expected
// number here is either a closed form derived in those sections, an identity
// this code checks against itself, or a published reference value read out of
// a PUBLIC-DOMAIN input deck (`reference/fds/Verification/HVAC/*.fds` and
// their `.csv`, NIST, US Government public domain). No FDS SOURCE is read
// here - only its case files and their published results, which are data.
// No GPL-licensed source was consulted.

use super::*;

use crate::field::{BcKind, GpuScalarField, GpuSurfaceScalarField};
use crate::fv::{self, FvKernels};
use crate::io::case::{LinearSolverKind, Preconditioner, SolverControls};
use crate::ldu::GpuLduMatrix;
use crate::ldu_ops::{self, LduKernels};
use crate::mesh::topology::tests::box_mesh;
use crate::solver::SolverWorkspace;
use crate::Vec3;

fn gpu() -> Option<Gpu> {
    Gpu::new(0).ok()
}

/// A hexahedral block of `n` cells of size `d`.
fn block(n: [usize; 3], d: Vec3) -> HostMesh {
    let (mut m, points, faces) = box_mesh(n, d);
    m.compute_geometry(&points, &faces).expect("geometry");
    m.build_cell_face_maps();
    m
}

/// `(start, size)` of a named patch, copied out so nothing holds a borrow of
/// the rig while the rig is being written to.
fn span(hm: &HostMesh, name: &str) -> (usize, usize) {
    let p = hm.patches.iter().find(|p| p.name == name).expect("patch");
    (p.start, p.size)
}

fn rel(a: Scalar, b: Scalar) -> Scalar {
    let s = a.abs().max(b.abs()).max(1e-300);
    (a - b).abs() / s
}

// ==========================================================================
//  The device contract
// ==========================================================================

/// The curve discriminants and the table bound are shared with
/// `cuda/fan.cu`, which switches on them. A drift here would silently route
/// a quadratic curve through the constant branch.
#[test]
fn curve_kind_values_match_the_device() {
    assert_eq!(CurveKind::Constant as i32, 0);
    assert_eq!(CurveKind::Quadratic as i32, 1);
    assert_eq!(CurveKind::Table as i32, 2);

    let src = std::fs::read_to_string(
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("cuda/fan.cu"),
    )
    .expect("cuda/fan.cu");
    for (macro_name, value) in [
        ("OFGPU_FAN_CONSTANT", 0),
        ("OFGPU_FAN_QUADRATIC", 1),
        ("OFGPU_FAN_TABLE", 2),
        ("OFGPU_FAN_MAX_POINTS", MAX_CURVE_POINTS as i32),
    ] {
        let want = format!("#define {macro_name}");
        let line = src
            .lines()
            .find(|l| l.trim_start().starts_with(&want))
            .unwrap_or_else(|| panic!("cuda/fan.cu does not define {macro_name}"));
        let got: i32 = line
            .split_whitespace()
            .next_back()
            .and_then(|t| t.parse().ok())
            .unwrap_or_else(|| panic!("cannot read a number out of \"{line}\""));
        assert_eq!(got, value, "{macro_name} disagrees with the Rust side");
    }
}

/// SPEC-LIT §52.7: no f64 atomic anywhere in this section's kernels. The
/// determinism claim is exactly this, and a grep is how it stays true.
#[test]
fn the_fan_kernels_contain_no_atomic() {
    let src = std::fs::read_to_string(
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("cuda/fan.cu"),
    )
    .expect("cuda/fan.cu");
    assert!(
        !src.contains("atomic"),
        "cuda/fan.cu mentions an atomic. SPEC-LIT S52.7: every sum here is a \
         gather plus solver::device_sum, whose partition is a pure function of n \
         and is therefore order-independent. An atomic would make the fan \
         operating point depend on the scheduler."
    );
}

// ==========================================================================
//  §52.12 Gate 52-A - the closed-form operating point
// ==========================================================================

/// (S52.15), evaluated rather than quoted, at FDS's own `FAN2` parameters.
///
/// The gate is not the constant: it is that the fan pressure and the system
/// pressure are the SAME number there, which is what "operating point"
/// means. A transcription error in (S52.15) breaks that equality.
#[test]
fn gate_52a_the_quadratic_operating_point_is_where_the_two_curves_cross() {
    let (dp_max, q_max, k_sys) = (3048.0 as Scalar, 2.4094 as Scalar, 400.0 as Scalar);
    let q = quadratic_operating_point(dp_max, q_max, k_sys);

    let dp_fan = dp_max * (1.0 - (q / q_max) * (q / q_max));
    let dp_sys = k_sys * q * q;
    assert!(
        rel(dp_fan, dp_sys) < 1e-14,
        "the closed form is not the crossing point: dp_fan = {dp_fan}, dp_sys = {dp_sys}"
    );

    // And the number the design note quotes, to all its digits - a second,
    // independent check that (S52.15) was transcribed correctly.
    assert!(rel(q, 1.8152058157833744) < 1e-14, "Q* = {q}");
    assert!(rel(dp_fan, 1317.9888614615143) < 1e-13, "dp = {dp_fan}");

    // The curve object must agree with the free function.
    let c = FanCurve::quadratic(dp_max, q_max);
    let (dp, s) = c.at(q);
    assert!(rel(dp, dp_fan) < 1e-14);
    // S = -dp' = 2 dp_max Q/Q_max^2 > 0 on the falling branch.
    assert!(rel(s, 2.0 * dp_max * q / (q_max * q_max)) < 1e-14, "S = {s}");
}

// ==========================================================================
//  §52.12 Gate 52-B - the FDS public-domain cross-check
// ==========================================================================

/// `reference/fds/Verification/HVAC/fan_test.fds` + `fan_test.csv`.
///
/// The fan duct there carries `LOSS=0,0`, so the fan's rise must equal the
/// compartment pressure difference exactly. Evaluating the quadratic curve at
/// FDS's own reported flow rate must reproduce FDS's own reported pressure.
///
/// Constants are read out of the vendored case deck; **no FDS source is
/// read**, only its input file and its published CSV, which are data.
#[test]
fn gate_52b_the_fds_fan_test_operating_point_is_reproduced() {
    // &HVAC ID='LEFT',TYPE_ID='FAN',MAX_FLOW=0.16,MAX_PRESSURE=10.
    let curve = FanCurve::quadratic(10.0, 0.16);
    // fan_test.csv: vflow1 = -0.0498253, pres_1 = 4.51513, pres_2 = -4.51513
    let q_fds = 0.0498253 as Scalar;
    let dp_fds = 2.0 * 4.51513 as Scalar;

    let (dp, _) = curve.at(q_fds);
    let err = rel(dp, dp_fds);
    assert!(
        err < 1e-5,
        "FDS fan_test: the curve at its own Q gives {dp} Pa, FDS reports {dp_fds} Pa \
         (relative {err:e})"
    );
}

/// `qfan_test.fds` + `qfan_test.csv`: the loss-only duct.
///
/// `dp = (1/2) rho K (Q/A)^2` with `K = 5` (the `LOSS=5,5` entry, forward
/// direction), `A = 0.04 m^2` and FDS's default air at 20 C. `rho` is
/// computed here from `p M/(R T)` rather than quoted, so a wrong molar mass
/// fails rather than agrees.
#[test]
fn gate_52b_the_fds_qfan_loss_duct_is_reproduced() {
    let rho = 101325.0 * 28.85034e-3 / (8.3145 * 293.15) as Scalar;
    assert!(rel(rho, 1.199338) < 1e-5, "FDS's air density comes out {rho}");

    let (q, a, k) = (0.04911 as Scalar, 0.04 as Scalar, 5.0 as Scalar);
    let dp = 0.5 * rho * k * (q / a) * (q / a);
    let dp_fds = 2.0 * 2.2592 as Scalar;
    let err = rel(dp, dp_fds);
    assert!(
        err < 3e-4,
        "FDS qfan_test loss duct: {dp} Pa against FDS's {dp_fds} Pa (relative {err:e})"
    );

    // And the same relation expressed through this module's own jump
    // coefficients, which is what makes it a check of THIS code rather than
    // of arithmetic: R phi in kinematic units, times rho, is dp in Pa.
    let c = PorousJumpCoeffs::from_loss_coefficient(k).expect("K");
    let r = c.resistance(q, a);
    assert!(
        rel(rho * r * q, dp) < 1e-12,
        "the jump coefficients do not reproduce (1/2) rho K u^2"
    );
}

// ==========================================================================
//  §52.12 Gate 52-D - the rank-1 identity, on the host
// ==========================================================================

/// (S52.8): the exact operator is symmetric, and it is a *bounded* downdate.
#[test]
fn gate_52d_the_exact_rank1_operator_is_symmetric() {
    let d = [0.3 as Scalar, 1.7, 0.55, 2.2, 0.9];
    for s in [0.0 as Scalar, 0.7, 12.0, 1e6] {
        let a = exact_rank1(&d, s);
        for i in 0..d.len() {
            for j in 0..d.len() {
                assert_eq!(
                    a[i][j], a[j][i],
                    "S52.2 consequence 1: A[{i}][{j}] and A[{j}][{i}] are the same \
                     product of the same two numbers and must be EQUAL, not close"
                );
            }
        }
    }

    // And the association is what buys that. `(kappa*d_i)*d_j` rounds twice
    // in a different order for each half and lands one ulp apart - measured,
    // not asserted, because this is the correction S52.2 had to be given.
    let sd: Scalar = d.iter().sum();
    let kappa = 0.7 / (1.0 + 0.7 * sd);
    let naive_02 = kappa * d[0] * d[2];
    let naive_20 = kappa * d[2] * d[0];
    assert_ne!(
        naive_02, naive_20,
        "if the naive association has become symmetric too, SPEC-LIT S52.2's \
         association note is no longer describing anything and should be removed"
    );
    assert!(
        rel(naive_02, naive_20) < 1e-15,
        "and it is one ulp, not a real disagreement: {naive_02} vs {naive_20}"
    );
    assert_eq!(kappa * (d[0] * d[2]), kappa * (d[2] * d[0]));
}

/// (S52.9): the row sum is `D_f/(1 + S SIGMA_D)`.
///
/// **The design note states this as `SIGMA_D/(1 + S SIGMA_D)` and that is
/// wrong** - at `S = 0` a `fixedValue` face contributes `D_f` to its own row,
/// not the patch total. This test is what the note's prose would fail.
#[test]
fn gate_52d_the_row_sum_is_d_f_over_one_plus_s_sigma_d() {
    let d = [0.3 as Scalar, 1.7, 0.55, 2.2, 0.9];
    let sd: Scalar = d.iter().sum();

    for s in [0.0 as Scalar, 0.7, 12.0] {
        let a = exact_rank1(&d, s);
        for (i, di) in d.iter().enumerate() {
            let row: Scalar = a[i].iter().sum();
            let want = di / (1.0 + s * sd);
            assert!(
                rel(row, want) < 1e-13,
                "row {i} at S = {s} sums to {row}, (S52.9) says {want}"
            );
            // The note's form would be `sd/(1 + s*sd)` - explicitly NOT this,
            // except in the degenerate case where every D is the patch total.
            if s > 0.0 {
                let note_form = sd / (1.0 + s * sd);
                assert!(
                    rel(row, note_form) > 1e-3,
                    "the note's SIGMA_D/(1+S SIGMA_D) happens to agree at row {i}; \
                     pick a D distribution where it does not, or the test proves \
                     nothing"
                );
            }
        }
    }

    // The two limits of (S52.9).
    let a0 = exact_rank1(&d, 0.0);
    for (i, di) in d.iter().enumerate() {
        let row: Scalar = a0[i].iter().sum();
        assert_eq!(row, *di, "at S = 0 the operator IS diag(D)");
    }
    let ainf = exact_rank1(&d, 1e12);
    for row in &ainf {
        let s: Scalar = row.iter().sum();
        assert!(s.abs() < 1e-11, "as S -> infinity the row sum -> 0 (pure Neumann)");
    }
}

/// (S52.10): the lumped `fr` reproduces (S52.9)'s row sum for **any** `D`,
/// and the design note's per-face form does not.
#[test]
fn gate_52d_the_lumped_fr_preserves_the_row_sum_and_the_notes_does_not() {
    let d = [0.3 as Scalar, 1.7, 0.55, 2.2, 0.9];
    let sd: Scalar = d.iter().sum();
    let s = 0.7 as Scalar;
    let a = exact_rank1(&d, s);

    let fr = 1.0 / (1.0 + s * sd);
    let mut worst_note = 0.0 as Scalar;
    for (i, di) in d.iter().enumerate() {
        let row: Scalar = a[i].iter().sum();
        assert!(
            rel(fr * di, row) < 1e-13,
            "(S52.10) row {i}: lumped {} vs exact {row}",
            fr * di
        );

        // The note's beta_f = S A rAU_f Delta_f. With equal face areas
        // a_f = A/N, that is S N D_f.
        let beta_note = s * d.len() as Scalar * di;
        let contrib_note = di / (1.0 + beta_note);
        // Normalised by the EXACT row sum - which is what "142 % high on that
        // row" in SPEC-LIT S52.3 means. `rel` divides by the larger of the
        // two and would report 0.59 for the same disagreement.
        worst_note = worst_note.max((contrib_note - row).abs() / row);
    }
    assert!(
        worst_note > 1.0,
        "the note's form is supposed to be badly wrong on a non-uniform patch; it \
         came out within {worst_note} of the exact row sum, so this test is not \
         measuring what SPEC-LIT S52.3 claims"
    );
    // S52.3 quotes 142 % on the worst row of exactly this D distribution.
    assert!(
        (worst_note - 1.417).abs() < 0.01,
        "S52.3 records the worst row as 142 % high; measured {:.1} %",
        100.0 * worst_note
    );

    // On a UNIFORM patch the two agree bitwise, which is what the note's own
    // numerical check found and what makes its form a legitimate
    // specialisation rather than an error.
    let du = [0.8 as Scalar; 5];
    let au = exact_rank1(&du, s);
    // The claim is about BETA, and on a uniform patch the two are the same
    // number: the note's `S A rAU_f Delta_f` is `S N D` and (S52.10)'s is
    // `S SIGMA_D`, and `SIGMA_D = N D`. (The row-sum PRODUCTS then differ by
    // one ulp, because `(1/x)*D` and `D/x` are different expression trees -
    // an artefact of how the check is written, not of either formula.)
    let beta_note = s * 5.0 * 0.8;
    let beta_ours = s * du.iter().sum::<Scalar>();
    assert!(
        rel(beta_note, beta_ours) < 1e-15,
        "on a uniform patch the note's beta and (S52.10)'s are the same number:          {beta_note} vs {beta_ours}"
    );
    // They are the same number and NOT the same bits: `S SIGMA_D` sums N
    // terms where `S A rAU Delta` multiplies, and the two round differently
    // (2.8000000000000003 against 2.8 here). "Agree on a uniform patch" is
    // therefore a round-off statement, not a bitwise one - which matters,
    // because everywhere else in S52 the bitwise claims are exact.
    assert_ne!(beta_note, beta_ours);
    let row: Scalar = au[0].iter().sum();
    assert!(rel(row, 0.8 / (1.0 + beta_ours)) < 1e-15);
    assert!(rel(row, 0.21052631578947367) < 1e-15, "the note's own check: {row}");
}

/// (S52.12): the lumped triple and the exact operator impose **the same**
/// relation between the patch flow rate and the `D`-weighted mean cell
/// pressure.
///
/// In exact arithmetic the difference is zero. In f64 the two are different
/// expression trees, so what is measured is the round-off - and that is what
/// this test reports, not "identically zero".
#[test]
fn gate_52d_the_lumped_and_exact_operators_impose_the_same_flow_rate() {
    let d = [0.3 as Scalar, 1.7, 0.55, 2.2, 0.9];
    let p_p = [1.0 as Scalar, -2.0, 0.5, 3.0, 0.25];
    let (c, phi) = (-3.0 as Scalar, 1.25 as Scalar);
    let sd: Scalar = d.iter().sum();

    let mut worst = 0.0 as Scalar;
    for s in [0.0 as Scalar, 0.31, 0.7, 5.0, 100.0] {
        let dp: Scalar = d.iter().zip(&p_p).map(|(a, b)| a * b).sum();

        // Exact (S52.7): one patch pressure pi, folded back.
        let pi = (c + s * phi + s * dp) / (1.0 + s * sd);
        let q_exact: Scalar =
            phi - d.iter().zip(&p_p).map(|(dg, pg)| dg * (pi - pg)).sum::<Scalar>();

        // Lumped (S52.11): a per-face Robin value.
        let fr = 1.0 / (1.0 + s * sd);
        let rv = c + s * phi;
        let q_lumped: Scalar = phi
            - d.iter()
                .zip(&p_p)
                .map(|(dg, pg)| dg * (fr * rv + (1.0 - fr) * pg - pg))
                .sum::<Scalar>();

        worst = worst.max(rel(q_exact, q_lumped));

        // And the linearised fan relation itself holds at the solution.
        assert!(
            rel(pi, c + s * q_exact) < 1e-12,
            "(S52.5) pi = c + S Q is not satisfied at S = {s}"
        );
    }
    assert!(
        worst < 1e-13,
        "(S52.12) should hold to round-off; the worst relative gap was {worst:e}"
    );
}

// ==========================================================================
//  §52.4 - the two endpoints
// ==========================================================================

/// `S = 0` gives `(1.0, c, 0.0)` **bitwise** - the `fixedValue` triple.
#[test]
fn a_flat_curve_gives_the_fixed_value_triple_bitwise() {
    let c = FanCurve::flat(37.5);
    for &(sigma_d, phi, q_star) in
        &[(0.0 as Scalar, 0.0 as Scalar, 0.0 as Scalar), (3.7, 12.25, -4.5), (1e9, -8.0, 1e3)]
    {
        for dir in [FanDirection::Outflow, FanDirection::Inflow] {
            let (fr, rv, s) =
                lumped_triple(&c, dir, 11.0, 1.2, q_star, phi, sigma_d);
            assert_eq!(s, 0.0, "a flat curve has S = 0 exactly");
            assert_eq!(fr, 1.0, "fr must be exactly 1.0, not 1.0 - eps");
            // refValue is exactly p_a - sigma*dp/rho, with nothing added.
            let want = 11.0 - dir.sigma() * 37.5 / 1.2;
            assert_eq!(rv, want, "refValue must be exactly c: {rv} vs {want}");
        }
    }
}

/// `S -> infinity` delivers the prescribed flow through an `fr -> 0` face.
#[test]
fn a_vertical_curve_delivers_the_prescribed_flow() {
    // A quadratic curve made almost vertical - but evaluated at its own free
    // delivery, so `F*` (and hence `pi*`) stays BOUNDED while `S` runs away.
    //
    // That premise is not decoration. (S52.4) reads
    // `fr(c + S Phi) = (pi* - S Q* + S Phi)/(1 + S SIGMA_D)`, and the limit
    // `(Phi - Q*)/SIGMA_D` needs `pi*/(S SIGMA_D) -> 0`. A curve whose value
    // at `Q*` grows with `S` - a quadratic with a tiny `Q_max` evaluated well
    // past it - leaves a residual `pi*/(S SIGMA_D)` behind, and a first draft
    // of this test measured exactly that: 0.258 against the limit's 0.236,
    // the difference being `1.008e15/4.583e16 = 0.022`.
    let q_star = 0.11 as Scalar;
    let c = FanCurve::quadratic(1.0e9, q_star);
    let (sigma_d, phi) = (2.5 as Scalar, 0.7 as Scalar);
    let (fr, rv, s) = lumped_triple(&c, FanDirection::Outflow, 0.0, 1.2, q_star, phi, sigma_d);

    assert!(fr > 0.0 && fr < 1e-9, "fr should be nearly zero, got {fr}");
    // (S52.4): fr*(c + S Phi) -> (Phi - Q*)/SIGMA_D, so the patch delivers Q*.
    let want = (phi - q_star) / sigma_d;
    assert!(
        rel(fr * rv, want) < 1e-6,
        "the S -> infinity limit gives {} where (S52.4) says {want} (S = {s})",
        fr * rv
    );
}

// ==========================================================================
//  §52.5 - the curve, its corrections, and its refusals
// ==========================================================================

/// The Fritsch-Carlson limiter: a monotone table stays monotone, where a
/// plain Catmull-Rom spline through the same points does not.
#[test]
fn the_hermite_curve_is_monotone_where_a_plain_spline_is_not() {
    // Four points with a near-flat stretch followed by a steep drop - the
    // classic overshoot configuration.
    let pts = vec![
        (0.0 as Scalar, 1000.0 as Scalar),
        (1.0, 999.0),
        (2.0, 995.0),
        (3.0, 300.0),
    ];
    let c = FanCurve::table(pts.clone());
    c.validate("t").expect("a falling table is legal");

    let mut worst_neg = 0.0 as Scalar;
    let mut prev = c.at(0.0).0;
    for i in 1..=3000 {
        let q = 3.0 * i as Scalar / 3000.0;
        let (dp, s) = c.at(q);
        assert!(dp <= prev + 1e-9, "the curve rose from {prev} to {dp} at Q = {q}");
        prev = dp;
        worst_neg = worst_neg.min(s);
    }
    assert!(
        worst_neg >= -1e-9,
        "S went negative ({worst_neg}) inside a monotone table - the Fritsch-Carlson \
         limiter is not doing its job, and a negative S is a stall branch the case \
         did not ask for (SPEC-LIT S52.5)"
    );

    // The unlimited three-point slope at the interior node adjacent to the
    // cliff is what a plain spline would use; show it overshoots.
    let d = [(999.0 - 1000.0) / 1.0, (995.0 - 999.0) / 1.0, (300.0 - 995.0) / 1.0];
    let m_unlimited = 0.5 * (d[1] + d[2]);
    let m_limited = c.hermite_slopes()[2];
    assert!(
        m_limited > m_unlimited,
        "the limiter must raise (flatten) the node slope at the cliff: {m_limited} \
         against the unlimited {m_unlimited}"
    );

    // The curve passes through its own data points.
    for (q, dp) in &pts {
        assert!(rel(c.at(*q).0, *dp) < 1e-12, "the interpolant misses ({q}, {dp})");
    }
}

/// (S52.13): the affinity laws, checked against their own statement.
#[test]
fn the_density_and_speed_corrections_are_the_affinity_laws() {
    let base = FanCurve::quadratic(500.0, 2.0);
    let q = 1.0 as Scalar;
    let (dp0, s0) = base.at(q);

    // Density: dp scales linearly, and so does the slope.
    let mut d = base.clone();
    d.rho = 0.9;
    d.rho_curve = 1.2;
    let (dp1, s1) = d.at(q);
    assert!(rel(dp1, dp0 * 0.75) < 1e-13, "dp did not scale by rho/rho_curve");
    assert!(rel(s1, s0 * 0.75) < 1e-13, "S did not scale by rho/rho_curve");

    // Speed: dp(Q; N) = dp_curve(Q N_c/N) (N/N_c)^2. At Q scaled with N the
    // pressure scales as N^2 exactly - the affinity law's own statement.
    let mut n = base.clone();
    n.n_speed = 1.5;
    n.n_curve = 1.0;
    let (dp2, _) = n.at(q * 1.5);
    assert!(
        rel(dp2, dp0 * 2.25) < 1e-13,
        "dp(1.5 Q; 1.5 N) should be 2.25 dp(Q; N); got {dp2} against {}",
        dp0 * 2.25
    );
    // And free delivery moves with the speed: dp = 0 at Q = N/N_c * Q_max.
    assert!(n.at(1.5 * 2.0).0.abs() < 1e-9, "free delivery did not move with N");
}

/// The curve's tails: held below `Q_min`, bounded above free delivery, and
/// `S > 0` on the way out so a reverse-flow excursion cannot run away.
#[test]
fn the_curve_tails_are_bounded_and_have_a_positive_slope() {
    let c = FanCurve::table(vec![(0.5, 900.0), (1.0, 700.0), (2.0, 100.0)]);
    c.validate("t").expect("legal");

    // Below Q_min the curve RISES and S stays positive - the tail opposes an
    // excursion instead of holding a value (SPEC-LIT S52.5). An earlier draft
    // held `dp(Q_min)` here, which is `S = 0`, i.e. a `fixedValue` at the
    // shut-off pressure - the stiffest condition the curve has, applied
    // exactly where the iterate is furthest from the operating point.
    let (dp, s) = c.at(-3.0);
    assert!(dp > 900.0, "below the first point dp must RISE: {dp}");
    assert!(s > 0.0, "and S must stay positive: {s}");

    // Above the last point: falling, and falling faster.
    let (dp_a, s_a) = c.at(2.5);
    let (dp_b, s_b) = c.at(6.0);
    assert!(dp_b < dp_a && dp_a < 100.0, "the tail must keep falling");
    assert!(s_b > s_a && s_a > 0.0, "S must GROW in the tail: {s_a} then {s_b}");
    // The slope is continuous at the join.
    let (_, s_at) = c.at(2.0 + 1e-9);
    let (_, s_in) = c.at(2.0 - 1e-9);
    assert!(rel(s_at, s_in) < 1e-4, "the slope jumps at Q_max: {s_in} then {s_at}");
}

#[test]
fn a_rising_curve_is_refused_by_name() {
    let c = FanCurve::table(vec![(0.0, 500.0), (1.0, 520.0), (2.0, 100.0)]);
    let e = c.validate("crac1").expect_err("a rising branch must be refused");
    let m = e.to_string();
    assert!(m.contains("crac1"), "the message must name the fan: {m}");
    assert!(m.contains("STALL"), "the message must say why: {m}");
    assert!(m.contains("quadratic"), "the message must name the alternatives: {m}");
}

#[test]
fn a_non_increasing_flow_axis_is_refused_by_name() {
    for pts in [
        vec![(1.0, 500.0), (1.0, 400.0)],
        vec![(0.0, 500.0), (2.0, 400.0), (1.0, 300.0)],
    ] {
        let e = FanCurve::table(pts)
            .validate("crac1")
            .expect_err("Q must be strictly increasing");
        assert!(e.to_string().contains("STRICTLY increasing"), "{e}");
    }
}

#[test]
fn a_short_or_oversized_table_is_refused_by_name() {
    let e = FanCurve::table(vec![(1.0, 500.0)]).validate("f").expect_err("one point");
    assert!(e.to_string().contains("at least two points"), "{e}");

    let big: Vec<(Scalar, Scalar)> =
        (0..MAX_CURVE_POINTS + 1).map(|i| (i as Scalar, -(i as Scalar))).collect();
    let e = FanCurve::table(big).validate("f").expect_err("too many points");
    assert!(e.to_string().contains("fixed-trip-count"), "{e}");
}

#[test]
fn the_curve_corrections_and_efficiency_are_validated_by_name() {
    let mut c = FanCurve::quadratic(500.0, 2.0);
    c.rho_curve = 0.0;
    assert!(c.validate("f").unwrap_err().to_string().contains("rhoCurve"));

    let mut c = FanCurve::quadratic(500.0, 2.0);
    c.n_speed = -1.0;
    assert!(c.validate("f").unwrap_err().to_string().contains("speed"));

    for eta in [0.0 as Scalar, -0.5, 1.5] {
        let mut c = FanCurve::quadratic(500.0, 2.0);
        c.efficiency = eta;
        assert!(
            c.validate("f").unwrap_err().to_string().contains("efficiency"),
            "efficiency {eta} must be refused"
        );
    }

    assert!(FanCurve::quadratic(0.0, 2.0).validate("f").is_err());
    assert!(FanCurve::quadratic(500.0, 0.0).validate("f").is_err());
}

#[test]
fn a_direction_this_solver_does_not_have_is_refused_by_name() {
    assert_eq!(FanDirection::from_name("outflow", "p").unwrap(), FanDirection::Outflow);
    assert_eq!(FanDirection::from_name("supply", "p").unwrap(), FanDirection::Inflow);
    let e = FanDirection::from_name("sideways", "crac1").expect_err("no such direction");
    let m = e.to_string();
    assert!(m.contains("crac1") && m.contains("outflow") && m.contains("inflow"), "{m}");
}

/// SPEC-LIT §52.5's field rule: a fan condition belongs on the pressure.
#[test]
fn a_fan_condition_on_a_field_that_is_not_the_pressure_is_refused_by_name() {
    crate::io::contract::set_permissive(false);
    assert_eq!(BcKind::from_name("fanPressure", "p", "outlet").unwrap(), BcKind::FanPressure);
    assert_eq!(BcKind::from_name("fan", "p_rgh", "outlet").unwrap(), BcKind::FanPressure);
    assert_eq!(
        BcKind::from_name("porousJumpPressure", "p", "tile").unwrap(),
        BcKind::PorousJumpPressure
    );
    assert_eq!(
        BcKind::from_name("porousBafflePressure", "p", "tile").unwrap(),
        BcKind::PorousJumpPressure
    );

    for (name, field) in
        [("fanPressure", "T"), ("fan", "U"), ("porousJumpPressure", "T"), ("fan", "pRef")]
    {
        let e = BcKind::from_name(name, field, "outlet")
            .unwrap_err()
            .to_string();
        assert!(
            e.contains("PRESSURE"),
            "`{name}` on `{field}` must be refused naming the pressure: {e}"
        );
    }

    // And the round trip through the published menu reaches the real kind,
    // not `Calculated` - S15.5's rule, extended again.
    for n in ["fanPressure", "fan", "porousJumpPressure", "porousBafflePressure"] {
        assert!(
            crate::field::IMPLEMENTED_BC_NAMES.contains(&n),
            "{n} is not in IMPLEMENTED_BC_NAMES"
        );
    }
}

/// SPEC-LIT §52.9 and §53.5: the two things this tranche refuses to build.
#[test]
fn the_woodbury_path_and_baffle_insertion_are_refused_by_name() {
    crate::io::contract::set_permissive(false);

    let e = refuse_capacitance_fft("pressureSolver").unwrap_err().to_string();
    assert!(e.contains("pbicgstab"), "must name the fallback backend: {e}");
    assert!(e.contains("S52.8"), "must name the derivation that would make it possible: {e}");
    assert!(e.contains("NOT implemented"), "must say plainly that it is not built: {e}");

    let e = refuse_baffle_insertion("devices/tile/baffle").unwrap_err().to_string();
    assert!(e.contains("TOPOLOGY MUTATION"), "{e}");
    assert!(e.contains("mesh-generation time"), "must name route one: {e}");
    assert!(e.contains("separate region"), "must name route two: {e}");
}

/// (S55.5): the shaft power, and that a reversed machine reports a negative
/// one rather than an absolute value.
#[test]
fn the_shaft_power_is_q_dp_over_eta() {
    let mut c = FanCurve::quadratic(500.0, 2.0);
    c.efficiency = 0.62;
    let q = 1.0 as Scalar;
    let (dp, _) = c.at(q);
    assert!(rel(c.shaft_power(q), q * dp / 0.62) < 1e-14);
    assert!(c.shaft_power(-0.4) < 0.0, "a machine driven backwards does negative work");
}

// ==========================================================================
//  §53 - the porous jump, on the host
// ==========================================================================

/// (S53.6) and its two limits, plus the contradiction §53.4 records.
#[test]
fn the_perforated_plate_loss_coefficient_has_the_right_limits() {
    let k25 = PorousJumpCoeffs::loss_coefficient_of_open_area(0.25).expect("0.25");
    assert!(
        rel(k25, 30.6782) < 1e-4,
        "K(0.25) = {k25}; the design note says ~30 and (S53.6) gives 30.68"
    );

    // The note's second claim, recorded as a contradiction rather than
    // silently accommodated.
    let k56 = PorousJumpCoeffs::loss_coefficient_of_open_area(0.56).expect("0.56");
    assert!(
        (k56 - 4.0).abs() > 0.5,
        "SPEC-LIT S53.4 records that the design note's `K ~= 4 at sigma = 0.56` \
         disagrees with (S53.6), which gives {k56}. If they now agree, the formula \
         changed and S53.4 needs rewriting"
    );
    assert!(rel(k56, 2.9367) < 1e-3, "K(0.56) = {k56}");
    assert!(
        rel(PorousJumpCoeffs::loss_coefficient_of_open_area(0.50).unwrap(), 4.3695) < 1e-3,
        "and 4.37 is (S53.6)'s value at sigma = 0.50, which is where the note's \
         number looks like it belongs"
    );

    // The limits.
    assert!(PorousJumpCoeffs::loss_coefficient_of_open_area(1.0).unwrap().abs() < 1e-30);
    let mut last = 0.0 as Scalar;
    for s in [0.5 as Scalar, 0.2, 0.05, 0.01, 0.001] {
        let k = PorousJumpCoeffs::loss_coefficient_of_open_area(s).unwrap();
        assert!(k > last, "K must grow without bound as sigma -> 0");
        last = k;
    }
    assert!(last > 1e6, "K(0.001) = {last} should be enormous");

    for bad in [0.0 as Scalar, -0.1, 1.5] {
        assert!(PorousJumpCoeffs::loss_coefficient_of_open_area(bad).is_err());
    }
}

/// (S53.2)/(S53.3): the two parameterisations are one.
#[test]
fn the_two_jump_parameterisations_agree() {
    let (c2, t_m, nu, k) = (17.0 as Scalar, 0.025 as Scalar, 1.5e-5 as Scalar, 0.425 as Scalar);
    assert!(rel(c2 * t_m, k) < 1e-14, "the test's own premise");

    // alpha -> infinity kills the viscous half.
    let a = PorousJumpCoeffs::from_darcy_forchheimer(1e30, c2, t_m, nu).expect("df");
    let b = PorousJumpCoeffs::from_loss_coefficient(k).expect("K");
    assert!(rel(a.r_inert, b.r_inert) < 1e-14);
    assert!(a.r_visc < 1e-30);

    for (phi, area) in [(0.1 as Scalar, 0.36 as Scalar), (-2.5, 1.0)] {
        assert!(rel(a.resistance(phi, area), b.resistance(phi, area)) < 1e-13);
    }

    // The viscous half is linear in phi and the inertial half is quadratic.
    let c = PorousJumpCoeffs::from_darcy_forchheimer(1e-9, c2, t_m, nu).expect("df");
    let (a1, a2) = (c.resistance(1.0, 1.0), c.resistance(2.0, 1.0));
    assert!(rel(a2 - c.r_visc, 2.0 * (a1 - c.r_visc)) < 1e-12, "R's inertial part is linear in |phi|");

    assert!(PorousJumpCoeffs::from_darcy_forchheimer(0.0, 1.0, 1.0, 1.0).is_err());
    assert!(PorousJumpCoeffs::from_darcy_forchheimer(1.0, -1.0, 1.0, 1.0).is_err());
    assert!(PorousJumpCoeffs::from_loss_coefficient(-1.0).is_err());

    // R = 0 exactly for the default - the bitwise-inert case of §53.2.
    assert_eq!(PorousJumpCoeffs::default().resistance(12.0, 3.0), 0.0);
}

// ==========================================================================
//  The device rigs
// ==========================================================================

/// A 1-D chain of `n` cells along `x`, length `l`, with the two `x` patches
/// available for boundary conditions.
fn chain(n: usize) -> HostMesh {
    block([n, 1, 1], Vec3::new(1.0 / n as Scalar, 0.4, 0.3))
}

struct Rig {
    hm: HostMesh,
    m: GpuMesh,
    fvk: FvKernels,
    lduk: LduKernels,
    solk: SolverKernels,
    ws: SolverWorkspace,
    p: GpuScalarField,
    phi: GpuSurfaceScalarField,
    phi_hbya: GpuSurfaceScalarField,
    rauf: GpuSurfaceScalarField,
    rauf_mag_sf: GpuSurfaceScalarField,
    a: GpuLduMatrix,
}

impl Rig {
    /// A pure-Laplacian pressure rig: `rAU_f` uniform, `phi_HbyA` zero,
    /// Dirichlet on `xmin`/`xmax` and zero-gradient everywhere else.
    ///
    /// That is exactly the operator §53.8 Gate 53-A's series law is written
    /// for, and it is ONE assembly and ONE solve.
    fn new(gpu: &Gpu, hm: HostMesh, rau: Scalar) -> Result<Self> {
        let m = GpuMesh::upload(gpu, &hm)?;
        let mut p = GpuScalarField::zeros(gpu, &m, "p")?;
        let mut kind = vec![BcKind::ZeroGradient as Label; hm.n_boundary_faces];
        let mut fr = vec![0.0 as Scalar; hm.n_boundary_faces];
        let rv = vec![0.0 as Scalar; hm.n_boundary_faces];
        for (i, k) in hm.b_kind.iter().enumerate() {
            if *k == crate::mesh::PatchKind::Empty as Label {
                kind[i] = BcKind::Empty as Label;
            }
        }
        for pi in &hm.patches {
            if pi.name == "xmin" || pi.name == "xmax" {
                for bf in pi.start..pi.start + pi.size {
                    kind[bf] = BcKind::FixedValue as Label;
                    fr[bf] = 1.0;
                }
            }
        }
        gpu.write(&mut p.bc_kind, &kind)?;
        gpu.write(&mut p.fr, &fr)?;
        gpu.write(&mut p.ref_value, &rv)?;

        let mut rauf = GpuSurfaceScalarField::zeros(gpu, &m, "rauf")?;
        gpu.write(&mut rauf.f, &vec![rau; hm.n_internal_faces.max(1)][..hm.n_internal_faces])?;
        gpu.write(&mut rauf.bf, &vec![rau; hm.n_boundary_faces])?;

        let mut rauf_mag_sf = GpuSurfaceScalarField::zeros(gpu, &m, "raufMagSf")?;
        let f: Vec<Scalar> = hm.mag_sf.iter().map(|a| rau * a).collect();
        let bf: Vec<Scalar> = hm.b_mag_sf.iter().map(|a| rau * a).collect();
        gpu.write(&mut rauf_mag_sf.f, &f)?;
        gpu.write(&mut rauf_mag_sf.bf, &bf)?;

        Ok(Self {
            fvk: FvKernels::new(gpu)?,
            lduk: LduKernels::new(gpu)?,
            solk: SolverKernels::new(gpu)?,
            ws: SolverWorkspace::for_mesh(gpu, &m)?,
            p,
            phi: GpuSurfaceScalarField::zeros(gpu, &m, "phi")?,
            phi_hbya: GpuSurfaceScalarField::zeros(gpu, &m, "phiHbyA")?,
            rauf,
            rauf_mag_sf,
            a: GpuLduMatrix::new(gpu, &m)?,
            m,
            hm,
        })
    }

    fn set_dirichlet(&mut self, gpu: &Gpu, patch: &str, v: Scalar) -> Result<()> {
        let mut rv = gpu.download(&self.p.ref_value)?;
        let pi = self.hm.patches.iter().find(|p| p.name == patch).expect("patch");
        for bf in pi.start..pi.start + pi.size {
            rv[bf] = v;
        }
        gpu.write(&mut self.p.ref_value, &rv)
    }

    fn assemble_and_solve(&mut self, gpu: &Gpu) -> Result<()> {
        self.a.zero(gpu)?;
        fv::fvm_laplacian(
            gpu,
            &self.fvk,
            &mut self.a,
            &self.m,
            &self.rauf_mag_sf.f,
            &self.rauf_mag_sf.bf,
            &self.p,
            1.0,
        )?;
        ldu_ops::add_boundary_contributions(gpu, &self.lduk, &mut self.a, &self.m)?;
        let ctrl = SolverControls {
            solver: LinearSolverKind::PCG,
            precon: Preconditioner::Dic,
            tolerance: 1e-30,
            rel_tol: 0.0,
            max_iter: 5000,
            ..SolverControls::default()
        };
        crate::solver::solve(gpu, &self.solk, &mut self.p.f, &self.a, &self.m, &mut self.ws, &ctrl)?;
        crate::field_ops::correct_boundary_conditions(
            gpu,
            &crate::field_ops::FieldKernels::new(gpu)?,
            &mut self.p,
            &self.m,
        )
    }

    /// `phi_f = phi_HbyA,f - rAU_f |Sf| Delta_f (p_N - p_P)` on internal
    /// faces - the same expression `momCorrectFlux` evaluates.
    fn internal_flux(&self, gpu: &Gpu) -> Result<Vec<Scalar>> {
        let p = gpu.download(&self.p.f)?;
        let g = gpu.download(&self.rauf_mag_sf.f)?;
        let ph = gpu.download(&self.phi_hbya.f)?;
        Ok((0..self.hm.n_internal_faces)
            .map(|f| {
                let (o, n) = (self.hm.owner[f] as usize, self.hm.neighbour[f] as usize);
                ph[f] - g[f] * self.hm.delta_coeffs[f] * (p[n] - p[o])
            })
            .collect())
    }
}

/// §53.8 Gate 53-A: resistances in series, exactly.
///
/// A 1-D chain with Dirichlet ends carries `Q = dp/SUM_i (1/D_i)`; a jump on
/// face `j` replaces `1/D_j` by `1/D_j + R`. ONE assembly, ONE solve.
#[test]
fn gate_53a_a_porous_jump_puts_resistances_in_series() {
    let Some(gpu) = gpu() else { return };
    let n = 12;
    let hm = chain(n);
    let rau = 0.017 as Scalar;
    let dp = 5.0 as Scalar;

    // The chain's own resistance, from the mesh, once.
    let base: Scalar = {
        let r = Rig::new(&gpu, chain(n), rau).expect("rig");
        let mut s = 0.0;
        for f in 0..r.hm.n_internal_faces {
            s += 1.0 / (rau * r.hm.mag_sf[f] * r.hm.delta_coeffs[f]);
        }
        // Plus the two boundary halves.
        for pi in &r.hm.patches {
            if pi.name == "xmin" || pi.name == "xmax" {
                for bf in pi.start..pi.start + pi.size {
                    s += 1.0 / (rau * r.hm.b_mag_sf[bf] * r.hm.b_delta_coeffs[bf]);
                }
            }
        }
        s
    };

    let mid = hm.n_internal_faces / 2;
    for r_jump in [0.0 as Scalar, 0.3 * base, 4.0 * base, 1e12 * base] {
        let mut rig = Rig::new(&gpu, chain(n), rau).expect("rig");
        rig.set_dirichlet(&gpu, "xmin", dp).expect("in");
        rig.set_dirichlet(&gpu, "xmax", 0.0).expect("out");

        // A purely VISCOUS jump, so R is independent of the flux and the
        // closed form is exact rather than a fixed point.
        let area = rig.hm.mag_sf[mid];
        let coeffs = PorousJumpCoeffs { r_visc: r_jump * area, r_inert: 0.0 };
        assert!(rel(coeffs.resistance(0.0, area), r_jump) < 1e-14 || r_jump == 0.0);

        let jumps =
            [PorousJump::Internal { faces: vec![mid as Label], coeffs }];
        let mut fd =
            FlowDevices::new(&gpu, &rig.hm, Vec::new(), &jumps, 1.2).expect("devices");
        fd.update(
            &gpu,
            &rig.m,
            &rig.phi,
            &mut rig.phi_hbya,
            &mut rig.rauf,
            &mut rig.rauf_mag_sf,
            &mut rig.p,
        )
        .expect("update");

        rig.assemble_and_solve(&gpu).expect("solve");
        let flux = rig.internal_flux(&gpu).expect("flux");
        let q: Scalar = flux[mid];

        let want = dp / (base + r_jump);
        if r_jump > 1e6 * base {
            assert!(
                q.abs() < 1e-10 * dp / base,
                "R -> infinity must be a wall: the face still carries {q}"
            );
        } else {
            assert!(
                rel(q, want) < 1e-10,
                "(S53.7) at R = {r_jump}: the face carries {q}, series says {want}"
            );
        }

        // Every face carries the same flux - the chain is conservative.
        //
        // The tolerance is tied to the UNJUMPED flux, not to `q`: at
        // `R = 1e12 base` the chain carries about 4e-12 and the solve's own
        // round-off is 5e-17, which is 1e-5 of it. A tolerance relative to a
        // vanishing quantity is a tolerance on round-off.
        let scale = dp / base;
        let spread = flux
            .iter()
            .fold(0.0 as Scalar, |m, v| m.max((v - q).abs()));
        assert!(
            spread < 1e-10 * scale,
            "the chain leaks: spread {spread} against the unjumped flux {scale}"
        );
    }
}

/// §53.2's bitwise gate: `R = 0` leaves all three arrays untouched, and the
/// solved field bit-for-bit the no-jump one.
#[test]
fn a_zero_resistance_jump_is_bitwise_inert() {
    let Some(gpu) = gpu() else { return };
    let n = 9;

    let solve = |with_jump: bool| -> (Vec<Scalar>, Vec<Scalar>, Vec<Scalar>, Vec<Scalar>) {
        let mut rig = Rig::new(&gpu, chain(n), 0.021).expect("rig");
        rig.set_dirichlet(&gpu, "xmin", 3.0).expect("in");
        rig.set_dirichlet(&gpu, "xmax", -1.0).expect("out");
        // A non-zero phi_HbyA, so the third array's division is exercised.
        let ph: Vec<Scalar> =
            (0..rig.hm.n_internal_faces).map(|f| 0.001 * (f as Scalar + 1.0)).collect();
        gpu.write(&mut rig.phi_hbya.f, &ph).expect("phiHbyA");

        if with_jump {
            let faces: Vec<Label> = (0..rig.hm.n_internal_faces as Label).collect();
            let jumps = [PorousJump::Internal {
                faces,
                coeffs: PorousJumpCoeffs::default(),
            }];
            let mut fd =
                FlowDevices::new(&gpu, &rig.hm, Vec::new(), &jumps, 1.2).expect("devices");
            fd.update(
                &gpu,
                &rig.m,
                &rig.phi,
                &mut rig.phi_hbya,
                &mut rig.rauf,
                &mut rig.rauf_mag_sf,
                &mut rig.p,
            )
            .expect("update");
        }
        rig.assemble_and_solve(&gpu).expect("solve");
        (
            gpu.download(&rig.p.f).expect("p"),
            gpu.download(&rig.rauf.f).expect("rauf"),
            gpu.download(&rig.rauf_mag_sf.f).expect("g"),
            gpu.download(&rig.phi_hbya.f).expect("ph"),
        )
    };

    let a = solve(false);
    let b = solve(true);
    assert_eq!(a.1, b.1, "rAU_f moved under a zero-resistance jump");
    assert_eq!(a.2, b.2, "rAU_f|Sf| moved under a zero-resistance jump");
    assert_eq!(a.3, b.3, "phi_HbyA moved under a zero-resistance jump");
    assert_eq!(
        a.0, b.0,
        "SPEC-LIT S53.2: x/(1 + 0*D) is x/1.0 which is x BITWISE, so the solved \
         field must be bit-for-bit the no-jump one"
    );
}

/// §53.7: `upper[f] == lower[f]` on a jump face, and the matrix stays
/// symmetric.
#[test]
fn a_jump_leaves_the_matrix_symmetric() {
    let Some(gpu) = gpu() else { return };
    let mut rig = Rig::new(&gpu, chain(10), 0.019).expect("rig");
    rig.set_dirichlet(&gpu, "xmin", 2.0).expect("in");

    let faces: Vec<Label> = vec![2, 5, 7];
    let jumps = [PorousJump::Internal {
        faces: faces.clone(),
        coeffs: PorousJumpCoeffs::from_loss_coefficient(30.68).expect("K"),
    }];
    // A non-zero flux, so the Forchheimer half is actually active.
    let phi: Vec<Scalar> = vec![0.02; rig.hm.n_internal_faces];
    gpu.write(&mut rig.phi.f, &phi).expect("phi");

    let mut fd = FlowDevices::new(&gpu, &rig.hm, Vec::new(), &jumps, 1.2).expect("devices");
    fd.update(
        &gpu,
        &rig.m,
        &rig.phi,
        &mut rig.phi_hbya,
        &mut rig.rauf,
        &mut rig.rauf_mag_sf,
        &mut rig.p,
    )
    .expect("update");

    // The coefficient really moved on the listed faces and nowhere else.
    let g = gpu.download(&rig.rauf_mag_sf.f).expect("g");
    for f in 0..rig.hm.n_internal_faces {
        let base = 0.019 * rig.hm.mag_sf[f];
        if faces.contains(&(f as Label)) {
            assert!(g[f] < base * 0.999, "face {f} should have been reduced");
        } else {
            assert_eq!(g[f], base, "face {f} is not a jump face and must not move");
        }
    }

    rig.a.zero(&gpu).expect("zero");
    fv::fvm_laplacian(
        &gpu,
        &rig.fvk,
        &mut rig.a,
        &rig.m,
        &rig.rauf_mag_sf.f,
        &rig.rauf_mag_sf.bf,
        &rig.p,
        1.0,
    )
    .expect("laplacian");
    ldu_ops::add_boundary_contributions(&gpu, &rig.lduk, &mut rig.a, &rig.m).expect("bnd");

    let upper = gpu.download(&rig.a.upper).expect("upper");
    let lower = gpu.download(&rig.a.lower).expect("lower");
    for f in 0..rig.hm.n_internal_faces {
        assert_eq!(upper[f], lower[f], "face {f}: upper != lower");
    }
    assert!(
        crate::solver::matrix_is_symmetric(&gpu, &rig.solk, &mut rig.ws, &rig.a, &rig.m)
            .expect("sym"),
        "SPEC-LIT S53.2: both halves get the same reduced coefficient, so symmetry \
         is preserved IDENTICALLY"
    );

    // And the M-matrix property: every reduced coefficient is in (0, D_f].
    for f in 0..rig.hm.n_internal_faces {
        let base = 0.019 * rig.hm.mag_sf[f];
        assert!(g[f] > 0.0 && g[f] <= base, "D_eff out of (0, D_f] on face {f}");
    }
}

/// §53.7: a jump face reverses sign when the pressure difference does. This
/// is the property a prescribed-flow tile lacks **by construction**, and it
/// is why §53.8's structural gate is worth having even without Karki's data.
#[test]
fn a_jump_face_reverses_with_the_pressure_difference() {
    let Some(gpu) = gpu() else { return };
    let mid = 5usize;
    let mut fluxes = Vec::new();
    for dp in [3.0 as Scalar, -3.0] {
        let mut rig = Rig::new(&gpu, chain(11), 0.023).expect("rig");
        rig.set_dirichlet(&gpu, "xmin", dp).expect("in");
        rig.set_dirichlet(&gpu, "xmax", 0.0).expect("out");

        let jumps = [PorousJump::Internal {
            faces: vec![mid as Label],
            coeffs: PorousJumpCoeffs::from_loss_coefficient(12.0).expect("K"),
        }];
        let mut fd =
            FlowDevices::new(&gpu, &rig.hm, Vec::new(), &jumps, 1.2).expect("devices");

        // Two Picard sweeps, so the Forchheimer R sees a real flux.
        for _ in 0..8 {
            let mut r2 = Rig::new(&gpu, chain(11), 0.023).expect("rig");
            r2.set_dirichlet(&gpu, "xmin", dp).expect("in");
            r2.set_dirichlet(&gpu, "xmax", 0.0).expect("out");
            let phi = gpu.download(&rig.phi.f).expect("phi");
            gpu.write(&mut r2.phi.f, &phi).expect("seed");
            fd.update(
                &gpu,
                &r2.m,
                &r2.phi,
                &mut r2.phi_hbya,
                &mut r2.rauf,
                &mut r2.rauf_mag_sf,
                &mut r2.p,
            )
            .expect("update");
            r2.assemble_and_solve(&gpu).expect("solve");
            let f = r2.internal_flux(&gpu).expect("flux");
            gpu.write(&mut rig.phi.f, &f).expect("write");
        }
        fluxes.push(gpu.download(&rig.phi.f).expect("phi")[mid]);
    }
    assert!(
        fluxes[0] > 0.0 && fluxes[1] < 0.0,
        "the jump must carry flow BOTH ways: {fluxes:?}. A prescribed-flow tile \
         cannot do this at all, which is SPEC-LIT S53.8's point"
    );
    assert!(
        rel(fluxes[0], -fluxes[1]) < 1e-9,
        "and symmetrically, since R depends on |phi|: {fluxes:?}"
    );
}

// ==========================================================================
//  §52 on the device
// ==========================================================================

/// A rig with a fan on `xmax` and a Dirichlet on `xmin`.
fn fan_rig(gpu: &Gpu, n: usize, fan: FanPatch, rau: Scalar) -> (Rig, FlowDevices) {
    let mut rig = Rig::new(gpu, chain(n), rau).expect("rig");
    // The fan patch is no longer a plain Dirichlet: `crate::fan` owns it.
    let mut kind = gpu.download(&rig.p.bc_kind).expect("kind");
    let pi = span(&rig.hm, "xmax");
    for bf in pi.0..pi.0 + pi.1 {
        kind[bf] = BcKind::FanPressure as Label;
    }
    gpu.write(&mut rig.p.bc_kind, &kind).expect("kind");
    let fd = FlowDevices::new(gpu, &rig.hm, vec![fan], &[], 1.2).expect("devices");
    (rig, fd)
}

/// The device triple is exactly [`lumped_triple`], for every curve kind and
/// both directions.
#[test]
fn the_device_triple_mirrors_the_host() {
    let Some(gpu) = gpu() else { return };
    let curves = [
        FanCurve::flat(42.0),
        FanCurve::quadratic(120.0, 0.5),
        FanCurve::table(vec![(0.0, 200.0), (0.2, 170.0), (0.5, 90.0), (0.8, 0.0)]),
    ];
    for c in &curves {
        for dir in [FanDirection::Outflow, FanDirection::Inflow] {
            let mut fan = FanPatch::new("xmax", c.clone(), dir);
            fan.ambient = 7.5;
            fan.relaxation = 1.0;
            let (mut rig, mut fd) = fan_rig(&gpu, 8, fan.clone(), 0.02);

            // Seed a non-trivial phi and phi_HbyA on the fan patch.
            let pi = span(&rig.hm, "xmax");
            let mut phi = vec![0.0 as Scalar; rig.hm.n_boundary_faces];
            let mut ph = vec![0.0 as Scalar; rig.hm.n_boundary_faces];
            for (i, bf) in (pi.0..pi.0 + pi.1).enumerate() {
                phi[bf] = 0.011 * (i as Scalar + 1.0);
                ph[bf] = 0.004 * (i as Scalar + 2.0);
            }
            gpu.write(&mut rig.phi.bf, &phi).expect("phi");
            gpu.write(&mut rig.phi_hbya.bf, &ph).expect("phiHbyA");

            fd.update(
                &gpu,
                &rig.m,
                &rig.phi,
                &mut rig.phi_hbya,
                &mut rig.rauf,
                &mut rig.rauf_mag_sf,
                &mut rig.p,
            )
            .expect("update");

            let q: Scalar = (pi.0..pi.0 + pi.1).map(|bf| phi[bf]).sum();
            let phi_sum: Scalar = (pi.0..pi.0 + pi.1).map(|bf| ph[bf]).sum();
            let sigma_d: Scalar = (pi.0..pi.0 + pi.1)
                .map(|bf| 0.02 * rig.hm.b_mag_sf[bf] * rig.hm.b_delta_coeffs[bf])
                .sum();
            let (fr, rv, _) =
                lumped_triple(c, dir, 7.5, 1.2, q, phi_sum, sigma_d);

            let dfr = gpu.download(&rig.p.fr).expect("fr");
            let drv = gpu.download(&rig.p.ref_value).expect("rv");
            let drg = gpu.download(&rig.p.ref_grad).expect("rg");
            for bf in pi.0..pi.0 + pi.1 {
                assert!(
                    rel(dfr[bf], fr) < 1e-12,
                    "fr: device {} host {fr} ({c:?}, {dir:?})",
                    dfr[bf]
                );
                assert!(
                    rel(drv[bf], rv) < 1e-11,
                    "refValue: device {} host {rv} ({c:?}, {dir:?})",
                    drv[bf]
                );
                assert_eq!(drg[bf], 0.0, "refGrad must be exactly zero");
            }

            // And the reported state matches.
            let st = fd.states(&gpu).expect("states");
            assert!(rel(st[0].q, q) < 1e-13, "reported Q {} vs {q}", st[0].q);
            assert!(rel(st[0].fr, fr) < 1e-12);
        }
    }
}

/// §52.4: a flat curve stamps `(1.0, c, 0.0)` on the device too, and the
/// solved field is **bit-for-bit** the `fixedValue` one.
#[test]
fn gate_52c_a_flat_curve_reproduces_the_fixed_value_field_bitwise() {
    let Some(gpu) = gpu() else { return };
    let dp = 30.0 as Scalar;
    let (p_a, rho) = (2.0 as Scalar, 1.2 as Scalar);
    let want_c = p_a - dp / rho;

    // (a) the fan.
    let mut fan = FanPatch::new("xmax", FanCurve::flat(dp), FanDirection::Outflow);
    fan.ambient = p_a;
    let (mut rig, mut fd) = fan_rig(&gpu, 14, fan, 0.02);
    rig.set_dirichlet(&gpu, "xmin", 5.0).expect("in");
    for _ in 0..3 {
        fd.update(
            &gpu,
            &rig.m,
            &rig.phi,
            &mut rig.phi_hbya,
            &mut rig.rauf,
            &mut rig.rauf_mag_sf,
            &mut rig.p,
        )
        .expect("update");
        rig.assemble_and_solve(&gpu).expect("solve");
    }
    let p_fan = gpu.download(&rig.p.f).expect("p");
    let fr = gpu.download(&rig.p.fr).expect("fr");
    let rv = gpu.download(&rig.p.ref_value).expect("rv");
    let pi = span(&rig.hm, "xmax");
    for bf in pi.0..pi.0 + pi.1 {
        assert_eq!(fr[bf], 1.0, "S = 0 must give fr exactly 1.0");
        assert_eq!(rv[bf], want_c, "refValue must be exactly p_a - dp/rho");
    }

    let sys_fan = (
        gpu.download(&rig.a.diag).expect("d"),
        gpu.download(&rig.a.upper).expect("u"),
        gpu.download(&rig.a.lower).expect("l"),
        gpu.download(&rig.a.source).expect("s"),
    );

    // (b) the plain fixedValue, same number, driven through the SAME solve
    //     sequence from the same starting field.
    //
    //     The sequence has to match. A Krylov solve is a fixed point of its
    //     own iteration only up to the tolerance it stopped at, so one solve
    //     from zero and three solves from each other's output land on two
    //     equally-converged but not bit-identical vectors - the same
    //     observation S47.11 records for a two-region mesh. What is bitwise
    //     is the SYSTEM, which is checked first and separately.
    let mut ref_rig = Rig::new(&gpu, chain(14), 0.02).expect("rig");
    ref_rig.set_dirichlet(&gpu, "xmin", 5.0).expect("in");
    ref_rig.set_dirichlet(&gpu, "xmax", want_c).expect("out");
    for _ in 0..3 {
        ref_rig.assemble_and_solve(&gpu).expect("solve");
    }
    let p_ref = gpu.download(&ref_rig.p.f).expect("p");
    let sys_ref = (
        gpu.download(&ref_rig.a.diag).expect("d"),
        gpu.download(&ref_rig.a.upper).expect("u"),
        gpu.download(&ref_rig.a.lower).expect("l"),
        gpu.download(&ref_rig.a.source).expect("s"),
    );

    assert_eq!(
        sys_fan, sys_ref,
        "SPEC-LIT S52.4: at S = 0 the fan stamps the fixedValue triple bit for bit, \
         so the ASSEMBLED SYSTEM - diag, upper, lower and source - must be \
         bit-for-bit the fixedValue one. This is the exact half of the claim."
    );
    assert_eq!(
        p_fan, p_ref,
        "SPEC-LIT S52.4: a flat fan curve IS fixedValue, bitwise. It is not nearly \
         fixedValue."
    );
}

/// The fan patch's converged state satisfies its own curve, and the flow it
/// delivers is the flow the system delivers at that pressure.
///
/// This is §52.12 Gate 52-A's real content on a real system: no model of the
/// duct is needed, because the identity is closed.
#[test]
fn the_fan_lands_where_its_curve_crosses_the_systems_own_characteristic() {
    let Some(gpu) = gpu() else { return };
    let (p_a, rho, rau) = (0.0 as Scalar, 1.2 as Scalar, 0.02 as Scalar);
    let mut fan = FanPatch::new("xmax", FanCurve::quadratic(60.0, 0.05), FanDirection::Outflow);
    fan.ambient = p_a;
    fan.relaxation = 0.6;

    let (mut rig, mut fd) = fan_rig(&gpu, 16, fan.clone(), rau);
    rig.set_dirichlet(&gpu, "xmin", 0.0).expect("in");

    for _ in 0..200 {
        fd.update(
            &gpu,
            &rig.m,
            &rig.phi,
            &mut rig.phi_hbya,
            &mut rig.rauf,
            &mut rig.rauf_mag_sf,
            &mut rig.p,
        )
        .expect("update");
        rig.assemble_and_solve(&gpu).expect("solve");
        // Recompute the boundary flux the pressure just produced.
        let p = gpu.download(&rig.p.f).expect("p");
        let pb = gpu.download(&rig.p.bf).expect("pb");
        let g = gpu.download(&rig.rauf_mag_sf.bf).expect("g");
        let mut phi = gpu.download(&rig.phi.bf).expect("phi");
        for bf in 0..rig.hm.n_boundary_faces {
            let c = rig.hm.b_face_cells[bf] as usize;
            phi[bf] = -g[bf] * rig.hm.b_delta_coeffs[bf] * (pb[bf] - p[c]);
        }
        gpu.write(&mut rig.phi.bf, &phi).expect("phi");
    }

    let st = fd.states(&gpu).expect("states");
    let q = st[0].q;
    assert!(q > 0.0, "the fan should be blowing out; Q = {q}");
    assert!(
        rel(q, st[0].q_star) < 1e-6,
        "the operating point has not converged: Q = {q}, Q* = {}",
        st[0].q_star
    );

    // 1. The patch face value satisfies (S52.3) at the converged flow.
    let pb = gpu.download(&rig.p.bf).expect("pb");
    let pi = span(&rig.hm, "xmax");
    let (dp, _) = fan.curve.at(q);
    let want = p_a - dp / rho;
    for bf in pi.0..pi.0 + pi.1 {
        assert!(
            rel(pb[bf], want) < 1e-8,
            "(S52.3): the face sits at {} where the curve says {want}",
            pb[bf]
        );
    }

    // 2. And a PLAIN fixedValue run at that same pressure delivers the same
    //    flow - which is what "the fan found the system's operating point"
    //    means, with no model of the system in the test at all.
    let mut r2 = Rig::new(&gpu, chain(16), rau).expect("rig");
    r2.set_dirichlet(&gpu, "xmin", 0.0).expect("in");
    r2.set_dirichlet(&gpu, "xmax", want).expect("out");
    r2.assemble_and_solve(&gpu).expect("solve");
    let p = gpu.download(&r2.p.f).expect("p");
    let pb2 = gpu.download(&r2.p.bf).expect("pb");
    let g = gpu.download(&r2.rauf_mag_sf.bf).expect("g");
    let q2: Scalar = (pi.0..pi.0 + pi.1)
        .map(|bf| {
            let c = r2.hm.b_face_cells[bf] as usize;
            -g[bf] * r2.hm.b_delta_coeffs[bf] * (pb2[bf] - p[c])
        })
        .sum();
    assert!(
        rel(q, q2) < 1e-7,
        "the fan delivers {q} but a fixedValue at its own pressure delivers {q2}"
    );
}

/// §52.11: the pressure matrix stays symmetric with a fan patch, and the
/// diagonal gains `fr D_f >= 0`.
#[test]
fn the_pressure_matrix_stays_symmetric_and_an_m_matrix_with_a_fan_patch() {
    let Some(gpu) = gpu() else { return };
    let mut fan = FanPatch::new("xmax", FanCurve::quadratic(60.0, 0.05), FanDirection::Outflow);
    fan.ambient = 1.0;
    let (mut rig, mut fd) = fan_rig(&gpu, 10, fan, 0.02);
    let phi: Vec<Scalar> = vec![0.004; rig.hm.n_boundary_faces];
    gpu.write(&mut rig.phi.bf, &phi).expect("phi");

    fd.update(
        &gpu,
        &rig.m,
        &rig.phi,
        &mut rig.phi_hbya,
        &mut rig.rauf,
        &mut rig.rauf_mag_sf,
        &mut rig.p,
    )
    .expect("update");

    let fr = gpu.download(&rig.p.fr).expect("fr");
    let pi = span(&rig.hm, "xmax");
    for bf in pi.0..pi.0 + pi.1 {
        assert!(fr[bf] > 0.0 && fr[bf] < 1.0, "fr must be in (0,1): {}", fr[bf]);
    }

    rig.a.zero(&gpu).expect("zero");
    fv::fvm_laplacian(
        &gpu,
        &rig.fvk,
        &mut rig.a,
        &rig.m,
        &rig.rauf_mag_sf.f,
        &rig.rauf_mag_sf.bf,
        &rig.p,
        1.0,
    )
    .expect("lap");
    ldu_ops::add_boundary_contributions(&gpu, &rig.lduk, &mut rig.a, &rig.m).expect("bnd");

    assert!(
        crate::solver::matrix_is_symmetric(&gpu, &rig.solk, &mut rig.ws, &rig.a, &rig.m)
            .expect("sym"),
        "SPEC-LIT S52.2 consequence 1: the fan does not break symmetry"
    );

    // The diagonal of a fan-adjacent cell gains fr*D_f, which is >= 0 and
    // smaller than a plain Dirichlet's D_f - the matrix gets EASIER.
    let diag = gpu.download(&rig.a.diag).expect("diag");
    let mut ref_rig = Rig::new(&gpu, chain(10), 0.02).expect("rig");
    ref_rig.a.zero(&gpu).expect("zero");
    fv::fvm_laplacian(
        &gpu,
        &ref_rig.fvk,
        &mut ref_rig.a,
        &ref_rig.m,
        &ref_rig.rauf_mag_sf.f,
        &ref_rig.rauf_mag_sf.bf,
        &ref_rig.p,
        1.0,
    )
    .expect("lap");
    ldu_ops::add_boundary_contributions(&gpu, &ref_rig.lduk, &mut ref_rig.a, &ref_rig.m)
        .expect("bnd");
    let diag_ref = gpu.download(&ref_rig.a.diag).expect("diag");
    let c = rig.hm.b_face_cells[pi.0] as usize;
    assert!(
        diag[c] > diag_ref[c],
        "the fan cell's diagonal must GAIN fr D_f over the pure-Neumann case: \
         {} vs {}",
        diag[c],
        diag_ref[c]
    );
}

/// SPEC-LIT §52.8: `S = 0` keeps the cuFFT path; a real curve loses it, with
/// a printed reason naming the face and its value fraction.
#[test]
fn a_flat_curve_keeps_the_fft_path_and_a_real_one_names_why_it_does_not() {
    let Some(gpu) = gpu() else { return };
    let probe = |curve: FanCurve| -> (bool, String) {
        let mut fan = FanPatch::new("xmax", curve, FanDirection::Outflow);
        fan.ambient = 1.0;
        let (mut rig, mut fd) = fan_rig(&gpu, 8, fan, 0.02);
        let phi: Vec<Scalar> = vec![0.004; rig.hm.n_boundary_faces];
        gpu.write(&mut rig.phi.bf, &phi).expect("phi");
        fd.update(
            &gpu,
            &rig.m,
            &rig.phi,
            &mut rig.phi_hbya,
            &mut rig.rauf,
            &mut rig.rauf_mag_sf,
            &mut rig.p,
        )
        .expect("update");
        let kind = gpu.download(&rig.p.bc_kind).expect("kind");
        let fr = gpu.download(&rig.p.fr).expect("fr");
        crate::pressure::cartesian::separable(&rig.hm, None, &kind, &fr)
    };

    let (ok, _) = probe(FanCurve::flat(20.0));
    assert!(
        ok,
        "SPEC-LIT S52.8: at S = 0 the face is uniformly Dirichlet and the FFT path \
         is NOT lost"
    );

    let (ok, why) = probe(FanCurve::quadratic(60.0, 0.05));
    assert!(!ok, "a real fan curve must disable the separable path");
    assert!(
        why.contains("Dirichlet") && why.contains("Neumann"),
        "the reason must say what was wrong: {why}"
    );
}

/// §52.7: two builds of the same fan case are bitwise identical.
#[test]
fn the_fan_update_is_bitwise_reproducible() {
    let Some(gpu) = gpu() else { return };
    let run = || -> (Vec<Scalar>, Vec<Scalar>) {
        let mut fan =
            FanPatch::new("xmax", FanCurve::table(vec![(0.0, 90.0), (0.03, 55.0), (0.07, 0.0)]), FanDirection::Outflow);
        fan.ambient = 1.5;
        let (mut rig, mut fd) = fan_rig(&gpu, 32, fan, 0.02);
        rig.set_dirichlet(&gpu, "xmin", 0.0).expect("in");
        for _ in 0..25 {
            fd.update(
                &gpu,
                &rig.m,
                &rig.phi,
                &mut rig.phi_hbya,
                &mut rig.rauf,
                &mut rig.rauf_mag_sf,
                &mut rig.p,
            )
            .expect("update");
            rig.assemble_and_solve(&gpu).expect("solve");
            let p = gpu.download(&rig.p.f).expect("p");
            let pb = gpu.download(&rig.p.bf).expect("pb");
            let g = gpu.download(&rig.rauf_mag_sf.bf).expect("g");
            let mut phi = gpu.download(&rig.phi.bf).expect("phi");
            for bf in 0..rig.hm.n_boundary_faces {
                let c = rig.hm.b_face_cells[bf] as usize;
                phi[bf] = -g[bf] * rig.hm.b_delta_coeffs[bf] * (pb[bf] - p[c]);
            }
            gpu.write(&mut rig.phi.bf, &phi).expect("phi");
        }
        (gpu.download(&rig.p.f).expect("p"), gpu.download(&rig.out_dummy()).expect("o"))
    };
    let a = run();
    let b = run();
    assert_eq!(a.0, b.0, "two identical fan runs must be BITWISE identical");
    assert_eq!(a.1, b.1);
}

impl Rig {
    /// The pressure boundary values, used only by the determinism test as a
    /// second array to compare.
    fn out_dummy(&self) -> &DevBuf<Scalar> {
        &self.p.bf
    }
}

// ==========================================================================
//  §55.6 - the pair tests
//
//  Each is two inputs identical in every entry but one, REQUIRED to produce
//  different output and failing by name if they do not.
// ==========================================================================

/// One fan run, reduced to the number a pair test compares.
fn fan_outcome(gpu: &Gpu, fan: FanPatch) -> (Scalar, Scalar, Scalar) {
    let (mut rig, mut fd) = fan_rig(gpu, 12, fan, 0.02);
    rig.set_dirichlet(gpu, "xmin", 0.0).expect("in");
    for _ in 0..60 {
        fd.update(
            gpu,
            &rig.m,
            &rig.phi,
            &mut rig.phi_hbya,
            &mut rig.rauf,
            &mut rig.rauf_mag_sf,
            &mut rig.p,
        )
        .expect("update");
        rig.assemble_and_solve(gpu).expect("solve");
        let p = gpu.download(&rig.p.f).expect("p");
        let pb = gpu.download(&rig.p.bf).expect("pb");
        let g = gpu.download(&rig.rauf_mag_sf.bf).expect("g");
        let mut phi = gpu.download(&rig.phi.bf).expect("phi");
        for bf in 0..rig.hm.n_boundary_faces {
            let c = rig.hm.b_face_cells[bf] as usize;
            phi[bf] = -g[bf] * rig.hm.b_delta_coeffs[bf] * (pb[bf] - p[c]);
        }
        gpu.write(&mut rig.phi.bf, &phi).expect("phi");
    }
    let st = fd.states(gpu).expect("states");
    let (_, total) = fd.shaft_power(gpu).expect("power");
    (st[0].q, st[0].fr, total)
}

fn base_fan() -> FanPatch {
    let mut f = FanPatch::new("xmax", FanCurve::quadratic(60.0, 0.05), FanDirection::Outflow);
    f.ambient = 0.0;
    f.relaxation = 0.6;
    f
}

macro_rules! pair {
    ($name:ident, $field:expr, $mutate:expr, $what:expr) => {
        #[test]
        fn $name() {
            let Some(gpu) = gpu() else { return };
            let a = base_fan();
            let mut b = base_fan();
            let f: &dyn Fn(&mut FanPatch) = &$mutate;
            f(&mut b);
            assert_ne!(a, b, "the two cases must actually differ in one entry");
            let ra = fan_outcome(&gpu, a);
            let rb = fan_outcome(&gpu, b);
            let picked: (Scalar, Scalar) = ($field(ra), $field(rb));
            assert!(
                rel(picked.0, picked.1) > 1e-6,
                "SPEC-LIT S13.4.1: two cases differing only in {} both produced \
                 {} = {} - the solver IGNORED the setting",
                $what,
                stringify!($field),
                picked.0
            );
        }
    };
}

pair!(
    pair_test_fan_dp_max,
    |r: (Scalar, Scalar, Scalar)| r.0,
    |f: &mut FanPatch| f.curve.dp_max = 90.0,
    "the fan curve's dpMax"
);
pair!(
    pair_test_fan_q_max,
    |r: (Scalar, Scalar, Scalar)| r.0,
    |f: &mut FanPatch| f.curve.q_max = 0.08,
    "the fan curve's QMax"
);
pair!(
    pair_test_ambient_pressure,
    |r: (Scalar, Scalar, Scalar)| r.0,
    |f: &mut FanPatch| f.ambient = 25.0,
    "ambientPressure"
);
pair!(
    pair_test_rho_curve,
    |r: (Scalar, Scalar, Scalar)| r.0,
    |f: &mut FanPatch| f.curve.rho_curve = 0.8,
    "rhoCurve, the (S52.13) density correction"
);
pair!(
    pair_test_speed,
    |r: (Scalar, Scalar, Scalar)| r.0,
    |f: &mut FanPatch| f.curve.n_speed = 1.4,
    "the shaft speed, the (S52.13) affinity correction"
);
pair!(
    pair_test_efficiency,
    |r: (Scalar, Scalar, Scalar)| r.2,
    |f: &mut FanPatch| f.curve.efficiency = 0.55,
    "efficiency, which divides the reported shaft power"
);
pair!(
    pair_test_curve_type,
    |r: (Scalar, Scalar, Scalar)| r.0,
    |f: &mut FanPatch| {
        f.curve = FanCurve::table(vec![(0.0, 60.0), (0.025, 45.0), (0.05, 0.0)]);
    },
    "the curve TYPE (quadratic against a table through the same endpoints)"
);

/// Moving ONE point of a tabulated curve must move the operating point and
/// the slope. The other pair tests change a scalar parameter; this one
/// changes a single number inside the table, which is the entry a
/// manufacturer's data sheet actually consists of.
#[test]
fn pair_test_one_table_point_moves_the_operating_point() {
    let Some(gpu) = gpu() else { return };
    let table = |mid: Scalar| {
        let mut f = base_fan();
        f.curve = FanCurve::table(vec![(0.0, 60.0), (0.025, mid), (0.05, 0.0)]);
        f
    };
    let a = table(45.0);
    let b = table(20.0);
    assert_ne!(a, b, "the two cases must differ in exactly one table entry");
    // The endpoints are identical, so anything that reads only them cannot
    // tell the two apart - which is what makes this the sharp version.
    assert_eq!(a.curve.points[0], b.curve.points[0]);
    assert_eq!(a.curve.points[2], b.curve.points[2]);

    let (qa, _, _) = fan_outcome(&gpu, a);
    let (qb, _, _) = fan_outcome(&gpu, b);
    assert!(
        rel(qa, qb) > 1e-6,
        "SPEC-LIT S13.4.1: two curves differing in ONE interior table point both          delivered {qa} - the interpolant read only the endpoints"
    );
}

/// `direction` must flip the **sign** of the patch flow, not just its size.
#[test]
fn pair_test_direction_flips_the_sign_of_the_flow() {
    let Some(gpu) = gpu() else { return };
    let mut a = base_fan();
    a.direction = FanDirection::Outflow;
    let mut b = base_fan();
    b.direction = FanDirection::Inflow;
    let (qa, _, _) = fan_outcome(&gpu, a);
    let (qb, _, _) = fan_outcome(&gpu, b);
    assert!(
        qa > 0.0 && qb < 0.0,
        "SPEC-LIT S13.4.1: `direction` must reverse the flow, not merely change it: \
         outflow gave {qa}, inflow gave {qb}"
    );
    assert!(
        rel(qa, -qb) < 1e-6,
        "and by symmetry of this rig the two should be mirror images: {qa}, {qb}"
    );
}

/// `fanRelaxation` must change the iterate HISTORY. The converged answer is
/// the same fixed point, which is why the history is what is compared.
#[test]
fn pair_test_fan_relaxation_changes_the_iterate_history() {
    let Some(gpu) = gpu() else { return };
    let history = |alpha: Scalar| -> Vec<Scalar> {
        let mut fan = base_fan();
        fan.relaxation = alpha;
        let (mut rig, mut fd) = fan_rig(&gpu, 12, fan, 0.02);
        rig.set_dirichlet(&gpu, "xmin", 0.0).expect("in");
        let mut h = Vec::new();
        for _ in 0..8 {
            fd.update(
                &gpu,
                &rig.m,
                &rig.phi,
                &mut rig.phi_hbya,
                &mut rig.rauf,
                &mut rig.rauf_mag_sf,
                &mut rig.p,
            )
            .expect("update");
            rig.assemble_and_solve(&gpu).expect("solve");
            let p = gpu.download(&rig.p.f).expect("p");
            let pb = gpu.download(&rig.p.bf).expect("pb");
            let g = gpu.download(&rig.rauf_mag_sf.bf).expect("g");
            let mut phi = gpu.download(&rig.phi.bf).expect("phi");
            for bf in 0..rig.hm.n_boundary_faces {
                let c = rig.hm.b_face_cells[bf] as usize;
                phi[bf] = -g[bf] * rig.hm.b_delta_coeffs[bf] * (pb[bf] - p[c]);
            }
            gpu.write(&mut rig.phi.bf, &phi).expect("phi");
            h.push(fd.states(&gpu).expect("st")[0].q_star);
        }
        h
    };
    let a = history(0.25);
    let b = history(1.0);
    let worst = a
        .iter()
        .zip(&b)
        .fold(0.0 as Scalar, |m, (x, y)| m.max(rel(*x, *y)));
    assert!(
        worst > 1e-6,
        "SPEC-LIT S13.4.1: two cases differing only in fanRelaxation produced the \
         same operating-point history - (S52.14) was ignored"
    );
}

/// Every porous-jump coefficient must move the face flux.
#[test]
fn pair_test_every_jump_coefficient_moves_the_flux() {
    let Some(gpu) = gpu() else { return };
    let flux = |c: PorousJumpCoeffs| -> Scalar {
        let mut rig = Rig::new(&gpu, chain(10), 0.02).expect("rig");
        rig.set_dirichlet(&gpu, "xmin", 4.0).expect("in");
        rig.set_dirichlet(&gpu, "xmax", 0.0).expect("out");
        let phi: Vec<Scalar> = vec![0.02; rig.hm.n_internal_faces];
        gpu.write(&mut rig.phi.f, &phi).expect("phi");
        let jumps = [PorousJump::Internal { faces: vec![4], coeffs: c }];
        let mut fd =
            FlowDevices::new(&gpu, &rig.hm, Vec::new(), &jumps, 1.2).expect("devices");
        fd.update(
            &gpu,
            &rig.m,
            &rig.phi,
            &mut rig.phi_hbya,
            &mut rig.rauf,
            &mut rig.rauf_mag_sf,
            &mut rig.p,
        )
        .expect("update");
        rig.assemble_and_solve(&gpu).expect("solve");
        rig.internal_flux(&gpu).expect("flux")[4]
    };

    let nu = 1.5e-5 as Scalar;
    let base = PorousJumpCoeffs::from_darcy_forchheimer(1e-7, 20.0, 0.02, nu).expect("df");
    let q0 = flux(base);

    for (what, c) in [
        ("K", PorousJumpCoeffs::from_loss_coefficient(30.0).expect("K")),
        (
            "alpha",
            PorousJumpCoeffs::from_darcy_forchheimer(2e-7, 20.0, 0.02, nu).expect("df"),
        ),
        (
            "C2",
            PorousJumpCoeffs::from_darcy_forchheimer(1e-7, 40.0, 0.02, nu).expect("df"),
        ),
        (
            "thickness",
            PorousJumpCoeffs::from_darcy_forchheimer(1e-7, 20.0, 0.04, nu).expect("df"),
        ),
        ("openAreaRatio", PorousJumpCoeffs::from_open_area_ratio(0.25).expect("sigma")),
    ] {
        let q = flux(c);
        assert!(
            rel(q, q0) > 1e-6,
            "SPEC-LIT S13.4.1: changing only `{what}` left the face flux at {q0} - \
             the solver IGNORED it"
        );
    }
}

/// The plenum pressure of a boundary jump must move the face flux.
#[test]
fn pair_test_plenum_pressure_moves_the_flux() {
    let Some(gpu) = gpu() else { return };
    let flux = |plenum: Scalar| -> Scalar {
        let mut rig = Rig::new(&gpu, chain(10), 0.02).expect("rig");
        rig.set_dirichlet(&gpu, "xmin", 4.0).expect("in");
        let mut kind = gpu.download(&rig.p.bc_kind).expect("kind");
        let pi = span(&rig.hm, "xmax");
        for bf in pi.0..pi.0 + pi.1 {
            kind[bf] = BcKind::PorousJumpPressure as Label;
        }
        gpu.write(&mut rig.p.bc_kind, &kind).expect("kind");

        let jumps = [PorousJump::Boundary {
            patch: "xmax".to_string(),
            coeffs: PorousJumpCoeffs::from_loss_coefficient(20.0).expect("K"),
            plenum,
        }];
        let mut fd =
            FlowDevices::new(&gpu, &rig.hm, Vec::new(), &jumps, 1.2).expect("devices");
        let phi: Vec<Scalar> = vec![0.02; rig.hm.n_boundary_faces];
        gpu.write(&mut rig.phi.bf, &phi).expect("phi");
        fd.update(
            &gpu,
            &rig.m,
            &rig.phi,
            &mut rig.phi_hbya,
            &mut rig.rauf,
            &mut rig.rauf_mag_sf,
            &mut rig.p,
        )
        .expect("update");
        rig.assemble_and_solve(&gpu).expect("solve");
        let p = gpu.download(&rig.p.f).expect("p");
        let pb = gpu.download(&rig.p.bf).expect("pb");
        let g = gpu.download(&rig.rauf_mag_sf.bf).expect("g");
        (pi.0..pi.0 + pi.1)
            .map(|bf| {
                let c = rig.hm.b_face_cells[bf] as usize;
                -g[bf] * rig.hm.b_delta_coeffs[bf] * (pb[bf] - p[c])
            })
            .sum()
    };
    let a = flux(0.0);
    let b = flux(-2.0);
    assert!(
        rel(a, b) > 1e-6,
        "SPEC-LIT S13.4.1: two cases differing only in plenumPressure both \
         delivered {a}"
    );
}

/// §53.3: `R = 0` on a boundary jump gives `fr = 1.0` exactly - a plain
/// `fixedValue` at the plenum pressure, bitwise.
#[test]
fn a_zero_resistance_boundary_jump_is_a_fixed_value_bitwise() {
    let Some(gpu) = gpu() else { return };
    let mut rig = Rig::new(&gpu, chain(8), 0.02).expect("rig");
    let jumps = [PorousJump::Boundary {
        patch: "xmax".to_string(),
        coeffs: PorousJumpCoeffs::default(),
        plenum: -1.25,
    }];
    let mut fd = FlowDevices::new(&gpu, &rig.hm, Vec::new(), &jumps, 1.2).expect("devices");
    fd.update(
        &gpu,
        &rig.m,
        &rig.phi,
        &mut rig.phi_hbya,
        &mut rig.rauf,
        &mut rig.rauf_mag_sf,
        &mut rig.p,
    )
    .expect("update");
    let fr = gpu.download(&rig.p.fr).expect("fr");
    let rv = gpu.download(&rig.p.ref_value).expect("rv");
    let pi = span(&rig.hm, "xmax");
    for bf in pi.0..pi.0 + pi.1 {
        assert_eq!(fr[bf], 1.0, "R = 0 must give fr exactly 1.0");
        assert_eq!(rv[bf], -1.25);
    }
}

// ==========================================================================
//  The five defects the shipped case found, each now a gate
// ==========================================================================

/// SPEC-LIT §52.1: the quadratic must be **odd** in `Q`, or a reversed fan
/// pushes harder the more it is pushed back.
#[test]
fn the_quadratic_curve_is_odd_in_q_so_s_never_goes_negative() {
    let c = FanCurve::quadratic(20.0, 3.0);

    // Forward branch: identical to the textbook even form, which is what
    // keeps every gate written against it valid.
    for q in [0.0 as Scalar, 0.5, 1.5, 3.0, 7.0] {
        let (dp, _) = c.at(q);
        assert!(
            rel(dp, 20.0 * (1.0 - (q / 3.0) * (q / 3.0))) < 1e-14,
            "the forward branch must be unchanged at Q = {q}"
        );
    }

    // The whole line: S >= 0 everywhere, and dp monotonically decreasing.
    let mut prev = c.at(-40.0).0;
    let mut worst_s = Scalar::INFINITY;
    for i in 1..=8000 {
        let q = -40.0 + 80.0 * i as Scalar / 8000.0;
        let (dp, s) = c.at(q);
        assert!(
            dp <= prev + 1e-9,
            "the curve rose from {prev} to {dp} at Q = {q} - that is the even \
             form's positive feedback loop coming back (SPEC-LIT S52.1)"
        );
        prev = dp;
        worst_s = worst_s.min(s);
    }
    assert!(worst_s >= 0.0, "S went to {worst_s} somewhere on the line");

    // And the even form really does fail this, so the test is measuring
    // something: at Q = -1 it gives S = -2 dpMax Q/QMax^2 = +13.3 > 0, i.e.
    // dp RISING with Q.
    let even_slope = -2.0 * 20.0 * (-1.0) / 9.0;
    assert!(even_slope > 0.0, "the even form's dp' at Q = -1 is {even_slope} > 0");
    assert!(c.at(-1.0).1 > 0.0, "the odd form's S at Q = -1 must be positive");
}

/// SPEC-LIT §52.5: both tails have `S > 0` and growing. The first draft held
/// the value below `Q_min`, which is `S = 0` - a `fixedValue` at shut-off.
#[test]
fn both_curve_tails_oppose_an_excursion() {
    let c = FanCurve::table(vec![(0.5, 900.0), (1.0, 700.0), (2.0, 100.0)]);
    c.validate("t").expect("legal");

    // Below the first point dp RISES and S stays positive and grows.
    let (dp_a, s_a) = c.at(0.4);
    let (dp_b, s_b) = c.at(-2.0);
    assert!(dp_a > 900.0 && dp_b > dp_a, "dp must rise below shut-off: {dp_a}, {dp_b}");
    assert!(s_b > s_a && s_a > 0.0, "S must grow going down: {s_a} then {s_b}");

    // Above the last point dp falls and S grows.
    let (dp_c, s_c) = c.at(2.5);
    let (dp_d, s_d) = c.at(8.0);
    assert!(dp_d < dp_c && dp_c < 100.0);
    assert!(s_d > s_c && s_c > 0.0);

    // Both joins are continuous in value and slope. The value tolerance is
    // the OFFSET times the slope - stepping 1e-9 either side of a join whose
    // slope is 600 Pa per m^3/s moves `dp` by 1.2e-6, and a tolerance tighter
    // than that would be testing arithmetic rather than continuity.
    for q in [0.5 as Scalar, 2.0] {
        let h = 1e-9 as Scalar;
        let lo = c.at(q - h);
        let hi = c.at(q + h);
        let slope = lo.1.abs().max(hi.1.abs());
        assert!(
            (lo.0 - hi.0).abs() <= 4.0 * h * slope,
            "dp jumps at Q = {q}: {} vs {}",
            lo.0,
            hi.0
        );
        assert!(rel(lo.1, hi.1) < 1e-4, "S jumps at Q = {q}: {} vs {}", lo.1, hi.1);
    }
}

/// SPEC-LIT §52.6: the first update linearises about free delivery when there
/// is no flux, and about the measured `Q` when there is.
#[test]
fn the_first_operating_point_is_free_delivery_when_there_is_no_flux() {
    let Some(gpu) = gpu() else { return };
    let q_max = 0.05 as Scalar;
    let mut fan = FanPatch::new("xmax", FanCurve::quadratic(60.0, q_max), FanDirection::Outflow);
    fan.ambient = 0.0;
    assert!(rel(fan.curve.free_delivery(), q_max) < 1e-15);

    // (a) no flux: Q* is free delivery, so dp is ZERO on the first update -
    //     the softest possible start.
    let (mut rig, mut fd) = fan_rig(&gpu, 8, fan.clone(), 0.02);
    fd.update(
        &gpu, &rig.m, &rig.phi, &mut rig.phi_hbya, &mut rig.rauf, &mut rig.rauf_mag_sf,
        &mut rig.p,
    )
    .expect("update");
    let st = fd.states(&gpu).expect("states");
    assert!(rel(st[0].q_star, q_max) < 1e-13, "Q* = {}, want {q_max}", st[0].q_star);
    assert!(
        st[0].dp.abs() < 1e-9,
        "at free delivery the curve delivers nothing, which is why it is the seed: \
         dp = {}",
        st[0].dp
    );

    // (b) a flux exists: Q* is the measured Q, not the seed.
    let (mut rig, mut fd) = fan_rig(&gpu, 8, fan, 0.02);
    let pi = span(&rig.hm, "xmax");
    let mut phi = vec![0.0 as Scalar; rig.hm.n_boundary_faces];
    for bf in pi.0..pi.0 + pi.1 {
        phi[bf] = 0.003;
    }
    gpu.write(&mut rig.phi.bf, &phi).expect("phi");
    fd.update(
        &gpu, &rig.m, &rig.phi, &mut rig.phi_hbya, &mut rig.rauf, &mut rig.rauf_mag_sf,
        &mut rig.p,
    )
    .expect("update");
    let st = fd.states(&gpu).expect("states");
    let q: Scalar = 0.003 * pi.1 as Scalar;
    assert!(rel(st[0].q_star, q) < 1e-13, "Q* = {}, want the measured {q}", st[0].q_star);
}

/// Free delivery of each curve kind, against its own definition.
#[test]
fn free_delivery_is_where_the_curve_first_reaches_zero() {
    assert_eq!(FanCurve::flat(300.0).free_delivery(), 0.0);
    assert!(rel(FanCurve::quadratic(10.0, 2.5).free_delivery(), 2.5) < 1e-15);

    // A table ending at zero: its last point.
    let t = FanCurve::table(vec![(0.0, 90.0), (1.0, 40.0), (2.0, 0.0)]);
    assert!(rel(t.free_delivery(), 2.0) < 1e-12);
    // A table crossing zero between points: the linear crossing.
    let t = FanCurve::table(vec![(0.0, 90.0), (1.0, 30.0), (2.0, -30.0)]);
    assert!(rel(t.free_delivery(), 1.5) < 1e-12, "{}", t.free_delivery());
    // A table that never reaches zero: its last point, as far as the case
    // was willing to describe.
    let t = FanCurve::table(vec![(0.0, 90.0), (1.0, 40.0), (2.0, 10.0)]);
    assert!(rel(t.free_delivery(), 2.0) < 1e-12);
    // And it scales with the shaft speed, like every other flow does.
    let mut t = FanCurve::quadratic(10.0, 2.5);
    t.n_speed = 2.0;
    assert!(rel(t.free_delivery(), 5.0) < 1e-14);
}

/// SPEC-LIT §52.4/§53.3: a fan or jump patch ALWAYS pins the pressure level,
/// and the seed has to say so before `crate::fan` has run.
#[test]
fn a_fan_or_jump_patch_is_seeded_as_a_dirichlet_because_it_pins_the_level() {
    let Some(gpu) = gpu() else { return };
    let hm = chain(6);
    let m = crate::mesh::GpuMesh::upload(&gpu, &hm).expect("upload");

    // The property that forces the seed: `fr` is in (0, 1] for every finite
    // curve slope and every finite resistance, so the patch is never a pure
    // Neumann face and the Poisson operator is never singular because of it.
    for s in [0.0 as Scalar, 1e-9, 1.0, 1e6, 1e300] {
        let fr = 1.0 / (1.0 + s * 3.7);
        assert!(fr > 0.0 && fr <= 1.0, "fr = {fr} at S = {s}");
    }
    for r in [0.0 as Scalar, 1e-9, 1.0, 1e6, 1e300] {
        let fr = 1.0 / (1.0 + r * 0.04);
        assert!(fr > 0.0 && fr <= 1.0, "fr = {fr} at R = {r}");
    }

    // And a `Simple` built with those kinds must NOT decide the pressure is
    // unpinned - which is what a zero seed would have made it do, and what
    // made `ofgpu-datacentre` diverge.
    let mut simple = crate::simple::Simple::new(
        &gpu,
        &hm,
        &m,
        crate::simple::SimpleControls::default(),
        crate::momentum::BuoyancyCoeffs { g: crate::Vec3::ZERO, ..Default::default() },
    )
    .expect("simple");
    {
        let p = simple.p_mut();
        let mut kind = vec![BcKind::ZeroGradient as Label; hm.n_boundary_faces];
        let mut fr = vec![0.0 as Scalar; hm.n_boundary_faces];
        let pi = span(&hm, "xmax");
        for bf in pi.0..pi.0 + pi.1 {
            kind[bf] = BcKind::FanPressure as Label;
            fr[bf] = 1.0;
        }
        gpu.write(&mut p.bc_kind, &kind).expect("kind");
        gpu.write(&mut p.fr, &fr).expect("fr");
    }
    simple.initialise(&gpu).expect("initialise");
    assert!(
        !simple.pressure_is_pinned(),
        "SPEC-LIT S52.4: a fan patch pins the pressure level, so `Simple` must NOT \
         pin a reference cell as well - `fix_pressure_level` would subtract that \
         cell after every solve and fight the absolute pressure the curve imposes"
    );
}

/// SPEC-LIT §53.3: the boundary jump divides ONLY `phi_HbyA`. Dividing the
/// coefficient too applies the resistance twice.
#[test]
fn the_boundary_jump_does_not_scale_the_coefficient_it_measured_d_from() {
    let Some(gpu) = gpu() else { return };
    let mut rig = Rig::new(&gpu, chain(8), 0.02).expect("rig");
    let pi = span(&rig.hm, "xmax");

    let g0 = gpu.download(&rig.rauf_mag_sf.bf).expect("g");
    let r0 = gpu.download(&rig.rauf.bf).expect("rauf");
    let ph: Vec<Scalar> = (0..rig.hm.n_boundary_faces).map(|i| 0.002 * (i as Scalar + 1.0)).collect();
    gpu.write(&mut rig.phi_hbya.bf, &ph).expect("phiHbyA");
    let phi: Vec<Scalar> = vec![0.05; rig.hm.n_boundary_faces];
    gpu.write(&mut rig.phi.bf, &phi).expect("phi");

    let coeffs = PorousJumpCoeffs::from_loss_coefficient(500.0).expect("K");
    let jumps = [PorousJump::Boundary {
        patch: "xmax".to_string(),
        coeffs,
        plenum: 1.5,
    }];
    let mut fd = FlowDevices::new(&gpu, &rig.hm, Vec::new(), &jumps, 1.2).expect("devices");
    fd.update(
        &gpu, &rig.m, &rig.phi, &mut rig.phi_hbya, &mut rig.rauf, &mut rig.rauf_mag_sf,
        &mut rig.p,
    )
    .expect("update");

    let g1 = gpu.download(&rig.rauf_mag_sf.bf).expect("g");
    let r1 = gpu.download(&rig.rauf.bf).expect("rauf");
    let ph1 = gpu.download(&rig.phi_hbya.bf).expect("phiHbyA");
    let fr = gpu.download(&rig.p.fr).expect("fr");

    assert_eq!(
        g0, g1,
        "SPEC-LIT S53.3: `bGammaMagSf` must come out BITWISE unchanged - `fr` \
         already carries the resistance into the assembly, and dividing the \
         coefficient as well applies 1/(1 + R D) twice"
    );
    assert_eq!(r0, r1, "and so must `rAU_b` - the flux corrector gets `fr` too");

    // `phi_HbyA` IS divided, by exactly 1/(1 + R D) = fr.
    for bf in pi.0..pi.0 + pi.1 {
        assert!(
            rel(ph1[bf], ph[bf] * fr[bf]) < 1e-13,
            "phi_HbyA at face {bf}: {} against {}",
            ph1[bf],
            ph[bf] * fr[bf]
        );
        // And `fr` is 1/(1 + R D) with D from the UNSCALED coefficient.
        let d = g0[bf] * rig.hm.b_delta_coeffs[bf];
        let r = coeffs.resistance(phi[bf], rig.hm.b_mag_sf[bf]);
        assert!(rel(fr[bf], 1.0 / (1.0 + r * d)) < 1e-13, "fr at {bf}");
    }
    // Faces off the patch are untouched.
    for bf in 0..pi.0 {
        assert_eq!(ph1[bf], ph[bf], "face {bf} is not a jump face");
    }
}

/// §53.8 Gate 53-A, the BOUNDARY form: the same series law with the plenum on
/// the far side. This is the gate that caught the double application.
#[test]
fn gate_53a_the_boundary_jump_puts_resistances_in_series() {
    let Some(gpu) = gpu() else { return };
    let n = 12;
    let rau = 0.017 as Scalar;
    let (p_in, p_plenum) = (5.0 as Scalar, 0.0 as Scalar);

    // The chain's resistance from `xmin` to the `xmax` FACE, i.e. everything
    // except the jump itself.
    let hm = chain(n);
    let mut base: Scalar = 0.0;
    for f in 0..hm.n_internal_faces {
        base += 1.0 / (rau * hm.mag_sf[f] * hm.delta_coeffs[f]);
    }
    for name in ["xmin", "xmax"] {
        let pi = span(&hm, name);
        for bf in pi.0..pi.0 + pi.1 {
            base += 1.0 / (rau * hm.b_mag_sf[bf] * hm.b_delta_coeffs[bf]);
        }
    }

    for r_jump in [0.0 as Scalar, 0.4 * base, 6.0 * base] {
        let mut rig = Rig::new(&gpu, chain(n), rau).expect("rig");
        rig.set_dirichlet(&gpu, "xmin", p_in).expect("in");
        let pi = span(&rig.hm, "xmax");
        let mut kind = gpu.download(&rig.p.bc_kind).expect("kind");
        for bf in pi.0..pi.0 + pi.1 {
            kind[bf] = BcKind::PorousJumpPressure as Label;
        }
        gpu.write(&mut rig.p.bc_kind, &kind).expect("kind");

        // A purely VISCOUS jump, so `R` does not depend on the flux and the
        // series law is a closed form rather than a fixed point. `R` is
        // per-face; the patch's faces are in parallel, so the patch-level
        // resistance is `R/n_faces`.
        let area = rig.hm.b_mag_sf[pi.0];
        let coeffs = PorousJumpCoeffs {
            r_visc: r_jump * pi.1 as Scalar * area,
            r_inert: 0.0,
        };
        let jumps = [PorousJump::Boundary {
            patch: "xmax".to_string(),
            coeffs,
            plenum: p_plenum,
        }];
        let mut fd =
            FlowDevices::new(&gpu, &rig.hm, Vec::new(), &jumps, 1.2).expect("devices");
        fd.update(
            &gpu, &rig.m, &rig.phi, &mut rig.phi_hbya, &mut rig.rauf, &mut rig.rauf_mag_sf,
            &mut rig.p,
        )
        .expect("update");
        rig.assemble_and_solve(&gpu).expect("solve");

        // The flux through the patch, from the same expression
        // `momCorrectFluxBoundary` evaluates.
        let p = gpu.download(&rig.p.f).expect("p");
        let pb = gpu.download(&rig.p.bf).expect("pb");
        let g = gpu.download(&rig.rauf_mag_sf.bf).expect("g");
        let q: Scalar = (pi.0..pi.0 + pi.1)
            .map(|bf| {
                let c = rig.hm.b_face_cells[bf] as usize;
                -g[bf] * rig.hm.b_delta_coeffs[bf] * (pb[bf] - p[c])
            })
            .sum();

        let want = (p_in - p_plenum) / (base + r_jump);
        assert!(
            rel(q, want) < 1e-9,
            "(S53.7) boundary form at R = {r_jump}: the patch carries {q}, series \
             says {want}. A factor of about (1 + R D) here is the double \
             application SPEC-LIT S53.3 records"
        );
    }
}

/// SPEC-LIT §52.10: `pressureInletOutletVelocity` on a fan patch carries ZERO
/// flux on inflow in this solver, which is why the driver uses
/// `zeroGradient`. The correction is a measurement, not an opinion.
#[test]
fn a_prescribed_velocity_face_carries_no_flux_whatever_the_pressure_says() {
    // `momFluxIsPrescribed(kind, fr)` is `kind == EMPTY || kind == SYMMETRY ||
    // fr >= 1`, and a prescribed face takes `phi = phi_HbyA` - so the
    // pressure's snGrad never reaches it.
    let src = std::fs::read_to_string(
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("cuda/momentum.cu"),
    )
    .expect("cuda/momentum.cu");
    assert!(
        src.contains("|| fr >= (ofscalar)1;"),
        "SPEC-LIT S52.10 rests on `momFluxIsPrescribed` treating `fr >= 1` as a \
         prescribed velocity. If that has changed, S52.10's correction needs \
         revisiting"
    );
    assert!(
        src.contains("phi[i] = momFluxIsPrescribed(bKind[i], fr[i])"),
        "and on `momCorrectFluxBoundary` asking it before applying snGrad(p)"
    );

    // And `field_setup` seeds kind 12 at `fr = 1` from the INTERIOR velocity,
    // once - there is no kernel that refreshes it from the flux, which is the
    // other half of why the note's claim does not hold here.
    let fs = std::fs::read_to_string(
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src/field_setup.rs"),
    )
    .expect("field_setup.rs");
    let at = fs
        .find("BcKind::PressureInletOutletVelocity => {")
        .expect("the kind 12 arm");
    let arm = &fs[at..at + 400];
    assert!(
        arm.contains("(1.0, n_hat * u.dot(n_hat), zero)"),
        "kind 12 is seeded as a Dirichlet at a fixed normal component: {arm}"
    );
}

// ==========================================================================
//  Setup-time refusals
// ==========================================================================

#[test]
fn a_fan_on_a_patch_the_mesh_does_not_have_is_refused_by_name() {
    let Some(gpu) = gpu() else { return };
    let hm = chain(6);
    let e = FlowDevices::new(
        &gpu,
        &hm,
        vec![FanPatch::new("crac", FanCurve::flat(1.0), FanDirection::Outflow)],
        &[],
        1.2,
    )
    .err()
    .expect("no such patch")
    .to_string();
    assert!(e.contains("crac") && e.contains("xmin"), "{e}");
}

#[test]
fn two_fans_on_one_patch_are_refused_by_name() {
    let Some(gpu) = gpu() else { return };
    let hm = chain(6);
    let e = FlowDevices::new(
        &gpu,
        &hm,
        vec![
            FanPatch::new("xmax", FanCurve::flat(1.0), FanDirection::Outflow),
            FanPatch::new("xmax", FanCurve::flat(2.0), FanDirection::Outflow),
        ],
        &[],
        1.2,
    )
    .err()
    .expect("two fans")
    .to_string();
    assert!(e.contains("stamp the"), "{e}");
}

#[test]
fn a_jump_on_a_face_that_is_not_internal_is_refused_by_name() {
    let Some(gpu) = gpu() else { return };
    let hm = chain(6);
    let jumps = [PorousJump::Internal {
        faces: vec![10_000],
        coeffs: PorousJumpCoeffs::default(),
    }];
    let e = FlowDevices::new(&gpu, &hm, Vec::new(), &jumps, 1.2)
        .err()
        .expect("bad face")
        .to_string();
    assert!(e.contains("internal face"), "{e}");

    // And a face named twice: the two resistances would not add.
    let jumps = [
        PorousJump::Internal { faces: vec![2], coeffs: PorousJumpCoeffs::default() },
        PorousJump::Internal { faces: vec![2], coeffs: PorousJumpCoeffs::default() },
    ];
    let e = FlowDevices::new(&gpu, &hm, Vec::new(), &jumps, 1.2)
        .err()
        .expect("duplicate")
        .to_string();
    assert!(e.contains("named by two jumps"), "{e}");
}

/// §53.6: the near-tile velocity caveat is produced, and only when there is
/// a jump to produce it for.
#[test]
fn the_jump_caveat_is_reported() {
    let Some(gpu) = gpu() else { return };
    let hm = chain(6);
    let fd = FlowDevices::new(&gpu, &hm, Vec::new(), &[], 1.2).expect("devices");
    assert!(fd.jump_caveat().is_none(), "no jump, no caveat");

    let jumps = [PorousJump::Internal {
        faces: vec![1, 2],
        coeffs: PorousJumpCoeffs::from_loss_coefficient(30.0).expect("K"),
    }];
    let fd = FlowDevices::new(&gpu, &hm, Vec::new(), &jumps, 1.2).expect("devices");
    let c = fd.jump_caveat().expect("a jump must produce the caveat");
    assert!(c.contains("FLOW RATE right"), "{c}");
    assert!(c.contains("VELOCITY FIELD wrong"), "{c}");
    assert!(c.contains("Abdelmaksoud"), "the caveat must cite its source: {c}");
    assert!(c.contains("Arghode"), "{c}");
}

#[test]
fn a_zero_reference_density_is_refused_by_name() {
    let Some(gpu) = gpu() else { return };
    let hm = chain(6);
    let e = FlowDevices::new(&gpu, &hm, Vec::new(), &[], 0.0)
        .err()
        .expect("rho_ref")
        .to_string();
    assert!(e.contains("kinematic"), "{e}");
}

// ----------------------------------------------------------------------
//  SPEC-LIT 81.7: the CUDA-graph capture gate
// ----------------------------------------------------------------------

/// SPEC-LIT 81: the fan/porous-jump face operator captures and replays
/// bitwise.
///
/// The operating point is under-relaxed from iteration to iteration, so this
/// also gates the thing most likely to have gone wrong: if the relaxation
/// were done on the host, capture would record one step of it and replay
/// that same step for ever, and only a bitwise comparison over several
/// replays would notice.
#[test]
fn the_fan_source_replays_bitwise() {
    let Some(gpu) = gpu() else { return };
    let faces: Vec<Label> = vec![2, 5, 7];
    let jumps = [PorousJump::Internal {
        faces: faces.clone(),
        coeffs: PorousJumpCoeffs::from_loss_coefficient(30.68).expect("K"),
    }];

    let report = crate::capture::capture_replays_bitwise(
        &gpu,
        "fan and porous jump (SPEC-LIT 52/53)",
        || {
            let mut rig = Rig::new(&gpu, chain(10), 0.019)?;
            rig.set_dirichlet(&gpu, "xmin", 2.0)?;
            let phi: Vec<Scalar> = vec![0.02; rig.hm.n_internal_faces];
            gpu.write(&mut rig.phi.f, &phi)?;
            let fd = FlowDevices::new(&gpu, &rig.hm, Vec::new(), &jumps, 1.2)?;
            Ok((fd, rig))
        },
        |(fd, rig): &mut (FlowDevices, Rig)| {
            fd.update(
                &gpu,
                &rig.m,
                &rig.phi,
                &mut rig.phi_hbya,
                &mut rig.rauf,
                &mut rig.rauf_mag_sf,
                &mut rig.p,
            )
        },
        |(_, rig): &(FlowDevices, Rig)| {
            Ok(vec![
                ("rAU_f", gpu.download(&rig.rauf.f)?),
                ("rAU_f|Sf|", gpu.download(&rig.rauf_mag_sf.f)?),
                ("phi_HbyA", gpu.download(&rig.phi_hbya.f)?),
                ("p", gpu.download(&rig.p.f)?),
            ])
        },
    )
    .expect("SPEC-LIT 81.7: the fan operator must capture and replay bitwise");
    println!("  fan / porous jump: {report}");
}
