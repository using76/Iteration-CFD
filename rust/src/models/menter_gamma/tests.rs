// meteor-cfd - Copyright (c) 2026 주식회사 메테오시뮬레이션 (Meteo Simulation Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.
//
// SPEC-LIT §90's "what must hold" table (§90.11), every row a host test can
// reach, each closed form checked against reference digits computed
// independently of this tree (Python, IEEE double) and against the limits
// the page prints. The device rows, the reduction, the plate and the pair
// tests are later units' work; nothing here is #[ignore]d for it.
// No GPL-licensed source was consulted.

#![allow(clippy::float_cmp)]
// The reference digits are quoted verbatim at 17 significant digits;
// clippy's shortest-round-trip respelling would obscure where each literal
// came from, so that lint stands down for this file alone.
#![allow(clippy::excessive_precision)]

use super::*;

/// Two doubles agree to the 1e-15 the reference digits are quoted to.
///
/// `exp` is the one operation these closed forms use whose bits IEEE does
/// not pin - `sqrt` and the four arithmetic operations are correctly
/// rounded, in any language - so the transcendental rows are asserted at
/// 1e-15 relative (about four ulp) and every rational row asserts bit
/// equality instead.
fn close(actual: Scalar, expected: Scalar) {
    let scale = expected.abs().max(actual.abs()).max(1e-300);
    let rel = (actual - expected).abs() / scale;
    assert!(
        rel <= 1e-15,
        "expected {expected} (0x{:016x}), got {actual} (0x{:016x}), \
         relative {rel:e}",
        expected.to_bits(),
        actual.to_bits()
    );
}

/// The default coefficients with `CPG3` moved - the pair §90.3's reading of
/// the page's orphan constant lives or dies by.
fn c_pg3(v: Scalar) -> GammaCoeffs {
    let mut c = GammaCoeffs::default();
    c.c_pg3 = v;
    c
}

/// `F_onset3 = max(1 - (R_T/3.5)^3, 0)` of (90.4), written directly.
///
/// The reference's digits are the formula's own rational arithmetic;
/// recovering the factor back out of [`f_onset`] - which `F_onset2` at its
/// cap of 2 would allow - costs the recovery a rounding that is not the
/// model's, so that consistency is asserted at 1e-15 in
/// `the_reference_f_turb_f_onset3_and_f_on_lim_rows_reproduce` instead.
fn f_onset3(r_t: Scalar) -> Scalar {
    (1.0 - (r_t / 3.5).powi(3)).max(0.0)
}

// ======================================================================
//  90.11  F_PG, and the reading of C_PG3
// ======================================================================

#[test]
fn the_reference_f_pg_rows_reproduce_bit_for_bit() {
    let c = GammaCoeffs::default();
    // The independent reference's 13 rows at C_PG3 = 0.00, digit for digit.
    // Every row is rational arithmetic - correctly rounded by IEEE - so the
    // assertion is bit equality and not a tolerance.
    let rows = [
        (0.0, 1.0),
        (0.02, 1.2936000000000001),
        (0.034059945504087197, 1.5),
        (0.1, 1.5),
        (1.0, 1.5),
        (2.0, 1.5),
        (-0.02, 1.1468),
        (-0.068099999999999994, 1.499854),
        (-0.1, 1.734),
        (-0.27250000000000002, 3.0),
        (-0.3, 3.0),
        (-1.0, 3.0),
        (-2.0, 3.0),
    ];
    for (lambda, f) in rows {
        assert_eq!(f_pg(lambda, &c), f, "lambda = {lambda}");
    }
}

#[test]
fn c_pg3_is_live_below_the_knot_and_inert_from_it_up() {
    let c0 = GammaCoeffs::default();
    let c1 = c_pg3(1.0);
    // At and above the knot min[lambda + 0.0681, 0] is exactly zero, so the
    // term C_PG3 carries cannot move F_PG - bitwise, which is the direction
    // §90.3's pair test fails in if the term sat on the wrong argument.
    for lambda in [-0.02, -0.068099999999999994, 0.0, 0.1, 1.0] {
        assert_eq!(f_pg(lambda, &c0), f_pg(lambda, &c1), "lambda = {lambda}");
    }
    // Below it, the reading moves F_PG - and by exactly the reference's
    // own numbers, not merely in sign.
    for (lambda, f) in [(-0.1, 1.7020999999999999), (-0.3, 2.9701)] {
        assert_eq!(f_pg(lambda, &c1), f, "lambda = {lambda}");
        assert!(f_pg(lambda, &c1) < f_pg(lambda, &c0));
    }
}

#[test]
fn the_reference_re_thetac_rows_reproduce() {
    let c = GammaCoeffs::default();
    // The reference's 11 rows. `exp` is not bit-pinned across languages, so
    // these carry the 1e-15 the digits were quoted to; the two rows 90.11
    // pins EXACTLY are asserted exactly below.
    let rows = [
        (0.0, 0.0, 1100.0),
        (0.5, 0.0, 706.53065971263345),
        (1.0, 0.0, 467.87944117144235),
        (2.244, 0.0, 206.03352089534809),
        (3.3, 0.0, 136.88316740124003),
        (3.3, 0.0128, 119.83944403395806),
        (5.788, 0.0, 103.06410426081986),
        (10.0, 0.0, 100.04539992976248),
        (1.0, 0.1, 323.13016014842981),
        (1.0, -0.1, 276.57668871613384),
    ];
    for (tu, lambda, re) in rows {
        close(re_thetac(tu, lambda, &c), re);
    }
    // Clean air is C_TU1 + C_TU2 - which is ReThetacLim to the digit, a
    // fact 90.3 states and nothing exploits. And at the Tu_L cap,
    // C_TU2 exp(-C_TU3 100) falls below the ulp of C_TU1 and the sum
    // rounds to C_TU1 exactly - 90.3's own explanation, not exp underflow.
    assert_eq!(re_thetac(0.0, 0.0, &c), 1100.0);
    assert_eq!(re_thetac(100.0, 0.0, &c), 100.0);
}

#[test]
fn f_pg_at_zero_is_exactly_one_and_re_thetac_two_constant_there() {
    let c = GammaCoeffs::default();
    // 90.11, first row: the positive branch before its cap - EXACTLY, the
    // property a pressure-gradient cap below 1 would deform (90.9).
    assert_eq!(f_pg(0.0, &c), 1.0);
    // 90.3: at lambda = 0, F_PG = 1 and Re_thetac is the two-constant form
    // C_TU1 + C_TU2 exp(-C_TU3 Tu_L) - the same exponent argument bit for
    // bit, so the identity is asserted bitwise.
    for tu in [0.0, 0.5, 1.0, 3.3, 10.0, 100.0] {
        assert_eq!(
            re_thetac(tu, 0.0, &c),
            c.c_tu1 + c.c_tu2 * (-(c.c_tu3 * tu)).exp(),
            "Tu_L = {tu}"
        );
    }
}

#[test]
fn the_positive_branch_reaches_its_cap_at_half_over_c_pg1() {
    let c = GammaCoeffs::default();
    // 90.11: F_PG reaches C_PG1lim where 1 + C_PG1 lambda = C_PG1lim, i.e.
    // lambda = 0.5/14.68 - the reference's third row, exactly 1.5 there.
    let knot = 0.5 / c.c_pg1;
    assert_eq!(f_pg(knot, &c), c.c_pg1_lim);
    assert!(f_pg(knot * (1.0 - 1e-12), &c) < c.c_pg1_lim);
    // and stays there to the end of the published range and past it.
    for lambda in [0.1, 0.5, 1.0, 2.0, 1.0e6] {
        assert_eq!(f_pg(lambda, &c), c.c_pg1_lim, "lambda = {lambda}");
    }
}

#[test]
fn the_negative_branch_reaches_its_cap_three() {
    let c = GammaCoeffs::default();
    // 90.11: the negative branch's cap is C_PG2lim = 3, reached where
    // 1 - 7.34 lambda = 3, i.e. lambda = -2/7.34. Either side of that
    // point, one part in 1e9, so no rounding question is being leaned on.
    let knot = 2.0 / -c.c_pg2;
    assert_eq!(f_pg(-knot * (1.0 + 1e-9), &c), 3.0);
    assert!(f_pg(-knot * (1.0 - 1e-9), &c) < c.c_pg2_lim);
    for lambda in [-0.27250000000000002, -0.3, -1.0, -2.0] {
        assert_eq!(f_pg(lambda, &c), 3.0, "lambda = {lambda}");
    }
}

#[test]
fn f_pg_is_never_negative_over_the_published_range() {
    // 90.11: F_PG >= 0 over a sweep of lambda_thL in [-1, 1] - the
    // published limiter max(F_PG, 0) of (90.8), with C_PG3 inert and live.
    let cs = [GammaCoeffs::default(), c_pg3(1.0)];
    let n = 100_001;
    for i in 0..=n {
        let lambda = -1.0 + 2.0 * (i as Scalar) / (n as Scalar);
        for c in &cs {
            assert!(f_pg(lambda, c) >= 0.0, "lambda = {lambda}");
        }
    }
}

#[test]
fn lambda_thl_is_clipped_at_both_ends() {
    // 90.11: lambda_thL clipped at +-1, both ends pinned - and just inside
    // the range the limiter is a no-op, not a quantiser.
    // both magnitudes far outside the limiter, one to each end
    for (dv, clipped) in [(1.0e10, -1.0), (-1.0e6, 1.0)] {
        assert_eq!(lambda_theta_l(dv, 1.0, 1.0), clipped);
    }
    let raw = lambda_theta_l_raw(100.0, 0.01, 1.5e-5);
    assert!(raw < -1.0);
    assert_eq!(lambda_theta_l(100.0, 0.01, 1.5e-5), -1.0);
    let inside = lambda_theta_l_raw(1.0, 0.01, 1.5e-5);
    assert_eq!(lambda_theta_l(1.0, 0.01, 1.5e-5), inside);
}

#[test]
fn re_thetac_is_monotone_decreasing_in_tu_l_and_bounded() {
    let c = GammaCoeffs::default();
    // 90.11: decreasing over [0, 100] wherever F_PG > 0, and bounded in
    // [C_TU1, C_TU1 + C_TU2] - the correlation's whole range.
    let n = 10_000;
    let mut prev = re_thetac(0.0, 0.0, &c);
    assert_eq!(prev, c.c_tu1 + c.c_tu2);
    for i in 1..=n {
        let tu = 100.0 * (i as Scalar) / (n as Scalar);
        let re = re_thetac(tu, 0.0, &c);
        assert!(re <= prev, "Tu_L = {tu}: {re} above {prev}");
        assert!(re >= c.c_tu1, "Tu_L = {tu}: {re} below C_TU1");
        assert!(re <= c.c_tu1 + c.c_tu2);
        // Strictly decreasing while the step still shows: with h = 0.01 the
        // step in C_TU2 exp(-Tu_L) falls to an ulp of C_TU1 near Tu_L ~ 29
        // (1000 e^-29 h ~ 1 ulp), and past there consecutive sums round to
        // the same double - 90.3's tail, not a plateau defect.
        if tu < 28.0 {
            assert!(re < prev, "Tu_L = {tu}: {re} not below {prev}");
        }
        prev = re;
    }
}

#[test]
fn f_onset_has_the_limits_the_page_prints() {
    // 90.11: F_onset2 <= 2, F_onset3 >= 0, F_onset >= 0, each pinned, and
    // F_onset = 0 at Re_V = 0 whatever R_T is doing.
    for r_t in [0.0, 0.5, 2.0, 3.0, 3.5, 10.0] {
        assert_eq!(f_onset(0.0, 1100.0, r_t), 0.0, "R_T = {r_t}");
    }
    // F_onset3 shuts OFF above R_T = 3.5 - 90.2's moved-up 40 % - which
    // with F_onset2 at its cap leaves F_onset = 2, the switch fully on.
    assert_eq!(f_onset3(0.0), 1.0);
    assert_eq!(f_onset3(3.5), 0.0);
    assert_eq!(f_onset3(7.0), 0.0);
    assert_eq!(f_onset(1.0e10, 1.0, 7.0), 2.0);
    for i in 0..=200 {
        let r_t = 10.0 * (i as Scalar) / 200.0;
        let fo3 = f_onset3(r_t);
        assert!((0.0..=1.0).contains(&fo3), "F_onset3({r_t}) = {fo3}");
        for j in 0..=200 {
            let re_v = 2.0e5 * (j as Scalar) / 200.0;
            let fo = f_onset(re_v, 137.0, r_t);
            assert!((0.0..=2.0).contains(&fo), "F_onset({re_v}, {r_t}) = {fo}");
        }
    }
}

#[test]
fn the_reference_f_turb_f_onset3_and_f_on_lim_rows_reproduce() {
    // F_turb's five rows: exp - 1e-15 - and its R_T = 2 row pinned again,
    // because 90.2 makes "F_turb = e^-1 at R_T = 2" the half-R_T statement.
    for (r_t, f) in [
        (0.0, 1.0),
        (1.0, 0.93941306281347581),
        (2.0, 0.36787944117144233),
        (3.0, 0.006329715427485747),
        (4.0, 1.1253517471925912e-07),
    ] {
        close(f_turb(r_t), f);
    }
    // F_onset3's four rows: rational arithmetic, bit-pinned. 90.11 also
    // pins F_on^lim's clip at 3 - its five rows, all rational.
    for (r_t, f) in [(0.0, 1.0), (1.0, 0.97667638483965014), (3.5, 0.0), (7.0, 0.0)] {
        assert_eq!(f_onset3(r_t), f, "R_T = {r_t}");
    }
    // And f_onset carries this factor: with F_onset2 at its cap,
    // F_onset = 2 - F_onset3, to within the RECOVERY'S rounding - two
    // subtractions near 2, whose ulp is 2.2e-16, so the bound here is a few
    // of those ABSOLUTE, not relative to F_onset3 itself.
    for i in 0..=100 {
        let r_t = 10.0 * (i as Scalar) / 100.0;
        let recovered = 2.0 - f_onset(1.0e10, 1.0, r_t);
        assert!(
            (recovered - f_onset3(r_t)).abs() <= 8.0 * Scalar::EPSILON,
            "R_T = {r_t}: recovered {recovered}"
        );
    }
    for (re_v, f) in [
        (0.0, 0.0),
        (2420.0, 0.0),
        (4840.0, 1.0),
        (9680.0, 3.0),
        (12100.0, 3.0),
    ] {
        assert_eq!(f_on_lim(re_v, 1100.0), f, "Re_V = {re_v}");
    }
}

#[test]
fn the_reference_tu_l_and_lambda_rows_reproduce_bit_for_bit() {
    // Tu_L's four rows: sqrt and the four operations are IEEE-pinned, so
    // bit equality - including both rows that sit ON the min(..., 100) cap,
    // one of them a wall cell, which is the guard's whole point.
    let rows = [
        (0.05, 50.0, 0.01, 36.514837167011073),
        (0.05, 50.0, 0.001, 100.0),
        (24.79, 6000.0, 0.001, 67.755005281829483),
        (1.0, 1.0, 1.0e-9, 100.0),
    ];
    for (k, omega, d, tu) in rows {
        assert_eq!(tu_l(k, omega, d), tu, "k = {k}, omega = {omega}, d = {d}");
    }
    // lambda_thL raw and clipped, all four rows.
    let lams = [
        (0.0, 0.012800000000000001, 0.012800000000000001),
        (100.0, -5.0338666666666665, -1.0),
        (-100.0, 5.0594666666666672, 1.0),
        (10000.0, -504.65386666666666, -1.0),
    ];
    for (dv, raw, clipped) in lams {
        assert_eq!(lambda_theta_l_raw(dv, 0.01, 1.5e-5), raw, "dV/dy = {dv}");
        assert_eq!(lambda_theta_l(dv, 0.01, 1.5e-5), clipped);
    }
}

#[test]
fn the_split_never_sinks_negative_and_reconstructs_the_source() {
    // 90.11, split row - 90.5's sweep: sp >= 0 at every state, over gamma
    // in [0, 1] including both ends (the absorbing state), and a, b in
    // {0, small, large}: production off, weak, and overwhelming.
    let ab = [(0.0, 0.0), (0.01, 0.06), (500.0, 25.0)];
    let ce2 = 50.0;
    for &(a, b) in &ab {
        for gi in 0..=20 {
            let gamma = (gi as Scalar) / 20.0;
            let (su, sp, susp) = gamma_source_split(gamma, a, b, ce2);
            assert_eq!(su, 0.0, "su at gamma = {gamma}");
            assert!(sp >= 0.0, "sp at gamma = {gamma}, A = {a}, B = {b}");
            assert_eq!(susp, -(a + b));
            // And the halves reconstruct the source. fvm_susp moves a
            // NON-POSITIVE coefficient to the right-hand side
            // (`source -= min(S, 0) psi`, SPEC-LIT 3.4), so the emitted net
            // source is su - susp gamma - sp gamma, and it must come back
            // as (90.12) to 1e-15.
            let emitted = su - susp * gamma - sp * gamma;
            let want = (a + b) * gamma - (a + b * ce2) * gamma * gamma;
            close(emitted, want);
        }
    }
}

// ======================================================================
//  Gate 90-G, host half
// ======================================================================

/// `grad(U)` by central differences - the stand-in for a Gauss gradient.
/// The velocity `U = (u0 + 0.25x + 1.5y, -0.25y, 2x)` is sampled at dyadic
/// points with a dyadic step, so the stencil is EXACT and the only thing a
/// Galilean boost can do to it is nothing. Row `i` is `dU_j/dx_i`,
/// `crate::types::Tensor`'s convention, the one `RasCore::grad_u` holds and
/// [`dv_dy`] reads.
fn grad_central(u0: Scalar, p: Vec3, h: Scalar) -> Tensor {
    let sample = |q: Vec3| Vec3::new(u0 + 0.25 * q.x + 1.5 * q.y, -0.25 * q.y, 2.0 * q.x);
    let axes = [Vec3::new(h, 0.0, 0.0), Vec3::new(0.0, h, 0.0), Vec3::new(0.0, 0.0, h)];
    let mut g = Tensor::ZERO;
    for (i, e) in axes.iter().enumerate() {
        let up = sample(p + *e);
        let um = sample(p - *e);
        let row = Vec3::new(
            (up.x - um.x) / (2.0 * h),
            (up.y - um.y) / (2.0 * h),
            (up.z - um.z) / (2.0 * h),
        );
        if i == 0 {
            g.xx = row.x;
            g.xy = row.y;
            g.xz = row.z;
        } else if i == 1 {
            g.yx = row.x;
            g.yy = row.y;
            g.yz = row.z;
        } else {
            g.zx = row.x;
            g.zy = row.y;
            g.zz = row.z;
        }
    }
    g
}

/// `S = sqrt(2 S_ij S_ij)` with `S_ij = (g_ij + g_ji)/2` (90.5, 90.2) - the
/// strain magnitude `Re_V` reads, built here rather than imported so the
/// gate's chain is visible: velocity field, gradient, closed form.
fn strain_mag(g: Tensor) -> Scalar {
    let sxy = 0.5 * (g.xy + g.yx);
    let sxz = 0.5 * (g.xz + g.zx);
    let syz = 0.5 * (g.yz + g.zy);
    let sum = g.xx * g.xx + g.yy * g.yy + g.zz * g.zz
        + 2.0 * (sxy * sxy + sxz * sxz + syz * syz);
    (2.0 * sum).sqrt()
}

#[test]
fn the_closed_forms_are_bitwise_galilean_invariant() {
    // Gate 90-G's closed-form half (90.7): 88.9's table - frame shifts of
    // +0.5, +1, +2, +5 m/s, the ones that moved LM2009's Re_theta_eq by
    // +7.9, +15.9, +31.9 and +82.4 % - run through these closed forms may
    // not move ONE BIT, pass/fail and not a tolerance, because no line of
    // (90.4)-(90.11) contains a velocity magnitude.
    let c = GammaCoeffs::default();
    let (k, omega, d, nu) = (0.05, 50.0, 0.01, 1.5e-5);
    let n = wall_normal(Vec3::new(0.0, 1.0, 0.0));
    let p = Vec3::new(1.0, 2.0, 0.5);
    type Row = (Tensor, Scalar, Scalar, Scalar, Scalar, Scalar, Scalar);
    let mut base: Option<Row> = None;
    for shift in [0.0, 0.5, 1.0, 2.0, 5.0] {
        let u0 = 5.0 + shift;
        let g = grad_central(u0, p, 0.25);
        let re_v = d * d * strain_mag(g) / nu;
        let r_t = k / (nu * omega);
        let tu = tu_l(k, omega, d);
        let lam = lambda_theta_l(dv_dy(n, g), d, nu);
        let rtc = re_thetac(tu, lam, &c);
        let got: Row = (g, tu, lam, f_pg(lam, &c), rtc, f_onset(re_v, rtc, r_t), re_v);
        match base {
            None => base = Some(got),
            Some(ref b) => assert_eq!(&got, b, "frame shift = {shift} m/s"),
        }
    }
    assert!(base.is_some());
}

#[test]
fn p_k_lim_switches_itself_off_and_is_never_negative() {
    let c = GammaCoeffs::default();
    // (90.15)'s factors are non-negative at a live state (90.6), and three
    // of its maxes make it switch itself off rather than being clamped:
    // below gamma = 0.2, at the fully-turbulent end, below the Re_V the
    // limiter switch needs, and in the log layer where nu_t passes
    // 3 C_sep nu.
    for gi in 0..=20 {
        let gamma = (gi as Scalar) / 20.0;
        for fol in [0.0, 0.5, 3.0] {
            for nu_t in [0.0, 3.0 * c.c_sep * 1.5e-5, 1.0e-3] {
                let p = p_k_lim(gamma, fol, 1.5e-5, nu_t, 100.0, 100.0, &c);
                assert!(p >= 0.0, "gamma = {gamma}, F_on^lim = {fol}, nu_t = {nu_t}");
            }
        }
    }
    for gamma in [0.0, 0.1, 0.2] {
        assert_eq!(p_k_lim(gamma, 3.0, 1.5e-5, 0.0, 100.0, 100.0, &c), 0.0);
    }
    assert_eq!(p_k_lim(1.0, 3.0, 1.5e-5, 0.0, 100.0, 100.0, &c), 0.0);
    assert_eq!(p_k_lim(0.5, 0.0, 1.5e-5, 0.0, 100.0, 100.0, &c), 0.0);
    assert_eq!(p_k_lim(0.5, 3.0, 1.5e-5, 1.0e-3, 100.0, 100.0, &c), 0.0);
    // and it is not identically zero where it is meant to live.
    assert!(p_k_lim(0.5, 3.0, 1.5e-5, 0.0, 100.0, 100.0, &c) > 0.0);
}

#[test]
fn f3_is_the_blending_floor_written_against_ry() {
    // (90.17): 88.6's F_3 to the character. 1 at the wall, e^-1 at
    // R_y = 120, monotone, and far below machine relevance by R_y = 240 -
    // which is why the argument carries the same clamp every exponential
    // here does.
    assert_eq!(f3(0.0), 1.0);
    close(f3(120.0), 0.36787944117144233);
    let mut prev = f3(0.0);
    for i in 1..=240 {
        let r_y = (i as Scalar) * (2.0 / 240.0) * 120.0;
        let f = f3(r_y);
        assert!(f <= prev, "R_y = {r_y}");
        assert!((0.0..=1.0).contains(&f));
        prev = f;
    }
    // exp(-256), far under any relevance, though representable - 90.3's
    // point about exp(-300) applies here too.
    assert!(f3(240.0) < 1.0e-100);
}

// ======================================================================
//  GammaCoeffs::check, and the banner
// ======================================================================

/// One refusal: a Config error whose message names the dictionary key.
fn refused(c: GammaCoeffs, key: &str) {
    let err = c.check().expect_err(key);
    assert!(matches!(err, Error::Config(_)), "{key}: {err}");
    let msg = err.to_string();
    assert!(msg.contains("momentumTransport/RAS/"), "{key}: {msg}");
    assert!(msg.contains(key), "{key}: {msg}");
}

#[test]
fn gamma_coeffs_check_refuses_each_impossible_set_by_name() {
    // 90.9's coefficient half, one line per refusal, each naming the key
    // the case wrote under momentumTransport/RAS.
    let mut c = GammaCoeffs::default();
    c.ce2 = 1.0;
    refused(c, "ce2");
    let mut c = GammaCoeffs::default();
    c.ce2 = 0.5;
    refused(c, "ce2");
    let mut c = GammaCoeffs::default();
    c.sigma_gamma = 0.0;
    refused(c, "sigmaGamma");
    let mut c = GammaCoeffs::default();
    c.f_length = -1.0;
    refused(c, "Flength");
    let mut c = GammaCoeffs::default();
    c.c_tu2 = -1.0;
    refused(c, "CTU2");
    let mut c = GammaCoeffs::default();
    c.re_thetac_lim = 0.0;
    refused(c, "ReThetacLim");
    let mut c = GammaCoeffs::default();
    c.gamma_min = 1.0;
    c.gamma_max = 0.0;
    refused(c, "gammaMin");
    let mut c = GammaCoeffs::default();
    c.c_pg1_lim = 0.5;
    refused(c, "CPG1lim");
    let mut c = GammaCoeffs::default();
    c.c_pg2_lim = 0.5;
    refused(c, "CPG2lim");
}

#[test]
fn the_defaults_pass_and_equal_bounds_are_a_freezing_not_a_defect() {
    // 90.9: gammaMin = gammaMax is deliberately NOT refused - it freezes
    // the intermittency, a real thing to ask for, and gammaMin = gammaMax =
    // 1 is Gate 90-R's fully-turbulent limit.
    GammaCoeffs::default().check().expect("the printed defaults hold");
    let mut frozen = GammaCoeffs::default();
    frozen.gamma_min = 1.0;
    frozen.gamma_max = 1.0;
    frozen.check().expect("gammaMin = gammaMax = 1 is the turbulent limit");
    let mut half = GammaCoeffs::default();
    half.gamma_min = 0.5;
    half.gamma_max = 0.5;
    half.check().expect("any equal pair is a frozen intermittency");
}

#[test]
fn the_banner_lists_every_constant_and_ends_with_the_gate() {
    let d = GammaCoeffs::default().describe();
    for key in [
        "Flength", "ce2", "ca2", "sigmaGamma", "CTU1", "CTU2", "CTU3", "CPG1", "CPG2", "CPG3",
        "CPG1lim", "CPG2lim", "ReThetacLim", "Ck", "Csep", "OURS:", "gamma in",
    ] {
        assert!(d.contains(key), "{key} missing from the banner: {d}");
    }
    // 90.7's promise, and the gate behind it.
    assert!(d.ends_with("Galilean invariant (Gate 90-G)"), "{d}");
    // The model name is the registry's business, not the banner's.
    assert!(!d.contains("kOmegaSSTGamma"), "{d}");
}

#[test]
fn the_controls_default_to_the_k_entries_the_case_falls_back_to() {
    // 91.1's fallback rule in its default form: before a registry exists to
    // read system/ with, gamma's solver settings ARE k's.
    let base = crate::io::case::TurbulenceControls::default();
    let g = GammaControls::default();
    assert_eq!(g.gamma_solver, base.k_solver);
    assert_eq!(g.gamma_relax, base.k_relax);
    assert_eq!(g.gamma_conv, base.k_conv());
}
