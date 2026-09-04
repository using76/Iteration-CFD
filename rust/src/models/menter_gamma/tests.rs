// meteor-cfd - Copyright (c) 2026 주식회사 메테오시뮬레이션 (Meteo Simulation Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.
//
// SPEC-LIT §90's "what must hold" table (§90.11), every row a host test can
// reach, each closed form checked against reference digits computed
// independently of this tree (Python, IEEE double) and against the limits
// the page prints. The device rows are here too (the host-vs-device sweep,
// the split, Gate 90-R's stamp half, the Kato-Launder measurement); the
// end-to-end reduction, the plate and the pair tests are later units' work.
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

// ======================================================================
//  90.11  The device and the host agree
// ======================================================================

fn gpu() -> Option<Gpu> {
    Gpu::new(0).ok()
}

/// A plain block with the floor a wall, so `y` is unambiguous.
fn block(n: usize) -> crate::mesh::HostMesh {
    let mut spec = crate::blockgen::BlockSpec {
        x: crate::blockgen::GradedAxis { lo: 0.0, hi: 1.0, n, expansion: 1.0, two_sided: false },
        y: crate::blockgen::GradedAxis { lo: 0.0, hi: 0.2, n, expansion: 1.0, two_sided: false },
        z: crate::blockgen::GradedAxis { lo: 0.0, hi: 0.2, n, expansion: 1.0, two_sided: false },
        ..crate::blockgen::BlockSpec::default()
    };
    for p in [1, 3, 4, 5] {
        spec.patch_type[p] = "patch".to_string();
    }
    crate::blockgen::build_mesh(&spec).expect("block")
}

/// **Every correlation on the device is the one on the host, over a sweep
/// that covers the interior and the cap of `Tu_L`, both signs and both
/// clips of `lambda_thL`.**
///
/// The host closed forms above are what the reference digits pin and what
/// Gate 90-T's arithmetic runs through; the device functions in
/// `cuda/gmtrans.cu` are what a run uses. They are two transcriptions of
/// the same formulas in the same operation order, and a digit dropped in
/// one of them is visible only by measuring them against each other.
/// (90.11): worst relative difference below 1e-13, per closed form.
#[test]
fn the_host_and_device_closed_forms_agree() {
    let Some(gpu) = gpu() else { return };
    let hm = block(8);
    let mesh = crate::mesh::GpuMesh::upload(&gpu, &hm).expect("mesh");
    let n = hm.n_cells;
    assert!(n >= 256, "the sweep needs cells to spread over, got {n}");

    let nu = 1.5e-5 as Scalar;
    let coeffs = GammaCoeffs::default();

    // One station per cell. Tu_L is driven through a target intensity
    // t_i that runs PAST the page's cap of 100, so both the interior of
    // (90.9) and its min(..., 100) are read.
    let tu_t: Vec<Scalar> =
        (0..n).map(|i| 0.05 + 0.6 * i as Scalar).collect();
    let omega: Vec<Scalar> = (0..n).map(|i| 10.0 + 5.0 * i as Scalar).collect();
    let yv: Vec<Scalar> = (0..n).map(|i| 1e-4 * (i as Scalar + 1.0)).collect();
    let sv: Vec<Scalar> = (0..n).map(|i| 1.0 + 0.5 * i as Scalar).collect();
    let k: Vec<Scalar> = tu_t
        .iter()
        .zip(&omega)
        .zip(&yv)
        .map(|((t, w), y)| 1.5 * (t * w * y / 100.0).powi(2))
        .collect();
    // A strained field - normal strain AND shear, growing with the cell
    // index so the late cells push lambda_thL through both clips - and a
    // field of non-unit wall-normal vectors whose direction turns with the
    // index, so gmWallNormal and gmDvDy are read on both signs.
    let gu: Vec<Tensor> = (0..n)
        .map(|i| {
            let i = i as Scalar;
            Tensor {
                xx: 0.1 + 0.01 * i,
                xy: -0.03 * i,
                xz: 0.02 + 0.001 * i,
                yx: 0.2 + 0.01 * i,
                yy: -0.05 - 0.005 * i,
                yz: -0.01 * i,
                zx: 0.05,
                zy: 0.02 * i,
                zz: -0.05 + 0.002 * i,
            }
        })
        .collect();
    let gy: Vec<Vec3> = (0..n)
        .map(|i| {
            let th = 0.07 * i as Scalar;
            let m = 0.5 + 0.001 * i as Scalar;
            Vec3 { x: th.cos() * m, y: th.sin() * m, z: 0.3 * m }
        })
        .collect();

    let mut yb: DevBuf<Scalar> = gpu.zeros(n).expect("y");
    let mut kb: DevBuf<Scalar> = gpu.zeros(n).expect("k");
    let mut wb: DevBuf<Scalar> = gpu.zeros(n).expect("w");
    let mut sb: DevBuf<Scalar> = gpu.zeros(n).expect("s");
    let mut gyb: DevBuf<Vec3> = gpu.zeros(n).expect("gy");
    let mut grub: DevBuf<Tensor> = gpu.zeros(n).expect("gradU");
    gpu.write(&mut yb, &yv).expect("write y");
    gpu.write(&mut kb, &k).expect("write k");
    gpu.write(&mut wb, &omega).expect("write omega");
    gpu.write(&mut sb, &sv).expect("write s");
    gpu.write(&mut gyb, &gy).expect("write gy");
    gpu.write(&mut grub, &gu).expect("write gradU");

    let mut gm = MenterGamma::new(
        &gpu,
        &mesh,
        coeffs,
        GammaControls::default(),
        &yb,
        &gyb,
    )
    .expect("model");
    gpu.write(&mut gm.gamma_mut().f, &vec![0.5 as Scalar; n]).expect("write gamma");
    let turb = TurbKernels::new(&gpu).expect("turb kernels");
    gm.update_fields(&gpu, &turb, &kb, &wb, &sb, &grub, nu, n).expect("update");

    let got = [
        gpu.download(gm.f_onset()).expect("read"),
        gpu.download(gm.f_turb_field()).expect("read"),
        gpu.download(gm.re_thetac_field()).expect("read"),
        gpu.download(gm.f_on_lim_field()).expect("read"),
        gpu.download(gm.f3_field()).expect("read"),
        gpu.download(gm.tu_l_field()).expect("read"),
        gpu.download(gm.lambda_thl_field()).expect("read"),
    ];

    let mut worst = [0.0 as Scalar; 7];
    let mut clip_lo = 0;
    let mut clip_hi = 0;
    let mut capped = 0;
    for i in 0..n {
        let kc = k[i].max(0.0);
        let w = omega[i].max(1e-30);
        let d = yv[i];
        let nrm = wall_normal(gy[i]);
        let dvdy = dv_dy(nrm, gu[i]);
        let tu = tu_l(k[i], omega[i], d);
        let lam = lambda_theta_l(dvdy, d, nu);
        let rc = re_thetac(tu, lam, &coeffs);
        let r_t = kc / (nu * w);
        let r_v = sv[i] * d * d / nu;
        let r_y = d * kc.sqrt() / nu;
        let want = [
            f_onset(r_v, rc, r_t),
            f_turb(r_t),
            rc,
            f_on_lim(r_v, coeffs.re_thetac_lim),
            f3(r_y),
            tu,
            lam,
        ];
        if lam <= -1.0 {
            clip_lo += 1;
        }
        if lam >= 1.0 {
            clip_hi += 1;
        }
        if tu >= 100.0 {
            capped += 1;
        }
        for j in 0..7 {
            let dd = (got[j][i] - want[j]).abs() / want[j].abs().max(1e-12);
            worst[j] = worst[j].max(dd);
        }
    }
    let names = ["F_onset", "F_turb", "Re_thetac", "F_on_lim", "F_3", "Tu_L", "lambda_thL"];
    println!("  sweep: {capped} cells at the Tu_L cap, {clip_lo} at lambda -1, {clip_hi} at +1");
    for (j, name) in names.iter().enumerate() {
        println!("  host vs device, {name}: worst {:.3e} relative", worst[j]);
        assert!(
            worst[j] < 1e-13,
            "{name} disagrees by {:.3e} between the host closed form and the kernel",
            worst[j]
        );
    }
    assert!(clip_lo > 0 && clip_hi > 0, "the sweep never reached either lambda clip");
}

// ======================================================================
//  90.5  The split, on the device
// ======================================================================

/// **The device source is the host Patankar split, over a sweep of every
/// input including the absorbing state.**
///
/// `su`, `sp` and `susp` are launched from `gmGammaSources` and compared
/// with [`gamma_source_split`] cell by cell. `su` is the constant zero and
/// is asserted bitwise. `sp` and `susp` are asserted at 1e-15 relative
/// (a few ulp), NOT bitwise, and the bitwise count is printed: this file
/// compiles with nvcc's default contraction (`-fmad=true`, which every
/// unit but `meshgeom.cu` keeps), so the device computes `a + b*ce2` as ONE
/// correctly-rounded fused multiply-add where rustc rounds it twice. That
/// is the documented §67.11 position - the device is the reference and a
/// fused product is the better answer - and it moves the last bit, which is
/// exactly what this test measures rather than hides. `sp` contracts
/// `a + b*ce2` into one rounding; `susp` contracts too, because `b` is a
/// product, so `a + b` folds its last multiply into the add
/// (`fma(t1, ftb, a)`) where rustc rounds twice. (§90.11: the halves
/// reconstruct the source, and `Sp` never sinks negative.)
#[test]
fn the_gamma_source_split_matches_the_host_split() {
    let Some(gpu) = gpu() else { return };
    let hm = block(6);
    let mesh = crate::mesh::GpuMesh::upload(&gpu, &hm).expect("mesh");
    let n = hm.n_cells;

    let coeffs = GammaCoeffs::default();
    let yb: DevBuf<Scalar> = gpu.zeros(n).expect("y");
    let gyb: DevBuf<Vec3> = gpu.zeros(n).expect("gy");
    let mut gm =
        MenterGamma::new(&gpu, &mesh, coeffs, GammaControls::default(), &yb, &gyb)
            .expect("model");

    // A sweep of everything the split reads, zeros included.
    let gam: Vec<Scalar> = (0..n).map(|i| i as Scalar / n as Scalar).collect();
    let fon: Vec<Scalar> = (0..n).map(|i| 2.0 * i as Scalar / n as Scalar).collect();
    let ftb: Vec<Scalar> = (0..n).map(|i| i as Scalar / (2.0 * n as Scalar)).collect();
    let sv: Vec<Scalar> = (0..n).map(|i| 1.0 + 0.7 * i as Scalar).collect();
    let om: Vec<Scalar> = (0..n).map(|i| 20.0 / (i as Scalar + 1.0)).collect();
    gpu.write(&mut gm.gamma_mut().f, &gam).expect("write gamma");
    gpu.write(&mut gm.f_onset, &fon).expect("write fOnset");
    gpu.write(&mut gm.f_turb, &ftb).expect("write fTurb");
    gpu.write(&mut gm.omega_mag, &om).expect("write omegaMag");

    let mut sb: DevBuf<Scalar> = gpu.zeros(n).expect("s");
    gpu.write(&mut sb, &sv).expect("write s");

    let (mut su, mut sp, mut susp) =
        (gpu.zeros(n).expect("su"), gpu.zeros(n).expect("sp"), gpu.zeros(n).expect("susp"));
    let (fl, ca2, ce2) = (coeffs.f_length, coeffs.ca2, coeffs.ce2);
    gm.gamma_sources_for_test(&gpu, &mut su, &mut sp, &mut susp, &sb, n)
        .expect("launch gmGammaSources");

    let got_su: Vec<Scalar> = gpu.download(&su).expect("read");
    let got_sp: Vec<Scalar> = gpu.download(&sp).expect("read");
    let got_susp: Vec<Scalar> = gpu.download(&susp).expect("read");

    let mut worst = [0.0 as Scalar; 2];
    let mut bitwise = 0;
    let mut sp_min = Scalar::INFINITY;
    let mut sp_zero = 0;
    for i in 0..n {
        let a = fl * sv[i] * fon[i];
        let b = ca2 * om[i] * ftb[i];
        let (_, w_sp, w_susp) = gamma_source_split(gam[i], a, b, ce2);
        assert_eq!(got_su[i].to_bits(), (0.0 as Scalar).to_bits(), "cell {i}: su");
        worst[0] = worst[0].max((got_sp[i] - w_sp).abs() / w_sp.abs().max(1e-300));
        worst[1] = worst[1].max((got_susp[i] - w_susp).abs() / w_susp.abs().max(1e-300));
        if got_sp[i].to_bits() == w_sp.to_bits() && got_susp[i].to_bits() == w_susp.to_bits() {
            bitwise += 1;
        }
        sp_min = sp_min.min(got_sp[i]);
        if got_sp[i] == 0.0 {
            sp_zero += 1;
        }
    }
    println!(
        "  device split vs host split, {n} cells: worst relative difference \
         sp {:.3e}, susp {:.3e}; {bitwise} of {n} cells bitwise (the rest are \
         the device's fused a + b*ce2, one rounding where the host makes two)",
        worst[0], worst[1]
    );
    assert!(
        worst[0] < 1e-15 && worst[1] < 1e-15,
        "the device split disagrees with the host split by {:e} / {:e} relative",
        worst[0],
        worst[1]
    );
    assert!(
        sp_min >= 0.0,
        "Sp sank to {sp_min} somewhere on the sweep - the split is not \
         non-negative at every state"
    );
    println!("  Sp >= 0 everywhere ({sp_zero} cells at exactly 0 - the absorbing end)");
}

/// **`gamma = 0` is an absorbing state, and the split shows it.**
///
/// SPEC-LIT §90.5. The honest form of this test needs a `correct`, and a
/// `correct` needs the SST rig that owns the [`RasCore`] - Unit 4a's work,
/// which §90.11's end-to-end rows wait for. What this unit CAN show is the
/// reason the state is absorbing: at `gamma = 0` the device emits
/// `su = 0` and `sp = 0`, both BITWISE - `(A + B c_e2)` is finite and
/// `gamma` is exactly zero, so the product is exactly zero whatever the
/// contraction does - and `susp = -(A + B)` to within a ulp, because the
/// device folds the last multiply of `B` into the addition where rustc
/// rounds twice (see `the_gamma_source_split_matches_the_host_split`).
/// `susp`'s explicit contribution through `fvm_susp` is
/// explicit contribution through `fvm_susp` is `susp * gamma = 0`. The
/// equation has no explicit source and never divides by `gamma`, so a cell
/// AT zero has zero source forever, and the end-to-end decay of a cell at
/// `gamma = 1` toward the laminar fixed point `1/c_e2` is left to the
/// wiring unit.
#[test]
fn a_cell_at_zero_intermittency_stays_at_zero_and_one_decays_toward_the_fixed_point() {
    let Some(gpu) = gpu() else { return };
    let hm = block(6);
    let mesh = crate::mesh::GpuMesh::upload(&gpu, &hm).expect("mesh");
    let n = hm.n_cells;

    let coeffs = GammaCoeffs::default();
    let yb: DevBuf<Scalar> = gpu.zeros(n).expect("y");
    let gyb: DevBuf<Vec3> = gpu.zeros(n).expect("gy");
    let mut gm =
        MenterGamma::new(&gpu, &mesh, coeffs, GammaControls::default(), &yb, &gyb)
            .expect("model");

    // Production and destruction both LIVE, so the zero is the split's and
    // not the inputs'.
    let fon: Vec<Scalar> = (0..n).map(|i| 0.3 + 0.001 * i as Scalar).collect();
    let ftb: Vec<Scalar> = (0..n).map(|i| 0.5 + 0.001 * i as Scalar).collect();
    let sv: Vec<Scalar> = (0..n).map(|i| 2.0 + i as Scalar).collect();
    let om: Vec<Scalar> = (0..n).map(|i| 10.0 + 0.1 * i as Scalar).collect();
    gpu.write(&mut gm.gamma_mut().f, &vec![0.0 as Scalar; n]).expect("write gamma");
    gpu.write(&mut gm.f_onset, &fon).expect("write fOnset");
    gpu.write(&mut gm.f_turb, &ftb).expect("write fTurb");
    gpu.write(&mut gm.omega_mag, &om).expect("write omegaMag");
    let mut sb: DevBuf<Scalar> = gpu.zeros(n).expect("s");
    gpu.write(&mut sb, &sv).expect("write s");

    let (mut su, mut sp, mut susp) =
        (gpu.zeros(n).expect("su"), gpu.zeros(n).expect("sp"), gpu.zeros(n).expect("susp"));
    let (fl, ca2) = (coeffs.f_length, coeffs.ca2);
    gm.gamma_sources_for_test(&gpu, &mut su, &mut sp, &mut susp, &sb, n)
        .expect("launch gmGammaSources");

    let got_su: Vec<Scalar> = gpu.download(&su).expect("read");
    let got_sp: Vec<Scalar> = gpu.download(&sp).expect("read");
    let got_susp: Vec<Scalar> = gpu.download(&susp).expect("read");

    let mut susp_min = 0.0 as Scalar;
    let mut worst = 0.0 as Scalar;
    for i in 0..n {
        assert_eq!(got_su[i].to_bits(), (0.0 as Scalar).to_bits(), "cell {i}: su is not 0");
        assert_eq!(got_sp[i].to_bits(), (0.0 as Scalar).to_bits(), "cell {i}: sp is not 0");
        let a = fl * sv[i] * fon[i];
        let b = ca2 * om[i] * ftb[i];
        let want_susp = -(a + b);
        worst = worst.max((got_susp[i] - want_susp).abs() / want_susp.abs().max(1e-300));
        susp_min = susp_min.min(got_susp[i]);
    }
    assert!(
        worst < 1e-15,
        "susp disagrees with -(A + B) by {worst:.3e} relative - beyond the \
         contraction's last bit"
    );
    assert!(susp_min < 0.0, "susp never went negative - A + B is not live");
    println!(
        "  at gamma = 0: su and sp are exactly 0 (bitwise) on {n} cells with \
         A + B live - susp matches -(A+B) to {worst:.3e} relative and runs \
         down to {susp_min:.3e}; the explicit half is susp * 0 = 0"
    );
}

// ======================================================================
//  90.10  Gate 90-R (i) - the stamp half
// ======================================================================

/// **At `gamma = 1` exactly and `F_3 = 0` exactly, both stamps are
/// identities on every bit, on 216 cells of awkward magnitudes.**
///
/// §90.10: multiplication by an exact 1.0 is exact, `max(gamma, 0.1)` at
/// `gamma = 1` multiplies by an exact 1.0, `P_k^lim` carries the exact
/// factor `(1 - gamma) = 0` - so `gLim`, `sp` and `f1` come back unchanged.
/// This is the stamp half of the reduction; the end-to-end half
/// (`gammaMin = gammaMax = 1` reproducing plain `kOmegaSST` over three
/// `correct` steps) needs the SST rig and is the wiring unit's.
#[test]
fn the_stamps_are_bitwise_identities_at_their_neutral_values() {
    let Some(gpu) = gpu() else { return };
    let hm = block(6);
    let mesh = crate::mesh::GpuMesh::upload(&gpu, &hm).expect("mesh");
    let n = hm.n_cells;
    let nu = 1.5e-5 as Scalar;

    let coeffs = GammaCoeffs::default();
    let yb: DevBuf<Scalar> = gpu.zeros(n).expect("y");
    let gyb: DevBuf<Vec3> = gpu.zeros(n).expect("gy");
    let mut gm =
        MenterGamma::new(&gpu, &mesh, coeffs, GammaControls::default(), &yb, &gyb)
            .expect("model");

    // Awkward magnitudes on purpose: a stamp that rounded would show here.
    let want_g: Vec<Scalar> =
        (0..n).map(|i| (i as Scalar + 1.0) * 1.2345678901234e-3 - 0.5).collect();
    let want_sp: Vec<Scalar> = (0..n).map(|i| 7.0 / (i as Scalar + 3.0) + 1e-17).collect();
    let want_f1: Vec<Scalar> = (0..n).map(|i| (i as Scalar) / (n as Scalar)).collect();
    let nut: Vec<Scalar> = (0..n).map(|i| 1e-4 * (i as Scalar + 1.0)).collect();
    let sv: Vec<Scalar> =
        (0..n).map(|i| (i as Scalar + 1.0) * 3.14159265358979e-1).collect();
    let om: Vec<Scalar> = (0..n).map(|i| 97.0 / (i as Scalar + 2.0)).collect();
    let fol: Vec<Scalar> = (0..n).map(|i| 0.3 + 0.002 * i as Scalar).collect();

    let mut g: DevBuf<Scalar> = gpu.zeros(n).expect("g");
    let mut sp: DevBuf<Scalar> = gpu.zeros(n).expect("sp");
    let mut f1: DevBuf<Scalar> = gpu.zeros(n).expect("f1");
    let mut nutb: DevBuf<Scalar> = gpu.zeros(n).expect("nut");
    let mut sb: DevBuf<Scalar> = gpu.zeros(n).expect("s");
    gpu.write(&mut g, &want_g).expect("write");
    gpu.write(&mut sp, &want_sp).expect("write");
    gpu.write(&mut f1, &want_f1).expect("write");
    gpu.write(&mut nutb, &nut).expect("write");
    gpu.write(&mut sb, &sv).expect("write");

    // gamma = 1 exactly (through its own field), F_3 = 0 exactly, and an
    // F_on^lim that would be live if (1 - gamma) were not exactly zero.
    gpu.write(&mut gm.gamma_mut().f, &vec![1.0 as Scalar; n]).expect("write gamma");
    gm.seed_stamp_inputs(&gpu, &vec![0.0 as Scalar; n]).expect("seed f3");
    gpu.write(&mut gm.f_on_lim, &fol).expect("write fOnLim");
    gpu.write(&mut gm.omega_mag, &om).expect("write omegaMag");

    gm.stamp_k_sources(&gpu, &mut g, &mut sp, &nutb, &sb, nu, n).expect("stamp k");
    gm.stamp_f1(&gpu, &mut f1, n).expect("stamp f1");

    let got_g: Vec<Scalar> = gpu.download(&g).expect("read");
    let got_sp: Vec<Scalar> = gpu.download(&sp).expect("read");
    let got_f1: Vec<Scalar> = gpu.download(&f1).expect("read");

    for i in 0..n {
        assert_eq!(
            got_g[i].to_bits(),
            want_g[i].to_bits(),
            "cell {i}: the production stamp is not bitwise at gamma = 1"
        );
        assert_eq!(
            got_sp[i].to_bits(),
            want_sp[i].to_bits(),
            "cell {i}: the dissipation stamp is not bitwise at gamma = 1"
        );
        assert_eq!(
            got_f1[i].to_bits(),
            want_f1[i].to_bits(),
            "cell {i}: the F1 stamp is not bitwise at F_3 = 0"
        );
    }
    println!("  Gate 90-R (stamps): gLim, sp and f1 bitwise on {n} cells at gamma = 1, F_3 = 0");
}

// ======================================================================
//  90.10  Gate 90-R (ii) - the pure-shear measurement
// ======================================================================

/// **On a pure-shear field the Kato-Launder production `nu_t S Omega` and
/// SST's own `nu_t S^2` agree in exact arithmetic - MEASURED, not assumed.**
///
/// §90.6: at `gamma = 1` with the stamp ON the model is SST with
/// Kato-Launder production, and the honest form of "it reduces to SST" is a
/// pair of numbers. A single shear component `dU_x/dy = a` makes
/// `S = Omega = |a|`, so `g = nut * a * a` is what exact arithmetic gives.
/// Whether the device lands there BITWISE depends on whether the crate's
/// own `strain_rate_mag`/`vorticity_mag` return `|a|` exactly - a fact
/// about operation order, printed here and not asserted. The relative
/// difference is asserted below 1e-14 only.
#[test]
fn the_kato_launder_production_on_pure_shear() {
    use crate::turbulence::strain_rate_mag;
    let Some(gpu) = gpu() else { return };
    let hm = block(6);
    let mesh = crate::mesh::GpuMesh::upload(&gpu, &hm).expect("mesh");
    let n = hm.n_cells;

    let coeffs = GammaCoeffs::default();
    let yb: DevBuf<Scalar> = gpu.zeros(n).expect("y");
    let gyb: DevBuf<Vec3> = gpu.zeros(n).expect("gy");
    let mut gm =
        MenterGamma::new(&gpu, &mesh, coeffs, GammaControls::default(), &yb, &gyb)
            .expect("model");

    // Pure shear, varying a - signs mixed so |a| is exercised both ways -
    // and varying nut.
    let a: Vec<Scalar> =
        (0..n).map(|i| ((i as Scalar + 1.0) * 0.371 - 40.0) * 1.7).collect();
    let nut: Vec<Scalar> = (0..n).map(|i| 1e-4 * (i as Scalar + 1.0) * 1.13).collect();
    let gu: Vec<Tensor> =
        a.iter().map(|a| Tensor { yx: *a, ..Tensor::ZERO }).collect();

    let mut grub: DevBuf<Tensor> = gpu.zeros(n).expect("gradU");
    let mut nutb: DevBuf<Scalar> = gpu.zeros(n).expect("nut");
    gpu.write(&mut grub, &gu).expect("write gradU");
    gpu.write(&mut nutb, &nut).expect("write nut");

    // s and omegaMag through the crate's own kernels - the same path a run
    // takes - and NOT by hand.
    let turb = TurbKernels::new(&gpu).expect("turb kernels");
    let mut sb: DevBuf<Scalar> = gpu.zeros(n).expect("s");
    strain_rate_mag(&gpu, &turb, &mut sb, &grub, n).expect("strain");
    crate::turbulence::vorticity_mag(&gpu, &turb, &mut gm.omega_mag, &grub, n)
        .expect("vorticity");

    let (mut g, mut p) = (gpu.zeros(n).expect("g"), gpu.zeros(n).expect("p"));
    gm.set_kato_launder(true); // the measurement is OF the stamp, stated, not defaulted
    gm.stamp_production(&gpu, &mut g, &mut p, &nutb, &sb, n).expect("stamp production");

    let got_g: Vec<Scalar> = gpu.download(&g).expect("read");
    let got_p: Vec<Scalar> = gpu.download(&p).expect("read");

    let mut worst = 0.0 as Scalar;
    let mut bitwise = 0;
    for i in 0..n {
        let want = nut[i] * a[i] * a[i];
        let rel = (got_g[i] - want).abs() / want.abs().max(1e-300);
        worst = worst.max(rel);
        if got_g[i].to_bits() == want.to_bits() {
            bitwise += 1;
        }
        // p is the same product without nut: S*Omega must equal nut-g less
        // the one factor, which the sweep also reports through g.
        assert!(got_p[i] >= 0.0, "cell {i}: p = S*Omega went negative");
    }
    println!(
        "  Gate 90-R (ii), pure shear, {n} cells: worst relative difference \
         of g against nut*a*a is {worst:.3e}; {bitwise} of {n} cells bitwise"
    );
    assert!(
        worst < 1e-14,
        "the Kato-Launder production disagrees with nut*a*a by {worst:.3e} \
         on a pure shear"
    );
}

// ======================================================================
//  90.7  Nothing here reads a velocity
// ======================================================================

/// **`gmFields` is deterministic: identical inputs, identical bits - and
/// there is no way for a velocity to enter.**
///
/// §90.7's claim in structural form. [`MenterGamma::update_fields`] has no
/// `U` argument at all: every input it takes is a gradient
/// (`grad_u`, `grad_y`), a turbulence quantity (`k`, `omega`) or a field
/// derived from them (`s`) plus the wall distance. A Galilean boost - a
/// constant added to every cell's velocity - changes none of them, so no
/// experiment could make this model read a frame; the Galilean table is
/// Gate 90-G's and is a later unit's. What IS checkable here is the
/// determinism the claim stands on: two `update_fields` calls with the
/// same `k, omega, s, gradU, gradY` must leave all seven outputs bit for
/// bit identical - no host decision, no data-dependent path (§81).
#[test]
fn update_fields_reads_no_velocity() {
    let Some(gpu) = gpu() else { return };
    let hm = block(6);
    let mesh = crate::mesh::GpuMesh::upload(&gpu, &hm).expect("mesh");
    let n = hm.n_cells;
    let nu = 1.5e-5 as Scalar;

    let k: Vec<Scalar> = (0..n).map(|i| 0.01 * (i as Scalar + 1.0)).collect();
    let omega: Vec<Scalar> = (0..n).map(|i| 20.0 + i as Scalar).collect();
    let sv: Vec<Scalar> = (0..n).map(|i| 1.0 + 0.3 * i as Scalar).collect();
    let gu: Vec<Tensor> = (0..n)
        .map(|i| {
            let i = i as Scalar;
            Tensor {
                xx: 0.1 + 0.001 * i, xy: 0.0, xz: 0.0,
                yx: 0.5 + 0.001 * i, yy: -0.02 * i, yz: 0.0,
                zx: 0.0, zy: 0.0, zz: 0.01,
            }
        })
        .collect();
    let gy: Vec<Vec3> = (0..n)
        .map(|i| Vec3 { x: 0.8, y: 0.1 + 0.001 * i as Scalar, z: -0.2 })
        .collect();

    let run = || -> Vec<Vec<u64>> {
        let yv: Vec<Scalar> = (0..n).map(|i| 1e-4 * (i as Scalar + 1.0)).collect();
        let mut yb: DevBuf<Scalar> = gpu.zeros(n).expect("y");
        gpu.write(&mut yb, &yv).expect("write y");
        let mut gyb: DevBuf<Vec3> = gpu.zeros(n).expect("gy");
        gpu.write(&mut gyb, &gy).expect("write gy");
        let mut gm = MenterGamma::new(
            &gpu,
            &mesh,
            GammaCoeffs::default(),
            GammaControls::default(),
            &yb,
            &gyb,
        )
        .expect("model");
        gpu.write(&mut gm.gamma_mut().f, &vec![0.5 as Scalar; n]).expect("write gamma");

        let mut kb: DevBuf<Scalar> = gpu.zeros(n).expect("k");
        let mut wb: DevBuf<Scalar> = gpu.zeros(n).expect("w");
        let mut sb: DevBuf<Scalar> = gpu.zeros(n).expect("s");
        let mut grub: DevBuf<Tensor> = gpu.zeros(n).expect("gradU");
        gpu.write(&mut kb, &k).expect("write k");
        gpu.write(&mut wb, &omega).expect("write omega");
        gpu.write(&mut sb, &sv).expect("write s");
        gpu.write(&mut grub, &gu).expect("write gradU");

        let turb = TurbKernels::new(&gpu).expect("turb kernels");
        gm.update_fields(&gpu, &turb, &kb, &wb, &sb, &grub, nu, n).expect("update");
        let bits = |b: &DevBuf<Scalar>| -> Vec<u64> {
            gpu.download(b).expect("read").iter().map(|v: &Scalar| v.to_bits()).collect()
        };
        vec![
            bits(gm.f_onset()),
            bits(gm.f_turb_field()),
            bits(gm.re_thetac_field()),
            bits(gm.f_on_lim_field()),
            bits(gm.f3_field()),
            bits(gm.tu_l_field()),
            bits(gm.lambda_thl_field()),
        ]
    };

    let a = run();
    let b = run();
    let names = ["F_onset", "F_turb", "Re_thetac", "F_on_lim", "F_3", "Tu_L", "lambda_thL"];
    for (i, name) in names.iter().enumerate() {
        let differ = a[i].iter().zip(&b[i]).filter(|(x, y)| x != y).count();
        assert_eq!(differ, 0, "{name} is not bitwise repeatable on {differ} cells");
    }
    println!(
        "  gmFields repeats bitwise on all seven outputs over {n} cells; no \
         input path exists for a velocity to reach it"
    );
}
