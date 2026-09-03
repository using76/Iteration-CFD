// meteor-cfd - Copyright (c) 2026 주식회사 메테오시뮬레이션 (Meteo Simulation Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.
//
// SPEC-LIT §88.11's table, one test per row, plus Gate 88-R (the bitwise
// reduction to plain SST) and Gate 88-T (the transition LOCATION, which is
// the only thing a transition model exists to predict).
// No GPL-licensed source was consulted.

#![allow(clippy::float_cmp)]

use super::*;
use crate::blockgen::{self, BlockSpec, GradedAxis};
use crate::field::GpuSurfaceScalarField;
use crate::mesh::HostMesh;

fn gpu() -> Option<Gpu> {
    Gpu::new(0).ok()
}

/// A plain block with the floor a wall, so `y` is unambiguous.
fn block(n: usize) -> HostMesh {
    let mut spec = BlockSpec {
        x: GradedAxis { lo: 0.0, hi: 1.0, n, expansion: 1.0, two_sided: false },
        y: GradedAxis { lo: 0.0, hi: 0.2, n, expansion: 1.0, two_sided: false },
        z: GradedAxis { lo: 0.0, hi: 0.2, n, expansion: 1.0, two_sided: false },
        ..BlockSpec::default()
    };
    for p in [1, 3, 4, 5] {
        spec.patch_type[p] = "patch".to_string();
    }
    blockgen::build_mesh(&spec).expect("block")
}

// ======================================================================
//  §88.2  A Blasius boundary layer, solved here, so that 2.193 is
//  DERIVED rather than accepted
// ======================================================================

/// The Blasius similarity solution `f''' + f f''/2 = 0`, integrated by
/// classical RK4 with a shooting correction on `f''(0)`.
///
/// Returns `(eta, f, f', f'')` sampled at every step. This is the only
/// numerical apparatus in this file that is not the model, and it exists so
/// that (88.7)'s `2.193` - the constant that lets a STRICTLY LOCAL vorticity
/// Reynolds number stand in for a momentum thickness, which is an integral
/// across the layer - is checked against a Blasius profile computed here
/// rather than taken on trust from the paper that prints it.
fn blasius(eta_max: Scalar, n: usize) -> Vec<(Scalar, Scalar, Scalar, Scalar)> {
    let h = eta_max / n as Scalar;

    // f''' = -f f''/2
    let step = |y: [Scalar; 3]| -> [Scalar; 3] { [y[1], y[2], -0.5 * y[0] * y[2]] };
    let rk4 = |mut y: [Scalar; 3], out: &mut Vec<(Scalar, Scalar, Scalar, Scalar)>| {
        out.clear();
        out.push((0.0, y[0], y[1], y[2]));
        for i in 0..n {
            let k1 = step(y);
            let k2 = step([y[0] + 0.5 * h * k1[0], y[1] + 0.5 * h * k1[1], y[2] + 0.5 * h * k1[2]]);
            let k3 = step([y[0] + 0.5 * h * k2[0], y[1] + 0.5 * h * k2[1], y[2] + 0.5 * h * k2[2]]);
            let k4 = step([y[0] + h * k3[0], y[1] + h * k3[1], y[2] + h * k3[2]]);
            for j in 0..3 {
                y[j] += h / 6.0 * (k1[j] + 2.0 * k2[j] + 2.0 * k3[j] + k4[j]);
            }
            out.push(((i + 1) as Scalar * h, y[0], y[1], y[2]));
        }
        y[1]
    };

    // Shoot on f''(0) by the secant method until f'(eta_max) = 1.
    let mut out = Vec::with_capacity(n + 1);
    let (mut a, mut b) = (0.3 as Scalar, 0.4 as Scalar);
    let mut fa = rk4([0.0, 0.0, a], &mut out) - 1.0;
    let mut fb = rk4([0.0, 0.0, b], &mut out) - 1.0;
    for _ in 0..80 {
        if (fb - fa).abs() < 1e-300 {
            break;
        }
        let c = b - fb * (b - a) / (fb - fa);
        a = b;
        fa = fb;
        b = c;
        fb = rk4([0.0, 0.0, b], &mut out) - 1.0;
        if fb.abs() < 1e-14 {
            break;
        }
    }
    rk4([0.0, 0.0, b], &mut out);
    out
}

/// **SPEC-LIT (88.7): `2.193` is `max_eta(eta^2 f'')/int f'(1 - f')`, and
/// this derives both halves from a Blasius solution computed here.**
///
/// The model's onset switch reads `F_onset1 = Re_V/(2.193 Re_thetac)` with
/// `Re_V = S d^2/nu`, a purely local quantity. It is a stand-in for
/// `Re_theta`, which is not local at all. The substitution is exact only
/// because on a Blasius profile
///
/// ```text
/// max_y Re_V = sqrt(Re_x) max_eta(eta^2 f'')  and  Re_theta = sqrt(Re_x) int f'(1 - f')
/// ```
///
/// so their ratio is a pure number. **If this test fails, the model's onset
/// criterion is not the published one**, and no amount of agreement on the
/// correlations would tell you.
#[test]
fn the_two_point_one_nine_three_is_a_property_of_the_blasius_profile() {
    let sol = blasius(10.0, 200_000);

    // f''(0), the classical 0.33206.
    let fpp0 = sol[0].3;
    assert!(
        (fpp0 - 0.332057).abs() < 2e-5,
        "Blasius f''(0) = {fpp0}, want 0.332057"
    );

    // int_0^inf f'(1 - f') d(eta) = 0.664, the momentum thickness.
    let h = sol[1].0 - sol[0].0;
    let mut theta = 0.0;
    for w in sol.windows(2) {
        let g = |s: &(Scalar, Scalar, Scalar, Scalar)| s.2 * (1.0 - s.2);
        theta += 0.5 * h * (g(&w[0]) + g(&w[1]));
    }
    assert!(
        (theta - 0.664).abs() < 1e-3,
        "Blasius theta/sqrt(nu x/U) = {theta}, want 0.664"
    );

    // max eta^2 f''.
    let mut rev = 0.0 as Scalar;
    for s in &sol {
        rev = rev.max(s.0 * s.0 * s.3);
    }

    let ratio = rev / theta;
    let rel = (ratio - BLASIUS_REV_OVER_RETHETA).abs() / BLASIUS_REV_OVER_RETHETA;
    println!(
        "  Blasius: f''(0) = {fpp0:.6}, theta = {theta:.6}, \
         max(eta^2 f'') = {rev:.6}, ratio = {ratio:.6} against the model's {} \
         -> {:.3} %",
        BLASIUS_REV_OVER_RETHETA,
        100.0 * rel
    );
    // **Confirmed to 0.21 %, and the residue is NOT accounted for here.**
    // The two halves of the ratio are each converged to five figures - the
    // integration is RK4 at h = 5e-5 and f''(0) lands on the classical
    // 0.332057 - so the gap is in the published constant or in a definition
    // this test has not reconstructed, not in the arithmetic. It is small
    // enough that it moves the onset criterion by 0.21 % and large enough
    // that claiming `2.193` is DERIVED here would be an overstatement, so
    // §88.7 says confirmed-to rather than derived.
    assert!(
        rel < 5e-3,
        "max(Re_V)/Re_theta on a Blasius profile is {ratio}, and the model \
         divides by {} - a {:.2} % gap, far more than the 0.21 % measured \
         when this was written",
        BLASIUS_REV_OVER_RETHETA,
        100.0 * rel
    );
}

// ======================================================================
//  §88.3  The two correlations
// ======================================================================

/// **The expanded form the TMR prints and the nested form Langtry & Menter
/// print are the same polynomial, and this measures what the rearrangement
/// costs.**
///
/// They are algebraically identical and numerically are not: the expanded
/// form adds `1.0120656 Re` and subtracts terms of comparable size, so it
/// loses significance where the nested form does not. The gap is REPORTED
/// rather than merely bounded, because it is the size of the only difference
/// between following the paper and following the documentation.
#[test]
fn the_two_forms_of_re_thetac_agree() {
    let mut worst = 0.0 as Scalar;
    let mut worst_at = 0.0 as Scalar;
    let mut r = 20.0 as Scalar;
    while r <= 1870.0 {
        let a = re_thetac(r);
        let b = re_thetac_nested(r);
        let rel = (a - b).abs() / a.abs().max(1e-30);
        if rel > worst {
            worst = rel;
            worst_at = r;
        }
        r += 0.5;
    }
    println!("  Re_thetac expanded vs nested: worst {worst:.3e} relative, at Re_theta~ = {worst_at}");
    assert!(
        worst < 1e-12,
        "the two published forms of Re_thetac disagree by {worst:.3e} at \
         Re_theta~ = {worst_at}, which is far more than a rearrangement costs"
    );
}

/// **`Re_thetac < Re_theta~` over the whole fitted range, and both are
/// positive.**
///
/// This is the correlation's own content: the critical momentum-thickness
/// Reynolds number, where the model may start producing intermittency, sits
/// BELOW the transition-onset one the experiment records. A sign error in
/// any coefficient breaks it.
#[test]
fn re_thetac_is_below_re_thetat_and_positive_over_the_fitted_range() {
    let mut r = 20.0 as Scalar;
    while r <= 4000.0 {
        let c = re_thetac(r);
        assert!(c > 0.0, "Re_thetac({r}) = {c} is not positive - F_onset1 divides by it");
        assert!(c < r, "Re_thetac({r}) = {c} is not below Re_theta~");
        r += 1.0;
    }
    // And the floor the coefficient check enforces is where the polynomial
    // is still comfortably positive.
    let at20 = re_thetac(20.0);
    println!("  Re_thetac(20) = {at20:.4}  (the ReThetatMin floor of 88.8)");
    assert!(at20 > 10.0, "Re_thetac at the floor is {at20}");
}

/// **The published `F_length` correlation is DISCONTINUOUS at two of its
/// three breakpoints, and this is the measurement.**
///
/// This test was written to assert continuity, on the reasoning that a jump
/// at a breakpoint is a transcription error in one of the two pieces meeting
/// there. **It failed, and the failure is not ours**: both pieces are
/// transcribed from the TMR verbatim and the jump is in Langtry & Menter's
/// own fit, whose four pieces were evidently fitted to reproduce values
/// rather than to meet.
///
/// | breakpoint | left | right | jump |
/// |---|---|---|---|
/// | 400 | 13.83738 | 13.84000 | 0.019 % |
/// | 596 | 0.495985 | 0.500000 | **0.81 %** |
/// | 1200 | 0.318800 | 0.318800 | 1e-12 %, i.e. the two pieces meet |
///
/// So the test now MEASURES the jumps, bounds them where the published fit
/// actually bounds them, and pins the one breakpoint that IS exact. What a
/// 0.81 % jump costs is a 0.81 % step in the intermittency production rate
/// across the cells that straddle `Re_theta~ = 596` - a shift in transition
/// LENGTH, not in onset, and far below the scatter of the data the
/// correlation was fitted to. `Re_thetac` is continuous at 1870, and that
/// one IS asserted, because it is the breakpoint whose two pieces were
/// written to meet.
#[test]
fn the_published_f_length_is_discontinuous_and_this_measures_it() {
    let eps = 1e-9 as Scalar;
    let mut jumps = Vec::new();
    for x in [400.0 as Scalar, 596.0, 1200.0] {
        let lo = f_length1(x - eps);
        let hi = f_length1(x + eps);
        let rel = (hi - lo).abs() / lo.abs().max(1e-30);
        println!(
            "  F_length1 at {x}: {lo:.8} -> {hi:.8}, jump {:.3e} ({:.3} %)",
            (hi - lo).abs(),
            100.0 * rel
        );
        jumps.push(rel);
    }
    // 1200 is exact: 0.5 - 3.0e-4 (1200 - 596) = 0.5 - 0.1812 = 0.3188.
    // 3e-13 absolute, which is `eps` times the slope: the two pieces meet.
    assert!(jumps[2] < 1e-11, "the 1200 breakpoint is not exact after all: {}", jumps[2]);
    assert!(jumps[0] < 1e-3, "the 400 jump grew to {}", jumps[0]);
    assert!(
        jumps[1] < 1e-2,
        "the 596 jump is {} - larger than the published fit's own",
        jumps[1]
    );
    // And it is a real jump, not round-off: it is many orders above eps.
    assert!(
        jumps[1] > 1e-3,
        "the 596 jump measured {} - if this has become round-off, one of the \
         two pieces has been changed and the transcription must be rechecked \
         against the TMR",
        jumps[1]
    );

    let lo = re_thetac(1870.0 - eps);
    let hi = re_thetac(1870.0 + eps);
    let jump = (hi - lo).abs();
    println!("  Re_thetac at 1870: {lo:.8} -> {hi:.8}, jump {jump:.3e}");
    assert!(jump < 1e-3 * lo, "Re_thetac jumps by {jump} at 1870");
}

/// **More free-stream turbulence transitions earlier, at every intensity.**
///
/// The single most basic thing a transition correlation must do, and the one
/// a compensating sign error breaks silently: a model that got T3A right and
/// the trend backwards would still produce a plausible converged answer on
/// that one case.
#[test]
fn the_zpg_correlation_is_monotone_decreasing_in_tu() {
    let mut prev = Scalar::INFINITY;
    let mut tu = 0.03 as Scalar;
    while tu <= 12.0 {
        let re = re_theta_eq_raw(tu, 0.0);
        assert!(
            re <= prev,
            "Re_theta_eq is not monotone in Tu: {re} at Tu = {tu} against {prev} just before"
        );
        prev = re;
        tu += 0.01;
    }
    // The three ERCOFTAC T3 intensities, printed for §88.10's table.
    for tu in [0.9 as Scalar, 3.3, 6.5] {
        let re = re_theta_eq_raw(tu, 0.0);
        println!("  Tu = {tu} %: Re_theta_eq = {re:.2}, Re_thetac = {:.2}", re_thetac(re));
    }
}

/// **The three published numerical limits are enforced, and they are the
/// TMR's and not ours.**
#[test]
fn the_published_limits_are_the_ones_enforced() {
    // Tu >= 0.027.
    assert_eq!(turbulence_intensity(0.0, 10.0), 0.027);
    assert_eq!(turbulence_intensity(1e-30, 10.0), 0.027);
    // Re_theta_eq >= 20. At an absurd Tu the raw correlation runs negative.
    let huge = re_theta_eq_raw(1000.0, 0.0);
    assert_eq!(huge, 20.0, "the Re_theta_eq floor of 20 is not applied");
    // lambda_theta clipped into [-0.1, 0.1]: past the clip the answer stops
    // moving.
    let a = re_theta_eq(3.3, 1e6, 1.5e-5, 5.0, 10);
    let b = re_theta_eq(3.3, 1e12, 1.5e-5, 5.0, 10);
    assert_eq!(a, b, "lambda_theta is not clipped at +0.1");
    let a = re_theta_eq(3.3, -1e6, 1.5e-5, 5.0, 10);
    let b = re_theta_eq(3.3, -1e12, 1.5e-5, 5.0, 10);
    assert_eq!(a, b, "lambda_theta is not clipped at -0.1");
}

/// **The `Re_theta_eq` fixed point has converged by ten sweeps, and this
/// measures how far past convergence ten is.**
///
/// SPEC-LIT §88.4: the sweep count is OURS, and it is a constant because a
/// convergence test inside the kernel would make the trip count depend on a
/// floating-point comparison - not capturable, not bitwise. The design note
/// asked for `N = 10` and `N = 20` to agree to `1e-10` relative; that is
/// what is measured here, over a sweep of pressure gradients broad enough to
/// hit both sides of the `lambda_theta` clip.
#[test]
fn ten_sweeps_of_the_fixed_point_is_past_convergence() {
    let (nu, u) = (1.5e-5 as Scalar, 5.0 as Scalar);
    let mut worst = 0.0 as Scalar;
    let mut worst_where = (0.0 as Scalar, 0.0 as Scalar);
    for tu in [0.05 as Scalar, 0.3, 0.9, 1.3, 3.3, 6.5, 10.0] {
        for du_ds in [-2e4 as Scalar, -2e3, -200.0, -20.0, 0.0, 20.0, 200.0, 2e3, 2e4] {
            let a = re_theta_eq(tu, du_ds, nu, u, 10);
            let b = re_theta_eq(tu, du_ds, nu, u, 20);
            let c = re_theta_eq(tu, du_ds, nu, u, 40);
            let rel = ((a - b).abs() / b.max(1e-30)).max((b - c).abs() / c.max(1e-30));
            if rel > worst {
                worst = rel;
                worst_where = (tu, du_ds);
            }
        }
    }
    println!(
        "  Re_theta_eq: worst |N=10 vs N=20| and |N=20 vs N=40| = {worst:.3e} relative, \
         at Tu = {} %, dU/ds = {}",
        worst_where.0, worst_where.1
    );
    assert!(
        worst < 1e-10,
        "ten sweeps is not past convergence: {worst:.3e} relative at Tu = {}, \
         dU/ds = {}",
        worst_where.0,
        worst_where.1
    );

    // **And the sweeps must do something.** A fixed point that agrees with
    // its own initial guess everywhere would pass the test above while
    // computing nothing at all - which is exactly how a loop whose body was
    // written against the wrong variable would look. So: at a real pressure
    // gradient the converged answer must differ from the zero-pressure-
    // gradient value the iteration starts from, and by a measured amount.
    let mut biggest = 0.0 as Scalar;
    for tu in [0.3 as Scalar, 3.3, 6.5] {
        for du_ds in [-2e4 as Scalar, -2e3, 2e3, 2e4] {
            let start = re_theta_eq_raw(tu, 0.0);
            let end = re_theta_eq(tu, du_ds, nu, u, 10);
            biggest = biggest.max((end - start).abs() / start);
        }
    }
    println!("    the sweeps move Re_theta_eq by up to {:.2} % off its ZPG guess", 100.0 * biggest);
    assert!(
        biggest > 0.01,
        "the fixed point never moves off its initial guess - the sweep body \
         is not reading the pressure gradient"
    );
}

/// **`re_thetat_inlet` IS the TMR's farfield boundary condition, to the
/// printed digit.**
///
/// The TMR states the `Re_theta~` farfield value as the same two-piece
/// correlation with `F(lambda) = 1`, which is exactly `re_theta_eq_raw` at
/// `lambda = 0`. This pins the two together so that a case writing an inlet
/// value by hand and a case computing one cannot drift.
#[test]
fn the_inlet_correlation_is_the_farfield_boundary_condition() {
    for tu in [0.1 as Scalar, 0.9, 1.3, 3.3, 5.855, 6.5] {
        let want = if tu <= 1.3 {
            1173.51 - 589.428 * tu + 0.2196 / (tu * tu)
        } else {
            331.50 * (tu - 0.5658).powf(-0.671)
        }
        .max(20.0);
        let got = re_thetat_inlet(tu);
        assert!(
            (got - want).abs() <= 8.0 * Scalar::EPSILON * want,
            "Tu = {tu}: inlet {got} vs farfield correlation {want}"
        );
    }
    println!(
        "  ReThetat inlet: Tu 0.9 % -> {:.2}, 3.3 % -> {:.2}, 6.5 % -> {:.2}",
        re_thetat_inlet(0.9),
        re_thetat_inlet(3.3),
        re_thetat_inlet(6.5)
    );
}

// ======================================================================
//  §88.5  The source terms
// ======================================================================

/// **`gamma = 1` is the fixed point where the flow is turbulent, and
/// `gamma = 1/c_e2` where it is not.**
///
/// `P_gamma - E_gamma` as a function of `gamma` alone:
///
/// ```text
/// A sqrt(gamma)(1 - c_e1 gamma) - B gamma (c_e2 gamma - 1)
/// ```
///
/// With `c_e1 = 1` the first term vanishes at `gamma = 1`, and `F_turb -> 0`
/// in a turbulent region kills the second - so a switched boundary layer sits
/// at exactly 1. Where `F_onset = 0` the first term is absent entirely and
/// the second vanishes at `gamma = 1/c_e2 = 0.02`. Both are properties of the
/// constants, and both are what makes the intermittency mean what its name
/// says.
#[test]
fn the_intermittency_has_the_two_fixed_points_the_constants_imply() {
    let c = LmCoeffs::default();
    let source = |gamma: Scalar, a: Scalar, b: Scalar| {
        a * gamma.sqrt() * (1.0 - c.ce1 * gamma) - b * gamma * (c.ce2 * gamma - 1.0)
    };

    // Turbulent: F_onset > 0 so A > 0, F_turb = 0 so B = 0.
    let (a, b) = (3.0 as Scalar, 0.0 as Scalar);
    assert_eq!(source(1.0, a, b), 0.0);
    assert!(source(0.5, a, b) > 0.0, "gamma below 1 must grow");

    // Laminar: F_onset = 0 so A = 0, F_turb = 1 so B > 0.
    let (a, b) = (0.0 as Scalar, 2.5 as Scalar);
    let g = 1.0 / c.ce2;
    assert!(source(g, a, b).abs() < 1e-15, "the laminar fixed point is not 1/ce2");
    assert!(source(0.5, a, b) < 0.0, "gamma above 1/ce2 must decay in a laminar layer");
    assert!(source(0.001, a, b) > 0.0, "gamma below 1/ce2 must grow back");
    println!("  gamma fixed points: 1.0 (F_turb = 0) and 1/ce2 = {g}");
}

/// **The Patankar split is clean: the diagonal contribution is non-negative
/// at every state.**
///
/// SPEC-LIT §88.5 emits `Su = A`, `Sp = A c_e1 + B c_e2 gamma`,
/// `Susp = -B`. The design note proposed a single lumped
/// `Susp = -(P - E)/gamma` instead; that form divides by `gamma`, and
/// `gamma` is zero in every cell of a laminar initial field. This is the
/// check that the split actually chosen never puts a negative number on the
/// diagonal.
#[test]
fn the_gamma_source_split_never_makes_the_diagonal_negative() {
    let c = LmCoeffs::default();
    for gamma in [0.0 as Scalar, 1e-12, 0.02, 0.5, 1.0] {
        for a in [0.0 as Scalar, 1e-8, 1.0, 1e6] {
            for b in [0.0 as Scalar, 1e-8, 1.0, 1e6] {
                let sp = a * c.ce1 + b * c.ce2 * gamma;
                assert!(sp >= 0.0, "Sp = {sp} at gamma = {gamma}, A = {a}, B = {b}");
                // And the two halves reconstruct the source exactly.
                let su = a;
                let susp = -b;
                let lhs = su - sp * gamma - susp * gamma;
                let rhs = a * (1.0 - c.ce1 * gamma) + b * gamma - b * c.ce2 * gamma * gamma;
                assert!(
                    (lhs - rhs).abs() <= 1e-9 * rhs.abs().max(1.0),
                    "the split does not reconstruct the source: {lhs} vs {rhs}"
                );
            }
        }
    }
}

// ======================================================================
//  §88.9  Coefficients and their refusals
// ======================================================================

#[test]
fn the_coefficient_checks_fire_by_name() {
    let d = LmCoeffs::default();
    d.check().expect("the published set is valid");

    for (c, needle) in [
        (LmCoeffs { ce2: 1.0, ..d }, "ce2"),
        (LmCoeffs { sigma_f: 0.0, ..d }, "sigmaf"),
        (LmCoeffs { sigma_tt: -1.0, ..d }, "sigmaThetat"),
        (LmCoeffs { n_sweeps: 0, ..d }, "nReThetaSweeps"),
        (LmCoeffs { n_sweeps: 1000, ..d }, "nReThetaSweeps"),
        (LmCoeffs { gamma_min: 2.0, ..d }, "gammaMin"),
        (LmCoeffs { re_thetat_min: 5.0, ..d }, "ReThetatMin"),
    ] {
        let e = c.check().expect_err("must be refused");
        let m = e.to_string();
        assert!(m.contains(needle), "the refusal does not name {needle}: {m}");
    }

    // gammaMin == gammaMax is ALLOWED and is a real setting: it freezes the
    // intermittency, which is what Gate 88-R runs the bitwise reduction on.
    LmCoeffs { gamma_min: 1.0, gamma_max: 1.0, ..d }
        .check()
        .expect("a frozen intermittency is a legitimate setting - SPEC-LIT 88.8");
}

/// **The model is NOT Galilean-invariant, and this is how much.**
///
/// `Tu = 100 sqrt(2k/3)/U` reads an ABSOLUTE velocity magnitude, so
/// translating the frame - adding a constant velocity to every cell, which
/// changes no derivative and no physics - changes `Re_theta_eq` and hence
/// where the model transitions. That is a property of LM2009, it is the
/// defect Menter et al. (2015) fixed, and §88.9 records it as a measurement
/// rather than as a remark.
#[test]
fn the_model_is_not_galilean_invariant_and_this_measures_it() {
    let k = 0.05 as Scalar; // Tu = 3.65 % at U = 5
    let base = 5.0 as Scalar;
    let re0 = re_theta_eq_raw(turbulence_intensity(k, base), 0.0);
    println!("  Galilean drift of Re_theta_eq (k = {k}, U = {base} m/s):");
    let mut worst = 0.0 as Scalar;
    for du in [0.5 as Scalar, 1.0, 2.0, 5.0] {
        let re = re_theta_eq_raw(turbulence_intensity(k, base + du), 0.0);
        let rel = (re - re0) / re0;
        worst = worst.max(rel.abs());
        println!("    +{du} m/s -> Re_theta_eq {re0:.2} -> {re:.2}  ({:+.1} %)", 100.0 * rel);
    }
    assert!(
        worst > 0.05,
        "the frame dependence measured {worst:.3e}, which is too small to be \
         LM2009's - either Tu stopped reading an absolute velocity or the \
         test is not exercising it"
    );
}

// ======================================================================
//  §88.10  Gate 88-T - the transition LOCATION
// ======================================================================

/// The NASA/TMBWG 2D T3A transitional flat-plate inflow, fetched live from
/// <https://tmbwg.github.io/turbmodels/t3_transition_mainpage.html> while
/// this section was written.
struct T3a;

impl T3a {
    const U: Scalar = 69.44; // m/s
    const RE_PER_M: Scalar = 2.0e5; // 1/m
    const TU_INLET: Scalar = 5.855; // %
    const NUT_OVER_NU: Scalar = 11.90;
    const TU_LEADING_EDGE: Scalar = 3.300; // %, the number this gate reproduces
    const INFLOW_TO_LE: Scalar = 0.250; // m

    fn nu() -> Scalar {
        Self::U / Self::RE_PER_M
    }

    /// The free-stream decay of `k` and `omega` under SST's outer
    /// coefficients, integrated exactly.
    ///
    /// In a uniform free stream the two transport equations reduce to
    /// `U dk/dx = -beta* k omega` and `U domega/dx = -beta omega^2`, whose
    /// solution is
    ///
    /// ```text
    /// tau      = 1 + beta omega_0 x/U
    /// omega(x) = omega_0/tau ,   k(x) = k_0 tau^(-beta*/beta)
    /// Tu(x)    = Tu_0 tau^(-beta*/(2 beta))
    /// ```
    ///
    /// with `F_1 = 0` in the free stream, so `beta = beta_2 = 0.0828`.
    fn state(x: Scalar) -> (Scalar, Scalar, Scalar) {
        let c = crate::models::KOmegaSstCoeffs::default();
        let nu = Self::nu();
        let k0 = 1.5 * (Self::TU_INLET / 100.0 * Self::U).powi(2);
        let omega0 = k0 / (Self::NUT_OVER_NU * nu);
        let tau = 1.0 + c.beta_2 * omega0 * x / Self::U;
        let k = k0 * tau.powf(-c.beta_star / c.beta_2);
        let omega = omega0 / tau;
        let tu = 100.0 * (2.0 * k / 3.0).sqrt() / Self::U;
        (k, omega, tu)
    }
}

/// **Gate 88-T, leg 1: the free-stream decay reproduces the TMR's published
/// leading-edge turbulence intensity.**
///
/// The TMR specifies T3A by FOUR numbers - `Tu = 5.855 %` and
/// `mu_t/mu = 11.90` at the inflow, `Tu = 3.300 %` at the leading edge
/// 0.250 m downstream - and the fourth is a consequence of the first three
/// under SST's own free-stream decay. Reproducing it is the sharpest
/// available check of the quantity that most strongly sets where transition
/// happens: **the local `Tu`, not the inlet one.**
#[test]
fn gate_88_t_the_free_stream_decay_reaches_the_published_leading_edge_tu() {
    let (k, omega, tu) = T3a::state(T3a::INFLOW_TO_LE);
    let rel = (tu - T3a::TU_LEADING_EDGE).abs() / T3a::TU_LEADING_EDGE;
    println!(
        "  Gate 88-T leg 1 (NASA/TMBWG T3A): Tu at the leading edge = {tu:.4} % \
         against the published {:.3} %  -> {:.2} %; there k = {k:.4} m2/s2, \
         omega = {omega:.1} 1/s",
        T3a::TU_LEADING_EDGE,
        100.0 * rel
    );
    assert!(
        rel < 0.05,
        "the free-stream decay puts Tu at {tu} % at the leading edge against \
         the TMR's {}, a {:.1} % gap",
        T3a::TU_LEADING_EDGE,
        100.0 * rel
    );
}

/// **Gate 88-T, leg 2: where the model says T3A transitions.**
///
/// On a Blasius layer `max_y Re_V = 2.193 Re_theta` by construction
/// (`the_two_point_one_nine_three_is_a_property_of_the_blasius_profile`), so
/// the model's onset switch `F_onset1 = Re_V/(2.193 Re_thetac)` reaches one
/// exactly where `Re_theta = Re_thetac`. With `Re_theta = 0.664 sqrt(Re_x)`
/// and the local `Tu` decaying along the plate, that is a single scalar
/// equation in `x`, and its root is the location the model predicts.
///
/// **What this gate does and does not claim.** It claims the ONSET CRITERION
/// fires where the model's own published correlations say it does, on an
/// exactly-known laminar profile with an exactly-known free-stream decay.
/// It does NOT claim to reproduce the measured skin-friction rise: the
/// intermittency still has to grow from its free-stream value through
/// `P_gamma` over a finite distance, so the measured `C_f` rise is
/// DOWNSTREAM of this point by an amount `F_length` sets. §88.10 records the
/// verdict as OPEN for exactly that reason, and prints both numbers.
#[test]
fn gate_88_t_the_onset_location_on_the_t3a_rig() {
    let _nu = T3a::nu();
    let re_theta_at = |s: Scalar| 0.664 * (T3a::RE_PER_M * s).sqrt();
    let re_thetac_at = |s: Scalar| {
        let (_, _, tu) = T3a::state(T3a::INFLOW_TO_LE + s);
        re_thetac(re_theta_eq_raw(tu, 0.0))
    };

    // Bisect on Re_theta(s) - Re_thetac(s), which is negative at the leading
    // edge (Re_theta = 0 there) and positive far downstream.
    let (mut lo, mut hi) = (1e-6 as Scalar, 50.0 as Scalar);
    assert!(re_theta_at(lo) < re_thetac_at(lo));
    assert!(
        re_theta_at(hi) > re_thetac_at(hi),
        "the onset criterion never fires within 50 m of the leading edge"
    );
    for _ in 0..200 {
        let mid = 0.5 * (lo + hi);
        if re_theta_at(mid) < re_thetac_at(mid) {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    let s = 0.5 * (lo + hi);
    let re_x = T3a::RE_PER_M * s;
    let (_, _, tu) = T3a::state(T3a::INFLOW_TO_LE + s);
    let re_tt = re_theta_eq_raw(tu, 0.0);

    println!(
        "  Gate 88-T leg 2 (T3A, Tu 3.300 % at the leading edge): onset at \
         x - x_LE = {s:.4} m, Re_x = {re_x:.4e}; there the local Tu has \
         decayed to {tu:.4} %, Re_theta~ = {re_tt:.2}, Re_thetac = {:.2}, \
         Re_theta = {:.2}",
        re_thetac(re_tt),
        re_theta_at(s)
    );
    println!(
        "    F_length there = {:.3}, so the C_f rise the experiment records \
         is downstream of this point",
        f_length(re_tt, 1e9)
    );

    // The gate proper: onset is ON the plate, and in the decade the T3
    // series occupies. A model whose correlations were mistranscribed by a
    // factor lands outside this by orders of magnitude.
    assert!(
        (2.0e4..1.0e6).contains(&re_x),
        "T3A onset at Re_x = {re_x:.3e} is nowhere near the T3 series' range"
    );
}

/// **Gate 88-T, leg 3: the trend with `Tu`, and the finding that refutes the
/// obvious way to run it.**
///
/// The T3 series' whole content is that raising the free-stream turbulence
/// from 0.9 % to 6.5 % moves transition by roughly an order of magnitude in
/// `Re_x`. The obvious gate is to run all three on one rig and check the
/// spread. **That gate does not measure what it looks like it measures**, and
/// this test is the measurement that says so: on a single rig the free-stream
/// DECAY is as strong a lever as `Tu` itself, because the low-`Tu` case
/// transitions so far downstream that its own `Tu` has decayed by another
/// factor by the time it gets there. The spread this construction produces is
/// therefore much larger than the measured one, and the excess is the
/// construction's, not the model's.
///
/// So the trend is REPORTED and the T3A-/T3B legs are NOT claimed - §88.10.
#[test]
fn gate_88_t_the_tu_trend_is_confounded_by_the_free_stream_decay() {
    let onset = |tu_le: Scalar| -> (Scalar, Scalar) {
        let scale = tu_le / T3a::TU_LEADING_EDGE;
        let re_theta_at = |s: Scalar| 0.664 * (T3a::RE_PER_M * s).sqrt();
        let re_thetac_at = |s: Scalar| {
            let (_, _, tu) = T3a::state(T3a::INFLOW_TO_LE + s);
            re_thetac(re_theta_eq_raw(tu * scale, 0.0))
        };
        let (mut lo, mut hi) = (1e-6 as Scalar, 500.0 as Scalar);
        for _ in 0..300 {
            let mid = 0.5 * (lo + hi);
            if re_theta_at(mid) < re_thetac_at(mid) {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        let s = 0.5 * (lo + hi);
        let (_, _, tu) = T3a::state(T3a::INFLOW_TO_LE + s);
        (T3a::RE_PER_M * s, tu * scale)
    };

    let mut rows = Vec::new();
    for (name, tu) in [("T3A-", 0.9 as Scalar), ("T3A", 3.3), ("T3B", 6.5)] {
        let (re_x, tu_local) = onset(tu);
        println!(
            "  {name}: Tu(leading edge) = {tu} %  ->  onset Re_x = {re_x:.3e}, \
             local Tu there = {tu_local:.3} %"
        );
        rows.push(re_x);
    }

    // Monotone: more free-stream turbulence, earlier transition. This half
    // IS a property of the model and is asserted.
    assert!(rows[0] > rows[1] && rows[1] > rows[2], "the Tu trend is not monotone: {rows:?}");

    // And the finding: the spread is far wider than the measured one, and
    // the reason is the decay, not the correlation.
    let spread = rows[0] / rows[2];
    println!(
        "    spread T3A-/T3B = {spread:.1}x on this ONE rig, against the ~10x \
         the T3 series measures. The excess is the free-stream decay: each \
         case's own Tu at its own onset point differs by far more than the \
         leading-edge values do, because the low-Tu case transitions much \
         further downstream. A per-case inflow omega - which the TMR \
         publishes for T3A and not for T3A- or T3B - is what would separate \
         the two effects, and it is why 88.10 claims the T3A leg only."
    );
    assert!(spread > 10.0, "spread {spread} - the confound this test names is absent");
}

// ======================================================================
//  §88.6  The two stamps, and Gate 88-R
// ======================================================================

/// **Gate 88-R, the by-construction half: each stamp is the IDENTITY at its
/// neutral value, on every bit.**
///
/// The transition model reaches SST's assembly through exactly two buffers -
/// `g_lim` and `sp` after `sstKSources`, and `f1` between `sstBlending` and
/// `sstBlendCoeffs` - and nothing else; `cuda/sst.cu` has a zero-line diff.
/// So "SST is unmoved" reduces to two statements about IEEE-754:
/// multiplication by an exact `1.0` is exact, and `max(a, 0.0) = a` for
/// `a >= 0`.
#[test]
fn gate_88_r_the_stamps_are_bitwise_identities_at_their_neutral_values() {
    let Some(gpu) = gpu() else { return };
    let hm = block(6);
    let mesh = crate::mesh::GpuMesh::upload(&gpu, &hm).expect("mesh");
    let n = hm.n_cells;

    let y: DevBuf<Scalar> = gpu.zeros(n).expect("y");
    let lm = LangtryMenter::new(&gpu, &mesh, LmCoeffs::default(), LmControls::default(), &y)
        .expect("model");

    // Awkward magnitudes on purpose: a stamp that rounded would show here.
    let want_g: Vec<Scalar> =
        (0..n).map(|i| (i as Scalar + 1.0) * 1.2345678901234e-3 - 0.5).collect();
    let want_sp: Vec<Scalar> =
        (0..n).map(|i| 7.0 / (i as Scalar + 3.0) + 1e-17).collect();
    let want_f1: Vec<Scalar> = (0..n).map(|i| (i as Scalar) / (n as Scalar)).collect();

    let mut g: DevBuf<Scalar> = gpu.zeros(n).expect("g");
    let mut sp: DevBuf<Scalar> = gpu.zeros(n).expect("sp");
    let mut f1: DevBuf<Scalar> = gpu.zeros(n).expect("f1");
    gpu.write(&mut g, &want_g).expect("write");
    gpu.write(&mut sp, &want_sp).expect("write");
    gpu.write(&mut f1, &want_f1).expect("write");

    // gamma_eff = 1 exactly, f3 = 0 exactly - the two neutral values.
    let mut lm = lm;
    lm.seed_stamp_inputs(&gpu, &vec![1.0; n], &vec![0.0; n]).expect("seed");

    lm.stamp_k_sources(&gpu, &mut g, &mut sp, n).expect("stamp k");
    lm.stamp_f1(&gpu, &mut f1, n).expect("stamp f1");

    let got_g: Vec<Scalar> = gpu.download(&g).expect("read");
    let got_sp: Vec<Scalar> = gpu.download(&sp).expect("read");
    let got_f1: Vec<Scalar> = gpu.download(&f1).expect("read");

    for i in 0..n {
        assert_eq!(
            got_g[i].to_bits(),
            want_g[i].to_bits(),
            "cell {i}: the production stamp is not bitwise at gamma_eff = 1"
        );
        assert_eq!(
            got_sp[i].to_bits(),
            want_sp[i].to_bits(),
            "cell {i}: the dissipation stamp is not bitwise at gamma_eff = 1"
        );
        assert_eq!(
            got_f1[i].to_bits(),
            want_f1[i].to_bits(),
            "cell {i}: the F1 stamp is not bitwise at F_3 = 0"
        );
    }
    println!("  Gate 88-R (by construction): both stamps bitwise on {n} cells");
}

/// **Gate 88-R, the end-to-end half: `kOmegaSSTLM` with the intermittency
/// frozen at 1 reproduces plain `kOmegaSST` BIT FOR BIT, over three full
/// `correct` steps, in `k`, `omega` and `nut`.**
///
/// `gammaMin = gammaMax = 1` is a real setting and not a test hook: it is
/// "run the transition model with transition switched off", which is exactly
/// the fully-turbulent limit. The other half of the neutrality is `F_3`,
/// which is `exp(-(R_y/120)^8)` and underflows to exactly `0.0` well before
/// the wall distances and `k` of this block - so `max(F_1, F_3)` returns
/// `F_1` on every bit.
///
/// This is the gate the brief calls "confirmed to reduce correctly". It is
/// necessary and it is not sufficient, which is why Gate 88-T exists.
#[test]
fn gate_88_r_a_frozen_intermittency_reproduces_plain_sst_bitwise() {
    let Some(gpu) = gpu() else { return };
    let hm = block(6);
    let mesh = crate::mesh::GpuMesh::upload(&gpu, &hm).expect("mesh");
    let n = hm.n_cells;
    let wf = crate::field_setup::WallFaces::none(hm.n_boundary_faces);
    let u = GpuVectorField::zeros(&gpu, &mesh, "U").expect("U");
    let phi = GpuSurfaceScalarField::zeros(&gpu, &mesh, "phi").expect("phi");
    let flow = FlowState::new(&u, &phi, 1e-5);
    let ctrl = crate::io::case::TurbulenceControls {
        ddt: crate::timescheme::DdtScheme::Euler,
        steady: false,
        delta_t: 1e-3,
        ..Default::default()
    };
    let wall = crate::wallfunctions::WallFunctionCoeffs::default();

    let (wy, _gy) = {
        let mut y: DevBuf<Scalar> = gpu.zeros(n).expect("y");
        gpu.write(&mut y, &vec![0.05 as Scalar; n]).expect("write");
        (y, ())
    };

    let run = |transition: bool| -> Vec<Vec<u64>> {
        let mut m = crate::models::KOmegaSst::new(
            &gpu,
            &hm,
            &mesh,
            Default::default(),
            ctrl,
            wall,
            &wf,
            &wy,
        )
        .expect("sst");
        gpu.write(&mut m.k_mut().f, &vec![0.05 as Scalar; n]).expect("k");
        gpu.write(&mut m.omega_mut().f, &vec![50.0 as Scalar; n]).expect("omega");
        if transition {
            let coeffs = LmCoeffs { gamma_min: 1.0, gamma_max: 1.0, ..Default::default() };
            let mut lm =
                LangtryMenter::new(&gpu, &mesh, coeffs, LmControls::default(), &wy).expect("lm");
            gpu.write(&mut lm.gamma_mut().f, &vec![1.0 as Scalar; n]).expect("gamma");
            gpu.write(&mut lm.re_thetat_mut().f, &vec![300.0 as Scalar; n]).expect("rtt");
            lm.initialise(&gpu, &mesh).expect("lm init");
            m.set_transition(Some(lm)).expect("attach");
        }
        m.initialise(&gpu, &flow).expect("init");
        for _ in 0..3 {
            m.correct(&gpu, &flow).expect("correct");
        }
        let bits = |f: &crate::field::GpuScalarField| -> Vec<u64> {
            gpu.download(&f.f).expect("read").iter().map(|v: &Scalar| v.to_bits()).collect()
        };
        vec![bits(m.k()), bits(m.omega()), bits(m.nut())]
    };

    let plain = run(false);
    let with_lm = run(true);
    for (i, name) in ["k", "omega", "nut"].iter().enumerate() {
        let differ = plain[i].iter().zip(&with_lm[i]).filter(|(a, b)| a != b).count();
        assert_eq!(
            differ, 0,
            "Gate 88-R: {differ} of {n} cells of {name} differ between plain \
             SST and kOmegaSSTLM with the intermittency frozen at 1"
        );
    }
    println!(
        "  Gate 88-R (end to end): kOmegaSSTLM with gamma frozen at 1 \
         reproduces kOmegaSST on every bit of k, omega and nut over three \
         correct steps, {n} cells"
    );
}

// ======================================================================
//  §88.11  The device and the host agree
// ======================================================================

/// **Every correlation on the device is the one on the host, over a sweep
/// that covers the whole fitted range.**
///
/// The host functions above are what Gate 88-T's arithmetic runs through,
/// and the device kernel is what a run uses. They are two transcriptions of
/// the same polynomials, and §80's measured limitation applies to prose but
/// not to this: a digit dropped in one of them is only visible by measuring
/// them against each other.
#[test]
fn the_host_and_device_correlations_agree() {
    let Some(gpu) = gpu() else { return };
    let hm = block(8);
    let mesh = crate::mesh::GpuMesh::upload(&gpu, &hm).expect("mesh");
    let n = hm.n_cells;
    assert!(n >= 256, "the sweep needs cells to spread over, got {n}");

    let nu = 1.5e-5 as Scalar;
    let u_mag = 5.0 as Scalar;

    // One station per cell: Re_theta~ swept over the whole fitted range, and
    // k swept so that Tu covers both branches of the Re_theta_eq correlation.
    let re_tt: Vec<Scalar> =
        (0..n).map(|i| 20.0 + 3000.0 * (i as Scalar) / (n as Scalar - 1.0)).collect();
    let tu: Vec<Scalar> =
        (0..n).map(|i| 0.05 + 8.0 * (i as Scalar) / (n as Scalar - 1.0)).collect();
    let k: Vec<Scalar> =
        tu.iter().map(|t| 1.5 * (t / 100.0 * u_mag).powi(2)).collect();
    let omega: Vec<Scalar> =
        (0..n).map(|i| 10.0 + 5.0 * i as Scalar).collect();
    let yv: Vec<Scalar> = (0..n).map(|i| 1e-4 * (i as Scalar + 1.0)).collect();
    let sv: Vec<Scalar> = (0..n).map(|i| 1.0 + 0.5 * i as Scalar).collect();

    let mut y: DevBuf<Scalar> = gpu.zeros(n).expect("y");
    gpu.write(&mut y, &yv).expect("write y");

    let coeffs = LmCoeffs::default();
    let mut lm =
        LangtryMenter::new(&gpu, &mesh, coeffs, LmControls::default(), &y).expect("model");
    gpu.write(&mut lm.re_thetat_mut().f, &re_tt).expect("write rtt");
    gpu.write(&mut lm.gamma_mut().f, &vec![0.5 as Scalar; n]).expect("write gamma");

    let mut kb: DevBuf<Scalar> = gpu.zeros(n).expect("k");
    let mut wb: DevBuf<Scalar> = gpu.zeros(n).expect("w");
    let mut sb: DevBuf<Scalar> = gpu.zeros(n).expect("s");
    gpu.write(&mut kb, &k).expect("write k");
    gpu.write(&mut wb, &omega).expect("write omega");
    gpu.write(&mut sb, &sv).expect("write s");

    // A uniform velocity, so dU/ds is zero and the fixed point stays at the
    // zero-pressure-gradient branch the host functions evaluate.
    let mut uf = GpuVectorField::zeros(&gpu, &mesh, "U").expect("U");
    gpu.write(&mut uf.f, &vec![Vec3 { x: u_mag, y: 0.0, z: 0.0 }; n]).expect("write U");
    let grad_u: DevBuf<Tensor> = gpu.zeros(n).expect("grad_u");

    let turb = TurbKernels::new(&gpu).expect("turb kernels");
    lm.update_fields(&gpu, &turb, &kb, &wb, &sb, &uf, &grad_u, nu, n)
        .expect("update");

    let got_c: Vec<Scalar> = gpu.download(lm.re_thetac_field()).expect("read");
    let got_l: Vec<Scalar> = gpu.download(lm.f_length_field()).expect("read");
    let got_e: Vec<Scalar> = gpu.download(lm.re_theta_eq_field()).expect("read");
    let got_o: Vec<Scalar> = gpu.download(lm.f_onset()).expect("read");
    let got_3: Vec<Scalar> = gpu.download(lm.f3_field()).expect("read");

    let mut worst = [0.0 as Scalar; 5];
    for i in 0..n {
        let re_w = omega[i] * yv[i] * yv[i] / nu;
        let r_t = k[i] / (nu * omega[i]);
        let r_v = sv[i] * yv[i] * yv[i] / nu;
        let r_y = yv[i] * k[i].sqrt() / nu;

        let want = [
            re_thetac(re_tt[i]),
            f_length(re_tt[i], re_w),
            re_theta_eq(turbulence_intensity(k[i], u_mag), 0.0, nu, u_mag, coeffs.n_sweeps),
            f_onset(r_v, re_thetac(re_tt[i]).max(1e-30), r_t),
            f3(r_y),
        ];
        let got = [got_c[i], got_l[i], got_e[i], got_o[i], got_3[i]];
        for j in 0..5 {
            let d = (got[j] - want[j]).abs() / want[j].abs().max(1e-12);
            worst[j] = worst[j].max(d);
        }
    }
    let names = ["Re_thetac", "F_length", "Re_theta_eq", "F_onset", "F_3"];
    for j in 0..5 {
        println!("  host vs device, {}: worst {:.3e} relative", names[j], worst[j]);
        assert!(
            worst[j] < 1e-12,
            "{} disagrees by {:.3e} between the host closed form and the kernel",
            names[j],
            worst[j]
        );
    }
}

/// **Two identical runs of a transitional `correct` produce identical bits.**
///
/// SPEC-LIT §88.4: the one loop in the model has a fixed trip count, so
/// nothing here is iteration-order dependent. This is the check that says so
/// on the whole model rather than on the loop.
#[test]
fn a_transitional_correct_is_bitwise_repeatable() {
    let Some(gpu) = gpu() else { return };
    let hm = block(6);
    let mesh = crate::mesh::GpuMesh::upload(&gpu, &hm).expect("mesh");
    let n = hm.n_cells;
    let wf = crate::field_setup::WallFaces::none(hm.n_boundary_faces);
    let u = GpuVectorField::zeros(&gpu, &mesh, "U").expect("U");
    let phi = GpuSurfaceScalarField::zeros(&gpu, &mesh, "phi").expect("phi");
    let flow = FlowState::new(&u, &phi, 1e-5);
    let ctrl = crate::io::case::TurbulenceControls {
        ddt: crate::timescheme::DdtScheme::Euler,
        steady: false,
        delta_t: 1e-3,
        ..Default::default()
    };

    let mut wy: DevBuf<Scalar> = gpu.zeros(n).expect("y");
    gpu.write(&mut wy, &vec![0.02 as Scalar; n]).expect("write");

    let run = || -> Vec<Vec<u64>> {
        let mut m = crate::models::KOmegaSst::new(
            &gpu,
            &hm,
            &mesh,
            Default::default(),
            ctrl,
            crate::wallfunctions::WallFunctionCoeffs::default(),
            &wf,
            &wy,
        )
        .expect("sst");
        gpu.write(&mut m.k_mut().f, &vec![0.02 as Scalar; n]).expect("k");
        gpu.write(&mut m.omega_mut().f, &vec![80.0 as Scalar; n]).expect("omega");
        let mut lm =
            LangtryMenter::new(&gpu, &mesh, LmCoeffs::default(), LmControls::default(), &wy)
                .expect("lm");
        gpu.write(&mut lm.gamma_mut().f, &vec![0.3 as Scalar; n]).expect("gamma");
        gpu.write(&mut lm.re_thetat_mut().f, &vec![250.0 as Scalar; n]).expect("rtt");
        lm.initialise(&gpu, &mesh).expect("lm init");
        m.set_transition(Some(lm)).expect("attach");
        m.initialise(&gpu, &flow).expect("init");
        for _ in 0..3 {
            m.correct(&gpu, &flow).expect("correct");
        }
        let bits = |f: &crate::field::GpuScalarField| -> Vec<u64> {
            gpu.download(&f.f).expect("read").iter().map(|v: &Scalar| v.to_bits()).collect()
        };
        let lm = m.transition().expect("attached");
        vec![bits(m.k()), bits(m.omega()), bits(m.nut()), bits(lm.gamma()), bits(lm.re_thetat())]
    };

    let a = run();
    let b = run();
    for (i, name) in ["k", "omega", "nut", "gamma", "ReThetat"].iter().enumerate() {
        assert_eq!(a[i], b[i], "{name} is not bitwise repeatable");
    }
    println!("  a transitional correct repeats bitwise on all five fields, {n} cells");
}

/// **SPEC-LIT §89.4 row 17: the intermittency reaches the `k` equation, and
/// this is the pair that says so.**
///
/// Gate 88-R shows that `gamma_eff = 1` leaves SST bitwise unmoved. On its
/// own that is also what a coupling that had been wired to nothing would
/// show. This is the other half: two runs identical in every byte but the
/// value `gamma` is frozen at, REQUIRED to produce a different `k`.
#[test]
fn the_intermittency_reaches_the_k_equation() {
    let Some(gpu) = gpu() else { return };
    let hm = block(6);
    let mesh = crate::mesh::GpuMesh::upload(&gpu, &hm).expect("mesh");
    let n = hm.n_cells;
    let wf = crate::field_setup::WallFaces::none(hm.n_boundary_faces);
    let u = GpuVectorField::zeros(&gpu, &mesh, "U").expect("U");
    let phi = GpuSurfaceScalarField::zeros(&gpu, &mesh, "phi").expect("phi");
    let flow = FlowState::new(&u, &phi, 1e-5);
    let ctrl = crate::io::case::TurbulenceControls {
        ddt: crate::timescheme::DdtScheme::Euler,
        steady: false,
        delta_t: 1e-3,
        ..Default::default()
    };
    let mut wy: DevBuf<Scalar> = gpu.zeros(n).expect("y");
    gpu.write(&mut wy, &vec![0.05 as Scalar; n]).expect("write");

    let run = |frozen: Scalar| -> Vec<u64> {
        let mut m = crate::models::KOmegaSst::new(
            &gpu,
            &hm,
            &mesh,
            Default::default(),
            ctrl,
            crate::wallfunctions::WallFunctionCoeffs::default(),
            &wf,
            &wy,
        )
        .expect("sst");
        gpu.write(&mut m.k_mut().f, &vec![0.05 as Scalar; n]).expect("k");
        gpu.write(&mut m.omega_mut().f, &vec![50.0 as Scalar; n]).expect("omega");
        let coeffs = LmCoeffs { gamma_min: frozen, gamma_max: frozen, ..Default::default() };
        let mut lm =
            LangtryMenter::new(&gpu, &mesh, coeffs, LmControls::default(), &wy).expect("lm");
        gpu.write(&mut lm.gamma_mut().f, &vec![frozen; n]).expect("gamma");
        gpu.write(&mut lm.re_thetat_mut().f, &vec![300.0 as Scalar; n]).expect("rtt");
        lm.initialise(&gpu, &mesh).expect("lm init");
        m.set_transition(Some(lm)).expect("attach");
        m.initialise(&gpu, &flow).expect("init");
        m.correct(&gpu, &flow).expect("correct");
        gpu.download(&m.k().f)
            .expect("read")
            .iter()
            .map(|v: &Scalar| v.to_bits())
            .collect()
    };

    let full = run(1.0);
    let part = run(0.3);
    let differ = full.iter().zip(&part).filter(|(a, b)| a != b).count();
    assert!(
        differ > 0,
        "SPEC-LIT 89.4 row 17: freezing gamma at 0.3 instead of 1.0 left k \
         bit-identical on all {n} cells - the intermittency does not reach \
         the k equation"
    );
    println!("  89.4 row 17: gamma 1.0 vs 0.3 moves k on {differ} of {n} cells after one correct");
}

/// **A transitional run writes four fields, and a pure SST run writes two.**
///
/// SPEC-LIT §89.1. A writer that emitted only `k` and `omega` would leave a
/// restart unable to reproduce the run it restarted from - the intermittency
/// carries the whole state of the transition.
#[test]
fn attaching_the_model_grows_the_written_field_set() {
    let Some(gpu) = gpu() else { return };
    let hm = block(4);
    let mesh = crate::mesh::GpuMesh::upload(&gpu, &hm).expect("mesh");
    let n = hm.n_cells;
    let wf = crate::field_setup::WallFaces::none(hm.n_boundary_faces);
    let y: DevBuf<Scalar> = gpu.zeros(n).expect("y");

    let mut m = crate::models::KOmegaSst::new(
        &gpu,
        &hm,
        &mesh,
        Default::default(),
        crate::io::case::TurbulenceControls::default(),
        crate::wallfunctions::WallFunctionCoeffs::default(),
        &wf,
        &y,
    )
    .expect("sst");
    let names: Vec<&str> = m.named_fields().iter().map(|(n, _)| *n).collect();
    assert_eq!(names, vec!["k", "omega", "nut"]);

    let lm = LangtryMenter::new(&gpu, &mesh, LmCoeffs::default(), LmControls::default(), &y)
        .expect("lm");
    m.set_transition(Some(lm)).expect("attach");
    let names: Vec<&str> = m.named_fields().iter().map(|(n, _)| *n).collect();
    assert_eq!(names, vec!["k", "omega", "nut", "gamma", "ReThetat"]);
    let names: Vec<&str> = m.named_fields_mut().iter().map(|(n, _)| *n).collect();
    assert_eq!(names, vec!["k", "omega", "nut", "gamma", "ReThetat"]);
}

/// **A transition model and a DES hybrid cannot both be attached, and the
/// refusal names both and says which buffer they fight over.**
#[test]
fn a_hybrid_and_a_transition_model_together_are_refused_by_name() {
    let Some(gpu) = gpu() else { return };
    let hm = block(4);
    let mesh = crate::mesh::GpuMesh::upload(&gpu, &hm).expect("mesh");
    let n = hm.n_cells;
    let wf = crate::field_setup::WallFaces::none(hm.n_boundary_faces);
    let y: DevBuf<Scalar> = gpu.zeros(n).expect("y");
    let gy: DevBuf<Vec3> = gpu.zeros(n).expect("grad y");

    let mut m = crate::models::KOmegaSst::new(
        &gpu,
        &hm,
        &mesh,
        Default::default(),
        crate::io::case::TurbulenceControls::default(),
        crate::wallfunctions::WallFunctionCoeffs::default(),
        &wf,
        &y,
    )
    .expect("sst");

    let des = crate::models::des::DesLengthScale::new(
        &gpu,
        &mesh,
        &y,
        &gy,
        crate::models::des::DesBranch::Ddes,
        crate::models::des::HybridDelta::MaxEdge,
        crate::models::des::HybridBackground::Sst,
        crate::models::des::DesCoeffs::sst(),
    )
    .expect("des");
    m.set_des(Some(des));

    let lm = LangtryMenter::new(&gpu, &mesh, LmCoeffs::default(), LmControls::default(), &y)
        .expect("lm");
    let e = m.set_transition(Some(lm)).expect_err("the combination must be refused");
    let s = e.to_string();
    assert!(s.contains("kOmegaSSTLM"), "{s}");
    assert!(s.contains("sstKSources"), "the refusal does not name the buffer: {s}");
    assert!(s.contains("88.9"), "the refusal does not cite the section: {s}");
}
