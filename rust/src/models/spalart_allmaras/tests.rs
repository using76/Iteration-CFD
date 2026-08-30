// meteor-cfd - Copyright (c) 2026 주식회사 메테오시뮬레이션 (Meteo Simulation Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.
//
// SPEC-LIT §56.10's table, one test per row.
// No GPL-licensed source was consulted.

#![allow(clippy::float_cmp)]

use super::*;
use crate::field::{GpuSurfaceScalarField, GpuVectorField};
use crate::mesh::HostMesh;
use crate::turbulence::{strain_rate_mag, TurbKernels};
use crate::Tensor;

fn gpu() -> Option<Gpu> {
    Gpu::new(0).ok()
}

fn quiet_box() -> HostMesh {
    let (mut m, points, faces) =
        crate::mesh::topology::tests::box_mesh([4, 4, 4], Vec3::new(0.25, 0.25, 0.25));
    m.compute_geometry(&points, &faces).expect("geometry");
    m.build_cell_face_maps();
    m
}

// ======================================================================
//  The two numbers the TMR publishes - SPEC-LIT §56.7, §56.10
// ======================================================================

/// **The one place in §56 where a published number is reproduced without a
/// flow solve.**
///
/// The TMR states the recommended far-field range as `nu~ = 3 nu` to `5 nu`
/// and, usefully, says what it means for the eddy viscosity: `nu_t/nu`
/// between `0.210438` and `1.294234`. Those are `chi f_v1(chi)` at the two
/// ends with `c_v1 = 7.1`, to six figures, and reproducing them pins `c_v1`
/// and the whole of `f_v1` at one stroke.
#[test]
fn the_tmr_far_field_eddy_viscosity_ratios() {
    let c = SaCoeffs::default();
    for (chi, want) in [(3.0 as Scalar, 0.210438 as Scalar), (5.0, 1.294234)] {
        let got = chi * fv1(chi, c.cv1);
        // The TMR prints six decimals, so the only statement those digits
        // support is that OUR value rounds to THEIRS. A `1e-6` RELATIVE
        // tolerance is tighter than the published precision itself at 0.21 -
        // the first draft of this test used one and failed on a correct
        // implementation, which is why the gate is stated this way.
        let rounded = (got * 1.0e6).round() / 1.0e6;
        assert_eq!(
            rounded.to_bits(),
            want.to_bits(),
            "TMR nu_t/nu at chi = {chi}: {got} rounds to {rounded}, published {want}"
        );
    }
}

/// `f_v1 -> 1` like `chi^-3` and `f_v2 -> 0` like `chi^-1` - measured as
/// RATES, because a wrong exponent is what a single-point check misses.
#[test]
fn the_two_viscous_functions_approach_their_limits_at_the_published_rates() {
    let c = SaCoeffs::default();
    let rate = |f: &dyn Fn(Scalar) -> Scalar| {
        let a = f(1.0e4).abs();
        let b = f(1.0e5).abs();
        (a / b).log10()
    };
    let r1 = rate(&|chi| 1.0 - fv1(chi, c.cv1));
    let r2 = rate(&|chi| fv2(chi, c.cv1));
    assert!((r1 - 3.0).abs() < 0.02, "1 - f_v1 decays like chi^-{r1}, want 3");
    assert!((r2 - 1.0).abs() < 0.02, "f_v2 decays like chi^-{r2}, want 1");
}

/// `f_v2` is NEGATIVE over a range of `chi`, which is the entire reason the
/// positivity fix of (56.9) exists. The minimum is located and reported
/// rather than merely asserted to be somewhere.
#[test]
fn fv2_is_negative_and_the_minimum_is_where_it_is() {
    let c = SaCoeffs::default();
    let mut worst = (0.0 as Scalar, 0.0 as Scalar);
    let mut chi = 0.01 as Scalar;
    while chi < 100.0 {
        let v = fv2(chi, c.cv1);
        if v < worst.1 {
            worst = (chi, v);
        }
        chi *= 1.001;
    }
    assert!(
        worst.1 < -0.5,
        "f_v2 never went meaningfully negative; minimum {} at chi = {}",
        worst.1,
        worst.0
    );
    // Reported, not asserted to three figures: the location depends on c_v1,
    // and a case may change it.
    println!("f_v2 minimum {:.6} at chi = {:.4}", worst.1, worst.0);
}

// ======================================================================
//  The S~ positivity fix - SPEC-LIT (56.9), §56.10
// ======================================================================

/// C0 AND C1 at the join, which the constants `c_v2 = 0.7` and `c_v3 = 0.9`
/// are exactly what arrange (SPEC-LIT §56.3).
#[test]
fn the_stilde_fix_is_c0_and_c1_at_the_join() {
    let c = SaCoeffs::default();
    let om = 3.0 as Scalar;
    let join = -c.cv2 * om;

    // C0: both branches give (1 - c_v2) Omega = 0.3 Omega.
    let want = (1.0 - c.cv2) * om;
    let above = stilde(om, join, c.cv2, c.cv3);
    let below = stilde(om, join - 1e-12, c.cv2, c.cv3);
    assert!((above - want).abs() <= 1e-13 * om, "join from above: {above} vs {want}");
    assert!((below - want).abs() <= 1e-11 * om, "join from below: {below} vs {want}");

    // C1: one-sided finite differences, both slope 1.
    let h = 1e-6 as Scalar;
    let up = (stilde(om, join + h, c.cv2, c.cv3) - above) / h;
    let dn = (above - stilde(om, join - h, c.cv2, c.cv3)) / h;
    assert!((up - 1.0).abs() < 1e-6, "slope from above {up}");
    assert!((dn - 1.0).abs() < 1e-5, "slope from below {dn}");
}

/// `S~ -> (1 - c_v3) Omega = 0.1 Omega` as `Sbar/Omega -> -inf`, so `S~` is
/// strictly positive wherever `Omega` is - the property that makes `r` finite
/// and `f_w` real.
#[test]
fn the_stilde_fix_asymptotes_to_one_tenth_omega_and_stays_positive() {
    let c = SaCoeffs::default();
    let om = 2.5 as Scalar;
    let far = stilde(om, -1e14 * om, c.cv2, c.cv3);
    assert!(
        (far - (1.0 - c.cv3) * om).abs() <= 1e-6 * om,
        "asymptote {far} vs {}",
        (1.0 - c.cv3) * om
    );

    let mut ratio = -1.0e10 as Scalar;
    while ratio < 1.0e10 {
        let s = stilde(om, ratio * om, c.cv2, c.cv3);
        assert!(
            s > 0.0,
            "S~ = {s} is not positive at Sbar/Omega = {ratio}"
        );
        ratio *= if ratio < -1.0 { 0.5 } else { 2.0 };
        if ratio > -1.0 && ratio < 0.0 {
            ratio = 1e-6;
        }
    }
}

/// Above the join the fix is the UNMODIFIED formula, bit for bit - so it is
/// not a different model on any flow the original could handle.
#[test]
fn the_stilde_fix_is_bitwise_the_plain_form_above_the_join() {
    let c = SaCoeffs::default();
    for om in [0.0 as Scalar, 1e-8, 1.0, 1e7] {
        for f in [-0.7 as Scalar, -0.3, 0.0, 0.5, 3.0, 1e4] {
            let sbar = f * om;
            if sbar < -c.cv2 * om {
                continue;
            }
            let got = stilde(om, sbar, c.cv2, c.cv3);
            assert_eq!(
                got.to_bits(),
                (om + sbar).to_bits(),
                "not bitwise at Omega = {om}, Sbar = {sbar}"
            );
        }
    }
}

// ======================================================================
//  f_w, c_w1, and the log layer - SPEC-LIT §56.4
// ======================================================================

/// `f_w(1) = 1` EXACTLY, and `f_w` is bounded above by `65^(1/6)`.
#[test]
fn fw_of_one_is_exactly_one_and_the_supremum_is_65_to_the_sixth() {
    let c = SaCoeffs::default();
    let one = fw(1.0, c.cw2, c.cw3);
    assert!(
        (one - 1.0).abs() <= 4.0 * Scalar::EPSILON,
        "f_w(1) = {one}, want 1"
    );

    let sup = fw_supremum(c.cw3);
    assert!(
        (sup - 2.005_174_7).abs() < 1e-6,
        "f_w supremum {sup}, want 65^(1/6) = 2.0051747"
    );
    let mut r = 1e-3 as Scalar;
    while r <= c.rlim {
        assert!(fw(r, c.cw2, c.cw3) <= sup + 1e-12, "f_w({r}) exceeds its supremum");
        r *= 1.05;
    }
    assert!(fw(1e6, c.cw2, c.cw3) <= sup + 1e-9);
}

/// `c_w1 = c_b1/kappa^2 + (1 + c_b2)/sigma = 3.2390678` - SPEC-LIT (56.6).
#[test]
fn cw1_is_the_derived_constant() {
    let c = SaCoeffs::default();
    assert!(
        (c.cw1() - 3.239_067_8).abs() < 1e-6,
        "c_w1 = {}, want 3.2390678",
        c.cw1()
    );
}

/// The three functions each collapse to `1` in the log layer, exactly.
#[test]
fn r_and_g_and_fw_are_all_exactly_one_in_the_log_layer() {
    let c = SaCoeffs::default();
    let (u_tau, y) = (0.37 as Scalar, 0.021 as Scalar);
    let nut = c.kappa * u_tau * y;
    let om = u_tau / (c.kappa * y);
    // The nu -> 0 limit: f_v2 = 0, so S~ = Omega.
    let stil = om;
    let r = nut / (stil * c.kappa * c.kappa * y * y);
    assert!((r - 1.0).abs() <= 8.0 * Scalar::EPSILON, "r = {r}");
    let g = r + c.cw2 * (r.powi(6) - r);
    assert!((g - 1.0).abs() <= 8.0 * Scalar::EPSILON, "g = {g}");
    assert!(
        (fw(r, c.cw2, c.cw3) - 1.0).abs() <= 8.0 * Scalar::EPSILON,
        "f_w = {}",
        fw(r, c.cw2, c.cw3)
    );
}

/// The three terms of (56.2) at `nu~ = kappa u_tau y`, on the host.
///
/// The diffusion is exact for this profile: `(nu + nu~) dnu~/dy` has
/// derivative `kappa^2 u_tau^2` whatever `nu` is, because the `nu` part is
/// constant.
fn log_layer_terms(c: &SaCoeffs, nu: Scalar, u_tau: Scalar, y: Scalar) -> (Scalar, Scalar, Scalar) {
    log_layer_terms_with(c, c.cw1(), nu, u_tau, y)
}

fn log_layer_terms_with(
    c: &SaCoeffs,
    cw1: Scalar,
    nu: Scalar,
    u_tau: Scalar,
    y: Scalar,
) -> (Scalar, Scalar, Scalar) {
    let nut = c.kappa * u_tau * y;
    let om = u_tau / (c.kappa * y);
    let chi = nut / nu;
    let f2 = fv2(chi, c.cv1);
    let k2d2 = c.kappa * c.kappa * y * y;
    let sbar = nut * f2 / k2d2;
    let stil = stilde(om, sbar, c.cv2, c.cv3);
    let r = if stil > 0.0 {
        (nut / (stil * k2d2)).min(c.rlim)
    } else {
        c.rlim
    };
    let ft2 = c.ct3_positive() * (-c.ct4 * chi * chi).exp();

    let prod = c.cb1 * (1.0 - ft2) * stil * nut;
    let dest = (cw1 * fw(r, c.cw2, c.cw3) - (c.cb1 / (c.kappa * c.kappa)) * ft2)
        * (nut / y)
        * (nut / y);
    let diff = (1.0 + c.cb2) / c.sigma * c.kappa * c.kappa * u_tau * u_tau;
    (prod, dest, diff)
}

/// **SPEC-LIT §56.4: the log layer is an exact solution, and the definition
/// of `c_w1` is what makes it one.**
///
/// This is the gate §56.11 runs INSTEAD of the TMR flat plate, and it is
/// sharper: it holds to round-off or it does not hold, and each of `f_v2`,
/// (56.9), `r`, `g`, `f_w`, `c_b2`, `sigma` and `c_w1` moves it.
#[test]
fn the_log_layer_is_an_exact_solution_in_the_high_reynolds_limit() {
    let c = SaCoeffs::default();
    let u_tau = 0.37 as Scalar;
    for y in [1e-4 as Scalar, 1e-3, 1e-2, 1e-1] {
        // nu -> 0 is taken as chi -> inf: nu small enough that f_v2 is under
        // the tolerance, which is what the limit means numerically.
        let nu = c.kappa * u_tau * y / 1e14;
        let (p, d, f) = log_layer_terms(&c, nu, u_tau, y);
        let res = p - d + f;
        let scale = p.abs().max(d.abs()).max(f.abs());
        assert!(
            res.abs() <= 1e-12 * scale,
            "log-layer residual {res} at y = {y} (production {p}, destruction {d}, diffusion {f})"
        );
    }
}

/// And at FINITE `nu` it is approached at the published rate, `O(1/chi)` -
/// which is what separates "the functions are right" from "the functions
/// happen to cancel".
#[test]
fn the_log_layer_residual_falls_like_one_over_chi() {
    let c = SaCoeffs::default();
    let u_tau = 0.37 as Scalar;
    let y = 0.01 as Scalar;
    let residual = |chi: Scalar| {
        let nu = c.kappa * u_tau * y / chi;
        let (p, d, f) = log_layer_terms(&c, nu, u_tau, y);
        (p - d + f).abs() / (c.kappa * c.kappa * u_tau * u_tau)
    };
    let a = residual(1e3);
    let b = residual(1e4);
    let rate = (a / b).log10();
    assert!(
        (rate - 1.0).abs() < 0.05,
        "residual falls like chi^-{rate} ({a:e} -> {b:e}), want 1"
    );
}

/// **The identity is a statement about `c_w1` ALONE - a finding this test
/// made, and the sharpest argument there is for deriving that constant.**
///
/// Perturbing `c_b2`, `c_b1`, `kappa` or `sigma` does NOT break it, because
/// (56.6) moves `c_w1` with them and the balance is exact for any consistent
/// set. The first draft of this test perturbed `c_b2` and measured a residual
/// of `2e-15` - it would have passed a completely different model. What
/// breaks the identity is `c_w1` set independently, which is exactly what
/// `RAS { Cw1 ...; }` is refused for.
#[test]
fn only_an_independently_perturbed_cw1_breaks_the_log_layer_identity() {
    let c = SaCoeffs::default();
    let u_tau = 0.37 as Scalar;
    let y = 0.01 as Scalar;
    let nu = c.kappa * u_tau * y / 1e14;

    // A CONSISTENT change: c_b2 up 1%, c_w1 derived from it. Still exact.
    let consistent = SaCoeffs { cb2: c.cb2 * 1.01, ..c };
    let (p, d, f) = log_layer_terms(&consistent, nu, u_tau, y);
    let scale = p.abs().max(d.abs()).max(f.abs());
    assert!(
        (p - d + f).abs() <= 1e-12 * scale,
        "a consistent coefficient change broke the identity ({}) - (56.6) is \
         not being applied",
        p - d + f
    );

    // c_w1 alone, 1% out: the identity must move, measurably.
    let (p, d, f) = log_layer_terms_with(&c, c.cw1() * 1.01, nu, u_tau, y);
    let res = (p - d + f).abs();
    let scale = p.abs().max(d.abs()).max(f.abs());
    assert!(
        res > 1e-3 * scale,
        "a 1% independent change in c_w1 left the identity intact ({res} against \
         {scale}) - the gate would pass a wrong model"
    );
}

// ======================================================================
//  The negative continuation - SPEC-LIT §56.5
// ======================================================================

/// `f_n` is EXACTLY `1` on the positive branch, which is what lets both
/// variants run the same diffusivity kernel.
#[test]
fn fn_is_exactly_one_on_the_positive_branch() {
    for chi in [0.0 as Scalar, 1e-30, 1.0, 1e6] {
        assert_eq!(fn_(chi, 16.0).to_bits(), (1.0 as Scalar).to_bits());
    }
}

/// **SPEC-LIT (56.14), derived here rather than quoted.**
///
/// `N(x) = x^4 + x^3 - c_n1 x + c_n1` is positive for all `x > 0` at
/// `c_n1 = 16`, and first touches zero at `c_n1 = 16.457746`, at
/// `x = (1 + sqrt(10))/3`.
#[test]
fn the_cn1_bound_is_where_the_derivation_says() {
    let x = cn1_bound_x();
    let bound = cn1_bound();
    assert!(
        (bound - 16.457_756_9).abs() < 1e-6,
        "c_n1 bound {bound}, want 16.4577569"
    );

    // At the bound, N and N' vanish together.
    assert!(neg_diffusivity_numerator(x, bound).abs() < 1e-12);
    let dn = 4.0 * x * x * x + 3.0 * x * x - bound;
    assert!(dn.abs() < 1e-12, "N'(x*) = {dn}");

    // Just above, N goes negative somewhere.
    let mut worst = Scalar::INFINITY;
    let mut xx = 1e-3 as Scalar;
    while xx < 10.0 {
        worst = worst.min(neg_diffusivity_numerator(xx, bound * 1.001));
        xx *= 1.0005;
    }
    assert!(worst < 0.0, "N stayed positive above the bound: min {worst}");
}

/// At the published `c_n1 = 16` the negative branch's diffusivity is strictly
/// positive, and the margin is reported rather than described.
#[test]
fn the_negative_diffusivity_stays_positive_at_cn1_sixteen() {
    let c = SaCoeffs::default();
    let mut worst = (0.0 as Scalar, Scalar::INFINITY);
    let mut x = 1e-4 as Scalar;
    while x < 20.0 {
        let n = neg_diffusivity_numerator(x, c.cn1);
        if n < worst.1 {
            worst = (x, n);
        }
        x *= 1.0002;
    }
    assert!(
        worst.1 > 0.0,
        "nu + nu~ f_n went non-positive: N = {} at x = {}",
        worst.1,
        worst.0
    );
    println!("min N(x) = {:.6} at x = {:.5} (c_n1 = 16)", worst.1, worst.0);

    // And the direct form agrees: nu + nu~ f_n > 0 for every nu~ < 0.
    let nu = 1.5e-5 as Scalar;
    let mut chi = -1e-3 as Scalar;
    while chi > -20.0 {
        let nt = chi * nu;
        assert!(
            nu + nt * fn_(chi, c.cn1) > 0.0,
            "diffusivity non-positive at chi = {chi}"
        );
        chi *= 1.001;
    }
}

/// **`P_n >= 0` for `nu~ < 0` requires `c_t3 > 1`** - the one place the two
/// branches must not share a constant.
#[test]
fn the_negative_production_is_non_negative_only_when_ct3_exceeds_one() {
    let c = SaCoeffs::default();
    let om = 4.0 as Scalar;
    let pn = |ct3: Scalar, nt: Scalar| c.cb1 * (1.0 - ct3) * om * nt;

    let mut nt = -1e-8 as Scalar;
    while nt > -1.0 {
        assert!(pn(c.ct3, nt) >= 0.0, "P_n < 0 at nu~ = {nt} with c_t3 = {}", c.ct3);
        assert!(
            pn(0.0, nt) < 0.0,
            "c_t3 = 0 gave a non-negative P_n at nu~ = {nt}, so the gate is vacuous"
        );
        nt *= 1.5;
    }

    // And the coefficient check refuses it rather than running it.
    let mut bad = c;
    bad.variant = SaVariant::Noft2Neg;
    bad.ct3 = 0.9;
    let e = bad.check().expect_err("Ct3 <= 1 under a -neg variant must be refused");
    assert!(format!("{e}").contains("Ct3"), "message does not name Ct3: {e}");
}

/// **The C1 correction of SPEC-LIT §56.5, pinned rather than tolerated.**
///
/// Four of the five terms are C1 at `nu~ = 0` under either variant. The
/// production is C1 for the FULL model and jumps by exactly
/// `1.2 c_b1 Omega` under SA-noft2, because the positive branch's `f_t2` is
/// zero there while the negative branch's `c_t3` is `1.2`.
#[test]
fn the_production_slope_jump_at_zero_is_exactly_1_2_cb1_omega() {
    let c = SaCoeffs::default();
    let om = 3.0 as Scalar;

    // SA-noft2: positive slope c_b1 Omega, negative slope -0.2 c_b1 Omega.
    let noft2 = SaCoeffs { variant: SaVariant::Noft2Neg, ..c };
    let up = noft2.cb1 * (1.0 - noft2.ct3_positive()) * om;
    let dn = noft2.cb1 * (1.0 - noft2.ct3) * om;
    let jump = up - dn;
    assert!(
        (jump - 1.2 * c.cb1 * om).abs() <= 1e-14 * om,
        "production slope jump {jump}, want 1.2 c_b1 Omega = {}",
        1.2 * c.cb1 * om
    );

    // The FULL model is C1 there: both slopes are c_b1(1 - c_t3) Omega.
    let full = SaCoeffs { variant: SaVariant::Ft2Neg, ..c };
    let up = full.cb1 * (1.0 - full.ct3_positive()) * om;
    let dn = full.cb1 * (1.0 - full.ct3) * om;
    assert_eq!(up.to_bits(), dn.to_bits(), "the full model is not C1 after all");
}

/// The diffusivity is C1 at `nu~ = 0`: value `nu` and slope `1` from both
/// sides, because `f_n(0) = 1` and `d(nu~ f_n)/dnu~ = 1` there.
#[test]
fn the_diffusivity_is_c1_at_zero() {
    let c = SaCoeffs::default();
    let nu = 1.5e-5 as Scalar;
    let g = |nt: Scalar| nu + nt * fn_(nt / nu, c.cn1);
    let h = nu * 1e-6;
    let up = (g(h) - g(0.0)) / h;
    let dn = (g(0.0) - g(-h)) / h;
    assert!((g(0.0) - nu).abs() <= Scalar::EPSILON * nu);
    assert!((up - 1.0).abs() < 1e-9, "slope from above {up}");
    assert!((dn - 1.0).abs() < 1e-9, "slope from below {dn}");
}

// ======================================================================
//  Device: the invariants, and the kernels
// ======================================================================

/// `Omega`, `S` and `F` are three DIFFERENT numbers, and `F^2 = (S^2 +
/// Omega^2)/2` exactly - SPEC-LIT (56.8), checked against a direct
/// nine-component sum.
#[test]
fn the_three_gradient_invariants_and_the_identity_between_them() -> Result<()> {
    let Some(gpu) = gpu() else {
        return Ok(());
    };
    let k = TurbKernels::new(&gpu)?;
    let t = |xx, xy, xz, yx, yy, yz, zx, zy, zz| Tensor {
        xx, xy, xz, yx, yy, yz, zx, zy, zz,
    };
    let cases: Vec<(&str, Tensor)> = vec![
        ("simple shear", t(0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0)),
        ("solid-body rotation", t(0.0, 5.0, 0.0, -5.0, 0.0, 0.0, 0.0, 0.0, 0.0)),
        ("plane strain", t(2.0, 0.0, 0.0, 0.0, -2.0, 0.0, 0.0, 0.0, 0.0)),
        ("general", t(0.7, -1.3, 0.4, 2.1, -0.2, 0.9, -0.6, 1.7, -0.5)),
    ];
    let n = cases.len();
    let g = gpu.upload(&cases.iter().map(|(_, t)| *t).collect::<Vec<_>>())?;
    let mut om: DevBuf<Scalar> = gpu.zeros(n)?;
    let mut s: DevBuf<Scalar> = gpu.zeros(n)?;
    let mut f: DevBuf<Scalar> = gpu.zeros(n)?;
    vorticity_mag(&gpu, &k, &mut om, &g, n)?;
    strain_rate_mag(&gpu, &k, &mut s, &g, n)?;
    crate::turbulence::grad_frobenius(&gpu, &k, &mut f, &g, n)?;
    let (om, s, f) = (gpu.download(&om)?, gpu.download(&s)?, gpu.download(&f)?);

    for (i, (name, gg)) in cases.iter().enumerate() {
        let direct = (gg.xx * gg.xx
            + gg.xy * gg.xy
            + gg.xz * gg.xz
            + gg.yx * gg.yx
            + gg.yy * gg.yy
            + gg.yz * gg.yz
            + gg.zx * gg.zx
            + gg.zy * gg.zy
            + gg.zz * gg.zz)
            .sqrt();
        assert!(
            (f[i] - direct).abs() <= 1e-14 * direct.max(1.0),
            "{name}: F = {} vs direct {direct}",
            f[i]
        );
        let ident = ((s[i] * s[i] + om[i] * om[i]) / 2.0).sqrt();
        assert!(
            (f[i] - ident).abs() <= 1e-14 * f[i].max(1.0),
            "{name}: F = {} but sqrt((S^2+Omega^2)/2) = {ident}",
            f[i]
        );
    }

    // Solid-body rotation at rate 5: Omega = 2*5 = 10, S = 0.
    assert!((om[1] - 10.0).abs() < 1e-12, "solid-body Omega = {}", om[1]);
    assert!(s[1].abs() < 1e-12, "solid-body S = {}", s[1]);
    // Plane strain: Omega = 0, S != 0. So Omega and S are not the same
    // number - which a shear-only test could never show.
    assert!(om[2].abs() < 1e-12, "plane-strain Omega = {}", om[2]);
    assert!(s[2] > 1.0, "plane-strain S = {}", s[2]);
    // Simple shear: the ONE state where S = Omega = F.
    assert!((s[0] - om[0]).abs() < 1e-12 && (s[0] - f[0]).abs() < 1e-12);
    Ok(())
}

/// The device `nu_t` reproduces the host `f_v1`, and is EXACTLY zero for
/// `nu~ < 0` - SPEC-LIT (56.13).
#[test]
fn the_device_nut_matches_fv1_and_is_exactly_zero_below_zero() -> Result<()> {
    let Some(gpu) = gpu() else {
        return Ok(());
    };
    let kern = SaKernels::new(&gpu)?;
    let c = SaCoeffs::default();
    let nu = 1.5e-5 as Scalar;
    let chis: Vec<Scalar> = vec![-10.0, -1.0, -1e-9, 0.0, 1e-3, 0.5, 3.0, 5.0, 50.0, 1e4];
    let nt: Vec<Scalar> = chis.iter().map(|c| c * nu).collect();
    let n = nt.len();
    let d = gpu.upload(&nt)?;
    let mut out: DevBuf<Scalar> = gpu.zeros(n)?;
    sa_nut(&gpu, &kern, &mut out, &d, nu, c.cv1, 1e30, n)?;
    let got = gpu.download(&out)?;

    for (i, &chi) in chis.iter().enumerate() {
        if chi < 0.0 {
            assert_eq!(got[i].to_bits(), (0.0 as Scalar).to_bits(), "chi = {chi}");
        } else {
            let want = nt[i] * fv1(chi, c.cv1);
            assert!(
                (got[i] - want).abs() <= 1e-15 * want.max(1e-30),
                "chi = {chi}: device {} vs host {want}",
                got[i]
            );
        }
    }
    Ok(())
}

/// The device source kernel reproduces the host closed forms, term by term,
/// on both branches and across the (56.9) join.
#[test]
fn the_device_sources_match_the_host_closed_forms() -> Result<()> {
    let Some(gpu) = gpu() else {
        return Ok(());
    };
    let kern = SaKernels::new(&gpu)?;
    let c = SaCoeffs { variant: SaVariant::Ft2Neg, ..SaCoeffs::default() };
    let nu = 1.5e-5 as Scalar;

    // A spread that crosses zero, crosses the (56.9) join (which needs a
    // strongly negative Sbar, i.e. a large nu~ with f_v2 < 0 and a small
    // Omega), and reaches r_lim.
    let states: Vec<(Scalar, Scalar, Scalar)> = vec![
        (3.0 * nu, 10.0, 1e-3),
        (-2.0 * nu, 10.0, 1e-3),
        (1e-3, 1e-4, 1e-3),
        (1e-2, 1.0, 5e-4),
        (5.0 * nu, 0.0, 1e-2),
        (-1e-4, 3.0, 2e-3),
        (1e-6, 100.0, 1e-2),
    ];
    let n = states.len();
    let nt = gpu.upload(&states.iter().map(|s| s.0).collect::<Vec<_>>())?;
    let om = gpu.upload(&states.iter().map(|s| s.1).collect::<Vec<_>>())?;
    let dd = gpu.upload(&states.iter().map(|s| s.2).collect::<Vec<_>>())?;
    let gr = gpu.upload(&vec![Vec3::new(1.0, -2.0, 0.5); n])?;
    let mut su: DevBuf<Scalar> = gpu.zeros(n)?;
    let mut sp: DevBuf<Scalar> = gpu.zeros(n)?;
    let mut susp: DevBuf<Scalar> = gpu.zeros(n)?;
    sa_sources(
        &gpu, &kern, &mut su, &mut sp, &mut susp, &nt, &gr, &om, &dd, &dd, nu, &c, n,
    )?;
    let (su, sp, susp) = (gpu.download(&su)?, gpu.download(&sp)?, gpu.download(&susp)?);

    for (i, &(ntv, omv, dv)) in states.iter().enumerate() {
        // su = (c_b2/sigma)|grad nu~|^2
        let want_su = (c.cb2 / c.sigma) * (1.0 + 4.0 + 0.25);
        assert!(
            (su[i] - want_su).abs() <= 1e-14 * want_su,
            "su[{i}] = {} vs {want_su}",
            su[i]
        );
        assert_eq!(sp[i].to_bits(), (0.0 as Scalar).to_bits(), "sp[{i}] is not zero");

        let a = if ntv >= 0.0 {
            let chi = ntv / nu;
            let f1 = fv1(chi, c.cv1);
            let f2 = 1.0 - chi / (1.0 + chi * f1);
            let k2d2 = c.kappa * c.kappa * dv * dv;
            let stil = stilde(omv, ntv * f2 / k2d2, c.cv2, c.cv3);
            let r = if stil > 0.0 {
                (ntv / (stil * k2d2)).min(c.rlim)
            } else {
                c.rlim
            };
            let ft2 = c.ct3_positive() * (-c.ct4 * chi * chi).exp();
            c.cb1 * (1.0 - ft2) * stil
                - (c.cw1() * fw(r, c.cw2, c.cw3) - (c.cb1 / (c.kappa * c.kappa)) * ft2) * ntv
                    / (dv * dv)
        } else {
            c.cb1 * (1.0 - c.ct3) * omv + c.cw1() * ntv / (dv * dv)
        };
        assert!(
            (susp[i] + a).abs() <= 1e-11 * a.abs().max(1.0),
            "susp[{i}] = {} but -A = {}",
            susp[i],
            -a
        );
    }
    Ok(())
}

/// **The TMR's `r = 10` rule for the `Omega == 0` corner**, where `S~` is
/// exactly zero and the quotient is `0/0`.
#[test]
fn r_is_the_limit_when_omega_and_stilde_are_both_zero() -> Result<()> {
    let Some(gpu) = gpu() else {
        return Ok(());
    };
    let kern = SaKernels::new(&gpu)?;
    let c = SaCoeffs::default();
    let nu = 1.5e-5 as Scalar;

    // Omega = 0 with Sbar < 0 makes (56.9)'s second branch give S~ = 0
    // exactly. f_v2 < 0 needs chi in the negative range, which chi = 3 is.
    let nt = 3.0 * nu;
    assert!(fv2(nt / nu, c.cv1) < 0.0, "the state does not exercise the corner");

    let d = 1e-3 as Scalar;
    let ntb = gpu.upload(&[nt])?;
    let omb = gpu.upload(&[0.0 as Scalar])?;
    let db = gpu.upload(&[d])?;
    let gr = gpu.upload(&[Vec3::ZERO])?;
    let mut su: DevBuf<Scalar> = gpu.zeros(1)?;
    let mut sp: DevBuf<Scalar> = gpu.zeros(1)?;
    let mut susp: DevBuf<Scalar> = gpu.zeros(1)?;
    sa_sources(
        &gpu, &kern, &mut su, &mut sp, &mut susp, &ntb, &gr, &omb, &db, &db, nu, &c, 1,
    )?;
    let got = gpu.download(&susp)?[0];
    assert!(got.is_finite(), "susp is {got} - the 0/0 corner produced a non-number");

    // With r = r_lim the destruction coefficient is c_w1 f_w(10) nu~/d^2 and
    // the production is zero (S~ = 0), so -A is exactly that.
    let want = c.cw1() * fw(c.rlim, c.cw2, c.cw3) * nt / (d * d);
    assert!(
        (got - want).abs() <= 1e-10 * want,
        "susp = {got}, want c_w1 f_w(r_lim) nu~/d^2 = {want} (r was not set to r_lim)"
    );
    Ok(())
}

/// The device log-layer terms reproduce §56.4's identity - the SAME balance,
/// through the kernels rather than the host closed forms.
#[test]
fn the_device_reproduces_the_log_layer_identity() -> Result<()> {
    let Some(gpu) = gpu() else {
        return Ok(());
    };
    let kern = SaKernels::new(&gpu)?;
    let c = SaCoeffs::default();
    let u_tau = 0.37 as Scalar;
    let ys: Vec<Scalar> = vec![1e-4, 1e-3, 1e-2, 1e-1];
    let n = ys.len();
    let nt: Vec<Scalar> = ys.iter().map(|y| c.kappa * u_tau * y).collect();
    let om: Vec<Scalar> = ys.iter().map(|y| u_tau / (c.kappa * y)).collect();
    // chi -> inf, taken numerically.
    let nu = c.kappa * u_tau * ys[0] / 1e14;

    let ntb = gpu.upload(&nt)?;
    let omb = gpu.upload(&om)?;
    let db = gpu.upload(&ys)?;
    let mut p: DevBuf<Scalar> = gpu.zeros(n)?;
    let mut d: DevBuf<Scalar> = gpu.zeros(n)?;
    sa_log_layer_terms(&gpu, &kern, &mut p, &mut d, &ntb, &omb, &db, nu, &c, n)?;
    let (p, d) = (gpu.download(&p)?, gpu.download(&d)?);

    let diff = (1.0 + c.cb2) / c.sigma * c.kappa * c.kappa * u_tau * u_tau;
    for i in 0..n {
        let res = p[i] - d[i] + diff;
        let scale = p[i].abs().max(d[i]).max(diff);
        assert!(
            res.abs() <= 1e-11 * scale,
            "device log-layer residual {res} at y = {} (P {} D {} diff {diff})",
            ys[i],
            p[i],
            d[i]
        );
    }
    Ok(())
}

// ======================================================================
//  Model level - passivity, the pair, determinism
// ======================================================================

struct Rig {
    hm: HostMesh,
}

fn rig() -> Rig {
    Rig { hm: quiet_box() }
}

#[allow(clippy::too_many_arguments)]
fn run_sa(
    gpu: &Gpu,
    hm: &HostMesh,
    coeffs: SaCoeffs,
    seed: &[Scalar],
    steps: usize,
) -> Result<Vec<Scalar>> {
    let mesh = GpuMesh::upload(gpu, hm)?;
    let no_walls = crate::field_setup::WallFaces::none(hm.n_boundary_faces);
    let no_rough = crate::field_setup::NutRoughness::none(hm.n_boundary_faces);
    let ctrl = TurbulenceControls {
        steady: false,
        delta_t: 1e-3,
        k_relax: 1.0,
        eps_relax: 1.0,
        ..Default::default()
    };
    let u = GpuVectorField::zeros(gpu, &mesh, "U")?;
    let phi = GpuSurfaceScalarField::zeros(gpu, &mesh, "phi")?;
    let flow = FlowState::new(&u, &phi, 1.5e-5);
    let y: DevBuf<Scalar> = gpu.upload(&vec![0.05 as Scalar; hm.n_cells])?;

    let mut m = SpalartAllmaras::new(
        gpu,
        hm,
        &mesh,
        coeffs,
        ctrl,
        WallFunctionCoeffs::default(),
        &no_walls,
        &no_rough,
        &y,
    )?;
    let fld = crate::field_ops::FieldKernels::new(gpu)?;
    let src = gpu.upload(seed)?;
    crate::field_ops::copy_field(gpu, &fld, &mut m.nu_tilda_mut().f, &src, hm.n_cells)?;
    m.initialise(gpu, &flow)?;
    for _ in 0..steps {
        m.correct(gpu, &flow)?;
    }
    gpu.download(&m.nu_tilda().f)
}

/// `run_sa`, returning `(nut, nuTilda)` instead of `nuTilda` alone.
fn run_sa_nut(
    gpu: &Gpu,
    hm: &HostMesh,
    coeffs: SaCoeffs,
    seed: &[Scalar],
    steps: usize,
) -> Result<(Vec<Scalar>, Vec<Scalar>)> {
    let mesh = GpuMesh::upload(gpu, hm)?;
    let no_walls = crate::field_setup::WallFaces::none(hm.n_boundary_faces);
    let no_rough = crate::field_setup::NutRoughness::none(hm.n_boundary_faces);
    let ctrl = TurbulenceControls {
        steady: false,
        delta_t: 1e-3,
        k_relax: 1.0,
        eps_relax: 1.0,
        ..Default::default()
    };
    let u = GpuVectorField::zeros(gpu, &mesh, "U")?;
    let phi = GpuSurfaceScalarField::zeros(gpu, &mesh, "phi")?;
    let flow = FlowState::new(&u, &phi, 1.5e-5);
    let y: DevBuf<Scalar> = gpu.upload(&vec![0.05 as Scalar; hm.n_cells])?;
    let mut m = SpalartAllmaras::new(
        gpu,
        hm,
        &mesh,
        coeffs,
        ctrl,
        WallFunctionCoeffs::default(),
        &no_walls,
        &no_rough,
        &y,
    )?;
    let fld = crate::field_ops::FieldKernels::new(gpu)?;
    let src = gpu.upload(seed)?;
    crate::field_ops::copy_field(gpu, &fld, &mut m.nu_tilda_mut().f, &src, hm.n_cells)?;
    m.initialise(gpu, &flow)?;
    for _ in 0..steps {
        m.correct(gpu, &flow)?;
    }
    Ok((gpu.download(&m.nut().f)?, gpu.download(&m.nu_tilda().f)?))
}

/// **SPEC-LIT §56.5's passivity property, measured bitwise.**
///
/// On a field where `nu~ >= 0` everywhere, the negative continuation must be
/// bit for bit the positive-only model: SA-neg's design goal is to be
/// PASSIVE on a resolved mesh, and a passivity failure is otherwise
/// invisible.
#[test]
fn the_negative_variant_is_bitwise_the_positive_one_where_nothing_goes_negative()
-> Result<()> {
    let Some(gpu) = gpu() else {
        return Ok(());
    };
    let r = rig();
    let seed: Vec<Scalar> = (0..r.hm.n_cells)
        .map(|c| 1e-4 + 1e-6 * c as Scalar)
        .collect();
    let a = run_sa(&gpu, &r.hm, SaCoeffs::default(), &seed, 3)?;
    let b = run_sa(
        &gpu,
        &r.hm,
        SaCoeffs { variant: SaVariant::Noft2Neg, ..SaCoeffs::default() },
        &seed,
        3,
    )?;
    for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
        assert!(
            x.is_finite() && *x > 0.0,
            "cell {i} went non-positive ({x}), so the passivity test is vacuous"
        );
        assert_eq!(
            x.to_bits(),
            y.to_bits(),
            "cell {i}: SA-noft2 {x} vs SA-noft2-neg {y} - the negative branch \
             is NOT passive on a non-negative field"
        );
    }
    Ok(())
}

/// **The §13.4.1 pair for `variant`: the same case with one cell seeded
/// negative MUST give different output.**
#[test]
fn the_two_variants_differ_when_a_cell_is_seeded_negative() -> Result<()> {
    let Some(gpu) = gpu() else {
        return Ok(());
    };
    let r = rig();
    let mut seed: Vec<Scalar> = vec![1e-4; r.hm.n_cells];
    seed[r.hm.n_cells / 2] = -3e-4;
    let a = run_sa(&gpu, &r.hm, SaCoeffs::default(), &seed, 2)?;
    let b = run_sa(
        &gpu,
        &r.hm,
        SaCoeffs { variant: SaVariant::Noft2Neg, ..SaCoeffs::default() },
        &seed,
        2,
    )?;
    let diff = a
        .iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0 as Scalar, Scalar::max);
    assert!(
        diff > 1e-12,
        "`variant noft2` and `variant noft2-neg` produced identical fields on a \
         mesh with a negative cell (max difference {diff}) - the setting is \
         being read and thrown away (SPEC-LIT §13.4.1)"
    );
    Ok(())
}

/// The §13.4.1 pair for `Cv1` - and the field it has to be read on.
///
/// **`c_v1` reaches `nu_t` and NOTHING else.** It does not appear in the
/// `nu~` equation at all: SA transports `nu~`, and `f_v1` only turns it into
/// an eddy viscosity for the momentum equation. The first draft of this test
/// compared `nuTilda`, found the two runs bit-identical - correctly - and
/// would have reported a real setting as inert. The pair is on `nut`.
#[test]
fn a_changed_cv1_changes_nut_and_leaves_nutilda_alone() -> Result<()> {
    let Some(gpu) = gpu() else {
        return Ok(());
    };
    let r = rig();
    let seed: Vec<Scalar> = vec![1e-4; r.hm.n_cells];
    let a = run_sa_nut(&gpu, &r.hm, SaCoeffs::default(), &seed, 2)?;
    let b = run_sa_nut(
        &gpu,
        &r.hm,
        SaCoeffs { cv1: 8.0, ..SaCoeffs::default() },
        &seed,
        2,
    )?;
    let diff = a
        .0
        .iter()
        .zip(b.0.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0 as Scalar, Scalar::max);
    assert!(
        diff > 1e-12,
        "`Cv1` was read and thrown away: nut identical (max difference {diff})"
    );
    for (i, (x, y)) in a.1.iter().zip(b.1.iter()).enumerate() {
        assert_eq!(
            x.to_bits(),
            y.to_bits(),
            "cell {i}: c_v1 moved nuTilda, which it does not appear in"
        );
    }
    Ok(())
}

/// Two runs of the same build produce bit-identical fields - SPEC-LIT §56.9.
#[test]
fn two_runs_are_bitwise_identical() -> Result<()> {
    let Some(gpu) = gpu() else {
        return Ok(());
    };
    let r = rig();
    let seed: Vec<Scalar> = (0..r.hm.n_cells)
        .map(|c| 1e-4 + 3e-6 * (c % 7) as Scalar)
        .collect();
    let a = run_sa(&gpu, &r.hm, SaCoeffs::default(), &seed, 4)?;
    let b = run_sa(&gpu, &r.hm, SaCoeffs::default(), &seed, 4)?;
    for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
        assert_eq!(x.to_bits(), y.to_bits(), "cell {i}: {x} vs {y}");
    }
    Ok(())
}

/// `cuda/sa.cu` calls no atomic of any kind - SPEC-LIT §56.9.
///
/// Comments are stripped first, because the file's own header says the word
/// while promising not to use one, and a test that could not tell those apart
/// would be satisfied by deleting a comment.
#[test]
fn the_sa_kernels_contain_no_atomics() {
    let code = strip_c_comments(include_str!("../../../cuda/sa.cu"));
    assert!(
        !code.contains("atomic"),
        "cuda/sa.cu calls an atomic; SPEC-LIT §56.9 forbids one"
    );
}

/// Remove `//` and block comments so an "atomic" audit reads CODE.
pub(crate) fn strip_c_comments(src: &str) -> String {
    let mut out = String::with_capacity(src.len());
    let b = src.as_bytes();
    let mut i = 0;
    while i < b.len() {
        if i + 1 < b.len() && b[i] == b'/' && b[i + 1] == b'/' {
            while i < b.len() && b[i] != b'\n' {
                i += 1;
            }
        } else if i + 1 < b.len() && b[i] == b'/' && b[i + 1] == b'*' {
            i += 2;
            while i + 1 < b.len() && !(b[i] == b'*' && b[i + 1] == b'/') {
                i += 1;
            }
            i = (i + 2).min(b.len());
        } else {
            out.push(b[i] as char);
            i += 1;
        }
    }
    out
}

/// The coefficient guard refuses a `c_n1` at or past (56.14)'s bound.
#[test]
fn a_cn1_past_the_bound_is_refused_by_name() {
    let c = SaCoeffs {
        variant: SaVariant::Noft2Neg,
        cn1: 17.0,
        ..SaCoeffs::default()
    };
    let e = c.check().expect_err("Cn1 = 17 must be refused");
    let m = format!("{e}");
    assert!(m.contains("Cn1"), "message does not name Cn1: {m}");
    assert!(m.contains("16.4"), "message does not quote the bound: {m}");
}

/// Every variant spelling the TMR uses parses, and an unknown one does not.
#[test]
fn the_variant_names_parse_and_an_unknown_one_does_not() {
    assert_eq!(SaVariant::parse("noft2"), Some(SaVariant::Noft2));
    assert_eq!(SaVariant::parse("SA-noft2"), Some(SaVariant::Noft2));
    assert_eq!(SaVariant::parse("SA-noft2-neg"), Some(SaVariant::Noft2Neg));
    assert_eq!(SaVariant::parse("SA-neg"), Some(SaVariant::Ft2Neg));
    assert_eq!(SaVariant::parse("SA"), Some(SaVariant::Ft2));
    assert_eq!(SaVariant::parse("nofT2"), None);
    assert_eq!(SaVariant::parse(""), None);
    for v in [
        SaVariant::Noft2,
        SaVariant::Noft2Neg,
        SaVariant::Ft2,
        SaVariant::Ft2Neg,
    ] {
        assert_eq!(SaVariant::parse(v.name()), Some(v), "{} round-trips", v.name());
        assert!(SaVariant::menu().contains(&v.name()));
    }
}
