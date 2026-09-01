// meteor-cfd - Copyright (c) 2026 주식회사 메테오시뮬레이션 (Meteo Simulation Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.

//! SPEC-LIT S76.12's host half: the property correlations against their own
//! anchors and derivatives, the two saturation curves against each other, the
//! `d^2`-law slope against the textbook closed form, the steady droplet
//! temperature against the psychrometric wet bulb of S54, and the S13.4
//! contract on every setting this section adds.
//!
//! Written from the same sources as `src/parcels/evaporation.rs`; see that
//! module's header. No GPL-licensed source was consulted.

use super::*;
use crate::psychro;

/// The reference gas for most of what follows: room air, dry-ish, still.
fn air(t: Scalar, rh: Scalar) -> GasState {
    GasState {
        t,
        y_vapour: psychro::yv_from_t_rh_p(t, rh, 101_325.0),
        rho: 101_325.0 * 28.966e-3 / (R_UNIVERSAL * t),
        mu: 1.8e-5,
        k: 0.026,
        cp: 1005.0,
        u_rel: 0.0,
    }
}

fn water() -> EvaporationControls {
    EvaporationControls::default()
}

// ======================================================================
//  (76.3): the saturation curves
// ======================================================================

/// The Hyland-Wexler slope this module carries has to be the derivative of
/// the polynomial S54 evaluates - checked against a central difference of
/// `psychro::p_ws` itself, so the duplicated coefficients cannot drift.
#[test]
fn the_hyland_wexler_slope_is_the_derivative_of_psychro() {
    let l = LiquidProperties::water();
    let mut worst: Scalar = 0.0;
    for &t in &[250.0, 260.0, 273.0, 280.0, 293.15, 313.15, 353.15, 373.15, 420.0] {
        // The bar is the CENTRAL DIFFERENCE's own truncation error, not the
        // analytic derivative's: `p_ws` is an exponential, so the second-order
        // remainder is `h^2 f'''/6` and no tighter comparison is available
        // without a third correlation to compare against.
        let h = 1e-4 * t;
        let fd = (psychro::p_ws(t + h) - psychro::p_ws(t - h)) / (2.0 * h);
        let an = p_sat_derivative(SaturationCurve::HylandWexler, &l, t);
        let rel = (an - fd).abs() / fd.abs();
        worst = worst.max(rel);
        assert!(rel < 1e-5, "dp_ws/dT at {t} K: analytic {an}, difference {fd}");
    }
    println!("[76.3] Hyland-Wexler slope vs central difference: worst {worst:.3e}");
}

/// Clausius-Clapeyron is anchored so that `p_sat(T_b) = p_b` EXACTLY, which
/// is what makes the boiling-point root find return `T_b` at one atmosphere
/// and what stops the surface mole fraction from exceeding one there.
#[test]
fn the_clausius_clapeyron_curve_is_anchored_at_its_boiling_point() {
    for l in [LiquidProperties::water(), LiquidProperties::benzene()] {
        let p = p_sat(SaturationCurve::ClausiusClapeyron, &l, l.t_boil);
        assert!(
            (p - l.p_boil).abs() <= 1e-9 * l.p_boil,
            "p_sat(T_b) = {p}, p_b = {}",
            l.p_boil
        );
    }
}

/// The gap between the general curve and the water one, measured and printed
/// rather than asserted away. It is the reason S76.12 runs the wet-bulb gate
/// on both legs.
#[test]
fn the_two_saturation_curves_differ_by_a_measured_amount() {
    let l = LiquidProperties::water();
    let mut worst: Scalar = 0.0;
    println!("[76.3] p_sat, Clausius-Clapeyron against Hyland-Wexler (water):");
    for &t in &[283.15, 293.15, 298.15, 313.15, 333.15, 353.15, 373.15] {
        let cc = p_sat(SaturationCurve::ClausiusClapeyron, &l, t);
        let hw = p_sat(SaturationCurve::HylandWexler, &l, t);
        let rel = (cc - hw) / hw;
        worst = worst.max(rel.abs());
        println!("        {t:8.2} K   CC {cc:10.2} Pa   HW {hw:10.2} Pa   {:+7.2} %", 100.0 * rel);
    }
    // The two agree at the boiling point by construction and diverge below
    // it; the bar here is only that the general curve is not WILD, because
    // the number itself is the finding.
    assert!(worst < 0.25, "the two saturation curves differ by {worst}");
    println!("        worst relative gap over 10-100 C: {:.2} %", 100.0 * worst);
}

#[test]
fn the_boiling_temperature_is_the_root_of_the_saturation_curve() {
    let l = LiquidProperties::water();
    for curve in [SaturationCurve::ClausiusClapeyron, SaturationCurve::HylandWexler] {
        for &p in &[50_000.0, 101_325.0, 200_000.0] {
            let tb = boiling_temperature(curve, &l, p).expect("boiling temperature");
            let ps = p_sat(curve, &l, tb);
            assert!(
                (ps - p).abs() <= 1e-9 * p,
                "{}: p_sat({tb}) = {ps}, wanted {p}",
                curve.name()
            );
        }
    }
    // The water numbers a reader can check: 100 C at one atmosphere on the
    // general curve by construction, and Hyland-Wexler's own 99.97 C, which
    // is the IAPWS value and NOT 373.15 K.
    let tb_cc =
        boiling_temperature(SaturationCurve::ClausiusClapeyron, &l, 101_325.0).unwrap();
    let tb_hw = boiling_temperature(SaturationCurve::HylandWexler, &l, 101_325.0).unwrap();
    println!("[76.9] T_boil at 1 atm: clausiusClapeyron {tb_cc:.4} K, hylandWexler {tb_hw:.4} K");
    assert!((tb_cc - 373.15).abs() < 1e-6);
    assert!((tb_hw - 373.124).abs() < 0.02, "Hyland-Wexler boiling point {tb_hw}");
}

/// A pressure at which the liquid does not boil anywhere below its critical
/// temperature is refused by name, at setup, where it can be said.
#[test]
fn a_pressure_with_no_boiling_point_is_refused() {
    let l = LiquidProperties::water();
    let e = boiling_temperature(SaturationCurve::HylandWexler, &l, 1e9).unwrap_err();
    let msg = format!("{e}");
    assert!(msg.contains("no boiling temperature"), "{msg}");
    assert!(msg.contains("S76.9"), "{msg}");
}

// ======================================================================
//  (76.4): the properties
// ======================================================================

#[test]
fn watsons_latent_heat_matches_its_anchor_and_its_derivative() {
    let l = LiquidProperties::water();
    assert!((latent_heat(&l, l.t_boil) - l.h_v_boil).abs() <= 1e-9 * l.h_v_boil);
    assert_eq!(latent_heat(&l, l.t_crit), 0.0);
    for &t in &[280.0, 300.0, 340.0, 370.0] {
        let h = 1e-5 * t;
        let fd = (latent_heat(&l, t + h) - latent_heat(&l, t - h)) / (2.0 * h);
        let an = latent_heat_derivative(&l, t);
        assert!(
            (an - fd).abs() < 1e-6 * fd.abs(),
            "dh_v/dT at {t}: analytic {an}, difference {fd}"
        );
    }
    // The number a reader can check: 2442 kJ/kg at 25 C is the steam-table
    // value; Watson gives 2474, 1.3 % high. Stated, not hidden.
    let hv25 = latent_heat(&l, 298.15);
    println!("[76.4] h_v(25 C) = {hv25:.0} J/kg; the steam table says 2442000 ({:+.2} %)",
             100.0 * (hv25 - 2.442e6) / 2.442e6);
    assert!((hv25 - 2.442e6).abs() / 2.442e6 < 0.02);
}

#[test]
fn the_diffusivity_carries_its_temperature_and_pressure_dependence() {
    let l = LiquidProperties::water();
    // Marrero & Mason's own correlation, evaluated independently here.
    let mm = |t: Scalar, p_atm: Scalar| 1.87e-10 * t.powf(2.072) / p_atm;
    for &t in &[290.0, 298.15, 350.0, 420.0] {
        let ours = diffusivity(&l, 101_325.0, t);
        let theirs = mm(t, 1.0);
        assert!(
            (ours - theirs).abs() < 1e-9 * theirs,
            "D({t}) = {ours}, Marrero & Mason {theirs}"
        );
    }
    // and the 1/p scaling
    let a = diffusivity(&l, 101_325.0, 300.0);
    let b = diffusivity(&l, 202_650.0, 300.0);
    assert!((b * 2.0 - a).abs() < 1e-12 * a);
}

// ======================================================================
//  (76.6)/(76.7): the closure
// ======================================================================

#[test]
fn the_blowing_factor_is_one_at_zero_and_smooth_across_the_series() {
    assert_eq!(blowing_factor(0.0), 1.0);
    for &b in &[-1e-9, 1e-9, 9.99e-7, 1.001e-6, -1.001e-6] {
        let f = blowing_factor(b);
        let exact = if b == 0.0 { 1.0 } else { b.ln_1p() / b };
        assert!((f - exact).abs() < 1e-12, "F({b}) = {f}, exact {exact}");
    }
}

/// The three blowing models differ by two named factors and by nothing else,
/// and one of the two is NOT the blowing correction.
///
/// This test was written expecting all three to collapse together as
/// `B_M -> 0`, and they do not: `ranzMarshall` and `spalding` differ by
/// `1/(1 - Y_s)` at *every* `B_M`, because `B_M = (Y_s - Y_g)/(1 - Y_s)` and
/// the un-blown form omits that Stefan-flow denominator entirely. At 25 C
/// that is 2 %, and at 90 C it is 40 % - so it is worth knowing that the
/// "no correction" leg is not the small-`B_M` limit of the others. What DOES
/// collapse as `B_M -> 0` is `spalding` against `abramzonSirignano`, which
/// share a mass rate exactly and differ only in `F(B_T)` against `F(B_M)`.
#[test]
fn the_blowing_models_differ_by_the_stefan_factor_and_by_f_of_b() {
    let gas = air(278.15, 0.99);
    let tb = water().boiling_temperature().unwrap();
    let rate = |transfer: MassTransfer| {
        let c = EvaporationControls { transfer, ..water() };
        droplet_rate(&c, tb, 1e-4, 278.0, &gas)
    };
    let rm = rate(MassTransfer::RanzMarshall);
    let sp = rate(MassTransfer::Spalding);
    let a_s = rate(MassTransfer::AbramzonSirignano);
    println!(
        "[76.6] at B_M = {:.3e}, Y_s = {:.5}: mdot ranzMarshall {:.6e}, spalding {:.6e}, \
         abramzonSirignano {:.6e} kg/s",
        sp.b_m, sp.y_surface, rm.mdot, sp.mdot, a_s.mdot
    );

    // The mass rates: exactly ln(1 + B_M) against (Y_s - Y_g), so the ratio
    // is exactly that of those two numbers.
    let ratio = sp.mdot / rm.mdot;
    let want = sp.b_m.ln_1p() / (sp.y_surface - gas.y_vapour);
    assert!((ratio - want).abs() < 1e-10 * want.abs(), "{ratio} vs {want}");

    // Spalding and Abramzon-Sirignano share the mass rate BIT FOR BIT: the
    // heat transfer number never enters it.
    assert_eq!(sp.mdot.to_bits(), a_s.mdot.to_bits());

    // ... and their conductances differ by exactly F(B_T)/F(B_M).
    let c_ratio = a_s.conductance / sp.conductance;
    let want_c = blowing_factor(a_s.b_t) / blowing_factor(sp.b_m);
    assert!((c_ratio - want_c).abs() < 1e-10, "{c_ratio} vs {want_c}");
    // which is within 1 % of one at this tiny B_M, and is what "the blowing
    // correction is negligible in the Ranz-Marshall data" means quantitatively.
    assert!((c_ratio - 1.0).abs() < 1e-2, "F(B_T)/F(B_M) = {c_ratio}");
}

/// (76.11): the slope this module reports IS the textbook `d^2` law, checked
/// against the closed form written out independently.
#[test]
fn the_d2_law_slope_is_the_textbook_closed_form() {
    let gas = air(298.15, 0.30);
    let ctrl = water();
    let tb = ctrl.boiling_temperature().unwrap();
    let t_p = steady_temperature(&ctrl, tb, 1e-3, &gas).unwrap();
    let r = droplet_rate(&ctrl, tb, 1e-3, t_p, &gas);

    // The textbook form, assembled here from the film state rather than from
    // `mdot`: K = 8 rho_f D_f ln(1 + B_M)/rho_l, valid at Re = 0 where
    // Sh_0 = 2 exactly.
    let l = &ctrl.liquid;
    let t_f = t_p + (gas.t - t_p) / 3.0;
    let rho_f = gas.rho * gas.t / t_f;
    let d_f = diffusivity(l, ctrl.p_ambient, t_f);
    let textbook = 8.0 * rho_f * d_f * r.b_m.ln_1p() / 1000.0;

    let ours = d2_law_slope(&ctrl, tb, 1000.0, 1e-3, t_p, &gas);
    println!(
        "[76.11] K = {ours:.6e} m2/s (textbook {textbook:.6e}); B_M {:.5}, T_p {:.3} K",
        r.b_m, t_p
    );
    assert!(
        (ours - textbook).abs() <= 1e-12 * textbook,
        "K {ours} against the closed form {textbook}"
    );

    // ... and it does not depend on d, which is the whole content of the law.
    for &d in &[1e-5, 1e-4, 1e-3, 3e-3] {
        let k = d2_law_slope(&ctrl, tb, 1000.0, d, t_p, &gas);
        assert!((k - ours).abs() <= 1e-12 * ours, "K({d}) = {k}, K(1e-3) = {ours}");
    }
}

/// The Ranz-Marshall leg has its own closed form, and it is a DIFFERENT
/// function of the same properties - which is why it is kept.
#[test]
fn the_ranz_marshall_slope_is_linear_in_the_driving_fraction() {
    let gas = air(298.15, 0.30);
    let ctrl = EvaporationControls { transfer: MassTransfer::RanzMarshall, ..water() };
    let tb = ctrl.boiling_temperature().unwrap();
    let t_p = steady_temperature(&ctrl, tb, 1e-3, &gas).unwrap();
    let r = droplet_rate(&ctrl, tb, 1e-3, t_p, &gas);
    let l = &ctrl.liquid;
    let t_f = t_p + (gas.t - t_p) / 3.0;
    let rho_f = gas.rho * gas.t / t_f;
    let d_f = diffusivity(l, ctrl.p_ambient, t_f);
    let textbook = 8.0 * rho_f * d_f * (r.y_surface - gas.y_vapour) / 1000.0;
    let ours = d2_law_slope(&ctrl, tb, 1000.0, 1e-3, t_p, &gas);
    assert!((ours - textbook).abs() <= 1e-12 * textbook, "{ours} vs {textbook}");
}

/// (76.11): the steady temperature is where the residual vanishes, and the
/// residual is the energy balance itself.
#[test]
fn the_steady_temperature_is_where_the_residual_vanishes() {
    let ctrl = water();
    let tb = ctrl.boiling_temperature().unwrap();
    for &(t_g, rh) in &[(290.0, 0.1), (298.15, 0.5), (330.0, 0.2), (373.0, 0.05)] {
        let gas = air(t_g, rh);
        let t_p = steady_temperature(&ctrl, tb, 5e-4, &gas).unwrap();
        let r = droplet_rate(&ctrl, tb, 5e-4, t_p, &gas);
        let residual = r.conductance * (gas.t - t_p) + r.mdot * r.h_v;
        let scale = (r.conductance * (gas.t - t_p)).abs();
        assert!(
            residual.abs() <= 1e-10 * scale,
            "T_g {t_g} rh {rh}: T_p {t_p}, residual {residual}, scale {scale}"
        );
    }
}

/// The steady temperature is independent of the droplet diameter, because
/// both sides of the balance scale as `d`. Not a numerical accident - it is
/// why a spray of many sizes reaches ONE wet-bulb temperature.
#[test]
fn the_steady_temperature_does_not_depend_on_the_diameter() {
    let ctrl = water();
    let tb = ctrl.boiling_temperature().unwrap();
    let gas = air(298.15, 0.3);
    let a = steady_temperature(&ctrl, tb, 1e-5, &gas).unwrap();
    for &d in &[1e-4, 1e-3, 5e-3] {
        let b = steady_temperature(&ctrl, tb, d, &gas).unwrap();
        assert!((a - b).abs() < 1e-9, "T_p({d}) = {b}, T_p(1e-5) = {a}");
    }
}

/// SPEC-LIT S76.13, host half. The droplet's steady temperature is NOT the
/// psychrometric wet bulb, and the gap is the Lewis number - measured here,
/// across humidity, and printed with the factor that explains it.
#[test]
fn the_steady_temperature_sits_below_the_psychrometric_wet_bulb() {
    let ctrl = water();
    let tb = ctrl.boiling_temperature().unwrap();
    println!(
        "[76.13] a 500 um water droplet in still 25 C air, against ASHRAE's wet bulb:"
    );
    println!("        rh      T_wb(C)   T_p(C)    gap(K)   gap/depression   Le");
    let mut worst: Scalar = 0.0;
    for &rh in &[0.1, 0.2, 0.3, 0.5, 0.7, 0.9] {
        let gas = air(298.15, rh);
        let t_p = steady_temperature(&ctrl, tb, 5e-4, &gas).unwrap();
        let w = psychro::w_from_t_rh_p(298.15, rh, 101_325.0);
        let t_wb = psychro::t_wb(298.15, w, 101_325.0).unwrap();
        let r = droplet_rate(&ctrl, tb, 5e-4, t_p, &gas);
        let gap = (t_p - 273.15) - t_wb;
        let depression = 25.0 - t_wb;
        worst = worst.max(gap.abs());
        println!(
            "        {rh:4.2}   {t_wb:7.3}   {:7.3}   {gap:+6.3}   {:+8.3}       {:.4}",
            t_p - 273.15,
            gap / depression,
            r.lewis
        );
    }
    // The claim is a SHAPE, not a tolerance: the droplet is always colder
    // than the psychrometric wet bulb, because the quiescent Nu_0 = Sh_0 = 2
    // gives a psychrometric ratio of Le rather than of one, and Le < 1 for
    // water in air. If this ever came out the other way the property set
    // would be wrong.
    assert!(worst < 3.0, "the wet-bulb gap reached {worst} K");
}

/// (76.9): above the boiling point the closure switches to Godsave's
/// heat-limited rate, and that rate is the one written out here.
#[test]
fn the_boiling_branch_is_godsaves_heat_limited_rate() {
    let ctrl = water();
    let tb = ctrl.boiling_temperature().unwrap();
    let gas = air(600.0, 0.0);
    let d = 5e-4;
    let r = droplet_rate(&ctrl, tb, d, tb + 50.0, &gas);
    assert!(r.boiling, "a droplet above T_b should take the boiling branch");
    let hv = latent_heat(&ctrl.liquid, tb);
    let b_t = ctrl.liquid.cp_vapour * (gas.t - tb) / hv;
    // Godsave: mdot = -pi d (k/c_pv) Nu_0 ln(1 + B_T), with Nu_0 = 2 at
    // Re = 0. Written from the correlation, not from `r`.
    let want = -std::f64::consts::PI * d * (gas.k / ctrl.liquid.cp_vapour) * 2.0 * (1.0 + b_t).ln();
    println!(
        "[76.9] boiling in 600 K air: B_T {b_t:.4}, mdot {:.6e} kg/s (Godsave {want:.6e})",
        r.mdot
    );
    assert!((r.mdot - want).abs() <= 1e-12 * want.abs(), "{} vs {want}", r.mdot);
    // ... and the temperature is capped, so the surface fraction can never
    // exceed one and B_M can never be a NaN.
    assert!(r.y_surface <= 1.0);
}

/// A droplet in a saturated gas does not evaporate; in a supersaturated one
/// it grows. Both branches of the sign, and no NaN in either.
#[test]
fn a_saturated_gas_stops_the_evaporation_and_a_wetter_one_reverses_it() {
    let ctrl = water();
    let tb = ctrl.boiling_temperature().unwrap();
    let t = 298.15;
    let y_sat = psychro::yv_from_t_rh_p(t, 1.0, 101_325.0);
    for (label, y, want_sign) in [
        ("dry", 0.0, -1.0),
        ("saturated", y_sat, -1.0),
        ("supersaturated", 1.5 * y_sat, 1.0),
    ] {
        let gas = GasState { y_vapour: y, ..air(t, 0.0) };
        let r = droplet_rate(&ctrl, tb, 1e-4, t, &gas);
        assert!(r.mdot.is_finite(), "{label}: mdot is {}", r.mdot);
        assert!(
            r.mdot * want_sign >= 0.0 || r.mdot.abs() < 1e-18,
            "{label}: mdot {} has the wrong sign",
            r.mdot
        );
    }
}

// ======================================================================
//  (76.2): the properties are a SET, not a decoration
// ======================================================================

/// Every liquid property is read. Perturb one, the answer moves; if it does
/// not, the property is plumbed in and ignored, which is the failure this
/// test exists for.
#[test]
fn every_liquid_property_moves_the_answer() {
    let gas = air(320.0, 0.2);
    let base = EvaporationControls {
        saturation: SaturationCurve::ClausiusClapeyron,
        ..water()
    };
    // The metric is the STEADY TEMPERATURE and not `mdot`, and the difference
    // matters: `cpVapour` enters only `B_T`, so it moves the conductance and
    // not the mass rate at fixed `T_p`. Measuring `mdot` alone reported that
    // property as dead when it is not, which is the mistake this test is
    // supposed to catch rather than commit.
    let rate = |c: &EvaporationControls| -> Scalar {
        let tb = c.boiling_temperature().unwrap();
        steady_temperature(c, tb, 1e-4, &gas).unwrap()
    };
    let m0 = rate(&base);
    let w = LiquidProperties::water();
    let cases: Vec<(&str, EvaporationControls)> = vec![
        ("wVapour", EvaporationControls {
            liquid: LiquidProperties { w_vapour: w.w_vapour * 1.05, ..w }, ..base }),
        ("tBoil", EvaporationControls {
            liquid: LiquidProperties { t_boil: w.t_boil + 5.0, ..w }, ..base }),
        ("pBoil", EvaporationControls {
            liquid: LiquidProperties { p_boil: w.p_boil * 1.02, ..w }, ..base }),
        ("hvBoil", EvaporationControls {
            liquid: LiquidProperties { h_v_boil: w.h_v_boil * 1.02, ..w }, ..base }),
        ("tCrit", EvaporationControls {
            liquid: LiquidProperties { t_crit: w.t_crit + 20.0, ..w }, ..base }),
        ("dVapour", EvaporationControls {
            liquid: LiquidProperties { d_vapour: w.d_vapour * 1.05, ..w }, ..base }),
        ("tRefD", EvaporationControls {
            liquid: LiquidProperties { t_ref_d: w.t_ref_d + 10.0, ..w }, ..base }),
        ("dExponent", EvaporationControls {
            liquid: LiquidProperties { d_exponent: w.d_exponent + 0.2, ..w }, ..base }),
        ("cpVapour", EvaporationControls {
            liquid: LiquidProperties { cp_vapour: w.cp_vapour * 1.1, ..w }, ..base }),
        ("wCarrier", EvaporationControls { w_carrier: base.w_carrier * 1.05, ..base }),
        ("pAmbient", EvaporationControls { p_ambient: 90_000.0, ..base }),
    ];
    for (name, c) in cases {
        let m = rate(&c);
        assert!(
            (m - m0).abs() > 1e-6,
            "{name} does not move the steady droplet temperature: {m} against {m0}"
        );
    }
}

/// The second fluid runs, and it does not run water's numbers.
#[test]
fn benzene_is_a_different_liquid_and_not_water_with_a_new_label() {
    let ctrl = EvaporationControls {
        saturation: SaturationCurve::ClausiusClapeyron,
        liquid: LiquidProperties::benzene(),
        ..water()
    };
    ctrl.validate().expect("benzene validates");
    let tb = ctrl.boiling_temperature().unwrap();
    assert!((tb - 353.25).abs() < 1e-6, "benzene boils at {tb} K");
    let gas = air(298.15, 0.0);
    let t_p = steady_temperature(&ctrl, tb, 1e-3, &gas).unwrap();
    let k = d2_law_slope(&ctrl, tb, 879.0, 1e-3, t_p, &gas);
    let kw = {
        let cw = water();
        let tbw = cw.boiling_temperature().unwrap();
        let tw = steady_temperature(&cw, tbw, 1e-3, &gas).unwrap();
        d2_law_slope(&cw, tbw, 1000.0, 1e-3, tw, &gas)
    };
    println!("[76.2] K: benzene {k:.4e} m2/s, water {kw:.4e} m2/s in dry 25 C air");
    assert!(k > 3.0 * kw, "benzene should evaporate much faster than water: {k} vs {kw}");
}

// ======================================================================
//  SPEC-LIT S13.4: the contract
// ======================================================================

#[test]
fn every_setting_is_recognised_by_name_and_the_rest_refused() {
    assert_eq!(
        SaturationCurve::from_name("clausiusClapeyron").unwrap(),
        SaturationCurve::ClausiusClapeyron
    );
    assert_eq!(
        SaturationCurve::from_name("hylandWexler").unwrap(),
        SaturationCurve::HylandWexler
    );
    for s in SaturationCurve::NAMES {
        assert_eq!(SaturationCurve::from_name(s).unwrap().name(), *s);
    }
    for s in MassTransfer::NAMES {
        assert_eq!(MassTransfer::from_name(s).unwrap().name(), *s);
    }
    for (bad, must) in [
        ("antoine", "coefficient set"),
        ("wagner", "coefficient set"),
        ("goff-gratch", "not supported"),
    ] {
        let e = SaturationCurve::from_name(bad).unwrap_err();
        let m = format!("{e}");
        assert!(m.contains(must), "{bad}: {m}");
        assert!(m.contains("hylandWexler"), "{bad} does not print the menu: {m}");
    }
    for (bad, must) in [
        ("filmThickness", "fixed point"),
        ("sazhin", "not supported"),
    ] {
        let e = MassTransfer::from_name(bad).unwrap_err();
        let m = format!("{e}");
        assert!(m.contains(must), "{bad}: {m}");
    }
}

/// Hyland-Wexler IS water. Asking for it with another liquid returns a
/// confident number from the wrong curve, so it is refused by name.
#[test]
fn hyland_wexler_is_refused_for_a_liquid_that_is_not_water() {
    let ctrl = EvaporationControls {
        saturation: SaturationCurve::HylandWexler,
        liquid: LiquidProperties::benzene(),
        ..water()
    };
    let e = ctrl.validate().unwrap_err();
    let m = format!("{e}");
    assert!(m.contains("fit to WATER"), "{m}");
    assert!(m.contains("clausiusClapeyron"), "{m}");
    assert!(m.contains("S76.3"), "{m}");
    // ... and the general curve is accepted for the same liquid.
    EvaporationControls {
        saturation: SaturationCurve::ClausiusClapeyron,
        ..ctrl
    }
    .validate()
    .expect("clausiusClapeyron accepts benzene");
}

#[test]
fn the_numbers_are_validated_and_each_one_is_named() {
    let w = LiquidProperties::water();
    let base = water();
    let cases: Vec<(&str, EvaporationControls)> = vec![
        ("wVapour", EvaporationControls {
            liquid: LiquidProperties { w_vapour: 0.0, ..w }, ..base }),
        ("hvBoil", EvaporationControls {
            liquid: LiquidProperties { h_v_boil: -1.0, ..w }, ..base }),
        ("dVapour", EvaporationControls {
            liquid: LiquidProperties { d_vapour: Scalar::NAN, ..w }, ..base }),
        ("cpVapour", EvaporationControls {
            liquid: LiquidProperties { cp_vapour: 0.0, ..w }, ..base }),
        ("wCarrier", EvaporationControls { w_carrier: 0.0, ..base }),
        ("pAmbient", EvaporationControls { p_ambient: -1.0, ..base }),
        ("cfl", EvaporationControls { cfl: 0.0, ..base }),
        ("cfl", EvaporationControls { cfl: 2.0, ..base }),
    ];
    for (name, c) in cases {
        let e = c.validate().unwrap_err();
        let m = format!("{e}");
        assert!(m.contains(name), "the error for {name} does not name it: {m}");
    }
    // tBoil above tCrit is its own message, because the failure is a
    // relation and not a number.
    let e = EvaporationControls {
        saturation: SaturationCurve::ClausiusClapeyron,
        liquid: LiquidProperties { t_boil: 700.0, ..w },
        ..base
    }
    .validate()
    .unwrap_err();
    assert!(format!("{e}").contains("tCrit"), "{e}");
}

/// SPEC-LIT S13.4.2: the banner names every setting the run will use.
#[test]
fn the_banner_names_every_setting() {
    let d = water().describe();
    for token in [
        "saturation=hylandWexler",
        "transfer=abramzonSirignano",
        "pAmbient=",
        "wVapour=",
        "wCarrier=",
        "tBoil=",
        "hvBoil=",
        "tCrit=",
        "dVapour=",
        "cpVapour=",
        "cfl=",
        "S76",
    ] {
        assert!(d.contains(token), "the banner is missing {token}: {d}");
    }
}

/// The device codes and the host enums are one mapping, and it is written
/// down in exactly one place on each side. `parcels::tests` checks the
/// device half against `cuda/parcels.cu`.
#[test]
fn the_enumeration_codes_are_dense_and_stable() {
    assert_eq!(SaturationCurve::ClausiusClapeyron.code(), 0);
    assert_eq!(SaturationCurve::HylandWexler.code(), 1);
    assert_eq!(MassTransfer::RanzMarshall.code(), 0);
    assert_eq!(MassTransfer::Spalding.code(), 1);
    assert_eq!(MassTransfer::AbramzonSirignano.code(), 2);
}
