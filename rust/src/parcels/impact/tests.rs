// meteor-cfd - Copyright (c) 2026 주식회사 메테오시뮬레이션 (Meteo Simulation Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.

//! SPEC-LIT S78.9's host half: the surface tension against IAPWS R1-76's own
//! published table, the dimensionless groups against each other's algebraic
//! identities, the map against its closed-form inverse, the two splash
//! criteria against each other, and the S13.4 contract on every setting this
//! section adds.
//!
//! Written from the same sources as `src/parcels/impact.rs`; see that
//! module's header. No GPL-licensed source was consulted.

use super::*;

/// Water at 20 C, as everything below is posed in.
fn water() -> WallImpactControls {
    WallImpactControls::default()
}

// ----------------------------------------------------------------------
//  (78.2): surface tension
// ----------------------------------------------------------------------

/// IAPWS R1-76 publishes a table of its own correlation. Reproducing that
/// table is the only way to say the four constants were transcribed
/// correctly, and it is a check against a PUBLISHED NUMBER rather than
/// against this crate's arithmetic twice.
///
/// The release's table is in mN/m to two decimals; the tolerance below is
/// half of that last digit, so a wrong constant cannot hide in it.
#[test]
fn the_surface_tension_reproduces_the_iapws_r1_76_table() {
    // (T [K], sigma [mN/m]) - IAPWS R1-76 (2014), Table 1.
    let table: &[(Scalar, Scalar)] = &[
        (273.15, 75.65),
        (293.15, 72.74),
        (298.15, 71.97),
        (313.15, 69.60),
        (373.15, 58.91),
        (473.15, 37.67),
        (573.15, 14.36),
    ];
    for &(t, want) in table {
        let got = 1000.0 * surface_tension(SurfaceTension::IapwsR176, 0.0, t);
        assert!(
            (got - want).abs() <= 5e-3,
            "IAPWS R1-76 at {t} K: got {got} mN/m, the release tabulates {want}"
        );
    }
}

/// The default `sigma` IS the correlation's own value at 293.15 K, to the
/// four decimals in mN/m the release prints. A default nobody can trace is
/// how a magic number gets into a spray model.
#[test]
fn the_default_surface_tension_is_the_iapws_value_at_twenty_celsius() {
    let iapws = surface_tension(SurfaceTension::IapwsR176, 0.0, 293.15);
    let c = water();
    assert!(
        (c.sigma - iapws).abs() <= 5e-6,
        "default sigma {} against IAPWS R1-76's {iapws} at 293.15 K",
        c.sigma
    );
}

/// Above the critical temperature there is no liquid surface and the
/// correlation's `tau^1.256` would be `pow` of a negative number. It returns
/// a clean zero, which (78.3) turns into `We = +inf` and the map turns into
/// a splash - the right answer, and reached without a special case anywhere
/// downstream.
#[test]
fn the_correlation_is_clamped_at_the_critical_point_and_a_splash_follows() {
    for t in [T_CRITICAL_WATER, T_CRITICAL_WATER + 1.0, 1000.0] {
        let s = surface_tension(SurfaceTension::IapwsR176, 0.072, t);
        assert_eq!(s, 0.0, "sigma at {t} K");
    }
    let c = WallImpactControls { tension: SurfaceTension::IapwsR176, ..water() };
    let (n, r) = c.classify(1000.0, 1e-4, 700.0, 1.0);
    assert!(n.we.is_infinite() && n.we > 0.0, "We = {}", n.we);
    assert_eq!(r, WallRegime::Splash);
}

/// The constant model does not read the temperature at all, which is the
/// whole of its contract: a case that sets `sigma` gets `sigma`.
#[test]
fn the_constant_model_ignores_the_droplet_temperature() {
    for t in [200.0, 293.15, 400.0, 900.0] {
        assert_eq!(surface_tension(SurfaceTension::Constant, 0.05, t), 0.05);
    }
}

// ----------------------------------------------------------------------
//  (78.3): the groups
// ----------------------------------------------------------------------

/// `Oh = sqrt(We)/Re`, `La = 1/Oh^2` and `K^4 = We^2 Re` are three algebraic
/// identities among the four groups, and they are what licenses (78.4b)'s
/// `pow`-free decision form. Checked over five decades of impact speed and
/// three of diameter, because an identity that only holds near one operating
/// point is a coincidence.
#[test]
fn the_four_groups_satisfy_their_algebraic_identities() {
    let c = water();
    let rho = 1000.0;
    let mut worst: (Scalar, Scalar, Scalar) = (0.0, 0.0, 0.0);
    for &d in &[1e-5, 1e-4, 1e-3] {
        for i in 0..50 {
            let u = 1e-2 * (10.0 as Scalar).powf(Scalar::from(i) / 10.0);
            let n = impact_numbers(rho, c.mu_liquid, c.sigma, d, u);
            let e_oh = ((n.oh - n.we.sqrt() / n.re) / n.oh).abs();
            let e_la = ((n.la - 1.0 / (n.oh * n.oh)) / n.la).abs();
            let k4 = n.k * n.k * n.k * n.k;
            let e_k4 = ((n.k4 - k4) / n.k4).abs();
            worst = (worst.0.max(e_oh), worst.1.max(e_la), worst.2.max(e_k4));
        }
    }
    assert!(worst.0 < 1e-14, "Oh = sqrt(We)/Re to {}", worst.0);
    assert!(worst.1 < 1e-14, "La = 1/Oh^2 to {}", worst.1);
    // `k` is formed with `powf` and `k4` without, so the two agree to the
    // accuracy of `powf` rather than to round-off. That gap is the REASON
    // (78.4b) exists and is measured here rather than assumed away.
    assert!(worst.2 < 1e-13, "K^4 = We^2 Re to {}", worst.2);
}

/// Mundo's parameter at a condition the paper's own regime plot covers: a
/// 100 um water droplet at about 9 m/s sits within a few per cent of
/// `K = 57.7`, which is what makes 57.7 a threshold a spray actually reaches
/// rather than a number in a table.
#[test]
fn mundos_parameter_is_the_published_order_of_magnitude() {
    let c = water();
    let n = impact_numbers(1000.0, c.mu_liquid, c.sigma, 1e-4, 9.0);
    assert!((n.k - 57.7).abs() < 6.0, "K = {} at 9 m/s", n.k);
    assert!(n.we > 100.0 && n.we < 130.0, "We = {}", n.we);
    assert!(n.re > 800.0 && n.re < 1000.0, "Re = {}", n.re);
    assert!(n.oh > 0.011 && n.oh < 0.013, "Oh = {}", n.oh);
}

// ----------------------------------------------------------------------
//  (78.4)/(78.5): the map and its inverse
// ----------------------------------------------------------------------

/// The map's three boundaries have closed-form inverses (78.4c). Straddling
/// each of them by one part in a million must flip the regime, and nothing
/// else in the sweep may. That is the gate S78.9 calls 78-A on the host side:
/// the map is exactly the published criterion, at the point where it
/// matters.
#[test]
fn every_regime_boundary_sits_where_the_closed_form_inverse_says() {
    for splash in [SplashCriterion::Mundo, SplashCriterion::BaiGosman] {
        let c = WallImpactControls { splash, ..water() };
        let (rho, d, t) = (1000.0 as Scalar, 1e-4 as Scalar, 293.15 as Scalar);
        let bounds = [
            (WallRegime::Stick, WallRegime::Rebound),
            (WallRegime::Rebound, WallRegime::Spread),
            (WallRegime::Spread, WallRegime::Splash),
        ];
        for (below, above) in bounds {
            let u = c.boundary_speed(rho, d, t, above).unwrap();
            let lo = c.classify(rho, d, t, u * (1.0 - 1e-6)).1;
            let hi = c.classify(rho, d, t, u * (1.0 + 1e-6)).1;
            assert_eq!(lo, below, "{splash:?}: just below {u} m/s");
            assert_eq!(hi, above, "{splash:?}: just above {u} m/s");
        }
        // ... and the three boundaries are ordered, so the four bands are
        // four bands.
        let u1 = c.boundary_speed(rho, d, t, WallRegime::Rebound).unwrap();
        let u2 = c.boundary_speed(rho, d, t, WallRegime::Spread).unwrap();
        let u3 = c.boundary_speed(rho, d, t, WallRegime::Splash).unwrap();
        assert!(u1 < u2 && u2 < u3, "{splash:?}: {u1} {u2} {u3}");
        assert!(c.boundary_speed(rho, d, t, WallRegime::Stick).is_none());
    }
}

/// The map is TOTAL and single-valued: every impact gets exactly one regime,
/// and the regime is monotone in the impact speed. A dense sweep, so that a
/// re-ordering of the three tests would be caught rather than argued about.
#[test]
fn the_map_is_monotone_in_the_impact_speed() {
    for splash in [SplashCriterion::Mundo, SplashCriterion::BaiGosman] {
        let c = WallImpactControls { splash, ..water() };
        let mut last = -1;
        for i in 0..4000 {
            let u = 1e-3 + Scalar::from(i) * 0.01;
            let code = c.classify(1000.0, 1e-4, 293.15, u).1.code();
            assert!(code >= last, "{splash:?}: regime fell back at u = {u}");
            last = code;
        }
        assert_eq!(last, WallRegime::Splash.code());
    }
}

/// (78.4b): the fourth-power decision form and the textbook `K > K_crit` are
/// the same test. They are not the same ARITHMETIC - one has a `powf` in it
/// and the other does not - so this is a real check and not a tautology, and
/// it is why the device can take the decision without `pow`.
#[test]
fn the_fourth_power_form_decides_exactly_what_k_does() {
    let c = water();
    let mut disagreements = 0;
    let mut closest = Scalar::INFINITY;
    for i in 0..20_000 {
        let u = 0.01 + Scalar::from(i) * 0.002;
        let n = impact_numbers(1000.0, c.mu_liquid, c.sigma, 1e-4, u);
        let naive = n.k > c.k_crit;
        if c.splashing(&n) != naive {
            disagreements += 1;
            closest = closest.min(((n.k - c.k_crit) / c.k_crit).abs());
        }
    }
    assert_eq!(
        disagreements, 0,
        "the two forms disagreed; the closest was {closest} in relative K"
    );
}

/// The two published criteria do NOT agree, and the size of the gap is the
/// honest content of S78.10. At 100 um in water Mundo's threshold is reached
/// at about a fifth of the Weber number Bai & Gosman's smooth dry wall needs.
#[test]
fn the_two_published_splash_criteria_disagree_by_a_factor_of_five() {
    let (rho, d, t) = (1000.0 as Scalar, 1e-4 as Scalar, 293.15 as Scalar);
    let m = WallImpactControls { splash: SplashCriterion::Mundo, ..water() };
    let b = WallImpactControls { splash: SplashCriterion::BaiGosman, ..water() };
    let um = m.boundary_speed(rho, d, t, WallRegime::Splash).unwrap();
    let ub = b.boundary_speed(rho, d, t, WallRegime::Splash).unwrap();
    let we = |u: Scalar| rho * d * u * u / m.sigma;
    let ratio = we(ub) / we(um);
    assert!(um < ub, "Mundo {um} should splash before Bai-Gosman {ub}");
    assert!(
        (3.0..8.0).contains(&ratio),
        "the criteria differ by {ratio} in We, which is not the factor S78.10 reports"
    );
}

/// Viscosity RAISES both splash thresholds, and by a lot. That is the
/// physically right direction - viscous dissipation is what stops the lamella
/// breaking up - and it is asserted here because the first version of this
/// test asserted the opposite from the sign of the `-0.18` exponent and was
/// wrong: `La` falls with viscosity, so `La^(-0.18)` RISES, and glycerol
/// needs twelve times the Weber number water does.
#[test]
fn viscosity_raises_the_splash_threshold_and_the_spread_band_never_closes() {
    let c = WallImpactControls { splash: SplashCriterion::BaiGosman, ..water() };
    let la = |mu: Scalar| 1000.0 * c.sigma * 1e-4 / (mu * mu);
    let water_we_c = c.bai_gosman_we_c(la(c.mu_liquid));
    let glycerol_we_c = c.bai_gosman_we_c(la(1.0));
    assert!(water_we_c > 400.0 && water_we_c < 700.0, "water We_c = {water_we_c}");
    assert!(
        glycerol_we_c > 10.0 * water_we_c,
        "a 1000x viscosity moved We_c from {water_we_c} only to {glycerol_we_c}"
    );
    // ... so for any liquid a case will actually name, the splash threshold
    // is far above `weSpread` and all four bands are non-empty. That is a
    // statement about the DEFAULTS and is worth having: the map does not
    // degenerate on the fluids it is for.
    for mu in [c.mu_liquid, 1e-2, 1e-1, 1.0] {
        let k = WallImpactControls { mu_liquid: mu, ..c };
        let u2 = k.boundary_speed(1000.0, 1e-4, 293.15, WallRegime::Spread).unwrap();
        let u3 = k.boundary_speed(1000.0, 1e-4, 293.15, WallRegime::Splash).unwrap();
        assert!(u3 > u2, "mu = {mu}: splash at {u3} m/s is not above spread at {u2}");
    }
}

/// The spread band CAN be closed - by the thresholds, which a case may set
/// freely - and when it is, the map is still single-valued because the splash
/// test is taken first. This is the case that proves the ordering is
/// load-bearing rather than cosmetic.
#[test]
fn a_closed_spread_band_leaves_the_map_single_valued() {
    // `weSpread` above the splash threshold: legal, and it says "there is no
    // spreading on this surface, it either bounces or it shatters".
    let c = WallImpactControls { we_spread: 1e5, ..water() };
    let u_splash = c
        .boundary_speed(1000.0, 1e-4, 293.15, WallRegime::Splash)
        .unwrap();
    let we_at = 1000.0 * 1e-4 * u_splash * u_splash / c.sigma;
    assert!(we_at < c.we_spread, "the splash threshold We = {we_at} is not below weSpread");
    let mut seen = [false; 4];
    for i in 0..4000 {
        let u = 1e-3 + Scalar::from(i) * 0.01;
        seen[c.classify(1000.0, 1e-4, 293.15, u).1.code() as usize] = true;
    }
    assert!(!seen[WallRegime::Spread.code() as usize], "the spread band was not empty");
    assert!(seen[WallRegime::Stick.code() as usize]);
    assert!(seen[WallRegime::Rebound.code() as usize]);
    assert!(seen[WallRegime::Splash.code() as usize]);
}

/// Three of the four outcomes leave the mass on the wall, and exactly one
/// does not. S78.6's ledger is a partition of the pool because of this one
/// line, so it is asserted rather than read off the enum.
#[test]
fn exactly_one_regime_gives_the_droplet_back() {
    let n: usize = [
        WallRegime::Stick,
        WallRegime::Rebound,
        WallRegime::Spread,
        WallRegime::Splash,
    ]
    .iter()
    .filter(|r| !r.deposits())
    .count();
    assert_eq!(n, 1);
    assert!(!WallRegime::Rebound.deposits());
    assert!(WallRegime::Splash.deposits());
}

// ----------------------------------------------------------------------
//  S13.4: the contract
// ----------------------------------------------------------------------

/// Every name round-trips, and every refused one is refused BY NAME with the
/// menu printed.
#[test]
fn every_impact_setting_goes_through_a_named_contract() {
    for s in SplashCriterion::NAMES {
        assert_eq!(SplashCriterion::from_name(s).unwrap().name(), *s);
    }
    for s in SurfaceTension::NAMES {
        assert_eq!(SurfaceTension::from_name(s).unwrap().name(), *s);
    }
    assert_eq!(
        SplashCriterion::from_name("bai").unwrap(),
        SplashCriterion::BaiGosman
    );
    assert_eq!(
        SurfaceTension::from_name("iapws").unwrap(),
        SurfaceTension::IapwsR176
    );
    for bad in ["cossali", "yarin", "stow"] {
        let m = format!("{}", SplashCriterion::from_name(bad).unwrap_err());
        assert!(m.contains("not supported"), "{bad}: {m}");
        assert!(m.contains("mundo"), "{bad} does not print the menu: {m}");
    }
    for bad in ["vargaftik", "linear"] {
        let m = format!("{}", SurfaceTension::from_name(bad).unwrap_err());
        assert!(m.contains("not supported"), "{bad}: {m}");
        assert!(m.contains("constant"), "{bad} does not print the menu: {m}");
    }
}

/// Every bad number is refused by name, and the ordering constraint on the
/// two Weber thresholds is one of them - an unordered pair is a different
/// map, not a tolerable input.
#[test]
fn every_bad_impact_number_is_refused_by_name() {
    let ok = water();
    ok.validate().unwrap();
    for (mutate, must) in [
        (
            WallImpactControls { sigma: 0.0, ..ok } as WallImpactControls,
            "sigma",
        ),
        (WallImpactControls { sigma: Scalar::NAN, ..ok }, "sigma"),
        (WallImpactControls { mu_liquid: -1.0, ..ok }, "muLiquid"),
        (WallImpactControls { k_crit: 0.0, ..ok }, "kCrit"),
        (WallImpactControls { splash_a: -3.0, ..ok }, "splashA"),
        (WallImpactControls { we_stick: -1.0, ..ok }, "weStick"),
        (
            WallImpactControls { we_stick: 30.0, we_spread: 20.0, ..ok },
            "weSpread",
        ),
    ] {
        let m = format!("{}", mutate.validate().unwrap_err());
        assert!(m.contains(must), "expected {must} to be named: {m}");
        assert!(m.contains("S78"), "expected a section number: {m}");
    }
    // Equal thresholds are legal and mean "no rebound band", which is a
    // model choice rather than an error.
    WallImpactControls { we_stick: 20.0, we_spread: 20.0, ..ok }.validate().unwrap();
}

/// S13.4.2: the banner says every setting the run will use, including the one
/// that the OTHER criterion would have read - and not the one it would not,
/// because a line that prints an unread number is a line that misleads.
#[test]
fn the_banner_prints_the_criterion_that_is_in_force() {
    let m = WallImpactControls { splash: SplashCriterion::Mundo, ..water() }.describe();
    assert!(m.contains("kCrit=57.7"), "{m}");
    assert!(!m.contains("splashA"), "{m}");
    let b = WallImpactControls { splash: SplashCriterion::BaiGosman, ..water() }.describe();
    assert!(b.contains("splashA=2630"), "{b}");
    assert!(!b.contains("kCrit"), "{b}");
    for s in [&m, &b] {
        assert!(s.contains("S78"), "{s}");
        assert!(s.contains("muLiquid=0.0010016"), "{s}");
    }
}
