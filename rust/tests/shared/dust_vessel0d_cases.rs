// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.
// Provenance: see PROVENANCE.md. No GPL-licensed source was consulted.

// The DST-08 part 1 gates (SPEC-LIT §157.6), written once and included both
// by src/dust/tests_vessel0d.rs (as dust::vessel0d::tests) and by
// tests/dust_vessel0d.rs, so both builds run them. No GPU: the 0-D vessel is
// host f64 in both builds, so the two builds read the same bits.
// The specification was informed by a review of FSCDustFoam (Huang &
// Lipatnikov, GPL) and no code was copied.
// No GPL-licensed source was consulted.

use ofgpu::chem::thermo::R_GAS;
use ofgpu::dust::thermo::{EquivalentGas, PrescribedInputs};
use ofgpu::dust::vessel0d::{
    integrate, metrics, BurningLaw, ThinFlameTrace, UnburnedDensity, Vessel,
};

/// A relative-band assert, `err <= rel |want|`, naming the quantity.
fn close(name: &str, got: f64, want: f64, rel: f64) {
    let err = (got - want).abs();
    assert!(
        err <= rel * want.abs().max(1e-300),
        "{name}: got {got}, want {want} (rel {rel})"
    );
}

/// The 20 L sphere's gamma_u (SPEC-LIT §157.6): `cp_u/(cp_u - R0/W_u)` at
/// the prescribed constant cp = 1000, W = 0.029.
fn gamma_u() -> f64 {
    1000.0 / (1000.0 - R_GAS / 0.029)
}

/// V20: R = 0.1684 m, p0 = 101325 Pa, pe = p0 (1 + 6 g) - the prescribed
/// gas's AICC ratio (test a), since 1 - R0/(W cp_u) = 1/gamma_u exactly.
fn v20() -> Vessel {
    let g = gamma_u();
    let p0 = 101325.0;
    Vessel { radius: 0.1684, p0, pe: p0 * (1.0 + 6.0 * g), gamma_u: g }
}

/// The case's law: `S0 = 0.5 m/s, alpha = 2.0, beta = -0.5`.
fn law() -> BurningLaw {
    BurningLaw { s0: 0.5, alpha: 2.0, beta: -0.5 }
}

/// The integration's start: `p_s = p0 + 1e-3 (pe - p0)` at `t_s = 0`, clear
/// of the `rate`'s zero at `p0`.
fn start() -> (f64, f64) {
    let v = v20();
    (0.0, v.p0 + 1e-3 * (v.pe - v.p0))
}

/// **SPEC-LIT §157.6 (a).** The vessel from the case's own thermo: the
/// prescribed gas (cp = 1000, W_u = W_b = 0.029, T_u = 300 K, T_b = 2100 K)
/// gives gamma_u = 1.4019456910621535 and pe/p0 = 9.41167414637292 - the
/// AICC ratio, which is V20's `p0 (1 + 6 g)` to the same band.
#[test]
fn vessel_from_gas_is_the_case_thermo() {
    let gas = EquivalentGas::prescribed(&PrescribedInputs {
        cp: 1000.0,
        w_u: 0.029,
        w_b: 0.029,
        t_u: 300.0,
        t_b: 2100.0,
    })
    .expect("the prescribed gas");
    let v = Vessel::from_gas(&gas, 300.0, 101325.0, 0.1684).expect("vessel");
    close("gamma_u", v.gamma_u, 1.4019456910621535, 1e-14);
    close("pe/p0", v.pe / 101325.0, 9.41167414637292, 1e-10);
    // V20's closed pe is the same AICC state the gas carries.
    close("pe/p0(case)", v.pe / 101325.0, v20().pe / 101325.0, 1e-12);
}

/// **SPEC-LIT §157.2-§157.3 (D).** The (157.2) fraction and the (157.3)
/// radius at the ends: `x_b` is 0 at `p0` and 1 at `pe`, the flame starts
/// at the centre and ends ON the wall.
#[test]
fn flame_radius_and_burned_fraction_end_points() {
    let v = v20();
    assert_eq!(v.burned_fraction(v.p0), 0.0);
    assert_eq!(v.burned_fraction(v.pe), 1.0);
    assert_eq!(v.flame_radius(v.p0), 0.0);
    close("flame_radius(pe)", v.flame_radius(v.pe), v.radius, 1e-15);
    close(
        "flame_radius(mid)",
        v.flame_radius((v.p0 + v.pe) / 2.0),
        0.15926087729212327,
        1e-12,
    );
}

/// **SPEC-LIT §157.6 (c).** The RK4 trace at h = 1e-5 s closes against the
/// Simpson quadrature: the time to the half pressure, the end time, the end
/// pressure ON `pe`, and `(dp/dt)_max` at the closed (157.5) value - under
/// the case's law and under the constant one.
#[test]
fn thin_flame_time_matches_the_quadrature() {
    let v = v20();
    let (t_s, p_s) = start();
    let tr = integrate(&v, &law(), UnburnedDensity::Compressed, t_s, p_s, 1e-5)
        .expect("the trace integrates");
    let m = metrics(&v, &law(), UnburnedDensity::Compressed, &tr).expect("metrics");
    close("t_half", m.t_half, 0.055709372114720164, 1e-10);
    close("t_end", m.t_end, 0.06831242328463713, 1e-10);
    assert_eq!(m.p_end, v.pe, "the trace ends on pe");
    close("dpdt_max", m.dpdt_max, 44293237.098994285, 1e-12);
    close("dpdt_max_closed", v.dpdt_max_closed(&law()), m.dpdt_max, 1e-12);
    let flat = BurningLaw::constant(0.5);
    let tr_c = integrate(&v, &flat, UnburnedDensity::Compressed, t_s, p_s, 1e-5)
        .expect("the trace integrates");
    let m_c = metrics(&v, &flat, UnburnedDensity::Compressed, &tr_c).expect("metrics");
    close("t_half(constant)", m_c.t_half, 0.058241649272052234, 1e-10);
    close("dpdt_max(constant)", m_c.dpdt_max, 37571570.778288685, 1e-12);
    close("k_st_closed(constant)", v.k_st_closed(&flat), 10199157.753370505, 1e-12);
}

/// **SPEC-LIT §157.6 (d).** Classical RK4 is fourth order: halving h twice
/// cuts the t_half error by ~16 each time - both successive ratios inside
/// [14, 18] (an oracle measured 15.5 and 15.8).
#[test]
fn rk4_is_fourth_order() {
    let v = v20();
    let (t_s, p_s) = start();
    let exact = 0.055709372114720164;
    let t_half = |h: f64| -> f64 {
        let tr = integrate(&v, &law(), UnburnedDensity::Compressed, t_s, p_s, h)
            .expect("the trace integrates");
        metrics(&v, &law(), UnburnedDensity::Compressed, &tr)
            .expect("metrics")
            .t_half
    };
    let e4 = (t_half(4e-4) - exact).abs();
    let e2 = (t_half(2e-4) - exact).abs();
    let e1 = (t_half(1e-4) - exact).abs();
    let (r42, r21) = (e4 / e2, e2 / e1);
    println!("  [157.6] RK4 error ratios: e(4e-4)/e(2e-4) = {r42}, e(2e-4)/e(1e-4) = {r21}");
    assert!((14.0..=18.0).contains(&r42), "e4/e2 = {r42} is outside [14, 18]");
    assert!((14.0..=18.0).contains(&r21), "e2/e1 = {r21} is outside [14, 18]");
}

/// **SPEC-LIT §157.5, §157.6 (e).** The cube-root law is EXACT for a thin
/// flame: the same law and pressure range on the 20 L and the 1 m3 sphere
/// gives one `K_st`, by (157.5)'s closed form and by `K_st = (dp/dt)_max
/// V^(1/3)` of the integrated traces; the times scale as the radii.
#[test]
fn cube_root_law_is_exact() {
    let v20 = v20();
    let v1 = Vessel { radius: 0.6204, ..v20 };
    let want = 1.2023817562643362e7;
    let (t_s, p_s) = start();
    let mut t_half = [0.0; 2];
    for (i, (name, v)) in [("20L", &v20), ("1m3", &v1)].into_iter().enumerate() {
        let tr = integrate(v, &law(), UnburnedDensity::Compressed, t_s, p_s, 1e-5)
            .expect("the trace integrates");
        let m = metrics(v, &law(), UnburnedDensity::Compressed, &tr).expect("metrics");
        close(&format!("k_st({name})"), v.k_st(m.dpdt_max), want, 1e-12);
        close(&format!("k_st_closed({name})"), v.k_st_closed(&law()), want, 1e-12);
        t_half[i] = m.t_half;
    }
    close("t_half(1m3)", t_half[1], 0.2052380906174132, 1e-10);
    close("t_half20/t_half1", t_half[0] / t_half[1], 0.1684 / 0.6204, 1e-10);
}

/// **SPEC-LIT §157.6 (f), INSIGHTS (g).** The negative control MISSES: with
/// `rho_u` held at its initial value, `(dp/dt)_max` falls by the (155.10)
/// end-state density ratio `(p0/pe)^(1/g)` - 20 %, far past the 10 % the
/// gate demands a control that passes to be a failed gate.
#[test]
fn the_negative_control_misses() {
    let v = v20();
    let (t_s, p_s) = start();
    let tr_c = integrate(&v, &law(), UnburnedDensity::Compressed, t_s, p_s, 1e-5)
        .expect("the trace integrates");
    let tr_i = integrate(&v, &law(), UnburnedDensity::Initial, t_s, p_s, 1e-5)
        .expect("the trace integrates");
    let m_c = metrics(&v, &law(), UnburnedDensity::Compressed, &tr_c).expect("metrics");
    let m_i = metrics(&v, &law(), UnburnedDensity::Initial, &tr_i).expect("metrics");
    let ratio = m_i.dpdt_max / m_c.dpdt_max;
    close("dpdt_max ratio", ratio, 0.20206396587822595, 1e-12);
    close(
        "= (p0/pe)^(1/g)",
        ratio,
        (v.p0 / v.pe).powf(1.0 / v.gamma_u),
        1e-12,
    );
    close("t_half(initial)", m_i.t_half, 0.09121777459992861, 1e-9);
    assert!(
        1.0 - ratio >= 0.10,
        "the control misses (dp/dt)_max by only {} < 10 %",
        1.0 - ratio
    );
}

/// **SPEC-LIT §157.6 (g).** Every refusal of the vessel names its input:
/// `Vessel::new`'s four, the law's three, `integrate`'s `p_start`, `h` and
/// law, and `metrics`' trace.
#[test]
fn vessel_refusals_name_the_input() {
    let v = v20();
    // Vessel::new: each input by name, every one of them house refusals.
    for (name, (r, p0, pe, g)) in [
        ("radius", (0.0, v.p0, v.pe, v.gamma_u)),
        ("p0", (v.radius, 0.0, v.pe, v.gamma_u)),
        ("pe", (v.radius, v.p0, v.p0, v.gamma_u)),
        ("gamma_u", (v.radius, v.p0, v.pe, 1.0)),
    ] {
        let e = Vessel::new(r, p0, pe, g).expect_err(name).to_string();
        assert!(e.contains("dust::vessel0d::Vessel::new"), "{name}: {e}");
        assert!(e.contains(name), "{name}: the refusal does not name it: {e}");
    }
    // BurningLaw::validate: s0, alpha and beta by name.
    for (name, bad) in [
        ("s0", BurningLaw { s0: 0.0, ..law() }),
        ("alpha", BurningLaw { alpha: f64::NAN, ..law() }),
        ("beta", BurningLaw { beta: f64::INFINITY, ..law() }),
    ] {
        let e = bad.validate().expect_err(name).to_string();
        assert!(e.contains("dust::vessel0d::BurningLaw::validate"), "{name}: {e}");
        assert!(e.contains(name), "{name}: the refusal does not name it: {e}");
    }
    // integrate: p_start, h and the law, each named.
    let (t_s, p_s) = start();
    let ok = BurningLaw::constant(0.5);
    let texts = [
        integrate(&v, &ok, UnburnedDensity::Compressed, t_s, v.p0, 1e-5)
            .expect_err("p_start")
            .to_string(),
        integrate(&v, &ok, UnburnedDensity::Compressed, t_s, p_s, 0.0)
            .expect_err("h")
            .to_string(),
        integrate(&v, &BurningLaw { s0: -1.0, ..ok }, UnburnedDensity::Compressed, t_s, p_s, 1e-5)
            .expect_err("law")
            .to_string(),
    ];
    assert!(texts[0].contains("p_start") && texts[0].contains("dust::vessel0d::integrate"), "{}", texts[0]);
    assert!(texts[1].contains("h") && texts[1].contains("dust::vessel0d::integrate"), "{}", texts[1]);
    assert!(texts[2].contains("s0"), "{}", texts[2]);
    // metrics: an empty trace, and one that starts past the half pressure.
    let e = metrics(&v, &ok, UnburnedDensity::Compressed, &ThinFlameTrace::default())
        .expect_err("empty")
        .to_string();
    assert!(e.contains("tr") && e.contains("dust::vessel0d::metrics"), "{e}");
    let past = ThinFlameTrace { t: vec![0.0], p: vec![0.5 * (v.p0 + v.pe)], dpdt: vec![1.0] };
    let e = metrics(&v, &ok, UnburnedDensity::Compressed, &past)
        .expect_err("past")
        .to_string();
    assert!(e.contains("tr.p[0]"), "{e}");
}
