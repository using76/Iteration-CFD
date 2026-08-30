// meteor-cfd - Copyright (c) 2026 주식회사 메테오시뮬레이션 (Meteo Simulation Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.
//
// Provenance: ORIGINAL - the tests for SPEC-LIT S54. The reference numbers
// are ASHRAE Handbook-Fundamentals (2021) Ch. 1 Table 2 and the IAPWS
// saturation pressure at 100 C, both PUBLISHED VALUES compared against, not
// transcribed code. Gate 54-A evaluates (S54.3) from an independent
// transcription of the thirteen coefficients rather than calling the module.
// No GPL-licensed source was consulted.

use super::*;

use crate::mesh::topology::tests::box_mesh;
use crate::Vec3;

fn gpu() -> Option<Gpu> {
    Gpu::new(0).ok()
}

fn rel(a: Scalar, b: Scalar) -> Scalar {
    let s = a.abs().max(b.abs()).max(1e-300);
    (a - b).abs() / s
}

fn block(n: [usize; 3], d: Vec3) -> crate::mesh::HostMesh {
    let (mut m, points, faces) = box_mesh(n, d);
    m.compute_geometry(&points, &faces).expect("geometry");
    m.build_cell_face_maps();
    m
}

// ==========================================================================
//  §54.8 Gate 54-A - the thirteen coefficients, against an independent
//  transcription of (S54.3)
// ==========================================================================

/// A SECOND transcription of Hyland & Wexler's coefficients, written out
/// here in a different order and a different expression tree.
///
/// The point is that a digit dropped in `crate::psychro`'s table shows up as
/// a failure rather than as agreement. Sharing the constants would make this
/// gate vacuous.
fn p_ws_reference(t: Scalar) -> Scalar {
    if t < 273.15 {
        // over ice
        let l = -5674.5359 / t
            + 6.3925247
            + t * (-0.009677843 + t * (6.2215701e-7 + t * (2.0747825e-9 + t * -9.484024e-13)))
            + 4.1635019 * t.ln();
        l.exp()
    } else {
        // over liquid water
        let l = -5800.2206 / t
            + 1.3914993
            + t * (-0.048640239 + t * (4.1764768e-5 + t * -1.4452093e-8))
            + 6.5459673 * t.ln();
        l.exp()
    }
}

#[test]
fn gate_54a_p_ws_matches_an_independent_transcription_of_the_formula() {
    let mut worst = 0.0 as Scalar;
    let mut t = 190.0 as Scalar;
    while t < 400.0 {
        worst = worst.max(rel(p_ws(t), p_ws_reference(t)));
        t += 0.37;
    }
    assert!(
        worst < 1e-6,
        "SPEC-LIT S54.8 Gate 54-A: the module's p_ws differs from an independent \
         transcription of (S54.3) by {worst:e}. Every other psychrometric quantity \
         is downstream of this one."
    );

    // The two branches meet at the split.
    let below = p_ws(273.15 - 1e-9);
    let above = p_ws(273.15);
    assert!(
        rel(below, above) < 2e-4,
        "the ice and liquid branches must nearly meet at 273.15 K: {below} and {above}"
    );
}

// ==========================================================================
//  §54.8 Gate 54-B - ASHRAE Handbook-Fundamentals Ch. 1 Table 2
// ==========================================================================

#[test]
fn gate_54b_the_ashrae_table_values_are_reproduced() {
    // p_ws, Pa. ASHRAE Table 2 quotes kPa to four figures.
    for (c, want, tol) in [
        (0.0 as Scalar, 611.2 as Scalar, 5e-4 as Scalar),
        (25.0, 3169.0, 5e-4),
        (50.0, 12349.9, 5e-4),
    ] {
        let got = p_ws(c + 273.15);
        assert!(
            rel(got, want) < tol,
            "p_ws({c} C) = {got} Pa against the table's {want} Pa"
        );
    }
    // The three numbers the design note quotes, to its own digits.
    assert!(rel(p_ws(273.15), 611.213) < 1e-5, "{}", p_ws(273.15));
    assert!(rel(p_ws(298.15), 3169.216) < 1e-5, "{}", p_ws(298.15));
    assert!(rel(p_ws(323.15), 12349.856) < 1e-5, "{}", p_ws(323.15));

    // 25 C, 50 % RH - the whole Table 2 row.
    let (t, rh) = (298.15 as Scalar, 0.5 as Scalar);
    let w = w_from_t_rh_p(t, rh, P_ATM);
    assert!(rel(w, 0.0098810) < 1e-5, "W = {w}");
    let h = h_from_t_w(t, w);
    assert!(rel(h, 50.322) < 1e-4, "h = {h} kJ/kg da");
    assert!((h - 50.4).abs() < 0.5 * 50.4 / 100.0, "and within 0.5 % of the table's ~50.4");
    let v = v_from_t_w_p(t, w, P_ATM);
    assert!(rel(v, 0.858043) < 1e-5, "v = {v}");
    assert!((v - 0.8586).abs() / 0.8586 < 5e-3, "and within 0.5 % of the table's ~0.8586");
    let td = t_d_from_pw(p_w_from_w_p(w, P_ATM));
    assert!(rel(td, 13.893) < 1e-4, "t_d = {td} C");
    assert!((td - 13.85).abs() < 0.1, "and within 0.1 K of the table's ~13.85 C");
}

/// §54.3: the ideal-gas bias, quantified and reported rather than tolerated.
#[test]
fn gate_54b_the_enhancement_factor_bias_is_the_documented_044_percent() {
    let ws = w_s(298.15, P_ATM);
    assert!(rel(ws, 0.0200811) < 1e-5, "W_s(25 C) ideal = {ws}");

    let (ideal, real, bias) = enhancement_bias(298.15, P_ATM, 1.0044);
    assert_eq!(ideal, ws, "enhancement_bias must report the same ideal value");
    assert!(
        rel(real, 0.020169) < 5e-4,
        "with f_e = 1.0044 the table value 0.020169 should come back; got {real}"
    );
    assert!(
        (bias - 0.0044).abs() < 5e-4,
        "SPEC-LIT S54.3: the documented bias is 0.44 %; measured {:.4} %",
        100.0 * bias
    );
    // It is a LOW bias - the ideal relations under-report.
    assert!(ideal < real, "the ideal relations must be BELOW the table");
    // And it is inside the half-percent gate S54.8 sets.
    assert!(bias.abs() < 5e-3);
}

/// §54.8 Gate 54-C: an independent reference for the liquid branch.
#[test]
fn gate_54c_p_ws_at_the_boiling_point_reproduces_iapws() {
    let got = p_ws(373.15);
    assert!(
        rel(got, 101418.0) < 1e-4,
        "p_ws(100 C) = {got} Pa; IAPWS gives 101.418 kPa. This is the one \
         psychrometric check in S54 whose reference is not ASHRAE, which is what \
         makes it worth having."
    );
    // And it is NOT 101325: water boils at 99.974 C at one atmosphere, so a
    // module that had been "corrected" to the round number would fail here.
    assert!(got > 101325.0, "p_ws(100 C) must exceed one atmosphere");
    // The ice branch, against ASHRAE's own -20 C row.
    assert!(rel(p_ws(253.15), 103.26) < 1e-3, "p_ws(-20 C) = {}", p_ws(253.15));
}

// ==========================================================================
//  The algebraic identities
// ==========================================================================

#[test]
fn the_humidity_ratio_round_trip_is_exact() {
    let mut worst = 0.0 as Scalar;
    for i in 1..2000 {
        let yv = i as Scalar * 4e-4;
        let w = w_from_yv(yv);
        worst = worst.max(rel(yv_from_w(w), yv));
    }
    assert!(worst < 1e-14, "W <-> Y_v round trip is off by {worst:e}");
    assert_eq!(w_from_yv(0.0), 0.0);
    assert_eq!(yv_from_w(0.0), 0.0);
}

#[test]
fn saturated_air_has_w_equal_to_w_s() {
    let mut worst = 0.0 as Scalar;
    for c in [5.0 as Scalar, 15.0, 25.0, 35.0, 45.0] {
        let t = c + 273.15;
        worst = worst.max(rel(w_from_t_rh_p(t, 1.0, P_ATM), w_s(t, P_ATM)));
        // and rh comes back as 1
        let w = w_s(t, P_ATM);
        worst = worst.max(rel(rh_from_t_w_p(t, w, P_ATM), 1.0));
    }
    assert!(worst < 1e-13, "rh = 1 does not give W = W_s: {worst:e}");
}

#[test]
fn the_dew_point_of_saturated_air_is_its_own_temperature() {
    for c in [2.0 as Scalar, 10.0, 25.0, 40.0] {
        let t = c + 273.15;
        let w = w_s(t, P_ATM);
        let td = t_d_from_pw(p_w_from_w_p(w, P_ATM));
        assert!(
            (td - c).abs() < 0.06,
            "saturated air at {c} C has dew point {td} C - ASHRAE's correlation is \
             not exact, but it is this good"
        );
    }
}

// ==========================================================================
//  §54.4 - the virtual temperature
// ==========================================================================

/// (S54.7) is EXACT, not a linearisation: checked against `rho = p M_mix/(RT)`
/// computed from the mixture molar mass, which is a different expression
/// from the identity the formula was derived from.
///
/// **And it is limited by one thing only, measured here rather than
/// asserted.** `EPS` is ASHRAE's published `0.621945`, a six-figure rounding
/// of `M_w/M_a = 0.6219453152` (Gatley et al. 2008's own molar masses). That
/// rounding is `5.07e-7` relative, and it is the entire residual: with a
/// mixture built from masses CONSISTENT with the rounded `EPS`, the identity
/// comes back at round-off. So (S54.7) is exact in the ratio it is written
/// in, and `3e-8` in practice because of a constant's last digit - which is
/// a different statement from "nearly exact", and the reason both halves are
/// checked here.
#[test]
fn gate_54d_the_virtual_temperature_is_the_density_ratio_exactly() {
    const R: Scalar = 8.314462618;
    const M_A: Scalar = 28.966e-3;
    const M_W: Scalar = 18.015268e-3;

    // (a) against the PUBLISHED molar masses, whose ratio is not exactly EPS.
    let identity = |mix: &dyn Fn(Scalar) -> Scalar| -> Scalar {
        let (t_ref, yv_ref) = (293.15 as Scalar, 0.006 as Scalar);
        let rho_ref = P_ATM * mix(yv_ref) / (R * t_ref);
        let tv_ref = virtual_temperature(t_ref, yv_ref);
        let mut worst = 0.0 as Scalar;
        for c in [10.0 as Scalar, 20.0, 25.0, 30.0, 40.0] {
            for rh in [0.0 as Scalar, 0.2, 0.5, 0.8, 1.0] {
                let t = c + 273.15;
                let yv = yv_from_t_rh_p(t, rh, P_ATM);
                let rho = P_ATM * mix(yv) / (R * t);
                worst = worst.max(rel(virtual_temperature(t, yv) / tv_ref, rho_ref / rho));
            }
        }
        worst
    };

    let published = identity(&molar_mass);
    assert!(
        published < 1e-7,
        "SPEC-LIT (S54.7): T_v/T_v,ref against rho_ref/rho came out {published:e}, \
         which is far more than the rounding of EPS can explain"
    );

    // (b) with the water mass made CONSISTENT with the published EPS. Now
    //     nothing is rounded relative to anything else and the identity is
    //     exact to f64.
    let consistent = |yv: Scalar| -> Scalar { 1.0 / (yv / (EPS * M_A) + (1.0 - yv) / M_A) };
    let exact = identity(&consistent);
    assert!(
        exact < 1e-14,
        "with masses consistent with EPS the identity must be exact to round-off; \
         measured {exact:e}"
    );

    // And the residual in (a) IS the rounding, to within a factor of a few:
    // EPS is 5.07e-7 relative from M_w/M_a, and the identity carries it
    // weighted by Y_v (at most about 0.05 here).
    let eps_error = (M_W / M_A - EPS).abs() / (M_W / M_A);
    assert!(
        (5.0e-7..6.0e-7).contains(&eps_error),
        "the published EPS should be 5.07e-7 from M_w/M_a; measured {eps_error:e}"
    );
    assert!(
        published < eps_error && published > 0.05 * eps_error,
        "the identity's residual {published:e} should be the EPS rounding \
         {eps_error:e} weighted down by Y_v, not something else"
    );
}

/// The bitwise gate that makes "the default is unmoved" a measurement.
#[test]
fn the_virtual_temperature_is_bitwise_t_at_zero_humidity() {
    for t in [1.0 as Scalar, 273.15, 300.0, 1234.5678, 1e-12] {
        assert_eq!(
            virtual_temperature(t, 0.0),
            t,
            "SPEC-LIT S54.4: T*(1.0 + c*0.0) is T*1.0 which is T BITWISE"
        );
    }
}

/// §54.4's own magnitude claim, reproduced rather than quoted.
#[test]
fn the_humidity_swing_is_worth_about_two_kelvin() {
    let t = 298.15 as Scalar;
    let yv20 = yv_from_t_rh_p(t, 0.2, P_ATM);
    let yv80 = yv_from_t_rh_p(t, 0.8, P_ATM);
    let dyv = yv80 - yv20;
    assert!(
        (dyv - 0.0118).abs() < 5e-4,
        "S54.4 says dY_v ~ 0.0118 across 20-80 % rh at 25 C; got {dyv}"
    );
    let dtv = virtual_temperature(t, yv80) - virtual_temperature(t, yv20);
    assert!(
        (dtv - 2.14).abs() < 0.05,
        "S54.4 says that is worth about 2.1 K of buoyancy; got {dtv} K"
    );
}

#[test]
fn the_molar_mass_caveat_fires_only_where_it_matters() {
    assert!(Psychrometrics::molar_mass_caveat(0.01).is_none());
    assert!(Psychrometrics::molar_mass_caveat(0.05).is_none());
    let c = Psychrometrics::molar_mass_caveat(0.09).expect("above 0.05 it must fire");
    assert!(c.contains("0.0900"), "the caveat must name the value reached: {c}");
    assert!(c.contains("EXACT for buoyancy"), "{c}");
}

// ==========================================================================
//  §54.5 - what is refused
// ==========================================================================

#[test]
fn wet_bulb_as_a_field_and_condensation_are_refused_by_name() {
    crate::io::contract::set_permissive(false);

    let e = refuse_wet_bulb_field("output/fields").unwrap_err().to_string();
    assert!(e.contains("CUDA-Graph"), "the reason must be the trip count: {e}");
    assert!(e.contains("0.3 K"), "and the accuracy of the alternative: {e}");
    assert!(e.contains("dewPoint"), "must name what IS available: {e}");

    let e = refuse_condensation("physics/humidity/condensation").unwrap_err().to_string();
    assert!(e.contains("REPORT supersaturation"), "{e}");
    assert!(e.contains("silently clipping"), "{e}");
}

/// The host-side wet bulb converges, and gives the right answer at the two
/// states where it is known in closed form.
#[test]
fn the_host_wet_bulb_converges_and_is_right_at_its_two_known_states() {
    // Saturated air: t* == t.
    for c in [10.0 as Scalar, 25.0, 35.0] {
        let t = c + 273.15;
        let w = w_s(t, P_ATM);
        let tw = t_wb(t, w, P_ATM).expect("converges");
        assert!((tw - c).abs() < 1e-6, "saturated air at {c} C has t* = {tw}");
    }
    // 25 C, 50 % rh: the psychrometric chart gives about 17.9 C.
    let t = 298.15 as Scalar;
    let w = w_from_t_rh_p(t, 0.5, P_ATM);
    let tw = t_wb(t, w, P_ATM).expect("converges");
    assert!(
        (tw - 17.9).abs() < 0.15,
        "25 C / 50 % rh should give t* near 17.9 C; got {tw}"
    );
    // And t* lies between the dew point and the dry bulb, always.
    let td = t_d_from_pw(p_w_from_w_p(w, P_ATM));
    assert!(td < tw && tw < 25.0, "t_d {td} < t* {tw} < t 25");
}

// ==========================================================================
//  The device mirror
// ==========================================================================

#[test]
fn the_device_mirrors_the_host() {
    let Some(gpu) = gpu() else { return };
    let hm = block([6, 5, 4], Vec3::new(0.1, 0.1, 0.1));
    let m = crate::mesh::GpuMesh::upload(&gpu, &hm).expect("upload");
    let n = hm.n_cells;

    let mut t = crate::field::GpuScalarField::zeros(&gpu, &m, "T").expect("T");
    let mut yv = crate::field::GpuScalarField::zeros(&gpu, &m, "Yv").expect("Yv");
    let tv: Vec<Scalar> = (0..n).map(|i| 283.0 + 0.31 * i as Scalar).collect();
    let yvv: Vec<Scalar> = (0..n).map(|i| 0.0005 * (i % 37) as Scalar).collect();
    gpu.write(&mut t.f, &tv).expect("T");
    gpu.write(&mut yv.f, &yvv).expect("Yv");

    let mut psy = Psychrometrics::new(&gpu, &m, P_ATM).expect("psy");
    psy.update(&gpu, &t, &yv).expect("update");

    let w = gpu.download(&psy.w).expect("w");
    let rh = gpu.download(&psy.rh).expect("rh");
    let h = gpu.download(&psy.h).expect("h");
    let v = gpu.download(&psy.v).expect("v");

    let mut worst = 0.0 as Scalar;
    for i in 0..n {
        let wi = w_from_yv(yvv[i]);
        worst = worst.max(rel(w[i], wi));
        worst = worst.max(rel(rh[i], rh_from_t_w_p(tv[i], wi, P_ATM)));
        worst = worst.max(rel(h[i], h_from_t_w(tv[i], wi)));
        worst = worst.max(rel(v[i], v_from_t_w_p(tv[i], wi, P_ATM)));
    }
    assert!(
        worst < 1e-14,
        "SPEC-LIT S54.7: the device kernel and the host functions must agree to \
         1e-14; measured {worst:e}"
    );

    // The virtual temperature, cells and boundary faces.
    let bt: Vec<Scalar> = (0..hm.n_boundary_faces).map(|i| 290.0 + 0.13 * i as Scalar).collect();
    let byv: Vec<Scalar> = (0..hm.n_boundary_faces).map(|i| 0.001 * (i % 11) as Scalar).collect();
    gpu.write(&mut t.bf, &bt).expect("bT");
    gpu.write(&mut yv.bf, &byv).expect("bYv");
    psy.update_virtual_temperature(&gpu, &t, &yv).expect("tv");
    let dtv = gpu.download(&psy.virtual_temperature_field().f).expect("tv");
    let dbtv = gpu.download(&psy.virtual_temperature_field().bf).expect("btv");
    for i in 0..n {
        assert!(rel(dtv[i], virtual_temperature(tv[i], yvv[i])) < 1e-15);
    }
    for i in 0..hm.n_boundary_faces {
        assert!(rel(dbtv[i], virtual_temperature(bt[i], byv[i])) < 1e-15);
    }
}

/// The buoyancy default, unmoved BY CONSTRUCTION: with `Y_v == 0` the field
/// handed to `momentum::update_buoyancy` is bit-for-bit `T`, so the buoyancy
/// flux it produces is bit-for-bit the dry one.
#[test]
fn the_buoyancy_flux_is_bitwise_the_dry_one_at_zero_humidity() {
    let Some(gpu) = gpu() else { return };
    let hm = block([5, 4, 4], Vec3::new(0.1, 0.1, 0.1));
    let m = crate::mesh::GpuMesh::upload(&gpu, &hm).expect("upload");

    let mut t = crate::field::GpuScalarField::zeros(&gpu, &m, "T").expect("T");
    let tv: Vec<Scalar> = (0..hm.n_cells).map(|i| 295.0 + 0.7 * i as Scalar).collect();
    gpu.write(&mut t.f, &tv).expect("T");
    let bt: Vec<Scalar> = (0..hm.n_boundary_faces).map(|i| 300.0 + 0.2 * i as Scalar).collect();
    gpu.write(&mut t.bf, &bt).expect("bT");
    let yv = crate::field::GpuScalarField::zeros(&gpu, &m, "Yv").expect("Yv");

    let mut psy = Psychrometrics::new(&gpu, &m, P_ATM).expect("psy");
    psy.update_virtual_temperature(&gpu, &t, &yv).expect("tv");

    let mut mom = crate::momentum::Momentum::new(
        &gpu,
        &m,
        crate::momentum::MomentumControls::default(),
        crate::momentum::BuoyancyCoeffs::default(),
    )
    .expect("momentum");
    let u = crate::field::GpuVectorField::zeros(&gpu, &m, "U").expect("U");

    mom.update_buoyancy(&gpu, &t, &u).expect("dry");
    let dry = (
        gpu.download(&mom.buoyancy_flux().f).expect("f"),
        gpu.download(&mom.buoyancy_flux().bf).expect("bf"),
    );
    mom.update_buoyancy(&gpu, psy.virtual_temperature_field(), &u).expect("moist");
    let moist = (
        gpu.download(&mom.buoyancy_flux().f).expect("f"),
        gpu.download(&mom.buoyancy_flux().bf).expect("bf"),
    );

    assert_eq!(
        dry, moist,
        "SPEC-LIT S54.4: at Y_v = 0 the virtual-temperature field is BITWISE T, so \
         the buoyancy flux must be bit-for-bit the dry one. `src/momentum.rs` is \
         not modified at all - this is what makes that provable."
    );
}

/// §54.5: supersaturation is REPORTED, with a cell count, and the field is
/// not clipped.
#[test]
fn supersaturation_is_reported_and_not_clipped() {
    let Some(gpu) = gpu() else { return };
    let hm = block([4, 4, 4], Vec3::new(0.1, 0.1, 0.1));
    let m = crate::mesh::GpuMesh::upload(&gpu, &hm).expect("upload");
    let n = hm.n_cells;

    let mut t = crate::field::GpuScalarField::zeros(&gpu, &m, "T").expect("T");
    let mut yv = crate::field::GpuScalarField::zeros(&gpu, &m, "Yv").expect("Yv");
    gpu.write(&mut t.f, &vec![293.15 as Scalar; n]).expect("T");

    // Half the cells dry, half well past saturation at 20 C.
    let ysat = yv_from_w(w_s(293.15, P_ATM));
    let vals: Vec<Scalar> =
        (0..n).map(|i| if i % 2 == 0 { 0.001 } else { ysat + 0.004 }).collect();
    gpu.write(&mut yv.f, &vals).expect("Yv");

    let mut psy = Psychrometrics::new(&gpu, &m, P_ATM).expect("psy");
    let s = psy.supersaturation(&gpu, &t, &yv).expect("supersat");
    assert_eq!(s.cells, n.div_ceil(2), "the count must be the number of wet cells");
    assert!(rel(s.worst, 0.004) < 1e-3, "the worst excess is {}", s.worst);

    // The field itself is untouched.
    let after = gpu.download(&yv.f).expect("Yv");
    assert_eq!(after[..n], vals[..n], "Y_v must NOT be clipped (S54.5)");
}

#[test]
fn a_non_positive_barometric_pressure_is_refused_by_name() {
    let Some(gpu) = gpu() else { return };
    let hm = block([2, 2, 2], Vec3::new(0.1, 0.1, 0.1));
    let m = crate::mesh::GpuMesh::upload(&gpu, &hm).expect("upload");
    let e = Psychrometrics::new(&gpu, &m, 0.0).err().expect("p_atm").to_string();
    assert!(e.contains("TOTAL barometric"), "{e}");
}

// ==========================================================================
//  §55.6 - the humidity pair tests
// ==========================================================================

/// An inlet that says `rh` must produce a different `Y_v` when the `rh`
/// changes. This is the conversion §54.6 says is printed rather than silent.
#[test]
fn pair_test_inlet_relative_humidity_moves_the_vapour_fraction() {
    let t = 291.15 as Scalar;
    let a = yv_from_t_rh_p(t, 0.35, P_ATM);
    let b = yv_from_t_rh_p(t, 0.60, P_ATM);
    assert!(
        rel(a, b) > 1e-6,
        "SPEC-LIT S13.4.1: two inlets differing only in `rh` both gave Y_v = {a}"
    );
    assert!(b > a, "more relative humidity is more vapour");
}

/// The supply temperature at fixed `rh` also moves `Y_v` - the other half of
/// the same conversion.
#[test]
fn pair_test_inlet_temperature_moves_the_vapour_fraction_at_fixed_rh() {
    let a = yv_from_t_rh_p(288.15, 0.5, P_ATM);
    let b = yv_from_t_rh_p(298.15, 0.5, P_ATM);
    assert!(rel(a, b) > 1e-6, "both gave Y_v = {a}");
    assert!(b > a, "warmer air at the same rh holds more vapour");
}

/// The virtual-temperature correction on and off must give different
/// buoyancy. This is the pair test for the ONE setting §54.4 adds to the
/// momentum path.
#[test]
fn pair_test_the_virtual_temperature_correction_moves_the_buoyancy() {
    let Some(gpu) = gpu() else { return };
    let hm = block([4, 4, 4], Vec3::new(0.1, 0.1, 0.1));
    let m = crate::mesh::GpuMesh::upload(&gpu, &hm).expect("upload");

    let mut t = crate::field::GpuScalarField::zeros(&gpu, &m, "T").expect("T");
    gpu.write(&mut t.f, &vec![300.0 as Scalar; hm.n_cells]).expect("T");
    gpu.write(&mut t.bf, &vec![300.0 as Scalar; hm.n_boundary_faces]).expect("bT");
    let mut yv = crate::field::GpuScalarField::zeros(&gpu, &m, "Yv").expect("Yv");
    let humid = yv_from_t_rh_p(300.0, 0.9, P_ATM);
    gpu.write(&mut yv.f, &vec![humid; hm.n_cells]).expect("Yv");
    gpu.write(&mut yv.bf, &vec![humid; hm.n_boundary_faces]).expect("bYv");

    let mut psy = Psychrometrics::new(&gpu, &m, P_ATM).expect("psy");
    psy.update_virtual_temperature(&gpu, &t, &yv).expect("tv");

    let mut mom = crate::momentum::Momentum::new(
        &gpu,
        &m,
        crate::momentum::MomentumControls::default(),
        crate::momentum::BuoyancyCoeffs::default(),
    )
    .expect("momentum");
    let u = crate::field::GpuVectorField::zeros(&gpu, &m, "U").expect("U");

    mom.update_buoyancy(&gpu, &t, &u).expect("off");
    let off = gpu.download(&mom.buoyancy_flux().f).expect("f");
    mom.update_buoyancy(&gpu, psy.virtual_temperature_field(), &u).expect("on");
    let on = gpu.download(&mom.buoyancy_flux().f).expect("f");

    let worst = off
        .iter()
        .zip(&on)
        .fold(0.0 as Scalar, |m, (a, b)| m.max((a - b).abs()));
    assert!(
        worst > 1e-9,
        "SPEC-LIT S13.4.1: turning the virtual-temperature correction on at 90 % rh \
         left the buoyancy flux unchanged - (S54.7) was ignored"
    );
}

/// SPEC-LIT §54.1: humidity is one more species, and its two transport
/// coefficients must each move the field. Checked on the species machinery
/// this section reuses **unmodified**.
#[test]
fn pair_test_the_humidity_transport_coefficients_move_the_field() {
    let a = crate::species::SpeciesCoeffs { d: 2.5e-5, sc_t: 0.7 };
    let b = crate::species::SpeciesCoeffs { d: 5.0e-5, sc_t: 0.7 };
    let c = crate::species::SpeciesCoeffs { d: 2.5e-5, sc_t: 1.4 };
    assert_ne!(a, b, "D_v must be a distinguishable setting");
    assert_ne!(a, c, "Sc_t must be a distinguishable setting");

    // The effective diffusivity the species equation assembles is
    // D + nu_t/Sc_t, so both entries move it - which is the whole reason
    // they are two entries and not one.
    let nut = 1e-3 as Scalar;
    let eff = |s: crate::species::SpeciesCoeffs| s.d + nut / s.sc_t;
    assert!(rel(eff(a), eff(b)) > 1e-6, "D_v does not reach D_eff");
    assert!(rel(eff(a), eff(c)) > 1e-6, "Sc_t does not reach D_eff");
}
