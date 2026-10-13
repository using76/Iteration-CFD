// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.
// Provenance: see PROVENANCE.md. No GPL-licensed source was consulted.

// The DST-01 S_L(phi) gates, written once and included both by
// src/dust/laminar.rs (as dust::laminar::tests) and by tests/dust_thermo.rs,
// so both builds run the same cases.
// No GPL-licensed source was consulted.

use ofgpu::dust::laminar::{LaminarSpeed, SlFactors};
use ofgpu::error::Error;

/// (152.13) at the three phi values SPEC-LIT quotes, against the printed
/// S_L(0.62) = 0.241107668 m/s.
#[test]
fn s_l_huang2022_eq3() -> Result<(), Error> {
    let l = LaminarSpeed::huang2022();
    let s = l.s_l_phi(0.62)?;
    println!("S_L(0.62) = {s} m/s");
    assert!(
        (s - 0.2411076681537468).abs() <= 1e-12 * 0.2411076681537468,
        "S_L(0.62) = {s}"
    );
    assert!(
        (s - 0.241107668).abs() <= 5e-10,
        "S_L(0.62) = {s} is not the printed 0.241107668 to 5e-10"
    );
    let s05 = l.s_l_phi(0.5)?;
    println!("S_L(0.5) = {s05} m/s");
    assert!(
        (s05 - 0.17569627011524560).abs() <= 1e-12 * 0.17569627011524560,
        "S_L(0.5) = {s05}"
    );
    let s10 = l.s_l_phi(1.0)?;
    println!("S_L(1.0) = {s10} m/s");
    assert!(
        (s10 - 4.061623747071102e-4).abs() <= 1e-12 * 4.061623747071102e-4,
        "S_L(1.0) = {s10}"
    );
    Ok(())
}

/// With the factors off, and with a Gulder set that multiplies by one,
/// (152.14) is (152.13) bit for bit.
#[test]
fn gulder_off_is_exactly_huang2022() -> Result<(), Error> {
    let l = LaminarSpeed::huang2022();
    let g = LaminarSpeed {
        factors: SlFactors::Gulder { alpha: 0.0, beta: 0.0, t_ref: 300.0, p_ref: 101325.0 },
        ..l
    };
    for phi in [0.3, 0.62, 1.0] {
        for t in [273.0, 600.0] {
            for p in [1e5, 5e5] {
                let base = l.s_l_phi(phi)?;
                let off = l.s_l(phi, t, p)?;
                let unit = g.s_l(phi, t, p)?;
                println!("S_L({phi}, {t}, {p}) = {off} against S_L(phi) = {base}");
                assert_eq!(off.to_bits(), base.to_bits(), "the off factors changed the bits");
                assert_eq!(unit.to_bits(), base.to_bits(), "a unit Gulder factor changed the bits");
            }
        }
    }
    Ok(())
}

/// The Gulder factors scale exactly as (152.14) says: alpha = 0.5 with
/// beta = -0.25 at twice T_ref and four times p_ref multiplies by one, and
/// alpha = 1.75 at twice T_ref multiplies by 2^1.75.
#[test]
fn gulder_factors_scale() -> Result<(), Error> {
    let l = LaminarSpeed::huang2022();
    let unit = LaminarSpeed {
        factors: SlFactors::Gulder { alpha: 0.5, beta: -0.25, t_ref: 300.0, p_ref: 101325.0 },
        ..l
    };
    let base = l.s_l_phi(0.62)?;
    let s = unit.s_l(0.62, 600.0, 405300.0)?;
    println!("Gulder 0.5, -0.25 at 600 K, 405300 Pa: {s} against {base}");
    assert!((s - base).abs() <= 1e-15 * base.abs(), "the unit factor is not one: {s}");
    let hot = LaminarSpeed {
        factors: SlFactors::Gulder { alpha: 1.75, beta: 0.0, t_ref: 300.0, p_ref: 101325.0 },
        ..l
    };
    let ratio = hot.s_l(0.62, 600.0, 101325.0)? / base;
    println!("Gulder 1.75, 0 at 600 K: ratio to S_L(phi) = {ratio}");
    assert!(
        (ratio - 3.363585661014858).abs() <= 1e-14 * 3.363585661014858,
        "the temperature ratio is not 2^1.75: {ratio}"
    );
    Ok(())
}

/// The refusals: phi at or below zero or not finite, a non-positive Gulder
/// reference temperature, and a non-positive unburned temperature.
#[test]
fn refusals() -> Result<(), Error> {
    let l = LaminarSpeed::huang2022();
    let e = l.s_l_phi(0.0).unwrap_err();
    println!("phi = 0: {e}");
    assert!(e.to_string().contains("phi"));
    let e = l.s_l_phi(f64::NAN).unwrap_err();
    println!("phi = NaN: {e}");
    assert!(e.to_string().contains("phi"));
    let cold = LaminarSpeed {
        factors: SlFactors::Gulder { alpha: 0.0, beta: 0.0, t_ref: 0.0, p_ref: 101325.0 },
        ..l
    };
    let e = cold.s_l(0.62, 300.0, 1e5).unwrap_err();
    println!("t_ref = 0: {e}");
    assert!(e.to_string().contains("t_ref"));
    let e = l.s_l(0.62, -1.0, 1e5).unwrap_err();
    println!("t_u = -1: {e}");
    assert!(e.to_string().contains("t_u"));
    Ok(())
}
