// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.
// Provenance: see PROVENANCE.md. No GPL-licensed source was consulted.

// The DST-01 thermo gates, written once and included both by
// src/dust/tests_thermo.rs (as dust::thermo::tests) and by
// tests/dust_thermo.rs, so both builds run the same nine cases.
// No GPL-licensed source was consulted.

use ofgpu::chem::thermo::{P_BAR, ThermoSet};
use ofgpu::dust::thermo::{
    EquivalentGas, FormulaInputs, PrescribedInputs, StarchFuel, tm4513_gas_set,
};
use ofgpu::error::Error;

/// The reference gas and charge of SPEC-LIT §152.10: TM-4513 records, the
/// H22 Table-1 fuel, phi = 0.62.
fn rembe_gas() -> Result<(ThermoSet, EquivalentGas), Error> {
    let gas = tm4513_gas_set()?;
    let g = EquivalentGas::formula(&FormulaInputs::huang2022_rembe()?, &gas)?;
    Ok((gas, g))
}

/// (152.11) and (152.10) at T_0 = 273 K, phi = 0.62, air 1 : 3.76, the dust
/// counted as gas moles, against the oracle values of SPEC-LIT §152.10.
#[test]
fn aicc_corn_starch_phi_0_62() -> Result<(), Error> {
    let (_, g) = rembe_gas()?;
    let a = g.aicc(273.0)?;
    println!("AICC phi = 0.62: P_max/p_0 = {}", a.p_ratio);
    assert!(
        8.57 * 0.99 <= a.p_ratio && a.p_ratio <= 8.57 * 1.01,
        "P_max/p_0 = {} is outside 8.57 +/- 1 %",
        a.p_ratio
    );
    println!("AICC phi = 0.62: T_v = {} K (oracle 2173.4)", a.t_v);
    assert!(
        (a.t_v - 2173.4).abs() <= 0.05,
        "T_v = {} K is outside 2173.4 +/- 0.05",
        a.t_v
    );
    let b = g.isobaric_burned(273.0)?;
    println!("isobaric phi = 0.62: T_b = {} K, sigma = {}", b.t_b, b.sigma);
    assert!(
        (b.t_b - 1779.5).abs() <= 0.05,
        "T_b = {} K is outside 1779.5 +/- 0.05",
        b.t_b
    );
    assert!(
        (b.sigma - 7.042).abs() <= 0.0005,
        "sigma = {} is outside 7.042 +/- 0.0005",
        b.sigma
    );
    assert!(a.t_v > b.t_b, "T_v = {} K must exceed the isobaric T_b = {} K", a.t_v, b.t_b);
    println!("W_U = {} g/mol, W_B = {} g/mol", g.w_u() * 1e3, g.w_b() * 1e3);
    assert!(
        (g.w_u() * 1e3 - 31.888).abs() <= 5e-4,
        "W_U = {} g/mol is outside 31.888 +/- 5e-4",
        g.w_u() * 1e3
    );
    assert!(
        (g.w_b() * 1e3 - 29.517).abs() <= 5e-4,
        "W_B = {} g/mol is outside 29.517 +/- 5e-4",
        g.w_b() * 1e3
    );
    Ok(())
}

/// (152.6) at 298.15 K against the printed 2521 kJ/mol, the derived a5, the
/// record's exact formation enthalpy, the formula molar mass, and the NASA
/// TP-2002 Table B1 formation enthalpies.
#[test]
fn heat_of_reaction_from_formation_enthalpies() -> Result<(), Error> {
    let (_, g) = rembe_gas()?;
    let dh = g.heat_of_reaction()?;
    println!("dH_r(298.15) = {} kJ/mol", dh / 1e3);
    assert!(
        (-dh - 2521e3).abs() <= 1e-3 * 2521e3,
        "dH_r = {} J/mol is outside 2521 kJ/mol +/- 0.1 %",
        dh
    );
    let f = StarchFuel::huang2022_table1()?;
    println!("derived a5 = {} K (SPEC-LIT §152.2: -98821.7)", f.a5());
    assert!(
        (f.a5() - (-98821.7)).abs() <= 0.05,
        "a5 = {} K is outside -98821.7 +/- 0.05",
        f.a5()
    );
    let ratio = f.species().h(298.15)? / f.hf;
    println!("h_STARCH(298.15)/Hf_cs = {ratio}");
    assert!(
        (ratio - 1.0).abs() <= 1e-12,
        "the record's h(298.15) is not the stated Hf_cs: ratio = {ratio}"
    );
    println!("W_cs = {} kg/mol", f.molar_mass());
    assert!(
        (f.molar_mass() - 0.15968406).abs() <= 1e-12,
        "W_cs = {} kg/mol is outside 0.15968406 +/- 1e-12",
        f.molar_mass()
    );
    let tp = 6.0 * (-393.510e3) + 3.94 * (-241.826e3) + 792.6e3;
    println!("dH_r = {} J/mol against TP-2002 B1 {tp} J/mol", dh);
    assert!(
        (dh - tp).abs() <= 1e-4 * tp.abs(),
        "dH_r = {} J/mol is outside 1e-4 relative of {tp}",
        dh
    );
    Ok(())
}

/// The pseudo-species are exact NASA-7 combinations: cp, h and s per kg at
/// 50 temperatures on [200, 6000] K equal the Y_k-weighted component sums to
/// 1e-12 relative, h(T, 0) and h(T, 1) are the end species bit for bit, the
/// element balance of the reaction is exact at five values of phi, and the
/// thermo set carries exactly the two pseudo-species.
#[test]
fn pseudo_species_equal_component_sums() -> Result<(), Error> {
    let (_, g) = rembe_gas()?;
    let sums =
        |comps: &[(ofgpu::chem::thermo::SpeciesThermo, f64)], t: f64| -> Result<(f64, f64, f64, f64), Error> {
            let (mut c, mut h, mut s, mut y) = (0.0, 0.0, 0.0, 0.0);
            for (r, y_k) in comps {
                c += y_k * r.cp(t)? / r.molar_mass;
                h += y_k * r.h(t)? / r.molar_mass;
                s += y_k * r.s(t)? / r.molar_mass;
                y += y_k;
            }
            Ok((c, h, s, y))
        };
    let mut worst = 0.0f64;
    for i in 0..50 {
        let t = 200.0 + (i as f64) * 5800.0 / 49.0;
        let (cu, hu, su, yu) = sums(&g.components_u, t)?;
        let (cb, hb, sb, yb) = sums(&g.components_b, t)?;
        for (side, pseudo, sum) in [
            ("cp_U", g.cp_u(t)?, cu),
            ("h_U", g.h_u(t)?, hu),
            ("s_U", g.s_u(t)?, su),
            ("cp_B", g.cp_b(t)?, cb),
            ("h_B", g.h_b(t)?, hb),
            ("s_B", g.s_b(t)?, sb),
        ] {
            let e = (pseudo - sum).abs() / sum.abs();
            worst = worst.max(e);
            assert!(
                e <= 1e-12,
                "{side} at {t} K: pseudo {pseudo} against component sum {sum}"
            );
        }
        assert!((yu - 1.0).abs() <= 1e-15, "sum Y_k^U = {yu}");
        assert!((yb - 1.0).abs() <= 1e-15, "sum Y_k^B = {yb}");
        assert_eq!(g.h(t, 0.0)?.to_bits(), g.h_u(t)?.to_bits(), "h({t}, 0) is not h_U");
        assert_eq!(g.h(t, 1.0)?.to_bits(), g.h_b(t)?.to_bits(), "h({t}, 1) is not h_B");
    }
    println!("worst pseudo-vs-sum relative error over 50 temperatures: {worst:.3e}");
    let base = FormulaInputs::huang2022_rembe()?;
    for (num, den) in [(62, 100), (1, 1), (1, 2), (31, 50), (3, 10)] {
        let inp = FormulaInputs {
            fuel: base.fuel.clone(),
            n2_per_o2_centi: base.n2_per_o2_centi,
            phi_num: num,
            phi_den: den,
        };
        let r = inp.element_residual_exact()?;
        println!("phi = {num}/{den}: element residual C, H, O, N = {r:?}");
        assert_eq!(r, [(0, 1); 4], "phi = {num}/{den}: the element balance is not exact");
    }
    let ts = g.thermo_set()?;
    let names: Vec<&str> = ts.species().iter().map(|s| s.name.as_str()).collect();
    println!("thermo_set holds {names:?}");
    assert_eq!(names, ["DUST_U", "DUST_B"]);
    Ok(())
}

/// The Leeds prescribed mode at two values of the stated c_p: sigma against
/// 5.06 and against its exact value 1500*0.0328/(328*0.0297), the burned
/// temperature, the straight adiabatic line, and the refusal of the heat of
/// reaction.
#[test]
fn prescribed_mode_sigma_5_06() -> Result<(), Error> {
    for cp in [1000.0, 1250.0] {
        let pg = EquivalentGas::prescribed(&PrescribedInputs::huang2020_leeds(cp))?;
        let b = pg.isobaric_burned(328.0)?;
        println!("prescribed cp = {cp}: sigma = {}, T_b = {} K", b.sigma, b.t_b);
        assert!(
            (b.sigma - 5.06).abs() <= 0.01,
            "sigma = {} is outside 5.06 +/- 0.01",
            b.sigma
        );
        let exact = 1500.0 * 0.0328 / (328.0 * 0.0297);
        assert!(
            (b.sigma - exact).abs() <= 1e-10 * exact,
            "sigma = {} is not the exact {exact}",
            b.sigma
        );
        assert!(
            (b.t_b - 1500.0).abs() <= 1e-7,
            "T_b = {} K is not the measured 1500 K",
            b.t_b
        );
        let h1500 = pg.h_u(1500.0)?.abs();
        for c in [0.25, 0.5, 0.75] {
            let t = 328.0 + c * (1500.0 - 328.0);
            let h = pg.h(t, c)?;
            println!("prescribed cp = {cp}: h({t}, {c}) = {h} against h_U(328)");
            assert!(
                (h - pg.h_u(328.0)?).abs() <= 1e-12 * h1500,
                "the adiabatic line is bent at c = {c}: h = {h}"
            );
        }
        assert!(
            pg.heat_of_reaction().is_err(),
            "the prescribed mode must refuse the heat of reaction"
        );
    }
    Ok(())
}

/// Every refusal names its input and its value: phi outside (0, 1], a
/// component the gas set lacks, a progress variable outside [0, 1], a
/// temperature below the records' range, a quadratic starch coefficient, and
/// a prescribed t_b not above t_u.
#[test]
fn refusals_name_the_input() -> Result<(), Error> {
    let (gas, g) = rembe_gas()?;
    let base = FormulaInputs::huang2022_rembe()?;
    let at = |num: i64, den: i64| FormulaInputs {
        fuel: base.fuel.clone(),
        n2_per_o2_centi: base.n2_per_o2_centi,
        phi_num: num,
        phi_den: den,
    };
    let e = at(3, 2).element_residual_exact().unwrap_err();
    println!("phi = 3/2: {e}");
    assert!(e.to_string().contains("phi"), "phi = 3/2: {e}");
    let e = at(1, 0).element_residual_exact().unwrap_err();
    println!("phi_den = 0: {e}");
    assert!(e.to_string().contains("phi"), "phi_den = 0: {e}");
    let e = at(0, 1).element_residual_exact().unwrap_err();
    println!("phi = 0: {e}");
    assert!(e.to_string().contains("phi"), "phi = 0: {e}");
    // The mole numbers are exact rationals, so the refusal of phi > 1 must be
    // exact too: (1e18 + 1)/1e18 rounds to exactly 1.0 in f64 yet is richer
    // than stoichiometric (SPEC-LIT §152.2).
    let rich = at(1_000_000_000_000_000_001, 1_000_000_000_000_000_000);
    let e = rich.element_residual_exact().unwrap_err();
    println!("phi = (1e18 + 1)/1e18: {e}");
    assert!(e.to_string().contains("phi"), "phi just above 1: {e}");
    assert!(
        EquivalentGas::formula(&rich, &gas).is_err(),
        "phi = (1e18 + 1)/1e18 must refuse the equivalent gas too"
    );
    let without: Vec<_> = gas.species().iter().filter(|s| s.name != "CO2").cloned().collect();
    let no_co2 = ThermoSet::from_species(without, P_BAR, "TM-4513 Table II without CO2")?;
    let e = EquivalentGas::formula(&base, &no_co2).unwrap_err();
    println!("gas set without CO2: {e}");
    assert!(e.to_string().contains("CO2"), "the missing species is not named: {e}");
    let e = g.h(1000.0, 1.5).unwrap_err();
    println!("c = 1.5: {e}");
    assert!(e.to_string().contains("c"), "the progress variable is not named: {e}");
    assert!(g.h_u(100.0).is_err(), "h_U below 200 K must be refused");
    let text = include_str!("../../../reference/huang2022-rembe/corn_starch_table1.csv");
    let mangled = text.replace(",0.10056,0,0,0,", ",0.10056,0.5,0,0,");
    let e = StarchFuel::from_table1_csv(&mangled).unwrap_err();
    println!("a2 = 0.5: {e}");
    assert!(e.to_string().contains("a2"), "the quadratic coefficient is not named: {e}");
    // A negative atom count is refused at or below zero, with the column
    // named (SPEC-LIT §152.1).
    let neg_c = text.replace("6,7.88,4.98,", "-6,7.88,4.98,");
    let e = StarchFuel::from_table1_csv(&neg_c).unwrap_err();
    println!("n_C = -6: {e}");
    assert!(e.to_string().contains("n_C"), "the negative atom count is not named: {e}");
    let neg_h = text.replace("6,7.88,4.98,", "6,-7.88,4.98,");
    let e = StarchFuel::from_table1_csv(&neg_h).unwrap_err();
    println!("n_H = -7.88: {e}");
    assert!(e.to_string().contains("n_H"), "the negative atom count is not named: {e}");
    let bad = PrescribedInputs { t_b: 300.0, ..PrescribedInputs::huang2020_leeds(1000.0) };
    assert!(
        EquivalentGas::prescribed(&bad).is_err(),
        "t_b = 300 K not above t_u = 328 K must be refused"
    );
    Ok(())
}
