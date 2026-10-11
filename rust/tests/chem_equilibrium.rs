// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.
// Provenance: see PROVENANCE.md. No GPL-licensed source was consulted.

//! CHR-04's gates: gas-phase chemical equilibrium by Gibbs minimisation
//! (SPEC-LIT §131, `src/chem/equilibrium.rs`) against the supervisor's
//! 40-digit oracle - the element-potential conditions of §131.1 solved
//! directly - on the ten-species TM-4513 fixture of §130.7, with element
//! conservation, detailed balance through [`rates_of_progress`], the
//! adiabatic flame temperature, and the §131.5 refusals. GRI-Mech 3.0 is
//! user-supplied by path (§129.6), so its test skips by name when
//! `OFGPU_GRI30_MECH` or `OFGPU_GRI30_THERMO` is unset, and
//! `OFGPU_REQUIRE_MECH=1` turns the skip into a failure.
//!
//! An integration test, so it runs in both builds: it links the normal
//! library, which is f64 in every build.
//! No GPL-licensed source was consulted.

// The oracle's values are kept to the digits it printed.
#![allow(clippy::excessive_precision)]

use std::env;
use std::fs;

use ofgpu::chem::equilibrium::element_moles;
use ofgpu::chem::kinetics::rates_of_progress;
use ofgpu::chem::{
    equilibrate_hp, equilibrate_tp, ElementTable, EqOptions, Equilibrium, Mechanism, SpeciesThermo,
    ThermoSet, P_ATM, P_BAR,
};

/// rel(x, t) = |x - t| / |t|, the class P comparison.
fn rel(x: f64, t: f64) -> f64 {
    (x - t).abs() / t.abs()
}

/// The ten TM-4513 records of `reference/mcbride-gordon-reno1993/
/// table_II.csv` (SPEC-LIT §128.6, §130.7).
const TABLE_II: &str = include_str!("../../reference/mcbride-gordon-reno1993/table_II.csv");

/// The table's species codes: name and elements (§128.6's code list).
#[allow(clippy::type_complexity)]
const CODES: &[(&str, &str, &[(&str, u32)])] = &[
    ("1", "N2", &[("N", 2)]),
    ("2", "O2", &[("O", 2)]),
    ("5", "CH4", &[("C", 1), ("H", 4)]),
    ("6", "CO2", &[("C", 1), ("O", 2)]),
    ("7", "H2O", &[("H", 2), ("O", 1)]),
    ("8", "CO", &[("C", 1), ("O", 1)]),
    ("9", "H2", &[("H", 2)]),
    ("10", "OH", &[("O", 1), ("H", 1)]),
    ("11", "O", &[("O", 1)]),
    ("12", "H", &[("H", 1)]),
];

/// The records as a set at `P_BAR`: TM-4513 tabulates at 1 bar (SPEC-LIT
/// §128.4), and the mechanism reader moves the set to `P_ATM` itself. The
/// row whose `t_lo` is 1000 is `upper`, the row whose `t_lo` is 200 is
/// `lower`; the bounds are 200/1000/6000 K for all ten species.
/// `molar_mass` starts as the table's own column - the mechanism reader
/// rebuilds it from the element weights.
fn tm_set() -> ThermoSet {
    let mut rows: Vec<(String, f64, f64, [f64; 7])> = Vec::new();
    for line in TABLE_II.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') || line.starts_with("species") {
            continue;
        }
        let c: Vec<&str> = line.split(',').collect();
        rows.push((
            c[0].to_string(),
            c[1].parse().unwrap(),
            c[3].parse().unwrap(),
            [
                c[4].parse().unwrap(),
                c[5].parse().unwrap(),
                c[6].parse().unwrap(),
                c[7].parse().unwrap(),
                c[8].parse().unwrap(),
                c[9].parse().unwrap(),
                c[10].parse().unwrap(),
            ],
        ));
    }
    let mut v = Vec::new();
    for (code, name, els) in CODES {
        let pick = |t_lo: f64| {
            rows.iter()
                .find(|(c, t, _, _)| c == code && *t == t_lo)
                .map(|r| (r.2, r.3))
                .unwrap()
        };
        let (mol_weight, upper) = pick(1000.0);
        let (_, lower) = pick(200.0);
        v.push(SpeciesThermo {
            name: name.to_string(),
            elements: els.iter().map(|(s, n)| (s.to_string(), *n)).collect(),
            molar_mass: mol_weight / 1000.0,
            t_low: 200.0,
            t_mid: 1000.0,
            t_high: 6000.0,
            upper,
            lower,
            line: 0,
        });
    }
    ThermoSet::from_species(v, P_BAR, "TM-4513").unwrap()
}

/// A mechanism of the given reaction lines over the ten TM-4513 species;
/// the concentration order is H2 O2 H2O OH O H CO CO2 CH4 N2 (0..9).
fn mech(rx: &[&str]) -> Mechanism {
    let mut text: Vec<&str> = vec![
        "ELEMENTS H O C N END",
        "SPECIES H2 O2 H2O OH O H CO CO2 CH4 N2 END",
        "REACTIONS",
    ];
    text.extend_from_slice(rx);
    text.push("END");
    Mechanism::read_chemkin(&text.join("\n"), "kin-fixture", Some(&tm_set()))
        .unwrap_or_else(|e| panic!("the fixture does not read: {e}"))
}

/// The 14-reaction fixture of SPEC-LIT §130.7. Its rate numbers are
/// placeholders, not any paper's; the R-numbers of the oracle below are
/// reaction index + 1.
fn fixture() -> Mechanism {
    mech(&[
        "H + O2 <=> O + OH                 1.0E+14  -0.5  15000.0",
        "O + H2 <=> H + OH                 1.0E+05   2.5   6000.0",
        "OH + H2 <=> H + H2O               2.0E+08   1.5   3000.0",
        "H + OH + M <=> H2O + M            2.0E+22  -2.0      0.0",
        "  H2O/6.0/ H2/2.0/",
        "O + O + M <=> O2 + M              1.0E+17  -1.0      0.0",
        "H + H (+M) <=> H2 (+M)            1.0E+13   0.0      0.0",
        "  LOW / 5.0E+18 -1.0 0.0 /",
        "  H2O/5.0/",
        "CO + O (+M) <=> CO2 (+M)          2.0E+10   0.0   2500.0",
        "  LOW / 5.0E+14 0.0 3000.0 /",
        "  TROE / 0.6 100.0 1500.0 5000.0 /",
        "O + H2 (+N2) <=> H2O (+N2)        1.0E+12   0.0   1000.0",
        "  LOW / 1.0E+18 -1.0 0.0 /",
        "  TROE / 0.7 200.0 1000.0 /",
        "H + O (+M) <=> OH (+M)            5.0E+12   0.0      0.0",
        "  LOW / 4.0E+18 -1.0 0.0 /",
        "  SRI / 0.5 800.0 200.0 1.2 0.1 /",
        "CO + OH <=> CO2 + H               4.0E+06   1.5   -500.0",
        "  REV / 2.0E+08 1.5 23000.0 /",
        "CH4 + 2 O2 => CO2 + 2 H2O         1.0E+12   0.0  30000.0",
        "  FORD /CH4 0.2/ FORD /O2 1.3/",
        "2 H2 + O2 => 2 H2O                1.0E+10   0.0  30000.0",
        "OH + OH <=> O + H2O               3.0E+04   2.4  -2000.0",
        "  DUP",
        "OH + OH <=> O + H2O               1.0E+12   0.0   1000.0",
        "  DUP",
    ])
}

/// Mole fractions to mass fractions against `set`: y_j = x_j W_j / sum x W.
fn y_from_x(set: &ThermoSet, x: &[f64]) -> Vec<f64> {
    let xw: Vec<f64> = x
        .iter()
        .zip(set.species())
        .map(|(xj, s)| xj * s.molar_mass)
        .collect();
    let s: f64 = xw.iter().sum();
    xw.iter().map(|v| v / s).collect()
}

/// Stoichiometric CH4-air by moles: 1 CH4 + 2 O2 + 7.52 N2 (phi = 1), in
/// the species order H2 O2 H2O OH O H CO CO2 CH4 N2.
fn y_ch4_air(set: &ThermoSet) -> Vec<f64> {
    let mut x = vec![0.0; set.species().len()];
    x[1] = 2.0;
    x[8] = 1.0;
    x[9] = 7.52;
    y_from_x(set, &x)
}

/// Stoichiometric H2-air by moles: 2 H2 + 1 O2 + 3.76 N2 (phi = 1).
fn y_h2_air(set: &ThermoSet) -> Vec<f64> {
    let mut x = vec![0.0; set.species().len()];
    x[0] = 2.0;
    x[1] = 1.0;
    x[9] = 3.76;
    y_from_x(set, &x)
}

/// The supervisor's 40-digit oracle (SPEC-LIT §131.4): the element-
/// potential conditions of §131.1 solved directly at the same thermo data,
/// not by the Newton above. Species order H2 O2 H2O OH O H CO CO2 CH4 N2;
/// pi order H O C N.
const OR_B0: [f64; 4] = [
    13.759808053099385481,
    13.759808053099385481,
    3.4399520132748463702,
    51.736878279653689408,
];
/// The CH4-air reactants' enthalpy at 298.15 K, J/kg.
const OR_H0: f64 = -256618.95614827009036;
/// CH4-air tp, T = 2000 K, p = 1 atm.
const OR_TP_NT: f64 = 36.272218301275177722;
const OR_TP_PI: [f64; 4] = [
    -13.072166930896799562,
    -17.527281773987463586,
    -22.677971679075514222,
    -13.632797148201778174,
];
const OR_TP_X: [f64; 10] = [
    0.0012595469463745432721,
    0.0018293687476956318061,
    0.18795875089814110185,
    0.00085447068058103909845,
    0.000028430533461381431057,
    0.000057335727307357177469,
    0.0028261885591149365937,
    0.092010911965114985037,
    1.5344998139458604572e-18,
    0.7131749959422090222,
];

/// CH4-air hp from 298.15 K, p = 1 atm.
const OR_HP_T: f64 = 2231.1042475877234514;
const OR_HP_NT: f64 = 36.452609749342997679;
const OR_HP_PI: [f64; 4] = [
    -12.743666237439135872,
    -17.192882843744583754,
    -21.578075888805278173,
    -13.822508907941732102,
];
const OR_HP_X: [f64; 10] = [
    0.0034614730726787496702,
    0.0053000349645764462549,
    0.18356089463982552693,
    0.0030348505825709250794,
    0.0002376735603284847911,
    0.00039155111305922052255,
    0.0086979201515583774117,
    0.085669864128601297287,
    2.5624325887745652953e-17,
    0.70964573778680094643,
];
/// H2-air tp, T = 2000 K, p = 1 atm; the species are H2 O2 H2O OH O H N2
/// (indices 0-5 and 9), and CO, CO2, CH4 are exactly 0. pi order H O N.
const OR_H2_NT: f64 = 40.818738890255862917;
const OR_H2_PI: [f64; 3] = [
    -12.661937648272184691,
    -17.745686577744385416,
    -13.677925751897957577,
];
const OR_H2_X: [f64; 7] = [
    0.0028611128432574666887,
    0.0011819449605453684846,
    0.34318727524416861168,
    0.0010351552237077072952,
    0.000022852462641092788743,
    0.00008641430465643343384,
    0.65162524496102331963,
];

/// The CH4-air pair of the oracle: the tp state at 2000 K and the hp state
/// from 298.15 K, both at 1 atm with the default options.
fn ch4_pair(set: &ThermoSet) -> (Equilibrium, Equilibrium) {
    let y0 = y_ch4_air(set);
    let h0 = set.h_mass(298.15, &y0).unwrap();
    let tp = equilibrate_tp(set, 2000.0, P_ATM, &y0, &EqOptions::default()).unwrap();
    let hp = equilibrate_hp(set, h0, P_ATM, &y0, &EqOptions::default()).unwrap();
    (tp, hp)
}

/// Test 2's oracle comparison: every X_j, pi_i and n_total - and T when the
/// oracle names one - within 1e-10 relative; the worst values are printed.
fn oracle_check(
    eq: &Equilibrium,
    name: &str,
    x: &[f64; 10],
    pi: &[f64; 4],
    nt: f64,
    t: Option<f64>,
) {
    let xt = eq.mole_fractions();
    let mut wx = 0.0f64;
    for j in 0..10 {
        let r = rel(xt[j], x[j]);
        wx = wx.max(r);
        assert!(
            r <= 1e-10,
            "{name}: X_{j} = {} against the oracle's {:e}, rel {r:.3e}",
            xt[j],
            x[j]
        );
    }
    let wn = rel(eq.n_total, nt);
    assert!(
        wn <= 1e-10,
        "{name}: n_total = {} against the oracle's {nt:e}, rel {wn:.3e}",
        eq.n_total
    );
    let mut wp = 0.0f64;
    for i in 0..4 {
        let r = rel(eq.pi[i], pi[i]);
        wp = wp.max(r);
        assert!(
            r <= 1e-10,
            "{name}: pi_{i} = {} against the oracle's {:e}, rel {r:.3e}",
            eq.pi[i],
            pi[i]
        );
    }
    let wt = t.map_or(0.0, |tt| {
        let r = rel(eq.t, tt);
        assert!(
            r <= 1e-10,
            "{name}: t = {} against the oracle's {tt:e}, rel {r:.3e}",
            eq.t
        );
        r
    });
    println!(
        "{name}: worst X rel {wx:.3e}, worst pi rel {wp:.3e}, n_total rel {wn:.3e}, \
         t rel {wt:.3e}, {} iterations",
        eq.iterations
    );
}

/// The (A) element check of test 1: the test's own sum of a_ij n_j over the
/// records meets b_i° to 1e-12 max b°, and sum n_j W_j = 1 kg.
fn element_balance(set: &ThermoSet, eq: &Equilibrium, name: &str) -> (f64, f64) {
    let bmax = eq.b0.iter().cloned().fold(0.0f64, f64::max);
    let mut worst = 0.0f64;
    for (i, el) in eq.elements.iter().enumerate() {
        let mut b_i = 0.0f64;
        for (nj, s) in eq.n.iter().zip(set.species()) {
            for (sym, cnt) in &s.elements {
                if sym == el {
                    b_i += *cnt as f64 * nj;
                }
            }
        }
        let r = (b_i - eq.b0[i]).abs();
        worst = worst.max(r);
        assert!(
            r <= 1e-12 * bmax,
            "{name}: element {el}: {b_i:e} against b0 = {:e}, residual {r:.3e}",
            eq.b0[i]
        );
    }
    let mass: f64 =
        eq.n.iter()
            .zip(set.species())
            .map(|(nj, s)| nj * s.molar_mass)
            .sum();
    let rm = (mass - 1.0).abs();
    assert!(
        rm <= 1e-12,
        "{name}: sum n_j W_j = {mass:e}, |1 - mass| = {rm:.3e}"
    );
    (worst, rm)
}

/// Test 1: the elements' moles are conserved at every gate state, b0 is
/// the mixture's own `element_moles` bitwise, and the mass sums to 1 kg
/// (SPEC-LIT §131.4, class A).
#[test]
fn eq_elements_conserved() {
    let m = fixture();
    let set = m.thermo();
    let y_ch4 = y_ch4_air(set);
    let y_h2 = y_h2_air(set);
    let h_ch4 = set.h_mass(298.15, &y_ch4).unwrap();
    let cases: [(&str, &Vec<f64>, Equilibrium); 3] = [
        (
            "ch4-air tp 2000 K",
            &y_ch4,
            equilibrate_tp(set, 2000.0, P_ATM, &y_ch4, &EqOptions::default()).unwrap(),
        ),
        (
            "ch4-air hp",
            &y_ch4,
            equilibrate_hp(set, h_ch4, P_ATM, &y_ch4, &EqOptions::default()).unwrap(),
        ),
        (
            "h2-air tp 2000 K",
            &y_h2,
            equilibrate_tp(set, 2000.0, P_ATM, &y_h2, &EqOptions::default()).unwrap(),
        ),
    ];
    let (mut worst_el, mut worst_mass) = (0.0f64, 0.0f64);
    for (name, y0, eq) in &cases {
        let (els, b) = element_moles(set, y0).unwrap();
        assert_eq!(&els, &eq.elements, "{name}: the element order");
        assert_eq!(b, eq.b0, "{name}: b0 is not bitwise element_moles");
        let (re, rm) = element_balance(set, eq, name);
        worst_el = worst_el.max(re);
        worst_mass = worst_mass.max(rm);
        println!(
            "{name}: {} iterations, element residual {re:.3e}",
            eq.iterations
        );
    }
    println!(
        "eq_elements_conserved: worst element residual {worst_el:.3e}, \
         worst |mass - 1| {worst_mass:.3e}"
    );
}

/// Test 2: the CH4-air tp and hp states against the supervisor's 40-digit
/// oracle (SPEC-LIT §131.4, class P).
#[test]
fn eq_matches_oracle() {
    let m = fixture();
    let set = m.thermo();
    let y0 = y_ch4_air(set);
    let h0 = set.h_mass(298.15, &y0).unwrap();
    let rh = rel(h0, OR_H0);
    println!("h0 = {h0} J/kg against the oracle's {OR_H0:e}, rel {rh:.3e}");
    assert!(rh <= 1e-12, "h0 rel {rh:.3e}");
    let (tp, hp) = ch4_pair(set);
    let mut worst_b = 0.0f64;
    for (i, b) in tp.b0.iter().enumerate() {
        let r = rel(*b, OR_B0[i]);
        worst_b = worst_b.max(r);
        assert!(
            r <= 1e-13,
            "b0_{i} = {b:e} against the oracle's {:e}, rel {r:.3e}",
            OR_B0[i]
        );
    }
    println!("b0 against the oracle: worst rel {worst_b:.3e}");
    oracle_check(&tp, "ch4-air tp", &OR_TP_X, &OR_TP_PI, OR_TP_NT, None);
    oracle_check(
        &hp,
        "ch4-air hp",
        &OR_HP_X,
        &OR_HP_PI,
        OR_HP_NT,
        Some(OR_HP_T),
    );
}

/// Test 3: detailed balance (SPEC-LIT §130.5, §131.4) at the equilibrium
/// states, through [`rates_of_progress`]; every reaction the rule cannot
/// check is printed EXCLUDED with its reason.
#[test]
fn eq_detailed_balance_holds() {
    let m = fixture();
    let set = m.thermo();
    let (tp, hp) = ch4_pair(set);
    for (name, eq) in [("ch4-air tp", &tp), ("ch4-air hp", &hp)] {
        let q = rates_of_progress(&m, eq.t, &eq.concentrations()).unwrap();
        let (mut checked, mut excluded) = (0usize, 0usize);
        let mut worst = 0.0f64;
        for (r, (rx, qr)) in m.reactions().iter().zip(&q).enumerate() {
            let checkable =
                rx.reversible && rx.reverse.is_none() && rx.ford.is_empty() && rx.rord.is_empty();
            if !checkable {
                let why = if !rx.reversible {
                    "irreversible"
                } else if rx.reverse.is_some() {
                    "REV"
                } else {
                    "FORD-RORD"
                };
                println!("EXCLUDED R{} {} ({why})", r + 1, rx.equation);
                excluded += 1;
                continue;
            }
            assert!(qr.forward > 0.0, "{name}: R{} q_f = {}", r + 1, qr.forward);
            let ratio = (qr.forward - qr.reverse).abs() / qr.forward;
            worst = worst.max(ratio);
            assert!(
                ratio < 1e-8,
                "{name}: R{} |q_f - q_r|/q_f = {ratio:.3e}",
                r + 1
            );
            checked += 1;
        }
        println!("{name}: checked {checked}, excluded {excluded}, worst ratio {worst:.3e}");
        assert_eq!(
            (checked, excluded),
            (11, 3),
            "{name}: the checked/excluded counts"
        );
    }
}

/// Test 4: the adiabatic flame temperature of stoichiometric CH4-air from
/// 298.15 K, 1 atm, against the oracle (P) and through `h_mass` (A); the
/// textbook value is printed NAMED, gating nothing.
#[test]
fn eq_tad_methane_air() {
    let m = fixture();
    let set = m.thermo();
    let y0 = y_ch4_air(set);
    let h0 = set.h_mass(298.15, &y0).unwrap();
    let hp = equilibrate_hp(set, h0, P_ATM, &y0, &EqOptions::default()).unwrap();
    let rt = rel(hp.t, OR_HP_T);
    println!(
        "T_ad = {} K against the oracle's {OR_HP_T:e}, rel {rt:.3e}, {} iterations",
        hp.t, hp.iterations
    );
    assert!(rt <= 1e-10, "T_ad rel {rt:.3e}");
    let ym = hp.mass_fractions(set);
    let h_back = set.h_mass(hp.t, &ym).unwrap();
    let scale: f64 = ym
        .iter()
        .zip(set.species())
        .map(|(yj, s)| yj * s.h(hp.t).unwrap().abs() / s.molar_mass)
        .sum();
    let dh = (h_back - h0).abs();
    println!(
        "|h(t, Y) - h0| = {dh:.3e} J/kg against the class A bound {:.3e} J/kg",
        1e-10 * scale
    );
    assert!(dh <= 1e-10 * scale, "enthalpy balance {dh:.3e} J/kg");
    println!(
        "NAMED, no gate: Turns (2000) quotes T_ad = 2226 K for stoichiometric CH4-air at \
         298 K, 1 atm; this 10-species set gives {} K (difference {} K)",
        hp.t,
        hp.t - 2226.0
    );
}

/// Test 5: at the hp state's T, the assigned-(T, p) problem reproduces the
/// hp state's composition and element potentials (SPEC-LIT §131.4).
#[test]
fn eq_tp_at_tad_reproduces_hp() {
    let m = fixture();
    let set = m.thermo();
    let y0 = y_ch4_air(set);
    let h0 = set.h_mass(298.15, &y0).unwrap();
    let hp = equilibrate_hp(set, h0, P_ATM, &y0, &EqOptions::default()).unwrap();
    let tp = equilibrate_tp(set, hp.t, P_ATM, &y0, &EqOptions::default()).unwrap();
    let (xh, xt) = (hp.mole_fractions(), tp.mole_fractions());
    let mut worst = 0.0f64;
    for (j, (a, b)) in xt.iter().zip(&xh).enumerate() {
        let r = rel(*a, *b);
        worst = worst.max(r);
        assert!(r <= 1e-10, "X_{j}: tp {a:e} against hp {b:e}, rel {r:.3e}");
    }
    for (i, (a, b)) in tp.pi.iter().zip(&hp.pi).enumerate() {
        let r = rel(*a, *b);
        worst = worst.max(r);
        assert!(r <= 1e-10, "pi_{i}: tp {a:e} against hp {b:e}, rel {r:.3e}");
    }
    println!(
        "eq_tp_at_tad_reproduces_hp: T = {}, worst rel {worst:.3e}",
        tp.t
    );
}

/// Test 6: with no carbon in the mixture, C is an absent element - its pi
/// and b0 are exactly 0 and every carbon species' mole fraction is exactly
/// 0 (SPEC-LIT §131.2 D1); the rest meets the H2-air oracle.
#[test]
fn eq_absent_element_is_exact_zero() {
    let m = fixture();
    let set = m.thermo();
    let y0 = y_h2_air(set);
    let eq = equilibrate_tp(set, 2000.0, P_ATM, &y0, &EqOptions::default()).unwrap();
    let x = eq.mole_fractions();
    assert_eq!(x[6], 0.0, "X_CO");
    assert_eq!(x[7], 0.0, "X_CO2");
    assert_eq!(x[8], 0.0, "X_CH4");
    assert_eq!(eq.pi[2], 0.0, "pi_C");
    assert_eq!(eq.b0[2], 0.0, "b0_C");
    // The elements are H O C N; the oracle's H2 order is H O N, and its
    // species are H2 O2 H2O OH O H N2 (indices 0-5 and 9).
    let h2_sp = [0usize, 1, 2, 3, 4, 5, 9];
    let mut worst = 0.0f64;
    for (k, &j) in h2_sp.iter().enumerate() {
        let r = rel(x[j], OR_H2_X[k]);
        worst = worst.max(r);
        assert!(
            r <= 1e-10,
            "X_{} = {} against the oracle's {:e}, rel {r:.3e}",
            j,
            x[j],
            OR_H2_X[k]
        );
    }
    for (k, &i) in [0usize, 1, 3].iter().enumerate() {
        let r = rel(eq.pi[i], OR_H2_PI[k]);
        worst = worst.max(r);
        assert!(
            r <= 1e-10,
            "pi_{i} = {} against the oracle's {:e}, rel {r:.3e}",
            eq.pi[i],
            OR_H2_PI[k]
        );
    }
    println!(
        "eq_absent_element_is_exact_zero: n_total = {} against the oracle's \
         {OR_H2_NT:e}, worst rel {worst:.3e}",
        eq.n_total
    );
}

/// Test 7: the §131.5 refusals, each an Err whose text carries the named
/// parts, and no unconverged state ever returned.
#[test]
fn eq_refuses_bad_input() {
    let m = fixture();
    let set = m.thermo();
    let y0 = y_ch4_air(set);
    let d = EqOptions::default();
    for t in [f64::NAN, 0.0, -5.0] {
        let e = equilibrate_tp(set, t, P_ATM, &y0, &d)
            .unwrap_err()
            .to_string();
        println!("T = {t}: {e}");
        assert!(e.contains("T =") && e.contains("§131.5"), "{e}");
    }
    let e = equilibrate_tp(set, 2000.0, 0.0, &y0, &d)
        .unwrap_err()
        .to_string();
    println!("p = 0: {e}");
    assert!(e.contains("p ="), "{e}");
    let e = equilibrate_hp(set, f64::NAN, P_ATM, &y0, &d)
        .unwrap_err()
        .to_string();
    println!("h0 = NaN: {e}");
    assert!(e.contains("h0 ="), "{e}");
    assert!(
        equilibrate_tp(set, 2000.0, P_ATM, &y0[..9], &d).is_err(),
        "a 9-value y0 must be refused"
    );
    let no_iters = EqOptions {
        max_iterations: 0,
        t_start: d.t_start,
    };
    assert!(equilibrate_hp(set, -1.0e6, P_ATM, &y0, &no_iters).is_err());
    let no_start = EqOptions {
        max_iterations: d.max_iterations,
        t_start: 0.0,
    };
    let e = equilibrate_hp(set, -1.0e6, P_ATM, &y0, &no_start)
        .unwrap_err()
        .to_string();
    println!("t_start = 0: {e}");
    assert!(e.contains("t_start"), "{e}");
    let three = EqOptions {
        max_iterations: 3,
        t_start: d.t_start,
    };
    let e = equilibrate_tp(set, 2000.0, P_ATM, &y0, &three)
        .unwrap_err()
        .to_string();
    println!("max_iterations = 3: {e}");
    assert!(e.contains("did not converge in 3 iterations"), "{e}");
    let e = equilibrate_tp(set, 7000.0, P_ATM, &y0, &d)
        .unwrap_err()
        .to_string();
    println!("T = 7000: {e}");
    assert!(e.contains("outside its range"), "{e}");
}

/// The env pair of SPEC-LIT §129.6, or a skip by name - as
/// `chem_kinetics.rs`'s `env_pair`: the missing variable prints SKIPPED
/// with the reason and the test returns, unless `OFGPU_REQUIRE_MECH=1`
/// makes it a failure.
fn env_pair(mech_var: &str, thermo_var: &str, test: &str) -> Option<(String, String)> {
    match (env::var(mech_var), env::var(thermo_var)) {
        (Ok(m), Ok(t)) => Some((m, t)),
        (mech, _) => {
            let var = if mech.is_err() { mech_var } else { thermo_var };
            let why = format!(
                "SKIPPED {test}: {var} is not set; GRI-Mech 3.0 is user-supplied by path \
                 (SPEC-LIT §129.6)"
            );
            if env::var("OFGPU_REQUIRE_MECH").as_deref() == Ok("1") {
                panic!("{why}");
            }
            eprintln!("{why}");
            None
        }
    }
}

/// Reads the GRI pair - as `chem_kinetics.rs`'s `read_pair`: the thermo
/// file as its own set at `P_ATM`, then the mechanism with it.
fn read_pair(mech: &str, thermo: &str) -> Mechanism {
    let t = fs::read_to_string(thermo)
        .unwrap_or_else(|e| panic!("the thermo file {thermo} does not read: {e}"));
    let set = ThermoSet::read_chemkin(&t, thermo, P_ATM, &ElementTable::standard())
        .unwrap_or_else(|e| panic!("the thermo file {thermo} does not read: {e}"));
    let m = fs::read_to_string(mech)
        .unwrap_or_else(|e| panic!("the mechanism file {mech} does not read: {e}"));
    Mechanism::read_chemkin(&m, mech, Some(&set))
        .unwrap_or_else(|e| panic!("the mechanism file {mech} does not read: {e}"))
}

/// Test 8: the same gates on GRI-Mech 3.0, user-supplied by path
/// (SPEC-LIT §129.6): element conservation as test 1, and detailed balance
/// as test 3 at the tp and hp states of stoichiometric CH4-air.
#[test]
fn eq_gri30_detailed_balance() {
    let Some((mech_path, thermo_path)) = env_pair(
        "OFGPU_GRI30_MECH",
        "OFGPU_GRI30_THERMO",
        "eq_gri30_detailed_balance",
    ) else {
        return;
    };
    let m = read_pair(&mech_path, &thermo_path);
    let set = m.thermo();
    let mut x = vec![0.0; set.species().len()];
    x[set.index_of("CH4").unwrap()] = 1.0;
    x[set.index_of("O2").unwrap()] = 2.0;
    x[set.index_of("N2").unwrap()] = 7.52;
    let y0 = y_from_x(set, &x);
    // The contract's "hp from 298.15 K": thermo30.dat's records start at
    // T_low = 300 K and are not extrapolated (SPEC-LIT §128.1), so the
    // reactants' enthalpy is formed at the set's common T_low instead,
    // 1.85 K above 298.15 K.
    let t_amb = 298.15f64.max(
        set.species()
            .iter()
            .map(|s| s.t_low)
            .fold(f64::NEG_INFINITY, f64::max),
    );
    let h0 = set.h_mass(t_amb, &y0).unwrap();
    println!("h0 formed at {t_amb} K (thermo30.dat's T_low; 298.15 K is below it)");
    let tp = equilibrate_tp(set, 2000.0, P_ATM, &y0, &EqOptions::default())
        .unwrap_or_else(|e| panic!("the GRI tp equilibrium refused: {e}"));
    let hp = equilibrate_hp(set, h0, P_ATM, &y0, &EqOptions::default())
        .unwrap_or_else(|e| panic!("the GRI hp equilibrium refused: {e}"));
    for (name, eq) in [("tp", &tp), ("hp", &hp)] {
        let (re, _) = element_balance(set, eq, name);
        println!(
            "{name}: {} species, {} elements, worst element residual {re:.3e}, {} iterations",
            set.species().len(),
            eq.elements.len(),
            eq.iterations
        );
        let q = rates_of_progress(&m, eq.t, &eq.concentrations()).unwrap();
        let (mut checked, mut worst) = (0usize, 0.0f64);
        for (r, (rx, qr)) in m.reactions().iter().zip(&q).enumerate() {
            let checkable =
                rx.reversible && rx.reverse.is_none() && rx.ford.is_empty() && rx.rord.is_empty();
            if !checkable {
                continue;
            }
            if qr.forward <= 0.0 {
                println!(
                    "EXCLUDED R{} {} (q_f = {:e})",
                    r + 1,
                    rx.equation,
                    qr.forward
                );
                continue;
            }
            let ratio = (qr.forward - qr.reverse).abs() / qr.forward;
            worst = worst.max(ratio);
            assert!(
                ratio < 1e-8,
                "{name}: R{} |q_f - q_r|/q_f = {ratio:.3e}",
                r + 1
            );
            checked += 1;
        }
        println!("{name}: checked {checked}, worst ratio {worst:.3e}");
        assert!(checked >= 300, "{name}: only {checked} reactions checked");
    }
    println!("eq_gri30_detailed_balance: hp t = {} K", hp.t);
}
