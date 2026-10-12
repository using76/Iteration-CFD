// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.
// Provenance: see PROVENANCE.md. No GPL-licensed source was consulted.

//! CHR-06's gates: the constant-pressure adiabatic 0-D reactor (SPEC-LIT
//! §132.4, `src/chem/reactor.rs`) against the supervisor's independent
//! float64 oracle - scipy's Radau on the same (132.7)-(132.9), made outside
//! the tree - on the ten-species TM-4513 fixture of §130.7: the right-hand
//! side at a fixed state, the landing on §131's `(h, p)` equilibrium, the
//! ignition delay's convergence and its oracle value, the zero-duration
//! identity, the §132.8 refusals, and the observer of
//! [`ofgpu::chem::rosenbrock::integrate_observed`]. GRI-Mech 3.0 is
//! user-supplied by path (§129.6), so its tests skip by name when
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

use ofgpu::chem::reactor::{ConstPressureReactor, ReactorOptions};
use ofgpu::chem::rosenbrock::{integrate, integrate_observed, RosOptions, RosStats};
use ofgpu::chem::{ElementTable, EqOptions, Mechanism, SpeciesThermo, ThermoSet, P_ATM, P_BAR};
use ofgpu::Error;

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

/// The 15-reaction reactor fixture of CHR-06. Every reaction is reversible,
/// the reverse from K_c (no REV, no FORD), so the ODE's rest point is the
/// Gibbs equilibrium of §131. Its rate numbers are placeholders, not any
/// paper's.
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
        "OH + OH <=> O + H2O               3.0E+04   2.4  -2000.0",
        "  DUP",
        "OH + OH <=> O + H2O               1.0E+12   0.0   1000.0",
        "  DUP",
        "H2 + O2 <=> OH + OH               1.7E+13   0.0  47780.0",
        "CH4 + O2 <=> CO + H2O + H2        1.0E+12   0.0  30000.0",
        "CH4 + O <=> CO + 2 H2             1.0E+13   0.0   8000.0",
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

/// Stoichiometric H2-air by moles: 2 H2 + 1 O2 + 3.76 N2 (phi = 1), in
/// the species order H2 O2 H2O OH O H CO CO2 CH4 N2.
fn h2_air(set: &ThermoSet) -> Vec<f64> {
    let mut x = vec![0.0; set.species().len()];
    x[0] = 2.0;
    x[1] = 1.0;
    x[9] = 3.76;
    y_from_x(set, &x)
}

/// Stoichiometric CH4-air by moles: 1 CH4 + 2 O2 + 7.52 N2 (phi = 1).
fn ch4_air(set: &ThermoSet) -> Vec<f64> {
    let mut x = vec![0.0; set.species().len()];
    x[1] = 2.0;
    x[8] = 1.0;
    x[9] = 7.52;
    y_from_x(set, &x)
}

/// The oracle's right-hand side at test 1's state (species order, then T).
const ORACLE_RHS: [f64; 11] = [
    -1432.4144287413787,
    -25.59632025826463,
    11809.963773312387,
    -10107.457647574522,
    -1187.811610014403,
    736.0923131413928,
    -44.91922950353602,
    356.29518998997605,
    -104.15204035164905,
    0.0,
    25828856.45591878,
];

/// The oracle's end state after 1 s at the defaults: (T, [Y_k]) for H2-air.
const ORACLE_END_H2: (f64, [f64; 10]) = (
    2693.827535802454,
    [
        3.0006981624e-03,
        1.7464580520e-02,
        2.1606791437e-01,
        1.5228277595e-02,
        2.6812734917e-03,
        4.4012740327e-04,
        0.0,
        0.0,
        0.0,
        7.4511712845e-01,
    ],
);

/// The oracle's end state after 1 s at the defaults: (T, [Y_k]) for CH4-air.
const ORACLE_END_CH4: (f64, [f64; 10]) = (
    2711.7999623963583,
    [
        1.2156675401e-03,
        2.5724001689e-02,
        1.0424789862e-01,
        1.1890015591e-02,
        3.3205775636e-03,
        2.8356439014e-04,
        3.9808346692e-02,
        8.8844925441e-02,
        1.0309182165e-15,
        7.2466500247e-01,
    ],
);

/// The oracle's ignition delays at rtol 1e-9 over 0.01 s: (H2-air, CH4-air).
const ORACLE_DELAY: (f64, f64) = (2.6552096919429e-3, 2.6042862928263e-3);

/// Every run's counters, printed as the contract asks.
fn print_stats(tag: &str, s: &RosStats) {
    eprintln!(
        "{tag}: accepted {} rejected {} nonfinite {} f_evals {} jac {} lu {} h_next {:e}",
        s.accepted, s.rejected, s.nonfinite, s.f_evals, s.jac_evals, s.lu_count, s.h_next
    );
}

/// The env pair of SPEC-LIT §129.6, or a skip by name - as
/// `chem_equilibrium.rs`'s `env_pair`: the missing variable prints SKIPPED
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

/// Reads the GRI pair - as `chem_equilibrium.rs`'s `read_pair`: the thermo
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

/// Test 1: (132.7)-(132.9) at one fixed state against the oracle.
#[test]
fn reactor_rhs_matches_oracle() {
    let m = fixture();
    let r = ConstPressureReactor::new(&m, P_ATM).unwrap();
    let set = m.thermo();
    let mut x = vec![0.0; 10];
    x[0] = 2.0; // H2
    x[1] = 1.5; // O2
    x[2] = 3.0; // H2O
    x[3] = 0.05; // OH
    x[4] = 0.02; // O
    x[5] = 0.03; // H
    x[6] = 0.4; // CO
    x[7] = 0.6; // CO2
    x[8] = 0.3; // CH4
    x[9] = 7.0; // N2
    let mut y = y_from_x(set, &x);
    y.push(1500.0);
    let mut dydt = vec![0.0; 11];
    r.rhs(&y, &mut dydt).unwrap();
    let mut worst = 0.0f64;
    for (l, (&got, &want)) in dydt.iter().zip(&ORACLE_RHS).enumerate() {
        eprintln!("dydt[{l}] = {:e}", got);
        if want == 0.0 {
            assert_eq!(got, 0.0, "component {l} must be exactly 0");
        } else {
            let e = rel(got, want);
            worst = worst.max(e);
            assert!(e <= 1e-9, "component {l}: rel {e:e} > 1e-9");
        }
    }
    eprintln!("rhs worst relative error {worst:e}");
}

/// Test 2's gates, shared with test 8: the run's end state against
/// §131's `(h, p)` equilibrium (0.5 K, 1e-6 in Y), against the oracle end
/// state when one is given, mass conservation at the clips, enthalpy
/// conservation, and an ignition delay. Every measure is printed.
fn eq_gates(
    tag: &str,
    set: &ThermoSet,
    rep: &ofgpu::chem::reactor::ReactorReport,
    y0: &[f64],
    h0: f64,
    opts: &ReactorOptions,
    oracle: Option<(f64, &[f64])>,
) {
    let eq = ofgpu::chem::equilibrate_hp(set, h0, P_ATM, y0, &EqOptions::default()).unwrap();
    let eq_y = eq.mass_fractions(set);
    let d_eq = rep
        .y
        .iter()
        .zip(&eq_y)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f64, f64::max);
    eprintln!(
        "{tag}: T_end = {:.9} K, T_eq = {:.9} K, dT = {dt:e}, max|Y - Y_eq| = {d_eq:e}",
        rep.temperature,
        eq.t,
        dt = (rep.temperature - eq.t).abs()
    );
    for (l, y) in rep.y.iter().enumerate() {
        eprintln!(
            "{tag}: Y[{}] = {y:e} (eq {eq:e})",
            set.species()[l].name,
            y = y,
            eq = eq_y[l]
        );
    }
    print_stats(tag, &rep.stats);
    assert!((rep.temperature - eq.t).abs() <= 0.5, "{tag}: dT");
    assert!(d_eq <= 1e-6, "{tag}: dY_eq");
    if let Some((t_or, y_or)) = oracle {
        let d_or = rep
            .y
            .iter()
            .zip(y_or)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f64, f64::max);
        eprintln!(
            "{tag}: oracle T = {t_or}, dT = {:e}, max|Y - Y_oracle| = {d_or:e}",
            (rep.temperature - t_or).abs()
        );
        assert!((rep.temperature - t_or).abs() <= 0.5, "{tag}: dT oracle");
        assert!(d_or <= 1e-6, "{tag}: dY oracle");
    }
    let d_sum = (rep.y.iter().sum::<f64>() - y0.iter().sum::<f64>()).abs();
    let h_end = set.h_mass(rep.temperature, &rep.y).unwrap();
    let d_h = (h_end - h0).abs();
    eprintln!(
        "{tag}: |sum Y - sum Y0| = {d_sum:e} (clipped {}), |h_end - h0|/|h0| = {:e}, \
         delay = {:?}",
        rep.clipped,
        d_h / h0.abs(),
        rep.ignition_delay
    );
    assert!(d_sum <= 1e-12 + rep.clipped as f64 * opts.atol_y, "{tag}: sum Y");
    assert!(d_h <= 1e-6 * h0.abs(), "{tag}: enthalpy");
    assert!(rep.ignition_delay.is_some(), "{tag}: no ignition");
}

/// Test 2: both mixtures land on the `(h, p)` equilibrium and on the
/// oracle's end state, 1 s at the default options.
#[test]
fn reactor_lands_on_equilibrium() {
    let m = fixture();
    let r = ConstPressureReactor::new(&m, P_ATM).unwrap();
    let set = m.thermo();
    let opts = ReactorOptions::default();
    let y_h2 = h2_air(set);
    let h_h2 = set.h_mass(1000.0, &y_h2).unwrap();
    let rep = r.advance(&y_h2, 1000.0, 1.0, &opts).unwrap();
    let (t_or, y_or) = ORACLE_END_H2;
    eq_gates("H2-air", set, &rep, &y_h2, h_h2, &opts, Some((t_or, &y_or)));
    let y_ch4 = ch4_air(set);
    let h_ch4 = set.h_mass(1400.0, &y_ch4).unwrap();
    let rep = r.advance(&y_ch4, 1400.0, 1.0, &opts).unwrap();
    let (t_or, y_or) = ORACLE_END_CH4;
    eq_gates("CH4-air", set, &rep, &y_ch4, h_ch4, &opts, Some((t_or, &y_or)));
}

/// Test 3: the ignition delay converges between rtol 1e-6 and 1e-9.
#[test]
fn reactor_delay_converges() {
    let m = fixture();
    let r = ConstPressureReactor::new(&m, P_ATM).unwrap();
    let set = m.thermo();
    let o6 = ReactorOptions {
        rtol: 1e-6,
        ..ReactorOptions::default()
    };
    let o9 = ReactorOptions {
        rtol: 1e-9,
        ..ReactorOptions::default()
    };
    let y_h2 = h2_air(set);
    let d6 = r.advance(&y_h2, 1000.0, 0.01, &o6).unwrap();
    let d9 = r.advance(&y_h2, 1000.0, 0.01, &o9).unwrap();
    let (a6, a9) = (d6.ignition_delay.unwrap(), d9.ignition_delay.unwrap());
    eprintln!("H2-air: delay(rtol 1e-6) = {a6:e}, delay(rtol 1e-9) = {a9:e}");
    print_stats("H2-air rtol 1e-6", &d6.stats);
    print_stats("H2-air rtol 1e-9", &d9.stats);
    assert!(rel(a6, a9) < 5e-3, "H2-air: rel {} >= 5e-3", rel(a6, a9));
    let y_ch4 = ch4_air(set);
    let d6 = r.advance(&y_ch4, 1400.0, 0.01, &o6).unwrap();
    let d9 = r.advance(&y_ch4, 1400.0, 0.01, &o9).unwrap();
    let (a6, a9) = (d6.ignition_delay.unwrap(), d9.ignition_delay.unwrap());
    eprintln!("CH4-air: delay(rtol 1e-6) = {a6:e}, delay(rtol 1e-9) = {a9:e}");
    print_stats("CH4-air rtol 1e-6", &d6.stats);
    print_stats("CH4-air rtol 1e-9", &d9.stats);
    assert!(rel(a6, a9) < 5e-3, "CH4-air: rel {} >= 5e-3", rel(a6, a9));
}

/// Test 4: the ignition delay against the oracle, rtol 1e-9 over 0.01 s.
#[test]
fn reactor_delay_matches_oracle() {
    let m = fixture();
    let r = ConstPressureReactor::new(&m, P_ATM).unwrap();
    let set = m.thermo();
    let o9 = ReactorOptions {
        rtol: 1e-9,
        ..ReactorOptions::default()
    };
    let y_h2 = h2_air(set);
    let rep = r.advance(&y_h2, 1000.0, 0.01, &o9).unwrap();
    let d = rep.ignition_delay.unwrap();
    eprintln!("H2-air: delay = {d:e}, oracle = {:e}", ORACLE_DELAY.0);
    print_stats("H2-air rtol 1e-9", &rep.stats);
    assert!(rel(d, ORACLE_DELAY.0) < 1e-3, "H2-air: rel {}", rel(d, ORACLE_DELAY.0));
    let y_ch4 = ch4_air(set);
    let rep = r.advance(&y_ch4, 1400.0, 0.01, &o9).unwrap();
    let d = rep.ignition_delay.unwrap();
    eprintln!("CH4-air: delay = {d:e}, oracle = {:e}", ORACLE_DELAY.1);
    print_stats("CH4-air rtol 1e-9", &rep.stats);
    assert!(rel(d, ORACLE_DELAY.1) < 1e-3, "CH4-air: rel {}", rel(d, ORACLE_DELAY.1));
}

/// Test 5: duration 0.0 returns the initial state untouched.
#[test]
fn reactor_zero_duration_is_identity() {
    let m = fixture();
    let r = ConstPressureReactor::new(&m, P_ATM).unwrap();
    let y0 = h2_air(m.thermo());
    let rep = r.advance(&y0, 1000.0, 0.0, &ReactorOptions::default()).unwrap();
    for (a, b) in rep.y.iter().zip(&y0) {
        assert_eq!(a.to_bits(), b.to_bits(), "y is not bitwise y0");
    }
    assert_eq!(rep.temperature, 1000.0);
    assert!(rep.ignition_delay.is_none());
    assert_eq!(rep.clipped, 0);
    assert_eq!(rep.stats.accepted, 0);
    eprintln!(
        "zero duration: y bitwise y0, T = {}, delay = {:?}, clipped {}, accepted {}",
        rep.temperature, rep.ignition_delay, rep.clipped, rep.stats.accepted
    );
}

/// Test 6: the refusals of R7, R8 and R9, each naming its value and ending
/// "(SPEC-LIT §132.8)"; and T0 = 100 K refused by the thermo record itself.
#[test]
fn reactor_refuses_bad_input() {
    let m = fixture();
    let y0 = h2_air(m.thermo());
    // R7: the pressure.
    let e = ConstPressureReactor::new(&m, 0.0).unwrap_err().to_string();
    eprintln!("p = 0: {e}");
    assert!(e.contains("reactor pressure p = 0 Pa"));
    assert!(e.ends_with("(SPEC-LIT §132.8)"));
    let e = ConstPressureReactor::new(&m, f64::NAN).unwrap_err().to_string();
    eprintln!("p = NaN: {e}");
    assert!(e.contains("reactor pressure p = NaN Pa"));
    // R8: the state's length.
    let r = ConstPressureReactor::new(&m, P_ATM).unwrap();
    let mut dy = vec![0.0; 11];
    let e = r.rhs(&[0.1, 0.2, 1000.0], &mut dy).unwrap_err().to_string();
    eprintln!("rhs 3 components: {e}");
    assert!(e.contains("the reactor state has 3 components, the mechanism needs 11"));
    // R9: the mass fractions' length.
    let e = r
        .advance(&[0.5, 0.5, 0.5], 1000.0, 1.0, &ReactorOptions::default())
        .unwrap_err()
        .to_string();
    eprintln!("y0 3 fractions: {e}");
    assert!(e.contains("the reactor was given 3 mass fractions, the mechanism has 10 species"));
    // R9: a negative mass fraction.
    let mut bad = y0.clone();
    bad[0] = -0.1;
    let e = r
        .advance(&bad, 1000.0, 1.0, &ReactorOptions::default())
        .unwrap_err()
        .to_string();
    eprintln!("H2 = -0.1: {e}");
    assert!(e.contains("species H2: the initial mass fraction -0.1"));
    // R9: the temperature, the duration, ignition_rise.
    let e = r
        .advance(&y0, 0.0, 1.0, &ReactorOptions::default())
        .unwrap_err()
        .to_string();
    eprintln!("T0 = 0: {e}");
    assert!(e.contains("initial temperature T0 = 0 K"));
    let e = r
        .advance(&y0, 1000.0, -1.0, &ReactorOptions::default())
        .unwrap_err()
        .to_string();
    eprintln!("duration = -1: {e}");
    assert!(e.contains("duration = -1 s"));
    let e = r
        .advance(
            &y0,
            1000.0,
            1.0,
            &ReactorOptions {
                ignition_rise: 0.0,
                ..ReactorOptions::default()
            },
        )
        .unwrap_err()
        .to_string();
    eprintln!("ignition_rise = 0: {e}");
    assert!(e.contains("ignition_rise = 0 K"));
    // A T0 below the thermo range is refused by the record itself
    // (§132.7 R5's pass-through).
    let e = r
        .advance(&y0, 100.0, 1.0, &ReactorOptions::default())
        .unwrap_err()
        .to_string();
    eprintln!("T0 = 100: {e}");
    assert!(e.contains("outside its range"));
}

/// The decay problem of test 7: y' = -y on [0, 1] from y(0) = 1.
fn decay(y: &[f64], dy: &mut [f64]) -> ofgpu::Result<()> {
    dy[0] = -y[0];
    Ok(())
}

/// Test 7: the observer of `integrate_observed` sees every accepted step,
/// `integrate` stays bitwise identical, and an observer's Err ends the call.
#[test]
fn ros_observer_sees_every_accepted_step() {
    let opts = RosOptions::new(1, 1e-8, 1e-12);
    let mut ts: Vec<f64> = Vec::new();
    let mut f = |y: &[f64], dy: &mut [f64]| decay(y, dy);
    let mut y = vec![1.0];
    let stats = integrate_observed(&mut f, &mut y, 0.0, 1.0, &opts, &mut |t, _| {
        ts.push(t);
        Ok(())
    })
    .unwrap();
    eprintln!("observer: {} calls, {} accepted", ts.len(), stats.accepted);
    print_stats("observed run", &stats);
    assert_eq!(ts.len(), stats.accepted);
    assert!(ts.windows(2).all(|w| w[0] < w[1]), "t not increasing");
    assert_eq!(*ts.last().unwrap(), 1.0, "the last t is not 1.0 exactly");
    // `integrate` on the same problem: bitwise the same y, equal counters.
    let mut f2 = |y: &[f64], dy: &mut [f64]| decay(y, dy);
    let mut y2 = vec![1.0];
    let stats2 = integrate(&mut f2, &mut y2, 0.0, 1.0, &opts).unwrap();
    assert_eq!(y[0].to_bits(), y2[0].to_bits(), "integrate's y differs");
    assert_eq!(stats, stats2, "integrate's counters differ");
    // An observer that refuses on its 3rd call ends the call with that Err.
    let mut calls = 0usize;
    let mut f3 = |y: &[f64], dy: &mut [f64]| decay(y, dy);
    let mut y3 = vec![1.0];
    let err = integrate_observed(&mut f3, &mut y3, 0.0, 1.0, &opts, &mut |_, _| {
        calls += 1;
        if calls == 3 {
            Err(Error::Config("stop here".into()))
        } else {
            Ok(())
        }
    })
    .unwrap_err();
    eprintln!("observer refused at call {calls}: {err}");
    assert_eq!(calls, 3);
    assert!(err.to_string().contains("stop here"));
}

/// The two GRI mixtures by name: stoichiometric H2-air from 1000 K and
/// stoichiometric CH4-air from 1400 K.
fn gri_mixtures(set: &ThermoSet) -> Vec<(&'static str, Vec<f64>, f64)> {
    let idx = |n: &str| {
        set.index_of(n)
            .unwrap_or_else(|| panic!("the mechanism has no species {n}"))
    };
    let mut h2 = vec![0.0; set.species().len()];
    h2[idx("H2")] = 2.0;
    h2[idx("O2")] = 1.0;
    h2[idx("N2")] = 3.76;
    let mut ch4 = vec![0.0; set.species().len()];
    ch4[idx("CH4")] = 1.0;
    ch4[idx("O2")] = 2.0;
    ch4[idx("N2")] = 7.52;
    vec![("H2-air", h2, 1000.0), ("CH4-air", ch4, 1400.0)]
}

/// Test 8: GRI-Mech 3.0 lands on the `(h, p)` equilibrium, 10 s at the
/// default options (SPEC-LIT §129.6: user-supplied by path).
#[test]
fn reactor_gri30_lands_on_equilibrium() {
    let Some((mech_path, thermo_path)) = env_pair(
        "OFGPU_GRI30_MECH",
        "OFGPU_GRI30_THERMO",
        "reactor_gri30_lands_on_equilibrium",
    ) else {
        return;
    };
    let m = read_pair(&mech_path, &thermo_path);
    let r = ConstPressureReactor::new(&m, P_ATM).unwrap();
    let set = m.thermo();
    let opts = ReactorOptions::default();
    for (name, x, t0) in gri_mixtures(set) {
        let y0 = y_from_x(set, &x);
        let h0 = set.h_mass(t0, &y0).unwrap();
        let rep = r.advance(&y0, t0, 10.0, &opts).unwrap();
        eq_gates(name, set, &rep, &y0, h0, &opts, None);
    }
}

/// Test 9: GRI-Mech 3.0's ignition delay converges between rtol 1e-6 and
/// 1e-9 over 0.01 s.
#[test]
fn reactor_gri30_delay_converges() {
    let Some((mech_path, thermo_path)) = env_pair(
        "OFGPU_GRI30_MECH",
        "OFGPU_GRI30_THERMO",
        "reactor_gri30_delay_converges",
    ) else {
        return;
    };
    let m = read_pair(&mech_path, &thermo_path);
    let r = ConstPressureReactor::new(&m, P_ATM).unwrap();
    let set = m.thermo();
    let o6 = ReactorOptions {
        rtol: 1e-6,
        ..ReactorOptions::default()
    };
    let o9 = ReactorOptions {
        rtol: 1e-9,
        ..ReactorOptions::default()
    };
    for (name, x, t0) in gri_mixtures(set) {
        let y0 = y_from_x(set, &x);
        let d6 = r.advance(&y0, t0, 0.01, &o6).unwrap();
        let d9 = r.advance(&y0, t0, 0.01, &o9).unwrap();
        let (a6, a9) = (d6.ignition_delay.unwrap(), d9.ignition_delay.unwrap());
        eprintln!("GRI {name}: delay(rtol 1e-6) = {a6:e}, delay(rtol 1e-9) = {a9:e}");
        print_stats(&format!("GRI {name} rtol 1e-6"), &d6.stats);
        print_stats(&format!("GRI {name} rtol 1e-9"), &d9.stats);
        assert!(rel(a6, a9) < 5e-3, "{name}: rel {} >= 5e-3", rel(a6, a9));
    }
}
