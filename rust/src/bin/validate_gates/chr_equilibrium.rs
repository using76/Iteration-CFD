// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.

//! **Gate CHR-EQ (SPEC-LIT §131.4): gas-phase equilibrium by Gibbs
//! minimisation on the TM-4513 fixture, against the 40-digit oracle, live.**
//!
//! The ten-species TM-4513 fixture of §130.7 (CHR-04's `tm_set`, through its
//! tracked key) over CHR-04's 14-reaction mechanism, ported from
//! `tests/chem_equilibrium.rs` tests 1-6: the elements' moles conserved at
//! every state with `b0` bitwise `element_moles` and the mass 1 kg (class A
//! and discrete), the CH4-air tp and hp states against the supervisor's
//! 40-digit oracle - every X_j, pi_i, n_total and T within 1e-10 relative,
//! b0 within 1e-13, h0 within 1e-12 (class A) - detailed balance through
//! `rates_of_progress` at the equilibrium states with the checkable/excluded
//! counts pinned (class A and discrete), the adiabatic flame temperature
//! against the oracle and through `h_mass` (class A), the assigned-(T, p)
//! problem reproducing the hp state at its own T (class A), and the absent
//! element C exactly zero in pi, b0 and every carbon species' mole fraction
//! while the rest meets the H2-air oracle (discrete and class A). The
//! textbook T_ad is printed beside the oracle row and gates nothing.
//!
//! Written from:
//!   ofgpu `SPEC-LIT.md` §131.4 (the gate table row), §131.1-§131.5 (the
//!     element-potential conditions, the oracle, the absent-element rule,
//!     the refusals), §130.5/§130.7 (detailed balance, the fixture)
//!   tests/chem_equilibrium.rs - tests `eq_elements_conserved`,
//!     `eq_matches_oracle`, `eq_detailed_balance_holds`,
//!     `eq_tad_methane_air`, `eq_tp_at_tad_reproduces_hp`,
//!     `eq_absent_element_is_exact_zero`; the tests stay as they are
//! No GPL-licensed source was consulted.

// The oracle's values are kept to the digits it printed.
#![allow(clippy::excessive_precision)]

use ofgpu::chem::equilibrium::element_moles;
use ofgpu::chem::kinetics::rates_of_progress;
use ofgpu::chem::{equilibrate_hp, equilibrate_tp, EqOptions, Equilibrium, ThermoSet, P_ATM};
use ofgpu::{Result, Scalar};

use crate::Checks;

use super::chr_fixture::{ch4_air, h2_air, mech, tm4513};

/// The gate's scope and registry name.
pub(crate) const GATE: &str = "SPEC-LIT 131.4 Gate CHR-EQ Gibbs equilibrium";

/// The 14-reaction fixture of SPEC-LIT §130.7. Its rate numbers are
/// placeholders, not any paper's; the R-numbers of the oracle below are
/// reaction index + 1.
const EQ_FIXTURE_RX: &[&str] = &[
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
];

/// rel(x, t) = |x - t| / |t|, the class A comparison.
fn rel(x: f64, t: f64) -> f64 {
    (x - t).abs() / t.abs()
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
fn ch4_pair(set: &ThermoSet, y0: &[f64], h0: f64) -> Result<(Equilibrium, Equilibrium)> {
    let d = EqOptions::default();
    let tp = equilibrate_tp(set, 2000.0, P_ATM, y0, &d)?;
    let hp = equilibrate_hp(set, h0, P_ATM, y0, &d)?;
    Ok((tp, hp))
}

/// Test 2's oracle comparison, one row per quantity: every X_j, pi_i and
/// n_total - and T when the oracle names one - within 1e-10 relative.
fn oracle_check(
    c: &mut Checks,
    eq: &Equilibrium,
    name: &str,
    x: &[f64; 10],
    pi: &[f64; 4],
    nt: f64,
    t: Option<f64>,
) {
    let xt = eq.mole_fractions();
    for j in 0..10 {
        c.check_a(
            &format!("Gate CHR-EQ: {name} X_{j} within 1e-10 relative of the oracle (SPEC-LIT 131.4)"),
            rel(xt[j], x[j]) as Scalar,
            1e-10,
        );
    }
    c.check_a(
        &format!("Gate CHR-EQ: {name} n_total within 1e-10 relative of the oracle (SPEC-LIT 131.4)"),
        rel(eq.n_total, nt) as Scalar,
        1e-10,
    );
    for i in 0..4 {
        c.check_a(
            &format!("Gate CHR-EQ: {name} pi_{i} within 1e-10 relative of the oracle (SPEC-LIT 131.4)"),
            rel(eq.pi[i], pi[i]) as Scalar,
            1e-10,
        );
    }
    if let Some(tt) = t {
        c.check_a(
            &format!("Gate CHR-EQ: {name} T within 1e-10 relative of the oracle (SPEC-LIT 131.4)"),
            rel(eq.t, tt) as Scalar,
            1e-10,
        );
    }
}

/// The (A) element check of test 1, one row per element plus the mass: the
/// test's own sum of a_ij n_j over the records meets b_i° to 1e-12 max b°,
/// and sum n_j W_j = 1 kg.
fn element_balance(c: &mut Checks, set: &ThermoSet, eq: &Equilibrium, name: &str) -> (f64, f64) {
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
        c.check_a(
            &format!("Gate CHR-EQ: {name} element {el} conserved to 1e-12 of the largest b0 (SPEC-LIT 131.4)"),
            r as Scalar,
            (1e-12 * bmax) as Scalar,
        );
    }
    let mass: f64 = eq.n.iter().zip(set.species()).map(|(nj, s)| nj * s.molar_mass).sum();
    let rm = (mass - 1.0).abs();
    c.check_a(
        &format!("Gate CHR-EQ: {name} sums n_j W_j to 1 kg (SPEC-LIT 131.4)"),
        rm as Scalar,
        1e-12,
    );
    (worst, rm)
}

/// **Gate CHR-EQ** (SPEC-LIT §131.4): Gibbs equilibrium on the TM-4513
/// fixture against the 40-digit oracle, live.
pub(crate) fn check_chr_equilibrium(c: &mut Checks) -> Result<()> {
    let set = tm4513(c)?;
    let m = mech(&set, EQ_FIXTURE_RX)?;
    let set = m.thermo();

    // ---- test 1: the elements' moles are conserved --------------------------
    let y_ch4 = ch4_air(&set);
    let y_h2 = h2_air(&set);
    let h_ch4 = set.h_mass(298.15, &y_ch4)?;
    let cases: [(&str, &Vec<f64>, Equilibrium); 3] = [
        (
            "ch4-air tp 2000 K",
            &y_ch4,
            equilibrate_tp(&set, 2000.0, P_ATM, &y_ch4, &EqOptions::default())?,
        ),
        (
            "ch4-air hp",
            &y_ch4,
            equilibrate_hp(&set, h_ch4, P_ATM, &y_ch4, &EqOptions::default())?,
        ),
        (
            "h2-air tp 2000 K",
            &y_h2,
            equilibrate_tp(&set, 2000.0, P_ATM, &y_h2, &EqOptions::default())?,
        ),
    ];
    for (name, y0, eq) in &cases {
        let (els, b) = element_moles(&set, y0)?;
        c.require(
            &format!("Gate CHR-EQ: {name} element order is element_moles' own (SPEC-LIT 131.4)"),
            els == eq.elements,
        );
        c.require(
            &format!("Gate CHR-EQ: {name} b0 is bitwise element_moles' own (SPEC-LIT 131.4)"),
            b == eq.b0,
        );
        element_balance(c, &set, eq, name);
    }

    // ---- test 2: the CH4-air tp and hp states against the oracle ------------
    let y0 = y_ch4.clone();
    let h0 = set.h_mass(298.15, &y0)?;
    c.check_a(
        "Gate CHR-EQ: the reactants' h0 within 1e-12 relative of the oracle's (SPEC-LIT 131.4)",
        rel(h0, OR_H0) as Scalar,
        1e-12,
    );
    let (tp, hp) = ch4_pair(&set, &y0, h0)?;
    for (i, b) in tp.b0.iter().enumerate() {
        c.check_a(
            &format!("Gate CHR-EQ: b0_{i} within 1e-13 relative of the oracle's (SPEC-LIT 131.4)"),
            rel(*b, OR_B0[i]) as Scalar,
            1e-13,
        );
    }
    oracle_check(c, &tp, "ch4-air tp", &OR_TP_X, &OR_TP_PI, OR_TP_NT, None);
    oracle_check(c, &hp, "ch4-air hp", &OR_HP_X, &OR_HP_PI, OR_HP_NT, Some(OR_HP_T));

    // ---- test 3: detailed balance at the equilibrium states -----------------
    for (name, eq) in [("ch4-air tp", &tp), ("ch4-air hp", &hp)] {
        let q = rates_of_progress(&m, eq.t, &eq.concentrations())?;
        let (mut checked, mut excluded) = (0usize, 0usize);
        let mut worst = 0.0f64;
        let mut all_positive = true;
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
                c.note(&format!("excluded R{} {} ({why})", r + 1, rx.equation));
                excluded += 1;
                continue;
            }
            all_positive &= qr.forward > 0.0;
            worst = worst.max((qr.forward - qr.reverse).abs() / qr.forward);
            checked += 1;
        }
        c.require(
            &format!("Gate CHR-EQ: {name} every checked reaction's forward rate is positive (SPEC-LIT 131.4)"),
            all_positive,
        );
        c.check_a(
            &format!("Gate CHR-EQ: {name} worst |q_f - q_r|/q_f over the {checked} checkable reactions (SPEC-LIT 131.4)"),
            worst as Scalar,
            1e-8,
        );
        c.require(
            &format!("Gate CHR-EQ: {name} checks 11 reactions and excludes 3 (SPEC-LIT 131.4)"),
            (checked, excluded) == (11, 3),
        );
    }

    // ---- test 4: the adiabatic flame temperature of CH4-air ------------------
    let hp = equilibrate_hp(&set, h0, P_ATM, &y0, &EqOptions::default())?;
    c.check_a(
        "Gate CHR-EQ: T_ad within 1e-10 relative of the oracle's (SPEC-LIT 131.4)",
        rel(hp.t, OR_HP_T) as Scalar,
        1e-10,
    );
    let ym = hp.mass_fractions(&set);
    let h_back = set.h_mass(hp.t, &ym)?;
    let scale: f64 = ym
        .iter()
        .zip(set.species())
        .map(|(yj, s)| yj * s.h(hp.t).unwrap().abs() / s.molar_mass)
        .sum();
    c.check_a(
        "Gate CHR-EQ: T_ad closes the enthalpy balance through h_mass (SPEC-LIT 131.4)",
        (h_back - h0).abs() as Scalar,
        (1e-10 * scale) as Scalar,
    );
    c.note(&format!(
        "NAMED, no gate: Turns (2000) quotes T_ad = 2226 K for stoichiometric CH4-air at 298 K, \
         1 atm; this 10-species set gives {} K (difference {} K)",
        hp.t,
        hp.t - 2226.0
    ));

    // ---- test 5: tp at T_ad reproduces hp ------------------------------------
    let tp = equilibrate_tp(&set, hp.t, P_ATM, &y0, &EqOptions::default())?;
    let (xh, xt) = (hp.mole_fractions(), tp.mole_fractions());
    for (j, (a, b)) in xt.iter().zip(&xh).enumerate() {
        c.check_a(
            &format!("Gate CHR-EQ: tp X_{j} reproduces hp's at T_ad within 1e-10 relative (SPEC-LIT 131.4)"),
            rel(*a, *b) as Scalar,
            1e-10,
        );
    }
    for (i, (a, b)) in tp.pi.iter().zip(&hp.pi).enumerate() {
        c.check_a(
            &format!("Gate CHR-EQ: tp pi_{i} reproduces hp's at T_ad within 1e-10 relative (SPEC-LIT 131.4)"),
            rel(*a, *b) as Scalar,
            1e-10,
        );
    }

    // ---- test 6: the absent element C is exactly zero ------------------------
    let eq = equilibrate_tp(&set, 2000.0, P_ATM, &y_h2, &EqOptions::default())?;
    let x = eq.mole_fractions();
    c.require("Gate CHR-EQ: X_CO is exactly 0 with no carbon in the mixture (SPEC-LIT 131.4)", x[6] == 0.0);
    c.require("Gate CHR-EQ: X_CO2 is exactly 0 with no carbon in the mixture (SPEC-LIT 131.4)", x[7] == 0.0);
    c.require("Gate CHR-EQ: X_CH4 is exactly 0 with no carbon in the mixture (SPEC-LIT 131.4)", x[8] == 0.0);
    c.require("Gate CHR-EQ: pi_C is exactly 0 with no carbon in the mixture (SPEC-LIT 131.4)", eq.pi[2] == 0.0);
    c.require("Gate CHR-EQ: b0_C is exactly 0 with no carbon in the mixture (SPEC-LIT 131.4)", eq.b0[2] == 0.0);
    // The elements are H O C N; the oracle's H2 order is H O N, and its
    // species are H2 O2 H2O OH O H N2 (indices 0-5 and 9).
    let h2_sp = [0usize, 1, 2, 3, 4, 5, 9];
    for (k, &j) in h2_sp.iter().enumerate() {
        c.check_a(
            &format!("Gate CHR-EQ: H2-air X_{j} within 1e-10 relative of the oracle (SPEC-LIT 131.4)"),
            rel(x[j], OR_H2_X[k]) as Scalar,
            1e-10,
        );
    }
    for (k, &i) in [0usize, 1, 3].iter().enumerate() {
        c.check_a(
            &format!("Gate CHR-EQ: H2-air pi_{i} within 1e-10 relative of the oracle (SPEC-LIT 131.4)"),
            rel(eq.pi[i], OR_H2_PI[k]) as Scalar,
            1e-10,
        );
    }
    c.note(&format!(
        "Gate CHR-EQ H2-air: n_total = {} against the oracle's {OR_H2_NT:e}",
        eq.n_total
    ));
    Ok(())
}


