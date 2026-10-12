// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.
// Provenance: see PROVENANCE.md. No GPL-licensed source was consulted.

//! CHR-03's gates: the finite-rate kinetics of SPEC-LIT §130
//! (`src/chem/kinetics.rs`) against hand evaluations and the supervisor's
//! 40-digit oracle. The mass identity (130.7) and the element identity run
//! over 1000 random `(T, p, Y)` states of the ten-species TM-4513 fixture
//! (§130.7); GRI-Mech 3.0 is user-supplied by path (§129.6), so its test
//! skips by name when `OFGPU_GRI30_MECH` or `OFGPU_GRI30_THERMO` is unset,
//! and `OFGPU_REQUIRE_MECH=1` turns the skip into a failure.
//!
//! An integration test, so it runs in both builds: it links the normal
//! library, which is f64 in every build (SPEC-LIT §130.6).
//! No GPL-licensed source was consulted.

// The oracle's values are kept to the 17 digits it printed.
#![allow(clippy::excessive_precision)]

use std::env;
use std::fs;

use ofgpu::chem::kinetics::{
    blending_f, ln_equilibrium_constant, production_rates, production_rates_tpy, rates_of_progress,
    sri_f, third_body_concentration, troe_f, troe_f_cent,
};
use ofgpu::chem::{Blending, ElementTable, Mechanism, SpeciesThermo, ThermoSet, P_ATM, R_GAS};

/// rel(x, t) = |x - t| / |t|, the class P comparison.
fn rel(x: f64, t: f64) -> f64 {
    (x - t).abs() / t.abs()
}

/// SplitMix64, seed 20261012; `u()` is uniform on [0, 1) as (x >> 11) 2^-53.
struct SplitMix64(u64);

impl SplitMix64 {
    fn u(&mut self) -> f64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        (z >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }
}

/// Random state i of 1000 (SPEC-LIT §130.7): T = 300 + 2700 u, then
/// p = 10^(4 + 3 u), then one Y draw per species, with every 10th state's
/// Y_[(i/10) % n] zeroed, normalised to sum 1.
fn random_state(i: usize, rng: &mut SplitMix64, n: usize) -> (f64, f64, Vec<f64>) {
    let t = 300.0 + 2700.0 * rng.u();
    let p = 10f64.powf(4.0 + 3.0 * rng.u());
    let mut y: Vec<f64> = (0..n).map(|_| rng.u()).collect();
    if i % 10 == 0 {
        y[(i / 10) % n] = 0.0;
    }
    let s: f64 = y.iter().sum();
    for v in y.iter_mut() {
        *v /= s;
    }
    (t, p, y)
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

/// The records as a set at `P_ATM`: the row whose `t_lo` is 1000 is
/// `upper`, the row whose `t_lo` is 200 is `lower`; the bounds are
/// 200/1000/6000 K for all ten species. `molar_mass` starts as the table's
/// own column - the mechanism reader rebuilds it from the element weights,
/// which is what makes (130.7) exact.
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
    ThermoSet::from_species(v, P_ATM, "TM-4513").unwrap()
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

/// State A: T = 1200 K, in species order.
const CONC_A: [f64; 10] = [2.0, 1.5, 3.0, 0.05, 0.02, 0.03, 0.4, 0.6, 0.3, 7.0];
/// State B: T = 2400 K, in species order.
const CONC_B: [f64; 10] = [0.5, 0.8, 1.2, 0.3, 0.2, 0.25, 0.15, 0.35, 0.0, 3.0];

/// The supervisor's 40-digit oracle (§130.7): `q_f q_r` per reaction at
/// state A (T = 1200 K, `CONC_A`).
const ORACLE_A: [(f64, f64); 14] = [
    (240.88156182967707, 339.84861106469931),
    (16117.058174659259, 618.78102540366087),
    (236286.11312158463, 1698.4129165873587),
    (664.58333333333333, 4.1983599538197297e-9),
    (0.49666666666666667, 1.1878440496349547e-11),
    (99.756891507684701, 8.7673440018713395e-8),
    (4.2987582005990414, 1.4315881227988802e-12),
    (111.3107378132099, 2.6997186140432921e-11),
    (54.956338790332583, 1.8543604066944032e-9),
    (4101.3087742410989, 9.6891629589098826),
    (4578.3208301947767, 0.0),
    (2.0630729539379391e-7, 0.0),
    (4259.3045891228756, 797.42999907817007),
    (1643.685736669582, 307.73199898041228),
];

/// The oracle at state B (T = 2400 K, `CONC_B`).
const ORACLE_B: [(f64, f64); 14] = [
    (17579.851760541829, 11877.424261191894),
    (801981.66186472398, 402775.49972439554),
    (1880426.1587066181, 691227.15243346493),
    (3450.5208333333333, 13.036875878123753),
    (11.25, 0.085954464699160705),
    (1500.2961623593229, 15.42060346836758),
    (15.166501879382089, 0.16536081335948993),
    (28.215933932801097, 0.053540437284517366),
    (1290.3320658314765, 6.6607704777512982),
    (23502.803695946126, 16556.207333954531),
    (0.0, 0.0),
    (3.7086142550715077e-6, 0.0),
    (532098.41642320449, 389455.52073313419),
    (72976.309764261483, 53413.101492522197),
];

/// The mass identity (130.7) and the element identity over `states` random
/// `(T, p, Y)`: the worst relative residual of each. W_k is the record's
/// `molar_mass` (rebuilt from the element weights), a_ek the count of
/// element e in species k.
fn conservation(m: &Mechanism, states: usize) -> (f64, f64) {
    let sp = m.thermo().species();
    let w: Vec<f64> = sp.iter().map(|s| s.molar_mass).collect();
    let mut rng = SplitMix64(20261012);
    let (mut worst_mass, mut worst_el) = (0.0f64, 0.0f64);
    for i in 0..states {
        let (t, p, y) = random_state(i, &mut rng, m.species().len());
        let omega = production_rates_tpy(m, t, p, &y).unwrap();
        let num: f64 = omega.iter().zip(&w).map(|(o, wk)| wk * o).sum();
        let den: f64 = omega.iter().zip(&w).map(|(o, wk)| wk * o.abs()).sum();
        worst_mass = worst_mass.max(num.abs() / den);
        for (e, _) in m.elements() {
            let mut num = 0.0;
            let mut den = 0.0;
            for (k, o) in omega.iter().enumerate() {
                let a = sp[k]
                    .elements
                    .iter()
                    .find(|(s, _)| s == e)
                    .map_or(0.0, |(_, n)| *n as f64);
                num += a * o;
                den += a * o.abs();
            }
            if den != 0.0 {
                worst_el = worst_el.max(num.abs() / den);
            }
        }
    }
    println!("worst mass rel {worst_mass:.3e}, worst element rel {worst_el:.3e}");
    (worst_mass, worst_el)
}

/// (130.7) over 1000 random states of the fixture (SPEC-LIT §130.6, A).
#[test]
fn kin_mass_is_conserved() {
    let m = fixture();
    let (worst, _) = conservation(&m, 1000);
    assert!(
        worst <= 1e-13,
        "mass identity (130.7) worst rel {worst:.3e} > 1e-13"
    );
}

/// `sum_k a_ek omega_k = 0` for every declared element, over the same 1000
/// states (SPEC-LIT §130.6, A).
#[test]
fn kin_elements_conserved() {
    let m = fixture();
    let (_, worst) = conservation(&m, 1000);
    assert!(
        worst <= 1e-13,
        "element identity worst rel {worst:.3e} > 1e-13"
    );
}

/// One Arrhenius reaction against (130.2)/(130.4) by hand, and the `+M`
/// third body of (130.3) (SPEC-LIT §130.6, P).
#[test]
fn kin_single_reaction_closed_form() {
    // (a) H2 + O2 => 2 OH at T = 1500, state-A conc: the RAW numbers are
    // A = 3.0E+12 (molecularity 2, so A_SI = A * 1e-6) and E = 20000 cal/mol.
    let m = mech(&["H2 + O2 => 2 OH   3.0E+12  0.5  20000.0"]);
    let a_si = 3.0e12 * 1e-6;
    let theta = 4.184 * 20000.0 / R_GAS;
    let q = a_si * 1500f64.powf(0.5) * (-theta / 1500.0).exp() * 2.0 * 1.5;
    let omega = production_rates(&m, 1500.0, &CONC_A).unwrap();
    println!(
        "(a) q = {q:.17e}, w_H2 = {}, w_O2 = {}, w_OH = {}",
        omega[0], omega[1], omega[3]
    );
    assert!(rel(omega[0], -q) <= 1e-14 && rel(omega[1], -q) <= 1e-14);
    assert!(rel(omega[3], 2.0 * q) <= 1e-14);
    for (k, o) in omega.iter().enumerate() {
        if !matches!(k, 0 | 1 | 3) {
            assert_eq!(*o, 0.0, "species {k} must be exactly untouched");
        }
    }
    // The supervisor's 40-digit oracle for q at c_H2 = 2.0, c_O2 = 1.5.
    println!(
        "(a) oracle q 4.2496051968742382e5 vs -w_H2, rel {:.3e}",
        rel(-omega[0], 424960.51968742382)
    );
    assert!(rel(-omega[0], 424960.51968742382) <= 1e-13);
    // (b) H + OH + M => H2O + M at T = 1500, state-A conc: C = 31.9,
    // forward 0.638, reverse exactly 0 (irreversible).
    let m = mech(&[
        "H + OH + M => H2O + M   2.0E+16 -1.0  0.0",
        "  H2O/6.0/ H2/2.0/",
    ]);
    let c_m = third_body_concentration(m.reactions()[0].third_body.as_ref().unwrap(), &CONC_A);
    println!("(b) C = {c_m}");
    assert!(rel(c_m, 31.9) <= 1e-14);
    let p = rates_of_progress(&m, 1500.0, &CONC_A).unwrap();
    let kf = 2.0e16 * 1e-12 * 1500f64.powf(-1.0);
    let q_hand = kf * c_m * 0.03 * 0.05;
    println!("(b) forward = {}, hand = {q_hand:.17e}", p[0].forward);
    assert!(rel(p[0].forward, q_hand) <= 1e-14 && rel(p[0].forward, 0.638) <= 1e-14);
    assert_eq!(p[0].reverse, 0.0);
}

/// (130.9)-(130.11) at one stated point, evaluated inline by hand
/// (SPEC-LIT §130.6, P).
#[test]
fn kin_troe_matches_hand_value() {
    let (t, pr) = (1200.0f64, 3.0f64);
    let f_cent_hand = 0.4 * (-t / 100.0).exp() + 0.6 * (-t / 1500.0).exp() + (-5000.0 / t).exp();
    let f_cent = troe_f_cent(0.6, 100.0, 1500.0, Some(5000.0), t);
    println!("F_cent = {f_cent}, hand = {f_cent_hand}");
    assert!(rel(f_cent, f_cent_hand) <= 1e-14);
    assert!(rel(f_cent, 0.28510368975428361) <= 1e-13);
    let f_hand = |fc: f64| {
        let log_fc = fc.log10();
        let c = -0.4 - 0.67 * log_fc;
        let n = 0.75 - 1.27 * log_fc;
        let x = pr.log10() + c;
        10f64.powf(log_fc / (1.0 + (x / (n - 0.14 * x)).powi(2)))
    };
    let f = troe_f(f_cent, pr);
    println!("F = {f}, hand = {}", f_hand(f_cent_hand));
    assert!(rel(f, f_hand(f_cent_hand)) <= 1e-14);
    assert!(rel(f, 0.32044292697628356) <= 1e-13);
    // Without T** the last term of (130.11) is absent (SAND89-8009 p. 23).
    let fc3_hand = 0.4 * (-t / 100.0).exp() + 0.6 * (-t / 1500.0).exp();
    let fc3 = troe_f_cent(0.6, 100.0, 1500.0, None, t);
    println!("F_cent(3 param) = {fc3}, hand = {fc3_hand}");
    assert!(rel(fc3, fc3_hand) <= 1e-14);
    assert!(rel(fc3, 0.26959983615527429) <= 1e-13);
    let f3 = troe_f(fc3, pr);
    println!("F(3 param) = {f3}, hand = {}", f_hand(fc3_hand));
    assert!(rel(f3, f_hand(fc3_hand)) <= 1e-14);
    assert!(rel(f3, 0.30566943426095627) <= 1e-13);
}

/// (130.6) against the direct `exp` form where neither overflows
/// (SPEC-LIT §130.6, P): the (b) reaction of test 3, made reversible.
#[test]
fn kin_reverse_from_log_kc() {
    let m = mech(&[
        "H + OH + M <=> H2O + M   2.0E+16 -1.0  0.0",
        "  H2O/6.0/ H2/2.0/",
    ]);
    let t = 1500.0;
    let g = |name: &str| m.thermo().get(name).unwrap().g_rt(t).unwrap();
    // nu: H2O +1, H -1, OH -1, so sum nu = -1 and Kc carries
    // (P_ATM / (R_GAS T))^(sum nu) = (P_ATM / (R_GAS T))^(-1).
    let sum_nu_g = g("H2O") - g("H") - g("OH");
    let kc = (-sum_nu_g).exp() * (P_ATM / (R_GAS * t)).powf(-1.0);
    let kf = 2.0e16 * 1e-12 * t.powf(-1.0);
    let expected = c_of_m() * kf / kc * 3.0;
    let p = rates_of_progress(&m, t, &CONC_A).unwrap();
    println!("reverse = {}, expected = {expected}", p[0].reverse);
    assert!(rel(p[0].reverse, expected) <= 1e-13);
    let ln_kc = ln_equilibrium_constant(&m, 0, t).unwrap();
    println!("ln_kc = {ln_kc}, ln(expected Kc) = {}", kc.ln());
    assert!(rel(ln_kc, kc.ln()) <= 1e-13);
}

/// (130.3)'s `[M]` of the `+M` reaction with `H2O/6.0/ H2/2.0/` at state A.
fn c_of_m() -> f64 {
    let m = mech(&[
        "H + OH + M <=> H2O + M   2.0E+16 -1.0  0.0",
        "  H2O/6.0/ H2/2.0/",
    ]);
    third_body_concentration(m.reactions()[0].third_body.as_ref().unwrap(), &CONC_A)
}

/// Every `q_f` and `q_r` of the fixture at states A and B against the
/// supervisor's 40-digit oracle, 1e-12 relative; an oracle 0.0 is exactly 0
/// (SPEC-LIT §130.7, P).
#[test]
fn kin_fixture_matches_oracle() {
    let m = fixture();
    let mut worst = 0.0f64;
    for (name, t, conc, oracle) in [
        ("A", 1200.0, &CONC_A[..], &ORACLE_A[..]),
        ("B", 2400.0, &CONC_B[..], &ORACLE_B[..]),
    ] {
        let q = rates_of_progress(&m, t, conc).unwrap();
        for (i, (got, want)) in q.iter().zip(oracle).enumerate() {
            println!(
                "state {name} R{}: fwd {} (oracle {}, rel {:.2e}), rev {} (oracle {}, rel {:.2e})",
                i + 1,
                got.forward,
                want.0,
                rel(got.forward, want.0),
                got.reverse,
                want.1,
                rel(got.reverse, want.1)
            );
            if want.0 == 0.0 {
                assert_eq!(got.forward, 0.0, "state {name} R{} forward", i + 1);
            } else {
                worst = worst.max(rel(got.forward, want.0));
            }
            if want.1 == 0.0 {
                assert_eq!(got.reverse, 0.0, "state {name} R{} reverse", i + 1);
            } else {
                worst = worst.max(rel(got.reverse, want.1));
            }
        }
    }
    println!("worst rel {worst:.3e}");
    assert!(
        worst <= 1e-12,
        "fixture vs oracle worst rel {worst:.3e} > 1e-12"
    );
}

/// Lindemann's limit `k = k_inf Pr/(1+Pr)` by hand, and `Pr = 0` giving
/// exactly 0 with `log10 Pr` never evaluated (SPEC-LIT §130.7, P).
#[test]
fn kin_falloff_limits() {
    // (a) only R6 (Lindemann) at state A: k_inf = 1e7 (A = 1e13, m = 2),
    // k_0 = 5e6/1200 (LOW = 5e18, m + 1 = 3, beta = -1), [M] = sum(conc)
    // + 4 c_H2O (H2O's alpha 5 adds 4).
    let m = mech(&[
        "H + H (+M) <=> H2 (+M)            1.0E+13   0.0      0.0",
        "  LOW / 5.0E+18 -1.0 0.0 /",
        "  H2O/5.0/",
    ]);
    let k_inf = 1e7;
    let k_0 = 5.0e6 / 1200.0;
    let m_conc: f64 = CONC_A.iter().sum::<f64>() + 4.0 * 3.0;
    let pr = k_0 * m_conc / k_inf;
    let hand = k_inf * pr / (1.0 + pr) * 0.03 * 0.03;
    let q = rates_of_progress(&m, 1200.0, &CONC_A).unwrap();
    println!("(a) forward = {}, hand = {hand} (Pr = {pr})", q[0].forward);
    assert!(rel(q[0].forward, hand) <= 1e-14);
    assert!(rel(q[0].forward, 99.756891507684701) <= 1e-13);
    // (b) only R8 ((+N2)) at state A but c_N2 = 0: the collider is absent,
    // so Pr = 0 and every entry of production_rates is exactly 0.
    let m = mech(&[
        "O + H2 (+N2) <=> H2O (+N2)        1.0E+12   0.0   1000.0",
        "  LOW / 1.0E+18 -1.0 0.0 /",
        "  TROE / 0.7 200.0 1000.0 /",
    ]);
    let mut conc = CONC_A;
    conc[9] = 0.0;
    let q = rates_of_progress(&m, 1200.0, &conc).unwrap();
    println!("(b) forward = {}, reverse = {}", q[0].forward, q[0].reverse);
    assert_eq!(q[0].forward, 0.0);
    assert_eq!(q[0].reverse, 0.0);
    for (k, o) in production_rates(&m, 1200.0, &conc)
        .unwrap()
        .iter()
        .enumerate()
    {
        assert_eq!(*o, 0.0, "omega[{k}]");
    }
}

/// (130.12) at one stated point, inline by hand, and Lindemann's F = 1
/// (SPEC-LIT §130.7, P).
#[test]
fn kin_sri_matches_hand_value() {
    let (t, pr) = (1200.0f64, 3.0f64);
    let x = 1.0 / (1.0 + pr.log10().powi(2));
    let hand = 1.2 * (0.5 * (-800.0 / t).exp() + (-t / 200.0).exp()).powf(x) * t.powf(0.1);
    let got = sri_f(0.5, 800.0, 200.0, 1.2, 0.1, t, pr);
    println!("SRI F = {got}, hand = {hand}");
    assert!(rel(got, hand) <= 1e-14);
    assert!(rel(got, 0.81179790110955226) <= 1e-13);
    assert_eq!(blending_f(&Blending::Lindemann, t, pr), 1.0);
}

/// (130.16) at 250 K, where `exp(ln K_c)` overflows f64 to `inf` and the
/// direct form would fail (SPEC-LIT §130.7, P).
#[test]
fn kin_reverse_survives_low_t() {
    let m = mech(&["4 H + 2 O <=> 2 H2O   1.0E+300  0.0  0.0"]);
    let ln_kc = ln_equilibrium_constant(&m, 0, 250.0).unwrap();
    println!("ln_kc = {ln_kc}");
    assert!(rel(ln_kc, 827.75739852010672) <= 1e-12);
    assert_eq!(ln_kc.exp(), f64::INFINITY);
    let conc = [1.0; 10];
    let q = rates_of_progress(&m, 250.0, &conc).unwrap();
    println!("reverse = {}", q[0].reverse);
    assert!(rel(q[0].reverse, 3.2324325257767398e-90) <= 1e-11);
    assert!(q[0].reverse.is_finite() && q[0].reverse > 0.0);
}

/// The §130.7 refusals, each by name.
#[test]
fn kin_refuses_bad_input() {
    let m = fixture();
    for t in [f64::NAN, 0.0, -5.0] {
        let e = rates_of_progress(&m, t, &CONC_A).unwrap_err().to_string();
        println!("T = {t}: {e}");
        assert!(e.contains("T =") && e.contains("§130.7"), "T = {t}: {e}");
    }
    let e = rates_of_progress(&m, 1200.0, &CONC_A[..9])
        .unwrap_err()
        .to_string();
    println!("length: {e}");
    assert!(e.contains("9") && e.contains("10"), "{e}");
    let mut conc = CONC_A;
    conc[1] = -0.001;
    let e = rates_of_progress(&m, 1200.0, &conc)
        .unwrap_err()
        .to_string();
    println!("negative: {e}");
    assert!(e.contains("O2") && e.contains("-0.001"), "{e}");
    conc[1] = 1.5;
    conc[6] = f64::INFINITY;
    let e = rates_of_progress(&m, 1200.0, &conc)
        .unwrap_err()
        .to_string();
    println!("infinite: {e}");
    assert!(e.contains("CO") && e.contains("inf"), "{e}");
    let e = rates_of_progress(&m, 7000.0, &CONC_A)
        .unwrap_err()
        .to_string();
    println!("range: {e}");
    assert!(e.contains("outside its range"), "{e}");
    let y = [0.02, 0.2, 0.1, 0.01, 0.005, 0.001, 0.05, 0.1, 0.03, 0.484];
    assert!(production_rates_tpy(&m, 1200.0, 0.0, &y).is_err());
}

/// The env pair of SPEC-LIT §129.6, or a skip by name - as
/// `chem_mechanism_files.rs`'s `env_pair`: the missing variable prints
/// `SKIPPED` with the reason and the test returns, unless
/// `OFGPU_REQUIRE_MECH=1` makes it a failure.
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

/// Reads the GRI pair - as `chem_mechanism_files.rs`'s `read_pair`: the
/// thermo file as its own set at `P_ATM`, then the mechanism with it.
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

/// (130.7) and the element identity on GRI-Mech 3.0 over the same 1000
/// random states, when its two paths are set (SPEC-LIT §130.7, A).
#[test]
fn kin_gri30_mass_and_elements() {
    let Some((mech, thermo)) = env_pair(
        "OFGPU_GRI30_MECH",
        "OFGPU_GRI30_THERMO",
        "kin_gri30_mass_and_elements",
    ) else {
        return;
    };
    let m = read_pair(&mech, &thermo);
    let (worst_mass, worst_el) = conservation(&m, 1000);
    println!("GRI-30: {worst_mass:.3e} mass, {worst_el:.3e} element");
    assert!(
        worst_mass <= 1e-13,
        "GRI mass identity worst rel {worst_mass:.3e}"
    );
    assert!(
        worst_el <= 1e-13,
        "GRI element identity worst rel {worst_el:.3e}"
    );
}

/// `production_rates_tpy` equals `production_rates` on the concentrations
/// `mech.thermo().concentrations` returns, bitwise (SPEC-LIT §130.7).
#[test]
fn kin_tpy_matches_concentrations() {
    let m = fixture();
    let (t, p) = (1200.0, 2.0e5);
    let y = [0.02, 0.2, 0.1, 0.01, 0.005, 0.001, 0.05, 0.1, 0.03, 0.484];
    let conc = m.thermo().concentrations(p, t, &y).unwrap();
    let a = production_rates_tpy(&m, t, p, &y).unwrap();
    let b = production_rates(&m, t, &conc).unwrap();
    for (k, (x, z)) in a.iter().zip(&b).enumerate() {
        println!("omega[{k}] = {x} (tpy) vs {z} (conc)");
        assert_eq!(x.to_bits(), z.to_bits(), "omega[{k}] differs bitwise");
    }
}
