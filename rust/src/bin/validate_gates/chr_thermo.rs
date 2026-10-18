// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.

//! **Gate CHR-THERMO (SPEC-LIT §128.7): the TM-4513 records' NASA-7
//! thermochemistry, held against TP-2002's Table B1, live.**
//!
//! The ten TM-4513 records of `reference/mcbride-gordon-reno1993/
//! table_II.csv` are rebuilt as CHEMKIN cards - the coefficient fields
//! carried as the CSV prints them - and read through
//! `ThermoSet::read_chemkin` at `P_BAR`, so the molar masses come from the
//! element weights, exactly as `src/chem/tests_thermo.rs::tm4513_set` builds
//! its set. Against that set, ported from its tests T2-T6: cp(298.15) of
//! every species within 0.1 % of `nasa-glenn2002/table_B1.csv`'s column
//! (class P), H(298.15) within 10 J/mol of the tabulated heats of formation
//! - OH held to its own record's printed heat, the two tables' adopted
//! values differing, with the gap pinned (class P) - the T_mid fit
//! constraint `cp/R`, `H/RT`, `S/R` equal on both intervals to 1e-6 relative
//! and the interval selection (class A / discrete), the out-of-range
//! refusals naming their values (discrete), and the molar masses within
//! 5e-6 g/mol of the printed column with the Ar/he/XE element weights
//! (discrete). Every key is read through the manifest; its digest line is
//! printed once above the rows it feeds.
//!
//! Written from:
//!   ofgpu `SPEC-LIT.md` §128.7 (the gate table row), §128.1-§128.6 (NASA-7,
//!     the ranges, the TM-4513 fixture and its species codes)
//!   src/chem/tests_thermo.rs - tests T2 `thermo_cp_matches_nasa_table_b1`,
//!     T3 `thermo_h_is_heat_of_formation`, T4 `thermo_ranges_meet_at_tmid`,
//!     T5 `thermo_out_of_range_is_refused`, T6
//!     `thermo_molar_mass_from_elements` (its first loop and the Ar/he/XE
//!     weight rows only); the tests stay as they are
//!   McBride, Gordon & Reno, NASA TM-4513 (1993), Table II; Gordon &
//!     McBride, RP-1311 (1994); NASA TP-2002-211556, Table B1 - both as the
//!     tracked transcriptions their digests pin
//! No GPL-licensed source was consulted.

use ofgpu::chem::thermo::{nasa7_cp_r, nasa7_h_rt, nasa7_s_r};
use ofgpu::chem::{ElementTable, Interval, SpeciesThermo, ThermoSet, P_BAR, R_GAS};
use ofgpu::{Error, Result, Scalar};

use crate::{key, Checks};

/// The gate's scope and registry name.
pub(crate) const GATE: &str = "SPEC-LIT 128.7 Gate CHR-THERMO NASA-7 thermochemistry";

/// One `SPECIES` entry: the TM-4513 species code, its name, and its element
/// counts, in this order - `src/chem/tests_thermo.rs`'s `SPECIES`.
type SpeciesEntry = (u32, &'static str, &'static [(&'static str, u32)]);

/// TM-4513 species code -> name and element counts, in this order.
const SPECIES: [SpeciesEntry; 10] = [
    (1, "N2", &[("N", 2)]),
    (2, "O2", &[("O", 2)]),
    (5, "CH4", &[("C", 1), ("H", 4)]),
    (6, "CO2", &[("C", 1), ("O", 2)]),
    (7, "H2O", &[("H", 2), ("O", 1)]),
    (8, "CO", &[("C", 1), ("O", 1)]),
    (9, "H2", &[("H", 2)]),
    (10, "OH", &[("O", 1), ("H", 1)]),
    (11, "O", &[("O", 1)]),
    (12, "H", &[("H", 1)]),
];

/// One row of Table II. The CSV's coefficient TEXT is kept, because a
/// record is built from the text as printed, never from a re-formatted
/// number; the parsed coefficients of the source test's `IiRow` are not
/// ported because no row of this gate reads them (T1 keeps them).
struct IiRow {
    code: u32,
    t_lo: f64,
    t_hi: f64,
    mol_weight: f64,
    text: [String; 7],
    h298_r_text: String,
    h298_r: f64,
}

fn ii_rows(text: &str) -> Vec<IiRow> {
    let mut rows = Vec::new();
    let mut header = false;
    for line in text.lines() {
        if line.starts_with('#') {
            continue;
        }
        if !header {
            header = true;
            continue;
        }
        let f: Vec<&str> = line.split(',').collect();
        rows.push(IiRow {
            code: f[0].trim().parse().unwrap(),
            t_lo: f[1].trim().parse().unwrap(),
            t_hi: f[2].trim().parse().unwrap(),
            mol_weight: f[3].trim().parse().unwrap(),
            text: std::array::from_fn(|k| f[4 + k].trim().to_string()),
            h298_r_text: f[11].trim().to_string(),
            h298_r: f[11].trim().parse().unwrap(),
        });
    }
    rows
}

/// One row of Table B1: `species,mol_weight,cp_298,dfh_298`, `dfh_298` in
/// kJ/mol.
struct B1Row {
    code: u32,
    mol_weight: f64,
    cp_298: f64,
    dfh_298: f64,
}

fn b1_rows(text: &str) -> Vec<B1Row> {
    let mut rows = Vec::new();
    let mut header = false;
    for line in text.lines() {
        if line.starts_with('#') {
            continue;
        }
        if !header {
            header = true;
            continue;
        }
        let f: Vec<&str> = line.split(',').collect();
        rows.push(B1Row {
            code: f[0].trim().parse().unwrap(),
            mol_weight: f[1].trim().parse().unwrap(),
            cp_298: f[2].trim().parse().unwrap(),
            dfh_298: f[3].trim().parse().unwrap(),
        });
    }
    rows
}

fn sp(code: u32) -> (u32, &'static str, &'static [(&'static str, u32)]) {
    *SPECIES.iter().find(|s| s.0 == code).unwrap()
}

/// The species' upper or lower CSV row: `t_lo` 1000.000 is the upper row,
/// 200.000 the lower, in the file's own order.
fn row_of(rows: &[IiRow], code: u32, t_lo: f64) -> &IiRow {
    rows.iter().find(|r| r.code == code && r.t_lo == t_lo).unwrap()
}

/// The record's species, refused by name rather than panicked over: a gate
/// stops with `?`, it does not abort the run.
fn sp_of<'a>(set: &'a ThermoSet, name: &str) -> Result<&'a SpeciesThermo> {
    set.get(name)
        .ok_or_else(|| Error::Config(format!("Gate CHR-THERMO: the TM-4513 set has no {name}")))
}

/// The four 80-character lines of one species' record, from its two CSV rows
/// (upper first, as the file has them). The coefficient fields carry the
/// CSV's text as printed, right-justified in 15 columns; the element slots
/// run from column 25, each symbol left-justified in 2 columns and its count
/// right-justified in 3 - `src/chem/tests_thermo.rs`'s `chemkin_record`.
fn chemkin_record(code: u32, rows: &[IiRow]) -> [String; 4] {
    let (.., name, els) = sp(code);
    let upper = row_of(rows, code, 1000.0);
    let lower = row_of(rows, code, 200.0);
    let mut card1 = format!("{name:<18}TM4513");
    for (sym, n) in els {
        card1.push_str(&format!("{sym:<2}{n:>3}"));
    }
    while card1.len() < 44 {
        card1.push(' ');
    }
    card1.push('G');
    card1.push_str(&format!(
        "{:>10.3}{:>10.3}{:>8.2}      1",
        lower.t_lo, upper.t_hi, upper.t_lo
    ));
    let e15 = |s: &str| format!("{s:>15}");
    let card2 = format!(
        "{}{}{}{}{}    2",
        e15(&upper.text[0]),
        e15(&upper.text[1]),
        e15(&upper.text[2]),
        e15(&upper.text[3]),
        e15(&upper.text[4])
    );
    let card3 = format!(
        "{}{}{}{}{}    3",
        e15(&upper.text[5]),
        e15(&upper.text[6]),
        e15(&lower.text[0]),
        e15(&lower.text[1]),
        e15(&lower.text[2])
    );
    let card4 = format!(
        "{}{}{}{}{}    4",
        e15(&lower.text[3]),
        e15(&lower.text[4]),
        e15(&lower.text[5]),
        e15(&lower.text[6]),
        e15(&upper.h298_r_text)
    );
    [card1, card2, card3, card4]
}

/// The ten records as one THERMO block - `tm4513_text`.
fn tm4513_text(rows: &[IiRow]) -> String {
    let mut text = String::from("THERMO ALL\n   200.000  1000.000  6000.000\n");
    for (code, ..) in SPECIES {
        for l in chemkin_record(code, rows) {
            text.push_str(&l);
            text.push('\n');
        }
    }
    text.push_str("END\n");
    text
}

/// **Gate CHR-THERMO** (SPEC-LIT §128.7): the TM-4513 records' NASA-7
/// thermochemistry against TP-2002 Table B1, live.
pub(crate) fn check_chr_thermo(c: &mut Checks) -> Result<()> {
    let (ii_line, ii_text) = key::load_text("mcbride-gordon-reno1993-table-II")?;
    c.note(&ii_line);
    let (b1_line, b1_text) = key::load_text("nasa-glenn2002-table-B1")?;
    c.note(&b1_line);
    let rows = ii_rows(&ii_text);
    let b1 = b1_rows(&b1_text);
    let set = ThermoSet::read_chemkin(
        &tm4513_text(&rows),
        "TM-4513 Table II",
        P_BAR,
        &ElementTable::standard(),
    )?;

    // ---- T2: cp(298.15) against Table B1 -----------------------------------
    let mut worst = (0.0f64, "N2");
    for entry in SPECIES {
        let (code, name, _) = entry;
        let s = sp_of(&set, name)?;
        let row = b1
            .iter()
            .find(|r| r.code == code)
            .ok_or_else(|| Error::Config(format!("Gate CHR-THERMO: Table B1 has no code {code}")))?;
        let cp = s.cp(298.15)?;
        let rel = (cp - row.cp_298).abs() / row.cp_298;
        c.check_p(
            &format!(
                "Gate CHR-THERMO: {name} cp(298.15) within 0.1 % of TP-2002 Table B1's {cp_g} J/(mol K) (SPEC-LIT 128.7)",
                cp_g = row.cp_298
            ),
            rel as Scalar,
            1.0e-3,
        );
        if rel > worst.0 {
            worst = (rel, name);
        }
    }
    c.note(&format!(
        "Gate CHR-THERMO T2 worst fit difference: {} at {:.3e} relative",
        worst.1, worst.0
    ));

    // ---- T3: H(298.15) against the heats of formation -----------------------
    for entry in SPECIES {
        let (code, name, _) = entry;
        let s = sp_of(&set, name)?;
        let h = s.h(298.15)?;
        // OH alone: TM-4513's TPIS78 record and TP-2002's Table B1 adopt
        // different heats of formation of OH, 39.347 against 37.278 kJ/mol -
        // a fact of the two printed tables that validate_key already pins.
        // So OH is held to its own record's printed heat of formation,
        // R_GAS * h298_r, and the gap against Table B1 is pinned on top.
        if code == 10 {
            let printed = R_GAS * row_of(&rows, 10, 1000.0).h298_r;
            c.check_p(
                "Gate CHR-THERMO: OH H(298.15) within 10 J/mol of its own record's printed heat of formation (SPEC-LIT 128.7)",
                (h - printed).abs() as Scalar,
                10.0,
            );
            let gap = h - 1000.0 * 37.278;
            c.check_p(
                "Gate CHR-THERMO: OH's gap to Table B1 sits at the pinned 2068.8818 J/mol (SPEC-LIT 128.7)",
                (gap - 2068.8818).abs() as Scalar,
                0.01,
            );
            continue;
        }
        let row = b1
            .iter()
            .find(|r| r.code == code)
            .ok_or_else(|| Error::Config(format!("Gate CHR-THERMO: Table B1 has no code {code}")))?;
        let want = 1000.0 * row.dfh_298;
        c.check_p(
            &format!(
                "Gate CHR-THERMO: {name} H(298.15) within 10 J/mol of Table B1's heat of formation (SPEC-LIT 128.7)"
            ),
            (h - want).abs() as Scalar,
            10.0,
        );
    }

    // ---- T4: the T_mid fit constraint and the interval selection ------------
    for entry in SPECIES {
        let (_, name, _) = entry;
        let s = sp_of(&set, name)?;
        let t = s.t_mid;
        for (label, fu, fl) in [
            ("cp/R", nasa7_cp_r(&s.upper, t), nasa7_cp_r(&s.lower, t)),
            ("H/RT", nasa7_h_rt(&s.upper, t), nasa7_h_rt(&s.lower, t)),
            ("S/R", nasa7_s_r(&s.upper, t), nasa7_s_r(&s.lower, t)),
        ] {
            let rel = (fu - fl).abs() / fu.abs().max(fl.abs());
            c.check_a(
                &format!(
                    "Gate CHR-THERMO: {name} {label} at T_mid meets between the two intervals (SPEC-LIT 128.7)"
                ),
                rel as Scalar,
                1.0e-6,
            );
        }
        c.require(
            &format!("Gate CHR-THERMO: {name} at T_mid evaluates the lower interval (SPEC-LIT 128.7)"),
            matches!(s.interval(t), Ok(Interval::Lower)),
        );
        c.require(
            &format!("Gate CHR-THERMO: {name} just above T_mid evaluates the upper interval (SPEC-LIT 128.7)"),
            matches!(s.interval(t + 1.0e-9), Ok(Interval::Upper)),
        );
        c.require(
            &format!("Gate CHR-THERMO: {name} at t_low evaluates the lower interval (SPEC-LIT 128.7)"),
            matches!(s.interval(200.0), Ok(Interval::Lower)),
        );
        c.require(
            &format!("Gate CHR-THERMO: {name} at t_high evaluates the upper interval (SPEC-LIT 128.7)"),
            matches!(s.interval(6000.0), Ok(Interval::Upper)),
        );
    }

    // ---- T5: the out-of-range refusals --------------------------------------
    let n2 = sp_of(&set, "N2")?;
    let e = match n2.cp_r(199.9) {
        Err(e) => e.to_string(),
        Ok(_) => String::new(),
    };
    c.require(
        "Gate CHR-THERMO: N2 cp/R at 199.9 K refused, naming N2, 199.9 and [200, 6000] (SPEC-LIT 128.7)",
        e.contains("N2") && e.contains("199.9") && e.contains("[200, 6000]"),
    );
    let e = match n2.h_rt(6000.1) {
        Err(e) => e.to_string(),
        Ok(_) => String::new(),
    };
    c.require(
        "Gate CHR-THERMO: N2 H/RT at 6000.1 K refused, naming 6000.1 (SPEC-LIT 128.7)",
        e.contains("6000.1"),
    );
    c.require(
        "Gate CHR-THERMO: N2 S/R at a NaN temperature refused (SPEC-LIT 128.7)",
        n2.s_r(f64::NAN).is_err(),
    );
    c.require(
        "Gate CHR-THERMO: N2 G/RT at an infinite temperature refused (SPEC-LIT 128.7)",
        n2.g_rt(f64::INFINITY).is_err(),
    );
    c.require(
        "Gate CHR-THERMO: N2 cp at t_low evaluates (SPEC-LIT 128.7)",
        n2.cp(200.0).is_ok(),
    );
    c.require(
        "Gate CHR-THERMO: N2 cp at t_high evaluates (SPEC-LIT 128.7)",
        n2.cp(6000.0).is_ok(),
    );

    // ---- T6: molar masses from the elements, and the Ar/he/XE weights -------
    for entry in SPECIES {
        let (code, name, _) = entry;
        let s = sp_of(&set, name)?;
        let mw = row_of(&rows, code, 1000.0).mol_weight;
        c.check_d(
            &format!(
                "Gate CHR-THERMO: {name} molar mass from the elements within 5e-6 g/mol of the printed column (SPEC-LIT 128.7)"
            ),
            (1000.0 * s.molar_mass - mw).abs() as Scalar,
            5.0e-6,
        );
    }
    let tab = ElementTable::standard();
    c.require(
        "Gate CHR-THERMO: the element table weighs Ar 39.948 (SPEC-LIT 128.7)",
        tab.weight("Ar") == Some(39.948),
    );
    let ar = b1
        .iter()
        .find(|r| r.code == 3)
        .ok_or_else(|| Error::Config("Gate CHR-THERMO: Table B1 has no Ar row".to_string()))?;
    c.check_d(
        "Gate CHR-THERMO: Table B1's Ar row weighs 39.948 g/mol (SPEC-LIT 128.7)",
        (39.948 - ar.mol_weight).abs() as Scalar,
        5.0e-6,
    );
    c.require(
        "Gate CHR-THERMO: the element table weighs he 4.002602 (SPEC-LIT 128.7)",
        tab.weight("he") == Some(4.002602),
    );
    c.require(
        "Gate CHR-THERMO: XE is not built in (SPEC-LIT 128.7)",
        tab.weight("XE").is_none(),
    );
    Ok(())
}




