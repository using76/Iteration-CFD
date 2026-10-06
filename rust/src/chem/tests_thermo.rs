// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.
// Provenance: see PROVENANCE.md. No GPL-licensed source was consulted.

//! The tests of `chem::thermo` - SPEC-LIT §128. The answer keys are the
//! TM-4513 Table II transcription and the TP-2002 Table B1 table already
//! held under `reference/`; their integrity is `ofgpu-validate`'s key
//! census, so no marker lines are added here.

use super::thermo::*;

const TABLE_II: &str = include_str!("../../../reference/mcbride-gordon-reno1993/table_II.csv");
const TABLE_B1: &str = include_str!("../../../reference/nasa-glenn2002/table_B1.csv");

/// One `SPECIES` entry: the TM-4513 species code, its name, and its element
/// counts, in this order.
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

/// One row of Table II. The CSV's coefficient TEXT is kept alongside its
/// value, because a record is built from the text as printed, never from a
/// re-formatted number.
struct IiRow {
    code: u32,
    t_lo: f64,
    t_hi: f64,
    mol_weight: f64,
    text: [String; 7],
    a: [f64; 7],
    h298_r_text: String,
    h298_r: f64,
}

fn ii_rows() -> Vec<IiRow> {
    let mut rows = Vec::new();
    let mut header = false;
    for line in TABLE_II.lines() {
        if line.starts_with('#') {
            continue;
        }
        if !header {
            header = true;
            continue;
        }
        let f: Vec<&str> = line.split(',').collect();
        let num = |i: usize| f[i].trim().parse::<f64>().unwrap();
        rows.push(IiRow {
            code: f[0].trim().parse().unwrap(),
            t_lo: num(1),
            t_hi: num(2),
            mol_weight: num(3),
            text: std::array::from_fn(|k| f[4 + k].trim().to_string()),
            a: std::array::from_fn(|k| num(4 + k)),
            h298_r_text: f[11].trim().to_string(),
            h298_r: num(11),
        });
    }
    rows
}

/// One row of Table B1: `species,mol_weight,cp_298,dfh_298`, `dfh_298` in kJ/mol.
struct B1Row {
    code: u32,
    mol_weight: f64,
    cp_298: f64,
    dfh_298: f64,
}

fn b1_rows() -> Vec<B1Row> {
    let mut rows = Vec::new();
    let mut header = false;
    for line in TABLE_B1.lines() {
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

/// The four 80-character lines of one species' record, from its two CSV rows
/// (upper first, as the file has them). The coefficient fields carry the
/// CSV's text as printed, right-justified in 15 columns; the element slots
/// run from column 25, each symbol left-justified in 2 columns and its count
/// right-justified in 3.
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

fn tm4513_text() -> String {
    let rows = ii_rows();
    let mut text = String::from("THERMO ALL\n   200.000  1000.000  6000.000\n");
    for (code, ..) in SPECIES {
        for l in chemkin_record(code, &rows) {
            text.push_str(&l);
            text.push('\n');
        }
    }
    text.push_str("END\n");
    text
}

fn tm4513_set() -> ThermoSet {
    ThermoSet::read_chemkin(&tm4513_text(), "TM-4513 Table II", P_BAR, &ElementTable::standard())
        .unwrap()
}

fn one_record(rec: &[String; 4]) -> String {
    let mut text = String::from("THERMO ALL\n   200.000  1000.000  6000.000\n");
    for l in rec {
        text.push_str(l);
        text.push('\n');
    }
    text.push_str("END\n");
    text
}

fn read_err(text: &str, source: &str) -> String {
    ThermoSet::read_chemkin(text, source, P_BAR, &ElementTable::standard())
        .unwrap_err()
        .to_string()
}

// ==========================================================================
//  T1
// ==========================================================================

#[test]
fn thermo_reads_a_chemkin_record_by_fixed_columns() {
    let rows = ii_rows();
    let n2 = chemkin_record(1, &rows);
    let oh = chemkin_record(10, &rows);
    // 1. the exact bytes, card by card
    assert_eq!(
        n2,
        [
            "N2                TM4513N   2               G   200.000  6000.000 1000.00      1",
            " 2.95257626E+00 1.39690057E-03-4.92631691E-07 7.86010367E-11-4.60755321E-15    2",
            "-9.23948645E+02 5.87189252E+00 3.53100528E+00-1.23660987E-04-5.02999437E-07    3",
            " 2.43530612E-09-1.40881235E-12-1.04697628E+03 2.96747468E+00 0.00000000E+00    4",
        ],
        "N2"
    );
    assert_eq!(
        oh,
        [
            "OH                TM4513O   1H   1          G   200.000  6000.000 1000.00      1",
            " 2.83864607E+00 1.10725586E-03-2.93914978E-07 4.20524247E-11-2.42169092E-15    2",
            " 3.94395852E+03 5.84452662E+00 3.99201543E+00-2.40131752E-03 4.61793841E-06    3",
            "-3.88113333E-09 1.36411470E-12 3.61508056E+03-1.03925458E-01 4.73234213E+03    4",
        ],
        "OH"
    );
    // 2. the whole ten-species set, exactly
    let set = tm4513_set();
    assert_eq!(set.species().len(), 10, "the TM-4513 set holds ten species");
    for (i, entry) in SPECIES.iter().enumerate() {
        let (code, name, _) = *entry;
        let s = &set.species()[i];
        let up = row_of(&rows, code, 1000.0);
        let lo = row_of(&rows, code, 200.0);
        assert_eq!(s.name, name, "species {i} in CSV order");
        assert_eq!(s.upper, up.a, "{name}: upper coefficients equal the CSV's numbers");
        assert_eq!(s.lower, lo.a, "{name}: lower coefficients equal the CSV's numbers");
        assert_eq!((s.t_low, s.t_mid, s.t_high), (200.0, 1000.0, 6000.0), "{name}");
    }
    let ch4 = set.get("CH4").unwrap();
    assert_eq!(
        ch4.elements,
        vec![("C".to_string(), 1u32), ("H".to_string(), 4u32)],
        "CH4"
    );
    assert!(set.notes().is_empty(), "a clean set carries no notes");
    assert_eq!(set.p_ref(), P_BAR);
    // 3. GRI's layout: plain THERMO, a line 2, and a blank T_mid field
    let mut c1 = n2[0].clone();
    c1.replace_range(65..73, "        ");
    let text = format!(
        "THERMO\n   300.000  1000.000  5000.000\n! a comment line\n\n{}\n{}\n{}\n{}\nEND\n",
        c1, n2[1], n2[2], n2[3]
    );
    let s = ThermoSet::read_chemkin(&text, "GRI layout", P_BAR, &ElementTable::standard()).unwrap();
    let gri = s.get("N2").unwrap();
    assert_eq!(
        (gri.t_low, gri.t_mid, gri.t_high),
        (200.0, 1000.0, 6000.0),
        "N2: t_mid from line 2, the range from card 1"
    );
    let text = format!("THERMO ALL\n{}\n{}\n{}\n{}\nEND\n", c1, n2[1], n2[2], n2[3]);
    let e = read_err(&text, "blank T_mid with no line 2");
    println!("blank T_mid with no line 2: {e}");
    assert!(e.contains("N2"), "{e}");
    // 4. a Fortran D exponent, and CRLF line endings
    let mut c2d = n2[1].clone();
    c2d.replace_range(0..15, " 2.95257626D+00");
    let s = ThermoSet::read_chemkin(
        &one_record(&[n2[0].clone(), c2d, n2[2].clone(), n2[3].clone()]),
        "D exponent",
        P_BAR,
        &ElementTable::standard(),
    )
    .unwrap();
    assert_eq!(
        s.get("N2").unwrap().upper[0],
        row_of(&rows, 1, 1000.0).a[0],
        "N2: the D exponent reads to the same value"
    );
    let text = tm4513_text().replace('\n', "\r\n");
    let crlf = ThermoSet::read_chemkin(&text, "TM-4513 Table II", P_BAR, &ElementTable::standard())
        .unwrap();
    assert_eq!(crlf, tm4513_set(), "N2: the CRLF set equals the LF set");
    // 5. a repeated record
    let mut text = String::from("THERMO ALL\n   200.000  1000.000  6000.000\n");
    for _ in 0..2 {
        for l in &n2 {
            text.push_str(l);
            text.push('\n');
        }
    }
    text.push_str("END\n");
    let s = ThermoSet::read_chemkin(&text, "repeat", P_BAR, &ElementTable::standard()).unwrap();
    assert_eq!(s.species().len(), 1, "the first N2 record is kept");
    assert_eq!(s.notes().len(), 1, "one note for the repeated N2 record");
    println!("note: {}", s.notes()[0]);
    assert!(s.notes()[0].contains("N2"), "{}", s.notes()[0]);
}

// ==========================================================================
//  T2
// ==========================================================================

#[test]
fn thermo_cp_matches_nasa_table_b1() {
    let set = tm4513_set();
    let b1 = b1_rows();
    let mut worst = (0.0f64, String::new());
    for entry in SPECIES {
        let (code, name, _) = entry;
        let s = set.get(name).unwrap();
        let row = b1.iter().find(|r| r.code == code).unwrap();
        let cp = s.cp(298.15).unwrap();
        let rel = (cp - row.cp_298).abs() / row.cp_298;
        println!(
            "{name}: cp(298.15) = {cp} J/(mol K) against Table B1's {}, relative {rel:.3e}",
            row.cp_298
        );
        assert!(
            rel <= 1.0e-3,
            "{name}: cp(298.15) = {cp} differs from Table B1's {} by {rel} relative, \
             above the 0.1 percent fit difference of SPEC-LIT §128.7",
            row.cp_298
        );
        if rel > worst.0 {
            worst = (rel, name.to_string());
        }
    }
    println!("T2 worst: {} at {:.3e} relative", worst.1, worst.0);
}

// ==========================================================================
//  T3
// ==========================================================================

#[test]
fn thermo_h_is_heat_of_formation() {
    let set = tm4513_set();
    let b1 = b1_rows();
    let rows = ii_rows();
    // OH alone: TM-4513's TPIS78 record and TP-2002's Table B1 adopt
    // different heats of formation of OH, 39.347 against 37.278 kJ/mol - a
    // fact of the two printed tables that validate_key already pins. So OH
    // is held to its own record's printed heat of formation, R_GAS * h298_r,
    // and the gap against Table B1 is pinned on top.
    for entry in SPECIES {
        let (code, name, _) = entry;
        let s = set.get(name).unwrap();
        let h = s.h(298.15).unwrap();
        if code == 10 {
            let printed = R_GAS * row_of(&rows, 10, 1000.0).h298_r;
            let d = (h - printed).abs();
            println!(
                "OH: H(298.15) = {h} J/mol against its own record's printed {printed}, \
                 |diff| = {d:.3e}"
            );
            assert!(d <= 10.0, "OH: H = {h} is {d} J/mol from its own printed heat of formation");
            let gap = h - 1000.0 * 37.278;
            println!("OH: the gap to Table B1 is {gap} J/mol");
            assert!(
                (gap - 2068.8818).abs() <= 0.01,
                "OH: the gap to Table B1 is {gap} J/mol, away from the pinned 2068.8818"
            );
            continue;
        }
        let row = b1.iter().find(|r| r.code == code).unwrap();
        let want = 1000.0 * row.dfh_298;
        let d = (h - want).abs();
        println!("{name}: H(298.15) = {h} J/mol against Table B1's {want}, |diff| = {d:.3e}");
        assert!(d <= 10.0, "{name}: H(298.15) = {h} is {d} J/mol from Table B1's {want}");
    }
}

// ==========================================================================
//  T4
// ==========================================================================

#[test]
fn thermo_ranges_meet_at_tmid() {
    let set = tm4513_set();
    let mut worst = (0.0f64, String::new());
    for entry in SPECIES {
        let (_, name, _) = entry;
        let s = set.get(name).unwrap();
        let t = s.t_mid;
        for (label, fu, fl) in [
            ("cp/R", nasa7_cp_r(&s.upper, t), nasa7_cp_r(&s.lower, t)),
            ("H/RT", nasa7_h_rt(&s.upper, t), nasa7_h_rt(&s.lower, t)),
            ("S/R", nasa7_s_r(&s.upper, t), nasa7_s_r(&s.lower, t)),
        ] {
            let rel = (fu - fl).abs() / fu.abs().max(fl.abs());
            println!("{name}: {label} at T_mid = {fu} upper against {fl} lower, {rel:.3e} relative");
            assert!(
                rel <= 1.0e-6,
                "{name}: {label} at T_mid differs by {rel} relative between the two \
                 intervals, above the fit constraint"
            );
            if rel > worst.0 {
                worst = (rel, format!("{name} {label}"));
            }
        }
        assert_eq!(s.interval(t).unwrap(), Interval::Lower, "{name} at T_mid");
        assert_eq!(s.interval(t + 1.0e-9).unwrap(), Interval::Upper, "{name} just above T_mid");
        assert_eq!(s.interval(200.0).unwrap(), Interval::Lower, "{name} at t_low");
        assert_eq!(s.interval(6000.0).unwrap(), Interval::Upper, "{name} at t_high");
    }
    println!("T4 worst: {} at {:.3e} relative", worst.1, worst.0);
}

// ==========================================================================
//  T5
// ==========================================================================

#[test]
fn thermo_out_of_range_is_refused() {
    let set = tm4513_set();
    let n2 = set.get("N2").unwrap();
    let e = n2.cp_r(199.9).unwrap_err().to_string();
    println!("N2 at 199.9 K: {e}");
    assert!(
        e.contains("N2") && e.contains("199.9") && e.contains("[200, 6000]"),
        "{e}"
    );
    let e = n2.h_rt(6000.1).unwrap_err().to_string();
    println!("N2 at 6000.1 K: {e}");
    assert!(e.contains("6000.1"), "{e}");
    assert!(n2.s_r(f64::NAN).is_err(), "N2 at NaN");
    assert!(n2.g_rt(f64::INFINITY).is_err(), "N2 at infinity");
    assert!(n2.cp(200.0).is_ok(), "N2 at t_low evaluates");
    assert!(n2.cp(6000.0).is_ok(), "N2 at t_high evaluates");
}

// ==========================================================================
//  T6
// ==========================================================================

#[test]
fn thermo_molar_mass_from_elements() {
    let set = tm4513_set();
    let rows = ii_rows();
    for entry in SPECIES {
        let (code, name, _) = entry;
        let s = set.get(name).unwrap();
        let mw = row_of(&rows, code, 1000.0).mol_weight;
        let d = (1000.0 * s.molar_mass - mw).abs();
        println!(
            "{name}: 1000 x molar_mass = {:.8} g/mol against the printed {mw}, |diff| = {d:.3e}",
            1000.0 * s.molar_mass
        );
        assert!(
            d <= 5.0e-6,
            "{name}: 1000 x molar_mass = {} is {d} g/mol from the printed {mw}",
            1000.0 * s.molar_mass
        );
    }
    let tab = ElementTable::standard();
    assert_eq!(tab.weight("Ar"), Some(39.948), "Ar");
    let ar = b1_rows().into_iter().find(|r| r.code == 3).unwrap();
    assert!(
        (39.948 - ar.mol_weight).abs() <= 5.0e-6,
        "Ar: 39.948 against Table B1's {}",
        ar.mol_weight
    );
    assert_eq!(tab.weight("he"), Some(4.002602), "he");
    assert_eq!(tab.weight("XE"), None, "XE is not built in");
    // an element the table does not know is refused by name, then read once given
    let mut rec = chemkin_record(1, &rows);
    rec[0].replace_range(24..26, "XE");
    let text = one_record(&rec);
    let e = read_err(&text, "unknown element");
    println!("XE in the formula: {e}");
    assert!(e.contains("XE") && e.contains("N2"), "{e}");
    let xe = ElementTable::standard().with("XE", 131.293).unwrap();
    let s = ThermoSet::read_chemkin(&text, "unknown element", P_BAR, &xe).unwrap();
    assert_eq!(
        s.get("N2").unwrap().molar_mass,
        2.0 * 131.293 / 1000.0,
        "N2 with the given XE weight"
    );
    assert!(ElementTable::standard().with("X1", 1.0).is_err(), "X1 is not a symbol");
    assert!(ElementTable::standard().with("D", -2.0).is_err(), "a weight below zero");
}

// ==========================================================================
//  T7
// ==========================================================================

#[test]
fn thermo_reader_refusals() {
    let rows = ii_rows();
    let n2 = chemkin_record(1, &rows);

    // 1. card 3 with a wrong number in column 80
    let mut rec = n2.clone();
    rec[2].replace_range(79..80, "5");
    let e = read_err(&one_record(&rec), "card number");
    println!("1. {e}");
    assert!(e.contains("line 5") && e.contains("N2"), "{e}");

    // 2. a condensed phase
    let mut rec = n2.clone();
    rec[0].replace_range(44..45, "S");
    let e = read_err(&one_record(&rec), "phase");
    println!("2. {e}");
    assert!(e.contains("line 3") && e.contains("N2") && e.contains("condensed"), "{e}");

    // 3. the record cut after card 2 by END
    let text = format!("THERMO ALL\n   200.000  1000.000  6000.000\n{}\n{}\nEND\n", n2[0], n2[1]);
    let e = read_err(&text, "cut by END");
    println!("3. {e}");
    assert!(e.contains("line 3") && e.contains("N2"), "{e}");

    // 4. a card 2 one character short
    let mut rec = n2.clone();
    rec[1].truncate(79);
    let e = read_err(&one_record(&rec), "short line");
    println!("4. {e}");
    assert!(e.contains("line 4") && e.contains("N2"), "{e}");

    // 5. a field that is not a number
    let mut rec = n2.clone();
    rec[1] = rec[1].replace("E+00", "X+00");
    let e = read_err(&one_record(&rec), "bad field");
    println!("5. {e}");
    assert!(e.contains("line 4") && e.contains("N2"), "{e}");

    // 6. T_low above T_mid
    let mut rec = n2.clone();
    rec[0].replace_range(45..55, "  1500.000");
    let e = read_err(&one_record(&rec), "temperature order");
    println!("6. {e}");
    assert!(e.contains("line 3") && e.contains("N2"), "{e}");

    // 7. no THERMO keyword line
    let e = read_err("REACTIONS\n", "keyword");
    println!("7. {e}");
    assert!(e.contains("line 1") && e.contains("REACTIONS"), "{e}");

    // 8. a stray line between two records
    let mut text = String::from("THERMO ALL\n   200.000  1000.000  6000.000\n");
    for l in &n2 {
        text.push_str(l);
        text.push('\n');
    }
    text.push_str("hello\n");
    for l in &n2 {
        text.push_str(l);
        text.push('\n');
    }
    text.push_str("END\n");
    let e = read_err(&text, "stray line");
    println!("8. {e}");
    assert!(e.contains("line 7"), "{e}");

    // 9. a standard pressure that is not a finite pressure above zero
    for p_ref in [0.0, f64::NAN] {
        let err = ThermoSet::read_chemkin(
            &one_record(&n2),
            "p_ref",
            p_ref,
            &ElementTable::standard(),
        )
        .unwrap_err();
        println!("9. p_ref = {p_ref}: {err}");
        assert!(matches!(err, crate::Error::Config(_)), "p_ref = {p_ref} is a Config refusal");
    }
}

// ==========================================================================
//  T8
// ==========================================================================

#[test]
fn thermo_p_ref_shifts_only_s_and_g() {
    let a = tm4513_set();
    let b = a.with_p_ref(P_ATM).unwrap();
    assert_eq!(b.p_ref(), P_ATM);
    let shift = -0.013162986526280862; // -ln(101325/100000)
    for entry in SPECIES {
        let (_, name, _) = entry;
        let sa = a.get(name).unwrap();
        let sb = b.get(name).unwrap();
        let mut ds_298 = 0.0;
        for t in [250.0, 298.15, 1000.0, 1500.0, 5000.0] {
            assert_eq!(
                sb.cp_r(t).unwrap().to_bits(),
                sa.cp_r(t).unwrap().to_bits(),
                "{name} at {t}: cp is bitwise unchanged"
            );
            assert_eq!(
                sb.h_rt(t).unwrap().to_bits(),
                sa.h_rt(t).unwrap().to_bits(),
                "{name} at {t}: H is bitwise unchanged"
            );
            let ds = sb.s_r(t).unwrap() - sa.s_r(t).unwrap();
            let dg = sb.g_rt(t).unwrap() - sa.g_rt(t).unwrap();
            assert!((ds - shift).abs() <= 1.0e-12, "{name} at {t}: S/R moved by {ds}");
            assert!((dg + shift).abs() <= 1.0e-12, "{name} at {t}: G/RT moved by {dg}");
            if t == 298.15 {
                ds_298 = ds;
            }
        }
        println!("{name}: S/R moved by {ds_298} at 298.15 K");
    }
    assert_eq!(a.with_p_ref(P_BAR).unwrap(), a, "the same pressure is a plain copy");
    let c = b.with_p_ref(P_BAR).unwrap();
    for entry in SPECIES {
        let (_, name, _) = entry;
        let sa = a.get(name).unwrap();
        let sc = c.get(name).unwrap();
        for (which, lo, hi) in
            [("upper", &sa.upper, &sc.upper), ("lower", &sa.lower, &sc.lower)]
        {
            for (k, (x, y)) in lo.iter().zip(hi.iter()).enumerate() {
                assert!(
                    (y - x).abs() <= 1.0e-14,
                    "{name} {which}[{k}]: the round trip moved a coefficient by {}",
                    y - x
                );
            }
        }
    }
    assert!(a.with_p_ref(-1.0).is_err(), "a negative standard pressure");
}

// ==========================================================================
//  T9
// ==========================================================================

/// A test-only monatomic species AR: element AR 1, 300-1000-5000 K, and
/// cp/R = 2.5 on both intervals, written in the same four-card layout.
fn ar_record() -> [String; 4] {
    let zero = |n: usize| -> String { (0..n).map(|_| format!("{:>15}", "0")).collect() };
    let card1 = format!(
        "{:<18}TM4513{:<20}G{:>10.3}{:>10.3}{:>8.2}      1",
        "AR", "AR  1", 300.0, 5000.0, 1000.0
    );
    let cp25 = format!("{:>15}", "2.5");
    let zeros4 = zero(4);
    let zero1 = zero(1);
    let card2 = format!("{cp25}{zeros4}    2");
    let card3 = format!("{}    3", zero(5));
    let card4 = format!("{zeros4}{zero1}    4");
    [card1, card2, card3, card4]
}

fn air_text(with_ar: bool) -> String {
    let rows = ii_rows();
    let mut text = String::from("THERMO ALL\n   200.000  1000.000  6000.000\n");
    for code in [1u32, 2u32] {
        for l in chemkin_record(code, &rows) {
            text.push_str(&l);
            text.push('\n');
        }
    }
    if with_ar {
        for l in ar_record() {
            text.push_str(&l);
            text.push('\n');
        }
    }
    text.push_str("END\n");
    text
}

fn read_air(with_ar: bool, source: &str) -> ThermoSet {
    ThermoSet::read_chemkin(&air_text(with_ar), source, P_BAR, &ElementTable::standard()).unwrap()
}

#[test]
fn thermo_mixture_of_air() {
    let s = read_air(false, "air");
    let y = [0.767, 0.233];

    let w = s.molar_mass_mix(&y).unwrap();
    println!("air: W = {w} kg/mol");
    assert!(
        (w - 0.02885070434421417).abs() / 0.02885070434421417 <= 1.0e-12,
        "air: W = {w}"
    );
    let x = s.mole_fractions(&y).unwrap();
    println!("air: X = [{}, {}], the sum is {}", x[0], x[1], x[0] + x[1]);
    assert!((x[0] - 0.7899229311036068).abs() / 0.7899229311036068 <= 1.0e-12, "X_N2 = {}", x[0]);
    assert!(
        (x[1] - 0.21007706889639305).abs() / 0.21007706889639305 <= 1.0e-12,
        "X_O2 = {}",
        x[1]
    );
    assert!((x[0] + x[1] - 1.0).abs() <= 1.0e-15, "the mole fractions sum to {}", x[0] + x[1]);

    let cp300 = s.cp_mass(300.0, &y).unwrap();
    println!("air: cp(300 K) = {cp300} J/(kg K)");
    assert!((cp300 - 1011.433578521991).abs() / 1011.433578521991 <= 1.0e-12, "cp(300 K)");
    let h300 = s.h_mass(300.0, &y).unwrap();
    println!("air: h(300 K) = {h300} J/kg");
    assert!((h300 - 1871.0549314323957).abs() / 1871.0549314323957 <= 1.0e-12, "h(300 K)");
    let cp1500 = s.cp_mass(1500.0, &y).unwrap();
    println!("air: cp(1500 K) = {cp1500} J/(kg K)");
    assert!(
        (cp1500 - 1217.7254737920775).abs() / 1217.7254737920775 <= 1.0e-12,
        "cp(1500 K)"
    );
    let h1500 = s.h_mass(1500.0, &y).unwrap();
    println!("air: h(1500 K) = {h1500} J/kg");
    assert!(
        (h1500 - 1345917.5205008553).abs() / 1345917.5205008553 <= 1.0e-12,
        "h(1500 K)"
    );

    let c = s.concentrations(101325.0, 300.0, &[1.0, 0.0]).unwrap();
    let want = 101325.0 / (R_GAS * 300.0);
    assert!(
        (c[0] - want).abs() / want <= 1.0e-14,
        "N2: the concentration is {} against {want}",
        c[0]
    );
    assert_eq!(c[1], 0.0, "O2 at Y = 0 contributes exactly nothing");

    let n2 = s.get("N2").unwrap();
    let wp = s.molar_mass_mix(&[1.0, 0.0]).unwrap();
    println!("pure N2: W = {wp} kg/mol against the record's {}", n2.molar_mass);
    assert!(
        (wp - n2.molar_mass).abs() / n2.molar_mass <= 4.5e-16,
        "N2: pure-species W = {wp} against {}",
        n2.molar_mass
    );
    let cpn = s.cp_mass(300.0, &[1.0, 0.0]).unwrap();
    let want = n2.cp(300.0).unwrap() / n2.molar_mass;
    assert!(
        (cpn - want).abs() / want <= 1.0e-15,
        "N2: pure-species cp = {cpn} against {want}"
    );

    // the refusals
    assert!(s.molar_mass_mix(&[0.767]).is_err(), "one value for two species");
    let e = s.mole_fractions(&[-0.1, 1.1]).unwrap_err().to_string();
    println!("a negative mass fraction: {e}");
    assert!(e.contains("N2"), "{e}");
    assert!(s.mole_fractions(&[f64::NAN, 1.0]).is_err(), "a NaN mass fraction");
    let e = s.cp_mass(300.0, &[0.0, 0.0]).unwrap_err().to_string();
    println!("all-zero Y: {e}");
    assert!(e.contains("all mass fractions are zero"), "{e}");
    assert!(s.concentrations(0.0, 300.0, &y).is_err(), "a pressure of zero");
    let e = s.cp_mass(7000.0, &y).unwrap_err().to_string();
    println!("air at 7000 K: {e}");
    assert!(e.contains("N2"), "{e}");

    // the Y = 0 rule: a third species at Y = 0 moves nothing, even outside
    // its own range; the same species at Y > 0 is refused by name
    let s3 = read_air(true, "air and AR");
    let y3 = [0.767, 0.233, 0.0];
    let cp250_3 = s3.cp_mass(250.0, &y3).unwrap();
    let cp250 = s.cp_mass(250.0, &y).unwrap();
    println!("air and AR at Y = 0: cp(250 K) = {cp250_3} against {cp250}");
    assert_eq!(cp250_3.to_bits(), cp250.to_bits(), "AR at Y = 0 is not evaluated at all");
    let e = s3.cp_mass(250.0, &[0.7, 0.2, 0.1]).unwrap_err().to_string();
    println!("AR above its Y: {e}");
    assert!(e.contains("AR"), "{e}");
    let cp_ar = s3.cp_mass(2000.0, &[0.0, 0.0, 1.0]).unwrap();
    let want = 2.5 * R_GAS / 0.039948;
    println!("pure AR: cp(2000 K) = {cp_ar} against {want} J/(kg K)");
    assert!((cp_ar - want).abs() / want <= 1.0e-15, "AR: pure-species cp = {cp_ar}");
}

// ==========================================================================
//  T10
// ==========================================================================

#[test]
fn thermo_reads_the_wide_tmid_field_and_the_fifth_element_slot() {
    let rows = ii_rows();
    let n2 = chemkin_record(1, &rows);

    // 1. the wide T_mid field, as GRI-Mech 3.0 writes it
    let mut rec = n2.clone();
    rec[0].replace_range(65..75, "  1000.000");
    let s = ThermoSet::read_chemkin(
        &one_record(&rec),
        "wide T_mid",
        P_BAR,
        &ElementTable::standard(),
    )
    .unwrap();
    let n = s.get("N2").unwrap();
    println!("1. wide T_mid = {}", n.t_mid);
    assert_eq!(n.t_mid, 1000.0, "the wide T_mid field reads");
    assert_eq!(n.elements, vec![("N".to_string(), 2u32)], "no fifth slot in the wide form");

    // 2. the wide field is read whole
    let mut rec = n2.clone();
    rec[0].replace_range(65..75, "  1000.125");
    let s = ThermoSet::read_chemkin(
        &one_record(&rec),
        "wide whole",
        P_BAR,
        &ElementTable::standard(),
    )
    .unwrap();
    assert_eq!(
        s.get("N2").unwrap().t_mid,
        1000.125,
        "an 8-wide read would have given 1000.1"
    );

    // 3. the wide field with junk after it
    let mut rec = n2.clone();
    rec[0].replace_range(65..75, "  1000.000");
    rec[0].replace_range(75..78, "  7");
    let e = read_err(&one_record(&rec), "junk after the wide field");
    println!("3. {e}");
    assert!(e.contains("N2") && e.contains("line 3"), "{e}");

    // 4. the fifth element slot still works
    let mut rec = n2.clone();
    rec[0].replace_range(73..78, "HE  1");
    let s = ThermoSet::read_chemkin(
        &one_record(&rec),
        "fifth slot",
        P_BAR,
        &ElementTable::standard(),
    )
    .unwrap();
    let n = s.get("N2").unwrap();
    assert_eq!(
        n.elements,
        vec![("N".to_string(), 2u32), ("HE".to_string(), 1u32)],
        "HE 1 in columns 74-78"
    );
    assert_eq!(n.t_mid, 1000.0, "T_mid is still columns 66-73");
    assert_eq!(n.molar_mass, (2.0 * 14.00674 + 4.002602) / 1000.0, "the exact sum");

    // 5. a wide blank T_mid takes line 2; column 74 is then a space, so the
    // 8-wide branch with columns 74-78 blank must read the same
    let mut rec = n2.clone();
    rec[0].replace_range(65..75, "          ");
    let text = format!(
        "THERMO\n   300.000  1000.000  5000.000\n{}\n{}\n{}\n{}\nEND\n",
        rec[0], rec[1], rec[2], rec[3]
    );
    let s = ThermoSet::read_chemkin(&text, "blank wide", P_BAR, &ElementTable::standard()).unwrap();
    assert_eq!(s.get("N2").unwrap().t_mid, 1000.0, "T_mid from line 2");
}

// ==========================================================================
//  T11
// ==========================================================================

#[test]
fn thermo_reader_block_edges() {
    // 1. THERMO END is an empty set
    let s = ThermoSet::read_chemkin("THERMO\nEND\n", "empty", P_BAR, &ElementTable::standard())
        .unwrap();
    assert_eq!(s.species().len(), 0, "THERMO END is an empty set");
    assert!(s.notes().is_empty(), "an empty block carries no notes");

    // 2. a line 2 and no records
    let s = ThermoSet::read_chemkin(
        "THERMO ALL\n   300.000  1000.000  5000.000\nEND\n",
        "line 2 only",
        P_BAR,
        &ElementTable::standard(),
    )
    .unwrap();
    assert_eq!(s.species().len(), 0, "a line 2 alone is still an empty set");

    // 3. the keyword line alone
    let s =
        ThermoSet::read_chemkin("THERMO\n", "keyword alone", P_BAR, &ElementTable::standard())
            .unwrap();
    assert_eq!(s.species().len(), 0, "nothing after the keyword line");

    // 4. a card 1 whose species name starts like a block keyword still reads
    let rows = ii_rows();
    for name in ["SPECX", "ELEM2"] {
        let mut rec = chemkin_record(1, &rows);
        rec[0].replace_range(0..18, &format!("{name:<18}"));
        let s = ThermoSet::read_chemkin(
            &one_record(&rec),
            name,
            P_BAR,
            &ElementTable::standard(),
        )
        .unwrap();
        assert_eq!(s.species().len(), 1, "{name}: the record reads");
        let sp = &s.species()[0];
        assert_eq!(sp.name, name, "{name}: read under its own name");
        assert_eq!(sp.elements, vec![("N".to_string(), 2u32)], "{name}");
    }

    // 5. the block ends at the next block keyword
    let s = ThermoSet::read_chemkin(
        "THERMO\nREACTIONS\n",
        "REACTIONS",
        P_BAR,
        &ElementTable::standard(),
    )
    .unwrap();
    assert_eq!(s.species().len(), 0, "REACTIONS ends the block");

    // 6. a card 1 whose five element slots are all empty is refused
    let mut rec = chemkin_record(1, &rows);
    rec[0].replace_range(24..29, "     ");
    let e = read_err(&one_record(&rec), "no element");
    println!("6. {e}");
    assert!(
        e.contains("N2") && e.contains("line 3") && e.contains("no element"),
        "{e}"
    );
}
