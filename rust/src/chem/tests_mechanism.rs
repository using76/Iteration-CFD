// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.
// Provenance: see PROVENANCE.md. No GPL-licensed source was consulted.

//! The tests of `chem::mechanism` - SPEC-LIT §129, with §130.2 (the units)
//! and §130.5 (FORD and RORD). The fixtures are made-up mechanisms shaped
//! like WD1 and JL4 with placeholder rate numbers and placeholder thermo
//! records: the tests measure parsing, the exact element balance, the unit
//! conversions and the refusals, never thermochemistry. The truth values of
//! the unit conversions are a 50-digit decimal oracle rounded to the nearest
//! f64.

use super::mechanism::*;
use super::thermo::*;

const TH_30K: f64 = 15096.586005241212;
const TH_40K: f64 = 20128.781340321613;
const TH_20K: f64 = 10064.390670160807;
const TH_47780: f64 = 24043.829311014168;
const A_1E12: f64 = 31622776.60168379;
const A_1E11: f64 = 3162277.6601683795;
const A_1E16: f64 = 316227766016.83795;
/// Class D: exact up to the conversion's one rounding.
const TOL: f64 = 1.0e-15;
const TEMPS: &str = "   300.000  1000.000  5000.000";

// ==========================================================================
//  Fixtures
// ==========================================================================

/// The element counts of the fixture species.
fn elems_of(name: &str) -> &'static [(&'static str, u32)] {
    match name {
        "CH4" => &[("C", 1), ("H", 4)],
        "O2" => &[("O", 2)],
        "CO2" => &[("C", 1), ("O", 2)],
        "H2O" => &[("H", 2), ("O", 1)],
        "N2" => &[("N", 2)],
        "CO" => &[("C", 1), ("O", 1)],
        "H2" => &[("H", 2)],
        "OH" => &[("O", 1), ("H", 1)],
        "H" => &[("H", 1)],
        "HO2" => &[("H", 1), ("O", 2)],
        other => panic!("the fixture has no species {other}"),
    }
}

/// One four-card THERMO record, four lines of exactly 80 bytes joined by `\n`, with no
/// trailing newline. Both intervals read [a1, 0, 0, 0, 0, 0, 0].
fn rec(name: &str, elems: &[(&str, u32)], a1: f64) -> String {
    let mut c1 = format!("{name:<18}FIXTUR");
    for k in 0..4 {
        match elems.get(k) {
            Some((sym, n)) => c1.push_str(&format!("{sym:<2}{n:>3}")),
            None => c1.push_str("     "),
        }
    }
    c1.push('G');
    c1.push_str(&format!("{:10.3}", 300.0));
    c1.push_str(&format!("{:10.3}", 5000.0));
    c1.push_str(&format!("{:8.2}", 1000.0));
    c1.push_str("      1");
    let zero = format!("{:>15}", "0.00000000E+00");
    let one = format!("{:>15}", format!("{:.8E}", a1));
    let c2 = format!("{one}{zero}{zero}{zero}{zero}    2");
    let c3 = format!("{zero}{zero}{one}{zero}{zero}    3");
    let c4 = format!("{zero}{zero}{zero}{zero}{}    4", " ".repeat(15));
    for c in [&c1, &c2, &c3, &c4] {
        assert_eq!(c.len(), 80, "a fixture card is 80 bytes: {c:?}");
    }
    format!("{c1}\n{c2}\n{c3}\n{c4}")
}

/// The four lines of the fixture record of a named species.
fn rec_lines(name: &str) -> Vec<String> {
    rec(name, elems_of(name), 2.5).split('\n').map(str::to_string).collect()
}

fn push_records(v: &mut Vec<String>, names: &[&str]) {
    for n in names {
        v.extend(rec_lines(n));
    }
}

fn s(x: &str) -> String {
    x.to_string()
}

fn text(lines: &[String]) -> String {
    lines.join("\n")
}

fn read(lines: &[String]) -> crate::Result<Mechanism> {
    Mechanism::read_chemkin(&text(lines), "t", None)
}

/// The mechanism, or a panic naming the text that failed to read.
fn must(lines: &[String]) -> Mechanism {
    match read(lines) {
        Ok(m) => m,
        Err(e) => panic!("the fixture should read, but: {e}\n{}", text(lines)),
    }
}

/// The refusal's text, or a panic when the fixture reads.
fn refusal(lines: &[String]) -> String {
    match read(lines) {
        Ok(_) => panic!("the fixture should be refused:\n{}", text(lines)),
        Err(e) => e.to_string(),
    }
}

fn has(msg: &str, parts: &[&str]) {
    for p in parts {
        assert!(msg.contains(p), "the message should contain {p:?}: {msg}");
    }
}

fn q(n: i64, d: i64) -> Rational {
    Rational::new(n, d).unwrap()
}

/// `|x - truth| <= tol * |truth|`.
fn within(x: f64, truth: f64) -> bool {
    (x - truth).abs() <= TOL * truth.abs()
}

fn names(m: &Mechanism) -> Vec<&str> {
    m.species().iter().map(String::as_str).collect()
}

/// T1's fixture: WD1-shaped, 36 numbered lines.
fn wd_lines() -> Vec<String> {
    let mut v = vec![
        s("! Westbrook-Dryer one-step shape: a CHR-02a parser fixture."),
        s("! Its rate numbers are placeholders, not the paper's: the paper is unread."),
        s("ELEMENTS"),
        s("C H O N"),
        s("END"),
        s("SPECIES"),
        s("CH4 O2 CO2 H2O N2"),
        s("END"),
        s("THERMO ALL"),
        s(TEMPS),
    ];
    push_records(&mut v, &["CH4", "O2", "CO2", "H2O", "N2"]);
    v.extend([
        s("END"),
        s("REACTIONS"),
        s("CH4 + 2 O2 => CO2 + 2 H2O     1.0E+12  0.0  30000.0"),
        s("  FORD /CH4 0.25/"),
        s("  FORD /O2 1.5/"),
        s("END"),
    ]);
    assert_eq!(v.len(), 36);
    v
}

/// T2's fixture: JL4-shaped, 42 numbered lines, no END.
fn jl_lines() -> Vec<String> {
    let mut v = vec![
        s("! Jones-Lindstedt four-step shape: a CHR-02a parser fixture."),
        s("! Its rate numbers are placeholders, not the paper's: the paper is unread."),
        s("ELEMENTS C H O N END"),
        s("SPECIES CH4 O2 CO H2 H2O CO2 N2 END"),
        s("THERMO"),
        s(TEMPS),
    ];
    push_records(&mut v, &["CH4", "O2", "CO", "H2", "H2O", "CO2", "N2"]);
    v.extend([
        s("END"),
        s("REACTIONS  KCAL/MOLE"),
        s("CH4 + 0.5 O2 => CO + 2 H2        1.0E+11  0.0  30.0"),
        s("  FORD /CH4 0.5/ FORD /O2 1.25/"),
        s("CH4+H2O=>CO+3H2                  1.0E+08  0.0  30.0   ! no spaces in the equation"),
        s("H2 + 0.5 O2 <=> H2O              1.0E+16  -1.0  40.0"),
        s("  FORD /H2 0.25/ FORD /O2 1.5/"),
        s("CO + H2O = CO2 + H2              1.0E+09  0.0  20.0"),
    ]);
    assert_eq!(v.len(), 42);
    v
}

/// T4's fixture: the reaction line is line 27.
fn t4_lines(equation: &str) -> Vec<String> {
    let mut v = vec![
        s("ELEMENTS H O N END"),
        s("SPECIES H2 O2 H2O N2 H END"),
        s("THERMO ALL"),
        s(TEMPS),
    ];
    push_records(&mut v, &["H2", "O2", "H2O", "N2", "H"]);
    v.extend([s("END"), s("REACTIONS"), format!("{equation}  1.0E+10  0.0  0.0")]);
    assert_eq!(v.len(), 27);
    v
}

/// T5's fixture up to and including the REACTIONS line (line 30); then `rest`.
fn t5_lines(reactions: &str, rest: &[&str]) -> Vec<String> {
    let mut v = vec![
        s("ELEMENTS H O N END"),
        s("SPECIES H2 O2 OH H HO2 N2 END"),
        s("THERMO ALL"),
        s(TEMPS),
    ];
    push_records(&mut v, &["H2", "O2", "OH", "H", "HO2", "N2"]);
    v.push(s("END"));
    v.push(s(reactions));
    assert_eq!(v.len(), 30);
    v.extend(rest.iter().map(|r| r.to_string()));
    v
}

const RX_A: &str = "H2 + O2 => 2 OH   1.7E+13  0.0  47780.0";
const RX_B: &str = "H + O2 + M => HO2 + M   2.0E+15  0.0  0.0";

// ==========================================================================
//  T1
// ==========================================================================

#[test]
fn ck_parses_westbrook_dryer() {
    let lines = wd_lines();
    let m = must(&lines);
    assert_eq!(names(&m), vec!["CH4", "O2", "CO2", "H2O", "N2"]);
    assert_eq!(
        m.elements().to_vec(),
        vec![(s("C"), 12.011), (s("H"), 1.00794), (s("O"), 15.9994), (s("N"), 14.00674)]
    );
    assert_eq!(m.source(), "t");
    assert_eq!(m.thermo().p_ref(), P_ATM);
    for (k, sp) in m.thermo().species().iter().enumerate() {
        assert_eq!(sp.name, m.species()[k]);
    }
    let record_lines: Vec<usize> = m.thermo().species().iter().map(|r| r.line).collect();
    assert_eq!(record_lines, vec![11, 15, 19, 23, 27]);

    assert_eq!(m.reactions().len(), 1);
    let r = &m.reactions()[0];
    assert_eq!(r.line, 33);
    assert_eq!(r.equation, "CH4 + 2 O2 => CO2 + 2 H2O");
    assert!(!r.reversible);
    assert_eq!(r.reactants, vec![(0, q(1, 1)), (1, q(2, 1))]);
    assert_eq!(r.products, vec![(2, q(1, 1)), (3, q(2, 1))]);
    assert_eq!(r.third_body, None);
    assert_eq!(r.ford, vec![(0, 0.25), (1, 1.5)]);
    assert!(r.rord.is_empty());
    assert_eq!(r.forward_orders(), vec![(0, 0.25), (1, 1.5)]);
    assert!(r.reverse_orders().is_empty());
    assert_eq!(r.molecularity(), 1.75);
    println!(
        "  [T1] WD1 shape: a = {:e} (truth {A_1E12:e}, rel {:.2e}), beta = {}, theta = {} \
         (truth {TH_30K}, rel {:.2e}), m = {}",
        r.rate.a,
        (r.rate.a - A_1E12).abs() / A_1E12,
        r.rate.beta,
        r.rate.theta,
        (r.rate.theta - TH_30K).abs() / TH_30K,
        r.molecularity()
    );
    assert!(within(r.rate.a, A_1E12), "a = {}", r.rate.a);
    assert_eq!(r.rate.beta, 0.0);
    assert!(within(r.rate.theta, TH_30K), "theta = {}", r.rate.theta);
    assert!(m.notes().is_empty(), "notes: {:?}", m.notes());

    // CRLF line ends read to an equal mechanism.
    let crlf = Mechanism::read_chemkin(&text(&lines).replace('\n', "\r\n"), "t", None).unwrap();
    assert_eq!(crlf, m);

    // A separate thermo set stands in for the in-file block: lines 9-31 deleted.
    let mut records = Vec::new();
    push_records(&mut records, &["CH4", "O2", "CO2", "H2O", "N2"]);
    let sep_text = format!("THERMO ALL\n{TEMPS}\n{}\nEND\n", text(&records));
    let sep = ThermoSet::read_chemkin(&sep_text, "sep", P_ATM, &ElementTable::standard()).unwrap();
    let mut without = lines.clone();
    without.drain(8..31);
    let m2 = Mechanism::read_chemkin(&text(&without), "t", Some(&sep)).unwrap();
    for k in 0..5 {
        assert_eq!(m2.thermo().species()[k].upper, m.thermo().species()[k].upper);
        assert_eq!(m2.thermo().species()[k].lower, m.thermo().species()[k].lower);
    }
    assert_eq!(m2.thermo().p_ref(), P_ATM);
    let mut expect = m.reactions()[0].clone();
    assert_eq!(m2.reactions()[0].line, 10);
    expect.line = 10;
    assert_eq!(m2.reactions()[0], expect);
}

// ==========================================================================
//  T2
// ==========================================================================

/// The fixture with line `no` replaced.
fn replace_line(mut v: Vec<String>, no: usize, with: &str) -> Vec<String> {
    v[no - 1] = s(with);
    v
}

#[test]
fn ck_parses_jones_lindstedt_shape() {
    let m = must(&jl_lines());
    assert_eq!(names(&m), vec!["CH4", "O2", "CO", "H2", "H2O", "CO2", "N2"]);
    assert_eq!(m.reactions().len(), 4);
    struct Row {
        line: usize,
        reversible: bool,
        reactants: Vec<(usize, Rational)>,
        products: Vec<(usize, Rational)>,
        forward: Vec<(usize, f64)>,
        m: f64,
        a: f64,
        beta: f64,
        theta: f64,
    }
    let rows = [
        Row {
            line: 37,
            reversible: false,
            reactants: vec![(0, q(1, 1)), (1, q(1, 2))],
            products: vec![(2, q(1, 1)), (3, q(2, 1))],
            forward: vec![(0, 0.5), (1, 1.25)],
            m: 1.75,
            a: A_1E11,
            beta: 0.0,
            theta: TH_30K,
        },
        Row {
            line: 39,
            reversible: false,
            reactants: vec![(0, q(1, 1)), (4, q(1, 1))],
            products: vec![(2, q(1, 1)), (3, q(3, 1))],
            forward: vec![(0, 1.0), (4, 1.0)],
            m: 2.0,
            a: 100.0,
            beta: 0.0,
            theta: TH_30K,
        },
        Row {
            line: 40,
            reversible: true,
            reactants: vec![(3, q(1, 1)), (1, q(1, 2))],
            products: vec![(4, q(1, 1))],
            forward: vec![(3, 0.25), (1, 1.5)],
            m: 1.75,
            a: A_1E16,
            beta: -1.0,
            theta: TH_40K,
        },
        Row {
            line: 42,
            reversible: true,
            reactants: vec![(2, q(1, 1)), (4, q(1, 1))],
            products: vec![(5, q(1, 1)), (3, q(1, 1))],
            forward: vec![(2, 1.0), (4, 1.0)],
            m: 2.0,
            a: 1000.0,
            beta: 0.0,
            theta: TH_20K,
        },
    ];
    for (i, row) in rows.iter().enumerate() {
        let r = &m.reactions()[i];
        println!(
            "  [T2] reaction {i} (line {}): a = {:e} (rel {:.2e}), beta = {}, theta = {} \
             (rel {:.2e}), m = {}",
            r.line,
            r.rate.a,
            (r.rate.a - row.a).abs() / row.a,
            r.rate.beta,
            r.rate.theta,
            (r.rate.theta - row.theta).abs() / row.theta,
            r.molecularity()
        );
        assert_eq!(r.line, row.line, "reaction {i}");
        assert_eq!(r.reversible, row.reversible, "reaction {i}");
        assert_eq!(r.reactants, row.reactants, "reaction {i}");
        assert_eq!(r.products, row.products, "reaction {i}");
        assert_eq!(r.forward_orders(), row.forward, "reaction {i}");
        assert_eq!(r.molecularity(), row.m, "reaction {i}");
        assert!(within(r.rate.a, row.a), "reaction {i}: a = {}", r.rate.a);
        assert_eq!(r.rate.beta, row.beta, "reaction {i}");
        assert!(within(r.rate.theta, row.theta), "reaction {i}: theta = {}", r.rate.theta);
    }
    assert_eq!(m.reactions()[1].equation, "CH4+H2O=>CO+3H2");
    assert_eq!(m.reactions()[2].reverse_orders(), vec![(4, 1.0)]);
    let o2 = m.reactions()[0].reactants[1].1;
    assert_eq!((o2.num(), o2.den(), o2.to_f64()), (1, 2, 0.5));
    // Reaction 2 is reversible with FORD; reaction 0 is irreversible; reaction 3 has no FORD.
    assert_eq!(m.notes().len(), 1, "notes: {:?}", m.notes());
    has(&m.notes()[0], &["line 40", "REV"]);

    // FORD and RORD refusals, each with the line of the replaced line.
    for (bad, why) in [
        ("  FORD /XX 1.0/", "an undeclared name"),
        ("  FORD /CH4 0.5/ FORD /CH4 0.7/", "a species given twice"),
        ("  FORD /CH4 -0.5/", "a negative order"),
        ("  RORD /CO 1.0/", "RORD on an irreversible reaction"),
    ] {
        let msg = refusal(&replace_line(jl_lines(), 38, bad));
        assert!(msg.contains("line 38"), "{why}: {msg}");
    }
    // FORD and RORD accepted.
    let m = must(&replace_line(jl_lines(), 41, "  RORD /H2O 0.75/"));
    assert_eq!(m.reactions()[2].reverse_orders(), vec![(4, 0.75)]);
    assert_eq!(m.notes().len(), 1, "notes: {:?}", m.notes());
    has(&m.notes()[0], &["line 40"]);
    let m = must(&replace_line(jl_lines(), 38, "  FORD /H2O 0.1/"));
    assert_eq!(m.reactions()[0].forward_orders(), vec![(0, 1.0), (1, 0.5), (4, 0.1)]);
    assert_eq!(m.reactions()[0].molecularity(), 1.6);
}

// ==========================================================================
//  T3
// ==========================================================================

#[test]
fn ck_refuses_unbalanced() {
    let msg = refusal(&replace_line(
        wd_lines(),
        33,
        "CH4 + 1.5 O2 => CO2 + 2 H2O     1.0E+12  0.0  30000.0",
    ));
    has(&msg, &["line 33", "element O: products minus reactants = 1 "]);
    assert!(!msg.contains("element H"), "{msg}");
    let msg = refusal(&replace_line(
        wd_lines(),
        33,
        "CH4 + 2 O2 => CO2 + H2O     1.0E+12  0.0  30000.0",
    ));
    // H comes before O in ELEMENTS order.
    has(&msg, &["line 33", "element H: products minus reactants = -2 "]);
}

// ==========================================================================
//  T4
// ==========================================================================

#[test]
fn ck_rational_balance_is_exact() {
    let ok = |eq: &str| must(&t4_lines(eq));
    let m = ok("H2 + 0.5 O2 => H2O");
    assert_eq!(m.reactions()[0].reactants[1], (1, Rational::new(1, 2).unwrap()));
    ok("2 H2 + O2 => 2 H2O");
    ok("H2 + .5 O2 => H2O");
    let m = ok("H2 + 0.50 O2 => H2O");
    assert_eq!(m.reactions()[0].reactants[1].1, q(1, 2));
    let m = ok("1.5 H2 + 0.75 O2 => 1.5 H2O");
    let r = &m.reactions()[0];
    assert_eq!((r.reactants[0].1, r.reactants[1].1, r.products[0].1), (q(3, 2), q(3, 4), q(3, 2)));
    let m = ok("H + H + N2 => H2 + N2");
    assert_eq!(m.reactions()[0].reactants, vec![(4, q(2, 1)), (3, q(1, 1))]);

    let bad = |eq: &str| refusal(&t4_lines(eq));
    let msg = bad("H2 + 0.49 O2 => H2O");
    has(&msg, &["line 27", "element O: products minus reactants = 1/50 "]);
    let msg = bad("0.5 H2 + 0.5 O2 => H2O");
    has(&msg, &["line 27", "element H: products minus reactants = 1 "]);
    has(&bad("H2 + 0 O2 => H2O"), &["line 27", "coefficient"]);
    has(&bad("H2 + O2 => H2O + XY"), &["line 27", "XY"]);
    has(&bad("H2 + 0.5 O2 <= H2O"), &["line 27"]);

    assert_eq!(Rational::parse_decimal("0.125"), Rational::new(1, 8));
    assert_eq!(Rational::parse_decimal("1."), Rational::new(1, 1));
    assert_eq!(Rational::parse_decimal(".5"), Rational::new(1, 2));
    assert_eq!(Rational::parse_decimal("0.49"), Rational::new(49, 100));
    assert_eq!(Rational::parse_decimal("0.50"), Rational::new(1, 2));
    for none in ["", ".", "1.2.3", "-1", "1e2", "1234567890123456"] {
        assert_eq!(Rational::parse_decimal(none), None, "{none:?}");
    }
    assert_eq!(Rational::new(1, 0), None);
    assert_eq!(Rational::new(4, -6), Rational::new(-2, 3));
    assert_eq!(Rational::new(-3, 6).unwrap().to_string(), "-1/2");
    assert_eq!(Rational::new(4, 2).unwrap().to_string(), "2");
}

// ==========================================================================
//  T5
// ==========================================================================

/// Reads the T5 fixture with one REACTIONS line and the reaction lines, and returns the
/// mechanism.
fn units(reactions: &str, rest: &[&str]) -> Mechanism {
    must(&t5_lines(reactions, rest))
}

fn rate0(m: &Mechanism) -> Arrhenius {
    m.reactions()[0].rate
}

/// Prints one case and checks `a` and `theta` against their truth.
fn check_units(case: &str, m: &Mechanism, a_truth: f64, theta_truth: f64) {
    let r = rate0(m);
    // A truth of zero has no relative error to print; its error is the value itself.
    let theta_err = (r.theta - theta_truth).abs() / theta_truth.abs().max(1.0);
    println!(
        "  [T5] case {case}: a = {:e} (rel {:.2e}), theta = {} (error {:.2e})",
        r.a,
        (r.a - a_truth).abs() / a_truth,
        r.theta,
        theta_err
    );
    assert!(within(r.a, a_truth), "case {case}: a = {}", r.a);
    assert!(within(r.theta, theta_truth), "case {case}: theta = {}", r.theta);
}

#[test]
fn ck_units_keywords() {
    let a_truth = 1.7e7;
    let base = units("REACTIONS", &[RX_A]);
    check_units("base", &base, a_truth, TH_47780);
    assert_eq!(base.reactions()[0].molecularity(), 2.0);

    // 1, 2: the defaults spelled out are bitwise the default.
    assert_eq!(rate0(&units("REACTIONS CAL/MOLE", &[RX_A])), rate0(&base));
    assert_eq!(rate0(&units("REACTIONS MOLES", &[RX_A])), rate0(&base));
    // 3 to 7: every energy unit, one reaction.
    let m = units("REACTIONS KCAL/MOLE", &["H2 + O2 => 2 OH   1.7E+13  0.0  47.78"]);
    check_units("3 KCAL/MOLE", &m, a_truth, TH_47780);
    let m = units("REACTIONS JOULES/MOLE", &["H2 + O2 => 2 OH   1.7E+13  0.0  199911.52"]);
    check_units("4 JOULES/MOLE", &m, a_truth, TH_47780);
    let m = units("REACTIONS KJOULES/MOLE", &["H2 + O2 => 2 OH   1.7E+13  0.0  199.91152"]);
    check_units("5 KJOULES/MOLE", &m, a_truth, TH_47780);
    assert_eq!(m.notes().len(), 1, "notes: {:?}", m.notes());
    has(&m.notes()[0], &["KJOULES/MOLE", "extension"]);
    let m = units("REACTIONS KELVINS", &["H2 + O2 => 2 OH   1.7E+13  0.0  24043.829311014168"]);
    check_units("6 KELVINS", &m, a_truth, TH_47780);
    let m = units("REACTIONS EVOLTS", &["H2 + O2 => 2 OH   1.7E+13  0.0  2.0719369007114357"]);
    check_units("7 EVOLTS", &m, a_truth, TH_47780);
    // 8, 9: A in molecules.
    let m = units("REACTIONS MOLECULES", &["H2 + O2 => 2 OH   2.8229164141955394E-11  0.0  47780.0"]);
    check_units("8 MOLECULES", &m, a_truth, TH_47780);
    let m = units(
        "REACTIONS MOLECULES KCAL/MOLE",
        &["H2 + O2 => 2 OH   2.8229164141955394E-11  0.0  47.78"],
    );
    check_units("9 MOLECULES KCAL/MOLE", &m, a_truth, TH_47780);
    // 10, 11: a third body adds one to the molecularity.
    let m = units("REACTIONS", &[RX_B]);
    check_units("10 +M", &m, 2000.0, 0.0);
    assert_eq!(rate0(&m).theta, 0.0);
    assert_eq!(m.reactions()[0].molecularity(), 3.0);
    let m = units("REACTIONS MOLECULES", &["H + O2 + M => HO2 + M   5.5147799872211776E-33  0.0  0.0"]);
    check_units("11 +M MOLECULES", &m, 2000.0, 0.0);
    // 12 to 16: per-reaction UNITS.
    let m = units("REACTIONS", &["H2 + O2 => 2 OH   1.7E+13  0.0  47.78", "  UNITS /KCAL/"]);
    check_units("12 UNITS KCAL", &m, a_truth, TH_47780);
    let m = units(
        "REACTIONS",
        &["H2 + O2 => 2 OH   1.7E+13  0.0  199.91152", "  UNITS /KJOU/"],
    );
    check_units("13 UNITS KJOU", &m, a_truth, TH_47780);
    assert!(m.notes().is_empty(), "notes: {:?}", m.notes());
    let m = units(
        "REACTIONS",
        &["H2 + O2 => 2 OH   1.7E+13  0.0  199911.52", "  UNITS /JOUL/"],
    );
    check_units("14 UNITS JOUL", &m, a_truth, TH_47780);
    for d in ["KELV", "KELVIN"] {
        let m = units(
            "REACTIONS",
            &["H2 + O2 => 2 OH   1.7E+13  0.0  24043.829311014168", &format!("  UNITS /{d}/")],
        );
        check_units(&format!("15 UNITS {d}"), &m, a_truth, TH_47780);
    }
    for d in ["EVOL", "EVOLTS"] {
        let m = units(
            "REACTIONS",
            &["H2 + O2 => 2 OH   1.7E+13  0.0  2.0719369007114357", &format!("  UNITS /{d}/")],
        );
        check_units(&format!("16 UNITS {d}"), &m, a_truth, TH_47780);
    }
    // 17: the override wins over the block default.
    let m = units("REACTIONS KELVINS", &[RX_A, "  UNITS /CAL/"]);
    check_units("17 override", &m, a_truth, TH_47780);
    // 18, 19: MOLE means MOLE(CULE).
    let m = units(
        "REACTIONS",
        &["H2 + O2 => 2 OH   2.8229164141955394E-11  0.0  47780.0", "  UNITS /MOLE/"],
    );
    check_units("18 UNITS MOLE", &m, a_truth, TH_47780);
    let m = units(
        "REACTIONS",
        &["H2 + O2 => 2 OH   2.8229164141955394E-11  0.0  47.78", "  UNITS /MOLECULE KCAL/"],
    );
    check_units("19 UNITS MOLECULE KCAL", &m, a_truth, TH_47780);
    // 20: the override does not leak into the next reaction.
    let m = units(
        "REACTIONS",
        &[
            "H2 + O2 => 2 OH   1.7E+13  0.0  47.78",
            "  UNITS /KCAL/",
            "2 OH => H2 + O2   1.7E+13  0.0  47780.0",
        ],
    );
    assert_eq!(m.reactions().len(), 2);
    let (r0, r1) = (m.reactions()[0].rate, m.reactions()[1].rate);
    println!("  [T5] case 20: theta {} then {} (truth {TH_47780})", r0.theta, r1.theta);
    assert!(within(r0.theta, TH_47780) && within(r1.theta, TH_47780));
    assert!(within(r1.a, a_truth));

    // The refusals each name the offending line.
    for (reactions, rest, line, word) in [
        ("REACTIONS KCAL/MOLE JOULES/MOLE", vec![], 30, "JOULES/MOLE"),
        ("REACTIONS MOLES MOLECULES", vec![], 30, "MOLECULES"),
        ("REACTIONS FURLONGS", vec![], 30, "FURLONGS"),
        ("REACTIONS", vec![RX_A, "  UNITS /FURLONG/"], 32, "FURLONG"),
        ("REACTIONS", vec![RX_A, "  UNITS /KCAL JOUL/"], 32, "JOUL"),
        ("REACTIONS", vec![RX_A, "  UNITS"], 32, "UNITS"),
    ] {
        let msg = refusal(&t5_lines(reactions, &rest));
        let at = format!("line {line}");
        has(&msg, &[at.as_str(), word]);
    }
}

// ==========================================================================
//  T6
// ==========================================================================

/// The base fixture with reaction A on line 31 and `aux` on line 32.
fn with_aux(aux: &str) -> Vec<String> {
    t5_lines("REACTIONS", &[RX_A, aux])
}

#[test]
fn ck_refuses_unknown_keyword() {
    for (aux, key) in [
        ("  HIGH / 1.0E+10 0.0 0.0 /", "HIGH"),
        ("  LT / 1.0 2.0 /", "LT"),
        ("  RLT / 1.0 2.0 /", "RLT"),
        ("  JAN / 1 2 3 4 5 6 7 8 9 /", "JAN"),
        ("  FIT1 / 1 2 3 4 /", "FIT1"),
        ("  HV / 2537.0 /", "HV"),
        ("  TDEP / O2 /", "TDEP"),
        ("  EXCI / 1.0 /", "EXCI"),
        ("  MOME", "MOME"),
        ("  XSMI", "XSMI"),
        ("  PLOG / 1.0 1.0E+13 0.0 0.0 /", "PLOG"),
        ("  CHEB / 1 2 /", "CHEB"),
        ("  plog / 1.0 1.0E+13 0.0 0.0 /", "PLOG"),
    ] {
        let msg = refusal(&with_aux(aux));
        has(&msg, &["line 32", key]);
    }
    for (aux, key) in [
        ("  LOW / 1.0E+16 0.0 0.0 /", "LOW"),
        ("  TROE / 0.5 100.0 1000.0 /", "TROE"),
        ("  SRI / 0.5 100.0 1000.0 /", "SRI"),
        ("  REV / 1.0E+12 0.0 0.0 /", "REV"),
        ("  DUP", "DUP"),
        ("  DUPLICATE", "DUPLICATE"),
    ] {
        let msg = refusal(&with_aux(aux));
        has(&msg, &["line 32", key, "not read yet"]);
    }
    has(&refusal(&with_aux("  FOO / 1.0 /")), &["line 32", "FOO", "FORD", "RORD", "UNITS", "LOW"]);
    has(&refusal(&with_aux("  FORD")), &["line 32", "FORD"]);
    // An auxiliary line before any reaction.
    has(&refusal(&t5_lines("REACTIONS", &["N2/2.0/"])), &["line 31"]);

    // Equation-level refusals, each on the reaction line (line 31).
    let on_line_31 = |rx: &str| refusal(&t5_lines("REACTIONS", &[rx]));
    has(&on_line_31("H + O2 (+M) => HO2 (+M)   1.0E+12 0.0 0.0"), &["line 31", "fall-off", "not read yet"]);
    has(
        &on_line_31("H + O2 (+N2) => HO2 (+N2)   1.0E+12 0.0 0.0"),
        &["line 31", "fall-off", "not read yet"],
    );
    has(&on_line_31("O2 + HV => O2   1.0 0.0 0.0"), &["line 31", "HV", "photon"]);
    has(&on_line_31("H2 + O2 => 2 OH   1.7E+13 0.0"), &["line 31", "three numbers"]);
    has(&on_line_31("H2 + O2 => 2 OH   1.7E+13 0.0 abc"), &["line 31", "three numbers"]);
}

// ==========================================================================
//  T7
// ==========================================================================

#[test]
fn ck_third_body_and_efficiencies() {
    let m = units("REACTIONS", &[RX_B, "  H2/2.5/ N2/ 1.0/  O2/0.4/"]);
    assert_eq!(
        m.reactions()[0].third_body,
        Some(ThirdBody { efficiencies: vec![(0, 2.5), (5, 1.0), (1, 0.4)] })
    );
    for eq in [
        "H + O2 + M => HO2   2.0E+15 0.0 0.0",
        "H + O2 => HO2 + M   2.0E+15 0.0 0.0",
        "H + O2 + m => HO2 + m   2.0E+15 0.0 0.0",
    ] {
        let m = units("REACTIONS", &[eq]);
        assert_eq!(m.reactions()[0].third_body, Some(ThirdBody { efficiencies: vec![] }), "{eq}");
    }
    for eq in [
        "H + O2 + M + M => HO2 + M   2.0E+15 0.0 0.0",
        "H + O2 + 2M => HO2 + 2M   2.0E+15 0.0 0.0",
    ] {
        has(&refusal(&t5_lines("REACTIONS", &[eq])), &["line 31"]);
    }
    has(&refusal(&t5_lines("REACTIONS", &[RX_A, "N2/2.0/"])), &["line 32", "N2"]);
    for aux in ["N2/2.0/ N2/3.0/", "N2/-1.0/", "N2/abc/", "N2/"] {
        has(&refusal(&t5_lines("REACTIONS", &[RX_B, aux])), &["line 32"]);
    }
    has(&refusal(&t5_lines("REACTIONS", &[RX_B, "AR/0.7/"])), &["line 32", "AR"]);
}

// ==========================================================================
//  T8
// ==========================================================================

/// T4's fixture with the first line replaced (the ELEMENTS line).
fn elements_line(l: &str) -> Vec<String> {
    replace_line(t4_lines("H2 + 0.5 O2 => H2O"), 1, l)
}

/// T4's fixture with the SPECIES line replaced.
fn species_line(l: &str) -> Vec<String> {
    replace_line(t4_lines("H2 + 0.5 O2 => H2O"), 2, l)
}

#[test]
fn ck_blocks_and_declarations() {
    let base = t4_lines("H2 + 0.5 O2 => H2O");
    // Blocks.
    let mut swapped = base.clone();
    swapped.swap(0, 1);
    has(&refusal(&swapped), &["line 1"]);
    let mut no_elements = base.clone();
    no_elements.remove(0);
    has(&refusal(&no_elements), &["end of text", "ELEMENTS"]);
    let mut no_species = base.clone();
    no_species.remove(1);
    has(&refusal(&no_species), &["SPECIES"]);
    let mut no_reactions = base.clone();
    no_reactions.truncate(25);
    assert_eq!(must(&no_reactions).reactions().len(), 0);
    no_reactions.push(s("REACTIONS"));
    no_reactions.push(s("END"));
    assert_eq!(must(&no_reactions).reactions().len(), 0);
    let mut hello = base.clone();
    hello.insert(0, s("hello"));
    has(&refusal(&hello), &["line 1"]);
    let mut after_end = base.clone();
    after_end.push(s("END"));
    after_end.push(s("H2 + 0.5 O2 => H2O  1.0E+10  0.0  0.0"));
    has(&refusal(&after_end), &["line 29"]);
    let mut repeated = base.clone();
    repeated.push(s("REACTIONS"));
    has(&refusal(&repeated), &["line 28"]);

    // Elements.
    let msg = refusal(&elements_line("ELEMENTS H O N XE END"));
    has(&msg, &["line 1", "XE", "no atomic weight"]);
    let m = must(&elements_line("ELEMENTS H O N XE /131.293/ D/2.014/ END"));
    let n = m.elements().len();
    assert_eq!(m.elements()[n - 2..].to_vec(), vec![(s("XE"), 131.293), (s("D"), 2.014)]);
    has(&refusal(&elements_line("ELEMENTS H O N E END")), &["line 1", "electron"]);
    has(&refusal(&elements_line("ELEMENTS H O N ABC END")), &["line 1", "ABC"]);
    let m = must(&elements_line("ELEMENTS H O N H END"));
    assert_eq!(m.elements().len(), 3);
    assert_eq!(m.notes().len(), 1, "notes: {:?}", m.notes());
    has(&m.notes()[0], &["element H"]);

    // Species.
    let m = must(&species_line("SPECIES H2 O2 H2O N2 H H2 END"));
    assert_eq!(m.species().len(), 5);
    assert_eq!(m.notes().len(), 1, "notes: {:?}", m.notes());
    has(&m.notes()[0], &["H2"]);
    for (name, word) in [
        ("2H", "2H"),
        ("OH+", "ion"),
        ("E", "electron"),
        ("M", "M"),
        ("ABCDEFGHIJKLMNOPQ", "ABCDEFGHIJKLMNOPQ"),
    ] {
        let msg = refusal(&species_line(&format!("SPECIES H2 O2 H2O N2 H {name} END")));
        has(&msg, &["line 2", word]);
    }
    has(&refusal(&species_line("SPECIES H2 O2 H2O N2 H HO2 END")), &["line 2", "HO2", "thermo"]);
    has(&refusal(&elements_line("ELEMENTS H O END")), &["line 2", "N2", "element N"]);

    // A separate set, its p_ref, and the in-file override.
    let mut recs = Vec::new();
    push_records(&mut recs, &["H2", "O2", "H2O", "N2", "H"]);
    recs.extend(rec_lines("CO2"));
    let sep_text = format!("THERMO ALL\n{TEMPS}\n{}\nEND\n", text(&recs));
    let sep = ThermoSet::read_chemkin(&sep_text, "sep", P_BAR, &ElementTable::standard()).unwrap();
    let mut no_thermo = base.clone();
    no_thermo.drain(2..25);
    assert_eq!(no_thermo.len(), 4);
    let m = Mechanism::read_chemkin(&text(&no_thermo), "t", Some(&sep)).unwrap();
    assert_eq!(m.thermo().p_ref(), P_ATM);
    assert_eq!(m.thermo().species().len(), 5);
    let h2 = m.thermo().get("H2").unwrap();
    let want = sep.get("H2").unwrap().upper[6] - (P_ATM / P_BAR).ln();
    assert!((h2.upper[6] - want).abs() <= 1e-15, "{} against {want}", h2.upper[6]);
    // A THERMO block holding only O2's record, with a1 = 3.5, overrides the set.
    let mut with_block = no_thermo.clone();
    let mut block = vec![s("THERMO"), s(TEMPS)];
    block.extend(rec("O2", elems_of("O2"), 3.5).split('\n').map(str::to_string));
    block.push(s("END"));
    for (k, l) in block.into_iter().enumerate() {
        with_block.insert(2 + k, l);
    }
    let m = Mechanism::read_chemkin(&text(&with_block), "t", Some(&sep)).unwrap();
    assert_eq!(m.thermo().get("O2").unwrap().upper[0], 3.5);
    assert_eq!(m.thermo().get("H2").unwrap().upper[0], 2.5);

    // Line numbers through the thermo reader: CO2's card 3 is line 21 of T1's fixture.
    let mut broken = wd_lines();
    broken[20].replace_range(79..80, "5");
    has(&refusal(&broken), &["line 21", "CO2"]);

    // ThermoSet::from_species.
    let one = sep.get("H2").unwrap().clone();
    let msg = ThermoSet::from_species(vec![one.clone(), one.clone()], P_ATM, "x")
        .unwrap_err()
        .to_string();
    has(&msg, &["H2"]);
    assert!(ThermoSet::from_species(vec![one.clone()], 0.0, "x").is_err());
    let set = ThermoSet::from_species(vec![one], P_ATM, "x").unwrap();
    assert_eq!(set.species().len(), 1);
    assert!(set.notes().is_empty());
}

// ==========================================================================
//  T9
// ==========================================================================

#[test]
fn ck_duplicates_are_refused() {
    for (first, second) in [
        ("H2 + O2 => 2 OH", "H2 + O2 => 2 OH"),
        ("H2 + O2 => 2 OH", "O2 + H2 => OH + OH"),
        ("H2 + O2 = 2 OH", "2 OH = H2 + O2"),
        ("H2 + O2 = 2 OH", "2 OH => H2 + O2"),
    ] {
        let rx = |eq: &str| format!("{eq}   1.7E+13  0.0  47780.0");
        let msg = refusal(&t5_lines("REACTIONS", &[&rx(first), &rx(second)]));
        assert!(msg.contains("line 32") && msg.contains("line 31"), "{first} / {second}: {msg}");
    }
    let rx = |eq: &str| format!("{eq}   1.7E+13  0.0  47780.0");
    // Two irreversible reactions in opposite directions.
    let m = units("REACTIONS", &[&rx("H2 + O2 => 2 OH"), &rx("2 OH => H2 + O2")]);
    assert_eq!(m.reactions().len(), 2);
    // The third body differs.
    let m = units("REACTIONS", &[RX_B, "H + O2 => HO2   2.0E+15  0.0  0.0"]);
    assert_eq!(m.reactions().len(), 2);
}
