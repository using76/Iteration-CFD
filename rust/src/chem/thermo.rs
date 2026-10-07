// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.
// Provenance: see PROVENANCE.md. No GPL-licensed source was consulted.

//! NASA seven-coefficient species thermochemistry read from a CHEMKIN
//! THERMO block by fixed columns, and the ideal-gas mixture built on it.
//! SPEC-LIT §128 is the contract; the record layout is §129.3.
//!
//! Written from:
//!   SPEC-LIT §128 and §129 - the polynomials, the two-interval choice and
//!     the refusal to extrapolate of §128.1, the element table of §128.2,
//!     the mixture of §128.3, the standard pressure a set carries of
//!     §128.4, and the fixed-column record of §129.3
//!   B. J. McBride, S. Gordon & M. A. Reno, NASA TM-4513 (1993),
//!     NTRS 19940013151 - eqs. (1)-(3), the seven-coefficient form with its
//!     two integration constants, and Table II, whose records are the test
//!     data of [`crate::chem::tests_thermo`]
//!   R. J. Kee, F. M. Rupley & J. A. Miller, *Chemkin-II*, Sandia
//!     SAND89-8009 (1989), DOI 10.2172/5681118 - eqs. (3), (6), (9),
//!     (19)-(21), (34), (38), (41) and Table III: the mixture relations and
//!     the THERMO record format, read as a format and not as code
//!   CODATA 2018 - the gas constant [`R_GAS`]
//! No GPL-licensed source was consulted.
//!
//! Host only, and f64 everywhere: no `Scalar`, no device types, so the
//! module and its tests behave identically in the default build and in
//! `--features single`.

use crate::error::Error;

/// J/(mol K), CODATA 2018, exact (SPEC-LIT §128.5).
pub const R_GAS: f64 = 8.314462618;
/// 1 atm in Pa: the standard pressure of a CHEMKIN-format file (SPEC-LIT §128.4).
pub const P_ATM: f64 = 101325.0;
/// 1 bar in Pa: the standard pressure of TM-4513 and TP-2002 (SPEC-LIT §128.4).
pub const P_BAR: f64 = 100000.0;

// ==========================================================================
//  The element table
// ==========================================================================

/// The atomic weights a molar mass is built from, g/mol (SPEC-LIT §128.2).
///
/// Nothing else is built in: an isotope or a non-listed element must be
/// given its weight before a record that names it can be read.
#[derive(Clone, Debug, PartialEq)]
pub struct ElementTable {
    weights: Vec<(String, f64)>,
}

impl ElementTable {
    /// The six elements of the §128.2 table: H, C, N, O, AR, HE.
    pub fn standard() -> ElementTable {
        ElementTable {
            weights: vec![
                ("H".to_string(), 1.00794),
                ("C".to_string(), 12.011),
                ("N".to_string(), 14.00674),
                ("O".to_string(), 15.9994),
                ("AR".to_string(), 39.948),
                ("HE".to_string(), 4.002602),
            ],
        }
    }

    /// Adds or overrides one element (an ELEMENTS-block weight such as `D /2.014/`).
    /// Refused (Error::Config naming the symbol and the value): a symbol that is not one or
    /// two ASCII letters, or a weight that is not finite and > 0.
    pub fn with(mut self, symbol: &str, weight_g_per_mol: f64) -> crate::Result<ElementTable> {
        let bytes = symbol.as_bytes();
        let named = (1..=2).contains(&bytes.len())
            && bytes.iter().all(|b| b.is_ascii_alphabetic());
        if !named {
            return Err(Error::Config(format!(
                "element symbol {symbol:?} is not one or two ASCII letters; \
                 its weight was given as {weight_g_per_mol} g/mol"
            )));
        }
        if !weight_g_per_mol.is_finite() || weight_g_per_mol <= 0.0 {
            return Err(Error::Config(format!(
                "element {symbol:?} was given the weight {weight_g_per_mol} g/mol, \
                 which is not a finite weight above zero"
            )));
        }
        let sym = symbol.to_ascii_uppercase();
        match self.weights.iter_mut().find(|(s, _)| *s == sym) {
            Some(entry) => entry.1 = weight_g_per_mol,
            None => self.weights.push((sym, weight_g_per_mol)),
        }
        Ok(self)
    }

    /// g/mol; the symbol is matched case-insensitively ("Ar" == "AR").
    pub fn weight(&self, symbol: &str) -> Option<f64> {
        let sym = symbol.to_ascii_uppercase();
        self.weights.iter().find(|(s, _)| *s == sym).map(|(_, w)| *w)
    }
}

// ==========================================================================
//  One species
// ==========================================================================

/// Which of the two coefficient sets (128.5) evaluates.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Interval {
    /// The set valid on `[t_low, t_mid]`.
    Lower,
    /// The set valid on `[t_mid, t_high]`.
    Upper,
}

/// One species' record (SPEC-LIT §128.1, layout §129.3).
#[derive(Clone, Debug, PartialEq)]
pub struct SpeciesThermo {
    pub name: String,
    /// Upper-case element symbols with their counts, in record order; zero counts dropped.
    pub elements: Vec<(String, u32)>,
    /// kg/mol, built from `elements` and the ElementTable (= g/mol / 1000).
    pub molar_mass: f64,
    pub t_low: f64,
    pub t_mid: f64,
    pub t_high: f64,
    /// a1..a7 of the interval [t_mid, t_high] (the record's FIRST seven numbers).
    pub upper: [f64; 7],
    /// a1..a7 of the interval [t_low, t_mid].
    pub lower: [f64; 7],
    /// 1-based line of the record's card 1 in the text it was read from.
    pub line: usize,
}

impl SpeciesThermo {
    /// (128.5). Err (Error::Config) when T is not finite or outside [t_low, t_high]; the
    /// message names the species, T and the range, see "Refusal messages".
    pub fn interval(&self, t: f64) -> crate::Result<Interval> {
        if !t.is_finite() || t < self.t_low || t > self.t_high {
            return Err(Error::Config(format!(
                "species {}: T = {t} K lies outside its range [{}, {}] K; \
                 a thermo curve is not extrapolated (SPEC-LIT §128.1)",
                self.name, self.t_low, self.t_high
            )));
        }
        Ok(if t <= self.t_mid { Interval::Lower } else { Interval::Upper })
    }

    /// a1..a7 of the interval's set.
    pub fn coeffs(&self, i: Interval) -> &[f64; 7] {
        match i {
            Interval::Lower => &self.lower,
            Interval::Upper => &self.upper,
        }
    }

    /// (128.1), dimensionless.
    pub fn cp_r(&self, t: f64) -> crate::Result<f64> {
        Ok(nasa7_cp_r(self.coeffs(self.interval(t)?), t))
    }

    /// (128.2), dimensionless.
    pub fn h_rt(&self, t: f64) -> crate::Result<f64> {
        Ok(nasa7_h_rt(self.coeffs(self.interval(t)?), t))
    }

    /// (128.3), dimensionless.
    pub fn s_r(&self, t: f64) -> crate::Result<f64> {
        Ok(nasa7_s_r(self.coeffs(self.interval(t)?), t))
    }

    /// (128.4), dimensionless.
    pub fn g_rt(&self, t: f64) -> crate::Result<f64> {
        Ok(nasa7_g_rt(self.coeffs(self.interval(t)?), t))
    }

    /// J/(mol K) = R_GAS * cp_r
    pub fn cp(&self, t: f64) -> crate::Result<f64> {
        Ok(R_GAS * self.cp_r(t)?)
    }

    /// J/mol    = R_GAS * t * h_rt
    pub fn h(&self, t: f64) -> crate::Result<f64> {
        Ok(R_GAS * t * self.h_rt(t)?)
    }

    /// J/(mol K) = R_GAS * s_r
    pub fn s(&self, t: f64) -> crate::Result<f64> {
        Ok(R_GAS * self.s_r(t)?)
    }

    /// J/mol    = R_GAS * t * g_rt
    pub fn g(&self, t: f64) -> crate::Result<f64> {
        Ok(R_GAS * t * self.g_rt(t)?)
    }

    /// cp - R_GAS, J/(mol K).
    pub fn cv(&self, t: f64) -> crate::Result<f64> {
        Ok(self.cp(t)? - R_GAS)
    }

    /// h - R_GAS * t, J/mol.
    pub fn u(&self, t: f64) -> crate::Result<f64> {
        Ok(self.h(t)? - R_GAS * t)
    }
}

// ==========================================================================
//  The raw polynomials
// ==========================================================================

/// (128.1): `cp/R = a1 + a2 T + a3 T^2 + a4 T^3 + a5 T^4`, no range check.
pub fn nasa7_cp_r(a: &[f64; 7], t: f64) -> f64 {
    a[0] + t * (a[1] + t * (a[2] + t * (a[3] + t * a[4])))
}

/// (128.2): `H/(R T) = a1 + a2 T/2 + a3 T^2/3 + a4 T^3/4 + a5 T^4/5 + a6/T`, no range check.
pub fn nasa7_h_rt(a: &[f64; 7], t: f64) -> f64 {
    let t2 = t * t;
    a[0] + a[1] * t / 2.0 + a[2] * t2 / 3.0 + a[3] * t2 * t / 4.0 + a[4] * t2 * t2 / 5.0
        + a[5] / t
}

/// (128.3): `S/R = a1 ln T + a2 T + a3 T^2/2 + a4 T^3/3 + a5 T^4/4 + a7`, no range check.
pub fn nasa7_s_r(a: &[f64; 7], t: f64) -> f64 {
    let t2 = t * t;
    a[0] * t.ln() + a[1] * t + a[2] * t2 / 2.0 + a[3] * t2 * t / 3.0 + a[4] * t2 * t2 / 4.0
        + a[6]
}

/// (128.4): `G/(R T) = H/(R T) - S/R`, no range check.
pub fn nasa7_g_rt(a: &[f64; 7], t: f64) -> f64 {
    nasa7_h_rt(a, t) - nasa7_s_r(a, t)
}

// ==========================================================================
//  The set, and the reader
// ==========================================================================

/// A species database with the standard pressure its numbers are tabulated at.
#[derive(Clone, Debug, PartialEq)]
pub struct ThermoSet {
    species: Vec<SpeciesThermo>,
    p_ref: f64,
    source: String,
    notes: Vec<String>,
}

/// The 1-based inclusive columns `a..=b` of `line`, as bytes: columns are
/// fixed columns, never a whitespace split, because two E15 fields touch
/// when the second is negative. A line shorter than the range pads with
/// spaces on the right, so a caller that cares about width checks it first.
fn columns(line: &str, a: usize, b: usize) -> &str {
    let start = a - 1;
    if start >= line.len() {
        return "";
    }
    &line[start..b.min(line.len())]
}

/// One E-format field: trimmed, a Fortran `D` exponent read as `E`, finite.
fn parse_e(field: &str) -> Option<f64> {
    let s = field.trim();
    if s.is_empty() {
        return None;
    }
    s.replace(['D', 'd'], "E").parse::<f64>().ok().filter(|v| v.is_finite())
}

/// A reader failure: `path` is the `source` argument, `msg` starts `line N: `
/// with N the 1-based line within `text`.
fn parse_err(source: &str, line: usize, msg: String) -> Error {
    Error::Parse { path: source.to_string(), msg: format!("line {line}: {msg}") }
}

/// The shape every record line must have before its columns are read.
fn check_line_shape(l: &str) -> Result<(), String> {
    if l.len() < 80 {
        return Err(format!(
            "the line reads \"{}\" and is {} characters; a THERMO record line is 80 \
             characters wide with the card number in column 80",
            l.trim_end(),
            l.len()
        ));
    }
    if !l.is_ascii() {
        return Err("the line holds a non-ASCII character; a THERMO record line is ASCII"
            .to_string());
    }
    if l.contains('\t') {
        return Err(
            "the line holds a tab character; a THERMO record line pads its fixed columns \
             with spaces"
                .to_string(),
        );
    }
    Ok(())
}

fn first_word(l: &str) -> &str {
    l.split_whitespace().next().unwrap_or("")
}

/// A card 1: a line of at least 80 characters whose column 80 is `1`.
fn is_card_1(l: &str) -> bool {
    l.len() >= 80 && l.is_ascii() && columns(l, 80, 80) == "1"
}

/// The end of the THERMO block (§129.1's four blocks): END, or the next
/// block keyword REAC, SPEC or ELEM. Everything after is ignored. A card 1
/// is never a block end, so a species named SPECX or ELEM2 still reads.
fn is_block_end(l: &str) -> bool {
    if is_card_1(l) {
        return false;
    }
    let t = first_word(l).to_ascii_uppercase();
    t == "END" || t.starts_with("REAC") || t.starts_with("SPEC") || t.starts_with("ELEM")
}

/// One 15-character coefficient field of a coefficient card, parsed.
fn read_coeff(
    source: &str,
    name: &str,
    card: char,
    card_line: &str,
    card_no: usize,
    field: usize,
) -> crate::Result<f64> {
    let a = 15 * field + 1;
    let b = a + 14;
    let f = columns(card_line, a, b);
    match parse_e(f) {
        Some(v) => Ok(v),
        None => Err(parse_err(
            source,
            card_no,
            format!(
                "species {name}: card {card} columns {a}-{b} read \"{f}\", which does not \
                 read as a finite coefficient"
            ),
        )),
    }
}

impl ThermoSet {
    /// Reads one THERMO block. `source` names the text in every error (a file
    /// path, or a label). `p_ref` (Pa, finite, > 0) is the standard pressure the numbers in
    /// `text` are tabulated at: P_ATM for a CHEMKIN file by format, P_BAR for a TM-4513 or
    /// TP-2002 transcription (SPEC-LIT §128.4).
    pub fn read_chemkin(
        text: &str,
        source: &str,
        p_ref: f64,
        elements: &ElementTable,
    ) -> crate::Result<ThermoSet> {
        if !p_ref.is_finite() || p_ref <= 0.0 {
            return Err(Error::Config(format!(
                "the thermo set is read with the standard pressure p_ref = {p_ref} Pa, \
                 which is not a finite pressure above zero (SPEC-LIT §128.4)"
            )));
        }

        // The significant lines: blank, whitespace-only and `!`-comment lines
        // are skipped, and `str::lines` strips one trailing `\r`, so CRLF and
        // LF read the same.
        let sig: Vec<(usize, &str)> = text
            .lines()
            .enumerate()
            .map(|(i, l)| (i + 1, l))
            .filter(|(_, l)| {
                let t = l.trim_start();
                !(t.is_empty() || t.starts_with('!'))
            })
            .collect();

        let Some(&(kw_no, kw_line)) = sig.first() else {
            return Err(parse_err(source, 1, "the text holds no THERMO keyword line".into()));
        };
        let tokens: Vec<String> =
            kw_line.split_whitespace().map(str::to_ascii_uppercase).collect();
        let keyword_ok = match tokens.as_slice() {
            [t] => t == "THERMO" || t == "THER",
            [t, u] => u == "ALL" && (t == "THERMO" || t == "THER"),
            _ => false,
        };
        if !keyword_ok {
            return Err(parse_err(
                source,
                kw_no,
                format!(
                    "the block must open with THERMO, THER, THERMO ALL or THER ALL; this \
                     line reads \"{}\"",
                    kw_line.trim()
                ),
            ));
        }

        // Line 2, present unless the next significant line is already a card 1.
        let mut idx = 1usize;
        let mut defaults: Option<[f64; 3]> = None;
        if let Some(&(no, l)) = sig.get(1) {
            // The line after the keyword line is line 2 only if it is
            // neither a card 1 nor a block end: THERMO\nEND\n is an empty
            // set with no line 2, and a block keyword there ends the block,
            // it is not a temperature line.
            if !is_card_1(l) && !is_block_end(l) {
                if !l.is_ascii() {
                    return Err(parse_err(
                        source,
                        no,
                        "the temperature line holds a non-ASCII character".into(),
                    ));
                }
                let mut d = [0.0f64; 3];
                for (j, slot) in d.iter_mut().enumerate() {
                    let a = 10 * j + 1;
                    let b = a + 9;
                    let f = columns(l, a, b);
                    match parse_e(f) {
                        Some(v) => *slot = v,
                        None => {
                            return Err(parse_err(
                                source,
                                no,
                                format!(
                                    "columns {a}-{b} of the temperature line read \"{f}\", \
                                     which is not a finite temperature"
                                ),
                            ));
                        }
                    }
                }
                defaults = Some(d);
                idx = 2;
            }
        }

        let mut species: Vec<SpeciesThermo> = Vec::new();
        let mut notes: Vec<String> = Vec::new();
        while idx < sig.len() {
            let (no1, l1) = sig[idx];
            // The card-1 shape is tested first: a card 1 is never a block
            // end, whatever its species name spells.
            if !is_card_1(l1) {
                if is_block_end(l1) {
                    break;
                }
                check_line_shape(l1).map_err(|m| parse_err(source, no1, m))?;
                return Err(parse_err(
                    source,
                    no1,
                    format!(
                        "column 80 reads \"{}\"; a record opens with its card 1, which \
                         carries the number 1 there",
                        columns(l1, 80, 80)
                    ),
                ));
            }
            check_line_shape(l1).map_err(|m| parse_err(source, no1, m))?;

            // ---- card 1: the name, the formula, the phase, the range ------
            let Some(name) = columns(l1, 1, 18).split_whitespace().next() else {
                return Err(parse_err(source, no1, "columns 1-18 hold no species name".into()));
            };
            // DESIGN: GRI-Mech 3.0's thermo30.dat writes T_mid as F10 in
            // columns 66-75; SAND89-8009 Table III says E8.0 in 66-73; a
            // digit or point in column 74 cannot begin an element symbol,
            // so it selects the wide field (SPEC-LIT §129.3).
            let wide_tmid = columns(l1, 74, 74)
                .bytes()
                .next()
                .is_some_and(|b| b.is_ascii_digit() || b == b'.');
            let mut elems: Vec<(String, u32)> = Vec::new();
            let slots = [(25usize, 26usize, 27, 29), (30, 31, 32, 34), (35, 36, 37, 39),
                (40, 41, 42, 44)];
            let fifth = (74usize, 75usize, 76, 78);
            for (sa, sb, ca, cb) in slots.into_iter().chain((!wide_tmid).then_some(fifth)) {
                let sym = columns(l1, sa, sb).trim();
                let cnt = columns(l1, ca, cb).trim();
                if sym.is_empty() {
                    if cnt.is_empty() || cnt == "0" {
                        continue; // an empty slot
                    }
                    return Err(parse_err(
                        source,
                        no1,
                        format!(
                            "species {name}: columns {ca}-{cb} hold the count {cnt} with \
                             no element symbol"
                        ),
                    ));
                }
                let Ok(n) = cnt.parse::<u32>() else {
                    return Err(parse_err(
                        source,
                        no1,
                        format!(
                            "species {name}: the count of {sym} in columns {ca}-{cb} \
                             reads \"{cnt}\", which is not a non-negative integer"
                        ),
                    ));
                };
                if n == 0 {
                    continue; // a zero count is dropped
                }
                if elements.weight(sym).is_none() {
                    return Err(parse_err(
                        source,
                        no1,
                        format!(
                            "species {name}: the element {sym} in columns {sa}-{sb} is \
                             not in the element table; give its weight in an ELEMENTS \
                             block (SPEC-LIT §129.2)"
                        ),
                    ));
                }
                elems.push((sym.to_ascii_uppercase(), n));
            }
            if elems.is_empty() {
                return Err(parse_err(
                    source,
                    no1,
                    format!(
                        "species {name}: card 1 carries no element in its five formula \
                         slots, so the species has no molar mass (SPEC-LIT §128.2)"
                    ),
                ));
            }
            if wide_tmid {
                let tail = columns(l1, 76, 78);
                if !tail.trim().is_empty() {
                    return Err(parse_err(
                        source,
                        no1,
                        format!(
                            "species {name}: the wide T_mid field runs to column 75, and \
                             columns 76-78 must be blank, but they read \"{tail}\""
                        ),
                    ));
                }
            }
            let phase = columns(l1, 45, 45);
            if phase != "G" && phase != "g" {
                return Err(parse_err(
                    source,
                    no1,
                    format!(
                        "species {name}: the phase in column 45 reads \"{phase}\"; \
                         condensed species are not supported; §131 is gas-phase only"
                    ),
                ));
            }
            let mut temps = [0.0f64; 3]; // t_low, t_high, t_mid, in column order
            let tmid = if wide_tmid { (66usize, 75usize) } else { (66, 73) };
            for (j, (a, b)) in [(46usize, 55usize), (56, 65), tmid].into_iter().enumerate()
            {
                let f = columns(l1, a, b);
                if let Some(v) = parse_e(f) {
                    temps[j] = v;
                    continue;
                }
                if f.trim().is_empty() {
                    let Some(d) = defaults else {
                        return Err(parse_err(
                            source,
                            no1,
                            format!(
                                "species {name}: columns {a}-{b} are blank and the block \
                                 carries no temperature line to take the value from"
                            ),
                        ));
                    };
                    // Line 2 runs lowest, common, highest (Table III), while the
                    // card 1 fields run T_low, T_high, T_mid.
                    temps[j] = d[[0usize, 2, 1][j]];
                    continue;
                }
                return Err(parse_err(
                    source,
                    no1,
                    format!(
                        "species {name}: columns {a}-{b} read \"{f}\", which is not a \
                         temperature"
                    ),
                ));
            }
            let (t_low, t_high, t_mid) = (temps[0], temps[1], temps[2]);
            if !(0.0 < t_low && t_low < t_mid && t_mid < t_high) {
                return Err(parse_err(
                    source,
                    no1,
                    format!(
                        "species {name}: the record temperatures are T_low = {t_low}, \
                         T_mid = {t_mid}, T_high = {t_high}; 0 < T_low < T_mid < T_high \
                         must hold (SPEC-LIT §129.3)"
                    ),
                ));
            }

            // ---- cards 2, 3, 4: five 15-character fields each of cards
            // 2 and 3, four of card 4. The UPPER interval comes first. ----
            let mut cards: [(usize, &str); 3] = [(0, ""), (0, ""), (0, "")];
            for (k, expect) in ['2', '3', '4'].into_iter().enumerate() {
                let Some(&(no, l)) = sig.get(idx + 1 + k) else {
                    return Err(parse_err(
                        source,
                        no1,
                        format!(
                            "species {name}: the record is cut short by the end of the \
                             text before its card {}",
                            k + 2
                        ),
                    ));
                };
                if is_block_end(l) {
                    return Err(parse_err(
                        source,
                        no1,
                        format!(
                            "species {name}: the record is cut short at line {no} by \
                             \"{}\" before its card {}",
                            first_word(l),
                            k + 2
                        ),
                    ));
                }
                check_line_shape(l)
                    .map_err(|m| parse_err(source, no, format!("species {name}: {m}")))?;
                if columns(l, 80, 80) != expect.to_string() {
                    return Err(parse_err(
                        source,
                        no,
                        format!(
                            "species {name}: column 80 reads \"{}\"; card {expect} of the \
                             record was expected there",
                            columns(l, 80, 80)
                        ),
                    ));
                }
                cards[k] = (no, l);
            }
            let mut upper = [0.0f64; 7];
            let mut lower = [0.0f64; 7];
            for (j, slot) in upper.iter_mut().enumerate().take(5) {
                *slot = read_coeff(source, name, '2', cards[0].1, cards[0].0, j)?;
            }
            upper[5] = read_coeff(source, name, '3', cards[1].1, cards[1].0, 0)?;
            upper[6] = read_coeff(source, name, '3', cards[1].1, cards[1].0, 1)?;
            lower[0] = read_coeff(source, name, '3', cards[1].1, cards[1].0, 2)?;
            lower[1] = read_coeff(source, name, '3', cards[1].1, cards[1].0, 3)?;
            lower[2] = read_coeff(source, name, '3', cards[1].1, cards[1].0, 4)?;
            for (j, k) in [(0usize, 3usize), (1, 4), (2, 5), (3, 6)] {
                lower[k] = read_coeff(source, name, '4', cards[2].1, cards[2].0, j)?;
            }

            // ---- the molar mass is BUILT from the element counts (§128.2),
            // never read as a number ----
            let mut mass = 0.0f64;
            for (sym, n) in &elems {
                let w = elements
                    .weight(sym)
                    .expect("element weight was checked when the slot was read");
                mass += *n as f64 * w;
            }

            let record = SpeciesThermo {
                name: name.to_string(),
                elements: elems,
                molar_mass: mass / 1000.0,
                t_low,
                t_mid,
                t_high,
                upper,
                lower,
                line: no1,
            };
            let dup_line = species.iter().find(|s| s.name == name).map(|s| s.line);
            match dup_line {
                Some(first_line) => notes.push(format!(
                    "line {no1}: species {name} repeats the record of line {first_line}; \
                     the first is kept"
                )),
                None => species.push(record),
            }
            idx += 4;
        }

        Ok(ThermoSet { species, p_ref, source: source.to_string(), notes })
    }

    /// A set made from records already read, in the given order, with `notes` empty.
    /// Err (Error::Config): `p_ref` not finite and > 0, or two records with the same name
    /// (the message names it).
    pub fn from_species(
        species: Vec<SpeciesThermo>,
        p_ref: f64,
        source: &str,
    ) -> crate::Result<ThermoSet> {
        if !p_ref.is_finite() || p_ref <= 0.0 {
            return Err(Error::Config(format!(
                "the thermo set is made with the standard pressure p_ref = {p_ref} Pa, \
                 which is not a finite pressure above zero (SPEC-LIT §128.4)"
            )));
        }
        let mut seen = std::collections::HashSet::new();
        for s in &species {
            if !seen.insert(s.name.as_str()) {
                return Err(Error::Config(format!(
                    "the thermo set made for {source} holds two records named {}; a species \
                     has one record (SPEC-LIT §128)",
                    s.name
                )));
            }
        }
        Ok(ThermoSet { species, p_ref, source: source.to_string(), notes: Vec::new() })
    }

    /// The species records, in file order.
    pub fn species(&self) -> &[SpeciesThermo] {
        &self.species
    }

    /// The record's position, by exact case-sensitive name.
    pub fn index_of(&self, name: &str) -> Option<usize> {
        self.species.iter().position(|s| s.name == name)
    }

    /// The record, by exact case-sensitive name.
    pub fn get(&self, name: &str) -> Option<&SpeciesThermo> {
        self.index_of(name).map(|i| &self.species[i])
    }

    /// Pa: the standard pressure the numbers are tabulated at (SPEC-LIT §128.4).
    pub fn p_ref(&self) -> f64 {
        self.p_ref
    }

    /// The `source` argument the set was read with.
    pub fn source(&self) -> &str {
        &self.source
    }

    /// Non-fatal reader notes (a duplicate record), in line order.
    pub fn notes(&self) -> &[String] {
        &self.notes
    }

    /// The same set at another standard pressure (128.10): a7 of both intervals of every
    /// species becomes a7 - ln(p_ref / self.p_ref()); everything else is copied unchanged.
    /// `p_ref == self.p_ref()` returns a clone equal under `==`. Err for p_ref not finite > 0.
    pub fn with_p_ref(&self, p_ref: f64) -> crate::Result<ThermoSet> {
        if !p_ref.is_finite() || p_ref <= 0.0 {
            return Err(Error::Config(format!(
                "the set is moved to the standard pressure p_ref = {p_ref} Pa, which is \
                 not a finite pressure above zero (SPEC-LIT §128.4)"
            )));
        }
        if p_ref == self.p_ref {
            return Ok(self.clone());
        }
        let shift = (p_ref / self.p_ref).ln();
        let mut species = self.species.clone();
        for s in &mut species {
            s.upper[6] -= shift;
            s.lower[6] -= shift;
        }
        Ok(ThermoSet { species, p_ref, source: self.source.clone(), notes: self.notes.clone() })
    }

    /// The mixture's mass fractions against the set, checked: one value per
    /// species, every one finite and at or above zero, and a positive sum of
    /// Y/W. `Y` is not normalised.
    fn check_y(&self, y: &[f64]) -> crate::Result<f64> {
        if y.len() != self.species.len() {
            return Err(Error::Config(format!(
                "the mixture is given {} mass fraction value(s) but the thermo set read \
                 from {} carries {} species",
                y.len(),
                self.source,
                self.species.len()
            )));
        }
        for (k, sp) in self.species.iter().enumerate() {
            let v = y[k];
            if !v.is_finite() || v < 0.0 {
                return Err(Error::Config(format!(
                    "species {}: the mass fraction {} is not a finite number at or above \
                     zero",
                    sp.name, v
                )));
            }
        }
        let sum: f64 = y
            .iter()
            .zip(&self.species)
            .map(|(v, sp)| v / sp.molar_mass)
            .sum();
        if sum.is_nan() || sum <= 0.0 {
            return Err(Error::Config(
                "all mass fractions are zero; a mixture needs a positive sum of Y/W"
                    .to_string(),
            ));
        }
        Ok(sum)
    }

    /// (128.6), kg/mol.
    pub fn molar_mass_mix(&self, y: &[f64]) -> crate::Result<f64> {
        Ok(1.0 / self.check_y(y)?)
    }

    /// (128.6).
    pub fn mole_fractions(&self, y: &[f64]) -> crate::Result<Vec<f64>> {
        let sum = self.check_y(y)?;
        let w = 1.0 / sum;
        Ok(y.iter()
            .zip(&self.species)
            .map(|(v, sp)| v * w / sp.molar_mass)
            .collect())
    }

    /// (128.7), J/(kg K). A species with `y[k] == 0.0` contributes exactly
    /// nothing and is not evaluated, so its temperature range is not checked.
    pub fn cp_mass(&self, t: f64, y: &[f64]) -> crate::Result<f64> {
        let _ = self.check_y(y)?;
        let mut acc = 0.0;
        for (k, sp) in self.species.iter().enumerate() {
            if y[k] == 0.0 {
                continue;
            }
            acc += y[k] * sp.cp(t)? / sp.molar_mass;
        }
        Ok(acc)
    }

    /// (128.7), J/kg. A species with `y[k] == 0.0` contributes exactly
    /// nothing and is not evaluated, so its temperature range is not checked.
    pub fn h_mass(&self, t: f64, y: &[f64]) -> crate::Result<f64> {
        let _ = self.check_y(y)?;
        let mut acc = 0.0;
        for (k, sp) in self.species.iter().enumerate() {
            if y[k] == 0.0 {
                continue;
            }
            acc += y[k] * sp.h(t)? / sp.molar_mass;
        }
        Ok(acc)
    }

    /// (128.9), mol/m^3: p X_k / (R_GAS T).
    pub fn concentrations(&self, p: f64, t: f64, y: &[f64]) -> crate::Result<Vec<f64>> {
        if !p.is_finite() || p <= 0.0 {
            return Err(Error::Config(format!(
                "concentrations need a pressure that is finite and above zero, given \
                 p = {p} Pa"
            )));
        }
        if !t.is_finite() || t <= 0.0 {
            return Err(Error::Config(format!(
                "concentrations need a temperature that is finite and above zero, given \
                 T = {t} K"
            )));
        }
        let x = self.mole_fractions(y)?;
        Ok(x.iter().map(|xk| p * xk / (R_GAS * t)).collect())
    }
}
