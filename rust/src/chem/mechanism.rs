// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.
// Provenance: see PROVENANCE.md. No GPL-licensed source was consulted.

//! A CHEMKIN mechanism (ELEMENTS, SPECIES, an optional THERMO block and an
//! optional REACTIONS block) read into a validated [`Mechanism`]: every
//! coefficient is an exact [`Rational`], every reaction's element balance is
//! checked exactly, and every rate is stored in SI as `A`, `beta` and
//! `theta = E/R`. SPEC-LIT §129 is the contract, with §130.2 for the units
//! and §130.5 for FORD and RORD.
//!
//! Written from:
//!   SPEC-LIT §129, §130.2 and §130.5
//!   R. J. Kee, F. M. Rupley & J. A. Miller, *Chemkin-II*, Sandia
//!     SAND89-8009 (1989), DOI 10.2172/5681118, Chapter IV and Tables I, II,
//!     IV and V, read as a format
//!   R. J. Kee, F. M. Rupley, E. Meeks & J. A. Miller, *CHEMKIN-III*, Sandia
//!     SAND96-8216 (1996), DOI 10.2172/481621, pp. 46-54 (non-integer
//!     coefficients, FORD, RORD, UNITS)
//!   NIST SP 811 (2008) Appendix B.8, for `1 cal = 4.184 J`, and the SI 2019
//!     exact values of `N_A`, `e` and `k_B`
//! No CHEMKIN, Cantera or other reader's source was opened. No GPL-licensed
//! source was consulted.
//!
//! The fall-off family (`(+M)`, `LOW`, `TROE`, `SRI`), `REV` and `DUP` are
//! refused here with the words `not read yet`: the next unit reads them.
//!
//! Host only, and f64 everywhere: no `Scalar`, no device types, so the module
//! and its tests behave identically in the default build and in
//! `--features single`.

use std::collections::HashMap;
use std::fmt;

use super::ck_lexer::{self, Block, SlashItem};
use super::thermo::{ElementTable, ThermoSet, P_ATM, R_GAS};
use crate::error::Error;

/// The thermochemical calorie, J (NIST SP 811 Appendix B.8, exact).
pub const CAL_J: f64 = 4.184;
/// Avogadro's number, 1/mol (SI 2019, exact).
pub const N_AVOGADRO: f64 = 6.02214076e23;
/// The elementary charge, C (SI 2019, exact).
pub const E_CHARGE: f64 = 1.602176634e-19;
/// The Boltzmann constant, J/K (SI 2019, exact).
pub const K_BOLTZMANN: f64 = 1.380649e-23;

// ==========================================================================
//  Exact rationals
// ==========================================================================

fn gcd(mut a: u128, mut b: u128) -> u128 {
    while b != 0 {
        let t = a % b;
        a = b;
        b = t;
    }
    a
}

/// An exact rational num/den: den > 0, gcd(|num|, den) == 1, so `==` is value equality.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Rational {
    num: i64,
    den: i64,
}

impl Rational {
    /// None when den == 0. Reduces and moves the sign to num: new(4, -6) == new(-2, 3).
    /// (None also for the one case whose reduced value does not fit an i64, such as
    /// `i64::MIN` over -1.)
    pub fn new(num: i64, den: i64) -> Option<Rational> {
        if den == 0 {
            return None;
        }
        let (mut n, mut d) = (num as i128, den as i128);
        if d < 0 {
            n = -n;
            d = -d;
        }
        let g = gcd(n.unsigned_abs(), d.unsigned_abs()) as i128;
        n /= g;
        d /= g;
        Some(Rational { num: i64::try_from(n).ok()?, den: i64::try_from(d).ok()? })
    }

    /// An unsigned decimal coefficient as written in a reaction: one or more digits with an
    /// optional `.` and fraction ("2", "0.5", ".5", "1.", "0.50", "1.25"). At least one digit
    /// in all. No sign, no exponent, at most 15 digits in all. "0.49" -> 49/100, "0.50" -> 1/2.
    /// None for anything else.
    pub fn parse_decimal(s: &str) -> Option<Rational> {
        let mut digits = 0u32;
        let mut dots = 0u32;
        let mut frac = 0u32;
        let mut num: i64 = 0;
        for c in s.chars() {
            match c {
                '0'..='9' => {
                    digits += 1;
                    if digits > 15 {
                        return None;
                    }
                    num = num * 10 + i64::from(c as u8 - b'0');
                    if dots == 1 {
                        frac += 1;
                    }
                }
                '.' => {
                    dots += 1;
                    if dots > 1 {
                        return None;
                    }
                }
                _ => return None,
            }
        }
        if digits == 0 {
            return None;
        }
        Rational::new(num, 10i64.pow(frac))
    }

    pub fn num(&self) -> i64 {
        self.num
    }

    pub fn den(&self) -> i64 {
        self.den
    }

    /// num as f64 / den as f64 (the one rounding of SPEC-LIT §129.4).
    pub fn to_f64(&self) -> f64 {
        self.num as f64 / self.den as f64
    }

    fn is_zero(&self) -> bool {
        self.num == 0
    }

    /// `a/b + c/d` over the least common denominator, every product checked.
    fn checked_add(self, o: Rational) -> Option<Rational> {
        let g = gcd(self.den as u128, o.den as u128) as i64;
        let self_part = self.den / g;
        let other_part = o.den / g;
        let num = self
            .num
            .checked_mul(other_part)?
            .checked_add(o.num.checked_mul(self_part)?)?;
        let den = self.den.checked_mul(other_part)?;
        Rational::new(num, den)
    }

    fn checked_sub(self, o: Rational) -> Option<Rational> {
        self.checked_add(Rational { num: o.num.checked_neg()?, den: o.den })
    }

    fn checked_mul_int(self, n: i64) -> Option<Rational> {
        Rational::new(self.num.checked_mul(n)?, self.den)
    }
}

/// "n" when den == 1, else "n/d": new(-3, 6) displays "-1/2", new(4, 2) displays "2".
impl fmt::Display for Rational {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.den == 1 {
            write!(f, "{}", self.num)
        } else {
            write!(f, "{}/{}", self.num, self.den)
        }
    }
}

// ==========================================================================
//  The reaction
// ==========================================================================

/// k(T) = a T^beta exp(-theta / T) in SI (SPEC-LIT §130.2).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Arrhenius {
    /// SI: (m^3/mol)^(m-1) / s, with m from SPEC-LIT §130.2.
    pub a: f64,
    pub beta: f64,
    /// E / R_c, in kelvin.
    pub theta: f64,
}

/// A `+M` third body (SPEC-LIT §129.4, (130.3)).
#[derive(Clone, Debug, PartialEq)]
pub struct ThirdBody {
    /// The enhanced efficiencies as written, in written order: (species index, alpha).
    /// Every species not listed has alpha = 1.
    pub efficiencies: Vec<(usize, f64)>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Reaction {
    /// The 1-based line of the reaction line in the mechanism text.
    pub line: usize,
    /// The reaction field as written, trimmed at both ends (inner spaces kept).
    pub equation: String,
    /// `=` or `<=>` is true; `=>` is false.
    pub reversible: bool,
    /// (species index, nu') merged per species, in order of first appearance.
    pub reactants: Vec<(usize, Rational)>,
    /// (species index, nu'') merged per species, in order of first appearance.
    pub products: Vec<(usize, Rational)>,
    pub third_body: Option<ThirdBody>,
    pub rate: Arrhenius,
    /// FORD overrides in written order: (species index, order).
    pub ford: Vec<(usize, f64)>,
    /// RORD overrides in written order: (species index, order).
    pub rord: Vec<(usize, f64)>,
}

/// (130.13) for one side: each species of `side` with its override if given, else its
/// coefficient as f64; then every override species that is not on the side, in override order.
fn orders(side: &[(usize, Rational)], over: &[(usize, f64)]) -> Vec<(usize, f64)> {
    let mut out: Vec<(usize, f64)> = side
        .iter()
        .map(|(k, nu)| {
            let o = over.iter().find(|(j, _)| j == k).map_or_else(|| nu.to_f64(), |(_, v)| *v);
            (*k, o)
        })
        .collect();
    for (j, v) in over {
        if !side.iter().any(|(k, _)| k == j) {
            out.push((*j, *v));
        }
    }
    out
}

impl Reaction {
    /// (130.13), forward side: for each reactant in `reactants` order, its FORD value if given,
    /// else nu' as f64; then every FORD species that is not a reactant, in FORD order.
    pub fn forward_orders(&self) -> Vec<(usize, f64)> {
        orders(&self.reactants, &self.ford)
    }

    /// The same for products and RORD. An irreversible reaction returns an empty Vec.
    pub fn reverse_orders(&self) -> Vec<(usize, f64)> {
        if self.reversible {
            orders(&self.products, &self.rord)
        } else {
            Vec::new()
        }
    }

    /// m of SPEC-LIT §130.2: forward_orders' values summed left to right from 0.0, then
    /// plus 1.0 with a third body.
    pub fn molecularity(&self) -> f64 {
        let m = self.forward_orders().iter().fold(0.0f64, |acc, (_, o)| acc + o);
        if self.third_body.is_some() {
            m + 1.0
        } else {
            m
        }
    }
}

// ==========================================================================
//  The mechanism
// ==========================================================================

#[derive(Clone, Debug, PartialEq)]
pub struct Mechanism {
    source: String,
    elements: Vec<(String, f64)>,
    species: Vec<String>,
    thermo: ThermoSet,
    reactions: Vec<Reaction>,
    notes: Vec<String>,
}

impl Mechanism {
    /// Reads one mechanism text (the rules below). `source` names the text in every error.
    /// `thermo` is an optional separate thermo set (the CHEMKIN database). A species' record is
    /// taken from the in-file THERMO block when it has one there, else from `thermo`.
    pub fn read_chemkin(
        text: &str,
        source: &str,
        thermo: Option<&ThermoSet>,
    ) -> crate::Result<Mechanism> {
        let mut rd = Reader::new(text, source, thermo);
        rd.run()?;
        rd.finish()
    }

    pub fn source(&self) -> &str {
        &self.source
    }

    /// The declared elements in declaration order: (upper-case symbol, weight g/mol).
    pub fn elements(&self) -> &[(String, f64)] {
        &self.elements
    }

    /// The declared species names in declaration order.
    pub fn species(&self) -> &[String] {
        &self.species
    }

    /// Exact, case-sensitive match.
    pub fn species_index(&self, name: &str) -> Option<usize> {
        self.species.iter().position(|s| s == name)
    }

    /// One record per species, in species() order, at p_ref == P_ATM.
    pub fn thermo(&self) -> &ThermoSet {
        &self.thermo
    }

    pub fn reactions(&self) -> &[Reaction] {
        &self.reactions
    }

    /// Non-fatal notes in line order (duplicated declarations, the KJOULES/MOLE extension,
    /// reversible reactions with FORD or RORD and no REV, and the thermo reader's notes).
    pub fn notes(&self) -> &[String] {
        &self.notes
    }
}

// ==========================================================================
//  The reader
// ==========================================================================

/// The energy unit of `E`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum EnergyUnit {
    Cal,
    Kcal,
    Joul,
    Kjou,
    Kelv,
    Evol,
}

/// The unit of `A`: moles, or molecules.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum AUnit {
    Moles,
    Molecules,
}

/// A reaction whose auxiliary lines are still being read: `a` and `theta` are computed only
/// when the next reaction line or the end of the block closes it, because FORD changes the
/// molecularity and UNITS changes the units.
struct Pending {
    rx: Reaction,
    a: f64,
    beta: f64,
    e: f64,
    energy: Option<EnergyUnit>,
    a_unit: Option<AUnit>,
    units_seen: bool,
}

/// One side of an equation.
struct Side {
    terms: Vec<(usize, Rational)>,
    third_body: bool,
}

/// A number as CHEMKIN writes it: a `D` or `d` exponent is read as `E`; finite only.
fn parse_f64(tok: &str) -> Option<f64> {
    tok.replace(['D', 'd'], "E").parse::<f64>().ok().filter(|v| v.is_finite())
}

/// `theta = E / R_c` in kelvin, one expression per unit and in this evaluation order
/// (SPEC-LIT §130.2).
fn theta_of(e: f64, unit: EnergyUnit) -> f64 {
    match unit {
        EnergyUnit::Cal => (CAL_J * e) / R_GAS,
        EnergyUnit::Kcal => (4184.0 * e) / R_GAS,
        EnergyUnit::Joul => e / R_GAS,
        EnergyUnit::Kjou => (1000.0 * e) / R_GAS,
        EnergyUnit::Kelv => e,
        EnergyUnit::Evol => (e * E_CHARGE) / K_BOLTZMANN,
    }
}

/// `A` in SI for a rate constant that multiplies molecularity `m` (SPEC-LIT §130.2).
fn a_si_of(a: f64, unit: AUnit, m: f64) -> f64 {
    match unit {
        AUnit::Moles => a * (1.0e-6_f64).powf(m - 1.0),
        AUnit::Molecules => a * (N_AVOGADRO * 1.0e-6).powf(m - 1.0),
    }
}

/// The auxiliary keywords the next unit reads (docs/17 CHR-02b).
const NEXT_UNIT: [&str; 6] = ["LOW", "TROE", "SRI", "REV", "DUP", "DUPLICATE"];
/// The auxiliary keywords SPEC-LIT §129.5 refuses by name.
const BY_NAME: [&str; 12] = [
    "HIGH", "LT", "RLT", "JAN", "FIT1", "HV", "TDEP", "EXCI", "MOME", "XSMI", "PLOG", "CHEB",
];

struct Reader<'a> {
    source: &'a str,
    lines: Vec<&'a str>,
    database: Option<&'a ThermoSet>,
    elements: Vec<(String, f64)>,
    table: ElementTable,
    /// Name and 1-based declaration line, in declaration order.
    species: Vec<(String, usize)>,
    species_ix: HashMap<String, usize>,
    in_file: Option<ThermoSet>,
    built: Option<ThermoSet>,
    notes: Vec<String>,
    reactions: Vec<Reaction>,
    block_energy: EnergyUnit,
    block_a: AUnit,
    seen_elements: bool,
    seen_species: bool,
    last: Option<Block>,
}

impl<'a> Reader<'a> {
    fn new(text: &'a str, source: &'a str, database: Option<&'a ThermoSet>) -> Reader<'a> {
        Reader {
            source,
            lines: text.lines().collect(),
            database,
            elements: Vec::new(),
            table: ElementTable::standard(),
            species: Vec::new(),
            species_ix: HashMap::new(),
            in_file: None,
            built: None,
            notes: Vec::new(),
            reactions: Vec::new(),
            block_energy: EnergyUnit::Cal,
            block_a: AUnit::Moles,
            seen_elements: false,
            seen_species: false,
            last: None,
        }
    }

    // ---- errors ----------------------------------------------------------

    fn err(&self, line: usize, msg: String) -> Error {
        Error::Parse { path: self.source.to_string(), msg: format!("line {line}: {msg}") }
    }

    fn end_err(&self, msg: String) -> Error {
        Error::Parse { path: self.source.to_string(), msg: format!("end of text: {msg}") }
    }

    /// A refusal of the element table, passed on with the line it belongs to.
    fn wrap(&self, line: usize, e: Error) -> Error {
        match e {
            Error::Config(m) => self.err(line, m),
            other => other,
        }
    }

    // ---- the block frame -------------------------------------------------

    fn run(&mut self) -> crate::Result<()> {
        let n = self.lines.len();
        let mut i = 0usize;
        let mut open: Option<Block> = None;
        let mut ended = false;
        let mut pending: Option<Pending> = None;
        while i < n {
            let lno = i + 1;
            let line = ck_lexer::code(self.lines[i]);
            if line.trim().is_empty() {
                i += 1;
                continue;
            }
            if ended {
                return Err(self.err(
                    lno,
                    format!(
                        "the text \"{}\" follows the END of the REACTIONS block, and nothing \
                         may follow it",
                        line.trim()
                    ),
                ));
            }
            if let Some(b) = ck_lexer::block_keyword(line) {
                self.enter(b, lno, i + 1)?;
                self.last = Some(b);
                open = None;
                match b {
                    Block::Elements => {
                        self.seen_elements = true;
                        if !self.read_element_items(lno, ck_lexer::after_first_token(line))? {
                            open = Some(Block::Elements);
                        }
                    }
                    Block::Species => {
                        self.seen_species = true;
                        if !self.read_species_tokens(lno, ck_lexer::after_first_token(line))? {
                            open = Some(Block::Species);
                        }
                    }
                    Block::Thermo => {
                        i = self.read_thermo_block(i)?;
                        continue;
                    }
                    Block::Reactions => {
                        self.build_thermo()?;
                        self.read_reactions_header(lno, ck_lexer::after_first_token(line))?;
                        open = Some(Block::Reactions);
                    }
                }
                i += 1;
                continue;
            }
            match open {
                None => {
                    let what = if self.last.is_none() {
                        "text comes before the ELEMENTS block; the first block of a mechanism \
                         is ELEMENTS"
                            .to_string()
                    } else {
                        "text comes between blocks; expected a block keyword (ELEMENTS, \
                         SPECIES, THERMO or REACTIONS)"
                            .to_string()
                    };
                    return Err(self.err(lno, format!("{what}: \"{}\"", line.trim())));
                }
                Some(Block::Elements) => {
                    if self.read_element_items(lno, line)? {
                        open = None;
                    }
                }
                Some(Block::Species) => {
                    if self.read_species_tokens(lno, line)? {
                        open = None;
                    }
                }
                Some(Block::Reactions) => {
                    let first = line.split_whitespace().next().unwrap_or("");
                    if first.eq_ignore_ascii_case("END") {
                        if line.split_whitespace().nth(1).is_some() {
                            return Err(self.err(
                                lno,
                                "an item follows END on the same line".to_string(),
                            ));
                        }
                        if let Some(p) = pending.take() {
                            self.complete(p)?;
                        }
                        ended = true;
                        open = None;
                    } else if line.contains('=') {
                        if let Some(p) = pending.take() {
                            self.complete(p)?;
                        }
                        pending = Some(self.read_reaction_line(lno, line)?);
                    } else {
                        match pending.as_mut() {
                            Some(p) => self.read_auxiliary(lno, line, p)?,
                            None => {
                                return Err(self.err(
                                    lno,
                                    format!(
                                        "the auxiliary line \"{}\" comes before the first \
                                         reaction; an auxiliary line belongs to the reaction \
                                         above it",
                                        line.trim()
                                    ),
                                ));
                            }
                        }
                    }
                }
                Some(Block::Thermo) => {
                    unreachable!("the THERMO block is consumed whole when its keyword is read")
                }
            }
            i += 1;
        }
        if let Some(p) = pending.take() {
            self.complete(p)?;
        }
        if !self.seen_elements {
            return Err(self.end_err(
                "the text holds no ELEMENTS block; a mechanism opens with ELEMENTS".to_string(),
            ));
        }
        if !self.seen_species {
            return Err(self.end_err(
                "the text holds no SPECIES block; SPECIES follows ELEMENTS".to_string(),
            ));
        }
        if self.built.is_none() {
            self.build_thermo()?;
        }
        self.check_duplicates()
    }

    /// Whether the block keyword `b` appears at or after line index `from`.
    fn starts_later(&self, b: Block, from: usize) -> bool {
        self.lines.iter().skip(from).any(|l| {
            !ck_lexer::thermo_card(l) && ck_lexer::block_keyword(ck_lexer::code(l)) == Some(b)
        })
    }

    /// The order rules of SPEC-LIT §129.1: ELEMENTS, SPECIES, THERMO (optional), REACTIONS
    /// (optional), each at most once, and ELEMENTS and SPECIES before anything after them.
    fn enter(&self, b: Block, lno: usize, from: usize) -> crate::Result<()> {
        if let Some(last) = self.last {
            if b.rank() <= last.rank() {
                return Err(self.err(
                    lno,
                    format!(
                        "the {} block cannot come after the {} block: the order is ELEMENTS, \
                         SPECIES, THERMO, REACTIONS, and each block appears at most once",
                        b.name(),
                        last.name()
                    ),
                ));
            }
        }
        for need in [Block::Elements, Block::Species] {
            let seen = match need {
                Block::Elements => self.seen_elements,
                _ => self.seen_species,
            };
            if need.rank() < b.rank() && !seen {
                if self.starts_later(need, from) {
                    return Err(self.err(
                        lno,
                        format!(
                            "the {} block comes before the {} block: the order is ELEMENTS, \
                             SPECIES, THERMO, REACTIONS",
                            b.name(),
                            need.name()
                        ),
                    ));
                }
                return Err(self.end_err(format!(
                    "the text holds no {} block, and the {} block at line {lno} needs it \
                     before it",
                    need.name(),
                    b.name()
                )));
            }
        }
        Ok(())
    }

    // ---- ELEMENTS and SPECIES -------------------------------------------

    /// Reads the items of one ELEMENTS line; true when an `END` item closed the block.
    fn read_element_items(&mut self, lno: usize, text: &str) -> crate::Result<bool> {
        let items = ck_lexer::slash_items(text).map_err(|m| self.err(lno, m))?;
        let mut ended = false;
        for it in items {
            if ended {
                return Err(self.err(
                    lno,
                    format!("the item {} follows END on the same line", it.name),
                ));
            }
            if it.name.eq_ignore_ascii_case("END") {
                if it.content.is_some() {
                    return Err(self.err(lno, "END takes no /.../ field".to_string()));
                }
                ended = true;
                continue;
            }
            self.declare_element(lno, &it)?;
        }
        Ok(ended)
    }

    fn declare_element(&mut self, lno: usize, it: &SlashItem) -> crate::Result<()> {
        let sym = it.name.to_ascii_uppercase();
        if sym == "E" {
            return Err(self.err(
                lno,
                "the element E is the electron, and this reader has no plasma machinery \
                 (SPEC-LIT §129.2)"
                    .to_string(),
            ));
        }
        let shaped = (1..=2).contains(&sym.len()) && sym.bytes().all(|b| b.is_ascii_alphabetic());
        if !shaped {
            return Err(self.err(
                lno,
                format!("the element symbol {} is not one or two ASCII letters", it.name),
            ));
        }
        if self.elements.iter().any(|(s, _)| *s == sym) {
            self.notes.push(format!(
                "line {lno}: element {sym} is declared again; the first declaration is kept"
            ));
            return Ok(());
        }
        if let Some(c) = &it.content {
            let toks: Vec<&str> = c.split_whitespace().collect();
            let weight = match toks.as_slice() {
                [t] => parse_f64(t),
                _ => None,
            };
            let Some(w) = weight else {
                return Err(self.err(
                    lno,
                    format!(
                        "element {sym}: the /.../ field reads \"{c}\", which is not one finite \
                         atomic weight"
                    ),
                ));
            };
            self.table = self
                .table
                .clone()
                .with(&sym, w)
                .map_err(|e| self.wrap(lno, e))?;
        }
        match self.table.weight(&sym) {
            Some(w) => {
                self.elements.push((sym, w));
                Ok(())
            }
            None => Err(self.err(
                lno,
                format!(
                    "element {sym} has no atomic weight in the built-in table; give one as \
                     {sym} /weight/ (SPEC-LIT §129.2)"
                ),
            )),
        }
    }

    /// Reads the tokens of one SPECIES line; true when an `END` token closed the block.
    fn read_species_tokens(&mut self, lno: usize, text: &str) -> crate::Result<bool> {
        let mut ended = false;
        for tok in text.split_whitespace() {
            if ended {
                return Err(self.err(lno, format!("the item {tok} follows END on the same line")));
            }
            if tok.eq_ignore_ascii_case("END") {
                ended = true;
                continue;
            }
            if let Some(why) = species_refusal(tok) {
                return Err(self.err(lno, format!("species {tok}: {why}")));
            }
            match self.species_ix.get(tok) {
                Some(&k) => {
                    let first = self.species[k].1;
                    self.notes.push(format!(
                        "line {lno}: species {tok} is declared again; the first declaration, \
                         line {first}, is kept"
                    ));
                }
                None => {
                    self.species_ix.insert(tok.to_string(), self.species.len());
                    self.species.push((tok.to_string(), lno));
                }
            }
        }
        Ok(ended)
    }

    // ---- THERMO ----------------------------------------------------------

    /// Reads the THERMO block whose keyword is at line index `i`; returns the index of the
    /// first line after the block.
    fn read_thermo_block(&mut self, i: usize) -> crate::Result<usize> {
        let n = self.lines.len();
        let (end, next) = {
            let mut found = None;
            for j in i + 1..n {
                let l = self.lines[j];
                if ck_lexer::thermo_card(l) {
                    continue;
                }
                let first = l.split_whitespace().next().unwrap_or("");
                if first.eq_ignore_ascii_case("END") {
                    found = Some((j, j + 1));
                    break;
                }
                if ck_lexer::block_keyword(l).is_some() {
                    found = Some((j - 1, j));
                    break;
                }
            }
            found.unwrap_or((n - 1, n))
        };
        let mut text = "\n".repeat(i);
        text.push_str(&self.lines[i..=end].join("\n"));
        let set = ThermoSet::read_chemkin(&text, self.source, P_ATM, &self.table)?;
        self.notes.extend(set.notes().iter().cloned());
        self.in_file = Some(set);
        Ok(next)
    }

    /// The thermo of the mechanism: one record per declared species, the in-file record first
    /// (SAND89-8009 Case 2: the file overrides the database), else the database's at P_ATM.
    fn build_thermo(&mut self) -> crate::Result<()> {
        let database = match self.database {
            Some(t) => Some(t.with_p_ref(P_ATM)?),
            None => None,
        };
        let mut records = Vec::with_capacity(self.species.len());
        for (name, decl) in &self.species {
            let from_file = self.in_file.as_ref().and_then(|s| s.get(name));
            let (mut rec, from_database) = match (from_file, database.as_ref().and_then(|s| s.get(name)))
            {
                (Some(r), _) => (r.clone(), false),
                (None, Some(r)) => (r.clone(), true),
                (None, None) => {
                    return Err(self.err(
                        *decl,
                        format!(
                            "species {name} has no thermo record: neither the THERMO block nor \
                             the separate thermo set holds one (SPEC-LIT §129.1)"
                        ),
                    ));
                }
            };
            let mut mass = 0.0f64;
            for (sym, count) in &rec.elements {
                let Some((_, w)) = self.elements.iter().find(|(s, _)| s == sym) else {
                    return Err(self.err(
                        *decl,
                        format!(
                            "species {name}: its thermo record names element {sym}, which the \
                             ELEMENTS block does not declare"
                        ),
                    ));
                };
                mass += f64::from(*count) * w;
            }
            if from_database {
                // The database was read with the built-in weights; the mechanism's ELEMENTS
                // may override them, and one mixture must carry one set of weights.
                rec.molar_mass = mass / 1000.0;
            }
            records.push(rec);
        }
        self.built = Some(ThermoSet::from_species(records, P_ATM, self.source)?);
        Ok(())
    }

    // ---- REACTIONS -------------------------------------------------------

    /// The unit keywords after REACTIONS: one energy unit and/or one A unit.
    fn read_reactions_header(&mut self, lno: usize, text: &str) -> crate::Result<()> {
        let mut energy: Option<EnergyUnit> = None;
        let mut a_unit: Option<AUnit> = None;
        for tok in text.split_whitespace() {
            let t = tok.to_ascii_uppercase();
            let e = match t.as_str() {
                "CAL/MOLE" => Some(EnergyUnit::Cal),
                "KCAL/MOLE" => Some(EnergyUnit::Kcal),
                "JOULES/MOLE" => Some(EnergyUnit::Joul),
                "KJOULES/MOLE" => Some(EnergyUnit::Kjou),
                "KELVINS" => Some(EnergyUnit::Kelv),
                "EVOLTS" => Some(EnergyUnit::Evol),
                _ => None,
            };
            let a = match t.as_str() {
                "MOLES" => Some(AUnit::Moles),
                "MOLECULES" => Some(AUnit::Molecules),
                _ => None,
            };
            match (e, a) {
                (Some(u), _) => {
                    if energy.is_some() {
                        return Err(self.err(
                            lno,
                            format!("the REACTIONS line names a second energy unit, {tok}"),
                        ));
                    }
                    energy = Some(u);
                    if u == EnergyUnit::Kjou {
                        self.notes.push(format!(
                            "line {lno}: KJOULES/MOLE is a DESIGN extension meaning 1000 J/mol; \
                             it is in neither CHEMKIN manual (SPEC-LIT §129.4)"
                        ));
                    }
                }
                (None, Some(u)) => {
                    if a_unit.is_some() {
                        return Err(self.err(
                            lno,
                            format!("the REACTIONS line names a second A unit, {tok}"),
                        ));
                    }
                    a_unit = Some(u);
                }
                (None, None) => {
                    return Err(self.err(
                        lno,
                        format!(
                            "{tok} is not a REACTIONS unit keyword; the energy units are \
                             CAL/MOLE, KCAL/MOLE, JOULES/MOLE, KJOULES/MOLE, KELVINS and \
                             EVOLTS, and the A units are MOLES and MOLECULES"
                        ),
                    ));
                }
            }
        }
        self.block_energy = energy.unwrap_or(EnergyUnit::Cal);
        self.block_a = a_unit.unwrap_or(AUnit::Moles);
        Ok(())
    }

    /// One reaction line (comment removed): the equation, then `A`, `beta` and `E`.
    fn read_reaction_line(&self, lno: usize, line: &str) -> crate::Result<Pending> {
        let three = |why: &str| {
            self.err(
                lno,
                format!(
                    "a reaction line ends with three numbers, A, beta and E; {why} (the line \
                     reads \"{}\")",
                    line.trim()
                ),
            )
        };
        // The last three whitespace tokens, taken from the right.
        let mut end = line.trim_end().len();
        let mut tail: Vec<&str> = Vec::with_capacity(3);
        for _ in 0..3 {
            let head = &line[..end];
            let start = head.rfind(char::is_whitespace).map_or(0, |p| {
                p + head[p..].chars().next().map_or(1, char::len_utf8)
            });
            tail.push(&head[start..]);
            end = head[..start].trim_end().len();
        }
        let equation = line[..end].trim();
        if equation.is_empty() {
            return Err(three("there are fewer than four tokens"));
        }
        tail.reverse();
        let mut nums = [0.0f64; 3];
        for (slot, tok) in nums.iter_mut().zip(&tail) {
            match parse_f64(tok) {
                Some(v) => *slot = v,
                None => return Err(three(&format!("\"{tok}\" is not a finite number"))),
            }
        }

        let (reversible, left, right) = self.split_equation(lno, equation)?;
        let (reactants, third_left) = self.read_side(lno, &left, equation)?;
        let (products, third_right) = self.read_side(lno, &right, equation)?;
        let third_body = (third_left || third_right).then(|| ThirdBody { efficiencies: Vec::new() });
        let rx = Reaction {
            line: lno,
            equation: equation.to_string(),
            reversible,
            reactants,
            products,
            third_body,
            rate: Arrhenius { a: 0.0, beta: 0.0, theta: 0.0 },
            ford: Vec::new(),
            rord: Vec::new(),
        };
        Ok(Pending {
            rx,
            a: nums[0],
            beta: nums[1],
            e: nums[2],
            energy: None,
            a_unit: None,
            units_seen: false,
        })
    }

    /// The delimiter: exactly one `=`, as `=`, `<=>` or `=>`. Returns (reversible, left, right)
    /// with the whitespace removed from both sides.
    fn split_equation(&self, lno: usize, equation: &str) -> crate::Result<(bool, String, String)> {
        let compact: String = equation.chars().filter(|c| !c.is_whitespace()).collect();
        if compact.contains("(+") {
            return Err(self.err(
                lno,
                format!(
                    "reaction {equation}: a fall-off or collider term (+...) is not read yet; \
                     the next unit, docs/17 CHR-02b, reads it"
                ),
            ));
        }
        let eq_count = compact.matches('=').count();
        if eq_count != 1 {
            return Err(self.err(
                lno,
                format!(
                    "reaction {equation}: the equation holds {eq_count} `=` characters and \
                     must hold exactly one, as `=`, `<=>` or `=>`"
                ),
            ));
        }
        let p = compact.find('=').unwrap_or(0);
        let (l, r) = (&compact[..p], &compact[p + 1..]);
        if let Some(l) = l.strip_suffix('<') {
            return match r.strip_prefix('>') {
                Some(r) => Ok((true, l.to_string(), r.to_string())),
                None => Err(self.err(
                    lno,
                    format!("reaction {equation}: `<=` has no `>`; the delimiters are `=`, `<=>` and `=>`"),
                )),
            };
        }
        Ok(match r.strip_prefix('>') {
            Some(r) => (false, l.to_string(), r.to_string()),
            None => (true, l.to_string(), r.to_string()),
        })
    }

    /// One side of an equation with its whitespace removed: the species with their exact
    /// coefficients, merged per species, and whether it holds `M`.
    fn read_side(
        &self,
        lno: usize,
        side: &str,
        equation: &str,
    ) -> crate::Result<(Vec<(usize, Rational)>, bool)> {
        let mut parsed = Side { terms: Vec::new(), third_body: false };
        if side.is_empty() {
            return Err(self.err(
                lno,
                format!("reaction {equation}: a side of the equation holds no species"),
            ));
        }
        for term in side.split('+') {
            if term.is_empty() {
                return Err(self.err(
                    lno,
                    format!("reaction {equation}: the equation holds an empty term"),
                ));
            }
            let cut = term
                .find(|c: char| !(c.is_ascii_digit() || c == '.'))
                .unwrap_or(term.len());
            let (coef_text, name) = term.split_at(cut);
            if name.is_empty() {
                return Err(self.err(
                    lno,
                    format!("reaction {equation}: the term {term} has a coefficient and no species"),
                ));
            }
            if name == "M" || name == "m" {
                if !coef_text.is_empty() {
                    return Err(self.err(
                        lno,
                        format!(
                            "reaction {equation}: the third body M takes no coefficient, but \
                             the term reads {term}"
                        ),
                    ));
                }
                if parsed.third_body {
                    return Err(self.err(
                        lno,
                        format!("reaction {equation}: M appears twice on one side"),
                    ));
                }
                parsed.third_body = true;
                continue;
            }
            if name.eq_ignore_ascii_case("HV") {
                return Err(self.err(
                    lno,
                    format!(
                        "reaction {equation}: HV is a photon, and this reader does not read \
                         radiative reactions (SPEC-LIT §129.5)"
                    ),
                ));
            }
            if name == "E" {
                return Err(self.err(
                    lno,
                    format!(
                        "reaction {equation}: E is the electron, and this reader has no plasma \
                         machinery (SPEC-LIT §129.2)"
                    ),
                ));
            }
            let Some(&k) = self.species_ix.get(name) else {
                return Err(self.err(
                    lno,
                    format!("reaction {equation}: the species {name} is not declared in SPECIES"),
                ));
            };
            let nu = if coef_text.is_empty() {
                Rational::new(1, 1)
            } else {
                Rational::parse_decimal(coef_text)
            };
            let nu = match nu {
                Some(v) if v.num() > 0 => v,
                _ => {
                    return Err(self.err(
                        lno,
                        format!(
                            "reaction {equation}: the coefficient \"{coef_text}\" of {name} is \
                             not a decimal number above zero"
                        ),
                    ));
                }
            };
            match parsed.terms.iter_mut().find(|(j, _)| *j == k) {
                Some((_, sum)) => {
                    *sum = sum.checked_add(nu).ok_or_else(|| {
                        self.err(
                            lno,
                            format!(
                                "reaction {equation}: the merged coefficient of {name} \
                                 overflows an exact rational"
                            ),
                        )
                    })?;
                }
                None => parsed.terms.push((k, nu)),
            }
        }
        if parsed.terms.is_empty() {
            return Err(self.err(
                lno,
                format!("reaction {equation}: a side of the equation holds no species"),
            ));
        }
        Ok((parsed.terms, parsed.third_body))
    }

    // ---- the auxiliary lines ---------------------------------------------

    fn read_auxiliary(&self, lno: usize, line: &str, p: &mut Pending) -> crate::Result<()> {
        let items = ck_lexer::slash_items(line).map_err(|m| self.err(lno, m))?;
        for it in items {
            self.apply_item(lno, &it, p)?;
        }
        Ok(())
    }

    fn apply_item(&self, lno: usize, it: &SlashItem, p: &mut Pending) -> crate::Result<()> {
        let key = it.name.to_ascii_uppercase();
        let params: Vec<&str> = it
            .content
            .as_deref()
            .map(|c| c.split_whitespace().collect())
            .unwrap_or_default();
        match key.as_str() {
            "FORD" | "RORD" => self.apply_order(lno, &key, it, &params, p),
            "UNITS" => self.apply_units(lno, it, &params, p),
            k if NEXT_UNIT.contains(&k) => Err(self.err(
                lno,
                format!(
                    "the keyword {key} is not read yet; the next unit, docs/17 CHR-02b, reads \
                     LOW, TROE, SRI, REV, DUP and DUPLICATE"
                ),
            )),
            k if BY_NAME.contains(&k) => Err(self.err(
                lno,
                format!("the keyword {key} is refused by name (SPEC-LIT §129.5)"),
            )),
            _ => match self.species_ix.get(it.name.as_str()) {
                Some(&k) => self.apply_efficiency(lno, it, &params, k, p),
                None => Err(self.err(
                    lno,
                    format!(
                        "{} is neither a recognised auxiliary keyword nor a declared species; \
                         the recognised keywords are FORD, RORD, UNITS, LOW, TROE, SRI, REV, \
                         DUP and DUPLICATE, and a declared species name with its efficiency",
                        it.name
                    ),
                )),
            },
        }
    }

    /// `FORD /species order/` or `RORD /species order/` (SPEC-LIT §130.5).
    fn apply_order(
        &self,
        lno: usize,
        key: &str,
        it: &SlashItem,
        params: &[&str],
        p: &mut Pending,
    ) -> crate::Result<()> {
        if it.content.is_none() {
            return Err(self.err(lno, format!("{key} needs a /species order/ field")));
        }
        let [sp, ord] = params else {
            return Err(self.err(
                lno,
                format!(
                    "{key} takes exactly two items, a species and an order; the field holds {}",
                    params.len()
                ),
            ));
        };
        let Some(&k) = self.species_ix.get(*sp) else {
            return Err(self.err(lno, format!("{key}: the species {sp} is not declared in SPECIES")));
        };
        let order = parse_f64(ord).filter(|v| *v >= 0.0);
        let Some(order) = order else {
            return Err(self.err(
                lno,
                format!("{key} {sp}: the order \"{ord}\" is not a finite number at or above zero"),
            ));
        };
        let list = if key == "FORD" {
            &mut p.rx.ford
        } else {
            if !p.rx.reversible {
                return Err(self.err(
                    lno,
                    format!(
                        "RORD on the irreversible reaction {}: an irreversible reaction has no \
                         reverse order",
                        p.rx.equation
                    ),
                ));
            }
            &mut p.rx.rord
        };
        if list.iter().any(|(j, _)| *j == k) {
            return Err(self.err(lno, format!("{key} gives the species {sp} twice")));
        }
        list.push((k, order));
        Ok(())
    }

    /// `UNITS /descriptors/`: overrides the block's units for this reaction only.
    fn apply_units(
        &self,
        lno: usize,
        it: &SlashItem,
        params: &[&str],
        p: &mut Pending,
    ) -> crate::Result<()> {
        if it.content.is_none() || params.is_empty() {
            return Err(self.err(
                lno,
                "UNITS needs one or more descriptors between slashes, such as UNITS /KCAL/"
                    .to_string(),
            ));
        }
        if p.units_seen {
            return Err(self.err(lno, "UNITS is given twice for one reaction".to_string()));
        }
        let mut energy: Option<EnergyUnit> = None;
        let mut a_unit: Option<AUnit> = None;
        for tok in params {
            let t = tok.to_ascii_uppercase();
            let e = match t.as_str() {
                "CAL" => Some(EnergyUnit::Cal),
                "KCAL" => Some(EnergyUnit::Kcal),
                "JOUL" => Some(EnergyUnit::Joul),
                "KJOU" => Some(EnergyUnit::Kjou),
                "KELV" | "KELVIN" => Some(EnergyUnit::Kelv),
                "EVOL" | "EVOLTS" => Some(EnergyUnit::Evol),
                _ => None,
            };
            let a = matches!(t.as_str(), "MOLE" | "MOLECULE").then_some(AUnit::Molecules);
            match (e, a) {
                (Some(u), _) => {
                    if energy.is_some() {
                        return Err(self.err(
                            lno,
                            format!("UNITS names a second energy descriptor, {tok}"),
                        ));
                    }
                    energy = Some(u);
                }
                (None, Some(u)) => {
                    if a_unit.is_some() {
                        return Err(self.err(
                            lno,
                            format!("UNITS names a second A descriptor, {tok}"),
                        ));
                    }
                    a_unit = Some(u);
                }
                (None, None) => {
                    return Err(self.err(
                        lno,
                        format!(
                            "UNITS: {tok} is not a descriptor; the descriptors are MOLE, \
                             MOLECULE, CAL, KCAL, JOUL, KJOU, KELV, KELVIN, EVOL and EVOLTS"
                        ),
                    ));
                }
            }
        }
        p.energy = energy;
        p.a_unit = a_unit;
        p.units_seen = true;
        Ok(())
    }

    /// `SPECIES /alpha/`: an enhanced third-body efficiency (130.3).
    fn apply_efficiency(
        &self,
        lno: usize,
        it: &SlashItem,
        params: &[&str],
        k: usize,
        p: &mut Pending,
    ) -> crate::Result<()> {
        let name = &it.name;
        let Some(tb) = p.rx.third_body.as_mut() else {
            return Err(self.err(
                lno,
                format!(
                    "the efficiency of {name} is given for reaction {}, which has no third \
                     body M",
                    p.rx.equation
                ),
            ));
        };
        if it.content.is_none() {
            return Err(self.err(lno, format!("the efficiency of {name} needs a /number/ field")));
        }
        let [tok] = params else {
            return Err(self.err(
                lno,
                format!(
                    "the efficiency of {name} takes exactly one number; the field holds {}",
                    params.len()
                ),
            ));
        };
        let Some(alpha) = parse_f64(tok).filter(|v| *v >= 0.0) else {
            return Err(self.err(
                lno,
                format!("the efficiency of {name} reads \"{tok}\", which is not a finite number at or above zero"),
            ));
        };
        if tb.efficiencies.iter().any(|(j, _)| *j == k) {
            return Err(self.err(lno, format!("the efficiency of {name} is given twice")));
        }
        tb.efficiencies.push((k, alpha));
        Ok(())
    }

    // ---- a finished reaction ---------------------------------------------

    /// The rate in SI, the exact element balance, and the reversible-with-orders note.
    fn complete(&mut self, p: Pending) -> crate::Result<()> {
        let Pending { mut rx, a, beta, e, energy, a_unit, .. } = p;
        let m = rx.molecularity();
        let theta = theta_of(e, energy.unwrap_or(self.block_energy));
        let a_si = a_si_of(a, a_unit.unwrap_or(self.block_a), m);
        if !a_si.is_finite() || !theta.is_finite() {
            return Err(self.err(
                rx.line,
                format!(
                    "reaction {}: the rate converts to a non-finite SI value (A = {a_si}, \
                     theta = {theta} K)",
                    rx.equation
                ),
            ));
        }
        rx.rate = Arrhenius { a: a_si, beta, theta };
        self.check_balance(&rx)?;
        if rx.reversible && (!rx.ford.is_empty() || !rx.rord.is_empty()) {
            self.notes.push(format!(
                "line {}: reaction {} is reversible and has FORD or RORD with no REV; its \
                 reverse rate from (130.5) is not consistent with equilibrium (SPEC-LIT §130.5)",
                rx.line, rx.equation
            ));
        }
        self.reactions.push(rx);
        Ok(())
    }

    /// `Σ_k a_ek (nu''_k - nu'_k) = 0` exactly, for every element in ELEMENTS order; the
    /// first element with a nonzero residual is refused.
    fn check_balance(&self, rx: &Reaction) -> crate::Result<()> {
        let Some(thermo) = self.built.as_ref() else {
            return Err(self.err(rx.line, "the species thermo is not built".to_string()));
        };
        let records = thermo.species();
        let overflow = || {
            self.err(
                rx.line,
                format!(
                    "reaction {}: the element balance overflows an exact rational",
                    rx.equation
                ),
            )
        };
        for (sym, _) in &self.elements {
            let count = |k: usize| -> i64 {
                records[k]
                    .elements
                    .iter()
                    .find(|(s, _)| s == sym)
                    .map_or(0, |(_, n)| i64::from(*n))
            };
            let mut residual = Rational::new(0, 1).ok_or_else(overflow)?;
            for (k, nu) in &rx.products {
                let term = nu.checked_mul_int(count(*k)).ok_or_else(overflow)?;
                residual = residual.checked_add(term).ok_or_else(overflow)?;
            }
            for (k, nu) in &rx.reactants {
                let term = nu.checked_mul_int(count(*k)).ok_or_else(overflow)?;
                residual = residual.checked_sub(term).ok_or_else(overflow)?;
            }
            if !residual.is_zero() {
                return Err(self.err(
                    rx.line,
                    format!(
                        "reaction {} does not balance: element {sym}: products minus \
                         reactants = {residual} (the balance must be exact, SPEC-LIT §129.4)",
                        rx.equation
                    ),
                ));
            }
        }
        Ok(())
    }

    /// Two reactions are the same when their sides agree, or agree reversed with at least one
    /// of them reversible, and both or neither have a third body (SPEC-LIT §129.5). DUP is
    /// not read yet, so none is permitted.
    fn check_duplicates(&self) -> crate::Result<()> {
        let sorted = |v: &[(usize, Rational)]| {
            let mut s = v.to_vec();
            s.sort_by_key(|(k, _)| *k);
            s
        };
        let keys: Vec<_> = self
            .reactions
            .iter()
            .map(|r| (sorted(&r.reactants), sorted(&r.products)))
            .collect();
        for (j, later) in self.reactions.iter().enumerate() {
            for (i, earlier) in self.reactions.iter().enumerate().take(j) {
                if later.third_body.is_some() != earlier.third_body.is_some() {
                    continue;
                }
                let same = keys[i] == keys[j];
                let swapped = keys[i].0 == keys[j].1
                    && keys[i].1 == keys[j].0
                    && (later.reversible || earlier.reversible);
                if same || swapped {
                    return Err(self.err(
                        later.line,
                        format!(
                            "reaction {} repeats the reaction of line {} ({}); DUP is not read \
                             yet, so no duplicate is permitted (SPEC-LIT §129.5)",
                            later.equation, earlier.line, earlier.equation
                        ),
                    ));
                }
            }
        }
        Ok(())
    }

    fn finish(self) -> crate::Result<Mechanism> {
        let Some(thermo) = self.built else {
            return Err(Error::Config("the mechanism's thermo was never built".to_string()));
        };
        Ok(Mechanism {
            source: self.source.to_string(),
            elements: self.elements,
            species: self.species.into_iter().map(|(n, _)| n).collect(),
            thermo,
            reactions: self.reactions,
            notes: self.notes,
        })
    }
}

/// Why a species name is refused, if it is (SPEC-LIT §129.2).
fn species_refusal(name: &str) -> Option<String> {
    if name == "E" {
        return Some(
            "E is the electron, and this reader has no plasma machinery (SPEC-LIT §129.2)"
                .to_string(),
        );
    }
    if name.eq_ignore_ascii_case("M") || name.eq_ignore_ascii_case("HV") {
        return Some(format!("{name} is a reaction symbol, the third body or the photon"));
    }
    if let Some(c) = name.chars().next().filter(|c| c.is_ascii_digit() || *c == '+' || *c == '=') {
        return Some(format!("a name may not begin with {c}"));
    }
    if name.contains('/') {
        return Some("a name may not contain /".to_string());
    }
    if name.chars().count() > 16 {
        return Some(format!(
            "the name is {} characters and the limit is 16",
            name.chars().count()
        ));
    }
    if name.ends_with(['+', '-']) {
        return Some(
            "a name ending in + or - is an ion, and this reader has no plasma machinery \
             (SPEC-LIT §129.2)"
                .to_string(),
        );
    }
    None
}
