// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.

//! Function1 - a profile of time or of one coordinate (SPEC-LIT 120.1).
//!
//! One function `F(x)` of a single real argument with `n_c` components, read
//! from a patch entry in any of the section's five forms and evaluated here
//! in exactly the operation order (120.1)-(120.11) is written in, so the host
//! twin is bit for bit the device evaluator of run 2: `a*b + c` stays two
//! roundings and every sum keeps its written order. Open questions of range
//! are settled at setup ([`F1::check_range`]), never on the device.
//!
//! Written from elementary mathematics and these open sources, cited with
//! SPEC-LIT 120.1:
//!
//! * linear interpolation <https://en.wikipedia.org/wiki/Linear_interpolation>
//! * the step function <https://en.wikipedia.org/wiki/Step_function>
//! * the sine wave <https://en.wikipedia.org/wiki/Sine_wave>
//! * the pulse wave and its duty cycle <https://en.wikipedia.org/wiki/Pulse_wave>
//! * the ramp function <https://en.wikipedia.org/wiki/Ramp_function>
//! * exponentiation by squaring
//!   <https://en.wikipedia.org/wiki/Exponentiation_by_squaring>
//! * Abramowitz, M. & Stegun, I. A. (1964), Handbook of Mathematical
//!   Functions, NBS AMS 55, 4.3.65-4.3.66 and 4.3.16 (periodicity), open
//!   scan <https://personal.math.ubc.ca/~cbm/aands/>
//! * the Taylor series <https://en.wikipedia.org/wiki/Taylor_series>
//!
//! Provenance: ORIGINAL. Type and keyword NAMES only from the public user
//! guides of SPEC-LIT 120. No GPL-licensed source was consulted.

use std::collections::BTreeMap;
use std::path::Path;

use crate::error::{Error, Result};
use crate::io::fields::PatchFieldSpec;
use crate::io::tokenizer::{Tok, Tokenizer};

/// The largest magnitude a polynomial exponent may have (SPEC-LIT 120.1).
pub const MAX_EXPONENT: i32 = 64;

/// One value; a scalar Function1 (n_c = 1) uses [0] only and leaves [1], [2]
/// at 0.0.
pub type F1Value = [f64; 3];

/// How a table interpolates between its points ((120.2), (120.3)).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Interp {
    Linear,
    Step,
}

/// What a table does with an argument outside [x_0, x_{N-1}] (120.4).
///
/// `Error` and `Warn` are settled on the host at setup
/// ([`F1::check_range`]); the device only ever sees `Clamp` or `Repeat`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OutOfBounds {
    Clamp,
    Error,
    Warn,
    Repeat,
}

/// The five ramp shapes of (120.8); scalar only.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RampKind {
    Linear,
    Quadratic,
    HalfCosine,
    QuarterSine,
    QuarterCosine,
}

/// A Function1: the type names and keywords are SPEC-LIT 120.1's, the
/// definitions are (120.1)-(120.9), evaluated in exactly that order.
#[derive(Debug, Clone, PartialEq)]
pub enum Function1 {
    /// F(x) = v (120.1).
    Constant { v: F1Value },
    /// The piecewise-linear or step table (120.2)-(120.4); x strictly
    /// increasing, y of the entry's rank per point.
    Table {
        x: Vec<f64>,
        y: Vec<F1Value>,
        interp: Interp,
        oob: OutOfBounds,
    },
    /// A generalised power series (120.5): one exponent per term, shared by
    /// every component, |e| <= MAX_EXPONENT.
    Polynomial { c: Vec<F1Value>, e: Vec<i32> },
    /// L + (A*S)*S2(phi) with phi = f*(x - x_s) (120.6); phi is a number of
    /// TURNS, so `frequency` is in hertz.
    Sine {
        amplitude: f64,
        frequency: f64,
        start: f64,
        level: F1Value,
        scale: F1Value,
    },
    /// The pulse (square) wave (120.7); `duty` is m/(1 + m), computed once
    /// here on the host and carried.
    Square {
        amplitude: f64,
        frequency: f64,
        start: f64,
        level: F1Value,
        scale: F1Value,
        duty: f64,
    },
    /// The ramps of (120.8), scalar only.
    Ramp {
        kind: RampKind,
        start: f64,
        duration: f64,
    },
    /// s(x)*v_c(x) (120.9); neither child is itself a scale.
    Scale {
        scale: Box<Function1>,
        value: Box<Function1>,
    },
}

/// One parsed entry: the function, its component count, and the setting name
/// every diagnostic starts with.
#[derive(Debug, Clone, PartialEq)]
pub struct F1 {
    pub n_c: usize,
    pub f: Function1,
    pub setting: String,
}

/// Each Taylor coefficient of (120.11) is the double the division `1.0/n!`
/// rounds to (n! is exact in f64 through 22!, SPEC-LIT 120.1); index k is
/// 1/k!, index 0 unused.
const INV_FACT: [f64; 20] = [
    1.0 / 1.0,
    1.0 / 1.0,
    1.0 / 2.0,
    1.0 / 6.0,
    1.0 / 24.0,
    1.0 / 120.0,
    1.0 / 720.0,
    1.0 / 5040.0,
    1.0 / 40320.0,
    1.0 / 362880.0,
    1.0 / 3628800.0,
    1.0 / 39916800.0,
    1.0 / 479001600.0,
    1.0 / 6227020800.0,
    1.0 / 87178291200.0,
    1.0 / 1307674368000.0,
    1.0 / 20922789888000.0,
    1.0 / 355687428096000.0,
    1.0 / 6402373705728000.0,
    1.0 / 121645100408832000.0,
];

/// (120.10)-(120.11): `(sin(2 pi c), cos(2 pi c))` from basic operations
/// only. `c` is a number of TURNS, which makes the range reduction exact:
/// `c - floor(c)`, `4*u`, `w - q` and `1 - r` are all exact, so the only
/// rounding before the polynomials is the caller's own argument, and the
/// polynomial argument never exceeds `pi/4`, where the truncated Taylor
/// series (next term < 1e-19) is far below an ulp. Neither the device's
/// `sin` nor the host's is called, because the two are different programs
/// that agree to an ulp or two, not bit for bit.
pub fn s2c2(c: f64) -> (f64, f64) {
    let u = c - c.floor();
    let w = 4.0 * u;
    let q = w.floor();
    let r = w - q;
    let q = if q == 4.0 { 0.0 } else { q };

    let h = std::f64::consts::FRAC_PI_2;
    let (s0, c0) = if r <= 0.5 {
        let a = r * h;
        (ps(a), pc(a))
    } else {
        let a = (1.0 - r) * h;
        (pc(a), ps(a))
    };

    match q as i32 {
        1 => (c0, -s0),
        2 => (-s0, -c0),
        3 => (-c0, s0),
        _ => (s0, c0),
    }
}

/// `sin(a)` for `0 <= a <= pi/4`, the truncated Taylor series of A&S 4.3.65
/// in Horner form, innermost coefficient first; every level is one multiply
/// and one add in the written order of (120.11).
fn ps(a: f64) -> f64 {
    let z = a * a;
    let mut acc = -INV_FACT[19];
    acc = INV_FACT[17] + z * acc;
    acc = -INV_FACT[15] + z * acc;
    acc = INV_FACT[13] + z * acc;
    acc = -INV_FACT[11] + z * acc;
    acc = INV_FACT[9] + z * acc;
    acc = -INV_FACT[7] + z * acc;
    acc = INV_FACT[5] + z * acc;
    acc = -INV_FACT[3] + z * acc;
    a * (1.0 + z * acc)
}

/// `cos(a)` for `0 <= a <= pi/4`, A&S 4.3.66 in Horner form, as [`ps`].
fn pc(a: f64) -> f64 {
    let z = a * a;
    let mut acc = -INV_FACT[18];
    acc = INV_FACT[16] + z * acc;
    acc = -INV_FACT[14] + z * acc;
    acc = INV_FACT[12] + z * acc;
    acc = -INV_FACT[10] + z * acc;
    acc = INV_FACT[8] + z * acc;
    acc = -INV_FACT[6] + z * acc;
    acc = INV_FACT[4] + z * acc;
    acc = -INV_FACT[2] + z * acc;
    1.0 + z * acc
}

// ==========================================================================
//  Reading nested lists of numbers
// ==========================================================================

/// One node of a nested list of numbers: a leaf value or a (possibly empty)
/// list.
#[derive(Debug, Clone)]
enum NumNode {
    Num(f64),
    List(Vec<NumNode>),
}

/// Read one node: either `(...)` - whose elements may nest - or a single
/// number. Any other token shape is refused.
fn read_node(tz: &mut Tokenizer) -> Result<NumNode> {
    if tz.is_punct('(') {
        tz.next()?;
        let mut v = Vec::new();
        while !tz.is_punct(')') {
            if tz.done() {
                return tz.err("unterminated '('");
            }
            v.push(read_node(tz)?);
        }
        tz.next()?;
        Ok(NumNode::List(v))
    } else if matches!(
        tz.peek_at(0),
        Some(Tok::Punct('{')) | Some(Tok::Punct('}')) | Some(Tok::Punct(';'))
            | Some(Tok::Punct('[')) | Some(Tok::Punct(']'))
    ) {
        tz.err("expected a number or '('")
    } else {
        Ok(NumNode::Num(tz.expect_num()?))
    }
}

/// Read one top-level value, accepting the optional leading element count
/// OpenFOAM's list form allows (`N ( ... )`); a count that disagrees with the
/// list it introduces is refused.
fn read_toplevel(tz: &mut Tokenizer) -> Result<NumNode> {
    let counted = matches!(tz.peek_at(0), Some(Tok::Num(_))) && tz.is_punct_at('(', 1);
    if !counted {
        return read_node(tz);
    }
    let n = match tz.peek_at(0) {
        Some(Tok::Num(v)) => *v,
        _ => unreachable!("checked above"),
    };
    tz.next()?;
    let node = read_node(tz)?;
    match &node {
        NumNode::List(v) if v.len() as f64 == n => Ok(node),
        NumNode::List(v) => tz.err(format!(
            "a list of {} elements where the leading count said {}",
            v.len(),
            n
        )),
        _ => unreachable!("read_node returned a list for '('"),
    }
}

/// A leaf value of the entry's rank: a bare number when n_c is 1, an
/// `(a b c)` list when n_c is 3 (SPEC-LIT 120.1).
fn node_value(node: &NumNode, n_c: usize, setting: &str, what: &str) -> Result<F1Value> {
    match node {
        NumNode::Num(v) if n_c == 1 => Ok([*v, 0.0, 0.0]),
        NumNode::Num(_) => Err(Error::Config(format!(
            "{setting}: {what} must be (a b c) on a vector entry, not a bare number"
        ))),
        NumNode::List(items) => {
            if items.len() != n_c {
                return Err(Error::Config(format!(
                    "{setting}: {what} has {} components where the entry has {n_c}",
                    items.len()
                )));
            }
            let mut out = [0.0; 3];
            for (i, it) in items.iter().enumerate() {
                out[i] = match it {
                    NumNode::Num(v) => *v,
                    NumNode::List(_) => {
                        return Err(Error::Config(format!(
                            "{setting}: {what} component {i} is a list, not a number"
                        )))
                    }
                };
            }
            Ok(out)
        }
    }
}

/// A rank-n_c value out of one parameter's raw text, skipping any leading
/// dimension set (`[0 1 -1 0 0 0 0]`) and `uniform`/`constant` word.
fn value_text(raw: &str, n_c: usize, setting: &str, what: &str) -> Result<F1Value> {
    let mut tz = Tokenizer::new(raw, setting);
    while tz.is_punct('[') || tz.is_word("uniform") || tz.is_word("constant") {
        if tz.is_punct('[') {
            while !tz.done() && !tz.is_punct(']') {
                tz.next()?;
            }
            tz.expect_punct(']')?;
        } else {
            tz.next()?;
        }
    }
    if tz.done() {
        return Err(Error::Config(format!(
            "{setting}: {what} carries no value"
        )));
    }
    let node = read_toplevel(&mut tz)?;
    node_value(&node, n_c, setting, what)
}

/// The last whitespace-separated token that parses as a number: an OpenFOAM
/// dimension set before the number (`[0 1 -1 0 0 0 0] 2`) drops out, exactly
/// as `FoamDict::scalar` reads one.
fn num_param(raw: &str) -> Option<f64> {
    raw.split_whitespace()
        .filter_map(|t| t.parse::<f64>().ok())
        .next_back()
}

/// OpenFOAM's `Switch`, as `FoamDict::bool` reads it.
fn bool_of(raw: &str) -> bool {
    matches!(
        raw.trim().split_whitespace().next(),
        Some(t) if matches!(
            t.to_ascii_lowercase().as_str(),
            "yes" | "true" | "on" | "y" | "t" | "1"
        )
    )
}

/// A type's parameters, when they live in a sub-dictionary (SPEC-LIT
/// 120.1's forms 4 and 5): every one of them at `<prefix>/<name>`.
#[derive(Debug, Clone, Copy)]
struct DictParams<'a> {
    entries: &'a BTreeMap<String, String>,
    prefix: &'a str,
}

impl DictParams<'_> {
    /// The raw text of one parameter, if the entry exists.
    fn get(&self, name: &str) -> Option<String> {
        self.entries
            .get(&format!("{}/{}", self.prefix, name))
            .cloned()
    }

    /// One scalar parameter, refusing an entry that carries no number rather
    /// than defaulting behind the user's back (SPEC-LIT 13.4).
    fn num(&self, name: &str, setting: &str) -> Result<Option<f64>> {
        match self.get(name) {
            None => Ok(None),
            Some(raw) => match num_param(&raw) {
                Some(v) => Ok(Some(v)),
                None => Err(Error::Config(format!(
                    "{setting}: {name}: \"{raw}\" is not a number"
                ))),
            },
        }
    }

    /// One rank-n_c parameter.
    fn value(
        &self,
        name: &str,
        n_c: usize,
        setting: &str,
    ) -> Result<Option<F1Value>> {
        match self.get(name) {
            None => Ok(None),
            Some(raw) => Ok(Some(value_text(&raw, n_c, setting, name)?)),
        }
    }
}

/// Resolve a `file` parameter against the case directory: a relative path,
/// or one starting `$FOAM_CASE/` or `<case>/`, resolves against `case_dir`;
/// `None` resolves against the current directory. Surrounding quotes an
/// in-place entry may still carry are dropped.
fn resolve_file(file: &str, case_dir: Option<&Path>) -> std::path::PathBuf {
    let bare = file.trim();
    let bare = bare
        .strip_prefix('"')
        .and_then(|s| s.strip_suffix('"'))
        .unwrap_or(bare);
    let rel = bare
        .strip_prefix("$FOAM_CASE/")
        .or_else(|| bare.strip_prefix("<case>/"))
        .unwrap_or(bare);
    match case_dir {
        Some(d) => d.join(rel),
        None => std::path::PathBuf::from(rel),
    }
}

// ==========================================================================
//  The five forms of an entry (SPEC-LIT 120.1)
// ==========================================================================

const KNOWN_TYPES: &[&str] = &[
    "constant",
    "table",
    "tableFile",
    "csvFile",
    "polynomial",
    "sine",
    "square",
    "linearRamp",
    "quadraticRamp",
    "halfCosineRamp",
    "quarterSineRamp",
    "quarterCosineRamp",
    "scale",
];

impl F1 {
    /// Read entry `key` from `entries` (a patch's flattened entries, see
    /// [`patch_entries`]) in any of SPEC-LIT 120.1's five forms.
    /// `Ok(None)` when neither `key` nor `key/type` is present. `n_c` is 1
    /// or 3. `setting` (e.g. "U: boundaryField/inlet/uniformValue") starts
    /// every error message. `case_dir` resolves a relative `file` (and
    /// `$FOAM_CASE/`, `<case>/`); `None` resolves against the current
    /// directory.
    pub fn parse(
        entries: &BTreeMap<String, String>,
        key: &str,
        n_c: usize,
        setting: &str,
        case_dir: Option<&Path>,
    ) -> Result<Option<F1>> {
        if n_c != 1 && n_c != 3 {
            return Err(Error::Config(format!(
                "{setting}: a Function1 has 1 or 3 components, not {n_c}"
            )));
        }

        // Form 5: the type and the parameters in one sub-dictionary.
        let type_key = format!("{key}/type");
        if let Some(raw) = entries.get(&type_key) {
            let name = raw.trim().to_string();
            let params = DictParams {
                entries,
                prefix: key,
            };
            let f = parse_typed(&name, &params, n_c, setting, case_dir)?;
            return Ok(Some(F1 {
                n_c,
                f,
                setting: setting.to_string(),
            }));
        }

        // Forms 1-4: the inline text of the entry itself.
        match entries.get(key) {
            None => Ok(None),
            Some(raw) => {
                let f = parse_inline(raw, key, entries, n_c, setting, case_dir)?;
                Ok(Some(F1 {
                    n_c,
                    f,
                    setting: setting.to_string(),
                }))
            }
        }
    }
}

/// Forms 1-4: the entry's inline text. Form 1 is a bare value, form 2 names
/// `constant` first, form 3 carries the parameters inline after the type
/// name, form 4 is the bare type word with the parameters in `keyCoeffs`.
fn parse_inline(
    raw: &str,
    key: &str,
    entries: &BTreeMap<String, String>,
    n_c: usize,
    setting: &str,
    case_dir: Option<&Path>,
) -> Result<Function1> {
    let mut tz = Tokenizer::new(raw, setting);

    // Form 1: a bare value of the entry's rank - a number, a vector, or a
    // dimension set before either. `uniform <v>` is accepted as a constant
    // too (as form 2 `uniform`).
    if tz.is_punct('(') || tz.is_punct('[') || tz.is_word("uniform")
        || matches!(tz.peek_at(0), Some(Tok::Num(_)))
    {
        let v = value_inline(&mut tz, n_c, setting)?;
        return Ok(Function1::Constant { v });
    }

    let name = tz.expect_word()?;
    if !KNOWN_TYPES.contains(&name.as_str()) {
        return Err(Error::Config(format!(
            "{setting}: unknown Function1 type '{name}'"
        )));
    }
    if tz.done() {
        // Form 4: `K sine;` with the parameters in `KCoeffs { ... }`.
        let params = DictParams {
            entries,
            prefix: &format!("{key}Coeffs"),
        };
        return parse_typed(&name, &params, n_c, setting, case_dir);
    }

    // Form 3: the parameters follow the type name inline.
    match name.as_str() {
        "constant" => {
            let v = value_inline(&mut tz, n_c, setting)?;
            Ok(Function1::Constant { v })
        }
        "table" => {
            let node = read_toplevel(&mut tz)?;
            build_table(&node, n_c, Interp::Linear, OutOfBounds::Clamp, setting)
        }
        "polynomial" => {
            let node = read_toplevel(&mut tz)?;
            build_polynomial(&node, n_c, setting)
        }
        other => Err(Error::Config(format!(
            "{setting}: {other} needs its parameters in a sub-dictionary"
        ))),
    }
}

/// A bare value where the reader sits now: any leading dimension set and
/// `uniform`/`constant` word skipped, then one nested list of numbers.
fn value_inline(tz: &mut Tokenizer, n_c: usize, setting: &str) -> Result<F1Value> {
    while tz.is_punct('[') || tz.is_word("uniform") || tz.is_word("constant") {
        if tz.is_punct('[') {
            while !tz.done() && !tz.is_punct(']') {
                tz.next()?;
            }
            tz.expect_punct(']')?;
        } else {
            tz.next()?;
        }
    }
    let node = read_toplevel(tz)?;
    node_value(&node, n_c, setting, "value")
}

/// Build one typed Function1 whose parameters live at `<prefix>/<name>`
/// (forms 4 and 5 of SPEC-LIT 120.1).
fn parse_typed(
    name: &str,
    params: &DictParams<'_>,
    n_c: usize,
    setting: &str,
    case_dir: Option<&Path>,
) -> Result<Function1> {
    // An argument scaling this reader does not carry (SPEC-LIT 120.1):
    // refused by name whatever the type, before anything else is read.
    if params.get("xScale").is_some() {
        return Err(Error::Config(format!(
            "{setting}: xScale is not carried by this reader"
        )));
    }

    match name {
        "constant" => {
            let v = params.value("value", n_c, setting)?.ok_or_else(|| {
                Error::Config(format!("{setting}: constant needs a 'value' entry"))
            })?;
            Ok(Function1::Constant { v })
        }
        "table" => {
            let raw = params.get("values").ok_or_else(|| {
                Error::Config(format!("{setting}: table needs a 'values' entry"))
            })?;
            let mut tz = Tokenizer::new(&raw, setting);
            let node = read_toplevel(&mut tz)?;
            let interp = interp_of(params, setting)?;
            let oob = oob_of(params, setting)?;
            build_table(&node, n_c, interp, oob, setting)
        }
        "tableFile" => {
            let interp = interp_of(params, setting)?;
            let oob = oob_of(params, setting)?;
            let node = read_data_file(params, setting, case_dir)?;
            build_table(&node, n_c, interp, oob, setting)
        }
        "csvFile" => build_csv(params, n_c, setting, case_dir),
        "polynomial" => {
            let raw = params.get("coeffs").ok_or_else(|| {
                Error::Config(format!("{setting}: polynomial needs a 'coeffs' entry"))
            })?;
            let mut tz = Tokenizer::new(&raw, setting);
            let node = read_toplevel(&mut tz)?;
            build_polynomial(&node, n_c, setting)
        }
        "sine" => build_wave(params, n_c, setting, false),
        "square" => build_wave(params, n_c, setting, true),
        "linearRamp" => build_ramp(RampKind::Linear, params, n_c, setting),
        "quadraticRamp" => build_ramp(RampKind::Quadratic, params, n_c, setting),
        "halfCosineRamp" => build_ramp(RampKind::HalfCosine, params, n_c, setting),
        "quarterSineRamp" => build_ramp(RampKind::QuarterSine, params, n_c, setting),
        "quarterCosineRamp" => build_ramp(RampKind::QuarterCosine, params, n_c, setting),
        "scale" => build_scale(params, n_c, setting, case_dir),
        other => Err(Error::Config(format!(
            "{setting}: unknown Function1 type '{other}'"
        ))),
    }
}

/// A table's `interpolationScheme` (default `linear`, SPEC-LIT 120.1).
fn interp_of(params: &DictParams<'_>, setting: &str) -> Result<Interp> {
    match params.get("interpolationScheme") {
        None => Ok(Interp::Linear),
        Some(s) => match s.trim() {
            "linear" => Ok(Interp::Linear),
            "step" => Ok(Interp::Step),
            other => Err(Error::Config(format!(
                "{setting}: interpolationScheme '{other}' is not linear or step"
            ))),
        },
    }
}

/// A table's `outOfBounds` (default `clamp`, SPEC-LIT 120.1).
fn oob_of(params: &DictParams<'_>, setting: &str) -> Result<OutOfBounds> {
    match params.get("outOfBounds") {
        None => Ok(OutOfBounds::Clamp),
        Some(s) => match s.trim() {
            "clamp" => Ok(OutOfBounds::Clamp),
            "warn" => Ok(OutOfBounds::Warn),
            "error" => Ok(OutOfBounds::Error),
            "repeat" => Ok(OutOfBounds::Repeat),
            other => Err(Error::Config(format!(
                "{setting}: outOfBounds '{other}' is not clamp, warn, error or repeat"
            ))),
        },
    }
}

/// The `( (x y) ... )` list of a `tableFile`, one (120.2) list: the file's
/// text with `//` and `/* */` comments removed. The comments go when the
/// [`Tokenizer`] strips them; the path in an error is the resolved one.
fn read_data_file(
    params: &DictParams<'_>,
    setting: &str,
    case_dir: Option<&Path>,
) -> Result<NumNode> {
    let raw = params
        .get("file")
        .ok_or_else(|| Error::Config(format!("{setting}: the file needs a 'file' entry")))?;
    let path = resolve_file(&raw, case_dir);
    let text = std::fs::read_to_string(&path).map_err(|e| {
        Error::Config(format!("{setting}: cannot read '{}': {e}", path.display()))
    })?;
    let mut tz = Tokenizer::new(&text, &path.display().to_string());
    let node = read_toplevel(&mut tz)?;
    tz.check_scan_error()?;
    Ok(node)
}

/// The table type (120.2)-(120.4) out of a parsed point list.
fn build_table(
    node: &NumNode,
    n_c: usize,
    interp: Interp,
    oob: OutOfBounds,
    setting: &str,
) -> Result<Function1> {
    let points = match node {
        NumNode::List(v) => v,
        _ => {
            return Err(Error::Config(format!(
                "{setting}: table values must be a list of (x y) points"
            )))
        }
    };
    if points.is_empty() {
        return Err(Error::Config(format!("{setting}: table has no points")));
    }

    let mut x = Vec::with_capacity(points.len());
    let mut y = Vec::with_capacity(points.len());
    for p in points {
        let pts = match p {
            NumNode::List(v) => v,
            _ => {
                return Err(Error::Config(format!(
                    "{setting}: each table point is (x y), not a bare number"
                )))
            }
        };
        if pts.len() != 2 {
            return Err(Error::Config(format!(
                "{setting}: each table point is (x y); got {} values",
                pts.len()
            )));
        }
        let px = match &pts[0] {
            NumNode::Num(v) => *v,
            NumNode::List(_) => {
                return Err(Error::Config(format!(
                    "{setting}: the table point's x is a list, not a number"
                )))
            }
        };
        let py = node_value(&pts[1], n_c, setting, "the table point's y")?;
        x.push(px);
        y.push(py);
    }

    for w in x.windows(2) {
        if w[1] <= w[0] {
            return Err(Error::Config(format!(
                "{setting}: table x must be strictly increasing ({} then {})",
                w[0], w[1]
            )));
        }
    }
    if oob == OutOfBounds::Repeat && x.len() < 2 {
        return Err(Error::Config(format!(
            "{setting}: outOfBounds repeat needs two points"
        )));
    }

    Ok(Function1::Table { x, y, interp, oob })
}

/// `csvFile`: a delimited text file read into a table (SPEC-LIT 120.1) -
/// `nHeaderLine` lines skipped, `refColumn` the column of x,
/// `componentColumns` the n_c columns of y, blank lines skipped, every
/// field trimmed. What comes out IS a table, with its `outOfBounds` and
/// `interpolationScheme`.
fn build_csv(
    params: &DictParams<'_>,
    n_c: usize,
    setting: &str,
    case_dir: Option<&Path>,
) -> Result<Function1> {
    let raw = params
        .get("file")
        .ok_or_else(|| Error::Config(format!("{setting}: csvFile needs a 'file' entry")))?;

    // The dictionary is read and checked before the file is touched, so a
    // malformed entry is refused by name rather than as a missing file.
    let n_head = params.num("nHeaderLine", setting)?.unwrap_or(0.0).max(0.0) as usize;
    let ref_col = params.num("refColumn", setting)?.unwrap_or(0.0).max(0.0) as usize;
    let comps_raw = params.get("componentColumns").ok_or_else(|| {
        Error::Config(format!("{setting}: csvFile needs a 'componentColumns' entry"))
    })?;
    let comps = columns_of(&comps_raw, setting)?;
    if comps.len() != n_c {
        return Err(Error::Config(format!(
            "{setting}: componentColumns has {} columns but the entry has {n_c} components",
            comps.len()
        )));
    }

    let sep_raw = params.get("separator").unwrap_or_default();
    let sep = sep_raw.trim().chars().next().unwrap_or(',');
    let merge = bool_of(&params.get("mergeSeparators").unwrap_or_default());

    let path = resolve_file(&raw, case_dir);
    let text = std::fs::read_to_string(&path).map_err(|e| {
        Error::Config(format!("{setting}: cannot read '{}': {e}", path.display()))
    })?;

    let mut x: Vec<f64> = Vec::new();
    let mut y: Vec<F1Value> = Vec::new();
    for line in text.lines().skip(n_head) {
        if line.trim().is_empty() {
            continue;
        }
        let mut fields: Vec<&str> = line.split(sep).collect();
        if merge {
            // A run of separators is one.
            fields.retain(|f| !f.trim().is_empty());
        }
        let need = [ref_col]
            .into_iter()
            .chain(comps.iter().copied())
            .max()
            .unwrap_or(0);
        if fields.len() <= need {
            return Err(Error::Config(format!(
                "{setting}: csv row '{}' has {} fields where column {} was asked for",
                line.trim(),
                fields.len(),
                need
            )));
        }
        let num = |f: &str| -> Result<f64> {
            f.trim().parse::<f64>().map_err(|_| {
                Error::Config(format!(
                    "{setting}: csv field \"{f}\" is not a number"
                ))
            })
        };
        let mut v = [0.0; 3];
        for (c, &col) in comps.iter().enumerate() {
            v[c] = num(&fields[col])?;
        }
        x.push(num(&fields[ref_col])?);
        y.push(v);
    }

    let node = NumNode::List(
        x.iter()
            .zip(&y)
            .map(|(px, py)| {
                NumNode::List(vec![
                    NumNode::Num(*px),
                    NumNode::List(py.iter().take(n_c).map(|v| NumNode::Num(*v)).collect()),
                ])
            })
            .collect(),
    );
    build_table(
        &node,
        n_c,
        interp_of(params, setting)?,
        oob_of(params, setting)?,
        setting,
    )
}

/// The `componentColumns` list: column indices, n_c of them.
fn columns_of(raw: &str, setting: &str) -> Result<Vec<usize>> {
    let mut tz = Tokenizer::new(raw, setting);
    let node = read_toplevel(&mut tz)?;
    let items = match node {
        NumNode::List(v) => v,
        NumNode::Num(_) => {
            return Err(Error::Config(format!(
                "{setting}: componentColumns must be a list of column numbers"
            )))
        }
    };
    items
        .iter()
        .map(|it| match it {
            NumNode::Num(v) => Ok((*v).max(0.0) as usize),
            NumNode::List(_) => Err(Error::Config(format!(
                "{setting}: componentColumns must be a list of column numbers"
            ))),
        })
        .collect()
}

/// The power series (120.5): each term is `(c e)` on a scalar entry and
/// `((c0 c1 c2) e)` - or the flat `(c0 c1 c2 e)` - on a vector one. An
/// exponent is an integer with |e| <= MAX_EXPONENT, refused otherwise.
fn build_polynomial(node: &NumNode, n_c: usize, setting: &str) -> Result<Function1> {
    let terms = match node {
        NumNode::List(v) => v,
        _ => {
            return Err(Error::Config(format!(
                "{setting}: polynomial coeffs must be a list of (c e) terms"
            )))
        }
    };

    let mut c: Vec<F1Value> = Vec::with_capacity(terms.len());
    let mut e: Vec<i32> = Vec::with_capacity(terms.len());
    for term in terms {
        let pts = match term {
            NumNode::List(v) => v,
            _ => {
                return Err(Error::Config(format!(
                    "{setting}: each polynomial term is (c e), not a bare number"
                )))
            }
        };

        // (coefficients..., exponent): the exponent is always the LAST value.
        let mut coeffs: Vec<f64> = Vec::with_capacity(pts.len());
        let exp = match pts.last() {
            Some(NumNode::Num(v)) => *v,
            _ => {
                return Err(Error::Config(format!(
                    "{setting}: each polynomial term ends in its integer exponent"
                )))
            }
        };
        for it in &pts[..pts.len() - 1] {
            match it {
                NumNode::Num(v) => coeffs.push(*v),
                NumNode::List(inner) => {
                    for iv in inner {
                        match iv {
                            NumNode::Num(v) => coeffs.push(*v),
                            NumNode::List(_) => {
                                return Err(Error::Config(format!(
                                    "{setting}: a polynomial coefficient nests too deep"
                                )))
                            }
                        }
                    }
                }
            }
        }
        if coeffs.len() != n_c {
            return Err(Error::Config(format!(
                "{setting}: a polynomial term has {} coefficients where the entry has {n_c}",
                coeffs.len()
            )));
        }
        if exp.fract() != 0.0 || exp.abs() > MAX_EXPONENT as f64 {
            return Err(Error::Config(format!(
                "{setting}: polynomial exponent {exp} must be an integer exponent \
                 with |e| <= {MAX_EXPONENT}"
            )));
        }

        let mut v = [0.0; 3];
        v[..n_c].copy_from_slice(&coeffs);
        c.push(v);
        e.push(exp as i32);
    }

    Ok(Function1::Polynomial { c, e })
}

/// `sine` and `square` (120.6), (120.7): `amplitude` and `frequency` are
/// required, `frequency` > 0, `start` defaults to 0, `level` and `scale`
/// default to 0 and 1 in every component, and `square` carries `markSpace`
/// (default 1, > 0) as the duty m/(1 + m) computed once here.
fn build_wave(
    params: &DictParams<'_>,
    n_c: usize,
    setting: &str,
    square: bool,
) -> Result<Function1> {
    let amplitude = params.num("amplitude", setting)?.ok_or_else(|| {
        Error::Config(format!("{setting}: sine needs an 'amplitude' entry"))
    })?;
    let frequency = params.num("frequency", setting)?.ok_or_else(|| {
        Error::Config(format!("{setting}: sine needs a 'frequency' entry"))
    })?;
    if !(frequency > 0.0) {
        return Err(Error::Config(format!(
            "{setting}: frequency must be > 0 (got {frequency})"
        )));
    }
    let start = params.num("start", setting)?.unwrap_or(0.0);
    let level = params.value("level", n_c, setting)?.unwrap_or([0.0; 3]);
    let scale = params.value("scale", n_c, setting)?.unwrap_or([1.0; 3]);

    if square {
        let m = params.num("markSpace", setting)?.unwrap_or(1.0);
        if !(m > 0.0) {
            return Err(Error::Config(format!(
                "{setting}: markSpace must be > 0 (got {m})"
            )));
        }
        Ok(Function1::Square {
            amplitude,
            frequency,
            start,
            level,
            scale,
            duty: m / (1.0 + m),
        })
    } else {
        Ok(Function1::Sine {
            amplitude,
            frequency,
            start,
            level,
            scale,
        })
    }
}

/// The ramps (120.8): scalar only, `start` default 0, `duration` required
/// and > 0.
fn build_ramp(
    kind: RampKind,
    params: &DictParams<'_>,
    n_c: usize,
    setting: &str,
) -> Result<Function1> {
    if n_c != 1 {
        return Err(Error::Config(format!(
            "{setting}: {} is a scalar Function1 and the entry has {n_c} components",
            kind_name(kind)
        )));
    }
    let start = params.num("start", setting)?.unwrap_or(0.0);
    let duration = params.num("duration", setting)?.ok_or_else(|| {
        Error::Config(format!(
            "{setting}: {} needs a 'duration' entry",
            kind_name(kind)
        ))
    })?;
    if !(duration > 0.0) {
        return Err(Error::Config(format!(
            "{setting}: duration must be > 0 (got {duration})"
        )));
    }
    Ok(Function1::Ramp {
        kind,
        start,
        duration,
    })
}

/// `scale` (120.9): its children are Function1s in their own right, read by
/// [`F1::parse`] at `<prefix>/scale` (scalar) and `<prefix>/value` (the
/// parent's rank); neither child may itself be a scale.
fn build_scale(
    params: &DictParams<'_>,
    n_c: usize,
    setting: &str,
    case_dir: Option<&Path>,
) -> Result<Function1> {
    let scale_key = format!("{}/scale", params.prefix);
    let value_key = format!("{}/value", params.prefix);
    let scale = F1::parse(params.entries, &scale_key, 1, &format!("{setting}/scale"), case_dir)?
        .ok_or_else(|| {
            Error::Config(format!(
                "{setting}: scale needs a 'scale' child (a scalar Function1)"
            ))
        })?;
    let value = F1::parse(params.entries, &value_key, n_c, &format!("{setting}/value"), case_dir)?
        .ok_or_else(|| Error::Config(format!("{setting}: scale needs a 'value' child")))?;

    if matches!(scale.f, Function1::Scale { .. }) || matches!(value.f, Function1::Scale { .. }) {
        return Err(Error::Config(format!("{setting}: scale inside a scale")));
    }
    Ok(Function1::Scale {
        scale: Box::new(scale.f),
        value: Box::new(value.f),
    })
}

/// A ramp's own name, for messages.
fn kind_name(kind: RampKind) -> &'static str {
    match kind {
        RampKind::Linear => "linearRamp",
        RampKind::Quadratic => "quadraticRamp",
        RampKind::HalfCosine => "halfCosineRamp",
        RampKind::QuarterSine => "quarterSineRamp",
        RampKind::QuarterCosine => "quarterCosineRamp",
    }
}

// ==========================================================================
//  Evaluation - (120.1)-(120.11), in the written order
// ==========================================================================

impl F1 {
    /// The host twin: (120.1)-(120.11) at argument `x`. Error/Warn tables
    /// clamp here (settled by [`F1::check_range`]).
    pub fn eval(&self, x: f64) -> F1Value {
        self.f.eval_at(x)
    }

    /// True only for [`Function1::Constant`].
    pub fn is_constant(&self) -> bool {
        matches!(self.f, Function1::Constant { .. })
    }

    /// SPEC-LIT 120.1 "Out of bounds is settled at setup": for every Table
    /// in the tree (recursing into Scale), when `[lo, hi]` is not inside
    /// [x_0, x_{N-1}], `Error` refuses naming the setting, "outOfBounds
    /// error", lo, hi and the table's ends; `Warn` prints that once and
    /// passes; `Clamp` and `Repeat` always pass.
    pub fn check_range(&self, lo: f64, hi: f64) -> Result<()> {
        self.f.check(lo, hi, &self.setting)
    }
}

impl Function1 {
    fn eval_at(&self, x: f64) -> F1Value {
        match self {
            Function1::Constant { v } => *v,
            Function1::Table { x: xs, y, interp, oob } => {
                table_eval(xs, y, *interp, *oob, x)
            }
            Function1::Polynomial { c, e } => {
                // Summed from 0 in k order (120.5); pw by repeated squaring.
                let mut acc = [0.0; 3];
                for (ck, &ek) in c.iter().zip(e.iter()) {
                    let p = if ek < 0 {
                        1.0 / pw(x, ek.unsigned_abs())
                    } else {
                        pw(x, ek as u32)
                    };
                    acc[0] += ck[0] * p;
                    acc[1] += ck[1] * p;
                    acc[2] += ck[2] * p;
                }
                acc
            }
            Function1::Sine { amplitude, frequency, start, level, scale } => {
                // phi = f*(x - x_s); F_c = L_c + (A*S_c)*S2(phi)  (120.6)
                let phi = frequency * (x - start);
                let (s, _) = s2c2(phi);
                [
                    level[0] + (amplitude * scale[0]) * s,
                    level[1] + (amplitude * scale[1]) * s,
                    level[2] + (amplitude * scale[2]) * s,
                ]
            }
            Function1::Square { amplitude, frequency, start, level, scale, duty } => {
                // phi = f*(x - x_s); u = phi - floor(phi); d = m/(1 + m);
                // q = 1 if u < d else -1; F_c = L_c + (A*S_c)*q  (120.7)
                let phi = frequency * (x - start);
                let u = phi - phi.floor();
                let q = if u < *duty { 1.0 } else { -1.0 };
                [
                    level[0] + (amplitude * scale[0]) * q,
                    level[1] + (amplitude * scale[1]) * q,
                    level[2] + (amplitude * scale[2]) * q,
                ]
            }
            Function1::Ramp { kind, start, duration } => {
                // u = (x - x_s)/D, then u = 0 if u < 0, u = 1 if u > 1 (120.8)
                let mut u = (x - start) / duration;
                if u < 0.0 {
                    u = 0.0;
                }
                if u > 1.0 {
                    u = 1.0;
                }
                let v = match kind {
                    RampKind::Linear => u,
                    RampKind::Quadratic => u * u,
                    RampKind::HalfCosine => {
                        let (_, c2) = s2c2(0.5 * u);
                        0.5 * (1.0 - c2)
                    }
                    RampKind::QuarterSine => {
                        let (s2, _) = s2c2(0.25 * u);
                        s2
                    }
                    RampKind::QuarterCosine => {
                        let (_, c2) = s2c2(0.25 * u);
                        1.0 - c2
                    }
                };
                [v, 0.0, 0.0]
            }
            Function1::Scale { scale, value } => {
                // F_c(x) = s(x)*v_c(x)  (120.9)
                let s = scale.eval_at(x)[0];
                let v = value.eval_at(x);
                [s * v[0], s * v[1], s * v[2]]
            }
        }
    }

    /// [`F1::check_range`]'s walk over the tree; `setting` names the whole
    /// entry in the refusal.
    fn check(&self, lo: f64, hi: f64, setting: &str) -> Result<()> {
        match self {
            Function1::Table { x, oob, .. } => {
                let inside = lo >= x[0] && hi <= x[x.len() - 1];
                if inside || matches!(oob, OutOfBounds::Clamp | OutOfBounds::Repeat) {
                    return Ok(());
                }
                let msg = format!(
                    "{setting}: outOfBounds error: the argument range [{lo}, {hi}] is not \
                     inside the table's [{}, {}]",
                    x[0],
                    x[x.len() - 1]
                );
                if *oob == OutOfBounds::Warn {
                    crate::io::contract::warn_once(setting, &msg);
                    return Ok(());
                }
                Err(Error::Config(msg))
            }
            Function1::Scale { scale, value } => {
                scale.check(lo, hi, setting)?;
                value.check(lo, hi, setting)
            }
            _ => Ok(()),
        }
    }
}

/// (120.2)-(120.4): map the argument by the out-of-bounds rule, then take
/// the end points or interpolate. `Error`/`Warn` arrive here already
/// settled, so they run as `Clamp` - the end rule of (120.2) does it.
fn table_eval(xs: &[f64], y: &[F1Value], interp: Interp, oob: OutOfBounds, x: f64) -> F1Value {
    let n = xs.len();
    let xp = if oob == OutOfBounds::Repeat && n >= 2 {
        // r = fmod(x - x_0, P); if r < 0 then r = r + P; x' = x_0 + r (120.4)
        let x0 = xs[0];
        let p = xs[n - 1] - x0;
        let mut r = (x - x0) % p;
        if r < 0.0 {
            r += p;
        }
        x0 + r
    } else {
        x
    };

    if xp <= xs[0] {
        return y[0];
    }
    if xp >= xs[n - 1] {
        return y[n - 1];
    }

    // x_0 < xp < x_{N-1}: the bracket j with x_j <= xp < x_{j+1}.
    let j = xs.partition_point(|&v| v <= xp) - 1;
    match interp {
        Interp::Step => y[j],
        Interp::Linear => {
            let t = (xp - xs[j]) / (xs[j + 1] - xs[j]);
            [
                y[j][0] + t * (y[j + 1][0] - y[j][0]),
                y[j][1] + t * (y[j + 1][1] - y[j][1]),
                y[j][2] + t * (y[j + 1][2] - y[j][2]),
            ]
        }
    }
}

/// `x^e` by exponentiation by squaring (120.5), least-significant bit
/// first: the result picks up one multiply per set bit, in order from the
/// bottom, and the base squares once between them. `pw(x, 0)` is 1.
fn pw(x: f64, e: u32) -> f64 {
    let mut result = 1.0;
    let mut base = x;
    let mut n = e;
    while n > 0 {
        if n & 1 == 1 {
            result *= base;
        }
        n >>= 1;
        if n > 0 {
            base *= base;
        }
    }
    result
}

/// A patch's entries flattened: `extra` as is, plus every `dicts[k]` parsed
/// with [`crate::io::dict::FoamDict::parse`] and inserted as `k/<sub>`
/// (skipping FoamDict's empty-dictionary marker keys that end in '/').
pub fn patch_entries(
    spec: &PatchFieldSpec,
    setting: &str,
) -> Result<BTreeMap<String, String>> {
    let mut out = spec.extra.clone();
    for (k, body) in &spec.dicts {
        let d = crate::io::dict::FoamDict::parse(body, &format!("{setting}/{k}"))?;
        for (key, val) in &d.entries {
            if key.ends_with('/') {
                continue;
            }
            out.insert(format!("{k}/{key}"), val.clone());
        }
    }
    Ok(out)
}

mod device;
pub use device::*;

#[cfg(test)]
mod tests;
