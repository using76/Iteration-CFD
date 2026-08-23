// meteor-cfd - Copyright (c) 2026 주식회사 메테오시뮬레이션 (Meteo Simulation Co., Ltd.)
// Source-available, not Open Source. Educational use is free; research,
// publication and commercial use require a licence - simul@msimul.com
// See LICENSE at the repository root.

//! `FoamDict` — any OpenFOAM dictionary, flattened.
//!
//! Nesting is thrown away and rebuilt into the key: `solvers { p { relTol
//! 0.01; } }` becomes the single entry `solvers/p/relTol`. That loses nothing
//! a solver setup needs and makes every lookup a one-liner, which matters
//! because the alternative — a recursive dictionary type — would have every
//! call site walking it by hand.
//!
//! The value is the entry's tokens rejoined with single spaces, minus the
//! terminating `;`:
//!
//! ```text
//! dimensions      [0 1 -1 0 0 0 0];   ->  "[0 1 -1 0 0 0 0]"
//! internalField   uniform (0 0 0);    ->  "uniform (0 0 0)"
//! default         Gauss linear;       ->  "Gauss linear"
//! ```
//!
//! A number is rejoined from its parsed value, not from the source text, so
//! `tolerance 1e-06` reads back as `0.000001`. That is deliberate: keeping the
//! source spelling would mean a `String` per token, and the same tokeniser has
//! to survive a `points` file with millions of them. The value round-trips
//! exactly, and an integer prints without a `.0`, so a dimension set is
//! unchanged.
//!
//! # Pattern keys
//!
//! A key written in quotes is a POSIX extended regular expression, not a name
//! - `boundaryField { ".*" { type zeroGradient; } }` is the commonest idiom
//! in the whole format. Such a key is flattened WITH its quotes, so
//! `solvers/"(U|k|epsilon)"/tolerance` is a different key from
//! `solvers/(U|k|epsilon)/tolerance`, and an exact lookup can never
//! accidentally hit a pattern. [`FoamDict::resolve`] is what turns a name into
//! the key that governs it: exact match first, then the patterns in **file**
//! order. That ordering is why [`FoamDict`] carries a `patterns` vector at all
//! - `entries` is a `BTreeMap` and has thrown the file order away.
//!
//! Provenance: carried across from this project's own earlier C++ I/O layer
//! when the crate moved to Rust. That C++ was written from the case format as
//! it appears in data files - not from any CFD code's source - and the format
//! itself, not another program, is the specification here. The pattern-key
//! resolution was written from ofgpu `SPEC-LIT.md` §13.4 and the POSIX ERE
//! definition; see [`crate::io::regex`]. No GPL-licensed source was consulted.

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

use super::regex::Regex;
use super::tokenizer::{check_ascii_format, slurp, Tok, Tokenizer, BINARY_MSG};
use crate::error::{parse_err, Error, Result};
use crate::{Label, Scalar};

/// How deep `#include` may nest before we call it a cycle. OpenFOAM has no
/// limit; a self-including file would otherwise blow the stack.
const MAX_INCLUDE_DEPTH: u32 = 31;

/// Every entry in the file, keyed by its flattened path.
///
/// A sub-dictionary with no entries of its own is recorded as the marker key
/// `"name/"` with an empty value, so [`FoamDict::sub_keys`] can still report
/// it — an empty `boundaryField` patch entry is legal and must not vanish.
#[derive(Debug, Default, Clone)]
pub struct FoamDict {
    pub entries: BTreeMap<String, String>,

    /// Flattened paths of the QUOTED keys, in the order the file wrote them:
    /// `["solvers/\"(U|k|epsilon)\"", "boundaryField/\".*\""]`.
    ///
    /// `entries` is sorted, so it cannot answer "which pattern did the author
    /// write first" - and that is exactly the tie-break OpenFOAM uses when
    /// two patterns match the same name.
    pub patterns: Vec<String>,
}

impl FoamDict {
    pub fn read(path: &Path) -> Result<Self> {
        let src = slurp(path)?;
        Self::parse(&src, &path.display().to_string())
    }

    /// `path` is used for error messages and to resolve `#include` relative to
    /// the file's own directory, exactly as OpenFOAM does.
    pub fn parse(src: &str, path: &str) -> Result<Self> {
        check_ascii_format(src, path)?;

        let mut entries: BTreeMap<String, String> = BTreeMap::new();
        let mut patterns: Vec<String> = Vec::new();
        let mut ts = Tokenizer::new(src, path);
        let ctx = ParseCtx {
            dir: dir_name(path),
            depth: 0,
        };

        // The FoamFile header is NOT skipped: it lands under "FoamFile/..."
        // so a caller can check `class` without a second pass over the file.
        parse_body(&mut ts, "", &mut entries, &mut patterns, false, &ctx)?;
        ts.check_scan_error()?;

        let d = FoamDict { entries, patterns };

        // check_ascii_format already looked at the raw text; this catches the
        // header written in some layout the string search did not match.
        if d.get_or("FoamFile/format", "").contains("binary") {
            return parse_err(path, BINARY_MSG);
        }
        Ok(d)
    }

    pub fn has(&self, k: &str) -> bool {
        self.entries.contains_key(k)
    }

    pub fn get(&self, k: &str) -> Option<&str> {
        self.entries.get(k).map(|s| s.as_str())
    }

    pub fn get_or<'a>(&'a self, k: &str, d: &'a str) -> &'a str {
        self.get(k).unwrap_or(d)
    }

    /// OpenFOAM writes dimensioned scalars as `nu [0 2 -1 0 0 0 0] 1e-05`, so
    /// the number wanted is always the LAST token that parses as one. A
    /// missing or unreadable entry falls back to `d` rather than failing,
    /// because every caller has a physical default.
    pub fn scalar(&self, k: &str, d: Scalar) -> Scalar {
        match self.entries.get(k) {
            None => d,
            Some(raw) => last_number(raw).map(|v| v as Scalar).unwrap_or(d),
        }
    }

    /// Same rule as [`FoamDict::scalar`], then rounded. Reading through f64 is
    /// deliberate: `nCorrectors 2` and `nCorrectors 2.0` both occur in the
    /// wild, and OpenFOAM accepts both.
    pub fn label(&self, k: &str, d: Label) -> Label {
        match self.entries.get(k) {
            None => d,
            // `+0.5` then truncate, matching the C++. Only ever applied to
            // counts, where the value is non-negative.
            Some(raw) => match last_number(raw) {
                Some(v) => (v + 0.5) as Label,
                None => d,
            },
        }
    }

    /// OpenFOAM's `Switch`: `yes/true/on/y/t/1` against `no/false/off/n/f/0`.
    pub fn bool(&self, k: &str, d: bool) -> bool {
        let Some(raw) = self.entries.get(k) else {
            return d;
        };
        let Some(tok) = raw.split_whitespace().next() else {
            return d;
        };
        match tok.to_ascii_lowercase().as_str() {
            "yes" | "true" | "on" | "y" | "t" | "1" => true,
            "no" | "false" | "off" | "n" | "f" | "0" => false,
            _ => d,
        }
    }

    // ----------------------------------------------------------------
    //  Pattern keys
    // ----------------------------------------------------------------

    /// Which key inside `prefix` governs `name`: the exact one if the file
    /// wrote it, otherwise the first QUOTED key, in file order, whose regular
    /// expression matches.
    ///
    /// Returns the key exactly as it is stored - a pattern comes back with
    /// its quotes - so the caller can build a full path from it:
    ///
    /// ```text
    /// let key = d.resolve("solvers", "epsilon")?;      // Some("\"(U|k|epsilon)\"")
    /// d.scalar(&format!("solvers/{key}/tolerance"), 1e-8)
    /// ```
    ///
    /// Exact-wins-over-pattern is the rule OpenFOAM uses, and it is the one
    /// that makes `"(U|k|epsilon)" {...}  epsilon { relTol 0; }` mean what it
    /// looks like it means.
    ///
    /// A malformed pattern is an error rather than a non-match: a key the
    /// author wrote in quotes was meant to match something, and quietly
    /// matching nothing is the failure this whole module exists to remove.
    pub fn resolve(&self, prefix: &str, name: &str) -> Result<Option<String>> {
        let p = with_slash(prefix);

        if self.has(&format!("{p}{name}")) || self.dict_exists(&format!("{p}{name}")) {
            return Ok(Some(name.to_string()));
        }

        for path in &self.patterns {
            let Some(seg) = path.strip_prefix(&p) else {
                continue;
            };
            // Only the patterns AT this level, not those in a sub-dictionary.
            if seg.contains('/') {
                continue;
            }
            let Some(pattern) = seg.strip_prefix('"').and_then(|s| s.strip_suffix('"')) else {
                continue;
            };

            let re = Regex::new(pattern).map_err(Error::Config)?;
            if re.is_match(name) {
                return Ok(Some(seg.to_string()));
            }
        }

        Ok(None)
    }

    /// True when `path` names a sub-dictionary - one with entries, or the
    /// marker key an empty one leaves behind.
    pub fn dict_exists(&self, path: &str) -> bool {
        let p = with_slash(path);
        if self.entries.contains_key(&p) {
            return true;
        }
        self.entries
            .range(p.clone()..)
            .next()
            .is_some_and(|(k, _)| k.starts_with(&p))
    }

    /// Every pattern key written directly inside `prefix`, in file order,
    /// with its quotes. For diagnostics and for tests.
    pub fn patterns_in(&self, prefix: &str) -> Vec<String> {
        let p = with_slash(prefix);
        self.patterns
            .iter()
            .filter_map(|path| path.strip_prefix(&p))
            .filter(|seg| !seg.contains('/'))
            .map(|seg| seg.to_string())
            .collect()
    }

    /// Immediate sub-keys of a dictionary path, in file-sorted order.
    /// `sub_keys("")` lists the top level; `sub_keys("boundaryField")` lists
    /// the patch names.
    pub fn sub_keys(&self, prefix: &str) -> Vec<String> {
        let mut p = prefix.to_string();
        if !p.is_empty() && !p.ends_with('/') {
            p.push('/');
        }

        let mut keys = Vec::new();
        let mut seen = BTreeSet::new();

        for k in self.entries.range(p.clone()..).map(|(k, _)| k) {
            // The marker key of the dictionary itself ("a/b/" when asked for
            // "a/b") is not one of its children.
            if k.len() <= p.len() {
                continue;
            }
            if !k.starts_with(&p) {
                break;
            }

            let rest = &k[p.len()..];
            let seg = match rest.find('/') {
                Some(s) => &rest[..s],
                None => rest,
            };
            if seg.is_empty() {
                continue;
            }
            if seen.insert(seg.to_string()) {
                keys.push(seg.to_string());
            }
        }

        keys
    }
}

// ==========================================================================
//  Parsing
// ==========================================================================

struct ParseCtx {
    /// Directory of the file being parsed, so `#include` resolves relative to
    /// it and not to the process's working directory.
    dir: String,
    depth: u32,
}

fn parse_body(
    ts: &mut Tokenizer,
    prefix: &str,
    out: &mut BTreeMap<String, String>,
    patterns: &mut Vec<String>,
    in_block: bool,
    ctx: &ParseCtx,
) -> Result<()> {
    loop {
        if ts.done() {
            if in_block {
                return ts.err("unterminated '{'");
            }
            return Ok(());
        }
        if ts.is_punct('}') {
            if !in_block {
                return ts.err("unexpected '}'");
            }
            ts.next()?;
            return Ok(());
        }
        // A stray ';' after a sub-dictionary is legal and common.
        if ts.is_punct(';') {
            ts.next()?;
            continue;
        }
        if ts.peek_at(0).is_some_and(|t| t.is_punct_any()) {
            return ts.err("expected a keyword");
        }

        // A QUOTED key is a regular expression, not a name, and the two must
        // stay distinguishable: the quotes are kept in the flattened path so
        // that `resolve` can tell a pattern from a patch called `.*`.
        let (raw_key, quoted) = ts.expect_key()?;
        let key = if quoted {
            format!("\"{raw_key}\"")
        } else {
            raw_key.clone()
        };

        // ---- preprocessor directives ---------------------------------------
        if key.starts_with('#') {
            handle_directive(ts, &key, prefix, out, patterns, ctx)?;
            continue;
        }

        // ---- $dict merge ----------------------------------------------------
        if key.len() > 1 && key.starts_with('$') {
            if ts.is_punct(';') {
                ts.next()?;
            }
            merge_var(out, prefix, &key[1..]);
            continue;
        }

        // ---- sub-dictionary --------------------------------------------------
        if ts.is_punct('{') {
            ts.next()?;
            if quoted {
                patterns.push(format!("{}{}", prefix, key));
            }
            let sub = format!("{}{}/", prefix, key);
            let before = out.len();
            parse_body(ts, &sub, out, patterns, true, ctx)?;
            if out.len() == before {
                out.insert(sub, String::new());
            }
            continue;
        }

        // ---- leaf entry -------------------------------------------------------
        let mut v = String::new();
        while !ts.done() && !ts.is_punct(';') {
            if ts.is_punct('}') {
                return ts.err(format!("missing ';' for entry '{}'", key));
            }
            let tk = match ts.next()? {
                Some(t) => t,
                None => break,
            };
            let text = match &tk {
                Tok::Word(w) if w.len() > 1 && w.starts_with('$') => {
                    lookup_var(out, prefix, &w[1..])
                }
                other => other.to_string(),
            };
            append_raw(&mut v, &tk, &text);
        }
        if ts.done() {
            return ts.err(format!("expected ';' for entry '{}'", key));
        }
        ts.next()?;

        if quoted {
            patterns.push(format!("{}{}", prefix, key));
        }
        out.insert(format!("{}{}", prefix, key), v);
    }
}

#[allow(clippy::too_many_arguments)]
fn handle_directive(
    ts: &mut Tokenizer,
    key: &str,
    prefix: &str,
    out: &mut BTreeMap<String, String>,
    patterns: &mut Vec<String>,
    ctx: &ParseCtx,
) -> Result<()> {
    if key == "#include" || key == "#includeIfPresent" {
        let f = ts.expect_word()?;
        // The file name carries no ';' in some cases and one in others.
        if ts.is_punct(';') {
            ts.next()?;
        }

        let p = if is_absolute(&f) {
            f
        } else {
            format!("{}/{}", ctx.dir, f)
        };

        if Path::new(&p).exists() {
            if ctx.depth > MAX_INCLUDE_DEPTH {
                return parse_err(ts.path(), format!("#include nested too deeply at {}", p));
            }
            let src = slurp(Path::new(&p))?;
            check_ascii_format(&src, &p)?;
            let mut sub_ts = Tokenizer::new(&src, &p);
            // An included file may carry its own FoamFile header; only the
            // outermost file's header is kept.
            sub_ts.skip_header()?;
            let sub_ctx = ParseCtx {
                dir: dir_name(&p),
                depth: ctx.depth + 1,
            };
            parse_body(&mut sub_ts, prefix, out, patterns, false, &sub_ctx)?;
            sub_ts.check_scan_error()?;
        } else if key == "#include" {
            return parse_err(ts.path(), format!("#include cannot open {}", p));
        }
        return Ok(());
    }

    // #includeEtc, #calc, #codeStream, ...: nothing here can evaluate them,
    // so skip the entry and say so once.
    eprintln!(
        "[ofgpu] {}: ignoring unsupported directive {}",
        ts.path(),
        key
    );

    // A `#include`-shaped directive carries a bare file name and NO ';', so
    // blindly skipping to the next ';' would swallow the entry that follows.
    if key.starts_with("#include") {
        if !ts.done() && !ts.peek_at(0).is_some_and(|t| t.is_punct_any()) {
            ts.next()?;
        }
        if ts.is_punct(';') {
            ts.next()?;
        }
        return Ok(());
    }

    if !ts.done() && !ts.is_punct('}') {
        ts.skip_entry()?;
    }
    Ok(())
}

/// Append one token to a raw value string with OpenFOAM-ish spacing, so
/// `[0 1 -1 0 0 0 0]` and `uniform (0 0 0)` come back out readable.
fn append_raw(v: &mut String, tk: &Tok, text: &str) {
    let opens = matches!(v.chars().last(), Some('(') | Some('['));
    let closes = tk.is_punct(')') || tk.is_punct(']');

    if !v.is_empty() && !opens && !closes {
        v.push(' ');
    }
    v.push_str(text);
}

/// A dictionary path with exactly one trailing `/`, and `""` left as `""`.
fn with_slash(prefix: &str) -> String {
    if prefix.is_empty() || prefix.ends_with('/') {
        prefix.to_string()
    } else {
        format!("{prefix}/")
    }
}

/// Strip one level off an `"a/b/"` scope, returning `""` at the top.
fn parent_scope(scope: &str) -> String {
    if scope.is_empty() {
        return String::new();
    }
    // The trailing '/' is the scope's own separator, so search before it.
    match scope[..scope.len() - 1].rfind('/') {
        Some(s) => scope[..s + 1].to_string(),
        None => String::new(),
    }
}

/// OpenFOAM's `$var` lookup: the current scope first, then outward.
/// An unresolved name is left visible in the value rather than dropped.
fn lookup_var(out: &BTreeMap<String, String>, prefix: &str, name: &str) -> String {
    let mut scope = prefix.to_string();
    loop {
        if let Some(v) = out.get(&format!("{}{}", scope, name)) {
            return v.clone();
        }
        if scope.is_empty() {
            break;
        }
        scope = parent_scope(&scope);
    }
    format!("${}", name)
}

/// `$p;` on its own line copies dictionary `p` into the current scope.
///
/// Entries already present WIN, which is what makes the
/// `pFinal { $p; relTol 0; }` idiom behave: `$p` fills in what is missing and
/// the explicit `relTol` that follows overwrites it.
fn merge_var(out: &mut BTreeMap<String, String>, prefix: &str, name: &str) {
    let mut scope = prefix.to_string();
    loop {
        let src = format!("{}{}/", scope, name);

        let add: Vec<(String, String)> = out
            .range(src.clone()..)
            .take_while(|(k, _)| k.starts_with(&src))
            .map(|(k, v)| (format!("{}{}", prefix, &k[src.len()..]), v.clone()))
            .collect();

        if !add.is_empty() {
            for (k, v) in add {
                out.entry(k).or_insert(v);
            }
            return;
        }

        if let Some(v) = out.get(&format!("{}{}", scope, name)).cloned() {
            out.entry(format!("{}{}", prefix, name)).or_insert(v);
            return;
        }

        if scope.is_empty() {
            return;
        }
        scope = parent_scope(&scope);
    }
}

/// Both separators, because a case path on Windows mixes them freely.
fn dir_name(path: &str) -> String {
    match path.rfind(['/', '\\']) {
        Some(s) => path[..s].to_string(),
        None => ".".to_string(),
    }
}

/// A leading `/`, or a Windows drive letter.
fn is_absolute(p: &str) -> bool {
    let mut c = p.chars();
    match (c.next(), c.next()) {
        (Some('/'), _) => true,
        (_, Some(':')) => true,
        _ => false,
    }
}

/// The last whitespace-separated token that parses as a number.
fn last_number(raw: &str) -> Option<f64> {
    let mut found = None;
    for tok in raw.split_whitespace() {
        if let Ok(v) = tok.parse::<f64>() {
            found = Some(v);
        }
    }
    found
}

// ==========================================================================
//  Tests
// ==========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// A cut-down but structurally real case dictionary: banner, header with
    /// a quoted path, a dimension set, an fvSchemes-style key, a regex solver
    /// key, and an empty sub-dictionary.
    const SRC: &str = r#"/*--------------------------------*- C++ -*----------------------------------*\
  =========                 |
\*---------------------------------------------------------------------------*/
FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    location    "constant/polyMesh";
    object      fvSolution;
}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

dimensions      [0 1 -1 0 0 0 0];
internalField   uniform (0 0 0);

divSchemes
{
    default         none;
    div(phi,U)      bounded Gauss limitedLinear 1;   // key must not split
}

solvers
{
    "(U|k|epsilon)"
    {
        solver          smoothSolver;
        tolerance       1e-05;
        nSweeps         2;
    }

    p
    {
        solver          GAMG;
        tolerance       1e-06;
        relTol          0.01;
    }

    pFinal
    {
        $p;
        relTol          0;
    }
}

nu              nu [0 2 -1 0 0 0 0] 1e-05;
turbulence      on;

emptyDict
{
}
"#;

    fn d() -> FoamDict {
        match FoamDict::parse(SRC, "case/system/fvSolution") {
            Ok(d) => d,
            Err(e) => panic!("{e}"),
        }
    }

    #[test]
    fn nesting_is_flattened_into_the_key() {
        let d = d();
        assert_eq!(d.get("solvers/p/solver"), Some("GAMG"));
        assert_eq!(d.get("divSchemes/default"), Some("none"));
        assert!(d.has("solvers/p/relTol"));
        assert!(!d.has("solvers/p"));
    }

    /// If this splits, every scheme lookup misses and the run silently uses
    /// a different discretisation.
    #[test]
    fn an_fvschemes_key_keeps_its_parentheses() {
        let d = d();
        assert_eq!(
            d.get("divSchemes/div(phi,U)"),
            Some("bounded Gauss limitedLinear 1")
        );
    }

    /// A quoted key keeps its quotes in the flattened path, which is what
    /// makes it a PATTERN and not a name. Losing that distinction would let a
    /// patch genuinely called `p` be found by the key `"p.*"`.
    #[test]
    fn a_regex_solver_key_survives_as_a_pattern() {
        let d = d();
        assert_eq!(
            d.get("solvers/\"(U|k|epsilon)\"/solver"),
            Some("smoothSolver")
        );
        assert_eq!(
            d.sub_keys("solvers/\"(U|k|epsilon)\""),
            vec!["nSweeps", "solver", "tolerance"]
        );
        assert_eq!(d.patterns_in("solvers"), vec!["\"(U|k|epsilon)\""]);
    }

    /// The whole point of the pattern machinery: `"(U|k|epsilon)"` in
    /// fvSolution must govern the `epsilon` equation. Before this existed the
    /// lookup missed and every one of those equations silently ran at the
    /// built-in default tolerance.
    #[test]
    fn resolve_finds_the_pattern_that_governs_a_name() {
        let d = d();

        let key = d.resolve("solvers", "epsilon").ok().flatten();
        assert_eq!(key.as_deref(), Some("\"(U|k|epsilon)\""));
        assert!((d.scalar(&format!("solvers/{}/tolerance", key.unwrap_or_default()), 0.0)
            as f64
            - 1e-5)
            .abs()
            < 1e-11);

        // An exact key wins over a pattern that would also match.
        assert_eq!(d.resolve("solvers", "p").ok().flatten().as_deref(), Some("p"));

        // Nothing matches `omega`, and saying so is the point: the caller
        // then knows it is using its own defaults.
        assert_eq!(d.resolve("solvers", "omega").ok().flatten(), None);
    }

    #[test]
    fn a_malformed_pattern_is_an_error_not_a_silent_miss() {
        let d = match FoamDict::parse(r#"solvers { "(unclosed" { relTol 0; } }"#, "fvSolution") {
            Ok(d) => d,
            Err(e) => panic!("{e}"),
        };
        assert!(d.resolve("solvers", "p").is_err());
    }

    #[test]
    fn a_dimension_set_rejoins_exactly() {
        let d = d();
        assert_eq!(d.get("dimensions"), Some("[0 1 -1 0 0 0 0]"));
        assert_eq!(d.get("internalField"), Some("uniform (0 0 0)"));
    }

    #[test]
    fn the_header_is_kept_and_its_quoted_slash_is_not_a_comment() {
        let d = d();
        assert_eq!(d.get("FoamFile/location"), Some("constant/polyMesh"));
        assert_eq!(d.get("FoamFile/object"), Some("fvSolution"));
        assert_eq!(d.get("FoamFile/format"), Some("ascii"));
    }

    /// An empty sub-dictionary has no entries to prove it exists, so the
    /// marker key is the only thing standing between it and disappearing.
    #[test]
    fn an_empty_sub_dictionary_is_still_visible() {
        let d = d();
        assert!(d.has("emptyDict/"));
        assert_eq!(d.get("emptyDict/"), Some(""));
        assert!(d.sub_keys("").iter().any(|k| k == "emptyDict"));
        assert!(d.sub_keys("emptyDict").is_empty());
    }

    #[test]
    fn sub_keys_lists_immediate_children_only() {
        let d = d();
        assert_eq!(
            d.sub_keys("solvers"),
            vec!["\"(U|k|epsilon)\"", "p", "pFinal"]
        );
        let top = d.sub_keys("");
        assert!(top.contains(&"divSchemes".to_string()));
        assert!(top.contains(&"solvers".to_string()));
        assert!(top.contains(&"FoamFile".to_string()));
        assert!(!top.iter().any(|k| k.contains('/')));
    }

    /// `nu [0 2 -1 0 0 0 0] 1e-05` must not read as 0 — it is the viscosity,
    /// and a 0 there would take the whole run laminar without saying so.
    #[test]
    fn a_dimensioned_scalar_takes_the_last_number() {
        let d = d();
        assert!(
            d.get_or("nu", "").starts_with("nu [0 2 -1 0 0 0 0] "),
            "{:?}",
            d.get("nu")
        );
        assert!((d.scalar("nu", 0.0) as f64 - 1e-5).abs() < 1e-11);
        assert_eq!(d.scalar("missing", 42.0), 42.0);
        assert_eq!(d.label("solvers/\"(U|k|epsilon)\"/nSweeps", 1), 2);
        assert!(d.bool("turbulence", false));
        assert!(!d.bool("missing", false));
    }

    /// `$p` fills in what pFinal does not say; what pFinal does say wins.
    #[test]
    fn a_dollar_merge_does_not_overwrite_the_explicit_entry() {
        let d = d();
        assert_eq!(d.get("solvers/pFinal/solver"), Some("GAMG"));
        assert!((d.scalar("solvers/pFinal/tolerance", 0.0) as f64 - 1e-6).abs() < 1e-12);
        assert_eq!(d.scalar("solvers/pFinal/relTol", 1.0), 0.0);
        assert_eq!(d.scalar("solvers/p/relTol", 0.0), 0.01);
    }

    /// Pinning the re-spelling so nobody is surprised by it: the value is the
    /// same number, spelled the way Rust prints it.
    #[test]
    fn a_float_is_rejoined_from_its_value_not_its_source_text() {
        let d = d();
        assert_eq!(d.get("solvers/p/tolerance"), Some("0.000001"));
        assert_eq!(d.get_or("solvers/p/tolerance", "").parse::<f64>().ok(), Some(1e-6));
        // Integers, which is all a dimension set ever holds, are untouched.
        assert_eq!(d.get("dimensions"), Some("[0 1 -1 0 0 0 0]"));
    }

    #[test]
    fn binary_files_are_refused_with_the_conversion_hint() {
        let src = "FoamFile\n{\n    version 2.0;\n    format binary;\n    class labelList;\n}\n";
        let e = match FoamDict::parse(src, "constant/polyMesh/owner") {
            Err(e) => e.to_string(),
            Ok(_) => panic!("a binary file was accepted"),
        };
        assert!(e.contains("foamFormatConvert"), "{e}");
        assert!(e.contains("constant/polyMesh/owner"), "{e}");
    }

    #[test]
    fn a_missing_semicolon_names_the_file_and_the_line() {
        let src = "a\n{\n    b 1;\n    c 2\n}\n";
        let e = match FoamDict::parse(src, "case/system/controlDict") {
            Err(e) => e.to_string(),
            Ok(_) => panic!("a missing ';' was accepted"),
        };
        assert!(e.contains("case/system/controlDict"), "{e}");
        assert!(e.contains("line 5"), "{e}");
        assert!(e.contains("entry 'c'"), "{e}");
    }

    #[test]
    fn an_unterminated_block_is_an_error_not_a_truncated_dictionary() {
        assert!(FoamDict::parse("a\n{\n  b 1;\n", "f").is_err());
        assert!(FoamDict::parse("}\n", "f").is_err());
    }

    /// Comment stripping runs before tokenising, so an entry that is entirely
    /// commented out must not leave a dangling key behind.
    #[test]
    fn commented_out_entries_leave_nothing_behind() {
        let src = "a 1;\n// b 2;\n/* c 3; */\nd 4;\n";
        let d = match FoamDict::parse(src, "f") {
            Ok(d) => d,
            Err(e) => panic!("{e}"),
        };
        assert_eq!(d.sub_keys(""), vec!["a", "d"]);
    }
}
