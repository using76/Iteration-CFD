// meteor-cfd - Copyright (c) 2026 주식회사 메테오시뮬레이션 (Meteo Simulation Co., Ltd.)
// Source-available, not Open Source. Educational use is free; research,
// publication and commercial use require a licence - simul@msimul.com
// See LICENSE at the repository root.

//! The OpenFOAM ASCII tokeniser.
//!
//! ofgpu reads `constant/polyMesh` and the `0/` directory itself so a case can
//! run on a machine with no OpenFOAM installed. That means reproducing enough
//! of `Foam::ISstream` to survive real case files, and three of its rules look
//! wrong until you meet those files:
//!
//! * comment stripping is **quote aware**, because every `FoamFile` header
//!   carries `location "constant/polyMesh"` and that `/` is not a comment;
//! * a word swallows **balanced parentheses**, so `div(phi,U)` and `grad(U)`
//!   arrive as single keywords — that is how OpenFOAM reads `fvSchemes`, and
//!   splitting them would lose the scheme's identity;
//! * but a token that *starts with a digit* is a number, so the compact face
//!   form `4(156 0 78 235)` still splits into `4` `(` `156` … `)`.
//!
//! Provenance: carried across from this project's own earlier C++ I/O layer
//! when the crate moved to Rust. That C++ was written from the case format as
//! it appears in data files - not from any CFD code's source - and the format
//! itself, not another program, is the specification here. No GPL-licensed
//! source was consulted.

use std::fmt;
use std::path::{Path, PathBuf};

use crate::error::{parse_err, IoContext, Result};
use crate::Label;

// ==========================================================================
//  Tokens
// ==========================================================================

/// One lexical token. `Punct` only ever holds one of `; { } ( ) [ ]`.
///
/// `Num` keeps the parsed value rather than the source text: the readers all
/// want the number, and [`fmt::Display`] renders it back in a form that
/// round-trips (`1.0` prints as `1`, so `dimensions [0 1 -1 0 0 0 0]` survives
/// a rejoin unchanged).
#[derive(Debug, Clone, PartialEq)]
pub enum Tok {
    Word(String),
    Str(String),
    Num(f64),
    Punct(char),
}

impl Tok {
    /// True for `; { } ( ) [ ]`. Keywords are *any* non-punctuation token, so
    /// this is the test a dictionary parser needs, not a match on `Word`.
    #[inline]
    pub fn is_punct_any(&self) -> bool {
        matches!(self, Tok::Punct(_))
    }

    #[inline]
    pub fn is_punct(&self, c: char) -> bool {
        matches!(self, Tok::Punct(p) if *p == c)
    }

    #[inline]
    pub fn is_word(&self, w: &str) -> bool {
        matches!(self, Tok::Word(s) if s == w)
    }
}

impl fmt::Display for Tok {
    /// A quoted string renders **without** its quotes, which is what makes
    /// `location "constant/polyMesh"` come back out of the dictionary as
    /// `constant/polyMesh`.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Tok::Word(s) | Tok::Str(s) => f.write_str(s),
            // `{}` on f64 never uses exponent notation and drops the trailing
            // ".0", so -1.0 prints as "-1" and a dimension set rejoins exactly.
            Tok::Num(v) => write!(f, "{}", v),
            Tok::Punct(c) => write!(f, "{}", c),
        }
    }
}

// ==========================================================================
//  Tokenizer
// ==========================================================================

/// A whole file's worth of tokens, with the line each was read from.
///
/// Scanning is eager: the only way it can fail is an unterminated string,
/// which by construction sits at the end of the file, so the failure is
/// parked in `scan_err` and surfaces from the first [`Tokenizer::next`] or
/// [`Tokenizer::peek`] that reaches the end. That is what lets `new` be
/// infallible, which in turn lets callers hold a `Tokenizer` by value.
pub struct Tokenizer {
    path: String,
    toks: Vec<Tok>,
    lines: Vec<usize>,
    pos: usize,
    eof_line: usize,
    scan_err: Option<(usize, String)>,
}

impl Tokenizer {
    /// Strips `//` and `/* */` comments (quote aware), then tokenises.
    ///
    /// Comments are blanked in place rather than deleted so every newline
    /// survives and the line numbers in error messages stay honest.
    pub fn new(src: &str, path: &str) -> Self {
        // Scanning over `char` rather than bytes: blanking a comment byte
        // inside a multi-byte sequence would produce invalid UTF-8, and some
        // real cases carry non-ASCII in their banner comments.
        let mut chars: Vec<char> = src.chars().collect();
        strip_comments(&mut chars);

        let s = scan(&chars);
        Tokenizer {
            path: path.to_string(),
            toks: s.toks,
            lines: s.lines,
            pos: 0,
            eof_line: s.eof_line,
            scan_err: s.err,
        }
    }

    /// The file name every error carries.
    pub fn path(&self) -> &str {
        &self.path
    }

    /// Line of the token about to be read, or of the end of the file.
    pub fn line(&self) -> usize {
        self.lines.get(self.pos).copied().unwrap_or(self.eof_line)
    }

    /// True once every token has been consumed. A pending scan failure does
    /// *not* show up here — call [`Tokenizer::check_scan_error`] at the end of
    /// a parse so a truncated file cannot pass silently.
    pub fn done(&self) -> bool {
        self.pos >= self.toks.len()
    }

    /// Re-raises a scan failure that the parse never ran into.
    pub fn check_scan_error(&self) -> Result<()> {
        match &self.scan_err {
            Some((line, msg)) => parse_err(&self.path, format!("line {}: {}", line, msg)),
            None => Ok(()),
        }
    }

    pub fn peek(&mut self) -> Result<Option<&Tok>> {
        if self.pos >= self.toks.len() {
            self.check_scan_error()?;
            return Ok(None);
        }
        Ok(self.toks.get(self.pos))
    }

    /// Lookahead without consuming. `off == 0` is the same token as `peek`.
    ///
    /// Needed because the field reader has to tell `uniform`'s single value
    /// from a bare list by looking at the token *after* the number.
    pub fn peek_at(&self, off: usize) -> Option<&Tok> {
        self.toks.get(self.pos + off)
    }

    pub fn next(&mut self) -> Result<Option<Tok>> {
        if self.pos >= self.toks.len() {
            self.check_scan_error()?;
            return Ok(None);
        }
        let t = self.toks[self.pos].clone();
        self.pos += 1;
        Ok(Some(t))
    }

    pub fn is_punct(&self, c: char) -> bool {
        self.is_punct_at(c, 0)
    }

    pub fn is_punct_at(&self, c: char, off: usize) -> bool {
        self.peek_at(off).is_some_and(|t| t.is_punct(c))
    }

    pub fn is_word(&self, w: &str) -> bool {
        self.is_word_at(w, 0)
    }

    pub fn is_word_at(&self, w: &str, off: usize) -> bool {
        self.peek_at(off).is_some_and(|t| t.is_word(w))
    }

    pub fn expect_punct(&mut self, c: char) -> Result<()> {
        if !self.is_punct(c) {
            return self.err(format!("expected '{}'", c));
        }
        self.pos += 1;
        Ok(())
    }

    /// A keyword or patch name: **any** non-punctuation token, taken as text.
    ///
    /// Not restricted to `Tok::Word` because `fvSolution` keys the solver
    /// blocks by regex — `"(U|k|epsilon)"` is a quoted string, and it is a
    /// perfectly ordinary dictionary key.
    pub fn expect_word(&mut self) -> Result<String> {
        Ok(self.expect_key()?.0)
    }

    /// [`Self::expect_word`], plus whether the name was written in QUOTES.
    ///
    /// That flag is the whole difference between a name and a pattern: only a
    /// quoted key is a regular expression in the case format, so a patch
    /// genuinely called `p` and the pattern `"p.*"` can never be confused.
    /// [`crate::io::dict::FoamDict`] keeps the quotes in the flattened path
    /// for exactly this reason.
    pub fn expect_key(&mut self) -> Result<(String, bool)> {
        let (text, quoted) = match self.toks.get(self.pos) {
            Some(Tok::Str(s)) => (s.clone(), true),
            Some(t) if !t.is_punct_any() => (t.to_string(), false),
            _ => return self.err("expected a name"),
        };
        self.pos += 1;
        Ok((text, quoted))
    }

    pub fn expect_num(&mut self) -> Result<f64> {
        let v = match self.toks.get(self.pos) {
            Some(Tok::Num(v)) => *v,
            // A word that happens to spell a number is accepted, matching the
            // C++ `strtod` path; `2ndOrder` still fails, because the whole
            // token has to convert.
            Some(t) if !t.is_punct_any() => match t.to_string().parse::<f64>() {
                Ok(v) => v,
                Err(_) => return self.err("expected a number"),
            },
            _ => return self.err("expected a number"),
        };
        self.pos += 1;
        Ok(v)
    }

    /// A list size or a mesh index. Rejects anything with a fractional part,
    /// so a corrupt `owner` file cannot quietly truncate into a valid label.
    pub fn expect_label(&mut self) -> Result<Label> {
        let v = match self.toks.get(self.pos) {
            Some(Tok::Num(v))
                if v.fract() == 0.0
                    && *v >= Label::MIN as f64
                    && *v <= Label::MAX as f64 =>
            {
                *v as Label
            }
            _ => return self.err("expected an integer"),
        };
        self.pos += 1;
        Ok(v)
    }

    /// Consume the tokens of an entry value up to and including its `;`, or
    /// the whole `{ ... }` block if the entry is a sub-dictionary.
    pub fn skip_entry(&mut self) -> Result<()> {
        if self.is_punct('{') {
            let mut depth = 0i32;
            while !self.done() {
                let tk = match self.next()? {
                    Some(t) => t,
                    None => break,
                };
                if tk.is_punct('{') {
                    depth += 1;
                } else if tk.is_punct('}') {
                    depth -= 1;
                    if depth == 0 {
                        return Ok(());
                    }
                }
            }
            return self.err("unterminated '{'");
        }

        while !self.done() && !self.is_punct(';') {
            if self.is_punct('}') {
                return self.err("missing ';' before '}'");
            }
            self.next()?;
        }
        if self.done() {
            return self.err("expected ';'");
        }
        self.next()?;
        Ok(())
    }

    /// Drop a leading `FoamFile { ... }` block if there is one.
    ///
    /// Used by the mesh and field readers, which do not want the header — and
    /// which would otherwise trip over the `note` entry blockMesh writes into
    /// `owner` and `neighbour`. `FoamDict` deliberately does *not* call this:
    /// it keeps the header under `FoamFile/...`.
    pub fn skip_header(&mut self) -> Result<()> {
        if !self.is_word("FoamFile") {
            return Ok(());
        }
        self.next()?;
        if !self.is_punct('{') {
            return self.err("expected '{' after FoamFile");
        }
        self.skip_entry()
    }

    /// A parse failure naming the file, the line and the offending token.
    pub fn err<T>(&self, msg: impl AsRef<str>) -> Result<T> {
        let line = self.line();
        let m = match self.toks.get(self.pos) {
            Some(t) => format!("line {}: {} (got '{}')", line, msg.as_ref(), t),
            None => format!("line {}: {} but hit the end of the file", line, msg.as_ref()),
        };
        parse_err(&self.path, m)
    }
}

// ==========================================================================
//  Source text
// ==========================================================================

/// Read a case file, with the two failures that actually happen spelled out.
///
/// Decoding is lossy on purpose: a stray non-UTF-8 byte in a banner comment
/// must not stop a case from running, and no byte that survives lossily can
/// change how the tokens come out.
pub fn slurp(path: &Path) -> Result<String> {
    match std::fs::read(path) {
        Ok(bytes) => Ok(String::from_utf8_lossy(&bytes).into_owned()),
        Err(e) => {
            let gz = gz_sibling(path);
            if gz.exists() {
                return parse_err(
                    &gz,
                    "this file is gzip-compressed. ofgpu reads plain ASCII only - \
                     run 'gunzip' on the case (or set writeCompression off) and try again.",
                );
            }
            Err(e).path(path)
        }
    }
}

fn gz_sibling(path: &Path) -> PathBuf {
    let mut s = path.as_os_str().to_os_string();
    s.push(".gz");
    PathBuf::from(s)
}

/// Refuse a binary-format file **before** tokenising it.
///
/// A binary blob would otherwise be shredded into millions of nonsense tokens
/// and fail somewhere deep inside the parse with a meaningless message. This
/// works on the raw text — the `format` entry is found by string search, and
/// truncating at the `;` keeps a trailing `// ... binary ...` comment on the
/// same line from triggering a false positive.
pub fn check_ascii_format(text: &str, path: &str) -> Result<()> {
    let Some(h) = text.find("FoamFile") else {
        return Ok(());
    };
    let Some(b) = text[h..].find('{').map(|i| i + h) else {
        return Ok(());
    };
    let Some(e) = text[b..].find('}').map(|i| i + b) else {
        return Ok(());
    };

    let hdr = &text[b..e];
    let Some(f) = hdr.find("format") else {
        return Ok(());
    };

    let mut entry = &hdr[f..];
    if let Some(semi) = entry.find(';') {
        entry = &entry[..semi];
    }

    if entry.contains("binary") {
        return parse_err(path, BINARY_MSG);
    }
    Ok(())
}

/// Shared so the dictionary can raise the identical message from the parsed
/// `FoamFile/format` entry.
pub(crate) const BINARY_MSG: &str = "written in binary format, which ofgpu does not read.\n    \
     Convert the case with:  foamFormatConvert -constant -allTime\n    \
     (or set  writeFormat ascii;  in system/controlDict before writing).";

// ==========================================================================
//  Scanner
// ==========================================================================

/// Blank out `//` and `/* */` comments in place, keeping every newline.
///
/// The quote arm is the whole point: `location "constant/polyMesh"` and
/// `#include "initialConditions"` both contain characters that would otherwise
/// start a comment.
fn strip_comments(s: &mut Vec<char>) {
    let n = s.len();
    let mut i = 0usize;

    while i < n {
        let c = s[i];

        if c == '"' || c == '\'' {
            let q = c;
            i += 1;
            while i < n && s[i] != q {
                if s[i] == '\\' && i + 1 < n {
                    i += 1;
                }
                i += 1;
            }
            if i < n {
                i += 1;
            }
        } else if c == '/' && i + 1 < n && s[i + 1] == '/' {
            while i < n && s[i] != '\n' {
                s[i] = ' ';
                i += 1;
            }
        } else if c == '/' && i + 1 < n && s[i + 1] == '*' {
            s[i] = ' ';
            s[i + 1] = ' ';
            i += 2;
            while i + 1 < n && !(s[i] == '*' && s[i + 1] == '/') {
                if s[i] != '\n' {
                    s[i] = ' ';
                }
                i += 1;
            }
            if i + 1 < n {
                s[i] = ' ';
                s[i + 1] = ' ';
                i += 2;
            } else {
                while i < n {
                    if s[i] != '\n' {
                        s[i] = ' ';
                    }
                    i += 1;
                }
            }
        } else {
            i += 1;
        }
    }
}

/// OpenFOAM's `word::valid()`, plus `[` and `]` which its token reader treats
/// as punctuation. `(` and `)` **are** word characters — they are handled by
/// the nesting counter in [`scan`], and that is what makes `div(phi,U)` come
/// out as a single keyword.
fn is_word_char(c: char) -> bool {
    !c.is_whitespace()
        && c != '"'
        && c != '\''
        && c != ';'
        && c != '{'
        && c != '}'
        && c != '['
        && c != ']'
        && c != '/'
}

fn is_punct_char(c: char) -> bool {
    matches!(c, '(' | ')' | '{' | '}' | '[' | ']' | ';')
}

struct Scan {
    toks: Vec<Tok>,
    lines: Vec<usize>,
    eof_line: usize,
    err: Option<(usize, String)>,
}

impl Scan {
    fn push(&mut self, t: Tok, line: usize) {
        self.toks.push(t);
        self.lines.push(line);
    }
}

fn scan(s: &[char]) -> Scan {
    let n = s.len();
    let mut out = Scan {
        toks: Vec::new(),
        lines: Vec::new(),
        eof_line: 1,
        err: None,
    };

    let mut i = 0usize;
    let mut line = 1usize;

    while i < n {
        let c = s[i];

        if c == '\n' {
            line += 1;
            i += 1;
            continue;
        }
        if c.is_whitespace() {
            i += 1;
            continue;
        }

        let tok_line = line;

        // ---- punctuation -------------------------------------------------
        if is_punct_char(c) {
            out.push(Tok::Punct(c), tok_line);
            i += 1;
            continue;
        }

        // ---- quoted string -----------------------------------------------
        if c == '"' || c == '\'' {
            let q = c;
            i += 1;
            let mut v = String::new();
            while i < n && s[i] != q {
                if s[i] == '\\' && i + 1 < n {
                    i += 1;
                }
                if s[i] == '\n' {
                    line += 1;
                }
                v.push(s[i]);
                i += 1;
            }
            if i >= n {
                out.eof_line = line;
                out.err = Some((tok_line, "unterminated string".to_string()));
                return out;
            }
            i += 1;
            out.push(Tok::Str(v), tok_line);
            continue;
        }

        // ---- number --------------------------------------------------------
        // Greedy about the exponent but NOT about a trailing letter, so
        // `1e-05` is one number while `2ndOrder` falls through to the word
        // scanner. A following '(' is fine, and that is exactly what makes the
        // compact face form `4(156 0 78 235)` split correctly.
        let mut num_start = c.is_ascii_digit();
        if !num_start && (c == '-' || c == '+' || c == '.') {
            if i + 1 < n && s[i + 1].is_ascii_digit() {
                num_start = true;
            } else if (c == '-' || c == '+')
                && i + 2 < n
                && s[i + 1] == '.'
                && s[i + 2].is_ascii_digit()
            {
                num_start = true;
            }
        }

        if num_start {
            let mut j = i;
            if s[j] == '-' || s[j] == '+' {
                j += 1;
            }
            while j < n && s[j].is_ascii_digit() {
                j += 1;
            }
            if j < n && s[j] == '.' {
                j += 1;
                while j < n && s[j].is_ascii_digit() {
                    j += 1;
                }
            }
            if j < n && (s[j] == 'e' || s[j] == 'E') {
                let mut k = j + 1;
                if k < n && (s[k] == '-' || s[k] == '+') {
                    k += 1;
                }
                if k < n && s[k].is_ascii_digit() {
                    k += 1;
                    while k < n && s[k].is_ascii_digit() {
                        k += 1;
                    }
                    j = k;
                }
            }

            let trailing_letter = j < n && (s[j].is_ascii_alphabetic() || s[j] == '_');

            if !trailing_letter {
                let text: String = s[i..j].iter().collect();
                match text.parse::<f64>() {
                    Ok(v) => {
                        out.push(Tok::Num(v), tok_line);
                        i = j;
                        continue;
                    }
                    // Unreachable for anything the scan above accepts, but a
                    // silent 0 here would corrupt a mesh, so keep the text.
                    Err(_) => {
                        out.push(Tok::Word(text), tok_line);
                        i = j;
                        continue;
                    }
                }
            }
        }

        // ---- word, swallowing balanced parentheses -------------------------
        {
            let mut j = i;
            let mut depth = 0i32;
            while j < n {
                let d = s[j];
                if d == '(' {
                    depth += 1;
                } else if d == ')' {
                    if depth == 0 {
                        break;
                    }
                    depth -= 1;
                } else if !is_word_char(d) {
                    break;
                }
                j += 1;
            }

            if j == i {
                // A delimiter this reader does not model (a bare '/', say).
                // Emit it as punctuation rather than looping forever.
                out.push(Tok::Punct(c), tok_line);
                i += 1;
                continue;
            }

            let text: String = s[i..j].iter().collect();
            out.push(Tok::Word(text), tok_line);
            i = j;
        }
    }

    out.eof_line = line;
    out
}

// ==========================================================================
//  Tests
// ==========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn toks(src: &str) -> Vec<Tok> {
        let mut t = Tokenizer::new(src, "test.dict");
        let mut v = Vec::new();
        while let Ok(Some(tk)) = t.next() {
            v.push(tk);
        }
        v
    }

    /// The single most load-bearing rule: an fvSchemes key must not split, or
    /// every scheme lookup silently falls back to the default.
    #[test]
    fn a_word_swallows_balanced_parens() {
        assert_eq!(
            toks("div(phi,U) Gauss linear;"),
            vec![
                Tok::Word("div(phi,U)".into()),
                Tok::Word("Gauss".into()),
                Tok::Word("linear".into()),
                Tok::Punct(';'),
            ]
        );
        assert_eq!(toks("grad(U)")[0], Tok::Word("grad(U)".into()));
    }

    /// ...but the compact face list must, or the mesh reader sees one giant
    /// nonsense word instead of a face.
    #[test]
    fn a_number_never_swallows_the_paren_after_it() {
        assert_eq!(
            toks("4(156 0 78 235)"),
            vec![
                Tok::Num(4.0),
                Tok::Punct('('),
                Tok::Num(156.0),
                Tok::Num(0.0),
                Tok::Num(78.0),
                Tok::Num(235.0),
                Tok::Punct(')'),
            ]
        );
    }

    #[test]
    fn a_dimension_set_is_punctuation_around_numbers() {
        assert_eq!(
            toks("[0 1 -1 0 0 0 0]"),
            vec![
                Tok::Punct('['),
                Tok::Num(0.0),
                Tok::Num(1.0),
                Tok::Num(-1.0),
                Tok::Num(0.0),
                Tok::Num(0.0),
                Tok::Num(0.0),
                Tok::Num(0.0),
                Tok::Punct(']'),
            ]
        );
    }

    #[test]
    fn exponents_stay_one_token_but_a_trailing_letter_does_not() {
        assert_eq!(toks("1e-05")[0], Tok::Num(1e-5));
        assert_eq!(toks("-1.5E+3")[0], Tok::Num(-1500.0));
        assert_eq!(toks("2ndOrder")[0], Tok::Word("2ndOrder".into()));
        // `1e` has no digits after the exponent marker, so it is a word.
        assert_eq!(toks("1e")[0], Tok::Word("1e".into()));
    }

    /// The header slash that broke every naive comment stripper.
    #[test]
    fn a_slash_inside_quotes_is_not_a_comment() {
        assert_eq!(
            toks("location \"constant/polyMesh\";"),
            vec![
                Tok::Word("location".into()),
                Tok::Str("constant/polyMesh".into()),
                Tok::Punct(';'),
            ]
        );
    }

    #[test]
    fn both_comment_forms_disappear_without_moving_the_line_numbers() {
        let src = "a 1; // trailing\n/* block\n   spanning */\nb 2;\n";
        assert_eq!(
            toks(src),
            vec![
                Tok::Word("a".into()),
                Tok::Num(1.0),
                Tok::Punct(';'),
                Tok::Word("b".into()),
                Tok::Num(2.0),
                Tok::Punct(';'),
            ]
        );

        let mut t = Tokenizer::new(src, "test.dict");
        for _ in 0..3 {
            let _ = t.next();
        }
        // `b` is on line 4 even though three lines of it were blanked.
        assert_eq!(t.line(), 4);
    }

    #[test]
    fn the_openfoam_banner_is_stripped_whole() {
        let src = "/*--------------------------------*- C++ -*-------------------------------*\\\n\
                   |  =========                                                              |\n\
                   \\*-------------------------------------------------------------------------*/\n\
                   FoamFile\n{\n    version 2.0;\n}\n";
        assert_eq!(toks(src)[0], Tok::Word("FoamFile".into()));
    }

    #[test]
    fn errors_name_the_file_and_the_line() {
        let mut t = Tokenizer::new("a 1;\nb 2;\nc\n", "case/system/fvSchemes");
        for _ in 0..6 {
            let _ = t.next();
        }
        let e = t.expect_punct(';').unwrap_err();
        let msg = e.to_string();
        assert!(msg.contains("case/system/fvSchemes"), "{msg}");
        assert!(msg.contains("line 3"), "{msg}");
        assert!(msg.contains("got 'c'"), "{msg}");
    }

    /// An unterminated string is at EOF by construction, so it can only be
    /// noticed on the way out; a parse that stops early must still report it.
    #[test]
    fn an_unterminated_string_is_not_lost() {
        let mut t = Tokenizer::new("a \"oops\nb 1;\n", "test.dict");
        assert!(t.check_scan_error().is_err());
        assert_eq!(t.next().ok().flatten(), Some(Tok::Word("a".into())));
        assert!(t.next().is_err());
    }

    #[test]
    fn a_quoted_regex_key_is_a_name() {
        let mut t = Tokenizer::new("\"(U|k|epsilon)\" { }", "test.dict");
        assert_eq!(t.expect_word().ok(), Some("(U|k|epsilon)".to_string()));
    }

    #[test]
    fn a_fractional_label_is_refused() {
        let mut t = Tokenizer::new("12 3.5", "test.dict");
        assert_eq!(t.expect_label().ok(), Some(12));
        assert!(t.expect_label().is_err());
    }

    #[test]
    fn numbers_rejoin_the_way_a_dimension_set_was_written() {
        // Tok::Num drops the source text, so Display has to give it back.
        assert_eq!(Tok::Num(-1.0).to_string(), "-1");
        assert_eq!(Tok::Num(0.0).to_string(), "0");
        assert_eq!(Tok::Num(1e-5).to_string().parse::<f64>().ok(), Some(1e-5));
    }

    #[test]
    fn binary_format_is_refused_with_the_conversion_hint() {
        let src = "FoamFile\n{\n    format      binary;\n    class       vectorField;\n}\n";
        let e = check_ascii_format(src, "constant/polyMesh/points").unwrap_err();
        assert!(e.to_string().contains("foamFormatConvert"), "{e}");
    }

    /// `format ascii; // written by a binary build` must not be rejected: the
    /// entry ends at its ';'.
    #[test]
    fn the_word_binary_after_the_semicolon_is_not_a_binary_file() {
        let src = "FoamFile\n{\n    format      ascii;   // was binary\n}\n";
        assert!(check_ascii_format(src, "f").is_ok());
    }

    #[test]
    fn skip_entry_takes_a_whole_nested_block() {
        let mut t = Tokenizer::new("{ a { b 1; } c 2; } after;", "test.dict");
        assert!(t.skip_entry().is_ok());
        assert_eq!(t.expect_word().ok(), Some("after".to_string()));
    }
}
