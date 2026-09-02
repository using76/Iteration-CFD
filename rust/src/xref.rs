// meteor-cfd - Copyright (c) 2026 주식회사 메테오시뮬레이션 (Meteo Simulation Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.
// Provenance: see PROVENANCE.md. No GPL-licensed source was consulted.

//! The citation audit - SPEC-LIT §80.
//!
//! `NOTICE`, `PROVENANCE.md` and the two READMEs are guarded by tests and stay
//! correct. Source comments were guarded by nothing, and that is where the
//! drift accumulated: §77 shipped 42 citation sites pointing at equations the
//! spec had renumbered underneath them, §79 about thirty pointing at a layout
//! that never existed, and both were corrected by hand. This module makes the
//! class of defect fail a test instead.
//!
//! It parses `SPEC-LIT.md` for the section numbers and equation labels that
//! actually exist, parses every `.rs`/`.cu`/`.cuh` file under `rust/` for the
//! symbols its comments, doc comments and string literals cite, and requires
//! the second set to be a subset of the first. §80.2 states the five citation
//! forms it recognises, §80.3 how a citation is attributed to the document it
//! belongs to, §80.4 the ambiguity ratchet, §80.5 the census, §80.8 what a
//! second reading of this module found in it, and §80.9 what the audit does
//! NOT catch.
//!
//! Written from: nothing. **ORIGINAL** - the rule, the lexer and the
//! attribution model are this project's own, and the only inputs are
//! `SPEC-LIT.md`'s own structure and the tree on disk. No GPL-licensed source
//! was consulted.

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};

// ==========================================================================
//  The document side: what SPEC-LIT.md actually contains
// ==========================================================================

/// Everything the audit knows about one Markdown document.
#[derive(Default)]
pub struct Doc {
    /// Every heading number, plus every ancestor of one (`13.4.1` implies
    /// `13.4` and `13`, whether or not those are spelled as headings).
    pub sections: BTreeSet<String>,
    /// Heading number -> heading text, for the census and the messages.
    pub headings: BTreeMap<String, String>,
    /// Equation labels: `(64.6)` and `(S47.3)` both reduce to `64.6`/`47.3`.
    pub equations: BTreeSet<String>,
    /// Top-level numbers of the sections that label equations at all -
    /// twenty-nine of them, all §40 or later. In a section that labels none
    /// the parenthesised form is a subsection reference (§80.2, form P).
    pub equation_sections: BTreeSet<String>,
}

impl Doc {
    /// Parse a Markdown document. Headings come from `#`-prefixed lines
    /// outside fenced blocks; equation labels from a `(NN.M)` or `(SNN.M)`
    /// closing a line INSIDE one, which is where the spec defines them. A
    /// label mentioned in prose is a reference, not a definition, so it does
    /// not create the symbol it cites.
    pub fn parse(text: &str) -> Doc {
        let mut d = Doc::default();
        let mut fenced = false;
        let mut section: Option<String> = None;
        for line in text.lines() {
            let t = line.trim_start();
            if t.starts_with("```") {
                fenced = !fenced;
                continue;
            }
            if fenced {
                // A label defines only IN ITS OWN SECTION. A cross-reference
                // to §46's equation closing a line of §59's fenced table is a
                // reference and not a definition; a validation row whose last
                // column is a measured value - §77 prints an expected 17.6
                // beside a computed 17.59 - is not a label at all. The shipped
                // parser took both, which is how §17, a section with no
                // subsections and no equations whatever, came to be counted
                // among the sections that label equations.
                if let Some(sym) = trailing_label(line) {
                    if top(&sym) == section {
                        d.equations.insert(sym);
                    }
                }
            } else if let Some((num, title)) = heading(t) {
                section = top(&num);
                d.headings.entry(num.clone()).or_insert(title);
                for a in ancestors(&num) {
                    d.sections.insert(a);
                }
                d.sections.insert(num);
            }
        }
        // §78 states the convention in the spec's own words: "a bare number
        // refers to a set by its bare number: `(78.2)` means the pair
        // (78.2a)/(78.2b)". So a lettered label registers its family too.
        for e in d.equations.clone() {
            if let Some(stem) = e.strip_suffix(|c: char| c.is_ascii_lowercase()) {
                if stem.contains('.') {
                    d.equations.insert(stem.to_string());
                }
            }
        }
        d.equation_sections = d.equations.iter().filter_map(|e| top(e)).collect();
        d
    }
}

/// `## 77. The vapour ...` -> `("77", "The vapour ...")`. The period after a
/// top-level number is optional, which is how the spec writes it.
fn heading(t: &str) -> Option<(String, String)> {
    let rest = t.strip_prefix('#')?.trim_start_matches('#');
    if !rest.starts_with(' ') {
        return None;
    }
    let rest = rest.trim_start();
    let (num, after) = read_number(rest.as_bytes(), 0)?;
    let after = &rest[after..];
    let after = after.strip_prefix('.').unwrap_or(after);
    if !after.starts_with(' ') {
        return None;
    }
    Some((num, after.trim().to_string()))
}

/// A label closing a line inside a fenced block: `... (64.6)` -> `64.6`.
/// `E_P = N_P / max(D_P, tiny)` also ends in a parenthesis and is not one.
fn trailing_label(line: &str) -> Option<String> {
    let inner = line.trim_end().strip_suffix(')')?;
    let open = inner.rfind('(')?;
    let body = &inner[open + 1..];
    let body = body.strip_prefix('S').unwrap_or(body);
    let (num, end) = read_number(body.as_bytes(), 0)?;
    if end != body.len() || !num.contains('.') {
        return None;
    }
    Some(num)
}

/// `13.4.1` -> `["13", "13.4"]`.
fn ancestors(num: &str) -> Vec<String> {
    let parts: Vec<&str> = num.split('.').collect();
    (1..parts.len()).map(|k| parts[..k].join(".")).collect()
}

/// The top-level section a symbol belongs to.
fn top(sym: &str) -> Option<String> {
    let first = sym.split('.').next()?;
    if first.is_empty() || !first.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    Some(first.to_string())
}

/// Read `NN(.NN)*[a-z]?` at `i`, returning the text and the byte after it.
///
/// The single-letter suffix is accepted only on a dotted number, which is the
/// only place the spec uses it (`42.5a`, `49.2b`, `78.3a`). Without that
/// restriction `S2s::update` scans as a citation of "§2s", which is how a
/// scanner acquires false positives it then gets weakened to silence.
fn read_number(b: &[u8], mut i: usize) -> Option<(String, usize)> {
    let start = i;
    if i >= b.len() || !b[i].is_ascii_digit() {
        return None;
    }
    while i < b.len() && b[i].is_ascii_digit() {
        i += 1;
    }
    let mut dotted = false;
    while i + 1 < b.len() && b[i] == b'.' && b[i + 1].is_ascii_digit() {
        dotted = true;
        i += 1;
        while i < b.len() && b[i].is_ascii_digit() {
            i += 1;
        }
    }
    if dotted && i < b.len() && b[i].is_ascii_lowercase() {
        i += 1;
    }
    Some((String::from_utf8_lossy(&b[start..i]).into_owned(), i))
}

/// A number written with a leading zero is arithmetic, not an address: the
/// spec has no §0 and no §07. This is what keeps `(0.25)`, `(0.50)` and the
/// DOI fragment `S0017-9310` out of the citation set.
fn leading_zero(sym: &str) -> bool {
    sym.split('.').any(|p| p.len() > 1 && p.starts_with('0')) || sym.starts_with('0')
}

// ==========================================================================
//  The source side: which bytes of a source file are prose
// ==========================================================================

/// The byte ranges of a source file that carry prose: line comments, block
/// comments and string literals. Everything else is code, where
/// `Tok::Num(78.0)` is a float and `S2s` is a type.
///
/// Every delimiter this walks (`/`, `*`, `"`, `'`, `\`, `\n`, `#`) is ASCII,
/// and no UTF-8 continuation byte is ASCII, so the ranges always fall on
/// character boundaries even in a file with Korean in it.
pub fn prose_spans(text: &str, c_style: bool) -> Vec<(usize, usize)> {
    // A span is a comment or a string; only comments merge with comments.
    #[derive(PartialEq, Eq, Clone, Copy)]
    enum Kind {
        Comment,
        Str,
    }
    let b = text.as_bytes();
    let n = b.len();
    let mut out: Vec<(Kind, usize, usize)> = Vec::new();
    let mut i = 0;
    while i < n {
        match b[i] {
            b'/' if i + 1 < n && b[i + 1] == b'/' => {
                let j = text[i..].find('\n').map(|k| i + k).unwrap_or(n);
                out.push((Kind::Comment, i, j));
                i = j;
            }
            b'/' if i + 1 < n && b[i + 1] == b'*' => {
                // Rust nests block comments; C does not.
                let mut depth = 1usize;
                let mut j = i + 2;
                while j < n && depth > 0 {
                    if !c_style && b[j] == b'/' && j + 1 < n && b[j + 1] == b'*' {
                        depth += 1;
                        j += 2;
                    } else if b[j] == b'*' && j + 1 < n && b[j + 1] == b'/' {
                        depth -= 1;
                        j += 2;
                    } else {
                        j += 1;
                    }
                }
                out.push((Kind::Comment, i, j.min(n)));
                i = j.min(n);
            }
            b'r' if !c_style && (i == 0 || !is_ident_byte(b[i - 1])) => {
                let mut k = i + 1;
                while k < n && b[k] == b'#' {
                    k += 1;
                }
                if k < n && b[k] == b'"' {
                    let mut close = vec![b'"'];
                    close.resize(1 + (k - i - 1), b'#');
                    let j = find_from(b, k + 1, &close).map(|p| p + close.len()).unwrap_or(n);
                    out.push((Kind::Str, i, j));
                    i = j;
                } else {
                    i += 1;
                }
            }
            b'"' => {
                let mut j = i + 1;
                while j < n {
                    if b[j] == b'\\' {
                        j += 2;
                    } else if b[j] == b'"' {
                        j += 1;
                        break;
                    } else {
                        j += 1;
                    }
                }
                out.push((Kind::Str, i, j.min(n)));
                i = j.min(n);
            }
            b'\'' => {
                // A char literal, so that `'"'` does not open a string. A
                // lifetime is left alone and costs one byte.
                let esc = i + 1 < n && b[i + 1] == b'\\';
                let end = if esc { i + 3 } else { i + 2 };
                i = if end < n && b[end] == b'\'' { end + 1 } else { i + 1 };
            }
            _ => i += 1,
        }
    }
    // Consecutive comments separated only by whitespace are one block, so a
    // `//!` bibliography reads as one piece of prose and §80.3's two-line
    // window can see the work name that a wrapped entry put on the line above.
    let mut merged: Vec<(Kind, usize, usize)> = Vec::new();
    for (kind, a, z) in out {
        let joins = match merged.last() {
            Some(&(prev, _, prev_end)) => {
                kind == Kind::Comment
                    && prev == Kind::Comment
                    && prev_end <= a
                    && text[prev_end..a].trim().is_empty()
            }
            None => false,
        };
        match merged.last_mut() {
            Some(last) if joins => last.2 = z,
            _ => merged.push((kind, a, z)),
        }
    }
    merged.into_iter().map(|(_, a, z)| (a, z)).collect()
}

fn is_ident_byte(c: u8) -> bool {
    c.is_ascii_alphanumeric() || c == b'_'
}

fn find_from(b: &[u8], from: usize, needle: &[u8]) -> Option<usize> {
    if needle.is_empty() || b.len() < needle.len() || from > b.len() - needle.len() {
        return None;
    }
    (from..=b.len() - needle.len()).find(|&k| &b[k..k + needle.len()] == needle)
}

// ==========================================================================
//  The citations themselves
// ==========================================================================

/// The five spellings §80.2 recognises.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Form {
    /// `§13.4` - a section, always.
    Section,
    /// `S13.4` - the ASCII spelling, which is a section OR an equation.
    Ascii,
    /// `(64.6)` - an equation where its section labels equations, a
    /// subsection where it does not.
    Paren,
    /// `(S47.3)` - the parenthesised form with the `S` the ASCII-only files
    /// spell it with, and resolved exactly like [`Form::Paren`]: an equation
    /// where its section labels equations, a subsection where it does not.
    ParenAscii,
    /// `SPEC-LIT 13.4`, `SPEC-LIT section 36` - a section, always.
    Bare,
}

/// One citation, with everything a failure message needs.
#[derive(Clone, Debug)]
pub struct Cite {
    pub form: Form,
    pub symbol: String,
    /// The document the citation names, per §80.3. `None` means SPEC-LIT.
    pub document: Option<String>,
    pub line: usize,
    pub context: String,
}

/// Scan one prose span. `names` is §80.3's attribution registry, longest
/// first.
fn scan_span(span: &str, names: &[String], out: &mut Vec<(Form, String, Option<String>, usize)>) {
    let b = span.as_bytes();
    let n = b.len();
    let mut i = 0;
    while i < n {
        // `§` is U+00A7 = C2 A7.
        if b[i] == 0xC2 && i + 1 < n && b[i + 1] == 0xA7 {
            let mut j = i + 2;
            if j < n && b[j] == b' ' {
                j += 1;
            }
            if let Some((sym, end)) = read_number(b, j) {
                if !leading_zero(&sym) {
                    push(span, names, Form::Section, sym, i, out);
                    i = end;
                    continue;
                }
            }
            i += 2;
            continue;
        }
        if b[i] == b'S' && (i == 0 || !is_ident_byte(b[i - 1])) {
            let mut matched = false;
            if let Some((sym, end)) = read_number(b, i + 1) {
                if (end == n || !b[end].is_ascii_alphanumeric()) && !leading_zero(&sym) {
                    push(span, names, Form::Ascii, sym, i, out);
                    i = end;
                    matched = true;
                }
            }
            if matched {
                continue;
            }
            if let Some(start) = bare_spec_lit(b, i) {
                if let Some((sym, end)) = read_number(b, start) {
                    if !leading_zero(&sym) {
                        push(span, names, Form::Bare, sym, start, out);
                        i = end;
                        continue;
                    }
                }
            }
        }
        if b[i] == b'(' && (i == 0 || !(is_ident_byte(b[i - 1]) || b[i - 1] == b'.')) {
            let mut j = i + 1;
            let s = j < n && b[j] == b'S';
            if s {
                j += 1;
            }
            if let Some((sym, end)) = read_number(b, j) {
                if end < n && b[end] == b')' && sym.contains('.') && !leading_zero(&sym) {
                    let f = if s { Form::ParenAscii } else { Form::Paren };
                    push(span, names, f, sym, i, out);
                    i = end + 1;
                    continue;
                }
            }
        }
        i += 1;
    }
}

/// Match ``SPEC-LIT[.md][`][,][:] [section[s] ]`` at `i`, returning the byte
/// where the number must start. Only spaces and tabs are crossed, so a match
/// can never run past the end of a comment line into the next one's marker.
fn bare_spec_lit(b: &[u8], i: usize) -> Option<usize> {
    const TAG: &[u8] = b"SPEC-LIT";
    if b.len() < i + TAG.len() || &b[i..i + TAG.len()] != TAG {
        return None;
    }
    let mut j = i + TAG.len();
    if b.len() >= j + 3 && &b[j..j + 3] == b".md" {
        j += 3;
    }
    while j < b.len() && matches!(b[j], b'`' | b',' | b':') {
        j += 1;
    }
    let before = j;
    while j < b.len() && matches!(b[j], b' ' | b'\t') {
        j += 1;
    }
    if j == before {
        return None;
    }
    for word in [&b"sections"[..], &b"section"[..]] {
        if b.len() >= j + word.len() && &b[j..j + word.len()] == word {
            let mut k = j + word.len();
            let after_word = k;
            while k < b.len() && matches!(b[k], b' ' | b'\t') {
                k += 1;
            }
            if k > after_word {
                return Some(k);
            }
        }
    }
    Some(j)
}

/// Attribute a citation and record it. §80.3: the document is the last
/// registry name appearing on the citation's own line or the line before it;
/// with nothing there, the citation is SPEC-LIT's.
fn push(
    span: &str,
    names: &[String],
    form: Form,
    symbol: String,
    at: usize,
    out: &mut Vec<(Form, String, Option<String>, usize)>,
) {
    let mut start = at;
    for _ in 0..2 {
        match span[..start].rfind('\n') {
            Some(k) => start = k,
            None => {
                start = 0;
                break;
            }
        }
    }
    let window = &span[start..at];
    let mut best: Option<(usize, &str)> = None;
    for name in names {
        if let Some(k) = window.rfind(name.as_str()) {
            if best.is_none_or(|(p, _)| k > p) {
                best = Some((k, name.as_str()));
            }
        }
    }
    let document = match best {
        Some((_, w)) if w != "SPEC-LIT" => Some(w.to_string()),
        _ => None,
    };
    out.push((form, symbol, document, at));
}

// ==========================================================================
//  The registry, read out of the spec so that it cannot drift either
// ==========================================================================

/// §80.3's registry: the names a citation may be attributed to, the top-level
/// numbers reserved for test fixtures, and §80.4's ratchet ceiling.
#[derive(Default)]
pub struct Registry {
    /// Author or standard names that mean "this citation is not SPEC-LIT's".
    pub works: Vec<String>,
    /// Repository-relative Markdown documents, resolved against their own
    /// headings rather than merely excused.
    pub documents: Vec<String>,
    /// Top-level numbers that must NOT exist in the spec, cited only by tests
    /// that need an obviously invented address.
    pub reserved: BTreeSet<String>,
    /// The most ambiguous bare-`S` citations §80.4 tolerates.
    pub ambiguous_ceiling: usize,
}

impl Registry {
    /// Parse the fenced block in §80.3 whose first line is
    /// `CITATION-AUDIT REGISTRY`.
    pub fn parse(spec: &str) -> Registry {
        let mut r = Registry::default();
        let mut fenced = false;
        let mut inside = false;
        for line in spec.lines() {
            let t = line.trim();
            if t.starts_with("```") {
                if fenced && inside {
                    break;
                }
                fenced = !fenced;
                inside = false;
                continue;
            }
            if !fenced {
                continue;
            }
            if t == "CITATION-AUDIT REGISTRY" {
                inside = true;
                continue;
            }
            if !inside || t.is_empty() {
                continue;
            }
            let Some((kind, value)) = t.split_once(char::is_whitespace) else { continue };
            let value = value.trim();
            match kind {
                "work" => r.works.push(value.to_string()),
                "document" => r.documents.push(value.to_string()),
                "reserved" => {
                    if let Some(num) = value.split_whitespace().next() {
                        r.reserved.insert(num.to_string());
                    }
                }
                "ambiguous-ceiling" => {
                    r.ambiguous_ceiling =
                        value.split_whitespace().next().and_then(|v| v.parse().ok()).unwrap_or(0);
                }
                _ => {}
            }
        }
        r
    }

    /// Every name the attribution window may match, longest first so that
    /// `docs/07-fire-solver.md` is preferred over any prefix of it.
    fn names(&self) -> Vec<String> {
        let mut v: Vec<String> = std::iter::once("SPEC-LIT".to_string())
            .chain(self.works.iter().cloned())
            .chain(self.documents.iter().cloned())
            .collect();
        v.sort_by_key(|s| std::cmp::Reverse(s.len()));
        v
    }
}

// ==========================================================================
//  The audit
// ==========================================================================

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).to_path_buf()
}

/// Every `.rs`/`.cu`/`.cuh` under `src/`, `cuda/`, `tests/`, plus `build.rs` -
/// the same four roots `NOTICE` names and `provenance_audit` walks. A file
/// added later is audited without this module being edited, which is the
/// property a hand-written `include_str!` list would not have.
fn sources() -> Vec<PathBuf> {
    fn walk(dir: &Path, out: &mut Vec<PathBuf>) {
        let Ok(rd) = fs::read_dir(dir) else { return };
        for e in rd.flatten() {
            let p = e.path();
            if p.is_dir() {
                walk(&p, out);
            } else if matches!(
                p.extension().and_then(|s| s.to_str()),
                Some("rs") | Some("cu") | Some("cuh")
            ) {
                out.push(p);
            }
        }
    }
    let mut out = Vec::new();
    for d in ["src", "cuda", "tests"] {
        walk(&root().join(d), &mut out);
    }
    out.push(root().join("build.rs"));
    out.sort();
    out
}

fn rel(p: &Path) -> String {
    let s = p.strip_prefix(root()).unwrap_or(p).display().to_string();
    s.replace(std::path::MAIN_SEPARATOR, "/")
}

fn is_c_style(p: &Path) -> bool {
    matches!(p.extension().and_then(|s| s.to_str()), Some("cu") | Some("cuh"))
}

/// Every citation in one file, with its line number and enough context that a
/// failure message is actionable without opening the file.
pub fn citations_in(text: &str, c_style: bool, names: &[String]) -> Vec<Cite> {
    let mut out = Vec::new();
    for (a, z) in prose_spans(text, c_style) {
        let span = &text[a..z];
        let mut hits = Vec::new();
        scan_span(span, names, &mut hits);
        for (form, symbol, document, at) in hits {
            let line = text[..a + at].bytes().filter(|&c| c == b'\n').count() + 1;
            let lo = span[..at].rfind('\n').map(|k| k + 1).unwrap_or(0);
            let hi = span[at..].find('\n').map(|k| at + k).unwrap_or(span.len());
            out.push(Cite {
                form,
                symbol,
                document,
                line,
                context: span[lo..hi].trim().to_string(),
            });
        }
    }
    out
}

/// Does the citation name something that exists?
fn resolves(cite: &Cite, spec: &Doc, docs: &BTreeMap<String, Doc>, reg: &Registry) -> bool {
    if let Some(d) = &cite.document {
        return match docs.get(d) {
            Some(doc) => doc.sections.contains(&cite.symbol),
            // A `work` has no document in this repository - Patankar, Jasak,
            // Saad. Their numbering is theirs and this audit cannot check it;
            // §80.9 says so rather than pretending otherwise. A `document`
            // that did not load is the opposite case: §80.3 promises it is
            // CHECKED, and a promise that evaporates when the file is renamed
            // is the no-op this section exists to make impossible.
            None => !reg.documents.iter().any(|x| x == d),
        };
    }
    let Some(t) = top(&cite.symbol) else { return true };
    if reg.reserved.contains(&t) {
        return true;
    }
    match cite.form {
        Form::Section | Form::Bare => spec.sections.contains(&cite.symbol),
        // §80.4: the parenthesised form means one thing, with or without the
        // `S` that the ASCII-only files (every `.cu` in this tree) must use.
        Form::Paren | Form::ParenAscii => {
            if spec.equation_sections.contains(&t) {
                spec.equations.contains(&cite.symbol)
            } else {
                spec.sections.contains(&cite.symbol)
            }
        }
        // The bare `SNN.M` cannot be tightened the same way: 586 sites in this
        // tree write it for a subsection of a section that also labels
        // equations. §80.4's ratchet is what stands in for a rule here.
        Form::Ascii => {
            spec.equations.contains(&cite.symbol) || spec.sections.contains(&cite.symbol)
        }
    }
}

/// The whole audit, in the shape the tests consume.
struct Audit {
    files: usize,
    checked: usize,
    attributed: usize,
    /// Citations by form, in [`Form`]'s own order, for §80.7's table.
    by_form: [usize; 5],
    /// Symbols that are BOTH an equation label and a subsection - the
    /// population §80.4's ratchet is drawn from.
    both: usize,
    stale: Vec<(String, Cite)>,
    ambiguous: Vec<(String, Cite)>,
    used_names: BTreeSet<String>,
    /// Symbol -> the files citing it, for §80.5's census.
    cited_in: BTreeMap<String, BTreeSet<String>>,
    /// Registry `document` -> how many heading numbers it parsed to. One that
    /// is missing or unreadable is absent from the map, and the test below
    /// that §80.3 property 2 names fails on it rather than excusing it.
    docs_read: BTreeMap<String, usize>,
    spec: Doc,
}

fn run() -> Audit {
    let spec_text = fs::read_to_string(root().join("SPEC-LIT.md")).expect("SPEC-LIT.md");
    let spec = Doc::parse(&spec_text);
    let reg = Registry::parse(&spec_text);
    let names = reg.names();
    let mut docs: BTreeMap<String, Doc> = BTreeMap::new();
    for d in &reg.documents {
        if let Ok(t) = fs::read_to_string(root().join("..").join(d)) {
            docs.insert(d.clone(), Doc::parse(&t));
        }
    }
    let docs_read: BTreeMap<String, usize> =
        docs.iter().map(|(k, v)| (k.clone(), v.sections.len())).collect();
    let both: BTreeSet<String> = spec.equations.intersection(&spec.sections).cloned().collect();

    let mut files = 0usize;
    let mut checked = 0usize;
    let mut attributed = 0usize;
    let mut stale = Vec::new();
    let mut ambiguous = Vec::new();
    let mut used_names = BTreeSet::new();
    let mut cited_in: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
    let mut by_form = [0usize; 5];
    for p in sources() {
        let Ok(text) = fs::read_to_string(&p) else { continue };
        files += 1;
        let name = rel(&p);
        for c in citations_in(&text, is_c_style(&p), &names) {
            by_form[match c.form {
                Form::Section => 0,
                Form::Ascii => 1,
                Form::Paren => 2,
                Form::ParenAscii => 3,
                Form::Bare => 4,
            }] += 1;
            match &c.document {
                Some(d) => {
                    attributed += 1;
                    used_names.insert(d.clone());
                }
                None => {
                    checked += 1;
                    if c.symbol.contains('.') {
                        cited_in.entry(c.symbol.clone()).or_default().insert(name.clone());
                    }
                }
            }
            if !resolves(&c, &spec, &docs, &reg) {
                stale.push((name.clone(), c.clone()));
            }
            if c.document.is_none() && c.form == Form::Ascii && both.contains(&c.symbol) {
                ambiguous.push((name.clone(), c));
            }
        }
    }
    let both = both.len();
    Audit {
        files,
        checked,
        attributed,
        by_form,
        both,
        stale,
        ambiguous,
        used_names,
        cited_in,
        docs_read,
        spec,
    }
}

// ==========================================================================
//  SPEC-LIT §80 - every claim it makes, as a test
// ==========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn registry() -> Registry {
        Registry::parse(&fs::read_to_string(root().join("SPEC-LIT.md")).unwrap())
    }

    /// The audit has to be reading a real tree. A path mistake that made
    /// `sources()` empty would turn every assertion below into a tautology,
    /// which is how a structural test dies quietly.
    #[test]
    fn the_audit_reads_the_tree_it_claims_to_read() {
        let a = run();
        assert!(a.files >= 170, "the walk found only {} source files", a.files);
        assert!(
            a.checked >= 7000,
            "only {} SPEC-LIT citations were found; the scanner is not scanning",
            a.checked
        );
        assert!(
            a.spec.sections.len() >= 500 && a.spec.equations.len() >= 250,
            "SPEC-LIT parsed to {} sections and {} equation labels",
            a.spec.sections.len(),
            a.spec.equations.len()
        );
        println!(
            "  [S80] {} files; {} citations = {} SPEC-LIT + {} attributed elsewhere; \
             by form S/A/P/PA/B = {:?}; the spec has {} section numbers and {} equation \
             labels, {} of which are both",
            a.files,
            a.checked + a.attributed,
            a.checked,
            a.attributed,
            a.by_form,
            a.spec.sections.len(),
            a.spec.equations.len(),
            a.both
        );
    }

    /// **§80.2.** Every section number and equation label cited by a comment,
    /// doc comment, error message or panic string exists in the document the
    /// citation names. This is the rule §77 and §79 broke.
    #[test]
    fn every_citation_names_something_that_exists() {
        let a = run();
        let list: Vec<String> = a
            .stale
            .iter()
            .map(|(f, c)| {
                let doc = c.document.as_deref().unwrap_or("SPEC-LIT.md");
                format!("{f}:{} cites {doc} {} | {}", c.line, c.symbol, c.context)
            })
            .collect();
        assert!(
            list.is_empty(),
            "{} citation(s) name a section or equation that does not exist. Fix the \
             citation, or - when it belongs to another work - name that work on the \
             citation's own line or the line above (SPEC-LIT §80.3):\n  {}",
            list.len(),
            list.join("\n  ")
        );
    }

    /// **§80.3.** The registry is the only way out of the audit, and every
    /// entry has to earn its place: a name nothing cites is a name that
    /// silently widens the escape hatch.
    #[test]
    fn every_registry_name_is_actually_used() {
        let a = run();
        let reg = registry();
        let unused: Vec<&String> = reg
            .works
            .iter()
            .chain(reg.documents.iter())
            .filter(|w| !a.used_names.contains(*w))
            .collect();
        assert!(
            unused.is_empty(),
            "SPEC-LIT §80.3's registry names {} work(s)/document(s) that no citation is \
             attributed to; remove them rather than leave an unused exemption: {unused:?}",
            unused.len()
        );
    }

    /// **§80.3.** A `work` is an excuse this audit cannot check; a `document`
    /// is a promise that it CAN, and §80.3 says so in those words. That
    /// promise is kept only while the file is where the registry says it is,
    /// and nothing used to notice when it was not. Hiding one of the two
    /// registered documents turned every citation into it into a silent pass -
    /// the no-op this whole section exists to make impossible. Those citations
    /// now fail, and this test fails first and names the file.
    ///
    /// (The document is deliberately not named on either of these two lines:
    /// §80.3's window would attribute the citations above to it.)
    #[test]
    fn a_registry_document_is_read_not_merely_excused() {
        let a = run();
        let reg = registry();
        assert!(!reg.documents.is_empty(), "SPEC-LIT §80.3's registry names no document");
        for d in &reg.documents {
            let n = a.docs_read.get(d).copied().unwrap_or(0);
            assert!(
                n > 0,
                "SPEC-LIT §80.3's registry names `{d}` as a document whose OWN headings a \
                 citation is resolved against, but it did not parse to a single heading - \
                 it has been moved, renamed or emptied. Fix the path, or demote the entry \
                 to `work`, which excuses a citation instead of checking it."
            );
        }
    }

    /// **§80.3.** A reserved number must not exist in the spec. §99 is the
    /// address §69's registry tests cite precisely because it is invented; the
    /// day someone writes a real §99 those fixtures stop being obviously fake,
    /// and this fails first.
    #[test]
    fn a_reserved_number_has_no_heading() {
        let a = run();
        let reg = registry();
        assert!(!reg.reserved.is_empty(), "SPEC-LIT §80.3 reserves no number at all");
        for r in &reg.reserved {
            assert!(
                !a.spec.sections.contains(r),
                "SPEC-LIT §80.3 reserves §{r} for test fixtures, but the spec now has a \
                 §{r}; renumber it or drop the reservation"
            );
        }
    }

    /// **§80.4, the ratchet.** 252 symbols in this spec are BOTH an equation
    /// label and a subsection number, and the bare `SNN.M` form does not say
    /// which is meant - that ambiguity is how one bare token for 77.5 came to
    /// mean §77.4 in one file, §77.6 in another and §77.5 in a third. The
    /// count may fall and may not rise: new code writes `§NN.M` for a
    /// subsection and `(NN.M)` for an equation.
    #[test]
    fn the_ambiguous_ascii_form_does_not_grow() {
        let a = run();
        let reg = registry();
        let n = a.ambiguous.len();
        println!("  [S80.4] {n} ambiguous bare-S citations, ceiling {}", reg.ambiguous_ceiling);
        assert!(
            n <= reg.ambiguous_ceiling,
            "{n} bare `SNN.M` citations name a symbol that is both an equation label and a \
             subsection, against SPEC-LIT §80.4's ceiling of {}. Write `§NN.M` for the \
             subsection or `(NN.M)` for the equation; the ceiling only comes down.\n  {}",
            reg.ambiguous_ceiling,
            a.ambiguous
                .iter()
                .rev()
                .take(20)
                .map(|(f, c)| format!("{f}:{} S{}", c.line, c.symbol))
                .collect::<Vec<_>>()
                .join("\n  ")
        );
    }

    /// **§80.5, the census.** Every symbol cited from more than one file, with
    /// the spec's own heading beside it, so a human reading the list can see a
    /// citation that survives §80.2 and still means the wrong thing. Printed,
    /// not asserted - §80.4 says why no automatic rule replaces it.
    #[test]
    fn the_multi_file_census_is_printed() {
        let a = run();
        let mut shared: Vec<(&String, &BTreeSet<String>)> =
            a.cited_in.iter().filter(|(_, f)| f.len() > 1).collect();
        shared.sort_by_key(|(sym, f)| (std::cmp::Reverse(f.len()), (*sym).clone()));
        println!("  [S80.5] {} symbols are cited from more than one file", shared.len());
        // EVERY one of them, not a head - §80.5 claims a list a human reads,
        // and a truncated list is one whose tail nobody has ever seen.
        for (sym, files) in &shared {
            let h = a.spec.headings.get(*sym).map(String::as_str).unwrap_or("(equation only)");
            println!("      {sym:<9} {:>2} files  {h}", files.len());
        }
        assert!(shared.len() > 100, "only {} shared symbols, which cannot be right", shared.len());
    }

    // ---- the parser itself, so that the gate cannot become a no-op ------

    #[test]
    fn headings_and_equation_labels_parse_the_way_the_spec_writes_them() {
        // The two strays are numbered §99 - §80.3's reserved address - for the
        // same reason §69's fixtures are: this file is itself audited, and a
        // fixture that spells a real stale citation to prove a point is a real
        // stale citation. The case it stands for is §77's validation row,
        // whose last column prints an expected 17.6 beside a computed 17.59.
        let d = Doc::parse(concat!(
            "## 77. The vapour\n",
            "### 77.1 What is deposited\n",
            "#### (a) `prescribedYield`\n",
            "```\n",
            "  m_P = sum n_p dm_p            kg      (77.4)\n",
            "  E_P = N_P / max(D_P, tiny)\n",
            "  cooling / expansion          99.59    (99.6)\n",
            "  from the solid side, see              (S46.5)\n",
            "```\n",
            "prose citing (77.9), which is a reference and not a definition\n",
            "#### 13.4.1 A setting must REACH the solver\n",
            "### 42.5a A correction\n",
            "## 78. Impact\n",
            "```\n",
            "  x = y                                 (78.3a)\n",
            "```\n",
        ));
        assert!(d.sections.contains("77") && d.sections.contains("77.1"));
        assert!(d.sections.contains("42.5a"));
        // ancestors are registered even where the spec skips the heading
        assert!(d.sections.contains("13.4") && d.sections.contains("13"));
        // an unnumbered heading creates nothing
        assert!(!d.sections.contains("a"));
        assert_eq!(d.headings.get("77.1").map(String::as_str), Some("What is deposited"));
        assert!(d.equations.contains("77.4"));
        // a line ending in an ordinary parenthesis is not a label
        assert!(d.equations.iter().all(|e| e != "D_P"));
        // §78's own convention: a lettered label registers its family
        assert!(d.equations.contains("78.3a") && d.equations.contains("78.3"));
        // a reference in prose does not define the symbol it cites
        assert!(!d.equations.contains("77.9"));
        // ... and neither does a measured value that happens to close its line
        // in parentheses, nor a cross-reference to another section's equation
        // inside a fenced block. Both shipped as definitions. The first
        // invented an equation for §17 - a section with no subsections and no
        // equations of its own - and put §17 into `equation_sections`, where
        // it would have turned the first honest parenthesised citation of a
        // §17 subsection into a spurious failure, and this rule into the next
        // thing somebody weakened to make the tree pass.
        assert!(!d.equations.contains("99.6"), "a measured value defined an equation");
        assert!(!d.equations.contains("46.5"), "a cross-reference defined an equation");
        assert_eq!(d.equation_sections, ["77".to_string(), "78".to_string()].into());
    }

    #[test]
    fn the_scanner_reads_prose_and_leaves_code_alone() {
        let names = vec!["SPEC-LIT".to_string(), "Patankar".to_string()];
        let src = concat!(
            "// SPEC-LIT §13.4 and (77.2), Patankar §4.2 on the same line\n",
            "let x = Limited(0.0);\n",
            "let y = cmu.powf(0.25);\n",
            "let s = \"refused by name (39.6), SPEC-LIT S26.1\";\n",
            "// SPEC-LIT section 36 and SPEC-LIT.md 13.4.1\n",
            "let z = S2s::update();\n",
        );
        let got: Vec<(Form, String, Option<String>)> = citations_in(src, false, &names)
            .into_iter()
            .map(|c| (c.form, c.symbol, c.document))
            .collect();
        let want = vec![
            (Form::Section, "13.4", None),
            (Form::Paren, "77.2", None),
            (Form::Section, "4.2", Some("Patankar")),
            (Form::Paren, "39.6", None),
            (Form::Ascii, "26.1", None),
            (Form::Bare, "36", None),
            (Form::Bare, "13.4.1", None),
        ];
        let want: Vec<(Form, String, Option<String>)> = want
            .into_iter()
            .map(|(f, s, d)| (f, s.to_string(), d.map(|d: &str| d.to_string())))
            .collect();
        assert_eq!(got, want);
    }

    #[test]
    fn arithmetic_in_parentheses_is_not_a_citation() {
        let names = vec!["SPEC-LIT".to_string()];
        for s in [
            "// unwrap_or(0.0) and atan(0.3) and ln(2.25)\n",
            "// K(0.25) = 4, K(0.50) = 5, K(0.56) = 6\n",
            "// DOI 10.1016/S0017-9310(02)00101-1\n",
            "// f_e1(0.25) - 1\n",
        ] {
            let got = citations_in(s, false, &names);
            assert!(got.is_empty(), "{s} scanned as {got:?}");
        }
    }

    #[test]
    fn a_citation_belongs_to_the_last_work_named_within_two_lines() {
        let names =
            vec!["SPEC-LIT".to_string(), "Saad".to_string(), "docs/07-fire-solver.md".to_string()];
        // One literal, so the fixture is one prose span and §80.3's window
        // sees the work name the way it does in a real bibliography entry.
        let src = r#"//!   Saad, *Iterative Methods*, 2nd ed. (2003),
//!     §6.7 (PCG) and §12.4, with `SPEC-LIT` §21 for the colouring
//!
//!   plain prose citing §8.4
/// see `docs/07-fire-solver.md` §1.1 for the run
"#;
        let got: Vec<(String, Option<String>)> = citations_in(src, false, &names)
            .into_iter()
            .map(|c| (c.symbol, c.document))
            .collect();
        let want: Vec<(String, Option<String>)> = [
            ("6.7", Some("Saad")),
            ("12.4", Some("Saad")),
            ("21", None),
            ("8.4", None),
            ("1.1", Some("docs/07-fire-solver.md")),
        ]
        .into_iter()
        .map(|(s, d): (&str, Option<&str>)| (s.to_string(), d.map(str::to_string)))
        .collect();
        assert_eq!(got, want);
    }

    #[test]
    fn a_block_comment_and_a_raw_string_are_both_prose() {
        let names = vec!["SPEC-LIT".to_string()];
        let src = "/* SPEC-LIT §31.2 */ let s = r#\"cites §22 too\"#;\n";
        let got: Vec<String> =
            citations_in(src, false, &names).into_iter().map(|c| c.symbol).collect();
        assert_eq!(got, vec!["31.2".to_string(), "22".to_string()]);
    }

    /// The registry has to parse; an empty one would excuse nothing and
    /// re-flag every legitimate citation of Patankar, Jasak and Saad.
    #[test]
    fn the_registry_block_parses() {
        let reg = registry();
        assert!(reg.works.len() >= 8, "SPEC-LIT §80.3 lists {} works", reg.works.len());
        assert!(reg.documents.len() >= 2, "SPEC-LIT §80.3 lists {} documents", reg.documents.len());
        assert!(reg.ambiguous_ceiling > 0, "SPEC-LIT §80.4's ceiling did not parse");
        assert!(reg.works.iter().any(|w| w == "Patankar"), "{:?}", reg.works);
    }
}
