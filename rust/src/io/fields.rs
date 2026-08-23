// meteor-cfd - Copyright (c) 2026 주식회사 메테오시뮬레이션 (Meteo Simulation Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.

//! `volScalarField` / `volVectorField` reader and writer.
//!
//! Provenance: carried across from this project's own earlier C++ I/O layer
//! when the crate moved to Rust. That C++ was written from the case format as
//! it appears in data files - not from any CFD code's source - and the format
//! itself, not another program, is the specification here. No GPL-licensed
//! source was consulted.
//!
//! ofgpu reads the `0/` directory itself so a case can run on a machine with
//! no OpenFOAM installed, which means accepting everything the format allows
//! rather than a convenient subset of it:
//!
//! * `uniform <v>;`
//! * `nonuniform List<scalar> N ( ... );` - the `List<scalar>` token is
//!   *optional*, because the degenerate form an empty patch gets is
//!   `nonuniform 0()`, with no type name at all;
//! * `N{v}`, OpenFOAM's compact all-equal list;
//! * a bare `( ... )` or `N ( ... )` with no `uniform`/`nonuniform` keyword,
//!   which older and hand-written files contain.
//!
//! Lists are stored **verbatim**: length 1 means the file said `uniform`.
//! Nothing here knows the patch sizes, so expanding one is the caller's job -
//! see [`expand_scalars`] / [`expand_vectors`].
//!
//! # Pattern keys
//!
//! `boundaryField { ".*" { type zeroGradient; } }` is how most real cases are
//! written, and a reader that only does exact lookup matches nothing in it. A
//! QUOTED patch key is therefore stored with its quotes - `"\".*\""` - so it
//! can never be confused with a patch genuinely called `.*`, and the file order
//! of those keys is kept in `boundary_patterns` because `BTreeMap` has thrown
//! it away. [`RawScalarField::spec`] is the lookup: exact match first, then the
//! patterns in file order.
//!
//! Extended from ofgpu `SPEC-LIT.md` §13.4 (a setting the solver cannot honour
//! must fail loudly, and one it *can* honour must actually be read) and the
//! POSIX ERE definition; see [`crate::io::regex`].
//!
//! # `surfaceScalarField`
//!
//! [`write_surface_scalar_field`] writes the conservative face flux `phi`.
//! Until it existed the writer produced no `phi` at all, so a restart from an
//! ofgpu-written time directory fell back to potential flow or to
//! `interpolate(U)·Sf` - and neither of those satisfies `Σ_f phi_f = 0` per
//! cell, which is the property the pressure equation of `SPEC-LIT` §5.1
//! assumes of the flux it is handed. See `SPEC-LIT` §5.1 and the restart check
//! of §22.
//!
//! The layout is the same file grammar the reader above already accepts - the
//! `internalField` list is one entry per INTERNAL face rather than per cell,
//! and each `boundaryField` entry is one per boundary face of that patch - so
//! the reader needs no new code path.

use std::collections::BTreeMap;
use std::path::Path;

use crate::error::{parse_err, Error, IoContext, Result};
use crate::io::regex::Regex;
use crate::io::tokenizer::{check_ascii_format, slurp, Tok, Tokenizer};
use crate::{Scalar, Vec3};

// ==========================================================================
//  Types
// ==========================================================================

/// One `boundaryField` entry, with whatever keys it carried.
///
/// Every list is verbatim - length 1 means `uniform`, any other length means
/// `nonuniform` and equals the patch size. An **empty** list therefore reads
/// as either "the key was absent" or "the patch has no faces"; the two cannot
/// be told apart, because this struct drops the `hasValue`, `hasGradient`, ...
/// flags the C++ version carries. The one visible consequence is on the way
/// out: a zero-sized patch does not get its `value nonuniform 0()` rewritten.
#[derive(Debug, Default, Clone, PartialEq)]
pub struct PatchFieldSpec {
    pub type_name: String,
    /// `cyclic` only.
    pub neighbour_patch: Option<String>,

    pub value: Vec<Scalar>,
    pub value_v: Vec<Vec3>,
    pub gradient: Vec<Scalar>,
    pub gradient_v: Vec<Vec3>,
    pub ref_value: Vec<Scalar>,
    pub ref_value_v: Vec<Vec3>,
    pub ref_gradient: Vec<Scalar>,
    pub ref_gradient_v: Vec<Vec3>,
    pub inlet_value: Vec<Scalar>,
    pub inlet_value_v: Vec<Vec3>,
    /// Scalar even on a vector field: the mixed condition blends every
    /// component with the same fraction, so OpenFOAM stores a `scalarField`
    /// here whatever the field's rank.
    pub value_fraction: Vec<Scalar>,

    /// Every other key the entry carried, verbatim, keyed by its name.
    ///
    /// `turbulentIntensity 0.05;`, `mixingLength 0.007;`, `p0 uniform 0;`,
    /// `volumetricFlowRate 0.1;` - each condition has one or two entries only
    /// it knows about, and a struct field per condition would mean editing
    /// this type every time one is added. Keeping the raw text also means a
    /// value that cannot be read is reported by the condition that wanted it,
    /// naming the key, rather than silently defaulting here.
    ///
    /// Sub-dictionaries are not captured; nothing needs one yet.
    pub extra: BTreeMap<String, String>,
}

impl PatchFieldSpec {
    /// One number out of [`Self::extra`], or an error naming the key.
    ///
    /// `None` when the entry is absent, so the caller can distinguish "not
    /// given" (where it may have a default) from "given and unreadable"
    /// (where it may not) - SPEC-LIT §13.4.
    pub fn number(&self, key: &str, patch: &str) -> Result<Option<Scalar>> {
        let Some(raw) = self.extra.get(key) else {
            return Ok(None);
        };

        // `uniform 0.05`, `0.05`, and the dimensioned `p0 [0 2 -2 0 0 0 0]
        // 0` all end with the number, exactly as FoamDict::scalar assumes.
        let last = raw
            .split_whitespace()
            .filter_map(|t| t.parse::<f64>().ok())
            .next_back();

        match last {
            Some(v) => Ok(Some(v as Scalar)),
            None => Err(Error::Field {
                field: patch.to_string(),
                msg: format!("{key}: \"{raw}\" is not a number"),
            }),
        }
    }

    /// [`Self::number`], with an error rather than `None` when the key is
    /// missing: a condition that is defined by its `mixingLength` cannot be
    /// built without one.
    pub fn required_number(&self, key: &str, patch: &str, wanted_by: &str) -> Result<Scalar> {
        match self.number(key, patch)? {
            Some(v) => Ok(v),
            None => Err(Error::Field {
                field: patch.to_string(),
                msg: format!("{wanted_by} needs a `{key}` entry and the patch has none"),
            }),
        }
    }
}

/// A `volScalarField` file as it sits on disk.
#[derive(Debug, Default, Clone, PartialEq)]
pub struct RawScalarField {
    pub name: String,
    /// Kept as text (`"[0 2 -2 0 0 0 0]"`). Nothing in ofgpu checks
    /// dimensions, and writing back exactly what was read keeps a diff against
    /// a foamRun result down to the numbers.
    pub dimensions: String,
    pub internal: Vec<Scalar>,
    /// Patch entries, keyed by patch name - or, for a QUOTED key, by the
    /// pattern with its quotes kept.
    pub boundary: BTreeMap<String, PatchFieldSpec>,
    /// The quoted keys of [`Self::boundary`], in file order, because that is
    /// the tie-break when two patterns match the same patch.
    pub boundary_patterns: Vec<String>,
}

/// A `volVectorField` file as it sits on disk.
#[derive(Debug, Default, Clone, PartialEq)]
pub struct RawVectorField {
    pub name: String,
    pub dimensions: String,
    pub internal: Vec<Vec3>,
    /// See [`RawScalarField::boundary`].
    pub boundary: BTreeMap<String, PatchFieldSpec>,
    /// See [`RawScalarField::boundary_patterns`].
    pub boundary_patterns: Vec<String>,
}

// ==========================================================================
//  Pattern keys
// ==========================================================================

/// Which `boundaryField` entry governs `patch`: the exact one if the file
/// wrote it, otherwise the first quoted key, in file order, whose regular
/// expression matches.
///
/// Returns the key as it is STORED, so a pattern comes back with its quotes.
/// A malformed pattern is an error rather than a non-match - a key written in
/// quotes was meant to match something, and quietly matching nothing is the
/// bug this exists to remove.
pub fn governing_key(
    boundary: &BTreeMap<String, PatchFieldSpec>,
    patterns: &[String],
    patch: &str,
) -> Result<Option<String>> {
    if boundary.contains_key(patch) {
        return Ok(Some(patch.to_string()));
    }

    for key in patterns {
        let Some(pattern) = key.strip_prefix('"').and_then(|s| s.strip_suffix('"')) else {
            continue;
        };
        let re = Regex::new(pattern).map_err(Error::Config)?;
        if re.is_match(patch) {
            return Ok(Some(key.clone()));
        }
    }

    Ok(None)
}

macro_rules! patch_lookup {
    ($t:ty) => {
        impl $t {
            /// The entry governing `patch`, exact key or matching pattern.
            pub fn spec(&self, patch: &str) -> Result<Option<&PatchFieldSpec>> {
                Ok(governing_key(&self.boundary, &self.boundary_patterns, patch)?
                    .and_then(|k| self.boundary.get(&k)))
            }

            /// The key under which [`Self::spec`] found it.
            pub fn spec_key(&self, patch: &str) -> Result<Option<String>> {
                governing_key(&self.boundary, &self.boundary_patterns, patch)
            }

            /// A copy carrying only the boundary *type* names - what a driver
            /// seeds an output field with so the written case still says
            /// `nutkWallFunction` where the input did.
            ///
            /// The patterns come across too, so a case written with `".*"`
            /// keeps its types on the way out; the write path then expands
            /// them to one explicit entry per patch.
            pub fn types_only(&self) -> Self {
                let mut dst = Self::default();
                for (name, spec) in &self.boundary {
                    dst.boundary.entry(name.clone()).or_default().type_name =
                        spec.type_name.clone();
                    dst.boundary
                        .entry(name.clone())
                        .or_default()
                        .extra
                        .clone_from(&spec.extra);
                }
                dst.boundary_patterns.clone_from(&self.boundary_patterns);
                dst
            }
        }
    };
}

patch_lookup!(RawScalarField);
patch_lookup!(RawVectorField);

// ==========================================================================
//  Expansion
// ==========================================================================

/// Grow a `uniform` (length 1) or verbatim list to exactly `n` entries.
///
/// An absent list expands to zeros rather than failing: OpenFOAM itself
/// defaults a missing `value` to zero on every patch type that does not
/// demand one.
pub fn expand_scalars(v: &[Scalar], n: usize, what: &str) -> Result<Vec<Scalar>> {
    if v.len() == n {
        return Ok(v.to_vec());
    }
    if v.len() == 1 {
        return Ok(vec![v[0]; n]);
    }
    if v.is_empty() {
        return Ok(vec![0.0; n]);
    }
    Err(size_mismatch(what, n, v.len()))
}

/// Vector counterpart of [`expand_scalars`].
pub fn expand_vectors(v: &[Vec3], n: usize, what: &str) -> Result<Vec<Vec3>> {
    if v.len() == n {
        return Ok(v.to_vec());
    }
    if v.len() == 1 {
        return Ok(vec![v[0]; n]);
    }
    if v.is_empty() {
        return Ok(vec![Vec3::ZERO; n]);
    }
    Err(size_mismatch(what, n, v.len()))
}

fn size_mismatch(what: &str, n: usize, got: usize) -> Error {
    Error::Field {
        field: what.to_string(),
        msg: format!("expected {} values (or one uniform value), got {}", n, got),
    }
}

// ==========================================================================
//  Reading
// ==========================================================================

/// Read a `volScalarField`.
///
/// `n_cells == 0` keeps the internal field exactly as written; any other value
/// expands a `uniform` entry to that length and rejects a list of the wrong
/// size. `phi` is read through here too, with `n_cells` set to the internal
/// face count.
pub fn read_scalar_field(path: &Path, n_cells: usize) -> Result<RawScalarField> {
    let text = slurp(path)?;
    parse_scalar_field(&text, &path.display().to_string(), base_name(path), n_cells)
}

/// Read a `volVectorField`. See [`read_scalar_field`] for `n_cells`.
pub fn read_vector_field(path: &Path, n_cells: usize) -> Result<RawVectorField> {
    let text = slurp(path)?;
    parse_vector_field(&text, &path.display().to_string(), base_name(path), n_cells)
}

fn parse_scalar_field(
    src: &str,
    path: &str,
    default_name: String,
    n_cells: usize,
) -> Result<RawScalarField> {
    check_ascii_format(src, path)?;
    let mut tz = Tokenizer::new(src, path);

    let hdr = read_header(&mut tz)?;
    let mut f = RawScalarField {
        name: hdr.get("object").cloned().unwrap_or(default_name),
        ..Default::default()
    };

    let mut have_internal = false;

    while !tz.done() {
        if tz.is_punct(';') {
            tz.next()?;
            continue;
        }
        if is_punct_any(&tz) {
            return tz.err("expected a keyword");
        }

        let k = tz.expect_word()?;
        if k.starts_with('#') {
            skip_directive(&mut tz, &k)?;
        } else if k == "dimensions" {
            f.dimensions = gather_raw(&mut tz)?;
        } else if k == "internalField" {
            f.internal = parse_scalar_entry(&mut tz)?;
            have_internal = true;
        } else if k == "boundaryField" {
            parse_boundary_field(&mut tz, &mut f.boundary, &mut f.boundary_patterns, false)?;
        } else {
            tz.skip_entry()?;
        }
    }
    tz.check_scan_error()?;

    if !have_internal {
        return parse_err(path, "no internalField entry");
    }
    if n_cells > 0 {
        f.internal = expand_scalars(&f.internal, n_cells, &format!("{}: internalField", path))?;
    }
    Ok(f)
}

fn parse_vector_field(
    src: &str,
    path: &str,
    default_name: String,
    n_cells: usize,
) -> Result<RawVectorField> {
    check_ascii_format(src, path)?;
    let mut tz = Tokenizer::new(src, path);

    let hdr = read_header(&mut tz)?;
    let mut f = RawVectorField {
        name: hdr.get("object").cloned().unwrap_or(default_name),
        ..Default::default()
    };

    let mut have_internal = false;

    while !tz.done() {
        if tz.is_punct(';') {
            tz.next()?;
            continue;
        }
        if is_punct_any(&tz) {
            return tz.err("expected a keyword");
        }

        let k = tz.expect_word()?;
        if k.starts_with('#') {
            skip_directive(&mut tz, &k)?;
        } else if k == "dimensions" {
            f.dimensions = gather_raw(&mut tz)?;
        } else if k == "internalField" {
            f.internal = parse_vector_entry(&mut tz)?;
            have_internal = true;
        } else if k == "boundaryField" {
            parse_boundary_field(&mut tz, &mut f.boundary, &mut f.boundary_patterns, true)?;
        } else {
            tz.skip_entry()?;
        }
    }
    tz.check_scan_error()?;

    if !have_internal {
        return parse_err(path, "no internalField entry");
    }
    if n_cells > 0 {
        f.internal = expand_vectors(&f.internal, n_cells, &format!("{}: internalField", path))?;
    }
    Ok(f)
}

// ==========================================================================
//  Entry-level parsing
// ==========================================================================

#[inline]
fn is_punct_any(tz: &Tokenizer) -> bool {
    tz.peek_at(0).is_some_and(|t| t.is_punct_any())
}

#[inline]
fn is_num_at(tz: &Tokenizer, off: usize) -> bool {
    matches!(tz.peek_at(off), Some(Tok::Num(_)))
}

#[inline]
fn is_word_tok(tz: &Tokenizer) -> bool {
    matches!(tz.peek_at(0), Some(Tok::Word(_)))
}

/// Append one token to a raw value string, using OpenFOAM-ish spacing so
/// `[0 1 -1 0 0 0 0]` and `uniform (0 0 0)` come back out readable.
fn append_raw(v: &mut String, t: &Tok) {
    let opens_group = matches!(v.chars().last(), Some('(') | Some('['));
    let closes_group = t.is_punct(')') || t.is_punct(']');
    if !v.is_empty() && !opens_group && !closes_group {
        v.push(' ');
    }
    v.push_str(&t.to_string());
}

/// Capture an entry's tokens verbatim, consuming the `;`.
fn gather_raw(tz: &mut Tokenizer) -> Result<String> {
    let mut v = String::new();
    while !tz.done() && !tz.is_punct(';') {
        if tz.is_punct('}') {
            return tz.err("missing ';'");
        }
        if let Some(t) = tz.peek_at(0) {
            append_raw(&mut v, t);
        }
        tz.next()?;
    }
    if tz.done() {
        return tz.err("expected ';'");
    }
    tz.next()?;
    Ok(v)
}

/// Consume a preprocessor directive this reader cannot evaluate.
///
/// `#include`-style directives carry a bare file name and NO `;`, so blindly
/// skipping to the next `;` would swallow the real entry that follows.
fn skip_directive(tz: &mut Tokenizer, key: &str) -> Result<()> {
    eprintln!(
        "[ofgpu] {}: ignoring unsupported directive {}",
        tz.path(),
        key
    );

    if key.starts_with("#include") {
        if !tz.done() && !is_punct_any(tz) {
            tz.next()?;
        }
        if tz.is_punct(';') {
            tz.next()?;
        }
        return Ok(());
    }

    if !tz.done() && !tz.is_punct('}') {
        tz.skip_entry()?;
    }
    Ok(())
}

/// Read the `FoamFile` header, keeping it: `object` is what names the field
/// when the file has been renamed. A file without a header is not an error -
/// hand-written `0/` entries often lack one.
fn read_header(tz: &mut Tokenizer) -> Result<BTreeMap<String, String>> {
    let mut out = BTreeMap::new();
    if !tz.is_word("FoamFile") {
        return Ok(out);
    }
    tz.next()?;
    tz.expect_punct('{')?;

    while !tz.done() && !tz.is_punct('}') {
        if tz.is_punct(';') {
            tz.next()?;
            continue;
        }
        let k = tz.expect_word()?;
        if tz.is_punct('{') {
            tz.skip_entry()?;
            continue;
        }

        let mut v = String::new();
        while !tz.done() && !tz.is_punct(';') {
            if tz.is_punct('}') {
                return tz.err("missing ';' in FoamFile header");
            }
            if let Some(t) = tz.peek_at(0) {
                append_raw(&mut v, t);
            }
            tz.next()?;
        }
        if tz.done() {
            return tz.err("expected ';' in FoamFile header");
        }
        tz.next()?;
        out.insert(k, v);
    }
    tz.expect_punct('}')?;
    Ok(out)
}

fn parse_vec3(tz: &mut Tokenizer) -> Result<Vec3> {
    tz.expect_punct('(')?;
    let x = tz.expect_num()? as Scalar;
    let y = tz.expect_num()? as Scalar;
    let z = tz.expect_num()? as Scalar;
    tz.expect_punct(')')?;
    Ok(Vec3::new(x, y, z))
}

/// `N ( a b c )`, `N{a}` (the compact all-equal form), or `( a b c )`.
fn parse_scalar_list(tz: &mut Tokenizer, out: &mut Vec<Scalar>) -> Result<()> {
    if tz.is_punct('(') {
        tz.next()?;
        while !tz.done() && !tz.is_punct(')') {
            let v = tz.expect_num()? as Scalar;
            out.push(v);
        }
        return tz.expect_punct(')');
    }

    let n = tz.expect_label()?;
    if n < 0 {
        return tz.err("negative list size");
    }
    let n = n as usize;

    if tz.is_punct('{') {
        tz.next()?;
        let v = tz.expect_num()? as Scalar;
        tz.expect_punct('}')?;
        out.clear();
        out.resize(n, v);
        return Ok(());
    }

    tz.expect_punct('(')?;
    // Bounded reserve: a corrupt count must not turn into a huge allocation
    // before a single value has been read.
    out.reserve(n.min(1 << 20));
    for _ in 0..n {
        let v = tz.expect_num()? as Scalar;
        out.push(v);
    }
    tz.expect_punct(')')
}

fn parse_vector_list(tz: &mut Tokenizer, out: &mut Vec<Vec3>) -> Result<()> {
    if tz.is_punct('(') {
        tz.next()?;
        while !tz.done() && !tz.is_punct(')') {
            let v = parse_vec3(tz)?;
            out.push(v);
        }
        return tz.expect_punct(')');
    }

    let n = tz.expect_label()?;
    if n < 0 {
        return tz.err("negative list size");
    }
    let n = n as usize;

    if tz.is_punct('{') {
        tz.next()?;
        let v = parse_vec3(tz)?;
        tz.expect_punct('}')?;
        out.clear();
        out.resize(n, v);
        return Ok(());
    }

    tz.expect_punct('(')?;
    out.reserve(n.min(1 << 20));
    for _ in 0..n {
        let v = parse_vec3(tz)?;
        out.push(v);
    }
    tz.expect_punct(')')
}

/// One `uniform x;` / `nonuniform List<scalar> N ( ... );` entry.
/// A length-1 result is how `uniform` is recorded.
fn parse_scalar_entry(tz: &mut Tokenizer) -> Result<Vec<Scalar>> {
    let mut out = Vec::new();

    if tz.is_word("uniform") {
        tz.next()?;
        let v = tz.expect_num()? as Scalar;
        out.push(v);
    } else if tz.is_word("nonuniform") {
        tz.next()?;
        // Optional `List<scalar>`: absent in the empty-patch form
        // `nonuniform 0()`, where the next token is already the count.
        if is_word_tok(tz) {
            tz.next()?;
        }
        parse_scalar_list(tz, &mut out)?;
    } else if tz.is_punct('(') {
        parse_scalar_list(tz, &mut out)?;
    } else if is_num_at(tz, 0) {
        // `4;` is a value, `4 (...)` is a sized list - one token is not
        // enough to tell them apart.
        if tz.is_punct_at(';', 1) {
            let v = tz.expect_num()? as Scalar;
            out.push(v);
        } else {
            parse_scalar_list(tz, &mut out)?;
        }
    } else {
        return tz.err("expected 'uniform' or 'nonuniform'");
    }

    tz.expect_punct(';')?;
    Ok(out)
}

fn parse_vector_entry(tz: &mut Tokenizer) -> Result<Vec<Vec3>> {
    let mut out = Vec::new();

    if tz.is_word("uniform") {
        tz.next()?;
        let v = parse_vec3(tz)?;
        out.push(v);
    } else if tz.is_word("nonuniform") {
        tz.next()?;
        if is_word_tok(tz) {
            tz.next()?;
        }
        parse_vector_list(tz, &mut out)?;
    } else if tz.is_punct('(') {
        // `(1 2 3)` is one vector; `((1 2 3) (4 5 6))` is a list of them.
        if tz.is_punct_at('(', 1) {
            parse_vector_list(tz, &mut out)?;
        } else {
            let v = parse_vec3(tz)?;
            out.push(v);
        }
    } else if is_num_at(tz, 0) {
        parse_vector_list(tz, &mut out)?;
    } else {
        return tz.err("expected 'uniform' or 'nonuniform'");
    }

    tz.expect_punct(';')?;
    Ok(out)
}

fn parse_patch_spec(tz: &mut Tokenizer, s: &mut PatchFieldSpec, is_vector: bool) -> Result<()> {
    tz.expect_punct('{')?;

    while !tz.done() && !tz.is_punct('}') {
        if tz.is_punct(';') {
            tz.next()?;
            continue;
        }
        if is_punct_any(tz) {
            return tz.err("expected a keyword");
        }

        let k = tz.expect_word()?;
        if k.starts_with('#') {
            skip_directive(tz, &k)?;
            continue;
        }

        match k.as_str() {
            "type" => s.type_name = gather_raw(tz)?,
            "neighbourPatch" => s.neighbour_patch = Some(gather_raw(tz)?),
            // Always a scalar, even on a vector field.
            "valueFraction" => s.value_fraction = parse_scalar_entry(tz)?,
            // `refGrad` is the spelling some older cases use for refGradient.
            "value" | "gradient" | "refValue" | "refGradient" | "refGrad" | "inletValue" => {
                if is_vector {
                    let v = parse_vector_entry(tz)?;
                    match k.as_str() {
                        "value" => s.value_v = v,
                        "gradient" => s.gradient_v = v,
                        "refValue" => s.ref_value_v = v,
                        "inletValue" => s.inlet_value_v = v,
                        _ => s.ref_gradient_v = v,
                    }
                } else {
                    let v = parse_scalar_entry(tz)?;
                    match k.as_str() {
                        "value" => s.value = v,
                        "gradient" => s.gradient = v,
                        "refValue" => s.ref_value = v,
                        "inletValue" => s.inlet_value = v,
                        _ => s.ref_gradient = v,
                    }
                }
            }
            // Everything else is kept verbatim rather than discarded.
            // `turbulentIntensity`, `mixingLength`, `p0`, `volumetricFlowRate`
            // and the rest are each read by exactly one condition, and a
            // condition that cannot see the entry it is defined by is the
            // silent substitution SPEC-LIT §13.4 forbids.
            _ => {
                if tz.is_punct('{') {
                    tz.skip_entry()?;
                } else {
                    let v = gather_raw(tz)?;
                    s.extra.insert(k, v);
                }
            }
        }
    }

    tz.expect_punct('}')
}

fn parse_boundary_field(
    tz: &mut Tokenizer,
    out: &mut BTreeMap<String, PatchFieldSpec>,
    patterns: &mut Vec<String>,
    is_vector: bool,
) -> Result<()> {
    tz.expect_punct('{')?;

    while !tz.done() && !tz.is_punct('}') {
        if tz.is_punct(';') {
            tz.next()?;
            continue;
        }
        if is_punct_any(tz) {
            return tz.err("expected a patch name");
        }

        // A quoted key is a regular expression, not a patch name. The quotes
        // are kept so the two can never be confused, and the order is kept
        // because it is the tie-break between two patterns that both match.
        let (raw, quoted) = tz.expect_key()?;
        let pname = if quoted {
            format!("\"{raw}\"")
        } else {
            raw.clone()
        };

        if pname.starts_with('#') {
            skip_directive(tz, &pname)?;
            continue;
        }

        if !tz.is_punct('{') {
            // e.g. `$otherPatch;` - not modelled, skip it.
            tz.skip_entry()?;
            continue;
        }

        let mut s = PatchFieldSpec::default();
        parse_patch_spec(tz, &mut s, is_vector)?;
        if quoted && !patterns.contains(&pname) {
            patterns.push(pname.clone());
        }
        out.insert(pname, s);
    }

    tz.expect_punct('}')
}

// ==========================================================================
//  Writing
// ==========================================================================

/// The face flux `phi`, as a `surfaceScalarField` file.
///
/// `f.internal` is one value per INTERNAL face and each `boundaryField`
/// entry's `value` is one per boundary face of its patch - the layout
/// [`read_scalar_field`] reads back when it is called with
/// `HostMesh::n_internal_faces`, so what this writes is exactly what a restart
/// consumes.
///
/// *DESIGN.* An `empty` patch keeps its `empty` type and carries no `value`:
/// its flux is identically zero, so nothing is lost, and a `calculated` face
/// value on an empty patch is not something another tool would accept. Every
/// other patch - `cyclic` included - is written `calculated` with its face
/// values, because dropping a cyclic patch's flux would break the round trip
/// this function exists for.
pub fn write_surface_scalar_field(path: &Path, f: &RawScalarField, time: &str) -> Result<()> {
    write_surface_scalar_field_prec(path, f, time, PHI_PRECISION, true)
}

/// *DESIGN.* `phi` is written at ROUND-TRIP precision, not at `controlDict`'s
/// `writePrecision`, and it is the only field in this solver that is.
///
/// Every other field is a number: losing the seventeenth digit of a
/// temperature costs nothing anyone can measure. `phi` is not a number, it is
/// a discrete CONSTRAINT - `Σ_f (±phi_f) = 0` in every cell - and that
/// constraint is a cancellation between face values of similar size. Rounded
/// to six significant digits it stops holding at the seventh, so a run
/// restarted from a six-digit `phi` begins with a continuity error of about
/// `1e-6 |phi|` per cell that the first pressure solve then has to remove.
/// Seventeen digits is what `f64` round-trips through decimal exactly, so the
/// flux read back is the flux written, bit for bit, and the restart's first
/// pressure residual is the one the run that wrote it would have seen.
pub const PHI_PRECISION: usize = 17;

/// As [`write_surface_scalar_field`], with `controlDict`'s `writePrecision`.
pub fn write_surface_scalar_field_prec(
    path: &Path,
    f: &RawScalarField,
    time: &str,
    precision: usize,
    collapse_uniform: bool,
) -> Result<()> {
    let obj = if f.name.is_empty() {
        base_name(path)
    } else {
        f.name.clone()
    };

    let mut out = String::new();
    write_foamfile_header(&mut out, "surfaceScalarField", time, &obj);
    write_dimensions(&mut out, &f.dimensions);

    write_scalar_entry(
        &mut out,
        "",
        "internalField",
        &f.internal,
        precision,
        collapse_uniform,
    );
    out.push('\n');

    write_boundary_field(&mut out, &f.boundary, false, precision, collapse_uniform);
    out.push_str(FOOTER);

    write_all(path, &out)
}

/// Write a `volScalarField` at OpenFOAM's default `writePrecision` of 6.
pub fn write_scalar_field(path: &Path, f: &RawScalarField, time: &str) -> Result<()> {
    write_scalar_field_prec(path, f, time, 6, true)
}

/// Write a `volVectorField` at OpenFOAM's default `writePrecision` of 6.
pub fn write_vector_field(path: &Path, f: &RawVectorField, time: &str) -> Result<()> {
    write_vector_field_prec(path, f, time, 6, true)
}

/// As [`write_scalar_field`], with `controlDict`'s `writePrecision` and the
/// choice of collapsing an all-equal list back to `uniform x`.
pub fn write_scalar_field_prec(
    path: &Path,
    f: &RawScalarField,
    time: &str,
    precision: usize,
    collapse_uniform: bool,
) -> Result<()> {
    let obj = if f.name.is_empty() {
        base_name(path)
    } else {
        f.name.clone()
    };

    let mut out = String::new();
    write_foamfile_header(&mut out, "volScalarField", time, &obj);
    write_dimensions(&mut out, &f.dimensions);

    write_scalar_entry(
        &mut out,
        "",
        "internalField",
        &f.internal,
        precision,
        collapse_uniform,
    );
    out.push('\n');

    write_boundary_field(&mut out, &f.boundary, false, precision, collapse_uniform);
    out.push_str(FOOTER);

    write_all(path, &out)
}

/// As [`write_vector_field`], with an explicit precision.
pub fn write_vector_field_prec(
    path: &Path,
    f: &RawVectorField,
    time: &str,
    precision: usize,
    collapse_uniform: bool,
) -> Result<()> {
    let obj = if f.name.is_empty() {
        base_name(path)
    } else {
        f.name.clone()
    };

    let mut out = String::new();
    write_foamfile_header(&mut out, "volVectorField", time, &obj);
    write_dimensions(&mut out, &f.dimensions);

    write_vector_entry(
        &mut out,
        "",
        "internalField",
        &f.internal,
        precision,
        collapse_uniform,
    );
    out.push('\n');

    write_boundary_field(&mut out, &f.boundary, true, precision, collapse_uniform);
    out.push_str(FOOTER);

    write_all(path, &out)
}

const BANNER: &str = r#"/*---------------------------------------------------------------------------*\
| ofgpu  --  GPU-native finite volume CFD                                     |
|                                                                             |
| Written in the OpenFOAM ASCII case format so that existing pre- and         |
| post-processing tools can read it. A file format is not a work: ofgpu is    |
| an independent implementation, neither derived from nor affiliated with     |
| OpenFOAM.                                                                   |
\*---------------------------------------------------------------------------*/
"#;

const SEPARATOR: &str =
    "// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //\n\n";

const FOOTER: &str =
    "\n\n// ************************************************************************* //\n";

/// OpenFOAM pads a keyword to 16 characters in the body of a file and to 12
/// inside the `FoamFile` header; matching that keeps a diff against a foamRun
/// result down to the numbers.
fn write_keyword(out: &mut String, indent: &str, kw: &str, width: usize) {
    out.push_str(indent);
    out.push_str(kw);
    if kw.len() < width {
        for _ in 0..(width - kw.len()) {
            out.push(' ');
        }
    } else {
        out.push(' ');
    }
}

fn write_foamfile_header(out: &mut String, cls: &str, loc: &str, obj: &str) {
    out.push_str(BANNER);
    out.push_str("FoamFile\n{\n");
    write_keyword(out, "    ", "format", 12);
    out.push_str("ascii;\n");
    write_keyword(out, "    ", "class", 12);
    out.push_str(cls);
    out.push_str(";\n");
    write_keyword(out, "    ", "location", 12);
    out.push('"');
    out.push_str(loc);
    out.push_str("\";\n");
    write_keyword(out, "    ", "object", 12);
    out.push_str(obj);
    out.push_str(";\n");
    out.push_str("}\n");
    out.push_str(SEPARATOR);
}

/// A dimensionless field still needs the entry - `foamToVTK` rejects a
/// `volScalarField` without one.
fn write_dimensions(out: &mut String, dims: &str) {
    write_keyword(out, "", "dimensions", 16);
    out.push_str(if dims.is_empty() {
        "[0 0 0 0 0 0 0]"
    } else {
        dims
    });
    out.push_str(";\n\n");
}

fn all_equal_scalar(v: &[Scalar]) -> bool {
    v.iter().all(|x| *x == v[0])
}

fn all_equal_vec3(v: &[Vec3]) -> bool {
    v.iter()
        .all(|x| x.x == v[0].x && x.y == v[0].y && x.z == v[0].z)
}

fn push_vec3(out: &mut String, v: &Vec3, p: usize) {
    out.push('(');
    out.push_str(&fmt_g(v.x, p));
    out.push(' ');
    out.push_str(&fmt_g(v.y, p));
    out.push(' ');
    out.push_str(&fmt_g(v.z, p));
    out.push(')');
}

fn write_scalar_entry(
    out: &mut String,
    indent: &str,
    kw: &str,
    v: &[Scalar],
    p: usize,
    collapse: bool,
) {
    write_keyword(out, indent, kw, 16);

    if v.is_empty() {
        out.push_str("nonuniform 0();\n");
        return;
    }
    if v.len() == 1 || (collapse && all_equal_scalar(v)) {
        out.push_str("uniform ");
        out.push_str(&fmt_g(v[0], p));
        out.push_str(";\n");
        return;
    }

    out.push_str("nonuniform List<scalar> \n");
    out.push_str(&v.len().to_string());
    out.push_str("\n(\n");
    for x in v {
        out.push_str(&fmt_g(*x, p));
        out.push('\n');
    }
    out.push_str(")\n;\n");
}

fn write_vector_entry(
    out: &mut String,
    indent: &str,
    kw: &str,
    v: &[Vec3],
    p: usize,
    collapse: bool,
) {
    write_keyword(out, indent, kw, 16);

    if v.is_empty() {
        out.push_str("nonuniform 0();\n");
        return;
    }
    if v.len() == 1 || (collapse && all_equal_vec3(v)) {
        out.push_str("uniform ");
        push_vec3(out, &v[0], p);
        out.push_str(";\n");
        return;
    }

    out.push_str("nonuniform List<vector> \n");
    out.push_str(&v.len().to_string());
    out.push_str("\n(\n");
    for x in v {
        push_vec3(out, x, p);
        out.push('\n');
    }
    out.push_str(")\n;\n");
}

/// Constraint and pure-derivative patch types for which OpenFOAM itself writes
/// no `value` entry; emitting one there is at best noise.
fn suppress_value(type_name: &str) -> bool {
    matches!(
        type_name,
        "empty"
            | "cyclic"
            | "cyclicSlip"
            | "processor"
            | "wedge"
            | "symmetry"
            | "symmetryPlane"
            | "zeroGradient"
            | "noSlip"
            | "slip"
    )
}

fn needs_quoting(s: &str) -> bool {
    if s.is_empty() {
        return true;
    }
    !s.chars()
        .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '.' || c == '-')
}

fn write_boundary_field(
    out: &mut String,
    b: &BTreeMap<String, PatchFieldSpec>,
    is_vector: bool,
    p: usize,
    collapse: bool,
) {
    out.push_str("boundaryField\n{\n");

    for (name, s) in b {
        let type_name = if s.type_name.is_empty() {
            "calculated"
        } else {
            s.type_name.as_str()
        };

        // A patch name that came from a regex key has to be re-quoted or it
        // will not tokenise back into one name.
        if needs_quoting(name) {
            out.push_str("    \"");
            out.push_str(name);
            out.push_str("\"\n");
        } else {
            out.push_str("    ");
            out.push_str(name);
            out.push('\n');
        }
        out.push_str("    {\n");
        write_keyword(out, "        ", "type", 16);
        out.push_str(type_name);
        out.push_str(";\n");

        if let Some(np) = &s.neighbour_patch {
            if !np.is_empty() {
                write_keyword(out, "        ", "neighbourPatch", 16);
                out.push_str(np);
                out.push_str(";\n");
            }
        }

        if is_vector {
            if !s.ref_value_v.is_empty() {
                write_vector_entry(out, "        ", "refValue", &s.ref_value_v, p, collapse);
            }
            if !s.ref_gradient_v.is_empty() {
                write_vector_entry(out, "        ", "refGradient", &s.ref_gradient_v, p, collapse);
            }
            if !s.value_fraction.is_empty() {
                write_scalar_entry(
                    out,
                    "        ",
                    "valueFraction",
                    &s.value_fraction,
                    p,
                    collapse,
                );
            }
            if !s.inlet_value_v.is_empty() {
                write_vector_entry(out, "        ", "inletValue", &s.inlet_value_v, p, collapse);
            }
            if !s.gradient_v.is_empty() {
                write_vector_entry(out, "        ", "gradient", &s.gradient_v, p, collapse);
            }
            if !s.value_v.is_empty() && !suppress_value(type_name) {
                write_vector_entry(out, "        ", "value", &s.value_v, p, collapse);
            }
        } else {
            if !s.ref_value.is_empty() {
                write_scalar_entry(out, "        ", "refValue", &s.ref_value, p, collapse);
            }
            if !s.ref_gradient.is_empty() {
                write_scalar_entry(out, "        ", "refGradient", &s.ref_gradient, p, collapse);
            }
            if !s.value_fraction.is_empty() {
                write_scalar_entry(
                    out,
                    "        ",
                    "valueFraction",
                    &s.value_fraction,
                    p,
                    collapse,
                );
            }
            if !s.inlet_value.is_empty() {
                write_scalar_entry(out, "        ", "inletValue", &s.inlet_value, p, collapse);
            }
            if !s.gradient.is_empty() {
                write_scalar_entry(out, "        ", "gradient", &s.gradient, p, collapse);
            }
            if !s.value.is_empty() && !suppress_value(type_name) {
                write_scalar_entry(out, "        ", "value", &s.value, p, collapse);
            }
        }

        out.push_str("    }\n");
    }

    out.push_str("}\n");
}

// ==========================================================================
//  Files and number formatting
// ==========================================================================

fn base_name(path: &Path) -> String {
    path.file_name()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_default()
}

fn write_all(path: &Path, text: &str) -> Result<()> {
    if let Some(dir) = path.parent() {
        if !dir.as_os_str().is_empty() {
            std::fs::create_dir_all(dir).path(dir)?;
        }
    }
    // Bytes, not a text stream: OpenFOAM files are LF everywhere, including
    // on Windows, and foamToVTK's parser is happier for it.
    std::fs::write(path, text.as_bytes()).path(path)
}

/// C++'s `operator<<(double)` under `setprecision(n)`, i.e. `printf("%.ng")`:
/// `n` significant digits, exponential form outside `[1e-4, 1e{n})`, trailing
/// zeros stripped.
///
/// Spelled out because Rust has no `%g`: `{}` would print `1e-5` as
/// `0.00001` (and a denormal in full), while `{:.6}` is six *decimals*, not
/// six significant digits - the first bloats the file, the second silently
/// flattens small values to zero.
fn fmt_g(v: Scalar, precision: usize) -> String {
    let x = v as f64;
    if x.is_nan() {
        return "nan".to_string();
    }
    if x.is_infinite() {
        return if x < 0.0 { "-inf" } else { "inf" }.to_string();
    }

    let p = precision.max(1);

    // The exponent AFTER rounding to p significant digits is what picks the
    // style, and that is exactly what `{:e}` reports.
    let sci = format!("{:.*e}", p - 1, x);
    let (mant, exp) = match sci.split_once('e') {
        Some((m, e)) => (m, e.parse::<i32>().unwrap_or(0)),
        None => return sci,
    };

    if exp < -4 || exp >= p as i32 {
        // printf pads the exponent to two digits; `{:e}` does not.
        let sign = if exp < 0 { '-' } else { '+' };
        return format!("{}e{}{:02}", trim_zeros(mant), sign, exp.abs());
    }

    let decimals = (p as i32 - 1 - exp).max(0) as usize;
    trim_zeros(&format!("{:.*}", decimals, x)).to_string()
}

fn trim_zeros(s: &str) -> &str {
    if !s.contains('.') {
        return s;
    }
    s.trim_end_matches('0').trim_end_matches('.')
}

// ==========================================================================
//  Tests
// ==========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn scratch(tag: &str) -> PathBuf {
        let mut d = std::env::temp_dir();
        d.push(format!("ofgpu_fields_{}_{}", std::process::id(), tag));
        d
    }

    fn read_scalar_str(src: &str, n: usize) -> Result<RawScalarField> {
        parse_scalar_field(src, "test.p", "p".to_string(), n)
    }

    fn read_vector_str(src: &str, n: usize) -> Result<RawVectorField> {
        parse_vector_field(src, "test.U", "U".to_string(), n)
    }

    fn patch(t: &str, value: &[Scalar]) -> PatchFieldSpec {
        PatchFieldSpec {
            type_name: t.to_string(),
            value: value.to_vec(),
            ..Default::default()
        }
    }

    /// The whole point of the writer is that what comes back is what went in:
    /// a drifting number format or a dropped keyword restarts a run from
    /// different data, and nothing downstream can notice.
    #[test]
    fn scalar_field_round_trips() {
        let mut f = RawScalarField {
            name: "k".to_string(),
            dimensions: "[0 2 -2 0 0 0 0]".to_string(),
            internal: vec![1.5, -2.25, 0.125, 1.0e-5, 3.0],
            boundary: BTreeMap::new(),
            boundary_patterns: Vec::new(),
        };
        f.boundary
            .insert("inlet".to_string(), patch("fixedValue", &[0.375]));
        f.boundary
            .insert("outlet".to_string(), patch("zeroGradient", &[]));
        f.boundary.insert(
            "wall".to_string(),
            patch("fixedValue", &[1.0, 2.0, 4.5, -0.5]),
        );

        let dir = scratch("scalar_rt");
        let path = dir.join("0.5").join("k");
        write_scalar_field(&path, &f, "0.5").unwrap();

        let back = read_scalar_field(&path, 0).unwrap();
        assert_eq!(back, f);

        // Expansion must leave an already-sized list alone.
        let sized = read_scalar_field(&path, 5).unwrap();
        assert_eq!(sized.internal, f.internal);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn vector_field_round_trips() {
        let mut f = RawVectorField {
            name: "U".to_string(),
            dimensions: "[0 1 -1 0 0 0 0]".to_string(),
            internal: vec![
                Vec3::new(1.0, 0.0, 0.0),
                Vec3::new(-0.25, 2.5, 0.0),
                Vec3::new(0.0, 0.0, 1.0e-5),
            ],
            boundary: BTreeMap::new(),
            boundary_patterns: Vec::new(),
        };
        f.boundary.insert(
            "inlet".to_string(),
            PatchFieldSpec {
                type_name: "fixedValue".to_string(),
                value_v: vec![Vec3::new(10.0, 0.0, 0.0)],
                ..Default::default()
            },
        );
        f.boundary.insert(
            "outlet".to_string(),
            PatchFieldSpec {
                type_name: "inletOutlet".to_string(),
                inlet_value_v: vec![Vec3::ZERO],
                value_v: vec![Vec3::new(1.0, 2.0, 3.0), Vec3::new(4.0, 5.0, 6.5)],
                ..Default::default()
            },
        );
        f.boundary.insert(
            "side".to_string(),
            PatchFieldSpec {
                type_name: "cyclic".to_string(),
                neighbour_patch: Some("side2".to_string()),
                ..Default::default()
            },
        );

        let dir = scratch("vector_rt");
        let path = dir.join("0").join("U");
        write_vector_field(&path, &f, "0").unwrap();

        let back = read_vector_field(&path, 0).unwrap();
        assert_eq!(back, f);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn uniform_internal_nonuniform_patch() {
        let src = "\
FoamFile { version 2.0; format ascii; class volScalarField; object p; }
dimensions      [0 2 -2 0 0 0 0];
internalField   uniform 3.5;
boundaryField
{
    inlet
    {
        type            fixedValue;
        value           nonuniform List<scalar> 3 (1 2 3);
    }
}
";
        let f = read_scalar_str(src, 4).unwrap();
        assert_eq!(f.name, "p");
        assert_eq!(f.dimensions, "[0 2 -2 0 0 0 0]");
        assert_eq!(f.internal, vec![3.5; 4]);
        // The patch list stays verbatim - expansion is the caller's job.
        assert_eq!(f.boundary["inlet"].value, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn nonuniform_internal_uniform_patch() {
        let src = "\
internalField   nonuniform List<scalar>
3
(
1.5
2.5
3.5
)
;
boundaryField
{
    wall
    {
        type            fixedValue;
        value           uniform 0;
    }
}
";
        let f = read_scalar_str(src, 3).unwrap();
        assert_eq!(f.internal, vec![1.5, 2.5, 3.5]);
        // Length 1 is how the reader records `uniform`.
        assert_eq!(f.boundary["wall"].value, vec![0.0]);
    }

    #[test]
    fn vector_uniform_internal_and_nonuniform_patch() {
        let src = "\
internalField   uniform (1 2 3);
boundaryField
{
    inlet
    {
        type            fixedValue;
        value           nonuniform List<vector> 2 ((1 0 0) (0 1 0));
    }
}
";
        let f = read_vector_str(src, 2).unwrap();
        assert_eq!(f.internal, vec![Vec3::new(1.0, 2.0, 3.0); 2]);
        assert_eq!(
            f.boundary["inlet"].value_v,
            vec![Vec3::new(1.0, 0.0, 0.0), Vec3::new(0.0, 1.0, 0.0)]
        );
    }

    /// inletOutlet's extra key drives the numerics - it becomes `refValue` in
    /// the mixed form - so dropping it changes the answer without any error.
    #[test]
    fn inlet_outlet_and_mixed_keys() {
        let src = "\
internalField   uniform 0;
boundaryField
{
    outlet
    {
        type            inletOutlet;
        inletValue      uniform 0.001;
        value           uniform 0.05;
    }
    mixedPatch
    {
        type            mixed;
        refValue        uniform 1;
        refGradient     uniform 0;
        valueFraction   nonuniform List<scalar> 2 (0.25 0.75);
        value           uniform 1;
    }
}
";
        let f = read_scalar_str(src, 1).unwrap();
        let o = &f.boundary["outlet"];
        assert_eq!(o.type_name, "inletOutlet");
        assert_eq!(o.inlet_value, vec![0.001]);
        assert_eq!(o.value, vec![0.05]);

        let m = &f.boundary["mixedPatch"];
        assert_eq!(m.ref_value, vec![1.0]);
        assert_eq!(m.ref_gradient, vec![0.0]);
        assert_eq!(m.value_fraction, vec![0.25, 0.75]);
    }

    /// What OpenFOAM writes for a patch with no faces. `List<scalar>` is
    /// missing here, so a reader that assumed it would desynchronise and
    /// mis-parse every entry that follows.
    #[test]
    fn empty_patch_lists_parse() {
        let src = "\
internalField   uniform 0;
boundaryField
{
    emptyPatch
    {
        type            calculated;
        value           nonuniform 0();
    }
    next
    {
        type            fixedValue;
        value           uniform 7;
    }
}
";
        let f = read_scalar_str(src, 1).unwrap();
        assert!(f.boundary["emptyPatch"].value.is_empty());
        assert_eq!(f.boundary["next"].value, vec![7.0]);

        let vsrc = "\
internalField   uniform (0 0 0);
boundaryField
{
    emptyPatch
    {
        type            calculated;
        value           nonuniform 0();
    }
    next
    {
        type            fixedValue;
        value           uniform (1 0 0);
    }
}
";
        let v = read_vector_str(vsrc, 1).unwrap();
        assert!(v.boundary["emptyPatch"].value_v.is_empty());
        assert_eq!(v.boundary["next"].value_v, vec![Vec3::new(1.0, 0.0, 0.0)]);
    }

    #[test]
    fn compact_all_equal_list_form() {
        let src = "internalField nonuniform List<scalar> 4{2.5};\nboundaryField {}\n";
        let f = read_scalar_str(src, 4).unwrap();
        assert_eq!(f.internal, vec![2.5; 4]);
    }

    #[test]
    fn unknown_entries_and_comments_are_skipped() {
        let src = "\
// leading comment with a / in it
FoamFile { format ascii; class volScalarField; location \"0/x\"; object nut; }
/* block
   comment */
dimensions      [0 2 -1 0 0 0 0];
boundaryValues  { nested { thing 1; } };
internalField   uniform 0.1;
boundaryField {}
";
        let f = read_scalar_str(src, 2).unwrap();
        assert_eq!(f.name, "nut");
        assert_eq!(f.dimensions, "[0 2 -1 0 0 0 0]");
        assert_eq!(f.internal, vec![0.1, 0.1]);
    }

    #[test]
    fn missing_internal_field_is_an_error() {
        let src = "dimensions [0 0 0 0 0 0 0];\nboundaryField {}\n";
        assert!(read_scalar_str(src, 1).is_err());
    }

    #[test]
    fn binary_format_is_rejected_with_advice() {
        let src = "FoamFile { format binary; class volScalarField; object p; }\n";
        let e = read_scalar_str(src, 1).unwrap_err().to_string();
        assert!(e.contains("foamFormatConvert"), "{}", e);
    }

    #[test]
    fn wrong_length_list_is_an_error() {
        let src = "internalField nonuniform List<scalar> 3 (1 2 3);\nboundaryField {}\n";
        assert!(read_scalar_str(src, 3).is_ok());
        // A field that does not match the mesh must not be silently padded.
        assert!(read_scalar_str(src, 4).is_err());
    }

    #[test]
    fn expansion_rules() {
        assert_eq!(expand_scalars(&[2.0], 3, "x").unwrap(), vec![2.0; 3]);
        assert_eq!(expand_scalars(&[], 2, "x").unwrap(), vec![0.0, 0.0]);
        assert_eq!(expand_scalars(&[1.0, 2.0], 2, "x").unwrap(), vec![1.0, 2.0]);
        assert!(expand_scalars(&[1.0, 2.0], 3, "x").is_err());

        assert_eq!(
            expand_vectors(&[Vec3::new(1.0, 2.0, 3.0)], 2, "x").unwrap(),
            vec![Vec3::new(1.0, 2.0, 3.0); 2]
        );
        assert_eq!(expand_vectors(&[], 1, "x").unwrap(), vec![Vec3::ZERO]);
        assert!(expand_vectors(&[Vec3::ZERO, Vec3::ZERO], 3, "x").is_err());
    }

    /// `%g` is not a Rust format, and getting it wrong writes files that
    /// either lose small numbers entirely or bloat to hundreds of megabytes.
    #[test]
    fn number_format_matches_printf_g() {
        assert_eq!(fmt_g(0.0, 6), "0");
        assert_eq!(fmt_g(1.0, 6), "1");
        assert_eq!(fmt_g(1.5, 6), "1.5");
        assert_eq!(fmt_g(-2.25, 6), "-2.25");
        assert_eq!(fmt_g(1.0e-5, 6), "1e-05");
        assert_eq!(fmt_g(1.0e20, 6), "1e+20");
        assert_eq!(fmt_g(0.0001, 6), "0.0001");
        assert_eq!(fmt_g(123456.0, 6), "123456");
        assert_eq!(fmt_g(1234567.0, 6), "1.23457e+06");
        assert_eq!(fmt_g(1.0 / 3.0, 6), "0.333333");
        assert_eq!(fmt_g(1.0 / 3.0, 12), "0.333333333333");
    }

    /// A collapsed list has to read back as `uniform`, and an unequal one must
    /// not collapse - that would quietly flatten a solved field.
    #[test]
    fn uniform_collapse_only_when_every_entry_matches() {
        let dir = scratch("collapse");

        let f = RawScalarField {
            name: "p".to_string(),
            dimensions: "[0 2 -2 0 0 0 0]".to_string(),
            internal: vec![4.0; 6],
            boundary: BTreeMap::new(),
            boundary_patterns: Vec::new(),
        };
        let path = dir.join("p");
        write_scalar_field(&path, &f, "0").unwrap();
        let text = std::fs::read_to_string(&path).unwrap();
        assert!(text.contains("internalField   uniform 4;"), "{}", text);
        // Collapsed on disk, re-expanded on the way back in.
        assert_eq!(read_scalar_field(&path, 6).unwrap().internal, f.internal);

        let g = RawScalarField {
            internal: vec![4.0, 4.0, 4.5],
            ..f.clone()
        };
        write_scalar_field(&path, &g, "0").unwrap();
        let text = std::fs::read_to_string(&path).unwrap();
        assert!(text.contains("nonuniform List<scalar>"), "{}", text);
        assert_eq!(read_scalar_field(&path, 3).unwrap().internal, g.internal);

        let _ = std::fs::remove_dir_all(&dir);
    }

    /// ParaView and foamToVTK both refuse a file whose FoamFile header is
    /// missing or malformed, and the failure reads like a corrupt case.
    #[test]
    fn written_header_is_complete() {
        let dir = scratch("header");
        let path = dir.join("2.5").join("epsilon");
        let f = RawScalarField {
            name: "epsilon".to_string(),
            dimensions: "[0 2 -3 0 0 0 0]".to_string(),
            internal: vec![1.0],
            boundary: BTreeMap::new(),
            boundary_patterns: Vec::new(),
        };
        write_scalar_field(&path, &f, "2.5").unwrap();

        let text = std::fs::read_to_string(&path).unwrap();
        assert!(text.starts_with("/*----"), "{}", text);
        assert!(text.contains("FoamFile\n{\n"));
        assert!(text.contains("format      ascii;"));
        assert!(text.contains("class       volScalarField;"));
        assert!(text.contains("location    \"2.5\";"));
        assert!(text.contains("object      epsilon;"));
        assert!(text.contains("dimensions      [0 2 -3 0 0 0 0];"));
        assert!(text.ends_with("//\n"));

        let _ = std::fs::remove_dir_all(&dir);
    }
}
