// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.
// Provenance: see PROVENANCE.md. No GPL-licensed source was consulted.

//! SPEC-LIT 120.11 - the inflow database `timeVaryingMappedFixedValue` reads:
//! `constant/boundaryData/<patch>/points` (the sample positions, a list of
//! vectors) and one folder per time, `<patch>/<time>/<field>` (one value per
//! sample point).
//!
//! FORMAT, as the files appear in data: an optional `FoamFile` header, then
//! `N ( ... )` or a bare `( ... )`. A value file may carry one value of the
//! field's rank before its list - the file's average - which is read past and
//! not used (`setAverage` is refused by name, SPEC-LIT 120.11). The point list
//! is the `points` reader of [`crate::io::polymesh`], straight into f64.
//!
//! No GPL-licensed source was consulted.

use std::path::{Path, PathBuf};

use crate::error::{parse_err, Error, Result};
use crate::io::tokenizer::{self, Tok, Tokenizer};
use crate::types::DVec3;

/// The sample positions of `<dir>/points`, in f64 (SPEC-LIT 118.2).
pub fn read_points(path: &Path) -> Result<Vec<DVec3>> {
    crate::io::polymesh::read_points_file(path)
}

/// The time folders of `dir` (a `boundaryData/<patch>` folder): every
/// sub-folder whose whole name parses as a finite number, sorted by that
/// number. Two folders naming the same time (`1` and `1.0`) are refused by
/// name, as is a folder holding no time at all.
pub fn time_folders(dir: &Path) -> Result<Vec<(f64, PathBuf)>> {
    let rd = std::fs::read_dir(dir).map_err(|e| Error::Parse {
        path: dir.display().to_string(),
        msg: format!("the boundaryData folder cannot be read: {e} (SPEC-LIT 120.11)"),
    })?;
    let mut out: Vec<(f64, PathBuf)> = Vec::new();
    for e in rd.flatten() {
        let p = e.path();
        if !p.is_dir() {
            continue;
        }
        let Some(name) = p.file_name().and_then(|s| s.to_str()) else { continue };
        let Ok(t) = name.parse::<f64>() else { continue };
        if !t.is_finite() {
            continue;
        }
        out.push((t, p));
    }
    out.sort_by(|a, b| a.0.total_cmp(&b.0));
    for w in out.windows(2) {
        if w[0].0 == w[1].0 {
            return parse_err(
                dir,
                format!(
                    "the time folders `{}` and `{}` name the same time {} (SPEC-LIT 120.11)",
                    w[0].1.display(),
                    w[1].1.display(),
                    w[0].0
                ),
            );
        }
    }
    if out.is_empty() {
        return parse_err(
            dir,
            "the boundaryData folder holds no time folder - at least one `<time>/<field>` \
             is needed (SPEC-LIT 120.11)",
        );
    }
    Ok(out)
}

/// One value of rank `rank` (1 or 3) into `out`.
fn read_value(ts: &mut Tokenizer, rank: usize, out: &mut Vec<f64>) -> Result<()> {
    if rank == 1 {
        out.push(ts.expect_num()?);
        return Ok(());
    }
    ts.expect_punct('(')?;
    for _ in 0..3 {
        out.push(ts.expect_num()?);
    }
    ts.expect_punct(')')
}

/// The values of a bare `( ... )` list, the `(` not yet consumed.
fn read_list(ts: &mut Tokenizer, rank: usize) -> Result<Vec<f64>> {
    ts.expect_punct('(')?;
    let mut out = Vec::new();
    while !ts.done() && !ts.is_punct(')') {
        read_value(ts, rank, &mut out)?;
    }
    ts.expect_punct(')')?;
    Ok(out)
}

/// A non-negative whole number that a list count can be.
fn as_count(x: f64) -> Option<usize> {
    if x >= 0.0 && x.fract() == 0.0 && x <= u32::MAX as f64 {
        Some(x as usize)
    } else {
        None
    }
}

/// The values of a value file of rank `rank` (1 scalar, 3 vector), flattened
/// component-fastest: an optional header, an optional leading average, then
/// `N ( ... )` or `( ... )`. A count that is not the list's length is refused.
pub fn read_values(path: &Path, rank: usize) -> Result<Vec<f64>> {
    let text = tokenizer::slurp(path)?;
    let name = path.display().to_string();
    tokenizer::check_ascii_format(&text, &name)?;
    let mut ts = Tokenizer::new(&text, &name);
    ts.skip_header()?;

    let counted = |ts: &mut Tokenizer, n: usize| -> Result<Vec<f64>> {
        let v = read_list(ts, rank)?;
        if v.len() != n * rank {
            return ts.err(format!(
                "the list's count says {n} values and it holds {} (SPEC-LIT 120.11)",
                v.len() / rank
            ));
        }
        Ok(v)
    };

    let v = if rank == 3 {
        if ts.is_punct('(') && ts.is_punct_at('(', 1) {
            read_list(&mut ts, 3)?
        } else if ts.is_punct('(') {
            // A leading average vector, then the list.
            let mut avg = Vec::new();
            read_value(&mut ts, 3, &mut avg)?;
            if ts.is_punct('(') {
                read_list(&mut ts, 3)?
            } else {
                let n = ts.expect_label()?;
                let Some(n) = as_count(n as f64) else {
                    return ts.err("a negative list count (SPEC-LIT 120.11)");
                };
                counted(&mut ts, n)?
            }
        } else {
            let n = ts.expect_label()?;
            let Some(n) = as_count(n as f64) else {
                return ts.err("a negative list count (SPEC-LIT 120.11)");
            };
            counted(&mut ts, n)?
        }
    } else if ts.is_punct('(') {
        read_list(&mut ts, 1)?
    } else {
        let x = ts.expect_num()?;
        if ts.is_punct('(') {
            // `x ( ... )`: `x` is the count when it is the list's length,
            // and otherwise the file's average before a bare list - read
            // past either way, as a value list carries its own length.
            let _ = x;
            read_list(&mut ts, 1)?
        } else {
            // `average N ( ... )`.
            let n = ts.expect_num()?;
            let Some(n) = as_count(n) else {
                return ts.err("a list count must be a non-negative integer (SPEC-LIT 120.11)");
            };
            counted(&mut ts, n)?
        }
    };
    if !ts.done() {
        let what = match ts.peek()? {
            Some(Tok::Punct(c)) => c.to_string(),
            Some(t) => t.to_string(),
            None => String::new(),
        };
        return ts.err(format!("unexpected `{what}` after the value list (SPEC-LIT 120.11)"));
    }
    ts.check_scan_error()?;
    Ok(v)
}
