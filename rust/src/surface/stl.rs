// meteor-cfd - Copyright (c) 2026 주식회사 메테오시뮬레이션 (Meteo Simulation Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.

//! STL reader: binary and ASCII, into a welded [`Surface`].
//!
//! Written from:
//!   ofgpu `SPEC-LIT.md` §23.1-23.2 (format, detection, patch identity,
//!     validation); the STL layout itself is the de facto public
//!     specification (3D Systems, 1987);
//!   Aftosmis, Berger & Melton, *AIAA J.* 36(6) (1998) 952;
//!   Barill, Dickson, Schmidt, Levin & Jacobson, *ACM TOG* 37(4) (2018).
//! No GPL-licensed source was consulted.
//!
//! Detection (§23.1): a file is ASCII only if it starts with "solid" AND
//! parses as ASCII. Binary exporters routinely write "solid ..." into the
//! 80-byte comment header, so an ASCII parse failure falls back to binary
//! rather than erroring - the two conditions together are unambiguous.
//!
//! Patch identity (§23.1): one patch per `solid` name in an ASCII file; a
//! binary file has no name, so the FILE STEM becomes the patch name.
//! Stored facet normals are read past and ignored - the winding is the
//! truth and normals are recomputed in [`Surface::from_soup`].

use std::path::Path;

use super::{Surface, SoupTri};
use crate::error::{Error, IoContext, Result};
use crate::{Scalar, Vec3};

/// Fixed offsets of the binary layout: 80-byte header, u32 count, then
/// 50 bytes per triangle (12 normal + 36 vertices + 2 attribute).
const BIN_HEADER: usize = 80;
const BIN_TRI_BYTES: usize = 50;

/// Read one STL file - binary or ASCII, detected per §23.1 - into a welded,
/// validated-for-degeneracy [`Surface`]. Closure is NOT checked here: the
/// caller merges multiple files first and then calls
/// [`Surface::require_closed`] on the result.
pub fn read_stl(path: impl AsRef<Path>) -> Result<Surface> {
    let path = path.as_ref();
    let bytes = std::fs::read(path).path(path)?;
    let stem = path
        .file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| "stl".to_string());
    parse_stl(&bytes, &stem, &path.display().to_string())
}

/// Parse STL from memory. `stem` names the patch for binary files (and for
/// anonymous `solid`s); `origin` is what error messages call the source.
///
/// Public so tests - and anything that already holds the bytes - need no
/// fixture files on disk.
pub fn parse_stl(bytes: &[u8], stem: &str, origin: &str) -> Result<Surface> {
    let starts_solid = bytes.len() >= 5 && bytes[..5].eq_ignore_ascii_case(b"solid");

    if starts_solid {
        // ASCII candidate. Binary garbage is not valid UTF-8 and does not
        // follow the facet grammar, so a real binary file drops through.
        let text = String::from_utf8_lossy(bytes);
        if let Ok((soup, names)) = parse_ascii(&text, stem) {
            return Surface::from_soup(soup, names);
        }
    }

    let (soup, names) = parse_binary(bytes, stem, origin)?;
    Surface::from_soup(soup, names)
}

// ==========================================================================
//  Binary
// ==========================================================================

/// Little-endian f32 at `o`; the caller has already validated the length.
#[inline]
fn f32_at(b: &[u8], o: usize) -> Scalar {
    let mut a = [0u8; 4];
    a.copy_from_slice(&b[o..o + 4]);
    f32::from_le_bytes(a) as Scalar
}

fn parse_binary(bytes: &[u8], stem: &str, origin: &str) -> Result<(Vec<SoupTri>, Vec<String>)> {
    if bytes.len() < BIN_HEADER + 4 {
        return Err(Error::Parse {
            path: origin.to_string(),
            msg: format!(
                "binary STL needs at least {} bytes (80-byte header + \
                 triangle count), found {}",
                BIN_HEADER + 4,
                bytes.len()
            ),
        });
    }

    let mut cnt = [0u8; 4];
    cnt.copy_from_slice(&bytes[BIN_HEADER..BIN_HEADER + 4]);
    let n = u32::from_le_bytes(cnt);

    // The format has no trailer: the length is exactly determined by the
    // count, so any mismatch means truncation or a lying header - both
    // corrupt geometry, so refuse rather than salvage (§23.1).
    let expected = (BIN_HEADER + 4) as u64 + BIN_TRI_BYTES as u64 * n as u64;
    if bytes.len() as u64 != expected {
        return Err(Error::Parse {
            path: origin.to_string(),
            msg: format!(
                "binary STL declares {n} triangles, which requires exactly \
                 {expected} bytes (84 + 50*{n}), but the file is {} bytes",
                bytes.len()
            ),
        });
    }

    let mut soup = Vec::with_capacity(n as usize);
    for t in 0..n as usize {
        // Skip the 12-byte stored normal - recomputed from winding (§23.1).
        let v = BIN_HEADER + 4 + BIN_TRI_BYTES * t + 12;
        soup.push((
            0u32,
            [
                Vec3::new(f32_at(bytes, v), f32_at(bytes, v + 4), f32_at(bytes, v + 8)),
                Vec3::new(f32_at(bytes, v + 12), f32_at(bytes, v + 16), f32_at(bytes, v + 20)),
                Vec3::new(f32_at(bytes, v + 24), f32_at(bytes, v + 28), f32_at(bytes, v + 32)),
            ],
        ));
        // The trailing u16 attribute is ignored, like the normal.
    }

    Ok((soup, vec![stem.to_string()]))
}

// ==========================================================================
//  ASCII
// ==========================================================================

/// Strict-enough ASCII parser: the exact keyword skeleton is required (so
/// that "parses as ASCII" is a meaningful test for the §23.1 detection),
/// keywords are matched case-insensitively, and the stored normal values
/// are validated as numbers but never used.
fn parse_ascii(text: &str, stem: &str) -> std::result::Result<(Vec<SoupTri>, Vec<String>), String> {
    let mut lines = text
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty())
        .peekable();

    let mut soup: Vec<SoupTri> = Vec::new();
    let mut names: Vec<String> = Vec::new();

    while let Some(line) = lines.next() {
        let rest = strip_keyword(line, "solid").ok_or("expected 'solid'")?;
        // An anonymous solid inherits the file stem, mirroring binary.
        let name = if rest.is_empty() { stem } else { rest };
        super::push_unique_name(&mut names, name);
        let patch = (names.len() - 1) as u32;

        loop {
            let line = lines.next().ok_or("missing 'endsolid'")?;
            if strip_keyword(line, "endsolid").is_some() {
                break;
            }

            let normal = strip_keyword(line, "facet").ok_or("expected 'facet' or 'endsolid'")?;
            if let Some(nums) = strip_keyword(normal, "normal") {
                // Present in every conforming file; parsed only to reject
                // non-STL text, the values themselves are ignored.
                parse_three(nums)?;
            }
            let outer = lines.next().ok_or("missing 'outer loop'")?;
            let loop_kw = strip_keyword(outer, "outer").ok_or("expected 'outer loop'")?;
            if !loop_kw.eq_ignore_ascii_case("loop") {
                return Err("expected 'outer loop'".into());
            }

            let mut v = [Vec3::ZERO; 3];
            for vert in &mut v {
                let line = lines.next().ok_or("missing 'vertex'")?;
                let nums = strip_keyword(line, "vertex").ok_or("expected 'vertex'")?;
                *vert = parse_three(nums)?;
            }

            let end = lines.next().ok_or("missing 'endloop'")?;
            if strip_keyword(end, "endloop").is_none() {
                return Err("expected 'endloop' (only triangles are legal STL)".into());
            }
            let end = lines.next().ok_or("missing 'endfacet'")?;
            if strip_keyword(end, "endfacet").is_none() {
                return Err("expected 'endfacet'".into());
            }

            soup.push((patch, v));
        }
    }

    if names.is_empty() {
        return Err("no 'solid' block found".into());
    }
    Ok((soup, names))
}

/// If `line` starts with `kw` as a whole word (case-insensitive), return
/// the trimmed remainder; solid names may contain spaces, so the remainder
/// is the rest of the line, not one token.
fn strip_keyword<'a>(line: &'a str, kw: &str) -> Option<&'a str> {
    if line.len() < kw.len() || !line[..kw.len()].eq_ignore_ascii_case(kw) {
        return None;
    }
    let rest = &line[kw.len()..];
    if !rest.is_empty() && !rest.starts_with(char::is_whitespace) {
        return None; // "solidify" is not "solid"
    }
    Some(rest.trim())
}

/// Exactly three whitespace-separated numbers.
fn parse_three(s: &str) -> std::result::Result<Vec3, String> {
    let mut it = s.split_whitespace();
    let mut c = [0.0 as Scalar; 3];
    for x in &mut c {
        let tok = it.next().ok_or("expected three numbers")?;
        *x = tok
            .parse::<f64>()
            .map_err(|_| format!("'{tok}' is not a number"))? as Scalar;
    }
    if it.next().is_some() {
        return Err("expected exactly three numbers".into());
    }
    Ok(Vec3::new(c[0], c[1], c[2]))
}

// ==========================================================================
//  Tests - all on in-memory buffers, no fixture files
// ==========================================================================

#[cfg(test)]
mod tests {
    use super::super::tests::{cube_points, CUBE_TRIS};
    use super::*;
    use std::fmt::Write as _;

    /// The unit cube as binary STL bytes, header filled with `header`
    /// (padded/truncated to 80 bytes), stored normals left at zero -
    /// which also proves they are ignored.
    fn binary_cube(header: &[u8]) -> Vec<u8> {
        let p = cube_points();
        let mut b = vec![0u8; 80];
        b[..header.len().min(80)].copy_from_slice(&header[..header.len().min(80)]);
        b.extend_from_slice(&(CUBE_TRIS.len() as u32).to_le_bytes());
        for &[i, j, k] in &CUBE_TRIS {
            b.extend_from_slice(&[0u8; 12]); // bogus stored normal
            for v in [p[i], p[j], p[k]] {
                b.extend_from_slice(&(v.x as f32).to_le_bytes());
                b.extend_from_slice(&(v.y as f32).to_le_bytes());
                b.extend_from_slice(&(v.z as f32).to_le_bytes());
            }
            b.extend_from_slice(&[0u8; 2]); // attribute byte count
        }
        b
    }

    /// The cube as ASCII, split into named solids by triangle ranges.
    fn ascii_cube(solids: &[(&str, std::ops::Range<usize>)]) -> String {
        let p = cube_points();
        let mut s = String::new();
        for (name, range) in solids {
            let _ = writeln!(s, "solid {name}");
            for &[i, j, k] in &CUBE_TRIS[range.clone()] {
                let _ = writeln!(s, "  facet normal 0 0 0");
                let _ = writeln!(s, "    outer loop");
                for v in [p[i], p[j], p[k]] {
                    let _ = writeln!(s, "      vertex {} {} {}", v.x, v.y, v.z);
                }
                let _ = writeln!(s, "    endloop");
                let _ = writeln!(s, "  endfacet");
            }
            let _ = writeln!(s, "endsolid {name}");
        }
        s
    }

    fn parsed(bytes: &[u8]) -> Surface {
        match parse_stl(bytes, "cube", "<memory>") {
            Ok(s) => s,
            Err(e) => panic!("parse failed: {e}"),
        }
    }

    #[test]
    fn binary_cube_reads_12_tris_and_welds_8_points() {
        let s = parsed(&binary_cube(b"any old comment"));
        assert_eq!(s.tris.len(), 12);
        assert_eq!(s.points.len(), 8, "bit-exact weld across repeated vertices");
        assert_eq!(s.patch_names, vec!["cube".to_string()], "file stem names the patch");
        assert!((s.patch_area[0] - 6.0).abs() < 1e-12);
        assert_eq!(s.edge_defects(), (0, 0));
        // Stored normals were zero; these are recomputed from winding.
        assert_eq!(s.normals[6], Vec3::new(1.0, 0.0, 0.0));
    }

    #[test]
    fn ascii_two_solids_yield_two_patches() {
        let text = ascii_cube(&[("walls", 0..10), ("lid", 10..12)]);
        let s = parsed(text.as_bytes());
        assert_eq!(s.patch_names, vec!["walls".to_string(), "lid".to_string()]);
        assert_eq!(s.tris.len(), 12);
        assert_eq!(s.points.len(), 8, "weld crosses solid boundaries");
        assert_eq!(s.tri_patch[..10], [0; 10]);
        assert_eq!(s.tri_patch[10..], [1, 1]);
        assert!((s.patch_area[1] - 1.0).abs() < 1e-12, "lid is one unit face");
    }

    #[test]
    fn binary_file_starting_with_solid_still_parses_as_binary() {
        // The §23.1 ambiguity: binary exporters write "solid ..." into the
        // comment header. Starts with "solid", does NOT parse as ASCII.
        let s = parsed(&binary_cube(b"solid exported by some tool"));
        assert_eq!(s.tris.len(), 12);
        assert_eq!(s.points.len(), 8);
        assert_eq!(s.patch_names, vec!["cube".to_string()]);
    }

    #[test]
    fn binary_length_mismatch_is_rejected_with_the_arithmetic() {
        let mut b = binary_cube(b"ok");
        b.pop(); // truncate one byte
        let e = match parse_stl(&b, "cube", "<memory>") {
            Err(e) => e.to_string(),
            Ok(_) => panic!("truncated binary STL was accepted"),
        };
        assert!(e.contains("12 triangles"), "{e}");
        assert!(e.contains("684"), "expected byte count named: {e}");
        assert!(e.contains("683"), "actual byte count named: {e}");
    }

    #[test]
    fn ascii_degenerate_triangle_is_dropped_with_count() {
        let mut text = ascii_cube(&[("cube", 0..12)]);
        // Splice a zero-area facet (repeated vertex) before "endsolid".
        let degen = "facet normal 0 0 0\nouter loop\n\
                     vertex 0 0 0\nvertex 0 0 0\nvertex 1 1 1\n\
                     endloop\nendfacet\nendsolid cube\n";
        text = text.replace("endsolid cube", degen);
        let s = parsed(text.as_bytes());
        assert_eq!(s.tris.len(), 12);
        assert_eq!(s.degenerate_dropped, 1);
        assert_eq!(s.points.len(), 8);
    }

    #[test]
    fn ascii_open_box_is_refused_through_the_contract() {
        crate::io::contract::set_permissive(false);
        let text = ascii_cube(&[("box", 0..10)]); // no lid
        let s = parsed(text.as_bytes());
        let e = match s.require_closed() {
            Err(e) => e.to_string(),
            Ok(()) => panic!("open surface was accepted"),
        };
        assert!(e.contains("4 open edge"), "{e}");
    }
}
