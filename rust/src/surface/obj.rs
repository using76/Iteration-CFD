// meteor-cfd - Copyright (c) 2026 주식회사 메테오시뮬레이션 (Meteo Simulation Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.

//! Wavefront OBJ surface reader; usemtl/g groups become patches
//! (material name = patch name, the docs/05 section 4.2 convention).
//!
//! Written from:
//!   `docs/05-io-redesign.md` section 4.2 - "one convention runs through
//!     every path: the MATERIAL name IS the patch name" - `usemtl` runs are
//!     the first-priority patch identity (what Blender actually writes),
//!     `g`/`o` groups are the fallback when a file carries no material at
//!     all;
//!   the Wavefront OBJ format itself, as documented in its long-public
//!     technical specification (Alias/Wavefront, 1992 - a de facto public
//!     file-format standard, not any program's source) - `v`/`f` syntax,
//!     1-based indices with the negative "relative to the current vertex
//!     count" form, and polygon (fan) faces;
//!   ofgpu `SPEC-LIT.md` §23.1-23.2 (patch identity, exact-bit welding,
//!     degenerate-triangle handling) via [`super::Surface::from_soup`], which
//!     this reader shares with [`super::stl::read_stl`].
//! No GPL-licensed source was consulted.
//!
//! # Patch identity
//!
//! * If the file contains AT LEAST ONE `usemtl` directive, every triangle's
//!   patch is the name of whichever material was active when its `f` line
//!   was read (a triangle read before the first `usemtl` gets `defaultFaces`).
//!   `g`/`o` are then ignored entirely for patch identity.
//! * Otherwise, if the file contains at least one `g` or `o` line, that name
//!   (whichever was most recently set) plays the same role, with the same
//!   `defaultFaces` fallback for anything before the first one.
//! * Otherwise every triangle is one patch named after the file stem, the
//!   same "no identity in the file" fallback [`super::stl::read_stl`] uses
//!   for an anonymous `solid`.
//!
//! # The `bc_` filter
//!
//! A Blender scene mixes materials assigned for CFD boundary identity with
//! materials assigned purely for shading. Once ANY material in the file is
//! named `bc_...`, that convention is assumed to be in force: only `bc_...`
//! materials keep their name as a patch, and every triangle whose material
//! is not `bc_...` (unset included) is folded into `defaultFaces`, so a
//! decorative material never fragments the boundary into spurious patches.
//! This filter applies to `usemtl`-derived names only - `g`/`o` fallback
//! grouping has no such convention to apply it to.
//!
//! Texture/normal indices (`v/vt/vn`) are read past and dropped: this reader
//! only ever wants the geometry.

use std::collections::HashMap;
use std::path::Path;

use super::{Surface, SoupTri};
use crate::error::{Error, IoContext, Result};
use crate::{Scalar, Vec3};

/// Read one OBJ file into a welded, patch-identified [`Surface`].
pub fn read_obj(path: impl AsRef<Path>) -> Result<Surface> {
    let path = path.as_ref();
    let text = std::fs::read_to_string(path).path(path)?;
    let stem = path
        .file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| "obj".to_string());
    parse_obj(&text, &stem, &path.display().to_string())
}

/// One triangle as read, with both possible patch sources still attached -
/// which one is used is decided only once the whole file has been read (a
/// `usemtl` seen on line 4000 still means "material mode" for line 10).
struct RawTri {
    verts: [Vec3; 3],
    material: Option<String>,
    group: Option<String>,
}

/// Parse from memory - public so tests need no fixture files on disk, the
/// same shape as [`crate::surface::stl::parse_stl`].
pub fn parse_obj(text: &str, stem: &str, origin: &str) -> Result<Surface> {
    let mut vertices: Vec<Vec3> = Vec::new();
    let mut tris: Vec<RawTri> = Vec::new();

    let mut material: Option<String> = None;
    let mut group: Option<String> = None;
    let mut seen_usemtl = false;

    let err = |line_no: usize, msg: String| -> Error {
        Error::Parse { path: format!("{origin}:{line_no}"), msg }
    };

    for (line_no, raw_line) in text.lines().enumerate() {
        let line_no = line_no + 1;
        let line = raw_line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let (kw, rest) = match line.split_once(char::is_whitespace) {
            Some((k, r)) => (k, r.trim()),
            None => (line, ""),
        };

        match kw {
            "v" => {
                let mut it = rest.split_whitespace();
                let mut c = [0.0 as Scalar; 3];
                for x in &mut c {
                    let tok = it
                        .next()
                        .ok_or_else(|| err(line_no, "'v' needs at least 3 coordinates".into()))?;
                    *x = tok
                        .parse::<f64>()
                        .map_err(|_| err(line_no, format!("'{tok}' is not a number")))?
                        as Scalar;
                }
                vertices.push(Vec3::new(c[0], c[1], c[2]));
            }
            "f" => {
                let mut idx: Vec<u32> = Vec::new();
                for tok in rest.split_whitespace() {
                    let vref = tok.split('/').next().unwrap_or("");
                    let n: i64 = vref
                        .parse()
                        .map_err(|_| err(line_no, format!("'{tok}' is not a valid face vertex")))?;
                    let zero_based = if n > 0 {
                        n - 1
                    } else if n < 0 {
                        vertices.len() as i64 + n
                    } else {
                        return Err(err(line_no, "vertex index 0 is not valid in OBJ (1-based)".into()));
                    };
                    if zero_based < 0 || zero_based as usize >= vertices.len() {
                        return Err(err(
                            line_no,
                            format!(
                                "face references vertex {n}, which is out of range \
                                 ({} vertices defined so far)",
                                vertices.len()
                            ),
                        ));
                    }
                    idx.push(zero_based as u32);
                }
                if idx.len() < 3 {
                    return Err(err(line_no, "a face needs at least 3 vertices".into()));
                }
                // Fan triangulation about the first vertex (§ module doc).
                for k in 1..idx.len() - 1 {
                    tris.push(RawTri {
                        verts: [
                            vertices[idx[0] as usize],
                            vertices[idx[k] as usize],
                            vertices[idx[k + 1] as usize],
                        ],
                        material: material.clone(),
                        group: group.clone(),
                    });
                }
            }
            "usemtl" => {
                if rest.is_empty() {
                    return Err(err(line_no, "'usemtl' needs a material name".into()));
                }
                seen_usemtl = true;
                material = Some(rest.to_string());
            }
            "g" | "o" => {
                group = if rest.is_empty() { None } else { Some(rest.to_string()) };
            }
            // Everything else (vt, vn, vp, mtllib, s, l, mg, comments handled
            // above) is irrelevant to a CFD boundary surface.
            _ => {}
        }
    }

    if tris.is_empty() {
        return Err(Error::Mesh(format!("{origin}: OBJ file has no faces")));
    }

    // ---- decide the patch-naming mode, once, over the whole file ----------
    let key_of: Box<dyn Fn(&RawTri) -> String> = if seen_usemtl {
        let mut used: std::collections::HashSet<&str> = std::collections::HashSet::new();
        for t in &tris {
            if let Some(m) = &t.material {
                used.insert(m.as_str());
            }
        }
        let has_bc = used.iter().any(|m| m.starts_with("bc_"));
        Box::new(move |t: &RawTri| match &t.material {
            Some(m) if !has_bc || m.starts_with("bc_") => m.clone(),
            _ => "defaultFaces".to_string(),
        })
    } else if tris.iter().any(|t| t.group.is_some()) {
        Box::new(|t: &RawTri| t.group.clone().unwrap_or_else(|| "defaultFaces".to_string()))
    } else {
        let stem = stem.to_string();
        Box::new(move |_: &RawTri| stem.clone())
    };

    let mut patch_names: Vec<String> = Vec::new();
    let mut patch_id: HashMap<String, u32> = HashMap::new();
    let mut soup: Vec<SoupTri> = Vec::with_capacity(tris.len());

    for t in &tris {
        let name = key_of(t);
        let id = *patch_id.entry(name.clone()).or_insert_with(|| {
            let id = patch_names.len() as u32;
            patch_names.push(name);
            id
        });
        soup.push((id, t.verts));
    }

    Surface::from_soup(soup, patch_names)
}

// ==========================================================================
//  Tests
// ==========================================================================

#[cfg(test)]
mod tests {
    use super::super::tests::{cube_points, CUBE_TRIS};
    use super::*;
    use std::fmt::Write as _;

    fn parsed(text: &str) -> Surface {
        match parse_obj(text, "cube", "<memory>") {
            Ok(s) => s,
            Err(e) => panic!("parse failed: {e}"),
        }
    }

    /// Write the cube's 8 vertices as `v` lines and, for each `(material,
    /// range)` pair, a `usemtl` switch followed by `f` lines for that
    /// triangle range - 1-based indices, the OBJ norm.
    fn cube_obj(mtls: &[(&str, std::ops::Range<usize>)]) -> String {
        let p = cube_points();
        let mut s = String::new();
        s += "# test cube\n";
        for v in p {
            let _ = writeln!(s, "v {} {} {}", v.x, v.y, v.z);
        }
        for (name, range) in mtls {
            let _ = writeln!(s, "usemtl {name}");
            for &[a, b, c] in &CUBE_TRIS[range.clone()] {
                let _ = writeln!(s, "f {} {} {}", a + 1, b + 1, c + 1);
            }
        }
        s
    }

    #[test]
    fn two_material_cube_yields_two_patches() {
        let text = cube_obj(&[("walls", 0..10), ("lid", 10..12)]);
        let s = parsed(&text);
        assert_eq!(s.patch_names, vec!["walls".to_string(), "lid".to_string()]);
        assert_eq!(s.tris.len(), 12);
        assert_eq!(s.points.len(), 8, "bit-exact weld across usemtl runs");
        assert_eq!(s.tri_patch[..10], [0; 10]);
        assert_eq!(s.tri_patch[10..], [1, 1]);
        assert!((s.patch_area[1] - 1.0).abs() < 1e-12, "lid is one unit face");
        assert_eq!(s.edge_defects(), (0, 0));
        assert!(s.require_closed().is_ok(), "closure check must pass on a closed cube");
    }

    #[test]
    fn bc_prefixed_materials_keep_their_name_others_become_default_faces() {
        // Two "bc_" materials plus one purely decorative material; the
        // decorative one must be folded into defaultFaces.
        let text = cube_obj(&[
            ("bc_inlet", 0..2),
            ("paint", 2..10),
            ("bc_outlet", 10..12),
        ]);
        let s = parsed(&text);
        let mut names = s.patch_names.clone();
        names.sort();
        assert_eq!(names, vec!["bc_inlet", "bc_outlet", "defaultFaces"]);

        let default_id = s.patch_names.iter().position(|n| n == "defaultFaces").unwrap() as u32;
        assert_eq!(&s.tri_patch[2..10], &vec![default_id; 8][..]);

        let inlet_id = s.patch_names.iter().position(|n| n == "bc_inlet").unwrap() as u32;
        assert_eq!(&s.tri_patch[0..2], &[inlet_id; 2]);
    }

    #[test]
    fn no_usemtl_falls_back_to_g_groups() {
        let p = cube_points();
        let mut s = String::new();
        for v in p {
            let _ = writeln!(s, "v {} {} {}", v.x, v.y, v.z);
        }
        let _ = writeln!(s, "g walls");
        for &[a, b, c] in &CUBE_TRIS[0..10] {
            let _ = writeln!(s, "f {} {} {}", a + 1, b + 1, c + 1);
        }
        let _ = writeln!(s, "g lid");
        for &[a, b, c] in &CUBE_TRIS[10..12] {
            let _ = writeln!(s, "f {} {} {}", a + 1, b + 1, c + 1);
        }

        let surf = parsed(&s);
        assert_eq!(surf.patch_names, vec!["walls".to_string(), "lid".to_string()]);
        assert_eq!(surf.tris.len(), 12);
    }

    #[test]
    fn no_grouping_at_all_names_the_single_patch_after_the_stem() {
        let text = cube_obj(&[]);
        // cube_obj with no mtls writes vertices but no faces via that path;
        // write faces directly with no usemtl/g/o at all.
        let mut s = String::new();
        for line in text.lines() {
            s.push_str(line);
            s.push('\n');
        }
        for &[a, b, c] in &CUBE_TRIS {
            let _ = writeln!(s, "f {} {} {}", a + 1, b + 1, c + 1);
        }
        let surf = parsed(&s);
        assert_eq!(surf.patch_names, vec!["cube".to_string()]);
        assert_eq!(surf.tris.len(), 12);
    }

    #[test]
    fn negative_indices_reference_the_most_recently_written_vertices() {
        // Eight `v` lines, then every face written with negative indices -
        // -8 is the first vertex, -1 the last, exactly like `1`/`8` would be.
        let p = cube_points();
        let mut s = String::new();
        for v in p {
            let _ = writeln!(s, "v {} {} {}", v.x, v.y, v.z);
        }
        s += "usemtl cube\n";
        for &[a, b, c] in &CUBE_TRIS {
            let neg = |i: usize| i as i64 - 8; // 0 -> -8, 7 -> -1
            let _ = writeln!(s, "f {} {} {}", neg(a), neg(b), neg(c));
        }

        let surf = parsed(&s);
        assert_eq!(surf.tris.len(), 12);
        assert_eq!(surf.points.len(), 8);
        assert_eq!(surf.edge_defects(), (0, 0));
    }

    #[test]
    fn a_quad_face_is_fan_triangulated() {
        // One planar quad face in the z = 0 plane of the cube, CCW from
        // above so its recomputed normal is +z: two triangles, same area.
        let p = cube_points();
        let mut s = String::new();
        for v in [p[0], p[1], p[2], p[3]] {
            let _ = writeln!(s, "v {} {} {}", v.x, v.y, v.z);
        }
        s += "f 1 2 3 4\n";

        let surf = parsed(&s);
        assert_eq!(surf.tris.len(), 2, "a quad fans into 2 triangles");
        assert_eq!(surf.points.len(), 4);
        assert!((surf.patch_area[0] - 1.0).abs() < 1e-12, "unit square area");
        for n in &surf.normals {
            assert_eq!(*n, Vec3::new(0.0, 0.0, 1.0));
        }
    }

    #[test]
    fn closure_check_passes_on_a_closed_cube_from_obj() {
        let text = cube_obj(&[("cube", 0..12)]);
        let surf = parsed(&text);
        assert!(surf.require_closed().is_ok());
    }
}
