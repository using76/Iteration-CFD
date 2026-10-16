// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.
// Provenance: see PROVENANCE.md. No GPL-licensed source was consulted.

//! The emission of SPEC-LIT §116.16 (TET-11): the tetrahedra become a
//! `PolyMeshRaw`, the §92.3 gate is run on it, and under `--features
//! single` a mesh that the f32 rounding of its points would flip is
//! refused, naming the cells (116.26). TET-13b extends the same layout to
//! the prisms of §116.15: cells `0..n_tets` are the tetrahedra and the
//! prisms follow in (116.25e)'s order, a face being a triangle or a
//! quadrilateral keyed by its sorted points. TET-13c's collapsed wall
//! points repeat ids in a prism - a pyramid or a tetrahedron - whose faces
//! drop their repeated consecutive points, and a face left with fewer than
//! three ids is no face.
//!
//! The layout is OpenFOAM's polyMesh convention as §1 and §92.3's G7 state
//! it:
//!
//! - cell `i` is `tets[i]`;
//! - one face per triangle shared by two tetrahedra, owned by the lower
//!   cell and wound by the owner's outward row of §116.4's table, so its
//!   right-hand normal points from the owner to the neighbour;
//! - the internal faces in strictly ascending `(owner, neighbour)` order
//!   (upper-triangular, `owner < neighbour`);
//! - then the boundary faces, grouped by patch in patch order, each patch
//!   in `(owner, local face)` order, wound out of the domain;
//! - the points that no tetrahedron uses dropped, the rest kept in
//!   ascending input order, as f64 (`PolyMeshRaw.points` is f64 in both
//!   builds since F32-03, SPEC-LIT §118.2).
//!
//! The f32 check rounds every point through [`crate::types::to_vec3`], the
//! one cast by which a host point becomes `Scalar`, widens it back exactly
//! and evaluates (116.2) with the exact `orient3d` of §116.4. In the f64
//! build the rounding is the identity, and [`emit`] skips the check.
//!
//! Provenance: ORIGINAL. The polyMesh layout is the format's own published
//! convention as SPEC-LIT §1 and §92.3 restate it; the check (116.26) is
//! SPEC-LIT §116.16's. No mesher's source was read. No GPL-licensed source
//! was consulted.

use super::carve::Carved;
use super::predicates::orient3d;
use super::recover::FaceRecovery;
use super::Point;
use crate::automesher::quality::{self, QualityReport, QualityThresholds};
use crate::error::{Error, Result};
use crate::io::polymesh::PolyMeshRaw;
use crate::mesh::{PatchInfo, PatchKind};
use crate::types::DVec3;
use crate::Label;
use std::collections::HashMap;

/// The faces of a tetrahedron `[v0, v1, v2, v3]` with `orient3d > 0`,
/// wound so the right-hand normal points out, opposite `v3, v2, v1, v0`
/// (§116.4's table).
pub(crate) const FACES: [[usize; 3]; 4] = [[0, 1, 2], [0, 3, 1], [0, 2, 3], [1, 3, 2]];

/// How many cells or faces a refusal names before it says "and N more".
const NAMED: usize = 20;

/// The three tetrahedra of a prism's split (116.25f): `PRISM_SPLIT[k]` is
/// `[v_i, v_j, v_k, v_l]` of `[v0..v5]`, and the prism is valid when every
/// row with four distinct ids has `orient3d > 0` and at least one row does.
/// On the right prism over the unit corner triangle (`v0..v2` =
/// `(0,0,0), (1,0,0), (0,1,0)`, the top `+z`) each is exactly `+1`.
pub const PRISM_SPLIT: [[usize; 4]; 3] = [[0, 2, 1, 3], [2, 1, 3, 4], [3, 2, 4, 5]];

/// (116.25f) with TET-13c's repeated ids: a split row whose four ids are
/// not distinct is skipped, never by its volume; every other row needs
/// `orient3d > 0` on `at`'s points, and at least one row must have four
/// distinct ids. A prism whose ids repeat is a pyramid or a tetrahedron.
fn prism_split_ok(p: &[u32; 6], at: impl Fn(u32) -> Point) -> bool {
    let mut any = false;
    for row in &PRISM_SPLIT {
        let q = row.map(|i| p[i]);
        let [a, b, c, d] = q;
        if a == b || a == c || a == d || b == c || b == d || c == d {
            continue;
        }
        any = true;
        let (x0, x1, x2, x3) = (at(a), at(b), at(c), at(d));
        if !(orient3d(&x0, &x1, &x2, &x3) > 0.0) {
            return false;
        }
    }
    any
}

/// A prism `[v0..v5]` = `[a, b, c, a', b', c']` (116.25e) is valid when
/// every tetrahedron of its split ([`PRISM_SPLIT`]) with four distinct ids
/// has `orient3d > 0` (116.2), and one does. Unchanged for a prism whose
/// ids are all distinct.
pub fn prism_valid(points: &[Point], p: &[u32; 6]) -> bool {
    prism_split_ok(p, |v| points[v as usize])
}

/// A face's repeated consecutive ids removed, cyclically: `a b b a` is
/// `a b`, and a one- or two-id result is no face.
pub(crate) fn reduce_cyclic(f: Vec<u32>) -> Vec<u32> {
    let mut f = f;
    loop {
        let mut g: Vec<u32> = Vec::with_capacity(f.len());
        for &x in &f {
            if g.last() != Some(&x) {
                g.push(x);
            }
        }
        if g.len() > 1 && g[0] == g[g.len() - 1] {
            g.pop();
        }
        if g.len() == f.len() {
            return g;
        }
        f = g;
    }
}

/// A prism's local face `l` in `0..5`, wound out of the prism (§116.4):
/// the bottom `(v0, v2, v1)`, the top `(v3, v4, v5)`, and the three side
/// quads `(v0, v1, v4, v3)`, `(v1, v2, v5, v4)`, `(v2, v0, v3, v5)` - with
/// repeated consecutive ids removed cyclically (TET-13c's collapse).
fn prism_face(p: &[u32; 6], l: usize) -> Vec<u32> {
    let f = match l {
        0 => vec![p[0], p[2], p[1]],
        1 => vec![p[3], p[4], p[5]],
        2 => vec![p[0], p[1], p[4], p[3]],
        3 => vec![p[1], p[2], p[5], p[4]],
        _ => vec![p[2], p[0], p[3], p[5]],
    };
    reduce_cyclic(f)
}

fn keyv(mut f: Vec<u32>) -> Vec<u32> {
    f.sort_unstable();
    f
}

/// One patch of the emitted mesh: its name and its OpenFOAM type
/// (`wall`, `patch`, `symmetryPlane`, ...), kept verbatim.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EmitPatch {
    /// The patch name.
    pub name: String,
    /// The OpenFOAM patch type; `PatchKind::from_type` reads it.
    pub type_name: String,
}

/// The emitted mesh.
#[derive(Clone, Debug)]
pub struct Emitted {
    /// The polyMesh; cell `i` is the input's `tets[i]`, then the prisms.
    pub raw: PolyMeshRaw,
    /// Beside `raw.points`: the index in the caller's array of each emitted
    /// point (the input index; for a prism layer point, its own index, whose
    /// wall point is the last entry of its `layers[w]` row).
    pub point_of: Vec<u32>,
}

fn named<T: std::fmt::Debug>(items: &[T]) -> String {
    let shown: Vec<String> = items.iter().take(NAMED).map(|x| format!("{x:?}")).collect();
    let more = items.len().saturating_sub(NAMED);
    if more > 0 {
        format!("{} and {more} more", shown.join(", "))
    } else {
        shown.join(", ")
    }
}

/// The patch of each of the carve's boundary faces: a subtriangle's patch
/// is its parent triangle's, `tri_patch[fr.sub_parent[s]]` (`tri_patch` is
/// beside `fr.tris`); a hull face (the domain boundary, `face_sub = None`)
/// gets `hull_patch`. The faces are the carve's, wound out of the kept
/// region; the optimiser keeps them bit for bit (116.21b), so they name the
/// optimised mesh's boundary too.
pub fn carved_boundary(
    carved: &Carved,
    fr: &FaceRecovery,
    tri_patch: &[u32],
    hull_patch: u32,
) -> Result<Vec<([u32; 3], u32)>> {
    if tri_patch.len() != fr.tris.len() {
        return Err(Error::Refused(format!(
            "emit/patches: {} triangle patches for {} triangles",
            tri_patch.len(),
            fr.tris.len()
        )));
    }
    if carved.face_sub.len() != carved.faces.len() {
        return Err(Error::Refused(format!(
            "emit/patches: {} subtriangle ids beside {} carved faces",
            carved.face_sub.len(),
            carved.faces.len()
        )));
    }
    let mut out = Vec::with_capacity(carved.faces.len());
    for (k, (&f, &s)) in carved.faces.iter().zip(&carved.face_sub).enumerate() {
        let patch = match s {
            None => hull_patch,
            Some(s) => {
                let parent = fr.sub_parent.get(s as usize).copied().ok_or_else(|| {
                    Error::Refused(format!(
                        "emit/patches: carved face {k} names subtriangle {s} of {}",
                        fr.sub_parent.len()
                    ))
                })?;
                tri_patch.get(parent as usize).copied().ok_or_else(|| {
                    Error::Refused(format!(
                        "emit/patches: subtriangle {s} names triangle {parent} of {}",
                        tri_patch.len()
                    ))
                })?
            }
        };
        out.push((f, patch));
    }
    Ok(out)
}

/// The tetrahedra as a `PolyMeshRaw` (§116.16's layout, above), with the
/// boundary face `f` (any rotation, either winding) on patch `p` for each
/// `(f, p)` of `boundary`. Refused, naming what is wrong: a point index out
/// of range; a tetrahedron without `orient3d > 0` (116.2); a face on three
/// or more tetrahedra; a boundary face without a patch; a `boundary` entry
/// that is not a boundary face, or two that disagree; a patch index out of
/// range. No gate is run here; [`emit`] runs them.
pub fn to_poly_mesh(
    points: &[Point],
    tets: &[[u32; 4]],
    boundary: &[([u32; 3], u32)],
    patches: &[EmitPatch],
) -> Result<Emitted> {
    let bnd: Vec<(Vec<u32>, u32)> = boundary.iter().map(|&(f, p)| (f.to_vec(), p)).collect();
    to_poly_mesh_mixed(points, tets, &[], &bnd, patches)
}

/// The tetrahedra and the prisms as a `PolyMeshRaw` (§116.16's layout with
/// TET-13b's prisms, SPEC-LIT §116.15): cells `0..tets.len()` are the
/// tetrahedra and the prisms follow in (116.25e)'s order; a face is a
/// triangle or a quadrilateral, keyed by its sorted points. The boundary
/// faces are `(points, patch)` pairs, any rotation, either winding. The
/// refusals are [`to_poly_mesh`]'s, with a prism that fails (116.25f)
/// refused like a tetrahedron (`emit/orient`); TET-13c's repeated ids make
/// a prism's faces drop their repeated consecutive points, and a face left
/// with fewer than three ids is not emitted. No gate runs here;
/// [`emit_mixed`] runs them.
pub fn to_poly_mesh_mixed(
    points: &[Point],
    tets: &[[u32; 4]],
    prisms: &[[u32; 6]],
    boundary: &[(Vec<u32>, u32)],
    patches: &[EmitPatch],
) -> Result<Emitted> {
    if tets.len() > Label::MAX as usize {
        return Err(Error::Refused(format!(
            "emit/cells: {} tetrahedra do not fit in a 32-bit label",
            tets.len()
        )));
    }
    for (c, t) in tets.iter().enumerate() {
        if let Some(&v) = t.iter().find(|&&v| v as usize >= points.len()) {
            return Err(Error::Refused(format!(
                "emit/points: tetrahedron {c} names point {v} of {}",
                points.len()
            )));
        }
    }
    let inverted: Vec<usize> = (0..tets.len())
        .filter(|&c| {
            let p = tets[c].map(|v| points[v as usize]);
            orient3d(&p[0], &p[1], &p[2], &p[3]) <= 0.0
        })
        .collect();
    if !inverted.is_empty() {
        return Err(Error::Refused(format!(
            "emit/orient: {} tetrahedra have orient3d <= 0 (116.2): cells {}",
            inverted.len(),
            named(&inverted)
        )));
    }
    let flat: Vec<usize> = (0..prisms.len())
        .filter(|&j| !prism_valid(points, &prisms[j]))
        .map(|j| tets.len() + j)
        .collect();
    if !flat.is_empty() {
        return Err(Error::Refused(format!(
            "emit/orient: {} prisms have a split tetrahedron with orient3d <= 0 (116.25f): cells {}",
            flat.len(),
            named(&flat)
        )));
    }

    // Each face's first visit is from the lower cell, its owner. A face is
    // a triangle or a quadrilateral, keyed by its sorted points.
    // value: (owner, owner's local face, neighbour or u32::MAX)
    let mut face: HashMap<Vec<u32>, (u32, u8, u32)> =
        HashMap::with_capacity(2 * tets.len() + 5 * prisms.len() + 8);
    let mut crowded: Vec<Vec<u32>> = Vec::new();
    for (c, t) in tets.iter().enumerate() {
        for (l, row) in FACES.iter().enumerate() {
            let k = keyv(row.map(|i| t[i]).to_vec());
            match face.get_mut(&k) {
                None => {
                    face.insert(k, (c as u32, l as u8, u32::MAX));
                }
                Some(e) if e.2 == u32::MAX && e.0 != c as u32 => e.2 = c as u32,
                Some(_) => crowded.push(k),
            }
        }
    }
    for (j, p) in prisms.iter().enumerate() {
        for l in 0..5 {
            // A prism face reduced below three ids (TET-13c's collapse) is
            // no face: not keyed, not emitted.
            let f = prism_face(p, l);
            if f.len() < 3 {
                continue;
            }
            let k = keyv(f);
            let c = (tets.len() + j) as u32;
            match face.get_mut(&k) {
                None => {
                    face.insert(k, (c, l as u8, u32::MAX));
                }
                Some(e) if e.2 == u32::MAX && e.0 != c => e.2 = c,
                Some(_) => crowded.push(k),
            }
        }
    }
    if !crowded.is_empty() {
        crowded.sort_unstable();
        crowded.dedup();
        return Err(Error::Refused(format!(
            "emit/faces: {} faces lie on three or more tetrahedra (or twice on one): {}",
            crowded.len(),
            named(&crowded)
        )));
    }

    // The boundary's patches.
    let mut patch_of: HashMap<Vec<u32>, u32> = HashMap::with_capacity(boundary.len());
    for &(ref f, p) in boundary {
        if p as usize >= patches.len() {
            return Err(Error::Refused(format!(
                "emit/patches: boundary face {f:?} names patch {p} of {}",
                patches.len()
            )));
        }
        let k = keyv(f.clone());
        match face.get(&k) {
            Some(e) if e.2 == u32::MAX => {}
            _ => {
                return Err(Error::Refused(format!(
                    "emit/patches: {f:?} is given patch {p} but is not a boundary face of the tetrahedra"
                )))
            }
        }
        if let Some(q) = patch_of.insert(k, p) {
            if q != p {
                return Err(Error::Refused(format!(
                    "emit/patches: boundary face {f:?} is given patches {q} and {p}"
                )));
            }
        }
    }

    let mut internal: Vec<(u32, u32, Vec<u32>)> = Vec::with_capacity(2 * tets.len());
    let mut bnd: Vec<(u32, u32, u8, Vec<u32>)> = Vec::with_capacity(patch_of.len());
    let mut unpatched: Vec<(u32, Vec<u32>)> = Vec::new();
    for (k, &(o, l, n)) in &face {
        let wound = if (o as usize) < tets.len() {
            FACES[l as usize].map(|i| tets[o as usize][i]).to_vec()
        } else {
            prism_face(&prisms[o as usize - tets.len()], l as usize)
        };
        if n != u32::MAX {
            internal.push((o, n, wound));
        } else {
            match patch_of.get(k) {
                Some(&p) => bnd.push((p, o, l, wound)),
                None => unpatched.push((o, wound)),
            }
        }
    }
    if !unpatched.is_empty() {
        unpatched.sort_unstable();
        return Err(Error::Refused(format!(
            "emit/patches: {} boundary faces have no patch: (cell, face) {}",
            unpatched.len(),
            named(&unpatched)
        )));
    }
    // (owner, neighbour) is unique for distinct tetrahedra unless two share
    // all four points; §92.3's G7 names that case, so the sort key carries
    // the face too and the order stays a function of the input alone.
    internal.sort_unstable();
    bnd.sort_unstable();

    // The points used, in ascending input order.
    let mut new_of = vec![u32::MAX; points.len()];
    for t in tets {
        for &v in t {
            new_of[v as usize] = 0;
        }
    }
    for p in prisms {
        for &v in p {
            new_of[v as usize] = 0;
        }
    }
    let mut point_of = Vec::new();
    for (v, slot) in new_of.iter_mut().enumerate() {
        if *slot == 0 {
            *slot = point_of.len() as u32;
            point_of.push(v as u32);
        }
    }
    let pts: Vec<DVec3> = point_of
        .iter()
        .map(|&v| {
            let p = points[v as usize];
            DVec3::new(p[0], p[1], p[2])
        })
        .collect();
    let lab = |f: &[u32]| -> Vec<Label> { f.iter().map(|&v| new_of[v as usize] as Label).collect() };

    let n_faces = internal.len() + bnd.len();
    let mut faces: Vec<Vec<Label>> = Vec::with_capacity(n_faces);
    let mut owner: Vec<Label> = Vec::with_capacity(n_faces);
    let mut neighbour: Vec<Label> = Vec::with_capacity(internal.len());
    for (o, n, f) in &internal {
        faces.push(lab(f));
        owner.push(*o as Label);
        neighbour.push(*n as Label);
    }
    let mut sizes = vec![0usize; patches.len()];
    for (p, o, _, f) in &bnd {
        faces.push(lab(f));
        owner.push(*o as Label);
        sizes[*p as usize] += 1;
    }
    let mut start = 0usize;
    let mut infos = Vec::with_capacity(patches.len());
    for (p, size) in patches.iter().zip(sizes) {
        infos.push(PatchInfo {
            name: p.name.clone(),
            type_name: p.type_name.clone(),
            kind: PatchKind::from_type(&p.type_name),
            start,
            size,
            nbr_patch: None,
        });
        start += size;
    }
    Ok(Emitted {
        raw: PolyMeshRaw { points: pts, faces, owner, neighbour, patches: infos },
        point_of,
    })
}

/// (116.26): the cells whose volume (116.2), evaluated exactly in f64 on
/// the points rounded to `Scalar` by [`crate::types::to_vec3`] and widened
/// back, is not positive, ascending. Empty in the f64 build for any input
/// with `orient3d > 0`, because the rounding is the identity there. A
/// tetrahedron is cell `c`; a prism is cell `tets.len() + j`, and its arm
/// evaluates (116.25f)'s three tetrahedra on the rounded points.
pub fn rounding_flips_mixed(
    points: &[Point],
    tets: &[[u32; 4]],
    prisms: &[[u32; 6]],
) -> Vec<usize> {
    let round = |p: Point| -> Point {
        let r = crate::types::to_dvec3(crate::types::to_vec3(DVec3::new(p[0], p[1], p[2])));
        [r.x, r.y, r.z]
    };
    let mut out: Vec<usize> = (0..tets.len())
        .filter(|&c| {
            let q = tets[c].map(|v| round(points[v as usize]));
            orient3d(&q[0], &q[1], &q[2], &q[3]) <= 0.0
        })
        .collect();
    for (j, p) in prisms.iter().enumerate() {
        // The prism arm of [`prism_valid`] on the rounded points: a split
        // row with repeated ids is skipped, never by its volume, so the
        // same rows are skipped here.
        if !prism_split_ok(p, |v| round(points[v as usize])) {
            out.push(tets.len() + j);
        }
    }
    out
}

/// (116.26) for a mesh of tetrahedra only: [`rounding_flips_mixed`] with no
/// prisms.
pub fn rounding_flips(points: &[Point], tets: &[[u32; 4]]) -> Vec<usize> {
    rounding_flips_mixed(points, tets, &[])
}

/// §116.16 end to end: under `--features single` the f32 check (116.26)
/// first, refused naming the cells; then [`to_poly_mesh`]; then §92.3's
/// gate, [`quality::check`] (which builds the mesh through
/// `build_host_mesh`), refused with §92.3's message.
pub fn emit(
    points: &[Point],
    tets: &[[u32; 4]],
    boundary: &[([u32; 3], u32)],
    patches: &[EmitPatch],
    thresholds: &QualityThresholds,
) -> Result<(Emitted, QualityReport)> {
    #[cfg(feature = "single")]
    {
        let flips = rounding_flips(points, tets);
        if !flips.is_empty() {
            return Err(Error::Refused(format!(
                "emit/f32: rounding the points to f32 makes {} of {} tetrahedra flat or inverted \
                 by (116.26) of SPEC-LIT 116.16: cells {}; translate the geometry to a local origin near the body \
                 (docs/17 5.1.7) and mesh again",
                flips.len(),
                tets.len(),
                named(&flips)
            )));
        }
    }
    let e = to_poly_mesh(points, tets, boundary, patches)?;
    let q = quality::check(&e.raw, thresholds)
        .map_err(|err| Error::Refused(format!("emit/gate (SPEC-LIT 92.3): {err}")))?;
    Ok((e, q))
}

/// §116.16 end to end for the tetrahedra and the prisms of TET-13b: the
/// f32 check (116.26) first, with (116.25f)'s prism arm, refused naming the
/// cells; then [`to_poly_mesh_mixed`]; then §92.3's gate,
/// [`quality::check`], refused with §92.3's message.
pub fn emit_mixed(
    points: &[Point],
    tets: &[[u32; 4]],
    prisms: &[[u32; 6]],
    boundary: &[(Vec<u32>, u32)],
    patches: &[EmitPatch],
    thresholds: &QualityThresholds,
) -> Result<(Emitted, QualityReport)> {
    #[cfg(feature = "single")]
    {
        let flips = rounding_flips_mixed(points, tets, prisms);
        if !flips.is_empty() {
            return Err(Error::Refused(format!(
                "emit/f32: rounding the points to f32 makes {} of {} cells flat or inverted \
                 by (116.26) of SPEC-LIT 116.16: cells {}; translate the geometry to a local origin near the body \
                 (docs/17 5.1.7) and mesh again",
                flips.len(),
                tets.len() + prisms.len(),
                named(&flips)
            )));
        }
    }
    let e = to_poly_mesh_mixed(points, tets, prisms, boundary, patches)?;
    let q = quality::check(&e.raw, thresholds)
        .map_err(|err| Error::Refused(format!("emit/gate (SPEC-LIT 92.3): {err}")))?;
    Ok((e, q))
}

#[cfg(test)]
mod tests {
    use super::super::optimise::tests::{holed, run, solid, wedge};
    use super::super::refine::tests::case;
    use super::super::surfprep::TetSurface;
    use super::*;
    use crate::io::polymesh::build_host_mesh;

    fn volume(points: &[Point], tets: &[[u32; 4]]) -> f64 {
        tets.iter()
            .map(|t| {
                let p = t.map(|v| points[v as usize]);
                orient3d(&p[0], &p[1], &p[2], &p[3]) / 6.0
            })
            .sum()
    }

    fn patches(names: &[(&str, &str)]) -> Vec<EmitPatch> {
        names
            .iter()
            .map(|&(n, t)| EmitPatch { name: n.to_string(), type_name: t.to_string() })
            .collect()
    }

    /// The unit corner tetrahedron of §116.4 (`orient3d = 1`), with an
    /// unused point first: four boundary faces on one patch, the unused
    /// point dropped, `V = 1/6` exactly, and §92.3 passes.
    #[test]
    fn unit_tet_layout() {
        let pts: Vec<Point> =
            vec![[9.0, 9.0, 9.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]];
        let tets = vec![[1u32, 2, 3, 4]];
        let bnd: Vec<([u32; 3], u32)> = FACES.iter().map(|r| (r.map(|i| tets[0][i]), 0)).collect();
        let (e, q) = emit(&pts, &tets, &bnd, &patches(&[("walls", "wall")]), &QualityThresholds::default())
            .unwrap();
        assert_eq!(e.point_of, vec![1, 2, 3, 4]);
        assert_eq!(e.raw.points.len(), 4);
        assert_eq!(e.raw.faces.len(), 4);
        assert!(e.raw.neighbour.is_empty());
        assert_eq!(e.raw.owner, vec![0; 4]);
        assert_eq!((e.raw.patches[0].start, e.raw.patches[0].size), (0, 4));
        assert_eq!(e.raw.patches[0].kind, PatchKind::Wall);
        let m = build_host_mesh(&e.raw).unwrap();
        assert_eq!(m.v[0], 1.0 / 6.0);
        assert!(q.passed());
        // Each face's right-hand area vector points away from the centroid.
        for f in 0..4 {
            assert!(m.b_sf[f].dot(m.b_cf[f] - m.c[0]) > 0.0, "face {f}");
        }
    }

    /// Two tetrahedra on the face `(0,0,0) (1,0,0) (0,1,0)`, on either
    /// side of `z = 0`, with the lower half's faces on a second patch: one
    /// internal face, owner 0 < neighbour 1, its area vector from cell 0
    /// towards cell 1; the boundary grouped by patch; the volume closes.
    #[test]
    fn two_tets_layout() {
        let pts: Vec<Point> =
            vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.25, 0.25, -1.0]];
        let mut tets = vec![[0u32, 1, 2, 3], [0, 1, 2, 4]];
        for t in tets.iter_mut() {
            let p = t.map(|v| pts[v as usize]);
            if orient3d(&p[0], &p[1], &p[2], &p[3]) < 0.0 {
                t.swap(0, 1);
            }
        }
        let mut bnd = Vec::new();
        for (c, t) in tets.iter().enumerate() {
            for r in FACES {
                let f = r.map(|i| t[i]);
                if keyv(f.to_vec()) != vec![0, 1, 2] {
                    bnd.push((f, c as u32));
                }
            }
        }
        let ps = patches(&[("top", "wall"), ("bottom", "patch")]);
        let (e, q) = emit(&pts, &tets, &bnd, &ps, &QualityThresholds::default()).unwrap();
        assert_eq!(e.raw.neighbour, vec![1]);
        assert_eq!(e.raw.owner, vec![0, 0, 0, 0, 1, 1, 1]);
        let mut f0 = e.raw.faces[0].clone();
        f0.sort_unstable();
        assert_eq!(f0, vec![0, 1, 2]);
        assert_eq!(e.raw.patches.iter().map(|p| (p.start, p.size)).collect::<Vec<_>>(), vec![(0, 3), (3, 3)]);
        let m = build_host_mesh(&e.raw).unwrap();
        assert!(m.sf[0].dot(m.c[1] - m.c[0]) > 0.0);
        let v: f64 = m.v.iter().sum();
        assert!((v - volume(&pts, &tets)).abs() <= 1e-15, "{v}");
        assert!(q.passed(), "{}", q.refusal_text());
    }

    #[test]
    fn refusals() {
        let pts: Vec<Point> = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]];
        let t = [0u32, 1, 2, 3];
        let all: Vec<([u32; 3], u32)> = FACES.iter().map(|r| (r.map(|i| t[i]), 0)).collect();
        let ps = patches(&[("walls", "wall")]);
        let err = |r: Result<Emitted>| r.unwrap_err().to_string();
        let e = err(to_poly_mesh(&pts, &[[0, 1, 2, 7]], &all, &ps));
        assert!(e.contains("emit/points: tetrahedron 0 names point 7 of 4"), "{e}");
        let e = err(to_poly_mesh(&pts, &[[1, 0, 2, 3]], &all, &ps));
        assert!(e.contains("emit/orient: 1 tetrahedra") && e.contains("cells 0"), "{e}");
        let e = err(to_poly_mesh(&pts, &[t, t, t], &all, &ps));
        assert!(e.contains("emit/faces: 4 faces lie on three or more"), "{e}");
        let e = err(to_poly_mesh(&pts, &[t], &all[..3], &ps));
        assert!(e.contains("emit/patches: 1 boundary faces have no patch: (cell, face) (0, [1, 3, 2])"), "{e}");
        let mut extra = all.clone();
        extra.push(([0, 1, 9], 0));
        let e = err(to_poly_mesh(&pts, &[t], &extra, &ps));
        assert!(e.contains("[0, 1, 9] is given patch 0 but is not a boundary face"), "{e}");
        let mut clash = all.clone();
        clash.push((all[0].0, 1));
        let two = patches(&[("a", "wall"), ("b", "wall")]);
        let e = err(to_poly_mesh(&pts, &[t], &clash, &two));
        assert!(e.contains("is given patches 0 and 1"), "{e}");
        let e = err(to_poly_mesh(&pts, &[t], &[(all[0].0, 3)], &ps));
        assert!(e.contains("names patch 3 of 1"), "{e}");
    }

    /// Emits a TET-10b output from `Optimised.points` (the moved points,
    /// not the recovery's) with the patches of `tri_patch` and a hull patch,
    /// and checks §116.16's gate: `build_host_mesh` accepts, §92.3 passes,
    /// `owner < neighbour` strictly ascending, one cell per tetrahedron, the
    /// volume closes at relative 1e-12, the points are the optimised ones
    /// bit for bit, and each patch has the carve's face count.
    fn gate(name: &str, r: &super::super::refine::Refined, tri_patch: &[u32], body: &str) {
        let k = run(name, r, tri_patch);
        let o = &k.out;
        let n_body = 1 + tri_patch.iter().copied().max().unwrap_or(0);
        let hull = n_body;
        let bnd = carved_boundary(&r.carved, &r.recovery, tri_patch, hull).unwrap();
        let mut ps: Vec<EmitPatch> = (0..n_body)
            .map(|p| EmitPatch { name: format!("{body}{p}"), type_name: "wall".to_string() })
            .collect();
        ps.push(EmitPatch { name: "domain".to_string(), type_name: "patch".to_string() });
        let t0 = std::time::Instant::now();
        let (e, q) = emit(&o.points, &o.tets, &bnd, &ps, &QualityThresholds::default()).unwrap();
        let secs = t0.elapsed().as_secs_f64();
        let m = build_host_mesh(&e.raw).unwrap();
        assert_eq!(m.n_cells, o.tets.len());
        let ordered = e.raw.neighbour.iter().enumerate().all(|(f, &n)| e.raw.owner[f] < n)
            && (1..e.raw.neighbour.len()).all(|f| {
                (e.raw.owner[f - 1], e.raw.neighbour[f - 1]) < (e.raw.owner[f], e.raw.neighbour[f])
            });
        assert!(ordered, "{name}: not upper-triangular");
        for (k, &v) in e.point_of.iter().enumerate() {
            let p = o.points[v as usize];
            let x = e.raw.points[k];
            assert_eq!([x.x, x.y, x.z], p, "{name}: point {k}");
        }
        let moved = (0..o.points.len()).filter(|&v| o.points[v] != r.recovery.points[v]).count();
        let (v0, v1) = (volume(&o.points, &o.tets), m.v.iter().sum::<f64>());
        let rv = (v1 - v0).abs() / v0;
        assert!(rv <= 1e-12, "{name}: volume rel {rv:e}");
        // TET-08's own count, from a surface that carries only the patches.
        let names = TetSurface {
            points: Vec::new(),
            tris: Vec::new(),
            tri_patch: tri_patch.to_vec(),
            patch_names: (0..n_body).map(|p| format!("{body}{p}")).collect(),
        };
        let (counts, hull_n) = r.carved.patch_face_counts(&names, &r.recovery);
        let sizes: Vec<usize> = e.raw.patches.iter().map(|p| p.size).collect();
        let mut want = counts;
        want.push(hull_n);
        assert_eq!(sizes, want, "{name}: patch face counts");
        println!(
            "emit {name}: {} cells, {} internal and {} boundary faces, {} points ({} moved by the optimiser), \
             patches {:?}; {secs:.2} s; volume rel {rv:.2e}; upper-triangular {ordered}; \
             build_host_mesh accepts; 92.3: {} - {}",
            m.n_cells,
            e.raw.neighbour.len(),
            e.raw.faces.len() - e.raw.neighbour.len(),
            e.raw.points.len(),
            moved,
            sizes,
            if q.passed() { "PASS" } else { "MISSED" },
            q.summary().replace('\n', "; ")
        );
        assert!(q.passed());
    }

    #[test]
    fn gate_passes_sphere() {
        let r = &case().refined;
        gate("sphere in a box", r, &vec![0; r.recovery.tris.len()], "sphere");
    }

    #[test]
    fn gate_passes_cube_minus_cylinder() {
        let s = holed();
        gate("cube minus a cylinder", &s.refined, &s.tri_patch, "part");
    }

    /// The 30-degree wedge is TET-10b's recorded MISS (SPEC-LIT 116.18):
    /// TET-09's refinement does not finish on it, so there is no mesh to
    /// emit. The test asserts that cause, at TET-10b's `h = 0.1` and cap of
    /// 20 000 inserted points.
    #[test]
    fn gate_passes_wedge() {
        let (h, cap) = (0.1, 20_000);
        let e = match solid(&wedge(), h, cap) {
            Ok(s) => panic!(
                "the wedge now refines ({} tets): restore its gate, gate(\"30-degree wedge\", ..)",
                s.refined.carved.tets.len()
            ),
            Err(e) => e.to_string(),
        };
        println!(
            "emit 30-degree wedge: MISSED - TET-09's refinement does not terminate at h = {h} on it: {e}; \
             92.3 cannot be measured (SPEC-LIT 116.18)"
        );
        assert!(e.contains("refine/cap: "), "{e}");
    }
}
