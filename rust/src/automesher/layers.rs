// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.
// Provenance: see PROVENANCE.md. No GPL-licensed source was consulted.

//! Wall layers - SPEC-LIT §92.13, §92.2's stage 6 end to end: the normal a
//! point grows along (92.40)-(92.42), the thickness it is allowed
//! (92.43)-(92.45), the shrink that moves the boundary inward
//! (92.46)-(92.47), and the extrusion that fills the gap it opened
//! (92.48)-(92.50), with (92.51)'s refusal in front of it. The stack itself
//! is (92.9) and the medial limit is (92.10), both of §92.2 stage 6.
//! [`shrink`] alone changes no topology - its mesh differs from its input in
//! `points` alone - and [`add_layers`] is the whole stage: shrink, extrude,
//! renumber, and §92.3's gate.
//!
//! The construction is the standard one - shrink the boundary inward, fill
//! the gap - and it is Garimella & Shephard's, *Int. J. Numer. Meth. Engng*
//! **49** (2000) 193-218 (DOI
//! `10.1002/1097-0207(20000910/20)49:1/2<193::AID-NME929>3.0.CO;2-R`), which
//! §92.2 cites for the same reason. The medial distance is measured by the
//! march (92.44) over [`TriIndex::closest_point`] alone. snappyHexMesh is
//! GPL and was not opened; the OpenFOAM *User Guide*'s prose description of
//! layer addition, cited in §92.2, is the only thing consulted about it.
//!
//! The shrink's interior relaxation is (92.29)'s shape, as §92.13 restates
//! it: fixed values on the boundary, a weighted neighbour mean on the whole
//! point graph behind it, and (92.33)'s hanging-node line applied last, for
//! (92.33)'s reason and the one §92.13 adds. §92.3's gate measures every
//! trial, and a patch the gate cannot be satisfied on loses its layers by
//! name rather than stopping the run.
//!
//! Provenance: ORIGINAL - the equations are stated in SPEC-LIT §92.13 and
//! §92.2, and the closest-point march is built from the crate's own query.
//! No GPL-licensed source was consulted.

use crate::error::{Error, Result};
use crate::io::polymesh::{build_host_mesh, PolyMeshRaw};
use crate::surface::{Surface, TriIndex};
use crate::{Scalar, Vec3};

use super::quality::{self, Gate, QualityThresholds};
use std::collections::{HashMap, HashSet};

use super::snap::{face_area_vector, find_hanging};
use crate::adapt::rebuild::ldu_permutation;
use super::{LayerSpec, LayerTerminate};

// ==========================================================================
//  The stack
// ==========================================================================

/// (92.44)'s `kappa`: a hit needs the surface to come back and MEET the
/// march - `|q - y| <= (1 - kappa) s` - and a flat wall gives exactly `s`,
/// so only a second piece of wall can satisfy it. This rejects the convex
/// corner, where the closest point is the launch point itself.
const KAPPA: Scalar = 1.0 / 8.0;

/// (92.44)'s `c`: a hit needs the closest point FAR along the surface from
/// where the march started - `|q - x| > c s`. This rejects the concave
/// corner, where the closest point is the foot of the launch neighbourhood
/// and `|q - x| = s / sqrt(2)`.
const MARCH_C: Scalar = 1.5;

/// (92.9) and (92.43): the stack, once, for the whole run.
///
/// `t` holds `t_1..t_n`, `total` their sum `T`, and `f` the fractions
/// `f_0..f_n` with `f[0] = 0` and `f[n] = 1` assigned rather than divided
/// into. Every point uses the same schedule, which is what keeps a hanging
/// node's copies at the exact midpoints of its parents' copies at every
/// level.
#[derive(Debug, Clone)]
pub struct Stack {
    pub n: usize,
    pub t: Vec<Scalar>,
    pub total: Scalar,
    pub f: Vec<Scalar>,
}

/// Validate the layer numbers and build the [`Stack`].
///
/// Refuses, naming the field and its value: `first_thickness` or `growth`
/// not positive and finite, `medial_frac` or `cell_frac` outside `(0, 1]`,
/// `smoothing` or `min_thickness` outside `[0, 1]`. `n == 0` is the empty
/// stack, not a refusal.
pub fn stack(spec: &LayerSpec) -> Result<Stack> {
    if spec.n == 0 {
        return Ok(Stack {
            n: 0,
            t: Vec::new(),
            total: 0.0,
            f: Vec::new(),
        });
    }
    if !(spec.first_thickness > 0.0) || !spec.first_thickness.is_finite() {
        return Err(Error::Mesh(format!(
            "layers: layers.first_thickness must be positive and finite, \
             got {} - it is the stack's first thickness t_1 of (92.9)",
            spec.first_thickness
        )));
    }
    if !(spec.growth > 0.0) || !spec.growth.is_finite() {
        return Err(Error::Mesh(format!(
            "layers: layers.growth must be positive and finite, got {} - \
             it is the expansion from one layer to the next of (92.9)",
            spec.growth
        )));
    }
    for (name, v) in [("medial_frac", spec.medial_frac), ("cell_frac", spec.cell_frac)] {
        if !(v > 0.0) || v > 1.0 || !v.is_finite() {
            return Err(Error::Mesh(format!(
                "layers: layers.{name} must lie in (0, 1], got {v}"
            )));
        }
    }
    for (name, v) in [
        ("smoothing", spec.smoothing),
        ("min_thickness", spec.min_thickness),
    ] {
        if !(v >= 0.0) || v > 1.0 || !v.is_finite() {
            return Err(Error::Mesh(format!(
                "layers: layers.{name} must lie in [0, 1], got {v}"
            )));
        }
    }
    let (n, t1, g) = (spec.n, spec.first_thickness, spec.growth);
    let mut t = Vec::with_capacity(n);
    let mut total = 0.0f64;
    for k in 0..n {
        let tk = t1 * g.powi(k as i32);
        t.push(tk as Scalar);
        total += tk;
    }
    let mut f = Vec::with_capacity(n + 1);
    let mut cum = 0.0f64;
    f.push(0.0);
    for k in 0..n {
        cum += t[k] as f64;
        f.push((cum / total) as Scalar);
    }
    f[n] = 1.0;
    Ok(Stack {
        n,
        t,
        total: total as Scalar,
        f,
    })
}

// ==========================================================================
//  The medial distance
// ==========================================================================

/// (92.44), on its own so it can be tested without a mesh.
///
/// Marches from `x` along `n`, sampling `s_max j / 8` for the first `j`
/// that hits and then bisecting six times between the last miss and the
/// first hit, keeping the invariant "lo does not hit, hi hits" and
/// returning the hi end. A hit is [`KAPPA`]/[`MARCH_C`]'s two-sided test on
/// [`TriIndex::closest_point`]: fourteen queries, bounded and
/// deterministic, and an APPROXIMATION - a gap narrower than `s_max / 8`
/// that opens and closes between two samples is missed, and what catches it
/// then is the gate, not this. Returns [`Scalar::INFINITY`] when nothing is
/// hit inside `s_max`, and when `s_max <= 0`.
pub fn medial_distance(idx: &TriIndex, x: Vec3, n: Vec3, s_max: Scalar) -> Scalar {
    if !(s_max > 0.0) || !s_max.is_finite() {
        return Scalar::INFINITY;
    }
    let hit = |s: Scalar| -> bool {
        let y = x + n * s;
        let (q, _, _) = idx.closest_point(y);
        (q - y).mag() <= (1.0 - KAPPA) * s && (q - x).mag() > MARCH_C * s
    };
    let mut lo = 0.0;
    let mut hi = None;
    for j in 1..=8 {
        let s = s_max * (j as Scalar) / 8.0;
        if hit(s) {
            hi = Some(s);
            break;
        }
        lo = s;
    }
    let Some(mut hi) = hi else {
        return Scalar::INFINITY;
    };
    for _ in 0..6 {
        let mid = 0.5 * (lo + hi);
        if hit(mid) {
            hi = mid;
        } else {
            lo = mid;
        }
    }
    hi
}

// ==========================================================================
//  The field
// ==========================================================================

/// The patch indices `layers.patches` names, ascending. Refuses a name that
/// is not a patch of the mesh, naming it: the geometry it names either was
/// never reached by a cell or is misspelled, and both are the user's to
/// fix.
pub fn resolve_patches(mesh: &PolyMeshRaw, spec: &LayerSpec) -> Result<Vec<usize>> {
    let mut out = Vec::new();
    for name in &spec.patches {
        match mesh.patches.iter().position(|p| p.name == *name) {
            Some(p) => out.push(p),
            None => {
                let named: Vec<&str> =
                    mesh.patches.iter().map(|p| p.name.as_str()).collect();
                return Err(Error::Mesh(format!(
                    "layers: patch \"{name}\" is not a patch of the mesh - \
                     the mesh carries {}, named {named:?}",
                    mesh.patches.len()
                )));
            }
        }
    }
    out.sort_unstable();
    out.dedup();
    Ok(out)
}

/// (92.40)-(92.45): everything about the points, for ONE set of layer
/// patches. `is_layer`, `normal`, `thickness`, `disp` and `pinned` are per
/// point; `faces` are GLOBAL face ids, ascending, with `face_patch`
/// parallel to them.
///
/// As [`field`] returns it, `disp` is (92.45)'s `D_i = T_i n_i` and
/// `thickness` is `T_i`. As [`shrink`] returns it inside [`Shrunk`], both
/// have been overwritten with what the mesh ACTUALLY got: `disp` is the
/// displacement the accepted iterate applied - after the retreats and after
/// (92.46)'s hanging line - and `thickness` is its magnitude. The extrusion
/// reads these, so `mesh.points[i] + disp[i]` is exactly where the shrunk
/// wall is.
#[derive(Debug, Clone)]
pub struct Field {
    pub patches: Vec<usize>,
    pub faces: Vec<usize>,
    pub face_patch: Vec<usize>,
    pub is_layer: Vec<bool>,
    pub normal: Vec<Vec3>,
    pub thickness: Vec<Scalar>,
    pub disp: Vec<Vec3>,
    pub pinned: Vec<bool>,
}

/// Build the field for ONE set of layer patches - (92.40) through (92.45).
///
/// `patches` are ascending patch indices, as [`resolve_patches`] returns
/// them. A LAYER face is a boundary face of one of them; its global id is
/// `n_internal + patch.start + j`, because `PatchInfo::start` counts from
/// the FIRST BOUNDARY FACE. The medial march runs against `idx`, built by
/// the caller once for the whole stage.
pub fn field(
    mesh: &PolyMeshRaw,
    idx: &TriIndex,
    patches: &[usize],
    spec: &LayerSpec,
    st: &Stack,
) -> Result<Field> {
    let n_points = mesh.points.len();
    let n_faces = mesh.faces.len();
    let n_internal = mesh.neighbour.len().min(n_faces);
    let mut is_layer_patch = vec![false; mesh.patches.len()];
    for &p in patches {
        is_layer_patch[p] = true;
    }
    let mut faces = Vec::new();
    let mut face_patch = Vec::new();
    let mut is_layer_face = vec![false; n_faces];
    for (p, patch) in mesh.patches.iter().enumerate() {
        if !is_layer_patch[p] {
            continue;
        }
        for j in 0..patch.size {
            let f = n_internal + patch.start + j;
            if f < n_faces {
                faces.push(f);
                face_patch.push(p);
                is_layer_face[f] = true;
            }
        }
    }
    let mut is_layer = vec![false; n_points];
    for &f in &faces {
        for &p in &mesh.faces[f] {
            is_layer[p as usize] = true;
        }
    }
    // (92.40): the area-weighted average is the sum of the inward area
    // vectors, normalised - no weight is written down, which is what makes
    // the normal at a 2:1 transition the normal of the surface and not of
    // the face count. A zero sum pins the point.
    let mut acc = vec![Vec3::ZERO; n_points];
    for &f in &faces {
        let sf = face_area_vector(&mesh.points, &mesh.faces[f]);
        for &p in &mesh.faces[f] {
            acc[p as usize] = acc[p as usize] - sf;
        }
    }
    let mut normal = vec![Vec3::ZERO; n_points];
    let mut pinned = vec![false; n_points];
    for i in 0..n_points {
        if !is_layer[i] {
            continue;
        }
        if acc[i].mag_sqr() > 0.0 {
            normal[i] = acc[i].normalised();
        } else {
            pinned[i] = true;
        }
    }
    // (92.41): smoothing passes over the wall's own graph - the edge
    // neighbours along LAYER faces only, of (92.27)'s W(i) restricted to L -
    // into a fresh vector each pass, renormalising after each. A point with
    // no layer neighbour keeps its own normal.
    let mut layer_nbrs: Vec<Vec<u32>> = vec![Vec::new(); n_points];
    for &f in &faces {
        let face = &mesh.faces[f];
        for k in 0..face.len() {
            let a = face[k] as usize;
            let b = face[(k + 1) % face.len()] as usize;
            layer_nbrs[a].push(b as u32);
            layer_nbrs[b].push(a as u32);
        }
    }
    for list in layer_nbrs.iter_mut() {
        list.sort_unstable();
        list.dedup();
    }
    for _ in 0..spec.normal_passes {
        let mut next = normal.clone();
        for i in 0..n_points {
            if !is_layer[i] {
                continue;
            }
            let mut mean = Vec3::ZERO;
            let mut cnt = 0usize;
            for &j in &layer_nbrs[i] {
                if is_layer[j as usize] {
                    mean = mean + normal[j as usize];
                    cnt += 1;
                }
            }
            if cnt == 0 {
                continue;
            }
            mean = mean / (cnt as Scalar);
            next[i] = ((1.0 - spec.smoothing) * normal[i]
                + spec.smoothing * mean)
                .normalised();
        }
        normal = next;
    }
    // (92.42): where a layer point is also carried by a boundary face that
    // is NOT a layer face, the normal is constrained into that face's plane
    // rather than the point pinned - the same instrument as (92.30),
    // stated for a general face normal. Collect the outward unit normals of
    // the non-layer boundary faces carrying each point, deduped.
    let mut us: Vec<Vec<Vec3>> = vec![Vec::new(); n_points];
    for f in n_internal..n_faces {
        if is_layer_face[f] {
            continue;
        }
        let sf = face_area_vector(&mesh.points, &mesh.faces[f]);
        let m = sf.mag();
        if !(m > 0.0) {
            continue;
        }
        let u = sf * (1.0 / m);
        for &p in &mesh.faces[f] {
            us[p as usize].push(u);
        }
    }
    for i in 0..n_points {
        if !is_layer[i] || pinned[i] {
            continue;
        }
        // (92.42) in patch mode, (92.69) in face mode: the merge tolerance
        // is the face mode's junction angle, the patch mode keeps the exact
        // 1e-6 dedupe it has always had.
        let tol = match spec.terminate {
            LayerTerminate::Patch => None,
            LayerTerminate::Face => Some(spec.junction_angle_deg),
        };
        match constrain_normal(normal[i], &us[i], tol) {
            Some(n) => normal[i] = n,
            None => pinned[i] = true, // three constraints leave nothing
        }
    }
    // (92.45): the thickness a point is allowed - the nominal T, the medial
    // limit of (92.44), and the local cell size `h_i`, the SHORTEST edge at
    // i among the layer faces. A pinned point gets nothing.
    let mut h = vec![Scalar::INFINITY; n_points];
    for &f in &faces {
        let face = &mesh.faces[f];
        for k in 0..face.len() {
            let a = face[k] as usize;
            let b = face[(k + 1) % face.len()] as usize;
            let d = (mesh.points[a] - mesh.points[b]).mag();
            if d < h[a] {
                h[a] = d;
            }
            if d < h[b] {
                h[b] = d;
            }
        }
    }
    // (92.45') in face mode: the height of the cell BEHIND the wall - the
    // owner cell's volume over the layer face's area, the smallest over the
    // layer faces carrying the point. Patch mode's array stays at INFINITY
    // and the limiter below stays bit for bit (92.45) as it was.
    let hh = match spec.terminate {
        LayerTerminate::Face => owner_heights(mesh, &faces)?,
        LayerTerminate::Patch => vec![Scalar::INFINITY; n_points],
    };
    let s_max = st.total / spec.medial_frac;
    let mut thickness = vec![0.0; n_points];
    let mut disp = vec![Vec3::ZERO; n_points];
    for i in 0..n_points {
        if !is_layer[i] || pinned[i] {
            continue;
        }
        let m_i = medial_distance(idx, mesh.points[i], normal[i], s_max);
        let t_medial = if m_i.is_finite() {
            spec.medial_frac * m_i
        } else {
            st.total
        };
        let t_i = st
            .total
            .min(t_medial)
            .min(spec.cell_frac * h[i])
            .min(spec.cell_frac * hh[i]);
        thickness[i] = t_i;
        disp[i] = normal[i] * t_i;
    }
    Ok(Field {
        patches: patches.to_vec(),
        faces,
        face_patch,
        is_layer,
        normal,
        thickness,
        disp,
        pinned,
    })
}

/// (92.45)'s face geometry, read the way `crate::mesh::geometry`'s own
/// sweep reads it: `Sf` the vector sum of the fan triangles' normals about
/// the vertex average, `Cf` their area-weighted centroid, the vertex
/// average when the face is too small to weight by. That fn is private to
/// the mesh module, and §92.3's volume is its reading, so this copy is its
/// formula character for character.
fn face_geometry_of(face: &[crate::Label], points: &[Vec3]) -> (Vec3, Vec3) {
    const SMALL: Scalar = 1.0e-19;
    let n = face.len();
    if n == 0 {
        return (Vec3::ZERO, Vec3::ZERO);
    }
    let mut x_avg = Vec3::ZERO;
    for &v in face {
        x_avg += points[v as usize];
    }
    x_avg = x_avg / n as Scalar;
    if n < 3 {
        return (Vec3::ZERO, x_avg);
    }
    let mut sf = Vec3::ZERO;
    let mut cf = Vec3::ZERO;
    let mut area: Scalar = 0.0;
    for i in 0..n {
        let a = points[face[i] as usize];
        let b = points[face[(i + 1) % n] as usize];
        let t_n = (a - x_avg).cross(b - x_avg);
        let t_c = (x_avg + a + b) / 3.0;
        let t_a = t_n.mag() * 0.5;
        sf += t_n * 0.5;
        cf += t_c * t_a;
        area += t_a;
    }
    if area > SMALL {
        (sf, cf / area)
    } else {
        (sf, x_avg)
    }
}

/// (92.45')'s face-mode heights: per point, the smallest owner-cell height
/// `V_own(f) / |Sf|` over the layer faces `faces` carrying it, INFINITY off
/// them. `V_own` is §92.3's own volume - the pyramid decomposition whose
/// apex is the mean of the cell's face centroids - of the owner cell on the
/// INPUT mesh, because (92.45)'s `h_i` reads the wall face's EDGES and a
/// sliver cut cell slips under every one of them: it is the height of the
/// cell BEHIND the wall that bounds how far the wall may move.
fn owner_heights(mesh: &PolyMeshRaw, faces: &[usize]) -> Result<Vec<Scalar>> {
    let n_internal = mesh.neighbour.len().min(mesh.faces.len());
    let mut n_cells = 0usize;
    for &o in &mesh.owner {
        n_cells = n_cells.max(o as usize + 1);
    }
    for &nb in &mesh.neighbour {
        n_cells = n_cells.max(nb as usize + 1);
    }
    // Only the owners of layer faces are wanted, so the sweep below
    // measures a face only when one of its cells is theirs - the near-wall
    // faces, not the mesh.
    let mut wanted = vec![false; n_cells];
    for &f in faces {
        wanted[mesh.owner[f] as usize] = true;
    }
    // The apex pass: the mean of the cell's face centroids.
    let mut apex = vec![Vec3::ZERO; n_cells];
    let mut n_cell_faces = vec![0u32; n_cells];
    for f in 0..mesh.faces.len() {
        let internal = f < n_internal;
        let own = mesh.owner[f] as usize;
        let nbr = if internal {
            mesh.neighbour[f] as usize
        } else {
            own
        };
        if !wanted[own] && !(internal && wanted[nbr]) {
            continue;
        }
        let (_, cf) = face_geometry_of(&mesh.faces[f], &mesh.points);
        if wanted[own] {
            apex[own] += cf;
            n_cell_faces[own] += 1;
        }
        if internal && wanted[nbr] {
            apex[nbr] += cf;
            n_cell_faces[nbr] += 1;
        }
    }
    for c in 0..n_cells {
        if n_cell_faces[c] > 0 {
            apex[c] = apex[c] / n_cell_faces[c] as Scalar;
        }
    }
    // The volume pass: V = (1/3) sum_f (s Sf) . (Cf - apex), the owner at
    // +Sf and the neighbour at -Sf, exactly the mesh module's decomposition.
    let mut vol = vec![0.0; n_cells];
    for f in 0..mesh.faces.len() {
        let internal = f < n_internal;
        let own = mesh.owner[f] as usize;
        let nbr = if internal {
            mesh.neighbour[f] as usize
        } else {
            own
        };
        if !wanted[own] && !(internal && wanted[nbr]) {
            continue;
        }
        let (sf, cf) = face_geometry_of(&mesh.faces[f], &mesh.points);
        if wanted[own] {
            vol[own] += sf.dot(cf - apex[own]) / 3.0;
        }
        if internal && wanted[nbr] {
            vol[nbr] -= sf.dot(cf - apex[nbr]) / 3.0;
        }
    }
    // The heights, and the smallest over each point's layer faces. A face
    // with no area sets no bound - its height is not a length, and the
    // faces around its points still are.
    let mut hh = vec![Scalar::INFINITY; mesh.points.len()];
    for &f in faces {
        let (sf, _) = face_geometry_of(&mesh.faces[f], &mesh.points);
        let a = sf.mag();
        if !(a > 0.0) {
            continue;
        }
        let height = vol[mesh.owner[f] as usize] / a;
        for &p in &mesh.faces[f] {
            let p = p as usize;
            if height < hh[p] {
                hh[p] = height;
            }
        }
    }
    Ok(hh)
}

/// The no-layers field: per-point arrays of the right length, nothing on L.
fn empty_field(n_points: usize) -> Field {
    Field {
        patches: Vec::new(),
        faces: Vec::new(),
        face_patch: Vec::new(),
        is_layer: vec![false; n_points],
        normal: vec![Vec3::ZERO; n_points],
        thickness: vec![0.0; n_points],
        disp: vec![Vec3::ZERO; n_points],
        pinned: vec![false; n_points],
    }
}

// ==========================================================================
//  The junction merge, and the per-face termination
// ==========================================================================

/// (92.42) and (92.69): the normal of a layer point after the non-layer
/// boundary faces carrying it constrain it. `us` are their outward unit
/// normals in face order. `tol_deg` None is (92.42) exactly as `field` does
/// it today (dedupe at u . v > 1 - 1e-6, |U| = 1 project with the 0.1 pin,
/// |U| = 2 the signed cross product with the 1e-9 pin, |U| >= 3 pinned);
/// `Some(theta)` is (92.69). `None` returned = the point is pinned (its
/// thickness is 0).
pub fn constrain_normal(n: Vec3, us: &[Vec3], tol_deg: Option<Scalar>) -> Option<Vec3> {
    let uniq: Vec<Vec3> = match tol_deg {
        // (92.42)'s own dedupe: exact parallelism to 1e-6 - a snapped plane
        // is not flat to that, which is why the face mode merges by angle
        // instead.
        None => {
            let mut uniq: Vec<Vec3> = Vec::new();
            for &u in us {
                if !uniq.iter().any(|v| v.dot(u) > 1.0 - 1e-6) {
                    uniq.push(u);
                }
            }
            uniq
        }
        // (92.69)'s C(i): u joins the first cluster whose representative
        // agrees with it to within theta_U, else opens one of its own; the
        // representatives stay in the order they opened.
        Some(theta) => {
            let cos = theta.to_radians().cos();
            let mut reps: Vec<Vec3> = Vec::new();
            for &u in us {
                match reps.iter().position(|r| u.dot(*r) >= cos) {
                    Some(_) => {}
                    None => reps.push(u),
                }
            }
            reps
        }
    };
    match uniq.len() {
        0 => Some(n), // nothing constrains it
        1 => {
            let u = uniq[0];
            let proj = n - u * n.dot(u);
            if proj.mag() < 0.1 {
                None
            } else {
                Some(proj.normalised())
            }
        }
        2 => {
            let cross = uniq[0].cross(uniq[1]);
            if cross.mag() < 1e-9 {
                // Two planes that do not meet in a line: pinned.
                None
            } else {
                let c = if cross.dot(n) < 0.0 {
                    cross * -1.0
                } else {
                    cross
                };
                Some(c.normalised())
            }
        }
        // (92.69)'s rank-2/rank-3 line, reached only in face mode: the pair
        // of representatives whose cross product is largest (ties to the
        // lowest a, then lowest b) spans the plane the constraints leave;
        // when every representative stays within theta_U's sine of it the
        // normal is that line, otherwise nothing is left.
        _ => {
            let theta = match tol_deg {
                Some(t) => t,
                None => return None, // (92.42): three constraints pin
            };
            let (mut a, mut b) = (0usize, 1usize);
            for i in 0..uniq.len() {
                for j in (i + 1)..uniq.len() {
                    if uniq[i].cross(uniq[j]).mag() > uniq[a].cross(uniq[b]).mag() {
                        a = i;
                        b = j;
                    }
                }
            }
            let cr = uniq[a].cross(uniq[b]);
            if !(cr.mag() > 1e-9) {
                return None;
            }
            let e = cr.normalised();
            let sin = theta.to_radians().sin();
            if uniq.iter().all(|r| r.dot(e).abs() <= sin) {
                let e = if e.dot(n) < 0.0 { e * -1.0 } else { e };
                Some(e.normalised())
            } else {
                None
            }
        }
    }
}

/// (92.70): a layer face's class from the anchored flags of its points.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FaceClass {
    /// No point of the face is anchored: it carries its `n` cells as (92.48).
    Keep,
    /// Some but not all of its points are anchored: it tapers to zero in ONE
    /// wedge cell.
    Ring,
    /// Every point of the face is anchored: it goes without layers.
    Off,
}

/// (92.70): `face`'s class from `anchored` (indexed by point id).
pub fn face_class(face: &[crate::Label], anchored: &[bool]) -> FaceClass {
    let mut any = false;
    let mut all = true;
    for &p in face {
        if anchored[p as usize] {
            any = true;
        } else {
            all = false;
        }
    }
    if all {
        FaceClass::Off
    } else if any {
        FaceClass::Ring
    } else {
        FaceClass::Keep
    }
}

/// (92.73): what one failing layer point gets.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LocalStep {
    /// `D_i <- D_i / 2`, its halving count up by one.
    Halve,
    /// `D_i <- 0`: the point is anchored.
    Anchor,
}

/// (92.73): the step one failing layer point takes - a point of a cell G5
/// names anchors at once (halving `t` halves `V`, so G5's `tau` falls), as
/// does one at its retreat limit or one whose halving would take it under
/// the floor; every other failing point halves.
pub fn local_step(
    in_g5: bool,
    halvings: usize,
    d_mag: Scalar,
    floor: Scalar,
    retreat_limit: usize,
) -> LocalStep {
    if in_g5 || halvings >= retreat_limit || d_mag / 2.0 < floor {
        LocalStep::Anchor
    } else {
        LocalStep::Halve
    }
}

/// (92.73): steps one ladder may take on one patch set in face mode before
/// (92.47)'s patch rule. This project's own number, like `BETA_RUNG_LIMIT`;
/// 12 and then 24 were not enough for the full-size F1 tunnel's body, whose ladder spent
/// its steps on the same squeezed cells the freeze now holds.
pub const TERMINATE_STEP_LIMIT: usize = 48;

/// (92.73): the OUTER ladder's per-patch-set state - every counter the
/// ladder runs on one patch set, and the M and X face marks it carries into
/// every attempt. A patch drop resets the lot, so the next patch set starts
/// clean and never at the last set's step limit.
pub struct FaceLadder {
    /// (92.47)'s per-point retreat caps. Patch mode halves and zeroes them;
    /// the face-mode outer ladder moves no point and they stay all ones.
    pub caps: Vec<Scalar>,
    /// (92.66)'s pull, carried into every attempt as the caps are.
    pub betas: Vec<Scalar>,
    /// (92.70): M, indexed by INPUT face id.
    pub merged: Vec<bool>,
    /// (92.70): X, indexed by INPUT face id.
    pub cut: Vec<bool>,
    /// (92.73): per-point halving counts. Written by patch mode and by the
    /// INNER ladder's own bookkeeping only; reset with the rest.
    pub out_h: Vec<usize>,
    /// The halvings this patch set took before the last measurement.
    pub halvings: usize,
    /// The steps this patch set took - face mode's merges and cuts included.
    pub steps: usize,
    /// The (92.66) rungs this patch set took.
    pub beta_rungs: usize,
}

impl FaceLadder {
    /// Fresh state for a run on a mesh of `n_points` points and `n_faces`
    /// faces.
    pub fn new(n_points: usize, n_faces: usize) -> Self {
        let mut fl = FaceLadder {
            caps: Vec::new(),
            betas: Vec::new(),
            merged: Vec::new(),
            cut: Vec::new(),
            out_h: Vec::new(),
            halvings: 0,
            steps: 0,
            beta_rungs: 0,
        };
        fl.reset(n_points, n_faces);
        fl
    }

    /// (92.73): the per-patch-set reset - every counter to zero, the caps
    /// and the pull back to one, M and X emptied.
    pub fn reset(&mut self, n_points: usize, n_faces: usize) {
        self.caps = vec![1.0 as Scalar; n_points];
        self.betas = vec![1.0 as Scalar; n_points];
        self.merged = vec![false; n_faces];
        self.cut = vec![false; n_faces];
        self.out_h = vec![0; n_points];
        self.halvings = 0;
        self.steps = 0;
        self.beta_rungs = 0;
    }
}

// ==========================================================================
//  The shrink
// ==========================================================================

/// (92.46) and (92.47): the boundary moved inward, the interior relaxed,
/// the gate satisfied, and the patches that had to give up their layers.
///
/// `mesh` differs from the input in `points` alone - the topology is
/// untouched. `retreats` counts the halvings actually taken, summed over
/// every round; `dropped` names each patch that lost its layers and why.
#[derive(Debug, Clone)]
pub struct Shrunk {
    pub mesh: PolyMeshRaw,
    pub field: Field,
    pub retreats: usize,
    pub dropped: Vec<(String, String)>,
    /// What drove each drop of `dropped`, by patch name, in the same order.
    pub drop_causes: Vec<(String, DropCause)>,
    /// Every measurement the inner ladder took, in order, all in round 0:
    /// the caller numbers the round.
    pub ladder: Vec<LadderEntry>,
    /// Per patch of the caller's patch set, by name in that set's order,
    /// the largest `|J_q|` over this call's rounds.
    pub reseated: Vec<(String, usize)>,
    /// Per input point, the (92.66) pull in force when the shrink returned -
    /// the caller's `betas` where no rung lowered it.
    pub beta: Vec<Scalar>,
    /// The `J` of the round the shrink accepted; empty when no round was
    /// accepted.
    pub reseat_points: Vec<usize>,
}

/// Which ladder of (92.47) a trace entry comes from: the inner one measures
/// the SHRUNK mesh, the outer one the EXTRUDED mesh (SPEC-LIT §92.13).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Ladder {
    Inner,
    Outer,
}

impl Ladder {
    /// The name the summary prints.
    pub fn as_str(self) -> &'static str {
        match self {
            Ladder::Inner => "inner",
            Ladder::Outer => "outer",
        }
    }
}

/// What one measurement of a ladder led to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Outcome {
    /// The gate passed, and on the inner ladder the floor held.
    Pass,
    /// The gate failed and the offending thickness was halved.
    Retreat,
    /// A rung of (92.66): the gate failed on cells carrying a re-seated
    /// point, whose pull was lowered; no thickness was halved.
    Beta,
    /// A (92.73) step in face mode: the OUTER ladder put faces in M and X -
    /// no point moved; the INNER ladder anchored failing layer points, their
    /// `D_i` set to zero.
    Terminate,
    /// The patch set gave up: one patch lost its layers.
    GiveUp,
}

impl Outcome {
    /// The name the summary prints.
    pub fn as_str(self) -> &'static str {
        match self {
            Outcome::Pass => "pass",
            Outcome::Retreat => "retreat",
            Outcome::Beta => "beta",
            Outcome::Terminate => "terminate",
            Outcome::GiveUp => "give_up",
        }
    }
}

/// What drove a patch's give-up in (92.47), whichever ladder took it: a
/// drop names the gate that caused it, however the two ladders nest.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DropCause {
    /// The inner ladder: the gate still failed on the shrunk mesh after the
    /// last retreat, or failed on cells no layer point reaches, or its own
    /// halvings took a point under the floor.
    InnerGate,
    /// The outer ladder: the gate still failed on the extruded mesh after
    /// the last retreat, or failed on cells no layer point reaches.
    OuterGate,
    /// The floor `min_thickness * T` at a point the outer ladder had capped:
    /// gate failures on the extruded mesh took it there.
    ThinAfterCaps,
    /// The floor at a point neither ladder had thinned: the thickness the
    /// field proposed, after its limiters, was under it.
    ThinProposed,
    /// A layer point whose applied displacement is zero.
    ZeroDisp,
    /// (92.73): every layer face of the patch was terminated - each of its
    /// points was anchored.
    Terminated,
}

impl DropCause {
    /// The name the summary prints.
    pub fn as_str(self) -> &'static str {
        match self {
            DropCause::InnerGate => "inner_gate",
            DropCause::OuterGate => "outer_gate",
            DropCause::ThinAfterCaps => "thin_after_caps",
            DropCause::ThinProposed => "thin_proposed",
            DropCause::ZeroDisp => "zero_disp",
            DropCause::Terminated => "terminated",
        }
    }
}

/// One measurement either ladder of (92.47) took, in the order taken. Read
/// off the ladders; it moves nothing, so no mesh depends on it.
#[derive(Debug, Clone, PartialEq)]
pub struct LadderEntry {
    pub ladder: Ladder,
    /// The outer round - one extrusion attempt of [`add_layers`] - counted
    /// from 0; an inner entry carries the round that ran it.
    pub round: usize,
    /// The halvings this ladder had taken on this patch set before the
    /// measurement.
    pub rung: usize,
    /// The (92.66) rungs this ladder had taken on this patch set before the
    /// measurement.
    pub beta_rung: usize,
    /// The patch set the measurement ran on, by name.
    pub patches: Vec<String>,
    /// Each failing gate of §92.3 with its `n_failed`, in the report's order.
    pub gates: Vec<(Gate, usize)>,
    /// Of the G4 subjects, the faces between an input cell and a layer cell:
    /// the level-n faces of (92.48). Always 0 on the inner ladder, whose mesh
    /// has no layer cell.
    pub g4_level_n: usize,
    pub outcome: Outcome,
    /// On a give-up, what drove it.
    pub give_up: Option<DropCause>,
    /// On a give-up, the patch that lost its layers.
    pub dropped: Option<String>,
    /// On a `Beta` entry, how many re-seated points it lowered; 0 on every
    /// other entry.
    pub beta_points: usize,
    /// (92.73): on a `Terminate` entry, how many points the step anchored or
    /// froze - a freeze's points, layer or interior, included; 0 on every
    /// other entry.
    pub terminate_points: usize,
}

impl LadderEntry {
    /// An entry of round 0 with no level-n count and no dropped patch; the
    /// ladders fill those in where they know them.
    pub fn new(
        ladder: Ladder,
        rung: usize,
        patches: &[String],
        gates: &[(Gate, usize)],
        outcome: Outcome,
        give_up: Option<DropCause>,
    ) -> Self {
        LadderEntry {
            ladder,
            round: 0,
            rung,
            beta_rung: 0,
            patches: patches.to_vec(),
            gates: gates.to_vec(),
            g4_level_n: 0,
            outcome,
            give_up,
            dropped: None,
            beta_points: 0,
            terminate_points: 0,
        }
    }
}

/// The short name of a gate - `G4` of `G4 (non-orthogonality)`.
pub fn gate_label(g: Gate) -> &'static str {
    g.name().split(' ').next().unwrap_or("")
}

/// Each failing gate of `rep` with its `n_failed`, in the report's order.
fn failing_gates(rep: &quality::QualityReport) -> Vec<(Gate, usize)> {
    rep.failures.iter().map(|g| (g.gate, g.n_failed)).collect()
}

/// How many of `rep`'s G4 subjects are level-n faces of `mesh`: internal
/// faces whose owner is an input cell and whose neighbour is a layer cell,
/// the layer cells being the ids from `first_cell` on (92.48).
fn level_n_g4(rep: &quality::QualityReport, mesh: &PolyMeshRaw, first_cell: usize) -> usize {
    let n_internal = mesh.neighbour.len().min(mesh.faces.len());
    let mut n = 0usize;
    for g in &rep.failures {
        if !matches!(g.gate, Gate::NonOrth) {
            continue;
        }
        for s in &g.subjects {
            let f = s.id;
            if f < n_internal
                && (mesh.owner[f] as usize) < first_cell
                && (mesh.neighbour[f] as usize) >= first_cell
            {
                n += 1;
            }
        }
    }
    n
}

/// Per patch of `mesh`, in order: the largest G4 angle over the patch's
/// level-n faces (92.48) on `out` - `None` where the patch carries no such
/// face, all `None` where no layer cell exists. The angle is quality's G4
/// expression on the host mesh of `out`, read owner -> neighbour.
fn level_n_non_orth_max(
    mesh: &PolyMeshRaw,
    out: &PolyMeshRaw,
    ex: &Extrusion,
) -> Result<Vec<Option<Scalar>>> {
    let mut res = vec![None; mesh.patches.len()];
    if ex.n == 0 || ex.layer_faces.is_empty() {
        return Ok(res);
    }
    let m = build_host_mesh(out)?;
    let n_internal_in = mesh.neighbour.len().min(mesh.faces.len());
    for f in 0..m.n_internal_faces {
        let (o, nb) = (m.owner[f] as usize, m.neighbour[f] as usize);
        if !(o < ex.first_cell && nb >= ex.first_cell) {
            continue;
        }
        let d = m.c[nb] - m.c[o];
        let (mag_s, mag_d) = (m.mag_sf[f], d.mag());
        if !(mag_s > 0.0) || !(mag_d > 0.0) {
            continue;
        }
        let theta = (m.sf[f].dot(d) / (mag_s * mag_d))
            .clamp(-1.0, 1.0)
            .acos()
            .to_degrees();
        // (92.72): the cell's own block, KEEP or RING - a RING cell's block
        // is one cell wide, so the /n of the patch mode cannot serve.
        let Some(jf) = ex.layer_face_of_cell(nb) else {
            continue;
        };
        let fa = ex.layer_faces[jf];
        if fa < n_internal_in {
            continue;
        }
        let rel = fa - n_internal_in;
        if let Some(k) = mesh
            .patches
            .iter()
            .position(|p| rel >= p.start && rel < p.start + p.size)
        {
            res[k] = Some(res[k].map_or(theta, |v: Scalar| v.max(theta)));
        }
    }
    Ok(res)
}

/// (92.63)'s `THETA_ON`, in degrees: a wall face whose predicted level-n
/// angle exceeds it has its row-1 cell re-seated by the shrink. This
/// project's own number, like `KAPPA` and `MARCH_C` of (92.44); on a wall
/// on the cell planes the angle is 0, so nothing is re-seated.
pub const RESEAT_THETA_ON_DEG: Scalar = 45.0;

/// (92.64): the interior points of row-1 cells the shrink holds at (92.65)'s
/// displacement, for ONE patch set.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Reseat {
    /// J, ascending.
    pub points: Vec<usize>,
    /// W(j) for each point of `points`, in the same order, ascending.
    pub wall: Vec<Vec<usize>>,
    /// `|J_q|` for each entry of the field's `patches`, in that order.
    pub per_patch: Vec<usize>,
}

/// (92.63): one predicted level-n angle per face of `mesh` - G4's own
/// expression, read on the INPUT mesh, where a wall face is still a
/// boundary face. Internal faces carry 0.0.
pub fn predicted_angles(mesh: &PolyMeshRaw) -> Result<Vec<Scalar>> {
    let m = build_host_mesh(mesh)?;
    let n_faces = mesh.faces.len();
    let n_internal = m.n_internal_faces;
    let mut theta = vec![0.0 as Scalar; n_faces];
    for fa in n_internal..n_faces {
        let bf = fa - n_internal;
        let s = m.b_sf[bf];
        let d = m.b_cf[bf] - m.c[m.b_face_cells[bf] as usize];
        let (mag_s, mag_d) = (s.mag(), d.mag());
        if !(mag_s > 0.0) || !(mag_d > 0.0) {
            continue;
        }
        theta[fa] = (s.dot(d) / (mag_s * mag_d))
            .clamp(-1.0, 1.0)
            .acos()
            .to_degrees();
    }
    Ok(theta)
}

/// (92.64): R1*, W and J, from the angles, the field, the shrink's own
/// boundary flags and the per-cell face lists. Pure: reads only its
/// arguments.
fn reseat_set(
    mesh: &PolyMeshRaw,
    theta: &[Scalar],
    f: &Field,
    is_b: &[bool],
    cell_faces: &[Vec<usize>],
) -> Reseat {
    let n_points = mesh.points.len();
    // R1*: the owner cells of the triggering layer faces.
    let mut r1 = vec![false; cell_faces.len()];
    for &fa in &f.faces {
        if theta[fa] > RESEAT_THETA_ON_DEG {
            r1[mesh.owner[fa] as usize] = true;
        }
    }
    // W: for every face of an R1* cell, every cyclic edge, each ordering.
    let mut w: Vec<Vec<usize>> = vec![Vec::new(); n_points];
    for (c, &bad) in r1.iter().enumerate() {
        if !bad {
            continue;
        }
        for &fa in &cell_faces[c] {
            let face = &mesh.faces[fa];
            for k in 0..face.len() {
                let a = face[k] as usize;
                let b = face[(k + 1) % face.len()] as usize;
                for (i, j) in [(a, b), (b, a)] {
                    if f.is_layer[i]
                        && !f.pinned[i]
                        && !f.is_layer[j]
                        && !is_b[j]
                        && !w[j].contains(&i)
                    {
                        w[j].push(i);
                    }
                }
            }
        }
    }
    for list in w.iter_mut() {
        list.sort_unstable();
    }
    let points: Vec<usize> = (0..n_points).filter(|&j| !w[j].is_empty()).collect();
    build_reseat(f, mesh, theta, &w, &points, cell_faces)
}

/// `|J_q|` per patch of the field, and the `Reseat` itself: the J points
/// lying on a face of the owner cell of some triggering face of `q`.
fn build_reseat(
    f: &Field,
    mesh: &PolyMeshRaw,
    theta: &[Scalar],
    w: &[Vec<usize>],
    points: &[usize],
    cell_faces: &[Vec<usize>],
) -> Reseat {
    let mut per_patch = Vec::with_capacity(f.patches.len());
    let mut on_q = vec![false; mesh.points.len()];
    for &q in &f.patches {
        for flag in on_q.iter_mut() {
            *flag = false;
        }
        for (k, &fa) in f.faces.iter().enumerate() {
            if f.face_patch[k] != q || theta[fa] <= RESEAT_THETA_ON_DEG {
                continue;
            }
            for &fc in &cell_faces[mesh.owner[fa] as usize] {
                for &p in &mesh.faces[fc] {
                    on_q[p as usize] = true;
                }
            }
        }
        per_patch.push(points.iter().filter(|&&j| on_q[j]).count());
    }
    Reseat {
        points: points.to_vec(),
        wall: w.iter().filter(|l| !l.is_empty()).cloned().collect(),
        per_patch,
    }
}

/// (92.65) for ONE point: `xj` the point itself, `wall` its
/// `(x_i, D_i, n_i)` in order, `beta` the pull toward the wall-following
/// position. Zero for an empty `wall`.
pub fn reseat_disp(xj: Vec3, wall: &[(Vec3, Vec3, Vec3)], beta: Scalar) -> Vec3 {
    if wall.is_empty() {
        return Vec3::ZERO;
    }
    let mut md = Vec3::ZERO;
    let mut y = Vec3::ZERO;
    for &(xi, di, ni) in wall {
        md = md + di;
        y = y + xi + di + (xj - xi).mag() * ni;
    }
    let over = wall.len() as Scalar;
    let md = md / over;
    let y = y / over;
    md + (y - xj - md) * beta
}

/// (92.66): the beta rungs one ladder may take on one patch set - the steps
/// 1 -> 1/2 -> 1/4 -> 0.
pub const BETA_RUNG_LIMIT: usize = 3;

/// (92.66)'s step: `b / 2` while `b > 1/4`, else 0 - so 1, 1/2, 1/4, 0, and
/// 0 stays 0.
pub fn beta_step(b: Scalar) -> Scalar {
    if b > 0.25 {
        b / 2.0
    } else {
        0.0
    }
}

/// What took the points under the floor: the outer ladder's cap where any
/// of them carries one, else the inner ladder's halving where any of them
/// took one, else the proposed thickness itself.
fn thin_cause(offenders: &[usize], caps: &[Scalar], halved: &[bool]) -> DropCause {
    if offenders.iter().any(|&i| caps[i] < 1.0) {
        DropCause::ThinAfterCaps
    } else if offenders.iter().any(|&i| halved[i]) {
        DropCause::InnerGate
    } else {
        DropCause::ThinProposed
    }
}

/// (92.73): `F` closed under "a hanging node in `F` adds its parents that
/// are layer points", recursively - [`find_hanging`]'s map, walked until no
/// parent is missing.
fn close_under_hanging(
    fset: &mut Vec<usize>,
    hanging: &[(u32, [u32; 2])],
    is_layer: &[bool],
) {
    // The hanging node -> parents map, built once per call: the closure
    // looks parents up instead of scanning the whole hanging list per
    // queued point.
    let mut parents: HashMap<usize, [u32; 2]> = HashMap::with_capacity(hanging.len());
    for &(h, ab) in hanging {
        parents.insert(h as usize, ab);
    }
    let mut in_set = vec![false; is_layer.len()];
    for &i in fset.iter() {
        in_set[i] = true;
    }
    let mut queue: Vec<usize> = fset.clone();
    while let Some(i) = queue.pop() {
        if let Some(ab) = parents.get(&i) {
            for &p in ab {
                let p = p as usize;
                if is_layer[p] && !in_set[p] {
                    in_set[p] = true;
                    fset.push(p);
                    queue.push(p);
                }
            }
        }
    }
    fset.sort_unstable();
    fset.dedup();
}

/// (92.73) amended: the freeze's consecutive-failure counts move only at a
/// failed measurement that FOLLOWS a local step - a retreat or terminate
/// step of this ladder. A measurement after a (92.66) beta rung, and the
/// round's first measurement before any step, neither increments nor resets
/// a count: a rung moves the pull, not the points, so the cells it fails
/// are the cells the step after it will have to answer for, not failures a
/// freeze may hold them for.
fn count_cell_failures(counts: &mut [usize], flags: &[bool], post_step: bool) {
    if !post_step {
        return;
    }
    for (c, &bad) in flags.iter().enumerate() {
        if bad {
            counts[c] += 1;
        } else {
            counts[c] = 0;
        }
    }
}

/// (92.73) amended: the frozen point set - the cells' points, closed under
/// "a hanging node in the set adds BOTH its parents", recursively, layer or
/// not: a frozen cell is held whole, interior points included, so (92.46)'s
/// relaxation cannot move it from its neighbours' wall points. Sorted, once.
fn frozen_point_set(
    cells: &[usize],
    cell_points: &[Vec<u32>],
    hanging: &[(u32, [u32; 2])],
) -> Vec<usize> {
    let mut parents: HashMap<usize, [u32; 2]> = HashMap::with_capacity(hanging.len());
    for &(h, ab) in hanging {
        parents.insert(h as usize, ab);
    }
    let mut in_set: HashSet<usize> = HashSet::new();
    let mut queue: Vec<usize> = Vec::new();
    for &c in cells {
        for &p in &cell_points[c] {
            if in_set.insert(p as usize) {
                queue.push(p as usize);
            }
        }
    }
    while let Some(i) = queue.pop() {
        if let Some(ab) = parents.get(&i) {
            for &p in ab {
                if in_set.insert(p as usize) {
                    queue.push(p as usize);
                }
            }
        }
    }
    let mut out: Vec<usize> = in_set.into_iter().collect();
    out.sort_unstable();
    out
}

/// (92.70)'s side faces carry a vertex ring that can close a triangle into
/// a line when one endpoint is anchored: consecutive repeats go first,
/// then the ring's own wrap-around repeat.
fn dedupe_cyclic(mut ps: Vec<crate::Label>) -> Vec<crate::Label> {
    ps.dedup();
    while ps.len() > 1 && ps[0] == ps[ps.len() - 1] {
        ps.pop();
    }
    ps
}

/// [`shrink`]'s body at a CALLER'S patch set and per-point retreat cap: the
/// ladder runs over `patches0` alone, and every layer point's proposed
/// displacement and thickness come in already scaled by `caps[i]`.
///
/// `betas` is the per-point pull (92.66) the caller carries in - one entry
/// per input point, the re-seat points' starting `beta_j`.
fn shrink_on(
    mesh: &PolyMeshRaw,
    surf: &Surface,
    spec: &LayerSpec,
    t: &QualityThresholds,
    patches0: &[usize],
    caps: &[Scalar],
    betas: &[Scalar],
) -> Result<Shrunk> {
    let n_faces = mesh.faces.len();
    let n_internal = mesh.neighbour.len().min(n_faces);
    let n_points = mesh.points.len();
    if spec.n == 0 || spec.patches.is_empty() {
        return Ok(Shrunk {
            mesh: mesh.clone(),
            field: empty_field(n_points),
            retreats: 0,
            dropped: Vec::new(),
            drop_causes: Vec::new(),
            ladder: Vec::new(),
            reseated: Vec::new(),
            beta: vec![1.0; n_points],
            reseat_points: Vec::new(),
        });
    }
    let st = stack(spec)?;
    // The two gates moving points cannot mend, refused at once on the
    // input, as stage 4 refuses them.
    let arrival = quality::measure_capped(mesh, t, quality::GATE_CELL_CAP)?;
    if arrival
        .failures
        .iter()
        .any(|f| matches!(f.gate, Gate::Regions | Gate::Addressing))
    {
        return Err(Error::Mesh(arrival.refusal_text()));
    }
    // The index is built ONCE: the hint is the mean layer-face edge length,
    // or a hundredth of the surface's diagonal when no face is a layer face.
    let mut is_layer_face = vec![false; n_faces];
    for &p in patches0 {
        let patch = &mesh.patches[p];
        for j in 0..patch.size {
            let f = n_internal + patch.start + j;
            if f < n_faces {
                is_layer_face[f] = true;
            }
        }
    }
    let mut sum = 0.0;
    let mut cnt = 0usize;
    for (f, &is) in is_layer_face.iter().enumerate() {
        if !is {
            continue;
        }
        let face = &mesh.faces[f];
        for k in 0..face.len() {
            let a = mesh.points[face[k] as usize];
            let b = mesh.points[face[(k + 1) % face.len()] as usize];
            sum += (a - b).mag();
            cnt += 1;
        }
    }
    let hint = if cnt > 0 {
        sum / (cnt as Scalar)
    } else {
        let (lo, hi) = surf.bbox;
        (hi - lo).mag() / 100.0
    };
    let idx = TriIndex::with_bvh(surf, hint)?;
    // The three graphs the ladder needs, built once: the whole point graph,
    // the boundary flag, and each cell's points.
    let mut all_nbrs: Vec<Vec<u32>> = vec![Vec::new(); n_points];
    let mut is_b = vec![false; n_points];
    for face in mesh.faces.iter() {
        for (k, &p) in face.iter().enumerate() {
            let i = p as usize;
            let nxt = face[(k + 1) % face.len()] as usize;
            all_nbrs[i].push(nxt as u32);
            all_nbrs[nxt].push(i as u32);
        }
    }
    for list in all_nbrs.iter_mut() {
        list.sort_unstable();
        list.dedup();
    }
    for f in n_internal..n_faces {
        for &p in &mesh.faces[f] {
            is_b[p as usize] = true;
        }
    }
    let n_cells = mesh
        .owner
        .iter()
        .chain(mesh.neighbour.iter())
        .copied()
        .max()
        .map_or(0, |m| m as usize + 1);
    let mut cell_points: Vec<Vec<u32>> = vec![Vec::new(); n_cells];
    for (f, face) in mesh.faces.iter().enumerate() {
        for &p in face {
            cell_points[mesh.owner[f] as usize].push(p as u32);
        }
        if f < n_internal {
            for &p in face {
                cell_points[mesh.neighbour[f] as usize].push(p as u32);
            }
        }
    }
    for list in cell_points.iter_mut() {
        list.sort_unstable();
        list.dedup();
    }
    let hanging = find_hanging(&mesh.points, &mesh.faces);
    // The re-seat plan's inputs, built ONCE: the predicted level-n angle of
    // every face, and each cell's faces ascending by id - `owner` runs in
    // face order, the internal faces add their neighbours in the same
    // order.
    let theta = predicted_angles(mesh)?;
    let mut cell_faces: Vec<Vec<usize>> = vec![Vec::new(); n_cells];
    for (fa, &c) in mesh.owner.iter().enumerate() {
        cell_faces[c as usize].push(fa);
    }
    for fa in 0..n_internal {
        cell_faces[mesh.neighbour[fa] as usize].push(fa);
    }
    // The outer patch loop: (92.40) through (92.47) for one patch set at a
    // time, dropping ONE patch per round - the one carrying the most
    // offending layer points, ties to the lower patch index - and
    // recomputing from the input mesh, because a point the dropped patch
    // shared is now constrained into that patch's plane and its normal is a
    // different vector. At most `all_patches.len()` rounds.
    let mut patches = patches0.to_vec();
    let mut dropped: Vec<(String, String)> = Vec::new();
    let mut retreats = 0usize;
    let mut drop_causes: Vec<(String, DropCause)> = Vec::new();
    let mut ladder: Vec<LadderEntry> = Vec::new();
    // (92.64)'s per-patch counts, one counter per entry of the caller's
    // patch set, the max taken over the rounds each patch ran in.
    let mut reseated_max = vec![0usize; patches0.len()];
    loop {
        if patches.is_empty() {
            let reseated = named_reseated(mesh, patches0, &reseated_max);
            return Ok(Shrunk {
                mesh: mesh.clone(),
                field: empty_field(n_points),
                retreats,
                dropped,
                drop_causes,
                ladder,
                reseated,
                beta: betas.to_vec(),
                reseat_points: Vec::new(),
            });
        }
        let mut f = field(mesh, &idx, &patches, spec, &st)?;
        // The trace of this round, read off the ladder and moving nothing
        // (SPEC-LIT §92.13): the patch set by name, the points the ladder
        // halved, and what drove the give-up if there is one.
        let names: Vec<String> = patches.iter().map(|&p| mesh.patches[p].name.clone()).collect();
        let mut halved = vec![false; n_points];
        let mut cause: Option<DropCause> = None;
        // (92.47)'s retreat, carried in from the EXTRUSION's own ladder: the
        // caller's cap scales what this round proposes, point by point.
        for i in 0..f.is_layer.len() {
            if f.is_layer[i] {
                f.disp[i] = f.disp[i] * caps[i];
                f.thickness[i] = f.thickness[i] * caps[i];
            }
        }
        // (92.64): the row-1 interior points this round holds, and the
        // per-patch counts merged into the run's counters.
        let rs = reseat_set(mesh, &theta, &f, &is_b, &cell_faces);
        for (k, &p) in patches.iter().enumerate() {
            let pos = patches0
                .iter()
                .position(|&q| q == p)
                .expect("a round's patch is one of the caller's");
            reseated_max[pos] = reseated_max[pos].max(rs.per_patch[k]);
        }
        // The retreat ladder: halve D at the layer points the failing cells
        // carry, re-run the relaxation, and try again, up to
        // `retreat_limit` times.
        let mut accepted = false;
        let mut pts_out = mesh.points.clone();
        let mut give_up: Option<(Vec<usize>, String)> = None;
        let mut halvings = 0usize;
        // (92.66): this round's pull starts from the caller's betas, and the
        // beta rungs count from 0 on every patch set.
        let mut beta = betas.to_vec();
        let mut beta_rungs = 0usize;
        // (92.73): face mode's per-point halving counts, its step count on
        // this patch set, and the floor - all no-ops in patch mode. The
        // anchors are the points the CALLER'S ladder already zeroed (c_i =
        // 0) plus what this round's pass path zeroes; the re-seat plan and
        // the hanging mean resurrect neither.
        let face_mode = spec.terminate == LayerTerminate::Face;
        let mut hcount = vec![0usize; n_points];
        let mut anchored: Vec<bool> = caps.iter().map(|c| *c == 0.0).collect();
        let mut steps = 0usize;
        // (92.73) amended: the freeze's two books, face mode's own - per
        // input cell, the consecutive measurements that named it, and per
        // point, whether a freeze already holds it. Untouched in patch mode.
        let mut cell_fails = vec![0usize; cell_points.len()];
        let mut frozen = vec![false; n_points];
        // (92.73) amended: did the measurement in front of the ladder FOLLOW
        // a local step of this ladder - a retreat or terminate step? A beta
        // rung's measurement, and the round's first, follow nothing, and
        // they leave the freeze's counts as they found them.
        let mut post_step = false;
        let limit = spec.min_thickness * st.total;
        loop {
            let d = relax(&f, &all_nbrs, &is_b, &hanging, spec, &rs, &mesh.points, &beta, &anchored);
            let mut work = mesh.clone();
            for i in 0..work.points.len() {
                work.points[i] = work.points[i] + d[i];
            }
            let rep = quality::measure_capped(&work, t, usize::MAX)?;
            let gates = failing_gates(&rep);
            if rep.passed() && face_mode {
                // (92.73) amended: a passed measurement is no cell failing -
                // the consecutive-failure counts start over - but like the
                // failure counts, only a measurement that follows a local
                // step touches them.
                if post_step {
                    for n in cell_fails.iter_mut() {
                        *n = 0;
                    }
                }
                // (92.73), at a pass: a point the relaxation left at zero is
                // ANCHORED, never a give-up - the faces around it taper in
                // the extrusion. A point under the floor is anchored too,
                // and the ladder measures again; with none under it, accept.
                let thin: Vec<usize> = (0..n_points)
                    .filter(|&i| {
                        f.is_layer[i] && d[i].mag_sqr() > 0.0 && d[i].mag() < limit
                    })
                    .collect();
                if !thin.is_empty() {
                    // (92.73): the anchored set is CLOSED under hanging
                    // parents first, so an anchored hanging node's forced
                    // zero is the mean of anchored (or non-layer, zero)
                    // parents and (92.46)'s hanging line holds for (92.49)'s
                    // split sides. The step counts toward the step limit,
                    // and one that changes no D_i ends the ladder by
                    // (92.47)'s patch rule.
                    if steps >= TERMINATE_STEP_LIMIT {
                        cause = Some(DropCause::InnerGate);
                        let mut e = LadderEntry::new(
                            Ladder::Inner, halvings, &names, &gates, Outcome::GiveUp, cause,
                        );
                        e.beta_rung = beta_rungs;
                        ladder.push(e);
                        give_up = Some((thin, format!(
                            "the pass still left thin points after {steps} terminate step(s)"
                        )));
                        break;
                    }
                    let mut fset = thin.clone();
                    close_under_hanging(&mut fset, &hanging, &f.is_layer);
                    let mut changed = false;
                    for &i in &fset {
                        if f.disp[i].mag_sqr() > 0.0 {
                            changed = true;
                        }
                        f.disp[i] = Vec3::ZERO;
                        f.thickness[i] = 0.0;
                        anchored[i] = true;
                    }
                    if !changed {
                        cause = Some(DropCause::InnerGate);
                        let mut e = LadderEntry::new(
                            Ladder::Inner, halvings, &names, &gates, Outcome::GiveUp, cause,
                        );
                        e.beta_rung = beta_rungs;
                        ladder.push(e);
                        give_up = Some((
                            fset,
                            "the thin step changed no layer point".to_string(),
                        ));
                        break;
                    }
                    let mut e = LadderEntry::new(
                        Ladder::Inner, halvings, &names, &gates, Outcome::Terminate, None,
                    );
                    e.beta_rung = beta_rungs;
                    e.terminate_points = fset.len();
                    ladder.push(e);
                    steps += 1;
                    post_step = true;
                    continue;
                }
                let mut e = LadderEntry::new(
                    Ladder::Inner, halvings, &names, &gates, Outcome::Pass, None,
                );
                e.beta_rung = beta_rungs;
                ladder.push(e);
                accepted = true;
                pts_out = work.points;
                for i in 0..n_points {
                    if f.is_layer[i] {
                        f.disp[i] = d[i];
                        f.thickness[i] = d[i].mag();
                    }
                }
                break;
            }
            if rep.passed() {
                // Written by the supervising session: the thickness a point
                // CARRIES is `|d_i|` after the relaxation's hanging line and
                // after the retreats, not the `T_i` (92.45) proposed, so both
                // the floor and the report read `d`. A hanging node on a
                // layer face whose PARENTS lie on a coarse face of a
                // non-layer patch takes the mean of two zeros, and a layer
                // point with `|d_i| = 0` would extrude a side face of zero
                // area - so that check is unconditional, whatever
                // `min_thickness` is set to (SPEC-LIT §92.13, (92.46)).
                let zero: Vec<usize> = (0..n_points)
                    .filter(|&i| f.is_layer[i] && !(d[i].mag() > 0.0))
                    .collect();
                let thin: Vec<usize> = (0..n_points)
                    .filter(|&i| f.is_layer[i] && d[i].mag() < limit)
                    .collect();
                if !zero.is_empty() {
                    cause = Some(DropCause::ZeroDisp);
                    let mut e = LadderEntry::new(
                        Ladder::Inner, halvings, &names, &gates, Outcome::GiveUp, cause,
                    );
                    e.beta_rung = beta_rungs;
                    ladder.push(e);
                    give_up = Some((
                        zero,
                        "the applied displacement is zero at a layer point -                          a layer cell there would have a side face of zero area"
                            .to_string(),
                    ));
                } else if !thin.is_empty() {
                    cause = Some(thin_cause(&thin, caps, &halved));
                    let mut e = LadderEntry::new(
                        Ladder::Inner, halvings, &names, &gates, Outcome::GiveUp, cause,
                    );
                    e.beta_rung = beta_rungs;
                    ladder.push(e);
                    give_up = Some((thin, format!(
                        "the thickness fell below min_thickness * T = {limit:.3e}"
                    )));
                } else {
                    let mut e = LadderEntry::new(
                        Ladder::Inner, halvings, &names, &gates, Outcome::Pass, None,
                    );
                    e.beta_rung = beta_rungs;
                    ladder.push(e);
                    accepted = true;
                    pts_out = work.points;
                    for i in 0..n_points {
                        if f.is_layer[i] {
                            f.disp[i] = d[i];
                            f.thickness[i] = d[i].mag();
                        }
                    }
                }
                break;
            }
            // (92.73) amended: the consecutive-failure counts update at a
            // failed measurement that FOLLOWS a local step of this ladder -
            // a retreat or terminate step - and a cell not named by it
            // starts over. A measurement after a (92.66) beta rung, and the
            // round's first measurement, touch nothing: the freeze would
            // otherwise take every cell the rungs had named before the
            // first step could answer them.
            let flags = failing_cell_flags(&rep, mesh, n_internal, cell_points.len());
            if face_mode {
                count_cell_failures(&mut cell_fails, &flags, post_step);
            }
            // (92.66): before any give-up check or halving, a failure whose
            // failing cells carry a live re-seat point takes the pull back
            // there - never a halving of `D`, never a retreat.
            let mut live = vec![false; n_points];
            for &j in &rs.points {
                live[j] = beta[j] > 0.0;
            }
            let jf = failing_points(&rep, mesh, n_internal, &live, &cell_points);
            if !jf.is_empty() && beta_rungs < BETA_RUNG_LIMIT {
                for &j in &jf {
                    beta[j] = beta_step(beta[j]);
                }
                let mut e = LadderEntry::new(
                    Ladder::Inner, halvings, &names, &gates, Outcome::Beta, None,
                );
                e.beta_rung = beta_rungs;
                e.beta_points = jf.len();
                ladder.push(e);
                beta_rungs += 1;
                // A rung is no step: the measurement after it follows
                // nothing, and the freeze's counts hold still for it.
                post_step = false;
                continue;
            }
            let fail_pts = failing_points(&rep, mesh, n_internal, &f.is_layer, &cell_points);
            if face_mode {
                // (92.73) amended, at a failure: the cells the freeze takes
                // are held whole at d = 0, the other failing cells' MOVING
                // layer points take their local step as before, and (92.47)'s
                // patch drop waits for a step that freezes, halves or anchors
                // nothing, or the step limit.
                let mut freeze_cells: Vec<usize> = Vec::new();
                for (c, &bad) in flags.iter().enumerate() {
                    if !bad {
                        continue;
                    }
                    // A cell whose layer points all stand at D = 0 is frozen
                    // at its first failure: no step on ITS points can mend
                    // it, (92.46) moves it from its neighbours' wall points.
                    // Otherwise two consecutive failures do.
                    let all_still = cell_points[c].iter().all(|&p| {
                        let i = p as usize;
                        !f.is_layer[i] || f.disp[i].mag_sqr() == 0.0
                    });
                    if all_still || cell_fails[c] >= 2 {
                        freeze_cells.push(c);
                    }
                }
                // The frozen points, closed under hanging parents - ALL
                // parents, layer or not - minus the points an earlier freeze
                // already holds: a cell is frozen at most once.
                let mut frozen_new =
                    frozen_point_set(&freeze_cells, &cell_points, &hanging);
                frozen_new.retain(|&i| !frozen[i]);
                let mut fset: Vec<usize> = fail_pts
                    .iter()
                    .copied()
                    .filter(|&i| f.disp[i].mag_sqr() > 0.0 && !frozen[i])
                    .collect();
                close_under_hanging(&mut fset, &hanging, &f.is_layer);
                fset.retain(|&i| !frozen[i]);
                // `in_g5`: the point is a point of a cell G5 names - halving
                // `t` halves `V`, so no halving can mend a G5 failure.
                let mut g5_pt = vec![false; n_points];
                for failure in &rep.failures {
                    if !matches!(failure.gate, Gate::Thickness) {
                        continue;
                    }
                    for s in &failure.subjects {
                        if (s.id as usize) < cell_points.len() {
                            for &p in &cell_points[s.id as usize] {
                                g5_pt[p as usize] = true;
                            }
                        }
                    }
                }
                if fset.is_empty() {
                    // (92.73)'s F empty: first every re-seat point of a
                    // failing cell takes its pull to ZERO at once - no
                    // BETA_RUNG_LIMIT here - and the ladder measures again.
                    if !jf.is_empty() {
                        for &j in &jf {
                            beta[j] = 0.0;
                        }
                        let mut e = LadderEntry::new(
                            Ladder::Inner, halvings, &names, &gates, Outcome::Beta, None,
                        );
                        e.beta_rung = beta_rungs;
                        e.beta_points = jf.len();
                        ladder.push(e);
                        // A rung is no step, the pull-to-zero either.
                        post_step = false;
                        continue;
                    }
                    // With no such point the freeze answers in run 5's place:
                    // F's cells all stand at D = 0, so the freeze above takes
                    // them whole - run 2 widened F to one ring, and that is
                    // gone. When the freeze holds nothing new either,
                    // (92.47)'s patch rule answers.
                    if frozen_new.is_empty() {
                        cause = Some(DropCause::InnerGate);
                        let mut e = LadderEntry::new(
                            Ladder::Inner, halvings, &names, &gates, Outcome::GiveUp, cause,
                        );
                        e.beta_rung = beta_rungs;
                        ladder.push(e);
                        give_up = Some((
                            fail_pts,
                            "the gate failed on cells no moving layer point reaches".to_string(),
                        ));
                        break;
                    }
                }
                if steps >= TERMINATE_STEP_LIMIT {
                    cause = Some(DropCause::InnerGate);
                    let mut e = LadderEntry::new(
                        Ladder::Inner, halvings, &names, &gates, Outcome::GiveUp, cause,
                    );
                    e.beta_rung = beta_rungs;
                    ladder.push(e);
                    give_up = Some((fail_pts, format!(
                        "the gate still failed after {steps} terminate step(s)"
                    )));
                    break;
                }
                let mut any_halved = false;
                let mut changed = false;
                // The freeze first: every point of a frozen cell, layer and
                // interior, held at d = 0 from now on - relax holds it as it
                // holds an anchored point, and (92.70) reads a frozen layer
                // point as anchored because its D_i is 0. Freezing a point is
                // the step's progress whatever it stood at.
                let mut held = 0usize;
                for &i in &frozen_new {
                    changed = true;
                    held += 1;
                    if f.is_layer[i] {
                        f.disp[i] = Vec3::ZERO;
                        f.thickness[i] = 0.0;
                    }
                    anchored[i] = true;
                    frozen[i] = true;
                }
                for &i in &fset {
                    // A point the freeze took this very step - a shared point
                    // of a frozen cell and a stepped one - is held, not
                    // stepped.
                    if frozen[i] {
                        continue;
                    }
                    match local_step(
                        g5_pt[i],
                        hcount[i],
                        f.disp[i].mag(),
                        limit,
                        spec.retreat_limit,
                    ) {
                        LocalStep::Halve => {
                            if f.disp[i].mag_sqr() > 0.0 {
                                changed = true;
                            }
                            f.disp[i] = f.disp[i] * 0.5;
                            f.thickness[i] = f.thickness[i] * 0.5;
                            hcount[i] += 1;
                            halved[i] = true;
                            any_halved = true;
                        }
                        LocalStep::Anchor => {
                            if f.disp[i].mag_sqr() > 0.0 {
                                changed = true;
                            }
                            f.disp[i] = Vec3::ZERO;
                            f.thickness[i] = 0.0;
                            anchored[i] = true;
                            held += 1;
                        }
                    }
                }
                if !changed {
                    cause = Some(DropCause::InnerGate);
                    let mut e = LadderEntry::new(
                        Ladder::Inner, halvings, &names, &gates, Outcome::GiveUp, cause,
                    );
                    e.beta_rung = beta_rungs;
                    ladder.push(e);
                    give_up = Some((
                        fail_pts,
                        "the terminate step changed no layer point".to_string(),
                    ));
                    break;
                }
                if any_halved {
                    let mut e = LadderEntry::new(
                        Ladder::Inner, halvings, &names, &gates, Outcome::Retreat, None,
                    );
                    e.beta_rung = beta_rungs;
                    ladder.push(e);
                    halvings += 1;
                } else {
                    let mut e = LadderEntry::new(
                        Ladder::Inner, halvings, &names, &gates, Outcome::Terminate, None,
                    );
                    e.beta_rung = beta_rungs;
                    e.terminate_points = held;
                    ladder.push(e);
                }
                steps += 1;
                post_step = true;
                continue;
            }
            if halvings >= spec.retreat_limit {
                cause = Some(DropCause::InnerGate);
                let mut e = LadderEntry::new(
                    Ladder::Inner, halvings, &names, &gates, Outcome::GiveUp, cause,
                );
                e.beta_rung = beta_rungs;
                ladder.push(e);
                give_up = Some((fail_pts, format!(
                    "the gate still failed after {halvings} retreat(s)"
                )));
                break;
            }
            if fail_pts.is_empty() {
                cause = Some(DropCause::InnerGate);
                let mut e = LadderEntry::new(
                    Ladder::Inner, halvings, &names, &gates, Outcome::GiveUp, cause,
                );
                e.beta_rung = beta_rungs;
                ladder.push(e);
                give_up = Some((
                    fail_pts,
                    "the gate failed on cells no layer point reaches".to_string(),
                ));
                break;
            }
            let mut e =
                LadderEntry::new(Ladder::Inner, halvings, &names, &gates, Outcome::Retreat, None);
            e.beta_rung = beta_rungs;
            ladder.push(e);
            for &i in &fail_pts {
                halved[i] = true;
                f.disp[i] = f.disp[i] * 0.5;
                f.thickness[i] = f.thickness[i] * 0.5;
            }
            halvings += 1;
        }
        retreats += halvings;
        if accepted {
            let reseated = named_reseated(mesh, patches0, &reseated_max);
            return Ok(Shrunk {
                mesh: PolyMeshRaw {
                    points: pts_out,
                    ..mesh.clone()
                },
                field: f,
                retreats,
                dropped,
                drop_causes,
                ladder,
                reseated,
                beta,
                reseat_points: rs.points.clone(),
            });
        }
        let (offenders, reason) = give_up.expect("the ladder ended in a give-up");
        // Which patch loses its layers: the one carrying the most offending
        // layer points, ties to the lower patch index.
        let mut counts = vec![0usize; patches.len()];
        for (k, &p) in patches.iter().enumerate() {
            let patch = &mesh.patches[p];
            let mut seen = vec![false; n_points];
            for j in 0..patch.size {
                let fa = n_internal + patch.start + j;
                if fa >= n_faces {
                    continue;
                }
                for &pt in &mesh.faces[fa] {
                    seen[pt as usize] = true;
                }
            }
            for &i in &offenders {
                if seen[i] {
                    counts[k] += 1;
                }
            }
        }
        let mut victim = 0usize;
        for k in 1..counts.len() {
            if counts[k] > counts[victim] {
                victim = k;
            }
        }
        let vp = patches[victim];
        let c = cause.expect("a give-up names its cause");
        if let Some(e) = ladder.last_mut() {
            e.dropped = Some(mesh.patches[vp].name.clone());
        }
        drop_causes.push((mesh.patches[vp].name.clone(), c));
        dropped.push((mesh.patches[vp].name.clone(), reason));
        patches.remove(victim);
    }
}

/// The run's re-seat counters, named: per entry of the caller's patch set,
/// in its order.
fn named_reseated(
    mesh: &PolyMeshRaw,
    patches0: &[usize],
    reseated_max: &[usize],
) -> Vec<(String, usize)> {
    patches0
        .iter()
        .zip(reseated_max.iter())
        .map(|(&p, &n)| (mesh.patches[p].name.clone(), n))
        .collect()
}

/// Shrink the boundary inward by the layer thickness and relax the interior
/// behind it, until the gate is satisfied or the patches give their layers
/// up. `spec.n == 0`, or an empty `layers.patches`, returns the input mesh
/// bit for bit.
pub fn shrink(
    mesh: &PolyMeshRaw,
    surf: &Surface,
    spec: &LayerSpec,
    t: &QualityThresholds,
) -> Result<Shrunk> {
    let n_points = mesh.points.len();
    if spec.n == 0 || spec.patches.is_empty() {
        return Ok(Shrunk {
            mesh: mesh.clone(),
            field: empty_field(n_points),
            retreats: 0,
            dropped: Vec::new(),
            drop_causes: Vec::new(),
            ladder: Vec::new(),
            reseated: Vec::new(),
            beta: vec![1.0; n_points],
            reseat_points: Vec::new(),
        });
    }
    let all_patches = resolve_patches(mesh, spec)?;
    shrink_on(
        mesh,
        surf,
        spec,
        t,
        &all_patches,
        &vec![1.0 as Scalar; n_points],
        &vec![1.0 as Scalar; n_points],
    )
}

/// The cells §92.3's failures blame - the failing-cell flags
/// [`failing_points`] reads its points off, kept as a vector the freeze of
/// (92.73)' walks.
fn failing_cell_flags(
    rep: &quality::QualityReport,
    mesh: &PolyMeshRaw,
    n_internal: usize,
    n_cells: usize,
) -> Vec<bool> {
    let mut is_fail = vec![false; n_cells];
    for failure in &rep.failures {
        match failure.gate {
            Gate::Regions | Gate::Addressing => continue,
            Gate::NonOrth => {
                for s in &failure.subjects {
                    let fa = s.id;
                    if fa < mesh.owner.len() {
                        is_fail[mesh.owner[fa] as usize] = true;
                        if fa < n_internal {
                            is_fail[mesh.neighbour[fa] as usize] = true;
                        }
                    }
                }
            }
            _ => {
                for s in &failure.subjects {
                    if s.id < is_fail.len() {
                        is_fail[s.id] = true;
                    }
                }
            }
        }
    }
    is_fail
}

/// The layer points the gate's failure blames: the subject cell for the
/// cell-named gates, both cells of the face for G4 - whose subject is a
/// FACE, not a cell id. G3 and G7 name no cell a retreat can serve. The
/// `is_layer` flags index the same points `cell_points` names.
fn failing_points(
    rep: &quality::QualityReport,
    mesh: &PolyMeshRaw,
    n_internal: usize,
    is_layer: &[bool],
    cell_points: &[Vec<u32>],
) -> Vec<usize> {
    let is_fail = failing_cell_flags(rep, mesh, n_internal, cell_points.len());
    let mut seen = vec![false; is_layer.len()];
    let mut out = Vec::new();
    for (c, bad) in is_fail.iter().enumerate() {
        if !bad {
            continue;
        }
        for &p in &cell_points[c] {
            let i = p as usize;
            if is_layer[i] && !seen[i] {
                seen[i] = true;
                out.push(i);
            }
        }
    }
    out.sort_unstable();
    out
}

/// (92.46): the fixed/free split and the interior relaxation, then the
/// hanging line. Layer points hold their `D_i` every pass, every other
/// BOUNDARY point holds zero, and interior points take `w * mean` over the
/// whole-graph neighbours - (92.29)'s shape, into a fresh vector each pass.
/// Then, longest parent edge first (as [`find_hanging`] orders them), every
/// hanging node rides its parents: `d_h <- (d_a + d_b) / 2`, applied to the
/// DISPLACEMENT and last, because the next unit's split faces close only if
/// a hanging node's copy stays the exact midpoint of its parents' copies at
/// every level - (92.33)'s line, for (92.33)'s reason and one more. The
/// points of `rs` hold (92.65)'s displacement as the layer points hold
/// theirs; each re-seat point's pull is its own `beta_j` (92.66).
fn relax(
    f: &Field,
    all_nbrs: &[Vec<u32>],
    is_b: &[bool],
    hanging: &[(u32, [u32; 2])],
    spec: &LayerSpec,
    rs: &Reseat,
    x: &[Vec3],
    beta: &[Scalar],
    anchored: &[bool],
) -> Vec<Vec3> {
    let n = f.disp.len();
    // Each re-seat point's displacement, computed ONCE from the `D_i` in
    // force at THIS call, so a retreat that halves `D_i` moves the point
    // with it while the re-seating part stays whole - (92.65). A point the
    // (92.73) ladder ANCHORED takes no displacement at any level, and the
    // re-seat must not resurrect one - it moves no level copy, so its zero
    // is the anchor.
    let held: HashMap<usize, Vec3> = rs
        .points
        .iter()
        .enumerate()
        .map(|(k, &j)| {
            let wall: Vec<(Vec3, Vec3, Vec3)> =
                rs.wall[k].iter().map(|&i| (x[i], f.disp[i], f.normal[i])).collect();
            (j, reseat_disp(x[j], &wall, beta[j]))
        })
        .collect();
    let mut d = f.disp.clone();
    for (&j, &dj) in held.iter() {
        if !anchored[j] {
            d[j] = dj;
        }
    }
    for _ in 0..spec.smoothing_passes {
        let mut next = vec![Vec3::ZERO; n];
        for i in 0..n {
            next[i] = if f.is_layer[i] {
                f.disp[i]
            } else if let Some(&dj) = held.get(&i) {
                dj
            } else if is_b[i] {
                Vec3::ZERO
            } else {
                let nbrs = &all_nbrs[i];
                if nbrs.is_empty() {
                    Vec3::ZERO
                } else {
                    let mut mean = Vec3::ZERO;
                    for &j in nbrs {
                        mean = mean + d[j as usize];
                    }
                    mean * (spec.smoothing / (nbrs.len() as Scalar))
                }
            };
        }
        d = next;
    }
    for &(h, ab) in hanging {
        d[h as usize] = (d[ab[0] as usize] + d[ab[1] as usize]) * 0.5;
    }
    // (92.73): an anchor is a zero at EVERY level, hanging parents and the
    // re-seat plan included - the pass path's own zeroing stays monotone.
    for (i, a) in anchored.iter().enumerate() {
        if *a {
            d[i] = Vec3::ZERO;
        }
    }
    d
}

// ==========================================================================
//  The extrusion
// ==========================================================================

/// (92.48): where every layer point's copies went, so a caller - and a
/// test - can find them without re-deriving the numbering.
#[derive(Debug, Clone)]
pub struct Extrusion {
    /// Per INPUT point: its slot in `L`, or -1.
    pub slot_of_point: Vec<i32>,
    /// `[n + 1][|L|]`: the point id of level `k` for slot `s`. Level `n` is
    /// the input point itself.
    pub level_point: Vec<Vec<u32>>,
    /// The input face id of each layer face, in the order the cell blocks
    /// were laid out: layer face `j` owns cells `first_cell + j*n .. +n`.
    pub layer_faces: Vec<usize>,
    /// (92.72): per layer face `j`, the first cell id of its block - in
    /// patch mode `first_cell + j*n`; in face mode a KEEP face still takes
    /// `n` cells but a RING face takes ONE wedge cell and an OFF face none.
    pub cell_start: Vec<usize>,
    /// (92.72): per layer face `j`, the number of cells its block carries -
    /// `n` on KEEP, 1 on RING, 0 on OFF.
    pub cell_count: Vec<usize>,
    /// The input point ids of the COPYING slots - the moving points of the
    /// KEEP and RING faces, `L_c ∩ L_m` of (92.72) - in the packed order the
    /// new level points were laid out in (a copying slot's level-`k` id is
    /// `n_points + mi*n + k` for its copy index `mi`); an anchored slot, or
    /// a moving point only on OFF faces, takes no new points at all.
    pub moving_slots: Vec<usize>,
    /// The input mesh's cell count - the first layer cell's id.
    pub first_cell: usize,
    pub n: usize,
}

impl Extrusion {
    /// The index `j` into `layer_faces` of the face whose block carries cell
    /// `c`, or `None` when `c` is not a layer cell.
    pub fn layer_face_of_cell(&self, c: usize) -> Option<usize> {
        if c < self.first_cell {
            return None;
        }
        let j = self.cell_start.partition_point(|&s| s <= c).checked_sub(1)?;
        if c < self.cell_start[j] + self.cell_count[j] {
            Some(j)
        } else {
            None
        }
    }
}

/// The thresholds `beta` the summary reports `area_frac_tau_ge` at: the
/// share of a patch's area whose face got at least `beta` of `T` (92.50).
pub const TAU_GE_BETAS: [Scalar; 3] = [0.5, 0.8, 0.95];

/// (92.50), per patch.
#[derive(Debug, Clone)]
pub struct PatchLayers {
    pub name: String,
    /// 0 when the patch was dropped.
    pub n_layers: usize,
    pub n_faces: usize,
    pub area: Scalar,
    /// The fraction of the patch's AREA that got the full stack.
    pub full_area_frac: Scalar,
    /// `frac_tau_ge(beta)` at each of `TAU_GE_BETAS`; `0.0` on a dropped
    /// patch.
    pub area_frac_tau_ge: [Scalar; 3],
    /// `(A_f, tau_f)` of (92.50) for every layer face of the patch, in the
    /// order `full` sums them; empty on a dropped patch.
    pub face_area_tau: Vec<(Scalar, Scalar)>,
    /// The area-weighted mean of the fraction of `T` actually achieved.
    pub mean_frac: Scalar,
    /// The first layer ASKED for, metres - `st.t[0]`, `0.0` when no stack
    /// exists (`spec.n == 0`).
    pub t1_requested: Scalar,
    /// The area-weighted mean first layer ACHIEVED, metres - `st.t[0] *
    /// mean_frac`. `0.0` on a dropped patch.
    pub t1_mean: Scalar,
    /// The smallest first layer ACHIEVED over the patch's faces, metres -
    /// `st.t[0] * min_f tau_f`. `0.0` on a dropped patch.
    pub t1_min: Scalar,
    /// `Some(reason)` when the patch lost its layers.
    pub dropped: Option<String>,
    /// What drove the give-up, when a ladder of (92.47) dropped the patch;
    /// `None` on a kept patch and where no ladder ran for it.
    pub drop_cause: Option<DropCause>,
    /// (92.64): how many interior points of row-1 cells the shrink re-seated for this patch - the largest count over every round the run ran with the patch in its set; 0 means the re-seat never acted on it.
    pub n_reseated_points: usize,
    /// (92.66): the beta rungs either ladder took on a patch set holding
    /// this patch, over the whole run - the count of the trace's `beta`
    /// entries whose patch set names it.
    pub beta_rungs: usize,
    /// The largest G4 angle over this patch's level-n faces (92.48) on the
    /// returned mesh - the near-wall non-orthogonality the solver sees;
    /// `None` on a patch that has no layers.
    pub level_n_non_orth_max_deg: Option<Scalar>,
    /// (92.74): the share of the patch's area whose face is KEEP - the full
    /// stack where it kept it. 1.0 on a patch-mode kept row, 0 on a dropped
    /// row.
    pub kept_area_frac: Scalar,
    /// (92.74): the share of the patch's area whose face is RING - tapered
    /// to zero through one wedge cell.
    pub ring_area_frac: Scalar,
    /// (92.70): the patch's KEEP / RING / OFF face counts.
    pub n_keep_faces: usize,
    pub n_ring_faces: usize,
    pub n_off_faces: usize,
    /// (92.70): how many of the patch's layer points are anchored.
    pub n_anchored_points: usize,
    /// (92.70'): the patch's RING faces with NO anchored point - the faces
    /// in M, each carrying one cell of the whole stack. 0 in patch mode and
    /// on a dropped row.
    pub n_merged_faces: usize,
    /// (92.70'): the share of the patch's area carried by the merged faces.
    pub merged_area_frac: Scalar,
    /// (92.70): the patch's OFF faces that are in X - the faces a ladder
    /// CUT, each leaving a step the wall carries. 0 in patch mode and on a
    /// dropped row.
    pub n_cut_faces: usize,
    /// (92.74): the share of the patch's area that carries the WHOLE stack
    /// thickness - the KEEP faces, plus the faces of M as one merged cell -
    /// as n cells or as one cell. `kept_area_frac` + the merged share.
    pub stack_area_frac: Scalar,
}

impl PatchLayers {
    /// The share of `area` whose face got at least `beta` of `T` - (92.50)'s
    /// `full` with `1` replaced by `beta`, over the same faces in the same
    /// order, so `frac_tau_ge(1.0)` is `full_area_frac` bit for bit.
    pub fn frac_tau_ge(&self, beta: Scalar) -> Scalar {
        let mut s = 0.0;
        for &(a, tau) in &self.face_area_tau {
            if tau >= beta - 1e-9 {
                s += a;
            }
        }
        if self.area > 0.0 {
            s / self.area
        } else {
            0.0
        }
    }
}

/// What the extrusion did, for the run log and the tests.
#[derive(Debug, Clone)]
pub struct LayerReport {
    pub patches: Vec<PatchLayers>,
    pub n_layer_cells: usize,
    pub n_layer_points: usize,
    pub n_side_internal: usize,
    pub n_side_boundary: usize,
    /// Side faces that came from cutting an edge at a hanging node (92.49).
    pub n_split_sides: usize,
    pub retreats: usize,
    /// Every measurement either ladder of (92.47) took, in the order taken.
    pub ladder: Vec<LadderEntry>,
}

impl LayerReport {
    /// A few lines for a run log, in `SnapReport::summary`'s style.
    pub fn summary(&self) -> String {
        let mut s = String::new();
        s.push_str(&format!(
            "layers: {} cell(s) behind {} face(s), {} new point(s), {} retreat(s)\n",
            self.n_layer_cells,
            self.patches.iter().map(|p| p.n_faces).sum::<usize>(),
            self.n_layer_points,
            self.retreats
        ));
        s.push_str(&format!(
            "layers: {} internal side face(s), {} boundary side face(s), {} from a split edge\n",
            self.n_side_internal, self.n_side_boundary, self.n_split_sides
        ));
        for p in &self.patches {
            match &p.dropped {
                Some(reason) => s.push_str(&format!("layers: {reason}\n")),
                None => {
                    let mut line = format!(
                        "layers: patch \"{}\": {} layer(s) on {} face(s), area {:.3e}, full {:.1}%, mean frac {:.3}, first layer {:.3e} m of {:.3e} requested (min {:.3e} m)",
                        p.name, p.n_layers, p.n_faces, p.area,
                        100.0 * p.full_area_frac, p.mean_frac,
                        p.t1_mean, p.t1_requested, p.t1_min
                    );
                    if p.full_area_frac == 0.0 {
                        line.push_str(" - NO face received the full stack");
                    }
                    // (92.74): a stack that ends per face says where, and
                    // (92.70') how much of the ring carries the whole stack
                    // as one merged cell.
                    if p.n_ring_faces + p.n_off_faces > 0 {
                        line.push_str(&format!(
                            ", kept {:.1}% of the area, stack {:.1}% (ring {:.1}% of which merged {:.1}%, {} cut face(s), {} anchored point(s))",
                            100.0 * p.kept_area_frac,
                            100.0 * p.stack_area_frac,
                            100.0 * p.ring_area_frac,
                            100.0 * p.merged_area_frac,
                            p.n_cut_faces,
                            p.n_anchored_points
                        ));
                    }
                    line.push('\n');
                    s.push_str(&line);
                }
            }
        }
        s
    }
}

/// What [`add_layers`] hands back: the mesh, the report, the map of the
/// extrusion, and §92.3's gate on the result.
#[derive(Debug, Clone)]
pub struct Layered {
    pub mesh: PolyMeshRaw,
    pub report: LayerReport,
    pub extrusion: Extrusion,
    pub quality: quality::QualityReport,
    /// The shrink's own (92.66) pull, handed to the outer ladder.
    pub beta: Vec<Scalar>,
    /// The shrink's own `J`.
    pub reseat_points: Vec<usize>,
    /// (92.70): the class of every layer face, in `extrusion.layer_faces`'s
    /// order - the classification this very attempt extruded with, which the
    /// OUTER ladder's M/X rules read.
    pub classes: Vec<FaceClass>,
}

/// SPEC-LIT §92.2 stage 6 / §92.13 end to end: shrink, extrude, renumber,
/// pass §92.3's gate.
///
/// The retreat ladder lives HERE, and it measures the EXTRUDED mesh, not the
/// shrunk one: each round runs the whole extrusion, reads §92.3's gate off
/// the mesh WITH its layer cells, and answers a failure with (92.47)'s
/// retreats - the offending layer points' thickness halved, the run
/// re-attempted - and, when the retreats run out, with the patch carrying
/// the most offending points losing its layers by name while the run
/// continues. What the shrink alone passes, the extrusion can still break -
/// a snapped wall carries its own non-orthogonality into the level-n face -
/// so no defect the extrusion introduces may end the run (SPEC-LIT §92.13,
/// (92.47)).
pub fn add_layers(
    mesh: &PolyMeshRaw,
    surf: &Surface,
    spec: &LayerSpec,
    t: &QualityThresholds,
) -> Result<Layered> {
    let mut patches = if spec.patches.is_empty() {
        Vec::new()
    } else {
        resolve_patches(mesh, spec)?
    };
    let mut extra_retreats = 0usize;
    let n_points = mesh.points.len();
    let n_faces = mesh.faces.len();
    let n_internal = mesh.neighbour.len().min(n_faces);
    // (92.73): the OUTER ladder's per-patch-set state - the caps, the pull,
    // M and X, and every counter - reset as a whole when a patch drops. The
    // OUTER ladder moves no point in face mode; the INPUT mesh's hanging map
    // is the INNER ladder's, inside `shrink_on`.
    let face_mode = spec.terminate == LayerTerminate::Face;
    let mut fl = FaceLadder::new(n_points, n_faces);
    let mut dropped: Vec<(String, String)> = Vec::new();
    // The whole trace, both ladders, read off and moving nothing (SPEC-LIT
    // §92.13), and the outer round it is at.
    let mut trace: Vec<LadderEntry> = Vec::new();
    let mut round = 0usize;
    // (92.64)'s per-patch re-seat counts, the max over every attempt the
    // run made, by patch name.
    let mut reseated: HashMap<String, usize> = HashMap::new();
    // (92.73)'s shrink reuse: the last attempt's patch set, caps and pull
    // with the Shrunk they produced. The shrink is a deterministic function
    // of those three, so a round whose three equal the last attempt's
    // reuses the answer - every mesh, report and trace entry bit for bit
    // what a fresh shrink returned - instead of running the shrink's ladder
    // again; the reused inner entries are pushed into the trace again with
    // the new round number, as always. Patch mode's caps change every
    // round, so only the face-mode rounds ever reuse.
    let mut shrink_cache: Option<(Vec<usize>, Vec<Scalar>, Vec<Scalar>, Shrunk)> = None;
    loop {
        let reuse = shrink_cache.as_ref().and_then(|(pp, pc, pb, s)| {
            if pp.as_slice() == patches.as_slice()
                && pc.as_slice() == fl.caps.as_slice()
                && pb.as_slice() == fl.betas.as_slice()
            {
                Some(s)
            } else {
                None
            }
        });
        let (mut a, shrunk) = attempt_on(
            mesh, surf, spec, t, &patches, &fl.caps, &fl.betas, &fl.merged, &fl.cut, reuse,
        )?;
        shrink_cache = Some((patches.clone(), fl.caps.clone(), fl.betas.clone(), shrunk));
        for row in &a.report.patches {
            let e = reseated.entry(row.name.clone()).or_insert(0);
            *e = (*e).max(row.n_reseated_points);
        }
        let this_round = round;
        round += 1;
        for mut e in std::mem::take(&mut a.report.ladder) {
            e.round = this_round;
            trace.push(e);
        }
        let names: Vec<String> = patches.iter().map(|&p| mesh.patches[p].name.clone()).collect();
        let gates = failing_gates(&a.quality);
        let g4_n = level_n_g4(&a.quality, &a.mesh, a.extrusion.first_cell);
        // The attempt passed the gate on the extruded mesh: the run
        // continues, and every patch an earlier round gave up carries its
        // row - zero layers, the reason, the input mesh's own face count and
        // area - alongside the patches that kept theirs.
        if a.quality.passed() {
            // A dropped row still names the first layer ASKED for, where a
            // stack exists at all.
            let t1_requested = if spec.n == 0 {
                0.0
            } else {
                stack(spec)?.t[0]
            };
            for (name, reason) in &dropped {
                if a
                    .report
                    .patches
                    .iter()
                    .any(|p| &p.name == name)
                {
                    continue;
                }
                let patch = mesh
                    .patches
                    .iter()
                    .find(|p| &p.name == name)
                    .expect("a dropped patch name is a patch of the mesh");
                a.report.patches.push(PatchLayers {
                    name: name.clone(),
                    n_layers: 0,
                    n_faces: patch.size,
                    area: patch_area(mesh, n_internal, patch),
                    full_area_frac: 0.0,
                    area_frac_tau_ge: [0.0; 3],
                    face_area_tau: Vec::new(),
                    mean_frac: 0.0,
                    t1_requested,
                    t1_mean: 0.0,
                    t1_min: 0.0,
                    dropped: Some(format!("patch \"{name}\": {reason}")),
                    drop_cause: Some(DropCause::OuterGate),
                    n_reseated_points: 0,
                    beta_rungs: 0,
                    level_n_non_orth_max_deg: None,
                    kept_area_frac: 0.0,
                    ring_area_frac: 0.0,
                    n_keep_faces: 0,
                    n_ring_faces: 0,
                    n_off_faces: 0,
                    n_anchored_points: 0,
                    n_merged_faces: 0,
                    merged_area_frac: 0.0,
                    n_cut_faces: 0,
                    stack_area_frac: 0.0,
                });
            }
            a.report.retreats += extra_retreats;
            let mut e = LadderEntry::new(Ladder::Outer, fl.halvings, &names, &gates, Outcome::Pass, None);
            e.round = this_round;
            e.g4_level_n = g4_n;
            e.beta_rung = fl.beta_rungs;
            trace.push(e);
            a.report.ladder = trace;
            for row in a.report.patches.iter_mut() {
                row.n_reseated_points = reseated.get(&row.name).copied().unwrap_or(0);
            }
            // (92.66)'s report fields, read off the final trace and the
            // returned mesh: the beta rungs each row's patch set took, and
            // the near-wall angle its level-n faces hand the solver.
            let non_orth = level_n_non_orth_max(mesh, &a.mesh, &a.extrusion)?;
            for row in a.report.patches.iter_mut() {
                row.beta_rungs = a
                    .report
                    .ladder
                    .iter()
                    .filter(|e| {
                        e.outcome == Outcome::Beta && e.patches.iter().any(|nm| *nm == row.name)
                    })
                    .count();
                row.level_n_non_orth_max_deg = if row.dropped.is_none() && row.n_layers > 0 {
                    mesh.patches
                        .iter()
                        .position(|p| p.name == row.name)
                        .and_then(|k| non_orth[k])
                } else {
                    None
                };
            }
            return Ok(a);
        }
        // No layer cell was inserted, so the gate's failure is the INPUT
        // mesh's own - no retreat on layer points can mend it.
        if patches.is_empty() || a.extrusion.layer_faces.is_empty() {
            return Err(Error::Mesh(a.quality.refusal_text()));
        }
        // The layer points to halve, read off the EXTRUDED mesh: a failing
        // cell may be a layer cell, whose points are level copies, so every
        // face point is mapped back to its INPUT id before the failing
        // cells' point sets are built.
        let n = a.extrusion.n;
        // (92.72): the new level points are laid out over the MOVING slots
        // in order, so new id p maps back through `moving_slots`.
        let moving_slots = &a.extrusion.moving_slots;
        let orig_of =
            |p: crate::Label| -> usize {
                let p = p as usize;
                if p < n_points {
                    p
                } else {
                    moving_slots[(p - n_points) / n]
                }
            };
        let n_faces_out = a.mesh.faces.len();
        let n_internal_out = a.mesh.neighbour.len().min(n_faces_out);
        let n_cells_out = a
            .mesh
            .owner
            .iter()
            .chain(a.mesh.neighbour.iter())
            .copied()
            .max()
            .map_or(0, |m| m as usize + 1);
        let mut cell_points: Vec<Vec<u32>> = vec![Vec::new(); n_cells_out];
        for (f, face) in a.mesh.faces.iter().enumerate() {
            for &p in face {
                cell_points[a.mesh.owner[f] as usize].push(orig_of(p) as u32);
            }
            if f < n_internal_out {
                for &p in face {
                    cell_points[a.mesh.neighbour[f] as usize].push(orig_of(p) as u32);
                }
            }
        }
        for list in cell_points.iter_mut() {
            list.sort_unstable();
            list.dedup();
        }
        let is_layer: Vec<bool> = a
            .extrusion
            .slot_of_point
            .iter()
            .map(|&s| s >= 0)
            .collect();
        let fail_pts =
            failing_points(&a.quality, &a.mesh, n_internal_out, &is_layer, &cell_points);
        // (92.66): before any give-up check or halving, a failure whose
        // failing cells carry a live re-seat point takes the pull back
        // there - the caps, the halvings and the drop are untouched.
        let mut live = vec![false; n_points];
        for &j in &a.reseat_points {
            live[j] = a.beta[j] > 0.0;
        }
        let jf = failing_points(&a.quality, &a.mesh, n_internal_out, &live, &cell_points);
        if !jf.is_empty() && fl.beta_rungs < BETA_RUNG_LIMIT {
            let mut e =
                LadderEntry::new(Ladder::Outer, fl.halvings, &names, &gates, Outcome::Beta, None);
            e.round = this_round;
            e.g4_level_n = g4_n;
            e.beta_rung = fl.beta_rungs;
            e.beta_points = jf.len();
            trace.push(e);
            for &j in &jf {
                fl.betas[j] = beta_step(a.beta[j]);
            }
            fl.beta_rungs += 1;
            continue;
        }
        let mut face_dropping = false;
        if face_mode {
            // (92.73) amended: the OUTER ladder moves no point. Every
            // G5-named layer cell of a KEEP face puts that face in M - the
            // merge, one cell carrying the whole stack; every other named
            // layer cell, and the layer face of every level-n face G4 names,
            // puts its face in X - the cut, no cell, a step the wall
            // carries. The next attempt re-extrudes with M and X; a round
            // that adds nothing to either, a failure on a cell that is
            // neither a layer cell nor at a level-n face, or the step limit
            // ends the ladder by (92.47)'s patch rule, whose block below
            // drops the victim with the same reason and the same reset.
            let mut added = false;
            let mut unanswerable = false;
            // The input cells a level-n face reaches: a failure ON one of
            // these is what the G4 rule's cut answers; a failure on any
            // other input cell is one no merge or cut can reach.
            let first_cell = a.extrusion.first_cell;
            let mut at_level_n = vec![false; first_cell];
            for fa in 0..n_internal_out {
                let o = a.mesh.owner[fa] as usize;
                if o < first_cell && (a.mesh.neighbour[fa] as usize) >= first_cell {
                    at_level_n[o] = true;
                }
            }
            // M first: a G5-named cell of a face this round puts in M is
            // answered by the merge and never reaches the X pass.
            for failure in &a.quality.failures {
                if !matches!(failure.gate, Gate::Thickness) {
                    continue;
                }
                for s in &failure.subjects {
                    if let Some(j) = a.extrusion.layer_face_of_cell(s.id) {
                        let fid = a.extrusion.layer_faces[j];
                        if a.classes[j] == FaceClass::Keep && !fl.merged[fid] {
                            fl.merged[fid] = true;
                            added = true;
                        }
                    }
                }
            }
            // X: a RING cell named by any gate (a merge that did not mend
            // its cell), or a KEEP cell named by a gate other than G5, cuts
            // its face; so does the layer face of every level-n face G4
            // names. X wins over M in (92.70)'s classification.
            for failure in &a.quality.failures {
                match failure.gate.subject() {
                    quality::Subject::Cell => {
                        for s in &failure.subjects {
                            match a.extrusion.layer_face_of_cell(s.id) {
                                Some(j) => {
                                    let answered_by_m = a.classes[j] == FaceClass::Keep
                                        && matches!(failure.gate, Gate::Thickness);
                                    let fid = a.extrusion.layer_faces[j];
                                    if !answered_by_m && !fl.cut[fid] {
                                        fl.cut[fid] = true;
                                        added = true;
                                    }
                                }
                                None => {
                                    let c = s.id;
                                    if c < first_cell && !at_level_n[c] {
                                        unanswerable = true;
                                    }
                                }
                            }
                        }
                    }
                    quality::Subject::Face => {
                        if !matches!(failure.gate, Gate::NonOrth) {
                            continue;
                        }
                        for s in &failure.subjects {
                            let fa = s.id;
                            if fa >= n_internal_out {
                                continue;
                            }
                            // G4's face names BOTH its cells, as
                            // `failing_cell_flags` reads it: a layer cell cuts
                            // its face - a level-n face's layer cell, or a
                            // wedge cell whose merged side face leans - and an
                            // input cell not at a level-n face is a failure
                            // nothing answers.
                            for &c in &[
                                a.mesh.owner[fa] as usize,
                                a.mesh.neighbour[fa] as usize,
                            ] {
                                if let Some(j) = a.extrusion.layer_face_of_cell(c) {
                                    let fid = a.extrusion.layer_faces[j];
                                    if !fl.cut[fid] {
                                        fl.cut[fid] = true;
                                        added = true;
                                    }
                                } else if c < first_cell && !at_level_n[c] {
                                    unanswerable = true;
                                }
                            }
                        }
                    }
                }
            }
            if unanswerable || !added || fl.steps >= TERMINATE_STEP_LIMIT {
                face_dropping = true;
            } else {
                let mut e = LadderEntry::new(
                    Ladder::Outer, fl.halvings, &names, &gates, Outcome::Terminate, None,
                );
                e.round = this_round;
                e.g4_level_n = g4_n;
                e.beta_rung = fl.beta_rungs;
                e.terminate_points = 0;
                trace.push(e);
                fl.steps += 1;
                continue;
            }
        }
        if face_dropping || fail_pts.is_empty() || fl.halvings >= spec.retreat_limit {
            // The patch that loses its layers: the one carrying the most
            // offending layer points, ties to the lower patch index. An
            // empty list blames the first patch - the gate failed on cells
            // no layer point reaches.
            let mut counts = vec![0usize; patches.len()];
            for (k, &p) in patches.iter().enumerate() {
                let patch = &mesh.patches[p];
                let mut seen = vec![false; n_points];
                for j in 0..patch.size {
                    let fa = n_internal + patch.start + j;
                    if fa >= n_faces {
                        continue;
                    }
                    for &pt in &mesh.faces[fa] {
                        seen[pt as usize] = true;
                    }
                }
                for &i in &fail_pts {
                    if seen[i] {
                        counts[k] += 1;
                    }
                }
            }
            let mut victim = 0usize;
            for k in 1..counts.len() {
                if counts[k] > counts[victim] {
                    victim = k;
                }
            }
            let vp = patches[victim];
            let reason = format!(
                "the gate still failed after {} retreat(s) on the layer \
                 cells themselves - the extruded mesh could not be brought inside the \
                 gate: layers are supported on a wall that castellates onto the cell \
                 planes, and a snapped wall carries its own non-orthogonality into the \
                 level-n face (SPEC-LIT 92.13)",
                fl.halvings
            );
            let mut e = LadderEntry::new(
                Ladder::Outer, fl.halvings, &names, &gates, Outcome::GiveUp, Some(DropCause::OuterGate),
            );
            e.round = this_round;
            e.g4_level_n = g4_n;
            e.beta_rung = fl.beta_rungs;
            e.dropped = Some(mesh.patches[vp].name.clone());
            trace.push(e);
            dropped.push((mesh.patches[vp].name.clone(), reason));
            patches.remove(victim);
            // (92.73): EVERY per-patch-set counter, M and X reset when a
            // patch drops - the next set starts clean, never at the last
            // set's step limit. (`reset` is one function; before it, a set
            // that dropped started at the limit and every later patch lost
            // its layers right after its beta rungs.)
            fl.reset(n_points, n_faces);
        } else {
            let mut e = LadderEntry::new(Ladder::Outer, fl.halvings, &names, &gates, Outcome::Retreat, None);
            e.round = this_round;
            e.g4_level_n = g4_n;
            e.beta_rung = fl.beta_rungs;
            trace.push(e);
            for &i in &fail_pts {
                fl.caps[i] = fl.caps[i] * 0.5;
            }
            fl.halvings += 1;
            extra_retreats += 1;
        }
    }
}

/// One run of the extrusion, at a FIXED patch set and per-point retreat cap,
/// with a fresh shrink: the tests' entry, reading `quality.passed()` - an
/// attempt does not refuse on the gate, because a gate failure on the layer
/// cells is what the ladder retreats on. [`add_layers`] itself calls
/// [`attempt_on`] with the shrink reuse.
#[cfg(test)]
#[allow(clippy::too_many_arguments)]
fn attempt(
    mesh: &PolyMeshRaw,
    surf: &Surface,
    spec: &LayerSpec,
    t: &QualityThresholds,
    patches0: &[usize],
    caps: &[Scalar],
    betas: &[Scalar],
    merged: &[bool],
    cut: &[bool],
) -> Result<Layered> {
    attempt_on(mesh, surf, spec, t, patches0, caps, betas, merged, cut, None)
        .map(|(a, _)| a)
}

/// [`attempt`] with the previous shrink handed in: `Some(s)` makes `s` the
/// attempt's shrink without running the shrink's ladder again - the shrink
/// is a deterministic function of the patch set, the caps and the pull, so
/// the reused answer is bit for bit the fresh one - and the attempt returns
/// the shrink it used, so [`add_layers`] can hand it back on the next round
/// whose inputs match.
#[allow(clippy::too_many_arguments)]
fn attempt_on(
    mesh: &PolyMeshRaw,
    surf: &Surface,
    spec: &LayerSpec,
    t: &QualityThresholds,
    patches0: &[usize],
    caps: &[Scalar],
    betas: &[Scalar],
    merged: &[bool],
    cut: &[bool],
    shrunk: Option<&Shrunk>,
) -> Result<(Layered, Shrunk)> {
    let st = stack(spec)?;
    let n = st.n;
    let named = patches0.to_vec();
    let n_points = mesh.points.len();
    let n_faces = mesh.faces.len();
    let n_internal = mesh.neighbour.len().min(n_faces);
    let first_cell = mesh
        .owner
        .iter()
        .chain(mesh.neighbour.iter())
        .copied()
        .max()
        .map_or(0, |m| m as usize + 1);
    let shrunk = match shrunk {
        Some(s) => s.clone(),
        None => shrink_on(mesh, surf, spec, t, patches0, caps, betas)?,
    };
    let field = &shrunk.field;
    // Nothing to do: no layers were asked for, or every named patch gave
    // its layers up in the shrink. The input mesh comes back bit for bit.
    if n == 0 || field.faces.is_empty() {
        let mut patches = Vec::new();
        for &p in &named {
            let patch = &mesh.patches[p];
            let drop_cause =
                shrunk.drop_causes.iter().find(|(nm, _)| nm == &patch.name).map(|(_, c)| *c);
            let reason = match shrunk.dropped.iter().find(|(nm, _)| nm == &patch.name) {
                Some((_, r)) => format!("patch \"{}\": {r}", patch.name),
                None if n == 0 => format!(
                    "patch \"{}\": layers.n is zero - no layers were requested",
                    patch.name
                ),
                None => format!("patch \"{}\": the patch carries no layer face", patch.name),
            };
            patches.push(PatchLayers {
                name: patch.name.clone(),
                n_layers: 0,
                n_faces: patch.size,
                area: patch_area(mesh, n_internal, patch),
                full_area_frac: 0.0,
                area_frac_tau_ge: [0.0; 3],
                face_area_tau: Vec::new(),
                mean_frac: 0.0,
                t1_requested: if n == 0 { 0.0 } else { st.t[0] },
                t1_mean: 0.0,
                t1_min: 0.0,
                dropped: Some(reason),
                drop_cause,
                n_reseated_points: shrunk
                    .reseated
                    .iter()
                    .find(|(nm, _)| nm == &patch.name)
                    .map_or(0, |(_, n)| *n),
                beta_rungs: 0,
                level_n_non_orth_max_deg: None,
                kept_area_frac: 0.0,
                ring_area_frac: 0.0,
                n_keep_faces: 0,
                n_ring_faces: 0,
                n_off_faces: 0,
                n_anchored_points: 0,
                n_merged_faces: 0,
                merged_area_frac: 0.0,
                n_cut_faces: 0,
                stack_area_frac: 0.0,
            });
        }
        let report = LayerReport {
            patches,
            n_layer_cells: 0,
            n_layer_points: 0,
            n_side_internal: 0,
            n_side_boundary: 0,
            n_split_sides: 0,
            retreats: shrunk.retreats,
            ladder: shrunk.ladder.clone(),
        };
        // §92.3's gate, run on the mesh that came back - which is the
        // input's, so this is a formality that costs nothing.
        let quality = quality::check(mesh, t)?;
        return Ok((
            Layered {
                mesh: shrunk.mesh.clone(),
                report,
                extrusion: Extrusion {
                    slot_of_point: vec![-1; n_points],
                    level_point: vec![Vec::new(); n + 1],
                    layer_faces: Vec::new(),
                    cell_start: Vec::new(),
                    cell_count: Vec::new(),
                    moving_slots: Vec::new(),
                    first_cell,
                    n,
                },
                quality,
                beta: shrunk.beta.clone(),
                reseat_points: shrunk.reseat_points.clone(),
                classes: Vec::new(),
            },
            shrunk,
        ));
    }

    // (92.51)'s early refusal, BEFORE any cell is inserted: G5 would refuse
    // every one of these cells, and the user must see the arithmetic, not a
    // gate failure on some cell id.
    let mut h_min = Scalar::INFINITY;
    for &f in &field.faces {
        let face = &mesh.faces[f];
        for k in 0..face.len() {
            let d = (mesh.points[face[k] as usize]
                - mesh.points[face[(k + 1) % face.len()] as usize])
                .mag();
            if d < h_min {
                h_min = d;
            }
        }
    }
    let ratio = 3.0 * st.t[0] / h_min;
    if !(ratio >= t.min_thickness_ratio) {
        return Err(Error::Mesh(format!(
            "layers: the first layer is thinner than the quality gate allows - \
             3 * {} / {} = {} < min_thickness_ratio = {} (92.51); every layer \
             cell would fail G5, so none is inserted",
            st.t[0], h_min, ratio, t.min_thickness_ratio
        )));
    }
    // The field's displacement is the APPLIED one - the doc comment on
    // `Field` says so - so the level positions of (92.48) are exact
    // interpolations between the wall and where the shrink put it. Checked,
    // because every level copy is wrong if this is not.
    for i in 0..n_points {
        if !field.is_layer[i] {
            continue;
        }
        let want = mesh.points[i] + field.disp[i];
        let e = (shrunk.mesh.points[i] - want).mag();
        if e > 1e-12 {
            return Err(Error::Mesh(format!(
                "layers: the shrink's points disagree with its own field at \
                 point {i} by {e:.3e} - the extrusion would not interpolate \
                 between the wall and the shrunk mesh"
            )));
        }
    }

    // The slots: L is the set of layer points, in input-point order. A
    // level copy of slot `s` is a new point after all the input's, so the
    // input's point ids are unchanged.
    let mut slot_of_point = vec![-1i32; n_points];
    let mut slots: Vec<usize> = Vec::new();
    for i in 0..n_points {
        if field.is_layer[i] {
            slot_of_point[i] = slots.len() as i32;
            slots.push(i);
        }
    }
    let n_l = slots.len();
    // (92.70): a layer point whose applied displacement is exactly the zero
    // vector is ANCHORED - it takes no level copies at all (its level-k
    // position is the point itself at every k). The patch mode never accepts
    // a zero, so `anchored` is empty there and everything below is the
    // patch-mode layout bit for bit.
    let anchored_slot: Vec<bool> = slots
        .iter()
        .map(|&i| field.disp[i].mag_sqr() == 0.0)
        .collect();
    let anchored_pt: Vec<bool> = (0..n_points)
        .map(|i| field.is_layer[i] && field.disp[i].mag_sqr() == 0.0)
        .collect();
    // (92.70): every layer face's class, read once - the cells, the faces
    // and the sides all branch on it. A face in X is OFF whatever its
    // points do; a face in M is RING with no anchored point; every point
    // anchored is OFF whatever M says.
    let classes: Vec<FaceClass> = field
        .faces
        .iter()
        .map(|&f| {
            let c = face_class(&mesh.faces[f], &anchored_pt);
            if cut[f] {
                FaceClass::Off
            } else if c == FaceClass::Keep && merged[f] {
                FaceClass::Ring
            } else {
                c
            }
        })
        .collect();
    // (92.72) amended: L_c, the points of the KEEP and RING faces. Only
    // L_c ∩ L_m takes level copies - an OFF face's moving point sits at its
    // shrunk position, the mesh's own point, and needs none.
    let mut on_stack = vec![false; n_points];
    for (j, &f) in field.faces.iter().enumerate() {
        if classes[j] != FaceClass::Off {
            for &p in &mesh.faces[f] {
                on_stack[p as usize] = true;
            }
        }
    }
    let copies_slot: Vec<bool> = (0..n_l)
        .map(|s| !anchored_slot[s] && on_stack[slots[s]])
        .collect();
    // (92.72): P + n |L_c ∩ L_m| - only the COPYING slots take level copies,
    // packed in slot order, so with nothing anchored or cut the ids are
    // today's.
    let moving_of_slot: Vec<usize> = {
        let mut mi = 0usize;
        (0..n_l)
            .map(|s| {
                let m = mi;
                if copies_slot[s] {
                    mi += 1;
                }
                m
            })
            .collect()
    };
    let n_moving = copies_slot.iter().filter(|c| **c).count();
    let mut points = shrunk.mesh.points.clone();
    points.reserve(n_moving * n);
    let mut level_point: Vec<Vec<u32>> = vec![vec![0u32; n_l]; n + 1];
    for (s, &i) in slots.iter().enumerate() {
        // Level n is the input point itself, already moved by the shrink -
        // and for a point that takes no copies (anchored, or only on OFF
        // faces) every level is that same point: the shrink put it where it
        // stands, or did not move it at all.
        level_point[n][s] = i as u32;
        if !copies_slot[s] {
            for lp in level_point.iter_mut().take(n) {
                lp[s] = i as u32;
            }
            continue;
        }
        let base = n_points + moving_of_slot[s] * n;
        for k in 0..n {
            level_point[k][s] = (base + k) as u32;
            // (92.48): x_i^(k) = x_i^orig + f_k D_i, from the INPUT point.
            points.push(mesh.points[i] + field.disp[i] * st.f[k]);
        }
    }
    // (92.72): the cell blocks, in field-face order - n cells on KEEP, one
    // wedge cell on RING, none on OFF. In patch mode this is today's
    // first_cell + j*n layout exactly.
    let mut cell_start: Vec<usize> = Vec::with_capacity(field.faces.len());
    let mut cell_count: Vec<usize> = Vec::with_capacity(field.faces.len());
    let mut next_cell = first_cell;
    for cls in &classes {
        let cnt = match cls {
            FaceClass::Keep => n,
            FaceClass::Ring => 1,
            FaceClass::Off => 0,
        };
        cell_start.push(next_cell);
        cell_count.push(cnt);
        next_cell += cnt;
    }

    // The faces' bookkeeping the sides and the boundary read: which input
    // face is a layer face, which layer face sits at which block index,
    // and the box diagonal the zero-area test is scaled by.
    let mut is_layer_face = vec![false; n_faces];
    for &f in &field.faces {
        is_layer_face[f] = true;
    }
    let mut layer_j = vec![-1i32; n_faces];
    for (j, &f) in field.faces.iter().enumerate() {
        layer_j[f] = j as i32;
    }
    let mut b_lo = mesh.points[0];
    let mut b_hi = mesh.points[0];
    for q in &mesh.points {
        b_lo = b_lo.cmpt_min(*q);
        b_hi = b_hi.cmpt_max(*q);
    }
    let diag2 = (b_hi - b_lo).mag_sqr();
    // (92.13): in face mode a side face of NONZERO area under the
    // `1e-14 * diag2` guard is emitted for the gate to judge - the
    // full-size F1 tunnel stopped on one of 4.287e-11 m^2 against a
    // 4.3e-11 m^2 threshold, a sliver edge of the snapped wall carrying a
    // real stack, not a stack that went to zero. Patch mode refuses it.
    let face_mode = spec.terminate == LayerTerminate::Face;


    // ---- the sides: (92.49)'s segments ------------------------------
    // NOT `snap::find_hanging`: that map is keyed by the hanging NODE and
    // keeps only the node's LONGEST parent edge, so a node that is the
    // midpoint of a wall edge and of a longer internal edge would be
    // missing from the wall's map and this segment would come out
    // unmatched. The cut is built here, over the BOUNDARY faces alone.
    let mut used_by_boundary = vec![false; n_points];
    for f in n_internal..n_faces {
        for &p in &mesh.faces[f] {
            used_by_boundary[p as usize] = true;
        }
    }
    let mp = Midpoints::new(&shrunk.mesh.points, &used_by_boundary);
    let mut face_segs: Vec<Vec<Vec<(u32, u32)>>> = vec![Vec::new(); n_faces];
    let mut seg_faces: HashMap<(u32, u32), Vec<usize>> = HashMap::new();
    for f in n_internal..n_faces {
        let face = &mesh.faces[f];
        let mut per_edge: Vec<Vec<(u32, u32)>> = Vec::with_capacity(face.len());
        for k in 0..face.len() {
            let a = face[k] as u32;
            let b = face[(k + 1) % face.len()] as u32;
            let segs = split_edge(&mp, a, b, 0)?;
            for &(u, v) in &segs {
                let key = if u < v { (u, v) } else { (v, u) };
                seg_faces.entry(key).or_default().push(f);
            }
            per_edge.push(segs);
        }
        face_segs[f] = per_edge;
    }
    let mut patch_of_bface = vec![usize::MAX; n_faces];
    for (p, patch) in mesh.patches.iter().enumerate() {
        for j in 0..patch.size {
            let f = n_internal + patch.start + j;
            if f < n_faces {
                patch_of_bface[f] = p;
            }
        }
    }
    let mut n_side_internal = 0usize;
    let mut n_side_boundary = 0usize;
    let mut n_split_sides = 0usize;
    let mut internal_sides: Vec<(usize, usize, Vec<crate::Label>)> = Vec::new();
    let mut boundary_sides: Vec<Vec<(usize, usize, usize, crate::Label, Vec<crate::Label>)>> =
        vec![Vec::new(); mesh.patches.len()];
    for (j, &f) in field.faces.iter().enumerate() {
        // (92.72) amended: an OFF face has no cell and no levels - the step
        // it leaves is built from its NEIGHBOURS' sides, so it contributes
        // no segment of its own.
        if classes[j] == FaceClass::Off {
            continue;
        }
        let face = &mesh.faces[f];
        for (e, _) in face.iter().enumerate() {
            let cut = face_segs[f][e].len() > 1;
            // The segments came back in order along the edge as the face
            // winds it, so each (u, v) is directed already.
            for (si, &(u, v)) in face_segs[f][e].iter().enumerate() {
                // Written by the supervising session. The cut is made at
                // any midpoint a BOUNDARY face carries, so a layer face
                // that is COARSER than the non-layer patch beside it is cut
                // at a point that has no normal, no thickness and no level
                // copies - and the indexing below would take slot -1. That
                // shape does not arise on the cases this stage is built for
                // (the wall is the refined patch and the box sides are at
                // base level), and supporting it needs a side face that
                // spans a whole edge while its partner is cut, which is the
                // terminating topology §92.13 defers to tranche 2. So it is
                // REFUSED by name rather than reached as a panic.
                if slot_of_point[u as usize] < 0 || slot_of_point[v as usize] < 0 {
                    return Err(Error::Mesh(format!(
                        "layers: the edge of layer face {f} was cut at a point                          that is not a layer point - segment ({u}, {v}), points                          {:?} and {:?}. A layer patch coarser than the patch                          beside it is not supported in tranche 1 (SPEC-LIT                          §92.13); refine the layer patch to at least the level                          of its neighbour, or take the layers off it",
                        points[u as usize], points[v as usize]
                    )));
                }
                let su = slot_of_point[u as usize] as usize;
                let sv = slot_of_point[v as usize] as usize;
                let key = if u < v { (u, v) } else { (v, u) };
                let partners: Vec<usize> = seg_faces[&key]
                    .iter()
                    .copied()
                    .filter(|&g| g != f)
                    .collect();
                if partners.len() != 1 {
                    return Err(Error::Mesh(format!(
                        "layers: the segment ({u}, {v}) of layer face {f} - points \
                         {:?}, {:?} - is carried by {} boundary face(s) besides \
                         itself, expected exactly one (92.49): the layer patch \
                         has an open rim",
                        points[u as usize],
                        points[v as usize],
                        partners.len()
                    )));
                }
                let g = partners[0];
                // (92.71): two anchored endpoints close the wedge cell by
                // themselves - the level-0 edge and the input edge coincide
                // - so no side face of any kind is emitted between them.
                if anchored_slot[su] && anchored_slot[sv] {
                    continue;
                }
                let jg = layer_j[g];
                let gcls = if jg >= 0 {
                    classes[jg as usize]
                } else {
                    FaceClass::Keep
                };
                let cf0 = cell_start[j];
                let cg0 = if jg >= 0 { cell_start[jg as usize] } else { 0 };
                // cell(f, k) of (92.71): the level-k cell on KEEP, the one
                // wedge cell on RING - the blocks of (92.72).
                let cell_k = |cls: FaceClass, s0: usize, k: usize| {
                    if cls == FaceClass::Keep {
                        s0 + k
                    } else {
                        s0
                    }
                };
                // The side face is wound by the topology, a CONSTANT, not by
                // a measurement. Directed as `f` winds its segment and
                // stepped inward by `w`, its area vector is d x w = n_out x d
                // - the INTERIOR direction of face `f`, always. So the list
                // as constructed points INTO `f`'s own layer cell `cf`, and a
                // face is wound out of its owner: `cf` takes the reversed
                // list, the other cell `cg` the list as constructed. (The
                // dot product against a cell centre an earlier form read
                // ~zero at the convex edges of a snapped wall and flipped
                // the quad there.)
                let mut emit =
                    |own_is_cf: bool,
                     own: usize,
                     nbr: usize,
                     mut ps: Vec<crate::Label>,
                     boundary: bool,
                     k_slot: usize,
                     at_ring: bool| {
                        // (92.71): a RING face's wedge tapers to ZERO at an
                        // anchored point, so a side face with NO area there
                        // is the taper's own shape - its area contributes
                        // nothing, and the closures of (92.54) hold without
                        // it. A face with a tiny-but-NONZERO area is EMITTED
                        // in face mode whatever `at_ring` says: dropping it
                        // would leave both its cells open by exactly that
                        // area, and the closure of (92.54) is topology, not
                        // scale - a snapped wall's sliver edge closes or
                        // nothing does - so the gate of §92.3 judges its
                        // cells, and (92.73)'s ladder cuts their faces if
                        // they fail. Patch mode still refuses it, and a
                        // KEEP face's side is a prism wall, so a zero area
                        // there is still the thickness going to zero where
                        // it was not allowed to.
                        let a_vec = face_area_vector(&points, &ps);
                        if ps.len() < 3 || a_vec.mag() == 0.0 {
                            if at_ring {
                                return Ok(());
                            }
                            return Err(Error::Mesh(format!(
                                "layers: the side face of segment ({u}, {v}) of layer \
                                 face {f} collapsed to {} point(s) - the thickness \
                                 went to zero where it was not allowed to",
                                ps.len()
                            )));
                        }
                        if !face_mode && !at_ring && a_vec.mag() < 1e-14 * diag2 {
                            return Err(Error::Mesh(format!(
                                "layers: the side face of segment ({u}, {v}) of layer \
                                 face {f} has area {:.3e} - the thickness went to zero \
                                 where it was not allowed to",
                                a_vec.mag()
                            )));
                        }
                        if own_is_cf {
                            ps.reverse();
                        }
                        if boundary {
                            boundary_sides[patch_of_bface[g]].push((f, k_slot, si, own as crate::Label, ps));
                            n_side_boundary += 1;
                        } else {
                            internal_sides.push((own, nbr, ps));
                            n_side_internal += 1;
                        }
                        if cut {
                            n_split_sides += 1;
                        }
                        Ok(())
                    };
                let piece = |k: usize| -> Vec<crate::Label> {
                    // (92.49)'s quad at level k, its consecutive duplicates
                    // removed cyclically - an anchored endpoint's level
                    // copies are all the point itself.
                    dedupe_cyclic(vec![
                        level_point[k][su] as crate::Label,
                        level_point[k][sv] as crate::Label,
                        level_point[k + 1][sv] as crate::Label,
                        level_point[k + 1][su] as crate::Label,
                    ])
                };
                let merged = || -> Vec<crate::Label> {
                    // (92.71)'s one merged face, a RING cell's whole side:
                    // dedupe(u^(0), v^(0)..v^(n), u^(n)..u^(1)).
                    let mut ps: Vec<crate::Label> = Vec::with_capacity(2 * n + 2);
                    ps.push(level_point[0][su] as crate::Label);
                    for k in 0..=n {
                        ps.push(level_point[k][sv] as crate::Label);
                    }
                    for k in (1..=n).rev() {
                        ps.push(level_point[k][su] as crate::Label);
                    }
                    dedupe_cyclic(ps)
                };
                if is_layer_face[g] && gcls != FaceClass::Off {
                    if f < g {
                        if classes[j] == FaceClass::Keep || gcls == FaceClass::Keep {
                            // (92.71): either face KEEP - the n pieces,
                            // internal, between cell(f,k) and cell(g,k).
                            for k in 0..n {
                                let cf = cell_k(classes[j], cf0, k);
                                let cg = cell_k(gcls, cg0, k);
                                let (own, nbr) = if cf < cg {
                                    (cf, cg)
                                } else {
                                    (cg, cf)
                                };
                                emit(own == cf, own, nbr, piece(k), false, 0, false)?;
                            }
                        } else {
                            // (92.71): both RING - ONE face, internal,
                            // between the two wedge cells.
                            let (own, nbr) = if cf0 < cg0 {
                                (cf0, cg0)
                            } else {
                                (cg0, cf0)
                            };
                            emit(own == cf0, own, nbr, merged(), false, 0, true)?;
                        }
                    }
                } else {
                    // (92.72) amended: the partner is OFF - a cut face, whose
                    // wall stepped down to the shrunk position - or on a
                    // non-layer patch: the side is a BOUNDARY face of that
                    // patch, owned by `f`'s cell.
                    if classes[j] == FaceClass::Keep {
                        for k in 0..n {
                            let own = cell_k(classes[j], cf0, k);
                            emit(true, own, 0, piece(k), true, k, false)?;
                        }
                    } else {
                        emit(true, cf0, 0, merged(), true, 0, true)?;
                    }
                }
            }
        }
    }

    // The assembly: the input's internal faces, unchanged, in order; then
    // every level 1..n face of (92.48); then the internal side faces.
    let mut faces: Vec<Vec<crate::Label>> = Vec::new();
    let mut owner: Vec<crate::Label> = Vec::new();
    let mut neighbour: Vec<crate::Label> = Vec::new();
    for f in 0..n_internal {
        faces.push(mesh.faces[f].clone());
        owner.push(mesh.owner[f]);
        neighbour.push(mesh.neighbour[f]);
    }
    for (j, &f) in field.faces.iter().enumerate() {
        let face = &mesh.faces[f];
        match classes[j] {
            // (92.70): an OFF face emits nothing here - it stays the
            // boundary face it was.
            FaceClass::Off => {}
            // (92.71): a RING face's level-n face is the input face itself,
            // internal, owner its input cell, neighbour the one wedge cell.
            FaceClass::Ring => {
                faces.push(face.clone());
                owner.push(mesh.owner[f]);
                neighbour.push(cell_start[j] as crate::Label);
            }
            FaceClass::Keep => {
                for k in 1..=n {
                    if k < n {
                        // Levels 1..n-1 carry the REVERSED point list: the
                        // owner is nearer the wall, so the normal has to
                        // point inward.
                        let mut ps: Vec<crate::Label> = face
                            .iter()
                            .map(|p| {
                                level_point[k]
                                    [slot_of_point[*p as usize] as usize]
                                    as crate::Label
                            })
                            .collect();
                        ps.reverse();
                        faces.push(ps);
                        owner.push((cell_start[j] + k - 1) as crate::Label);
                        neighbour.push((cell_start[j] + k) as crate::Label);
                    } else {
                        // Level n is the input's own face, unchanged, now
                        // internal with the layer's last cell as neighbour.
                        faces.push(face.clone());
                        owner.push(mesh.owner[f]);
                        neighbour.push((cell_start[j] + n - 1) as crate::Label);
                    }
                }
            }
        }
    }
    for (o, nb, ps) in internal_sides {
        faces.push(ps);
        owner.push(o as crate::Label);
        neighbour.push(nb as crate::Label);
    }
    // §2's upper-triangular order, restored over the whole internal block:
    // the sort runs on a copy of the face list, the owner and the
    // neighbour, so the three stay aligned.
    let perm = ldu_permutation(&owner, &neighbour)?;
    let mut sorted_faces: Vec<Vec<crate::Label>> = Vec::with_capacity(faces.len());
    let mut sorted_owner: Vec<crate::Label> = Vec::with_capacity(owner.len());
    let mut sorted_neighbour: Vec<crate::Label> = Vec::with_capacity(neighbour.len());
    for pi in perm {
        let pi = pi as usize;
        sorted_faces.push(faces[pi].clone());
        sorted_owner.push(owner[pi]);
        sorted_neighbour.push(neighbour[pi]);
    }
    let (mut faces, mut owner, neighbour) = (sorted_faces, sorted_owner, sorted_neighbour);

    // The boundary, patch by patch in the INPUT's patch order: a layer
    // patch's level-0 copies in the input's order, a non-layer patch's own
    // faces in the input's order, then the side faces (92.49) put on that
    // patch, in (layer face id, level, segment index) order.
    let mut patches_out: Vec<crate::mesh::PatchInfo> = Vec::with_capacity(mesh.patches.len());
    let mut n_boundary = 0usize;
    for (p, patch) in mesh.patches.iter().enumerate() {
        let start = n_boundary;
        if field.patches.contains(&p) {
            for jj in 0..patch.size {
                let f = n_internal + patch.start + jj;
                let j = layer_j[f] as usize;
                // (92.70)-(92.71): a KEEP or RING face's level-0 boundary
                // face, owned by its first cell (the wedge cell on RING); an
                // OFF face stays the input face it was, owner its input
                // cell, carrying no stack at all.
                if classes[j] == FaceClass::Off {
                    faces.push(mesh.faces[f].clone());
                    owner.push(mesh.owner[f]);
                    n_boundary += 1;
                    continue;
                }
                let ps: Vec<crate::Label> = mesh.faces[f]
                    .iter()
                    .map(|q| level_point[0][slot_of_point[*q as usize] as usize] as crate::Label)
                    .collect();
                faces.push(ps);
                owner.push(cell_start[j] as crate::Label);
                n_boundary += 1;
            }
        } else {
            for jj in 0..patch.size {
                let f = n_internal + patch.start + jj;
                faces.push(mesh.faces[f].clone());
                owner.push(mesh.owner[f]);
                n_boundary += 1;
            }
        }
        let mut sides = std::mem::take(&mut boundary_sides[p]);
        sides.sort_by_key(|s| (s.0, s.1, s.2));
        for s in sides {
            faces.push(s.4);
            owner.push(s.3 as crate::Label);
            n_boundary += 1;
        }
        patches_out.push(crate::mesh::PatchInfo {
            name: patch.name.clone(),
            type_name: patch.type_name.clone(),
            kind: patch.kind,
            start,
            size: n_boundary - start,
            nbr_patch: patch.nbr_patch,
        });
    }

    let out = PolyMeshRaw {
        points,
        faces,
        owner,
        neighbour,
        patches: patches_out,
    };
    // This stage's own closure check first, so a face it mis-wound is
    // blamed here rather than read by the gate as bad geometry.
    check_extrusion_closes(&out, first_cell)?;
    // §92.3's gate, MEASURED only: an attempt does not refuse on it - the
    // caller reads `quality.passed()` and retreats. `usize::MAX` lifts the
    // per-cell cap, so the report names every failing cell the ladder's
    // `failing_points` needs.
    let quality = quality::measure_capped(&out, t, usize::MAX)?;

    // (92.50), per patch, area-weighted, `A_f` off the LEVEL-0 face, and
    // `tau_f` the fraction of the nominal `T` the face actually got.
    let mut patches_rep = Vec::new();
    for &p in &named {
        let patch = &mesh.patches[p];
        match field.patches.iter().position(|&q| q == p) {
            None => {
                let drop_cause =
                    shrunk.drop_causes.iter().find(|(nm, _)| nm == &patch.name).map(|(_, c)| *c);
                let reason = match shrunk.dropped.iter().find(|(nm, _)| nm == &patch.name) {
                    Some((_, r)) => format!("patch \"{}\": {r}", patch.name),
                    None => {
                        format!("patch \"{}\": the patch carries no layer face", patch.name)
                    }
                };
                patches_rep.push(PatchLayers {
                    name: patch.name.clone(),
                    n_layers: 0,
                    n_faces: patch.size,
                    area: patch_area(mesh, n_internal, patch),
                    full_area_frac: 0.0,
                    area_frac_tau_ge: [0.0; 3],
                    face_area_tau: Vec::new(),
                    mean_frac: 0.0,
                    t1_requested: st.t[0],
                    t1_mean: 0.0,
                    t1_min: 0.0,
                    dropped: Some(reason),
                    drop_cause,
                    n_reseated_points: shrunk
                        .reseated
                        .iter()
                        .find(|(nm, _)| nm == &patch.name)
                        .map_or(0, |(_, n)| *n),
                    beta_rungs: 0,
                    level_n_non_orth_max_deg: None,
                    kept_area_frac: 0.0,
                    ring_area_frac: 0.0,
                    n_keep_faces: 0,
                    n_ring_faces: 0,
                    n_off_faces: 0,
                    n_anchored_points: 0,
                    n_merged_faces: 0,
                    merged_area_frac: 0.0,
                    n_cut_faces: 0,
                    stack_area_frac: 0.0,
                });
            }
            Some(_) => {
                let mut area = 0.0;
                let mut full = 0.0;
                let mut wsum = 0.0;
                let mut tau_min_all = Scalar::INFINITY;
                let mut nf = 0usize;
                let mut face_area_tau: Vec<(Scalar, Scalar)> = Vec::new();
                // (92.74)'s coverage, over the patch's own layer faces.
                let mut kept_a = 0.0;
                let mut ring_a = 0.0;
                let mut n_keep_f = 0usize;
                let mut n_ring_f = 0usize;
                let mut n_off_f = 0usize;
                let mut n_anch = 0usize;
                // (92.70')'s merge, and (92.70)'s cut, over the same faces.
                let mut n_merged_f = 0usize;
                let mut merged_a = 0.0;
                let mut n_cut_f = 0usize;
                let mut seen_pt = vec![false; n_points];
                for (j, &f) in field.faces.iter().enumerate() {
                    if field.face_patch[j] != p {
                        continue;
                    }
                    nf += 1;
                    if cut[f] {
                        n_cut_f += 1;
                    }
                    match classes[j] {
                        FaceClass::Keep => n_keep_f += 1,
                        FaceClass::Ring => n_ring_f += 1,
                        FaceClass::Off => n_off_f += 1,
                    }
                    let ps0: Vec<crate::Label> = mesh.faces[f]
                        .iter()
                        .map(|q| level_point[0][slot_of_point[*q as usize] as usize] as crate::Label)
                        .collect();
                    let a = face_area_vector(&out.points, &ps0).mag();
                    let tau_min = mesh
                        .faces[f]
                        .iter()
                        .map(|q| field.disp[*q as usize].mag())
                        .fold(Scalar::INFINITY, Scalar::min);
                    // A CUT face carries no stack whatever its points' pull
                    // was - the wall stepped away from it - so its tau is
                    // zero, as an all-anchored face's is.
                    let tau = if cut[f] {
                        0.0
                    } else {
                        tau_min / st.total
                    };
                    face_area_tau.push((a, tau));
                    area += a;
                    if tau >= 1.0 - 1e-9 {
                        full += a;
                    }
                    wsum += a * tau;
                    tau_min_all = tau_min_all.min(tau);
                    match classes[j] {
                        FaceClass::Keep => kept_a += a,
                        FaceClass::Ring => {
                            ring_a += a;
                            // (92.70'): a RING face with no anchored point
                            // is a face of M - one cell of the whole stack.
                            if !mesh.faces[f]
                                .iter()
                                .any(|q| anchored_pt[*q as usize])
                            {
                                n_merged_f += 1;
                                merged_a += a;
                            }
                        }
                        FaceClass::Off => {}
                    }
                    for q in &mesh.faces[f] {
                        let q = *q as usize;
                        if !seen_pt[q] {
                            seen_pt[q] = true;
                            if anchored_pt[q] {
                                n_anch += 1;
                            }
                        }
                    }
                }
                let mean_frac = if area > 0.0 { wsum / area } else { 0.0 };
                // (92.73): a patch whose every layer face is OFF has had its
                // whole stack terminated - reported, and the mesh goes on.
                let all_off = nf > 0 && n_off_f == nf;
                let kept_frac = if area > 0.0 { kept_a / area } else { 0.0 };
                let ring_frac = if area > 0.0 { ring_a / area } else { 0.0 };
                // (92.74): the share of the wall that carries the WHOLE stack
                // thickness - the KEEP faces, plus the faces of M as one
                // merged cell each.
                let stack_frac = if area > 0.0 {
                    (kept_a + merged_a) / area
                } else {
                    0.0
                };
                let mut row = PatchLayers {
                    name: patch.name.clone(),
                    n_layers: if all_off { 0 } else { n },
                    n_faces: nf,
                    area,
                    full_area_frac: if all_off || area == 0.0 {
                        0.0
                    } else {
                        full / area
                    },
                    area_frac_tau_ge: [0.0; 3],
                    face_area_tau: if all_off { Vec::new() } else { face_area_tau },
                    mean_frac: if all_off { 0.0 } else { mean_frac },
                    t1_requested: st.t[0],
                    t1_mean: if all_off { 0.0 } else { st.t[0] * mean_frac },
                    // The achieved first layer the WORST face got: the
                    // limiter's own number, in metres.
                    t1_min: if all_off {
                        0.0
                    } else {
                        st.t[0] * if area > 0.0 { tau_min_all } else { 0.0 }
                    },
                    dropped: if all_off {
                        Some(format!(
                            "patch \"{}\": every layer face terminated - anchored \
                             or cut (SPEC-LIT 92.13, (92.73))",
                            patch.name
                        ))
                    } else {
                        None
                    },
                    drop_cause: if all_off {
                        Some(DropCause::Terminated)
                    } else {
                        None
                    },
                    n_reseated_points: shrunk
                        .reseated
                        .iter()
                        .find(|(nm, _)| nm == &patch.name)
                        .map_or(0, |(_, n)| *n),
                    beta_rungs: 0,
                    level_n_non_orth_max_deg: None,
                    kept_area_frac: kept_frac,
                    ring_area_frac: ring_frac,
                    n_keep_faces: n_keep_f,
                    n_ring_faces: n_ring_f,
                    n_off_faces: n_off_f,
                    n_anchored_points: n_anch,
                    n_merged_faces: n_merged_f,
                    merged_area_frac: if area > 0.0 { merged_a / area } else { 0.0 },
                    n_cut_faces: n_cut_f,
                    stack_area_frac: stack_frac,
                };
                if !all_off {
                    row.area_frac_tau_ge = TAU_GE_BETAS.map(|b| row.frac_tau_ge(b));
                }
                patches_rep.push(row);
            }
        }
    }
    let report = LayerReport {
        patches: patches_rep,
        // (92.72): C + n |KEEP| + |RING| cells, P + n |L_c ∩ L_m| points.
        n_layer_cells: classes.iter().fold(0usize, |s, c| {
            s + match c {
                FaceClass::Keep => n,
                FaceClass::Ring => 1,
                FaceClass::Off => 0,
            }
        }),
        n_layer_points: n * n_moving,
        n_side_internal,
        n_side_boundary,
        n_split_sides,
        retreats: shrunk.retreats,
        ladder: shrunk.ladder.clone(),
    };
    let moving_slots: Vec<usize> = (0..n_l)
        .filter(|&s| copies_slot[s])
        .map(|s| slots[s])
        .collect();
    Ok((
        Layered {
            mesh: out,
            report,
            extrusion: Extrusion {
                slot_of_point,
                level_point,
                layer_faces: field.faces.clone(),
                cell_start,
                cell_count,
                moving_slots,
                first_cell,
                n,
            },
            quality,
            beta: shrunk.beta.clone(),
            reseat_points: shrunk.reseat_points.clone(),
            classes,
        },
        shrunk,
    ))
}

/// The extrusion's own check on its own output, run BEFORE §92.3's gate:
/// every cell's faces must close to 1e-12 RELATIVE - `|sum +-Sf| / sum |Sf|`
/// - which a closed polyhedron satisfies exactly whatever its shape. A cell
/// that fails did not get a bad geometry from the user: it got a face this
/// stage wound backwards or emitted with no area, and the message says so,
/// because G2's threshold is absolute against `V^(2/3)` and a small
/// mis-wound face in a large cell slips under it (SPEC-LIT 92.13).
fn check_extrusion_closes(out: &PolyMeshRaw, first_cell: usize) -> Result<()> {
    // A closed polyhedron closes exactly whatever its shape, so the bound
    // needs no threshold from the config.
    const RELATIVE_CLOSURE: Scalar = 1.0e-12;
    let n_internal = out.neighbour.len().min(out.faces.len());
    let n_cells = out
        .owner
        .iter()
        .chain(out.neighbour.iter())
        .copied()
        .max()
        .map_or(0, |c| c as usize + 1);
    let mut s = vec![Vec3::ZERO; n_cells];
    let mut a = vec![0.0; n_cells];
    // The mean of the points the cell's faces carry - message-grade only.
    let mut centroid = vec![Vec3::ZERO; n_cells];
    let mut n_pts = vec![0usize; n_cells];
    for f in 0..out.faces.len() {
        let sf = face_area_vector(&out.points, &out.faces[f]);
        let o = out.owner[f] as usize;
        s[o] = s[o] + sf;
        a[o] += sf.mag();
        for &p in &out.faces[f] {
            centroid[o] = centroid[o] + out.points[p as usize];
            n_pts[o] += 1;
        }
        if f < n_internal {
            let nb = out.neighbour[f] as usize;
            s[nb] = s[nb] - sf;
            a[nb] += sf.mag();
            for &p in &out.faces[f] {
                centroid[nb] = centroid[nb] + out.points[p as usize];
                n_pts[nb] += 1;
            }
        }
    }
    for c in 0..n_cells {
        if a[c] == 0.0 {
            continue;
        }
        let ratio = s[c].mag() / a[c];
        if ratio > RELATIVE_CLOSURE {
            let kind = if c >= first_cell {
                "a LAYER cell"
            } else {
                "an input cell"
            };
            let ctr = centroid[c] / n_pts[c] as Scalar;
            return Err(Error::Mesh(format!(
                "layers: cell {c} ({kind}) does not close: \
                 |sum +-Sf| / sum |Sf| = {ratio:.3e} > {RELATIVE_CLOSURE:.0e}; \
                 the cell sits at ({:.6}, {:.6}, {:.6}), the mean of the \
                 points its faces carry. Its faces were assembled by the \
                 layer extrusion itself, so this is a fault of the mesher, \
                 not of the geometry, and should be reported.",
                ctr.x, ctr.y, ctr.z
            )));
        }
    }
    Ok(())
}

/// The patch's total boundary area, off the INPUT mesh's own faces - what a
/// dropped row reports, and what (92.50) sums.
fn patch_area(mesh: &PolyMeshRaw, n_internal: usize, patch: &crate::mesh::PatchInfo) -> Scalar {
    let n_faces = mesh.faces.len();
    let mut area = 0.0;
    for j in 0..patch.size {
        let f = n_internal + patch.start + j;
        if f < n_faces {
            area += face_area_vector(&mesh.points, &mesh.faces[f]).mag();
        }
    }
    area
}

/// The nearest boundary-carried point within tolerance of an edge's
/// midpoint - `find_hanging`'s hash, keyed by the EDGE rather than by the
/// node, for the reason the segments above state. One tolerance for the
/// whole mesh, a bucket a thousand tolerances wide, a 27-bucket probe:
/// `find_hanging`'s shape, copied.
struct Midpoints<'a> {
    points: &'a [Vec3],
    used: &'a [bool],
    tol: Scalar,
    lo: Vec3,
    h: Scalar,
    grid: HashMap<[i64; 3], Vec<u32>>,
}

impl<'a> Midpoints<'a> {
    fn new(points: &'a [Vec3], used: &'a [bool]) -> Self {
        if points.is_empty() {
            return Self {
                points,
                used,
                tol: 1.0,
                lo: Vec3::ZERO,
                h: 1000.0,
                grid: HashMap::new(),
            };
        }
        let mut lo = points[0];
        let mut hi = points[0];
        for p in points {
            lo = lo.cmpt_min(*p);
            hi = hi.cmpt_max(*p);
        }
        let diag = (hi - lo).mag();
        let tol = 1e-9 * diag.max(1.0);
        let h = 1000.0 * tol;
        let key = |p: Vec3| -> [i64; 3] {
            [
                ((p.x - lo.x) / h).floor() as i64,
                ((p.y - lo.y) / h).floor() as i64,
                ((p.z - lo.z) / h).floor() as i64,
            ]
        };
        let mut grid: HashMap<[i64; 3], Vec<u32>> = HashMap::new();
        for (i, p) in points.iter().enumerate() {
            grid.entry(key(*p)).or_default().push(i as u32);
        }
        Self {
            points,
            used,
            tol,
            lo,
            h,
            grid,
        }
    }

    /// The nearest boundary-carried point within tolerance of the midpoint
    /// of `(a, b)`, ties to the lower point id. The endpoints themselves
    /// never come back.
    fn at(&self, a: u32, b: u32) -> Option<u32> {
        let m = (self.points[a as usize] + self.points[b as usize]) * 0.5;
        let c = [
            ((m.x - self.lo.x) / self.h).floor() as i64,
            ((m.y - self.lo.y) / self.h).floor() as i64,
            ((m.z - self.lo.z) / self.h).floor() as i64,
        ];
        let mut best = (Scalar::INFINITY, u32::MAX);
        for dz in -1..=1 {
            for dy in -1..=1 {
                for dx in -1..=1 {
                    let Some(bucket) = self.grid.get(&[c[0] + dx, c[1] + dy, c[2] + dz])
                    else {
                        continue;
                    };
                    for &i in bucket {
                        if i == a || i == b || !self.used[i as usize] {
                            continue;
                        }
                        let d = (self.points[i as usize] - m).mag();
                        if d <= self.tol && (d, i) < best {
                            best = (d, i);
                        }
                    }
                }
            }
        }
        if best.1 == u32::MAX {
            None
        } else {
            Some(best.1)
        }
    }
}

/// (92.49)'s cut: `(a, b)` is split at every boundary-carried point lying
/// at its midpoint, recursively, until no hanging node lies inside a
/// segment. The segments come back in order along `(a, b)`. The recursion
/// is capped at four cuts - generous over the one a 2:1 transition needs -
/// and a deeper chain is a refusal naming the edge.
fn split_edge(mp: &Midpoints, a: u32, b: u32, depth: u32) -> Result<Vec<(u32, u32)>> {
    let Some(m) = mp.at(a, b) else {
        return Ok(vec![(a, b)]);
    };
    if depth >= 4 {
        return Err(Error::Mesh(format!(
            "layers: the edge ({a}, {b}) - points {:?}, {:?} - needed more \
             than four cuts at hanging nodes (92.49)",
            mp.points[a as usize],
            mp.points[b as usize]
        )));
    }
    let mut segs = split_edge(mp, a, m, depth + 1)?;
    segs.extend(split_edge(mp, m, b, depth + 1)?);
    Ok(segs)
}

// ==========================================================================
//  Tests
// ==========================================================================

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::automesher::castellate::castellate;
    use crate::automesher::castellate::tests::{box_soup, sphere_soup, thresholds};
    use crate::automesher::octree::{
        patch_names, refine_to_surface, Background, Octree,
    };
    use crate::automesher::snap;
    use crate::automesher::{
        CastellationSpec, DistanceBand, DomainSpec, LayerTerminate, RefinementBand,
        RefinementSpec, SnapSpec,
    };

    /// The background over `extent` at `base`, and the tree `max_level` deep
    /// everywhere - the common front half of every case here.
    fn setup(extent: [f64; 6], base: f64, max_level: u32) -> (Octree, Background) {
        let bg = Background::from_domain(&DomainSpec {
            extent,
            base_size: base,
            grading: [1.0; 3],
        })
        .expect("background");
        let tree = Octree::uniform(bg.base_n(), max_level).expect("uniform");
        (tree, bg)
    }

    /// The cube castellated and then SNAPPED - layers are tested on a
    /// snapped mesh, the way the stage sees it.
    fn snapped_cube_case() -> (Surface, PolyMeshRaw) {
        let (mut tree, bg) = setup([0.0, 4.0, 0.0, 4.0, 0.0, 4.0], 1.0, 1);
        let surf = Surface::from_soup(
            box_soup([1.3; 3], [2.3; 3]),
            vec!["cube".to_string()],
        )
        .expect("surface");
        let spec = RefinementSpec {
            levels: vec![RefinementBand {
                patch: "cube".to_string(),
                bands: vec![DistanceBand { distance: 0.0, level: 1 }],
                feature_level: 0,
            }],
            feature_angle_deg: 30.0,
            max_level: 1,
            boxes: Vec::new(),
        };
        refine_to_surface(&mut tree, &bg, &surf, &spec).expect("refine");
        let cast = castellate(
            &tree,
            &bg,
            &surf,
            &patch_names(),
            &CastellationSpec::default(),
            &thresholds(),
        )
        .expect("castellate");
        let snapped =
            snap::snap(&cast.mesh, &surf, 1.0, 30.0, &SnapSpec::default(), &thresholds())
                .expect("snap");
        (surf, snapped.mesh)
    }

    /// A box standing on the domain floor, castellated and SNAPPED: the
    /// shape the extrusion actually meets, since stage 4 always runs before
    /// stage 6. Its convex edges are where the side quads' winding is
    /// decided.
    fn snapped_floor_box_case() -> (Surface, PolyMeshRaw) {
        let (mut tree, bg) = setup([0.0, 4.0, 0.0, 4.0, 0.0, 4.0], 1.0, 1);
        let surf = Surface::from_soup(
            box_soup([1.13, 1.13, -1.0], [2.63, 2.63, 2.13]),
            vec!["cube".to_string()],
        )
        .expect("surface");
        let spec = RefinementSpec {
            levels: vec![RefinementBand {
                patch: "cube".to_string(),
                bands: vec![DistanceBand { distance: 0.0, level: 1 }],
                feature_level: 0,
            }],
            feature_angle_deg: 30.0,
            max_level: 1,
            boxes: Vec::new(),
        };
        refine_to_surface(&mut tree, &bg, &surf, &spec).expect("refine");
        let cast = castellate(
            &tree,
            &bg,
            &surf,
            &patch_names(),
            &CastellationSpec::default(),
            &thresholds(),
        )
        .expect("castellate");
        let snapped =
            snap::snap(&cast.mesh, &surf, 1.0, 30.0, &SnapSpec::default(), &thresholds())
                .expect("snap");
        (surf, snapped.mesh)
    }

    /// The sphere castellated and then SNAPPED.
    fn snapped_sphere_case() -> (Surface, PolyMeshRaw) {
        let (mut tree, bg) = setup([0.0, 8.0, 0.0, 8.0, 0.0, 8.0], 1.0, 2);
        let surf = Surface::from_soup(
            sphere_soup(3.0, [4.0; 3]),
            vec!["sphere".to_string()],
        )
        .expect("surface");
        let spec = RefinementSpec {
            levels: vec![RefinementBand {
                patch: "sphere".to_string(),
                bands: vec![DistanceBand { distance: 0.0, level: 2 }],
                feature_level: 0,
            }],
            feature_angle_deg: 30.0,
            max_level: 2,
            boxes: Vec::new(),
        };
        refine_to_surface(&mut tree, &bg, &surf, &spec).expect("refine");
        let cast = castellate(
            &tree,
            &bg,
            &surf,
            &patch_names(),
            &CastellationSpec::default(),
            &thresholds(),
        )
        .expect("castellate");
        let snapped =
            snap::snap(&cast.mesh, &surf, 1.0, 30.0, &SnapSpec::default(), &thresholds())
                .expect("snap");
        (surf, snapped.mesh)
    }

    /// The layer spec the sphere cases run with: three layers of the
    /// nominal thickness.
    fn sphere_layers(first_thickness: f64) -> LayerSpec {
        LayerSpec {
            patches: vec!["sphere".to_string()],
            n: 3,
            first_thickness,
            growth: 1.3,
            ..LayerSpec::default()
        }
    }

    /// (92.69) against (92.42): the angle merge answers where the 1e-6
    /// dedupe pins, and agrees with it where the planes are independent.
    #[test]
    fn a_junction_normal_merges_within_the_angle_tolerance() {
        let s5 = 5.0f64.to_radians().sin();
        let c5 = 5.0f64.to_radians().cos();
        // (a) three near-parallel constraints: (92.42) pins, (92.69) merges
        // into the one plane and projects.
        let n = Vec3::new(0.6, 0.0, 0.8);
        let us = vec![
            Vec3::new(0.0, 0.0, -1.0),
            Vec3::new(s5, 0.0, -c5),
            Vec3::new(0.0, s5, -c5),
        ];
        assert_eq!(constrain_normal(n, &us, None), None);
        let got = constrain_normal(n, &us, Some(15.0)).expect("(a) merges");
        assert!((got - Vec3::new(1.0, 0.0, 0.0)).mag() < 1e-12, "{got:?}");
        // (b) three independent-ish normals within 45 deg of each other.
        let n = Vec3::new(0.3, 0.2, 0.9).normalised();
        let fr = std::f64::consts::FRAC_1_SQRT_2;
        let us = vec![
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(fr, fr, 0.0),
        ];
        assert_eq!(constrain_normal(n, &us, None), None);
        let got = constrain_normal(n, &us, Some(15.0)).expect("(b) merges");
        assert!((got - Vec3::new(0.0, 0.0, 1.0)).mag() < 1e-12, "{got:?}");
        // (c) three orthogonal constraints: both pin.
        let n = Vec3::new(1.0, 1.0, 1.0).normalised();
        let us = vec![
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
        ];
        assert_eq!(constrain_normal(n, &us, None), None);
        assert_eq!(constrain_normal(n, &us, Some(15.0)), None);
        // (d) the normal against its own plane: both pin.
        let us = vec![Vec3::new(0.0, 0.0, -1.0)];
        assert_eq!(constrain_normal(Vec3::new(0.0, 0.0, 1.0), &us, None), None);
        assert_eq!(
            constrain_normal(Vec3::new(0.0, 0.0, 1.0), &us, Some(15.0)),
            None
        );
        // (e) two orthogonal constraints: both take the cross product.
        let n = Vec3::new(0.5, 0.5, 0.7071).normalised();
        let us = vec![Vec3::new(1.0, 0.0, 0.0), Vec3::new(0.0, 1.0, 0.0)];
        for tol in [None, Some(15.0)] {
            let got = constrain_normal(n, &us, tol).expect("(e) cross");
            assert!((got - Vec3::new(0.0, 0.0, 1.0)).mag() < 1e-12, "{got:?}");
        }
        // (f) nothing constrains the point: unchanged, both.
        for tol in [None, Some(15.0)] {
            assert_eq!(constrain_normal(n, &[], tol), Some(n));
        }
    }

    /// (92.70) and (92.73): the face class reads the anchored flags, and the
    /// local step anchors a G5 point, a spent point, or one whose halving
    /// would fall under the floor.
    #[test]
    fn the_face_class_and_the_local_step() {
        let a = [false, false, false, true];
        assert_eq!(face_class(&[0, 1, 2, 3], &[false; 4]), FaceClass::Keep);
        assert_eq!(face_class(&[0, 1, 2, 3], &a), FaceClass::Ring);
        assert_eq!(face_class(&[0, 1, 2, 3], &[true; 4]), FaceClass::Off);
        assert_eq!(
            local_step(false, 0, 1.0, 0.1, 4),
            LocalStep::Halve
        );
        assert_eq!(
            local_step(false, 4, 1.0, 0.1, 4),
            LocalStep::Anchor
        );
        assert_eq!(
            local_step(false, 0, 0.15, 0.1, 4),
            LocalStep::Anchor
        );
        assert_eq!(
            local_step(false, 0, 0.2, 0.1, 4),
            LocalStep::Halve
        );
        assert_eq!(local_step(true, 0, 1.0, 0.1, 4), LocalStep::Anchor);
        assert_eq!(local_step(false, 3, 1.0, 0.0, 4), LocalStep::Halve);
        assert_eq!(local_step(false, 4, 1.0, 0.0, 4), LocalStep::Anchor);
    }

    /// (92.73)'s closure, which the pass branch anchors through: a hanging
    /// node in the set pulls its layer-point parents in, recursively, so an
    /// anchored hanging node's forced zero is the mean of anchored parents.
    #[test]
    fn the_pass_branch_anchors_a_hanging_node_with_its_parents() {
        let hanging: Vec<(u32, [u32; 2])> = vec![(5, [1, 2]), (6, [5, 3])];
        let is_layer = [true; 8];
        let mut fset = vec![6usize];
        close_under_hanging(&mut fset, &hanging, &is_layer);
        assert_eq!(fset, vec![1, 2, 3, 5, 6]);
        let mut fset = vec![4usize];
        close_under_hanging(&mut fset, &hanging, &is_layer);
        assert_eq!(fset, vec![4]);
        let mut is_layer = [true; 8];
        is_layer[3] = false;
        let mut fset = vec![6usize];
        close_under_hanging(&mut fset, &hanging, &is_layer);
        assert_eq!(fset, vec![1, 2, 5, 6]);
    }

    /// (92.73) amended: the freeze's point set - a frozen cell's points, all
    /// of them, layer or not, closed under hanging parents recursively, so
    /// the whole cell and everything hanging off it is held.
    #[test]
    fn a_frozen_cell_holds_every_point_and_its_hanging_parents() {
        let cell_points: Vec<Vec<u32>> = vec![vec![0, 1, 2, 3], vec![2, 3, 4, 5]];
        let hanging: Vec<(u32, [u32; 2])> = vec![(4, [6, 7]), (7, [8, 9])];
        // Cell 1 freezes: its four points, then 4's parents 6 and 7, then
        // 7's parents 8 and 9 - all of them, layer or not.
        assert_eq!(
            frozen_point_set(&[1], &cell_points, &hanging),
            vec![2, 3, 4, 5, 6, 7, 8, 9]
        );
        // Cell 0 freezes: no hanging node among its points, so just the four.
        assert_eq!(
            frozen_point_set(&[0], &cell_points, &hanging),
            vec![0, 1, 2, 3]
        );
        // Both at once: the shared 2 and 3 once, 0's set untouched by 4's
        // parents.
        assert_eq!(
            frozen_point_set(&[0, 1], &cell_points, &hanging),
            vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        );
    }

    /// (92.73) amended: the freeze's consecutive-failure counts move only at
    /// a measurement that FOLLOWS a local step. Three (92.66) beta-rung
    /// measurements that name cell 0 leave it at zero - a rung moves the
    /// pull, not the points, so the cells it fails are not a step's
    /// failure - and the first post-step measurement that names it puts it
    /// at ONE: under the freeze's two, so the cell is not frozen.
    #[test]
    fn beta_rungs_do_not_count_toward_a_freeze() {
        let mut counts = vec![0usize; 2];
        let cell0_fails = [true, false];
        for _ in 0..3 {
            count_cell_failures(&mut counts, &cell0_fails, false);
        }
        assert_eq!(counts[0], 0, "a beta rung neither counts nor resets");
        count_cell_failures(&mut counts, &cell0_fails, true);
        assert_eq!(counts[0], 1, "one post-step failure");
        assert!(counts[0] < 2, "cell 0 froze at its first counted failure");
        // A post-step measurement the cell passes still starts it over.
        count_cell_failures(&mut counts, &[false, true], true);
        assert_eq!(counts[0], 0);
    }

    #[test]
    fn the_module_doc_ends_with_the_provenance_line() {
        let src = include_str!("layers.rs");
        let last = src
            .lines()
            .filter(|l| l.starts_with("//!"))
            .last()
            .expect("module doc");
        assert_eq!(
            last.trim_start_matches("//!").trim(),
            "No GPL-licensed source was consulted."
        );
    }

    #[test]
    fn the_stack_sums_to_the_nominal_thickness() {
        let mut spec = LayerSpec {
            patches: vec!["sphere".to_string()],
            n: 3,
            first_thickness: 0.02,
            growth: 1.3,
            ..LayerSpec::default()
        };
        let st = stack(&spec).expect("stack");
        assert_eq!(st.n, 3);
        assert_eq!(st.t.len(), 3);
        for (k, want) in [0.02, 0.026, 0.0338].iter().enumerate() {
            assert!(
                (st.t[k] - *want).abs() < 1e-12,
                "t[{}] = {}, want {}",
                k,
                st.t[k],
                want
            );
        }
        assert!((st.total - 0.0798).abs() < 1e-12, "total = {}", st.total);
        assert_eq!(st.f.len(), 4);
        assert_eq!(st.f[0], 0.0);
        assert!(st.f[1] > st.f[0] && st.f[2] > st.f[1] && st.f[2] < 1.0);
        assert_eq!(st.f[3], 1.0);
        // Uniform growth: n copies of the first thickness.
        spec.growth = 1.0;
        let st = stack(&spec).expect("stack");
        assert!((st.total - 3.0 * 0.02).abs() < 1e-12);
        for k in 0..3 {
            assert!((st.t[k] - 0.02).abs() < 1e-12);
        }
        assert!((st.f[3] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn a_bad_stack_is_refused_by_field_name() {
        let mut spec = LayerSpec {
            patches: vec!["sphere".to_string()],
            n: 3,
            first_thickness: 0.02,
            growth: 1.3,
            ..LayerSpec::default()
        };
        spec.first_thickness = 0.0;
        let e = stack(&spec).unwrap_err();
        assert!(e.to_string().contains("first_thickness"), "{e}");
        assert!(e.to_string().contains('0'), "{e}");
        spec.first_thickness = 0.02;
        spec.growth = -1.0;
        let e = stack(&spec).unwrap_err();
        assert!(e.to_string().contains("growth"), "{e}");
        spec.growth = 1.3;
        spec.medial_frac = 2.0;
        let e = stack(&spec).unwrap_err();
        assert!(e.to_string().contains("medial_frac"), "{e}");
        assert!(e.to_string().contains('2'), "{e}");
    }

    #[test]
    fn two_plates_give_half_the_gap() {
        let g = 0.4;
        let mut soup = box_soup([0.0, 0.0, 0.0], [2.0, 2.0, 0.3]);
        soup.extend(
            box_soup([0.0, 0.0, 0.3 + g], [2.0, 2.0, 1.0])
                .into_iter()
                .map(|(p, t)| (p + 1, t)),
        );
        let surf = Surface::from_soup(
            soup,
            vec!["lower".to_string(), "upper".to_string()],
        )
        .expect("surface");
        let idx = TriIndex::new(&surf, 0.05).expect("index");
        let x = Vec3::new(1.0, 1.0, 0.3);
        let n = Vec3::new(0.0, 0.0, 1.0);
        let m = medial_distance(&idx, x, n, g);
        assert!(m.is_finite(), "the march found no second wall");
        // (92.44)'s hit needs |q - y| <= (1 - kappa) s, and the second wall
        // gives |q - y| = g - s, so the march is caught at
        // s >= g / (2 - kappa) = 8 g / 15 - kappa's margin over the medial
        // axis at g / 2, and never before it. The bisection returns the hi
        // end, within the bracket's s_max / 8 over 2^6.
        let expect = g / (2.0 - KAPPA);
        assert!(
            (m - expect).abs() <= g / 8.0 / 64.0 + 1e-9,
            "m = {m}, want {expect} to within {}",
            g / 8.0 / 64.0
        );
        assert!(m > g / 2.0, "the march stopped before the gap's midpoint");
    }

    #[test]
    fn a_convex_edge_is_not_cut() {
        let surf = Surface::from_soup(
            box_soup([1.3; 3], [2.3; 3]),
            vec!["cube".to_string()],
        )
        .expect("surface");
        let idx = TriIndex::new(&surf, 0.05).expect("index");
        // On the vertical convex edge x = y = 1.3, marching along the
        // outward diagonal: the closest point is the launch point itself,
        // so no s hits, and the layers grow.
        let x = Vec3::new(1.3, 1.3, 1.8);
        let n = Vec3::new(-1.0, -1.0, 0.0).normalised();
        let m = medial_distance(&idx, x, n, 4.0);
        assert_eq!(m, Scalar::INFINITY);
        // The middle of a flat face, marching straight out: the closest
        // point is the foot, at |q - y| = s exactly.
        let x = Vec3::new(1.8, 1.8, 2.3);
        let n = Vec3::new(0.0, 0.0, 1.0);
        let m = medial_distance(&idx, x, n, 4.0);
        assert_eq!(m, Scalar::INFINITY);
    }

    #[test]
    fn a_flat_wall_normal_is_the_face_normal() {
        let (surf, mesh) = snapped_cube_case();
        let mut spec = LayerSpec {
            patches: vec!["cube".to_string()],
            n: 1,
            first_thickness: 0.01,
            normal_passes: 0,
            ..LayerSpec::default()
        };
        spec.smoothing = 0.5;
        let st = stack(&spec).expect("stack");
        let patches = resolve_patches(&mesh, &spec).expect("patches");
        let idx = TriIndex::new(&surf, 0.05).expect("index");
        let f = field(&mesh, &idx, &patches, &spec, &st).expect("field");
        // The side each layer face belongs to, by the dominant component of
        // its outward area vector; a point qualifies when EVERY layer face
        // carrying it is of one side, so its normal is the sum of parallel
        // area vectors - the side's inward unit normal, in every bit.
        let sides: Vec<(usize, bool)> = f
            .faces
            .iter()
            .map(|&fa| {
                let sf = face_area_vector(&mesh.points, &mesh.faces[fa]);
                let c = [sf.x.abs(), sf.y.abs(), sf.z.abs()];
                let ax = if c[0] >= c[1] && c[0] >= c[2] {
                    0
                } else if c[1] >= c[2] {
                    1
                } else {
                    2
                };
                (ax, sf.component(ax) > 0.0)
            })
            .collect();
        let mut carries: Vec<Vec<usize>> = vec![Vec::new(); mesh.points.len()];
        for (fi, &fa) in f.faces.iter().enumerate() {
            for &p in &mesh.faces[fa] {
                carries[p as usize].push(fi);
            }
        }
        let mut n_qual = 0usize;
        for i in 0..mesh.points.len() {
            let fis = &carries[i];
            if fis.is_empty() {
                continue;
            }
            let (ax, sign) = sides[fis[0]];
            if !fis.iter().all(|&fi| sides[fi] == (ax, sign)) {
                continue;
            }
            let mut v = [0.0; 3];
            v[ax] = if sign { -1.0 } else { 1.0 };
            let want = Vec3::new(v[0], v[1], v[2]);
            let got = f.normal[i];
            // The dominant component is exact in every bit: the sum is the
            // side's area vector alone, and normalise divides the dust
            // away. The transverse components carry up to one ulp of dust
            // from the CENTROID inside `face_area_vector` - on a triangle
            // face the mean of three equal z does not round back to z - so
            // the whole-vector bit-for-bit equality §92.13 claims for a
            // flat wall is out of reach for triangle faces without
            // touching that helper, which this unit may not. Measured
            // worst dust on this case: 1.2e-16.
            let dom = got.component(ax);
            assert_eq!(dom, want.component(ax), "point {i} at {:?}", mesh.points[i]);
            let dust = (got - want).mag();
            assert!(
                dust <= 1e-12,
                "point {i}: dust {dust:.3e} past 1e-12, got {got:?}"
            );
            n_qual += 1;
        }
        assert!(n_qual > 0, "no point sat on a single flat side");
    }

    #[test]
    fn the_shrunk_mesh_keeps_its_topology_and_passes_the_gate() {
        let (surf, mesh) = snapped_sphere_case();
        let spec = sphere_layers(0.02);
        let shrunk = shrink(&mesh, &surf, &spec, &thresholds()).expect("shrink");
        assert_eq!(shrunk.mesh.faces, mesh.faces);
        assert_eq!(shrunk.mesh.owner, mesh.owner);
        assert_eq!(shrunk.mesh.neighbour, mesh.neighbour);
        assert_eq!(shrunk.mesh.points.len(), mesh.points.len());
        assert_eq!(shrunk.mesh.patches.len(), mesh.patches.len());
        for (a, b) in shrunk.mesh.patches.iter().zip(mesh.patches.iter()) {
            assert_eq!(a.name, b.name);
            assert_eq!(a.start, b.start);
            assert_eq!(a.size, b.size);
        }
        let rep = quality::check(&shrunk.mesh, &thresholds()).expect("gate");
        assert_eq!(rep.n_regions, 1);
        eprintln!(
            "sphere shrink: retreats {}, dropped {:?}",
            shrunk.retreats, shrunk.dropped
        );
        // Written by the supervising session: without these two, a run that
        // dropped the patch would pass every assertion above - the points
        // would simply be the input's - and the test would be measuring
        // nothing.
        assert!(shrunk.dropped.is_empty(), "dropped {:?}", shrunk.dropped);
        let moved = (0..mesh.points.len())
            .filter(|&i| shrunk.field.is_layer[i] && shrunk.field.thickness[i] > 0.0)
            .count();
        assert!(moved > 0, "no layer point carried a thickness");
        for i in 0..mesh.points.len() {
            if !shrunk.field.is_layer[i] {
                continue;
            }
            let want = mesh.points[i] + shrunk.field.disp[i];
            let got = shrunk.mesh.points[i];
            assert!(
                (got - want).mag() <= 1e-12,
                "point {i}: the field's displacement does not put the wall                  where the shrink put it ({got:?} vs {want:?})"
            );
        }
    }

    #[test]
    fn a_hanging_node_stays_on_its_parents_segment() {
        let (surf, mesh) = snapped_sphere_case();
        let spec = sphere_layers(0.02);
        let shrunk = shrink(&mesh, &surf, &spec, &thresholds()).expect("shrink");
        assert!(shrunk.dropped.is_empty(), "dropped {:?}", shrunk.dropped);
        let hanging = find_hanging(&mesh.points, &mesh.faces);
        assert!(!hanging.is_empty(), "the case has no hanging nodes to test");
        let size = 8.0;
        for (h, ab) in hanging {
            let want = (shrunk.mesh.points[ab[0] as usize]
                + shrunk.mesh.points[ab[1] as usize])
                * 0.5;
            let got = shrunk.mesh.points[h as usize];
            assert!(
                (got - want).mag() < 1e-12 * size,
                "hanging node {h}: {got:?} off its parents' midpoint {want:?}"
            );
        }
    }

    #[test]
    fn a_patch_name_that_is_not_a_patch_is_refused() {
        let (_surf, mesh) = snapped_cube_case();
        let mut spec = LayerSpec::default();
        spec.patches = vec!["not_a_patch".to_string()];
        spec.n = 2;
        let e = resolve_patches(&mesh, &spec).unwrap_err();
        assert!(e.to_string().contains("not_a_patch"), "{e}");
    }

    #[test]
    fn no_layers_is_the_identity() {
        let (surf, mesh) = snapped_sphere_case();
        let mut spec = sphere_layers(0.02);
        spec.n = 0;
        let shrunk = shrink(&mesh, &surf, &spec, &thresholds()).expect("shrink");
        assert_eq!(shrunk.mesh.points, mesh.points);
        assert_eq!(shrunk.mesh.faces, mesh.faces);
        assert_eq!(shrunk.mesh.owner, mesh.owner);
        assert_eq!(shrunk.mesh.neighbour, mesh.neighbour);
        assert_eq!(shrunk.mesh.patches.len(), mesh.patches.len());
        assert!(shrunk.field.patches.is_empty());
        assert_eq!(shrunk.retreats, 0);
        assert!(shrunk.dropped.is_empty());
    }

    #[test]
    fn the_thickness_never_exceeds_half_the_local_cell() {
        let (surf, mesh) = snapped_sphere_case();
        // `min_thickness = 0` on purpose: at `first_thickness = 1.0` the
        // nominal T is 4.3 and the cell limit is about 0.125, so the default
        // floor of 0.1 T would drop the patch under (92.47) and the assertion
        // below would range over an empty set. What this case is for is that
        // the CELL limit binds and the mesh survives it, so the floor is
        // taken off and the layers are kept, thin. (Supervising session: the
        // coding agent reported this test as vacuous and it was.)
        let mut spec = sphere_layers(1.0);
        spec.min_thickness = 0.0;
        let shrunk = shrink(&mesh, &surf, &spec, &thresholds()).expect("shrink");
        // h_i again, on the input mesh, over the field's own layer faces.
        let mut h = vec![Scalar::INFINITY; mesh.points.len()];
        for &fa in &shrunk.field.faces {
            let face = &mesh.faces[fa];
            for k in 0..face.len() {
                let a = face[k] as usize;
                let b = face[(k + 1) % face.len()] as usize;
                let d = (mesh.points[a] - mesh.points[b]).mag();
                h[a] = h[a].min(d);
                h[b] = h[b].min(d);
            }
        }
        let mut n_thick = 0usize;
        for i in 0..mesh.points.len() {
            if shrunk.field.is_layer[i] {
                assert!(
                    shrunk.field.thickness[i] <= 0.5 * h[i] + 1e-12,
                    "T[{}] = {} exceeds cell_frac * h_i = {}",
                    i,
                    shrunk.field.thickness[i],
                    0.5 * h[i]
                );
                if shrunk.field.thickness[i] > 0.0 {
                    n_thick += 1;
                }
            }
        }
        eprintln!(
            "huge-thickness shrink: {n_thick} thick points, retreats {}, \
             dropped {:?}",
            shrunk.retreats, shrunk.dropped
        );
        assert!(
            shrunk.dropped.is_empty(),
            "the patch was dropped, so the bound above ranged over \n             nothing: {:?}",
            shrunk.dropped
        );
        assert!(n_thick > 0, "no layer point kept a positive thickness");
        let rep = quality::check(&shrunk.mesh, &thresholds()).expect("gate");
        assert!(rep.passed());
        assert_eq!(rep.n_regions, 1);
    }

    // ---- AM-U6b: the extrusion --------------------------------------

    use crate::io::polymesh::build_host_mesh;

    /// The cube castellated but NOT snapped: snap is what grinds grazing
    /// slivers into the wall - sub-cell edges a tenth of a cell wide - and
    /// the layer cells that grow behind them then fail the gate's G4 in
    /// droves (360 faces on the sphere, 6 on the cube, at any thickness or
    /// smoothing the spec offers). The castellated wall lies flat on the
    /// cell planes, the layer cells behind it are clean prisms, and the
    /// gate passes with room to spare (33.9 deg worst non-orthogonality on
    /// the cube). The curve cases keep their snap tests in the shrink half
    /// of the module; the extrusion's own promises are tested where they
    /// can hold.
    fn castellated_cube_case() -> (Surface, PolyMeshRaw) {
        let (mut tree, bg) = setup([0.0, 4.0, 0.0, 4.0, 0.0, 4.0], 1.0, 1);
        // The cube's faces sit ON the level-1 cell planes (0.5), so the
        // castellation cuts nothing: the body comes out an exact 3x3x3
        // stack of cells, and each face carries a 3x3 grid of wall faces.
        // That middle matters: on a cube that straddles the planes the
        // castellated wall is ONE cell, every wall point is a corner of
        // three walls at once, and (92.40)'s average gives each point a
        // diagonal normal - no face would carry a single direction at all.
        let surf = Surface::from_soup(
            box_soup([1.0, 1.0, 1.0], [2.5, 2.5, 2.5]),
            vec!["cube".to_string()],
        )
        .expect("surface");
        let spec = RefinementSpec {
            levels: vec![RefinementBand {
                patch: "cube".to_string(),
                bands: vec![DistanceBand { distance: 0.0, level: 1 }],
                feature_level: 0,
            }],
            feature_angle_deg: 30.0,
            max_level: 1,
            boxes: Vec::new(),
        };
        refine_to_surface(&mut tree, &bg, &surf, &spec).expect("refine");
        let cast = castellate(
            &tree,
            &bg,
            &surf,
            &patch_names(),
            &CastellationSpec::default(),
            &thresholds(),
        )
        .expect("castellate");
        (surf, cast.mesh)
    }

    /// The gap case: two slabs in a [0,5]^3 domain at base size 1, no
    /// refinement. Each slab must hold a background leaf CENTRE for the
    /// leaf classification to keep any cell and give it walls at all
    /// ((92.23) classifies by leaf centre), so the slabs sit astride the
    /// cell planes z = 2 and z = 3: the fluid between the facing walls is
    /// the one cell z in [2, 3], the walls 1.0 apart. Castellated, not
    /// snapped: the wall faces lie on the cell planes, which is all the
    /// shrink and the extrusion read.
    fn castellated_gap_case() -> (Surface, PolyMeshRaw) {
        let (tree, bg) = setup([0.0, 5.0, 0.0, 5.0, 0.0, 5.0], 1.0, 1);
        let mut soup = box_soup([1.0, 1.0, 1.05], [3.0, 3.0, 1.55]);
        soup.extend(
            box_soup([1.0, 1.0, 3.45], [3.0, 3.0, 3.95])
                .into_iter()
                .map(|(p, t)| (p + 1, t)),
        );
        let surf = Surface::from_soup(
            soup,
            vec!["lower".to_string(), "upper".to_string()],
        )
        .expect("surface");
        let cast = castellate(
            &tree,
            &bg,
            &surf,
            &patch_names(),
            &CastellationSpec::default(),
            &thresholds(),
        )
        .expect("castellate");
        (surf, cast.mesh)
    }

    /// The gap spec: a nominal first thickness the cell limit and the
    /// march's medial limit both clamp - the wall cells are 1.0 across, so
    /// the cell limit is 0.5 a wall, and the wall cell between the slabs
    /// puts the medial axis within 0.5 * g of each wall ((92.45)'s bound).
    fn gap_layers(min_thickness: f64) -> LayerSpec {
        LayerSpec {
            patches: vec!["lower".to_string(), "upper".to_string()],
            n: 3,
            first_thickness: 0.4,
            growth: 1.3,
            min_thickness,
            ..LayerSpec::default()
        }
    }

    /// The cube spec the extrusion tests run with: three thin layers, the
    /// flat-wall normals left exact, the thickness floor off so the patch
    /// always survives the shrink.
    fn cube_layers(first_thickness: f64) -> LayerSpec {
        LayerSpec {
            patches: vec!["cube".to_string()],
            n: 3,
            first_thickness,
            normal_passes: 0,
            min_thickness: 0.0,
            ..LayerSpec::default()
        }
    }

    fn cell_count(m: &PolyMeshRaw) -> usize {
        m.owner
            .iter()
            .chain(m.neighbour.iter())
            .copied()
            .max()
            .map_or(0, |x| x as usize + 1)
    }

    /// (92.45')'s owner heights: per point the smallest owner-cell height
    /// `V_own / |Sf|` over the layer faces carrying it, INFINITY off them.
    /// On the castellated cube every owner cell of the cube patch is an
    /// exact 0.5 m cube, so every cube layer point reads `V/A` =
    /// 0.125 / 0.25 = 0.5 to 1e-12 - and there the owner height EQUALS
    /// (92.45)'s shortest edge, which is why the face-mode run of
    /// `face_mode_with_nothing_anchored_is_patch_mode` stays bit for bit.
    #[test]
    fn the_owner_height_limits_a_point_over_a_flat_cell() {
        let (_surf, mesh) = castellated_cube_case();
        let spec = cube_layers(0.02);
        let patches = resolve_patches(&mesh, &spec).expect("patches");
        let n_internal = mesh.neighbour.len().min(mesh.faces.len());
        let mut faces = Vec::new();
        for &p in &patches {
            let patch = &mesh.patches[p];
            for j in 0..patch.size {
                let f = n_internal + patch.start + j;
                if f < mesh.faces.len() {
                    faces.push(f);
                }
            }
        }
        assert_eq!(faces.len(), 54, "the 3x3 faces of the cube's six walls");
        let hh = owner_heights(&mesh, &faces).expect("heights");
        assert_eq!(hh.len(), mesh.points.len());
        let mut n_pts = 0usize;
        for (i, &hh_i) in hh.iter().enumerate() {
            if hh_i.is_finite() {
                n_pts += 1;
                assert!(
                    (hh_i - 0.5).abs() <= 1e-12,
                    "H[{i}] = {hh_i}, want 0.5"
                );
            }
        }
        assert!(n_pts > 0, "no layer point carried an owner height");
        // Off the layer faces the helper sets no bound at all.
        assert!(hh.iter().filter(|h| h.is_infinite()).count() > 0);
    }

    /// (92.70)-(92.72) on the castellated cube: one anchored point rings its
    /// four faces, every cell closes, and the counts are (92.72)'s.
    #[test]
    fn an_anchored_point_rings_its_four_faces_and_every_cell_closes() {
        let (surf, mesh) = castellated_cube_case();
        let mut spec = cube_layers(0.02);
        spec.terminate = LayerTerminate::Face;
        // The patch-mode run reads F and |L| off, per the brief.
        let base = add_layers(&mesh, &surf, &cube_layers(0.02), &thresholds())
            .expect("patch mode");
        let big_f = base.report.patches[0].n_faces;
        let big_l = base.report.n_layer_points / spec.n;
        let big_c = cell_count(&mesh);
        let n_internal = mesh.neighbour.len().min(mesh.faces.len());
        let patches = resolve_patches(&mesh, &spec).expect("patches");
        let n_pts = mesh.points.len();
        let mut caps = vec![1.0 as Scalar; n_pts];
        let i0 = (0..n_pts)
            .find(|&i| (mesh.points[i] - Vec3::new(1.5, 1.5, 2.5)).mag() < 1e-12)
            .expect("the interior top point");
        caps[i0] = 0.0;
        let ones = vec![1.0 as Scalar; n_pts];
        let out = attempt(
            &mesh,
            &surf,
            &spec,
            &thresholds(),
            &patches,
            &caps,
            &ones,
            &vec![false; mesh.faces.len()],
            &vec![false; mesh.faces.len()],
        )
        .expect("attempt");
        assert!(out.quality.passed(), "{:?}", out.quality.max_non_orth_deg);
        let anchored: Vec<bool> = (0..n_pts)
            .map(|i| out.extrusion.slot_of_point[i] >= 0 && caps[i] == 0.0)
            .collect();
        let row = &out.report.patches[0];
        assert_eq!(row.n_ring_faces, 4, "{row:?}");
        assert_eq!(row.n_off_faces, 0);
        assert_eq!(row.n_keep_faces, big_f - 4);
        assert_eq!(row.n_anchored_points, 1);
        assert_eq!(cell_count(&out.mesh), big_c + 3 * (big_f - 4) + 4);
        assert_eq!(out.mesh.points.len(), n_pts + spec.n * (big_l - 1));
        // The RING cells, and the merged faces between two of them.
        let mut ring_cells = std::collections::HashSet::new();
        for (j, &f) in out.extrusion.layer_faces.iter().enumerate() {
            if face_class(&mesh.faces[f], &anchored) == FaceClass::Ring {
                for k in 0..out.extrusion.cell_count[j] {
                    ring_cells.insert(out.extrusion.cell_start[j] + k);
                }
            }
        }
        let n_internal_out = out.mesh.neighbour.len().min(out.mesh.faces.len());
        let mut merged = 0usize;
        for f in 0..n_internal_out {
            let (o, nb) = (out.mesh.owner[f] as usize, out.mesh.neighbour[f] as usize);
            if ring_cells.contains(&o) && ring_cells.contains(&nb) {
                merged += 1;
                assert_eq!(out.mesh.faces[f].len(), 5, "face {f}");
            }
        }
        assert_eq!(merged, 4, "four RING-RING merged faces");
        // (92.74): the ring is the four faces carrying i0, of the patch.
        assert!((row.kept_area_frac + row.ring_area_frac - 1.0).abs() < 1e-12);
        let pidx = mesh
            .patches
            .iter()
            .position(|p| p.name == "cube")
            .expect("the cube patch");
        let patch = &mesh.patches[pidx];
        let mut ring_area = 0.0;
        let mut n_ring = 0usize;
        for j in 0..patch.size {
            let f = n_internal + patch.start + j;
            if mesh.faces[f].iter().any(|&p| p as usize == i0) {
                ring_area += face_area_vector(&mesh.points, &mesh.faces[f]).mag();
                n_ring += 1;
            }
        }
        assert_eq!(n_ring, 4);
        assert!((row.ring_area_frac - ring_area / row.area).abs() < 1e-12);
    }

    /// (92.13): a side face whose area is NONZERO but under the
    /// `1e-14 * diag2` guard is EMITTED in face mode - the gate of §92.3
    /// judges its cells, and (92.73)'s ladder cuts their faces if they fail
    /// - and is REFUSED in patch mode, the thickness going to zero where it
    /// was not allowed to. A cap of `1e-12` is not an anchor (`0.0` is):
    /// the point keeps level copies a hair apart. The sliver lives on the
    /// segment BETWEEN two such points - a side face is the segment's
    /// length times the stack at it, so one hair-thin point alone still
    /// has full-thickness sides along its neighbours, the shape of the
    /// full-size tunnel's sliver edge of ~1e-7 m carrying a ~4e-4 m stack.
    #[test]
    fn a_tiny_side_face_is_emitted_in_face_mode_and_refused_in_patch_mode() {
        let (surf, mesh) = castellated_cube_case();
        let mut spec = cube_layers(0.02);
        spec.terminate = LayerTerminate::Face;
        let patches = resolve_patches(&mesh, &spec).expect("patches");
        let n_pts = mesh.points.len();
        let mut caps = vec![1.0 as Scalar; n_pts];
        let i0 = (0..n_pts)
            .find(|&i| (mesh.points[i] - Vec3::new(1.5, 1.5, 2.5)).mag() < 1e-12)
            .expect("the interior top point");
        let i1 = (0..n_pts)
            .find(|&i| (mesh.points[i] - Vec3::new(2.0, 1.5, 2.5)).mag() < 1e-12)
            .expect("the neighbour on the segment");
        caps[i0] = 1e-12;
        caps[i1] = 1e-12;
        let ones = vec![1.0 as Scalar; n_pts];
        let no_marks = vec![false; mesh.faces.len()];
        let out = attempt(
            &mesh,
            &surf,
            &spec,
            &thresholds(),
            &patches,
            &caps,
            &ones,
            &no_marks,
            &no_marks,
        )
        .expect("face mode emits a side face of nonzero area");
        let row = &out.report.patches[0];
        assert_eq!(row.n_keep_faces, row.n_faces, "{row:?}");
        assert_eq!(row.n_ring_faces, 0, "{row:?}");
        assert_eq!(row.n_cut_faces, 0, "{row:?}");
        // (92.54) ran inside `attempt` - its Ok is every cell closing.
        let err = attempt(
            &mesh,
            &surf,
            &cube_layers(0.02),
            &thresholds(),
            &patches,
            &caps,
            &ones,
            &no_marks,
            &no_marks,
        )
        .expect_err("patch mode refuses the same call");
        assert!(err.to_string().contains("has area"), "{err}");
    }

    /// (92.70)-(92.72) with a whole face OFF: the input face stays the
    /// boundary face it was, and no KEEP face touches it.
    #[test]
    fn a_face_whose_points_are_all_anchored_is_left_off() {
        let (surf, mesh) = castellated_cube_case();
        let mut spec = cube_layers(0.02);
        spec.terminate = LayerTerminate::Face;
        let base = add_layers(&mesh, &surf, &cube_layers(0.02), &thresholds())
            .expect("patch mode");
        let big_f = base.report.patches[0].n_faces;
        let big_l = base.report.n_layer_points / spec.n;
        let big_c = cell_count(&mesh);
        let n_internal = mesh.neighbour.len().min(mesh.faces.len());
        let patches = resolve_patches(&mesh, &spec).expect("patches");
        let n_pts = mesh.points.len();
        let mut caps = vec![1.0 as Scalar; n_pts];
        let mut anchored_ids = Vec::new();
        for (i, p) in mesh.points.iter().enumerate() {
            if (1.45..2.05).contains(&p.x)
                && (1.45..2.05).contains(&p.y)
                && (p.z - 2.5).abs() < 1e-12
            {
                caps[i] = 0.0;
                anchored_ids.push(i);
            }
        }
        assert_eq!(anchored_ids.len(), 4, "the four interior top points");
        let ones = vec![1.0 as Scalar; n_pts];
        let out = attempt(
            &mesh,
            &surf,
            &spec,
            &thresholds(),
            &patches,
            &caps,
            &ones,
            &vec![false; mesh.faces.len()],
            &vec![false; mesh.faces.len()],
        )
        .expect("attempt");
        assert!(out.quality.passed(), "{:?}", out.quality.max_non_orth_deg);
        let anchored: Vec<bool> = (0..n_pts)
            .map(|i| out.extrusion.slot_of_point[i] >= 0 && caps[i] == 0.0)
            .collect();
        let row = &out.report.patches[0];
        assert_eq!(row.n_off_faces, 1, "{row:?}");
        assert_eq!(row.n_ring_faces, 8);
        assert_eq!(row.n_keep_faces, big_f - 9);
        assert_eq!(row.n_anchored_points, 4);
        assert_eq!(cell_count(&out.mesh), big_c + 3 * (big_f - 9) + 8);
        assert_eq!(out.mesh.points.len(), n_pts + spec.n * (big_l - 4));
        // The OFF face: the input face whose every point is anchored, back as
        // a boundary face with its input point list and its input owner.
        let pidx = mesh
            .patches
            .iter()
            .position(|p| p.name == "cube")
            .expect("the cube patch");
        let patch = &mesh.patches[pidx];
        let mut off_input = None;
        for j in 0..patch.size {
            let f = n_internal + patch.start + j;
            if mesh.faces[f].iter().all(|&p| anchored[p as usize]) {
                off_input = Some(f);
            }
        }
        let off_input = off_input.expect("the all-anchored input face");
        let n_internal_out = out.mesh.neighbour.len().min(out.mesh.faces.len());
        let off_out = n_internal_out + patch.start
            + (off_input - n_internal - patch.start);
        assert_eq!(out.mesh.faces[off_out], mesh.faces[off_input]);
        assert_eq!(out.mesh.owner[off_out], mesh.owner[off_input]);
        // No KEEP face shares a point with the OFF face: RING cells and the
        // input's own cells are all that reach an anchored point.
        let anchored_set: std::collections::HashSet<usize> =
            anchored_ids.iter().copied().collect();
        let mut ring_or_input = ring_cells_of(&mesh, &out.extrusion, &anchored);
        for c in 0..big_c {
            ring_or_input.insert(c);
        }
        for (fi, face) in out.mesh.faces.iter().enumerate() {
            if !face.iter().any(|p| anchored_set.contains(&(*p as usize))) {
                continue;
            }
            let o = out.mesh.owner[fi] as usize;
            let nb = out.mesh.neighbour.get(fi).map(|v| *v as usize);
            assert!(
                ring_or_input.contains(&o)
                    && nb.map_or(true, |nb| ring_or_input.contains(&nb)),
                "face {fi} at an anchored point is not KEEP-made"
            );
        }
    }

    /// The RING cells of an extrusion, from the anchored flags.
    fn ring_cells_of(
        mesh: &PolyMeshRaw,
        ex: &Extrusion,
        anchored: &[bool],
    ) -> std::collections::HashSet<usize> {
        let mut out = std::collections::HashSet::new();
        for (j, &f) in ex.layer_faces.iter().enumerate() {
            if face_class(&mesh.faces[f], anchored) == FaceClass::Ring {
                for k in 0..ex.cell_count[j] {
                    out.insert(ex.cell_start[j] + k);
                }
            }
        }
        out
    }

    /// (92.70')'s merge: a KEEP face in M carries ONE cell of the whole
    /// stack with NO anchored point - the G5-failing stack becomes a single
    /// fat cell, and no point of it moves.
    #[test]
    fn a_merged_face_carries_one_cell_of_the_whole_stack() {
        let (surf, mesh) = castellated_cube_case();
        let mut spec = cube_layers(0.02);
        spec.terminate = LayerTerminate::Face;
        let base = add_layers(&mesh, &surf, &cube_layers(0.02), &thresholds())
            .expect("patch mode");
        let big_f = base.report.patches[0].n_faces;
        let big_l = base.report.n_layer_points / spec.n;
        let big_c = cell_count(&mesh);
        let patches = resolve_patches(&mesh, &spec).expect("patches");
        let n_pts = mesh.points.len();
        // The top side's centre face: [1.5, 2.0]^2 x {2.5}.
        let n_internal = mesh.neighbour.len().min(mesh.faces.len());
        let centre = (n_internal..mesh.faces.len())
            .find(|&f| {
                let face = &mesh.faces[f];
                face.iter().all(|&p| {
                    let q = mesh.points[p as usize];
                    (q.z - 2.5).abs() < 1e-12
                        && (1.45..2.05).contains(&q.x)
                        && (1.45..2.05).contains(&q.y)
                })
            })
            .expect("the top centre face");
        let mut merged = vec![false; mesh.faces.len()];
        merged[centre] = true;
        let ones = vec![1.0 as Scalar; n_pts];
        let caps = vec![1.0 as Scalar; n_pts];
        let none = vec![false; mesh.faces.len()];
        let out = attempt(
            &mesh, &surf, &spec, &thresholds(), &patches, &caps, &ones, &merged, &none,
        )
        .expect("attempt");
        assert!(out.quality.passed(), "{:?}", out.quality.max_non_orth_deg);
        let row = &out.report.patches[0];
        assert_eq!(row.n_ring_faces, 1, "{row:?}");
        assert_eq!(row.n_merged_faces, 1, "{row:?}");
        assert_eq!(row.n_keep_faces, big_f - 1);
        assert_eq!(row.n_anchored_points, 0);
        assert_eq!(cell_count(&out.mesh), big_c + 3 * (big_f - 1) + 1);
        assert_eq!(out.mesh.points.len(), n_pts + spec.n * big_l);
        // The merged cell: exactly level 0, level n, and three side pieces
        // against each of its four KEEP neighbours.
        let j = out
            .extrusion
            .layer_faces
            .iter()
            .position(|&f| f == centre)
            .expect("the merged face's block");
        assert_eq!(out.extrusion.cell_count[j], 1);
        let cell = out.extrusion.cell_start[j];
        let mut n_faces_of_cell = 0usize;
        for (fi, _) in out.mesh.faces.iter().enumerate() {
            let o = out.mesh.owner[fi] as usize;
            let nb = out.mesh.neighbour.get(fi).map(|v| *v as usize);
            if o == cell || nb == Some(cell) {
                n_faces_of_cell += 1;
            }
        }
        assert_eq!(n_faces_of_cell, 2 + 4 * 3, "the merged cell's faces");
    }

    /// (92.70) amended: a CUT face goes OFF with no anchored point at all -
    /// the input face stays a boundary face of its patch, one step down the
    /// stack, and the sides its neighbours had against it become boundary
    /// faces of the step, on the cut face's own patch.
    #[test]
    fn a_cut_face_leaves_its_input_face_and_a_step_of_boundary_sides() {
        let (surf, mesh) = castellated_cube_case();
        let mut spec = cube_layers(0.02);
        spec.terminate = LayerTerminate::Face;
        let base = add_layers(&mesh, &surf, &cube_layers(0.02), &thresholds())
            .expect("patch mode");
        let big_f = base.report.patches[0].n_faces;
        let big_l = base.report.n_layer_points / spec.n;
        let big_c = cell_count(&mesh);
        let n_internal = mesh.neighbour.len().min(mesh.faces.len());
        let patches = resolve_patches(&mesh, &spec).expect("patches");
        let n_pts = mesh.points.len();
        let ones = vec![1.0 as Scalar; n_pts];
        let caps = vec![1.0 as Scalar; n_pts];
        // The top side's centre face: [1.5, 2.0]^2 x {2.5}, T7's face - the
        // one stack whose cells G5 refuse at this thickness.
        let centre = (n_internal..mesh.faces.len())
            .find(|&f| {
                let face = &mesh.faces[f];
                face.iter().all(|&p| {
                    let q = mesh.points[p as usize];
                    (q.z - 2.5).abs() < 1e-12
                        && (1.45..2.05).contains(&q.x)
                        && (1.45..2.05).contains(&q.y)
                })
            })
            .expect("the top centre face");
        let none: Vec<bool> = vec![false; mesh.faces.len()];
        let keep = attempt(
            &mesh, &surf, &spec, &thresholds(), &patches, &caps, &ones, &none, &none,
        )
        .expect("all KEEP");
        let mut cut = vec![false; mesh.faces.len()];
        cut[centre] = true;
        let out = attempt(
            &mesh, &surf, &spec, &thresholds(), &patches, &caps, &ones, &none, &cut,
        )
        .expect("cut");
        assert!(out.quality.passed(), "{:?}", out.quality.max_non_orth_deg);
        let row = &out.report.patches[0];
        assert_eq!(row.n_off_faces, 1, "{row:?}");
        assert_eq!(row.n_cut_faces, 1, "{row:?}");
        assert_eq!(row.n_ring_faces, 0, "{row:?}");
        assert_eq!(row.n_keep_faces, big_f - 1, "{row:?}");
        assert_eq!(row.n_anchored_points, 0, "{row:?}");
        assert_eq!(cell_count(&out.mesh), big_c + 3 * (big_f - 1), "{row:?}");
        assert_eq!(out.mesh.points.len(), n_pts + 3 * big_l);
        assert!(
            (row.stack_area_frac - row.kept_area_frac).abs() < 1e-12,
            "{row:?}"
        );
        // The cut face is back on the cube patch, its INPUT point list and
        // input owner, and the cube patch carries exactly the step's sides
        // more than the all-KEEP run: 4 segments x 3 levels.
        let pidx = mesh
            .patches
            .iter()
            .position(|p| p.name == "cube")
            .expect("the cube patch");
        let n_internal_out = out.mesh.neighbour.len().min(out.mesh.faces.len());
        let opatch = &out.mesh.patches[pidx];
        let keep_patch = &keep.mesh.patches[pidx];
        assert_eq!(opatch.size - keep_patch.size, 4 * 3, "{opatch:?}");
        let mut cut_out = None;
        for b in 0..opatch.size {
            let fi = n_internal_out + opatch.start + b;
            if out.mesh.faces[fi] == mesh.faces[centre] {
                cut_out = Some(fi);
            }
        }
        let cut_out = cut_out.expect("the cut face back on the cube patch");
        assert_eq!(out.mesh.owner[cut_out], mesh.owner[centre]);
    }

    /// (92.73) amended: the OUTER ladder in face mode moves no point - a
    /// step only merges and cuts - so no entry is a retreat, every terminate
    /// entry anchors nothing and names its gates, the trace ends on an outer
    /// pass, and one patch set takes at most TERMINATE_STEP_LIMIT entries.
    #[test]
    fn the_outer_ladder_in_face_mode_moves_no_point() {
        let (surf, mesh) = snapped_cube_case();
        let spec = LayerSpec {
            patches: vec!["cube".to_string()],
            n: 3,
            first_thickness: 0.06,
            growth: 1.3,
            min_thickness: 0.0,
            terminate: LayerTerminate::Face,
            ..LayerSpec::default()
        };
        let out =
            add_layers(&mesh, &surf, &spec, &thresholds()).expect("layers");
        let last = out.report.ladder.last().expect("a trace");
        assert!(
            last.ladder == Ladder::Outer && last.outcome == Outcome::Pass,
            "{last:?}"
        );
        let mut per_set = 0usize;
        let mut max_per_set = 0usize;
        for e in &out.report.ladder {
            if e.ladder != Ladder::Outer {
                continue;
            }
            if e.outcome == Outcome::GiveUp {
                per_set = 0;
                continue;
            }
            per_set += 1;
            max_per_set = max_per_set.max(per_set);
            assert!(e.outcome != Outcome::Retreat, "{e:?}");
            if e.outcome == Outcome::Terminate {
                assert_eq!(e.terminate_points, 0, "{e:?}");
                assert!(!e.gates.is_empty(), "{e:?}");
            }
        }
        assert!(max_per_set <= TERMINATE_STEP_LIMIT, "{max_per_set}");
        let row = out
            .report
            .patches
            .iter()
            .find(|p| p.name == "cube")
            .expect("the row");
        eprintln!("face outer row: {row:?}");
        for e in &out.report.ladder {
            eprintln!("face outer ladder {e:?}");
        }
    }

    /// (92.73) amended: EVERY per-patch-set counter, M and X reset when a
    /// patch drops - one function zeroes the lot, so the next patch set
    /// starts clean and never at the last set's step limit.
    #[test]
    fn a_patch_drop_resets_the_face_mode_counters() {
        let n_points = 7usize;
        let n_faces = 5usize;
        let mut fl = FaceLadder::new(n_points, n_faces);
        fl.caps[3] = 0.5;
        fl.betas[4] = 0.25;
        fl.merged[1] = true;
        fl.cut[2] = true;
        fl.out_h[5] = 2;
        fl.halvings = 1;
        fl.steps = TERMINATE_STEP_LIMIT;
        fl.beta_rungs = 1;
        fl.reset(n_points, n_faces);
        assert!(fl.caps.iter().all(|&c| c == 1.0));
        assert!(fl.betas.iter().all(|&b| b == 1.0));
        assert!(fl.merged.iter().all(|m| !m));
        assert!(fl.cut.iter().all(|c| !c));
        assert!(fl.out_h.iter().all(|&h| h == 0));
        assert_eq!(fl.halvings, 0);
        assert_eq!(fl.steps, 0);
        assert_eq!(fl.beta_rungs, 0);
    }

    /// (92.73) on the snapped cube: face mode keeps a ring of stack where
    /// patch mode gave the whole patch up.
    #[test]
    fn the_snapped_cube_in_face_mode_keeps_its_layers_where_it_can() {
        let (surf, mesh) = snapped_cube_case();
        let spec = LayerSpec {
            patches: vec!["cube".to_string()],
            n: 3,
            first_thickness: 0.06,
            growth: 1.3,
            min_thickness: 0.0,
            terminate: LayerTerminate::Face,
            ..LayerSpec::default()
        };
        let out =
            add_layers(&mesh, &surf, &spec, &thresholds()).expect("layers");
        assert!(
            out.quality.passed(),
            "max non-orth {:?}",
            out.quality.max_non_orth_deg
        );
        let last = out.report.ladder.last().expect("a trace");
        assert!(
            last.ladder == Ladder::Outer && last.outcome == Outcome::Pass,
            "{last:?}"
        );
        // (92.73) amended: the OUTER ladder moves no point - a step merges
        // and cuts, and anchors nothing.
        for e in &out.report.ladder {
            if e.ladder == Ladder::Outer && e.outcome == Outcome::Terminate {
                assert_eq!(e.terminate_points, 0, "{e:?}");
                assert!(!e.gates.is_empty(), "{e:?}");
            }
        }
        let row = out
            .report
            .patches
            .iter()
            .find(|p| p.name == "cube")
            .expect("the row");
        assert_eq!(
            row.n_keep_faces + row.n_ring_faces + row.n_off_faces,
            row.n_faces,
            "{row:?}"
        );
        assert!(row.kept_area_frac + row.ring_area_frac <= 1.0 + 1e-12);
        for e in &out.report.ladder {
            eprintln!("face cube ladder {e:?}");
        }
        eprintln!("face cube row: {row:?}");
        assert!(row.kept_area_frac > 0.0, "{row:?}");
    }

    /// (92.73) on the snapped floor box, whose pull is what fails: face mode
    /// anchors its way to a mesh the gate takes.
    #[test]
    fn the_floor_box_in_face_mode_meets_the_ground() {
        let (surf, mesh) = snapped_floor_box_case();
        let spec = LayerSpec {
            patches: vec!["cube".to_string()],
            n: 3,
            first_thickness: 0.06,
            growth: 1.3,
            min_thickness: 0.0,
            terminate: LayerTerminate::Face,
            ..LayerSpec::default()
        };
        let out =
            add_layers(&mesh, &surf, &spec, &thresholds()).expect("layers");
        assert!(
            out.quality.passed(),
            "max non-orth {:?}",
            out.quality.max_non_orth_deg
        );
        let last = out.report.ladder.last().expect("a trace");
        assert!(
            last.ladder == Ladder::Outer && last.outcome == Outcome::Pass,
            "{last:?}"
        );
        let row = out
            .report
            .patches
            .iter()
            .find(|p| p.name == "cube")
            .expect("the row");
        assert_eq!(
            row.n_keep_faces + row.n_ring_faces + row.n_off_faces,
            row.n_faces,
            "{row:?}"
        );
        assert!(row.kept_area_frac > 0.0, "{row:?}");
        eprintln!("face floor row: {row:?}");
        for e in &out.report.ladder {
            eprintln!("face floor ladder {e:?}");
        }
    }

    /// R8: with nothing anchored, face mode IS patch mode, bit for bit.
    #[test]
    fn face_mode_with_nothing_anchored_is_patch_mode() {
        let (surf, mesh) = castellated_cube_case();
        let base = cube_layers(0.02);
        let mut face_spec = base.clone();
        face_spec.terminate = LayerTerminate::Face;
        let a = add_layers(&mesh, &surf, &base, &thresholds()).expect("patch");
        let b = add_layers(&mesh, &surf, &face_spec, &thresholds()).expect("face");
        assert_eq!(a.mesh.points, b.mesh.points);
        assert_eq!(a.mesh.faces, b.mesh.faces);
        assert_eq!(a.mesh.owner, b.mesh.owner);
        assert_eq!(a.mesh.neighbour, b.mesh.neighbour);
        for (pa, pb) in a.mesh.patches.iter().zip(b.mesh.patches.iter()) {
            assert_eq!(pa.start, pb.start, "patch {}", pa.name);
            assert_eq!(pa.size, pb.size, "patch {}", pa.name);
        }
    }

    #[test]
    fn add_layers_with_no_layers_is_the_identity() {
        let (surf, mesh) = snapped_sphere_case();
        let mut spec = sphere_layers(0.02);
        spec.n = 0;
        let out = add_layers(&mesh, &surf, &spec, &thresholds()).expect("layers");
        assert_eq!(out.mesh.points, mesh.points);
        assert_eq!(out.mesh.faces, mesh.faces);
        assert_eq!(out.mesh.owner, mesh.owner);
        assert_eq!(out.mesh.neighbour, mesh.neighbour);
        assert_eq!(out.mesh.patches.len(), mesh.patches.len());
        for (a, b) in out.mesh.patches.iter().zip(mesh.patches.iter()) {
            assert_eq!(a.name, b.name);
            assert_eq!(a.start, b.start);
            assert_eq!(a.size, b.size);
        }
        assert_eq!(out.report.n_layer_cells, 0);
        assert_eq!(out.extrusion.first_cell, cell_count(&mesh));
        assert_eq!(out.extrusion.n, 0);
        assert!(out.quality.passed());
    }

    #[test]
    fn a_body_gets_three_layers_behind_every_wall_face() {
        let (surf, mesh) = castellated_cube_case();
        let spec = cube_layers(0.02);
        let big_c = cell_count(&mesh);
        let p_in = mesh.points.len();
        let out = add_layers(&mesh, &surf, &spec, &thresholds()).expect("layers");
        let n_layers = 3usize;
        assert_eq!(
            out.report.n_layer_cells,
            n_layers * out.extrusion.layer_faces.len()
        );
        let f = out.extrusion.layer_faces.len();
        let slot_count = out
            .extrusion
            .slot_of_point
            .iter()
            .filter(|&&s| s >= 0)
            .count();
        assert_eq!(cell_count(&out.mesh), big_c + n_layers * f);
        assert_eq!(out.mesh.points.len(), p_in + n_layers * slot_count);
        // The layer patch's boundary faces all own layer cells, one per
        // input face, in the input's order.
        let n_internal = out.mesh.neighbour.len();
        let patch = out
            .mesh
            .patches
            .iter()
            .find(|p| p.name == "cube")
            .expect("cube patch")
            .clone();
        assert_eq!(patch.size, f, "one level-0 face per layer face");
        for b in 0..patch.size {
            let fi = n_internal + patch.start + b;
            assert!(
                out.mesh.owner[fi] as usize >= big_c,
                "boundary face {b} of the layer patch owns cell {}",
                out.mesh.owner[fi]
            );
        }
        assert!(out.quality.passed());
        assert_eq!(out.quality.n_regions, 1);
        // The achieved first-layer thickness, off the host geometry, on
        // the faces whose points carry ONE normal - a face at a stepped
        // corner carries two or three walls' average normal ((92.40)'s
        // rule), and its v/A reads only the one wall's share of it, not
        // the thickness.
        let st = stack(&spec).expect("stack");
        let patches = resolve_patches(&mesh, &spec).expect("patches");
        let idx = TriIndex::new(&surf, 0.05).expect("index");
        let fd = field(&mesh, &idx, &patches, &spec, &st).expect("field");
        let host = build_host_mesh(&out.mesh).expect("host");
        let mut n_prism = 0usize;
        for j in 0..f {
            let pts = &mesh.faces[fd.faces[j]];
            let u = fd.normal[pts[0] as usize];
            if pts.iter().any(|&p| (fd.normal[p as usize] - u).mag() > 1e-12) {
                continue;
            }
            let dom = u.x.abs().max(u.y.abs()).max(u.z.abs());
            if dom < 1.0 - 1e-12 {
                continue;
            }
            let cell = out.extrusion.first_cell + j * n_layers;
            let bf = patch.start + j;
            let h = host.v[cell] / host.b_mag_sf[bf];
            assert!(
                (h - 0.02).abs() <= 0.05 * 0.02,
                "layer {j}: v/A = {h}, want 0.02 to 5%"
            );
            n_prism += 1;
        }
        assert!(n_prism > 0, "no wall face carried a single normal");
    }

    #[test]
    fn a_cube_gets_exact_prisms_on_its_flat_faces() {
        let (surf, mesh) = castellated_cube_case();
        let spec = cube_layers(0.02);
        let st = stack(&spec).expect("stack");
        let out = add_layers(&mesh, &surf, &spec, &thresholds()).expect("layers");
        // The exactness below reads the PROPOSED field, so the case must
        // not have retreated.
        assert_eq!(out.report.retreats, 0, "the case retreated");
        let patches = resolve_patches(&mesh, &spec).expect("patches");
        let idx = TriIndex::new(&surf, 0.05).expect("index");
        let fd = field(&mesh, &idx, &patches, &spec, &st).expect("field");
        let mut n_cell_faces = vec![0u32; cell_count(&out.mesh)];
        for (fi, _) in out.mesh.faces.iter().enumerate() {
            n_cell_faces[out.mesh.owner[fi] as usize] += 1;
            if fi < out.mesh.neighbour.len() {
                n_cell_faces[out.mesh.neighbour[fi] as usize] += 1;
            }
        }
        // A point's normal is a flat side's inward unit normal when the
        // dominant component is the whole vector.
        let flat = |nn: Vec3| -> Option<Vec3> {
            let c = [nn.x.abs(), nn.y.abs(), nn.z.abs()];
            let ax = if c[0] >= c[1] && c[0] >= c[2] {
                0
            } else if c[1] >= c[2] {
                1
            } else {
                2
            };
            if (c[ax] - 1.0).abs() > 1e-12 {
                return None;
            }
            let mut v = [0.0; 3];
            v[ax] = nn.component(ax).signum();
            Some(Vec3::new(v[0], v[1], v[2]))
        };
        let mut n_qual = 0usize;
        for (j, &fa) in fd.faces.iter().enumerate() {
            let pts = &mesh.faces[fa];
            let mut sides: Vec<Vec3> = Vec::new();
            let mut ok = true;
            for &p in pts {
                match flat(fd.normal[p as usize]) {
                    Some(u) => sides.push(u),
                    None => {
                        ok = false;
                        break;
                    }
                }
            }
            if !ok || sides.iter().any(|u| (*u - sides[0]).mag() > 1e-12) {
                continue;
            }
            // And ONE displacement, so the prism is exact.
            let d0 = fd.disp[pts[0] as usize];
            if pts
                .iter()
                .any(|&p| (fd.disp[p as usize] - d0).mag() > 1e-12)
            {
                continue;
            }
            n_qual += 1;
            let nu = sides[0];
            for k in 0..3 {
                let cell = out.extrusion.first_cell + j * 3 + k;
                assert_eq!(
                    n_cell_faces[cell], 6,
                    "layer cell {cell} of face {j} is not a hexahedron"
                );
            }
            let slots: Vec<usize> = pts
                .iter()
                .map(|&p| out.extrusion.slot_of_point[p as usize] as usize)
                .collect();
            for k in 0..3 {
                let mut step = Vec3::ZERO;
                for &s in &slots {
                    let a = out.mesh.points[out.extrusion.level_point[k][s] as usize];
                    let b = out.mesh.points[out.extrusion.level_point[k + 1][s] as usize];
                    step = step + (b - a);
                }
                let h = step.dot(nu) / (slots.len() as Scalar);
                assert!(
                    (h - st.t[k]).abs() <= 1e-9,
                    "face {j} level {k}: spacing {h} vs t = {}",
                    st.t[k]
                );
            }
        }
        assert!(n_qual > 0, "no layer face sat wholly inside a flat side");
    }

    #[test]
    fn the_counts_and_the_ordering_hold() {
        let cube = {
            let (surf, mesh) = castellated_cube_case();
            (surf, mesh, cube_layers(0.02))
        };
        let gap = {
            let (surf, mesh) = castellated_gap_case();
            (surf, mesh, gap_layers(0.0))
        };
        for (surf, mesh, spec) in [cube, gap] {
            let out = add_layers(&mesh, &surf, &spec, &thresholds()).expect("layers");
            let n_internal = out.mesh.neighbour.len();
            let mut prev: Option<(crate::Label, crate::Label)> = None;
            for fi in 0..n_internal {
                let pair = (out.mesh.owner[fi], out.mesh.neighbour[fi]);
                assert!(pair.0 < pair.1, "face {fi}: owner not below neighbour");
                if let Some(p) = prev {
                    assert!(p < pair, "face {fi}: (owner, neighbour) not increasing");
                }
                prev = Some(pair);
            }
            let mut run = 0usize;
            for p in &out.mesh.patches {
                assert_eq!(p.start, run, "patch {} start", p.name);
                run += p.size;
            }
            assert_eq!(out.mesh.faces.len() - n_internal, run);
        }
    }

    #[test]
    fn a_gap_narrower_than_the_stack_retreats_instead_of_inverting() {
        let (surf, mesh) = castellated_gap_case();
        let spec = gap_layers(0.0);
        let p_in = mesh.points.len();
        let out = add_layers(&mesh, &surf, &spec, &thresholds()).expect("layers");
        assert!(out.quality.passed());
        assert!(
            out.quality.min_volume > 0.0,
            "min cell volume {}",
            out.quality.min_volume
        );
        for p in &out.report.patches {
            assert!(p.dropped.is_none(), "patch dropped: {:?}", p.dropped);
            assert!(p.n_layers > 0);
            assert!(
                p.mean_frac > 0.0 && p.mean_frac < 1.0,
                "patch {}: mean_frac {} - not a retreat",
                p.name,
                p.mean_frac
            );
        }
        assert_eq!(out.report.patches.len(), 2);
        // The SPEC's bounds, point by point. No displacement past its own
        // half-cell limit - the wall cells are 1.0 - and no layer point
        // past (92.45)'s T_i <= medial_frac * d_medial, the march asked
        // again along the very normal the field grew along. Between two
        // plates the march is caught at s = g / (2 - kappa) - kappa's
        // margin over the medial axis at g / 2, the bias
        // `two_plates_give_half_the_gap` documents - so a wall point that
        // FACES the other slab lands under medial_frac * g / 2; at the
        // slabs' rim the march meets a SIDE wall instead, farther out,
        // and the half-cell bound is the one that holds.
        let st = stack(&spec).expect("stack");
        let patches = resolve_patches(&mesh, &spec).expect("patches");
        let idx = TriIndex::new(&surf, 0.05).expect("index");
        let fd = field(&mesh, &idx, &patches, &spec, &st).expect("field");
        for i in 0..p_in {
            let d = out.mesh.points[i] - mesh.points[i];
            assert!(
                d.mag() <= 0.5 + 1e-12,
                "point {i}: |disp| = {} past the half-cell limit",
                d.mag()
            );
            if !fd.is_layer[i] || fd.pinned[i] {
                continue;
            }
            let m = medial_distance(
                &idx,
                mesh.points[i],
                fd.normal[i],
                st.total / spec.medial_frac,
            );
            let lim = if m.is_finite() {
                spec.medial_frac * m
            } else {
                0.5
            };
            assert!(
                d.mag() <= lim + 1e-12,
                "point {i}: |disp| = {d:.6} past {lim:.6}, march {m:.6}"
            );
        }
    }

    #[test]
    fn the_same_gap_with_a_floor_drops_the_patch_by_name() {
        let (surf, mesh) = castellated_gap_case();
        let spec = gap_layers(0.9);
        let out = add_layers(&mesh, &surf, &spec, &thresholds()).expect("layers");
        assert_eq!(out.report.n_layer_cells, 0);
        assert_eq!(out.report.patches.len(), 2);
        for p in &out.report.patches {
            assert_eq!(p.n_layers, 0);
            let d = p
                .dropped
                .as_ref()
                .unwrap_or_else(|| panic!("patch {} was not dropped", p.name));
            assert!(d.contains(&p.name), "the reason does not name it: {d}");
        }
        // The returned mesh is the input, bit for bit.
        assert_eq!(out.mesh.points, mesh.points);
        assert_eq!(out.mesh.faces, mesh.faces);
        assert_eq!(out.mesh.owner, mesh.owner);
        assert_eq!(out.mesh.neighbour, mesh.neighbour);
        assert_eq!(out.mesh.patches.len(), mesh.patches.len());
        for (a, b) in out.mesh.patches.iter().zip(mesh.patches.iter()) {
            assert_eq!(a.size, b.size);
        }
    }

    #[test]
    fn a_two_to_one_transition_emits_split_sides() {
        // A distance band cannot make a 2:1 transition on a wall - every
        // cell carrying a wall face is already at the band's finest level.
        // What does is FEATURE refinement: every cell the surface passes
        // through is at least level 1 (0.5), and the feature-edge pull of
        // the refinement stage takes the cells near one of the cube's 12
        // feature edges to level 2 (0.25). The middle of each face stays
        // at level 1, so the wall carries 2:1 - 48 T-junction edges on
        // this case. Castellated, not snapped: the snap grinds slivers
        // into the wall that the layer cells behind fail the gate on.
        let (mut tree, bg) = setup([0.0, 4.0, 0.0, 4.0, 0.0, 4.0], 1.0, 2);
        let surf = Surface::from_soup(
            box_soup([1.1; 3], [3.1; 3]),
            vec!["cube".to_string()],
        )
        .expect("surface");
        let spec = RefinementSpec {
            levels: vec![RefinementBand {
                patch: "cube".to_string(),
                bands: vec![DistanceBand {
                    distance: 0.0,
                    level: 1,
                }],
                feature_level: 2,
            }],
            feature_angle_deg: 30.0,
            max_level: 2,
            boxes: Vec::new(),
        };
        refine_to_surface(&mut tree, &bg, &surf, &spec).expect("refine");
        let cast = castellate(
            &tree,
            &bg,
            &surf,
            &patch_names(),
            &CastellationSpec::default(),
            &thresholds(),
        )
        .expect("castellate");
        let lspec = LayerSpec {
            patches: vec!["cube".to_string()],
            n: 2,
            first_thickness: 0.01,
            normal_passes: 0,
            min_thickness: 0.0,
            ..LayerSpec::default()
        };
        let out =
            add_layers(&cast.mesh, &surf, &lspec, &thresholds()).expect("layers");
        // If this is zero the case did not do what it was built for, and
        // the test must fail, not pass quietly.
        assert!(
            out.report.n_split_sides > 0,
            "n_split_sides = 0 - the case has no 2:1 transition on the wall"
        );
        assert!(
            out.report.patches.iter().all(|p| p.dropped.is_none()),
            "dropped: {:?}",
            out.report.patches
        );
        build_host_mesh(&out.mesh).expect("host");
        assert!(out.quality.passed());
        assert_eq!(out.quality.n_regions, 1);
        assert!(
            out.quality.max_closure < 1e-12,
            "max closure {:.3e}",
            out.quality.max_closure
        );
    }

    #[test]
    fn a_layer_thinner_than_the_gate_allows_is_refused_with_the_arithmetic() {
        let (surf, mesh) = castellated_cube_case();
        // A two-hundredth of the wall face size: 3 t_1 / h is far under
        // G5's 0.05, so the run is refused before any cell is inserted.
        let spec = cube_layers(0.5 / 200.0);
        let e = add_layers(&mesh, &surf, &spec, &thresholds()).unwrap_err();
        let msg = e.to_string();
        // h_min, the same scan the refusal runs: the shortest edge among
        // the layer faces, on the input mesh.
        let n_internal = mesh.neighbour.len().min(mesh.faces.len());
        let patch = mesh
            .patches
            .iter()
            .find(|p| p.name == "cube")
            .expect("cube patch");
        let mut h_min = f64::INFINITY;
        for j in 0..patch.size {
            let face = &mesh.faces[n_internal + patch.start + j];
            for k in 0..face.len() {
                let d = (mesh.points[face[k] as usize]
                    - mesh.points[face[(k + 1) % face.len()] as usize])
                    .mag();
                h_min = h_min.min(d);
            }
        }
        let t1 = spec.first_thickness;
        let ratio = 3.0 * t1 / h_min;
        assert!(ratio < 0.05, "the case is not thin enough: {ratio}");
        assert!(msg.contains("min_thickness_ratio"), "{msg}");
        assert!(
            msg.contains(&format!("3 * {t1} / {h_min}")),
            "no arithmetic in: {msg}"
        );
        assert!(msg.contains(&format!("{ratio}")), "{msg}");
    }

    #[test]
    fn a_snapped_wall_closes_exactly() {
        let (surf, mesh) = snapped_floor_box_case();
        let spec = LayerSpec {
            patches: vec!["cube".to_string()],
            n: 3,
            first_thickness: 0.02,
            growth: 1.3,
            min_thickness: 0.0,
            ..LayerSpec::default()
        };
        // The thresholds are loosened past any refusal on purpose: the
        // assertion below is CLOSURE, not a quality number, and a closed
        // polyhedron satisfies it exactly whatever its shape - so this
        // test catches a mis-wound face and nothing else.
        let t = QualityThresholds {
            max_closure: 1e30,
            max_non_orth_deg: 179.9,
            min_thickness_ratio: 0.0,
            max_cond: 1e300,
            ..thresholds()
        };
        let out = add_layers(&mesh, &surf, &spec, &t)
            .expect("layers on a snapped box standing on the floor");
        let m = &out.mesh;
        let n_internal = m.neighbour.len().min(m.faces.len());
        let n_cells = m
            .owner
            .iter()
            .chain(m.neighbour.iter())
            .copied()
            .max()
            .map_or(0, |c| c as usize + 1);
        let mut s = vec![Vec3::ZERO; n_cells];
        let mut a = vec![0.0; n_cells];
        for fi in 0..m.faces.len() {
            let sf = face_area_vector(&m.points, &m.faces[fi]);
            let o = m.owner[fi] as usize;
            s[o] = s[o] + sf;
            a[o] += sf.mag();
            if fi < n_internal {
                let nb = m.neighbour[fi] as usize;
                s[nb] = s[nb] - sf;
                a[nb] += sf.mag();
            }
        }
        for c in 0..n_cells {
            let ratio = s[c].mag() / a[c];
            assert!(
                ratio < 1e-12,
                "cell {c} does not close: |sum +-Sf| / sum |Sf| = {ratio:.3e}"
            );
        }
    }

    #[test]
    fn a_backwards_side_face_is_caught_before_the_gate() {
        let (surf, mesh) = castellated_cube_case();
        let out = add_layers(&mesh, &surf, &cube_layers(0.02), &thresholds())
            .expect("layers on the castellated cube");
        let mut m = out.mesh;
        let first_cell = out.extrusion.first_cell;
        let n_internal = m.neighbour.len().min(m.faces.len());
        // The mesh as assembled must close, or the case did not do what it
        // was built for and the rest of the test means nothing.
        assert!(
            check_extrusion_closes(&m, first_cell).is_ok(),
            "the assembled mesh does not close"
        );
        // Reverse the first internal face a layer cell owns: one mis-wound
        // face is exactly the fault the check exists to catch.
        let f = (0..n_internal)
            .find(|&f| m.owner[f] as usize >= first_cell)
            .expect("the case has layer cells on internal faces");
        m.faces[f].reverse();
        let msg = check_extrusion_closes(&m, first_cell)
            .expect_err("a reversed face must fail the closure check")
            .to_string();
        // Only the two cells the face belongs to lost closure, so the cell
        // the check stops on is one of them; `(` anchors the id against a
        // longer id sharing the prefix.
        assert!(msg.contains("layers"), "{msg}");
        assert!(
            [m.owner[f] as usize, m.neighbour[f] as usize]
                .iter()
                .any(|&c| msg.contains(&format!("cell {c} ("))),
            "no incident cell named in: {msg}"
        );
    }

    /// A snapped wall the layers cannot survive at G4's 70 degrees loses them
    /// BY NAME and the run continues (92.47), rather than the run stopping on a
    /// list of faces the user cannot act on. The case is the snapped floor box:
    /// the re-seat of (92.63)-(92.65) acts on its row-1 cells and the shrunk
    /// mesh still fails the gate after every retreat.
    #[test]
    fn a_wall_the_layers_cannot_survive_loses_them_by_name() {
        let (surf, mesh) = snapped_floor_box_case();
        let spec = LayerSpec {
            patches: vec!["cube".to_string()],
            n: 3,
            first_thickness: 0.02,
            growth: 1.3,
            min_thickness: 0.0,
            ..LayerSpec::default()
        };
        let before = cell_count(&mesh);
        let out = add_layers(&mesh, &surf, &spec, &thresholds())
            .expect("the patch loses its layers by name; the run continues");
        assert_eq!(cell_count(&out.mesh), before, "no layer cell survives");
        let row = out
            .report
            .patches
            .iter()
            .find(|p| p.name == "cube")
            .expect("the report carries a row for the cube");
        assert_eq!(row.n_layers, 0);
        let reason = row.dropped.as_ref().expect("the row says why");
        assert!(!reason.is_empty());
        assert!(row.n_reseated_points > 0);
        assert!(row.drop_cause.is_some());
        println!(
            "layers: the cube row: reason {reason}; n_reseated_points {}; drop_cause {:?}",
            row.n_reseated_points,
            row.drop_cause
        );
        assert!(
            out.report.summary().contains("cube"),
            "{}",
            out.report.summary()
        );
    }

    /// Written by the supervising session. The OUTER ladder of (92.47), the
    /// one that measures the EXTRUDED mesh: the snapped cube - every wall
    /// face under (92.63)'s 45 degrees, so the re-seat does not act - passes
    /// the shrink's own gate and then fails on the layer cells themselves,
    /// so the thickness retreats `retreat_limit` times and the patch loses
    /// its layers BY NAME - `add_layers` returns the snapped mesh and the
    /// reason, and does NOT refuse with a list of faces the user cannot act
    /// on.
    #[test]
    fn a_snapped_wall_retreats_on_the_extruded_mesh_then_gives_up_by_name() {
        let (surf, mesh) = snapped_cube_case();
        let spec = LayerSpec {
            patches: vec!["cube".to_string()],
            n: 3,
            first_thickness: 0.02,
            growth: 1.3,
            min_thickness: 0.0,
            ..LayerSpec::default()
        };
        let before = cell_count(&mesh);
        let out = add_layers(&mesh, &surf, &spec, &thresholds())
            .expect("a wall that cannot carry layers is not a refusal");
        assert_eq!(
            cell_count(&out.mesh),
            before,
            "the patch gave its layers up, so no cell was added"
        );
        assert_eq!(
            out.report.retreats, spec.retreat_limit,
            "the ladder took every retreat before giving up"
        );
        let row = out
            .report
            .patches
            .iter()
            .find(|p| p.name == "cube")
            .expect("the named patch has a row");
        assert_eq!(row.n_layers, 0);
        assert_eq!(row.n_reseated_points, 0);
        let reason = row.dropped.as_ref().expect("the row says why");
        assert!(reason.contains("retreat"), "{reason}");
        assert!(reason.contains("92.13"), "{reason}");
        assert!(
            reason.contains("castellates onto the cell planes"),
            "the reason tells the user what IS supported: {reason}"
        );
        assert!(out.report.summary().contains("cube"), "{}", out.report.summary());
    }

    /// The report says the first layer it ACHIEVED, in metres, because the
    /// fraction alone hides a limiter the user did not write down
    /// (SPEC-LIT 92.13, (92.50)).
    #[test]
    fn the_report_says_the_achieved_first_layer_in_metres() {
        let (surf, mesh) = castellated_cube_case();
        let out = add_layers(&mesh, &surf, &cube_layers(0.02), &thresholds())
            .expect("layers");
        let row = out
            .report
            .patches
            .iter()
            .find(|p| p.name == "cube")
            .expect("the named patch has a row");
        assert!(row.dropped.is_none(), "{:?}", row.dropped);
        assert!(
            (row.t1_requested - 0.02).abs() < 1e-12,
            "t1_requested = {}",
            row.t1_requested
        );
        assert!(
            (row.t1_mean - 0.02 * row.mean_frac).abs() < 1e-12,
            "t1_mean {} vs 0.02 * mean_frac {}",
            row.t1_mean,
            row.mean_frac
        );
        assert!(
            0.0 < row.t1_min && row.t1_min <= row.t1_mean + 1e-12,
            "t1_min {} vs t1_mean {}",
            row.t1_min,
            row.t1_mean
        );
        let s = out.report.summary();
        assert!(s.contains("first layer"), "{s}");
        assert!(s.contains("requested"), "{s}");
        assert!(!s.contains("NO face received the full stack"), "{s}");

        // The same wall, a limiter chosen to BITE without tripping G5:
        // `cell_frac * h = 0.10 * 0.5 = 0.05` binds against the stack's
        // T = 0.02 + 0.026 + 0.0338 = 0.0798, so tau = 0.627 and the
        // achieved first layer is about 0.0125 m, not the 0.02 asked for.
        let spec = LayerSpec {
            cell_frac: 0.10,
            ..cube_layers(0.02)
        };
        let out = add_layers(&mesh, &surf, &spec, &thresholds()).expect("layers");
        let row = out
            .report
            .patches
            .iter()
            .find(|p| p.name == "cube")
            .expect("the named patch has a row");
        assert!(
            row.dropped.is_none(),
            "the limiter bites but the patch survives: {:?}",
            row.dropped
        );
        assert!(
            row.t1_mean < row.t1_requested,
            "t1_mean {} vs t1_requested {}",
            row.t1_mean,
            row.t1_requested
        );
        assert!(
            (row.t1_mean - 0.05 / 0.0798 * 0.02).abs() < 1e-3,
            "t1_mean {} vs 0.02 * tau, tau = 0.05/0.0798",
            row.t1_mean
        );
    }

    /// The row `area` of (92.50) is the patch's own area: on the on-plane
    /// cube the rows' areas sum to the STL's area for the patch to 1e-12
    /// relative, and the recorded `(A_f, tau_f)` pairs re-sum to the row's
    /// `area` bit for bit, in the order `full` sums them.
    #[test]
    fn the_row_areas_sum_to_the_cube_area() {
        let (surf, mesh) = castellated_cube_case();
        let out = add_layers(&mesh, &surf, &cube_layers(0.02), &thresholds())
            .expect("layers");
        let k = surf
            .patch_names
            .iter()
            .position(|n| n == "cube")
            .expect("the surface names the cube");
        let a_stl = surf.patch_area[k];
        let sum: Scalar = out.report.patches.iter().map(|p| p.area).sum();
        eprintln!("row areas sum {sum:.9} vs STL area {a_stl:.9}");
        assert!(((sum - a_stl) / a_stl).abs() <= 1e-12, "{sum} vs {a_stl}");
        let row = out
            .report
            .patches
            .iter()
            .find(|p| p.name == "cube")
            .expect("the named patch has a row");
        assert!(row.dropped.is_none(), "{:?}", row.dropped);
        assert_eq!(row.face_area_tau.len(), row.n_faces);
        let mut from_pairs = 0.0;
        for &(a, _) in &row.face_area_tau {
            from_pairs += a;
        }
        assert_eq!(from_pairs.to_bits(), row.area.to_bits());
    }

    /// The tau shares at every beta of `TAU_GE_BETAS` are the rows'
    /// `frac_tau_ge` recomputed, order down to `full_area_frac` at beta = 1
    /// bit for bit, sit under the Markov bound `beta * share <= mean_frac`,
    /// and vanish with no `(A_f, tau_f)` pairs on a dropped row.
    #[test]
    fn tau_ge_one_is_the_full_area_fraction() {
        let (surf, mesh) = castellated_cube_case();
        let a = add_layers(&mesh, &surf, &cube_layers(0.02), &thresholds())
            .expect("layers");
        let lim = LayerSpec { cell_frac: 0.10, ..cube_layers(0.02) };
        let b = add_layers(&mesh, &surf, &lim, &thresholds()).expect("layers");
        let (surf_g, mesh_g) = castellated_gap_case();
        let c = add_layers(&mesh_g, &surf_g, &gap_layers(0.0), &thresholds())
            .expect("layers");
        let (surf_s, mesh_s) = snapped_sphere_case();
        let d = add_layers(&mesh_s, &surf_s, &sphere_layers(0.05), &thresholds())
            .expect("layers on the snapped sphere");
        let (surf_c, mesh_c) = snapped_cube_case();
        let spec_e = LayerSpec {
            patches: vec!["cube".to_string()],
            n: 3,
            first_thickness: 0.02,
            growth: 1.3,
            min_thickness: 0.0,
            ..LayerSpec::default()
        };
        let e = add_layers(&mesh_c, &surf_c, &spec_e, &thresholds())
            .expect("the patch loses its layers by name; the run continues");
        let cases: [(&str, &LayerReport, bool); 5] = [
            ("a", &a.report, true),
            ("b", &b.report, true),
            ("c", &c.report, true),
            ("d", &d.report, true),
            ("e", &e.report, false),
        ];
        for (case, report, want_kept) in cases {
            let mut kept = 0usize;
            let mut dropped = 0usize;
            for row in &report.patches {
                eprintln!(
                    "{case} \"{}\": full {:.6}, ge {:.6} {:.6} {:.6}, mean {:.6}",
                    row.name,
                    row.full_area_frac,
                    row.area_frac_tau_ge[0],
                    row.area_frac_tau_ge[1],
                    row.area_frac_tau_ge[2],
                    row.mean_frac
                );
                check_tau_ge_row(row);
                if row.dropped.is_some() {
                    dropped += 1;
                } else {
                    kept += 1;
                }
            }
            if want_kept {
                assert!(kept > 0, "{case}: no non-dropped row");
            } else {
                assert!(dropped > 0, "{case}: no dropped row");
            }
        }
        let cube = b
            .report
            .patches
            .iter()
            .find(|p| p.name == "cube")
            .expect("the named patch has a row");
        assert!(
            cube.area_frac_tau_ge[0] > cube.area_frac_tau_ge[1],
            "the limiter's tau sits between 0.5 and 0.8: {:.6} vs {:.6}",
            cube.area_frac_tau_ge[0],
            cube.area_frac_tau_ge[1]
        );
    }

    /// The per-row half of `tau_ge_one_is_the_full_area_fraction`, shared by
    /// all five of its cases.
    fn check_tau_ge_row(row: &PatchLayers) {
        assert_eq!(
            row.frac_tau_ge(1.0).to_bits(),
            row.full_area_frac.to_bits()
        );
        for i in 0..3 {
            assert_eq!(
                row.area_frac_tau_ge[i].to_bits(),
                row.frac_tau_ge(TAU_GE_BETAS[i]).to_bits()
            );
        }
        assert!(row.area_frac_tau_ge[0] >= row.area_frac_tau_ge[1]);
        assert!(row.area_frac_tau_ge[1] >= row.area_frac_tau_ge[2]);
        assert!(row.area_frac_tau_ge[2] >= row.full_area_frac);
        assert!(row.area_frac_tau_ge[0] <= 1.0 + 1e-12);
        for i in 0..3 {
            assert!(
                TAU_GE_BETAS[i] * row.area_frac_tau_ge[i] <= row.mean_frac + 1e-12,
                "beta {} share {} vs mean {}",
                TAU_GE_BETAS[i],
                row.area_frac_tau_ge[i],
                row.mean_frac
            );
        }
        if row.dropped.is_some() {
            assert_eq!(row.area_frac_tau_ge, [0.0; 3]);
            assert!(row.face_area_tau.is_empty());
        }
    }

    /// Byte-for-byte goldens of stage 6's output on the castellated-wall
    /// cases, where every wall face lies on a cell plane, and a printed (not
    /// asserted) report of the snapped cases. A golden is SHA-256 over the
    /// emitted mesh, written at the commit this module landed on; it changes
    /// only when the layer stage's output on these cases does.
    pub(crate) mod castellated_goldens {
        use super::*;

        /// SHA-256, FIPS 180-4 (NIST, 2015) section 6.2, as 64 lowercase
        /// hex digits.
        pub(crate) fn sha256_hex(msg: &[u8]) -> String {
            const K: [u32; 64] = [
                0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5,
                0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
                0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
                0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
                0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc,
                0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
                0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
                0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
                0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
                0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
                0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3,
                0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
                0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5,
                0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
                0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
                0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
            ];
            let mut h: [u32; 8] = [
                0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
                0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
            ];
            let mut m = msg.to_vec();
            let bit_len = (msg.len() as u64).wrapping_mul(8);
            m.push(0x80);
            while m.len() % 64 != 56 {
                m.push(0);
            }
            m.extend_from_slice(&bit_len.to_be_bytes());
            for block in m.chunks_exact(64) {
                let mut w = [0u32; 64];
                for t in 0..16 {
                    let q = &block[4 * t..4 * t + 4];
                    w[t] = u32::from_be_bytes([q[0], q[1], q[2], q[3]]);
                }
                for t in 16..64 {
                    let (a, b) = (w[t - 15], w[t - 2]);
                    let s0 = a.rotate_right(7) ^ a.rotate_right(18) ^ (a >> 3);
                    let s1 = b.rotate_right(17) ^ b.rotate_right(19) ^ (b >> 10);
                    w[t] = w[t - 16]
                        .wrapping_add(s0)
                        .wrapping_add(w[t - 7])
                        .wrapping_add(s1);
                }
                let mut v = h;
                for t in 0..64 {
                    let [a, b, c, d, e, f, g, hh] = v;
                    let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
                    let ch = (e & f) ^ (!e & g);
                    let t1 = hh
                        .wrapping_add(s1)
                        .wrapping_add(ch)
                        .wrapping_add(K[t])
                        .wrapping_add(w[t]);
                    let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
                    let maj = (a & b) ^ (a & c) ^ (b & c);
                    let t2 = s0.wrapping_add(maj);
                    v = [t1.wrapping_add(t2), a, b, c, d.wrapping_add(t1), e, f, g];
                }
                for k in 0..8 {
                    h[k] = h[k].wrapping_add(v[k]);
                }
            }
            h.iter().map(|x| format!("{x:08x}")).collect()
        }

        /// SHA-256 over every count, every point's f64 bits, every face,
        /// owner, neighbour and patch-table entry of `m`, in order,
        /// little-endian.
        pub(crate) fn mesh_sha256(m: &PolyMeshRaw) -> String {
            let mut b: Vec<u8> = b"PolyMeshRaw/v1\0".to_vec();
            let n = [m.points.len(), m.faces.len(), m.owner.len()];
            for k in n.into_iter().chain([m.neighbour.len(), m.patches.len()]) {
                b.extend_from_slice(&(k as u64).to_le_bytes());
            }
            for p in &m.points {
                for v in [p.x, p.y, p.z] {
                    b.extend_from_slice(&(v as f64).to_bits().to_le_bytes());
                }
            }
            for f in &m.faces {
                b.extend_from_slice(&(f.len() as u64).to_le_bytes());
                for q in f {
                    b.extend_from_slice(&(*q as i64).to_le_bytes());
                }
            }
            for o in m.owner.iter().chain(m.neighbour.iter()) {
                b.extend_from_slice(&(*o as i64).to_le_bytes());
            }
            for p in &m.patches {
                let kind = format!("{:?}", p.kind);
                for s in [p.name.as_str(), p.type_name.as_str(), kind.as_str()] {
                    b.extend_from_slice(s.as_bytes());
                    b.push(0);
                }
                b.extend_from_slice(&(p.start as u64).to_le_bytes());
                b.extend_from_slice(&(p.size as u64).to_le_bytes());
                let nbr = p.nbr_patch.map_or(-1i64, |k| k as i64);
                b.extend_from_slice(&nbr.to_le_bytes());
            }
            sha256_hex(&b)
        }

        /// What stage 6 returns on one case, hashed: the mesh when it returns
        /// one, `refused:` and the hash of the refusal's text when it refuses.
        fn run_case(surf: &Surface, mesh: &PolyMeshRaw, spec: &LayerSpec) -> String {
            match add_layers(mesh, surf, spec, &thresholds()) {
                Ok(out) => mesh_sha256(&out.mesh),
                Err(e) => format!("refused:{}", sha256_hex(e.to_string().as_bytes())),
            }
        }

        /// The case of `a_two_to_one_transition_emits_split_sides`, its setup
        /// and layer spec copied unchanged: feature refinement takes the cells
        /// along the cube's edges to level 2, so the castellated wall carries
        /// 2:1 transitions and the extrusion emits split sides.
        fn two_to_one_case() -> (Surface, PolyMeshRaw, LayerSpec) {
            let (mut tree, bg) = setup([0.0, 4.0, 0.0, 4.0, 0.0, 4.0], 1.0, 2);
            let surf = Surface::from_soup(box_soup([1.1; 3], [3.1; 3]), vec!["cube".to_string()])
                .expect("surface");
            let spec = RefinementSpec {
                levels: vec![RefinementBand {
                    patch: "cube".to_string(),
                    bands: vec![DistanceBand { distance: 0.0, level: 1 }],
                    feature_level: 2,
                }],
                feature_angle_deg: 30.0,
                max_level: 2,
                boxes: Vec::new(),
            };
            refine_to_surface(&mut tree, &bg, &surf, &spec).expect("refine");
            let cast = castellate(
                &tree,
                &bg,
                &surf,
                &patch_names(),
                &CastellationSpec::default(),
                &thresholds(),
            )
            .expect("castellate");
            let lspec = LayerSpec {
                patches: vec!["cube".to_string()],
                n: 2,
                first_thickness: 0.01,
                normal_passes: 0,
                min_thickness: 0.0,
                ..LayerSpec::default()
            };
            (surf, cast.mesh, lspec)
        }

        /// The castellated-wall cases, by golden name; each is built afresh.
        fn castellated_cases() -> Vec<(&'static str, Surface, PolyMeshRaw, LayerSpec)> {
            let mut smoothed = cube_layers(0.02);
            smoothed.normal_passes = LayerSpec::default().normal_passes;
            let (c1s, c1m) = castellated_cube_case();
            let (c2s, c2m) = castellated_cube_case();
            let (c3s, c3m) = castellated_cube_case();
            let (g1s, g1m) = castellated_gap_case();
            let (g2s, g2m) = castellated_gap_case();
            let (ts, tm, tl) = two_to_one_case();
            vec![
                ("box_minus_cube", c1s, c1m, smoothed),
                ("box_minus_cube_normal_passes_0", c2s, c2m, cube_layers(0.02)),
                ("slot", g1s, g1m, gap_layers(0.0)),
                ("slot_with_floor", g2s, g2m, gap_layers(0.9)),
                ("two_to_one_split_sides", ts, tm, tl),
                ("thin_t1_refusal", c3s, c3m, cube_layers(0.5 / 200.0)),
            ]
        }

        /// Written from the output of the commit this module landed on.
        const GOLDENS: [(&str, &str); 6] = [
            ("box_minus_cube", "50a122b585c3808af769fa7c10e8685561e5c014c65686cfd2e6832926fb4e76"),
            ("box_minus_cube_normal_passes_0", "6a66264cf1b2822b7d317440a2cd5f43c83ff7e531c72d8d0cd4f6fa8224d617"),
            ("slot", "8864f6af86014970c34146d02a572c8e38d30d8cd5f40697822494c6a894e8b9"),
            ("slot_with_floor", "3ce1d2e9882accd2859b8f51d0e65e9eb158366ec27224d4c99ab2838f54a4a7"),
            ("two_to_one_split_sides", "b131581b6e4411aa5d274ec376153ac7a38806dab82832b4b590ae3153cad72e"),
            ("thin_t1_refusal", "refused:307046d9b46bd9358add4273f97ea40beda48fc10292a7c73cb7bab51069fe8e"),
        ];

        /// The FIPS 180-4 example messages give their published digests.
        #[test]
        fn sha256_matches_the_published_vectors() {
            let two_block = b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq";
            let million = vec![b'a'; 1_000_000];
            let cases: [(&[u8], &str); 4] = [
                (&b""[..], "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"),
                (&b"abc"[..], "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"),
                (&two_block[..], "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1"),
                (&million[..], "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0"),
            ];
            for (msg, want) in cases {
                assert_eq!(sha256_hex(msg), want, "message of {} bytes", msg.len());
            }
        }

        /// Every castellated-wall case returns the mesh (or the refusal) it
        /// returned when the goldens were written, bit for bit.
        #[test]
        fn the_castellated_layer_cases_are_golden() {
            let mut bad: Vec<String> = Vec::new();
            for (name, surf, mesh, spec) in castellated_cases() {
                let got = run_case(&surf, &mesh, &spec);
                eprintln!("golden {name} = {got}");
                let want = GOLDENS.iter().find(|g| g.0 == name).map(|g| g.1);
                if want != Some(got.as_str()) {
                    bad.push(format!("{name}: got {got}, want {want:?}"));
                }
            }
            assert!(bad.is_empty(), "the goldens differ: {bad:#?}");
            for (name, h) in GOLDENS {
                assert_eq!(
                    name == "thin_t1_refusal",
                    h.starts_with("refused:"),
                    "{name}: only the thin first layer is a refusal"
                );
            }
        }

        /// One flipped bit in one emitted point changes the hash, and
        /// flipping it back restores it: the goldens are not vacuous.
        #[test]
        fn a_flipped_point_bit_changes_the_golden() {
            let (surf, mesh) = castellated_cube_case();
            let mut out = add_layers(&mesh, &surf, &cube_layers(0.02), &thresholds())
                .expect("layers");
            let name = "box_minus_cube_normal_passes_0";
            let want = GOLDENS.iter().find(|g| g.0 == name).expect("golden").1;
            assert_eq!(mesh_sha256(&out.mesh), want);
            let i = out.mesh.points.len() - 1;
            let x = out.mesh.points[i].x;
            out.mesh.points[i].x = Scalar::from_bits(x.to_bits() ^ 1);
            assert_ne!(mesh_sha256(&out.mesh), want);
            out.mesh.points[i].x = x;
            assert_eq!(mesh_sha256(&out.mesh), want);
        }

        /// The snapped cases' hashes, printed for the record and NOT
        /// asserted: a snapped wall is what later layer work may change.
        #[test]
        fn the_snapped_layer_cases_are_reported() {
            let floor = LayerSpec {
                patches: vec!["cube".to_string()],
                n: 3,
                first_thickness: 0.02,
                growth: 1.3,
                min_thickness: 0.0,
                ..LayerSpec::default()
            };
            let cases = [
                ("snapped_cube", snapped_cube_case(), cube_layers(0.02)),
                ("snapped_floor_box", snapped_floor_box_case(), floor),
                ("snapped_sphere_t1_0.02", snapped_sphere_case(), sphere_layers(0.02)),
                ("snapped_sphere_t1_0.05", snapped_sphere_case(), sphere_layers(0.05)),
            ];
            for (name, (surf, mesh), spec) in cases {
                let got = run_case(&surf, &mesh, &spec);
                eprintln!("snapped {name} = {got}");
                assert!(got.len() == 64 || got.starts_with("refused:"), "{name}: {got}");
            }
        }
    }

    /// The shape every trace has, whatever the case: rounds numbered in
    /// order with each inner entry ahead of its round's outer one, gates
    /// listed exactly where a gate failed, a cause and a patch exactly on a
    /// give-up, level-n faces only on the outer ladder and never more than
    /// G4 failed on, and a last entry that is the outer pass the run
    /// returned on.
    fn check_ladder(trace: &[LadderEntry]) {
        let mut next_outer = 0usize;
        for e in trace {
            assert_eq!(e.round, next_outer, "{e:?}");
            if e.ladder == Ladder::Outer {
                next_outer += 1;
            }
            let quiet = e.give_up.is_none() && e.dropped.is_none();
            match e.outcome {
                Outcome::Pass => assert!(e.gates.is_empty() && quiet, "{e:?}"),
                Outcome::Retreat => assert!(!e.gates.is_empty() && quiet, "{e:?}"),
                Outcome::Beta => {
                    assert!(!e.gates.is_empty() && quiet && e.beta_points > 0, "{e:?}")
                }
                Outcome::GiveUp => assert!(e.give_up.is_some() && e.dropped.is_some(), "{e:?}"),
                Outcome::Terminate => {
                    assert!(!e.gates.is_empty() && quiet && e.terminate_points > 0, "{e:?}")
                }
            }
            match e.give_up {
                Some(DropCause::OuterGate) => {
                    assert!(e.ladder == Ladder::Outer && !e.gates.is_empty(), "{e:?}")
                }
                Some(DropCause::InnerGate) => assert_eq!(e.ladder, Ladder::Inner, "{e:?}"),
                Some(_) => assert!(e.ladder == Ladder::Inner && e.gates.is_empty(), "{e:?}"),
                None => {}
            }
            let g4: usize =
                e.gates.iter().filter(|(g, _)| *g == Gate::NonOrth).map(|(_, n)| *n).sum();
            assert!(e.g4_level_n <= g4, "{e:?}");
            if e.ladder == Ladder::Inner {
                assert_eq!(e.g4_level_n, 0, "{e:?}");
            }
        }
        let last = trace.last().expect("a run that returned has a trace");
        assert!(last.ladder == Ladder::Outer && last.outcome == Outcome::Pass, "{last:?}");
    }

    /// Each row's `drop_cause` is the cause of the LAST give-up in the trace
    /// that names its patch, and a row that was not dropped has none.
    fn check_rows(out: &Layered) {
        for p in &out.report.patches {
            let last = out
                .report
                .ladder
                .iter()
                .rev()
                .find(|e| e.dropped.as_deref() == Some(p.name.as_str()));
            if p.dropped.is_none() {
                assert!(p.drop_cause.is_none(), "{p:?}");
            }
            if let Some(c) = p.drop_cause {
                assert!(p.dropped.is_some(), "{p:?}");
                assert_eq!(last.and_then(|e| e.give_up), Some(c), "{}", p.name);
            }
        }
    }

    /// A stack the castellated cube keeps leaves the shortest trace there
    /// is: one inner pass and one outer pass, in round 0, on the one patch,
    /// and no row names a cause.
    #[test]
    fn a_kept_stack_leaves_one_inner_and_one_outer_pass() {
        let (surf, mesh) = castellated_cube_case();
        let out = add_layers(&mesh, &surf, &cube_layers(0.02), &thresholds()).expect("layers");
        check_ladder(&out.report.ladder);
        check_rows(&out);
        let names = vec!["cube".to_string()];
        assert_eq!(
            out.report.ladder,
            vec![
                LadderEntry::new(Ladder::Inner, 0, &names, &[], Outcome::Pass, None),
                LadderEntry::new(Ladder::Outer, 0, &names, &[], Outcome::Pass, None),
            ]
        );
        assert!(out.report.patches.iter().all(|p| p.drop_cause.is_none()));
    }

    /// The snapped cube with no floor - no wall face past (92.63)'s 45
    /// degrees, so the re-seat does not act - gives up in the OUTER ladder:
    /// `retreat_limit` outer retreats at rungs 0, 1, ..., each naming the
    /// gates that failed, then the give-up that drops the cube as
    /// `outer_gate`, then the pass on the empty set the run returns on.
    #[test]
    fn the_outer_ladder_names_the_gates_it_gave_up_on() {
        let (surf, mesh) = snapped_cube_case();
        let spec = LayerSpec {
            patches: vec!["cube".to_string()],
            n: 3,
            first_thickness: 0.02,
            growth: 1.3,
            min_thickness: 0.0,
            ..LayerSpec::default()
        };
        let out = add_layers(&mesh, &surf, &spec, &thresholds()).expect("layers");
        for e in &out.report.ladder {
            eprintln!("snapped cube ladder {e:?}");
        }
        check_ladder(&out.report.ladder);
        check_rows(&out);
        let outer: Vec<&LadderEntry> =
            out.report.ladder.iter().filter(|e| e.ladder == Ladder::Outer).collect();
        let rungs: Vec<usize> =
            outer.iter().filter(|e| e.outcome == Outcome::Retreat).map(|e| e.rung).collect();
        assert_eq!(rungs, (0..spec.retreat_limit).collect::<Vec<usize>>());
        assert_eq!(outer.len(), spec.retreat_limit + 2);
        let quit = outer[spec.retreat_limit];
        assert_eq!(quit.outcome, Outcome::GiveUp, "{quit:?}");
        assert_eq!(quit.give_up, Some(DropCause::OuterGate), "{quit:?}");
        assert_eq!(quit.dropped.as_deref(), Some("cube"), "{quit:?}");
        assert_eq!(quit.rung, spec.retreat_limit, "{quit:?}");
        assert!(outer[spec.retreat_limit + 1].patches.is_empty());
        let row = out.report.patches.iter().find(|p| p.name == "cube").expect("row");
        assert_eq!(row.drop_cause, Some(DropCause::OuterGate));
    }

    /// The snapped level-2 sphere with eight layers of 0.006 at growth 1.0:
    /// the re-seat acts on its row-1 cells, the outer ladder still retreats
    /// on the extruded mesh, and the caps take the stack under the floor -
    /// thin_after_caps, after at least one outer retreat. Printed for the
    /// record.
    #[test]
    fn the_sphere_drop_names_what_drove_it() {
        let (surf, mesh) = snapped_sphere_case();
        let spec = LayerSpec {
            patches: vec!["sphere".to_string()],
            n: 8,
            first_thickness: 0.006,
            growth: 1.0,
            ..LayerSpec::default()
        };
        let out = add_layers(&mesh, &surf, &spec, &thresholds()).expect("layers");
        for e in &out.report.ladder {
            eprintln!("sphere ladder {e:?}");
        }
        check_ladder(&out.report.ladder);
        check_rows(&out);
        let row = out.report.patches.iter().find(|p| p.name == "sphere").expect("row");
        let cause = row.drop_cause.expect("the sphere loses its layers and says what drove it");
        eprintln!("sphere drop_cause {}", cause.as_str());
        assert_eq!(cause, DropCause::ThinAfterCaps);
        assert_eq!(row.n_reseated_points, 888);
        let quit = out
            .report
            .ladder
            .iter()
            .rposition(|e| e.dropped.as_deref() == Some("sphere"))
            .expect("a give-up names the sphere");
        if cause == DropCause::ThinAfterCaps {
            assert!(out.report.ladder[..quit]
                .iter()
                .any(|e| e.ladder == Ladder::Outer && e.outcome == Outcome::Retreat));
        }
    }

    /// The floor's classes on their own: a capped point names the caps
    /// whatever else holds, a halved one the inner ladder, neither the
    /// proposal; and the short gate names the summary prints.
    #[test]
    fn the_floor_names_what_thinned_the_point() {
        let caps = [1.0, 0.5, 1.0];
        let halved = [false, false, true];
        assert_eq!(thin_cause(&[0, 1], &caps, &halved), DropCause::ThinAfterCaps);
        assert_eq!(thin_cause(&[1, 2], &caps, &halved), DropCause::ThinAfterCaps);
        assert_eq!(thin_cause(&[0, 2], &caps, &halved), DropCause::InnerGate);
        assert_eq!(thin_cause(&[0], &caps, &halved), DropCause::ThinProposed);
        assert_eq!(gate_label(Gate::NonOrth), "G4");
        assert_eq!(gate_label(Gate::Thickness), "G5");
        assert_eq!(DropCause::ThinAfterCaps.as_str(), "thin_after_caps");
    }

    /// `level_n_g4` against a count by point lists: on the floor box's
    /// first extruded attempt, the G4 subjects whose face has the point list
    /// of one of the input's layer faces - the level-n face is that face.
    #[test]
    fn the_level_n_count_matches_the_layer_faces_by_point_list() {
        let (surf, mesh) = snapped_floor_box_case();
        let spec = LayerSpec {
            patches: vec!["cube".to_string()],
            n: 3,
            first_thickness: 0.02,
            growth: 1.3,
            min_thickness: 0.0,
            ..LayerSpec::default()
        };
        let patches = resolve_patches(&mesh, &spec).expect("patches");
        let caps = vec![1.0 as Scalar; mesh.points.len()];
        let a = attempt(
            &mesh,
            &surf,
            &spec,
            &thresholds(),
            &patches,
            &caps,
            &vec![1.0 as Scalar; mesh.points.len()],
            &vec![false; mesh.faces.len()],
            &vec![false; mesh.faces.len()],
        )
        .expect("attempt");
        let key = |ps: &[crate::Label]| {
            let mut v = ps.to_vec();
            v.sort_unstable();
            v
        };
        let layer: std::collections::HashSet<Vec<crate::Label>> =
            a.extrusion.layer_faces.iter().map(|&f| key(&mesh.faces[f])).collect();
        let (mut g4, mut brute) = (0usize, 0usize);
        for g in &a.quality.failures {
            if g.gate == Gate::NonOrth {
                g4 += g.n_failed;
                for s in &g.subjects {
                    if layer.contains(&key(&a.mesh.faces[s.id])) {
                        brute += 1;
                    }
                }
            }
        }
        let got = level_n_g4(&a.quality, &a.mesh, a.extrusion.first_cell);
        eprintln!("floor box attempt 0: G4 {g4}, level-n {got}, by point list {brute}");
        assert_eq!(got, brute);
    }

    /// What the shrink's FIRST round re-seats: the field and graphs of
    /// `shrink_on`'s first round, and the (92.64) plan over them.
    fn reseat_of(mesh: &PolyMeshRaw, surf: &Surface, spec: &LayerSpec) -> Reseat {
        let patches = resolve_patches(mesh, spec).expect("resolve_patches");
        let st = stack(spec).expect("stack");
        let n_faces = mesh.faces.len();
        let n_internal = mesh.neighbour.len().min(n_faces);
        let n_points = mesh.points.len();
        let hint = (surf.bbox.1 - surf.bbox.0).mag() / 100.0;
        let idx = TriIndex::new(surf, hint).expect("TriIndex");
        let f = field(mesh, &idx, &patches, spec, &st).expect("field");
        let mut is_b = vec![false; n_points];
        for fi in n_internal..n_faces {
            for &p in &mesh.faces[fi] {
                is_b[p as usize] = true;
            }
        }
        let n_cells = mesh
            .owner
            .iter()
            .chain(mesh.neighbour.iter())
            .copied()
            .max()
            .map_or(0, |m| m as usize + 1);
        let mut cell_faces: Vec<Vec<usize>> = vec![Vec::new(); n_cells];
        for (fa, &c) in mesh.owner.iter().enumerate() {
            cell_faces[c as usize].push(fa);
        }
        for fa in 0..n_internal {
            cell_faces[mesh.neighbour[fa] as usize].push(fa);
        }
        let theta = predicted_angles(mesh).expect("predicted_angles");
        reseat_set(mesh, &theta, &f, &is_b, &cell_faces)
    }

    /// The largest predicted angle over the patch faces `shrink_on` walks.
    fn worst_theta(mesh: &PolyMeshRaw, spec: &LayerSpec) -> (usize, Scalar) {
        let theta = predicted_angles(mesh).expect("predicted_angles");
        let n_internal = mesh.neighbour.len().min(mesh.faces.len());
        let patches = resolve_patches(mesh, spec).expect("resolve_patches");
        let mut worst: Scalar = 0.0;
        let mut nf = 0usize;
        for &p in &patches {
            let patch = &mesh.patches[p];
            for j in 0..patch.size {
                let fa = n_internal + patch.start + j;
                if fa < theta.len() {
                    worst = worst.max(theta[fa]);
                    nf += 1;
                }
            }
        }
        (nf, worst)
    }

    #[test]
    fn the_reseat_moves_a_row_one_point_along_the_wall_normal() {
        let xj = Vec3::new(0.0, 0.0, 1.0);
        let wall = vec![
            (Vec3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 0.1), Vec3::new(0.0, 0.0, 1.0)),
            (Vec3::new(1.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 0.2), Vec3::new(0.0, 0.0, 1.0)),
        ];
        let d1 = reseat_disp(xj, &wall, 1.0);
        eprintln!("reseat beta=1 {d1:?}");
        assert!((d1.x - 0.5).abs() < 1e-15);
        assert!((d1.y - 0.0).abs() < 1e-15);
        assert!((d1.z - 0.3571067811865476).abs() < 1e-15);
        let d2 = reseat_disp(xj, &wall, 0.5);
        eprintln!("reseat beta=0.5 {d2:?}");
        assert!((d2.x - 0.25).abs() < 1e-15);
        assert!((d2.y - 0.0).abs() < 1e-15);
        assert!((d2.z - 0.2535533905932738).abs() < 1e-15);
        let d0 = reseat_disp(xj, &wall, 0.0);
        eprintln!("reseat beta=0 {d0:?}");
        assert!((d0.x).abs() < 1e-15 && (d0.y).abs() < 1e-15);
        assert!((d0.z - 0.15).abs() < 1e-15);

        let d3 = reseat_disp(
            Vec3::new(0.0, 1.0, 0.0),
            &[(
                Vec3::new(0.0, 0.0, 0.0),
                Vec3::new(0.1, 0.0, 0.0),
                Vec3::new(1.0, 0.0, 0.0),
            )],
            1.0,
        );
        eprintln!("reseat x-diag {d3:?}");
        assert!((d3.x - 1.1).abs() < 1e-15);
        assert!((d3.y + 1.0).abs() < 1e-15);
        assert!((d3.z).abs() < 1e-15);
        let d4 = reseat_disp(
            Vec3::new(0.5, 0.5, 1.5),
            &[(
                Vec3::new(0.5, 0.5, 1.0),
                Vec3::new(0.0, 0.0, 0.02),
                Vec3::new(0.0, 0.0, 1.0),
            )],
            1.0,
        );
        eprintln!("reseat straight above {d4:?}");
        assert!((d4.x).abs() < 1e-15 && (d4.y).abs() < 1e-15);
        assert!((d4.z - 0.02).abs() < 1e-15);
        let d5 = reseat_disp(xj, &[], 1.0);
        eprintln!("reseat empty wall {d5:?}");
        assert_eq!(d5, Vec3::ZERO);
    }

    #[test]
    fn a_wall_on_the_cell_planes_predicts_no_angle_and_reseats_nothing() {
        let cases = [
            ("cube", castellated_cube_case(), cube_layers(0.02)),
            ("gap", castellated_gap_case(), gap_layers(0.0)),
        ];
        for (name, (surf, mesh), spec) in cases {
            let (nf, worst) = worst_theta(&mesh, &spec);
            eprintln!("theta {name}: worst {worst:.3e} deg over {nf} layer faces");
            assert!(worst < 1e-4, "{name}: worst theta {worst}");
            let rs = reseat_of(&mesh, &surf, &spec);
            eprintln!("reseat {name}: {} points", rs.points.len());
            assert!(rs.points.is_empty(), "{name}: J is not empty");
            assert!(rs.wall.is_empty(), "{name}: W is not empty");
            assert!(
                rs.per_patch.iter().all(|&c| c == 0),
                "{name}: per_patch {:?}",
                rs.per_patch
            );
            let out = add_layers(&mesh, &surf, &spec, &thresholds()).expect("layers");
            assert!(
                out.report.patches.iter().all(|r| r.n_reseated_points == 0),
                "{name}: a row reseated points: {:?}",
                out.report
                    .patches
                    .iter()
                    .map(|r| (&r.name, r.n_reseated_points))
                    .collect::<Vec<_>>()
            );
            eprintln!(
                "reseat {name}: all n_reseated_points 0 across {} row(s)",
                out.report.patches.len()
            );
        }
    }

    #[test]
    fn the_snapped_cube_stays_under_the_threshold() {
        let (surf, mesh) = snapped_cube_case();
        let spec = cube_layers(0.02);
        let (nf, worst) = worst_theta(&mesh, &spec);
        eprintln!("theta snapped cube: worst {worst:.6} deg over {nf} faces");
        assert_eq!(nf, 24, "the cube patch has 24 faces");
        assert!(worst > 38.93 && worst < 38.94, "worst theta {worst}");
        let rs = reseat_of(&mesh, &surf, &spec);
        eprintln!("reseat snapped cube: {} points", rs.points.len());
        assert!(rs.points.is_empty(), "the cube re-seats nothing");
    }

    #[test]
    fn the_snapped_sphere_reseats_its_row_one_points() {
        let (surf, mesh) = snapped_sphere_case();
        let spec = sphere_layers(0.05);
        let n_internal = mesh.neighbour.len().min(mesh.faces.len());
        let patch = mesh
            .patches
            .iter()
            .find(|p| p.name == "sphere")
            .expect("the sphere patch");
        assert_eq!(patch.size, 2592, "the sphere patch has 2592 faces");
        let theta = predicted_angles(&mesh).expect("predicted_angles");
        let faces: Vec<usize> =
            (0..patch.size).map(|j| n_internal + patch.start + j).collect();
        let worst = faces.iter().map(|&fa| theta[fa]).fold(0.0, Scalar::max);
        let n_trigger = faces.iter().filter(|&&fa| theta[fa] > RESEAT_THETA_ON_DEG).count();
        let mut owners: Vec<usize> = faces
            .iter()
            .filter(|&&fa| theta[fa] > RESEAT_THETA_ON_DEG)
            .map(|&fa| mesh.owner[fa] as usize)
            .collect();
        owners.sort_unstable();
        owners.dedup();
        eprintln!("theta sphere: worst {worst:.6} deg, {n_trigger} past 45, {} owners", owners.len());
        assert!(worst > 80.25 && worst < 80.26, "worst theta {worst}");
        assert_eq!(n_trigger, 1632);
        assert_eq!(owners.len(), 804);

        let rs = reseat_of(&mesh, &surf, &spec);
        eprintln!("reseat sphere: {} points, per_patch {:?}", rs.points.len(), rs.per_patch);
        assert_eq!(rs.points.len(), 888);
        assert_eq!(rs.per_patch, vec![888]);
        let mut worst_w = 0usize;
        for w in &rs.wall {
            assert!((1..=3).contains(&w.len()), "a W(j) has {} entries", w.len());
            worst_w = worst_w.max(w.len());
        }
        eprintln!("reseat sphere: largest W(j) holds {worst_w} wall points");
        assert_eq!(worst_w, 3);
        assert!(
            rs.points.windows(2).all(|w| w[0] < w[1]),
            "J is not strictly ascending"
        );
        let mut is_b = vec![false; mesh.points.len()];
        for fi in n_internal..mesh.faces.len() {
            for &p in &mesh.faces[fi] {
                is_b[p as usize] = true;
            }
        }
        assert!(
            rs.points.iter().all(|&j| !is_b[j]),
            "a J point sits on a boundary face"
        );
        let out = add_layers(&mesh, &surf, &spec, &thresholds()).expect("layers");
        let row = out
            .report
            .patches
            .iter()
            .find(|p| p.name == "sphere")
            .expect("the sphere row");
        eprintln!(
            "reseat sphere row: n_reseated_points {}, n_layers {}, full_area_frac {:.6}, drop_cause {:?}",
            row.n_reseated_points,
            row.n_layers,
            row.full_area_frac,
            row.drop_cause.as_ref().map(|c| c.as_str())
        );
        assert_eq!(row.n_reseated_points, 888);
        for e in &out.report.ladder {
            eprintln!("sphere ladder {e:?}");
        }
    }

    /// The snapped level-2 sphere with three layers of 0.05 KEEPS them: the
    /// re-seat of (92.63)-(92.65) moves 888 row-1 points, the gate passes at
    /// its own thresholds with every level-n face under 70 degrees, and the
    /// trace is one inner and one outer pass. The (92.45) limiter, not G4,
    /// now sets the thickness: full 0, mean_frac 0.346.
    #[test]
    fn the_snapped_sphere_keeps_its_layers_where_the_reseat_acts() {
        let (surf, mesh) = snapped_sphere_case();
        assert_eq!(cell_count(&mesh), 4920);
        let out = add_layers(&mesh, &surf, &sphere_layers(0.05), &thresholds())
            .expect("layers on the snapped sphere");
        assert_eq!(cell_count(&out.mesh), 4920 + 3 * 2592);
        assert!(out.quality.passed());
        assert!(
            (out.quality.max_non_orth_deg - 66.05797649243732).abs() < 1e-9,
            "max_non_orth_deg {}",
            out.quality.max_non_orth_deg
        );
        assert!(out.quality.max_non_orth_deg < 70.0);
        let row = out
            .report
            .patches
            .iter()
            .find(|p| p.name == "sphere")
            .expect("the report carries a row for the sphere");
        assert_eq!(row.n_layers, 3);
        assert!(row.dropped.is_none(), "{:?}", row.dropped);
        assert!(row.drop_cause.is_none(), "{:?}", row.drop_cause);
        assert_eq!(row.n_reseated_points, 888);
        assert_eq!(row.full_area_frac, 0.0);
        assert!(
            (row.mean_frac - 0.34603400766355).abs() < 1e-9,
            "mean_frac {}",
            row.mean_frac
        );
        check_ladder(&out.report.ladder);
        check_rows(&out);
        let names = vec!["sphere".to_string()];
        assert_eq!(
            out.report.ladder,
            vec![
                LadderEntry::new(Ladder::Inner, 0, &names, &[], Outcome::Pass, None),
                LadderEntry::new(Ladder::Outer, 0, &names, &[], Outcome::Pass, None),
            ]
        );
        eprintln!(
            "reseat keeps: cells {} -> {}, max_non_orth_deg {:.6}, mean_frac {:.12}, full_area_frac {}",
            cell_count(&mesh),
            cell_count(&out.mesh),
            out.quality.max_non_orth_deg,
            row.mean_frac,
            row.full_area_frac
        );
    }

    /// The shape (92.66) puts on a run, whatever the case. Also runs
    /// [`check_ladder`] and [`check_rows`].
    fn check_beta(out: &Layered) {
        check_ladder(&out.report.ladder);
        check_rows(out);
        for e in &out.report.ladder {
            if e.outcome == Outcome::Beta {
                assert!(!e.gates.is_empty(), "{e:?}");
                assert!(e.beta_points > 0, "{e:?}");
                assert!(e.give_up.is_none() && e.dropped.is_none(), "{e:?}");
            }
        }
        // Per ladder, the rungs taken on the patch set in force: the count
        // resets when the set changes - and, on the inner ladder, when the
        // round does - and no count passes the limit.
        let mut counts: Vec<((Ladder, Vec<String>, Option<usize>), usize)> = Vec::new();
        for e in &out.report.ladder {
            let key = (
                e.ladder,
                e.patches.clone(),
                if e.ladder == Ladder::Inner { Some(e.round) } else { None },
            );
            let c = counts
                .iter()
                .find(|(k, _)| *k == key)
                .map_or(0, |(_, c)| *c);
            assert_eq!(e.beta_rung, c, "{e:?}");
            let c = if e.outcome == Outcome::Beta { c + 1 } else { c };
            assert!(c <= BETA_RUNG_LIMIT, "{e:?}");
            match counts.iter_mut().find(|(k, _)| *k == key) {
                Some(slot) => slot.1 = c,
                None => counts.push((key, c)),
            }
        }
        for p in &out.report.patches {
            let n = out
                .report
                .ladder
                .iter()
                .filter(|e| e.outcome == Outcome::Beta && e.patches.iter().any(|nm| *nm == p.name))
                .count();
            assert_eq!(p.beta_rungs, n, "{}", p.name);
        }
        for p in &out.report.patches {
            if p.dropped.is_some() || p.n_layers == 0 {
                assert_eq!(p.level_n_non_orth_max_deg, None, "{}", p.name);
            } else {
                let v = p
                    .level_n_non_orth_max_deg
                    .expect("a kept row carries its level-n angle");
                assert!(v.is_finite() && v >= 0.0, "{} {v}", p.name);
                assert!(v <= out.quality.max_non_orth_deg + 1e-9, "{} {v}", p.name);
            }
        }
    }

    /// (92.66)'s step and its limit: 1, 1/2, 1/4, 0, and 0 stays 0; the
    /// pull at beta 1/4 is a quarter of the way to the wall-following
    /// position; the trace names the rung `beta`.
    #[test]
    fn the_beta_step_runs_one_half_quarter_zero() {
        assert_eq!(beta_step(1.0), 0.5);
        assert_eq!(beta_step(0.5), 0.25);
        assert_eq!(beta_step(0.25), 0.0);
        assert_eq!(beta_step(0.0), 0.0);
        assert_eq!(BETA_RUNG_LIMIT, 3);
        let wall = vec![
            (Vec3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 0.1), Vec3::new(0.0, 0.0, 1.0)),
            (Vec3::new(1.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 0.2), Vec3::new(0.0, 0.0, 1.0)),
        ];
        let d = reseat_disp(Vec3::new(0.0, 0.0, 1.0), &wall, 0.25);
        eprintln!("beta step: reseat_disp at beta 1/4 = ({}, {}, {})", d.x, d.y, d.z);
        assert!((d.x - 0.125).abs() < 1e-15, "{d:?}");
        assert!(d.y.abs() < 1e-15, "{d:?}");
        assert!((d.z - 0.20177669529663692).abs() < 1e-15, "{d:?}");
        assert_eq!(Outcome::Beta.as_str(), "beta");
    }

    /// A wall on the cell planes has no re-seat point, so no beta rung is
    /// ever taken: no `beta` entry, every counter 0, and the kept rows
    /// still report their level-n angle.
    #[test]
    fn a_wall_on_the_cell_planes_takes_no_beta_rung() {
        for (name, case, spec) in [
            ("cube", castellated_cube_case(), cube_layers(0.02)),
            ("gap", castellated_gap_case(), gap_layers(0.0)),
        ] {
            let (surf, mesh) = case;
            let out = add_layers(&mesh, &surf, &spec, &thresholds()).expect(name);
            check_beta(&out);
            for e in &out.report.ladder {
                assert!(e.outcome != Outcome::Beta, "{e:?}");
                assert_eq!(e.beta_rung, 0, "{e:?}");
                assert_eq!(e.beta_points, 0, "{e:?}");
            }
            for p in &out.report.patches {
                assert_eq!(p.beta_rungs, 0, "{}", p.name);
                if p.dropped.is_none() && p.n_layers > 0 {
                    eprintln!(
                        "beta {name} patch \"{}\": level_n_non_orth_max_deg {:?}",
                        p.name, p.level_n_non_orth_max_deg
                    );
                }
            }
        }
    }

    /// The snapped cube has no wall face past 45 degrees, so `J` is empty
    /// and the outer ladder gives up as before: no `beta` entry anywhere,
    /// `retreats` 4, and the row reports null.
    #[test]
    fn the_snapped_cube_takes_no_beta_rung() {
        let (surf, mesh) = snapped_cube_case();
        let spec = LayerSpec {
            patches: vec!["cube".to_string()],
            n: 3,
            first_thickness: 0.02,
            growth: 1.3,
            min_thickness: 0.0,
            ..LayerSpec::default()
        };
        let out = add_layers(&mesh, &surf, &spec, &thresholds()).expect("layers");
        check_beta(&out);
        for e in &out.report.ladder {
            assert!(e.outcome != Outcome::Beta, "{e:?}");
        }
        let row = out
            .report
            .patches
            .iter()
            .find(|p| p.name == "cube")
            .expect("the report carries a row for the cube");
        assert_eq!(row.beta_rungs, 0);
        assert_eq!(row.n_layers, 0);
        assert_eq!(row.level_n_non_orth_max_deg, None);
        assert_eq!(row.drop_cause, Some(DropCause::OuterGate));
        assert_eq!(out.report.retreats, 4);
        eprintln!(
            "beta snapped cube: retreats {}, beta_rungs {}, level_n_non_orth_max_deg {:?}",
            out.report.retreats, row.beta_rungs, row.level_n_non_orth_max_deg
        );
    }

    /// The snapped floor box's pull is what fails the gate, so the FIRST
    /// measurement is a beta rung on the inner ladder - the volume gate
    /// names cells carrying re-seated points - and only then do the
    /// halvings follow.
    #[test]
    fn the_floor_box_takes_the_pull_back_before_the_thickness() {
        let (surf, mesh) = snapped_floor_box_case();
        let spec = LayerSpec {
            patches: vec!["cube".to_string()],
            n: 3,
            first_thickness: 0.02,
            growth: 1.3,
            min_thickness: 0.0,
            ..LayerSpec::default()
        };
        let out = add_layers(&mesh, &surf, &spec, &thresholds()).expect("layers");
        check_beta(&out);
        let first = out.report.ladder.first().expect("a trace");
        assert!(first.ladder == Ladder::Inner, "{first:?}");
        assert_eq!(first.round, 0, "{first:?}");
        assert_eq!(first.rung, 0, "{first:?}");
        assert_eq!(first.beta_rung, 0, "{first:?}");
        assert!(first.outcome == Outcome::Beta, "{first:?}");
        assert!(first.beta_points > 0, "{first:?}");
        assert!(
            first.gates.iter().any(|(g, _)| *g == Gate::Volume),
            "the first entry does not name G1: {first:?}"
        );
        for e in &out.report.ladder {
            eprintln!("beta floor ladder {e:?}");
        }
        let row = out
            .report
            .patches
            .iter()
            .find(|p| p.name == "cube")
            .expect("the report carries a row for the cube");
        eprintln!(
            "beta floor cube: n_layers {} full_area_frac {} drop_cause {:?} dropped {:?} n_reseated_points {} beta_rungs {} level_n_non_orth_max_deg {:?} max_non_orth_deg {}",
            row.n_layers,
            row.full_area_frac,
            row.drop_cause,
            row.dropped,
            row.n_reseated_points,
            row.beta_rungs,
            row.level_n_non_orth_max_deg,
            out.quality.max_non_orth_deg
        );
    }

    /// The snapped sphere with eight layers lowers its pull on the OUTER
    /// ladder: a failing extruded mesh's cells carry re-seated points, so
    /// a `beta` entry names the outer ladder before any cap is halved.
    #[test]
    fn the_sphere_with_eight_layers_lowers_its_pull_on_the_outer_ladder() {
        let (surf, mesh) = snapped_sphere_case();
        let spec = LayerSpec {
            patches: vec!["sphere".to_string()],
            n: 8,
            first_thickness: 0.006,
            growth: 1.0,
            ..LayerSpec::default()
        };
        let out = add_layers(&mesh, &surf, &spec, &thresholds()).expect("layers");
        check_beta(&out);
        assert!(
            out.report
                .ladder
                .iter()
                .any(|e| e.ladder == Ladder::Outer && e.outcome == Outcome::Beta),
            "no outer beta rung: {:?}",
            out.report.ladder
        );
        let row = out
            .report
            .patches
            .iter()
            .find(|p| p.name == "sphere")
            .expect("the report carries a row for the sphere");
        assert!(row.beta_rungs >= 1, "{row:?}");
        assert_eq!(row.n_reseated_points, 888);
        for e in &out.report.ladder {
            eprintln!("beta sphere8 ladder {e:?}");
        }
        eprintln!(
            "beta sphere8: n_layers {} full_area_frac {} drop_cause {:?} dropped {:?} n_reseated_points {} beta_rungs {} level_n_non_orth_max_deg {:?} max_non_orth_deg {}",
            row.n_layers,
            row.full_area_frac,
            row.drop_cause,
            row.dropped,
            row.n_reseated_points,
            row.beta_rungs,
            row.level_n_non_orth_max_deg,
            out.quality.max_non_orth_deg
        );
    }

    /// The snapped sphere that KEEPS its three layers reports the worst G4
    /// angle over its level-n faces - above the 45-degree re-seat trigger,
    /// never above the mesh's worst non-orthogonality.
    #[test]
    fn the_kept_sphere_reports_its_level_n_angle() {
        let (surf, mesh) = snapped_sphere_case();
        let out = add_layers(&mesh, &surf, &sphere_layers(0.05), &thresholds())
            .expect("layers on the snapped sphere");
        check_beta(&out);
        for e in &out.report.ladder {
            assert!(e.outcome != Outcome::Beta, "{e:?}");
        }
        let row = out
            .report
            .patches
            .iter()
            .find(|p| p.name == "sphere")
            .expect("the report carries a row for the sphere");
        assert_eq!(row.n_layers, 3);
        assert_eq!(row.beta_rungs, 0);
        let v = row
            .level_n_non_orth_max_deg
            .expect("the kept sphere reports its level-n angle");
        assert!(45.0 < v, "{v}");
        assert!(v <= 66.05797649243732 + 1e-9, "{v}");
        eprintln!("beta sphere level_n_non_orth_max_deg {v:.17}");
        eprintln!("beta sphere max_non_orth_deg {:.17}", out.quality.max_non_orth_deg);
    }

    // ---- §92.13's probe ---------------------------------------------------
    //
    // Diagnosis only: which cells fail G5 on a snapped layer config, cell by
    // cell and round by round, printed. It asserts nothing but that its own
    // step-by-step reproduction of add_layers's outer ladder of (92.47)
    // agrees with the trace the real call took. No threshold, limiter or
    // ladder rule is changed by it, and nothing here runs unless
    // AUTOMESHER_G5_PROBE names a config path.

    use crate::mesh::HostMesh;

    /// The median of `v`, destroying its order: the two middles' mean on an
    /// even length.
    fn g5p_p50(v: &mut Vec<Scalar>) -> Scalar {
        v.sort_by(|x, y| x.partial_cmp(y).unwrap_or(std::cmp::Ordering::Equal));
        if v.is_empty() {
            return 0.0;
        }
        if v.len() % 2 == 1 {
            v[v.len() / 2]
        } else {
            0.5 * (v[v.len() / 2 - 1] + v[v.len() / 2])
        }
    }

    /// The hint the shrink gives its index: the mean edge length over the
    /// layer patches' faces.
    fn g5p_hint(m: &PolyMeshRaw, patches: &[usize]) -> Scalar {
        let n_internal = m.neighbour.len().min(m.faces.len());
        let mut sum = 0.0 as Scalar;
        let mut cnt = 0usize;
        for &p in patches {
            let patch = &m.patches[p];
            for j in 0..patch.size {
                let f = n_internal + patch.start + j;
                if f >= m.faces.len() {
                    continue;
                }
                let face = &m.faces[f];
                for k in 0..face.len() {
                    let a = m.points[face[k] as usize];
                    let b = m.points[face[(k + 1) % face.len()] as usize];
                    sum += (a - b).mag();
                    cnt += 1;
                }
            }
        }
        assert!(cnt > 0, "no layer face to build the index hint from");
        sum / (cnt as Scalar)
    }

    /// G5's planar grouping (§92.3): the largest planar face group's area
    /// over a cell's faces, given as (outward unit normal, |Sf|).
    fn g5p_a_max(c_faces: &[(Vec3, Scalar)]) -> Scalar {
        let cos_tol = quality::PLANAR_GROUP_DEG.to_radians().cos();
        let mut groups: Vec<(Vec3, Scalar)> = Vec::new();
        for (n, area) in c_faces {
            match groups
                .iter_mut()
                .find(|(g, _): &&mut (Vec3, Scalar)| g.dot(*n) >= cos_tol)
            {
                Some((_, a)) => *a += *area,
                None => groups.push((*n, *area)),
            }
        }
        groups.iter().map(|(_, a)| *a).fold(0.0, Scalar::max)
    }

    /// The A_max of (92.14) for one cell of a host mesh.
    fn g5p_a_max_of(hm: &HostMesh, c: usize) -> Scalar {
        let mut c_faces: Vec<(Vec3, Scalar)> = Vec::new();
        for k in hm.cf_offset[c] as usize..hm.cf_offset[c + 1] as usize {
            let f = hm.cf_face[k] as usize;
            let mag = hm.mag_sf[f];
            if mag <= 0.0 {
                continue;
            }
            let sign = if hm.cf_own[k] != 0 { 1.0 } else { -1.0 };
            c_faces.push((hm.sf[f] * (sign / mag), mag));
        }
        for k in hm.bcf_offset[c] as usize..hm.bcf_offset[c + 1] as usize {
            let bf = hm.bcf_face[k] as usize;
            let mag = hm.b_mag_sf[bf];
            if mag <= 0.0 {
                continue;
            }
            c_faces.push((hm.b_sf[bf] * (1.0 / mag), mag));
        }
        g5p_a_max(&c_faces)
    }

    /// `3 V / A_max^1.5` of (92.14), the value G5 fails a cell on.
    fn g5p_tau(hm: &HostMesh, c: usize) -> Scalar {
        if hm.v[c] <= 0.0 {
            return Scalar::INFINITY;
        }
        let amax = g5p_a_max_of(hm, c);
        if amax <= 0.0 {
            return Scalar::INFINITY;
        }
        3.0 * hm.v[c] / amax.powf(1.5)
    }

    /// A variation the stack or the ladder refused, named short.
    fn g5p_refused(name: &str, value: &str, e: &Error) {
        let msg: String = format!("{e}").chars().take(120).collect();
        println!("g5probe vary {name} {value} refused {msg}");
    }

    #[test]
    #[ignore]
    fn a_probe_measures_which_cells_fail_g5_on_a_snapped_layer_config() {
        let Ok(cfg_str) = std::env::var("AUTOMESHER_G5_PROBE") else {
            println!("g5probe skipped: AUTOMESHER_G5_PROBE is unset");
            return;
        };
        let cfg_path = std::path::PathBuf::from(cfg_str.clone());
        let cfg =
            crate::automesher::read_config(&cfg_path).expect("read the probe config");
        cfg.validate().expect("validate the probe config");
        let dir = cfg_path
            .parent()
            .map(std::path::Path::to_path_buf)
            .unwrap_or_default();
        let mut parts: Vec<Surface> = Vec::new();
        for s in &cfg.input.surfaces {
            let mut part = crate::surface::stl::read_stl(dir.join(&s.path))
                .expect("read the probe surface");
            if let Some(name) = &s.name {
                part.patch_names = vec![name.clone()];
                part.tri_patch = vec![0; part.tris.len()];
                part.patch_area = vec![part.patch_area.iter().sum()];
            }
            parts.push(part);
        }
        let surf = if parts.len() == 1 {
            parts.pop().expect("one surface")
        } else {
            Surface::merge(parts).expect("merge the probe surfaces")
        };
        surf.require_closed().expect("the probe surface is closed");
        let mut quiet = |_: &str| {};
        let out = crate::automesher::driver::run(
            &cfg,
            &surf,
            Some(crate::automesher::driver::Stage::Snap),
            &mut quiet,
        )
        .expect("the stages up to snap");
        let m = out.mesh;
        let thr = cfg.quality.thresholds();
        let spec0 = cfg.layers.clone();
        let st0 = stack(&spec0).expect("the probe stack");
        let patches = if spec0.patches.is_empty() {
            Vec::new()
        } else {
            resolve_patches(&m, &spec0).expect("the probe patches")
        };
        let hm_m = build_host_mesh(&m).expect("host mesh of the snapped mesh");
        let n_internal_m = m.neighbour.len().min(m.faces.len());
        println!(
            "g5probe config {} n {} t1 {:.6e} g {:.6e} T {:.6e} cell_frac {:.6e} medial_frac {:.6e} retreat_limit {}",
            cfg_str, spec0.n, spec0.first_thickness, spec0.growth, st0.total,
            spec0.cell_frac, spec0.medial_frac, spec0.retreat_limit
        );
        let mut wall: Vec<(usize, Scalar)> = Vec::new();
        let mut short_min = Scalar::INFINITY;
        for &p in &patches {
            let patch = &m.patches[p];
            for j in 0..patch.size {
                let f = n_internal_m + patch.start + j;
                if f >= m.faces.len() {
                    continue;
                }
                wall.push((f, hm_m.b_mag_sf[f - n_internal_m]));
                let pts = &m.faces[f];
                for k in 0..pts.len() {
                    let a = m.points[pts[k] as usize];
                    let b = m.points[pts[(k + 1) % pts.len()] as usize];
                    short_min = short_min.min((a - b).mag());
                }
            }
        }
        let mut areas: Vec<Scalar> = wall.iter().map(|&(_, a)| a).collect();
        let area_max = areas.iter().copied().fold(0.0, Scalar::max);
        let flat_floor = area_max.sqrt() / 60.0;
        println!(
            "g5probe wall faces {} area_max {:.6e} flat_floor {:.6e} area_p50 {:.6e} short_edge_min {:.6e}",
            wall.len(),
            area_max,
            flat_floor,
            g5p_p50(&mut areas),
            short_min
        );
        let layered = add_layers(&m, &surf, &spec0, &thr).expect("layers end to end");
        for e in layered
            .report
            .ladder
            .iter()
            .filter(|e| e.ladder == Ladder::Outer)
        {
            let gates = e
                .gates
                .iter()
                .map(|(g, c)| format!("{}:{}", gate_label(*g), c))
                .collect::<Vec<_>>()
                .join(" ");
            println!(
                "g5probe trace round {} outcome {} rung {} beta_rung {} gates {}",
                e.round,
                e.outcome.as_str(),
                e.rung,
                e.beta_rung,
                gates
            );
        }
        let row0 = layered
            .report
            .patches
            .iter()
            .find(|r| Some(&r.name) == spec0.patches.first())
            .or_else(|| layered.report.patches.first());
        let (nl, fu, dc, br) = match row0 {
            Some(r) => (
                r.n_layers,
                r.full_area_frac,
                r.drop_cause.map(|c| c.as_str()).unwrap_or("none"),
                r.beta_rungs,
            ),
            None => (0, 0.0, "none", 0),
        };
        println!(
            "g5probe final n_layers {} full {:.6e} drop_cause {} retreats {} beta_rungs {}",
            nl, fu, dc, layered.report.retreats, br
        );
        let outer_trace: Vec<LadderEntry> = layered
            .report
            .ladder
            .iter()
            .filter(|e| e.ladder == Ladder::Outer)
            .cloned()
            .collect();
        let f0 = {
            let hint = g5p_hint(&m, &patches);
            let idx = TriIndex::new(&surf, hint).expect("the probe's tri index");
            field(&m, &idx, &patches, &spec0, &st0).expect("the probe's field")
        };
        g5p_emulate(
            &m, &surf, &spec0, &thr, &patches, &st0, &hm_m, n_internal_m,
            &outer_trace, &f0,
        );
        let n_points = m.points.len();
        let b1 = vec![1.0 as Scalar; n_points];
        for mult in [0.90f64, 0.95, 1.00, 1.05, 1.10, 1.20, 1.43] {
            let mut s = spec0.clone();
            s.first_thickness = spec0.first_thickness * mult;
            g5p_vary(&m, &surf, &thr, &patches, &s, "t1", &format!("{mult:.2}"), &b1);
        }
        for nn in [1usize, 2, 4, 8] {
            let mut s = spec0.clone();
            s.n = nn;
            g5p_vary(&m, &surf, &thr, &patches, &s, "n", &nn.to_string(), &b1);
        }
        for gv in [1.0f64, 1.05, 1.1, 1.2] {
            let mut s = spec0.clone();
            s.growth = gv;
            g5p_vary(&m, &surf, &thr, &patches, &s, "g", &format!("{gv:.2}"), &b1);
        }
        for cf in [spec0.cell_frac, 1.0] {
            let mut s = spec0.clone();
            s.cell_frac = cf;
            g5p_vary(&m, &surf, &thr, &patches, &s, "cell_frac", &format!("{cf:.2}"), &b1);
        }
        let b0 = vec![0.0 as Scalar; n_points];
        let s1 = g5p_vary(&m, &surf, &thr, &patches, &spec0, "beta", "1", &b1);
        let s0 = g5p_vary(&m, &surf, &thr, &patches, &spec0, "beta", "0", &b0);
        if let (Some(x), Some(y)) = (&s1, &s0) {
            println!("g5probe beta_same_set {}", x == y);
        }
        g5p_bisect(&m, &surf, &thr, &patches, &spec0, flat_floor);
    }

    /// The outer ladder of (92.47), reproduced step by step from the same
    /// `attempt` calls `add_layers` makes, asserting every round against the
    /// trace the real call took and printing every failing G5 cell.
    #[allow(clippy::too_many_arguments)]
    fn g5p_emulate(
        m: &PolyMeshRaw,
        surf: &Surface,
        spec: &LayerSpec,
        thr: &quality::QualityThresholds,
        patches: &[usize],
        st: &Stack,
        hm_m: &HostMesh,
        n_internal_m: usize,
        trace: &[LadderEntry],
        f0: &Field,
    ) {
        let n_points = m.points.len();
        let mut patches_e = patches.to_vec();
        let mut caps = vec![1.0 as Scalar; n_points];
        let mut betas = vec![1.0 as Scalar; n_points];
        let mut halvings = 0usize;
        let mut beta_rungs = 0usize;
        let mut prev: Option<std::collections::BTreeSet<usize>> = None;
        let mut r = 0usize;
        loop {
            let a = attempt(
                m, surf, spec, thr, &patches_e, &caps, &betas,
                &vec![false; m.faces.len()], &vec![false; m.faces.len()],
            )
            .expect("the probe's attempt");
            let hm_a = build_host_mesh(&a.mesh).expect("host mesh of the attempt");
            let rep =
                quality::measure_capped(&a.mesh, thr, usize::MAX).expect("the probe's measure");
            let g5f = rep.failures.iter().find(|f| f.gate == Gate::Thickness);
            let g5_total = g5f.map(|f| f.n_failed).unwrap_or(0);
            let cells: Vec<usize> = g5f
                .map(|f| f.subjects.iter().map(|s| s.id).collect())
                .unwrap_or_default();
            let te = trace
                .get(r)
                .unwrap_or_else(|| panic!("the trace has no outer entry for round {r}"));
            let (emu, fail_pts, jf): (Outcome, Vec<usize>, Vec<usize>) = if a.quality.passed() {
                (Outcome::Pass, Vec::new(), Vec::new())
            } else {
                let n = a.extrusion.n;
                let mut slots = vec![
                    0usize;
                    a.extrusion
                        .slot_of_point
                        .iter()
                        .filter(|&&s| s >= 0)
                        .count()
                ];
                for (i, &s) in a.extrusion.slot_of_point.iter().enumerate() {
                    if s >= 0 {
                        slots[s as usize] = i;
                    }
                }
                let orig_of = |p: crate::Label| -> usize {
                    let p = p as usize;
                    if p < n_points {
                        p
                    } else {
                        slots[(p - n_points) / n]
                    }
                };
                let n_faces_out = a.mesh.faces.len();
                let n_internal_out = a.mesh.neighbour.len().min(n_faces_out);
                let n_cells_out = a
                    .mesh
                    .owner
                    .iter()
                    .chain(a.mesh.neighbour.iter())
                    .copied()
                    .max()
                    .map_or(0, |x| x as usize + 1);
                let mut cell_points: Vec<Vec<u32>> = vec![Vec::new(); n_cells_out];
                for (f, face) in a.mesh.faces.iter().enumerate() {
                    for &p in face {
                        cell_points[a.mesh.owner[f] as usize].push(orig_of(p) as u32);
                    }
                    if f < n_internal_out {
                        for &p in face {
                            cell_points[a.mesh.neighbour[f] as usize].push(orig_of(p) as u32);
                        }
                    }
                }
                for list in cell_points.iter_mut() {
                    list.sort_unstable();
                    list.dedup();
                }
                let is_layer: Vec<bool> = a
                    .extrusion
                    .slot_of_point
                    .iter()
                    .map(|&s| s >= 0)
                    .collect();
                let fp = failing_points(&a.quality, &a.mesh, n_internal_out, &is_layer, &cell_points);
                let mut live = vec![false; n_points];
                for &j in &a.reseat_points {
                    live[j] = a.beta[j] > 0.0;
                }
                let jf = failing_points(&a.quality, &a.mesh, n_internal_out, &live, &cell_points);
                let oc = if !jf.is_empty() && beta_rungs < BETA_RUNG_LIMIT {
                    Outcome::Beta
                } else if fp.is_empty() || halvings >= spec.retreat_limit {
                    Outcome::GiveUp
                } else {
                    Outcome::Retreat
                };
                (oc, fp, jf)
            };
            assert_eq!(te.round, r, "round");
            assert_eq!(te.outcome, emu, "round {r} outcome");
            assert_eq!(te.rung, halvings, "round {r} rung");
            assert_eq!(te.beta_rung, beta_rungs, "round {r} beta_rung");
            assert_eq!(&failing_gates(&a.quality), &te.gates, "round {r} gates");
            let mut step_pts = 0usize;
            match emu {
                Outcome::Beta => {
                    step_pts = jf.len();
                    for &j in &jf {
                        betas[j] = beta_step(a.beta[j]);
                    }
                    beta_rungs += 1;
                }
                Outcome::GiveUp => {
                    let mut counts = vec![0usize; patches_e.len()];
                    for (k, &p) in patches_e.iter().enumerate() {
                        let patch = &m.patches[p];
                        let mut seen = vec![false; n_points];
                        for j in 0..patch.size {
                            let fa = n_internal_m + patch.start + j;
                            if fa >= m.faces.len() {
                                continue;
                            }
                            for &pt in &m.faces[fa] {
                                seen[pt as usize] = true;
                            }
                        }
                        for &i in &fail_pts {
                            if seen[i] {
                                counts[k] += 1;
                            }
                        }
                    }
                    let mut victim = 0usize;
                    for k in 1..counts.len() {
                        if counts[k] > counts[victim] {
                            victim = k;
                        }
                    }
                    patches_e.remove(victim);
                    caps = vec![1.0 as Scalar; n_points];
                    halvings = 0;
                    betas = vec![1.0 as Scalar; n_points];
                    beta_rungs = 0;
                }
                Outcome::Retreat => {
                    step_pts = fail_pts.len();
                    for &i in &fail_pts {
                        caps[i] *= 0.5;
                    }
                    halvings += 1;
                }
                Outcome::Pass => {}
                Outcome::Terminate => unreachable!("the oracle emulates patch mode"),
            }
            let mut hist: std::collections::BTreeMap<u64, usize> = Default::default();
            for (i, &s) in a.extrusion.slot_of_point.iter().enumerate() {
                if s >= 0 {
                    *hist.entry(caps[i].to_bits()).or_insert(0) += 1;
                }
            }
            let mut keys: Vec<Scalar> = hist
                .keys()
                .map(|b| Scalar::from_bits(*b))
                .collect();
            keys.sort_by(|x, y| y.partial_cmp(x).unwrap_or(std::cmp::Ordering::Equal));
            let caps_s = keys
                .iter()
                .map(|k| format!("{}:{}", k, hist[&k.to_bits()]))
                .collect::<Vec<_>>()
                .join(",");
            let n = a.extrusion.n;
            let first = a.extrusion.first_cell;
            let mut by_k = vec![0usize; n];
            let mut jset: std::collections::BTreeSet<usize> = Default::default();
            let owners: std::collections::BTreeSet<usize> = a
                .extrusion
                .layer_faces
                .iter()
                .map(|&f| m.owner[f] as usize)
                .collect();
            let mut row1 = 0usize;
            let mut other = 0usize;
            let mut taus: Vec<Scalar> = Vec::new();
            for &c in &cells {
                taus.push(g5p_tau(&hm_a, c));
                if c >= first {
                    by_k[(c - first) % n] += 1;
                    jset.insert((c - first) / n);
                } else if owners.contains(&c) {
                    row1 += 1;
                } else {
                    other += 1;
                }
            }
            let (persist, fresh) = match &prev {
                None => (0usize, cells.len()),
                Some(p) => {
                    let cur: std::collections::BTreeSet<usize> = cells.iter().copied().collect();
                    (cur.intersection(p).count(), cur.difference(p).count())
                }
            };
            let mut ts = taus.clone();
            let tau_p50 = g5p_p50(&mut ts);
            let tau_min = ts.first().copied().unwrap_or(0.0);
            println!(
                "g5probe round {} g5_total {} g5_seen_by_ladder {} layer_by_k [{}] row1 {} other {} faces {} persist {} new {} tau_min {:.6e} tau_p50 {:.6e} fail_pts {} caps [{}]",
                r,
                g5_total,
                g5_total.min(quality::GATE_CELL_CAP),
                by_k.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(","),
                row1,
                other,
                jset.len(),
                persist,
                fresh,
                tau_min,
                tau_p50,
                step_pts,
                caps_s
            );
            if r == 0 {
                g5p_face_table(&a, m, patches, st, f0, &hm_a, hm_m, n_internal_m, &cells);
            }
            prev = Some(cells.iter().copied().collect());
            match emu {
                Outcome::Pass | Outcome::GiveUp => break,
                _ => {}
            }
            if patches_e.is_empty() || a.extrusion.layer_faces.is_empty() {
                panic!("the emulation reached the ladder's refusal - the trace should have ended");
            }
            r += 1;
        }
    }

    /// The round-0 wall faces with a failing layer cell, worst first, with
    /// the failing cell's tau decomposed against its wall face.
    #[allow(clippy::too_many_arguments)]
    fn g5p_face_table(
        a: &Layered,
        m: &PolyMeshRaw,
        patches: &[usize],
        st: &Stack,
        f0: &Field,
        hm_a: &HostMesh,
        hm_m: &HostMesh,
        n_internal_m: usize,
        cells: &[usize],
    ) {
        let n = a.extrusion.n;
        let first = a.extrusion.first_cell;
        let mut per_face: std::collections::BTreeMap<usize, Vec<usize>> = Default::default();
        for &c in cells {
            if c >= first {
                per_face.entry((c - first) / n).or_default().push(c);
            }
        }
        let mut rows: Vec<(Scalar, usize, usize, Scalar, usize)> = Vec::new();
        for (&block, cs) in &per_face {
            let fa = a.extrusion.layer_faces[block];
            let a_wall = hm_m.b_mag_sf[fa - n_internal_m];
            let mut best = (Scalar::INFINITY, 0usize, 0usize);
            for &c in cs.iter() {
                let t = g5p_tau(hm_a, c);
                if t < best.0 {
                    best = (t, (c - first) % n, c);
                }
            }
            rows.push((best.0, best.1, best.2, a_wall, fa));
        }
        rows.sort_by(|x, y| x.0.partial_cmp(&y.0).unwrap_or(std::cmp::Ordering::Equal));
        for (i, &(tmin, kmin, cell, a_wall, fa)) in rows.iter().enumerate().take(300) {
            let block = (cell - first) / n;
            let mut d_min = Scalar::INFINITY;
            let mut d_max = 0.0 as Scalar;
            let mut ti_min = Scalar::INFINITY;
            let mut dir = Vec3::ZERO;
            let mut dir_n = 0usize;
            for &p in &m.faces[fa] {
                let p = p as usize;
                let s = a.extrusion.slot_of_point[p];
                if s < 0 {
                    continue;
                }
                let s = s as usize;
                let x0 = a.mesh.points[a.extrusion.level_point[0][s] as usize];
                let xn = a.mesh.points[a.extrusion.level_point[n][s] as usize];
                let d = x0 - xn;
                let dm = d.mag();
                d_min = d_min.min(dm / st.total);
                d_max = d_max.max(dm / st.total);
                if dm > 0.0 {
                    dir = dir + d * (1.0 / dm);
                    dir_n += 1;
                }
                ti_min = ti_min.min(f0.thickness[p] / st.total);
            }
            let where_face = patches.iter().enumerate().find_map(|(pi, &p)| {
                let patch = &m.patches[p];
                (n_internal_m + patch.start..n_internal_m + patch.start + patch.size)
                    .position(|f| f == fa)
                    .map(|o| (pi, o))
            });
            let (pidx, pj) = where_face.expect("a failing face of a layer patch");
            let name = &m.patches[patches[pidx]].name;
            let p_a = a
                .mesh
                .patches
                .iter()
                .find(|q| &q.name == name)
                .expect("the wall patch on the extruded mesh");
            let normal = hm_a.b_sf[p_a.start + pj].normalised();
            let mean_dir = if dir_n > 0 {
                dir * (1.0 / dir_n as Scalar)
            } else {
                Vec3::ZERO
            };
            let tilt = if dir_n > 0 {
                normal
                    .dot(mean_dir.normalised())
                    .clamp(-1.0, 1.0)
                    .acos()
                    .to_degrees()
            } else {
                0.0
            };
            let v = hm_a.v[cell];
            let amax = g5p_a_max_of(hm_a, cell);
            let t_k = st.t[kmin];
            let amax_over = amax / a_wall;
            let v_over = v / (a_wall * t_k);
            println!(
                "g5probe face j {} area {:.6e} tau_min {:.6e} k_min {} amax_over_awall {:.6e} v_over_awall_tk {:.6e} d_over_t_min {:.6e} d_over_t_max {:.6e} tilt_deg {:.6e} ti_over_t_min {:.6e}",
                block, a_wall, tmin, kmin, amax_over, v_over, d_min, d_max, tilt, ti_min
            );
            if i < 5 {
                let est = 3.0 * t_k / a_wall.sqrt() * v_over / amax_over.powf(1.5);
                println!(
                    "g5probe face_check j {} tau {:.6e} estimate {:.6e}",
                    block, tmin, est
                );
            }
        }
        let mut fail_pts: std::collections::BTreeSet<usize> = Default::default();
        for &block in per_face.keys() {
            for &p in &m.faces[a.extrusion.layer_faces[block]] {
                fail_pts.insert(p as usize);
            }
        }
        let mut limited = 0usize;
        let mut limited_on = 0usize;
        let mut n_slots = 0usize;
        for (i, &s) in a.extrusion.slot_of_point.iter().enumerate() {
            if s < 0 {
                continue;
            }
            n_slots += 1;
            if f0.thickness[i] < st.total * (1.0 - 1e-12) {
                limited += 1;
                if fail_pts.contains(&i) {
                    limited_on += 1;
                }
            }
        }
        println!(
            "g5probe limiter points {} limited {} limited_on_failing_faces {} of {}",
            n_slots,
            limited,
            limited_on,
            fail_pts.len()
        );
    }

    /// One variation: one attempt at caps 1 and the given betas, then the
    /// full run, on the one changed spec value.
    fn g5p_vary(
        m: &PolyMeshRaw,
        surf: &Surface,
        thr: &quality::QualityThresholds,
        patches: &[usize],
        spec: &LayerSpec,
        name: &str,
        value: &str,
        betas: &[Scalar],
    ) -> Option<std::collections::BTreeSet<usize>> {
        let caps = vec![1.0 as Scalar; m.points.len()];
        let r0 = match attempt(m, surf, spec, thr, patches, &caps, betas, &vec![false; m.faces.len()], &vec![false; m.faces.len()]) {
            Ok(a) => a,
            Err(e) => {
                g5p_refused(name, value, &e);
                return None;
            }
        };
        let rep = quality::measure_capped(&r0.mesh, thr, usize::MAX).expect("the probe's measure");
        let g5f = rep.failures.iter().find(|f| f.gate == Gate::Thickness);
        let g5_total = g5f.map(|f| f.n_failed).unwrap_or(0);
        let set: std::collections::BTreeSet<usize> = g5f
            .map(|f| f.subjects.iter().map(|s| s.id).collect())
            .unwrap_or_default();
        let n = r0.extrusion.n;
        let first = r0.extrusion.first_cell;
        let mut by_k = vec![0usize; n];
        let mut js: std::collections::BTreeSet<usize> = Default::default();
        for &c in &set {
            if c >= first {
                by_k[(c - first) % n] += 1;
                js.insert((c - first) / n);
            }
        }
        let full = match add_layers(m, surf, spec, thr) {
            Ok(f) => f,
            Err(e) => {
                g5p_refused(name, value, &e);
                return Some(set);
            }
        };
        let row = full
            .report
            .patches
            .iter()
            .find(|r| Some(&r.name) == spec.patches.first())
            .or_else(|| full.report.patches.first());
        let (nl, fu, dc, br) = match row {
            Some(r) => (
                r.n_layers,
                r.full_area_frac,
                r.drop_cause.map(|c| c.as_str()).unwrap_or("none"),
                r.beta_rungs,
            ),
            None => (0, 0.0, "none", 0),
        };
        let rounds = full
            .report
            .ladder
            .iter()
            .filter(|e| e.ladder == Ladder::Outer)
            .count();
        println!(
            "g5probe vary {name} {value} r0_g5 {} r0_faces {} r0_layer_by_k [{}] final_n_layers {} full {:.6e} drop_cause {} outer_rounds {} retreats {} beta_rungs {}",
            g5_total,
            js.len(),
            by_k.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(","),
            nl,
            fu,
            dc,
            rounds,
            full.report.retreats,
            br
        );
        Some(set)
    }

    /// The smallest t1 in `[t1, 1.5 t1]` whose round-0 attempt has zero G5
    /// cells, by 14 halvings, and what the full run does there.
    fn g5p_bisect(
        m: &PolyMeshRaw,
        surf: &Surface,
        thr: &quality::QualityThresholds,
        patches: &[usize],
        spec: &LayerSpec,
        flat_floor: Scalar,
    ) {
        let zero = |t1: f64| -> bool {
            let mut s = spec.clone();
            s.first_thickness = t1;
            let caps = vec![1.0 as Scalar; m.points.len()];
            let betas = vec![1.0 as Scalar; m.points.len()];
            match attempt(m, surf, &s, thr, patches, &caps, &betas, &vec![false; m.faces.len()], &vec![false; m.faces.len()]) {
                Ok(a) => {
                    let rep = quality::measure_capped(&a.mesh, thr, usize::MAX)
                        .expect("the probe's measure");
                    !rep.failures.iter().any(|f| f.gate == Gate::Thickness)
                }
                Err(_) => false,
            }
        };
        let hi0 = spec.first_thickness * 1.5;
        if !zero(hi0) {
            println!("g5probe bisect none_below {:.6e}", hi0);
            return;
        }
        let mut lo = spec.first_thickness;
        let mut hi = hi0;
        for _ in 0..14 {
            let mid = 0.5 * (lo + hi);
            if zero(mid) {
                hi = mid;
            } else {
                lo = mid;
            }
        }
        let mut s = spec.clone();
        s.first_thickness = hi;
        let full = add_layers(m, surf, &s, thr).expect("the bisect's full run");
        let row = full
            .report
            .patches
            .iter()
            .find(|r| Some(&r.name) == spec.patches.first())
            .or_else(|| full.report.patches.first());
        let (nl, fu) = match row {
            Some(r) => (r.n_layers, r.full_area_frac),
            None => (0, 0.0),
        };
        println!(
            "g5probe bisect t1_zero_g5 {:.6e} over_flat_floor {:.6e} final_n_layers {} full {:.6e}",
            hi,
            hi / flat_floor,
            nl,
            fu
        );
    }
}
