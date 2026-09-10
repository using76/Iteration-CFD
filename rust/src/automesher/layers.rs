// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.
// Provenance: see PROVENANCE.md. No GPL-licensed source was consulted.

//! Wall layers - SPEC-LIT §92.13: the normal a point grows along
//! (92.40)-(92.42), the thickness it is allowed (92.43)-(92.45), and the
//! shrink that moves the boundary inward (92.46)-(92.47). The stack itself
//! is (92.9) and the medial limit is (92.10), both of §92.2 stage 6. This
//! unit inserts NO cells and changes NO topology: `Shrunk::mesh` differs
//! from its input in `points` alone, and the extrusion that fills the gap
//! is the next unit's.
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
use crate::io::polymesh::PolyMeshRaw;
use crate::surface::{Surface, TriIndex};
use crate::{Scalar, Vec3};

use super::quality::{self, Gate, QualityThresholds};
use super::snap::{face_area_vector, find_hanging};
use super::LayerSpec;

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
        let mut uniq: Vec<Vec3> = Vec::new();
        for &u in &us[i] {
            if !uniq.iter().any(|v| v.dot(u) > 1.0 - 1e-6) {
                uniq.push(u);
            }
        }
        match uniq.len() {
            0 => {} // nothing constrains it
            1 => {
                let u = uniq[0];
                let proj = normal[i] - u * normal[i].dot(u);
                if proj.mag() < 0.1 {
                    pinned[i] = true;
                } else {
                    normal[i] = proj.normalised();
                }
            }
            2 => {
                let cross = uniq[0].cross(uniq[1]);
                if cross.mag() < 1e-9 {
                    // Two planes that do not meet in a line: pinned.
                    pinned[i] = true;
                } else {
                    let c = if cross.dot(normal[i]) < 0.0 {
                        cross * -1.0
                    } else {
                        cross
                    };
                    normal[i] = c.normalised();
                }
            }
            _ => pinned[i] = true, // three constraints leave nothing
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
        let t_i = st.total.min(t_medial).min(spec.cell_frac * h[i]);
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
    let n_faces = mesh.faces.len();
    let n_internal = mesh.neighbour.len().min(n_faces);
    let n_points = mesh.points.len();
    if spec.n == 0 || spec.patches.is_empty() {
        return Ok(Shrunk {
            mesh: mesh.clone(),
            field: empty_field(n_points),
            retreats: 0,
            dropped: Vec::new(),
        });
    }
    let st = stack(spec)?;
    let all_patches = resolve_patches(mesh, spec)?;
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
    for &p in &all_patches {
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
    let idx = TriIndex::new(surf, hint)?;
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
    // The outer patch loop: (92.40) through (92.47) for one patch set at a
    // time, dropping ONE patch per round - the one carrying the most
    // offending layer points, ties to the lower patch index - and
    // recomputing from the input mesh, because a point the dropped patch
    // shared is now constrained into that patch's plane and its normal is a
    // different vector. At most `all_patches.len()` rounds.
    let mut patches = all_patches.clone();
    let mut dropped: Vec<(String, String)> = Vec::new();
    let mut retreats = 0usize;
    loop {
        if patches.is_empty() {
            return Ok(Shrunk {
                mesh: mesh.clone(),
                field: empty_field(n_points),
                retreats,
                dropped,
            });
        }
        let mut f = field(mesh, &idx, &patches, spec, &st)?;
        // The retreat ladder: halve D at the layer points the failing cells
        // carry, re-run the relaxation, and try again, up to
        // `retreat_limit` times.
        let mut accepted = false;
        let mut pts_out = mesh.points.clone();
        let mut give_up: Option<(Vec<usize>, String)> = None;
        let mut halvings = 0usize;
        loop {
            let d = relax(&f, &all_nbrs, &is_b, &hanging, spec);
            let mut work = mesh.clone();
            for i in 0..work.points.len() {
                work.points[i] = work.points[i] + d[i];
            }
            let rep = quality::measure_capped(&work, t, usize::MAX)?;
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
                let limit = spec.min_thickness * st.total;
                let zero: Vec<usize> = (0..n_points)
                    .filter(|&i| f.is_layer[i] && !(d[i].mag() > 0.0))
                    .collect();
                let thin: Vec<usize> = (0..n_points)
                    .filter(|&i| f.is_layer[i] && d[i].mag() < limit)
                    .collect();
                if !zero.is_empty() {
                    give_up = Some((
                        zero,
                        "the applied displacement is zero at a layer point -                          a layer cell there would have a side face of zero area"
                            .to_string(),
                    ));
                } else if !thin.is_empty() {
                    give_up = Some((thin, format!(
                        "the thickness fell below min_thickness * T = {limit:.3e}"
                    )));
                } else {
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
            let fail_pts = failing_points(&rep, mesh, n_internal, &f, &cell_points);
            if halvings >= spec.retreat_limit {
                give_up = Some((fail_pts, format!(
                    "the gate still failed after {halvings} retreat(s)"
                )));
                break;
            }
            if fail_pts.is_empty() {
                give_up = Some((
                    fail_pts,
                    "the gate failed on cells no layer point reaches".to_string(),
                ));
                break;
            }
            for &i in &fail_pts {
                f.disp[i] = f.disp[i] * 0.5;
                f.thickness[i] = f.thickness[i] * 0.5;
            }
            halvings += 1;
        }
        retreats += halvings;
        if accepted {
            return Ok(Shrunk {
                mesh: PolyMeshRaw {
                    points: pts_out,
                    ..mesh.clone()
                },
                field: f,
                retreats,
                dropped,
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
        dropped.push((mesh.patches[vp].name.clone(), reason));
        patches.remove(victim);
    }
}

/// The layer points the gate's failure blames: the subject cell for the
/// cell-named gates, both cells of the face for G4 - whose subject is a
/// FACE, not a cell id. G3 and G7 name no cell a retreat can serve.
fn failing_points(
    rep: &quality::QualityReport,
    mesh: &PolyMeshRaw,
    n_internal: usize,
    f: &Field,
    cell_points: &[Vec<u32>],
) -> Vec<usize> {
    let mut is_fail = vec![false; cell_points.len()];
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
    let mut seen = vec![false; f.is_layer.len()];
    let mut out = Vec::new();
    for (c, bad) in is_fail.iter().enumerate() {
        if !bad {
            continue;
        }
        for &p in &cell_points[c] {
            let i = p as usize;
            if f.is_layer[i] && !seen[i] {
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
/// every level - (92.33)'s line, for (92.33)'s reason and one more.
fn relax(
    f: &Field,
    all_nbrs: &[Vec<u32>],
    is_b: &[bool],
    hanging: &[(u32, [u32; 2])],
    spec: &LayerSpec,
) -> Vec<Vec3> {
    let n = f.disp.len();
    let mut d = f.disp.clone();
    for _ in 0..spec.smoothing_passes {
        let mut next = vec![Vec3::ZERO; n];
        for i in 0..n {
            next[i] = if f.is_layer[i] {
                f.disp[i]
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
    d
}

// ==========================================================================
//  Tests
// ==========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::automesher::castellate::castellate;
    use crate::automesher::castellate::tests::{box_soup, sphere_soup, thresholds};
    use crate::automesher::octree::{
        patch_names, refine_to_surface, Background, Octree,
    };
    use crate::automesher::snap;
    use crate::automesher::{
        CastellationSpec, DistanceBand, DomainSpec, RefinementBand, RefinementSpec,
        SnapSpec,
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
}
