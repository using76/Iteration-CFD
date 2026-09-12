// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.
// Provenance: see PROVENANCE.md. No GPL-licensed source was consulted.

//! Snapping - SPEC-LIT §92.11, the implementation-level companion to §92.2
//! stage 4. Stage 3 leaves a staircase; this stage moves the boundary points
//! onto the STL and undoes any move that breaks a cell. The stage moves
//! `points` and nothing else: the face lists, `owner`, `neighbour` and the
//! patches leave as they entered them, so `PolyMeshRaw::points` is the only
//! array this file writes.
//!
//! The displacement is (92.5) as the dead band (92.28) restates it, measured
//! by [`TriIndex::closest_point`]; the smoothing is (92.6) as (92.29)
//! restates it, with the interior extension; the guarded step is (92.7) as
//! (92.31) restates it. The box constrains where the geometry meets it,
//! (92.30), and a patch the cells never resolved is refused by (92.32)
//! before any point moves.
//!
//! §92.12's feature snapping runs inside the same loop: (92.28)'s target is
//! replaced - when a feature edge, or a corner some one point claims, lies
//! within `snap.feature_tolerance * base_size` of the surface point - by
//! that edge or corner (92.38). A corner admits exactly one point, its
//! claim under (92.39), recomputed from the positions of every iterate.
//!
//! (92.33) is the one this file found rather than implemented: a castellated
//! mesh's 2:1 transitions carry hanging nodes, points that are the midpoint
//! of another face's edge. §92.3's closure gate cancels a cell's face
//! vectors edge by edge, and that cancellation is exact only while every
//! hanging node sits on its parents' segment - true on the lattice,
//! destroyed by any move that does not keep the three collinear. So the
//! guarded step re-seats every hanging node on its parents' midpoint before
//! (92.31) measures, longest parent edge first. Without it the first trial
//! fails closure mesh-wide, every iterate is abandoned whole, and stage 4
//! never moves a point on any refined mesh.
//!
//! Provenance: ORIGINAL - the equations are stated in SPEC-LIT §92.2, §92.3
//! and §92.11, and the closest-point test is Ericson, Real-Time Collision
//! Detection (2005), the barycentric region test, a textbook read as a
//! textbook. No GPL-licensed source was consulted.

use std::collections::{HashMap, HashSet};

use crate::error::{Error, Result};
use crate::io::polymesh::PolyMeshRaw;
use crate::surface::{Surface, TriIndex};
use crate::{Label, Scalar, Vec3};

use super::features::{self, FeatureIndex};
use super::quality::{self, Gate, QualityReport, QualityThresholds};
use super::SnapSpec;

// ==========================================================================
//  The report
// ==========================================================================

/// What stage 4 did, per SPEC-LIT §92.11.
#[derive(Debug, Clone, Default)]
pub struct SnapReport {
    /// |B| of (92.27) - the points at least one wall face carries.
    pub n_boundary_points: usize,
    /// Iterations of (92.7) actually run.
    pub iterations: usize,
    /// True when the loop stopped on `max_i |alpha_i d_i| <= eps` rather
    /// than on the iteration count.
    pub converged: bool,
    /// The largest `|alpha_i d_i|` of the last iterate.
    pub max_step: Scalar,
    /// Residual `|closest_point(x_i) - x_i|` over `B` after the last
    /// iterate: the largest, and the 99th percentile.
    pub max_residual: Scalar,
    pub p99_residual: Scalar,
    /// Points whose alpha was halved at least once by (92.31).
    pub n_scaled_back: usize,
    /// Points that ended PINNED - alpha driven to zero, or fixed by (92.30).
    pub n_pinned: usize,
    /// Iterates abandoned whole by (92.31).
    pub n_abandoned: usize,
    /// Feature edges (92.34) the surface carried, and corners (92.35).
    pub n_feature_edges: usize,
    pub n_feature_corners: usize,
    /// Points whose LAST iterate took (92.38)'s edge branch, and its corner
    /// branch. Counted from the last iterate that ran, not cumulatively.
    pub n_snapped_to_edge: usize,
    pub n_snapped_to_corner: usize,
}

impl SnapReport {
    /// A few lines for a run log, in [`QualityReport::summary`]'s style.
    pub fn summary(&self) -> String {
        let mut s = String::new();
        s.push_str(&format!(
            "snap: {} boundary points, {} iteration(s), {}\n",
            self.n_boundary_points,
            self.iterations,
            if self.converged {
                "converged"
            } else {
                "out of iterations"
            },
        ));
        s.push_str(&format!(
            "snap: max step {:.3e}, residual max {:.3e}, p99 {:.3e}\n",
            self.max_step, self.max_residual, self.p99_residual
        ));
        s.push_str(&format!(
            "snap: {} point(s) scaled back, {} pinned, {} iterate(s) abandoned\n",
            self.n_scaled_back, self.n_pinned, self.n_abandoned
        ));
        s.push_str(&format!(
            "snap: {} feature edge(s), {} corner(s); last iterate {} point(s) \
             to an edge, {} to a corner\n",
            self.n_feature_edges,
            self.n_feature_corners,
            self.n_snapped_to_edge,
            self.n_snapped_to_corner
        ));
        s
    }
}

/// Stage 4's output: the moved mesh and the gate it passed.
#[derive(Debug, Clone)]
pub struct Snapped {
    pub mesh: PolyMeshRaw,
    pub report: SnapReport,
    pub quality: QualityReport,
}

// ==========================================================================
//  Stage 4 itself
// ==========================================================================

/// SPEC-LIT §92.2 stage 4 / §92.11: displace the boundary points of `mesh`
/// onto `surf`, smooth the displacement, and undo whatever breaks §92.3's
/// gate. `base_size` is `domain.base_size`, the length `snap.tolerance`
/// scales (92.28). `feature_angle_deg` is §92.12's dihedral, the angle
/// (92.34) classifies the feature edges at and (92.38) snaps onto them by.
/// Only `mesh.points` changes.
pub fn snap(
    mesh: &PolyMeshRaw,
    surf: &Surface,
    base_size: Scalar,
    feature_angle_deg: Scalar,
    spec: &SnapSpec,
    t: &QualityThresholds,
) -> Result<Snapped> {
    // Refusals before anything.
    if !(base_size > 0.0) || !base_size.is_finite() {
        return Err(Error::Mesh(format!(
            "snap: domain.base_size must be positive and finite, got \
             {base_size} - snap.tolerance and the closest-point index both \
             scale by it"
        )));
    }
    let n_faces = mesh.faces.len();
    let n_internal = mesh.neighbour.len().min(n_faces);
    let n_cells = mesh
        .owner
        .iter()
        .chain(mesh.neighbour.iter())
        .copied()
        .max()
        .map_or(0, |m| m as usize + 1);
    if n_cells == 0 {
        return Err(Error::Mesh(
            "snap: the mesh has no cells - there is no boundary to snap"
                .to_string(),
        ));
    }
    if !(0.0..=1.0).contains(&spec.smoothing) {
        return Err(Error::Mesh(format!(
            "snap: snap.smoothing is {} - it weights (92.29)'s neighbour \
             mean and must lie in [0, 1]",
            spec.smoothing
        )));
    }
    if spec.max_area_ratio < 1.0 {
        return Err(Error::Mesh(format!(
            "snap: snap.max_area_ratio is {} - a castellated wall \
             over-reports a smooth surface's area by up to sqrt(3), so the \
             limit cannot be under 1",
            spec.max_area_ratio
        )));
    }
    // Which faces are which. A boundary face's patch comes from
    // `mesh.patches` by start/size; a patch is a WALL patch when its name is
    // one of the surface's own patch names, and a DOMAIN patch otherwise.
    // Internal faces are neither.
    let wall_name: HashSet<&str> =
        surf.patch_names.iter().map(|s| s.as_str()).collect();
    let mut patch_of_face = vec![usize::MAX; n_faces];
    for (p, patch) in mesh.patches.iter().enumerate() {
        for j in 0..patch.size {
            let f = n_internal + patch.start + j;
            if f < n_faces {
                patch_of_face[f] = p;
            }
        }
    }
    let mut wall_face = vec![false; n_faces];
    let mut domain_face = vec![false; n_faces];
    for f in n_internal..n_faces {
        let p = patch_of_face[f];
        if p == usize::MAX {
            continue;
        }
        if wall_name.contains(mesh.patches[p].name.as_str()) {
            wall_face[f] = true;
        } else {
            domain_face[f] = true;
        }
    }
    // (92.32), before any point moves: a wall patch carrying more than
    // max_area_ratio times its own surface area is geometry the cells never
    // resolved, and snapping it would collapse the cell that reached it.
    // The test is one-sided on purpose: a_mesh <= a_surf is a patch running
    // partly outside the domain, and is legal.
    for patch in &mesh.patches {
        if !wall_name.contains(patch.name.as_str()) {
            continue;
        }
        let mut a_mesh = 0.0;
        for j in 0..patch.size {
            let f = n_internal + patch.start + j;
            if f < n_faces {
                a_mesh += face_area_vector(&mesh.points, &mesh.faces[f]).mag();
            }
        }
        let a_surf = surf
            .patch_names
            .iter()
            .position(|n| *n == patch.name)
            .map(|k| surf.patch_area[k])
            .unwrap_or(0.0);
        let unresolved = a_surf <= 0.0 && a_mesh > 0.0
            || a_surf > 0.0 && a_mesh > spec.max_area_ratio * a_surf;
        if unresolved {
            let ratio = if a_surf > 0.0 {
                a_mesh / a_surf
            } else {
                Scalar::INFINITY
            };
            return Err(Error::Mesh(format!(
                "snap: patch '{}' carries {} m^2 of wall where its geometry \
                 has {} m^2 - a ratio of {}, over the max_area_ratio of {}; \
                 the cells that reached this patch never resolved it, so \
                 snapping would collapse them (SPEC-LIT §92.11, 92.32)",
                patch.name,
                sig3(a_mesh),
                sig3(a_surf),
                sig3(ratio),
                sig3(spec.max_area_ratio),
            )));
        }
    }
    // The arrival check. G3 and G7 are topological: no motion of the points
    // can mend either, so a mesh that fails one on arrival is refused at
    // once rather than halved at.
    let arrival = quality::measure_capped(mesh, t, quality::GATE_CELL_CAP)?;
    if arrival
        .failures
        .iter()
        .any(|f| matches!(f.gate, Gate::Regions | Gate::Addressing))
    {
        return Err(Error::Mesh(arrival.refusal_text()));
    }
    // The four sets of (92.27), built once. An EDGE is a consecutive pair
    // in a face's point list, wrapping from last to first.
    let n_points = mesh.points.len();
    let mut all_nbrs: Vec<Vec<u32>> = vec![Vec::new(); n_points];
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
    let mut wall_nbrs: Vec<Vec<u32>> = vec![Vec::new(); n_points];
    let mut is_b = vec![false; n_points];
    for f in n_internal..n_faces {
        if !wall_face[f] {
            continue;
        }
        let face = &mesh.faces[f];
        for k in 0..face.len() {
            let a = face[k] as usize;
            let b = face[(k + 1) % face.len()] as usize;
            wall_nbrs[a].push(b as u32);
            wall_nbrs[b].push(a as u32);
            is_b[a] = true;
        }
    }
    for list in wall_nbrs.iter_mut() {
        list.sort_unstable();
        list.dedup();
    }
    let mut cell_points: Vec<Vec<u32>> = vec![Vec::new(); n_cells];
    for (f, face) in mesh.faces.iter().enumerate() {
        let owner = mesh.owner[f] as usize;
        let nbr = if f < n_internal {
            Some(mesh.neighbour[f] as usize)
        } else {
            None
        };
        for &p in face {
            cell_points[owner].push(p as u32);
            if let Some(c) = nbr {
                cell_points[c].push(p as u32);
            }
        }
    }
    for list in cell_points.iter_mut() {
        list.sort_unstable();
        list.dedup();
    }
    // (92.30), taken once: every DOMAIN face locks the axis its unit normal
    // points along, or pins its points outright when the normal is not
    // axis-aligned to 1e-6. A point on a box edge loses two components and
    // one on a box corner all three, by applying this once per face.
    let mut lock = vec![[false; 3]; n_points];
    let mut pinned = vec![false; n_points];
    for f in n_internal..n_faces {
        if !domain_face[f] {
            continue;
        }
        let n = face_area_vector(&mesh.points, &mesh.faces[f]).normalised();
        let axis = [0, 1, 2].map(|a| n.component(a).abs() >= 1.0 - 1e-6);
        for &p in &mesh.faces[f] {
            if axis[0] || axis[1] || axis[2] {
                let l = &mut lock[p as usize];
                l[0] |= axis[0];
                l[1] |= axis[1];
                l[2] |= axis[2];
            } else {
                pinned[p as usize] = true;
            }
        }
    }
    // The hanging nodes: points that are the midpoint of another face's
    // edge, each mapped to its parent edge, longest first so a hanging node
    // of a hanging node is re-seated after its parents.
    let hanging = find_hanging(&mesh.points, &mesh.faces);
    // The index, built once, outside the loop.
    let idx = TriIndex::new(surf, base_size)?;
    // (92.38)'s attraction, prepared once: the surface's sharp edges
    // (92.34) chained into polylines and indexed for (92.36) queries. A
    // `feature_tolerance` of zero turns the stage off entirely - no
    // extraction, no index - and a surface carrying no feature edge is the
    // identity either way. `tau` measures the branch tests in `base_size`s.
    let tau = spec.feature_tolerance * base_size;
    let fset = if spec.feature_tolerance > 0.0 {
        Some(features::extract(surf, feature_angle_deg)?)
    } else {
        None
    };
    let fidx = match &fset {
        Some(fs) if !fs.is_empty() => Some(FeatureIndex::new(fs, base_size)?),
        _ => None,
    };
    // The corners by their slot in the feature set, so (92.39)'s claim can
    // be a Vec over the corners while `closest_corner` answers in point ids.
    let corner_slot: HashMap<u32, usize> = fset.as_ref().map_or(HashMap::new(), |fs| {
        fs.corners.iter().enumerate().map(|(s, &c)| (c, s)).collect()
    });
    let n_corners = fset.as_ref().map_or(0, |fs| fs.corners.len());
    let mut claim: Vec<Option<u32>> = vec![None; n_corners];
    let mut corner_of_point: Vec<i32> = vec![-1; n_points];
    let eps = spec.tolerance * base_size;
    let w = spec.smoothing;
    let mut pts = mesh.points.clone();
    let mut work = mesh.clone();
    let mut scaled_back = vec![false; n_points];
    let mut report = SnapReport {
        n_boundary_points: is_b.iter().filter(|&&b| b).count(),
        n_feature_edges: fset.as_ref().map_or(0, |fs| fs.edges.len()),
        n_feature_corners: n_corners,
        ..SnapReport::default()
    };
    for k in 0..spec.iterations {
        // (92.39): the corner claim, recomputed from the positions THIS
        // iterate starts from - a total function of them, so no order of
        // visiting decides who owns a corner, and a point that drifts past
        // another does not keep a corner it is no longer nearest to.
        if let Some(fi) = &fidx {
            claim_corners(
                fi,
                &corner_slot,
                &pts,
                &is_b,
                &pinned,
                &mut claim,
                &mut corner_of_point,
            );
        }
        // (92.28): the pull toward the closest point, dead-banded by eps -
        // the band measured against the FINAL target of (92.38), so a point
        // already on the feature it belongs to is not pulled again.
        let mut delta = vec![Vec3::ZERO; n_points];
        let mut took_edge = 0usize;
        let mut took_corner = 0usize;
        for i in 0..n_points {
            if !is_b[i] || pinned[i] {
                continue;
            }
            let (q_surf, _, d) = idx.closest_point(pts[i]);
            // (92.38): corner first, but only the corner this point claims;
            // else the feature edge, when one is within tau of the surface
            // point - the measurement is from `q`, never from `x_i`, which
            // sits up to half a diagonal off the geometry.
            let mut q = q_surf;
            let mut branch = 0u8;
            if let Some(fi) = &fidx {
                if let Some((cq, dc, c)) = fi.closest_corner(q) {
                    let claims = corner_slot
                        .get(&c)
                        .map_or(false, |&s| corner_of_point[i] == s as i32);
                    if dc <= tau && claims {
                        q = cq;
                        branch = 2;
                    }
                }
                if branch == 0 {
                    if let Some((eq, de, _)) = fi.closest_edge_point(q) {
                        if de <= tau {
                            q = eq;
                            branch = 1;
                        }
                    }
                }
            }
            match branch {
                2 => took_corner += 1,
                1 => took_edge += 1,
                _ => {}
            }
            if branch == 0 {
                if d > eps {
                    delta[i] = q - pts[i];
                }
            } else {
                let off = q - pts[i];
                if off.mag() > eps {
                    delta[i] = off;
                }
            }
        }
        report.n_snapped_to_edge = took_edge;
        report.n_snapped_to_corner = took_corner;
        // (92.29): Jacobi sweeps over the DISPLACEMENT - on the wall's own
        // surface graph inside B, on the whole point graph outside it - into
        // a fresh vector each pass, never in place. A point with no
        // neighbours keeps the (1 - w) delta term alone off B.
        let mut d_field = delta.clone();
        for _ in 0..spec.smoothing_passes {
            let mut next = vec![Vec3::ZERO; n_points];
            for i in 0..n_points {
                let nbrs: &Vec<u32> = if is_b[i] { &wall_nbrs[i] } else { &all_nbrs[i] };
                let mut mean = Vec3::ZERO;
                if !nbrs.is_empty() {
                    for &j in nbrs {
                        mean = mean + d_field[j as usize];
                    }
                    mean = mean / (nbrs.len() as Scalar);
                }
                next[i] = if is_b[i] {
                    (1.0 - w) * delta[i] + w * mean
                } else {
                    w * mean
                };
            }
            d_field = next;
        }
        // (92.30): the box constrains the direction; a pin kills the step.
        for i in 0..n_points {
            if pinned[i] {
                d_field[i] = Vec3::ZERO;
            } else {
                let d = d_field[i];
                d_field[i] = Vec3::new(
                    if lock[i][0] { 0.0 } else { d.x },
                    if lock[i][1] { 0.0 } else { d.y },
                    if lock[i][2] { 0.0 } else { d.z },
                );
            }
        }
        // (92.31): the guarded step. Halve what breaks a cell, pin what
        // halving cannot mend, and abandon the iterate whole - pts is then
        // unchanged, and it passed the gate on arrival.
        let mut alpha = vec![1.0; n_points];
        for i in 0..n_points {
            if pinned[i] {
                alpha[i] = 0.0;
            }
        }
        let mut halvings = 0usize;
        let mut accepted = false;
        loop {
            let mut pts_try: Vec<Vec3> = (0..n_points)
                .map(|i| pts[i] + d_field[i] * alpha[i])
                .collect();
            // (92.33): the hanging nodes ride their parents - exact
            // midpoints, longest parent edge first, so §92.3's closure
            // cancellation survives whatever alpha the halving has left
            // behind. A pinned hanging node is re-seated too: its seat is
            // the mesh's, not its own motion.
            for &(h, ab) in &hanging {
                pts_try[h as usize] =
                    (pts_try[ab[0] as usize] + pts_try[ab[1] as usize]) * 0.5;
            }
            work.points = pts_try.clone();
            let rep = quality::measure_capped(&work, t, usize::MAX)?;
            // The failing CELLS: the subject itself for a cell-named gate,
            // both cells of the face for G4. G3 and G7 are skipped - step 3
            // already refused them, and they cannot appear here.
            let mut fail: Vec<u32> = Vec::new();
            for failure in &rep.failures {
                match failure.gate {
                    Gate::Regions | Gate::Addressing => continue,
                    Gate::NonOrth => {
                        for s in &failure.subjects {
                            // The subject is the FACE; the cells it blames
                            // are its owner and, internally, its neighbour.
                            // The face id itself is NOT a cell id.
                            let f = s.id;
                            fail.push(mesh.owner[f] as u32);
                            if f < n_internal {
                                fail.push(mesh.neighbour[f] as u32);
                            }
                        }
                    }
                    _ => {
                        for s in &failure.subjects {
                            fail.push(s.id as u32);
                        }
                    }
                }
            }
            fail.sort_unstable();
            fail.dedup();
            if fail.is_empty() {
                pts = pts_try;
                accepted = true;
                break;
            }
            if halvings < spec.undo_limit {
                for &c in &fail {
                    for &i in &cell_points[c as usize] {
                        if !pinned[i as usize] {
                            alpha[i as usize] *= 0.5;
                            scaled_back[i as usize] = true;
                        }
                    }
                }
                halvings += 1;
            } else {
                for &c in &fail {
                    for &i in &cell_points[c as usize] {
                        pinned[i as usize] = true;
                    }
                }
                report.n_abandoned += 1;
                break;
            }
        }
        let max_step = if accepted {
            let mut m: Scalar = 0.0;
            for i in 0..n_points {
                m = m.max((d_field[i] * alpha[i]).mag());
            }
            m
        } else {
            0.0
        };
        report.iterations = k + 1;
        report.max_step = max_step;
        // An abandoned iterate moved nothing, but a zero step is only
        // convergence when the step was ACCEPTED: the abandonment pinned
        // points, and the points it did not pin may still have somewhere to
        // go next pass.
        if accepted && max_step <= eps {
            report.converged = true;
            break;
        }
    }
    // The residuals over B after the last iterate.
    let mut residuals: Vec<Scalar> = Vec::new();
    for i in 0..n_points {
        if is_b[i] {
            let (_, _, d) = idx.closest_point(pts[i]);
            residuals.push(d);
        }
    }
    residuals.sort_by(Scalar::total_cmp);
    if let Some(last) = residuals.last() {
        report.max_residual = *last;
        report.p99_residual =
            residuals[(((residuals.len() - 1) as f64) * 0.99).round() as usize];
    }
    report.n_scaled_back = scaled_back.iter().filter(|&&s| s).count();
    report.n_pinned = pinned.iter().filter(|&&p| p).count();
    // The gate: a mesh that still fails leaves as §92.3's own refusal text.
    let mut out = mesh.clone();
    out.points = pts;
    let quality = quality::check(&out, t)?;
    Ok(Snapped {
        mesh: out,
        report,
        quality,
    })
}

// ==========================================================================
//  Helpers
// ==========================================================================

/// (92.39): the corner claim, from the CURRENT positions - for every corner
/// `k`, the boundary point `i` of `B`, not pinned, minimising
/// `|x_i - corner_k|`, ties to the lower `i`, stored as `claim[k]` with
/// `corner_of_point` its reverse. Overwrites both arrays whole.
///
/// A corner admits exactly one point: two points sent to one corner
/// position are a zero-area face, a zero-volume cell, and a mesh the gate
/// refuses. The candidate set is (92.39)'s `cand(k)` - the points whose
/// NEAREST corner is `k`, read off one `closest_corner` query each, so the
/// whole assignment costs |B| queries and not |B| times the corner count.
/// That set also makes the claim a matching by construction: a point is a
/// candidate for exactly one corner, so no point can be sent to two, and no
/// second pass is needed to resolve contention. What it costs is stated in
/// §92.12: a corner whose own nearest point is nearer to a different corner
/// goes unclaimed for that iterate, and its points take the edge branch.
///
/// The candidate rule, and the removal of a second-choice fallback that
/// could never fire (a point appears in exactly one corner's candidate
/// list), are the supervising session's, not the coding agent's.
fn claim_corners(
    fi: &FeatureIndex,
    corner_slot: &HashMap<u32, usize>,
    pts: &[Vec3],
    is_b: &[bool],
    pinned: &[bool],
    claim: &mut [Option<u32>],
    corner_of_point: &mut [i32],
) {
    for c in claim.iter_mut() {
        *c = None;
    }
    for o in corner_of_point.iter_mut() {
        *o = -1;
    }
    let n_corners = claim.len();
    if n_corners == 0 {
        return;
    }
    // Per corner the nearest of its own candidates, ordered by (distance,
    // point index) so a tie goes to the lower point.
    let mut best = vec![(Scalar::INFINITY, u32::MAX); n_corners];
    for (i, b) in is_b.iter().enumerate() {
        if !b || pinned[i] {
            continue;
        }
        if let Some((_, d, c)) = fi.closest_corner(pts[i]) {
            let s = match corner_slot.get(&c) {
                Some(&s) => s,
                None => continue,
            };
            let cand = (d, i as u32);
            if cand < best[s] {
                best[s] = cand;
            }
        }
    }
    for (s, &(d, i)) in best.iter().enumerate() {
        if d.is_finite() {
            claim[s] = Some(i);
            corner_of_point[i as usize] = s as i32;
        }
    }
}

/// The polygon's area vector, fanned about the mean of its own points -
/// the construction `mesh::geometry` runs, on the raw arrays.
pub(crate) fn face_area_vector(points: &[Vec3], face: &[Label]) -> Vec3 {
    let mut c = Vec3::ZERO;
    for &p in face {
        c = c + points[p as usize];
    }
    c = c / (face.len() as Scalar);
    let mut sf = Vec3::ZERO;
    for k in 0..face.len() {
        let a = points[face[k] as usize] - c;
        let b = points[face[(k + 1) % face.len()] as usize] - c;
        sf = sf + a.cross(b);
    }
    sf * 0.5
}

/// The mesh's hanging nodes: each point that is the midpoint of some face's
/// edge, mapped to that parent edge. Ordered by parent-edge length, longest
/// first, so a hanging node of a hanging node is re-seated after its own
/// parents.
///
/// (92.33)'s parents. The lookup is a spatial hash and NOT a scan: (92.20)'s
/// lattice midpoint
/// is the average of two node coordinates only up to rounding - `coord`
/// interpolates, and `(a + b) / 2` and `coord((qa + qb) / 2)` are two
/// different roundings of one number - so a bit-exact map misses most of the
/// hanging nodes and a linear fallback would cost one pass over every point
/// per edge. Bucketed, the query touches the 27 buckets around the midpoint
/// and nothing else.
///
/// Written by the supervising session, not by the coding agent: the rule and
/// the ordering are the coding agent's, the bucketing replaced its O(edges x
/// points) fallback scan.
pub(crate) fn find_hanging(points: &[Vec3], faces: &[Vec<Label>]) -> Vec<(u32, [u32; 2])> {
    if points.is_empty() {
        return Vec::new();
    }
    let mut lo = points[0];
    let mut hi = points[0];
    for p in points {
        lo = lo.cmpt_min(*p);
        hi = hi.cmpt_max(*p);
    }
    // One tolerance for the whole mesh: no edge is longer than the diagonal,
    // so a per-edge tolerance can only be smaller than this one. The bucket
    // is a thousand tolerances wide, which is what makes the 27-bucket probe
    // exhaustive for everything within tolerance of the query.
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
    let mut seen: HashSet<(u32, u32)> = HashSet::new();
    let mut out: HashMap<u32, ([u32; 2], Scalar)> = HashMap::new();
    for face in faces {
        for k in 0..face.len() {
            let a = face[k] as u32;
            let b = face[(k + 1) % face.len()] as u32;
            let edge = if a < b { (a, b) } else { (b, a) };
            if !seen.insert(edge) {
                continue;
            }
            let (pa, pb) = (points[a as usize], points[b as usize]);
            let m = (pa + pb) * 0.5;
            let c = key(m);
            let mut best: Option<(u32, Scalar)> = None;
            for dz in -1..=1 {
                for dy in -1..=1 {
                    for dx in -1..=1 {
                        let Some(bucket) =
                            grid.get(&[c[0] + dx, c[1] + dy, c[2] + dz])
                        else {
                            continue;
                        };
                        for &i in bucket {
                            if i == a || i == b {
                                continue;
                            }
                            let d = (points[i as usize] - m).mag();
                            if d <= tol && best.is_none_or(|(_, bd)| d < bd) {
                                best = Some((i, d));
                            }
                        }
                    }
                }
            }
            if let Some((hnode, _)) = best {
                let len2 = (pb - pa).mag_sqr();
                match out.get(&hnode) {
                    Some((_, old)) if *old >= len2 => {}
                    _ => {
                        out.insert(hnode, ([a, b], len2));
                    }
                }
            }
        }
    }
    let mut v: Vec<(u32, [u32; 2], Scalar)> =
        out.into_iter().map(|(h, (ab, len2))| (h, ab, len2)).collect();
    v.sort_by(|x, y| y.2.total_cmp(&x.2).then(x.0.cmp(&y.0)));
    v.into_iter().map(|(h, ab, _)| (h, ab)).collect()
}

/// A number to three significant figures, as the (92.32) refusal prints it.
fn sig3(v: Scalar) -> String {
    if !v.is_finite() || v == 0.0 {
        return format!("{v}");
    }
    let e = v.abs().log10().floor() as i32;
    let dec = (2 - e).max(0) as usize;
    let mut s = format!("{:.*}", dec, v);
    if s.contains('.') {
        while s.ends_with('0') {
            s.pop();
        }
        if s.ends_with('.') {
            s.pop();
        }
    }
    s
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
    use crate::automesher::{
        CastellationSpec, DistanceBand, DomainSpec, RefinementBand, RefinementSpec,
    };

    /// The background over `extent` at `base`, and the tree `max_level` deep
    /// everywhere - the common front half of every test here.
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

    /// The volume a closed, outward-wound soup encloses: |sum a . (b x c)| / 6.
    fn soup_volume(soup: &[(u32, [Vec3; 3])]) -> f64 {
        let mut v = 0.0;
        for (_, t) in soup {
            v += t[0].dot(t[1].cross(t[2]));
        }
        v.abs() / 6.0
    }

    #[test]
    fn a_cube_on_the_cell_planes_is_snapped_bit_for_bit() {
        let (tree, bg) = setup([0.0, 4.0, 0.0, 4.0, 0.0, 4.0], 1.0, 0);
        let surf = Surface::from_soup(
            box_soup([1.0; 3], [3.0; 3]),
            vec!["cube".to_string()],
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
        let snapped =
            snap(&cast.mesh, &surf, 1.0, 30.0, &SnapSpec::default(), &thresholds())
                .expect("snap");
        assert_eq!(snapped.mesh.points.len(), cast.mesh.points.len());
        for (a, b) in snapped.mesh.points.iter().zip(cast.mesh.points.iter()) {
            assert_eq!(a.x.to_bits(), b.x.to_bits());
            assert_eq!(a.y.to_bits(), b.y.to_bits());
            assert_eq!(a.z.to_bits(), b.z.to_bits());
        }
        assert_eq!(snapped.mesh.faces, cast.mesh.faces);
        assert_eq!(snapped.mesh.owner, cast.mesh.owner);
        assert_eq!(snapped.mesh.neighbour, cast.mesh.neighbour);
        assert_eq!(snapped.mesh.patches.len(), cast.mesh.patches.len());
        for (a, b) in snapped.mesh.patches.iter().zip(cast.mesh.patches.iter()) {
            assert_eq!(a.name, b.name);
            assert_eq!(a.start, b.start);
            assert_eq!(a.size, b.size);
        }
        assert!(
            snapped.report.max_residual <= 1e-12,
            "residual {} - every wall point is already on the geometry",
            snapped.report.max_residual
        );
        assert_eq!(snapped.report.n_pinned, 0);
        assert_eq!(snapped.report.n_scaled_back, 0);
        assert_eq!(snapped.report.n_abandoned, 0);
        assert!(snapped.report.converged);
        assert_eq!(snapped.report.iterations, 1);
    }

    #[test]
    fn a_sphere_is_snapped_onto_the_sphere() {
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
            snap(&cast.mesh, &surf, 1.0, 30.0, &SnapSpec::default(), &thresholds())
                .expect("snap");
        eprintln!("{}", snapped.report.summary());
        assert!(
            snapped.report.p99_residual < 0.02 * 3.0,
            "p99 residual {} not under {}",
            snapped.report.p99_residual,
            0.02 * 3.0
        );
        // The topology is unchanged: only points moved.
        assert_eq!(snapped.mesh.faces, cast.mesh.faces);
        assert_eq!(snapped.mesh.owner, cast.mesh.owner);
        assert_eq!(snapped.mesh.neighbour, cast.mesh.neighbour);
        let mut host =
            crate::io::polymesh::build_host_mesh(&snapped.mesh).expect("host mesh");
        host.compute_geometry(&snapped.mesh.points, &snapped.mesh.faces)
            .expect("geometry");
        let total = host.check().total_volume;
        let want = 512.0 - soup_volume(&sphere_soup(3.0, [4.0; 3]));
        eprintln!(
            "snap: kept volume {total:.6} vs the STL's own {want:.6} ({:.3} % off)",
            100.0 * (total - want).abs() / want
        );
        let rel = (total - want).abs() / want;
        assert!(
            rel < 0.01,
            "kept volume {total} vs the STL's own enclosed {want} ({:.2} % off)",
            100.0 * rel
        );
        assert!(snapped.quality.passed());
    }

    #[test]
    fn a_sphere_finer_than_a_cell_is_refused_by_name() {
        let (tree, bg) = setup([0.0, 4.0, 0.0, 4.0, 0.0, 4.0], 1.0, 0);
        let surf = Surface::from_soup(
            sphere_soup(0.05, [1.5, 1.5, 1.5]),
            vec!["pinhead".to_string()],
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
        let err = snap(&cast.mesh, &surf, 1.0, 30.0, &SnapSpec::default(), &thresholds())
            .expect_err("a sphere finer than a cell is refused");
        let msg = format!("{err}");
        assert!(
            msg.contains("pinhead"),
            "the refusal must name the patch: {msg}"
        );
        assert!(
            msg.contains("max_area_ratio"),
            "the refusal must name the limit: {msg}"
        );
    }

    /// (92.31)'s undo, on a gate no displacement can satisfy. A UNIFORM
    /// tree's castellated mesh is exactly orthogonal - every internal face
    /// separates two cells whose centres differ along one axis - so a limit
    /// of a thousandth of a degree passes on arrival and fails on the first
    /// move. The step is halved `undo_limit` times, the blamed points are
    /// pinned, the iterate is abandoned whole, and what comes back is the
    /// mesh that went in, bit for bit, still through §92.3's gate. This is
    /// the path the other four tests never reach: they all snap cleanly.
    ///
    /// Written by the supervising session, not by the coding agent.
    #[test]
    fn a_gate_no_step_can_satisfy_leaves_the_mesh_where_it_started() {
        let (tree, bg) = setup([0.0, 8.0, 0.0, 8.0, 0.0, 8.0], 1.0, 0);
        let surf = Surface::from_soup(
            sphere_soup(3.0, [4.0; 3]),
            vec!["sphere".to_string()],
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
        let mut t = thresholds();
        t.max_non_orth_deg = 1e-3;
        t.report_non_orth_deg = 1e-3;
        let snapped = snap(&cast.mesh, &surf, 1.0, 30.0, &SnapSpec::default(), &t)
            .expect("the arrival mesh is orthogonal, so the gate is satisfiable");
        eprintln!("{}", snapped.report.summary());
        assert!(
            snapped.report.n_scaled_back > 0,
            "the guard must have halved something"
        );
        assert!(snapped.report.n_pinned > 0, "and then pinned it");
        assert!(
            snapped.report.n_abandoned > 0,
            "and abandoned the iterate whole"
        );
        for (a, b) in snapped.mesh.points.iter().zip(cast.mesh.points.iter()) {
            assert_eq!(a.x.to_bits(), b.x.to_bits());
            assert_eq!(a.y.to_bits(), b.y.to_bits());
            assert_eq!(a.z.to_bits(), b.z.to_bits());
        }
        assert!(snapped.quality.passed());
    }

    #[test]
    fn a_sphere_cut_by_the_box_keeps_the_box_plane() {
        let (mut tree, bg) = setup([0.0, 8.0, 0.0, 8.0, 0.0, 8.0], 1.0, 1);
        let surf = Surface::from_soup(
            sphere_soup(3.0, [4.0, 4.0, 0.0]),
            vec!["dome".to_string()],
        )
        .expect("surface");
        let spec = RefinementSpec {
            levels: vec![RefinementBand {
                patch: "dome".to_string(),
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
            snap(&cast.mesh, &surf, 1.0, 30.0, &SnapSpec::default(), &thresholds())
                .expect("snap - the one-sided area test lets the dome through");
        eprintln!("{}", snapped.report.summary());
        // (92.30) holds the box plane exactly, on every point the zMin
        // patch's faces carry.
        let n_internal = snapped.mesh.neighbour.len();
        let zmin = snapped
            .mesh
            .patches
            .iter()
            .position(|p| p.name == "zMin")
            .expect("zMin patch");
        let patch = &snapped.mesh.patches[zmin];
        for j in 0..patch.size {
            let f = n_internal + patch.start + j;
            for &p in &snapped.mesh.faces[f] {
                assert_eq!(
                    snapped.mesh.points[p as usize].z.to_bits(),
                    (0.0f64).to_bits(),
                    "point {p} of zMin face {f} left the box plane"
                );
            }
        }
        assert!(snapped.quality.passed());
    }

    /// The unit's common case: the [0,4]^3 domain at base size 1, the tree
    /// at level 1 around the geometry, and a cube spanning [1.3, 2.3]^3 -
    /// offset by 0.3 of a BASE cell, so no cube face lies on a cell plane.
    /// Returns the surface and the castellated mesh; the caller snaps.
    fn cube_case() -> (Surface, PolyMeshRaw) {
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
        (surf, cast.mesh)
    }

    /// The `a_sphere_is_snapped_onto_the_sphere` case, built once: the
    /// refined tree, the surface, the castellated mesh.
    fn sphere_case() -> (Surface, PolyMeshRaw) {
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
        (surf, cast.mesh)
    }

    /// Point-to-SEGMENT distance - the acceptance test measures along an
    /// edge, not to its infinite line.
    fn seg_dist(p: Vec3, a: Vec3, b: Vec3) -> f64 {
        let ab = b - a;
        let t = ((p - a).dot(ab) / ab.mag_sqr()).clamp(0.0, 1.0);
        (p - a - ab * t).mag()
    }

    /// The cube's 12 edges as segments, and its 8 corners, from its corners
    /// `lo` and `hi`.
    fn cube_frame(lo: Vec3, hi: Vec3) -> (Vec<(Vec3, Vec3)>, Vec<Vec3>) {
        let mut segs = Vec::new();
        for d in 0..3 {
            for m0 in [false, true] {
                for m1 in [false, true] {
                    let e1 = (d + 1) % 3;
                    let e2 = (d + 2) % 3;
                    let at = |on: bool, ax: usize| {
                        if on { hi.component(ax) } else { lo.component(ax) }
                    };
                    let mut a = [0.0; 3];
                    a[d] = lo.component(d);
                    a[e1] = at(m0, e1);
                    a[e2] = at(m1, e2);
                    let mut b = a;
                    b[d] = hi.component(d);
                    segs.push((Vec3::new(a[0], a[1], a[2]), Vec3::new(b[0], b[1], b[2])));
                }
            }
        }
        let mut corners = Vec::new();
        for cx in [lo.x, hi.x] {
            for cy in [lo.y, hi.y] {
                for cz in [lo.z, hi.z] {
                    corners.push(Vec3::new(cx, cy, cz));
                }
            }
        }
        (segs, corners)
    }

    #[test]
    fn a_cube_off_the_lattice_gets_its_edges_and_corners() {
        let (surf, cast) = cube_case();
        // The run at SnapSpec::default(). (92.28)'s dead band bounds stage
        // 4's own precision at eps = tolerance * base_size - 1e-3 at the
        // default - and the band measures the FINAL target of (92.38), so a
        // point stops within eps of the edge or corner it was pulled to.
        // The defaults still occupy every entity, and pass the gate with
        // the topology unchanged.
        let defaults =
            snap(&cast, &surf, 1.0, 30.0, &SnapSpec::default(), &thresholds())
                .expect("snap at the defaults");
        eprintln!("{}", defaults.report.summary());
        // The cube's triangulation carries its 12 edges and 8 corners
        // exactly, by (92.34) and (92.35).
        assert_eq!(defaults.report.n_feature_edges, 12);
        assert_eq!(defaults.report.n_feature_corners, 8);
        assert_eq!(defaults.report.n_snapped_to_corner, 8);
        assert!(defaults.report.n_snapped_to_edge > 0);
        let (segs, corners) =
            cube_frame(Vec3::new(1.3, 1.3, 1.3), Vec3::new(2.3, 2.3, 2.3));
        assert_eq!(segs.len(), 12);
        assert_eq!(corners.len(), 8);
        // Occupied to the precision the dead band defines: every entity
        // carries a point within 2 eps.
        for (a, b) in &segs {
            let best = defaults
                .mesh
                .points
                .iter()
                .map(|p| seg_dist(*p, *a, *b))
                .fold(f64::INFINITY, f64::min);
            assert!(
                best <= 2e-3,
                "at the defaults no point within 2 eps of the edge ({:?}) - ({:?}): {:e}",
                a, b, best
            );
        }
        for c in &corners {
            let best = defaults
                .mesh
                .points
                .iter()
                .map(|p| (*p - *c).mag())
                .fold(f64::INFINITY, f64::min);
            assert!(
                best <= 2e-3,
                "at the defaults no point within 2 eps of the corner {:?}: {:e}",
                c, best
            );
        }
        assert!(defaults.quality.passed());
        assert_eq!(defaults.mesh.faces, cast.faces);
        assert_eq!(defaults.mesh.owner, cast.owner);
        assert_eq!(defaults.mesh.neighbour, cast.neighbour);
        // The micron gate: the same case with (92.28)'s dead band tightened
        // to 1e-9 and the iteration cap raised to 60 - every other field
        // the default - so the pull runs to its fixed point instead of
        // stopping at the band. Then every point the (92.38) branches took
        // sits ON its target: within 1e-6 of each edge SEGMENT, and each of
        // the 8 corners occupied - by one point, which
        // two_points_cannot_take_one_corner asserts directly.
        let spec = SnapSpec {
            tolerance: 1e-9,
            iterations: 60,
            ..SnapSpec::default()
        };
        let snapped = snap(&cast, &surf, 1.0, 30.0, &spec, &thresholds())
            .expect("snap with the band tightened");
        eprintln!("{}", snapped.report.summary());
        for (a, b) in &segs {
            let best = snapped
                .mesh
                .points
                .iter()
                .map(|p| seg_dist(*p, *a, *b))
                .fold(f64::INFINITY, f64::min);
            assert!(
                best <= 1e-6,
                "no point on the edge ({:?}) - ({:?}): closest {:e}",
                a, b, best
            );
        }
        for c in &corners {
            let best = snapped
                .mesh
                .points
                .iter()
                .map(|p| (*p - *c).mag())
                .fold(f64::INFINITY, f64::min);
            assert!(
                best <= 1e-6,
                "no point on the corner {:?}: closest {:e}",
                c, best
            );
        }
        assert!(snapped.quality.passed());
        assert_eq!(snapped.mesh.faces, cast.faces);
        assert_eq!(snapped.mesh.owner, cast.owner);
        assert_eq!(snapped.mesh.neighbour, cast.neighbour);
    }

    #[test]
    fn with_feature_snapping_off_the_corners_are_not_occupied() {
        let (surf, cast) = cube_case();
        let spec = SnapSpec {
            feature_tolerance: 0.0,
            ..SnapSpec::default()
        };
        let a = snap(&cast, &surf, 1.0, 30.0, &spec, &thresholds()).expect("snap a");
        let b = snap(&cast, &surf, 1.0, 30.0, &spec, &thresholds()).expect("snap b");
        eprintln!("{}", a.report.summary());
        assert_eq!(a.report.n_snapped_to_edge, 0);
        assert_eq!(a.report.n_snapped_to_corner, 0);
        // Determinism: what the pre-change code produced is what the stage
        // at zero tolerance still produces, bit for bit.
        assert_eq!(a.mesh.points.len(), b.mesh.points.len());
        for (p, q) in a.mesh.points.iter().zip(b.mesh.points.iter()) {
            assert_eq!(p.x.to_bits(), q.x.to_bits());
            assert_eq!(p.y.to_bits(), q.y.to_bits());
            assert_eq!(p.z.to_bits(), q.z.to_bits());
        }
        // The chamfer (92.38) exists to mend: with the attraction off, the
        // nearest-point map never returns an edge, and no point is TAKEN to
        // a corner. One caveat this geometry really has: the solid block's
        // corner (2.5, 2.5, 2.5) faces the cube's own (2.3, 2.3, 2.3) up
        // the body diagonal from 0.2 of a cell, and to a point outside a
        // convex corner the closest point of the triangulation IS the
        // corner - so (92.28) alone walks it there and the dead band stops
        // it within eps. That is the one point the feature stage would have
        // claimed; every other corner keeps every point at least 0.1 away,
        // and nowhere does a point come within the 1e-6 of occupation the
        // acceptance test holds the stage to.
        let (_, corners) = cube_frame(Vec3::new(1.3, 1.3, 1.3), Vec3::new(2.3, 2.3, 2.3));
        // The one corner (92.28) reaches on its own is named by its
        // coordinates, not by its position in the list.
        let free = Vec3::new(2.3, 2.3, 2.3);
        for c in corners.iter() {
            let best = a
                .mesh
                .points
                .iter()
                .map(|p| (*p - *c).mag())
                .fold(f64::INFINITY, f64::min);
            if (*c - free).mag() < 1e-12 {
                assert!(
                    best > 1e-6 && best <= 2e-3,
                    "the body-diagonal corner {:?}: nearest point {:e} - \
                     expected the dead-banded face pull, nothing more",
                    c, best
                );
            } else {
                assert!(
                    best >= 0.1,
                    "a point sits {:e} from the corner {:?} with the stage off",
                    best, c
                );
            }
        }
    }

    #[test]
    fn a_sphere_is_unchanged_by_the_feature_stage() {
        let (surf, cast) = sphere_case();
        // `sphere_soup` folds no edge further than 22.08 degrees, so at 30
        // the feature set is empty and the stage is the identity - with the
        // tolerance on or off, bit for bit the same mesh.
        let on = snap(&cast, &surf, 1.0, 30.0, &SnapSpec::default(), &thresholds())
            .expect("snap with the stage on");
        let off_spec = SnapSpec {
            feature_tolerance: 0.0,
            ..SnapSpec::default()
        };
        let off = snap(&cast, &surf, 1.0, 30.0, &off_spec, &thresholds())
            .expect("snap with the stage off");
        eprintln!("{}", on.report.summary());
        assert_eq!(on.report.n_feature_edges, 0);
        assert_eq!(on.report.n_feature_corners, 0);
        assert_eq!(on.report.n_snapped_to_edge, 0);
        assert_eq!(on.report.n_snapped_to_corner, 0);
        assert_eq!(on.mesh.points.len(), off.mesh.points.len());
        for (p, q) in on.mesh.points.iter().zip(off.mesh.points.iter()) {
            assert_eq!(p.x.to_bits(), q.x.to_bits());
            assert_eq!(p.y.to_bits(), q.y.to_bits());
            assert_eq!(p.z.to_bits(), q.z.to_bits());
        }
    }

    /// What (92.39) buys: a corner admits one point, so no two points of
    /// the snapped mesh share a position - two at one position are a
    /// zero-area face and a zero-volume cell. O(n^2) on the unit's own
    /// small case.
    #[test]
    fn two_points_cannot_take_one_corner() {
        let (surf, cast) = cube_case();
        let snapped =
            snap(&cast, &surf, 1.0, 30.0, &SnapSpec::default(), &thresholds())
                .expect("snap");
        let pts = &snapped.mesh.points;
        for i in 0..pts.len() {
            for j in (i + 1)..pts.len() {
                let d = (pts[i] - pts[j]).mag();
                assert!(
                    d >= 1e-9,
                    "points {i} and {j} collapsed onto each other: {d:e}"
                );
            }
        }
    }
}
