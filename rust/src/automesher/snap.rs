// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.
// Provenance: see PROVENANCE.md. No GPL-licensed source was consulted.

//! Snapping - SPEC-LIT §92.11, the implementation-level companion to §92.2
//! stage 4. A kept body's interface points join `B` as §92.15 says.
//! Stage 3 leaves a staircase; this stage moves the boundary points
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

/// One surface patch's wall area, as (92.32) measures it: the mesh's wall
/// faces carrying the patch name plus the region-interface faces (92.32)
/// assigns to it, before the first move and on the points the stage returns.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct PatchArea {
    pub name: String,
    /// The patch's own area on the surface, `surf.patch_area[k]`.
    pub stl_area: Scalar,
    pub castellated_area: Scalar,
    pub snapped_area: Scalar,
}

impl PatchArea {
    /// `snapped_area / stl_area`; `None` when the surface area is not positive.
    pub fn ratio(&self) -> Option<Scalar> {
        (self.stl_area > 0.0).then(|| self.snapped_area / self.stl_area)
    }
    /// `castellated_area / stl_area` - the ratio (92.32) refuses on.
    pub fn castellated_ratio(&self) -> Option<Scalar> {
        (self.stl_area > 0.0).then(|| self.castellated_area / self.stl_area)
    }
}

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
    /// The pinned points that lie in `B` of (92.27). `n_pinned` also counts
    /// the non-wall points (92.31) pins with a failing cell when it abandons
    /// an iterate, and the domain points (92.30) pins, so it can exceed
    /// `n_boundary_points`; this count cannot.
    pub n_pinned_boundary: usize,
    /// Iterates abandoned whole by (92.31).
    pub n_abandoned: usize,
    /// Feature edges (92.34) the surface carried, and corners (92.35).
    pub n_feature_edges: usize,
    pub n_feature_corners: usize,
    /// Points whose LAST iterate took (92.38)'s edge branch, and its corner
    /// branch. Counted from the last iterate that ran, not cumulatively.
    pub n_snapped_to_edge: usize,
    pub n_snapped_to_corner: usize,
    /// One row per surface patch, in `surf.patch_names` order.
    pub patch_areas: Vec<PatchArea>,
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
/// Only `mesh.points` changes. Every cell is fluid - see [`snap_regions`]
/// for the region-aware entry point.
pub fn snap(
    mesh: &PolyMeshRaw,
    surf: &Surface,
    base_size: Scalar,
    feature_angle_deg: Scalar,
    spec: &SnapSpec,
    t: &QualityThresholds,
) -> Result<Snapped> {
    snap_regions(mesh, surf, None, base_size, feature_angle_deg, spec, t)
}

/// [`snap`] over a mesh whose cells belong to regions: `region_of_cell`
/// (`None` = every cell fluid; `Some` must have one entry per cell) adds every
/// INTERNAL face whose two cells lie in different regions to the wall-face
/// set - its points join `B` of (92.27), move by (92.28) onto the merged
/// surface, which holds the body's own triangles, and count toward (92.32)'s
/// area under the surface patch (92.26)'s nearest triangle names. The
/// interface stays internal: only `points` change, as in [`snap`].
pub fn snap_regions(
    mesh: &PolyMeshRaw,
    surf: &Surface,
    region_of_cell: Option<&[i32]>,
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
    if let Some(r) = region_of_cell {
        if r.len() != n_cells {
            return Err(Error::Mesh(format!(
                "snap: region_of_cell has {} entries but the mesh has {} cells",
                r.len(),
                n_cells
            )));
        }
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
    // A declared body's interface: an INTERNAL face whose two cells lie in
    // different regions joins the wall-face set, so its points move with
    // (92.28) onto the surface that holds the body's own triangles. The face
    // itself stays internal - only its points move.
    if let Some(r) = region_of_cell {
        for f in 0..n_internal {
            if r[mesh.owner[f] as usize] != r[mesh.neighbour[f] as usize] {
                wall_face[f] = true;
            }
        }
    }
    // (92.32), before any point moves: a patch of the SURFACE carrying more
    // than max_area_ratio times its own area over the mesh's wall faces AND
    // the region interfaces is geometry the cells never resolved, and
    // snapping it would collapse the cell that reached it. The test is
    // one-sided on purpose: a_mesh <= a_surf is a patch running partly
    // outside the domain, and is legal.
    // The index, built once, outside the loop - (92.26)'s nearest triangle
    // names the surface patch an interface face's area lands on.
    let idx = TriIndex::new(surf, base_size)?;
    let mut iface_area = vec![0.0; surf.patch_names.len()];
    // The same faces, kept with the patch (92.32) assigned them, so the
    // report can re-measure them on the returned points.
    let mut iface_patch: Vec<(usize, usize)> = Vec::new();
    for f in 0..n_internal {
        if !wall_face[f] {
            continue;
        }
        let ps = &mesh.faces[f];
        let mut c = [0.0; 3];
        for &p in ps {
            let q = mesh.points[p as usize];
            c[0] += q.x;
            c[1] += q.y;
            c[2] += q.z;
        }
        let n = ps.len() as Scalar;
        let centre = Vec3::new(c[0] / n, c[1] / n, c[2] / n);
        let (t, _) = idx.nearest_triangle(centre);
        iface_area[surf.tri_patch[t] as usize] +=
            face_area_vector(&mesh.points, ps).mag();
        iface_patch.push((f, surf.tri_patch[t] as usize));
    }
    let mut patch_areas: Vec<PatchArea> = Vec::new();
    for (k, name) in surf.patch_names.iter().enumerate() {
        let mut a_mesh = iface_area[k];
        if let Some(patch) = mesh.patches.iter().find(|p| p.name == *name) {
            for j in 0..patch.size {
                let f = n_internal + patch.start + j;
                if f < n_faces {
                    a_mesh += face_area_vector(&mesh.points, &mesh.faces[f]).mag();
                }
            }
        }
        let a_surf = surf.patch_area[k];
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
                name,
                sig3(a_mesh),
                sig3(a_surf),
                sig3(ratio),
                sig3(spec.max_area_ratio),
            )));
        }
        patch_areas.push(PatchArea {
            name: name.clone(),
            stl_area: a_surf,
            castellated_area: a_mesh,
            snapped_area: 0.0,
        });
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
    // Every face: the loop skips the non-wall faces itself, and an interface
    // face IS one since the classification above.
    for f in 0..n_faces {
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
    report.n_pinned_boundary = (0..n_points).filter(|&i| pinned[i] && is_b[i]).count();
    // The gate: a mesh that still fails leaves as §92.3's own refusal text.
    let mut out = mesh.clone();
    out.points = pts;
    // Each patch's area once more, on the points the stage returns: the
    // (92.32) walk unchanged, measured twice.
    patch_areas_on(&out.points, &out, n_internal, &iface_patch, &mut patch_areas);
    report.patch_areas = patch_areas;
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

/// Each surface patch's area on `points`: the (92.32) walk - the region
/// interface faces assigned to the patch, in the order they were recorded,
/// then the boundary faces of the mesh patch carrying its name - with
/// `face_area_vector` measured on `points` rather than the mesh's own.
fn patch_areas_on(
    points: &[Vec3],
    mesh: &PolyMeshRaw,
    n_internal: usize,
    iface_patch: &[(usize, usize)],
    rows: &mut [PatchArea],
) {
    for (k, row) in rows.iter_mut().enumerate() {
        let mut a = iface_patch
            .iter()
            .filter(|&(_, kk)| *kk == k)
            .map(|&(f, _)| face_area_vector(points, &mesh.faces[f]).mag())
            .sum::<Scalar>();
        if let Some(patch) = mesh.patches.iter().find(|p| p.name == row.name) {
            for j in 0..patch.size {
                let f = n_internal + patch.start + j;
                if f < mesh.faces.len() {
                    a += face_area_vector(points, &mesh.faces[f]).mag();
                }
            }
        }
        row.snapped_area = a;
    }
}

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
//  Feature-edge capture - read off a mesh, moves nothing
// ==========================================================================

/// How much of the surface's sharp-edge length the mesh's wall edges lie
/// along - SPEC-LIT §92.12, equation (92.62). It is read off a mesh and
/// moves nothing, so no stage's output depends on it.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct FeatureCapture {
    /// The summed length of the feature edges (92.34), in metres.
    pub sharp_length: Scalar,
    /// Per capture chain, the arclength the union of its covered intervals
    /// spans, summed, in metres; never above `sharp_length`.
    pub captured_length: Scalar,
    /// The distance both ends of a covering wall edge lie within.
    pub tol: Scalar,
}

/// A covering wall edge runs within this many degrees of the feature
/// edge's direction.
pub const CAPTURE_ANGLE_DEG: Scalar = 30.0;

/// The cover test of (92.62): the parameter interval `[t0, t1]` of the
/// segment `a`-`b` that the mesh edge `p`-`q` covers, or `None` when an end
/// lies further than `tol` from the segment, the edge runs more than
/// `CAPTURE_ANGLE_DEG` off the segment's direction, or either is of zero
/// length.
/// Kept as the one-segment reference the chain cover is tested against.
#[cfg(test)]
pub(crate) fn covered_interval(
    p: Vec3,
    q: Vec3,
    a: Vec3,
    b: Vec3,
    tol: Scalar,
) -> Option<(Scalar, Scalar)> {
    let ab = b - a;
    let l2 = ab.mag_sqr();
    let pq = q - p;
    let lpq = pq.mag();
    if !(l2 > 0.0) || !(lpq > 0.0) {
        return None;
    }
    let tp = ((p - a).dot(ab) / l2).clamp(0.0, 1.0);
    let tq = ((q - a).dot(ab) / l2).clamp(0.0, 1.0);
    if (p - (a + ab * tp)).mag() > tol || (q - (a + ab * tq)).mag() > tol {
        return None;
    }
    let cos_min = CAPTURE_ANGLE_DEG.to_radians().cos();
    if pq.dot(ab).abs() < cos_min * lpq * l2.sqrt() {
        return None;
    }
    Some((tp.min(tq), tp.max(tq)))
}

/// The length of the union of the intervals `iv`, which it sorts in place.
pub(crate) fn union_length(iv: &mut [(Scalar, Scalar)]) -> Scalar {
    iv.sort_by(|x, y| x.0.total_cmp(&y.0).then(x.1.total_cmp(&y.1)));
    let mut total: Scalar = 0.0;
    let mut cur: Option<(Scalar, Scalar)> = None;
    for &(s, e) in iv.iter() {
        cur = match cur {
            Some((cs, ce)) if s <= ce => Some((cs, ce.max(e))),
            Some((cs, ce)) => {
                total += ce - cs;
                Some((s, e))
            }
            None => Some((s, e)),
        };
    }
    if let Some((cs, ce)) = cur {
        total += ce - cs;
    }
    total
}

/// The faces the stage calls wall faces: a boundary face whose patch name
/// is one of the surface's, and - with regions - an internal face between
/// two cells of different regions.
fn wall_face_mask(
    mesh: &PolyMeshRaw,
    surf: &Surface,
    region_of_cell: Option<&[i32]>,
) -> Vec<bool> {
    let n_faces = mesh.faces.len();
    let n_internal = mesh.neighbour.len().min(n_faces);
    let names: HashSet<&str> = surf.patch_names.iter().map(|s| s.as_str()).collect();
    let mut wall = vec![false; n_faces];
    for patch in &mesh.patches {
        if !names.contains(patch.name.as_str()) {
            continue;
        }
        for j in 0..patch.size {
            let f = n_internal + patch.start + j;
            if f < n_faces {
                wall[f] = true;
            }
        }
    }
    if let Some(r) = region_of_cell {
        for f in 0..n_internal {
            let (o, n) = (mesh.owner[f] as usize, mesh.neighbour[f] as usize);
            if o < r.len() && n < r.len() && r[o] != r[n] {
                wall[f] = true;
            }
        }
    }
    wall
}

/// One capture chain of (92.62): feature points in order, closed iff the
/// first point is the last, with the arclength at each point.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct CaptureChain {
    /// The chain's points in order along it; `points[k]` and
    /// `points[k + 1]` join the segment of index `k`.
    pub(crate) points: Vec<Vec3>,
    /// The arclength at each point, `arc[0]` 0 and strictly rising.
    pub(crate) arc: Vec<Scalar>,
    /// True when the chain is a closed loop with no cut point: the first
    /// point is the last, its neighbours the second and the
    /// second-to-last point.
    pub(crate) closed: bool,
}

impl CaptureChain {
    /// The chain's length: its last `arc` value, 0 when it has none.
    pub(crate) fn length(&self) -> Scalar {
        self.arc.last().copied().unwrap_or(0.0)
    }
}

/// The open chain of `pts[a..=b]`, its arclength accumulated from `a`.
fn chain_piece(pts: &[Vec3], a: usize, b: usize, closed: bool) -> CaptureChain {
    let points = pts[a..=b].to_vec();
    let mut arc = Vec::with_capacity(points.len());
    arc.push(0.0);
    for k in 0..points.len() - 1 {
        arc.push(arc[k] + (points[k + 1] - points[k]).mag());
    }
    CaptureChain { points, arc, closed }
}

/// The capture chains of (92.62): every polyline of (92.35), cut again at
/// each interior point whose turn exceeds `CAPTURE_ANGLE_DEG` - a closed
/// polyline with no such point stays one closed chain, one with any is
/// re-rooted at its first cut point and cut open there.
pub(crate) fn capture_chains(fs: &features::FeatureSet) -> Vec<CaptureChain> {
    // The turn at `v` between its neighbours `u` and `w`, in degrees; a
    // zero-length neighbour vector is a turn above any angle.
    let turn = |u: Vec3, v: Vec3, w: Vec3| -> Scalar {
        let (a, b) = (u - v, w - v);
        let (la, lb) = (a.mag(), b.mag());
        if !(la > 0.0) || !(lb > 0.0) {
            return CAPTURE_ANGLE_DEG + 1.0;
        }
        180.0 - (a.dot(b) / (la * lb)).clamp(-1.0, 1.0).acos().to_degrees()
    };
    let mut chains: Vec<CaptureChain> = Vec::new();
    for pl in &fs.polylines {
        if pl.len() < 2 {
            continue;
        }
        let pts: Vec<Vec3> = pl.iter().map(|&i| fs.points[i as usize]).collect();
        if pts.first() != pts.last() {
            // Open: a cut point ends one chain and starts the next.
            let mut start = 0usize;
            for i in 1..pts.len() - 1 {
                if turn(pts[i - 1], pts[i], pts[i + 1]) > CAPTURE_ANGLE_DEG {
                    chains.push(chain_piece(&pts, start, i, false));
                    start = i;
                }
            }
            chains.push(chain_piece(&pts, start, pts.len() - 1, false));
            continue;
        }
        insert_closed_chains(&pts, &turn, &mut chains);
    }
    chains
}

/// The closed half of `capture_chains`: `pts[0]` is `pts[m]`, every point
/// interior. No cut is one closed chain; any cut re-roots the ring at its
/// first cut point and cuts it open there.
fn insert_closed_chains(
    pts: &[Vec3],
    turn: &impl Fn(Vec3, Vec3, Vec3) -> Scalar,
    chains: &mut Vec<CaptureChain>,
) {
    let m = pts.len() - 1;
    let ring = &pts[..m];
    let cuts: Vec<usize> = (0..m)
        .filter(|&i| turn(ring[(i + m - 1) % m], ring[i], ring[(i + 1) % m]) > CAPTURE_ANGLE_DEG)
        .collect();
    if cuts.is_empty() {
        chains.push(chain_piece(pts, 0, m, true));
        return;
    }
    // Re-root at the first cut point, then cut as an open polyline.
    let mut open = ring[cuts[0]..].to_vec();
    open.extend_from_slice(&ring[..cuts[0]]);
    open.push(open[0]);
    let mut start = 0usize;
    for i in 1..open.len() - 1 {
        if turn(open[i - 1], open[i], open[i + 1]) > CAPTURE_ANGLE_DEG {
            chains.push(chain_piece(&open, start, i, false));
            start = i;
        }
    }
    chains.push(chain_piece(&open, start, open.len() - 1, false));
}

/// The nearest point of the listed segments of `c` to `p`: its arclength
/// `s`, its distance `d` and the foot, the smaller `s` on an exact tie of
/// `d`. `None` when no listed segment has a nonzero length.
pub(crate) fn project_on_chain(
    p: Vec3,
    c: &CaptureChain,
    segs: &[usize],
) -> Option<(Scalar, Scalar, Vec3)> {
    let mut best: Option<(Scalar, Scalar, Vec3)> = None;
    for &k in segs {
        let (a, b) = (c.points[k], c.points[k + 1]);
        let ab = b - a;
        let l2 = ab.mag_sqr();
        if !(l2 > 0.0) {
            continue;
        }
        let l = l2.sqrt();
        let t = ((p - a).dot(ab) / l2).clamp(0.0, 1.0);
        let f = a + ab * t;
        let s = c.arc[k] + t * l;
        let d = (p - f).mag();
        let better = match best {
            None => true,
            Some((bs, bd, _)) => d < bd || (d == bd && s < bs),
        };
        if better {
            best = Some((s, d, f));
        }
    }
    best
}

/// The cover test of (92.62) along a chain, `fp`/`fq` the projections of
/// `p`/`q`: the covered arc as `[s0, s1]`, `s0 <= s1` always, or `None`
/// when an end lies further than `tol` from the chain, the edge or the
/// chord of the two feet is of zero length, the edge runs more than
/// `CAPTURE_ANGLE_DEG` off that chord, or the arc exceeds the edge by more
/// than `2 tol`. Across the seam of a closed chain the interval runs from
/// `hi` to `lo + L`.
pub(crate) fn chain_cover(
    p: Vec3,
    q: Vec3,
    fp: (Scalar, Scalar, Vec3),
    fq: (Scalar, Scalar, Vec3),
    c: &CaptureChain,
    tol: Scalar,
) -> Option<(Scalar, Scalar)> {
    let (s_p, d_p, f_p) = fp;
    let (s_q, d_q, f_q) = fq;
    if d_p > tol || d_q > tol {
        return None;
    }
    let pq = q - p;
    let lpq = pq.mag();
    if !(lpq > 0.0) {
        return None;
    }
    let k = f_q - f_p;
    let lk = k.mag();
    if !(lk > 0.0) {
        return None;
    }
    if pq.dot(k).abs() < CAPTURE_ANGLE_DEG.to_radians().cos() * lpq * lk {
        return None;
    }
    let (lo, hi) = if s_p <= s_q { (s_p, s_q) } else { (s_q, s_p) };
    let span = hi - lo;
    let l = c.length();
    let across = c.closed && l - span < span;
    if (if across { l - span } else { span }) > lpq + 2.0 * tol {
        return None;
    }
    Some(if across { (hi, lo + l) } else { (lo, hi) })
}

/// (92.62): the sharp length of `surf` at `feature_angle_deg`, and the part
/// of it the wall edges of `mesh` cover within `tol`. Reads; moves nothing.
/// The cover is measured along the capture chains, so a wall edge across
/// the joint of two collinear feature segments counts.
pub fn feature_capture(
    mesh: &PolyMeshRaw,
    surf: &Surface,
    region_of_cell: Option<&[i32]>,
    feature_angle_deg: Scalar,
    tol: Scalar,
) -> Result<FeatureCapture> {
    if !(tol > 0.0) || !tol.is_finite() {
        return Err(Error::Mesh(format!(
            "feature capture: the tolerance must be positive and finite, got {tol}"
        )));
    }
    let fs = features::extract(surf, feature_angle_deg)?;
    let seg = |e: usize| (fs.points[fs.edges[e][0] as usize], fs.points[fs.edges[e][1] as usize]);
    let mut out = FeatureCapture { tol, ..FeatureCapture::default() };
    for e in 0..fs.edges.len() {
        let (a, b) = seg(e);
        out.sharp_length += (b - a).mag();
    }
    if fs.edges.is_empty() {
        return Ok(out);
    }
    let chains = capture_chains(&fs);
    // Buckets of side g >= 2 tol, the chains' segments sampled at most g
    // apart: a point within tol of a segment is within g of a sample, so
    // in one of the 27 buckets around that sample's.
    let g = (2.0 * tol).max(out.sharp_length / 4.0e6);
    let key = |p: Vec3| {
        [(p.x / g).floor() as i64, (p.y / g).floor() as i64, (p.z / g).floor() as i64]
    };
    let mut grid: HashMap<[i64; 3], Vec<(u32, u32)>> = HashMap::new();
    for (ci, c) in chains.iter().enumerate() {
        for k in 0..c.points.len() - 1 {
            let (a, b) = (c.points[k], c.points[k + 1]);
            let n = (((b - a).mag() / g).ceil().max(1.0)) as usize;
            for j in 0..=n {
                let c3 = key(a + (b - a) * ((j as Scalar) / (n as Scalar)));
                let list = grid.entry(c3).or_default();
                if list.last() != Some(&(ci as u32, k as u32)) {
                    list.push((ci as u32, k as u32));
                }
            }
        }
    }
    let wall = wall_face_mask(mesh, surf, region_of_cell);
    let mut edges: Vec<(u32, u32)> = Vec::new();
    for (f, face) in mesh.faces.iter().enumerate() {
        if wall[f] {
            for k in 0..face.len() {
                let (a, b) = (face[k] as u32, face[(k + 1) % face.len()] as u32);
                edges.push((a.min(b), a.max(b)));
            }
        }
    }
    edges.sort_unstable();
    edges.dedup();
    let mut covered: Vec<Vec<(Scalar, Scalar)>> = vec![Vec::new(); chains.len()];
    let mut cand_p: Vec<(u32, u32)> = Vec::new();
    let mut cand_q: Vec<(u32, u32)> = Vec::new();
    let mut segs_p: Vec<usize> = Vec::new();
    let mut segs_q: Vec<usize> = Vec::new();
    let around = |cx: [i64; 3], out: &mut Vec<(u32, u32)>| {
        for dz in -1i64..=1 {
            for dy in -1i64..=1 {
                for dx in -1i64..=1 {
                    if let Some(list) = grid.get(&[cx[0] + dx, cx[1] + dy, cx[2] + dz]) {
                        out.extend_from_slice(list);
                    }
                }
            }
        }
        out.sort_unstable();
        out.dedup();
    };
    for &(i, j) in &edges {
        let (p, q) = (mesh.points[i as usize], mesh.points[j as usize]);
        cand_p.clear();
        around(key(p), &mut cand_p);
        cand_q.clear();
        around(key(q), &mut cand_q);
        // The chains both ends name, each end projected over its own
        // candidate segments only, never the whole chain.
        let mut a = 0usize;
        while a < cand_p.len() {
            let ci = cand_p[a].0;
            let mut b = a;
            while b < cand_p.len() && cand_p[b].0 == ci {
                b += 1;
            }
            let qs = cand_q.partition_point(|&(c, _)| c < ci);
            if qs < cand_q.len() && cand_q[qs].0 == ci {
                let mut qe = qs;
                while qe < cand_q.len() && cand_q[qe].0 == ci {
                    qe += 1;
                }
                segs_p.clear();
                segs_p.extend(cand_p[a..b].iter().map(|&(_, s)| s as usize));
                segs_q.clear();
                segs_q.extend(cand_q[qs..qe].iter().map(|&(_, s)| s as usize));
                let c = &chains[ci as usize];
                if let (Some(fp), Some(fq)) = (
                    project_on_chain(p, c, &segs_p),
                    project_on_chain(q, c, &segs_q),
                ) {
                    if let Some((s0, s1)) = chain_cover(p, q, fp, fq, c, tol) {
                        // An interval ending past L runs across the seam:
                        // store the two pieces either side of it.
                        let l = c.length();
                        if s1 > l {
                            covered[ci as usize].push((s0, l));
                            covered[ci as usize].push((0.0, s1 - l));
                        } else {
                            covered[ci as usize].push((s0, s1));
                        }
                    }
                }
            }
            a = b;
        }
    }
    for c in covered.iter_mut() {
        out.captured_length += union_length(c);
    }
    Ok(out)
}

// ==========================================================================
//  Tests
// ==========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::automesher::castellate::castellate;
    use crate::automesher::castellate::tests::{
        box_soup, mesh_fingerprint, sphere_soup, thresholds,
    };
    use crate::automesher::octree::{
        patch_names, refine_to_surface, Background, Octree,
    };
    use crate::automesher::{
        BodySpec, CastellationSpec, DistanceBand, DomainSpec, RefinementBand,
        RefinementSpec,
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

    /// Requirement 2's pin (M6 Run 1): with no bodies the snapped sphere
    /// mesh is bit for bit what it was before the run's edits. The constant
    /// was pinned before any other edit.
    const PINNED_SPHERE_SNAP_NO_BODY: u64 = 0x0637e67a829f905e;

    #[test]
    fn the_snapped_mesh_without_bodies_is_pinned() {
        let (surf, mesh) = sphere_case();
        let snapped =
            snap(&mesh, &surf, 1.0, 30.0, &SnapSpec::default(), &thresholds())
                .expect("snap");
        let v = mesh_fingerprint(&snapped.mesh);
        eprintln!("fingerprint snapped_sphere_no_body = {v:#018x}");
        assert_eq!(v, PINNED_SPHERE_SNAP_NO_BODY);
        // `snap` IS `snap_regions(.., None, ..)`, and an all-`-1` slice is
        // the same no-body walk: the same mesh bit for bit, the same pin.
        let n_cells = mesh
            .owner
            .iter()
            .chain(mesh.neighbour.iter())
            .copied()
            .max()
            .unwrap() as usize
            + 1;
        let all_fluid = vec![-1i32; n_cells];
        let via_regions = snap_regions(
            &mesh,
            &surf,
            Some(&all_fluid),
            1.0,
            30.0,
            &SnapSpec::default(),
            &thresholds(),
        )
        .expect("snap_regions");
        assert_eq!(mesh_fingerprint(&via_regions.mesh), PINNED_SPHERE_SNAP_NO_BODY);
    }

    /// Requirement 11 (M6 Run 1): a region slice that is not one entry per
    /// cell is refused by name, with both lengths.
    #[test]
    fn a_region_slice_of_the_wrong_length_is_refused() {
        let (_surf, mesh) = sphere_case();
        let n_cells = mesh
            .owner
            .iter()
            .chain(mesh.neighbour.iter())
            .copied()
            .max()
            .unwrap() as usize
            + 1;
        let err = snap_regions(
            &mesh,
            &_surf,
            Some(&vec![-1i32; 3]),
            1.0,
            30.0,
            &SnapSpec::default(),
            &thresholds(),
        )
        .unwrap_err();
        let text = err.to_string();
        assert!(text.contains("region_of_cell"), "{err}");
        assert!(text.contains(&format!("{n_cells}")), "{err}");
        assert!(text.contains('3'), "{err}");
    }

    /// Requirement 8 (M6 Run 1): with the sphere declared a body, the
    /// interface points join `B` of (92.27) and land on the sphere; the
    /// topology and the total volume are the castellated mesh's own, and the
    /// fluid/body split of the volume is (about) the block minus the sphere.
    #[test]
    fn a_kept_sphere_is_snapped_along_its_interface() {
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
        let cast_spec = CastellationSpec {
            bodies: vec![BodySpec {
                name: "sphere".to_string(),
                patches: vec!["sphere".to_string()],
            }],
            ..Default::default()
        };
        let cast = castellate(
            &tree,
            &bg,
            &surf,
            &patch_names(),
            &cast_spec,
            &thresholds(),
        )
        .expect("castellate");
        let snapped = snap_regions(
            &cast.mesh,
            &surf,
            Some(&cast.region_of_cell),
            1.0,
            30.0,
            &SnapSpec::default(),
            &thresholds(),
        )
        .expect("snap_regions");
        eprintln!("{}", snapped.report.summary());
        assert!(snapped.report.n_boundary_points > 0);
        assert!(
            snapped.report.p99_residual < 0.06,
            "p99 residual {} not under 0.06",
            snapped.report.p99_residual
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
        eprintln!("snap: total volume {total:.6} vs the block's 512.0");
        assert!(
            (total - 512.0).abs() / 512.0 < 1e-9,
            "nothing was removed, so the total is 512 to 1e-9: {total}"
        );
        assert!(snapped.quality.passed());
        // The split: the fluid keeps (about) the block minus the sphere, the
        // body holds (about) the sphere.
        let mut fluid = 0.0;
        let mut body = 0.0;
        for (c, r) in cast.region_of_cell.iter().enumerate() {
            if *r < 0 {
                fluid += host.v[c];
            } else {
                body += host.v[c];
            }
        }
        let want = soup_volume(&sphere_soup(3.0, [4.0; 3]));
        eprintln!(
            "snap: fluid {fluid:.6} vs {:.6} ({:.3} % off); body {body:.6} vs \
             {want:.6} ({:.3} % off)",
            512.0 - want,
            100.0 * (fluid - (512.0 - want)).abs() / (512.0 - want),
            100.0 * (body - want).abs() / want
        );
        assert!(
            (fluid - (512.0 - want)).abs() / (512.0 - want) < 0.01,
            "fluid volume {fluid} vs {}",
            512.0 - want
        );
        assert!(
            (body - want).abs() / want < 0.05,
            "body volume {body} vs {want}"
        );
    }

    /// Requirement 9 (M6 Run 1): a kept body far finer than a cell is still
    /// refused by (92.32), whose area now counts the interface faces.
    #[test]
    fn a_kept_body_finer_than_a_cell_is_refused_by_area() {
        let (tree, bg) = setup([0.0, 8.0, 0.0, 8.0, 0.0, 8.0], 1.0, 0);
        let surf = Surface::from_soup(
            sphere_soup(0.2, [4.5; 3]),
            vec!["sphere".to_string()],
        )
        .expect("surface");
        let cast_spec = CastellationSpec {
            bodies: vec![BodySpec {
                name: "sphere".to_string(),
                patches: vec!["sphere".to_string()],
            }],
            ..Default::default()
        };
        let cast = castellate(
            &tree,
            &bg,
            &surf,
            &patch_names(),
            &cast_spec,
            &thresholds(),
        )
        .expect("castellate");
        assert_eq!(
            cast.region_of_cell.iter().filter(|r| **r == 0).count(),
            1,
            "the one leaf whose centre (4.5, 4.5, 4.5) is inside"
        );
        let err = snap_regions(
            &cast.mesh,
            &surf,
            Some(&cast.region_of_cell),
            1.0,
            30.0,
            &SnapSpec::default(),
            &thresholds(),
        )
        .unwrap_err();
        let text = err.to_string();
        eprintln!("refusal: {text}");
        assert!(text.contains("sphere"), "{err}");
        assert!(text.contains("max_area_ratio"), "{err}");
        assert!(text.contains("92.32"), "{err}");
    }

    /// The on-plane cube's report carries its one patch with castellated
    /// and snapped areas each the geometry's own 24 m^2 - the snap moved
    /// nothing, so both walks measure the same unit quads - both ratios 1,
    /// and no pinned point of any kind, wall or otherwise.
    #[test]
    fn the_cube_on_the_cell_planes_reports_its_whole_area() {
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
        assert_eq!(snapped.report.patch_areas.len(), 1);
        let row = &snapped.report.patch_areas[0];
        assert_eq!(row.name, "cube");
        let near = |a: Scalar, b: Scalar| (a - b).abs() <= 1e-12 * b.abs();
        assert!(near(row.stl_area, 24.0), "stl area {}", row.stl_area);
        assert!(
            near(row.castellated_area, 24.0),
            "castellated area {}",
            row.castellated_area
        );
        assert!(
            near(row.snapped_area, 24.0),
            "snapped area {}",
            row.snapped_area
        );
        let r = row.ratio().expect("ratio");
        let cr = row.castellated_ratio().expect("castellated ratio");
        assert!((r - 1.0).abs() <= 1e-12, "ratio {r}");
        assert!((cr - 1.0).abs() <= 1e-12, "castellated ratio {cr}");
        assert_eq!(snapped.report.n_pinned_boundary, 0);
    }

    /// An abandoned iterate pins more than the boundary: the pinned points
    /// that lie in B of (92.27) stay inside the boundary count and under
    /// the pinned total, and with the mesh returned unmoved each patch's
    /// re-measured area is bit for bit the castellated one.
    #[test]
    fn an_abandoned_iterate_pins_more_than_the_boundary() {
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
        eprintln!(
            "n_pinned {} n_pinned_boundary {} n_boundary_points {}",
            snapped.report.n_pinned,
            snapped.report.n_pinned_boundary,
            snapped.report.n_boundary_points
        );
        assert!(snapped.report.n_pinned_boundary > 0);
        assert!(
            snapped.report.n_pinned_boundary <= snapped.report.n_boundary_points,
            "a boundary-only count cannot exceed the boundary points"
        );
        assert!(
            snapped.report.n_pinned_boundary < snapped.report.n_pinned,
            "the pinned total also counts points outside B"
        );
        for row in &snapped.report.patch_areas {
            assert_eq!(
                row.snapped_area.to_bits(),
                row.castellated_area.to_bits(),
                "the mesh came back unmoved, so the two walks agree exactly"
            );
        }
    }

    /// The snapped sphere's mesh area lands within ten per cent of its
    /// surface's own, and the snap moved the castellated mesh toward that:
    /// a staircase over-reports a curved surface's area (by up to sqrt(3)),
    /// so the castellated ratio sits above the snapped one.
    #[test]
    fn the_snapped_sphere_area_is_near_its_surface_area() {
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
        assert_eq!(snapped.report.patch_areas.len(), 1);
        let row = &snapped.report.patch_areas[0];
        assert_eq!(row.name, "sphere");
        let r = row.ratio().expect("ratio");
        let cr = row.castellated_ratio().expect("castellated ratio");
        eprintln!("sphere snapped ratio {r} castellated ratio {cr}");
        assert!(0.9 < r && r < 1.1, "ratio {r} not in (0.9, 1.1)");
        assert!(cr > r, "castellated ratio {cr} not under snapped ratio {r}");
    }

    /// With the sphere declared a body, the interface faces (92.32) assigns
    /// to the patch count toward its areas, and the snapped mesh still
    /// covers the geometry: one row, a positive castellated area that
    /// includes those interfaces, and a snapped ratio within ten per cent.
    #[test]
    fn the_interface_area_counts_toward_the_snapped_ratio() {
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
        let cast_spec = CastellationSpec {
            bodies: vec![BodySpec {
                name: "sphere".to_string(),
                patches: vec!["sphere".to_string()],
            }],
            ..Default::default()
        };
        let cast = castellate(
            &tree,
            &bg,
            &surf,
            &patch_names(),
            &cast_spec,
            &thresholds(),
        )
        .expect("castellate");
        let snapped = snap_regions(
            &cast.mesh,
            &surf,
            Some(&cast.region_of_cell),
            1.0,
            30.0,
            &SnapSpec::default(),
            &thresholds(),
        )
        .expect("snap_regions");
        assert_eq!(snapped.report.patch_areas.len(), 1);
        let row = &snapped.report.patch_areas[0];
        assert_eq!(row.name, "sphere");
        let r = row.ratio().expect("ratio");
        let cr = row.castellated_ratio().expect("castellated ratio");
        eprintln!("body snapped ratio {r} castellated ratio {cr}");
        assert!(row.castellated_area > 0.0, "interface area must count");
        assert!(0.9 < r && r < 1.1, "ratio {r} not in (0.9, 1.1)");
    }

    /// The `a_cube_on_the_cell_planes_is_snapped_bit_for_bit` case, its
    /// castellated mesh only: a cube spanning [1, 3]^3 at base size 1 and
    /// level 0, so every wall face lies on a cube face and every wall edge
    /// along a cube edge lies on that edge.
    fn plane_cube_case() -> (Surface, PolyMeshRaw) {
        let (tree, bg) = setup([0.0, 4.0, 0.0, 4.0, 0.0, 4.0], 1.0, 0);
        let surf = Surface::from_soup(box_soup([1.0; 3], [3.0; 3]), vec!["cube".to_string()])
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

    /// On the cell planes the capture is the whole sharp length: 12 edges
    /// of 2, every one held by the wall edges along it.
    #[test]
    fn the_capture_on_the_cell_planes_is_the_whole_sharp_length() {
        let (surf, mesh) = plane_cube_case();
        let c = feature_capture(&mesh, &surf, None, 30.0, 0.1).expect("capture");
        eprintln!("plane cube capture {c:?}");
        assert!((c.sharp_length - 24.0).abs() <= 1e-12 * 24.0, "{c:?}");
        assert!((c.captured_length - c.sharp_length).abs() <= 1e-12 * 24.0, "{c:?}");
        assert_eq!(c.tol, 0.1);
    }

    /// Pull the mesh point at the middle of one cube edge half a cell off
    /// it, in a copy of the mesh: the two wall edges through it no longer
    /// cover that edge, and the capture falls by exactly its length, 2.
    #[test]
    fn a_point_pulled_off_an_edge_uncovers_that_edge() {
        let (surf, mut mesh) = plane_cube_case();
        let mid = Vec3::new(2.0, 1.0, 1.0);
        let i = (0..mesh.points.len())
            .min_by(|&a, &b| {
                (mesh.points[a] - mid).mag().total_cmp(&(mesh.points[b] - mid).mag())
            })
            .expect("points");
        assert!((mesh.points[i] - mid).mag() <= 1e-12, "no mesh point at the edge's middle");
        mesh.points[i] = Vec3::new(2.0, 0.5, 0.5);
        let c = feature_capture(&mesh, &surf, None, 30.0, 0.1).expect("capture");
        eprintln!("pulled point capture {c:?}");
        assert!((c.sharp_length - 24.0).abs() <= 1e-12 * 24.0, "{c:?}");
        assert!((c.captured_length - 22.0).abs() <= 1e-12 * 24.0, "{c:?}");
    }

    /// A smooth sphere has no sharp length, so nothing to capture, and the
    /// tolerance is reported as given.
    #[test]
    fn a_sphere_has_no_sharp_length() {
        let (surf, cast) = sphere_case();
        let snapped = snap(&cast, &surf, 1.0, 30.0, &SnapSpec::default(), &thresholds())
            .expect("snap");
        let c = feature_capture(&snapped.mesh, &surf, None, 30.0, 0.025).expect("capture");
        assert_eq!(c, FeatureCapture { sharp_length: 0.0, captured_length: 0.0, tol: 0.025 });
    }

    /// The off-lattice cube holds more of its edges with the attraction on
    /// than with it off - the chamfer of `feature_tolerance` 0 is what the
    /// capture exists to show. Both are printed for the record.
    #[test]
    fn the_attraction_raises_the_capture_on_an_off_lattice_cube() {
        let (surf, cast) = cube_case();
        let on = snap(&cast, &surf, 1.0, 30.0, &SnapSpec::default(), &thresholds())
            .expect("snap on");
        let off_spec = SnapSpec { feature_tolerance: 0.0, ..SnapSpec::default() };
        let off = snap(&cast, &surf, 1.0, 30.0, &off_spec, &thresholds()).expect("snap off");
        let c_on = feature_capture(&on.mesh, &surf, None, 30.0, 0.05).expect("on");
        let c_off = feature_capture(&off.mesh, &surf, None, 30.0, 0.05).expect("off");
        eprintln!("off-lattice cube capture on {c_on:?} off {c_off:?}");
        for c in [c_on, c_off] {
            assert!((c.sharp_length - 12.0).abs() <= 1e-12 * 12.0, "{c:?}");
            assert!(0.0 <= c.captured_length, "{c:?}");
            assert!(c.captured_length <= c.sharp_length * (1.0 + 1e-12), "{c:?}");
        }
        assert!(c_on.captured_length > c_off.captured_length, "on {c_on:?} off {c_off:?}");
    }

    /// The union counts an overlap once and a gap not at all.
    #[test]
    fn the_union_counts_an_overlap_once() {
        let mut a = vec![(0.5, 1.0), (0.0, 0.5), (0.2, 0.3)];
        assert_eq!(union_length(&mut a), 1.0);
        let mut b = vec![(0.5, 0.75), (0.0, 0.25)];
        assert_eq!(union_length(&mut b), 0.5);
        let mut c: Vec<(Scalar, Scalar)> = Vec::new();
        assert_eq!(union_length(&mut c), 0.0);
    }

    /// The cover test on its own: both ends within `tol` AND the direction
    /// within 30 degrees, or no cover; the interval is clamped to the
    /// segment and ordered.
    #[test]
    fn the_cover_test_needs_both_ends_near_and_the_direction_along() {
        let a = Vec3::new(0.0, 0.0, 0.0);
        let b = Vec3::new(1.0, 0.0, 0.0);
        let v = |x: f64, y: f64| Vec3::new(x, y, 0.0);
        let cov = |p: Vec3, q: Vec3| covered_interval(p, q, a, b, 0.1);
        assert_eq!(cov(v(0.2, 0.05), v(0.6, -0.05)), Some((0.2, 0.6)));
        assert_eq!(cov(v(0.6, -0.05), v(0.2, 0.05)), Some((0.2, 0.6)));
        assert_eq!(cov(v(0.2, 0.05), v(0.6, 0.2)), None);
        assert_eq!(cov(v(0.5, 0.0), v(0.5, 0.09)), None);
        assert_eq!(cov(v(0.5, 0.0), v(0.55, 0.05)), None);
        assert!(cov(v(0.5, 0.0), v(0.6, 0.05)).is_some());
        assert_eq!(cov(v(-0.05, 0.0), v(0.3, 0.0)), Some((0.0, 0.3)));
        assert_eq!(cov(v(0.3, 0.0), v(0.3, 0.0)), None);
    }

    /// The closed regular 16-gon of radius 1 in z = 0, point k at angle
    /// 2 pi k / 16, ids 0..16 with the last the first again.
    fn gon16_set() -> features::FeatureSet {
        let mut pts = Vec::new();
        for k in 0..16 {
            let a = 2.0 * std::f64::consts::PI * (k as f64) / 16.0;
            pts.push(Vec3::new(a.cos(), a.sin(), 0.0));
        }
        let mut pl: Vec<u32> = (0..16u32).collect();
        pl.push(0);
        features::FeatureSet {
            points: pts,
            edges: Vec::new(),
            edge_patches: Vec::new(),
            polylines: vec![pl],
            corners: Vec::new(),
            feature_angle_deg: 60.0,
        }
    }

    /// The six faces of the box, each split into `k` x `k` squares, each
    /// square two triangles wound OUTWARD like `box_soup`'s faces, patch 0.
    /// The coordinate along axis `d` at grid index `i` is
    /// `lo[d] + (hi[d] - lo[d]) * i / k`, the same expression on every
    /// face, so shared points weld bit for bit.
    fn grid_box_soup(lo: [f64; 3], hi: [f64; 3], k: usize) -> Vec<(u32, [Vec3; 3])> {
        let g = |d: usize, i: usize| lo[d] + (hi[d] - lo[d]) * (i as f64) / (k as f64);
        let mut soup: Vec<(u32, [Vec3; 3])> = Vec::new();
        for &(a, b, f, hi_f) in &[
            (0usize, 1usize, 2usize, false),
            (0, 1, 2, true),
            (0, 2, 1, false),
            (0, 2, 1, true),
            (1, 2, 0, false),
            (1, 2, 0, true),
        ] {
            let p = |u: usize, v: usize| {
                let mut x = [0.0; 3];
                x[a] = g(a, u);
                x[b] = g(b, v);
                x[f] = if hi_f { hi[f] } else { lo[f] };
                Vec3::new(x[0], x[1], x[2])
            };
            for i in 0..k {
                for j in 0..k {
                    let (c00, c10, c11, c01) =
                        (p(i, j), p(i + 1, j), p(i + 1, j + 1), p(i, j + 1));
                    // The two triangles, wound outward like box_soup's.
                    let (t1, t2) = match (a, f, hi_f) {
                        (0, 2, false) => ([c00, c11, c10], [c00, c01, c11]),
                        (0, 1, true) => ([c10, c00, c01], [c10, c01, c11]),
                        (1, 0, false) => ([c00, c01, c11], [c00, c11, c10]),
                        _ => ([c00, c10, c11], [c00, c11, c01]),
                    };
                    soup.push((0u32, t1));
                    soup.push((0u32, t2));
                }
            }
        }
        soup
    }

    /// `plane_cube_case` with `grid_box_soup([1, 3]^3, 3)` in place of
    /// `box_soup`: the same setup and castellation, the STL splitting every
    /// cube edge into 3 collinear segments.
    fn grid_cube_case() -> (Surface, PolyMeshRaw) {
        let (tree, bg) = setup([0.0, 4.0, 0.0, 4.0, 0.0, 4.0], 1.0, 0);
        let surf = Surface::from_soup(
            grid_box_soup([1.0; 3], [3.0; 3], 3),
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
        (surf, cast.mesh)
    }

    /// The chains cut a polyline where it turns further than the capture
    /// angle: a 45-degree kink into 2 and 1, a closed square into 4 open
    /// chains of 1, a closed 16-gon one closed chain.
    #[test]
    fn the_chains_cut_a_polyline_where_it_turns_further_than_the_capture_angle() {
        let fs_of = |points: Vec<Vec3>, pl: Vec<u32>| features::FeatureSet {
            points,
            edges: Vec::new(),
            edge_patches: Vec::new(),
            polylines: vec![pl],
            corners: Vec::new(),
            feature_angle_deg: 60.0,
        };
        let s05 = 0.5f64.sqrt();
        let a = fs_of(
            vec![
                Vec3::new(0.0, 0.0, 0.0),
                Vec3::new(1.0, 0.0, 0.0),
                Vec3::new(2.0, 0.0, 0.0),
                Vec3::new(2.0 + s05, s05, 0.0),
            ],
            vec![0, 1, 2, 3],
        );
        let ch = capture_chains(&a);
        assert_eq!(ch.len(), 2, "{ch:?}");
        assert!(ch.iter().all(|c| !c.closed), "{ch:?}");
        assert_eq!(ch[0].points.len(), 3, "{:?}", ch[0].points);
        assert!((ch[0].length() - 2.0).abs() <= 1e-12, "{}", ch[0].length());
        assert_eq!(ch[1].points.len(), 2, "{:?}", ch[1].points);
        assert!((ch[1].length() - 1.0).abs() <= 1e-12, "{}", ch[1].length());
        eprintln!("chain cut open: lengths {} {}", ch[0].length(), ch[1].length());
        let b = fs_of(
            vec![
                Vec3::new(0.0, 0.0, 0.0),
                Vec3::new(1.0, 0.0, 0.0),
                Vec3::new(1.0, 1.0, 0.0),
                Vec3::new(0.0, 1.0, 0.0),
            ],
            vec![0, 1, 2, 3, 0],
        );
        let ch = capture_chains(&b);
        assert_eq!(ch.len(), 4, "{ch:?}");
        for c in &ch {
            assert!(!c.closed, "{c:?}");
            assert_eq!(c.points.len(), 2, "{:?}", c.points);
            assert!((c.length() - 1.0).abs() <= 1e-12, "{}", c.length());
        }
        assert_eq!(ch[0].points[0], Vec3::new(0.0, 0.0, 0.0), "{:?}", ch[0].points);
        assert_eq!(ch[0].points[1], Vec3::new(1.0, 0.0, 0.0), "{:?}", ch[0].points);
        eprintln!("chain cut closed square: {} chains of 1", ch.len());
        let ch = capture_chains(&gon16_set());
        assert_eq!(ch.len(), 1, "{ch:?}");
        assert!(ch[0].closed, "{:?}", ch[0]);
        assert_eq!(ch[0].points.len(), 17, "{} points", ch[0].points.len());
        for w in ch[0].arc.windows(2) {
            assert!(w[0] < w[1], "arc not rising: {:?}", ch[0].arc);
        }
        let want = 32.0 * (std::f64::consts::PI / 16.0).sin();
        assert!((ch[0].length() - want).abs() <= 1e-12, "{} vs {want}", ch[0].length());
        eprintln!("chain 16-gon: closed length {}", ch[0].length());
    }

    /// On one segment the chain cover is the segment cover: the same
    /// intervals where that one is of positive length, no cover where it
    /// has none.
    #[test]
    fn on_one_segment_the_chain_cover_is_the_segment_cover() {
        let fs = features::FeatureSet {
            points: vec![Vec3::new(0.0, 0.0, 0.0), Vec3::new(1.0, 0.0, 0.0)],
            edges: Vec::new(),
            edge_patches: Vec::new(),
            polylines: vec![vec![0, 1]],
            corners: Vec::new(),
            feature_angle_deg: 60.0,
        };
        let ch = &capture_chains(&fs)[0];
        let segs = [0usize];
        let v = |x: f64, y: f64| Vec3::new(x, y, 0.0);
        let cases: [(Vec3, Vec3, Option<(f64, f64)>); 8] = [
            (v(0.2, 0.05), v(0.6, -0.05), Some((0.2, 0.6))),
            (v(0.6, -0.05), v(0.2, 0.05), Some((0.2, 0.6))),
            (v(0.2, 0.05), v(0.6, 0.2), None),
            (v(0.5, 0.0), v(0.5, 0.09), None),
            (v(0.5, 0.0), v(0.55, 0.05), None),
            (v(0.5, 0.0), v(0.6, 0.05), Some((0.5, 0.6))),
            (v(-0.05, 0.0), v(0.3, 0.0), Some((0.0, 0.3))),
            (v(0.3, 0.0), v(0.3, 0.0), None),
        ];
        for (p, q, want) in cases {
            let fp = project_on_chain(p, ch, &segs).expect("p projects");
            let fq = project_on_chain(q, ch, &segs).expect("q projects");
            let got = chain_cover(p, q, fp, fq, ch, 0.1);
            match (got, want) {
                (Some((s0, s1)), Some((t0, t1))) => {
                    assert!((s0 - t0).abs() <= 1e-15 && (s1 - t1).abs() <= 1e-15, "{got:?}");
                }
                (None, None) => {}
                _ => panic!("cover {got:?} vs segment {want:?} for {p:?} {q:?}"),
            }
            let seg = covered_interval(p, q, ch.points[0], ch.points[1], 0.1)
                .filter(|&(t0, t1)| t1 > t0);
            match (got, seg) {
                (None, None) => {}
                (Some((s0, s1)), Some((t0, t1))) => {
                    assert!((s0 - t0).abs() <= 1e-15 && (s1 - t1).abs() <= 1e-15, "{got:?}");
                }
                _ => panic!("agreement {got:?} vs {seg:?} for {p:?} {q:?}"),
            }
        }
        eprintln!("capture on one segment: {} cases agree", cases.len());
    }

    /// A closed chain is covered across its seam: the edge between the
    /// midpoints of the last and first sides covers one side's length
    /// across the seam, and the 16 edges cover the whole loop.
    #[test]
    fn a_closed_chain_is_covered_across_its_seam() {
        let ch = &capture_chains(&gon16_set())[0];
        let l = 2.0 * (std::f64::consts::PI / 16.0).sin();
        let segs: Vec<usize> = (0..16).collect();
        let mid = |k: usize| {
            let (a, b) = (ch.points[k], ch.points[(k + 1) % 16]);
            a + (b - a) * 0.5
        };
        let (p, q) = (mid(15), mid(0));
        let fp = project_on_chain(p, ch, &segs).expect("p projects");
        let fq = project_on_chain(q, ch, &segs).expect("q projects");
        assert!(fp.1 <= 1e-15, "{}", fp.1);
        assert!(fq.1 <= 1e-15, "{}", fq.1);
        assert!((fp.0 - 15.5 * l).abs() <= 1e-12, "{} vs {}", fp.0, 15.5 * l);
        assert!((fq.0 - 0.5 * l).abs() <= 1e-12, "{} vs {}", fq.0, 0.5 * l);
        let iv = chain_cover(p, q, fp, fq, ch, 0.05).expect("seam cover");
        assert!(
            (iv.0 - 15.5 * l).abs() <= 1e-12 && (iv.1 - 16.5 * l).abs() <= 1e-12,
            "{iv:?}"
        );
        eprintln!("capture seam interval {iv:?} against L {}", ch.length());
        let mut ivs: Vec<(Scalar, Scalar)> = Vec::new();
        for k in 0..16 {
            let (p, q) = (mid(k), mid((k + 1) % 16));
            let fp = project_on_chain(p, ch, &segs).expect("p projects");
            let fq = project_on_chain(q, ch, &segs).expect("q projects");
            if let Some((s0, s1)) = chain_cover(p, q, fp, fq, ch, 0.05) {
                if s1 > ch.length() {
                    ivs.push((s0, ch.length()));
                    ivs.push((0.0, s1 - ch.length()));
                } else {
                    ivs.push((s0, s1));
                }
            }
        }
        let total = union_length(&mut ivs);
        assert!((total - 16.0 * l).abs() <= 1e-12, "{total} vs {}", 16.0 * l);
        assert!((total - ch.length()).abs() <= 1e-12, "{total} vs {}", ch.length());
        eprintln!("capture seam loop: union {total} of L {}", ch.length());
    }

    /// A hairpin does not carry a short edge round its bend: the arc from
    /// one arm to the other exceeds the edge by more than `2 tol`.
    #[test]
    fn a_hairpin_does_not_carry_a_short_edge_round_its_bend() {
        let mut pts = vec![Vec3::new(0.0, 0.0, 0.0), Vec3::new(1.0, 0.0, 0.0)];
        for j in 1..=8 {
            let a = (-90.0 + 22.5 * (j as f64)).to_radians();
            pts.push(Vec3::new(1.0 + 0.04 * a.cos(), 0.04 + 0.04 * a.sin(), 0.0));
        }
        pts.push(Vec3::new(0.0, 0.08, 0.0));
        assert_eq!(pts.len(), 11, "{} points", pts.len());
        let fs = features::FeatureSet {
            points: pts,
            edges: Vec::new(),
            edge_patches: Vec::new(),
            polylines: vec![(0..11u32).collect()],
            corners: Vec::new(),
            feature_angle_deg: 60.0,
        };
        let chains = capture_chains(&fs);
        assert_eq!(chains.len(), 1, "{chains:?}");
        assert!(!chains[0].closed, "{:?}", chains[0]);
        let ch = &chains[0];
        let segs: Vec<usize> = (0..ch.points.len() - 1).collect();
        let p = Vec3::new(0.5, 0.0, 0.0);
        let q = Vec3::new(0.55, 0.08, 0.0);
        let fp = project_on_chain(p, ch, &segs).expect("p projects");
        let fq = project_on_chain(q, ch, &segs).expect("q projects");
        assert!(fp.1 <= 1e-15, "{}", fp.1);
        assert!(fq.1 <= 1e-15, "{}", fq.1);
        assert!(chain_cover(p, q, fp, fq, ch, 0.05).is_none(), "hairpin cover");
        let q2 = Vec3::new(0.6, 0.0, 0.0);
        let fq2 = project_on_chain(q2, ch, &segs).expect("q2 projects");
        let iv = chain_cover(p, q2, fp, fq2, ch, 0.05).expect("straight cover");
        assert!((iv.0 - 0.5).abs() <= 1e-12 && (iv.1 - 0.6).abs() <= 1e-12, "{iv:?}");
        eprintln!("capture hairpin: length {} straight {iv:?}", ch.length());
    }

    /// A subdivided cube on the cell planes is captured whole: 36 feature
    /// edges in 12 chains, `captured_length` 24 where the per-segment
    /// measure reads 0.
    #[test]
    fn a_subdivided_cube_on_the_cell_planes_is_captured_whole() {
        let (surf, mesh) = grid_cube_case();
        let fs = features::extract(&surf, 30.0).expect("features");
        assert_eq!(fs.edges.len(), 36, "{} edges", fs.edges.len());
        assert_eq!(fs.polylines.len(), 12, "{} polylines", fs.polylines.len());
        assert_eq!(fs.corners.len(), 8, "{} corners", fs.corners.len());
        let chains = capture_chains(&fs);
        assert_eq!(chains.len(), 12, "{} chains", chains.len());
        for c in &chains {
            assert!(!c.closed, "{c:?}");
            assert!((c.length() - 2.0).abs() <= 1e-12, "{}", c.length());
        }
        let c = feature_capture(&mesh, &surf, None, 30.0, 0.1).expect("capture");
        eprintln!("subdivided cube capture {c:?}");
        assert!((c.sharp_length - 24.0).abs() <= 1e-12 * 24.0, "{c:?}");
        assert!((c.captured_length - 24.0).abs() <= 1e-12 * 24.0, "{c:?}");
        // The per-segment reading on the same mesh, for the record and as
        // proof the chain matters: every wall edge is 1 long and every
        // segment 2/3, so no wall edge has both ends within 0.1 of one
        // segment.
        let wall = wall_face_mask(&mesh, &surf, None);
        let mut wedges: Vec<(u32, u32)> = Vec::new();
        for (f, face) in mesh.faces.iter().enumerate() {
            if wall[f] {
                for k in 0..face.len() {
                    let (a, b) = (face[k] as u32, face[(k + 1) % face.len()] as u32);
                    wedges.push((a.min(b), a.max(b)));
                }
            }
        }
        wedges.sort_unstable();
        wedges.dedup();
        let mut per_seg: Vec<Vec<(Scalar, Scalar)>> = vec![Vec::new(); fs.edges.len()];
        for &(i, j) in &wedges {
            let (p, q) = (mesh.points[i as usize], mesh.points[j as usize]);
            for e in 0..fs.edges.len() {
                let (a, b) =
                    (fs.points[fs.edges[e][0] as usize], fs.points[fs.edges[e][1] as usize]);
                if let Some(iv) = covered_interval(p, q, a, b, 0.1) {
                    per_seg[e].push(iv);
                }
            }
        }
        let mut total = 0.0;
        for e in 0..fs.edges.len() {
            let (a, b) =
                (fs.points[fs.edges[e][0] as usize], fs.points[fs.edges[e][1] as usize]);
            total += union_length(&mut per_seg[e]) * (b - a).mag();
        }
        assert_eq!(total, 0.0, "per-segment capture {total}");
        eprintln!("capture per-segment reading on the same mesh: {total}");
    }

    /// Pull the mesh point at the middle of one subdivided cube edge half
    /// a cell off it: the chain measure takes that edge's 2 away, 24 to 22.
    #[test]
    fn a_point_pulled_off_a_subdivided_edge_uncovers_that_edge() {
        let (surf, mut mesh) = grid_cube_case();
        let mid = Vec3::new(2.0, 1.0, 1.0);
        let i = (0..mesh.points.len())
            .min_by(|&a, &b| {
                (mesh.points[a] - mid).mag().total_cmp(&(mesh.points[b] - mid).mag())
            })
            .expect("points");
        assert!((mesh.points[i] - mid).mag() <= 1e-12, "no mesh point at the edge's middle");
        mesh.points[i] = Vec3::new(2.0, 0.5, 0.5);
        let c = feature_capture(&mesh, &surf, None, 30.0, 0.1).expect("capture");
        eprintln!("subdivided pulled point capture {c:?}");
        assert!((c.sharp_length - 24.0).abs() <= 1e-12 * 24.0, "{c:?}");
        assert!((c.captured_length - 22.0).abs() <= 1e-12 * 24.0, "{c:?}");
    }
}
