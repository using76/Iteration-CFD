// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.
// Provenance: see PROVENANCE.md. No GPL-licensed source was consulted.

//! The castellation keep-set - SPEC-LIT §92.10's removal walk.
//!
//! §92.2 stage 3 decides, of the leaves [`super::octree::emit`] emitted, which
//! survive to be cells. A leaf is FLUID or SOLID by its centre alone,
//! eq. (92.23): `inside(surf, c(P))`, §23.3's three-axis parity pipeline
//! (`surface::classify::classify_points`) evaluated at a point. The fluid
//! leaves then go through the removal walk eq. (92.24): W1, the pinch repair
//! of eq. (92.25), and W2, the region rule of eq. (92.4) - `largest`, or the
//! component containing the seed - to a joint fixed point, then W3, the
//! starved-cell check. Every rule only removes; the walk refuses a whole
//! domain rather than emit an empty mesh, and reports what it removed in
//! [`Removed`].
//!
//! [`castellate`] then turns that keep mask into the castellated mesh
//! itself: it drops the removed cells, walls every face they exposed by
//! (92.26), renumbers, and passes §92.3's gate before returning.
//!
//! Provenance: ORIGINAL - the walk is stated in SPEC-LIT §92.10 with
//! equations (92.23)-(92.25) against §92.3's gate. No GPL-licensed source was
//! consulted.

use std::collections::HashMap;

use crate::error::{Error, Result};
use crate::io::polymesh::PolyMeshRaw;
use crate::surface::classify::classify_points;
use crate::surface::{Surface, TriIndex};
use crate::{Scalar, Vec3};

use super::octree::{Background, Octree};
use super::{CastellationSpec, KeepRegion};

/// How many passes of (92.24) the walk will run before it refuses. K only
/// ever shrinks, so a walk that is still moving after this many passes is a
/// walk that will not converge, and must say so.
const MAX_PASSES: usize = 64;

/// What the walk (92.24) removed, by the rule that removed it.
#[derive(Debug, Clone, Default)]
pub struct Removed {
    /// Leaves whose CENTRE was inside the surface (92.23).
    pub solid: usize,
    /// Cells removed by the pinch rule (92.25).
    pub pinch: usize,
    /// Cells dropped with a component (92.4) that was not kept.
    pub off_region: usize,
    /// Cells with fewer than `min_faces` faces. Cannot fire on the hex
    /// path; counted so that the stage which makes it fire is caught.
    pub starved: usize,
    /// Four-face edges the rule could not name two cells for; left alone
    /// and reported rather than guessed at.
    pub unrepaired_pinch: usize,
    /// One `[xlo, xhi, ylo, yhi, zlo, zhi]` per dropped component, so a
    /// mesher that deletes 900 cells under a building has said where.
    pub dropped_boxes: Vec<[Scalar; 6]>,
}

/// The centres of `tree.leaves()`, in that order, and the finest leaf edge
/// in the tree - (92.23)'s `c(P)` and its jitter length `h_min`.
///
/// The arithmetic is [`super::octree::leaf_centre_edges`]'s, written there
/// once; `h_min` is the smallest edge over all leaves and all three axes,
/// because the points come from leaves of different sizes and one length has
/// to serve them all. It is `classify_leaves` that refuses a non-positive
/// one, by name.
pub fn leaf_centres(tree: &Octree, bg: &Background) -> (Vec<Vec3>, Scalar) {
    let l = tree.max_level();
    let mut centres = Vec::with_capacity(tree.len());
    let mut h_min = Scalar::INFINITY;
    for k in tree.leaves() {
        let (c, e) = super::octree::leaf_centre_edges(bg, l, k);
        for d in e {
            if d < h_min {
                h_min = d;
            }
        }
        centres.push(c);
    }
    (centres, h_min)
}

/// (92.23): `true` where the leaf's CENTRE is inside `surf`, in
/// `tree.leaves()` order.
///
/// The surface cutting through a leaf does not split it and does not remove
/// it - the cut is stage 4's business. The classification is §23.3's
/// pipeline, jitted by the tree's finest leaf edge; the surface's
/// orientation is irrelevant, as it is in §23.3.
pub fn classify_leaves(
    tree: &Octree,
    bg: &Background,
    surf: &Surface,
) -> Result<Vec<bool>> {
    let (centres, h_min) = leaf_centres(tree, bg);
    if !centres.is_empty() && !(h_min > 0.0) {
        return Err(Error::Mesh(format!(
            "castellate: the finest leaf edge is {h_min:?} - (92.23) needs a \
             positive jitter length, so the background block needs positive cell sizes"
        )));
    }
    let mask = classify_points(surf, &centres, h_min)?;
    Ok(mask.solid)
}

/// The cell's face neighbours currently in K - (92.25)'s count. Two leaves
/// share at most one face, so this counts neighbours, not face incidences.
fn nbrs_in_k(
    keep: &[bool],
    cell_faces: &[Vec<usize>],
    owner: &[crate::Label],
    neighbour: &[crate::Label],
    n_int: usize,
    c: usize,
) -> usize {
    cell_faces[c]
        .iter()
        .filter(|&&f| {
            if f >= n_int {
                return false;
            }
            let other = if owner[f] as usize == c {
                neighbour[f] as usize
            } else {
                owner[f] as usize
            };
            keep[other]
        })
        .count()
}

/// (92.24)/(92.25): the keep mask over the cells of `full` (which must be
/// `octree::emit`'s mesh for the same tree, so cell `i` is leaf `i`).
/// `solid[i]` is (92.23)'s answer for cell `i`, `centres[i]` its centre.
pub fn keep_set(
    full: &PolyMeshRaw,
    solid: &[bool],
    centres: &[Vec3],
    spec: &CastellationSpec,
) -> Result<(Vec<bool>, Removed)> {
    let n_faces = full.faces.len();
    let n_int = full.neighbour.len();
    if full.owner.len() != n_faces {
        return Err(Error::Mesh(format!(
            "castellate: owner has {} entries but faces has {} - the mesh's \
             addressing is broken, so no keep-set can be walked",
            full.owner.len(),
            n_faces
        )));
    }
    // polyMesh stores no cell count; max(owner, neighbour) + 1 is it, the way
    // `io::polymesh` derives it too.
    let n_cells = full
        .owner
        .iter()
        .chain(full.neighbour.iter())
        .copied()
        .max()
        .map_or(0, |m| m as usize + 1);
    if solid.len() != n_cells || centres.len() != n_cells {
        return Err(Error::Mesh(format!(
            "castellate: the mesh has {} cells, but solid has {} answers and \
             centres has {} points - not the same leaves in the same order",
            n_cells,
            solid.len(),
            centres.len()
        )));
    }
    if spec.keep_region == KeepRegion::Seed && spec.seed_point.is_none() {
        return Err(Error::Mesh(
            "castellate: castellation.keep_region is \"seed\" but \
             castellation.seed_point is not set - the seed rule (92.4) has \
             nothing to seed on"
                .to_string(),
        ));
    }

    // Every cell's faces, once: the boundary of K, the neighbour counts and
    // W3's counts all read this one array.
    let mut cell_faces: Vec<Vec<usize>> = vec![Vec::new(); n_cells];
    for f in 0..n_faces {
        cell_faces[full.owner[f] as usize].push(f);
        if f < n_int {
            cell_faces[full.neighbour[f] as usize].push(f);
        }
    }

    // K_0, and `Removed::solid` counted here.
    let mut keep: Vec<bool> = solid.iter().map(|s| !s).collect();
    let mut rep = Removed::default();
    rep.solid = solid.iter().filter(|s| **s).count();

    let mut pass = 0;
    while pass < MAX_PASSES {
        let mut removed = 0;

        // W1, the pinch (92.25). Every edge the boundary of K carries, with
        // its carry count and the kept cell behind each carry.
        let mut edges: HashMap<(usize, usize), (usize, HashMap<usize, usize>)> = HashMap::new();
        for f in 0..n_faces {
            let own = full.owner[f] as usize;
            let on_k = if f < n_int {
                let nbr = full.neighbour[f] as usize;
                keep[own] != keep[nbr]
            } else {
                keep[own]
            };
            if !on_k {
                continue;
            }
            let held = if f < n_int {
                let nbr = full.neighbour[f] as usize;
                if keep[own] { own } else { nbr }
            } else {
                own
            };
            let ps = &full.faces[f];
            for i in 0..ps.len() {
                let a = ps[i] as usize;
                let b = ps[(i + 1) % ps.len()] as usize;
                let key = if a < b { (a, b) } else { (b, a) };
                let e = edges.entry(key).or_insert_with(|| (0, HashMap::new()));
                e.0 += 1;
                *e.1.entry(held).or_insert(0) += 1;
            }
        }
        // Decide for the WHOLE pass first, then apply together, so the result
        // does not depend on the order the edges were visited in.
        let mut decisions: Vec<usize> = Vec::new();
        for (carry, cells) in edges.values() {
            if *carry < 4 {
                continue;
            }
            let on_it_twice: Vec<usize> = cells
                .iter()
                .filter(|(_, &n)| n >= 2)
                .map(|(&cell, _)| cell)
                .collect();
            let (p, q) = match on_it_twice.as_slice() {
                [p, q] => (*p, *q),
                _ => {
                    rep.unrepaired_pinch += 1;
                    continue;
                }
            };
            let np = nbrs_in_k(&keep, &cell_faces, &full.owner, &full.neighbour, n_int, p);
            let nq = nbrs_in_k(&keep, &cell_faces, &full.owner, &full.neighbour, n_int, q);
            let drop = if np < nq {
                p
            } else if nq < np {
                q
            } else {
                p.max(q)
            };
            if !decisions.contains(&drop) {
                decisions.push(drop);
            }
        }
        for c in decisions {
            keep[c] = false;
            rep.pinch += 1;
            removed += 1;
        }

        // W2, the region (92.4).
        removed += keep_one_region(&mut keep, centres, spec, full, n_int, &mut rep)?;

        if removed == 0 {
            break;
        }
        pass += 1;
    }
    if pass == MAX_PASSES {
        return Err(Error::Mesh(format!(
            "castellate: the removal walk (92.24) is still moving after \
             {MAX_PASSES} passes - it will not converge, so refusing rather \
             than spinning"
        )));
    }

    // W3, starved. One pass: every leaf keeps its six faces or more whatever
    // its neighbours do - a removed neighbour leaves a WALL face where the
    // internal face was - so this cannot fire on the hex path today. It is
    // checked so the stage that makes it fire is caught on the day it does.
    for c in 0..n_cells {
        if keep[c] && cell_faces[c].len() < spec.min_faces {
            keep[c] = false;
            rep.starved += 1;
        }
    }

    if !keep.iter().any(|k| *k) {
        return Err(Error::Mesh(
            "castellate: the removal walk kept no cell - a castellation that \
             removes the whole domain is a refusal, not an empty mesh"
                .to_string(),
        ));
    }
    Ok((keep, rep))
}

/// W2 (92.4): of `keep`'s connected components - cells joined by the internal
/// faces whose BOTH cells are kept - leave one standing and drop the rest,
/// counted and boxed in `rep`. `Largest` keeps the biggest, ties to the
/// component with the lowest cell id; `Seed` keeps the one holding the kept
/// cell whose centre is nearest `seed_point` (`keep_set` has already refused
/// a seed with no point).
fn keep_one_region(
    keep: &mut [bool],
    centres: &[Vec3],
    spec: &CastellationSpec,
    full: &PolyMeshRaw,
    n_int: usize,
    rep: &mut Removed,
) -> Result<usize> {
    // Union-find over the kept cells, joined across internal faces.
    let n = keep.len();
    let mut parent: Vec<usize> = (0..n).collect();
    fn find(parent: &[usize], mut x: usize) -> usize {
        while parent[x] != x {
            x = parent[x];
        }
        x
    }
    for f in 0..n_int {
        let a = full.owner[f] as usize;
        let b = full.neighbour[f] as usize;
        if keep[a] && keep[b] {
            let (ra, rb) = (find(&parent, a), find(&parent, b));
            if ra != rb {
                parent[ra.max(rb)] = parent[ra.min(rb)];
            }
        }
    }
    // One cell list per component, each ascending (cells are scanned in id
    // order), then the components ordered by their lowest cell id - so the
    // answer never depends on the map's iteration order.
    let mut comps: HashMap<usize, Vec<usize>> = HashMap::new();
    for c in 0..n {
        if keep[c] {
            comps.entry(find(&parent, c)).or_default().push(c);
        }
    }
    let mut comps: Vec<Vec<usize>> = comps.into_values().collect();
    comps.sort_by_key(|c| c[0]);
    if comps.len() <= 1 {
        return Ok(0);
    }

    let held = match spec.keep_region {
        KeepRegion::Largest => {
            let mut best = 0;
            for i in 1..comps.len() {
                if comps[i].len() > comps[best].len() {
                    best = i;
                }
            }
            best
        }
        KeepRegion::Seed => {
            let sp = spec.seed_point.expect("keep_set checked the seed point");
            let p = Vec3::new(sp[0] as Scalar, sp[1] as Scalar, sp[2] as Scalar);
            let mut best: Option<(Scalar, usize)> = None;
            for (i, cells) in comps.iter().enumerate() {
                let d = cells
                    .iter()
                    .map(|&c| (centres[c] - p).mag())
                    .fold(Scalar::INFINITY, Scalar::min);
                if best.map_or(true, |(bd, _)| d < bd) {
                    best = Some((d, i));
                }
            }
            best.expect("comps is not empty").1
        }
    };

    let mut dropped = 0;
    for (i, cells) in comps.iter().enumerate() {
        if i == held {
            continue;
        }
        let mut lo = [Scalar::INFINITY; 3];
        let mut hi = [Scalar::NEG_INFINITY; 3];
        for &c in cells {
            keep[c] = false;
            dropped += 1;
            let p = [centres[c].x, centres[c].y, centres[c].z];
            for a in 0..3 {
                lo[a] = lo[a].min(p[a]);
                hi[a] = hi[a].max(p[a]);
            }
        }
        rep.dropped_boxes.push([lo[0], hi[0], lo[1], hi[1], lo[2], hi[2]]);
    }
    rep.off_region += dropped;
    Ok(dropped)
}

// ===========================================================================
//  The castellated mesh - (92.2) stage 3's rebuild of `emit`'s leaf mesh
// ===========================================================================

/// SPEC-LIT §92.2 stage 3's output: the castellated mesh and what it cost.
#[derive(Debug, Clone)]
pub struct Castellated {
    /// The mesh, real points, ready for stage 4.
    pub mesh: PolyMeshRaw,
    /// What the walk (92.24) removed, by rule.
    pub removed: Removed,
    /// How many faces (92.26) turned into walls.
    pub wall_faces: usize,
    /// The gate of §92.3, which this mesh has passed.
    pub report: super::quality::QualityReport,
}

/// §92.2 stage 3 end to end: emit the leaf mesh, classify by (92.23), run
/// the walk (92.24), wall the exposed faces by (92.26), renumber, and pass
/// §92.3's gate before returning.
pub fn castellate(
    tree: &Octree,
    bg: &Background,
    surf: &Surface,
    names: &[String; 6],
    spec: &CastellationSpec,
    thresholds: &super::quality::QualityThresholds,
) -> Result<Castellated> {
    let full = super::octree::emit(tree, bg, names)?;
    let n_internal = full.neighbour.len();

    // The centres, walked ONCE: (92.23)'s classification, the keep-set walk
    // and the wall index's hint all read the one pass - which is why the
    // classification is `classify_points` here and not `classify_leaves`,
    // which would walk the tree a second time. The refusal is
    // `classify_leaves`'s, made here because it is this stage that needs the
    // length.
    let (centres, h_min) = leaf_centres(tree, bg);
    if !centres.is_empty() && !(h_min > 0.0) {
        return Err(Error::Mesh(format!(
            "castellate: the finest leaf edge is {h_min:?} - (92.23) needs a \
             positive jitter length, so the background block needs positive cell sizes"
        )));
    }
    let solid = classify_points(surf, &centres, h_min)?.solid;
    let (keep, removed) = keep_set(&full, &solid, &centres, spec)?;

    // The renumbering: old ids scanned ascending, so the map is MONOTONE -
    // which is what lets the surviving internal faces inherit (2)'s
    // upper-triangular order instead of being re-sorted into it.
    let mut new_of_old: Vec<Option<crate::Label>> = Vec::with_capacity(keep.len());
    let mut next: crate::Label = 0;
    for &k in &keep {
        if k {
            new_of_old.push(Some(next));
            next += 1;
        } else {
            new_of_old.push(None);
        }
    }
    if next == 0 {
        return Err(Error::Mesh(
            "castellate: no cell survived the walk - refusing rather than \
             emitting an empty mesh"
                .to_string(),
        ));
    }

    // The wall index (92.26), one for the whole run, hinted by the tree's
    // finest leaf edge - the length the classifier's jitter uses.
    let tidx = TriIndex::new(surf, h_min.max(Scalar::MIN_POSITIVE))?;

    let mut internal: Vec<(crate::Label, crate::Label, Vec<crate::Label>)> = Vec::new();
    let mut box_faces: [Vec<(crate::Label, Vec<crate::Label>)>; 6] =
        std::array::from_fn(|_| Vec::new());
    let mut walls: Vec<Vec<(crate::Label, Vec<crate::Label>)>> =
        vec![Vec::new(); surf.patch_names.len()];

    for f in 0..full.faces.len() {
        let ps = &full.faces[f];
        let own = full.owner[f];
        if f < n_internal {
            let nbr = full.neighbour[f];
            match (keep[own as usize], keep[nbr as usize]) {
                (true, true) => {
                    let o = new_of_old[own as usize].expect("kept cell is mapped");
                    let n = new_of_old[nbr as usize].expect("kept cell is mapped");
                    if o >= n {
                        return Err(Error::Mesh(format!(
                            "castellate: face {f} came out owner {o} >= neighbour {n} - \
                             the renumbering is not monotone"
                        )));
                    }
                    internal.push((o, n, ps.clone()));
                }
                (true, false) | (false, true) => {
                    // A WALL face (92.26). The kept side owns it; if that
                    // side was the neighbour, the point list reverses with
                    // the ownership, so `Sf` keeps pointing out of the
                    // fluid.
                    let (holder, ps) = if keep[own as usize] {
                        (new_of_old[own as usize].expect("kept cell is mapped"), ps.clone())
                    } else {
                        (
                            new_of_old[nbr as usize].expect("kept cell is mapped"),
                            ps.iter().rev().copied().collect(),
                        )
                    };
                    let (t, _) = tidx.nearest_triangle(face_centre(&full.points, &ps));
                    walls[surf.tri_patch[t] as usize].push((holder, ps));
                }
                (false, false) => {}
            }
        } else if keep[own as usize] {
            // A box face of a kept cell: it keeps its patch, named in
            // `emit`'s own list. A removed cell's box face goes with it.
            let b = f - n_internal;
            let slot = match full
                .patches
                .iter()
                .position(|p| b >= p.start && b < p.start + p.size)
            {
                Some(s) => s,
                None => {
                    return Err(Error::Mesh(format!(
                        "castellate: boundary face {f} lies in no patch of the \
                         emitted mesh - the patch list is broken"
                    )));
                }
            };
            box_faces[slot].push((
                new_of_old[own as usize].expect("kept cell is mapped"),
                ps.clone(),
            ));
        }
    }

    // (92.26)'s "Order": the monotone map leaves the walked order sorted by
    // (owner, neighbour) - assert it; neither sort nor `ldu_permutation`.
    for (i, w) in internal.windows(2).enumerate() {
        if (w[0].0, w[0].1) >= (w[1].0, w[1].1) {
            return Err(Error::Mesh(format!(
                "castellate: internal faces {i} and {} came out (({}, {}) then \
                 ({}, {})) - not in (2)'s upper-triangular order",
                i + 1,
                w[0].0,
                w[0].1,
                w[1].0,
                w[1].1
            )));
        }
    }

    // The assembly: internal faces first, in the order they were walked;
    // then the six box patches in slot order, each keeping its faces'
    // original relative order and emitted even when empty; then the wall
    // patches in the SURFACE's patch order - the boundary file is a function
    // of the STL, not of the order the walk exposed the faces in.
    let mut faces: Vec<Vec<crate::Label>> = Vec::new();
    let mut owner: Vec<crate::Label> = Vec::new();
    let mut neighbour: Vec<crate::Label> = Vec::with_capacity(internal.len());
    for (o, n, ps) in &internal {
        faces.push(ps.clone());
        owner.push(*o);
        neighbour.push(*n);
    }
    // `PatchInfo::start` counts from the FIRST boundary face - `emit`'s
    // arithmetic, copied rather than re-derived.
    let mut patches: Vec<crate::mesh::PatchInfo> = Vec::new();
    for (slot, bf) in box_faces.iter().enumerate() {
        let start = faces.len() - internal.len();
        for (o, ps) in bf {
            faces.push(ps.clone());
            owner.push(*o);
        }
        patches.push(crate::mesh::PatchInfo {
            name: names[slot].clone(),
            type_name: "patch".to_string(),
            kind: crate::mesh::PatchKind::from_type("patch"),
            start,
            size: bf.len(),
            nbr_patch: None,
        });
    }

    let mut n_wall = 0;
    for (p, wf) in walls.into_iter().enumerate() {
        if wf.is_empty() {
            continue;
        }
        let name = surf.patch_names[p].clone();
        if names.iter().any(|q| q == &name) {
            // A silent rename would detach the user's boundary conditions
            // from their geometry, and a duplicate patch name in the
            // boundary file is a mesh the solver reads wrongly - `carve`'s
            // own rule (23.4).
            return Err(Error::Mesh(format!(
                "castellate: surface patch '{name}' collides with a domain \
                 patch of the same name - rename the STL solid"
            )));
        }
        let start = faces.len() - internal.len();
        for (o, ps) in &wf {
            faces.push(ps.clone());
            owner.push(*o);
        }
        n_wall += wf.len();
        patches.push(crate::mesh::PatchInfo {
            name,
            type_name: "wall".to_string(),
            kind: crate::mesh::PatchKind::from_type("wall"),
            start,
            size: wf.len(),
            nbr_patch: None,
        });
    }

    // The points are `full.points` unchanged: an unused point is legal and
    // costs one Vec3, while renumbering them would touch every face list
    // for nothing.
    let mesh = PolyMeshRaw {
        points: full.points,
        faces,
        owner,
        neighbour,
        patches,
    };
    // §92.3's gate: its refusal is this stage's, unwrapped no further.
    let report = super::quality::check(&mesh, thresholds)?;
    Ok(Castellated {
        mesh,
        removed,
        wall_faces: n_wall,
        report,
    })
}

/// A face's point average - (92.26)'s `x_f`.
fn face_centre(points: &[Vec3], ps: &[crate::Label]) -> Vec3 {
    let mut x = [0.0 as Scalar; 3];
    for p in ps {
        let q = points[*p as usize];
        x[0] += q.x;
        x[1] += q.y;
        x[2] += q.z;
    }
    let n = ps.len() as Scalar;
    Vec3::new(x[0] / n, x[1] / n, x[2] / n)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::automesher::octree::patch_names;
    use crate::automesher::DomainSpec;

    /// The uniform tree over a `base_size` block, and its `emit` mesh.
    fn setup(
        extent: [f64; 6],
        base: f64,
    ) -> (Octree, Background, PolyMeshRaw, Vec<Vec3>) {
        let bg = Background::from_domain(&DomainSpec {
            extent,
            base_size: base,
            grading: [1.0; 3],
        })
        .expect("background");
        let tree = Octree::uniform(bg.base_n(), 0).expect("uniform");
        let full =
            crate::automesher::octree::emit(&tree, &bg, &patch_names()).expect("emit");
        let (centres, _) = leaf_centres(&tree, &bg);
        (tree, bg, full, centres)
    }

    #[test]
    fn no_solid_keeps_every_cell() {
        let (_tree, _bg, full, centres) = setup([0.0, 3.0, 0.0, 3.0, 0.0, 3.0], 1.0);
        let solid = vec![false; centres.len()];
        let (keep, rep) =
            keep_set(&full, &solid, &centres, &CastellationSpec::default()).expect("keep set");
        assert_eq!(keep.iter().filter(|k| **k).count(), 27);
        assert_eq!(rep.solid, 0);
        assert_eq!(rep.pinch, 0);
        assert_eq!(rep.off_region, 0);
        assert_eq!(rep.starved, 0);
        assert_eq!(rep.unrepaired_pinch, 0);
        assert!(rep.dropped_boxes.is_empty());
    }

    /// The [2,2,1] tree's leaf ids are (k, j, i) order, so `solid` names the
    /// cells directly. 0 and 3 are the diagonal pair meeting at the middle
    /// vertical edge only.
    #[test]
    fn the_hourglass_loses_one_cell() {
        let (_tree, _bg, full, centres) = setup([0.0, 2.0, 0.0, 2.0, 0.0, 1.0], 1.0);
        assert_eq!(centres.len(), 4);
        let solid = [false, true, true, false].to_vec();
        let (keep, rep) =
            keep_set(&full, &solid, &centres, &CastellationSpec::default()).expect("keep set");
        assert_eq!(rep.pinch, 1);
        assert!(!keep[1] && !keep[2], "the solid pair stays gone");
        assert!(keep[0], "cell 0 survives - the tie goes to the LARGER id");
        assert!(!keep[3], "cell 3 is the one (92.25) removes");
        assert_eq!(keep.iter().filter(|k| **k).count(), 1);
        assert_eq!(rep.off_region, 0);
        assert_eq!(rep.unrepaired_pinch, 0, "a box corner is not a pinch");
    }

    /// The whole middle layer solid seals two 9-cell pockets; the tie on size
    /// goes to the one with the lowest cell id, the k == 0 layer.
    #[test]
    fn a_sealed_pocket_is_dropped() {
        let (_tree, _bg, full, centres) = setup([0.0, 3.0, 0.0, 3.0, 0.0, 3.0], 1.0);
        let solid: Vec<bool> = centres
            .iter()
            .map(|c| c.z > 0.5 && c.z < 2.5)
            .collect();
        assert_eq!(solid.iter().filter(|s| **s).count(), 9);
        let (keep, rep) =
            keep_set(&full, &solid, &centres, &CastellationSpec::default()).expect("keep set");
        assert_eq!(keep.iter().filter(|k| **k).count(), 9);
        assert_eq!(rep.off_region, 9);
        assert_eq!(rep.dropped_boxes.len(), 1);
        for id in 0..9usize {
            assert!(keep[id], "cell {id} of the k == 0 layer should be kept");
        }
        assert!(rep.solid == 9 && rep.pinch == 0);
        assert_eq!(rep.unrepaired_pinch, 0, "a box corner is not a pinch");
    }

    #[test]
    fn seed_chooses_the_other_pocket() {
        let (_tree, _bg, full, centres) = setup([0.0, 3.0, 0.0, 3.0, 0.0, 3.0], 1.0);
        let solid: Vec<bool> = centres.iter().map(|c| c.z > 0.5 && c.z < 2.5).collect();
        let spec = CastellationSpec {
            keep_region: KeepRegion::Seed,
            seed_point: Some([1.5, 1.5, 2.5]),
            ..CastellationSpec::default()
        };
        let (keep, rep) = keep_set(&full, &solid, &centres, &spec).expect("keep set");
        assert_eq!(keep.iter().filter(|k| **k).count(), 9);
        assert_eq!(rep.off_region, 9);
        for id in 18..27usize {
            assert!(keep[id], "cell {id} of the k == 2 layer should be kept");
        }
        for id in 0..18usize {
            assert!(!keep[id], "cell {id} should have left with its layer");
        }
        let bad = CastellationSpec {
            keep_region: KeepRegion::Seed,
            seed_point: None,
            ..CastellationSpec::default()
        };
        let err = keep_set(&full, &solid, &centres, &bad).expect_err("a seed needs a point");
        assert!(format!("{err}").contains("seed_point"), "named: {err}");
    }

    #[test]
    fn a_full_solid_is_refused() {
        let (_tree, _bg, full, centres) = setup([0.0, 3.0, 0.0, 3.0, 0.0, 3.0], 1.0);
        let solid = vec![true; centres.len()];
        let err = keep_set(&full, &solid, &centres, &CastellationSpec::default())
            .expect_err("an empty domain is a refusal");
        assert!(
            format!("{err}").contains("kept no cell"),
            "the refusal must name the empty domain: {err}"
        );
    }

    /// The cube [1,3]^3 as 12 OUTWARD-wound triangles, named corners.
    fn cube_soup() -> Vec<(u32, [Vec3; 3])> {
        let v: [[f64; 3]; 8] = [
            [1.0, 1.0, 1.0],
            [3.0, 1.0, 1.0],
            [3.0, 3.0, 1.0],
            [1.0, 3.0, 1.0],
            [1.0, 1.0, 3.0],
            [3.0, 1.0, 3.0],
            [3.0, 3.0, 3.0],
            [1.0, 3.0, 3.0],
        ];
        let t: [[usize; 3]; 12] = [
            [0, 2, 1],
            [0, 3, 2], // -z
            [4, 5, 6],
            [4, 6, 7], // +z
            [0, 1, 5],
            [0, 5, 4], // -y
            [2, 3, 7],
            [2, 7, 6], // +y
            [0, 4, 7],
            [0, 7, 3], // -x
            [1, 2, 6],
            [1, 6, 5], // +x
        ];
        t.iter()
            .map(|tr| (0u32, tr.map(|i| Vec3::new(v[i][0], v[i][1], v[i][2]))))
            .collect()
    }

    #[test]
    fn classify_leaves_removes_the_middle() {
        let (tree, bg, _full, centres) = setup([0.0, 4.0, 0.0, 4.0, 0.0, 4.0], 1.0);
        assert_eq!(centres.len(), 64);
        let surf = Surface::from_soup(cube_soup(), vec!["cube".to_string()]).expect("surface");
        let solid = classify_leaves(&tree, &bg, &surf).expect("classify");
        assert_eq!(solid.iter().filter(|s| **s).count(), 8, "the 8 centre leaves");
        for (i, c) in centres.iter().enumerate() {
            let inside = (1.0..3.0).contains(&c.x)
                && (1.0..3.0).contains(&c.y)
                && (1.0..3.0).contains(&c.z);
            assert_eq!(solid[i], inside, "leaf {i} at ({},{},{})", c.x, c.y, c.z);
        }
    }

    /// The box `[lo, hi]` as 12 OUTWARD-wound triangles, all in patch 0 -
    /// `cube_soup` generalised off `[1, 3]^3`.
    fn box_soup(lo: [f64; 3], hi: [f64; 3]) -> Vec<(u32, [Vec3; 3])> {
        let v: [[f64; 3]; 8] = [
            [lo[0], lo[1], lo[2]],
            [hi[0], lo[1], lo[2]],
            [hi[0], hi[1], lo[2]],
            [lo[0], hi[1], lo[2]],
            [lo[0], lo[1], hi[2]],
            [hi[0], lo[1], hi[2]],
            [hi[0], hi[1], hi[2]],
            [lo[0], hi[1], hi[2]],
        ];
        let t: [[usize; 3]; 12] = [
            [0, 2, 1], [0, 3, 2], // -z
            [4, 5, 6], [4, 6, 7], // +z
            [0, 1, 5], [0, 5, 4], // -y
            [2, 3, 7], [2, 7, 6], // +y
            [0, 4, 7], [0, 7, 3], // -x
            [1, 2, 6], [1, 6, 5], // +x
        ];
        t.iter()
            .map(|tr| (0u32, tr.map(|i| Vec3::new(v[i][0], v[i][1], v[i][2]))))
            .collect()
    }

    /// Every triangle reversed: the same closed surface, wound inward.
    fn inward(soup: &[(u32, [Vec3; 3])]) -> Vec<(u32, [Vec3; 3])> {
        soup.iter().map(|(p, t)| (*p, [t[0], t[2], t[1]])).collect()
    }

    /// §92.3's default thresholds.
    fn thresholds() -> crate::automesher::quality::QualityThresholds {
        crate::automesher::quality::QualityThresholds::default()
    }

    /// A sphere of radius `r` at `c`: the octahedron subdivided twice - 128
    /// triangles, every vertex ON the sphere, wound outward.
    fn sphere_soup(r: f64, c: [f64; 3]) -> Vec<(u32, [Vec3; 3])> {
        fn mid(
            a: usize,
            b: usize,
            v: &mut Vec<[f64; 3]>,
            cache: &mut HashMap<(usize, usize), usize>,
        ) -> usize {
            let key = if a < b { (a, b) } else { (b, a) };
            if let Some(&m) = cache.get(&key) {
                return m;
            }
            let m = [
                0.5 * (v[a][0] + v[b][0]),
                0.5 * (v[a][1] + v[b][1]),
                0.5 * (v[a][2] + v[b][2]),
            ];
            let n = (m[0] * m[0] + m[1] * m[1] + m[2] * m[2]).sqrt();
            v.push([m[0] / n, m[1] / n, m[2] / n]);
            let id = v.len() - 1;
            cache.insert(key, id);
            id
        }
        let mut v: Vec<[f64; 3]> = vec![
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, -1.0],
        ];
        let mut f: Vec<[usize; 3]> = vec![
            [0, 2, 4], [2, 1, 4], [1, 3, 4], [3, 0, 4], // +z half
            [2, 0, 5], [1, 2, 5], [3, 1, 5], [0, 3, 5], // -z half
        ];
        for _ in 0..2 {
            let mut cache: HashMap<(usize, usize), usize> = HashMap::new();
            let mut nf = Vec::with_capacity(f.len() * 4);
            for t in &f {
                let [a, b, cc] = *t;
                let ab = mid(a, b, &mut v, &mut cache);
                let bc = mid(b, cc, &mut v, &mut cache);
                let ca = mid(cc, a, &mut v, &mut cache);
                nf.push([a, ab, ca]);
                nf.push([ab, b, bc]);
                nf.push([ca, bc, cc]);
                nf.push([ab, bc, ca]);
            }
            f = nf;
        }
        assert_eq!(f.len(), 128, "a twice-subdivided octahedron");
        f.iter()
            .map(|t| {
                (
                    0u32,
                    t.map(|i| {
                        Vec3::new(c[0] + r * v[i][0], c[1] + r * v[i][1], c[2] + r * v[i][2])
                    }),
                )
            })
            .collect()
    }

    /// §92.2 stage 3 must not touch a mesh with nothing to remove.
    #[test]
    fn nothing_solid_reproduces_the_octree_mesh() {
        let (tree, bg, full, _) = setup([0.0, 8.0, 0.0, 8.0, 0.0, 8.0], 1.0);
        let surf = Surface::from_soup(
            box_soup([20.0; 3], [22.0; 3]),
            vec!["cube".to_string()],
        )
        .expect("surface");
        let out = castellate(
            &tree,
            &bg,
            &surf,
            &patch_names(),
            &CastellationSpec::default(),
            &thresholds(),
        )
        .expect("castellate");
        assert_eq!(out.mesh.points, full.points);
        assert_eq!(out.mesh.faces, full.faces);
        assert_eq!(out.mesh.owner, full.owner);
        assert_eq!(out.mesh.neighbour, full.neighbour);
        assert_eq!(out.mesh.patches.len(), 6, "six box patches, no wall patch");
        assert_eq!(out.wall_faces, 0);
        assert_eq!(out.removed.solid, 0);
        assert_eq!(out.removed.pinch, 0);
        assert_eq!(out.removed.off_region, 0);
        assert_eq!(out.removed.starved, 0);
        assert_eq!(out.removed.unrepaired_pinch, 0);
        assert!(out.removed.dropped_boxes.is_empty());
        assert_eq!(out.report.n_cells, 512);
    }

    /// A solid whose faces ARE cell planes: the walls sit exactly on them.
    #[test]
    fn a_cube_leaves_its_walls_on_the_cell_planes() {
        let (tree, bg, _full, _) = setup([0.0, 8.0, 0.0, 8.0, 0.0, 8.0], 1.0);
        let surf = Surface::from_soup(
            box_soup([2.0; 3], [6.0; 3]),
            vec!["cube".to_string()],
        )
        .expect("surface");
        let out = castellate(
            &tree,
            &bg,
            &surf,
            &patch_names(),
            &CastellationSpec::default(),
            &thresholds(),
        )
        .expect("castellate");
        assert_eq!(out.report.n_cells, 448, "512 - the 64 solid cells");
        assert_eq!(out.mesh.patches.len(), 7, "six box patches and one wall");
        let wall = &out.mesh.patches[6];
        assert_eq!(wall.name, "cube");
        assert_eq!(wall.type_name, "wall");
        assert_eq!(wall.size, 6 * 16, "the cube's six 4x4 faces");
        assert_eq!(out.wall_faces, 96);
        let n_int = out.mesh.neighbour.len();
        for b in wall.start..wall.start + wall.size {
            for &p in &out.mesh.faces[n_int + b] {
                let q = out.mesh.points[p as usize];
                assert!(
                    q.x == 2.0 || q.x == 6.0 || q.y == 2.0 || q.y == 6.0
                        || q.z == 2.0 || q.z == 6.0,
                    "wall point ({}, {}, {}) is off the cell planes",
                    q.x, q.y, q.z
                );
            }
        }
        assert_eq!(out.report.n_regions, 1);
        assert!(out.report.passed());
    }

    /// Centre classification keeps a volume within 5 % of the sphere's.
    #[test]
    fn a_sphere_gives_the_analytic_volume() {
        let bg = Background::from_domain(&DomainSpec {
            extent: [0.0, 8.0, 0.0, 8.0, 0.0, 8.0],
            base_size: 0.25,
            grading: [1.0; 3],
        })
        .expect("background");
        assert_eq!(bg.base_n(), [32, 32, 32]);
        let tree = Octree::uniform(bg.base_n(), 0).expect("uniform");
        let surf =
            Surface::from_soup(sphere_soup(3.0, [4.0; 3]), vec!["sphere".to_string()])
                .expect("surface");
        let out = castellate(
            &tree,
            &bg,
            &surf,
            &patch_names(),
            &CastellationSpec::default(),
            &thresholds(),
        )
        .expect("castellate");
        let mut host = crate::io::polymesh::build_host_mesh(&out.mesh).expect("host mesh");
        host.compute_geometry(&out.mesh.points, &out.mesh.faces)
            .expect("geometry");
        let total = host.check().total_volume;
        let want = 512.0 - 4.0 / 3.0 * std::f64::consts::PI * 27.0;
        let rel = (total - want).abs() / want;
        assert!(
            rel < 0.05,
            "kept volume {total} vs the analytic {want} ({:.1} % off)",
            100.0 * rel
        );
    }

    /// §92.1's tetrahedral failure: a sealed pocket under a solid is
    /// disconnected fluid, so the region rule drops it - and the dropped
    /// cells take the cavity surface's faces with them.
    #[test]
    fn a_sealed_cavity_is_dropped_and_leaves_one_region() {
        let (tree, bg, _full, _) = setup([0.0, 8.0, 0.0, 8.0, 0.0, 8.0], 1.0);
        let mut soup = box_soup([1.0; 3], [7.0; 3]);
        soup.extend(
            inward(&box_soup([3.0; 3], [5.0; 3]))
                .into_iter()
                .map(|(_, t)| (1u32, t)),
        );
        let surf = Surface::from_soup(
            soup,
            vec!["shell".to_string(), "cavity".to_string()],
        )
        .expect("surface");
        let out = castellate(
            &tree,
            &bg,
            &surf,
            &patch_names(),
            &CastellationSpec::default(),
            &thresholds(),
        )
        .expect("castellate");
        assert_eq!(out.removed.off_region, 8, "the eight cavity cells");
        assert_eq!(out.removed.dropped_boxes.len(), 1);
        assert_eq!(out.report.n_regions, 1);
        let wall_names: Vec<&str> =
            out.mesh.patches[6..].iter().map(|p| p.name.as_str()).collect();
        assert_eq!(wall_names, ["shell"], "the cavity keeps no face");
        assert!(out.report.passed());
    }

    /// §92.1's own failure, in miniature: a slab that runs PAST the domain
    /// wall - the site's ground under a building - seals the layer beneath it
    /// against the box's own patches. The pocket is a component of (92.4)
    /// with no path to the rest, so the walk drops it and the mesh that
    /// leaves the stage has one region. The tetrahedral run this mesher
    /// exists to replace kept such a pocket and diverged on it.
    #[test]
    fn a_slab_across_the_domain_seals_a_pocket_and_it_is_dropped() {
        let (tree, bg, _full, _) = setup([0.0, 8.0, 0.0, 8.0, 0.0, 8.0], 1.0);
        // Closed, and deliberately wider than the domain on x and y: the
        // solid meets the box wall instead of ending inside the fluid.
        let surf = Surface::from_soup(
            box_soup([-1.0, -1.0, 1.0], [9.0, 9.0, 2.0]),
            vec!["ground".to_string()],
        )
        .expect("surface");
        let out = castellate(
            &tree,
            &bg,
            &surf,
            &patch_names(),
            &CastellationSpec::default(),
            &thresholds(),
        )
        .expect("castellate");

        assert_eq!(out.removed.solid, 64, "the k == 1 layer is inside the slab");
        assert_eq!(out.removed.off_region, 64, "the sealed layer below it");
        assert_eq!(out.removed.dropped_boxes.len(), 1, "one pocket, reported");
        assert_eq!(
            out.mesh.patches.iter().map(|p| p.size).sum::<usize>(),
            out.mesh.owner.len() - out.mesh.neighbour.len(),
            "every boundary face is on exactly one patch"
        );
        assert_eq!(out.report.n_cells, 384, "the six layers above the slab");
        assert_eq!(out.report.n_regions, 1, "no sealed pocket survives");
        assert!(out.report.passed(), "{}", out.report.summary());

        // Every wall face is the slab's top, exactly on the cell plane z = 2:
        // the pocket's own walls left with the pocket.
        let wall: Vec<&crate::mesh::PatchInfo> =
            out.mesh.patches.iter().filter(|p| p.type_name == "wall").collect();
        assert_eq!(wall.len(), 1);
        assert_eq!(wall[0].name, "ground");
        assert_eq!(wall[0].size, 64, "one face per cell of the layer above");
        let n_int = out.mesh.neighbour.len();
        for f in 0..wall[0].size {
            for &p in &out.mesh.faces[n_int + wall[0].start + f] {
                assert_eq!(
                    out.mesh.points[p as usize].z,
                    2.0,
                    "wall face {f} is not on the slab's top plane"
                );
            }
        }
    }

    /// `carve`'s own rule (23.4): a surface patch may not take a domain
    /// patch's name.
    #[test]
    fn a_patch_name_that_collides_with_the_box_is_refused() {
        let (tree, bg, _full, _) = setup([0.0, 8.0, 0.0, 8.0, 0.0, 8.0], 1.0);
        let surf = Surface::from_soup(
            box_soup([2.0; 3], [6.0; 3]),
            vec!["xMin".to_string()],
        )
        .expect("surface");
        let err = castellate(
            &tree,
            &bg,
            &surf,
            &patch_names(),
            &CastellationSpec::default(),
            &thresholds(),
        )
        .expect_err("a colliding patch name is a refusal");
        assert!(format!("{err}").contains("xMin"), "named the collision: {err}");
    }
}
