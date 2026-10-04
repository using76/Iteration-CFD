// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.
// Provenance: see PROVENANCE.md. No GPL-licensed source was consulted.

//! The castellation keep-set - SPEC-LIT §92.10's removal walk.
//! A declared body kept as its own region, and the per-region walk, are §92.15.
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

use std::collections::{HashMap, HashSet};

use crate::error::{Error, Result};
use crate::io::polymesh::{check_patch_name, PolyMeshRaw};
use crate::surface::classify::classify_points;
use crate::surface::{SoupTri, Surface, TriIndex};
use crate::{Scalar, Vec3};

use super::octree::{Background, Octree};
use super::{BodySpec, CastellationSpec, KeepRegion};

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
    /// Pinched SEGMENTS the walk left standing at its fixed point, counted
    /// once by a final scan over the whole boundary of K. A non-zero value
    /// is a refusal, not a warning.
    pub unrepaired_pinch: usize,
    /// One `[xlo, xhi, ylo, yhi, zlo, zhi]` per dropped component, so a
    /// mesher that deletes 900 cells under a building has said where.
    pub dropped_boxes: Vec<[Scalar; 6]>,
    /// Cells each declared body lost to W1/W2/W3 on its own mask, in
    /// `castellation.bodies` order - a body that lost nothing has a row of 0.
    pub per_body: Vec<(String, usize)>,
    /// Cells the merged surface put inside a body: kept, and counted here
    /// so `solid` stays "removed as solid" exactly as before.
    pub kept_in_bodies: usize,
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

/// The region of every leaf centre, in `centres` order: `-2` removed as
/// solid, `-1` fluid, `k >= 0` the k-th declared body. `region_of_cell` on a
/// [`Castellated`] never holds `-2`.
///
/// Each body is classified on its OWN sub-surface (its patches' triangles
/// alone, `octree::band_surfaces`' pattern) by `classify_points`; a leaf
/// inside two bodies, a body naming a patch the surface lacks, and a body
/// whose shell is not closed are refused by name. Returns the region ids and
/// the body names in `bodies` order.
pub fn classify_regions(
    surf: &Surface,
    centres: &[Vec3],
    h_min: Scalar,
    bodies: &[BodySpec],
) -> Result<(Vec<i32>, Vec<String>)> {
    // (a) One sub-surface per body, checked before any classification.
    let mut subs: Vec<Surface> = Vec::with_capacity(bodies.len());
    for b in bodies {
        let mut ids: Vec<u32> = Vec::with_capacity(b.patches.len());
        for p in &b.patches {
            let Some(id) = surf.patch_names.iter().position(|n| n == p) else {
                return Err(Error::Mesh(format!(
                    "castellate: body \"{}\" names patch \"{}\", but the surface's \
                     patches are {:?} - a patch the surface lacks encloses nothing",
                    b.name, p, surf.patch_names
                )));
            };
            ids.push(id as u32);
        }
        let soup: Vec<SoupTri> = surf
            .tris
            .iter()
            .enumerate()
            .filter(|(t, _)| ids.contains(&surf.tri_patch[*t]))
            .map(|(_, tri)| {
                (
                    0u32,
                    [
                        surf.points[tri[0] as usize],
                        surf.points[tri[1] as usize],
                        surf.points[tri[2] as usize],
                    ],
                )
            })
            .collect();
        let sub = Surface::from_soup(soup, vec![b.name.clone()])?;
        let (open, nonman) = sub.edge_defects();
        if open + nonman > 0 {
            return Err(Error::Mesh(format!(
                "castellate: body \"{}\" is not a closed shell - {open} open \
                 edge(s), {nonman} non-manifold edge(s) among its patches {:?}; \
                 (92.23) cannot classify a leaf against an open body",
                b.name, b.patches
            )));
        }
        subs.push(sub);
    }
    // (b) The merged classification, the SAME call today's line makes - with
    // `bodies` empty this function returns today's classification exactly.
    let merged = classify_points(surf, centres, h_min)?;
    let mut region0: Vec<i32> =
        merged.solid.iter().map(|s| if *s { -2 } else { -1 }).collect();
    let names: Vec<String> = bodies.iter().map(|b| b.name.clone()).collect();
    // (c) Per body in order: a leaf inside two bodies is refused, a leaf
    // inside one body belongs to it even where the merged surface called it
    // solid, and a body that encloses no leaf centre is finer than the mesh.
    for (k, (b, sub)) in bodies.iter().zip(subs.iter()).enumerate() {
        let inside = classify_points(sub, centres, h_min)?.solid;
        for (i, &ins) in inside.iter().enumerate() {
            if !ins {
                continue;
            }
            if region0[i] >= 0 {
                return Err(Error::Mesh(format!(
                    "castellate: leaf centre ({:.3}, {:.3}, {:.3}) is inside body \
                     \"{}\" and inside body \"{}\" - bodies overlap",
                    centres[i].x,
                    centres[i].y,
                    centres[i].z,
                    names[region0[i] as usize],
                    b.name
                )));
            }
            region0[i] = k as i32;
        }
        if !inside.iter().any(|&x| x) {
            return Err(Error::Mesh(format!(
                "castellate: body \"{}\" holds no leaf centre - it is finer than \
                 the cells that reached it; raise the level near its patches or \
                 remove it from castellation.bodies",
                b.name
            )));
        }
    }
    Ok((region0, names))
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

/// The kept cells whose CLOSED leaf box contains the point `q`, given in
/// DOUBLED finest-lattice coordinates (`q = qa + qb` for the two ends of a
/// segment is that segment's midpoint, with no fractions).
fn star_at(
    tree: &Octree,
    cell_of: &HashMap<u64, usize>,
    keep: &[bool],
    q: [u64; 3],
) -> Vec<usize> {
    let l = tree.max_level();
    let n_fine = [
        (tree.base_n()[0] as u64) << l,
        (tree.base_n()[1] as u64) << l,
        (tree.base_n()[2] as u64) << l,
    ];
    let mut star: Vec<usize> = Vec::new();
    for s in 0..8u32 {
        let mut idx = [0u64; 3];
        let mut skip = false;
        for a in 0..3 {
            // Even: the point sits ON a lattice plane, and a cell on each
            // side of it has the plane in its closed box. Odd: strictly
            // inside a finest cell, and that one cell is both sides.
            let half = q[a] / 2;
            if q[a] % 2 == 0 {
                if s & (1 << a) != 0 {
                    if half >= n_fine[a] {
                        skip = true;
                        break;
                    }
                    idx[a] = half;
                } else {
                    if q[a] == 0 {
                        skip = true;
                        break;
                    }
                    idx[a] = half - 1;
                }
            } else {
                idx[a] = half;
            }
        }
        if skip {
            continue;
        }
        // The leaf covering that finest cell - the level the tree actually
        // resolved it at, which is what makes the test LEVEL-AWARE: at a 2:1
        // interface both sides reach it, whole edge or half edge alike.
        let Some(leaf) = tree.leaf_at(l, [idx[0] as u32, idx[1] as u32, idx[2] as u32]) else {
            continue;
        };
        let Some(&c) = cell_of.get(&leaf.pack()) else {
            continue;
        };
        if keep[c] && !star.contains(&c) {
            star.push(c);
        }
    }
    star
}

/// The star's cells split into face-connected components: two star cells are
/// joined when an internal face has BOTH of them as its two sides. Each
/// component sorted ascending, the components sorted by (len, first) - the
/// stable order the repair reads.
fn star_components(
    star: &[usize],
    cell_faces: &[Vec<usize>],
    owner: &[crate::Label],
    neighbour: &[crate::Label],
    n_int: usize,
) -> Vec<Vec<usize>> {
    let in_star: HashSet<usize> = star.iter().copied().collect();
    let mut seen: HashSet<usize> = HashSet::new();
    let mut comps: Vec<Vec<usize>> = Vec::new();
    for &root in star {
        if !seen.insert(root) {
            continue;
        }
        let mut comp = vec![root];
        let mut i = 0;
        while i < comp.len() {
            let c = comp[i];
            i += 1;
            for &f in &cell_faces[c] {
                if f >= n_int {
                    continue;
                }
                let other = if owner[f] as usize == c {
                    neighbour[f] as usize
                } else {
                    owner[f] as usize
                };
                if in_star.contains(&other) && seen.insert(other) {
                    comp.push(other);
                }
            }
        }
        comp.sort_unstable();
        comps.push(comp);
    }
    comps.sort_by_key(|c| (c.len(), c[0]));
    comps
}

/// A segment of the boundary of K whose star is not connected: (92.25)'s
/// hourglass, stated so that a 2:1 hanging node cannot hide it.
struct PinchSite {
    /// The segment's midpoint in REAL coordinates, for the message.
    mid: Vec3,
    /// The star's components, each sorted ascending, sorted by (len, first).
    comps: Vec<Vec<usize>>,
}

/// (92.25) as a segment test: walk every face on the boundary of K, probe
/// the DOUBLED-lattice midpoint of each of its segments once, and keep the
/// probes whose star - [`star_at`]'s kept cells around the segment - falls
/// into two or more face-connected components. The old rule counted the
/// carries of the whole `(point, point)` edge, which a hanging node splits
/// in two; a segment midpoint is a point, and no level hides it.
fn pinch_sites(
    full: &PolyMeshRaw,
    keep: &[bool],
    tree: &Octree,
    lattice: &[[u64; 3]],
    cell_of: &HashMap<u64, usize>,
    cell_faces: &[Vec<usize>],
    n_int: usize,
) -> Vec<PinchSite> {
    let n_faces = full.faces.len();
    let mut probed: HashSet<[u64; 3]> = HashSet::new();
    let mut sites: Vec<PinchSite> = Vec::new();
    for f in 0..n_faces {
        let own = full.owner[f] as usize;
        let on_k = if f < n_int {
            keep[own] != keep[full.neighbour[f] as usize]
        } else {
            keep[own]
        };
        if !on_k {
            continue;
        }
        let ps = &full.faces[f];
        for i in 0..ps.len() {
            let a = ps[i] as usize;
            let b = ps[(i + 1) % ps.len()] as usize;
            let q = [
                lattice[a][0] + lattice[b][0],
                lattice[a][1] + lattice[b][1],
                lattice[a][2] + lattice[b][2],
            ];
            if !probed.insert(q) {
                continue;
            }
            let star = star_at(tree, cell_of, keep, q);
            if star.len() < 2 {
                continue;
            }
            let comps = star_components(&star, cell_faces, &full.owner, &full.neighbour, n_int);
            if comps.len() >= 2 {
                let (pa, pb) = (full.points[a], full.points[b]);
                sites.push(PinchSite {
                    mid: Vec3::new(
                        0.5 * (pa.x + pb.x),
                        0.5 * (pa.y + pb.y),
                        0.5 * (pa.z + pb.z),
                    ),
                    comps,
                });
            }
        }
    }
    sites
}

/// (92.24)/(92.25): the keep mask over the cells of `full` (which must be
/// `octree::emit`'s mesh for the same `tree` - cell `i` is leaf `i`, and
/// `tree` is where the pinch rule reads the point lattice and the levels).
/// `solid[i]` is (92.23)'s answer for cell `i`, `centres[i]` its centre.
pub fn keep_set(
    full: &PolyMeshRaw,
    solid: &[bool],
    centres: &[Vec3],
    spec: &CastellationSpec,
    tree: &Octree,
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
    // The pinch rule reads the tree: its point lattice, in `emit`'s own
    // point order, and the packed key -> cell id map. A length mismatch
    // means the caller passed a mesh that is not this tree's emission.
    let lattice = super::octree::point_lattice(tree);
    if lattice.len() != full.points.len() {
        return Err(Error::Mesh(format!(
            "castellate: the tree holds {} points but the mesh has {} - the \
             tree and the mesh are not the same partition",
            lattice.len(),
            full.points.len()
        )));
    }
    let leaves = tree.leaves();
    if leaves.len() != n_cells {
        return Err(Error::Mesh(format!(
            "castellate: the tree has {} leaves but the mesh has {} cells - \
             the tree and the mesh are not the same partition",
            leaves.len(),
            n_cells
        )));
    }
    let mut cell_of: HashMap<u64, usize> = HashMap::with_capacity(leaves.len());
    for (id, key) in leaves.iter().enumerate() {
        cell_of.insert(key.pack(), id);
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

        // W1, the pinch (92.25). Every segment of the boundary of K whose
        // star is not connected - LEVEL-AWARE, so a 2:1 hanging node cannot
        // split an edge in two and hide an hourglass the way a whole-edge
        // carry count did.
        let sites = pinch_sites(full, &keep, tree, &lattice, &cell_of, &cell_faces, n_int);
        // Decide for the WHOLE pass first, then apply together, so the result
        // does not depend on the order the segments were visited in.
        let mut decisions: Vec<usize> = Vec::new();
        for site in &sites {
            let m = site
                .comps
                .iter()
                .map(|c| c.len())
                .min()
                .expect("a site has at least two components");
            // The candidates are the cells of EVERY smallest component; the
            // one to go is the least connected, ties to the LARGEST id.
            let mut drop: Option<(usize, usize)> = None;
            for comp in &site.comps {
                if comp.len() != m {
                    continue;
                }
                for &c in comp {
                    let n = nbrs_in_k(&keep, &cell_faces, &full.owner, &full.neighbour, n_int, c);
                    let better = match drop {
                        None => true,
                        Some((d, nd)) => n < nd || (n == nd && c > d),
                    };
                    if better {
                        drop = Some((c, n));
                    }
                }
            }
            let (drop, _) = drop.expect("a smallest component is non-empty");
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

    // (92.25)'s residual, counted ONCE over the fixed point - not once per
    // pass, which would report passes x sites. What the walk could not
    // repair, stage 4's polyhedra would invert, so a survivor is a refusal.
    let left = pinch_sites(full, &keep, tree, &lattice, &cell_of, &cell_faces, n_int);
    rep.unrepaired_pinch = left.len();
    if !left.is_empty() {
        let mut msg = format!(
            "castellate: {} pinched segment{} survived the walk (92.24)",
            left.len(),
            if left.len() == 1 { "" } else { "s" }
        );
        for site in left.iter().take(3) {
            let mut cells: Vec<usize> = site.comps.iter().flatten().copied().collect();
            cells.sort_unstable();
            msg.push_str(&format!(
                ": at ({:.3}, {:.3}, {:.3}) cells {cells:?} meet only along a segment",
                site.mid.x, site.mid.y, site.mid.z
            ));
        }
        msg.push_str(" - stage 4 would invert them, so refusing");
        return Err(Error::Mesh(msg));
    }
    Ok((keep, rep))
}

/// (92.24) run ONCE PER REGION: the fluid (`-1`) with `spec` as written, then
/// each body `k` with `keep_region = "largest"` and no seed - each region is
/// a mesh of its own after the split, so W1's pinch (92.25) and W2's
/// component rule (92.4) hold on every region's own boundary, the interface
/// included. A cell any region's walk removes becomes `-2`. Afterwards the
/// union of what survived is scanned once with `keep_set` as a CHECK: a pinch
/// or a disconnected component that appears only across regions is refused
/// by name, never repaired. Returns the region per cell (`-2` removed) and
/// the totals over every region.
pub fn keep_set_regions(
    full: &PolyMeshRaw,
    region0: &[i32],
    centres: &[Vec3],
    spec: &CastellationSpec,
    body_names: &[String],
    tree: &Octree,
) -> Result<(Vec<i32>, Removed)> {
    // (a) The fluid's own walk - the ONE call today's stage makes when no
    // body is declared, and the answer is this function's whole answer then.
    let solid_f: Vec<bool> = region0.iter().map(|r| *r != -1).collect();
    let (keep_f, rep_f) = keep_set(full, &solid_f, centres, spec, tree)?;
    let mut region: Vec<i32> =
        keep_f.iter().map(|k| if *k { -1 } else { -2 }).collect();
    if body_names.is_empty() {
        return Ok((region, Removed { per_body: Vec::new(), kept_in_bodies: 0, ..rep_f }));
    }
    // The fluid walk saw the bodies as solid; a body's own cells take their
    // region back here, and only the body's own walk can take it away.
    for (i, r) in region.iter_mut().enumerate() {
        if *r == -2 && region0[i] >= 0 {
            *r = region0[i];
        }
    }
    let mut rep = Removed { per_body: Vec::new(), ..rep_f };
    // (b) Each body's own walk: `largest`, no seed - a body is a mesh of its
    // own after the split and keeps its biggest component.
    for (k, name) in body_names.iter().enumerate() {
        let solid_k: Vec<bool> = region0.iter().map(|r| *r != k as i32).collect();
        let spec_k = CastellationSpec {
            keep_region: KeepRegion::Largest,
            seed_point: None,
            ..spec.clone()
        };
        let (keep_k, rep_k) = keep_set(full, &solid_k, centres, &spec_k, tree)
            .map_err(|e| Error::Mesh(format!("body \"{name}\": {e}")))?;
        let mut lost = 0;
        for (i, kflag) in keep_k.iter().enumerate() {
            if region0[i] == k as i32 && !kflag {
                region[i] = -2;
                lost += 1;
            }
        }
        rep.pinch += rep_k.pinch;
        rep.off_region += rep_k.off_region;
        rep.starved += rep_k.starved;
        rep.dropped_boxes.extend(rep_k.dropped_boxes);
        rep.per_body.push((name.clone(), lost));
    }
    // (c) The totals: `solid` stays the merged classification's removals -
    // NOT the fluid walk's count, which saw the bodies as solid - and every
    // `keep_set` above has already refused its own residual pinch.
    rep.solid = region0.iter().filter(|r| **r == -2).count();
    rep.kept_in_bodies = region.iter().filter(|r| **r >= 0).count();
    rep.unrepaired_pinch = 0;
    // (d) The union check: whatever the regions kept together must survive
    // ONE walk on its own, or the layout is a pinch no region's walk saw.
    let solid_u: Vec<bool> = region.iter().map(|r| *r == -2).collect();
    let spec_largest = CastellationSpec {
        keep_region: KeepRegion::Largest,
        seed_point: None,
        ..spec.clone()
    };
    let rep_u = match keep_set(full, &solid_u, centres, &spec_largest, tree) {
        Ok((_, r)) => r,
        Err(e) => {
            return Err(Error::Mesh(format!(
                "castellate: the fluid and the bodies together are not one \
                 repairable region - {e}"
            )));
        }
    };
    let across = rep_u.pinch + rep_u.off_region + rep_u.starved;
    if across > 0 {
        return Err(Error::Mesh(format!(
            "castellate: {across} cell(s) would be removed by (92.24) on the \
             UNION of the fluid and the bodies but by no region's own walk - a \
             body is pinched against the fluid or is not face-connected to it \
             (the fluid and {body_names:?}); such a layout is refused, not repaired"
        )));
    }
    Ok((region, rep))
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
    /// How many wall faces (92.26) each surface patch got, in the SURFACE's own
    /// patch order and including the patches that got NONE - the shape of
    /// `blockgen::CarveSummary::wall_faces`. A row of 0 is a named patch that
    /// is NOT in `mesh.patches`: the geometry it names was finer than the cells
    /// that reached it, and a caller whose inlet is that patch must see the row
    /// rather than discover the patch missing.
    pub wall_patches: Vec<(String, usize)>,
    /// The region of every cell of `mesh`, in `mesh`'s (renumbered) order:
    /// `-1` fluid, `k >= 0` the k-th entry of `body_names`. All `-1` when
    /// `castellation.bodies` is empty.
    pub region_of_cell: Vec<i32>,
    /// `castellation.bodies[].name`, in config order.
    pub body_names: Vec<String>,
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
    // The boundary file is only as good as its patch names: a name the
    // crate's own reader refuses, or one name on two of the six box patches
    // (read back as one patch), is a case that cannot be read - refused
    // before any tree is walked.
    for (i, name) in names.iter().enumerate() {
        check_patch_name(name, "domain patch")?;
        if let Some(j) = names[..i].iter().position(|q| q == name) {
            return Err(Error::Mesh(format!(
                "castellate: domain patches {j} and {i} are both named \
                 '{name}' - a boundary file with two identically named patches \
                 is read as one"
            )));
        }
    }
    // The surface's names too, ALL of them and before the walk: an illegal
    // STL solid name is illegal at every resolution, so the refusal must not
    // wait on whether the geometry it names happened to reach a cell.
    for name in &surf.patch_names {
        check_patch_name(name, "surface patch")?;
    }
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
    let (region0, body_names) =
        classify_regions(surf, &centres, h_min, &spec.bodies)?;
    let (region, removed) =
        keep_set_regions(&full, &region0, &centres, spec, &body_names, tree)?;
    let keep: Vec<bool> = region.iter().map(|r| *r != -2).collect();

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
    // The regions ride the same monotone renumbering: the map keeps order,
    // so dropping the removed cells IS the new numbering.
    let region_of_cell: Vec<i32> =
        region.iter().copied().filter(|r| *r != -2).collect();

    // The wall index (92.26), one for the whole run, hinted by the tree's
    // finest leaf edge - the length the classifier's jitter uses.
    let tidx = TriIndex::with_bvh(surf, h_min.max(Scalar::MIN_POSITIVE))?;

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

    // The count is taken from the same `walls` the boundary list is, before
    // the empty ones are skipped: a patch with no face is still a row of 0
    // in `wall_patches`, while the mesh itself carries no such patch - §92.10
    // keeps `carve`'s rule.
    let wall_patches: Vec<(String, usize)> = surf
        .patch_names
        .iter()
        .zip(walls.iter())
        .map(|(name, wf)| (name.clone(), wf.len()))
        .collect();

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
        wall_patches,
        region_of_cell,
        body_names,
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
pub(crate) mod tests {
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
        let (tree, _bg, full, centres) = setup([0.0, 3.0, 0.0, 3.0, 0.0, 3.0], 1.0);
        let solid = vec![false; centres.len()];
        let (keep, rep) =
            keep_set(&full, &solid, &centres, &CastellationSpec::default(), &tree)
                .expect("keep set");
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
        let (tree, _bg, full, centres) = setup([0.0, 2.0, 0.0, 2.0, 0.0, 1.0], 1.0);
        assert_eq!(centres.len(), 4);
        let solid = [false, true, true, false].to_vec();
        let (keep, rep) =
            keep_set(&full, &solid, &centres, &CastellationSpec::default(), &tree)
                .expect("keep set");
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
        let (tree, _bg, full, centres) = setup([0.0, 3.0, 0.0, 3.0, 0.0, 3.0], 1.0);
        let solid: Vec<bool> = centres
            .iter()
            .map(|c| c.z > 0.5 && c.z < 2.5)
            .collect();
        assert_eq!(solid.iter().filter(|s| **s).count(), 9);
        let (keep, rep) =
            keep_set(&full, &solid, &centres, &CastellationSpec::default(), &tree)
                .expect("keep set");
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
        let (tree, _bg, full, centres) = setup([0.0, 3.0, 0.0, 3.0, 0.0, 3.0], 1.0);
        let solid: Vec<bool> = centres.iter().map(|c| c.z > 0.5 && c.z < 2.5).collect();
        let spec = CastellationSpec {
            keep_region: KeepRegion::Seed,
            seed_point: Some([1.5, 1.5, 2.5]),
            ..CastellationSpec::default()
        };
        let (keep, rep) =
            keep_set(&full, &solid, &centres, &spec, &tree).expect("keep set");
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
        let err =
            keep_set(&full, &solid, &centres, &bad, &tree).expect_err("a seed needs a point");
        assert!(format!("{err}").contains("seed_point"), "named: {err}");
    }

    #[test]
    fn a_full_solid_is_refused() {
        let (tree, _bg, full, centres) = setup([0.0, 3.0, 0.0, 3.0, 0.0, 3.0], 1.0);
        let solid = vec![true; centres.len()];
        let err = keep_set(&full, &solid, &centres, &CastellationSpec::default(), &tree)
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
            .map(|tr| (0u32, tr.map(|i| Vec3::new(v[i][0] as Scalar, v[i][1] as Scalar, v[i][2] as Scalar))))
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
    pub(crate) fn box_soup(lo: [f64; 3], hi: [f64; 3]) -> Vec<(u32, [Vec3; 3])> {
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
            .map(|tr| (0u32, tr.map(|i| Vec3::new(v[i][0] as Scalar, v[i][1] as Scalar, v[i][2] as Scalar))))
            .collect()
    }

    /// Every triangle reversed: the same closed surface, wound inward.
    fn inward(soup: &[(u32, [Vec3; 3])]) -> Vec<(u32, [Vec3; 3])> {
        soup.iter().map(|(p, t)| (*p, [t[0], t[2], t[1]])).collect()
    }

    /// §92.3's default thresholds.
    pub(crate) fn thresholds() -> crate::automesher::quality::QualityThresholds {
        crate::automesher::quality::QualityThresholds::default()
    }

    /// FNV-1a 64 over points (f64 bits), faces, owner, neighbour and the
    /// patch list - equal iff the two meshes are the same mesh bit for bit.
    pub(crate) fn mesh_fingerprint(m: &PolyMeshRaw) -> u64 {
        let mut b: Vec<u8> = Vec::new();
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
        for o in &m.owner {
            b.extend_from_slice(&(*o as i64).to_le_bytes());
        }
        for n in &m.neighbour {
            b.extend_from_slice(&(*n as i64).to_le_bytes());
        }
        for p in &m.patches {
            b.extend_from_slice(p.name.as_bytes());
            b.push(0);
            b.extend_from_slice(p.type_name.as_bytes());
            b.push(0);
            b.extend_from_slice(&(p.start as u64).to_le_bytes());
            b.extend_from_slice(&(p.size as u64).to_le_bytes());
        }
        crate::automesher::identity::fnv1a64(&b)
    }

    /// A sphere of radius `r` at `c`: the octahedron subdivided twice - 128
    /// triangles, every vertex ON the sphere, wound outward.
    pub(crate) fn sphere_soup(r: f64, c: [f64; 3]) -> Vec<(u32, [Vec3; 3])> {
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
                        Vec3::new(
                            (c[0] + r * v[i][0]) as Scalar,
                            (c[1] + r * v[i][1]) as Scalar,
                            (c[2] + r * v[i][2]) as Scalar,
                        )
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
        let want = 512.0 - 4.0 / 3.0 * std::f64::consts::PI as Scalar * 27.0;
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
        assert_eq!(
            out.wall_patches,
            vec![("shell".to_string(), 216), ("cavity".to_string(), 0)],
            "every surface patch is counted, a patch that got no face included"
        );
        assert!(
            !out.mesh.patches.iter().any(|p| p.name == "cavity"),
            "a patch with no face is still not emitted"
        );
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

    /// A legal STL solid name can still be a patch name this crate's own
    /// reader refuses - `solid pump house` is a legal STL line. Refused
    /// here, with the name in it, not by the case directory after the write.
    #[test]
    fn a_surface_patch_name_with_a_space_is_refused() {
        let (tree, bg, _full, _) = setup([0.0, 8.0, 0.0, 8.0, 0.0, 8.0], 1.0);
        let surf = Surface::from_soup(
            box_soup([2.0; 3], [6.0; 3]),
            vec!["pump house".to_string()],
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
        .expect_err("a space in a patch name is a refusal");
        assert!(
            format!("{err}").contains("pump house"),
            "named the offending patch: {err}"
        );
    }

    /// The six box patches go into one boundary file: two of one name are
    /// read back as one patch, so the repeat is refused, naming the name.
    #[test]
    fn two_domain_patches_with_one_name_are_refused() {
        let (tree, bg, _full, _) = setup([0.0, 8.0, 0.0, 8.0, 0.0, 8.0], 1.0);
        let surf = Surface::from_soup(
            box_soup([20.0; 3], [22.0; 3]),
            vec!["cube".to_string()],
        )
        .expect("surface");
        let mut names = patch_names();
        names[1] = "xMin".to_string();
        let err = castellate(
            &tree,
            &bg,
            &surf,
            &names,
            &CastellationSpec::default(),
            &thresholds(),
        )
        .expect_err("two domain patches of one name are a refusal");
        assert!(format!("{err}").contains("xMin"), "named the repeat: {err}");
    }

    /// Brute force: the pinched segments of `keep` over `full`. Cell boxes are
    /// the min/max of the points of each cell's faces; the probes are the edge
    /// midpoints of every face on the boundary of K; a probe's star is every
    /// kept cell whose closed box contains it (a 1e-12 slack); the star must be
    /// connected by internal faces whose two cells are both kept and both in
    /// the star.
    fn pinched_segments(full: &PolyMeshRaw, keep: &[bool]) -> Vec<Vec3> {
        let n_faces = full.faces.len();
        let n_int = full.neighbour.len();
        let n_cells = full
            .owner
            .iter()
            .chain(full.neighbour.iter())
            .copied()
            .max()
            .map_or(0, |m| m as usize + 1);
        let mut lo = vec![[Scalar::INFINITY; 3]; n_cells];
        let mut hi = vec![[Scalar::NEG_INFINITY; 3]; n_cells];
        let mut cell_faces: Vec<Vec<usize>> = vec![Vec::new(); n_cells];
        for f in 0..n_faces {
            let own = full.owner[f] as usize;
            cell_faces[own].push(f);
            let mut cs = [own, own];
            if f < n_int {
                let nbr = full.neighbour[f] as usize;
                cell_faces[nbr].push(f);
                cs[1] = nbr;
            }
            for c in cs {
                for &p in &full.faces[f] {
                    let q = full.points[p as usize];
                    let xyz = [q.x, q.y, q.z];
                    for a in 0..3 {
                        lo[c][a] = lo[c][a].min(xyz[a]);
                        hi[c][a] = hi[c][a].max(xyz[a]);
                    }
                }
            }
        }
        const SLACK: Scalar = 1e-12;
        let mut found: Vec<Vec3> = Vec::new();
        for f in 0..n_faces {
            let own = full.owner[f] as usize;
            let on_k = if f < n_int {
                keep[own] != keep[full.neighbour[f] as usize]
            } else {
                keep[own]
            };
            if !on_k {
                continue;
            }
            let ps = &full.faces[f];
            for i in 0..ps.len() {
                let (a, b) = (ps[i] as usize, ps[(i + 1) % ps.len()] as usize);
                let (pa, pb) = (full.points[a], full.points[b]);
                let probe = [
                    0.5 * (pa.x + pb.x),
                    0.5 * (pa.y + pb.y),
                    0.5 * (pa.z + pb.z),
                ];
                let star: Vec<usize> = (0..n_cells)
                    .filter(|&c| {
                        keep[c]
                            && (0..3).all(|a| {
                                probe[a] >= lo[c][a] - SLACK && probe[a] <= hi[c][a] + SLACK
                            })
                    })
                    .collect();
                if star.len() < 2 {
                    continue;
                }
                let in_star: std::collections::HashSet<usize> = star.iter().copied().collect();
                let mut seen = vec![false; n_cells];
                let mut stack = vec![star[0]];
                seen[star[0]] = true;
                let mut n_seen = 1;
                while let Some(c) = stack.pop() {
                    for &f2 in &cell_faces[c] {
                        if f2 >= n_int {
                            continue;
                        }
                        let other = if full.owner[f2] as usize == c {
                            full.neighbour[f2] as usize
                        } else {
                            full.owner[f2] as usize
                        };
                        if in_star.contains(&other) && !seen[other] {
                            seen[other] = true;
                            n_seen += 1;
                            stack.push(other);
                        }
                    }
                }
                if n_seen < star.len() {
                    found.push(Vec3::new(probe[0], probe[1], probe[2]));
                }
            }
        }
        found
    }

    /// The reviewer's blind spot, made a test: an hourglass straddling a 2:1
    /// interface, where the coarse side carries the whole edge and the fine
    /// side carries its halves - a carry count never reaches four. The
    /// segment test sees it, the walk repairs it, and an INDEPENDENT checker
    /// confirms the castellated mesh has no pinched segment left.
    #[test]
    fn a_two_to_one_hourglass_is_seen_and_repaired() {
        let bg = Background::from_domain(&DomainSpec {
            extent: [0.0, 4.0, 0.0, 4.0, 0.0, 4.0],
            base_size: 1.0,
            grading: [1.0; 3],
        })
        .expect("background");
        let mut tree = Octree::uniform(bg.base_n(), 1).expect("uniform");
        // Level 1 where the base cell's centre has x > 2, level 0 elsewhere.
        tree.refine(|k| if (k.idx[0] as f64 + 0.5) > 2.0 { 1 } else { 0 });
        tree.balance_2to1();
        tree.check_partition().expect("partition");
        // Two boxes touching along one EDGE across the 2:1 interface at
        // x = 2: the slab sits in the coarse half, the tooth in the fine.
        let mut soup = box_soup([1.0, 1.0, 0.0], [2.0, 2.0, 1.0]);
        soup.extend(
            box_soup([2.0, 0.5, 0.5], [2.5, 1.0, 1.0])
                .into_iter()
                .map(|(_, t)| (1u32, t)),
        );
        let surf = Surface::from_soup(soup, vec!["slab".to_string(), "tooth".to_string()])
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
        assert!(
            out.removed.pinch >= 1,
            "the straddling hourglass must be seen, pinch={}",
            out.removed.pinch
        );
        assert_eq!(out.removed.unrepaired_pinch, 0, "the walk repairs it");
        assert!(out.report.passed());
        // The point of the test: an INDEPENDENT scan of the final mesh.
        let n = out
            .mesh
            .owner
            .iter()
            .chain(out.mesh.neighbour.iter())
            .copied()
            .max()
            .expect("the mesh has faces") as usize
            + 1;
        let all = vec![true; n];
        let left = pinched_segments(&out.mesh, &all);
        assert!(
            left.is_empty(),
            "{} pinched segment(s) survived in the castellated mesh, e.g. {:?}",
            left.len(),
            left.first().map(|m| (m.x, m.y, m.z))
        );
        let mut host = crate::io::polymesh::build_host_mesh(&out.mesh).expect("host mesh");
        host.compute_geometry(&out.mesh.points, &out.mesh.faces)
            .expect("geometry");
        assert_eq!(host.check().n_regions, 1, "one connected region");
    }

    /// On a band-refined tree, random keep sets used to leave a pinch in
    /// every single one while `unrepaired_pinch` reported 0 - the carry
    /// count was blind at every 2:1 interface. The segment test is not: the
    /// walk must leave the INDEPENDENT checker nothing to find.
    #[test]
    fn random_keep_sets_on_a_two_to_one_tree_leave_no_pinch() {
        let bg = Background::from_domain(&DomainSpec {
            extent: [0.0, 6.0, 0.0, 6.0, 0.0, 6.0],
            base_size: 1.0,
            grading: [1.0; 3],
        })
        .expect("background");
        let mut tree = Octree::uniform(bg.base_n(), 1).expect("uniform");
        // Level 1 where the base cell's centre has x > 3, level 0 elsewhere.
        tree.refine(|k| if (k.idx[0] as f64 + 0.5) > 3.0 { 1 } else { 0 });
        tree.balance_2to1();
        let full = crate::automesher::octree::emit(&tree, &bg, &patch_names()).expect("emit");
        let (centres, _) = leaf_centres(&tree, &bg);
        let n = centres.len();
        let mut checked = 0;
        for run in 0..60usize {
            // A 64-bit LCG, one stream per run, seeded by `0x2545F491 + run`.
            let mut s = 0x2545F491u64.wrapping_add(run as u64);
            let solid: Vec<bool> = (0..n)
                .map(|_| {
                    s = s
                        .wrapping_mul(6364136223846793005)
                        .wrapping_add(1442695040888963407);
                    // Top 53 bits, scaled to [0, 1).
                    (s >> 11) as f64 / 9007199254740992.0 < 0.30
                })
                .collect();
            let (keep, rep) = match
                keep_set(&full, &solid, &centres, &CastellationSpec::default(), &tree)
            {
                Ok(v) => v,
                Err(e) => {
                    let msg = format!("{e}");
                    assert!(
                        msg.contains("kept no cell"),
                        "run {run}: an unexpected refusal: {msg}"
                    );
                    continue;
                }
            };
            checked += 1;
            assert_eq!(rep.unrepaired_pinch, 0, "run {run}");
            let left = pinched_segments(&full, &keep);
            assert!(
                left.is_empty(),
                "run {run}: {} pinched segment(s) left, e.g. {:?}",
                left.len(),
                left.first().map(|m| (m.x, m.y, m.z))
            );
        }
        assert!(checked > 30, "only {checked} of 60 runs produced a keep set");
    }

    /// Requirement 1's pin (M6 Run 1): with `castellation.bodies` empty the
    /// castellated mesh is bit for bit what it was before the field existed -
    /// the cube on the cell planes and the 0.25 m sphere, each to its own
    /// constant, pinned before any other edit of the run.
    const PINNED_CUBE_NO_BODY: u64 = 0x1941cd5cc045b35d;
    const PINNED_SPHERE_NO_BODY: u64 = 0x4133037a9c4931e8;

    #[test]
    fn the_castellated_mesh_without_bodies_is_pinned() {
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
        let cube = mesh_fingerprint(&out.mesh);
        eprintln!("fingerprint castellated_cube_no_body = {cube:#018x}");
        assert_eq!(cube, PINNED_CUBE_NO_BODY);
        // Requirement 7: the default config is the same walk - every cell
        // fluid, no body named.
        assert!(out.region_of_cell.iter().all(|r| *r == -1));
        assert!(out.body_names.is_empty());

        let bg2 = Background::from_domain(&DomainSpec {
            extent: [0.0, 8.0, 0.0, 8.0, 0.0, 8.0],
            base_size: 0.25,
            grading: [1.0; 3],
        })
        .expect("background");
        assert_eq!(bg2.base_n(), [32, 32, 32]);
        let tree2 = Octree::uniform(bg2.base_n(), 0).expect("uniform");
        let surf2 = Surface::from_soup(
            sphere_soup(3.0, [4.0; 3]),
            vec!["sphere".to_string()],
        )
        .expect("surface");
        let out2 = castellate(
            &tree2,
            &bg2,
            &surf2,
            &patch_names(),
            &CastellationSpec::default(),
            &thresholds(),
        )
        .expect("castellate");
        let sphere = mesh_fingerprint(&out2.mesh);
        eprintln!("fingerprint castellated_sphere_no_body = {sphere:#018x}");
        assert_eq!(sphere, PINNED_SPHERE_NO_BODY);
    }

    /// Requirement 4 (M6 Run 1): the per-leaf region answers, and the three
    /// refusals, each naming the body and what it lacks.
    #[test]
    fn classify_regions_names_the_body_and_refuses_what_it_cannot() {
        let (tree, bg, _full, _) = setup([0.0, 8.0, 0.0, 8.0, 0.0, 8.0], 1.0);
        let (centres, h_min) = leaf_centres(&tree, &bg);
        let surf = Surface::from_soup(
            box_soup([2.0; 3], [6.0; 3]),
            vec!["cube".to_string()],
        )
        .expect("surface");
        let cube_body =
            || BodySpec { name: "cube".to_string(), patches: vec!["cube".to_string()] };

        // The answers: the 64 leaves inside the cube are region 0, the rest
        // fluid, none removed - the body is the exception to (92.23).
        let (region, names) = classify_regions(&surf, &centres, h_min, &[cube_body()])
            .expect("regions");
        assert_eq!(names, vec!["cube".to_string()]);
        assert_eq!(region.iter().filter(|r| **r == 0).count(), 64);
        assert_eq!(region.iter().filter(|r| **r == -1).count(), 448);
        assert!(!region.iter().any(|r| *r == -2));

        // A patch the surface lacks, named.
        let bad = BodySpec {
            name: "cube".to_string(),
            patches: vec!["cubee".to_string()],
        };
        let err = classify_regions(&surf, &centres, h_min, &[bad]).unwrap_err();
        let text = err.to_string();
        assert!(text.contains("cubee") && text.contains("cube"), "{err}");

        // An open shell: two triangles short of the cube.
        let mut open_soup = box_soup([2.0; 3], [6.0; 3]);
        open_soup.truncate(open_soup.len() - 2);
        let open = Surface::from_soup(open_soup, vec!["cube".to_string()])
            .expect("an open soup is still a surface");
        let err = classify_regions(&open, &centres, h_min, &[cube_body()]).unwrap_err();
        let text = err.to_string();
        assert!(text.contains("open edge"), "{err}");

        // Two overlapping bodies: the sphere wholly inside the cube. The
        // merged parity puts the sphere's leaves OUTSIDE - two crossings -
        // which is why the overlap check reads the per-body answers.
        let mut soup = box_soup([2.0; 3], [6.0; 3]);
        soup.extend(sphere_soup(1.0, [4.0; 3]).into_iter().map(|(_, t)| (1u32, t)));
        let two = Surface::from_soup(
            soup,
            vec!["cube".to_string(), "blob".to_string()],
        )
        .expect("surface");
        let both = vec![
            cube_body(),
            BodySpec { name: "blob".to_string(), patches: vec!["blob".to_string()] },
        ];
        let err = classify_regions(&two, &centres, h_min, &both).unwrap_err();
        let text = err.to_string();
        assert!(
            text.contains("blob") && text.contains("cube") && text.contains("overlap"),
            "{err}"
        );
    }

    /// Requirement 5 (M6 Run 1): W1 runs on the body's own mask, so a
    /// two-cell-wide diagonal pinch loses one cell OF THE BODY, counted in
    /// `per_body`, and the union check finds nothing left to refuse.
    #[test]
    fn a_body_hourglass_loses_one_cell_of_the_body() {
        let (tree, _bg, full, centres) = setup([0.0, 2.0, 0.0, 2.0, 0.0, 2.0], 1.0);
        assert_eq!(centres.len(), 8);
        // Leaf ids are (k, j, i) order: x fastest. The upper four carry the
        // body on the (x < 1) == (y < 1) diagonal; the other diagonal is
        // solid to remove, like §92.10's own hourglass.
        let region0: Vec<i32> = centres
            .iter()
            .map(|c| {
                if c.z < 1.0 {
                    -1
                } else if (c.x < 1.0) == (c.y < 1.0) {
                    0
                } else {
                    -2
                }
            })
            .collect();
        let spec = CastellationSpec::default();
        let (region, rep) = keep_set_regions(
            &full,
            &region0,
            &centres,
            &spec,
            &["blob".to_string()],
            &tree,
        )
        .expect("regions");
        assert_eq!(rep.per_body, vec![("blob".to_string(), 1)]);
        assert_eq!(rep.pinch, 1);
        assert_eq!(rep.solid, 2);
        assert_eq!(region.iter().filter(|r| **r != -2).count(), 5);
        // The surviving body cell is the diagonal cell with the LOWER id;
        // all four fluid cells are kept.
        let body: Vec<usize> = region
            .iter()
            .enumerate()
            .filter(|(_, r)| **r == 0)
            .map(|(i, _)| i)
            .collect();
        assert_eq!(body, vec![4]);
        for r in &region[..4] {
            assert_eq!(*r, -1, "all four fluid cells are kept");
        }
    }

    /// Requirement 5 (M6 Run 1): a body touching no fluid cell - nine cells
    /// sealed off by a solid layer - passes its own walk and the fluid's,
    /// and is refused only by the union check, by name.
    #[test]
    fn a_body_disconnected_from_the_fluid_is_refused() {
        let (tree, _bg, full, centres) = setup([0.0, 3.0, 0.0, 3.0, 0.0, 3.0], 1.0);
        assert_eq!(centres.len(), 27);
        let region0: Vec<i32> = centres
            .iter()
            .map(|c| {
                if c.z < 1.0 {
                    -1
                } else if c.z < 2.0 {
                    -2
                } else {
                    0
                }
            })
            .collect();
        let spec = CastellationSpec::default();
        let err = keep_set_regions(
            &full,
            &region0,
            &centres,
            &spec,
            &["blob".to_string()],
            &tree,
        )
        .unwrap_err();
        let text = err.to_string();
        assert!(
            text.contains("blob") && text.contains("not face-connected"),
            "{err}"
        );
    }

    /// Requirement 6 (M6 Run 1): a declared cube ON the cell planes is kept
    /// as its own region - nothing removed, no wall, the interface faces
    /// internal, and §92.3's gate passed on the full 512-cell block.
    #[test]
    fn a_declared_cube_is_kept_as_its_own_region() {
        let (tree, bg, _full, _) = setup([0.0, 8.0, 0.0, 8.0, 0.0, 8.0], 1.0);
        let surf = Surface::from_soup(
            box_soup([2.0; 3], [6.0; 3]),
            vec!["cube".to_string()],
        )
        .expect("surface");
        let spec = CastellationSpec {
            bodies: vec![BodySpec {
                name: "cube".to_string(),
                patches: vec!["cube".to_string()],
            }],
            ..Default::default()
        };
        let out =
            castellate(&tree, &bg, &surf, &patch_names(), &spec, &thresholds())
                .expect("castellate");
        assert_eq!(out.report.n_cells, 512);
        assert_eq!(out.removed.solid, 0);
        assert_eq!(out.wall_faces, 0);
        assert_eq!(out.wall_patches, vec![("cube".to_string(), 0)]);
        assert_eq!(out.mesh.patches.len(), 6);
        assert_eq!(out.body_names, vec!["cube".to_string()]);
        assert_eq!(out.region_of_cell.iter().filter(|r| **r == 0).count(), 64);
        assert_eq!(out.report.n_regions, 1);
        assert!(out.report.passed());
        assert_eq!(
            out.mesh.neighbour.len(),
            3 * 8 * 8 * 7,
            "nothing was removed, so the interface faces are among the full \
             block's internal faces"
        );
        // Every body cell's centre - the mean of its face points - lies
        // inside [2, 6]^3.
        let n_int = out.mesh.neighbour.len();
        let n_cells = out.report.n_cells;
        let mut sum = vec![[0.0f64; 3]; n_cells];
        let mut cnt = vec![0.0f64; n_cells];
        for (f, ps) in out.mesh.faces.iter().enumerate() {
            let mut cells = vec![out.mesh.owner[f] as usize];
            if f < n_int {
                cells.push(out.mesh.neighbour[f] as usize);
            }
            for c in cells {
                for &p in ps {
                    let q = out.mesh.points[p as usize];
                    sum[c][0] += q.x as f64;
                    sum[c][1] += q.y as f64;
                    sum[c][2] += q.z as f64;
                    cnt[c] += 1.0;
                }
            }
        }
        for (c, r) in out.region_of_cell.iter().enumerate() {
            if *r != 0 {
                continue;
            }
            let ctr = [sum[c][0] / cnt[c], sum[c][1] / cnt[c], sum[c][2] / cnt[c]];
            assert!(
                ctr[0] >= 2.0
                    && ctr[0] <= 6.0
                    && ctr[1] >= 2.0
                    && ctr[1] <= 6.0
                    && ctr[2] >= 2.0
                    && ctr[2] <= 6.0,
                "body cell {c} has centre {ctr:?}, outside [2,6]^3"
            );
        }
    }
}
