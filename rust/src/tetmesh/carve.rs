// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.
// Provenance: see PROVENANCE.md. No GPL-licensed source was consulted.

//! The carve of SPEC-LIT §116.12 (TET-08): the tetrahedra of a conforming
//! boundary recovery are split into regions by a flood fill across every
//! face that is NOT in the split surface, and one fluid region is kept.
//!
//! 1. The domain box enters as points. The convex hull of the
//!    triangulation is the box, so its hull faces tile the box planes with
//!    no recovery; a hull face is the domain boundary.
//! 2. The regions are the live tetrahedra, finite and hull, joined across
//!    faces that are not subtriangles of the split surface (§116.11). All
//!    hull tetrahedra share one region, because a face through the
//!    infinite vertex is never a subtriangle.
//! 3. Each subtriangle `(a, b, c)`, wound as its input triangle, faces a
//!    tetrahedron with its front iff the tetrahedron's apex `d` has
//!    `orient3d(a, b, c, d) < 0` (116.16a): the normal points out of the
//!    solid, into the fluid, as an STL shell's does. A region is fluid if
//!    every subtriangle that bounds it faces it with its front, solid if
//!    every one faces it with its back, and mixed otherwise.
//! 4. The kept region is the one holding the hull tetrahedra
//!    ([`CarveStart::Hull`], `keep_region = "largest"`, the external flow)
//!    or the one holding the tetrahedron the walk locates the seed in
//!    ([`CarveStart::Seed`]). A seed outside the hull, a kept region on the
//!    solid side (a seed on the wrong side), a mixed kept region and an
//!    empty one are refusals by name.
//! 5. The kept region's boundary faces are wound outward from it, and its
//!    volume closes by the divergence theorem (116.16b). Every other
//!    region is reported with its side, size, volume and bounding box: the
//!    fluid ones are §92.4's R4 pockets, dropped.
//!
//! Every coordinate is `[f64; 3]` in both builds (§116.2), and every side
//! decision is the exact `orient3d` of §116.4.
//!
//! Provenance: ORIGINAL. A flood fill over the tetrahedra's face
//! adjacency; the volume identity is the divergence theorem of §116.9's
//! (116.13), and the region rule is §92.4's R4 (pocket removal) carried to
//! tetrahedra; the choices are derived in SPEC-LIT §116.12. No GPL-licensed
//! source was consulted.

use super::delaunay::{Delaunay, INF, NONE};
use super::predicates::orient3d;
use super::recover::FaceRecovery;
use super::surfprep::TetSurface;
use super::Point;
use crate::error::{Error, Result};
use std::collections::{HashMap, VecDeque};

fn sub(a: Point, b: Point) -> Point {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn dot(a: Point, b: Point) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn cross(a: Point, b: Point) -> Point {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn sorted3(mut f: [u32; 3]) -> [u32; 3] {
    f.sort_unstable();
    f
}

/// The face opposite slot `i`, wound so its right-hand normal points out of
/// the tetrahedron: §116.4's table.
fn face_out(v: [u32; 4], i: usize) -> [u32; 3] {
    match i {
        0 => [v[1], v[3], v[2]],
        1 => [v[0], v[2], v[3]],
        2 => [v[0], v[3], v[1]],
        3 => [v[0], v[1], v[2]],
        _ => unreachable!(),
    }
}

/// Where the flood starts (SPEC-LIT §116.12).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum CarveStart {
    /// The region holding the domain hull's tetrahedra:
    /// `castellation.keep_region = "largest"`, the external flow.
    Hull,
    /// The region holding the tetrahedron that contains the point:
    /// `castellation.keep_region = "seed"` with its `seed_point`.
    Seed(Point),
}

/// Which side of the surface a region lies on (116.16a).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Side {
    /// Every subtriangle that bounds the region faces it with its front
    /// (also a region that no subtriangle bounds).
    Fluid,
    /// Every subtriangle that bounds the region faces it with its back.
    Solid,
    /// Some face it with their front and some with their back.
    Mixed,
}

/// One region of the flood fill (SPEC-LIT §116.12).
#[derive(Clone, Debug, PartialEq)]
pub struct CarveRegion {
    pub side: Side,
    /// Whether this is the kept region.
    pub kept: bool,
    /// Whether the region holds the hull tetrahedra.
    pub touches_hull: bool,
    /// The region's finite tetrahedra.
    pub n_tets: usize,
    /// (116.2) summed over the region's finite tetrahedra in slot order.
    pub volume: f64,
    /// The bounding box of the region's finite tetrahedra's vertices; with
    /// no finite tetrahedron, `lo` is `+inf` and `hi` is `-inf`.
    pub lo: Point,
    pub hi: Point,
}

/// What the carve kept (SPEC-LIT §116.12).
#[derive(Clone, Debug)]
pub struct Carved {
    /// The kept tetrahedra's slots in the recovery's `delaunay`, ascending.
    pub kept_slots: Vec<u32>,
    /// Beside `kept_slots`: the tetrahedra in stored order, (116.2) > 0.
    pub tets: Vec<[u32; 4]>,
    /// The kept region's boundary faces, each wound so its right-hand normal
    /// points out of the region, in (slot, face) order.
    pub faces: Vec<[u32; 3]>,
    /// Beside `faces`: the subtriangle index into the recovery's `subtris`,
    /// or `None` for a hull face (the domain boundary).
    pub face_sub: Vec<Option<u32>>,
    /// Every region, numbered in the order of its lowest slot.
    pub regions: Vec<CarveRegion>,
    /// The index of the kept region in `regions`.
    pub kept_region: usize,
    /// The kept volume: `regions[kept_region].volume`.
    pub volume: f64,
}

impl Carved {
    /// (116.16b): the kept volume by the divergence theorem over the kept
    /// region's outward-wound boundary faces, about the first face's first
    /// point.
    pub fn boundary_volume(&self, points: &[Point]) -> f64 {
        let Some(f0) = self.faces.first() else {
            return 0.0;
        };
        let o = points[f0[0] as usize];
        let mut s = 0.0;
        for f in &self.faces {
            let x0 = sub(points[f[0] as usize], o);
            let x1 = sub(points[f[1] as usize], o);
            let x2 = sub(points[f[2] as usize], o);
            s += dot(x0, cross(x1, x2));
        }
        s / 6.0
    }

    /// The kept region's boundary faces per patch of `surf` (a subtriangle
    /// counts on its parent triangle's patch), and the hull faces.
    pub fn patch_face_counts(&self, surf: &TetSurface, fr: &FaceRecovery) -> (Vec<usize>, usize) {
        let mut per = vec![0usize; surf.patch_names.len()];
        let mut hull = 0usize;
        for s in &self.face_sub {
            match s {
                Some(k) => {
                    let t = fr.sub_parent[*k as usize] as usize;
                    per[surf.tri_patch[t] as usize] += 1;
                }
                None => hull += 1,
            }
        }
        (per, hull)
    }

    /// The dropped fluid regions with finite tetrahedra: §92.4's R4 pockets.
    pub fn dropped(&self) -> Vec<&CarveRegion> {
        self.regions
            .iter()
            .filter(|r| r.side == Side::Fluid && !r.kept && r.n_tets > 0)
            .collect()
    }
}

/// The name of subtriangle `k` in a refusal.
fn name_sub(surf: &TetSurface, fr: &FaceRecovery, k: u32) -> String {
    let s = fr.subtris[k as usize];
    let t = fr.sub_parent[k as usize];
    let patch = surf
        .tri_patch
        .get(t as usize)
        .and_then(|&p| surf.patch_names.get(p as usize))
        .map(|s| s.as_str())
        .unwrap_or("?");
    format!(
        "subtriangle {k} [{}, {}, {}] of input triangle {t} (patch \"{patch}\")",
        s[0], s[1], s[2]
    )
}

/// SPEC-LIT §116.12: carves the triangulation of a face recovery `fr` of
/// `surf` into regions by a flood fill across every face that is not a
/// subtriangle, and keeps the fluid region `start` names. Refuses by name a
/// subtriangle that is not a face of exactly two tetrahedra
/// (`carve/input`), a seed outside the domain hull or on the solid side of
/// the surface (`carve/seed`), a hull region on the solid side
/// (`carve/hull`), a kept region on both sides (`carve/side`) and a kept
/// region with no finite tetrahedron (`carve/empty`).
pub fn carve(surf: &TetSurface, fr: &FaceRecovery, start: &CarveStart) -> Result<Carved> {
    let d: &Delaunay = &fr.delaunay;
    let pts = d.points();
    if surf.tris.len() != fr.tris.len()
        || surf.tri_patch.len() != surf.tris.len()
        || fr.subtris.len() != fr.sub_parent.len()
    {
        return Err(Error::Refused(format!(
            "carve/input: the recovery has {} input triangles and {} subtriangles with {} \
             parents, the surface {} triangles with {} patch ids",
            fr.tris.len(),
            fr.subtris.len(),
            fr.sub_parent.len(),
            surf.tris.len(),
            surf.tri_patch.len()
        )));
    }
    if let Some(t) = surf.tri_patch.iter().position(|&p| p as usize >= surf.patch_names.len()) {
        return Err(Error::Refused(format!(
            "carve/input: triangle {t} has patch {} of {} patches",
            surf.tri_patch[t],
            surf.patch_names.len()
        )));
    }
    let mut sub_of: HashMap<[u32; 3], u32> = HashMap::with_capacity(fr.subtris.len());
    for (k, s) in fr.subtris.iter().enumerate() {
        if let Some(j) = sub_of.insert(sorted3(*s), k as u32) {
            return Err(Error::Refused(format!(
                "carve/input: {} repeats the vertices of subtriangle {j}",
                name_sub(surf, fr, k as u32)
            )));
        }
    }
    let n_slots = d.slots();
    let face_sub = |t: u32, i: usize| -> Option<u32> {
        let f = face_out(d.tet(t), i);
        if f.contains(&INF) {
            None
        } else {
            sub_of.get(&sorted3(f)).copied()
        }
    };

    // The regions: a breadth-first flood over the live slots in slot order,
    // across faces that are not subtriangles.
    let mut region = vec![NONE; n_slots];
    let mut n_regions = 0u32;
    let mut queue: VecDeque<u32> = VecDeque::new();
    for s0 in 0..n_slots as u32 {
        if !d.is_live(s0) || region[s0 as usize] != NONE {
            continue;
        }
        region[s0 as usize] = n_regions;
        queue.push_back(s0);
        while let Some(t) = queue.pop_front() {
            for (i, &n) in d.neighbours(t).iter().enumerate() {
                if n == NONE || region[n as usize] != NONE || face_sub(t, i).is_some() {
                    continue;
                }
                region[n as usize] = n_regions;
                queue.push_back(n);
            }
        }
        n_regions += 1;
    }
    let nr = n_regions as usize;

    // Each subtriangle's two tetrahedra, and which one its front faces
    // (116.16a).
    let mut sides: Vec<Vec<(u32, usize)>> = vec![Vec::new(); fr.subtris.len()];
    for t in 0..n_slots as u32 {
        if !d.is_live(t) {
            continue;
        }
        for i in 0..4 {
            if let Some(k) = face_sub(t, i) {
                sides[k as usize].push((t, i));
            }
        }
    }
    let mut first_front = vec![NONE; nr];
    let mut first_back = vec![NONE; nr];
    for (k, on) in sides.iter().enumerate() {
        if on.len() != 2 {
            return Err(Error::Refused(format!(
                "carve/input: {} is a face of {} tetrahedra, not 2",
                name_sub(surf, fr, k as u32),
                on.len()
            )));
        }
        // A face through INF is never a subtriangle, so a subtriangle's
        // apex in a hull tetrahedron is the slot-3 INF and the finite side
        // decides.
        let (fin, other) = if d.is_hull(on[0].0) { (on[1], on[0]) } else { (on[0], on[1]) };
        let s = fr.subtris[k];
        let apex = d.tet(fin.0)[fin.1];
        let o = orient3d(
            &pts[s[0] as usize],
            &pts[s[1] as usize],
            &pts[s[2] as usize],
            &pts[apex as usize],
        );
        let (front, back) = if o < 0.0 {
            (fin.0, other.0)
        } else if o > 0.0 {
            (other.0, fin.0)
        } else {
            return Err(Error::Refused(format!(
                "carve/input: {} is coplanar with the apex {apex} of tetrahedron {}",
                name_sub(surf, fr, k as u32),
                fin.0
            )));
        };
        let (rf, rb) = (region[front as usize] as usize, region[back as usize] as usize);
        if first_front[rf] == NONE {
            first_front[rf] = k as u32;
        }
        if first_back[rb] == NONE {
            first_back[rb] = k as u32;
        }
    }

    // The regions' sides, sizes, volumes and boxes.
    let mut regions: Vec<CarveRegion> = (0..nr)
        .map(|r| CarveRegion {
            side: match (first_front[r] != NONE, first_back[r] != NONE) {
                (_, false) => Side::Fluid,
                (false, true) => Side::Solid,
                (true, true) => Side::Mixed,
            },
            kept: false,
            touches_hull: false,
            n_tets: 0,
            volume: 0.0,
            lo: [f64::INFINITY; 3],
            hi: [f64::NEG_INFINITY; 3],
        })
        .collect();
    let mut six_v = vec![0.0f64; nr];
    let mut hull_region = NONE;
    let mut first_finite = NONE;
    for t in 0..n_slots as u32 {
        if !d.is_live(t) {
            continue;
        }
        let r = region[t as usize] as usize;
        if d.is_hull(t) {
            regions[r].touches_hull = true;
            if hull_region == NONE {
                hull_region = r as u32;
            }
            continue;
        }
        if first_finite == NONE {
            first_finite = t;
        }
        let v = d.tet(t);
        let p = [pts[v[0] as usize], pts[v[1] as usize], pts[v[2] as usize], pts[v[3] as usize]];
        six_v[r] += orient3d(&p[0], &p[1], &p[2], &p[3]);
        let g = &mut regions[r];
        g.n_tets += 1;
        for x in &p {
            for ((lo, hi), &xa) in g.lo.iter_mut().zip(g.hi.iter_mut()).zip(x) {
                *lo = lo.min(xa);
                *hi = hi.max(xa);
            }
        }
    }
    for (g, s) in regions.iter_mut().zip(&six_v) {
        g.volume = s / 6.0;
    }
    if first_finite == NONE || hull_region == NONE {
        return Err(Error::Refused(
            "carve/empty: the triangulation has no finite or no hull tetrahedron".to_string(),
        ));
    }

    // The kept region.
    let kept = match *start {
        CarveStart::Hull => {
            let r = hull_region as usize;
            match regions[r].side {
                Side::Solid => {
                    return Err(Error::Refused(format!(
                        "carve/hull: the region at the domain hull lies behind {}, on the \
                         solid side of the surface, whose normals point into the fluid; a \
                         shell wound inward holds its fluid inside and is kept from a seed",
                        name_sub(surf, fr, first_back[r])
                    )))
                }
                Side::Mixed => {
                    return Err(Error::Refused(format!(
                        "carve/side: the region at the domain hull lies on both sides of the \
                         surface, in front of {} and behind {}: the surface does not separate \
                         the domain (open, or wound inconsistently)",
                        name_sub(surf, fr, first_front[r]),
                        name_sub(surf, fr, first_back[r])
                    )))
                }
                Side::Fluid => {}
            }
            if regions[r].n_tets == 0 {
                return Err(Error::Refused(
                    "carve/empty: the region at the domain hull holds no finite tetrahedron: \
                     every hull face is a surface face, so no domain box surrounds the \
                     surface; keep a region by a seed"
                        .to_string(),
                ));
            }
            r
        }
        CarveStart::Seed(p) => {
            if !(p[0].is_finite() && p[1].is_finite() && p[2].is_finite()) {
                return Err(Error::Refused(format!(
                    "carve/seed: the seed point ({}, {}, {}) is not finite",
                    p[0], p[1], p[2]
                )));
            }
            let at = d.locate(&p, first_finite)?.tet;
            if d.is_hull(at) {
                return Err(Error::Refused(format!(
                    "carve/seed: the seed point ({}, {}, {}) lies outside the domain hull (the \
                     convex hull of the {} points); it must be inside the domain box",
                    p[0],
                    p[1],
                    p[2],
                    pts.len()
                )));
            }
            let r = region[at as usize] as usize;
            match regions[r].side {
                Side::Solid => {
                    return Err(Error::Refused(format!(
                        "carve/seed: the seed point ({}, {}, {}) is on the wrong side: its \
                         region ({} tetrahedra, volume {:e}) lies behind {}, on the solid side \
                         of the surface, whose normals point into the fluid",
                        p[0],
                        p[1],
                        p[2],
                        regions[r].n_tets,
                        regions[r].volume,
                        name_sub(surf, fr, first_back[r])
                    )))
                }
                Side::Mixed => {
                    return Err(Error::Refused(format!(
                        "carve/side: the seed point ({}, {}, {})'s region lies on both sides of \
                         the surface, in front of {} and behind {}: the surface does not \
                         separate the domain (open, or wound inconsistently)",
                        p[0],
                        p[1],
                        p[2],
                        name_sub(surf, fr, first_front[r]),
                        name_sub(surf, fr, first_back[r])
                    )))
                }
                Side::Fluid => {}
            }
            r
        }
    };
    regions[kept].kept = true;

    // The kept tetrahedra and the region's outward-wound boundary.
    let mut kept_slots = Vec::with_capacity(regions[kept].n_tets);
    let mut tets = Vec::with_capacity(regions[kept].n_tets);
    let mut faces = Vec::new();
    let mut face_subs = Vec::new();
    for t in 0..n_slots as u32 {
        if !d.is_live(t) || d.is_hull(t) || region[t as usize] as usize != kept {
            continue;
        }
        kept_slots.push(t);
        tets.push(d.tet(t));
        for (i, &n) in d.neighbours(t).iter().enumerate() {
            let k = face_sub(t, i);
            // A fluid region never meets itself across a subtriangle, so a
            // boundary face is a subtriangle or a face onto a hull tetrahedron.
            if k.is_some() || n == NONE || d.is_hull(n) {
                faces.push(face_out(d.tet(t), i));
                face_subs.push(k);
            }
        }
    }
    let volume = regions[kept].volume;
    Ok(Carved {
        kept_slots,
        tets,
        faces,
        face_sub: face_subs,
        regions,
        kept_region: kept,
        volume,
    })
}

#[cfg(test)]
mod tests {
    use super::super::recover::recover_faces;
    use super::super::remesh::{remesh, RemeshOptions};
    use super::super::surfprep::prepare;
    use super::super::TetRecoverSpec;
    use super::*;
    use std::collections::HashSet;

    fn norm(a: Point) -> f64 {
        dot(a, a).sqrt()
    }

    /// The level-`level` icosphere of radius 1 about the origin, one patch
    /// per hemisphere when `two` (by the sign of the triangle's centroid z),
    /// wound inward when `inward`.
    fn ico(level: u32, two: bool, inward: bool) -> TetSurface {
        let t = (1.0 + 5.0f64.sqrt()) / 2.0;
        let v: [Point; 12] = [
            [-1.0, t, 0.0],
            [1.0, t, 0.0],
            [-1.0, -t, 0.0],
            [1.0, -t, 0.0],
            [0.0, -1.0, t],
            [0.0, 1.0, t],
            [0.0, -1.0, -t],
            [0.0, 1.0, -t],
            [t, 0.0, -1.0],
            [t, 0.0, 1.0],
            [-t, 0.0, -1.0],
            [-t, 0.0, 1.0],
        ];
        let f: [[u32; 3]; 20] = [
            [0, 11, 5],
            [0, 5, 1],
            [0, 1, 7],
            [0, 7, 10],
            [0, 10, 11],
            [1, 5, 9],
            [5, 11, 4],
            [11, 10, 2],
            [10, 7, 6],
            [7, 1, 8],
            [3, 9, 4],
            [3, 4, 2],
            [3, 2, 6],
            [3, 6, 8],
            [3, 8, 9],
            [4, 9, 5],
            [2, 4, 11],
            [6, 2, 10],
            [8, 6, 7],
            [9, 8, 1],
        ];
        let unit = |p: Point| {
            let n = norm(p);
            [p[0] / n, p[1] / n, p[2] / n]
        };
        let mut pts: Vec<Point> = v.iter().map(|&p| unit(p)).collect();
        let mut tris: Vec<[u32; 3]> = f.to_vec();
        for _ in 0..level {
            let mut cache: HashMap<(u32, u32), u32> = HashMap::new();
            let mut next = Vec::with_capacity(tris.len() * 4);
            for [a, b, c] in tris {
                let mut mid = |x: u32, y: u32| -> u32 {
                    *cache.entry((x.min(y), x.max(y))).or_insert_with(|| {
                        let (p, q) = (pts[x as usize], pts[y as usize]);
                        pts.push(unit([
                            0.5 * (p[0] + q[0]),
                            0.5 * (p[1] + q[1]),
                            0.5 * (p[2] + q[2]),
                        ]));
                        pts.len() as u32 - 1
                    })
                };
                let (ab, bc, ca) = (mid(a, b), mid(b, c), mid(c, a));
                next.extend([[a, ab, ca], [b, bc, ab], [c, ca, bc], [ab, bc, ca]]);
            }
            tris = next;
        }
        if inward {
            for t in &mut tris {
                t.swap(1, 2);
            }
        }
        let tri_patch: Vec<u32> = tris
            .iter()
            .map(|t| {
                let z = pts[t[0] as usize][2] + pts[t[1] as usize][2] + pts[t[2] as usize][2];
                u32::from(two && z < 0.0)
            })
            .collect();
        let patch_names = if two {
            vec!["north".to_string(), "south".to_string()]
        } else {
            vec!["sphere".to_string()]
        };
        TetSurface { points: pts, tris, tri_patch, patch_names }
    }

    /// The axis-aligned cube `[lo, hi]^3`, two triangles and one patch per
    /// face (`prefix` + xmin ... zmax), appended to `s`; wound outward, or
    /// inward when `inward`.
    fn add_cube(s: &mut TetSurface, lo: f64, hi: f64, prefix: &str, inward: bool) {
        let base = s.points.len() as u32;
        for k in 0..8u32 {
            let c = |b: u32| if (k >> b) & 1 == 1 { hi } else { lo };
            s.points.push([c(0), c(1), c(2)]);
        }
        let faces: [[u32; 4]; 6] = [
            [0, 4, 6, 2],
            [1, 3, 7, 5],
            [0, 1, 5, 4],
            [2, 6, 7, 3],
            [0, 2, 3, 1],
            [4, 5, 7, 6],
        ];
        let p0 = s.patch_names.len() as u32;
        for (j, f) in faces.iter().enumerate() {
            for t in [[f[0], f[1], f[2]], [f[0], f[2], f[3]]] {
                let mut t = [base + t[0], base + t[1], base + t[2]];
                if inward {
                    t.swap(1, 2);
                }
                s.tris.push(t);
                s.tri_patch.push(p0 + j as u32);
            }
        }
        for n in ["xmin", "xmax", "ymin", "ymax", "zmin", "zmax"] {
            s.patch_names.push(format!("{prefix}{n}"));
        }
    }

    /// The domain box `[lo, hi]^3` as its 8 corner points, on no triangle.
    fn add_box(s: &mut TetSurface, lo: f64, hi: f64) {
        for k in 0..8u32 {
            let c = |b: u32| if (k >> b) & 1 == 1 { hi } else { lo };
            s.points.push([c(0), c(1), c(2)]);
        }
    }

    /// TET-06b's 30 degree wedge: the prism over (0,0), (1,0),
    /// (cos 30, sin 30) from z = 0 to z = 1.
    fn wedge() -> TetSurface {
        let c = 30f64.to_radians().cos();
        let s = 30f64.to_radians().sin();
        TetSurface {
            points: vec![
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [c, s, 0.0],
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 1.0],
                [c, s, 1.0],
            ],
            tris: vec![
                [0, 2, 1],
                [3, 4, 5],
                [0, 1, 4],
                [0, 4, 3],
                [1, 2, 5],
                [1, 5, 4],
                [2, 0, 3],
                [2, 3, 5],
            ],
            tri_patch: vec![0; 8],
            patch_names: vec!["wedge".to_string()],
        }
    }

    /// (116.13a): the enclosed volume of the split surface's subtriangles
    /// whose parent is in `on`, about the first one's first point.
    fn enclosed(fr: &FaceRecovery, on: &dyn Fn(u32) -> bool) -> f64 {
        let subs: Vec<[u32; 3]> = fr
            .subtris
            .iter()
            .zip(&fr.sub_parent)
            .filter(|(_, &p)| on(p))
            .map(|(s, _)| *s)
            .collect();
        let o = fr.points[subs[0][0] as usize];
        let mut v = 0.0;
        for s in &subs {
            let x0 = sub(fr.points[s[0] as usize], o);
            let x1 = sub(fr.points[s[1] as usize], o);
            let x2 = sub(fr.points[s[2] as usize], o);
            v += dot(x0, cross(x1, x2));
        }
        v / 6.0
    }

    fn rel(a: f64, b: f64) -> f64 {
        (a - b).abs() / b.abs()
    }

    fn spec() -> TetRecoverSpec {
        TetRecoverSpec::default()
    }

    /// The volume gate (116.16) and (116.16b) on one body in a box, with
    /// the body's triangles `0..n_body`: the kept volume is the box minus
    /// the split body's enclosed volume, relative 1e-12; the boundary
    /// closes on it; the kept and solid regions fill the box; a fluid seed
    /// keeps the same tetrahedra bit for bit. Returns the carve.
    fn volume_gate(
        name: &str,
        s: &TetSurface,
        fr: &FaceRecovery,
        v_box: f64,
        n_body: u32,
        fluid_seed: Point,
    ) -> Carved {
        let c = carve(s, fr, &CarveStart::Hull).unwrap();
        let v_body = enclosed(fr, &|p| p < n_body);
        let target = v_box - v_body;
        let e1 = rel(c.volume, target);
        let e2 = rel(c.boundary_volume(&fr.points), c.volume);
        let solid: Vec<&CarveRegion> =
            c.regions.iter().filter(|r| r.side == Side::Solid).collect();
        assert_eq!(solid.len(), 1, "{name}: one solid region");
        let e3 = rel(solid[0].volume, v_body);
        let total: f64 = c.regions.iter().map(|r| r.volume).sum();
        let e4 = rel(total, v_box);
        println!(
            "carve {name}: {} points, {} subtriangles, {} regions, kept {} tets, V {:.15} \
             target {:.15}, rel {e1:.2e}, (116.16b) {e2:.2e}, solid {e3:.2e}, box {e4:.2e}, \
             {} boundary faces",
            fr.points.len(),
            fr.subtris.len(),
            c.regions.len(),
            c.tets.len(),
            c.volume,
            target,
            c.faces.len()
        );
        assert!(e1 <= 1e-12, "{name}: (116.16) rel {e1:e}");
        assert!(e2 <= 1e-12, "{name}: (116.16b) rel {e2:e}");
        assert!(e3 <= 1e-12, "{name}: solid rel {e3:e}");
        assert!(e4 <= 1e-12, "{name}: box rel {e4:e}");
        assert!(c.regions[c.kept_region].touches_hull);
        assert!(c.dropped().is_empty(), "{name}: no pocket");
        let cs = carve(s, fr, &CarveStart::Seed(fluid_seed)).unwrap();
        assert_eq!(cs.kept_slots, c.kept_slots, "{name}: the seed keeps the hull region");
        assert_eq!(cs.faces, c.faces);
        assert_eq!(cs.volume.to_bits(), c.volume.to_bits());
        c
    }

    #[test]
    fn volume_closes() {
        // The level-3 icosphere in [-2, 2]^3.
        let mut s = ico(3, false, false);
        let n_body = s.tris.len() as u32;
        add_box(&mut s, -2.0, 2.0);
        let fr = recover_faces(&s, &[], &spec()).unwrap();
        volume_gate("icosphere", &s, &fr, 64.0, n_body, [-1.9, -1.9, -1.9]);

        // The same sphere remeshed at h = 0.15.
        let prep = prepare(&ico(3, false, false), 60.0).unwrap();
        let r = remesh(&prep, &|_: Point| 0.15, &RemeshOptions::default()).unwrap();
        let mut s = r.surface.clone();
        let n_body = s.tris.len() as u32;
        add_box(&mut s, -2.0, 2.0);
        let fr = recover_faces(&s, &r.feature_edges, &spec()).unwrap();
        volume_gate("remeshed sphere", &s, &fr, 64.0, n_body, [1.9, 0.0, 0.0]);

        // The 30 degree wedge remeshed at h = 0.1 in [-1, 2]^3: Steiner
        // points on its slanted faces, none on the hull.
        let prep = prepare(&wedge(), 60.0).unwrap();
        let r = remesh(&prep, &|_: Point| 0.1, &RemeshOptions::default()).unwrap();
        let mut s = r.surface.clone();
        let n_body = s.tris.len() as u32;
        add_box(&mut s, -1.0, 2.0);
        let fr = recover_faces(&s, &r.feature_edges, &spec()).unwrap();
        assert!(fr.steiner() > 0, "the wedge needs Steiner points");
        volume_gate("wedge", &s, &fr, 27.0, n_body, [-0.5, 1.5, 0.5]);

        // A cube body [-0.5, 0.5]^3 in [-2, 2]^3.
        let mut s = TetSurface {
            points: Vec::new(),
            tris: Vec::new(),
            tri_patch: Vec::new(),
            patch_names: Vec::new(),
        };
        add_cube(&mut s, -0.5, 0.5, "", false);
        add_box(&mut s, -2.0, 2.0);
        let fr = recover_faces(&s, &[], &spec()).unwrap();
        volume_gate("cube", &s, &fr, 64.0, 12, [0.0, 0.0, 1.0]);
    }

    #[test]
    fn seed_refusal() {
        let mut s = ico(3, false, false);
        add_box(&mut s, -2.0, 2.0);
        let fr = recover_faces(&s, &[], &spec()).unwrap();
        let err = format!("{}", carve(&s, &fr, &CarveStart::Seed([0.0, 0.0, 0.0])).unwrap_err());
        println!("carve seed inside: {err}");
        assert!(err.contains("carve/seed") && err.contains("wrong side"), "{err}");
        assert!(err.contains("patch \"sphere\""), "{err}");
        let err = format!("{}", carve(&s, &fr, &CarveStart::Seed([5.0, 0.0, 0.0])).unwrap_err());
        println!("carve seed outside: {err}");
        assert!(err.contains("carve/seed") && err.contains("outside the domain hull"), "{err}");
        let err =
            format!("{}", carve(&s, &fr, &CarveStart::Seed([f64::NAN, 0.0, 0.0])).unwrap_err());
        assert!(err.contains("carve/seed") && err.contains("not finite"), "{err}");

        // The sphere wound inward holds its fluid inside: the hull region is
        // on the solid side, and a seed inside keeps the sphere.
        let mut s = ico(3, false, true);
        let n_body = s.tris.len() as u32;
        add_box(&mut s, -2.0, 2.0);
        let fr = recover_faces(&s, &[], &spec()).unwrap();
        let err = format!("{}", carve(&s, &fr, &CarveStart::Hull).unwrap_err());
        println!("carve inward hull: {err}");
        assert!(err.contains("carve/hull") && err.contains("patch \"sphere\""), "{err}");
        let c = carve(&s, &fr, &CarveStart::Seed([0.0, 0.0, 0.0])).unwrap();
        let v_in = -enclosed(&fr, &|p| p < n_body);
        let e = rel(c.volume, v_in);
        let eb = rel(c.boundary_volume(&fr.points), c.volume);
        println!("carve inward seed: V {:.15}, rel {e:.2e}, (116.16b) {eb:.2e}", c.volume);
        assert!(v_in > 0.0);
        assert!(e <= 1e-12 && eb <= 1e-12, "{e:e} {eb:e}");
        assert!(!c.regions[c.kept_region].touches_hull);
        assert!(c.face_sub.iter().all(|k| k.is_some()), "the sphere bounds the kept region");
    }

    /// Checks the kept boundary of a body in a box: each patch's face count
    /// is its subtriangle count, the hull faces are the triangulation's,
    /// every face appears once and every subtriangle face is the reverse of
    /// its input winding (the body's front faces the fluid, so the fluid's
    /// outward normal points into the body).
    fn count_gate(name: &str, s: &TetSurface, fr: &FaceRecovery) {
        let c = carve(s, fr, &CarveStart::Hull).unwrap();
        let (per, hull) = c.patch_face_counts(s, fr);
        let mut want = vec![0usize; s.patch_names.len()];
        for &p in &fr.sub_parent {
            want[s.tri_patch[p as usize] as usize] += 1;
        }
        println!("carve {name}: per patch {per:?}, hull {hull}");
        assert_eq!(per, want, "{name}: patch face counts");
        assert_eq!(hull, fr.delaunay.hull_faces().len(), "{name}: hull faces");
        let set: HashSet<[u32; 3]> = c.faces.iter().map(|f| sorted3(*f)).collect();
        assert_eq!(set.len(), c.faces.len(), "{name}: every face once");
        for (f, k) in c.faces.iter().zip(&c.face_sub) {
            if let Some(k) = k {
                let t = fr.subtris[*k as usize];
                let rev = [t[0], t[2], t[1]];
                let rot = |x: [u32; 3]| [[x[0], x[1], x[2]], [x[1], x[2], x[0]], [x[2], x[0], x[1]]];
                assert!(rot(rev).contains(f), "{name}: face {f:?} of subtriangle {t:?}");
            }
        }
    }

    #[test]
    fn patch_counts() {
        let mut s = TetSurface {
            points: Vec::new(),
            tris: Vec::new(),
            tri_patch: Vec::new(),
            patch_names: Vec::new(),
        };
        add_cube(&mut s, -0.5, 0.5, "", false);
        add_box(&mut s, -2.0, 2.0);
        let fr = recover_faces(&s, &[], &spec()).unwrap();
        count_gate("cube", &s, &fr);

        let mut s = ico(2, true, false);
        add_box(&mut s, -2.0, 2.0);
        let fr = recover_faces(&s, &[], &spec()).unwrap();
        count_gate("two-patch sphere", &s, &fr);
    }

    #[test]
    fn pocket() {
        // A hollow cube: outer [-1, 1]^3 wound outward, inner [-0.5, 0.5]^3
        // wound inward, in [-2, 2]^3.
        let mut s = TetSurface {
            points: Vec::new(),
            tris: Vec::new(),
            tri_patch: Vec::new(),
            patch_names: Vec::new(),
        };
        add_cube(&mut s, -1.0, 1.0, "out_", false);
        add_cube(&mut s, -0.5, 0.5, "in_", true);
        add_box(&mut s, -2.0, 2.0);
        let fr = recover_faces(&s, &[], &spec()).unwrap();
        let c = carve(&s, &fr, &CarveStart::Hull).unwrap();
        let drop = c.dropped();
        println!(
            "carve pocket: kept V {:.15}, {} regions, dropped {:?}",
            c.volume,
            c.regions.len(),
            drop
        );
        assert!(rel(c.volume, 56.0) <= 1e-12, "{}", c.volume);
        assert!(rel(c.boundary_volume(&fr.points), 56.0) <= 1e-12);
        assert_eq!(drop.len(), 1);
        assert_eq!(drop[0].lo, [-0.5; 3]);
        assert_eq!(drop[0].hi, [0.5; 3]);
        assert!(rel(drop[0].volume, 1.0) <= 1e-12);
        let solid: Vec<&CarveRegion> =
            c.regions.iter().filter(|r| r.side == Side::Solid).collect();
        assert_eq!(solid.len(), 1);
        assert!(rel(solid[0].volume, 7.0) <= 1e-12);
        assert_eq!(solid[0].lo, [-1.0; 3]);
        assert_eq!(solid[0].hi, [1.0; 3]);

        let cp = carve(&s, &fr, &CarveStart::Seed([0.1, 0.2, 0.3])).unwrap();
        assert!(rel(cp.volume, 1.0) <= 1e-12);
        assert!(rel(cp.boundary_volume(&fr.points), 1.0) <= 1e-12);
        let err =
            format!("{}", carve(&s, &fr, &CarveStart::Seed([0.75, 0.0, 0.0])).unwrap_err());
        println!("carve pocket seed in the shell: {err}");
        assert!(err.contains("carve/seed") && err.contains("wrong side"), "{err}");
    }
}
