// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.
// Provenance: see PROVENANCE.md. No GPL-licensed source was consulted.

//! Conforming boundary recovery of SPEC-LIT §116.11, the edge half
//! (TET-07a): every edge of a closed surface triangulation becomes a chain
//! of Delaunay edges after Steiner points are placed on it.
//!
//! The steps, as §116.11 writes them in (116.15a)-(116.15e):
//!
//! 1. The segments are every undirected edge of the surface's triangles.
//!    The input's features (the feature edges of `Remeshed`) and its
//!    facets (the regions of triangles joined across non-feature edges)
//!    decide which vertices are protected: a vertex where two feature
//!    segments, or a feature segment and a facet, meet below
//!    `protect_angle_deg` (116.15a)/(116.15b).
//! 2. A protected vertex `v` gets a ball of radius (116.15c), a third of
//!    its shortest incident surface edge or of its distance to the nearest
//!    triangle not incident to it, whichever is smaller (Murphy, Mount &
//!    Gable 2001; Cohen-Steiner, Colin de Verdière & Yvinec 2004). Every
//!    surface edge from `v` is split once on the ball's surface (116.15d).
//! 3. A pass triangulates every point with [`Delaunay::build_brio`] and
//!    splits each subsegment that is not a Delaunay edge at the midpoint of
//!    its parameter interval on its parent edge (116.15e). Next to a
//!    protected vertex those midpoints sit at `r_v / 2^k`, on concentric
//!    shells, so splits never cascade toward `v`.
//! 4. The passes are capped at `max_passes`. A pass that still finds a
//!    missing subsegment at the cap is a refusal by name, never a silent
//!    pass. A subsegment that is present but encroached (116.15) is counted
//!    and reported, not split; §116.11 records why.
//!
//! Every coordinate is `[f64; 3]` in both builds (§116.2); the Delaunay
//! decisions are the exact predicates of §116.4-§116.5.
//!
//! Provenance: ORIGINAL. Written from Shewchuk (1998), Proc. 14th SoCG
//! 86-95 (the encroachment rule), Murphy, Mount & Gable (2001), Int. J.
//! Comput. Geom. Appl. 11(6):669-682 and Cohen-Steiner, Colin de Verdière &
//! Yvinec (2004), Comput. Geom. 28(2-3):217-233 (protecting balls and
//! concentric shells); the choices are derived in SPEC-LIT §116.11. No
//! GPL-licensed source was consulted.

use super::delaunay::Delaunay;
use super::remesh::Remeshed;
use super::sizefield::point_triangle_distance;
use super::surfprep::TetSurface;
use super::{Point, TetRecoverSpec};
use crate::error::{Error, Result};
use std::collections::{HashMap, HashSet};

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

fn norm(a: Point) -> f64 {
    dot(a, a).sqrt()
}

/// The undirected edge key `(min, max)`.
fn key(a: u32, b: u32) -> (u32, u32) {
    if a < b {
        (a, b)
    } else {
        (b, a)
    }
}

/// `p + t (q - p)`, the point at parameter `t` on `[p, q]`.
fn lerp(p: Point, q: Point, t: f64) -> Point {
    [
        p[0] + t * (q[0] - p[0]),
        p[1] + t * (q[1] - p[1]),
        p[2] + t * (q[2] - p[2]),
    ]
}

/// The cosine threshold of `protect_angle_deg`: an angle is below the
/// threshold iff its cosine is above this. It is exactly 0 at 90 degrees,
/// so an exact right angle (a dot product of exactly 0) is not below 90
/// (§116.11, (116.15a)).
fn cos_threshold(deg: f64) -> f64 {
    if deg == 90.0 {
        0.0
    } else {
        deg.to_radians().cos()
    }
}

/// The cosine of the angle between the ray `d` and the planar sector
/// spanned at its apex by `e1` and `e2` (a triangle's corner), (116.15b):
/// the projection of `d` onto the sector's plane when it falls inside the
/// sector, else the nearer of the two bounding rays.
fn ray_sector_cos(d: Point, e1: Point, e2: Point) -> f64 {
    let dl = norm(d);
    let c1 = dot(d, e1) / (dl * norm(e1));
    let c2 = dot(d, e2) / (dl * norm(e2));
    let rays = c1.max(c2);
    let n = cross(e1, e2);
    let nn = dot(n, n);
    if nn == 0.0 {
        return rays;
    }
    let dn = dot(d, n);
    let k = dn / nn;
    let dp = [d[0] - k * n[0], d[1] - k * n[1], d[2] - k * n[2]];
    if dot(cross(e1, dp), n) >= 0.0 && dot(cross(dp, e2), n) >= 0.0 {
        // cos = |d_p| / |d| = sqrt(1 - sin^2), sin = |d . n| / (|d| |n|):
        // a ray normal to the plane gives sin = 1 and cos = 0 exactly,
        // where |d_p| would keep the rounding of the projection.
        let sin = (dn.abs() / (dl * nn.sqrt())).min(1.0);
        (1.0 - sin * sin).sqrt().max(rays)
    } else {
        rays
    }
}

/// Every undirected edge of `tris`, each `[min, max]`, ascending.
pub fn surface_edges(tris: &[[u32; 3]]) -> Vec<[u32; 2]> {
    let mut v: Vec<[u32; 2]> = Vec::with_capacity(3 * tris.len());
    for t in tris {
        for (a, b) in [(t[0], t[1]), (t[1], t[2]), (t[2], t[0])] {
            let (a, b) = key(a, b);
            v.push([a, b]);
        }
    }
    v.sort_unstable();
    v.dedup();
    v
}

/// Every edge of the finite tetrahedra of `d`, as `(min, max)`.
pub fn delaunay_edges(d: &Delaunay) -> HashSet<(u32, u32)> {
    let mut s = HashSet::new();
    for t in d.finite_tets() {
        for i in 0..4 {
            for j in i + 1..4 {
                s.insert(key(t[i], t[j]));
            }
        }
    }
    s
}

/// The facets of (116.15b): triangles joined across every edge that is not
/// a feature edge, numbered by their smallest triangle, then compacted in
/// order of first appearance.
fn regions(tris: &[[u32; 3]], features: &HashSet<(u32, u32)>) -> Vec<u32> {
    let n = tris.len();
    let mut parent: Vec<usize> = (0..n).collect();
    fn find(p: &mut [usize], mut x: usize) -> usize {
        while p[x] != x {
            p[x] = p[p[x]];
            x = p[x];
        }
        x
    }
    let mut by_edge: HashMap<(u32, u32), usize> = HashMap::new();
    for (ti, t) in tris.iter().enumerate() {
        for (a, b) in [(t[0], t[1]), (t[1], t[2]), (t[2], t[0])] {
            let k = key(a, b);
            if features.contains(&k) {
                continue;
            }
            match by_edge.get(&k) {
                Some(&o) => {
                    let (ra, rb) = (find(&mut parent, o), find(&mut parent, ti));
                    if ra != rb {
                        let (lo, hi) = (ra.min(rb), ra.max(rb));
                        parent[hi] = lo;
                    }
                }
                None => {
                    by_edge.insert(k, ti);
                }
            }
        }
    }
    let mut id: HashMap<usize, u32> = HashMap::new();
    let mut out = Vec::with_capacity(n);
    for ti in 0..n {
        let r = find(&mut parent, ti);
        let next = id.len() as u32;
        out.push(*id.entry(r).or_insert(next));
    }
    out
}

/// The largest cosine of the input angles (116.15a)/(116.15b) at every
/// vertex with a feature edge: feature segment against feature segment,
/// and feature segment against every facet at the vertex that does not
/// hold the segment. Ascending by vertex.
fn feature_angle_cos(surf: &TetSurface, feature_edges: &[[u32; 2]]) -> Vec<(u32, f64)> {
    let fset: HashSet<(u32, u32)> = feature_edges.iter().map(|e| key(e[0], e[1])).collect();
    let region = regions(&surf.tris, &fset);
    // Each feature edge once, in ascending order, whatever the caller's
    // order or repeats.
    let mut fe: Vec<(u32, u32)> = fset.iter().copied().collect();
    fe.sort_unstable();
    let mut fe_at: HashMap<u32, Vec<u32>> = HashMap::new();
    for &(a, b) in &fe {
        fe_at.entry(a).or_default().push(b);
        fe_at.entry(b).or_default().push(a);
    }
    let mut tris_at: HashMap<u32, Vec<usize>> = HashMap::new();
    let mut edge_regions: HashMap<(u32, u32), Vec<u32>> = HashMap::new();
    for (ti, t) in surf.tris.iter().enumerate() {
        for k in 0..3 {
            tris_at.entry(t[k]).or_default().push(ti);
            let ek = key(t[k], t[(k + 1) % 3]);
            edge_regions.entry(ek).or_default().push(region[ti]);
        }
    }
    let p = |v: u32| surf.points[v as usize];
    let mut verts: Vec<u32> = fe_at.keys().copied().collect();
    verts.sort_unstable();
    let mut out = Vec::with_capacity(verts.len());
    for v in verts {
        let ws = &fe_at[&v];
        let pv = p(v);
        let mut best = f64::NEG_INFINITY;
        for i in 0..ws.len() {
            let di = sub(p(ws[i]), pv);
            for &wj in &ws[i + 1..] {
                let dj = sub(p(wj), pv);
                best = best.max(dot(di, dj) / (norm(di) * norm(dj)));
            }
            let holding: &[u32] = edge_regions
                .get(&key(v, ws[i]))
                .map(|r| r.as_slice())
                .unwrap_or(&[]);
            for &ti in tris_at.get(&v).map(|x| x.as_slice()).unwrap_or(&[]) {
                if holding.contains(&region[ti]) {
                    continue;
                }
                let t = surf.tris[ti];
                let k = t.iter().position(|&x| x == v).unwrap();
                let e1 = sub(p(t[(k + 1) % 3]), pv);
                let e2 = sub(p(t[(k + 2) % 3]), pv);
                best = best.max(ray_sector_cos(di, e1, e2));
            }
        }
        out.push((v, best));
    }
    out
}

/// The protected vertices of §116.11, ascending, each with its smallest
/// input angle in degrees: the vertices whose input angle (116.15a)/
/// (116.15b) is below `protect_angle_deg`.
pub fn protected_vertices(
    surf: &TetSurface,
    feature_edges: &[[u32; 2]],
    protect_angle_deg: f64,
) -> Vec<(u32, f64)> {
    let ct = cos_threshold(protect_angle_deg);
    feature_angle_cos(surf, feature_edges)
        .into_iter()
        .filter(|&(_, c)| c > ct)
        .map(|(v, c)| (v, c.clamp(-1.0, 1.0).acos().to_degrees()))
        .collect()
}

/// The protecting-ball radius (116.15c) of vertex `v`: a third of the
/// smaller of its shortest incident surface edge and its distance to the
/// nearest triangle that does not contain it.
pub fn protect_radius(surf: &TetSurface, v: u32) -> f64 {
    let pv = surf.points[v as usize];
    let mut lmin = f64::INFINITY;
    let mut dmin = f64::INFINITY;
    for t in &surf.tris {
        if t.contains(&v) {
            for &w in t {
                if w != v {
                    lmin = lmin.min(norm(sub(surf.points[w as usize], pv)));
                }
            }
        } else {
            let tri = [
                surf.points[t[0] as usize],
                surf.points[t[1] as usize],
                surf.points[t[2] as usize],
            ];
            dmin = dmin.min(point_triangle_distance(pv, &tri));
        }
    }
    lmin.min(dmin) / 3.0
}

/// What the edge recovery built (SPEC-LIT §116.11).
#[derive(Debug)]
pub struct EdgeRecovery {
    /// The input points bit for bit, then the Steiner points in creation order.
    pub points: Vec<Point>,
    /// The surface edges, each `[min, max]` in the input numbering, ascending.
    pub edges: Vec<[u32; 2]>,
    /// Beside `edges`: the edge's chain of point ids from `edges[e][0]` to
    /// `edges[e][1]`, in increasing parameter.
    pub chains: Vec<Vec<u32>>,
    /// Steiner point `k` (point id `n_input + k`) lies on `edges[steiner_edge[k]]`
    /// at parameter `steiner_t[k]` from its first end.
    pub steiner_edge: Vec<u32>,
    pub steiner_t: Vec<f64>,
    /// The protected vertices, ascending, with their ball radii (116.15c)
    /// and their smallest input angles in degrees.
    pub protected: Vec<u32>,
    pub protect_radius: Vec<f64>,
    pub protect_min_angle_deg: Vec<f64>,
    /// Steiner points placed on ball surfaces (116.15d).
    pub ball_steiner: usize,
    /// Steiner points placed by missing-subsegment splits (116.15e).
    pub split_steiner: usize,
    /// Delaunay triangulations built, the last one included.
    pub passes: usize,
    /// Subsegments present at the end but encroached (116.15): counted,
    /// not split (§116.11).
    pub encroached: usize,
    /// The Delaunay triangulation of `points` in which every subsegment is an edge.
    pub delaunay: Delaunay,
}

impl EdgeRecovery {
    /// The number of Steiner points.
    pub fn steiner(&self) -> usize {
        self.steiner_edge.len()
    }

    /// Every subsegment, `[min, max]`, in chain order.
    pub fn subsegments(&self) -> Vec<[u32; 2]> {
        let mut out = Vec::new();
        for ch in &self.chains {
            for w in ch.windows(2) {
                let (a, b) = key(w[0], w[1]);
                out.push([a, b]);
            }
        }
        out
    }

    /// `(present, total)`: the subsegments that are edges of `delaunay`.
    pub fn recovered(&self) -> (usize, usize) {
        let de = delaunay_edges(&self.delaunay);
        let subs = self.subsegments();
        let present = subs.iter().filter(|s| de.contains(&(s[0], s[1]))).count();
        (present, subs.len())
    }
}

/// The subsegments present in `d` whose diametral ball holds a vertex
/// strictly inside (116.15). A Delaunay edge's diametral ball is empty iff
/// it holds none of the edge's link vertices (its midpoint lies in its
/// Voronoi face, whose sides are the link's bisectors), so only the link
/// is tested.
fn count_encroached(points: &[Point], subs: &[[u32; 2]], d: &Delaunay) -> usize {
    let set: HashSet<(u32, u32)> = subs.iter().map(|s| (s[0], s[1])).collect();
    let mut link: HashMap<(u32, u32), Vec<u32>> = HashMap::new();
    for t in d.finite_tets() {
        for i in 0..4 {
            for j in i + 1..4 {
                let k = key(t[i], t[j]);
                if set.contains(&k) {
                    let l = link.entry(k).or_default();
                    for (m, &tm) in t.iter().enumerate() {
                        if m != i && m != j {
                            l.push(tm);
                        }
                    }
                }
            }
        }
    }
    let mut n = 0;
    for s in subs {
        let (a, b) = (points[s[0] as usize], points[s[1] as usize]);
        if let Some(l) = link.get(&(s[0], s[1])) {
            if l
                .iter()
                .any(|&c| dot(sub(points[c as usize], a), sub(points[c as usize], b)) < 0.0)
            {
                n += 1;
            }
        }
    }
    n
}

/// Recovers every edge of `surf` (SPEC-LIT §116.11, (116.15a)-(116.15e)).
/// `feature_edges` are the input's feature segments; each must be an edge
/// of `surf`. Refuses by name an invalid `spec`, an empty or out-of-range
/// triangulation, a feature edge that is not a surface edge, a protected
/// vertex without a ball, a split that no longer separates its ends, and a
/// subsegment still missing at `spec.max_passes`; the Delaunay kernel's own
/// refusals pass through.
pub fn recover_edges(
    surf: &TetSurface,
    feature_edges: &[[u32; 2]],
    spec: &TetRecoverSpec,
) -> Result<EdgeRecovery> {
    if !(spec.protect_angle_deg > 0.0 && spec.protect_angle_deg < 180.0) {
        return Err(Error::Refused(format!(
            "tet.recover.protect_angle_deg: must be in (0, 180), got {}",
            spec.protect_angle_deg
        )));
    }
    if spec.max_passes == 0 {
        return Err(Error::Refused(
            "tet.recover.max_passes: must be >= 1, got 0".to_string(),
        ));
    }
    if surf.tris.is_empty() {
        return Err(Error::Refused("recover/input: no triangles".to_string()));
    }
    let n_in = surf.points.len();
    for (ti, t) in surf.tris.iter().enumerate() {
        for (k, &v) in t.iter().enumerate() {
            if v as usize >= n_in {
                return Err(Error::Refused(format!(
                    "recover/input: triangle {ti} vertex {k} = {v} is not below {n_in}"
                )));
            }
        }
        if t[0] == t[1] || t[1] == t[2] || t[2] == t[0] {
            return Err(Error::Refused(format!(
                "recover/input: triangle {ti} repeats a vertex"
            )));
        }
    }
    let edges = surface_edges(&surf.tris);
    let eset: HashSet<(u32, u32)> = edges.iter().map(|e| (e[0], e[1])).collect();
    for e in feature_edges {
        if !eset.contains(&key(e[0], e[1])) {
            return Err(Error::Refused(format!(
                "recover/input: feature edge [{}, {}] is not a surface edge",
                e[0], e[1]
            )));
        }
    }

    // Protection, (116.15a)-(116.15c).
    let prot = protected_vertices(surf, feature_edges, spec.protect_angle_deg);
    let protected: Vec<u32> = prot.iter().map(|&(v, _)| v).collect();
    let protect_min_angle_deg: Vec<f64> = prot.iter().map(|&(_, a)| a).collect();
    let radius: Vec<f64> = protected.iter().map(|&v| protect_radius(surf, v)).collect();
    for (&v, &r) in protected.iter().zip(&radius) {
        if !(r > 0.0 && r.is_finite()) {
            return Err(Error::Refused(format!(
                "recover/protect: vertex {v} has ball radius {r}; a triangle not on it touches it"
            )));
        }
    }

    let mut points: Vec<Point> = surf.points.clone();
    let mut steiner_edge: Vec<u32> = Vec::new();
    let mut steiner_t: Vec<f64> = Vec::new();
    // Per edge: (point id, parameter from edges[e][0]), increasing parameter.
    let mut chains: Vec<Vec<(u32, f64)>> =
        edges.iter().map(|e| vec![(e[0], 0.0), (e[1], 1.0)]).collect();

    // The ball splits (116.15d): each edge from a protected vertex gets its
    // point at distance r_v from the vertex, computed from the vertex side.
    let rad_of: HashMap<u32, f64> = protected.iter().copied().zip(radius.iter().copied()).collect();
    for (e, ed) in edges.iter().enumerate() {
        for (end, &v) in ed.iter().enumerate() {
            let Some(&r) = rad_of.get(&v) else { continue };
            let w = ed[1 - end];
            let (pv, pw) = (points[v as usize], points[w as usize]);
            let s = r / norm(sub(pw, pv));
            let x = lerp(pv, pw, s);
            let t = if end == 0 { s } else { 1.0 - s };
            let id = points.len() as u32;
            points.push(x);
            steiner_edge.push(e as u32);
            steiner_t.push(t);
            let ch = &mut chains[e];
            let pos = ch.iter().position(|&(_, ct)| ct > t).unwrap();
            ch.insert(pos, (id, t));
        }
    }
    let ball_steiner = steiner_edge.len();

    // The passes (116.15e).
    let mut passes = 0usize;
    loop {
        passes += 1;
        let d = Delaunay::build_brio(&points)?;
        let de = delaunay_edges(&d);
        let mut missing: Vec<(usize, usize)> = Vec::new();
        for (e, ch) in chains.iter().enumerate() {
            for i in 0..ch.len() - 1 {
                if !de.contains(&key(ch[i].0, ch[i + 1].0)) {
                    missing.push((e, i));
                }
            }
        }
        if missing.is_empty() {
            let chains: Vec<Vec<u32>> =
                chains.iter().map(|c| c.iter().map(|&(id, _)| id).collect()).collect();
            let mut subs: Vec<[u32; 2]> = Vec::new();
            for ch in &chains {
                for w in ch.windows(2) {
                    let (a, b) = key(w[0], w[1]);
                    subs.push([a, b]);
                }
            }
            let encroached = count_encroached(&points, &subs, &d);
            let split_steiner = steiner_edge.len() - ball_steiner;
            return Ok(EdgeRecovery {
                points,
                edges,
                chains,
                steiner_edge,
                steiner_t,
                protected,
                protect_radius: radius,
                protect_min_angle_deg,
                ball_steiner,
                split_steiner,
                passes,
                encroached,
                delaunay: d,
            });
        }
        if passes >= spec.max_passes {
            let total: usize = chains.iter().map(|c| c.len() - 1).sum();
            let (e, i) = missing[0];
            let (a, b) = key(chains[e][i].0, chains[e][i + 1].0);
            return Err(Error::Refused(format!(
                "recover/cap: {} of {} subsegments missing after {} passes, first [{}, {}]",
                missing.len(),
                total,
                passes,
                a,
                b
            )));
        }
        // New points in (edge, position) order, then inserted back to front
        // so the earlier positions of an edge stay valid.
        let mut new: Vec<(usize, usize, u32, f64)> = Vec::with_capacity(missing.len());
        for &(e, i) in &missing {
            let (ta, tb) = (chains[e][i].1, chains[e][i + 1].1);
            let t = 0.5 * (ta + tb);
            if !(t > ta && t < tb) {
                let (a, b) = key(chains[e][i].0, chains[e][i + 1].0);
                return Err(Error::Refused(format!(
                    "recover/split: subsegment [{a}, {b}] of edge [{}, {}] is too short to split",
                    edges[e][0], edges[e][1]
                )));
            }
            let ed = edges[e];
            let x = lerp(points[ed[0] as usize], points[ed[1] as usize], t);
            let id = points.len() as u32;
            points.push(x);
            steiner_edge.push(e as u32);
            steiner_t.push(t);
            new.push((e, i, id, t));
        }
        for &(e, i, id, t) in new.iter().rev() {
            chains[e].insert(i + 1, (id, t));
        }
    }
}

/// [`recover_edges`] on a remesh's output, its feature edges as the input
/// features.
pub fn recover_remeshed(r: &Remeshed, spec: &TetRecoverSpec) -> Result<EdgeRecovery> {
    recover_edges(&r.surface, &r.feature_edges, spec)
}

#[cfg(test)]
mod tests {
    use super::super::remesh::{remesh, RemeshOptions};
    use super::super::surfprep::prepare;
    use super::*;

    /// The unit cube with a patch per face, as TET-06b's fixture.
    fn cube6() -> TetSurface {
        let points: Vec<Point> = (0..8u32)
            .map(|k| [(k & 1) as f64, ((k >> 1) & 1) as f64, ((k >> 2) & 1) as f64])
            .collect();
        let faces: [[u32; 4]; 6] = [
            [0, 4, 6, 2],
            [1, 3, 7, 5],
            [0, 1, 5, 4],
            [2, 6, 7, 3],
            [0, 2, 3, 1],
            [4, 5, 7, 6],
        ];
        let mut tris: Vec<[u32; 3]> = Vec::new();
        for f in &faces {
            tris.push([f[0], f[1], f[2]]);
            tris.push([f[0], f[2], f[3]]);
        }
        TetSurface {
            points,
            tris,
            tri_patch: (0..12u32).map(|t| t / 2).collect(),
            patch_names: ["xmin", "xmax", "ymin", "ymax", "zmin", "zmax"]
                .iter()
                .map(|s| s.to_string())
                .collect(),
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

    /// The 10 degree spike: a triangular pyramid whose three apex edges
    /// have length 1 and meet pairwise at 10 degrees; the base is the
    /// equilateral triangle of circumradius rho = sqrt(2 (1 - cos 10) / 3)
    /// at z = 0, the apex (point 3) at z = sqrt(1 - rho^2).
    fn spike() -> TetSurface {
        let rho = (2.0 * (1.0 - 10f64.to_radians().cos()) / 3.0).sqrt();
        let h = (1.0 - rho * rho).sqrt();
        let mut points: Vec<Point> = (0..3)
            .map(|k| {
                let a = 2.0 * std::f64::consts::PI * k as f64 / 3.0;
                [rho * a.cos(), rho * a.sin(), 0.0]
            })
            .collect();
        points.push([0.0, 0.0, h]);
        TetSurface {
            points,
            tris: vec![[0, 2, 1], [0, 1, 3], [1, 2, 3], [2, 0, 3]],
            tri_patch: vec![0; 4],
            patch_names: vec!["spike".to_string()],
        }
    }

    /// The rhombus pyramid of the cap test: the top z = 0 is split along
    /// A C, and no sphere through A (-2,0,0) and C (2,0,0) avoids both
    /// B (0.3,-1,0) and D (-0.2,1,0), so A C is in no Delaunay
    /// triangulation; its midpoint (0,0,0) recovers it.
    fn rhombus() -> TetSurface {
        TetSurface {
            points: vec![
                [-2.0, 0.0, 0.0],
                [0.3, -1.0, 0.0],
                [2.0, 0.0, 0.0],
                [-0.2, 1.0, 0.0],
                [0.0, 0.0, -1.0],
            ],
            tris: vec![[0, 1, 2], [0, 2, 3], [1, 0, 4], [2, 1, 4], [3, 2, 4], [0, 3, 4]],
            tri_patch: vec![0; 6],
            patch_names: vec!["rhombus".to_string()],
        }
    }

    /// The spike's smallest input angles, degrees, from its geometry: at
    /// the apex an apex edge against the opposite side face,
    /// asin(1.5 rho cos(phi)) (below the 10 degrees between two apex
    /// edges); at a base corner a base edge against the far side face,
    /// asin(sin(60) cos(phi)) (below the 60 degrees between two base
    /// edges); phi = atan(rho / (2 h)) is the side faces' lean.
    fn spike_angles() -> (f64, f64) {
        let rho = (2.0 * (1.0 - 10f64.to_radians().cos()) / 3.0).sqrt();
        let h = (1.0 - rho * rho).sqrt();
        let phi = (rho / (2.0 * h)).atan();
        let apex = (1.5 * rho * phi.cos()).asin().to_degrees();
        let base = (60f64.to_radians().sin() * phi.cos()).asin().to_degrees();
        (apex, base)
    }

    /// A concave fixture, not a plan gate: the L-shaped prism over
    /// (0,0), (2,0), (2,1), (1,1), (1,2), (0,2) from z = 0 to z = 1, whose
    /// edge over (1,1) is a 90 degree re-entrant edge.
    fn lprism() -> TetSurface {
        let xy = [[0.0, 0.0], [2.0, 0.0], [2.0, 1.0], [1.0, 1.0], [1.0, 2.0], [0.0, 2.0]];
        let mut points: Vec<Point> = Vec::new();
        for z in [0.0, 1.0] {
            for q in &xy {
                points.push([q[0], q[1], z]);
            }
        }
        let mut tris: Vec<[u32; 3]> = Vec::new();
        for k in 1..5u32 {
            tris.push([0, k + 1, k]);
            tris.push([6, 6 + k, 6 + k + 1]);
        }
        for i in 0..6u32 {
            let j = (i + 1) % 6;
            tris.push([i, j, j + 6]);
            tris.push([i, j + 6, i + 6]);
        }
        let n = tris.len();
        TetSurface {
            points,
            tris,
            tri_patch: vec![0; n],
            patch_names: vec!["lprism".to_string()],
        }
    }

    fn remeshed(s: &TetSurface, h: f64) -> Remeshed {
        let prep = prepare(s, 60.0).unwrap();
        remesh(&prep, &|_: Point| h, &RemeshOptions::default()).unwrap()
    }

    /// The edge gate (§116.18 TET-07a): every subsegment a Delaunay edge,
    /// counted independently of the recovery's own loop; the cap not
    /// reached; each Steiner point on its parent edge.
    fn gate(name: &str, r: &EdgeRecovery, n_input: usize, spec: &TetRecoverSpec) {
        let (present, total) = r.recovered();
        assert_eq!(present, total, "{name}: {present} of {total} subsegments");
        assert!(r.passes < spec.max_passes, "{name}: cap reached ({} passes)", r.passes);
        assert_eq!(r.points.len(), n_input + r.steiner());
        assert_eq!(r.steiner(), r.ball_steiner + r.split_steiner);
        for (k, (&e, &t)) in r.steiner_edge.iter().zip(&r.steiner_t).enumerate() {
            let [a, b] = r.edges[e as usize];
            let (pa, pb) = (r.points[a as usize], r.points[b as usize]);
            let x = r.points[n_input + k];
            let l = norm(sub(pb, pa));
            let off = norm(sub(x, lerp(pa, pb, t)));
            assert!(off <= 1e-12 * l, "{name}: Steiner {k} off its edge by {off}");
            assert!(t > 0.0 && t < 1.0);
            assert!(r.chains[e as usize].contains(&((n_input + k) as u32)));
        }
        assert_eq!(r.delaunay.points(), &r.points[..], "{name}: the triangulated points");
        r.delaunay.check().unwrap();
        println!(
            "{name}: {} points in, {} edges, {} subsegments, {} protected, Steiner {} (ball {}, split {}), passes {}, encroached {}",
            n_input,
            r.edges.len(),
            total,
            r.protected.len(),
            r.steiner(),
            r.ball_steiner,
            r.split_steiner,
            r.passes,
            r.encroached
        );
    }

    #[test]
    fn edges_cube() {
        let out = remeshed(&cube6(), 0.1);
        let spec = TetRecoverSpec::default();
        let r = recover_remeshed(&out, &spec).unwrap();
        assert!(r.protected.is_empty(), "cube corners are right angles: {:?}", r.protected);
        assert_eq!(r.ball_steiner, 0);
        assert_eq!(r.edges, surface_edges(&out.surface.tris));
        gate("edges_cube", &r, out.surface.points.len(), &spec);
    }

    #[test]
    fn edges_wedge() {
        let out = remeshed(&wedge(), 0.1);
        let spec = TetRecoverSpec::default();
        let r = recover_remeshed(&out, &spec).unwrap();
        assert_eq!(r.protected, vec![0, 1, 2, 3, 4, 5], "the six corners");
        for (k, &a) in r.protect_min_angle_deg.iter().enumerate() {
            let want = if k % 3 == 0 { 30.0 } else { 75.0 };
            assert!((a - want).abs() <= 1e-9, "corner {k}: {a} degrees");
        }
        gate("edges_wedge", &r, out.surface.points.len(), &spec);
    }

    #[test]
    fn edges_spike() {
        let out = remeshed(&spike(), 0.1);
        let spec = TetRecoverSpec::default();
        let r = recover_remeshed(&out, &spec).unwrap();
        assert_eq!(r.protected, vec![0, 1, 2, 3], "the four corners");
        let (apex, base) = spike_angles();
        for (k, &a) in r.protect_min_angle_deg.iter().enumerate() {
            let want = if k == 3 { apex } else { base };
            assert!((a - want).abs() <= 1e-9, "corner {k}: {a} degrees, want {want}");
        }
        // Each ball point sits at r_v from its protected vertex.
        let n_in = out.surface.points.len();
        let mut seen = 0;
        for k in 0..r.ball_steiner {
            let [a, b] = r.edges[r.steiner_edge[k] as usize];
            let x = r.points[n_in + k];
            let mut hit = false;
            for (i, &v) in r.protected.iter().enumerate() {
                if v == a || v == b {
                    let dist = norm(sub(x, r.points[v as usize]));
                    if (dist - r.protect_radius[i]).abs() <= 1e-12 * r.protect_radius[i] {
                        hit = true;
                    }
                }
            }
            assert!(hit, "ball point {k} is on no ball surface");
            seen += 1;
        }
        let incident: usize = r
            .edges
            .iter()
            .map(|e| r.protected.iter().filter(|&&v| e.contains(&v)).count())
            .sum();
        assert_eq!(seen, incident, "one ball point per protected end");
        gate("edges_spike", &r, n_in, &spec);
    }

    #[test]
    fn edges_concave() {
        let out = remeshed(&lprism(), 0.2);
        let spec = TetRecoverSpec::default();
        let r = recover_remeshed(&out, &spec).unwrap();
        assert!(r.protected.is_empty(), "every input angle is 90 degrees or more");
        gate("edges_concave", &r, out.surface.points.len(), &spec);
    }

    /// Termination beyond the named fixtures: the four fixtures at five
    /// sizes, each recovered in full with the cap (64) never reached.
    #[test]
    fn cap_never_reached_sweep() {
        let spec = TetRecoverSpec::default();
        let mut worst = 0usize;
        let mut steiner = 0usize;
        let fixtures: [(&str, TetSurface); 4] =
            [("cube", cube6()), ("wedge", wedge()), ("spike", spike()), ("lprism", lprism())];
        for (name, s) in &fixtures {
            for h in [0.05, 0.07, 0.1, 0.13, 0.2] {
                let out = remeshed(s, h);
                let r = recover_remeshed(&out, &spec).unwrap();
                let (present, total) = r.recovered();
                assert_eq!(present, total, "{name} at h = {h}");
                assert!(r.passes < spec.max_passes, "{name} at h = {h}: {} passes", r.passes);
                worst = worst.max(r.passes);
                steiner += r.steiner();
            }
        }
        println!("cap_never_reached_sweep: 20 runs, at most {worst} passes, {steiner} Steiner points in all");
    }

    #[test]
    fn cap_is_a_refusal() {
        let s = rhombus();
        let one = TetRecoverSpec { protect_angle_deg: 90.0, max_passes: 1 };
        let e = recover_edges(&s, &[], &one).unwrap_err().to_string();
        assert!(
            e.contains("recover/cap: 1 of 9 subsegments missing after 1 passes, first [0, 2]"),
            "{e}"
        );
        let two = TetRecoverSpec { protect_angle_deg: 90.0, max_passes: 2 };
        let r = recover_edges(&s, &[], &two).unwrap();
        assert_eq!(r.passes, 2);
        assert_eq!(r.steiner(), 1);
        assert_eq!(r.split_steiner, 1);
        assert_eq!(r.points[5], [0.0, 0.0, 0.0]);
        assert_eq!(r.chains[r.steiner_edge[0] as usize], vec![0, 5, 2]);
        assert_eq!(r.recovered(), (10, 10));

        // The cap is exact on the concave prism, which needs more than one
        // round of splits: `passes` succeeds with the same Steiner points,
        // one pass fewer refuses.
        let out = remeshed(&lprism(), 0.2);
        let spec = TetRecoverSpec::default();
        let r = recover_remeshed(&out, &spec).unwrap();
        assert!(r.passes >= 3, "{} passes", r.passes);
        let at = TetRecoverSpec { max_passes: r.passes, ..spec.clone() };
        assert_eq!(recover_remeshed(&out, &at).unwrap().points, r.points);
        let below = TetRecoverSpec { max_passes: r.passes - 1, ..spec };
        let e = recover_remeshed(&out, &below).unwrap_err().to_string();
        assert!(e.contains("recover/cap: "), "{e}");
        println!("cap_is_a_refusal: {e}");
    }

    #[test]
    fn protection_rules() {
        // The spike's raw surface: the smallest input angles are 8.67
        // degrees at the apex and 59.87 at the base corners (`spike_angles`);
        // at 8 degrees nothing is protected, at 11 the apex, at 61 all four.
        let s = spike();
        let fe = surface_edges(&s.tris);
        let (apex, base) = spike_angles();
        let p = protected_vertices(&s, &fe, 61.0);
        for &(v, a) in &p {
            let want = if v == 3 { apex } else { base };
            assert!((a - want).abs() <= 1e-9, "raw corner {v}: {a}, want {want}");
        }
        assert!(protected_vertices(&s, &fe, 8.0).is_empty());
        assert_eq!(p.iter().map(|x| x.0).collect::<Vec<_>>(), vec![0, 1, 2, 3]);
        let p11 = protected_vertices(&s, &fe, 11.0);
        assert_eq!(p11.iter().map(|x| x.0).collect::<Vec<_>>(), vec![3]);
        // The cube: every input angle is exactly 90 degrees.
        let c = cube6();
        let fc = prepare(&c, 60.0).unwrap().feature_edges;
        assert!(protected_vertices(&c, &fc, 90.0).is_empty());
        assert_eq!(protected_vertices(&c, &fc, 90.5).len(), 8);
        // The radius (116.15c) on the raw spike's apex: a third of its
        // distance to the base triangle, the height sqrt(1 - rho^2), which
        // is shorter than the apex edges (1).
        let rho = (2.0 * (1.0 - 10f64.to_radians().cos()) / 3.0).sqrt();
        let h = (1.0 - rho * rho).sqrt();
        let r3 = protect_radius(&s, 3);
        assert!((r3 - h / 3.0).abs() <= 1e-15, "{r3}");
        // A segment against a facet: the ray along +z against the sector
        // of the xy quadrant is 90 degrees, along (1,1,1) 35.26 degrees.
        let e1 = [1.0, 0.0, 0.0];
        let e2 = [0.0, 1.0, 0.0];
        assert_eq!(ray_sector_cos([0.0, 0.0, 1.0], e1, e2), 0.0);
        let c111 = ray_sector_cos([1.0, 1.0, 1.0], e1, e2);
        assert!((c111 - (2.0f64 / 3.0).sqrt()).abs() <= 1e-15, "{c111}");
        // Outside the sector the nearer bounding ray decides.
        let cm = ray_sector_cos([-1.0, -1.0, 0.0], e1, e2);
        assert!((cm + 0.5f64.sqrt()).abs() <= 1e-15, "{cm}");
    }

    #[test]
    fn refusals() {
        let spec = TetRecoverSpec::default();
        let mut s = rhombus();
        s.tris[2][1] = 9;
        let e = recover_edges(&s, &[], &spec).unwrap_err().to_string();
        assert!(e.contains("recover/input: triangle 2 vertex 1 = 9 is not below 5"), "{e}");
        let s = rhombus();
        let e = recover_edges(&s, &[[1, 3]], &spec).unwrap_err().to_string();
        assert!(e.contains("recover/input: feature edge [1, 3] is not a surface edge"), "{e}");
        // A protected vertex touched by a triangle it is not on has no ball.
        let mut t = spike();
        let apex = t.points[3];
        t.points.extend([apex, [1.0, 0.0, 2.0], [0.0, 1.0, 2.0]]);
        t.tris.push([4, 5, 6]);
        t.tri_patch.push(0);
        let fe = surface_edges(&t.tris[..4]);
        let e = recover_edges(&t, &fe, &spec).unwrap_err().to_string();
        assert!(e.contains("recover/protect: vertex 3 has ball radius 0"), "{e}");
        let bad = TetRecoverSpec { protect_angle_deg: 0.0, max_passes: 4 };
        let e = recover_edges(&s, &[], &bad).unwrap_err().to_string();
        assert!(e.contains("tet.recover.protect_angle_deg: must be in (0, 180), got 0"), "{e}");
        let bad = TetRecoverSpec { protect_angle_deg: 90.0, max_passes: 0 };
        let e = recover_edges(&s, &[], &bad).unwrap_err().to_string();
        assert!(e.contains("tet.recover.max_passes: must be >= 1, got 0"), "{e}");
    }
}
