// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.
// Provenance: see PROVENANCE.md. No GPL-licensed source was consulted.

//! Conforming boundary recovery of SPEC-LIT §116.11. The edge half
//! (TET-07a): every edge of a closed surface triangulation becomes a chain
//! of Delaunay edges after Steiner points are placed on it. The face half
//! (TET-07b): every surface triangle becomes a union of Delaunay faces after
//! Steiner points are placed on its edges and inside it, so every Steiner
//! point lies on the piecewise-linear surface.
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
//! 5. The face half, (116.15f)-(116.15h): each input triangle is its own
//!    facet. Its corners, chain points and facet points are triangulated in
//!    its plane frame (Lawson flips with the exact 2D predicates lifted to
//!    §116.4's 3D ones). A pass with no missing subsegment places, for each
//!    subtriangle that is not a Delaunay face, its circumcentre, or splits
//!    the subsegments of the triangle that point would encroach, or (the
//!    circumcentre outside and encroaching none) places the centroid. Both
//!    halves share the cap; a present encroached subtriangle is counted.
//!
//! Every coordinate is `[f64; 3]` in both builds (§116.2); the Delaunay
//! decisions are the exact predicates of §116.4-§116.5.
//!
//! Provenance: ORIGINAL. Written from Shewchuk (1998), Proc. 14th SoCG
//! 86-95 (the encroachment rule), Murphy, Mount & Gable (2001), Int. J.
//! Comput. Geom. Appl. 11(6):669-682 and Cohen-Steiner, Colin de Verdière &
//! Yvinec (2004), Comput. Geom. 28(2-3):217-233 (protecting balls and
//! concentric shells), Lawson (1977), Mathematical Software III 161-194
//! (the flips), Ruppert (1995), J. Algorithms 18(3):548-585 (the
//! circumcentre split) and Schönhardt (1928), Math. Ann. 98:309-312 (the
//! twisted prism); the choices are derived in SPEC-LIT §116.11. No
//! GPL-licensed source was consulted.

use super::delaunay::Delaunay;
use super::predicates::{insphere, orient3d};
use super::remesh::Remeshed;
use super::sizefield::point_triangle_distance;
use super::surfprep::TetSurface;
use super::{Point, TetRecoverSpec};
use crate::error::{Error, Result};
use std::collections::{BTreeSet, HashMap, HashSet};

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
pub(super) fn key(a: u32, b: u32) -> (u32, u32) {
    if a < b {
        (a, b)
    } else {
        (b, a)
    }
}

/// `p + t (q - p)`, the point at parameter `t` on `[p, q]`.
pub(super) fn lerp(p: Point, q: Point, t: f64) -> Point {
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

// ---------------------------------------------------------------------
// The face half (TET-07b): every surface triangle a union of Delaunay faces.

/// Where a Steiner point lies (SPEC-LIT §116.11).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum SteinerSite {
    /// On `edges[edge]` at parameter `t` from its first end, (116.15d) or
    /// (116.15e).
    Edge { edge: u32, t: f64 },
    /// Inside input triangle `tri = (a, b, c)`, at `a + u (b - a) + v (c - a)`
    /// (116.15f)-(116.15g).
    Facet { tri: u32, u: f64, v: f64 },
    /// A point of the Delaunay refinement of §116.13 that lies on no input
    /// triangle: inside the domain, or on the domain box.
    Volume,
}

/// What the face recovery built (SPEC-LIT §116.11): the edge recovery's
/// points and chains carried on, the facet points added, and the split
/// surface whose every triangle is a face of `delaunay`.
#[derive(Debug)]
pub struct FaceRecovery {
    /// The input points bit for bit, then the Steiner points in creation order.
    pub points: Vec<Point>,
    /// The input triangles, as given.
    pub tris: Vec<[u32; 3]>,
    /// The surface edges, each `[min, max]` in the input numbering, ascending.
    pub edges: Vec<[u32; 2]>,
    /// Beside `edges`: the edge's chain of point ids, in increasing parameter.
    pub chains: Vec<Vec<u32>>,
    /// Steiner point `k` (point id `n_input + k`) lies at `sites[k]`.
    pub sites: Vec<SteinerSite>,
    /// The protected vertices of the edge half, ascending.
    pub protected: Vec<u32>,
    /// Steiner points on ball surfaces (116.15d).
    pub ball_steiner: usize,
    /// Passes of the edge half, its last triangulation included.
    pub edge_passes: usize,
    /// Steiner points from missing-subsegment splits (116.15e), both halves.
    pub split_steiner: usize,
    /// Steiner points from subsegment splits a facet point would encroach (116.15g).
    pub encroach_split_steiner: usize,
    /// Steiner points inside triangles (116.15g), the centroids included.
    pub facet_steiner: usize,
    /// Of `facet_steiner`, the centroids placed when a circumcentre fell
    /// outside its triangle and encroached no subsegment of it.
    pub centroid_steiner: usize,
    /// Delaunay triangulations built, both halves, the last one included.
    pub passes: usize,
    /// The split surface: every triangle wound as its parent input triangle.
    pub subtris: Vec<[u32; 3]>,
    /// Beside `subtris`: the index of the parent input triangle.
    pub sub_parent: Vec<u32>,
    /// Subsegments present at the end but encroached (116.15): counted.
    pub encroached: usize,
    /// Subtriangles present at the end but encroached (116.15): counted.
    pub encroached_faces: usize,
    /// The Delaunay triangulation of `points` holding every subsegment and
    /// every subtriangle.
    pub delaunay: Delaunay,
}

impl FaceRecovery {
    /// The number of Steiner points.
    pub fn steiner(&self) -> usize {
        self.sites.len()
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

    /// The input triangles each point lies on: a corner's incident
    /// triangles, an edge point's two (or more) triangles of its edge, a
    /// facet point's own triangle.
    pub fn point_tris(&self) -> Vec<Vec<u32>> {
        let n_in = self.points.len() - self.sites.len();
        let mut on: Vec<Vec<u32>> = vec![Vec::new(); self.points.len()];
        let mut edge_tris: HashMap<(u32, u32), Vec<u32>> = HashMap::new();
        for (ti, t) in self.tris.iter().enumerate() {
            for k in 0..3 {
                on[t[k] as usize].push(ti as u32);
                edge_tris.entry(key(t[k], t[(k + 1) % 3])).or_default().push(ti as u32);
            }
        }
        for (k, s) in self.sites.iter().enumerate() {
            on[n_in + k] = match *s {
                SteinerSite::Edge { edge, .. } => {
                    let e = self.edges[edge as usize];
                    edge_tris[&(e[0], e[1])].clone()
                }
                SteinerSite::Facet { tri, .. } => vec![tri],
                SteinerSite::Volume => Vec::new(),
            };
        }
        on
    }

    /// The boundary face set: every face of `delaunay`'s finite tets whose
    /// three vertices lie on one common input triangle, each sorted
    /// ascending, the list sorted.
    pub fn surface_faces(&self) -> Vec<[u32; 3]> {
        let on = self.point_tris();
        let mut out: Vec<[u32; 3]> = delaunay_faces(&self.delaunay)
            .into_iter()
            .filter(|f| {
                on[f[0] as usize].iter().any(|t| {
                    on[f[1] as usize].contains(t) && on[f[2] as usize].contains(t)
                })
            })
            .collect();
        out.sort_unstable();
        out
    }

    /// `(present, total)`: the subtriangles that are faces of `delaunay`.
    pub fn recovered(&self) -> (usize, usize) {
        let df = delaunay_faces(&self.delaunay);
        let present = self.subtris.iter().filter(|t| df.contains(&sorted3(**t))).count();
        (present, self.subtris.len())
    }
}

fn sorted3(mut f: [u32; 3]) -> [u32; 3] {
    f.sort_unstable();
    f
}

/// Every face of the finite tets of `d`, sorted ascending.
pub fn delaunay_faces(d: &Delaunay) -> HashSet<[u32; 3]> {
    let mut out = HashSet::new();
    for t in d.finite_tets() {
        for skip in 0..4 {
            let mut f = [0u32; 3];
            let mut m = 0;
            for (i, &v) in t.iter().enumerate() {
                if i != skip {
                    f[m] = v;
                    m += 1;
                }
            }
            out.insert(sorted3(f));
        }
    }
    out
}

/// The plane point under every 2D predicate: `orient3d(a, b, c, LIFT) > 0`
/// for `a, b, c` counter-clockwise on `z = 0` (116.15f).
const LIFT: Point = [0.0, 0.0, -1.0];

fn on_plane(a: [f64; 2]) -> Point {
    [a[0], a[1], 0.0]
}

/// The exact 2D orientation, positive when `a, b, c` turn counter-clockwise:
/// (116.1) of the three points on `z = 0` and `(0, 0, -1)`, whose
/// determinant is the 2D one exactly.
pub(super) fn orient2(a: [f64; 2], b: [f64; 2], c: [f64; 2]) -> f64 {
    orient3d(&on_plane(a), &on_plane(b), &on_plane(c), &LIFT)
}

/// The exact 2D in-circle sign, positive when `p` lies strictly inside the
/// circle through the counter-clockwise `a, b, c`: (116.3) of the sphere
/// through them and `(0, 0, -1)`, which meets `z = 0` in that circle.
fn incircle2(a: [f64; 2], b: [f64; 2], c: [f64; 2], p: [f64; 2]) -> f64 {
    insphere(&on_plane(a), &on_plane(b), &on_plane(c), &LIFT, &on_plane(p))
}

/// The plane frame of input triangle `(a, b, c)` (116.15f).
pub(super) struct Frame {
    pub(super) a: Point,
    pub(super) e1: Point,
    pub(super) e2: Point,
    /// The corners in the plane: `(0, 0)`, `(|e1|, 0)`, `(e2 . e1 / |e1|, |e1 x e2| / |e1|)`.
    pub(super) c2: [[f64; 2]; 3],
}

impl Frame {
    pub(super) fn new(p: [Point; 3]) -> Frame {
        let e1 = sub(p[1], p[0]);
        let e2 = sub(p[2], p[0]);
        let l = norm(e1);
        Frame {
            a: p[0],
            e1,
            e2,
            c2: [[0.0, 0.0], [l, 0.0], [dot(e2, e1) / l, norm(cross(e1, e2)) / l]],
        }
    }

    pub(super) fn at2(&self, u: f64, v: f64) -> [f64; 2] {
        [u * self.c2[1][0] + v * self.c2[2][0], v * self.c2[2][1]]
    }

    pub(super) fn uv(&self, x: [f64; 2]) -> (f64, f64) {
        let v = x[1] / self.c2[2][1];
        ((x[0] - v * self.c2[2][0]) / self.c2[1][0], v)
    }

    pub(super) fn at3(&self, u: f64, v: f64) -> Point {
        [
            self.a[0] + u * self.e1[0] + v * self.e2[0],
            self.a[1] + u * self.e1[1] + v * self.e2[1],
            self.a[2] + u * self.e1[2] + v * self.e2[2],
        ]
    }
}

/// One input triangle's points in its plane frame: the boundary cycle
/// first (corner `a`, the chain to `b`, `b`, the chain to `c`, `c`, the
/// chain back), then its interior points; `side[k]` is the
/// `(edge, reversed)` of side `k` and `start[k]` its corner's position.
pub(super) struct FacetPoints {
    pub(super) ids: Vec<u32>,
    pub(super) xy: Vec<[f64; 2]>,
    pub(super) nb: usize,
    pub(super) start: [usize; 3],
    pub(super) side: [(usize, bool); 3],
}

/// The 2D triangulation of (116.15f): boundary points by edge splits,
/// interior points by location, then Lawson flips of every non-boundary
/// edge that fails the exact in-circle test. Counter-clockwise local
/// triangles, or `None` when a point cannot be placed.
pub(super) fn triangulate_facet(fp: &FacetPoints) -> Option<Vec<[usize; 3]>> {
    let xy = &fp.xy;
    let nb = fp.nb;
    let n = xy.len();
    let mut tris: Vec<[usize; 3]> = vec![[fp.start[0], fp.start[1], fp.start[2]]];
    // The boundary points, side by side, each splitting the boundary edge
    // that holds it.
    for k in 0..3 {
        let end = fp.start[(k + 1) % 3];
        let last = if k == 2 { nb } else { end };
        let mut prev = fp.start[k];
        for j in fp.start[k] + 1..last {
            let (ti, r) = tris.iter().enumerate().find_map(|(ti, t)| {
                (0..3).find_map(|m| {
                    (t[m] == prev && t[(m + 1) % 3] == end).then_some((ti, t[(m + 2) % 3]))
                })
            })?;
            tris[ti] = [prev, j, r];
            tris.push([j, end, r]);
            prev = j;
        }
    }
    // The interior points by location.
    for j in nb..n {
        let x = xy[j];
        let mut placed = false;
        for ti in 0..tris.len() {
            let t = tris[ti];
            let o = [
                orient2(xy[t[1]], xy[t[2]], x),
                orient2(xy[t[2]], xy[t[0]], x),
                orient2(xy[t[0]], xy[t[1]], x),
            ];
            if o.iter().any(|&s| s < 0.0) {
                continue;
            }
            let zeros: Vec<usize> = (0..3).filter(|&m| o[m] == 0.0).collect();
            match zeros.len() {
                0 => {
                    tris[ti] = [t[0], t[1], j];
                    tris.push([t[1], t[2], j]);
                    tris.push([t[2], t[0], j]);
                }
                1 => {
                    let m = zeros[0];
                    let (p, q, r) = (t[(m + 1) % 3], t[(m + 2) % 3], t[m]);
                    let (tj, s) = tris.iter().enumerate().find_map(|(tj, u)| {
                        (0..3).find_map(|w| {
                            (u[w] == q && u[(w + 1) % 3] == p).then_some((tj, u[(w + 2) % 3]))
                        })
                    })?;
                    tris[ti] = [p, j, r];
                    tris[tj] = [q, j, s];
                    tris.push([j, q, r]);
                    tris.push([j, p, s]);
                }
                _ => return None,
            }
            placed = true;
            break;
        }
        if !placed {
            return None;
        }
    }
    // Lawson flips until every interior edge is locally Delaunay.
    let mut sweeps = 0usize;
    loop {
        sweeps += 1;
        if sweeps > n * n + 10 {
            return None;
        }
        let mut owner: HashMap<(usize, usize), (usize, usize)> = HashMap::new();
        for (ti, t) in tris.iter().enumerate() {
            for m in 0..3 {
                owner.insert((t[(m + 1) % 3], t[(m + 2) % 3]), (ti, m));
            }
        }
        let mut touched = vec![false; tris.len()];
        let mut flips = 0usize;
        for ti in 0..tris.len() {
            for m in 0..3 {
                if touched[ti] {
                    break;
                }
                let t = tris[ti];
                let (p, q, r) = (t[(m + 1) % 3], t[(m + 2) % 3], t[m]);
                let Some(&(tj, w)) = owner.get(&(q, p)) else { continue };
                if touched[tj] {
                    continue;
                }
                let s = tris[tj][w];
                if incircle2(xy[p], xy[q], xy[r], xy[s]) <= 0.0 {
                    continue;
                }
                if orient2(xy[p], xy[s], xy[r]) <= 0.0 || orient2(xy[s], xy[q], xy[r]) <= 0.0 {
                    continue;
                }
                tris[ti] = [p, s, r];
                tris[tj] = [s, q, r];
                touched[ti] = true;
                touched[tj] = true;
                flips += 1;
            }
        }
        if flips == 0 {
            break;
        }
    }
    let ok = tris.len() + 2 == nb + 2 * (n - nb)
        && tris.iter().all(|t| orient2(xy[t[0]], xy[t[1]], xy[t[2]]) > 0.0);
    ok.then_some(tris)
}

/// The 2D circumcentre of `p, q, r`.
pub(super) fn circumcentre2(p: [f64; 2], q: [f64; 2], r: [f64; 2]) -> [f64; 2] {
    let (bx, by) = (q[0] - p[0], q[1] - p[1]);
    let (cx, cy) = (r[0] - p[0], r[1] - p[1]);
    let d = 2.0 * (bx * cy - by * cx);
    let (b2, c2) = (bx * bx + by * by, cx * cx + cy * cy);
    [p[0] + (cy * b2 - by * c2) / d, p[1] + (bx * c2 - cx * b2) / d]
}

/// The 3D circumcentre and circumradius of a triangle.
pub(super) fn circumsphere3(a: Point, b: Point, c: Point) -> (Point, f64) {
    let ab = sub(b, a);
    let ac = sub(c, a);
    let n = cross(ab, ac);
    let n2 = dot(n, n);
    let t1 = cross(n, ab);
    let t2 = cross(ac, n);
    let (lb, lc) = (dot(ab, ab), dot(ac, ac));
    let off = [
        (lc * t1[0] + lb * t2[0]) / (2.0 * n2),
        (lc * t1[1] + lb * t2[1]) / (2.0 * n2),
        (lc * t1[2] + lb * t2[2]) / (2.0 * n2),
    ];
    ([a[0] + off[0], a[1] + off[1], a[2] + off[2]], norm(off))
}

/// The subtriangles present in `d` whose diametral ball holds a vertex
/// strictly inside (116.15). A Delaunay face's diametral ball lies in the
/// pencil of spheres through its circumcircle, between or beyond its two
/// tets' empty circumspheres, so it is empty iff it holds neither apex.
fn count_encroached_faces(points: &[Point], subs: &[[u32; 3]], d: &Delaunay) -> usize {
    let set: HashSet<[u32; 3]> = subs.iter().map(|t| sorted3(*t)).collect();
    let mut hit: HashSet<[u32; 3]> = HashSet::new();
    for t in d.finite_tets() {
        for skip in 0..4 {
            let f = sorted3([t[(skip + 1) % 4], t[(skip + 2) % 4], t[(skip + 3) % 4]]);
            if !set.contains(&f) || hit.contains(&f) {
                continue;
            }
            let (c, r) = circumsphere3(
                points[f[0] as usize],
                points[f[1] as usize],
                points[f[2] as usize],
            );
            if norm(sub(points[t[skip] as usize], c)) < r {
                hit.insert(f);
            }
        }
    }
    hit.len()
}

/// Recovers every edge and then every triangle of `surf` (SPEC-LIT
/// §116.11, (116.15a)-(116.15h)): [`recover_edges`] first, then passes
/// that split each missing subsegment (116.15e) or, with none missing,
/// each missing subtriangle at its circumcentre in its plane, or the
/// subsegments of its triangle that point would encroach. The passes of
/// both halves share `spec.max_passes`. Refuses by name what
/// [`recover_edges`] refuses, a triangle that cannot be triangulated in
/// its plane (`recover/facet`) and a subsegment or subtriangle still
/// missing at the cap (`recover/cap`).
pub fn recover_faces(
    surf: &TetSurface,
    feature_edges: &[[u32; 2]],
    spec: &TetRecoverSpec,
) -> Result<FaceRecovery> {
    let er = recover_edges(surf, feature_edges, spec)?;
    let n_in = surf.points.len();
    let EdgeRecovery {
        mut points,
        edges,
        chains: id_chains,
        steiner_edge,
        steiner_t,
        protected,
        ball_steiner,
        mut split_steiner,
        passes: edge_passes,
        delaunay,
        ..
    } = er;
    let mut sites: Vec<SteinerSite> = steiner_edge
        .iter()
        .zip(&steiner_t)
        .map(|(&edge, &t)| SteinerSite::Edge { edge, t })
        .collect();
    let mut chains: Vec<Vec<(u32, f64)>> = id_chains
        .iter()
        .enumerate()
        .map(|(e, ch)| {
            ch.iter()
                .map(|&id| {
                    let t = if (id as usize) < n_in {
                        if id == edges[e][0] { 0.0 } else { 1.0 }
                    } else {
                        steiner_t[id as usize - n_in]
                    };
                    (id, t)
                })
                .collect()
        })
        .collect();
    let eidx: HashMap<(u32, u32), usize> =
        edges.iter().enumerate().map(|(e, ed)| ((ed[0], ed[1]), e)).collect();
    let tris = surf.tris.clone();
    let frames: Vec<Frame> = tris
        .iter()
        .map(|t| Frame::new([0, 1, 2].map(|k| surf.points[t[k] as usize])))
        .collect();
    for (ti, f) in frames.iter().enumerate() {
        if !(f.c2[2][1] > 0.0 && f.c2[2][1].is_finite()) {
            return Err(Error::Refused(format!("recover/input: triangle {ti} has no area")));
        }
    }
    let mut interior: Vec<Vec<(u32, f64, f64)>> = vec![Vec::new(); tris.len()];
    let mut encroach_split_steiner = 0usize;
    let mut facet_steiner = 0usize;
    let mut centroid_steiner = 0usize;
    let mut d = delaunay;
    let mut passes = edge_passes;
    loop {
        // The pass's subsegments and subtriangles against `d`.
        let de = delaunay_edges(&d);
        let mut missing_seg: Vec<(usize, usize)> = Vec::new();
        for (e, ch) in chains.iter().enumerate() {
            for i in 0..ch.len() - 1 {
                if !de.contains(&key(ch[i].0, ch[i + 1].0)) {
                    missing_seg.push((e, i));
                }
            }
        }
        let df = delaunay_faces(&d);
        let mut facets: Vec<(FacetPoints, Vec<[usize; 3]>)> = Vec::with_capacity(tris.len());
        let mut missing_tri: Vec<(usize, usize)> = Vec::new();
        for (ti, t) in tris.iter().enumerate() {
            let f = &frames[ti];
            let mut fp = FacetPoints {
                ids: Vec::new(),
                xy: Vec::new(),
                nb: 0,
                start: [0; 3],
                side: [(0, false); 3],
            };
            for k in 0..3 {
                let (p, q) = (t[k], t[(k + 1) % 3]);
                let e = eidx[&key(p, q)];
                let rev = edges[e][0] != p;
                let (ka, kb) = if rev { ((k + 1) % 3, k) } else { (k, (k + 1) % 3) };
                let (p2, q2) = (f.c2[ka], f.c2[kb]);
                fp.start[k] = fp.ids.len();
                fp.side[k] = (e, rev);
                let ch = &chains[e];
                let m = ch.len();
                for j in 0..m - 1 {
                    let (id, tt) = if rev { ch[m - 1 - j] } else { ch[j] };
                    fp.ids.push(id);
                    fp.xy.push(if j == 0 {
                        f.c2[k]
                    } else {
                        [p2[0] + tt * (q2[0] - p2[0]), p2[1] + tt * (q2[1] - p2[1])]
                    });
                }
            }
            fp.nb = fp.ids.len();
            for &(id, u, v) in &interior[ti] {
                fp.ids.push(id);
                fp.xy.push(f.at2(u, v));
            }
            let Some(local) = triangulate_facet(&fp) else {
                return Err(Error::Refused(format!(
                    "recover/facet: triangle {ti} [{}, {}, {}] cannot be triangulated in its plane",
                    t[0], t[1], t[2]
                )));
            };
            for (li, lt) in local.iter().enumerate() {
                if !df.contains(&sorted3(lt.map(|x| fp.ids[x]))) {
                    missing_tri.push((ti, li));
                }
            }
            facets.push((fp, local));
        }
        if missing_seg.is_empty() && missing_tri.is_empty() {
            let mut subtris: Vec<[u32; 3]> = Vec::new();
            let mut sub_parent: Vec<u32> = Vec::new();
            for (ti, (fp, local)) in facets.iter().enumerate() {
                for lt in local {
                    subtris.push(lt.map(|x| fp.ids[x]));
                    sub_parent.push(ti as u32);
                }
            }
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
            let encroached_faces = count_encroached_faces(&points, &subtris, &d);
            return Ok(FaceRecovery {
                points,
                tris,
                edges,
                chains,
                sites,
                protected,
                ball_steiner,
                edge_passes,
                split_steiner,
                encroach_split_steiner,
                facet_steiner,
                centroid_steiner,
                passes,
                subtris,
                sub_parent,
                encroached,
                encroached_faces,
                delaunay: d,
            });
        }
        if passes >= spec.max_passes {
            let ns: usize = chains.iter().map(|c| c.len() - 1).sum();
            let nf: usize = facets.iter().map(|(_, l)| l.len()).sum();
            let first = if let Some(&(e, i)) = missing_seg.first() {
                let (a, b) = key(chains[e][i].0, chains[e][i + 1].0);
                format!("subsegment [{a}, {b}]")
            } else {
                let (ti, li) = missing_tri[0];
                let (fp, local) = &facets[ti];
                let s = local[li].map(|x| fp.ids[x]);
                format!("subtriangle [{}, {}, {}] of triangle {ti}", s[0], s[1], s[2])
            };
            return Err(Error::Refused(format!(
                "recover/cap: {} of {} subsegments and {} of {} subtriangles missing after {} passes, first {}",
                missing_seg.len(),
                ns,
                missing_tri.len(),
                nf,
                passes,
                first
            )));
        }
        // The splits of this pass: subsegments (edge, position) and facet
        // points (triangle, u, v).
        let mut seg_split: BTreeSet<(usize, usize)> = BTreeSet::new();
        let mut facet_new: Vec<(usize, f64, f64, bool)> = Vec::new();
        if !missing_seg.is_empty() {
            seg_split.extend(missing_seg.iter().copied());
        } else {
            let mut taken: Vec<Vec<([f64; 2], f64)>> = vec![Vec::new(); tris.len()];
            for &(ti, li) in &missing_tri {
                let f = &frames[ti];
                let (fp, local) = &facets[ti];
                let lt = local[li];
                let (p, q, r) = (fp.xy[lt[0]], fp.xy[lt[1]], fp.xy[lt[2]]);
                let c2 = circumcentre2(p, q, r);
                let rad = ((c2[0] - p[0]).powi(2) + (c2[1] - p[1]).powi(2)).sqrt();
                let inside = orient2(f.c2[0], f.c2[1], c2) > 0.0
                    && orient2(f.c2[1], f.c2[2], c2) > 0.0
                    && orient2(f.c2[2], f.c2[0], c2) > 0.0;
                let (u, v) = f.uv(c2);
                let x = f.at3(u, v);
                let mut enc: Vec<(usize, usize)> = Vec::new();
                for &(e, _) in &fp.side {
                    let ch = &chains[e];
                    for i in 0..ch.len() - 1 {
                        let (a, b) = (points[ch[i].0 as usize], points[ch[i + 1].0 as usize]);
                        if dot(sub(x, a), sub(x, b)) < 0.0 {
                            enc.push((e, i));
                        }
                    }
                }
                if !enc.is_empty() {
                    seg_split.extend(enc);
                    continue;
                }
                let (pick, rr, centroid) = if inside {
                    (c2, rad, false)
                } else {
                    let g = [(p[0] + q[0] + r[0]) / 3.0, (p[1] + q[1] + r[1]) / 3.0];
                    (g, rad, true)
                };
                let near = taken[ti].iter().any(|&(y, ry)| {
                    let dd = ((y[0] - pick[0]).powi(2) + (y[1] - pick[1]).powi(2)).sqrt();
                    dd < 1e-6 * rr.min(ry)
                });
                if near {
                    continue;
                }
                taken[ti].push((pick, rr));
                let (u, v) = f.uv(pick);
                facet_new.push((ti, u, v, centroid));
            }
        }
        // Subsegment splits in (edge, position) order, inserted back to
        // front; then the facet points in (triangle, candidate) order.
        let mut new: Vec<(usize, usize, u32, f64)> = Vec::with_capacity(seg_split.len());
        for &(e, i) in &seg_split {
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
            let id = points.len() as u32;
            points.push(lerp(points[ed[0] as usize], points[ed[1] as usize], t));
            sites.push(SteinerSite::Edge { edge: e as u32, t });
            new.push((e, i, id, t));
        }
        if missing_seg.is_empty() {
            encroach_split_steiner += new.len();
        } else {
            split_steiner += new.len();
        }
        for &(e, i, id, t) in new.iter().rev() {
            chains[e].insert(i + 1, (id, t));
        }
        for &(ti, u, v, centroid) in &facet_new {
            let id = points.len() as u32;
            points.push(frames[ti].at3(u, v));
            sites.push(SteinerSite::Facet { tri: ti as u32, u, v });
            interior[ti].push((id, u, v));
            facet_steiner += 1;
            if centroid {
                centroid_steiner += 1;
            }
        }
        passes += 1;
        d = Delaunay::build_brio(&points)?;
    }
}

/// [`recover_faces`] on a remesh's output, its feature edges as the input
/// features.
pub fn recover_remeshed_faces(r: &Remeshed, spec: &TetRecoverSpec) -> Result<FaceRecovery> {
    recover_faces(&r.surface, &r.feature_edges, spec)
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

    // -----------------------------------------------------------------
    // TET-07b: the face half.

    /// Schönhardt's twisted prism (1928): the bottom `a_i` at 120 i
    /// degrees on the unit circle at `z = 0`, the top `b_i` at 120 i + 30
    /// degrees at `z = 1`, each side quad split along its reflex diagonal
    /// `a_i b_(i+1)`, wound outward.
    fn schonhardt_prism() -> TetSurface {
        let mut points: Vec<Point> = Vec::new();
        for (z, twist) in [(0.0, 0.0), (1.0, 30.0)] {
            for i in 0..3 {
                let a = (120.0 * i as f64 + twist).to_radians();
                points.push([a.cos(), a.sin(), z]);
            }
        }
        let mut tris: Vec<[u32; 3]> = vec![[0, 2, 1], [3, 4, 5]];
        for i in 0..3u32 {
            let j = (i + 1) % 3;
            tris.push([i, j, 3 + j]);
            tris.push([i, 3 + j, 3 + i]);
        }
        TetSurface {
            points,
            tris,
            tri_patch: vec![0; 8],
            patch_names: vec!["schonhardt".to_string()],
        }
    }

    /// The icosphere at `level` subdivisions on the unit sphere, outward.
    fn ico(level: u32) -> TetSurface {
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
                    *cache.entry(key(x, y)).or_insert_with(|| {
                        let (p, q) = (pts[x as usize], pts[y as usize]);
                        pts.push(unit(lerp(p, q, 0.5)));
                        pts.len() as u32 - 1
                    })
                };
                let (ab, bc, ca) = (mid(a, b), mid(b, c), mid(c, a));
                next.extend([[a, ab, ca], [b, bc, ab], [c, ca, bc], [ab, bc, ca]]);
            }
            tris = next;
        }
        let n = tris.len();
        TetSurface {
            points: pts,
            tris,
            tri_patch: vec![0; n],
            patch_names: vec!["sphere".to_string()],
        }
    }

    fn tri_area(a: Point, b: Point, c: Point) -> f64 {
        0.5 * norm(cross(sub(b, a), sub(c, a)))
    }

    /// The face gate (§116.18 TET-07b), counted from the final
    /// triangulation independently of the pass loop: every subsegment an
    /// edge and every subtriangle a face of it; each input triangle tiled
    /// by its subtriangles (relative 1e-12); each Steiner point within
    /// `1e-12 h` of every input triangle it lies on, `h` that triangle's
    /// longest edge, a facet point strictly inside its triangle. With
    /// `exact`, the boundary face set equals the split surface; without,
    /// it holds the split surface and every face beyond it is a face of a
    /// tetrahedron flat to rounding (`6 V <= 1e-12 L^3`, `L` its longest
    /// edge). Returns the faces beyond the split surface.
    fn face_gate(name: &str, r: &FaceRecovery, n_input: usize, exact: bool) -> usize {
        assert_eq!(r.points.len(), n_input + r.steiner());
        assert_eq!(r.delaunay.points(), &r.points[..]);
        r.delaunay.check().unwrap();
        assert!(r.passes < 64, "{name}: cap reached ({} passes)", r.passes);
        let (present, total) = r.recovered();
        assert_eq!(present, total, "{name}: {present} of {total} subtriangles");
        let de = delaunay_edges(&r.delaunay);
        let subs = r.subsegments();
        assert!(subs.iter().all(|s| de.contains(&(s[0], s[1]))), "{name}: subsegments");
        let mut split: Vec<[u32; 3]> = r.subtris.iter().map(|t| sorted3(*t)).collect();
        split.sort_unstable();
        let n_split = split.len();
        split.dedup();
        assert_eq!(split.len(), n_split, "{name}: a repeated subtriangle");
        let faces = r.surface_faces();
        let in_split: HashSet<[u32; 3]> = split.iter().copied().collect();
        assert!(split.iter().all(|f| faces.binary_search(f).is_ok()), "{name}: split surface");
        let extra: Vec<[u32; 3]> = faces.iter().filter(|f| !in_split.contains(*f)).copied().collect();
        if exact {
            assert_eq!(faces, split, "{name}: the boundary face set");
        } else {
            let mut flat: HashSet<[u32; 3]> = HashSet::new();
            for t in r.delaunay.finite_tets() {
                let p = t.map(|k| r.points[k as usize]);
                let six = dot(sub(p[1], p[0]), cross(sub(p[2], p[0]), sub(p[3], p[0]))).abs();
                let mut l = 0.0f64;
                for i in 0..4 {
                    for j in i + 1..4 {
                        l = l.max(norm(sub(p[j], p[i])));
                    }
                }
                if six <= 1e-12 * l * l * l {
                    for skip in 0..4 {
                        flat.insert(sorted3([t[(skip + 1) % 4], t[(skip + 2) % 4], t[(skip + 3) % 4]]));
                    }
                }
            }
            assert!(extra.iter().all(|f| flat.contains(f)), "{name}: a face beyond the split surface");
        }
        let mut tiled = vec![0.0f64; r.tris.len()];
        for (t, &p) in r.subtris.iter().zip(&r.sub_parent) {
            let [a, b, c] = t.map(|k| r.points[k as usize]);
            tiled[p as usize] += tri_area(a, b, c);
        }
        for (ti, t) in r.tris.iter().enumerate() {
            let [a, b, c] = t.map(|k| r.points[k as usize]);
            let want = tri_area(a, b, c);
            assert!((tiled[ti] - want).abs() <= 1e-12 * want, "{name}: triangle {ti} tiled {}", tiled[ti]);
        }
        let on = r.point_tris();
        let mut worst = 0.0f64;
        for k in 0..r.steiner() {
            let x = r.points[n_input + k];
            assert!(!on[n_input + k].is_empty(), "{name}: Steiner {k} on no triangle");
            for &ti in &on[n_input + k] {
                let tri = r.tris[ti as usize].map(|v| r.points[v as usize]);
                let h = (0..3).map(|m| norm(sub(tri[(m + 1) % 3], tri[m]))).fold(0.0, f64::max);
                let off = point_triangle_distance(x, &tri) / h;
                worst = worst.max(off);
                assert!(off <= 1e-12, "{name}: Steiner {k} off triangle {ti} by {off} h");
            }
            if let SteinerSite::Facet { u, v, .. } = r.sites[k] {
                assert!(u > 0.0 && v > 0.0 && u + v < 1.0, "{name}: Steiner {k} at ({u}, {v})");
            }
        }
        assert_eq!(
            r.steiner(),
            r.ball_steiner + r.split_steiner + r.encroach_split_steiner + r.facet_steiner
        );
        println!(
            "{name}: {} triangles -> {} subtriangles, {} subsegments, Steiner {} (ball {}, split {}, encroach-split {}, facet {} with {} centroids), passes {} (edge {}), encroached {} subsegments and {} subtriangles, {} faces beyond the split surface, worst off-surface {worst:.3e} h",
            r.tris.len(),
            r.subtris.len(),
            subs.len(),
            r.steiner(),
            r.ball_steiner,
            r.split_steiner,
            r.encroach_split_steiner,
            r.facet_steiner,
            r.centroid_steiner,
            r.passes,
            r.edge_passes,
            r.encroached,
            r.encroached_faces,
            extra.len()
        );
        extra.len()
    }

    #[test]
    fn schonhardt() {
        let s = schonhardt_prism();
        // Not vacuous: the input points' own triangulation misses a
        // surface edge or triangle.
        let d = Delaunay::build_brio(&s.points).unwrap();
        let (de, df) = (delaunay_edges(&d), delaunay_faces(&d));
        let have = s.tris.iter().all(|t| {
            df.contains(&sorted3(*t)) && (0..3).all(|k| de.contains(&key(t[k], t[(k + 1) % 3])))
        });
        assert!(!have, "the six points alone recover the prism");
        let r = recover_faces(&s, &[], &TetRecoverSpec::default()).unwrap();
        assert!(r.steiner() >= 1, "Schönhardt needs a Steiner point");
        let reflex: Vec<(u32, u32)> = (0..3u32).map(|i| key(i, 3 + (i + 1) % 3)).collect();
        for s in &r.sites {
            let SteinerSite::Edge { edge, .. } = *s else { panic!("a facet point: {s:?}") };
            let e = r.edges[edge as usize];
            assert!(reflex.contains(&(e[0], e[1])), "a Steiner point on edge {e:?}");
        }
        face_gate("schonhardt", &r, 6, true);
    }

    #[test]
    fn faces_exact_sphere() {
        let spec = TetRecoverSpec::default();
        let out = remeshed(&ico(3), 0.15);
        let r = recover_remeshed_faces(&out, &spec).unwrap();
        face_gate("sphere h 0.15", &r, out.surface.points.len(), true);
        for h in [0.1, 0.05] {
            let out = remeshed(&cube6(), h);
            let r = recover_remeshed_faces(&out, &spec).unwrap();
            face_gate(&format!("cube h {h}"), &r, out.surface.points.len(), true);
        }
    }

    #[test]
    fn steiner_on_surface() {
        let spec = TetRecoverSpec::default();
        let r = recover_faces(&schonhardt_prism(), &[], &spec).unwrap();
        face_gate("schonhardt", &r, 6, true);
        for (name, s, h) in [
            ("wedge", wedge(), 0.1),
            ("spike", spike(), 0.1),
            ("lprism", lprism(), 0.2),
        ] {
            let out = remeshed(&s, h);
            let r = recover_remeshed_faces(&out, &spec).unwrap();
            face_gate(name, &r, out.surface.points.len(), false);
            if name != "lprism" {
                assert!(r.facet_steiner >= 1, "{name}: the face half placed no point");
            }
        }
    }

    #[test]
    fn faces_cap_is_exact() {
        let out = remeshed(&wedge(), 0.1);
        let spec = TetRecoverSpec::default();
        let r = recover_remeshed_faces(&out, &spec).unwrap();
        assert!(r.passes > r.edge_passes, "the wedge needs face passes");
        let at = TetRecoverSpec { max_passes: r.passes, ..spec.clone() };
        assert_eq!(recover_remeshed_faces(&out, &at).unwrap().points, r.points);
        let below = TetRecoverSpec { max_passes: r.passes - 1, ..spec };
        let e = recover_remeshed_faces(&out, &below).unwrap_err().to_string();
        let tail = format!("subtriangles missing after {} passes, first ", r.passes - 1);
        assert!(e.contains("recover/cap: ") && e.contains(&tail), "{e}");
        println!("faces cap: {e}");
    }
}
