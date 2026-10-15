// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.
// Provenance: see PROVENANCE.md. No GPL-licensed source was consulted.

//! The isotropic surface remesh of SPEC-LIT §116.10 (TET-06a): split /
//! collapse / flip / tangential relaxation, every vertex projected back
//! onto the input surface, until the edge lengths sit at the target
//! `h(x)`. Every coordinate is [`Point`] = `[f64; 3]` in both builds
//! (§116.2). Each iteration runs the five steps of §116.10 in order:
//! splits by (116.14a) in repeated sweeps, collapses by (116.14b) with
//! the link-condition and normal guards, flips by (116.14c) with the
//! valence-energy criterion, tangential relaxation by (116.14) in Jacobi
//! order, and projection by (116.14d) through [`Projector`], the f64
//! closest-point query over a BVH of the input's f64 boxes. Ties go to
//! the smallest triangle index, every pass visits edges and vertices in
//! a fixed index order, and no `HashMap`/`HashSet` iteration reaches a
//! result, so two runs are bit-identical. TET-06b adds the features:
//! corners never move, feature vertices move only along their input
//! polyline ((116.14f)-(116.14j)), feature and seam edges are never
//! flipped and never collapsed across, and the output carries the
//! features in its own numbering. (116.14k) is the flat test that
//! takes the exact-zero area test's place and also refuses a flip onto
//! a flat triangle, and (116.14l) is the cut-triangle refusal that
//! keeps three vertices of one feature line from ending up as one
//! triangle.
//!
//! Provenance: ORIGINAL. Written from Botsch & Kobbelt (2004), Proc. SGP
//! 185-192; the choices are derived in SPEC-LIT §116.10. No GPL-licensed
//! source was consulted.

use std::collections::{BTreeMap, BTreeSet};

use super::sizefield::{par_map, point_box_distance};
use super::surfprep::{PreparedSurface, TetSurface};
use super::Point;
use crate::error::{Error, Result};

// ==========================================================================
//  Vector helpers - mirrors of sizefield.rs's private ones, same order
// ==========================================================================

fn dot(a: Point, b: Point) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn sub(a: Point, b: Point) -> Point {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn add(a: Point, b: Point) -> Point {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

fn scale(a: Point, t: f64) -> Point {
    [a[0] * t, a[1] * t, a[2] * t]
}

/// The Euclidean norm, as `sizefield::norm`.
fn norm(x: Point) -> f64 {
    dot(x, x).sqrt()
}

/// The f64 cross product.
fn cross(a: Point, b: Point) -> Point {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

/// The midpoint, per component `(a + b) * 0.5`.
fn mid(a: Point, b: Point) -> Point {
    [(a[0] + b[0]) * 0.5, (a[1] + b[1]) * 0.5, (a[2] + b[2]) * 0.5]
}

/// The triangle's f64 normal `(x1 - x0) x (x2 - x0)`, unnormalised.
fn tri_normal(p: &[Point; 3]) -> Point {
    cross(sub(p[1], p[0]), sub(p[2], p[0]))
}

/// (116.14k): the triangle is flat when
/// `|(x1 - x0) x (x2 - x0)| <= 1e-6 max(l01^2, l12^2, l02^2)`.
/// A degeneracy floor (height below one millionth of the longest edge),
/// not a quality bar; it replaces the TET-06a exact-zero area test,
/// whose cross product is round-off noise of either sign on a straight
/// feature line that is not axis-aligned.
fn tri_flat(t: &[Point; 3]) -> bool {
    let l01 = sub(t[1], t[0]);
    let l12 = sub(t[2], t[1]);
    let l20 = sub(t[0], t[2]);
    let l2max = dot(l01, l01).max(dot(l12, l12)).max(dot(l20, l20));
    norm(tri_normal(t)) <= 1e-6 * l2max
}

/// The remesh's every `h` call goes through here: a value that is not
/// finite or is not positive is refused by name, with the point and the
/// value in the message (§116.10).
fn h_checked(h: &dyn Fn(Point) -> f64, x: Point) -> Result<f64> {
    let v = h(x);
    if !v.is_finite() || v <= 0.0 {
        return Err(Error::Refused(format!(
            "remesh/size: h({:?}) = {v} must be finite and positive",
            x
        )));
    }
    Ok(v)
}

// ==========================================================================
//  The closest point on a triangle (116.14d)'s per-triangle query
// ==========================================================================

/// The closest point to `x` on the segment `p q`, the point form of
/// `sizefield::segment_distance` (same arithmetic, same order).
fn segment_closest(x: Point, p: Point, q: Point) -> Point {
    let pq = sub(q, p);
    let l2 = dot(pq, pq);
    if l2 == 0.0 {
        return p;
    }
    let t = (dot(sub(x, p), pq) / l2).clamp(0.0, 1.0);
    add(p, scale(pq, t))
}

/// (116.14j): the closest point on the input polyline `L_k`, over its
/// consecutive input segments; the first strict minimum in increasing
/// segment order. A feature vertex of `L_k` is projected here.
fn closest_on_polyline(x: Point, segs: &[(Point, Point)]) -> Point {
    let mut best = segment_closest(x, segs[0].0, segs[0].1);
    let mut d = norm(sub(x, best));
    for (p, q) in &segs[1..] {
        let q2 = segment_closest(x, *p, *q);
        let d2 = norm(sub(x, q2));
        if d2 < d {
            best = q2;
            d = d2;
        }
    }
    best
}

/// The closest point on the triangle `t` to `x`: the same Voronoi-region
/// walk as `sizefield::point_triangle_distance`, returning the point
/// instead of the distance, so `norm(x - here)` equals that distance bit
/// for bit. A zero-area triangle falls back to the best of its three
/// segments, first strict minimum in the order AB, BC, CA.
pub fn closest_point_on_triangle(x: Point, t: &[Point; 3]) -> Point {
    let a = t[0];
    let b = t[1];
    let c = t[2];
    let ab = sub(b, a);
    let ac = sub(c, a);
    let n = [ab[1] * ac[2] - ab[2] * ac[1], ab[2] * ac[0] - ab[0] * ac[2], ab[0] * ac[1] - ab[1] * ac[0]];
    if dot(n, n) == 0.0 {
        let q1 = segment_closest(x, a, b);
        let d1 = norm(sub(x, q1));
        let q2 = segment_closest(x, b, c);
        let d2 = norm(sub(x, q2));
        let q3 = segment_closest(x, c, a);
        let d3 = norm(sub(x, q3));
        let mut q = q1;
        let mut d = d1;
        if d2 < d {
            q = q2;
            d = d2;
        }
        if d3 < d {
            q = q3;
        }
        return q;
    }
    let ap = sub(x, a);
    let d1 = dot(ab, ap);
    let d2 = dot(ac, ap);
    if d1 <= 0.0 && d2 <= 0.0 {
        return a; // vertex region A
    }
    let bp = sub(x, b);
    let d3 = dot(ab, bp);
    let d4 = dot(ac, bp);
    if d3 >= 0.0 && d4 <= d3 {
        return b; // vertex region B
    }
    let vc = d1 * d4 - d3 * d2;
    if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
        return add(a, scale(ab, d1 / (d1 - d3))); // edge AB
    }
    let cp = sub(x, c);
    let d5 = dot(ab, cp);
    let d6 = dot(ac, cp);
    if d6 >= 0.0 && d5 <= d6 {
        return c; // vertex region C
    }
    let vb = d5 * d2 - d1 * d6;
    if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
        return add(a, scale(ac, d2 / (d2 - d6))); // edge AC
    }
    let va = d3 * d6 - d5 * d4;
    if va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0 {
        // edge BC
        let den = (d4 - d3) + (d5 - d6);
        return add(b, scale(sub(c, b), (d4 - d3) / den));
    }
    // Face interior: all three sub-areas positive, so the sum is too.
    let denom = 1.0 / (va + vb + vc);
    add(add(a, scale(ab, vb * denom)), scale(ac, vc * denom))
}

// ==========================================================================
//  The projector: the input surface and a BVH over its f64 boxes
// ==========================================================================

struct BvhNode {
    lo: Point,
    hi: Point,
    /// Children; `u32::MAX` on a leaf.
    left: u32,
    right: u32,
    /// A leaf's triangle indices, ascending.
    tris: Vec<u32>,
}

struct BvhItem {
    idx: u32,
    lo: Point,
    hi: Point,
    cen: Point,
}

/// The closest-point query of (116.14d): every input triangle, a BVH over
/// their f64 boxes, and the tie rule - among equally close triangles the
/// smallest index wins. A node's box is pruned only when its distance is
/// STRICTLY greater than the best so far, so an equal-distance triangle is
/// never hidden behind a pruned node.
pub struct Projector {
    tris: Vec<[Point; 3]>,
    nodes: Vec<BvhNode>,
    root: u32,
}

const LEAF_CAP: usize = 8;

fn box_of(t: &[Point; 3]) -> (Point, Point) {
    let mut lo = t[0];
    let mut hi = t[0];
    for p in t {
        for d in 0..3 {
            lo[d] = lo[d].min(p[d]);
            hi[d] = hi[d].max(p[d]);
        }
    }
    (lo, hi)
}

impl Projector {
    /// Builds the BVH: median split on the node box's longest axis, ties
    /// of the centroid order broken by the triangle index, so the tree is
    /// the same on every run.
    pub fn new(surf: &TetSurface) -> Projector {
        let tris: Vec<[Point; 3]> = surf
            .tris
            .iter()
            .map(|t| {
                [
                    surf.points[t[0] as usize],
                    surf.points[t[1] as usize],
                    surf.points[t[2] as usize],
                ]
            })
            .collect();
        let mut items: Vec<BvhItem> = tris
            .iter()
            .enumerate()
            .map(|(i, t)| {
                let (lo, hi) = box_of(t);
                BvhItem {
                    idx: i as u32,
                    lo,
                    hi,
                    cen: mid(lo, hi),
                }
            })
            .collect();
        let mut nodes: Vec<BvhNode> = Vec::new();
        let root = build_bvh(&mut nodes, &mut items);
        Projector { tris, nodes, root }
    }

    /// `(closest point, triangle index, distance)`, distance = norm of the
    /// difference. Ties go to the smaller triangle index.
    pub fn closest(&self, x: Point) -> (Point, u32, f64) {
        let mut best: Option<(Point, u32, f64)> = None;
        let mut stack: Vec<u32> = vec![self.root];
        while let Some(id) = stack.pop() {
            let node = &self.nodes[id as usize];
            let bd = point_box_distance(x, node.lo, node.hi);
            if let Some((_, _, d)) = &best {
                if bd > *d {
                    continue;
                }
            }
            if node.left == u32::MAX {
                for &ti in &node.tris {
                    let p = closest_point_on_triangle(x, &self.tris[ti as usize]);
                    let d = norm(sub(x, p));
                    let better = match &best {
                        None => true,
                        Some((_, bj, dj)) => d < *dj || (d == *dj && ti < *bj),
                    };
                    if better {
                        best = Some((p, ti, d));
                    }
                }
            } else {
                let l = &self.nodes[node.left as usize];
                let r = &self.nodes[node.right as usize];
                let dl = point_box_distance(x, l.lo, l.hi);
                let dr = point_box_distance(x, r.lo, r.hi);
                // Pop the nearer child first; the answer does not depend
                // on the order.
                if dl <= dr {
                    stack.push(node.right);
                    stack.push(node.left);
                } else {
                    stack.push(node.left);
                    stack.push(node.right);
                }
            }
        }
        best.unwrap_or(([0.0; 3], 0, 0.0))
    }

    /// The same answer by scanning every triangle in index order, keeping
    /// the first strict minimum.
    pub fn closest_brute(&self, x: Point) -> (Point, u32, f64) {
        let mut best: Option<(Point, u32, f64)> = None;
        for (i, t) in self.tris.iter().enumerate() {
            let p = closest_point_on_triangle(x, t);
            let d = norm(sub(x, p));
            let better = match &best {
                None => true,
                Some((_, _, dj)) => d < *dj,
            };
            if better {
                best = Some((p, i as u32, d));
            }
        }
        best.unwrap_or(([0.0; 3], 0, 0.0))
    }
}

/// The recursive BVH build over a mutable slice of items.
fn build_bvh(nodes: &mut Vec<BvhNode>, items: &mut [BvhItem]) -> u32 {
    let (lo, hi) = items_box(items);
    let id = nodes.len() as u32;
    nodes.push(BvhNode {
        lo,
        hi,
        left: u32::MAX,
        right: u32::MAX,
        tris: Vec::new(),
    });
    if items.len() <= LEAF_CAP {
        nodes[id as usize].tris = items.iter().map(|it| it.idx).collect();
        nodes[id as usize].tris.sort_unstable();
        return id;
    }
    let mut axis = 0usize;
    for d in 1..3 {
        if hi[d] - lo[d] > hi[axis] - lo[axis] {
            axis = d;
        }
    }
    items.sort_by(|p, q| {
        p.cen[axis]
            .total_cmp(&q.cen[axis])
            .then(p.idx.cmp(&q.idx))
    });
    let m = items.len() / 2;
    let (lft, rgt) = items.split_at_mut(m);
    let l = build_bvh(nodes, lft);
    let r = build_bvh(nodes, rgt);
    nodes[id as usize].left = l;
    nodes[id as usize].right = r;
    id
}

/// The union of the items' boxes.
fn items_box(items: &[BvhItem]) -> (Point, Point) {
    let mut lo = items[0].lo;
    let mut hi = items[0].hi;
    for it in items {
        for d in 0..3 {
            lo[d] = lo[d].min(it.lo[d]);
            hi[d] = hi[d].max(it.hi[d]);
        }
    }
    (lo, hi)
}

// ==========================================================================
//  The remesh itself
// ==========================================================================

/// The remesh's knobs (§116.10): how many split/collapse/flip/relax/
/// project iterations to run, and on how many workers the relaxation and
/// the projection run, `0` meaning every core (TET-15). The result is the
/// same for every worker count, bit for bit.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RemeshOptions {
    pub iterations: u32,
    pub threads: usize,
}

impl Default for RemeshOptions {
    fn default() -> Self {
        RemeshOptions {
            iterations: 10,
            threads: 0,
        }
    }
}

/// What one `remesh` run did, over all its iterations.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct RemeshStats {
    pub iterations: u32,
    pub splits: usize,
    pub collapses: usize,
    pub flips: usize,
    pub n_points: usize,
    pub n_tris: usize,
}

/// The remeshed surface and its stats, and the features of the output in
/// the output's numbering (TET-06b).
#[derive(Debug, Clone, PartialEq)]
pub struct Remeshed {
    pub surface: TetSurface,
    pub stats: RemeshStats,
    /// The output's feature edges, each [min, max], in ascending order, in the OUTPUT numbering.
    pub feature_edges: Vec<[u32; 2]>,
    /// Beside feature_edges: the seam flag each edge carries from its input edge.
    pub edge_is_seam: Vec<bool>,
    /// Beside feature_edges: the index k of the input polyline (input.polylines[k]) the edge lies on.
    pub edge_polyline: Vec<u32>,
    /// The corners, ascending, in the output numbering.
    pub corners: Vec<u32>,
}

/// The undirected edge key `(min, max)`.
fn edge_key(a: u32, b: u32) -> (u32, u32) {
    if a < b {
        (a, b)
    } else {
        (b, a)
    }
}

/// The triangle rotated so it starts with the directed edge `x -> y`, if
/// its winding traverses it that way.
fn rotated(t: [u32; 3], x: u32, y: u32) -> Option<[u32; 3]> {
    for k in 0..3 {
        if t[k] == x && t[(k + 1) % 3] == y {
            return Some([t[k], t[(k + 1) % 3], t[(k + 2) % 3]]);
        }
    }
    None
}

/// The edge `(a, b)`'s two triangles in winding order: the one that
/// traverses it `a -> b` first, the one that traverses `b -> a` second,
/// in either incidence-list order. `None` when the pair does not
/// traverse the edge in both directions.
fn edge_oriented(t1: [u32; 3], t2: [u32; 3], a: u32, b: u32) -> Option<([u32; 3], [u32; 3])> {
    if let Some(r1) = rotated(t1, a, b) {
        Some((r1, rotated(t2, b, a)?))
    } else {
        Some((rotated(t2, a, b)?, rotated(t1, b, a)?))
    }
}

/// The working mesh: tombstoned vertices and triangles, per-vertex
/// neighbour sets, and the edge -> incident-triangles map, all updated
/// incrementally per operation (§116.10's suggested structure).
struct Mesh {
    pts: Vec<Point>,
    tris: Vec<[u32; 3]>,
    tri_patch: Vec<u32>,
    live_v: Vec<bool>,
    live_t: Vec<bool>,
    nbrs: Vec<BTreeSet<u32>>,
    edges: BTreeMap<(u32, u32), Vec<u32>>,
    n_live_v: usize,
    /// A corner never moves and is never removed (116.14g)-(116.14j).
    corner: Vec<bool>,
    /// A feature vertex (not a corner): an end of a feature edge; it
    /// moves only along its polyline.
    feat_v: Vec<bool>,
    /// The polyline `k` a feature vertex lies on (its class' index k).
    fpoly: Vec<u32>,
    /// The feature edges, one map per vertex: `feat_idx[v][w]` is the
    /// `(polyline k, seam flag)` of the feature edge `(v, w)`, stored in
    /// both directions. A lookup at `v` costs its own feature edges,
    /// whatever the table's size; only `(v + 1)..` and `..v` ranges of
    /// it are ever walked, in that order, so `feat_at`'s neighbours come
    /// out exactly as the old whole-table walk produced them (the
    /// larger indices first in ascending order, then the smaller ones).
    feat_idx: Vec<BTreeMap<u32, (u32, bool)>>,
}

impl Mesh {
    fn from_surface(s: &TetSurface) -> Mesh {
        let n = s.points.len();
        let mut edges: BTreeMap<(u32, u32), Vec<u32>> = BTreeMap::new();
        let mut nbrs: Vec<BTreeSet<u32>> = vec![BTreeSet::new(); n];
        for (t, tri) in s.tris.iter().enumerate() {
            for k in 0..3 {
                let a = tri[k];
                let b = tri[(k + 1) % 3];
                nbrs[a as usize].insert(b);
                edges.entry(edge_key(a, b)).or_default().push(t as u32);
            }
        }
        Mesh {
            pts: s.points.clone(),
            tris: s.tris.clone(),
            tri_patch: s.tri_patch.clone(),
            live_v: vec![true; n],
            live_t: vec![true; s.tris.len()],
            nbrs,
            edges,
            n_live_v: n,
            corner: vec![false; n],
            feat_v: vec![false; n],
            fpoly: vec![0; n],
            feat_idx: vec![BTreeMap::new(); n],
        }
    }

    /// Sets the feature edge `(a, b)` to `(k, seam)`, both directions.
    fn feat_insert(&mut self, a: u32, b: u32, d: (u32, bool)) {
        self.feat_idx[a as usize].insert(b, d);
        self.feat_idx[b as usize].insert(a, d);
    }

    /// Removes the feature edge `(a, b)` from both directions.
    fn feat_remove(&mut self, a: u32, b: u32) {
        self.feat_idx[a as usize].remove(&b);
        self.feat_idx[b as usize].remove(&a);
    }

    /// The feature edges at `v`, each as `(other vertex, (k, seam))`:
    /// the ones with a larger index first in ascending order, then the
    /// smaller ones in ascending order. Two BTreeMap ranges of `v`'s
    /// own map, so the cost is in proportion to `v`'s feature edges,
    /// whatever the table's size.
    fn feat_at(&self, v: u32) -> Vec<(u32, (u32, bool))> {
        let mut out: Vec<(u32, (u32, bool))> = Vec::new();
        for (&w, &d) in self.feat_idx[v as usize].range((v + 1)..) {
            out.push((w, d));
        }
        for (&w, &d) in self.feat_idx[v as usize].range(..v) {
            out.push((w, d));
        }
        out
    }

    /// (116.14l): the triangle is a cut triangle when two of its edges
    /// are feature edges that meet at a feature vertex (not a corner;
    /// `feat_v` is false at corners).
    fn tri_is_cut(&self, tri: [u32; 3]) -> bool {
        self.tri_is_cut_by(tri, &|x, y| {
            self.feat_idx[x as usize].contains_key(&y)
        })
    }

    /// (116.14l) as [`Mesh::tri_is_cut`], with the feature edges judged
    /// by `feat`: the collapse guards pass the post-collapse table.
    fn tri_is_cut_by(&self, tri: [u32; 3], feat: &dyn Fn(u32, u32) -> bool) -> bool {
        for k in 0..3 {
            let (a, b, c) = (tri[k], tri[(k + 1) % 3], tri[(k + 2) % 3]);
            if self.feat_v[a as usize] && feat(a, b) && feat(a, c) {
                return true;
            }
        }
        false
    }

    /// The output's feature tables: the edges each `[min, max]` in
    /// ascending order, with the seam flag and the polyline `k` beside
    /// them. Walking each vertex's `(v + 1)..` range in ascending `v`
    /// visits every edge once, in the `(min, max)` order the old
    /// whole-table keys() gave.
    fn feature_tables(&self) -> (Vec<[u32; 2]>, Vec<bool>, Vec<u32>) {
        let mut feature_edges = Vec::new();
        let mut edge_is_seam = Vec::new();
        let mut edge_polyline = Vec::new();
        for v in 0..self.pts.len() as u32 {
            for (&w, &(k, seam)) in self.feat_idx[v as usize].range((v + 1)..) {
                feature_edges.push([v, w]);
                edge_is_seam.push(seam);
                edge_polyline.push(k);
            }
        }
        (feature_edges, edge_is_seam, edge_polyline)
    }

    /// The triangle's points, in its winding order.
    fn tri_pts(&self, t: u32) -> [Point; 3] {
        let tri = self.tris[t as usize];
        [
            self.pts[tri[0] as usize],
            self.pts[tri[1] as usize],
            self.pts[tri[2] as usize],
        ]
    }

    /// A test-only consistency check: every live triangle's vertices are
    /// live and in range, and the edge map matches the live triangles.
    #[cfg(test)]
    #[allow(dead_code)]
    fn audit(&self, tag: &str, it: u32, sweep: u32) {
        for (key, inc) in &self.edges {
            for &t in inc {
                assert!(
                    self.live_t[t as usize],
                    "{tag} it {it} sweep {sweep}: edge {key:?} holds dead tri {t}"
                );
            }
            let mut sorted = inc.clone();
            sorted.sort_unstable();
            sorted.dedup();
            assert_eq!(
                sorted.len(),
                inc.len(),
                "{tag} it {it} sweep {sweep}: edge {key:?} has duplicates ({inc:?})"
            );
            assert!(
                inc.len() == 2,
                "{tag} it {it} sweep {sweep}: edge {key:?} has {} tris ({inc:?})",
                inc.len()
            );
        }
        for (t, tri) in self.tris.iter().enumerate() {
            if !self.live_t[t] {
                continue;
            }
            for k in 0..3 {
                let v = tri[k] as usize;
                assert!(
                    v < self.pts.len() && self.live_v[v],
                    "{tag} it {it} sweep {sweep}: tri {t} vertex {v} not live (pts {})",
                    self.pts.len()
                );
            }
            for k in 0..3 {
                let key = edge_key(tri[k], tri[(k + 1) % 3]);
                let inc = self.edges.get(&key).unwrap_or_else(|| {
                    panic!("{tag} it {it} sweep {sweep}: tri {t} edge {key:?} missing")
                });
                assert!(
                    inc.contains(&(t as u32)),
                    "{tag} it {it} sweep {sweep}: tri {t} not on edge {key:?} ({inc:?})"
                );
            }
        }
    }


    /// Drops triangle `t` from the edge's incidence list, removing an
    /// emptied edge.
    fn edge_drop(&mut self, key: (u32, u32), t: u32) {
        if let Some(v) = self.edges.get_mut(&key) {
            v.retain(|&x| x != t);
            if v.is_empty() {
                self.edges.remove(&key);
            }
        }
    }

    /// Adds triangle `t` to the edge's incidence list.
    fn edge_add(&mut self, key: (u32, u32), t: u32) {
        self.edges.entry(key).or_default().push(t);
    }

    /// (116.14a): one sweep over the edges in ascending `(min, max)`
    /// order, splitting each edge longer than `(4/3) h` at its midpoint.
    /// Returns how many splits happened.
    fn split_sweep(&mut self, h: &dyn Fn(Point) -> f64) -> Result<usize> {
        let cand: Vec<(u32, u32)> = self.edges.keys().copied().collect();
        let mut n = 0usize;
        for (a, b) in cand {
            if !self.live_v[a as usize] || !self.live_v[b as usize] {
                continue;
            }
            let pa = self.pts[a as usize];
            let pb = self.pts[b as usize];
            let m = mid(pa, pb);
            let hm = h_checked(h, m)?;
            if norm(sub(pa, pb)) > (4.0 / 3.0) * hm && self.split(a, b, m) {
                n += 1;
            }
        }
        Ok(n)
    }

    /// Splits the edge `(a, b)` at `m`. Each of the two triangles becomes
    /// two; each child keeps its parent's patch (116.14a).
    fn split(&mut self, a: u32, b: u32, m: Point) -> bool {
        let (t1, t2) = match self.edges.get(&edge_key(a, b)) {
            Some(v) if v.len() == 2 => (v[0], v[1]),
            _ => return false,
        };
        let g1 = self.tris[t1 as usize];
        let g2 = self.tris[t2 as usize];
        // Whichever triangle traverses the edge a -> b keeps its patch
        // with its children, in either incidence-list order.
        let (t_ab, t_ba) = if rotated(g1, a, b).is_some() {
            (t1, t2)
        } else {
            (t2, t1)
        };
        let (r1, r2) = match edge_oriented(g1, g2, a, b) {
            Some(pair) => pair,
            None => return false,
        };
        let c = r1[2];
        let d = r2[2];
        if c == d {
            return false;
        }
        let mi = self.pts.len() as u32;
        self.pts.push(m);
        self.live_v.push(true);
        self.nbrs.push(BTreeSet::new());
        self.corner.push(false);
        self.feat_v.push(false);
        self.fpoly.push(0);
        self.feat_idx.push(BTreeMap::new());
        self.n_live_v += 1;
        let patch_ab = self.tri_patch[t_ab as usize];
        let patch_ba = self.tri_patch[t_ba as usize];
        self.live_t[t1 as usize] = false;
        self.live_t[t2 as usize] = false;
        let t1a = self.push_tri([a, mi, c], patch_ab);
        let t1b = self.push_tri([mi, b, c], patch_ab);
        let t2a = self.push_tri([b, mi, d], patch_ba);
        let t2b = self.push_tri([mi, a, d], patch_ba);
        self.edges.remove(&edge_key(a, b));
        self.edge_add(edge_key(a, mi), t1a);
        self.edge_add(edge_key(a, mi), t2b);
        self.edge_add(edge_key(mi, b), t1b);
        self.edge_add(edge_key(mi, b), t2a);
        self.edge_add(edge_key(mi, c), t1a);
        self.edge_add(edge_key(mi, c), t1b);
        self.edge_add(edge_key(mi, d), t2a);
        self.edge_add(edge_key(mi, d), t2b);
        self.edge_drop(edge_key(a, c), t_ab);
        self.edge_add(edge_key(a, c), t1a);
        self.edge_drop(edge_key(b, c), t_ab);
        self.edge_add(edge_key(b, c), t1b);
        self.edge_drop(edge_key(a, d), t_ba);
        self.edge_add(edge_key(a, d), t2b);
        self.edge_drop(edge_key(b, d), t_ba);
        self.edge_add(edge_key(b, d), t2a);
        for v in [a, b, c, d] {
            self.nbrs[v as usize].insert(mi);
        }
        self.nbrs[a as usize].remove(&b);
        self.nbrs[b as usize].remove(&a);
        self.nbrs[mi as usize] = [a, b, c, d].into_iter().collect();
        // (116.14f): a split feature edge leaves the new vertex a feature
        // vertex of the same polyline, its two halves carry the edge's
        // polyline and seam flag, and (m, c), (m, d) are free.
        if let Some(&(k, seam)) = self.feat_idx[a as usize].get(&b) {
            self.feat_remove(a, b);
            self.feat_insert(a, mi, (k, seam));
            self.feat_insert(mi, b, (k, seam));
            self.feat_v[mi as usize] = true;
            self.fpoly[mi as usize] = k;
        }
        true
    }

    /// Appends a live triangle with its patch.
    fn push_tri(&mut self, tri: [u32; 3], patch: u32) -> u32 {
        let id = self.tris.len() as u32;
        self.tris.push(tri);
        self.tri_patch.push(patch);
        self.live_t.push(true);
        id
    }

    /// (116.14b): one pass over the edges in ascending `(min, max)`
    /// order, collapsing each edge shorter than `(4/5) h` whose guards
    /// all hold. Returns the count.
    fn collapse_pass(&mut self, h: &dyn Fn(Point) -> f64) -> Result<usize> {
        let cand: Vec<(u32, u32)> = self.edges.keys().copied().collect();
        let mut n = 0usize;
        for key in cand {
            if !self.live_v[key.0 as usize] || !self.live_v[key.1 as usize] {
                continue;
            }
            if !self.edges.contains_key(&key) {
                continue;
            }
            if self.collapse(key.0, key.1, h)? {
                n += 1;
            }
        }
        Ok(n)
    }

    /// The collapse of `(a, b)` per (116.14g): the survivor `s` and its
    /// position `p` follow the class table - on a free edge the smaller
    /// free index survives at the midpoint, the non-free vertex survives
    /// in place, a free edge with neither end free is refused; on a
    /// feature edge the corner survives in place, two feature vertices
    /// collapse onto the smaller index at the midpoint, two corners are
    /// refused. The length test stays `l_ab < (4/5) h(m_ab)`; `p`
    /// replaces `m_ab` in the guards. A feature edge of the dying vertex
    /// may not land on a feature edge of the survivor (the merged-edge
    /// rule). Every guard of §116.10 is checked before anything moves.
    fn collapse(&mut self, a: u32, b: u32, h: &dyn Fn(Point) -> f64) -> Result<bool> {
        let (t1, t2) = match self.edges.get(&edge_key(a, b)) {
            Some(v) if v.len() == 2 => (v[0], v[1]),
            _ => return Ok(false),
        };
        let (r1, r2) = match edge_oriented(self.tris[t1 as usize], self.tris[t2 as usize], a, b) {
            Some(pair) => pair,
            None => return Ok(false),
        };
        let (c, d) = (r1[2], r2[2]);
        if c == d {
            return Ok(false);
        }
        let pa = self.pts[a as usize];
        let pb = self.pts[b as usize];
        let m = mid(pa, pb);
        if !(norm(sub(pa, pb)) < (4.0 / 5.0) * h_checked(h, m)?) {
            return Ok(false);
        }
        // The link condition: the only common neighbours of `a` and `b`
        // are the two opposite vertices.
        let common: Vec<u32> = self.nbrs[a as usize]
            .intersection(&self.nbrs[b as usize])
            .copied()
            .collect();
        if common.len() != 2 || !common.contains(&c) || !common.contains(&d) {
            return Ok(false);
        }
        // The surface keeps more than four vertices.
        if self.n_live_v - 1 <= 4 {
            return Ok(false);
        }
        // (116.14g): the survivor, its position, the refusal.
        let feat_ab = self.feat_idx[a as usize].get(&b).copied();
        let cls = |v: u32| {
            if self.corner[v as usize] {
                2u8
            } else if self.feat_v[v as usize] {
                1
            } else {
                0
            }
        };
        let (ca, cb) = (cls(a), cls(b));
        let (s, t, p) = match (feat_ab, ca, cb) {
            (None, 0, 0) => (a, b, m),
            (None, 0, _) => (b, a, self.pts[b as usize]),
            (None, _, 0) => (a, b, self.pts[a as usize]),
            (None, _, _) => return Ok(false),
            (Some(_), 2, 2) => return Ok(false),
            (Some(_), 2, _) => (a, b, self.pts[a as usize]),
            (Some(_), _, 2) => (b, a, self.pts[b as usize]),
            (Some(_), _, _) => (a, b, m),
        };
        // The merged-edge rule: a feature edge of the dying vertex `t`
        // may not land on a feature edge of the survivor `s`.
        for (w, _) in self.feat_at(t) {
            if w != s && self.feat_idx[s as usize].contains_key(&w) {
                return Ok(false);
            }
        }
        if !self.collapse_guards(s, t, p, t1, t2, h)? {
            return Ok(false);
        }
        self.collapse_apply(s, t, p, t1, t2);
        Ok(true)
    }

    /// The guards of §116.10's collapse: every `w` in
    /// `N(s) u N(t) \ {s, t}` stays within `(4/3) h` of the survivor's
    /// position `p` (116.14b, with (116.14g)'s `p` in place of `m_ab`),
    /// and every surviving triangle of the two one-rings keeps
    /// non-flat area (116.14k, in the exact-zero test's place) and
    /// turns its normal by less than 90 degrees, and a triangle that is
    /// not a cut triangle does not become one (116.14l). `Ok(false)` is
    /// a refused collapse, never an error.
    fn collapse_guards(
        &self,
        s: u32,
        t: u32,
        p: Point,
        t1: u32,
        t2: u32,
        h: &dyn Fn(Point) -> f64,
    ) -> Result<bool> {
        let ws: BTreeSet<u32> = self.nbrs[s as usize]
            .union(&self.nbrs[t as usize])
            .copied()
            .filter(|&w| w != s && w != t)
            .collect();
        for w in &ws {
            let xw = self.pts[*w as usize];
            let hw = h_checked(h, mid(p, xw))?;
            if norm(sub(p, xw)) > (4.0 / 3.0) * hw {
                return Ok(false);
            }
        }
        let mut ring: BTreeSet<u32> = BTreeSet::new();
        for v in [s, t] {
            for &w in &self.nbrs[v as usize] {
                if let Some(ts) = self.edges.get(&edge_key(v, w)) {
                    ring.extend(ts.iter().copied());
                }
            }
        }
        ring.remove(&t1);
        ring.remove(&t2);
        // (116.14l) with the feature edges as they would be after the
        // collapse: the dying vertex's edges pass to the survivor, so
        // a merged edge at `s` is a feature edge when either of its two
        // parts was (the both-parts case is refused upstream). `t` is
        // already replaced by `s` in the indices `feat` is fed.
        let feat_after = |x: u32, y: u32| -> bool {
            if x == s || y == s {
                let w = if x == s { y } else { x };
                self.feat_idx[s as usize].contains_key(&w)
                    || self.feat_idx[t as usize].contains_key(&w)
            } else {
                self.feat_idx[x as usize].contains_key(&y)
            }
        };
        for tid in &ring {
            let old = self.tri_pts(*tid);
            let old_n = tri_normal(&old);
            let tri = self.tris[*tid as usize];
            let mut new = old;
            let mut post = tri;
            for k in 0..3 {
                let u = tri[k];
                if u == s || u == t {
                    new[k] = p;
                    post[k] = s;
                }
            }
            let new_n = tri_normal(&new);
            // (116.14k): the flat test takes the exact-zero area test's
            // place; the 90 degree test stays.
            if tri_flat(&new) || dot(new_n, old_n) <= 0.0 {
                return Ok(false);
            }
            // (116.14l): a surviving triangle that is not a cut triangle
            // may not become one.
            if !self.tri_is_cut(tri) && self.tri_is_cut_by(post, &feat_after) {
                return Ok(false);
            }
        }
        Ok(true)
    }

    /// Applies the collapse: the survivor `s` takes the position `p`
    /// (a corner is never written; (116.14g) gives it `x_s` there
    /// anyway), the two dead triangles go, every other triangle around
    /// the dying vertex `t` is re-pointed to `s`, and the edge and
    /// neighbour maps follow. The feature edges of `t` pass to `s` with
    /// their polyline and seam flag (116.14g); the refusal has already
    /// ruled out a landing on a feature edge of `s`.
    fn collapse_apply(&mut self, s: u32, t: u32, p: Point, t1: u32, t2: u32) {
        let pass: Vec<(u32, (u32, bool))> = self
            .feat_at(t)
            .into_iter()
            .filter(|(w, _)| *w != s)
            .collect();
        self.feat_remove(s, t);
        for (w, d) in pass {
            self.feat_remove(t, w);
            self.feat_insert(s, w, d);
        }
        for tid in [t1, t2] {
            let tri = self.tris[tid as usize];
            for k in 0..3 {
                self.edge_drop(edge_key(tri[k], tri[(k + 1) % 3]), tid);
            }
            self.live_t[tid as usize] = false;
        }
        let mut b_tris: Vec<u32> = Vec::new();
        for &w in &self.nbrs[t as usize] {
            if let Some(ts) = self.edges.get(&edge_key(t, w)) {
                b_tris.extend(ts.iter().copied());
            }
        }
        b_tris.sort_unstable();
        b_tris.dedup();
        for tid in b_tris {
            let tri = self.tris[tid as usize];
            let others: Vec<u32> =
                (0..3).filter(|&k| tri[k] != t).map(|k| tri[k]).collect();
            self.edge_drop(edge_key(t, others[0]), tid);
            self.edge_drop(edge_key(t, others[1]), tid);
            self.edge_add(edge_key(s, others[0]), tid);
            self.edge_add(edge_key(s, others[1]), tid);
            let mut tri = tri;
            for k in 0..3 {
                if tri[k] == t {
                    tri[k] = s;
                }
            }
            self.tris[tid as usize] = tri;
        }
        let nb: Vec<u32> = self.nbrs[t as usize].iter().copied().collect();
        for w in nb.iter().copied() {
            if w == s {
                continue;
            }
            self.nbrs[w as usize].remove(&t);
            self.nbrs[w as usize].insert(s);
        }
        self.nbrs[s as usize].remove(&t);
        for w in nb.iter().copied() {
            if w != s {
                self.nbrs[s as usize].insert(w);
            }
        }
        self.nbrs[t as usize].clear();
        if !self.corner[s as usize] {
            self.pts[s as usize] = p;
        }
        self.live_v[t as usize] = false;
        self.n_live_v -= 1;
    }

    /// (116.14c)/(116.14h): one pass over the edges in ascending order,
    /// flipping each interior edge whose flip lowers the valence energy
    /// `E = sum (val(v) - val*(v))^2` over `{a, b, c, d}`, with `val*`
    /// 4 at a corner and 6 at any other vertex, and passing the guards:
    /// the edge is not a feature edge, `(c, d)` not already an edge,
    /// neither `val(a)` nor `val(b)` is 3, no new triangle is flat
    /// (116.14k) or a cut triangle (116.14l), and each new triangle's
    /// normal makes less than 90 degrees with the sum of the two old
    /// ones.
    fn flip_pass(&mut self) -> usize {
        let cand: Vec<(u32, u32)> = self.edges.keys().copied().collect();
        let mut n = 0usize;
        for key in cand {
            let (a, b) = key;
            if !self.live_v[a as usize] || !self.live_v[b as usize] {
                continue;
            }
            if self.feat_idx[a as usize].contains_key(&b) {
                continue; // (116.14h): a feature edge is never flipped.
            }
            let (t1, t2) = match self.edges.get(&key) {
                Some(v) if v.len() == 2 => (v[0], v[1]),
                _ => continue,
            };
            let g1 = self.tris[t1 as usize];
            let g2 = self.tris[t2 as usize];
            // The new triangle (a, d, c) replaces whichever old triangle
            // traversed a -> b; the slots follow that role (and the patch
            // with them), in either incidence-list order.
            let (t_ab, t_ba) = if rotated(g1, a, b).is_some() {
                (t1, t2)
            } else {
                (t2, t1)
            };
            let (r1, r2) = match edge_oriented(g1, g2, a, b) {
                Some(pair) => pair,
                None => continue,
            };
            let (c, d) = (r1[2], r2[2]);
            if c == d || self.edges.contains_key(&edge_key(c, d)) {
                continue;
            }
            let val = |v: u32| self.nbrs[v as usize].len() as i64;
            let valstar = |v: u32| if self.corner[v as usize] { 4i64 } else { 6i64 };
            if val(a) == 3 || val(b) == 3 {
                continue;
            }
            let e = (val(a) - valstar(a)).pow(2)
                + (val(b) - valstar(b)).pow(2)
                + (val(c) - valstar(c)).pow(2)
                + (val(d) - valstar(d)).pow(2);
            let e2 = (val(a) - 1 - valstar(a)).pow(2)
                + (val(b) - 1 - valstar(b)).pow(2)
                + (val(c) + 1 - valstar(c)).pow(2)
                + (val(d) + 1 - valstar(d)).pow(2);
            if e2 >= e {
                continue;
            }
            let n1 = tri_normal(&self.tri_pts(t1));
            let n2 = tri_normal(&self.tri_pts(t2));
            let s = add(n1, n2);
            let pa = self.pts[a as usize];
            let pb = self.pts[b as usize];
            let pc = self.pts[c as usize];
            let pd = self.pts[d as usize];
            let m1 = tri_normal(&[pa, pd, pc]);
            let m2 = tri_normal(&[pb, pc, pd]);
            // (116.14k): a flip is refused when a new triangle is flat;
            // (116.14l): a flip is refused when a new triangle would be
            // a cut triangle. Both before the 90 degree tests, whose
            // verdicts the two rules do not change.
            if tri_flat(&[pa, pd, pc]) || tri_flat(&[pb, pc, pd]) {
                continue;
            }
            if self.tri_is_cut([a, d, c]) || self.tri_is_cut([b, c, d]) {
                continue;
            }
            if dot(m1, s) <= 0.0 || dot(m2, s) <= 0.0 {
                continue;
            }
            self.flip(a, b, c, d, t_ab, t_ba);
            n += 1;
        }
        n
    }

    /// Applies the flip: the pair `(a, b, c)`, `(b, a, d)` becomes
    /// `(a, d, c)`, `(b, c, d)`, each triangle keeping its patch, and
    /// the edge `(a, b)` becomes `(c, d)`. `t_ab` is the slot of the old
    /// `a -> b` triangle, `t_ba` that of the old `b -> a` one.
    fn flip(&mut self, a: u32, b: u32, c: u32, d: u32, t_ab: u32, t_ba: u32) {
        self.tris[t_ab as usize] = [a, d, c];
        self.tris[t_ba as usize] = [b, c, d];
        self.edge_drop(edge_key(a, b), t_ab);
        self.edge_drop(edge_key(a, b), t_ba);
        self.edge_drop(edge_key(b, c), t_ab);
        self.edge_add(edge_key(b, c), t_ba);
        self.edge_drop(edge_key(a, d), t_ba);
        self.edge_add(edge_key(a, d), t_ab);
        self.edge_add(edge_key(c, d), t_ab);
        self.edge_add(edge_key(c, d), t_ba);
        self.nbrs[a as usize].remove(&b);
        self.nbrs[b as usize].remove(&a);
        self.nbrs[c as usize].insert(d);
        self.nbrs[d as usize].insert(c);
    }

    /// (116.14)/(116.14i): tangential relaxation in Jacobi order - every
    /// vertex's new position is computed from the old positions before
    /// any moves. A corner stays. A feature vertex goes to the midpoint
    /// of its two feature neighbours' old positions, staying on its
    /// polyline. A free vertex follows (116.14): `x <- x + (I - n n^T)
    /// (c - x)` with `n` the normalised sum of the one-ring's triangle
    /// normals and `c` the area-weighted centroid. Each vertex reads only
    /// the old positions, so the vertices run on `threads` workers through
    /// [`par_map`] (TET-15) with the sequential result.
    fn relax(&mut self, threads: usize) {
        let me = &*self;
        let new_pts = par_map(me.pts.len(), threads, |v| me.relaxed(v));
        self.pts = new_pts;
    }

    /// One vertex of [`Mesh::relax`]: its new position from the old ones.
    fn relaxed(&self, v: usize) -> Point {
        let old = &self.pts;
        if !self.live_v[v] {
            return old[v];
        }
        if self.corner[v] {
            return old[v]; // a corner never moves.
        }
        if self.feat_v[v] {
            let fnbrs: Vec<u32> = self.feat_at(v as u32).into_iter().map(|(w, _)| w).collect();
            if fnbrs.len() == 2 {
                return mid(old[fnbrs[0] as usize], old[fnbrs[1] as usize]);
            }
            return old[v];
        }
        let mut ring: BTreeSet<u32> = BTreeSet::new();
        for &w in &self.nbrs[v] {
            if let Some(ts) = self.edges.get(&edge_key(v as u32, w)) {
                ring.extend(ts.iter().copied());
            }
        }
        if ring.is_empty() {
            return old[v];
        }
        let mut n_sum = [0.0; 3];
        let mut a_sum = 0.0;
        let mut c_sum = [0.0; 3];
        for t in &ring {
            let tri = self.tris[*t as usize];
            let p = [
                old[tri[0] as usize],
                old[tri[1] as usize],
                old[tri[2] as usize],
            ];
            let cr = cross(sub(p[1], p[0]), sub(p[2], p[0]));
            n_sum = add(n_sum, cr);
            let a = norm(cr) * 0.5;
            a_sum += a;
            let g = [
                (p[0][0] + p[1][0] + p[2][0]) / 3.0,
                (p[0][1] + p[1][1] + p[2][1]) / 3.0,
                (p[0][2] + p[1][2] + p[2][2]) / 3.0,
            ];
            c_sum = [
                c_sum[0] + a * g[0],
                c_sum[1] + a * g[1],
                c_sum[2] + a * g[2],
            ];
        }
        if !(a_sum > 0.0) {
            return old[v];
        }
        let nn = norm(n_sum);
        if nn == 0.0 {
            return old[v];
        }
        let n = scale(n_sum, 1.0 / nn);
        let c = scale(c_sum, 1.0 / a_sum);
        let d0 = sub(c, old[v]);
        add(old[v], sub(d0, scale(n, dot(n, d0))))
    }

    /// (116.14d)/(116.14j): every live vertex projected - a corner
    /// stays, a feature vertex of polyline `L_k` onto `L_k`'s input
    /// segments (the first strict minimum over them), a free vertex onto
    /// the input surface through the one `Projector` built on it; among
    /// equally close triangles the smallest index wins. Each vertex moves
    /// alone, so the vertices run on `threads` workers through
    /// [`par_map`] (TET-15) with the sequential result.
    fn project(&mut self, proj: &Projector, polys: &[Vec<(Point, Point)>], threads: usize) {
        let me = &*self;
        let new_pts = par_map(me.pts.len(), threads, |v| {
            let x = me.pts[v];
            if !me.live_v[v] || me.corner[v] {
                x
            } else if me.feat_v[v] {
                closest_on_polyline(x, &polys[me.fpoly[v] as usize])
            } else {
                proj.closest(x).0
            }
        });
        self.pts = new_pts;
    }

    /// Compacts: no unused vertices, vertices renumbered in increasing
    /// old-index order, adjacency rebuilt, and the feature state (the
    /// vertex classes, the per-vertex polyline index, the feature-edge
    /// table) carried across the renumbering.
    fn compact(&mut self) {
        let mut map: Vec<u32> = Vec::with_capacity(self.pts.len());
        let mut corner: Vec<bool> = Vec::new();
        let mut feat_v: Vec<bool> = Vec::new();
        let mut fpoly: Vec<u32> = Vec::new();
        for i in 0..self.pts.len() {
            if self.live_v[i] {
                map.push(corner.len() as u32);
                corner.push(self.corner[i]);
                feat_v.push(self.feat_v[i]);
                fpoly.push(self.fpoly[i]);
            } else {
                map.push(u32::MAX);
            }
        }
        // Each edge once, ascending (min, max), re-inserted after the
        // renumbering in the same order.
        let mut feat: Vec<(u32, u32, (u32, bool))> = Vec::new();
        for v in 0..self.pts.len() as u32 {
            for (&w, &d) in self.feat_idx[v as usize].range((v + 1)..) {
                feat.push((v, w, d));
            }
        }
        let names = Vec::new();
        let surf = self.to_surface(names);
        *self = Mesh::from_surface(&surf);
        self.corner = corner;
        self.feat_v = feat_v;
        self.fpoly = fpoly;
        for (a, b, d) in feat {
            let (ma, mb) = (map[a as usize], map[b as usize]);
            if ma != u32::MAX && mb != u32::MAX {
                self.feat_insert(ma, mb, d);
            }
        }
    }

    /// The live part as a `TetSurface`; `patch_names` comes from the
    /// caller (the output carries the input's).
    fn to_surface(&self, patch_names: Vec<String>) -> TetSurface {
        let mut map: Vec<u32> = Vec::with_capacity(self.pts.len());
        let mut points: Vec<Point> = Vec::new();
        for i in 0..self.pts.len() {
            if self.live_v[i] {
                map.push(points.len() as u32);
                points.push(self.pts[i]);
            } else {
                map.push(u32::MAX);
            }
        }
        let mut tris: Vec<[u32; 3]> = Vec::new();
        let mut tri_patch: Vec<u32> = Vec::new();
        for (t, tri) in self.tris.iter().enumerate() {
            if !self.live_t[t] {
                continue;
            }
            tris.push([
                map[tri[0] as usize],
                map[tri[1] as usize],
                map[tri[2] as usize],
            ]);
            tri_patch.push(self.tri_patch[t]);
        }
        TetSurface {
            points,
            tris,
            tri_patch,
            patch_names,
        }
    }
}

/// The isotropic remesh of §116.10: `iterations` rounds of splits
/// (116.14a, repeated sweeps, at most 32 per iteration), collapses
/// (116.14b with (116.14g)'s survivor table), flips (116.14c/(116.14h)),
/// tangential relaxation (116.14/(116.14i)) in Jacobi order, and
/// projection (116.14d/(116.14j)). Corners never move, feature vertices
/// move only along their input polyline, feature and seam edges are
/// never flipped and never collapsed across. Refuses `iterations == 0`
/// and any `h` value that is not finite and positive, and returns the
/// output only once [`check_manifold`] accepts it. `Remeshed` carries
/// the output's features in the output's numbering.
pub fn remesh(
    input: &PreparedSurface,
    h: &dyn Fn(Point) -> f64,
    opts: &RemeshOptions,
) -> Result<Remeshed> {
    if opts.iterations == 0 {
        return Err(Error::Refused(
            "remesh/iterations: 0 must be at least 1".into(),
        ));
    }
    let proj = Projector::new(&input.surface);
    // (116.14j)'s input: each polyline's consecutive input segments.
    let polys: Vec<Vec<(Point, Point)>> = input
        .polylines
        .iter()
        .map(|chain| {
            chain
                .windows(2)
                .map(|w| {
                    (
                        input.surface.points[w[0] as usize],
                        input.surface.points[w[1] as usize],
                    )
                })
                .collect()
        })
        .collect();
    // The feature state of the input: each polyline edge's k, the
    // feature edges with their (k, seam), the corners and feature
    // vertices with their k.
    let mut edge_poly: BTreeMap<(u32, u32), u32> = BTreeMap::new();
    for (k, chain) in input.polylines.iter().enumerate() {
        for w in chain.windows(2) {
            edge_poly.insert(edge_key(w[0], w[1]), k as u32);
        }
    }
    let mut mesh = Mesh::from_surface(&input.surface);
    for (i, e) in input.feature_edges.iter().enumerate() {
        let k = edge_poly.get(&edge_key(e[0], e[1])).copied().unwrap_or(0);
        mesh.feat_insert(e[0], e[1], (k, input.edge_is_seam[i]));
    }
    for &c in &input.corners {
        mesh.corner[c as usize] = true;
    }
    // Each edge once, in ascending (min, max) order, as the old
    // whole-table walk visited them.
    let mut tops: Vec<(u32, u32, u32)> = Vec::new();
    for v in 0..mesh.pts.len() as u32 {
        for (&w, &(k, _)) in mesh.feat_idx[v as usize].range((v + 1)..) {
            tops.push((v, w, k));
        }
    }
    for (v, w, k) in tops {
        for u in [v, w] {
            if !mesh.corner[u as usize] {
                mesh.feat_v[u as usize] = true;
                mesh.fpoly[u as usize] = k;
            }
        }
    }
    let mut stats = RemeshStats::default();
    for _ in 0..opts.iterations {
        for _ in 0..32 {
            let n = mesh.split_sweep(h)?;
            stats.splits += n;
            if n == 0 {
                break;
            }
        }
        stats.collapses += mesh.collapse_pass(h)?;
        stats.flips += mesh.flip_pass();
        mesh.relax(opts.threads);
        mesh.project(&proj, &polys, opts.threads);
        mesh.compact();
    }
    let surface = mesh.to_surface(input.surface.patch_names.clone());
    check_manifold(&surface)?;
    stats.iterations = opts.iterations;
    stats.n_points = surface.points.len();
    stats.n_tris = surface.tris.len();
    let (feature_edges, edge_is_seam, edge_polyline) = mesh.feature_tables();
    let corners: Vec<u32> = (0..mesh.pts.len() as u32)
        .filter(|&v| mesh.corner[v as usize])
        .collect();
    Ok(Remeshed {
        surface,
        stats,
        feature_edges,
        edge_is_seam,
        edge_polyline,
        corners,
    })
}

// ==========================================================================
//  The measures of §116.18 and the manifold check
// ==========================================================================

/// `l_e / h(m_e)`, one per undirected edge, edges in ascending
/// `(min, max)` vertex-index order.
pub fn edge_length_ratios(s: &TetSurface, h: &dyn Fn(Point) -> f64) -> Vec<f64> {
    let mut keys: BTreeSet<(u32, u32)> = BTreeSet::new();
    for tri in &s.tris {
        for k in 0..3 {
            keys.insert(edge_key(tri[k], tri[(k + 1) % 3]));
        }
    }
    keys.into_iter()
        .map(|(a, b)| {
            let pa = s.points[a as usize];
            let pb = s.points[b as usize];
            norm(sub(pa, pb)) / h(mid(pa, pb))
        })
        .collect()
}

/// The smallest interior angle of each triangle, in degrees, in triangle
/// order.
pub fn min_angles_deg(s: &TetSurface) -> Vec<f64> {
    s.tris.iter()
        .map(|t| {
            let p = [
                s.points[t[0] as usize],
                s.points[t[1] as usize],
                s.points[t[2] as usize],
            ];
            let mut m = 180.0f64;
            for k in 0..3 {
                let u = sub(p[(k + 2) % 3], p[(k + 1) % 3]);
                let w = sub(p[k], p[(k + 1) % 3]);
                let cu = norm(u);
                let cw = norm(w);
                let ang = if cu == 0.0 || cw == 0.0 {
                    0.0
                } else {
                    (dot(u, w) / (cu * cw)).clamp(-1.0, 1.0).acos()
                        * 180.0
                        / std::f64::consts::PI
                };
                m = m.min(ang);
            }
            m
        })
        .collect()
}

/// (116.14e): max over the vertices of `s` of the distance from the
/// vertex to the input surface.
pub fn vertex_hausdorff(s: &TetSurface, input: &Projector) -> f64 {
    s.points
        .iter()
        .fold(0.0f64, |m, p| m.max(input.closest(*p).2))
}

/// Two-sided sampled distance: each triangle is sampled at its three
/// vertices, its three edge midpoints and its centroid; the answer is
/// the larger of the two one-sided maxima, each through a [`Projector`].
pub fn sampled_hausdorff(a: &TetSurface, b: &TetSurface) -> f64 {
    one_sided(&Projector::new(b), a).max(one_sided(&Projector::new(a), b))
}

/// max over the samples of `s` of the distance to the surface `proj`
/// was built on.
fn one_sided(proj: &Projector, s: &TetSurface) -> f64 {
    s.tris.iter().fold(0.0f64, |m, t| {
        let p = [
            s.points[t[0] as usize],
            s.points[t[1] as usize],
            s.points[t[2] as usize],
        ];
        let mut mm = m;
        for k in 0..3 {
            mm = mm.max(proj.closest(p[k]).2);
        }
        for k in 0..3 {
            let q = p[k];
            let r = p[(k + 1) % 3];
            mm = mm.max(proj.closest(mid(q, r)).2);
        }
        let cen = [
            (p[0][0] + p[1][0] + p[2][0]) / 3.0,
            (p[0][1] + p[1][1] + p[2][1]) / 3.0,
            (p[0][2] + p[1][2] + p[2][2]) / 3.0,
        ];
        mm.max(proj.closest(cen).2)
    })
}

/// What [`check_manifold`] measured on success.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ManifoldReport {
    pub n_points: usize,
    pub n_edges: usize,
    pub n_tris: usize,
    pub euler: i64,
}

fn dsu_find(parent: &mut Vec<usize>, mut x: usize) -> usize {
    while parent[x] != x {
        parent[x] = parent[parent[x]];
        x = parent[x];
    }
    x
}

fn dsu_union(parent: &mut Vec<usize>, a: usize, b: usize) {
    let ra = dsu_find(parent, a);
    let rb = dsu_find(parent, b);
    if ra != rb {
        parent[ra.max(rb)] = ra.min(rb);
    }
}

/// The closed-manifold check of §116.10: the five checks in order, the
/// first failure wins, every failure a named `Error::Refused`.
pub fn check_manifold(s: &TetSurface) -> Result<ManifoldReport> {
    // 1. no triangle out of range or with a repeated index.
    for (t, tri) in s.tris.iter().enumerate() {
        let repeated = tri[0] == tri[1] || tri[1] == tri[2] || tri[0] == tri[2];
        let out = tri.iter().any(|&i| i as usize >= s.points.len());
        if repeated || out {
            return Err(Error::Refused(format!(
                "remesh/manifold: triangle {t} is degenerate"
            )));
        }
    }
    // 2./3. the undirected edges, ascending (min, max): exactly two
    // triangles each, traversed in opposite directions.
    let mut edges: BTreeMap<(u32, u32), Vec<(u32, bool)>> = BTreeMap::new();
    for (t, tri) in s.tris.iter().enumerate() {
        for k in 0..3 {
            let a = tri[k];
            let b = tri[(k + 1) % 3];
            let (key, fwd) = if a < b { ((a, b), true) } else { ((b, a), false) };
            edges.entry(key).or_default().push((t as u32, fwd));
        }
    }
    for (&(a, b), inc) in &edges {
        if inc.len() != 2 {
            return Err(Error::Refused(format!(
                "remesh/manifold: edge ({a}, {b}) is on {} triangle(s)",
                inc.len()
            )));
        }
    }
    for (&(a, b), inc) in &edges {
        if inc[0].1 == inc[1].1 {
            return Err(Error::Refused(format!(
                "remesh/manifold: edge ({a}, {b}) is used twice in the same direction"
            )));
        }
    }
    // 4. the triangles around each vertex form one fan cycle.
    let mut vtris: Vec<Vec<u32>> = vec![Vec::new(); s.points.len()];
    for (t, tri) in s.tris.iter().enumerate() {
        for &i in tri {
            vtris[i as usize].push(t as u32);
        }
    }
    for v in 0..s.points.len() {
        if vtris[v].is_empty() {
            continue;
        }
        let mut ids: BTreeMap<u32, usize> = BTreeMap::new();
        let mut parent: Vec<usize> = Vec::new();
        for &t in &vtris[v] {
            let tri = s.tris[t as usize];
            let others: Vec<u32> = (0..3)
                .filter(|&k| tri[k] as usize != v)
                .map(|k| tri[k])
                .collect();
            for &w in &others {
                let n = ids.len();
                let e = ids.entry(w).or_insert(n);
                if *e == n {
                    parent.push(n);
                }
            }
            let ia = ids[&others[0]];
            let ib = ids[&others[1]];
            dsu_union(&mut parent, ia, ib);
        }
        let mut roots: Vec<usize> =
            (0..parent.len()).map(|i| dsu_find(&mut parent, i)).collect();
        roots.sort_unstable();
        roots.dedup();
        if roots.len() > 1 {
            return Err(Error::Refused(format!(
                "remesh/manifold: vertex {v} has a link of {} cycles",
                roots.len()
            )));
        }
    }
    // 5. no vertex off every triangle.
    for (v, ts) in vtris.iter().enumerate() {
        if ts.is_empty() {
            return Err(Error::Refused(format!(
                "remesh/manifold: vertex {v} is on no triangle"
            )));
        }
    }
    Ok(ManifoldReport {
        n_points: s.points.len(),
        n_edges: edges.len(),
        n_tris: s.tris.len(),
        euler: s.points.len() as i64 - edges.len() as i64 + s.tris.len() as i64,
    })
}

#[cfg(test)]
mod tests {
    use super::super::sizefield::point_triangle_distance;
    use super::super::sizefield::{
        OctreeOptions, SizeField, SizeParams, SizeSource,
    };
    use super::super::surfprep::prepare;
    use super::*;
    use std::f64::consts::PI;
    use std::sync::OnceLock;

    /// SplitMix64, Steele, Lea & Flood 2014: every operation wrapping.
    struct SplitMix64 {
        state: u64,
    }

    impl SplitMix64 {
        fn new(seed: u64) -> SplitMix64 {
            SplitMix64 { state: seed }
        }

        fn next(&mut self) -> u64 {
            self.state = self.state.wrapping_add(0x9E3779B97F4A7C15);
            let mut z = self.state;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
            z ^ (z >> 31)
        }

        /// Uniform in [0, 1).
        fn unit(&mut self) -> f64 {
            (self.next() >> 11) as f64 / 9007199254740992.0
        }

        fn uni(&mut self, lo: f64, hi: f64) -> f64 {
            lo + (hi - lo) * self.unit()
        }

        /// A uniform point in the box: x, then y, then z.
        fn point(&mut self, lo: Point, hi: Point) -> Point {
            [
                self.uni(lo[0], hi[0]),
                self.uni(lo[1], hi[1]),
                self.uni(lo[2], hi[2]),
            ]
        }
    }

    /// One midpoint per unordered pair: the normalised sum, cached.
    fn ico_mid(
        pts: &mut Vec<Point>,
        cache: &mut BTreeMap<(u32, u32), u32>,
        a: u32,
        b: u32,
    ) -> u32 {
        let key = if a < b { (a, b) } else { (b, a) };
        if let Some(&m) = cache.get(&key) {
            return m;
        }
        let m = [
            pts[a as usize][0] + pts[b as usize][0],
            pts[a as usize][1] + pts[b as usize][1],
            pts[a as usize][2] + pts[b as usize][2],
        ];
        let n = norm(m);
        let id = pts.len() as u32;
        pts.push([m[0] / n, m[1] / n, m[2] / n]);
        cache.insert(key, id);
        id
    }

    /// The icosphere at `level` subdivisions, on the unit sphere: level 5
    /// is the SPHERE fixture, 10242 points and 20480 triangles.
    fn ico(level: u32) -> (Vec<Point>, Vec<[u32; 3]>) {
        let t = (1.0 + 5.0f64.sqrt()) / 2.0;
        let v: [Point; 12] = [
            [-1.0, t, 0.0], [1.0, t, 0.0], [-1.0, -t, 0.0], [1.0, -t, 0.0],
            [0.0, -1.0, t], [0.0, 1.0, t], [0.0, -1.0, -t], [0.0, 1.0, -t],
            [t, 0.0, -1.0], [t, 0.0, 1.0], [-t, 0.0, -1.0], [-t, 0.0, 1.0],
        ];
        let f: [[u32; 3]; 20] = [
            [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
            [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
            [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
            [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1],
        ];
        let mut pts: Vec<Point> = v.iter().map(|&p| {
            let n = norm(p);
            [p[0] / n, p[1] / n, p[2] / n]
        }).collect();
        let mut tris: Vec<[u32; 3]> =
            f.iter().map(|g| [g[0], g[1], g[2]]).collect();
        for _ in 0..level {
            let mut cache: BTreeMap<(u32, u32), u32> = BTreeMap::new();
            let mut next: Vec<[u32; 3]> = Vec::with_capacity(tris.len() * 4);
            for [a, b, c] in tris.clone() {
                let ab = ico_mid(&mut pts, &mut cache, a, b);
                let bc = ico_mid(&mut pts, &mut cache, b, c);
                let ca = ico_mid(&mut pts, &mut cache, c, a);
                next.push([a, ab, ca]);
                next.push([b, bc, ab]);
                next.push([c, ca, bc]);
                next.push([ab, bc, ca]);
            }
            tris = next;
        }
        (pts, tris)
    }

    /// The SPHERE input: level-5 icosphere of radius 1, one patch.
    fn sphere_input() -> &'static TetSurface {
        static S: OnceLock<TetSurface> = OnceLock::new();
        S.get_or_init(|| {
            let (pts, tris) = ico(5);
            assert_eq!(pts.len(), 10242);
            assert_eq!(tris.len(), 20480);
            TetSurface {
                points: pts,
                tris,
                tri_patch: vec![0; 20480],
                patch_names: vec!["sphere".to_string()],
            }
        })
    }

    /// SPHERE through `prepare(.., 60.0)`, once.
    fn sphere_prepared() -> &'static PreparedSurface {
        static P: OnceLock<PreparedSurface> = OnceLock::new();
        P.get_or_init(|| prepare(sphere_input(), 60.0).unwrap())
    }

    /// SPHERE remeshed once at the constant h = 0.05 with the default
    /// options, shared across the tests.
    fn sphere_out() -> &'static Remeshed {
        static R: OnceLock<Remeshed> = OnceLock::new();
        R.get_or_init(|| {
            remesh(
                sphere_prepared(),
                &|_: Point| 0.05,
                &RemeshOptions::default(),
            )
            .unwrap()
        })
    }

    /// The TORUS input: R 1, r 0.4, nu 160 x nv 64, outward.
    fn torus_input() -> &'static TetSurface {
        static T: OnceLock<TetSurface> = OnceLock::new();
        T.get_or_init(|| {
            let (r_big, r_small, nu, nv) = (1.0f64, 0.4f64, 160u32, 64u32);
            let pt = |i: u32, j: u32| {
                let u = 2.0 * PI * i as f64 / nu as f64;
                let v = 2.0 * PI * j as f64 / nv as f64;
                [
                    (r_big + r_small * v.cos()) * u.cos(),
                    (r_big + r_small * v.cos()) * u.sin(),
                    r_small * v.sin(),
                ]
            };
            let mut points = Vec::new();
            for i in 0..nu {
                for j in 0..nv {
                    points.push(pt(i, j));
                }
            }
            let idx = |i: u32, j: u32| (i % nu) * nv + (j % nv);
            let mut tris: Vec<[u32; 3]> = Vec::new();
            for i in 0..nu {
                for j in 0..nv {
                    let (p00, p10, p11, p01) =
                        (idx(i, j), idx(i + 1, j), idx(i + 1, j + 1), idx(i, j + 1));
                    tris.push([p00, p10, p11]);
                    tris.push([p00, p11, p01]);
                }
            }
            let nt = tris.len();
            TetSurface {
                points,
                tris,
                tri_patch: vec![0; nt],
                patch_names: vec!["torus".to_string()],
            }
        })
    }

    fn band_frac(ratios: &[f64]) -> f64 {
        let n = ratios.len();
        let k = ratios.iter().filter(|&&r| r >= 0.8 && r <= 4.0 / 3.0).count();
        k as f64 / n as f64
    }

    /// A coarse-sphere regression: the level-3 icosphere remeshed at
    /// h = 0.25 for 3 iterations stays a closed manifold of Euler 2 that
    /// `prepare` accepts. This caught the edge-map keys that landed under
    /// an unsorted `(max, min)` pair after a split.
    #[test]
    fn coarse_sphere_regression() {
        let (pts, tris) = ico(3);
        let surf = TetSurface {
            points: pts,
            tri_patch: vec![0; tris.len()],
            tris,
            patch_names: vec!["sphere".to_string()],
        };
        let prep = prepare(&surf, 60.0).unwrap();
        let out = remesh(
            &prep,
            &|_: Point| 0.25,
            &RemeshOptions { iterations: 3, threads: 0 },
        )
        .unwrap();
        let rep = check_manifold(&out.surface).unwrap();
        assert_eq!(rep.euler, 2);
        assert!(prepare(&out.surface, 60.0).is_ok());
        println!(
            "coarse_sphere_regression: points {}, tris {}, splits {}, collapses {}, flips {}",
            out.stats.n_points,
            out.stats.n_tris,
            out.stats.splits,
            out.stats.collapses,
            out.stats.flips
        );
    }


    #[test]
    fn sphere_lengths() {
        let out = sphere_out();
        let ratios = edge_length_ratios(&out.surface, &|_: Point| 0.05);
        let n = ratios.len();
        let frac = band_frac(&ratios);
        let min = ratios.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = ratios.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let mean = ratios.iter().sum::<f64>() / n as f64;
        println!(
            "sphere_lengths: fraction {frac:.6}, min {min:.4}, max {max:.4}, mean {mean:.4}, edges {n}"
        );
        assert!(frac >= 0.95, "in-band fraction {frac} below 0.95");
    }

    #[test]
    fn sphere_angles() {
        let out = sphere_out();
        let angs = min_angles_deg(&out.surface);
        let n = angs.len();
        let k = angs.iter().filter(|&&a| a >= 25.0).count();
        let frac = k as f64 / n as f64;
        let smallest = angs.iter().cloned().fold(f64::INFINITY, f64::min);
        println!(
            "sphere_angles: fraction {frac:.6}, smallest {smallest:.3} deg, triangles {n}"
        );
        assert!(frac >= 0.99, "angle fraction {frac} below 0.99");
    }

    #[test]
    fn sphere_hausdorff() {
        let out = sphere_out();
        let proj = Projector::new(sphere_input());
        let vh = vertex_hausdorff(&out.surface, &proj);
        println!("sphere_hausdorff: vertex {vh:.3e} bar {}", 0.01 * 0.05);
        assert!(vh <= 0.01 * 0.05, "vertex hausdorff {vh} above bar");
        let sh = sampled_hausdorff(&out.surface, sphere_input()) / 0.05;
        println!("sphere_hausdorff: sampled/h = {sh:.6} (measured, not gated)");
    }

    #[test]
    fn stays_closed() {
        let out = sphere_out();
        let rep = check_manifold(&out.surface).unwrap();
        assert_eq!(rep.euler, 2);
        let p = prepare(&out.surface, 60.0).unwrap();
        assert_eq!(p.surface.tris.len(), out.surface.tris.len());
        assert!(out.surface.tri_patch.iter().all(|&q| q == 0));
        assert_eq!(out.stats.n_points, out.surface.points.len());
        assert_eq!(out.stats.n_tris, out.surface.tris.len());
        println!(
            "stays_closed: points {}, tris {}, splits {}, collapses {}, flips {}",
            out.stats.n_points,
            out.stats.n_tris,
            out.stats.splits,
            out.stats.collapses,
            out.stats.flips
        );
    }

    #[test]
    fn torus_stays_closed() {
        let tin = torus_input();
        let prep = prepare(tin, 60.0).unwrap();
        let out = remesh(&prep, &|_: Point| 0.05, &RemeshOptions::default()).unwrap();
        let rep = check_manifold(&out.surface).unwrap();
        assert_eq!(rep.euler, 0);
        assert!(prepare(&out.surface, 60.0).is_ok());
        let ratios = edge_length_ratios(&out.surface, &|_: Point| 0.05);
        let frac = band_frac(&ratios);
        let angs = min_angles_deg(&out.surface);
        let afrac = angs.iter().filter(|&&a| a >= 25.0).count() as f64 / angs.len() as f64;
        println!(
            "torus_stays_closed: length fraction {frac:.6}, angle fraction {afrac:.6}, points {}, tris {}",
            out.stats.n_points,
            out.stats.n_tris
        );
    }

    #[test]
    fn graded_lengths() {
        let field = SizeField::build(
            SizeParams::new(0.1, 1.2, None).unwrap(),
            &[SizeSource::Point {
                at: [1.0, 0.0, 0.0],
                size: 0.025,
            }],
            [-2.0, -2.0, -2.0],
            [2.0, 2.0, 2.0],
            OctreeOptions::default(),
        )
        .unwrap();
        let hfun = |x: Point| field.h(x);
        let out = remesh(sphere_prepared(), &hfun, &RemeshOptions::default()).unwrap();
        let ratios = edge_length_ratios(&out.surface, &hfun);
        let frac = band_frac(&ratios);
        println!("graded_lengths: fraction {frac:.6}, edges {}", ratios.len());
        assert!(frac >= 0.95, "in-band fraction {frac} below 0.95");
        let rep = check_manifold(&out.surface).unwrap();
        assert_eq!(rep.euler, 2);
    }

    #[test]
    fn deterministic() {
        let a = remesh(
            sphere_prepared(),
            &|_: Point| 0.05,
            &RemeshOptions::default(),
        )
        .unwrap();
        let b = remesh(
            sphere_prepared(),
            &|_: Point| 0.05,
            &RemeshOptions::default(),
        )
        .unwrap();
        assert_eq!(a.surface.points.len(), b.surface.points.len());
        for (p, q) in a.surface.points.iter().zip(b.surface.points.iter()) {
            for k in 0..3 {
                assert_eq!(p[k].to_bits(), q[k].to_bits());
            }
        }
        assert_eq!(a.surface.tris, b.surface.tris);
        assert_eq!(a.surface.tri_patch, b.surface.tri_patch);
    }

    #[test]
    fn refusals() {
        let err = remesh(
            sphere_prepared(),
            &|_: Point| 0.0,
            &RemeshOptions::default(),
        )
        .unwrap_err();
        assert!(err.to_string().contains("remesh/size: h("), "got {err}");
        let err = remesh(
            sphere_prepared(),
            &|_: Point| f64::NAN,
            &RemeshOptions::default(),
        )
        .unwrap_err();
        assert!(err.to_string().contains("remesh/size: h("), "got {err}");
        let err = remesh(
            sphere_prepared(),
            &|_: Point| 0.05,
            &RemeshOptions { iterations: 0, threads: 0 },
        )
        .unwrap_err();
        assert!(
            err.to_string().contains("remesh/iterations"),
            "got {err}"
        );
    }

    #[test]
    fn manifold_refusals() {
        let pts: Vec<Point> = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ];
        let surf = |ps: &[Point], tris: &[[u32; 3]]| TetSurface {
            points: ps.to_vec(),
            tris: tris.to_vec(),
            tri_patch: vec![0; tris.len()],
            patch_names: vec!["tet".to_string()],
        };
        let full: [[u32; 3]; 4] = [[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]];
        let rep = check_manifold(&surf(&pts, &full)).unwrap();
        assert_eq!(rep.n_points, 4);
        assert_eq!(rep.n_edges, 6);
        assert_eq!(rep.n_tris, 4);
        assert_eq!(rep.euler, 2);
        let refused = |r: Result<ManifoldReport>| match r {
            Err(Error::Refused(m)) => m,
            other => panic!("expected a refusal, got {other:?}"),
        };
        let err = refused(check_manifold(&surf(&pts, &full[..3])));
        assert_eq!(err, "remesh/manifold: edge (1, 2) is on 1 triangle(s)");
        let flipped: [[u32; 3]; 4] = [[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 3, 2]];
        let err = refused(check_manifold(&surf(&pts, &flipped)));
        assert_eq!(
            err,
            "remesh/manifold: edge (1, 2) is used twice in the same direction"
        );
        // Bowtie: the tetrahedron plus a copy translated by (0, 0, 1)
        // whose first vertex IS point 3.
        let mut bpts = pts.clone();
        bpts.push([1.0, 0.0, 1.0]);
        bpts.push([0.0, 1.0, 1.0]);
        bpts.push([0.0, 0.0, 2.0]);
        let mut btris = full.to_vec();
        btris.push([3, 5, 4]);
        btris.push([3, 4, 6]);
        btris.push([3, 6, 5]);
        btris.push([4, 5, 6]);
        let err = refused(check_manifold(&surf(&bpts, &btris)));
        assert_eq!(err, "remesh/manifold: vertex 3 has a link of 2 cycles");
        // A repeated index.
        let mut dtris = full.to_vec();
        dtris.push([0, 0, 1]);
        let err = refused(check_manifold(&surf(&pts, &dtris)));
        assert_eq!(err, "remesh/manifold: triangle 4 is degenerate");
        // An unused fifth point.
        let mut upts = pts.clone();
        upts.push([0.0, 0.0, -1.0]);
        let err = refused(check_manifold(&surf(&upts, &full)));
        assert_eq!(err, "remesh/manifold: vertex 4 is on no triangle");
    }

    #[test]
    fn projector_matches_brute() {
        for surf in [sphere_input(), torus_input()] {
            let proj = Projector::new(surf);
            let mut rng = SplitMix64::new(2026);
            for _ in 0..10_000 {
                let x = rng.point([-1.5, -1.5, -1.5], [1.5, 1.5, 1.5]);
                let (p, ti, d) = proj.closest(x);
                let (pb, tib, db) = proj.closest_brute(x);
                for k in 0..3 {
                    assert_eq!(p[k].to_bits(), pb[k].to_bits(), "x {x:?}");
                }
                assert_eq!(ti, tib, "x {x:?}");
                assert_eq!(d.to_bits(), db.to_bits(), "x {x:?}");
            }
        }
    }

    #[test]
    fn closest_point_agrees_with_distance() {
        let mut rng = SplitMix64::new(2027);
        let mut cases: Vec<(Point, [Point; 3])> = Vec::new();
        for _ in 0..10_000 {
            let t = [
                rng.point([-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]),
                rng.point([-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]),
                rng.point([-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]),
            ];
            let x = rng.point([-2.0, -2.0, -2.0], [2.0, 2.0, 2.0]);
            cases.push((x, t));
        }
        for _ in 0..100 {
            let a = rng.point([-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]);
            let t = [a, a, rng.point([-1.0, -1.0, -1.0], [1.0, 1.0, 1.0])];
            let x = rng.point([-2.0, -2.0, -2.0], [2.0, 2.0, 2.0]);
            cases.push((x, t));
        }
        for _ in 0..100 {
            let a = rng.point([-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]);
            let b = rng.point([-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]);
            let m = [(a[0] + b[0]) * 0.5, (a[1] + b[1]) * 0.5, (a[2] + b[2]) * 0.5];
            let t = [a, b, m];
            let x = rng.point([-2.0, -2.0, -2.0], [2.0, 2.0, 2.0]);
            cases.push((x, t));
        }
        for (x, t) in &cases {
            let p = closest_point_on_triangle(*x, t);
            let d = norm(sub(*x, p));
            let want = point_triangle_distance(*x, t);
            assert_eq!(d.to_bits(), want.to_bits(), "x {x:?} t {t:?}");
        }
    }

    // ==================================================================
    //  TET-06b: features, corners and seams survive the remesh
    // ==================================================================

    /// FNV-1a 64 over a surface's points (each coordinate's bits, little
    /// endian), triangles (each index, little endian) and patches.
    fn fnv1a(s: &TetSurface) -> u64 {
        let mut h: u64 = 0xcbf29ce484222325;
        for p in &s.points {
            for k in 0..3 {
                for b in p[k].to_bits().to_le_bytes() {
                    h ^= b as u64;
                    h = h.wrapping_mul(0x100000001b3);
                }
            }
        }
        for t in &s.tris {
            for k in 0..3 {
                for b in t[k].to_le_bytes() {
                    h ^= b as u64;
                    h = h.wrapping_mul(0x100000001b3);
                }
            }
        }
        for q in &s.tri_patch {
            for b in q.to_le_bytes() {
                h ^= b as u64;
                h = h.wrapping_mul(0x100000001b3);
            }
        }
        h
    }

    /// The distance from `x` to the segment `p q`.
    fn seg_dist(x: Point, p: Point, q: Point) -> f64 {
        let pq = sub(q, p);
        let l2 = dot(pq, pq);
        let t = if l2 == 0.0 {
            0.0
        } else {
            (dot(sub(x, p), pq) / l2).clamp(0.0, 1.0)
        };
        norm(sub(x, add(p, scale(pq, t))))
    }

    /// The patch's area: the sum of the triangle areas, in triangle
    /// order.
    fn patch_area(s: &TetSurface, patch: u32) -> f64 {
        let mut a = 0.0f64;
        for (t, &p) in s.tris.iter().zip(&s.tri_patch) {
            if p != patch {
                continue;
            }
            let x = [
                s.points[t[0] as usize],
                s.points[t[1] as usize],
                s.points[t[2] as usize],
            ];
            a += norm(cross(sub(x[1], x[0]), sub(x[2], x[0]))) * 0.5;
        }
        a
    }

    /// The unit cube's 8 points: point k has coordinate 1.0 on axis q
    /// when bit q of k is set, else 0.0.
    fn cube_points() -> Vec<Point> {
        (0..8u32)
            .map(|k| {
                [
                    (k & 1) as f64,
                    ((k >> 1) & 1) as f64,
                    ((k >> 2) & 1) as f64,
                ]
            })
            .collect()
    }

    /// The cube surface: one patch "box", or a patch per face
    /// ("xmin" .. "zmax", patch t / 2 for triangle t).
    fn cube_surface(per_face: bool) -> TetSurface {
        let points = cube_points();
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
        let (tri_patch, names): (Vec<u32>, Vec<String>) = if per_face {
            (
                (0..12u32).map(|t| t / 2).collect(),
                ["xmin", "xmax", "ymin", "ymax", "zmin", "zmax"]
                    .iter()
                    .map(|s| s.to_string())
                    .collect(),
            )
        } else {
            (vec![0; 12], vec!["box".to_string()])
        };
        TetSurface {
            points,
            tris,
            tri_patch,
            patch_names: names,
        }
    }

    /// The capped 16-sided cylinder: two apex points, three rings, the
    /// bottom and top fans first, then the two side bands. Patch 0
    /// ("lower") on the bottom fan and the lower band, patch 1 ("upper")
    /// on the top fan and the upper band.
    fn cyl2() -> TetSurface {
        let mut points = vec![[0.0, 0.0, -1.0], [0.0, 0.0, 1.0]];
        for r in 0..3u32 {
            let z = match r {
                0 => -1.0,
                1 => 0.0,
                _ => 1.0,
            };
            for k in 0..16u32 {
                let t = 2.0 * PI * k as f64 / 16.0;
                points.push([t.cos(), t.sin(), z]);
            }
        }
        let ring = |r: u32, k: u32| 2 + 16 * r + (k % 16);
        let mut tris: Vec<[u32; 3]> = Vec::new();
        for k in 0..16u32 {
            tris.push([0, ring(0, k + 1), ring(0, k)]);
        }
        for k in 0..16u32 {
            tris.push([1, ring(2, k), ring(2, k + 1)]);
        }
        for r in 0..2u32 {
            for k in 0..16u32 {
                let (a, b, c, d) =
                    (ring(r, k), ring(r, k + 1), ring(r + 1, k + 1), ring(r + 1, k));
                tris.push([a, b, c]);
                tris.push([a, c, d]);
            }
        }
        let tri_patch: Vec<u32> = (0..96u32)
            .map(|t| if (16..32).contains(&t) || t >= 64 { 1 } else { 0 })
            .collect();
        TetSurface {
            points,
            tris,
            tri_patch,
            patch_names: vec!["lower".to_string(), "upper".to_string()],
        }
    }

    /// The 30 degree wedge: the prism over the triangle (0,0), (1,0),
    /// (cos 30, sin 30) from z = 0 to z = 1, one patch.
    fn wedge() -> TetSurface {
        let c = 30f64.to_radians().cos();
        let s = 30f64.to_radians().sin();
        let points: Vec<Point> = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [c, s, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [c, s, 1.0],
        ];
        let tris: Vec<[u32; 3]> = vec![
            [0, 2, 1],
            [3, 4, 5],
            [0, 1, 4],
            [0, 4, 3],
            [1, 2, 5],
            [1, 5, 4],
            [2, 0, 3],
            [2, 3, 5],
        ];
        TetSurface {
            points,
            tris,
            tri_patch: vec![0; 8],
            patch_names: vec!["wedge".to_string()],
        }
    }

    /// The cube turned by Rz(0.3) Ry(0.5) Rx(0.7), one patch.
    fn rcube() -> TetSurface {
        let (ga, gb, gg) = (0.7f64, 0.5f64, 0.3f64);
        let (ca, sa) = (ga.cos(), ga.sin());
        let (cb, sb) = (gb.cos(), gb.sin());
        let (cg, sg) = (gg.cos(), gg.sin());
        let points: Vec<Point> = cube_points()
            .into_iter()
            .map(|p| {
                let (x, y, z) = (p[0], p[1], p[2]);
                let (y, z) = (y * ca - z * sa, y * sa + z * ca);
                let (x, z) = (x * cb + z * sb, -x * sb + z * cb);
                let (x, y) = (x * cg - y * sg, x * sg + y * cg);
                [x, y, z]
            })
            .collect();
        let base = cube_surface(false);
        TetSurface {
            points,
            tris: base.tris,
            tri_patch: base.tri_patch,
            patch_names: base.patch_names,
        }
    }

    /// The assertions the two cube fixtures share (TET-06b's cube gate):
    /// a closed manifold of Euler 2, 8 corners and 12 open polylines in
    /// the re-extraction, the 8 input points bit for bit, the output's
    /// feature tables equal to the re-extraction's, and every polyline
    /// running along one cube edge with 8..=12 edges in band.
    fn cube_gate(input: &TetSurface, out: &Remeshed, per_face: bool) {
        let q = prepare(&out.surface, 60.0).unwrap();
        let rep = check_manifold(&out.surface).unwrap();
        assert_eq!(rep.euler, 2, "euler");
        assert_eq!(q.report.n_corners, 8, "corners");
        assert_eq!(q.report.n_polylines, 12, "polylines");
        assert_eq!(q.report.n_closed_polylines, 0, "closed polylines");
        for k in 0..8 {
            for d in 0..3 {
                assert_eq!(
                    out.surface.points[k][d].to_bits(),
                    input.points[k][d].to_bits(),
                    "point {k} coordinate {d}"
                );
            }
        }
        assert_eq!(out.corners, vec![0, 1, 2, 3, 4, 5, 6, 7], "out corners");
        assert_eq!(out.corners, q.corners, "corners equal the re-extraction");
        assert_eq!(out.feature_edges, q.feature_edges, "feature edges");
        assert_eq!(out.edge_is_seam, q.edge_is_seam, "seam flags");
        let all_seam = out.edge_is_seam.iter().all(|&s| s);
        let no_seam = out.edge_is_seam.iter().all(|&s| !s);
        assert!(if per_face { all_seam } else { no_seam }, "seam flags");
        assert_eq!(out.edge_polyline.len(), out.feature_edges.len());
        assert!(out.edge_polyline.iter().all(|&k| k < 12), "polyline k");
        let mut pairs: BTreeSet<(u32, u32)> = BTreeSet::new();
        for chain in &q.polylines {
            let (i, j) = (chain[0], *chain.last().unwrap());
            assert!(i < 8 && j < 8, "chain ends {i} {j}");
            let x = i ^ j;
            assert!(x == 1 || x == 2 || x == 4, "ends {i} {j} not one bit");
            let q0 = x.trailing_zeros() as usize;
            let p0 = out.surface.points[i as usize];
            assert_eq!(p0[q0], 0.0, "first corner on the axis");
            let mut prev = -1.0f64;
            for &v in chain {
                let p = out.surface.points[v as usize];
                for d in 0..3 {
                    if d != q0 {
                        assert_eq!(p[d].to_bits(), p0[d].to_bits(), "off-axis");
                    }
                }
                assert!(p[q0] > prev, "not increasing");
                assert!(p[q0] <= 1.0, "past the far corner");
                prev = p[q0];
            }
            assert_eq!(prev, 1.0, "far corner at 1.0");
            let edges = chain.len() - 1;
            assert!((8..=12).contains(&edges), "chain of {edges} edges");
            pairs.insert((i, j));
        }
        assert_eq!(pairs.len(), 12, "12 different cube edges");
        for [a, b] in &out.feature_edges {
            let pa = out.surface.points[*a as usize];
            let pb = out.surface.points[*b as usize];
            let r = norm(sub(pa, pb)) / 0.1;
            assert!(r >= 0.8 && r <= 4.0 / 3.0, "edge ratio {r}");
        }
    }

    /// TET-06b's cube gate: the feature edges, corners and patch seams
    /// of the unit cube at h = 0.1 survive the remesh, once with one
    /// patch and once with a patch per face.
    #[test]
    fn cube_features_exact() {
        for per_face in [false, true] {
            let input = cube_surface(per_face);
            let prep = prepare(&input, 60.0).unwrap();
            let out = remesh(&prep, &|_: Point| 0.1, &RemeshOptions::default()).unwrap();
            cube_gate(&input, &out, per_face);
            let per_edge: BTreeMap<u32, usize> =
                out.edge_polyline.iter().fold(BTreeMap::new(), |mut m, &k| {
                    *m.entry(k).or_default() += 1;
                    m
                });
            let ratios = edge_length_ratios(&out.surface, &|_: Point| 0.1);
            let frac = band_frac(&ratios);
            let angs = min_angles_deg(&out.surface);
            let afrac = angs.iter().filter(|&&a| a >= 25.0).count() as f64
                / angs.len() as f64;
            println!(
                "cube_features_exact(per_face {per_face}): points {}, tris {}, \
                 splits {}, collapses {}, flips {}, edges per cube edge {:?}, \
                 band_frac {frac:.6}, angle frac {afrac:.6}",
                out.stats.n_points,
                out.stats.n_tris,
                out.stats.splits,
                out.stats.collapses,
                out.stats.flips,
                per_edge,
            );
        }
    }

    /// TET-06b's patch-area gate: with a patch per face each patch's
    /// area equals the input's within 1e-12 relative and every triangle
    /// of a face's patch sits bit for bit on that face; with one patch
    /// the total area is 6 within 6e-12.
    #[test]
    fn patch_area_kept() {
        let input = cube_surface(true);
        let prep = prepare(&input, 60.0).unwrap();
        let out = remesh(&prep, &|_: Point| 0.1, &RemeshOptions::default()).unwrap();
        let names = ["x", "y", "z"];
        for p in 0..6u32 {
            let ia = patch_area(&input, p);
            let oa = patch_area(&out.surface, p);
            assert!(
                (oa - ia).abs() <= 1e-12 * ia,
                "patch {p}: {oa} vs input {ia}"
            );
            let (axis, val) = (p as usize / 2, if p % 2 == 0 { 0.0f64 } else { 1.0 });
            for (ti, (t, &tp)) in
                out.surface.tris.iter().zip(&out.surface.tri_patch).enumerate()
            {
                if tp != p {
                    continue;
                }
                for k in 0..3 {
                    assert_eq!(
                        out.surface.points[t[k] as usize][axis].to_bits(),
                        val.to_bits(),
                        "patch {p} ({}{}) triangle {ti}",
                        names[axis],
                        val
                    );
                }
            }
            println!("patch_area_kept: patch {p} diff {:.3e}", oa - ia);
        }
        let input1 = cube_surface(false);
        let prep1 = prepare(&input1, 60.0).unwrap();
        let out1 =
            remesh(&prep1, &|_: Point| 0.1, &RemeshOptions::default()).unwrap();
        let total = patch_area(&out1.surface, 0);
        assert!((total - 6.0).abs() <= 6e-12, "total area {total}");
        println!("patch_area_kept: single-patch total {total:.16}");
    }

    /// TET-06b's wedge gate: the 30 degree prism's 6 corners and 9
    /// polylines survive; the polyline on the 30 degree edge runs from
    /// (0,0,0) to (0,0,1) with x = y = 0 bit for bit, and every point of
    /// the other polylines lies on its segment.
    #[test]
    fn wedge_edge() {
        let input = wedge();
        let prep = prepare(&input, 60.0).unwrap();
        assert_eq!(prep.report.n_corners, 6, "input corners");
        assert_eq!(prep.report.n_polylines, 9, "input polylines");
        let out = remesh(&prep, &|_: Point| 0.1, &RemeshOptions::default()).unwrap();
        let rep = check_manifold(&out.surface).unwrap();
        assert_eq!(rep.euler, 2);
        let q = prepare(&out.surface, 60.0).unwrap();
        assert_eq!(q.report.n_corners, 6, "output corners");
        assert_eq!(q.report.n_polylines, 9, "output polylines");
        for k in 0..6 {
            for d in 0..3 {
                assert_eq!(
                    out.surface.points[k][d].to_bits(),
                    input.points[k][d].to_bits(),
                    "corner {k} coordinate {d}"
                );
            }
        }
        assert_eq!(out.corners, vec![0, 1, 2, 3, 4, 5], "out corners");
        assert_eq!(out.corners, q.corners, "corners");
        assert_eq!(out.feature_edges, q.feature_edges, "feature edges");
        let pts = |v: u32| out.surface.points[v as usize];
        let zchain = q
            .polylines
            .iter()
            .find(|c| c[0] == 0 && *c.last().unwrap() == 3)
            .expect("the z-edge polyline");
        let mut prev = 0.0f64;
        for (i, &v) in zchain.iter().enumerate() {
            let p = pts(v);
            assert_eq!(p[0], 0.0, "x on the z edge");
            assert_eq!(p[1], 0.0, "y on the z edge");
            if i == 0 {
                assert_eq!(p[2], 0.0, "z starts at 0.0");
            } else {
                assert!(p[2] > prev, "z strictly increasing");
            }
            assert!(p[2] <= 1.0);
            prev = p[2];
        }
        assert_eq!(prev, 1.0, "top corner");
        let edges = zchain.len() - 1;
        assert!((8..=12).contains(&edges), "z chain of {edges} edges");
        let mut lens: Vec<usize> = Vec::new();
        let mut worst = 0.0f64;
        for chain in &q.polylines {
            lens.push(chain.len() - 1);
            if chain[0] == 0 && *chain.last().unwrap() == 3 {
                continue;
            }
            let (p0, p1) = (pts(chain[0]), pts(*chain.last().unwrap()));
            for &v in chain {
                worst = worst.max(seg_dist(pts(v), p0, p1));
            }
        }
        assert!(worst <= 1e-12, "off-segment by {worst}");
        let ia: f64 = patch_area(&input, 0);
        let oa: f64 = patch_area(&out.surface, 0);
        println!(
            "wedge_edge: chain lengths {lens:?}, area change {:.3e} (relative {:.3e})",
            oa - ia,
            (oa - ia) / ia
        );
    }

    /// TET-06b's cylinder gate: the two-patch cylinder keeps its three
    /// closed polylines with no corners, every end of a seam edge sits
    /// bit for bit at z = 0 on the input's 16-sided ring, and each
    /// triangle stays on its patch's side of the seam.
    #[test]
    fn cylinder_seam_kept() {
        let input = cyl2();
        let prep = prepare(&input, 60.0).unwrap();
        let out = remesh(&prep, &|_: Point| 0.1, &RemeshOptions::default()).unwrap();
        let rep = check_manifold(&out.surface).unwrap();
        assert_eq!(rep.euler, 2);
        let q = prepare(&out.surface, 60.0).unwrap();
        assert_eq!(q.report.n_corners, 0, "corners");
        assert_eq!(q.report.n_polylines, 3, "polylines");
        assert_eq!(q.report.n_closed_polylines, 3, "closed polylines");
        assert!(out.corners.is_empty(), "no corners");
        assert_eq!(out.feature_edges, q.feature_edges, "feature edges");
        assert_eq!(out.edge_is_seam, q.edge_is_seam, "seam flags");
        let ring = |k: u32| {
            let t = 2.0 * PI * (k % 16) as f64 / 16.0;
            [t.cos(), t.sin(), 0.0]
        };
        let mut n_seam = 0usize;
        for (i, seam) in out.edge_is_seam.iter().enumerate() {
            if !*seam {
                continue;
            }
            n_seam += 1;
            for e in 0..2 {
                let p = out.surface.points[out.feature_edges[i][e] as usize];
                assert_eq!(p[2].to_bits(), 0.0f64.to_bits(), "seam end z");
                let d = (0..16u32)
                    .map(|k| seg_dist(p, ring(k), ring(k + 1)))
                    .fold(f64::INFINITY, f64::min);
                assert!(d <= 1e-12, "seam end off the ring by {d}");
            }
        }
        for (ti, (t, &tp)) in
            out.surface.tris.iter().zip(&out.surface.tri_patch).enumerate()
        {
            for k in 0..3 {
                let z = out.surface.points[t[k] as usize][2];
                if tp == 0 {
                    assert!(z <= 0.0, "lower patch triangle {ti} at z {z}");
                } else {
                    assert!(z >= 0.0, "upper patch triangle {ti} at z {z}");
                }
            }
        }
        let ratios = edge_length_ratios(&out.surface, &|_: Point| 0.1);
        let frac = band_frac(&ratios);
        let angs = min_angles_deg(&out.surface);
        let afrac =
            angs.iter().filter(|&&a| a >= 25.0).count() as f64 / angs.len() as f64;
        for p in 0..2u32 {
            let ia = patch_area(&input, p);
            let oa = patch_area(&out.surface, p);
            println!(
                "cylinder_seam_kept: patch {p} area {oa:.6} vs input {ia:.6} (diff {:.3e}, not asserted)",
                oa - ia
            );
        }
        println!(
            "cylinder_seam_kept: seam edges {n_seam}, band_frac {frac:.6}, angle frac {afrac:.6}, points {}, tris {}",
            out.stats.n_points,
            out.stats.n_tris
        );
    }

    /// TET-06b's rotated-cube gate: no face axis-aligned, the 8 input
    /// points still bit for bit, 8 corners and 12 polylines, every
    /// polyline point on its segment, the total area kept.
    #[test]
    fn rotated_cube_features() {
        let input = rcube();
        let prep = prepare(&input, 60.0).unwrap();
        let out = remesh(&prep, &|_: Point| 0.1, &RemeshOptions::default()).unwrap();
        let rep = check_manifold(&out.surface).unwrap();
        assert_eq!(rep.euler, 2);
        let q = prepare(&out.surface, 60.0).unwrap();
        assert_eq!(q.report.n_corners, 8, "corners");
        assert_eq!(q.report.n_polylines, 12, "polylines");
        for k in 0..8 {
            for d in 0..3 {
                assert_eq!(
                    out.surface.points[k][d].to_bits(),
                    input.points[k][d].to_bits(),
                    "corner {k} coordinate {d}"
                );
            }
        }
        assert_eq!(out.feature_edges, q.feature_edges, "feature edges");
        assert_eq!(out.corners, q.corners, "corners");
        let mut worst = 0.0f64;
        for chain in &q.polylines {
            let (p0, p1) = (
                out.surface.points[chain[0] as usize],
                out.surface.points[*chain.last().unwrap() as usize],
            );
            for &v in chain {
                worst = worst.max(seg_dist(out.surface.points[v as usize], p0, p1));
            }
        }
        assert!(worst <= 1e-12, "polyline point off its segment by {worst}");
        let ia = patch_area(&input, 0);
        let oa = patch_area(&out.surface, 0);
        assert!(
            (oa - ia).abs() <= 1e-12 * ia,
            "total area {oa} vs input {ia}"
        );
        println!(
            "rotated_cube_features: largest seg_dist {worst:.3e}, area diff {:.3e}, points {}, tris {}",
            oa - ia,
            out.stats.n_points,
            out.stats.n_tris
        );
    }

    /// The free remesh is unchanged by TET-06b: the TET-06a sphere's
    /// output holds bit for bit (FNV-1a over points, triangles and
    /// patches) and the new tables are empty.
    #[test]
    fn free_surface_unchanged() {
        let out = sphere_out();
        assert_eq!(fnv1a(&out.surface), 0xf76943aa6d947000);
        assert!(out.feature_edges.is_empty(), "feature edges");
        assert!(out.edge_is_seam.is_empty(), "seam flags");
        assert!(out.edge_polyline.is_empty(), "edge polylines");
        assert!(out.corners.is_empty(), "corners");
    }

    // ==================================================================
    //  TET-06b: no flat and no cut triangle (116.14k)/(116.14l)
    // ==================================================================

    /// The flat measure of (116.14k): |n| / L^2, the triangle's normal
    /// length over the longest edge's square.
    fn flat_measure(t: &[Point; 3]) -> f64 {
        let l01 = sub(t[1], t[0]);
        let l12 = sub(t[2], t[1]);
        let l20 = sub(t[0], t[2]);
        let l2max = dot(l01, l01).max(dot(l12, l12)).max(dot(l20, l20));
        norm(tri_normal(t)) / l2max
    }

    /// (116.14k): how many of the surface's triangles are flat.
    fn n_flat(s: &TetSurface) -> usize {
        s.tris
            .iter()
            .filter(|t| {
                tri_flat(&[
                    s.points[t[0] as usize],
                    s.points[t[1] as usize],
                    s.points[t[2] as usize],
                ])
            })
            .count()
    }

    /// (116.14l): how many of the output's triangles are cut triangles
    /// - two edges of `out.feature_edges` meeting at a vertex that is
    /// not in `out.corners`.
    fn n_cut(out: &Remeshed) -> usize {
        let feat: BTreeSet<(u32, u32)> =
            out.feature_edges.iter().map(|&e| (e[0], e[1])).collect();
        let corners: BTreeSet<u32> = out.corners.iter().copied().collect();
        let mut n = 0usize;
        for t in &out.surface.tris {
            for k in 0..3 {
                let (u, a, b) = (t[k], t[(k + 1) % 3], t[(k + 2) % 3]);
                if !corners.contains(&u)
                    && feat.contains(&edge_key(u, a))
                    && feat.contains(&edge_key(u, b))
                {
                    n += 1;
                    break;
                }
            }
        }
        n
    }

    /// The point turned exactly as `rcube` turns the cube's: Rx(0.7),
    /// then Ry(0.5), then Rz(0.3).
    fn turn(p: Point) -> Point {
        let (ga, gb, gg) = (0.7f64, 0.5f64, 0.3f64);
        let (ca, sa) = (ga.cos(), ga.sin());
        let (cb, sb) = (gb.cos(), gb.sin());
        let (cg, sg) = (gg.cos(), gg.sin());
        let (x, y, z) = (p[0], p[1], p[2]);
        let (y, z) = (y * ca - z * sa, y * sa + z * ca);
        let (x, z) = (x * cb + z * sb, -x * sb + z * cb);
        let (x, y) = (x * cg - y * sg, x * sg + y * cg);
        [x, y, z]
    }

    /// The 30 degree wedge with every point turned exactly as RCUBE
    /// turns CUBE1.
    fn rwedge() -> TetSurface {
        let base = wedge();
        TetSurface {
            points: base.points.iter().map(|&p| turn(p)).collect(),
            tris: base.tris,
            tri_patch: base.tri_patch,
            patch_names: base.patch_names,
        }
    }

    /// The two-patch cylinder with every point turned exactly as RCUBE
    /// turns CUBE1.
    fn rcyl2() -> TetSurface {
        let base = cyl2();
        TetSurface {
            points: base.points.iter().map(|&p| turn(p)).collect(),
            tris: base.tris,
            tri_patch: base.tri_patch,
            patch_names: base.patch_names,
        }
    }

    /// One named row of the no-flat-no-cut gate: remesh and assert the
    /// "expected in every row" list - a closed 2-manifold of Euler 2,
    /// `Q = prepare(.., 60)` ok on the output, 0 flat and 0 cut
    /// triangles, and the output's feature tables equal to `Q`'s. Every
    /// assert names the input, h and n. Returns the smallest flat
    /// measure |n| / L^2 seen over the output's triangles.
    fn feature_run(name: &str, input: &TetSurface, h: f64, n: u32) -> f64 {
        let tag = format!("{name} h {h} n {n}");
        let prep = prepare(input, 60.0)
            .unwrap_or_else(|e| panic!("{tag}: input refused: {e}"));
        let out = remesh(&prep, &|_: Point| h, &RemeshOptions { iterations: n, threads: 0 })
            .unwrap_or_else(|e| panic!("{tag}: remesh refused: {e}"));
        let rep = check_manifold(&out.surface)
            .unwrap_or_else(|e| panic!("{tag}: manifold: {e}"));
        assert_eq!(rep.euler, 2, "{tag}: euler");
        let q = prepare(&out.surface, 60.0)
            .unwrap_or_else(|e| panic!("{tag}: output refused: {e}"));
        let nf = n_flat(&out.surface);
        let nc = n_cut(&out);
        assert_eq!(nf, 0, "{tag}: flat triangles");
        assert_eq!(nc, 0, "{tag}: cut triangles");
        assert_eq!(out.feature_edges, q.feature_edges, "{tag}: feature edges");
        assert_eq!(out.edge_is_seam, q.edge_is_seam, "{tag}: seam flags");
        assert_eq!(out.corners, q.corners, "{tag}: corners");
        let mut m = f64::INFINITY;
        for t in &out.surface.tris {
            m = m.min(flat_measure(&[
                out.surface.points[t[0] as usize],
                out.surface.points[t[1] as usize],
                out.surface.points[t[2] as usize],
            ]));
        }
        println!(
            "feature_run({tag}): flat {nf}, cut {nc}, min |n|/L^2 {m:.3e}, \
             points {}, tris {}, feature edges {}",
            out.stats.n_points,
            out.stats.n_tris,
            out.feature_edges.len()
        );
        m
    }

    /// The five named rows the first build failed (it had neither
    /// rule): 0 flat and 0 cut triangles in every one, with the
    /// manifold, `Q` and the tables as in [`feature_run`].
    #[test]
    fn no_flat_no_cut_named() {
        let rcube_s = rcube();
        let rwedge_s = rwedge();
        let cyl2_s = cyl2();
        let rcyl2_s = rcyl2();
        feature_run("rcube", &rcube_s, 0.07, 3);
        feature_run("rwedge", &rwedge_s, 0.13, 3);
        feature_run("cyl2", &cyl2_s, 0.2, 2);
        feature_run("cyl2", &cyl2_s, 0.07, 7);
        feature_run("rcyl2", &rcyl2_s, 0.07, 11);
    }

    /// The sweep: CUBE1, RCUBE, WEDGE, RWEDGE, CYL2, RCYL2 at
    /// h in {0.07, 0.1, 0.13, 0.2} and 1..=12 iterations, 288 runs.
    #[test]
    fn feature_sweep() {
        let inputs: Vec<(&str, TetSurface)> = vec![
            ("cube1", cube_surface(false)),
            ("rcube", rcube()),
            ("wedge", wedge()),
            ("rwedge", rwedge()),
            ("cyl2", cyl2()),
            ("rcyl2", rcyl2()),
        ];
        let mut runs = 0usize;
        let mut smallest = f64::INFINITY;
        for (name, input) in &inputs {
            for h in [0.07f64, 0.1, 0.13, 0.2] {
                for n in 1..=12u32 {
                    let m = feature_run(name, input, h, n);
                    smallest = smallest.min(m);
                    runs += 1;
                }
            }
        }
        println!("feature_sweep: {runs} runs, smallest |n|/L^2 {smallest:.3e}");
        assert_eq!(runs, 288);
    }
}
