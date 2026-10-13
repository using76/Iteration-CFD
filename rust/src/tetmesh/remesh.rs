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
//! result, so two runs are bit-identical. TET-06a treats every vertex
//! as free; features, corners and seams are TET-06b's job.
//!
//! Provenance: ORIGINAL. Written from Botsch & Kobbelt (2004), Proc. SGP
//! 185-192; the choices are derived in SPEC-LIT §116.10. No GPL-licensed
//! source was consulted.

use std::collections::{BTreeMap, BTreeSet};

use super::sizefield::point_box_distance;
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

/// Exactly zero, all three components.
fn is_zero(v: Point) -> bool {
    v[0] == 0.0 && v[1] == 0.0 && v[2] == 0.0
}

/// The triangle's f64 normal `(x1 - x0) x (x2 - x0)`, unnormalised.
fn tri_normal(p: &[Point; 3]) -> Point {
    cross(sub(p[1], p[0]), sub(p[2], p[0]))
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
/// project iterations to run.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RemeshOptions {
    pub iterations: u32,
}

impl Default for RemeshOptions {
    fn default() -> Self {
        RemeshOptions { iterations: 10 }
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

/// The remeshed surface and its stats.
#[derive(Debug, Clone, PartialEq)]
pub struct Remeshed {
    pub surface: TetSurface,
    pub stats: RemeshStats,
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
        }
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

    /// The collapse of `(a, b)` into its midpoint: the survivor is `a`
    /// (the smaller index, since the pass visits keys ascending), `b`
    /// dies. Every guard of §116.10 is checked before anything moves.
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
        if !self.collapse_guards(a, b, m, t1, t2, h)? {
            return Ok(false);
        }
        self.collapse_apply(a, b, m, t1, t2);
        Ok(true)
    }

    /// The guards of §116.10's collapse: every `w` in
    /// `N(a) u N(b) \ {a, b}` stays within `(4/3) h` of `m` (116.14b),
    /// and every surviving triangle of the two one-rings keeps positive
    /// area and turns its normal by less than 90 degrees. `Ok(false)` is
    /// a refused collapse, never an error.
    fn collapse_guards(
        &self,
        a: u32,
        b: u32,
        m: Point,
        t1: u32,
        t2: u32,
        h: &dyn Fn(Point) -> f64,
    ) -> Result<bool> {
        let ws: BTreeSet<u32> = self.nbrs[a as usize]
            .union(&self.nbrs[b as usize])
            .copied()
            .filter(|&w| w != a && w != b)
            .collect();
        for w in &ws {
            let xw = self.pts[*w as usize];
            let hw = h_checked(h, mid(m, xw))?;
            if norm(sub(m, xw)) > (4.0 / 3.0) * hw {
                return Ok(false);
            }
        }
        let mut ring: BTreeSet<u32> = BTreeSet::new();
        for v in [a, b] {
            for &w in &self.nbrs[v as usize] {
                if let Some(ts) = self.edges.get(&edge_key(v, w)) {
                    ring.extend(ts.iter().copied());
                }
            }
        }
        ring.remove(&t1);
        ring.remove(&t2);
        for t in &ring {
            let old = self.tri_pts(*t);
            let old_n = tri_normal(&old);
            let tri = self.tris[*t as usize];
            let mut new = old;
            for k in 0..3 {
                let u = tri[k];
                if u == a || u == b {
                    new[k] = m;
                }
            }
            let new_n = tri_normal(&new);
            if is_zero(new_n) || dot(new_n, old_n) <= 0.0 {
                return Ok(false);
            }
        }
        Ok(true)
    }

    /// Applies the collapse: `a` moves to `m`, the two dead triangles
    /// go, every other triangle around `b` is re-pointed to `a`, and the
    /// edge and neighbour maps follow.
    fn collapse_apply(&mut self, a: u32, b: u32, m: Point, t1: u32, t2: u32) {
        for t in [t1, t2] {
            let tri = self.tris[t as usize];
            for k in 0..3 {
                self.edge_drop(edge_key(tri[k], tri[(k + 1) % 3]), t);
            }
            self.live_t[t as usize] = false;
        }
        let mut b_tris: Vec<u32> = Vec::new();
        for &w in &self.nbrs[b as usize] {
            if let Some(ts) = self.edges.get(&edge_key(b, w)) {
                b_tris.extend(ts.iter().copied());
            }
        }
        b_tris.sort_unstable();
        b_tris.dedup();
        for t in b_tris {
            let tri = self.tris[t as usize];
            let others: Vec<u32> =
                (0..3).filter(|&k| tri[k] != b).map(|k| tri[k]).collect();
            self.edge_drop(edge_key(b, others[0]), t);
            self.edge_drop(edge_key(b, others[1]), t);
            self.edge_add(edge_key(a, others[0]), t);
            self.edge_add(edge_key(a, others[1]), t);
            let mut tri = tri;
            for k in 0..3 {
                if tri[k] == b {
                    tri[k] = a;
                }
            }
            self.tris[t as usize] = tri;
        }
        let nb: Vec<u32> = self.nbrs[b as usize].iter().copied().collect();
        for w in nb.iter().copied() {
            if w == a {
                continue;
            }
            self.nbrs[w as usize].remove(&b);
            self.nbrs[w as usize].insert(a);
        }
        self.nbrs[a as usize].remove(&b);
        for w in nb.iter().copied() {
            if w != a {
                self.nbrs[a as usize].insert(w);
            }
        }
        self.nbrs[b as usize].clear();
        self.pts[a as usize] = m;
        self.live_v[b as usize] = false;
        self.n_live_v -= 1;
    }

    /// (116.14c): one pass over the edges in ascending order, flipping
    /// each interior edge whose flip lowers the valence energy
    /// `E = sum (val(v) - 6)^2` over `{a, b, c, d}` and passes the
    /// guards: `(c, d)` not already an edge, neither `val(a)` nor
    /// `val(b)` is 3, and each new triangle's normal makes less than 90
    /// degrees with the sum of the two old ones.
    fn flip_pass(&mut self) -> usize {
        let cand: Vec<(u32, u32)> = self.edges.keys().copied().collect();
        let mut n = 0usize;
        for key in cand {
            let (a, b) = key;
            if !self.live_v[a as usize] || !self.live_v[b as usize] {
                continue;
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
            if val(a) == 3 || val(b) == 3 {
                continue;
            }
            let e = (val(a) - 6).pow(2)
                + (val(b) - 6).pow(2)
                + (val(c) - 6).pow(2)
                + (val(d) - 6).pow(2);
            let e2 = (val(a) - 7).pow(2)
                + (val(b) - 7).pow(2)
                + (val(c) - 5).pow(2)
                + (val(d) - 5).pow(2);
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

    /// (116.14): tangential relaxation in Jacobi order - every vertex's
    /// new position is computed from the old positions before any moves:
    /// `x <- x + (I - n n^T) (c - x)` with `n` the normalised sum of the
    /// one-ring's triangle normals and `c` the area-weighted centroid.
    fn relax(&mut self) {
        let old = self.pts.clone();
        let mut new_pts = old.clone();
        for v in 0..self.pts.len() {
            if !self.live_v[v] {
                continue;
            }
            let mut ring: BTreeSet<u32> = BTreeSet::new();
            for &w in &self.nbrs[v] {
                if let Some(ts) = self.edges.get(&edge_key(v as u32, w)) {
                    ring.extend(ts.iter().copied());
                }
            }
            if ring.is_empty() {
                continue;
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
                continue;
            }
            let nn = norm(n_sum);
            if nn == 0.0 {
                continue;
            }
            let n = scale(n_sum, 1.0 / nn);
            let c = scale(c_sum, 1.0 / a_sum);
            let d0 = sub(c, old[v]);
            new_pts[v] = add(old[v], sub(d0, scale(n, dot(n, d0))));
        }
        self.pts = new_pts;
    }

    /// (116.14d): every live vertex projected onto the input surface
    /// through the one `Projector` built on it; among equally close
    /// triangles the smallest index wins.
    fn project(&mut self, proj: &Projector) {
        for v in 0..self.pts.len() {
            if self.live_v[v] {
                self.pts[v] = proj.closest(self.pts[v]).0;
            }
        }
    }

    /// Compacts: no unused vertices, vertices renumbered in increasing
    /// old-index order, adjacency rebuilt.
    fn compact(&mut self) {
        let names = Vec::new();
        let surf = self.to_surface(names);
        *self = Mesh::from_surface(&surf);
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
/// (116.14b with the guards), flips (116.14c with the guards),
/// tangential relaxation (116.14) in Jacobi order, and projection
/// (116.14d). Refuses `iterations == 0` and any `h` value that is not
/// finite and positive, and returns the output only once
/// [`check_manifold`] accepts it.
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
    let mut mesh = Mesh::from_surface(&input.surface);
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
        mesh.relax();
        mesh.project(&proj);
        mesh.compact();
    }
    let surface = mesh.to_surface(input.surface.patch_names.clone());
    check_manifold(&surface)?;
    stats.iterations = opts.iterations;
    stats.n_points = surface.points.len();
    stats.n_tris = surface.tris.len();
    Ok(Remeshed { surface, stats })
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
            &RemeshOptions { iterations: 3 },
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
            &RemeshOptions { iterations: 0 },
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
}





























