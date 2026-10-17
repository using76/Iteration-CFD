// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.
// Provenance: see PROVENANCE.md. No GPL-licensed source was consulted.

//! The size field h(x) of SPEC-LIT §116.8: the edge length, in metres, that
//! the later tetmesh units ask for at a point. The field is the minimum of
//! cones over the flattened primitives, and a query evaluates (116.7b)
//! exactly over the candidate list of the one octree leaf holding the query
//! point; it never interpolates. The octree itself only culls: it drops a
//! primitive from a child by the per-leaf bounds of (116.7c) under the
//! margin rule of (116.7d), then drops a patch source's other triangles
//! where one of them fills the whole child inside the band, (116.7e), so
//! the leaf answer equals the evaluation over every primitive bit for bit.
//! The split rule (116.7f) stops the tree at the scale of the mesh it
//! serves, within the leaf-count bound (116.7g). The curvature and
//! proximity sources of a `TetSurface`, (116.9a)-(116.9c) and
//! (116.10a)-(116.10b), are per-triangle patch primitives, so the octree
//! and the gradation limit (116.11) hold for them unchanged.
//!
//! Provenance: ORIGINAL. Written from Persson (2006), Eng. Comput. 22:95-109; the culling bounds are derived in SPEC-LIT §116.8. No GPL-licensed source was consulted.

use super::{Point, TetSizeSpec};
use crate::automesher::RefinementSpec;
use crate::error::{Error, Result};
use std::collections::HashMap;

/// The validated numbers of the size field: the base size `H`, the
/// gradation `g`, and the floor `h_min` of (116.12), which defaults to
/// `H / 64` when no `min_size` is given.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SizeParams {
    pub base_size: f64,
    pub gradation: f64,
    pub min_size: f64,
}

impl SizeParams {
    /// Validate `H`, `g` and the floor. `g = 1` is a uniform size and is
    /// legal; the floor may equal `H` but not exceed it.
    pub fn new(base_size: f64, gradation: f64, min_size: Option<f64>) -> Result<SizeParams> {
        if !base_size.is_finite() || base_size <= 0.0 {
            return Err(Error::Mesh(format!(
                "size field: base_size {base_size} must be finite and greater than 0"
            )));
        }
        if !gradation.is_finite() || gradation < 1.0 {
            return Err(Error::Mesh(format!(
                "size field: gradation {gradation} must be finite and at least 1"
            )));
        }
        let min_size = min_size.unwrap_or(base_size / 64.0);
        if !min_size.is_finite() || min_size <= 0.0 || min_size > base_size {
            return Err(Error::Mesh(format!(
                "size field: min_size {min_size} must be finite, greater than 0, \
                 and no greater than base_size {base_size}"
            )));
        }
        Ok(SizeParams {
            base_size,
            gradation,
            min_size,
        })
    }

    /// Validate from the `tet.size` config block's gradation and floor.
    pub fn from_spec(base_size: f64, spec: &TetSizeSpec) -> Result<SizeParams> {
        SizeParams::new(base_size, spec.gradation, spec.min_size)
    }
}

/// One size source before flattening: a point, an axis-aligned box, or a
/// named patch with a flat band out to `distance`, per §116.8.
#[derive(Debug, Clone, PartialEq)]
pub enum SizeSource {
    Point { at: Point, size: f64 },
    Box { min: Point, max: Point, size: f64 },
    Patch {
        name: String,
        tris: Vec<[Point; 3]>,
        distance: f64,
        size: f64,
    },
}

/// A patch's triangles under its name, as `sources_from_refinement` matches
/// the refinement config's patch names against them.
#[derive(Debug, Clone, PartialEq)]
pub struct PatchTris {
    pub name: String,
    pub tris: Vec<[Point; 3]>,
}

/// The background octree's DESIGN knobs: they change the cost of a query,
/// never its answer, per §116.8. `threads` is the number of workers the
/// build uses, `0` meaning every core `available_parallelism` reports
/// (TET-15); the tree is the same for every count, bit for bit.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OctreeOptions {
    pub leaf_cap: usize,
    pub max_depth: u32,
    pub threads: usize,
}

impl Default for OctreeOptions {
    fn default() -> Self {
        OctreeOptions {
            leaf_cap: 16,
            max_depth: 12,
            threads: 0,
        }
    }
}

/// The worker count `threads` asks for: `0` is every core the system
/// reports (one when it reports none), any other value itself.
pub fn resolve_threads(threads: usize) -> usize {
    if threads > 0 {
        return threads;
    }
    std::thread::available_parallelism().map_or(1, |n| n.get())
}

/// `f(0), f(1), .., f(n - 1)` in index order, computed by up to `threads`
/// scoped workers (TET-15, SPEC-LIT §116.8). The workers take the next
/// index from one atomic counter and each result is stored under its own
/// index, so the output does not depend on the schedule or on the worker
/// count: it is the sequential map whenever `f` reads only shared state.
/// One worker, or one item, runs on the calling thread.
pub fn par_map<T, F>(n: usize, threads: usize, f: F) -> Vec<T>
where
    T: Send,
    F: Fn(usize) -> T + Sync,
{
    let workers = resolve_threads(threads).min(n);
    if workers <= 1 {
        return (0..n).map(f).collect();
    }
    let next = std::sync::atomic::AtomicUsize::new(0);
    let mut parts: Vec<Vec<(usize, T)>> = std::thread::scope(|scope| {
        let handles: Vec<_> = (0..workers)
            .map(|_| {
                scope.spawn(|| {
                    let mut got: Vec<(usize, T)> = Vec::new();
                    loop {
                        let i = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        if i >= n {
                            return got;
                        }
                        got.push((i, f(i)));
                    }
                })
            })
            .collect();
        handles
            .into_iter()
            .map(|h| h.join().unwrap_or_else(|e| std::panic::resume_unwind(e)))
            .collect()
    });
    let mut slots: Vec<Option<T>> = (0..n).map(|_| None).collect();
    for part in parts.iter_mut() {
        for (i, v) in part.drain(..) {
            slots[i] = Some(v);
        }
    }
    slots
        .into_iter()
        .map(|v| v.expect("par_map: every index is taken exactly once"))
        .collect()
}

/// What a build produced, over the leaves of the finished tree.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SizeFieldStats {
    pub primitives: usize,
    pub leaves: usize,
    pub depth: u32,
    pub max_candidates: usize,
    pub mean_candidates: f64,
}

/// A level's size, (116.8): `H` halved `l` times, exactly.
pub fn level_size(base_size: f64, level: u32) -> f64 {
    base_size * 0.5f64.powi(level as i32)
}

/// The automesher's `refinement` block translated to size sources, in the
/// config's order, per §116.8: one patch source per band element with
/// `h = H 2^-min(level, max_level)`, which is (116.8) under the automesher's
/// own level cap (eq. 92.1); then one box source per box. A missing patch,
/// an empty one, a `feature_level` above zero and a negative or non-finite
/// band distance are refused by name.
pub fn sources_from_refinement(
    spec: &RefinementSpec,
    base_size: f64,
    patches: &[PatchTris],
) -> Result<Vec<SizeSource>> {
    let mut out: Vec<SizeSource> = Vec::new();
    for entry in &spec.levels {
        let patch = match patches.iter().find(|p| p.name == entry.patch) {
            Some(p) => p,
            None => {
                let names: Vec<String> = patches.iter().map(|p| p.name.clone()).collect();
                return Err(Error::Mesh(format!(
                    "size field: refinement names patch {:?}, which is not among the patches {:?}",
                    entry.patch, names
                )));
            }
        };
        if patch.tris.is_empty() {
            return Err(Error::Mesh(format!(
                "size field: refinement patch {:?} has no triangle",
                patch.name
            )));
        }
        if entry.feature_level > 0 {
            return Err(Error::Mesh(format!(
                "size field: refinement feature_level {} on patch {:?} has no tet source and \
                 is refused, since ignoring it would mesh something the user did not ask for",
                entry.feature_level, patch.name
            )));
        }
        for band in &entry.bands {
            if !band.distance.is_finite() || band.distance < 0.0 {
                return Err(Error::Mesh(format!(
                    "size field: refinement band distance {} on patch {:?} must be finite \
                     and not negative",
                    band.distance, patch.name
                )));
            }
            out.push(SizeSource::Patch {
                name: patch.name.clone(),
                tris: patch.tris.clone(),
                distance: band.distance,
                size: level_size(base_size, band.level.min(spec.max_level)),
            });
        }
    }
    for b in &spec.boxes {
        out.push(SizeSource::Box {
            min: b.min,
            max: b.max,
            size: level_size(base_size, b.level.min(spec.max_level)),
        });
    }
    Ok(out)
}

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

/// The Euclidean norm.
fn norm(x: Point) -> f64 {
    dot(x, x).sqrt()
}

/// Distance from `x` to the segment `p q`: the degenerate segment is the
/// point `p`; otherwise the projection parameter is clamped to the segment.
fn segment_distance(x: Point, p: Point, q: Point) -> f64 {
    let pq = sub(q, p);
    let l2 = dot(pq, pq);
    if l2 == 0.0 {
        return norm(sub(x, p));
    }
    let t = (dot(sub(x, p), pq) / l2).clamp(0.0, 1.0);
    norm(sub(x, add(p, scale(pq, t))))
}

/// Distance from `x` to the triangle `t`, via the classic Voronoi-region
/// walk: the vertex regions, then the edge regions, then the face interior.
/// A zero-area triangle, including one with repeated vertices, falls back to
/// the minimum over its three segments.
pub fn point_triangle_distance(x: Point, t: &[Point; 3]) -> f64 {
    let a = t[0];
    let b = t[1];
    let c = t[2];
    let ab = sub(b, a);
    let ac = sub(c, a);
    let n = [ab[1] * ac[2] - ab[2] * ac[1], ab[2] * ac[0] - ab[0] * ac[2], ab[0] * ac[1] - ab[1] * ac[0]];
    if dot(n, n) == 0.0 {
        return segment_distance(x, a, b)
            .min(segment_distance(x, b, c))
            .min(segment_distance(x, c, a));
    }
    let ap = sub(x, a);
    let d1 = dot(ab, ap);
    let d2 = dot(ac, ap);
    if d1 <= 0.0 && d2 <= 0.0 {
        return norm(ap); // vertex region A
    }
    let bp = sub(x, b);
    let d3 = dot(ab, bp);
    let d4 = dot(ac, bp);
    if d3 >= 0.0 && d4 <= d3 {
        return norm(bp); // vertex region B
    }
    let vc = d1 * d4 - d3 * d2;
    if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
        return norm(sub(x, add(a, scale(ab, d1 / (d1 - d3))))); // edge AB
    }
    let cp = sub(x, c);
    let d5 = dot(ab, cp);
    let d6 = dot(ac, cp);
    if d6 >= 0.0 && d5 <= d6 {
        return norm(cp); // vertex region C
    }
    let vb = d5 * d2 - d1 * d6;
    if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
        return norm(sub(x, add(a, scale(ac, d2 / (d2 - d6))))); // edge AC
    }
    let va = d3 * d6 - d5 * d4;
    if va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0 {
        // edge BC
        let den = (d4 - d3) + (d5 - d6);
        return norm(sub(x, add(b, scale(sub(c, b), (d4 - d3) / den))));
    }
    // Face interior: all three sub-areas positive, so the sum is too.
    let denom = 1.0 / (va + vb + vc);
    let q = add(add(a, scale(ab, vb * denom)), scale(ac, vc * denom));
    norm(sub(x, q))
}

/// Distance from `x` to the axis-aligned box: per axis the outside gap, zero
/// inside or on the box.
pub fn point_box_distance(x: Point, lo: Point, hi: Point) -> f64 {
    let e0 = (lo[0] - x[0]).max(0.0).max(x[0] - hi[0]);
    let e1 = (lo[1] - x[1]).max(0.0).max(x[1] - hi[1]);
    let e2 = (lo[2] - x[2]).max(0.0).max(x[2] - hi[2]);
    (e0 * e0 + e1 * e1 + e2 * e2).sqrt()
}

/// One primitive's support: a point, an axis-aligned box, or one triangle.
#[derive(Debug, Clone, Copy)]
enum Support {
    Point(Point),
    Box { lo: Point, hi: Point },
    Tri([Point; 3]),
}

impl Support {
    /// `d_k(x)`, the distance to the support, zero inside or on a box.
    fn distance(&self, x: Point) -> f64 {
        match *self {
            Support::Point(p) => norm(sub(x, p)),
            Support::Box { lo, hi } => point_box_distance(x, lo, hi),
            Support::Tri(t) => point_triangle_distance(x, &t),
        }
    }

    /// Every support vertex, for the `A` of (116.7d).
    fn vertices(&self) -> Vec<Point> {
        match *self {
            Support::Point(p) => vec![p],
            Support::Box { lo, hi } => (0..8usize)
                .map(|b| {
                    [
                        if b & 1 != 0 { hi[0] } else { lo[0] },
                        if b & 2 != 0 { hi[1] } else { lo[1] },
                        if b & 4 != 0 { hi[2] } else { lo[2] },
                    ]
                })
                .collect(),
            Support::Tri(t) => t.to_vec(),
        }
    }
}

/// One flattened primitive: a cone height `h`, the flat distance `d_flat`,
/// a support, and the index in `sources` of the source the primitive came
/// from, so the triangles of one patch source can be told from another's
/// even when both carry the same name. A point or box source is one
/// primitive; a patch source is one per triangle.
#[derive(Debug, Clone, Copy)]
struct Prim {
    h: f64,
    d_flat: f64,
    support: Support,
    src: u32,
}

/// The one cone evaluation, (116.7a), shared by the octree query and the
/// brute force so the two agree bit for bit.
fn cone(p: &SizeParams, k: &Prim, x: Point) -> f64 {
    let s = p.gradation - 1.0;
    k.h + s * (k.support.distance(x) - k.d_flat).max(0.0)
}

/// One octree node: its box, its candidate primitives in the parent's
/// order, and its eight children when split. A leaf holds the candidates a
/// query evaluates; an internal node keeps an empty list.
#[derive(Debug, Clone)]
struct Node {
    lo: Point,
    hi: Point,
    cand: Vec<u32>,
    children: Option<Box<[Node; 8]>>,
}

/// The size field of §116.8 over one domain box: the params, the flattened
/// primitives, the octree and the build's stats.
pub struct SizeField {
    params: SizeParams,
    prims: Vec<Prim>,
    lo: Point,
    hi: Point,
    root: Node,
    stats: SizeFieldStats,
}

impl SizeField {
    /// Build the field. Refuses a bad domain, then each bad source in order,
    /// then flattens and splits the octree while the three DESIGN rules of
    /// §116.8 all hold, on `opts.threads` workers (TET-15); the tree, the
    /// stats and every query answer are the same for every worker count.
    pub fn build(
        params: SizeParams,
        sources: &[SizeSource],
        lo: Point,
        hi: Point,
        opts: OctreeOptions,
    ) -> Result<SizeField> {
        if (0..3).any(|a| !lo[a].is_finite() || !hi[a].is_finite()) {
            return Err(Error::Mesh(format!(
                "size field: domain lo {lo:?} hi {hi:?} has a component that is not finite"
            )));
        }
        if (0..3).any(|a| lo[a] >= hi[a]) {
            return Err(Error::Mesh(format!(
                "size field: domain lo {lo:?} hi {hi:?} needs lo < hi on every axis"
            )));
        }
        for (i, src) in sources.iter().enumerate() {
            let size = match src {
                SizeSource::Point { size, .. }
                | SizeSource::Box { size, .. }
                | SizeSource::Patch { size, .. } => *size,
            };
            if !size.is_finite() || size <= 0.0 {
                return Err(Error::Mesh(format!(
                    "size field: source {i} size {size} must be finite and greater than 0"
                )));
            }
            let bad_coord = match src {
                SizeSource::Point { at, .. } => !at.iter().all(|v| v.is_finite()),
                SizeSource::Box { min, max, .. } => {
                    !min.iter().chain(max.iter()).all(|v| v.is_finite())
                }
                SizeSource::Patch { tris, .. } => tris
                    .iter()
                    .flat_map(|t| t.iter())
                    .any(|v| !v.iter().all(|c| c.is_finite())),
            };
            if bad_coord {
                return Err(Error::Mesh(format!(
                    "size field: source {i} has a coordinate that is not finite"
                )));
            }
            if let SizeSource::Box { min, max, .. } = src {
                if (0..3).any(|a| min[a] > max[a]) {
                    return Err(Error::Mesh(format!(
                        "size field: source {i} box min {min:?} max {max:?} has min > max on \
                         an axis"
                    )));
                }
            }
            if let SizeSource::Patch { name, tris, .. } = src {
                if tris.is_empty() {
                    return Err(Error::Mesh(format!(
                        "size field: source {i} patch {name:?} has no triangle"
                    )));
                }
            }
            if let SizeSource::Patch { distance, .. } = src {
                if !distance.is_finite() || *distance < 0.0 {
                    return Err(Error::Mesh(format!(
                        "size field: source {i} patch distance {distance} must be finite and \
                         not negative"
                    )));
                }
            }
        }
        // Flatten, in source order, a patch one primitive per triangle.
        let mut prims: Vec<Prim> = Vec::new();
        for (si, src) in sources.iter().enumerate() {
            match *src {
                SizeSource::Point { at, size } => prims.push(Prim {
                    h: size,
                    d_flat: 0.0,
                    support: Support::Point(at),
                    src: si as u32,
                }),
                SizeSource::Box { min, max, size } => prims.push(Prim {
                    h: size,
                    d_flat: 0.0,
                    support: Support::Box { lo: min, hi: max },
                    src: si as u32,
                }),
                SizeSource::Patch { ref tris, distance, size, .. } => {
                    for t in tris {
                        prims.push(Prim {
                            h: size,
                            d_flat: distance,
                            support: Support::Tri(*t),
                            src: si as u32,
                        });
                    }
                }
            }
        }
        // The culling margins of (116.7d) and (116.7e), computed once: `A`
        // is the largest absolute coordinate among the root's corners and
        // every support vertex, plus the root diagonal; `D_max` the largest
        // flat distance, zero with no patch.
        let mut a_max: f64 = 0.0;
        for x in corners_of(lo, hi) {
            for a in 0..3 {
                a_max = a_max.max(x[a].abs());
            }
        }
        for k in &prims {
            for v in k.support.vertices() {
                for a in 0..3 {
                    a_max = a_max.max(v[a].abs());
                }
            }
        }
        let dx = hi[0] - lo[0];
        let dy = hi[1] - lo[1];
        let dz = hi[2] - lo[2];
        a_max += (dx * dx + dy * dy + dz * dz).sqrt();
        let d_max = prims.iter().map(|k| k.d_flat).fold(0.0f64, f64::max);
        let s = params.gradation - 1.0;
        let eta = 1e-9 * (params.base_size + s * (a_max + d_max));
        let eta_d = 1e-9 * (a_max + d_max);
        let mut stats = SizeFieldStats {
            primitives: prims.len(),
            leaves: 0,
            depth: 0,
            max_candidates: 0,
            mean_candidates: 0.0,
        };
        let all: Vec<u32> = (0..prims.len() as u32).collect();
        let grow = Grow {
            prims: &prims,
            p: params,
            opts,
            eta,
            eta_d,
        };
        let root = build_root(lo, hi, all, &grow, &mut stats);
        // The mean accumulated a running sum over the leaves; divide it now.
        if stats.leaves > 0 {
            stats.mean_candidates /= stats.leaves as f64;
        }
        Ok(SizeField {
            params,
            prims,
            lo,
            hi,
            root,
            stats,
        })
    }

    pub fn params(&self) -> SizeParams {
        self.params
    }

    pub fn stats(&self) -> SizeFieldStats {
        self.stats
    }

    /// The leaf holding `x`, or `None` when `x` is outside the root box. The
    /// descent takes the child whose bit `a` is set iff `x[a] >= mid[a]`,
    /// the same `mid` expression the build split with.
    fn leaf_of(&self, x: Point) -> Option<&Node> {
        for a in 0..3 {
            if x[a] < self.lo[a] || x[a] > self.hi[a] {
                return None;
            }
        }
        let mut node = &self.root;
        loop {
            match &node.children {
                None => return Some(node),
                Some(ch) => {
                    let mut bit = 0usize;
                    for a in 0..3 {
                        let mid = 0.5 * (node.lo[a] + node.hi[a]);
                        if x[a] >= mid {
                            bit |= 1 << a;
                        }
                    }
                    node = &ch[bit];
                }
            }
        }
    }

    /// The field at `x`, per (116.7b) over the one leaf's candidates. Outside
    /// the root every primitive is evaluated instead.
    pub fn h(&self, x: Point) -> f64 {
        match self.leaf_of(x) {
            None => self.h_brute(x),
            Some(node) => {
                let mut m = f64::INFINITY;
                for &j in &node.cand {
                    m = m.min(cone(&self.params, &self.prims[j as usize], x));
                }
                self.params.min_size.max(self.params.base_size.min(m))
            }
        }
    }

    /// The same evaluation over every primitive, the oracle the octree must
    /// match bit for bit.
    pub fn h_brute(&self, x: Point) -> f64 {
        let mut m = f64::INFINITY;
        for k in &self.prims {
            m = m.min(cone(&self.params, k, x));
        }
        self.params.min_size.max(self.params.base_size.min(m))
    }

    /// The queried leaf's box, or `None` outside the root.
    pub fn leaf_box(&self, x: Point) -> Option<(Point, Point)> {
        self.leaf_of(x).map(|n| (n.lo, n.hi))
    }

    /// How many candidates the queried leaf holds, or `None` outside the
    /// root. The plain count of what `h` evaluates there.
    pub fn leaf_candidates(&self, x: Point) -> Option<usize> {
        self.leaf_of(x).map(|n| n.cand.len())
    }
}

/// The eight corners of a box.
fn corners_of(lo: Point, hi: Point) -> Vec<Point> {
    (0..8usize)
        .map(|b| {
            [
                if b & 1 != 0 { hi[0] } else { lo[0] },
                if b & 2 != 0 { hi[1] } else { lo[1] },
                if b & 4 != 0 { hi[2] } else { lo[2] },
            ]
        })
        .collect()
}

/// The octree build's shared, read-only inputs: the primitives, the
/// params, the options and the two culling margins of (116.7d), (116.7e).
#[derive(Clone, Copy)]
struct Grow<'a> {
    prims: &'a [Prim],
    p: SizeParams,
    opts: OctreeOptions,
    eta: f64,
    eta_d: f64,
}

/// The depth down to which a threaded build expands the tree on the
/// calling thread before it hands the subtrees below to the workers: at
/// most 8^2 = 64 subtrees (TET-15, DESIGN).
const PAR_DEPTH: u32 = 2;

/// A child's box and its candidates, as [`split_children`] returns them.
type Child = (Point, Point, Vec<u32>);

/// One node's split decision. `None` when the node is a leaf; otherwise
/// its eight children in bit order, each with its box and its candidates.
/// The node splits while the candidate count is above `leaf_cap`, the
/// depth is below `max_depth`, and the longest edge exceeds the node's own
/// lower bound of `h`, (116.7f); a child keeps its parent's candidates
/// minus those (116.7d) drops against the child's own bounds of (116.7c),
/// then minus the tied band-interior triangles that (116.7e) drops.
fn split_children(lo: Point, hi: Point, depth: u32, cand: &[u32], g: &Grow<'_>) -> Option<Vec<Child>> {
    let (prims, p, opts, eta, eta_d) = (g.prims, g.p, g.opts, g.eta, g.eta_d);
    let dx = hi[0] - lo[0];
    let dy = hi[1] - lo[1];
    let dz = hi[2] - lo[2];
    let s = p.gradation - 1.0;
    // (116.7f): `lambda(C)`, the lower bound of `h` on this node's own box,
    // over its own candidates on its own centre and half-diagonal; `+inf`
    // for an empty list, so `lambda` is `min(H, +inf)` = `H` then.
    let c = [
        0.5 * (lo[0] + hi[0]),
        0.5 * (lo[1] + hi[1]),
        0.5 * (lo[2] + hi[2]),
    ];
    let r = 0.5 * (dx * dx + dy * dy + dz * dz).sqrt();
    let mut m = f64::INFINITY;
    for &j in cand {
        let k = &prims[j as usize];
        let d = k.support.distance(c);
        m = m.min(k.h + s * (d - r - k.d_flat).max(0.0));
    }
    let lambda = p.min_size.max(p.base_size.min(m));
    let split = cand.len() > opts.leaf_cap && depth < opts.max_depth && dx.max(dy).max(dz) > lambda;
    if !split {
        return None;
    }
    let mid = c;
    let mut children: Vec<Child> = Vec::with_capacity(8);
    for bit in 0..8usize {
        let clo = [
            if bit & 1 != 0 { mid[0] } else { lo[0] },
            if bit & 2 != 0 { mid[1] } else { lo[1] },
            if bit & 4 != 0 { mid[2] } else { lo[2] },
        ];
        let chi = [
            if bit & 1 != 0 { hi[0] } else { mid[0] },
            if bit & 2 != 0 { hi[1] } else { mid[1] },
            if bit & 4 != 0 { hi[2] } else { mid[2] },
        ];
        let c = [
            0.5 * (clo[0] + chi[0]),
            0.5 * (clo[1] + chi[1]),
            0.5 * (clo[2] + chi[2]),
        ];
        let ex = chi[0] - clo[0];
        let ey = chi[1] - clo[1];
        let ez = chi[2] - clo[2];
        // The child's half-diagonal.
        let r = 0.5 * (ex * ex + ey * ey + ez * ez).sqrt();
        let mut u_min = f64::INFINITY;
        let mut ds: Vec<f64> = Vec::with_capacity(cand.len());
        for &j in cand {
            let k = &prims[j as usize];
            let d = k.support.distance(c);
            ds.push(d);
            u_min = u_min.min(k.h + s * (d + r - k.d_flat).max(0.0));
        }
        // Drop k when its lower bound passes this from above.
        let bound = p.base_size.min(u_min) + eta;
        let mut kept: Vec<u32> = Vec::new();
        let mut kd: Vec<f64> = Vec::new();
        for (idx, &j) in cand.iter().enumerate() {
            let k = &prims[j as usize];
            let l = k.h + s * (ds[idx] - r - k.d_flat).max(0.0);
            if l <= bound {
                kept.push(j);
                kd.push(ds[idx]);
            }
        }
        // (116.7e): inside a band every triangle of a patch source ties at
        // h_sigma, so (116.7d) can drop none of them. The first triangle of
        // a source, in candidate order, that fills the whole child inside
        // the band represents it, and the source's other triangles drop;
        // points and boxes never drop here.
        let mut rep: HashMap<u32, u32> = HashMap::new();
        for (idx, &j) in kept.iter().enumerate() {
            let k = &prims[j as usize];
            if matches!(k.support, Support::Tri(_))
                && kd[idx] + r + eta_d <= k.d_flat
                && !rep.contains_key(&k.src)
            {
                rep.insert(k.src, j);
            }
        }
        if !rep.is_empty() {
            kept = kept
                .into_iter()
                .filter(|&j| rep.get(&prims[j as usize].src).map_or(true, |&rj| rj == j))
                .collect();
        }
        children.push((clo, chi, kept));
    }
    Some(children)
}

/// A leaf, counted into the stats.
fn leaf(lo: Point, hi: Point, depth: u32, cand: Vec<u32>, st: &mut SizeFieldStats) -> Node {
    st.leaves += 1;
    st.depth = st.depth.max(depth);
    st.max_candidates = st.max_candidates.max(cand.len());
    st.mean_candidates += cand.len() as f64;
    Node {
        lo,
        hi,
        cand,
        children: None,
    }
}

/// An internal node over its eight children in bit order.
fn internal(lo: Point, hi: Point, children: Vec<Node>) -> Node {
    Node {
        lo,
        hi,
        cand: Vec::new(),
        children: Some(Box::new(
            children.try_into().expect("eight children built"),
        )),
    }
}

/// Build one node and everything below it on the calling thread.
fn build_node(
    lo: Point,
    hi: Point,
    depth: u32,
    cand: Vec<u32>,
    g: &Grow<'_>,
    st: &mut SizeFieldStats,
) -> Node {
    match split_children(lo, hi, depth, &cand, g) {
        None => leaf(lo, hi, depth, cand, st),
        Some(kids) => {
            let children: Vec<Node> = kids
                .into_iter()
                .map(|(clo, chi, kept)| build_node(clo, chi, depth + 1, kept, g, st))
                .collect();
            internal(lo, hi, children)
        }
    }
}

/// The tree above `PAR_DEPTH` while a threaded build runs: a finished
/// leaf, a subtree waiting in the worker list, or a split node.
enum Shape {
    Done(Node),
    Pending(usize),
    Split(Point, Point, Vec<Shape>),
}

/// A subtree [`expand`] queued: its box, its depth and its candidates.
type Queued = (Point, Point, u32, Vec<u32>);

/// Expand down to `PAR_DEPTH` on the calling thread, queueing each node
/// that reaches it, in the order the sequential build would visit them.
fn expand(
    lo: Point,
    hi: Point,
    depth: u32,
    cand: Vec<u32>,
    g: &Grow<'_>,
    pending: &mut Vec<Queued>,
    st: &mut SizeFieldStats,
) -> Shape {
    if depth >= PAR_DEPTH {
        pending.push((lo, hi, depth, cand));
        return Shape::Pending(pending.len() - 1);
    }
    match split_children(lo, hi, depth, &cand, g) {
        None => Shape::Done(leaf(lo, hi, depth, cand, st)),
        Some(kids) => Shape::Split(
            lo,
            hi,
            kids.into_iter()
                .map(|(clo, chi, kept)| expand(clo, chi, depth + 1, kept, g, pending, st))
                .collect(),
        ),
    }
}

/// Put the workers' subtrees back where [`expand`] queued them.
fn assemble(shape: Shape, built: &mut [Option<Node>]) -> Node {
    match shape {
        Shape::Done(n) => n,
        Shape::Pending(i) => built[i].take().expect("each queued subtree is built once"),
        Shape::Split(lo, hi, kids) => {
            internal(lo, hi, kids.into_iter().map(|k| assemble(k, built)).collect())
        }
    }
}

/// The root build (TET-15): on one worker exactly [`build_node`]; on more,
/// [`expand`] to `PAR_DEPTH`, the queued subtrees built by [`par_map`] each
/// into its own stats, and the stats merged. Every node is built by the
/// same code from the same inputs, so the tree is the same for every
/// worker count. The merge sums the leaf counts and the candidate total
/// (an integer held in an f64, exact below 2^53, so the order of the sum
/// does not matter) and takes the maxima of the depth and the candidate
/// count.
fn build_root(lo: Point, hi: Point, cand: Vec<u32>, g: &Grow<'_>, st: &mut SizeFieldStats) -> Node {
    let threads = resolve_threads(g.opts.threads);
    if threads <= 1 {
        return build_node(lo, hi, 0, cand, g, st);
    }
    let mut pending: Vec<Queued> = Vec::new();
    let shape = expand(lo, hi, 0, cand, g, &mut pending, st);
    let built: Vec<(Node, SizeFieldStats)> = par_map(pending.len(), threads, |i| {
        let (clo, chi, depth, ref kept) = pending[i];
        let mut sub = SizeFieldStats {
            primitives: 0,
            leaves: 0,
            depth: 0,
            max_candidates: 0,
            mean_candidates: 0.0,
        };
        let n = build_node(clo, chi, depth, kept.clone(), g, &mut sub);
        (n, sub)
    });
    let mut nodes: Vec<Option<Node>> = Vec::with_capacity(built.len());
    for (n, sub) in built {
        st.leaves += sub.leaves;
        st.depth = st.depth.max(sub.depth);
        st.max_candidates = st.max_candidates.max(sub.max_candidates);
        st.mean_candidates += sub.mean_candidates;
        nodes.push(Some(n));
    }
    assemble(shape, &mut nodes)
}

// ==========================================================================
//  Curvature and proximity on a triangulation (TET-04b), SPEC-LIT §116.8
// ==========================================================================

use super::surfprep::{segment_meets_triangle, TetSurface};
use std::collections::BTreeSet;

/// Which side of each triangle the gap of (116.10a) is measured on (DESIGN):
/// `Outward` a fluid gap between closed bodies whose normals point into the
/// fluid, `Inward` a channel inside a shell, `Both` also a solid's own
/// thickness, for when the fluid side is not yet known.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GapSide {
    /// Along `+n_t`, away from the solid.
    Outward,
    /// Along `-n_t`, into the solid.
    Inward,
    /// Along both directions.
    Both,
}

/// The input checks every TET-04b entry point runs, per §116.8: the patch
/// list matches the triangles, every triangle names existing points, its
/// patch id exists, and every point a triangle uses is finite. Open and
/// non-manifold surfaces are fine - those edges just carry no curvature.
fn validate_size_surface(surf: &TetSurface) -> Result<()> {
    if surf.tri_patch.len() != surf.tris.len() {
        return Err(Error::Mesh(format!(
            "size field: tri_patch has {} entries for {} triangles",
            surf.tri_patch.len(),
            surf.tris.len()
        )));
    }
    for (t, tri) in surf.tris.iter().enumerate() {
        for &i in tri.iter() {
            if i as usize >= surf.points.len() {
                return Err(Error::Mesh(format!(
                    "size field: triangle {t} references point {i}, but there are {} points",
                    surf.points.len()
                )));
            }
        }
        let p = surf.tri_patch[t];
        if p as usize >= surf.patch_names.len() {
            return Err(Error::Mesh(format!(
                "size field: triangle {t} has patch id {p}, but there are {} patches",
                surf.patch_names.len()
            )));
        }
        for &i in tri.iter() {
            let x = surf.points[i as usize];
            if !x.iter().all(|v| v.is_finite()) {
                return Err(Error::Mesh(format!(
                    "size field: triangle {t} uses point {i} with a coordinate that is not finite"
                )));
            }
        }
    }
    Ok(())
}

/// The crease angle's validity, §116.8: finite and inside (0, 180].
fn validate_crease_deg(crease_deg: f64) -> Result<()> {
    if !crease_deg.is_finite() || crease_deg <= 0.0 || crease_deg > 180.0 {
        return Err(Error::Mesh(format!(
            "size field: crease_deg {crease_deg} must be finite and inside (0, 180]"
        )));
    }
    Ok(())
}

/// The undirected edge key, as `surfprep`'s.
fn edge_key(a: u32, b: u32) -> (u32, u32) {
    if a < b {
        (a, b)
    } else {
        (b, a)
    }
}

/// The f64 cross product, as `surfprep`'s.
fn cross3(a: Point, b: Point) -> Point {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

/// A triangle's unit normal `(x1 - x0) x (x2 - x0) / |..|` and its area,
/// half the norm. The exactly-zero cross is the zero-area triangle,
/// reported as the zero normal and area 0: it carries no curvature, no
/// gap, and is never a hit target.
fn unit_normal_area(t: &[Point; 3]) -> (Point, f64) {
    let c = cross3(sub(t[1], t[0]), sub(t[2], t[0]));
    let nn = norm(c);
    if nn == 0.0 {
        return ([0.0; 3], 0.0);
    }
    ([c[0] / nn, c[1] / nn, c[2] / nn], 0.5 * nn)
}

/// The dihedral angle (116.9a) between two unit normals, in [0, pi].
fn dihedral_angle(n1: Point, n2: Point) -> f64 {
    norm(cross3(n1, n2)).atan2(dot(n1, n2))
}

/// A triangle's three points.
fn tri_pts(surf: &TetSurface, t: usize) -> [Point; 3] {
    [
        surf.points[surf.tris[t][0] as usize],
        surf.points[surf.tris[t][1] as usize],
        surf.points[surf.tris[t][2] as usize],
    ]
}

/// A triangle's patch name, valid by `validate_size_surface`.
fn patch_of(surf: &TetSurface, t: usize) -> &str {
    &surf.patch_names[surf.tri_patch[t] as usize]
}

/// kappa_t per triangle, (116.9a)-(116.9c), in triangle order; 0.0 where
/// none. An edge shared by exactly two non-zero-area triangles carries the
/// estimate (116.9b) unless it is a crease, `phi_e >= crease_deg`; a
/// triangle takes the largest estimate over its edge star, its own three
/// edges and every edge of each triangle sharing a non-crease edge with
/// it. Open and non-manifold edges carry no estimate and never panic.
pub fn triangle_curvatures(surf: &TetSurface, crease_deg: f64) -> Result<Vec<f64>> {
    validate_size_surface(surf)?;
    validate_crease_deg(crease_deg)?;
    Ok(curvatures_validated(surf, crease_deg.to_radians(), 0))
}

/// The curvature pass on a validated surface, the crease angle in radians;
/// the per-triangle stars run on `threads` workers through [`par_map`].
fn curvatures_validated(surf: &TetSurface, crease_rad: f64, threads: usize) -> Vec<f64> {
    let n = surf.tris.len();
    let mut normal = vec![[0.0; 3]; n];
    let mut area = vec![0.0f64; n];
    for i in 0..n {
        let (nrm, a) = unit_normal_area(&tri_pts(surf, i));
        normal[i] = nrm;
        area[i] = a;
    }
    let mut edges: HashMap<(u32, u32), Vec<u32>> = HashMap::new();
    for (i, t) in surf.tris.iter().enumerate() {
        for (a, b) in [(t[0], t[1]), (t[1], t[2]), (t[2], t[0])] {
            edges.entry(edge_key(a, b)).or_default().push(i as u32);
        }
    }

    // The defined estimates (116.9b): an edge with exactly two triangles,
    // both of positive area, that is not a crease. A crease, an edge with
    // one or with three or more triangles, and an edge next to a zero-area
    // triangle stay out, so they carry no estimate.
    let mut ekappa: HashMap<(u32, u32), f64> = HashMap::new();
    for (e, ts) in &edges {
        if ts.len() != 2 {
            continue;
        }
        let (i, j) = (ts[0] as usize, ts[1] as usize);
        if area[i] <= 0.0 || area[j] <= 0.0 {
            continue;
        }
        let phi = dihedral_angle(normal[i], normal[j]);
        if phi >= crease_rad {
            continue;
        }
        let el = norm(sub(surf.points[e.0 as usize], surf.points[e.1 as usize]));
        ekappa.insert(*e, 3.0 * phi * el / (2.0 * (area[i] + area[j])));
    }

    // kappa_t (116.9c): the largest estimate over the edge star, the
    // triangle's own three edges plus every edge of each triangle sharing
    // a non-crease (estimated) edge with it.
    par_map(n, threads, |i| {
        let mut out_i = 0.0f64;
        if area[i] == 0.0 {
            return out_i;
        }
        let t = &surf.tris[i];
        let own = [(t[0], t[1]), (t[1], t[2]), (t[2], t[0])];
        let mut star: BTreeSet<(u32, u32)> = BTreeSet::new();
        for (a, b) in own {
            star.insert(edge_key(a, b));
        }
        for (a, b) in own {
            let e = edge_key(a, b);
            if !ekappa.contains_key(&e) {
                continue;
            }
            for &j in &edges[&e] {
                if j as usize == i {
                    continue;
                }
                let tn = &surf.tris[j as usize];
                for (a2, b2) in [(tn[0], tn[1]), (tn[1], tn[2]), (tn[2], tn[0])] {
                    star.insert(edge_key(a2, b2));
                }
            }
        }
        for e in &star {
            if let Some(&k) = ekappa.get(e) {
                out_i = out_i.max(k);
            }
        }
        out_i
    })
}

/// gap_t per triangle, (116.10a) with the facing rule; None where there is
/// no gap. From each triangle's centroid, along every direction the side
/// allows, the closed segment to `c_t + L v` with `L` twice the bounding
/// box's diagonal takes its FIRST hit; it counts when the two sheets face
/// each other, and gap_t is the smallest gap over the directions.
pub fn triangle_gaps(surf: &TetSurface, side: GapSide) -> Result<Vec<Option<f64>>> {
    validate_size_surface(surf)?;
    let scanner = GapScanner::new(surf);
    Ok(par_map(surf.tris.len(), 0, |t| scanner.gap_of(t, side)))
}

/// One node of the gap BVH: the union box of its triangles and either its
/// two children or its range in `GapBvh::order`. The median split of the
/// centroids runs on the node box's longest axis; a leaf holds at most
/// four triangles.
struct GapNode {
    lo: Point,
    hi: Point,
    children: Option<(u32, u32)>,
    start: u32,
    len: u32,
}

/// A bounding-volume hierarchy over the non-zero-area triangles' boxes
/// (DESIGN, §116.8): it only culls. The slab test accepts every box the
/// closed segment truly meets, so its candidate set is a superset of the
/// brute scan's, the per-candidate arithmetic is (116.10a)'s own, and the
/// reduction takes the lowest index on a tie by rule, not by order - so a
/// first hit through the hierarchy equals the scan over every triangle
/// bit for bit.
struct GapBvh {
    nodes: Vec<GapNode>,
    order: Vec<u32>,
}

impl GapBvh {
    fn build(tri: &[[Point; 3]], ok: &[bool]) -> GapBvh {
        let mut boxes: Vec<(Point, Point)> = Vec::new();
        let mut cents: Vec<Point> = Vec::new();
        for (i, t) in tri.iter().enumerate() {
            if !ok[i] {
                continue;
            }
            boxes.push((
                [
                    t[0][0].min(t[1][0]).min(t[2][0]),
                    t[0][1].min(t[1][1]).min(t[2][1]),
                    t[0][2].min(t[1][2]).min(t[2][2]),
                ],
                [
                    t[0][0].max(t[1][0]).max(t[2][0]),
                    t[0][1].max(t[1][1]).max(t[2][1]),
                    t[0][2].max(t[1][2]).max(t[2][2]),
                ],
            ));
            cents.push([
                (t[0][0] + t[1][0] + t[2][0]) / 3.0,
                (t[0][1] + t[1][1] + t[2][1]) / 3.0,
                (t[0][2] + t[1][2] + t[2][2]) / 3.0,
            ]);
        }
        let mut idx: Vec<u32> = (0..boxes.len() as u32).collect();
        let mut bvh = GapBvh {
            nodes: Vec::new(),
            order: Vec::new(),
        };
        Self::split(&boxes, &cents, &mut idx, &mut bvh);
        bvh
    }

    fn split(
        boxes: &[(Point, Point)],
        cents: &[Point],
        idx: &mut [u32],
        bvh: &mut GapBvh,
    ) -> u32 {
        let mut lo = [f64::INFINITY; 3];
        let mut hi = [f64::NEG_INFINITY; 3];
        for &i in idx.iter() {
            let (blo, bhi) = &boxes[i as usize];
            for a in 0..3 {
                lo[a] = lo[a].min(blo[a]);
                hi[a] = hi[a].max(bhi[a]);
            }
        }
        let id = bvh.nodes.len() as u32;
        bvh.nodes.push(GapNode {
            lo,
            hi,
            children: None,
            start: 0,
            len: 0,
        });
        if idx.len() <= 4 {
            let node = &mut bvh.nodes[id as usize];
            node.start = bvh.order.len() as u32;
            node.len = idx.len() as u32;
            bvh.order.extend_from_slice(idx);
            return id;
        }
        let mut axis = 0usize;
        let mut wide = hi[0] - lo[0];
        for a in 1..3 {
            let w = hi[a] - lo[a];
            if w > wide {
                wide = w;
                axis = a;
            }
        }
        idx.sort_by(|&i, &j| {
            cents[i as usize][axis]
                .partial_cmp(&cents[j as usize][axis])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let mid = idx.len() / 2;
        let (l, r) = idx.split_at_mut(mid);
        let lc = Self::split(boxes, cents, l, bvh);
        let rc = Self::split(boxes, cents, r, bvh);
        bvh.nodes[id as usize].children = Some((lc, rc));
        id
    }

    /// Every leaf triangle whose box the closed segment `p q` meets.
    fn collect(&self, p: Point, q: Point, out: &mut Vec<u32>) {
        let mut stack = vec![0u32];
        while let Some(id) = stack.pop() {
            let n = &self.nodes[id as usize];
            if !segment_meets_box(p, q, n.lo, n.hi) {
                continue;
            }
            match n.children {
                None => out.extend_from_slice(
                    &self.order[n.start as usize..(n.start + n.len) as usize],
                ),
                Some((l, r)) => {
                    stack.push(l);
                    stack.push(r);
                }
            }
        }
    }
}

/// The slab test of the closed segment against the closed box, with a
/// tolerance of 1e-12 on the parameter interval, far above the test's own
/// rounding: it accepts every box the segment truly meets, and the exact
/// triangle test then decides on the extra few.
fn segment_meets_box(p: Point, q: Point, lo: Point, hi: Point) -> bool {
    let mut tmin = 0.0f64;
    let mut tmax = 1.0f64;
    for a in 0..3 {
        let d = q[a] - p[a];
        if d == 0.0 {
            if p[a] < lo[a] || p[a] > hi[a] {
                return false;
            }
        } else {
            let (mut t1, mut t2) = ((lo[a] - p[a]) / d, (hi[a] - p[a]) / d);
            if t1 > t2 {
                std::mem::swap(&mut t1, &mut t2);
            }
            tmin = tmin.max(t1);
            tmax = tmax.min(t2);
            if tmin > tmax + 1e-12 {
                return false;
            }
        }
    }
    true
}

/// Everything the gap query needs, precomputed once per surface: the f64
/// triangles, their unit normals (the zero normal marks a zero-area
/// triangle), their centroids, the segment length `L`, and the BVH.
pub(super) struct GapScanner {
    tri: Vec<[Point; 3]>,
    normal: Vec<Point>,
    centroid: Vec<Point>,
    ok: Vec<bool>,
    len: f64,
    bvh: GapBvh,
}

impl GapScanner {
    pub(super) fn new(surf: &TetSurface) -> GapScanner {
        let n = surf.tris.len();
        let mut tri: Vec<[Point; 3]> = Vec::with_capacity(n);
        let mut normal = vec![[0.0; 3]; n];
        let mut centroid = vec![[0.0; 3]; n];
        let mut ok = vec![false; n];
        for i in 0..n {
            let t = tri_pts(surf, i);
            let (nrm, a) = unit_normal_area(&t);
            tri.push(t);
            normal[i] = nrm;
            ok[i] = a > 0.0;
            centroid[i] = [
                (t[0][0] + t[1][0] + t[2][0]) / 3.0,
                (t[0][1] + t[1][1] + t[2][1]) / 3.0,
                (t[0][2] + t[1][2] + t[2][2]) / 3.0,
            ];
        }
        let mut lo = surf.points.first().copied().unwrap_or([0.0; 3]);
        let mut hi = lo;
        for p in &surf.points {
            for a in 0..3 {
                lo[a] = lo[a].min(p[a]);
                hi[a] = hi[a].max(p[a]);
            }
        }
        let len = 2.0 * norm(sub(hi, lo));
        GapScanner {
            bvh: GapBvh::build(&tri, &ok),
            tri,
            normal,
            centroid,
            ok,
            len,
        }
    }

    /// The reduction both paths share: the first hit among `cands`, the
    /// smallest tau (116.10a), ties to the lowest index by rule, skipping
    /// the start triangle, zero-area triangles, misses and non-finite
    /// taus. Order-independent, so the BVH and the brute scan agree bit
    /// for bit.
    fn scan(
        &self,
        p: Point,
        q: Point,
        v: Point,
        skip: &dyn Fn(usize) -> bool,
        cands: &[u32],
    ) -> Option<(usize, f64)> {
        let mut best: Option<(usize, f64)> = None;
        for &si in cands {
            let s = si as usize;
            if skip(s) || !self.ok[s] {
                continue;
            }
            if !segment_meets_triangle(&p, &q, &self.tri[s]) {
                continue;
            }
            let ns = self.normal[s];
            let tau = dot(ns, sub(self.tri[s][0], p)) / dot(ns, v);
            if !tau.is_finite() {
                continue;
            }
            best = match best {
                Some((bs, bt)) if bt < tau || (bt == tau && bs < s) => Some((bs, bt)),
                _ => Some((s, tau)),
            };
        }
        best
    }

    /// The first hit through the BVH.
    fn first_hit(&self, p: Point, q: Point, v: Point, skip: usize) -> Option<(usize, f64)> {
        let mut cands: Vec<u32> = Vec::new();
        self.bvh.collect(p, q, &mut cands);
        self.scan(p, q, v, &|s| s == skip, &cands)
    }

    /// The first hit of the segment from `p` to `p + L v` (`L` of
    /// [`GapScanner::new`]) over the triangles `skip` does not reject,
    /// through the BVH, by the same reduction as [`GapScanner::first_hit`].
    /// §116.15's gap (116.24a) starts at a wall point and skips every
    /// triangle incident to it.
    pub(super) fn first_hit_except(
        &self,
        p: Point,
        v: Point,
        skip: &dyn Fn(usize) -> bool,
    ) -> Option<(usize, f64)> {
        let q = add(p, scale(v, self.len));
        let mut cands: Vec<u32> = Vec::new();
        self.bvh.collect(p, q, &mut cands);
        self.scan(p, q, v, skip, &cands)
    }

    /// The gap of one triangle, (116.10a) with the facing rule: only the
    /// first hit counts, and only when `|n_s . n_t| >= cos 45 deg`; the
    /// gap is the smallest over the directions that gave one.
    fn gap_of(&self, t: usize, side: GapSide) -> Option<f64> {
        if !self.ok[t] {
            return None;
        }
        let nt = self.normal[t];
        let dirs: &[Point] = match side {
            GapSide::Outward => &[nt],
            GapSide::Inward => &[scale(nt, -1.0)],
            GapSide::Both => &[nt, scale(nt, -1.0)],
        };
        let mut gap: Option<f64> = None;
        for &v in dirs {
            let p = self.centroid[t];
            let q = add(p, scale(v, self.len));
            if let Some((s, tau)) = self.first_hit(p, q, v, t) {
                if dot(self.normal[s], nt).abs() >= std::f64::consts::FRAC_1_SQRT_2 {
                    gap = Some(gap.map_or(tau, |g| g.min(tau)));
                }
            }
        }
        gap
    }
}

/// The brute-force twin the BVH must match bit for bit: the same scan
/// over every triangle. Test-only, so the shipped library carries no dead
/// linear scan.
#[cfg(test)]
impl GapScanner {
    fn first_hit_brute(&self, p: Point, q: Point, v: Point, skip: usize) -> Option<(usize, f64)> {
        let all: Vec<u32> = (0..self.tri.len() as u32).collect();
        self.scan(p, q, v, &|s| s == skip, &all)
    }
}

/// One `SizeSource::Patch` per triangle with a curvature primitive,
/// §116.8: `h_k = 2 pi / (n_curv kappa_t)` at `D_k = 0` on the triangle
/// alone, named `curvature:<patch name>`. `n_curv == 0` turns the source
/// off and returns an empty vector; a triangle with `kappa_t == 0`, and
/// one whose size has reached `base_size`, get none, by §116.8's cap
/// argument.
pub fn curvature_sources(
    surf: &TetSurface,
    params: &SizeParams,
    n_curv: u32,
    crease_deg: f64,
) -> Result<Vec<SizeSource>> {
    curvature_sources_t(surf, params, n_curv, crease_deg, 0)
}

/// [`curvature_sources`] on `threads` workers (`0`: every core).
fn curvature_sources_t(
    surf: &TetSurface,
    params: &SizeParams,
    n_curv: u32,
    crease_deg: f64,
    threads: usize,
) -> Result<Vec<SizeSource>> {
    validate_size_surface(surf)?;
    validate_crease_deg(crease_deg)?;
    if n_curv == 0 {
        return Ok(Vec::new());
    }
    let kap = curvatures_validated(surf, crease_deg.to_radians(), threads);
    let mut out: Vec<SizeSource> = Vec::new();
    for (i, &k) in kap.iter().enumerate() {
        if k == 0.0 {
            continue;
        }
        let size = std::f64::consts::TAU / (f64::from(n_curv) * k);
        if size >= params.base_size {
            continue;
        }
        out.push(SizeSource::Patch {
            name: format!("curvature:{}", patch_of(surf, i)),
            tris: vec![tri_pts(surf, i)],
            distance: 0.0,
            size,
        });
    }
    Ok(out)
}

/// One `SizeSource::Patch` per triangle with a gap, §116.8: `h_k =
/// gap_t / n_gap` held flat out to `D_k = gap_t / 2` (116.10b), named
/// `proximity:<patch name>`. `n_gap == 0` turns the source off; a
/// triangle with no gap, one with `gap_t <= 0`, and one whose size has
/// reached `base_size`, get none.
pub fn proximity_sources(
    surf: &TetSurface,
    params: &SizeParams,
    n_gap: u32,
    side: GapSide,
) -> Result<Vec<SizeSource>> {
    proximity_sources_t(surf, params, n_gap, side, 0)
}

/// [`proximity_sources`] on `threads` workers (`0`: every core): the gaps
/// of (116.10a) through [`par_map`], the sources from them in triangle
/// order on the calling thread.
fn proximity_sources_t(
    surf: &TetSurface,
    params: &SizeParams,
    n_gap: u32,
    side: GapSide,
    threads: usize,
) -> Result<Vec<SizeSource>> {
    validate_size_surface(surf)?;
    if n_gap == 0 {
        return Ok(Vec::new());
    }
    let scanner = GapScanner::new(surf);
    let gaps = par_map(surf.tris.len(), threads, |t| scanner.gap_of(t, side));
    let mut out: Vec<SizeSource> = Vec::new();
    for (t, g) in gaps.into_iter().enumerate() {
        let gap = match g {
            Some(g) if g > 0.0 => g,
            _ => continue,
        };
        let size = gap / f64::from(n_gap);
        if size >= params.base_size {
            continue;
        }
        out.push(SizeSource::Patch {
            name: format!("proximity:{}", patch_of(surf, t)),
            tris: vec![scanner.tri[t]],
            distance: gap / 2.0,
            size,
        });
    }
    Ok(out)
}

/// `curvature_sources` on `spec.curvature_cells`, then
/// `proximity_sources` on `spec.gap_cells`, in that order.
pub fn surface_sources(
    surf: &TetSurface,
    params: &SizeParams,
    spec: &TetSizeSpec,
    crease_deg: f64,
    side: GapSide,
) -> Result<Vec<SizeSource>> {
    surface_sources_threads(surf, params, spec, crease_deg, side, 0)
}

/// [`surface_sources`] on `threads` workers (`0`: every core, TET-15). The
/// list is the same, source for source and bit for bit, for every count.
pub fn surface_sources_threads(
    surf: &TetSurface,
    params: &SizeParams,
    spec: &TetSizeSpec,
    crease_deg: f64,
    side: GapSide,
    threads: usize,
) -> Result<Vec<SizeSource>> {
    let mut out = curvature_sources_t(surf, params, spec.curvature_cells, crease_deg, threads)?;
    out.extend(proximity_sources_t(surf, params, spec.gap_cells, side, threads)?);
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::automesher::{DistanceBand, RefinementBand, RefinementBox, RefinementSpec};
    use std::f64::consts::{PI, SQRT_2};
    use std::time::Instant;

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

    fn rel_ok(got: f64, want: f64, e: f64) -> bool {
        (got - want).abs() <= e * want.abs()
    }

    fn sq() -> Vec<[Point; 3]> {
        vec![
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0]],
            [[0.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]],
        ]
    }

    fn corners(lo: Point, hi: Point) -> Vec<Point> {
        (0..8usize)
            .map(|b| {
                [
                    if b & 1 != 0 { hi[0] } else { lo[0] },
                    if b & 2 != 0 { hi[1] } else { lo[1] },
                    if b & 4 != 0 { hi[2] } else { lo[2] },
                ]
            })
            .collect()
    }

    /// The UV sphere: 9 rows by 17 columns, the poles exact, 256 triangles
    /// of which the 32 pole ones are zero-area and stay.
    fn uv_sphere(r: f64) -> Vec<[Point; 3]> {
        let mut v: Vec<Point> = Vec::with_capacity(9 * 17);
        for i in 0..=8u32 {
            for j in 0..=16u32 {
                let p = if i == 0 {
                    [0.0, 0.0, r]
                } else if i == 8 {
                    [0.0, 0.0, -r]
                } else {
                    let th = PI * f64::from(i) / 8.0;
                    let ph = 2.0 * PI * f64::from(j) / 16.0;
                    [
                        r * th.sin() * ph.cos(),
                        r * th.sin() * ph.sin(),
                        r * th.cos(),
                    ]
                };
                v.push(p);
            }
        }
        let at = |i: u32, j: u32| v[(i * 17 + j) as usize];
        let mut tris: Vec<[Point; 3]> = Vec::with_capacity(256);
        for i in 0..8u32 {
            for j in 0..16u32 {
                tris.push([at(i, j), at(i + 1, j), at(i + 1, j + 1)]);
                tris.push([at(i, j), at(i + 1, j + 1), at(i, j + 1)]);
            }
        }
        tris
    }

    /// The mixed fixture: two ball patches, the floor patch, a box and a
    /// point, 516 primitives in all.
    fn mix_sources() -> Vec<SizeSource> {
        let f = |p: Point| [4.0 * p[0] - 2.0, 4.0 * p[1] - 2.0, -1.0];
        vec![
            SizeSource::Patch {
                name: "ball".to_string(),
                tris: uv_sphere(0.5),
                distance: 0.0,
                size: 0.02,
            },
            SizeSource::Patch {
                name: "ball".to_string(),
                tris: uv_sphere(0.5),
                distance: 0.4,
                size: 0.06,
            },
            SizeSource::Patch {
                name: "floor".to_string(),
                tris: sq().iter().map(|t| [f(t[0]), f(t[1]), f(t[2])]).collect(),
                distance: 0.0,
                size: 0.08,
            },
            SizeSource::Box {
                min: [0.5, -0.25, -0.25],
                max: [1.5, 0.25, 0.25],
                size: 0.04,
            },
            SizeSource::Point {
                at: [-1.0, 1.0, 0.5],
                size: 0.01,
            },
        ]
    }

    fn mix() -> SizeField {
        SizeField::build(
            SizeParams::new(0.5, 1.15, None).unwrap(),
            &mix_sources(),
            [-2.0, -2.0, -1.5],
            [2.0, 2.0, 1.5],
            OctreeOptions::default(),
        )
        .unwrap()
    }

    fn rebuild(leaf_cap: usize) -> SizeField {
        SizeField::build(
            SizeParams::new(0.5, 1.15, None).unwrap(),
            &mix_sources(),
            [-2.0, -2.0, -1.5],
            [2.0, 2.0, 1.5],
            OctreeOptions {
                leaf_cap,
                max_depth: 12,
                threads: 0,
            },
        )
        .unwrap()
    }

    fn mesh_msg(r: Result<Vec<SizeSource>>) -> String {
        match r {
            Err(Error::Mesh(m)) => m,
            _ => panic!("expected Error::Mesh"),
        }
    }

    fn field_msg(r: Result<SizeField>) -> String {
        match r {
            Err(Error::Mesh(m)) => m,
            _ => panic!("expected Error::Mesh"),
        }
    }

    fn params_msg(r: Result<SizeParams>) -> String {
        match r {
            Err(Error::Mesh(m)) => m,
            _ => panic!("expected Error::Mesh"),
        }
    }

    #[test]
    fn point_source_cone() {
        let s = 1.2f64 - 1.0;
        let h_min: f64 = 1.0 / 64.0;
        // a: one point at the origin, the gate of §116.18.
        let field = SizeField::build(
            SizeParams::new(1.0, 1.2, None).unwrap(),
            &[SizeSource::Point {
                at: [0.0, 0.0, 0.0],
                size: 0.05,
            }],
            [-6.0, -6.0, -6.0],
            [6.0, 6.0, 6.0],
            OctreeOptions::default(),
        )
        .unwrap();
        let want = |x: Point| h_min.max(1.0f64.min(0.05 + s * norm(sub(x, [0.0; 3]))));
        let mut rng = SplitMix64::new(41);
        let mut max_rel_a = 0.0f64;
        for _ in 0..100_000 {
            let x = rng.point([-6.0; 3], [6.0; 3]);
            let got = field.h(x);
            let w = want(x);
            assert!(rel_ok(got, w, 0.02), "a: h({x:?}) = {got}, want {w}");
            max_rel_a = max_rel_a.max((got - w).abs() / w.abs());
        }
        for _ in 0..10_000 {
            let x = rng.point([-0.1; 3], [0.1; 3]);
            let got = field.h(x);
            let w = want(x);
            assert!(rel_ok(got, w, 0.02), "a near: h({x:?}) = {got}, want {w}");
            max_rel_a = max_rel_a.max((got - w).abs() / w.abs());
        }
        for x in corners([-6.0; 3], [6.0; 3]).into_iter().chain([[0.0; 3]]) {
            let got = field.h(x);
            let w = want(x);
            assert!(rel_ok(got, w, 0.02), "a corner: h({x:?}) = {got}, want {w}");
            max_rel_a = max_rel_a.max((got - w).abs() / w.abs());
        }
        assert!(max_rel_a <= 1e-12, "a max rel {max_rel_a}");
        assert_eq!(field.h([0.0, 0.0, 0.0]), 0.05);
        assert_eq!(field.h([6.0, 6.0, 6.0]), 1.0);
        assert!(rel_ok(field.h([2.0, 0.0, 0.0]), 0.05 + s * 2.0, 1e-15));
        // b: the same at a far offset.
        let o = [1000.25, -2000.5, 30.125];
        let field = SizeField::build(
            SizeParams::new(1.0, 1.2, None).unwrap(),
            &[SizeSource::Point { at: o, size: 0.05 }],
            [o[0] - 6.0, o[1] - 6.0, o[2] - 6.0],
            [o[0] + 6.0, o[1] + 6.0, o[2] + 6.0],
            OctreeOptions::default(),
        )
        .unwrap();
        let want = |x: Point| h_min.max(1.0f64.min(0.05 + s * norm(sub(x, o))));
        let mut rng = SplitMix64::new(42);
        let mut max_rel_b = 0.0f64;
        for _ in 0..100_000 {
            let x = rng.point([o[0] - 6.0, o[1] - 6.0, o[2] - 6.0], [o[0] + 6.0, o[1] + 6.0, o[2] + 6.0]);
            let got = field.h(x);
            let w = want(x);
            assert!(rel_ok(got, w, 0.02), "b: h({x:?}) = {got}, want {w}");
            max_rel_b = max_rel_b.max((got - w).abs() / w.abs());
        }
        assert!(max_rel_b <= 1e-12, "b max rel {max_rel_b}");
        // c: 64 point sources of graded sizes.
        let mut pts: Vec<(Point, f64)> = Vec::new();
        for k in 0..4u32 {
            for j in 0..4u32 {
                for i in 0..4u32 {
                    let n = i + 4 * j + 16 * k;
                    pts.push((
                        [
                            f64::from(i) - 1.5,
                            f64::from(j) - 1.5,
                            f64::from(k) - 1.5,
                        ],
                        0.02 + 0.001 * f64::from(n),
                    ));
                }
            }
        }
        let sources: Vec<SizeSource> = pts
            .iter()
            .map(|(at, size)| SizeSource::Point { at: *at, size: *size })
            .collect();
        let field = SizeField::build(
            SizeParams::new(1.0, 1.2, None).unwrap(),
            &sources,
            [-3.0; 3],
            [3.0; 3],
            OctreeOptions::default(),
        )
        .unwrap();
        let want = |x: Point| {
            let mut m = f64::INFINITY;
            for (p, size) in &pts {
                m = m.min(size + s * norm(sub(x, *p)));
            }
            h_min.max(1.0f64.min(m))
        };
        let mut rng = SplitMix64::new(43);
        let mut max_rel_c = 0.0f64;
        for _ in 0..100_000 {
            let x = rng.point([-3.0; 3], [3.0; 3]);
            let got = field.h(x);
            let w = want(x);
            assert!(rel_ok(got, w, 0.02), "c: h({x:?}) = {got}, want {w}");
            max_rel_c = max_rel_c.max((got - w).abs() / w.abs());
            let (a, b) = field.leaf_box(x).expect("c: leaf box inside the root");
            for q in 0..3 {
                assert!(a[q] <= x[q] && x[q] <= b[q], "c: leaf box misses {x:?}");
            }
        }
        assert!(max_rel_c <= 1e-12, "c max rel {max_rel_c}");
        assert!(field.stats().leaves > 1);
        println!(
            "TET04A-CONE max_rel_a={max_rel_a} max_rel_b={max_rel_b} max_rel_c={max_rel_c} leaves_c={}",
            field.stats().leaves
        );
    }

    #[test]
    fn box_exact() {
        let s = 1.2f64 - 1.0;
        let h_min: f64 = 1.0 / 64.0;
        let blo = [-1.0, -0.5, 0.0];
        let bhi = [1.0, 0.5, 2.0];
        let sources = vec![
            SizeSource::Box {
                min: blo,
                max: bhi,
                size: 0.125,
            },
            SizeSource::Point {
                at: [3.0, 0.0, 1.0],
                size: 0.02,
            },
        ];
        let field = SizeField::build(
            SizeParams::new(1.0, 1.2, None).unwrap(),
            &sources,
            [-4.0; 3],
            [4.0; 3],
            OctreeOptions::default(),
        )
        .unwrap();
        // a: inside the box the answer is the box size, bit for bit.
        let mut rng = SplitMix64::new(44);
        let want_bits = 0.125f64.to_bits();
        let mut check_inside = |x: Point| {
            assert_eq!(field.h(x).to_bits(), want_bits, "inside {x:?}");
        };
        for _ in 0..100_000 {
            check_inside(rng.point(blo, bhi));
        }
        for x in corners(blo, bhi) {
            check_inside(x);
        }
        for a in 0..3 {
            for edge in [blo[a], bhi[a]] {
                let mut f = [0.0, 0.0, 1.0];
                f[a] = edge;
                check_inside(f);
            }
        }
        // b: outside, the cone of (116.7a) over the box distance.
        let mut rng = SplitMix64::new(45);
        let mut kept = 0usize;
        while kept < 10_000 {
            let x = rng.point([-4.0; 3], [4.0; 3]);
            let dbox = point_box_distance(x, blo, bhi);
            if dbox > 0.0 {
                kept += 1;
                let dp = norm(sub(x, [3.0, 0.0, 1.0]));
                let w = h_min.max(1.0f64.min((0.125 + s * dbox).min(0.02 + s * dp)));
                let got = field.h(x);
                assert!(rel_ok(got, w, 1e-12), "b: h({x:?}) = {got}, want {w}");
            }
        }
        assert!(rel_ok(field.h([2.0, 0.0, 1.0]), 0.02 + s * 1.0, 1e-15));
        assert!(rel_ok(field.h([-3.0, 0.0, 1.0]), 0.125 + s * 2.0, 1e-15));
        // d: a smaller point source on the box face wins inside.
        let mut sources3 = sources.clone();
        sources3.push(SizeSource::Point {
            at: [0.0, 0.0, 1.0],
            size: 0.03,
        });
        let field3 = SizeField::build(
            SizeParams::new(1.0, 1.2, None).unwrap(),
            &sources3,
            [-4.0; 3],
            [4.0; 3],
            OctreeOptions::default(),
        )
        .unwrap();
        assert_eq!(field3.h([0.0, 0.0, 1.0]), 0.03);
        let mut rng = SplitMix64::new(46);
        for _ in 0..10_000 {
            let x = rng.point(blo, bhi);
            let got = field3.h(x);
            assert!(got <= 0.125, "d: h({x:?}) = {got} above the box size");
        }
    }

    #[test]
    fn patch_surface_and_band() {
        let s = 1.2f64 - 1.0;
        let params = SizeParams::new(1.0, 1.2, None).unwrap();
        let mk = |distance: f64, size: f64| {
            SizeField::build(
                params,
                &[SizeSource::Patch {
                    name: "floor".to_string(),
                    tris: sq(),
                    distance,
                    size,
                }],
                [-3.0; 3],
                [3.0; 3],
                OctreeOptions::default(),
            )
            .unwrap()
        };
        let fa = mk(0.0, 0.05);
        let fb = mk(0.3, 0.1);
        let fab = SizeField::build(
            params,
            &[
                SizeSource::Patch {
                    name: "floor".to_string(),
                    tris: sq(),
                    distance: 0.0,
                    size: 0.05,
                },
                SizeSource::Patch {
                    name: "floor".to_string(),
                    tris: sq(),
                    distance: 0.3,
                    size: 0.1,
                },
            ],
            [-3.0; 3],
            [3.0; 3],
            OctreeOptions::default(),
        )
        .unwrap();
        let cases: &[(&SizeField, Point, f64)] = &[
            (&fa, [0.5, 0.5, 0.0], 0.05),
            (&fa, [0.3, 0.7, 0.0], 0.05),
            (&fa, [0.5, 0.5, 0.1], 0.05 + s * 0.1),
            (&fa, [0.5, 0.5, -0.25], 0.05 + s * 0.25),
            (&fa, [0.5, 0.5, 1.0], 0.05 + s * 1.0),
            (&fa, [1.5, 0.5, 0.0], 0.05 + s * 0.5),
            (&fa, [2.0, 2.0, 0.0], 0.05 + s * SQRT_2),
            (&fb, [0.5, 0.5, 0.2], 0.1),
            (&fb, [0.5, 0.5, 0.3], 0.1),
            (&fb, [0.5, 0.5, 0.8], 0.1 + s * 0.5),
            (&fb, [0.5, 0.5, 2.9], 0.1 + s * 2.6),
            (&fab, [0.5, 0.5, 0.1], 0.05 + s * 0.1),
            (&fab, [0.5, 0.5, 0.3], 0.1),
            (&fab, [0.5, 0.5, 0.8], 0.1 + s * 0.5),
        ];
        for (f, x, w) in cases {
            let got = f.h(*x);
            assert!(rel_ok(got, *w, 1e-12), "h({x:?}) = {got}, want {w}");
        }
    }

    #[test]
    fn refinement_translation() {
        let spec = RefinementSpec {
            levels: vec![RefinementBand {
                patch: "wall".to_string(),
                bands: vec![
                    DistanceBand {
                        distance: 0.5,
                        level: 1,
                    },
                    DistanceBand {
                        distance: 0.1,
                        level: 3,
                    },
                ],
                feature_level: 0,
            }],
            feature_angle_deg: 30.0,
            max_level: 2,
            boxes: vec![RefinementBox {
                min: [0.0, 0.0, 0.0],
                max: [1.0, 1.0, 1.0],
                level: 2,
            }],
        };
        let patches = |wall_tris: Vec<[Point; 3]>| {
            vec![
                PatchTris {
                    name: "wall".to_string(),
                    tris: wall_tris,
                },
                PatchTris {
                    name: "other".to_string(),
                    tris: vec![sq()[0]],
                },
            ]
        };
        let got = sources_from_refinement(&spec, 1.0, &patches(sq())).unwrap();
        let want = vec![
            SizeSource::Patch {
                name: "wall".to_string(),
                tris: sq(),
                distance: 0.5,
                size: 0.5,
            },
            SizeSource::Patch {
                name: "wall".to_string(),
                tris: sq(),
                distance: 0.1,
                size: 0.25,
            },
            SizeSource::Box {
                min: [0.0, 0.0, 0.0],
                max: [1.0, 1.0, 1.0],
                size: 0.25,
            },
        ];
        assert_eq!(got, want);
        for l in 0..=10u32 {
            let mut w = 1.0f64;
            for _ in 0..l {
                w *= 0.5;
            }
            assert_eq!(level_size(1.0, l).to_bits(), w.to_bits(), "level {l}");
        }
        assert_eq!(level_size(0.3, 3).to_bits(), (0.3f64 * 0.125).to_bits());
        // Refusals, each by name.
        let missing = RefinementSpec {
            levels: vec![RefinementBand {
                patch: "nope".to_string(),
                bands: vec![DistanceBand {
                    distance: 0.5,
                    level: 1,
                }],
                feature_level: 0,
            }],
            feature_angle_deg: 30.0,
            max_level: 2,
            boxes: vec![],
        };
        let msg = mesh_msg(sources_from_refinement(&missing, 1.0, &patches(sq())));
        assert!(
            msg.contains("nope") && msg.contains("wall") && msg.contains("other"),
            "{msg}"
        );
        let msg = mesh_msg(sources_from_refinement(&spec, 1.0, &patches(vec![])));
        assert!(msg.contains("wall") && msg.contains("no triangle"), "{msg}");
        let mut fl = spec.clone();
        fl.levels[0].feature_level = 1;
        let msg = mesh_msg(sources_from_refinement(&fl, 1.0, &patches(sq())));
        assert!(msg.contains("feature_level") && msg.contains("wall"), "{msg}");
        let mut neg = spec.clone();
        neg.levels[0].bands[0].distance = -1.0;
        let msg = mesh_msg(sources_from_refinement(&neg, 1.0, &patches(sq())));
        assert!(msg.contains("distance") && msg.contains("wall"), "{msg}");
        let mut nan = spec.clone();
        nan.levels[0].bands[0].distance = f64::NAN;
        let msg = mesh_msg(sources_from_refinement(&nan, 1.0, &patches(sq())));
        assert!(msg.contains("distance"), "{msg}");
    }

    #[test]
    fn octree_matches_brute_force() {
        let field = mix();
        let lo = [-2.0, -2.0, -1.5];
        let hi = [2.0, 2.0, 1.5];
        let mut probes: Vec<Point> = Vec::new();
        let mut rng = SplitMix64::new(47);
        for _ in 0..100_000 {
            probes.push(rng.point(lo, hi));
        }
        for _ in 0..10_000 {
            probes.push(rng.point([-2.5, -2.5, -2.0], [2.5, 2.5, 2.0]));
        }
        for x in &probes {
            assert_eq!(
                field.h(*x).to_bits(),
                field.h_brute(*x).to_bits(),
                "h vs brute at {x:?}"
            );
            let inside = (0..3).all(|a| x[a] >= lo[a] && x[a] <= hi[a]);
            assert_eq!(field.leaf_box(*x).is_some(), inside, "leaf_box at {x:?}");
        }
        let st = field.stats();
        assert_eq!(st.primitives, 516);
        assert!(st.leaves > 8, "leaves {}", st.leaves);
        assert!(st.mean_candidates < 516.0, "mean {}", st.mean_candidates);
        let f1 = rebuild(1);
        let f4 = rebuild(4);
        let fmax = rebuild(usize::MAX);
        for x in probes.iter().take(20_000) {
            let b = field.h(*x).to_bits();
            assert_eq!(f1.h(*x).to_bits(), b, "cap 1 at {x:?}");
            assert_eq!(f4.h(*x).to_bits(), b, "cap 4 at {x:?}");
            assert_eq!(fmax.h(*x).to_bits(), b, "cap MAX at {x:?}");
        }
        assert_eq!(fmax.stats().leaves, 1);
        // Wall times of the 10^5 in-domain probes, informative only.
        let mut acc = 0.0f64;
        let t0 = Instant::now();
        for x in probes.iter().take(100_000) {
            acc += field.h(*x);
        }
        let t_query = t0.elapsed().as_millis();
        let t0 = Instant::now();
        for x in probes.iter().take(100_000) {
            acc += field.h_brute(*x);
        }
        let t_brute = t0.elapsed().as_millis();
        assert!(acc.is_finite());
        println!(
            "TET04A-OCTREE leaves={} depth={} max_cand={} mean_cand={:.3} prims=516 \
             t_query_ms={t_query} t_brute_ms={t_brute}",
            st.leaves, st.depth, st.max_candidates, st.mean_candidates
        );
    }

    #[test]
    fn distance_functions() {
        let t = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
        let cases: &[(Point, f64)] = &[
            ([0.2, 0.2, 0.5], 0.5),
            ([-1.0, -1.0, 0.0], SQRT_2),
            ([0.5, -1.0, 0.0], 1.0),
            ([1.0, 1.0, 0.0], SQRT_2 / 2.0),
            ([2.0, 0.0, 0.0], 1.0),
            ([0.0, 2.0, 0.0], 1.0),
            ([0.25, 0.25, 0.0], 0.0),
            ([0.5, 0.5, 3.0], 3.0),
        ];
        for (x, w) in cases {
            let got = point_triangle_distance(*x, &t);
            if *w == 0.0 {
                assert!(got.abs() <= 1e-15, "{x:?}: {got}");
            } else {
                assert!(rel_ok(got, *w, 1e-15), "{x:?}: {got} vs {w}");
            }
        }
        let degenerate: &[(&[Point; 3], Point, f64)] = &[
            (
                &[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
                [1.0, 1.0, 0.0],
                1.0,
            ),
            (
                &[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
                [3.0, 0.0, 0.0],
                1.0,
            ),
            (&[[1.0, 1.0, 1.0]; 3], [1.0, 1.0, 2.0], 1.0),
            (
                &[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
                [0.5, 2.0, 0.0],
                2.0,
            ),
        ];
        for (tri, x, w) in degenerate {
            let got = point_triangle_distance(*x, tri);
            assert!(rel_ok(got, *w, 1e-15), "{x:?}: {got} vs {w}");
        }
        let blo = [0.0; 3];
        let bhi = [1.0; 3];
        let box_cases: &[(Point, f64)] = &[
            ([0.5, 0.5, 0.5], 0.0),
            ([2.0, 0.5, 0.5], 1.0),
            ([2.0, 2.0, 0.5], SQRT_2),
            ([2.0, 2.0, 2.0], 3.0f64.sqrt()),
            ([1.0, 1.0, 1.0], 0.0),
            ([-1.0, 0.5, 3.0], 5.0f64.sqrt()),
        ];
        for (x, w) in box_cases {
            let got = point_box_distance(*x, blo, bhi);
            if *w == 0.0 {
                assert!(got.abs() <= 1e-15, "{x:?}: {got}");
            } else {
                assert!(rel_ok(got, *w, 1e-15), "{x:?}: {got} vs {w}");
            }
        }

        let mut rng = SplitMix64::new(48);
        for _ in 0..2000 {
            let a = rng.point([-1.0; 3], [1.0; 3]);
            let b = rng.point([-1.0; 3], [1.0; 3]);
            let c = rng.point([-1.0; 3], [1.0; 3]);
            let x = rng.point([-2.0; 3], [2.0; 3]);
            let tri = [a, b, c];
            let d = point_triangle_distance(x, &tri);
            let mut dg = f64::INFINITY;
            for i in 0..=100i32 {
                for j in 0..=(100 - i) {
                    let fi = f64::from(i) / 100.0;
                    let fj = f64::from(j) / 100.0;
                    let q = [
                        a[0] + fi * (b[0] - a[0]) + fj * (c[0] - a[0]),
                        a[1] + fi * (b[1] - a[1]) + fj * (c[1] - a[1]),
                        a[2] + fi * (b[2] - a[2]) + fj * (c[2] - a[2]),
                    ];
                    dg = dg.min(norm(sub(x, q)));
                }
            }
            let e = norm(sub(b, a))
                .max(norm(sub(c, b)))
                .max(norm(sub(a, c)));
            assert!(
                d <= dg * (1.0 + 1e-12) + 1e-15,
                "d {d} above the grid min {dg}"
            );
            assert!(dg - d <= e / 100.0, "grid min {dg} below d {d} past e/100");
        }
    }

    #[test]
    fn lipschitz_smoke() {
        let field = mix();
        let lo = [-2.0, -2.0, -1.5];
        let hi = [2.0, 2.0, 1.5];
        let s = 1.15f64 - 1.0;
        let mut rng = SplitMix64::new(49);
        let mut max_ratio = 0.0f64;
        for _ in 0..10_000 {
            let a = rng.point(lo, hi);
            let v = loop {
                let w = [rng.uni(-1.0, 1.0), rng.uni(-1.0, 1.0), rng.uni(-1.0, 1.0)];
                if norm(w) >= 1e-3 {
                    break w;
                }
            };
            let n = norm(v);
            let v = [v[0] / n, v[1] / n, v[2] / n];
            let t = rng.uni(1e-4, 0.2);
            let b = [a[0] + t * v[0], a[1] + t * v[1], a[2] + t * v[2]];
            let ha = field.h(a);
            let hb = field.h(b);
            let d = norm(sub(b, a));
            assert!(
                (ha - hb).abs() <= s * d * (1.0 + 1e-9),
                "|{ha} - {hb}| above s d at {a:?} {b:?}"
            );
            max_ratio = max_ratio.max((ha - hb).abs() / (s * d));
        }
        println!("TET04A-LIP max_ratio={max_ratio}");
    }

    #[test]
    fn floor_and_cap() {
        let pt = SizeSource::Point {
            at: [0.0; 3],
            size: 0.001,
        };
        let field = SizeField::build(
            SizeParams::new(1.0, 1.2, None).unwrap(),
            &[pt.clone()],
            [-12.0; 3],
            [12.0; 3],
            OctreeOptions::default(),
        )
        .unwrap();
        assert_eq!(field.h([0.0; 3]), 0.015625);
        assert_eq!(field.h([0.05, 0.0, 0.0]), 0.015625);
        assert!(rel_ok(field.h([1.0, 0.0, 0.0]), 0.001 + (1.2f64 - 1.0), 1e-15));
        assert_eq!(field.h([10.0, 0.0, 0.0]), 1.0);
        let field = SizeField::build(
            SizeParams::new(1.0, 1.2, Some(0.0005)).unwrap(),
            &[pt.clone()],
            [-12.0; 3],
            [12.0; 3],
            OctreeOptions::default(),
        )
        .unwrap();
        assert_eq!(field.h([0.0; 3]), 0.001);
        let field = SizeField::build(
            SizeParams::new(1.0, 1.2, None).unwrap(),
            &[],
            [-12.0; 3],
            [12.0; 3],
            OctreeOptions::default(),
        )
        .unwrap();
        let mut rng = SplitMix64::new(50);
        for _ in 0..100 {
            let x = rng.point([-12.0; 3], [12.0; 3]);
            assert_eq!(field.h(x), 1.0);
        }
        assert_eq!(field.stats().leaves, 1);
        assert_eq!(field.stats().primitives, 0);
        let field = SizeField::build(
            SizeParams::new(1.0, 1.0, None).unwrap(),
            &[pt.clone()],
            [-12.0; 3],
            [12.0; 3],
            OctreeOptions::default(),
        )
        .unwrap();
        assert_eq!(field.h([5.0, 0.0, 0.0]), 0.015625);
    }

    #[test]
    fn refusals() {
        for base in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            let msg = params_msg(SizeParams::new(base, 1.2, None));
            assert!(msg.contains("base_size"), "{msg}");
        }
        for g in [0.9, f64::NAN, f64::INFINITY] {
            let msg = params_msg(SizeParams::new(1.0, g, None));
            assert!(msg.contains("gradation"), "{msg}");
        }
        assert!(params_msg(SizeParams::new(1.0, 1.2, Some(2.0))).contains("min_size"));
        assert!(params_msg(SizeParams::new(1.0, 1.2, Some(0.0))).contains("min_size"));
        assert!(SizeParams::new(1.0, 1.2, Some(1.0)).is_ok());
        let spec_default = TetSizeSpec::default();
        let p = SizeParams::from_spec(1.0, &spec_default).unwrap();
        assert_eq!(p.gradation, 1.2);
        assert_eq!(p.min_size, 1.0 / 64.0);
        let params = SizeParams::new(1.0, 1.2, None).unwrap();
        let opts = OctreeOptions::default();
        let msg = field_msg(SizeField::build(
            params,
            &[],
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 1.0],
            opts,
        ));
        assert!(msg.contains("domain"), "{msg}");
        let msg = field_msg(SizeField::build(
            params,
            &[],
            [0.0; 3],
            [1.0, f64::NAN, 1.0],
            opts,
        ));
        assert!(msg.contains("domain"), "{msg}");
        let msg = field_msg(SizeField::build(
            params,
            &[SizeSource::Point {
                at: [0.5; 3],
                size: 0.0,
            }],
            [0.0; 3],
            [1.0; 3],
            opts,
        ));
        assert!(msg.contains("source 0") && msg.contains("size"), "{msg}");
        let msg = field_msg(SizeField::build(
            params,
            &[
                SizeSource::Point {
                    at: [0.5; 3],
                    size: 0.1,
                },
                SizeSource::Point {
                    at: [0.5; 3],
                    size: -1.0,
                },
            ],
            [0.0; 3],
            [1.0; 3],
            opts,
        ));
        assert!(msg.contains("source 1") && msg.contains("size"), "{msg}");
        let msg = field_msg(SizeField::build(
            params,
            &[SizeSource::Point {
                at: [f64::NAN, 0.0, 0.0],
                size: 0.1,
            }],
            [0.0; 3],
            [1.0; 3],
            opts,
        ));
        assert!(msg.contains("source 0") && msg.contains("finite"), "{msg}");
        let msg = field_msg(SizeField::build(
            params,
            &[SizeSource::Box {
                min: [1.0, 0.0, 0.0],
                max: [0.0, 1.0, 1.0],
                size: 0.1,
            }],
            [0.0; 3],
            [1.0; 3],
            opts,
        ));
        assert!(msg.contains("source 0") && msg.contains("box"), "{msg}");
        let msg = field_msg(SizeField::build(
            params,
            &[SizeSource::Patch {
                name: "empty".to_string(),
                tris: vec![],
                distance: 0.0,
                size: 0.1,
            }],
            [0.0; 3],
            [1.0; 3],
            opts,
        ));
        assert!(msg.contains("source 0") && msg.contains("empty"), "{msg}");
        let msg = field_msg(SizeField::build(
            params,
            &[SizeSource::Patch {
                name: "p".to_string(),
                tris: sq(),
                distance: -0.1,
                size: 0.1,
            }],
            [0.0; 3],
            [1.0; 3],
            opts,
        ));
        assert!(msg.contains("source 0") && msg.contains("distance"), "{msg}");
        let msg = field_msg(SizeField::build(
            params,
            &[SizeSource::Patch {
                name: "p".to_string(),
                tris: vec![[[0.0, 0.0, 0.0], [f64::INFINITY, 0.0, 0.0], [0.0, 1.0, 0.0]]],
                distance: 0.0,
                size: 0.1,
            }],
            [0.0; 3],
            [1.0; 3],
            opts,
        ));
        assert!(msg.contains("source 0") && msg.contains("finite"), "{msg}");
        let ok = SizeField::build(
            params,
            &[SizeSource::Box {
                min: [0.0; 3],
                max: [0.0; 3],
                size: 0.1,
            }],
            [0.0; 3],
            [1.0; 3],
            opts,
        );
        assert!(ok.is_ok());
    }

    /// The Monte Carlo estimate of `integral dV / h^3` over the root box:
    /// 10^5 uniform points, the mean of `1 / h_brute^3`, times the volume.
    fn n_h(field: &SizeField, lo: Point, hi: Point, seed: u64) -> f64 {
        let mut rng = SplitMix64::new(seed);
        let n = 100_000usize;
        let mut acc = 0.0f64;
        for _ in 0..n {
            acc += 1.0 / field.h_brute(rng.point(lo, hi)).powi(3);
        }
        let vol = (hi[0] - lo[0]) * (hi[1] - lo[1]) * (hi[2] - lo[2]);
        vol * acc / n as f64
    }

    /// The leaf-count bound of (116.7g) over the root box, with 1.5 for the
    /// Monte Carlo error and 1 for an unsplit root.
    fn leaf_bound(lo: Point, hi: Point, s: f64, n_h: f64) -> f64 {
        let e = (hi[0] - lo[0]).max(hi[1] - lo[1]).max(hi[2] - lo[2]);
        let v = (hi[0] - lo[0]) * (hi[1] - lo[1]) * (hi[2] - lo[2]);
        1.5 * (e * e * e / v) * (2.0 + 2.0 * 3.0f64.sqrt() * s).powi(3) * n_h + 1.0
    }

    #[test]
    fn octree_is_mesh_scale() {
        let s = 1.2f64 - 1.0;
        // a: 40 tied point sources must not split the whole domain down to
        // the smallest leaves: the tree stops at the mesh's scale.
        let sources: Vec<SizeSource> = (0..40)
            .map(|_| SizeSource::Point {
                at: [0.0; 3],
                size: 0.05,
            })
            .collect();
        let field = SizeField::build(
            SizeParams::new(1.0, 1.2, None).unwrap(),
            &sources,
            [-1.0; 3],
            [1.0; 3],
            OctreeOptions::default(),
        )
        .unwrap();
        let lo = [-1.0; 3];
        let hi = [1.0; 3];
        let leaves_a = field.stats().leaves;
        let n_a = n_h(&field, lo, hi, 51);
        let bound_a = leaf_bound(lo, hi, s, n_a);
        assert!(leaves_a as f64 <= bound_a, "cluster leaves {leaves_a} above bound {bound_a}");
        let mut rng = SplitMix64::new(52);
        for _ in 0..10_000 {
            let x = rng.point(lo, hi);
            assert_eq!(field.h(x).to_bits(), field.h_brute(x).to_bits(), "at {x:?}");
        }
        // b: the MIX fixture, whose first run built 3.2 M leaves.
        let field = mix();
        let lo = [-2.0, -2.0, -1.5];
        let hi = [2.0, 2.0, 1.5];
        let n_b = n_h(&field, lo, hi, 53);
        // MIX has g = 1.15, so its own s, not part (a)'s.
        let bound_b = leaf_bound(lo, hi, 1.15f64 - 1.0, n_b);
        assert!(
            field.stats().leaves as f64 <= bound_b,
            "mix leaves {} above bound {bound_b}",
            field.stats().leaves
        );
        println!(
            "TET04A-SCALE cluster leaves={leaves_a} bound={bound_a:.0} | mix leaves={} \
             N_h={n_b:.1} bound={bound_b:.0}",
            field.stats().leaves
        );
    }

    #[test]
    fn band_interior_collapses() {
        // The fine square: 20 by 20 cells, 800 triangles on z = 0.
        let mut fs: Vec<[Point; 3]> = Vec::new();
        for i in 0..20u32 {
            for j in 0..20u32 {
                let x0 = f64::from(i) / 20.0;
                let y0 = f64::from(j) / 20.0;
                let x1 = f64::from(i + 1) / 20.0;
                let y1 = f64::from(j + 1) / 20.0;
                let a = [x0, y0, 0.0];
                let b = [x1, y0, 0.0];
                let c = [x1, y1, 0.0];
                let d = [x0, y1, 0.0];
                fs.push([a, b, c]);
                fs.push([a, c, d]);
            }
        }
        let params = SizeParams::new(1.0, 1.2, None).unwrap();
        let opts = OctreeOptions {
            leaf_cap: 1,
            max_depth: 12,
            threads: 0,
        };
        let src = |tris: Vec<[Point; 3]>| SizeSource::Patch {
            name: "fine".to_string(),
            tris,
            distance: 0.5,
            size: 0.1,
        };
        let field = SizeField::build(params, &[src(fs.clone())], [-1.0; 3], [2.0; 3], opts)
            .unwrap();
        let probe = [0.5, 0.5, 0.1];
        assert_eq!(field.leaf_candidates(probe), Some(1));
        assert_eq!(field.h(probe), 0.1);
        let mut rng = SplitMix64::new(54);
        let probes: Vec<Point> = (0..10_000).map(|_| rng.point([-1.0; 3], [2.0; 3])).collect();
        for x in &probes {
            assert_eq!(
                field.h(*x).to_bits(),
                field.h_brute(*x).to_bits(),
                "h vs brute at {x:?}"
            );
        }
        // A duplicate patch source under the same name: a separate source,
        // so its triangles must survive as a second candidate — the
        // collapse of (116.7e) is per source, never per name.
        let field2 = SizeField::build(
            params,
            &[src(fs.clone()), src(fs)],
            [-1.0; 3],
            [2.0; 3],
            opts,
        )
        .unwrap();
        assert_eq!(field2.leaf_candidates(probe), Some(2));
        for x in &probes {
            assert_eq!(
                field2.h(*x).to_bits(),
                field2.h_brute(*x).to_bits(),
                "dup h vs brute at {x:?}"
            );
        }
        println!(
            "TET04A-BAND leaves={} cand_at_probe={} | dup leaves={} cand_at_probe={}",
            field.stats().leaves,
            field.leaf_candidates(probe).unwrap(),
            field2.stats().leaves,
            field2.leaf_candidates(probe).unwrap()
        );
    }

    // ==================================================================
    //  TET-04b: curvature and proximity (§116.8)
    // ==================================================================

    use crate::tetmesh::surfprep::prepare;

    /// The icosahedron's 12 unit vertices and 20 outward faces, before
    /// subdivision.
    fn ico_base() -> (Vec<Point>, Vec<[u32; 3]>) {
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
        let unit = |p: Point| {
            let n = norm(p);
            [p[0] / n, p[1] / n, p[2] / n]
        };
        let f: [[usize; 3]; 20] = [
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
        (
            v.iter().map(|&p| unit(p)).collect(),
            f.iter()
                .map(|g| [g[0] as u32, g[1] as u32, g[2] as u32])
                .collect(),
        )
    }

    /// One midpoint per unordered pair: the normalised sum, cached.
    fn ico_mid(pts: &mut Vec<Point>, cache: &mut HashMap<(u32, u32), u32>, a: u32, b: u32) -> u32 {
        let key = if a < b { (a, b) } else { (b, a) };
        if let Some(&m) = cache.get(&key) {
            return m;
        }
        let m = add(pts[a as usize], pts[b as usize]);
        let n = norm(m);
        let id = pts.len() as u32;
        pts.push([m[0] / n, m[1] / n, m[2] / n]);
        cache.insert(key, id);
        id
    }

    /// The icosphere at `level` subdivisions, on the unit sphere.
    fn ico_mesh(level: u32) -> (Vec<Point>, Vec<[u32; 3]>) {
        let (mut pts, mut tris) = ico_base();
        for _ in 0..level {
            let mut cache: HashMap<(u32, u32), u32> = HashMap::new();
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

    /// `ico(level, r, centre)` as a one-patch TetSurface named "sphere".
    fn ico_surface(level: u32, r: f64, centre: Point) -> TetSurface {
        let (pts, tris) = ico_mesh(level);
        TetSurface {
            points: pts
                .iter()
                .map(|p| [centre[0] + r * p[0], centre[1] + r * p[1], centre[2] + r * p[2]])
                .collect(),
            tri_patch: vec![0; tris.len()],
            patch_names: vec!["sphere".to_string()],
            tris,
        }
    }

    /// Two icospheres in one surface: ico(3, 0.5, origin) then
    /// ico(3, 0.2, (0, 0, 0.75)); the gap between them is 0.05.
    fn two_sphere_surface() -> TetSurface {
        let (pa, ta) = ico_mesh(3);
        let (pb, tb) = ico_mesh(3);
        let off = pa.len() as u32;
        let mut points: Vec<Point> = pa
            .iter()
            .map(|p| [0.5 * p[0], 0.5 * p[1], 0.5 * p[2]])
            .collect();
        points.extend(pb.iter().map(|p| [0.2 * p[0], 0.2 * p[1], 0.75 + 0.2 * p[2]]));
        let mut tris = ta;
        tris.extend(tb.iter().map(|t| [t[0] + off, t[1] + off, t[2] + off]));
        TetSurface {
            points,
            tri_patch: vec![0; tris.len()],
            patch_names: vec!["spheres".to_string()],
            tris,
        }
    }

    /// The N = 32, R = 0.5 cylinder: 32 side quads (64 triangles, side
    /// first), then the two caps; one patch "cyl".
    fn cylinder_surface() -> TetSurface {
        let mut points: Vec<Point> = Vec::with_capacity(66);
        for j in 0..32u32 {
            let a = 2.0 * PI * f64::from(j) / 32.0;
            points.push([0.5 * a.cos(), 0.5 * a.sin(), 0.0]);
        }
        for j in 0..32u32 {
            let a = 2.0 * PI * f64::from(j) / 32.0;
            points.push([0.5 * a.cos(), 0.5 * a.sin(), 1.0]);
        }
        points.push([0.0, 0.0, 0.0]);
        points.push([0.0, 0.0, 1.0]);
        let mut tris: Vec<[u32; 3]> = Vec::with_capacity(128);
        for j in 0..32u32 {
            let jj = (j + 1) % 32;
            tris.push([j, jj, 32 + jj]);
            tris.push([j, 32 + jj, 32 + j]);
        }
        for j in 0..32u32 {
            let jj = (j + 1) % 32;
            tris.push([64, jj, j]);
            tris.push([65, 32 + j, 32 + jj]);
        }
        TetSurface {
            points,
            tri_patch: vec![0; tris.len()],
            patch_names: vec!["cyl".to_string()],
            tris,
        }
    }

    /// The unit cube's corners, `b = x + 2y + 4z`.
    fn cube_points() -> Vec<Point> {
        (0..8u32)
            .map(|b| {
                [
                    f64::from(b & 1),
                    f64::from((b & 2) >> 1),
                    f64::from((b & 4) >> 2),
                ]
            })
            .collect()
    }

    /// The unit cube's 12 outward triangles.
    const CUBE_TRIS: [[u32; 3]; 12] = [
        [0, 2, 3],
        [0, 3, 1],
        [4, 5, 7],
        [4, 7, 6],
        [0, 1, 5],
        [0, 5, 4],
        [2, 6, 7],
        [2, 7, 3],
        [0, 4, 6],
        [0, 6, 2],
        [1, 3, 7],
        [1, 7, 5],
    ];

    fn cube_surface() -> TetSurface {
        TetSurface {
            points: cube_points(),
            tris: CUBE_TRIS.to_vec(),
            tri_patch: vec![0; 12],
            patch_names: vec!["cube".to_string()],
        }
    }

    /// The 1 mm gap: box A on z in [-0.005, 0], box B on z in
    /// [0.001, 0.006], the same 0.01 x 0.01 footprint, one surface,
    /// patches "a" and "b". A's top z = 0 faces B's bottom z = 0.001.
    fn box_pair_surface() -> TetSurface {
        let map = |b: u32, z0: f64| {
            [
                0.01 * f64::from(b & 1),
                0.01 * f64::from((b & 2) >> 1),
                z0 + 0.005 * f64::from((b & 4) >> 2),
            ]
        };
        let mut points: Vec<Point> = (0..8u32).map(|b| map(b, -0.005)).collect();
        points.extend((0..8u32).map(|b| map(b, 0.001)));
        let mut tris: Vec<[u32; 3]> = CUBE_TRIS.to_vec();
        tris.extend(CUBE_TRIS.map(|t| [t[0] + 8, t[1] + 8, t[2] + 8]));
        let mut tri_patch: Vec<u32> = vec![0; 12];
        tri_patch.extend(vec![1; 12]);
        TetSurface {
            points,
            tris,
            tri_patch,
            patch_names: vec!["a".to_string(), "b".to_string()],
        }
    }

    /// Every fixture passes `prepare(.., 60.0)` before use.
    fn prepared(surf: &TetSurface) {
        assert!(
            prepare(surf, 60.0).is_ok(),
            "prepare refused {:?}",
            surf.patch_names
        );
    }

    /// A facet point: triangle `next() % n`, barycentric u, v folded back
    /// into the triangle.
    fn facet_point(rng: &mut SplitMix64, pts: &[Point], tris: &[[u32; 3]]) -> Point {
        let n = tris.len() as u64;
        let t = &tris[(rng.next() % n) as usize];
        let x0 = pts[t[0] as usize];
        let x1 = pts[t[1] as usize];
        let x2 = pts[t[2] as usize];
        let mut u = rng.unit();
        let mut v = rng.unit();
        if u + v > 1.0 {
            u = 1.0 - u;
            v = 1.0 - v;
        }
        [
            x0[0] + u * (x1[0] - x0[0]) + v * (x2[0] - x0[0]),
            x0[1] + u * (x1[1] - x0[1]) + v * (x2[1] - x0[1]),
            x0[2] + u * (x1[2] - x0[2]) + v * (x2[2] - x0[2]),
        ]
    }

    /// A true sphere point by rejection: w in [-1, 1]^3 with
    /// 1e-3 <= |w| <= 1, scaled to the radius about c.
    fn sphere_point(rng: &mut SplitMix64, c: Point, r: f64) -> Point {
        loop {
            let w = [rng.uni(-1.0, 1.0), rng.uni(-1.0, 1.0), rng.uni(-1.0, 1.0)];
            let n = norm(w);
            if (1e-3..=1.0).contains(&n) {
                return [
                    c[0] + r * w[0] / n,
                    c[1] + r * w[1] / n,
                    c[2] + r * w[2] / n,
                ];
            }
        }
    }

    fn curve_msg(r: Result<Vec<f64>>) -> String {
        match r {
            Err(Error::Mesh(m)) => m,
            _ => panic!("expected Error::Mesh"),
        }
    }

    fn gaps_msg(r: Result<Vec<Option<f64>>>) -> String {
        match r {
            Err(Error::Mesh(m)) => m,
            _ => panic!("expected Error::Mesh"),
        }
    }

    #[test]
    fn curvature_sphere() {
        let surf = ico_surface(3, 0.5, [0.0; 3]);
        prepared(&surf);
        assert_eq!(surf.points.len(), 642);
        assert_eq!(surf.tris.len(), 1280);
        let kap = triangle_curvatures(&surf, 60.0).unwrap();
        assert_eq!(kap.len(), 1280);
        assert!(kap.iter().all(|k| *k > 0.0), "all kappa above 0");
        let mut kmin = f64::INFINITY;
        let mut kmax = 0.0f64;
        for k in &kap {
            kmin = kmin.min(k * 0.5);
            kmax = kmax.max(k * 0.5);
        }
        assert!(
            (0.9965..=0.9975).contains(&kmin),
            "min kappa R {kmin} outside [0.9965, 0.9975]"
        );
        assert!(
            (1.1455..=1.1470).contains(&kmax),
            "max kappa R {kmax} outside [1.1455, 1.1470]"
        );
        let p = SizeParams::new(1.0, 1.2, Some(1e-3)).unwrap();
        let sources = curvature_sources(&surf, &p, 16, 60.0).unwrap();
        assert_eq!(sources.len(), 1280);
        let field = SizeField::build(
            p,
            &sources,
            [-1.0; 3],
            [1.0; 3],
            OctreeOptions::default(),
        )
        .unwrap();
        let b = 2.0 * PI * 0.5 / 16.0;
        let mut rng = SplitMix64::new(41);
        let mut facet_max = 0.0f64;
        for _ in 0..10_000 {
            let x = facet_point(&mut rng, &surf.points, &surf.tris);
            let h = field.h(x);
            assert!(h / b <= 1.02, "facet {x:?}: h {h} is {} b", h / b);
            facet_max = facet_max.max(h / b);
            assert_eq!(field.h(x).to_bits(), field.h_brute(x).to_bits(), "at {x:?}");
        }
        assert!(facet_max >= 0.9, "facet_max {facet_max} below 0.9");
        let mut rng = SplitMix64::new(43);
        let mut sphere_max = 0.0f64;
        for _ in 0..10_000 {
            let x = sphere_point(&mut rng, [0.0; 3], 0.5);
            let h = field.h(x);
            assert!(h / b <= 1.02, "sphere {x:?}: h {h} is {} b", h / b);
            sphere_max = sphere_max.max(h / b);
            assert_eq!(field.h(x).to_bits(), field.h_brute(x).to_bits(), "at {x:?}");
        }
        println!("TET04B-CURV facet_max={facet_max} sphere_max={sphere_max}");
    }

    #[test]
    fn curvature_creases() {
        let surf = cylinder_surface();
        prepared(&surf);
        let kap = triangle_curvatures(&surf, 60.0).unwrap();
        assert_eq!(kap.len(), 128);
        let want = 1.5 * (2.0 * PI / 32.0) / (2.0 * 0.5 * (PI / 32.0).sin());
        for (i, k) in kap.iter().enumerate() {
            if i < 64 {
                assert!(rel_ok(*k, want, 1e-12), "side {i}: {k} vs {want}");
            } else {
                assert_eq!(k.to_bits(), 0.0f64.to_bits(), "cap {i}: {k}");
            }
        }
        let cube = cube_surface();
        prepared(&cube);
        let kapc = triangle_curvatures(&cube, 60.0).unwrap();
        assert_eq!(kapc.len(), 12);
        for (i, k) in kapc.iter().enumerate() {
            assert_eq!(k.to_bits(), 0.0f64.to_bits(), "cube {i}: {k}");
        }
        let p = SizeParams::new(1.0, 1.2, None).unwrap();
        assert!(curvature_sources(&cube, &p, 16, 60.0).unwrap().is_empty());
    }

    #[test]
    fn proximity_gap() {
        let surf = box_pair_surface();
        prepared(&surf);
        let facing = [2usize, 3, 12, 13];
        let out = triangle_gaps(&surf, GapSide::Outward).unwrap();
        assert_eq!(out.len(), 24);
        for (i, g) in out.iter().enumerate() {
            if facing.contains(&i) {
                let v = g.unwrap_or_else(|| panic!("outward {i}: expected Some(1e-3)"));
                assert!(rel_ok(v, 1e-3, 1e-12), "outward {i}: {v}");
            } else {
                assert_eq!(*g, None, "outward {i}");
            }
        }
        let inw = triangle_gaps(&surf, GapSide::Inward).unwrap();
        for (i, g) in inw.iter().enumerate() {
            let want = if i % 12 < 4 { 0.005 } else { 0.01 };
            let v = g.unwrap_or_else(|| panic!("inward {i}: expected Some({want})"));
            assert!(rel_ok(v, want, 1e-12), "inward {i}: {v} vs {want}");
        }
        let both = triangle_gaps(&surf, GapSide::Both).unwrap();
        for (i, g) in both.iter().enumerate() {
            if facing.contains(&i) {
                let v = g.unwrap_or_else(|| panic!("both {i}: expected Some(1e-3)"));
                assert!(rel_ok(v, 1e-3, 1e-12), "both {i}: {v}");
            } else {
                assert_eq!(g.map(f64::to_bits), inw[i].map(f64::to_bits), "both {i}");
            }
        }

        // The field: the gap primitives, h <= gap/3 through the gap.
        let p = SizeParams::new(0.02, 1.2, Some(1e-5)).unwrap();
        let curv = curvature_sources(&surf, &p, 16, 60.0).unwrap();
        assert!(curv.is_empty());
        let mut sources = proximity_sources(&surf, &p, 3, GapSide::Both).unwrap();
        sources.extend(curv);
        let lo = [-0.01, -0.01, -0.01];
        let hi = [0.02, 0.02, 0.016];
        let field =
            SizeField::build(p, &sources, lo, hi, OctreeOptions::default()).unwrap();
        let mut rng = SplitMix64::new(47);
        let want = 1e-3 / 3.0;
        for _ in 0..10_000 {
            let x = rng.point([0.0; 3], [0.01, 0.01, 0.001]);
            let h = field.h(x);
            assert!(rel_ok(h, want, 1e-12), "gap h({x:?}) = {h} vs {want}");
            assert_eq!(field.h(x).to_bits(), field.h_brute(x).to_bits(), "at {x:?}");
        }
        let x = [0.005, 0.005, 0.0105];
        let want = 0.005 / 3.0 + 0.2 * 0.002;
        assert!(rel_ok(field.h(x), want, 1e-12), "above: {} vs {want}", field.h(x));
        // Inward alone does not see the fluid gap; nothing at all is the base.
        let inw_sources = proximity_sources(&surf, &p, 3, GapSide::Inward).unwrap();
        let field_in =
            SizeField::build(p, &inw_sources, lo, hi, OctreeOptions::default()).unwrap();
        let x = [0.005, 0.005, 0.0005];
        assert!(
            rel_ok(field_in.h(x), 0.005 / 3.0, 1e-12),
            "inward h({x:?}) = {}",
            field_in.h(x)
        );
        let field_no = SizeField::build(p, &[], lo, hi, OctreeOptions::default()).unwrap();
        assert_eq!(field_no.h(x), 0.02);
    }

    #[test]
    fn gradation_bounded() {
        let surf = two_sphere_surface();
        prepared(&surf);
        let gaps = triangle_gaps(&surf, GapSide::Outward).unwrap();
        let n_some = gaps.iter().filter(|g| g.is_some()).count();
        assert!((70..=90).contains(&n_some), "{n_some} gaps, want [70, 90]");
        let gmin = gaps.iter().filter_map(|g| *g).fold(f64::INFINITY, f64::min);
        assert!(
            (0.0530..=0.0540).contains(&gmin),
            "min gap {gmin} outside [0.0530, 0.0540]"
        );
        for (g, seed) in [(1.2f64, 53u64), (1.5, 59u64)] {
            let p = SizeParams::new(1.0, g, Some(1e-3)).unwrap();
            let spec = TetSizeSpec {
                gradation: g,
                curvature_cells: 16,
                gap_cells: 3,
                min_size: Some(1e-3),
            };
            let mut sources = surface_sources(&surf, &p, &spec, 60.0, GapSide::Both).unwrap();
            sources.push(SizeSource::Point {
                at: [0.8, 0.8, -0.8],
                size: 0.02,
            });
            sources.push(SizeSource::Box {
                min: [-0.9; 3],
                max: [-0.6; 3],
                size: 0.05,
            });
            sources.push(SizeSource::Patch {
                name: "floor".to_string(),
                tris: vec![
                    [[-1.0, -1.0, -0.95], [1.0, -1.0, -0.95], [1.0, 1.0, -0.95]],
                    [[-1.0, -1.0, -0.95], [1.0, 1.0, -0.95], [-1.0, 1.0, -0.95]],
                ],
                distance: 0.05,
                size: 0.04,
            });

            let field = SizeField::build(
                p,
                &sources,
                [-1.0; 3],
                [1.0; 3],
                OctreeOptions::default(),
            )
            .unwrap();
            let s = g - 1.0;
            let mut rng = SplitMix64::new(seed);
            let mut max_ratio = 0.0f64;
            for _ in 0..100_000 {
                let a = rng.point([-1.0; 3], [1.0; 3]);
                let v = loop {
                    let w = [rng.uni(-1.0, 1.0), rng.uni(-1.0, 1.0), rng.uni(-1.0, 1.0)];
                    if norm(w) >= 1e-3 {
                        break w;
                    }
                };
                let n = norm(v);
                let u = [v[0] / n, v[1] / n, v[2] / n];
                let t = 10f64.powf(rng.uni(-4.0, 0.5f64.log10()));
                let b = [a[0] + t * u[0], a[1] + t * u[1], a[2] + t * u[2]];
                let (ha, hb) = (field.h(a), field.h(b));
                let d = norm(sub(b, a));
                assert!(
                    (ha - hb).abs() <= s * d * (1.0 + 1e-9),
                    "g {g}: |{ha} - {hb}| above {s} d at {a:?} {b:?}"
                );
                assert_eq!(field.h(a).to_bits(), field.h_brute(a).to_bits(), "at {a:?}");
                max_ratio = max_ratio.max((ha - hb).abs() / (s * d));
            }
            assert!(max_ratio >= 0.99, "g {g}: max_ratio {max_ratio}");
            println!(
                "TET04B-GRAD g={g} max_ratio={max_ratio} leaves={}",
                field.stats().leaves
            );
        }
    }

    /// One BVH-vs-brute comparison of a first-hit query.
    fn check_hit(
        sc: &GapScanner,
        p: Point,
        q: Point,
        v: Point,
        skip: usize,
        queries: &mut usize,
        hits: &mut usize,
    ) {
        let a = sc.first_hit(p, q, v, skip);
        let b = sc.first_hit_brute(p, q, v, skip);
        *queries += 1;
        match (a, b) {
            (None, None) => {}
            (Some((ia, ta)), Some((ib, tb))) => {
                assert_eq!(ia, ib, "index at {p:?} -> {q:?}");
                assert_eq!(ta.to_bits(), tb.to_bits(), "tau at {p:?} -> {q:?}");
                *hits += 1;
            }
            _ => panic!("bvh {a:?} vs brute {b:?} at {p:?} -> {q:?}"),
        }
    }

    #[test]
    fn proximity_bvh_matches_brute() {
        let mut queries = 0usize;
        let mut hits = 0usize;
        for surf in [two_sphere_surface(), box_pair_surface()] {
            prepared(&surf);
            let sc = GapScanner::new(&surf);
            for t in 0..surf.tris.len() {
                for sgn in [1.0f64, -1.0] {
                    let v = scale(sc.normal[t], sgn);
                    let p = sc.centroid[t];
                    let q = add(p, scale(v, sc.len));
                    check_hit(&sc, p, q, v, t, &mut queries, &mut hits);
                }
            }
        }
        let surf = two_sphere_surface();
        let sc = GapScanner::new(&surf);
        let mut rng = SplitMix64::new(61);
        for _ in 0..1000 {
            let p = rng.point([-1.0; 3], [1.0; 3]);
            let q = rng.point([-1.0; 3], [1.0; 3]);
            let d = sub(q, p);
            let n = norm(d);
            if n == 0.0 {
                continue;
            }
            let v = scale(d, 1.0 / n);
            check_hit(&sc, p, q, v, usize::MAX, &mut queries, &mut hits);
        }
        println!("TET04B-BVH queries={queries} hits={hits}");
    }

    #[test]
    fn surface_source_refusals() {
        let cube = cube_surface();
        prepared(&cube);
        for cd in [0.0, -1.0, f64::NAN, 181.0] {
            let msg = curve_msg(triangle_curvatures(&cube, cd));
            assert!(msg.starts_with("size field:"), "{msg}");
        }
        assert!(triangle_curvatures(&cube, 180.0).is_ok());
        let mut bad = cube_surface();
        bad.tris[0] = [999, 2, 3];
        let msg = curve_msg(triangle_curvatures(&bad, 60.0));
        assert!(msg.contains("triangle 0") && msg.contains("999"), "{msg}");
        let msg = gaps_msg(triangle_gaps(&bad, GapSide::Both));
        assert!(msg.contains("triangle 0") && msg.contains("999"), "{msg}");
        // The patch checks, both directions of bad.
        let mut bp = cube_surface();
        bp.tri_patch[0] = 7;
        assert!(triangle_curvatures(&bp, 60.0).is_err());
        bp.tri_patch = vec![0; 11];
        assert!(triangle_gaps(&bp, GapSide::Both).is_err());
        // Switched-off sources return empty vectors.
        let p = SizeParams::new(1.0, 1.2, None).unwrap();
        assert!(curvature_sources(&cube, &p, 0, 60.0).unwrap().is_empty());
        assert!(proximity_sources(&cube, &p, 0, GapSide::Both)
            .unwrap()
            .is_empty());
        // A zero-area triangle: kappa 0, no gap, no panic.
        let mut z = cube_surface();
        let n0 = z.points.len() as u32;
        z.points.push([2.0, 0.0, 0.0]);
        z.points.push([3.0, 0.0, 0.0]);
        z.points.push([4.0, 0.0, 0.0]);
        z.tris.push([n0, n0 + 1, n0 + 2]);
        z.tri_patch.push(0);
        let kap = triangle_curvatures(&z, 60.0).unwrap();
        assert_eq!(kap[12].to_bits(), 0.0f64.to_bits());
        let gaps = triangle_gaps(&z, GapSide::Both).unwrap();
        assert_eq!(gaps[12], None);
    }
}
