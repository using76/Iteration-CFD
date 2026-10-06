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
//! serves, within the leaf-count bound (116.7g).
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
/// never its answer, per §116.8.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OctreeOptions {
    pub leaf_cap: usize,
    pub max_depth: u32,
}

impl Default for OctreeOptions {
    fn default() -> Self {
        OctreeOptions {
            leaf_cap: 16,
            max_depth: 12,
        }
    }
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
    /// §116.8 all hold. Single-threaded and deterministic.
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
        let root = build_node(lo, hi, 0, all, &prims, params, opts, eta, eta_d, &mut stats);
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

/// Build one node. Split into eight children while the candidate count is
/// above `leaf_cap`, the depth is below `max_depth`, and the longest edge
/// exceeds the node's own lower bound of `h`, (116.7f); a child keeps its
/// parent's candidates minus those (116.7d) drops against the child's own
/// bounds of (116.7c), then minus the tied band-interior triangles that
/// (116.7e) drops.
#[allow(clippy::too_many_arguments)]
fn build_node(
    lo: Point,
    hi: Point,
    depth: u32,
    cand: Vec<u32>,
    prims: &[Prim],
    p: SizeParams,
    opts: OctreeOptions,
    eta: f64,
    eta_d: f64,
    st: &mut SizeFieldStats,
) -> Node {
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
    for &j in &cand {
        let k = &prims[j as usize];
        let d = k.support.distance(c);
        m = m.min(k.h + s * (d - r - k.d_flat).max(0.0));
    }
    let lambda = p.min_size.max(p.base_size.min(m));
    let split = cand.len() > opts.leaf_cap && depth < opts.max_depth && dx.max(dy).max(dz) > lambda;
    if !split {
        st.leaves += 1;
        st.depth = st.depth.max(depth);
        st.max_candidates = st.max_candidates.max(cand.len());
        st.mean_candidates += cand.len() as f64;
        return Node {
            lo,
            hi,
            cand,
            children: None,
        };
    }
    let mid = c;
    let mut children: Vec<Node> = Vec::with_capacity(8);
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
        for &j in &cand {
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
        children.push(build_node(
            clo, chi, depth + 1, kept, prims, p, opts, eta, eta_d, st,
        ));
    }
    Node {
        lo,
        hi,
        cand: Vec::new(),
        children: Some(Box::new(
            children.try_into().expect("eight children built"),
        )),
    }
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
}
