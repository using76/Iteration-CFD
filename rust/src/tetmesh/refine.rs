// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.
// Provenance: see PROVENANCE.md. No GPL-licensed source was consulted.

//! The Delaunay refinement of SPEC-LIT §116.13 (TET-09).
//!
//! The input is a face recovery of §116.11 and its surface, with the
//! domain box as eight corner points (§116.12). Each pass works on the
//! triangulation, which is updated by insertion, never rebuilt:
//!
//! 1. Conformity. The split surface is re-triangulated facet by facet
//!    (116.15f). A subsegment or subtriangle that is not in the
//!    triangulation is split by the recovery's rules (116.15e), (116.15h).
//! 2. Encroachment first (Shewchuk 1998). A present subsegment with a link
//!    vertex in its diametral ball, then a present subtriangle with an apex
//!    in its diametral ball (116.15), is split.
//! 3. Quality. The carve of §116.12 names the kept tetrahedra. Each kept
//!    tetrahedron with `rho > B` (116.17) or `R > c h(x_T)` (116.18) is
//!    taken worst first. Its circumcentre (116.19) is inserted, unless it
//!    lies outside the domain box (the box face it projects to is split),
//!    or it encroaches a subsegment, a subtriangle or a box face of its
//!    cavity, or its cavity holds both tetrahedra of a subtriangle (those
//!    are split instead), or its cavity touches the cavity of an earlier
//!    point of the pass (deferred). Tetrahedra next to a protecting ball
//!    (§116.11) are exempt, counted and reported.
//!
//! A subsegment with one input vertex as an end is split on a concentric
//! shell about that end (Ruppert 1995): at the distance `2^k` with
//! `k = ceil(log2(len / 3))`, (116.20a). The loop ends when a pass finds
//! nothing to do; `refine.max_inserted_points` is an exact cap, a refusal
//! by name. Every coordinate is `[f64; 3]` in both builds (§116.2).
//!
//! Provenance: ORIGINAL. Written from Shewchuk (1998), Proc. 14th SoCG
//! 86-95 (the radius-edge refinement and its encroachment rules) and
//! Ruppert (1995), J. Algorithms 18(3):548-585 (the circumcentre split and
//! the concentric shells); the choices are derived in SPEC-LIT §116.13. No
//! GPL-licensed source was consulted.

use super::carve::{carve, CarveStart, Carved};
use super::delaunay::{Delaunay, INF};
use super::recover::{
    circumcentre2, circumsphere3, delaunay_edges, delaunay_faces, key, lerp, orient2,
    protect_radius, triangulate_facet, FaceRecovery, FacetPoints, Frame, SteinerSite,
};
use super::surfprep::TetSurface;
use super::{Point, TetRefineSpec};
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

fn sorted3(mut f: [u32; 3]) -> [u32; 3] {
    f.sort_unstable();
    f
}

/// The circumcentre and circumradius of a tetrahedron, (116.19): with
/// `a, b, c` the edges from `p[0]`, the centre is `p[0]` plus
/// `(|a|^2 (b x c) + |b|^2 (c x a) + |c|^2 (a x b)) / (2 a . (b x c))`.
pub fn circumsphere(p: [Point; 4]) -> (Point, f64) {
    let a = sub(p[1], p[0]);
    let b = sub(p[2], p[0]);
    let c = sub(p[3], p[0]);
    let bc = cross(b, c);
    let ca = cross(c, a);
    let ab = cross(a, b);
    let den = 2.0 * dot(a, bc);
    let (la, lb, lc) = (dot(a, a), dot(b, b), dot(c, c));
    let off = [
        (la * bc[0] + lb * ca[0] + lc * ab[0]) / den,
        (la * bc[1] + lb * ca[1] + lc * ab[1]) / den,
        (la * bc[2] + lb * ca[2] + lc * ab[2]) / den,
    ];
    (
        [p[0][0] + off[0], p[0][1] + off[1], p[0][2] + off[2]],
        norm(off),
    )
}

/// One tetrahedron's measures of (116.17)-(116.19).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TetQuality {
    /// The circumcentre `x_T` (116.19).
    pub centre: Point,
    /// The circumradius `R` (116.19).
    pub radius: f64,
    /// The shortest of the six edges, `l_min`.
    pub l_min: f64,
    /// `rho = R / l_min` (116.17).
    pub rho: f64,
    /// `h(x_T)`, the size field at the circumcentre (116.18).
    pub h: f64,
}

impl TetQuality {
    /// Whether the tetrahedron violates (116.17), `rho > B`, or (116.18),
    /// `R > c h(x_T)`.
    pub fn violates(&self, spec: &TetRefineSpec) -> bool {
        self.rho > spec.radius_edge || self.radius > spec.size_factor * self.h
    }

    /// The pass order's key: `max(rho / B, R / (c h))`, worst first.
    fn score(&self, spec: &TetRefineSpec) -> f64 {
        (self.rho / spec.radius_edge).max(self.radius / (spec.size_factor * self.h))
    }
}

/// The measures of the tetrahedron `p` under the size field `h`.
pub fn quality(p: [Point; 4], h: &dyn Fn(Point) -> f64) -> TetQuality {
    let (centre, radius) = circumsphere(p);
    let mut l_min = f64::INFINITY;
    for i in 0..4 {
        for j in i + 1..4 {
            l_min = l_min.min(norm(sub(p[i], p[j])));
        }
    }
    TetQuality {
        centre,
        radius,
        l_min,
        rho: radius / l_min,
        h: h(centre),
    }
}

fn tet_points(points: &[Point], t: [u32; 4]) -> [Point; 4] {
    t.map(|v| points[v as usize])
}

/// The size gate's measure, (116.20b): the mean over the unique edges of
/// `tets` of `|e| / h(m_e)`, `m_e` the edge's midpoint, summed in
/// ascending `[min, max]` order.
pub fn mean_edge_ratio(points: &[Point], tets: &[[u32; 4]], h: &dyn Fn(Point) -> f64) -> f64 {
    let mut edges: BTreeSet<(u32, u32)> = BTreeSet::new();
    for t in tets {
        for i in 0..4 {
            for j in i + 1..4 {
                edges.insert(key(t[i], t[j]));
            }
        }
    }
    let mut s = 0.0;
    for &(a, b) in &edges {
        let (p, q) = (points[a as usize], points[b as usize]);
        s += norm(sub(p, q)) / h(lerp(p, q, 0.5));
    }
    s / edges.len() as f64
}

/// (116.20) by the midpoint rule on the tetrahedra themselves, (116.20c):
/// `1.1 sum_T V_T / (h(g_T)^3 / (6 sqrt 2))`, `g_T` the centroid, summed in
/// the order given.
pub fn predicted_count(points: &[Point], tets: &[[u32; 4]], h: &dyn Fn(Point) -> f64) -> f64 {
    let mut s = 0.0;
    for t in tets {
        let p = tet_points(points, *t);
        let v = dot(sub(p[0], p[3]), cross(sub(p[1], p[3]), sub(p[2], p[3]))) / 6.0;
        let g = [0, 1, 2].map(|k| 0.25 * (p[0][k] + p[1][k] + p[2][k] + p[3][k]));
        let hg = h(g);
        s += v / (hg * hg * hg / (6.0 * 2f64.sqrt()));
    }
    1.1 * s
}

/// What a refinement inserted, by kind (SPEC-LIT §116.13).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RefineStats {
    /// Passes run, the last (empty) one included.
    pub passes: usize,
    /// Circumcentres inserted in the domain.
    pub interior: usize,
    /// Subsegment splits: missing (116.15e), encroached by a vertex, and
    /// encroached by a rejected circumcentre or facet point.
    pub segment_splits: usize,
    /// Facet points (116.15h), centroids included.
    pub facet_splits: usize,
    /// Of `facet_splits`, the centroids.
    pub facet_centroids: usize,
    /// Points on a box face, centroids included.
    pub box_face_splits: usize,
    /// Midpoints on a box edge.
    pub box_edge_splits: usize,
    /// Circumcentres not inserted because they lay outside the box or
    /// encroached, each causing splits instead.
    pub rejected: usize,
    /// Circumcentres left to a later pass because their cavity touched an
    /// earlier one's.
    pub deferred: usize,
}

impl RefineStats {
    /// Every point the refinement inserted: the quantity the cap bounds.
    pub fn inserted(&self) -> usize {
        self.interior
            + self.segment_splits
            + self.facet_splits
            + self.box_face_splits
            + self.box_edge_splits
    }
}

/// The refined mesh (SPEC-LIT §116.13).
#[derive(Debug)]
pub struct Refined {
    /// The recovery carried on: its points, chains, sites (a refinement
    /// point off the surface is [`SteinerSite::Volume`]), split surface and
    /// triangulation are the refined ones.
    pub recovery: FaceRecovery,
    /// The carve of the final triangulation.
    pub carved: Carved,
    /// What was inserted.
    pub stats: RefineStats,
    /// The kept tetrahedra (indices into `carved.tets`) that violate
    /// (116.17) or (116.18) and are exempt: a vertex or the circumcentre
    /// lies in a protecting ball (§116.11). Ascending.
    pub exempt: Vec<usize>,
    /// The protecting balls `(vertex, r_v)` of (116.15c).
    pub balls: Vec<(u32, f64)>,
}

/// The splits a pass schedules.
#[derive(Default)]
struct Plan {
    interior: Vec<Point>,
    seg: BTreeSet<(usize, usize)>,
    /// `(triangle, u, v, centroid)`.
    facet: Vec<(usize, f64, f64, bool)>,
    /// Per triangle: the facet points taken, `(xy, r)`, for the `1e-6 r`
    /// duplicate rule of (116.15h).
    taken: HashMap<usize, Vec<([f64; 2], f64)>>,
    box_faces: BTreeSet<[u32; 3]>,
    box_pts: Vec<Point>,
    box_keys: BTreeSet<[u64; 3]>,
    box_edges: BTreeSet<(u32, u32)>,
}

impl Plan {
    fn len(&self) -> usize {
        self.interior.len() + self.seg.len() + self.facet.len() + self.box_pts.len() + self.box_edges.len()
    }
}

/// The refinement's fixed state: the input triangles' frames, the edge
/// index, the box.
struct Ctx<'a> {
    tris: &'a [[u32; 3]],
    edges: &'a [[u32; 2]],
    frames: Vec<Frame>,
    eidx: HashMap<(u32, u32), usize>,
    edge_len: Vec<f64>,
    protected: HashSet<u32>,
    lo: Point,
    hi: Point,
}

fn bits(p: Point) -> [u64; 3] {
    p.map(|x| if x == 0.0 { 0.0f64.to_bits() } else { x.to_bits() })
}

/// Each input triangle's points in its frame and its triangulation, as
/// (116.15f) builds them.
fn build_facets(
    cx: &Ctx,
    chains: &[Vec<u32>],
    chain_t: &[Vec<f64>],
    interior: &[Vec<(u32, f64, f64)>],
) -> Result<Vec<(FacetPoints, Vec<[usize; 3]>)>> {
    let mut out = Vec::with_capacity(cx.tris.len());
    for (ti, t) in cx.tris.iter().enumerate() {
        let f = &cx.frames[ti];
        let mut fp = FacetPoints {
            ids: Vec::new(),
            xy: Vec::new(),
            nb: 0,
            start: [0; 3],
            side: [(0, false); 3],
        };
        for k in 0..3 {
            let (p, q) = (t[k], t[(k + 1) % 3]);
            let e = cx.eidx[&key(p, q)];
            let rev = cx.edges[e][0] != p;
            let (ka, kb) = if rev { ((k + 1) % 3, k) } else { (k, (k + 1) % 3) };
            let (p2, q2) = (f.c2[ka], f.c2[kb]);
            fp.start[k] = fp.ids.len();
            fp.side[k] = (e, rev);
            let (ch, ct) = (&chains[e], &chain_t[e]);
            let m = ch.len();
            for j in 0..m - 1 {
                let (id, tt) = if rev { (ch[m - 1 - j], ct[m - 1 - j]) } else { (ch[j], ct[j]) };
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
                "refine/facet: triangle {ti} [{}, {}, {}] cannot be triangulated in its plane",
                t[0], t[1], t[2]
            )));
        };
        out.push((fp, local));
    }
    Ok(out)
}

/// The split parameter of subsegment `i` of edge `e`, (116.20a): on the
/// concentric shell `2^k` from the end that is an unprotected input vertex
/// when exactly one end is, else the midpoint (116.15e).
fn split_t(cx: &Ctx, chain_t: &[Vec<f64>], e: usize, i: usize) -> Result<f64> {
    let ct = &chain_t[e];
    let (ta, tb) = (ct[i], ct[i + 1]);
    let a_in = i == 0 && !cx.protected.contains(&cx.edges[e][0]);
    let b_in = i + 2 == ct.len() && !cx.protected.contains(&cx.edges[e][1]);
    let l = cx.edge_len[e];
    let t = if a_in != b_in {
        let len = (tb - ta) * l;
        let s = 2f64.powi((len / 3.0).log2().ceil() as i32);
        if a_in {
            ta + s / l
        } else {
            tb - s / l
        }
    } else {
        0.5 * (ta + tb)
    };
    if !(t > ta && t < tb) {
        return Err(Error::Refused(format!(
            "refine/split: subsegment {i} of edge [{}, {}] is too short to split",
            cx.edges[e][0], cx.edges[e][1]
        )));
    }
    Ok(t)
}

/// The facet split of subtriangle `li` of triangle `ti`, (116.15h): its
/// circumcentre in the plane, or the subsegments of the triangle's sides
/// that point encroaches, or (outside and encroaching none) the centroid.
fn plan_facet(
    cx: &Ctx,
    points: &[Point],
    chains: &[Vec<u32>],
    facets: &[(FacetPoints, Vec<[usize; 3]>)],
    ti: usize,
    li: usize,
    plan: &mut Plan,
) {
    let f = &cx.frames[ti];
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
    let mut enc = false;
    for &(e, _) in &fp.side {
        let ch = &chains[e];
        for i in 0..ch.len() - 1 {
            let (a, b) = (points[ch[i] as usize], points[ch[i + 1] as usize]);
            if dot(sub(x, a), sub(x, b)) < 0.0 {
                plan.seg.insert((e, i));
                enc = true;
            }
        }
    }
    if enc {
        return;
    }
    let (pick, centroid) = if inside {
        (c2, false)
    } else {
        ([(p[0] + q[0] + r[0]) / 3.0, (p[1] + q[1] + r[1]) / 3.0], true)
    };
    let taken = plan.taken.entry(ti).or_default();
    let near = taken.iter().any(|&(y, ry)| {
        let dd = ((y[0] - pick[0]).powi(2) + (y[1] - pick[1]).powi(2)).sqrt();
        dd < 1e-6 * rad.min(ry)
    });
    if near {
        return;
    }
    taken.push((pick, rad));
    let (u, v) = f.uv(pick);
    plan.facet.push((ti, u, v, centroid));
}

/// The box plane a hull face lies on: `(axis, value)`.
fn face_plane(cx: &Ctx, pts: &[Point], f: [u32; 3]) -> Option<(usize, f64)> {
    for k in 0..3 {
        for v in [cx.lo[k], cx.hi[k]] {
            if f.iter().all(|&i| pts[i as usize][k] == v) {
                return Some((k, v));
            }
        }
    }
    None
}

/// The hull faces by box plane, and the box subsegments (hull edges on a
/// box edge) by plane, for this pass.
struct BoxState {
    faces: Vec<([u32; 3], usize, f64)>,
}

impl BoxState {
    fn new(cx: &Ctx, d: &Delaunay) -> Result<BoxState> {
        let pts = d.points();
        let mut faces = Vec::new();
        for f in d.hull_faces() {
            let Some((k, v)) = face_plane(cx, pts, f) else {
                return Err(Error::Refused(format!(
                    "refine/box: hull face [{}, {}, {}] lies on no plane of the domain box",
                    f[0], f[1], f[2]
                )));
            };
            faces.push((f, k, v));
        }
        Ok(BoxState { faces })
    }

    /// The box subsegments on plane `(k, v)`: edges of its hull faces whose
    /// ends share a second extreme coordinate.
    fn edges_on(&self, cx: &Ctx, pts: &[Point], k: usize, v: f64) -> BTreeSet<(u32, u32)> {
        let mut out = BTreeSet::new();
        for &(f, fk, fv) in &self.faces {
            if fk != k || fv != v {
                continue;
            }
            for m in 0..3 {
                let (a, b) = (f[m], f[(m + 1) % 3]);
                let (pa, pb) = (pts[a as usize], pts[b as usize]);
                let on = (0..3).any(|j| {
                    j != k
                        && pa[j] == pb[j]
                        && (pa[j] == cx.lo[j] || pa[j] == cx.hi[j])
                });
                if on {
                    out.insert(key(a, b));
                }
            }
        }
        out
    }
}

/// The split of hull face `f` on plane `(k, v)`: its circumcentre moved
/// onto the plane exactly, or the box subsegments that point encroaches,
/// or (outside the face's rectangle) its centroid.
fn plan_box_face(cx: &Ctx, pts: &[Point], bs: &BoxState, f: [u32; 3], k: usize, v: f64, plan: &mut Plan) {
    if !plan.box_faces.insert(sorted3(f)) {
        return;
    }
    let (mut c, _) = circumsphere3(pts[f[0] as usize], pts[f[1] as usize], pts[f[2] as usize]);
    c[k] = v;
    let mut enc = false;
    for (a, b) in bs.edges_on(cx, pts, k, v) {
        if dot(sub(c, pts[a as usize]), sub(c, pts[b as usize])) < 0.0 {
            plan.box_edges.insert((a, b));
            enc = true;
        }
    }
    if enc {
        return;
    }
    let inside = (0..3).all(|j| j == k || (c[j] > cx.lo[j] && c[j] < cx.hi[j]));
    let x = if inside {
        c
    } else {
        let mut g = [0, 1, 2].map(|j| {
            (pts[f[0] as usize][j] + pts[f[1] as usize][j] + pts[f[2] as usize][j]) / 3.0
        });
        g[k] = v;
        g
    };
    if plan.box_keys.insert(bits(x)) {
        plan.box_pts.push(x);
    }
}

/// A circumcentre `x` outside the open box: the hull face that its
/// projection onto the box falls in, on the plane of the largest excess.
fn plan_outside(cx: &Ctx, pts: &[Point], bs: &BoxState, x: Point, plan: &mut Plan) -> Result<()> {
    let mut best: Option<(f64, usize, f64)> = None;
    for k in 0..3 {
        for (v, ex) in [(cx.lo[k], cx.lo[k] - x[k]), (cx.hi[k], x[k] - cx.hi[k])] {
            if ex >= 0.0 && best.map_or(true, |(b, _, _)| ex > b) {
                best = Some((ex, k, v));
            }
        }
    }
    let (_, k, v) = best.expect("a point outside the open box has an excess");
    let xc = [0, 1, 2].map(|j| if j == k { v } else { x[j].clamp(cx.lo[j], cx.hi[j]) });
    let (j1, j2) = ((k + 1) % 3, (k + 2) % 3);
    let p2 = |p: Point| [p[j1], p[j2]];
    for &(f, fk, fv) in &bs.faces {
        if fk != k || fv != v {
            continue;
        }
        let (a, b, c) = (p2(pts[f[0] as usize]), p2(pts[f[1] as usize]), p2(pts[f[2] as usize]));
        let y = p2(xc);
        let o = [orient2(a, b, y), orient2(b, c, y), orient2(c, a, y)];
        if o.iter().all(|&s| s >= 0.0) || o.iter().all(|&s| s <= 0.0) {
            plan_box_face(cx, pts, bs, f, k, v, plan);
            return Ok(());
        }
    }
    Err(Error::Refused(format!(
        "refine/box: the projection {xc:?} of a circumcentre lies in no hull face of its box plane"
    )))
}

/// SPEC-LIT §116.13: refines the face recovery `fr` of `surf` (the domain
/// box among its points as eight corners) until no kept tetrahedron of the
/// carve from `start` violates (116.17) or (116.18) under the size field
/// `h`, except the exempt ones next to a protecting ball. Refuses by name
/// an invalid `spec` (with `TetSpec`'s messages), a domain box without its
/// eight corners (`refine/box`), a size that is not finite and positive
/// (`refine/h`), a kept tetrahedron without a finite circumcentre
/// (`refine/centre`), a chain point that is not an edge site
/// (`refine/input`), a facet that cannot be triangulated (`refine/facet`), a
/// split that no longer separates its ends (`refine/split`), a pass with
/// nothing to insert (`refine/stall`) and a pass that would take the
/// insertions past `spec.max_inserted_points` (`refine/cap`); the carve's
/// and the kernel's refusals pass through.
pub fn refine(
    surf: &TetSurface,
    fr: FaceRecovery,
    start: &CarveStart,
    spec: &TetRefineSpec,
    h: &dyn Fn(Point) -> f64,
) -> Result<Refined> {
    if !(spec.radius_edge >= 2.0 && spec.radius_edge.is_finite()) {
        return Err(Error::Refused(format!(
            "tet.refine.radius_edge: must be >= 2 (Shewchuk 1998's termination bound, SPEC-LIT §116.13), got {}",
            spec.radius_edge
        )));
    }
    if !(spec.size_factor.is_finite() && spec.size_factor > 0.0) {
        return Err(Error::Refused(format!(
            "tet.refine.size_factor: must be finite and > 0, got {}",
            spec.size_factor
        )));
    }
    if spec.max_inserted_points == 0 {
        return Err(Error::Refused(
            "tet.refine.max_inserted_points: must be >= 1, got 0".to_string(),
        ));
    }
    let mut fr = fr;
    let n_in = fr.points.len() - fr.sites.len();
    let mut lo = [f64::INFINITY; 3];
    let mut hi = [f64::NEG_INFINITY; 3];
    for p in &fr.points {
        for k in 0..3 {
            lo[k] = lo[k].min(p[k]);
            hi[k] = hi[k].max(p[k]);
        }
    }
    let have: HashSet<[u64; 3]> = fr.points.iter().map(|&p| bits(p)).collect();
    for m in 0..8u32 {
        let c = [0, 1, 2].map(|k| if (m >> k) & 1 == 1 { hi[k] } else { lo[k] });
        if !have.contains(&bits(c)) {
            return Err(Error::Refused(format!(
                "refine/box: the corner {c:?} of the points' bounding box {lo:?} .. {hi:?} is not a \
                 point, so the convex hull is not the domain box"
            )));
        }
    }
    let tris = fr.tris.clone();
    let edges = fr.edges.clone();
    let cx = Ctx {
        tris: &tris,
        edges: &edges,
        frames: tris
            .iter()
            .map(|t| Frame::new([0, 1, 2].map(|k| fr.points[t[k] as usize])))
            .collect(),
        eidx: edges.iter().enumerate().map(|(e, ed)| ((ed[0], ed[1]), e)).collect(),
        edge_len: edges
            .iter()
            .map(|ed| norm(sub(fr.points[ed[1] as usize], fr.points[ed[0] as usize])))
            .collect(),
        protected: fr.protected.iter().copied().collect(),
        lo,
        hi,
    };
    let balls: Vec<(u32, f64)> = fr.protected.iter().map(|&v| (v, protect_radius(surf, v))).collect();
    // The chain parameters and the facet points, from the sites.
    let mut chain_t: Vec<Vec<f64>> = fr
        .chains
        .iter()
        .enumerate()
        .map(|(e, ch)| {
            ch.iter()
                .map(|&id| {
                    if (id as usize) < n_in {
                        if id == edges[e][0] { 0.0 } else { 1.0 }
                    } else {
                        match fr.sites[id as usize - n_in] {
                            SteinerSite::Edge { t, .. } => t,
                            _ => f64::NAN,
                        }
                    }
                })
                .collect()
        })
        .collect();
    if let Some(e) = chain_t.iter().position(|ct| ct.iter().any(|t| t.is_nan())) {
        return Err(Error::Refused(format!(
            "refine/input: edge {e}'s chain holds a point that is not an edge site"
        )));
    }
    let mut interior: Vec<Vec<(u32, f64, f64)>> = vec![Vec::new(); tris.len()];
    for (k, s) in fr.sites.iter().enumerate() {
        if let SteinerSite::Facet { tri, u, v } = *s {
            interior[tri as usize].push(((n_in + k) as u32, u, v));
        }
    }
    let near_ball = |pts: &[Point], x: Point| {
        balls
            .iter()
            .any(|&(v, r)| norm(sub(x, pts[v as usize])) <= r * (1.0 + 1e-9))
    };
    let mut stats = RefineStats::default();
    loop {
        stats.passes += 1;
        let facets = build_facets(&cx, &fr.chains, &chain_t, &interior)?;
        fr.subtris.clear();
        fr.sub_parent.clear();
        let mut submap: HashMap<[u32; 3], (usize, usize)> = HashMap::new();
        for (ti, (fp, local)) in facets.iter().enumerate() {
            for (li, lt) in local.iter().enumerate() {
                let s = lt.map(|x| fp.ids[x]);
                submap.insert(sorted3(s), (ti, li));
                fr.subtris.push(s);
                fr.sub_parent.push(ti as u32);
            }
        }
        let mut segmap: HashMap<(u32, u32), (usize, usize)> = HashMap::new();
        for (e, ch) in fr.chains.iter().enumerate() {
            for i in 0..ch.len() - 1 {
                segmap.insert(key(ch[i], ch[i + 1]), (e, i));
            }
        }
        let mut plan = Plan::default();
        let mut carved: Option<Carved> = None;
        let mut exempt: Vec<usize> = Vec::new();
        // 1. Conformity.
        let de = delaunay_edges(&fr.delaunay);
        for (e, ch) in fr.chains.iter().enumerate() {
            for i in 0..ch.len() - 1 {
                if !de.contains(&key(ch[i], ch[i + 1])) {
                    plan.seg.insert((e, i));
                }
            }
        }
        if plan.seg.is_empty() {
            let df = delaunay_faces(&fr.delaunay);
            for (ti, (fp, local)) in facets.iter().enumerate() {
                for (li, lt) in local.iter().enumerate() {
                    if !df.contains(&sorted3(lt.map(|x| fp.ids[x]))) {
                        plan_facet(&cx, &fr.points, &fr.chains, &facets, ti, li, &mut plan);
                    }
                }
            }
        }
        // 2. Encroachment by a vertex: subsegments by their link, then
        // subtriangles by their apexes (116.15).
        if plan.len() == 0 {
            let tets = fr.delaunay.finite_tets();
            let pts = &fr.points;
            for t in &tets {
                for i in 0..4 {
                    for j in i + 1..4 {
                        let Some(&(e, k)) = segmap.get(&key(t[i], t[j])) else { continue };
                        let (a, b) = (pts[t[i] as usize], pts[t[j] as usize]);
                        let hit = (0..4).filter(|&m| m != i && m != j).any(|m| {
                            let c = pts[t[m] as usize];
                            dot(sub(c, a), sub(c, b)) < 0.0
                        });
                        if hit {
                            plan.seg.insert((e, k));
                        }
                    }
                }
            }
            if plan.seg.is_empty() {
                let mut hit: BTreeSet<(usize, usize)> = BTreeSet::new();
                for t in &tets {
                    for skip in 0..4 {
                        let f = sorted3([t[(skip + 1) % 4], t[(skip + 2) % 4], t[(skip + 3) % 4]]);
                        let Some(&tl) = submap.get(&f) else { continue };
                        let (c, r) = circumsphere3(pts[f[0] as usize], pts[f[1] as usize], pts[f[2] as usize]);
                        if norm(sub(pts[t[skip] as usize], c)) < r {
                            hit.insert(tl);
                        }
                    }
                }
                for &(ti, li) in &hit {
                    plan_facet(&cx, &fr.points, &fr.chains, &facets, ti, li, &mut plan);
                }
            }
        }
        // 3. Quality.
        if plan.len() == 0 {
            let c = carve(surf, &fr, start)?;
            let mut bad: Vec<(f64, usize, TetQuality)> = Vec::new();
            for (ki, (&slot, &t)) in c.kept_slots.iter().zip(&c.tets).enumerate() {
                let p = tet_points(&fr.points, t);
                let q = quality(p, h);
                if !q.centre.iter().all(|x| x.is_finite()) {
                    return Err(Error::Refused(format!(
                        "refine/centre: kept tetrahedron {t:?} has no finite circumcentre"
                    )));
                }
                if !(q.h.is_finite() && q.h > 0.0) {
                    return Err(Error::Refused(format!(
                        "refine/h: the size at {:?} is {}, not finite and positive",
                        q.centre, q.h
                    )));
                }
                if !q.violates(spec) {
                    continue;
                }
                if p.iter().any(|&x| near_ball(&fr.points, x)) || near_ball(&fr.points, q.centre) {
                    exempt.push(ki);
                    continue;
                }
                bad.push((q.score(spec), slot as usize, q));
            }
            bad.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
            if !bad.is_empty() {
                let d = &fr.delaunay;
                let pts = d.points();
                let mut bs: Option<BoxState> = None;
                let mut marked: HashSet<u32> = HashSet::new();
                let pid0 = pts.len();
                for &(_, slot, q) in &bad {
                    let x = q.centre;
                    let inside = (0..3).all(|k| x[k] > lo[k] && x[k] < hi[k]);
                    if !inside {
                        if bs.is_none() {
                            bs = Some(BoxState::new(&cx, d)?);
                        }
                        plan_outside(&cx, pts, bs.as_ref().unwrap(), x, &mut plan)?;
                        stats.rejected += 1;
                        continue;
                    }
                    let pid = pid0 + plan.interior.len();
                    let cav = d.cavity(&x, pid, slot as u32)?;
                    let cavset: HashSet<u32> = cav.iter().copied().collect();
                    let mut enc_seg: Vec<(usize, usize)> = Vec::new();
                    let mut enc_tri: Vec<(usize, usize)> = Vec::new();
                    let mut enc_hull: Vec<[u32; 3]> = Vec::new();
                    for &t in &cav {
                        let v = d.tet(t);
                        if v[3] == INF {
                            continue;
                        }
                        let nb = d.neighbours(t);
                        for i in 0..4 {
                            let f = sorted3([v[(i + 1) % 4], v[(i + 2) % 4], v[(i + 3) % 4]]);
                            if let Some(&tl) = submap.get(&f) {
                                let (c, r) = circumsphere3(pts[f[0] as usize], pts[f[1] as usize], pts[f[2] as usize]);
                                if norm(sub(x, c)) < r || cavset.contains(&nb[i]) {
                                    enc_tri.push(tl);
                                }
                            } else if d.is_hull(nb[i]) {
                                let (c, r) = circumsphere3(pts[f[0] as usize], pts[f[1] as usize], pts[f[2] as usize]);
                                if norm(sub(x, c)) < r {
                                    enc_hull.push(f);
                                }
                            }
                        }
                        for i in 0..4 {
                            for j in i + 1..4 {
                                if let Some(&ek) = segmap.get(&key(v[i], v[j])) {
                                    let (a, b) = (pts[v[i] as usize], pts[v[j] as usize]);
                                    if dot(sub(x, a), sub(x, b)) < 0.0 {
                                        enc_seg.push(ek);
                                    }
                                }
                            }
                        }
                    }
                    if !enc_seg.is_empty() {
                        plan.seg.extend(enc_seg);
                        stats.rejected += 1;
                        continue;
                    }
                    if !enc_tri.is_empty() {
                        enc_tri.sort_unstable();
                        enc_tri.dedup();
                        for (ti, li) in enc_tri {
                            plan_facet(&cx, &fr.points, &fr.chains, &facets, ti, li, &mut plan);
                        }
                        stats.rejected += 1;
                        continue;
                    }
                    if !enc_hull.is_empty() {
                        if bs.is_none() {
                            bs = Some(BoxState::new(&cx, d)?);
                        }
                        let bsr = bs.as_ref().unwrap();
                        for f in enc_hull {
                            let (k, v) = face_plane(&cx, pts, f).ok_or_else(|| {
                                Error::Refused(format!(
                                    "refine/box: hull face [{}, {}, {}] lies on no plane of the domain box",
                                    f[0], f[1], f[2]
                                ))
                            })?;
                            plan_box_face(&cx, pts, bsr, f, k, v, &mut plan);
                        }
                        stats.rejected += 1;
                        continue;
                    }
                    if cav.iter().any(|t| marked.contains(t)) {
                        stats.deferred += 1;
                        continue;
                    }
                    for &t in &cav {
                        marked.insert(t);
                        for u in d.neighbours(t) {
                            marked.insert(u);
                        }
                    }
                    plan.interior.push(x);
                }
            }
            carved = Some(c);
            if bad.is_empty() {
                let carved = carved.take().unwrap();
                return Ok(Refined {
                    recovery: fr,
                    carved,
                    stats,
                    exempt,
                    balls,
                });
            }
        }
        // Apply the plan.
        let new = plan.len();
        if new == 0 {
            return Err(Error::Refused(format!(
                "refine/stall: pass {} found work but scheduled no point",
                stats.passes
            )));
        }
        let cap = spec.max_inserted_points;
        if (stats.inserted() + new) as u64 > cap {
            let left = carved
                .as_ref()
                .map(|c| format!("; {} kept tetrahedra", c.tets.len()))
                .unwrap_or_default();
            return Err(Error::Refused(format!(
                "refine/cap: {} points inserted after {} passes, the next pass needs {} more, cap \
                 {cap}{left}",
                stats.inserted(),
                stats.passes,
                new
            )));
        }
        // Interior points first, in acceptance order, so each takes the id
        // its cavity was computed with.
        for x in plan.interior {
            let id = fr.delaunay.insert_point(x)?;
            debug_assert_eq!(id as usize, fr.points.len());
            fr.points.push(x);
            fr.sites.push(SteinerSite::Volume);
            stats.interior += 1;
        }
        // Subsegment splits in (edge, position) order, inserted back to front.
        let mut seg_new: Vec<(usize, usize, u32, f64)> = Vec::with_capacity(plan.seg.len());
        for &(e, i) in &plan.seg {
            let t = split_t(&cx, &chain_t, e, i)?;
            let ed = edges[e];
            let x = lerp(fr.points[ed[0] as usize], fr.points[ed[1] as usize], t);
            let id = fr.delaunay.insert_point(x)?;
            fr.points.push(x);
            fr.sites.push(SteinerSite::Edge { edge: e as u32, t });
            seg_new.push((e, i, id, t));
            stats.segment_splits += 1;
        }
        for &(e, i, id, t) in seg_new.iter().rev() {
            fr.chains[e].insert(i + 1, id);
            chain_t[e].insert(i + 1, t);
        }
        for &(ti, u, v, centroid) in &plan.facet {
            let x = cx.frames[ti].at3(u, v);
            let id = fr.delaunay.insert_point(x)?;
            fr.points.push(x);
            fr.sites.push(SteinerSite::Facet { tri: ti as u32, u, v });
            interior[ti].push((id, u, v));
            stats.facet_splits += 1;
            if centroid {
                stats.facet_centroids += 1;
            }
        }
        for &x in &plan.box_pts {
            fr.delaunay.insert_point(x)?;
            fr.points.push(x);
            fr.sites.push(SteinerSite::Volume);
            stats.box_face_splits += 1;
        }
        for &(a, b) in &plan.box_edges {
            let x = lerp(fr.points[a as usize], fr.points[b as usize], 0.5);
            fr.delaunay.insert_point(x)?;
            fr.points.push(x);
            fr.sites.push(SteinerSite::Volume);
            stats.box_edge_splits += 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::recover::recover_faces;
    use super::super::remesh::{remesh, RemeshOptions};
    use super::super::sizefield::{OctreeOptions, SizeField, SizeParams, SizeSource};
    use super::super::surfprep::prepare;
    use super::super::TetRecoverSpec;
    use super::*;
    use std::sync::OnceLock;

    /// The level-`level` icosphere of radius 1, one patch "sphere".
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
                    *cache.entry((x.min(y), x.max(y))).or_insert_with(|| {
                        pts.push(unit(lerp(pts[x as usize], pts[y as usize], 0.5)));
                        pts.len() as u32 - 1
                    })
                };
                let (ab, bc, ca) = (mid(a, b), mid(b, c), mid(c, a));
                next.extend([[a, ab, ca], [b, bc, ab], [c, ca, bc], [ab, bc, ca]]);
            }
            tris = next;
        }
        let tri_patch = vec![0; tris.len()];
        TetSurface { points: pts, tris, tri_patch, patch_names: vec!["sphere".to_string()] }
    }

    /// The domain box `[lo, hi]^3` as its 8 corner points, on no triangle.
    fn add_box(s: &mut TetSurface, lo: f64, hi: f64) {
        for k in 0..8u32 {
            let c = |b: u32| if (k >> b) & 1 == 1 { hi } else { lo };
            s.points.push([c(0), c(1), c(2)]);
        }
    }

    /// The axis-aligned cube `[lo, hi]^3`, two triangles per face, wound
    /// outward, one patch.
    fn cube(lo: f64, hi: f64) -> TetSurface {
        let points: Vec<Point> = (0..8u32)
            .map(|k| {
                let c = |b: u32| if (k >> b) & 1 == 1 { hi } else { lo };
                [c(0), c(1), c(2)]
            })
            .collect();
        let faces: [[u32; 4]; 6] = [
            [0, 4, 6, 2],
            [1, 3, 7, 5],
            [0, 1, 5, 4],
            [2, 6, 7, 3],
            [0, 2, 3, 1],
            [4, 5, 7, 6],
        ];
        let mut tris = Vec::new();
        for f in faces {
            tris.push([f[0], f[1], f[2]]);
            tris.push([f[0], f[2], f[3]]);
        }
        let tri_patch = vec![0; tris.len()];
        TetSurface { points, tris, tri_patch, patch_names: vec!["cube".to_string()] }
    }

    fn rel(a: f64, b: f64) -> f64 {
        (a - b).abs() / b.abs()
    }

    /// The fixture: the level-3 icosphere remeshed to the size field, in
    /// the box `[-2, 2]^3`, refined from the hull. The size field has one
    /// patch source on the icosphere, `h = 0.25` on it, `H = 0.5` and
    /// `g = 1.2`.
    struct Case {
        n_body: u32,
        field: SizeField,
        refined: Refined,
        seconds: f64,
    }

    fn case() -> &'static Case {
        static CASE: OnceLock<Case> = OnceLock::new();
        CASE.get_or_init(|| {
            let ico3 = ico(3);
            let tris: Vec<[Point; 3]> = ico3
                .tris
                .iter()
                .map(|t| t.map(|v| ico3.points[v as usize]))
                .collect();
            let field = SizeField::build(
                SizeParams::new(0.5, 1.2, None).unwrap(),
                &[SizeSource::Patch { name: "sphere".to_string(), tris, distance: 0.0, size: 0.25 }],
                [-2.0; 3],
                [2.0; 3],
                OctreeOptions::default(),
            )
            .unwrap();
            let prep = prepare(&ico3, 60.0).unwrap();
            let r = remesh(&prep, &|x: Point| field.h(x), &RemeshOptions::default()).unwrap();
            let mut surf = r.surface.clone();
            let n_body = surf.tris.len() as u32;
            add_box(&mut surf, -2.0, 2.0);
            let fr = recover_faces(&surf, &r.feature_edges, &TetRecoverSpec::default()).unwrap();
            let t0 = std::time::Instant::now();
            let refined = refine(
                &surf,
                fr,
                &CarveStart::Hull,
                &TetRefineSpec::default(),
                &|x: Point| field.h(x),
            )
            .unwrap();
            let seconds = t0.elapsed().as_secs_f64();
            Case { n_body, field, refined, seconds }
        })
    }

    #[test]
    fn radius_edge_bound() {
        let c = case();
        let r = &c.refined;
        let fr = &r.recovery;
        let spec = TetRefineSpec::default();
        let h = |x: Point| c.field.h(x);
        let exempt: HashSet<usize> = r.exempt.iter().copied().collect();
        let mut violators = 0usize;
        let mut worst_rho = 0.0f64;
        let mut worst_size = 0.0f64;
        for (k, t) in r.carved.tets.iter().enumerate() {
            let q = quality(tet_points(&fr.points, *t), &h);
            worst_rho = worst_rho.max(q.rho);
            worst_size = worst_size.max(q.radius / (spec.size_factor * q.h));
            if q.violates(&spec) && !exempt.contains(&k) {
                violators += 1;
            }
        }
        let (present, total) = fr.recovered();
        let de = delaunay_edges(&fr.delaunay);
        let subs = fr.subsegments();
        let seg_present = subs.iter().filter(|s| de.contains(&(s[0], s[1]))).count();
        // (116.16) on the refined carve.
        let subs_body: Vec<[u32; 3]> = fr
            .subtris
            .iter()
            .zip(&fr.sub_parent)
            .filter(|(_, &p)| p < c.n_body)
            .map(|(s, _)| *s)
            .collect();
        let o = fr.points[subs_body[0][0] as usize];
        let mut v_body = 0.0;
        for s in &subs_body {
            let x = [0, 1, 2].map(|k| sub(fr.points[s[k] as usize], o));
            v_body += dot(x[0], cross(x[1], x[2]));
        }
        v_body /= 6.0;
        let e1 = rel(r.carved.volume, 64.0 - v_body);
        let e2 = rel(r.carved.boundary_volume(&fr.points), r.carved.volume);
        println!(
            "refine sphere: {} points, {} kept tets, {:?}, {:.2} s; violators {violators} outside \
             the reported set of {} (balls {}); worst rho {worst_rho:.4}, worst R/(c h) \
             {worst_size:.4}; subtriangles {present}/{total}, subsegments {seg_present}/{}; \
             (116.16) rel {e1:.2e}, (116.16b) rel {e2:.2e}",
            fr.points.len(),
            r.carved.tets.len(),
            r.stats,
            c.seconds,
            r.exempt.len(),
            r.balls.len(),
            subs.len()
        );
        assert_eq!(violators, 0, "no tet violates (116.17) or (116.18) outside the reported set");
        assert!(r.balls.is_empty() && r.exempt.is_empty(), "the sphere has no protected vertex");
        assert_eq!(present, total, "every subtriangle is a face");
        assert_eq!(seg_present, subs.len(), "every subsegment is an edge");
        assert!(e1 <= 1e-12, "(116.16) rel {e1:e}");
        assert!(e2 <= 1e-12, "(116.16b) rel {e2:e}");
        assert!(r.stats.inserted() > 0 && r.stats.interior > 0);
    }

    #[test]
    fn size_conformance() {
        let c = case();
        let r = &c.refined;
        let m = mean_edge_ratio(&r.recovery.points, &r.carved.tets, &|x: Point| c.field.h(x));
        let ok = (0.85..=1.15).contains(&m);
        println!(
            "refine size: mean edge/h(x) = {m:.4} over the kept tets' edges, band [0.85, 1.15]: {}",
            if ok { "PASS" } else { "MISSED" }
        );
        assert!(ok, "mean edge/h {m}");
    }

    #[test]
    fn count_predicted() {
        let c = case();
        let r = &c.refined;
        let n = r.carved.tets.len() as f64;
        let np = predicted_count(&r.recovery.points, &r.carved.tets, &|x: Point| c.field.h(x));
        let dev = n / np - 1.0;
        let ok = dev.abs() <= 0.2;
        println!(
            "refine count: {n} kept tets against N_pred = {np:.1} (116.20), deviation {:+.2} %: {}",
            100.0 * dev,
            if ok { "PASS" } else { "MISSED" }
        );
        assert!(ok, "count deviation {dev}");
    }

    /// The cube `[-0.5, 0.5]^3` remeshed at a uniform `h = 0.4` (§116.10,
    /// its 12 edges held as features), in `[-1, 1]^3`.
    fn small() -> (TetSurface, FaceRecovery) {
        let prep = prepare(&cube(-0.5, 0.5), 60.0).unwrap();
        let r = remesh(&prep, &|_: Point| 0.4, &RemeshOptions::default()).unwrap();
        let mut s = r.surface.clone();
        add_box(&mut s, -1.0, 1.0);
        let fr = recover_faces(&s, &r.feature_edges, &TetRecoverSpec::default()).unwrap();
        (s, fr)
    }

    #[test]
    fn cap_and_refusals() {
        let h = |_: Point| 0.4;
        let (s, fr) = small();
        let r = refine(&s, fr, &CarveStart::Hull, &TetRefineSpec::default(), &h).unwrap();
        let k = r.stats.inserted();
        println!("refine small cube: {} kept tets, {:?}", r.carved.tets.len(), r.stats);
        assert!(k > 0);
        let mut spec = TetRefineSpec::default();
        spec.max_inserted_points = k as u64;
        let (s, fr) = small();
        let r2 = refine(&s, fr, &CarveStart::Hull, &spec, &h).unwrap();
        assert_eq!(r2.stats, r.stats, "the cap at k passes with the same work");
        assert_eq!(r2.carved.tets, r.carved.tets);
        spec.max_inserted_points = k as u64 - 1;
        let (s, fr) = small();
        let e = refine(&s, fr, &CarveStart::Hull, &spec, &h).unwrap_err().to_string();
        println!("{e}");
        assert!(e.contains("refine/cap: "), "{e}");
        let bad = |f: &dyn Fn(&mut TetRefineSpec), want: &str| {
            let mut sp = TetRefineSpec::default();
            f(&mut sp);
            let (s, fr) = small();
            let e = refine(&s, fr, &CarveStart::Hull, &sp, &h).unwrap_err().to_string();
            assert!(e.contains(want), "{e}");
        };
        bad(
            &|s| s.radius_edge = 1.9,
            "tet.refine.radius_edge: must be >= 2 (Shewchuk 1998's termination bound, SPEC-LIT §116.13), got 1.9",
        );
        bad(&|s| s.size_factor = -1.0, "tet.refine.size_factor: must be finite and > 0, got -1");
        bad(&|s| s.max_inserted_points = 0, "tet.refine.max_inserted_points: must be >= 1, got 0");
        // No domain box: the cube alone has its own corners as the box, so
        // drop one box corner from a cube in a larger box instead.
        let mut s = cube(-0.5, 0.5);
        add_box(&mut s, -1.0, 1.0);
        s.points.pop();
        s.points.push([0.9, 0.9, 0.9]);
        let fr = recover_faces(&s, &[], &TetRecoverSpec::default()).unwrap();
        let e = refine(&s, fr, &CarveStart::Hull, &TetRefineSpec::default(), &h)
            .unwrap_err()
            .to_string();
        assert!(e.contains("refine/box: the corner [1.0, 1.0, 1.0]"), "{e}");
        let (s, fr) = small();
        let e = refine(&s, fr, &CarveStart::Hull, &TetRefineSpec::default(), &|_: Point| f64::NAN)
            .unwrap_err()
            .to_string();
        assert!(e.contains("refine/h: "), "{e}");
    }
}
