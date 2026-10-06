// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.
// Provenance: see PROVENANCE.md. No GPL-licensed source was consulted.

//! The surface preparation of SPEC-LIT §116.9: the checks that refuse a bad
//! input surface by name before any meshing, and the sharp edges, corners
//! and patch seams the later units hold. The input is a [`TetSurface`], the
//! f64 widening of a reader's `Surface`, and every check is decided in f64
//! by the exact `orient3d` of §116.4, so each answer is the same in both
//! builds (§116.2). Each refusal is a named `Error::Refused`, the name
//! starting the message, in the order of §116.9's table:
//! `surface/feature_angle_deg`, `surface/input`, `surface/degenerate`,
//! `surface/closed`, `surface/orientation`, `surface/inward`,
//! `surface/self-intersection`. The overlap predicates are (116.13b),
//! (116.13c), (116.13d), (116.13e) and (116.13f); the shell volume is
//! (116.13a); the features are (92.34) and (92.35) on §92.12's reading, with
//! the patch seams joined into the feature set before the corners are found.
//! The chaining into polylines is §92.12's.
//!
//! Provenance: ORIGINAL. Written from Möller (1997), J. Graphics Tools 2(2):25-30, with the exact predicates of Shewchuk (1997); the exact overlap test is derived in SPEC-LIT §116.9. No GPL-licensed source was consulted.

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet, VecDeque};

use super::predicates::{in_exact_domain, orient3d};
use super::Point;
use crate::error::{Error, Result};
use crate::surface::Surface;

/// An input surface in pure f64 - §116.2's rule that every coordinate inside
/// `tetmesh` is a `[f64; 3]`, whatever `Scalar` is.
#[derive(Debug, Clone, PartialEq)]
pub struct TetSurface {
    pub points: Vec<Point>,
    pub tris: Vec<[u32; 3]>,
    pub tri_patch: Vec<u32>,
    pub patch_names: Vec<String>,
}

impl TetSurface {
    /// The exact f64 widening of §116.9: each coordinate widens with
    /// `as f64`, the identity in the f64 build and exact under `single`, so
    /// this carries the reader's triangulation bit for bit.
    pub fn from_surface(s: &Surface) -> TetSurface {
        TetSurface {
            points: s
                .points
                .iter()
                .map(|p| [p.x as f64, p.y as f64, p.z as f64])
                .collect(),
            tris: s.tris.clone(),
            tri_patch: s.tri_patch.clone(),
            patch_names: s.patch_names.clone(),
        }
    }
}

// ==========================================================================
//  The exact pair predicates of §116.9
// ==========================================================================

/// `a - b`, in f64.
fn sub3(a: &Point, b: &Point) -> Point {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

/// The f64 cross product.
fn cross3(a: &Point, b: &Point) -> Point {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

/// The f64 dot product.
fn dot3(a: &Point, b: &Point) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// The sign of an f64, with 0 (and -0.0) its own sign.
fn sgn(x: f64) -> i32 {
    if x > 0.0 {
        1
    } else if x < 0.0 {
        -1
    } else {
        0
    }
}

/// The undirected edge key: `(min, max)`, so maps are independent of the
/// order the triangles arrived in.
fn edge_key(a: u32, b: u32) -> (u32, u32) {
    if a < b {
        (a, b)
    } else {
        (b, a)
    }
}

/// The apex (116.13e): `w = a + (L_e / |n|) n` with `n` the triangle's f64
/// normal and `L_e` its longest edge - a point one longest-edge-length above
/// the plane. Any point off it gives the same in-plane signs, so the
/// rounding costs nothing.
pub fn triangle_apex(t: &[Point; 3]) -> Point {
    let n = cross3(&sub3(&t[1], &t[0]), &sub3(&t[2], &t[0]));
    let nn = dot3(&n, &n).sqrt();
    let e0 = dot3(&sub3(&t[1], &t[0]), &sub3(&t[1], &t[0]));
    let e1 = dot3(&sub3(&t[2], &t[1]), &sub3(&t[2], &t[1]));
    let e2 = dot3(&sub3(&t[0], &t[2]), &sub3(&t[0], &t[2]));
    let le = e0.max(e1).max(e2).sqrt();
    let s = le / nn;
    [t[0][0] + s * n[0], t[0][1] + s * n[1], t[0][2] + s * n[2]]
}

/// Degenerate by §116.9's order 2: the f64 cross product exactly zero, or
/// the apex (116.13e) on the triangle's own plane.
pub fn is_degenerate(t: &[Point; 3]) -> bool {
    let n = cross3(&sub3(&t[1], &t[0]), &sub3(&t[2], &t[0]));
    if n[0] == 0.0 && n[1] == 0.0 && n[2] == 0.0 {
        return true;
    }
    orient3d(&t[0], &t[1], &t[2], &triangle_apex(t)) == 0.0
}

/// Whether the closed segment `[p, q]` meets the closed triangle `t` when
/// the segment is not coplanar with it - (116.13d). Both end signs 0 is the
/// skipped in-plane case; the same non-zero sign is a miss; otherwise the
/// three side signs decide, all one sign or the other, touching allowed.
pub fn segment_meets_triangle(p: &Point, q: &Point, t: &[Point; 3]) -> bool {
    let sp = sgn(orient3d(&t[0], &t[1], &t[2], p));
    let sq = sgn(orient3d(&t[0], &t[1], &t[2], q));
    if sp == 0 && sq == 0 {
        return false;
    }
    if sp != 0 && sp == sq {
        return false;
    }
    let s1 = sgn(orient3d(p, q, &t[0], &t[1]));
    let s2 = sgn(orient3d(p, q, &t[1], &t[2]));
    let s3 = sgn(orient3d(p, q, &t[2], &t[0]));
    (s1 >= 0 && s2 >= 0 && s3 >= 0) || (s1 <= 0 && s2 <= 0 && s3 <= 0)
}

/// The in-plane sign `o_w` of (116.13e), against the lifted point `w`.
fn ow(w: &Point, a: &Point, b: &Point, c: &Point) -> i32 {
    sgn(orient3d(a, b, c, w))
}

/// Whether two coplanar closed segments meet - (116.13e)'s rule, with the
/// all-four-zero (collinear) case of §116.9 decided by closed interval
/// overlap on the axis where `|q - p|` has its largest component.
fn coplanar_segments_meet(p: &Point, q: &Point, r: &Point, s: &Point, w: &Point) -> bool {
    let s1 = ow(w, p, q, r);
    let s2 = ow(w, p, q, s);
    let s3 = ow(w, r, s, p);
    let s4 = ow(w, r, s, q);
    if s1 == 0 && s2 == 0 && s3 == 0 && s4 == 0 {
        let d = sub3(q, p);
        let axis = if d[0].abs() >= d[1].abs() && d[0].abs() >= d[2].abs() {
            0
        } else if d[1].abs() >= d[2].abs() {
            1
        } else {
            2
        };
        let lo1 = p[axis].min(q[axis]);
        let hi1 = p[axis].max(q[axis]);
        let lo2 = r[axis].min(s[axis]);
        let hi2 = r[axis].max(s[axis]);
        lo1 <= hi2 && lo2 <= hi1
    } else {
        !((s1 != 0 && s1 == s2) || (s3 != 0 && s3 == s4))
    }
}

/// Whether a point of the common plane lies in the closed triangle `t`, by
/// (116.13e)'s three in-plane signs against the same `w`.
fn coplanar_point_in_tri(x: &Point, t: &[Point; 3], w: &Point) -> bool {
    let s1 = ow(w, &t[0], &t[1], x);
    let s2 = ow(w, &t[1], &t[2], x);
    let s3 = ow(w, &t[2], &t[0], x);
    (s1 >= 0 && s2 >= 0 && s3 >= 0) || (s1 <= 0 && s2 <= 0 && s3 <= 0)
}

/// Whether two closed triangles meet, touching included - §116.9's exact
/// overlap test. The plane rejection is (116.13b) with each signed distance
/// replaced by the exact sign of `orient3d`; if every `d_1` is 0 the
/// triangles are coplanar and (116.13e) decides with `w` the apex of `t2`
/// used for every in-plane sign; otherwise (116.13c) decides through
/// [`segment_meets_triangle`]. Neither triangle may be degenerate: the
/// caller refuses degenerates at order 2.
pub fn tri_tri_intersect(t1: &[Point; 3], t2: &[Point; 3]) -> bool {
    let d1 = [
        sgn(orient3d(&t2[0], &t2[1], &t2[2], &t1[0])),
        sgn(orient3d(&t2[0], &t2[1], &t2[2], &t1[1])),
        sgn(orient3d(&t2[0], &t2[1], &t2[2], &t1[2])),
    ];
    let d2 = [
        sgn(orient3d(&t1[0], &t1[1], &t1[2], &t2[0])),
        sgn(orient3d(&t1[0], &t1[1], &t1[2], &t2[1])),
        sgn(orient3d(&t1[0], &t1[1], &t1[2], &t2[2])),
    ];
    for d in [&d1, &d2] {
        if d.iter().all(|&x| x == 1) || d.iter().all(|&x| x == -1) {
            return false;
        }
    }
    if d1.iter().all(|&x| x == 0) {
        let w = triangle_apex(t2);
        for i in 0..3 {
            for j in 0..3 {
                if coplanar_segments_meet(&t1[i], &t1[(i + 1) % 3], &t2[j], &t2[(j + 1) % 3], &w) {
                    return true;
                }
            }
        }
        for v in t1 {
            if coplanar_point_in_tri(v, t2, &w) {
                return true;
            }
        }
        for v in t2 {
            if coplanar_point_in_tri(v, t1, &w) {
                return true;
            }
        }
        return false;
    }
    for i in 0..3 {
        if segment_meets_triangle(&t1[i], &t1[(i + 1) % 3], t2)
            || segment_meets_triangle(&t2[i], &t2[(i + 1) % 3], t1)
        {
            return true;
        }
    }
    false
}

/// Whether the two triangles sharing the edge `(u, v)` fold onto each other
/// across it - (116.13f): `b` on the plane of `(u, v, a)`, and both opposite
/// points on the same side of that plane, judged against the apex of
/// `(u, v, a)`.
pub fn fold(u: &Point, v: &Point, a: &Point, b: &Point) -> bool {
    if orient3d(u, v, a, b) != 0.0 {
        return false;
    }
    let w = triangle_apex(&[*u, *v, *a]);
    let sa = sgn(orient3d(u, v, a, &w));
    let sb = sgn(orient3d(u, v, b, &w));
    sa != 0 && sa == sb
}

/// The shell's enclosed volume (116.13a): the sum of the signed tetrahedra
/// to the origin `o`, the first point of the shell's lowest triangle, so the
/// terms stay small far from the coordinate origin. `members` ascending, as
/// the shell's own triangle list is.
pub fn shell_volume(points: &[Point], tris: &[[u32; 3]], members: &[u32]) -> f64 {
    let o = &points[tris[members[0] as usize][0] as usize];
    let mut sum = 0.0;
    for &m in members {
        let t = &tris[m as usize];
        let x0 = &points[t[0] as usize];
        let x1 = &points[t[1] as usize];
        let x2 = &points[t[2] as usize];
        sum += dot3(&sub3(x0, o), &cross3(&sub3(x1, o), &sub3(x2, o)));
    }
    sum / 6.0
}

/// What §116.9's pass found, one row per check, for the run log.
#[derive(Debug, Clone, PartialEq)]
pub struct SurfprepReport {
    pub n_points: usize,
    pub n_tris: usize,
    pub n_shells: usize,
    pub shell_volumes: Vec<f64>,
    pub n_feature_edges: usize,
    pub n_seam_edges: usize,
    pub n_polylines: usize,
    pub n_closed_polylines: usize,
    pub n_corners: usize,
    pub n_candidate_pairs: usize,
    pub n_vertex_sharing_pairs: usize,
}

impl SurfprepReport {
    /// One line for a run log, in the report summaries' style.
    pub fn summary(&self) -> String {
        format!(
            "surfprep: {} triangle(s), {} shell(s), {} feature edge(s) ({} seam), {} polyline(s) ({} closed), {} corner(s); {} pair(s) tested, {} vertex-sharing pair(s) not tested\n",
            self.n_tris,
            self.n_shells,
            self.n_feature_edges,
            self.n_seam_edges,
            self.n_polylines,
            self.n_closed_polylines,
            self.n_corners,
            self.n_candidate_pairs,
            self.n_vertex_sharing_pairs,
        )
    }
}

/// The prepared surface: the input, and everything the later units hold -
/// the shell of each triangle, the feature edges with their seam flags, the
/// polylines chained from them, the corners, and the report.
#[derive(Debug, Clone, PartialEq)]
pub struct PreparedSurface {
    pub surface: TetSurface,
    pub shell_of_tri: Vec<u32>,
    pub feature_edges: Vec<[u32; 2]>,
    pub edge_is_seam: Vec<bool>,
    pub polylines: Vec<Vec<u32>>,
    pub corners: Vec<u32>,
    pub report: SurfprepReport,
}

// ==========================================================================
//  The §116.9 pass
// ==========================================================================

/// Runs §116.9's table in order - the angle, the input, the degenerates,
/// closedness, orientation, the enclosed volumes, self-intersection - and
/// returns the first failure, named. On success everything the later units
/// hold: the shells, the feature edges and seams, the corners, the
/// polylines, and the report.
pub fn prepare(surf: &TetSurface, feature_angle_deg: f64) -> Result<PreparedSurface> {
    // Order 0: the angle.
    if !feature_angle_deg.is_finite() || feature_angle_deg <= 0.0 || feature_angle_deg >= 180.0 {
        return Err(Error::Refused(format!(
            "surface/feature_angle_deg: {feature_angle_deg} must be finite and inside (0, 180)"
        )));
    }
    let n = surf.tris.len();
    // Order 1: the input's shape first, then its values.
    if n == 0 {
        return Err(Error::Refused("surface/input: no triangles".into()));
    }
    if surf.patch_names.is_empty() {
        return Err(Error::Refused("surface/input: no patches".into()));
    }
    if surf.tri_patch.len() != n {
        return Err(Error::Refused(format!(
            "surface/input: tri_patch has {} entries for {} triangles",
            surf.tri_patch.len(),
            n
        )));
    }
    let np = surf.points.len();
    for (t, tri) in surf.tris.iter().enumerate() {
        for &i in tri.iter() {
            if i as usize >= np {
                return Err(Error::Refused(format!(
                    "surface/input: triangle {t} references point {i}, but there are {np} points"
                )));
            }
        }
        let p = surf.tri_patch[t];
        if p as usize >= surf.patch_names.len() {
            return Err(Error::Refused(format!(
                "surface/input: triangle {t} has patch id {p}, but there are {} patches",
                surf.patch_names.len()
            )));
        }
        for (a, b) in [(0usize, 1usize), (1, 2), (0, 2)] {
            if tri[a] == tri[b] {
                return Err(Error::Refused(format!(
                    "surface/input: triangle {t} repeats point {}",
                    tri[a]
                )));
            }
        }
    }
    for (i, p) in surf.points.iter().enumerate() {
        for d in 0..3 {
            if !in_exact_domain(p[d]) {
                return Err(Error::Refused(format!(
                    "surface/input: point {i} has coordinate {}, outside the exact domain (116.3f)",
                    p[d]
                )));
            }
        }
    }
    let tri: Vec<[Point; 3]> = surf
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
    // Order 2: degenerate triangles, counted all, named at the lowest.
    let degenerate: Vec<bool> = tri.iter().map(|t| is_degenerate(t)).collect();
    let n_degen = degenerate.iter().filter(|&&d| d).count();
    if n_degen > 0 {
        let t = degenerate.iter().position(|&d| d).unwrap();
        let name = &surf.patch_names[surf.tri_patch[t] as usize];
        return Err(Error::Refused(format!(
            "surface/degenerate: {n_degen} degenerate triangle(s); first: triangle {t} (patch \"{name}\")"
        )));
    }
    // Order 3: every undirected edge on exactly two triangles - §23.2's
    // edge counting, re-derived because the tetrahedral path has no
    // -permissive fallback (§116.9).
    let mut edges: BTreeMap<(u32, u32), Vec<(u32, bool)>> = BTreeMap::new();
    for (t, tr) in surf.tris.iter().enumerate() {
        for k in 0..3 {
            let a = tr[k];
            let b = tr[(k + 1) % 3];
            let (key, fwd) = if a < b { ((a, b), true) } else { ((b, a), false) };
            edges.entry(key).or_default().push((t as u32, fwd));
        }
    }
    let mut n_open = 0usize;
    let mut n_nonman = 0usize;
    let mut first_bad: Option<((u32, u32), u32)> = None;
    for (&key, inc) in &edges {
        match inc.len() {
            1 => n_open += 1,
            2 => {}
            _ => n_nonman += 1,
        }
        if inc.len() != 2 && first_bad.is_none() {
            let t = inc.iter().map(|e| e.0).min().unwrap();
            first_bad = Some((key, t));
        }
    }
    if let Some(((a, b), t)) = first_bad {
        let name = &surf.patch_names[surf.tri_patch[t as usize] as usize];
        return Err(Error::Refused(format!(
            "surface/closed: {n_open} open edge(s), {n_nonman} non-manifold edge(s); first: edge ({a}, {b}) of triangle {t} (patch \"{name}\"). Surface::require_closed (§23.2) refuses this surface, and the tetrahedral path has no -permissive fallback (§116.9)"
        )));
    }
    // Order 4: shells and orientation. A shell is a connected component of
    // triangles joined across edges, numbered by its lowest triangle.
    fn find(parent: &mut Vec<u32>, x: u32) -> u32 {
        let mut r = x;
        while parent[r as usize] != r {
            parent[r as usize] = parent[parent[r as usize] as usize];
            r = parent[r as usize];
        }
        r
    }
    let mut parent: Vec<u32> = (0..n as u32).collect();
    for inc in edges.values() {
        if inc.len() == 2 {
            let a = find(&mut parent, inc[0].0);
            let b = find(&mut parent, inc[1].0);
            if a != b {
                parent[a as usize] = b;
            }
        }
    }
    let mut shell_of_tri = vec![0u32; n];
    let mut shells: Vec<Vec<u32>> = Vec::new();
    let mut root_shell: HashMap<u32, u32> = HashMap::new();
    for t in 0..n as u32 {
        let r = find(&mut parent, t);
        let s = *root_shell.entry(r).or_insert_with(|| {
            shells.push(Vec::new());
            (shells.len() - 1) as u32
        });
        shells[s as usize].push(t);
        shell_of_tri[t as usize] = s;
    }
    for (s, members) in shells.iter().enumerate() {
        let lo = members[0];
        let mut parity: HashMap<u32, bool> = HashMap::new();
        parity.insert(lo, false);
        let mut queue: VecDeque<u32> = VecDeque::new();
        queue.push_back(lo);
        let mut non_orientable = false;
        while let Some(t) = queue.pop_front() {
            let pt = parity[&t];
            let tr = &surf.tris[t as usize];
            for k in 0..3 {
                let (a, b) = (tr[k], tr[(k + 1) % 3]);
                let key = if a < b { (a, b) } else { (b, a) };
                let fwd = a < b;
                for &(u, f2) in &edges[&key] {
                    if u == t {
                        continue;
                    }
                    let want = if fwd == f2 { !pt } else { pt };
                    match parity.get(&u) {
                        Some(&p) if p != want => non_orientable = true,
                        Some(_) => {}
                        None => {
                            parity.insert(u, want);
                            queue.push_back(u);
                        }
                    }
                }
            }
            if non_orientable {
                break;
            }
        }
        if non_orientable {
            let name = &surf.patch_names[surf.tri_patch[lo as usize] as usize];
            return Err(Error::Refused(format!(
                "surface/orientation: shell {s} (lowest triangle {lo}, patch \"{name}\") is non-orientable"
            )));
        }
        // The smaller class is the flipped set; on a tie the class without
        // the lowest triangle, whose parity is false.
        let n_true = parity.values().filter(|&&p| p).count();
        let n_false = members.len() - n_true;
        let want = n_true <= n_false;
        let flipped: Vec<u32> = members
            .iter()
            .copied()
            .filter(|&t| parity[&t] == want)
            .collect();
        if let Some(&t) = flipped.first() {
            let name = &surf.patch_names[surf.tri_patch[t as usize] as usize];
            return Err(Error::Refused(format!(
                "surface/orientation: {} triangle(s) wound against shell {s}; first: triangle {t} (patch \"{name}\")",
                flipped.len()
            )));
        }
    }
    // Order 5: every shell encloses a positive volume, (116.13a).
    let mut shell_volumes = Vec::with_capacity(shells.len());
    for members in &shells {
        shell_volumes.push(shell_volume(&surf.points, &surf.tris, members));
    }
    for (s, members) in shells.iter().enumerate() {
        if shell_volumes[s] <= 0.0 {
            let lo = members[0];
            let name = &surf.patch_names[surf.tri_patch[lo as usize] as usize];
            return Err(Error::Refused(format!(
                "surface/inward: shell {s} (lowest triangle {lo}, patch \"{name}\") encloses volume {}, which is not positive (116.13a)",
                shell_volumes[s]
            )));
        }
    }
    // Order 6: self-intersection. The broad phase sweeps the f64 boxes on
    // the axis of the surface's largest extent, ties to the lowest axis; a
    // pair is a candidate when the closed boxes overlap on all three axes,
    // which is the brute-force set exactly.
    let mut blo: Vec<Point> = Vec::with_capacity(n);
    let mut bhi: Vec<Point> = Vec::with_capacity(n);
    for t in &tri {
        let mut lo = t[0];
        let mut hi = t[0];
        for p in t {
            for d in 0..3 {
                lo[d] = lo[d].min(p[d]);
                hi[d] = hi[d].max(p[d]);
            }
        }
        blo.push(lo);
        bhi.push(hi);
    }
    let mut ext = [0.0f64; 3];
    for d in 0..3 {
        let mut lo = f64::INFINITY;
        let mut hi = f64::NEG_INFINITY;
        for b in &blo {
            lo = lo.min(b[d]);
        }
        for b in &bhi {
            hi = hi.max(b[d]);
        }
        ext[d] = hi - lo;
    }
    let axis = if ext[0] >= ext[1] && ext[0] >= ext[2] {
        0
    } else if ext[1] >= ext[2] {
        1
    } else {
        2
    };
    let mut order: Vec<u32> = (0..n as u32).collect();
    order.sort_by(|&a, &b| blo[a as usize][axis].total_cmp(&blo[b as usize][axis]));
    let mut tested = 0usize;
    let mut n_vs = 0usize;
    let mut meeting: BTreeSet<(u32, u32)> = BTreeSet::new();
    for ii in 0..n {
        let i = order[ii] as usize;
        for jj in (ii + 1)..n {
            let j = order[jj] as usize;
            if blo[j][axis] > bhi[i][axis] {
                break;
            }
            let mut overlap = true;
            for d in 0..3 {
                if blo[i][d] > bhi[j][d] || blo[j][d] > bhi[i][d] {
                    overlap = false;
                    break;
                }
            }
            if !overlap {
                continue;
            }
            let shared = (0..3)
                .filter(|&k| (0..3).any(|l| surf.tris[i][k] == surf.tris[j][l]))
                .count();
            if shared == 0 {
                tested += 1;
                if tri_tri_intersect(&tri[i], &tri[j]) {
                    meeting.insert((i.min(j) as u32, i.max(j) as u32));
                }
            } else if shared == 1 {
                n_vs += 1;
            }
        }
    }
    // Triangles sharing an edge: folds, by (116.13f).
    for (&(u, v), inc) in &edges {
        if inc.len() != 2 {
            continue;
        }
        let (t1, t2) = (inc[0].0, inc[1].0);
        let a = surf.tris[t1 as usize]
            .iter()
            .copied()
            .find(|&x| x != u && x != v)
            .unwrap();
        let b = surf.tris[t2 as usize]
            .iter()
            .copied()
            .find(|&x| x != u && x != v)
            .unwrap();
        if fold(
            &surf.points[u as usize],
            &surf.points[v as usize],
            &surf.points[a as usize],
            &surf.points[b as usize],
        ) {
            meeting.insert((t1.min(t2), t1.max(t2)));
        }
    }
    if let Some(&(i, j)) = meeting.iter().next() {
        let pi = &surf.patch_names[surf.tri_patch[i as usize] as usize];
        let pj = &surf.patch_names[surf.tri_patch[j as usize] as usize];
        return Err(Error::Refused(format!(
            "surface/self-intersection: {} meeting triangle pair(s); first: triangles {i} and {j} (patches \"{pi}\" and \"{pj}\")",
            meeting.len()
        )));
    }
    // Order 7: features in f64 - (92.34) with the patch seams joined in,
    // then corners by (92.35), then §92.12's chaining.
    let mut normals: Vec<Point> = Vec::with_capacity(n);
    for t in &tri {
        let c = cross3(&sub3(&t[1], &t[0]), &sub3(&t[2], &t[0]));
        let m = dot3(&c, &c).sqrt();
        normals.push([c[0] / m, c[1] / m, c[2] / m]);
    }
    let mut feature_edges: Vec<[u32; 2]> = Vec::new();
    let mut edge_is_seam: Vec<bool> = Vec::new();
    for (&(a, b), inc) in &edges {
        let (i, j) = (inc[0].0 as usize, inc[1].0 as usize);
        let d = (normals[i][0] * normals[j][0]
            + normals[i][1] * normals[j][1]
            + normals[i][2] * normals[j][2])
            .clamp(-1.0, 1.0);
        let seam = surf.tri_patch[i] != surf.tri_patch[j];
        if seam || d.acos().to_degrees() > feature_angle_deg {
            feature_edges.push([a, b]);
            edge_is_seam.push(seam);
        }
    }
    let mut nbrs: HashMap<u32, Vec<u32>> = HashMap::new();
    for &[a, b] in &feature_edges {
        nbrs.entry(a).or_default().push(b);
        nbrs.entry(b).or_default().push(a);
    }
    let mut verts: Vec<u32> = nbrs.keys().copied().collect();
    verts.sort_unstable();
    for list in nbrs.values_mut() {
        list.sort_unstable();
    }
    let mut corners: Vec<u32> = Vec::new();
    for &v in &verts {
        let ns = &nbrs[&v];
        let corner = if ns.len() != 2 {
            true
        } else {
            let xv = &surf.points[v as usize];
            let ua = sub3(&surf.points[ns[0] as usize], xv);
            let ub = sub3(&surf.points[ns[1] as usize], xv);
            let la = dot3(&ua, &ua).sqrt();
            let lb = dot3(&ub, &ub).sqrt();
            if !(la > 0.0) || !(lb > 0.0) {
                true
            } else {
                let d = ((ua[0] * ub[0] + ua[1] * ub[1] + ua[2] * ub[2]) / (la * lb))
                    .clamp(-1.0, 1.0);
                180.0 - d.acos().to_degrees() > feature_angle_deg
            }
        };
        if corner {
            corners.push(v);
        }
    }
    let corner_set: HashSet<u32> = corners.iter().copied().collect();
    let mut edge_at: HashMap<(u32, u32), usize> = HashMap::new();
    for (i, e) in feature_edges.iter().enumerate() {
        edge_at.insert((e[0], e[1]), i);
    }
    let mut consumed = vec![false; feature_edges.len()];
    let mut polylines: Vec<Vec<u32>> = Vec::new();
    for &c in &corners {
        for &nn in &nbrs[&c] {
            let first = edge_at[&edge_key(c, nn)];
            if consumed[first] {
                continue;
            }
            consumed[first] = true;
            let mut chain = vec![c, nn];
            let mut prev = c;
            let mut cur = nn;
            while !corner_set.contains(&cur) {
                if chain.len() > feature_edges.len() {
                    return Err(Error::Mesh(format!(
                        "feature chaining ran off the graph at point {cur}"
                    )));
                }
                let next = nbrs[&cur]
                    .iter()
                    .copied()
                    .find(|&x| x != prev)
                    .expect("a degree-2 interior point has exactly one other neighbour");
                consumed[edge_at[&edge_key(cur, next)]] = true;
                chain.push(next);
                prev = cur;
                cur = next;
            }
            polylines.push(chain);
        }
    }
    // Every edge still unconsumed lies on a corner-free cycle: walk it
    // round, cut it at its lowest point id, close it.
    for e in 0..feature_edges.len() {
        if consumed[e] {
            continue;
        }
        let start = feature_edges[e][0];
        consumed[e] = true;
        let mut chain = vec![start];
        let mut prev = start;
        let mut cur = feature_edges[e][1];
        while cur != start {
            if chain.len() > feature_edges.len() {
                return Err(Error::Mesh(format!(
                    "feature chaining failed to close a cycle at point {cur}"
                )));
            }
            chain.push(cur);
            let next = nbrs[&cur]
                .iter()
                .copied()
                .find(|&x| x != prev)
                .expect("a corner-free cycle turns through degree-2 points only");
            consumed[edge_at[&edge_key(cur, next)]] = true;
            prev = cur;
            cur = next;
        }
        chain.push(start);
        let pos = (0..chain.len() - 1)
            .min_by_key(|&i| chain[i])
            .expect("a cycle has at least one point");
        chain.rotate_left(pos);
        polylines.push(chain);
    }
    polylines.sort_by(|p, q| (p[0], p[1]).cmp(&(q[0], q[1])));
    let n_closed = polylines
        .iter()
        .filter(|p| p.len() >= 2 && p.first() == p.last())
        .count();
    let report = SurfprepReport {
        n_points: surf.points.len(),
        n_tris: n,
        n_shells: shells.len(),
        shell_volumes,
        n_feature_edges: feature_edges.len(),
        n_seam_edges: edge_is_seam.iter().filter(|&&s| s).count(),
        n_polylines: polylines.len(),
        n_closed_polylines: n_closed,
        n_corners: corners.len(),
        n_candidate_pairs: tested,
        n_vertex_sharing_pairs: n_vs,
    };
    Ok(PreparedSurface {
        surface: surf.clone(),
        shell_of_tri,
        feature_edges,
        edge_is_seam,
        polylines,
        corners,
        report,
    })
}

// ==========================================================================
//  Tests - the fixtures and the exact values of §116.9
// ==========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::automesher::features;
    use crate::{Scalar, Vec3};
    use std::collections::HashMap;

    fn pt(x: f64, y: f64, z: f64) -> Point {
        [x, y, z]
    }

    /// The test cube: point `k` takes `hi[q]` on axis `q` when bit `q` of
    /// `k` is set and `lo[q]` otherwise; the six faces, CCW from outside,
    /// split into two triangles each, so triangles 10 and 11 are +z.
    fn cube(lo: [f64; 3], hi: [f64; 3], base: u32) -> (Vec<Point>, Vec<[u32; 3]>) {
        let mut points = Vec::new();
        for k in 0..8u32 {
            let mut p = [0.0; 3];
            for (q, c) in p.iter_mut().enumerate() {
                *c = if k & (1 << q) != 0 { hi[q] } else { lo[q] };
            }
            points.push(p);
        }
        let faces = [
            [0u32, 4, 6, 2],
            [1, 3, 7, 5],
            [0, 1, 5, 4],
            [2, 6, 7, 3],
            [0, 2, 3, 1],
            [4, 5, 7, 6],
        ];
        let mut tris = Vec::new();
        for f in &faces {
            tris.push([f[0] + base, f[1] + base, f[2] + base]);
            tris.push([f[0] + base, f[2] + base, f[3] + base]);
        }
        (points, tris)
    }

    /// The icosphere: the 12 normalised icosahedron points; each level
    /// replaces every triangle in order by four, the edge midpoints created
    /// once per undirected edge as `normalise((pa + pb) * 0.5)`, appended.
    fn icosphere(level: u32) -> (Vec<Point>, Vec<[u32; 3]>) {
        let phi = (1.0 + 5.0f64.sqrt()) / 2.0;
        let raw = [
            [-1.0, phi, 0.0], [1.0, phi, 0.0], [-1.0, -phi, 0.0], [1.0, -phi, 0.0],
            [0.0, -1.0, phi], [0.0, 1.0, phi], [0.0, -1.0, -phi], [0.0, 1.0, -phi],
            [phi, 0.0, -1.0], [phi, 0.0, 1.0], [-phi, 0.0, -1.0], [-phi, 0.0, 1.0],
        ];
        let mut points: Vec<Point> = raw
            .iter()
            .map(|p| {
                let m = (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt();
                [p[0] / m, p[1] / m, p[2] / m]
            })
            .collect();
        let mut faces: Vec<[u32; 3]> = vec![
            [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
            [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
            [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
            [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1],
        ];
        fn midpoint(
            points: &mut Vec<Point>,
            mid: &mut HashMap<(u32, u32), u32>,
            a: u32,
            b: u32,
        ) -> u32 {
            let key = if a < b { (a, b) } else { (b, a) };
            if let Some(&m) = mid.get(&key) {
                return m;
            }
            let pa = points[a as usize];
            let pb = points[b as usize];
            let p = [
                (pa[0] + pb[0]) * 0.5,
                (pa[1] + pb[1]) * 0.5,
                (pa[2] + pb[2]) * 0.5,
            ];
            let m = (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt();
            let id = points.len() as u32;
            points.push([p[0] / m, p[1] / m, p[2] / m]);
            mid.insert(key, id);
            id
        }
        for _ in 0..level {
            let mut mid: HashMap<(u32, u32), u32> = HashMap::new();
            let mut next = Vec::with_capacity(faces.len() * 4);
            for [a, b, c] in faces.drain(..) {
                let ab = midpoint(&mut points, &mut mid, a, b);
                let bc = midpoint(&mut points, &mut mid, b, c);
                let ca = midpoint(&mut points, &mut mid, c, a);
                next.push([a, ab, ca]);
                next.push([b, bc, ab]);
                next.push([c, ca, bc]);
                next.push([ab, bc, ca]);
            }
            faces = next;
        }
        (points, faces)
    }

    /// The capped 16-sided cylinder: two apex points, three rings, the
    /// bottom and top fans first, then the two side bands.
    fn cylinder() -> (Vec<Point>, Vec<[u32; 3]>) {
        let mut points = vec![pt(0.0, 0.0, -1.0), pt(0.0, 0.0, 1.0)];
        for r in 0..3u32 {
            let z = match r {
                0 => -1.0,
                1 => 0.0,
                _ => 1.0,
            };
            for k in 0..16u32 {
                let t = 2.0 * std::f64::consts::PI * k as f64 / 16.0;
                points.push([t.cos(), t.sin(), z]);
            }
        }
        let ring = |r: u32, k: u32| 2 + 16 * r + (k % 16);
        let mut tris = Vec::new();
        for k in 0..16u32 {
            tris.push([0, ring(0, k + 1), ring(0, k)]);
        }
        for k in 0..16u32 {
            tris.push([1, ring(2, k), ring(2, k + 1)]);
        }
        for r in 0..2u32 {
            for k in 0..16u32 {
                let (a, b, c, d) = (ring(r, k), ring(r, k + 1), ring(r + 1, k + 1), ring(r + 1, k));
                tris.push([a, b, c]);
                tris.push([a, c, d]);
            }
        }
        (points, tris)
    }

    /// The real projective plane's 6-point triangulation: non-orientable.
    fn rp2() -> (Vec<Point>, Vec<[u32; 3]>) {
        let points: Vec<Point> = (0..6u32)
            .map(|i| {
                let f = i as f64;
                [f, f * f, f * f * f]
            })
            .collect();
        let tris = vec![
            [0u32, 1, 2],
            [0, 2, 3],
            [0, 3, 4],
            [0, 4, 5],
            [0, 5, 1],
            [1, 2, 4],
            [2, 3, 5],
            [3, 4, 1],
            [4, 5, 2],
            [5, 1, 3],
        ];
        (points, tris)
    }

    /// A `TetSurface` from fixture parts, with the given patch ids and names.
    fn tet(
        points: Vec<Point>,
        tris: Vec<[u32; 3]>,
        patches: &[u32],
        names: &[&str],
    ) -> TetSurface {
        TetSurface {
            points,
            tris,
            tri_patch: patches.to_vec(),
            patch_names: names.iter().map(|s| s.to_string()).collect(),
        }
    }

    /// A reader's `Surface` from fixture parts, with dummy derived fields;
    /// [`Surface::merge`] rebuilds everything and renumbers by first
    /// appearance, so compare against `from_surface` of the merged one.
    fn to_surface(
        points: &[Point],
        tris: &[[u32; 3]],
        patches: &[u32],
        names: &[&str],
    ) -> Surface {
        let raw = Surface {
            points: points
                .iter()
                .map(|p| Vec3::new(p[0] as Scalar, p[1] as Scalar, p[2] as Scalar))
                .collect(),
            tris: tris.to_vec(),
            tri_patch: patches.to_vec(),
            patch_names: names.iter().map(|s| s.to_string()).collect(),
            normals: vec![Vec3::ZERO; tris.len()],
            bbox: (Vec3::ZERO, Vec3::ZERO),
            patch_area: vec![0.0; names.len()],
            degenerate_dropped: 0,
        };
        Surface::merge(vec![raw]).unwrap()
    }

    /// The brute-force counts: closed box overlap for every i < j, then the
    /// shared-point classification of §116.9's order 6.
    fn brute_pairs(ts: &TetSurface) -> (usize, usize) {
        let boxes: Vec<([f64; 3], [f64; 3])> = ts
            .tris
            .iter()
            .map(|t| {
                let mut lo = ts.points[t[0] as usize];
                let mut hi = lo;
                for &i in t {
                    let p = ts.points[i as usize];
                    for d in 0..3 {
                        lo[d] = lo[d].min(p[d]);
                        hi[d] = hi[d].max(p[d]);
                    }
                }
                (lo, hi)
            })
            .collect();
        let mut tested = 0;
        let mut vs = 0;
        for i in 0..ts.tris.len() {
            for j in (i + 1)..ts.tris.len() {
                let mut touch = true;
                for d in 0..3 {
                    if boxes[i].0[d] > boxes[j].1[d] || boxes[j].0[d] > boxes[i].1[d] {
                        touch = false;
                        break;
                    }
                }
                if !touch {
                    continue;
                }
                let shared = (0..3)
                    .filter(|&k| (0..3).any(|l| ts.tris[i][k] == ts.tris[j][l]))
                    .count();
                if shared == 0 {
                    tested += 1;
                }
                if shared == 1 {
                    vs += 1;
                }
            }
        }
        (tested, vs)
    }

    /// `prepare` is `Err(Error::Refused(m))` and `m` carries every text.
    fn refused(p: &TetSurface, angle: f64, texts: &[&str]) {
        match prepare(p, angle) {
            Err(Error::Refused(m)) => {
                for t in texts {
                    assert!(m.contains(t), "refusal lacks {t:?}: {m}");
                }
            }
            other => panic!("expected a refusal of {texts:?}, got {other:?}"),
        }
    }

    #[test]
    fn refuses_open() {
        let (pts, mut tris) = cube([0.0, 0.0, 0.0], [1.0, 1.0, 1.0], 0);
        tris.truncate(10);
        let s = tet(pts, tris, &[0; 10], &["box"]);
        refused(
            &s,
            30.0,
            &[
                "surface/closed",
                "require_closed",
                "4 open edge(s), 0 non-manifold edge(s)",
                "edge (4, 5) of triangle 5",
                "patch \"box\"",
            ],
        );
        let (pts, mut tris) = cube([0.0, 0.0, 0.0], [1.0, 1.0, 1.0], 0);
        tris.push([0, 1, 7]);
        let s = tet(pts, tris, &[0; 13], &["box"]);
        refused(
            &s,
            30.0,
            &[
                "surface/closed",
                "1 open edge(s), 2 non-manifold edge(s)",
                "edge (0, 1) of triangle 4",
            ],
        );
        // Through a Surface, whose merge renumbers by first appearance.
        let (pts, tris) = cube([0.0, 0.0, 0.0], [1.0, 1.0, 1.0], 0);
        let mut open = tris.clone();
        open.truncate(10);
        let so = to_surface(&pts, &open, &[0; 10], &["box"]);
        assert_eq!(so.edge_defects(), (4, 0));
        refused(
            &TetSurface::from_surface(&so),
            30.0,
            &["4 open edge(s), 0 non-manifold"],
        );
        let mut with13 = tris.clone();
        with13.push([0, 1, 7]);
        let s13 = to_surface(&pts, &with13, &[0; 13], &["box"]);
        assert_eq!(s13.edge_defects(), (1, 2));
        refused(
            &TetSurface::from_surface(&s13),
            30.0,
            &["1 open edge(s), 2 non-manifold"],
        );
    }

    #[test]
    fn refuses_flipped() {
        let (pts, mut tris) = cube([0.0, 0.0, 0.0], [1.0, 1.0, 1.0], 0);
        tris[5].reverse();
        let s = tet(pts.clone(), tris.clone(), &[0; 12], &["box"]);
        refused(
            &s,
            30.0,
            &[
                "surface/orientation",
                "1 triangle(s) wound against shell 0",
                "triangle 5",
            ],
        );
        let sf = to_surface(&pts, &tris, &[0; 12], &["box"]);
        assert_eq!(sf.edge_defects(), (0, 3));
        refused(
            &TetSurface::from_surface(&sf),
            30.0,
            &["surface/orientation"],
        );
        let (pts, mut tris) = icosphere(3);
        tris[37].swap(1, 2);
        let s = tet(pts, tris, &[0; 1280], &["sphere"]);
        refused(
            &s,
            30.0,
            &[
                "1 triangle(s) wound against shell 0",
                "triangle 37",
            ],
        );
        let (pts, mut tris) = cube([0.0, 0.0, 0.0], [1.0, 1.0, 1.0], 0);
        for t in tris.iter_mut() {
            t.reverse();
        }
        let s = tet(pts, tris, &[0; 12], &["box"]);
        refused(&s, 30.0, &["surface/inward", "shell 0", "lowest triangle 0"]);
        let (pts, tris) = rp2();
        let s = tet(pts, tris, &[0; 10], &["rp2"]);
        refused(
            &s,
            30.0,
            &["surface/orientation", "shell 0", "non-orientable"],
        );
    }

    /// Two bodies, the first patch 0 "a", the second patch 1 "b".
    fn two_cubes(lo2: [f64; 3], hi2: [f64; 3]) -> TetSurface {
        let mut points = Vec::new();
        let mut tris = Vec::new();
        let mut patches = Vec::new();
        let (p1, t1) = cube([0.0, 0.0, 0.0], [1.0, 1.0, 1.0], 0);
        points.extend(p1);
        tris.extend(t1);
        patches.extend(std::iter::repeat(0u32).take(12));
        let (p2, t2) = cube(lo2, hi2, 8);
        points.extend(p2);
        tris.extend(t2);
        patches.extend(std::iter::repeat(1u32).take(12));
        tet(points, tris, &patches, &["a", "b"])
    }

    #[test]
    fn refuses_self_intersection() {
        let s = two_cubes([0.5, 0.25, 0.25], [1.5, 0.75, 0.75]);
        refused(
            &s,
            30.0,
            &[
                "surface/self-intersection",
                "12 meeting triangle pair(s)",
                "triangles 2 and 16",
                "patches \"a\" and \"b\"",
            ],
        );
        let s = two_cubes([1.0, 0.25, 0.25], [2.0, 0.75, 0.75]);
        refused(
            &s,
            30.0,
            &[
                "surface/self-intersection",
                "18 meeting triangle pair(s)",
                "triangles 2 and 12",
            ],
        );
    }

    /// The cube's feature answer, shared by the one-patch and six-patch
    /// fixtures of [`cube_features`].
    fn cube_want() -> (Vec<[u32; 2]>, Vec<Vec<u32>>) {
        let edges = vec![
            [0u32, 1],
            [0, 2],
            [0, 4],
            [1, 3],
            [1, 5],
            [2, 3],
            [2, 6],
            [3, 7],
            [4, 5],
            [4, 6],
            [5, 7],
            [6, 7],
        ];
        let polys: Vec<Vec<u32>> = edges.iter().map(|e| e.to_vec()).collect();
        (edges, polys)
    }

    #[test]
    fn cube_features() {
        let (want_edges, want_polys) = cube_want();
        let (pts, tris) = cube([0.0, 0.0, 0.0], [1.0, 1.0, 1.0], 0);
        let s = tet(pts, tris, &[0; 12], &["box"]);
        let p = prepare(&s, 30.0).unwrap();
        assert_eq!(p.feature_edges, want_edges);
        assert_eq!(p.polylines, want_polys);
        assert_eq!(p.corners, vec![0, 1, 2, 3, 4, 5, 6, 7]);
        assert!(p.edge_is_seam.iter().all(|&s| !s));
        let r = &p.report;
        assert_eq!(r.n_shells, 1);
        assert_eq!(r.shell_volumes[0].to_bits(), 1.0f64.to_bits());
        assert_eq!(r.n_feature_edges, 12);
        assert_eq!(r.n_seam_edges, 0);
        assert_eq!(r.n_polylines, 12);
        assert_eq!(r.n_closed_polylines, 0);
        assert_eq!(r.n_corners, 8);
        assert_eq!(r.n_candidate_pairs, 6);
        assert_eq!(r.n_vertex_sharing_pairs, 30);
        print!("TET05-CUBE {}", r.summary());
        // cube6: the same edges, polylines and corners, every edge a seam.
        let (want_edges, want_polys) = cube_want();
        let (pts, tris) = cube([0.0, 0.0, 0.0], [1.0, 1.0, 1.0], 0);
        let patches: Vec<u32> = (0..12u32).map(|t| t / 2).collect();
        let s = tet(
            pts,
            tris,
            &patches,
            &["xmin", "xmax", "ymin", "ymax", "zmin", "zmax"],
        );
        let p = prepare(&s, 30.0).unwrap();
        assert_eq!(p.feature_edges, want_edges);
        assert_eq!(p.polylines, want_polys);
        assert_eq!(p.corners, vec![0, 1, 2, 3, 4, 5, 6, 7]);
        assert!(p.edge_is_seam.iter().all(|&s| s));
        assert_eq!(p.report.n_seam_edges, 12);
    }

    #[test]
    fn cylinder_features() {
        let (pts, tris) = cylinder();
        let s = tet(pts.clone(), tris.clone(), &[0; 96], &["cyl"]);
        let p = prepare(&s, 30.0).unwrap();
        assert_eq!(p.report.n_feature_edges, 32);
        assert_eq!(p.report.n_seam_edges, 0);
        let rim_lo: Vec<u32> = (0..=16u32).map(|k| 2 + (k % 16)).collect();
        let rim_hi: Vec<u32> = (0..=16u32).map(|k| 34 + (k % 16)).collect();
        assert_eq!(p.polylines, vec![rim_lo.clone(), rim_hi.clone()]);
        assert!(p.corners.is_empty());
        let want = 6.1229349178414365f64;
        assert!((p.report.shell_volumes[0] - want).abs() <= want * 1e-12);
        let (cand, vs) = brute_pairs(&s);
        assert_eq!(p.report.n_candidate_pairs, cand);
        assert_eq!(p.report.n_vertex_sharing_pairs, vs);
        // cylinder2: the side split at mid-height, patch 1 above.
        let mut patches = vec![0u32; 96];
        for t in 16..32 {
            patches[t] = 1;
        }
        for t in 64..96 {
            patches[t] = 1;
        }
        let s2 = tet(pts, tris, &patches, &["lower", "upper"]);
        let p2 = prepare(&s2, 30.0).unwrap();
        assert_eq!(p2.report.n_feature_edges, 48);
        assert_eq!(p2.report.n_seam_edges, 16);
        let rim_mid: Vec<u32> = (0..=16u32).map(|k| 18 + (k % 16)).collect();
        assert_eq!(p2.polylines, vec![rim_lo, rim_mid, rim_hi]);
        assert!(p2.corners.is_empty());
        assert_eq!(p2.report.n_closed_polylines, 3);
        let (cand, vs) = brute_pairs(&s2);
        assert_eq!(p2.report.n_candidate_pairs, cand);
        assert_eq!(p2.report.n_vertex_sharing_pairs, vs);
        println!("TET05-CYL cand={cand} vs={vs}");
    }

    #[test]
    fn sphere_passes() {
        let (pts, tris) = icosphere(3);
        assert_eq!(pts.len(), 642);
        assert_eq!(tris.len(), 1280);
        let s = tet(pts, tris, &[0; 1280], &["sphere"]);
        let p = prepare(&s, 30.0).unwrap();
        assert_eq!(p.report.n_points, 642);
        assert_eq!(p.report.n_tris, 1280);
        assert_eq!(p.report.n_shells, 1);
        let want = 4.152740817093058f64;
        assert!((p.report.shell_volumes[0] - want).abs() <= want * 1e-12);
        assert_eq!(p.report.n_feature_edges, 0);
        assert_eq!(p.report.n_corners, 0);
        assert_eq!(p.report.n_candidate_pairs, 216);
        assert_eq!(p.report.n_vertex_sharing_pairs, 5730);
        // Level 5: the pair counts against the brute force, wall time
        // recorded but informative only.
        let (pts, tris) = icosphere(5);
        assert_eq!(pts.len(), 10242);
        assert_eq!(tris.len(), 20480);
        let patches = vec![0u32; 20480];
        let s = tet(pts, tris, &patches, &["sphere"]);
        let t0 = std::time::Instant::now();
        let p = prepare(&s, 30.0).unwrap();
        let ms = t0.elapsed().as_millis();
        assert_eq!(p.report.n_shells, 1);
        assert_eq!(p.report.n_feature_edges, 0);
        assert_eq!(p.report.n_corners, 0);
        let (cand, vs) = brute_pairs(&s);
        assert_eq!(p.report.n_candidate_pairs, cand);
        assert_eq!(p.report.n_vertex_sharing_pairs, vs);
        println!("TET05-SPHERE5 tris=20480 cand={cand} vs={vs} prepare_ms={ms}");
    }

    #[test]
    fn disjoint_boxes_pass() {
        let mut points = Vec::new();
        let mut tris = Vec::new();
        let (p1, t1) = cube([0.0, 0.0, 0.0], [1.0, 1.0, 1.0], 0);
        let (p2, t2) = cube([2.0, 0.0, 0.0], [3.0, 1.0, 1.0], 8);
        points.extend(p1);
        points.extend(p2);
        tris.extend(t1);
        tris.extend(t2);
        let s = tet(points, tris, &[0; 24], &["box"]);
        let p = prepare(&s, 30.0).unwrap();
        assert_eq!(p.report.n_shells, 2);
        assert_eq!(p.report.shell_volumes, vec![1.0, 1.0]);
        assert!(p.shell_of_tri.iter().take(12).all(|&x| x == 0));
        assert!(p.shell_of_tri.iter().skip(12).all(|&x| x == 1));
        assert_eq!(p.report.n_feature_edges, 24);
        assert_eq!(p.report.n_polylines, 24);
        assert_eq!(p.report.n_corners, 16);
        assert_eq!(p.report.n_candidate_pairs, 12);
        assert_eq!(p.report.n_vertex_sharing_pairs, 60);
    }

    #[test]
    fn tri_tri_random() {
        struct SplitMix64 {
            state: u64,
        }
        impl SplitMix64 {
            fn new(seed: u64) -> Self {
                SplitMix64 { state: seed }
            }
            fn next(&mut self) -> u64 {
                self.state = self.state.wrapping_add(0x9E3779B97F4A7C15);
                let mut z = self.state;
                z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
                z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
                z ^ (z >> 31)
            }
            fn range(&mut self, lo: i64, hi: i64) -> i64 {
                lo + (self.next() % ((hi - lo) as u64)) as i64
            }
        }

        fn draw_pair(
            r: &mut SplitMix64,
            lo: i64,
            hi: i64,
            coplanar: bool,
            scaled: bool,
        ) -> ([Point; 3], [Point; 3]) {
            let two30 = (1u64 << 30) as f64;
            let map = |c: f64| if scaled { c / two30 + 1024.0 } else { c };
            let mut t1 = [[0.0; 3]; 3];
            let mut t2 = [[0.0; 3]; 3];
            if coplanar {
                for t in [&mut t1, &mut t2] {
                    for v in t.iter_mut() {
                        let xi = r.range(lo, hi);
                        let yi = r.range(lo, hi);
                        let zi = xi + yi;
                        *v = [map(xi as f64), map(yi as f64), map(zi as f64)];
                    }
                }
            } else {
                for t in [&mut t1, &mut t2] {
                    for v in t.iter_mut() {
                        for c in v.iter_mut() {
                            *c = map(r.range(lo, hi) as f64);
                        }
                    }
                }
            }
            (t1, t2)
        }

        fn count_stream(
            seed: u64,
            lo: i64,
            hi: i64,
            coplanar: bool,
            scaled: bool,
        ) -> (usize, usize) {
            let mut r = SplitMix64::new(seed);
            let mut degen = 0usize;
            let mut inter = 0usize;
            for _ in 0..100_000 {
                let (t1, t2) = draw_pair(&mut r, lo, hi, coplanar, scaled);
                if is_degenerate(&t1) || is_degenerate(&t2) {
                    degen += 1;
                } else if tri_tri_intersect(&t1, &t2) {
                    inter += 1;
                }
            }
            (degen, inter)
        }
        let (da, ia) = count_stream(0x7E705A, -4, 4, false, false);
        assert_eq!((da, ia), (1505, 29874));
        let (db, ib) = count_stream(0x7E705B, -2, 2, false, false);
        assert_eq!((db, ib), (10623, 38875));
        let (dc, ic) = count_stream(0x7E705C, -4, 4, true, false);
        assert_eq!((dc, ic), (15835, 60197));
        // Invariance on stream A: symmetry, a rotation of t1, a reversal
        // of t2 - the answers never move.
        let mut r = SplitMix64::new(0x7E705A);
        for _ in 0..100_000 {
            let (t1, t2) = draw_pair(&mut r, -4, 4, false, false);
            if is_degenerate(&t1) || is_degenerate(&t2) {
                continue;
            }
            let want = tri_tri_intersect(&t1, &t2);
            assert_eq!(tri_tri_intersect(&t2, &t1), want);
            let rot = [t1[1], t1[2], t1[0]];
            assert_eq!(tri_tri_intersect(&rot, &t2), want);
            let rev = [t2[0], t2[2], t2[1]];
            assert_eq!(tri_tri_intersect(&t1, &rev), want);
        }
        // The same streams with every coordinate mapped onto a large
        // offset: the exact domain is scale-free, so the counts repeat.
        let (das, ias) = count_stream(0x7E705A, -4, 4, false, true);
        assert_eq!((das, ias), (da, ia));
        let (dcs, ics) = count_stream(0x7E705C, -4, 4, true, true);
        assert_eq!((dcs, ics), (dc, ic));
        println!("TET05-RANDOM a={ia} b={ib} c={ic} a_scaled={ias} c_scaled={ics}");
    }

    #[test]
    fn pair_predicate_cases() {
        let t: [Point; 3] = [pt(0.0, 0.0, 0.0), pt(1.0, 0.0, 0.0), pt(0.0, 1.0, 0.0)];
        assert_eq!(triangle_apex(&t), [0.0, 0.0, 1.4142135623730951]);
        assert!(is_degenerate(&[pt(0.0, 0.0, 0.0), pt(1.0, 0.0, 0.0), pt(2.0, 0.0, 0.0)]));
        assert!(is_degenerate(&[pt(1.0, 1.0, 1.0), pt(1.0, 1.0, 1.0), pt(2.0, 0.0, 0.0)]));
        assert!(!is_degenerate(&t));
        assert!(segment_meets_triangle(&pt(0.2, 0.2, -1.0), &pt(0.2, 0.2, 1.0), &t));
        assert!(!segment_meets_triangle(&pt(2.0, 2.0, -1.0), &pt(2.0, 2.0, 1.0), &t));
        assert!(segment_meets_triangle(&pt(0.0, 0.0, -1.0), &pt(0.0, 0.0, 1.0), &t));
        assert!(segment_meets_triangle(&pt(0.5, 0.0, -1.0), &pt(0.5, 0.0, 1.0), &t));
        assert!(segment_meets_triangle(&pt(0.2, 0.2, 0.0), &pt(0.2, 0.2, 1.0), &t));
        assert!(!segment_meets_triangle(&pt(0.2, 0.2, 0.5), &pt(0.2, 0.2, 1.0), &t));
        assert!(!segment_meets_triangle(&pt(0.1, 0.1, 0.0), &pt(0.3, 0.3, 0.0), &t));
        assert!(tri_tri_intersect(
            &t,
            &[pt(0.0, 0.0, 0.0), pt(0.0, 0.0, 1.0), pt(-1.0, -1.0, 1.0)]
        ));
        assert!(tri_tri_intersect(
            &t,
            &[pt(0.5, -1.0, 0.0), pt(0.5, 0.0, 0.0), pt(0.5, -1.0, 1.0)]
        ));
        assert!(tri_tri_intersect(
            &t,
            &[pt(0.25, 0.25, -1.0), pt(0.25, 0.25, 1.0), pt(2.0, 2.0, 0.5)]
        ));
        assert!(!tri_tri_intersect(
            &t,
            &[pt(0.0, 0.0, 1.0), pt(1.0, 0.0, 1.0), pt(0.0, 1.0, 1.0)]
        ));
        // The coplanar meets, the folds, and the shell volume.
        assert!(tri_tri_intersect(
            &t,
            &[pt(0.25, 0.25, 0.0), pt(2.0, 0.25, 0.0), pt(0.25, 2.0, 0.0)]
        ));
        assert!(!tri_tri_intersect(
            &t,
            &[pt(1.0, 1.0, 0.0), pt(2.0, 1.0, 0.0), pt(1.0, 2.0, 0.0)]
        ));
        assert!(tri_tri_intersect(
            &t,
            &[pt(1.0, 0.0, 0.0), pt(0.0, 1.0, 0.0), pt(1.0, 1.0, 0.0)]
        ));
        assert!(tri_tri_intersect(
            &t,
            &[pt(0.1, 0.1, 0.0), pt(0.2, 0.1, 0.0), pt(0.1, 0.2, 0.0)]
        ));
        assert!(!tri_tri_intersect(
            &t,
            &[pt(0.5, 0.5000001, -1.0), pt(0.5, 0.5000001, 1.0), pt(3.0, 3.0, 0.0)]
        ));
        assert!(fold(
            &pt(0.0, 0.0, 0.0),
            &pt(1.0, 0.0, 0.0),
            &pt(0.0, 1.0, 0.0),
            &pt(0.5, 0.5, 0.0)
        ));
        assert!(!fold(
            &pt(0.0, 0.0, 0.0),
            &pt(1.0, 0.0, 0.0),
            &pt(0.0, 1.0, 0.0),
            &pt(0.5, -0.5, 0.0)
        ));
        assert!(!fold(
            &pt(0.0, 0.0, 0.0),
            &pt(1.0, 0.0, 0.0),
            &pt(0.0, 1.0, 0.0),
            &pt(0.5, 0.5, 1.0)
        ));
        let (pts, tris) = cube([0.0, 0.0, 0.0], [1.0, 1.0, 1.0], 0);
        let members: Vec<u32> = (0..12).collect();
        assert_eq!(shell_volume(&pts, &tris, &members), 1.0);
    }

    #[test]
    fn refuses_input() {
        let (pts, tris) = cube([0.0, 0.0, 0.0], [1.0, 1.0, 1.0], 0);
        let base = tet(pts.clone(), tris.clone(), &[0; 12], &["box"]);
        for angle in [0.0f64, 180.0, f64::NAN] {
            refused(&base, angle, &["surface/feature_angle_deg"]);
        }
        refused(
            &tet(pts.clone(), Vec::new(), &[], &["box"]),
            30.0,
            &["surface/input", "no triangles"],
        );
        refused(
            &tet(pts.clone(), tris.clone(), &[], &[]),
            30.0,
            &["surface/input", "no patches"],
        );
        let mut tp = vec![0u32; 12];
        tp.pop();
        refused(
            &tet(pts.clone(), tris.clone(), &tp, &["box"]),
            30.0,
            &["surface/input", "11 entries for 12"],
        );
        let mut t99 = tris.clone();
        t99[0] = [0, 4, 99];
        refused(
            &tet(pts.clone(), t99, &[0; 12], &["box"]),
            30.0,
            &["triangle 0 references point 99"],
        );
        let mut tp5 = vec![0u32; 12];
        tp5[3] = 5;
        refused(
            &tet(pts.clone(), tris.clone(), &tp5, &["box"]),
            30.0,
            &["triangle 3 has patch id 5"],
        );
        let mut trep = tris.clone();
        trep[0] = [0, 0, 6];
        refused(
            &tet(pts.clone(), trep, &[0; 12], &["box"]),
            30.0,
            &["triangle 0 repeats point 0"],
        );
        let mut pnan = pts.clone();
        pnan[3] = [1.0, f64::NAN, 0.0];
        refused(
            &tet(pnan, tris.clone(), &[0; 12], &["box"]),
            30.0,
            &["point 3", "exact domain"],
        );
        let mut pbig = pts.clone();
        pbig[2] = [0.0, 1e200, 0.0];
        refused(
            &tet(pbig, tris.clone(), &[0; 12], &["box"]),
            30.0,
            &["point 2"],
        );
        // The degenerate check comes before the closed one.
        let mut p9 = pts.clone();
        p9.push([2.0, 0.0, 0.0]);
        let mut t13 = tris.clone();
        t13.push([0, 1, 8]);
        refused(
            &tet(p9, t13, &[0; 13], &["box"]),
            30.0,
            &[
                "surface/degenerate",
                "1 degenerate triangle(s)",
                "triangle 12",
            ],
        );
    }

    #[test]
    fn matches_hex_features() {
        for (points, tris) in [
            cube([0.0, 0.0, 0.0], [1.0, 1.0, 1.0], 0),
            cylinder(),
            icosphere(3),
        ] {
            let patches = vec![0u32; tris.len()];
            let s = to_surface(&points, &tris, &patches, &["one"]);
            let ts = TetSurface::from_surface(&s);
            assert_eq!(ts.points.len(), s.points.len());
            for (i, p) in s.points.iter().enumerate() {
                assert_eq!(ts.points[i], [p.x as f64, p.y as f64, p.z as f64]);
            }
            assert_eq!(ts.tris, s.tris);
            assert_eq!(ts.tri_patch, s.tri_patch);
            assert_eq!(ts.patch_names, s.patch_names);
            let prep = prepare(&ts, 30.0).unwrap();
            let hex = features::extract(&s, 30.0).unwrap();
            assert_eq!(prep.feature_edges, hex.edges);
            assert_eq!(prep.polylines, hex.polylines);
            assert_eq!(prep.corners, hex.corners);
        }
    }
}
