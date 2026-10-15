// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.
// Provenance: see PROVENANCE.md. No GPL-licensed source was consulted.

//! Prism layers by post-inflation, SPEC-LIT §116.15: the wall point
//! normals and the per-point layer thicknesses (TET-13a).
//!
//! The wall is every triangle of the surface whose patch carries layers.
//! At each wall point the inflation direction is the "most normal" normal
//! (116.22) of the incident wall triangles' unit normals: the unit vector
//! that maximises the smallest cosine with them. It is the centre of the
//! smallest spherical cap that holds the face normals, so its support is
//! one, two or three of them (116.22a), and [`most_normal`] takes the best
//! of those candidates with the objective evaluated directly. No tolerance
//! decides which candidate is valid.
//!
//! The total thickness of the stack (116.23) is then limited at each wall
//! point (116.24): by the gap along the normal to the first triangle it
//! meets (116.24a), by the radius of curvature of a concave wall (116.24b),
//! and by the ratio to its wall neighbours (116.24c). Every limit only
//! lowers a thickness. The geometry is f64 in both builds (§116.2).
//!
//! Provenance: ORIGINAL. Written from Aubry & Löhner (2008), "On the 'most
//! normal' normal" (the definition (116.22) only; its solution by the
//! support set is this module's), and SPEC-LIT §116.8's curvature (116.9b)
//! and gap (116.10a) estimates. No GPL-licensed source was consulted.

use super::sizefield::GapScanner;
use super::surfprep::TetSurface;
use super::{Point, TetLayerSpec};
use crate::error::{Error, Result};
use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap};

fn dot(a: Point, b: Point) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn sub(a: Point, b: Point) -> Point {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn add(a: Point, b: Point) -> Point {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
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

/// `a / |a|`, or None for the zero or a non-finite vector.
fn unit(a: Point) -> Option<Point> {
    let n = norm(a);
    if n > 0.0 && n.is_finite() {
        Some([a[0] / n, a[1] / n, a[2] / n])
    } else {
        None
    }
}

/// The angle between two vectors, `atan2(|a x b|, a . b)` in `[0, pi]`,
/// accurate near 0 where `acos` is not.
pub fn angle(a: Point, b: Point) -> f64 {
    norm(cross(a, b)).atan2(dot(a, b))
}

/// The objective of (116.22): the smallest cosine of `n` with the normals.
fn min_cos(n: Point, normals: &[Point]) -> f64 {
    normals.iter().map(|&m| dot(n, m)).fold(f64::INFINITY, f64::min)
}

/// The most normal normal (116.22) of unit face normals: the unit `n*`
/// that maximises `min_i n . n_i`, and that smallest cosine.
///
/// The optimum is the centre of the smallest spherical cap that holds the
/// `n_i`, fixed by one, two or three of them (116.22a): `n_i` itself,
/// `(n_i + n_j) / |n_i + n_j|`, and the direction equidistant from three,
/// `(n_j - n_i) x (n_k - n_i)` oriented to `n_i`'s side. Every candidate's
/// objective is evaluated over all the normals, so the largest is the
/// optimum; ties keep the first candidate in that order. Normals equal
/// bit for bit are taken once. None when there is no normal, or no
/// direction has a positive cosine with every one (no visible side, which
/// TET-13c's taper handles). The cost is cubic in the distinct normals.
pub fn most_normal(normals: &[Point]) -> Option<(Point, f64)> {
    let mut ns: Vec<Point> = Vec::with_capacity(normals.len());
    for &n in normals {
        if !ns.contains(&n) {
            ns.push(n);
        }
    }
    let m = ns.len();
    let mut best: Option<(Point, f64)> = None;
    let mut consider = |c: Point| {
        let f = min_cos(c, &ns);
        if best.map_or(true, |(_, b)| f > b) {
            best = Some((c, f));
        }
    };
    for i in 0..m {
        if let Some(c) = unit(ns[i]) {
            consider(c);
        }
    }
    for i in 0..m {
        for j in (i + 1)..m {
            if let Some(c) = unit(add(ns[i], ns[j])) {
                consider(c);
            }
        }
    }
    for i in 0..m {
        for j in (i + 1)..m {
            for k in (j + 1)..m {
                let s = cross(sub(ns[j], ns[i]), sub(ns[k], ns[i]));
                let side = dot(s, ns[i]);
                if side == 0.0 || !side.is_finite() {
                    continue;
                }
                let s = if side < 0.0 { [-s[0], -s[1], -s[2]] } else { s };
                if let Some(c) = unit(s) {
                    consider(c);
                }
            }
        }
    }
    best.filter(|&(_, f)| f > 0.0)
}

/// The checks every entry point runs: the patch list matches the
/// triangles, the layered mask matches the patches, every triangle names
/// existing points and an existing patch, and every point is finite.
fn validate(surf: &TetSurface, layered: &[bool]) -> Result<()> {
    if surf.tri_patch.len() != surf.tris.len() {
        return Err(Error::Mesh(format!(
            "inflate: tri_patch has {} entries for {} triangles",
            surf.tri_patch.len(),
            surf.tris.len()
        )));
    }
    if layered.len() != surf.patch_names.len() {
        return Err(Error::Mesh(format!(
            "inflate: the layered mask has {} entries for {} patches",
            layered.len(),
            surf.patch_names.len()
        )));
    }
    for (t, tri) in surf.tris.iter().enumerate() {
        for &i in tri {
            if i as usize >= surf.points.len() {
                return Err(Error::Mesh(format!(
                    "inflate: triangle {t} references point {i}, but there are {} points",
                    surf.points.len()
                )));
            }
        }
        if surf.tri_patch[t] as usize >= surf.patch_names.len() {
            return Err(Error::Mesh(format!(
                "inflate: triangle {t} has patch id {}, but there are {} patches",
                surf.tri_patch[t],
                surf.patch_names.len()
            )));
        }
    }
    for (i, p) in surf.points.iter().enumerate() {
        if !p.iter().all(|v| v.is_finite()) {
            return Err(Error::Mesh(format!(
                "inflate: point {i} has a coordinate that is not finite"
            )));
        }
    }
    Ok(())
}

/// Triangle `t`'s three points.
fn tri_pts(surf: &TetSurface, t: usize) -> [Point; 3] {
    let [a, b, c] = surf.tris[t];
    [
        surf.points[a as usize],
        surf.points[b as usize],
        surf.points[c as usize],
    ]
}

/// Triangle `t`'s unit normal `(x1 - x0) x (x2 - x0) / |..|` and its area;
/// None for a zero-area triangle.
fn tri_normal_area(surf: &TetSurface, t: usize) -> Option<(Point, f64)> {
    let [a, b, c] = tri_pts(surf, t);
    let x = cross(sub(b, a), sub(c, a));
    unit(x).map(|n| (n, 0.5 * norm(x)))
}

fn is_wall(surf: &TetSurface, layered: &[bool], t: usize) -> bool {
    layered[surf.tri_patch[t] as usize]
}

/// The wall points: every point of a layered triangle, ascending.
fn wall_points(surf: &TetSurface, layered: &[bool]) -> Vec<u32> {
    let mut on = vec![false; surf.points.len()];
    for t in 0..surf.tris.len() {
        if is_wall(surf, layered, t) {
            for &i in &surf.tris[t] {
                on[i as usize] = true;
            }
        }
    }
    (0..surf.points.len() as u32).filter(|&i| on[i as usize]).collect()
}

/// The wall point normals of §116.15.
#[derive(Debug, Clone, PartialEq)]
pub struct WallNormals {
    /// The wall points, the surface's point ids in ascending order.
    pub points: Vec<u32>,
    /// `n*` of (116.22) per wall point, in `points` order: the most normal
    /// normal of the unit normals of its incident layered triangles of
    /// positive area. None where no direction sees every one of them, or
    /// every incident layered triangle has zero area.
    pub normal: Vec<Option<Point>>,
    /// The objective `min_i n* . n_i` per wall point, 0.0 where `normal`
    /// is None.
    pub min_cos: Vec<f64>,
}

/// The most normal normal (116.22) at every wall point: the points of the
/// triangles whose patch `layered` marks (one entry per patch). The
/// triangles must be wound so that their right-hand normals point into
/// the fluid, which is the side the layers grow on.
pub fn wall_normals(surf: &TetSurface, layered: &[bool]) -> Result<WallNormals> {
    validate(surf, layered)?;
    let points = wall_points(surf, layered);
    let mut faces: HashMap<u32, Vec<Point>> = HashMap::new();
    for t in 0..surf.tris.len() {
        if !is_wall(surf, layered, t) {
            continue;
        }
        if let Some((n, _)) = tri_normal_area(surf, t) {
            for &i in &surf.tris[t] {
                faces.entry(i).or_default().push(n);
            }
        }
    }
    let mut normal = Vec::with_capacity(points.len());
    let mut mc = Vec::with_capacity(points.len());
    for p in &points {
        match faces.get(p).and_then(|f| most_normal(f)) {
            Some((n, f)) => {
                normal.push(Some(n));
                mc.push(f);
            }
            None => {
                normal.push(None);
                mc.push(0.0);
            }
        }
    }
    Ok(WallNormals {
        points,
        normal,
        min_cos: mc,
    })
}

/// The layer stack of (116.23): `n` layers, the first `first_thickness`
/// thick, each `growth` times the one below.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LayerStack {
    pub n: usize,
    pub first_thickness: f64,
    pub growth: f64,
}

impl LayerStack {
    /// Refuses, naming the field and its value, `first_thickness` or
    /// `growth` not positive and finite. `n == 0` is the empty stack.
    pub fn validate(&self) -> Result<()> {
        if !(self.first_thickness > 0.0 && self.first_thickness.is_finite()) {
            return Err(Error::Mesh(format!(
                "inflate: layers.first_thickness must be positive and finite, got {}",
                self.first_thickness
            )));
        }
        if !(self.growth > 0.0 && self.growth.is_finite()) {
            return Err(Error::Mesh(format!(
                "inflate: layers.growth must be positive and finite, got {}",
                self.growth
            )));
        }
        Ok(())
    }

    /// The total thickness `T` of (116.23):
    /// `t_1 (r^N - 1) / (r - 1)`, and `N t_1` when `r = 1`.
    pub fn total(&self) -> f64 {
        let n = self.n as f64;
        if self.growth == 1.0 {
            n * self.first_thickness
        } else {
            self.first_thickness * (self.growth.powi(self.n as i32) - 1.0) / (self.growth - 1.0)
        }
    }
}

/// What set a wall point's thickness (§116.15).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Limit {
    /// The stack's own total (116.23).
    Requested,
    /// The gap along the normal, (116.24a).
    Gap,
    /// A concave wall's radius of curvature, (116.24b).
    Concave,
    /// The ratio to a wall neighbour, (116.24c).
    Neighbour,
    /// No normal (116.22): the point gets no layers and is TET-13c's.
    NoNormal,
}

/// The per-point layer thicknesses of §116.15, in [`WallNormals`] order.
#[derive(Debug, Clone, PartialEq)]
pub struct Thicknesses {
    /// The stack's total `T` of (116.23), before any limit.
    pub requested: f64,
    /// The limited total thickness `T(p)`.
    pub total: Vec<f64>,
    /// `gap(p)` of (116.24a), None where the normal meets nothing.
    pub gap: Vec<Option<f64>>,
    /// `rho_c(p)` of (116.24b), None where no incident edge is concave.
    pub rho_c: Vec<Option<f64>>,
    /// The limit that set `total`, the last one to lower it.
    pub limit: Vec<Limit>,
}

impl Thicknesses {
    /// How many wall points each limit set.
    pub fn count(&self, l: Limit) -> usize {
        self.limit.iter().filter(|&&x| x == l).count()
    }
}

/// `1 / rho_c` per surface point, (116.24d): the largest concave curvature
/// estimate (116.9b) over the point's edges. An edge counts when exactly
/// two triangles of positive area share it, it is not a crease
/// (`phi_e < crease`), and it is concave seen from the normals' side: the
/// far point `c_2` of the second triangle lies on the positive side of the
/// first's plane, `n_1 . (c_2 - a) > 0`. 0.0 where none counts.
fn concave_curvature(surf: &TetSurface, crease_rad: f64) -> Vec<f64> {
    let mut edges: HashMap<(u32, u32), Vec<(usize, u32)>> = HashMap::new();
    for (t, tri) in surf.tris.iter().enumerate() {
        for k in 0..3 {
            let (a, b, c) = (tri[k], tri[(k + 1) % 3], tri[(k + 2) % 3]);
            let key = if a < b { (a, b) } else { (b, a) };
            edges.entry(key).or_default().push((t, c));
        }
    }
    let mut kmax = vec![0.0f64; surf.points.len()];
    for (&(a, b), ts) in &edges {
        if ts.len() != 2 {
            continue;
        }
        let ((t1, _), (t2, c2)) = if ts[0].0 < ts[1].0 { (ts[0], ts[1]) } else { (ts[1], ts[0]) };
        let (Some((n1, a1)), Some((n2, a2))) = (tri_normal_area(surf, t1), tri_normal_area(surf, t2)) else {
            continue;
        };
        let phi = angle(n1, n2);
        if phi >= crease_rad {
            continue;
        }
        let pa = surf.points[a as usize];
        if !(dot(n1, sub(surf.points[c2 as usize], pa)) > 0.0) {
            continue;
        }
        let el = norm(sub(surf.points[b as usize], pa));
        let kappa = 3.0 * phi * el / (2.0 * (a1 + a2));
        for i in [a, b] {
            kmax[i as usize] = kmax[i as usize].max(kappa);
        }
    }
    kmax
}

/// The per-point total layer thickness of §116.15 at the wall points of
/// `normals` (from [`wall_normals`] on the same surface and mask).
///
/// Every wall point starts at the stack's total (116.23) and is lowered by
/// (116.24a), `gap_fraction * gap(p)`, where `gap(p)` is (116.10a)'s `tau`
/// of the first triangle that the segment from `p` along `n*(p)` meets,
/// skipping the triangles incident to `p`, from either side, layered or
/// not; then by (116.24b), `concave_fraction * rho_c(p)`. A point with no
/// normal gets 0. Last, (116.24c): `T(p) <= max_neighbour_ratio * T(q)`
/// over every edge of a layered triangle whose two points both have a
/// normal and a positive thickness, imposed exactly by a smallest-first
/// sweep, so each point ends at the smallest of its own limit and
/// `ratio^k` times any point `k` edges away. `crease_deg` is the feature
/// angle: an edge turning by at least it is a crease, which (116.24b) does
/// not see (TET-13c's).
pub fn thicknesses(
    surf: &TetSurface,
    layered: &[bool],
    normals: &WallNormals,
    stack: &LayerStack,
    spec: &TetLayerSpec,
    crease_deg: f64,
) -> Result<Thicknesses> {
    validate(surf, layered)?;
    stack.validate()?;
    if !(spec.max_neighbour_ratio >= 1.0 && spec.max_neighbour_ratio.is_finite()) {
        return Err(Error::Mesh(format!(
            "inflate: tet.layers.max_neighbour_ratio must be >= 1, got {}",
            spec.max_neighbour_ratio
        )));
    }
    for (name, v) in [
        ("gap_fraction", spec.gap_fraction),
        ("concave_fraction", spec.concave_fraction),
    ] {
        if !(v > 0.0 && v <= 1.0) {
            return Err(Error::Mesh(format!(
                "inflate: tet.layers.{name} must be in (0, 1], got {v}"
            )));
        }
    }
    if !(crease_deg > 0.0 && crease_deg <= 180.0) {
        return Err(Error::Mesh(format!(
            "inflate: crease_deg must be finite and inside (0, 180], got {crease_deg}"
        )));
    }
    let wp = wall_points(surf, layered);
    if wp != normals.points
        || normals.normal.len() != wp.len()
        || normals.min_cos.len() != wp.len()
    {
        return Err(Error::Mesh(
            "inflate: the wall normals were not built from this surface and layered mask".into(),
        ));
    }

    let requested = stack.total();
    let np = wp.len();
    let mut incident: Vec<Vec<u32>> = vec![Vec::new(); surf.points.len()];
    for (t, tri) in surf.tris.iter().enumerate() {
        for &i in tri {
            incident[i as usize].push(t as u32);
        }
    }
    let scanner = GapScanner::new(surf);
    let kconc = concave_curvature(surf, crease_deg.to_radians());

    let mut total = vec![requested; np];
    let mut gap = vec![None; np];
    let mut rho_c = vec![None; np];
    let mut limit = vec![Limit::Requested; np];
    for (k, &p) in wp.iter().enumerate() {
        let Some(n) = normals.normal[k] else {
            total[k] = 0.0;
            limit[k] = Limit::NoNormal;
            continue;
        };
        let inc = &incident[p as usize];
        let x = surf.points[p as usize];
        if let Some((_, tau)) = scanner.first_hit_except(x, n, &|s| inc.contains(&(s as u32))) {
            // A hit at the start point can round to a tiny negative tau.
            let tau = tau.max(0.0);
            gap[k] = Some(tau);
            let cap = spec.gap_fraction * tau;
            if cap < total[k] {
                total[k] = cap;
                limit[k] = Limit::Gap;
            }
        }
        if kconc[p as usize] > 0.0 {
            let rho = 1.0 / kconc[p as usize];
            rho_c[k] = Some(rho);
            let cap = spec.concave_fraction * rho;
            if cap < total[k] {
                total[k] = cap;
                limit[k] = Limit::Concave;
            }
        }
    }

    // (116.24c): the smallest-first sweep over the layered edges.
    let slot: HashMap<u32, usize> = wp.iter().enumerate().map(|(k, &p)| (p, k)).collect();
    let live = |k: usize, total: &[f64]| normals.normal[k].is_some() && total[k] > 0.0;
    let mut adj: Vec<Vec<usize>> = vec![Vec::new(); np];
    for t in 0..surf.tris.len() {
        if !is_wall(surf, layered, t) {
            continue;
        }
        let tri = surf.tris[t];
        for e in 0..3 {
            let (i, j) = (slot[&tri[e]], slot[&tri[(e + 1) % 3]]);
            if !adj[i].contains(&j) {
                adj[i].push(j);
            }
            if !adj[j].contains(&i) {
                adj[j].push(i);
            }
        }
    }
    let ratio = spec.max_neighbour_ratio;
    let mut heap: BinaryHeap<Reverse<(u64, usize)>> = BinaryHeap::new();
    for k in 0..np {
        if live(k, &total) {
            heap.push(Reverse((total[k].to_bits(), k)));
        }
    }
    while let Some(Reverse((bits, i))) = heap.pop() {
        if bits != total[i].to_bits() {
            continue;
        }
        let cap = ratio * total[i];
        for &j in &adj[i] {
            if live(j, &total) && cap < total[j] {
                total[j] = cap;
                limit[j] = Limit::Neighbour;
                heap.push(Reverse((cap.to_bits(), j)));
            }
        }
    }
    Ok(Thicknesses {
        requested,
        total,
        gap,
        rho_c,
        limit,
    })
}

/// TET-13a's gates, shared with tests/tetmesh_inflate.rs so the `single`
/// build runs them too.
#[cfg(test)]
mod tests {
    use crate as ofgpu;
    include!("../../tests/shared/tetmesh_inflate_cases.rs");
}
