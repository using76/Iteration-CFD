// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.
// Provenance: see PROVENANCE.md. No GPL-licensed source was consulted.

//! Prism layers by post-inflation, SPEC-LIT §116.15: the wall point
//! normals and the per-point layer thicknesses (TET-13a), then the
//! deformation, its inversion cuts and the prisms (TET-13b, [`inflate`]),
//! and the collapse and taper at sharp edges and corners (TET-13c).
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
//! [`inflate`] then moves the wall points by `T(p) n*(p)` and every other
//! point a tetrahedron uses by the explicit inverse-distance interpolation
//! (116.25) with the two-term weight (116.25a); where a tetrahedron or a
//! prism (116.25f) fails, it halves the thickness of the wall points that
//! cell follows and moves again from the undeformed points. The gap is then
//! filled with `N` prisms per wall triangle (116.25e), whose top layer is
//! the wall triangle itself, so the prism-tet interface is conformal.
//!
//! At a sharp edge or corner (TET-13c) a wall point whose stack would
//! leave the wall at more than 60 degrees to an incident face collapses
//! (116.25g): it carries no layers, its neighbours are capped by half
//! their distance to it (116.25h)'s taper, and a wall triangle whose three
//! points have all collapsed gets no prisms. [`Inflated`] names the
//! collapsed and tapered points, and [`Inflated::coverage`] is (116.25i).
//!
//! Provenance: ORIGINAL. Written from Aubry & Löhner (2008), "On the 'most
//! normal' normal" (the definition (116.22) only; its solution by the
//! support set is this module's), and SPEC-LIT §116.8's curvature (116.9b)
//! and gap (116.10a) estimates; the deformation from Luke, Collins &
//! Blades (2012) as SPEC-LIT §105.9 and §116.15 read it, and the prism
//! fill from Kallinderis, Khawaja & McMorris (1996), the published
//! descriptions only. No GPL-licensed source was consulted.

use super::emit::{prism_valid, reduce_cyclic};
use super::predicates::orient3d;
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
/// TET-13c's collapse handles). The cost is cubic in the distinct normals.
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
    /// A cell the move inverted cut the point's thickness, §116.15's
    /// "The cut": its `T` was halved and the move redone.
    Inversion,
    /// The sharp edge or corner (116.25g), TET-13c's: the point collapses
    /// and carries no layers.
    Collapsed,
    /// The taper cap (116.25h), TET-13c's: the distance to the nearest
    /// collapsed point lowered the point's thickness.
    Taper,
    /// No normal (116.22), set by [`thicknesses`]; the collapse (116.25g)
    /// then marks the point [`Limit::Collapsed`].
    NoNormal,
}

/// The cosine of 60 degrees, (116.25g)'s collapse bar: a wall point whose
/// stack leaves the wall at more than 60 degrees to an incident face
/// collapses.
pub const COLLAPSE_MIN_COS: f64 = 0.5;

/// (116.25h)'s taper slope `s_T`: a wall point's thickness is capped by
/// `s_T` times its distance to the nearest collapsed point.
pub const TAPER_SLOPE: f64 = 0.5;

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
/// not see; (116.25g)'s collapse handles it.
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
    let mut adj = wall_adjacency(surf, layered, &slot, np);
    neighbour_sweep(
        &mut adj,
        spec.max_neighbour_ratio,
        &normals.normal,
        &mut total,
        &mut limit,
    );
    Ok(Thicknesses {
        requested,
        total,
        gap,
        rho_c,
        limit,
    })
}

/// The wall-point adjacency of (116.24c): `adj[i]` holds the wall points
/// sharing an edge of a layered triangle with wall point `i`, first push
/// per edge, in triangle and edge order.
fn wall_adjacency(
    surf: &TetSurface,
    layered: &[bool],
    slot: &HashMap<u32, usize>,
    np: usize,
) -> Vec<Vec<usize>> {
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
    adj
}

/// (116.24c)'s exact smallest-first sweep over the layered edges: the
/// smallest unfinished `T(q)` caps each neighbour at `ratio * T(q)`. A
/// point with no normal, or with `T = 0`, takes no part. Every limit only
/// lowers `T`; a point the sweep lowers, and that no cut has claimed,
/// is marked [`Limit::Neighbour`].
fn neighbour_sweep(
    adj: &mut [Vec<usize>],
    ratio: f64,
    normal: &[Option<Point>],
    total: &mut [f64],
    limit: &mut [Limit],
) {
    let live = |k: usize, total: &[f64]| normal[k].is_some() && total[k] > 0.0;
    let mut heap: BinaryHeap<Reverse<(u64, usize)>> = BinaryHeap::new();
    for k in 0..total.len() {
        if live(k, total) {
            heap.push(Reverse((total[k].to_bits(), k)));
        }
    }
    while let Some(Reverse((bits, i))) = heap.pop() {
        if bits != total[i].to_bits() {
            continue;
        }
        let cap = ratio * total[i];
        for &j in &adj[i] {
            if live(j, total) && cap < total[j] {
                total[j] = cap;
                if limit[j] != Limit::Inversion {
                    limit[j] = Limit::Neighbour;
                }
                heap.push(Reverse((cap.to_bits(), j)));
            }
        }
    }
}

/// What the post-inflation built (§116.15, TET-13b and TET-13c).
#[derive(Debug, Clone)]
pub struct Inflated {
    /// The input points moved by `s` (unused points unchanged), then the
    /// new layer points `x_0(p) .. x_{N-1}(p)` of (116.25d) of the
    /// uncollapsed wall points (116.25g), appended in ascending wall-point
    /// order and, per point, in layer order.
    pub points: Vec<Point>,
    /// The input tetrahedra, same ids: they sit on the displaced wall.
    pub tets: Vec<[u32; 4]>,
    /// The prisms of (116.25e), triangle by triangle and in each from the
    /// wall out; a collapsed point's ids repeat in them, and a wall
    /// triangle whose three points have all collapsed has none.
    pub prisms: Vec<[u32; 6]>,
    /// The boundary after the inflation, §116.15's order: each input face
    /// (a wall triangle replaced by its bottom, the rest unchanged), then
    /// the side quads.
    pub boundary: Vec<(Vec<u32>, u32)>,
    /// The wall normals over the oriented boundary surface.
    pub normals: WallNormals,
    /// The thicknesses over that surface, FINAL: after every cut.
    pub thickness: Thicknesses,
    /// Per wall point, in [`WallNormals::points`] order: the `N + 1` ids of
    /// `x_0 .. x_N`, the last the wall point's own id; a collapsed point's
    /// row is `N + 1` copies of its own id.
    pub layers: Vec<Vec<u32>>,
    /// The moves made (1 when nothing was cut).
    pub rounds: usize,
    /// The wall points whose `T` a cut lowered.
    pub n_cut: usize,
    /// The collapsed wall points (116.25g) by id, ascending: they carry no
    /// layers, and a wall triangle whose three points have all collapsed
    /// gets no prisms.
    pub collapsed: Vec<u32>,
    /// The tapered wall points (116.25h) by id, ascending: every point
    /// whose `T` the cap lowered, even if the sweep lowered it further.
    pub tapered: Vec<u32>,
}

impl Inflated {
    /// (116.25i): the fraction of wall points carrying the full `N`
    /// layers, `(n_wall - n_collapsed) / n_wall`; 1 when there is no wall
    /// point.
    pub fn coverage(&self) -> f64 {
        let n_wall = self.normals.points.len();
        if n_wall == 0 {
            1.0
        } else {
            (n_wall - self.collapsed.len()) as f64 / n_wall as f64
        }
    }
}

/// Up to 20 items, then "and N more".
fn named20<T: std::fmt::Debug>(items: &[T]) -> String {
    let shown: Vec<String> = items.iter().take(20).map(|x| format!("{x:?}")).collect();
    let more = items.len().saturating_sub(20);
    if more > 0 {
        format!("{} and {more} more", shown.join(", "))
    } else {
        shown.join(", ")
    }
}

/// The explicit inverse-distance displacement of Luke, Collins & Blades
/// (2012), §116.15's (116.25) with the weight (116.25a): the wall points
/// move by `T(p) n*(p)`, every other boundary node by 0, and every other
/// point a tetrahedron uses by `sum w_i s_i / sum w_i`, the sums in
/// ascending node order over the undeformed points; a point that coincides
/// with a node takes that node's `s_i`. A point on no boundary face that no
/// tetrahedron uses stays where it is.
#[allow(clippy::too_many_arguments)]
fn displacement(
    points: &[Point],
    used: &[bool],
    is_node: &[bool],
    area: &[f64],
    l_len: f64,
    alpha: f64,
    normals: &WallNormals,
    total: &[f64],
) -> Vec<Point> {
    let n = points.len();
    let mut s_node = vec![[0.0; 3]; n];
    for (k, &p) in normals.points.iter().enumerate() {
        if let Some(m) = normals.normal[k] {
            let t = total[k];
            s_node[p as usize] = [t * m[0], t * m[1], t * m[2]];
        }
    }
    // The boundary nodes in ascending order, so each sum runs over them
    // alone in the order (116.25) fixes.
    let nodes: Vec<usize> = (0..n).filter(|&j| is_node[j]).collect();
    let mut out = vec![[0.0; 3]; n];
    for i in 0..n {
        if is_node[i] {
            out[i] = s_node[i];
            continue;
        }
        if !used[i] {
            continue;
        }
        let x = points[i];
        let mut num = [0.0; 3];
        let mut den = 0.0f64;
        for &j in &nodes {
            let r = norm(sub(x, points[j]));
            if r == 0.0 {
                num = s_node[j];
                den = 1.0;
                break;
            }
            let u = l_len / r;
            let v = alpha * u;
            let w = area[j] * (u * u * u + v * v * v * v * v);
            num[0] += w * s_node[j][0];
            num[1] += w * s_node[j][1];
            num[2] += w * s_node[j][2];
            den += w;
        }
        if den > 0.0 {
            out[i] = [num[0] / den, num[1] / den, num[2] / den];
        }
    }
    out
}

/// The prism layers by post-inflation (§116.15, TET-13b and TET-13c): the
/// optimised tetrahedra, their boundary faces with patches, and the
/// per-patch mask of the layered patches (one entry per patch; a boundary
/// patch id past the mask is refused by name). The wall moves by
/// `s_i = T(p_i) n*_i`, every other point a tetrahedron uses by (116.25),
/// every tetrahedron keeps `orient3d > 0` and every prism (116.25f) -
/// where one does not, the thickness is cut locally (`Limit::Inversion`)
/// and the move is redone, 30 rounds at most - and the gap is filled with
/// `N` prisms per wall triangle (116.25e), conformal by construction. A
/// wall point at a sharp edge or corner collapses (116.25g): no layers,
/// its neighbours tapered by (116.25h), and a fully collapsed wall
/// triangle gets no prisms.
pub fn inflate(
    points: &[Point],
    tets: &[[u32; 4]],
    boundary: &[([u32; 3], u32)],
    layered: &[bool],
    stack: &LayerStack,
    spec: &TetLayerSpec,
    crease_deg: f64,
) -> Result<Inflated> {
    stack.validate()?;
    for (k, &(_, p)) in boundary.iter().enumerate() {
        if p as usize >= layered.len() {
            return Err(Error::Mesh(format!(
                "inflate/boundary: face {k} names patch {p}, but the layered mask has {} entries",
                layered.len()
            )));
        }
    }
    if stack.n == 0 || !boundary.iter().any(|&(_, p)| layered[p as usize]) {
        let faces: Vec<(Vec<u32>, u32)> =
            boundary.iter().map(|&(f, p)| (f.to_vec(), p)).collect();
        return Ok(Inflated {
            points: points.to_vec(),
            tets: tets.to_vec(),
            prisms: Vec::new(),
            boundary: faces,
            normals: WallNormals { points: Vec::new(), normal: Vec::new(), min_cos: Vec::new() },
            thickness: Thicknesses {
                requested: stack.total(),
                total: Vec::new(),
                gap: Vec::new(),
                rho_c: Vec::new(),
                limit: Vec::new(),
            },
            layers: Vec::new(),
            rounds: 1,
            n_cut: 0,
            collapsed: Vec::new(),
            tapered: Vec::new(),
        });
    }
    build_inflated(points, tets, boundary, layered, stack, spec, crease_deg)
}

/// Everything [`inflate`] does past its empty path.
#[allow(clippy::too_many_arguments)]
fn build_inflated(
    points: &[Point],
    tets: &[[u32; 4]],
    boundary: &[([u32; 3], u32)],
    layered: &[bool],
    stack: &LayerStack,
    spec: &TetLayerSpec,
    crease_deg: f64,
) -> Result<Inflated> {
    // The surface of TET-13a's normals and thicknesses: every boundary face
    // wound so that its right-hand normal points into the fluid, on all the
    // mesh's points, with its patch. A listed face must be the face of
    // exactly one tetrahedron; §116.4's outward row, reversed, is the
    // winding (orient3d < 0 against the tet's fourth point).
    let mut on_tet: HashMap<[u32; 3], (u32, u8, u32)> = HashMap::with_capacity(4 * tets.len());
    for (c, t) in tets.iter().enumerate() {
        for (l, row) in super::emit::FACES.iter().enumerate() {
            let mut k = row.map(|i| t[i]);
            k.sort_unstable();
            let e = on_tet.entry(k).or_insert((c as u32, l as u8, 0));
            e.2 += 1;
        }
    }
    let mut surf = TetSurface {
        points: points.to_vec(),
        tris: Vec::with_capacity(boundary.len()),
        tri_patch: Vec::with_capacity(boundary.len()),
        patch_names: (0..layered.len()).map(|i| i.to_string()).collect(),
    };
    let mut bad: Vec<[u32; 3]> = Vec::new();
    for &(f, p) in boundary {
        let mut k = f;
        k.sort_unstable();
        match on_tet.get(&k) {
            Some(&(c, l, 1)) => {
                let w = super::emit::FACES[l as usize].map(|i| tets[c as usize][i]);
                surf.tris.push([w[0], w[2], w[1]]);
                surf.tri_patch.push(p);
            }
            _ => bad.push(f),
        }
    }
    if !bad.is_empty() {
        return Err(Error::Mesh(format!(
            "inflate/boundary: {} listed face(s) are not the face of exactly one tetrahedron: {}",
            bad.len(),
            named20(&bad)
        )));
    }
    inflate_rest(points, tets, boundary, layered, stack, spec, crease_deg, surf)
}

/// The normals, the thicknesses, the collapse and the taper, the move and
/// its cuts, then the prisms.
#[allow(clippy::too_many_arguments)]
fn inflate_rest(
    points: &[Point],
    tets: &[[u32; 4]],
    boundary: &[([u32; 3], u32)],
    layered: &[bool],
    stack: &LayerStack,
    spec: &TetLayerSpec,
    crease_deg: f64,
    surf: TetSurface,
) -> Result<Inflated> {
    let n = points.len();
    let normals = wall_normals(&surf, layered)?;
    let mut th = thicknesses(&surf, layered, &normals, stack, spec, crease_deg)?;
    let np = normals.points.len();
    // (116.25g), TET-13c's collapse, after TET-13a's thicknesses and
    // before the first move: a wall point collapses when it has no normal,
    // or no thickness, or its stack would leave the wall at more than 60
    // degrees to an incident face (`min_cos < COLLAPSE_MIN_COS`). A
    // collapsed point carries no layers: its `T` is 0, its limit is
    // [`Limit::Collapsed`], and it stays where it is.
    let collapsed: Vec<bool> = (0..np)
        .map(|k| {
            normals.normal[k].is_none()
                || !(th.total[k] > 0.0)
                || normals.min_cos[k] < COLLAPSE_MIN_COS
        })
        .collect();
    let collapsed_ids: Vec<u32> = {
        let mut ids = Vec::new();
        for (k, c) in collapsed.iter().enumerate() {
            if *c {
                th.total[k] = 0.0;
                th.limit[k] = Limit::Collapsed;
                ids.push(normals.points[k]);
            }
        }
        ids
    };
    // (116.25h), only when a point has collapsed: every other wall point
    // is capped by `TAPER_SLOPE` times its distance to the nearest
    // collapsed point, on the undeformed points, the minimum over the
    // collapsed points in ascending wall-point order. A point the cap
    // lowers is tapered: its limit becomes [`Limit::Taper`] and the report
    // names it. [`build_final`]'s (116.24c) sweep then runs again. With no
    // collapse none of this runs, and TET-13b's numbers stand bit for bit.
    let mut tapered_ids: Vec<u32> = Vec::new();
    if !collapsed_ids.is_empty() {
        for (k, &p) in normals.points.iter().enumerate() {
            if collapsed[k] {
                continue;
            }
            let x = points[p as usize];
            let mut near = f64::INFINITY;
            for &c in &collapsed_ids {
                let d = norm(sub(x, points[c as usize]));
                if d < near {
                    near = d;
                }
            }
            let cap = TAPER_SLOPE * near;
            if cap < th.total[k] {
                th.total[k] = cap;
                th.limit[k] = Limit::Taper;
                tapered_ids.push(p);
            }
        }
    }
    // The boundary nodes, their (116.25c) areas, (116.25b)'s length, and
    // which points a tetrahedron uses.
    let mut is_node = vec![false; n];
    let mut area = vec![0.0f64; n];
    for tri in &surf.tris {
        let (pa, pb, pc) =
            (points[tri[0] as usize], points[tri[1] as usize], points[tri[2] as usize]);
        let ar = 0.5 * norm(cross(sub(pb, pa), sub(pc, pa)));
        for &i in tri {
            is_node[i as usize] = true;
            area[i as usize] += ar;
        }
    }
    for a in area.iter_mut() {
        *a /= 3.0;
    }
    let (mut lo, mut hi) = ([f64::INFINITY; 3], [-f64::INFINITY; 3]);
    for (i, &on) in is_node.iter().enumerate() {
        if on {
            for a in 0..3 {
                lo[a] = lo[a].min(points[i][a]);
                hi[a] = hi[a].max(points[i][a]);
            }
        }
    }
    let l_len = norm(sub(hi, lo));
    let mut used = vec![false; n];
    for t in tets {
        for &v in t {
            used[v as usize] = true;
        }
    }
    let slot: HashMap<u32, usize> =
        normals.points.iter().enumerate().map(|(k, &p)| (p, k)).collect();
    // The wall triangles in boundary order, their wall slots, and the map
    // from an edge to the boundary faces on it, for the side quads.
    let mut wall_tri: Vec<usize> = Vec::new();
    let mut tri_slots: Vec<[usize; 3]> = Vec::new();
    let mut wall_ix: HashMap<usize, usize> = HashMap::new();
    for (bi, &(_, p)) in boundary.iter().enumerate() {
        if layered[p as usize] {
            let tri = surf.tris[bi];
            wall_ix.insert(bi, wall_tri.len());
            wall_tri.push(bi);
            tri_slots.push([slot[&tri[0]], slot[&tri[1]], slot[&tri[2]]]);
        }
    }
    let mut edge_faces: HashMap<[u32; 2], Vec<usize>> = HashMap::new();
    for (bi, &(f, _)) in boundary.iter().enumerate() {
        for e in 0..3 {
            let (a, b) = (f[e], f[(e + 1) % 3]);
            edge_faces.entry([a.min(b), a.max(b)]).or_default().push(bi);
        }
    }
    // The layer-point offsets of (116.25e): only uncollapsed slots gain
    // points, so slot `k`'s layer `j` sits at `n + off[k] + j`, with
    // `off[k]` `N` times the uncollapsed slots before it (`k * N` with no
    // collapse, TET-13b's numbering).
    let mut off = vec![0usize; np];
    let mut un = 0usize;
    for k in 0..np {
        off[k] = stack.n * un;
        if !collapsed[k] {
            un += 1;
        }
    }
    let up = Setup {
        points,
        tets,
        boundary,
        layered,
        stack,
        spec,
        surf: &surf,
        normals: &normals,
        np,
        n,
        is_node,
        area,
        l_len,
        used,
        slot,
        wall_tri,
        tri_slots,
        wall_ix,
        edge_faces,
        collapsed,
        off,
        collapsed_ids,
        tapered_ids,
    };
    build_final(&up, &mut th)
}

/// Everything the move and the prisms are built from, derived once from the
/// undeformed input.
struct Setup<'a> {
    points: &'a [Point],
    tets: &'a [[u32; 4]],
    boundary: &'a [([u32; 3], u32)],
    layered: &'a [bool],
    stack: &'a LayerStack,
    spec: &'a TetLayerSpec,
    surf: &'a TetSurface,
    normals: &'a WallNormals,
    np: usize,
    n: usize,
    is_node: Vec<bool>,
    area: Vec<f64>,
    l_len: f64,
    used: Vec<bool>,
    slot: HashMap<u32, usize>,
    wall_tri: Vec<usize>,
    tri_slots: Vec<[usize; 3]>,
    wall_ix: HashMap<usize, usize>,
    edge_faces: HashMap<[u32; 2], Vec<usize>>,
    /// Whether wall slot `k` collapsed (116.25g).
    collapsed: Vec<bool>,
    /// Slot `k`'s layer-point offset: its layer `j` is `n + off[k] + j`.
    off: Vec<usize>,
    /// The collapsed points, ascending.
    collapsed_ids: Vec<u32>,
    /// The tapered points, ascending.
    tapered_ids: Vec<u32>,
}

/// (116.25a)'s `alpha = 5 max_i |s_i| / L`, over the current thicknesses.
fn alpha_of(normals: &WallNormals, total: &[f64], l_len: f64) -> f64 {
    let mut max_s = 0.0f64;
    for (k, m) in normals.normal.iter().enumerate() {
        if let Some(m) = m {
            let t = total[k];
            max_s = max_s.max(norm([t * m[0], t * m[1], t * m[2]]));
        }
    }
    if l_len > 0.0 {
        5.0 * max_s / l_len
    } else {
        0.0
    }
}

/// The wall point a tetrahedron's point follows, §116.15's "The cut": the
/// point itself when it is a wall point, else the wall point with `T > 0`
/// of largest `w_i` at the point's undeformed position, the lowest
/// wall-point index on a tie.
fn follower(
    up: &Setup,
    total: &[f64],
    alpha: f64,
    v: u32,
) -> Option<usize> {
    if let Some(&k) = up.slot.get(&v) {
        return Some(k);
    }
    let x = up.points[v as usize];
    let mut best: Option<(f64, usize)> = None;
    for (k, &p) in up.normals.points.iter().enumerate() {
        if !(total[k] > 0.0) {
            continue;
        }
        let r = norm(sub(x, up.points[p as usize]));
        if r == 0.0 {
            return Some(k);
        }
        let u = up.l_len / r;
        let b = alpha * u;
        let w = up.area[p as usize] * (u * u * u + b * b * b * b * b);
        if best.map_or(true, |(bw, _)| w > bw) {
            best = Some((w, k));
        }
    }
    best.map(|(_, k)| k)
}

/// The move, its cuts, and the prisms, over the setup.
fn build_final(up: &Setup, th: &mut Thicknesses) -> Result<Inflated> {
    let (n, np, n_layers) = (up.n, up.np, up.stack.n);
    let mut adj = wall_adjacency(up.surf, up.layered, &up.slot, np);
    // (116.25h)'s follow-up: (116.24c)'s sweep runs again over the capped
    // thicknesses. A collapsed point, with `T = 0`, takes no part in it,
    // as no point with `T = 0` ever does. With no collapse the
    // thicknesses are TET-13a's and TET-13b's bit for bit, and the sweep
    // is skipped.
    if !up.collapsed_ids.is_empty() {
        neighbour_sweep(
            &mut adj,
            up.spec.max_neighbour_ratio,
            &up.normals.normal,
            &mut th.total,
            &mut th.limit,
        );
    }
    // The id of layer `j` of wall slot `k`; `j == N` is the wall point. A
    // collapsed slot has no layer points: every `j` names the wall point
    // itself (116.25g).
    let id_of = |k: usize, j: usize| -> u32 {
        if up.collapsed[k] || j >= n_layers {
            up.normals.points[k]
        } else {
            (n + up.off[k] + j) as u32
        }
    };
    // The prism index of wall triangle `pi`'s first prism, in (116.25e)'s
    // order: a fully collapsed wall triangle gets none.
    let mut prism_base: Vec<usize> = Vec::with_capacity(up.tri_slots.len());
    let mut nb = 0usize;
    for slots in &up.tri_slots {
        prism_base.push(nb);
        if !(up.collapsed[slots[0]] && up.collapsed[slots[1]] && up.collapsed[slots[2]]) {
            nb += n_layers;
        }
    }
    let t_total = up.stack.total();
    let mut rounds = 0usize;
    let mut cut = vec![false; np];
    let (moved, layer_pts) = loop {
        rounds += 1;
        let alpha = alpha_of(up.normals, &th.total, up.l_len);
        let s = displacement(
            up.points, &up.used, &up.is_node, &up.area, up.l_len, alpha, up.normals, &th.total,
        );
        let moved: Vec<Point> = (0..n)
            .map(|i| if s[i] == [0.0; 3] { up.points[i] } else { add(up.points[i], s[i]) })
            .collect();
        // (116.25d), on the undeformed points; x_0 is the undeformed wall.
        // The layer points are built for the uncollapsed slots only: a
        // collapsed point's `x_k` is the point itself for every `k`
        // (116.25g).
        let mut layer_pts: Vec<Point> = Vec::with_capacity(np * n_layers);
        for (k, &p) in up.normals.points.iter().enumerate() {
            let Some(m) = up.normals.normal[k] else { continue };
            if up.collapsed[k] {
                continue;
            }
            for j in 0..n_layers {
                if j == 0 {
                    layer_pts.push(up.points[p as usize]);
                    continue;
                }
                let frac = th.total[k] / t_total;
                let d = if up.stack.growth == 1.0 {
                    frac * j as f64 * up.stack.first_thickness
                } else {
                    frac * up.stack.first_thickness
                        * (up.stack.growth.powi(j as i32) - 1.0)
                        / (up.stack.growth - 1.0)
                };
                layer_pts
                    .push([up.points[p as usize][0] + d * m[0], up.points[p as usize][1] + d * m[1], up.points[p as usize][2] + d * m[2]]);
            }
        }
        let mut all = moved.clone();
        all.extend_from_slice(&layer_pts);
        let bad_tets: Vec<usize> = (0..up.tets.len())
            .filter(|&c| {
                let q = up.tets[c].map(|v| all[v as usize]);
                orient3d(&q[0], &q[1], &q[2], &q[3]) <= 0.0
            })
            .collect();
        let mut bad_prisms: Vec<(usize, usize)> = Vec::new();
        for (pi, slots) in up.tri_slots.iter().enumerate() {
            // A wall triangle whose three points have all collapsed gets
            // no prisms (116.25e); the others keep their `N`, with a
            // collapsed point's ids repeating.
            if up.collapsed[slots[0]] && up.collapsed[slots[1]] && up.collapsed[slots[2]] {
                continue;
            }
            for k in 1..=n_layers {
                let p = [
                    id_of(slots[0], k - 1), id_of(slots[1], k - 1), id_of(slots[2], k - 1),
                    id_of(slots[0], k), id_of(slots[1], k), id_of(slots[2], k),
                ];
                if !prism_valid(&all, &p) {
                    bad_prisms.push((pi, k - 1));
                }
            }
        }
        if bad_tets.is_empty() && bad_prisms.is_empty() {
            break (moved, layer_pts);
        }
        if rounds >= 30 {
            let mut names: Vec<String> =
                bad_tets.iter().map(|c| format!("tet {c}")).collect();
            names.extend(
                bad_prisms.iter().map(|&(pi, k)| format!("prism {}", prism_base[pi] + k)),
            );
            return Err(Error::Mesh(format!(
                "inflate/invert: the move still fails {} cells after 30 rounds: {}",
                bad_tets.len() + bad_prisms.len(),
                named20(&names)
            )));
        }
        // The cut: each named point's T is halved, once per round, its
        // limit becomes Inversion, the sweep runs again, the move is
        // redone from the undeformed points. The cut never names a
        // collapsed point: a failing prism names only its uncollapsed wall
        // points, and a collapsed point is never halved (116.25g).
        let mut hits: Vec<usize> = Vec::new();
        for &c in &bad_tets {
            for &v in &up.tets[c] {
                if let Some(k) = follower(up, &th.total, alpha, v) {
                    hits.push(k);
                }
            }
        }
        for &(pi, _) in &bad_prisms {
            for &s in &up.tri_slots[pi] {
                if !up.collapsed[s] {
                    hits.push(s);
                }
            }
        }
        hits.sort_unstable();
        hits.dedup();
        for &k in &hits {
            if up.collapsed[k] {
                continue;
            }
            th.total[k] *= 0.5;
            th.limit[k] = Limit::Inversion;
            cut[k] = true;
        }
        neighbour_sweep(
            &mut adj,
            up.spec.max_neighbour_ratio,
            &up.normals.normal,
            &mut th.total,
            &mut th.limit,
        );
    };
    assemble(up, th, &moved, &layer_pts, rounds, &cut)
}

/// The layer points appended, the prisms of (116.25e), and the boundary in
/// §116.15's order: each input face (a wall triangle replaced by its
/// bottom), then the side quads in wall-triangle, edge and layer order.
fn assemble(
    up: &Setup,
    th: &Thicknesses,
    moved: &[Point],
    layer_pts: &[Point],
    rounds: usize,
    cut: &[bool],
) -> Result<Inflated> {
    let (n, np, n_layers) = (up.n, up.np, up.stack.n);
    // The id of layer `j` of wall slot `k`, [`build_final`]'s: a collapsed
    // slot names the wall point itself for every `j` (116.25g).
    let id_of = |k: usize, j: usize| -> u32 {
        if up.collapsed[k] || j >= n_layers {
            up.normals.points[k]
        } else {
            (n + up.off[k] + j) as u32
        }
    };
    let mut points_out: Vec<Point> = moved.to_vec();
    points_out.extend_from_slice(layer_pts);
    let mut layers_out: Vec<Vec<u32>> = Vec::with_capacity(np);
    for k in 0..np {
        let mut row = Vec::with_capacity(n_layers + 1);
        for j in 0..=n_layers {
            row.push(id_of(k, j));
        }
        layers_out.push(row);
    }
    let mut prisms_out: Vec<[u32; 6]> = Vec::with_capacity(up.wall_tri.len() * n_layers);
    let mut boundary_out: Vec<(Vec<u32>, u32)> = Vec::with_capacity(up.boundary.len());
    for (bi, &(_, p)) in up.boundary.iter().enumerate() {
        if !up.layered[p as usize] {
            boundary_out.push((up.boundary[bi].0.to_vec(), p));
            continue;
        }
        let wi = up.wall_ix[&bi];
        let slots = up.tri_slots[wi];
        // A wall triangle whose three points have all collapsed gets no
        // prisms: it stays the face of its tetrahedron, and its bottom,
        // the same triangle, stays on its patch. Every other one gets its
        // `N` prisms, a collapsed point's ids repeating (116.25e).
        let full = up.collapsed[slots[0]] && up.collapsed[slots[1]] && up.collapsed[slots[2]];
        if !full {
            for k in 1..=n_layers {
                prisms_out.push([
                    id_of(slots[0], k - 1), id_of(slots[1], k - 1), id_of(slots[2], k - 1),
                    id_of(slots[0], k), id_of(slots[1], k), id_of(slots[2], k),
                ]);
            }
        }
        boundary_out
            .push((vec![id_of(slots[0], 0), id_of(slots[2], 0), id_of(slots[1], 0)], p));
    }
    for &bi in up.wall_tri.iter() {
        let tri = up.surf.tris[bi];
        for e in 0..3 {
            let (a, b) = (tri[e], tri[(e + 1) % 3]);
            let key = [a.min(b), a.max(b)];
            let list = up.edge_faces.get(&key).map_or(0usize, Vec::len);
            if list != 2 {
                return Err(Error::Mesh(format!(
                    "inflate/edges: edge ({a}, {b}) lies on {list} boundary faces, not 2"
                )));
            }
            let other = *up.edge_faces[&key].iter().find(|&&x| x != bi).unwrap();
            let q = up.boundary[other].1;
            if up.layered[q as usize] {
                continue;
            }
            let (ia, ib) = (up.slot[&a], up.slot[&b]);
            for k in 1..=n_layers {
                // A collapsed point's repeated ids drop out of the side
                // quad (116.25e): the face is reduced by its repeated
                // consecutive ids, cyclically, and a face left with fewer
                // than three ids is no face.
                let f = reduce_cyclic(vec![
                    id_of(ia, k - 1), id_of(ib, k - 1), id_of(ib, k), id_of(ia, k),
                ]);
                if f.len() >= 3 {
                    boundary_out.push((f, q));
                }
            }
        }
    }
    Ok(Inflated {
        points: points_out,
        tets: up.tets.to_vec(),
        prisms: prisms_out,
        boundary: boundary_out,
        normals: up.normals.clone(),
        thickness: th.clone(),
        layers: layers_out,
        rounds,
        n_cut: cut.iter().filter(|&&c| c).count(),
        collapsed: up.collapsed_ids.clone(),
        tapered: up.tapered_ids.clone(),
    })
}

/// TET-13a's and TET-13b's gates, shared with tests/tetmesh_inflate.rs so
/// the `single` build runs them too.
#[cfg(test)]
mod tests {
    use crate as ofgpu;
    include!("../../tests/shared/tetmesh_inflate_cases.rs");
}
