// meteor-cfd - Copyright (c) 2026 주식회사 메테오시뮬레이션 (Meteo Simulation Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.

//! Triangulated surfaces: the intake side of castellated meshing.
//!
//! Written from:
//!   ofgpu `SPEC-LIT.md` §23.1-23.4 (surface intake, validation, the uniform
//!     grid bucket for nearest-triangle and column-crossing queries);
//!   Aftosmis, Berger & Melton, *AIAA J.* 36(6) (1998) 952 - castellation
//!     context;
//!   Barill, Dickson, Schmidt, Levin & Jacobson, *ACM TOG* 37(4) (2018) -
//!     why parity voting is the fallback for imperfect surfaces.
//! No GPL-licensed source was consulted.
//!
//! A [`Surface`] is what the reader ([`stl::read_stl`]) produces: welded
//! points, indexed triangles, a patch id per triangle, and geometry
//! recomputed from the winding - stored STL normals are untrustworthy
//! (§23.1) so they are never used. Welding is by exact bit-equality only,
//! marked *DESIGN* in §23.1: STL repeats every vertex per triangle, and a
//! file written by one tool repeats bit-identical coordinates. An epsilon
//! weld would be a silent geometry edit.
//!
//! [`TriIndex`] is the §23.4 uniform grid bucket over the bounding box.
//! No tree: at the sizes a castellated case carries (1e4-1e6 triangles,
//! queried once per near-surface cell at setup) a flat grid with cell size
//! near the mesh spacing is both simpler and fast enough.

pub mod classify;
pub mod cutcell;
pub mod obj;
pub mod stl;

use std::collections::HashMap;

use crate::error::{Error, Result};
use crate::io::contract;
use crate::{Scalar, Vec3};

// ==========================================================================
//  Surface
// ==========================================================================

/// A welded, indexed triangle surface with per-triangle patch identity.
///
/// Everything derived (normals, areas, bounding box) is recomputed from the
/// points and the winding at construction; nothing stored in the source file
/// beyond coordinates and solid names survives into this struct (§23.1).
#[derive(Debug, Clone)]
pub struct Surface {
    /// Welded points. Bit-exact weld only - see the module doc.
    pub points: Vec<Vec3>,
    /// Triangles as indices into `points`, source winding preserved.
    pub tris: Vec<[u32; 3]>,
    /// Patch id of each triangle, an index into `patch_names`.
    pub tri_patch: Vec<u32>,
    /// One name per patch: `solid` names from ASCII STL, the file stem for
    /// binary STL, uniquified on merge (§23.1 patch identity).
    pub patch_names: Vec<String>,
    /// Unit normal per triangle, recomputed as `(v1-v0)x(v2-v0)` normalised.
    pub normals: Vec<Vec3>,
    /// Axis-aligned bounding box `(lo, hi)` of the welded points.
    pub bbox: (Vec3, Vec3),
    /// Total triangle area per patch, same indexing as `patch_names`.
    pub patch_area: Vec<Scalar>,
    /// How many zero-area (at f64, §23.2) triangles were dropped on build.
    pub degenerate_dropped: usize,
}

/// One raw triangle before welding: patch id + three vertices as read.
pub(crate) type SoupTri = (u32, [Vec3; 3]);

/// Bit-pattern key for the exact weld. `to_bits` keeps `-0.0 != 0.0` and
/// distinct NaNs distinct, which is precisely what "no silent geometry
/// edit" requires.
#[inline]
fn weld_key(p: Vec3) -> [u64; 3] {
    [p.x.to_bits() as u64, p.y.to_bits() as u64, p.z.to_bits() as u64]
}

/// Push `name` onto `names`, appending `_2`, `_3`, ... if it is already
/// taken, and warning once per collision - a silent rename would detach the
/// user's boundary conditions from their geometry without a trace.
fn push_unique_name(names: &mut Vec<String>, name: &str) {
    if !names.iter().any(|n| n == name) {
        names.push(name.to_string());
        return;
    }
    let mut k = 2usize;
    let unique = loop {
        let cand = format!("{name}_{k}");
        if !names.iter().any(|n| n == &cand) {
            break cand;
        }
        k += 1;
    };
    contract::warn_once(
        &format!("surface/patch/{name}"),
        &format!("duplicate surface patch name \"{name}\" renamed to \"{unique}\""),
    );
    names.push(unique);
}

impl Surface {
    /// Build a `Surface` from a triangle soup: weld, drop degenerates,
    /// recompute normals, bounding box and per-patch areas.
    ///
    /// The single constructor - both STL flavours and [`Surface::merge`]
    /// funnel through here so every invariant lives in one place.
    pub(crate) fn from_soup(soup: Vec<SoupTri>, patch_names: Vec<String>) -> Result<Surface> {
        if patch_names.is_empty() {
            return Err(Error::Mesh("surface has no patches".into()));
        }

        let mut points: Vec<Vec3> = Vec::new();
        let mut weld: HashMap<[u64; 3], u32> = HashMap::new();
        let mut tris: Vec<[u32; 3]> = Vec::new();
        let mut tri_patch: Vec<u32> = Vec::new();
        let mut normals: Vec<Vec3> = Vec::new();
        let mut patch_area = vec![0.0 as Scalar; patch_names.len()];
        let mut degenerate_dropped = 0usize;

        for (patch, [a, b, c]) in soup {
            if patch as usize >= patch_names.len() {
                return Err(Error::Mesh(format!(
                    "surface triangle references patch {patch} but only {} patches exist",
                    patch_names.len()
                )));
            }
            // Degenerate: zero area at f64 exactly (§23.2). Anything with a
            // nonzero cross product carries a usable normal and stays.
            let cross = (b - a).cross(c - a);
            let two_area = cross.mag();
            if two_area == 0.0 {
                degenerate_dropped += 1;
                continue;
            }

            let mut idx = [0u32; 3];
            for (i, v) in [a, b, c].into_iter().enumerate() {
                let next = points.len() as u32;
                idx[i] = *weld.entry(weld_key(v)).or_insert_with(|| {
                    points.push(v);
                    next
                });
            }

            tris.push(idx);
            tri_patch.push(patch);
            normals.push(cross / two_area);
            patch_area[patch as usize] += 0.5 * two_area;
        }

        if tris.is_empty() {
            return Err(Error::Mesh(
                "surface has no non-degenerate triangles".into(),
            ));
        }
        if degenerate_dropped > 0 {
            eprintln!(
                "[ofgpu] surface: dropped {degenerate_dropped} degenerate \
                 (zero-area) triangle(s)"
            );
        }

        let mut lo = points[0];
        let mut hi = points[0];
        for &p in &points[1..] {
            lo = lo.cmpt_min(p);
            hi = hi.cmpt_max(p);
        }

        Ok(Surface {
            points,
            tris,
            tri_patch,
            patch_names,
            normals,
            bbox: (lo, hi),
            patch_area,
            degenerate_dropped,
        })
    }

    /// Merge several surfaces (one per `-stl` argument) into one, keeping
    /// each input's patches distinct. Duplicate patch names get a numeric
    /// suffix and a warning (§23.1 patch identity).
    pub fn merge(parts: Vec<Surface>) -> Result<Surface> {
        if parts.is_empty() {
            return Err(Error::Mesh("no surfaces to merge".into()));
        }

        let mut names: Vec<String> = Vec::new();
        let mut soup: Vec<SoupTri> = Vec::new();
        let mut dropped = 0usize;

        for part in &parts {
            let base = names.len() as u32;
            for n in &part.patch_names {
                push_unique_name(&mut names, n);
            }
            for (t, tri) in part.tris.iter().enumerate() {
                soup.push((
                    base + part.tri_patch[t],
                    [
                        part.points[tri[0] as usize],
                        part.points[tri[1] as usize],
                        part.points[tri[2] as usize],
                    ],
                ));
            }
            dropped += part.degenerate_dropped;
        }

        let mut merged = Surface::from_soup(soup, names)?;
        // from_soup counts only what IT dropped (nothing: the parts already
        // dropped theirs); carry the original counts forward so the total
        // reported to the user is honest.
        merged.degenerate_dropped += dropped;
        Ok(merged)
    }

    /// Count edge defects: `(open, non_manifold)`.
    ///
    /// §23.2: closed means every undirected edge appears exactly twice, with
    /// opposite orientations. An edge seen once is open; seen more than
    /// twice, or twice with the SAME orientation (a flipped triangle), it is
    /// counted as non-manifold - either way parity ray casting through it is
    /// unreliable.
    pub fn edge_defects(&self) -> (usize, usize) {
        // (forward, reverse) appearance counts per undirected edge (a < b).
        let mut edges: HashMap<(u32, u32), (u32, u32)> = HashMap::new();
        for tri in &self.tris {
            for e in 0..3 {
                let (a, b) = (tri[e], tri[(e + 1) % 3]);
                let entry = edges.entry((a.min(b), a.max(b))).or_insert((0, 0));
                if a < b {
                    entry.0 += 1;
                } else {
                    entry.1 += 1;
                }
            }
        }

        let mut open = 0usize;
        let mut non_manifold = 0usize;
        for &(fwd, rev) in edges.values() {
            match fwd + rev {
                1 => open += 1,
                2 if fwd == 1 && rev == 1 => {}
                _ => non_manifold += 1,
            }
        }
        (open, non_manifold)
    }

    /// Refuse a surface that cannot classify inside/outside.
    ///
    /// Closed-ness is REQUIRED (§23.2, via the §13.4 contract): strict mode
    /// errors naming the defect counts; `-permissive` downgrades to a
    /// warning, and the classifier then leans on parity voting (§23.3,
    /// Barill et al. 2018) to tolerate the holes.
    pub fn require_closed(&self) -> Result<()> {
        let (open, non_manifold) = self.edge_defects();
        if open == 0 && non_manifold == 0 {
            return Ok(());
        }
        contract::unsupported_note(
            "surface/closed",
            &format!("{open} open edge(s), {non_manifold} non-manifold edge(s)"),
            &[],
            "inside/outside classification (SPEC-LIT §23.2) requires a closed \
             surface: every undirected edge shared by exactly two oppositely \
             oriented triangles",
            "parity voting over the open surface (SPEC-LIT §23.3)",
            (),
        )
    }
}

// ==========================================================================
//  TriIndex - the §23.4 uniform grid bucket
// ==========================================================================

/// Uniform grid over the surface bounding box: each bucket lists the
/// triangles whose bounding boxes overlap it.
///
/// Two queries, both of which must say WHICH triangle, because the carver
/// (§23.4) assigns the new wall faces to the surface patch of the triangle
/// it hit or the one it is nearest to.
pub struct TriIndex<'s> {
    surf: &'s Surface,
    lo: Vec3,
    hi: Vec3,
    dims: [usize; 3],
    /// Bucket edge length per axis (extent / dims; a sentinel 1.0 on an axis
    /// the surface does not span, so index arithmetic stays finite).
    cell: [Scalar; 3],
    buckets: Vec<Vec<u32>>,
}

impl<'s> TriIndex<'s> {
    /// Build the grid with bucket size near `cell_hint` - "~ the mesh
    /// spacing" per §23.4, so a bucket holds the handful of triangles a
    /// nearby cell face could care about.
    pub fn new(surf: &'s Surface, cell_hint: Scalar) -> Result<TriIndex<'s>> {
        if !(cell_hint > 0.0) {
            return Err(Error::Mesh(format!(
                "TriIndex cell size must be positive, got {cell_hint}"
            )));
        }
        let (lo, hi) = surf.bbox;
        let ext = hi - lo;

        let mut dims = [0usize; 3];
        for ax in 0..3 {
            let n = (ext.component(ax) / cell_hint).ceil();
            dims[ax] = if n.is_finite() { (n as usize).clamp(1, 4096) } else { 1 };
        }
        // Bound total memory: halve the largest axis until the bucket count
        // is sane. The grid is an accelerator, not a truth - a coarser grid
        // is merely slower.
        while dims[0] * dims[1] * dims[2] > (1 << 20) {
            let ax = (0..3).max_by_key(|&a| dims[a]).unwrap_or(0);
            dims[ax] = (dims[ax] + 1) / 2;
        }

        let mut cell = [1.0 as Scalar; 3];
        for ax in 0..3 {
            let e = ext.component(ax);
            if e > 0.0 {
                cell[ax] = e / dims[ax] as Scalar;
            }
        }

        let mut buckets = vec![Vec::new(); dims[0] * dims[1] * dims[2]];
        for (t, tri) in surf.tris.iter().enumerate() {
            let (a, b, c) = (
                surf.points[tri[0] as usize],
                surf.points[tri[1] as usize],
                surf.points[tri[2] as usize],
            );
            let tlo = a.cmpt_min(b).cmpt_min(c);
            let thi = a.cmpt_max(b).cmpt_max(c);
            let ilo = Self::coords(lo, cell, dims, tlo);
            let ihi = Self::coords(lo, cell, dims, thi);
            for k in ilo[2]..=ihi[2] {
                for j in ilo[1]..=ihi[1] {
                    for i in ilo[0]..=ihi[0] {
                        buckets[i + dims[0] * (j + dims[1] * k)].push(t as u32);
                    }
                }
            }
        }

        Ok(TriIndex { surf, lo, hi, dims, cell, buckets })
    }

    /// Bucket coordinates of `p`, clamped into the grid.
    fn coords(lo: Vec3, cell: [Scalar; 3], dims: [usize; 3], p: Vec3) -> [usize; 3] {
        let mut c = [0usize; 3];
        for ax in 0..3 {
            let f = ((p.component(ax) - lo.component(ax)) / cell[ax]).floor();
            c[ax] = if f > 0.0 { (f as usize).min(dims[ax] - 1) } else { 0 };
        }
        c
    }

    /// The triangle nearest to `p` and its (unsigned) distance.
    ///
    /// Expanding-ring search: scan buckets in shells of increasing Chebyshev
    /// radius around `p`'s bucket, stopping once no unscanned bucket can
    /// hold anything closer than the best found. Distance is exact
    /// point-to-triangle; the grid only orders the candidates.
    pub fn nearest_triangle(&self, p: Vec3) -> (usize, Scalar) {
        let base = Self::coords(self.lo, self.cell, self.dims, p);
        // Correction for a query point outside the bbox: shell radius r
        // guarantees distance >= r*min_cell measured from the CLAMPED point,
        // so the bound seen from p is weaker by |p - clamp(p)|.
        let clamped = p.cmpt_max(self.lo).cmpt_min(self.hi);
        let d0 = (p - clamped).mag();
        let min_cell = self.cell[0].min(self.cell[1]).min(self.cell[2]);
        let max_r = *self.dims.iter().max().unwrap_or(&1);

        let mut best = (0usize, Scalar::INFINITY);
        for r in 0..=max_r {
            self.scan_shell(base, r, |tri| {
                let d = self.dist_to_tri(p, tri);
                if d < best.1 {
                    best = (tri, d);
                }
            });
            // After finishing shell r, every unscanned triangle lives in a
            // bucket at Chebyshev index distance > r, hence at geometric
            // distance >= r*min_cell - d0 from p.
            if best.1 <= r as Scalar * min_cell - d0 {
                break;
            }
        }
        best
    }

    /// Visit every triangle in buckets at Chebyshev radius exactly `r`
    /// (buckets outside the grid skipped). Triangles spanning several
    /// buckets are visited more than once; the visitor must be idempotent.
    fn scan_shell(&self, base: [usize; 3], r: usize, mut visit: impl FnMut(usize)) {
        let lo_i = |b: usize| b.saturating_sub(r);
        let hi_i = |b: usize, ax: usize| (b + r).min(self.dims[ax] - 1);
        let ri = r as isize;
        for k in lo_i(base[2])..=hi_i(base[2], 2) {
            for j in lo_i(base[1])..=hi_i(base[1], 1) {
                for i in lo_i(base[0])..=hi_i(base[0], 0) {
                    let cheb = (i as isize - base[0] as isize)
                        .abs()
                        .max((j as isize - base[1] as isize).abs())
                        .max((k as isize - base[2] as isize).abs());
                    if cheb != ri {
                        continue;
                    }
                    for &t in &self.buckets[i + self.dims[0] * (j + self.dims[1] * k)] {
                        visit(t as usize);
                    }
                }
            }
        }
    }

    /// Exact distance from `p` to triangle `t`.
    fn dist_to_tri(&self, p: Vec3, t: usize) -> Scalar {
        let tri = self.surf.tris[t];
        let cp = closest_point_on_triangle(
            p,
            self.surf.points[tri[0] as usize],
            self.surf.points[tri[1] as usize],
            self.surf.points[tri[2] as usize],
        );
        (p - cp).mag()
    }

    /// All crossings of the line `{(t, y, z) : t in R}` with the surface,
    /// sorted by `t`, each with the triangle it pierced.
    ///
    /// This is the §23.3 column ray: parity between consecutive crossings
    /// classifies inside/outside, and the pierced triangle's patch labels
    /// the wall face the carver creates there (§23.4). The intersection test
    /// is watertight for the triangulation as a whole: 2D edge functions in
    /// the (y,z) projection with a top-left style tie-break, so a line
    /// through a shared edge or vertex is counted by exactly one of the
    /// incident triangles, never zero or two.
    pub fn crossings_x(&self, y: Scalar, z: Scalar) -> Vec<(Scalar, usize)> {
        // A column outside the bounding box cannot cross anything.
        if y < self.lo.y || y > self.hi.y || z < self.lo.z || z > self.hi.z {
            return Vec::new();
        }
        let base = Self::coords(self.lo, self.cell, self.dims, Vec3::new(self.lo.x, y, z));
        let (j, k) = (base[1], base[2]);

        // Gather candidates along the whole x-row of buckets, deduplicated -
        // a triangle spanning several buckets must be tested once.
        let mut cand: Vec<u32> = Vec::new();
        for i in 0..self.dims[0] {
            cand.extend_from_slice(&self.buckets[i + self.dims[0] * (j + self.dims[1] * k)]);
        }
        cand.sort_unstable();
        cand.dedup();

        let mut hits: Vec<(Scalar, usize)> = Vec::new();
        for t in cand {
            let tri = self.surf.tris[t as usize];
            if let Some(tx) = x_line_hit(
                self.surf.points[tri[0] as usize],
                self.surf.points[tri[1] as usize],
                self.surf.points[tri[2] as usize],
                y,
                z,
            ) {
                hits.push((tx, t as usize));
            }
        }
        hits.sort_by(|a, b| a.0.total_cmp(&b.0));
        hits
    }
}

/// Watertight x-line/triangle intersection: does the line `(t, y, z)` pierce
/// triangle `(a, b, c)`, and at which `x = t`?
///
/// Project to (y,z); the three edge functions are the signed sub-areas, and
/// the line is inside when all three share the triangle's orientation sign.
/// A zero edge function (line exactly through an edge or vertex) is resolved
/// by a fill-rule on the directed edge, the rasteriser construction: of the
/// two triangles sharing that edge - which traverse it in opposite
/// directions when the surface is consistently wound - exactly one accepts.
/// That keeps column parity exact through shared features; the residual
/// floating-point disagreements are what §23.3's jitter-and-vote absorbs.
fn x_line_hit(a: Vec3, b: Vec3, c: Vec3, y: Scalar, z: Scalar) -> Option<Scalar> {
    let (u0, v0) = (a.y - y, a.z - z);
    let (u1, v1) = (b.y - y, b.z - z);
    let (u2, v2) = (c.y - y, c.z - z);

    // Edge functions: w0 spans edge b->c, w1 spans c->a, w2 spans a->b.
    let w0 = u1 * v2 - u2 * v1;
    let w1 = u2 * v0 - u0 * v2;
    let w2 = u0 * v1 - u1 * v0;

    let area = w0 + w1 + w2;
    if area == 0.0 {
        // Projected-degenerate: the triangle is parallel to the x-axis. Its
        // crossing is grazing; the neighbouring triangles carry the parity.
        return None;
    }
    // Normalise to positive orientation so one fill-rule serves both
    // windings; flipping the signs is flipping the traversal direction.
    let s: Scalar = if area > 0.0 { 1.0 } else { -1.0 };

    let edges = [
        (w0, (u2 - u1, v2 - v1)),
        (w1, (u0 - u2, v0 - v2)),
        (w2, (u1 - u0, v1 - v0)),
    ];
    for (w, (du, dv)) in edges {
        let w = s * w;
        if w < 0.0 {
            return None;
        }
        if w == 0.0 {
            // Fill rule: a zero-area edge counts only when directed "up",
            // or horizontal and directed "left". Opposite traversal fails
            // the same test, so a shared edge is claimed exactly once.
            let (du, dv) = (s * du, s * dv);
            if !(dv > 0.0 || (dv == 0.0 && du < 0.0)) {
                return None;
            }
        }
    }

    // Barycentric interpolation of x at the hit; w_i/area is the weight of
    // vertex i regardless of the orientation sign.
    Some((w0 * a.x + w1 * b.x + w2 * c.x) / area)
}

/// Closest point on triangle `(a, b, c)` to `p`.
///
/// The classic Voronoi-region walk: test the vertex regions, then the edge
/// regions, then fall through to the face interior - each test is a pair of
/// dot-product signs, so no division happens until the region is known and
/// its denominator is provably positive.
fn closest_point_on_triangle(p: Vec3, a: Vec3, b: Vec3, c: Vec3) -> Vec3 {
    let ab = b - a;
    let ac = c - a;

    let ap = p - a;
    let d1 = ab.dot(ap);
    let d2 = ac.dot(ap);
    if d1 <= 0.0 && d2 <= 0.0 {
        return a; // vertex region A
    }

    let bp = p - b;
    let d3 = ab.dot(bp);
    let d4 = ac.dot(bp);
    if d3 >= 0.0 && d4 <= d3 {
        return b; // vertex region B
    }

    let vc = d1 * d4 - d3 * d2;
    if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
        return a + ab * (d1 / (d1 - d3)); // edge region AB
    }

    let cp = p - c;
    let d5 = ab.dot(cp);
    let d6 = ac.dot(cp);
    if d6 >= 0.0 && d5 <= d6 {
        return c; // vertex region C
    }

    let vb = d5 * d2 - d1 * d6;
    if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
        return a + ac * (d2 / (d2 - d6)); // edge region AC
    }

    let va = d3 * d6 - d5 * d4;
    if va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0 {
        return b + (c - b) * ((d4 - d3) / ((d4 - d3) + (d5 - d6))); // edge BC
    }

    // Face interior: all three sub-areas positive, so the sum is too.
    let denom = 1.0 / (va + vb + vc);
    a + ab * (vb * denom) + ac * (vc * denom)
}

// ==========================================================================
//  Tests
// ==========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// The unit cube [0,1]^3 as a consistently outward-wound triangle soup.
    /// 8 distinct corners repeated across 12 triangles - the weld test bed.
    pub(super) fn cube_points() -> [Vec3; 8] {
        [
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(1.0, 1.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
            Vec3::new(1.0, 0.0, 1.0),
            Vec3::new(1.0, 1.0, 1.0),
            Vec3::new(0.0, 1.0, 1.0),
        ]
    }

    /// Outward winding throughout; verified by the normal assertions below.
    pub(super) const CUBE_TRIS: [[usize; 3]; 12] = [
        [0, 3, 2], [0, 2, 1], // z = 0
        [4, 5, 6], [4, 6, 7], // z = 1
        [0, 4, 7], [0, 7, 3], // x = 0
        [1, 2, 6], [1, 6, 5], // x = 1
        [0, 1, 5], [0, 5, 4], // y = 0
        [3, 7, 6], [3, 6, 2], // y = 1
    ];

    fn cube_soup() -> Vec<SoupTri> {
        let p = cube_points();
        CUBE_TRIS
            .iter()
            .map(|&[a, b, c]| (0u32, [p[a], p[b], p[c]]))
            .collect()
    }

    fn cube() -> Surface {
        match Surface::from_soup(cube_soup(), vec!["cube".into()]) {
            Ok(s) => s,
            Err(e) => panic!("cube build failed: {e}"),
        }
    }

    #[test]
    fn cube_welds_geometry_and_areas() {
        let s = cube();
        assert_eq!(s.points.len(), 8);
        assert_eq!(s.tris.len(), 12);
        assert_eq!(s.degenerate_dropped, 0);
        assert_eq!(s.bbox.0, Vec3::ZERO);
        assert_eq!(s.bbox.1, Vec3::new(1.0, 1.0, 1.0));
        // One patch, six unit faces.
        assert!((s.patch_area[0] - 6.0).abs() < 1e-12);
        // Recomputed normals are outward unit vectors: x = 0 face -> -x.
        assert_eq!(s.normals[4], Vec3::new(-1.0, 0.0, 0.0));
        assert_eq!(s.normals[6], Vec3::new(1.0, 0.0, 0.0));
        // Consistently wound and closed.
        assert_eq!(s.edge_defects(), (0, 0));
        assert!(s.require_closed().is_ok());
    }

    #[test]
    fn open_box_is_refused_naming_four_open_edges() {
        crate::io::contract::set_permissive(false);
        // Drop the z = 1 lid: 10 triangles, the top rim's 4 edges open.
        let p = cube_points();
        let soup: Vec<SoupTri> = CUBE_TRIS
            .iter()
            .filter(|&&t| !matches!(t, [4, 5, 6] | [4, 6, 7]))
            .map(|&[a, b, c]| (0u32, [p[a], p[b], p[c]]))
            .collect();
        assert_eq!(soup.len(), 10);
        let s = match Surface::from_soup(soup, vec!["box".into()]) {
            Ok(s) => s,
            Err(e) => panic!("build failed: {e}"),
        };
        assert_eq!(s.edge_defects(), (4, 0));

        let e = match s.require_closed() {
            Err(e) => e.to_string(),
            Ok(()) => panic!("open box was accepted"),
        };
        assert!(e.contains("4 open edge"), "{e}");
        assert!(e.contains("-permissive"), "{e}");
    }

    #[test]
    fn degenerate_triangle_dropped_with_count() {
        let p = cube_points();
        let mut soup = cube_soup();
        soup.push((0, [p[0], p[0], p[6]])); // zero area exactly
        let s = match Surface::from_soup(soup, vec!["cube".into()]) {
            Ok(s) => s,
            Err(e) => panic!("build failed: {e}"),
        };
        assert_eq!(s.tris.len(), 12);
        assert_eq!(s.degenerate_dropped, 1);
        assert_eq!(s.points.len(), 8);
    }

    #[test]
    fn merge_suffixes_duplicate_patch_names() {
        let merged = match Surface::merge(vec![cube(), cube()]) {
            Ok(s) => s,
            Err(e) => panic!("merge failed: {e}"),
        };
        assert_eq!(merged.patch_names, vec!["cube".to_string(), "cube_2".to_string()]);
        assert_eq!(merged.tris.len(), 24);
        // Identical coordinates weld across the two inputs - bit-exact.
        assert_eq!(merged.points.len(), 8);
        assert_eq!(merged.tri_patch[0], 0);
        assert_eq!(merged.tri_patch[12], 1);
    }

    #[test]
    fn nearest_triangle_matches_hand_values() {
        let s = cube();
        let idx = match TriIndex::new(&s, 0.5) {
            Ok(i) => i,
            Err(e) => panic!("index build failed: {e}"),
        };

        // Outside, facing the x = 1 face at (1, 0.25, 0.75): distance 1.
        // In that face's (y,z) plane the point is above the diagonal, so of
        // the two x = 1 triangles only [1,6,5] (index 7) contains the foot.
        let (t, d) = idx.nearest_triangle(Vec3::new(2.0, 0.25, 0.75));
        assert_eq!(t, 7);
        assert!((d - 1.0).abs() < 1e-12, "d = {d}");

        // Inside, nearest the z = 1 lid: (0.5, 0.25) is below the lid's
        // (x,y) diagonal, so triangle [4,5,6] (index 2), distance 0.05.
        let (t, d) = idx.nearest_triangle(Vec3::new(0.5, 0.25, 0.95));
        assert_eq!(t, 2);
        assert!((d - 0.05).abs() < 1e-12, "d = {d}");
    }

    #[test]
    fn crossings_x_matches_hand_values() {
        let s = cube();
        let idx = match TriIndex::new(&s, 0.5) {
            Ok(i) => i,
            Err(e) => panic!("index build failed: {e}"),
        };

        // The column (y,z) = (0.25, 0.75) avoids every face diagonal: it
        // pierces x = 0 in triangle [0,4,7] (index 4) and x = 1 in triangle
        // [1,6,5] (index 7), at t = 0 and t = 1.
        let hits = idx.crossings_x(0.25, 0.75);
        assert_eq!(hits.len(), 2, "{hits:?}");
        assert!((hits[0].0 - 0.0).abs() < 1e-12);
        assert_eq!(hits[0].1, 4);
        assert!((hits[1].0 - 1.0).abs() < 1e-12);
        assert_eq!(hits[1].1, 7);

        // Parity between the crossings: inside.
        assert!(hits[0].0 < 0.5 && 0.5 < hits[1].0);

        // A column through the x = 0 face's diagonal (y = z) must count
        // each surface crossing exactly once - the fill rule assigns the
        // shared edge to one triangle, never zero or both.
        let hits = idx.crossings_x(0.5, 0.5);
        assert_eq!(hits.len(), 2, "{hits:?}");
        assert!((hits[0].0 - 0.0).abs() < 1e-12);
        assert!((hits[1].0 - 1.0).abs() < 1e-12);

        // Outside the bounding box: no crossings.
        assert!(idx.crossings_x(1.5, 0.5).is_empty());
    }
}
