// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
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
//! [`TriIndex`] is the §23.4 uniform grid bucket over the bounding box, and
//! the grid still decides WHICH of several equidistant triangles a query
//! returns - its expanding-shell visit order is kept as the tie-break. What
//! [`TriIndex::with_bvh`] adds is a median-split bounding-volume hierarchy
//! over the same triangles, Ericson 2005 ch. 6 read as a textbook: the same
//! answer, found faster, which the automesher (§92.2) needs because its
//! surfaces mix a small dense body with domain-sized triangles a flat grid's
//! shells walk past.

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
    /// The bounding-volume hierarchy [`TriIndex::with_bvh`] builds; `new`
    /// leaves it out and queries stay on the grid alone.
    bvh: Option<Bvh>,
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

        Ok(TriIndex { surf, lo, hi, dims, cell, buckets, bvh: None })
    }

    /// [`TriIndex::new`]'s grid PLUS a bounding-volume hierarchy over the
    /// triangles' boxes - what the automesher builds, whose surfaces mix a
    /// small dense body with domain-sized triangles. The grid stays the
    /// authority on WHICH equidistant triangle a query returns; the hierarchy
    /// only gets there faster.
    pub fn with_bvh(surf: &'s Surface, cell_hint: Scalar) -> Result<TriIndex<'s>> {
        let mut idx = TriIndex::new(surf, cell_hint)?;
        let boxes: Vec<(Vec3, Vec3)> = surf
            .tris
            .iter()
            .map(|tri| {
                let a = surf.points[tri[0] as usize];
                let b = surf.points[tri[1] as usize];
                let c = surf.points[tri[2] as usize];
                (a.cmpt_min(b).cmpt_min(c), a.cmpt_max(b).cmpt_max(c))
            })
            .collect();
        idx.bvh = Some(Bvh::build(&boxes));
        Ok(idx)
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
    ///
    /// With a BVH ( [`TriIndex::with_bvh`]) the same minimum is found by
    /// `Bvh::visit_near`, and among triangles tied at the bitwise-equal
    /// minimum the one the grid scan would have reached FIRST is returned -
    /// the shell it is first met in and the scan's own (k, j, i, id) order
    /// inside that shell, [`TriIndex::legacy_key`]. An early grid stop can
    /// only leave a tie unvisited in a LATER shell, which loses the key
    /// comparison, so the two paths pick the same triangle.
    pub fn nearest_triangle(&self, p: Vec3) -> (usize, Scalar) {
        if let Some(bvh) = &self.bvh {
            let mut best = Scalar::INFINITY;
            let mut ties: Vec<u32> = Vec::new();
            let g = bvh.guard(p);
            bvh.visit_near(p, Scalar::INFINITY, |t| {
                let d = self.dist_to_tri(p, t as usize);
                if d < best {
                    best = d;
                    ties.clear();
                    ties.push(t);
                } else if d == best {
                    ties.push(t);
                }
                best + g
            });
            if ties.is_empty() {
                return (0, Scalar::INFINITY);
            }
            let pick = if ties.len() == 1 {
                ties[0]
            } else {
                let base = Self::coords(self.lo, self.cell, self.dims, p);
                let mut pick = ties[0];
                let mut pick_key = self.legacy_key(base, pick);
                for &t in &ties[1..] {
                    let k = self.legacy_key(base, t);
                    if k < pick_key {
                        pick_key = k;
                        pick = t;
                    }
                }
                pick
            };
            return (pick as usize, best);
        }
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

    /// The closest POINT on the surface to `p`, the triangle carrying it and
    /// the distance - what `nearest_triangle` measures but does not return.
    ///
    /// The triangle is `nearest_triangle`'s and the point is
    /// `closest_point_on_triangle` on that triangle, so the distance this
    /// returns is the distance `nearest_triangle` returns, to the bit.
    /// SPEC-LIT §92.11 (92.28): the target of the automesher's snap.
    pub fn closest_point(&self, p: Vec3) -> (Vec3, usize, Scalar) {
        let (t, d) = self.nearest_triangle(p);
        let tri = self.surf.tris[t];
        let q = closest_point_on_triangle(
            p,
            self.surf.points[tri[0] as usize],
            self.surf.points[tri[1] as usize],
            self.surf.points[tri[2] as usize],
        );
        (q, t, d)
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

    /// The order the §23.4 shell scan first reaches triangle `t` from
    /// `base`, as a comparable key: the shell radius it is first met in,
    /// then the scan's own (k, j, i, id) order of the first of its buckets
    /// that shell covers. Ranking bitwise ties by this key ranks them the
    /// way the scan visits them, so the smallest key IS the triangle the
    /// scan would return.
    fn legacy_key(&self, base: [usize; 3], t: u32) -> (usize, usize, usize, usize, u32) {
        let tri = self.surf.tris[t as usize];
        let (a, b, c) = (
            self.surf.points[tri[0] as usize],
            self.surf.points[tri[1] as usize],
            self.surf.points[tri[2] as usize],
        );
        let ilo = Self::coords(self.lo, self.cell, self.dims, a.cmpt_min(b).cmpt_min(c));
        let ihi = Self::coords(self.lo, self.cell, self.dims, a.cmpt_max(b).cmpt_max(c));
        // The smallest shell whose ring of buckets meets the triangle's
        // range: the largest per-axis index gap from `base`.
        let mut r_t = 0usize;
        for ax in 0..3 {
            let gap = if base[ax] < ilo[ax] {
                ilo[ax] - base[ax]
            } else if base[ax] > ihi[ax] {
                base[ax] - ihi[ax]
            } else {
                0
            };
            r_t = r_t.max(gap);
        }
        // The scan visits shell `r_t` in ascending (k, j, i), so the first
        // of the triangle's buckets it touches is the per-axis low end of
        // range and shell intersected.
        let mut first = [0usize; 3];
        for ax in 0..3 {
            first[ax] = base[ax].saturating_sub(r_t).max(ilo[ax]);
        }
        (r_t, first[2], first[1], first[0], t)
    }

    /// The exact nearest-triangle distance, answered only when it is
    /// `<= r_max`: `Some(d)` with the same bits [`TriIndex::nearest_triangle`]
    /// returns whenever that distance is within `r_max`, `None` otherwise.
    /// The bound is what lets a caller stop the search early - the refinement
    /// bands (§92.2) only ever ask whether a leaf is inside a distance they
    /// already know.
    pub fn nearest_distance_within(&self, p: Vec3, r_max: Scalar) -> Option<Scalar> {
        match &self.bvh {
            Some(bvh) => {
                let g = bvh.guard(p);
                let mut best = Scalar::INFINITY;
                bvh.visit_near(p, r_max + g, |t| {
                    let d = self.dist_to_tri(p, t as usize);
                    best = best.min(d);
                    best.min(r_max) + g
                });
                if best <= r_max {
                    Some(best)
                } else {
                    None
                }
            }
            None => {
                let (_, d) = self.nearest_triangle(p);
                if d <= r_max {
                    Some(d)
                } else {
                    None
                }
            }
        }
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

/// Distance from `p` to the axis-aligned box `[lo, hi]`: the square root of
/// the sum over the axes of the squared gap - zero on every axis `p` already
/// spans, so 0 inside.
pub fn box_distance(p: Vec3, lo: Vec3, hi: Vec3) -> Scalar {
    let gap = |a: Scalar, l: Scalar, h: Scalar| {
        if a < l {
            l - a
        } else if a > h {
            a - h
        } else {
            0.0
        }
    };
    let (dx, dy, dz) = (gap(p.x, lo.x, hi.x), gap(p.y, lo.y, hi.y), gap(p.z, lo.z, hi.z));
    (dx * dx + dy * dy + dz * dz).sqrt()
}

/// One node of a [`Bvh`]: the union box of its items, and either two
/// children or a run of items.
struct BvhNode {
    lo: Vec3,
    hi: Vec3,
    /// Inner node: the node index of the left child. Leaf: the item run's
    /// first index into [`Bvh::items`].
    a: u32,
    /// Inner node: the node index of the right child. Leaf: the item run's
    /// length.
    b: u32,
    /// Items held; 0 marks an inner node.
    count: u32,
}

/// A median-split bounding-volume hierarchy over axis-aligned boxes -
/// Ericson 2005 ch. 6 read as a textbook. Deterministic by construction: the
/// same input builds the same tree, and [`Bvh::visit_near`] walks it in a
/// fixed order. The automesher (§92.2) builds one over a surface's
/// triangles, whose small dense body defeats the §23.4 grid's shell walk;
/// the grid keeps the authority on ties, the hierarchy only gets there
/// faster.
pub struct Bvh {
    nodes: Vec<BvhNode>,
    /// Item ids, reordered by the build; a leaf holds a run of them in
    /// ascending id.
    items: Vec<u32>,
    /// Each item's own box, what a leaf re-checks before visiting.
    item_boxes: Vec<(Vec3, Vec3)>,
}

impl Bvh {
    /// Build over `boxes` - item `i` is `boxes[i]`. A node over more than
    /// [`Bvh::LEAF`] items splits on the longest axis of the bounding box of
    /// its items' box CENTRES (ties to the lowest axis), orders its items by
    /// centre on that axis then id, and halves them; a node over at most
    /// that many is a leaf holding its items in ascending id.
    pub fn build(boxes: &[(Vec3, Vec3)]) -> Bvh {
        let mut bvh = Bvh {
            nodes: Vec::new(),
            items: (0..boxes.len() as u32).collect(),
            item_boxes: boxes.to_vec(),
        };
        if !boxes.is_empty() {
            bvh.build_node(boxes, 0, boxes.len());
        }
        bvh
    }

    const LEAF: usize = 4;

    fn build_node(&mut self, boxes: &[(Vec3, Vec3)], first: usize, last: usize) -> u32 {
        let mut lo = boxes[self.items[first] as usize].0;
        let mut hi = boxes[self.items[first] as usize].1;
        for &it in &self.items[first..last] {
            let (blo, bhi) = boxes[it as usize];
            lo = lo.cmpt_min(blo);
            hi = hi.cmpt_max(bhi);
        }
        let count = last - first;
        if count <= Self::LEAF {
            self.items[first..last].sort_unstable();
            let idx = self.nodes.len() as u32;
            self.nodes
                .push(BvhNode { lo, hi, a: first as u32, b: count as u32, count: count as u32 });
            return idx;
        }
        // The split axis: the longest extent of the box around the items'
        // box centres, ties to the lowest axis.
        let mut clo = Vec3::new(Scalar::INFINITY, Scalar::INFINITY, Scalar::INFINITY);
        let mut chi = Vec3::new(Scalar::NEG_INFINITY, Scalar::NEG_INFINITY, Scalar::NEG_INFINITY);
        for &it in &self.items[first..last] {
            let (blo, bhi) = boxes[it as usize];
            let c = Vec3::new(
                0.5 * (blo.x + bhi.x),
                0.5 * (blo.y + bhi.y),
                0.5 * (blo.z + bhi.z),
            );
            clo = clo.cmpt_min(c);
            chi = chi.cmpt_max(c);
        }
        let (ex, ey, ez) =
            (chi.x - clo.x, chi.y - clo.y, chi.z - clo.z);
        let axis = if ex >= ey && ex >= ez {
            0
        } else if ey >= ez {
            1
        } else {
            2
        };
        self.items[first..last].sort_by(|&x, &y| {
            let cx = |it: u32| {
                let (blo, bhi) = boxes[it as usize];
                0.5 * (blo.component(axis) + bhi.component(axis))
            };
            cx(x).total_cmp(&cx(y)).then(x.cmp(&y))
        });
        let mid = first + count / 2;
        // The parent takes its slot BEFORE the children are built, so node 0
        // is the root - what `visit_near`'s walk starts from.
        let idx = self.nodes.len() as u32;
        self.nodes.push(BvhNode { lo, hi, a: 0, b: 0, count: 0 });
        self.nodes[idx as usize].a = self.build_node(boxes, first, mid);
        self.nodes[idx as usize].b = self.build_node(boxes, mid, last);
        idx
    }

    /// How many items the tree was built over.
    pub fn len(&self) -> usize {
        self.items.len()
    }

    /// Whether the tree holds no items.
    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }

    /// The root node's box, or `None` on an empty tree.
    pub fn root_box(&self) -> Option<(Vec3, Vec3)> {
        self.nodes.first().map(|n| (n.lo, n.hi))
    }

    /// The slack a bounded query adds to a distance bound so that a triangle
    /// exactly at the bound is never pruned by rounding in the box test:
    /// 1e-9 scaled by 1, the root box's diagonal and how far `p` sits from
    /// the origin.
    pub fn guard(&self, p: Vec3) -> Scalar {
        let m = p.x.abs().max(p.y.abs()).max(p.z.abs());
        match self.nodes.first() {
            Some(n) => 1e-9 * (1.0 + (n.hi - n.lo).mag() + m),
            None => 1e-9 * (1.0 + m),
        }
    }

    /// Visit the items whose boxes can touch the ball of radius `radius0`
    /// about `p` - and everything a shrinking bound still allows: each call
    /// `visit(item)` makes RETURNS the bound's new value, so the caller
    /// tightens the search as better items come in. Nodes are skipped with
    /// their whole subtree once their box sits past the current bound; at an
    /// inner node the nearer child is entered first; a leaf visits its items
    /// in ascending id.
    pub fn visit_near(&self, p: Vec3, radius0: Scalar, mut visit: impl FnMut(u32) -> Scalar) {
        if self.nodes.is_empty() {
            return;
        }
        let mut radius = radius0;
        let mut stack = vec![0u32];
        while let Some(ni) = stack.pop() {
            let n = &self.nodes[ni as usize];
            if box_distance(p, n.lo, n.hi) > radius {
                continue;
            }
            if n.count == 0 {
                let ln = &self.nodes[n.a as usize];
                let rn = &self.nodes[n.b as usize];
                // The nearer child is entered first: push the farther one so
                // it pops second.
                if box_distance(p, ln.lo, ln.hi) <= box_distance(p, rn.lo, rn.hi) {
                    stack.push(n.b);
                    stack.push(n.a);
                } else {
                    stack.push(n.a);
                    stack.push(n.b);
                }
            } else {
                for k in 0..n.count {
                    let item = self.items[n.a as usize + k as usize];
                    let (blo, bhi) = self.item_boxes[item as usize];
                    if box_distance(p, blo, bhi) > radius {
                        continue;
                    }
                    radius = visit(item);
                }
            }
        }
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
        let _guard = crate::io::contract::permissive_test_guard();
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
    #[cfg_attr(feature = "single", ignore = "fails at f32: SPEC-LIT 112.3")]
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
    fn closest_point_agrees_with_nearest_triangle() {
        let s = cube();
        let idx = match TriIndex::new(&s, 0.5) {
            Ok(i) => i,
            Err(e) => panic!("index build failed: {e}"),
        };

        // Inside, outside a face, outside an edge, outside a corner, and
        // exactly on a triangle: whichever triangle wins, the point and the
        // distance must be two views of the same measurement.
        let queries = [
            Vec3::new(0.5, 0.25, 0.5),
            Vec3::new(2.0, 0.25, 0.75),
            Vec3::new(2.0, 2.0, 0.5),
            Vec3::new(2.0, 2.0, 2.0),
            Vec3::new(1.0, 0.25, 0.75),
        ];
        for (i, p) in queries.iter().enumerate() {
            let (t, d) = idx.nearest_triangle(*p);
            let (q, t2, d2) = idx.closest_point(*p);
            assert_eq!(t2, t, "query {i}: triangle");
            assert_eq!(d2, d, "query {i}: distance must match to the bit");
            let mag = (*p - q).mag();
            assert!((mag - d).abs() < 1e-12, "query {i}: mag {mag} vs d {d}");
        }
    }

    #[test]
    fn closest_point_lands_in_the_right_voronoi_region() {
        let s = cube();
        let idx = match TriIndex::new(&s, 0.5) {
            Ok(i) => i,
            Err(e) => panic!("index build failed: {e}"),
        };

        // Face region of the x = 1 face.
        let (q, _, _) = idx.closest_point(Vec3::new(2.0, 0.25, 0.75));
        assert!((q.x - 1.0).abs() < 1e-12, "x = {}", q.x);
        assert!((q.y - 0.25).abs() < 1e-12, "y = {}", q.y);
        assert!((q.z - 0.75).abs() < 1e-12, "z = {}", q.z);

        // Edge region of x = 1, y = 1.
        let (q, _, _) = idx.closest_point(Vec3::new(2.0, 2.0, 0.5));
        assert!((q.x - 1.0).abs() < 1e-12, "x = {}", q.x);
        assert!((q.y - 1.0).abs() < 1e-12, "y = {}", q.y);
        assert!((q.z - 0.5).abs() < 1e-12, "z = {}", q.z);

        // Vertex region of the (1, 1, 1) corner.
        let (q, _, _) = idx.closest_point(Vec3::new(2.0, 2.0, 2.0));
        assert!((q.x - 1.0).abs() < 1e-12, "x = {}", q.x);
        assert!((q.y - 1.0).abs() < 1e-12, "y = {}", q.y);
        assert!((q.z - 1.0).abs() < 1e-12, "z = {}", q.z);
    }

    #[test]
    fn a_point_on_the_surface_returns_itself() {
        let s = cube();
        let idx = match TriIndex::new(&s, 0.5) {
            Ok(i) => i,
            Err(e) => panic!("index build failed: {e}"),
        };

        // (1, 0.25, 0.75) lies on the x = 1 face: the foot is the point
        // itself and the distance is zero. The barycentric reconstruction
        // rounds, so the coordinates are checked to 1e-12, not bit-exactly.
        let p = Vec3::new(1.0, 0.25, 0.75);
        let (q, _, d) = idx.closest_point(p);
        assert!((q.x - 1.0).abs() < 1e-12, "x = {}", q.x);
        assert!((q.y - 0.25).abs() < 1e-12, "y = {}", q.y);
        assert!((q.z - 0.75).abs() < 1e-12, "z = {}", q.z);
        assert!(d <= 1e-12, "d = {d}");
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

    /// A closed UV sphere - `bands` latitude bands of `segs` segments, both
    /// pole caps fan-triangulated, wound outward - as a soup.
    fn uv_sphere(c: Vec3, r: Scalar, bands: usize, segs: usize) -> Vec<SoupTri> {
        let at = |k: usize, s: usize| {
            let th = std::f64::consts::PI * k as f64 / bands as f64;
            let ph = 2.0 * std::f64::consts::PI * (s % segs) as f64 / segs as f64;
            Vec3::new(
                (c.x as f64 + r as f64 * th.sin() * ph.cos()) as Scalar,
                (c.y as f64 + r as f64 * th.sin() * ph.sin()) as Scalar,
                (c.z as f64 + r as f64 * th.cos()) as Scalar,
            )
        };
        let mut soup: Vec<SoupTri> = Vec::new();
        for k in 0..bands {
            for s in 0..segs {
                if k == 0 {
                    // North cap: fan from the pole, which is at(k, s) for
                    // every s.
                    soup.push((0u32, [at(0, s), at(1, s), at(1, s + 1)]));
                } else if k == bands - 1 {
                    // South cap: the mirrored winding.
                    soup.push((0u32, [at(bands, s), at(bands - 1, s + 1), at(bands - 1, s)]));
                } else {
                    let (a, b, d) = (at(k, s), at(k + 1, s), at(k, s + 1));
                    let cc = at(k + 1, s + 1);
                    soup.push((0u32, [a, b, d]));
                    soup.push((0u32, [b, cc, d]));
                }
            }
        }
        soup
    }

    /// The deterministic LCG the features tests use.
    fn lcg_points(n: usize, scale: f64, shift: f64) -> Vec<Vec3> {
        let mut s: u64 = 0x853C49E6748FEA9B;
        let mut pts = Vec::with_capacity(n);
        for _ in 0..n {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let u = ((s >> 11) as f64) / (1u64 << 53) as f64;
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let v = ((s >> 11) as f64) / (1u64 << 53) as f64;
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let w = ((s >> 11) as f64) / (1u64 << 53) as f64;
            pts.push(Vec3::new(
                (scale * u + shift) as Scalar,
                (scale * v + shift) as Scalar,
                (scale * w + shift) as Scalar,
            ));
        }
        pts
    }

    /// The T1/T2 query points: the cube's lattice plus the sphere's LCG
    /// cloud and its far corners.
    fn t1_points() -> (Vec<Vec3>, Vec<Vec3>) {
        let cube = (-4..=8i64)
            .flat_map(|i| {
                (-4..=8i64).flat_map(move |j| {
                    (-4..=8i64).map(move |k| {
                        Vec3::new(i as Scalar / 4.0, j as Scalar / 4.0, k as Scalar / 4.0)
                    })
                })
            })
            .collect();
        let mut sphere = lcg_points(3000, 6.0, -3.0);
        for &x in &[-10.0, 10.0] {
            for &y in &[-10.0, 10.0] {
                for &z in &[-10.0, 10.0] {
                    sphere.push(Vec3::new(x, y, z));
                }
            }
        }
        (cube, sphere)
    }

    #[test]
    fn the_bvh_index_answers_exactly_what_the_grid_answers() {
        let (cube_pts, sphere_pts) = t1_points();

        // (a) The unit cube at four cell hints, on a lattice dense enough to
        // land on faces, edges and corners.
        let cube = cube();
        let mut tie_points = 0usize;
        for &p in &cube_pts {
            let mut min_d = Scalar::INFINITY;
            let mut n_at_min = 0usize;
            for t in 0..cube.tris.len() {
                let tri = cube.tris[t];
                let q = closest_point_on_triangle(
                    p,
                    cube.points[tri[0] as usize],
                    cube.points[tri[1] as usize],
                    cube.points[tri[2] as usize],
                );
                let d = (p - q).mag();
                if d < min_d {
                    min_d = d;
                    n_at_min = 1;
                } else if d == min_d {
                    n_at_min += 1;
                }
            }
            if n_at_min >= 2 {
                tie_points += 1;
            }
        }
        assert!(tie_points > 100, "only {tie_points} lattice ties - the tie path is not exercised");

        for h in [0.25, 0.3, 0.5, 1.0] {
            let grid = TriIndex::new(&cube, h).expect("cube grid");
            let bvh = TriIndex::with_bvh(&cube, h).expect("cube bvh");
            for &p in &cube_pts {
                let (gt, gd) = grid.nearest_triangle(p);
                let (bt, bd) = bvh.nearest_triangle(p);
                assert_eq!(gt, bt, "cube hint {h}: index at ({},{},{})", p.x, p.y, p.z);
                assert_eq!(gd.to_bits(), bd.to_bits(), "cube hint {h}: distance bits");
                let (gq, gt2, gd2) = grid.closest_point(p);
                let (bq, bt2b, bd2b) = bvh.closest_point(p);
                assert_eq!(gt2, bt2b, "cube hint {h}: closest_point index");
                for (a, b) in [(gq, bq), (Vec3::new(gd2, 0.0, 0.0), Vec3::new(bd2b, 0.0, 0.0))] {
                    for ax in 0..3 {
                        assert_eq!(
                            a.component(ax).to_bits(),
                            b.component(ax).to_bits(),
                            "cube hint {h}: closest_point bits axis {ax}"
                        );
                    }
                }
            }
        }

        // (b) A sphere whose dense body sits beside nothing - the opposite
        // shape for the shell walk - at three hints, including one so coarse
        // the grid collapses to a single bucket.
        let sph_soup = uv_sphere(Vec3::new(0.3, -0.2, 0.1), 1.0, 24, 48);
        let sph = Surface::from_soup(sph_soup, vec!["sph".into()]).expect("sphere builds");
        assert!(sph.tris.len() > 2000, "the sphere must be dense: {}", sph.tris.len());
        for h in [0.05, 0.4, 3.0] {
            let grid = TriIndex::new(&sph, h).expect("sphere grid");
            let bvh = TriIndex::with_bvh(&sph, h).expect("sphere bvh");
            for &p in &sphere_pts {
                let (gt, gd) = grid.nearest_triangle(p);
                let (bt, bd) = bvh.nearest_triangle(p);
                assert_eq!(gt, bt, "sphere hint {h}: index at ({},{},{})", p.x, p.y, p.z);
                assert_eq!(gd.to_bits(), bd.to_bits(), "sphere hint {h}: distance bits");
                let (gq, gt2, gd2) = grid.closest_point(p);
                let (bq, bt2b, bd2b) = bvh.closest_point(p);
                assert_eq!(gt2, bt2b, "sphere hint {h}: closest_point index");
                assert_eq!(gd2.to_bits(), bd2b.to_bits(), "sphere hint {h}: distance bits");
                for ax in 0..3 {
                    assert_eq!(
                        gq.component(ax).to_bits(),
                        bq.component(ax).to_bits(),
                        "sphere hint {h}: point bits axis {ax}"
                    );
                }
            }
        }
    }

    #[test]
    fn nearest_distance_within_agrees_with_nearest_triangle() {
        let (cube_pts, sphere_pts) = t1_points();
        let cube = cube();
        let sph_soup = uv_sphere(Vec3::new(0.3, -0.2, 0.1), 1.0, 24, 48);
        let sph = Surface::from_soup(sph_soup, vec!["sph".into()]).expect("sphere builds");
        for (surf, pts) in [(&cube, &cube_pts), (&sph, &sphere_pts)] {
            for h in if surf.patch_names[0] == "cube" {
                vec![0.25, 1.0]
            } else {
                vec![0.05, 3.0]
            } {
                let grid = TriIndex::new(surf, h).expect("grid");
                let bvh = TriIndex::with_bvh(surf, h).expect("bvh");
                for &p in pts {
                    let (_, d) = grid.nearest_triangle(p);
                    for &r in &[0.0, 0.01, 0.1, 0.5, 2.0] {
                        for (name, idx) in [("grid", &grid), ("bvh", &bvh)] {
                            match idx.nearest_distance_within(p, r) {
                                Some(got) => {
                                    assert!(
                                        d <= r,
                                        "{name} hint {h}: Some at r={r} but nearest is {d}"
                                    );
                                    assert_eq!(
                                        got.to_bits(),
                                        d.to_bits(),
                                        "{name} hint {h}: bits at r={r}"
                                    );
                                }
                                None => {
                                    assert!(
                                        d > r,
                                        "{name} hint {h}: None at r={r} but nearest is {d}"
                                    );
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}
