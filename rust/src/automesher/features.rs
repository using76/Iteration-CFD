// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.
// Provenance: see PROVENANCE.md. No GPL-licensed source was consulted.

//! Feature edges, corners, polylines and their index - SPEC-LIT §92.12,
//! equations (92.34)-(92.36). The nearest-point map (92.28) is discontinuous
//! across a sharp edge and never returns the edge itself, so a surface snapped
//! to it alone comes back chamfered along every sharp edge by a fixed fraction
//! of the cell size. The remedy this file builds the data for is explicit
//! feature snapping: extract the sharp edges as curves, refine to them, and
//! snap the points that land near one onto the curve rather than onto the
//! faces. Of snappyHexMesh the only thing consulted was the OpenFOAM User
//! Guide's prose description of that remedy; everything here is stated
//! normatively by §92.12 and implemented from it.
//!
//! `extract` walks the welded triangulation (`Surface`, §23.1), marks an
//! undirected edge a feature when its two triangles fold further than
//! `feature_angle_deg` or when it carries any number of triangles but two -
//! an open sheet's boundary, a non-manifold junction (92.34), the edges
//! §23.2 already counts as defects - marks a point a corner by degree and by
//! the turn of the curve through it (92.35), and chains the feature edges
//! into polylines between corners, cycles closed at their lowest point.
//! `FeatureIndex` answers (92.36) - the closest point of the indexed
//! segments - through bounding-volume hierarchies over the segments' and
//! the corners' boxes, the same median-split tree `TriIndex::with_bvh`
//! builds (§23.4): a bounded walk prunes whole subtrees once their box sits
//! past the best found, and the answers stay what a linear scan with the
//! lower-id tie-break returns, bit for bit.
//!
//! What refinement (§92.2 stage 1's `l_feat` from (92.2)) and feature
//! snapping (§92.2 stage 4, consuming (92.28)'s output) then do with these
//! curves is the next unit; this file classifies, chains and queries, and
//! touches nothing else in the tree.
//!
//! Provenance: ORIGINAL - the classification, the chaining and the index are
//! this document's own, stated by SPEC-LIT §92.12 with equations (92.34),
//! (92.35), (92.36), reading (92.1) and (92.2) for where the results go.
//!
//! No GPL-licensed source was consulted.

use std::collections::{HashMap, HashSet};

use crate::error::{Error, Result};
use crate::surface::{Bvh, Surface};
use crate::{Scalar, Vec3};

// ==========================================================================
//  The feature set
// ==========================================================================

/// The sharp edges of a surface, chained into polylines - SPEC-LIT §92.12.
#[derive(Debug, Clone)]
pub struct FeatureSet {
    /// The surface's own points, copied.
    pub points: Vec<Vec3>,
    /// Feature edges as point ids, `a < b`, sorted, deduplicated.
    pub edges: Vec<[u32; 2]>,
    /// Per edge: the patch ids of its triangles, sorted and deduplicated.
    pub edge_patches: Vec<Vec<u32>>,
    /// Chains of point ids; closed iff the first id is the last.
    pub polylines: Vec<Vec<u32>>,
    /// Corner point ids, sorted.
    pub corners: Vec<u32>,
    /// The angle the set was extracted at.
    pub feature_angle_deg: Scalar,
}

impl FeatureSet {
    /// True when the surface carries no feature edge at all - the case in
    /// which the feature stages must leave a mesh exactly as they found it.
    pub fn is_empty(&self) -> bool {
        self.edges.is_empty()
    }

    /// The empty set at `feature_angle_deg` - what stage 1 carries when no
    /// `refinement.levels[]` entry asks for feature refinement, so the walk
    /// over a 200 000-triangle surface is not run for an answer nothing
    /// reads.
    ///
    /// Written by the supervising session, not by the coding agent.
    pub fn empty(feature_angle_deg: Scalar) -> FeatureSet {
        FeatureSet {
            points: Vec::new(),
            edges: Vec::new(),
            edge_patches: Vec::new(),
            polylines: Vec::new(),
            corners: Vec::new(),
            feature_angle_deg,
        }
    }

    /// A line for a run log, in the report summaries' style.
    pub fn summary(&self) -> String {
        format!(
            "features: {} edge(s), {} corner(s), {} polyline(s) ({} closed) at {:.1} deg\n",
            self.edges.len(),
            self.corners.len(),
            self.polylines.len(),
            self.n_closed_polylines(),
            self.feature_angle_deg,
        )
    }

    /// How many of the polylines are closed loops.
    pub fn n_closed_polylines(&self) -> usize {
        self.polylines
            .iter()
            .filter(|p| p.len() >= 2 && p.first() == p.last())
            .count()
    }
}

/// The undirected edge key: `(min, max)`, so the map is independent of the
/// order the triangles arrived in.
fn edge_key(a: u32, b: u32) -> (u32, u32) {
    if a < b {
        (a, b)
    } else {
        (b, a)
    }
}

// ==========================================================================
//  Extraction - (92.34) and (92.35)
// ==========================================================================

/// Extract the feature edges and corners of a welded triangulation at
/// `feature_angle_deg` - SPEC-LIT §92.12, equations (92.34) and (92.35).
/// A non-finite angle, or one outside (0, 180), is refused by name.
pub fn extract(surf: &Surface, feature_angle_deg: Scalar) -> Result<FeatureSet> {
    if !feature_angle_deg.is_finite() || feature_angle_deg <= 0.0 || feature_angle_deg >= 180.0 {
        return Err(Error::Mesh(format!(
            "feature_angle_deg must be finite and inside (0, 180), got {feature_angle_deg}"
        )));
    }

    // (92.34): the undirected edge -> incident-triangle map, built once.
    let mut incident: HashMap<(u32, u32), Vec<u32>> = HashMap::new();
    for (t, tri) in surf.tris.iter().enumerate() {
        for k in 0..3 {
            incident
                .entry(edge_key(tri[k], tri[(k + 1) % 3]))
                .or_default()
                .push(t as u32);
        }
    }

    // A feature when the two normals fold further than the angle, in
    // degrees; a feature unconditionally when the edge carries any other
    // number of triangles - one is an open sheet's boundary, three or more
    // a non-manifold junction. Normals come from the surface, recomputed
    // from the winding, never from the file.
    let mut keys: Vec<(u32, u32)> = incident.keys().copied().collect();
    keys.sort_unstable();
    let mut edges: Vec<[u32; 2]> = Vec::new();
    let mut edge_patches: Vec<Vec<u32>> = Vec::new();
    for key in keys {
        let tris = &incident[&key];
        let feature = if tris.len() == 2 {
            let dot = surf.normals[tris[0] as usize].dot(surf.normals[tris[1] as usize]);
            dot.clamp(-1.0, 1.0).acos().to_degrees() > feature_angle_deg
        } else {
            true
        };
        if feature {
            let mut patches: Vec<u32> = tris.iter().map(|&t| surf.tri_patch[t as usize]).collect();
            patches.sort_unstable();
            patches.dedup();
            edges.push([key.0, key.1]);
            edge_patches.push(patches);
        }
    }

    // (92.35): the graph on the feature edges - degree, and the turn the
    // curve makes through each of its points.
    let mut nbrs: HashMap<u32, Vec<u32>> = HashMap::new();
    for &[a, b] in &edges {
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
            // Branching or ending: no single tangent exists.
            true
        } else {
            let xv = surf.points[v as usize];
            let ua = surf.points[ns[0] as usize] - xv;
            let ub = surf.points[ns[1] as usize] - xv;
            let la = ua.mag();
            let lb = ub.mag();
            if !(la > 0.0) || !(lb > 0.0) {
                // A zero-length edge cannot happen on a welded surface;
                // refuse to divide by zero and call the point a corner.
                true
            } else {
                let d = (ua.dot(ub) / (la * lb)).clamp(-1.0, 1.0);
                // turn = 180 - the angle at v; a straight-through point
                // has ua.ub = -1 and turn 0.
                180.0 - d.acos().to_degrees() > feature_angle_deg
            }
        };
        if corner {
            corners.push(v);
        }
    }
    let corner_set: HashSet<u32> = corners.iter().copied().collect();

    // The polylines: maximal chains whose interior points are degree-2
    // non-corners. Walk from every corner along each of its feature edges,
    // consuming edges as you go, until another corner arrives.
    let mut edge_at: HashMap<(u32, u32), usize> = HashMap::new();
    for (i, e) in edges.iter().enumerate() {
        edge_at.insert((e[0], e[1]), i);
    }
    let mut consumed = vec![false; edges.len()];
    let mut polylines: Vec<Vec<u32>> = Vec::new();
    for &c in &corners {
        for &n in &nbrs[&c] {
            let first = edge_at[&edge_key(c, n)];
            if consumed[first] {
                continue;
            }
            consumed[first] = true;
            let mut chain = vec![c, n];
            let mut prev = c;
            let mut cur = n;
            while !corner_set.contains(&cur) {
                if chain.len() > edges.len() {
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
    for e in 0..edges.len() {
        if consumed[e] {
            continue;
        }
        let start = edges[e][0];
        consumed[e] = true;
        let mut chain = vec![start];
        let mut prev = start;
        let mut cur = edges[e][1];
        while cur != start {
            if chain.len() > edges.len() {
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

    // Every feature edge lies on exactly one polyline - checked, not
    // assumed, with the mismatch named.
    let unused = consumed.iter().filter(|&&c| !c).count();
    let laid: usize = polylines.iter().map(|p| p.len() - 1).sum();
    if unused != 0 || laid != edges.len() {
        return Err(Error::Mesh(format!(
            "feature chaining put {laid} of {} feature edge(s) on polylines, {unused} unconsumed",
            edges.len()
        )));
    }

    // Emitted by first point id, then by second, so running extract twice
    // on the same surface gives the same order whatever the walk found.
    polylines.sort_by(|p, q| (p[0], p[1]).cmp(&(q[0], q[1])));

    Ok(FeatureSet {
        points: surf.points.clone(),
        edges,
        edge_patches,
        polylines,
        corners,
        feature_angle_deg,
    })
}

// ==========================================================================
//  The index - (92.36)
// ==========================================================================

/// The feature edges indexed for closest-point queries - SPEC-LIT §92.12,
/// equation (92.36), through two bounding-volume hierarchies: one over the
/// indexed segments' boxes, one over the corners they carry. Both walks
/// answer exactly what a linear scan with the lower-id tie-break answers.
pub struct FeatureIndex<'f> {
    fs: &'f FeatureSet,
    /// Edge ids - indices into `fs.edges` - this index carries, ascending.
    edge_ids: Vec<usize>,
    /// `edge_ids` position -> box of the segment's two end points; the
    /// hierarchy's items.
    edge_bvh: Bvh,
    /// `fs.corners` the indexed edges carry, ascending.
    corners: Vec<u32>,
    /// Position -> the corner point itself, as a degenerate box.
    corner_bvh: Bvh,
}

impl<'f> FeatureIndex<'f> {
    /// Every feature edge, bucketed at ~`cell_hint`.
    pub fn new(fs: &'f FeatureSet, cell_hint: Scalar) -> Result<FeatureIndex<'f>> {
        let edge_ids = (0..fs.edges.len()).collect();
        Self::build(fs, edge_ids, cell_hint)
    }

    /// Only the edges whose `edge_patches` carries `patch` - the per-patch
    /// edge set the next unit's refinement band filters to.
    pub fn for_patch(fs: &'f FeatureSet, cell_hint: Scalar, patch: u32) -> Result<FeatureIndex<'f>> {
        let edge_ids = (0..fs.edges.len())
            .filter(|&e| fs.edge_patches[e].contains(&patch))
            .collect();
        Self::build(fs, edge_ids, cell_hint)
    }

    fn build(
        fs: &'f FeatureSet,
        edge_ids: Vec<usize>,
        cell_hint: Scalar,
    ) -> Result<FeatureIndex<'f>> {
        if !(cell_hint > 0.0) {
            return Err(Error::Mesh(format!(
                "FeatureIndex cell size must be positive, got {cell_hint}"
            )));
        }
        let edge_bvh = Bvh::build(
            &edge_ids
                .iter()
                .map(|&e| {
                    let a = fs.points[fs.edges[e][0] as usize];
                    let b = fs.points[fs.edges[e][1] as usize];
                    (a.cmpt_min(b), a.cmpt_max(b))
                })
                .collect::<Vec<(Vec3, Vec3)>>(),
        );

        // Corners are useful only where the indexed edges carry them. The
        // membership is taken through a set of the indexed endpoints and not
        // by scanning the edges once per corner: a site's geometry carries
        // both by the hundred thousand, and the scan is quadratic in them.
        //
        // Written by the supervising session, not by the coding agent.
        let mut carried: HashSet<u32> = HashSet::with_capacity(2 * edge_ids.len());
        for &e in &edge_ids {
            carried.insert(fs.edges[e][0]);
            carried.insert(fs.edges[e][1]);
        }
        let corners: Vec<u32> = fs
            .corners
            .iter()
            .copied()
            .filter(|c| carried.contains(c))
            .collect();
        let corner_bvh = Bvh::build(
            &corners
                .iter()
                .map(|&c| {
                    let p = fs.points[c as usize];
                    (p, p)
                })
                .collect::<Vec<(Vec3, Vec3)>>(),
        );

        Ok(FeatureIndex { fs, edge_ids, edge_bvh, corners, corner_bvh })
    }

    /// (92.36): the closest point of the indexed edges to `p`, its distance,
    /// and the edge id - an index into `fs.edges`, not into the filtered
    /// list `for_patch` built. `None` when the index carries no edge.
    ///
    /// A walk of the segments' hierarchy, the bound tightening to the best
    /// squared distance found; a pruned segment's distance exceeds the best
    /// found strictly, so no tie can hide behind a prune and the answer is
    /// the linear scan's: the minimum squared distance, ties to the LOWER
    /// edge id - the same tie-break the unit test's scan uses.
    pub fn closest_edge_point(&self, p: Vec3) -> Option<(Vec3, Scalar, usize)> {
        self.edge_nearest(p, Scalar::INFINITY)
            .map(|(d2, e)| {
                let (q, _) = self.project(p, e);
                (q, d2.sqrt(), e)
            })
    }

    /// The walk both edge queries share: the minimum `(squared distance,
    /// edge id)` with the lower-id tie-break, searched only where the bound
    /// `bound` allows - `INFINITY` for the unbounded query, a radius for
    /// [`FeatureIndex::closest_edge_within`]. A pruned item's distance is
    /// `> sqrt(bd) + guard` with `bd` the best so far, so it can neither
    /// beat nor tie; every item at the true minimum is visited.
    fn edge_nearest(&self, p: Vec3, bound: Scalar) -> Option<(Scalar, usize)> {
        if self.edge_ids.is_empty() {
            return None;
        }
        let g = self.edge_bvh.guard(p);
        let mut best: Option<(Scalar, usize)> = None;
        self.edge_bvh.visit_near(p, bound + g, |item| {
            let e = self.edge_ids[item as usize];
            let (_, d2) = self.project(p, e);
            let take = match best {
                None => true,
                Some((bd, be)) => d2 < bd || (d2 == bd && e < be),
            };
            if take {
                best = Some((d2, e));
            }
            let bd = best.map_or(Scalar::INFINITY, |(bd, _)| bd);
            bd.sqrt().min(bound) + g
        });
        best
    }

    /// (92.36) under a bound: the closest point of the indexed edges to `p`
    /// whenever [`FeatureIndex::closest_edge_point`] answers within `r` of
    /// it - all three values bit for bit its answer - and `None` when the
    /// nearest edge sits past `r`. What the refinement bands (§92.2) need:
    /// they ask whether a leaf is within its own cell size of a feature
    /// edge, never how far past it the nearest edge is.
    pub fn closest_edge_within(
        &self,
        p: Vec3,
        r: Scalar,
    ) -> Option<(Vec3, Scalar, usize)> {
        self.edge_nearest(p, r).and_then(|(d2, e)| {
            let d = d2.sqrt();
            if d <= r {
                let (q, _) = self.project(p, e);
                Some((q, d, e))
            } else {
                None
            }
        })
    }

    /// (92.36): the closest point of feature edge `e` to `p` and the
    /// squared distance to it - the segment parameter clamped into its
    /// own length.
    fn project(&self, p: Vec3, e: usize) -> (Vec3, Scalar) {
        let [a, b] = self.fs.edges[e];
        let xa = self.fs.points[a as usize];
        let xb = self.fs.points[b as usize];
        let ab = xb - xa;
        let denom = ab.mag_sqr();
        let t = if denom > 0.0 {
            ((p - xa).dot(ab) / denom).clamp(0.0, 1.0)
        } else {
            0.0
        };
        let q = xa + ab * t;
        let d = p - q;
        (q, d.mag_sqr())
    }

    /// The nearest corner the indexed edges carry: its position, its
    /// distance, its point id. A walk of the corners' hierarchy with the
    /// same tightening bound the segments' walk uses - a pruned corner sits
    /// strictly past the best found, so the answer is the linear scan's:
    /// the minimum squared distance, ties to the lower point id.
    pub fn closest_corner(&self, p: Vec3) -> Option<(Vec3, Scalar, u32)> {
        if self.corners.is_empty() {
            return None;
        }
        let g = self.corner_bvh.guard(p);
        let mut best: Option<(Scalar, u32)> = None;
        self.corner_bvh.visit_near(p, Scalar::INFINITY, |item| {
            let c = self.corners[item as usize];
            let d = p - self.fs.points[c as usize];
            let d2 = d.mag_sqr();
            let take = match best {
                None => true,
                Some((bd, bc)) => d2 < bd || (d2 == bd && c < bc),
            };
            if take {
                best = Some((d2, c));
            }
            let bd = best.map_or(Scalar::INFINITY, |(bd, _)| bd);
            bd.sqrt() + g
        });
        best.map(|(d2, c)| (self.fs.points[c as usize], d2.sqrt(), c))
    }

    /// How many edges the index carries.
    pub fn n_edges(&self) -> usize {
        self.edge_ids.len()
    }

    /// How many corners the indexed edges carry.
    pub fn n_corners(&self) -> usize {
        self.corners.len()
    }
}

// ==========================================================================
//  Tests
// ==========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::automesher::castellate::tests::{box_soup, sphere_soup};

    /// A closed cylinder about the z axis: `n` side quads as `2n`
    /// triangles, both caps fan-triangulated from a centre vertex, wound
    /// OUTWARD. The side seams fold by 360/n degrees and the caps are
    /// flat, so at any feature angle above 360/n exactly the two rims are
    /// features.
    fn cylinder_soup(r: f64, h: f64, n: usize, c: [f64; 3]) -> Vec<(u32, [Vec3; 3])> {
        let z0 = c[2] - 0.5 * h;
        let z1 = c[2] + 0.5 * h;
        let pt = |k: usize, z: f64| {
            let th = 2.0 * std::f64::consts::PI * (k % n) as f64 / n as f64;
            Vec3::new(
                (c[0] + r * th.cos()) as Scalar,
                (c[1] + r * th.sin()) as Scalar,
                z as Scalar,
            )
        };
        let mut soup: Vec<(u32, [Vec3; 3])> = Vec::new();
        let bottom = Vec3::new(c[0] as Scalar, c[1] as Scalar, z0 as Scalar);
        let top = Vec3::new(c[0] as Scalar, c[1] as Scalar, z1 as Scalar);
        for k in 0..n {
            let a0 = pt(k, z0);
            let b0 = pt(k + 1, z0);
            let a1 = pt(k, z1);
            let b1 = pt(k + 1, z1);
            soup.push((0, [a0, b0, b1]));
            soup.push((0, [a0, b1, a1]));
            soup.push((0, [bottom, b0, a0]));
            soup.push((0, [top, a1, b1]));
        }
        soup
    }

    fn cube() -> Surface {
        Surface::from_soup(box_soup([1.0; 3], [3.0; 3]), vec!["box".into()])
            .expect("the cube soup builds")
    }

    #[test]
    fn a_cube_gives_twelve_edges_and_eight_corners() {
        let surf = cube();
        let fs = extract(&surf, 30.0).expect("a cube extracts");
        assert_eq!(fs.edges.len(), 12);
        assert_eq!(fs.corners.len(), 8);
        assert_eq!(fs.polylines.len(), 12);
        assert!(fs.polylines.iter().all(|p| p.len() == 2), "cube polylines are single edges");
        assert!(!fs.is_empty());
        for &c in &fs.corners {
            let carried = fs.edges.iter().filter(|e| e.contains(&c)).count();
            assert_eq!(carried, 3, "corner {c} carries 3 feature edges");
        }
    }

    #[test]
    fn the_cube_is_the_same_at_five_and_at_eighty_five_degrees() {
        let surf = cube();
        for angle in [5.0, 85.0] {
            let fs = extract(&surf, angle).expect("the fold is 90 degrees");
            assert_eq!(fs.edges.len(), 12, "edges at {angle}");
            assert_eq!(fs.corners.len(), 8, "corners at {angle}");
            assert_eq!(fs.polylines.len(), 12, "polylines at {angle}");
        }
    }

    #[test]
    fn a_capped_cylinder_gives_exactly_its_two_rims() {
        let surf = Surface::from_soup(cylinder_soup(1.0, 2.0, 32, [0.0; 3]), vec!["cyl".into()])
            .expect("the cylinder soup builds");
        let fs = extract(&surf, 30.0).expect("the cylinder extracts");
        assert_eq!(fs.edges.len(), 64, "the two rims, 32 edges each");
        assert_eq!(fs.polylines.len(), 2);
        assert_eq!(fs.n_closed_polylines(), 2);
        assert!(fs.polylines.iter().all(|p| p.first() == p.last()), "both rims closed");
        assert_eq!(fs.corners.len(), 0);
    }

    #[test]
    fn a_sphere_has_no_feature_edges() {
        let surf = Surface::from_soup(sphere_soup(3.0, [4.0; 3]), vec!["sph".into()])
            .expect("the sphere soup builds");
        let fs = extract(&surf, 30.0).expect("a smooth sphere extracts");
        assert!(fs.is_empty());
        assert_eq!(fs.edges.len(), 0);
        assert_eq!(fs.corners.len(), 0);
        assert_eq!(fs.polylines.len(), 0);
        let idx = FeatureIndex::new(&fs, 1.0).expect("an empty index builds");
        let p = Vec3::new(4.0, 4.0, 4.0);
        assert!(idx.closest_edge_point(p).is_none());
        assert!(idx.closest_corner(p).is_none());
    }

    #[test]
    fn an_open_sheets_boundary_is_a_feature() {
        let v = [
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(1.0, 1.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
        ];
        let soup = vec![(0u32, [v[0], v[1], v[2]]), (0u32, [v[0], v[2], v[3]])];
        let surf = Surface::from_soup(soup, vec!["sheet".into()]).expect("the sheet builds");
        let fs = extract(&surf, 30.0).expect("the sheet extracts");
        assert_eq!(fs.edges.len(), 4, "the four boundary edges, not the diagonal");
        assert_eq!(fs.corners.len(), 4, "each square corner turns 90 degrees");
        assert_eq!(fs.polylines.len(), 4);
        let diagonal = edge_key(0, 2);
        assert!(
            !fs.edges.iter().any(|e| (e[0], e[1]) == diagonal),
            "the flat shared diagonal is not a feature"
        );
    }

    /// A linear scan over `ids` - the reference the index is held to.
    /// Lexicographic minimum of (squared distance, edge id), the same
    /// tie-break the index uses, so the two agree exactly on ties.
    fn scan_edges(fs: &FeatureSet, p: Vec3, ids: &[usize]) -> (f64, usize, Vec3) {
        let mut best: Option<(f64, usize, Vec3)> = None;
        for &e in ids {
            let [a, b] = fs.edges[e];
            let xa = fs.points[a as usize];
            let xb = fs.points[b as usize];
            let ab = xb - xa;
            let denom = ab.mag_sqr() as f64;
            let t = ((p - xa).dot(ab) as f64 / denom).clamp(0.0, 1.0);
            let q = xa + ab * (t as Scalar);
            let d2 = (p - q).mag_sqr() as f64;
            let take = match best {
                None => true,
                Some((bd, be, _)) => d2 < bd || (d2 == bd && e < be),
            };
            if take {
                best = Some((d2, e, q));
            }
        }
        let (d2, e, q) = best.expect("at least one edge to scan");
        (d2.sqrt(), e, q)
    }

    #[test]
    fn the_index_agrees_with_a_linear_scan() {
        let surf = cube();
        let fs = extract(&surf, 30.0).expect("a cube extracts");
        let idx = FeatureIndex::new(&fs, 0.25).expect("the cube index builds");
        let ids: Vec<usize> = (0..fs.edges.len()).collect();

        // A deterministic LCG, spread over a box that swallows the cube.
        let mut s: u64 = 0x853C49E6748FEA9B;
        let next = |s: &mut u64| -> f64 {
            *s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((*s >> 11) as f64) / (1u64 << 53) as f64
        };
        let mut pts: Vec<Vec3> = Vec::new();
        for _ in 0..300 {
            pts.push(Vec3::new(
                (4.0 * next(&mut s)) as Scalar,
                (4.0 * next(&mut s)) as Scalar,
                (4.0 * next(&mut s)) as Scalar,
            ));
        }
        // Beyond a segment's end, exactly on a segment, at a corner, and
        // far outside the grid, where the query must not clamp first.
        pts.push(Vec3::new(1.0, 1.0, 3.5));
        pts.push(Vec3::new(0.5, 1.0, 1.0));
        pts.push(Vec3::new(1.0, 1.75, 1.0));
        pts.push(Vec3::new(2.0, 1.0, 1.0));
        pts.push(Vec3::new(1.0, 1.0, 1.0));
        pts.push(Vec3::new(-2.0, -2.0, -2.0));
        pts.push(Vec3::new(10.0, 10.0, 10.0));

        let tol = 1e-9 * 4.0; // relative to the box size
        for p in &pts {
            let (q, dist, e) = idx.closest_edge_point(*p).expect("the cube carries edges");
            let (rd, re, rq) = scan_edges(&fs, *p, &ids);
            assert_eq!(e, re, "edge id at ({},{},{})", p.x, p.y, p.z);
            assert!(
                (dist as f64 - rd).abs() <= tol,
                "distance at ({},{},{}): {dist} vs {rd}",
                p.x, p.y, p.z
            );
            for ax in 0..3 {
                assert!(
                    (q.component(ax) as f64 - rq.component(ax) as f64).abs() <= tol,
                    "position axis {ax} at ({},{},{})",
                    p.x, p.y, p.z
                );
            }
        }
    }

    #[test]
    fn for_patch_sees_only_its_own_patch() {
        let mut soup = box_soup([0.0; 3], [2.0; 3]);
        for (_, t) in box_soup([10.0; 3], [12.0; 3]) {
            soup.push((1, t));
        }
        let surf = Surface::from_soup(soup, vec!["near".into(), "far".into()])
            .expect("the two-box surface builds");
        let fs = extract(&surf, 30.0).expect("two boxes extract");
        assert_eq!(fs.edges.len(), 24);
        assert_eq!(fs.corners.len(), 16);
        // The first box's points welded first, so its edges sort first.
        let near: Vec<usize> = (0..12).collect();
        let far: Vec<usize> = (12..24).collect();
        assert!(
            fs.edge_patches[..12].iter().all(|ps| ps.as_slice() == [0].as_slice()),
            "the first box's edges carry patch 0"
        );

        let all = FeatureIndex::new(&fs, 0.5).expect("the full index builds");
        let patch0 = FeatureIndex::for_patch(&fs, 0.5, 0).expect("the patch index builds");
        assert_eq!(patch0.n_edges(), 12);
        assert_eq!(patch0.n_corners(), 8);

        // A point inside the second box: the full index answers with a
        // second-box edge, for_patch with the first box's, and that answer
        // is the true nearest first-box edge.
        let p = Vec3::new(11.0, 11.0, 11.0);
        let (_, d_all, e_all) = all.closest_edge_point(p).expect("edges exist");
        let (q0, d0, e0) = patch0.closest_edge_point(p).expect("patch edges exist");
        assert!(e_all >= 12, "the full index answers with the second box first");
        assert!(e0 < 12, "for_patch answers with the first box's edge");
        assert!(d_all < d0, "the second box's own edge is the nearer one");
        let (rd, re, rq) = scan_edges(&fs, p, &near);
        assert_eq!(e0, re);
        assert!((d0 as f64 - rd).abs() <= 1e-9 * 4.0);
        for ax in 0..3 {
            assert!((q0.component(ax) as f64 - rq.component(ax) as f64).abs() <= 1e-9 * 4.0);
        }
        // And the far edges are still there, under their own patch.
        let patch1 = FeatureIndex::for_patch(&fs, 0.5, 1).expect("the far index builds");
        assert_eq!(patch1.n_edges(), 12);
        assert_eq!(patch1.n_corners(), 8);
        let (_, _, e1) = patch1.closest_edge_point(p).expect("far edges exist");
        let (_, re1, _) = scan_edges(&fs, p, &far);
        assert_eq!(e1, re1);
        assert!(e1 >= 12);
    }

    #[test]
    fn extraction_is_deterministic() {
        let surf = Surface::from_soup(cylinder_soup(1.0, 1.0, 16, [0.0; 3]), vec!["cyl".into()])
            .expect("the cylinder soup builds");
        let a = extract(&surf, 30.0).expect("first extraction");
        let b = extract(&surf, 30.0).expect("second extraction");
        assert_eq!(a.edges, b.edges, "edges");
        assert_eq!(a.edge_patches, b.edge_patches, "edge patches");
        assert_eq!(a.corners, b.corners, "corners");
        assert_eq!(a.polylines, b.polylines, "polylines");
        assert_eq!(a.feature_angle_deg, b.feature_angle_deg, "angle");
    }

    #[test]
    fn a_bad_angle_is_refused_by_name() {
        let surf = cube();
        for bad in [0.0, 180.0, -1.0, 360.0, Scalar::NAN, Scalar::INFINITY] {
            let err = extract(&surf, bad).expect_err("the angle is refused");
            let msg = format!("{err}");
            assert!(
                msg.contains("feature_angle_deg"),
                "the refusal names the angle: {msg}"
            );
        }
    }

    #[test]
    fn the_feature_bvh_answers_exactly_what_a_linear_scan_answers() {
        // Twenty-seven flush cubes on patch 0 (a slab of shared faces, so
        // the extract sees open-sheet and junction edges too) plus a capped
        // cylinder on patch 1.
        let mut soup = Vec::new();
        for k in 0..3usize {
            for j in 0..3usize {
                for i in 0..3usize {
                    let lo = [1.0 + i as f64, 1.0 + j as f64, 1.0 + k as f64];
                    let hi = [lo[0] + 0.5, lo[1] + 0.5, lo[2] + 0.5];
                    for (_, t) in box_soup(lo, hi) {
                        soup.push((0, t));
                    }
                }
            }
        }
        for (_, t) in cylinder_soup(1.0, 2.0, 32, [5.0, 5.0, 5.0]) {
            soup.push((1, t));
        }
        let surf = Surface::from_soup(soup, vec!["cubes".into(), "cyl".into()])
            .expect("the two-patch surface builds");
        let fs = extract(&surf, 30.0).expect("the soup extracts");
        assert!(fs.edges.len() > 100, "{} edges - too thin a test", fs.edges.len());

        // A deterministic LCG spread over a box that swallows both shapes,
        // plus every corner point and every edge midpoint - the exact-tie
        // sites.
        let mut s: u64 = 0x853C49E6748FEA9B;
        let next = |s: &mut u64| -> f64 {
            *s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((*s >> 11) as f64) / (1u64 << 53) as f64
        };
        let mut pts: Vec<Vec3> = Vec::new();
        for _ in 0..2000 {
            pts.push(Vec3::new(
                (10.0 * next(&mut s) - 2.0) as Scalar,
                (10.0 * next(&mut s) - 2.0) as Scalar,
                (10.0 * next(&mut s) - 2.0) as Scalar,
            ));
        }
        for h in [0.1, 1.0] {
            let idx = FeatureIndex::new(&fs, h).expect("the full index builds");
            let patch1 = FeatureIndex::for_patch(&fs, h, 1).expect("the patch index builds");
            for index in [&idx, &patch1] {
                pts.extend(fs.corners.iter().map(|&c| fs.points[c as usize]));
                pts.extend(index.edge_ids.iter().map(|&e| {
                    let [a, b] = fs.edges[e];
                    Vec3::new(
                        0.5 * (fs.points[a as usize].x + fs.points[b as usize].x),
                        0.5 * (fs.points[a as usize].y + fs.points[b as usize].y),
                        0.5 * (fs.points[a as usize].z + fs.points[b as usize].z),
                    )
                }));
                for p in &pts {
                    // Edges: the walk against a linear scan with the same
                    // lower-id tie-break, all three values bit for bit.
                    let (q, dist, e) = index
                        .closest_edge_point(*p)
                        .expect("the index carries edges");
                    let mut best: Option<(f64, usize, Vec3)> = None;
                    for &eid in &index.edge_ids {
                        let (rq, rd2) = index.project(*p, eid);
                        let take = match best {
                            None => true,
                            Some((bd, be, _)) => {
                                rd2 < bd || (rd2 == bd && eid < be)
                            }
                        };
                        if take {
                            best = Some((rd2, eid, rq));
                        }
                    }
                    let (rd2, re, rq) = best.expect("edges to scan");
                    assert_eq!(e, re, "edge id at ({},{},{}) h={h}", p.x, p.y, p.z);
                    assert_eq!(
                        dist.to_bits(),
                        (rd2 as Scalar).sqrt().to_bits(),
                        "distance at ({},{},{}) h={h}",
                        p.x,
                        p.y,
                        p.z
                    );
                    for ax in 0..3 {
                        assert_eq!(
                            q.component(ax).to_bits(),
                            rq.component(ax).to_bits(),
                            "position axis {ax} at ({},{},{}) h={h}",
                            p.x,
                            p.y,
                            p.z
                        );
                    }
                    // Corners: the walk against a linear scan. A patch
                    // index over rim edges only carries none.
                    if index.corners.is_empty() {
                        assert!(index.closest_corner(*p).is_none());
                        continue;
                    }
                    let (cp, cd, cc) = index
                        .closest_corner(*p)
                        .expect("the index carries corners");
                    let mut cbest: Option<(f64, u32)> = None;
                    for &cand in &index.corners {
                        let d2 = (*p - fs.points[cand as usize]).mag_sqr() as f64;
                        let take = match cbest {
                            None => true,
                            Some((bd, bc)) => d2 < bd || (d2 == bd && cand < bc),
                        };
                        if take {
                            cbest = Some((d2, cand));
                        }
                    }
                    let (cd2, cc2) = cbest.expect("corners to scan");
                    assert_eq!(cc, cc2, "corner id at ({},{},{})", p.x, p.y, p.z);
                    assert_eq!(
                        cd.to_bits(),
                        (cd2 as Scalar).sqrt().to_bits(),
                        "corner distance at ({},{},{})",
                        p.x,
                        p.y,
                        p.z
                    );
                    for ax in 0..3 {
                        assert_eq!(
                            cp.component(ax).to_bits(),
                            fs.points[cc2 as usize].component(ax).to_bits(),
                            "corner position axis {ax}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn closest_edge_within_agrees_with_closest_edge_point() {
        let surf = cube();
        let fs = extract(&surf, 30.0).expect("a cube extracts");
        let idx = FeatureIndex::new(&fs, 0.25).expect("the cube index builds");
        let mut s: u64 = 0x853C49E6748FEA9B;
        let next = |s: &mut u64| -> f64 {
            *s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((*s >> 11) as f64) / (1u64 << 53) as f64
        };
        let mut pts: Vec<Vec3> = Vec::new();
        for _ in 0..2000 {
            pts.push(Vec3::new(
                (10.0 * next(&mut s) - 2.0) as Scalar,
                (10.0 * next(&mut s) - 2.0) as Scalar,
                (10.0 * next(&mut s) - 2.0) as Scalar,
            ));
        }
        // The exact-tie sites: every corner point and every edge midpoint.
        pts.extend(fs.corners.iter().map(|&c| fs.points[c as usize]));
        pts.extend((0..fs.edges.len()).map(|e| {
            let [a, b] = fs.edges[e];
            Vec3::new(
                0.5 * (fs.points[a as usize].x + fs.points[b as usize].x),
                0.5 * (fs.points[a as usize].y + fs.points[b as usize].y),
                0.5 * (fs.points[a as usize].z + fs.points[b as usize].z),
            )
        }));
        for p in &pts {
            let full = idx.closest_edge_point(*p);
            let (_, d, _) = full.expect("the cube carries edges");
            for &r in &[0.0, 0.05, 0.3, 2.0] {
                let got = idx.closest_edge_within(*p, r);
                if d <= r {
                    let (q, dist, e) = got.expect("the nearest edge is within r");
                    let (fq, fd, fe) = full.unwrap();
                    assert_eq!(e, fe, "edge id at ({},{},{}) r={r}", p.x, p.y, p.z);
                    assert_eq!(dist.to_bits(), fd.to_bits(), "distance bits at r={r}");
                    for ax in 0..3 {
                        assert_eq!(
                            q.component(ax).to_bits(),
                            fq.component(ax).to_bits(),
                            "position bits axis {ax} at r={r}"
                        );
                    }
                } else {
                    assert!(
                        got.is_none(),
                        "Some at r={r} but the nearest edge is {d} away"
                    );
                }
            }
        }
    }

    #[test]
    fn the_module_doc_ends_with_the_provenance_sentence() {
        let src = include_str!("features.rs");
        let last = src
            .lines()
            .filter(|l| l.starts_with("//!"))
            .last()
            .expect("the module doc is present");
        assert_eq!(
            last.trim_start_matches("//!").trim(),
            "No GPL-licensed source was consulted."
        );
    }
}
