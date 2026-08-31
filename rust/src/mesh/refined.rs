// meteor-cfd - Copyright (c) 2026 주식회사 메테오시뮬레이션 (Meteo Simulation Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.

//! A hexahedral box mesh that is BORN with 2:1 refinement interfaces.
//!
//! Written from:
//!   ofgpu SPEC-LIT.md sections 1, 2 and 74
//!   M. J. Berger and P. Colella, "Local adaptive mesh refinement for shock
//!     hydrodynamics", J. Comput. Phys. 82 (1989) 64-84 - for what a 2:1
//!     interface IS, and for the flux-register construction that SPEC-LIT
//!     section 74.1 argues a face-based cell-centred code does not need
//!   T. Isaac, C. Burstedde and O. Ghattas, "Low-cost parallel algorithms for
//!     2:1 octree balance", IPDPS 2012, 426-437 - the balance condition,
//!     enforced here by the obvious fixed-point sweep because the base grid
//!     is small and static
//!   S. Muzaferija and D. Gosman, J. Comput. Phys. 138 (1997) 766-787 - a
//!     locally refined arbitrary-topology FV mesh is an ordinary polyhedral
//!     mesh, which is the whole reason this module produces a plain HostMesh
//! No GPL-licensed source was consulted.
//!
//! # What this is for
//!
//! This module does not adapt anything: **nothing here changes a mesh after it
//! is built.** What it does is produce, statically, the mesh an adapt would
//! have produced, so that the operators of SPEC-LIT section 3 can be measured
//! on one. `crate::adapt` (SPEC-LIT section 75) is where a mesh does change,
//! and its emitter is required to agree with this one BIT FOR BIT on every
//! mesh both can express - which is what stops the two from drifting into
//! being different mesh generators.
//!
//! The point of doing that separately, and first, is SPEC-LIT section 74: a
//! 2:1 hexahedral interface carries 25.24 degrees of non-orthogonality and a
//! relative skewness of 0.1421, and until the discretisation is shown to
//! survive those numbers, no amount of adaptation machinery is worth writing.
//!
//! # The construction
//!
//! A base grid of `n = [nx, ny, nz]` hexahedra of size `d`. Each base cell
//! carries an integer refinement level; a base cell at level `l` is split
//! isotropically into `(2^l)^3` leaf cells. Levels are 2:1 balanced across
//! FACE-adjacent base cells before anything is emitted
//! (`|l(A) - l(B)| <= 1`), by the monotone fixed-point sweep of SPEC-LIT
//! section 74.2.
//!
//! The leaf mesh is then a plain polyMesh:
//!
//! - a coarse cell's face onto a finer neighbour is **not one face**. It is
//!   the four (in 3-D) sub-faces of the finer cells, each a full face of the
//!   finer cell, each carrying its own `owner`/`neighbour` entry. The coarse
//!   cell is a polyhedron with up to 24 faces, and `sum_f Sf = 0` still holds
//!   exactly because the four sub-areas sum to the parent area exactly.
//! - **There is no hanging node to treat.** A node hanging on the coarse
//!   cell's *other* faces is simply a point that face's polygon does not list;
//!   the face is still planar and its area and centroid are still exact.
//! - **There is no flux register.** One flux per face, written to `upper[f]`
//!   and `lower[f]` for the same `f` and picked up with opposite signs by the
//!   two cells' gathers, is conservative for the same reason it is on a
//!   uniform mesh.
//!
//! # Ordering
//!
//! Cells are numbered base-cell-major (`I + nx*(J + ny*K)`), leaves within a
//! base cell in `(z, y, x)` order. That makes the `+x`, `+y` and `+z`
//! neighbour of any leaf strictly later in the numbering, so every internal
//! face this module emits already has `owner < neighbour` and the
//! upper-triangular order SPEC-LIT section 1 requires is a sort by
//! `(owner, neighbour)` and nothing more. The generator asserts it rather
//! than assuming it.

use std::collections::BTreeMap;

use crate::error::{Error, Result};
use crate::mesh::{HostMesh, PatchInfo, PatchKind};
use crate::{Label, Scalar, Vec3};

/// A statically refined box, with everything `mesh::geometry::compute` needed
/// kept alongside so a caller can re-run the sweep or write the mesh out.
pub struct RefinedBox {
    /// The mesh, with `build_cell_face_maps` and `compute_geometry` already
    /// run.
    pub mesh: HostMesh,
    /// Points, in the order `mesh`'s face point lists index them.
    pub points: Vec<Vec3>,
    /// One point list per face, internal faces first.
    pub faces: Vec<Vec<Label>>,
    /// The refinement level of every leaf cell.
    pub level: Vec<u32>,
    /// The base grid this was built from.
    pub base_n: [usize; 3],
    /// The base cell size this was built from.
    pub base_d: Vec3,
}

impl RefinedBox {
    /// The internal faces whose two cells are at different levels - the 2:1
    /// interface itself.
    pub fn interface_faces(&self) -> Vec<usize> {
        (0..self.mesh.n_internal_faces)
            .filter(|&f| {
                self.level[self.mesh.owner[f] as usize]
                    != self.level[self.mesh.neighbour[f] as usize]
            })
            .collect()
    }

    /// The largest level difference across any internal face. 2:1 balance is
    /// the statement that this is at most 1.
    pub fn max_level_jump(&self) -> u32 {
        (0..self.mesh.n_internal_faces)
            .map(|f| {
                let a = self.level[self.mesh.owner[f] as usize];
                let b = self.level[self.mesh.neighbour[f] as usize];
                a.abs_diff(b)
            })
            .max()
            .unwrap_or(0)
    }
}

/// Raise levels until every face-adjacent pair of base cells differs by at
/// most one - SPEC-LIT section 74.2.
///
/// The update is monotone (levels only rise) and integer `max` is associative
/// and exact, so the fixed point does not depend on the order the cells are
/// visited. It is reached in at most `max(level)` sweeps.
pub fn balance_2to1(n: [usize; 3], level: &mut [u32]) {
    let (nx, ny, nz) = (n[0], n[1], n[2]);
    let at = |i: usize, j: usize, k: usize| i + nx * (j + ny * k);
    loop {
        let mut changed = false;
        for k in 0..nz {
            for j in 0..ny {
                for i in 0..nx {
                    let mut want = 0u32;
                    let mut look = |a: usize, b: usize, c: usize| {
                        want = want.max(level[at(a, b, c)]);
                    };
                    if i > 0 {
                        look(i - 1, j, k);
                    }
                    if i + 1 < nx {
                        look(i + 1, j, k);
                    }
                    if j > 0 {
                        look(i, j - 1, k);
                    }
                    if j + 1 < ny {
                        look(i, j + 1, k);
                    }
                    if k > 0 {
                        look(i, j, k - 1);
                    }
                    if k + 1 < nz {
                        look(i, j, k + 1);
                    }
                    let me = &mut level[at(i, j, k)];
                    if want > 0 && *me + 1 < want {
                        *me = want - 1;
                        changed = true;
                    }
                }
            }
        }
        if !changed {
            return;
        }
    }
}

/// Build the leaf mesh of a base grid whose cells carry `level`.
///
/// `level` is 2:1 balanced first, so a caller may pass any level field it
/// likes; what comes back in `RefinedBox::level` is the balanced one, per
/// LEAF rather than per base cell.
pub fn build(n: [usize; 3], d: Vec3, level: &[u32]) -> Result<RefinedBox> {
    let (nx, ny, nz) = (n[0], n[1], n[2]);
    if nx == 0 || ny == 0 || nz == 0 {
        return Err(Error::Mesh(
            "a refined box needs at least one cell on every axis".to_string(),
        ));
    }
    if level.len() != nx * ny * nz {
        return Err(Error::Mesh(format!(
            "the level field has {} entries, but the base grid has {} cells",
            level.len(),
            nx * ny * nz
        )));
    }

    let mut lev = level.to_vec();
    balance_2to1(n, &mut lev);
    let lmax = lev.iter().copied().max().unwrap_or(0);
    if lmax > 6 {
        return Err(Error::Mesh(format!(
            "refinement level {lmax} is past this generator's limit of 6"
        )));
    }

    // ---- leaf cells, in the order the module header fixes ------------------
    let fac = 1usize << lmax; // voxels per base cell, on the finest grid
    let vn = [nx * fac, ny * fac, nz * fac];
    let vox = |i: usize, j: usize, k: usize| i + vn[0] * (j + vn[1] * k);

    // Every leaf's voxel box [lo, hi) on the finest grid.
    let mut lo: Vec<[usize; 3]> = Vec::new();
    let mut hi: Vec<[usize; 3]> = Vec::new();
    let mut cell_level: Vec<u32> = Vec::new();
    for k in 0..nz {
        for j in 0..ny {
            for i in 0..nx {
                let l = lev[i + nx * (j + ny * k)];
                let s = 1usize << l; // leaves per direction
                let step = fac >> l; // voxels per leaf
                for cc in 0..s {
                    for bb in 0..s {
                        for aa in 0..s {
                            let p = [i * fac + aa * step, j * fac + bb * step, k * fac + cc * step];
                            lo.push(p);
                            hi.push([p[0] + step, p[1] + step, p[2] + step]);
                            cell_level.push(l);
                        }
                    }
                }
            }
        }
    }
    let n_cells = lo.len();

    // Which leaf owns each finest-grid voxel. This is the whole neighbour
    // search: one O(1) lookup per voxel, no tree walk and no hash table.
    let mut owner_of = vec![-1 as Label; vn[0] * vn[1] * vn[2]];
    for (c, (a, b)) in lo.iter().zip(hi.iter()).enumerate() {
        for k in a[2]..b[2] {
            for j in a[1]..b[1] {
                for i in a[0]..b[0] {
                    owner_of[vox(i, j, k)] = c as Label;
                }
            }
        }
    }

    // ---- points, allocated on first use -----------------------------------
    let h = Vec3::new(d.x / fac as Scalar, d.y / fac as Scalar, d.z / fac as Scalar);
    let pn = [vn[0] + 1, vn[1] + 1, vn[2] + 1];
    let mut point_id = vec![-1 as Label; pn[0] * pn[1] * pn[2]];
    let mut points: Vec<Vec3> = Vec::new();

    // The four corners of an axis-normal rectangle, wound so that `Sf` points
    // along +axis. `q` is the constant coordinate; `(a0, a1)` and `(b0, b1)`
    // span the transverse directions in the order (axis+1, axis+2). That
    // reproduces `topology::tests::box_mesh`'s winding on all three axes.
    let corners = |axis: usize, q: usize, a0: usize, a1: usize, b0: usize, b1: usize| {
        let mk = |a: usize, b: usize| -> [usize; 3] {
            let mut p = [0usize; 3];
            p[axis] = q;
            p[(axis + 1) % 3] = a;
            p[(axis + 2) % 3] = b;
            p
        };
        [mk(a0, b0), mk(a1, b0), mk(a1, b1), mk(a0, b1)]
    };

    // ---- faces ------------------------------------------------------------
    let mut internal: Vec<(Label, Label, Vec<Label>)> = Vec::new();
    let mut patch_faces: Vec<Vec<(Label, Vec<Label>)>> = vec![Vec::new(); 6];

    for c in 0..n_cells {
        let (a, b) = (lo[c], hi[c]);
        for axis in 0..3 {
            let (t1, t2) = ((axis + 1) % 3, (axis + 2) % 3);

            let mut emit = |cs: [[usize; 3]; 4], points: &mut Vec<Vec3>| -> Vec<Label> {
                cs.iter()
                    .map(|p| {
                        let s = p[0] + pn[0] * (p[1] + pn[1] * p[2]);
                        if point_id[s] < 0 {
                            point_id[s] = points.len() as Label;
                            points.push(Vec3::new(
                                p[0] as Scalar * h.x,
                                p[1] as Scalar * h.y,
                                p[2] as Scalar * h.z,
                            ));
                        }
                        point_id[s]
                    })
                    .collect()
            };

            // ---- the minus side: a boundary face, or the neighbour's job ---
            if a[axis] == 0 {
                let cs = corners(axis, a[axis], a[t1], b[t1], a[t2], b[t2]);
                let mut ps = emit(cs, &mut points);
                ps.reverse(); // Sf out of the domain, i.e. along -axis
                patch_faces[2 * axis].push((c as Label, ps));
            }

            // ---- the plus side --------------------------------------------
            if b[axis] == vn[axis] {
                let cs = corners(axis, b[axis], a[t1], b[t1], a[t2], b[t2]);
                let ps = emit(cs, &mut points);
                patch_faces[2 * axis + 1].push((c as Label, ps));
                continue;
            }

            // Group the voxels of this face by the leaf on the far side. A
            // BTreeMap, so the emitted order is the neighbour's cell order
            // and never a hash order.
            let mut groups: BTreeMap<Label, [usize; 4]> = BTreeMap::new();
            for u in a[t1]..b[t1] {
                for v in a[t2]..b[t2] {
                    let mut p = [0usize; 3];
                    p[axis] = b[axis];
                    p[t1] = u;
                    p[t2] = v;
                    let nb = owner_of[vox(p[0], p[1], p[2])];
                    let e = groups.entry(nb).or_insert([u, u, v, v]);
                    e[0] = e[0].min(u);
                    e[1] = e[1].max(u);
                    e[2] = e[2].min(v);
                    e[3] = e[3].max(v);
                }
            }

            for (nb, r) in groups {
                let nb_u = nb as usize;
                // Under 2:1 balance the shared region is a full rectangle -
                // one whole face of whichever cell is finer. If it is not,
                // the mesh will not close and everything downstream is
                // invalid, so say so here rather than emit it.
                let want = (r[1] + 1 - r[0]) * (r[3] + 1 - r[2]);
                let got = (a[t1].max(lo[nb_u][t1])..b[t1].min(hi[nb_u][t1])).len()
                    * (a[t2].max(lo[nb_u][t2])..b[t2].min(hi[nb_u][t2])).len();
                if want != got {
                    return Err(Error::Mesh(format!(
                        "cells {c} and {nb_u} share a non-rectangular region on axis \
                         {axis}; the level field is not 2:1 balanced"
                    )));
                }
                if nb_u <= c {
                    return Err(Error::Mesh(format!(
                        "cell {c}'s +{axis} neighbour is {nb_u}, which is not later in \
                         the numbering; the leaf ordering this module documents is broken"
                    )));
                }
                let cs = corners(axis, b[axis], r[0], r[1] + 1, r[2], r[3] + 1);
                let ps = emit(cs, &mut points);
                internal.push((c as Label, nb, ps));
            }
        }
    }

    internal.sort_by_key(|&(o, nb, _)| (o, nb));

    // ---- assemble ---------------------------------------------------------
    let names = ["xmin", "xmax", "ymin", "ymax", "zmin", "zmax"];
    let mut faces: Vec<Vec<Label>> = Vec::with_capacity(internal.len());
    let mut owner = Vec::with_capacity(internal.len());
    let mut neighbour = Vec::with_capacity(internal.len());
    for (o, nb, fp) in internal {
        owner.push(o);
        neighbour.push(nb);
        faces.push(fp);
    }

    let mut b_face_cells = Vec::new();
    let mut patches = Vec::new();
    for (p, mut pf) in patch_faces.into_iter().enumerate() {
        pf.sort_by_key(|(c, _)| *c);
        let start = b_face_cells.len();
        let size = pf.len();
        for (c, fp) in pf {
            b_face_cells.push(c);
            faces.push(fp);
        }
        patches.push(PatchInfo {
            name: names[p].to_string(),
            type_name: "patch".to_string(),
            kind: PatchKind::Generic,
            start,
            size,
            nbr_patch: None,
        });
    }

    let mut mesh = HostMesh {
        n_cells,
        n_internal_faces: owner.len(),
        n_boundary_faces: b_face_cells.len(),
        n_points: points.len(),
        owner,
        neighbour,
        b_face_cells,
        patches,
        ..Default::default()
    };
    mesh.build_cell_face_maps();
    mesh.compute_geometry(&points, &faces)?;

    Ok(RefinedBox {
        mesh,
        points,
        faces,
        level: cell_level,
        base_n: n,
        base_d: d,
    })
}

/// The mesh SPEC-LIT section 74.5 measures on: a box whose central block is
/// `levels` finer, so the coarse-fine interface has faces, edges AND corners.
///
/// `frac` is the half-width of the refined block as a fraction of the domain,
/// measured from the centre. The block is snapped to base cells and then 2:1
/// balanced.
pub fn refined_core(n: [usize; 3], d: Vec3, frac: Scalar, levels: u32) -> Result<RefinedBox> {
    let mut level = vec![0u32; n[0] * n[1] * n[2]];
    let inside = |idx: usize, dim: usize| -> bool {
        let c = (idx as Scalar + 0.5) / dim as Scalar;
        (c - 0.5).abs() < frac
    };
    for k in 0..n[2] {
        for j in 0..n[1] {
            for i in 0..n[0] {
                if inside(i, n[0]) && inside(j, n[1]) && inside(k, n[2]) {
                    level[i + n[0] * (j + n[1] * k)] = levels;
                }
            }
        }
    }
    build(n, d, &level)
}

/// The mesh the design note analysed: one flat 2:1 interface at `x = Lx/2`,
/// the upper half a level finer, and nothing else. Everything about it is
/// symmetric, which is exactly what makes it the *easy* case - SPEC-LIT
/// section 74.5 measures both and the difference is the finding.
pub fn refined_half(n: [usize; 3], d: Vec3, levels: u32) -> Result<RefinedBox> {
    let mut level = vec![0u32; n[0] * n[1] * n[2]];
    for k in 0..n[2] {
        for j in 0..n[1] {
            for i in n[0] / 2..n[0] {
                level[i + n[0] * (j + n[1] * k)] = levels;
            }
        }
    }
    build(n, d, &level)
}

#[cfg(test)]
mod tests {
    use super::*;

    const CUBE: Vec3 = Vec3::new(0.25, 0.25, 0.25);

    /// Everything SPEC-LIT section 74.3 tabulates for a 2:1 hexahedral
    /// interface, measured on a mesh that has one.
    #[test]
    fn a_2to1_hex_interface_carries_the_numbers_section_74_tabulates() {
        let r = refined_half([4, 4, 4], CUBE, 1).unwrap();
        let m = &r.mesh;
        let h = CUBE.x;

        assert_eq!(r.max_level_jump(), 1, "the mesh is not 2:1 balanced");
        let iface = r.interface_faces();
        // 4x4 coarse cells face the refined half, each split into four.
        assert_eq!(iface.len(), 4 * 4 * 4, "interface face count");

        let mut worst_deg: Scalar = 0.0;
        for &f in &iface {
            let p = m.owner[f] as usize;
            let nb = m.neighbour[f] as usize;
            let d = m.c[nb] - m.c[p];
            let nf = m.sf[f].normalised();
            let deg = (nf.dot(d) / d.mag()).acos().to_degrees();
            worst_deg = worst_deg.max(deg);

            assert!(
                (m.weights[f] - 1.0 / 3.0).abs() < 1e-12,
                "face {f}: owner weight {} is not 1/3",
                m.weights[f]
            );
            assert!(
                (m.delta_coeffs[f] - 4.0 / (3.0 * h)).abs() < 1e-12,
                "face {f}: delta {} is not 4/(3h)",
                m.delta_coeffs[f]
            );
            assert!(
                (m.non_orth_corr[f].mag() - 0.4714045207910317).abs() < 1e-12,
                "face {f}: |k| is {}",
                m.non_orth_corr[f].mag()
            );
            let skew = m.skew_corr[f].mag() / d.mag();
            assert!(
                (skew - 0.14213381090374028).abs() < 1e-12,
                "face {f}: relative skewness {skew}"
            );
        }
        assert!(
            (worst_deg - 25.239401820678103).abs() < 1e-9,
            "the interface non-orthogonality is {worst_deg} degrees, not 25.2394"
        );

        // And every face that is NOT an interface face is orthogonal and
        // unskewed, so the numbers above are the interface's alone.
        for f in 0..m.n_internal_faces {
            if iface.contains(&f) {
                continue;
            }
            assert!(m.non_orth_corr[f].mag() < 1e-14, "face {f} is not orthogonal");
            assert!(m.skew_corr[f].mag() < 1e-14, "face {f} is skewed");
        }
    }

    /// The mesh closes: `sum_f Sf = 0` per cell, volumes sum to the box, and
    /// the faces are in the upper-triangular order SPEC-LIT section 1 wants.
    /// This is the claim that a 2:1 interface needs no flux register - four
    /// sub-areas summing to the parent area is the entire argument.
    #[test]
    fn a_refined_box_closes_and_is_ldu_ordered() {
        for r in [
            refined_half([4, 4, 4], CUBE, 1).unwrap(),
            refined_core([6, 6, 6], CUBE, 0.2, 1).unwrap(),
            refined_core([8, 6, 4], Vec3::new(0.2, 0.3, 0.5), 0.3, 2).unwrap(),
        ] {
            let rep = r.mesh.check();
            let want = r.base_n[0] as Scalar
                * r.base_n[1] as Scalar
                * r.base_n[2] as Scalar
                * r.base_d.x
                * r.base_d.y
                * r.base_d.z;
            assert!(
                (rep.total_volume - want).abs() < 1e-12 * want,
                "total volume {} vs {want}",
                rep.total_volume
            );
            assert!(
                rep.max_closure_error < 1e-14,
                "closure error {} at cell {}",
                rep.max_closure_error,
                rep.max_closure_cell
            );
            assert!(rep.ldu_ordered, "faces are not upper-triangular");
            assert!(rep.min_volume > 0.0, "a cell has non-positive volume");
            assert_eq!(r.max_level_jump(), 1, "not 2:1 balanced");
        }
    }

    /// A cell whose six neighbours are all one level finer has 24 faces, and
    /// under 2:1 balance no cell can have more. SPEC-LIT section 74.3 item 4
    /// is what the multi-colour preconditioner would pay for.
    #[test]
    fn the_cell_degree_is_bounded_by_twenty_four() {
        let r = refined_core([6, 6, 6], CUBE, 0.2, 1).unwrap();
        let m = &r.mesh;
        let mut worst = 0usize;
        for c in 0..m.n_cells {
            let deg = (m.cf_offset[c + 1] - m.cf_offset[c]) as usize
                + (m.bcf_offset[c + 1] - m.bcf_offset[c]) as usize;
            worst = worst.max(deg);
        }
        assert!(worst <= 24, "a cell has {worst} faces, past the 2:1 bound of 24");
        assert!(worst > 6, "no cell gained a face; the mesh is not refined at all");
    }

    /// 2:1 balance is a fixed point that does not depend on the sweep order,
    /// because levels only ever rise and integer `max` is exact.
    #[test]
    fn balancing_is_idempotent_and_order_independent() {
        let n = [7usize, 5, 3];
        let mut a = vec![0u32; n[0] * n[1] * n[2]];
        let at = |i: usize, j: usize, k: usize| i + n[0] * (j + n[1] * k);
        a[at(3, 2, 1)] = 3;
        let mut b = a.clone();
        b.reverse();

        balance_2to1(n, &mut a);
        let once = a.clone();
        balance_2to1(n, &mut a);
        assert_eq!(a, once, "a second balance pass changed the answer");

        // The mirrored problem must give the mirrored answer.
        balance_2to1(n, &mut b);
        b.reverse();
        assert_eq!(a, b, "balance depends on the direction it is swept");
    }
}
