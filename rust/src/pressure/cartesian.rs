// meteor-cfd - Copyright (c) 2026 주식회사 메테오시뮬레이션 (Meteo Simulation Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.

//! Recognising a uniform Cartesian box in an unstructured mesh, and the
//! per-side boundary conditions a separable solve needs.
//!
//! Nothing here trusts the cell numbering. `HostMesh` is an `lduAddressing`
//! plus geometry: it has no idea it came from a block, and a mesh read off
//! disk may be numbered any way the generator felt like. So the structure is
//! *recovered* from the geometry - face areas and cell centres - and every
//! recovered fact is then checked against the addressing. Either the mesh
//! really is a `nx*ny*nz` box of identical hexahedra, in which case there is a
//! bijection between cells and `(i,j,k)` and every internal face joins two
//! cells adjacent along one axis, or [`detect`] says why not.
//!
//! Getting this wrong would be invisible: a permutation that is a bijection
//! but not the *right* bijection still round-trips through an FFT and still
//! produces a smooth, plausible, completely wrong pressure field. Hence the
//! checks, and hence the fact that the FFT backend additionally verifies the
//! assembled matrix against the operator it thinks it is inverting.

use crate::mesh::{HostMesh, PatchKind};
use crate::{Label, Scalar, Vec3};

use super::round_off_tol;

/// The two conditions a separable direction can carry at one end.
///
/// `Neumann` covers `zeroGradient`, `fixedGradient`, a wall, a symmetry plane
/// and an `empty` patch: all four leave the *operator* alone and put whatever
/// they have to say into the source. `Dirichlet` is `fixedValue`, which does
/// change the operator - it adds `-2*gamma*magSf*deltaCoeffs` to the diagonal.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SideBc {
    Neumann,
    Dirichlet,
}

impl SideBc {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Neumann => "Neumann",
            Self::Dirichlet => "Dirichlet",
        }
    }
}

/// Names of the six sides in the order they are indexed everywhere here.
pub const SIDE_NAMES: [&str; 6] = ["-x", "+x", "-y", "+y", "-z", "+z"];

/// A recovered uniform Cartesian box.
#[derive(Debug, Clone)]
pub struct CartesianGrid {
    pub nx: usize,
    pub ny: usize,
    pub nz: usize,
    pub dx: Scalar,
    pub dy: Scalar,
    pub dz: Scalar,

    /// `cell_of[i + nx*(j + ny*k)]` is the mesh cell at `(i,j,k)`.
    pub cell_of: Vec<Label>,
    /// `cart_of[cell]` is that cell's `i + nx*(j + ny*k)`.
    pub cart_of: Vec<Label>,

    /// `[n_internal_faces]` which axis each internal face is normal to.
    pub face_axis: Vec<u8>,
    /// `[n_boundary_faces]` which of the six sides each boundary face lies on.
    pub b_side: Vec<u8>,
}

impl CartesianGrid {
    #[inline]
    pub fn n(&self) -> usize {
        self.nx * self.ny * self.nz
    }

    #[inline]
    pub fn dim(&self, axis: usize) -> usize {
        [self.nx, self.ny, self.nz][axis]
    }

    #[inline]
    pub fn spacing(&self, axis: usize) -> Scalar {
        [self.dx, self.dy, self.dz][axis]
    }

    /// `(i,j,k)` of a Cartesian index.
    #[inline]
    pub fn ijk(&self, t: usize) -> (usize, usize, usize) {
        (t % self.nx, (t / self.nx) % self.ny, t / (self.nx * self.ny))
    }

    /// The index along `axis` of a Cartesian index.
    #[inline]
    pub fn coord(&self, t: usize, axis: usize) -> usize {
        let (i, j, k) = self.ijk(t);
        [i, j, k][axis]
    }
}

/// Recover the box, or say what stopped it.
///
/// The error is a `String` rather than a crate `Error` because "this mesh is
/// not a Cartesian box" is not a failure - it is one of the two normal
/// answers, and it ends up in the decision table as the reason a backend was
/// not a candidate.
pub fn detect(hm: &HostMesh) -> std::result::Result<CartesianGrid, String> {
    let n = hm.n_cells;
    if n == 0 {
        return Err("the mesh has no cells".into());
    }
    if hm.v.len() < n || hm.c.len() < n {
        return Err("the mesh geometry has not been computed".into());
    }

    let tol = round_off_tol();

    // ---- every cell is the same hexahedron -------------------------------
    let v0 = hm.v[0];
    if !(v0 > 0.0) {
        return Err("cell 0 has non-positive volume".into());
    }
    for (c, v) in hm.v[..n].iter().enumerate() {
        if (v - v0).abs() > 1e3 * tol * v0 {
            return Err(format!(
                "cell volumes are not uniform (cell {c} is {v:.6e}, cell 0 is {v0:.6e})"
            ));
        }
    }

    // ---- every face is axis-aligned, and one area per axis ---------------
    let mut face_axis = vec![0u8; hm.n_internal_faces];
    let mut area = [None::<Scalar>; 3];

    let mut record_area = |axis: usize, a: Scalar| -> std::result::Result<(), String> {
        match area[axis] {
            None => {
                area[axis] = Some(a);
                Ok(())
            }
            Some(a0) => {
                if (a - a0).abs() > 1e3 * tol * a0 {
                    Err(format!(
                        "faces normal to {} do not all have the same area \
                         ({a:.6e} vs {a0:.6e})",
                        ["x", "y", "z"][axis]
                    ))
                } else {
                    Ok(())
                }
            }
        }
    };

    for f in 0..hm.n_internal_faces {
        let axis = axis_of(hm.sf[f], tol).ok_or_else(|| {
            format!("internal face {f} is not axis-aligned (Sf = {:?})", hm.sf[f])
        })?;
        face_axis[f] = axis as u8;
        record_area(axis, hm.mag_sf[f])?;
    }

    let mut b_side = vec![0u8; hm.n_boundary_faces];
    for bf in 0..hm.n_boundary_faces {
        let sf = hm.b_sf[bf];
        let axis = axis_of(sf, tol)
            .ok_or_else(|| format!("boundary face {bf} is not axis-aligned (Sf = {sf:?})"))?;
        let positive = [sf.x, sf.y, sf.z][axis] > 0.0;
        b_side[bf] = (2 * axis + usize::from(positive)) as u8;
        record_area(axis, hm.b_mag_sf[bf])?;
    }

    let mut d = [0.0 as Scalar; 3];
    for axis in 0..3 {
        // dx = V/(dy dz), and dy dz is exactly the area of an x-face. This is
        // the only route that also works when an axis has a single cell and
        // therefore no internal faces to measure a spacing from.
        let a = area[axis]
            .ok_or_else(|| format!("no face is normal to {}", ["x", "y", "z"][axis]))?;
        if !(a > 0.0) {
            return Err(format!("faces normal to {} have zero area", ["x", "y", "z"][axis]));
        }
        d[axis] = v0 / a;
    }
    if (d[0] * d[1] * d[2] - v0).abs() > 1e4 * tol * v0 {
        return Err(format!(
            "the recovered spacings {:.6e} x {:.6e} x {:.6e} do not multiply to the \
             cell volume {v0:.6e}",
            d[0], d[1], d[2]
        ));
    }

    // ---- cell centres land on a regular lattice --------------------------
    let mut lo = hm.c[0];
    for c in &hm.c[..n] {
        lo = lo.cmpt_min(*c);
    }

    let mut idx = vec![[0usize; 3]; n];
    let mut dim = [0usize; 3];

    for (cell, centre) in hm.c[..n].iter().enumerate() {
        let comp = [centre.x, centre.y, centre.z];
        let lo_c = [lo.x, lo.y, lo.z];
        for axis in 0..3 {
            let s = (comp[axis] - lo_c[axis]) / d[axis];
            let i = s.round();
            if (s - i).abs() > 1e4 * tol.max(1e-9) {
                return Err(format!(
                    "cell {cell}'s centre is {s:.6} spacings along {} from the corner, \
                     which is not a lattice point",
                    ["x", "y", "z"][axis]
                ));
            }
            if i < 0.0 {
                return Err(format!("cell {cell} sits before the corner along {}", ["x", "y", "z"][axis]));
            }
            let i = i as usize;
            idx[cell][axis] = i;
            dim[axis] = dim[axis].max(i + 1);
        }
    }

    let (nx, ny, nz) = (dim[0], dim[1], dim[2]);
    if nx * ny * nz != n {
        return Err(format!(
            "the lattice is {nx} x {ny} x {nz} = {} points but the mesh has {n} cells",
            nx * ny * nz
        ));
    }

    // ---- and the lattice is a bijection ----------------------------------
    let mut cell_of = vec![-1 as Label; n];
    let mut cart_of = vec![-1 as Label; n];
    for (cell, ijk) in idx.iter().enumerate() {
        let t = ijk[0] + nx * (ijk[1] + ny * ijk[2]);
        if cell_of[t] >= 0 {
            return Err(format!(
                "cells {} and {cell} both sit at lattice point ({}, {}, {})",
                cell_of[t], ijk[0], ijk[1], ijk[2]
            ));
        }
        cell_of[t] = cell as Label;
        cart_of[cell] = t as Label;
    }

    // ---- the addressing agrees with the lattice --------------------------
    for f in 0..hm.n_internal_faces {
        let o = hm.owner[f] as usize;
        let nb = hm.neighbour[f] as usize;
        if o >= n || nb >= n {
            return Err(format!("internal face {f} addresses a cell outside the mesh"));
        }
        let axis = face_axis[f] as usize;
        for b in 0..3 {
            let delta = idx[nb][b] as isize - idx[o][b] as isize;
            let want = if b == axis { 1 } else { 0 };
            if delta != want {
                return Err(format!(
                    "internal face {f} is normal to {} but joins lattice points that \
                     differ by {delta} along {}",
                    ["x", "y", "z"][axis],
                    ["x", "y", "z"][b]
                ));
            }
        }
    }

    for bf in 0..hm.n_boundary_faces {
        let c = hm.b_face_cells[bf] as usize;
        if c >= n {
            return Err(format!("boundary face {bf} addresses a cell outside the mesh"));
        }
        let side = b_side[bf] as usize;
        let axis = side / 2;
        let at_top = side % 2 == 1;
        let want = if at_top { dim[axis] - 1 } else { 0 };
        if idx[c][axis] != want {
            return Err(format!(
                "boundary face {bf} points along {}{} but its cell is at index {} of {}",
                if at_top { "+" } else { "-" },
                ["x", "y", "z"][axis],
                idx[c][axis],
                dim[axis]
            ));
        }
    }

    Ok(CartesianGrid {
        nx,
        ny,
        nz,
        dx: d[0],
        dy: d[1],
        dz: d[2],
        cell_of,
        cart_of,
        face_axis,
        b_side,
    })
}

/// The axis a face normal points along, or `None` if it points somewhere else.
fn axis_of(sf: Vec3, tol: Scalar) -> Option<usize> {
    let c = [sf.x.abs(), sf.y.abs(), sf.z.abs()];
    let mag = (c[0] * c[0] + c[1] * c[1] + c[2] * c[2]).sqrt();
    if !(mag > 0.0) {
        return None;
    }
    let axis = (0..3).max_by(|a, b| c[*a].total_cmp(&c[*b]))?;
    for b in 0..3 {
        if b != axis && c[b] > 1e4 * tol.max(1e-10) * mag {
            return None;
        }
    }
    Some(axis)
}

// ==========================================================================
//  Boundary conditions
// ==========================================================================

/// Which condition one boundary face imposes on the *operator*.
///
/// Only `fr` (the mixed form's `valueFraction`) reaches the matrix -
/// `refValue` and `refGrad` go into the source - so this is a two-way
/// classification with nothing in between. A genuinely mixed face,
/// `0 < fr < 1`, is neither and makes the direction non-separable.
fn face_bc(hm: &HostMesh, bc_kind: &[Label], fr: &[Scalar], bf: usize) -> Option<SideBc> {
    use crate::field::BcKind;

    let kind = bc_kind.get(bf).copied().unwrap_or(BcKind::ZeroGradient as Label);

    if kind == BcKind::Empty as Label
        || hm.b_kind.get(bf).copied() == Some(PatchKind::Empty as Label)
    {
        // `fvLapBoundary` returns before touching an empty face, so it adds
        // nothing to the diagonal - which is what a Neumann face does too.
        return Some(SideBc::Neumann);
    }
    if kind == BcKind::Cyclic as Label
        || hm.b_nbr_cell.get(bf).copied().unwrap_or(-1) >= 0
    {
        // Periodic separates too, with a plain C2C transform, but the FFT
        // backend here does not implement it and a coupled coefficient stays
        // inside `Amul` where none of this can see it.
        return None;
    }

    let f = fr.get(bf).copied().unwrap_or(0.0);
    if f.abs() <= 1e-12 {
        Some(SideBc::Neumann)
    } else if (f - 1.0).abs() <= 1e-12 {
        Some(SideBc::Dirichlet)
    } else {
        None
    }
}

/// The condition on each of the six sides, or `None` if any side is mixed.
pub fn side_bcs(
    hm: &HostMesh,
    grid: &CartesianGrid,
    bc_kind: &[Label],
    fr: &[Scalar],
) -> Option<[SideBc; 6]> {
    let mut sides = [None::<SideBc>; 6];
    for bf in 0..hm.n_boundary_faces {
        let s = *grid.b_side.get(bf)? as usize;
        let bc = face_bc(hm, bc_kind, fr, bf)?;
        match sides[s] {
            None => sides[s] = Some(bc),
            Some(existing) if existing == bc => {}
            Some(_) => return None,
        }
    }
    // A side with no faces at all cannot happen on a box that `detect`
    // accepted, but defaulting it to Neumann rather than panicking keeps this
    // function total.
    Some(sides.map(|s| s.unwrap_or(SideBc::Neumann)))
}

/// `(separable, why not)`.
///
/// On a recovered box the test is per SIDE, not per patch: the plume's floor
/// carries two patches - the burner window and the rest of the floor - and
/// they have to agree with each other, while a single `walls` patch spanning
/// five sides only has to be uniform within each of them. Per-side is the
/// condition that actually makes the operator separate, and it is the one
/// checked. Without a box there is no side to speak of, so the fallback is the
/// per-patch statement.
pub fn separable(
    hm: &HostMesh,
    grid: Option<&CartesianGrid>,
    bc_kind: &[Label],
    fr: &[Scalar],
) -> (bool, String) {
    match grid {
        Some(g) => {
            for bf in 0..hm.n_boundary_faces {
                if face_bc(hm, bc_kind, fr, bf).is_none() {
                    return (
                        false,
                        format!(
                            "boundary face {bf} is neither uniformly Dirichlet nor \
                             uniformly Neumann (valueFraction {:.3}{})",
                            fr.get(bf).copied().unwrap_or(0.0),
                            if hm.b_nbr_cell.get(bf).copied().unwrap_or(-1) >= 0 {
                                ", coupled"
                            } else {
                                ""
                            }
                        ),
                    );
                }
            }
            match side_bcs(hm, g, bc_kind, fr) {
                Some(_) => (true, String::new()),
                None => {
                    let s = mixed_side(hm, g, bc_kind, fr);
                    (
                        false,
                        format!(
                            "side {} carries both Dirichlet and Neumann faces",
                            SIDE_NAMES.get(s).copied().unwrap_or("?")
                        ),
                    )
                }
            }
        }
        None => {
            for p in &hm.patches {
                let mut seen: Option<SideBc> = None;
                for bf in p.start..p.start + p.size {
                    match face_bc(hm, bc_kind, fr, bf) {
                        None => {
                            return (
                                false,
                                format!("patch \"{}\" is neither Dirichlet nor Neumann", p.name),
                            )
                        }
                        Some(bc) => match seen {
                            None => seen = Some(bc),
                            Some(x) if x == bc => {}
                            Some(_) => {
                                return (
                                    false,
                                    format!("patch \"{}\" mixes Dirichlet and Neumann faces", p.name),
                                )
                            }
                        },
                    }
                }
            }
            (true, String::new())
        }
    }
}

fn mixed_side(hm: &HostMesh, g: &CartesianGrid, bc_kind: &[Label], fr: &[Scalar]) -> usize {
    let mut sides = [None::<SideBc>; 6];
    for bf in 0..hm.n_boundary_faces {
        let s = g.b_side.get(bf).copied().unwrap_or(0) as usize;
        if let Some(bc) = face_bc(hm, bc_kind, fr, bf) {
            match sides[s] {
                None => sides[s] = Some(bc),
                Some(x) if x == bc => {}
                Some(_) => return s,
            }
        }
    }
    6
}

// ==========================================================================
//  Tests
// ==========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesh::topology::tests::box_mesh;
    use crate::Vec3;

    fn built(n: [usize; 3], d: Vec3) -> HostMesh {
        let (mut m, points, faces) = box_mesh(n, d);
        m.build_cell_face_maps();
        m.compute_geometry(&points, &faces).expect("geometry");
        m
    }

    #[test]
    fn a_block_is_recognised_with_its_own_spacings() {
        let m = built([4, 3, 2], Vec3::new(0.5, 0.25, 2.0));
        let g = detect(&m).expect("should be a box");

        assert_eq!((g.nx, g.ny, g.nz), (4, 3, 2));
        assert!((g.dx - 0.5).abs() < 1e-12);
        assert!((g.dy - 0.25).abs() < 1e-12);
        assert!((g.dz - 2.0).abs() < 1e-12);
    }

    /// The permutation being a bijection is not enough - it has to be the
    /// bijection the geometry says, or the FFT solves a scrambled problem
    /// that still looks smooth.
    #[test]
    fn the_permutation_matches_the_cell_centres() {
        let d = Vec3::new(0.5, 0.25, 2.0);
        let m = built([4, 3, 2], d);
        let g = detect(&m).expect("should be a box");

        let mut seen = vec![false; g.n()];
        for t in 0..g.n() {
            let cell = g.cell_of[t] as usize;
            assert!(!seen[cell], "cell {cell} appears twice");
            seen[cell] = true;
            assert_eq!(g.cart_of[cell] as usize, t);

            let (i, j, k) = g.ijk(t);
            let c = m.c[cell];
            assert!((c.x - (i as Scalar + 0.5) * d.x).abs() < 1e-12);
            assert!((c.y - (j as Scalar + 0.5) * d.y).abs() < 1e-12);
            assert!((c.z - (k as Scalar + 0.5) * d.z).abs() < 1e-12);
        }
        assert!(seen.iter().all(|s| *s));
    }

    #[test]
    fn a_single_cell_thick_direction_still_gets_its_spacing() {
        // nz == 1 means no internal z faces, so dz can only come from the
        // volume and the z-face area.
        let m = built([3, 2, 1], Vec3::new(0.5, 0.25, 2.0));
        let g = detect(&m).expect("should be a box");
        assert_eq!((g.nx, g.ny, g.nz), (3, 2, 1));
        assert!((g.dz - 2.0).abs() < 1e-12);
    }

    #[test]
    fn every_boundary_face_lands_on_the_side_its_normal_points_at() {
        let m = built([3, 2, 2], Vec3::new(1.0, 1.0, 1.0));
        let g = detect(&m).expect("should be a box");

        let mut count = [0usize; 6];
        for bf in 0..m.n_boundary_faces {
            count[g.b_side[bf] as usize] += 1;
        }
        assert_eq!(count, [2 * 2, 2 * 2, 3 * 2, 3 * 2, 3 * 2, 3 * 2]);
    }

    #[test]
    fn a_stretched_mesh_is_rejected_and_says_why() {
        let mut m = built([3, 2, 2], Vec3::new(1.0, 1.0, 1.0));
        // One fatter cell: the volumes stop being uniform.
        m.v[5] *= 1.5;
        let why = detect(&m).expect_err("must be rejected");
        assert!(why.contains("volumes are not uniform"), "{why}");
    }

    #[test]
    fn a_rotated_face_is_rejected() {
        let mut m = built([3, 2, 2], Vec3::new(1.0, 1.0, 1.0));
        m.sf[0] = Vec3::new(0.7, 0.7, 0.0);
        let why = detect(&m).expect_err("must be rejected");
        assert!(why.contains("axis-aligned"), "{why}");
    }

    #[test]
    fn all_walls_is_all_neumann_and_separable() {
        let m = built([3, 2, 2], Vec3::new(1.0, 1.0, 1.0));
        let g = detect(&m).expect("box");
        let kind = vec![crate::field::BcKind::ZeroGradient as Label; m.n_boundary_faces];
        let fr = vec![0.0 as Scalar; m.n_boundary_faces];

        let (ok, why) = separable(&m, Some(&g), &kind, &fr);
        assert!(ok, "{why}");
        assert_eq!(
            side_bcs(&m, &g, &kind, &fr),
            Some([SideBc::Neumann; 6])
        );
    }

    #[test]
    fn one_dirichlet_side_is_still_separable() {
        let m = built([3, 2, 2], Vec3::new(1.0, 1.0, 1.0));
        let g = detect(&m).expect("box");
        let kind = vec![crate::field::BcKind::ZeroGradient as Label; m.n_boundary_faces];
        let mut fr = vec![0.0 as Scalar; m.n_boundary_faces];
        for bf in 0..m.n_boundary_faces {
            if g.b_side[bf] == 1 {
                fr[bf] = 1.0;
            }
        }

        let (ok, why) = separable(&m, Some(&g), &kind, &fr);
        assert!(ok, "{why}");
        let s = side_bcs(&m, &g, &kind, &fr).expect("sides");
        assert_eq!(s[1], SideBc::Dirichlet);
        assert_eq!(s[0], SideBc::Neumann);
    }

    #[test]
    fn half_a_dirichlet_side_is_not_separable() {
        let m = built([3, 2, 2], Vec3::new(1.0, 1.0, 1.0));
        let g = detect(&m).expect("box");
        let kind = vec![crate::field::BcKind::ZeroGradient as Label; m.n_boundary_faces];
        let mut fr = vec![0.0 as Scalar; m.n_boundary_faces];

        let mut done = false;
        for bf in 0..m.n_boundary_faces {
            if g.b_side[bf] == 1 && !done {
                fr[bf] = 1.0;
                done = true;
            }
        }

        let (ok, why) = separable(&m, Some(&g), &kind, &fr);
        assert!(!ok);
        assert!(why.contains("+x"), "{why}");
    }

    #[test]
    fn a_genuinely_mixed_face_is_not_separable() {
        let m = built([3, 2, 2], Vec3::new(1.0, 1.0, 1.0));
        let g = detect(&m).expect("box");
        let kind = vec![crate::field::BcKind::Mixed as Label; m.n_boundary_faces];
        let mut fr = vec![0.0 as Scalar; m.n_boundary_faces];
        fr[0] = 0.5;

        let (ok, why) = separable(&m, Some(&g), &kind, &fr);
        assert!(!ok);
        assert!(why.contains("valueFraction"), "{why}");
    }
}
