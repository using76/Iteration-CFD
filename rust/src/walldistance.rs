// meteor-cfd - Copyright (c) 2026 주식회사 메테오시뮬레이션 (Meteo Simulation Co., Ltd.)
// Source-available, not Open Source. Educational use is free; research,
// publication and commercial use require a licence - simul@msimul.com
// See LICENSE at the repository root.

//! The distance to the nearest wall, from one Poisson solve - SPEC-LIT §6.6.
//!
//! Written from:
//!   Tucker, *Applied Mathematical Modelling* 22 (1998) 293-305 - the
//!     Poisson wall distance and the algebraic step that turns the potential
//!     into a length
//!   Jasak (1996) ch. 3 - the laplacian this assembles, which is the operator
//!     of SPEC-LIT §3.2 and not a new one
//!   ofgpu `SPEC-LIT.md` §6.6, and §6.3 and §16.4 for the two things that
//!     consume the answer
//! No GPL-licensed source was consulted.
//!
//! # What is solved
//!
//! ```text
//! laplacian(phi) = -1 ,   phi = 0 on walls ,   dphi/dn = 0 everywhere else
//! y = -|grad phi| + sqrt( |grad phi|^2 + 2 phi )
//! ```
//!
//! No search, no k-d tree, no nearest-face loop: it is the machinery of §3.2
//! run once at setup, on the same matrix, the same preconditioner and the same
//! solver every other equation uses. On a GPU that matters twice over - a
//! nearest-neighbour search is the one part of a wall-distance implementation
//! that does not vectorise, and a Poisson solve is the part that does.
//!
//! # Why the formula has the shape it has
//!
//! Between two parallel walls a distance `L` apart the solution is
//! `phi = s(L - s)/2` in the wall-normal coordinate `s`, so
//! `|grad phi| = |L/2 - s|` and
//!
//! ```text
//! |grad phi|^2 + 2 phi = (L/2 - s)^2 + s(L - s) = L^2/4
//! ```
//!
//! exactly, leaving `y = L/2 - |L/2 - s| = min(s, L - s)`: the true distance,
//! identically, for every `L`. That identity is what
//! `tests::a_channel_reproduces_the_analytic_distance` measures, and it is why
//! the expression is written as the square root of a sum rather than as a
//! series in `phi`. Away from the one-dimensional limit it is an
//! approximation - one that keeps `y = 0` on the wall and `|grad y| = 1`
//! leaving it, which is what §6.3 and §16.4 need of it.
//!
//! # Which faces are walls
//!
//! `PatchKind::Wall`, from the mesh, not a field's boundary condition. The
//! wall distance is a property of the geometry: it is the same number for
//! `k`, for `omega` and for the LES filter width, and reading it off one
//! field's `boundaryField` would make it that field's opinion. A caller that
//! needs something else - an internal baffle, or a patch that is
//! geometrically a wall but should not attract the model - names its own
//! faces through [`wall_distance_from_faces`].
//!
//! # A domain with no walls at all
//!
//! Decaying isotropic turbulence in a periodic box has no wall, and the
//! Poisson problem is then all-Neumann and singular. Rather than solve it,
//! [`wall_distance`] fills `y` with [`NO_WALL`] and reports
//! `n_wall_faces == 0`. That is the physically right answer as well as the
//! safe one: SST's `F1` tends to zero as `y` grows, which is its free-shear
//! branch, and §16.4's van Driest length grows without bound, so the `min`
//! picks the geometric filter width. Both models therefore degrade to exactly
//! what a wall-free flow should get.

use crate::device::{DevBuf, Gpu};
use crate::error::{Error, Result};
use crate::field::{BcKind, GpuScalarField};
use crate::field_ops::{self, FieldKernels};
use crate::fv::{self, FvKernels, SnGradScheme};
use crate::ldu::GpuLduMatrix;
use crate::ldu_ops::{self, LduKernels};
use crate::mesh::{GpuMesh, HostMesh, PatchKind};
use crate::solver::{self, SolverKernels, SolverPerformance, SolverWorkspace};
use crate::turbulence::{wall_distance_from_potential, TurbKernels};
use crate::{Label, Scalar, Vec3};

pub use crate::io::case::SolverControls;

/// The distance reported in a domain that has no wall in it.
///
/// *DESIGN.* Any number large enough that `sqrt(k)/(beta* omega y)`,
/// `500 nu/(y^2 omega)` and `(kappa/C_delta) y` are all past the point where
/// the models stop caring will do. `1e10` m is past it for anything with a
/// mesh, and is small enough to square in double precision with room to spare
/// (`1e20` against a range of `1e308`), which matters because `y^2` appears
/// in both of SST's blending arguments.
pub const NO_WALL: Scalar = 1.0e10;

/// The wall distance, and the two things a caller does with it.
pub struct WallDistance {
    /// `y`, as a field rather than a bare buffer: it carries `y = 0` on the
    /// walls and zero-gradient elsewhere, so [`Self::grad_y`] is the gradient
    /// of something whose boundary values are the physical ones, and a driver
    /// can write it out or probe it like any other volume field.
    pub y: GpuScalarField,

    /// `grad y`. Near a wall this is the outward unit wall normal - `y` is a
    /// distance function there, so `|grad y| = 1` - which is what lets
    /// §16.4's van Driest damping find the wall-normal direction without a
    /// search.
    pub grad_y: DevBuf<Vec3>,

    /// The potential the distance came from, kept because it is the thing to
    /// look at when a distance field looks wrong.
    pub phi: GpuScalarField,

    /// How many boundary faces were treated as walls, counting only faces the
    /// matrix can feel. Zero means [`NO_WALL`] was filled in and nothing was
    /// solved.
    pub n_wall_faces: usize,

    /// The last Poisson pass's iteration count and final residual.
    pub iterations: usize,
    pub final_residual: Scalar,
}

impl WallDistance {
    /// `max(y)`, downloaded. A setup-time diagnostic; nothing in a time loop
    /// calls it.
    pub fn max(&self, gpu: &Gpu) -> Result<Scalar> {
        let v = gpu.download(&self.y.f)?;
        Ok(v.iter().copied().fold(0.0 as Scalar, Scalar::max))
    }
}

/// The wall distance, with every `PatchKind::Wall` face treated as a wall.
pub fn wall_distance(
    gpu: &Gpu,
    hm: &HostMesh,
    m: &GpuMesh,
    ctrl: &SolverControls,
    n_non_orth: usize,
) -> Result<WallDistance> {
    let is_wall: Vec<bool> = hm
        .b_kind
        .iter()
        .map(|&k| k == PatchKind::Wall as Label)
        .collect();
    wall_distance_from_faces(gpu, hm, m, &is_wall, ctrl, n_non_orth)
}

/// The wall distance with the wall faces named explicitly.
///
/// `is_wall` is indexed by *flattened boundary face*, the same indexing
/// `HostMesh::b_face_cells` uses, and must have exactly `n_boundary_faces`
/// entries - a short slice would silently drop the last patch, and a wall left
/// out of this list becomes a zero-gradient boundary, which is to say the
/// solve stops knowing it is there.
///
/// `n_non_orth` is the number of *extra* deferred passes over the laplacian's
/// non-orthogonal correction (SPEC-LIT §2.4, §12.3). Zero is exact on an
/// orthogonal mesh and first order on anything else; one or two passes is what
/// a skewed mesh wants. Each pass rebuilds the correction from the previous
/// pass's `grad phi` and costs one gradient and one solve.
pub fn wall_distance_from_faces(
    gpu: &Gpu,
    hm: &HostMesh,
    m: &GpuMesh,
    is_wall: &[bool],
    ctrl: &SolverControls,
    n_non_orth: usize,
) -> Result<WallDistance> {
    if hm.n_cells != m.n_cells || hm.n_boundary_faces != m.n_boundary_faces {
        return Err(Error::Config(format!(
            "wall_distance: the host mesh has ({}, {}) cells/boundary faces \
             and the device mesh ({}, {})",
            hm.n_cells, hm.n_boundary_faces, m.n_cells, m.n_boundary_faces
        )));
    }
    if is_wall.len() != hm.n_boundary_faces {
        return Err(Error::Config(format!(
            "wall_distance: the wall-face flag has {} entries, the mesh has \
             {} boundary faces",
            is_wall.len(),
            hm.n_boundary_faces
        )));
    }

    let fvk = FvKernels::new(gpu)?;
    let fldk = FieldKernels::new(gpu)?;
    let turb = TurbKernels::new(gpu)?;

    let mut y = GpuScalarField::zeros(gpu, m, "y")?;
    let mut phi = GpuScalarField::zeros(gpu, m, "wallDistancePotential")?;
    let mut grad_y: DevBuf<Vec3> = gpu.zeros(m.n_cells.max(1))?;

    // An `empty` face contributes nothing to any surface integral, so a wall
    // flag on one names a wall the matrix never sees. Counting the faces the
    // SOLVE will actually feel - rather than the faces the caller named - is
    // what makes the wall-free branch below trustworthy.
    let n_wall_faces = is_wall
        .iter()
        .enumerate()
        .filter(|(bf, on)| **on && hm.b_kind[*bf] != PatchKind::Empty as Label)
        .count();

    // ---- the wall-free case ----------------------------------------------
    if n_wall_faces == 0 || m.n_cells == 0 {
        field_ops::set_field(gpu, &fldk, &mut y.f, NO_WALL, m.n_cells)?;
        if m.n_boundary_faces > 0 {
            gpu.write(&mut y.bf, &vec![NO_WALL; m.n_boundary_faces])?;
        }
        gpu.fill_zero(&mut grad_y)?;
        return Ok(WallDistance {
            y,
            grad_y,
            phi,
            n_wall_faces: 0,
            iterations: 0,
            final_residual: 0.0,
        });
    }

    // ---- boundary conditions ---------------------------------------------
    // A wall is `phi = 0`; every other face keeps whatever
    // `GpuScalarField::zeros` seeded from the mesh topology, which is
    // zero-gradient on an ordinary patch, nothing at all on an `empty` one, a
    // couple on a cyclic one, and zero-gradient - the scalar meaning of
    // symmetry - on a symmetry plane. All four are already the right condition
    // for this problem, which is why they are read back rather than rebuilt.
    stamp_wall_dirichlet(gpu, &mut phi, is_wall, hm.n_boundary_faces)?;
    stamp_wall_dirichlet(gpu, &mut y, is_wall, hm.n_boundary_faces)?;

    // ---- laplacian(phi) = -1 ---------------------------------------------
    //
    // In this crate's shape - `ddt + div - laplacian + Sp psi = Su`, assembled
    // as `A psi = source` - that is `-laplacian(phi) = 1`: the laplacian with
    // `sign = -1` and a unit explicit source. Signed that way round on
    // purpose, because it leaves a positive diagonal and negative
    // off-diagonals, which is the M-matrix form PCG and the DIC
    // preconditioner both want. `gamma = 1`, so `gammaMagSf` is the mesh's own
    // `magSf` and no scratch face field is needed.
    let mut a = GpuLduMatrix::new(gpu, m)?;
    let mut ws = SolverWorkspace::for_mesh(gpu, m)?;
    let lduk = LduKernels::new(gpu)?;
    let solk = SolverKernels::new(gpu)?;

    let mut unit: DevBuf<Scalar> = gpu.zeros(m.n_cells)?;
    field_ops::set_field(gpu, &fldk, &mut unit, 1.0, m.n_cells)?;

    let mut grad_phi: DevBuf<Vec3> = gpu.zeros(m.n_cells)?;

    let solve_ctrl = SolverControls { report_residuals: true, ..*ctrl };
    let mut perf = SolverPerformance::default();

    for pass in 0..=n_non_orth {
        a.zero(gpu)?;
        fv::fvm_laplacian(gpu, &fvk, &mut a, m, &m.mag_sf, &m.b_mag_sf, &phi, -1.0)?;
        fv::fvm_su(gpu, &fvk, &mut a, m, &unit, 1.0)?;

        // The deferred correction reads the PREVIOUS pass's gradient, so
        // there is nothing to add on the first one: `phi` is still zero and
        // the term would be identically zero anyway.
        if pass > 0 {
            fv::fvc_grad_scalar(gpu, &fvk, &mut grad_phi, &phi, m)?;
            fv::fvm_laplacian_non_orth_correction(
                gpu,
                &fvk,
                &mut a,
                m,
                &m.mag_sf,
                &m.b_mag_sf,
                &phi,
                &grad_phi,
                SnGradScheme::Corrected,
                -1.0,
            )?;
        }

        ldu_ops::add_boundary_contributions(gpu, &lduk, &mut a, m)?;
        perf = solver::solve(gpu, &solk, &mut phi.f, &a, m, &mut ws, &solve_ctrl)?;
        field_ops::correct_boundary_conditions(gpu, &fldk, &mut phi, m)?;
    }

    // ---- and the distance it implies -------------------------------------
    fv::fvc_grad_scalar(gpu, &fvk, &mut grad_phi, &phi, m)?;
    wall_distance_from_potential(gpu, &turb, &mut y.f, &grad_phi, &phi.f, m.n_cells)?;

    // `y = 0` on a wall face and zero-gradient elsewhere - the same stamp the
    // potential carries, which is what makes `grad y` the wall normal rather
    // than something with a step in it at the boundary.
    field_ops::correct_boundary_conditions(gpu, &fldk, &mut y, m)?;
    fv::fvc_grad_scalar(gpu, &fvk, &mut grad_y, &y, m)?;

    Ok(WallDistance {
        y,
        grad_y,
        phi,
        n_wall_faces,
        iterations: perf.n_iterations,
        final_residual: perf.final_residual,
    })
}

/// `psi = 0` on every named wall face; every other face left as seeded.
fn stamp_wall_dirichlet(
    gpu: &Gpu,
    psi: &mut GpuScalarField,
    is_wall: &[bool],
    n_bf: usize,
) -> Result<()> {
    let mut kind = gpu.download(&psi.bc_kind)?;
    let mut fr = vec![0.0 as Scalar; n_bf];
    let mut ref_value = vec![0.0 as Scalar; n_bf];
    let ref_grad = vec![0.0 as Scalar; n_bf];

    for (bf, &on) in is_wall.iter().enumerate() {
        if !on {
            continue;
        }
        kind[bf] = BcKind::FixedValue as Label;
        fr[bf] = 1.0;
        ref_value[bf] = 0.0;
    }

    if n_bf > 0 {
        gpu.write(&mut psi.bc_kind, &kind)?;
        gpu.write(&mut psi.fr, &fr)?;
        gpu.write(&mut psi.ref_value, &ref_value)?;
        gpu.write(&mut psi.ref_grad, &ref_grad)?;
    }
    gpu.fill_zero(&mut psi.f)?;
    gpu.fill_zero(&mut psi.f0)?;
    gpu.fill_zero(&mut psi.bf)?;
    Ok(())
}

// ==========================================================================
//  Tests
// ==========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::case::{LinearSolverKind, Preconditioner};

    fn gpu() -> Option<Gpu> {
        Gpu::new(0).ok()
    }

    /// Tight, because the whole point of these tests is that the answer is
    /// analytic and the only error left should be discretisation.
    fn tight() -> SolverControls {
        SolverControls {
            solver: LinearSolverKind::PCG,
            precon: Preconditioner::Diagonal,
            tolerance: 1e-14,
            rel_tol: 0.0,
            max_iter: 5000,
            report_residuals: true,
            ..Default::default()
        }
    }

    /// `box_mesh` already walls `ymin`/`ymax` and empties `zmin`/`zmax`, which
    /// is a plane channel in `y` - the geometry SPEC-LIT §22 asks for.
    fn channel(ny: usize, dy: Scalar) -> HostMesh {
        let (mut m, points, faces) =
            crate::mesh::topology::tests::box_mesh([3, ny, 1], Vec3::new(0.5, dy, 0.5));
        m.compute_geometry(&points, &faces).expect("geometry");
        m.build_cell_face_maps();
        m
    }

    #[test]
    fn a_channel_reproduces_the_analytic_distance() -> Result<()> {
        let Some(gpu) = gpu() else {
            return Ok(());
        };

        let ny = 20;
        let dy: Scalar = 0.05;
        let ly = ny as Scalar * dy;

        let hm = channel(ny, dy);
        let m = GpuMesh::upload(&gpu, &hm)?;

        let wd = wall_distance(&gpu, &hm, &m, &tight(), 0)?;
        assert_eq!(wd.n_wall_faces, 2 * 3);

        let y = gpu.download(&wd.y.f)?;

        for (c, &v) in y.iter().enumerate() {
            let yc = hm.c[c].y;
            let want = yc.min(ly - yc);

            // §6.6's algebra is exact here - see the module header - so the
            // only error left is the Poisson solve's own, and it is worth
            // being precise about where that comes from. The discrete
            // laplacian reproduces the quadratic `phi` exactly in the
            // interior, but the wall's Dirichlet flux is differenced across
            // `h/2` and comes out `h/4` short of the true face gradient. That
            // defect sits in the two wall rows, and the discrete Green's
            // function spreads it over the whole channel, so the error is
            // O(h²) everywhere rather than confined to the first cell. At
            // `h/L = 1/20` it is 6.2e-4, which is `h²/4`; the tolerance below
            // is that number with a factor of two on it, and
            // `refining_the_channel_does_not_degrade_the_distance` measures
            // the order rather than trusting this line.
            let tol = 0.5 * dy * dy;

            assert!(
                (v - want).abs() < tol,
                "cell {c} at y = {yc}: distance {v}, analytic {want}, tolerance {tol}"
            );
        }

        Ok(())
    }

    /// Refining must not make it worse, and must make it better at a rate a
    /// second-order operator implies.
    #[test]
    fn refining_the_channel_does_not_degrade_the_distance() -> Result<()> {
        let Some(gpu) = gpu() else {
            return Ok(());
        };

        let mut worst = Vec::new();
        for ny in [10usize, 20, 40] {
            let dy = 1.0 as Scalar / ny as Scalar;
            let hm = channel(ny, dy);
            let m = GpuMesh::upload(&gpu, &hm)?;
            let wd = wall_distance(&gpu, &hm, &m, &tight(), 0)?;
            let y = gpu.download(&wd.y.f)?;

            let mut e: Scalar = 0.0;
            for (c, &v) in y.iter().enumerate() {
                let yc = hm.c[c].y;
                e = e.max((v - yc.min(1.0 - yc)).abs());
            }
            worst.push(e);
        }

        // Two halvings, so two order estimates, and both must land on 2: the
        // wall's one-sided Dirichlet flux is the only defect in the problem
        // and it is a second-order one.
        for w in worst.windows(2) {
            let order = (w[0] / w[1]).log2();
            assert!(
                (order - 2.0).abs() < 0.15,
                "observed order {order}, errors {worst:?}"
            );
        }
        assert!(worst[2] < 2e-4, "{worst:?}");

        Ok(())
    }

    /// A quarter annulus: `xmin` is the cylinder, `xmax` the far field,
    /// `ymin`/`ymax` the two radial cuts and `zmin`/`zmax` empty.
    ///
    /// The cuts are ordinary patches, which for a scalar is zero-gradient and
    /// therefore exactly the symmetry condition an axisymmetric solution
    /// satisfies there - so this quarter is the whole annulus as far as the
    /// solve is concerned, and no cyclic addressing is needed to say so.
    fn quarter_annulus(nr: usize, nt: usize, r_in: Scalar, r_out: Scalar) -> HostMesh {
        let (mut m, mut points, faces) =
            crate::mesh::topology::tests::box_mesh([nr, nt, 1], Vec3::new(1.0, 1.0, 1.0));

        let dr = (r_out - r_in) / nr as Scalar;
        let dth = (std::f64::consts::FRAC_PI_2 as Scalar) / nt as Scalar;

        // (i, j, k) -> (r cos th, r sin th, z). The map has a positive
        // Jacobian everywhere in r > 0, so the polyMesh winding - and with it
        // every outward normal - survives it unchanged.
        for p in points.iter_mut() {
            let r = r_in + p.x * dr;
            let th = p.y * dth;
            *p = Vec3::new(r * th.cos(), r * th.sin(), p.z * dr);
        }

        let kinds = [
            PatchKind::Wall,    // xmin: the cylinder
            PatchKind::Generic, // xmax: the far field
            PatchKind::Generic, // ymin: a radial cut
            PatchKind::Generic, // ymax: the other one
            PatchKind::Empty,
            PatchKind::Empty,
        ];
        for (p, k) in m.patches.iter_mut().zip(kinds) {
            p.kind = k;
            p.type_name = match k {
                PatchKind::Wall => "wall",
                PatchKind::Empty => "empty",
                _ => "patch",
            }
            .to_string();
        }

        m.compute_geometry(&points, &faces).expect("geometry");
        m.build_cell_face_maps();
        m
    }

    /// The closed-form Tucker distance for an annulus with a wall at `r_in`
    /// and zero gradient at `r_out`.
    ///
    /// `(1/r)(r phi')' = -1` integrates to `r phi' = -r²/2 + C`, and
    /// `phi'(r_out) = 0` fixes `C = r_out²/2`, so
    ///
    /// ```text
    /// phi'(r) = (r_out² - r²)/(2r)
    /// phi (r) = (r_in² - r²)/4 + (r_out²/2) ln(r/r_in)
    /// ```
    ///
    /// and §6.6's algebra is then applied to those two exactly. This is the
    /// analytic answer for the CONTINUOUS problem the solve discretises - not
    /// the geometric distance `r - R`, which Tucker's formula does not claim
    /// to reproduce once the wall is curved.
    fn annulus_tucker(r: Scalar, r_in: Scalar, r_out: Scalar) -> Scalar {
        let g = (r_out * r_out - r * r) / (2.0 * r);
        let phi = (r_in * r_in - r * r) / 4.0 + 0.5 * r_out * r_out * (r / r_in).ln();
        -g + (g * g + 2.0 * phi).sqrt()
    }

    /// §6.6's formula IS the geometric distance at a wall, however curved the
    /// wall is: as `r -> r_in` the relative departure goes to zero.
    ///
    /// A statement about the algebra, so it is checked on the host against the
    /// closed form above and needs no device. It is also the reason the device
    /// test below compares against `annulus_tucker` rather than against
    /// `r - r_in`: away from the wall the two genuinely differ, by 11 % at one
    /// cylinder radius out, and a test that hid that would be asserting
    /// something false about the method.
    #[test]
    fn tucker_tends_to_the_geometric_distance_at_a_wall() {
        let (r_in, r_out) = (0.5 as Scalar, 2.5 as Scalar);

        let mut prev = Scalar::INFINITY;
        for &eps in &[0.2 as Scalar, 0.1, 0.05, 0.02, 0.01, 0.005] {
            let r = r_in + eps;
            let rel = (annulus_tucker(r, r_in, r_out) - eps).abs() / eps;
            assert!(rel < prev, "the departure grew as the wall was approached");
            prev = rel;
        }
        assert!(prev < 0.02, "at the wall the departure is still {prev}");
    }

    /// SPEC-LIT §22: "a cylinder in a box - radial, to discretisation error".
    ///
    /// Two claims, and they are different. The first is that the answer
    /// depends only on the radius; that is a statement about the solve and the
    /// mesh, and it holds to discretisation error. The second is that it is
    /// the distance; that is a statement about §6.6's algebra, and it holds
    /// exactly at the wall and approximately away from it - so what is
    /// compared here is the closed-form solution of the same continuous
    /// problem, which isolates the discretisation from the model.
    #[test]
    fn a_cylinder_gives_a_radial_distance() -> Result<()> {
        let Some(gpu) = gpu() else {
            return Ok(());
        };

        let (nr, nt) = (24usize, 24usize);
        let (r_in, r_out) = (0.5 as Scalar, 2.5 as Scalar);
        let dr = (r_out - r_in) / nr as Scalar;

        let hm = quarter_annulus(nr, nt, r_in, r_out);
        let m = GpuMesh::upload(&gpu, &hm)?;

        // A polygonal annulus is not orthogonal, so the correction earns its
        // keep here in a way it does not in the channel.
        let wd = wall_distance(&gpu, &hm, &m, &tight(), 2)?;
        assert_eq!(wd.n_wall_faces, nt);

        let y = gpu.download(&wd.y.f)?;

        // Cells are numbered i + nr*(j + nt*k), so a whole ring shares one i.
        for i in 0..nr {
            let mut lo = Scalar::INFINITY;
            let mut hi: Scalar = 0.0;
            for j in 0..nt {
                let v = y[i + nr * j];
                lo = lo.min(v);
                hi = hi.max(v);
            }
            assert!(
                hi - lo < 1e-3 * (hi + dr),
                "ring {i}: y ranges over [{lo}, {hi}], which is not radial"
            );
        }

        // Ring by ring outward: strictly increasing, and on the closed-form
        // solution of the continuous problem to within the discretisation.
        //
        // The radius used is the CIRCUMSCRIBED one - a polygonal ring's cell
        // centres sit inside the circle its points lie on - so the comparison
        // is against the analytic solution evaluated where the cell centre
        // actually is, not where the ideal annulus would have put it.
        let mid = nt / 2;
        let mut prev: Scalar = -1.0;
        for i in 0..nr {
            let c = hm.c[i + nr * mid];
            let r = (c.x * c.x + c.y * c.y).sqrt();
            let v = y[i + nr * mid];

            assert!(v > prev, "ring {i}: y = {v} did not increase past {prev}");
            prev = v;

            let want = annulus_tucker(r, r_in, r_out);
            assert!(
                (v - want).abs() < 0.04 * want.max(dr),
                "ring {i} at r = {r}: y = {v}, analytic Tucker distance {want}"
            );
        }

        Ok(())
    }

    /// A mesh with no wall in it gets [`NO_WALL`], not a singular solve.
    #[test]
    fn a_wall_free_mesh_reports_no_wall() -> Result<()> {
        let Some(gpu) = gpu() else {
            return Ok(());
        };

        let hm = channel(6, 0.1);
        let m = GpuMesh::upload(&gpu, &hm)?;

        let none = vec![false; hm.n_boundary_faces];
        let wd = wall_distance_from_faces(&gpu, &hm, &m, &none, &tight(), 0)?;

        assert_eq!(wd.n_wall_faces, 0);
        for (c, &v) in gpu.download(&wd.y.f)?.iter().enumerate() {
            assert!((v - NO_WALL).abs() < 1e-6, "cell {c}: y = {v}");
        }

        Ok(())
    }

    /// A wall flag on an `empty` face names a wall the matrix cannot feel, so
    /// it must not be counted as one - otherwise the wall-free branch is
    /// skipped and an all-Neumann laplacian gets solved instead.
    #[test]
    fn a_wall_on_an_empty_patch_is_not_a_wall() -> Result<()> {
        let Some(gpu) = gpu() else {
            return Ok(());
        };

        let hm = channel(6, 0.1);
        let m = GpuMesh::upload(&gpu, &hm)?;

        let mut flags = vec![false; hm.n_boundary_faces];
        for p in &hm.patches {
            if p.kind == PatchKind::Empty {
                for bf in p.start..p.start + p.size {
                    flags[bf] = true;
                }
            }
        }

        let wd = wall_distance_from_faces(&gpu, &hm, &m, &flags, &tight(), 0)?;
        assert_eq!(wd.n_wall_faces, 0);

        Ok(())
    }

    #[test]
    fn a_mismatched_flag_length_is_an_error() -> Result<()> {
        let Some(gpu) = gpu() else {
            return Ok(());
        };

        let hm = channel(4, 0.1);
        let m = GpuMesh::upload(&gpu, &hm)?;
        let short = vec![false; hm.n_boundary_faces - 1];

        assert!(wall_distance_from_faces(&gpu, &hm, &m, &short, &tight(), 0).is_err());
        Ok(())
    }
}
