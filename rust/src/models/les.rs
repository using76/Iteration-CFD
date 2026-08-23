// meteor-cfd - Copyright (c) 2026 주식회사 메테오시뮬레이션 (Meteo Simulation Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.

//! The three subgrid-scale models of SPEC-LIT §6.5.
//!
//! Written from:
//!   Smagorinsky, *Mon. Weather Rev.* 91 (1963) 99-164
//!   Nicoud & Ducros, *Flow Turbul. Combust.* 62 (1999) 183-200 - WALE
//!   Deardorff, *Boundary-Layer Meteorol.* 18 (1980) 495-527, in the algebraic
//!     form used by FDS (NIST, public domain; see `reference/fds`
//!     `Source/velo.f90` and the FDS Technical Reference Guide), which
//!     SPEC-LIT §6.5 names as the reference implementation
//!   Lilly, in *Proc. IBM Sci. Comput. Symp. Environ. Sci.* (1967) - the value
//!     of `C_s` that the inertial-range argument gives, and why the wall-bounded
//!     value is lower
//!   ofgpu `SPEC-LIT.md` §6.5 for the models and §16 for the filter width they
//!     share, which lives in [`crate::les`]
//! No GPL-licensed source was consulted.
//!
//! # Acknowledgement
//!
//! The Deardorff form implemented here - subgrid kinetic energy estimated from
//! a test-filtered velocity, `nu_t = C_D Delta sqrt(k_sgs)` - follows FDS, a
//! work of the United States National Institute of Standards and Technology
//! and in the public domain. It is used with thanks. The unstructured-mesh
//! test filter that feeds it is ours; `cuda/les.cu` says exactly what was
//! adapted and what was derived.
//!
//! # Why these three are one type and the RAS models are not
//!
//! `src/models/mod.rs` explains why `KEpsilon` and `KOmega` share no trait:
//! one carries `epsilon` and the other `omega`, and a common accessor would
//! mean two different things. That argument does not apply here. All three of
//! these are **algebraic**: they solve no transport equation, they carry no
//! state between steps, and they differ only in the expression that turns
//! `grad U` and `Delta` into `nu_t`. So they are one type and an enum, and the
//! enum is switched on once per time step - outside every kernel launch.
//!
//! # What an LES `correct` does, and what it does not
//!
//! ```text
//! grad U        ->  the strain rate, and WALE's gd = g.g
//! Delta         ->  §16, rebuilt if van Driest damping is on
//! nu_t          ->  one kernel, chosen by the enum
//! boundary      ->  zero-gradient, then any wall function nut's own patch
//!                   type asked for (SPEC-LIT §15.5)
//! ```
//!
//! There is no matrix, no linear solve and no bounding, because there is
//! nothing to bound: every one of the three expressions is non-negative by
//! construction. What there is instead is `nut_max`, the same *DESIGN* ceiling
//! §6.1 puts on the RAS models, because a filter width that has just been
//! smoothed across a mesh jump can otherwise hand the momentum equation a
//! viscosity the time step cannot carry.

use crate::device::{DevBuf, Gpu};
use crate::error::{Error, Result};
use crate::field::GpuScalarField;
use crate::field_ops::{correct_boundary_conditions, FieldKernels};
use crate::fv::{fvc_grad_vector, FvKernels};
use crate::les::{
    nut_deardorff, nut_smagorinsky, nut_wale, test_filter, DeltaSpec, LesDelta, LesKernels,
};
use crate::mesh::{GpuMesh, HostMesh};
use crate::turbulence::{nut_boundary, strain_rate_mag, FlowState, TurbKernels, TurbulenceControls};
use crate::wallfunctions::{WallData, WallFunctionCoeffs};
use crate::{Scalar, Tensor, Vec3};

/// Which subgrid model - SPEC-LIT §6.5.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LesModel {
    /// `nu_t = (C_s Delta)² sqrt(2 S:S)` - Smagorinsky (1963).
    Smagorinsky,
    /// Nicoud & Ducros (1999). Recovers the `y³` near-wall scaling with no
    /// damping function, which is its reason for existing.
    Wale,
    /// Deardorff (1980), in FDS's algebraic form.
    Deardorff,
}

impl LesModel {
    /// The name a case writes in `LES { model ...; }`.
    pub fn name(self) -> &'static str {
        match self {
            Self::Smagorinsky => "Smagorinsky",
            Self::Wale => "WALE",
            Self::Deardorff => "Deardorff",
        }
    }

    /// Whether this model needs a near-wall damping function to get the
    /// near-wall limit right.
    ///
    /// Smagorinsky does: its `nu_t` goes like `y⁰` where the true subgrid
    /// stress goes like `y³`, which is what van Driest damping (§16.4) is for.
    /// WALE does not, by construction. Deardorff's estimate inherits whatever
    /// the resolved velocity difference does, which vanishes at a wall because
    /// the velocity does. Reported rather than enforced, so a driver can say
    /// so in its log instead of a user finding out from a profile.
    pub fn wants_van_driest(self) -> bool {
        matches!(self, Self::Smagorinsky)
    }
}

/// The one coefficient each model carries.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LesCoeffs {
    /// Smagorinsky's `C_s`. SPEC-LIT §6.5 gives the range 0.1-0.2; Lilly's
    /// inertial-range argument gives 0.17, and the value that works in a
    /// wall-bounded flow is nearer 0.1 because the model has no idea a wall is
    /// there. **We default to 0.168**, the inertial-range value, because it is
    /// the one that follows from a derivation rather than from a fit - and
    /// because the near-wall deficiency it exposes is the one van Driest
    /// damping is there to repair. A case that runs Smagorinsky without
    /// damping in a channel should lower it, and the number is a dictionary
    /// entry so that it can.
    pub cs: Scalar,
    /// WALE's `C_w`, 0.325 - Nicoud & Ducros (1999), SPEC-LIT §6.5.
    pub cw: Scalar,
    /// Deardorff's `C_D`, 0.1 - the value FDS uses (`C_DEARDORFF`).
    pub cd: Scalar,
}

impl Default for LesCoeffs {
    fn default() -> Self {
        Self {
            cs: 0.168,
            cw: 0.325,
            cd: 0.1,
        }
    }
}

// ==========================================================================
//  The model
// ==========================================================================

/// A large-eddy simulation closure, resident on the device.
pub struct Les<'m> {
    mesh: &'m GpuMesh,
    ctrl: TurbulenceControls,
    wall: WallFunctionCoeffs,

    model: LesModel,
    coeffs: LesCoeffs,

    fv: FvKernels,
    fld: FieldKernels,
    turb: TurbKernels,
    /// `cuda/les.cu`'s entry points. [`LesDelta`] resolved its own copy for
    /// the filter-width kernels; this one is here so that a `nu_t` pass can
    /// borrow the kernels and the field it writes at the same time, which a
    /// reach-through accessor on `delta` would not allow.
    les: LesKernels,

    /// Which faces `nu_t` gets a wall value on - from `nut`'s OWN patch type
    /// and nothing else's, SPEC-LIT §15.5.
    wd: WallData,

    /// The filter width of SPEC-LIT §16, and everything it needs.
    delta: LesDelta,

    nut: GpuScalarField,

    /// `[n_cells]` the wall distance, and its gradient - the wall normal
    /// §16.4 uses. Copied in at construction; both are `NO_WALL`/zero in a
    /// domain with no wall, which is what makes van Driest damping inert
    /// there.
    y: DevBuf<Scalar>,
    grad_y: DevBuf<Vec3>,

    /// `[n_cells]` scratch: `grad U`, the strain-rate magnitude, the
    /// test-filtered velocity and the subgrid kinetic energy.
    grad_u: DevBuf<Tensor>,
    s: DevBuf<Scalar>,
    u_hat: DevBuf<Vec3>,
    k_sgs: DevBuf<Scalar>,
}

impl<'m> Les<'m> {
    /// `y` and `grad_y` come from [`crate::walldistance::wall_distance`] and
    /// are copied, not borrowed. Both are needed only by van Driest damping;
    /// a spec without it never reads them, and a wall-free domain's `NO_WALL`
    /// and zero gradient are the right values for it anyway.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        gpu: &Gpu,
        hm: &HostMesh,
        mesh: &'m GpuMesh,
        model: LesModel,
        coeffs: LesCoeffs,
        delta: DeltaSpec,
        ctrl: TurbulenceControls,
        wall: WallFunctionCoeffs,
        wall_faces: &crate::field_setup::WallFaces,
        y: &DevBuf<Scalar>,
        grad_y: &DevBuf<Vec3>,
    ) -> Result<Self> {
        if hm.n_cells != mesh.n_cells || hm.n_boundary_faces != mesh.n_boundary_faces {
            return Err(Error::Config(format!(
                "Les::new: the host mesh has ({}, {}) cells/boundary faces and \
                 the device mesh ({}, {})",
                hm.n_cells, hm.n_boundary_faces, mesh.n_cells, mesh.n_boundary_faces
            )));
        }
        for (what, len) in [("y", y.len()), ("grad y", grad_y.len())] {
            if len != mesh.n_cells && mesh.n_cells > 0 {
                return Err(Error::Config(format!(
                    "Les::new: `{what}` has {len} entries, the mesh has {} cells \
                     (SPEC-LIT 6.6)",
                    mesh.n_cells
                )));
            }
        }
        for (name, v) in [("Cs", coeffs.cs), ("Cw", coeffs.cw), ("Cd", coeffs.cd)] {
            if !(v >= 0.0) || !v.is_finite() {
                return Err(Error::Config(format!(
                    "LES: {name} = {v}; a model constant must be finite and \
                     non-negative"
                )));
            }
        }

        let nc = mesh.n_cells.max(1);
        let fld = FieldKernels::new(gpu)?;

        let mut y_own: DevBuf<Scalar> = gpu.zeros(nc)?;
        let mut gy_own: DevBuf<Vec3> = gpu.zeros(nc)?;
        crate::field_ops::copy_field(gpu, &fld, &mut y_own, y, mesh.n_cells)?;
        crate::field_ops::copy_field_vector(gpu, &fld, &mut gy_own, grad_y, mesh.n_cells)?;

        Ok(Self {
            mesh,
            ctrl,
            wall,
            model,
            coeffs,

            fv: FvKernels::new(gpu)?,
            fld,
            turb: TurbKernels::new(gpu)?,
            les: LesKernels::new(gpu)?,

            wd: WallData::build(gpu, hm, wall_faces)?,
            delta: LesDelta::new(gpu, mesh, delta)?,

            nut: GpuScalarField::zeros(gpu, mesh, "nut")?,

            y: y_own,
            grad_y: gy_own,

            grad_u: gpu.zeros(nc)?,
            s: gpu.zeros(nc)?,
            u_hat: gpu.zeros(nc)?,
            k_sgs: gpu.zeros(nc)?,
        })
    }

    // ---- accessors --------------------------------------------------------

    pub fn model(&self) -> LesModel {
        self.model
    }
    pub fn coeffs(&self) -> &LesCoeffs {
        &self.coeffs
    }
    pub fn nut(&self) -> &GpuScalarField {
        &self.nut
    }
    pub fn nut_mut(&mut self) -> &mut GpuScalarField {
        &mut self.nut
    }
    /// The filter width in force, and everything §16 computed on the way to
    /// it.
    pub fn delta(&self) -> &LesDelta {
        &self.delta
    }
    /// The subgrid kinetic energy. Deardorff's own estimate when that is the
    /// model; for the other two it is left at zero, because neither has one
    /// and inventing one would be a number a user could plot and believe.
    pub fn k_sgs(&self) -> &DevBuf<Scalar> {
        &self.k_sgs
    }
    /// The strain-rate magnitude `sqrt(2 S:S)` of the last `correct`.
    pub fn strain_rate(&self) -> &DevBuf<Scalar> {
        &self.s
    }

    /// The eddy-viscosity ceiling of SPEC-LIT §6.1's *DESIGN* note, applied
    /// here for the reason given in the module header.
    #[inline]
    pub fn nut_max(&self, nu: Scalar) -> Scalar {
        self.ctrl.nut_max_coeff * nu
    }

    /// `nu_t = 0` everywhere, and the boundary triple rewritten so nothing can
    /// put it back - what `simulationType laminar;` means for an LES case, and
    /// what a DNS on a resolved mesh wants.
    pub fn freeze_nut(&mut self, gpu: &Gpu) -> Result<()> {
        let zeros_c = vec![0.0 as Scalar; self.nut.f.len()];
        let zeros_b = vec![0.0 as Scalar; self.nut.bf.len()];
        let ones_b = vec![1.0 as Scalar; self.nut.fr.len()];

        gpu.write(&mut self.nut.f, &zeros_c)?;
        gpu.write(&mut self.nut.f0, &zeros_c)?;
        gpu.write(&mut self.nut.bf, &zeros_b)?;
        gpu.write(&mut self.nut.fr, &ones_b)?;
        gpu.write(&mut self.nut.ref_value, &zeros_b)?;
        gpu.write(&mut self.nut.ref_grad, &zeros_b)?;
        Ok(())
    }

    // ---- one step ---------------------------------------------------------

    /// Rebuild `nu_t` from the current velocity field.
    ///
    /// One call per time step. There is no outer iteration to converge and no
    /// old time level to rotate: an algebraic model has no memory.
    pub fn correct(&mut self, gpu: &Gpu, flow: &FlowState) -> Result<()> {
        let n = self.mesh.n_cells;
        if n == 0 {
            return Ok(());
        }
        let nut_max = self.nut_max(flow.nu);

        fvc_grad_vector(gpu, &self.fv, &mut self.grad_u, flow.u, self.mesh)?;
        strain_rate_mag(gpu, &self.turb, &mut self.s, &self.grad_u, n)?;

        // §16.4 reads the PREVIOUS step's nu_t, which is the ordinary
        // segregated lag and is what makes u_tau the total shear velocity
        // rather than the molecular one. A spec with no van Driest term
        // returns immediately.
        self.delta.update(
            gpu,
            self.mesh,
            &self.grad_u,
            &self.grad_y,
            &self.nut.f,
            &self.y,
            flow.nu,
        )?;

        match self.model {
            LesModel::Smagorinsky => nut_smagorinsky(
                gpu,
                &self.les,
                &mut self.nut.f,
                &self.s,
                self.delta.delta(),
                self.coeffs.cs,
                nut_max,
                n,
            )?,
            LesModel::Wale => nut_wale(
                gpu,
                &self.les,
                &mut self.nut.f,
                &self.grad_u,
                self.delta.delta(),
                self.coeffs.cw,
                nut_max,
                n,
            )?,
            LesModel::Deardorff => {
                test_filter(
                    gpu,
                    &self.les,
                    &mut self.u_hat,
                    &flow.u.f,
                    &flow.u.bf,
                    self.mesh,
                )?;
                nut_deardorff(
                    gpu,
                    &self.les,
                    &mut self.nut.f,
                    &mut self.k_sgs,
                    &flow.u.f,
                    &self.u_hat,
                    self.delta.delta(),
                    self.coeffs.cd,
                    nut_max,
                    n,
                )?;
            }
        }

        // Zero-gradient everywhere the model owns the face, then the wall
        // function on the faces `nut`'s OWN patch type asked for - SPEC-LIT
        // §15.5. An LES on a resolved mesh names none of them and this is a
        // no-op, which is the point: it is the case file that decides, not the
        // model.
        correct_boundary_conditions(gpu, &self.fld, &mut self.nut, self.mesh)?;
        nut_boundary(gpu, &self.turb, &mut self.nut, self.mesh)?;

        Ok(())
    }

    /// The wall-function pass, for a driver that runs one.
    ///
    /// Separate from [`Self::correct`] because it needs `k`, and an LES has no
    /// `k` unless the model is Deardorff. A driver with a modelled `k` - or
    /// one that wants the Deardorff estimate used this way - passes it here;
    /// one without simply never calls it, and the wall faces keep the
    /// zero-gradient value `correct` left.
    pub fn apply_nut_wall_function(
        &mut self,
        gpu: &Gpu,
        k: &DevBuf<Scalar>,
        nu: Scalar,
    ) -> Result<()> {
        self.wd.update_nut(
            gpu,
            &mut self.nut.bf,
            k,
            self.mesh,
            &self.wall,
            nu,
            self.ctrl.k_min,
        )
    }
}

// ==========================================================================
//  Tests
// ==========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::field::{BcKind, GpuSurfaceScalarField, GpuVectorField};
    use crate::field_ops::correct_boundary_conditions_vector;
    use crate::les::{BaseDelta, SmoothSpec};
    use crate::mesh::PatchKind;
    use crate::Label;

    fn gpu() -> Option<Gpu> {
        Gpu::new(0).ok()
    }

    /// A closed two-dimensional box: `nz = 1`, so the `empty` patches
    /// `box_mesh` puts on `zmin`/`zmax` describe a real 2-D case and every
    /// surface integral closes. On a mesh several cells deep with `empty` ends
    /// the Green-Gauss gradient of a uniform field is not zero, and every
    /// closed-form check below would be measuring the mesh instead of the
    /// model.
    fn flat_box(n: usize, h: Scalar) -> HostMesh {
        let (mut m, points, faces) =
            crate::mesh::topology::tests::box_mesh([n, n, 1], Vec3::new(h, h, h));
        m.compute_geometry(&points, &faces).expect("geometry");
        m.build_cell_face_maps();
        m
    }

    /// A velocity field with EXACT Dirichlet boundary values.
    ///
    /// Green-Gauss reproduces the gradient of a linear field exactly only if
    /// the face values are exact, so a zero-gradient boundary would put an
    /// O(h) error into the one quantity every test here is measuring. `empty`
    /// faces keep their own condition, which evaluates to the cell value and
    /// is what a direction the mesh does not resolve should contribute.
    fn upload_u(
        gpu: &Gpu,
        hm: &HostMesh,
        mesh: &GpuMesh,
        fld: &FieldKernels,
        f: impl Fn(Vec3) -> Vec3,
    ) -> Result<GpuVectorField> {
        let mut u = GpuVectorField::zeros(gpu, mesh, "U")?;

        let cells: Vec<Vec3> = hm.c.iter().map(|c| f(*c)).collect();
        gpu.write(&mut u.f, &cells)?;

        let n_bf = hm.n_boundary_faces;
        let mut kind = gpu.download(&u.bc_kind)?;
        let mut fr = vec![0.0 as Scalar; n_bf];
        let mut rv = vec![Vec3::ZERO; n_bf];

        for bf in 0..n_bf {
            if hm.b_kind[bf] == PatchKind::Empty as Label {
                continue;
            }
            kind[bf] = BcKind::FixedValue as Label;
            fr[bf] = 1.0;
            rv[bf] = f(hm.b_cf[bf]);
        }

        gpu.write(&mut u.bc_kind, &kind)?;
        gpu.write(&mut u.fr, &fr)?;
        gpu.write(&mut u.ref_value, &rv)?;
        correct_boundary_conditions_vector(gpu, fld, &mut u, mesh)?;

        Ok(u)
    }

    fn controls() -> TurbulenceControls {
        TurbulenceControls {
            steady: false,
            delta_t: 1e-3,
            nut_max_coeff: 1e8,
            ..Default::default()
        }
    }

    /// Build a model on a flat box and correct it once against `f`.
    fn run(
        gpu: &Gpu,
        hm: &HostMesh,
        mesh: &GpuMesh,
        model: LesModel,
        delta: DeltaSpec,
        f: impl Fn(Vec3) -> Vec3,
    ) -> Result<(Vec<Scalar>, Vec<Scalar>, Vec<Scalar>)> {
        let fld = FieldKernels::new(gpu)?;
        let u = upload_u(gpu, hm, mesh, &fld, f)?;
        let phi = GpuSurfaceScalarField::zeros(gpu, mesh, "phi")?;
        let flow = FlowState::new(&u, &phi, 1e-5);

        let no_walls = crate::field_setup::WallFaces::none(hm.n_boundary_faces);
        let y = gpu.upload(&vec![crate::walldistance::NO_WALL; mesh.n_cells])?;
        let grad_y = gpu.upload(&vec![Vec3::ZERO; mesh.n_cells])?;

        let mut les = Les::new(
            gpu,
            hm,
            mesh,
            model,
            LesCoeffs::default(),
            delta,
            controls(),
            WallFunctionCoeffs::default(),
            &no_walls,
            &y,
            &grad_y,
        )?;

        les.correct(gpu, &flow)?;
        gpu.sync()?;

        Ok((
            gpu.download(&les.nut().f)?,
            gpu.download(les.k_sgs())?,
            gpu.download(les.delta().delta())?,
        ))
    }

    /// The cells with no non-`empty` boundary face - the ones whose seven-point
    /// stencil is complete.
    fn interior(hm: &HostMesh, n: usize) -> Vec<usize> {
        let _ = hm;
        let mut v = Vec::new();
        for j in 1..n - 1 {
            for i in 1..n - 1 {
                v.push(i + n * j);
            }
        }
        v
    }

    // ----------------------------------------------------------------------
    //  Smagorinsky
    // ----------------------------------------------------------------------

    /// `nu_t = (C_s Delta)² sqrt(2 S:S)`, against the closed form.
    ///
    /// `U = (a y, 0, 0)` gives `grad U` with one entry, `dU_x/dy = a`, so
    /// `S:S = a²/2` and the strain-rate magnitude is exactly `a`. On a uniform
    /// box the filter width is the edge, so the whole answer is
    /// `(C_s h)² a` - one number, checkable to round-off in every cell.
    #[test]
    fn smagorinsky_is_the_closed_form_in_a_uniform_shear() -> Result<()> {
        let Some(gpu) = gpu() else {
            return Ok(());
        };

        let (n, h): (usize, Scalar) = (5, 0.2);
        let a: Scalar = 3.0;
        let hm = flat_box(n, h);
        let mesh = GpuMesh::upload(&gpu, &hm)?;

        let (nut, _, delta) = run(
            &gpu,
            &hm,
            &mesh,
            LesModel::Smagorinsky,
            DeltaSpec::default(),
            |c| Vec3::new(a * c.y, 0.0, 0.0),
        )?;

        let cs = LesCoeffs::default().cs;
        let want = (cs * h) * (cs * h) * a;

        for c in 0..hm.n_cells {
            assert!(
                (delta[c] - h).abs() < 1e-13 * h,
                "cell {c}: Delta = {}",
                delta[c]
            );
            assert!(
                (nut[c] - want).abs() < 1e-12 * want,
                "cell {c}: nu_t = {}, (Cs h)² a = {want}",
                nut[c]
            );
        }

        Ok(())
    }

    /// And it scales with the square of the filter width, which is the only
    /// thing SPEC-LIT §16 changes about it.
    #[test]
    fn smagorinsky_scales_with_the_square_of_delta() -> Result<()> {
        let Some(gpu) = gpu() else {
            return Ok(());
        };

        let (n, h): (usize, Scalar) = (5, 0.2);
        let hm = flat_box(n, h);
        let mesh = GpuMesh::upload(&gpu, &hm)?;

        let plain = DeltaSpec::default();
        let doubled = DeltaSpec {
            delta_coeff: 2.0,
            ..Default::default()
        };

        let (a, _, _) = run(&gpu, &hm, &mesh, LesModel::Smagorinsky, plain, |c| {
            Vec3::new(3.0 * c.y, 0.0, 0.0)
        })?;
        let (b, _, _) = run(&gpu, &hm, &mesh, LesModel::Smagorinsky, doubled, |c| {
            Vec3::new(3.0 * c.y, 0.0, 0.0)
        })?;

        for c in 0..hm.n_cells {
            assert!(
                (b[c] - 4.0 * a[c]).abs() < 1e-12 * b[c].max(1e-30),
                "cell {c}: doubling Delta multiplied nu_t by {}",
                b[c] / a[c]
            );
        }

        Ok(())
    }

    // ----------------------------------------------------------------------
    //  WALE
    // ----------------------------------------------------------------------

    /// WALE gives EXACTLY zero in a pure shear, and that is the property it
    /// exists for.
    ///
    /// With `dU_x/dy` the only non-zero entry, `grad U` is nilpotent:
    /// `(grad U)·(grad U) = 0`, so `Sd = 0` and the numerator vanishes
    /// identically. Smagorinsky, given the same field, returns a positive
    /// eddy viscosity - which is precisely its near-wall failure, and the
    /// second half of this test measures the difference rather than asserting
    /// zero against zero.
    #[test]
    fn wale_vanishes_in_a_pure_shear_where_smagorinsky_does_not() -> Result<()> {
        let Some(gpu) = gpu() else {
            return Ok(());
        };

        let (n, h): (usize, Scalar) = (5, 0.2);
        let hm = flat_box(n, h);
        let mesh = GpuMesh::upload(&gpu, &hm)?;
        let shear = |c: Vec3| Vec3::new(4.0 * c.y, 0.0, 0.0);

        let (wale, _, _) = run(&gpu, &hm, &mesh, LesModel::Wale, DeltaSpec::default(), shear)?;
        let (smag, _, _) = run(
            &gpu,
            &hm,
            &mesh,
            LesModel::Smagorinsky,
            DeltaSpec::default(),
            shear,
        )?;

        for c in 0..hm.n_cells {
            assert!(
                wale[c].abs() < 1e-20,
                "cell {c}: WALE returned {} in a pure shear",
                wale[c]
            );
            assert!(
                smag[c] > 1e-4,
                "cell {c}: Smagorinsky returned {}, so the contrast is not real",
                smag[c]
            );
        }

        Ok(())
    }

    /// In a pure strain it must return the expression of Nicoud & Ducros, and
    /// here that is a number this test computes from the tensor algebra
    /// independently of the kernel.
    ///
    /// `U = (a x, -a y, 0)` gives `grad U = diag(a, -a, 0)`, so
    /// `S:S = 2a²`, `gd = diag(a², a², 0)`, `Sd = diag(a²/3, a²/3, -2a²/3)`
    /// and `Sd:Sd = (2/3) a⁴`.
    #[test]
    fn wale_is_the_closed_form_in_a_pure_strain() -> Result<()> {
        let Some(gpu) = gpu() else {
            return Ok(());
        };

        let (n, h): (usize, Scalar) = (5, 0.2);
        let a: Scalar = 2.5;
        let hm = flat_box(n, h);
        let mesh = GpuMesh::upload(&gpu, &hm)?;

        let (nut, _, _) = run(&gpu, &hm, &mesh, LesModel::Wale, DeltaSpec::default(), |c| {
            Vec3::new(a * c.x, -a * c.y, 0.0)
        })?;

        let ss = 2.0 * a * a;
        let sdsd = (2.0 / 3.0) * a.powi(4);
        let cw = LesCoeffs::default().cw;
        let want = (cw * h) * (cw * h) * sdsd.powf(1.5) / (ss.powf(2.5) + sdsd.powf(1.25));

        assert!(want > 0.0, "the reference value is {want}");

        for c in 0..hm.n_cells {
            assert!(
                (nut[c] - want).abs() < 1e-11 * want,
                "cell {c}: nu_t = {}, Nicoud & Ducros give {want}",
                nut[c]
            );
        }

        Ok(())
    }

    // ----------------------------------------------------------------------
    //  Deardorff
    // ----------------------------------------------------------------------

    /// The test filter annihilates a linear field, so a resolved shear carries
    /// no subgrid energy at all - which is what makes the estimate an estimate
    /// of the UNRESOLVED motion rather than of the motion.
    ///
    /// Checked on the interior cells only: a cell with a Dirichlet face sees
    /// that face's value at half the spacing of a cell centre, so its stencil
    /// is not symmetric and the cancellation is not exact there.
    #[test]
    fn deardorff_sees_no_subgrid_energy_in_a_resolved_linear_field() -> Result<()> {
        let Some(gpu) = gpu() else {
            return Ok(());
        };

        let (n, h): (usize, Scalar) = (7, 0.2);
        let hm = flat_box(n, h);
        let mesh = GpuMesh::upload(&gpu, &hm)?;

        let (nut, k_sgs, _) = run(
            &gpu,
            &hm,
            &mesh,
            LesModel::Deardorff,
            DeltaSpec::default(),
            |c| Vec3::new(3.0 * c.y + 2.0 * c.x, -1.5 * c.y, 0.0),
        )?;

        for c in interior(&hm, n) {
            assert!(
                k_sgs[c] < 1e-26,
                "interior cell {c}: k_sgs = {} in a linear field",
                k_sgs[c]
            );
            assert!(nut[c] < 1e-13, "interior cell {c}: nu_t = {}", nut[c]);
        }

        Ok(())
    }

    /// And on a field the mesh cannot resolve it must return the closed form.
    ///
    /// With `U = (a y², 0, 0)` the seven-point filter leaves
    /// `u - u_hat = -a h²/6` in an interior cell - the two `y` neighbours
    /// contribute `2 a h²` of curvature between them, spread over the six
    /// faces and halved by the filter's own weight - so
    /// `k_sgs = (a h²)²/72` and `nu_t = C_D h sqrt(k_sgs)`.
    #[test]
    fn deardorff_is_the_closed_form_on_an_unresolved_field() -> Result<()> {
        let Some(gpu) = gpu() else {
            return Ok(());
        };

        let (n, h): (usize, Scalar) = (7, 0.2);
        let a: Scalar = 5.0;
        let hm = flat_box(n, h);
        let mesh = GpuMesh::upload(&gpu, &hm)?;

        let (nut, k_sgs, _) = run(
            &gpu,
            &hm,
            &mesh,
            LesModel::Deardorff,
            DeltaSpec::default(),
            |c| Vec3::new(a * c.y * c.y, 0.0, 0.0),
        )?;

        let want_k = (a * h * h) * (a * h * h) / 72.0;
        let want_nut = LesCoeffs::default().cd * h * want_k.sqrt();

        for c in interior(&hm, n) {
            assert!(
                (k_sgs[c] - want_k).abs() < 1e-11 * want_k,
                "cell {c}: k_sgs = {}, closed form {want_k}",
                k_sgs[c]
            );
            assert!(
                (nut[c] - want_nut).abs() < 1e-11 * want_nut,
                "cell {c}: nu_t = {}, C_D Delta sqrt(k_sgs) = {want_nut}",
                nut[c]
            );
        }

        Ok(())
    }

    // ----------------------------------------------------------------------
    //  The contract
    // ----------------------------------------------------------------------

    /// A uniform flow has no subgrid anything, whichever model is asked.
    #[test]
    fn a_uniform_flow_has_no_eddy_viscosity() -> Result<()> {
        let Some(gpu) = gpu() else {
            return Ok(());
        };

        let hm = flat_box(5, 0.2);
        let mesh = GpuMesh::upload(&gpu, &hm)?;

        for model in [LesModel::Smagorinsky, LesModel::Wale, LesModel::Deardorff] {
            let (nut, _, _) = run(&gpu, &hm, &mesh, model, DeltaSpec::default(), |_| {
                Vec3::new(1.7, -0.3, 0.0)
            })?;
            // Not zero to the last bit: the Green-Gauss gradient of a uniform
            // field cancels to round-off rather than identically, so
            // Smagorinsky's sqrt(2 S:S) lands near 1e-15 s^-1 and its nu_t
            // near 1e-18 m²/s. That is fifteen orders below the sheared case
            // in `smagorinsky_is_the_closed_form_in_a_uniform_shear`, which is
            // what "no eddy viscosity" can mean in floating point.
            for (c, &v) in nut.iter().enumerate() {
                assert!(
                    v.abs() < 1e-15,
                    "{}: cell {c} has nu_t = {v} in a uniform flow",
                    model.name()
                );
            }
        }

        Ok(())
    }

    /// The eddy-viscosity ceiling of SPEC-LIT §6.1, applied here for the
    /// reason the module header gives.
    #[test]
    fn nut_is_capped() -> Result<()> {
        let Some(gpu) = gpu() else {
            return Ok(());
        };

        let hm = flat_box(4, 0.25);
        let mesh = GpuMesh::upload(&gpu, &hm)?;

        let fld = FieldKernels::new(&gpu)?;
        let u = upload_u(&gpu, &hm, &mesh, &fld, |c| Vec3::new(1e6 * c.y, 0.0, 0.0))?;
        let phi = GpuSurfaceScalarField::zeros(&gpu, &mesh, "phi")?;
        let nu: Scalar = 1e-5;
        let flow = FlowState::new(&u, &phi, nu);

        let no_walls = crate::field_setup::WallFaces::none(hm.n_boundary_faces);
        let y = gpu.upload(&vec![crate::walldistance::NO_WALL; mesh.n_cells])?;
        let grad_y = gpu.upload(&vec![Vec3::ZERO; mesh.n_cells])?;

        let ctrl = TurbulenceControls {
            nut_max_coeff: 1e3,
            ..controls()
        };

        let mut les = Les::new(
            &gpu,
            &hm,
            &mesh,
            LesModel::Smagorinsky,
            LesCoeffs::default(),
            DeltaSpec::default(),
            ctrl,
            WallFunctionCoeffs::default(),
            &no_walls,
            &y,
            &grad_y,
        )?;
        les.correct(&gpu, &flow)?;
        gpu.sync()?;

        let cap = les.nut_max(nu);
        for (c, &v) in gpu.download(&les.nut().f)?.iter().enumerate() {
            assert!(v <= cap * (1.0 + 1e-12), "cell {c}: nu_t = {v}, cap {cap}");
        }
        assert!(
            gpu.download(&les.nut().f)?[0] >= cap * (1.0 - 1e-12),
            "the cap was never reached, so this measures nothing"
        );

        Ok(())
    }

    /// `simulationType laminar;` on an LES case, and a DNS on a resolved mesh.
    #[test]
    fn freezing_nut_zeroes_it_everywhere() -> Result<()> {
        let Some(gpu) = gpu() else {
            return Ok(());
        };

        let hm = flat_box(4, 0.25);
        let mesh = GpuMesh::upload(&gpu, &hm)?;

        let no_walls = crate::field_setup::WallFaces::none(hm.n_boundary_faces);
        let y = gpu.upload(&vec![crate::walldistance::NO_WALL; mesh.n_cells])?;
        let grad_y = gpu.upload(&vec![Vec3::ZERO; mesh.n_cells])?;

        let mut les = Les::new(
            &gpu,
            &hm,
            &mesh,
            LesModel::Smagorinsky,
            LesCoeffs::default(),
            DeltaSpec::default(),
            controls(),
            WallFunctionCoeffs::default(),
            &no_walls,
            &y,
            &grad_y,
        )?;
        les.freeze_nut(&gpu)?;
        gpu.sync()?;

        for v in gpu.download(&les.nut().f)? {
            assert_eq!(v, 0.0);
        }
        for v in gpu.download(&les.nut().bf)? {
            assert_eq!(v, 0.0);
        }

        Ok(())
    }

    /// Smagorinsky is the model that needs van Driest damping and WALE is not,
    /// and a driver has to be able to say so in its log.
    #[test]
    fn only_smagorinsky_asks_for_van_driest() {
        assert!(LesModel::Smagorinsky.wants_van_driest());
        assert!(!LesModel::Wale.wants_van_driest());
        assert!(!LesModel::Deardorff.wants_van_driest());
    }

    /// The published constants, pinned. `C_w = 0.325` and `C_D = 0.1` are
    /// facts about two papers; `C_s` is a choice and the doc comment on it
    /// says so.
    #[test]
    fn the_model_constants_are_the_published_ones() {
        let c = LesCoeffs::default();
        assert!((c.cw - 0.325).abs() < 1e-15);
        assert!((c.cd - 0.1).abs() < 1e-15);
        assert!((0.1..=0.2).contains(&c.cs), "Cs = {} is outside SPEC-LIT §6.5's range", c.cs);
    }

    /// A filter-width description that mentions a stage the spec does not have
    /// would be a lie in the log; this is the cheap end of that check.
    #[test]
    fn a_wrong_length_wall_distance_is_refused() -> Result<()> {
        let Some(gpu) = gpu() else {
            return Ok(());
        };

        let hm = flat_box(3, 0.25);
        let mesh = GpuMesh::upload(&gpu, &hm)?;
        let no_walls = crate::field_setup::WallFaces::none(hm.n_boundary_faces);

        let short: DevBuf<Scalar> = gpu.zeros(hm.n_cells - 1)?;
        let grad_y = gpu.upload(&vec![Vec3::ZERO; mesh.n_cells])?;

        assert!(Les::new(
            &gpu,
            &hm,
            &mesh,
            LesModel::Wale,
            LesCoeffs::default(),
            DeltaSpec {
                base: BaseDelta::MaxEdge,
                smooth: Some(SmoothSpec::default()),
                ..Default::default()
            },
            controls(),
            WallFunctionCoeffs::default(),
            &no_walls,
            &short,
            &grad_y,
        )
        .is_err());

        Ok(())
    }
}
