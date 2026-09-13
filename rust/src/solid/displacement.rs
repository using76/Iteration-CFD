// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.

//! The displacement operator on the device: ONE application of the
//! segregated thermo-elastic fixed-point map that [`super::prototype`]
//! measures host-side, in the crate's own kernel plumbing. The outer loop,
//! the relaxation and the stress field are later units' work - nothing here
//! iterates on its own.
//!
//! Written from:
//!   I. Demirdžić, S. Muzaferija, *Int. J. Numer. Methods Eng.* 37 (1994)
//!     3751-3766, DOI 10.1002/nme.1620372110 - the segregated cell-centred
//!     formulation, and the statement that a boundary face's contribution to
//!     the equilibrium sum IS its prescribed traction
//!   I. Demirdžić, S. Muzaferija, *Comput. Methods Appl. Mech. Eng.* 125
//!     (1995) 235-255, DOI 10.1016/0045-7825(95)00800-G
//!   H. Jasak, H. G. Weller, *Int. J. Numer. Methods Eng.* 48 (2000) 267-287,
//!     DOI 10.1002/(SICI)1097-0207(20000520)48:2<267::AID-NME884>3.0.CO;2-Q -
//!     the `(2 mu + lambda)` implicit split and its convergence behaviour
//!   I. Demirdžić, D. Martinović, *Comput. Methods Appl. Mech. Eng.* 109
//!     (1993) 331-349, DOI 10.1016/0045-7825(93)90085-C - the thermal-strain
//!     term in finite-volume form
//!   B. A. Boley, J. H. Weiner, *Theory of Thermal Stresses*, Wiley (1960),
//!     ch. 1 - Duhamel-Neumann, and the free-expansion state
//!   S. P. Timoshenko, J. N. Goodier, *Theory of Elasticity*, 3rd ed.,
//!     McGraw-Hill (1970) ch. 1 - small-strain isotropic elasticity and the
//!     Lamé conversion
//!   Y. Saad, *Iterative Methods for Sparse Linear Systems*, 2nd ed., SIAM
//!     (2003), DOI 10.1137/1.9780898718003, §10.2-10.3 - the diagonal
//!     incomplete factorisation the linear solve's preconditioner rebuilds
//!   ofgpu `SPEC-LIT.md` §1, §2.4, §3.2, §3.5, §4, §8.2, §8.4, §21, §81
//!
//! OpenFOAM and solids4foam are GPL and were not opened; the kernels are the
//! device port of [`super::prototype`], diffed against it stage by stage to
//! 1e-12. No GPL-licensed source was consulted.
//!
//! # One application of the map
//!
//! `apply` is [`super::prototype::Prototype::apply_map`] with the device
//! standing in for the host: `passes` boundary sub-passes (gradient, the
//! traction condition solved AT the face, the boundary values), then the
//! gradient again and the boundary gradient, then the deferred stress
//! gather and the thermal load into ONE right-hand side, then three scalar
//! systems solved by [`crate::solver::solve`]. The matrix is re-assembled
//! per component per application rather than built once, and the
//! preconditioner is rebuilt by `solve` on every solve - the moment a later
//! unit lets the mesh or the material move, nothing here is stale.
//!
//! The thermal load is a masked face gather, not the volumetric
//! `-(3 lambda + 2 mu) alpha V_P (grad T)_P`: a traction face's thermal
//! contribution is excluded (its prescribed traction IS its whole
//! contribution, Demirdžić & Muzaferija 1994), and with a free face excluded
//! the closed-cell sum and the volumetric form differ by the entire thermal
//! driving force.
//!
//! The displacement field [`Displacement::u`] is a plain
//! [`GpuVectorField`], but its boundary values are evaluated by THIS
//! module's kernel and not by `field_ops::correct_boundary_conditions`,
//! because a vector field carries one `fr` per FACE while a symmetry plane
//! needs `fr = 1` in its normal component and `fr = 0` in the tangential
//! ones - the per-component mask is what carries that. `u.fr`,
//! `u.ref_value`, `u.bc_kind`, `u.f0` and `u.f00` are allocated by `zeros`
//! and never read here. The per-component scalar the operators assemble
//! with carries `ref_grad = 0` on every face: with `fr` in {0,1} the
//! laplacian then folds the fixed value into the source through the
//! boundary coefficients and nothing on a traction component, which is
//! exactly the manual fold of the prototype; the correction's limiter scale
//! is 1 under the corrected scheme, so the zero never reaches the numbers.
//!
//! A coupled patch (`Cyclic`, `Processor`, `Interface`) is refused by name
//! at construction - no kernel here reads `b_nbr_cell`; empty patches are
//! accepted and skipped by the mesh's patch kind, never by the mask. And
//! `apply` writes an internal `next` buffer that `picard_step` and `apply`
//! copy out of, so both share one body without borrowing it and `&mut self`
//! at once.

use cudarc::driver::{CudaFunction, PushKernelArg};

use crate::device::{cfg_for, DevBuf, Gpu, KernelSet};
use crate::error::{Error, Result};
use crate::field::{GpuScalarField, GpuVectorField};
use crate::field_ops::{self, FieldKernels};
use crate::fv::{self, FvKernels, SnGradScheme};
use crate::io::case::SolverControls;
use crate::ldu::GpuLduMatrix;
use crate::ldu_ops::{self, LduKernels};
use crate::mesh::{GpuMesh, HostMesh};
use crate::solver::{self, SolverKernels, SolverPerformance, SolverWorkspace};
use crate::{Label, Scalar, Tensor, Vec3};

use super::bc::{CompBc, SolidBcs};
use super::materials::{Bonds, BondTreatment, MaterialMap, PerCell};
use super::Material;

/// The per-cell material arrays the displacement and stress kernels read,
/// resident for the operator's whole life. On a one-material map every
/// entry is that material's own constant, and the kernels' arithmetic is
/// the one-material operator's to the bit.
pub struct DeviceMaterials {
    pub mu: DevBuf<Scalar>,
    pub lambda: DevBuf<Scalar>,
    pub alpha: DevBuf<Scalar>,
    pub beta_alpha: DevBuf<Scalar>,
    pub t_ref: DevBuf<Scalar>,
}

impl DeviceMaterials {
    /// Resident once, at construction; `t_ref` is rewritten by
    /// [`Displacement::set_temperature`], the rest never moves.
    pub fn upload(gpu: &Gpu, pc: &PerCell, t_ref: &[Scalar]) -> Result<Self> {
        Ok(Self {
            mu: gpu.upload(&pc.mu)?,
            lambda: gpu.upload(&pc.lambda)?,
            alpha: gpu.upload(&pc.alpha)?,
            beta_alpha: gpu.upload(&pc.beta_alpha)?,
            t_ref: gpu.upload(t_ref)?,
        })
    }
}

/// Entry points of `cuda/solid.cu`, resolved once.
struct SolidKernels {
    vec_component: CudaFunction,
    set_component: CudaFunction,
    add_component: CudaFunction,
    grad_component: CudaFunction,
    evaluate_boundary: CudaFunction,
    boundary_gradient: CudaFunction,
    traction_ref_grad: CudaFunction,
    div_sigma_exp: CudaFunction,
    thermal_load: CudaFunction,
    boundary_traction: CudaFunction,
    bond_face: CudaFunction,
    bond_grad_corr: CudaFunction,
}

impl SolidKernels {
    fn new(gpu: &Gpu) -> Result<Self> {
        let k = KernelSet::new(gpu, crate::kernels::SOLID)?;
        Ok(Self {
            vec_component: k.func("solidVecComponent")?,
            set_component: k.func("solidSetComponent")?,
            add_component: k.func("solidAddComponent")?,
            grad_component: k.func("solidGradComponent")?,
            evaluate_boundary: k.func("solidEvaluateBoundary")?,
            boundary_gradient: k.func("solidBoundaryGradient")?,
            traction_ref_grad: k.func("solidTractionRefGrad")?,
            div_sigma_exp: k.func("solidDivSigmaExp")?,
            thermal_load: k.func("solidThermalLoad")?,
            boundary_traction: k.func("solidBoundaryTraction")?,
            bond_face: k.func("solidBondFace")?,
            bond_grad_corr: k.func("solidBondGradCorr")?,
        })
    }
}

fn launch_component(
    gpu: &Gpu,
    k: &CudaFunction,
    out: &mut DevBuf<Scalar>,
    src: &DevBuf<Vec3>,
    cmpt: Label,
    n: usize,
) -> Result<()> {
    if n == 0 {
        return Ok(());
    }
    let nl = n as Label;
    let f = k.clone();
    unsafe {
        gpu.stream()
            .launch_builder(&f)
            .arg(out)
            .arg(src)
            .arg(&cmpt)
            .arg(&nl)
            .launch(cfg_for(n))?;
    }
    Ok(())
}

fn launch_set_component(
    gpu: &Gpu,
    k: &CudaFunction,
    out: &mut DevBuf<Vec3>,
    src: &DevBuf<Scalar>,
    cmpt: Label,
    n: usize,
) -> Result<()> {
    if n == 0 {
        return Ok(());
    }
    let nl = n as Label;
    let f = k.clone();
    unsafe {
        gpu.stream()
            .launch_builder(&f)
            .arg(out)
            .arg(src)
            .arg(&cmpt)
            .arg(&nl)
            .launch(cfg_for(n))?;
    }
    Ok(())
}

/// The displacement operator: the prototype's map, once per call, on the
/// device.
pub struct Displacement<'m> {
    m: &'m GpuMesh,
    pub material: Material,
    pub bcs: SolidBcs,
    pub ctrl: SolverControls,
    /// The scheme selector of the non-orthogonal correction; `Corrected`,
    /// the prototype's default.
    pub sn_grad: SnGradScheme,
    /// Boundary sub-passes per application; 3, the prototype's default.
    pub passes: usize,
    pub t_ref: Scalar,
    /// Displacement. `f` = cell values, `bf` = evaluated boundary values,
    /// `ref_grad` = the value the traction condition solved for. `fr`,
    /// `ref_value`, `bc_kind`, `f0` and `f00` are never read here (the
    /// module doc says why the boundary is this module's own kernel).
    pub u: GpuVectorField,
    /// `[n_cells]` temperature.
    pub t: DevBuf<Scalar>,
    /// `[n_bf]` boundary temperature.
    pub bt: DevBuf<Scalar>,
    /// `[n_cells]` `grad(u)`, `G_ij = du_j/dx_i` (SPEC-LIT §1, §3.5).
    pub grad: DevBuf<Tensor>,
    /// `[n_bf]` the boundary gradient, its normal row replaced; zero on
    /// empty faces.
    pub b_grad: DevBuf<Tensor>,
    /// `[n_cells]` deferred stress + thermal load, computed once per
    /// application and folded into all three components.
    pub rhs: DevBuf<Vec3>,
    /// The material map the operator was built from - host-only; the
    /// kernels see [`Self::cells`].
    pub map: MaterialMap,
    /// The per-cell material arrays on the device.
    pub cells: DeviceMaterials,
    /// The bond faces, as the host sees them.
    pub bonds: Bonds,
    face_bond: DevBuf<Label>,
    bond_face: DevBuf<Label>,
    /// `[n_bond]` what each bond face says: the face displacement and the
    /// face traction, rewritten by every sub-pass.
    pub bond_u: DevBuf<Vec3>,
    pub bond_t: DevBuf<Vec3>,
    /// The treatment the bond faces carry: 0 Series, 1 Linear.
    bond_mode: Label,
    /// `F(u)`, written by `apply_inner`; `picard_step` and `apply` copy it
    /// out, so both share one body.
    next: DevBuf<Vec3>,
    a: GpuLduMatrix,
    ws: SolverWorkspace,
    /// The per-component scalar view the operators assemble with. `ref_grad`
    /// stays zero (the module doc says why that changes nothing); `f` is the
    /// warm start AND the solution of the last component solved.
    uc: GpuScalarField,
    grad_uc: DevBuf<Vec3>,
    gamma_mag_sf: DevBuf<Scalar>,
    b_gamma_mag_sf: DevBuf<Scalar>,
    fvk: FvKernels,
    lduk: LduKernels,
    fldk: FieldKernels,
    solk: SolverKernels,
    sk: SolidKernels,
}

impl<'m> Displacement<'m> {
    /// Validate the material, check the host mesh against the device one,
    /// scatter the boundary statement, size the solver workspace, and
    /// allocate every buffer - once; nothing here allocates again.
    /// The one-material constructor, kept so no existing call site changes:
    /// one material everywhere, no bond faces. The map-taking constructor
    /// is [`Self::with_materials`].
    pub fn new(
        gpu: &Gpu,
        m: &'m GpuMesh,
        host: &HostMesh,
        material: Material,
        per_patch: &[[CompBc; 3]],
        ctrl: SolverControls,
    ) -> Result<Self> {
        Self::with_materials(
            gpu,
            m,
            host,
            &MaterialMap::uniform("material", material, host.n_cells),
            per_patch,
            ctrl,
        )
    }

    /// The operator on a region carrying several bonded materials. Every
    /// material is validated; the implicit coefficients come from the map -
    /// the series value at a bond face, the one-material number to the bit
    /// on every other face; the per-cell arrays are uploaded once; the bond
    /// faces are found on the host mesh and refused if it is degenerate
    /// there (SPEC-LIT §95.8). `T_ref` starts at zero on every cell and is
    /// written by [`Self::set_temperature`].
    #[allow(clippy::too_many_lines)]
    pub fn with_materials(
        gpu: &Gpu,
        m: &'m GpuMesh,
        host: &HostMesh,
        map: &MaterialMap,
        per_patch: &[[CompBc; 3]],
        ctrl: SolverControls,
    ) -> Result<Self> {
        for mat in &map.materials {
            mat.validate()?;
        }
        if host.n_cells != m.n_cells {
            return Err(Error::Config(format!(
                "solid: the host mesh has {} cells but the device mesh has {}",
                host.n_cells,
                m.n_cells
            )));
        }
        let bcs = SolidBcs::new(gpu, host, per_patch)?;
        let ws = SolverWorkspace::for_mesh(gpu, m)?;

        // The implicit coefficient per face, host-built from the map: the
        // series value at a bond face, coefficient times magnitude to the
        // bit on every same-material face. Never changes after this.
        let (gamma_mag_sf, b_gamma_mag_sf) = map.implicit_coefficients(host);
        let bonds = map.bonds(host)?;
        let nb = bonds.n_bond().max(1);
        let mut bond_face = bonds.bond_face.clone();
        bond_face.resize(nb, 0);
        let cells = DeviceMaterials::upload(gpu, &map.per_cell(), &vec![0.0 as Scalar; m.n_cells])?;

        Ok(Self {
            m,
            material: map.materials[0],
            bcs,
            ctrl,
            sn_grad: SnGradScheme::Corrected,
            passes: 3,
            t_ref: 0.0,
            u: GpuVectorField::zeros(gpu, m, "solid displacement")?,
            t: gpu.zeros(m.n_cells)?,
            bt: gpu.zeros(m.n_boundary_faces)?,
            grad: gpu.zeros(m.n_cells)?,
            b_grad: gpu.zeros(m.n_boundary_faces)?,
            rhs: gpu.zeros(m.n_cells)?,
            map: map.clone(),
            cells,
            face_bond: gpu.upload(&bonds.face_bond)?,
            bond_face: gpu.upload(&bond_face)?,
            bond_u: gpu.zeros(nb)?,
            bond_t: gpu.zeros(nb)?,
            bond_mode: match map.bond {
                BondTreatment::Series => 0,
                BondTreatment::Linear => 1,
            },
            bonds,
            next: gpu.zeros(m.n_cells)?,
            a: GpuLduMatrix::new(gpu, m)?,
            ws,
            uc: GpuScalarField::zeros(gpu, m, "solid displacement component")?,
            grad_uc: gpu.zeros(m.n_cells)?,
            gamma_mag_sf: gpu.upload(&gamma_mag_sf)?,
            b_gamma_mag_sf: gpu.upload(&b_gamma_mag_sf)?,
            fvk: FvKernels::new(gpu)?,
            lduk: LduKernels::new(gpu)?,
            fldk: FieldKernels::new(gpu)?,
            solk: SolverKernels::new(gpu)?,
            sk: SolidKernels::new(gpu)?,
        })
    }

    /// How many bond faces the region carries - zero on a one-material map,
    /// which is the switch every bond kernel launch here reads.
    pub fn n_bond(&self) -> usize {
        self.bonds.n_bond()
    }

    /// The two bond kernels, run after a plain gradient sweep: what each
    /// bond face says (its displacement and traction, from the cell
    /// gradients the sweep just produced), then the Green-Gauss correction
    /// that carries the face's own displacement into the two bond cells'
    /// gradients. Skipped entirely on a one-material map - a host branch on
    /// a host integer, so a capture around this sequence stays valid.
    fn run_bond_kernels(&mut self, gpu: &Gpu) -> Result<()> {
        if self.bonds.n_bond() == 0 {
            return Ok(());
        }
        let nbond = self.bonds.n_bond() as Label;
        let nl = self.m.n_cells as Label;
        let n = self.m.n_cells;
        let mode = self.bond_mode;
        unsafe {
            let f = self.sk.bond_face.clone();
            gpu.stream()
                .launch_builder(&f)
                .arg(&mut self.bond_u)
                .arg(&mut self.bond_t)
                .arg(&self.bond_face)
                .arg(&self.u.f)
                .arg(&self.grad)
                .arg(&self.t)
                .arg(&self.cells.mu)
                .arg(&self.cells.lambda)
                .arg(&self.cells.beta_alpha)
                .arg(&self.cells.t_ref)
                .arg(&self.m.weights)
                .arg(&self.m.sf)
                .arg(&self.m.mag_sf)
                .arg(&self.m.cf)
                .arg(&self.m.c)
                .arg(&self.m.owner)
                .arg(&self.m.neighbour)
                .arg(&mode)
                .arg(&nbond)
                .launch(cfg_for(self.bonds.n_bond()))?;
        }
        unsafe {
            let f = self.sk.bond_grad_corr.clone();
            gpu.stream()
                .launch_builder(&f)
                .arg(&mut self.grad)
                .arg(&self.bond_u)
                .arg(&self.face_bond)
                .arg(&self.u.f)
                .arg(&self.m.weights)
                .arg(&self.m.sf)
                .arg(&self.m.v)
                .arg(&self.m.owner)
                .arg(&self.m.neighbour)
                .arg(&self.m.cf_offset)
                .arg(&self.m.cf_face)
                .arg(&self.m.cf_own)
                .arg(&nl)
                .launch(cfg_for(n))?;
        }
        Ok(())
    }

    /// Setup only (`gpu.write`): the temperature field and its boundary
    /// values, and `t_ref` with them. The conjugate solve hands this the
    /// real `T`; a gate that wants `T - T_ref = 0` writes `t_ref` into both.
    pub fn set_temperature(
        &mut self,
        gpu: &Gpu,
        t: &[Scalar],
        bt: &[Scalar],
        t_ref: Scalar,
    ) -> Result<()> {
        if t.len() != self.m.n_cells {
            return Err(Error::Config(format!(
                "solid: temperature has {} values for {} cells",
                t.len(),
                self.m.n_cells
            )));
        }
        if bt.len() != self.m.n_boundary_faces {
            return Err(Error::Config(format!(
                "solid: boundary temperature has {} values for {} boundary \
                 faces",
                bt.len(),
                self.m.n_boundary_faces
            )));
        }
        gpu.write(&mut self.t, t)?;
        gpu.write(&mut self.bt, bt)?;
        // The per-cell `T_ref`: a material's own value where the map
        // overrides, the region's scalar where it does not.
        gpu.write(&mut self.cells.t_ref, &self.map.t_ref_per_cell(t_ref))?;
        self.t_ref = t_ref;
        Ok(())
    }

    /// Setup only (`gpu.write`): displacement, cells and boundary faces.
    pub fn set_displacement(
        &mut self,
        gpu: &Gpu,
        u: &[Vec3],
        ub: &[Vec3],
    ) -> Result<()> {
        if u.len() != self.m.n_cells {
            return Err(Error::Config(format!(
                "solid: displacement has {} values for {} cells",
                u.len(),
                self.m.n_cells
            )));
        }
        if ub.len() != self.m.n_boundary_faces {
            return Err(Error::Config(format!(
                "solid: boundary displacement has {} values for {} boundary \
                 faces",
                ub.len(),
                self.m.n_boundary_faces
            )));
        }
        gpu.write(&mut self.u.f, u)?;
        gpu.write(&mut self.u.bf, ub)
    }

    /// Mirror of `Prototype::correct_boundary`: `passes` rounds of [gradient,
    /// traction condition, boundary values] to close the circle between
    /// SPEC-LIT §3.5's gradient and §4's boundary values, then the gradient
    /// once more and the boundary gradient from it.
    pub fn correct_boundary(&mut self, gpu: &Gpu) -> Result<()> {
        let nbf = self.m.n_boundary_faces;
        for _ in 0..self.passes.max(1) {
            fv::fvc_grad_vector(gpu, &self.fvk, &mut self.grad, &self.u, self.m)?;
            self.run_bond_kernels(gpu)?;
            if nbf > 0 {
                let nl = nbf as Label;
                let f = self.sk.traction_ref_grad.clone();
                unsafe {
                    gpu.stream()
                        .launch_builder(&f)
                        .arg(&mut self.u.ref_grad)
                        .arg(&self.grad)
                        .arg(&self.bcs.traction)
                        .arg(&self.bcs.mask)
                        .arg(&self.m.b_face_cells)
                        .arg(&self.m.b_sf)
                        .arg(&self.bt)
                        .arg(&self.m.b_kind)
                        .arg(&self.cells.mu)
                        .arg(&self.cells.lambda)
                        .arg(&self.cells.beta_alpha)
                        .arg(&self.cells.t_ref)
                        .arg(&nl)
                        .launch(cfg_for(nbf))?;
                }
                let f = self.sk.evaluate_boundary.clone();
                unsafe {
                    gpu.stream()
                        .launch_builder(&f)
                        .arg(&mut self.u.bf)
                        .arg(&self.u.f)
                        .arg(&self.bcs.ref_value)
                        .arg(&self.u.ref_grad)
                        .arg(&self.bcs.mask)
                        .arg(&self.m.b_face_cells)
                        .arg(&self.m.b_delta_coeffs)
                        .arg(&nl)
                        .launch(cfg_for(nbf))?;
                }
            }
        }
        fv::fvc_grad_vector(gpu, &self.fvk, &mut self.grad, &self.u, self.m)?;
        self.run_bond_kernels(gpu)?;
        if nbf > 0 {
            let nl = nbf as Label;
            let f = self.sk.boundary_gradient.clone();
            unsafe {
                gpu.stream()
                    .launch_builder(&f)
                    .arg(&mut self.b_grad)
                    .arg(&self.grad)
                    .arg(&self.u.f)
                    .arg(&self.u.bf)
                    .arg(&self.m.b_face_cells)
                    .arg(&self.m.b_sf)
                    .arg(&self.m.b_delta_coeffs)
                    .arg(&self.m.b_kind)
                    .arg(&nl)
                    .launch(cfg_for(nbf))?;
            }
        }
        Ok(())
    }

    /// The right-hand side that does not depend on the component: zero it
    /// (a memset node, legal inside a capture), gather the deferred stress,
    /// subtract the thermal load.
    pub fn assemble_rhs(&mut self, gpu: &Gpu) -> Result<()> {
        gpu.fill_zero(&mut self.rhs)?;
        let n = self.m.n_cells;
        if n == 0 {
            return Ok(());
        }
        let nl = n as Label;
        {
            let f = self.sk.div_sigma_exp.clone();
            unsafe {
                gpu.stream()
                    .launch_builder(&f)
                    .arg(&mut self.rhs)
                    .arg(&self.grad)
                    .arg(&self.b_grad)
                    .arg(&self.bcs.mask)
                    .arg(&self.m.sf)
                    .arg(&self.m.weights)
                    .arg(&self.m.owner)
                    .arg(&self.m.neighbour)
                    .arg(&self.m.b_sf)
                    .arg(&self.m.b_kind)
                    .arg(&self.m.cf_offset)
                    .arg(&self.m.cf_face)
                    .arg(&self.m.cf_own)
                    .arg(&self.m.bcf_offset)
                    .arg(&self.m.bcf_face)
                    .arg(&self.cells.mu)
                    .arg(&self.cells.lambda)
                    .arg(&self.face_bond)
                    .arg(&self.bond_t)
                    .arg(&self.u.f)
                    .arg(&self.gamma_mag_sf)
                    .arg(&self.m.delta_coeffs)
                    .arg(&self.m.non_orth_corr)
                    .arg(&self.m.mag_sf)
                    .arg(&nl)
                    .launch(cfg_for(n))?;
            }
        }
        let f = self.sk.thermal_load.clone();
        unsafe {
            gpu.stream()
                .launch_builder(&f)
                .arg(&mut self.rhs)
                .arg(&self.t)
                .arg(&self.bt)
                .arg(&self.bcs.mask)
                .arg(&self.m.sf)
                .arg(&self.m.weights)
                .arg(&self.m.owner)
                .arg(&self.m.neighbour)
                .arg(&self.m.b_sf)
                .arg(&self.m.b_kind)
                .arg(&self.m.cf_offset)
                .arg(&self.m.cf_face)
                .arg(&self.m.cf_own)
                .arg(&self.m.bcf_offset)
                .arg(&self.m.bcf_face)
                .arg(&self.cells.beta_alpha)
                .arg(&self.cells.t_ref)
                .arg(&self.face_bond)
                .arg(&nl)
                .launch(cfg_for(n))?;
        }
        Ok(())
    }

    /// One component's scalar system, in the prototype's assembly order:
    /// zero the matrix; `uc.f <- u.f[cmpt]` (which is also the warm start
    /// `solve` reads); the boundary triple from the mask; the Gauss
    /// laplacian of `(2 mu + lambda)|Sf|`; the correction contracted against
    /// this component's gradient column; the traction fold; the shared
    /// right-hand side; the boundary-coefficient fold. Deterministic in the
    /// current state, so a test may call it again after `apply` to read the
    /// matrix of any component.
    pub fn assemble_component(&mut self, gpu: &Gpu, cmpt: Label) -> Result<()> {
        let m = self.m;
        let n = m.n_cells;
        let nbf = m.n_boundary_faces;

        self.a.zero(gpu)?;
        launch_component(gpu, &self.sk.vec_component, &mut self.uc.f, &self.u.f, cmpt, n)?;
        field_ops::copy_field(
            gpu,
            &self.fldk,
            &mut self.uc.fr,
            &self.bcs.fr[cmpt as usize],
            nbf,
        )?;
        launch_component(
            gpu,
            &self.sk.vec_component,
            &mut self.uc.ref_value,
            &self.bcs.ref_value,
            cmpt,
            nbf,
        )?;
        // uc.ref_grad stays zero: on a fixed component the boundary
        // coefficients carry the value, and on a traction component the
        // traction kernel below adds `t|Sf|` instead - the manual fold of
        // the prototype, with the SOLVED ref_grad living on u.ref_grad for
        // the boundary values only.
        fv::fvm_laplacian(
            gpu,
            &self.fvk,
            &mut self.a,
            m,
            &self.gamma_mag_sf,
            &self.b_gamma_mag_sf,
            &self.uc,
            -1.0,
        )?;
        if n > 0 {
            let nl = n as Label;
            let f = self.sk.grad_component.clone();
            unsafe {
                gpu.stream()
                    .launch_builder(&f)
                    .arg(&mut self.grad_uc)
                    .arg(&self.grad)
                    .arg(&cmpt)
                    .arg(&nl)
                    .launch(cfg_for(n))?;
            }
        }
        if self.sn_grad.applies() {
            fv::fvm_laplacian_non_orth_correction(
                gpu,
                &self.fvk,
                &mut self.a,
                m,
                &self.gamma_mag_sf,
                &self.b_gamma_mag_sf,
                &self.uc,
                &self.grad_uc,
                self.sn_grad,
                -1.0,
            )?;
        }
        if n > 0 {
            let nl = n as Label;
            let f = self.sk.boundary_traction.clone();
            unsafe {
                gpu.stream()
                    .launch_builder(&f)
                    .arg(&mut self.a.source)
                    .arg(&self.bcs.traction)
                    .arg(&self.bcs.mask)
                    .arg(&self.m.b_mag_sf)
                    .arg(&self.m.b_kind)
                    .arg(&self.m.bcf_offset)
                    .arg(&self.m.bcf_face)
                    .arg(&cmpt)
                    .arg(&nl)
                    .launch(cfg_for(n))?;
            }
        }
        launch_component(gpu, &self.sk.add_component, &mut self.a.source, &self.rhs, cmpt, n)?;
        ldu_ops::add_boundary_contributions(gpu, &self.lduk, &mut self.a, m)
    }

    /// ONE application of the map into `self.next`: the boundary sub-passes,
    /// the shared right-hand side, and per component an assembly, a warm
    /// start and a solve. `u.f` is not written - the outer loop owns the
    /// update. Allocates nothing, downloads nothing, so it captures.
    fn apply_inner(&mut self, gpu: &Gpu) -> Result<[SolverPerformance; 3]> {
        self.correct_boundary(gpu)?;
        self.assemble_rhs(gpu)?;
        let n = self.m.n_cells;
        let mut perf = [SolverPerformance::default(); 3];
        for cmpt in 0..3 {
            self.assemble_component(gpu, cmpt)?;
            perf[cmpt as usize] = solver::solve(
                gpu,
                &self.solk,
                &mut self.uc.f,
                &self.a,
                self.m,
                &mut self.ws,
                &self.ctrl,
            )?;
            launch_set_component(gpu, &self.sk.set_component, &mut self.next, &self.uc.f, cmpt, n)?;
        }
        Ok(perf)
    }

    /// `apply_inner`, then `next` copied out to `out`: the map, leaving `u`
    /// untouched. `out.len()` must be the cell count or refused by name.
    pub fn apply(&mut self, gpu: &Gpu, out: &mut DevBuf<Vec3>) -> Result<[SolverPerformance; 3]> {
        if out.len() != self.m.n_cells {
            return Err(Error::Config(format!(
                "solid: apply: out has {} elements for {} cells",
                out.len(),
                self.m.n_cells
            )));
        }
        let perf = self.apply_inner(gpu)?;
        field_ops::copy_field_vector(gpu, &self.fldk, out, &self.next, self.m.n_cells)?;
        Ok(perf)
    }

    /// `apply_inner`, then `next` copied into `u.f`: a plain Picard step,
    /// relaxation factor 1. The capture gate iterates THIS; the outer loop
    /// of the next unit drives `apply` and puts its relaxation between.
    pub fn picard_step(&mut self, gpu: &Gpu) -> Result<[SolverPerformance; 3]> {
        let perf = self.apply_inner(gpu)?;
        field_ops::copy_field_vector(gpu, &self.fldk, &mut self.u.f, &self.next, self.m.n_cells)?;
        Ok(perf)
    }

    /// The last component's assembled matrix.
    pub fn matrix(&self) -> &GpuLduMatrix {
        &self.a
    }

    /// `F(u)` from the last application.
    pub fn next(&self) -> &DevBuf<Vec3> {
        &self.next
    }

    /// The mesh the operator was built on.
    pub fn mesh(&self) -> &GpuMesh {
        self.m
    }
}
