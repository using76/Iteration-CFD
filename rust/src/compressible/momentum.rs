// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.
// Provenance: see PROVENANCE.md. No GPL-licensed source was consulted.

//! The rho-weighted momentum equation of the compressible solver, its
//! explicit Newtonian stress and its Rhie-Chow face flux - SPEC-LIT
//! §123.1-§123.3 (docs/17 CMP-09), on the §5.1 schedule, with the time term
//! of §86.3 and the face treatment of body forces §5.1 requires.
//!
//! ```text
//! d(rho u)/dt + div(mdot u) = -grad p + div(tau) + rho g                (123.1)
//! ```
//!
//! `mdot` is a face MASS flux, and every scalar operator is one this crate
//! already ships: the assembly CALLS `fv::` and `ldu_ops::` and the
//! Rhie-Chow kernels of `cuda/momentum.cu` rather than re-deriving any of
//! them, in the order `momentum::Momentum::assemble_component` uses, so
//! that equal inputs give equal bits - which is what the bitwise gate
//! against the kinematic assembly measures.
//!
//! Written from:
//!   F. Moukalled, L. Mangani, M. Darwish, *The Finite Volume Method in
//!     Computational Fluid Dynamics*, Springer (2016), ch. 15-16 - the
//!     conservative-form momentum equation and the pressure-based route,
//!     *not read*, as §124.1 states of the same book; the assembly itself
//!     is this tree's
//!   ofgpu `SPEC-LIT.md` §123.1-§123.3 - the equation, its discretisation,
//!     and the assembly as built, which is the contract of this file
//!   ofgpu `SPEC-LIT.md` §5.1 - Rhie-Chow with a density, and why the
//!     pressure gradient and the body force are held back as `force`
//!   ofgpu `SPEC-LIT.md` §86.3 - `d(rho psi)/dt` and its scheme
//!   ofgpu `SPEC-LIT.md` §7, §12 - the convection weights and the snGrad
//! No GPL-licensed source was consulted.

use cudarc::driver::{CudaFunction, PushKernelArg};

use crate::device::{cfg_for, DevBuf, Gpu, KernelSet};
use crate::error::{Error, Result};
use crate::field::{GpuScalarField, GpuSurfaceScalarField, GpuVectorField};
use crate::field_ops::{self, FieldKernels};
use crate::fv::{self, FvKernels, GradScheme, SnGradScheme};
use crate::io::case::SolverControls;
use crate::ldu::GpuLduMatrix;
use crate::ldu_ops::{self, LduKernels};
use crate::mesh::GpuMesh;
use crate::solver::{self, SolverKernels, SolverPerformance, SolverWorkspace};
use crate::timescheme::{DdtScheme, LtsControls};
use crate::types::Tensor;
use crate::{Label, Scalar, Vec3};

/// A refusal in the one house format.
fn refuse(msg: String) -> Error {
    Error::Refused(format!("compressible::momentum::new: {msg}"))
}

// ==========================================================================
//  Controls
// ==========================================================================

/// Everything the rho-weighted momentum equation reads out of a case.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RhoMomentumControls {
    pub u_solver: SolverControls,

    /// Implicit under-relaxation of the momentum matrix (Patankar §4.9).
    /// §5.2 recommends `≈ 0.7` against `alpha_p ≈ 0.3`.
    pub u_relax: Scalar,

    /// The convection scheme of §7, on the face MASS flux.
    pub div_scheme: crate::fv::DivScheme,

    /// `gradSchemes` for `grad(u)` - §12.1, §12.2. Used for the deferred
    /// correction of `linearUpwind`/`cubic` and for the explicit
    /// non-orthogonal correction.
    pub grad_scheme: GradScheme,

    /// How much of §12.3's explicit over-relaxed correction is applied.
    /// Identically zero on an orthogonal mesh whatever this says.
    pub sn_grad: SnGradScheme,

    /// The time scheme `ddtSchemes` named - §13, on the mass-weighted form
    /// of §86.3. `LocalEuler` and `CrankNicolson` are refused by name in
    /// [`RhoMomentum::new`]; §125.4's per-cell step is CMP-13's.
    pub ddt: DdtScheme,

    pub delta_t: Scalar,

    /// Form the explicit transpose stress of (123.4), `E` of (123.10) and
    /// (123.11). `false` is the laplacian-only operator - what the bitwise
    /// gate against the kinematic assembly compares with, whose own
    /// explicit stress term is a different expression.
    pub full_stress: bool,

    /// Gravitational acceleration. `Vec3::ZERO`, the default, launches no
    /// gravity kernel at all and leaves `phig` a zero buffer.
    pub g: Vec3,
}

impl Default for RhoMomentumControls {
    fn default() -> Self {
        Self {
            u_solver: SolverControls::default(),
            u_relax: 0.7,
            div_scheme: crate::fv::DivScheme::Upwind,
            grad_scheme: GradScheme::GAUSS,
            sn_grad: SnGradScheme::Corrected,
            ddt: DdtScheme::SteadyState,
            delta_t: 1.0,
            full_stress: true,
            g: Vec3::ZERO,
        }
    }
}

// ==========================================================================
//  Kernels
// ==========================================================================

/// The entry points, resolved once. Three modules' worth, because three
/// modules' kernels are reused verbatim: `compressible.cu` for the six of
/// CMP-09, `momentum.cu` for the component views, the convection sensor and
/// the Rhie-Chow chain that take their face force as an argument.
struct RhoMomentumKernels {
    // momentum.cu
    vec_component: CudaFunction,
    set_component: CudaFunction,
    copy_label: CudaFunction,
    mul: CudaFunction,
    mag: CudaFunction,
    rau: CudaFunction,
    hbya: CudaFunction,
    solve_source: CudaFunction,
    face_interp: CudaFunction,
    face_interp_boundary: CudaFunction,
    phi_hbya: CudaFunction,
    phi_hbya_boundary: CudaFunction,
    force_flux: CudaFunction,
    force_flux_boundary: CudaFunction,
    // (121.3)'s kernel pair, with beta = -1 and T_ref = 0: (123.12).
    boussinesq: CudaFunction,
    boussinesq_boundary: CudaFunction,

    // compressible.cu, CMP-09
    cmp_explicit_stress: CudaFunction,
    cmp_add_component: CudaFunction,
    cmp_residual_component: CudaFunction,
    cmp_mass_flux: CudaFunction,
    cmp_mass_flux_boundary: CudaFunction,
    cmp_boundary_traction: CudaFunction,
}

impl RhoMomentumKernels {
    fn new(gpu: &Gpu) -> Result<Self> {
        let m = KernelSet::new(gpu, crate::kernels::MOMENTUM)?;
        let c = KernelSet::new(gpu, crate::kernels::COMPRESSIBLE)?;
        Ok(Self {
            vec_component: m.func("momVecComponent")?,
            set_component: m.func("momSetComponent")?,
            copy_label: m.func("momCopyLabel")?,
            mul: m.func("momMul")?,
            mag: m.func("momMag")?,
            rau: m.func("momRau")?,
            hbya: m.func("momHbyA")?,
            solve_source: m.func("momSolveSource")?,
            face_interp: m.func("momFaceInterp")?,
            face_interp_boundary: m.func("momFaceInterpBoundary")?,
            phi_hbya: m.func("momPhiHbyA")?,
            phi_hbya_boundary: m.func("momPhiHbyABoundary")?,
            force_flux: m.func("momForceFlux")?,
            force_flux_boundary: m.func("momForceFluxBoundary")?,
            boussinesq: m.func("momBoussinesqFlux")?,
            boussinesq_boundary: m.func("momBoussinesqFluxBoundary")?,

            cmp_explicit_stress: c.func("cmpExplicitStress")?,
            cmp_add_component: c.func("cmpAddComponent")?,
            cmp_residual_component: c.func("cmpResidualComponent")?,
            cmp_mass_flux: c.func("cmpMassFlux")?,
            cmp_mass_flux_boundary: c.func("cmpMassFluxBoundary")?,
            cmp_boundary_traction: c.func("cmpBoundaryTraction")?,
        })
    }
}

// ==========================================================================
//  The solver
// ==========================================================================

/// The rho-weighted momentum equation: `d(rho u)/dt + div(mdot u) = -grad p
/// + div(tau) + rho g`, one matrix and three right-hand sides.
///
/// Borrows the mesh for its whole life. Inputs are the face mass flux
/// `mdot` [kg/s], the density `rho` at its three time levels with its
/// boundary values, the effective viscosity `mu_eff` of (122.8), and the
/// gauge `p`; nothing here evaluates the gas law (§123.3).
///
/// The pressure gradient and the body force are held back as
/// [`Self::force`] until the solve, as §5.1 requires of `H`: folding them
/// into the source would put a cell-centred force on the faces, and the
/// collocated checkerboard mode Rhie and Chow removed would walk straight
/// back in.
pub struct RhoMomentum<'m> {
    m: &'m GpuMesh,
    ctrl: RhoMomentumControls,

    fvk: FvKernels,
    lduk: LduKernels,
    fldk: FieldKernels,
    solk: SolverKernels,
    k: RhoMomentumKernels,

    /// The time term of §86.3, `d(rho u)/dt` - SPEC-LIT §13 on the
    /// mass-weighted form.
    pub ddt: crate::timescheme::Ddt,

    a: GpuLduMatrix,
    ws: SolverWorkspace,

    /// The scalar view of one velocity component; refilled per component.
    uc: GpuScalarField,

    /// Per-component source, WITHOUT the pressure gradient and the body
    /// force. Packed as a `Vec3` so one buffer holds all three.
    su: DevBuf<Vec3>,
    /// `-grad p + rho g`, reconstructed from faces (123.13).
    force: DevBuf<Vec3>,
    hbya: DevBuf<Vec3>,
    /// (123.11): the explicit stress `E_P`, an integrated force [N].
    explicit_stress: DevBuf<Vec3>,
    /// (123.14), written one component per [`Self::residual`] pass.
    residual: DevBuf<Vec3>,
    /// (123.16), one entry per boundary face [N].
    traction: DevBuf<Vec3>,

    grad_u: DevBuf<Tensor>,
    grad_uc: DevBuf<Vec3>,
    grad_p: DevBuf<Vec3>,

    /// `rAU = V/diag` [m^3 s/kg], (123.5).
    rau: DevBuf<Scalar>,
    /// `A.u` for one component.
    au: DevBuf<Scalar>,

    /// `|u|`, the scalar a limited convection scheme forms its ratio from.
    u_mag: GpuScalarField,
    grad_u_mag: DevBuf<Vec3>,

    /// §7's convection weights, shared by the three components.
    w: DevBuf<Scalar>,
    bw: DevBuf<Scalar>,

    /// (123.9): `mu_f` on faces, `G_f = mu_f |Sf|` beside it.
    mu_face: GpuSurfaceScalarField,
    mu_mag_sf: GpuSurfaceScalarField,

    rauf: GpuSurfaceScalarField,
    /// (123.6) [m^3/s].
    phi_hbya: GpuSurfaceScalarField,
    /// (123.15) [kg/s].
    mdot_hbya: GpuSurfaceScalarField,
    /// `|Sf| snGrad(p)`.
    sn_grad_p: GpuSurfaceScalarField,
    /// (123.13)'s face flux: `phig - |Sf| snGrad(p)`.
    force_flux: GpuSurfaceScalarField,
    /// (123.12): `rho_f (g . Sf)`. Zero where the velocity prescribes the
    /// flux, and never written at all when `g = 0`.
    phig: GpuSurfaceScalarField,
    /// The body-force flux `momPhiHbyA` reads in [`Self::rhie_chow`]: all
    /// zero and never written, because the gravity part of (123.7) belongs
    /// to the pressure equation, §124.
    zero_body_flux: GpuSurfaceScalarField,
    /// The linear interpolate of `rho` on faces, for (123.12). Written only
    /// when `g != 0`.
    rho_face: GpuSurfaceScalarField,
}

impl<'m> RhoMomentum<'m> {
    /// Build the equation and allocate everything it will ever touch: no
    /// method below allocates, downloads or syncs, so a captured region can
    /// hold them.
    pub fn new(gpu: &Gpu, m: &'m GpuMesh, ctrl: RhoMomentumControls) -> Result<Self> {
        // §123.3: `localEuler` is refused by name - its step is a per-cell
        // preconditioner, and `d(rho u)/dt` with a different dt in every
        // cell conserves nothing (§86.7); §125.4's per-cell step is CMP-13's.
        if matches!(ctrl.ddt, DdtScheme::LocalEuler) {
            return Err(refuse(
                "ddtSchemes named localEuler, and this equation is mass-weighted: \
                 d(rho u)/dt with a different dt in every cell conserves nothing, \
                 and the steady run it serves is CMP-13's to specify (SPEC-LIT \
                 §125.4). Alternative: `Euler` or `backward` for a transient run, \
                 `steadyState` for a steady one"
                    .to_string(),
            ));
        }
        // `CrankNicolson` is refused too: the theta weighting wants the same
        // slot in the assembly the relaxation wants, and a mass-weighted
        // equation under-relaxed by anything below 1 cannot reach it
        // (SPEC-LIT §13.1).
        if matches!(ctrl.ddt, DdtScheme::CrankNicolson(_)) {
            return Err(refuse(
                "ddtSchemes named CrankNicolson: the theta weighting and the \
                 implicit under-relaxation want the same slot in this assembly, \
                 so it is not reachable here. Alternative: `Euler`, `backward`, \
                 or `steadyState`"
                    .to_string(),
            ));
        }
        if !(ctrl.u_relax > 0.0 && ctrl.u_relax <= 1.0) {
            return Err(refuse(format!(
                "u_relax is {}; implicit under-relaxation needs 0 < alpha <= 1 \
                 (SPEC-LIT §5.2)",
                ctrl.u_relax
            )));
        }
        if !ctrl.delta_t.is_finite() || !(ctrl.delta_t > 0.0) {
            return Err(refuse(format!(
                "delta_t is {}; a transient mass-weighted equation needs a \
                 positive finite step",
                ctrl.delta_t
            )));
        }
        if !ctrl.g.x.is_finite() || !ctrl.g.y.is_finite() || !ctrl.g.z.is_finite() {
            return Err(refuse(format!(
                "g = ({}, {}, {}) is not finite; a NaN or infinite gravity would \
                 propagate through (123.12) into every face of the run",
                ctrl.g.x, ctrl.g.y, ctrl.g.z
            )));
        }

        let n = m.n_cells;
        let nif = m.n_internal_faces;
        let nbf = m.n_boundary_faces;

        // A zero-length device allocation is an error rather than an empty
        // buffer, so degenerate counts get one element nothing reads.
        let one = |k: usize| k.max(1);

        Ok(Self {
            m,
            ctrl,

            fvk: FvKernels::new(gpu)?,
            lduk: LduKernels::new(gpu)?,
            fldk: FieldKernels::new(gpu)?,
            solk: SolverKernels::new(gpu)?,
            k: RhoMomentumKernels::new(gpu)?,
            // `relaxed = u_relax < 1.0`, the condition under which
            // `CrankNicolson` would be reachable - refused above, but the
            // flag is what `new_with_relax` asks for.
            ddt: crate::timescheme::Ddt::new_with_relax(
                gpu,
                m,
                ctrl.ddt,
                ctrl.delta_t,
                LtsControls::default(),
                ctrl.u_relax < 1.0,
            )?,

            a: GpuLduMatrix::new(gpu, m)?,
            ws: SolverWorkspace::for_mesh(gpu, m)?,

            uc: GpuScalarField::zeros(gpu, m, "Ucmpt")?,

            su: gpu.zeros(one(n))?,
            force: gpu.zeros(one(n))?,
            hbya: gpu.zeros(one(n))?,
            explicit_stress: gpu.zeros(one(n))?,
            residual: gpu.zeros(one(n))?,
            traction: gpu.zeros(one(nbf))?,

            grad_u: gpu.zeros(one(n))?,
            grad_uc: gpu.zeros(one(n))?,
            grad_p: gpu.zeros(one(n))?,

            rau: gpu.zeros(one(n))?,
            au: gpu.zeros(one(n))?,

            u_mag: GpuScalarField::zeros(gpu, m, "magU")?,
            grad_u_mag: gpu.zeros(one(n))?,

            w: gpu.zeros(one(nif))?,
            bw: gpu.zeros(one(nbf))?,

            mu_face: GpuSurfaceScalarField::zeros(gpu, m, "muEfff")?,
            mu_mag_sf: GpuSurfaceScalarField::zeros(gpu, m, "muEffMagSf")?,

            rauf: GpuSurfaceScalarField::zeros(gpu, m, "rAUf")?,
            phi_hbya: GpuSurfaceScalarField::zeros(gpu, m, "phiHbyA")?,
            mdot_hbya: GpuSurfaceScalarField::zeros(gpu, m, "mdotHbyA")?,
            sn_grad_p: GpuSurfaceScalarField::zeros(gpu, m, "snGradP")?,
            force_flux: GpuSurfaceScalarField::zeros(gpu, m, "forceFlux")?,
            phig: GpuSurfaceScalarField::zeros(gpu, m, "phig")?,
            zero_body_flux: GpuSurfaceScalarField::zeros(gpu, m, "phigZero")?,
            rho_face: GpuSurfaceScalarField::zeros(gpu, m, "rhof")?,
        })
    }

    // ---- accessors --------------------------------------------------------

    pub fn controls(&self) -> &RhoMomentumControls {
        &self.ctrl
    }

    /// The matrix the last component assembly left behind - bit-identical
    /// to the first two passes', which is what all three solves use.
    pub fn matrix(&self) -> &GpuLduMatrix {
        &self.a
    }

    /// `rAU = V/diag` [m^3 s/kg], as of the last [`Self::rhie_chow`].
    pub fn rau(&self) -> &DevBuf<Scalar> {
        &self.rau
    }

    /// `HbyA = rAU H`, as of the last [`Self::rhie_chow`].
    pub fn hbya(&self) -> &DevBuf<Vec3> {
        &self.hbya
    }

    /// `rAU_f`, the face interpolation of [`Self::rau`].
    pub fn rauf(&self) -> &GpuSurfaceScalarField {
        &self.rauf
    }

    /// (123.6) [m^3/s].
    pub fn phi_hbya(&self) -> &GpuSurfaceScalarField {
        &self.phi_hbya
    }

    /// (123.15) [kg/s]: the convective half of (123.7) with the retarded
    /// density of the current iterate.
    pub fn mdot_hbya(&self) -> &GpuSurfaceScalarField {
        &self.mdot_hbya
    }

    /// (123.9): `G_f = mu_f |Sf|`, the laplacian's coefficient.
    pub fn mu_mag_sf(&self) -> &GpuSurfaceScalarField {
        &self.mu_mag_sf
    }

    /// (123.11): the explicit stress `E_P` [N], as of the last
    /// [`Self::set_viscosity`] under `full_stress`.
    pub fn explicit_stress(&self) -> &DevBuf<Vec3> {
        &self.explicit_stress
    }

    /// `|Sf| snGrad(p)`, as of the last [`Self::set_force`].
    pub fn sn_grad_p(&self) -> &GpuSurfaceScalarField {
        &self.sn_grad_p
    }

    /// (123.12): `phig = rho_f (g . Sf)`, zero everywhere when `g = 0`.
    pub fn gravity_flux(&self) -> &GpuSurfaceScalarField {
        &self.phig
    }

    /// `F_f = phig_f - |Sf| snGrad(p)_f`, (123.13)'s face flux.
    pub fn force_flux(&self) -> &GpuSurfaceScalarField {
        &self.force_flux
    }

    /// `reconstruct(F)` [N/m^3], as of the last [`Self::set_force`].
    pub fn force(&self) -> &DevBuf<Vec3> {
        &self.force
    }

    /// (123.14), as of the last [`Self::residual`].
    pub fn residual_field(&self) -> &DevBuf<Vec3> {
        &self.residual
    }

    /// (123.16) [N], one entry per boundary face, as of the last
    /// [`Self::boundary_traction`].
    pub fn traction(&self) -> &DevBuf<Vec3> {
        &self.traction
    }

    // ---- launch helpers ---------------------------------------------------
    //
    //  The four of `momentum.rs`, local because that module's are private.
    //  Every one guards `n == 0`: a zero-block grid is an invalid launch
    //  configuration and not a no-op.

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

    fn launch_mul(
        gpu: &Gpu,
        k: &CudaFunction,
        out: &mut DevBuf<Scalar>,
        a: &DevBuf<Scalar>,
        b: &DevBuf<Scalar>,
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
                .arg(a)
                .arg(b)
                .arg(&nl)
                .launch(cfg_for(n))?;
        }
        Ok(())
    }

    fn launch_mag(
        gpu: &Gpu,
        k: &CudaFunction,
        out: &mut DevBuf<Scalar>,
        src: &DevBuf<Vec3>,
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
                .arg(&nl)
                .launch(cfg_for(n))?;
        }
        Ok(())
    }

    // ---- the component view ----------------------------------------------

    /// Fill [`Self::uc`] with component `cmpt` of `u`, boundary state and
    /// all, so every scalar operator in [`crate::fv`] can be applied to it
    /// unchanged. `momentum.rs`'s own comment on the symmetry-plane
    /// approximation carries over word for word.
    fn fill_component(&mut self, gpu: &Gpu, u: &GpuVectorField, cmpt: Label) -> Result<()> {
        let n = self.m.n_cells;
        let nbf = self.m.n_boundary_faces;

        let Self { k, uc, fldk, .. } = self;

        Self::launch_component(gpu, &k.vec_component, &mut uc.f, &u.f, cmpt, n)?;
        Self::launch_component(gpu, &k.vec_component, &mut uc.f0, &u.f0, cmpt, n)?;
        // psi^{n-2} as well, or BDF2 differences against whatever the
        // allocation held - which is zero, and looks like a solution that
        // jumped.
        Self::launch_component(gpu, &k.vec_component, &mut uc.f00, &u.f00, cmpt, n)?;
        Self::launch_component(gpu, &k.vec_component, &mut uc.bf, &u.bf, cmpt, nbf)?;
        Self::launch_component(gpu, &k.vec_component, &mut uc.ref_value, &u.ref_value, cmpt, nbf)?;
        Self::launch_component(gpu, &k.vec_component, &mut uc.ref_grad, &u.ref_grad, cmpt, nbf)?;

        field_ops::copy_field(gpu, fldk, &mut uc.fr, &u.fr, nbf)?;

        if nbf > 0 {
            let nl = nbf as Label;
            let f = k.copy_label.clone();
            unsafe {
                gpu.stream()
                    .launch_builder(&f)
                    .arg(&mut uc.bc_kind)
                    .arg(&u.bc_kind)
                    .arg(&nl)
                    .launch(cfg_for(nbf))?;
            }
        }

        Ok(())
    }

    // ---- viscosity --------------------------------------------------------

    /// (123.9): `mu_f = w mu_P + (1 - w) mu_N` on internal faces, `mu_b` on
    /// boundary ones, and `G_f = mu_f |Sf|` beside it - the two calls
    /// `momentum::Momentum::update_viscosity` makes for `nu_eff_mag_sf`,
    /// fed by `mu_eff` where that one adds `nu_lam + nu_t`. Under
    /// [`RhoMomentumControls::full_stress`] the cell gradient of `u` and
    /// (123.4)'s explicit stress follow; without it, neither is launched.
    pub fn set_viscosity(&mut self, gpu: &Gpu, u: &GpuVectorField, mu_eff: &GpuScalarField) -> Result<()> {
        let m = self.m;
        let nif = m.n_internal_faces;
        let nbf = m.n_boundary_faces;

        fv::interpolate_linear(gpu, &self.fvk, &mut self.mu_face, mu_eff, m)?;

        {
            let Self { k, mu_face, mu_mag_sf, .. } = self;
            Self::launch_mul(gpu, &k.mul, &mut mu_mag_sf.f, &mu_face.f, &m.mag_sf, nif)?;
            Self::launch_mul(gpu, &k.mul, &mut mu_mag_sf.bf, &mu_face.bf, &m.b_mag_sf, nbf)?;
        }

        if self.ctrl.full_stress {
            {
                let Self { fvk, grad_u, .. } = self;
                fv::fvc_grad_vector_scheme(gpu, fvk, grad_u, u, m, self.ctrl.grad_scheme)?;
            }
            let n = m.n_cells;
            if n > 0 {
                let nl = n as Label;
                let f = self.k.cmp_explicit_stress.clone();
                unsafe {
                    gpu.stream()
                        .launch_builder(&f)
                        .arg(&mut self.explicit_stress)
                        .arg(&self.grad_u)
                        .arg(&self.mu_face.f)
                        .arg(&self.mu_face.bf)
                        .arg(&m.weights)
                        .arg(&m.b_weights)
                        .arg(&m.sf)
                        .arg(&m.b_sf)
                        .arg(&m.owner)
                        .arg(&m.neighbour)
                        .arg(&m.b_nbr_cell)
                        .arg(&m.b_kind)
                        .arg(&m.cf_offset)
                        .arg(&m.cf_face)
                        .arg(&m.cf_own)
                        .arg(&m.bcf_offset)
                        .arg(&m.bcf_face)
                        .arg(&nl)
                        .launch(cfg_for(n))?;
                }
            }
        }

        Ok(())
    }

    // ---- assembly and solve -----------------------------------------------

    /// Assemble the three component systems WITHOUT solving them.
    ///
    /// `PIMPLE/momentumPredictor no` skips the momentum solve, but not the
    /// assembly: `rAU` is `V/diag` and `H` is `(b - sum_N a_N u_N)/V`, so
    /// both come out of a matrix that has to exist whether or not it was
    /// ever inverted (§5.1, §14). [`RhoMomentumControls::full_stress`] must
    /// have had its [`Self::set_viscosity`] first.
    pub fn assemble_only(
        &mut self,
        gpu: &Gpu,
        u: &GpuVectorField,
        mdot: &GpuSurfaceScalarField,
        rho: &GpuScalarField,
    ) -> Result<()> {
        let n = self.m.n_cells;
        if n == 0 {
            return Ok(());
        }
        self.update_div_weights(gpu, u, mdot)?;
        for c in 0..3 {
            self.assemble_component(gpu, u, mdot, rho, c as Label)?;
        }
        Ok(())
    }

    /// Assemble and solve the three component systems. `u` is both the
    /// current value the relaxation is measured against and where the answer
    /// lands. All three sources are built first, with `u` still holding the
    /// previous iterate, so the three systems are the three components of
    /// ONE linearisation rather than a Gauss-Seidel sweep.
    pub fn solve(
        &mut self,
        gpu: &Gpu,
        u: &mut GpuVectorField,
        mdot: &GpuSurfaceScalarField,
        rho: &GpuScalarField,
    ) -> Result<[SolverPerformance; 3]> {
        let n = self.m.n_cells;
        if n == 0 {
            return Ok([SolverPerformance::default(); 3]);
        }

        self.assemble_only(gpu, u, mdot, rho)?;

        let mut perf = [SolverPerformance::default(); 3];
        for c in 0..3 {
            let cmpt = c as Label;

            // The initial guess is the current component, which is what makes
            // a converged SIMPLE iteration cost one sweep.
            {
                let Self { k, uc, .. } = self;
                Self::launch_component(gpu, &k.vec_component, &mut uc.f, &u.f, cmpt, n)?;
            }

            // source = su_cmpt + V force_cmpt - the held-back force joins
            // here and only here (§5.1).
            {
                let nl = n as Label;
                let f = self.k.solve_source.clone();
                unsafe {
                    gpu.stream()
                        .launch_builder(&f)
                        .arg(&mut self.a.source)
                        .arg(&self.su)
                        .arg(&self.force)
                        .arg(&self.m.v)
                        .arg(&cmpt)
                        .arg(&nl)
                        .launch(cfg_for(n))?;
                }
            }

            let Self { solk, uc, a, ws, .. } = self;
            perf[c] = solver::solve(gpu, solk, &mut uc.f, a, self.m, ws, &self.ctrl.u_solver)?;

            {
                let Self { k, uc, .. } = self;
                Self::launch_set_component(gpu, &k.set_component, &mut u.f, &uc.f, cmpt, n)?;
            }
        }

        field_ops::correct_boundary_conditions_vector(gpu, &self.fldk, u, self.m)?;

        Ok(perf)
    }

    /// (123.14), component by component, against the matrix as assembled:
    /// `r_P,c = (su_P,c + V_P force_P,c) - (A u_c)_P`. `u` is not moved.
    pub fn residual(
        &mut self,
        gpu: &Gpu,
        u: &GpuVectorField,
        mdot: &GpuSurfaceScalarField,
        rho: &GpuScalarField,
    ) -> Result<()> {
        let n = self.m.n_cells;
        if n == 0 {
            return Ok(());
        }

        self.assemble_only(gpu, u, mdot, rho)?;

        for c in 0..3 {
            let cmpt = c as Label;
            {
                let Self { k, uc, .. } = self;
                Self::launch_component(gpu, &k.vec_component, &mut uc.f, &u.f, cmpt, n)?;
            }
            {
                let Self { lduk, au, uc, a, .. } = self;
                ldu_ops::amul(gpu, lduk, au, &uc.f, a, self.m)?;
            }
            let nl = n as Label;
            let f = self.k.cmp_residual_component.clone();
            unsafe {
                gpu.stream()
                    .launch_builder(&f)
                    .arg(&mut self.residual)
                    .arg(&self.su)
                    .arg(&self.force)
                    .arg(&self.m.v)
                    .arg(&self.au)
                    .arg(&cmpt)
                    .arg(&nl)
                    .launch(cfg_for(n))?;
            }
        }

        Ok(())
    }

    /// One component's matrix and source, in the order
    /// `momentum::Momentum::assemble_component` uses, so that equal inputs
    /// give equal bits:
    ///
    /// ```text
    /// ddt(rho, u_c) + div(mdot, u_c) - laplacian(G_f, u_c) = ... + E_c
    /// ```
    ///
    /// There is NO bounded-convection correction: in conservative form
    /// `sum_f mdot_f` is the other half of `d(rho u)/dt` (§86.4), not a
    /// spurious source. The relaxation runs ALWAYS - also at 1, where it
    /// only enforces dominance - and the boundary fold comes after it.
    fn assemble_component(
        &mut self,
        gpu: &Gpu,
        u: &GpuVectorField,
        mdot: &GpuSurfaceScalarField,
        rho: &GpuScalarField,
        cmpt: Label,
    ) -> Result<()> {
        let m = self.m;
        let n = m.n_cells;

        self.fill_component(gpu, u, cmpt)?;

        self.a.zero(gpu)?;

        // §86.3: the mass-weighted time derivative, on THIS component's two
        // old levels with `rho` at its three.
        {
            let Self { ddt, a, uc, .. } = self;
            ddt.add_rho(gpu, a, m, &rho.f, &rho.f0, &rho.f00, &uc.f0, &uc.f00, 1.0)?;
        }

        {
            let Self { fvk, a, uc, .. } = self;
            fv::fvm_div_gauss(gpu, fvk, a, m, mdot, &self.w, &self.bw, uc, 1.0)?;
        }

        // The explicit half of a deferred-correction scheme (§11.1) - the
        // gradient of THIS component, a scalar equation's correction three
        // times over (§11.2).
        let scheme = self.ctrl.div_scheme;
        if scheme.correction().is_some() {
            {
                let Self { fvk, grad_uc, uc, .. } = self;
                fv::fvc_grad_scalar_scheme(gpu, fvk, grad_uc, uc, m, self.ctrl.grad_scheme)?;
            }
            let Self { fvk, a, grad_uc, .. } = self;
            fv::fvm_div_correction(gpu, fvk, a, m, mdot, grad_uc, scheme, 1.0)?;
        }

        {
            let Self { fvk, a, uc, mu_mag_sf, .. } = self;
            fv::fvm_laplacian(gpu, fvk, a, m, &mu_mag_sf.f, &mu_mag_sf.bf, uc, -1.0)?;
        }

        if self.ctrl.sn_grad.applies() {
            {
                let Self { fvk, grad_uc, uc, .. } = self;
                fv::fvc_grad_scalar_scheme(gpu, fvk, grad_uc, uc, m, self.ctrl.grad_scheme)?;
            }
            let Self { fvk, a, uc, mu_mag_sf, grad_uc, .. } = self;
            fv::fvm_laplacian_non_orth_correction(
                gpu,
                fvk,
                a,
                m,
                &mu_mag_sf.f,
                &mu_mag_sf.bf,
                uc,
                grad_uc,
                self.ctrl.sn_grad,
                -1.0,
            )?;
        }

        // (123.11) into the source: `E_P` is an integrated force, so it
        // enters as it is, with no volume factor.
        if self.ctrl.full_stress && n > 0 {
            let nl = n as Label;
            let f = self.k.cmp_add_component.clone();
            unsafe {
                gpu.stream()
                    .launch_builder(&f)
                    .arg(&mut self.a.source)
                    .arg(&self.explicit_stress)
                    .arg(&cmpt)
                    .arg(&nl)
                    .launch(cfg_for(n))?;
            }
        }

        // Relaxation BEFORE the boundary fold, for the reason
        // `ldu_ops::relax` states: the increment is (diag' - diag)*psi and
        // does not read the source at all. ALWAYS, also at 1.
        {
            let Self { lduk, a, uc, .. } = self;
            ldu_ops::relax(gpu, lduk, a, m, &uc.f, self.ctrl.u_relax)?;
        }
        ldu_ops::add_boundary_contributions(gpu, &self.lduk, &mut self.a, m)?;

        let Self { k, su, a, .. } = self;
        Self::launch_set_component(gpu, &k.set_component, su, &a.source, cmpt, n)
    }

    /// §7's convection weights, shared by the three components - the same
    /// sensor `momentum::Momentum::update_div_weights` uses (`|u|`), with
    /// `mdot` where it has `phi`.
    fn update_div_weights(
        &mut self,
        gpu: &Gpu,
        u: &GpuVectorField,
        mdot: &GpuSurfaceScalarField,
    ) -> Result<()> {
        let m = self.m;
        let scheme: fv::DivScheme = self.ctrl.div_scheme;

        if scheme.needs_gradient() {
            let n = m.n_cells;
            let nbf = m.n_boundary_faces;
            {
                let Self { k, u_mag, .. } = self;
                Self::launch_mag(gpu, &k.mag, &mut u_mag.f, &u.f, n)?;
                Self::launch_mag(gpu, &k.mag, &mut u_mag.bf, &u.bf, nbf)?;
            }
            fv::fvc_grad_scalar(gpu, &self.fvk, &mut self.grad_u_mag, &self.u_mag, m)?;

            let Self { fvk, w, bw, u_mag, grad_u_mag, .. } = self;
            fv::div_scheme_weights(
                gpu,
                fvk,
                Some(w),
                Some(bw),
                scheme,
                mdot,
                u_mag,
                Some(grad_u_mag),
                m,
            )
        } else {
            let Self { fvk, w, bw, u_mag, .. } = self;
            fv::div_scheme_weights(gpu, fvk, Some(w), Some(bw), scheme, mdot, u_mag, None, m)
        }
    }

    // ---- the face force ---------------------------------------------------

    /// (123.12) and (123.13): `phig = rho_f (g . Sf)`, `F_f = phig_f - |Sf|
    /// snGrad(p)_f`, and `force = reconstruct(F)`.
    ///
    /// The pressure gradient is taken as a FACE difference and only then
    /// reconstructed; `momentum.rs`'s note on the collocated checkerboard
    /// carries over. `phig` is (121.3) with `beta = -1` and `T_ref = 0` -
    /// the subtraction of zero and the two sign changes are exact, so
    /// (123.12) is `rho_f (g . Sf)` with no rounding added. With `g = 0` no
    /// gravity kernel is launched and `phig` stays a zero buffer.
    pub fn set_force(
        &mut self,
        gpu: &Gpu,
        p: &GpuScalarField,
        u: &GpuVectorField,
        rho: &GpuScalarField,
    ) -> Result<()> {
        let m = self.m;

        fv::sn_grad_flux(gpu, &self.fvk, &mut self.sn_grad_p, p, &m.mag_sf, &m.b_mag_sf, m)?;

        if self.ctrl.sn_grad.applies() {
            fv::fvc_grad_scalar(gpu, &self.fvk, &mut self.grad_p, p, m)?;
            fv::sn_grad_flux_correction(
                gpu,
                &self.fvk,
                &mut self.sn_grad_p,
                p,
                &m.mag_sf,
                &m.b_mag_sf,
                &self.grad_p,
                self.ctrl.sn_grad,
                m,
            )?;
        }

        if self.ctrl.g.mag_sqr() > 0.0 {
            let g = self.ctrl.g;
            let t_ref = 0.0 as Scalar;
            let beta = -1.0 as Scalar;
            let nif = m.n_internal_faces;
            let nbf = m.n_boundary_faces;

            fv::interpolate_linear(gpu, &self.fvk, &mut self.rho_face, rho, m)?;

            if nif > 0 {
                let nl = nif as Label;
                let f = self.k.boussinesq.clone();
                unsafe {
                    gpu.stream()
                        .launch_builder(&f)
                        .arg(&mut self.phig.f)
                        .arg(&self.rho_face.f)
                        .arg(&m.sf)
                        .arg(&g.x)
                        .arg(&g.y)
                        .arg(&g.z)
                        .arg(&t_ref)
                        .arg(&beta)
                        .arg(&nl)
                        .launch(cfg_for(nif))?;
                }
            }

            if nbf > 0 {
                let nl = nbf as Label;
                let f = self.k.boussinesq_boundary.clone();
                unsafe {
                    gpu.stream()
                        .launch_builder(&f)
                        .arg(&mut self.phig.bf)
                        .arg(&self.rho_face.bf)
                        .arg(&m.b_sf)
                        .arg(&m.b_kind)
                        .arg(&u.fr)
                        .arg(&u.bc_kind)
                        .arg(&g.x)
                        .arg(&g.y)
                        .arg(&g.z)
                        .arg(&t_ref)
                        .arg(&beta)
                        .arg(&nl)
                        .launch(cfg_for(nbf))?;
                }
            }
        }

        let nif = m.n_internal_faces;
        if nif > 0 {
            let nl = nif as Label;
            let f = self.k.force_flux.clone();
            unsafe {
                gpu.stream()
                    .launch_builder(&f)
                    .arg(&mut self.force_flux.f)
                    .arg(&self.phig.f)
                    .arg(&self.sn_grad_p.f)
                    .arg(&nl)
                    .launch(cfg_for(nif))?;
            }
        }

        let nbf = m.n_boundary_faces;
        if nbf > 0 {
            let nl = nbf as Label;
            let f = self.k.force_flux_boundary.clone();
            unsafe {
                gpu.stream()
                    .launch_builder(&f)
                    .arg(&mut self.force_flux.bf)
                    .arg(&self.phig.bf)
                    .arg(&self.sn_grad_p.bf)
                    .arg(&m.b_kind)
                    .arg(&u.fr)
                    .arg(&u.bc_kind)
                    .arg(&nl)
                    .launch(cfg_for(nbf))?;
            }
        }

        fv::fvc_reconstruct(gpu, &self.fvk, &mut self.force, &self.force_flux, m)
    }

    // ---- Rhie-Chow --------------------------------------------------------

    /// `rAU`, `HbyA`, `rAU_f`, `phi_HbyA` of (123.6) and `mdot_HbyA` of
    /// (123.15) - §5.1's structure with `rho` written where the incompressible
    /// solver has a constant.
    ///
    /// Must be called with the velocity the momentum solve produced, because
    /// `H` is evaluated at that velocity. `phi_HbyA` forms with a ZERO
    /// body-force flux: the gravity part of (123.7) belongs to the pressure
    /// equation, §124. There is no SIMPLEC here.
    pub fn rhie_chow(&mut self, gpu: &Gpu, u: &GpuVectorField, rho: &GpuScalarField) -> Result<()> {
        let m = self.m;
        let n = m.n_cells;
        if n == 0 {
            return Ok(());
        }
        let nl = n as Label;

        // ---- rAU = V/diag, guarded against a zero diagonal ----------------
        {
            let f = self.k.rau.clone();
            unsafe {
                gpu.stream()
                    .launch_builder(&f)
                    .arg(&mut self.rau)
                    .arg(&m.v)
                    .arg(&self.a.diag)
                    .arg(&nl)
                    .launch(cfg_for(n))?;
            }
        }

        // ---- HbyA, one component at a time --------------------------------
        for c in 0..3 {
            let cmpt = c as Label;
            {
                let Self { k, uc, .. } = self;
                Self::launch_component(gpu, &k.vec_component, &mut uc.f, &u.f, cmpt, n)?;
            }
            {
                let Self { lduk, au, uc, a, .. } = self;
                ldu_ops::amul(gpu, lduk, au, &uc.f, a, m)?;
            }
            let f = self.k.hbya.clone();
            unsafe {
                gpu.stream()
                    .launch_builder(&f)
                    .arg(&mut self.hbya)
                    .arg(&self.su)
                    .arg(&self.au)
                    .arg(&self.a.diag)
                    .arg(&self.uc.f)
                    .arg(&cmpt)
                    .arg(&nl)
                    .launch(cfg_for(n))?;
            }
        }

        // ---- rAU on faces -------------------------------------------------
        let nif = m.n_internal_faces;
        if nif > 0 {
            let nfl = nif as Label;
            let f = self.k.face_interp.clone();
            unsafe {
                gpu.stream()
                    .launch_builder(&f)
                    .arg(&mut self.rauf.f)
                    .arg(&self.rau)
                    .arg(&m.weights)
                    .arg(&m.owner)
                    .arg(&m.neighbour)
                    .arg(&nfl)
                    .launch(cfg_for(nif))?;
            }
        }

        let nbf = m.n_boundary_faces;
        if nbf > 0 {
            let nbl = nbf as Label;
            let f = self.k.face_interp_boundary.clone();
            unsafe {
                gpu.stream()
                    .launch_builder(&f)
                    .arg(&mut self.rauf.bf)
                    .arg(&self.rau)
                    .arg(&m.b_weights)
                    .arg(&m.b_face_cells)
                    .arg(&m.b_nbr_cell)
                    .arg(&m.b_kind)
                    .arg(&nbl)
                    .launch(cfg_for(nbf))?;
            }
        }

        // ---- phi_HbyA of (123.6), with a ZERO body-force flux -------------
        if nif > 0 {
            let nfl = nif as Label;
            let f = self.k.phi_hbya.clone();
            unsafe {
                gpu.stream()
                    .launch_builder(&f)
                    .arg(&mut self.phi_hbya.f)
                    .arg(&self.hbya)
                    .arg(&self.rauf.f)
                    .arg(&self.zero_body_flux.f)
                    .arg(&m.weights)
                    .arg(&m.sf)
                    .arg(&m.owner)
                    .arg(&m.neighbour)
                    .arg(&nfl)
                    .launch(cfg_for(nif))?;
            }
        }

        if nbf > 0 {
            let nbl = nbf as Label;
            let f = self.k.phi_hbya_boundary.clone();
            unsafe {
                gpu.stream()
                    .launch_builder(&f)
                    .arg(&mut self.phi_hbya.bf)
                    .arg(&self.hbya)
                    .arg(&u.bf)
                    .arg(&self.rau)
                    .arg(&self.zero_body_flux.bf)
                    .arg(&m.b_sf)
                    .arg(&m.b_weights)
                    .arg(&m.b_face_cells)
                    .arg(&m.b_nbr_cell)
                    .arg(&m.b_kind)
                    .arg(&u.fr)
                    .arg(&u.bc_kind)
                    .arg(&nbl)
                    .launch(cfg_for(nbf))?;
            }
        }

        // ---- mdot_HbyA of (123.15) ----------------------------------------
        if nif > 0 {
            let nfl = nif as Label;
            let f = self.k.cmp_mass_flux.clone();
            unsafe {
                gpu.stream()
                    .launch_builder(&f)
                    .arg(&mut self.mdot_hbya.f)
                    .arg(&self.phi_hbya.f)
                    .arg(&rho.f)
                    .arg(&m.owner)
                    .arg(&m.neighbour)
                    .arg(&nfl)
                    .launch(cfg_for(nif))?;
            }
        }

        if nbf > 0 {
            let nbl = nbf as Label;
            let f = self.k.cmp_mass_flux_boundary.clone();
            unsafe {
                gpu.stream()
                    .launch_builder(&f)
                    .arg(&mut self.mdot_hbya.bf)
                    .arg(&self.phi_hbya.bf)
                    .arg(&rho.f)
                    .arg(&rho.bf)
                    .arg(&m.b_face_cells)
                    .arg(&m.b_nbr_cell)
                    .arg(&m.b_kind)
                    .arg(&nbl)
                    .launch(cfg_for(nbf))?;
            }
        }

        Ok(())
    }

    // ---- the boundary traction --------------------------------------------

    /// (123.16): the viscous traction on every boundary face [N], from the
    /// gradient and the `mu_f` boundary values [`Self::set_viscosity`] left
    /// behind. With `full_stress = false` only the wall-normal term is
    /// formed - the laplacian-only operator's traction.
    pub fn boundary_traction(&mut self, gpu: &Gpu, u: &GpuVectorField) -> Result<()> {
        let m = self.m;
        let nbf = m.n_boundary_faces;
        if nbf == 0 {
            return Ok(());
        }
        let nbl = nbf as Label;
        let full: Label = if self.ctrl.full_stress { 1 } else { 0 };

        let f = self.k.cmp_boundary_traction.clone();
        unsafe {
            gpu.stream()
                .launch_builder(&f)
                .arg(&mut self.traction)
                .arg(&u.f)
                .arg(&u.bf)
                .arg(&self.grad_u)
                .arg(&self.mu_face.bf)
                .arg(&m.b_sf)
                .arg(&m.b_mag_sf)
                .arg(&m.b_delta_coeffs)
                .arg(&m.b_face_cells)
                .arg(&m.b_kind)
                .arg(&full)
                .arg(&nbl)
                .launch(cfg_for(nbf))?;
        }

        Ok(())
    }
}

// ==========================================================================
//  The capture gate - SPEC-LIT 81.7's row for this module (CMP-09)
// ==========================================================================

#[cfg(test)]
mod tests {
    use crate as ofgpu;

    include!("../../tests/shared/compressible_momentum_cases.rs");

    /// §81.7's row for `src/compressible/momentum.rs`: one rho-weighted
    /// predictor - `set_viscosity`, `set_force`, a fixed-sweep `solve` and
    /// `rhie_chow` - captures into one graph and replays bitwise, on the
    /// precedent of `the_momentum_predictor_replays_bitwise` and thermo's
    /// gate. The seed `u`, `rho` and `mu_eff` are non-uniform, the linear
    /// solver sweeps a fixed 4 iterations with residuals unreported
    /// (§81.7's capture conditions), and the scheme is Euler.
    #[test]
    fn the_compressible_momentum_predictor_replays_bitwise() {
        let Some(g) = gpu() else { return };
        let hm = blockgen::build_mesh(&box_spec(
            [4, 3, 2],
            [0.0, 0.0, 0.0],
            [1.0, 0.8, 0.6],
            [1.0, 2.0, 1.0],
        ))
        .expect("mesh");
        let gm = GpuMesh::upload(&g, &hm).expect("mesh");
        let fvk = FvKernels::new(&g).expect("fv kernels");

        let mut ctrl = RhoMomentumControls {
            u_relax: 1.0,
            ddt: DdtScheme::Euler,
            delta_t: 1.0e-3,
            ..Default::default()
        };
        ctrl.u_solver.fixed_iters = true;
        ctrl.u_solver.max_iter = 4;
        ctrl.u_solver.report_residuals = false;

        // The constant inputs, outside the captured state: `p` stays zero,
        // `rho` and `mu_eff` are non-uniform, and `mdot` is the flux of a
        // throwaway non-uniform seed.
        let n = hm.n_cells;
        let nbf = hm.n_boundary_faces;
        let p = GpuScalarField::zeros(&g, &gm, "p").expect("p");
        let mut rho = GpuScalarField::zeros(&g, &gm, "rho").expect("rho");
        let rv: Vec<Scalar> = (0..n).map(|i| 1.2 + 0.01 * (i % 7) as Scalar).collect();
        g.write(&mut rho.f, &rv).expect("rho");
        g.write(&mut rho.f0, &rv).expect("rho");
        let rb: Vec<Scalar> = (0..nbf).map(|i| 1.2 + 0.01 * (i % 5) as Scalar).collect();
        g.write(&mut rho.bf, &rb).expect("rho bf");
        let mut mu_eff = GpuScalarField::zeros(&g, &gm, "muEff").expect("muEff");
        let mv: Vec<Scalar> = (0..n)
            .map(|i| 1.8e-5 * (1.0 + 0.1 * (i % 3) as Scalar))
            .collect();
        g.write(&mut mu_eff.f, &mv).expect("mu");
        let mb: Vec<Scalar> = (0..nbf)
            .map(|i| 1.8e-5 * (1.0 + 0.1 * (i % 4) as Scalar))
            .collect();
        g.write(&mut mu_eff.bf, &mb).expect("mu bf");
        let mut mdot = GpuSurfaceScalarField::zeros(&g, &gm, "mdot").expect("mdot");
        {
            let mut seed = GpuVectorField::zeros(&g, &gm, "seed").expect("seed");
            let sv: Vec<Vec3> = (0..n)
                .map(|i| Vec3::new(0.1 + 0.01 * (i % 5) as Scalar, 0.02, 0.0))
                .collect();
            g.write(&mut seed.f, &sv).expect("seed");
            fv::interpolate_vector_flux(&g, &fvk, &mut mdot, &seed, &gm)
                .expect("mdot");
        }

        let report = crate::capture::capture_replays_bitwise(
            &g,
            "the compressible rho-weighted momentum predictor (SPEC-LIT 123.3)",
            || {
                let m = RhoMomentum::new(&g, &gm, ctrl)?;
                let mut u = GpuVectorField::zeros(&g, &gm, "U")?;
                let uv: Vec<Vec3> = (0..n)
                    .map(|i| {
                        Vec3::new(
                            0.1 + 0.01 * (i % 5) as Scalar,
                            0.02 - 0.005 * (i % 3) as Scalar,
                            0.0,
                        )
                    })
                    .collect();
                g.write(&mut u.f, &uv)?;
                let u0: Vec<Vec3> = uv.iter().map(|v| *v * 0.9).collect();
                g.write(&mut u.f0, &u0)?;
                Ok((m, u))
            },
            |(m, u): &mut (_, GpuVectorField)| {
                m.set_viscosity(&g, u, &mu_eff)?;
                m.set_force(&g, &p, u, &rho)?;
                m.solve(&g, u, &mdot, &rho)?;
                m.rhie_chow(&g, u, &rho)
            },
            |(m, u): &(_, GpuVectorField)| {
                let v = g.download(&u.f)?;
                let mut flat = Vec::with_capacity(v.len() * 3);
                for c in &v {
                    flat.push(c.x);
                    flat.push(c.y);
                    flat.push(c.z);
                }
                let h = g.download(m.hbya())?;
                let mut hf = Vec::with_capacity(h.len() * 3);
                for c in &h {
                    hf.push(c.x);
                    hf.push(c.y);
                    hf.push(c.z);
                }
                Ok(vec![
                    ("U", flat),
                    ("rau", g.download(m.rau())?),
                    ("HbyA", hf),
                    ("mdot_HbyA", g.download(&m.mdot_hbya().f)?),
                ])
            },
        )
        .expect(
            "SPEC-LIT 81.7: the compressible momentum predictor must capture \
             and replay bitwise",
        );
        println!("  compressible momentum: {report}");
    }
}
