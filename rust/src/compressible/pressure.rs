// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.
// Provenance: see PROVENANCE.md. No GPL-licensed source was consulted.

//! The all-speed pressure equation of the compressible solver: continuity
//! with the gas law, the retarded density, and the incompressible limit -
//! SPEC-LIT §124.1-§124.4 (docs/17 CMP-10), built on [`super::momentum`]'s
//! Rhie-Chow face fields.
//!
//! ```text
//! A p = b
//! A p = laplacian(rho_f rAU_f, p) - sum_f psiPhi_f p~_f - a_n V_P psi_P p_P  (124.7)
//! b   = sum_f s_f + V_P ( a_n (rho_P* - psi_P p_P*) + a_0 rho_P^o + a_00 rho_P^oo )
//! psiPhi_f = psi~_f phi_HbyA,f
//! s0_f     = (rho~_f* - psi~_f p~_f*) phi_HbyA,f ,  s_f = s0_f + rho_f rAU_f phig_f
//! ```
//!
//! The system assembled is (124.4) multiplied by -1, so its laplacian enters
//! as `fvm_laplacian(.., +1)` exactly as the incompressible pressure
//! equation's does, and at `psi = 0` with uniform `rho` the matrix IS that
//! equation's. Everything §3, §5.1, §7 and §12 already compute is CALLED,
//! not re-derived; the seven kernels of `cuda/compressible.cu` do only what
//! no existing kernel does - the retarded face products, the shared time
//! term, the face axpy, the mass flux (124.10), and the linearisation
//! (124.11) of `rho` about `p*`.
//!
//! Written from:
//!   ofgpu `SPEC-LIT.md` §124.1-§124.4 - the derivation from continuity,
//!     the matrix, the incompressible limit, and the assembly as built,
//!     which is the contract of this file
//!   ofgpu `SPEC-LIT.md` §123.3 - the `RhoMomentum` face fields this module
//!     consumes: `rAU`, `HbyA`, `rAU_f`, `phi_HbyA`, `phig`, `force`
//!   ofgpu `SPEC-LIT.md` §5.1 - the incompressible pressure equation whose
//!     sign, laplacian and `snGrad` flux this module copies
//!   ofgpu `SPEC-LIT.md` §7 - the convection weights, upwind or limited,
//!     that serve both the convective operator and every retarded value
//!   ofgpu `SPEC-LIT.md` §12.3 - the explicit non-orthogonal correction
//!   Karki & Patankar (1989); Demirdžić, Lilek & Perić (1993); Hafez, South
//!     & Murman (1979) - the all-speed pressure-based literature, *not
//!     read*, as §124.1 states of them
//! No GPL-licensed source was consulted.

use cudarc::driver::{CudaFunction, PushKernelArg};

use crate::device::{cfg_for, DevBuf, Gpu, KernelSet};
use crate::error::{Error, Result};
use crate::field::{GpuScalarField, GpuSurfaceScalarField, GpuVectorField};
use crate::field_ops::{self, FieldKernels};
use crate::fv::{self, DivScheme, FvKernels, Limiter, SnGradScheme};
use crate::ldu::GpuLduMatrix;
use crate::ldu_ops::{self, LduKernels};
use crate::mesh::GpuMesh;
use crate::solver::{
    self, LinearSolverKind, Preconditioner, SolverControls, SolverKernels, SolverPerformance,
    SolverWorkspace,
};
use crate::timescheme::DdtCoeffs;
use crate::{Label, Scalar, Vec3};

/// A refusal in the one house format.
fn refuse(msg: String) -> Error {
    Error::Refused(format!("compressible::pressure::new: {msg}"))
}

// ==========================================================================
//  Controls
// ==========================================================================

/// How the retarded face values of (124.3) are taken: §124.4 "Face values".
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Retarded {
    /// The cell upwind of `phi_HbyA,f` - §3.1's upwind weights, the default.
    Upwind,
    /// A §7-limited face value, the limiter named as §7 names it - §124.1's
    /// option `limitedRetarded`, with `p` as the sensor.
    Limited(Limiter),
}

/// Everything the all-speed pressure equation reads out of a case.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RhoPressureControls {
    /// The linear solver of §8.1. `PCG`, `DIC` and `GAMG` are refused by
    /// name in [`RhoPressure::new`]: the matrix is asymmetric, and only the
    /// incompressible pressure equation has the AMGX backend (§8.3). The
    /// default is `PBiCGStab` with `DILU`.
    pub p_solver: SolverControls,

    /// The retarded face values of (124.3).
    pub retarded: Retarded,

    /// §12.3's snGrad scheme, for the laplacian's explicit correction and
    /// `correct`'s flux.
    pub sn_grad: SnGradScheme,

    /// §5.2's non-orthogonal correctors. The linearisation point does not
    /// move between them (§124.4 "The order").
    pub n_non_orth_corr: usize,
}

impl Default for RhoPressureControls {
    fn default() -> Self {
        Self {
            p_solver: SolverControls {
                precon: Preconditioner::Dilu,
                ..Default::default()
            },
            retarded: Retarded::Upwind,
            sn_grad: SnGradScheme::Corrected,
            n_non_orth_corr: 0,
        }
    }
}

// ==========================================================================
//  Kernels
// ==========================================================================

/// The entry points, resolved once. Three modules' worth, because three
/// modules' kernels are reused verbatim: `compressible.cu` for the seven of
/// CMP-10, `momentum.cu` for the face product and the velocity correction,
/// `simple.cu` for the face-flux sum.
struct RhoPressureKernels {
    // compressible.cu, CMP-10
    retarded_flux: CudaFunction,
    retarded_flux_boundary: CudaFunction,
    pressure_ddt: CudaFunction,
    face_axpy: CudaFunction,
    pressure_mass_flux: CudaFunction,
    pressure_mass_flux_boundary: CudaFunction,
    linearized_rho: CudaFunction,
    // momentum.cu
    mul: CudaFunction,
    correct_velocity: CudaFunction,
    // simple.cu
    face_flux_sum: CudaFunction,
}

impl RhoPressureKernels {
    fn new(gpu: &Gpu) -> Result<Self> {
        let c = KernelSet::new(gpu, crate::kernels::COMPRESSIBLE)?;
        let mo = KernelSet::new(gpu, crate::kernels::MOMENTUM)?;
        let si = KernelSet::new(gpu, crate::kernels::SIMPLE)?;
        Ok(Self {
            retarded_flux: c.func("cmpRetardedFlux")?,
            retarded_flux_boundary: c.func("cmpRetardedFluxBoundary")?,
            pressure_ddt: c.func("cmpPressureDdt")?,
            face_axpy: c.func("cmpFaceAxpy")?,
            pressure_mass_flux: c.func("cmpPressureMassFlux")?,
            pressure_mass_flux_boundary: c.func("cmpPressureMassFluxBoundary")?,
            linearized_rho: c.func("cmpLinearizedRho")?,
            mul: mo.func("momMul")?,
            correct_velocity: mo.func("momCorrectVelocity")?,
            face_flux_sum: si.func("smpFaceFluxSum")?,
        })
    }
}

// ==========================================================================
//  The solver
// ==========================================================================

/// The all-speed pressure equation: (124.4) multiplied by -1, `(124.7)`
/// with its source `(124.8)`, solved for the gauge `p` alone.
///
/// Borrows the mesh for its whole life, and reads
/// [`RhoMomentum`](super::momentum::RhoMomentum)'s face fields - `rAU_f`,
/// `phi_HbyA`, `phig` - plus the time coefficients of its own ddt, so that
/// continuity and momentum share one time scheme, as (124.1) requires. The
/// gas law is never evaluated here: `psi`, `rho*` and `p` arrive as fields,
/// exactly as §123.3 kept the momentum equation free of them.
///
/// No reference level is fixed and nothing is pinned (§124.4): with
/// `psi > 0` a transient system has its time diagonal, and a steady one
/// needs a face where `p` is a Dirichlet value.
pub struct RhoPressure<'m> {
    m: &'m GpuMesh,
    ctrl: RhoPressureControls,

    fvk: FvKernels,
    lduk: LduKernels,
    fldk: FieldKernels,
    solk: SolverKernels,
    k: RhoPressureKernels,

    a: GpuLduMatrix,
    ws: SolverWorkspace,

    /// §7's weights, shared by the convective operator and every retarded
    /// value (§124.4 "Face values").
    w: DevBuf<Scalar>,
    bw: DevBuf<Scalar>,
    /// The Gauss `grad p` scratch: the limiter's `r` under
    /// [`Retarded::Limited`], and step 5's gradient otherwise.
    grad_p: DevBuf<Vec3>,

    /// `rho_f` - linear on internal faces, `rho*_b` on the boundary.
    rho_face: GpuSurfaceScalarField,
    /// `rho_f rAU_f`, formed in that order.
    rho_rauf: GpuSurfaceScalarField,
    /// `(rho_f rAU_f) |Sf|`, the laplacian's coefficient.
    rho_rauf_mag_sf: GpuSurfaceScalarField,

    /// `psiPhi_f = psi~_f phi_HbyA,f` (124.8).
    psi_phi: GpuSurfaceScalarField,
    /// `s0_f = (rho~_f* - psi~_f p~_f*) phi_HbyA,f` (124.8).
    s0: GpuSurfaceScalarField,
    /// `s_f = s0_f + rho_f rAU_f phig_f` (124.8).
    s: GpuSurfaceScalarField,

    /// The linearisation point `p*`, cell and boundary values.
    p_star: DevBuf<Scalar>,
    p_star_b: DevBuf<Scalar>,

    /// (124.10)'s `lap_f`, a scratch of [`Self::correct`].
    lap: GpuSurfaceScalarField,
    /// `A p` scratch of [`Self::residual`].
    au: DevBuf<Scalar>,
    /// `r = source - A p`, written by [`Self::residual`].
    residual: DevBuf<Scalar>,
}

impl<'m> RhoPressure<'m> {
    /// Build the equation and allocate everything it will ever touch: no
    /// method below allocates, downloads or syncs, so a captured region can
    /// hold [`Self::solve`] and [`Self::correct`].
    pub fn new(gpu: &Gpu, m: &'m GpuMesh, ctrl: RhoPressureControls) -> Result<Self> {
        // The matrix carries (124.7)'s convective term, `psi~ phi_HbyA` on
        // one side and no transpose of it anywhere: it is asymmetric, so the
        // symmetric-only machinery is refused by name (SPEC-LIT §8.2).
        if ctrl.p_solver.solver == LinearSolverKind::PCG {
            return Err(refuse(
                "solvers/p/solver named PCG, and the all-speed pressure matrix \
                 is ASYMMETRIC: (124.7)'s retarded convection term has no \
                 transpose, so conjugate gradients - defined only for symmetric \
                 systems (SPEC-LIT §8.2) - are not guaranteed to converge on \
                 it. Alternative: `PBiCGStab` with `DILU`"
                    .to_string(),
            ));
        }
        if ctrl.p_solver.precon == Preconditioner::Dic {
            return Err(refuse(
                "solvers/p/preconditioner named DIC, and the all-speed pressure \
                 matrix is asymmetric: DIC is the incomplete CHOLESKY \
                 factorisation (SPEC-LIT §21) and an asymmetric matrix has no \
                 Cholesky factor to approximate. Alternative: `PBiCGStab` with \
                 `DILU`, the asymmetric case of the same algorithm"
                    .to_string(),
            ));
        }
        // The AMGX backend is the incompressible pressure equation's alone
        // (SPEC-LIT §8.3); this equation is a PBiCGStab system.
        if ctrl.p_solver.solver == LinearSolverKind::Gamg {
            return Err(refuse(
                "solvers/p/solver named GAMG, and only the incompressible \
                 pressure equation has the AMGX backend (SPEC-LIT §8.3); the \
                 all-speed equation is not served by it. Alternative: \
                 `PBiCGStab` with `DILU`"
                    .to_string(),
            ));
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
            k: RhoPressureKernels::new(gpu)?,

            a: GpuLduMatrix::new(gpu, m)?,
            ws: SolverWorkspace::for_mesh(gpu, m)?,

            w: gpu.zeros(one(nif))?,
            bw: gpu.zeros(one(nbf))?,
            grad_p: gpu.zeros(one(n))?,

            rho_face: GpuSurfaceScalarField::zeros(gpu, m, "rhoFRauf")?,
            rho_rauf: GpuSurfaceScalarField::zeros(gpu, m, "rhoRauf")?,
            rho_rauf_mag_sf: GpuSurfaceScalarField::zeros(gpu, m, "rhoRaufMagSf")?,

            psi_phi: GpuSurfaceScalarField::zeros(gpu, m, "psiPhi")?,
            s0: GpuSurfaceScalarField::zeros(gpu, m, "s0")?,
            s: GpuSurfaceScalarField::zeros(gpu, m, "sRhoP")?,

            p_star: gpu.zeros(one(n))?,
            p_star_b: gpu.zeros(one(nbf))?,

            lap: GpuSurfaceScalarField::zeros(gpu, m, "lapRhoP")?,
            au: gpu.zeros(one(n))?,
            residual: gpu.zeros(one(n))?,
        })
    }

    // ---- accessors --------------------------------------------------------

    pub fn controls(&self) -> &RhoPressureControls {
        &self.ctrl
    }

    /// The matrix the last assembly left behind - the last non-orthogonal
    /// corrector's, after a [`Self::solve`].
    pub fn matrix(&self) -> &GpuLduMatrix {
        &self.a
    }

    /// `psiPhi_f` of (124.8), as of the last preparation.
    pub fn psi_phi(&self) -> &GpuSurfaceScalarField {
        &self.psi_phi
    }

    /// `s0_f` of (124.8), as of the last preparation.
    pub fn s0(&self) -> &GpuSurfaceScalarField {
        &self.s0
    }

    /// `s_f = s0_f + rho_f rAU_f phig_f` of (124.8).
    pub fn source_flux(&self) -> &GpuSurfaceScalarField {
        &self.s
    }

    /// `rho_f rAU_f`, as of the last preparation.
    pub fn rho_rauf(&self) -> &GpuSurfaceScalarField {
        &self.rho_rauf
    }

    /// `(rho_f rAU_f) |Sf|`, the laplacian's coefficient.
    pub fn rho_rauf_mag_sf(&self) -> &GpuSurfaceScalarField {
        &self.rho_rauf_mag_sf
    }

    /// The linearisation point `p*` in the cells, as of the last
    /// preparation.
    pub fn p_star(&self) -> &DevBuf<Scalar> {
        &self.p_star
    }

    /// `r = source - A p`, as of the last [`Self::residual`].
    pub fn residual_field(&self) -> &DevBuf<Scalar> {
        &self.residual
    }

    // ---- launch helpers ---------------------------------------------------
    //
    //  Every one guards n == 0: a zero-block grid is an invalid launch
    //  configuration and not a no-op.

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

    /// `out = a + x*y` on one array of faces - `cmpFaceAxpy`.
    fn launch_axpy(
        gpu: &Gpu,
        k: &CudaFunction,
        out: &mut DevBuf<Scalar>,
        a: &DevBuf<Scalar>,
        x: &DevBuf<Scalar>,
        y: &DevBuf<Scalar>,
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
                .arg(x)
                .arg(y)
                .arg(&nl)
                .launch(cfg_for(n))?;
        }
        Ok(())
    }

    // ---- the preparation --------------------------------------------------

    /// Everything steps 1-7 read: `p*` (cell and boundary values), the
    /// weights, `psiPhi`, `s0`, `s` and `rho_f rAU_f` - formed ONCE per
    /// [`Self::assemble_only`] or [`Self::solve`], from the `p`, `rho*` and
    /// `psi` handed in. The linearisation point does not move through the
    /// non-orthogonal correctors (§124.4).
    ///
    /// `p`'s, `psi`'s and `rho`'s boundary values must be evaluated - the
    /// retarded inflow selection of (124.9) and `rho_f` read `bf` directly.
    fn prepare(
        &mut self,
        gpu: &Gpu,
        mom: &super::momentum::RhoMomentum<'_>,
        p: &GpuScalarField,
        rho: &GpuScalarField,
        psi: &GpuScalarField,
    ) -> Result<()> {
        let m = self.m;
        let n = m.n_cells;
        let nif = m.n_internal_faces;
        let nbf = m.n_boundary_faces;

        // p* : the linearisation point, cell and boundary values.
        solver::vec_copy(gpu, &self.solk, &mut self.p_star, &p.f, n)?;
        solver::vec_copy(gpu, &self.solk, &mut self.p_star_b, &p.bf, nbf)?;

        // One set of weights serves the convective operator and every
        // retarded value (§124.4 "Face values"). `p` is the limiter's
        // sensor, with its Gauss gradient forming `r`.
        match self.ctrl.retarded {
            Retarded::Upwind => {
                let Self { fvk, w, bw, .. } = self;
                fv::div_scheme_weights(
                    gpu,
                    fvk,
                    Some(w),
                    Some(bw),
                    DivScheme::Upwind,
                    mom.phi_hbya(),
                    p,
                    None,
                    m,
                )?;
            }
            Retarded::Limited(l) => {
                fv::fvc_grad_scalar(gpu, &self.fvk, &mut self.grad_p, p, m)?;
                let Self { fvk, w, bw, grad_p, .. } = self;
                fv::div_scheme_weights(
                    gpu,
                    fvk,
                    Some(w),
                    Some(bw),
                    DivScheme::Limited(l),
                    mom.phi_hbya(),
                    p,
                    Some(grad_p),
                    m,
                )?;
            }
        }

        // rho_f: linear on internal faces, rho*_b on the boundary. Then
        // rho_f rAU_f, and the laplacian's coefficient beside it, each
        // formed as one product in that order (§124.4).
        fv::interpolate_linear(gpu, &self.fvk, &mut self.rho_face, rho, m)?;
        {
            let Self {
                k,
                rho_face,
                rho_rauf,
                rho_rauf_mag_sf,
                ..
            } = self;
            Self::launch_mul(
                gpu,
                &k.mul,
                &mut rho_rauf.f,
                &rho_face.f,
                &mom.rauf().f,
                nif,
            )?;
            Self::launch_mul(
                gpu,
                &k.mul,
                &mut rho_rauf.bf,
                &rho_face.bf,
                &mom.rauf().bf,
                nbf,
            )?;
            Self::launch_mul(
                gpu,
                &k.mul,
                &mut rho_rauf_mag_sf.f,
                &rho_rauf.f,
                &m.mag_sf,
                nif,
            )?;
            Self::launch_mul(
                gpu,
                &k.mul,
                &mut rho_rauf_mag_sf.bf,
                &rho_rauf.bf,
                &m.b_mag_sf,
                nbf,
            )?;
        }

        // psiPhi and s0 of (124.8), the retarded face products.
        {
            let Self {
                k,
                psi_phi,
                s0,
                w,
                bw,
                p_star,
                p_star_b,
                ..
            } = self;
            if nif > 0 {
                let nl = nif as Label;
                let f = k.retarded_flux.clone();
                unsafe {
                    gpu.stream()
                        .launch_builder(&f)
                        .arg(&mut psi_phi.f)
                        .arg(&mut s0.f)
                        .arg(&mom.phi_hbya().f)
                        .arg(w)
                        .arg(&psi.f)
                        .arg(&rho.f)
                        .arg(&mut *p_star)
                        .arg(&m.owner)
                        .arg(&m.neighbour)
                        .arg(&nl)
                        .launch(cfg_for(nif))?;
                }
            }

            if nbf > 0 {
                let nl = nbf as Label;
                let f = k.retarded_flux_boundary.clone();
                unsafe {
                    gpu.stream()
                        .launch_builder(&f)
                        .arg(&mut psi_phi.bf)
                        .arg(&mut s0.bf)
                        .arg(&mom.phi_hbya().bf)
                        .arg(bw)
                        .arg(&psi.f)
                        .arg(&psi.bf)
                        .arg(&rho.f)
                        .arg(&rho.bf)
                        .arg(&mut *p_star)
                        .arg(&mut *p_star_b)
                        .arg(&m.b_face_cells)
                        .arg(&m.b_nbr_cell)
                        .arg(&m.b_kind)
                        .arg(&nl)
                        .launch(cfg_for(nbf))?;
                }
            }
        }

        // s = s0 + rho_f rAU_f phig, on internal and boundary faces. With
        // g = 0 `phig` is the zero buffer `RhoMomentum` left behind, and the
        // sum adds nothing.
        {
            let Self {
                k,
                s,
                s0,
                rho_rauf,
                ..
            } = self;
            Self::launch_axpy(
                gpu,
                &k.face_axpy,
                &mut s.f,
                &s0.f,
                &rho_rauf.f,
                &mom.gravity_flux().f,
                nif,
            )?;
            Self::launch_axpy(
                gpu,
                &k.face_axpy,
                &mut s.bf,
                &s0.bf,
                &rho_rauf.bf,
                &mom.gravity_flux().bf,
                nbf,
            )?;
        }

        Ok(())
    }

    // ---- the matrix -------------------------------------------------------

    /// (124.7)'s ONE system, in the order §124.4 fixes:
    ///
    /// 1. zero the matrix;
    /// 2. the time term, with the momentum equation's own coefficients;
    /// 3. the retarded convection of `p` at flux `psiPhi`;
    /// 4. `fvm_laplacian((rho_f rAU_f) |Sf|, p, +1)`;
    /// 5. §12.3's explicit correction with the Gauss `grad p` of the
    ///    current `p`;
    /// 6. `source += sum_f ±s_f`, accumulating;
    /// 7. the boundary fold.
    fn assemble_matrix(
        &mut self,
        gpu: &Gpu,
        mom: &super::momentum::RhoMomentum<'_>,
        p: &GpuScalarField,
        rho: &GpuScalarField,
        psi: &GpuScalarField,
    ) -> Result<()> {
        let m = self.m;
        let n = m.n_cells;

        // 1. zero the matrix, as `Simple::assemble_pressure` does.
        self.a.zero(gpu)?;

        // 2. the time term - the momentum equation's own ddt, so that
        // continuity and momentum share one time scheme (§124.4). A
        // steadyState ddt returns `DdtCoeffs::ZERO`, and no kernel runs.
        let co: DdtCoeffs = mom.ddt.state.coeffs(mom.ddt.scheme)?;
        if let Some(ls) = mom.local_step() {
            // §125.7: the momentum equation's per-cell step, the same sign.
            let sign = -1.0 as Scalar;
            ls.add_pressure_ddt(gpu, &mut self.a, m, &psi.f, &rho.f, &self.p_star, &rho.f0, sign)?;
        } else if !(co.a_n == 0.0 && co.a_0 == 0.0 && co.a_00 == 0.0) && n > 0 {
            let sign = -1.0 as Scalar;
            let nl = n as Label;
            let f = self.k.pressure_ddt.clone();
            unsafe {
                gpu.stream()
                    .launch_builder(&f)
                    .arg(&mut self.a.diag)
                    .arg(&mut self.a.source)
                    .arg(&m.v)
                    .arg(&psi.f)
                    .arg(&rho.f)
                    .arg(&self.p_star)
                    .arg(&rho.f0)
                    .arg(&rho.f00)
                    .arg(&co.a_n)
                    .arg(&co.a_0)
                    .arg(&co.a_00)
                    .arg(&sign)
                    .arg(&nl)
                    .launch(cfg_for(n))?;
            }
        }

        // 3. the convective operator on `p` at face flux `psiPhi`, with the
        // same weights - the boundary faces carry p's triple.
        {
            let Self {
                fvk,
                a,
                psi_phi,
                w,
                bw,
                ..
            } = self;
            fv::fvm_div_gauss(gpu, fvk, a, m, psi_phi, w, bw, p, -1.0)?;
        }

        // 4. the laplacian, at +1 exactly as §5.1's pressure equation's.
        {
            let Self {
                fvk,
                a,
                rho_rauf_mag_sf,
                ..
            } = self;
            fv::fvm_laplacian(
                gpu,
                fvk,
                a,
                m,
                &rho_rauf_mag_sf.f,
                &rho_rauf_mag_sf.bf,
                p,
                1.0,
            )?;
        }

        // 5. §12.3's explicit correction, with the gradient of the CURRENT
        // p - which is the previous corrector's answer inside `solve`.
        if self.ctrl.sn_grad.applies() {
            fv::fvc_grad_scalar(gpu, &self.fvk, &mut self.grad_p, p, m)?;
            let Self {
                fvk,
                a,
                rho_rauf_mag_sf,
                grad_p,
                ..
            } = self;
            fv::fvm_laplacian_non_orth_correction(
                gpu,
                fvk,
                a,
                m,
                &rho_rauf_mag_sf.f,
                &rho_rauf_mag_sf.bf,
                p,
                grad_p,
                self.ctrl.sn_grad,
                1.0,
            )?;
        }

        // 6. `source += sum_f ±s_f`, accumulating - the same face-flux sum
        // `Simple::assemble_pressure` puts `div(phi_HbyA)` through, on `s`.
        {
            if n > 0 {
                let nl = n as Label;
                let acc: Label = 1;
                let f = self.k.face_flux_sum.clone();
                unsafe {
                    gpu.stream()
                        .launch_builder(&f)
                        .arg(&mut self.a.source)
                        .arg(&self.s.f)
                        .arg(&self.s.bf)
                        .arg(&m.cf_offset)
                        .arg(&m.cf_face)
                        .arg(&m.cf_own)
                        .arg(&m.bcf_offset)
                        .arg(&m.bcf_face)
                        .arg(&acc)
                        .arg(&nl)
                        .launch(cfg_for(n))?;
                }
            }
        }

        // 7. the boundary fold, last, as everywhere else in this crate.
        ldu_ops::add_boundary_contributions(gpu, &self.lduk, &mut self.a, m)
    }

    // ---- assembly, solve, residual ----------------------------------------

    /// The preparation, then ONE matrix (§124.4's steps 1-7). Solves
    /// nothing: `rAU` and `HbyA`'s consumer exists whether or not `p` is
    /// ever inverted, and `residual` measures against the matrix as it
    /// stands.
    pub fn assemble_only(
        &mut self,
        gpu: &Gpu,
        mom: &super::momentum::RhoMomentum<'_>,
        p: &GpuScalarField,
        rho: &GpuScalarField,
        psi: &GpuScalarField,
    ) -> Result<()> {
        let n = self.m.n_cells;
        if n == 0 {
            return Ok(());
        }
        self.prepare(gpu, mom, p, rho, psi)?;
        self.assemble_matrix(gpu, mom, p, rho, psi)
    }

    /// The same preparation, formed ONCE, then for every non-orthogonal
    /// corrector: steps 1-7, a solve into `p.f`, and `p`'s boundary values
    /// evaluated. Returns the LAST solve's performance. The linearisation
    /// point does not move between correctors (§124.4).
    pub fn solve(
        &mut self,
        gpu: &Gpu,
        mom: &super::momentum::RhoMomentum<'_>,
        p: &mut GpuScalarField,
        rho: &GpuScalarField,
        psi: &GpuScalarField,
    ) -> Result<SolverPerformance> {
        let n = self.m.n_cells;
        if n == 0 {
            return Ok(SolverPerformance::default());
        }

        self.prepare(gpu, mom, p, rho, psi)?;

        let mut perf = SolverPerformance::default();
        for _nc in 0..=self.ctrl.n_non_orth_corr {
            self.assemble_matrix(gpu, mom, p, rho, psi)?;
            {
                let Self {
                    solk,
                    a,
                    ws,
                    ctrl,
                    ..
                } = self;
                perf = solver::solve(gpu, solk, &mut p.f, a, self.m, ws, &ctrl.p_solver)?;
            }
            field_ops::correct_boundary_conditions(gpu, &self.fldk, p, self.m)?;
        }
        Ok(perf)
    }

    /// `r = source - A p`, against the matrix AS LAST ASSEMBLED - no
    /// reassembly, so a caller can perturb `p` and see only the perturbation
    /// move. `p` is not moved.
    pub fn residual(&mut self, gpu: &Gpu, p: &GpuScalarField) -> Result<()> {
        let n = self.m.n_cells;
        if n == 0 {
            return Ok(());
        }
        {
            let Self {
                solk,
                au,
                a,
                ..
            } = self;
            solver::amul(gpu, solk, au, &p.f, a, self.m)?;
        }
        let Self {
            solk,
            residual,
            au,
            a,
            ..
        } = self;
        solver::vec_sub(gpu, solk, residual, &a.source, au, n)
    }

    // ---- the corrector ----------------------------------------------------

    /// (124.10), (123.8) and (124.11), in §124.4's order: `p`'s boundary
    /// values; `lap_f = (rho_f rAU_f |Sf|) snGrad(p)_f`, plus §12.3's
    /// correction when it applies; `mdot` of (124.10); the force at the new
    /// `p` with `rho` still `rho*`; `u = HbyA + rAU force`; and
    /// `rho = rho* + psi (p - p*)` on cells and boundary. Call after
    /// [`Self::solve`]: the face fields it reuses are the preparation's.
    #[allow(clippy::too_many_arguments)]
    pub fn correct(
        &mut self,
        gpu: &Gpu,
        mom: &mut super::momentum::RhoMomentum<'_>,
        p: &mut GpuScalarField,
        u: &mut GpuVectorField,
        rho: &mut GpuScalarField,
        psi: &GpuScalarField,
        mdot: &mut GpuSurfaceScalarField,
    ) -> Result<()> {
        let m = self.m;
        let n = m.n_cells;
        if n == 0 {
            return Ok(());
        }
        let nif = m.n_internal_faces;
        let nbf = m.n_boundary_faces;

        // p on its boundary first: the face values (124.10) reads are the
        // evaluated triple, and the snGrad below is built on them.
        field_ops::correct_boundary_conditions(gpu, &self.fldk, p, m)?;

        // lap_f = (rho_f rAU_f |Sf|) snGrad(p)_f, the matrix's own face
        // coefficient through sn_grad_flux, plus §12.3's correction when
        // the scheme applies.
        fv::sn_grad_flux(
            gpu,
            &self.fvk,
            &mut self.lap,
            p,
            &self.rho_rauf_mag_sf.f,
            &self.rho_rauf_mag_sf.bf,
            m,
        )?;
        if self.ctrl.sn_grad.applies() {
            fv::fvc_grad_scalar(gpu, &self.fvk, &mut self.grad_p, p, m)?;
            fv::sn_grad_flux_correction(
                gpu,
                &self.fvk,
                &mut self.lap,
                p,
                &self.rho_rauf_mag_sf.f,
                &self.rho_rauf_mag_sf.bf,
                &self.grad_p,
                self.ctrl.sn_grad,
                m,
            )?;
        }

        // mdot of (124.10), on internal and boundary faces, with `phig`
        // the face field (123.12) `RhoMomentum::set_force` left behind.
        {
            let Self {
                k,
                s0,
                psi_phi,
                w,
                bw,
                lap,
                rho_rauf,
                ..
            } = self;
            if nif > 0 {
                let nl = nif as Label;
                let f = k.pressure_mass_flux.clone();
                unsafe {
                    gpu.stream()
                        .launch_builder(&f)
                        .arg(&mut mdot.f)
                        .arg(&s0.f)
                        .arg(&psi_phi.f)
                        .arg(w)
                        .arg(&p.f)
                        .arg(&lap.f)
                        .arg(&rho_rauf.f)
                        .arg(&mom.gravity_flux().f)
                        .arg(&m.owner)
                        .arg(&m.neighbour)
                        .arg(&nl)
                        .launch(cfg_for(nif))?;
                }
            }

            if nbf > 0 {
                let nl = nbf as Label;
                let f = k.pressure_mass_flux_boundary.clone();
                unsafe {
                    gpu.stream()
                        .launch_builder(&f)
                        .arg(&mut mdot.bf)
                        .arg(&s0.bf)
                        .arg(&psi_phi.bf)
                        .arg(bw)
                        .arg(&p.f)
                        .arg(&p.bf)
                        .arg(&lap.bf)
                        .arg(&rho_rauf.bf)
                        .arg(&mom.gravity_flux().bf)
                        .arg(&m.b_face_cells)
                        .arg(&m.b_nbr_cell)
                        .arg(&m.b_kind)
                        .arg(&nl)
                        .launch(cfg_for(nbf))?;
                }
            }
        }

        // The force at the new p, with rho still rho* - (124.11) has not run
        // yet - and then u = HbyA + rAU force with the CELL rAU, which is
        // what momCorrectVelocity does for §5.1. `p` is not relaxed here;
        // that is step 6 of §125.3's loop.
        mom.set_force(gpu, p, u, rho)?;

        {
            let nl = n as Label;
            let f = self.k.correct_velocity.clone();
            unsafe {
                gpu.stream()
                    .launch_builder(&f)
                    .arg(&mut u.f)
                    .arg(mom.hbya())
                    .arg(mom.rau())
                    .arg(mom.force())
                    .arg(&nl)
                    .launch(cfg_for(n))?;
            }
        }
        field_ops::correct_boundary_conditions_vector(gpu, &self.fldk, u, m)?;

        // (124.11): rho = rho* + psi (p - p*), at frozen psi, on the cells
        // and on the boundary. At psi = 0 it holds rho, which (124.6)
        // needs; when the caller handed in rho* = psi (p* + p_op) it is
        // the gas law (122.3) to round-off.
        {
            let nl = n as Label;
            let f = self.k.linearized_rho.clone();
            unsafe {
                gpu.stream()
                    .launch_builder(&f)
                    .arg(&mut rho.f)
                    .arg(&psi.f)
                    .arg(&p.f)
                    .arg(&self.p_star)
                    .arg(&nl)
                    .launch(cfg_for(n))?;
            }
        }

        if nbf > 0 {
            let nl = nbf as Label;
            let f = self.k.linearized_rho.clone();
            unsafe {
                gpu.stream()
                    .launch_builder(&f)
                    .arg(&mut rho.bf)
                    .arg(&psi.bf)
                    .arg(&p.bf)
                    .arg(&self.p_star_b)
                    .arg(&nl)
                    .launch(cfg_for(nbf))?;
            }
        }

        Ok(())
    }
}

// ==========================================================================
//  The capture gate - SPEC-LIT 81.7's row for this module (CMP-10)
// ==========================================================================

#[cfg(test)]
mod tests {
    use crate as ofgpu;

    include!("../../tests/shared/compressible_pressure_cases.rs");

    /// §81.7's row for `src/compressible/pressure.rs`: one corrector -
    /// `set_force`, a fixed-sweep momentum `solve`, `rhie_chow`, a
    /// fixed-sweep pressure `solve` and `correct` - captures into one graph
    /// and replays bitwise, on the precedent of the momentum gate. The seed
    /// `u`, `rho` (with its old level) and `mu_eff` are non-uniform, `mdot`
    /// is the flux of a throwaway seed, `psi` is small but NOT zero - so
    /// every one of (124.7)'s terms is live - and the linear solvers sweep
    /// a fixed 4 iterations with residuals unreported (§81.7's capture
    /// conditions), the pressure one with Jacobi: DILU's factorisation
    /// reads the coloured ordering back, and `simple.rs`'s gate uses Jacobi
    /// for the same reason.
    #[test]
    fn the_compressible_pressure_corrector_replays_bitwise() {
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

        let mut pctrl = RhoPressureControls::default();
        pctrl.p_solver.precon = Preconditioner::Diagonal;
        pctrl.p_solver.fixed_iters = true;
        pctrl.p_solver.max_iter = 4;
        pctrl.p_solver.report_residuals = false;

        // The constant inputs, outside the captured state: `psi` and
        // `mu_eff` are non-uniform, `p` is zero with a fixedValue 0 on +x,
        // and `mdot` is the flux of a throwaway non-uniform seed.
        let n = hm.n_cells;
        let nbf = hm.n_boundary_faces;
        let mut psi = GpuScalarField::zeros(&g, &gm, "psi").expect("psi");
        let psiv: Vec<Scalar> = (0..n)
            .map(|i| 1.0e-5 * (1.0 + 0.01 * (i % 3) as Scalar))
            .collect();
        g.write(&mut psi.f, &psiv).expect("psi");
        let psib: Vec<Scalar> = (0..nbf)
            .map(|i| 1.0e-5 * (1.0 + 0.01 * (i % 3) as Scalar))
            .collect();
        g.write(&mut psi.bf, &psib).expect("psi bf");
        zero_gradient_scalar(&g, &mut psi, &hm).expect("psi bc");
        let mut mu_eff = GpuScalarField::zeros(&g, &gm, "muEff").expect("muEff");
        let mv: Vec<Scalar> = (0..n)
            .map(|i| 1.8e-5 * (1.0 + 0.1 * (i % 3) as Scalar))
            .collect();
        g.write(&mut mu_eff.f, &mv).expect("mu");
        let mb: Vec<Scalar> = (0..nbf)
            .map(|i| 1.8e-5 * (1.0 + 0.1 * (i % 4) as Scalar))
            .collect();
        g.write(&mut mu_eff.bf, &mb).expect("mu bf");

        let report = crate::capture::capture_replays_bitwise(
            &g,
            "the compressible all-speed pressure corrector (SPEC-LIT 124.4)",
            || {
                let mom = RhoMomentum::new(&g, &gm, ctrl)?;
                let prs = RhoPressure::new(&g, &gm, pctrl)?;
                let mut p = GpuScalarField::zeros(&g, &gm, "p")?;
                let mut kind = vec![BcKind::ZeroGradient as Label; nbf];
                let mut fr = vec![0.0 as Scalar; nbf];
                for i in 0..nbf {
                    let (axis, sgn) = face_axis(hm.b_sf[i]);
                    if axis == 0 && sgn > 0.0 {
                        kind[i] = BcKind::FixedValue as Label;
                        fr[i] = 1.0;
                    }
                }
                g.write(&mut p.bc_kind, &kind)?;
                g.write(&mut p.fr, &fr)?;
                g.write(&mut p.ref_value, &vec![0.0 as Scalar; nbf])?;
                g.write(&mut p.ref_grad, &vec![0.0 as Scalar; nbf])?;
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
                let mut rho = GpuScalarField::zeros(&g, &gm, "rho")?;
                let rv: Vec<Scalar> = (0..n)
                    .map(|i| 1.2 + 0.01 * (i % 7) as Scalar)
                    .collect();
                g.write(&mut rho.f, &rv)?;
                g.write(&mut rho.f0, &rv)?;
                let rb: Vec<Scalar> = (0..nbf)
                    .map(|i| 1.2 + 0.01 * (i % 5) as Scalar)
                    .collect();
                g.write(&mut rho.bf, &rb)?;
                let mut mdot = GpuSurfaceScalarField::zeros(&g, &gm, "mdot")?;
                let mut seed = GpuVectorField::zeros(&g, &gm, "seed")?;
                let sv: Vec<Vec3> = (0..n)
                    .map(|i| Vec3::new(0.1 + 0.01 * (i % 5) as Scalar, 0.02, 0.0))
                    .collect();
                g.write(&mut seed.f, &sv)?;
                fv::interpolate_vector_flux(&g, &fvk, &mut mdot, &seed, &gm)?;
                Ok((mom, prs, p, u, rho, mdot))
            },
            |(mom, prs, p, u, rho, mdot): &mut (
                _,
                RhoPressure,
                GpuScalarField,
                GpuVectorField,
                GpuScalarField,
                GpuSurfaceScalarField,
            )| {
                mom.set_viscosity(&g, u, &mu_eff)?;
                mom.set_force(&g, p, u, rho)?;
                mom.solve(&g, u, &mdot, rho)?;
                mom.rhie_chow(&g, u, rho)?;
                prs.solve(&g, mom, p, rho, &psi)?;
                prs.correct(&g, mom, p, u, rho, &psi, mdot)
            },
            |(_mom, prs, p, u, rho, mdot): &(
                _,
                RhoPressure,
                GpuScalarField,
                GpuVectorField,
                GpuScalarField,
                GpuSurfaceScalarField,
            )| {
                let flat = |v: &ofgpu::DevBuf<Vec3>| -> Result<Vec<Scalar>> {
                    let t = g.download(v)?;
                    let mut out = Vec::with_capacity(t.len() * 3);
                    for c in &t {
                        out.push(c.x);
                        out.push(c.y);
                        out.push(c.z);
                    }
                    Ok(out)
                };
                Ok(vec![
                    ("p", g.download(&p.f)?),
                    ("U", flat(&u.f)?),
                    ("rho", g.download(&rho.f)?),
                    ("mdot.f", g.download(&mdot.f)?),
                    ("mdot.bf", g.download(&mdot.bf)?),
                    ("psiPhi", g.download(&prs.psi_phi().f)?),
                ])
            },
        )
        .expect(
            "SPEC-LIT 81.7: the compressible pressure corrector must capture \
             and replay bitwise",
        );
        println!("  compressible pressure: {report}");
    }
}
