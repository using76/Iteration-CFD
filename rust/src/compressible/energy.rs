// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.
// Provenance: see PROVENANCE.md. No GPL-licensed source was consulted.

//! The energy equation of the compressible solver, in the `c_p T + K` form,
//! solved for `T` - SPEC-LIT §125.1, §125.2 and §125.5 (docs/17 CMP-11).
//!
//! ```text
//! d(rho c_p T)/dt + div(mdot c_p T) - div(kappa_eff grad T)
//!       =  dp/dt  -  [ d(rho K)/dt + div(mdot K) ]  +  div(tau . u)  +  rho g . u   (125.1)
//! ```
//!
//! `c_p` is constant (§122.1, `hConst`), so the system assembled is (125.1)
//! divided by `c_p` (§125.5): the time term is `Ddt::add_rho` with weight
//! `rho`, the convection is `fvm_div_gauss` on the mass flux `mdot`, the
//! laplacian's coefficient is `kappa_f |Sf| / c_p`, and every explicit term
//! carries `rcp = fl(1/c_p)`. `K_f` is taken on the convection weights of
//! `T_f` (§125.1's DESIGN), so the face sum `mdot_f (T_f + K_f/c_p)` is a
//! face flux of `h_0/c_p` and steady, inviscid, adiabatic flow conserves
//! `h_0` to the solver's tolerance. Everything §3, §7 and §12 already
//! compute is CALLED; the three kernels of `cuda/compressible.cu` form only
//! the explicit side - the face flux of the `K` convection and the viscous
//! work, and the cell source of `dp/dt`, `d(rho K)/dt` and the gravity work.
//!
//! Written from:
//!   ofgpu `SPEC-LIT.md` §125.1-§125.2 - the equation solved for `T`, its
//!     discrete terms and what is not in it, after Carlson's eq. (36) and
//!     NASA Glenn's *Navier-Stokes Equations* page, both *read*
//!   ofgpu `SPEC-LIT.md` §125.5 - the assembly as built, which is the
//!     contract of this file
//!   ofgpu `SPEC-LIT.md` §123.2 - the stress (123.2) the viscous work
//!     contracts, and §123.3's face tensor (123.10)
//!   ofgpu `SPEC-LIT.md` §86.3 - the mass-weighted time term
//!   ofgpu `SPEC-LIT.md` §7, §12.3 - the convection weights and the explicit
//!     non-orthogonal correction
//!   Moukalled, Mangani & Darwish (2016) ch. 16; Ferziger & Perić (2002)
//!     ch. 11 - the face-consistency argument for `K_f`, *not read*, as
//!     §125.1 states of them
//! No GPL-licensed source was consulted.

use cudarc::driver::{CudaFunction, PushKernelArg};

use crate::device::{cfg_for, DevBuf, Gpu, KernelSet};
use crate::error::{Error, Result};
use crate::field::{GpuScalarField, GpuSurfaceScalarField, GpuVectorField};
use crate::field_ops::{self, FieldKernels};
use crate::fv::{self, DivScheme, FvKernels, GradScheme, SnGradScheme};
use crate::io::case::SolverControls;
use crate::ldu::GpuLduMatrix;
use crate::ldu_ops::{self, LduKernels};
use crate::mesh::GpuMesh;
use crate::solver::{
    self, LinearSolverKind, Preconditioner, SolverKernels, SolverPerformance, SolverWorkspace,
};
use crate::timescheme::{DdtCoeffs, DdtScheme, LtsControls};
use crate::types::Tensor;
use crate::{Label, Scalar, Vec3};

use super::lts::LocalStep;

/// A refusal in the one house format.
fn refuse(msg: String) -> Error {
    Error::Refused(format!("compressible::energy::new: {msg}"))
}

// ==========================================================================
//  Controls
// ==========================================================================

/// Everything the energy equation reads out of a case.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RhoEnergyControls {
    /// The linear solver of §8.1. `PCG`, `DIC` and `GAMG` are refused by
    /// name in [`RhoEnergy::new`]: the convection term makes the matrix
    /// asymmetric. The default is `PBiCGStab` with `DILU`.
    pub t_solver: SolverControls,

    /// `alpha_T` of §125.3 step 3: implicit under-relaxation, applied
    /// always (also at 1, where it only enforces dominance), as the
    /// momentum equation applies its own. The outer loop sets 1 on a final
    /// corrector through [`RhoEnergy::set_t_relax`].
    pub t_relax: Scalar,

    /// The convection scheme of `T`. Only schemes whose face value IS a
    /// weight are accepted, because `K_f` must use the weights of `T_f`
    /// (§125.1's DESIGN): `linearUpwind`, `linearUpwindBlended` and `cubic`
    /// are refused by name.
    pub div_scheme: DivScheme,

    /// The gradient scheme of `grad u` (the viscous work) and `grad T` (a
    /// limiter's sensor and §12.3's correction).
    pub grad_scheme: GradScheme,

    /// §12.3's snGrad scheme, for the laplacian's explicit correction.
    pub sn_grad: SnGradScheme,

    /// `ddtSchemes`: `steadyState`, `Euler` or `backward`. `localEuler` and
    /// `CrankNicolson` are refused by name, as §123.3 refuses them for the
    /// momentum equation; §125.4's per-cell step is attached to an `Euler`
    /// equation by `compressible::Compressible` (CMP-13, §125.7).
    pub ddt: DdtScheme,

    /// The time step of a transient run.
    pub delta_t: Scalar,

    /// `viscousWork` of §125.1: the term `div(tau . u)`, default on.
    pub viscous_work: bool,

    /// Gravity; the work `rho g . u` is on only when `g != 0`.
    pub g: Vec3,

    /// `c_p` [J/(kg K)], constant (§122.1) - `PerfectGas::cp`. The
    /// assembly divides (125.1) by it; `fl(1/c_p)` is formed in f64 and
    /// rounded once.
    pub cp: f64,
}

impl Default for RhoEnergyControls {
    fn default() -> Self {
        Self {
            t_solver: SolverControls {
                precon: Preconditioner::Dilu,
                ..Default::default()
            },
            t_relax: 1.0,
            div_scheme: DivScheme::Upwind,
            grad_scheme: GradScheme::GAUSS,
            sn_grad: SnGradScheme::Corrected,
            ddt: DdtScheme::SteadyState,
            delta_t: 1.0,
            viscous_work: true,
            g: Vec3::ZERO,
            cp: super::thermo::PerfectGas::air(1.0e5).cp(),
        }
    }
}

/// The fields the energy equation reads - each with its old levels and
/// boundary values where the term reads them. The gas law is never
/// evaluated here.
pub struct EnergyInputs<'a> {
    /// `u`, three time levels and its boundary values: `K`, `u_f`, `grad u`.
    pub u: &'a GpuVectorField,
    /// The gauge `p`, three time levels: `dp/dt`.
    pub p: &'a GpuScalarField,
    /// The face mass flux [kg/s].
    pub mdot: &'a GpuSurfaceScalarField,
    /// `rho`, three time levels.
    pub rho: &'a GpuScalarField,
    /// `mu_eff` with boundary values: the viscous work.
    pub mu_eff: &'a GpuScalarField,
    /// `kappa_eff` with boundary values: the laplacian.
    pub kappa_eff: &'a GpuScalarField,
}

// ==========================================================================
//  The mixture mode (SPEC-LIT §158.4, off by default)
// ==========================================================================

/// The energy equation's MIXTURE state: the equation then runs in ENERGY
/// units - `d(rho cp(c~) T)/dt + div(mdot cp(c~_f) T_f) - div(lambda grad T)
/// = ... + S_dust` - instead of divided by the gas's constant `c_p`. `w`,
/// `w0` and `w00` are `rho cp(c~)` at the three time levels on the cells
/// (the time term's weights), `flux` the face flux `mdot cp(c~_f)` (the
/// convection), and `source` S_dust = `Δh_c wdot_c` plus the
/// `div(rho_f Gamma_f T_f (cp_b - cp_u) snGrad c~)` enthalpy-of-mixture
/// transport, in W/m3 on the cells. Attached by
/// [`RhoEnergy::enable_mixture`]; every launcher below fills it.
pub struct EnergyMixture {
    pub w: DevBuf<Scalar>,
    pub w0: DevBuf<Scalar>,
    pub w00: DevBuf<Scalar>,
    pub flux: GpuSurfaceScalarField,
    pub source: DevBuf<Scalar>,
}

// ==========================================================================
//  Kernels
// ==========================================================================

/// The entry points, resolved once: CMP-11's three of `compressible.cu`,
/// the face product of `momentum.cu`, the face-flux sum of `simple.cu`.
struct RhoEnergyKernels {
    face_flux: CudaFunction,
    face_flux_boundary: CudaFunction,
    source: CudaFunction,
    mul: CudaFunction,
    face_flux_sum: CudaFunction,
}

impl RhoEnergyKernels {
    fn new(gpu: &Gpu) -> Result<Self> {
        let c = KernelSet::new(gpu, crate::kernels::COMPRESSIBLE)?;
        let mo = KernelSet::new(gpu, crate::kernels::MOMENTUM)?;
        let si = KernelSet::new(gpu, crate::kernels::SIMPLE)?;
        Ok(Self {
            face_flux: c.func("cmpEnergyFaceFlux")?,
            face_flux_boundary: c.func("cmpEnergyFaceFluxBoundary")?,
            source: c.func("cmpEnergySource")?,
            mul: mo.func("momMul")?,
            face_flux_sum: si.func("smpFaceFluxSum")?,
        })
    }
}

// ==========================================================================
//  The equation
// ==========================================================================

/// The energy equation (125.1) divided by `c_p`, assembled and solved for
/// `T` alone (§125.5).
///
/// Borrows the mesh for its whole life. Owns its time term ([`Self::ddt`]);
/// the outer loop advances its state once per step beside the momentum
/// equation's.
pub struct RhoEnergy<'m> {
    m: &'m GpuMesh,
    ctrl: RhoEnergyControls,
    /// `fl(1/c_p)`.
    rcp: Scalar,

    fvk: FvKernels,
    lduk: LduKernels,
    fldk: FieldKernels,
    solk: SolverKernels,
    k: RhoEnergyKernels,

    /// The time term of §86.3, `d(rho T)/dt`.
    pub ddt: crate::timescheme::Ddt,
    /// §125.7: the per-cell step of a local-time-step run, attached to an
    /// `Euler` equation by [`Self::attach_local_step`]; `None` runs on
    /// `ddt`'s own coefficients.
    local: Option<LocalStep>,

    a: GpuLduMatrix,
    ws: SolverWorkspace,

    /// §7's convection weights of `T`, shared by `K_f` (§125.1's DESIGN).
    w: DevBuf<Scalar>,
    bw: DevBuf<Scalar>,
    /// `grad T`: a limiter's sensor and §12.3's correction.
    grad_t: DevBuf<Vec3>,
    /// `grad u`, for the viscous work.
    grad_u: DevBuf<Tensor>,

    /// `kappa_f` and then `rcp kappa_f |Sf|`, the laplacian's coefficient.
    kappa_face: GpuSurfaceScalarField,
    gamma: GpuSurfaceScalarField,
    /// `mu_f`, linear, for the viscous work.
    mu_face: GpuSurfaceScalarField,

    /// `e_f = rcp (W_f - mdot_f K~_f)` (125.4), the explicit face flux.
    e: GpuSurfaceScalarField,

    au: DevBuf<Scalar>,
    residual: DevBuf<Scalar>,

    /// §158.4's mixture mode, off by default: `Some` puts the equation in
    /// energy units on the dust model's `rho cp(c~)`.
    mixture: Option<EnergyMixture>,
}

impl<'m> RhoEnergy<'m> {
    /// Build the equation and allocate everything it will ever touch: no
    /// method below allocates, downloads or syncs, so a captured region can
    /// hold [`Self::solve`].
    pub fn new(gpu: &Gpu, m: &'m GpuMesh, ctrl: RhoEnergyControls) -> Result<Self> {
        if matches!(ctrl.ddt, DdtScheme::LocalEuler) {
            return Err(refuse(
                "ddtSchemes named localEuler, and this equation is mass-weighted: \
                 d(rho T)/dt with a different dt in every cell conserves nothing \
                 alone. The steady run it serves is `compressible::Compressible`'s, \
                 which builds this equation on Euler with the local step of SPEC-LIT \
                 §125.4 attached. Alternative: `Euler` or `backward` for a transient \
                 run, `steadyState` for a steady one"
                    .to_string(),
            ));
        }
        if matches!(ctrl.ddt, DdtScheme::CrankNicolson(_)) {
            return Err(refuse(
                "ddtSchemes named CrankNicolson: the theta weighting and the \
                 implicit under-relaxation want the same slot in this assembly, \
                 so it is not reachable here. Alternative: `Euler`, `backward`, \
                 or `steadyState`"
                    .to_string(),
            ));
        }
        if ctrl.div_scheme.correction().is_some() {
            return Err(refuse(format!(
                "divSchemes for T named {:?}, a deferred-correction scheme: its \
                 face value is not a weight, and K_f must use the face weights of \
                 T_f so that the face sum is a flux of h0 = c_p T + K (SPEC-LIT \
                 §125.1). Alternative: `upwind`, `linear` or a §7 limiter",
                ctrl.div_scheme
            )));
        }
        if ctrl.t_solver.solver == LinearSolverKind::PCG {
            return Err(refuse(
                "solvers/T/solver named PCG, and the energy matrix is ASYMMETRIC: \
                 its convection term div(mdot T) has no transpose, so conjugate \
                 gradients - defined only for symmetric systems (SPEC-LIT §8.2) - \
                 are not guaranteed to converge on it. Alternative: `PBiCGStab` \
                 with `DILU`"
                    .to_string(),
            ));
        }
        if ctrl.t_solver.precon == Preconditioner::Dic {
            return Err(refuse(
                "solvers/T/preconditioner named DIC, and the energy matrix is \
                 asymmetric: DIC is the incomplete CHOLESKY factorisation \
                 (SPEC-LIT §21) and an asymmetric matrix has no Cholesky factor \
                 to approximate. Alternative: `PBiCGStab` with `DILU`"
                    .to_string(),
            ));
        }
        if ctrl.t_solver.solver == LinearSolverKind::Gamg {
            return Err(refuse(
                "solvers/T/solver named GAMG, and the energy matrix is asymmetric; \
                 the compressible equations are PBiCGStab systems (SPEC-LIT \
                 §124.4, §125.5). Alternative: `PBiCGStab` with `DILU`"
                    .to_string(),
            ));
        }
        if !(ctrl.t_relax > 0.0 && ctrl.t_relax <= 1.0) {
            return Err(refuse(format!(
                "t_relax is {}; implicit under-relaxation needs 0 < alpha <= 1 \
                 (SPEC-LIT §5.2)",
                ctrl.t_relax
            )));
        }
        if !ctrl.cp.is_finite() || !(ctrl.cp > 0.0) {
            return Err(refuse(format!(
                "cp is {}; (125.1) is divided by a positive, finite c_p \
                 (SPEC-LIT §125.5)",
                ctrl.cp
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
                "g = ({}, {}, {}) is not finite; the gravity work rho g . u would \
                 carry it into every cell",
                ctrl.g.x, ctrl.g.y, ctrl.g.z
            )));
        }

        let n = m.n_cells;
        let nif = m.n_internal_faces;
        let nbf = m.n_boundary_faces;
        let one = |k: usize| k.max(1);

        Ok(Self {
            m,
            ctrl,
            rcp: (1.0 / ctrl.cp) as Scalar,

            fvk: FvKernels::new(gpu)?,
            lduk: LduKernels::new(gpu)?,
            fldk: FieldKernels::new(gpu)?,
            solk: SolverKernels::new(gpu)?,
            k: RhoEnergyKernels::new(gpu)?,

            ddt: crate::timescheme::Ddt::new_with_relax(
                gpu,
                m,
                ctrl.ddt,
                ctrl.delta_t,
                LtsControls::default(),
                ctrl.t_relax < 1.0,
            )?,
            local: None,

            a: GpuLduMatrix::new(gpu, m)?,
            ws: SolverWorkspace::for_mesh(gpu, m)?,

            w: gpu.zeros(one(nif))?,
            bw: gpu.zeros(one(nbf))?,
            grad_t: gpu.zeros(one(n))?,
            grad_u: gpu.zeros(one(n))?,

            kappa_face: GpuSurfaceScalarField::zeros(gpu, m, "kappaF")?,
            gamma: GpuSurfaceScalarField::zeros(gpu, m, "kappaMagSfByCp")?,
            mu_face: GpuSurfaceScalarField::zeros(gpu, m, "muFEnergy")?,

            e: GpuSurfaceScalarField::zeros(gpu, m, "energyFlux")?,

            au: gpu.zeros(one(n))?,
            residual: gpu.zeros(one(n))?,

            mixture: None,
        })
    }

    // ---- accessors --------------------------------------------------------

    pub fn controls(&self) -> &RhoEnergyControls {
        &self.ctrl
    }

    /// `alpha_T` for the next assembly - 1 on a final transient corrector
    /// (§125.3 step 3).
    pub fn set_t_relax(&mut self, alpha: Scalar) -> Result<()> {
        if !(alpha > 0.0 && alpha <= 1.0) {
            return Err(Error::Refused(format!(
                "compressible::energy::set_t_relax: t_relax is {alpha}; implicit \
                 under-relaxation needs 0 < alpha <= 1 (SPEC-LIT §5.2)"
            )));
        }
        self.ctrl.t_relax = alpha;
        Ok(())
    }

    /// §125.7: attach the per-cell step of a local-time-step run, the twin
    /// of `RhoMomentum::attach_local_step`. The time term and the cell
    /// source then take `a_n = r_P`, `a_0 = -r_P` from
    /// [`LocalStep::r_delta_t`]. Refused unless the equation was built on
    /// `Euler`. Allocates: call it before a captured region.
    pub fn attach_local_step(&mut self, gpu: &Gpu) -> Result<()> {
        if self.ddt.scheme != DdtScheme::Euler {
            return Err(Error::Refused(format!(
                "compressible::energy::attach_local_step: the equation was built \
                 on {:?}; the local step of SPEC-LIT §125.4 is the Euler scheme \
                 with a per-cell dt",
                self.ddt.scheme
            )));
        }
        if self.mixture.is_some() {
            return Err(Error::Refused(
                "compressible::energy::attach_local_step: the mixture energy \
                 equation (SPEC-LIT §158.4) is attached, and it is built for Euler \
                 transient runs, not for a localEuler step"
                    .to_string(),
            ));
        }
        self.local = Some(LocalStep::new(gpu, self.m)?);
        Ok(())
    }

    /// The attached local step, if any.
    pub fn local_step(&self) -> Option<&LocalStep> {
        self.local.as_ref()
    }

    /// The attached local step, for the outer loop to fill.
    pub fn local_step_mut(&mut self) -> Option<&mut LocalStep> {
        self.local.as_mut()
    }

    /// §158.4: put the equation in ENERGY units on `rho cp(c~)` - the dust
    /// model's mixture mode. Allocates, so call it before a captured region.
    /// From then on every `rcp` use in the assembly is 1 (the equation is
    /// not divided by `c_p`), the time term weights on `rho cp(c~)`, the
    /// convection on `mdot cp(c~_f)`, and S_dust enters by `fvm_su`.
    /// Refused by name while a `localEuler` local step is attached: the
    /// mixture equation is built for Euler transient runs (§158.6).
    pub fn enable_mixture(&mut self, gpu: &Gpu) -> Result<()> {
        if self.local.is_some() {
            return Err(Error::Refused(
                "compressible::energy::enable_mixture: a localEuler local step is \
                 attached, and the mixture energy equation (SPEC-LIT §158.4) is \
                 built for Euler transient runs. Alternative: the dust model on an \
                 Euler run (§158.6)"
                    .to_string(),
            ));
        }
        if self.mixture.is_some() {
            return Ok(());
        }
        let one = |k: usize| k.max(1);
        let n = self.m.n_cells;
        self.mixture = Some(EnergyMixture {
            w: gpu.zeros(one(n))?,
            w0: gpu.zeros(one(n))?,
            w00: gpu.zeros(one(n))?,
            flux: GpuSurfaceScalarField::zeros(gpu, self.m, "dustEnergyFlux")?,
            source: gpu.zeros(one(n))?,
        });
        Ok(())
    }

    /// The mixture state, when the mode is on.
    pub fn mixture(&self) -> Option<&EnergyMixture> {
        self.mixture.as_ref()
    }

    /// The mixture state, for the coupling to fill each outer iteration.
    pub fn mixture_mut(&mut self) -> Option<&mut EnergyMixture> {
        self.mixture.as_mut()
    }

    /// The matrix the last assembly left behind.
    pub fn matrix(&self) -> &GpuLduMatrix {
        &self.a
    }

    /// `e_f = rcp (W_f - mdot_f K~_f)` of (125.4), as of the last
    /// preparation.
    pub fn explicit_flux(&self) -> &GpuSurfaceScalarField {
        &self.e
    }

    /// The convection weights of `T` (internal, boundary) the last
    /// preparation formed - the weights `K_f` shares.
    pub fn weights(&self) -> (&DevBuf<Scalar>, &DevBuf<Scalar>) {
        (&self.w, &self.bw)
    }

    /// `r = source - A T`, as of the last [`Self::residual`].
    pub fn residual_field(&self) -> &DevBuf<Scalar> {
        &self.residual
    }

    // ---- launch helper ----------------------------------------------------

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

    /// The scalar every explicit term carries: `fl(1/c_p)` in the divided
    /// (125.1) form, 1 in the mixture mode's energy units (§158.4).
    fn asm_rcp(&self) -> Scalar {
        if self.mixture.is_some() {
            1.0 as Scalar
        } else {
            self.rcp
        }
    }

    // ---- the preparation --------------------------------------------------

    /// The weights of `T` on `mdot`, the laplacian's coefficient
    /// `rcp kappa_f |Sf|`, and the explicit face flux `e_f` (with `mu_f`
    /// and `grad u` before it when the viscous work is on) - formed once per
    /// [`Self::assemble_only`] or [`Self::solve`]. `T`'s, `u`'s, `mu_eff`'s
    /// and `kappa_eff`'s boundary values must be evaluated.
    fn prepare(&mut self, gpu: &Gpu, t: &GpuScalarField, inp: &EnergyInputs<'_>) -> Result<()> {
        let m = self.m;
        let nif = m.n_internal_faces;
        let nbf = m.n_boundary_faces;
        let scheme = self.ctrl.div_scheme;

        // §7's weights of T on the mass flux, T its own sensor.
        if scheme.needs_gradient() {
            fv::fvc_grad_scalar_scheme(gpu, &self.fvk, &mut self.grad_t, t, m, self.ctrl.grad_scheme)?;
            let Self { fvk, w, bw, grad_t, .. } = self;
            fv::div_scheme_weights(gpu, fvk, Some(w), Some(bw), scheme, inp.mdot, t, Some(grad_t), m)?;
        } else {
            let Self { fvk, w, bw, .. } = self;
            fv::div_scheme_weights(gpu, fvk, Some(w), Some(bw), scheme, inp.mdot, t, None, m)?;
        }

        // rcp kappa_f |Sf|: the linear kappa_f, the product with |Sf|, then
        // the one scaling by fl(1/c_p) - by 1 in the mixture mode (§158.4).
        fv::interpolate_linear(gpu, &self.fvk, &mut self.kappa_face, inp.kappa_eff, m)?;
        {
            let rcp = self.asm_rcp();
            let Self { k, kappa_face, gamma, fldk, .. } = self;
            Self::launch_mul(gpu, &k.mul, &mut gamma.f, &kappa_face.f, &m.mag_sf, nif)?;
            Self::launch_mul(gpu, &k.mul, &mut gamma.bf, &kappa_face.bf, &m.b_mag_sf, nbf)?;
            field_ops::scale_field(gpu, fldk, &mut gamma.f, rcp, nif)?;
            field_ops::scale_field(gpu, fldk, &mut gamma.bf, rcp, nbf)?;
        }

        // The viscous work's face viscosity and cell gradient.
        let work: Label = if self.ctrl.viscous_work { 1 } else { 0 };
        if self.ctrl.viscous_work {
            fv::interpolate_linear(gpu, &self.fvk, &mut self.mu_face, inp.mu_eff, m)?;
            let Self { fvk, grad_u, .. } = self;
            fv::fvc_grad_vector_scheme(gpu, fvk, grad_u, inp.u, m, self.ctrl.grad_scheme)?;
        }

        // e_f of (125.4), internal and boundary faces.
        let rcp = self.asm_rcp();
        if nif > 0 {
            let nl = nif as Label;
            let f = self.k.face_flux.clone();
            unsafe {
                gpu.stream()
                    .launch_builder(&f)
                    .arg(&mut self.e.f)
                    .arg(&inp.mdot.f)
                    .arg(&self.w)
                    .arg(&m.weights)
                    .arg(&inp.u.f)
                    .arg(&self.grad_u)
                    .arg(&self.mu_face.f)
                    .arg(&m.sf)
                    .arg(&m.owner)
                    .arg(&m.neighbour)
                    .arg(&rcp)
                    .arg(&work)
                    .arg(&nl)
                    .launch(cfg_for(nif))?;
            }
        }
        if nbf > 0 {
            let nl = nbf as Label;
            let f = self.k.face_flux_boundary.clone();
            unsafe {
                gpu.stream()
                    .launch_builder(&f)
                    .arg(&mut self.e.bf)
                    .arg(&inp.mdot.bf)
                    .arg(&self.bw)
                    .arg(&m.b_weights)
                    .arg(&inp.u.f)
                    .arg(&inp.u.bf)
                    .arg(&self.grad_u)
                    .arg(&self.mu_face.bf)
                    .arg(&m.b_sf)
                    .arg(&m.b_face_cells)
                    .arg(&m.b_nbr_cell)
                    .arg(&m.b_kind)
                    .arg(&rcp)
                    .arg(&work)
                    .arg(&nl)
                    .launch(cfg_for(nbf))?;
            }
        }
        Ok(())
    }

    // ---- the matrix -------------------------------------------------------

    /// (125.3)'s ONE system, in its order: zero; the time term; the
    /// convection; the laplacian; §12.3's correction; `source += sum_f ±e_f`;
    /// the cell source; the relaxation; the boundary fold.
    fn assemble_matrix(&mut self, gpu: &Gpu, t: &GpuScalarField, inp: &EnergyInputs<'_>) -> Result<()> {
        let m = self.m;
        let n = m.n_cells;

        self.a.zero(gpu)?;

        // §86.3: d(rho T)/dt on T's two old levels with rho at its three -
        // on rho cp(c~) in the mixture mode (§158.4's energy units).
        {
            let Self { ddt, local, a, mixture, .. } = self;
            match local {
                Some(ls) => ls.add_ddt_rho(gpu, a, m, &inp.rho.f, &inp.rho.f0, &t.f0, 1.0)?,
                None => match mixture {
                    Some(mx) => ddt.add_rho(gpu, a, m, &mx.w, &mx.w0, &mx.w00, &t.f0, &t.f00, 1.0)?,
                    None => ddt.add_rho(gpu, a, m, &inp.rho.f, &inp.rho.f0, &inp.rho.f00, &t.f0, &t.f00, 1.0)?,
                },
            }
        }

        // div(mdot T) - on mdot cp(c~_f) in the mixture mode (§158.4); the
        // weights w, bw of T are still `prepare`'s, formed on inp.mdot.
        {
            let Self { fvk, a, w, bw, mixture, .. } = self;
            match mixture {
                Some(mx) => fv::fvm_div_gauss(gpu, fvk, a, m, &mx.flux, w, bw, t, 1.0)?,
                None => fv::fvm_div_gauss(gpu, fvk, a, m, inp.mdot, w, bw, t, 1.0)?,
            }
        }

        {
            let Self { fvk, a, gamma, .. } = self;
            fv::fvm_laplacian(gpu, fvk, a, m, &gamma.f, &gamma.bf, t, -1.0)?;
        }

        if self.ctrl.sn_grad.applies() {
            fv::fvc_grad_scalar_scheme(gpu, &self.fvk, &mut self.grad_t, t, m, self.ctrl.grad_scheme)?;
            let Self { fvk, a, gamma, grad_t, .. } = self;
            fv::fvm_laplacian_non_orth_correction(
                gpu,
                fvk,
                a,
                m,
                &gamma.f,
                &gamma.bf,
                t,
                grad_t,
                self.ctrl.sn_grad,
                -1.0,
            )?;
        }

        // source += sum_f ±e_f, accumulating, outward sign.
        if n > 0 {
            let nl = n as Label;
            let acc: Label = 1;
            let f = self.k.face_flux_sum.clone();
            unsafe {
                gpu.stream()
                    .launch_builder(&f)
                    .arg(&mut self.a.source)
                    .arg(&self.e.f)
                    .arg(&self.e.bf)
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

        // The cell source: dp/dt - d(rho K)/dt + rho g . u, on this
        // equation's own time coefficients. A steadyState run has none, and
        // without gravity nothing is launched.
        let co: DdtCoeffs = self.ddt.state.coeffs(self.ddt.scheme)?;
        let timed = !(co.a_n == 0.0 && co.a_0 == 0.0 && co.a_00 == 0.0);
        let g = self.ctrl.g;
        let gravity = !(g.x == 0.0 && g.y == 0.0 && g.z == 0.0);
        if let Some(ls) = &self.local {
            // §125.7: the same source on the per-cell step.
            ls.add_energy_source(
                gpu,
                &mut self.a.source,
                m,
                (&inp.rho.f, &inp.rho.f0),
                (&inp.p.f, &inp.p.f0),
                (&inp.u.f, &inp.u.f0),
                g,
                self.rcp,
            )?;
        } else if (timed || gravity) && n > 0 {
            let nl = n as Label;
            let rcp = self.asm_rcp();
            let f = self.k.source.clone();
            unsafe {
                gpu.stream()
                    .launch_builder(&f)
                    .arg(&mut self.a.source)
                    .arg(&m.v)
                    .arg(&inp.rho.f)
                    .arg(&inp.rho.f0)
                    .arg(&inp.rho.f00)
                    .arg(&inp.p.f)
                    .arg(&inp.p.f0)
                    .arg(&inp.p.f00)
                    .arg(&inp.u.f)
                    .arg(&inp.u.f0)
                    .arg(&inp.u.f00)
                    .arg(&co.a_n)
                    .arg(&co.a_0)
                    .arg(&co.a_00)
                    .arg(&g.x)
                    .arg(&g.y)
                    .arg(&g.z)
                    .arg(&rcp)
                    .arg(&nl)
                    .launch(cfg_for(n))?;
            }
        }

        // §158.4: S_dust = Δh_c wdot_c + div(rho_f Gamma_f T_f (cp_b - cp_u)
        // snGrad c~) in W/m3, a SOURCE of energy - the mixture mode only.
        if let Some(mx) = &self.mixture {
            fv::fvm_su(gpu, &self.fvk, &mut self.a, m, &mx.source, 1.0)?;
        }

        // Relaxation BEFORE the boundary fold, always, as the momentum
        // equation does it.
        {
            let Self { lduk, a, .. } = self;
            ldu_ops::relax(gpu, lduk, a, m, &t.f, self.ctrl.t_relax)?;
        }
        ldu_ops::add_boundary_contributions(gpu, &self.lduk, &mut self.a, m)
    }

    // ---- assembly, solve, residual ----------------------------------------

    /// The preparation, then ONE matrix. Solves nothing.
    pub fn assemble_only(&mut self, gpu: &Gpu, t: &GpuScalarField, inp: &EnergyInputs<'_>) -> Result<()> {
        if self.m.n_cells == 0 {
            return Ok(());
        }
        self.prepare(gpu, t, inp)?;
        self.assemble_matrix(gpu, t, inp)
    }

    /// The preparation and the matrix, a solve into `t.f`, and `T`'s
    /// boundary values evaluated. The performance's initial residual is
    /// the `h_0` residual §125.3 step 8 reports.
    pub fn solve(
        &mut self,
        gpu: &Gpu,
        t: &mut GpuScalarField,
        inp: &EnergyInputs<'_>,
    ) -> Result<SolverPerformance> {
        if self.m.n_cells == 0 {
            return Ok(SolverPerformance::default());
        }
        self.assemble_only(gpu, t, inp)?;
        let perf = {
            let Self { solk, a, ws, ctrl, .. } = self;
            solver::solve(gpu, solk, &mut t.f, a, self.m, ws, &ctrl.t_solver)?
        };
        field_ops::correct_boundary_conditions(gpu, &self.fldk, t, self.m)?;
        Ok(perf)
    }

    /// `r = source - A T`, against the matrix AS LAST ASSEMBLED. `T` is not
    /// moved.
    pub fn residual(&mut self, gpu: &Gpu, t: &GpuScalarField) -> Result<()> {
        let n = self.m.n_cells;
        if n == 0 {
            return Ok(());
        }
        {
            let Self { solk, au, a, .. } = self;
            solver::amul(gpu, solk, au, &t.f, a, self.m)?;
        }
        let Self { solk, residual, au, a, .. } = self;
        solver::vec_sub(gpu, solk, residual, &a.source, au, n)
    }
}

#[cfg(test)]
mod tests {
    use crate as ofgpu;

    include!("../../tests/shared/compressible_energy_cases.rs");

    /// §81.7's row for `src/compressible/energy.rs`: one energy solve - the
    /// weights, the laplacian's coefficient, the viscous work's `mu_f` and
    /// `grad u`, the explicit face flux, the matrix with every term of
    /// (125.3)-(125.4) live (Euler, gravity, a non-uniform `u`, `rho`, `p` and `T`)
    /// and a fixed-sweep solve - captures into one graph and replays
    /// bitwise. The linear solver sweeps a fixed 4 iterations with Jacobi
    /// and residuals unreported, the capture conditions the pressure gate
    /// states.
    #[test]
    fn the_compressible_energy_equation_replays_bitwise() {
        let Some(g) = gpu() else { return };
        let hm = blockgen::build_mesh(&box_spec([4, 3, 2], [0.0; 3], [1.0, 0.8, 0.6], [1.0, 2.0, 1.0]))
            .expect("mesh");
        let gm = GpuMesh::upload(&g, &hm).expect("mesh");
        let cp = PerfectGas::air(1.0e5).cp();

        let mut ctrl = RhoEnergyControls {
            ddt: DdtScheme::Euler,
            delta_t: 1.0e-3,
            g: Vec3::new(0.0, -9.81, 0.0),
            cp,
            ..Default::default()
        };
        ctrl.t_solver.precon = Preconditioner::Diagonal;
        ctrl.t_solver.fixed_iters = true;
        ctrl.t_solver.max_iter = 4;
        ctrl.t_solver.report_residuals = false;

        // The constant inputs, outside the captured state.
        let vel = |x: Vec3| Vec3::new(30.0 + 10.0 * x.x, 5.0 * x.y * x.y, -2.0 * x.z);
        let u = vector_field(&g, &gm, &hm, vel, |_| false).expect("U");
        let mut uu = u;
        let u0: Vec<Vec3> = g.download(&uu.f).expect("U").iter().map(|v| *v * 0.9).collect();
        g.write(&mut uu.f0, &u0).expect("U0");
        let u = uu;
        let rho = scalar_field(&g, &gm, &hm, "rho", |x| 1.2 + 0.05 * w64(x.x), |_| None).expect("rho");
        let p = scalar_field(&g, &gm, &hm, "p", |x| 50.0 + 20.0 * w64(x.y), |_| None).expect("p");
        let mu = scalar_field(&g, &gm, &hm, "mu", |x| 1.8e-5 * (1.0 + w64(x.z)), |_| None).expect("mu");
        let kap = scalar_field(&g, &gm, &hm, "kappa", |x| 0.026 * (1.0 + w64(x.x)), |_| None)
            .expect("kappa");
        let mdot = mass_flux(&g, &u, 1.2, &gm).expect("mdot");
        let mb = g.download(&mdot.bf).expect("mdot");

        let report = crate::capture::capture_replays_bitwise(
            &g,
            "the compressible energy equation (SPEC-LIT 125.5)",
            || {
                let en = RhoEnergy::new(&g, &gm, ctrl)?;
                let t = scalar_field(
                    &g,
                    &gm,
                    &hm,
                    "T",
                    |x| 300.0 + 20.0 * w64(x.x) - 5.0 * w64(x.y),
                    |i| (mb[i] < 0.0).then_some(310.0),
                )?;
                Ok((en, t))
            },
            |(en, t): &mut (RhoEnergy, GpuScalarField)| {
                let inp = EnergyInputs {
                    u: &u,
                    p: &p,
                    mdot: &mdot,
                    rho: &rho,
                    mu_eff: &mu,
                    kappa_eff: &kap,
                };
                en.solve(&g, t, &inp).map(|_| ())
            },
            |(en, t): &(RhoEnergy, GpuScalarField)| {
                Ok(vec![
                    ("T", g.download(&t.f)?),
                    ("T.bf", g.download(&t.bf)?),
                    ("e.f", g.download(&en.explicit_flux().f)?),
                    ("e.bf", g.download(&en.explicit_flux().bf)?),
                ])
            },
        )
        .expect(
            "SPEC-LIT 81.7: the compressible energy equation must capture and \
             replay bitwise",
        );
        println!("  compressible energy: {report}");
    }
}
