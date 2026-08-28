// meteor-cfd - Copyright (c) 2026 주식회사 메테오시뮬레이션 (Meteo Simulation Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.

//! Volume of fluid: two immiscible, incompressible phases sharing one
//! momentum equation.
//!
//! Written from:
//!   C. W. Hirt, B. D. Nichols, *J. Comput. Phys.* 39 (1981) 201-225 - the
//!     volume-of-fluid method and the phase-fraction equation
//!   O. Ubbink, PhD thesis, Imperial College London (1997) - the
//!     interface-compressed finite-volume form on an unstructured mesh
//!   H. Rusche, PhD thesis, Imperial College London (2002) - the compression
//!     velocity tied to the local flux
//!   S. T. Zalesak, *J. Comput. Phys.* 31 (1979) 335-362 - flux-corrected
//!     transport, the limiter that keeps `alpha` in `[0, 1]` exactly
//!   J. U. Brackbill, D. B. Kothe, C. Zemach, *J. Comput. Phys.* 100 (1992)
//!     335-354 - the continuum surface force
//!   C. M. Rhie, W. L. Chow, *AIAA J.* 21 (1983) 1525 - the collocated face
//!     flux this pressure equation is built on
//!   R. I. Issa, *J. Comput. Phys.* 62 (1986) 40 - PISO
//!   S. V. Patankar, *Numerical Heat Transfer and Fluid Flow* (1980) ch. 4-6
//!   J. H. Ferziger, M. Peric, *Computational Methods for Fluid Dynamics*,
//!     §7.5 - body forces on faces
//!   ofgpu `SPEC-LIT.md` §20 (all five subsections) and §22 (the tests),
//!     together with §2.4, §3, §4, §5.1 and §5.4, which the assembly here is
//!     built out of.
//! No GPL-licensed source was consulted.
//!
//! # What this module owns
//!
//! Everything between "here is a mesh, a phase fraction and two fluids" and
//! "here is `alpha`, `U`, `p_rgh` and a conservative `phi` one time step
//! later". It is a self-contained transient solver rather than an extension of
//! [`crate::simple`], because every one of its equations differs from that
//! module's:
//!
//! * the momentum equation is variable-density and in conservative form, so
//!   its `ddt` carries `rho`/`rho0` and its convection carries `rho phi`;
//! * the pressure variable is `p_rgh`, not `p` (§20.5);
//! * the body force on faces is gravity across a density jump plus surface
//!   tension across a curved interface, not the plume's `g (T_ref/T - 1)`.
//!
//! What it does *not* duplicate is the arithmetic. `fvm_ddt_euler` already
//! takes `rho`/`rho0` (`cuda/fv.cu`'s `fvDdtEulerRho`) and had no caller until
//! this one; the Rhie-Chow kernels in `cuda/momentum.cu` take the face body
//! force as an argument and neither know nor care that this module's is a
//! different force; `smpFaceFluxSum` in `cuda/simple.cu` is the pressure
//! equation's right-hand side whatever the pressure means. All three are
//! loaded here and used unchanged.
//!
//! # The order of a step, and why it is that order
//!
//! ```text
//! 1  U^{n-1} <- U ;  rho^{n-1} <- rho
//! 2  solve alpha             sub-cycled, FCT-limited        §20.1, §20.2
//! 3  rho, mu   from alpha^{n}                               §20.3
//! 4  rho_phi   from the SAME limited fluxes                 §20.3
//! 5  kappa, and the face body force                         §20.4, §20.5
//! 6  momentum predictor                                     §3, §5.1
//! 7  n_correctors x [ rAU, HbyA, phi_HbyA | p_rgh | correct ]   §5.4
//! ```
//!
//! Step 2 comes first because steps 3 and 4 are statements about the *new*
//! `alpha`, and because the momentum equation must see a `rho_phi` consistent
//! with the `rho` its `ddt` differences: with `rho` affine in `alpha` and
//! `rho_phi` the same affine function of the accumulated limited `alpha` flux,
//!
//! ```text
//! (rho - rho0) V/dt + Σ_f (±rho_phi_f) = 0
//! ```
//!
//! holds to round-off. That identity is the whole of §20.3, and without it the
//! interface manufactures velocity.
//!
//! # What "bounded exactly" is conditional on
//!
//! §20.2 asks for `alpha` in `[0, 1]` **exactly**, and the Zalesak limiter
//! delivers that - *given a discretely solenoidal flux*. The limiter bounds
//! the ANTIDIFFUSIVE correction; the low-order solution it corrects is bounded
//! by a different argument, that the upwind update is a convex combination,
//! and that argument needs `Σ_f phi_f = 0` in every cell. Hand it a flux
//! carrying a pressure solver's residual instead of zero and a full cell walks
//! past one by about `(dt/V)·Σ_f phi_f` each step, with nothing downstream able
//! to take it back.
//!
//! This is not a caveat wriggling out of the specification, it is a measured
//! and controllable property, and both halves are in the tests. Measured on
//! the dam break of §22, one thousand steps:
//!
//! ```text
//! flux                                        max(alpha) - 1
//! analytic, solenoidal to round-off              0            (Zalesak disc,
//!                                                              2000 steps)
//! dam break, p_rgh at tolerance 1e-13 relTol 0   4.0e-12
//! dam break, p_rgh at tolerance 1e-9 relTol 1e-3 4.3e-07
//! ```
//!
//! So the excursion is the pressure solve's, it scales with the pressure
//! solve's stopping criterion, and it is a knob in `fvSolution` rather than a
//! property of §20.2's machinery.
//!
//! It would be easy to make this exact for any flux by subtracting
//! `alpha_P Σ_f phi_f` from the update - the bounded-convection correction of
//! §3.1, applied explicitly. That is deliberately NOT done here: it is the
//! non-conservative form, and it would break the identity §20.3 exists to
//! guarantee, `(rho - rho0) V/dt + Σ_f (±rho_phi_f) = 0`, by exactly the same
//! residual. Between an `alpha` that is bounded to a part in ten million and a
//! mass flux inconsistent with the density it advects, §20.3 is explicit about
//! which one matters: the second makes the interface generate velocity out of
//! nothing.
//!
//! # `p_rgh`, in one paragraph
//!
//! With `p = p_rgh + rho (g·x)`,
//!
//! ```text
//! -grad(p) + rho g = -grad(p_rgh) - (g·x) grad(rho)
//! ```
//!
//! exactly - the `rho grad(g·x)` and `rho g` cancel identically because
//! `grad(g·x) = g`. Both remaining terms are the size of the physics rather
//! than the size of the hydrostatic field, which is the point of §20.5. On a
//! face the second one is `-(g·x)_f (rho_N - rho_P) Δ_f |Sf|`, built from the
//! same `Δ_f |Sf|` the pressure laplacian uses, so in hydrostatic equilibrium
//! the two cancel *face by face* and a sealed tank of stratified fluid does
//! not merely stay nearly at rest, it stays at rest.

use std::path::Path;

use cudarc::driver::{CudaFunction, PushKernelArg};

use crate::device::{cfg_for, DevBuf, Gpu, KernelSet};
use crate::error::{Error, Result};
use crate::field::{BcKind, GpuScalarField, GpuSurfaceScalarField, GpuVectorField};
use crate::field_ops::{self, FieldKernels};
use crate::fv::{self, FvKernels};
use crate::io::case::{read_solver_controls, SolverControls};
use crate::io::contract::{unreadable, unsupported};
use crate::io::dict::FoamDict;
use crate::io::schemes::FvSchemes;
use crate::ldu::GpuLduMatrix;
use crate::ldu_ops::{self, LduKernels};
use crate::mesh::{GpuMesh, HostMesh};
use crate::momentum::BuoyancyCoeffs;
use crate::solver::{self, SolverKernels, SolverPerformance, SolverWorkspace};
use crate::{Label, Scalar, Vec3};

// ==========================================================================
//  Physical properties
// ==========================================================================

/// The two fluids, gravity, and how hard the interface is compressed.
///
/// Phase 1 is the one `alpha = 1` names. Densities are absolute (kg/m^3) and
/// viscosities DYNAMIC (Pa s), not kinematic: with two fluids there is no
/// single density to divide by, and a kinematic formulation would have to pick
/// one and be wrong about the other.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VofProperties {
    pub rho1: Scalar,
    pub rho2: Scalar,
    pub mu1: Scalar,
    pub mu2: Scalar,
    /// Surface tension coefficient, N/m. Zero switches §20.4 off entirely -
    /// the curvature is still computed but multiplied by nothing.
    pub sigma: Scalar,
    /// Gravitational acceleration, `constant/g`.
    pub g: Vec3,
    /// `c_alpha` of §20.1: `0` no compression, `1` conservative compression,
    /// `> 1` enhanced. The term it scales is multiplied by
    /// `alpha_f (1 - alpha_f)` and so vanishes in both pure phases whatever
    /// this is.
    pub c_alpha: Scalar,
}

impl Default for VofProperties {
    /// Water against air at 20 C, earth gravity down `-z`, conservative
    /// compression. Every number is printed by the driver that reads them.
    ///
    /// `mu` is DYNAMIC, so air is 1.8e-5 Pa s and not the 1.5e-5 m^2/s of its
    /// kinematic viscosity - a factor of a thousand apart once multiplied by
    /// the density, and the sort of thing a default can hide for a long time.
    fn default() -> Self {
        Self {
            rho1: 998.2,
            rho2: 1.2,
            mu1: 1.002e-3,
            mu2: 1.8e-5,
            sigma: 0.0728,
            g: Vec3::new(0.0, 0.0, -9.81),
            c_alpha: 1.0,
        }
    }
}

impl VofProperties {
    fn validate(&self) -> Result<()> {
        if !(self.rho1 > 0.0) || !(self.rho2 > 0.0) {
            return Err(Error::Config(format!(
                "vof: rho1 = {} and rho2 = {}; a mixture density \
                 alpha rho1 + (1 - alpha) rho2 is only positive for every \
                 alpha in [0, 1] when both are",
                self.rho1, self.rho2
            )));
        }
        if self.mu1 < 0.0 || self.mu2 < 0.0 {
            return Err(Error::Config(format!(
                "vof: mu1 = {} and mu2 = {}; a negative viscosity makes the \
                 momentum laplacian anti-diffusive",
                self.mu1, self.mu2
            )));
        }
        if !self.c_alpha.is_finite() || self.c_alpha < 0.0 {
            return Err(Error::Config(format!(
                "vof: cAlpha = {}; SPEC-LIT 20.1 defines it on [0, inf) with \
                 0 meaning no compression and 1 conservative compression",
                self.c_alpha
            )));
        }
        if !self.sigma.is_finite() || self.sigma < 0.0 {
            return Err(Error::Config(format!(
                "vof: sigma = {}; surface tension is not negative",
                self.sigma
            )));
        }
        Ok(())
    }
}

// ==========================================================================
//  Controls
// ==========================================================================

/// Everything about *how* the step is taken, as opposed to what is in the
/// tank.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VofControls {
    /// Time step. A VOF run is transient by construction: §20.2's whole
    /// argument is about the size of the explicit `alpha` step, and a steady
    /// two-phase interface problem is not a thing this module solves.
    pub delta_t: Scalar,

    /// The Courant number one `alpha` SUB-CYCLE is allowed to reach, §20.2.
    /// The explicit update is stable and bounded up to 1; `0.5` leaves room
    /// for the compression flux to be underestimated by the bound described
    /// in `vofCourant`.
    pub max_alpha_co: Scalar,

    /// Ceiling on the sub-cycle count, so a flux that has gone to infinity
    /// fails as a diagnosable error instead of hanging the run.
    pub max_sub_cycles: Label,

    /// *DESIGN*, SPEC-LIT §20.2: how many times the Zalesak limiter is
    /// recomputed against the room its own previous pass left. Three. Zero is
    /// legal and is exactly first-order upwind - useful only to show what the
    /// machinery buys.
    pub n_limiter_iters: Label,

    /// PISO pressure correctors per step (§5.4). Two is Issa's own
    /// recommendation and the default here.
    pub n_correctors: Label,

    /// Extra pressure passes per corrector for the explicit non-orthogonal
    /// correction (§3.2). Zero is right on an orthogonal mesh.
    pub n_non_orth_correctors: Label,

    /// Solve the momentum predictor, or take `U^n` as the predictor and let
    /// the pressure correction do all the work.
    ///
    /// *DESIGN.* Both are standard. The predictor costs three linear solves
    /// and is worth it when convection dominates; with strong surface tension
    /// on a fine mesh it can be the noisiest part of the step, because the
    /// predictor moves the velocity with a force the pressure has not yet
    /// balanced. On by default.
    pub momentum_predictor: bool,

    /// Under-relaxation of the momentum matrix. `1` - no relaxation - is
    /// right for a transient run and is the default; the knob exists because
    /// a badly-started case sometimes needs it.
    pub u_relax: Scalar,

    /// `div(rho phi, U)`.
    pub div_scheme: fv::DivScheme,

    /// `gradSchemes/grad(U)` - the gradient the momentum equation's deferred
    /// correction and TVD limiter read (SPEC-LIT §11.1, §11.2, §12.1).
    ///
    /// Three separate entries rather than one, because SPEC-LIT §13.4.1(a)
    /// is exactly the rule that each equation reads the entry named for ITS
    /// OWN field. Every one of them used to be `GradScheme::GAUSS`
    /// unconditionally: `gradSchemes` was not read by this module at all.
    pub grad_u: fv::GradScheme,

    /// `gradSchemes/grad(p_rgh)` - the gradient the pressure equation's
    /// non-orthogonal correction reads (§20.5, §2.4).
    pub grad_p: fv::GradScheme,

    /// `gradSchemes/grad(alpha.<phase1>)` - the gradient the interface
    /// normal `n_hat` of §20.1/§20.4 is built from.
    pub grad_alpha: fv::GradScheme,

    /// The `snGrad` correction, §12.3. Applies to the momentum laplacian and
    /// to the pressure equation alike.
    pub sn_grad: fv::SnGradScheme,

    pub u_solver: SolverControls,
    pub p_solver: SolverControls,

    /// `controlDict/adjustTimeStep` - SPEC-LIT §20.2. `ofgpu-vof` is the one
    /// driver in this crate whose step is adaptive, so the case's own entry
    /// is honoured here rather than refused; `read_control_dict` refuses it
    /// for every driver that has no such loop.
    pub adjust_time_step: bool,

    /// `controlDict/maxCo` - the material Courant number the adaptive step
    /// holds. Zero means "the case named none", which leaves the step fixed
    /// even under `adjustTimeStep yes`.
    pub max_co: Scalar,

    /// `controlDict/maxDeltaT` - the ceiling the adaptive step may not rise
    /// past. Infinite means "the case named none".
    pub max_delta_t: Scalar,

    /// Print nothing; the driver does the printing. Kept so a caller can ask
    /// for the continuity error without paying for it in a timed run.
    pub report_continuity: bool,
}

impl Default for VofControls {
    fn default() -> Self {
        Self {
            delta_t: 1e-3,
            max_alpha_co: 0.5,
            max_sub_cycles: 100,
            n_limiter_iters: 3,
            n_correctors: 2,
            n_non_orth_correctors: 0,
            momentum_predictor: true,
            u_relax: 1.0,
            div_scheme: fv::DivScheme::Upwind,
            grad_u: fv::GradScheme::GAUSS,
            grad_p: fv::GradScheme::GAUSS,
            grad_alpha: fv::GradScheme::GAUSS,
            sn_grad: fv::SnGradScheme::Corrected,
            u_solver: SolverControls::default(),
            p_solver: SolverControls::default(),
            adjust_time_step: false,
            max_co: 0.0,
            max_delta_t: Scalar::INFINITY,
            report_continuity: true,
        }
    }
}

impl VofControls {
    fn validate(&self) -> Result<()> {
        if !(self.delta_t > 0.0) || !self.delta_t.is_finite() {
            return Err(Error::Config(format!(
                "vof: deltaT = {}; a VOF run is transient and needs a finite \
                 positive time step",
                self.delta_t
            )));
        }
        if !(self.max_alpha_co > 0.0) || self.max_alpha_co > 1.0 {
            return Err(Error::Config(format!(
                "vof: maxAlphaCo = {}; the explicit alpha update is bounded \
                 only up to a Courant number of 1 (SPEC-LIT 20.2)",
                self.max_alpha_co
            )));
        }
        if self.max_sub_cycles < 1 {
            return Err(Error::Config(format!(
                "vof: maxAlphaSubCycles = {}; at least one sub-cycle has to \
                 be taken",
                self.max_sub_cycles
            )));
        }
        if self.n_limiter_iters < 0 {
            return Err(Error::Config(format!(
                "vof: nAlphaLimiterIters = {}; negative is not a count",
                self.n_limiter_iters
            )));
        }
        if self.n_correctors < 1 {
            return Err(Error::Config(format!(
                "vof: nCorrectors = {}; PISO needs at least one pressure \
                 corrector or the flux never satisfies continuity",
                self.n_correctors
            )));
        }
        if self.n_non_orth_correctors < 0 {
            return Err(Error::Config(format!(
                "vof: nNonOrthogonalCorrectors = {} is not a count",
                self.n_non_orth_correctors
            )));
        }
        if !(self.u_relax > 0.0) || self.u_relax > 1.0 {
            return Err(Error::Config(format!(
                "vof: relaxationFactors/equations/U = {}; Patankar 4.9 needs \
                 it in (0, 1]",
                self.u_relax
            )));
        }
        Ok(())
    }
}

/// What one step cost and what it left behind.
#[derive(Debug, Clone, Copy, Default)]
pub struct VofPerformance {
    pub u: [SolverPerformance; 3],
    pub p_rgh: SolverPerformance,
    /// Sub-cycles the `alpha` equation took (§20.2).
    pub n_sub_cycles: Label,
    /// The `alpha` Courant number over the whole step, before sub-cycling.
    pub alpha_courant: Scalar,
    /// `max_c |Σ_f phi_f|` after the last corrector.
    pub continuity_error: Scalar,
}

// ==========================================================================
//  Kernels
// ==========================================================================

/// The entry points, resolved once.
///
/// Three modules' worth, because three modules' kernels are reused verbatim:
/// `vof.cu` for everything §20 adds, `momentum.cu` for the Rhie-Chow flux and
/// the component views, `simple.cu` for the pressure right-hand side and the
/// reference-level fix.
struct VofKernels {
    // vof.cu
    mixture: CudaFunction,
    face_unit_normal: CudaFunction,
    face_unit_normal_boundary: CudaFunction,
    curvature: CudaFunction,
    compression_flux: CudaFunction,
    alpha_flux: CudaFunction,
    alpha_flux_boundary: CudaFunction,
    advance: CudaFunction,
    limiter_room: CudaFunction,
    apply_limiter: CudaFunction,
    accumulate: CudaFunction,
    rho_phi: CudaFunction,
    body_force_flux: CudaFunction,
    body_force_flux_boundary: CudaFunction,
    sn_grad_mag_sf: CudaFunction,
    sn_grad_mag_sf_boundary: CudaFunction,
    courant: CudaFunction,
    phase_volume: CudaFunction,

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
    correct_flux: CudaFunction,
    correct_flux_boundary: CudaFunction,
    correct_velocity: CudaFunction,

    // simple.cu
    face_flux_sum: CudaFunction,
    pick_value: CudaFunction,
    sub_scalar: CudaFunction,
}

impl VofKernels {
    fn new(gpu: &Gpu) -> Result<Self> {
        let v = KernelSet::new(gpu, crate::kernels::VOF)?;
        let m = KernelSet::new(gpu, crate::kernels::MOMENTUM)?;
        let s = KernelSet::new(gpu, crate::kernels::SIMPLE)?;

        Ok(Self {
            mixture: v.func("vofMixture")?,
            face_unit_normal: v.func("vofFaceUnitNormal")?,
            face_unit_normal_boundary: v.func("vofFaceUnitNormalBoundary")?,
            curvature: v.func("vofCurvature")?,
            compression_flux: v.func("vofCompressionFlux")?,
            alpha_flux: v.func("vofAlphaFlux")?,
            alpha_flux_boundary: v.func("vofAlphaFluxBoundary")?,
            advance: v.func("vofAdvance")?,
            limiter_room: v.func("vofLimiterRoom")?,
            apply_limiter: v.func("vofApplyLimiter")?,
            accumulate: v.func("vofAccumulate")?,
            rho_phi: v.func("vofRhoPhi")?,
            body_force_flux: v.func("vofBodyForceFlux")?,
            body_force_flux_boundary: v.func("vofBodyForceFluxBoundary")?,
            sn_grad_mag_sf: v.func("vofSnGradMagSf")?,
            sn_grad_mag_sf_boundary: v.func("vofSnGradMagSfBoundary")?,
            courant: v.func("vofCourant")?,
            phase_volume: v.func("vofPhaseVolume")?,

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
            correct_flux: m.func("momCorrectFlux")?,
            correct_flux_boundary: m.func("momCorrectFluxBoundary")?,
            correct_velocity: m.func("momCorrectVelocity")?,

            face_flux_sum: s.func("smpFaceFluxSum")?,
            pick_value: s.func("smpPickValue")?,
            sub_scalar: s.func("smpSubScalar")?,
        })
    }
}

// ==========================================================================
//  The solver
// ==========================================================================

/// The two-phase solver: `alpha`, `U`, `p_rgh` and the flux that ties them.
///
/// Borrows the mesh for its whole life. Setup writes the fields through the
/// `_mut` accessors, calls [`Vof::initialise`] once, and then only ever calls
/// [`Vof::step`].
pub struct Vof<'m> {
    m: &'m GpuMesh,
    props: VofProperties,
    ctrl: VofControls,

    /// §20.1's `eps`: the stabilisation in `n_hat = n/(|n| + eps)`. Set from
    /// the mesh in [`Vof::new`] and reported by [`Vof::interface_eps`].
    eps_n: Scalar,

    fvk: FvKernels,
    lduk: LduKernels,
    fldk: FieldKernels,
    solk: SolverKernels,
    vk: VofKernels,

    // ---- the state -------------------------------------------------------
    alpha: GpuScalarField,
    u: GpuVectorField,
    p_rgh: GpuScalarField,
    phi: GpuSurfaceScalarField,

    // ---- mixture properties (§20.3) --------------------------------------
    /// Only `.f` and `.bf` are used; the boundary triple is never read,
    /// because `rho` is derived from `alpha` and solves for nothing.
    rho: GpuScalarField,
    rho0: DevBuf<Scalar>,
    mu: GpuScalarField,
    mu_face: GpuSurfaceScalarField,
    /// `mu_f |Sf|`, the momentum laplacian's coefficient.
    mu_mag_sf: GpuSurfaceScalarField,
    /// The mass flux, from the same limited fluxes that advanced `alpha`.
    rho_phi: GpuSurfaceScalarField,

    // ---- the alpha equation (§20.1, §20.2) -------------------------------
    grad_alpha: DevBuf<Vec3>,
    n_hatf: DevBuf<Scalar>,
    b_n_hatf: DevBuf<Scalar>,
    phir: DevBuf<Scalar>,
    phi_l: DevBuf<Scalar>,
    b_phi_l: DevBuf<Scalar>,
    anti: DevBuf<Scalar>,
    d_f: DevBuf<Scalar>,
    r_plus: DevBuf<Scalar>,
    r_minus: DevBuf<Scalar>,
    /// The limited flux of one sub-cycle, and its accumulation over the step.
    alpha_phi: GpuSurfaceScalarField,
    alpha_phi_sum: GpuSurfaceScalarField,
    /// Never written. The limiter correction has no boundary flux at all, and
    /// `vofAdvance` still needs an array to read.
    b_zero: DevBuf<Scalar>,
    co_cell: DevBuf<Scalar>,

    // ---- surface tension and gravity (§20.4, §20.5) ----------------------
    kappa: GpuScalarField,
    kappa_f: GpuSurfaceScalarField,
    sn_grad_rho: GpuSurfaceScalarField,
    sn_grad_alpha: GpuSurfaceScalarField,
    /// The face body force, `-(g·x)_f |Sf| snGrad(rho) + sigma kappa_f |Sf|
    /// snGrad(alpha)`.
    phib: GpuSurfaceScalarField,

    // ---- momentum and pressure -------------------------------------------
    a: GpuLduMatrix,
    a_p: GpuLduMatrix,
    ws: SolverWorkspace,
    uc: GpuScalarField,
    u_mag: GpuScalarField,
    grad_u_mag: DevBuf<Vec3>,
    grad_uc: DevBuf<Vec3>,
    grad_p: DevBuf<Vec3>,
    su: DevBuf<Vec3>,
    force: DevBuf<Vec3>,
    hbya: DevBuf<Vec3>,
    rau: DevBuf<Scalar>,
    au: DevBuf<Scalar>,
    w: DevBuf<Scalar>,
    bw: DevBuf<Scalar>,
    rauf: GpuSurfaceScalarField,
    rauf_mag_sf: GpuSurfaceScalarField,
    sn_grad_p: GpuSurfaceScalarField,
    phi_hbya: GpuSurfaceScalarField,
    force_flux: GpuSurfaceScalarField,
    div_phi: DevBuf<Scalar>,
    red: DevBuf<Scalar>,

    pinned: bool,
    reference_cell: usize,
}

impl<'m> Vof<'m> {
    pub fn new(
        gpu: &Gpu,
        hm: &HostMesh,
        m: &'m GpuMesh,
        props: VofProperties,
        ctrl: VofControls,
    ) -> Result<Self> {
        props.validate()?;
        ctrl.validate()?;

        if hm.n_cells != m.n_cells || hm.n_boundary_faces != m.n_boundary_faces {
            return Err(Error::Config(format!(
                "Vof::new: the host mesh has ({}, {}) cells/boundary faces \
                 and the device mesh ({}, {})",
                hm.n_cells, hm.n_boundary_faces, m.n_cells, m.n_boundary_faces
            )));
        }

        let n = m.n_cells;
        let nif = m.n_internal_faces;
        let nbf = m.n_boundary_faces;
        let one = |k: usize| k.max(1);

        // SPEC-LIT §20.1 asks for `eps` to be "a small fraction of 1/(mean
        // cell size)" and for its value to be stated. L is the cube root of
        // the mean cell volume, so 1e-8/L is eight orders below the
        // |grad alpha| ~ 1/L an interface carries and eight orders above the
        // ~1e-16/L round-off a pure phase carries.
        let eps_n = if n > 0 && m.total_volume > 0.0 {
            let l = (f64::from(m.total_volume) / n as f64).cbrt();
            (1e-8 / l) as Scalar
        } else {
            1e-8
        };

        Ok(Self {
            m,
            props,
            ctrl,
            eps_n,

            fvk: FvKernels::new(gpu)?,
            lduk: LduKernels::new(gpu)?,
            fldk: FieldKernels::new(gpu)?,
            solk: SolverKernels::new(gpu)?,
            vk: VofKernels::new(gpu)?,

            alpha: GpuScalarField::zeros(gpu, m, "alpha")?,
            u: GpuVectorField::zeros(gpu, m, "U")?,
            p_rgh: GpuScalarField::zeros(gpu, m, "p_rgh")?,
            phi: GpuSurfaceScalarField::zeros(gpu, m, "phi")?,

            rho: GpuScalarField::zeros(gpu, m, "rho")?,
            rho0: gpu.zeros(one(n))?,
            mu: GpuScalarField::zeros(gpu, m, "mu")?,
            mu_face: GpuSurfaceScalarField::zeros(gpu, m, "muf")?,
            mu_mag_sf: GpuSurfaceScalarField::zeros(gpu, m, "muMagSf")?,
            rho_phi: GpuSurfaceScalarField::zeros(gpu, m, "rhoPhi")?,

            grad_alpha: gpu.zeros(one(n))?,
            n_hatf: gpu.zeros(one(nif))?,
            b_n_hatf: gpu.zeros(one(nbf))?,
            phir: gpu.zeros(one(nif))?,
            phi_l: gpu.zeros(one(nif))?,
            b_phi_l: gpu.zeros(one(nbf))?,
            anti: gpu.zeros(one(nif))?,
            d_f: gpu.zeros(one(nif))?,
            r_plus: gpu.zeros(one(n))?,
            r_minus: gpu.zeros(one(n))?,
            alpha_phi: GpuSurfaceScalarField::zeros(gpu, m, "alphaPhi")?,
            alpha_phi_sum: GpuSurfaceScalarField::zeros(gpu, m, "alphaPhiSum")?,
            b_zero: gpu.zeros(one(nbf))?,
            co_cell: gpu.zeros(one(n))?,

            kappa: GpuScalarField::zeros(gpu, m, "kappa")?,
            kappa_f: GpuSurfaceScalarField::zeros(gpu, m, "kappaf")?,
            sn_grad_rho: GpuSurfaceScalarField::zeros(gpu, m, "snGradRho")?,
            sn_grad_alpha: GpuSurfaceScalarField::zeros(gpu, m, "snGradAlpha")?,
            phib: GpuSurfaceScalarField::zeros(gpu, m, "phib")?,

            a: GpuLduMatrix::new(gpu, m)?,
            a_p: GpuLduMatrix::new(gpu, m)?,
            ws: SolverWorkspace::for_mesh(gpu, m)?,
            uc: GpuScalarField::zeros(gpu, m, "Ucmpt")?,
            u_mag: GpuScalarField::zeros(gpu, m, "magU")?,
            grad_u_mag: gpu.zeros(one(n))?,
            grad_uc: gpu.zeros(one(n))?,
            grad_p: gpu.zeros(one(n))?,
            su: gpu.zeros(one(n))?,
            force: gpu.zeros(one(n))?,
            hbya: gpu.zeros(one(n))?,
            rau: gpu.zeros(one(n))?,
            au: gpu.zeros(one(n))?,
            w: gpu.zeros(one(nif))?,
            bw: gpu.zeros(one(nbf))?,
            rauf: GpuSurfaceScalarField::zeros(gpu, m, "rAUf")?,
            rauf_mag_sf: GpuSurfaceScalarField::zeros(gpu, m, "rAUfMagSf")?,
            sn_grad_p: GpuSurfaceScalarField::zeros(gpu, m, "snGradPrgh")?,
            phi_hbya: GpuSurfaceScalarField::zeros(gpu, m, "phiHbyA")?,
            force_flux: GpuSurfaceScalarField::zeros(gpu, m, "forceFlux")?,
            div_phi: gpu.zeros(one(n))?,
            red: gpu.zeros(1)?,

            pinned: false,
            reference_cell: 0,
        })
    }

    // ---- accessors --------------------------------------------------------

    pub fn alpha(&self) -> &GpuScalarField {
        &self.alpha
    }
    pub fn alpha_mut(&mut self) -> &mut GpuScalarField {
        &mut self.alpha
    }
    pub fn u(&self) -> &GpuVectorField {
        &self.u
    }
    pub fn u_mut(&mut self) -> &mut GpuVectorField {
        &mut self.u
    }
    pub fn p_rgh(&self) -> &GpuScalarField {
        &self.p_rgh
    }
    pub fn p_rgh_mut(&mut self) -> &mut GpuScalarField {
        &mut self.p_rgh
    }
    pub fn phi(&self) -> &GpuSurfaceScalarField {
        &self.phi
    }
    pub fn phi_mut(&mut self) -> &mut GpuSurfaceScalarField {
        &mut self.phi
    }
    /// The mixture density. `.f` and `.bf` only.
    pub fn rho(&self) -> &GpuScalarField {
        &self.rho
    }
    /// The advective mass flux `alpha_phi` this step's (or the last step's)
    /// `solve_alpha` produced - the Zalesak/limited flux, NOT the plain
    /// upwind [`Vof::initialise`] seeds it with on a cold start.
    ///
    /// Exposed `_mut` for exactly one reason: a restart. `initialise` cannot
    /// tell a resumed run from a cold one, so calling it again mid-run
    /// replaces this field's limited flux with the cold-start upwind
    /// approximation - a restart driver that only restores `alpha`/`U`/
    /// `p_rgh`/`phi` and then calls `initialise` gets a plausible but not
    /// bit-identical continuation, because this field's own history did not
    /// come along. Restoring it directly from a checkpoint after
    /// `initialise` runs is what makes a restart reproduce the continuous
    /// run's next pressure residual to round-off.
    pub fn alpha_phi_mut(&mut self) -> &mut GpuSurfaceScalarField {
        &mut self.alpha_phi_sum
    }
    /// The mass flux `rho_phi = alpha_phi`-mixed. See [`Vof::alpha_phi_mut`]
    /// for why a restart needs to set this directly.
    pub fn rho_phi_mut(&mut self) -> &mut GpuSurfaceScalarField {
        &mut self.rho_phi
    }
    /// `rho0`, the density level [`Vof::step`]'s momentum ddt differences
    /// against. See [`Vof::alpha_phi_mut`] for why a restart needs to set
    /// this directly rather than let `initialise` recompute it from the
    /// just-restored `alpha` (which gives `rho0 == rho`, a zero density ddt
    /// for the FIRST step after the restart - wrong whenever `alpha` was
    /// still changing at the moment the checkpoint was written).
    pub fn rho_old_mut(&mut self) -> &mut DevBuf<Scalar> {
        &mut self.rho0
    }
    /// The mixture dynamic viscosity. `.f` and `.bf` only.
    pub fn mu(&self) -> &GpuScalarField {
        &self.mu
    }
    /// `kappa = -div(n_hat)`, as of the last step (§20.4).
    pub fn curvature(&self) -> &GpuScalarField {
        &self.kappa
    }
    /// The face body force flux, gravity plus surface tension (§20.4, §20.5).
    pub fn body_force_flux(&self) -> &GpuSurfaceScalarField {
        &self.phib
    }
    pub fn properties(&self) -> &VofProperties {
        &self.props
    }
    pub fn controls(&self) -> &VofControls {
        &self.ctrl
    }
    /// §20.1's `eps`, stated as the section asks.
    pub fn interface_eps(&self) -> Scalar {
        self.eps_n
    }
    /// True when `p_rgh` has no `fixedValue` anywhere, so its level is ours.
    pub fn pressure_is_pinned(&self) -> bool {
        self.pinned
    }

    // ---- setup ------------------------------------------------------------

    /// Evaluate the boundary faces, build the mixture properties from the
    /// initial `alpha`, seed the old-time levels and decide whether `p_rgh`
    /// needs a reference level.
    ///
    /// Call once, after the fields have been written through the `_mut`
    /// accessors, and before the first [`Vof::step`].
    pub fn initialise(&mut self, gpu: &Gpu) -> Result<()> {
        let m = self.m;

        field_ops::correct_boundary_conditions(gpu, &self.fldk, &mut self.alpha, m)?;
        field_ops::correct_boundary_conditions_vector(gpu, &self.fldk, &mut self.u, m)?;
        field_ops::correct_boundary_conditions(gpu, &self.fldk, &mut self.p_rgh, m)?;

        field_ops::store_old_time_vector(gpu, &self.fldk, &mut self.u)?;
        field_ops::store_old_time(gpu, &self.fldk, &mut self.p_rgh)?;
        field_ops::store_old_time(gpu, &self.fldk, &mut self.alpha)?;

        self.update_properties(gpu)?;
        field_ops::copy_field(gpu, &self.fldk, &mut self.rho0, &self.rho.f, m.n_cells)?;

        // Nothing has been advected yet, so the only mass flux is whatever the
        // initial phi carries through the phase already there. Upwind is the
        // right answer for a flux that has not been limited by anything.
        self.seed_alpha_flux(gpu)?;
        self.update_rho_phi(gpu)?;

        self.pinned = !self.pressure_has_a_dirichlet(gpu)?;
        self.reference_cell = 0;

        Ok(())
    }

    /// `phi = interpolate(U) · Sf`, from whatever velocity is currently in
    /// the fields.
    ///
    /// NOT a conservative flux - nothing constrains an interpolated cell
    /// velocity to satisfy discrete continuity - and it does not need to be:
    /// the first pressure correction makes it one, and this only decides how
    /// much work that correction has to do. From a velocity of zero it is
    /// exactly zero, which is what a released column of liquid starts from.
    ///
    /// Call before [`Vof::initialise`], which seeds the mass flux from it.
    pub fn initialise_flux_from_velocity(&mut self, gpu: &Gpu) -> Result<()> {
        // The face values first: this reads `U.bf`, and a field that has been
        // written through `u_mut` but never evaluated carries whatever its
        // file's `value` entry held - which for a `zeroGradient` patch is
        // nothing at all.
        field_ops::correct_boundary_conditions_vector(gpu, &self.fldk, &mut self.u, self.m)?;

        let Self { fvk, phi, u, m, .. } = self;
        fv::interpolate_vector_flux(gpu, fvk, phi, u, m)
    }

    /// Does any boundary face give `p_rgh` a value rather than a gradient?
    ///
    /// The same test [`crate::simple`] makes, for the same reason: `empty`
    /// and `cyclic` faces cannot fix a level, and everywhere else it is `fr`
    /// that puts a value into the diagonal (§4).
    fn pressure_has_a_dirichlet(&self, gpu: &Gpu) -> Result<bool> {
        let nbf = self.m.n_boundary_faces;
        if nbf == 0 {
            return Ok(false);
        }

        let fr = gpu.download(&self.p_rgh.fr)?;
        let kinds = gpu.download(&self.p_rgh.bc_kind)?;

        for i in 0..nbf {
            let k = kinds[i];
            if k == BcKind::Empty as Label || k == BcKind::Cyclic as Label {
                continue;
            }
            if fr[i] > 0.0 {
                return Ok(true);
            }
        }
        Ok(false)
    }

    // ---- §20.3  mixture properties ---------------------------------------

    /// `rho` and `mu` on cells and on boundary faces, from the current
    /// `alpha`. Also the face viscosity the momentum laplacian needs.
    pub fn update_properties(&mut self, gpu: &Gpu) -> Result<()> {
        let m = self.m;
        let (n, nif, nbf) = (m.n_cells, m.n_internal_faces, m.n_boundary_faces);
        let (r1, r2, m1, m2) = (self.props.rho1, self.props.rho2, self.props.mu1, self.props.mu2);

        {
            let Self { vk, rho, mu, alpha, .. } = self;
            launch_mixture(gpu, &vk.mixture, &mut rho.f, &mut mu.f, &alpha.f, r1, r2, m1, m2, n)?;
            launch_mixture(
                gpu, &vk.mixture, &mut rho.bf, &mut mu.bf, &alpha.bf, r1, r2, m1, m2, nbf,
            )?;
        }

        {
            let Self { fvk, mu_face, mu, .. } = self;
            fv::interpolate_linear(gpu, fvk, mu_face, mu, m)?;
        }
        {
            let Self { vk, mu_mag_sf, mu_face, .. } = self;
            launch_mul(gpu, &vk.mul, &mut mu_mag_sf.f, &mu_face.f, &m.mag_sf, nif)?;
            launch_mul(gpu, &vk.mul, &mut mu_mag_sf.bf, &mu_face.bf, &m.b_mag_sf, nbf)?;
        }

        Ok(())
    }

    /// `rho_phi = rho2 phi + (rho1 - rho2) alpha_phi` (§20.3).
    pub fn update_rho_phi(&mut self, gpu: &Gpu) -> Result<()> {
        let m = self.m;
        let (r1, r2) = (self.props.rho1, self.props.rho2);

        let Self { vk, rho_phi, phi, alpha_phi_sum, .. } = self;
        launch_rho_phi(
            gpu,
            &vk.rho_phi,
            &mut rho_phi.f,
            &phi.f,
            &alpha_phi_sum.f,
            r1,
            r2,
            m.n_internal_faces,
        )?;
        launch_rho_phi(
            gpu,
            &vk.rho_phi,
            &mut rho_phi.bf,
            &phi.bf,
            &alpha_phi_sum.bf,
            r1,
            r2,
            m.n_boundary_faces,
        )
    }

    /// The upwind `alpha` flux of the *current* state, with no compression and
    /// no limiting. Used once, at [`Vof::initialise`], to give `rho_phi`
    /// something consistent with the initial `phi`.
    fn seed_alpha_flux(&mut self, gpu: &Gpu) -> Result<()> {
        let m = self.m;
        let (nif, nbf) = (m.n_internal_faces, m.n_boundary_faces);

        gpu.fill_zero(&mut self.phir)?;
        self.build_alpha_fluxes(gpu)?;

        field_ops::copy_field(gpu, &self.fldk, &mut self.alpha_phi_sum.f, &self.phi_l, nif)?;
        field_ops::copy_field(gpu, &self.fldk, &mut self.alpha_phi_sum.bf, &self.b_phi_l, nbf)
    }

    // ---- §20.1, §20.2  the phase fraction equation ------------------------

    /// The face interface normal `n_hat_f · Sf`, on internal and boundary
    /// faces (§20.1, §20.4). Reads the current `alpha` and its gradient.
    fn update_face_normal(&mut self, gpu: &Gpu) -> Result<()> {
        let m = self.m;
        let eps = self.eps_n;

        {
            let Self { fvk, grad_alpha, alpha, ctrl, .. } = self;
            fv::fvc_grad_scalar_scheme(gpu, fvk, grad_alpha, alpha, m, ctrl.grad_alpha)?;
        }

        let nif = m.n_internal_faces;
        if nif > 0 {
            let nl = nif as Label;
            let f = self.vk.face_unit_normal.clone();
            unsafe {
                gpu.stream()
                    .launch_builder(&f)
                    .arg(&mut self.n_hatf)
                    .arg(&self.grad_alpha)
                    .arg(&m.weights)
                    .arg(&m.sf)
                    .arg(&m.owner)
                    .arg(&m.neighbour)
                    .arg(&eps)
                    .arg(&nl)
                    .launch(cfg_for(nif))?;
            }
        }

        let nbf = m.n_boundary_faces;
        if nbf > 0 {
            let nl = nbf as Label;
            let f = self.vk.face_unit_normal_boundary.clone();
            unsafe {
                gpu.stream()
                    .launch_builder(&f)
                    .arg(&mut self.b_n_hatf)
                    .arg(&self.grad_alpha)
                    .arg(&m.b_weights)
                    .arg(&m.b_sf)
                    .arg(&m.b_face_cells)
                    .arg(&m.b_nbr_cell)
                    .arg(&m.b_kind)
                    .arg(&eps)
                    .arg(&nl)
                    .launch(cfg_for(nbf))?;
            }
        }

        Ok(())
    }

    /// `phi_r = c_alpha |phi_f/|Sf|| (n_f · Sf)` (§20.1).
    fn update_compression_flux(&mut self, gpu: &Gpu) -> Result<()> {
        let nif = self.m.n_internal_faces;
        if nif == 0 {
            return Ok(());
        }
        let nl = nif as Label;
        let ca = self.props.c_alpha;

        let f = self.vk.compression_flux.clone();
        unsafe {
            gpu.stream()
                .launch_builder(&f)
                .arg(&mut self.phir)
                .arg(&self.phi.f)
                .arg(&self.n_hatf)
                .arg(&self.m.mag_sf)
                .arg(&ca)
                .arg(&nl)
                .launch(cfg_for(nif))?;
        }
        Ok(())
    }

    /// Zalesak steps 1 to 3: `phi_L`, and `A = phi_H - phi_L`.
    fn build_alpha_fluxes(&mut self, gpu: &Gpu) -> Result<()> {
        let m = self.m;

        let nif = m.n_internal_faces;
        if nif > 0 {
            let nl = nif as Label;
            let f = self.vk.alpha_flux.clone();
            unsafe {
                gpu.stream()
                    .launch_builder(&f)
                    .arg(&mut self.phi_l)
                    .arg(&mut self.anti)
                    .arg(&self.alpha.f)
                    .arg(&self.phi.f)
                    .arg(&self.phir)
                    .arg(&m.weights)
                    .arg(&m.owner)
                    .arg(&m.neighbour)
                    .arg(&nl)
                    .launch(cfg_for(nif))?;
            }
        }

        let nbf = m.n_boundary_faces;
        if nbf > 0 {
            let nl = nbf as Label;
            let f = self.vk.alpha_flux_boundary.clone();
            unsafe {
                gpu.stream()
                    .launch_builder(&f)
                    .arg(&mut self.b_phi_l)
                    .arg(&self.alpha.f)
                    .arg(&self.alpha.bf)
                    .arg(&self.phi.bf)
                    .arg(&m.b_face_cells)
                    .arg(&m.b_nbr_cell)
                    .arg(&m.b_kind)
                    .arg(&nl)
                    .launch(cfg_for(nbf))?;
            }
        }

        Ok(())
    }

    /// `alpha -= (dtau/V) Σ_f (±F_f)`.
    fn advance_alpha(
        &mut self,
        gpu: &Gpu,
        internal: usize,
        dtau: Scalar,
    ) -> Result<()> {
        let m = self.m;
        let n = m.n_cells;
        if n == 0 {
            return Ok(());
        }
        let nl = n as Label;

        // 0 = the low-order flux (with its boundary half), 1 = a limiter
        // correction (which has no boundary half at all).
        let f = self.vk.advance.clone();
        let (fi, fb): (&DevBuf<Scalar>, &DevBuf<Scalar>) = if internal == 0 {
            (&self.phi_l, &self.b_phi_l)
        } else {
            (&self.d_f, &self.b_zero)
        };

        unsafe {
            gpu.stream()
                .launch_builder(&f)
                .arg(&mut self.alpha.f)
                .arg(fi)
                .arg(fb)
                .arg(&m.v)
                .arg(&m.b_kind)
                .arg(&m.cf_offset)
                .arg(&m.cf_face)
                .arg(&m.cf_own)
                .arg(&m.bcf_offset)
                .arg(&m.bcf_face)
                .arg(&dtau)
                .arg(&nl)
                .launch(cfg_for(n))?;
        }
        Ok(())
    }

    /// The `alpha` Courant number over `dt`, and the sub-cycle count it
    /// implies (§20.2).
    ///
    /// The one host transfer the step makes for its own sake: eight bytes, and
    /// unavoidable, because `n = ceil(Co/Co_max)` is a *count of kernel
    /// launches* and the host is what issues them.
    fn alpha_sub_cycles(&mut self, gpu: &Gpu, dt: Scalar) -> Result<(Label, Scalar)> {
        let m = self.m;
        let n = m.n_cells;
        if n == 0 {
            return Ok((1, 0.0));
        }
        let nl = n as Label;

        {
            let f = self.vk.courant.clone();
            unsafe {
                gpu.stream()
                    .launch_builder(&f)
                    .arg(&mut self.co_cell)
                    .arg(&self.phi.f)
                    .arg(&self.phi.bf)
                    .arg(&m.v)
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

        {
            let Self { solk, ws, co_cell, .. } = self;
            solver::device_max_mag(gpu, solk, &mut ws.den, co_cell, &mut ws.partials, n)?;
        }
        let co_rate = gpu.download(&self.ws.den)?[0];

        // The compression flux is bounded by c_alpha times the advective one
        // (|n_f·Sf| <= |Sf|), so this never underestimates the total.
        let co = co_rate * dt * (1.0 + self.props.c_alpha);

        if !co.is_finite() {
            return Err(Error::Config(format!(
                "vof: the alpha Courant number came out {co}; the flux has \
                 gone non-finite, which means the previous pressure solve \
                 diverged"
            )));
        }

        let want = (co / self.ctrl.max_alpha_co).ceil();
        let n_sub = if want < 1.0 { 1 } else { want as i64 };

        if n_sub > i64::from(self.ctrl.max_sub_cycles) {
            return Err(Error::Config(format!(
                "vof: the alpha equation would need {n_sub} sub-cycles at \
                 Courant number {co:.3} to stay under maxAlphaCo = {}, and \
                 maxAlphaSubCycles is {}. Reduce deltaT.",
                self.ctrl.max_alpha_co, self.ctrl.max_sub_cycles
            )));
        }

        Ok((n_sub as Label, co))
    }

    /// Advance `alpha` over `dt`, sub-cycled and FCT-limited, accumulating the
    /// limited flux the mass flux is then built from (§20.1 to §20.3).
    ///
    /// Returns the sub-cycle count and the Courant number that chose it.
    pub fn solve_alpha(&mut self, gpu: &Gpu, dt: Scalar) -> Result<(Label, Scalar)> {
        let m = self.m;
        let (n, nif, nbf) = (m.n_cells, m.n_internal_faces, m.n_boundary_faces);
        if n == 0 {
            return Ok((1, 0.0));
        }

        let (n_sub, co) = self.alpha_sub_cycles(gpu, dt)?;
        let dtau = dt / n_sub as Scalar;
        let weight = 1.0 / n_sub as Scalar;

        gpu.fill_zero(&mut self.alpha_phi_sum.f)?;
        gpu.fill_zero(&mut self.alpha_phi_sum.bf)?;

        for _ in 0..n_sub {
            field_ops::correct_boundary_conditions(gpu, &self.fldk, &mut self.alpha, m)?;

            // §20.1: the compression direction, from the interface normal.
            self.update_face_normal(gpu)?;
            self.update_compression_flux(gpu)?;

            // §20.2 steps 1-3.
            self.build_alpha_fluxes(gpu)?;

            // The flux that will be applied, before any correction: the
            // low-order one.
            field_ops::copy_field(gpu, &self.fldk, &mut self.alpha_phi.f, &self.phi_l, nif)?;
            field_ops::copy_field(gpu, &self.fldk, &mut self.alpha_phi.bf, &self.b_phi_l, nbf)?;

            // §20.2 step 4: the bounded low-order solution.
            self.advance_alpha(gpu, 0, dtau)?;

            // §20.2 steps 5-7, iterated. Each pass measures the room the
            // previous one left and adds as much of what remains as fits.
            for _ in 0..self.ctrl.n_limiter_iters {
                self.limiter_room(gpu, dtau)?;
                self.apply_limiter(gpu)?;
                self.advance_alpha(gpu, 1, dtau)?;
            }

            // §20.2: accumulate, so the momentum equation sees one consistent
            // flux for the whole step.
            {
                let Self { vk, alpha_phi_sum, alpha_phi, .. } = self;
                launch_accumulate(
                    gpu,
                    &vk.accumulate,
                    &mut alpha_phi_sum.f,
                    &alpha_phi.f,
                    weight,
                    nif,
                )?;
                launch_accumulate(
                    gpu,
                    &vk.accumulate,
                    &mut alpha_phi_sum.bf,
                    &alpha_phi.bf,
                    weight,
                    nbf,
                )?;
            }
        }

        field_ops::correct_boundary_conditions(gpu, &self.fldk, &mut self.alpha, m)?;

        Ok((n_sub, co))
    }

    fn limiter_room(&mut self, gpu: &Gpu, dtau: Scalar) -> Result<()> {
        let m = self.m;
        let n = m.n_cells;
        if n == 0 {
            return Ok(());
        }
        let nl = n as Label;
        let (lo, hi) = (0.0 as Scalar, 1.0 as Scalar);

        let f = self.vk.limiter_room.clone();
        unsafe {
            gpu.stream()
                .launch_builder(&f)
                .arg(&mut self.r_plus)
                .arg(&mut self.r_minus)
                .arg(&self.alpha.f)
                .arg(&self.anti)
                .arg(&m.v)
                .arg(&m.cf_offset)
                .arg(&m.cf_face)
                .arg(&m.cf_own)
                .arg(&lo)
                .arg(&hi)
                .arg(&dtau)
                .arg(&nl)
                .launch(cfg_for(n))?;
        }
        Ok(())
    }

    fn apply_limiter(&mut self, gpu: &Gpu) -> Result<()> {
        let m = self.m;
        let nif = m.n_internal_faces;
        if nif == 0 {
            return Ok(());
        }
        let nl = nif as Label;

        let f = self.vk.apply_limiter.clone();
        unsafe {
            gpu.stream()
                .launch_builder(&f)
                .arg(&mut self.d_f)
                .arg(&mut self.anti)
                .arg(&mut self.alpha_phi.f)
                .arg(&self.r_plus)
                .arg(&self.r_minus)
                .arg(&m.owner)
                .arg(&m.neighbour)
                .arg(&nl)
                .launch(cfg_for(nif))?;
        }
        Ok(())
    }

    // ---- §20.4, §20.5  curvature and the face body force ------------------

    /// `|Sf| snGrad(rho)`, `|Sf| snGrad(alpha)`, `kappa`, `kappa_f` and the
    /// face body force built out of them.
    ///
    /// Call after `alpha`, `rho` and the face normal are current.
    pub fn update_body_force(&mut self, gpu: &Gpu) -> Result<()> {
        let m = self.m;
        let (n, nif, nbf) = (m.n_cells, m.n_internal_faces, m.n_boundary_faces);

        // The face normal is a function of alpha alone and the alpha solve
        // left it at the last sub-cycle's value, which IS the new alpha.
        // Recomputed anyway: initialise() calls this without an alpha solve in
        // front of it, and a stale normal is a wrong curvature.
        self.update_face_normal(gpu)?;

        // kappa = -div(n_hat), from the FACE normals (§20.4).
        if n > 0 {
            let nl = n as Label;
            let f = self.vk.curvature.clone();
            unsafe {
                gpu.stream()
                    .launch_builder(&f)
                    .arg(&mut self.kappa.f)
                    .arg(&self.n_hatf)
                    .arg(&self.b_n_hatf)
                    .arg(&m.v)
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

        // kappa on faces. A boundary face takes the adjacent cell's value,
        // which is what momFaceInterpBoundary gives for everything but a
        // cyclic couple - and there it interpolates properly.
        {
            let Self { vk, kappa_f, kappa, .. } = self;
            launch_face_interp(gpu, &vk.face_interp, &mut kappa_f.f, &kappa.f, m, nif)?;
            launch_face_interp_boundary(
                gpu,
                &vk.face_interp_boundary,
                &mut kappa_f.bf,
                &kappa.f,
                m,
                nbf,
            )?;
        }

        // The two face differences the body force is made of.
        {
            let Self { vk, sn_grad_rho, rho, .. } = self;
            launch_sn_grad(gpu, &vk.sn_grad_mag_sf, &mut sn_grad_rho.f, &rho.f, m, nif)?;
            launch_sn_grad_boundary(
                gpu,
                &vk.sn_grad_mag_sf_boundary,
                &mut sn_grad_rho.bf,
                &rho.f,
                &rho.bf,
                m,
                nbf,
            )?;
        }
        {
            let Self { vk, sn_grad_alpha, alpha, .. } = self;
            launch_sn_grad(gpu, &vk.sn_grad_mag_sf, &mut sn_grad_alpha.f, &alpha.f, m, nif)?;
            launch_sn_grad_boundary(
                gpu,
                &vk.sn_grad_mag_sf_boundary,
                &mut sn_grad_alpha.bf,
                &alpha.f,
                &alpha.bf,
                m,
                nbf,
            )?;
        }

        let g = self.props.g;
        let sigma = self.props.sigma;

        if nif > 0 {
            let nl = nif as Label;
            let f = self.vk.body_force_flux.clone();
            unsafe {
                gpu.stream()
                    .launch_builder(&f)
                    .arg(&mut self.phib.f)
                    .arg(&self.sn_grad_rho.f)
                    .arg(&self.sn_grad_alpha.f)
                    .arg(&self.kappa_f.f)
                    .arg(&m.cf)
                    .arg(&g.x)
                    .arg(&g.y)
                    .arg(&g.z)
                    .arg(&sigma)
                    .arg(&nl)
                    .launch(cfg_for(nif))?;
            }
        }

        if nbf > 0 {
            let nl = nbf as Label;
            let f = self.vk.body_force_flux_boundary.clone();
            unsafe {
                gpu.stream()
                    .launch_builder(&f)
                    .arg(&mut self.phib.bf)
                    .arg(&self.sn_grad_rho.bf)
                    .arg(&self.sn_grad_alpha.bf)
                    .arg(&self.kappa_f.bf)
                    .arg(&m.b_cf)
                    .arg(&m.b_kind)
                    .arg(&self.u.fr)
                    .arg(&g.x)
                    .arg(&g.y)
                    .arg(&g.z)
                    .arg(&sigma)
                    .arg(&nl)
                    .launch(cfg_for(nbf))?;
            }
        }

        Ok(())
    }

    /// `|Sf| snGrad(p_rgh)`, the face force flux, and the cell force
    /// `reconstruct` gives back from it (§5.1).
    ///
    /// The pressure gradient is a FACE difference and only then reconstructed,
    /// for the reason [`crate::momentum::Momentum::update_force`] states at
    /// length: a cell-centred `grad p` cannot see the checkerboard mode, so a
    /// hydrostatic case grows a sawtooth that still contours smoothly.
    fn update_force(&mut self, gpu: &Gpu) -> Result<()> {
        let m = self.m;
        let (nif, nbf) = (m.n_internal_faces, m.n_boundary_faces);

        {
            let Self { fvk, sn_grad_p, p_rgh, .. } = self;
            fv::sn_grad_flux(gpu, fvk, sn_grad_p, p_rgh, &m.mag_sf, &m.b_mag_sf, m)?;
        }

        if self.ctrl.sn_grad.applies() {
            {
                let Self { fvk, grad_p, p_rgh, ctrl, .. } = self;
                fv::fvc_grad_scalar_scheme(gpu, fvk, grad_p, p_rgh, m, ctrl.grad_p)?;
            }
            let Self { fvk, sn_grad_p, p_rgh, grad_p, ctrl, .. } = self;
            fv::sn_grad_flux_correction(
                gpu,
                fvk,
                sn_grad_p,
                p_rgh,
                &m.mag_sf,
                &m.b_mag_sf,
                grad_p,
                ctrl.sn_grad,
                m,
            )?;
        }

        if nif > 0 {
            let nl = nif as Label;
            let f = self.vk.force_flux.clone();
            unsafe {
                gpu.stream()
                    .launch_builder(&f)
                    .arg(&mut self.force_flux.f)
                    .arg(&self.phib.f)
                    .arg(&self.sn_grad_p.f)
                    .arg(&nl)
                    .launch(cfg_for(nif))?;
            }
        }

        if nbf > 0 {
            let nl = nbf as Label;
            let f = self.vk.force_flux_boundary.clone();
            unsafe {
                gpu.stream()
                    .launch_builder(&f)
                    .arg(&mut self.force_flux.bf)
                    .arg(&self.phib.bf)
                    .arg(&self.sn_grad_p.bf)
                    .arg(&m.b_kind)
                    .arg(&self.u.fr)
                    .arg(&nl)
                    .launch(cfg_for(nbf))?;
            }
        }

        let Self { fvk, force, force_flux, .. } = self;
        fv::fvc_reconstruct(gpu, fvk, force, force_flux, m)
    }

    // ---- momentum ---------------------------------------------------------

    /// Fill [`Vof::uc`] with component `cmpt` of `U`, boundary state and all.
    fn fill_component(&mut self, gpu: &Gpu, cmpt: Label) -> Result<()> {
        let n = self.m.n_cells;
        let nbf = self.m.n_boundary_faces;

        let Self { vk, uc, u, fldk, .. } = self;

        launch_component(gpu, &vk.vec_component, &mut uc.f, &u.f, cmpt, n)?;
        launch_component(gpu, &vk.vec_component, &mut uc.f0, &u.f0, cmpt, n)?;
        launch_component(gpu, &vk.vec_component, &mut uc.f00, &u.f00, cmpt, n)?;
        launch_component(gpu, &vk.vec_component, &mut uc.bf, &u.bf, cmpt, nbf)?;
        launch_component(gpu, &vk.vec_component, &mut uc.ref_value, &u.ref_value, cmpt, nbf)?;
        launch_component(gpu, &vk.vec_component, &mut uc.ref_grad, &u.ref_grad, cmpt, nbf)?;

        field_ops::copy_field(gpu, fldk, &mut uc.fr, &u.fr, nbf)?;

        if nbf > 0 {
            let nl = nbf as Label;
            let f = vk.copy_label.clone();
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

    /// The convection weights, shared by the three components.
    ///
    /// The limiter sensor is `|U|`, for the reason
    /// [`crate::momentum::Momentum`] gives: one matrix cannot carry three sets
    /// of weights, and `|U|` is the one scalar the vector equation agrees on.
    fn update_div_weights(&mut self, gpu: &Gpu) -> Result<()> {
        let m = self.m;
        let scheme: fv::DivScheme = self.ctrl.div_scheme;

        if scheme.needs_gradient() {
            let (n, nbf) = (m.n_cells, m.n_boundary_faces);
            {
                let Self { vk, u_mag, u, .. } = self;
                launch_mag(gpu, &vk.mag, &mut u_mag.f, &u.f, n)?;
                launch_mag(gpu, &vk.mag, &mut u_mag.bf, &u.bf, nbf)?;
            }
            {
                let Self { fvk, grad_u_mag, u_mag, ctrl, .. } = self;
                fv::fvc_grad_scalar_scheme(gpu, fvk, grad_u_mag, u_mag, m, ctrl.grad_u)?;
            }

            let Self { fvk, w, bw, u_mag, grad_u_mag, rho_phi, .. } = self;
            fv::div_scheme_weights(
                gpu,
                fvk,
                Some(w),
                Some(bw),
                scheme,
                rho_phi,
                u_mag,
                Some(grad_u_mag),
                m,
            )
        } else {
            let Self { fvk, w, bw, u_mag, rho_phi, .. } = self;
            fv::div_scheme_weights(gpu, fvk, Some(w), Some(bw), scheme, rho_phi, u_mag, None, m)
        }
    }

    /// One component's matrix and source:
    ///
    /// ```text
    /// ddt(rho, U) + div(rho phi, U) - laplacian(mu, U) = force
    /// ```
    ///
    /// with `force` held back until the solve, because `H` is defined without
    /// it (§5.1) and folding it in would put a cell-centred body force on the
    /// faces.
    fn assemble_component(&mut self, gpu: &Gpu, dt: Scalar, cmpt: Label) -> Result<()> {
        let m = self.m;
        let n = m.n_cells;
        let r_dt = 1.0 / dt;

        self.fill_component(gpu, cmpt)?;
        self.a.zero(gpu)?;

        // §3.3 with rho/rho0: the conservative-form time derivative
        // d(rho U)/dt. `fvDdtEulerRho` has been in cuda/fv.cu, tested, since
        // the operators were written; this is its first caller.
        {
            let Self { fvk, a, uc, rho, rho0, .. } = self;
            fv::fvm_ddt_euler(gpu, fvk, a, m, Some(&rho.f), Some(rho0), &uc.f0, r_dt, 1.0)?;
        }

        // The convecting flux is the MASS flux (§20.3), not `phi`.
        //
        // And there is deliberately no `fvm_div_bounded_correction` here, for
        // a reason particular to the conservative form. That correction
        // subtracts `V_P (div u)_P` from the diagonal to cancel the spurious
        // source a non-solenoidal flux injects into `div(phi, psi)`. In this
        // equation the corresponding quantity is `Σ_f (±rho_phi_f)`, and
        // §20.3's construction makes that equal to `-(rho - rho0) V/dt`
        // exactly - which is not spurious at all, it is the other half of
        // `d(rho psi)/dt = rho d(psi)/dt + psi d(rho)/dt`. Subtracting it
        // would cancel the `ddt` term's own density weight and leave the
        // equation neither in conservative nor in non-conservative form.
        // `the_mass_flux_is_consistent_with_the_density_it_advects` measures
        // the identity this rests on.
        {
            let Self { fvk, a, uc, rho_phi, w, bw, .. } = self;
            fv::fvm_div_gauss(gpu, fvk, a, m, rho_phi, w, bw, uc, 1.0)?;
        }

        let scheme = self.ctrl.div_scheme;
        if scheme.correction().is_some() {
            {
                let Self { fvk, grad_uc, uc, ctrl, .. } = self;
                fv::fvc_grad_scalar_scheme(gpu, fvk, grad_uc, uc, m, ctrl.grad_u)?;
            }
            let Self { fvk, a, grad_uc, rho_phi, .. } = self;
            fv::fvm_div_correction(gpu, fvk, a, m, rho_phi, grad_uc, scheme, 1.0)?;
        }

        {
            let Self { fvk, a, uc, mu_mag_sf, .. } = self;
            fv::fvm_laplacian(gpu, fvk, a, m, &mu_mag_sf.f, &mu_mag_sf.bf, uc, -1.0)?;
        }

        if self.ctrl.sn_grad.applies() {
            {
                let Self { fvk, grad_uc, uc, ctrl, .. } = self;
                fv::fvc_grad_scalar_scheme(gpu, fvk, grad_uc, uc, m, ctrl.grad_u)?;
            }
            let Self { fvk, a, uc, mu_mag_sf, grad_uc, ctrl, .. } = self;
            fv::fvm_laplacian_non_orth_correction(
                gpu,
                fvk,
                a,
                m,
                &mu_mag_sf.f,
                &mu_mag_sf.bf,
                uc,
                grad_uc,
                ctrl.sn_grad,
                -1.0,
            )?;
        }

        if self.ctrl.u_relax < 1.0 {
            let alpha = self.ctrl.u_relax;
            let Self { lduk, a, uc, .. } = self;
            ldu_ops::relax(gpu, lduk, a, m, &uc.f, alpha)?;
        }

        ldu_ops::add_boundary_contributions(gpu, &self.lduk, &mut self.a, m)?;

        let Self { vk, su, a, .. } = self;
        launch_set_component(gpu, &vk.set_component, su, &a.source, cmpt, n)
    }

    /// Assemble the three component systems, and solve them if the predictor
    /// is on.
    fn momentum_predictor(&mut self, gpu: &Gpu, dt: Scalar) -> Result<[SolverPerformance; 3]> {
        let m = self.m;
        let n = m.n_cells;
        let mut perf = [SolverPerformance::default(); 3];
        if n == 0 {
            return Ok(perf);
        }

        self.update_div_weights(gpu)?;

        for c in 0..3 {
            self.assemble_component(gpu, dt, c as Label)?;
        }

        if !self.ctrl.momentum_predictor {
            // The matrix and `su` are still needed - they are what rAU and
            // HbyA are read out of - so the assembly above is not wasted. Only
            // the three solves are skipped.
            return Ok(perf);
        }

        for c in 0..3 {
            let cmpt = c as Label;
            {
                let Self { vk, uc, u, .. } = self;
                launch_component(gpu, &vk.vec_component, &mut uc.f, &u.f, cmpt, n)?;
            }
            {
                let nl = n as Label;
                let f = self.vk.solve_source.clone();
                unsafe {
                    gpu.stream()
                        .launch_builder(&f)
                        .arg(&mut self.a.source)
                        .arg(&self.su)
                        .arg(&self.force)
                        .arg(&m.v)
                        .arg(&cmpt)
                        .arg(&nl)
                        .launch(cfg_for(n))?;
                }
            }

            {
                let Self { solk, uc, a, ws, ctrl, .. } = self;
                perf[c] = solver::solve(gpu, solk, &mut uc.f, a, m, ws, &ctrl.u_solver)?;
            }

            let Self { vk, uc, u, .. } = self;
            launch_set_component(gpu, &vk.set_component, &mut u.f, &uc.f, cmpt, n)?;
        }

        field_ops::correct_boundary_conditions_vector(gpu, &self.fldk, &mut self.u, m)?;

        Ok(perf)
    }

    /// `rAU`, `HbyA`, `rAU_f` and `phi_HbyA` (§5.1).
    fn rhie_chow(&mut self, gpu: &Gpu) -> Result<()> {
        let m = self.m;
        let n = m.n_cells;
        if n == 0 {
            return Ok(());
        }
        let nl = n as Label;

        {
            let f = self.vk.rau.clone();
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

        for c in 0..3 {
            let cmpt = c as Label;
            {
                let Self { vk, uc, u, .. } = self;
                launch_component(gpu, &vk.vec_component, &mut uc.f, &u.f, cmpt, n)?;
            }
            {
                let Self { lduk, au, uc, a, .. } = self;
                ldu_ops::amul(gpu, lduk, au, &uc.f, a, m)?;
            }
            let f = self.vk.hbya.clone();
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

        let (nif, nbf) = (m.n_internal_faces, m.n_boundary_faces);

        {
            let Self { vk, rauf, rau, .. } = self;
            launch_face_interp(gpu, &vk.face_interp, &mut rauf.f, rau, m, nif)?;
            launch_face_interp_boundary(gpu, &vk.face_interp_boundary, &mut rauf.bf, rau, m, nbf)?;
        }
        {
            let Self { vk, rauf_mag_sf, rauf, .. } = self;
            launch_mul(gpu, &vk.mul, &mut rauf_mag_sf.f, &rauf.f, &m.mag_sf, nif)?;
            launch_mul(gpu, &vk.mul, &mut rauf_mag_sf.bf, &rauf.bf, &m.b_mag_sf, nbf)?;
        }

        if nif > 0 {
            let nfl = nif as Label;
            let f = self.vk.phi_hbya.clone();
            unsafe {
                gpu.stream()
                    .launch_builder(&f)
                    .arg(&mut self.phi_hbya.f)
                    .arg(&self.hbya)
                    .arg(&self.rauf.f)
                    .arg(&self.phib.f)
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
            let f = self.vk.phi_hbya_boundary.clone();
            unsafe {
                gpu.stream()
                    .launch_builder(&f)
                    .arg(&mut self.phi_hbya.bf)
                    .arg(&self.hbya)
                    .arg(&self.u.bf)
                    .arg(&self.rau)
                    .arg(&self.phib.bf)
                    .arg(&m.b_sf)
                    .arg(&m.b_weights)
                    .arg(&m.b_face_cells)
                    .arg(&m.b_nbr_cell)
                    .arg(&m.b_kind)
                    .arg(&self.u.fr)
                    .arg(&nbl)
                    .launch(cfg_for(nbf))?;
            }
        }

        Ok(())
    }

    /// `laplacian(rAU_f, p_rgh) = div(phi_HbyA)`.
    fn assemble_pressure(&mut self, gpu: &Gpu) -> Result<()> {
        let m = self.m;
        let n = m.n_cells;

        if self.ctrl.sn_grad.applies() {
            let Self { fvk, grad_p, p_rgh, ctrl, .. } = self;
            fv::fvc_grad_scalar_scheme(gpu, fvk, grad_p, p_rgh, m, ctrl.grad_p)?;
        }

        self.a_p.zero(gpu)?;

        {
            let Self { a_p, p_rgh, fvk, ctrl, rauf_mag_sf, grad_p, .. } = self;
            fv::fvm_laplacian(gpu, fvk, a_p, m, &rauf_mag_sf.f, &rauf_mag_sf.bf, p_rgh, 1.0)?;

            if ctrl.sn_grad.applies() {
                fv::fvm_laplacian_non_orth_correction(
                    gpu,
                    fvk,
                    a_p,
                    m,
                    &rauf_mag_sf.f,
                    &rauf_mag_sf.bf,
                    p_rgh,
                    grad_p,
                    ctrl.sn_grad,
                    1.0,
                )?;
            }
        }

        {
            let Self { vk, a_p, phi_hbya, .. } = self;
            launch_face_flux_sum(
                gpu,
                &vk.face_flux_sum,
                &mut a_p.source,
                &phi_hbya.f,
                &phi_hbya.bf,
                m,
                true,
            )?;
        }

        // A singular system needs a consistent right-hand side - the same
        // argument `smpSubScalar` in cuda/simple.cu spells out.
        if self.pinned && n > 0 {
            let Self { solk, ws, a_p, vk, .. } = self;
            solver::device_sum(gpu, solk, &mut ws.num, &a_p.source, &mut ws.partials, n)?;
            let scale = 1.0 / n as Scalar;
            launch_sub_scalar(gpu, &vk.sub_scalar, &mut a_p.source, &ws.num, scale, n)?;
        }

        ldu_ops::add_boundary_contributions(gpu, &self.lduk, &mut self.a_p, m)
    }

    fn fix_pressure_level(&mut self, gpu: &Gpu) -> Result<()> {
        let n = self.m.n_cells;
        if n == 0 {
            return Ok(());
        }
        let idx = self.reference_cell.min(n - 1) as Label;

        let f = self.vk.pick_value.clone();
        unsafe {
            gpu.stream()
                .launch_builder(&f)
                .arg(&mut self.red)
                .arg(&self.p_rgh.f)
                .arg(&idx)
                .launch(cfg_for(1))?;
        }

        let Self { vk, p_rgh, red, .. } = self;
        launch_sub_scalar(gpu, &vk.sub_scalar, &mut p_rgh.f, red, 1.0, n)
    }

    /// `phi = phi_HbyA - rAU_f |Sf| snGrad(p_rgh)` and
    /// `U = HbyA + rAU · force`.
    fn correct_flux_and_velocity(&mut self, gpu: &Gpu) -> Result<()> {
        let m = self.m;
        let (n, nif, nbf) = (m.n_cells, m.n_internal_faces, m.n_boundary_faces);

        if nif > 0 {
            let nl = nif as Label;
            let f = self.vk.correct_flux.clone();
            unsafe {
                gpu.stream()
                    .launch_builder(&f)
                    .arg(&mut self.phi.f)
                    .arg(&self.phi_hbya.f)
                    .arg(&self.rauf.f)
                    .arg(&self.sn_grad_p.f)
                    .arg(&nl)
                    .launch(cfg_for(nif))?;
            }
        }

        if nbf > 0 {
            let nl = nbf as Label;
            let f = self.vk.correct_flux_boundary.clone();
            unsafe {
                gpu.stream()
                    .launch_builder(&f)
                    .arg(&mut self.phi.bf)
                    .arg(&self.phi_hbya.bf)
                    .arg(&self.rauf.bf)
                    .arg(&self.sn_grad_p.bf)
                    .arg(&m.b_kind)
                    .arg(&self.u.fr)
                    .arg(&nl)
                    .launch(cfg_for(nbf))?;
            }
        }

        if n > 0 {
            let nl = n as Label;
            let f = self.vk.correct_velocity.clone();
            unsafe {
                gpu.stream()
                    .launch_builder(&f)
                    .arg(&mut self.u.f)
                    .arg(&self.hbya)
                    .arg(&self.rau)
                    .arg(&self.force)
                    .arg(&nl)
                    .launch(cfg_for(n))?;
            }
        }

        field_ops::correct_boundary_conditions_vector(gpu, &self.fldk, &mut self.u, m)
    }

    // ---- one step ---------------------------------------------------------

    /// One time step: `alpha`, then the properties it sets, then PISO.
    ///
    /// See the module header for why the order is the order.
    pub fn step(&mut self, gpu: &Gpu, dt: Scalar) -> Result<VofPerformance> {
        let m = self.m;
        let n = m.n_cells;
        if n == 0 {
            return Ok(VofPerformance::default());
        }
        if !(dt > 0.0) || !dt.is_finite() {
            return Err(Error::Config(format!(
                "Vof::step: deltaT = {dt}; a time step has to be finite and \
                 positive"
            )));
        }

        // The old levels this step differences against, taken BEFORE anything
        // moves: U^{n-1} for the momentum ddt, rho^{n-1} for the same ddt's
        // density weight.
        field_ops::store_old_time_vector(gpu, &self.fldk, &mut self.u)?;
        field_ops::store_old_time(gpu, &self.fldk, &mut self.alpha)?;
        field_ops::copy_field(gpu, &self.fldk, &mut self.rho0, &self.rho.f, n)?;

        // inletOutlet switches on the sign of the face flux, so the fractions
        // follow the flux the last step produced before anything reads them.
        field_ops::update_inlet_outlet_vector(gpu, &self.fldk, &mut self.u, &self.phi)?;
        field_ops::update_inlet_outlet_scalar(gpu, &self.fldk, &mut self.p_rgh, &self.phi)?;
        field_ops::update_inlet_outlet_scalar(gpu, &self.fldk, &mut self.alpha, &self.phi)?;
        field_ops::correct_boundary_conditions_vector(gpu, &self.fldk, &mut self.u, m)?;
        field_ops::correct_boundary_conditions(gpu, &self.fldk, &mut self.p_rgh, m)?;

        // §20.1 to §20.3.
        let (n_sub, co) = self.solve_alpha(gpu, dt)?;
        self.update_properties(gpu)?;
        self.update_rho_phi(gpu)?;

        // §20.4 and §20.5.
        self.update_body_force(gpu)?;
        self.update_force(gpu)?;

        // §3, §5.1.
        let u_perf = self.momentum_predictor(gpu, dt)?;

        // §5.4.
        let mut p_perf = SolverPerformance::default();
        for _ in 0..self.ctrl.n_correctors {
            self.rhie_chow(gpu)?;

            for _ in 0..=self.ctrl.n_non_orth_correctors {
                self.assemble_pressure(gpu)?;
                {
                    let Self { solk, p_rgh, a_p, ws, ctrl, .. } = self;
                    p_perf = solver::solve(gpu, solk, &mut p_rgh.f, a_p, m, ws, &ctrl.p_solver)?;
                }
                if self.pinned {
                    self.fix_pressure_level(gpu)?;
                }
                field_ops::correct_boundary_conditions(gpu, &self.fldk, &mut self.p_rgh, m)?;
            }

            self.update_force(gpu)?;
            self.correct_flux_and_velocity(gpu)?;
        }

        let continuity_error = if self.ctrl.report_continuity {
            self.continuity_error(gpu)?
        } else {
            0.0
        };

        Ok(VofPerformance {
            u: u_perf,
            p_rgh: p_perf,
            n_sub_cycles: n_sub,
            alpha_courant: co,
            continuity_error,
        })
    }

    // ---- diagnostics ------------------------------------------------------

    /// `max_c |Σ_f phi_f|` - how much volume the worst cell is inventing.
    pub fn continuity_error(&mut self, gpu: &Gpu) -> Result<Scalar> {
        let m = self.m;
        let n = m.n_cells;
        if n == 0 {
            return Ok(0.0);
        }

        {
            let Self { vk, div_phi, phi, .. } = self;
            launch_face_flux_sum(gpu, &vk.face_flux_sum, div_phi, &phi.f, &phi.bf, m, false)?;
        }
        {
            let Self { solk, ws, div_phi, .. } = self;
            solver::device_max_mag(gpu, solk, &mut ws.den, div_phi, &mut ws.partials, n)?;
        }
        Ok(gpu.download(&self.ws.den)?[0])
    }

    /// `Σ_c alpha_c V_c` - the volume of phase 1. Conserved exactly by a
    /// flux-form scheme, which is what makes it worth measuring.
    pub fn phase_volume(&mut self, gpu: &Gpu) -> Result<Scalar> {
        let m = self.m;
        let n = m.n_cells;
        if n == 0 {
            return Ok(0.0);
        }

        {
            let nl = n as Label;
            let f = self.vk.phase_volume.clone();
            unsafe {
                gpu.stream()
                    .launch_builder(&f)
                    .arg(&mut self.div_phi)
                    .arg(&self.alpha.f)
                    .arg(&m.v)
                    .arg(&nl)
                    .launch(cfg_for(n))?;
            }
        }
        {
            let Self { solk, ws, div_phi, .. } = self;
            solver::device_sum(gpu, solk, &mut ws.num, div_phi, &mut ws.partials, n)?;
        }
        Ok(gpu.download(&self.ws.num)?[0])
    }

    /// `rho^{n-1}`, as of the last [`Vof::step`]. Exposed so the §20.3
    /// consistency identity can be MEASURED rather than asserted in a comment.
    pub fn rho_old(&self) -> &DevBuf<Scalar> {
        &self.rho0
    }

    /// The mass flux the last step advected momentum with (§20.3).
    pub fn rho_phi(&self) -> &GpuSurfaceScalarField {
        &self.rho_phi
    }

    /// The accumulated limited `alpha` flux `rho_phi` was built from.
    pub fn alpha_phi(&self) -> &GpuSurfaceScalarField {
        &self.alpha_phi_sum
    }

    /// `(min alpha, max alpha)` over the cells. The measurement SPEC-LIT
    /// §20.2 says is "the entire justification for the machinery".
    pub fn alpha_bounds(&self, gpu: &Gpu) -> Result<(Scalar, Scalar)> {
        let n = self.m.n_cells;
        if n == 0 {
            return Ok((0.0, 0.0));
        }
        let a = gpu.download(&self.alpha.f)?;
        let mut lo = Scalar::INFINITY;
        let mut hi = Scalar::NEG_INFINITY;
        for v in a.iter().take(n) {
            if *v < lo {
                lo = *v;
            }
            if *v > hi {
                hi = *v;
            }
        }
        Ok((lo, hi))
    }
}

// ==========================================================================
//  Free launch helpers
//
//  Outside `impl` so a caller can hold `&mut` on one field of `Self` and `&`
//  on another. Every one guards `n == 0`, because a zero-block grid is an
//  invalid launch configuration and not a no-op.
// ==========================================================================

#[allow(clippy::too_many_arguments)]
fn launch_mixture(
    gpu: &Gpu,
    k: &CudaFunction,
    rho: &mut DevBuf<Scalar>,
    mu: &mut DevBuf<Scalar>,
    alpha: &DevBuf<Scalar>,
    rho1: Scalar,
    rho2: Scalar,
    mu1: Scalar,
    mu2: Scalar,
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
            .arg(rho)
            .arg(mu)
            .arg(alpha)
            .arg(&rho1)
            .arg(&rho2)
            .arg(&mu1)
            .arg(&mu2)
            .arg(&nl)
            .launch(cfg_for(n))?;
    }
    Ok(())
}

fn launch_rho_phi(
    gpu: &Gpu,
    k: &CudaFunction,
    out: &mut DevBuf<Scalar>,
    phi: &DevBuf<Scalar>,
    alpha_phi: &DevBuf<Scalar>,
    rho1: Scalar,
    rho2: Scalar,
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
            .arg(phi)
            .arg(alpha_phi)
            .arg(&rho1)
            .arg(&rho2)
            .arg(&nl)
            .launch(cfg_for(n))?;
    }
    Ok(())
}

fn launch_accumulate(
    gpu: &Gpu,
    k: &CudaFunction,
    sum: &mut DevBuf<Scalar>,
    x: &DevBuf<Scalar>,
    weight: Scalar,
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
            .arg(sum)
            .arg(x)
            .arg(&weight)
            .arg(&nl)
            .launch(cfg_for(n))?;
    }
    Ok(())
}

fn launch_sn_grad(
    gpu: &Gpu,
    k: &CudaFunction,
    out: &mut DevBuf<Scalar>,
    psi: &DevBuf<Scalar>,
    m: &GpuMesh,
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
            .arg(psi)
            .arg(&m.delta_coeffs)
            .arg(&m.mag_sf)
            .arg(&m.owner)
            .arg(&m.neighbour)
            .arg(&nl)
            .launch(cfg_for(n))?;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn launch_sn_grad_boundary(
    gpu: &Gpu,
    k: &CudaFunction,
    out: &mut DevBuf<Scalar>,
    psi: &DevBuf<Scalar>,
    bpsi: &DevBuf<Scalar>,
    m: &GpuMesh,
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
            .arg(psi)
            .arg(bpsi)
            .arg(&m.b_delta_coeffs)
            .arg(&m.b_mag_sf)
            .arg(&m.b_face_cells)
            .arg(&m.b_kind)
            .arg(&nl)
            .launch(cfg_for(n))?;
    }
    Ok(())
}

fn launch_face_interp(
    gpu: &Gpu,
    k: &CudaFunction,
    out: &mut DevBuf<Scalar>,
    x: &DevBuf<Scalar>,
    m: &GpuMesh,
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
            .arg(x)
            .arg(&m.weights)
            .arg(&m.owner)
            .arg(&m.neighbour)
            .arg(&nl)
            .launch(cfg_for(n))?;
    }
    Ok(())
}

fn launch_face_interp_boundary(
    gpu: &Gpu,
    k: &CudaFunction,
    out: &mut DevBuf<Scalar>,
    x: &DevBuf<Scalar>,
    m: &GpuMesh,
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
            .arg(x)
            .arg(&m.b_weights)
            .arg(&m.b_face_cells)
            .arg(&m.b_nbr_cell)
            .arg(&m.b_kind)
            .arg(&nl)
            .launch(cfg_for(n))?;
    }
    Ok(())
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

fn launch_face_flux_sum(
    gpu: &Gpu,
    k: &CudaFunction,
    out: &mut DevBuf<Scalar>,
    phi: &DevBuf<Scalar>,
    bphi: &DevBuf<Scalar>,
    m: &GpuMesh,
    accumulate: bool,
) -> Result<()> {
    let n = m.n_cells;
    if n == 0 {
        return Ok(());
    }
    let nl = n as Label;
    let acc: Label = if accumulate { 1 } else { 0 };
    let f = k.clone();
    unsafe {
        gpu.stream()
            .launch_builder(&f)
            .arg(out)
            .arg(phi)
            .arg(bphi)
            .arg(&m.cf_offset)
            .arg(&m.cf_face)
            .arg(&m.cf_own)
            .arg(&m.bcf_offset)
            .arg(&m.bcf_face)
            .arg(&acc)
            .arg(&nl)
            .launch(cfg_for(n))?;
    }
    Ok(())
}

fn launch_sub_scalar(
    gpu: &Gpu,
    k: &CudaFunction,
    x: &mut DevBuf<Scalar>,
    v: &DevBuf<Scalar>,
    scale: Scalar,
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
            .arg(x)
            .arg(v)
            .arg(&scale)
            .arg(&nl)
            .launch(cfg_for(n))?;
    }
    Ok(())
}

// ==========================================================================
//  Reading a case
//
//  SPEC-LIT §13.4 governs every lookup here: a setting the solver needs and
//  the case does not give is an ERROR naming the setting, not a guess, and
//  `-permissive` downgrades it to a warning that says what was substituted.
//  A setting the case does not mention at all and the solver has a documented
//  default for is a default, which is a different thing.
// ==========================================================================

/// The two phase names, from `phases (water air);`.
///
/// Phase 1 is the FIRST in the list, and it is the one `alpha.<name>` counts
/// and the one `rho1`/`mu1` describe. A case with no `phases` entry gets
/// `(water, air)`, which is what [`VofProperties::default`] is, so the two
/// cannot disagree.
pub fn phase_names(case_dir: &Path) -> Result<[String; 2]> {
    let mut names = ["water".to_string(), "air".to_string()];

    for nm in ["transportProperties", "physicalProperties"] {
        let p = case_dir.join("constant").join(nm);
        if !p.exists() {
            continue;
        }
        let d = FoamDict::read(&p)?;
        let Some(raw) = d.get("phases") else { break };

        let inner = raw
            .trim()
            .trim_start_matches('(')
            .trim_end_matches(')')
            .trim();
        let got: Vec<&str> = inner.split_whitespace().collect();

        if got.len() != 2 {
            return Err(Error::Config(format!(
                "constant/{nm}: phases ({}) names {} phase(s); this solver is \
                 the two-phase VOF of SPEC-LIT 20 and needs exactly two",
                inner,
                got.len()
            )));
        }
        names = [got[0].to_string(), got[1].to_string()];
        break;
    }

    Ok(names)
}

/// A dimensioned scalar the solver cannot proceed without.
///
/// `fallback_name` is printed by `-permissive`, so it has to be the truth
/// about `fallback` and not a phrase that sounds like one.
fn required(
    d: &FoamDict,
    key: &str,
    setting: &str,
    fallback: Scalar,
    fallback_name: &str,
) -> Result<Scalar> {
    if !d.has(key) {
        return unsupported(setting, "<missing>", &[], fallback_name, fallback);
    }
    let v = d.scalar(key, Scalar::NAN);
    if !v.is_finite() {
        return unreadable(setting, d.get_or(key, ""), "a number", fallback);
    }
    Ok(v)
}

impl VofProperties {
    /// Read `constant/transportProperties`, `constant/g` and the compression
    /// coefficient out of `system/fvSolution`.
    ///
    /// Viscosity may be given as the dynamic `mu` or the kinematic `nu`; the
    /// second is converted with that phase's own `rho`, which is the only
    /// density it could mean. `mu` wins where both are present.
    pub fn from_case(case_dir: &Path) -> Result<Self> {
        let mut c = Self::default();
        let names = phase_names(case_dir)?;

        let mut found = false;
        for nm in ["transportProperties", "physicalProperties"] {
            let p = case_dir.join("constant").join(nm);
            if !p.exists() {
                continue;
            }
            let d = FoamDict::read(&p)?;
            found = true;

            let phase = |i: usize, rho_d: Scalar, mu_d: Scalar| -> Result<(Scalar, Scalar)> {
                let n = &names[i];
                let rho = required(
                    &d,
                    &format!("{n}/rho"),
                    &format!("constant/{nm}: {n}/rho"),
                    rho_d,
                    "the built-in default for this phase",
                )?;
                if !(rho > 0.0) {
                    return Err(Error::Config(format!(
                        "constant/{nm}: {n}/rho = {rho} is not a positive density"
                    )));
                }

                let mu = if d.has(&format!("{n}/mu")) {
                    required(
                        &d,
                        &format!("{n}/mu"),
                        &format!("constant/{nm}: {n}/mu"),
                        mu_d,
                        "the built-in default for this phase",
                    )?
                } else if d.has(&format!("{n}/nu")) {
                    // Kinematic: mu = rho nu, with that phase's own rho.
                    rho * required(
                        &d,
                        &format!("{n}/nu"),
                        &format!("constant/{nm}: {n}/nu"),
                        mu_d / rho,
                        "the built-in default for this phase",
                    )?
                } else {
                    unsupported(
                        &format!("constant/{nm}: {n}/mu"),
                        "<missing>",
                        &["mu (dynamic)", "nu (kinematic)"],
                        "the built-in default",
                        mu_d,
                    )?
                };

                Ok((rho, mu))
            };

            let (r1, m1) = phase(0, c.rho1, c.mu1)?;
            let (r2, m2) = phase(1, c.rho2, c.mu2)?;
            c.rho1 = r1;
            c.mu1 = m1;
            c.rho2 = r2;
            c.mu2 = m2;

            // `sigma` is REQUIRED, and gets the same treatment `nu` gets in
            // `read_case_controls`. A surface tension the case did not state
            // is exactly the kind of setting a default hides: §20.4's force is
            // `sigma kappa grad(alpha)`, so a wrong `sigma` changes every
            // interface in the run and shows up as nothing but a slightly
            // different answer. Defaulting it to water-against-air would be
            // asserting a fluid pair on the user's behalf, and defaulting it
            // to zero would silently delete a term.
            //
            // `sigma 0;` is one line, and it says "this case models no surface
            // tension" out loud - which is what `-permissive` substitutes.
            c.sigma = required(
                &d,
                "sigma",
                &format!("constant/{nm}: sigma"),
                0.0,
                "zero, i.e. no surface tension at all",
            )?;
            break;
        }

        if !found {
            return Err(Error::Config(format!(
                "{}: no constant/transportProperties or \
                 constant/physicalProperties, so there is nothing that says \
                 what the two fluids are",
                case_dir.display()
            )));
        }

        // `constant/g`, through the reader that already knows both spellings.
        c.g = BuoyancyCoeffs::from_case(case_dir)?.g;

        // §20.1's c_alpha. `PIMPLE/cAlpha` is where this crate writes it;
        // `solvers/alpha.<phase1>/cAlpha` is where OpenFOAM's own cases put
        // it, and both are accepted because a user should not have to move an
        // entry to run a case they already have.
        let fvs = case_dir.join("system").join("fvSolution");
        if fvs.exists() {
            let d = FoamDict::read(&fvs)?;
            let alpha_key = format!("alpha.{}", names[0]);
            let mut ca = c.c_alpha;
            if let Some(k) = d.resolve("solvers", &alpha_key)? {
                ca = d.scalar(&format!("solvers/{k}/cAlpha"), ca);
            }
            c.c_alpha = d.scalar("PIMPLE/cAlpha", ca);
        }

        c.validate()?;
        Ok(c)
    }

    /// A one-line summary, for the banner every driver prints.
    pub fn describe(&self, names: &[String; 2]) -> String {
        format!(
            "{}: rho {} mu {} | {}: rho {} mu {} | sigma {} | g ({} {} {}) | cAlpha {}",
            names[0],
            self.rho1,
            self.mu1,
            names[1],
            self.rho2,
            self.mu2,
            self.sigma,
            self.g.x,
            self.g.y,
            self.g.z,
            self.c_alpha
        )
    }

    /// The Atwood number, `(rho1 - rho2)/(rho1 + rho2)`. Printed because it is
    /// the one number that says how hard the case is going to be on the
    /// pressure equation.
    pub fn atwood(&self) -> Scalar {
        (self.rho1 - self.rho2) / (self.rho1 + self.rho2)
    }
}

impl VofControls {
    /// Read `system/fvSolution` and `system/fvSchemes`.
    ///
    /// The keys, and where SPEC-LIT puts each of them:
    ///
    /// ```text
    /// solvers/p_rgh/...                  §8    the pressure linear solver
    /// solvers/U/...                      §8    the momentum linear solver
    /// <algo>/nCorrectors                §5.4  PISO correctors
    /// <algo>/nNonOrthogonalCorrectors   §3.2
    /// <algo>/momentumPredictor          §5.4  *DESIGN*, see the field
    /// <algo>/maxAlphaCo                 §20.2 the sub-cycle Courant limit
    /// <algo>/maxAlphaSubCycles          §20.2
    /// <algo>/nAlphaLimiterIters         §20.2 *DESIGN*, three
    /// ddtSchemes/default                §3.3  Euler, and only Euler
    /// divSchemes/div(rhoPhi,U)          §20.3 the MASS flux, not phi
    /// gradSchemes/grad(U)               §12.1 the momentum gradient
    /// gradSchemes/grad(p_rgh)           §12.1 the pressure gradient
    /// gradSchemes/grad(alpha.<phase1>)  §20.1 the interface normal
    /// laplacianSchemes / snGradSchemes  §12.3 through `resolve_sn_grad`
    /// relaxationFactors/equations/U     §5.2
    /// controlDict/deltaT
    /// controlDict/adjustTimeStep, maxCo, maxDeltaT   §20.2, the alpha step
    /// ```
    ///
    /// `<algo>` is whichever of `SIMPLE`/`PISO`/`PIMPLE` the case wrote, via
    /// [`crate::io::case::AlgorithmControls::read`]. This module used to look
    /// up the literal `PIMPLE/...` spellings only, so a two-phase case whose
    /// entries sat in a `PISO` dictionary - which is what this solver's loop
    /// actually is - ran on [`VofControls::default`]'s numbers with nothing
    /// printed.
    ///
    /// `deltaT` is read here so a driver with no `-deltaT` on its command line
    /// still has the case's own step rather than a number this file invented.
    ///
    /// **SPEC-LIT §13.4.1, instance 5.** Everything in the table above except
    /// the solvers, the three `<algo>` alpha entries and `deltaT` was read by
    /// nobody until this sweep; the settings this solver CANNOT honour
    /// (`ddtSchemes` other than `Euler`, `nOuterCorrectors > 1`,
    /// `relaxationFactors/fields/p_rgh`, a `bounded` prefix on the momentum
    /// convection, `residualControl`) are now §13.4 errors naming the
    /// alternative rather than entries that parse and vanish.
    pub fn from_case(case_dir: &Path) -> Result<Self> {
        let mut c = Self::default();
        let names = phase_names(case_dir)?;

        let fvs = case_dir.join("system").join("fvSolution");
        if fvs.exists() {
            let d = FoamDict::read(&fvs)?;

            read_solver_controls(&mut c.p_solver, &d, "p_rgh")?;
            read_solver_controls(&mut c.u_solver, &d, "U")?;

            // Whichever algorithm dictionary the case wrote, not the literal
            // `PIMPLE` spelling - SPEC-LIT §14, and the same reader every
            // other driver uses.
            let algo = crate::io::case::AlgorithmControls::read(&d);
            c.n_correctors = algo.n_correctors as Label;
            c.n_non_orth_correctors = algo.n_non_orth_correctors as Label;
            c.momentum_predictor = algo.momentum_predictor;

            // SPEC-LIT §13.4, "recognised, not implemented". §20's step is
            // PISO: one alpha sub-cycle sequence, one momentum predictor and
            // `nCorrectors` pressure correctors, with no outer
            // re-linearisation. `nOuterCorrectors > 1` asks for a loop that
            // does not exist here.
            if algo.n_outer_correctors > 1 {
                crate::io::contract::unsupported(
                    &format!("{}/nOuterCorrectors", algo.dict),
                    &algo.n_outer_correctors.to_string(),
                    &["1"],
                    "one outer corrector - ofgpu-vof's step is PISO, and nCorrectors is the corrector count it does have",
                    (),
                )?;
            }

            // The alpha-equation entries live in the same dictionary. Looked
            // up under whichever one carried the rest, falling back to
            // `PIMPLE` for a case that names none.
            let dict = if algo.dict.is_empty() { "PIMPLE" } else { algo.dict };
            c.max_alpha_co = d.scalar(&format!("{dict}/maxAlphaCo"), c.max_alpha_co);
            c.max_sub_cycles = d.label(&format!("{dict}/maxAlphaSubCycles"), c.max_sub_cycles);
            c.n_limiter_iters = d.label(&format!("{dict}/nAlphaLimiterIters"), c.n_limiter_iters);

            // Through the pattern resolver, so `equations { ".*" 1; }` reaches
            // `U` (SPEC-LIT §13.4.1 and `relaxation_factor`'s own doc).
            c.u_relax = crate::io::case::relaxation_factor(&d, "U", c.u_relax)?;

            // SPEC-LIT §13.4. `p_rgh` is corrected, not relaxed: §5.4's PISO
            // applies the whole correction because the equation is solved
            // afresh inside every corrector, and there is no `p_relax` in
            // this module to put the number in. A case asking for one is
            // asking for SIMPLE's pressure relaxation in a PISO loop.
            for spelling in ["fields", "equations"] {
                let key = format!("relaxationFactors/{spelling}/p_rgh");
                if d.has(&key) {
                    crate::io::contract::unsupported_note(
                        &key,
                        d.get_or(&key, "").trim(),
                        &[],
                        "ofgpu-vof's step is PISO (SPEC-LIT §5.4): the pressure correction is applied whole, so there is no under-relaxation of p_rgh to set. relaxationFactors/equations/U is the one relaxation this loop has",
                        "no pressure relaxation - the correction is applied whole",
                        (),
                    )?;
                }
            }

            // SPEC-LIT §13.4. `residualControl` stops an OUTER loop on the
            // initial residuals; this driver marches to `-endTime` and has no
            // outer loop for it to stop.
            let rc = crate::io::case::ResidualControl::read(&d);
            if !rc.is_empty() {
                let list: Vec<String> = rc.iter().map(|(f, t)| format!("{f} {t:e}")).collect();
                crate::io::contract::unsupported_note(
                    "residualControl",
                    &list.join(", "),
                    &[],
                    "ofgpu-vof marches to -endTime and has no outer loop for a residual test to stop; ofgpu-buoyant, ofgpu-plume, ofgpu-k-epsilon and ofgpu-k-omega do honour it",
                    "no residual-based stopping - the run ends on -endTime",
                    (),
                )?;
            }
        }

        let sch = case_dir.join("system").join("fvSchemes");
        if sch.exists() {
            let s = FvSchemes::from_dict(FoamDict::read(&sch)?);

            // The convecting flux of the two-phase momentum equation is
            // `rhoPhi` (§20.3), so that is the key looked up. `div(phi,U)` is
            // accepted as a fallback for a case written for a single-phase
            // solver, and the fallback is silent only because the two entries
            // mean the same scheme applied to different fluxes - the flux
            // itself is never taken from the dictionary.
            let key = if s.dict().has("divSchemes/div(rhoPhi,U)") {
                "div(rhoPhi,U)"
            } else {
                "div(phi,U)"
            };
            let conv = s.div(key)?;
            c.div_scheme = conv.scheme;

            // SPEC-LIT §13.4. The `bounded` prefix used to be parsed and
            // dropped, which is the silent substitution §13.4 forbids - and
            // here the substituted answer is the RIGHT one, which makes it
            // worse rather than better, because nothing tells the user their
            // entry was overruled. `assemble_component`'s own comment derives
            // why the conservative form must NOT subtract `Σ_f rho_phi_f`
            // from the diagonal: §20.3 makes that quantity equal
            // `-(rho - rho0) V/dt` exactly, which is the other half of
            // `d(rho psi)/dt`, not a spurious source. A case asking for the
            // correction is asking for an equation that is neither
            // conservative nor non-conservative, and has to be told.
            if conv.bounded {
                crate::io::contract::unsupported_note(
                    &format!("divSchemes/{key}"),
                    &format!("bounded {}", conv.scheme.describe()),
                    &[],
                    "the two-phase momentum equation is in CONSERVATIVE form (SPEC-LIT §20.3): its ddt carries rho/rho0 and the face sum of rhoPhi is exactly -(rho - rho0)V/dt, not a spurious source, so the bounded correction would cancel the ddt term's own density weight. Write the same entry without the bounded prefix",
                    "the same scheme UNbounded, which is what §20.3 requires",
                    (),
                )?;
            }

            // Each field's own gradient - SPEC-LIT §13.4.1(a). `gradSchemes`
            // was not read by this module at all, so every one of the six
            // `fvc_grad_scalar` call sites ran plain Gauss linear whatever
            // the case asked for.
            c.grad_u = s.grad("grad(U)")?;
            c.grad_p = s.grad("grad(p_rgh)")?;
            c.grad_alpha = s.grad(&format!("grad(alpha.{})", names[0]))?;

            // `laplacianSchemes` where the case names one, `snGradSchemes`
            // otherwise - `resolve_sn_grad`'s own rule, and the reason it
            // exists. This module used to read `snGradSchemes/default` only,
            // so a case writing `laplacianSchemes { default Gauss linear
            // corrected; }` beside `snGradSchemes { default uncorrected; }`
            // ran its laplacians uncorrected and said nothing.
            c.sn_grad = crate::io::case::resolve_sn_grad(&s, "laplacian(muEff,U)")?;

            // SPEC-LIT §13.4 and §3.3. `fvm_ddt_euler` is the only time
            // derivative §20 assembles, and `assemble_component`'s density
            // weighting is written for it specifically: `backward` would need
            // a second old-time level of `rho U`, which this module does not
            // keep, and `steadyState` would drop the term a moving interface
            // is entirely made of.
            let raw = s.dict().get_or("ddtSchemes/default", "Euler");
            let ddt = crate::timescheme::DdtScheme::parse(raw)
                .unwrap_or(crate::timescheme::DdtScheme::Euler);
            if ddt != crate::timescheme::DdtScheme::Euler {
                crate::io::contract::unsupported(
                    "ddtSchemes/default",
                    raw.trim(),
                    &["Euler"],
                    "Euler - the only time derivative SPEC-LIT §20 assembles",
                    (),
                )?;
            }
        }

        let cd = case_dir.join("system").join("controlDict");
        if cd.exists() {
            let d = FoamDict::read(&cd)?;
            c.delta_t = d.scalar("deltaT", c.delta_t);

            // SPEC-LIT §13.4 and §20.2. `ofgpu-vof` is the one driver in this
            // crate that DOES adapt its step, so these three are honoured here
            // rather than refused. `-maxCo`/`-maxDeltaT` on the command line
            // still win; the case's own entries are what a run with neither
            // flag gets. `read_control_dict` refuses `adjustTimeStep yes` for
            // every driver that has no such loop.
            c.adjust_time_step = d.bool("adjustTimeStep", c.adjust_time_step);
            c.max_co = d.scalar("maxCo", c.max_co);
            c.max_delta_t = d.scalar("maxDeltaT", c.max_delta_t);
        }

        c.validate()?;
        Ok(c)
    }
}

// ==========================================================================
//  Tests - SPEC-LIT §22's VOF rows, and the identities §20 rests on
//
//  Nothing here compares against another CFD code. Every check is either an
//  analytic identity (the curvature of a circle, the Laplace jump, a
//  hydrostatic balance), a conservation statement the discretisation must
//  satisfy exactly, or a boundedness statement that is the entire reason the
//  Zalesak machinery exists.
// ==========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::case::{LinearSolverKind, Preconditioner};
    use crate::mesh::PatchKind;
    use crate::GpuMesh;

    /// A machine without a card makes every device test pass vacuously, which
    /// is the convention the rest of the crate follows.
    fn gpu() -> Option<Gpu> {
        Gpu::new(0).ok()
    }

    /// A structured box, origin at zero, `n` cells of size `d`, with the six
    /// patch kinds the caller asks for. Patch order is `-x +x -y +y -z +z`.
    fn boxed(n: [usize; 3], d: Vec3, kinds: [PatchKind; 6]) -> HostMesh {
        let (mut m, points, faces) = crate::mesh::topology::tests::box_mesh(n, d);

        for (p, k) in m.patches.iter_mut().zip(kinds) {
            p.kind = k;
            p.type_name = match k {
                PatchKind::Wall => "wall",
                PatchKind::Empty => "empty",
                PatchKind::Symmetry => "symmetry",
                _ => "patch",
            }
            .to_string();
        }

        m.compute_geometry(&points, &faces).expect("box geometry");
        m.build_cell_face_maps();
        m
    }

    /// Which patch each boundary face belongs to.
    fn patch_of(hm: &HostMesh) -> Vec<usize> {
        let mut v = vec![0usize; hm.n_boundary_faces];
        for (p, pi) in hm.patches.iter().enumerate() {
            for k in 0..pi.size {
                v[pi.start + k] = p;
            }
        }
        v
    }

    /// `Some(value)` is a Dirichlet, `None` zero-gradient. An `empty` patch
    /// stays empty whatever is asked for.
    fn set_scalar_bcs(
        gpu: &Gpu,
        s: &mut GpuScalarField,
        hm: &HostMesh,
        per_patch: &[Option<Scalar>],
    ) -> Result<()> {
        let nbf = hm.n_boundary_faces;
        let owner = patch_of(hm);

        let mut kind = vec![BcKind::ZeroGradient as Label; nbf];
        let mut fr = vec![0.0 as Scalar; nbf];
        let mut rv = vec![0.0 as Scalar; nbf];
        let rg = vec![0.0 as Scalar; nbf];

        for i in 0..nbf {
            let p = owner[i];
            if hm.patches[p].kind == PatchKind::Empty {
                kind[i] = BcKind::Empty as Label;
                continue;
            }
            if let Some(v) = per_patch[p] {
                kind[i] = BcKind::FixedValue as Label;
                fr[i] = 1.0;
                rv[i] = v;
            }
        }

        gpu.write(&mut s.bc_kind, &kind)?;
        gpu.write(&mut s.fr, &fr)?;
        gpu.write(&mut s.ref_value, &rv)?;
        gpu.write(&mut s.ref_grad, &rg)
    }

    fn set_velocity_bcs(
        gpu: &Gpu,
        u: &mut GpuVectorField,
        hm: &HostMesh,
        per_patch: &[Option<Vec3>],
    ) -> Result<()> {
        let nbf = hm.n_boundary_faces;
        let owner = patch_of(hm);

        let mut kind = vec![BcKind::ZeroGradient as Label; nbf];
        let mut fr = vec![0.0 as Scalar; nbf];
        let mut rv = vec![Vec3::ZERO; nbf];
        let rg = vec![Vec3::ZERO; nbf];

        for i in 0..nbf {
            let p = owner[i];
            if hm.patches[p].kind == PatchKind::Empty {
                kind[i] = BcKind::Empty as Label;
                continue;
            }
            if let Some(v) = per_patch[p] {
                kind[i] = BcKind::FixedValue as Label;
                fr[i] = 1.0;
                rv[i] = v;
            }
        }

        gpu.write(&mut u.bc_kind, &kind)?;
        gpu.write(&mut u.fr, &fr)?;
        gpu.write(&mut u.ref_value, &rv)?;
        gpu.write(&mut u.ref_grad, &rg)
    }

    /// The face flux of an analytic velocity field, `phi_f = u(Cf)·Sf`.
    ///
    /// For a solid-body rotation or a uniform stream on a Cartesian mesh this
    /// is discretely solenoidal to round-off: the two faces normal to `x`
    /// share a face-centre `y` and therefore the same `u_x`, so their
    /// contributions cancel exactly, and likewise for `y` and `z`. That is
    /// what makes the advection tests below tests of the SCHEME rather than of
    /// a flux that was not conservative to begin with.
    fn write_analytic_flux(
        gpu: &Gpu,
        phi: &mut GpuSurfaceScalarField,
        hm: &HostMesh,
        u: impl Fn(Vec3) -> Vec3,
    ) -> Result<()> {
        let f: Vec<Scalar> = (0..hm.n_internal_faces)
            .map(|i| u(hm.cf[i]).dot(hm.sf[i]))
            .collect();
        let bf: Vec<Scalar> = (0..hm.n_boundary_faces)
            .map(|i| {
                if hm.b_kind[i] == PatchKind::Empty as Label {
                    0.0
                } else {
                    u(hm.b_cf[i]).dot(hm.b_sf[i])
                }
            })
            .collect();

        gpu.write(&mut phi.f, &f)?;
        gpu.write(&mut phi.bf, &bf)
    }

    /// The largest `|Σ_f phi_f|` the analytic flux leaves in any cell. Should
    /// be round-off; if it is not, an advection test measures the flux and not
    /// the scheme.
    fn host_div_phi(hm: &HostMesh, f: &[Scalar], bf: &[Scalar]) -> Scalar {
        let mut d = vec![0.0 as Scalar; hm.n_cells];
        for i in 0..hm.n_internal_faces {
            d[hm.owner[i] as usize] += f[i];
            d[hm.neighbour[i] as usize] -= f[i];
        }
        for i in 0..hm.n_boundary_faces {
            d[hm.b_face_cells[i] as usize] += bf[i];
        }
        d.iter().fold(0.0 as Scalar, |a, x| a.max(x.abs()))
    }

    /// The cell whose centre is nearest `p`.
    fn nearest_cell(hm: &HostMesh, p: Vec3) -> usize {
        let mut best = 0;
        let mut bd = Scalar::MAX;
        for c in 0..hm.n_cells {
            let d = (hm.c[c] - p).mag_sqr();
            if d < bd {
                bd = d;
                best = c;
            }
        }
        best
    }

    fn tight() -> SolverControls {
        SolverControls {
            tolerance: 1e-12,
            rel_tol: 0.0,
            max_iter: 1000,
            check_interval: 1,
            solver: LinearSolverKind::PBiCGStab,
            precon: Preconditioner::Diagonal,
            ..SolverControls::default()
        }
    }

    /// Advection only: no momentum, no pressure, no gravity, and both phases
    /// given the same properties so nothing but `alpha` can move.
    fn advection_only(c_alpha: Scalar) -> (VofProperties, VofControls) {
        (
            VofProperties {
                rho1: 1.0,
                rho2: 1.0,
                mu1: 0.0,
                mu2: 0.0,
                sigma: 0.0,
                g: Vec3::ZERO,
                c_alpha,
            },
            VofControls {
                max_alpha_co: 0.5,
                n_limiter_iters: 3,
                report_continuity: false,
                ..VofControls::default()
            },
        )
    }

    fn max_speed(gpu: &Gpu, vof: &Vof<'_>) -> Result<Scalar> {
        let u = gpu.download(&vof.u().f)?;
        Ok(u.iter().fold(0.0 as Scalar, |a, v| a.max(v.mag())))
    }

    // ----------------------------------------------------------------------
    //  SPEC-LIT §22: "VOF boundedness - a rotating slotted disc (Zalesak
    //  1979) - alpha in [0,1] exactly, shape preserved after one revolution"
    // ----------------------------------------------------------------------

    /// Zalesak's own test case, and the one that justifies the limiter.
    ///
    /// A slotted disc in a unit square is carried once round by a solid-body
    /// rotation about the centre and must come back where it started. Three
    /// things are measured:
    ///
    /// 1. `alpha` never leaves `[0, 1]` - to round-off, not to a tolerance.
    ///    This is the whole of §20.2: a value of `-1e-3` gives a negative
    ///    density and the pressure solve diverges.
    /// 2. the phase volume is conserved, because the scheme is in flux form
    ///    and the limiter scales fluxes rather than values;
    /// 3. the slot is still there - a probe in the middle of it reads below
    ///    a half while probes either side of it, inside the disc, read above.
    ///    A scheme that has diffused the disc into a blur fails 3 while
    ///    passing 1 and 2, which is why all three are here.
    #[test]
    fn zalesaks_rotating_slotted_disc_stays_bounded_and_keeps_its_slot() -> Result<()> {
        let Some(g) = gpu() else { return Ok(()) };

        let nx = 100usize;
        let h = 1.0 / nx as Scalar;
        let hm = boxed(
            [nx, nx, 1],
            Vec3::new(h, h, h),
            [
                PatchKind::Wall,
                PatchKind::Wall,
                PatchKind::Wall,
                PatchKind::Wall,
                PatchKind::Empty,
                PatchKind::Empty,
            ],
        );
        let m = GpuMesh::upload(&g, &hm)?;

        let (props, mut ctrl) = advection_only(1.0);
        ctrl.delta_t = 1.0;
        // §20.2's sub-cycle count is chosen from a Courant number that BOUNDS
        // the compression flux by c_alpha times the advective one, so at
        // c_alpha = 1 the estimate is twice the advective Courant number. The
        // step below is Co = 0.31 advective, 0.62 bounded; 0.8 keeps this one
        // sub-cycle per step and leaves the sub-cycling itself to the dam
        // break, where the flux really does vary.
        ctrl.max_alpha_co = 0.8;
        let mut vof = Vof::new(&g, &hm, &m, props, ctrl)?;

        // Zalesak's disc: radius 0.15 about (0.5, 0.75), with a slot 0.05
        // wide cut in from the bottom to y = 0.85.
        let centre = Vec3::new(0.5, 0.75, 0.0);
        let radius = 0.15 as Scalar;
        let in_disc = |p: Vec3| -> bool {
            let r = ((p.x - centre.x).powi(2) + (p.y - centre.y).powi(2)).sqrt();
            if r > radius {
                return false;
            }
            !((p.x - centre.x).abs() < 0.025 && p.y < 0.85)
        };

        let a0: Vec<Scalar> = (0..hm.n_cells)
            .map(|c| if in_disc(hm.c[c]) { 1.0 } else { 0.0 })
            .collect();
        g.write(&mut vof.alpha_mut().f, &a0)?;
        set_scalar_bcs(&g, vof.alpha_mut(), &hm, &[None; 6])?;

        // Solid-body rotation about the centre of the box, omega = 1 rad/s.
        let omega = 1.0 as Scalar;
        let rot = |p: Vec3| Vec3::new(-omega * (p.y - 0.5), omega * (p.x - 0.5), 0.0);
        write_analytic_flux(&g, vof.phi_mut(), &hm, rot)?;
        vof.initialise(&g)?;

        // The flux has to be conservative or this measures the wrong thing.
        let fi = g.download(&vof.phi().f)?;
        let fb = g.download(&vof.phi().bf)?;
        let dmax = host_div_phi(&hm, &fi, &fb);
        assert!(
            dmax < 1e-15,
            "the rotation flux is not solenoidal: max |sum phi| = {dmax:e}"
        );

        let v0 = vof.phase_volume(&g)?;

        // One full revolution, split so the alpha Courant number is about a
        // third and no sub-cycling is needed.
        let steps = 2000;
        let dt = (2.0 * std::f64::consts::PI / steps as f64) as Scalar;
        for _ in 0..steps {
            let (n_sub, co) = vof.solve_alpha(&g, dt)?;
            assert_eq!(n_sub, 1, "unexpected sub-cycling at Co = {co}");
        }

        let (lo, hi) = vof.alpha_bounds(&g)?;
        let v1 = vof.phase_volume(&g)?;

        println!(
            "  Zalesak disc after one revolution: alpha in [{lo:e}, {hi}], \
             volume {v0:e} -> {v1:e}"
        );

        // 1. Boundedness, to round-off.
        assert!(lo >= -1e-12, "alpha went negative: min = {lo:e}");
        assert!(hi <= 1.0 + 1e-12, "alpha exceeded one: max = {hi:e}");

        // 2. Conservation. Flux form plus a face-wise limiter, so this is an
        //    identity of the scheme and not an accident of the case.
        //
        // The bound is 1e-8 rather than machine epsilon because 2000 steps of
        // O(1) arithmetic on 10^4 cells accumulate round-off: what the scheme
        // guarantees is that the flux leaving one cell is the flux entering
        // the next, and the residue is the floating-point sum of two million
        // such cancellations, not a leak.
        let rel = ((v1 - v0) / v0).abs();
        assert!(rel < 1e-8, "phase volume drifted by {rel:e}");

        // 3. The slot. The disc is back where it started, so the four probes
        //    are the ones the initial condition would answer 0, 1, 1, 1.
        let a = g.download(&vof.alpha().f)?;
        let slot = a[nearest_cell(&hm, Vec3::new(0.5, 0.70, 0.0))];
        let left = a[nearest_cell(&hm, Vec3::new(0.44, 0.70, 0.0))];
        let right = a[nearest_cell(&hm, Vec3::new(0.56, 0.70, 0.0))];
        let top = a[nearest_cell(&hm, Vec3::new(0.5, 0.88, 0.0))];

        println!(
            "  probes: slot {slot:.4}  left {left:.4}  right {right:.4}  \
             bridge {top:.4}"
        );

        assert!(slot < 0.5, "the slot has filled in: alpha = {slot}");
        assert!(left > 0.5, "the disc left of the slot has gone: {left}");
        assert!(right > 0.5, "the disc right of the slot has gone: {right}");
        assert!(top > 0.5, "the bridge over the slot has gone: {top}");

        Ok(())
    }

    // ----------------------------------------------------------------------
    //  SPEC-LIT §22: "VOF compression - a translating interface - interface
    //  width does not grow"
    // ----------------------------------------------------------------------

    /// A step in `alpha` carried along a uniform stream.
    ///
    /// Two runs, identical but for `c_alpha`. The compressed one must end no
    /// wider than it was after the first few steps, and must be strictly
    /// sharper than the uncompressed one - which is the measurement that the
    /// term of §20.1 is doing what the section says it does, rather than being
    /// present and inert.
    #[test]
    fn a_translating_interface_does_not_widen() -> Result<()> {
        let Some(g) = gpu() else { return Ok(()) };

        let nx = 200usize;
        let h = 0.005 as Scalar;
        let hm = boxed(
            [nx, 1, 1],
            Vec3::new(h, 0.01, 0.01),
            [
                PatchKind::Generic,
                PatchKind::Generic,
                PatchKind::Wall,
                PatchKind::Wall,
                PatchKind::Empty,
                PatchKind::Empty,
            ],
        );
        let m = GpuMesh::upload(&g, &hm)?;

        // 0.01 < alpha < 0.99 - the cells that are neither phase.
        let width = |a: &[Scalar]| -> usize {
            a.iter().filter(|v| **v > 0.01 && **v < 0.99).count()
        };

        let run = |c_alpha: Scalar| -> Result<(usize, usize, Scalar, Scalar)> {
            let (props, mut ctrl) = advection_only(c_alpha);
            ctrl.delta_t = 1.0;
            let mut vof = Vof::new(&g, &hm, &m, props, ctrl)?;

            let a0: Vec<Scalar> = (0..hm.n_cells)
                .map(|c| if hm.c[c].x < 0.25 { 1.0 } else { 0.0 })
                .collect();
            g.write(&mut vof.alpha_mut().f, &a0)?;
            // Phase 1 flows in at x = 0 and everything leaves at x = 1.
            set_scalar_bcs(
                &g,
                vof.alpha_mut(),
                &hm,
                &[Some(1.0), None, None, None, None, None],
            )?;

            let stream = |_p: Vec3| Vec3::new(1.0, 0.0, 0.0);
            write_analytic_flux(&g, vof.phi_mut(), &hm, stream)?;
            vof.initialise(&g)?;

            let dt = 0.25 * h; // |u| = 1, so this is Co = 0.25
            for _ in 0..20 {
                vof.solve_alpha(&g, dt)?;
            }
            let early = width(&g.download(&vof.alpha().f)?);

            for _ in 20..200 {
                vof.solve_alpha(&g, dt)?;
            }
            let a = g.download(&vof.alpha().f)?;
            let late = width(&a);
            let (lo, hi) = vof.alpha_bounds(&g)?;
            Ok((early, late, lo, hi))
        };

        let (e_off, l_off, lo_off, hi_off) = run(0.0)?;
        let (e_on, l_on, lo_on, hi_on) = run(1.0)?;

        println!(
            "  interface width in cells: cAlpha 0  {e_off} -> {l_off} ; \
             cAlpha 1  {e_on} -> {l_on}"
        );

        // Bounded either way - FCT does not need the compression term for
        // that, and the two facts are independent.
        for (lo, hi) in [(lo_off, hi_off), (lo_on, hi_on)] {
            assert!(lo >= -1e-12 && hi <= 1.0 + 1e-12, "alpha left [0,1]");
        }

        assert!(
            l_on <= e_on,
            "the compressed interface widened from {e_on} to {l_on} cells"
        );
        assert!(
            l_on < l_off,
            "compression made no difference: {l_on} cells against {l_off}"
        );

        Ok(())
    }

    // ----------------------------------------------------------------------
    //  SPEC-LIT §20.4: the curvature, against the one shape whose curvature
    //  is known in closed form
    // ----------------------------------------------------------------------

    /// A smooth radial `alpha` profile whose level sets are circles (2-D) or
    /// spheres (3-D).
    ///
    /// `alpha = (1 - tanh((r - R)/w))/2` decreases with `r`, so
    /// `grad(alpha)` points inward, `n_hat = -r_hat`, and
    ///
    /// ```text
    /// kappa = -div(n_hat) = +div(r_hat) = (d - 1)/r
    /// ```
    ///
    /// which is `1/R` on the interface in two dimensions and `2/R` in three -
    /// exactly the two numbers §22 asks the Laplace jump to reproduce.
    fn radial_alpha(hm: &HostMesh, centre: Vec3, r: Scalar, w: Scalar, two_d: bool) -> Vec<Scalar> {
        (0..hm.n_cells)
            .map(|c| {
                let d = hm.c[c] - centre;
                let rr = if two_d {
                    (d.x * d.x + d.y * d.y).sqrt()
                } else {
                    d.mag()
                };
                0.5 * (1.0 - ((rr - r) / w).tanh())
            })
            .collect()
    }

    /// The root-mean-square relative error of the computed curvature over the
    /// interface band, against the ANALYTIC curvature of the level set that
    /// passes through each cell.
    ///
    /// `(d - 1)/r_c`, not `(d - 1)/R`: the level sets of a radial profile are
    /// concentric circles or spheres and each has its own curvature, so
    /// comparing a cell at `r_c` with the value at `R` would charge the scheme
    /// for a difference that is exactly right.
    fn curvature_rms_error(
        gpu: &Gpu,
        vof: &mut Vof<'_>,
        hm: &HostMesh,
        centre: Vec3,
        two_d: bool,
    ) -> Result<(Scalar, usize)> {
        vof.update_body_force(gpu)?;
        let k = gpu.download(&vof.curvature().f)?;
        let a = gpu.download(&vof.alpha().f)?;

        let d_minus_1: Scalar = if two_d { 1.0 } else { 2.0 };

        let mut sum = 0.0 as Scalar;
        let mut n = 0usize;
        for c in 0..hm.n_cells {
            if a[c] <= 0.3 || a[c] >= 0.7 {
                continue;
            }
            let d = hm.c[c] - centre;
            let r = if two_d {
                (d.x * d.x + d.y * d.y).sqrt()
            } else {
                d.mag()
            };
            let exact = d_minus_1 / r;
            let e = (k[c] - exact) / exact;
            sum += e * e;
            n += 1;
        }

        if n == 0 {
            return Ok((Scalar::INFINITY, 0));
        }
        Ok(((sum / n as Scalar).sqrt(), n))
    }

    /// A circular interface, refined, against `kappa = 1/r`.
    ///
    /// A CONVERGENCE test rather than a threshold. The curvature is a second
    /// derivative of a field that is nearly a step, and at a fixed number of
    /// cells across the interface its relative error is a few per cent whatever
    /// the mesh - so a threshold test either passes a broken operator on a fine
    /// mesh or fails a correct one on a coarse mesh. Holding the interface
    /// thickness FIXED IN METRES and refining measures the thing that is
    /// actually true: the error is `O((h/w)^2)` and falls by four when the mesh
    /// is halved.
    ///
    /// This is the test that caught the face-gradient variant `cuda/vof.cu`
    /// documents and rejects; it showed up there as an error that did not fall
    /// at all.
    #[test]
    fn the_curvature_of_a_circular_interface_converges_to_one_over_r() -> Result<()> {
        let Some(g) = gpu() else { return Ok(()) };

        // Fixed in metres, so refining the mesh really does resolve it better.
        let w = 0.06 as Scalar;
        let r = 0.25 as Scalar;

        let measure = |nx: usize| -> Result<(Scalar, usize)> {
            let h = 1.0 / nx as Scalar;
            let hm = boxed(
                [nx, nx, 1],
                Vec3::new(h, h, h),
                [
                    PatchKind::Wall,
                    PatchKind::Wall,
                    PatchKind::Wall,
                    PatchKind::Wall,
                    PatchKind::Empty,
                    PatchKind::Empty,
                ],
            );
            let m = GpuMesh::upload(&g, &hm)?;

            let (props, ctrl) = advection_only(0.0);
            let mut vof = Vof::new(&g, &hm, &m, props, ctrl)?;

            let centre = Vec3::new(0.5, 0.5, 0.5 * h);
            let a = radial_alpha(&hm, centre, r, w, true);
            g.write(&mut vof.alpha_mut().f, &a)?;
            set_scalar_bcs(&g, vof.alpha_mut(), &hm, &[None; 6])?;
            vof.initialise(&g)?;

            curvature_rms_error(&g, &mut vof, &hm, centre, true)
        };

        let (e64, n64) = measure(64)?;
        let (e128, n128) = measure(128)?;
        let order = (e64 / e128).log2();

        println!(
            "  2-D curvature: rms error {e64:.5} on 64^2 ({n64} band cells) ->              {e128:.5} on 128^2 ({n128}); observed order {order:.2}"
        );

        assert!(n64 > 100 && n128 > 200, "the interface band is too thin to measure");
        assert!(order > 1.5, "the curvature is converging at order {order:.2}");
        assert!(e128 < 0.02, "the fine-mesh curvature error is {e128:.4}");

        Ok(())
    }

    /// The same in three dimensions, where the answer is `2/r`.
    ///
    /// Both numbers matter: §22 asks the Laplace jump to come out `sigma/R` in
    /// 2-D and `2 sigma/R` in 3-D, and the factor of two between them is
    /// entirely this curvature.
    #[test]
    fn the_curvature_of_a_spherical_interface_converges_to_two_over_r() -> Result<()> {
        let Some(g) = gpu() else { return Ok(()) };

        let w = 0.1 as Scalar;
        let r = 0.25 as Scalar;

        let measure = |nx: usize| -> Result<(Scalar, usize)> {
            let h = 1.0 / nx as Scalar;
            let hm = boxed([nx, nx, nx], Vec3::new(h, h, h), [PatchKind::Wall; 6]);
            let m = GpuMesh::upload(&g, &hm)?;

            let (props, ctrl) = advection_only(0.0);
            let mut vof = Vof::new(&g, &hm, &m, props, ctrl)?;

            let centre = Vec3::new(0.5, 0.5, 0.5);
            let a = radial_alpha(&hm, centre, r, w, false);
            g.write(&mut vof.alpha_mut().f, &a)?;
            set_scalar_bcs(&g, vof.alpha_mut(), &hm, &[None; 6])?;
            vof.initialise(&g)?;

            curvature_rms_error(&g, &mut vof, &hm, centre, false)
        };

        let (e32, n32) = measure(32)?;
        let (e64, n64) = measure(64)?;
        let order = (e32 / e64).log2();

        println!(
            "  3-D curvature: rms error {e32:.5} on 32^3 ({n32} band cells) ->              {e64:.5} on 64^3 ({n64}); observed order {order:.2}"
        );

        assert!(n32 > 100 && n64 > 500, "the interface band is too thin to measure");
        assert!(order > 1.5, "the curvature is converging at order {order:.2}");
        assert!(e64 < 0.02, "the fine-mesh curvature error is {e64:.4}");

        Ok(())
    }

    // ----------------------------------------------------------------------
    //  SPEC-LIT §22: "CSF - a static drop in zero gravity - spurious currents
    //  small and bounded; Laplace pressure sigma/R (2-D)"
    // ----------------------------------------------------------------------

    /// A drop at rest, held together by surface tension alone.
    ///
    /// With `g = 0` the only body force is the CSF, and at equilibrium the
    /// face balance is
    ///
    /// ```text
    /// |Sf| snGrad(p_rgh) = sigma kappa_f |Sf| snGrad(alpha)
    /// ```
    ///
    /// so where `kappa` is constant along the interface, `p_rgh = sigma kappa
    /// alpha + const` solves it EXACTLY and the flux correction is identically
    /// zero. Two things follow, and both are checked:
    ///
    /// * the pressure difference between the two phases is `sigma kappa`,
    ///   which is `sigma/R` in 2-D;
    /// * whatever velocity does appear comes from the VARIATION of `kappa`
    ///   along the interface - the classic spurious interface current - and
    ///   must stay bounded rather than growing step on step.
    #[test]
    fn a_static_drop_has_the_laplace_pressure_jump() -> Result<()> {
        let Some(g) = gpu() else { return Ok(()) };

        let nx = 64usize;
        let h = 1.0 / nx as Scalar;
        let hm = boxed(
            [nx, nx, 1],
            Vec3::new(h, h, h),
            [
                PatchKind::Wall,
                PatchKind::Wall,
                PatchKind::Wall,
                PatchKind::Wall,
                PatchKind::Empty,
                PatchKind::Empty,
            ],
        );
        let m = GpuMesh::upload(&g, &hm)?;

        let sigma = 1.0 as Scalar;
        let r = 0.2 as Scalar;

        let props = VofProperties {
            rho1: 1000.0,
            rho2: 1.0,
            mu1: 0.1,
            mu2: 0.01,
            sigma,
            // Zero gravity: the drop is held together by nothing else.
            g: Vec3::ZERO,
            c_alpha: 1.0,
        };
        let ctrl = VofControls {
            delta_t: 1e-4,
            n_correctors: 2,
            u_solver: tight(),
            p_solver: tight(),
            sn_grad: fv::SnGradScheme::Uncorrected,
            report_continuity: true,
            ..VofControls::default()
        };

        let mut vof = Vof::new(&g, &hm, &m, props, ctrl)?;

        let a = radial_alpha(&hm, Vec3::new(0.5, 0.5, 0.5 * h), r, 1.5 * h, true);
        g.write(&mut vof.alpha_mut().f, &a)?;
        set_scalar_bcs(&g, vof.alpha_mut(), &hm, &[None; 6])?;
        set_velocity_bcs(&g, vof.u_mut(), &hm, &[Some(Vec3::ZERO); 6])?;
        set_scalar_bcs(&g, vof.p_rgh_mut(), &hm, &[None; 6])?;
        vof.initialise(&g)?;

        // Sampled at three times in geometric progression, so the SHAPE of
        // the growth can be read off and not just its size.
        let dt = 1e-4 as Scalar;
        let mut u = [0.0 as Scalar; 3];
        for step in 1..=120 {
            vof.step(&g, dt)?;
            match step {
                30 => u[0] = max_speed(&g, &vof)?,
                60 => u[1] = max_speed(&g, &vof)?,
                120 => u[2] = max_speed(&g, &vof)?,
                _ => {}
            }
        }
        let u_end = u[2];

        // The pressure jump. p_rgh IS p here, because g = 0.
        let p = g.download(&vof.p_rgh().f)?;
        let a = g.download(&vof.alpha().f)?;
        let mut inside = (0.0 as Scalar, 0usize);
        let mut outside = (0.0 as Scalar, 0usize);
        for c in 0..hm.n_cells {
            if a[c] > 0.99 {
                inside.0 += p[c];
                inside.1 += 1;
            } else if a[c] < 0.01 {
                outside.0 += p[c];
                outside.1 += 1;
            }
        }
        assert!(inside.1 > 50 && outside.1 > 50);
        let jump = inside.0 / inside.1 as Scalar - outside.0 / outside.1 as Scalar;
        let exact = sigma / r;

        // The capillary number the spurious current amounts to.
        let ca = props.mu1 * u_end / sigma;

        // Growing no faster than linearly is the claim, and it is the right
        // one. The residual force is the variation of `kappa` along the
        // interface, which the mesh fixes once and for all, so it drives a
        // CONSTANT acceleration until viscosity balances it - and this drop's
        // viscous time `rho h^2/mu` is two seconds against the twelve
        // milliseconds simulated. Doubling the interval therefore doubles the
        // speed, and what would be a failure is growth that ACCELERATES: a
        // CSF feeding its own velocity field. So the two successive ratios are
        // compared with each other, not with 2.
        let r1 = u[1] / u[0].max(Scalar::MIN_POSITIVE);
        let r2 = u[2] / u[1].max(Scalar::MIN_POSITIVE);

        println!(
            "  static drop: Laplace jump {jump:.4} against sigma/R = {exact:.4} \
             ({:.2}%);  spurious |U| {:e} -> {:e} -> {:e}, Ca = {ca:e}, \
             growth ratios {r1:.3} then {r2:.3}",
            100.0 * (jump - exact) / exact,
            u[0],
            u[1],
            u[2]
        );

        assert!(
            ((jump - exact) / exact).abs() < 0.10,
            "the Laplace jump is {jump}, and sigma/R is {exact}"
        );
        assert!(ca < 1e-2, "the spurious current is not small: Ca = {ca:e}");
        assert!(
            r2 <= 1.05 * r1,
            "the spurious current is accelerating: ratios {r1} then {r2}"
        );

        Ok(())
    }

    // ----------------------------------------------------------------------
    //  SPEC-LIT §22: "p_rgh - two stratified fluids, sealed, at rest - stays
    //  at rest to round-off"
    //
    //  §20.5: "That test fails immediately if p_rgh is not used, and it is the
    //  one test that proves this section is right."
    // ----------------------------------------------------------------------

    /// Heavy fluid under light fluid in a sealed tank, released from rest.
    ///
    /// The discrete statement being tested is a face identity. Within either
    /// layer `rho` is uniform, so `snGrad(rho) = 0` and `p_rgh` is constant.
    /// Across the interface face, with `P` below and `N` above,
    ///
    /// ```text
    /// p_rgh_P - p_rgh_N = (rho_P - rho_N) g z_f
    /// ```
    ///
    /// and the gravity flux `-(g·x)_f (rho_N - rho_P) Δ_f |Sf|` is exactly
    /// `|Sf| snGrad(p_rgh)` with the SAME `Δ_f |Sf|`. The two cancel to the
    /// last bit, so the flux correction is zero and nothing moves.
    ///
    /// Solve for `p` instead of `p_rgh` and both terms are the size of
    /// `rho g H`, their difference is the size of the physics, and the
    /// cancellation is only as good as the ratio of the two - which is what
    /// §20.5 is about.
    #[test]
    fn two_sealed_stratified_fluids_stay_at_rest() -> Result<()> {
        let Some(g) = gpu() else { return Ok(()) };

        let n = 24usize;
        let h = 1.0 / n as Scalar;
        let hm = boxed(
            [n, 1, n],
            Vec3::new(h, h, h),
            [
                PatchKind::Wall,
                PatchKind::Wall,
                PatchKind::Empty,
                PatchKind::Empty,
                PatchKind::Wall,
                PatchKind::Wall,
            ],
        );
        let m = GpuMesh::upload(&g, &hm)?;

        let props = VofProperties {
            rho1: 1000.0,
            rho2: 1.0,
            mu1: 1.002e-3,
            mu2: 1.8e-5,
            // No surface tension: a flat interface has zero curvature, and
            // this test is about §20.5 alone.
            sigma: 0.0,
            g: Vec3::new(0.0, 0.0, -9.81),
            c_alpha: 1.0,
        };
        let ctrl = VofControls {
            delta_t: 1e-3,
            n_correctors: 3,
            u_solver: tight(),
            p_solver: tight(),
            sn_grad: fv::SnGradScheme::Uncorrected,
            ..VofControls::default()
        };

        let mut vof = Vof::new(&g, &hm, &m, props, ctrl)?;

        // The interface sits exactly on a face, so the initial alpha is a
        // clean 1/0 split with no partly-filled cell anywhere.
        let a: Vec<Scalar> = (0..hm.n_cells)
            .map(|c| if hm.c[c].z < 0.5 { 1.0 } else { 0.0 })
            .collect();
        g.write(&mut vof.alpha_mut().f, &a)?;
        set_scalar_bcs(&g, vof.alpha_mut(), &hm, &[None; 6])?;
        set_velocity_bcs(&g, vof.u_mut(), &hm, &[Some(Vec3::ZERO); 6])?;
        set_scalar_bcs(&g, vof.p_rgh_mut(), &hm, &[None; 6])?;
        vof.initialise(&g)?;

        assert!(
            vof.pressure_is_pinned(),
            "a sealed tank has no pressure Dirichlet and must be pinned"
        );

        let dt = 1e-3 as Scalar;
        let mut worst = 0.0 as Scalar;
        let mut last = 0.0 as Scalar;
        for _ in 0..20 {
            let perf = vof.step(&g, dt)?;
            last = max_speed(&g, &vof)?;
            worst = worst.max(last);
            assert!(perf.continuity_error.is_finite());
        }

        let (lo, hi) = vof.alpha_bounds(&g)?;
        println!(
            "  sealed stratified tank after 20 steps: max |U| {last:e} \
             (worst over the run {worst:e}), alpha in [{lo}, {hi}]"
        );

        // The interface has not moved. `alpha` is still 0 and 1 to within
        // the flux that is left, which is the pressure solver's residual and
        // not a physical motion - so the bound is stated against that residual
        // rather than as an exact equality.
        assert!(
            lo > -1e-9 && hi < 1.0 + 1e-9,
            "the interface moved: alpha in [{lo:e}, {hi}]"
        );

        // The velocity scale of the problem: a gravity wave on a tank of this
        // depth, sqrt(g H). Anything the solver leaves behind has to be
        // negligible NEXT TO THE PHYSICS, which is what that compares against;
        // `dt` does not enter, because the claim is about the state and not
        // about one step.
        let scale = (9.81 as Scalar).sqrt();
        assert!(
            last < 1e-8 * scale,
            "the tank did not stay at rest: |U| = {last:e} against the \
             gravity-wave scale {scale:e}"
        );

        // And it is not growing: the last step is no worse than the worst of
        // the run, which is the statement that the balance is a fixed point
        // rather than a slow leak.
        assert!(last <= worst + 1e-30);

        Ok(())
    }

    // ----------------------------------------------------------------------
    //  SPEC-LIT §20.3: the mass flux and the density must be consistent
    // ----------------------------------------------------------------------

    /// `(rho - rho0) V/dt + Σ_f (±rho_phi_f) = 0`, measured.
    ///
    /// This is the identity §20.3 exists to guarantee, and it holds only
    /// because `rho_phi` is built from the accumulated LIMITED `alpha` flux
    /// rather than from re-interpolating `rho`. Re-interpolating would leave a
    /// residual proportional to the interpolation error at the interface, and
    /// the momentum equation would answer it with velocity.
    #[test]
    fn the_mass_flux_is_consistent_with_the_density_it_advects() -> Result<()> {
        let Some(g) = gpu() else { return Ok(()) };

        let nx = 40usize;
        let h = 1.0 / nx as Scalar;
        let hm = boxed(
            [nx, nx, 1],
            Vec3::new(h, h, h),
            [
                PatchKind::Wall,
                PatchKind::Wall,
                PatchKind::Wall,
                PatchKind::Wall,
                PatchKind::Empty,
                PatchKind::Empty,
            ],
        );
        let m = GpuMesh::upload(&g, &hm)?;

        let props = VofProperties {
            rho1: 1000.0,
            rho2: 1.0,
            mu1: 0.0,
            mu2: 0.0,
            sigma: 0.0,
            g: Vec3::ZERO,
            c_alpha: 1.0,
        };
        let ctrl = VofControls {
            max_alpha_co: 0.5,
            report_continuity: false,
            ..VofControls::default()
        };
        let mut vof = Vof::new(&g, &hm, &m, props, ctrl)?;

        // A blob, off-centre, so the rotation actually carries it.
        let a: Vec<Scalar> = (0..hm.n_cells)
            .map(|c| {
                let d = hm.c[c] - Vec3::new(0.5, 0.7, 0.0);
                if (d.x * d.x + d.y * d.y).sqrt() < 0.15 {
                    1.0
                } else {
                    0.0
                }
            })
            .collect();
        g.write(&mut vof.alpha_mut().f, &a)?;
        set_scalar_bcs(&g, vof.alpha_mut(), &hm, &[None; 6])?;

        let rot = |p: Vec3| Vec3::new(-(p.y - 0.5), p.x - 0.5, 0.0);
        write_analytic_flux(&g, vof.phi_mut(), &hm, rot)?;
        vof.initialise(&g)?;

        let dt = 0.2 * h;
        let rho_before = g.download(&vof.rho().f)?;
        vof.solve_alpha(&g, dt)?;
        vof.update_properties(&g)?;
        vof.update_rho_phi(&g)?;
        let rho_after = g.download(&vof.rho().f)?;

        let rp = g.download(&vof.rho_phi().f)?;
        let brp = g.download(&vof.rho_phi().bf)?;

        let mut worst = 0.0 as Scalar;
        let mut scale = 0.0 as Scalar;
        let mut div = vec![0.0 as Scalar; hm.n_cells];
        for f in 0..hm.n_internal_faces {
            div[hm.owner[f] as usize] += rp[f];
            div[hm.neighbour[f] as usize] -= rp[f];
        }
        for b in 0..hm.n_boundary_faces {
            if hm.b_kind[b] != PatchKind::Empty as Label {
                div[hm.b_face_cells[b] as usize] += brp[b];
            }
        }
        for c in 0..hm.n_cells {
            let ddt = (rho_after[c] - rho_before[c]) * hm.v[c] / dt;
            worst = worst.max((ddt + div[c]).abs());
            scale = scale.max(ddt.abs().max(div[c].abs()));
        }

        println!("  mass consistency: worst residual {worst:e} against a scale of {scale:e}");
        assert!(scale > 0.0, "nothing moved, so nothing was tested");
        assert!(
            worst < 1e-10 * scale.max(1.0),
            "d(rho)/dt + div(rho phi) = {worst:e}, which is not round-off"
        );

        Ok(())
    }

    // ----------------------------------------------------------------------
    //  SPEC-LIT §22: "Dam break - Martin & Moyce, Phil. Trans. R. Soc. A 244
    //  (1952) 312 - surge front position vs time"
    //
    //  `#[ignore]`d: it runs a thousand steps of the full two-phase solver and
    //  takes a minute. Run it with
    //
    //      cargo test --release --lib -- --ignored --nocapture dam_break
    // ----------------------------------------------------------------------

    /// A column of water of width `a` and height `2a`, released.
    ///
    /// **What this asserts, and what it does not.** Martin & Moyce photographed
    /// the surge front of exactly this experiment and tabulated `Z = z/a`
    /// against a dimensionless time. Their table is NOT transcribed here,
    /// because nobody working on this file has the paper in front of them, and
    /// a table of numbers attributed to a 1952 experiment and actually
    /// remembered would be worse than no table at all. What the test prints is
    /// the solver's own `Z(T)` in those variables, so a reader who does have
    /// the paper can lay one on the other.
    ///
    /// What it *asserts* is three things that are true independently of any
    /// experiment:
    ///
    /// 1. `alpha` stays in `[0, 1]` for every one of the thousand steps;
    /// 2. the water volume is conserved exactly, for as long as no water has
    ///    reached the open top - after that, water leaving is physics;
    /// 3. the surge front never outruns `a + 2 sqrt(g h0) t`. That is the
    ///    Ritter (1892) characteristic speed of the frictionless shallow-water
    ///    dam break, an analytic upper bound on how fast a released column of
    ///    depth `h0` can possibly spread, and a front that beats it is a front
    ///    the numerics invented.
    #[test]
    #[ignore]
    fn a_dam_break_surge_front_stays_under_the_ritter_bound() -> Result<()> {
        let Some(g) = gpu() else { return Ok(()) };

        // Martin & Moyce's geometry: width a, height n^2 a with n^2 = 2, in a
        // tank 5a x 3a. `a` is a free choice - their variables are
        // dimensionless - and 0.05 m is laboratory scale.
        let a = 0.05 as Scalar;
        let h0 = 2.0 * a;
        let gmag = 9.81 as Scalar;

        let (nx, ny) = (150usize, 90usize);
        let h = 5.0 * a / nx as Scalar;
        let hm = boxed(
            [nx, ny, 1],
            Vec3::new(h, h, h),
            [
                PatchKind::Wall,
                PatchKind::Wall,
                PatchKind::Wall,
                // The open top. Everything else is solid.
                PatchKind::Generic,
                PatchKind::Empty,
                PatchKind::Empty,
            ],
        );
        let m = GpuMesh::upload(&g, &hm)?;

        let props = VofProperties {
            rho1: 998.2,
            rho2: 1.2,
            mu1: 1.002e-3,
            mu2: 1.8e-5,
            sigma: 0.0728,
            // The resolved plane of this 2-D block is x-y, so down is -y.
            g: Vec3::new(0.0, -gmag, 0.0),
            c_alpha: 1.0,
        };
        let ctrl = VofControls {
            delta_t: 2e-4,
            n_correctors: 3,
            u_solver: SolverControls {
                tolerance: 1e-8,
                rel_tol: 0.0,
                max_iter: 200,
                ..SolverControls::default()
            },
            p_solver: SolverControls {
                tolerance: 1e-9,
                rel_tol: 1e-3,
                max_iter: 2000,
                precon: Preconditioner::Dic,
                ..SolverControls::default()
            },
            sn_grad: fv::SnGradScheme::Uncorrected,
            // Measured, because the bound on the alpha excursion below is
            // derived from it.
            report_continuity: true,
            ..VofControls::default()
        };
        let mut vof = Vof::new(&g, &hm, &m, props, ctrl)?;

        let a0: Vec<Scalar> = (0..hm.n_cells)
            .map(|c| {
                if hm.c[c].x < a && hm.c[c].y < h0 {
                    1.0
                } else {
                    0.0
                }
            })
            .collect();
        g.write(&mut vof.alpha_mut().f, &a0)?;
        // Walls zero-gradient; the open top takes air back in if it draws any.
        set_scalar_bcs(&g, vof.alpha_mut(), &hm, &[None; 6])?;
        // Solid everywhere but the top, where the velocity is whatever the
        // prescribed pressure asks for.
        set_velocity_bcs(
            &g,
            vof.u_mut(),
            &hm,
            &[
                Some(Vec3::ZERO),
                Some(Vec3::ZERO),
                Some(Vec3::ZERO),
                None,
                None,
                None,
            ],
        )?;
        // The one pressure Dirichlet in the case: the level the tank is open
        // to. Without it the system is singular AND water could not leave.
        set_scalar_bcs(
            &g,
            vof.p_rgh_mut(),
            &hm,
            &[None, None, None, Some(0.0), None, None],
        )?;
        vof.initialise(&g)?;

        assert!(
            !vof.pressure_is_pinned(),
            "the open top should give p_rgh a Dirichlet"
        );

        // The bottom row of cells, in ascending x: the surge front runs along
        // the floor, and reading it off the whole field would give the outline
        // of the collapsing bulk instead.
        let mut floor: Vec<usize> = (0..hm.n_cells).filter(|c| hm.c[*c].y < h).collect();
        floor.sort_by(|p, q| hm.c[*p].x.total_cmp(&hm.c[*q].x));
        assert_eq!(floor.len(), nx);

        let front = |alpha: &[Scalar]| -> Scalar {
            let mut last = 0usize;
            for (i, c) in floor.iter().enumerate() {
                if alpha[*c] >= 0.5 {
                    last = i;
                }
            }
            // Linear between the last wet cell and the first dry one, so the
            // answer moves smoothly rather than in cell-sized jumps.
            let a0 = alpha[floor[last]];
            let a1 = if last + 1 < floor.len() {
                alpha[floor[last + 1]]
            } else {
                a0
            };
            let f = if (a0 - a1).abs() > 1e-30 {
                ((a0 - 0.5) / (a0 - a1)).clamp(0.0, 1.0)
            } else {
                0.0
            };
            hm.c[floor[last]].x + f * h
        };

        let v0 = vof.phase_volume(&g)?;
        let dt = 2e-4 as Scalar;
        let steps = 1000; // t = 0.2 s, T = 2.8
        let scale_t = (gmag / a).sqrt();

        println!("\n  dam break, Martin & Moyce variables: Z = z/a, T = t sqrt(g/a)");
        println!("  a = {a} m, column height {h0} m, tank {} x {} m", 5.0 * a, 3.0 * a);
        println!("  {:>8} {:>8} {:>10} {:>8} {:>10}", "t", "T", "z", "Z", "Ritter Z");

        let mut worst_lo = 0.0 as Scalar;
        let mut worst_hi = 0.0 as Scalar;
        let mut worst_cont = 0.0 as Scalar;

        for step in 1..=steps {
            let perf = vof.step(&g, dt)?;
            worst_cont = worst_cont.max(perf.continuity_error);
            let t = step as Scalar * dt;

            let (lo, hi) = vof.alpha_bounds(&g)?;
            worst_lo = worst_lo.max(-lo);
            worst_hi = worst_hi.max(hi - 1.0);

            let alpha = g.download(&vof.alpha().f)?;
            let z = front(&alpha);

            // Ritter (1892): the front of a frictionless shallow-water dam
            // break of depth h0 moves at 2 sqrt(g h0). Starting from the
            // column's own edge, that is the furthest the surge can be.
            let ritter = a + 2.0 * (gmag * h0).sqrt() * t;
            assert!(
                z <= ritter + h,
                "the surge front is at {z} m at t = {t} s, past the Ritter \
                 bound {ritter} m"
            );

            if step % 100 == 0 {
                println!(
                    "  {:>8.4} {:>8.3} {:>10.5} {:>8.3} {:>10.3}",
                    t,
                    t * scale_t,
                    z,
                    z / a,
                    ritter / a
                );
            }
        }

        let v1 = vof.phase_volume(&g)?;
        let z_end = front(&g.download(&vof.alpha().f)?);

        // The bound one step of a non-solenoidal flux can push a FULL cell
        // past one by, times an allowance for the few steps it takes before
        // the interface moves on and the excess is advected away.
        //
        // `10` is that allowance, and it is the one fitted number here; the
        // rest is arithmetic. It is worth having because the bound is
        // otherwise TIGHT - it tracks the measurement across five orders of
        // magnitude of solver tolerance - and a test that tracks is worth more
        // than a threshold that does not.
        let v_cell = hm.v.iter().fold(Scalar::MAX, |x, v| x.min(*v));
        let bound = 10.0 * worst_cont * dt / v_cell;

        println!(
            "  alpha excursions: {:e} below zero, {:e} above one",
            worst_lo, worst_hi
        );
        println!(
            "  worst continuity error {:e} m^3/s, so one step can move a full \
             cell by {:e} and the bound is {:e}",
            worst_cont,
            worst_cont * dt / v_cell,
            bound
        );
        println!(
            "  water volume {v0:e} -> {v1:e} (relative change {:e})",
            (v1 - v0) / v0
        );

        // Below zero, this really is round-off: an excursion downward
        // needs a cell to give away more than it has, and the upwind flux out
        // of a cell is proportional to what is in it.
        assert!(worst_lo < 1e-12, "alpha went below zero by {worst_lo:e}");

        // Above one it is NOT round-off, and the module header says why:
        // this flux satisfies continuity to the pressure solver's stopping
        // criterion rather than to machine epsilon, and the low-order upwind
        // step is bounded only for a flux that satisfies it exactly.
        //
        // So what is asserted is that the excursion is EXPLAINED BY THE FLUX
        // rather than that it is zero, which would be a claim about a flux
        // this case has not got. The two ends of the scaling, measured:
        //
        //     p_rgh to 1e-9, relTol 1e-3   ->  max(alpha) - 1 = 4.3e-07
        //     p_rgh to 1e-13, relTol 0     ->  max(alpha) - 1 = 4.0e-12
        //
        // five orders of magnitude, tracking the solver and not the scheme.
        // The exact `[0, 1]` statement of §20.2 is tested where it can be:
        // on the Zalesak disc, whose flux is analytic and solenoidal to
        // round-off, and which comes back after a full revolution with
        // `max(alpha)` exactly 1.
        assert!(
            worst_hi <= bound,
            "alpha went above one by {worst_hi:e}, past the {bound:e} that a \
             continuity error of {worst_cont:e} explains"
        );
        // And small enough to be physically irrelevant either way: the mixture
        // density is positive and right to a part in a hundred thousand.
        assert!(worst_hi < 1e-5, "alpha went above one by {worst_hi:e}");

        // Over these 0.2 s the surge is still crossing the floor and no water
        // has reached the open top, so the volume is conserved exactly.
        assert!(
            ((v1 - v0) / v0).abs() < 1e-10,
            "water volume changed by {:e} before any of it could leave",
            (v1 - v0) / v0
        );

        // And it really did collapse: a column that had not moved would still
        // read Z = 1.
        assert!(z_end > 3.0 * a, "the column did not collapse: z = {z_end}");

        Ok(())
    }

    // ----------------------------------------------------------------------
    //  Configuration
    // ----------------------------------------------------------------------

    #[test]
    fn a_negative_density_is_refused() {
        let p = VofProperties { rho2: -1.0, ..VofProperties::default() };
        let e = p.validate().unwrap_err().to_string();
        assert!(e.contains("rho2"), "{e}");
    }

    #[test]
    fn an_alpha_courant_number_above_one_is_refused() {
        let c = VofControls { max_alpha_co: 1.5, ..VofControls::default() };
        let e = c.validate().unwrap_err().to_string();
        assert!(e.contains("maxAlphaCo"), "{e}");
    }

    #[test]
    fn a_pressure_corrector_count_of_zero_is_refused() {
        let c = VofControls { n_correctors: 0, ..VofControls::default() };
        let e = c.validate().unwrap_err().to_string();
        assert!(e.contains("nCorrectors"), "{e}");
    }
}
