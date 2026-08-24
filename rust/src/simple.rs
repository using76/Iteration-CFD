// meteor-cfd - Copyright (c) 2026 주식회사 메테오시뮬레이션 (Meteo Simulation Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.

//! SIMPLE: the pressure-velocity coupling.
//!
//! Written from:
//!   S. V. Patankar, D. B. Spalding, *Int. J. Heat Mass Transfer* 15 (1972)
//!     1787-1806, and S. V. Patankar, *Numerical Heat Transfer and Fluid
//!     Flow*, Hemisphere (1980), ch. 6 - the algorithm, and §6.7 for the
//!     relaxation pair
//!   C. M. Rhie, W. L. Chow, *AIAA J.* 21 (1983) 1525-1532 - the face flux
//!   J. P. Van Doormaal, G. D. Raithby, *Numer. Heat Transfer* 7 (1984) 147 -
//!     SIMPLEC
//!   H. Jasak, PhD thesis, Imperial College (1996), §3.4.3 - iterating the
//!     explicit non-orthogonal correction
//!   R. I. Issa, *J. Comput. Phys.* 62 (1986) 40-65, and Ferziger & Perić
//!     §7.4 - PISO, and the outer-corrector wrapper that is PIMPLE
//!   ofgpu `SPEC-LIT.md` §5.2, §5.3, §8.5 and §14
//! No GPL-licensed source was consulted.
//!
//! # One algorithm, two switches
//!
//! SPEC-LIT §14 asks for ONE loop rather than three algorithms, and this is
//! it:
//!
//! ```text
//! for outer in 1..=nOuterCorrectors:                 <- PIMPLE
//!     final = transient && outer == nOuterCorrectors
//!     assemble and solve momentum, relaxed unless final
//!     for corrector in 1..=nCorrectors:              <- PISO
//!         rAU, HbyA, phiHbyA        <- H RE-EVALUATED from the latest U
//!         for nc in 0..=nNonOrthogonalCorrectors:    <- Jasak §3.4.3
//!             solve laplacian(rAU_f, p) = div(phiHbyA)
//!         phi, U <- corrected by the p that satisfies continuity
//!     relax p unless final
//! ```
//!
//! * `nOuterCorrectors = 1`, `nCorrectors = 1`, a steady `ddt`   -> SIMPLE
//! * `nOuterCorrectors = 1`, `nCorrectors >= 2`, a transient `ddt` -> PISO
//! * `nOuterCorrectors >= 2`                                     -> PIMPLE
//!
//! The distinction that makes the middle line worth anything is that `H` is
//! rebuilt at the top of every PISO corrector, from the velocity the previous
//! corrector produced. A loop that computes `HbyA` once and only repeats the
//! pressure solve is doing NON-ORTHOGONAL correctors - the inner loop - and
//! will not reach the transient accuracy PISO exists for. The two loops are
//! nested here precisely because they are different things.
//!
//! Relaxation is switched off on the final outer corrector of a TRANSIENT run
//! only. In a steady run relaxation is what replaces the time derivative
//! (Patankar §6.7); switching it off there because "this is the last outer
//! iteration" - and in steady SIMPLE every iteration is the last - would
//! remove the only thing holding the fixed-point iteration together.
//!
//! # One iteration
//!
//! ```text
//! b_f          <- face buoyancy flux from T                       (§9)
//! forceFlux    <- b_f·Sf - |Sf|·snGrad(p)                         (§5.1)
//! solve        <- momentum, relaxed by alpha_U, three components  (§5.2)
//! rAU, HbyA    <- from the matrix just solved                     (§5.1)
//! phi_HbyA     <- interpolate(HbyA)·Sf + rAU_f·(b_f·Sf)           (§5.1)
//! repeat nNonOrth+1 times:
//!     solve    <- laplacian(rAU_f, p) = div(phi_HbyA)
//! phi          <- phi_HbyA - rAU_f·|Sf|·snGrad(p)
//! U            <- HbyA + reconstruct(rAU_f·forceFlux)
//! p            <- p_old + alpha_p·(p - p_old)                     (§5.2)
//! ```
//!
//! The order is SPEC-LIT §5.2's, including the placement of the pressure
//! relaxation *after* the flux correction: the flux has to be corrected with
//! the pressure that actually satisfies continuity, and only the field carried
//! into the next momentum predictor is relaxed.
//!
//! # The pressure equation is a Poisson equation and nothing else
//!
//! `laplacian(rAU_f, p) = div(phi_HbyA)` is assembled with the ordinary
//! [`crate::fv::fvm_laplacian`], is left unrelaxed, and - crucially - is never
//! modified to pin a reference cell. That last point is what lets the direct
//! cuFFT backend of [`crate::pressure`] recognise it: that backend re-derives
//! the whole operator from `diag`, `upper` and `lower` on every solve and
//! refuses anything that is not the separable Poisson operator, a pinned row
//! included. When the pressure has no Dirichlet boundary anywhere the level is
//! instead fixed *after* the solve, by subtracting the reference cell's value,
//! which changes no gradient and therefore no flux and no velocity. See
//! `smpSubScalar` in `cuda/simple.cu`.
//!
//! # What "converged" looks like
//!
//! The number to watch is not the pressure residual - which a Poisson solver
//! will drive to whatever tolerance it is given every iteration, converged or
//! not - but `max_c |Σ_f phi_f|`, the largest amount of volume a cell is
//! gaining or losing per second. [`SimplePerformance::continuity_error`]
//! reports it.

use cudarc::driver::{CudaFunction, PushKernelArg};

use crate::device::{cfg_for, DevBuf, Gpu, KernelSet};
use crate::error::{Error, Result};
use crate::field::{BcKind, GpuScalarField, GpuSurfaceScalarField, GpuVectorField};
use crate::field_ops::{self, FieldKernels};
use crate::fv::{self, FvKernels};
use crate::io::case::SolverControls;
use crate::ldu::GpuLduMatrix;
use crate::ldu_ops::{self, LduKernels};
use crate::mesh::{GpuMesh, HostMesh};
use crate::momentum::{BuoyancyCoeffs, Momentum, MomentumControls};
use crate::pressure::PressureBackend;
use crate::solver::{self, SolverKernels, SolverPerformance};
use crate::turbulence::FlowState;
use crate::{Label, Scalar};

// ==========================================================================
//  Controls
// ==========================================================================

/// Everything the outer loop reads out of `fvSolution`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SimpleControls {
    pub momentum: MomentumControls,

    pub p_solver: SolverControls,

    /// Explicit relaxation of the pressure FIELD (Patankar §6.7). SPEC-LIT
    /// §5.2 recommends `alpha_p ≈ 1 - alpha_U`, so 0.3 against the momentum
    /// equation's 0.7.
    ///
    /// SIMPLEC ([`MomentumControls::simplec`]) keeps the neighbour corrections
    /// the plain algorithm drops and permits `alpha_p = 1`.
    pub p_relax: Scalar,

    /// Extra pressure solves per iteration, each with the explicit
    /// non-orthogonal correction rebuilt from the latest `p` (Jasak §3.4.3).
    /// Zero is right on an orthogonal mesh, where the correction is
    /// identically zero.
    pub n_non_orth_correctors: usize,

    /// `PISO/nCorrectors` (or `PIMPLE/nCorrectors`): how many times the
    /// pressure is corrected per outer iteration, with `H` re-evaluated
    /// before each - SPEC-LIT §14. One is SIMPLE; two is the usual PISO
    /// setting and gives second-order splitting error (Issa 1986).
    ///
    /// Not to be confused with [`Self::n_non_orth_correctors`], which repeats
    /// the pressure SOLVE at a fixed `HbyA` to iterate the explicit
    /// non-orthogonal correction. Repeating the solve is not correcting the
    /// pressure: `H` has to move for that.
    pub n_correctors: usize,

    /// `PIMPLE/nOuterCorrectors`: how many times the whole momentum-pressure
    /// system is re-linearised within one time step - SPEC-LIT §14. One is
    /// PISO (or SIMPLE); more lets a transient run take a step past the
    /// Courant limit.
    pub n_outer_correctors: usize,

    /// `PIMPLE/momentumPredictor`. Off skips the momentum solve and lets the
    /// pressure correctors do all the work, which is worth having only where
    /// convection is weak. *DESIGN*: default on, because the predictor is what
    /// makes `H` mean anything on the first corrector.
    pub momentum_predictor: bool,

    /// Read `max_c |Σ_f phi_f|` back to the host at the end of every
    /// iteration. One eight-byte copy, and the only host traffic the loop has
    /// once the linear solvers are in fixed-iteration mode - so it must be off
    /// for a genuinely transfer-free run or for CUDA-graph capture.
    pub report_continuity: bool,
}

impl Default for SimpleControls {
    fn default() -> Self {
        Self {
            momentum: MomentumControls::default(),
            p_solver: SolverControls::default(),
            p_relax: 0.3,
            n_non_orth_correctors: 0,
            n_correctors: 1,
            n_outer_correctors: 1,
            momentum_predictor: true,
            report_continuity: true,
        }
    }
}

impl SimpleControls {
    fn validate(&self) -> Result<()> {
        if !(self.p_relax > 0.0 && self.p_relax <= 1.0) {
            return Err(Error::Config(format!(
                "relaxationFactors/fields/p is {}; the pressure is relaxed as a \
                 field and needs 0 < alpha <= 1 (SPEC-LIT §5.2)",
                self.p_relax
            )));
        }
        if self.n_non_orth_correctors > 64 {
            return Err(Error::Config(format!(
                "nNonOrthogonalCorrectors is {}; that is a pressure solve per \
                 corrector and cannot be what was meant",
                self.n_non_orth_correctors
            )));
        }
        // Zero pressure correctors would solve momentum and never make the
        // flux conservative, which is not an algorithm anybody asked for.
        if self.n_correctors == 0 || self.n_correctors > 64 {
            return Err(Error::Config(format!(
                "nCorrectors is {}; PISO needs at least one pressure corrector, and more than a handful is not an algorithm (SPEC-LIT §14)",
                self.n_correctors
            )));
        }
        if self.n_outer_correctors == 0 || self.n_outer_correctors > 1024 {
            return Err(Error::Config(format!(
                "nOuterCorrectors is {}; PIMPLE needs at least one outer corrector (SPEC-LIT §14)",
                self.n_outer_correctors
            )));
        }
        Ok(())
    }
}

/// What one SIMPLE iteration did.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct SimplePerformance {
    /// The three momentum components, in `x, y, z` order.
    pub u: [SolverPerformance; 3],
    /// The last pressure corrector's.
    pub p: SolverPerformance,
    /// `max_c |Σ_f phi_f|`, in m³/s. Zero when
    /// [`SimpleControls::report_continuity`] is off - nothing was measured,
    /// and reporting a number nobody computed would be worse.
    pub continuity_error: Scalar,
}

/// What one call to [`Simple::solve_step`] did.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct PimplePerformance {
    /// The LAST outer corrector's, which is the only one whose systems were
    /// assembled from coefficients the step had settled on.
    pub last: SimplePerformance,
    /// The FIRST outer corrector's initial residuals - the ones
    /// `residualControl` is tested against and the ones a convergence history
    /// should plot, because they measure the step's starting error rather than
    /// how hard the linear solver was pushed.
    pub first: SimplePerformance,
    /// How many outer correctors actually ran.
    pub n_outer: usize,
    /// True when `residualControl` stopped the outer loop early.
    pub converged: bool,
}

// ==========================================================================
//  Kernels
// ==========================================================================

struct SimpleKernels {
    face_flux_sum: CudaFunction,
    relax_field: CudaFunction,
    pick_value: CudaFunction,
    sub_scalar: CudaFunction,
}

impl SimpleKernels {
    fn new(gpu: &Gpu) -> Result<Self> {
        let k = KernelSet::new(gpu, crate::kernels::SIMPLE)?;
        Ok(Self {
            face_flux_sum: k.func("smpFaceFluxSum")?,
            relax_field: k.func("smpRelaxField")?,
            pick_value: k.func("smpPickValue")?,
            sub_scalar: k.func("smpSubScalar")?,
        })
    }
}

// ==========================================================================
//  Simple
// ==========================================================================

/// The coupled `U`-`p`-`phi` system.
///
/// Owns the three fields, because they are only meaningful together: a `phi`
/// that is not the one the last pressure solve produced is not a conservative
/// flux, and handing the three out separately would make that easy to get
/// wrong. Setup writes through [`Simple::u_mut`] and friends before
/// [`Simple::initialise`]; after that the loop owns them.
pub struct Simple<'m> {
    m: &'m GpuMesh,
    ctrl: SimpleControls,

    momentum: Momentum<'m>,

    u: GpuVectorField,
    p: GpuScalarField,
    phi: GpuSurfaceScalarField,

    /// The pressure Poisson system. Separate from the momentum matrix because
    /// [`crate::pressure`] holds on to its structure between solves.
    a_p: GpuLduMatrix,
    p_old: DevBuf<Scalar>,
    /// `Σ_f phi_f` per cell.
    div_phi: DevBuf<Scalar>,
    /// One-element landing pads for the two device reductions.
    red: DevBuf<Scalar>,

    /// `residualControl`, when the case gave one - the outer loop's stopping
    /// criterion. Not part of [`SimpleControls`] because that struct is
    /// `Copy` and this one owns a list of names.
    residual_control: Option<crate::io::case::ResidualControl>,

    /// True when `p` has no Dirichlet boundary anywhere, so the Poisson
    /// operator is singular and its level is ours to choose.
    pinned: bool,
    reference_cell: usize,

    fvk: FvKernels,
    lduk: LduKernels,
    fldk: FieldKernels,
    solk: SolverKernels,
    sk: SimpleKernels,
}

impl<'m> Simple<'m> {
    pub fn new(
        gpu: &Gpu,
        hm: &HostMesh,
        m: &'m GpuMesh,
        ctrl: SimpleControls,
        buoyancy: BuoyancyCoeffs,
    ) -> Result<Self> {
        ctrl.validate()?;

        if hm.n_cells != m.n_cells || hm.n_boundary_faces != m.n_boundary_faces {
            return Err(Error::Config(format!(
                "Simple::new: the host mesh has ({}, {}) cells/boundary faces \
                 and the device mesh ({}, {})",
                hm.n_cells, hm.n_boundary_faces, m.n_cells, m.n_boundary_faces
            )));
        }

        let nc = m.n_cells.max(1);

        Ok(Self {
            m,
            ctrl,

            momentum: Momentum::new(gpu, m, ctrl.momentum, buoyancy)?,

            u: GpuVectorField::zeros(gpu, m, "U")?,
            p: GpuScalarField::zeros(gpu, m, "p")?,
            phi: GpuSurfaceScalarField::zeros(gpu, m, "phi")?,

            a_p: GpuLduMatrix::new(gpu, m)?,
            p_old: gpu.zeros(nc)?,
            div_phi: gpu.zeros(nc)?,
            red: gpu.zeros(1)?,

            residual_control: None,
            pinned: false,
            reference_cell: 0,

            fvk: FvKernels::new(gpu)?,
            lduk: LduKernels::new(gpu)?,
            fldk: FieldKernels::new(gpu)?,
            solk: SolverKernels::new(gpu)?,
            sk: SimpleKernels::new(gpu)?,
        })
    }

    // ---- the fields -------------------------------------------------------

    pub fn u(&self) -> &GpuVectorField {
        &self.u
    }

    pub fn u_mut(&mut self) -> &mut GpuVectorField {
        &mut self.u
    }

    pub fn p(&self) -> &GpuScalarField {
        &self.p
    }

    pub fn p_mut(&mut self) -> &mut GpuScalarField {
        &mut self.p
    }

    pub fn phi(&self) -> &GpuSurfaceScalarField {
        &self.phi
    }

    pub fn phi_mut(&mut self) -> &mut GpuSurfaceScalarField {
        &mut self.phi
    }

    pub fn controls(&self) -> &SimpleControls {
        &self.ctrl
    }

    /// Give the outer loop a stopping criterion.
    ///
    /// Tested on the INITIAL residual of each outer corrector, per SPEC-LIT
    /// §14: the final residual measures the linear solver, and a loop that
    /// watched it would stop as soon as the linear solves became easy.
    pub fn set_residual_control(&mut self, rc: crate::io::case::ResidualControl) {
        self.residual_control = if rc.is_empty() { None } else { Some(rc) };
    }

    pub fn momentum(&self) -> &Momentum<'m> {
        &self.momentum
    }

    pub fn momentum_mut(&mut self) -> &mut Momentum<'m> {
        &mut self.momentum
    }

    /// What a turbulence model or a scalar transport equation needs from the
    /// flow: the velocity, the conservative flux, and the molecular viscosity.
    pub fn flow_state(&self) -> FlowState<'_> {
        FlowState::new(&self.u, &self.phi, self.ctrl.momentum.nu)
    }

    /// The assembled pressure system - what [`crate::pressure::SystemProbe`]
    /// is meant to be shown.
    pub fn pressure_matrix(&self) -> &GpuLduMatrix {
        &self.a_p
    }

    /// The pressure laplacian's coefficient, `(rAU_f·|Sf|, rAU_b·|Sf_b|)`.
    pub fn pressure_laplacian_coeffs(&self) -> (&DevBuf<Scalar>, &DevBuf<Scalar>) {
        self.momentum.pressure_laplacian_coeffs()
    }

    /// True when the pressure has no `fixedValue` anywhere, so only its
    /// gradients mean anything and the level is set by this module.
    pub fn pressure_is_pinned(&self) -> bool {
        self.pinned
    }

    /// The cell whose pressure is held at zero when the problem is pinned.
    pub fn reference_cell(&self) -> usize {
        self.reference_cell
    }

    // ---- setup ------------------------------------------------------------

    /// Evaluate the boundary faces, seed the old-time levels, and work out
    /// whether the pressure needs a reference level.
    ///
    /// Call once, after the fields have been written through
    /// [`Simple::u_mut`], [`Simple::p_mut`] and [`Simple::phi_mut`], and
    /// before the first [`Simple::correct`].
    pub fn initialise(&mut self, gpu: &Gpu) -> Result<()> {
        let m = self.m;

        field_ops::correct_boundary_conditions_vector(gpu, &self.fldk, &mut self.u, m)?;
        field_ops::correct_boundary_conditions(gpu, &self.fldk, &mut self.p, m)?;

        field_ops::store_old_time_vector(gpu, &self.fldk, &mut self.u)?;
        field_ops::store_old_time(gpu, &self.fldk, &mut self.p)?;

        self.pinned = !self.pressure_has_a_dirichlet(gpu)?;
        self.reference_cell = 0;

        Ok(())
    }

    /// Does any boundary face give the pressure a value rather than a
    /// gradient?
    ///
    /// `empty` and `cyclic` faces are skipped: neither contributes a matrix
    /// coefficient that could fix a level - an empty patch contributes nothing
    /// at all, and a cyclic couple relates two unknowns without pinning
    /// either. Everything else is judged by `fr`, because that is exactly the
    /// coefficient that puts a value into the diagonal (SPEC-LIT §4).
    fn pressure_has_a_dirichlet(&self, gpu: &Gpu) -> Result<bool> {
        let nbf = self.m.n_boundary_faces;
        if nbf == 0 {
            return Ok(false);
        }

        let fr = gpu.download(&self.p.fr)?;
        let kinds = gpu.download(&self.p.bc_kind)?;

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

    // ---- one iteration ----------------------------------------------------

    /// Open a new TIME step: rotate `U`'s two old levels and move the ddt
    /// term's step history on.
    ///
    /// Separate from [`Simple::correct`] on purpose. `correct` refreshes
    /// `U^{n-1}` once per outer corrector, which is what SIMPLE wants - each
    /// corrector differences against the start of its own sub-step. `U^{n-2}`
    /// is a property of the TIME step, and rotating it per corrector would
    /// collapse it onto `U^{n-1}` and make BDF2 quietly first order
    /// (`SPEC-LIT` §13.3). A transient driver calls this once per step, before
    /// the first `correct`; a steady one need not call it at all.
    pub fn begin_time_step(&mut self, gpu: &Gpu, dt: Scalar) -> Result<()> {
        field_ops::advance_time_levels_vector(gpu, &self.fldk, &mut self.u)?;
        field_ops::advance_time_levels(gpu, &self.fldk, &mut self.p)?;
        self.momentum.ddt.advance(dt);
        Ok(())
    }

    /// One TIME STEP (or, in a steady run, one outer iteration), as
    /// SPEC-LIT §14's single loop.
    ///
    /// `nOuterCorrectors` passes of [`Simple::correct_outer`], with
    /// relaxation switched off on the last one in a transient run, stopping
    /// early when every field named in `residualControl` has met its
    /// tolerance on its INITIAL residual.
    ///
    /// `nut` is the eddy viscosity from the turbulence model and `t` the
    /// temperature the buoyancy is built from. Both are read, never written.
    /// A laminar isothermal case passes a zero `nut` and a uniform `t` at the
    /// reference temperature, where the body force is zero to the last bit.
    ///
    /// The caller still owns the time-level rotation: call
    /// [`Simple::begin_time_step`] once per step, before this.
    pub fn solve_step(
        &mut self,
        gpu: &Gpu,
        backend: &mut dyn PressureBackend,
        nut: &GpuScalarField,
        t: &GpuScalarField,
    ) -> Result<PimplePerformance> {
        if self.m.n_cells == 0 {
            return Ok(PimplePerformance::default());
        }

        let n_outer = self.ctrl.n_outer_correctors.max(1);
        // Relaxation replaces the time derivative in a steady run (Patankar
        // §6.7), so "final" - the iteration that runs unrelaxed - exists only
        // where there IS a time derivative. See the module note.
        let transient = !self.ctrl.momentum.steady;

        let mut out = PimplePerformance::default();

        for outer in 0..n_outer {
            let is_final = transient && outer + 1 == n_outer;
            let perf = self.correct_outer(gpu, backend, nut, t, is_final)?;

            if outer == 0 {
                out.first = perf;
            }
            out.last = perf;
            out.n_outer = outer + 1;

            // residualControl is tested on the INITIAL residual, and on the
            // residual of THIS outer iteration - the error the step still has,
            // not how hard the last linear solve was pushed (SPEC-LIT §14).
            if self.outer_converged(&perf) {
                out.converged = true;
                break;
            }
        }

        Ok(out)
    }

    /// One outer corrector: the momentum predictor, then `nCorrectors`
    /// pressure correctors, then the pressure relaxation.
    ///
    /// `is_final` switches BOTH relaxations off, which is what makes the last
    /// PIMPLE iteration end on the unrelaxed equations (SPEC-LIT §14). A
    /// steady run never passes `true`.
    pub fn correct_outer(
        &mut self,
        gpu: &Gpu,
        backend: &mut dyn PressureBackend,
        nut: &GpuScalarField,
        t: &GpuScalarField,
        is_final: bool,
    ) -> Result<SimplePerformance> {
        self.correct_outer_impl(gpu, backend, nut, t, None, is_final)
    }

    /// [`Self::correct_outer`] plus SPEC-LIT §25.1's target divergence in the
    /// pressure equation's source - the ONE seam §25.3 names for a low-Mach
    /// solver ("SIMPLE/PISO change in ONE place"). `target_div` is
    /// [`crate::energy::Energy::target_divergence`], refreshed for this outer
    /// iteration before this is called.
    ///
    /// Everything else is identical to [`Self::correct_outer`] - which is
    /// exactly [`Self::correct_outer_impl`] called with `target_div = None`,
    /// so a driver that never calls this one gets bit-identical behaviour to
    /// before this method existed.
    pub fn correct_outer_low_mach(
        &mut self,
        gpu: &Gpu,
        backend: &mut dyn PressureBackend,
        nut: &GpuScalarField,
        t: &GpuScalarField,
        target_div: &DevBuf<Scalar>,
        is_final: bool,
    ) -> Result<SimplePerformance> {
        self.correct_outer_impl(gpu, backend, nut, t, Some(target_div), is_final)
    }

    fn correct_outer_impl(
        &mut self,
        gpu: &Gpu,
        backend: &mut dyn PressureBackend,
        nut: &GpuScalarField,
        t: &GpuScalarField,
        target_div: Option<&DevBuf<Scalar>>,
        is_final: bool,
    ) -> Result<SimplePerformance> {
        let m = self.m;
        let n = m.n_cells;
        if n == 0 {
            return Ok(SimplePerformance::default());
        }

        // inletOutlet switches on the sign of the face flux, so the fractions
        // have to follow the flux the last iteration produced before anything
        // reads them.
        field_ops::update_inlet_outlet_vector(gpu, &self.fldk, &mut self.u, &self.phi)?;
        field_ops::update_inlet_outlet_scalar(gpu, &self.fldk, &mut self.p, &self.phi)?;
        field_ops::correct_boundary_conditions_vector(gpu, &self.fldk, &mut self.u, m)?;
        field_ops::correct_boundary_conditions(gpu, &self.fldk, &mut self.p, m)?;

        // The velocity's old-time level - in a STEADY run only.
        //
        // A transient run gets `U^{n-1}` from `begin_time_step`, once per time
        // step, and refreshing it here would make every corrector difference
        // against the previous corrector instead of against the start of the
        // step: `nCorrectors 2` would advance `U` by two Euler sub-steps while
        // `T` and `k`/`epsilon` advanced by one, and the fields would drift
        // apart in time with nothing in the output saying so. That is a
        // time-integration error, not a wiring detail, and it is what this
        // branch removes.
        //
        // Steady is the other way round: `r_delta_t` is zero, so `f0` is never
        // read at all and this copy only keeps a driver that never calls
        // `begin_time_step` from carrying a stale level around.
        if self.ctrl.momentum.steady {
            field_ops::store_old_time_vector(gpu, &self.fldk, &mut self.u)?;
        }

        // Relaxation off on the final outer corrector, and restored on the way
        // out so the controls still describe the case.
        let alpha_u = self.ctrl.momentum.u_relax;
        if is_final {
            self.momentum.set_relaxation(1.0)?;
        }

        // ---- the forces, on faces -----------------------------------------
        self.momentum.update_buoyancy(gpu, t, &self.u)?;
        self.momentum.update_force(gpu, &self.p, &self.u)?;

        // ---- momentum predictor -------------------------------------------
        let u_perf = if self.ctrl.momentum_predictor {
            self.momentum.solve(gpu, &mut self.u, &self.phi, nut)?
        } else {
            // No predictor still needs the matrix that `H` and `rAU` come out
            // of, and that matrix needs the eddy viscosity and the convection
            // weights - so the assembly happens either way; only the linear
            // solve is skipped.
            self.momentum.assemble_only(gpu, &self.u, &self.phi, nut)?;
            [SolverPerformance::default(); 3]
        };

        if is_final {
            self.momentum.set_relaxation(alpha_u)?;
        }

        // The pressure this outer iteration is relaxed towards. Taken once,
        // before the correctors, because `alpha_p` relaxes the OUTER
        // iteration's change and not each corrector's (SPEC-LIT §14).
        field_ops::copy_field(gpu, &self.fldk, &mut self.p_old, &self.p.f, n)?;

        let mut p_perf = SolverPerformance::default();

        // ---- the PISO correctors ------------------------------------------
        for _ in 0..self.ctrl.n_correctors.max(1) {
            // rAU, HbyA, phi_HbyA - and this is the line that makes it PISO:
            // `H` is re-evaluated from the velocity the PREVIOUS corrector
            // produced, with the momentum matrix held frozen (Issa 1986).
            self.momentum.rhie_chow(gpu, &self.u)?;

            // The non-orthogonal correctors: the same `phi_HbyA`, the
            // explicit correction re-evaluated against the latest `p`
            // (Jasak §3.4.3). A different loop doing a different job.
            for _ in 0..=self.ctrl.n_non_orth_correctors {
                self.assemble_pressure(gpu, target_div)?;
                p_perf = backend.solve(gpu, &mut self.p.f, &self.a_p, m)?;

                if self.pinned {
                    self.fix_pressure_level(gpu)?;
                }
                field_ops::correct_boundary_conditions(gpu, &self.fldk, &mut self.p, m)?;
            }

            // ---- flux and velocity, from the pressure that satisfies
            //      continuity ------------------------------------------------
            self.momentum.update_force(gpu, &self.p, &self.u)?;
            self.momentum
                .correct_flux_and_velocity(gpu, &mut self.u, &mut self.phi)?;
        }

        // ---- relax the pressure FIELD, for the next predictor ---------------
        if !is_final {
            self.relax_pressure(gpu)?;
            field_ops::correct_boundary_conditions(gpu, &self.fldk, &mut self.p, m)?;
        }

        let continuity_error = if self.ctrl.report_continuity {
            self.continuity_error(gpu)?
        } else {
            0.0
        };

        Ok(SimplePerformance {
            u: u_perf,
            p: p_perf,
            continuity_error,
        })
    }

    /// One SIMPLE iteration - one outer corrector, relaxed.
    ///
    /// Kept as the entry point for a steady driver, which has no outer loop to
    /// speak of: with the default `nCorrectors 1` this is exactly the
    /// algorithm of SPEC-LIT §5.2, term for term.
    pub fn correct(
        &mut self,
        gpu: &Gpu,
        backend: &mut dyn PressureBackend,
        nut: &GpuScalarField,
        t: &GpuScalarField,
    ) -> Result<SimplePerformance> {
        self.correct_outer(gpu, backend, nut, t, false)
    }

    /// Have all the fields named in `residualControl` met their tolerance?
    ///
    /// False when no control was given: an outer loop with no stopping
    /// criterion runs its full count, which is what `nOuterCorrectors` means
    /// on its own.
    fn outer_converged(&self, perf: &SimplePerformance) -> bool {
        let Some(rc) = &self.residual_control else {
            return false;
        };
        if rc.is_empty() {
            return false;
        }

        // `U` is reported as the WORST of the three components: a control on
        // "U" that only watched `Ux` would stop a run whose cross-flow had not
        // converged at all.
        let u_res = perf
            .u
            .iter()
            .map(|q| q.initial_residual)
            .fold(0.0 as Scalar, Scalar::max);

        rc.all_satisfied(&[
            ("U", u_res),
            ("Ux", perf.u[0].initial_residual),
            ("Uy", perf.u[1].initial_residual),
            ("Uz", perf.u[2].initial_residual),
            ("p", perf.p.initial_residual),
        ])
    }

    /// `laplacian(rAU_f, p) = div(phi_HbyA)`, or with `target_div` supplied,
    /// SPEC-LIT §25.3's low-Mach source:
    /// `laplacian(rAU_f, p) = div(phi_HbyA) - (div u)_target`.
    ///
    /// The right-hand side is the volume integral `Σ_f (±phi_HbyA_f)`, not
    /// `(1/V)Σ_f` - it is added straight to the matrix source, which is what
    /// the operator's own `A·psi` is measured against. `target_div` is a
    /// per-cell (not volume-integrated) field, so it goes in through
    /// [`crate::fv::fvm_su`] exactly like any other §18 source, which does
    /// that integral itself.
    fn assemble_pressure(&mut self, gpu: &Gpu, target_div: Option<&DevBuf<Scalar>>) -> Result<()> {
        let m = self.m;
        let n = m.n_cells;

        // Before the coefficients are borrowed: this needs `&mut momentum`.
        if self.ctrl.momentum.sn_grad.applies() {
            self.momentum.update_p_gradient(gpu, &self.p)?;
        }

        self.a_p.zero(gpu)?;

        {
            let Self { momentum, a_p, p, fvk, ctrl, .. } = self;
            let (g, bg) = momentum.pressure_laplacian_coeffs();

            fv::fvm_laplacian(gpu, fvk, a_p, m, g, bg, p, 1.0)?;

            if ctrl.momentum.sn_grad.applies() {
                fv::fvm_laplacian_non_orth_correction(
                    gpu,
                    fvk,
                    a_p,
                    m,
                    g,
                    bg,
                    p,
                    momentum.grad_p(),
                    ctrl.momentum.sn_grad,
                    1.0,
                )?;
            }
        }

        // `accumulate`, because the non-orthogonal correction is already in
        // the source by now.
        {
            let Self { sk, a_p, momentum, .. } = self;
            let phi_hbya = momentum.phi_hbya();
            launch_face_flux_sum(
                gpu,
                &sk.face_flux_sum,
                &mut a_p.source,
                &phi_hbya.f,
                &phi_hbya.bf,
                m,
                true,
            )?;
        }

        // SPEC-LIT §25.1/§25.3: subtract the target divergence, so the
        // velocity this pressure equation produces satisfies
        // `div u = (div u)_target` rather than `div u = 0`. `sign = -1`
        // against `fvm_su`'s `source += sign*V*su` gives exactly
        // `source -= V*(div u)_target`. Absent for every existing caller
        // ([`Self::correct_outer`] passes `None`), so this is a no-op unless
        // a driver opts in through [`Self::correct_outer_low_mach`].
        if let Some(td) = target_div {
            fv::fvm_su(gpu, &self.fvk, &mut self.a_p, m, td, -1.0)?;
        }

        // A singular system needs a consistent right-hand side; see
        // `smpSubScalar` in `cuda/simple.cu`. Done before the boundary fold,
        // which on a pinned problem adds nothing to the source anyway (every
        // face has `fr = 0` and `refGrad = 0`), but the order is fixed so the
        // mean removed is the mean of the whole source.
        if self.pinned && n > 0 {
            let Self { solk, momentum, a_p, sk, .. } = self;
            let ws = momentum.workspace_mut();
            solver::device_sum(gpu, solk, &mut ws.num, &a_p.source, &mut ws.partials, n)?;
            let scale = 1.0 / n as Scalar;
            launch_sub_scalar(gpu, &sk.sub_scalar, &mut a_p.source, &ws.num, scale, n)?;
        }

        ldu_ops::add_boundary_contributions(gpu, &self.lduk, &mut self.a_p, m)
    }

    /// Subtract the reference cell's value, so a pressure defined only up to a
    /// constant has a definite one.
    fn fix_pressure_level(&mut self, gpu: &Gpu) -> Result<()> {
        let n = self.m.n_cells;
        if n == 0 {
            return Ok(());
        }
        let idx = self.reference_cell.min(n - 1) as Label;

        // Read the reference out first. Every thread subtracting `p[ref]`
        // while one of them writes it is a race.
        let f = self.sk.pick_value.clone();
        unsafe {
            gpu.stream()
                .launch_builder(&f)
                .arg(&mut self.red)
                .arg(&self.p.f)
                .arg(&idx)
                .launch(cfg_for(1))?;
        }

        let Self { sk, p, red, .. } = self;
        launch_sub_scalar(gpu, &sk.sub_scalar, &mut p.f, red, 1.0, n)
    }

    /// `p = p_old + alpha_p·(p - p_old)` - Patankar §6.7, SPEC-LIT §5.2.
    fn relax_pressure(&mut self, gpu: &Gpu) -> Result<()> {
        let n = self.m.n_cells;
        if n == 0 {
            return Ok(());
        }
        let alpha = self.ctrl.p_relax;
        let nl = n as Label;
        let f = self.sk.relax_field.clone();
        unsafe {
            gpu.stream()
                .launch_builder(&f)
                .arg(&mut self.p.f)
                .arg(&self.p_old)
                .arg(&alpha)
                .arg(&nl)
                .launch(cfg_for(n))?;
        }
        Ok(())
    }

    /// `max_c |Σ_f phi_f|` - how much volume the worst cell is inventing.
    ///
    /// The one host transfer in the iteration, eight bytes.
    pub fn continuity_error(&mut self, gpu: &Gpu) -> Result<Scalar> {
        let m = self.m;
        let n = m.n_cells;
        if n == 0 {
            return Ok(0.0);
        }

        {
            let Self { sk, div_phi, phi, .. } = self;
            launch_face_flux_sum(
                gpu,
                &sk.face_flux_sum,
                div_phi,
                &phi.f,
                &phi.bf,
                m,
                false,
            )?;
        }

        {
            let Self { solk, momentum, div_phi, .. } = self;
            let ws = momentum.workspace_mut();
            solver::device_max_mag(gpu, solk, &mut ws.den, div_phi, &mut ws.partials, n)?;
        }

        let v = gpu.download(&self.momentum.workspace().den)?;
        Ok(v.first().copied().unwrap_or(0.0))
    }
}

// ==========================================================================
//  Launch helpers
// ==========================================================================

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
//  Tests
//
//  Nothing here compares against another CFD code. The checks are the ones
//  SPEC-LIT §10 names for this module: a sealed box that must not move, a
//  hydrostatic column whose pressure must be smooth rather than merely
//  plausible, and the lid-driven cavity against the tabulated profiles of
//  Ghia, Ghia & Shin (1982). The sign of the buoyancy force is checked in
//  `crate::momentum`.
// ==========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesh::PatchKind;
    use crate::pressure::{PbicgstabBackend, SystemProbe};
    use crate::scalar_transport::weighted_stats;
    use crate::Vec3;

    /// A machine without a card makes every device test pass vacuously, which
    /// is the convention the rest of the crate follows.
    fn gpu() -> Option<Gpu> {
        Gpu::new(0).ok()
    }

    /// A structured box with the six patch kinds the caller asks for, with its
    /// geometry computed the way a real case's is.
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

    /// Every boundary face's patch index, so a test can say "this patch is the
    /// moving lid" without re-deriving the flattening.
    fn patch_of(hm: &HostMesh) -> Vec<usize> {
        let mut v = vec![0usize; hm.n_boundary_faces];
        for (p, pi) in hm.patches.iter().enumerate() {
            for k in 0..pi.size {
                v[pi.start + k] = p;
            }
        }
        v
    }

    /// Write a velocity boundary condition patch by patch: `Some(value)` is a
    /// Dirichlet, `None` leaves the patch zero-gradient. An `empty` patch is
    /// always left as such.
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

    /// The same for a scalar. `None` is zero-gradient.
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

    /// A zero eddy viscosity with calculated boundary values - a laminar run.
    fn laminar_nut(gpu: &Gpu, m: &GpuMesh, hm: &HostMesh) -> Result<GpuScalarField> {
        let mut nut = GpuScalarField::zeros(gpu, m, "nut")?;
        let kinds = vec![BcKind::Calculated as Label; hm.n_boundary_faces];
        gpu.write(&mut nut.bc_kind, &kinds)?;
        Ok(nut)
    }

    /// A uniform temperature field with the boundary condition the caller
    /// wants, already evaluated.
    fn temperature(
        gpu: &Gpu,
        m: &GpuMesh,
        hm: &HostMesh,
        cells: &[Scalar],
        per_patch: &[Option<Scalar>],
    ) -> Result<GpuScalarField> {
        let mut t = GpuScalarField::zeros(gpu, m, "T")?;
        gpu.write(&mut t.f, cells)?;
        set_scalar_bcs(gpu, &mut t, hm, per_patch)?;
        crate::field_ops::correct_boundary_conditions(gpu, &FieldKernels::new(gpu)?, &mut t, m)?;
        Ok(t)
    }

    fn max_mag(v: &[Vec3]) -> Scalar {
        v.iter().fold(0.0 as Scalar, |a, x| a.max(x.mag()))
    }

    fn max_abs(v: &[Scalar]) -> Scalar {
        v.iter().fold(0.0 as Scalar, |a, x| a.max(x.abs()))
    }

    /// Tight enough that what the tests measure is the discretisation and not
    /// the linear solver's stopping criterion.
    fn tight() -> SolverControls {
        SolverControls {
            tolerance: 1e-12,
            rel_tol: 0.0,
            max_iter: 500,
            check_interval: 1,
            ..SolverControls::default()
        }
    }

    /// The two buoyancy tests run TRANSIENT and in a very viscous fluid, and
    /// both choices are deliberate.
    ///
    /// *Transient*, because "stays at rest" is a statement about a fluid that
    /// starts at rest. A steady solve started from `p = 0` is not that: a
    /// uniform pressure against a 4 m/s2 body force is a violently
    /// out-of-equilibrium state, the momentum predictor answers it with the
    /// Stokes flow that force would drive with no pressure at all, and what
    /// follows is a real startup transient rather than a statement about the
    /// discretisation. With a time derivative the predictor can only move by
    /// `dt*b`, and the pressure removes that on the same step.
    ///
    /// *Viscous* (`nu = 1`), so the momentum diffusion timescale across the
    /// box is milliseconds rather than a quarter of an hour. Whatever residue
    /// the first step leaves then decays inside the test, instead of being
    /// carried undecayed to the last step where it is indistinguishable from a
    /// bug.
    fn at_rest_controls() -> SimpleControls {
        SimpleControls {
            momentum: MomentumControls {
                nu: 1.0,
                u_solver: tight(),
                steady: false,
                delta_t: 0.01,
                sn_grad: crate::fv::SnGradScheme::Uncorrected,
                variable_viscosity_stress: false,
                ..MomentumControls::default()
            },
            p_solver: tight(),
            ..SimpleControls::default()
        }
    }

    // ----------------------------------------------------------------------
    //  A sealed box at the reference temperature
    // ----------------------------------------------------------------------

    /// With `T = T_ref` everywhere the body force `g·(T_ref/T - 1)` is zero
    /// EXACTLY, not merely small, and every stage downstream of it is exactly
    /// zero too. Asserting equality rather than a tolerance is the point: the
    /// buoyancy term cannot inject a drift into an isothermal run at any
    /// amplitude at all, which a `beta·(T - T_ref)` written with a rounded
    /// `beta` could not promise.
    #[test]
    fn an_isothermal_sealed_box_stays_exactly_at_rest() -> Result<()> {
        let Some(g) = gpu() else { return Ok(()) };

        let hm = boxed([5, 4, 6], Vec3::new(0.1, 0.1, 0.1), [PatchKind::Wall; 6]);
        let m = GpuMesh::upload(&g, &hm)?;

        let t_ref: Scalar = 293.15;
        let buoy = BuoyancyCoeffs {
            g: Vec3::new(0.0, 0.0, -9.81),
            t_ref,
            t_min: 1.0,
        };

        let ctrl = SimpleControls {
            momentum: MomentumControls {
                nu: 1.5e-5,
                sn_grad: crate::fv::SnGradScheme::Uncorrected,
                variable_viscosity_stress: false,
                ..MomentumControls::default()
            },
            ..SimpleControls::default()
        };

        let mut s = Simple::new(&g, &hm, &m, ctrl, buoy)?;
        set_velocity_bcs(&g, s.u_mut(), &hm, &[Some(Vec3::ZERO); 6])?;
        set_scalar_bcs(&g, s.p_mut(), &hm, &[None; 6])?;
        s.initialise(&g)?;

        assert!(
            s.pressure_is_pinned(),
            "a pressure with no fixedValue anywhere must be recognised as pinned"
        );

        let nut = laminar_nut(&g, &m, &hm)?;
        let t = temperature(&g, &m, &hm, &vec![t_ref; hm.n_cells], &[Some(t_ref); 6])?;

        let mut backend = PbicgstabBackend::new(ctrl.p_solver);
        backend.setup(&g, &hm, &m, &SystemProbe::default())?;

        for _ in 0..5 {
            s.correct(&g, &mut backend, &nut, &t)?;
        }

        let u = g.download(&s.u().f)?;
        let phi = g.download(&s.phi().f)?;
        let p = g.download(&s.p().f)?;

        assert!(
            u.iter().all(|v| *v == Vec3::ZERO),
            "U moved: max |U| = {}",
            max_mag(&u)
        );
        assert!(phi.iter().all(|f| *f == 0.0), "phi is not identically zero");
        assert!(p.iter().all(|q| *q == 0.0), "p is not identically zero");
        Ok(())
    }

    // ----------------------------------------------------------------------
    //  A sealed box at a uniform temperature that is NOT the reference
    // ----------------------------------------------------------------------

    /// A uniform body force is a pure gradient, so the pressure must absorb it
    /// entirely and the fluid must not move.
    ///
    /// The force here is `-9.81·(293.15/500 - 1) = +4.06 m/s²` - four tenths
    /// of gravity, on every cell - and the velocity that survives it measures
    /// whether the body force and the pressure gradient are discretised
    /// consistently. They are, only because both live on faces.
    #[test]
    fn a_uniform_body_force_is_absorbed_by_the_pressure() -> Result<()> {
        let Some(g) = gpu() else { return Ok(()) };

        let (nx, ny, nz) = (4usize, 4usize, 8usize);
        let d = Vec3::new(0.1, 0.1, 0.1);
        let hm = boxed([nx, ny, nz], d, [PatchKind::Wall; 6]);
        let m = GpuMesh::upload(&g, &hm)?;

        let t_ref: Scalar = 293.15;
        let t_box: Scalar = 500.0;
        let gz: Scalar = -9.81;
        let buoy = BuoyancyCoeffs { g: Vec3::new(0.0, 0.0, gz), t_ref, t_min: 1.0 };

        let ctrl = at_rest_controls();

        let mut s = Simple::new(&g, &hm, &m, ctrl, buoy)?;
        set_velocity_bcs(&g, s.u_mut(), &hm, &[Some(Vec3::ZERO); 6])?;
        set_scalar_bcs(&g, s.p_mut(), &hm, &[None; 6])?;
        s.initialise(&g)?;

        let nut = laminar_nut(&g, &m, &hm)?;
        let t = temperature(&g, &m, &hm, &vec![t_box; hm.n_cells], &[Some(t_box); 6])?;

        let mut backend = PbicgstabBackend::new(ctrl.p_solver);
        backend.setup(&g, &hm, &m, &SystemProbe::default())?;

        let mut last = SimplePerformance::default();
        for _ in 0..60 {
            last = s.correct(&g, &mut backend, &nut, &t)?;
        }

        let u = g.download(&s.u().f)?;
        let p = g.download(&s.p().f)?;

        // The acceleration the pressure had to balance, for scale.
        let b = gz * (t_ref / t_box - 1.0);
        assert!(b > 4.0, "the test is not exercising anything: b = {b}");

        let speed = max_mag(&u);
        assert!(
            speed < 1.0e-9,
            "a uniform body force set the fluid moving at {speed} m/s against a \
             driving acceleration of {b} m/s2"
        );
        assert!(
            last.continuity_error < 1e-12,
            "the flux is not conservative: max |sum_f phi| = {}",
            last.continuity_error
        );

        // p must be the exact discrete hydrostatic field. On a uniform mesh
        // the face temperature is the box temperature, so the balance
        // `|Sf| snGrad(p) = (T_ref/T_f - 1)(g·Sf)` reads
        // `p_N - p_P = h·g_z·(T_ref/T - 1)` across every z face and `0` across
        // every x and y face.
        let layer = nx * ny;
        let want = d.z * b;
        for c in 0..hm.n_cells {
            let k = c / layer;
            let dp = p[c] - p[c % layer];
            let expect = want * k as Scalar;
            assert!(
                (dp - expect).abs() < 1e-9 * (1.0 + expect.abs()),
                "cell {c}: p - p_bottom = {dp}, hydrostatic says {expect}"
            );
        }
        Ok(())
    }

    // ----------------------------------------------------------------------
    //  A stratified column: the checkerboard test
    // ----------------------------------------------------------------------

    /// A hot-above-cold column in gravity. Nothing may move, and the pressure
    /// that holds it up must be SMOOTH.
    ///
    /// This is the test the module exists to pass. A body force interpolated
    /// from cell values balances a cell-centred pressure gradient, and a
    /// cell-centred gradient is blind to the mode that alternates sign from
    /// one cell to the next: `(p_{k+1} - p_{k-1})/2h` is identical for a
    /// smooth field and for the same field plus any sawtooth. The residual
    /// therefore converges, the velocity stays near zero, and the pressure
    /// field is nonetheless wrong - by an amount a contour plot will not show,
    /// because the sawtooth is precisely the mode that plots smoothly.
    ///
    /// So the assertion is not on the residual. It is on the amplitude of the
    /// alternating mode itself, extracted as the second difference of the
    /// pressure up the column and measured against the smooth variation of the
    /// first difference; and, more strongly, on the discrete hydrostatic
    /// balance holding FACE BY FACE,
    ///
    /// ```text
    /// p_{k+1} - p_k = h·g_z·(T_ref/T_f - 1),   T_f = (T_k + T_{k+1})/2
    /// ```
    ///
    /// which is what a face body force buys and what an interpolated one
    /// cannot deliver.
    #[test]
    fn a_hydrostatic_column_has_no_checkerboard() -> Result<()> {
        let Some(g) = gpu() else { return Ok(()) };

        let (nx, ny, nz) = (3usize, 3usize, 32usize);
        let h: Scalar = 0.05;
        let d = Vec3::new(h, h, h);
        let hm = boxed([nx, ny, nz], d, [PatchKind::Wall; 6]);
        let m = GpuMesh::upload(&g, &hm)?;

        let t_ref: Scalar = 293.15;
        let gz: Scalar = -9.81;
        let buoy = BuoyancyCoeffs { g: Vec3::new(0.0, 0.0, gz), t_ref, t_min: 1.0 };

        // Stably stratified - hot on top - so buoyancy cannot drive convection
        // and "at rest" is the true answer rather than a slow transient.
        let t_of = |k: usize| -> Scalar {
            let z = (k as Scalar + 0.5) * h;
            t_ref + 400.0 * z / (nz as Scalar * h)
        };

        let ctrl = at_rest_controls();

        let mut s = Simple::new(&g, &hm, &m, ctrl, buoy)?;
        set_velocity_bcs(&g, s.u_mut(), &hm, &[Some(Vec3::ZERO); 6])?;
        set_scalar_bcs(&g, s.p_mut(), &hm, &[None; 6])?;
        s.initialise(&g)?;

        let nut = laminar_nut(&g, &m, &hm)?;
        let layer = nx * ny;
        let tcells: Vec<Scalar> = (0..hm.n_cells).map(|c| t_of(c / layer)).collect();
        // zeroGradient on every wall, so the boundary temperature is the
        // adjacent cell's and the column is not driven from its ends.
        let t = temperature(&g, &m, &hm, &tcells, &[None; 6])?;

        let mut backend = PbicgstabBackend::new(ctrl.p_solver);
        backend.setup(&g, &hm, &m, &SystemProbe::default())?;

        for _ in 0..80 {
            s.correct(&g, &mut backend, &nut, &t)?;
        }

        let u = g.download(&s.u().f)?;
        let p = g.download(&s.p().f)?;

        let speed = max_mag(&u);
        assert!(speed < 1e-9, "a stratified column started moving at {speed} m/s");

        // ---- one column of cells, bottom to top --------------------------
        let col: Vec<Scalar> = (0..nz).map(|k| p[k * layer]).collect();

        // ---- 1. the discrete hydrostatic balance, face by face -----------
        for k in 0..nz - 1 {
            let tf = 0.5 * (t_of(k) + t_of(k + 1));
            let want = h * gz * (t_ref / tf - 1.0);
            let got = col[k + 1] - col[k];
            assert!(
                (got - want).abs() < 1e-8 * want.abs().max(1e-3),
                "face {k}: dp = {got}, the face body force says {want}"
            );
        }

        // ---- 2. the amplitude of the alternating mode --------------------
        //
        // dp_k varies smoothly with height because T does, so the second
        // difference is that smooth variation plus twice whatever sawtooth is
        // present. Comparing the two measures the checkerboard directly.
        let dp: Vec<Scalar> = (0..nz - 1).map(|k| col[k + 1] - col[k]).collect();
        let smooth = max_abs(&dp);
        let d2: Vec<Scalar> = (0..dp.len() - 1).map(|k| dp[k + 1] - dp[k]).collect();
        let osc = max_abs(&d2);

        // What a smooth hydrostatic field's own second difference has to be.
        let smooth_d2 = (0..dp.len() - 1)
            .map(|k| {
                let a = h * gz * (t_ref / (0.5 * (t_of(k) + t_of(k + 1))) - 1.0);
                let b = h * gz * (t_ref / (0.5 * (t_of(k + 1) + t_of(k + 2))) - 1.0);
                (b - a).abs()
            })
            .fold(0.0 as Scalar, Scalar::max);

        assert!(
            osc <= smooth_d2 * 1.05 + 1e-12,
            "the pressure oscillates cell to cell: max |d2 p| = {osc}, but a \
             smooth hydrostatic field can only reach {smooth_d2} \
             (max |dp| = {smooth})"
        );

        // A genuine sawtooth of amplitude A has |dp| ~ 2A and |d2 p| ~ 4A, so
        // the indicator sits near 2 for a checkerboarded field and near zero
        // for a smooth one. Well below one is the qualitative statement; the
        // assertion above is the quantitative one.
        let indicator = osc / smooth;
        assert!(
            indicator < 0.2,
            "checkerboard indicator max|d2 p| / max|dp| = {indicator}"
        );

        // And a control, so the indicator is known to be able to SEE a
        // checkerboard rather than merely to be small. Two per cent of the
        // pressure range, alternating cell to cell, is a sawtooth far too
        // small to notice in a contour plot and far too large to survive this.
        let range = col[col.len() - 1] - col[0];
        let amp = 0.02 * range;
        let dirty: Vec<Scalar> = col
            .iter()
            .enumerate()
            .map(|(k, q)| q + if k % 2 == 0 { amp } else { -amp })
            .collect();
        let ddp: Vec<Scalar> = (0..dirty.len() - 1).map(|k| dirty[k + 1] - dirty[k]).collect();
        let dd2: Vec<Scalar> = (0..ddp.len() - 1).map(|k| ddp[k + 1] - ddp[k]).collect();
        let dirty_indicator = max_abs(&dd2) / max_abs(&ddp);
        assert!(
            dirty_indicator > 5.0 * indicator,
            "the checkerboard indicator cannot tell a sawtooth from a smooth \
             field: clean {indicator}, deliberately dirtied {dirty_indicator}"
        );

        println!(
            "hydrostatic column: max |U| = {speed:.3e} m/s, checkerboard \
             indicator {indicator:.4} (a 2% sawtooth reads {dirty_indicator:.4}), \
             worst face-balance error {:.3e}",
            (0..nz - 1)
                .map(|k| {
                    let tf = 0.5 * (t_of(k) + t_of(k + 1));
                    ((col[k + 1] - col[k]) - h * gz * (t_ref / tf - 1.0)).abs()
                })
                .fold(0.0 as Scalar, Scalar::max)
        );

        // Every layer must be uniform ACROSS the column, because g has no
        // horizontal component.
        for k in 0..nz {
            let base = k * layer;
            for c in 0..layer {
                assert!(
                    (p[base + c] - p[base]).abs() < 1e-10,
                    "layer {k} is not horizontally uniform"
                );
            }
        }

        // The volume-weighted statistics helper, on real data.
        let stats = weighted_stats(&tcells, &hm.v)?;
        assert!(stats.min < stats.max);
        assert!(stats.mean > stats.min && stats.mean < stats.max);

        Ok(())
    }

    // ----------------------------------------------------------------------
    //  Lid-driven cavity - Ghia, Ghia & Shin (1982)
    // ----------------------------------------------------------------------

    /// `u` along the vertical centreline at `Re = 100`, from Table I of
    /// **U. Ghia, K. N. Ghia, C. T. Shin, *J. Comput. Phys.* 48 (1982)
    /// 387-411**. `(y, u)`, `y` measured from the stationary floor and `u`
    /// scaled by the lid speed.
    const GHIA_U_RE100: &[(f64, f64)] = &[
        (1.0000, 1.00000),
        (0.9766, 0.84123),
        (0.9688, 0.78871),
        (0.9609, 0.73722),
        (0.9531, 0.68717),
        (0.8516, 0.23151),
        (0.7344, 0.00332),
        (0.6172, -0.13641),
        (0.5000, -0.20581),
        (0.4531, -0.21090),
        (0.2813, -0.15662),
        (0.1719, -0.10150),
        (0.1016, -0.06434),
        (0.0703, -0.04775),
        (0.0625, -0.04192),
        (0.0547, -0.03717),
        (0.0000, 0.00000),
    ];

    /// `v` along the horizontal centreline at `Re = 100`, Table II of the same
    /// paper. `(x, v)`.
    const GHIA_V_RE100: &[(f64, f64)] = &[
        (1.0000, 0.00000),
        (0.9688, -0.05906),
        (0.9609, -0.07391),
        (0.9531, -0.08864),
        (0.9453, -0.10313),
        (0.9063, -0.16914),
        (0.8594, -0.22445),
        (0.8047, -0.24533),
        (0.5000, 0.05454),
        (0.2344, 0.17527),
        (0.2266, 0.17507),
        (0.1563, 0.16077),
        (0.0938, 0.12317),
        (0.0781, 0.10890),
        (0.0703, 0.10091),
        (0.0625, 0.09233),
        (0.0000, 0.00000),
    ];

    /// Linear interpolation of a profile sampled at cell centres, extended to
    /// the two walls by the boundary values the case prescribes.
    fn sample(profile: &[(Scalar, Scalar)], at: Scalar) -> Scalar {
        if at <= profile[0].0 {
            return profile[0].1;
        }
        for w in profile.windows(2) {
            let (x0, v0) = w[0];
            let (x1, v1) = w[1];
            if at <= x1 {
                let t = (at - x0) / (x1 - x0);
                return v0 + t * (v1 - v0);
            }
        }
        profile[profile.len() - 1].1
    }

    /// The lid-driven cavity at `Re = 100`, against the tabulated centreline
    /// profiles of Ghia, Ghia & Shin (1982).
    ///
    /// This is the module's real validation: it exercises the momentum
    /// assembly, the Rhie-Chow flux, the pressure equation and the whole
    /// SIMPLE loop against a published measurement of the *solution*, not
    /// against another program's output. A checkerboarding pressure, a
    /// mis-signed pressure gradient or a Rhie-Chow term with the wrong
    /// coefficient all show up here as a recirculation of the wrong strength
    /// or in the wrong place.
    ///
    /// Second-order central convection: the cell Reynolds number is
    /// `U·h/nu = 1.6` on this mesh, comfortably inside the range where
    /// central differencing is bounded, and upwind on 64 cells would be too
    /// diffusive to say anything about a 129-point reference.
    #[test]
    fn lid_driven_cavity_matches_ghia_ghia_and_shin() -> Result<()> {
        let Some(g) = gpu() else { return Ok(()) };

        const N: usize = 64;
        let h: Scalar = 1.0 / N as Scalar;

        // 2-D: one cell deep with `empty` front and back.
        let kinds = [
            PatchKind::Wall,
            PatchKind::Wall,
            PatchKind::Wall,
            PatchKind::Wall,
            PatchKind::Empty,
            PatchKind::Empty,
        ];
        let hm = boxed([N, N, 1], Vec3::new(h, h, h), kinds);
        let m = GpuMesh::upload(&g, &hm)?;

        // Re = U L / nu = 1 * 1 / 0.01
        let nu: Scalar = 0.01;

        let ctrl = SimpleControls {
            momentum: MomentumControls {
                nu,
                u_solver: SolverControls {
                    tolerance: 1e-10,
                    max_iter: 200,
                    check_interval: 5,
                    ..SolverControls::default()
                },
                u_relax: 0.7,
                div_scheme: crate::io::case::DivScheme::Central,
                bounded_convection: true,
                sn_grad: crate::fv::SnGradScheme::Uncorrected,
                variable_viscosity_stress: false,
                ..MomentumControls::default()
            },
            p_solver: SolverControls {
                tolerance: 1e-10,
                max_iter: 600,
                check_interval: 5,
                ..SolverControls::default()
            },
            p_relax: 0.3,
            ..SimpleControls::default()
        };

        let mut s = Simple::new(&g, &hm, &m, ctrl, BuoyancyCoeffs {
            g: Vec3::ZERO,
            ..BuoyancyCoeffs::default()
        })?;

        // xmin xmax ymin: stationary walls. ymax: the lid, moving in +x.
        let lid = Vec3::new(1.0, 0.0, 0.0);
        set_velocity_bcs(
            &g,
            s.u_mut(),
            &hm,
            &[
                Some(Vec3::ZERO),
                Some(Vec3::ZERO),
                Some(Vec3::ZERO),
                Some(lid),
                None,
                None,
            ],
        )?;
        set_scalar_bcs(&g, s.p_mut(), &hm, &[None; 6])?;
        s.initialise(&g)?;

        let nut = laminar_nut(&g, &m, &hm)?;
        let t_ref = BuoyancyCoeffs::default().t_ref;
        let t = temperature(&g, &m, &hm, &vec![t_ref; hm.n_cells], &[Some(t_ref); 6])?;

        let mut backend = PbicgstabBackend::new(ctrl.p_solver);
        backend.setup(&g, &hm, &m, &SystemProbe::default())?;

        // A steady cavity at Re = 100 settles well inside this. The loop stops
        // on the momentum residual rather than on the continuity error,
        // because the pressure equation drives the flux to machine zero on
        // every iteration whether the outer loop has converged or not - the
        // continuity error is therefore about 1e-17 from the very first
        // iteration and says nothing at all about how far the fixed point
        // still is.
        let mut iters = 0usize;
        let mut last = SimplePerformance::default();
        for _ in 0..4000 {
            last = s.correct(&g, &mut backend, &nut, &t)?;
            iters += 1;
            let worst = last
                .u
                .iter()
                .map(|p| p.initial_residual)
                .fold(0.0 as Scalar, Scalar::max);
            if iters > 20 && worst < 1e-7 {
                break;
            }
        }

        assert!(
            last.continuity_error < 1e-8,
            "after {iters} iterations the flux still loses {} m3/s from a cell",
            last.continuity_error
        );

        // ---- centreline profiles -----------------------------------------
        let u = g.download(&s.u().f)?;
        let at = |i: usize, j: usize| u[i + N * j];

        // x = 0.5 and y = 0.5 both fall exactly between two cell centres on an
        // even mesh, so the centreline value is the average of the two
        // neighbouring lines - second-order accurate and unbiased.
        let (ia, ib) = (N / 2 - 1, N / 2);

        // u(y) along the vertical centreline, with the two walls appended.
        let mut u_prof: Vec<(Scalar, Scalar)> = vec![(0.0, 0.0)];
        for j in 0..N {
            let y = (j as Scalar + 0.5) * h;
            u_prof.push((y, 0.5 * (at(ia, j).x + at(ib, j).x)));
        }
        u_prof.push((1.0, 1.0));

        // v(x) along the horizontal centreline.
        let mut v_prof: Vec<(Scalar, Scalar)> = vec![(0.0, 0.0)];
        for i in 0..N {
            let x = (i as Scalar + 0.5) * h;
            v_prof.push((x, 0.5 * (at(i, ia).y + at(i, ib).y)));
        }
        v_prof.push((1.0, 0.0));

        let mut worst_u: Scalar = 0.0;
        for &(y, want) in GHIA_U_RE100 {
            let got = sample(&u_prof, y as Scalar);
            let e = (got - want as Scalar).abs();
            if e > worst_u {
                worst_u = e;
            }
            assert!(
                e < 0.035,
                "u at y = {y}: {got} against Ghia's {want} (error {e})"
            );
        }

        let mut worst_v: Scalar = 0.0;
        for &(x, want) in GHIA_V_RE100 {
            let got = sample(&v_prof, x as Scalar);
            let e = (got - want as Scalar).abs();
            if e > worst_v {
                worst_v = e;
            }
            assert!(
                e < 0.035,
                "v at x = {x}: {got} against Ghia's {want} (error {e})"
            );
        }

        // The primary vortex, as three independent facts about the SHAPE of
        // the solution rather than about any single sample.
        let u_min = u_prof
            .iter()
            .fold((0.0 as Scalar, 0.0 as Scalar), |a, &(y, v)| if v < a.1 { (y, v) } else { a });
        assert!(
            (u_min.1 - -0.2109).abs() < 0.03,
            "the back-flow under the primary vortex peaks at {} (Ghia: -0.2109)",
            u_min.1
        );
        assert!(
            (u_min.0 - 0.4531).abs() < 0.08,
            "that peak sits at y = {} (Ghia: 0.4531)",
            u_min.0
        );

        let v_min = v_prof.iter().fold(0.0 as Scalar, |a, &(_, v)| a.min(v));
        let v_max = v_prof.iter().fold(0.0 as Scalar, |a, &(_, v)| a.max(v));
        assert!(
            (v_min - -0.2453).abs() < 0.03 && (v_max - 0.1753).abs() < 0.03,
            "the vertical velocity spans [{v_min}, {v_max}], Ghia [-0.2453, 0.1753]"
        );

        println!(
            "lid-driven cavity Re=100, {N}x{N}: converged in {iters} SIMPLE \
             iterations, worst |u - Ghia| = {worst_u:.4}, \
             worst |v - Ghia| = {worst_v:.4}, \
             max |sum_f phi| = {:.3e}",
            last.continuity_error
        );

        Ok(())
    }

    // ----------------------------------------------------------------------
    //  Controls
    // ----------------------------------------------------------------------


    // ----------------------------------------------------------------------
    //  SPEC-LIT §14 and the "PISO vs SIMPLE" row of §22
    // ----------------------------------------------------------------------

    /// The cavity, small and upwind, shared by the two algorithm tests.
    ///
    /// Upwind rather than central on purpose: the point of these tests is that
    /// two ALGORITHMS reach the same fixed point, and a scheme whose cell
    /// Peclet number is above two on this mesh would put its own wobble
    /// between them. The accuracy of the discretisation is
    /// `lid_driven_cavity_matches_ghia_ghia_and_shin`'s business.
    fn cavity_mesh(n: usize) -> HostMesh {
        let h: Scalar = 1.0 / n as Scalar;
        boxed(
            [n, n, 1],
            Vec3::new(h, h, h),
            [
                PatchKind::Wall,
                PatchKind::Wall,
                PatchKind::Wall,
                PatchKind::Wall,
                PatchKind::Empty,
                PatchKind::Empty,
            ],
        )
    }

    fn cavity_controls(nu: Scalar) -> SimpleControls {
        SimpleControls {
            momentum: MomentumControls {
                nu,
                u_solver: SolverControls {
                    tolerance: 1e-12,
                    max_iter: 300,
                    check_interval: 5,
                    ..SolverControls::default()
                },
                div_scheme: crate::io::case::DivScheme::Upwind,
                bounded_convection: true,
                sn_grad: crate::fv::SnGradScheme::Uncorrected,
                variable_viscosity_stress: false,
                ..MomentumControls::default()
            },
            p_solver: SolverControls {
                tolerance: 1e-12,
                max_iter: 800,
                check_interval: 5,
                ..SolverControls::default()
            },
            ..SimpleControls::default()
        }
    }

    /// Put the lid on and evaluate the boundaries.
    fn seat_cavity(g: &Gpu, s: &mut Simple<'_>, hm: &HostMesh) -> Result<()> {
        let lid = Vec3::new(1.0, 0.0, 0.0);
        set_velocity_bcs(
            g,
            s.u_mut(),
            hm,
            &[
                Some(Vec3::ZERO),
                Some(Vec3::ZERO),
                Some(Vec3::ZERO),
                Some(lid),
                None,
                None,
            ],
        )?;
        set_scalar_bcs(g, s.p_mut(), hm, &[None; 6])?;
        s.initialise(g)
    }

    /// March a steady run to convergence and return the velocity.
    fn run_steady(
        g: &Gpu,
        hm: &HostMesh,
        m: &GpuMesh,
        nu: Scalar,
        alpha_u: Scalar,
        nut: &GpuScalarField,
        t: &GpuScalarField,
    ) -> Result<Vec<Vec3>> {
        let mut ctrl = cavity_controls(nu);
        ctrl.momentum.u_relax = alpha_u;
        ctrl.p_relax = 0.3;
        ctrl.momentum.steady = true;
        ctrl.momentum.ddt = crate::timescheme::DdtScheme::SteadyState;

        let mut s = Simple::new(g, hm, m, ctrl, BuoyancyCoeffs {
            g: Vec3::ZERO,
            ..BuoyancyCoeffs::default()
        })?;
        seat_cavity(g, &mut s, hm)?;

        let mut backend = PbicgstabBackend::new(ctrl.p_solver);
        backend.setup(g, hm, m, &SystemProbe::default())?;

        let mut done = 0usize;
        for i in 0..4000 {
            let perf = s.correct(g, &mut backend, nut, t)?;
            done = i + 1;
            let worst = perf
                .u
                .iter()
                .map(|q| q.initial_residual)
                .fold(0.0 as Scalar, Scalar::max);
            if i > 30 && worst < 1e-11 {
                break;
            }
        }
        assert!(done < 4000, "SIMPLE (alpha_U = {alpha_u}) did not converge");
        Ok(g.download(&s.u().f)?)
    }

    /// March a transient run until the field stops moving and return the
    /// velocity. `n_outer > 1` makes it PIMPLE and turns relaxation on for
    /// every outer corrector but the last; `n_outer == 1` makes it PISO.
    #[allow(clippy::too_many_arguments)]
    fn run_transient(
        g: &Gpu,
        hm: &HostMesh,
        m: &GpuMesh,
        nu: Scalar,
        dt: Scalar,
        n_correctors: usize,
        n_outer: usize,
        nut: &GpuScalarField,
        t: &GpuScalarField,
    ) -> Result<Vec<Vec3>> {
        let mut ctrl = cavity_controls(nu);
        ctrl.momentum.steady = false;
        ctrl.momentum.ddt = crate::timescheme::DdtScheme::Euler;
        ctrl.momentum.delta_t = dt;
        ctrl.n_correctors = n_correctors;
        ctrl.n_outer_correctors = n_outer;
        // PISO is a non-iterative splitting and relaxing it destroys the time
        // accuracy that justifies it (SPEC-LIT §14); PIMPLE relaxes every
        // outer corrector but the last, and `correct_outer` switches it off
        // there.
        if n_outer == 1 {
            ctrl.momentum.u_relax = 1.0;
            ctrl.p_relax = 1.0;
        } else {
            ctrl.momentum.u_relax = 0.7;
            ctrl.p_relax = 0.3;
        }

        let mut s = Simple::new(g, hm, m, ctrl, BuoyancyCoeffs {
            g: Vec3::ZERO,
            ..BuoyancyCoeffs::default()
        })?;
        seat_cavity(g, &mut s, hm)?;

        let mut backend = PbicgstabBackend::new(ctrl.p_solver);
        backend.setup(g, hm, m, &SystemProbe::default())?;

        let mut prev = g.download(&s.u().f)?;
        let mut steps = 0usize;
        for i in 0..3000 {
            s.begin_time_step(g, dt)?;
            s.solve_step(g, &mut backend, nut, t)?;
            steps = i + 1;

            // Every twenty steps, ask whether the field is still moving. A
            // transient run has no residual that means "converged to a steady
            // state"; the change per step is the only thing that does.
            if i % 20 == 19 {
                let now = g.download(&s.u().f)?;
                let moved = now
                    .iter()
                    .zip(&prev)
                    .fold(0.0 as Scalar, |w, (a, b)| w.max((*a - *b).mag()));
                prev = now;
                if moved < 1e-10 {
                    break;
                }
            }
        }
        assert!(steps < 3000, "the transient run had not settled after {steps} steps");
        Ok(g.download(&s.u().f)?)
    }

    fn worst_difference(a: &[Vec3], b: &[Vec3]) -> Scalar {
        a.iter()
            .zip(b)
            .fold(0.0 as Scalar, |w, (x, y)| w.max((*x - *y).mag()))
    }

    /// SPEC-LIT §22: PISO and SIMPLE converge to the same steady answer.
    ///
    /// # What "the same" can mean here, and what it cannot
    ///
    /// The two are different splittings of one system, and at the fixed point
    /// the splitting error and the relaxation both multiply zero - so the
    /// MOMENTUM equation they converge to is identical. The flux is not.
    /// SPEC-LIT §5.1 builds it as
    ///
    /// ```text
    /// phi = interp(HbyA)·Sf + rAU_f(b_f·Sf) - rAU_f|Sf|snGrad(p)
    /// ```
    ///
    /// and substituting the converged `HbyA = U - rAU·reconstruct(forceFlux)`
    /// leaves
    ///
    /// ```text
    /// phi = interp(U)·Sf + [ rAU_f·forceFlux_f
    ///                        - interp(rAU·reconstruct(forceFlux))·Sf ]
    /// ```
    ///
    /// The bracket is the Rhie-Chow term itself - the whole reason the flux is
    /// not simply `interp(U)·Sf` - and it is PROPORTIONAL TO `rAU`. `rAU` is
    /// `V/a_P` of the matrix that was actually solved, so it carries the
    /// implicit relaxation factor (`a_P/alpha_U`, Patankar §4.9) in a steady
    /// run and the time-step term (`a_P + V/dt`) in a transient one. Two runs
    /// with different `alpha_U`, or a steady and a transient run, therefore
    /// converge to fluxes that differ by an O(`h²`) discretisation term - and
    /// so to velocity fields that differ by the same order.
    ///
    /// This is a property of the formulation, not of the corrector loop, and
    /// this test MEASURES it rather than asserting it away: the control is
    /// SIMPLE against ITSELF at two legal relaxation factors. The claim then
    /// is the sharp one -
    ///
    /// **PISO differs from SIMPLE by no more than SIMPLE differs from itself
    /// under a legal change of its own relaxation factor.**
    ///
    /// - if PISO were solving a different equation the gap would be large and
    ///   would not scale with the control;
    /// - if the pressure correctors were wrong the flux would not be
    ///   conservative, which `lid_driven_cavity_matches_ghia_ghia_and_shin`
    ///   asserts on this same problem and `ofgpu-validate` asserts on four
    ///   more;
    /// - and the two flows must be the same flow, which the peak velocity
    ///   check says.
    #[test]
    fn piso_and_simple_reach_the_same_steady_answer() -> Result<()> {
        let Some(g) = gpu() else { return Ok(()) };

        const N: usize = 16;
        let nu: Scalar = 0.01; // Re = U L / nu = 100
        let hm = cavity_mesh(N);
        let m = GpuMesh::upload(&g, &hm)?;

        let t_ref = BuoyancyCoeffs::default().t_ref;
        let nut = laminar_nut(&g, &m, &hm)?;
        let t = temperature(&g, &m, &hm, &vec![t_ref; hm.n_cells], &[Some(t_ref); 6])?;

        // The control: one algorithm, two relaxation factors, both legal.
        let simple_07 = run_steady(&g, &hm, &m, nu, 0.7, &nut, &t)?;
        let simple_04 = run_steady(&g, &hm, &m, nu, 0.4, &nut, &t)?;
        let control = worst_difference(&simple_07, &simple_04);

        // PISO: one outer corrector, two pressure correctors, no relaxation,
        // marched to a steady state.
        let piso = run_transient(&g, &hm, &m, nu, 0.025, 2, 1, &nut, &t)?;

        let gap = worst_difference(&simple_07, &piso);

        assert!(
            control > 0.0,
            "SIMPLE gave the same answer at two relaxation factors, so this \
             test has no scale to measure against and something else is wrong"
        );
        assert!(
            gap <= 2.0 * control,
            "SIMPLE and PISO differ by {gap} of a lid speed, more than twice \
             SIMPLE's own sensitivity to alpha_U ({control}). The two are no \
             longer the same discrete problem"
        );

        // The same flow, not merely a nearby one.
        let peak_s = max_mag(&simple_07);
        let peak_p = max_mag(&piso);
        assert!(peak_s > 0.2, "the cavity never got moving: peak |U| = {peak_s}");
        assert!(
            (peak_s - peak_p).abs() < 1e-3,
            "peak |U|: SIMPLE {peak_s}, PISO {peak_p}"
        );

        Ok(())
    }

    /// SPEC-LIT §14: "with `nOuterCorrectors = 1` and no relaxation PIMPLE is
    /// exactly PISO".
    ///
    /// This is the sharp half of the comparison above. PIMPLE and PISO at the
    /// SAME time step have the same `rAU` on the iteration that sets the
    /// answer - PIMPLE's last outer corrector runs unrelaxed - so the
    /// Rhie-Chow term is identical and the two steady states must agree to
    /// solver tolerance, not merely to discretisation error.
    ///
    /// It is also the test that exercises the outer loop end to end: four
    /// outer correctors, each re-linearising momentum, each running two
    /// pressure correctors, with relaxation switched off on the fourth.
    #[test]
    fn pimple_and_piso_reach_the_same_steady_answer() -> Result<()> {
        let Some(g) = gpu() else { return Ok(()) };

        // Coarser than the SIMPLE comparison above, on purpose. What is being
        // checked here is an IDENTITY - two loops that reduce to the same
        // arithmetic on the iteration that sets the answer - and an identity
        // does not need a resolved flow to show itself, so this runs on the
        // smallest mesh that shows it.
        const N: usize = 8;
        let nu: Scalar = 0.01;
        let hm = cavity_mesh(N);
        let m = GpuMesh::upload(&g, &hm)?;

        let t_ref = BuoyancyCoeffs::default().t_ref;
        let nut = laminar_nut(&g, &m, &hm)?;
        let t = temperature(&g, &m, &hm, &vec![t_ref; hm.n_cells], &[Some(t_ref); 6])?;

        // Courant ~ 0.4 at the lid: h = 1/8, U = 1.
        let dt: Scalar = 0.05;
        let piso = run_transient(&g, &hm, &m, nu, dt, 2, 1, &nut, &t)?;
        let pimple = run_transient(&g, &hm, &m, nu, dt, 2, 4, &nut, &t)?;

        let gap = worst_difference(&piso, &pimple);
        assert!(
            gap < 1e-3,
            "PIMPLE and PISO reached different steady states, {gap} apart"
        );
        assert!(max_mag(&piso) > 0.2, "the cavity never got moving");

        Ok(())
    }

    /// The time-integration bug the audit found: `nCorrectors 2` used to
    /// advance `U` by TWO Euler sub-steps per time step.
    ///
    /// `Simple::correct` refreshed the velocity's old-time level on entry, so
    /// a driver calling it twice per step differenced the second call against
    /// the first call's answer rather than against the start of the step. `T`
    /// and `k`/`epsilon` advanced by one `dt` in the same wall-clock time, and
    /// the fields came apart with nothing in the output saying so.
    ///
    /// What pins it: after a whole time step with several outer correctors and
    /// several pressure correctors, `U^{n-1}` must still be the velocity the
    /// step STARTED from, bit for bit. `begin_time_step` is the only thing
    /// entitled to move it.
    #[test]
    fn a_time_step_stores_one_old_level_however_many_correctors_it_takes() -> Result<()> {
        let Some(g) = gpu() else { return Ok(()) };

        const N: usize = 8;
        let hm = cavity_mesh(N);
        let m = GpuMesh::upload(&g, &hm)?;

        let mut ctrl = cavity_controls(0.01);
        ctrl.momentum.steady = false;
        ctrl.momentum.ddt = crate::timescheme::DdtScheme::Euler;
        ctrl.momentum.delta_t = 0.01;
        ctrl.momentum.u_relax = 1.0;
        ctrl.p_relax = 1.0;
        // Both loops on: three outer correctors, two pressure correctors each.
        ctrl.n_outer_correctors = 3;
        ctrl.n_correctors = 2;

        let mut s = Simple::new(&g, &hm, &m, ctrl, BuoyancyCoeffs {
            g: Vec3::ZERO,
            ..BuoyancyCoeffs::default()
        })?;
        seat_cavity(&g, &mut s, &hm)?;

        let mut backend = PbicgstabBackend::new(ctrl.p_solver);
        backend.setup(&g, &hm, &m, &SystemProbe::default())?;

        // One step first, so the field is not uniformly zero and a stale
        // old-time level would be visibly different from a fresh one.
        s.begin_time_step(&g, ctrl.momentum.delta_t)?;
        s.solve_step(&g, &mut backend, &nut_for(&g, &m, &hm)?, &t_for(&g, &m, &hm)?)?;

        let at_step_start = g.download(&s.u().f)?;

        s.begin_time_step(&g, ctrl.momentum.delta_t)?;
        let f0_after_rotation = g.download(&s.u().f0)?;
        assert_eq!(
            f0_after_rotation, at_step_start,
            "begin_time_step must make U^(n-1) the velocity the step starts from"
        );

        s.solve_step(&g, &mut backend, &nut_for(&g, &m, &hm)?, &t_for(&g, &m, &hm)?)?;

        let f0_after_step = g.download(&s.u().f0)?;
        assert_eq!(
            f0_after_step, at_step_start,
            "the correctors moved U^(n-1): the step advanced U by more than one dt"
        );

        // And the field really did move, so the check is not vacuous.
        let now = g.download(&s.u().f)?;
        let moved = now
            .iter()
            .zip(&at_step_start)
            .fold(0.0 as Scalar, |w, (a, b)| w.max((*a - *b).mag()));
        assert!(moved > 0.0, "the step changed nothing, so nothing was tested");

        Ok(())
    }

    fn nut_for(g: &Gpu, m: &GpuMesh, hm: &HostMesh) -> Result<GpuScalarField> {
        laminar_nut(g, m, hm)
    }

    fn t_for(g: &Gpu, m: &GpuMesh, hm: &HostMesh) -> Result<GpuScalarField> {
        let t_ref = BuoyancyCoeffs::default().t_ref;
        temperature(g, m, hm, &vec![t_ref; hm.n_cells], &[Some(t_ref); 6])
    }

    /// A steady run never switches its relaxation off, however many outer
    /// correctors it is given - SPEC-LIT §14's note that relaxation is what
    /// replaces the time derivative there.
    #[test]
    fn a_steady_run_keeps_its_relaxation_on_every_iteration() -> Result<()> {
        let Some(g) = gpu() else { return Ok(()) };

        let hm = cavity_mesh(6);
        let m = GpuMesh::upload(&g, &hm)?;

        let mut ctrl = cavity_controls(0.01);
        ctrl.momentum.steady = true;
        ctrl.momentum.ddt = crate::timescheme::DdtScheme::SteadyState;
        ctrl.momentum.u_relax = 0.7;
        ctrl.n_outer_correctors = 4;

        let mut s = Simple::new(&g, &hm, &m, ctrl, BuoyancyCoeffs {
            g: Vec3::ZERO,
            ..BuoyancyCoeffs::default()
        })?;
        seat_cavity(&g, &mut s, &hm)?;

        let mut backend = PbicgstabBackend::new(ctrl.p_solver);
        backend.setup(&g, &hm, &m, &SystemProbe::default())?;

        s.solve_step(&g, &mut backend, &nut_for(&g, &m, &hm)?, &t_for(&g, &m, &hm)?)?;

        assert_eq!(
            s.momentum().relaxation(),
            0.7,
            "the relaxation factor must be back to the case's value"
        );
        Ok(())
    }

    /// `nCorrectors 0` and `nOuterCorrectors 0` are refused rather than
    /// quietly turned into one - SPEC-LIT §13.4.
    #[test]
    fn a_zero_corrector_count_is_refused() {
        let mut c = SimpleControls { n_correctors: 0, ..SimpleControls::default() };
        assert!(c.validate().is_err());
        c = SimpleControls { n_outer_correctors: 0, ..SimpleControls::default() };
        assert!(c.validate().is_err());
        assert!(SimpleControls::default().validate().is_ok());
    }

    #[test]
    fn a_pressure_relaxation_outside_the_valid_range_is_refused() {
        let bad = SimpleControls { p_relax: 0.0, ..SimpleControls::default() };
        assert!(bad.validate().is_err());
        let bad = SimpleControls { p_relax: 1.2, ..SimpleControls::default() };
        assert!(bad.validate().is_err());
        assert!(SimpleControls::default().validate().is_ok());
    }

    /// SPEC-LIT §5.2 recommends `alpha_p ≈ 1 - alpha_U`; the defaults are that
    /// pair, and a change to one that forgets the other should be visible.
    #[test]
    fn the_default_relaxation_pair_is_the_recommended_one() {
        let c = SimpleControls::default();
        assert!((c.momentum.u_relax - 0.7).abs() < 1e-12);
        assert!((c.p_relax - 0.3).abs() < 1e-12);
        assert!((c.momentum.u_relax + c.p_relax - 1.0).abs() < 1e-12);
    }

    // ----------------------------------------------------------------------
    //  §25.1/§25.3 - the low-Mach pressure source, checked independently of
    //  crate::energy
    // ----------------------------------------------------------------------

    /// A uniform target divergence `S`, fed through
    /// [`Simple::correct_outer_low_mach`], with every patch a wall except ONE
    /// open outlet: the only place the volume `S` generates in every cell can
    /// leave is that outlet, so the net boundary outflow must equal
    /// `S * V_dom` exactly - the discrete statement that SPEC-LIT §25.3's
    /// "the pressure equation's source acquires the target divergence"
    /// actually holds. [`Simple::correct_outer`] (`target_div = None`) is
    /// untouched by this method's existence - every OTHER test in this module
    /// stays green, which is `correct_outer_low_mach` calling
    /// `correct_outer_impl(..., None, ...)` reducing to exactly the code path
    /// those tests already exercise.
    #[test]
    fn a_uniform_target_divergence_drives_the_matching_net_outflow() -> Result<()> {
        let Some(g) = gpu() else { return Ok(()) };

        let hm = boxed([5, 5, 5], Vec3::new(0.1, 0.1, 0.1), [PatchKind::Wall; 6]);
        let m = GpuMesh::upload(&g, &hm)?;

        let ctrl = SimpleControls {
            momentum: MomentumControls {
                nu: 1.0,
                sn_grad: crate::fv::SnGradScheme::Uncorrected,
                variable_viscosity_stress: false,
                ..MomentumControls::default()
            },
            p_solver: tight(),
            ..SimpleControls::default()
        };

        let mut s = Simple::new(&g, &hm, &m, ctrl, BuoyancyCoeffs {
            g: Vec3::ZERO,
            ..BuoyancyCoeffs::default()
        })?;

        // Every patch a wall except patch 1, open: zero-gradient U, fixedValue
        // p = 0 - the one place the generated volume can leave. Which
        // GEOMETRIC face patch 1 is does not matter: the test only needs one
        // consistent open patch, walls everywhere else.
        set_velocity_bcs(
            &g,
            s.u_mut(),
            &hm,
            &[Some(Vec3::ZERO), None, Some(Vec3::ZERO), Some(Vec3::ZERO), Some(Vec3::ZERO), Some(Vec3::ZERO)],
        )?;
        set_scalar_bcs(&g, s.p_mut(), &hm, &[None, Some(0.0), None, None, None, None])?;
        s.initialise(&g)?;
        assert!(
            !s.pressure_is_pinned(),
            "a fixedValue face must be recognised as pinning the level"
        );

        let nut = laminar_nut(&g, &m, &hm)?;
        let t_ref = BuoyancyCoeffs::default().t_ref;
        let t = temperature(&g, &m, &hm, &vec![t_ref; hm.n_cells], &[Some(t_ref); 6])?;

        let s_target: Scalar = 0.5; // 1/s - an arbitrary uniform expansion rate
        let target_div = g.upload(&vec![s_target; hm.n_cells])?;

        let mut backend = PbicgstabBackend::new(ctrl.p_solver);
        backend.setup(&g, &hm, &m, &SystemProbe::default())?;

        for _ in 0..800 {
            s.correct_outer_low_mach(&g, &mut backend, &nut, &t, &target_div, false)?;
        }

        let bf = g.download(&s.phi().bf)?;
        let net_outflow: Scalar = bf.iter().sum();
        let want = s_target * m.total_volume;

        assert!(
            (net_outflow - want).abs() < 1e-3 * want.abs(),
            "net boundary outflow = {net_outflow}, S*V_dom = {want}"
        );
        Ok(())
    }
}
