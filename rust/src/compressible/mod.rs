// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.
// Provenance: see PROVENANCE.md. No GPL-licensed source was consulted.

//! The pressure-based all-speed compressible solver of SPEC-LIT §122 to
//! §127 (`ofgpu-compressible`): one segregated `(U, p, T)` solver in a
//! self-contained module, on the `vof.rs` precedent, with its kernels in a
//! `cuda/compressible.cu` of its own.
//!
//! The module root holds [`Compressible`], the outer loop of §125.3 as built
//! in §125.6 (CMP-12): transient PIMPLE (`nOuterCorrectors >= 1`,
//! `nCorrectors >= 1`) and steady SIMPLE in one loop with two switches, the
//! energy equation inside the outer loop, the old time levels refreshed once
//! per step, the bounds of §122.3 counted (never clipped), and the mass
//! imbalance and the `h_0` residual reported on every outer iteration. Its
//! submodules:
//!
//! - [`thermo`] (§122.1-§122.3, §122.5), the perfect-gas state, CMP-08;
//! - [`momentum`] (§123), momentum in conservative form, CMP-09;
//! - [`pressure`] (§124), the all-speed pressure equation, CMP-10;
//! - [`energy`] (§125.1, §125.5), the energy equation solved for `T`,
//!   CMP-11;
//! - [`exact`] (§127.2), the analytic answer keys, CMP-06, with
//!   `exact/riemann.rs`, the exact Riemann solver, Ringleb flow (§127.3)
//!   and compressible Couette flow (§127.4), CMP-07;
//! - [`gatemesh`] (§127.5), the gate meshes - with CMP-26, the airfoil
//!   C-mesh of §127.5.1;
//! - `tests`, the outer loop's gate table (CMP-12);
//! - still to come: `bc` (§126), the compressible boundary conditions,
//!   CMP-16 to CMP-18; `lts` (§125.4), the local time step of a steady run,
//!   CMP-13.
//!
//! Written from:
//!   ofgpu SPEC-LIT.md §125.3 - the outer loop, its order and its reports,
//!     which is the contract of [`Compressible::step`]
//!   ofgpu SPEC-LIT.md §125.6 - the loop as built (CMP-12)
//!   ofgpu SPEC-LIT.md §122.3 - bounds are counted, never clipped
//!   ofgpu SPEC-LIT.md §93.5 - (93.7)'s `M_max` and `M_mean`
//!   ofgpu SPEC-LIT.md §14, §13.3 - PIMPLE's final corrector and the
//!     once-per-step rotation of the time levels
//!
//! Provenance: ORIGINAL - the loop is this tree's own composition of the
//! module's equations, in the order §125.3 states; it carries no numerics
//! of its own beyond the explicit relaxation of `p` (§125.3 step 6, the
//! `smpRelaxField` kernel `simple.rs` uses). No GPL-licensed source was
//! consulted.

pub mod energy;
pub mod exact;
pub mod gatemesh;
pub mod momentum;
pub mod pressure;
pub mod thermo;

#[cfg(test)]
mod tests;

use cudarc::driver::{CudaFunction, PushKernelArg};

use crate::device::{cfg_for, DevBuf, Gpu, KernelSet};
use crate::error::{Error, Result};
use crate::field::{GpuScalarField, GpuSurfaceScalarField, GpuVectorField};
use crate::field_ops::{self, FieldKernels};
use crate::fv::{self, FvKernels};
use crate::mesh::GpuMesh;
use crate::solver::SolverPerformance;
use crate::{Label, Scalar};

use energy::{EnergyInputs, RhoEnergy, RhoEnergyControls};
use momentum::{RhoMomentum, RhoMomentumControls};
use pressure::{RhoPressure, RhoPressureControls};
use thermo::{GasConstants, PerfectGas, ThermoKernels};

/// A refusal in the one house format.
fn refuse(who: &str, msg: String) -> Error {
    Error::Refused(format!("compressible::{who}: {msg}"))
}

/// The build's `Scalar` widened to the f64 the reports are formed in.
#[allow(clippy::useless_conversion)]
fn w64(x: Scalar) -> f64 {
    f64::from(x)
}

// ==========================================================================
//  Controls
// ==========================================================================

/// §14's `residualControl` of a steady run: the run stops once every
/// INITIAL residual of an outer iteration is below its bar. A bar of 0
/// never stops.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ResidualControl {
    /// The bar on each velocity component's initial residual.
    pub u: Scalar,
    /// The bar on the first pressure corrector's initial residual.
    pub p: Scalar,
    /// The bar on the energy equation's initial residual (the `h_0`
    /// residual).
    pub t: Scalar,
}

/// Everything the outer loop reads out of a case.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CompressibleControls {
    /// The gas of §122.1. `energy.cp` is never read: `c_p` follows from
    /// `gamma` and `W` (122.5), and [`Compressible::new`] writes it.
    pub gas: PerfectGas,
    /// The momentum equation; its `u_relax` is `alpha_U` of step 2.
    pub momentum: RhoMomentumControls,
    /// The pressure equation; its `n_non_orth_corr` is step 5's inner loop.
    pub pressure: RhoPressureControls,
    /// The energy equation; its `t_relax` is `alpha_T` of step 3. Its
    /// `ddt`, `delta_t` and `g` must equal the momentum equation's.
    pub energy: RhoEnergyControls,
    /// `nOuterCorrectors`, at least 1. A steady run takes one per SIMPLE
    /// iteration.
    pub n_outer: usize,
    /// `nCorrectors`, at least 1.
    pub n_corr: usize,
    /// `alpha_p` of step 6, in (0, 1]: explicit relaxation of `p`, applied
    /// in a steady run and on the non-final outer iterations of a transient
    /// one.
    pub p_relax: Scalar,
    /// The steady run's stop (step 8); `None` runs every iteration asked.
    pub residual_control: Option<ResidualControl>,
    /// Print [`OuterReport::line`] after every outer iteration.
    pub log: bool,
}

impl Default for CompressibleControls {
    fn default() -> Self {
        Self {
            gas: PerfectGas::air(1.0e5),
            momentum: RhoMomentumControls::default(),
            pressure: RhoPressureControls::default(),
            energy: RhoEnergyControls::default(),
            n_outer: 1,
            n_corr: 2,
            p_relax: 0.3,
            residual_control: None,
            log: false,
        }
    }
}

// ==========================================================================
//  State and reports
// ==========================================================================

/// The fields of one compressible run. `u`, `p` (gauge, about `p_op`) and
/// `T` carry the caller's boundary conditions; `rho`, `psi`, `mu` and
/// `kappa` are the gas law's and the transport laws' of them, cell AND
/// boundary values formed elementwise from `p` and `T` (their boundary
/// conditions are never evaluated); `mdot` is the face mass flux [kg/s].
pub struct CompressibleState {
    pub u: GpuVectorField,
    pub p: GpuScalarField,
    pub t: GpuScalarField,
    pub rho: GpuScalarField,
    pub psi: GpuScalarField,
    pub mu: GpuScalarField,
    pub kappa: GpuScalarField,
    pub mdot: GpuSurfaceScalarField,
}

impl CompressibleState {
    /// Take the caller's `u`, `p` and `T` and allocate the rest as zeros;
    /// [`Compressible::initialise`] fills them.
    pub fn new(
        gpu: &Gpu,
        m: &GpuMesh,
        u: GpuVectorField,
        p: GpuScalarField,
        t: GpuScalarField,
    ) -> Result<Self> {
        Ok(Self {
            u,
            p,
            t,
            rho: GpuScalarField::zeros(gpu, m, "rho")?,
            psi: GpuScalarField::zeros(gpu, m, "psi")?,
            mu: GpuScalarField::zeros(gpu, m, "mu")?,
            kappa: GpuScalarField::zeros(gpu, m, "kappa")?,
            mdot: GpuSurfaceScalarField::zeros(gpu, m, "mdot")?,
        })
    }
}

/// One out-of-bounds cell (§122.3): the field, the cell and the value.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Violation {
    /// `"T"`, `"p_abs"` (`p + p_op`) or `"rho"`.
    pub field: &'static str,
    pub cell: usize,
    pub value: f64,
}

/// §122.3's count: the cells with `T <= 0`, `p_abs <= 0` or `rho <= 0`
/// (a NaN counts, because the test is `!(x > 0)`), and the first one in
/// the order T, then p_abs, then rho, each by cell index.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Bounds {
    pub t: usize,
    pub p: usize,
    pub rho: usize,
    pub first: Option<Violation>,
}

impl Bounds {
    /// The number of violations, summed over the three fields.
    pub fn total(&self) -> usize {
        self.t + self.p + self.rho
    }
}

/// What one outer iteration reports (§125.3 step 8).
#[derive(Debug, Clone, PartialEq)]
pub struct OuterReport {
    /// The step this iteration belongs to, from 1.
    pub step: u64,
    /// The outer iteration, from 1.
    pub outer: usize,
    /// True on a transient run's last outer iteration (no relaxation).
    pub is_final: bool,
    /// The three momentum solves.
    pub u: [SolverPerformance; 3],
    /// One pressure solve per corrector (the last non-orthogonal one each).
    pub p: Vec<SolverPerformance>,
    /// The energy solve.
    pub t: SolverPerformance,
    /// `sum_P |ddt(rho)_P + sum_f mdot_f| / sum_P rho_P V_P/dt`, in f64 on
    /// the host after the correctors; a steady run has no `ddt(rho)` and
    /// divides by its `delta_t`.
    pub mass_imbalance: f64,
    /// The initial residual of (125.1).
    pub h0_residual: Scalar,
    /// §122.3's count at the end of the iteration.
    pub bounds: Bounds,
    /// (93.7): `max_P M_P` and `sum_P M_P V_P / sum_P V_P`.
    pub mach_max: f64,
    pub mach_mean: f64,
}

impl OuterReport {
    /// The iteration line.
    pub fn line(&self) -> String {
        let p0 = self.p.first().map(|p| p.initial_residual).unwrap_or(0.0);
        let mut s = format!(
            "step {} outer {}{}: Ux {:.3e} Uy {:.3e} Uz {:.3e} p {:.3e} h0 {:.3e} \
             mass {:.3e} M_max {:.4} M_mean {:.4}",
            self.step,
            self.outer,
            if self.is_final { " (final)" } else { "" },
            self.u[0].initial_residual,
            self.u[1].initial_residual,
            self.u[2].initial_residual,
            p0,
            self.h0_residual,
            self.mass_imbalance,
            self.mach_max,
            self.mach_mean,
        );
        if self.bounds.total() > 0 {
            s.push_str(&format!(
                " BOUNDS {} (T {}, p_abs {}, rho {})",
                self.bounds.total(),
                self.bounds.t,
                self.bounds.p,
                self.bounds.rho
            ));
        }
        s
    }
}

/// One step's outer iterations, and whether the residual control was met.
#[derive(Debug, Clone, PartialEq)]
pub struct StepReport {
    pub step: u64,
    pub outers: Vec<OuterReport>,
    /// The last outer iteration met [`CompressibleControls::residual_control`].
    pub converged: bool,
}

/// What [`Compressible::run`] did.
#[derive(Debug, Clone, PartialEq)]
pub struct RunReport {
    /// The steps taken by this call.
    pub steps: usize,
    /// The last step's report.
    pub last: Option<StepReport>,
    /// A steady run stopped on its residual control.
    pub converged: bool,
}

// ==========================================================================
//  The solver
// ==========================================================================

/// The mesh's topology on the host, downloaded once in [`Compressible::new`]
/// for the reports of step 8.
struct HostTopology {
    owner: Vec<Label>,
    neighbour: Vec<Label>,
    b_face_cells: Vec<Label>,
    v: Vec<Scalar>,
}

/// The outer loop of §125.3 over the module's three equations.
pub struct Compressible<'m> {
    m: &'m GpuMesh,
    ctrl: CompressibleControls,
    gas: GasConstants,
    p_op: f64,
    /// Steady SIMPLE when the ddt scheme is `steadyState`.
    steady: bool,

    thk: ThermoKernels,
    fvk: FvKernels,
    fldk: FieldKernels,
    relax_field: CudaFunction,
    mul: CudaFunction,

    pub mom: RhoMomentum<'m>,
    pub prs: RhoPressure<'m>,
    pub energy: RhoEnergy<'m>,

    /// `p` before the correctors, for step 6.
    p_prev: DevBuf<Scalar>,
    /// `rho_f` and `u_f . Sf`, for the initial `mdot`.
    rho_face: GpuSurfaceScalarField,
    phi: GpuSurfaceScalarField,

    host: HostTopology,
    /// Steps completed.
    steps: u64,
}

impl<'m> Compressible<'m> {
    /// Build the three equations and everything the loop touches. Refused by
    /// name: `n_outer` or `n_corr` of 0, `p_relax` outside (0, 1], a gas
    /// `PerfectGas::validate` refuses, and a momentum and energy equation
    /// that disagree on `ddt`, `delta_t` or `g`.
    pub fn new(gpu: &Gpu, m: &'m GpuMesh, mut ctrl: CompressibleControls) -> Result<Self> {
        let who = "Compressible::new";
        if ctrl.n_outer == 0 {
            return Err(refuse(
                who,
                "nOuterCorrectors is 0; the outer loop runs at least once \
                 (SPEC-LIT §125.3). Alternative: 1 for PISO-like stepping or a \
                 SIMPLE iteration"
                    .to_string(),
            ));
        }
        if ctrl.n_corr == 0 {
            return Err(refuse(
                who,
                "nCorrectors is 0; without a pressure corrector mdot, u and rho \
                 are never corrected (SPEC-LIT §125.3 step 5). Alternative: 1 or 2"
                    .to_string(),
            ));
        }
        if !(ctrl.p_relax > 0.0 && ctrl.p_relax <= 1.0) {
            return Err(refuse(
                who,
                format!(
                    "p_relax is {}; explicit relaxation of p needs 0 < alpha_p <= 1 \
                     (SPEC-LIT §125.3 step 6)",
                    ctrl.p_relax
                ),
            ));
        }
        ctrl.gas.validate()?;
        let (mc, ec) = (&ctrl.momentum, &ctrl.energy);
        if mc.ddt != ec.ddt {
            return Err(refuse(
                who,
                format!(
                    "ddtSchemes differ: momentum {:?}, energy {:?}; one time step \
                     advances both equations (SPEC-LIT §125.3). Alternative: name \
                     one scheme for both",
                    mc.ddt, ec.ddt
                ),
            ));
        }
        if mc.delta_t != ec.delta_t {
            return Err(refuse(
                who,
                format!(
                    "delta_t differs: momentum {}, energy {}; one time step \
                     advances both equations (SPEC-LIT §125.3)",
                    mc.delta_t, ec.delta_t
                ),
            ));
        }
        if mc.g != ec.g {
            return Err(refuse(
                who,
                format!(
                    "g differs: momentum ({}, {}, {}), energy ({}, {}, {}); the \
                     body force and its work are one gravity (SPEC-LIT §125.1)",
                    mc.g.x, mc.g.y, mc.g.z, ec.g.x, ec.g.y, ec.g.z
                ),
            ));
        }
        // c_p is never an input (§122.1): the energy equation divides by
        // the gas's own.
        ctrl.energy.cp = ctrl.gas.cp();

        let n = m.n_cells;
        let one = |k: usize| k.max(1);
        let si = KernelSet::new(gpu, crate::kernels::SIMPLE)?;
        let mo = KernelSet::new(gpu, crate::kernels::MOMENTUM)?;

        let host = HostTopology {
            owner: gpu.download(&m.owner)?,
            neighbour: gpu.download(&m.neighbour)?,
            b_face_cells: gpu.download(&m.b_face_cells)?,
            v: gpu.download(&m.v)?,
        };

        Ok(Self {
            m,
            ctrl,
            gas: ctrl.gas.constants(),
            p_op: ctrl.gas.p_op,
            steady: ctrl.momentum.ddt.is_steady(),

            thk: ThermoKernels::new(gpu)?,
            fvk: FvKernels::new(gpu)?,
            fldk: FieldKernels::new(gpu)?,
            relax_field: si.func("smpRelaxField")?,
            mul: mo.func("momMul")?,

            mom: RhoMomentum::new(gpu, m, ctrl.momentum)?,
            prs: RhoPressure::new(gpu, m, ctrl.pressure)?,
            energy: RhoEnergy::new(gpu, m, ctrl.energy)?,

            p_prev: gpu.zeros(one(n))?,
            rho_face: GpuSurfaceScalarField::zeros(gpu, m, "rhoF")?,
            phi: GpuSurfaceScalarField::zeros(gpu, m, "phiU")?,

            host,
            steps: 0,
        })
    }

    pub fn controls(&self) -> &CompressibleControls {
        &self.ctrl
    }

    /// Steps completed since [`Self::new`].
    pub fn steps_taken(&self) -> u64 {
        self.steps
    }

    // ---- the gas law and the transport laws, cells and boundary faces ----

    /// (122.10) on cells and boundary faces: `psi` from `T`, and
    /// `rho = psi (p + p_op)` - step 4's `psi` and the `rho*` CMP-10's
    /// corrector needs. `p`'s and `T`'s boundary values must be evaluated.
    fn gas_state(&self, gpu: &Gpu, s: &mut CompressibleState) -> Result<()> {
        let (n, nbf) = (self.m.n_cells, self.m.n_boundary_faces);
        let CompressibleState { psi, rho, p, t, .. } = s;
        thermo::gas_state(gpu, &self.thk, &self.gas, &mut psi.f, &mut rho.f, &p.f, &t.f, n)?;
        thermo::gas_state(gpu, &self.thk, &self.gas, &mut psi.bf, &mut rho.bf, &p.bf, &t.bf, nbf)
    }

    /// (122.11) on cells and boundary faces: `mu` and `kappa` from `T`.
    fn transport(&self, gpu: &Gpu, s: &mut CompressibleState) -> Result<()> {
        let (n, nbf) = (self.m.n_cells, self.m.n_boundary_faces);
        let CompressibleState { mu, kappa, t, .. } = s;
        thermo::transport(gpu, &self.thk, &self.gas, &mut mu.f, &mut kappa.f, &t.f, n)?;
        thermo::transport(gpu, &self.thk, &self.gas, &mut mu.bf, &mut kappa.bf, &t.bf, nbf)
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

    /// The start of a run: `u`'s, `p`'s and `T`'s boundary values; `mu`,
    /// `kappa`, `psi` and `rho` from them; `mdot = rho_f (u_f . Sf)` with
    /// `rho_f` linear; every old level seeded from the current field.
    pub fn initialise(&mut self, gpu: &Gpu, s: &mut CompressibleState) -> Result<()> {
        let m = self.m;
        field_ops::correct_boundary_conditions_vector(gpu, &self.fldk, &mut s.u, m)?;
        field_ops::correct_boundary_conditions(gpu, &self.fldk, &mut s.p, m)?;
        field_ops::correct_boundary_conditions(gpu, &self.fldk, &mut s.t, m)?;
        self.transport(gpu, s)?;
        self.gas_state(gpu, s)?;

        fv::interpolate_vector_flux(gpu, &self.fvk, &mut self.phi, &s.u, m)?;
        fv::interpolate_linear(gpu, &self.fvk, &mut self.rho_face, &s.rho, m)?;
        let Self { mul, rho_face, phi, .. } = self;
        Self::launch_mul(gpu, mul, &mut s.mdot.f, &rho_face.f, &phi.f, m.n_internal_faces)?;
        Self::launch_mul(gpu, mul, &mut s.mdot.bf, &rho_face.bf, &phi.bf, m.n_boundary_faces)?;

        for f in [&mut s.rho, &mut s.p, &mut s.t] {
            field_ops::seed_old_time(gpu, &self.fldk, f)?;
        }
        field_ops::seed_old_time_vector(gpu, &self.fldk, &mut s.u)
    }

    // ---- the loop ---------------------------------------------------------

    /// One time step of a transient run, or one SIMPLE iteration of a
    /// steady one: §125.3's loop as §125.6 builds it. Errs with
    /// `Error::Refused` at the end of the first outer iteration whose bounds
    /// count (§122.3) is not zero, leaving every field as it stands.
    pub fn step(&mut self, gpu: &Gpu, s: &mut CompressibleState) -> Result<StepReport> {
        let m = self.m;
        let steady = self.steady;

        // Refresh the old levels ONCE, at the top of the step: close the
        // step just taken in both time terms (not before the first), and
        // rotate rho, u, T and p (backward reads ^oo; K^o is |u^o|^2/2,
        // formed from u^o by the energy source).
        if self.steps > 0 {
            let dt = self.mom.ddt.state.dt;
            self.mom.ddt.advance(dt);
            self.energy.ddt.advance(dt);
        }
        for f in [&mut s.rho, &mut s.p, &mut s.t] {
            field_ops::advance_time_levels(gpu, &self.fldk, f)?;
        }
        field_ops::advance_time_levels_vector(gpu, &self.fldk, &mut s.u)?;
        let step = self.steps + 1;

        let n_outer = self.ctrl.n_outer;
        let mut outers = Vec::with_capacity(n_outer);
        for outer in 1..=n_outer {
            let is_final = !steady && outer == n_outer;

            // 1. mu and kappa from T (psi follows at step 4).
            self.transport(gpu, s)?;

            // 2. The momentum predictor, relaxed unless final.
            self.mom.set_u_relax(if is_final { 1.0 } else { self.ctrl.momentum.u_relax })?;
            self.mom.set_viscosity(gpu, &s.u, &s.mu)?;
            self.mom.set_force(gpu, &s.p, &s.u, &s.rho)?;
            let perf_u = self.mom.solve(gpu, &mut s.u, &s.mdot, &s.rho)?;

            // 3. Energy, relaxed unless final.
            self.energy
                .set_t_relax(if is_final { 1.0 } else { self.ctrl.energy.t_relax })?;
            let perf_t = {
                let inp = EnergyInputs {
                    u: &s.u,
                    p: &s.p,
                    mdot: &s.mdot,
                    rho: &s.rho,
                    mu_eff: &s.mu,
                    kappa_eff: &s.kappa,
                };
                self.energy.solve(gpu, &mut s.t, &inp)?
            };

            // 4. psi from the new T, and rho* = psi (p* + p_op).
            self.gas_state(gpu, s)?;

            // 5. The correctors, H re-evaluated in each.
            field_ops::copy_field(gpu, &self.fldk, &mut self.p_prev, &s.p.f, m.n_cells)?;
            let mut perf_p = Vec::with_capacity(self.ctrl.n_corr);
            for _corr in 0..self.ctrl.n_corr {
                self.mom.rhie_chow(gpu, &s.u, &s.rho)?;
                perf_p.push(self.prs.solve(gpu, &self.mom, &mut s.p, &s.rho, &s.psi)?);
                self.prs.correct(
                    gpu,
                    &mut self.mom,
                    &mut s.p,
                    &mut s.u,
                    &mut s.rho,
                    &s.psi,
                    &mut s.mdot,
                )?;
            }

            // The mass imbalance, on the corrected rho and mdot.
            let mass_imbalance = self.mass_imbalance(gpu, s)?;

            // 6. p relaxed (steady, and the non-final transient outer
            // iterations), its boundary evaluated, the gas law restored.
            let alpha = self.ctrl.p_relax;
            if (steady || !is_final) && alpha < 1.0 {
                self.relax_p(gpu, &mut s.p, alpha)?;
                field_ops::correct_boundary_conditions(gpu, &self.fldk, &mut s.p, m)?;
                self.gas_state(gpu, s)?;
            }

            // 7. Turbulence: none yet (SA is CMP-14); the gas is laminar.

            // 8. Bounds, residuals, the report.
            let (bounds, mach_max, mach_mean) = self.bounds_and_mach(gpu, s)?;
            let rep = OuterReport {
                step,
                outer,
                is_final,
                u: perf_u,
                p: perf_p,
                t: perf_t,
                mass_imbalance,
                h0_residual: perf_t.initial_residual,
                bounds,
                mach_max,
                mach_mean,
            };
            if self.ctrl.log {
                println!("{}", rep.line());
            }
            if let Some(v) = bounds.first {
                self.steps = step;
                return Err(refuse(
                    "Compressible::step",
                    format!(
                        "{} cell(s) out of bounds at the end of outer iteration {outer} \
                         of step {step} (T <= 0: {}, p_abs <= 0: {}, rho <= 0: {}); the \
                         first is {} at cell {}, {} = {}. The gas law has no state \
                         there, so the run stops, and -permissive does not lift it; \
                         nothing was clipped (SPEC-LIT §122.3)",
                        bounds.total(),
                        bounds.t,
                        bounds.p,
                        bounds.rho,
                        v.field,
                        v.cell,
                        v.field,
                        v.value
                    ),
                ));
            }
            outers.push(rep);
        }

        self.steps = step;
        let converged = match (self.ctrl.residual_control, outers.last()) {
            (Some(rc), Some(last)) => {
                let p0 = last.p.first().map(|p| p.initial_residual).unwrap_or(0.0);
                last.u.iter().all(|u| u.initial_residual < rc.u)
                    && p0 < rc.p
                    && last.h0_residual < rc.t
            }
            _ => false,
        };
        Ok(StepReport { step, outers, converged })
    }

    /// Up to `n_steps` steps; a steady run stops early once its residual
    /// control is met.
    pub fn run(&mut self, gpu: &Gpu, s: &mut CompressibleState, n_steps: usize) -> Result<RunReport> {
        let mut out = RunReport { steps: 0, last: None, converged: false };
        for _ in 0..n_steps {
            let rep = self.step(gpu, s)?;
            out.steps += 1;
            let stop = self.steady && rep.converged;
            out.converged = rep.converged;
            out.last = Some(rep);
            if stop {
                break;
            }
        }
        Ok(out)
    }

    // ---- step 6 and step 8 -----------------------------------------------

    /// `p = p_prev + alpha (p - p_prev)` on the cells (`smpRelaxField`).
    fn relax_p(&mut self, gpu: &Gpu, p: &mut GpuScalarField, alpha: Scalar) -> Result<()> {
        let n = self.m.n_cells;
        if n == 0 {
            return Ok(());
        }
        let nl = n as Label;
        let f = self.relax_field.clone();
        unsafe {
            gpu.stream()
                .launch_builder(&f)
                .arg(&mut p.f)
                .arg(&self.p_prev)
                .arg(&alpha)
                .arg(&nl)
                .launch(cfg_for(n))?;
        }
        Ok(())
    }

    /// §125.3's mass imbalance, in f64 on the host from the corrected
    /// `rho` (three levels) and `mdot`, on the momentum equation's time
    /// coefficients - the ones (124.1) was solved with.
    fn mass_imbalance(&self, gpu: &Gpu, s: &CompressibleState) -> Result<f64> {
        let n = self.m.n_cells;
        let h = &self.host;
        let co = self.mom.ddt.state.coeffs(self.mom.ddt.scheme)?;
        let dt = w64(self.mom.ddt.state.dt);
        let rf = gpu.download(&s.rho.f)?;
        let r0 = gpu.download(&s.rho.f0)?;
        let r2 = gpu.download(&s.rho.f00)?;
        let mf = gpu.download(&s.mdot.f)?;
        let mb = gpu.download(&s.mdot.bf)?;
        let mut e = vec![0.0f64; n];
        let mut den = 0.0f64;
        for i in 0..n {
            let v = w64(h.v[i]);
            e[i] = v * (w64(co.a_n) * w64(rf[i]) + w64(co.a_0) * w64(r0[i]) + w64(co.a_00) * w64(r2[i]));
            den += w64(rf[i]) * v / dt;
        }
        for f in 0..self.m.n_internal_faces {
            e[h.owner[f] as usize] += w64(mf[f]);
            e[h.neighbour[f] as usize] -= w64(mf[f]);
        }
        for b in 0..self.m.n_boundary_faces {
            e[h.b_face_cells[b] as usize] += w64(mb[b]);
        }
        let num: f64 = e.iter().map(|x| x.abs()).sum();
        Ok(if den != 0.0 { num / den } else { num })
    }

    /// §122.3's count and (93.7)'s Mach numbers, from one read-back of
    /// `T`, `p`, `rho` and `u`.
    fn bounds_and_mach(&self, gpu: &Gpu, s: &CompressibleState) -> Result<(Bounds, f64, f64)> {
        let n = self.m.n_cells;
        let tf = gpu.download(&s.t.f)?;
        let pf = gpu.download(&s.p.f)?;
        let rf = gpu.download(&s.rho.f)?;
        let uf = gpu.download(&s.u.f)?;
        let mut b = Bounds::default();
        let (mut ft, mut fp, mut fr) = (None, None, None);
        for i in 0..n {
            let t = w64(tf[i]);
            if !(t > 0.0) {
                b.t += 1;
                ft.get_or_insert(Violation { field: "T", cell: i, value: t });
            }
            let pa = w64(pf[i]) + self.p_op;
            if !(pa > 0.0) {
                b.p += 1;
                fp.get_or_insert(Violation { field: "p_abs", cell: i, value: pa });
            }
            let r = w64(rf[i]);
            if !(r > 0.0) {
                b.rho += 1;
                fr.get_or_insert(Violation { field: "rho", cell: i, value: r });
            }
        }
        b.first = ft.or(fp).or(fr);

        let (mut m_max, mut mv, mut v_sum) = (0.0f64, 0.0f64, 0.0f64);
        for i in 0..n {
            let mach = w64(self.gas.mach(uf[i], tf[i]));
            let v = w64(self.host.v[i]);
            m_max = m_max.max(mach);
            mv += mach * v;
            v_sum += v;
        }
        let m_mean = if v_sum > 0.0 { mv / v_sum } else { 0.0 };
        Ok((b, m_max, m_mean))
    }
}
