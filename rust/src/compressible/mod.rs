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
//! - [`lts`] (§125.4, §125.7), the local time step of a steady run,
//!   CMP-13: `localEuler` on both equations builds them on Euler with a
//!   per-cell step (125.2) attached;
//! - `tests`, the outer loop's gate table (CMP-12);
//! - [`bc`] (§126), the compressible boundary conditions: the inlets of
//!   §126.8 (CMP-16), the outlets of §126.9 (CMP-17) and the
//!   characteristic far-field of §126.10 (CMP-18).
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

pub mod bc;
pub mod energy;
pub mod exact;
pub mod gatemesh;
pub mod lts;
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
use crate::timescheme::DdtScheme;
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
    /// The local time step of §125.4, read only when the momentum and the
    /// energy equation name `localEuler` (§125.7).
    pub lts: lts::LtsControls,
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
            lts: lts::LtsControls::default(),
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
    /// Steady SIMPLE when the ddt scheme is `steadyState` or `localEuler`.
    steady: bool,
    /// `localEuler`: the equations run on Euler with §125.7's local step.
    local: bool,

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
    /// SPEC-LIT §126.8: the compressible inlets, rewritten at the top of
    /// every outer iteration; `None` (no rule) rewrites nothing.
    bcs: Option<bc::CompressibleBcs>,
    /// SPEC-LIT §126.9: the outlets (the supersonic switch of a fixed `p`,
    /// `waveTransmissive`); `None` (no rule) rewrites nothing.
    outlets: Option<bc::CompressibleOutlets>,
    /// SPEC-LIT §126.10: the characteristic far-fields; `None` (no rule)
    /// rewrites nothing.
    farfields: Option<bc::CompressibleFarfields>,
    /// SPEC-LIT §126.11: the free-stream family, if any.
    freestreams: Option<bc::CompressibleFreestreams>,
}

impl<'m> Compressible<'m> {
    /// Build the three equations and everything the loop touches. Refused by
    /// name: `n_outer` or `n_corr` of 0, `p_relax` outside (0, 1], a gas
    /// `PerfectGas::validate` refuses, and a momentum and energy equation
    /// that disagree on `ddt`, `delta_t` or `g`, and under `localEuler` a `cfl`
    /// that `lts::LtsControls::validate` refuses (§125.7).
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
        // §125.7: `localEuler` builds both equations on Euler and attaches
        // the per-cell step; `ctrl` keeps the scheme the case named.
        let local = ctrl.momentum.ddt == DdtScheme::LocalEuler;
        let (mut mom_ctrl, mut energy_ctrl) = (ctrl.momentum, ctrl.energy);
        if local {
            ctrl.lts.validate()?;
            mom_ctrl.ddt = DdtScheme::Euler;
            energy_ctrl.ddt = DdtScheme::Euler;
        }

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

        let mut mom = RhoMomentum::new(gpu, m, mom_ctrl)?;
        let mut energy = RhoEnergy::new(gpu, m, energy_ctrl)?;
        if local {
            mom.attach_local_step(gpu)?;
            energy.attach_local_step(gpu)?;
        }

        Ok(Self {
            m,
            ctrl,
            gas: ctrl.gas.constants(),
            p_op: ctrl.gas.p_op,
            steady: ctrl.momentum.ddt.is_steady(),
            local,

            thk: ThermoKernels::new(gpu)?,
            fvk: FvKernels::new(gpu)?,
            fldk: FieldKernels::new(gpu)?,
            relax_field: si.func("smpRelaxField")?,
            mul: mo.func("momMul")?,

            mom,
            prs: RhoPressure::new(gpu, m, ctrl.pressure)?,
            energy,

            p_prev: gpu.zeros(one(n))?,
            rho_face: GpuSurfaceScalarField::zeros(gpu, m, "rhoF")?,
            phi: GpuSurfaceScalarField::zeros(gpu, m, "phiU")?,

            host,
            steps: 0,
            bcs: None,
            outlets: None,
            farfields: None,
            freestreams: None,
        })
    }

    /// SPEC-LIT §126.8: attach the compressible inlets `bc::lower` found;
    /// an empty list attaches nothing, so the loop is the one without them.
    /// Call before [`Self::initialise`].
    pub fn set_boundary_rules(&mut self, gpu: &Gpu, rules: Vec<bc::BcRule>) -> Result<()> {
        self.bcs = if rules.is_empty() {
            None
        } else {
            Some(bc::CompressibleBcs::new(gpu, self.m, &self.ctrl.gas, rules)?)
        };
        Ok(())
    }

    /// The attached inlet rules (empty when none).
    pub fn boundary_rules(&self) -> &[bc::BcRule] {
        self.bcs.as_ref().map_or(&[], |b| b.rules())
    }

    /// SPEC-LIT §126.9: attach the outlets `bc::lower` found; an empty list
    /// attaches nothing. `waveTransmissive` needs a time step, so it is
    /// refused by name under `steadyState` and `CrankNicolson` (§126.5,
    /// and (120.4c) of SPEC-LIT 120.4 is defined for Euler and backward
    /// only); `localEuler` takes the owner's `dt_P`. Call before
    /// [`Self::initialise`].
    pub fn set_outlet_rules(&mut self, gpu: &Gpu, rules: Vec<bc::OutletRule>) -> Result<()> {
        if let Some(r) = rules.iter().find(|r| r.kind == bc::OutletKind::WaveTransmissive) {
            let ddt = self.ctrl.momentum.ddt;
            let why = match ddt {
                DdtScheme::SteadyState => Some("a steady SIMPLE run without localEuler has no time step"),
                DdtScheme::CrankNicolson(_) => {
                    Some("CrankNicolson weights the spatial operator, and (126.7) is formed for Euler and backward only")
                }
                _ => None,
            };
            if let Some(why) = why {
                return Err(refuse(
                    "Compressible::set_outlet_rules",
                    format!(
                        "patch {}: waveTransmissive is (126.6), a condition in time, and {why} \
                         (ddtSchemes {ddt:?}); name Euler, backward or localEuler (SPEC-LIT §126.5)",
                        r.patch
                    ),
                ));
            }
        }
        self.outlets = if rules.is_empty() {
            None
        } else {
            Some(bc::CompressibleOutlets::new(gpu, self.m, &self.ctrl.gas, rules)?)
        };
        Ok(())
    }

    /// The attached outlet rules (empty when none).
    pub fn outlet_rules(&self) -> &[bc::OutletRule] {
        self.outlets.as_ref().map_or(&[], |o| o.rules())
    }

    /// SPEC-LIT §126.10: attach the characteristic far-fields `bc::lower`
    /// found; an empty list attaches nothing. Call before
    /// [`Self::initialise`].
    pub fn set_farfield_rules(&mut self, gpu: &Gpu, rules: Vec<bc::FarfieldRule>) -> Result<()> {
        self.farfields = if rules.is_empty() {
            None
        } else {
            Some(bc::CompressibleFarfields::new(gpu, self.m, &self.ctrl.gas, rules)?)
        };
        Ok(())
    }

    /// The attached far-field rules (empty when none).
    pub fn farfield_rules(&self) -> &[bc::FarfieldRule] {
        self.farfields.as_ref().map_or(&[], |f| f.rules())
    }

    /// SPEC-LIT §126.11: attach the `supersonicFreestream` and
    /// `freestreamPressure` rules `bc::lower` found; an empty list attaches
    /// nothing. Call before [`Self::initialise`].
    pub fn set_freestream_rules(&mut self, gpu: &Gpu, rules: Vec<bc::FreestreamRule>) -> Result<()> {
        self.freestreams = if rules.is_empty() {
            None
        } else {
            Some(bc::CompressibleFreestreams::new(gpu, self.m, &self.ctrl.gas, rules)?)
        };
        Ok(())
    }

    /// The attached free-stream rules (empty when none).
    pub fn freestream_rules(&self) -> &[bc::FreestreamRule] {
        self.freestreams.as_ref().map_or(&[], |f| f.rules())
    }

    /// Whether any boundary rule of §126.8-§126.11 is attached.
    fn has_boundary_rules(&self) -> bool {
        self.bcs.is_some() || self.outlets.is_some() || self.farfields.is_some() || self.freestreams.is_some()
    }

    /// §126.11, §126.8, §126.10 and §126.9: rewrite the free streams', the
    /// inlets', the far-fields' and then the outlets' triples from the
    /// current state (the free streams first, so an inlet `T` on the same
    /// patch reads the new `u_b`), evaluate their faces, and re-form
    /// `psi` and `rho` on the boundary faces (the cells are not touched).
    /// `wave` is false at the start of a run, which has no step yet: the
    /// `waveTransmissive` faces then keep their set-up value. Nothing
    /// without a rule.
    fn apply_boundary_rules(&mut self, gpu: &Gpu, s: &mut CompressibleState, wave: bool) -> Result<()> {
        if !self.has_boundary_rules() {
            return Ok(());
        }
        if let Some(f) = &self.freestreams {
            f.apply(gpu, &self.fldk, s, self.m)?;
        }
        if let Some(b) = &self.bcs {
            b.apply(gpu, &self.fldk, s, self.m)?;
        }
        if let Some(f) = &self.farfields {
            f.apply(gpu, &self.fldk, s, self.m)?;
        }
        if let Some(o) = &mut self.outlets {
            let time = if wave && o.has_wave() {
                Some(match self.mom.local_step() {
                    Some(l) => bc::WaveTime {
                        c: crate::timescheme::DdtCoeffs { a_n: 1.0, a_0: -1.0, a_00: 0.0 },
                        r_delta_t: Some(&l.r_delta_t),
                    },
                    None => bc::WaveTime { c: self.mom.ddt.face_coeffs("waveTransmissive")?, r_delta_t: None },
                })
            } else {
                None
            };
            o.apply(gpu, &self.fldk, s, self.m, time)?;
        }
        let nbf = self.m.n_boundary_faces;
        let CompressibleState { psi, rho, p, t, .. } = s;
        thermo::gas_state(gpu, &self.thk, &self.gas, &mut psi.bf, &mut rho.bf, &p.bf, &t.bf, nbf)
    }

    pub fn controls(&self) -> &CompressibleControls {
        &self.ctrl
    }

    /// Steps completed since [`Self::new`].
    pub fn steps_taken(&self) -> u64 {
        self.steps
    }

    /// §125.7: the per-cell reciprocal step `r_P = 1/dt_P` of the pseudo
    /// step last taken (the momentum equation's; the energy equation holds
    /// a copy). `None` unless the run names `localEuler`.
    pub fn local_step(&self) -> Option<&DevBuf<Scalar>> {
        self.mom.local_step().map(|l| &l.r_delta_t)
    }

    /// (125.2) on every cell from the cell `u` and `T`, into the momentum
    /// equation's step, then copied into the energy equation's.
    fn update_local_step(&mut self, gpu: &Gpu, s: &CompressibleState) -> Result<()> {
        let (m, gas, cfl) = (self.m, self.gas, self.ctrl.lts.cfl);
        let (Some(lm), Some(le)) = (self.mom.local_step_mut(), self.energy.local_step_mut()) else {
            return Err(refuse(
                "Compressible::step",
                "localEuler is named and an equation has no local step attached".to_string(),
            ));
        };
        lm.update(gpu, m, &gas, cfl, &s.u.f, &s.t.f)?;
        le.copy_from(gpu, lm)
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
    /// `rho_f` linear; every old level seeded from the current field. The
    /// inlets of [`Self::set_boundary_rules`] and the far-fields of
    /// [`Self::set_farfield_rules`], if any, are applied once after the
    /// first gas state (SPEC-LIT §126.8, §126.10), and so are the free
    /// streams of [`Self::set_freestream_rules`] (§126.11).
    pub fn initialise(&mut self, gpu: &Gpu, s: &mut CompressibleState) -> Result<()> {
        let m = self.m;
        field_ops::correct_boundary_conditions_vector(gpu, &self.fldk, &mut s.u, m)?;
        field_ops::correct_boundary_conditions(gpu, &self.fldk, &mut s.p, m)?;
        field_ops::correct_boundary_conditions(gpu, &self.fldk, &mut s.t, m)?;
        self.transport(gpu, s)?;
        self.gas_state(gpu, s)?;
        // §126.8: the inlets from the state just formed, then mu and kappa
        // of the new boundary T.
        if let Some(o) = &mut self.outlets {
            o.store_initial(gpu, &self.fldk, s)?;
        }
        if self.has_boundary_rules() {
            self.apply_boundary_rules(gpu, s, false)?;
            self.transport(gpu, s)?;
        }

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
        // §126.9: the waveTransmissive faces' phi_b^o, once per step.
        if let Some(o) = &mut self.outlets {
            o.store_old_time(gpu, &self.fldk, s)?;
        }
        // §125.7: the pseudo step's own dt_P, once, from the state it
        // starts from.
        if self.local {
            self.update_local_step(gpu, s)?;
        }
        let step = self.steps + 1;

        let n_outer = self.ctrl.n_outer;
        let mut outers = Vec::with_capacity(n_outer);
        for outer in 1..=n_outer {
            let is_final = !steady && outer == n_outer;

            // 0. §126.11, §126.8, §126.10, §126.9: the compressible free
            // streams, inlets, far-fields and outlets from the latest state.
            self.apply_boundary_rules(gpu, s, true)?;

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
    /// coefficients - the ones (124.1) was solved with. Under §125.7's
    /// local step those are `a_n = r_P`, `a_0 = -r_P` per cell, and the
    /// denominator is `sum_P rho_P V_P r_P`.
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
        let rdt = match self.mom.local_step() {
            Some(l) => Some(gpu.download(&l.r_delta_t)?),
            None => None,
        };
        let mut e = vec![0.0f64; n];
        let mut den = 0.0f64;
        for i in 0..n {
            let v = w64(h.v[i]);
            if let Some(r) = &rdt {
                let ri = w64(r[i]);
                e[i] = v * (ri * w64(rf[i]) - ri * w64(r0[i]));
                den += w64(rf[i]) * v * ri;
                continue;
            }
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
