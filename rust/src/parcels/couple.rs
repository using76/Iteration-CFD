// meteor-cfd - Copyright (c) 2026 주식회사 메테오시뮬레이션 (Meteo Simulation Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.

//! Two-way coupling: what the parcels took from the gas, given back to it -
//! SPEC-LIT `SPEC-LIT.md` section 68.
//!
//! S66 moves parcels and never touches a cell field. S67 sorts them onto the
//! total order `(cell, uid)` and gathers per cell without an atomic. This
//! module is the last step: it turns the gathered sums into the two things
//! the gas equations take - an acceleration and a linear sink on momentum,
//! a power density and a linear sink on energy - and hands them over through
//! SPEC-LIT S18's source registries, which know nothing about parcels.
//!
//! # The one idea
//!
//! S66's integrator applies a drag impulse to each parcel. This module hands
//! the gas **the negative of that same number** - not a re-linearisation of
//! it, not a `(m_p/tau_p)(u_p - u)` recomputed at some other velocity, but
//! the quantity the parcel kernel accumulated. So
//!
//! ```text
//!   sum_cells V_P f_P dt  +  sum_parcels n_p imp_p  =  0
//! ```
//!
//! holds to round-off, and holds whatever the sub-step count, the drag law,
//! the added-mass setting or the cell crossings did. The design note this
//! section was written from recommends the re-linearised Patankar split
//! instead (its S2.1); that split is here too, as
//! [`CouplingMode::SemiImplicit`], but posed as an increment about the
//! linearisation point so that it buys diagonal dominance **without changing
//! what was exchanged**.
//!
//! # What is refused, by name
//!
//! * **Evaporation** and therefore every mass source. `dm_p/dt` is
//!   identically zero for both supported physics, so there is no vapour to
//!   put anywhere; [`MassCoupling::from_name`] says so rather than letting a
//!   case believe a sprinkler is wetting the air. A sprinkler that does not
//!   evaporate is not a sprinkler.
//! * **Radiation absorption by droplets** - S68.13. It needs `kappa_p`,
//!   `sigmabar_p`, Mie efficiencies and a face-interpolated `Gamma` in
//!   `radiation.rs`; none of it is here.
//! * **Wall splash** - S68.13. Population growth, and S66 has no capacity
//!   policy for it.
//!
//! Written from:
//!   C. T. Crowe, M. P. Sharma, D. E. Stock, *The particle-source-in-cell
//!     (PSI-CELL) model for gas-droplet flows*, J. Fluids Eng. 99 (1977)
//!     325, DOI `10.1115/1.3448756` - the per-cell source construction
//!   S. V. Patankar, *Numerical Heat Transfer and Fluid Flow*, Hemisphere
//!     (1980), S4.2 - the `S_u + S_p psi` linearisation and `S_p <= 0`
//!   W. E. Ranz, W. R. Marshall, *Evaporation from drops*, Chem. Eng. Prog.
//!     48 (1952) 141 and 173 - `Nu = 2 + 0.6 Re^(1/2) Pr^(1/3)`, whose
//!     sensible-heat half is what (68.8) integrates
//!   S. Elghobashi, *On predicting particle-laden turbulent flows*, Appl.
//!     Sci. Res. 52 (1994) 309, DOI `10.1007/BF00936835` - the coupling map
//!     that says when two-way coupling is required at all
//!   R. C. Theobald, *The effect of nozzle design on the stability and
//!     performance of turbulent water jets*, Fire Safety Journal 4 (1981)
//!     1-13 - the ~90 hose-stream experiments S68.12 reports against
//!   ofgpu `SPEC-LIT.md` S68
//! No GPL-licensed source was consulted. OpenFOAM's `src/lagrangian` tree,
//! which contains the obvious reference implementation of a parcel-to-cell
//! source, was not opened.

use cudarc::driver::{CudaFunction, LaunchConfig, PushKernelArg};

use crate::device::{cfg_for, DevBuf, Gpu, KernelSet};
use crate::energy::EnergySources;
use crate::error::{Error, Result};
use crate::field::GpuVectorField;
use crate::io::contract;
use crate::mesh::GpuMesh;
use crate::momentum::MomentumSources;
use crate::{Scalar, Vec3};

use super::deposit::ParcelDeposition;
use super::{ParcelPhysics, Parcels};

#[cfg(test)]
mod tests;

// ==========================================================================
//  The menu
// ==========================================================================

/// How a deposited source enters its equation - SPEC-LIT S68.6, equation
/// (68.10).
///
/// The three are not three models. They are one deposit, entering the matrix
/// three ways, and the deposit is the same in all three.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CouplingMode {
    /// Nothing is registered. The equation is bit for bit the equation of a
    /// run with no parcels in it - not because the source is zero, but
    /// because no kernel of this module touches its matrix.
    #[default]
    Off,

    /// The whole exchange on the right-hand side: `S_u = f/rho`, `S_p = 0`.
    ///
    /// Conservative to round-off (68.4): what the gas is given is exactly
    /// what the parcels lost. Explicit, so a cell whose parcel loading is
    /// heavy enough that `beta dt/rho >~ 1` can oscillate - which is what
    /// [`Self::SemiImplicit`] is for.
    Explicit,

    /// Patankar's split about the velocity the parcels actually saw:
    /// `S_p = -beta/rho` on the diagonal and `S_u = (f + beta u^n)/rho` on
    /// the right, so that **at `u = u^n` the two are identically
    /// [`Self::Explicit`]**.
    ///
    /// `beta >= 0` by construction (it is a sum of `n_p m_eff (1 - e^-x)`
    /// with `x >= 0`), so `S_p <= 0` with no clamp and no sign branch, and
    /// Patankar's rule is satisfied by the arithmetic rather than by a
    /// guard. What it costs is stated rather than hidden: the momentum the
    /// gas ends the solve with differs from the momentum the parcels lost by
    /// `V_P beta_P (u^{n+1} - u^n) dt`, which is zero at convergence of the
    /// outer iteration and is reported by
    /// [`ParcelCoupling::linearisation_defect`] when it is not.
    SemiImplicit,
}

impl CouplingMode {
    pub const NAMES: &'static [&'static str] = &["off", "explicit", "semiImplicit"];

    pub fn name(self) -> &'static str {
        match self {
            Self::Off => "off",
            Self::Explicit => "explicit",
            Self::SemiImplicit => "semiImplicit",
        }
    }

    /// The `OFC_MODE_*` code `cuda/parcelcouple.cu` switches on.
    pub fn code(self) -> i32 {
        match self {
            Self::Off => 0,
            Self::Explicit => 1,
            Self::SemiImplicit => 2,
        }
    }

    pub fn is_on(self) -> bool {
        self != Self::Off
    }

    pub fn from_name(s: &str) -> Result<Self> {
        match s {
            "off" | "none" => Ok(Self::Off),
            "explicit" => Ok(Self::Explicit),
            "semiImplicit" | "semiimplicit" | "implicit" => Ok(Self::SemiImplicit),
            other => contract::unsupported(
                "parcels/coupling",
                other,
                Self::NAMES,
                "explicit",
                Self::Explicit,
            ),
        }
    }
}

/// Mass exchange between the phases - SPEC-LIT S68.13.
///
/// There is exactly one supported value, and it is an enum rather than an
/// absence for the reason S66's [`ParcelPhysics`] is: a case that asks for
/// evaporation is refused **by name**, with what it would need, instead of
/// running an inert spray and producing a plausible wrong answer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum MassCoupling {
    /// `dm_p/dt = 0`. No species source, no `D_src` contribution to the
    /// pressure equation, no change in `n_p` or `d`.
    #[default]
    None,
}

impl MassCoupling {
    pub const NAMES: &'static [&'static str] = &["none"];

    pub fn name(self) -> &'static str {
        "none"
    }

    pub fn from_name(s: &str) -> Result<Self> {
        match s {
            "none" | "off" => Ok(Self::None),
            "evaporation" | "evaporating" | "species" | "vapour" | "vapor" => {
                contract::unsupported_note(
                    "parcels/massCoupling",
                    s,
                    Self::NAMES,
                    "a mass source between the phases IS evaporation, and evaporation \
                     needs the semi-implicit 3x3 closure, liquid property tables, a \
                     species source that follows a moving spray and the D_src term in \
                     `Energy::target_divergence` - none of which exist. SPEC-LIT S68.13 \
                     names all four. A spray that does not evaporate is not a sprinkler, \
                     and this refusal is what says so",
                    "none",
                    Self::None,
                )
            }
            other => contract::unsupported(
                "parcels/massCoupling",
                other,
                Self::NAMES,
                "none",
                Self::None,
            ),
        }
    }
}

/// Everything a case can say about the coupling - SPEC-LIT S68.6.
///
/// Written out field by field wherever it is built, never closed with
/// `..Default::default()`, per SPEC-LIT S13.4.1(b).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CouplingControls {
    pub momentum: CouplingMode,
    pub energy: CouplingMode,
    /// One value. See [`MassCoupling`].
    pub mass: MassCoupling,
}

impl Default for CouplingControls {
    fn default() -> Self {
        Self {
            momentum: CouplingMode::Explicit,
            energy: CouplingMode::Off,
            mass: MassCoupling::None,
        }
    }
}

impl CouplingControls {
    /// One line for the startup banner - SPEC-LIT S13.4.2.
    pub fn describe(&self) -> String {
        format!(
            "parcels/coupling: momentum={} energy={} mass={} (SPEC-LIT S68)",
            self.momentum.name(),
            self.energy.name(),
            self.mass.name(),
        )
    }
}

// ==========================================================================
//  Kernels
// ==========================================================================

struct CoupleKernels {
    gather: CudaFunction,
    cell_integral: CudaFunction,
}

impl CoupleKernels {
    fn new(gpu: &Gpu) -> Result<Self> {
        let k = KernelSet::new(gpu, crate::kernels::PARCELCOUPLE)?;
        Ok(Self {
            gather: k.func("parcelCoupleGather")?,
            cell_integral: k.func("parcelCoupleCellIntegral")?,
        })
    }
}

// ==========================================================================
//  What comes back to the host
// ==========================================================================

/// A host copy of the coupled fields - what the S68.11 gates read and what
/// an output writer would draw.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct CouplingSnapshot {
    /// `f_P`, N/m3: the drag force density the parcels put on the gas.
    pub force: Vec<Vec3>,
    /// `beta_P`, kg/(m3 s): the momentum exchange rate. `>= 0`.
    pub exchange: Vec<Scalar>,
    /// `q_P`, W/m3: the sensible heat the parcels put into the gas.
    pub heat: Vec<Scalar>,
    /// `alpha_T,P`, W/(m3 K): the heat exchange rate. `>= 0`.
    pub heat_exchange: Vec<Scalar>,
    /// What [`MomentumSources`] is handed, m/s2.
    pub momentum_su: Vec<Vec3>,
    /// What [`MomentumSources`] is handed, 1/s, `<= 0`.
    pub momentum_sp: Vec<Scalar>,
    /// What [`EnergySources`] is handed, W/m3.
    pub energy_q: Vec<Scalar>,
    /// What [`EnergySources`] is handed, W/(m3 K), `<= 0`.
    pub energy_sp: Vec<Scalar>,
}

// ==========================================================================
//  The coupling
// ==========================================================================

/// The per-cell coupled sources, and the two registrations that hand them to
/// the gas - SPEC-LIT S68.
///
/// Owns eight `n_cells` arrays: the four physical deposits (`f`, `beta`,
/// `q`, `alpha_T`) and the four the registries take. The split is not
/// redundancy - the deposits are what S68.9's conservation statement is
/// posed on, and they are what an output writer wants to draw, while the
/// registry fields carry the mode's linearisation and the division by the
/// gas density that the kinematic momentum equation needs.
///
/// Twelve `f64` per cell, plus the three of the read-back scratch (68.4) is
/// posed on: fifteen, 120 B, so a million-cell mesh spends 120 MB here.
/// S68.8 states it rather than discovering it as an OOM.
pub struct ParcelCoupling<'m> {
    m: &'m GpuMesh,
    ctrl: CouplingControls,
    k: CoupleKernels,
    cfg_cells: LaunchConfig,

    /// The physics the pool was built with, remembered so that a mismatched
    /// pool is refused rather than silently reading a length-1 buffer.
    physics: ParcelPhysics,
    capacity: usize,
    dt: Scalar,

    f: DevBuf<Vec3>,
    beta: DevBuf<Scalar>,
    q: DevBuf<Scalar>,
    alpha_t: DevBuf<Scalar>,

    mom_su: DevBuf<Vec3>,
    mom_sp: DevBuf<Scalar>,
    nrg_q: DevBuf<Scalar>,
    nrg_sp: DevBuf<Scalar>,

    /// Scratch for [`Self::total_impulse`]. Diagnostics only.
    integral: DevBuf<Vec3>,
}

impl<'m> ParcelCoupling<'m> {
    /// Build the coupling for one pool.
    ///
    /// Refuses, by name and at setup: energy coupling on inert parcels (an
    /// inert droplet is an infinite heat bath and coupling one would create
    /// energy from nothing); and a mass coupling other than
    /// [`MassCoupling::None`], which cannot be constructed anyway.
    pub fn new(gpu: &Gpu, p: &Parcels<'m>, ctrl: CouplingControls) -> Result<Self> {
        if ctrl.energy.is_on() && p.ctrl.physics != ParcelPhysics::Heating {
            return Err(Error::Config(format!(
                "parcels/coupling: energy coupling is \"{}\" but the parcels are \
                 \"{}\". An inert parcel's temperature never moves, so it would be an \
                 INFINITE heat bath: the gas would relax towards it for ever and the \
                 energy the run gained would come from nowhere. Say `physics heating` \
                 (SPEC-LIT S68.5), or `energy off`",
                ctrl.energy.name(),
                p.ctrl.physics.name(),
            )));
        }

        let n = p.m.n_cells;
        let one = n.max(1);
        Ok(Self {
            m: p.m,
            ctrl,
            k: CoupleKernels::new(gpu)?,
            cfg_cells: cfg_for(one),
            physics: p.ctrl.physics,
            capacity: p.ctrl.capacity,
            dt: p.dt,

            f: gpu.zeros(one)?,
            beta: gpu.zeros(one)?,
            q: gpu.zeros(one)?,
            alpha_t: gpu.zeros(one)?,

            mom_su: gpu.zeros(one)?,
            mom_sp: gpu.zeros(one)?,
            nrg_q: gpu.zeros(one)?,
            nrg_sp: gpu.zeros(one)?,

            integral: gpu.zeros(one)?,
        })
    }

    pub fn controls(&self) -> &CouplingControls {
        &self.ctrl
    }

    /// `15 * 8` bytes per cell - what this object costs before it is paid
    /// for. SPEC-LIT S68.8.
    pub fn device_bytes(&self) -> usize {
        let n = self.m.n_cells.max(1);
        n * (3 * 8 + 8 + 8 + 8 + 3 * 8 + 8 + 8 + 8 + 3 * 8)
    }

    // ---- the step -----------------------------------------------------

    /// The gather of (68.7): one thread per cell, walking the S67 CSR.
    ///
    /// Call **after** [`ParcelDeposition::update`] in the same step, so that
    /// the CSR describes where the parcels ended and the accumulators
    /// describe what they exchanged getting there. Both are true of the same
    /// instant; S68.3 says why depositing into the end-of-step cell is the
    /// only gather-shaped choice and what it costs.
    ///
    /// `t_gas` is required exactly when energy coupling is on, and refused
    /// when it is not.
    // Eight, and the lint's bar is seven. Every one is a distinct object the
    // gather reads and none can be folded into another without this module
    // taking ownership of something that is not its: the pool, the CSR, and
    // the three gas fields all have different owners and different lifetimes.
    #[allow(clippy::too_many_arguments)]
    pub fn update(
        &mut self,
        gpu: &Gpu,
        p: &Parcels<'m>,
        dep: &ParcelDeposition<'m>,
        rho: &DevBuf<Scalar>,
        u_gas: &GpuVectorField,
        t_gas: Option<&DevBuf<Scalar>>,
        dt: Scalar,
    ) -> Result<()> {
        self.check(p)?;
        if dt != self.dt {
            return Err(Error::Config(format!(
                "parcels/coupling: update was given dt = {dt} but the pool was built \
                 for dt = {}. (68.5) accumulated the impulse over THAT step, and \
                 dividing it by another is a silently wrong force (SPEC-LIT S68.3)",
                self.dt
            )));
        }
        let n = self.m.n_cells;
        if rho.len() < n || u_gas.n_cells != n {
            return Err(Error::Config(format!(
                "parcels/coupling: the gas fields have {} / {} cells, the mesh has {n}",
                rho.len(),
                u_gas.n_cells
            )));
        }
        match (self.ctrl.energy.is_on(), t_gas) {
            (true, None) => {
                return Err(Error::Config(
                    "parcels/coupling: energy coupling is on and no gas temperature was \
                     given; (68.10)'s split linearises about it (SPEC-LIT S68.5)"
                        .to_string(),
                ))
            }
            (false, Some(_)) => {
                return Err(Error::Config(
                    "parcels/coupling: energy coupling is off and a gas temperature was \
                     given, which would be read and ignored (SPEC-LIT S13.4)"
                        .to_string(),
                ))
            }
            (true, Some(t)) if t.len() < n => {
                return Err(Error::Config(format!(
                    "parcels/coupling: the gas temperature has {} cells, the mesh has {n}",
                    t.len()
                )))
            }
            _ => {}
        }

        if n == 0 {
            return Ok(());
        }

        // Inert: `q`/`atr` are length-1 stand-ins and `tGas` is the caller's
        // absent field, so both are passed as something valid the kernel
        // never dereferences. The launch has ONE shape whatever the mode is,
        // which is what a captured graph needs.
        let t_field: &DevBuf<Scalar> = t_gas.unwrap_or(rho);
        let n_cells = n as i32;
        let mom = self.ctrl.momentum.code();
        let nrg = self.ctrl.energy.code();
        let cfg = self.cfg_cells;
        let f = self.k.gather.clone();
        let m = self.m;
        let Self {
            f: f_src,
            beta,
            q,
            alpha_t,
            mom_su,
            mom_sp,
            nrg_q,
            nrg_sp,
            ..
        } = self;
        unsafe {
            gpu.stream()
                .launch_builder(&f)
                .arg(dep.offsets())
                .arg(dep.index())
                .arg(&p.np)
                .arg(&p.imp)
                .arg(&p.axr)
                .arg(&p.qim)
                .arg(&p.atr)
                .arg(&m.v)
                .arg(rho)
                .arg(&u_gas.f)
                .arg(t_field)
                .arg(&mut *f_src)
                .arg(&mut *beta)
                .arg(&mut *q)
                .arg(&mut *alpha_t)
                .arg(&mut *mom_su)
                .arg(&mut *mom_sp)
                .arg(&mut *nrg_q)
                .arg(&mut *nrg_sp)
                .arg(&dt)
                .arg(&mom)
                .arg(&nrg)
                .arg(&n_cells)
                .launch(cfg)?;
        }
        Ok(())
    }

    /// Hand the momentum equation its two arrays - SPEC-LIT S18.
    ///
    /// A no-op when momentum coupling is off, and a no-op that registers
    /// **nothing**: the assembly then launches no kernel over the registry
    /// at all, so the matrix is bit for bit the matrix of a run without this
    /// module (S68.10).
    pub fn register_momentum(&self, gpu: &Gpu, s: &mut MomentumSources) -> Result<()> {
        match self.ctrl.momentum {
            CouplingMode::Off => Ok(()),
            CouplingMode::Explicit => s.register_explicit(gpu, &self.mom_su),
            CouplingMode::SemiImplicit => {
                s.register_explicit(gpu, &self.mom_su)?;
                s.register_implicit_sink(gpu, &self.mom_sp)
            }
        }
    }

    /// Hand the energy equation its two arrays - SPEC-LIT S18/S26.
    pub fn register_energy(&self, gpu: &Gpu, s: &mut EnergySources) -> Result<()> {
        match self.ctrl.energy {
            CouplingMode::Off => Ok(()),
            CouplingMode::Explicit => s.register_explicit(gpu, &self.nrg_q),
            CouplingMode::SemiImplicit => {
                s.register_explicit(gpu, &self.nrg_q)?;
                s.register_implicit_sink(gpu, &self.nrg_sp)
            }
        }
    }

    // ---- accessors ----------------------------------------------------

    /// `f_P`, N/m3.
    pub fn force_density(&self) -> &DevBuf<Vec3> {
        &self.f
    }

    /// `beta_P`, kg/(m3 s), `>= 0`.
    pub fn exchange_rate(&self) -> &DevBuf<Scalar> {
        &self.beta
    }

    /// `q_P`, W/m3.
    pub fn heat_density(&self) -> &DevBuf<Scalar> {
        &self.q
    }

    /// `alpha_T,P`, W/(m3 K), `>= 0`.
    pub fn heat_exchange(&self) -> &DevBuf<Scalar> {
        &self.alpha_t
    }

    pub fn momentum_su(&self) -> &DevBuf<Vec3> {
        &self.mom_su
    }

    pub fn momentum_sp(&self) -> &DevBuf<Scalar> {
        &self.mom_sp
    }

    pub fn energy_q(&self) -> &DevBuf<Scalar> {
        &self.nrg_q
    }

    pub fn energy_sp(&self) -> &DevBuf<Scalar> {
        &self.nrg_sp
    }

    // ---- read-back, for reporting and for the gates -------------------

    /// `sum_P V_P f_P dt` - the total impulse this step handed to the gas,
    /// kg m/s. SPEC-LIT (68.4)'s left-hand side.
    ///
    /// A device read-back and a host reduction in cell order: call it when a
    /// driver reports, never inside the step.
    pub fn total_impulse(&mut self, gpu: &Gpu) -> Result<Vec3> {
        let n = self.m.n_cells;
        if n == 0 {
            return Ok(Vec3::ZERO);
        }
        let dt = self.dt;
        let n_cells = n as i32;
        let cfg = self.cfg_cells;
        let func = self.k.cell_integral.clone();
        let m = self.m;
        let Self { f, integral, .. } = self;
        unsafe {
            gpu.stream()
                .launch_builder(&func)
                .arg(&*f)
                .arg(&m.v)
                .arg(&mut *integral)
                .arg(&dt)
                .arg(&n_cells)
                .launch(cfg)?;
        }
        let host = gpu.download(&self.integral)?;
        let mut acc = Vec3::ZERO;
        for v in host.iter().take(n) {
            acc.x += v.x;
            acc.y += v.y;
            acc.z += v.z;
        }
        Ok(acc)
    }

    /// The whole coupled state, on the host.
    pub fn snapshot(&self, gpu: &Gpu) -> Result<CouplingSnapshot> {
        Ok(CouplingSnapshot {
            force: gpu.download(&self.f)?,
            exchange: gpu.download(&self.beta)?,
            heat: gpu.download(&self.q)?,
            heat_exchange: gpu.download(&self.alpha_t)?,
            momentum_su: gpu.download(&self.mom_su)?,
            momentum_sp: gpu.download(&self.mom_sp)?,
            energy_q: gpu.download(&self.nrg_q)?,
            energy_sp: gpu.download(&self.nrg_sp)?,
        })
    }

    /// (68.10)'s cost, measured: `sum_P V_P beta_P (u_P - u_P^n) dt`, the
    /// momentum the semi-implicit split moved relative to the explicit one.
    ///
    /// Exactly zero under [`CouplingMode::Explicit`], and zero under
    /// [`CouplingMode::SemiImplicit`] when the outer iteration has converged
    /// so that `u^{n+1} = u^n`. Reported rather than assumed away.
    pub fn linearisation_defect(
        &self,
        gpu: &Gpu,
        u_now: &GpuVectorField,
        u_lin: &[Vec3],
    ) -> Result<Vec3> {
        if self.ctrl.momentum != CouplingMode::SemiImplicit {
            return Ok(Vec3::ZERO);
        }
        let n = self.m.n_cells;
        let beta = gpu.download(&self.beta)?;
        let vol = gpu.download(&self.m.v)?;
        let u = gpu.download(&u_now.f)?;
        if u_lin.len() < n {
            return Err(Error::Config(format!(
                "parcels/coupling: linearisation_defect was given {} cells of the \
                 linearisation velocity, the mesh has {n}",
                u_lin.len()
            )));
        }
        let mut acc = Vec3::ZERO;
        for c in 0..n {
            let w = vol[c] * beta[c] * self.dt;
            acc.x += w * (u[c].x - u_lin[c].x);
            acc.y += w * (u[c].y - u_lin[c].y);
            acc.z += w * (u[c].z - u_lin[c].z);
        }
        Ok(acc)
    }

    fn check(&self, p: &Parcels<'m>) -> Result<()> {
        if !std::ptr::eq(self.m, p.m) {
            return Err(Error::Config(
                "parcels/coupling: this coupling was built against a different GpuMesh \
                 from the pool it was handed (SPEC-LIT S68.6)"
                    .to_string(),
            ));
        }
        if self.capacity != p.ctrl.capacity || self.physics != p.ctrl.physics {
            return Err(Error::Config(format!(
                "parcels/coupling: built for a pool of {} slots with physics \"{}\", \
                 handed one of {} with \"{}\" (SPEC-LIT S68.6)",
                self.capacity,
                self.physics.name(),
                p.ctrl.capacity,
                p.ctrl.physics.name(),
            )));
        }
        Ok(())
    }
}

// ==========================================================================
//  The host reference, for the gates
// ==========================================================================

/// `sum_p n_p imp_p` over the LIVE parcels of a snapshot, kg m/s - the
/// right-hand side of (68.4), summed on the host in slot order.
///
/// Live only, and that is the whole subtlety: a parcel that left the domain
/// during the step is no longer in the CSR, so the gas was never given the
/// impulse it exchanged on its way out. S68.9 row 4 measures exactly that
/// difference rather than defining it away.
#[must_use]
pub fn live_parcel_impulse(s: &super::ParcelSnapshot) -> Vec3 {
    let mut acc = Vec3::ZERO;
    for i in 0..s.n_slots.min(s.cell.len()) {
        if s.cell[i] < 0 {
            continue;
        }
        let np = s.n_p[i];
        acc.x += np * s.impulse[i].x;
        acc.y += np * s.impulse[i].y;
        acc.z += np * s.impulse[i].z;
    }
    acc
}

/// `sum_p n_p qim_p` over the live parcels, J - the heat the gas gave the
/// droplets, and (68.4)'s energy twin.
#[must_use]
pub fn live_parcel_heat(s: &super::ParcelSnapshot) -> Scalar {
    let mut acc = 0.0;
    if s.heat.is_empty() {
        return acc;
    }
    for i in 0..s.n_slots.min(s.cell.len()) {
        if s.cell[i] >= 0 {
            acc += s.n_p[i] * s.heat[i];
        }
    }
    acc
}

/// The drag impulse one droplet takes over one step of the exact update of
/// (66.5), in the frozen-gas single-sub-step case - the closed form
/// `cuda/parcels.cu` accumulates and the one a test can check it against.
///
/// `m_eff` is `(rho_l + C_am rho) (pi/6) d^3`, `beta = h/tau_p` and
/// `a_g` the buoyancy-corrected gravity of (66.2).
#[must_use]
pub fn drag_impulse(m_eff: Scalar, beta: Scalar, h: Scalar, u_rel: Vec3, a_g: Vec3) -> Vec3 {
    // w = 1 - exp(-beta); q = w/beta, both exact at beta = 0.
    let w = -(-beta).exp_m1();
    let q = if beta > 1e-8 {
        w / beta
    } else {
        1.0 - beta / 2.0 + beta * beta / 6.0
    };
    let k = h * (1.0 - q);
    Vec3::new(
        m_eff * (w * u_rel.x - k * a_g.x),
        m_eff * (w * u_rel.y - k * a_g.y),
        m_eff * (w * u_rel.z - k * a_g.z),
    )
}
