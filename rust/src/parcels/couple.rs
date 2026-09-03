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
//! # And the same idea for the mass - SPEC-LIT S77
//!
//! S76 made the droplets evaporate. S77 hands the gas what leaves them, and
//! it is three things at once because a phase change owes the gas three:
//! the vapour itself (a source on `Y_v`), the enthalpy that mass carries in
//! (`cp_g mdot (T_p - T_g)`, on the SAME energy registry), and the volume it
//! occupies at fixed `p0` (`mdot/rho`, in `Energy::target_divergence`).
//!
//! Two things about that are not what they look like, and both are S77's:
//!
//! * **There is no second latent-heat sink.** The droplet's own budget
//!   (76.10) is `Q_c = C dT_p + dm h_v`, so the convective heat S68 already
//!   deposits contains every joule the phase change consumed. Depositing
//!   `q_lat` again would count it twice. What the gas is actually owed is the
//!   arriving mass's SENSIBLE enthalpy, 12 % of the latent heat, not 100 %.
//! * **The energy half of the divergence arrives on its own.**
//!   `energyTargetDivergence` reads `EnergySources::q`, so registering the
//!   energy source puts it in `(div u)_target` too, with no code. The mass
//!   half is not a heat source, cannot be written as one, and has to be
//!   handed over - S25.1's conduction omission with the halves swapped.
//!
//! # What is refused, by name
//!
//! * **The half-coupled evaporating pool**, in both directions
//!   ([`ParcelCoupling::new`]): the vapour without the heat is a spray that
//!   humidifies and does not cool; the heat without the vapour cools the gas
//!   for a mass transfer that never happened. `mass evaporation` carries all
//!   three couplings or none of them.
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

/// Mass exchange between the phases - SPEC-LIT S77.
///
/// S68 had one value and refused everything else because there was nothing
/// to give; S76 made the droplets evaporate and the refusal's reason became
/// "there is nowhere to put it"; S77 built the somewhere. There are now two
/// values, and [`Self::Evaporation`] carries all three of the couplings a
/// phase change owes the gas at once - the vapour, the enthalpy it brings,
/// and the volume it makes - because a case that took one and left another
/// would have a run whose mass and energy do not close and no way to know.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum MassCoupling {
    /// No species source, no vapour enthalpy and no `D_src` contribution to
    /// the pressure equation. The parcels may still evaporate: `dm_p/dt` is
    /// theirs and this setting is about whether the GAS hears about it.
    #[default]
    None,

    /// The vapour reaches the gas - SPEC-LIT S77:
    ///
    /// * `Y_v` gains `mdot'''(1 - Y_v)/rho` (77.1), through the whole-field
    ///   explicit seam [`crate::scalar_transport::ScalarTransport::correct_with_source`]
    ///   provides;
    /// * the energy registry gains `cp_g mdot''' (T_p - T_g)` (77.2), the
    ///   enthalpy the arriving mass carries, on top of S68's convective
    ///   exchange - and **not** a second latent-heat sink, which would be
    ///   counting the same joules twice (S77.4);
    /// * `Energy::target_divergence` gains `mdot'''/rho` (77.3), the volume
    ///   the mass occupies at fixed `p0`.
    ///
    /// Requires `physics evaporating` and energy coupling on, in both
    /// directions and by name.
    Evaporation,
}

impl MassCoupling {
    pub const NAMES: &'static [&'static str] = &["none", "evaporation"];

    pub fn name(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::Evaporation => "evaporation",
        }
    }

    /// The `OFC_MASS_*` code `cuda/parcelcouple.cu` switches on.
    pub fn code(self) -> i32 {
        match self {
            Self::None => 0,
            Self::Evaporation => 1,
        }
    }

    pub fn is_on(self) -> bool {
        self != Self::None
    }

    pub fn from_name(s: &str) -> Result<Self> {
        match s {
            "none" | "off" => Ok(Self::None),
            "evaporation" | "evaporating" | "species" | "vapour" | "vapor" => {
                Ok(Self::Evaporation)
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
            "parcels/coupling: momentum={} energy={} mass={} (SPEC-LIT S68{})",
            self.momentum.name(),
            self.energy.name(),
            self.mass.name(),
            if self.mass.is_on() { "/S77" } else { "" },
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
    /// (77.6) `mdot'''`, kg/(m3 s): the vapour the parcels put into the gas.
    /// Positive when they are evaporating, negative when they are growing.
    pub vapour: Vec<Scalar>,
    /// (77.2) `cp_g mdot''' (T_p - T_g)`, W/m3: the enthalpy the arriving
    /// mass carries, already summed into [`Self::energy_q`].
    pub vapour_enthalpy: Vec<Scalar>,
    /// (77.1) what the `Y_v` equation is handed, 1/s.
    pub species_su: Vec<Scalar>,
    /// (77.3) what `Energy::target_divergence` is handed, 1/s.
    pub divergence: Vec<Scalar>,
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
/// S68.8 states it rather than discovering it as an OOM. A mass coupling
/// adds four more - `mdot`, `q_vap`, `S_Y` and `D` - for 152 B, and they are
/// allocated at length ONE when it is off, so a momentum-only spray costs
/// exactly what S68 measured (S77.8).
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

    // ---- S77, the vapour ----------------------------------------------
    /// (77.6) `mdot'''`, kg/(m3 s). Length 1 unless mass coupling is on.
    mdot_v: DevBuf<Scalar>,
    /// (77.2) the enthalpy the arriving mass carries, W/m3. Length 1 unless
    /// mass coupling is on. Already inside [`Self::nrg_q`]; kept separately
    /// because S77.9's ledger is posed on it and a round trip through a sum
    /// is not exact.
    q_vap: DevBuf<Scalar>,
    /// (77.1) the `Y_v` source, 1/s. Length 1 unless mass coupling is on.
    y_su: DevBuf<Scalar>,
    /// (77.3) the target-divergence source, 1/s. Length 1 unless mass
    /// coupling is on.
    d_src: DevBuf<Scalar>,
    /// The gas specific heat the enthalpy source is formed with - S66's own
    /// `ParcelControls::cp_gas` (S77.3).
    ///
    /// **It is not the same object as `GasProperties::cp`, and nothing here
    /// can check that they agree.** S66 uses this one as a FILM property, in
    /// `Pr = mu cp/k` for the Nusselt number; S26 books the gas's own
    /// enthalpy as `rho cp T` with the other. (77.2) is the enthalpy the
    /// arriving mass carries relative to that booking, so the constant it
    /// wants is S26's - and this module never sees a `GasProperties`.
    /// [`Self::gas_cp`] exists so that a driver can assert the two agree;
    /// S77.12 says what it costs when they do not.
    cp_gas: Scalar,

    /// Scratch for [`Self::total_impulse`]. Diagnostics only.
    integral: DevBuf<Vec3>,
}

impl<'m> ParcelCoupling<'m> {
    /// Build the coupling for one pool.
    ///
    /// Refuses, by name and at setup: energy coupling on inert parcels (an
    /// inert droplet is an infinite heat bath and coupling one would create
    /// energy from nothing); a mass coupling on a pool that cannot
    /// evaporate; and - in both directions, per S13.4 - the half-coupled
    /// evaporating pool, which is a run whose energy does not close.
    pub fn new(gpu: &Gpu, p: &Parcels<'m>, ctrl: CouplingControls) -> Result<Self> {
        let evaporating = p.ctrl.physics == ParcelPhysics::Evaporating;

        // SPEC-LIT S77.7, and it replaces S76.14's blanket refusal. An
        // evaporating droplet's energy leaves it in TWO parts - the
        // convective heat and the mass that carries enthalpy away - and
        // giving the gas one without the other is not "an approximation": it
        // is a run whose mass and energy do not close, reported as one that
        // does. So the two are refused apart and supported together.
        if ctrl.mass.is_on() && !evaporating {
            return Err(Error::Config(format!(
                "parcels/coupling: mass coupling is \"{}\" and the parcels are \"{}\". \
                 There is no vapour to give: `dm_p/dt` is identically zero unless the \
                 physics is \"evaporating\" (SPEC-LIT S76). Say `physics evaporating`, \
                 or `mass none`",
                ctrl.mass.name(),
                p.ctrl.physics.name(),
            )));
        }
        if ctrl.mass.is_on() && !ctrl.energy.is_on() {
            return Err(Error::Config(format!(
                "parcels/coupling: mass coupling is \"{}\" and energy coupling is \
                 \"off\". The vapour would arrive in the gas without the heat its \
                 phase change took out of it, so the run would gain water and lose no \
                 energy - a spray that humidifies without cooling. (77.2)'s enthalpy \
                 source rides the SAME registry S68's convective exchange does, and \
                 turning that registry off turns both off. Say `energy explicit` \
                 (SPEC-LIT S77.7)",
                ctrl.mass.name(),
            )));
        }
        if ctrl.energy.is_on() && evaporating && !ctrl.mass.is_on() {
            return Err(Error::Config(format!(
                "parcels/coupling: energy coupling is \"{}\", the parcels are \
                 \"evaporating\" and mass coupling is \"none\". S68's energy gather \
                 carries the CONVECTIVE heat, which for an evaporating droplet already \
                 contains every joule the phase change consumed (76.10); depositing it \
                 with the vapour thrown away would cool the gas for a mass transfer \
                 that never happened, and the run's energy would not close. Say \
                 `mass evaporation` and the gas gets both (SPEC-LIT S77), or \
                 `energy off` and it gets neither. `ParcelSnapshot::heat`, `::latent` \
                 and `::mass_lost` carry all three halves for a case that only wants \
                 to see them",
                ctrl.energy.name(),
            )));
        }
        if ctrl.energy.is_on() && !evaporating && p.ctrl.physics != ParcelPhysics::Heating {
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
        // S68.8's rule, one section on: an array nothing will read is
        // allocated at length one rather than at n_cells, so the memory a
        // momentum-only spray costs is the memory S68 measured.
        let vap = if ctrl.mass.is_on() { one } else { 1 };
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

            mdot_v: gpu.zeros(vap)?,
            q_vap: gpu.zeros(vap)?,
            y_su: gpu.zeros(vap)?,
            d_src: gpu.zeros(vap)?,
            cp_gas: p.ctrl.cp_gas,

            integral: gpu.zeros(one)?,
        })
    }

    pub fn controls(&self) -> &CouplingControls {
        &self.ctrl
    }

    /// `15 * 8` bytes per cell without a mass coupling and `19 * 8` with
    /// one - what this object costs before it is paid for. SPEC-LIT S68.8
    /// and S77.8.
    pub fn device_bytes(&self) -> usize {
        let n = self.m.n_cells.max(1);
        let base = n * (3 * 8 + 8 + 8 + 8 + 3 * 8 + 8 + 8 + 8 + 3 * 8);
        base + if self.ctrl.mass.is_on() { n * 4 * 8 } else { 0 }
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
    /// when it is not; `y_vapour` exactly when mass coupling is on (77.4
    /// reads it for the dilution factor), and refused when it is not.
    // Nine, and the lint's bar is seven. Every one is a distinct object the
    // gather reads and none can be folded into another without this module
    // taking ownership of something that is not its: the pool, the CSR, and
    // the four gas fields all have different owners and different lifetimes.
    #[allow(clippy::too_many_arguments)]
    pub fn update(
        &mut self,
        gpu: &Gpu,
        p: &Parcels<'m>,
        dep: &ParcelDeposition<'m>,
        rho: &DevBuf<Scalar>,
        u_gas: &GpuVectorField,
        t_gas: Option<&DevBuf<Scalar>>,
        y_vapour: Option<&DevBuf<Scalar>>,
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
        // S13.4 again, on the vapour field, and in both directions. (77.1)'s
        // dilution factor is the only thing that reads it, so a coupling that
        // is not coupling mass would read it and ignore it.
        match (self.ctrl.mass.is_on(), y_vapour) {
            (true, None) => {
                return Err(Error::Config(
                    "parcels/coupling: mass coupling is on and no gas vapour mass                      fraction was given; (77.1)'s source is mdot(1 - Y_v)/rho and the                      dilution factor is not a detail - without it the vapour goes in                      faster than the mixture it is going into grew (SPEC-LIT S77.3)"
                        .to_string(),
                ))
            }
            (false, Some(_)) => {
                return Err(Error::Config(
                    "parcels/coupling: mass coupling is off and a gas vapour mass                      fraction was given, which would be read and ignored (SPEC-LIT                      S13.4)"
                        .to_string(),
                ))
            }
            (true, Some(y)) if y.len() < n => {
                return Err(Error::Config(format!(
                    "parcels/coupling: the gas vapour mass fraction has {} cells, the                      mesh has {n}",
                    y.len()
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
        let y_field: &DevBuf<Scalar> = y_vapour.unwrap_or(rho);
        let n_cells = n as i32;
        let mom = self.ctrl.momentum.code();
        let nrg = self.ctrl.energy.code();
        let mss = self.ctrl.mass.code();
        let cp_gas = self.cp_gas;
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
            mdot_v,
            q_vap,
            y_su,
            d_src,
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
                .arg(&p.dmv)
                .arg(&p.t)
                .arg(&m.v)
                .arg(rho)
                .arg(&u_gas.f)
                .arg(t_field)
                .arg(y_field)
                .arg(&mut *f_src)
                .arg(&mut *beta)
                .arg(&mut *q)
                .arg(&mut *alpha_t)
                .arg(&mut *mdot_v)
                .arg(&mut *q_vap)
                .arg(&mut *mom_su)
                .arg(&mut *mom_sp)
                .arg(&mut *nrg_q)
                .arg(&mut *nrg_sp)
                .arg(&mut *y_su)
                .arg(&mut *d_src)
                .arg(&dt)
                .arg(&cp_gas)
                .arg(&mom)
                .arg(&nrg)
                .arg(&mss)
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

    /// (77.6) `mdot'''`, kg/(m3 s) - the vapour the parcels put into the gas.
    ///
    /// The raw deposit, and what gate 77-A's mass identity is posed on. Length
    /// one and never written unless mass coupling is on.
    pub fn vapour_production(&self) -> &DevBuf<Scalar> {
        &self.mdot_v
    }

    /// (77.2) `cp_g mdot''' (T_p - T_g)`, W/m3 - the enthalpy the arriving
    /// mass carries, already summed into [`Self::energy_q`].
    ///
    /// `cp_g` here is the pool's [`crate::parcels::ParcelControls::cp_gas`],
    /// not `GasProperties::cp`. They are the same number in every fixture in
    /// this repository and they must be: see [`Self::gas_cp`] and S77.12.
    pub fn vapour_enthalpy_density(&self) -> &DevBuf<Scalar> {
        &self.q_vap
    }

    /// The gas specific heat (77.2)'s enthalpy source was formed with,
    /// J/(kg K) - the pool's own `cp_gas`.
    ///
    /// A driver that registers this coupling on an [`EnergySources`] whose
    /// equation carries a DIFFERENT `cp` has a seam that does not close:
    /// gate 77-B still passes, because it is posed on this same constant on
    /// both sides, but the gas's temperature moves by the wrong amount and
    /// `Q/(rho cp T)` in the divergence divides by the other one. Nothing in
    /// this module can see a `GasProperties`, so the check belongs to the
    /// caller and this is the handle for it:
    ///
    /// ```text
    ///   assert_eq!(coupling.gas_cp(), props.cp);
    /// ```
    pub fn gas_cp(&self) -> Scalar {
        self.cp_gas
    }

    /// (77.1) what the `Y_v` equation is handed, 1/s.
    ///
    /// Pass it to [`crate::scalar_transport::ScalarTransport::correct_with_source`]
    /// on the vapour species - the whole-field explicit seam that solver
    /// provides, applied as `fvm_su(su, +1)` in the same place every other
    /// source in that equation goes.
    ///
    /// **The source is the one for the NON-CONSERVATIVE form** `DY/Dt = S`,
    /// which the species equation is when its convection scheme is
    /// `bounded` - S19's own requirement for a bounded scalar and what every
    /// species entry in this crate carries. With an unbounded scheme the
    /// assembled operator is `dY/dt + div(phi, Y)` and this source is short
    /// by `Y div(u)`; S77.3 says so rather than leaving it to be discovered.
    pub fn vapour_source(&self) -> &DevBuf<Scalar> {
        &self.y_su
    }

    /// (77.3) what `Energy::update_target_divergence_with` is handed, 1/s -
    /// the volume the arriving mass occupies at fixed `p0` and `T`.
    ///
    /// **This is the half of evaporation's effect on the divergence that
    /// needs code.** The other half - the energy the phase change moved -
    /// arrives on its own, because `energyTargetDivergence` reads
    /// `EnergySources::q` and [`Self::register_energy`] has already put it
    /// there. S25.1 lost its conduction term once for exactly this kind of
    /// asymmetry, and S77.6 states which half is which.
    pub fn divergence_source(&self) -> &DevBuf<Scalar> {
        &self.d_src
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
            vapour: gpu.download(&self.mdot_v)?,
            vapour_enthalpy: gpu.download(&self.q_vap)?,
            species_su: gpu.download(&self.y_su)?,
            divergence: gpu.download(&self.d_src)?,
        })
    }

    /// `sum_P V_P mdot'''_P dt` - the vapour mass this step handed the gas,
    /// kg. SPEC-LIT S77.9's gate 77-A, left-hand side.
    ///
    /// A device read-back and a host reduction in cell order, exactly like
    /// [`Self::total_impulse`]: call it when a driver reports, never inside
    /// the step. Zero when mass coupling is off, and it says so by being the
    /// sum of an array that was never written.
    pub fn total_vapour_mass(&self, gpu: &Gpu) -> Result<Scalar> {
        let n = self.m.n_cells;
        if n == 0 || !self.ctrl.mass.is_on() {
            return Ok(0.0);
        }
        let mdot = gpu.download(&self.mdot_v)?;
        let vol = gpu.download(&self.m.v)?;
        let mut acc = 0.0;
        for c in 0..n {
            acc += vol[c] * mdot[c] * self.dt;
        }
        Ok(acc)
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

/// `sum_p n_p qim_p` over the live parcels, J - the CONVECTIVE heat the gas
/// gave the droplets, and (68.4)'s energy twin.
///
/// For a heating parcel that is the whole exchange. For an evaporating one it
/// is half of it, the other half being the latent heat the vapour carried off
/// (`ParcelSnapshot::latent`); which is why S76.14 refuses to couple the
/// energy of an evaporating pool rather than deposit this and call it closed.
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

/// `cp_g sum_p n_p dm_p (T_p - T_g)` over the live parcels, J - the enthalpy
/// the arriving vapour carries into a gas at the uniform temperature
/// `t_gas`, and gate 77-B's energy twin of [`live_parcel_impulse`].
///
/// `t_gas` is one number because the gates that read this run in a uniform
/// gas; the kernel does it per cell, where `T_g` is a per-cell constant that
/// factors out of the cell's own sum (77.2).
///
/// The sign is the physics: droplets are colder than the gas they evaporate
/// into, so the vapour they hand over is a small extra COOLING on top of the
/// convective exchange [`live_parcel_heat`] measures.
#[must_use]
pub fn live_parcel_vapour_enthalpy(
    s: &super::ParcelSnapshot,
    cp_gas: Scalar,
    t_gas: Scalar,
) -> Scalar {
    if s.mass_lost.is_empty() {
        return 0.0;
    }
    let mut acc = 0.0;
    for i in 0..s.n_slots.min(s.cell.len()) {
        if s.cell[i] >= 0 {
            acc += s.n_p[i] * s.mass_lost[i] * (s.temperature[i] - t_gas);
        }
    }
    cp_gas * acc
}

/// The energy the vapour carried out of BOTH sensible pools over one step, J
/// - SPEC-LIT gate 77-B (S77.9), summed on the host over the live parcels.
///
/// ```text
///   E_vap = sum_p n_p [ q_lat,p + dm_p (c_l - cp_g) T_p ]
/// ```
///
/// The first term is the latent heat (76.11) accumulated. The second is not
/// physics: it is the offset between two sensible pools whose enthalpy data
/// are `c_l T` for the liquid and `cp_g T` for the gas, both referred to
/// absolute zero, which is the reference S26's energy equation actually
/// carries. With it, the ledger
///
/// ```text
///   dE_gas + dE_liquid + E_vap = 0
/// ```
///
/// closes to the accuracy of the droplet's OWN budget - `4.8e-12` relative,
/// S76.12 row 7 - and not to round-off. The remainder is the same thing
/// S76.10 already reports: the two accumulators are endpoint differences
/// while the budget is a sum over sub-steps whose mass changes between them,
/// so the ledger inherits exactly that gap and no more. It is measured at
/// `1.3e-12`. Without the second term the ledger is short by
/// `dm (c_l - cp_g) T_p`, which for water at 290 K is 38 % of the latent
/// heat and is not a rounding.
///
/// S77.11 says what this costs in physical terms: the crate's gas energy
/// equation gives water vapour dry air's `cp`, so the number here is not
/// `h_v` and a reader should not read it as one.
#[must_use]
pub fn live_parcel_vapour_energy(
    s: &super::ParcelSnapshot,
    c_liquid: Scalar,
    cp_gas: Scalar,
) -> Scalar {
    if s.mass_lost.is_empty() || s.latent.is_empty() {
        return 0.0;
    }
    let mut acc = 0.0;
    for i in 0..s.n_slots.min(s.cell.len()) {
        if s.cell[i] >= 0 {
            acc += s.n_p[i]
                * (s.latent[i] + s.mass_lost[i] * (c_liquid - cp_gas) * s.temperature[i]);
        }
    }
    acc
}

/// `sum_p n_p m_p c_l T_p` over the live parcels, J - the liquid's own
/// sensible energy, on the reference S26 carries (absolute zero).
///
/// The middle term of gate 77-B's ledger, and the one a test differences
/// across a step.
#[must_use]
pub fn live_parcel_liquid_energy(
    s: &super::ParcelSnapshot,
    rho_liquid: Scalar,
    c_liquid: Scalar,
) -> Scalar {
    let mut acc = 0.0;
    for i in s.live() {
        let d = s.d[i];
        let m = rho_liquid * std::f64::consts::FRAC_PI_6 as Scalar * d * d * d;
        acc += s.n_p[i] * m * c_liquid * s.temperature[i];
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
