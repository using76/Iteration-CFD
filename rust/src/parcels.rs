// meteor-cfd - Copyright (c) 2026 주식회사 메테오시뮬레이션 (Meteo Simulation Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.

//! Lagrangian parcels (SPEC-LIT `SPEC-LIT.md` section 66): the structure-of-
//! arrays pool, the exponential drag update, the face-crossing mesh walk over
//! the cell -> face CSR section 1 already built, deterministic injection, and
//! parcel output.
//!
//! **One-way coupling only.** Parcels read the gas; the gas does not know they
//! exist. No cell field enters a matrix from anywhere in this module, so every
//! Eulerian result in this crate is bitwise unchanged by its presence - not
//! by measurement, but because there is no code path from a parcel to a
//! matrix. [`deposit`] (S67) adds the `(cell, uid)` sort, the per-cell CSR and
//! a gather that produces four per-cell quantities, and couples none of them
//! into an equation either. Evaporation is a later section still;
//! [`ParcelPhysics`] and [`ParcelControls::validate`] refuse it by name rather
//! than quietly doing something else (S13.4).
//!
//! Written from:
//!   J. K. Dukowicz, *A particle-fluid numerical model for liquid sprays*,
//!     J. Comput. Phys. 35 (1980) 229-253 - the discrete droplet model: a
//!     parcel stands for `n_p` identical physical droplets, `n_p` real-valued
//!   C. Crowe, M. Sommerfeld, Y. Tsuji, *Multiphase Flows with Droplets and
//!     Particles*, CRC Press (1998) - the equation of motion, and which of its
//!     terms survive at `rho/rho_l ~ 1e-3`
//!   M. R. Maxey, J. J. Riley, *Phys. Fluids* 26 (1983) 883 - the derivation
//!     behind the added-mass coefficient `C_am = 1/2`
//!   L. Schiller, A. Naumann, *Z. Ver. Deutsch. Ing.* 77 (1933) 318, in the
//!     form compiled by R. Clift, J. R. Grace, M. E. Weber, *Bubbles, Drops,
//!     and Particles*, Academic Press (1978) - the drag correlation
//!   K. McGrattan, S. Hostikka, R. McDermott, J. Floyd, M. Vanella et al.,
//!     *Fire Dynamics Simulator Technical Reference Guide*, NIST SP 1018-1
//!     (NIST, US-Government public domain; `reference/fds/LICENSE.md` read
//!     verbatim) - chapter "Lagrangian Particles" and appendix
//!     "Fluid-Particle Momentum Transfer": the exponential integration of the
//!     linearised drag, and the sub-step CFL bound
//!   G. B. Macpherson, N. Nordin, H. G. Weller, *Commun. Numer. Meth. Engng*
//!     25 (2009) 263 - barycentric tracking, which is the fix S66.6 names for
//!     the one case the plane-crossing walk cannot do. The PAPER was read;
//!     the OpenFOAM implementation of it is GPL and was NOT opened
//!   G. L. Steele Jr., D. Lea, C. H. Flood, *Fast splittable pseudorandom
//!     number generators*, OOPSLA 2014, ACM SIGPLAN Notices 49(10) 453 - the
//!     SplitMix64 finalising mix, used here as a BIJECTION so that parcel
//!     identity is unique by construction. Vigna's reference implementation
//!     is public domain (CC0)
//!   ofgpu `SPEC-LIT.md` S66 - the section this module implements; S1 (the
//!     cell -> face CSR the walk gathers over), S13.4 (the refusal contract)
//!
//! No GPL-licensed source was consulted, and in particular OpenFOAM's
//! `src/lagrangian` tree - the obvious reference, and GPL-3.0 - was not
//! opened.
//!
//! # The two things that make this reproducible
//!
//! **Identity is a bijection, not a counter.** `uid = mix64(injector, event,
//! index)` where the packing is injective and `mix64` is invertible, so two
//! parcels can never share a `uid` and the assignment does not depend on
//! thread scheduling. An atomic counter would hand out ids in hardware
//! scheduling order and silently poison the `(cell, uid)` sort that
//! [`deposit`] keys the whole per-cell CSR on.
//!
//! **Nothing that changes step to step is a kernel argument.** The active
//! count and the step number live in device memory and are read inside the
//! kernels. Every launch geometry is fixed at setup. That is what lets birth
//! and death coexist with CUDA-graph capture, which records a fixed geometry
//! and has no update path.

use std::path::Path;

use cudarc::driver::{CudaFunction, LaunchConfig, PushKernelArg};

use crate::device::{DevBuf, Gpu, KernelSet, BLOCK};
use crate::error::{Error, Result};
use crate::field::GpuVectorField;
use crate::io::contract;
use crate::mesh::{GpuMesh, HostMesh, PatchKind};
use crate::{Label, Scalar, Vec3};

#[cfg(test)]
mod tests;

pub mod couple;
pub mod deposit;

pub use couple::{
    CouplingControls, CouplingMode, CouplingSnapshot, MassCoupling, ParcelCoupling,
};
pub use deposit::{
    DepositSnapshot, DeviceScan, ParcelCsrSnapshot, ParcelDeposition, RADIX_BITS, RADIX_DIGITS,
    SORT_TILE, UID_PASSES,
};

// ==========================================================================
//  SPEC-LIT (66.9): parcel identity
// ==========================================================================

/// Reserved injector id for a parcel placed directly by [`Parcels::seed`]
/// rather than emitted by an injector. `4095` is the largest value the 12-bit
/// injector field can hold, so [`ParcelControls::validate`] caps the injector
/// count at `4095` and the two spaces cannot collide.
pub const SEEDED_INJECTOR_ID: u64 = 4095;

/// Widths of the three fields packed into a parcel's identity, in bits.
pub const UID_INJECTOR_BITS: u32 = 12;
pub const UID_EVENT_BITS: u32 = 32;
pub const UID_INDEX_BITS: u32 = 20;

/// SPEC-LIT (66.9). `uid = mix64((injector << 52) | (event << 20) | index)`.
///
/// The mix is SplitMix64's finaliser (Steele, Lea & Flood 2014; Vigna's
/// reference implementation is public domain), used here for a property that
/// has nothing to do with randomness: **it is a bijection on `u64`**. Three
/// `x ^= x >> k` steps are invertible, and multiplication by an odd constant
/// is invertible mod `2^64`, so distinct packed triples give distinct `uid`s
/// *exactly*. Uniqueness is by construction, not by a birthday argument - and
/// a 32-bit identity over `10^6` parcels would collide with near-certainty,
/// which is the failure this avoids.
///
/// [`tests::the_uid_mix_is_a_bijection`] inverts it and checks.
#[must_use]
pub fn parcel_uid(injector: u64, event: u64, index: u64) -> u64 {
    let mut z = (injector << (UID_EVENT_BITS + UID_INDEX_BITS))
        | (event << UID_INDEX_BITS)
        | index;
    z ^= z >> 30;
    z = z.wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z ^= z >> 27;
    z = z.wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^= z >> 31;
    z
}

// ==========================================================================
//  The settings, and the S13.4 contract they serve
// ==========================================================================

/// SPEC-LIT (66.3): which drag correlation the parcels feel.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DragModel {
    /// No drag at all: the parcel is ballistic. Not a physical model - it is
    /// the verification setting S66.12's mesh-walk gate needs, because a
    /// straight line is the only trajectory whose destination cell can be
    /// computed independently of the solver being tested.
    None,
    /// `C_d = 24/Re` everywhere, i.e. the creeping-flow branch extended past
    /// its range. Useful when a case is known to be Stokesian and the
    /// `pow(Re, 0.687)` of the general branch is not wanted.
    Stokes,
    /// Schiller-Naumann with the `Re = 1` continuity fix (66.3). The default.
    #[default]
    SchillerNaumann,
}

impl DragModel {
    pub const NAMES: &'static [&'static str] = &["none", "stokes", "schillerNaumann"];

    pub fn name(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::Stokes => "stokes",
            Self::SchillerNaumann => "schillerNaumann",
        }
    }

    fn code(self) -> i32 {
        match self {
            Self::None => 0,
            Self::Stokes => 1,
            Self::SchillerNaumann => 2,
        }
    }

    /// SPEC-LIT S13.4: recognised and implemented, or an error naming the
    /// setting and the menu.
    pub fn from_name(s: &str) -> Result<Self> {
        match s {
            "none" => Ok(Self::None),
            "stokes" | "Stokes" => Ok(Self::Stokes),
            "schillerNaumann" | "SchillerNaumann" => Ok(Self::SchillerNaumann),
            other => contract::unsupported(
                "parcels/dragModel",
                other,
                Self::NAMES,
                "schillerNaumann",
                Self::SchillerNaumann,
            ),
        }
    }
}

/// SPEC-LIT (66.10): what a parcel does when the walk takes it through a
/// `wall` patch.
///
/// This is deliberately *two* outcomes, not the Bai-Gosman four. Stick,
/// spread and splash all need a wall film to receive the mass, and splash is
/// a population-growth event with no deterministic capacity policy designed
/// yet; both are refused by name rather than approximated.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum WallAction {
    /// The parcel stops where it met the face and is removed from the working
    /// set, counted into [`ParcelStats::n_wall`]. This is where a film would
    /// receive it.
    #[default]
    Remove,
    /// Specular rebound with normal restitution `e` and tangential loss
    /// `f_t`: `u_n' = -e u_n`, `u_t' = (1 - f_t) u_t`.
    Rebound,
}

impl WallAction {
    pub const NAMES: &'static [&'static str] = &["remove", "rebound"];

    pub fn name(self) -> &'static str {
        match self {
            Self::Remove => "remove",
            Self::Rebound => "rebound",
        }
    }

    fn code(self) -> i32 {
        match self {
            Self::Remove => 0,
            Self::Rebound => 1,
        }
    }

    pub fn from_name(s: &str) -> Result<Self> {
        match s {
            "remove" | "escape" => Ok(Self::Remove),
            "rebound" => Ok(Self::Rebound),
            "stick" | "spread" | "film" => contract::unsupported_note(
                "parcels/wallInteraction",
                s,
                Self::NAMES,
                "a wall film is needed to receive the mass, and this crate has none; \
                 SPEC-LIT S66.10 names it as the next step",
                "remove",
                Self::Remove,
            ),
            "splash" => contract::unsupported_note(
                "parcels/wallInteraction",
                s,
                Self::NAMES,
                "splash multiplies one parcel into N and no deterministic capacity \
                 policy for population growth is designed yet (SPEC-LIT S66.10)",
                "remove",
                Self::Remove,
            ),
            other => contract::unsupported(
                "parcels/wallInteraction",
                other,
                Self::NAMES,
                "remove",
                Self::Remove,
            ),
        }
    }
}

/// SPEC-LIT S66.11: what the parcel state variables *do* over a step.
///
/// There is exactly one supported value. It is an enum, and not simply
/// absent, so that a case asking for evaporation is refused **by name** with
/// the alternative printed, rather than silently running an inert spray and
/// producing a plausible wrong answer - which is the S13.4 failure mode this
/// project keeps finding.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ParcelPhysics {
    /// Position, velocity and cell evolve. Diameter, temperature and `n_p` do
    /// not.
    #[default]
    Inert,

    /// ... and the temperature evolves too, by the lumped-capacity
    /// relaxation of (68.9) with the Ranz-Marshall Nusselt number, driven by
    /// the gas temperature in the parcel's own cell.
    ///
    /// This is what makes S68's energy coupling **conservative**: the heat
    /// the gas loses is exactly the heat one droplet gained, times `n_p`. An
    /// inert parcel held at a fixed temperature would be an infinite heat
    /// bath, and coupling one to the gas would put energy into a run from
    /// nowhere - which is why this is an enum value and not a flag on the
    /// coupling.
    ///
    /// Still no mass transfer: `d` and `n_p` do not move, and evaporation is
    /// refused by name (S68.13).
    Heating,
}

impl ParcelPhysics {
    pub const NAMES: &'static [&'static str] = &["inert", "heating"];

    pub fn name(self) -> &'static str {
        match self {
            Self::Inert => "inert",
            Self::Heating => "heating",
        }
    }

    /// The `OFP_PHYS_*` code `cuda/parcels.cu` switches on.
    pub fn code(self) -> i32 {
        match self {
            Self::Inert => 0,
            Self::Heating => 1,
        }
    }

    pub fn from_name(s: &str) -> Result<Self> {
        match s {
            "inert" => Ok(Self::Inert),
            "heating" | "heatTransfer" => Ok(Self::Heating),
            "evaporating" | "evaporation" | "heatAndMassTransfer" => {
                contract::unsupported_note(
                    "parcels/physics",
                    s,
                    Self::NAMES,
                    "evaporation needs the semi-implicit 3x3 closure of the design note's \
                     S1.5, liquid property tables and a species source, none of which \
                     exist; SPEC-LIT S68.13 names them and this module does NOT implement \
                     them. `heating` is the sensible-heat half of it and IS implemented",
                    "heating",
                    Self::Heating,
                )
            }
            "reacting" | "combusting" => contract::unsupported_note(
                "parcels/physics",
                s,
                Self::NAMES,
                "a reacting parcel needs evaporation first (SPEC-LIT S68.13)",
                "inert",
                Self::Inert,
            ),
            other => contract::unsupported(
                "parcels/physics",
                other,
                Self::NAMES,
                "inert",
                Self::Inert,
            ),
        }
    }
}

/// Everything a case can say about the parcel model.
///
/// Written out field by field wherever it is built - never closed with
/// `..Default::default()`, per SPEC-LIT S13.4.1(b) - so that the next field
/// added is a compile error at every construction site instead of a default
/// nobody reviewed.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ParcelControls {
    /// Slots in the pool. Fixed for the pool's life: growing it would need a
    /// reallocation and a re-capture, both host operations, and neither may
    /// happen inside the time loop.
    pub capacity: usize,
    pub drag: DragModel,
    pub physics: ParcelPhysics,
    pub wall: WallAction,
    /// Normal restitution for [`WallAction::Rebound`], in `[0, 1]`.
    pub restitution: Scalar,
    /// Tangential loss for [`WallAction::Rebound`], in `[0, 1]`.
    pub tangential_loss: Scalar,
    pub gravity: Vec3,
    /// Liquid (or solid) density of the parcel material, kg/m3.
    pub rho_liquid: Scalar,
    /// Gas dynamic viscosity, Pa s. Constant: the parcel model reads the gas
    /// velocity and density as fields, but a variable `mu` would have to come
    /// from a transport model this module does not own.
    pub mu_gas: Scalar,
    /// Liquid (or solid) specific heat capacity, J/(kg K). Read only by
    /// [`ParcelPhysics::Heating`].
    pub c_liquid: Scalar,
    /// Gas thermal conductivity, W/(m K). Read only by
    /// [`ParcelPhysics::Heating`], where it sets `h_g = Nu k_g/d`.
    pub k_gas: Scalar,
    /// Gas specific heat capacity at constant pressure, J/(kg K). Read only
    /// by [`ParcelPhysics::Heating`], and only to form `Pr = mu c_p/k_g`.
    pub cp_gas: Scalar,
    /// Include the added-mass inertia `C_am = 1/2` in the response time.
    ///
    /// It changes the **approach** to terminal velocity and, by (66.4), not
    /// the terminal velocity itself - so its pair test must be measured on a
    /// transient. That is not a caveat, it is the arithmetic, and S66.12
    /// records it because a pair test taken at terminal velocity would report
    /// this setting as inert and be wrong about why.
    pub added_mass: bool,
    /// Sub-step target: no parcel may cross more than this fraction of a cell
    /// per sub-step (66.7).
    pub cfl: Scalar,
    /// Hard cap on sub-steps per parcel per time step. A cap, not a
    /// convergence test: a data-dependent trip count is what a captured graph
    /// cannot express.
    pub max_substeps: u32,
    /// Hard cap on face crossings per sub-step. Exceeding it marks the parcel
    /// `lost` and counts it (66.6).
    pub max_walk: u32,
    /// Blocks in the fixed persistent grid, or `None` to derive it from
    /// `capacity`.
    ///
    /// **This is the one setting whose contract is the opposite of
    /// S13.4.1's.** It is launch geometry, not physics, and it is REQUIRED
    /// not to change the answer;
    /// [`tests::the_persistent_grid_geometry_does_not_change_the_answer`]
    /// asserts exactly that, which is the admissible-exception treatment
    /// S13.4.1 sets out for a setting whose effect must be identically zero.
    pub persistent_blocks: Option<u32>,
}

impl Default for ParcelControls {
    fn default() -> Self {
        Self {
            capacity: 1024,
            drag: DragModel::SchillerNaumann,
            physics: ParcelPhysics::Inert,
            wall: WallAction::Remove,
            restitution: 1.0,
            tangential_loss: 0.0,
            gravity: Vec3::new(0.0, 0.0, -9.81),
            rho_liquid: 1000.0,
            mu_gas: 1.8e-5,
            c_liquid: 4182.0,
            k_gas: 0.026,
            cp_gas: 1005.0,
            added_mass: false,
            cfl: 0.9,
            max_substeps: 64,
            max_walk: 16,
            persistent_blocks: None,
        }
    }
}

impl ParcelControls {
    /// Everything that can be wrong with these numbers, named. Called by
    /// [`Parcels::new`] before a single byte is allocated.
    pub fn validate(&self) -> Result<()> {
        let bad = |what: &str, v: Scalar| {
            Err(Error::Config(format!(
                "parcels: {what} is {v}; SPEC-LIT S66.11 requires it finite and positive"
            )))
        };
        if self.capacity == 0 {
            return Err(Error::Config(
                "parcels: capacity is 0 - an empty pool is the ABSENCE of the parcel model, \
                 not a mode of it (SPEC-LIT S66.11)"
                    .to_string(),
            ));
        }
        if self.capacity > i32::MAX as usize {
            return Err(Error::Config(format!(
                "parcels: capacity {} exceeds the i32 slot index the kernels use",
                self.capacity
            )));
        }
        if !(self.rho_liquid > 0.0) || !self.rho_liquid.is_finite() {
            return bad("rhoLiquid", self.rho_liquid);
        }
        if !(self.mu_gas > 0.0) || !self.mu_gas.is_finite() {
            return bad("muGas", self.mu_gas);
        }
        // S68.5 reads these three only when the parcels are heating, but
        // they are validated always: a number that is nonsense when it is
        // read is nonsense when it is written, and the error is more use at
        // setup than three steps into a run.
        if !(self.c_liquid > 0.0) || !self.c_liquid.is_finite() {
            return bad("cLiquid", self.c_liquid);
        }
        if !(self.k_gas > 0.0) || !self.k_gas.is_finite() {
            return bad("kGas", self.k_gas);
        }
        if !(self.cp_gas > 0.0) || !self.cp_gas.is_finite() {
            return bad("cpGas", self.cp_gas);
        }
        if !(self.cfl > 0.0) || !self.cfl.is_finite() {
            return bad("cfl", self.cfl);
        }
        if self.max_substeps == 0 {
            return Err(Error::Config(
                "parcels: maxSubSteps is 0; (66.7) needs at least one".to_string(),
            ));
        }
        if self.max_walk == 0 {
            return Err(Error::Config(
                "parcels: maxWalk is 0; (66.6) needs at least one face test".to_string(),
            ));
        }
        if !(0.0..=1.0).contains(&self.restitution) {
            return Err(Error::Config(format!(
                "parcels: restitution {} is outside [0, 1]; e > 1 would create energy at \
                 a wall (SPEC-LIT S66.10)",
                self.restitution
            )));
        }
        if !(0.0..=1.0).contains(&self.tangential_loss) {
            return Err(Error::Config(format!(
                "parcels: tangentialLoss {} is outside [0, 1] (SPEC-LIT S66.10)",
                self.tangential_loss
            )));
        }
        if !self.gravity.x.is_finite()
            || !self.gravity.y.is_finite()
            || !self.gravity.z.is_finite()
        {
            return Err(Error::Config("parcels: gravity is not finite".to_string()));
        }
        Ok(())
    }

    /// One line for the startup banner - SPEC-LIT S13.4.2. Every setting the
    /// run will actually use, so that a log says what was in force without
    /// anyone having to infer it from the case files.
    pub fn describe(&self) -> String {
        format!(
            "parcels: physics={} drag={} wall={}{} capacity={} rhoLiquid={} muGas={} \
             g=({}, {}, {}) addedMass={} cfl={} maxSubSteps={} maxWalk={}{} \
             (SPEC-LIT S66)",
            self.physics.name(),
            self.drag.name(),
            self.wall.name(),
            if self.wall == WallAction::Rebound {
                format!(" e={} ft={}", self.restitution, self.tangential_loss)
            } else {
                String::new()
            },
            self.capacity,
            self.rho_liquid,
            self.mu_gas,
            self.gravity.x,
            self.gravity.y,
            self.gravity.z,
            self.added_mass,
            self.cfl,
            self.max_substeps,
            self.max_walk,
            if self.physics == ParcelPhysics::Heating {
                format!(
                    " cLiquid={} kGas={} cpGas={}",
                    self.c_liquid, self.k_gas, self.cp_gas
                )
            } else {
                String::new()
            },
        )
    }

    fn c_am(&self) -> Scalar {
        if self.added_mass {
            0.5
        } else {
            0.0
        }
    }
}

// ==========================================================================
//  Injectors, SPEC-LIT (66.8)
// ==========================================================================

/// One nozzle. Fixed at setup: its cell is located once, on the host, and
/// every parcel it emits is a pure function of `(index, event)`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Injector {
    pub position: Vec3,
    /// Cone axis. Normalised by [`Parcels::new`]; need not be a unit vector.
    pub axis: Vec3,
    /// Half-angle of the hollow cone, radians. `0` is a straight jet.
    pub cone_half_angle: Scalar,
    /// Radius of the stand-off sphere the parcels appear on, m. `0` puts them
    /// at the nozzle point itself.
    pub standoff: Scalar,
    pub speed: Scalar,
    pub diameter: Scalar,
    pub temperature: Scalar,
    /// Liquid mass flow rate, kg/s. Sets `n_p` so that the mass emitted per
    /// event is exactly `mass_flow * dt * stride` (66.8).
    pub mass_flow: Scalar,
    pub parcels_per_event: u32,
    /// Seconds between injection events. Rounded at setup to an integer
    /// number of time steps, because the event index must be integer
    /// arithmetic on the device - a floating-point `floor(t/interval)` would
    /// make the whole spray depend on how the accumulated time was summed.
    pub interval: Scalar,
}

impl Default for Injector {
    fn default() -> Self {
        Self {
            position: Vec3::ZERO,
            axis: Vec3::new(0.0, 0.0, -1.0),
            cone_half_angle: 0.0,
            standoff: 0.0,
            speed: 1.0,
            diameter: 1e-4,
            temperature: 293.15,
            mass_flow: 0.0,
            parcels_per_event: 1,
            interval: 0.0,
        }
    }
}

// ==========================================================================
//  What a run reports
// ==========================================================================

/// SPEC-LIT S66.12's reported quantities. Downloaded by [`Parcels::stats`],
/// which a driver calls when it prints - never inside the step, because a
/// read-back is exactly what a captured graph cannot contain.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ParcelStats {
    /// Slots in use, i.e. the high-water mark of the pool. Dead slots are
    /// still counted: nothing reclaims them in this section.
    pub n_slots: usize,
    pub capacity: usize,
    /// Parcels removed through a non-wall patch.
    pub n_escaped: i64,
    /// Parcels removed at a `wall` patch under [`WallAction::Remove`].
    pub n_wall: i64,
    /// Parcels the walk could not place within `max_walk` crossings. **Zero
    /// on any hex or Cartesian mesh**; a non-zero count is the measurable
    /// defect S66.6 promises instead of a silent wrong answer.
    pub n_lost: i64,
    /// Parcels an injector wanted to emit and the pool had no room for.
    pub n_dropped: i64,
    pub n_injected: i64,
}

impl ParcelStats {
    /// SPEC-LIT S66.11's capacity rule: dropping parcels changes what the
    /// case means, so it is an error rather than a warning - refused where a
    /// human can be told, which is outside the step loop.
    pub fn check_capacity(&self) -> Result<()> {
        if self.n_dropped == 0 {
            return Ok(());
        }
        Err(Error::Config(format!(
            "parcels: the pool is full - {} parcel(s) were not injected because capacity \
             is {}. Raise capacity, lower parcelsPerEvent, or lengthen the injection \
             interval; a spray that silently emits fewer parcels than the case asked for \
             is a different case (SPEC-LIT S66.11)",
            self.n_dropped, self.capacity
        )))
    }
}

/// A host-side copy of the whole pool - what output and every test reads.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ParcelSnapshot {
    pub x: Vec<Vec3>,
    pub u: Vec<Vec3>,
    pub d: Vec<Scalar>,
    pub temperature: Vec<Scalar>,
    pub n_p: Vec<Scalar>,
    pub cell: Vec<Label>,
    pub uid: Vec<u64>,
    pub flags: Vec<u32>,
    /// (68.5): the drag impulse ON ONE DROPLET over the last step, kg m/s.
    /// Multiply by `n_p` for the parcel's own.
    pub impulse: Vec<Vec3>,
    /// (68.6): the momentum exchange rate for one droplet, kg/s.
    pub exchange: Vec<Scalar>,
    /// (68.9): the heat into ONE DROPLET over the last step, J. Empty
    /// unless the parcels are heating.
    pub heat: Vec<Scalar>,
    /// (68.9): the heat exchange rate for one droplet, W/K. Empty unless
    /// the parcels are heating.
    pub heat_exchange: Vec<Scalar>,
    /// Slots in use; entries beyond this are untouched pool.
    pub n_slots: usize,
}

impl ParcelSnapshot {
    /// Slot indices that are still alive, in slot order.
    #[must_use]
    pub fn live(&self) -> Vec<usize> {
        (0..self.n_slots).filter(|&i| self.cell[i] >= 0).collect()
    }

    /// Total liquid mass the live parcels stand for, kg:
    /// `sum n_p rho_l (pi/6) d^3`.
    #[must_use]
    pub fn liquid_mass(&self, rho_liquid: Scalar) -> Scalar {
        let mut m = 0.0;
        for i in self.live() {
            let d = self.d[i];
            m += self.n_p[i] * rho_liquid * std::f64::consts::FRAC_PI_6 as Scalar * d * d * d;
        }
        m
    }
}

// ==========================================================================
//  Kernels - cuda/parcels.cu
// ==========================================================================

struct ParcelKernels {
    begin_step: CudaFunction,
    inject: CudaFunction,
    integrate: CudaFunction,
    end_step: CudaFunction,
}

impl ParcelKernels {
    fn new(gpu: &Gpu) -> Result<Self> {
        let k = KernelSet::new(gpu, crate::kernels::PARCELS)?;
        Ok(Self {
            begin_step: k.func("parcelBeginStep")?,
            inject: k.func("parcelInject")?,
            integrate: k.func("parcelIntegrate")?,
            end_step: k.func("parcelEndStep")?,
        })
    }
}

/// Counter slots, mirroring the `OFP_N_*` defines in `cuda/parcels.cu`.
const N_ESCAPED: usize = 0;
const N_WALL: usize = 1;
const N_LOST: usize = 2;
const N_DROPPED: usize = 3;
const N_INJECTED: usize = 4;
const N_COUNTERS: usize = 5;

/// Upper bound on the fixed persistent grid. A grid-stride kernel needs only
/// enough blocks to fill the device; more is waste, and the number must be a
/// setup-time constant so the captured graph never needs updating.
const MAX_PERSISTENT_BLOCKS: u32 = 1024;

// ==========================================================================
//  The pool
// ==========================================================================

/// The structure-of-arrays parcel pool and the four kernels that move it -
/// SPEC-LIT S66.
///
/// `Vec3` is stored interleaved, as `DevBuf<Vec3>`, matching `GpuMesh`'s own
/// `c`, `sf` and `cf`. (The design note that preceded this section asserted
/// the mesh stores vectors as three separate arrays and recommended matching
/// it; it does not, and matching what the crate actually does keeps the
/// `#[repr(C)]` mirror in `types.rs` doing the marshalling.)
pub struct Parcels<'m> {
    m: &'m GpuMesh,
    ctrl: ParcelControls,
    k: ParcelKernels,

    /// Fixed launch geometry - the whole reason birth and death can coexist
    /// with graph capture.
    grid: LaunchConfig,

    // ---- the pool, one entry per slot ---------------------------------
    x: DevBuf<Vec3>,
    u: DevBuf<Vec3>,
    d: DevBuf<Scalar>,
    t: DevBuf<Scalar>,
    np: DevBuf<Scalar>,
    cell: DevBuf<Label>,
    uid: DevBuf<u64>,
    flags: DevBuf<u32>,

    // ---- the coupling accumulators, S68.2 -----------------------------
    /// Drag impulse on ONE droplet over the last step, kg m/s.
    imp: DevBuf<Vec3>,
    /// Momentum exchange rate for one droplet, kg/s.
    axr: DevBuf<Scalar>,
    /// Heat into ONE droplet over the last step, J. Length 1 unless the
    /// parcels are heating.
    qim: DevBuf<Scalar>,
    /// Heat exchange rate for one droplet, W/K. Length 1 unless heating.
    atr: DevBuf<Scalar>,
    /// A one-element stand-in for the gas temperature, passed when the
    /// parcels are inert so that the kernel signature never changes and the
    /// pointer is never dereferenced.
    t_null: DevBuf<Scalar>,

    // ---- device-resident step state -----------------------------------
    n_active: DevBuf<i32>,
    step: DevBuf<i64>,
    counters: DevBuf<i64>,
    total: DevBuf<i32>,

    // ---- injectors, fixed at setup ------------------------------------
    injectors: Vec<Injector>,
    inj_base: DevBuf<i32>,
    inj_count: DevBuf<i32>,
    inj_event: DevBuf<i64>,
    inj_stride: DevBuf<i32>,
    inj_per_event: DevBuf<i32>,
    inj_pos: DevBuf<Vec3>,
    inj_axis: DevBuf<Vec3>,
    inj_t1: DevBuf<Vec3>,
    inj_t2: DevBuf<Vec3>,
    inj_cell: DevBuf<Label>,
    inj_speed: DevBuf<Scalar>,
    inj_diameter: DevBuf<Scalar>,
    inj_temperature: DevBuf<Scalar>,
    inj_weight: DevBuf<Scalar>,
    inj_half_angle: DevBuf<Scalar>,
    inj_standoff: DevBuf<Scalar>,
    n_inj: i32,

    /// The `dt` the injection strides were computed against. A captured graph
    /// freezes it, so [`Parcels::step`] refuses a different one.
    dt: Scalar,
}

impl<'m> Parcels<'m> {
    /// Build the pool. `dt` is needed here, not only at `step`, because the
    /// injection interval has to be reduced to an integer number of steps
    /// once, deterministically, on the host.
    pub fn new(
        gpu: &Gpu,
        hm: &HostMesh,
        m: &'m GpuMesh,
        ctrl: ParcelControls,
        injectors: &[Injector],
        dt: Scalar,
    ) -> Result<Self> {
        ctrl.validate()?;
        if !(dt > 0.0) || !dt.is_finite() {
            return Err(Error::Config(format!(
                "parcels: dt is {dt}; SPEC-LIT S66.8 reduces the injection interval to an \
                 integer number of steps and needs a positive, finite one"
            )));
        }
        if hm.n_cells != m.n_cells {
            return Err(Error::Config(
                "parcels: the host and device meshes disagree on the cell count".to_string(),
            ));
        }

        // SPEC-LIT S66.6: a coupled patch would need the parcel and its
        // velocity transformed through the couple, and this section has no
        // transform. Refused by name at setup rather than discovered as a
        // parcel that vanished.
        for p in &m.patches {
            if matches!(
                p.kind,
                PatchKind::Cyclic | PatchKind::Processor | PatchKind::Interface
            ) {
                return Err(Error::Config(format!(
                    "parcels: patch \"{}\" is {}, and parcel transport across a coupled \
                     patch is not implemented - the parcel and its velocity would have to \
                     be transformed through the couple (SPEC-LIT S66.6). Available \
                     boundary outcomes: escape on a generic patch, {} at a wall, specular \
                     reflection at symmetry and empty",
                    p.name,
                    p.kind.as_str(),
                    ctrl.wall.name(),
                )));
            }
        }

        if injectors.len() >= SEEDED_INJECTOR_ID as usize {
            return Err(Error::Config(format!(
                "parcels: {} injectors, but the identity of (66.9) has {UID_INJECTOR_BITS} \
                 bits for the injector index and reserves {SEEDED_INJECTOR_ID} for a \
                 seeded parcel",
                injectors.len()
            )));
        }

        let cap = ctrl.capacity;
        let blocks = ctrl.persistent_blocks.unwrap_or_else(|| {
            (cap.div_ceil(BLOCK as usize) as u32).clamp(1, MAX_PERSISTENT_BLOCKS)
        });
        if blocks == 0 {
            return Err(Error::Config(
                "parcels: persistentBlocks is 0; a grid dimension of zero is an invalid \
                 launch configuration, not a no-op"
                    .to_string(),
            ));
        }
        let grid = LaunchConfig {
            grid_dim: (blocks, 1, 1),
            block_dim: (BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };

        // ---- injector descriptors, all resolved on the host ----------
        let n = injectors.len();
        let one = n.max(1);
        let mut pos = vec![Vec3::ZERO; one];
        let mut axis = vec![Vec3::ZERO; one];
        let mut t1 = vec![Vec3::ZERO; one];
        let mut t2 = vec![Vec3::ZERO; one];
        let mut icell = vec![-1 as Label; one];
        let mut speed = vec![0.0 as Scalar; one];
        let mut diameter = vec![0.0 as Scalar; one];
        let mut temperature = vec![0.0 as Scalar; one];
        let mut weight = vec![0.0 as Scalar; one];
        let mut half_angle = vec![0.0 as Scalar; one];
        let mut standoff = vec![0.0 as Scalar; one];
        let mut per_event = vec![0i32; one];
        let mut stride = vec![0i32; one];

        for (j, inj) in injectors.iter().enumerate() {
            if inj.parcels_per_event == 0 {
                return Err(Error::Config(format!(
                    "parcels: injector {j} emits 0 parcels per event; an injector that \
                     injects nothing is the absence of an injector"
                )));
            }
            if inj.parcels_per_event as u64 > (1u64 << UID_INDEX_BITS) {
                return Err(Error::Config(format!(
                    "parcels: injector {j} emits {} parcels per event, above the \
                     {} the identity of (66.9) can index",
                    inj.parcels_per_event,
                    1u64 << UID_INDEX_BITS
                )));
            }
            if !(inj.diameter > 0.0) || !inj.diameter.is_finite() {
                return Err(Error::Config(format!(
                    "parcels: injector {j} has diameter {}; it must be finite and positive",
                    inj.diameter
                )));
            }
            if !(inj.mass_flow >= 0.0) || !inj.mass_flow.is_finite() {
                return Err(Error::Config(format!(
                    "parcels: injector {j} has massFlow {}; it must be finite and \
                     non-negative",
                    inj.mass_flow
                )));
            }
            let ax = inj.axis.normalised();
            if ax.mag_sqr() == 0.0 {
                return Err(Error::Config(format!(
                    "parcels: injector {j} has a zero cone axis"
                )));
            }
            let (a, b) = tangent_frame(ax);

            let c = locate_cell(hm, inj.position).ok_or_else(|| {
                Error::Config(format!(
                    "parcels: injector {j} at ({}, {}, {}) is not inside any cell of this \
                     mesh. An injector's cell is located once, here, on the host; the \
                     tracking kernel never searches (SPEC-LIT S66.6)",
                    inj.position.x, inj.position.y, inj.position.z
                ))
            })?;

            // (66.8): the interval reduced to whole steps, and the weight
            // that makes the emitted mass exactly mdot * dt * stride.
            let st = if inj.interval <= 0.0 {
                1i32
            } else {
                let s = (inj.interval / dt).round();
                if !(s >= 1.0) || !s.is_finite() || s > i32::MAX as Scalar {
                    return Err(Error::Config(format!(
                        "parcels: injector {j} has interval {} against dt {dt}, which is \
                         not a usable whole number of steps",
                        inj.interval
                    )));
                }
                s as i32
            };
            let m_droplet = ctrl.rho_liquid
                * std::f64::consts::FRAC_PI_6 as Scalar
                * inj.diameter
                * inj.diameter
                * inj.diameter;

            pos[j] = inj.position;
            axis[j] = ax;
            t1[j] = a;
            t2[j] = b;
            icell[j] = c as Label;
            speed[j] = inj.speed;
            diameter[j] = inj.diameter;
            temperature[j] = inj.temperature;
            weight[j] = inj.mass_flow * dt * st as Scalar
                / (inj.parcels_per_event as Scalar * m_droplet);
            half_angle[j] = inj.cone_half_angle;
            standoff[j] = inj.standoff;
            per_event[j] = inj.parcels_per_event as i32;
            stride[j] = st;
        }

        Ok(Self {
            m,
            ctrl,
            k: ParcelKernels::new(gpu)?,
            grid,

            x: gpu.zeros(cap)?,
            u: gpu.zeros(cap)?,
            d: gpu.zeros(cap)?,
            t: gpu.zeros(cap)?,
            np: gpu.zeros(cap)?,
            cell: gpu.upload(&vec![-1 as Label; cap])?,
            uid: gpu.zeros(cap)?,
            flags: gpu.zeros(cap)?,

            imp: gpu.zeros(cap)?,
            axr: gpu.zeros(cap)?,
            // Sized only when something reads them - S68.2's memory note.
            qim: gpu.zeros(if ctrl.physics == ParcelPhysics::Heating { cap } else { 1 })?,
            atr: gpu.zeros(if ctrl.physics == ParcelPhysics::Heating { cap } else { 1 })?,
            t_null: gpu.zeros(1)?,

            n_active: gpu.zeros(1)?,
            step: gpu.zeros(1)?,
            counters: gpu.zeros(N_COUNTERS)?,
            total: gpu.zeros(1)?,

            injectors: injectors.to_vec(),
            inj_base: gpu.zeros(one)?,
            inj_count: gpu.zeros(one)?,
            inj_event: gpu.zeros(one)?,
            inj_stride: gpu.upload(&stride)?,
            inj_per_event: gpu.upload(&per_event)?,
            inj_pos: gpu.upload(&pos)?,
            inj_axis: gpu.upload(&axis)?,
            inj_t1: gpu.upload(&t1)?,
            inj_t2: gpu.upload(&t2)?,
            inj_cell: gpu.upload(&icell)?,
            inj_speed: gpu.upload(&speed)?,
            inj_diameter: gpu.upload(&diameter)?,
            inj_temperature: gpu.upload(&temperature)?,
            inj_weight: gpu.upload(&weight)?,
            inj_half_angle: gpu.upload(&half_angle)?,
            inj_standoff: gpu.upload(&standoff)?,
            n_inj: n as i32,

            dt,
        })
    }

    pub fn controls(&self) -> &ParcelControls {
        &self.ctrl
    }

    pub fn injectors(&self) -> &[Injector] {
        &self.injectors
    }

    /// Blocks in the fixed persistent grid, as resolved at setup.
    pub fn persistent_blocks(&self) -> u32 {
        self.grid.grid_dim.0
    }

    /// Place parcels directly, at setup, with no injector - what a
    /// verification case needs (a single droplet released in still air) and
    /// what a restart will use.
    ///
    /// Setup only: it writes device memory from the host, which is illegal
    /// inside the time loop and impossible inside a captured graph.
    pub fn seed(&mut self, gpu: &Gpu, hm: &HostMesh, seeds: &[SeedParcel]) -> Result<()> {
        if seeds.len() > self.ctrl.capacity {
            return Err(Error::Config(format!(
                "parcels: {} seed parcels but capacity is {}",
                seeds.len(),
                self.ctrl.capacity
            )));
        }
        if seeds.len() as u64 > (1u64 << UID_INDEX_BITS) {
            return Err(Error::Config(format!(
                "parcels: {} seed parcels, above the {} the identity of (66.9) can index",
                seeds.len(),
                1u64 << UID_INDEX_BITS
            )));
        }

        let cap = self.ctrl.capacity;
        let mut x = vec![Vec3::ZERO; cap];
        let mut u = vec![Vec3::ZERO; cap];
        let mut d = vec![0.0 as Scalar; cap];
        let mut t = vec![0.0 as Scalar; cap];
        let mut np = vec![0.0 as Scalar; cap];
        let mut cell = vec![-1 as Label; cap];
        let mut uid = vec![0u64; cap];
        let mut flags = vec![0u32; cap];

        for (i, s) in seeds.iter().enumerate() {
            if !(s.diameter > 0.0) || !s.diameter.is_finite() {
                return Err(Error::Config(format!(
                    "parcels: seed {i} has diameter {}; it must be finite and positive",
                    s.diameter
                )));
            }
            let c = locate_cell(hm, s.position).ok_or_else(|| {
                Error::Config(format!(
                    "parcels: seed {i} at ({}, {}, {}) is not inside any cell of this mesh",
                    s.position.x, s.position.y, s.position.z
                ))
            })?;
            x[i] = s.position;
            u[i] = s.velocity;
            d[i] = s.diameter;
            t[i] = s.temperature;
            np[i] = s.n_p;
            cell[i] = c as Label;
            uid[i] = s.uid.unwrap_or_else(|| parcel_uid(SEEDED_INJECTOR_ID, 0, i as u64));
            flags[i] = 1;
        }

        // (66.9)/(67.1): `(cell, uid)` is a total order only if `uid` is
        // unique. A derived identity is unique by construction; an explicit
        // one is the caller's, so it is checked here - at setup, on the host,
        // where a duplicate can be named - rather than discovered as a
        // deposition that silently depends on slot order.
        {
            let mut seen = std::collections::HashMap::with_capacity(seeds.len());
            for (i, &u) in uid.iter().take(seeds.len()).enumerate() {
                if let Some(prev) = seen.insert(u, i) {
                    return Err(Error::Config(format!(
                        "parcels: seed {i} and seed {prev} both carry identity {u}. \
                         (cell, uid) is a TOTAL order only if uid is unique, and S67's \
                         deposition canonicalisation rests on that - two parcels sharing \
                         one identity would make the order of their contributions depend \
                         on which slot they happened to land in (SPEC-LIT S67.1)"
                    )));
                }
            }
        }

        gpu.write(&mut self.x, &x)?;
        gpu.write(&mut self.u, &u)?;
        gpu.write(&mut self.d, &d)?;
        gpu.write(&mut self.t, &t)?;
        gpu.write(&mut self.np, &np)?;
        gpu.write(&mut self.cell, &cell)?;
        gpu.write(&mut self.uid, &uid)?;
        gpu.write(&mut self.flags, &flags)?;
        gpu.write(&mut self.n_active, &[seeds.len() as i32])?;
        gpu.write(&mut self.step, &[0i64])?;
        gpu.write(&mut self.counters, &[0i64; N_COUNTERS])?;
        Ok(())
    }

    /// Advance every parcel by one time step: inject, integrate, walk.
    ///
    /// Four launches, all of fixed geometry, none of which reads anything
    /// back to the host. Capturable by [`crate::Gpu::capture`] exactly as it
    /// stands, and `tests::the_graph_is_captured_once_and_replayed` proves
    /// the replay reproduces the eager path bit for bit while the working set
    /// grows underneath it.
    pub fn step(
        &mut self,
        gpu: &Gpu,
        u_gas: &GpuVectorField,
        rho_gas: &DevBuf<Scalar>,
        t_gas: Option<&DevBuf<Scalar>>,
        dt: Scalar,
    ) -> Result<()> {
        if dt != self.dt {
            return Err(Error::Config(format!(
                "parcels: step was given dt = {dt} but the pool was built for dt = {}. The \
                 injection interval was reduced to whole steps against the second, and a \
                 captured graph has frozen it; rebuild the pool to change dt (SPEC-LIT \
                 S66.8)",
                self.dt
            )));
        }
        if u_gas.n_cells != self.m.n_cells || rho_gas.len() != self.m.n_cells {
            return Err(Error::Config(format!(
                "parcels: the gas fields have {} / {} cells, the mesh has {}",
                u_gas.n_cells,
                rho_gas.len(),
                self.m.n_cells
            )));
        }

        // SPEC-LIT S13.4 on both sides. A heating parcel handed no gas
        // temperature would silently freeze at its injection value and
        // couple a constant heat source for ever; an inert one handed a
        // temperature field would read it and ignore it. Both are refused
        // by name.
        match (self.ctrl.physics, t_gas) {
            (ParcelPhysics::Heating, Some(t)) if t.len() != self.m.n_cells => {
                return Err(Error::Config(format!(
                    "parcels: the gas temperature has {} cells, the mesh has {}",
                    t.len(),
                    self.m.n_cells
                )))
            }
            (ParcelPhysics::Heating, None) => {
                return Err(Error::Config(
                    "parcels: physics is \"heating\" and no gas temperature was given. \
                     (68.9) relaxes every droplet towards the temperature of its own \
                     cell, and without that field the parcels would hold their injection \
                     temperature and couple a constant heat source into the gas for ever \
                     (SPEC-LIT S68.5)"
                        .to_string(),
                ))
            }
            (ParcelPhysics::Inert, Some(_)) => {
                return Err(Error::Config(
                    "parcels: physics is \"inert\" and a gas temperature was given. An \
                     inert parcel's temperature does not move, so the field would be \
                     read and ignored; say `physics heating` to have it used (SPEC-LIT \
                     S13.4)"
                        .to_string(),
                ))
            }
            _ => {}
        }

        let cap = self.ctrl.capacity as i32;
        let n_inj = self.n_inj;
        let cfg = self.grid;

        if n_inj > 0 {
            let f = self.k.begin_step.clone();
            let Self {
                n_active,
                step,
                counters,
                inj_base,
                inj_count,
                inj_event,
                inj_stride,
                inj_per_event,
                total,
                ..
            } = self;
            unsafe {
                gpu.stream()
                    .launch_builder(&f)
                    .arg(&mut *n_active)
                    .arg(&*step)
                    .arg(&mut *counters)
                    .arg(&mut *inj_base)
                    .arg(&mut *inj_count)
                    .arg(&mut *inj_event)
                    .arg(&*inj_stride)
                    .arg(&*inj_per_event)
                    .arg(&n_inj)
                    .arg(&cap)
                    .arg(&mut *total)
                    .launch(cfg)?;
            }

            let f = self.k.inject.clone();
            let max_walk = self.ctrl.max_walk as i32;
            let m = self.m;
            let Self {
                x,
                u,
                d,
                t,
                np,
                cell,
                uid,
                flags,
                counters,
                total,
                inj_base,
                inj_count,
                inj_event,
                inj_pos,
                inj_axis,
                inj_t1,
                inj_t2,
                inj_cell,
                inj_speed,
                inj_diameter,
                inj_temperature,
                inj_weight,
                inj_half_angle,
                inj_standoff,
                inj_per_event,
                ..
            } = self;
            unsafe {
                gpu.stream()
                    .launch_builder(&f)
                    .arg(&mut *x)
                    .arg(&mut *u)
                    .arg(&mut *d)
                    .arg(&mut *t)
                    .arg(&mut *np)
                    .arg(&mut *cell)
                    .arg(&mut *uid)
                    .arg(&mut *flags)
                    .arg(&mut *counters)
                    .arg(&*total)
                    .arg(&*inj_base)
                    .arg(&*inj_count)
                    .arg(&*inj_event)
                    .arg(&*inj_pos)
                    .arg(&*inj_axis)
                    .arg(&*inj_t1)
                    .arg(&*inj_t2)
                    .arg(&*inj_cell)
                    .arg(&*inj_speed)
                    .arg(&*inj_diameter)
                    .arg(&*inj_temperature)
                    .arg(&*inj_weight)
                    .arg(&*inj_half_angle)
                    .arg(&*inj_standoff)
                    .arg(&*inj_per_event)
                    .arg(&n_inj)
                    .arg(&m.owner)
                    .arg(&m.neighbour)
                    .arg(&m.sf)
                    .arg(&m.cf)
                    .arg(&m.cf_offset)
                    .arg(&m.cf_face)
                    .arg(&m.cf_own)
                    .arg(&m.bcf_offset)
                    .arg(&m.bcf_face)
                    .arg(&m.b_sf)
                    .arg(&m.b_cf)
                    .arg(&m.b_kind)
                    .arg(&max_walk)
                    .launch(cfg)?;
            }
        }

        let f = self.k.integrate.clone();
        let mu = self.ctrl.mu_gas;
        let rho_l = self.ctrl.rho_liquid;
        let cam = self.ctrl.c_am();
        let g = self.ctrl.gravity;
        let drag = self.ctrl.drag.code();
        let wall = self.ctrl.wall.code();
        let e = self.ctrl.restitution;
        let ft = self.ctrl.tangential_loss;
        let cfl = self.ctrl.cfl;
        let max_sub = self.ctrl.max_substeps as i32;
        let max_walk = self.ctrl.max_walk as i32;
        let physics = self.ctrl.physics.code();
        let c_liquid = self.ctrl.c_liquid;
        let k_gas = self.ctrl.k_gas;
        let cp_gas = self.ctrl.cp_gas;
        let m = self.m;
        let Self {
            x,
            u,
            d,
            t,
            cell,
            flags,
            counters,
            n_active,
            imp,
            axr,
            qim,
            atr,
            t_null,
            ..
        } = self;
        // Inert: a one-element stand-in the kernel never dereferences, so
        // the launch has one shape whatever the physics is - which is what
        // a captured graph needs.
        let t_field: &DevBuf<Scalar> = match t_gas {
            Some(t) => t,
            None => &*t_null,
        };
        unsafe {
            gpu.stream()
                .launch_builder(&f)
                .arg(&mut *x)
                .arg(&mut *u)
                .arg(&*d)
                .arg(&mut *cell)
                .arg(&mut *flags)
                .arg(&mut *counters)
                .arg(&*n_active)
                .arg(&u_gas.f)
                .arg(rho_gas)
                .arg(&m.v)
                .arg(&m.owner)
                .arg(&m.neighbour)
                .arg(&m.sf)
                .arg(&m.cf)
                .arg(&m.cf_offset)
                .arg(&m.cf_face)
                .arg(&m.cf_own)
                .arg(&m.bcf_offset)
                .arg(&m.bcf_face)
                .arg(&m.b_sf)
                .arg(&m.b_cf)
                .arg(&m.b_kind)
                .arg(&dt)
                .arg(&mu)
                .arg(&rho_l)
                .arg(&cam)
                .arg(&g)
                .arg(&drag)
                .arg(&wall)
                .arg(&e)
                .arg(&ft)
                .arg(&cfl)
                .arg(&max_sub)
                .arg(&max_walk)
                .arg(&mut *imp)
                .arg(&mut *axr)
                .arg(&mut *qim)
                .arg(&mut *atr)
                .arg(&mut *t)
                .arg(t_field)
                .arg(&physics)
                .arg(&c_liquid)
                .arg(&k_gas)
                .arg(&cp_gas)
                .launch(cfg)?;
        }

        let f = self.k.end_step.clone();
        let Self { step, .. } = self;
        unsafe {
            gpu.stream().launch_builder(&f).arg(&mut *step).launch(cfg)?;
        }

        Ok(())
    }

    /// SPEC-LIT S66.12's counters. A device read-back: call it when the
    /// driver reports, never inside the step.
    pub fn stats(&self, gpu: &Gpu) -> Result<ParcelStats> {
        let c = gpu.download(&self.counters)?;
        let n = gpu.download(&self.n_active)?;
        Ok(ParcelStats {
            n_slots: n[0].max(0) as usize,
            capacity: self.ctrl.capacity,
            n_escaped: c[N_ESCAPED],
            n_wall: c[N_WALL],
            n_lost: c[N_LOST],
            n_dropped: c[N_DROPPED],
            n_injected: c[N_INJECTED],
        })
    }

    /// The whole pool, on the host. Output and every test read this.
    pub fn snapshot(&self, gpu: &Gpu) -> Result<ParcelSnapshot> {
        let n = gpu.download(&self.n_active)?[0].max(0) as usize;
        Ok(ParcelSnapshot {
            x: gpu.download(&self.x)?,
            u: gpu.download(&self.u)?,
            d: gpu.download(&self.d)?,
            temperature: gpu.download(&self.t)?,
            n_p: gpu.download(&self.np)?,
            cell: gpu.download(&self.cell)?,
            uid: gpu.download(&self.uid)?,
            flags: gpu.download(&self.flags)?,
            impulse: gpu.download(&self.imp)?,
            exchange: gpu.download(&self.axr)?,
            heat: if self.ctrl.physics == ParcelPhysics::Heating {
                gpu.download(&self.qim)?
            } else {
                Vec::new()
            },
            heat_exchange: if self.ctrl.physics == ParcelPhysics::Heating {
                gpu.download(&self.atr)?
            } else {
                Vec::new()
            },
            n_slots: n.min(self.ctrl.capacity),
        })
    }

    /// Write the live parcels as a VTK PolyData (`.vtp`) file - SPEC-LIT
    /// S66.13. Without this a spray cannot be debugged at all.
    pub fn write_vtp(&self, gpu: &Gpu, path: &Path, time: Option<Scalar>) -> Result<()> {
        let s = self.snapshot(gpu)?;
        crate::io::vtu::write_parcels_vtp(path, &s, time)
    }
}

/// A parcel placed directly at setup - see [`Parcels::seed`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SeedParcel {
    pub position: Vec3,
    pub velocity: Vec3,
    pub diameter: Scalar,
    pub temperature: Scalar,
    /// How many physical droplets this parcel stands for.
    pub n_p: Scalar,
    /// The identity this parcel carries (66.9), or `None` to derive it from
    /// the slot as `mix64(SEEDED, 0, slot)`.
    ///
    /// `None` is what a verification case wants: it needs a parcel somewhere,
    /// not a particular one. `Some` exists for the two things that need the
    /// identity to be a property of the **parcel** and not of the array:
    ///
    /// * a restart, which must reconstruct the pool with the identities the
    ///   checkpoint recorded, or the deposition order changes across it;
    /// * S67.10's canonicalisation gate, which holds the parcel *set* fixed
    ///   and permutes the *slot order*. With a derived identity the two are
    ///   the same thing and the gate cannot be posed at all.
    ///
    /// [`Parcels::seed`] refuses duplicates: `(cell, uid)` is only a total
    /// order if `uid` is unique, and the whole of S67 rests on that.
    pub uid: Option<u64>,
}

impl Default for SeedParcel {
    fn default() -> Self {
        Self {
            position: Vec3::ZERO,
            velocity: Vec3::ZERO,
            diameter: 1e-4,
            temperature: 293.15,
            n_p: 1.0,
            uid: None,
        }
    }
}

// ==========================================================================
//  Host-side geometry helpers
// ==========================================================================

/// An orthonormal pair spanning the plane perpendicular to the unit vector
/// `n`. The seed axis is whichever coordinate direction is least aligned with
/// `n`, so the cross product never degenerates.
fn tangent_frame(n: Vec3) -> (Vec3, Vec3) {
    let seed = if n.x.abs() <= n.y.abs() && n.x.abs() <= n.z.abs() {
        Vec3::new(1.0, 0.0, 0.0)
    } else if n.y.abs() <= n.z.abs() {
        Vec3::new(0.0, 1.0, 0.0)
    } else {
        Vec3::new(0.0, 0.0, 1.0)
    };
    let t1 = n.cross(seed).normalised();
    let t2 = n.cross(t1);
    (t1, t2)
}

/// Which cell contains `p`, or `None`.
///
/// SPEC-LIT S66.6: **this runs on the host, at setup, and nowhere else.** The
/// tracking kernel never searches - every parcel is born in a known cell and
/// keeps its cell current by walking - so the `O(n_cells x faces)` cost here
/// is paid once per injector and never in the time loop. Written as a
/// half-space test over the same cell -> face CSR the walk uses, so a point
/// exactly on a face belongs to the lower-numbered cell that claims it, which
/// makes the answer independent of the order the cells are visited.
#[must_use]
pub fn locate_cell(hm: &HostMesh, p: Vec3) -> Option<usize> {
    for c in 0..hm.n_cells {
        let mut inside = true;
        for k in hm.cf_offset[c] as usize..hm.cf_offset[c + 1] as usize {
            let f = hm.cf_face[k] as usize;
            let sgn: Scalar = if hm.cf_own[k] != 0 { 1.0 } else { -1.0 };
            let n = hm.sf[f] * sgn;
            if (p - hm.cf[f]).dot(n) > 0.0 {
                inside = false;
                break;
            }
        }
        if inside {
            for k in hm.bcf_offset[c] as usize..hm.bcf_offset[c + 1] as usize {
                let bf = hm.bcf_face[k] as usize;
                if (p - hm.b_cf[bf]).dot(hm.b_sf[bf]) > 0.0 {
                    inside = false;
                    break;
                }
            }
        }
        if inside {
            return Some(c);
        }
    }
    None
}

// ==========================================================================
//  SPEC-LIT (66.4): the analytic terminal velocity, on the host
// ==========================================================================

/// `K = rho_g C_d |u_rel|`, kg/(m2 s) - the host mirror of `parcelDragK` in
/// `cuda/parcels.cu`, and the reason the kernel has no removable singularity
/// at `Re -> 0`: the Stokes branch is `24 mu / d`, with no division by a
/// relative speed anywhere.
#[must_use]
pub fn drag_k(model: DragModel, rho_g: Scalar, mu: Scalar, d: Scalar, mag_urel: Scalar) -> Scalar {
    if model == DragModel::None {
        return 0.0;
    }
    let re = rho_g * mag_urel * d / mu;
    if model == DragModel::Stokes || re < 1.0 {
        return 24.0 * mu / d;
    }
    if re <= 1000.0 {
        return 24.0 * mu * (0.85 + 0.15 * re.powf(0.687)) / d;
    }
    0.44 * rho_g * mag_urel
}

/// SPEC-LIT (66.4): the terminal velocity a parcel released in still gas must
/// reach, from the exact force balance
///
/// ```text
///   (1/2) rho C_d(Re_t) A_pc u_t^2 = m_p |g| (1 - rho/rho_l)
/// ```
///
/// solved as the fixed point `u_t = |a_g| tau_p(u_t)` - which is the same
/// fixed point the exponential update of (66.5) converges to, so this is the
/// analytic statement of what the kernel must produce and not a
/// re-implementation of it.
///
/// **It is independent of the added-mass coefficient**: `a_g` carries
/// `1/(1 + C_am rho/rho_l)` and `tau_p` carries `(rho_l + C_am rho)`, and the
/// two cancel exactly. Added mass changes the approach, never the
/// destination.
///
/// Note the buoyancy factor `(1 - rho/rho_l)`. The design note that preceded
/// this section quotes `u_t = sqrt(4 rho_l g d/(3 rho C_d))`, which drops it;
/// for water in air that is a 0.06 % error, and for a droplet in a liquid
/// carrier it is a first-order one.
#[must_use]
pub fn terminal_velocity(
    model: DragModel,
    rho_g: Scalar,
    rho_l: Scalar,
    mu: Scalar,
    d: Scalar,
    g: Scalar,
) -> Scalar {
    let a_g = g * (1.0 - rho_g / rho_l);
    if a_g <= 0.0 || model == DragModel::None {
        return 0.0;
    }
    // u = a_g * tau = a_g * (4/3) rho_l d / K(u). Damped fixed point, since
    // the Schiller-Naumann branch's map has |f'| = 0.687 < 1 but the first
    // iterate from u = 0 lands in the Stokes branch and can overshoot.
    let mut u = 1e-6;
    for _ in 0..500 {
        let k = drag_k(model, rho_g, mu, d, u);
        let next = a_g * (4.0 / 3.0) * rho_l * d / k;
        let step = 0.5 * (next - u);
        u += step;
        if step.abs() <= 1e-15 * u.abs() {
            break;
        }
    }
    u
}
