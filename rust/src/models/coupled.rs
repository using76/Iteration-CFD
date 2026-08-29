// meteor-cfd - Copyright (c) 2026 주식회사 메테오시뮬레이션 (Meteo Simulation Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.

//! One turbulence closure, driven by a coupled solver - SPEC-LIT §30.2.
//!
//! Written from:
//!   ofgpu `SPEC-LIT.md` §30.2 - the trait this file defines, and the
//!     requirement that a buoyant/fire case asking for SST or LES must
//!     actually get it rather than the k-epsilon every coupled driver built
//!     directly until now
//!   ofgpu `SPEC-LIT.md` §17 - the buoyancy production `G_b`, and which
//!     equation of which model it enters
//!   ofgpu `SPEC-LIT.md` §6.3, §6.6 - SST and the wall distance it needs at
//!     setup
//! No GPL-licensed source was consulted.
//!
//! # The failure this removes
//!
//! `src/bin/buoyant.rs` and `src/bin/fire.rs` used to call `KEpsilon::new`
//! directly, unconditionally - the case's own `constant/momentumTransport`
//! was never consulted. A case asking for `kOmegaSST` or `LES` got standard
//! k-epsilon, silently, which is exactly the substitution SPEC-LIT §13.4
//! forbids. `crate::models::registry::select_turbulence_model` already reads
//! the setting correctly (`src/bin/k_epsilon.rs` and `src/bin/k_omega.rs`
//! use it); the two coupled drivers just never called it.
//!
//! # Why a trait, and why THIS one
//!
//! `KEpsilon`, `KOmega` and `KOmegaSst` deliberately share no trait today
//! (`src/models/mod.rs` explains why: one carries `epsilon`, the others
//! `omega`, and forcing a `dissipation_field()` accessor would paper over
//! that). That reasoning is about hand-written call sites that already know
//! which concrete model they are driving - `src/bin/k_omega.rs` is one. A
//! coupled driver is not: which model it runs is a runtime fact read out of
//! a case file, so IT needs the `dyn` dispatch the standalone drivers do not,
//! and can afford the one virtual call per OUTER iteration this costs -
//! `src/bin/bench.rs` already made the identical trade for the same reason
//! (see that file's own two-method trait).
//!
//! [`CoupledTurbulence`] is kept to exactly what a coupled driver's outer
//! loop, its writer seam and its `.mcr` checkpoint need, and no more:
//!
//! * `correct` - advance the model one outer step, buoyancy included when the
//!   case has gravity and a temperature.
//! * `nut` - what the momentum and energy equations read back.
//! * `name` - the run banner (SPEC-LIT §30.3's selection test reads this).
//! * `output_fields` / `output_fields_mut` - name -> field, for the writer
//!   seam ([`crate::io::WriteCtx`]/`OutputField`) and the `.mcr` restart,
//!   which both want the SAME set of fields whichever model is running and
//!   otherwise have to be told about each one by name in the driver.
//! * `initialise` - the one-time setup every model needs after its `0/`
//!   fields are uploaded, before the first `correct`.
//!
//! # `ThermalCtx`, and the T_ref that is deliberately not in it
//!
//! SPEC-LIT §17 gives the buoyancy production as
//! `G_b = (nu_t/Pr_t) g.grad(T)/T` - the LOCAL temperature, nowhere a
//! reference temperature. `T_ref` enters SPEC-LIT §9's momentum body force
//! `b = g(T_ref/T - 1)`, which the coupled driver already applies to the
//! velocity equation on its own, upstream of anything a turbulence model
//! touches. Carrying an unused `t_ref` here so the struct "looked complete"
//! would be the kind of dead field this codebase's own house rules warn
//! against; the reason it is absent is this paragraph.
//!
//! # Which buoyancy settings live where
//!
//! `g` and `Pr_t` travel on [`ThermalCtx`] because they are properties of the
//! CASE's thermal setup that every model-wrapper needs on every call, exactly
//! as the temperature field itself does. `C_3`'s convention, the stable-branch
//! switch and the temperature floor are case settings too, but they are read
//! ONCE at construction (`registry::build_coupled`) and carried in each
//! wrapper's own small [`BuoyancySettings`], because there is no reason to
//! thread three rarely-changed knobs through the hot per-iteration call when
//! one struct field does it once.

use crate::device::{DevBuf, Gpu};
use crate::error::Result;
use crate::field::GpuScalarField;
use crate::models::les::Les;
use crate::models::{KEpsilon, KOmega, KOmegaSst, LaunderSharmaKE, RealizableKe, RngKe};
use crate::solver::SolverPerformance;
use crate::turbulence::{BuoyancyProduction, C3Mode, FlowState};
use crate::{Scalar, Vec3};

// ==========================================================================
//  ThermalCtx
// ==========================================================================

/// What the buoyancy production of SPEC-LIT §17 needs from the ENERGY side of
/// a coupled solver, gathered so `CoupledTurbulence::correct` takes one
/// optional argument instead of three.
///
/// `None` at the call site (rather than an isothermal `ThermalCtx`) is what
/// an isothermal case passes - `buoyant`/`fire` always solve a temperature,
/// so today `None` in practice means "this case has zero gravity", handled
/// exactly as the direct-`KEpsilon` code path always has: the models below
/// skip the whole term rather than compute a zero.
pub struct ThermalCtx<'a> {
    /// The temperature field, with its boundary conditions already
    /// evaluated - `crate::turbulence::RasCore::update_buoyancy_production`
    /// reads `t.bf` directly, so a stale boundary near a hot inlet is where
    /// this term's sign would go wrong first.
    pub t: &'a GpuScalarField,
    /// `constant/g`.
    pub g: Vec3,
    /// The turbulent Prandtl number for heat - the SAME constant the energy
    /// equation itself diffuses `T` with (SPEC-LIT §15.6: one constant, one
    /// value, reaching every equation that uses it).
    pub prt: Scalar,
}

/// The buoyancy knobs SPEC-LIT §17 leaves to the case, read once at
/// construction rather than threaded through every `correct` call - see the
/// module doc's "Which buoyancy settings live where".
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BuoyancySettings {
    pub c3: C3Mode,
    pub epsilon_stable_branch: bool,
    pub t_min: Scalar,
}

impl Default for BuoyancySettings {
    fn default() -> Self {
        let d = BuoyancyProduction::default();
        Self {
            c3: d.c3,
            epsilon_stable_branch: d.epsilon_stable_branch,
            t_min: d.t_min,
        }
    }
}

impl BuoyancySettings {
    fn production(&self, ctx: &ThermalCtx) -> BuoyancyProduction {
        BuoyancyProduction {
            g: ctx.g,
            prt: ctx.prt,
            c3: self.c3,
            epsilon_stable_branch: self.epsilon_stable_branch,
            t_min: self.t_min,
        }
    }
}

// ==========================================================================
//  The trait
// ==========================================================================

/// What SPEC-LIT §27's eddy-dissipation combustion model needs from a
/// turbulence closure to build its mixing rate, in whatever form THIS
/// model's own closure gives it - so a caller like `ofgpu-fire` reads one of
/// these off [`CoupledTurbulence::combustion_mixing`] and hands it straight
/// to the matching [`crate::combustion::Combustion`] entry point, never
/// downcasting to a concrete model to get there (the same discipline
/// [`ThermalCtx`]/`nut()` already keep).
///
/// Every field/coefficient here is borrowed, not computed - this is a
/// dispatch key, not a rate. Computing the rate itself (`eps/k`,
/// `beta_star*omega`, `C_EDM'*|S|`) stays inside `Combustion`, which is
/// where SPEC-LIT §27 lives and where the availability clip and species
/// bookkeeping already are.
pub enum CombustionMixing<'a> {
    /// k-epsilon (SPEC-LIT §6.1): feed [`crate::combustion::Combustion::react_rans`].
    Epsilon {
        k: &'a GpuScalarField,
        epsilon: &'a GpuScalarField,
    },
    /// k-omega / k-omega-SST (SPEC-LIT §6.2, §6.3): neither transports
    /// `epsilon`, but both carry the Wilcox identity
    /// `epsilon = beta_star*k*omega` in their own `k` equation - feed
    /// [`crate::combustion::Combustion::react_rans_omega`], which applies
    /// exactly that substitution.
    Omega { omega: &'a GpuScalarField, beta_star: Scalar },
    /// LES (SPEC-LIT §6.5): the resolved strain-rate magnitude stands in for
    /// the eddy turnover rate, SPEC-LIT §27's own *DESIGN* note for an LES
    /// cell - feed [`crate::combustion::Combustion::react_les`].
    Strain(&'a DevBuf<Scalar>),
    /// laminar: `nu_t = 0` carries no mixing time scale at all. A caller
    /// that lets `-combustion` run against this model would be reporting a
    /// reaction rate with no closure behind it - SPEC-LIT §13.4: refuse the
    /// COMBINATION by name rather than silently returning a zero rate that
    /// reads as "correctly modelled, and simply not reacting".
    None,
}

/// One turbulence closure, driven generically by a coupled solver -
/// SPEC-LIT §30.2. See the module doc for why this exists and why it is
/// shaped the way it is.
pub trait CoupledTurbulence {
    /// Bound the initial fields, evaluate their boundaries, and build the
    /// first `nu_t`. Call once, after every field `output_fields_mut` names
    /// has been uploaded from the case's `0/` directory (or a `.mcr`
    /// restart), before the first `correct`.
    fn initialise(&mut self, gpu: &Gpu, flow: &FlowState) -> Result<()>;

    /// Advance the model one outer step. `thermal` carries the buoyancy
    /// production of SPEC-LIT §17; pass `None` for an isothermal case or one
    /// with no gravity, and the term is skipped exactly as it always was.
    ///
    /// Returns `(dissipation, k)` linear-solver performance, in that order -
    /// `(epsilon, k)` for k-epsilon, `(omega, k)` for k-omega and SST. Every
    /// concrete model already returns exactly this pair from its own
    /// `correct`/`correct_buoyant` (see e.g. [`KEpsilon::correct_buoyant`]'s
    /// doc for why that order), so a driver that used to match on the
    /// concrete type keeps printing the same two residual columns after
    /// switching to this trait.
    fn correct(
        &mut self,
        gpu: &Gpu,
        flow: &FlowState,
        thermal: Option<&ThermalCtx>,
    ) -> Result<(SolverPerformance, SolverPerformance)>;

    /// The eddy viscosity the momentum and energy equations read.
    fn nut(&self) -> &GpuScalarField;

    /// The model's name, for the run banner - SPEC-LIT §30.3's selection
    /// test reads this alongside a field difference against a k-epsilon run
    /// on the same case.
    fn name(&self) -> &str;

    /// name -> field, every field this model owns - the writer seam and the
    /// `.mcr` restart checkpoint's complete view of it, `nut` included.
    fn output_fields(&self) -> Vec<(&'static str, &GpuScalarField)>;

    /// [`Self::output_fields`]'s mutable twin - for `0/` upload and `.mcr`
    /// restore. The set of names is identical; see the module doc for why
    /// this cannot be the same method as the shared-reference one.
    fn output_fields_mut(&mut self) -> Vec<(&'static str, &mut GpuScalarField)>;

    /// SPEC-LIT §27's combustion mixing rate, in whatever form this model's
    /// own closure provides it - see [`CombustionMixing`]'s doc for why this
    /// exists and what each arm feeds.
    fn combustion_mixing(&self) -> CombustionMixing<'_>;
}

// ==========================================================================
//  kEpsilon
// ==========================================================================

/// [`KEpsilon`] behind [`CoupledTurbulence`].
pub struct CoupledKEpsilon<'m> {
    model: KEpsilon<'m>,
    buoy: Option<BuoyancySettings>,
}

impl<'m> CoupledKEpsilon<'m> {
    /// `buoy = None` is a case with no gravity: `correct` never builds a
    /// [`BuoyancyProduction`] and the underlying model's `G_b` machinery is
    /// never switched on, which is what an isothermal or zero-`g` case wants
    /// - no gradient computed and multiplied by zero every iteration.
    pub fn new(model: KEpsilon<'m>, buoy: Option<BuoyancySettings>) -> Self {
        Self { model, buoy }
    }

    pub fn model(&self) -> &KEpsilon<'m> {
        &self.model
    }
    pub fn model_mut(&mut self) -> &mut KEpsilon<'m> {
        &mut self.model
    }
}

impl<'m> CoupledTurbulence for CoupledKEpsilon<'m> {
    fn initialise(&mut self, gpu: &Gpu, flow: &FlowState) -> Result<()> {
        self.model.initialise(gpu, flow)
    }

    fn correct(
        &mut self,
        gpu: &Gpu,
        flow: &FlowState,
        thermal: Option<&ThermalCtx>,
    ) -> Result<(SolverPerformance, SolverPerformance)> {
        match (self.buoy, thermal) {
            (Some(settings), Some(ctx)) => {
                self.model.set_buoyancy(settings.production(ctx))?;
                self.model.correct_buoyant(gpu, flow, Some(ctx.t))
            }
            _ => self.model.correct(gpu, flow),
        }
    }

    fn nut(&self) -> &GpuScalarField {
        self.model.nut()
    }
    fn name(&self) -> &str {
        "kEpsilon"
    }
    fn output_fields(&self) -> Vec<(&'static str, &GpuScalarField)> {
        self.model.named_fields()
    }
    fn output_fields_mut(&mut self) -> Vec<(&'static str, &mut GpuScalarField)> {
        self.model.named_fields_mut()
    }
    fn combustion_mixing(&self) -> CombustionMixing<'_> {
        CombustionMixing::Epsilon {
            k: self.model.k(),
            epsilon: self.model.epsilon(),
        }
    }
}

// ==========================================================================
//  LaunderSharmaKE - SPEC-LIT §33
// ==========================================================================

/// [`LaunderSharmaKE`] behind [`CoupledTurbulence`].
pub struct CoupledLaunderSharmaKE<'m> {
    model: LaunderSharmaKE<'m>,
    buoy: Option<BuoyancySettings>,
}

impl<'m> CoupledLaunderSharmaKE<'m> {
    pub fn new(model: LaunderSharmaKE<'m>, buoy: Option<BuoyancySettings>) -> Self {
        Self { model, buoy }
    }

    pub fn model(&self) -> &LaunderSharmaKE<'m> {
        &self.model
    }
    pub fn model_mut(&mut self) -> &mut LaunderSharmaKE<'m> {
        &mut self.model
    }
}

impl<'m> CoupledTurbulence for CoupledLaunderSharmaKE<'m> {
    fn initialise(&mut self, gpu: &Gpu, flow: &FlowState) -> Result<()> {
        self.model.initialise(gpu, flow)
    }

    fn correct(
        &mut self,
        gpu: &Gpu,
        flow: &FlowState,
        thermal: Option<&ThermalCtx>,
    ) -> Result<(SolverPerformance, SolverPerformance)> {
        match (self.buoy, thermal) {
            (Some(settings), Some(ctx)) => {
                self.model.set_buoyancy(settings.production(ctx))?;
                self.model.correct_buoyant(gpu, flow, Some(ctx.t))
            }
            _ => self.model.correct(gpu, flow),
        }
    }

    fn nut(&self) -> &GpuScalarField {
        self.model.nut()
    }
    fn name(&self) -> &str {
        "LaunderSharmaKE"
    }
    fn output_fields(&self) -> Vec<(&'static str, &GpuScalarField)> {
        self.model.named_fields()
    }
    fn output_fields_mut(&mut self) -> Vec<(&'static str, &mut GpuScalarField)> {
        self.model.named_fields_mut()
    }
    fn combustion_mixing(&self) -> CombustionMixing<'_> {
        // epsilon_tilde stands in for epsilon here (see the model's own
        // module doc): the difference is D = 2 nu |grad(sqrt k)|^2, an O(nu)
        // near-wall correction that a combustion mixing-time estimate - which
        // is never evaluated inside the viscous sublayer to begin with - does
        // not need resolved any more finely than that.
        CombustionMixing::Epsilon {
            k: self.model.k(),
            epsilon: self.model.epsilon(),
        }
    }
}

// ==========================================================================
//  realizableKE - SPEC-LIT §40
// ==========================================================================

/// [`RealizableKe`] behind [`CoupledTurbulence`].
///
/// No `buoy` field, unlike every other wrapper here, and that absence is the
/// point: SPEC-LIT §40.5 has no `G_b` term for the `C_1 S epsilon` production
/// form, Shih et al. specify none, and inventing `C_1 (eps/k) C_3 G_b` for a
/// model whose `epsilon` production is not proportional to `G` would be the
/// silent substitution §13.4 forbids. `registry::build_coupled` therefore
/// REFUSES a buoyant case under this model by name, so a `ThermalCtx` can
/// never reach here in the first place - and if one did, the `correct` below
/// would still not read it, which is why there is no field for it to be read
/// from.
pub struct CoupledRealizableKe<'m> {
    model: RealizableKe<'m>,
}

impl<'m> CoupledRealizableKe<'m> {
    pub fn new(model: RealizableKe<'m>) -> Self {
        Self { model }
    }

    pub fn model(&self) -> &RealizableKe<'m> {
        &self.model
    }
    pub fn model_mut(&mut self) -> &mut RealizableKe<'m> {
        &mut self.model
    }
}

impl<'m> CoupledTurbulence for CoupledRealizableKe<'m> {
    fn initialise(&mut self, gpu: &Gpu, flow: &FlowState) -> Result<()> {
        self.model.initialise(gpu, flow)
    }

    fn correct(
        &mut self,
        gpu: &Gpu,
        flow: &FlowState,
        _thermal: Option<&ThermalCtx>,
    ) -> Result<(SolverPerformance, SolverPerformance)> {
        self.model.correct(gpu, flow)
    }

    fn nut(&self) -> &GpuScalarField {
        self.model.nut()
    }
    fn name(&self) -> &str {
        "realizableKE"
    }
    fn output_fields(&self) -> Vec<(&'static str, &GpuScalarField)> {
        self.model.named_fields()
    }
    fn output_fields_mut(&mut self) -> Vec<(&'static str, &mut GpuScalarField)> {
        self.model.named_fields_mut()
    }
    fn combustion_mixing(&self) -> CombustionMixing<'_> {
        CombustionMixing::Epsilon {
            k: self.model.k(),
            epsilon: self.model.epsilon(),
        }
    }
}

// ==========================================================================
//  RNGkEpsilon - SPEC-LIT §41
// ==========================================================================

/// [`RngKe`] behind [`CoupledTurbulence`].
///
/// Buoyancy IS carried here (SPEC-LIT §41.5): `C_e1 (eps/k) G` is §6.1's
/// production form exactly, so §17's `C_1 (eps/k) C_3 G_b` transfers with
/// `C_1 = C_e1` and no new physics is invented.
pub struct CoupledRngKe<'m> {
    model: RngKe<'m>,
    buoy: Option<BuoyancySettings>,
}

impl<'m> CoupledRngKe<'m> {
    pub fn new(model: RngKe<'m>, buoy: Option<BuoyancySettings>) -> Self {
        Self { model, buoy }
    }

    pub fn model(&self) -> &RngKe<'m> {
        &self.model
    }
    pub fn model_mut(&mut self) -> &mut RngKe<'m> {
        &mut self.model
    }
}

impl<'m> CoupledTurbulence for CoupledRngKe<'m> {
    fn initialise(&mut self, gpu: &Gpu, flow: &FlowState) -> Result<()> {
        self.model.initialise(gpu, flow)
    }

    fn correct(
        &mut self,
        gpu: &Gpu,
        flow: &FlowState,
        thermal: Option<&ThermalCtx>,
    ) -> Result<(SolverPerformance, SolverPerformance)> {
        match (self.buoy, thermal) {
            (Some(settings), Some(ctx)) => {
                self.model.set_buoyancy(settings.production(ctx))?;
                self.model.correct_buoyant(gpu, flow, Some(ctx.t))
            }
            _ => self.model.correct(gpu, flow),
        }
    }

    fn nut(&self) -> &GpuScalarField {
        self.model.nut()
    }
    fn name(&self) -> &str {
        "RNGkEpsilon"
    }
    fn output_fields(&self) -> Vec<(&'static str, &GpuScalarField)> {
        self.model.named_fields()
    }
    fn output_fields_mut(&mut self) -> Vec<(&'static str, &mut GpuScalarField)> {
        self.model.named_fields_mut()
    }
    fn combustion_mixing(&self) -> CombustionMixing<'_> {
        CombustionMixing::Epsilon {
            k: self.model.k(),
            epsilon: self.model.epsilon(),
        }
    }
}

// ==========================================================================
//  kOmega
// ==========================================================================

/// [`KOmega`] behind [`CoupledTurbulence`].
pub struct CoupledKOmega<'m> {
    model: KOmega<'m>,
    buoy: Option<BuoyancySettings>,
}

impl<'m> CoupledKOmega<'m> {
    pub fn new(model: KOmega<'m>, buoy: Option<BuoyancySettings>) -> Self {
        Self { model, buoy }
    }

    pub fn model(&self) -> &KOmega<'m> {
        &self.model
    }
    pub fn model_mut(&mut self) -> &mut KOmega<'m> {
        &mut self.model
    }
}

impl<'m> CoupledTurbulence for CoupledKOmega<'m> {
    fn initialise(&mut self, gpu: &Gpu, flow: &FlowState) -> Result<()> {
        self.model.initialise(gpu, flow)
    }

    fn correct(
        &mut self,
        gpu: &Gpu,
        flow: &FlowState,
        thermal: Option<&ThermalCtx>,
    ) -> Result<(SolverPerformance, SolverPerformance)> {
        match (self.buoy, thermal) {
            (Some(settings), Some(ctx)) => {
                self.model.set_buoyancy(settings.production(ctx))?;
                self.model.correct_buoyant(gpu, flow, Some(ctx.t))
            }
            _ => self.model.correct(gpu, flow),
        }
    }

    fn nut(&self) -> &GpuScalarField {
        self.model.nut()
    }
    fn name(&self) -> &str {
        "kOmega"
    }
    fn output_fields(&self) -> Vec<(&'static str, &GpuScalarField)> {
        self.model.named_fields()
    }
    fn output_fields_mut(&mut self) -> Vec<(&'static str, &mut GpuScalarField)> {
        self.model.named_fields_mut()
    }
    fn combustion_mixing(&self) -> CombustionMixing<'_> {
        CombustionMixing::Omega {
            omega: self.model.omega(),
            beta_star: self.model.coeffs().beta_star,
        }
    }
}

// ==========================================================================
//  kOmegaSST
// ==========================================================================

/// [`KOmegaSst`] behind [`CoupledTurbulence`]. See
/// `registry::build_coupled` for where the wall distance SPEC-LIT §6.6
/// requires is computed - once, at setup, before this wrapper exists.
pub struct CoupledKOmegaSst<'m> {
    model: KOmegaSst<'m>,
    buoy: Option<BuoyancySettings>,
}

impl<'m> CoupledKOmegaSst<'m> {
    pub fn new(model: KOmegaSst<'m>, buoy: Option<BuoyancySettings>) -> Self {
        Self { model, buoy }
    }

    pub fn model(&self) -> &KOmegaSst<'m> {
        &self.model
    }
    pub fn model_mut(&mut self) -> &mut KOmegaSst<'m> {
        &mut self.model
    }
}

impl<'m> CoupledTurbulence for CoupledKOmegaSst<'m> {
    fn initialise(&mut self, gpu: &Gpu, flow: &FlowState) -> Result<()> {
        self.model.initialise(gpu, flow)
    }

    fn correct(
        &mut self,
        gpu: &Gpu,
        flow: &FlowState,
        thermal: Option<&ThermalCtx>,
    ) -> Result<(SolverPerformance, SolverPerformance)> {
        match (self.buoy, thermal) {
            (Some(settings), Some(ctx)) => {
                self.model.set_buoyancy(settings.production(ctx))?;
                self.model.correct_buoyant(gpu, flow, Some(ctx.t))
            }
            _ => self.model.correct(gpu, flow),
        }
    }

    fn nut(&self) -> &GpuScalarField {
        self.model.nut()
    }
    fn name(&self) -> &str {
        "kOmegaSST"
    }
    fn output_fields(&self) -> Vec<(&'static str, &GpuScalarField)> {
        self.model.named_fields()
    }
    fn output_fields_mut(&mut self) -> Vec<(&'static str, &mut GpuScalarField)> {
        self.model.named_fields_mut()
    }
    fn combustion_mixing(&self) -> CombustionMixing<'_> {
        CombustionMixing::Omega {
            omega: self.model.omega(),
            beta_star: self.model.coeffs().beta_star,
        }
    }
}

// ==========================================================================
//  LES
// ==========================================================================

/// [`Les`] behind [`CoupledTurbulence`] - SPEC-LIT §30.2.
///
/// # Buoyancy: why `thermal` is read by NONE of the three submodels
///
/// SPEC-LIT §17 gives `G_b` as a production term entering a TRANSPORT
/// equation - `k`'s, `epsilon`'s, `omega`'s. Smagorinsky and WALE have no
/// such equation (SPEC-LIT §30.2 says so explicitly: "no transport equation,
/// so no G_b term ... buoyant force still acts through the resolved
/// momentum equation"), and this solver's Deardorff, as [`Les`] actually
/// implements it (see that module's own doc, and FDS, its cited reference),
/// is the SAME shape: `k_sgs` is a diagnostic estimate rebuilt from the
/// test-filtered velocity fresh every step, with no state carried between
/// steps for a production term to accumulate into. Adding `G_b` to that
/// estimate would not be "taking buoyancy into the SGS-TKE equation" - there
/// is no equation, so there is nothing to add it to - it would be inventing
/// a fudge on top of a number that is discarded and recomputed next step
/// regardless, which is exactly the kind of invented term SPEC-LIT §30.2
/// asks the algebraic models to avoid. All three LES submodels therefore
/// take the buoyant force through the resolved momentum equation ONLY, and
/// `correct` below never builds a [`BuoyancyProduction`] - `thermal` is
/// accepted (the trait requires it) and ignored, exactly as
/// [`CoupledLaminar::correct`] ignores it.
pub struct CoupledLes<'m> {
    model: Les<'m>,
}

impl<'m> CoupledLes<'m> {
    pub fn new(model: Les<'m>) -> Self {
        Self { model }
    }

    pub fn model(&self) -> &Les<'m> {
        &self.model
    }
    pub fn model_mut(&mut self) -> &mut Les<'m> {
        &mut self.model
    }

    /// Rebuild `nu_t` from the current resolved velocity, then the
    /// Werner-Wengle wall pass on whichever faces `nut`'s own patch type
    /// asked for one (SPEC-LIT §30.1) - the one step both `initialise` and
    /// `correct` need, with no buoyancy term either reads (see the type doc).
    fn step(&mut self, gpu: &Gpu, flow: &FlowState) -> Result<()> {
        self.model.correct(gpu, flow)?;
        self.model.apply_werner_wengle_wall_function(gpu, flow.u, flow.nu)
    }
}

impl<'m> CoupledTurbulence for CoupledLes<'m> {
    fn initialise(&mut self, gpu: &Gpu, flow: &FlowState) -> Result<()> {
        self.step(gpu, flow)
    }

    fn correct(
        &mut self,
        gpu: &Gpu,
        flow: &FlowState,
        _thermal: Option<&ThermalCtx>,
    ) -> Result<(SolverPerformance, SolverPerformance)> {
        self.step(gpu, flow)?;
        // No linear solve: an algebraic model has nothing to report here,
        // which is the truth rather than a placeholder - see
        // `CoupledLaminar::correct`'s identical choice.
        Ok((SolverPerformance::default(), SolverPerformance::default()))
    }

    fn nut(&self) -> &GpuScalarField {
        self.model.nut()
    }
    fn name(&self) -> &str {
        self.model.model().name()
    }
    fn output_fields(&self) -> Vec<(&'static str, &GpuScalarField)> {
        // Deardorff's `k_sgs` is a diagnostic `DevBuf<Scalar>`
        // ([`Les::k_sgs`]), not a `GpuScalarField` with its own boundary
        // machinery, so it cannot be named here the way `nut` can - the
        // writer seam and the `.mcr` restart both want a real field, and
        // inventing boundary conditions for a purely-diagnostic buffer would
        // be a number a user could plot and believe is more than it is.
        vec![("nut", self.model.nut())]
    }
    fn output_fields_mut(&mut self) -> Vec<(&'static str, &mut GpuScalarField)> {
        vec![("nut", self.model.nut_mut())]
    }
    fn combustion_mixing(&self) -> CombustionMixing<'_> {
        // The strain rate of the LAST `correct`/`initialise` call - the same
        // one-iteration lag `nut()` itself already carries into momentum and
        // energy, and true of all three LES submodels alike (SPEC-LIT §27's
        // *DESIGN* note names no distinction between them here).
        CombustionMixing::Strain(self.model.strain_rate())
    }
}

// ==========================================================================
//  Laminar
// ==========================================================================

/// `nu_t = 0` everywhere, for `simulationType laminar;` or
/// `RAS { turbulence off; }` reaching a coupled driver through the same
/// registry path as every real model - SPEC-LIT §13.4: this used to be a
/// separate, easy-to-forget branch in each driver, or not handled at all.
pub struct CoupledLaminar {
    nut: GpuScalarField,
}

impl CoupledLaminar {
    pub fn new(gpu: &Gpu, mesh: &crate::mesh::GpuMesh) -> Result<Self> {
        Ok(Self {
            nut: GpuScalarField::zeros(gpu, mesh, "nut")?,
        })
    }
}

impl CoupledTurbulence for CoupledLaminar {
    fn initialise(&mut self, _gpu: &Gpu, _flow: &FlowState) -> Result<()> {
        Ok(())
    }
    fn correct(
        &mut self,
        _gpu: &Gpu,
        _flow: &FlowState,
        _thermal: Option<&ThermalCtx>,
    ) -> Result<(SolverPerformance, SolverPerformance)> {
        // Nothing is solved: `nu_t = 0` is the whole model. Both performance
        // records report zero iterations at zero residual, which is the
        // truth rather than a placeholder - a driver printing them alongside
        // a real model's sees exactly that: no work was done here.
        Ok((SolverPerformance::default(), SolverPerformance::default()))
    }
    fn nut(&self) -> &GpuScalarField {
        &self.nut
    }
    fn name(&self) -> &str {
        "laminar"
    }
    fn output_fields(&self) -> Vec<(&'static str, &GpuScalarField)> {
        vec![("nut", &self.nut)]
    }
    fn output_fields_mut(&mut self) -> Vec<(&'static str, &mut GpuScalarField)> {
        vec![("nut", &mut self.nut)]
    }
    fn combustion_mixing(&self) -> CombustionMixing<'_> {
        CombustionMixing::None
    }
}
