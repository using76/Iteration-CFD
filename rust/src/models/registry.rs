// meteor-cfd - Copyright (c) 2026 주식회사 메테오시뮬레이션 (Meteo Simulation Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.

//! Which turbulence model the case asked for, and whether it runs at all.
//!
//! Written from:
//!   ofgpu `SPEC-LIT.md` §6 (the models), §13.4 (a setting the solver cannot
//!     honour must fail loudly, naming the setting and the alternatives)
//!   Launder & Spalding, *Comput. Methods Appl. Mech. Eng.* 3 (1974) 269-289
//!   Wilcox, *Turbulence Modeling for CFD*, DCW Industries - the 1988 k-omega
//!   Menter, *AIAA J.* 32 (1994) 1598-1605, and Menter, Kuntz & Langtry,
//!     *Turbulence, Heat and Mass Transfer* 4 (2003) - `kOmegaSST`, which this
//!     registry now dispatches to rather than refusing
//!   Smagorinsky, *Mon. Weather Rev.* 91 (1963) 99-164; Nicoud & Ducros,
//!     *Flow Turbul. Combust.* 62 (1999) 183-200; Deardorff,
//!     *Boundary-Layer Meteorol.* 18 (1980) 495-527 - the three LES closures
//!     `simulationType LES;` now reaches
//!   ofgpu `SPEC-LIT.md` §6.3, §6.5 and §16
//! No GPL-licensed source was consulted.
//!
//! # The failure this removes
//!
//! `RAS { model ...; }` used to have **no effect at all**: the name was read
//! and used only as a string prefix for the coefficient lookup, so
//! `kOmegaSST`, `SpalartAllmaras`, `realizableKE` and the typo `kepsilon` all
//! behaved identically, and which model ran was decided by which binary was
//! invoked. `RAS { turbulence off; }` and `simulationType laminar;` were
//! ignored the same way - the model ran regardless, bit for bit.
//!
//! Both are the silent substitution SPEC-LIT §13.4 forbids: a case that asks
//! for SST and gets standard k-epsilon converges, plots and is believed.
//!
//! # What a registry buys when there is no trait
//!
//! `KEpsilon` and `KOmega` deliberately share no trait
//! (`src/models/mod.rs` says why: one carries `epsilon` and the other
//! `omega`, and a `dissipation_field()` accessor would mean two different
//! things). So this is a registry of *identities*, not of constructors that
//! return a boxed object: [`RasModel`] says which model the case named, and
//! the driver - which knows the concrete type it can build - matches on it.
//! The dispatch is real either way; what is avoided is a virtual call in an
//! inner loop that runs a hundred kernel launches deep.

use crate::device::Gpu;
use crate::error::{Error, Result};
use crate::field_setup::{wall_coeffs_from_case, NutRoughness, WallFaces};
use crate::io::case::{model_coeff, CaseControls};
use crate::io::contract::{unsupported, unsupported_note};
use crate::io::dict::FoamDict;
use crate::les::{BaseDelta, DeltaSpec, SmoothSpec};
use crate::mesh::{GpuMesh, HostMesh};
use crate::models::coupled::{
    BuoyancySettings, CoupledKEpsilon, CoupledKOmega, CoupledKOmegaSst, CoupledLaminar,
    CoupledLaunderSharmaKE, CoupledLes, CoupledRealizableKe, CoupledRngKe,
    CoupledSpalartAllmaras, CoupledTurbulence,
};
use crate::models::des::{DesBranch, DesCoeffs, HybridBackground, HybridDelta};
use crate::models::les::{Les, LesCoeffs, LesModel};
use crate::models::spalart_allmaras::{SaCoeffs, SaVariant};
use crate::models::transition::{LangtryMenter, LmCoeffs, LmControls};
use crate::models::{
    KEpsilon, KEpsilonCoeffs, KOmega, KOmegaCoeffs, KOmegaSst, KOmegaSstCoeffs, LaunderSharmaKE,
    RealizableKe, RealizableKeCoeffs, RngKe, RngKeCoeffs, SpalartAllmaras,
};
use crate::turbulence::C3Mode;
use crate::{Label, Scalar, Vec3};

/// A model ofgpu implements.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RasModel {
    /// No closure at all: `nu_t = 0` everywhere, and the momentum equation
    /// sees the molecular viscosity alone.
    Laminar,
    /// Standard k-epsilon - SPEC-LIT §6.1, Launder & Spalding (1974).
    KEpsilon,
    /// Launder-Sharma low-Reynolds-number k-epsilon - SPEC-LIT §33, Launder
    /// & Sharma (1974). The only model `wallTreatment lowRe` is valid under
    /// (`io::case::validate_low_re_wall_treatment`'s `LOW_RE_VALID`): it
    /// integrates through the viscous sublayer instead of assuming a log
    /// layer, which is what `lowRe`'s "no wall model, the mesh resolves it"
    /// requires and neither `KEpsilon` nor `KOmega`/`KOmegaSST` can provide.
    LaunderSharmaKE,
    /// Shih, Liou, Shabbir, Yang & Zhu's realizable k-epsilon - SPEC-LIT
    /// §40. §6.1's two equations with `C_mu` a FIELD, the `epsilon`
    /// production written `C_1 S eps`, and the sink denominator
    /// `k + sqrt(nu eps)`. It has no buoyancy production of its own
    /// ([`refuse_realizable_ke_buoyancy`] says why), which is the one thing
    /// that stops it being a drop-in replacement for `kEpsilon` everywhere.
    RealizableKE,
    /// Yakhot & Orszag's RNG k-epsilon - SPEC-LIT §41. §6.1's two equations
    /// with the `R` term absorbed into a per-cell `C_e2*`, and diffusivities
    /// `alpha (nu + nu_t)` rather than `nu + nu_t/sigma`.
    RNGkEpsilon,
    /// Wilcox k-omega, the 1988 form - SPEC-LIT §6.2.
    KOmega,
    /// Menter k-omega SST, the 2003 revision - SPEC-LIT §6.3. Needs the wall
    /// distance of §6.6, which a driver must compute before constructing it.
    KOmegaSST,
    /// Langtry & Menter's four-equation gamma-Re_theta transition model on
    /// the k-omega SST background - SPEC-LIT §88.
    ///
    /// Two more transport equations than §6.3, `gamma` and `ReThetat`, and
    /// three stamps into SST's own assembly. It is a separate variant rather
    /// than a flag on `KOmegaSST` for the reason `HybridSst` is: what
    /// changes is the set of `0/` files a driver must find, and a driver
    /// that matched `RasModel::KOmegaSST` would run a transitional case
    /// fully turbulent from the leading edge and say nothing.
    KOmegaSstLM,
    /// Spalart-Allmaras - SPEC-LIT §56. ONE transport equation, for `nuTilda`,
    /// which is not a dissipation rate and not an eddy viscosity; and no wall
    /// function at all, because `nu~ = 0` is an exact Dirichlet condition.
    /// Needs the wall distance of §6.6.
    SpalartAllmaras,
    /// A detached-eddy hybrid on the Spalart-Allmaras background - SPEC-LIT
    /// §57. Which of DES97/DDES/IDDES, which filter width and which
    /// calibration are on [`TurbulenceSelection::des`]; this variant carries
    /// only "the background is SA", which is what decides the `0/` files and
    /// the model a driver builds.
    HybridSa,
    /// A detached-eddy hybrid on the k-omega SST background - SPEC-LIT §57.
    /// `sst_k_sources` is untouched: the hybrid overwrites `sp` afterwards
    /// with (57.4), and a pure SST run does not launch that kernel at all.
    HybridSst,
    /// Not a RANS model at all: the case said `simulationType LES;` and
    /// [`TurbulenceSelection::les`] carries which subgrid model and which
    /// filter width.
    ///
    /// It is a variant of this enum rather than a separate flag so that a
    /// driver which knows only about RANS *cannot* run an LES case by
    /// accident. `ofgpu-k-epsilon` matches `RasModel::KEpsilon |
    /// RasModel::Laminar` and errors on everything else; an LES case
    /// therefore stops at that match with a message naming the model, instead
    /// of being read as laminar and running with `nu_t = 0` - which is a
    /// plausible, converged, wrong answer of exactly the kind SPEC-LIT §13.4
    /// exists to prevent.
    Les,
}

impl RasModel {
    /// The name a case writes to get this model.
    pub fn name(self) -> &'static str {
        match self {
            Self::Laminar => "laminar",
            Self::KEpsilon => "kEpsilon",
            Self::LaunderSharmaKE => "LaunderSharmaKE",
            Self::RealizableKE => "realizableKE",
            Self::RNGkEpsilon => "RNGkEpsilon",
            Self::KOmega => "kOmega",
            Self::KOmegaSST => "kOmegaSST",
            Self::KOmegaSstLM => "kOmegaSSTLM",
            Self::SpalartAllmaras => "SpalartAllmaras",
            // The branch is on `TurbulenceSelection::des`; a bare
            // `RasModel` cannot name it, and `TurbulenceSelection::describe`
            // is what the run banner prints.
            Self::HybridSa => "SpalartAllmaras (hybrid)",
            Self::HybridSst => "kOmegaSST (hybrid)",
            Self::Les => "LES",
        }
    }

    /// The dissipation variable's field name, which is also the `0/` file a
    /// driver has to find.
    ///
    /// `LaunderSharmaKE` answers `epsilon`, the same name `KEpsilon` does -
    /// the field it transports is `epsilon_tilde`, but there is no separate
    /// `epsilonTilde` file; see [`crate::models::launder_sharma`]'s module
    /// doc for why.
    pub fn dissipation_field(self) -> Option<&'static str> {
        match self {
            Self::Laminar => None,
            Self::KEpsilon => Some("epsilon"),
            Self::LaunderSharmaKE => Some("epsilon"),
            // Both variants transport the same two fields §6.1 does, under
            // the same two names - which is the whole reason they are cheap.
            Self::RealizableKE => Some("epsilon"),
            Self::RNGkEpsilon => Some("epsilon"),
            Self::KOmega => Some("omega"),
            Self::KOmegaSST => Some("omega"),
            // The transition model transports FOUR fields and `omega` is
            // still the dissipation variable among them - `gamma` and
            // `ReThetat` are neither dissipations nor working viscosities,
            // and `transported_fields` below is where a driver reads them.
            Self::KOmegaSstLM => Some("omega"),
            // SPEC-LIT §58.1, following the design note's own recommendation:
            // `nu~` is NOT a dissipation rate, and returning `Some("nuTilda")`
            // here would make this accessor mean two different things
            // depending on who answers it - the exact overloading
            // `models/mod.rs` argues against at length. The honest answer is
            // `None`, and [`RasModel::transported_fields`] is the accessor
            // that says what a driver actually wants to know.
            Self::SpalartAllmaras | Self::HybridSa => None,
            Self::HybridSst => Some("omega"),
            // An algebraic subgrid model solves for nothing, so there is no
            // `0/` file to find. Deardorff reports a `k_sgs`, but it is an
            // estimate the model makes rather than a field it transports.
            Self::Les => None,
        }
    }

    /// Every field this model TRANSPORTS, which is also the set of `0/` files
    /// a driver has to find - SPEC-LIT §58.1.
    ///
    /// Added rather than overloading [`Self::dissipation_field`], for the
    /// reason that accessor's own doc gives: `nu~` is not a dissipation rate,
    /// and an accessor that meant "the dissipation variable" for four models
    /// and "the working variable" for a fifth would be exactly the drift
    /// `models/mod.rs` warns about. **Two accessors that mean two different
    /// things, and both honest.**
    pub fn transported_fields(self) -> &'static [&'static str] {
        match self {
            Self::Laminar | Self::Les => &[],
            Self::KEpsilon
            | Self::LaunderSharmaKE
            | Self::RealizableKE
            | Self::RNGkEpsilon => &["k", "epsilon"],
            Self::KOmega | Self::KOmegaSST | Self::HybridSst => &["k", "omega"],
            // SPEC-LIT §89.1. The order is the order they are solved in.
            Self::KOmegaSstLM => &["k", "omega", "gamma", "ReThetat"],
            Self::SpalartAllmaras | Self::HybridSa => &["nuTilda"],
        }
    }
}

/// Every spelling that selects an implemented model.
///
/// A table rather than a `match` arm because the same table is what the error
/// message prints, and a menu that has drifted from the code is worse than no
/// menu.
const REGISTRY: &[(&str, RasModel)] = &[
    ("laminar", RasModel::Laminar),
    ("kEpsilon", RasModel::KEpsilon),
    ("KEpsilon", RasModel::KEpsilon),
    ("LaunderSharmaKE", RasModel::LaunderSharmaKE),
    ("realizableKE", RasModel::RealizableKE),
    ("RealizableKE", RasModel::RealizableKE),
    ("RNGkEpsilon", RasModel::RNGkEpsilon),
    ("RNGKEpsilon", RasModel::RNGkEpsilon),
    ("kOmega", RasModel::KOmega),
    ("KOmega", RasModel::KOmega),
    ("kOmegaSST", RasModel::KOmegaSST),
    ("kOmegaSSTLM", RasModel::KOmegaSstLM),
    ("KOmegaSST", RasModel::KOmegaSST),
    ("SpalartAllmaras", RasModel::SpalartAllmaras),
    ("SpalartAllmarras", RasModel::SpalartAllmaras),
];

/// The hybrid RANS-LES models - SPEC-LIT §57, §58.2.
///
/// Reachable through `simulationType LES; LES { model ...; }` and through
/// `simulationType DES|DDES|IDDES; DES { model ...; }` alike; both spellings
/// go through the same reader, and where `simulationType` names a branch it
/// must AGREE with the one the model name carries (§58.1).
const HYBRID_REGISTRY: &[(&str, DesBranch, HybridBackground)] = &[
    ("SpalartAllmarasDES", DesBranch::Des97, HybridBackground::Sa),
    ("SpalartAllmarasDDES", DesBranch::Ddes, HybridBackground::Sa),
    ("SpalartAllmarasIDDES", DesBranch::Iddes, HybridBackground::Sa),
    ("kOmegaSSTDES", DesBranch::Des97, HybridBackground::Sst),
    ("kOmegaSSTDDES", DesBranch::Ddes, HybridBackground::Sst),
    ("kOmegaSSTIDDES", DesBranch::Iddes, HybridBackground::Sst),
];

/// Names this solver RECOGNISES but does not implement.
///
/// SPEC-LIT §13.4 distinguishes these from a name nobody has heard of, and the
/// distinction is worth keeping: telling a user that `kOmegaSST` is a real
/// model ofgpu has not got is a different message from telling them
/// `kepsilon` is not a model at all, and the second one is a typo they can fix
/// in five seconds once they are told.
const RECOGNISED_NOT_IMPLEMENTED: &[(&str, &str)] = &[
    // SPEC-LIT 89.3. `kOmegaSSTLM` used to sit here, at the head of this
    // list; it is now in REGISTRY. What replaces it is its own successor,
    // refused with the reason the successor is the better model rather than
    // with a shrug.
    (
        "kOmegaSSTGamma",
        "Menter, Smirnov, Liu & Avancha's ONE-equation gamma transition model \
         (Flow Turbul. Combust. 95 (2015) 583-619) - the successor to \
         kOmegaSSTLM, which ofgpu HAS (SPEC-LIT 88). The 2015 model drops the \
         Re_theta~ equation and with it the implicit Re_theta_eq fixed point \
         (SPEC-LIT 88.4), and it is Galilean-invariant where LM2009 is not: \
         LM2009's Tu and its time scale T read an ABSOLUTE velocity \
         magnitude, so its answer changes if the frame is translated. That is \
         a real defect of the model ofgpu does have, SPEC-LIT 88.9 measures \
         how large it is, and this refusal names it rather than pretending \
         kOmegaSSTLM is the last word",
    ),
    (
        "kOmegaSSTSAS",
        "a scale-adaptive model, which reads the von Karman length scale from \
         the SECOND velocity derivative rather than switching on the grid. \
         ofgpu's grid-switched hybrids are kOmegaSSTDDES and kOmegaSSTIDDES \
         (SPEC-LIT 57)",
    ),
    (
        "kEpsilonPhitF",
        "four equations with an elliptic relaxation whose wall boundary \
         condition couples two of them. `LaunderSharmaKE` is the low-Reynolds \
         model ofgpu has (SPEC-LIT 33)",
    ),
    (
        "v2f",
        "as kEpsilonPhitF: an elliptic-relaxation model. `LaunderSharmaKE` is \
         the low-Reynolds model ofgpu has (SPEC-LIT 33)",
    ),
    // SPEC-LIT 89.6 - the one family with no representative in this crate at
    // all, refused with WHAT IT WOULD TAKE rather than with "nothing is
    // close". The two messages differ where the two models differ and nowhere
    // else, so neither can be read as a restatement of the other.
    (
        "LRR",
        "Launder, Reece & Rodi's Reynolds-stress transport (J. Fluid Mech. 68 \
         (1975) 537-566): SEVEN transport equations - the six independent \
         components of <u_i u_j>, plus epsilon - closed by a LINEAR \
         pressure-strain model. It is not an eddy-viscosity closure, so there \
         is no nu_t to hand to the momentum equation. SPEC-LIT 89.6 sets out \
         what it would take here: a symmetric-tensor transported field, which \
         this crate has not got (every field it solves is a scalar); a \
         momentum equation that takes a stress DIVERGENCE rather than a \
         viscosity; six coupled solves per outer iteration against \
         kOmegaSST's two; wall-reflection terms that need the wall NORMAL and \
         not only the distance 6.6 computes; and a realizability guard on the \
         solved tensor, because a Reynolds-stress model can produce a \
         negative normal stress and an eddy-viscosity one cannot. Use \
         kOmegaSST (SPEC-LIT 6.3), or LaunderSharmaKE (SPEC-LIT 33) where the \
         near-wall behaviour is what matters",
    ),
    (
        "SSG",
        "Speziale, Sarkar & Gatski's Reynolds-stress transport (J. Fluid \
         Mech. 227 (1991) 245-272) - as LRR, and everything in that entry \
         applies, with a QUADRATIC pressure-strain model in place of the \
         linear one. The quadratic terms need the anisotropy tensor's second \
         and third invariants, so SSG needs the eigenstructure of the solved \
         stress and not only the stress (SPEC-LIT 89.6). Use kOmegaSST \
         (SPEC-LIT 6.3)",
    ),
];

/// Names that ARE implemented, but in the other branch of `simulationType`.
///
/// `RAS { model Smagorinsky; }` is not a typo and it is not an unimplemented
/// model - it is an LES model written under the RANS heading, and the fix is
/// one line in the same file. Telling a user "ofgpu has not got Smagorinsky"
/// when it has would send them looking for the wrong thing.
const LES_MODEL_UNDER_RAS: &[&str] = &["Smagorinsky", "WALE", "Deardorff"];

/// LES models this solver implements - SPEC-LIT §6.5.
const LES_REGISTRY: &[(&str, LesModel)] = &[
    ("Smagorinsky", LesModel::Smagorinsky),
    ("smagorinsky", LesModel::Smagorinsky),
    ("WALE", LesModel::Wale),
    ("Wale", LesModel::Wale),
    ("Deardorff", LesModel::Deardorff),
];

/// LES models this solver recognises and has not got.
///
/// Every one of them solves a transport equation or runs a dynamic procedure,
/// which is the line the three implemented models sit on the other side of:
/// all three of ours are algebraic.
const LES_RECOGNISED_NOT_IMPLEMENTED: &[&str] = &[
    "kEqn",
    "dynamicKEqn",
    "dynamicLagrangian",
    "DeardorffDiffStress",
    "Vreman",
];

/// Filter widths this solver implements - SPEC-LIT §16.
///
/// `vanDriest` and `smooth` are not widths but wrappers: each reads a base
/// width out of its own `Coeffs` sub-dictionary and modifies it, which is how
/// they are written in a case file and how §16 describes them.
const DELTA_NAMES: &[&str] = &[
    "cubeRootVol",
    "maxDeltaxyz",
    "Scotti",
    "vanDriest",
    "smooth",
];

/// Filter widths this solver recognises and has not got.
const DELTA_RECOGNISED_NOT_IMPLEMENTED: &[&str] = &[
    "PrandtlDelta",
    "maxDeltaxyzCubeRootLES",
    "cubeRootVolDelta",
];

/// Filter widths that exist here but ONLY inside a hybrid - SPEC-LIT §58.2.
///
/// (57.17) reads the wall distance and the wall-normal grid step and is
/// defined only inside IDDES's own length-scale blend, so it does not join
/// [`DELTA_NAMES`]: a pure-LES case naming it is refused with a message
/// saying where it does live, the same shape as [`LES_MODEL_UNDER_RAS`]'s.
const DELTA_HYBRID_ONLY: &[&str] = &["IDDESDelta", "IDDESDeltaSimple"];

/// The filter widths a hybrid may name - SPEC-LIT §57.4, §57.10.
const HYBRID_DELTA_NAMES: &[(&str, HybridDelta)] = &[
    ("maxDeltaxyz", HybridDelta::MaxEdge),
    ("IDDESDelta", HybridDelta::IddesFull),
    ("IDDESDeltaSimple", HybridDelta::IddesSimple),
];

/// The menu a rejected name is shown.
pub fn available_models() -> Vec<&'static str> {
    let mut v: Vec<&'static str> = vec![
        "kEpsilon",
        "LaunderSharmaKE",
        "realizableKE",
        "RNGkEpsilon",
        "kOmega",
        "kOmegaSST",
        "kOmegaSSTLM",
        "SpalartAllmaras",
        "laminar",
    ];
    v.dedup();
    v
}

/// The menu a rejected hybrid name is shown - SPEC-LIT §58.2.
pub fn available_hybrid_models() -> Vec<&'static str> {
    HYBRID_REGISTRY.iter().map(|(n, _, _)| *n).collect()
}

/// The menu a rejected `DES { delta ...; }` is shown.
pub fn available_hybrid_deltas() -> Vec<&'static str> {
    HYBRID_DELTA_NAMES.iter().map(|(n, _)| *n).collect()
}

/// The menu a rejected `LES { model ...; }` is shown.
pub fn available_les_models() -> Vec<&'static str> {
    vec!["Smagorinsky", "WALE", "Deardorff"]
}

/// The menu a rejected `LES { delta ...; }` is shown.
pub fn available_deltas() -> Vec<&'static str> {
    DELTA_NAMES.to_vec()
}

/// What the case asked for.
///
/// `Eq` is deliberately absent: [`LesSelection`] carries floating-point
/// coefficients, and two filter widths that differ in the last bit of
/// `deltaCoeff` are not the same setting however tempting it is to say so.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TurbulenceSelection {
    /// The model named in `constant/momentumTransport`.
    pub model: RasModel,
    /// False when the case switched the closure off - `simulationType
    /// laminar;` or `RAS { turbulence off; }`.
    ///
    /// A driver that sees `false` must freeze `nu_t` at zero and never call
    /// `correct`. Both settings used to be read and discarded.
    pub active: bool,

    /// `Some` exactly when `model == RasModel::Les`: which subgrid model, its
    /// coefficient, and the filter width of SPEC-LIT §16.
    pub les: Option<LesSelection>,

    /// `Some` exactly when `model` is `HybridSa` or `HybridSst` - SPEC-LIT
    /// §57. Which branch, which filter width, and the per-background
    /// calibration.
    pub des: Option<HybridSelection>,

    /// `Some` exactly when `model == RasModel::KOmegaSstLM` - SPEC-LIT §88,
    /// §89.2. The transition model's own constants and its two equations'
    /// `system/` settings.
    pub transition: Option<LmSelection>,
}

/// What a transition case asked for - SPEC-LIT §88, §89.2.
///
/// Two records rather than one, because they come from two files:
/// `constant/momentumTransport` carries the coefficients and `system/`
/// carries the solver, relaxation and convection entries for `gamma` and
/// `ReThetat`. Keeping them apart is what makes each of §89.4's pair tests
/// able to name the file it exercises.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LmSelection {
    pub coeffs: LmCoeffs,
    pub controls: LmControls,
}

impl LmSelection {
    /// The run banner's line.
    #[must_use]
    pub fn describe(&self) -> String {
        format!("kOmegaSSTLM (SPEC-LIT 88): {}", self.coeffs.describe())
    }
}

/// What a hybrid case asked for - SPEC-LIT §57, §58.1.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HybridSelection {
    pub branch: DesBranch,
    pub background: HybridBackground,
    pub delta: HybridDelta,
    pub coeffs: DesCoeffs,
    /// The background model's own constants. Only the SA background reads
    /// this; the SST one takes its coefficients through `KOmegaSstCoeffs` as
    /// it always has.
    pub sa: SaCoeffs,
}

impl HybridSelection {
    /// The run banner's line - SPEC-LIT §57.5 requires the calibration in use
    /// to be visible, because the same three names carry different numbers on
    /// the two backgrounds.
    #[must_use]
    pub fn describe(&self) -> String {
        format!(
            "{} on {} ({} branch), delta {}, {}",
            self.model_name(),
            self.background.name(),
            self.branch.name(),
            self.delta.name(),
            self.coeffs.describe(self.background)
        )
    }

    /// The name a case writes to get exactly this hybrid.
    #[must_use]
    pub fn model_name(&self) -> &'static str {
        HYBRID_REGISTRY
            .iter()
            .find(|(_, b, g)| *b == self.branch && *g == self.background)
            .map_or("<unnamed hybrid>", |(n, _, _)| *n)
    }
}

/// What an LES case asked for - SPEC-LIT §6.5 and §16.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LesSelection {
    pub model: LesModel,
    pub coeffs: LesCoeffs,
    pub delta: DeltaSpec,
}

impl TurbulenceSelection {
    /// A laminar run: no closure, `nu_t = 0`.
    pub fn laminar() -> Self {
        Self {
            model: RasModel::Laminar,
            active: false,
            les: None,
            des: None,
            transition: None,
        }
    }

    /// Every `0/` file this selection needs - SPEC-LIT §58.1.
    #[must_use]
    pub fn transported_fields(&self) -> &'static [&'static str] {
        self.model.transported_fields()
    }

    /// The run banner's model line, branch included.
    #[must_use]
    pub fn describe(&self) -> String {
        match (&self.des, &self.transition) {
            (Some(h), _) => h.describe(),
            (None, Some(t)) => t.describe(),
            (None, None) => self.model.name().to_string(),
        }
    }
}

/// Read `constant/momentumTransport` and say which model runs.
///
/// The three settings, in the order they override each other:
///
/// 1. `simulationType` - `laminar` switches the closure off outright;
///    `RAS`/`RASModel` selects the RANS branch. `LES` and `DES` are
///    recognised and not implemented, so they are an error.
/// 2. `RAS { model <name>; }` (or the older `RASModel`) names the model.
/// 3. `RAS { turbulence off; }` runs the case with the model's fields present
///    but frozen - which is how a case is restarted from a converged
///    turbulent field without letting the closure move.
pub fn select_turbulence_model(c: &CaseControls) -> Result<TurbulenceSelection> {
    let d = &c.momentum_transport;

    // ---- simulationType ---------------------------------------------------
    let sim = d
        .get("simulationType")
        .unwrap_or("RAS")
        .split_whitespace()
        .next()
        .unwrap_or("RAS")
        .to_string();

    match sim.as_str() {
        "laminar" => return Ok(TurbulenceSelection::laminar()),
        "RAS" | "RASModel" => {
            // SPEC-LIT §30.2: a `LES { model ...; }` block left in a case
            // that runs `simulationType RAS;` is not read, so it cannot be
            // what the RAS answer used - which means it is either dead
            // leftover the case should drop, or the setting the run author
            // actually wanted and typed in the wrong place. Both are worth
            // stopping for rather than silently taking the RAS branch.
            if let Some(les_name) = les_block_model_name(d) {
                unsupported::<()>(
                    "momentumTransport (simulationType RAS with an LES model \
                     block also present)",
                    &les_name,
                    &[],
                    "RAS (the LES block is ignored)",
                    (),
                )?;
            }
        }
        "LES" => {
            // The mirror image: `RAS { model ...; }` left beside
            // `simulationType LES;` is never read either.
            if let Some(ras_name) = ras_block_model_name(d) {
                unsupported::<()>(
                    "momentumTransport (simulationType LES with a RAS model \
                     block also present)",
                    &ras_name,
                    &[],
                    "LES (the RAS block is ignored)",
                    (),
                )?;
            }
            // A hybrid written under `simulationType LES;` is the OpenFOAM
            // spelling and is honoured here - SPEC-LIT §58.1. The name alone
            // says which branch, so there is nothing for it to disagree with.
            if let Some(n) = les_block_model_name(d) {
                if HYBRID_REGISTRY.iter().any(|(m, _, _)| *m == n) {
                    return select_hybrid(c, d, None);
                }
            }
            return select_les(d);
        }
        // A detached-eddy hybrid is a RANS model and an LES model with a
        // switch between them, and the switch is the model - which is now
        // `models::des`, and is what this arm reaches (SPEC-LIT §57, §58.2).
        //
        // The branch `simulationType` names must AGREE with the one the model
        // name carries: `simulationType DDES;` beside
        // `model SpalartAllmarasIDDES;` is a case that says two things, and
        // taking either would be the silent substitution §13.4 forbids.
        "DES" | "DDES" | "IDDES" => {
            return select_hybrid(c, d, Some(sim.as_str()));
        }
        other => {
            return unsupported(
                "momentumTransport/simulationType",
                other,
                &["RAS", "LES", "laminar"],
                "laminar (nu_t = 0)",
                TurbulenceSelection::laminar(),
            );
        }
    }

    // ---- the model name ---------------------------------------------------
    // An absent name is not an error: a case with no momentumTransport file at
    // all is a laminar case, and that is what `foamRun` makes of it too.
    let raw = c.model_name.trim();
    if raw.is_empty() {
        return Ok(TurbulenceSelection::laminar());
    }
    let name = raw.split_whitespace().next().unwrap_or(raw);

    let model = match REGISTRY.iter().find(|(n, _)| *n == name) {
        Some((_, m)) => *m,
        None => {
            let menu = available_models();
            let hint = if LES_MODEL_UNDER_RAS.contains(&name) {
                "an LES model; ofgpu has it, but it needs `simulationType LES;`"
                    .to_string()
            } else if HYBRID_REGISTRY.iter().any(|(n, _, _)| *n == name) {
                "a hybrid RANS-LES model; ofgpu has it, but it needs \
                 `simulationType LES;` or `simulationType DES|DDES|IDDES;` and \
                 its own block"
                    .to_string()
            } else if let Some((_, why)) =
                RECOGNISED_NOT_IMPLEMENTED.iter().find(|(n, _)| *n == name)
            {
                format!("a published model ofgpu has not got - {why}")
            } else {
                "not a model name ofgpu knows".to_string()
            };
            return unsupported(
                &format!("momentumTransport/RAS/model ({hint})"),
                name,
                &menu,
                "laminar (nu_t = 0)",
                TurbulenceSelection::laminar(),
            );
        }
    };

    if model == RasModel::Laminar {
        return Ok(TurbulenceSelection::laminar());
    }

    // ---- turbulence on/off ------------------------------------------------
    // `printCoeffs` and friends are switches too, and reading this one the
    // same way keeps `turbulence off;` from being the one entry in the file
    // that does nothing.
    let active = d.bool("RAS/turbulence", true);

    // SPEC-LIT 89.2: the transition model's own record, read only when the
    // case named it. `lm_selection` also fires every refusal 89.3 lists, so
    // a case that writes a setting kOmegaSSTLM does not read stops here
    // rather than having it thrown away.
    let transition = if model == RasModel::KOmegaSstLM {
        Some(lm_selection(c)?)
    } else {
        None
    };

    Ok(TurbulenceSelection {
        model,
        active,
        les: None,
        des: None,
        transition,
    })
}

// ==========================================================================
//  LES - SPEC-LIT §6.5 and §16
// ==========================================================================

/// Read the `LES { ... }` sub-dictionary.
///
/// ```text
/// simulationType  LES;
/// LES
/// {
///     model            WALE;
///     delta            vanDriest;
///     WALECoeffs       { Cw 0.325; }
///     vanDriestCoeffs  { delta cubeRootVol; Aplus 26; Cdelta 0.158; }
///     turbulence       on;
/// }
/// ```
///
/// Every name is looked up in a table and an unknown one is an error naming
/// the setting and the menu - SPEC-LIT §13.4, the same rule the RAS branch
/// above obeys. In particular `LES { delta cubeRootVolDelta; }` - a plausible
/// mis-spelling - is refused rather than being read as `cubeRootVol`, because
/// a filter width that is not the one the case asked for changes `nu_t`
/// everywhere and shows up as nothing but a slightly different answer.
fn select_les(d: &FoamDict) -> Result<TurbulenceSelection> {
    let raw = first_word(d.get_or("LES/model", d.get_or("LES/LESModel", "")));

    if raw.is_empty() {
        return unsupported(
            "momentumTransport/LES/model",
            "<missing>",
            &available_les_models(),
            "laminar (nu_t = 0)",
            TurbulenceSelection::laminar(),
        );
    }

    let model = match LES_REGISTRY.iter().find(|(n, _)| *n == raw) {
        Some((_, m)) => *m,
        None => {
            let hint = if LES_RECOGNISED_NOT_IMPLEMENTED.contains(&raw.as_str()) {
                "a published subgrid model ofgpu has not got"
            } else {
                "not a subgrid model name ofgpu knows"
            };
            return unsupported(
                &format!("momentumTransport/LES/model ({hint})"),
                &raw,
                &available_les_models(),
                "laminar (nu_t = 0)",
                TurbulenceSelection::laminar(),
            );
        }
    };

    // The coefficients, from `<model>Coeffs` or from the LES dict itself -
    // the same two-place lookup `io::case::model_coeff` does for RAS, so a
    // case can write either and neither is silently ignored.
    let def = LesCoeffs::default();
    let coeffs = LesCoeffs {
        cs: les_coeff(d, raw.as_str(), "Cs", def.cs),
        cw: les_coeff(d, raw.as_str(), "Cw", def.cw),
        cd: les_coeff(d, raw.as_str(), "Ck", les_coeff(d, raw.as_str(), "Cd", def.cd)),
    };

    let delta = parse_delta(d, first_word(d.get_or("LES/delta", "cubeRootVol")), 0)?;

    // `LES { turbulence off; }` freezes the closure exactly as its RAS
    // counterpart does - which for an algebraic model means `nu_t = 0` and no
    // `correct`, i.e. a DNS on whatever mesh the case has.
    let active = d.bool("LES/turbulence", true);

    Ok(TurbulenceSelection {
        model: RasModel::Les,
        active,
        les: Some(LesSelection {
            model,
            coeffs,
            delta,
        }),
        des: None,
        transition: None,
    })
}

// ==========================================================================
//  Section 57 / 58 - the hybrid RANS-LES family
// ==========================================================================

/// Read a hybrid out of `DES { ... }` (or `LES { ... }`, the OpenFOAM
/// spelling) - SPEC-LIT §57, §58.1.
///
/// `sim_branch` is the branch `simulationType` named, when it named one. The
/// model name carries a branch too, and the two must agree: a case that says
/// `simulationType DDES;` beside `model SpalartAllmarasIDDES;` has said two
/// different things, and taking either would be exactly the silent
/// substitution SPEC-LIT §13.4 forbids.
///
/// Four guards run before anything is built, and they are what keep this
/// capability honest rather than merely present (SPEC-LIT §57.10): a steady
/// run, an upwind-biased `div(phi,U)`, `cubeRootVol` as the width, and the
/// two LES width WRAPPERS are each refused by name. The 2-D refusal needs the
/// mesh and lives in [`build_coupled`].
fn select_hybrid(
    c: &CaseControls,
    d: &FoamDict,
    sim_branch: Option<&str>,
) -> Result<TurbulenceSelection> {
    // Which block: `DES { ... }` if the case wrote one, else `LES { ... }`.
    let prefix = if d.has("DES/model") || d.has("DES/DESModel") {
        "DES"
    } else {
        "LES"
    };
    let raw = first_word(d.get_or(
        &format!("{prefix}/model"),
        d.get_or(&format!("{prefix}/{prefix}Model"), ""),
    ));

    if raw.is_empty() {
        return unsupported(
            &format!("momentumTransport/{prefix}/model"),
            "<missing>",
            &available_hybrid_models(),
            "laminar (nu_t = 0)",
            TurbulenceSelection::laminar(),
        );
    }

    let Some((_, branch, background)) =
        HYBRID_REGISTRY.iter().find(|(n, _, _)| *n == raw).copied()
    else {
        let hint = if LES_REGISTRY.iter().any(|(n, _)| *n == raw) {
            "an ALGEBRAIC subgrid model, not a hybrid; it needs \
             `simulationType LES;` with no DES block"
        } else if LES_RECOGNISED_NOT_IMPLEMENTED.contains(&raw.as_str()) {
            "a published subgrid model ofgpu has not got"
        } else {
            "not a hybrid model name ofgpu knows"
        };
        return unsupported(
            &format!("momentumTransport/{prefix}/model ({hint})"),
            &raw,
            &available_hybrid_models(),
            "laminar (nu_t = 0)",
            TurbulenceSelection::laminar(),
        );
    };

    // SPEC-LIT §58.1: the two spellings of the branch must agree.
    if let Some(sim) = sim_branch {
        if sim != branch.name() {
            return unsupported_note(
                "momentumTransport/simulationType (against the model name)",
                sim,
                &[branch.name()],
                &format!(
                    "`{prefix} {{ model {raw}; }}` is the {} branch; \
                     `simulationType {sim};` names another. The case says two \
                     different things and this solver will not pick one \
                     (SPEC-LIT 58.1)",
                    branch.name()
                ),
                "laminar (nu_t = 0)",
                TurbulenceSelection::laminar(),
            );
        }
    }

    // ---- guard 1: a steady run ------------------------------------------
    // DES is unsteady by construction; a steady DES is a RANS model with a
    // corrupted length scale (SPEC-LIT §57.10).
    if c.turb.steady {
        return unsupported_note(
            "ddtSchemes/default (under a DES-family model)",
            "steadyState",
            &["Euler", "backward", "CrankNicolson <theta>"],
            "a detached-eddy hybrid is UNSTEADY by construction: its LES branch \
             represents resolved turbulence, and a steady solve has none. A \
             steady run of this model is a RANS model whose destruction term \
             has been divided by the wrong length (SPEC-LIT 57.10). Run one of \
             the RANS models instead, or write a time scheme",
            "laminar (nu_t = 0)",
            TurbulenceSelection::laminar(),
        );
    }

    // ---- guard 2: an upwind-biased convection scheme ---------------------
    let u_conv = c.schemes.div("div(phi,U)")?;
    if !hybrid_scheme_is_low_dissipation(u_conv.scheme) {
        return unsupported_note(
            "divSchemes/div(phi,U) (under a DES-family model)",
            &format!("{:?}", u_conv.scheme),
            &[
                "Gauss linear",
                "Gauss cubic",
                "Gauss linearUpwindBlended 0.75 (LUST)",
                "Gauss blended <gamma >= 0.5>",
            ],
            "an upwind-biased scheme damps the resolved content the LES branch \
             exists to produce, and the run then looks converged and plausible \
             and is wrong - the same class of silent substitution SPEC-LIT 13.4 \
             forbids, which is why this is an error and not a warning. Travin \
             et al. (2002) publish a scheme-blending function for a case that \
             genuinely wants an upwind-biased RANS region; ofgpu has NOT got it \
             (SPEC-LIT 57.10)",
            "laminar (nu_t = 0)",
            TurbulenceSelection::laminar(),
        );
    }

    // ---- guard 3 and 4: the filter width ---------------------------------
    let delta_raw = first_word(d.get_or(
        &format!("{prefix}/delta"),
        &HybridDelta::default_for(branch, background).name().to_string(),
    ));
    let Some((_, delta)) = HYBRID_DELTA_NAMES
        .iter()
        .find(|(n, _)| *n == delta_raw)
        .copied()
    else {
        let note = match delta_raw.as_str() {
            "cubeRootVol" | "cubeRootVolDelta" | "Scotti" => {
                "C_DES = 0.65 is calibrated with Delta = h_max (Shur et al. \
                 1999). On an anisotropic mesh V^(1/3) is smaller than h_max by \
                 the cell aspect ratio, so accepting this would run a DES with \
                 an uncalibrated constant - a refusal, not a preference \
                 (SPEC-LIT 57.10)"
            }
            "vanDriest" | "smooth" => {
                "both are WRAPPERS that damp or smooth a base width for a pure \
                 LES. Neither is defined for a hybrid, whose RANS branch \
                 already carries the near-wall treatment (SPEC-LIT 57.10)"
            }
            _ => {
                "a hybrid takes h_max, or one of IDDES's two published widths - \
                 SPEC-LIT (57.17) and (57.18)"
            }
        };
        return unsupported_note(
            &format!("momentumTransport/{prefix}/delta"),
            &delta_raw,
            &available_hybrid_deltas(),
            note,
            "laminar (nu_t = 0)",
            TurbulenceSelection::laminar(),
        );
    };

    let coeffs = des_coeffs(d, prefix, background)?;
    let sa = sa_coeffs_from(d, &format!("{prefix}/"))?;
    let active = d.bool(&format!("{prefix}/turbulence"), true);

    Ok(TurbulenceSelection {
        model: match background {
            HybridBackground::Sa => RasModel::HybridSa,
            HybridBackground::Sst => RasModel::HybridSst,
        },
        active,
        les: None,
        transition: None,
        des: Some(HybridSelection {
            branch,
            background,
            delta,
            coeffs,
            sa,
        }),
    })
}

/// SPEC-LIT §57.10's guard 3: a 2-D mesh.
///
/// The LES branch of a hybrid is a three-dimensional turbulence model. A 2-D
/// DES resolves nothing at all - it produces a plausible converged answer with
/// no resolved content, which is exactly the failure the whole guard set
/// exists to stop. `PatchKind::Empty` is what a 2-D mesh has and a 3-D one
/// does not.
pub fn refuse_two_dimensional_hybrid(
    hm: &HostMesh,
    sel: &HybridSelection,
) -> Result<()> {
    let empty = crate::mesh::PatchKind::Empty as Label;
    if !hm.b_kind.iter().any(|&k| k == empty) {
        return Ok(());
    }
    unsupported_note::<()>(
        "the mesh (under a DES-family model)",
        "a 2-D mesh: it has `empty` patches",
        &["a 3-D mesh"],
        &format!(
            "{} is a hybrid RANS-LES model, and its LES branch is a \
             three-dimensional turbulence model: in two dimensions there is no \
             vortex stretching and nothing for it to resolve. A 2-D run of this \
             model converges, plots and means nothing (SPEC-LIT 57.10). Run a \
             RANS model on this mesh, or extrude it",
            sel.model_name()
        ),
        "nothing - a 2-D hybrid is refused",
        (),
    )
}

/// SPEC-LIT §57.10's guard 2: is this `div(phi,U)` low-dissipation enough for
/// the LES branch to have anything to resolve?
///
/// Central and cubic are the unbounded second- and fourth-order schemes;
/// `linearUpwindBlended` is the usual LES choice (LUST is that blend at
/// `0.75`); a plain `blended` counts once the central half is at least half.
/// Everything else is upwind-biased.
fn hybrid_scheme_is_low_dissipation(sch: crate::fv::DivScheme) -> bool {
    use crate::fv::DivScheme as D;
    match sch {
        D::Central | D::Cubic | D::LinearUpwindBlended(_) => true,
        D::Blended(g) => g >= 0.5,
        _ => false,
    }
}

/// Every key a hybrid reads out of its own block, per background.
const DES_KEYS_SA: &[&str] = &["CDES", "Cdt1", "Cdt2", "ct", "cl", "Cw", "kappa", "delta"];
const DES_KEYS_SST: &[&str] =
    &["CDES1", "CDES2", "Cdt1", "Cdt2", "ct", "cl", "Cw", "kappa", "delta"];

/// SPEC-LIT §57.5's per-background refusals, read from the hybrid's own
/// block.
fn des_coeffs(
    d: &FoamDict,
    prefix: &str,
    background: HybridBackground,
) -> Result<DesCoeffs> {
    let def = DesCoeffs::for_background(background);

    let inert: &[(&str, &str)] = match background {
        HybridBackground::Sa => &[
            (
                "CDES1",
                "CDES1 and CDES2 are the SST background's, where C_DES is \
                 blended by F1 (SPEC-LIT 57.5). The SA background has ONE \
                 C_DES; write `CDES 0.65`",
            ),
            (
                "CDES2",
                "as CDES1: the SA background has one C_DES, not two blended by \
                 F1. Write `CDES 0.65`",
            ),
        ],
        HybridBackground::Sst => &[(
            "CDES",
            "the SST background's C_DES is C_DES1 F1 + C_DES2 (1 - F1), a FIELD \
             (SPEC-LIT 57.5 and arXiv:2603.08875 eq. 15) - a single value \
             cannot express it. Write `CDES1 0.78; CDES2 0.61;`",
        )],
    };
    for (key, why) in inert {
        let path = format!("{prefix}/{key}");
        if d.has(&path) {
            let read = match background {
                HybridBackground::Sa => DES_KEYS_SA,
                HybridBackground::Sst => DES_KEYS_SST,
            };
            unsupported_note::<()>(
                &format!("momentumTransport/{prefix}/{key} (on the {} background)", background.name()),
                d.get(&path).unwrap_or("").trim(),
                read,
                why,
                "nothing - the entry is not read by this background",
                (),
            )?;
        }
    }

    let g = |k: &str, fallback: Scalar| d.scalar(&format!("{prefix}/{k}"), fallback);
    let c = DesCoeffs {
        cdes: g("CDES", def.cdes),
        cdes1: g("CDES1", def.cdes1),
        cdes2: g("CDES2", def.cdes2),
        cdt1: g("Cdt1", def.cdt1),
        cdt2: g("Cdt2", def.cdt2),
        ct: g("ct", def.ct),
        cl: g("cl", def.cl),
        cw: g("Cw", def.cw),
        kappa: g("kappa", def.kappa),
    };
    c.check()?;
    Ok(c)
}

// ==========================================================================
//  Section 56 - Spalart-Allmaras's coefficients, and what it does NOT read
// ==========================================================================

/// Every key `SpalartAllmaras` reads out of `RAS { ... }`.
const SA_KEYS: &[&str] = &[
    "variant", "Cb1", "Cb2", "Cv1", "Cv2", "Cv3", "Cw2", "Cw3", "Ct3", "Ct4", "Cn1",
    "sigmaNut", "kappa", "rlim", "nutMaxCoeff",
];

/// Keys a case might plausibly carry from a k-epsilon or k-omega setup that
/// Spalart-Allmaras does not read - SPEC-LIT §56.8.
///
/// `Cw1` is the interesting one and it is refused for a reason no other entry
/// in this file has: it is DERIVED, and the derivation IS the log layer.
const SA_INERT: &[(&str, &str)] = &[
    (
        "Cw1",
        "c_w1 = Cb1/kappa^2 + (1 + Cb2)/sigmaNut is DERIVED, not read (SPEC-LIT \
         (56.6)). That identity is exactly what makes the log layer an exact \
         solution of the model (SPEC-LIT 56.4), so a case that could set c_w1 \
         independently could ask for a Spalart-Allmaras with no log layer. \
         Change Cb1, Cb2, kappa or sigmaNut and c_w1 moves with them",
    ),
    (
        "Cmu",
        "SpalartAllmaras has no C_mu: nu_t = nu~ f_v1, not C_mu k^2/eps \
         (SPEC-LIT (56.1)). `kEpsilon`, `realizableKE` and `RNGkEpsilon` have \
         one",
    ),
    (
        "C1",
        "SpalartAllmaras has no epsilon equation for C_1 to appear in \
         (SPEC-LIT 56.1)",
    ),
    (
        "C2",
        "SpalartAllmaras has no epsilon equation for C_2 to appear in \
         (SPEC-LIT 56.1)",
    ),
    (
        "C3",
        "the Favre dilatation coefficient belongs to the epsilon equation \
         SpalartAllmaras has not got. There is also no buoyancy term here: a \
         case with gravity naming this model is refused by name (SPEC-LIT 56.8)",
    ),
    (
        "sigmak",
        "SpalartAllmaras transports nu~, not k. Its own diffusivity constant is \
         `sigmaNut` (2/3), which multiplies (nu + nu~ f_n) (SPEC-LIT (56.2))",
    ),
    (
        "sigmaEps",
        "SpalartAllmaras has no epsilon equation. Its diffusivity constant is \
         `sigmaNut` (SPEC-LIT (56.2))",
    ),
    (
        "alphak",
        "alphak and alphaEps are RNGkEpsilon's inverse Prandtl numbers \
         (SPEC-LIT 41.2). SpalartAllmaras's is `sigmaNut`",
    ),
    (
        "alphaEps",
        "as alphak: RNGkEpsilon's. SpalartAllmaras's is `sigmaNut`",
    ),
    (
        "betaStar",
        "betaStar belongs to the k-omega family (SPEC-LIT 6.2, 6.3). \
         SpalartAllmaras has no omega equation",
    ),
    (
        "A0",
        "A0 is realizableKE's variable-C_mu constant (SPEC-LIT 40.3). \
         SpalartAllmaras has no C_mu",
    ),
];

/// SPEC-LIT §56.8, read from the case with every entry §56 does not use
/// refused by name first.
pub fn sa_coeffs(c: &CaseControls) -> Result<SaCoeffs> {
    refuse_inert_coefficients(c, "SpalartAllmaras", SA_KEYS, SA_INERT)?;
    sa_coeffs_from(&c.momentum_transport, "RAS/")
}

/// [`sa_coeffs`]'s dictionary half, so a hybrid can read the same constants
/// out of `DES { ... }` and there is ONE transcription of them.
fn sa_coeffs_from(d: &FoamDict, prefix: &str) -> Result<SaCoeffs> {
    let def = SaCoeffs::default();
    let g = |k: &str, fallback: Scalar| d.scalar(&format!("{prefix}{k}"), fallback);

    let raw = first_word(d.get_or(&format!("{prefix}variant"), def.variant.name()));
    let Some(variant) = SaVariant::parse(&raw) else {
        return unsupported(
            &format!("momentumTransport/{}variant", prefix.trim_end_matches('/')),
            &raw,
            &SaVariant::menu(),
            "noft2",
            def,
        );
    };

    let c = SaCoeffs {
        variant,
        cb1: g("Cb1", def.cb1),
        cb2: g("Cb2", def.cb2),
        cv1: g("Cv1", def.cv1),
        cv2: g("Cv2", def.cv2),
        cv3: g("Cv3", def.cv3),
        cw2: g("Cw2", def.cw2),
        cw3: g("Cw3", def.cw3),
        ct3: g("Ct3", def.ct3),
        ct4: g("Ct4", def.ct4),
        cn1: g("Cn1", def.cn1),
        sigma: g("sigmaNut", def.sigma),
        kappa: g("kappa", def.kappa),
        rlim: g("rlim", def.rlim),
    };
    c.check()?;
    Ok(c)
}

/// SPEC-LIT §56.8: `SpalartAllmaras` has no buoyancy production, so a case
/// that names gravity AND runs a coupled driver is refused by name rather
/// than run with `G_b` silently at zero.
///
/// §17's `G_b` enters a `k` equation and there is none here. Spalart &
/// Allmaras specify no buoyant extension and this solver will not invent one
/// - the same refusal §40.5 makes for `realizableKE`, one model further.
pub fn refuse_sa_buoyancy(c: &CaseControls) -> Result<()> {
    if !c.buoyancy.is_active() {
        return Ok(());
    }
    unsupported_note::<()>(
        "momentumTransport/RAS/model (`SpalartAllmaras` in a case with gravity)",
        "SpalartAllmaras",
        &["kEpsilon", "LaunderSharmaKE", "kOmega", "kOmegaSST", "RNGkEpsilon"],
        "SPEC-LIT 56.8: section 17's buoyancy production G_b enters a k \
         equation, and SpalartAllmaras has none - it transports nu~, which is \
         not an energy. Spalart & Allmaras specify no buoyant extension and \
         this solver will not invent one",
        "nothing - a buoyant SpalartAllmaras run is refused",
        (),
    )
}

// ==========================================================================
//  §89.2  What a transition case says
// ==========================================================================

/// Every `RAS { ... }` key `kOmegaSSTLM` READS - SPEC-LIT §89.2.
///
/// The SST coefficients are here too, because the transition model runs on
/// the SST background and every one of them still reaches the two equations
/// SST owns.
const LM_KEYS: &[&str] = &[
    "model",
    "turbulence",
    "printCoeffs",
    // §6.3's own, unchanged
    "sigmaK1", "sigmaOmega1", "beta1", "gamma1",
    "sigmaK2", "sigmaOmega2", "beta2", "gamma2",
    "betaStar", "a1", "b1", "c1", "nutMaxCoeff",
    // §88's
    "ca1", "ca2", "ce1", "ce2", "cThetat", "s1", "sigmaf", "sigmaThetat",
    // §88's, and OURS
    "nReThetaSweeps", "gammaMin", "gammaMax", "ReThetatMin",
];

/// Keys a case might plausibly carry that `kOmegaSSTLM` does NOT read -
/// SPEC-LIT §89.3.
const LM_INERT: &[(&str, &str)] = &[
    (
        "Cmu",
        "kOmegaSSTLM runs on the k-omega SST background, whose corresponding \
         constant is `betaStar` (SPEC-LIT 6.3). `kEpsilon`, `realizableKE` \
         and `RNGkEpsilon` read Cmu",
    ),
    (
        "C1",
        "there is no epsilon equation here for C_1 to appear in. The \
         transition model's own production constants are ca1 and ce1 \
         (SPEC-LIT (88.5)), and SST's cross-diffusion limiter is c1 - a \
         DIFFERENT constant with a lower-case spelling, which is exactly why \
         C1 is refused rather than being read as it",
    ),
    (
        "C2",
        "there is no epsilon equation here for C_2 to appear in. The \
         transition model's own destruction constants are ca2 and ce2 \
         (SPEC-LIT (88.6))",
    ),
    (
        "C3",
        "the Favre dilatation coefficient belongs to an epsilon equation this \
         model has not got. SST's own dilatation term is unscaled by the \
         intermittency and SPEC-LIT 88.6 says why: Langtry & Menter write \
         nothing about it, and inventing a factor is what 13.4 forbids",
    ),
    (
        "sigmak",
        "SST blends TWO of these and they are `sigmaK1` and `sigmaK2` \
         (SPEC-LIT 6.3). A single sigmak would be read into neither",
    ),
    (
        "sigmaEps",
        "there is no epsilon equation here. SST's omega-equation diffusivities \
         are `sigmaOmega1` and `sigmaOmega2` (SPEC-LIT 6.3)",
    ),
    (
        "FlengthCoeff",
        "F_length is a published CORRELATION in Re_theta~, four pieces of \
         polynomial with a viscous-sublayer blend (SPEC-LIT (88.4)), not a \
         constant. There is nothing for a coefficient to multiply that would \
         still be Langtry & Menter's F_length",
    ),
    (
        "ReThetacCoeff",
        "Re_thetac is a published CORRELATION in Re_theta~ (SPEC-LIT (88.3)), \
         not a constant. Scaling it would move the transition location by an \
         amount no published calibration supports",
    ),
    (
        "Tu",
        "the free-stream turbulence intensity is not a model constant: it is \
         computed per cell from the local k and |U| (SPEC-LIT (88.9)), which \
         is the whole point of a LOCAL correlation-based transition model. \
         What a case DOES set from a free-stream Tu is the inlet value of \
         ReThetat, and `models::transition::re_thetat_inlet` computes it \
         (SPEC-LIT 89.2)",
    ),
];

/// Read `constant/momentumTransport` and `system/` for `kOmegaSSTLM` -
/// SPEC-LIT §88.9, §89.2.
pub fn lm_selection(c: &CaseControls) -> Result<LmSelection> {
    refuse_inert_coefficients(c, "kOmegaSSTLM", LM_KEYS, LM_INERT)?;
    Ok(LmSelection {
        coeffs: lm_coeffs(c)?,
        controls: lm_controls(c)?,
    })
}

/// SPEC-LIT §88.9: a transitional case with gravity is refused by name.
///
/// §17's `G_b` reaches a `k` equation model-independently, and this model has
/// one - so unlike `realizableKE` (§40.5) and `SpalartAllmaras` (§56.8) the
/// term is not *missing*. What is missing is a published answer to the one
/// question the coupling asks: **does `gamma_eff` multiply `G_b` as it
/// multiplies `P_k`?** Langtry & Menter write nothing about buoyancy, both
/// answers are defensible, and picking one silently is precisely the
/// substitution §13.4 exists to stop - the more so because a laminar buoyant
/// layer with an unscaled `G_b` would generate turbulence the intermittency
/// says is not there yet.
pub fn refuse_lm_buoyancy(c: &CaseControls) -> Result<()> {
    if !c.buoyancy.is_active() {
        return Ok(());
    }
    unsupported_note::<()>(
        "momentumTransport/RAS/model (`kOmegaSSTLM` in a case with gravity)",
        "kOmegaSSTLM",
        &["kOmegaSST", "kEpsilon", "LaunderSharmaKE", "kOmega", "RNGkEpsilon"],
        "SPEC-LIT 88.9: section 17's buoyancy production G_b enters the k \
         equation, and (88.13) scales that equation's production by the \
         effective intermittency. Whether G_b is scaled with it is a question \
         Langtry & Menter do not answer - they publish no buoyant extension - \
         and both answers change where a buoyant layer transitions. This \
         solver will not invent one",
        "nothing - a buoyant kOmegaSSTLM run is refused",
        (),
    )
}

/// §88's own constants, and the three that are ours.
pub fn lm_coeffs(c: &CaseControls) -> Result<LmCoeffs> {
    let d = &c.momentum_transport;
    let def = LmCoeffs::default();
    let g = |k: &str, fallback: Scalar| d.scalar(&format!("RAS/{k}"), fallback);

    let coeffs = LmCoeffs {
        ca1: g("ca1", def.ca1),
        ca2: g("ca2", def.ca2),
        ce1: g("ce1", def.ce1),
        ce2: g("ce2", def.ce2),
        ctt: g("cThetat", def.ctt),
        s1: g("s1", def.s1),
        sigma_f: g("sigmaf", def.sigma_f),
        sigma_tt: g("sigmaThetat", def.sigma_tt),
        n_sweeps: d.label("RAS/nReThetaSweeps", def.n_sweeps as crate::Label).max(0) as usize,
        gamma_min: g("gammaMin", def.gamma_min),
        gamma_max: g("gammaMax", def.gamma_max),
        re_thetat_min: g("ReThetatMin", def.re_thetat_min),
    };
    coeffs.check()?;
    Ok(coeffs)
}

/// The two new equations' `system/` settings - SPEC-LIT §89.2.
///
/// **Each entry is read for ITSELF.** `TurbulenceControls::epsilon_solver` is
/// documented as "also used for omega - the two never coexist"; three
/// dissipation-like variables DO coexist under this model, and reaching for
/// that slot a third time would make `solvers/gamma`,
/// `relaxationFactors/equations/gamma` and `divSchemes/div(phi,gamma)` inert,
/// which is the exact §13.4.1 failure `dissipation_from_model` was written to
/// fix for `nuTilda`. Where the case writes no entry of its own the fallback
/// is `k`'s, the closest bounded scalar in the run.
pub fn lm_controls(c: &CaseControls) -> Result<LmControls> {
    let mut ctrl = LmControls {
        gamma_solver: c.turb.k_solver,
        gamma_relax: c.turb.k_relax,
        gamma_conv: c.turb.k_conv(),
        re_thetat_solver: c.turb.k_solver,
        re_thetat_relax: c.turb.k_relax,
        re_thetat_conv: c.turb.k_conv(),
    };

    crate::io::case::read_solver_controls(&mut ctrl.gamma_solver, &c.fv_solution, "gamma")?;
    crate::io::case::read_solver_controls(
        &mut ctrl.re_thetat_solver,
        &c.fv_solution,
        "ReThetat",
    )?;

    ctrl.gamma_relax =
        crate::io::case::relaxation_factor(&c.fv_solution, "gamma", ctrl.gamma_relax)?;
    ctrl.re_thetat_relax =
        crate::io::case::relaxation_factor(&c.fv_solution, "ReThetat", ctrl.re_thetat_relax)?;

    ctrl.gamma_conv = crate::io::case::div_entry(c, "div(phi,gamma)")?;
    ctrl.re_thetat_conv = crate::io::case::div_entry(c, "div(phi,ReThetat)")?;

    Ok(ctrl)
}

/// `LES/<model>Coeffs/<name>`, then `LES/<name>`.
fn les_coeff(d: &FoamDict, model: &str, name: &str, fallback: Scalar) -> Scalar {
    let a = format!("LES/{model}Coeffs/{name}");
    if d.has(&a) {
        return d.scalar(&a, fallback);
    }
    let b = format!("LES/{name}");
    if d.has(&b) {
        return d.scalar(&b, fallback);
    }
    fallback
}

/// The filter width named by `name`, recursively - SPEC-LIT §16.
///
/// `vanDriest` and `smooth` wrap a base width named in their own `Coeffs`
/// sub-dictionary, so the grammar is recursive and the recursion has to be
/// bounded: `depth` stops a case whose `vanDriestCoeffs { delta vanDriest; }`
/// refers to itself from recursing until the stack runs out. Three levels is
/// one more than any combination §16 describes needs.
fn parse_delta(d: &FoamDict, name: String, depth: usize) -> Result<DeltaSpec> {
    if depth > 3 {
        return unsupported(
            "momentumTransport/LES/delta",
            &name,
            &available_deltas(),
            "cubeRootVol",
            DeltaSpec::default(),
        );
    }

    let base_of = |key: &str, fallback: &str| -> String {
        first_word(d.get_or(key, fallback))
    };

    let spec = match name.as_str() {
        "cubeRootVol" => DeltaSpec {
            base: BaseDelta::CubeRootVol,
            delta_coeff: d.scalar("LES/cubeRootVolCoeffs/deltaCoeff", 1.0),
            ..Default::default()
        },
        "maxDeltaxyz" => DeltaSpec {
            base: BaseDelta::MaxEdge,
            delta_coeff: d.scalar("LES/maxDeltaxyzCoeffs/deltaCoeff", 1.0),
            ..Default::default()
        },
        "Scotti" => DeltaSpec {
            base: BaseDelta::CubeRootVol,
            anisotropy: true,
            delta_coeff: d.scalar("LES/ScottiCoeffs/deltaCoeff", 1.0),
            ..Default::default()
        },
        "vanDriest" => {
            let inner = parse_delta(d, base_of("LES/vanDriestCoeffs/delta", "cubeRootVol"), depth + 1)?;
            DeltaSpec {
                van_driest: true,
                kappa: d.scalar("LES/vanDriestCoeffs/kappa", inner.kappa),
                a_plus: d.scalar("LES/vanDriestCoeffs/Aplus", inner.a_plus),
                c_delta: d.scalar("LES/vanDriestCoeffs/Cdelta", inner.c_delta),
                ..inner
            }
        }
        "smooth" => {
            let inner = parse_delta(d, base_of("LES/smoothCoeffs/delta", "cubeRootVol"), depth + 1)?;
            let def = SmoothSpec::default();
            DeltaSpec {
                smooth: Some(SmoothSpec {
                    max_ratio: d.scalar("LES/smoothCoeffs/maxDeltaRatio", def.max_ratio),
                    sweeps: d.scalar("LES/smoothCoeffs/sweeps", def.sweeps as Scalar).max(0.0)
                        as usize,
                }),
                ..inner
            }
        }
        other => {
            let hint = if DELTA_HYBRID_ONLY.contains(&other) {
                "a filter width ofgpu HAS, but only inside a hybrid: SPEC-LIT \
                 (57.17) reads the wall distance and the wall-normal grid step \
                 and is defined only inside IDDES's own length-scale blend. \
                 Write it under `DES { delta ...; }` with an IDDES model"
            } else if DELTA_RECOGNISED_NOT_IMPLEMENTED.contains(&other) {
                "a published filter width ofgpu has not got"
            } else {
                "not a filter width ofgpu knows"
            };
            return unsupported(
                &format!("momentumTransport/LES/delta ({hint})"),
                other,
                &available_deltas(),
                "cubeRootVol",
                DeltaSpec::default(),
            );
        }
    };

    Ok(spec)
}

/// The first whitespace-separated token, which is how every entry in this
/// file is read: `delta cubeRootVol;` and `delta cubeRootVol  ;` are the same
/// setting, and a trailing comment is not part of the name.
fn first_word(s: &str) -> String {
    s.trim().split_whitespace().next().unwrap_or("").to_string()
}

/// `RAS/model` (or `RAS/RASModel`), non-empty - SPEC-LIT §30.2's conflict
/// check. `None` means there is nothing named there, which is different from
/// "an empty `RAS { turbulence off; }` block": the latter selects nothing
/// either, so it is not the ambiguity this check exists for.
fn ras_block_model_name(d: &FoamDict) -> Option<String> {
    let n = first_word(d.get_or("RAS/model", d.get_or("RAS/RASModel", "")));
    (!n.is_empty()).then_some(n)
}

/// [`ras_block_model_name`]'s mirror for `LES/model` / `LES/LESModel`.
fn les_block_model_name(d: &FoamDict) -> Option<String> {
    let n = first_word(d.get_or("LES/model", d.get_or("LES/LESModel", "")));
    (!n.is_empty()).then_some(n)
}

// ==========================================================================
//  §30.2 - the coupled solvers (buoyant, fire)
// ==========================================================================

/// Read `RAS { C3Buoyancy ...; }` into the [`BuoyancySettings`] every
/// buoyancy-capable model shares - the same reading `src/bin/buoyant.rs` did
/// inline before this registry grew a constructor, moved here so the two
/// coupled drivers do not each carry their own copy of it.
///
/// `None` when the case has no gravity: a coupled driver's model then never
/// switches its `G_b` machinery on, which is the isothermal/zero-`g` case's
/// correct behaviour (`ThermalCtx`'s doc explains why `None` is the signal
/// rather than a zeroed [`BuoyancySettings`]).
pub fn buoyancy_settings(c: &CaseControls) -> Option<BuoyancySettings> {
    if !c.buoyancy.is_active() {
        return None;
    }
    let d = BuoyancySettings::default();
    Some(BuoyancySettings {
        c3: match model_coeff(c, "C3Buoyancy", Scalar::NAN) {
            v if v.is_nan() => C3Mode::Henkes,
            v => C3Mode::Constant(v),
        },
        ..d
    })
}

// ==========================================================================
//  §40 / §41 - the coefficients of the two k-epsilon variants, and the
//  entries each of them does NOT read
// ==========================================================================

/// Every key `realizableKE` reads out of `RAS { ... }`.
///
/// The list is here rather than inline because it is what the refusal below
/// prints, and a menu that has drifted from the code is worse than no menu -
/// the same argument [`REGISTRY`]'s own doc makes.
const REALIZABLE_KE_KEYS: &[&str] =
    &["A0", "C2", "sigmak", "sigmaEps", "Cmu", "kappa", "E", "nutMaxCoeff"];

/// Keys a case might plausibly write under `realizableKE` that the model does
/// not read, with what to write instead.
///
/// SPEC-LIT §40.6. `C_1` is (40.5) - `max(0.43, eta/(eta+5))`, computed per
/// cell from the local strain - and there is no dilatation term in (40.3) for
/// `C_3` to multiply, so both would be read into a struct field and thrown
/// away. That is the sixth instance of the failure this project has now found
/// five times, and the fix is the same one: refuse by name.
const REALIZABLE_KE_INERT: &[(&str, &str)] = &[
    (
        "C1",
        "realizableKE computes C_1 = max(0.43, eta/(eta+5)) per cell from the \
         local strain (SPEC-LIT 40.5); it is not a constant. Set A0 to move \
         C_mu, or run `kEpsilon`, whose C_1 IS a constant",
    ),
    (
        "C3",
        "realizableKE's epsilon production is C_1 S eps, which is not \
         proportional to G, so there is no Favre dilatation term for C_3 to \
         multiply (SPEC-LIT 40.5). `kEpsilon` and `RNGkEpsilon` both have one",
    ),
];

/// Every key `RNGkEpsilon` reads.
const RNG_KE_KEYS: &[&str] = &[
    "Cmu", "C1", "C2", "C3", "alphak", "alphaEps", "eta0", "beta", "kappa", "E", "nutMaxCoeff",
];

/// SPEC-LIT §41.4: this model's diffusivity is `alpha (nu + nu_t)`, not
/// `nu + nu_t/sigma`, so a case that writes `sigmaEps 1.3` here has written a
/// number nothing reads.
const RNG_KE_INERT: &[(&str, &str)] = &[
    (
        "sigmak",
        "RNGkEpsilon diffuses k with alphak (nu + nu_t), where the inverse \
         Prandtl number multiplies the EFFECTIVE viscosity (SPEC-LIT 41.2). \
         Write `alphak 1.39` instead. The two are NOT the same setting with \
         two names: alphak (nu + nu_t) against nu + nu_t/sigmak differ by \
         (alphak - 1) nu on every face, which is nothing in the free stream \
         and is the whole diffusivity in the first cell off a wall",
    ),
    (
        "sigmaEps",
        "RNGkEpsilon diffuses epsilon with alphaEps (nu + nu_t) (SPEC-LIT \
         41.2). Write `alphaEps 1.39` instead - numerically 1/0.71942, but \
         multiplying nu + nu_t rather than nu_t alone, which is a different \
         diffusivity near a wall and the same one far from it",
    ),
    (
        "A0",
        "A0 is realizableKE's (SPEC-LIT 40.3). RNGkEpsilon's C_mu is the \
         constant 0.0845",
    ),
];

/// Refuse, by name, every `RAS { ... }` entry the named model does not read -
/// SPEC-LIT §13.4.
///
/// `model_coeff` is a silent lookup: it returns the fallback for a key that is
/// not there and says nothing about a key that IS there and is never asked
/// for. That asymmetry is how five settings in this project's history came to
/// be written by a generator, parsed by a reader and consulted by nobody. The
/// two new models close it for themselves by listing what they read and
/// refusing the rest.
///
/// Only keys in `inert` are refused, not every unrecognised key: a case
/// dictionary legitimately carries `printCoeffs`, `turbulence`,
/// `wallTreatment`, `roughness` and whatever else the rest of the reader
/// consumes, and refusing those would refuse every real case. What is refused
/// is the specific set of coefficient names that BELONG to a sibling model and
/// would look, to a reader of the case file, as though they were in force.
fn refuse_inert_coefficients(
    c: &CaseControls,
    model: &str,
    read: &[&str],
    inert: &[(&str, &str)],
) -> Result<()> {
    for (key, why) in inert {
        let direct = format!("RAS/{key}");
        let in_coeffs = format!("RAS/{model}Coeffs/{key}");
        let d = &c.momentum_transport;
        if d.has(&direct) || d.has(&in_coeffs) {
            let value = d
                .get(&direct)
                .or_else(|| d.get(&in_coeffs))
                .unwrap_or("")
                .trim()
                .to_string();
            unsupported_note::<()>(
                &format!("momentumTransport/RAS/{key} (under `model {model}`)"),
                &value,
                read,
                why,
                "nothing - the entry is not read by this model",
                (),
            )?;
        }
    }
    Ok(())
}

/// SPEC-LIT §40.6, read from the case with every entry §40 does not use
/// refused by name first.
pub fn realizable_ke_coeffs(c: &CaseControls) -> Result<RealizableKeCoeffs> {
    refuse_inert_coefficients(c, "realizableKE", REALIZABLE_KE_KEYS, REALIZABLE_KE_INERT)?;
    let d = RealizableKeCoeffs::default();
    Ok(RealizableKeCoeffs {
        a0: model_coeff(c, "A0", d.a0),
        c2: model_coeff(c, "C2", d.c2),
        sigmak: model_coeff(c, "sigmak", d.sigmak),
        sigma_eps: model_coeff(c, "sigmaEps", d.sigma_eps),
    })
}

/// SPEC-LIT §41.4, same discipline.
pub fn rng_ke_coeffs(c: &CaseControls) -> Result<RngKeCoeffs> {
    refuse_inert_coefficients(c, "RNGkEpsilon", RNG_KE_KEYS, RNG_KE_INERT)?;
    let d = RngKeCoeffs::default();
    Ok(RngKeCoeffs {
        cmu: model_coeff(c, "Cmu", d.cmu),
        c1: model_coeff(c, "C1", d.c1),
        c2: model_coeff(c, "C2", d.c2),
        alpha_k: model_coeff(c, "alphak", d.alpha_k),
        alpha_eps: model_coeff(c, "alphaEps", d.alpha_eps),
        eta0: model_coeff(c, "eta0", d.eta0),
        beta: model_coeff(c, "beta", d.beta),
        c3: model_coeff(c, "C3", d.c3),
    })
}

/// SPEC-LIT §40.5: `realizableKE` has no buoyancy production, so a case that
/// names gravity AND runs a coupled driver is refused by name rather than run
/// with `G_b` silently at zero.
///
/// Shih et al. specify no buoyant extension. The one every code uses,
/// `C_1 (eps/k) C_3 G_b`, presupposes that the `epsilon` production is
/// proportional to `G`; §40's is `C_1 S eps` and is not. Writing it anyway
/// would be inventing a model and attributing it to a paper, which is what
/// §13.4 and §0 between them forbid.
pub fn refuse_realizable_ke_buoyancy(c: &CaseControls) -> Result<()> {
    if !c.buoyancy.is_active() {
        return Ok(());
    }
    unsupported_note::<()>(
        "momentumTransport/RAS/model (`realizableKE` in a case with gravity)",
        "realizableKE",
        &["kEpsilon", "LaunderSharmaKE", "kOmega", "kOmegaSST", "RNGkEpsilon"],
        "SPEC-LIT 40.5: realizableKE's epsilon production is C_1 S eps, which \
         is not proportional to G, so section 17's buoyancy term \
         C_1 (eps/k) C_3 G_b has nothing to attach to. Shih et al. (NASA \
         TM-106721) specify no buoyant extension and this solver will not \
         invent one. RNGkEpsilon keeps section 6.1's production form exactly \
         and DOES carry G_b (SPEC-LIT 41.5)",
        "nothing - a buoyant realizableKE run is refused",
        (),
    )
}

/// Build the turbulence closure a coupled solver (`ofgpu-buoyant`,
/// `ofgpu-lowmach`) drives, from the case's own `constant/momentumTransport` -
/// SPEC-LIT §30.2.
///
/// This is the fix for the failure the module doc of
/// [`crate::models::coupled`] describes: before this function existed, both
/// coupled drivers built `KEpsilon` directly and never consulted the case at
/// all, so `RAS { model kOmegaSST; }` and `simulationType LES;` were both
/// silently read as k-epsilon. Every branch below goes through
/// [`select_turbulence_model`] first, exactly as `ofgpu-k-epsilon` and
/// `ofgpu-k-omega` already do, so an unknown or unimplemented name errors
/// here exactly as it does there.
///
/// `wall_faces` and `roughness` are NOT rebuilt here: SPEC-LIT §15.5 requires
/// them to come from the DISSIPATION field's own patch types
/// (`epsilon`'s for k-epsilon, `omega`'s for k-omega/SST), and only the
/// caller knows which `0/` file that is once it has read
/// `selection.model.dissipation_field()` - so the caller reads it, builds
/// these two, and hands them in unchanged, exactly as
/// `src/bin/k_omega.rs` does inline. `RasModel::Laminar` and a §13.4 error
/// never look at either.
///
/// SST additionally needs the wall distance of SPEC-LIT §6.6, computed here
/// with the case's own pressure-solver tolerance and non-orthogonal
/// corrector count (`p_solver`, `turb.n_non_orth_correctors`) - the same
/// Poisson machinery §3.2 assembles everything else with, run once at setup.
/// `wall_distance` reads `HostMesh::b_kind == PatchKind::Wall` directly, so a
/// carved castellated or cut-cell mesh's wall patches are walls to it like
/// any other - SPEC-LIT §23.4's *DESIGN* note that such faces get the
/// ordinary `wall` type is exactly what makes this work with no special case
/// here.
pub fn build_coupled<'m>(
    gpu: &Gpu,
    hm: &HostMesh,
    mesh: &'m GpuMesh,
    cc: &CaseControls,
    selection: &TurbulenceSelection,
    wall_faces: &WallFaces,
    roughness: &NutRoughness,
) -> Result<Box<dyn CoupledTurbulence + 'm>> {
    let buoy = buoyancy_settings(cc);
    let wall = wall_coeffs_from_case(&cc.wall);

    match selection.model {
        RasModel::Laminar => Ok(Box::new(CoupledLaminar::new(gpu, mesh)?)),

        RasModel::KEpsilon => {
            let d = KEpsilonCoeffs::default();
            let coeffs = KEpsilonCoeffs {
                cmu: model_coeff(cc, "Cmu", d.cmu),
                c1: model_coeff(cc, "C1", d.c1),
                c2: model_coeff(cc, "C2", d.c2),
                c3: model_coeff(cc, "C3", d.c3),
                sigmak: model_coeff(cc, "sigmak", d.sigmak),
                sigma_eps: model_coeff(cc, "sigmaEps", d.sigma_eps),
            };
            let mut model =
                KEpsilon::new(gpu, hm, mesh, coeffs, cc.turb, wall, wall_faces, roughness)?;
            if !selection.active {
                model.freeze_nut(gpu)?;
            }
            Ok(Box::new(CoupledKEpsilon::new(model, buoy)))
        }

        // SPEC-LIT §33: the coefficients are §6.1's own, unchanged -
        // `KEpsilonCoeffs` reused rather than a second, near-identical
        // struct (see `models::launder_sharma`'s module doc). `wall_faces`/
        // `roughness` should be the empty ones here: SPEC-LIT §33.2 needs no
        // wall-function machinery at all, and a caller that built non-empty
        // ones from `epsilon`'s/`nut`'s patch types under `wallTreatment
        // lowRe` will find them already empty by construction (`lowRe`
        // pins no wall-function BcKind on either field).
        RasModel::LaunderSharmaKE => {
            let d = KEpsilonCoeffs::default();
            let coeffs = KEpsilonCoeffs {
                cmu: model_coeff(cc, "Cmu", d.cmu),
                c1: model_coeff(cc, "C1", d.c1),
                c2: model_coeff(cc, "C2", d.c2),
                c3: model_coeff(cc, "C3", d.c3),
                sigmak: model_coeff(cc, "sigmak", d.sigmak),
                sigma_eps: model_coeff(cc, "sigmaEps", d.sigma_eps),
            };
            let mut model =
                LaunderSharmaKE::new(gpu, hm, mesh, coeffs, cc.turb, wall, wall_faces, roughness)?;
            if !selection.active {
                model.freeze_nut(gpu)?;
            }
            Ok(Box::new(CoupledLaunderSharmaKE::new(model, buoy)))
        }

        // SPEC-LIT §40. `buoy` is refused rather than ignored: §40.5 has no
        // G_b term for the `C_1 S eps` production form, and a coupled driver
        // that ran this model with the buoyancy silently at zero would be
        // producing exactly the plausible wrong answer §13.4 exists to stop.
        RasModel::RealizableKE => {
            refuse_realizable_ke_buoyancy(cc)?;
            let coeffs = realizable_ke_coeffs(cc)?;
            let mut model =
                RealizableKe::new(gpu, hm, mesh, coeffs, cc.turb, wall, wall_faces, roughness)?;
            if !selection.active {
                model.freeze_nut(gpu)?;
            }
            Ok(Box::new(CoupledRealizableKe::new(model)))
        }

        // SPEC-LIT §41. Buoyancy IS carried here - `C_e1 (eps/k) G` is §6.1's
        // production form exactly, so §17's term transfers unchanged.
        RasModel::RNGkEpsilon => {
            let coeffs = rng_ke_coeffs(cc)?;
            let mut model =
                RngKe::new(gpu, hm, mesh, coeffs, cc.turb, wall, wall_faces, roughness)?;
            if !selection.active {
                model.freeze_nut(gpu)?;
            }
            Ok(Box::new(CoupledRngKe::new(model, buoy)))
        }

        RasModel::KOmega => {
            let d = KOmegaCoeffs::default();
            let coeffs = KOmegaCoeffs {
                beta_star: model_coeff(cc, "betaStar", d.beta_star),
                beta: model_coeff(cc, "beta", d.beta),
                gamma: model_coeff(cc, "gamma", d.gamma),
                alpha_k: model_coeff(cc, "alphaK", d.alpha_k),
                alpha_omega: model_coeff(cc, "alphaOmega", d.alpha_omega),
            };
            let mut model =
                KOmega::new(gpu, hm, mesh, coeffs, cc.turb, wall, wall_faces, roughness)?;
            if !selection.active {
                model.freeze_nut(gpu)?;
            }
            Ok(Box::new(CoupledKOmega::new(model, buoy)))
        }

        RasModel::KOmegaSST => {
            let wd = crate::walldistance::wall_distance(
                gpu,
                hm,
                mesh,
                &cc.p_solver,
                cc.turb.n_non_orth_correctors,
            )?;

            let d = KOmegaSstCoeffs::default();
            let coeffs = KOmegaSstCoeffs {
                sigma_k1: model_coeff(cc, "sigmaK1", d.sigma_k1),
                sigma_w1: model_coeff(cc, "sigmaOmega1", d.sigma_w1),
                beta_1: model_coeff(cc, "beta1", d.beta_1),
                gamma_1: model_coeff(cc, "gamma1", d.gamma_1),
                sigma_k2: model_coeff(cc, "sigmaK2", d.sigma_k2),
                sigma_w2: model_coeff(cc, "sigmaOmega2", d.sigma_w2),
                beta_2: model_coeff(cc, "beta2", d.beta_2),
                gamma_2: model_coeff(cc, "gamma2", d.gamma_2),
                beta_star: model_coeff(cc, "betaStar", d.beta_star),
                a1: model_coeff(cc, "a1", d.a1),
                b1: model_coeff(cc, "b1", d.b1),
                c1: model_coeff(cc, "c1", d.c1),
            };
            let mut model = KOmegaSst::new(
                gpu, hm, mesh, coeffs, cc.turb, wall, wall_faces, &wd.y.f,
            )?;
            if !selection.active {
                model.freeze_nut(gpu)?;
            }
            Ok(Box::new(CoupledKOmegaSst::new(model, buoy)))
        }

        // SPEC-LIT §88. The same SST above, with §88's two equations bolted
        // on - not a second model. `KOmegaSstCoeffs` is read through the
        // same `model_coeff` calls, because a transitional case's `k` and
        // `omega` equations ARE §6.3's and a coefficient that stopped being
        // read when a transition model was named would be the §13.4.1
        // failure one level up.
        RasModel::KOmegaSstLM => {
            refuse_lm_buoyancy(cc)?;
            let sel = selection.transition.as_ref().ok_or_else(|| {
                Error::Config(
                    "momentumTransport: kOmegaSSTLM was selected with no \
                     transition record - an internal registry error \
                     (select_turbulence_model should have built one before \
                     build_coupled ever saw it), not a setting this case file \
                     can fix"
                        .to_string(),
                )
            })?;

            let wd = crate::walldistance::wall_distance(
                gpu,
                hm,
                mesh,
                &cc.p_solver,
                cc.turb.n_non_orth_correctors,
            )?;

            let d = KOmegaSstCoeffs::default();
            let coeffs = KOmegaSstCoeffs {
                sigma_k1: model_coeff(cc, "sigmaK1", d.sigma_k1),
                sigma_w1: model_coeff(cc, "sigmaOmega1", d.sigma_w1),
                beta_1: model_coeff(cc, "beta1", d.beta_1),
                gamma_1: model_coeff(cc, "gamma1", d.gamma_1),
                sigma_k2: model_coeff(cc, "sigmaK2", d.sigma_k2),
                sigma_w2: model_coeff(cc, "sigmaOmega2", d.sigma_w2),
                beta_2: model_coeff(cc, "beta2", d.beta_2),
                gamma_2: model_coeff(cc, "gamma2", d.gamma_2),
                beta_star: model_coeff(cc, "betaStar", d.beta_star),
                a1: model_coeff(cc, "a1", d.a1),
                b1: model_coeff(cc, "b1", d.b1),
                c1: model_coeff(cc, "c1", d.c1),
            };
            let mut model = KOmegaSst::new(
                gpu, hm, mesh, coeffs, cc.turb, wall, wall_faces, &wd.y.f,
            )?;
            let lm = LangtryMenter::new(gpu, mesh, sel.coeffs, sel.controls, &wd.y.f)?;
            model.set_transition(Some(lm))?;
            if !selection.active {
                model.freeze_nut(gpu)?;
            }
            Ok(Box::new(CoupledKOmegaSst::new(model, buoy)))
        }

        // SPEC-LIT §56. One transport equation and one solve, and no
        // wall-function machinery at all on `nu~` - which is why
        // `wall_faces.constrained_cells` is never read here. `nut`'s own set
        // still is, because `nu_t`'s wall value is `nut`'s business whichever
        // model computed the interior (§15.5).
        //
        // Buoyancy is REFUSED rather than ignored, exactly as §40.5 refuses it
        // for `realizableKE`: §17's `G_b` enters a `k` equation and this model
        // has none.
        RasModel::SpalartAllmaras => {
            refuse_sa_buoyancy(cc)?;
            let coeffs = sa_coeffs(cc)?;
            let wd = crate::walldistance::wall_distance(
                gpu,
                hm,
                mesh,
                &cc.p_solver,
                cc.turb.n_non_orth_correctors,
            )?;
            let mut model = SpalartAllmaras::new(
                gpu, hm, mesh, coeffs, cc.turb, wall, wall_faces, roughness, &wd.y.f,
            )?;
            if !selection.active {
                model.freeze_nut(gpu)?;
            }
            Ok(Box::new(CoupledSpalartAllmaras::new(model)))
        }

        // SPEC-LIT §57. The hybrids, on either background. The three guards
        // `select_hybrid` could apply without a mesh have already fired; the
        // fourth - a 2-D mesh - needs one and fires here.
        RasModel::HybridSa | RasModel::HybridSst => {
            let sel = selection.des.as_ref().ok_or_else(|| {
                Error::Config(
                    "momentumTransport: a hybrid was selected with no DES record \
                     - an internal registry error (select_turbulence_model should \
                     have refused this case before build_coupled ever saw it), \
                     not a setting this case file can fix"
                        .to_string(),
                )
            })?;
            refuse_two_dimensional_hybrid(hm, sel)?;

            let wd = crate::walldistance::wall_distance(
                gpu,
                hm,
                mesh,
                &cc.p_solver,
                cc.turb.n_non_orth_correctors,
            )?;
            let des = crate::models::des::DesLengthScale::new(
                gpu,
                mesh,
                &wd.y.f,
                &wd.grad_y,
                sel.branch,
                sel.delta,
                sel.background,
                sel.coeffs,
            )?;

            match sel.background {
                HybridBackground::Sa => {
                    refuse_sa_buoyancy(cc)?;
                    let mut model = SpalartAllmaras::new(
                        gpu, hm, mesh, sel.sa, cc.turb, wall, wall_faces, roughness,
                        &wd.y.f,
                    )?;
                    model.set_des(Some(des));
                    if !selection.active {
                        model.freeze_nut(gpu)?;
                    }
                    Ok(Box::new(CoupledSpalartAllmaras::new(model)))
                }
                HybridBackground::Sst => {
                    let d = KOmegaSstCoeffs::default();
                    let coeffs = KOmegaSstCoeffs {
                        sigma_k1: model_coeff(cc, "sigmaK1", d.sigma_k1),
                        sigma_w1: model_coeff(cc, "sigmaOmega1", d.sigma_w1),
                        beta_1: model_coeff(cc, "beta1", d.beta_1),
                        gamma_1: model_coeff(cc, "gamma1", d.gamma_1),
                        sigma_k2: model_coeff(cc, "sigmaK2", d.sigma_k2),
                        sigma_w2: model_coeff(cc, "sigmaOmega2", d.sigma_w2),
                        beta_2: model_coeff(cc, "beta2", d.beta_2),
                        gamma_2: model_coeff(cc, "gamma2", d.gamma_2),
                        beta_star: model_coeff(cc, "betaStar", d.beta_star),
                        a1: model_coeff(cc, "a1", d.a1),
                        b1: model_coeff(cc, "b1", d.b1),
                        c1: model_coeff(cc, "c1", d.c1),
                    };
                    let mut model = KOmegaSst::new(
                        gpu, hm, mesh, coeffs, cc.turb, wall, wall_faces, &wd.y.f,
                    )?;
                    model.set_des(Some(des));
                    if !selection.active {
                        model.freeze_nut(gpu)?;
                    }
                    Ok(Box::new(CoupledKOmegaSst::new(model, buoy)))
                }
            }
        }

        // SPEC-LIT §30.2: the LES family, over `CoupledLes`/`Les`. `wall_faces`
        // means something different here than it does for the RAS arms above
        // - `select_les`/`select_turbulence_model` never populate it (an LES
        // case has no dissipation field to read a `constrained_cells` set
        // from), so the CALLER is the one who must have built
        // `wall_faces.nut` from `nut`'s own patch types via
        // `crate::field_setup::les_nut_wall_faces` (SPEC-LIT §30.1) rather
        // than the RAS `nut_wall_faces` - exactly as `build_coupled`'s own
        // doc already requires for the dissipation-keyed RAS case.
        // `roughness` is never read: SPEC-LIT §30.1 has no rough LES wall
        // model yet, so there is nothing for it to feed.
        RasModel::Les => {
            let sel = selection.les.as_ref().ok_or_else(|| {
                Error::Config(
                    "momentumTransport: simulationType LES selected with no LES \
                     model recorded - an internal registry error (select_turbulence_model \
                     should have refused this case before build_coupled ever saw it), \
                     not a setting this case file can fix"
                        .to_string(),
                )
            })?;

            // SPEC-LIT §16.4/§30.2: the wall distance is a prerequisite for
            // van Driest damping and NOTHING else in an LES - unlike SST, it
            // is not paid for unconditionally. A delta spec that never wraps
            // `vanDriest` gets the same `NO_WALL`/zero sentinel a wall-free
            // domain gets, which is what makes the damping inert without a
            // Poisson solve nobody asked for.
            let n = hm.n_cells.max(1);
            let (y, grad_y) = if sel.delta.van_driest {
                let wd = crate::walldistance::wall_distance(
                    gpu,
                    hm,
                    mesh,
                    &cc.p_solver,
                    cc.turb.n_non_orth_correctors,
                )?;
                (wd.y.f, wd.grad_y)
            } else {
                (
                    gpu.upload(&vec![crate::walldistance::NO_WALL; n])?,
                    gpu.upload(&vec![Vec3::ZERO; n])?,
                )
            };

            let mut model = Les::new(
                gpu,
                hm,
                mesh,
                sel.model,
                sel.coeffs,
                sel.delta,
                cc.turb,
                wall_faces,
                &y,
                &grad_y,
            )?;
            if !selection.active {
                model.freeze_nut(gpu)?;
            }
            Ok(Box::new(CoupledLes::new(model)))
        }
    }
}

// ==========================================================================
//  Tests
// ==========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn case(src: &str) -> CaseControls {
        let d = match FoamDict::parse(src, "momentumTransport") {
            Ok(d) => d,
            Err(e) => panic!("{e}"),
        };
        let name = d
            .get_or("RAS/model", d.get_or("RAS/RASModel", ""))
            .to_string();

        CaseControls {
            model_name: name,
            momentum_transport: d,
            ..Default::default()
        }
    }

    /// A case whose `fvSchemes` names a scheme, so the hybrid guards of
    /// SPEC-LIT §57.10 have something to read.
    fn hybrid_case(momentum: &str, div_u: &str, steady: bool) -> CaseControls {
        let d = FoamDict::parse(momentum, "momentumTransport").expect("momentumTransport");
        let sch = FoamDict::parse(
            &format!("divSchemes {{ div(phi,U) {div_u}; default Gauss linear; }}"),
            "fvSchemes",
        )
        .expect("fvSchemes");
        let name = d
            .get_or("RAS/model", d.get_or("RAS/RASModel", ""))
            .to_string();
        let mut c = CaseControls {
            model_name: name,
            momentum_transport: d,
            schemes: crate::io::schemes::FvSchemes::from_dict(sch),
            ..Default::default()
        };
        c.turb.steady = steady;
        c
    }

    /// A hybrid case with everything the four guards want.
    fn good_hybrid(momentum: &str) -> CaseControls {
        hybrid_case(momentum, "Gauss linear", false)
    }

    // ------------------------------------------------------------------
    //  SPEC-LIT §89.4 - the transition model's pair tests
    // ------------------------------------------------------------------

    /// The base transitional case: every §88 setting written explicitly, plus
    /// the `system/` entries the two new equations read, so that a pair test
    /// can replace exactly one substring and change nothing else.
    const LM_MOMENTUM: &str = "\
        simulationType RAS; \
        RAS { model kOmegaSSTLM; betaStar 0.09; \
              ca1 2.0; ca2 0.06; ce1 1.0; ce2 50; cThetat 0.03; s1 2; \
              sigmaf 1.0; sigmaThetat 2.0; \
              nReThetaSweeps 10; gammaMin 0; gammaMax 1; ReThetatMin 20; }";

    const LM_SCHEMES: &str = "\
        divSchemes { default Gauss linear; \
                     div(phi,k) bounded Gauss upwind; \
                     div(phi,omega) bounded Gauss upwind; \
                     div(phi,gamma) bounded Gauss upwind; \
                     div(phi,ReThetat) bounded Gauss upwind; }";

    const LM_SOLUTION: &str = "\
        solvers { gamma { solver PBiCGStab; tolerance 1e-9; maxIter 100; } \
                  ReThetat { solver PBiCGStab; tolerance 1e-9; maxIter 100; } } \
        relaxationFactors { equations { gamma 0.7; ReThetat 0.7; } }";

    /// A whole transitional case, from three dictionary sources.
    fn lm_case(momentum: &str, schemes: &str, solution: &str) -> CaseControls {
        let d = FoamDict::parse(momentum, "momentumTransport").expect("momentumTransport");
        let sch = FoamDict::parse(schemes, "fvSchemes").expect("fvSchemes");
        let sol = FoamDict::parse(solution, "fvSolution").expect("fvSolution");
        let name = d
            .get_or("RAS/model", d.get_or("RAS/RASModel", ""))
            .to_string();
        CaseControls {
            model_name: name,
            momentum_transport: d,
            schemes: crate::io::schemes::FvSchemes::from_dict(sch),
            fv_solution: sol,
            ..Default::default()
        }
    }

    fn lm_base() -> CaseControls {
        lm_case(LM_MOMENTUM, LM_SCHEMES, LM_SOLUTION)
    }

    /// **SPEC-LIT §89.4, rows 1-15: fifteen pairs differing in one entry,
    /// each REQUIRED to reach the solver.**
    ///
    /// Rows 10-15 are the §13.4.1 instances this section exists to prevent.
    /// Every one of them would have been inert had `LmControls` reached for
    /// `TurbulenceControls`' `k`/`epsilon` slots the way
    /// `TurbulenceControls::epsilon_solver` already reaches for `omega`'s -
    /// and every one of them is a setting a real case writes.
    #[test]
    fn every_transition_setting_reaches_the_solver() {
        let base = select_turbulence_model(&lm_base()).expect("the base case is valid");
        let base_t = base.transition.expect("a transition record");

        // Rows 2-9: one coefficient each, out of `constant/momentumTransport`.
        for (from, to, what) in [
            ("ca1 2.0", "ca1 4.0", "the intermittency production"),
            ("ce2 50", "ce2 25", "the destruction and F_thetat"),
            ("cThetat 0.03", "cThetat 0.06", "the ReThetat source"),
            ("sigmaThetat 2.0", "sigmaThetat 1.0", "the ReThetat diffusivity"),
            ("nReThetaSweeps 10", "nReThetaSweeps 3", "the fixed point"),
            ("gammaMax 1", "gammaMax 0.5", "the bound"),
            ("ReThetatMin 20", "ReThetatMin 50", "the bound"),
        ] {
            let src = LM_MOMENTUM.replace(from, to);
            assert_ne!(src, LM_MOMENTUM, "the pair did not differ: {from}");
            let s = select_turbulence_model(&lm_case(&src, LM_SCHEMES, LM_SOLUTION))
                .expect("the varied case is valid");
            let t = s.transition.expect("a transition record");
            assert_ne!(
                t.coeffs, base_t.coeffs,
                "`{from}` -> `{to}` is INERT: {what} does not move (SPEC-LIT 89.4)"
            );
        }

        // Row 9: §6.3's own constant, still read under this model.
        let src = LM_MOMENTUM.replace("betaStar 0.09", "betaStar 0.08");
        let c = lm_case(&src, LM_SCHEMES, LM_SOLUTION);
        assert_ne!(
            model_coeff(&c, "betaStar", 0.09),
            model_coeff(&lm_base(), "betaStar", 0.09),
            "betaStar is inert under kOmegaSSTLM - the k and omega equations \
             ARE section 6.3's (SPEC-LIT 89.1)"
        );

        // Rows 10-13: `system/fvSolution`, per equation and per entry.
        for (from, to, field) in [
            ("gamma { solver PBiCGStab; tolerance 1e-9;", "gamma { solver PBiCGStab; tolerance 1e-5;", "gamma solver"),
            ("ReThetat { solver PBiCGStab; tolerance 1e-9; maxIter 100;", "ReThetat { solver PBiCGStab; tolerance 1e-9; maxIter 7;", "ReThetat solver"),
            ("gamma 0.7;", "gamma 0.4;", "gamma relaxation"),
            ("ReThetat 0.7;", "ReThetat 0.3;", "ReThetat relaxation"),
        ] {
            let src = LM_SOLUTION.replace(from, to);
            assert_ne!(src, LM_SOLUTION, "the pair did not differ: {field}");
            let s = select_turbulence_model(&lm_case(LM_MOMENTUM, LM_SCHEMES, &src))
                .expect("the varied case is valid");
            let t = s.transition.expect("a transition record");
            assert_ne!(
                t.controls, base_t.controls,
                "`{field}` is INERT - it was read and thrown away (SPEC-LIT 89.4)"
            );
        }

        // Rows 14-15: `system/fvSchemes`, per equation.
        for (from, to, field) in [
            ("div(phi,gamma) bounded Gauss upwind", "div(phi,gamma) Gauss linear", "div(phi,gamma)"),
            (
                "div(phi,ReThetat) bounded Gauss upwind",
                "div(phi,ReThetat) Gauss linear",
                "div(phi,ReThetat)",
            ),
        ] {
            let src = LM_SCHEMES.replace(from, to);
            assert_ne!(src, LM_SCHEMES, "the pair did not differ: {field}");
            let s = select_turbulence_model(&lm_case(LM_MOMENTUM, &src, LM_SOLUTION))
                .expect("the varied case is valid");
            let t = s.transition.expect("a transition record");
            assert_ne!(
                t.controls, base_t.controls,
                "`{field}` is INERT - the gamma equation took some other \
                 entry's scheme (SPEC-LIT 89.4)"
            );
        }
    }

    /// **The two new equations read their OWN entries, not each other's.**
    ///
    /// The sharper half of §89.4: a reader that took `div(phi,gamma)` for
    /// both equations would pass every row above - each pair would still
    /// differ - and would be wrong. This asserts that moving `gamma`'s entry
    /// leaves `ReThetat`'s alone, and the reverse.
    #[test]
    fn the_two_transition_equations_do_not_share_an_entry() {
        let base = select_turbulence_model(&lm_base())
            .expect("base")
            .transition
            .expect("record");

        let only_gamma = LM_SCHEMES.replace(
            "div(phi,gamma) bounded Gauss upwind",
            "div(phi,gamma) Gauss linear",
        );
        let t = select_turbulence_model(&lm_case(LM_MOMENTUM, &only_gamma, LM_SOLUTION))
            .expect("varied")
            .transition
            .expect("record");
        assert_ne!(t.controls.gamma_conv, base.controls.gamma_conv);
        assert_eq!(
            t.controls.re_thetat_conv, base.controls.re_thetat_conv,
            "changing div(phi,gamma) moved the ReThetat equation too"
        );

        let only_rtt = LM_SOLUTION.replace("ReThetat 0.7;", "ReThetat 0.3;");
        let t = select_turbulence_model(&lm_case(LM_MOMENTUM, LM_SCHEMES, &only_rtt))
            .expect("varied")
            .transition
            .expect("record");
        assert_ne!(t.controls.re_thetat_relax, base.controls.re_thetat_relax);
        assert_eq!(
            t.controls.gamma_relax, base.controls.gamma_relax,
            "changing ReThetat's relaxation moved gamma's too"
        );
    }

    /// **SPEC-LIT §89.4: every inert key is refused by name.**
    #[test]
    fn the_transition_models_inert_keys_are_refused_by_name() {
        for (key, value) in [
            ("Cmu", "0.09"),
            ("C1", "1.44"),
            ("C2", "1.92"),
            ("C3", "0.0"),
            ("sigmak", "1.0"),
            ("sigmaEps", "1.3"),
            ("FlengthCoeff", "1.0"),
            ("ReThetacCoeff", "1.0"),
            ("Tu", "3.3"),
        ] {
            let src = LM_MOMENTUM.replace("betaStar 0.09;", &format!("betaStar 0.09; {key} {value};"));
            assert_ne!(src, LM_MOMENTUM);
            let e = select_turbulence_model(&lm_case(&src, LM_SCHEMES, LM_SOLUTION))
                .expect_err("an inert key must be refused");
            let m = e.to_string();
            assert!(m.contains(key), "the refusal does not name {key}: {m}");
        }
    }

    /// **SPEC-LIT §88.9: a buoyant transitional case is refused by name.**
    ///
    /// Not because the term is missing - `G_b` enters a `k` equation and this
    /// model has one - but because whether `gamma_eff` multiplies it is a
    /// question Langtry & Menter do not answer, and both answers change where
    /// a buoyant layer transitions.
    #[test]
    fn a_buoyant_transitional_case_is_refused_by_name() {
        let mut cc = lm_base();
        cc.buoyancy.g = crate::Vec3::new(0.0, -9.81, 0.0);
        assert!(cc.buoyancy.is_active());
        let e = refuse_lm_buoyancy(&cc).expect_err("gravity under kOmegaSSTLM must be refused");
        let m = e.to_string();
        assert!(m.contains("kOmegaSSTLM"), "{m}");
        assert!(m.contains("G_b"), "the refusal does not name the term: {m}");
        assert!(m.contains("kOmegaSST"), "the refusal names no alternative: {m}");
        // And with no gravity it is silent.
        let mut cc = lm_base();
        cc.buoyancy.g = crate::Vec3::ZERO;
        refuse_lm_buoyancy(&cc).expect("a case with no gravity is fine");
    }

    /// **SPEC-LIT §89.1: the dissipation slot a transitional case's `omega`
    /// equation reads is `omega`'s own, through the string route too.**
    #[test]
    fn the_transition_model_reads_the_omega_entries() {
        use crate::io::case::dissipation_from_model;
        assert_eq!(dissipation_from_model("kOmegaSSTLM"), Some("omega"));
    }

    /// **SPEC-LIT §88.9: the coefficient guards fire through the case
    /// reader, not only through `LmCoeffs::check`.**
    #[test]
    fn the_transition_coefficient_guards_fire_from_a_case_file() {
        for (from, to, needle) in [
            ("ce2 50", "ce2 1", "ce2"),
            ("sigmaf 1.0", "sigmaf 0", "sigmaf"),
            ("sigmaThetat 2.0", "sigmaThetat -1", "sigmaThetat"),
            ("nReThetaSweeps 10", "nReThetaSweeps 0", "nReThetaSweeps"),
            ("nReThetaSweeps 10", "nReThetaSweeps 500", "nReThetaSweeps"),
            ("gammaMin 0", "gammaMin 2", "gammaMin"),
            ("ReThetatMin 20", "ReThetatMin 5", "ReThetatMin"),
        ] {
            let src = LM_MOMENTUM.replace(from, to);
            assert_ne!(src, LM_MOMENTUM, "{from}");
            let e = select_turbulence_model(&lm_case(&src, LM_SCHEMES, LM_SOLUTION))
                .expect_err("the guard must fire");
            let m = e.to_string();
            assert!(m.contains(needle), "the refusal does not name {needle}: {m}");
        }
        // And the one that is ALLOWED: gammaMin == gammaMax freezes the
        // intermittency, which is Gate 88-R's fully-turbulent limit.
        let src = LM_MOMENTUM.replace("gammaMin 0", "gammaMin 1");
        select_turbulence_model(&lm_case(&src, LM_SCHEMES, LM_SOLUTION))
            .expect("a frozen intermittency is a legitimate setting - SPEC-LIT 88.8");
    }

    /// **SPEC-LIT §89.5: a transitional case builds, end to end, and the
    /// banner and the written field set both say what it is.**
    #[test]
    fn a_transitional_case_builds_through_the_coupled_route() {
        let Ok(gpu) = Gpu::new(0) else { return };
        let hm = {
            let mut spec = crate::blockgen::BlockSpec {
                x: crate::blockgen::GradedAxis {
                    lo: 0.0, hi: 1.0, n: 4, expansion: 1.0, two_sided: false,
                },
                y: crate::blockgen::GradedAxis {
                    lo: 0.0, hi: 0.2, n: 4, expansion: 1.0, two_sided: false,
                },
                z: crate::blockgen::GradedAxis {
                    lo: 0.0, hi: 0.2, n: 4, expansion: 1.0, two_sided: false,
                },
                ..Default::default()
            };
            spec.patch_type[4] = "patch".to_string();
            spec.patch_type[5] = "patch".to_string();
            crate::blockgen::build_mesh(&spec).expect("block")
        };
        let mesh = crate::mesh::GpuMesh::upload(&gpu, &hm).expect("mesh");
        let no_walls = crate::field_setup::WallFaces::none(hm.n_boundary_faces);
        let no_rough = crate::field_setup::NutRoughness::none(hm.n_boundary_faces);

        let mut cc = lm_base();
        cc.buoyancy.g = crate::Vec3::ZERO;
        let sel = select_turbulence_model(&cc).expect("selection");
        assert_eq!(sel.model, RasModel::KOmegaSstLM);
        assert!(sel.describe().contains("kOmegaSSTLM"), "{}", sel.describe());

        let turb = build_coupled(&gpu, &hm, &mesh, &cc, &sel, &no_walls, &no_rough)
            .expect("kOmegaSSTLM must build");
        assert_eq!(turb.name(), "kOmegaSSTLM");
        let names: Vec<&str> = turb.output_fields().iter().map(|(n, _)| *n).collect();
        assert_eq!(names, vec!["k", "omega", "nut", "gamma", "ReThetat"]);

        // And plain SST out of the same route is untouched: three names, and
        // the banner it always printed.
        let mut cc = case("RAS { model kOmegaSST; }");
        cc.buoyancy.g = crate::Vec3::ZERO;
        let sel = select_turbulence_model(&cc).expect("selection");
        let turb = build_coupled(&gpu, &hm, &mesh, &cc, &sel, &no_walls, &no_rough)
            .expect("kOmegaSST must still build");
        assert_eq!(turb.name(), "kOmegaSST");
        let names: Vec<&str> = turb.output_fields().iter().map(|(n, _)| *n).collect();
        assert_eq!(names, vec!["k", "omega", "nut"]);
    }

    // ------------------------------------------------------------------
    //  SPEC-LIT §56 / §57 / §58 - the refusals that became capabilities
    // ------------------------------------------------------------------

    /// `SpalartAllmaras` used to be in `RECOGNISED_NOT_IMPLEMENTED`. It now
    /// selects a model, and both lists say so.
    #[test]
    fn spalart_allmaras_now_selects_a_model() {
        let s = select_turbulence_model(&case("RAS { model SpalartAllmaras; }"))
            .expect("SpalartAllmaras is implemented");
        assert_eq!(s.model, RasModel::SpalartAllmaras);
        assert!(s.active);
        assert!(s.des.is_none());
        // SPEC-LIT §58.1: two accessors, two meanings, both honest.
        assert_eq!(s.model.dissipation_field(), None);
        assert_eq!(s.model.transported_fields(), &["nuTilda"]);
    }

    /// Every hybrid name selects the branch and background it carries, under
    /// BOTH spellings - SPEC-LIT §58.1.
    #[test]
    fn every_hybrid_name_selects_its_own_branch_under_both_spellings() {
        for (name, branch, background) in HYBRID_REGISTRY {
            // `simulationType LES;` with the hybrid name in the LES block.
            let c = good_hybrid(&format!(
                "simulationType LES; LES {{ model {name}; }}"
            ));
            let s = select_turbulence_model(&c).unwrap_or_else(|e| panic!("{name}: {e}"));
            let h = s.des.expect("a hybrid selection");
            assert_eq!(h.branch, *branch, "{name}");
            assert_eq!(h.background, *background, "{name}");
            assert_eq!(h.model_name(), *name);

            // `simulationType <branch>;` with a DES block.
            let c = good_hybrid(&format!(
                "simulationType {}; DES {{ model {name}; }}",
                branch.name()
            ));
            let s = select_turbulence_model(&c).unwrap_or_else(|e| panic!("{name}: {e}"));
            let h = s.des.expect("a hybrid selection");
            assert_eq!(h.branch, *branch, "{name}");
            assert_eq!(h.background, *background, "{name}");

            // The `0/` files follow the background, not the branch.
            let want: &[&str] = match background {
                HybridBackground::Sa => &["nuTilda"],
                HybridBackground::Sst => &["k", "omega"],
            };
            assert_eq!(s.model.transported_fields(), want, "{name}");
        }
    }

    /// SPEC-LIT §58.2's bookkeeping, in both directions and for all four
    /// lists - the drift those lists' own doc comments warn about, checked.
    #[test]
    fn the_refusal_lists_and_the_registries_do_not_overlap() {
        for (name, _, _) in HYBRID_REGISTRY {
            assert!(
                !LES_RECOGNISED_NOT_IMPLEMENTED.contains(name),
                "{name} is both implemented and refused"
            );
            assert!(
                RECOGNISED_NOT_IMPLEMENTED.iter().all(|(n, _)| n != name),
                "{name} is both implemented and refused"
            );
            assert!(available_hybrid_models().contains(name));
        }
        for name in available_hybrid_models() {
            assert!(HYBRID_REGISTRY.iter().any(|(n, _, _)| *n == name));
        }
        // `IDDESDelta` left the delta refusal list, and did NOT join the
        // pure-LES menu: (57.17) is defined only inside IDDES.
        for name in DELTA_HYBRID_ONLY {
            assert!(!DELTA_RECOGNISED_NOT_IMPLEMENTED.contains(name));
            assert!(!DELTA_NAMES.contains(name), "{name} must not be a pure-LES width");
            assert!(available_hybrid_deltas().contains(name));
        }
        // Every still-refused RAS name names an alternative that IS
        // implemented, or says plainly that none is close - SPEC-LIT §58.2.
        // The task that produced §56 asked for exactly this to be checked in
        // both directions, and "a published model ofgpu has not got" - the
        // one bare hint every name carried before - passes neither half.
        let menu: Vec<&str> = available_models()
            .into_iter()
            .chain(available_hybrid_models())
            .collect();
        for (name, why) in RECOGNISED_NOT_IMPLEMENTED {
            assert!(why.len() > 40, "{name}'s refusal says nothing useful: {why}");
            let names_one = menu.iter().any(|m| why.contains(m));
            let says_none = why.contains("Nothing in ofgpu is close");
            assert!(
                names_one || says_none,
                "{name}'s refusal names no implemented alternative and does \
                 not say that none is close: {why}"
            );
        }
    }

    /// **SPEC-LIT §89.3: `kOmegaSSTLM` is no longer refused, and its
    /// successor is - with the reason the successor is the better model.**
    ///
    /// This test used to be
    /// `the_transition_model_is_refused_by_name_and_names_its_successor` and
    /// it asserted the opposite. It is REWRITTEN rather than deleted,
    /// because §58.3's refusal message named three things - the paper, the
    /// successor, and the specific way `kOmegaSST` would be wrong in its
    /// place - and all three have to survive the move: the first two into
    /// §88's own file headers, the third into the successor's refusal.
    #[test]
    fn the_transition_model_now_selects_a_model() {
        let s = select_turbulence_model(&case("RAS { model kOmegaSSTLM; }"))
            .expect("kOmegaSSTLM is implemented - SPEC-LIT 88");
        assert_eq!(s.model, RasModel::KOmegaSstLM);
        assert_eq!(s.model.transported_fields(), &["k", "omega", "gamma", "ReThetat"]);
        assert_eq!(s.model.dissipation_field(), Some("omega"));
        assert!(s.transition.is_some(), "a kOmegaSSTLM case must carry a transition record");
        assert!(available_models().contains(&"kOmegaSSTLM"));
        assert!(RECOGNISED_NOT_IMPLEMENTED.iter().all(|(n, _)| *n != "kOmegaSSTLM"));

        // And the successor is refused, naming the model that IS here and
        // the reason it is nevertheless the weaker of the two.
        let e = select_turbulence_model(&case("RAS { model kOmegaSSTGamma; }"))
            .expect_err("the 2015 one-equation gamma model is not implemented");
        let m = e.to_string();
        assert!(m.contains("kOmegaSSTGamma"), "{m}");
        assert!(m.contains("kOmegaSSTLM"), "the refusal does not name what IS here: {m}");
        assert!(m.contains("2015"), "the refusal does not cite Menter et al. (2015): {m}");
        assert!(
            m.contains("Galilean"),
            "the refusal does not say why the successor is better: {m}"
        );
    }

    /// SPEC-LIT §89.6: the Reynolds-stress family is refused by name, with
    /// what it would take.
    ///
    /// "Nothing in ofgpu is close" was true and was all the old message
    /// said. A refusal that says what the gap IS is a different message: it
    /// lets a reader decide whether the gap is one they can close.
    #[test]
    fn the_reynolds_stress_models_are_refused_with_what_it_would_take() {
        for (name, year) in [("LRR", "1975"), ("SSG", "1991")] {
            let e = select_turbulence_model(&case(&format!("RAS {{ model {name}; }}")))
                .expect_err("Reynolds-stress transport is not implemented");
            let m = e.to_string();
            assert!(m.contains(name), "{m}");
            assert!(m.contains(year), "{name}'s refusal does not cite its paper: {m}");
            assert!(
                m.contains("SEVEN") || m.contains("as LRR"),
                "{name}'s refusal does not say how many equations, nor defer                  to the entry that does: {m}"
            );
            assert!(
                m.contains("kOmegaSST"),
                "{name}'s refusal names no implemented alternative: {m}"
            );
        }
        // The two messages must differ where the two models differ. SSG's
        // whole content beyond LRR is the quadratic pressure-strain term, so
        // a message that did not mention it would be LRR's with the name
        // changed - which is the drift this crate's refusals exist to avoid.
        let lrr = RECOGNISED_NOT_IMPLEMENTED.iter().find(|(n, _)| *n == "LRR").unwrap().1;
        let ssg = RECOGNISED_NOT_IMPLEMENTED.iter().find(|(n, _)| *n == "SSG").unwrap().1;
        assert!(lrr.contains("LINEAR"), "{lrr}");
        assert!(ssg.contains("QUADRATIC"), "{ssg}");
        assert!(ssg.contains("invariants"), "{ssg}");
    }

    /// SPEC-LIT §57.10's guard 1: a steady run is refused by name.
    ///
    /// DES is unsteady by construction, and a steady DES is a RANS model
    /// whose destruction term has been divided by the wrong length.
    #[test]
    fn a_steady_hybrid_is_refused_by_name() {
        let m = "simulationType LES; LES { model SpalartAllmarasDDES; }";
        let e = select_turbulence_model(&hybrid_case(m, "Gauss linear", true))
            .expect_err("a steady hybrid must be refused");
        let t = e.to_string();
        assert!(t.contains("ddtSchemes"), "{t}");
        assert!(t.contains("UNSTEADY") || t.contains("unsteady"), "{t}");
        assert!(t.contains("Euler"), "the menu is missing: {t}");
        // And the same case with a time scheme is accepted.
        select_turbulence_model(&hybrid_case(m, "Gauss linear", false))
            .expect("a transient hybrid is fine");
    }

    /// The upwind refusal, separated out so its message is checked once.
    #[test]
    fn an_upwind_biased_momentum_scheme_is_refused_under_a_hybrid() {
        let m = "simulationType LES; LES { model SpalartAllmarasDDES; }";
        for sch in ["Gauss upwind", "Gauss linearUpwind grad(U)", "Gauss blended 0.2"] {
            let e = select_turbulence_model(&hybrid_case(m, sch, false))
                .expect_err("an upwind-biased scheme under a hybrid must be refused");
            let t = e.to_string();
            assert!(t.contains("div(phi,U)"), "{sch}: {t}");
            assert!(t.contains("Travin"), "{sch}: the refusal does not name the fix: {t}");
        }
        // And the low-dissipation ones are accepted.
        for sch in ["Gauss linear", "Gauss cubic", "Gauss linearUpwindBlended 0.75"] {
            select_turbulence_model(&hybrid_case(m, sch, false))
                .unwrap_or_else(|e| panic!("{sch} should be accepted under a hybrid: {e}"));
        }
    }

    /// `cubeRootVol`, `Scotti` and the two LES width WRAPPERS are refused
    /// under a hybrid, each with its own reason - SPEC-LIT §57.10.
    #[test]
    fn the_hybrid_width_refusals_say_why() {
        for (delta, needle) in [
            ("cubeRootVol", "calibrated"),
            ("vanDriest", "WRAPPER"),
            ("smooth", "WRAPPER"),
            ("Scotti", "calibrated"),
        ] {
            let c = good_hybrid(&format!(
                "simulationType LES; LES {{ model SpalartAllmarasDDES; delta {delta}; }}"
            ));
            let e = select_turbulence_model(&c).expect_err("must be refused");
            let t = e.to_string();
            assert!(t.contains(delta), "{delta}: {t}");
            assert!(t.contains(needle), "{delta}: the reason is missing: {t}");
            assert!(t.contains("maxDeltaxyz"), "{delta}: no menu: {t}");
        }
    }

    /// `IDDESDelta` under a PURE LES is refused with a message saying where
    /// it does live - SPEC-LIT §58.2.
    #[test]
    fn the_iddes_width_under_a_pure_les_names_where_it_lives() {
        let c = case("simulationType LES; LES { model WALE; delta IDDESDelta; }");
        let e = select_turbulence_model(&c).expect_err("IDDESDelta is not a pure-LES width");
        let t = e.to_string();
        assert!(t.contains("IDDESDelta"), "{t}");
        assert!(t.contains("IDDES"), "{t}");
        assert!(t.contains("HAS"), "the message does not say ofgpu has it: {t}");
    }

    /// **SPEC-LIT §58.1: the branch `simulationType` names and the branch the
    /// model name carries must AGREE.**
    #[test]
    fn a_branch_mismatch_between_the_two_spellings_is_refused() {
        let c = good_hybrid("simulationType DDES; DES { model SpalartAllmarasIDDES; }");
        let e = select_turbulence_model(&c).expect_err("a branch mismatch must be refused");
        let t = e.to_string();
        assert!(t.contains("DDES"), "{t}");
        assert!(t.contains("IDDES"), "{t}");
        assert!(t.contains("two different things"), "{t}");

        // And the matching pair is fine.
        select_turbulence_model(&good_hybrid(
            "simulationType IDDES; DES { model SpalartAllmarasIDDES; }",
        ))
        .expect("a matching pair is not a conflict");
    }

    /// SPEC-LIT §56.8's inert-coefficient refusals, `Cw1` first because it is
    /// refused for a reason no other entry has: it is DERIVED.
    #[test]
    fn spalart_allmaras_refuses_the_coefficients_it_does_not_read() {
        for (key, needle) in [
            ("Cw1", "DERIVED"),
            ("Cmu", "no C_mu"),
            ("sigmak", "sigmaNut"),
            ("sigmaEps", "sigmaNut"),
            ("alphak", "RNGkEpsilon"),
            ("betaStar", "k-omega"),
            ("A0", "realizableKE"),
            ("C1", "epsilon equation"),
        ] {
            let c = case(&format!("RAS {{ model SpalartAllmaras; {key} 1.0; }}"));
            let e = sa_coeffs(&c).expect_err("an inert coefficient must be refused");
            let t = e.to_string();
            assert!(t.contains(key), "{key}: {t}");
            assert!(t.contains(needle), "{key}: the reason is missing: {t}");
        }
        // The keys it DOES read are not refused.
        let c = case(
            "RAS { model SpalartAllmaras; Cb1 0.1355; Cb2 0.622; Cv1 7.1; \
             sigmaNut 0.6666666666; kappa 0.41; Cn1 16; }",
        );
        let got = sa_coeffs(&c).expect("the keys SA reads must be accepted");
        assert_eq!(got.cv1, 7.1);
    }

    /// An unknown `variant` is refused with the menu - SPEC-LIT §56.8.
    #[test]
    fn an_unknown_sa_variant_is_refused_with_the_menu() {
        let c = case("RAS { model SpalartAllmaras; variant SA-noft3; }");
        let e = sa_coeffs(&c).expect_err("an unknown variant must be refused");
        let t = e.to_string();
        assert!(t.contains("SA-noft3"), "{t}");
        assert!(t.contains("noft2-neg"), "the menu is missing: {t}");

        for (spelling, want) in [
            ("noft2", SaVariant::Noft2),
            ("SA-noft2-neg", SaVariant::Noft2Neg),
            ("SA-neg", SaVariant::Ft2Neg),
            ("ft2", SaVariant::Ft2),
        ] {
            let c = case(&format!(
                "RAS {{ model SpalartAllmaras; variant {spelling}; }}"
            ));
            assert_eq!(sa_coeffs(&c).expect("a known variant").variant, want);
        }
    }

    /// SPEC-LIT §57.5: the calibrations are per-background, and mixing them
    /// is refused by name.
    #[test]
    fn the_per_background_des_constants_are_refused_on_the_wrong_background() {
        // CDES under an SST hybrid.
        let c = good_hybrid("simulationType LES; LES { model kOmegaSSTDDES; CDES 0.65; }");
        let e = select_turbulence_model(&c).expect_err("CDES under SST must be refused");
        let t = e.to_string();
        assert!(t.contains("CDES"), "{t}");
        assert!(t.contains("CDES1"), "the alternative is missing: {t}");

        // CDES1 under an SA hybrid.
        let c =
            good_hybrid("simulationType LES; LES { model SpalartAllmarasDDES; CDES1 0.78; }");
        let e = select_turbulence_model(&c).expect_err("CDES1 under SA must be refused");
        let t = e.to_string();
        assert!(t.contains("CDES1"), "{t}");
        assert!(t.contains("ONE C_DES") || t.contains("one C_DES"), "{t}");
    }

    /// **SPEC-LIT §58.4's case-document pairs: two documents differing in one
    /// entry, REQUIRED to select something different.**
    #[test]
    fn the_hybrid_case_document_pairs_each_change_the_selection() {
        let base = "simulationType LES; LES { model SpalartAllmarasDDES; delta maxDeltaxyz; \
                    CDES 0.65; Cdt1 8; Cw 0.15; }";
        let read = |src: &str| {
            select_turbulence_model(&good_hybrid(src))
                .unwrap_or_else(|e| panic!("{src}: {e}"))
                .des
                .expect("a hybrid")
        };
        let b = read(base);

        for (from, to, what) in [
            ("SpalartAllmarasDDES", "SpalartAllmarasDES", "model (branch)"),
            ("SpalartAllmarasDDES", "SpalartAllmarasIDDES", "model (branch)"),
            ("SpalartAllmarasDDES", "kOmegaSSTDDES", "model (background)"),
            ("CDES 0.65", "CDES 0.30", "CDES"),
            ("Cdt1 8", "Cdt1 2", "Cdt1"),
            ("Cw 0.15", "Cw 0.30", "Cw"),
            ("delta maxDeltaxyz", "delta IDDESDeltaSimple", "delta"),
        ] {
            let src = base.replacen(from, to, 1);
            assert_ne!(src, base, "the pair did not change one entry: {what}");
            // A background swap drops `CDES`, which the SST arm refuses.
            let src = if to == "kOmegaSSTDDES" {
                src.replacen("CDES 0.65; ", "", 1)
            } else {
                src
            };
            // `IDDESDeltaSimple` is only meaningful under IDDES, and the
            // branch must not change with it.
            let other = read(&src);
            assert_ne!(
                (b.branch, b.background, b.delta, b.coeffs),
                (other.branch, other.background, other.delta, other.coeffs),
                "`{what}` was read and thrown away: the two selections are \
                 identical (SPEC-LIT §13.4.1)"
            );
        }
    }

    /// The same for `RAS { variant ...; }` and the SA coefficients.
    #[test]
    fn the_sa_case_document_pairs_each_change_the_coefficients() {
        let base = "RAS { model SpalartAllmaras; variant noft2; Cb1 0.1355; Cv1 7.1; \
                    Cn1 16; }";
        let b = sa_coeffs(&case(base)).expect("the base case");
        for (from, to, what) in [
            ("variant noft2", "variant noft2-neg", "variant"),
            ("variant noft2", "variant ft2", "variant"),
            ("Cb1 0.1355", "Cb1 0.14", "Cb1"),
            ("Cv1 7.1", "Cv1 8.0", "Cv1"),
            ("Cn1 16", "Cn1 12", "Cn1"),
        ] {
            let src = base.replacen(from, to, 1);
            assert_ne!(src, base);
            let other = sa_coeffs(&case(&src)).unwrap_or_else(|e| panic!("{src}: {e}"));
            assert_ne!(
                b, other,
                "`{what}` was read and thrown away (SPEC-LIT §13.4.1)"
            );
        }
    }

    /// **SPEC-LIT §58.1's seventh instance: `div(phi,nuTilda)`,
    /// `solvers/nuTilda` and `relaxationFactors/equations/nuTilda` reach the
    /// `nu~` equation, and they did not before.**
    #[test]
    fn the_nutilda_dictionary_entries_reach_the_nutilda_equation() {
        use crate::io::case::dissipation_from_model;
        assert_eq!(dissipation_from_model("SpalartAllmaras"), Some("nuTilda"));
        assert_eq!(dissipation_from_model("SpalartAllmarasDDES"), Some("nuTilda"));
        // The SST-background hybrids contain "omega" and are caught by the
        // first arm, which is right: they transport k and omega.
        assert_eq!(dissipation_from_model("kOmegaSSTIDDES"), Some("omega"));
        // And nothing that already answered has moved.
        assert_eq!(dissipation_from_model("kEpsilon"), Some("epsilon"));
        assert_eq!(dissipation_from_model("realizableKE"), Some("epsilon"));
        assert_eq!(dissipation_from_model("kOmegaSST"), Some("omega"));
        assert_eq!(dissipation_from_model("laminar"), None);
    }

    #[test]
    fn the_name_selects_the_model() {
        assert_eq!(
            select_turbulence_model(&case("simulationType RAS; RAS { model kEpsilon; }"))
                .ok()
                .map(|s| s.model),
            Some(RasModel::KEpsilon)
        );
        assert_eq!(
            select_turbulence_model(&case("RAS { RASModel kOmega; }"))
                .ok()
                .map(|s| s.model),
            Some(RasModel::KOmega)
        );
    }

    /// The whole point of the registry. These used to run standard k-epsilon
    /// or standard k-omega, whichever binary was invoked, and say nothing.
    ///
    /// `kOmegaSST` was the headline example here and has been removed from the
    /// list because it is now implemented -
    /// `k_omega_sst_now_selects_a_model` is what took its place, and the two
    /// must not both be true. `realizableKE` and `RNGkEpsilon` left the list
    /// the same way, for `the_two_k_epsilon_variants_now_select_a_model`.
    /// So did `SpalartAllmaras`, for `spalart_allmaras_now_selects_a_model`.
    /// And so did `kOmegaSSTLM`, for
    /// `the_transition_model_now_selects_a_model`; what stands in its place
    /// here is its own successor, `kOmegaSSTGamma`.
    #[test]
    fn an_unimplemented_model_errors_and_names_the_alternatives() {
        for name in ["kOmegaSSTGamma", "LRR", "v2f"] {
            let e = select_turbulence_model(&case(&format!("RAS {{ model {name}; }}")))
                .expect_err("must not silently substitute");
            let s = e.to_string();
            assert!(s.contains(name), "{s}");
            assert!(s.contains("kEpsilon"), "{s}");
            assert!(s.contains("kOmega"), "{s}");
            // The menu grew when the two variants landed; a refusal that
            // still printed the old menu would send a user looking for a
            // model this solver now has.
            assert!(s.contains("realizableKE"), "{s}");
            assert!(s.contains("RNGkEpsilon"), "{s}");
        }
    }

    /// SPEC-LIT §40 and §41: the two names that used to be
    /// RECOGNISED-AND-REFUSED now select real models, and the refusal list
    /// and the menu moved together.
    #[test]
    fn the_two_k_epsilon_variants_now_select_a_model() {
        for (name, want) in [
            ("realizableKE", RasModel::RealizableKE),
            ("RealizableKE", RasModel::RealizableKE),
            ("RNGkEpsilon", RasModel::RNGkEpsilon),
            ("RNGKEpsilon", RasModel::RNGkEpsilon),
        ] {
            let s = select_turbulence_model(&case(&format!("RAS {{ model {name}; }}")))
                .expect("the variant is implemented");
            assert_eq!(s.model, want);
            assert_eq!(s.model.dissipation_field(), Some("epsilon"));
            assert!(s.active);
        }

        // Both halves of the §13.4 bookkeeping, checked rather than assumed:
        // out of the refusal list, into the menu.
        for gone in ["realizableKE", "RNGkEpsilon", "SpalartAllmaras"] {
            assert!(RECOGNISED_NOT_IMPLEMENTED.iter().all(|(n, _)| *n != gone));
            assert!(available_models().contains(&gone));
        }

        // And every name still refused must be one the registry cannot build,
        // or the menu is lying in the other direction.
        for (name, _) in RECOGNISED_NOT_IMPLEMENTED {
            assert!(
                REGISTRY.iter().all(|(n, _)| n != name),
                "{name} is in both the refusal list and the registry"
            );
        }
        for name in available_models() {
            assert!(
                REGISTRY.iter().any(|(n, _)| *n == name),
                "the menu offers {name}, which the registry cannot build"
            );
        }
    }

    /// SPEC-LIT §40.6 and §41.4 - **the test that stops the sixth instance.**
    ///
    /// `model_coeff` returns the fallback for an absent key and says nothing
    /// about a present key nobody asks for. So `RAS { model realizableKE; C1
    /// 1.44; }` would parse, run, and use `max(0.43, eta/(eta+5))` anyway.
    /// Each of these is refused by name, with the reason and the alternative.
    #[test]
    fn a_coefficient_the_named_model_does_not_read_is_refused_by_name() {
        let _g = crate::io::contract::permissive_test_guard();
        crate::io::contract::set_permissive(false);
        crate::io::contract::reset_warnings();

        for (model, key, must_mention) in [
            ("realizableKE", "C1", "40.5"),
            ("realizableKE", "C3", "40.5"),
            ("RNGkEpsilon", "sigmak", "alphak"),
            ("RNGkEpsilon", "sigmaEps", "alphaEps"),
            ("RNGkEpsilon", "A0", "realizableKE"),
        ] {
            let cc = case(&format!("RAS {{ model {model}; {key} 1.5; }}"));
            let e = match model {
                "realizableKE" => realizable_ke_coeffs(&cc).err(),
                _ => rng_ke_coeffs(&cc).err(),
            }
            .unwrap_or_else(|| panic!("{model}: `{key}` must not be read and discarded"));
            let m = e.to_string();
            assert!(m.contains(key), "{m}");
            assert!(m.contains(must_mention), "{m}");
        }

        // The keys each model DOES read must not be refused, or the check is
        // simply an allow-nothing list.
        assert!(realizable_ke_coeffs(&case(
            "RAS { model realizableKE; A0 4.0; C2 1.9; sigmak 1.0; sigmaEps 1.2; Cmu 0.09; }"
        ))
        .is_ok());
        assert!(rng_ke_coeffs(&case(
            "RAS { model RNGkEpsilon; Cmu 0.0845; C1 1.42; C2 1.68; alphak 1.39; alphaEps 1.39; eta0 4.38; beta 0.012; C3 0; }"
        ))
        .is_ok());
    }

    /// SPEC-LIT §40.6: the coefficients a case writes must REACH the struct.
    #[test]
    fn the_variant_coefficients_are_read_from_the_case() {
        let c = realizable_ke_coeffs(&case(
            "RAS { model realizableKE; A0 4.0; C2 1.85; sigmaEps 1.25; }",
        ))
        .expect("reads");
        assert_eq!(c.a0, 4.0);
        assert_eq!(c.c2, 1.85);
        assert_eq!(c.sigma_eps, 1.25);
        // The default is the DERIVED value, not the NASA TM's printed 4.0 -
        // SPEC-LIT §40.3.
        assert_eq!(
            realizable_ke_coeffs(&case("RAS { model realizableKE; }")).expect("reads").a0,
            4.04
        );

        let r = rng_ke_coeffs(&case(
            "RAS { model RNGkEpsilon; alphaEps 1.2; eta0 4.0; beta 0.02; }",
        ))
        .expect("reads");
        assert_eq!(r.alpha_eps, 1.2);
        assert_eq!(r.eta0, 4.0);
        assert_eq!(r.beta, 0.02);
        assert_eq!(r.alpha_k, 1.39, "alphak must keep its own default");

        // And the <model>Coeffs sub-dictionary route, which is the other
        // place OpenFOAM cases put these.
        let c = realizable_ke_coeffs(&case(
            "RAS { model realizableKE; realizableKECoeffs { A0 4.2; } }",
        ))
        .expect("reads");
        assert_eq!(c.a0, 4.2);
    }

    /// SPEC-LIT §40.5: a buoyant `realizableKE` case is refused by name, and
    /// the refusal names `RNGkEpsilon` - which DOES have the term.
    #[test]
    fn a_buoyant_realizable_case_is_refused_and_names_a_model_that_has_gb() {
        let _g = crate::io::contract::permissive_test_guard();
        crate::io::contract::set_permissive(false);
        crate::io::contract::reset_warnings();

        // `CaseControls::default()` carries `BuoyancyCoeffs::default()`, which
        // is Earth gravity - a case with no `constant/g` gets a zeroed one
        // from `read_case_controls`, so that is what "no gravity" means here.
        let mut cc = case("RAS { model realizableKE; }");
        cc.buoyancy.g = Vec3::ZERO;
        assert!(refuse_realizable_ke_buoyancy(&cc).is_ok(), "no gravity, no refusal");

        cc.buoyancy.g = Vec3::new(0.0, 0.0, -9.81);
        assert!(cc.buoyancy.is_active(), "the knob must reach the controls");
        let e = refuse_realizable_ke_buoyancy(&cc)
            .expect_err("a buoyant realizableKE run must be refused");
        let m = e.to_string();
        assert!(m.contains("realizableKE"), "{m}");
        assert!(m.contains("RNGkEpsilon"), "{m}");
        assert!(m.contains("40.5"), "{m}");

        // RNG in the same case is fine.
        assert!(rng_ke_coeffs(&cc).is_ok());
    }

    #[test]
    fn a_typo_errors_and_names_the_setting() {
        let e = select_turbulence_model(&case("RAS { model kepsilon; }"))
            .expect_err("a typo must not run a model");
        let s = e.to_string();
        assert!(s.contains("kepsilon"), "{s}");
        assert!(s.contains("not a model name"), "{s}");
    }

    #[test]
    fn turbulence_off_and_laminar_both_stop_the_model() {
        let s = select_turbulence_model(&case(
            "simulationType RAS; RAS { model kEpsilon; turbulence off; }",
        ))
        .expect("selects");
        assert_eq!(s.model, RasModel::KEpsilon);
        assert!(!s.active, "`turbulence off` must stop the model");

        let s = select_turbulence_model(&case("simulationType laminar; RAS { model kEpsilon; }"))
            .expect("selects");
        assert!(!s.active, "`simulationType laminar` must stop the model");
        assert_eq!(s.model, RasModel::Laminar);

        // The default is on, or a case that says nothing would run laminar.
        let s = select_turbulence_model(&case("RAS { model kEpsilon; }")).expect("selects");
        assert!(s.active);
    }

    #[test]
    fn an_empty_case_is_laminar_not_an_error() {
        let s = select_turbulence_model(&case("")).expect("an empty case is laminar");
        assert_eq!(s.model, RasModel::Laminar);
        assert!(!s.active);
    }

    /// `simulationType LES;` with no `LES { model ...; }` names nothing, and
    /// guessing one would be the silent substitution §13.4 forbids.
    #[test]
    fn les_without_a_model_is_refused() {
        let e = select_turbulence_model(&case("simulationType LES;"))
            .expect_err("an LES case must name a subgrid model");
        let m = e.to_string();
        assert!(m.contains("LES/model"), "{m}");
        assert!(m.contains("Smagorinsky"), "{m}");
        assert!(m.contains("WALE"), "{m}");
    }

    #[test]
    fn the_les_name_selects_the_subgrid_model() {
        for (name, want) in [
            ("Smagorinsky", LesModel::Smagorinsky),
            ("WALE", LesModel::Wale),
            ("Deardorff", LesModel::Deardorff),
        ] {
            let s = select_turbulence_model(&case(&format!(
                "simulationType LES; LES {{ model {name}; }}"
            )))
            .expect("selects");
            assert_eq!(s.model, RasModel::Les);
            assert!(s.active);
            assert_eq!(s.les.map(|l| l.model), Some(want));
        }
    }

    /// The whole point of the `Les` variant: a RANS-only driver must not be
    /// able to read an LES case as laminar.
    #[test]
    fn an_les_case_is_not_a_laminar_case() {
        let s = select_turbulence_model(&case("simulationType LES; LES { model WALE; }"))
            .expect("selects");
        assert_ne!(s.model, RasModel::Laminar);
        assert_ne!(s.model, RasModel::KEpsilon);
        assert_eq!(s.model.name(), "LES");
        assert_eq!(s.model.dissipation_field(), None);
    }

    #[test]
    fn an_unimplemented_subgrid_model_errors_and_names_the_alternatives() {
        for name in ["kEqn", "dynamicKEqn", "Vreman"] {
            let e = select_turbulence_model(&case(&format!(
                "simulationType LES; LES {{ model {name}; }}"
            )))
            .expect_err("must not silently substitute");
            let m = e.to_string();
            assert!(m.contains(name), "{m}");
            assert!(m.contains("WALE"), "{m}");
        }
    }

    /// An LES model written under the RANS heading is a one-line fix, and the
    /// message has to say which line.
    #[test]
    fn an_les_model_under_ras_says_so() {
        let e = select_turbulence_model(&case("simulationType RAS; RAS { model WALE; }"))
            .expect_err("WALE is not a RAS model");
        let m = e.to_string();
        assert!(m.contains("simulationType LES"), "{m}");
    }

    /// SPEC-LIT §30.2: `simulationType LES;` beside a leftover
    /// `RAS { model ...; }` block is an ambiguity, not a preference -
    /// neither dictionary says which one the run author meant, and reading
    /// the `simulationType` alone (which is all this registry did before)
    /// silently drops the other one.
    #[test]
    fn les_with_a_leftover_ras_block_is_a_conflict_error() {
        let _g = crate::io::contract::permissive_test_guard();
        crate::io::contract::set_permissive(false);
        crate::io::contract::reset_warnings();

        let e = select_turbulence_model(&case(
            "simulationType LES; LES { model WALE; } RAS { model kEpsilon; }",
        ))
        .expect_err("LES with a RAS block present must not silently pick LES");
        let m = e.to_string();
        assert!(m.contains("kEpsilon"), "{m}");
        assert!(m.contains("LES"), "{m}");
    }

    /// The mirror image: `simulationType RAS;` beside a leftover
    /// `LES { model ...; }` block.
    #[test]
    fn ras_with_a_leftover_les_block_is_a_conflict_error() {
        let _g = crate::io::contract::permissive_test_guard();
        crate::io::contract::set_permissive(false);
        crate::io::contract::reset_warnings();

        let e = select_turbulence_model(&case(
            "simulationType RAS; RAS { model kEpsilon; } LES { model WALE; }",
        ))
        .expect_err("RAS with an LES block present must not silently pick RAS");
        let m = e.to_string();
        assert!(m.contains("WALE"), "{m}");
        assert!(m.contains("RAS"), "{m}");
    }

    /// `-permissive` substitutes the branch the case's own `simulationType`
    /// named, and says so - it does not average the two dictionaries or pick
    /// a third thing.
    #[test]
    fn permissive_ignores_the_other_blocks_dictionary() {
        let _g = crate::io::contract::permissive_test_guard();
        crate::io::contract::set_permissive(true);
        crate::io::contract::reset_warnings();

        let s = select_turbulence_model(&case(
            "simulationType RAS; RAS { model kOmega; } LES { model WALE; }",
        ))
        .expect("-permissive continues past the conflict");
        assert_eq!(s.model, RasModel::KOmega);

        crate::io::contract::set_permissive(false);
    }

    /// An empty `RAS { turbulence off; }` beside `simulationType LES;` is not
    /// this ambiguity: nothing is named there, so there is nothing the LES
    /// answer could have silently overridden.
    #[test]
    fn an_empty_ras_block_beside_les_is_not_a_conflict() {
        let s = select_turbulence_model(&case(
            "simulationType LES; LES { model WALE; } RAS { turbulence off; }",
        ))
        .expect("an empty RAS block names no model, so this is not the conflict");
        assert_eq!(s.model, RasModel::Les);
    }

    #[test]
    fn k_omega_sst_now_selects_a_model() {
        for name in ["kOmegaSST", "KOmegaSST"] {
            let s = select_turbulence_model(&case(&format!("RAS {{ model {name}; }}")))
                .expect("kOmegaSST is implemented");
            assert_eq!(s.model, RasModel::KOmegaSST);
            assert_eq!(s.model.dissipation_field(), Some("omega"));
            assert!(s.active);
        }
    }

    // ------------------------------------------------------------------
    //  §16 - the filter width
    // ------------------------------------------------------------------

    #[test]
    fn the_default_filter_width_is_the_cube_root_of_the_volume() {
        let s = select_turbulence_model(&case("simulationType LES; LES { model WALE; }"))
            .expect("selects");
        let d = s.les.expect("an LES selection").delta;
        assert_eq!(d.base, BaseDelta::CubeRootVol);
        assert!(!d.anisotropy);
        assert!(!d.van_driest);
        assert!(d.smooth.is_none());
    }

    #[test]
    fn every_filter_width_is_reachable() {
        let d = |src: &str| {
            select_turbulence_model(&case(src))
                .expect("selects")
                .les
                .expect("an LES selection")
                .delta
        };

        assert_eq!(
            d("simulationType LES; LES { model WALE; delta maxDeltaxyz; }").base,
            BaseDelta::MaxEdge
        );
        assert!(d("simulationType LES; LES { model WALE; delta Scotti; }").anisotropy);

        let vd = d("simulationType LES; LES { model Smagorinsky; delta vanDriest;                    vanDriestCoeffs { delta maxDeltaxyz; Aplus 30; Cdelta 0.2; } }");
        assert!(vd.van_driest);
        assert_eq!(vd.base, BaseDelta::MaxEdge, "the wrapped base was lost");
        assert!((vd.a_plus - 30.0).abs() < 1e-12);
        assert!((vd.c_delta - 0.2).abs() < 1e-12);

        let sm = d("simulationType LES; LES { model WALE; delta smooth;                    smoothCoeffs { delta cubeRootVol; maxDeltaRatio 1.05; sweeps 4; } }");
        let sm = sm.smooth.expect("smoothing");
        assert!((sm.max_ratio - 1.05).abs() < 1e-12);
        assert_eq!(sm.sweeps, 4);

        // And the two wrappers compose, which is the only reason the parser
        // is recursive.
        let both = d("simulationType LES; LES { model Smagorinsky; delta smooth;                      smoothCoeffs { delta vanDriest; }                      vanDriestCoeffs { delta maxDeltaxyz; } }");
        assert!(both.van_driest);
        assert!(both.smooth.is_some());
        assert_eq!(both.base, BaseDelta::MaxEdge);
    }

    /// A mis-spelled filter width changes `nu_t` in every cell and is
    /// invisible in the log, so it is an error rather than a fallback.
    #[test]
    fn a_wrong_filter_width_errors_and_names_the_menu() {
        for name in ["cubeRootVolDelta", "PrandtlDelta", "banana"] {
            let e = select_turbulence_model(&case(&format!(
                "simulationType LES; LES {{ model WALE; delta {name}; }}"
            )))
            .expect_err("must not substitute a filter width");
            let m = e.to_string();
            assert!(m.contains(name), "{m}");
            assert!(m.contains("cubeRootVol"), "{m}");
        }
    }

    /// A self-referential `delta` must terminate rather than exhaust the
    /// stack.
    #[test]
    fn a_recursive_filter_width_terminates() {
        let e = select_turbulence_model(&case(
            "simulationType LES; LES { model WALE; delta vanDriest;              vanDriestCoeffs { delta vanDriest; } }",
        ));
        assert!(e.is_err(), "a self-referential delta must be refused");
    }

    #[test]
    fn les_coefficients_are_read_from_either_place() {
        let s = select_turbulence_model(&case(
            "simulationType LES; LES { model Smagorinsky; SmagorinskyCoeffs { Cs 0.12; } Ck 0.07; }",
        ))
        .expect("selects");
        let l = s.les.expect("an LES selection");
        assert!((l.coeffs.cs - 0.12).abs() < 1e-12);
        assert!((l.coeffs.cd - 0.07).abs() < 1e-12, "Ck was not read");
    }

    #[test]
    fn les_turbulence_off_freezes_the_closure() {
        let s = select_turbulence_model(&case(
            "simulationType LES; LES { model WALE; turbulence off; }",
        ))
        .expect("selects");
        assert_eq!(s.model, RasModel::Les);
        assert!(!s.active);
    }

    /// `simulationType DES;` used to be a hard refusal. It is now a
    /// capability (SPEC-LIT §57, §58.2) - but one that still refuses a case
    /// naming no model, with the hybrid menu rather than the LES one.
    #[test]
    fn simulation_type_des_now_reaches_the_hybrids_and_still_needs_a_model() {
        let e = select_turbulence_model(&case("simulationType DES;"))
            .expect_err("a hybrid with no model named is still an error");
        let m = e.to_string();
        assert!(m.contains("missing"), "{m}");
        assert!(m.contains("SpalartAllmarasDDES"), "the hybrid menu is missing: {m}");

        // And with a model it selects one.
        let s = select_turbulence_model(&good_hybrid(
            "simulationType DES; DES { model SpalartAllmarasDES; }",
        ))
        .expect("simulationType DES now reaches the hybrid family");
        assert_eq!(s.model, RasModel::HybridSa);
        assert_eq!(s.des.expect("a hybrid").branch, DesBranch::Des97);
    }

    // ------------------------------------------------------------------
    //  §30.2 - build_coupled
    // ------------------------------------------------------------------

    fn gpu() -> Option<crate::device::Gpu> {
        crate::device::Gpu::new(0).ok()
    }

    /// A closed box with real walls - see `k_omega_sst.rs::tests::quiet_box`
    /// for why `nz = 1` (an `empty` mesh SST's cross-diffusion term can tell
    /// from an unclosed one).
    fn wall_box() -> crate::mesh::HostMesh {
        let (mut m, points, faces) = crate::mesh::topology::tests::box_mesh(
            [4, 4, 1],
            crate::Vec3::new(0.25, 0.25, 0.25),
        );
        m.compute_geometry(&points, &faces).expect("geometry");
        m.build_cell_face_maps();
        m
    }

    /// SPEC-LIT §30.2's whole point: the case's own `RAS { model ...; }` (or
    /// `simulationType laminar;`) must reach the CONCRETE model
    /// `build_coupled` allocates, not just the string `select_turbulence_model`
    /// returns. Before this function existed, `ofgpu-buoyant`/`ofgpu-lowmach`
    /// built `KEpsilon` regardless of what this loop iterates over.
    #[test]
    fn build_coupled_constructs_the_model_the_case_names() -> Result<()> {
        let Some(gpu) = gpu() else {
            return Ok(());
        };

        let hm = wall_box();
        let mesh = crate::mesh::GpuMesh::upload(&gpu, &hm)?;
        let no_walls = crate::field_setup::WallFaces::none(hm.n_boundary_faces);
        let no_roughness = crate::field_setup::NutRoughness::none(hm.n_boundary_faces);

        for (src, want_name) in [
            ("", "laminar"),
            ("simulationType laminar; RAS { model kEpsilon; }", "laminar"),
            ("RAS { model kEpsilon; }", "kEpsilon"),
            ("RAS { model kOmega; }", "kOmega"),
            ("RAS { model kOmegaSST; }", "kOmegaSST"),
        ] {
            let cc = case(src);
            let selection = select_turbulence_model(&cc)?;
            let turb =
                build_coupled(&gpu, &hm, &mesh, &cc, &selection, &no_walls, &no_roughness)?;
            assert_eq!(turb.name(), want_name, "case {src:?}");
            assert_eq!(turb.nut().f.len(), hm.n_cells);
            assert!(
                !turb.output_fields().is_empty(),
                "{want_name}: output_fields must name at least nut"
            );
        }

        Ok(())
    }

    /// SPEC-LIT §30.2's LES requirement, landed: the coupled solvers must
    /// actually BUILD the LES closure a case asks for, not the k-epsilon
    /// `build_coupled` used to hand back nothing to and then refuse outright.
    /// One test per submodel, since each is a different code path through
    /// `Les::new`/`CoupledLes`.
    #[test]
    fn build_coupled_constructs_the_les_model_the_case_names() -> Result<()> {
        let Some(gpu) = gpu() else {
            return Ok(());
        };

        let hm = wall_box();
        let mesh = crate::mesh::GpuMesh::upload(&gpu, &hm)?;
        let no_walls = crate::field_setup::WallFaces::none(hm.n_boundary_faces);
        let no_roughness = crate::field_setup::NutRoughness::none(hm.n_boundary_faces);

        for (src, want_name) in [
            ("simulationType LES; LES { model Smagorinsky; }", "Smagorinsky"),
            ("simulationType LES; LES { model WALE; }", "WALE"),
            ("simulationType LES; LES { model Deardorff; }", "Deardorff"),
            // §16.4/§30.2: van Driest damping needs the wall distance -
            // exercises the branch in `build_coupled` that runs the Poisson
            // solve, not just the `NO_WALL` sentinel path the other three
            // take.
            (
                "simulationType LES; LES { model Smagorinsky; delta vanDriest; }",
                "Smagorinsky",
            ),
        ] {
            let cc = case(src);
            let selection = select_turbulence_model(&cc)?;
            let turb = build_coupled(&gpu, &hm, &mesh, &cc, &selection, &no_walls, &no_roughness)?;
            assert_eq!(turb.name(), want_name, "case {src:?}");
            assert_eq!(turb.nut().f.len(), hm.n_cells);
            assert!(
                !turb.output_fields().is_empty(),
                "{want_name}: output_fields must name at least nut"
            );
        }

        Ok(())
    }

    /// SPEC-LIT §56/§57: `build_coupled` really builds Spalart-Allmaras and
    /// both hybrid backgrounds - the whole point of turning a refusal into a
    /// capability.
    #[test]
    fn build_coupled_constructs_spalart_allmaras_and_both_hybrids() -> Result<()> {
        let Some(gpu) = gpu() else {
            return Ok(());
        };
        // Three-dimensional AND free of `empty` patches, because SPEC-LIT
        // §57.10's guard 3 refuses a 2-D hybrid and reads the patch KINDS
        // rather than the cell count - `box_mesh` marks front and back
        // `empty` whatever `nz` is. That the first two drafts of this test
        // tripped over its own guard is the guard working.
        let hm = {
            let mut spec = crate::blockgen::BlockSpec {
                x: crate::blockgen::GradedAxis { n: 4, ..Default::default() },
                y: crate::blockgen::GradedAxis { n: 4, ..Default::default() },
                z: crate::blockgen::GradedAxis { n: 4, ..Default::default() },
                ..Default::default()
            };
            spec.patch_type[4] = "patch".to_string();
            spec.patch_type[5] = "patch".to_string();
            crate::blockgen::build_mesh(&spec)?
        };
        let mesh = crate::mesh::GpuMesh::upload(&gpu, &hm)?;
        let no_walls = crate::field_setup::WallFaces::none(hm.n_boundary_faces);
        let no_roughness = crate::field_setup::NutRoughness::none(hm.n_boundary_faces);

        // Plain RANS SA. Gravity is zeroed because `CaseControls::default()`
        // carries OpenFOAM's own `g` and SPEC-LIT §56.8 refuses a buoyant SA
        // run by name - which is why `ofgpu-sa` exists and is the driver
        // `common::driver_for` points at.
        let mut cc = case("RAS { model SpalartAllmaras; }");
        cc.buoyancy.g = Vec3::ZERO;
        let sel = select_turbulence_model(&cc)?;
        let turb = build_coupled(&gpu, &hm, &mesh, &cc, &sel, &no_walls, &no_roughness)?;
        assert_eq!(turb.name(), "SpalartAllmaras");
        let names: Vec<&str> = turb.output_fields().iter().map(|(n, _)| *n).collect();
        assert!(names.contains(&"nuTilda") && names.contains(&"nut"), "{names:?}");
        // A one-equation model carries no mixing time scale, and this says
        // so rather than inventing one - SPEC-LIT §13.4.
        assert!(matches!(
            turb.mixing_rate(),
            crate::models::coupled::MixingRate::None
        ));

        // Both hybrid backgrounds, all three branches.
        for name in available_hybrid_models() {
            let mut cc = good_hybrid(&format!(
                "simulationType LES; LES {{ model {name}; }}"
            ));
            cc.buoyancy.g = Vec3::ZERO;
            let sel = select_turbulence_model(&cc)?;
            let turb = build_coupled(&gpu, &hm, &mesh, &cc, &sel, &no_walls, &no_roughness)
                .unwrap_or_else(|e| panic!("{name}: {e}"));
            assert!(turb.name().contains("hybrid"), "{name}: {}", turb.name());
            assert_eq!(turb.nut().f.len(), hm.n_cells);
        }
        Ok(())
    }

    /// **SPEC-LIT §57.10's guard 3, which needs a mesh: a 2-D hybrid is
    /// refused by name.**
    ///
    /// The LES branch of a hybrid is a three-dimensional turbulence model; in
    /// two dimensions there is no vortex stretching and nothing to resolve.
    /// The run would converge, plot and mean nothing.
    #[test]
    fn a_two_dimensional_hybrid_is_refused_by_name() -> Result<()> {
        let Some(gpu) = gpu() else {
            return Ok(());
        };
        // `blockgen`'s default block has `empty` front and back - a 2-D mesh.
        let hm = crate::blockgen::build_mesh(&crate::blockgen::BlockSpec {
            x: crate::blockgen::GradedAxis { n: 4, ..Default::default() },
            y: crate::blockgen::GradedAxis { n: 4, ..Default::default() },
            z: crate::blockgen::GradedAxis { n: 1, ..Default::default() },
            ..Default::default()
        })?;
        assert!(
            hm.b_kind
                .iter()
                .any(|&k| k == crate::mesh::PatchKind::Empty as Label),
            "the test mesh is not 2-D"
        );
        let mesh = crate::mesh::GpuMesh::upload(&gpu, &hm)?;
        let no_walls = crate::field_setup::WallFaces::none(hm.n_boundary_faces);
        let no_roughness = crate::field_setup::NutRoughness::none(hm.n_boundary_faces);

        let cc = good_hybrid("simulationType LES; LES { model SpalartAllmarasDDES; }");
        let sel = select_turbulence_model(&cc)?;
        let t = match build_coupled(&gpu, &hm, &mesh, &cc, &sel, &no_walls, &no_roughness) {
            Ok(_) => panic!("a 2-D hybrid must be refused"),
            Err(e) => e.to_string(),
        };
        assert!(t.contains("2-D"), "{t}");
        assert!(t.contains("SpalartAllmarasDDES"), "{t}");
        assert!(t.contains("vortex stretching"), "{t}");
        Ok(())
    }

    /// SPEC-LIT §56.8: a buoyant `SpalartAllmaras` case is refused by name,
    /// exactly as §40.5 refuses a buoyant `realizableKE`.
    #[test]
    fn a_buoyant_spalart_allmaras_case_is_refused_by_name() {
        let mut cc = case("RAS { model SpalartAllmaras; }");
        cc.buoyancy.g = Vec3::new(0.0, -9.81, 0.0);
        assert!(cc.buoyancy.is_active(), "the test case has no gravity");
        let e = refuse_sa_buoyancy(&cc).expect_err("a buoyant SA run must be refused");
        let t = e.to_string();
        assert!(t.contains("SpalartAllmaras"), "{t}");
        assert!(t.contains("kEpsilon"), "the alternatives are missing: {t}");
        assert!(t.contains("not an energy"), "the reason is missing: {t}");
        // And with no gravity it passes - `CaseControls::default()` carries
        // OpenFOAM's own `g`, so the negative half has to zero it explicitly
        // or the test would be vacuous in the other direction.
        let mut quiet = case("RAS { model SpalartAllmaras; }");
        quiet.buoyancy.g = Vec3::ZERO;
        assert!(!quiet.buoyancy.is_active());
        assert!(refuse_sa_buoyancy(&quiet).is_ok());
    }

    /// SPEC-LIT §30.3: Deardorff, built through the SAME registry path a
    /// buoyant case uses, must run NaN-free next to a k-epsilon build on the
    /// identical mesh and flow - with mean `nut` reported, honestly, rather
    /// than asserted equal or better (SPEC-LIT §30.3 asks only that it
    /// differ from the RAS number, which a genuinely different model
    /// trivially does).
    ///
    /// The velocity field stands in for a small rising plume's own shear -
    /// upward in the centre, sheared at the edges - the resolved-momentum
    /// signature buoyancy leaves behind once the momentum equation itself has
    /// carried the body force (SPEC-LIT §30.2's own point: an algebraic LES
    /// has no `G_b` term of its own, so this is exactly how buoyancy is
    /// supposed to reach it).
    #[test]
    fn deardorff_via_build_coupled_runs_a_small_plume_nan_free_and_reports_nut_against_kepsilon()
    -> Result<()> {
        let Some(gpu) = gpu() else {
            return Ok(());
        };

        let hm = wall_box();
        let mesh = crate::mesh::GpuMesh::upload(&gpu, &hm)?;
        let no_walls = crate::field_setup::WallFaces::none(hm.n_boundary_faces);
        let no_roughness = crate::field_setup::NutRoughness::none(hm.n_boundary_faces);

        // A plume-like vertical (y) shear: fastest at the centre, at rest at
        // the ymin/ymax walls - nonzero grad U everywhere, which is the only
        // thing any of these models reads.
        let ny = 4usize;
        let mut u = crate::field::GpuVectorField::zeros(&gpu, &mesh, "U")?;
        let u_vals: Vec<crate::Vec3> = (0..hm.n_cells)
            .map(|c| {
                let j = (c / 4) % ny; // wall_box is a 4x4x1 box - see its own doc
                let center = (ny as Scalar - 1.0) / 2.0;
                let r = 1.0 - ((j as Scalar - center).abs() / (center + 1.0));
                crate::Vec3::new(0.0, 2.0 * r.max(0.0), 0.0)
            })
            .collect();
        gpu.write(&mut u.f, &u_vals)?;
        let phi = crate::field::GpuSurfaceScalarField::zeros(&gpu, &mesh, "phi")?;
        let flow = crate::turbulence::FlowState::new(&u, &phi, 1e-3);

        let mean_nut = |src: &str| -> Result<(String, Scalar)> {
            let cc = case(src);
            let selection = select_turbulence_model(&cc)?;
            let mut turb =
                build_coupled(&gpu, &hm, &mesh, &cc, &selection, &no_walls, &no_roughness)?;
            for (name, f) in turb.output_fields_mut() {
                if name == "k" {
                    gpu.write(&mut f.f, &vec![1.0 as Scalar; hm.n_cells])?;
                }
            }
            turb.initialise(&gpu, &flow)?;
            for _ in 0..10 {
                turb.correct(&gpu, &flow, None)?;
            }
            gpu.sync()?;
            let nut = gpu.download(&turb.nut().f)?;
            assert!(
                nut.iter().all(|v| v.is_finite()),
                "{}: nu_t has a non-finite value",
                turb.name()
            );
            let mean = nut.iter().sum::<Scalar>() / nut.len().max(1) as Scalar;
            Ok((turb.name().to_string(), mean))
        };

        let (ke_name, ke_mean) = mean_nut("RAS { model kEpsilon; }")?;
        let (les_name, les_mean) =
            mean_nut("simulationType LES; LES { model Deardorff; }")?;

        println!(
            "SPEC-LIT 30.3: mean nut - {ke_name} {ke_mean:e}, {les_name} {les_mean:e} \
             (ratio {les_name}/{ke_name} = {:e})",
            les_mean / ke_mean.max(1e-300)
        );

        assert!(ke_mean.is_finite() && ke_mean >= 0.0);
        assert!(les_mean.is_finite() && les_mean >= 0.0);

        Ok(())
    }

    /// SPEC-LIT §30.3: `kOmegaSST` built through the registry must actually
    /// BE SST - measured the way the batch's own gate is worded, by running
    /// it next to a k-epsilon build on the identical case and mesh and
    /// checking `nut` differs. A same-named field that happened to match
    /// would mean the dispatch silently fell back to k-epsilon regardless of
    /// what the case asked for, which is the exact failure this whole file
    /// exists to close.
    #[test]
    fn komega_sst_via_build_coupled_is_not_bit_identical_to_kepsilon() -> Result<()> {
        let Some(gpu) = gpu() else {
            return Ok(());
        };

        let hm = wall_box();
        let mesh = crate::mesh::GpuMesh::upload(&gpu, &hm)?;
        let no_walls = crate::field_setup::WallFaces::none(hm.n_boundary_faces);
        let no_roughness = crate::field_setup::NutRoughness::none(hm.n_boundary_faces);

        // A non-trivial flow, so the models actually have something to
        // differ ON - `nu_t = 0` from both would agree for the wrong reason.
        let mut u = crate::field::GpuVectorField::zeros(&gpu, &mesh, "U")?;
        let u_vals = vec![crate::Vec3::new(1.0, 0.3, 0.0); hm.n_cells];
        gpu.write(&mut u.f, &u_vals)?;
        let phi = crate::field::GpuSurfaceScalarField::zeros(&gpu, &mesh, "phi")?;
        let flow = crate::turbulence::FlowState::new(&u, &phi, 1e-3);

        let k0 = 1.0 as Scalar;

        let ke_nut = {
            let cc = case("RAS { model kEpsilon; }");
            let selection = select_turbulence_model(&cc)?;
            let mut turb =
                build_coupled(&gpu, &hm, &mesh, &cc, &selection, &no_walls, &no_roughness)?;
            for (name, f) in turb.output_fields_mut() {
                if name == "k" {
                    gpu.write(&mut f.f, &vec![k0; hm.n_cells])?;
                }
            }
            turb.initialise(&gpu, &flow)?;
            for _ in 0..20 {
                turb.correct(&gpu, &flow, None)?;
            }
            gpu.sync()?;
            gpu.download(&turb.nut().f)?
        };

        let sst_nut = {
            let cc = case("RAS { model kOmegaSST; }");
            let selection = select_turbulence_model(&cc)?;
            let mut turb =
                build_coupled(&gpu, &hm, &mesh, &cc, &selection, &no_walls, &no_roughness)?;
            for (name, f) in turb.output_fields_mut() {
                if name == "k" {
                    gpu.write(&mut f.f, &vec![k0; hm.n_cells])?;
                }
            }
            turb.initialise(&gpu, &flow)?;
            for _ in 0..20 {
                turb.correct(&gpu, &flow, None)?;
            }
            gpu.sync()?;
            gpu.download(&turb.nut().f)?
        };

        assert_eq!(ke_nut.len(), sst_nut.len());
        let max_diff = ke_nut
            .iter()
            .zip(sst_nut.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0 as Scalar, Scalar::max);
        let scale = ke_nut.iter().cloned().fold(0.0 as Scalar, Scalar::max).max(1e-30);
        assert!(
            max_diff > 1e-6 * scale,
            "kOmegaSST's nut is bit-identical to kEpsilon's (max diff {max_diff}); the \
             registry did not actually build a different model"
        );

        Ok(())
    }
}
