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
use crate::io::contract::unsupported;
use crate::io::dict::FoamDict;
use crate::les::{BaseDelta, DeltaSpec, SmoothSpec};
use crate::mesh::{GpuMesh, HostMesh};
use crate::models::coupled::{
    BuoyancySettings, CoupledKEpsilon, CoupledKOmega, CoupledKOmegaSst, CoupledLaminar,
    CoupledLes, CoupledTurbulence,
};
use crate::models::les::{Les, LesCoeffs, LesModel};
use crate::models::{KEpsilon, KEpsilonCoeffs, KOmega, KOmegaCoeffs, KOmegaSst, KOmegaSstCoeffs};
use crate::turbulence::C3Mode;
use crate::{Scalar, Vec3};

/// A model ofgpu implements.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RasModel {
    /// No closure at all: `nu_t = 0` everywhere, and the momentum equation
    /// sees the molecular viscosity alone.
    Laminar,
    /// Standard k-epsilon - SPEC-LIT §6.1, Launder & Spalding (1974).
    KEpsilon,
    /// Wilcox k-omega, the 1988 form - SPEC-LIT §6.2.
    KOmega,
    /// Menter k-omega SST, the 2003 revision - SPEC-LIT §6.3. Needs the wall
    /// distance of §6.6, which a driver must compute before constructing it.
    KOmegaSST,
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
            Self::KOmega => "kOmega",
            Self::KOmegaSST => "kOmegaSST",
            Self::Les => "LES",
        }
    }

    /// The dissipation variable's field name, which is also the `0/` file a
    /// driver has to find.
    pub fn dissipation_field(self) -> Option<&'static str> {
        match self {
            Self::Laminar => None,
            Self::KEpsilon => Some("epsilon"),
            Self::KOmega => Some("omega"),
            Self::KOmegaSST => Some("omega"),
            // An algebraic subgrid model solves for nothing, so there is no
            // `0/` file to find. Deardorff reports a `k_sgs`, but it is an
            // estimate the model makes rather than a field it transports.
            Self::Les => None,
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
    ("kOmega", RasModel::KOmega),
    ("KOmega", RasModel::KOmega),
    ("kOmegaSST", RasModel::KOmegaSST),
    ("KOmegaSST", RasModel::KOmegaSST),
];

/// Names this solver RECOGNISES but does not implement.
///
/// SPEC-LIT §13.4 distinguishes these from a name nobody has heard of, and the
/// distinction is worth keeping: telling a user that `kOmegaSST` is a real
/// model ofgpu has not got is a different message from telling them
/// `kepsilon` is not a model at all, and the second one is a typo they can fix
/// in five seconds once they are told.
const RECOGNISED_NOT_IMPLEMENTED: &[&str] = &[
    "kOmegaSSTLM",
    "kOmegaSSTSAS",
    "SpalartAllmaras",
    "realizableKE",
    "RNGkEpsilon",
    "LaunderSharmaKE",
    "kEpsilonPhitF",
    "v2f",
    "LRR",
    "SSG",
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
    "SpalartAllmarasDES",
    "SpalartAllmarasDDES",
    "SpalartAllmarasIDDES",
    "kOmegaSSTDES",
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
    "IDDESDelta",
    "maxDeltaxyzCubeRootLES",
    "cubeRootVolDelta",
];

/// The menu a rejected name is shown.
pub fn available_models() -> Vec<&'static str> {
    let mut v: Vec<&'static str> = vec!["kEpsilon", "kOmega", "kOmegaSST", "laminar"];
    v.dedup();
    v
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
            return select_les(d);
        }
        // A detached-eddy hybrid is a RANS model and an LES model with a
        // switch between them, and the switch is the model. Refused by name
        // rather than run as either half.
        "DES" | "DDES" | "IDDES" => {
            return unsupported(
                "momentumTransport/simulationType",
                &sim,
                &["RAS", "LES", "laminar"],
                "laminar (nu_t = 0)",
                TurbulenceSelection::laminar(),
            );
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
            } else if RECOGNISED_NOT_IMPLEMENTED.contains(&name) {
                "a published model ofgpu has not got"
            } else {
                "not a model name ofgpu knows"
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

    Ok(TurbulenceSelection {
        model,
        active,
        les: None,
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
    })
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
            let hint = if DELTA_RECOGNISED_NOT_IMPLEMENTED.contains(&other) {
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

/// Build the turbulence closure a coupled solver (`ofgpu-buoyant`,
/// `ofgpu-fire`) drives, from the case's own `constant/momentumTransport` -
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
    /// must not both be true.
    #[test]
    fn an_unimplemented_model_errors_and_names_the_alternatives() {
        for name in ["kOmegaSSTLM", "SpalartAllmaras", "realizableKE"] {
            let e = select_turbulence_model(&case(&format!("RAS {{ model {name}; }}")))
                .expect_err("must not silently substitute");
            let s = e.to_string();
            assert!(s.contains(name), "{s}");
            assert!(s.contains("kEpsilon"), "{s}");
            assert!(s.contains("kOmega"), "{s}");
        }
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

    #[test]
    fn des_is_still_refused_and_names_les() {
        let e = select_turbulence_model(&case("simulationType DES;"))
            .expect_err("DES is not implemented");
        let m = e.to_string();
        assert!(m.contains("DES"), "{m}");
        assert!(m.contains("LES"), "{m}");
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
    /// returns. Before this function existed, `ofgpu-buoyant`/`ofgpu-fire`
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
