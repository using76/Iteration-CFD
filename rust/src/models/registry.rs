// meteor-cfd - Copyright (c) 2026 주식회사 메테오시뮬레이션 (Meteo Simulation Co., Ltd.)
// Source-available, not Open Source. Educational use is free; research,
// publication and commercial use require a licence - simul@msimul.com
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

use crate::error::Result;
use crate::io::case::CaseControls;
use crate::io::contract::unsupported;
use crate::io::dict::FoamDict;
use crate::les::{BaseDelta, DeltaSpec, SmoothSpec};
use crate::models::les::{LesCoeffs, LesModel};
use crate::Scalar;

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
        "RAS" | "RASModel" => {}
        "LES" => return select_les(d),
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
}
