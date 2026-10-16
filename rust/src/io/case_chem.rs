// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.
// Provenance: see PROVENANCE.md. No GPL-licensed source was consulted.

//! The `species`, `chemistry` and `combustion` case blocks - JSONC and the
//! three dictionaries `constant/thermophysicalProperties`,
//! `constant/chemistryProperties` and `constant/combustionProperties` -
//! lowered onto one [`LoweredChem`] (docs/17 CHR-12a, SPEC-LIT §137.2 and
//! §137.7).
//!
//! Both routes fill the same three `Json*` blocks: the dictionary route turns
//! each dictionary into the JSON object the JSONC route reads and hands it to
//! the same `deny_unknown_fields` deserialiser, so an unknown key is refused
//! by name on both, and [`lower_blocks`] is the one lowering. What needs the
//! mechanism and thermo files themselves (a name outside the thermo set, the
//! set against the mechanism, EDM's global-mechanism rule, and §129's own
//! refusals naming the line) is [`LoweredChem::load`], which reads them.
//!
//! Only `ofgpu-lowmach` runs a chemistry case (CHR-12b, SPEC-LIT §133.11),
//! and only from a JSONC case: every other driver, and every driver on the
//! dictionary route, refuses one by name ([`refuse_chemistry_case`]), the
//! CHR-25 parcels seam's shape.
//!
//! Written from:
//!   ofgpu SPEC-LIT.md §137.2 - the three tables and their refusals
//!   ofgpu SPEC-LIT.md §137.7 - the as-built rules of this file
//!   ofgpu SPEC-LIT.md §134.1-§134.4 - the closures' keys and the
//!     combinations refused against `MixingRate` (`src/models/coupled.rs`)
//!   ofgpu SPEC-LIT.md §128, §129, §132.8, §133.4, §19 - the thermo set, the
//!     CHEMKIN reader, the reactor's defaults, the Lewis route, `Sc_t`
//!
//! Provenance: ORIGINAL. No external source. No GPL-licensed source was
//! consulted.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::chem::mechanism::Mechanism;
use crate::chem::reactor::ReactorOptions;
use crate::chem::rosenbrock::Method;
use crate::chem::thermo::{ElementTable, ThermoSet, P_ATM};
use crate::error::{Error, Result};
use crate::io::case_json::{
    patch_class, resolve_patch_rule, JsonPatchRule, LoweredScalarField, PatchPresetKind, ScalarBc,
    TurbulenceKind,
};
use crate::io::dict::FoamDict;
use crate::io::fields::PatchFieldSpec;
use crate::io::regex::Regex;
use crate::models::coupled::MixingRate;
use crate::models::registry::RasModel;
use crate::Scalar;

/// The dictionary route's files inside a case directory.
pub const SPECIES_FILE: &str = "constant/thermophysicalProperties";
pub const CHEMISTRY_FILE: &str = "constant/chemistryProperties";
pub const COMBUSTION_FILE: &str = "constant/combustionProperties";
/// The foam-format reaction list, which this reader does not read: the
/// mechanism is a CHEMKIN file (§129), named by `chemistry.mechanism`.
pub const REACTIONS_FILE: &str = "constant/reactions";

const TAG: &str = "(SPEC-LIT §137.2, §137.7)";

// ==========================================================================
//  The JSON blocks
// ==========================================================================

/// The `species` block (SPEC-LIT §137.2).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct JsonSpecies {
    /// Every species, the inert one included.
    pub names: Vec<String>,
    /// The species closed by `1 - sum` (SPEC-LIT §19). No default.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub inert: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub transport: Option<JsonSpeciesTransport>,
    /// name -> the uniform initial mass fraction of every solved species.
    /// The inert one's is `1 - sum` and is not stated.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub initial: BTreeMap<String, f64>,
    /// Ordered, first-match-wins, as `patches[]`.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub patches: Vec<JsonSpeciesPatch>,
}

/// `species.transport`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct JsonSpeciesTransport {
    /// `lewis` (the default). `mixtureAveraged` is refused (§133.4).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    /// name -> Lewis number; 1 for every species not named.
    #[serde(rename = "Le", default, skip_serializing_if = "BTreeMap::is_empty")]
    pub le: BTreeMap<String, f64>,
    /// The turbulent Schmidt number; SPEC-LIT §19's 0.7 when absent.
    #[serde(rename = "Sct", default, skip_serializing_if = "Option::is_none")]
    pub sct: Option<f64>,
}

/// One `species.patches[]` rule: the mass-fraction condition of each named
/// species on every boundary patch `match` matches.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct JsonSpeciesPatch {
    #[serde(rename = "match")]
    pub pattern: String,
    #[serde(rename = "Y", default)]
    pub y: BTreeMap<String, ScalarBc>,
}

/// The `chemistry` block (SPEC-LIT §137.2).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct JsonChemistry {
    /// The CHEMKIN mechanism, relative to the case.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mechanism: Option<String>,
    /// The CHEMKIN thermo file, relative to the case.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub thermo: Option<String>,
    /// A TRAN file. Refused: the Lewis route reads none (§137.2).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub transport: Option<String>,
    /// `rodas3` (the default) | `ros3`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub integrator: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rtol: Option<f64>,
    /// The absolute tolerance of every mass fraction.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub atol: Option<f64>,
    /// `lie` (the default) | `strang`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub splitting: Option<String>,
}

/// A word the case may write as a string or as a number (`variant 1989`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(untagged)]
pub enum Word {
    Text(String),
    Number(f64),
}

impl Word {
    fn text(&self) -> String {
        match self {
            Word::Text(s) => s.clone(),
            Word::Number(x) => format!("{x}"),
        }
    }
}

/// The `combustion` block (SPEC-LIT §137.2; the closures are §134).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct JsonCombustion {
    /// `laminar` (the default) | `EDM` | `EDC` | `PaSR` | `FSC` | `TFC`.
    /// `FSC` and `TFC` are the dust model's closures (§154); they lower in
    /// `crate::io::case_dust` and are refused here, as is every one of the
    /// four keys below under any of the four species-closure names.
    #[serde(alias = "combustionModel", default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    /// EDM's `A` (§134.2). No default.
    #[serde(rename = "A", default, skip_serializing_if = "Option::is_none")]
    pub a: Option<f64>,
    /// EDM's `B`; absent removes the product-limiting argument.
    #[serde(rename = "B", default, skip_serializing_if = "Option::is_none")]
    pub b: Option<f64>,
    /// EDC's `1989` (alias `1996`) | `2005` (§134.3). No default.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub variant: Option<Word>,
    #[serde(rename = "CD1", default, skip_serializing_if = "Option::is_none")]
    pub cd1: Option<f64>,
    #[serde(rename = "CD2", default, skip_serializing_if = "Option::is_none")]
    pub cd2: Option<f64>,
    #[serde(rename = "gammaMax", default, skip_serializing_if = "Option::is_none")]
    pub gamma_max: Option<f64>,
    #[serde(rename = "psrSpan", default, skip_serializing_if = "Option::is_none")]
    pub psr_span: Option<f64>,
    /// PaSR's mixing time by name (§134.4). No default.
    #[serde(rename = "tauMix", default, skip_serializing_if = "Option::is_none")]
    pub tau_mix: Option<String>,
    /// PaSR's chemical time by name (§134.4). No default.
    #[serde(rename = "tauChem", default, skip_serializing_if = "Option::is_none")]
    pub tau_chem: Option<String>,
    /// `C_mix` of `tauMix integral`. No default.
    #[serde(rename = "Cmix", default, skip_serializing_if = "Option::is_none")]
    pub c_mix: Option<f64>,
    /// FSC/TFC's Pr_t (§151.5: each case its own - Leeds 0.4, REMBE 0.28).
    /// REQUIRED with `model` FSC or TFC; there is no default.
    #[serde(rename = "PrT", default, skip_serializing_if = "Option::is_none")]
    pub pr_t: Option<f64>,
    /// The extended FSC's t_flacc (§154.7). FSC only - TFC has no clock
    /// (§154.6).
    #[serde(rename = "tFlacc", default, skip_serializing_if = "Option::is_none")]
    pub t_flacc: Option<f64>,
    /// The flame-Courant limit Co_f,max of (155.9); the default is
    /// `ofgpu::dust::courant::CO_F_MAX` = 0.5.
    #[serde(rename = "maxFlameCourant", default, skip_serializing_if = "Option::is_none")]
    pub max_flame_courant: Option<f64>,
    /// The ignition source of (155.4); absent is no ignition source.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ignition: Option<JsonIgnition>,
}

/// `combustion.ignition`, the source of (155.4) (SPEC-LIT §155.3). `sigmaT`
/// omitted is t0 (§151.7 item 8).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct JsonIgnition {
    /// The ignition point in world coordinates, m. Required.
    pub point: [f64; 3],
    /// sigma_r, m (H20 §3.3's 0.25 mm). Required, finite and above zero.
    #[serde(rename = "sigmaR")]
    pub sigma_r: f64,
    /// W_0, 1/s (H20 §3.3's 1e13). Required, finite and above zero.
    #[serde(rename = "W0")]
    pub w0: f64,
    /// t_0, s - the end of the ramp and the origin of t_fd. Required, finite
    /// and at or above zero.
    #[serde(rename = "t0")]
    pub t0: f64,
    /// sigma_t, s; omitted is t0 (§151.7 item 8).
    #[serde(rename = "sigmaT", default, skip_serializing_if = "Option::is_none")]
    pub sigma_t: Option<f64>,
}

// ==========================================================================
//  The lowered config
// ==========================================================================

/// The species set, lowered.
#[derive(Debug, Clone, PartialEq)]
pub struct LoweredSpecies {
    /// Every species, the inert one included, in the case's order.
    pub names: Vec<String>,
    pub inert: String,
    /// Lewis number per entry of [`Self::names`] (the Lewis route, §133.4).
    pub le: Vec<f64>,
    /// The turbulent Schmidt number.
    pub sct: f64,
    /// One field per SOLVED species, in [`Self::names`] order without the
    /// inert one. Empty on the dictionary route, whose fields are the case's
    /// `0/` files.
    pub fields: Vec<LoweredScalarField>,
}

/// `chemistry.splitting`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Splitting {
    Lie,
    Strang,
}

/// The chemistry, lowered. The two paths are as the case wrote them;
/// [`LoweredChem::load`] resolves them against the case's directory.
#[derive(Debug, Clone, PartialEq)]
pub struct LoweredChemistry {
    pub mechanism: PathBuf,
    pub thermo: PathBuf,
    /// The reactor's controls: `integrator`, `rtol` and `atol` (as `atol_y`)
    /// over [`ReactorOptions::default`] (§132.8).
    pub reactor: ReactorOptions,
    pub splitting: Splitting,
}

/// EDC's two variants (§134.3).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EdcVariant {
    V1989,
    V2005,
}

/// PaSR's mixing times (§134.4).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TauMix {
    Integral,
    Kolmogorov,
    Geometric,
    Strain,
}

/// PaSR's chemical times (§134.4).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TauChem {
    Sfr,
    Ffr,
    Gfr,
}

/// The combustion closure, lowered (§134). CHR-14 and CHR-15 build the
/// three closures; until then the driver seam refuses every one.
#[derive(Debug, Clone, PartialEq)]
pub enum LoweredCombustion {
    Laminar,
    Edm { a: f64, b: Option<f64> },
    Edc { variant: EdcVariant, cd1: f64, cd2: f64, gamma_max: f64, psr_span: f64 },
    Pasr { tau_mix: TauMix, tau_chem: TauChem, c_mix: Option<f64> },
}

/// Everything [`lower_blocks`] produces.
#[derive(Debug, Clone, PartialEq)]
pub struct LoweredChem {
    pub species: LoweredSpecies,
    pub chemistry: LoweredChemistry,
    pub combustion: LoweredCombustion,
}

/// [`MixingRate`]'s four arms, by name: what the case's turbulence model
/// gives a closure (§134.1).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MixingKind {
    None,
    Epsilon,
    Omega,
    Strain,
}

impl MixingKind {
    /// The arm a running model returns.
    pub fn of(rate: &MixingRate<'_>) -> Self {
        match rate {
            MixingRate::Epsilon { .. } => Self::Epsilon,
            MixingRate::Omega { .. } => Self::Omega,
            MixingRate::Strain(_) => Self::Strain,
            MixingRate::None => Self::None,
        }
    }

    /// The arm a case's turbulence selection will give, by the model's name
    /// and [`RasModel::dissipation_field`]. `None` when the name is not a
    /// model this crate builds (the model's own constructor refuses it).
    pub fn of_model(kind: Option<TurbulenceKind>, model: Option<&str>) -> Option<Self> {
        if kind == Some(TurbulenceKind::Les) {
            return Some(Self::Strain);
        }
        let Some(name) = model else { return Some(Self::None) };
        const MODELS: [RasModel; 10] = [
            RasModel::Laminar,
            RasModel::KEpsilon,
            RasModel::LaunderSharmaKE,
            RasModel::RealizableKE,
            RasModel::RNGkEpsilon,
            RasModel::KOmega,
            RasModel::KOmegaSST,
            RasModel::KOmegaSstLM,
            RasModel::KOmegaSstGamma,
            RasModel::SpalartAllmaras,
        ];
        let m = MODELS.iter().find(|m| m.name() == name)?;
        Some(match m.dissipation_field() {
            Some("epsilon") => Self::Epsilon,
            Some(_) => Self::Omega,
            None => Self::None,
        })
    }
}

// ==========================================================================
//  Refusals
// ==========================================================================

/// The refusal a driver gives a case that states species, chemistry or
/// combustion when it does not run them (SPEC-LIT §137.7, §133.11). `source`
/// names where.
#[must_use]
pub fn refuse_chemistry_case(source: &str) -> Error {
    Error::Refused(format!(
        "{source}: this case states species, chemistry or combustion, which this driver does \
         not run; ofgpu-lowmach runs a reacting JSONC case (CHR-12b, SPEC-LIT §133.11), and no driver runs \
         one from a case directory yet, so it is refused rather than run without its chemistry \
         {TAG}"
    ))
}

fn refused(key: &str, msg: &str) -> Error {
    Error::Refused(format!("{key}: {msg} {TAG}"))
}

fn keyed(key: &str, e: Error) -> Error {
    match e {
        Error::Refused(m) => Error::Refused(format!("{key}: {m}")),
        Error::Config(m) => Error::Config(format!("{key}: {m}")),
        other => other,
    }
}

/// A finite number above zero, or the refusal naming `key`.
fn positive(key: &str, v: f64) -> Result<f64> {
    if v.is_finite() && v > 0.0 {
        Ok(v)
    } else {
        Err(refused(key, &format!("{v} is not a finite number above zero")))
    }
}

/// A mass fraction: finite and in `[0, 1]`.
fn fraction(key: &str, v: f64) -> Result<f64> {
    if v.is_finite() && (0.0..=1.0).contains(&v) {
        Ok(v)
    } else {
        Err(refused(key, &format!("{v} is not a mass fraction in [0, 1]")))
    }
}

// ==========================================================================
//  The one lowering
// ==========================================================================

/// The three blocks as a case states them; each `None` when absent.
#[derive(Debug, Clone, Copy, Default)]
pub struct ChemBlocks<'a> {
    pub species: Option<&'a JsonSpecies>,
    pub chemistry: Option<&'a JsonChemistry>,
    pub combustion: Option<&'a JsonCombustion>,
}

/// The one lowering of both routes (SPEC-LIT §137.2, §137.7). `Ok(None)`
/// when the case states none of the three blocks.
///
/// `mixing` is [`MixingKind::of_model`]'s answer for the case's turbulence
/// (`None`: a model name this crate does not build). `compressible` is
/// whether the case states `physics.compressible`. `boundary` is the JSONC
/// route's patch names and `patches[]` rules, from which each solved
/// species' field is built; `None` on the dictionary route, whose fields are
/// the case's `0/` files.
pub fn lower_blocks(
    blocks: ChemBlocks<'_>,
    mixing: Option<MixingKind>,
    compressible: bool,
    boundary: Option<(&[String], &[JsonPatchRule])>,
) -> Result<Option<LoweredChem>> {
    let ChemBlocks { species, chemistry, combustion } = blocks;
    let Some(sp) = species else {
        if chemistry.is_some() {
            return Err(refused("chemistry", "stated without a species block; the mechanism reacts the case's species"));
        }
        if combustion.is_some() {
            return Err(refused("combustion", "stated without a species block; a closure acts on the case's species"));
        }
        return Ok(None);
    };
    if compressible {
        return Err(refused(
            "species",
            "stated with physics.compressible; compressible reacting flow is out of scope \
             (docs/17 section 2), and ofgpu-lowmach carries the reacting case",
        ));
    }
    let species = lower_species(sp, boundary)?;
    let Some(ch) = chemistry else {
        return Err(refused(
            "chemistry",
            "absent; a species block needs chemistry.mechanism and chemistry.thermo",
        ));
    };
    let chemistry = lower_chemistry(ch)?;
    let combustion = match combustion {
        Some(c) => lower_combustion(c, mixing)?,
        None => LoweredCombustion::Laminar,
    };
    Ok(Some(LoweredChem { species, chemistry, combustion }))
}

fn lower_species(sp: &JsonSpecies, boundary: Option<(&[String], &[JsonPatchRule])>) -> Result<LoweredSpecies> {
    let names = &sp.names;
    if names.len() < 2 {
        return Err(refused(
            "species.names",
            &format!("{} named; with fewer than two the mass fractions are not a transported set (SPEC-LIT §19)", names.len()),
        ));
    }
    for (i, n) in names.iter().enumerate() {
        if n.is_empty() {
            return Err(refused("species.names", "an empty name"));
        }
        if names[..i].contains(n) {
            return Err(refused("species.names", &format!("\"{n}\" is named twice")));
        }
    }
    let Some(inert) = sp.inert.clone() else {
        return Err(refused(
            "species.inert",
            "absent; the case names the species closed by 1 - sum (there is no default)",
        ));
    };
    if !names.contains(&inert) {
        return Err(refused("species.inert", &format!("\"{inert}\" is not one of species.names {names:?}")));
    }
    let named = |key: &str, n: &str| -> Result<()> {
        if names.iter().any(|m| m == n) {
            Ok(())
        } else {
            Err(refused(key, &format!("\"{n}\" is not one of species.names {names:?}")))
        }
    };

    // ---- transport: the Lewis route (§133.4) -----------------------------
    let mut le = vec![1.0f64; names.len()];
    let mut sct = f64::from(crate::species::SpeciesCoeffs::default().sc_t);
    if let Some(t) = &sp.transport {
        match t.model.as_deref() {
            None | Some("lewis") => {}
            Some("mixtureAveraged") => {
                return Err(refused(
                    "species.transport.model",
                    "\"mixtureAveraged\" needs the mixture diffusivities of §133.4's route (b), \
                     which wait for a read copy of Neufeld, Janzen & Aziz (1972); the recognised \
                     model is \"lewis\"",
                ))
            }
            Some(other) => {
                return Err(refused(
                    "species.transport.model",
                    &format!("\"{other}\": the recognised models are \"lewis, mixtureAveraged\""),
                ))
            }
        }
        for (n, v) in &t.le {
            let key = format!("species.transport.Le.{n}");
            named(&key, n)?;
            let i = names.iter().position(|m| m == n).unwrap_or(0);
            le[i] = positive(&key, *v)?;
        }
        if let Some(v) = t.sct {
            sct = positive("species.transport.Sct", v)?;
        }
    }

    // ---- fields: the JSONC route only ------------------------------------
    let mut fields = Vec::new();
    if let Some((patch_names, rules)) = boundary {
        for (n, v) in &sp.initial {
            let key = format!("species.initial.{n}");
            named(&key, n)?;
            if *n == inert {
                return Err(refused(
                    &key,
                    "the inert species' initial value is 1 - sum of the others, so a stated one \
                     would be read and ignored",
                ));
            }
            fraction(&key, *v)?;
        }
        let mut sum = 0.0f64;
        for n in names.iter().filter(|n| **n != inert) {
            match sp.initial.get(n) {
                Some(v) => sum += v,
                None => {
                    return Err(refused(
                        "species.initial",
                        &format!("\"{n}\" has no initial value"),
                    ))
                }
            }
        }
        if sum > 1.0 {
            return Err(refused(
                "species.initial",
                &format!("the solved species sum to {sum}, so the inert \"{inert}\" would be negative"),
            ));
        }
        let mut compiled = Vec::with_capacity(sp.patches.len());
        for (r, rule) in sp.patches.iter().enumerate() {
            for (n, bc) in &rule.y {
                let key = format!("species.patches[{r}].Y.{n}");
                named(&key, n)?;
                if *n == inert {
                    return Err(refused(
                        &key,
                        "the inert species is closed by 1 - sum on every face, so a stated \
                         condition would be read and ignored",
                    ));
                }
                match bc {
                    ScalarBc::FixedValue { value } => {
                        fraction(&key, *value)?;
                    }
                    ScalarBc::InletOutlet { inlet_value } => {
                        fraction(&key, *inlet_value)?;
                    }
                    ScalarBc::ZeroGradient {} => {}
                    _ => {
                        return Err(refused(
                            &key,
                            "a temperature condition; a mass fraction takes fixedValue, \
                             inletOutlet or zeroGradient",
                        ))
                    }
                }
            }
            let re = Regex::new(&rule.pattern)
                .map_err(|e| Error::Config(format!("species.patches[{r}].match: {e}")))?;
            compiled.push(re);
        }
        for n in names.iter().filter(|n| **n != inert) {
            let y0 = sp.initial[n];
            let mut boundary = BTreeMap::new();
            for pname in patch_names {
                let own = sp
                    .patches
                    .iter()
                    .zip(&compiled)
                    .find(|(_, re)| re.is_match(pname))
                    .and_then(|(rule, _)| rule.y.get(n));
                let kind = resolve_patch_rule(rules, pname)?.kind;
                boundary.insert(pname.clone(), species_spec(own, kind, y0, n, pname)?);
            }
            fields.push(LoweredScalarField {
                name: n.clone(),
                dimensions: "[0 0 0 0 0 0 0]".to_string(),
                internal_uniform: y0 as Scalar,
                boundary,
            });
        }
    } else if !sp.initial.is_empty() || !sp.patches.is_empty() {
        // Unreachable from the dictionary route's reader, which fills neither.
        return Err(refused(
            "species.initial",
            "the dictionary route's fields are the case's 0/ files",
        ));
    }
    Ok(LoweredSpecies { names: names.clone(), inert, le, sct, fields })
}

/// One solved species' condition on one patch: the first matching
/// `species.patches` rule's, else the patch kind's default.
fn species_spec(
    own: Option<&ScalarBc>,
    kind: PatchPresetKind,
    y0: f64,
    name: &str,
    patch: &str,
) -> Result<PatchFieldSpec> {
    let spec = |t: &str| PatchFieldSpec { type_name: t.to_string(), ..PatchFieldSpec::default() };
    if let Some(bc) = own {
        return Ok(match bc {
            ScalarBc::FixedValue { value } => {
                let mut s = spec("fixedValue");
                s.value = vec![*value as Scalar];
                s
            }
            ScalarBc::InletOutlet { inlet_value } => {
                let mut s = spec("inletOutlet");
                s.inlet_value = vec![*inlet_value as Scalar];
                s.value = vec![*inlet_value as Scalar];
                s
            }
            _ => spec("zeroGradient"),
        });
    }
    match kind {
        PatchPresetKind::Wall => Ok(spec("zeroGradient")),
        PatchPresetKind::Open => {
            let mut s = spec("inletOutlet");
            s.inlet_value = vec![y0 as Scalar];
            s.value = vec![y0 as Scalar];
            Ok(s)
        }
        PatchPresetKind::Inlet => Err(refused(
            "species.patches",
            &format!("patch \"{patch}\" is an inlet and no rule states \"{name}\" on it"),
        )),
        PatchPresetKind::Empty | PatchPresetKind::Symmetry | PatchPresetKind::Wedge => {
            Ok(spec(patch_class(kind)))
        }
        PatchPresetKind::Farfield | PatchPresetKind::SupersonicInlet => Err(refused(
            "species.patches",
            &format!("patch \"{patch}\" is a compressible preset, and a species case is not compressible"),
        )),
    }
}

fn lower_chemistry(ch: &JsonChemistry) -> Result<LoweredChemistry> {
    let Some(mechanism) = &ch.mechanism else {
        return Err(refused("chemistry.mechanism", "absent; it is required with a species block"));
    };
    let Some(thermo) = &ch.thermo else {
        return Err(refused("chemistry.thermo", "absent; it is required with a species block"));
    };
    if ch.transport.is_some() {
        return Err(refused(
            "chemistry.transport",
            "a TRAN file with species.transport.model lewis (the one model built) would be \
             read and ignored",
        ));
    }
    let d = ReactorOptions::default();
    let method = match &ch.integrator {
        Some(n) => Method::from_name(n).map_err(|e| keyed("chemistry.integrator", e))?,
        None => d.method,
    };
    let rtol = ch.rtol.map(|v| positive("chemistry.rtol", v)).transpose()?.unwrap_or(d.rtol);
    let atol_y = ch.atol.map(|v| positive("chemistry.atol", v)).transpose()?.unwrap_or(d.atol_y);
    let splitting = match ch.splitting.as_deref() {
        None | Some("lie") => Splitting::Lie,
        Some("strang") => Splitting::Strang,
        Some(other) => {
            return Err(refused(
                "chemistry.splitting",
                &format!("\"{other}\": the recognised splittings are \"lie, strang\""),
            ))
        }
    };
    Ok(LoweredChemistry {
        mechanism: PathBuf::from(mechanism),
        thermo: PathBuf::from(thermo),
        reactor: ReactorOptions { method, rtol, atol_y, ..d },
        splitting,
    })
}

fn lower_combustion(c: &JsonCombustion, mixing: Option<MixingKind>) -> Result<LoweredCombustion> {
    let model = c.model.as_deref().unwrap_or("laminar");
    // §13.4.1: a key of a closure the case did not select is refused. The
    // FSC/TFC keys lower in `crate::io::case_dust`; under laminar/EDM/EDC/
    // PaSR they are refused here by the same table.
    let keys: [(&str, bool, &str); 14] = [
        ("A", c.a.is_some(), "EDM"),
        ("B", c.b.is_some(), "EDM"),
        ("variant", c.variant.is_some(), "EDC"),
        ("CD1", c.cd1.is_some(), "EDC"),
        ("CD2", c.cd2.is_some(), "EDC"),
        ("gammaMax", c.gamma_max.is_some(), "EDC"),
        ("psrSpan", c.psr_span.is_some(), "EDC"),
        ("tauMix", c.tau_mix.is_some(), "PaSR"),
        ("tauChem", c.tau_chem.is_some(), "PaSR"),
        ("Cmix", c.c_mix.is_some(), "PaSR"),
        ("PrT", c.pr_t.is_some(), "FSC/TFC"),
        ("tFlacc", c.t_flacc.is_some(), "FSC/TFC"),
        ("maxFlameCourant", c.max_flame_courant.is_some(), "FSC/TFC"),
        ("ignition", c.ignition.is_some(), "FSC/TFC"),
    ];
    if !["laminar", "EDM", "EDC", "PaSR"].contains(&model) {
        return Err(refused(
            "combustion.model",
            &format!("\"{model}\": the recognised models are \"laminar, EDM, EDC, PaSR\""),
        ));
    }
    for (key, set, owner) in keys {
        if set && owner != model {
            return Err(refused(
                &format!("combustion.{key}"),
                &format!("a setting of {owner}, and combustion.model is {model}, so it would be read and ignored"),
            ));
        }
    }
    if model == "laminar" {
        return Ok(LoweredCombustion::Laminar);
    }
    let combination = |what: &str| refused("combustion.model", &format!("{model} with {what} (SPEC-LIT §134.1)"));
    match mixing {
        None => {
            return Err(combination(
                "a turbulence model this crate does not build, whose mixing rate is unknown",
            ))
        }
        Some(MixingKind::None) => {
            return Err(combination(
                "a turbulence model that gives no mixing time scale (laminar, Spalart-Allmaras: \
                 MixingRate::None)",
            ))
        }
        Some(MixingKind::Strain) if model != "PaSR" => {
            return Err(combination(
                "LES (MixingRate::Strain): EDM and EDC are RANS closures in every read source, \
                 and LES-EDC is named and not built",
            ))
        }
        Some(_) => {}
    }
    Ok(match model {
        "EDM" => {
            let Some(a) = c.a else {
                return Err(refused(
                    "combustion.A",
                    "absent; EDM's A has no default, because Magnussen & Hjertager (1977), \
                     which states it, was not read (SPEC-LIT §134.2)",
                ));
            };
            LoweredCombustion::Edm {
                a: positive("combustion.A", a)?,
                b: c.b.map(|b| positive("combustion.B", b)).transpose()?,
            }
        }
        "EDC" => {
            let variant = match c.variant.as_ref().map(Word::text).as_deref() {
                None => {
                    return Err(refused(
                        "combustion.variant",
                        "absent; EDC's variant (1989, alias 1996, or 2005) has no default (SPEC-LIT §134.3)",
                    ))
                }
                Some("1989") | Some("1996") => EdcVariant::V1989,
                Some("2005") => EdcVariant::V2005,
                Some("batch") => {
                    return Err(refused(
                        "combustion.variant",
                        "\"batch\": integrating the fine structures as a batch reactor is not \
                         (134.9)'s PSR, so it is refused (SPEC-LIT §134.3)",
                    ))
                }
                Some(other) => {
                    return Err(refused(
                        "combustion.variant",
                        &format!("\"{other}\": the recognised variants are \"1989, 1996, 2005\""),
                    ))
                }
            };
            let gamma_max = c.gamma_max.unwrap_or(0.95);
            if !(gamma_max > 0.0 && gamma_max < 1.0) {
                return Err(refused("combustion.gammaMax", &format!("{gamma_max} is outside (0, 1)")));
            }
            LoweredCombustion::Edc {
                variant,
                cd1: positive("combustion.CD1", c.cd1.unwrap_or(0.135))?,
                cd2: positive("combustion.CD2", c.cd2.unwrap_or(0.50))?,
                gamma_max,
                psr_span: positive("combustion.psrSpan", c.psr_span.unwrap_or(20.0))?,
            }
        }
        _ => {
            let tau_mix = match c.tau_mix.as_deref() {
                None => {
                    return Err(refused(
                        "combustion.tauMix",
                        "absent; PaSR's mixing time has no default (SPEC-LIT §134.4)",
                    ))
                }
                Some("integral") => TauMix::Integral,
                Some("kolmogorov") => TauMix::Kolmogorov,
                Some("geometric") => TauMix::Geometric,
                Some("strain") => TauMix::Strain,
                Some(other) => {
                    return Err(refused(
                        "combustion.tauMix",
                        &format!("\"{other}\": the recognised mixing times are \"integral, kolmogorov, geometric, strain\""),
                    ))
                }
            };
            let tau_chem = match c.tau_chem.as_deref() {
                None => {
                    return Err(refused(
                        "combustion.tauChem",
                        "absent; PaSR's chemical time has no default (SPEC-LIT §134.4)",
                    ))
                }
                Some("sfr") => TauChem::Sfr,
                Some("ffr") => TauChem::Ffr,
                Some("gfr") => TauChem::Gfr,
                Some(other) => {
                    return Err(refused(
                        "combustion.tauChem",
                        &format!("\"{other}\": the recognised chemical times are \"sfr, ffr, gfr\""),
                    ))
                }
            };
            let strain = mixing == Some(MixingKind::Strain);
            if (tau_mix == TauMix::Strain) != strain {
                return Err(refused(
                    "combustion.tauMix",
                    if strain {
                        "an epsilon-based mixing time under LES, which transports no epsilon; \
                         LES takes \"strain\" (SPEC-LIT §134.4)"
                    } else {
                        "\"strain\" reads LES's |S| (MixingRate::Strain), and this case's \
                         turbulence model is RANS (SPEC-LIT §134.4)"
                    },
                ));
            }
            let c_mix = match (tau_mix, c.c_mix) {
                (TauMix::Integral, None) => {
                    return Err(refused(
                        "combustion.Cmix",
                        "absent with tauMix integral; C_mix has no default, because the source \
                         says it must be tuned (SPEC-LIT §134.4)",
                    ))
                }
                (TauMix::Integral, Some(v)) => Some(positive("combustion.Cmix", v)?),
                (_, Some(_)) => {
                    return Err(refused(
                        "combustion.Cmix",
                        "only tauMix integral reads C_mix, so it would be read and ignored",
                    ))
                }
                (_, None) => None,
            };
            LoweredCombustion::Pasr { tau_mix, tau_chem, c_mix }
        }
    })
}

// ==========================================================================
//  The file-dependent refusals
// ==========================================================================

fn read_text(key: &str, p: &Path) -> Result<String> {
    std::fs::read_to_string(p)
        .map_err(|e| Error::Config(format!("{key}: cannot read {}: {e} {TAG}", p.display())))
}

impl LoweredChem {
    /// Read the mechanism and the thermo set (paths relative to `base`, the
    /// case's directory) and apply the refusals that need them: §129's,
    /// naming the line; a species name outside the thermo set (§128); a
    /// mechanism species outside `species.names` and a name outside the
    /// mechanism; EDM with a mechanism that is not one or two global
    /// reactions (§134.1).
    pub fn load(&self, base: &Path) -> Result<Mechanism> {
        let tp = base.join(&self.chemistry.thermo);
        let thermo = ThermoSet::read_chemkin(
            &read_text("chemistry.thermo", &tp)?,
            &tp.display().to_string(),
            P_ATM,
            &ElementTable::standard(),
        )
        .map_err(|e| keyed("chemistry.thermo", e))?;
        let mp = base.join(&self.chemistry.mechanism);
        let mech = Mechanism::read_chemkin(
            &read_text("chemistry.mechanism", &mp)?,
            &mp.display().to_string(),
            Some(&thermo),
        )
        .map_err(|e| keyed("chemistry.mechanism", e))?;
        for n in &self.species.names {
            if thermo.index_of(n).is_none() && mech.thermo().index_of(n).is_none() {
                return Err(refused(
                    "species.names",
                    &format!("\"{n}\" is not in the thermo set of {} (SPEC-LIT §128)", tp.display()),
                ));
            }
        }
        for n in mech.species() {
            if !self.species.names.contains(n) {
                return Err(refused(
                    "species.names",
                    &format!("the mechanism's species \"{n}\" is not one of species.names"),
                ));
            }
        }
        for n in &self.species.names {
            if mech.species_index(n).is_none() {
                return Err(refused(
                    "species.names",
                    &format!("\"{n}\" is not a species of the mechanism {}", mp.display()),
                ));
            }
        }
        if let LoweredCombustion::Edm { .. } = self.combustion {
            let r = mech.reactions();
            if r.len() > 2 {
                return Err(refused(
                    "combustion.model",
                    &format!(
                        "EDM with a mechanism of {} reactions; EDM takes one or two global \
                         reactions (SPEC-LIT §134.1, §134.2)",
                        r.len()
                    ),
                ));
            }
            if let Some(x) = r.iter().find(|x| {
                x.reversible || x.reverse.is_some() || x.third_body.is_some() || x.falloff.is_some()
            }) {
                return Err(refused(
                    "combustion.model",
                    &format!(
                        "EDM with the reaction \"{}\" (line {}), which is not global (a reverse \
                         reaction, a third body or a fall-off) (SPEC-LIT §134.1)",
                        x.equation, x.line
                    ),
                ));
            }
        }
        Ok(mech)
    }
}

// ==========================================================================
//  The dictionary route
// ==========================================================================

/// One dictionary entry as the JSON value the JSONC route would hold: `(a b
/// c)` a list (of numbers when every item is one, else of words), a number a
/// number, and anything else a word.
fn dict_value(raw: &str) -> serde_json::Value {
    use serde_json::Value;
    let t = raw.trim().trim_matches('"');
    if let Some(inner) = t.strip_prefix('(').and_then(|s| s.strip_suffix(')')) {
        let items: Vec<&str> = inner.split_whitespace().collect();
        let nums: Option<Vec<f64>> = items.iter().map(|s| s.parse::<f64>().ok()).collect();
        return match nums {
            Some(v) if !v.is_empty() => Value::Array(v.into_iter().map(|x| serde_json::json!(x)).collect()),
            _ => Value::Array(items.into_iter().map(|s| Value::String(s.trim_matches('"').to_string())).collect()),
        };
    }
    if let Ok(i) = t.parse::<i64>() {
        return serde_json::json!(i);
    }
    match t.parse::<f64>() {
        Ok(x) => serde_json::json!(x),
        Err(_) => Value::String(t.to_string()),
    }
}

/// The entries under `prefix` (`""` for the whole file, the header skipped)
/// as a nested JSON object.
fn dict_object(d: &FoamDict, prefix: &str, what: &str) -> Result<serde_json::Map<String, serde_json::Value>> {
    use serde_json::{Map, Value};
    let mut root = Map::new();
    for (key, raw) in &d.entries {
        let Some(rest) = key.strip_prefix(prefix) else { continue };
        if rest.is_empty() || rest == "FoamFile" || rest.starts_with("FoamFile/") {
            continue;
        }
        let parts: Vec<&str> = rest.split('/').filter(|s| !s.is_empty()).collect();
        let Some((leaf, dirs)) = parts.split_last() else { continue };
        let mut node = &mut root;
        for dir in dirs {
            let slot = node.entry((*dir).to_string()).or_insert_with(|| Value::Object(Map::new()));
            node = match slot {
                Value::Object(m) => m,
                _ => return Err(Error::Config(format!("{what}: {dir} is both an entry and a sub-dictionary {TAG}"))),
            };
        }
        if rest.ends_with('/') {
            node.entry((*leaf).to_string()).or_insert_with(|| Value::Object(Map::new()));
            continue;
        }
        node.insert((*leaf).to_string(), dict_value(raw));
    }
    Ok(root)
}

fn from_object<T: serde::de::DeserializeOwned>(m: serde_json::Map<String, serde_json::Value>, what: &str) -> Result<T> {
    serde_json::from_value(serde_json::Value::Object(m)).map_err(|e| Error::Refused(format!("{what}: {e} {TAG}")))
}

/// `constant/thermophysicalProperties`'s species entries as the `species`
/// block: `species` -> `names`, `inertSpecie` -> `inert`, and the
/// sub-dictionary `speciesTransport` -> `transport`. The file's other
/// entries belong to other readers and are not looked at. `None` when the
/// file names no `species`.
pub fn species_from_dict(d: &FoamDict, what: &str) -> Result<Option<JsonSpecies>> {
    use serde_json::{Map, Value};
    let Some(list) = d.get("species") else {
        for key in ["inertSpecie", "speciesTransport"] {
            if d.has(key) || d.entries.keys().any(|k| k.starts_with(&format!("{key}/"))) {
                return Err(refused(&format!("{what}: {key}"), "stated with no species list, so it would be read and ignored"));
            }
        }
        return Ok(None);
    };
    let mut m = Map::new();
    let names = match dict_value(list) {
        Value::Array(v) => v.into_iter().map(|x| Value::String(x.as_str().map_or_else(|| x.to_string(), str::to_string))).collect(),
        _ => return Err(refused(&format!("{what}: species"), "is not a list of names")),
    };
    m.insert("names".into(), Value::Array(names));
    if let Some(inert) = d.get("inertSpecie") {
        m.insert("inert".into(), Value::String(inert.trim().trim_matches('"').to_string()));
    }
    let tr = dict_object(d, "speciesTransport/", what)?;
    if !tr.is_empty() || d.has("speciesTransport/") {
        m.insert("transport".into(), Value::Object(tr));
    }
    from_object(m, what).map(Some)
}

/// A whole dictionary (the header skipped) as one block.
pub fn block_from_dict<T: serde::de::DeserializeOwned>(d: &FoamDict, what: &str) -> Result<T> {
    from_object(dict_object(d, "", what)?, what)
}

/// The dictionary route (SPEC-LIT §137.7): the three files of `case_dir`,
/// through [`lower_blocks`] with no `boundary` (the fields are the case's
/// `0/` files, and a solved species with no `0/<name>` is refused). A
/// `constant/reactions` file is refused, naming the CHEMKIN route.
pub fn read_case_chem(case_dir: &Path, mixing: Option<MixingKind>) -> Result<Option<LoweredChem>> {
    if case_dir.join(REACTIONS_FILE).exists() {
        return Err(refused(
            REACTIONS_FILE,
            "a foam-format reaction list, which this reader does not read; name a CHEMKIN \
             mechanism and thermo file in constant/chemistryProperties (mechanism, thermo)",
        ));
    }
    let read = |f: &str| -> Result<Option<FoamDict>> {
        let p = case_dir.join(f);
        if p.exists() {
            FoamDict::read(&p).map(Some)
        } else {
            Ok(None)
        }
    };
    let species = match read(SPECIES_FILE)? {
        Some(d) => species_from_dict(&d, SPECIES_FILE)?,
        None => None,
    };
    let chemistry: Option<JsonChemistry> =
        read(CHEMISTRY_FILE)?.map(|d| block_from_dict(&d, CHEMISTRY_FILE)).transpose()?;
    let combustion: Option<JsonCombustion> =
        read(COMBUSTION_FILE)?.map(|d| block_from_dict(&d, COMBUSTION_FILE)).transpose()?;
    let blocks = ChemBlocks {
        species: species.as_ref(),
        chemistry: chemistry.as_ref(),
        combustion: combustion.as_ref(),
    };
    let lowered = lower_blocks(blocks, mixing, false, None)?;
    if let Some(l) = &lowered {
        for n in l.species.names.iter().filter(|n| **n != l.species.inert) {
            if !case_dir.join("0").join(n).exists() {
                return Err(refused(
                    "species",
                    &format!("\"{n}\" has no initial value: 0/{n} is missing"),
                ));
            }
        }
    }
    Ok(lowered)
}

/// Whether a case directory states chemistry on the dictionary route: the
/// file that does, for [`refuse_chemistry_case`]'s message.
pub fn dict_route_states_chemistry(case_dir: &Path) -> Result<Option<&'static str>> {
    for f in [CHEMISTRY_FILE, COMBUSTION_FILE, REACTIONS_FILE] {
        if case_dir.join(f).exists() {
            return Ok(Some(f));
        }
    }
    // A thermophysicalProperties that does not parse is left to the readers
    // that use it (§122.6, §37.4): a driver that never read it must not
    // start failing on it here.
    let p = case_dir.join(SPECIES_FILE);
    if p.exists() && FoamDict::read(&p).is_ok_and(|d| d.has("species")) {
        return Ok(Some(SPECIES_FILE));
    }
    Ok(None)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::case_json::{read_case_jsonc_str, JsonCase, LoweredCase};

    /// One CHEMKIN THERMO record with `a1 = 3.5` and every other
    /// coefficient 0 in both ranges: a format-valid record, read as data.
    fn record(name: &str, els: &[(&str, u32)]) -> String {
        let mut c1 = format!("{name:<18}TEST  ");
        for (s, n) in els {
            c1.push_str(&format!("{s:<2}{n:>3}"));
        }
        while c1.len() < 44 {
            c1.push(' ');
        }
        c1.push_str(&format!("G{:>10.3}{:>10.3}{:>8.2}      1", 200.0, 6000.0, 1000.0));
        let a = |x: f64| format!("{x:>15.8E}");
        let z = a(0.0);
        let c2 = format!("{}{z}{z}{z}{z}    2", a(3.5));
        let c3 = format!("{z}{z}{}{z}{z}    3", a(3.5));
        let c4 = format!("{z}{z}{z}{z}                   4");
        format!("{c1}\n{c2}\n{c3}\n{c4}\n")
    }

    fn thermo_text(with_h2o: bool, with_ar: bool) -> String {
        let mut t = String::from("THERMO ALL\n   200.000  1000.000  6000.000\n");
        t.push_str(&record("N2", &[("N", 2)]));
        t.push_str(&record("O2", &[("O", 2)]));
        t.push_str(&record("H2", &[("H", 2)]));
        if with_h2o {
            t.push_str(&record("H2O", &[("H", 2), ("O", 1)]));
        }
        if with_ar {
            t.push_str(&record("AR", &[("AR", 1)]));
        }
        t.push_str("END\n");
        t
    }

    const GLOBAL: &str = "ELEMENTS H O N END\nSPECIES H2 O2 H2O N2 END\nREACTIONS\n\
        2H2 + O2 => 2H2O   1.0E+10 0.0 10000.0\nEND\n";
    const REVERSIBLE: &str = "ELEMENTS H O N END\nSPECIES H2 O2 H2O N2 END\nREACTIONS\n\
        2H2 + O2 <=> 2H2O   1.0E+10 0.0 10000.0\nEND\n";

    /// A temp case directory holding `files`.
    fn temp_case(tag: &str, files: &[(&str, &str)]) -> PathBuf {
        static N: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let n = N.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let d = std::env::temp_dir().join(format!("case_chem_{tag}_{}_{n}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        for (f, text) in files {
            let p = d.join(f);
            std::fs::create_dir_all(p.parent().unwrap()).unwrap();
            std::fs::write(p, text).unwrap();
        }
        d
    }

    /// Every key of the three blocks set to a value that is not its default.
    const EVERY: &str = r#"
        "species": {
            "names": ["H2", "O2", "H2O", "N2"], "inert": "N2",
            "transport": { "model": "lewis", "Le": { "H2": 0.3, "H2O": 0.8 }, "Sct": 0.9 },
            "initial": { "H2": 0.02, "O2": 0.22, "H2O": 0.01 },
            "patches": [
                { "match": "inlet", "Y": { "H2": { "type": "fixedValue", "value": 0.03 },
                    "O2": { "type": "fixedValue", "value": 0.23 },
                    "H2O": { "type": "zeroGradient" } } },
                { "match": "outlet", "Y": { "H2": { "type": "inletOutlet", "inletValue": 0.0 } } }
            ]
        },
        "chemistry": { "mechanism": "mech.inp", "thermo": "therm.dat", "integrator": "ros3",
            "rtol": 1e-6, "atol": 1e-12, "splitting": "strang" },
        "combustion": { "model": "laminar" }"#;

    fn case_text(blocks: &str) -> String {
        format!(
            r#"{{
                "name": "chemTest",
                "mesh": {{
                    "kind": "cartesian",
                    "bounds": {{ "min": [0,0,0], "max": [1,1,1] }},
                    "cells": [4, 2, 2],
                    "boundaries": {{
                        "xmin": "inlet", "xmax": "outlet", "ymin": "ya",
                        "ymax": "yb", "zmin": "za", "zmax": "zb"
                    }}
                }},
                "physics": {{
                    "gravity": [0,0,0],
                    "fluid": {{ "nu": 1e-5, "Pr": 0.71, "Prt": 0.85, "TRef": 293.15 }},
                    "buoyancy": "densityRatio"
                }},
                "patches": [
                    {{ "match": "inlet", "kind": "inlet", "U": {{ "type": "fixedValue", "value": [1,0,0] }} }},
                    {{ "match": "outlet", "kind": "open" }},
                    {{ "match": ".*", "kind": "wall" }}
                ],
                "initial": {{ "U": [0,0,0], "p": 0.0 }},
                "numerics": {{
                    "algorithm": {{ "kind": "SIMPLE" }},
                    "ddt": "steadyState",
                    "div": {{ "default": "Gauss upwind" }},
                    "grad": "Gauss linear",
                    "laplacian": {{ "snGrad": "corrected", "nonOrthogonalCorrectors": 0 }},
                    "solvers": []
                }},
                "run": {{ "endTime": 1.0, "deltaT": 1.0 }}{blocks}
            }}"#
        )
    }

    fn lower(blocks: &str) -> Result<LoweredCase> {
        read_case_jsonc_str(&case_text(blocks), "chemTest").and_then(|c| c.lower())
    }

    fn refuses<T: std::fmt::Debug>(r: Result<T>, key: &str) {
        match r {
            Err(Error::Refused(m)) => assert!(m.contains(key), "the refusal does not name {key}: {m}"),
            other => panic!("{key}: expected a refusal, got {other:?}"),
        }
    }

    /// The three blocks from JSON text (`co` may be empty), lowered with the
    /// JSONC route's boundary `inlet` (inlet), `outlet` (open), `w` (wall).
    fn lb(sp: &str, ch: &str, co: &str, mixing: Option<MixingKind>) -> Result<Option<LoweredChem>> {
        let sp: Option<JsonSpecies> = (!sp.is_empty()).then(|| serde_json::from_str(sp).unwrap());
        let ch: Option<JsonChemistry> = (!ch.is_empty()).then(|| serde_json::from_str(ch).unwrap());
        let co: Option<JsonCombustion> = (!co.is_empty()).then(|| serde_json::from_str(co).unwrap());
        let rules: Vec<JsonPatchRule> = serde_json::from_str(
            r#"[ { "match": "inlet", "kind": "inlet", "U": { "type": "fixedValue", "value": [1,0,0] } },
                 { "match": "outlet", "kind": "open" }, { "match": ".*", "kind": "wall" } ]"#,
        )
        .unwrap();
        let names: Vec<String> = ["inlet", "outlet", "w"].iter().map(|s| s.to_string()).collect();
        let blocks = ChemBlocks { species: sp.as_ref(), chemistry: ch.as_ref(), combustion: co.as_ref() };
        lower_blocks(blocks, mixing, false, Some((&names, &rules)))
    }

    const SP: &str = r#"{ "names": ["H2", "O2", "H2O", "N2"], "inert": "N2",
        "initial": { "H2": 0.02, "O2": 0.22, "H2O": 0.0 },
        "patches": [ { "match": "inlet", "Y": { "H2": { "type": "fixedValue", "value": 0.03 },
            "O2": { "type": "fixedValue", "value": 0.23 }, "H2O": { "type": "zeroGradient" } } } ] }"#;
    const CH: &str = r#"{ "mechanism": "mech.inp", "thermo": "therm.dat" }"#;

    /// SPEC-LIT §137.7 `case_chem_lowers`: every key of the three blocks
    /// set to a non-default value is read back; the dictionary route equals
    /// the JSONC route; a case with the blocks survives a serde round trip;
    /// a case without them lowers to no chemistry; `load` reads the files.
    #[test]
    fn case_chem_lowers() {
        let l = lower(&format!(",{EVERY}")).expect("the chemistry case lowers");
        l.refuse_chemistry().expect_err("every driver refuses a chemistry case before CHR-12b");
        let c = l.chem.clone().expect("the blocks lower");
        let s = &c.species;
        assert_eq!(s.names, ["H2", "O2", "H2O", "N2"]);
        assert_eq!(s.inert, "N2");
        assert_eq!(s.le, [0.3, 1.0, 0.8, 1.0]);
        assert_eq!(s.sct, 0.9);
        let names: Vec<&str> = s.fields.iter().map(|f| f.name.as_str()).collect();
        assert_eq!(names, ["H2", "O2", "H2O"], "one field per solved species, in order");
        let h2 = &s.fields[0];
        assert_eq!(h2.internal_uniform, 0.02 as Scalar);
        assert_eq!(h2.boundary["inlet"].type_name, "fixedValue");
        assert_eq!(h2.boundary["inlet"].value, vec![0.03 as Scalar]);
        assert_eq!(h2.boundary["outlet"].type_name, "inletOutlet");
        assert_eq!(h2.boundary["outlet"].inlet_value, vec![0.0 as Scalar]);
        let o2 = &s.fields[1];
        assert_eq!(o2.boundary["outlet"].type_name, "inletOutlet", "the open default");
        assert_eq!(o2.boundary["outlet"].inlet_value, vec![0.22 as Scalar]);
        for w in ["ya", "yb", "za", "zb"] {
            assert_eq!(o2.boundary[w].type_name, "zeroGradient", "the wall default");
        }
        assert_eq!(s.fields[2].boundary["inlet"].type_name, "zeroGradient");
        let ch = &c.chemistry;
        assert_eq!(ch.mechanism, PathBuf::from("mech.inp"));
        assert_eq!(ch.thermo, PathBuf::from("therm.dat"));
        let d = ReactorOptions::default();
        assert_eq!(ch.reactor, ReactorOptions { method: Method::Ros3, rtol: 1e-6, atol_y: 1e-12, ..d });
        assert_eq!(ch.splitting, Splitting::Strang);
        assert_eq!(c.combustion, LoweredCombustion::Laminar);

        // The parcels' vapour may now name a species.
        let parcels = r#", "parcels": { "capacity": 64, "physics": "evaporating",
            "coupling": { "mass": "evaporation", "vapour": "H2O" },
            "injectors": [ { "position": [0.5, 0.5, 0.5], "massFlow": 1e-6 } ] }"#;
        let lp = lower(&format!(",{EVERY}{parcels}")).expect("a vapour in species.names lowers");
        assert_eq!(lp.parcels.expect("parcels").vapour.as_deref(), Some("H2O"));

        // Defaults, and the closures' keys.
        let m = lb(SP, CH, "", Some(MixingKind::None)).unwrap().unwrap();
        assert_eq!(m.species.le, [1.0; 4]);
        assert_eq!(m.species.sct, f64::from(crate::species::SpeciesCoeffs::default().sc_t));
        assert_eq!(m.chemistry.reactor, d);
        assert_eq!(m.chemistry.splitting, Splitting::Lie);
        let e = lb(SP, CH, r#"{ "model": "EDM", "A": 4.0, "B": 0.5 }"#, Some(MixingKind::Epsilon));
        assert_eq!(e.unwrap().unwrap().combustion, LoweredCombustion::Edm { a: 4.0, b: Some(0.5) });
        let e = lb(SP, CH, r#"{ "model": "EDC", "variant": 2005, "CD1": 0.134, "CD2": 0.4,
            "gammaMax": 0.9, "psrSpan": 10 }"#, Some(MixingKind::Omega));
        assert_eq!(
            e.unwrap().unwrap().combustion,
            LoweredCombustion::Edc { variant: EdcVariant::V2005, cd1: 0.134, cd2: 0.4, gamma_max: 0.9, psr_span: 10.0 }
        );
        let e = lb(SP, CH, r#"{ "model": "EDC", "variant": "1996" }"#, Some(MixingKind::Epsilon));
        assert_eq!(
            e.unwrap().unwrap().combustion,
            LoweredCombustion::Edc { variant: EdcVariant::V1989, cd1: 0.135, cd2: 0.50, gamma_max: 0.95, psr_span: 20.0 }
        );
        let e = lb(SP, CH, r#"{ "model": "PaSR", "tauMix": "integral", "tauChem": "gfr", "Cmix": 0.3 }"#, Some(MixingKind::Omega));
        assert_eq!(
            e.unwrap().unwrap().combustion,
            LoweredCombustion::Pasr { tau_mix: TauMix::Integral, tau_chem: TauChem::Gfr, c_mix: Some(0.3) }
        );
        let e = lb(SP, CH, r#"{ "model": "PaSR", "tauMix": "strain", "tauChem": "sfr" }"#, Some(MixingKind::Strain));
        assert_eq!(
            e.unwrap().unwrap().combustion,
            LoweredCombustion::Pasr { tau_mix: TauMix::Strain, tau_chem: TauChem::Sfr, c_mix: None }
        );

        // MixingKind by the turbulence model's name.
        let k = |kind, m| MixingKind::of_model(kind, m);
        assert_eq!(k(None, None), Some(MixingKind::None));
        assert_eq!(k(Some(TurbulenceKind::Ras), Some("kEpsilon")), Some(MixingKind::Epsilon));
        assert_eq!(k(Some(TurbulenceKind::Ras), Some("realizableKE")), Some(MixingKind::Epsilon));
        assert_eq!(k(Some(TurbulenceKind::Ras), Some("kOmegaSST")), Some(MixingKind::Omega));
        assert_eq!(k(Some(TurbulenceKind::Ras), Some("SpalartAllmaras")), Some(MixingKind::None));
        assert_eq!(k(Some(TurbulenceKind::Ras), Some("laminar")), Some(MixingKind::None));
        assert_eq!(k(Some(TurbulenceKind::Les), Some("Smagorinsky")), Some(MixingKind::Strain));
        assert_eq!(k(Some(TurbulenceKind::Ras), Some("v2f")), None);

        // The dictionary route equals the JSONC route (its fields are 0/).
        let dir = temp_case(
            "dict",
            &[
                (SPECIES_FILE, "thermoType { type heRhoThermo; }\nspecies (H2 O2 H2O N2);\ninertSpecie N2;\n\
                    speciesTransport { model lewis; Le { H2 0.3; H2O 0.8; } Sct 0.9; }\n"),
                (CHEMISTRY_FILE, "mechanism \"mech.inp\"; thermo \"therm.dat\"; integrator ros3;\n\
                    rtol 1e-6; atol 1e-12; splitting strang;\n"),
                (COMBUSTION_FILE, "combustionModel laminar;\n"),
                ("0/H2", ""),
                ("0/O2", ""),
                ("0/H2O", ""),
            ],
        );
        let dict = read_case_chem(&dir, Some(MixingKind::None)).unwrap().expect("the dictionaries state chemistry");
        let mut jsonc = c.clone();
        jsonc.species.fields.clear();
        assert_eq!(dict, jsonc, "the dictionary route equals the JSONC route");
        assert_eq!(dict_route_states_chemistry(&dir).unwrap(), Some(CHEMISTRY_FILE));
        std::fs::remove_file(dir.join("0/H2O")).unwrap();
        refuses(read_case_chem(&dir, Some(MixingKind::None)), "0/H2O is missing");
        let _ = std::fs::remove_dir_all(&dir);

        // Serde round trip of the case with all three blocks.
        let case = read_case_jsonc_str(&case_text(&format!(",{EVERY}")), "chemTest").unwrap();
        let back: JsonCase = serde_json::from_str(&serde_json::to_string(&case).unwrap()).unwrap();
        assert_eq!(back, case, "the blocks survive a serde round trip");

        // Absent blocks: no chemistry, and nothing serialised.
        let bare = lower("").expect("a case without the blocks lowers");
        assert!(bare.chem.is_none());
        bare.refuse_chemistry().expect("a case without the blocks is not refused");
        let text = serde_json::to_string(&read_case_jsonc_str(&case_text(""), "chemTest").unwrap()).unwrap();
        for key in ["species", "chemistry", "combustion"] {
            assert!(!text.contains(key), "an absent {key} block is not serialised");
        }

        // load(): the files are read, and the set matches the mechanism.
        let dir = temp_case("load", &[("therm.dat", &thermo_text(true, false)), ("mech.inp", GLOBAL)]);
        let mech = c.load(&dir).expect("the mechanism and thermo load");
        assert_eq!(mech.species().len(), 4);
        assert_eq!(mech.reactions().len(), 1);
        let edm = lb(SP, CH, r#"{ "model": "EDM", "A": 4.0 }"#, Some(MixingKind::Epsilon)).unwrap().unwrap();
        edm.load(&dir).expect("EDM takes one global reaction");
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// SPEC-LIT §137.7 `case_chem_refusals`: every refusal of §137.2 and
    /// §134.1 this lowering owns, each naming its key.
    #[test]
    fn case_chem_refusals() {
        let mix = Some(MixingKind::Epsilon);
        let sp = |patch: &str| SP.replace(r#""inert": "N2","#, patch);
        // species
        refuses(lb("", CH, "", mix), "chemistry");
        refuses(lb("", "", r#"{ "model": "laminar" }"#, mix), "combustion");
        refuses(lb(r#"{ "names": ["N2"], "inert": "N2" }"#, CH, "", mix), "species.names");
        refuses(lb(r#"{ "names": ["N2", "N2"], "inert": "N2" }"#, CH, "", mix), "species.names");
        refuses(lb(&sp(""), CH, "", mix), "species.inert");
        refuses(lb(&sp(r#""inert": "AR","#), CH, "", mix), "species.inert");
        refuses(lb(&sp(r#""inert": "N2", "transport": { "model": "mixtureAveraged" },"#), CH, "", mix), "species.transport.model");
        refuses(lb(&sp(r#""inert": "N2", "transport": { "model": "fickian" },"#), CH, "", mix), "species.transport.model");
        refuses(lb(&sp(r#""inert": "N2", "transport": { "Le": { "AR": 1.0 } },"#), CH, "", mix), "species.transport.Le.AR");
        refuses(lb(&sp(r#""inert": "N2", "transport": { "Le": { "H2": 0.0 } },"#), CH, "", mix), "species.transport.Le.H2");
        refuses(lb(&sp(r#""inert": "N2", "transport": { "Sct": -1.0 },"#), CH, "", mix), "species.transport.Sct");
        let init = |i: &str| SP.replace(r#""initial": { "H2": 0.02, "O2": 0.22, "H2O": 0.0 }"#, i);
        refuses(lb(&init(r#""initial": { "H2": 0.02, "O2": 0.22 }"#), CH, "", mix), "species.initial");
        refuses(lb(&init(r#""initial": { "H2": 0.02, "O2": 0.22, "H2O": 0.0, "N2": 0.76 }"#), CH, "", mix), "species.initial.N2");
        refuses(lb(&init(r#""initial": { "H2": 0.02, "O2": 1.5, "H2O": 0.0 }"#), CH, "", mix), "species.initial.O2");
        refuses(lb(&init(r#""initial": { "H2": 0.6, "O2": 0.5, "H2O": 0.0 }"#), CH, "", mix), "species.initial");
        refuses(lb(&init(r#""initial": { "H2": 0.02, "O2": 0.22, "H2O": 0.0, "AR": 0.0 }"#), CH, "", mix), "species.initial.AR");
        let pat = |y: &str| SP.replace(r#""H2O": { "type": "zeroGradient" }"#, y);
        refuses(lb(&pat(r#""H2O": { "type": "zeroGradient" }, "AR": { "type": "zeroGradient" }"#), CH, "", mix), "species.patches[0].Y.AR");
        refuses(lb(&pat(r#""H2O": { "type": "zeroGradient" }, "N2": { "type": "zeroGradient" }"#), CH, "", mix), "species.patches[0].Y.N2");
        refuses(lb(&pat(r#""H2O": { "type": "fixedFluxTemperature", "q": 1.0 }"#), CH, "", mix), "species.patches[0].Y.H2O");
        refuses(lb(&pat(r#""H2O": { "type": "fixedValue", "value": -0.1 }"#), CH, "", mix), "species.patches[0].Y.H2O");
        refuses(lb(&pat(r#""X": { "type": "zeroGradient" }"#), CH, "", mix), "species.patches[0].Y.X");
        refuses(lb(&SP.replace(r#", "H2O": { "type": "zeroGradient" }"#, ""), CH, "", mix), "\"H2O\" on it");
        {
            let spj: JsonSpecies = serde_json::from_str(SP).unwrap();
            let chj: JsonChemistry = serde_json::from_str(CH).unwrap();
            let b = ChemBlocks { species: Some(&spj), chemistry: Some(&chj), combustion: None };
            refuses(lower_blocks(b, mix, true, None), "physics.compressible");
        }
        refuses(lower(&format!(",{}", EVERY.replace("\"inert\": \"N2\",", ""))).map(|_| ()), "species.inert");
        // chemistry
        refuses(lb(SP, "", "", mix), "chemistry");
        refuses(lb(SP, r#"{ "thermo": "t" }"#, "", mix), "chemistry.mechanism");
        refuses(lb(SP, r#"{ "mechanism": "m" }"#, "", mix), "chemistry.thermo");
        refuses(lb(SP, r#"{ "mechanism": "m", "thermo": "t", "transport": "tran.dat" }"#, "", mix), "chemistry.transport");
        refuses(lb(SP, r#"{ "mechanism": "m", "thermo": "t", "integrator": "ros4" }"#, "", mix), "chemistry.integrator");
        refuses(lb(SP, r#"{ "mechanism": "m", "thermo": "t", "integrator": "euler" }"#, "", mix), "chemistry.integrator");
        refuses(lb(SP, r#"{ "mechanism": "m", "thermo": "t", "rtol": 0 }"#, "", mix), "chemistry.rtol");
        refuses(lb(SP, r#"{ "mechanism": "m", "thermo": "t", "atol": -1e-9 }"#, "", mix), "chemistry.atol");
        refuses(lb(SP, r#"{ "mechanism": "m", "thermo": "t", "splitting": "godunov" }"#, "", mix), "chemistry.splitting");
        // combustion against MixingRate (§134.1)
        refuses(lb(SP, CH, r#"{ "model": "flamelet" }"#, mix), "combustion.model");
        for co in [r#"{ "model": "EDM", "A": 4 }"#, r#"{ "model": "EDC", "variant": 1989 }"#,
                   r#"{ "model": "PaSR", "tauMix": "kolmogorov", "tauChem": "sfr" }"#] {
            refuses(lb(SP, CH, co, Some(MixingKind::None)), "MixingRate::None");
            refuses(lb(SP, CH, co, None), "combustion.model");
        }
        refuses(lb(SP, CH, r#"{ "model": "EDM", "A": 4 }"#, Some(MixingKind::Strain)), "MixingRate::Strain");
        refuses(lb(SP, CH, r#"{ "model": "EDC", "variant": 1989 }"#, Some(MixingKind::Strain)), "MixingRate::Strain");
        // the closures' own keys
        refuses(lb(SP, CH, r#"{ "model": "EDM" }"#, mix), "combustion.A");
        refuses(lb(SP, CH, r#"{ "model": "EDM", "A": 0 }"#, mix), "combustion.A");
        refuses(lb(SP, CH, r#"{ "model": "EDM", "A": 4, "B": -1 }"#, mix), "combustion.B");
        refuses(lb(SP, CH, r#"{ "model": "laminar", "A": 4 }"#, mix), "combustion.A");
        refuses(lb(SP, CH, r#"{ "model": "EDM", "A": 4, "variant": 1989 }"#, mix), "combustion.variant");
        refuses(lb(SP, CH, r#"{ "model": "EDC" }"#, mix), "combustion.variant");
        refuses(lb(SP, CH, r#"{ "model": "EDC", "variant": "batch" }"#, mix), "combustion.variant");
        refuses(lb(SP, CH, r#"{ "model": "EDC", "variant": "2010" }"#, mix), "combustion.variant");
        refuses(lb(SP, CH, r#"{ "model": "EDC", "variant": 1989, "CD1": 0 }"#, mix), "combustion.CD1");
        refuses(lb(SP, CH, r#"{ "model": "EDC", "variant": 1989, "CD2": -1 }"#, mix), "combustion.CD2");
        refuses(lb(SP, CH, r#"{ "model": "EDC", "variant": 1989, "gammaMax": 1.0 }"#, mix), "combustion.gammaMax");
        refuses(lb(SP, CH, r#"{ "model": "EDC", "variant": 1989, "psrSpan": 0 }"#, mix), "combustion.psrSpan");
        refuses(lb(SP, CH, r#"{ "model": "EDC", "variant": 1989, "Cmix": 1 }"#, mix), "combustion.Cmix");
        refuses(lb(SP, CH, r#"{ "model": "PaSR", "tauChem": "sfr" }"#, mix), "combustion.tauMix");
        refuses(lb(SP, CH, r#"{ "model": "PaSR", "tauMix": "batchelor", "tauChem": "sfr" }"#, mix), "combustion.tauMix");
        refuses(lb(SP, CH, r#"{ "model": "PaSR", "tauMix": "kolmogorov" }"#, mix), "combustion.tauChem");
        refuses(lb(SP, CH, r#"{ "model": "PaSR", "tauMix": "kolmogorov", "tauChem": "batchelor" }"#, mix), "combustion.tauChem");
        refuses(lb(SP, CH, r#"{ "model": "PaSR", "tauMix": "integral", "tauChem": "sfr" }"#, mix), "combustion.Cmix");
        refuses(lb(SP, CH, r#"{ "model": "PaSR", "tauMix": "geometric", "tauChem": "sfr", "Cmix": 1 }"#, mix), "combustion.Cmix");
        refuses(lb(SP, CH, r#"{ "model": "PaSR", "tauMix": "strain", "tauChem": "sfr" }"#, mix), "combustion.tauMix");
        refuses(lb(SP, CH, r#"{ "model": "PaSR", "tauMix": "kolmogorov", "tauChem": "sfr" }"#, Some(MixingKind::Strain)), "combustion.tauMix");
        // an unknown key, on both routes
        assert!(serde_json::from_str::<JsonCombustion>(r#"{ "model": "EDM", "C": 1 }"#).is_err());
        let dir = temp_case("unknown", &[(CHEMISTRY_FILE, "chemistry on;\n")]);
        refuses(read_case_chem(&dir, mix), "chemistry");
        let _ = std::fs::remove_dir_all(&dir);
        let dir = temp_case("reactions", &[(REACTIONS_FILE, "reactions {}\n")]);
        refuses(read_case_chem(&dir, mix), REACTIONS_FILE);
        let _ = std::fs::remove_dir_all(&dir);
        let dir = temp_case("inertonly", &[(SPECIES_FILE, "inertSpecie N2;\n")]);
        refuses(read_case_chem(&dir, mix), "inertSpecie");
        let _ = std::fs::remove_dir_all(&dir);
        // the driver seam of the dictionary route
        for f in [CHEMISTRY_FILE, COMBUSTION_FILE, REACTIONS_FILE] {
            let dir = temp_case("seam", &[("constant/polyMesh/owner", ""), (f, "model laminar;\n")]);
            refuses(crate::io::case::read_case_controls(&dir).map(|_| ()), f);
            let _ = std::fs::remove_dir_all(&dir);
        }
        let dir = temp_case("seam", &[("constant/polyMesh/owner", ""), (SPECIES_FILE, "species (H2 N2);\n")]);
        refuses(crate::io::case::read_case_controls(&dir).map(|_| ()), "CHR-12b");
        let _ = std::fs::remove_dir_all(&dir);

        // the file-dependent refusals of load()
        let ok = lb(SP, CH, "", mix).unwrap().unwrap();
        let dir = temp_case("nothermo", &[("therm.dat", &thermo_text(false, false)), ("mech.inp", GLOBAL)]);
        // §129's own refusal, naming the mechanism file and the line.
        match ok.load(&dir) {
            Err(Error::Parse { path, msg }) => {
                assert!(path.ends_with("mech.inp") && msg.contains("line 2") && msg.contains("H2O"), "{path}: {msg}")
            }
            other => panic!("a mechanism species missing in thermo: expected §129's refusal, got {other:?}"),
        }
        let _ = std::fs::remove_dir_all(&dir);
        let dir = temp_case("ar", &[("therm.dat", &thermo_text(true, true)), ("mech.inp", GLOBAL)]);
        // The dictionary route's lowering (no fields), so a set can differ.
        let with = |names: &str| {
            let spj: JsonSpecies =
                serde_json::from_str(&format!(r#"{{ "names": {names}, "inert": "N2" }}"#)).unwrap();
            let chj: JsonChemistry = serde_json::from_str(CH).unwrap();
            let b = ChemBlocks { species: Some(&spj), chemistry: Some(&chj), combustion: None };
            lower_blocks(b, mix, false, None).unwrap().unwrap()
        };
        refuses(with(r#"["H2", "O2", "H2O", "N2", "XE"]"#).load(&dir), "\"XE\" is not in the thermo set");
        refuses(with(r#"["H2", "O2", "H2O", "N2", "AR"]"#).load(&dir), "\"AR\" is not a species of the mechanism");
        refuses(with(r#"["H2", "O2", "N2"]"#).load(&dir), "the mechanism's species \"H2O\"");
        let _ = std::fs::remove_dir_all(&dir);
        let dir = temp_case("rev", &[("therm.dat", &thermo_text(true, false)), ("mech.inp", REVERSIBLE)]);
        let edm = lb(SP, CH, r#"{ "model": "EDM", "A": 4 }"#, mix).unwrap().unwrap();
        refuses(edm.load(&dir), "EDM with the reaction");
        ok.load(&dir).expect("laminar takes a reversible mechanism");
        let _ = std::fs::remove_dir_all(&dir);
        let dir = temp_case("missing", &[]);
        assert!(matches!(ok.load(&dir), Err(Error::Config(m)) if m.contains("chemistry.thermo")));
        let _ = std::fs::remove_dir_all(&dir);
    }
}
