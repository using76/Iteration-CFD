// meteor-cfd - Copyright (c) 2026 주식회사 메테오시뮬레이션 (Meteo Simulation Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.

//! JSONC case reader: `docs/case-example.json` -> [`JsonCase`] -> [`LoweredCase`].
//!
//! Written from `docs/05-io-redesign.md` section 4.1 (the JSONC case format
//! decision) and `docs/case-example.json` (the format itself, read as data,
//! not as anyone's source). The mapping onto the solver's own control types
//! is `SPEC-LIT.md`'s §13.4 contract ("an unsupported setting is a loud
//! error naming the alternatives"), applied through [`crate::io::contract`]
//! exactly as `crate::io::case` and `crate::io::schemes` already apply it to
//! an OpenFOAM case - this reader calls the SAME parsing functions
//! ([`crate::io::schemes::parse_div`], `parse_grad`, `parse_sn_grad`,
//! [`crate::timescheme::DdtScheme::parse`], [`crate::io::case::LinearSolverKind::from_name`],
//! [`crate::io::case::Preconditioner::from_name`]) on the scheme strings a
//! JSONC case carries, so a case written either way is refused or accepted
//! for the same reasons. The patch-kind presets (wall/inlet/open) reproduce
//! the boundary conditions `crate::blockgen::write_case`'s `write_initial_fields`
//! gives an uncarved case's walls, inlet and open boundary - read there, not
//! copied from anywhere else. No GPL-licensed source was consulted.
//!
//! # Three layers
//!
//! 1. [`JsonCase`] and its fields mirror `docs/case-example.json` field for
//!    field, `#[serde(deny_unknown_fields)]` throughout, and derive
//!    [`schemars::JsonSchema`] so [`emit_schema`] is generated from the SAME
//!    types that parse the file - the two cannot disagree the way a
//!    hand-maintained schema and a hand-maintained reader can (the SU2
//!    failure mode `docs/05-io-redesign.md` section 4.1 names).
//! 2. [`read_case_jsonc`] strips comments and trailing commas
//!    ([`jsonc_parser`]) and deserialises through [`serde_path_to_error`], so
//!    a bad case names the JSON path (`patches[3].kind`) rather than just
//!    "invalid value".
//! 3. [`JsonCase::lower`] maps the parsed tree onto the solver's own control
//!    types - `crate::blockgen::BlockSpec`, `crate::io::case::TurbulenceControls`,
//!    `SolverControls`, `crate::timescheme::DdtScheme`, `crate::io::schemes::DivEntry`
//!    - plus the small new types this format needs and the old ones have no
//!    place for ([`RunControl`], [`LoweredScalarField`], [`LoweredVectorField`]).
//!
//! Patch resolution - both `patches[]` against a boundary's name and
//! `numerics.solvers[]` against an equation's name - is **first-match-wins in
//! file order**, via [`crate::io::regex::Regex`] (anchored both ends, same
//! engine `crate::io::dict::FoamDict::resolve` uses for a quoted OpenFOAM
//! key). A JSON object cannot carry that order, which is exactly why
//! `docs/05-io-redesign.md` section 4.1 point 1 makes both of them arrays.
//!
//! Wiring this into a driver (building an actual `HostMesh`, running the
//! solver) is the B3 agent's job, not this file's.

use std::collections::BTreeMap;
use std::path::Path;

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::blockgen::{BlockSpec, GradedAxis, PatchWindow};
use crate::error::{Error, IoContext, Result};
use crate::field::BcKind;
use crate::io::case::{
    compute_y_plus_lam, format_time_name, AlgorithmControls, CaseControls, LinearSolverKind,
    Preconditioner, SolverControls, TurbulenceControls, WallFunctionCoeffs,
};
use crate::io::contract::unsupported;
use crate::io::fields::PatchFieldSpec;
use crate::io::regex::Regex;
use crate::io::schemes::{parse_div, parse_grad, parse_sn_grad, DivEntry};
use crate::fv::{GradScheme, SnGradScheme};
use crate::momentum::BuoyancyCoeffs;
use crate::scalar_transport::ScalarTransportCoeffs;
use crate::timescheme::DdtScheme;
use crate::combustion::CombustionCoeffs;
use crate::radiation::{RadiationModel, RadiationProps};
use crate::{Label, Scalar, Vec3};

// ==========================================================================
//  1. The JSON tree - mirrors docs/case-example.json field for field
// ==========================================================================

/// The whole case file. `$schema` is accepted and ignored - it points a
/// human's editor at [`emit_schema`]'s output, and carries no information the
/// reader needs.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct JsonCase {
    #[serde(rename = "$schema", default, skip_serializing_if = "Option::is_none")]
    pub schema: Option<String>,
    pub name: String,
    pub mesh: JsonMesh,
    pub physics: JsonPhysics,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub turbulence: Option<JsonTurbulence>,
    /// ORDERED, matched first-to-last - section 4.1 point 1. A JSON object's
    /// keys cannot carry this, which is the entire reason this is an array of
    /// `{match, ...}` records rather than a `{"pattern": {...}}` map.
    pub patches: Vec<JsonPatchRule>,
    pub initial: JsonInitial,
    pub numerics: JsonNumerics,
    pub run: JsonRun,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output: Option<JsonOutput>,
}

// ---- mesh ----------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "lowercase")]
pub enum MeshKind {
    Cartesian,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct JsonBounds {
    pub min: [f64; 3],
    pub max: [f64; 3],
}

/// Which patch each of the six box faces belongs to, `-x +x -y +y -z +z` -
/// [`crate::blockgen::BlockSpec`]'s own slot order.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct JsonBoundaries {
    pub xmin: String,
    pub xmax: String,
    pub ymin: String,
    pub ymax: String,
    pub zmin: String,
    pub zmax: String,
}

/// A rectangular window carved out of one box face, replacing the
/// `PatchWindow` workaround `docs/05-io-redesign.md` section 4.1's opening
/// comment complains about: here it is just a region.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct JsonRegion {
    pub name: String,
    /// Which `JsonBoundaries` face this window is cut from: `xmin`, `xmax`,
    /// `ymin`, `ymax`, `zmin` or `zmax`.
    pub on: String,
    pub shape: JsonShape,
}

/// Internally tagged on `kind`, so today's one variant already carries its
/// own `oneOf`/`const` discriminator in [`emit_schema`]'s output and a second
/// shape (`cylinder`, say) is additive.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "kind", rename_all = "lowercase", deny_unknown_fields)]
pub enum JsonShape {
    Box { min: [f64; 3], max: [f64; 3] },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct JsonMesh {
    pub kind: MeshKind,
    pub bounds: JsonBounds,
    /// Cell counts along `x, y, z`.
    pub cells: [u32; 3],
    pub boundaries: JsonBoundaries,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub regions: Vec<JsonRegion>,
}

// ---- physics ---------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct JsonFluid {
    pub nu: f64,
    #[serde(rename = "Pr")]
    pub pr: f64,
    #[serde(rename = "Prt")]
    pub prt: f64,
    #[serde(rename = "TRef")]
    pub t_ref: f64,
}

/// `b = g·(TRef/T - 1)` (density-ratio, `SPEC-LIT` §9) is the only model this
/// solver implements - see [`crate::momentum::BuoyancyCoeffs`]. `boussinesq`
/// is accepted here, so the schema documents that OpenFOAM cases spell it
/// that way too, and [`JsonCase::lower`] rejects it under the §13.4 contract
/// rather than silently linearising a fire plume's `ΔT/T ≈ 3`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub enum BuoyancyModel {
    DensityRatio,
    Boussinesq,
}

/// SPEC-LIT §27's mixing-controlled combustion, overridable per case.
/// Presence of this block (`physics.fire.combustion`) is what turns
/// combustion ON in `ofgpu-fire`; everything inside it is optional and falls
/// back to [`CombustionCoeffs::default`]'s propane values (SPEC-LIT §27's own
/// *DESIGN* default fuel).
///
/// The species set combustion needs (`Y_F`, `Y_O2`, `Y_P`, inert `N2`) is
/// NOT named here - it is fixed by [`JsonCase::lower`], because SPEC-LIT §27
/// names exactly these three reacting species and there is nothing for a
/// case to choose about their identities, only about the coefficients above.
/// A case turns combustion on by giving `initial.Y_F` and an inlet patch's
/// own `Y_F` condition - see [`y_f_spec_for`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct JsonCombustion {
    #[serde(rename = "CEDM", default, skip_serializing_if = "Option::is_none")]
    pub c_edm: Option<f64>,
    #[serde(rename = "CEDMLES", default, skip_serializing_if = "Option::is_none")]
    pub c_edm_les: Option<f64>,
    /// Stoichiometric O2/fuel MASS ratio `s`. Propane's `3.63` by default.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub s: Option<f64>,
    /// Heat of combustion, J/kg. Propane's `46.45e6` by default.
    #[serde(rename = "dhc", default, skip_serializing_if = "Option::is_none")]
    pub dh_c: Option<f64>,
}

/// SPEC-LIT §28's P1 gray radiation. `model` is validated by
/// [`RadiationModel::from_name`] - naming `fvDOM` here is the §13.4 gate that
/// section documents, not a silent fallback to P1.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct JsonRadiation {
    /// `"P1"` - the only model this crate implements; see
    /// [`RadiationModel::from_name`] for what happens with anything else.
    pub model: String,
    /// Gray absorption coefficient `a`, 1/m. Constant, case-supplied
    /// (SPEC-LIT §28 v1; WSGG is later work).
    #[serde(rename = "a")]
    pub absorption: f64,
    /// The radiant-fraction floor `chi_r`. Defaults to
    /// [`crate::radiation::CHI_R_DEFAULT`] (`0.35`, FDS practice).
    #[serde(rename = "chiR", default, skip_serializing_if = "Option::is_none")]
    pub chi_r: Option<f64>,
    /// Marshak wall emissivity, applied to every `wall`-kind patch. Defaults
    /// to `1.0` (black wall) when absent.
    #[serde(rename = "wallEmissivity", default, skip_serializing_if = "Option::is_none")]
    pub wall_emissivity: Option<f64>,
}

/// `physics.fire`: the two S27/S28 blocks `ofgpu-fire` reads. Either or both
/// may be absent - a case with neither runs the plain S25/S26 low-Mach solver
/// `ofgpu-fire` already had, unchanged.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct JsonFire {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub combustion: Option<JsonCombustion>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub radiation: Option<JsonRadiation>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct JsonPhysics {
    pub gravity: [f64; 3],
    pub fluid: JsonFluid,
    pub buoyancy: BuoyancyModel,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fire: Option<JsonFire>,
}

// ---- turbulence ------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub enum TurbulenceKind {
    #[serde(rename = "RAS")]
    Ras,
    #[serde(rename = "LES")]
    Les,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct JsonWallFunctions {
    pub kappa: f64,
    #[serde(rename = "E")]
    pub e: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct JsonTurbulence {
    pub kind: TurbulenceKind,
    /// `kEpsilon`, `kOmegaSST`, ... - validated at model-construction time,
    /// same as an OpenFOAM `momentumTransport`'s `RAS/model`; not here, so a
    /// build without a given model is not a reason to refuse every case.
    pub model: String,
    #[serde(rename = "wallFunctions")]
    pub wall_functions: JsonWallFunctions,
    /// SPEC-LIT §29.1 route (b): the case-level `wallTreatment` default every
    /// wall patch's four turbulence-closure types expand to, unless a patch's
    /// own `treatment` ([`JsonPatchRule::wall_treatment`]) or its own
    /// explicit `k`/`epsilon`/`omega`/`nut` overrides it. `standard` (the
    /// OpenFOAM-compatible default) when the case names none.
    #[serde(rename = "wallTreatment", default)]
    pub wall_treatment: WallTreatmentKind,
}

/// SPEC-LIT §29.1's four presets, as the JSONC format spells them -
/// `crate::io::case::WallTreatment`'s twin, kept separate so this module's
/// [`schemars`] derive is what puts the four names in [`emit_schema`]'s
/// output rather than a hand-written schema fragment that could drift from
/// the reader.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub enum WallTreatmentKind {
    Standard,
    Spalding,
    Rough,
    LowRe,
}

impl Default for WallTreatmentKind {
    fn default() -> Self {
        Self::Standard
    }
}

impl WallTreatmentKind {
    pub fn to_case(self) -> crate::io::case::WallTreatment {
        match self {
            Self::Standard => crate::io::case::WallTreatment::Standard,
            Self::Spalding => crate::io::case::WallTreatment::Spalding,
            Self::Rough => crate::io::case::WallTreatment::Rough,
            Self::LowRe => crate::io::case::WallTreatment::LowRe,
        }
    }
}

// ---- patches -----------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "lowercase")]
pub enum PatchPresetKind {
    Wall,
    Inlet,
    Open,
}

/// One `{"type": "fixedValue", "value": ...}`-shaped scalar boundary
/// condition. Internally tagged, so `emit_schema` renders it as `oneOf` with
/// a `const` discriminator on `type` - deliverable 4 of this module.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "type", deny_unknown_fields)]
pub enum ScalarBc {
    #[serde(rename = "fixedValue")]
    FixedValue { value: f64 },
    #[serde(rename = "inletOutlet")]
    InletOutlet {
        #[serde(rename = "inletValue")]
        inlet_value: f64,
    },
    #[serde(rename = "zeroGradient")]
    ZeroGradient {},
    /// SPEC-LIT §29.3's Jayatilleke thermal wall function, nameable
    /// explicitly on `T` with its own wall target temperature `T_w` -
    /// [`TurbBc`]'s wall-function variants already carry a `value` this same
    /// way for `nut`/`k`/`epsilon`/`omega`; `T` had no such variant, only the
    /// case-level `wallTreatment` auto-completion of [`t_spec_for`], which
    /// (deliberately, until this variant existed) never has a wall
    /// temperature of its own to write and falls back to the neighbour
    /// cell's value - fine for an adiabatic-ish plume floor, useless for a
    /// genuinely hot wall against a cooler bulk (SPEC-LIT §29.3's own
    /// deferred gate: a channel with fixed-T hot walls). An explicit
    /// `T: { "type": "thermalWallFunction", "value": T_w }` on a wall rule is
    /// the highest-precedence route of §29.1's table and is honoured here the
    /// same way an explicit `nutkWallFunction { value }` already is.
    #[serde(rename = "thermalWallFunction")]
    ThermalWallFunction { value: f64 },
}

/// Vector counterpart of [`ScalarBc`], for `U`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "type", deny_unknown_fields)]
pub enum VectorBc {
    #[serde(rename = "fixedValue")]
    FixedValue { value: [f64; 3] },
    #[serde(rename = "inletOutlet")]
    InletOutlet {
        #[serde(rename = "inletValue")]
        inlet_value: [f64; 3],
    },
    #[serde(rename = "zeroGradient")]
    ZeroGradient {},
}

/// One `k`/`epsilon`/`omega`/`nut` boundary condition. A superset of
/// [`ScalarBc`]: a turbulence closure field's wall treatment is a wall
/// FUNCTION, not `fixedValue`/`zeroGradient`, and which one governs whether
/// [`crate::field_setup::WallFaces::from_case`] treats a patch as a wall at
/// all (it reads `epsilon`'s and `nut`'s own patch *type strings* - see that
/// function and `crate::field_setup::wall_cell_faces`/`nut_wall_faces`). This
/// type exists because the B3 gate (`docs/05-io-redesign.md` phase 1) needs a
/// JSONC case to reproduce a generated OpenFOAM case's `k`/`epsilon`/`nut`
/// EXACTLY, wall-function type strings included - `JsonPatchTurbulence`'s
/// intensity/mixing-length inlet estimate has nothing to say about a wall
/// patch's type, only an inlet's value. `value` is the uniform value written
/// on every variant, including the wall-function ones: OpenFOAM itself writes
/// a `value` entry there too (the field's own initial guess before the wall
/// function first runs), and `docs/case-example.json`'s own generated
/// counterpart does the same - see e.g. `cases/plumeB/0/k`'s `floor` entry.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "type", deny_unknown_fields)]
pub enum TurbBc {
    #[serde(rename = "fixedValue")]
    FixedValue { value: f64 },
    #[serde(rename = "zeroGradient")]
    ZeroGradient {},
    /// `nut`'s own non-wall-function type (an inlet/outlet with no
    /// prescribed eddy viscosity) - OpenFOAM's `calculated`.
    #[serde(rename = "calculated")]
    Calculated { value: f64 },
    #[serde(rename = "kqRWallFunction")]
    KqRWallFunction { value: f64 },
    #[serde(rename = "epsilonWallFunction")]
    EpsilonWallFunction { value: f64 },
    #[serde(rename = "omegaWallFunction")]
    OmegaWallFunction { value: f64 },
    #[serde(rename = "nutkWallFunction")]
    NutkWallFunction { value: f64 },
    /// SPEC-LIT §15.1/§29.1: `y+` from the local velocity rather than `k` -
    /// an explicit per-patch override of the `spalding` preset row's `nut`,
    /// nameable on its own patch under any case-level default (§29.1's
    /// precedence: "explicit per-field BCs...override").
    #[serde(rename = "nutUWallFunction")]
    NutUWallFunction { value: f64 },
    /// SPEC-LIT §15.2/§29.1: `nu_t,w = 0`, no wall model - the `lowRe`
    /// preset row's `nut`, nameable explicitly on one patch.
    #[serde(rename = "nutLowReWallFunction")]
    NutLowReWallFunction { value: f64 },
    /// SPEC-LIT §15.4/§29.1: the resolved-sublayer `k` completion - the
    /// `lowRe` preset row's `k`, nameable explicitly on one patch.
    #[serde(rename = "kLowReWallFunction")]
    KLowReWallFunction { value: f64 },
}

/// The turbulence intensity/mixing-length estimate an inlet carries, in the
/// same units OpenFOAM's own `turbulentIntensityKineticEnergyInlet` /
/// `turbulentMixingLengthDissipationRateInlet` take: `intensity` a fraction
/// of the inlet speed, `mixingLength` an absolute length in metres (NOT a
/// fraction - `crate::blockgen`'s canned cases multiply a fraction by a
/// hydraulic diameter internally; a case file names the length directly).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields, rename_all = "camelCase")]
pub struct JsonPatchTurbulence {
    pub intensity: f64,
    pub mixing_length: f64,
}

/// One ordered rule of the `patches` array.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct JsonPatchRule {
    #[serde(rename = "match")]
    pub pattern: String,
    pub kind: PatchPresetKind,
    #[serde(rename = "U", default, skip_serializing_if = "Option::is_none")]
    pub u: Option<VectorBc>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub p: Option<ScalarBc>,
    #[serde(rename = "T", default, skip_serializing_if = "Option::is_none")]
    pub t: Option<ScalarBc>,
    /// The fuel mass fraction at this patch - SPEC-LIT §27's "an inlet
    /// carrying `Y_F`". Only required on the patch(es) that inject fuel; see
    /// [`y_f_spec_for`] for the wall/open defaults and why `Y_O2`/`Y_P` are
    /// not separately named here.
    #[serde(rename = "Y_F", default, skip_serializing_if = "Option::is_none")]
    pub y_f: Option<ScalarBc>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub turbulence: Option<JsonPatchTurbulence>,
    /// `k`'s own condition. Only required when `initial.k` is given (this
    /// case solves `k`) - see [`turb_field_spec`].
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub k: Option<TurbBc>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub epsilon: Option<TurbBc>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub omega: Option<TurbBc>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub nut: Option<TurbBc>,
    /// SPEC-LIT §29.1 route (b): this patch's own `wallTreatment` override,
    /// on a `wall`-kind patch only - `None` falls back to
    /// `turbulence.wallTreatment`, the case-level default. Meaningless (and
    /// simply unused) on an `inlet`/`open` rule.
    #[serde(rename = "treatment", default, skip_serializing_if = "Option::is_none")]
    pub wall_treatment: Option<WallTreatmentKind>,
    /// Sand-grain height, m - required when the effective treatment (this
    /// rule's own [`Self::wall_treatment`], or the case default) is `rough`.
    #[serde(rename = "Ks", default, skip_serializing_if = "Option::is_none")]
    pub ks: Option<f64>,
    /// The roughness constant; defaults to `0.5` (SPEC-LIT §15.3's "uniform
    /// sand") when `Ks` is given and this is not.
    #[serde(rename = "Cs", default, skip_serializing_if = "Option::is_none")]
    pub cs: Option<f64>,
}

// ---- initial -----------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct JsonInitial {
    #[serde(rename = "U")]
    pub u: [f64; 3],
    /// Absent for an isothermal case - not every case solves a temperature
    /// equation, and a `0` here would claim one does.
    #[serde(rename = "T", default, skip_serializing_if = "Option::is_none")]
    pub t: Option<f64>,
    /// The ambient fuel mass fraction, and the presence check that turns
    /// combustion (SPEC-LIT §27) on - same convention as [`Self::t`]:
    /// absent means this case does not solve species at all. Usually `0.0`
    /// (no fuel in the ambient air); the fuel inlet's own value comes from
    /// its patch rule's `Y_F` ([`y_f_spec_for`]), not from here.
    #[serde(rename = "Y_F", default, skip_serializing_if = "Option::is_none")]
    pub y_f: Option<f64>,
    pub p: f64,
    /// The four turbulence closure fields, each absent when the case does
    /// not solve it - same reasoning as [`Self::t`]. Present exactly when a
    /// `patches[]` rule must carry the matching [`TurbBc`] - see
    /// [`turb_field_spec`].
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub k: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub epsilon: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub omega: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub nut: Option<f64>,
}

// ---- numerics ------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub enum AlgorithmKind {
    #[serde(rename = "SIMPLE")]
    Simple,
    #[serde(rename = "PISO")]
    Piso,
    #[serde(rename = "PIMPLE")]
    Pimple,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct JsonAlgorithm {
    pub kind: AlgorithmKind,
    #[serde(rename = "outerCorrectors", default, skip_serializing_if = "Option::is_none")]
    pub outer_correctors: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub correctors: Option<usize>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct JsonLaplacian {
    /// The `snGradSchemes` half of a `laplacianSchemes` entry -
    /// `crate::io::schemes::parse_sn_grad`'s grammar (`corrected`,
    /// `uncorrected`, `limited <alpha>`). The `Gauss <interpolation>` half is
    /// not exposed: `linear` is the only interpolation this solver has, so
    /// there is nothing for a case to choose.
    #[serde(rename = "snGrad")]
    pub sn_grad: String,
    #[serde(rename = "nonOrthogonalCorrectors")]
    pub non_orthogonal_correctors: usize,
}

/// One `numerics.solvers` rule.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct JsonSolverRule {
    #[serde(rename = "match")]
    pub pattern: String,
    pub solver: String,
    pub preconditioner: String,
    pub tolerance: f64,
    /// `SolverControls::default`'s own `0.0` (no relative tolerance) when
    /// absent - the same fallback an OpenFOAM `solvers/<var>` entry with no
    /// `relTol` gets from `read_solver_controls`.
    #[serde(rename = "relTol", default)]
    pub rel_tol: f64,
    /// `SolverControls::default`'s own `1000` when absent, for the same
    /// reason as [`Self::rel_tol`].
    #[serde(rename = "maxIter", default = "default_max_iter")]
    pub max_iter: Label,
}

fn default_max_iter() -> Label {
    SolverControls::default().max_iter
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct JsonNumerics {
    pub algorithm: JsonAlgorithm,
    /// `ddtSchemes/default`'s grammar - `crate::timescheme::DdtScheme::parse`.
    pub ddt: String,
    /// `divSchemes`, keyed by entry name (`"default"`, `"div(phi,U)"`, ...).
    /// A map rather than named fields: the set of equations a case names here
    /// is open-ended, and `crate::io::schemes::parse_div` is what validates
    /// each value, not this type.
    pub div: BTreeMap<String, String>,
    /// `gradSchemes/default` - `crate::io::schemes::parse_grad`'s grammar.
    pub grad: String,
    pub laplacian: JsonLaplacian,
    /// `relaxationFactors/equations`, keyed by field name.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub relaxation: BTreeMap<String, f64>,
    /// ORDERED, matched first-to-last against the equation's name - the same
    /// reason `patches` is an array rather than a map.
    pub solvers: Vec<JsonSolverRule>,
}

// ---- run -----------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct JsonRun {
    #[serde(rename = "endTime")]
    pub end_time: f64,
    #[serde(rename = "deltaT")]
    pub delta_t: f64,
    #[serde(rename = "adjustTimeStep", default)]
    pub adjust_time_step: bool,
    #[serde(rename = "maxCo", default, skip_serializing_if = "Option::is_none")]
    pub max_co: Option<f64>,
}

// ---- output ----------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct JsonVisualisation {
    pub format: String,
    pub interval: f64,
    pub fields: Vec<String>,
    pub precision: String,
    #[serde(rename = "usdScene", default)]
    pub usd_scene: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct JsonExact {
    pub format: String,
    pub interval: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct JsonRestart {
    pub interval: f64,
    pub keep: usize,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct JsonOutput {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub visualisation: Option<JsonVisualisation>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub exact: Option<JsonExact>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub restart: Option<JsonRestart>,
}

// ==========================================================================
//  2. Reading: strip JSONC, deserialise with a JSON-path error
// ==========================================================================

/// The dialect this reader accepts: JSON plus `//`/`/* */` comments plus
/// trailing commas, and nothing else - `docs/05-io-redesign.md` section 4.1's
/// own words for the decision. `jsonc_parser`'s defaults are far looser
/// (single-quoted strings, hex numbers, unary `+`, missing commas), so every
/// other relaxation is turned off explicitly rather than inherited.
fn parse_options() -> jsonc_parser::ParseOptions {
    jsonc_parser::ParseOptions {
        allow_comments: true,
        allow_trailing_commas: true,
        allow_loose_object_property_names: false,
        allow_missing_commas: false,
        allow_single_quoted_strings: false,
        allow_hexadecimal_numbers: false,
        allow_unary_plus_numbers: false,
    }
}

/// A JSONC number is kept as source text (`"1.5e-5"`, `"98"`, `"-1"`); this is
/// what makes an integer literal deserialise into a `u32`/`usize` field and a
/// decimal literal deserialise into an `f64` one - `serde_json::Number`
/// distinguishes the two internally and a `Number` built only through
/// `from_f64` can never satisfy `as_u64`/`as_i64` again, whatever the value.
/// Losing that distinction here would make `"cells": [98, 42, 20]` fail to
/// parse into `[u32; 3]`.
fn number_from_raw(raw: &str) -> Result<serde_json::Number> {
    if !raw.contains(['.', 'e', 'E']) {
        if let Ok(i) = raw.parse::<i64>() {
            return Ok(serde_json::Number::from(i));
        }
        if let Ok(u) = raw.parse::<u64>() {
            return Ok(serde_json::Number::from(u));
        }
    }
    let f: f64 = raw
        .parse()
        .map_err(|_| Error::Config(format!("\"{raw}\" is not a number")))?;
    serde_json::Number::from_f64(f)
        .ok_or_else(|| Error::Config(format!("\"{raw}\" is not a finite number")))
}

/// [`jsonc_parser::JsonValue`] -> [`serde_json::Value`], so the rest of the
/// pipeline (deserialisation, the JSON-path error reporter) can be the
/// ordinary `serde_json`/`serde_path_to_error` combination and does not have
/// to know a JSONC-specific value type exists.
fn jsonc_to_serde(v: jsonc_parser::JsonValue) -> Result<serde_json::Value> {
    Ok(match v {
        jsonc_parser::JsonValue::Null => serde_json::Value::Null,
        jsonc_parser::JsonValue::Boolean(b) => serde_json::Value::Bool(b),
        jsonc_parser::JsonValue::String(s) => serde_json::Value::String(s.into_owned()),
        jsonc_parser::JsonValue::Number(raw) => serde_json::Value::Number(number_from_raw(raw)?),
        jsonc_parser::JsonValue::Array(arr) => {
            let mut out = Vec::new();
            for item in arr {
                out.push(jsonc_to_serde(item)?);
            }
            serde_json::Value::Array(out)
        }
        jsonc_parser::JsonValue::Object(obj) => {
            let mut map = serde_json::Map::new();
            for (k, val) in obj {
                map.insert(k.into_owned(), jsonc_to_serde(val)?);
            }
            serde_json::Value::Object(map)
        }
    })
}

/// Read and parse a `.jsonc` case file into the typed tree.
///
/// Unknown fields and unknown enum variants (an unrecognised `patches[].kind`,
/// a BC `type` this format does not define, ...) are the `#[serde(deny_unknown_fields)]`
/// and internally-tagged-enum machinery's own errors, which already name what
/// exists - the `SPEC-LIT` §13.4 diagnostic this reader owes a case with a
/// typo in it. [`serde_path_to_error`] adds the piece `serde` alone does not
/// give: WHERE in the file, as a JSON path (`patches[3].kind`), rather than
/// just what.
pub fn read_case_jsonc(path: &Path) -> Result<JsonCase> {
    let text = std::fs::read_to_string(path).path(path)?;

    let parsed = jsonc_parser::parse_to_value(&text, &parse_options())
        .map_err(|e| Error::Parse { path: path.display().to_string(), msg: e.to_string() })?;
    let Some(value) = parsed else {
        return Err(Error::Parse {
            path: path.display().to_string(),
            msg: "empty JSONC document".to_string(),
        });
    };

    let value = jsonc_to_serde(value)?;

    serde_path_to_error::deserialize(value).map_err(|e| {
        let json_path = e.path().to_string();
        Error::Parse {
            path: path.display().to_string(),
            msg: format!("{json_path}: {}", e.into_inner()),
        }
    })
}

// ==========================================================================
//  3. Lowering: the JSON tree -> the solver's own control types
// ==========================================================================

/// `numerics.run`, lowered. No existing type in the crate models "a plain
/// transient run's own end time and adjustable step" -
/// `crate::timescheme::LtsControls` is specifically `localEuler`'s pseudo-time
/// ceiling (`SPEC-LIT` §13.2) and would misname this - so this is a new,
/// small one rather than a strained reuse of that.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RunControl {
    pub end_time: Scalar,
    pub delta_t: Scalar,
    pub adjust_time_step: bool,
    /// Only meaningful when [`Self::adjust_time_step`] is set.
    pub max_co: Scalar,
}

/// A window region, resolved to cell indices against the block's own node
/// arrays - the geometric content of a [`PatchWindow`], kept alongside it
/// because `BlockSpec` carries only one `window` slot today and a case may
/// name more than one `mesh.regions` entry.
#[derive(Debug, Clone, PartialEq)]
pub struct WindowRegionSpec {
    pub name: String,
    pub on: String,
    pub window: PatchWindow,
}

/// A `volScalarField`'s worth of a case, before a mesh exists to size its
/// internal field: the uniform initial value plus one resolved
/// [`PatchFieldSpec`] per patch NAME actually present in the mesh (boundary
/// slots plus any window). Once a driver has `n_cells`, building the real
/// `crate::io::fields::RawScalarField` is `vec![internal_uniform; n_cells]`
/// plus this `boundary` map verbatim - `boundary_patterns` stays empty
/// because every pattern here has already been resolved to a concrete patch.
#[derive(Debug, Clone, PartialEq)]
pub struct LoweredScalarField {
    pub name: String,
    pub dimensions: String,
    pub internal_uniform: Scalar,
    pub boundary: BTreeMap<String, PatchFieldSpec>,
}

impl LoweredScalarField {
    /// The `n_cells`-sized [`crate::io::fields::RawScalarField`] this field's
    /// worth of a case becomes, once a mesh exists to size it - see this
    /// type's own doc comment. `boundary_patterns` stays empty: every pattern
    /// here has already been resolved to a concrete patch name by
    /// [`JsonCase::lower`].
    pub fn to_raw(&self, n_cells: usize) -> crate::io::fields::RawScalarField {
        crate::io::fields::RawScalarField {
            name: self.name.clone(),
            dimensions: self.dimensions.clone(),
            internal: vec![self.internal_uniform; n_cells],
            boundary: self.boundary.clone(),
            boundary_patterns: Vec::new(),
        }
    }
}

/// Vector counterpart of [`LoweredScalarField`], for `U`.
#[derive(Debug, Clone, PartialEq)]
pub struct LoweredVectorField {
    pub name: String,
    pub dimensions: String,
    pub internal_uniform: Vec3,
    pub boundary: BTreeMap<String, PatchFieldSpec>,
}

impl LoweredVectorField {
    /// Vector counterpart of [`LoweredScalarField::to_raw`].
    pub fn to_raw(&self, n_cells: usize) -> crate::io::fields::RawVectorField {
        crate::io::fields::RawVectorField {
            name: self.name.clone(),
            dimensions: self.dimensions.clone(),
            internal: vec![self.internal_uniform; n_cells],
            boundary: self.boundary.clone(),
            boundary_patterns: Vec::new(),
        }
    }
}

/// Everything [`JsonCase::lower`] produces. Deliberately not a mesh and not a
/// running solver - wiring either is the B3 agent's job.
pub struct LoweredCase {
    pub name: String,

    pub block: BlockSpec,
    /// Every `mesh.regions` entry, resolved. `block.window` is a COPY of
    /// `windows[0]` when there is exactly one (today's only case
    /// `BlockSpec` can express); a case with more than one window has already
    /// gone through the §13.4 contract by the time this is populated - see
    /// [`JsonCase::lower`].
    pub windows: Vec<WindowRegionSpec>,

    pub nu: Scalar,
    pub fluid: ScalarTransportCoeffs,
    pub buoyancy: BuoyancyCoeffs,
    pub wall: WallFunctionCoeffs,
    pub turbulence_model: Option<String>,

    /// `physics.fire.combustion`, resolved to [`CombustionCoeffs`] - `None`
    /// when the case has no `fire.combustion` block. Combustion is actually
    /// SOLVED only when [`Self::y_f_field`] is also `Some` - see
    /// [`JsonCase::lower`]'s cross-check between the two.
    pub combustion: Option<CombustionCoeffs>,
    /// `physics.fire.radiation`, resolved - `None` when the case has no
    /// `fire.radiation` block.
    pub radiation: Option<RadiationProps>,
    /// The Marshak wall emissivity every `wall`-kind patch gets when
    /// `radiation` is `Some`; meaningless otherwise.
    pub radiation_wall_emissivity: Scalar,

    pub turb: TurbulenceControls,
    pub algorithm: AlgorithmControls,
    pub p_solver: SolverControls,
    pub u_solver: SolverControls,
    /// `numerics.relaxation`, verbatim - a driver reading a field this crate
    /// has no dedicated relaxation slot for (a passive scalar, say) still
    /// gets its factor.
    pub relaxation: BTreeMap<String, Scalar>,
    /// `numerics.div`, parsed - `div_for("div(phi,U)")` falls back to the
    /// entry's own `"default"`, exactly as
    /// `crate::io::schemes::FvSchemes::div` falls back to
    /// `divSchemes/default`.
    div: BTreeMap<String, DivEntry>,
    pub grad: GradScheme,
    pub laplacian_sn_grad: SnGradScheme,

    pub run: RunControl,

    /// The ordered rules, verbatim - a driver building a field this layer did
    /// not (turbulence, say) still has the patch-kind and per-field data to
    /// build it from, resolved with the same first-match order via
    /// [`resolve_patch_rule`].
    pub patch_rules: Vec<JsonPatchRule>,

    pub u_field: LoweredVectorField,
    pub p_field: LoweredScalarField,
    /// `None` for an isothermal case - `initial.T` was absent.
    pub t_field: Option<LoweredScalarField>,
    /// `None` when the case does not solve `k` - `initial.k` was absent. Set
    /// together with [`Self::epsilon_field`] by a `kEpsilon` case; a
    /// `kOmega`/`kOmegaSST` one would set [`Self::omega_field`] instead.
    pub k_field: Option<LoweredScalarField>,
    pub epsilon_field: Option<LoweredScalarField>,
    pub omega_field: Option<LoweredScalarField>,
    /// `None` when the case gives no `nut` - the same "derive the wall
    /// treatment from the dissipation field instead" fallback
    /// `crate::field_setup::WallFaces::from_case` already applies to an
    /// OpenFOAM case with no `nut` file.
    pub nut_field: Option<LoweredScalarField>,

    /// `None` when the case has no `initial.Y_F` - see [`JsonInitial::y_f`].
    /// The species set combustion needs is exactly `{y_f, o2, products}`
    /// solved with `N2` as the inert closure (SPEC-LIT §27, §19) -
    /// `crate::species::Species::new` is given `["Y_F", "Y_O2", "Y_P", "N2"]`
    /// with `inert = "N2"` verbatim, never a case choice.
    pub y_f_field: Option<LoweredScalarField>,
    pub o2_field: Option<LoweredScalarField>,
    pub products_field: Option<LoweredScalarField>,

    pub output: Option<JsonOutput>,
}

impl LoweredCase {
    /// The `divSchemes` entry for `key` (`"div(phi,U)"`, ...), falling back to
    /// the case's own `"default"` and then to [`DivEntry::UPWIND`] if the case
    /// gave no default at all - which cannot happen through [`JsonCase::lower`]
    /// (`numerics.div.default` is required there) but keeps this fallible-free
    /// for a caller holding a `LoweredCase` built some other way.
    pub fn div_for(&self, key: &str) -> DivEntry {
        self.div
            .get(key)
            .or_else(|| self.div.get("default"))
            .copied()
            .unwrap_or(DivEntry::UPWIND)
    }

    /// The solver's own [`CaseControls`], field for field the same as
    /// [`crate::io::case::read_case_controls`] builds from an OpenFOAM case
    /// directory - this is the piece that lets a driver run from either
    /// format through the SAME `CaseControls`-shaped code from here on. Every
    /// field a JSONC case has no dictionary for (`momentum_transport`,
    /// `schemes`, `residual_control`, `lts`) keeps `CaseControls::default`'s
    /// value, which is exactly what an OpenFOAM case with no matching
    /// dictionary file gets too - see `read_case_controls`'s own "fails only
    /// if `case_dir` is not an OpenFOAM case; everything else falls back to
    /// the OpenFOAM default".
    pub fn to_case_controls(&self) -> CaseControls {
        let mut c = CaseControls {
            nu: self.nu,
            turb: self.turb,
            wall: self.wall,
            buoyancy: self.buoyancy,
            model_name: self.turbulence_model.clone().unwrap_or_default(),
            p_solver: self.p_solver,
            u_solver: self.u_solver,
            algorithm: self.algorithm,
            ..CaseControls::default()
        };
        if self.run.end_time > 0.0 {
            c.write_time = format_time_name(self.run.end_time);
        }
        c
    }
}

// -------------------------------------------------------------------- mesh

/// `-x +x -y +y -z +z`, `BlockSpec`'s own slot order.
const SLOT_NAMES: [&str; 6] = ["xmin", "xmax", "ymin", "ymax", "zmin", "zmax"];

fn slot_of(on: &str) -> Result<usize> {
    SLOT_NAMES
        .iter()
        .position(|s| *s == on)
        .ok_or_else(|| {
            Error::Config(format!(
                "mesh.regions[].on: \"{on}\" is not one of {}",
                SLOT_NAMES.join(", ")
            ))
        })
}

/// The two axis indices (0 = x, 1 = y, 2 = z) a slot's faces are spanned by,
/// in the `(fast, slow)` order `crate::blockgen::PatchWindow`'s own doc
/// comment gives for `boundary_quad`'s decomposition: `(j, k)` on `-x`/`+x`,
/// `(i, k)` on `-y`/`+y`, `(i, j)` on `-z`/`+z`.
fn tangential_axes(slot: usize) -> (usize, usize) {
    match slot {
        0 | 1 => (1, 2), // -x / +x : (y, z)
        2 | 3 => (0, 2), // -y / +y : (x, z)
        _ => (0, 1),     // -z / +z : (x, y)
    }
}

/// The half-open cell-index range along one axis whose cell CENTRES fall
/// inside `[lo, hi]` - the same criterion `crate::blockgen`'s own
/// `centred_cell_range` uses for the plume's burner window, generalised from
/// a centre-and-width pair to an explicit `[lo, hi]` because a JSON region is
/// given as a box, not a centre and a side length. Falls back to the single
/// cell nearest the box's own centre when the box is narrower than one cell,
/// so a thin window never silently disappears.
fn cell_range_from_bounds(nodes: &[Scalar], lo: Scalar, hi: Scalar) -> (usize, usize) {
    let n = nodes.len().saturating_sub(1);
    if n == 0 {
        return (0, 0);
    }
    let mid = |i: usize| 0.5 * (nodes[i] + nodes[i + 1]);

    let mut a = n;
    let mut b = 0;
    for i in 0..n {
        let c = mid(i);
        if c >= lo && c <= hi {
            a = a.min(i);
            b = i + 1;
        }
    }
    if a < b {
        return (a, b);
    }

    let centre = 0.5 * (lo + hi);
    let mut best = 0;
    let mut best_d = Scalar::MAX;
    for i in 0..n {
        let d = (mid(i) - centre).abs();
        if d < best_d {
            best_d = d;
            best = i;
        }
    }
    (best, best + 1)
}

/// OpenFOAM's own boundary-patch class (`wall`/`patch`) for one of this
/// format's `patches[].kind` presets - `crate::mesh::PatchKind::from_type`
/// is what reads it back.
fn patch_class(kind: PatchPresetKind) -> &'static str {
    match kind {
        PatchPresetKind::Wall => "wall",
        PatchPresetKind::Inlet | PatchPresetKind::Open => "patch",
    }
}

fn build_block(mesh: &JsonMesh, patches: &[JsonPatchRule]) -> Result<(BlockSpec, Vec<WindowRegionSpec>)> {
    let MeshKind::Cartesian = mesh.kind;

    let axis = |lo: f64, hi: f64, n: u32| GradedAxis {
        lo: lo as Scalar,
        hi: hi as Scalar,
        n: n as usize,
        expansion: 1.0,
        two_sided: false,
    };

    let mut block = BlockSpec {
        x: axis(mesh.bounds.min[0], mesh.bounds.max[0], mesh.cells[0]),
        y: axis(mesh.bounds.min[1], mesh.bounds.max[1], mesh.cells[1]),
        z: axis(mesh.bounds.min[2], mesh.bounds.max[2], mesh.cells[2]),
        ..BlockSpec::default()
    };

    let names = [
        mesh.boundaries.xmin.clone(),
        mesh.boundaries.xmax.clone(),
        mesh.boundaries.ymin.clone(),
        mesh.boundaries.ymax.clone(),
        mesh.boundaries.zmin.clone(),
        mesh.boundaries.zmax.clone(),
    ];
    for (slot, name) in names.iter().enumerate() {
        let rule = resolve_patch_rule(patches, name)?;
        block.patch_name[slot] = name.clone();
        block.patch_type[slot] = patch_class(rule.kind).to_string();
    }

    // Only one region reaches `BlockSpec::window` - it has one slot. Every
    // region is still resolved and returned in `windows`, so a case with more
    // than one is a §13.4 substitution (keep the first, warn or refuse) and
    // not a silent truncation.
    let mut windows = Vec::with_capacity(mesh.regions.len());
    for region in &mesh.regions {
        let slot = slot_of(&region.on)?;
        let JsonShape::Box { min, max } = &region.shape;
        let (fast, slow) = tangential_axes(slot);
        let axis_nodes = [
            crate::blockgen::graded_nodes(&block.x),
            crate::blockgen::graded_nodes(&block.y),
            crate::blockgen::graded_nodes(&block.z),
        ];
        let (lo0, hi0) = cell_range_from_bounds(&axis_nodes[fast], min[fast] as Scalar, max[fast] as Scalar);
        let (lo1, hi1) = cell_range_from_bounds(&axis_nodes[slow], min[slow] as Scalar, max[slow] as Scalar);

        let rule = resolve_patch_rule(patches, &region.name)?;
        let window = PatchWindow {
            slot,
            lo: [lo0, lo1],
            hi: [hi0, hi1],
            name: region.name.clone(),
            type_name: patch_class(rule.kind).to_string(),
        };
        windows.push(WindowRegionSpec { name: region.name.clone(), on: region.on.clone(), window });
    }

    if let Some(first) = windows.first() {
        if windows.len() > 1 {
            unsupported::<()>(
                "mesh.regions",
                &format!("{} regions", windows.len()),
                &["exactly one region (blockgen::BlockSpec has a single window slot)"],
                "the first region only; the rest are resolved but not carved",
                (),
            )?;
        }
        block.window = Some(first.window.clone());
    }

    Ok((block, windows))
}

// ------------------------------------------------------------ patch rules

/// Which rule of an ORDERED `patches`-shaped array governs `name`: the first
/// whose `match` (a POSIX ERE, anchored at both ends -
/// `crate::io::regex::Regex`) matches, in FILE order. This is the mechanism
/// `docs/05-io-redesign.md` section 4.1 point 1 asks for in place of a JSON
/// object's unordered keys.
pub fn resolve_patch_rule<'a>(rules: &'a [JsonPatchRule], name: &str) -> Result<&'a JsonPatchRule> {
    for rule in rules {
        let re = Regex::new(&rule.pattern).map_err(Error::Config)?;
        if re.is_match(name) {
            return Ok(rule);
        }
    }
    Err(Error::Config(format!(
        "patches: no rule matches boundary patch \"{name}\" - every case should \
         end with a catch-all, e.g. {{ \"match\": \".*\", \"kind\": \"wall\" }}"
    )))
}

fn spec(type_name: &str) -> PatchFieldSpec {
    PatchFieldSpec { type_name: type_name.to_string(), ..PatchFieldSpec::default() }
}

fn to_vec3(v: [f64; 3]) -> Vec3 {
    Vec3::new(v[0] as Scalar, v[1] as Scalar, v[2] as Scalar)
}

fn scalar_bc_spec(bc: &ScalarBc) -> PatchFieldSpec {
    match bc {
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
        ScalarBc::ZeroGradient {} => spec("zeroGradient"),
        ScalarBc::ThermalWallFunction { value } => {
            let mut s = spec("thermalWallFunction");
            s.value = vec![*value as Scalar];
            s
        }
    }
}

fn vector_bc_spec(bc: &VectorBc) -> PatchFieldSpec {
    match bc {
        VectorBc::FixedValue { value } => {
            let mut s = spec("fixedValue");
            s.value_v = vec![to_vec3(*value)];
            s
        }
        VectorBc::InletOutlet { inlet_value } => {
            let mut s = spec("inletOutlet");
            s.inlet_value_v = vec![to_vec3(*inlet_value)];
            s.value_v = vec![to_vec3(*inlet_value)];
            s
        }
        VectorBc::ZeroGradient {} => spec("zeroGradient"),
    }
}

/// `U` at one resolved rule: the rule's own `U` if it gave one, otherwise the
/// preset [`crate::blockgen::write_case`]'s `write_initial_fields` gives a
/// wall (`noSlip 0`) or an open boundary (`inletOutlet`, inlet value zero). An
/// `inlet` preset with no `U` has nothing to fall back to - a burner with no
/// velocity is a case error, not a default - so that is a §13.4-shaped error
/// naming the rule.
fn u_spec_for(rule: &JsonPatchRule) -> Result<PatchFieldSpec> {
    if let Some(bc) = &rule.u {
        return Ok(vector_bc_spec(bc));
    }
    match rule.kind {
        PatchPresetKind::Wall => {
            let mut s = spec("noSlip");
            s.value_v = vec![Vec3::ZERO];
            Ok(s)
        }
        PatchPresetKind::Open => {
            let mut s = spec("inletOutlet");
            s.inlet_value_v = vec![Vec3::ZERO];
            s.value_v = vec![Vec3::ZERO];
            Ok(s)
        }
        PatchPresetKind::Inlet => Err(Error::Config(format!(
            "patches: rule \"{}\" has kind \"inlet\" but no \"U\" condition; \
             an inlet needs a velocity",
            rule.pattern
        ))),
    }
}

/// `p` at one resolved rule: the rule's own `p` if given, otherwise
/// `zeroGradient` - the preset every kind gets in
/// `crate::blockgen::write_case`'s `write_initial_fields` unless the case
/// names the one Dirichlet face itself (the plume's `outlet`, here `open`
/// patches carrying an explicit `p`).
fn p_spec_for(rule: &JsonPatchRule) -> PatchFieldSpec {
    match &rule.p {
        Some(bc) => scalar_bc_spec(bc),
        None => spec("zeroGradient"),
    }
}

/// `T` at one resolved rule: the rule's own `T` if given; otherwise a wall
/// gets the effective `wallTreatment`'s thermal completion - SPEC-LIT §29.3:
/// `thermalWallFunction` for every row except `lowRe`, which is `zeroGradient`
/// (adiabatic, matching `write_initial_fields`'s floor and ceiling; `lowRe`
/// "pins the molecular resistance the resolved mesh already provides", which
/// on a mesh with no wall function at all IS the adiabatic default) - and an
/// `open` boundary is `inletOutlet` at `ambient` (the initial/ambient
/// temperature the case as a whole gave). An `inlet` preset with no `T` has no
/// honest ambient to fall back to when the case IS solving a temperature
/// equation, so that is an error naming the rule - same reasoning as
/// [`u_spec_for`]'s inlet-with-no-`U`.
fn t_spec_for(
    rule: &JsonPatchRule,
    ambient: Scalar,
    wall_treatment: crate::io::case::WallTreatment,
) -> Result<PatchFieldSpec> {
    if let Some(bc) = &rule.t {
        return Ok(scalar_bc_spec(bc));
    }
    match rule.kind {
        PatchPresetKind::Wall => Ok(spec(wall_treatment.thermal_type().unwrap_or("zeroGradient"))),
        PatchPresetKind::Open => {
            let mut s = spec("inletOutlet");
            s.inlet_value = vec![ambient];
            s.value = vec![ambient];
            Ok(s)
        }
        PatchPresetKind::Inlet => Err(Error::Config(format!(
            "patches: rule \"{}\" has kind \"inlet\" but no \"T\" condition, \
             and this case solves a temperature equation (initial.T is given)",
            rule.pattern
        ))),
    }
}

/// *DESIGN*: the ambient air composition, by mass, that every non-fuel
/// patch's `Y_O2`/`Y_P` fall back to. `0.232` is the standard mass fraction of
/// O2 in dry air; the case has no field for it because there is nothing to
/// choose - it is a physical constant of the ambient the case is already
/// immersed in, not a tunable.
pub const AMBIENT_Y_O2: Scalar = 0.232;

/// `Y_F` at one resolved rule - SPEC-LIT §27's "an inlet carrying `Y_F`".
/// Same fallback shape as [`t_spec_for`]: a wall is impermeable
/// (`zeroGradient`), an `open` boundary is `inletOutlet` at the ambient
/// `Y_F` (`initial.Y_F`, usually `0`), and an `inlet` with no `Y_F` condition
/// is an error - a fuel inlet with no fuel fraction is a case error, not a
/// default.
fn y_f_spec_for(rule: &JsonPatchRule, ambient: Scalar) -> Result<PatchFieldSpec> {
    if let Some(bc) = &rule.y_f {
        return Ok(scalar_bc_spec(bc));
    }
    match rule.kind {
        PatchPresetKind::Wall => Ok(spec("zeroGradient")),
        PatchPresetKind::Open => {
            let mut s = spec("inletOutlet");
            s.inlet_value = vec![ambient];
            s.value = vec![ambient];
            Ok(s)
        }
        PatchPresetKind::Inlet => Err(Error::Config(format!(
            "patches: rule \"{}\" has kind \"inlet\" but no \"Y_F\" condition, \
             and this case solves combustion (initial.Y_F is given)",
            rule.pattern
        ))),
    }
}

/// `Y_O2`/`Y_P` at one resolved rule. Neither is a field a case names
/// directly (see [`JsonPatchRule`]'s doc on `y_f`): a wall or open boundary
/// carries the ambient composition (`AMBIENT_Y_O2`/`0` respectively, the same
/// `inletOutlet`-at-ambient shape as `Y_F`), and a fuel `inlet` carries pure
/// fuel - `fixedValue 0` for both, since the same patch's `Y_F` is already
/// pinned by `fixedValue` to whatever the case named. *DESIGN*: a partially
/// diluted fuel stream (`Y_F < 1` at the inlet with nonzero `Y_O2`) is not
/// expressible this way; SPEC-LIT §27 only asks for a burner supplying fuel,
/// which this covers.
fn oxidiser_product_spec_for(rule: &JsonPatchRule, ambient: Scalar) -> PatchFieldSpec {
    match rule.kind {
        PatchPresetKind::Wall => spec("zeroGradient"),
        PatchPresetKind::Open => {
            let mut s = spec("inletOutlet");
            s.inlet_value = vec![ambient];
            s.value = vec![ambient];
            s
        }
        PatchPresetKind::Inlet => {
            let mut s = spec("fixedValue");
            s.value = vec![0.0];
            s
        }
    }
}

fn turb_bc_spec(bc: &TurbBc) -> PatchFieldSpec {
    match bc {
        TurbBc::FixedValue { value } => {
            let mut s = spec("fixedValue");
            s.value = vec![*value as Scalar];
            s
        }
        TurbBc::ZeroGradient {} => spec("zeroGradient"),
        TurbBc::Calculated { value } => {
            let mut s = spec("calculated");
            s.value = vec![*value as Scalar];
            s
        }
        TurbBc::KqRWallFunction { value } => {
            let mut s = spec("kqRWallFunction");
            s.value = vec![*value as Scalar];
            s
        }
        TurbBc::EpsilonWallFunction { value } => {
            let mut s = spec("epsilonWallFunction");
            s.value = vec![*value as Scalar];
            s
        }
        TurbBc::OmegaWallFunction { value } => {
            let mut s = spec("omegaWallFunction");
            s.value = vec![*value as Scalar];
            s
        }
        TurbBc::NutkWallFunction { value } => {
            let mut s = spec("nutkWallFunction");
            s.value = vec![*value as Scalar];
            s
        }
        TurbBc::NutUWallFunction { value } => {
            let mut s = spec("nutUWallFunction");
            s.value = vec![*value as Scalar];
            s
        }
        TurbBc::NutLowReWallFunction { value } => {
            let mut s = spec("nutLowReWallFunction");
            s.value = vec![*value as Scalar];
            s
        }
        TurbBc::KLowReWallFunction { value } => {
            let mut s = spec("kLowReWallFunction");
            s.value = vec![*value as Scalar];
            s
        }
    }
}

/// The `wallTreatment` preset row's entry for one turbulence field on a wall
/// patch - SPEC-LIT §29.1's table, expanded to a [`PatchFieldSpec`] the same
/// shape [`turb_bc_spec`] produces for an explicit one. `nut` under `rough`
/// carries `Ks`/`Cs` as [`PatchFieldSpec::extra`] entries, read the same way
/// an OpenFOAM `nutkRoughWallFunction` patch's would be
/// (`crate::field_setup::NutRoughness::from_case`).
fn wall_preset_bc_spec(
    treatment: crate::io::case::WallTreatment,
    roughness: Option<crate::io::case::Roughness>,
    field_name: &str,
) -> PatchFieldSpec {
    use crate::io::case::WallTreatment;

    let type_name = match field_name {
        "nut" => treatment.nut_type(),
        "k" => treatment.k_type(),
        "epsilon" => treatment.epsilon_type(),
        "omega" => treatment.omega_type(),
        _ => "zeroGradient",
    };
    let mut s = spec(type_name);
    if field_name == "nut" && treatment == WallTreatment::Rough {
        if let Some(r) = roughness {
            s.extra.insert("Ks".to_string(), r.ks.to_string());
            s.extra.insert("Cs".to_string(), r.cs.to_string());
        }
    }
    s
}

/// `k`/`epsilon`/`omega`/`nut` at one resolved rule - SPEC-LIT §29.1's three
/// routes, precedence most-specific-wins: the rule's own field-specific
/// [`TurbBc`] if it has one; otherwise, on a `wall`-kind patch, the
/// `wallTreatment` preset row (the rule's own `treatment` override if it has
/// one, else `case_default`, the case-level `turbulence.wallTreatment`).
/// A non-wall patch with neither is an error naming the rule - there is no
/// honest default for an inlet or open boundary's own turbulence quantity,
/// unlike a wall's (SPEC-LIT §29.1 exists precisely to give walls one).
fn turb_field_spec(
    case_default: WallTreatmentKind,
    rule: &JsonPatchRule,
    field_name: &str,
) -> Result<PatchFieldSpec> {
    let opt = match field_name {
        "k" => &rule.k,
        "epsilon" => &rule.epsilon,
        "omega" => &rule.omega,
        "nut" => &rule.nut,
        _ => unreachable!("turb_field_spec only called with k/epsilon/omega/nut"),
    };
    if let Some(bc) = opt {
        return Ok(turb_bc_spec(bc));
    }
    if rule.kind != PatchPresetKind::Wall {
        return Err(Error::Config(format!(
            "patches: rule \"{}\" has no \"{field_name}\" condition, \
             and this case solves {field_name} (initial.{field_name} is given)",
            rule.pattern
        )));
    }

    let treatment = rule.wall_treatment.unwrap_or(case_default).to_case();
    let setting = format!("patches: rule \"{}\"", rule.pattern);
    let roughness = crate::io::case::Roughness::resolve(
        treatment,
        rule.ks.map(|v| v as Scalar),
        rule.cs.map(|v| v as Scalar),
        &setting,
    )?;
    Ok(wall_preset_bc_spec(treatment, roughness, field_name))
}

/// Every patch NAME actually present in the mesh: the six boundary slots plus
/// any window - the keys [`LoweredScalarField::boundary`] /
/// [`LoweredVectorField::boundary`] must cover.
fn mesh_patch_names(block: &BlockSpec) -> Vec<String> {
    let mut names: Vec<String> = block.patch_name.to_vec();
    if let Some(w) = &block.window {
        names.push(w.name.clone());
    }
    names
}

// --------------------------------------------------------- numerics/controls

fn build_algorithm(a: &JsonAlgorithm, non_orth: usize) -> AlgorithmControls {
    AlgorithmControls {
        dict: match a.kind {
            AlgorithmKind::Simple => "SIMPLE",
            AlgorithmKind::Piso => "PISO",
            AlgorithmKind::Pimple => "PIMPLE",
        },
        consistent: false,
        n_correctors: a.correctors.unwrap_or(1),
        n_outer_correctors: a.outer_correctors.unwrap_or(1),
        momentum_predictor: true,
        n_non_orth_correctors: non_orth,
    }
}

/// One equation's solver settings out of the ORDERED `numerics.solvers`
/// array - first match wins, same mechanism as [`resolve_patch_rule`]. A
/// `var` no rule matches keeps [`SolverControls::default`], exactly as an
/// OpenFOAM case with no `solvers/<var>` entry at all does in
/// `crate::io::case::read_solver_controls`.
fn solver_for(solvers: &[JsonSolverRule], var: &str) -> Result<SolverControls> {
    for rule in solvers {
        let re = Regex::new(&rule.pattern).map_err(Error::Config)?;
        if !re.is_match(var) {
            continue;
        }
        let mut sc = SolverControls::default();
        sc.solver = LinearSolverKind::from_name(&rule.solver)?;
        sc.precon = Preconditioner::from_name(&rule.preconditioner)?;
        sc.tolerance = rule.tolerance as Scalar;
        sc.rel_tol = rule.rel_tol as Scalar;
        sc.max_iter = rule.max_iter;
        return Ok(sc);
    }
    Ok(SolverControls::default())
}

// ============================================================== JsonCase

impl JsonCase {
    /// Map the parsed tree onto the solver's own control types.
    pub fn lower(&self) -> Result<LoweredCase> {
        let (block, windows) = build_block(&self.mesh, &self.patches)?;

        // ---- physics --------------------------------------------------
        let nu = self.physics.fluid.nu as Scalar;
        let fluid = ScalarTransportCoeffs {
            pr: self.physics.fluid.pr as Scalar,
            prt: self.physics.fluid.prt as Scalar,
        };
        let buoyancy = match self.physics.buoyancy {
            BuoyancyModel::DensityRatio => BuoyancyCoeffs {
                g: to_vec3(self.physics.gravity),
                t_ref: self.physics.fluid.t_ref as Scalar,
                t_min: BuoyancyCoeffs::default().t_min,
            },
            BuoyancyModel::Boussinesq => unsupported(
                "physics.buoyancy",
                "boussinesq",
                &["densityRatio"],
                "densityRatio",
                BuoyancyCoeffs {
                    g: to_vec3(self.physics.gravity),
                    t_ref: self.physics.fluid.t_ref as Scalar,
                    t_min: BuoyancyCoeffs::default().t_min,
                },
            )?,
        };

        // ---- fire: combustion (S27) and radiation (S28) -----------------
        let combustion = self
            .physics
            .fire
            .as_ref()
            .and_then(|f| f.combustion.as_ref())
            .map(|c| {
                let mut coeffs = CombustionCoeffs::default();
                if let Some(v) = c.c_edm {
                    coeffs.c_edm = v as Scalar;
                }
                if let Some(v) = c.c_edm_les {
                    coeffs.c_edm_les = v as Scalar;
                }
                if let Some(v) = c.s {
                    coeffs.s = v as Scalar;
                }
                if let Some(v) = c.dh_c {
                    coeffs.dh_c = v as Scalar;
                }
                coeffs.validate()?;
                Ok::<_, Error>(coeffs)
            })
            .transpose()?;
        if combustion.is_some() && self.initial.y_f.is_none() {
            return Err(Error::Config(
                "physics.fire.combustion is given but initial.Y_F is not - \
                 combustion (SPEC-LIT S27) needs the species set built, and \
                 that needs an ambient Y_F to build it from"
                    .to_string(),
            ));
        }

        let (radiation, radiation_wall_emissivity) = match self.physics.fire.as_ref().and_then(|f| f.radiation.as_ref())
        {
            Some(r) => {
                RadiationModel::from_name(&r.model)?;
                let mut props = RadiationProps::new(r.absorption as Scalar)?;
                if let Some(v) = r.chi_r {
                    props.chi_r = v as Scalar;
                }
                props.validate()?;
                (Some(props), r.wall_emissivity.unwrap_or(1.0) as Scalar)
            }
            None => (None, 1.0),
        };

        // ---- turbulence -------------------------------------------------
        let mut wall = WallFunctionCoeffs::default();
        let turbulence_model = if let Some(t) = &self.turbulence {
            wall.kappa = t.wall_functions.kappa as Scalar;
            wall.e = t.wall_functions.e as Scalar;
            Some(t.model.clone())
        } else {
            None
        };
        wall.y_plus_lam = compute_y_plus_lam(wall.kappa, wall.e);
        // SPEC-LIT §29.1 route (b): the case-level default every wall
        // patch's per-field types (and, per §29.3, `T`'s) expand to unless a
        // patch or a field overrides it. `standard` when the case solves no
        // turbulence at all - a laminar case with walls still needs a `T`
        // completion if it carries a temperature equation.
        let wall_treatment_default = self
            .turbulence
            .as_ref()
            .map(|t| t.wall_treatment)
            .unwrap_or_default();

        let mut turb = TurbulenceControls::default();
        turb.k_solver = solver_for(&self.numerics.solvers, "k")?;
        turb.epsilon_solver = solver_for(&self.numerics.solvers, "epsilon")?;
        if let Some(v) = self.numerics.relaxation.get("k") {
            turb.k_relax = *v as Scalar;
        }
        if let Some(v) = self
            .numerics
            .relaxation
            .get("epsilon")
            .or_else(|| self.numerics.relaxation.get("omega"))
        {
            turb.eps_relax = *v as Scalar;
        }

        // ---- numerics: schemes ------------------------------------------
        let mut div: BTreeMap<String, DivEntry> = BTreeMap::new();
        for (key, raw) in &self.numerics.div {
            div.insert(key.clone(), parse_div(&format!("numerics.div.{key}"), raw)?);
        }
        if !div.contains_key("default") {
            return Err(Error::Config(
                "numerics.div: missing the required \"default\" entry".to_string(),
            ));
        }
        let default_div = div["default"];

        // `TurbulenceControls::div_scheme`/`bounded_convection` are the `k`
        // EQUATION's own convection scheme, not the momentum equation's -
        // `crate::io::case::read_fv_schemes`'s own comment says so
        // explicitly ("every equation reads ITS OWN entry"), and its code
        // reads `div(phi,k)` here, never `div(phi,U)`. An earlier version of
        // this function read `div(phi,U)` instead, which the B3 phase-1 gate
        // (`docs/05-io-redesign.md`) caught: `cases/plume.jsonc` and
        // `cases/plumeB` converged to visibly different `k`/`epsilon` because
        // the k equation was being assembled with U's scheme
        // (`linearUpwind`, unbounded) instead of its own
        // (`bounded upwind`) - a different number of matrix coefficients
        // from the very first outer iteration. `div(phi,U)` itself is still
        // available to a driver that solves momentum, through
        // `LoweredCase::div_for("div(phi,U)")`; it was never right for this
        // field.
        let kk = div.get("div(phi,k)").copied().unwrap_or(default_div);
        turb.div_scheme = kk.scheme;
        turb.bounded_convection = kk.bounded;

        // epsilon and omega never coexist - same reasoning and same
        // fallback order as `read_fv_schemes`'s `eps_key`: the key that is
        // PRESENT decides which is looked up, `div(phi,epsilon)` when
        // neither is (so a case solving neither still gets a real key name
        // in its own error rather than a silently-picked default).
        let eps_key = if div.contains_key("div(phi,omega)") && !div.contains_key("div(phi,epsilon)")
        {
            "div(phi,omega)"
        } else {
            "div(phi,epsilon)"
        };
        let ee = div.get(eps_key).copied().unwrap_or(default_div);
        turb.eps_div_scheme = ee.scheme;
        turb.eps_bounded_convection = ee.bounded;

        let grad = parse_grad("numerics.grad", &self.numerics.grad)?;
        turb.grad_scheme = grad;

        let laplacian_sn_grad =
            parse_sn_grad("numerics.laplacian.snGrad", &self.numerics.laplacian.sn_grad)?;
        turb.sn_grad = laplacian_sn_grad;
        turb.n_non_orth_correctors = self.numerics.laplacian.non_orthogonal_correctors;

        let ddt = DdtScheme::parse(&self.numerics.ddt)?;
        turb.ddt = ddt;
        turb.steady = ddt.is_steady();

        let algorithm = build_algorithm(&self.numerics.algorithm, turb.n_non_orth_correctors);

        let p_solver = solver_for(&self.numerics.solvers, "p")?;
        let u_solver = solver_for(&self.numerics.solvers, "U")?;

        // ---- run ----------------------------------------------------------
        turb.delta_t = self.run.delta_t as Scalar;
        if self.run.end_time > 0.0 {
            turb.n_outer_iterations = if turb.steady {
                self.run.end_time as Label
            } else {
                (self.run.end_time / self.run.delta_t + 0.5) as Label
            };
        }
        let run = RunControl {
            end_time: self.run.end_time as Scalar,
            delta_t: self.run.delta_t as Scalar,
            adjust_time_step: self.run.adjust_time_step,
            max_co: self.run.max_co.unwrap_or(0.0) as Scalar,
        };

        // ---- fields ---------------------------------------------------
        let names = mesh_patch_names(&block);
        let ambient_t = self.initial.t.map(|v| v as Scalar).unwrap_or(0.0);

        let mut u_boundary = BTreeMap::new();
        let mut p_boundary = BTreeMap::new();
        let mut t_boundary: BTreeMap<String, PatchFieldSpec> = BTreeMap::new();
        for name in &names {
            let rule = resolve_patch_rule(&self.patches, name)?;
            u_boundary.insert(name.clone(), u_spec_for(rule)?);
            p_boundary.insert(name.clone(), p_spec_for(rule));
            if self.initial.t.is_some() {
                let treatment = rule.wall_treatment.unwrap_or(wall_treatment_default).to_case();
                t_boundary.insert(name.clone(), t_spec_for(rule, ambient_t, treatment)?);
            }
        }

        let u_field = LoweredVectorField {
            name: "U".to_string(),
            dimensions: "[0 1 -1 0 0 0 0]".to_string(),
            internal_uniform: to_vec3(self.initial.u),
            boundary: u_boundary,
        };
        let p_field = LoweredScalarField {
            name: "p".to_string(),
            dimensions: "[0 2 -2 0 0 0 0]".to_string(),
            internal_uniform: self.initial.p as Scalar,
            boundary: p_boundary,
        };
        let t_field = self.initial.t.map(|t| LoweredScalarField {
            name: "T".to_string(),
            dimensions: "[0 0 0 1 0 0 0]".to_string(),
            internal_uniform: t as Scalar,
            boundary: t_boundary,
        });

        // ---- species (SPEC-LIT S19/S27) --------------------------------
        let (y_f_field, o2_field, products_field) = if let Some(y_f0) = self.initial.y_f {
            let mut y_f_boundary = BTreeMap::new();
            let mut o2_boundary = BTreeMap::new();
            let mut p_boundary_sp = BTreeMap::new();
            for name in &names {
                let rule = resolve_patch_rule(&self.patches, name)?;
                y_f_boundary.insert(name.clone(), y_f_spec_for(rule, y_f0 as Scalar)?);
                o2_boundary.insert(name.clone(), oxidiser_product_spec_for(rule, AMBIENT_Y_O2));
                p_boundary_sp.insert(name.clone(), oxidiser_product_spec_for(rule, 0.0));
            }
            let dims = "[0 0 0 0 0 0 0]"; // mass fraction, dimensionless
            (
                Some(LoweredScalarField {
                    name: "Y_F".to_string(),
                    dimensions: dims.to_string(),
                    internal_uniform: y_f0 as Scalar,
                    boundary: y_f_boundary,
                }),
                Some(LoweredScalarField {
                    name: "Y_O2".to_string(),
                    dimensions: dims.to_string(),
                    internal_uniform: AMBIENT_Y_O2,
                    boundary: o2_boundary,
                }),
                Some(LoweredScalarField {
                    name: "Y_P".to_string(),
                    dimensions: dims.to_string(),
                    internal_uniform: 0.0,
                    boundary: p_boundary_sp,
                }),
            )
        } else {
            (None, None, None)
        };

        // ---- turbulence closure fields ---------------------------------
        // Dimensions match what `crate::field_setup`'s own drivers write
        // back out - see e.g. `src/bin/k_epsilon.rs`'s `out_k.dimensions`.
        let turb_field = |value: Option<f64>, name: &str, dims: &str| -> Result<Option<LoweredScalarField>> {
            let Some(v0) = value else { return Ok(None) };
            let mut boundary = BTreeMap::new();
            for pname in &names {
                let rule = resolve_patch_rule(&self.patches, pname)?;
                boundary.insert(pname.clone(), turb_field_spec(wall_treatment_default, rule, name)?);
            }
            Ok(Some(LoweredScalarField {
                name: name.to_string(),
                dimensions: dims.to_string(),
                internal_uniform: v0 as Scalar,
                boundary,
            }))
        };

        let mut k_field = turb_field(self.initial.k, "k", "[0 2 -2 0 0 0 0]")?;
        let mut epsilon_field = turb_field(self.initial.epsilon, "epsilon", "[0 2 -3 0 0 0 0]")?;
        let mut omega_field = turb_field(self.initial.omega, "omega", "[0 0 -1 0 0 0 0]")?;
        let mut nut_field = turb_field(self.initial.nut, "nut", "[0 2 -1 0 0 0 0]")?;

        // ---- SPEC-LIT §29.1 point 3: the consistency contract -------------
        //
        // Wired here because this is the one place all four turbulence
        // fields' resolved patch types are available together, after every
        // route above (explicit per-field, per-patch `treatment`, case
        // default) has already had its say on each field independently.
        for pname in &names {
            let rule = resolve_patch_rule(&self.patches, pname)?;
            if rule.kind != PatchPresetKind::Wall {
                continue;
            }

            let kind_of = |f: &Option<LoweredScalarField>, field: &str| -> Result<Option<BcKind>> {
                match f {
                    Some(lf) => {
                        let s = lf
                            .boundary
                            .get(pname)
                            .expect("every patch name was inserted into this field above");
                        Ok(Some(BcKind::from_name(&s.type_name, field, pname)?))
                    }
                    None => Ok(None),
                }
            };

            let nut_kind = kind_of(&nut_field, "nut")?;
            let k_kind = kind_of(&k_field, "k")?;
            let epsilon_kind = kind_of(&epsilon_field, "epsilon")?;
            let omega_kind = kind_of(&omega_field, "omega")?;

            let corrected = crate::field_setup::validate_wall_row(
                pname,
                crate::field_setup::WallRow {
                    nut: nut_kind,
                    k: k_kind,
                    epsilon: epsilon_kind,
                    omega: omega_kind,
                },
            )?;

            let write_back =
                |f: &mut Option<LoweredScalarField>, before: Option<BcKind>, after: Option<BcKind>| {
                    if before == after {
                        return;
                    }
                    if let (Some(lf), Some(k)) = (f.as_mut(), after) {
                        lf.boundary.get_mut(pname).expect("present above").type_name =
                            crate::field_setup::bc_kind_name(k).to_string();
                    }
                };
            write_back(&mut nut_field, nut_kind, corrected.nut);
            write_back(&mut k_field, k_kind, corrected.k);
            write_back(&mut epsilon_field, epsilon_kind, corrected.epsilon);
            write_back(&mut omega_field, omega_kind, corrected.omega);
        }

        Ok(LoweredCase {
            name: self.name.clone(),
            block,
            windows,
            nu,
            fluid,
            buoyancy,
            wall,
            turbulence_model,
            combustion,
            radiation,
            radiation_wall_emissivity,
            turb,
            algorithm,
            p_solver,
            u_solver,
            relaxation: self
                .numerics
                .relaxation
                .iter()
                .map(|(k, v)| (k.clone(), *v as Scalar))
                .collect(),
            div,
            grad,
            laplacian_sn_grad,
            run,
            patch_rules: self.patches.clone(),
            u_field,
            p_field,
            t_field,
            k_field,
            epsilon_field,
            omega_field,
            nut_field,
            y_f_field,
            o2_field,
            products_field,
            output: self.output.clone(),
        })
    }
}

// ==========================================================================
//  4. Schema
// ==========================================================================

/// The JSON Schema for [`JsonCase`], generated from these same types by
/// `schemars` - `docs/05-io-redesign.md` section 4.1's whole reason for using
/// `schemars` rather than hand-writing one, since the two cannot then
/// disagree.
pub fn emit_schema() -> String {
    let schema = schemars::schema_for!(JsonCase);
    serde_json::to_string_pretty(&schema).unwrap_or_default()
}

// ==========================================================================
//  Tests
// ==========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn example_path() -> std::path::PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../docs/case-example.json")
    }

    #[test]
    fn case_example_parses_and_lowers() {
        let case = read_case_jsonc(&example_path()).expect("docs/case-example.json should parse");
        assert_eq!(case.name, "plumeB");
        assert_eq!(case.mesh.cells, [98, 42, 20]);

        let lowered = case.lower().expect("docs/case-example.json should lower");
        assert_eq!(lowered.block.x.n, 98);
        assert_eq!(lowered.block.y.n, 42);
        assert_eq!(lowered.block.z.n, 20);
        assert_eq!(lowered.block.patch_name[4], "floor");
        assert_eq!(lowered.block.patch_type[4], "wall");
        assert!(lowered.block.window.is_some(), "the burner window should be carved");
        let w = lowered.block.window.as_ref().unwrap();
        assert_eq!(w.name, "inlet");
        assert_eq!(w.type_name, "patch");

        // p solves PCG+DIC by exact match; everything else falls to the
        // catch-all PBiCGStab+DILU - first-match-wins on the ORDERED array.
        assert_eq!(lowered.p_solver.solver, crate::io::case::LinearSolverKind::PCG);
        assert_eq!(lowered.u_solver.solver, crate::io::case::LinearSolverKind::PBiCGStab);

        assert!(lowered.t_field.is_some(), "the plume case solves T");
        assert!((lowered.nu - 1.5e-5).abs() < 1e-12);
    }

    /// The B3 phase-1 gate's mesh precondition: `cases/plume.jsonc`'s
    /// in-memory mesh ([`crate::blockgen::build_mesh`], no disk polyMesh at
    /// all) must be BIT-IDENTICAL to `cases/plumeB`'s on-disk one (written by
    /// `ofgpu-generate-mesh plume ../cases/plumeB 98 42 20`, read back
    /// through the ordinary `read_poly_mesh` + `build_host_mesh` path). This
    /// is what makes the driver-level comparison
    /// (`ofgpu_k_epsilon_produces_bit_identical_fields_from_either_format`,
    /// run by the phase-1 gate script) meaningful rather than coincidental:
    /// if the mesh geometry itself is not identical, no field comparison
    /// downstream of it can be either.
    ///
    /// This test is also what caught the actual bug the gate exists to find:
    /// `TextOut::real`'s original 15-significant-digit points format did not
    /// round-trip an `f64` exactly for most nodes of this mesh's x and y
    /// axes (98 and 42 cells - see `blockgen.rs`'s `real` for the fix, 17
    /// digits).
    #[test]
    fn plume_jsonc_mesh_matches_the_generated_openfoam_case_exactly() {
        let jsonc_path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../cases/plume.jsonc");
        let case = read_case_jsonc(&jsonc_path).expect("cases/plume.jsonc should parse");
        let lowered = case.lower().expect("cases/plume.jsonc should lower");
        let direct = crate::blockgen::build_mesh(&lowered.block).expect("build_mesh");

        let plumeb_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../cases/plumeB");
        let raw = crate::io::polymesh::read_poly_mesh(&plumeb_dir)
            .expect("read cases/plumeB - run `ofgpu-generate-mesh plume ../cases/plumeB 98 42 20` first");
        let via_disk = crate::io::polymesh::build_host_mesh(&raw).expect("build_host_mesh");

        assert_eq!(direct.n_cells, via_disk.n_cells);
        assert_eq!(direct.n_internal_faces, via_disk.n_internal_faces);
        assert_eq!(direct.n_boundary_faces, via_disk.n_boundary_faces);
        assert_eq!(direct.owner, via_disk.owner, "face-owner addressing must match exactly");
        assert_eq!(direct.neighbour, via_disk.neighbour, "face-neighbour addressing must match exactly");
        assert_eq!(direct.v, via_disk.v, "cell volumes must be bit-identical");
        assert_eq!(direct.c, via_disk.c, "cell centres must be bit-identical");
        assert_eq!(direct.sf, via_disk.sf, "internal face area vectors must be bit-identical");
        assert_eq!(direct.mag_sf, via_disk.mag_sf, "internal face areas must be bit-identical");
        assert_eq!(direct.cf, via_disk.cf, "internal face centres must be bit-identical");
        assert_eq!(direct.weights, via_disk.weights, "interpolation weights must be bit-identical");
        assert_eq!(direct.delta_coeffs, via_disk.delta_coeffs, "delta coefficients must be bit-identical");
        assert_eq!(direct.b_face_cells, via_disk.b_face_cells);
        assert_eq!(direct.b_sf, via_disk.b_sf, "boundary face area vectors must be bit-identical");
        assert_eq!(direct.b_mag_sf, via_disk.b_mag_sf);
        assert_eq!(direct.b_cf, via_disk.b_cf, "boundary face centres must be bit-identical");
        assert_eq!(direct.b_delta_coeffs, via_disk.b_delta_coeffs);

        // Patch identity, not just geometry: same names, same kind, same
        // face ranges, in the same order.
        assert_eq!(direct.patches.len(), via_disk.patches.len());
        for (a, b) in direct.patches.iter().zip(via_disk.patches.iter()) {
            assert_eq!(a.name, b.name);
            assert_eq!(a.kind, b.kind);
            assert_eq!(a.start, b.start);
            assert_eq!(a.size, b.size);
        }
    }

    // ------------------------------------------------------------------
    //  SPEC-LIT §29.1: wallTreatment presets (route b, JSONC)
    // ------------------------------------------------------------------

    fn plume_case() -> JsonCase {
        let jsonc_path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../cases/plume.jsonc");
        read_case_jsonc(&jsonc_path).expect("cases/plume.jsonc should parse")
    }

    /// The wall rule in `cases/plume.jsonc` (`{"match": ".*", "kind": "wall"}`)
    /// carries explicit k/epsilon/omega/nut of its own; the preset tests want
    /// to see what the CASE DEFAULT (or a patch's own `treatment`) expands to
    /// instead, so this clears them.
    fn wall_rule_mut(case: &mut JsonCase) -> &mut JsonPatchRule {
        case.patches
            .iter_mut()
            .find(|r| r.kind == PatchPresetKind::Wall)
            .expect("plume.jsonc has a wall rule")
    }

    fn clear_explicit_turbulence(rule: &mut JsonPatchRule) {
        rule.k = None;
        rule.epsilon = None;
        rule.omega = None;
        rule.nut = None;
    }

    /// Any patch name the `".*"` wall rule governs - every wall-kind field's
    /// boundary map carries the same row for it since none of the rules
    /// carry a per-patch override in these tests.
    fn a_wall_patch_name(lowered: &LoweredCase) -> String {
        mesh_patch_names(&lowered.block)
            .into_iter()
            .find(|n| resolve_patch_rule(&lowered.patch_rules, n).unwrap().kind == PatchPresetKind::Wall)
            .expect("plume.jsonc has at least one wall patch")
    }

    /// Each preset expands to exactly its row - SPEC-LIT §29.1's table,
    /// string-level, through the JSONC lowering path.
    #[test]
    fn each_preset_expands_to_exactly_its_row_through_jsonc() {
        for (wt, nut_want, k_want, eps_want, omega_want) in [
            (WallTreatmentKind::Standard, "nutkWallFunction", "kqRWallFunction", "epsilonWallFunction", "omegaWallFunction"),
            (WallTreatmentKind::Spalding, "nutUWallFunction", "kqRWallFunction", "epsilonWallFunction", "omegaWallFunction"),
            (WallTreatmentKind::Rough, "nutkRoughWallFunction", "kqRWallFunction", "epsilonWallFunction", "omegaWallFunction"),
            (WallTreatmentKind::LowRe, "nutLowReWallFunction", "kLowReWallFunction", "zeroGradient", "zeroGradient"),
        ] {
            let mut case = plume_case();
            case.turbulence.as_mut().unwrap().wall_treatment = wt;
            {
                let r = wall_rule_mut(&mut case);
                clear_explicit_turbulence(r);
                if wt == WallTreatmentKind::Rough {
                    r.ks = Some(0.001);
                }
            }
            let lowered = case.lower().unwrap_or_else(|e| panic!("{wt:?} should lower: {e}"));
            let p = a_wall_patch_name(&lowered);

            let ty = |f: &Option<LoweredScalarField>| f.as_ref().unwrap().boundary[&p].type_name.clone();
            assert_eq!(ty(&lowered.nut_field), nut_want, "{wt:?} nut");
            assert_eq!(ty(&lowered.k_field), k_want, "{wt:?} k");
            assert_eq!(ty(&lowered.epsilon_field), eps_want, "{wt:?} epsilon");
            assert_eq!(ty(&lowered.omega_field), omega_want, "{wt:?} omega");
        }
    }

    /// §29.3 through T's own row: every preset except `lowRe` defaults `T` on
    /// a wall to `thermalWallFunction`; `lowRe` stays adiabatic.
    #[test]
    fn thermal_wall_function_follows_the_same_row_except_low_re() {
        for (wt, want) in [
            (WallTreatmentKind::Standard, "thermalWallFunction"),
            (WallTreatmentKind::Spalding, "thermalWallFunction"),
            (WallTreatmentKind::Rough, "thermalWallFunction"),
            (WallTreatmentKind::LowRe, "zeroGradient"),
        ] {
            let mut case = plume_case();
            case.turbulence.as_mut().unwrap().wall_treatment = wt;
            {
                let r = wall_rule_mut(&mut case);
                clear_explicit_turbulence(r);
                r.t = None;
                if wt == WallTreatmentKind::Rough {
                    r.ks = Some(0.001);
                }
            }
            let lowered = case.lower().unwrap_or_else(|e| panic!("{wt:?} should lower: {e}"));
            let p = a_wall_patch_name(&lowered);
            assert_eq!(lowered.t_field.unwrap().boundary[&p].type_name, want, "{wt:?}");
        }
    }

    /// The gap `thermal_wall_function_follows_the_same_row_except_low_re`
    /// does NOT cover: that row's auto-completion never has a wall
    /// temperature of its own, so a wall left to the case default gets
    /// whatever the neighbour cell reads (fine for an adiabatic-ish floor,
    /// useless for a genuinely HOT wall). `ScalarBc::ThermalWallFunction`
    /// exists so a case can name `T_w` explicitly - highest precedence,
    /// SPEC-LIT §29.1's table - and this is the round trip: the type stays
    /// `thermalWallFunction` (not demoted to plain `fixedValue`, the way an
    /// explicit `ScalarBc::FixedValue` deliberately would be) and the value
    /// written is exactly the `T_w` given, on every wall-treatment row.
    #[test]
    fn explicit_thermal_wall_function_carries_its_own_t_w() {
        for wt in [
            WallTreatmentKind::Standard,
            WallTreatmentKind::Spalding,
            WallTreatmentKind::Rough,
            WallTreatmentKind::LowRe,
        ] {
            let mut case = plume_case();
            case.turbulence.as_mut().unwrap().wall_treatment = wt;
            {
                let r = wall_rule_mut(&mut case);
                clear_explicit_turbulence(r);
                r.t = Some(ScalarBc::ThermalWallFunction { value: 400.0 });
                if wt == WallTreatmentKind::Rough {
                    r.ks = Some(0.001);
                }
            }
            let lowered = case.lower().unwrap_or_else(|e| panic!("{wt:?} should lower: {e}"));
            let p = a_wall_patch_name(&lowered);
            let t = lowered.t_field.unwrap();
            assert_eq!(t.boundary[&p].type_name, "thermalWallFunction", "{wt:?}");
            assert_eq!(t.boundary[&p].value, vec![400.0 as Scalar], "{wt:?}");
        }
    }

    /// `rough` needs `Ks` - naming it, whether the treatment came from the
    /// case default or a patch's own `treatment` override.
    #[test]
    fn rough_without_ks_through_jsonc_is_an_error_naming_it() {
        let mut case = plume_case();
        case.turbulence.as_mut().unwrap().wall_treatment = WallTreatmentKind::Rough;
        clear_explicit_turbulence(wall_rule_mut(&mut case));
        let err = match case.lower() {
            Err(e) => e,
            Ok(_) => panic!("rough with no Ks must be refused"),
        };
        assert!(err.to_string().contains("Ks"), "{err}");
    }

    /// Precedence: an explicit `nutUWallFunction` on one patch wins on that
    /// patch only, even though the case default is `standard`.
    #[test]
    fn explicit_nutu_under_a_standard_default_wins_on_its_own_patch_only() {
        let mut case = plume_case();
        // Case default stays `standard` (the file names none).
        assert_eq!(case.turbulence.as_ref().unwrap().wall_treatment, WallTreatmentKind::Standard);

        // Split the wall rule into two: an explicit override on one named
        // patch, the untouched catch-all everywhere else.
        let wall_idx = case
            .patches
            .iter()
            .position(|r| r.kind == PatchPresetKind::Wall)
            .unwrap();
        let mut overridden = case.patches[wall_idx].clone();
        overridden.pattern = "floor".to_string();
        overridden.nut = Some(TurbBc::NutUWallFunction { value: 0.0 });
        case.patches.insert(wall_idx, overridden);

        let lowered = case.lower().expect("should lower");
        assert_eq!(
            lowered.nut_field.as_ref().unwrap().boundary["floor"].type_name,
            "nutUWallFunction",
            "the override must win on its own patch"
        );
        // Every other wall patch keeps the case default's row, untouched by
        // the override on `floor`.
        let other = a_wall_patch_name(&lowered);
        assert_ne!(other, "floor");
        assert_eq!(
            lowered.nut_field.as_ref().unwrap().boundary[&other].type_name,
            "nutkWallFunction"
        );
    }

    /// The consistency contract, wired into the JSONC path: an explicit
    /// `nutLowReWallFunction` on one patch, left to inherit `epsilon` from a
    /// `standard` case default, is SPEC-LIT §29.1's own contradiction and is
    /// refused naming both types.
    #[test]
    fn jsonc_wall_row_contradiction_is_refused() {
        let _g = crate::io::contract::permissive_test_guard();
        crate::io::contract::set_permissive(false);

        let mut case = plume_case();
        {
            // `k` agrees with `nut` (both `lowRe`); `epsilon`/`omega` fall to
            // the case default (`standard`, `*WallFunction`) and disagree -
            // the exact contradiction SPEC-LIT §29.1 names.
            let r = wall_rule_mut(&mut case);
            r.nut = Some(TurbBc::NutLowReWallFunction { value: 0.0 });
            r.k = Some(TurbBc::KLowReWallFunction { value: 0.0 });
            r.epsilon = None;
            r.omega = None;
        }
        let err = match case.lower() {
            Err(e) => e,
            Ok(_) => panic!("nutLowRe + epsilonWallFunction must be refused"),
        };
        let msg = err.to_string();
        assert!(msg.contains("nutLowReWallFunction"), "{msg}");
        assert!(msg.contains("epsilonWallFunction"), "{msg}");
    }

    /// `-permissive` resolves the same contradiction to the `lowRe` row and
    /// prints the substitution, rather than refusing the case outright.
    #[test]
    fn jsonc_wall_row_contradiction_is_resolved_under_permissive() {
        let _g = crate::io::contract::permissive_test_guard();
        crate::io::contract::reset_warnings();
        crate::io::contract::set_permissive(true);

        let mut case = plume_case();
        {
            let r = wall_rule_mut(&mut case);
            r.nut = Some(TurbBc::NutLowReWallFunction { value: 0.0 });
            r.k = Some(TurbBc::KLowReWallFunction { value: 0.0 });
            r.epsilon = None;
            r.omega = None;
        }
        let lowered = case.lower().expect("-permissive resolves it");
        let p = a_wall_patch_name(&lowered);
        assert_eq!(
            lowered.epsilon_field.unwrap().boundary[&p].type_name,
            "zeroGradient",
            "epsilon must be corrected to the row nut implied"
        );

        crate::io::contract::set_permissive(false);
    }

    /// The schema names all four presets - deliverable 4 of SPEC-LIT §29.1's
    /// JSONC route.
    #[test]
    fn schema_names_the_four_wall_treatment_presets() {
        let text = emit_schema();
        for name in ["standard", "spalding", "rough", "lowRe"] {
            assert!(text.contains(&format!("\"{name}\"")), "schema missing {name}: {text}");
        }
    }

    #[test]
    fn ordered_patch_resolution_is_first_match_not_best_match() {
        let rules = vec![
            JsonPatchRule {
                pattern: "inlet".to_string(),
                kind: PatchPresetKind::Inlet,
                u: Some(VectorBc::FixedValue { value: [0.0, 0.0, 1.0] }),
                p: None,
                t: None,
                y_f: None,
                turbulence: None,
                k: None,
                epsilon: None,
                omega: None,
                nut: None,
                wall_treatment: None,
                ks: None,
                cs: None,
            },
            JsonPatchRule {
                pattern: ".*".to_string(),
                kind: PatchPresetKind::Wall,
                u: None,
                p: None,
                t: None,
                y_f: None,
                turbulence: None,
                k: None,
                epsilon: None,
                omega: None,
                nut: None,
                wall_treatment: None,
                ks: None,
                cs: None,
            },
        ];

        // "inlet" matches BOTH the exact rule and the catch-all; the first
        // rule in file order must win.
        let r = resolve_patch_rule(&rules, "inlet").unwrap();
        assert_eq!(r.kind, PatchPresetKind::Inlet);

        // Anything else only matches the catch-all.
        let r = resolve_patch_rule(&rules, "floor").unwrap();
        assert_eq!(r.kind, PatchPresetKind::Wall);

        // Swap the order: now the catch-all governs "inlet" too, because it
        // comes first in the file.
        let mut swapped = rules;
        swapped.swap(0, 1);
        let r = resolve_patch_rule(&swapped, "inlet").unwrap();
        assert_eq!(r.kind, PatchPresetKind::Wall);
    }

    #[test]
    fn unknown_key_error_names_the_json_path() {
        let text = r#"{
            "name": "bad",
            "mesh": {
                "kind": "cartesian",
                "bounds": { "min": [0,0,0], "max": [1,1,1] },
                "cells": [1,1,1],
                "boundaries": {
                    "xmin": "a", "xmax": "a", "ymin": "a",
                    "ymax": "a", "zmin": "a", "zmax": "a"
                },
                "regions": []
            },
            "physics": {
                "gravity": [0,0,0],
                "fluid": { "nu": 1e-5, "Pr": 0.71, "Prt": 0.85, "TRef": 293.15 },
                "buoyancy": "densityRatio"
            },
            "patches": [
                { "match": ".*", "kind": "wall", "bogusField": 1 }
            ],
            "initial": { "U": [0,0,0], "p": 0.0 },
            "numerics": {
                "algorithm": { "kind": "SIMPLE" },
                "ddt": "steadyState",
                "div": { "default": "Gauss upwind" },
                "grad": "Gauss linear",
                "laplacian": { "snGrad": "corrected", "nonOrthogonalCorrectors": 0 },
                "solvers": []
            },
            "run": { "endTime": 1.0, "deltaT": 1.0 }
        }"#;

        let path = std::env::temp_dir().join("case_json_unknown_key_test.jsonc");
        std::fs::write(&path, text).unwrap();
        let err = read_case_jsonc(&path).unwrap_err().to_string();
        let _ = std::fs::remove_file(&path);

        // The JSON path names exactly where the bad key is, not just that one
        // exists somewhere in the file.
        assert!(err.contains("patches[0]"), "{err}");
        assert!(err.contains("bogusField"), "{err}");
    }

    #[test]
    fn schema_uses_one_of_const_for_bc_types() {
        let text = emit_schema();
        let value: serde_json::Value = serde_json::from_str(&text).unwrap();

        // schemars renders a `#[serde(tag = "type")]` enum as a `oneOf` whose
        // branches each pin the tag with `const` - the machine-checkable half
        // of "the schema documents every BC type this solver accepts".
        assert!(text.contains("oneOf"), "schema should contain oneOf: {text}");
        assert!(text.contains("\"const\""), "schema should contain const discriminators");
        assert!(text.contains("fixedValue"));
        assert!(text.contains("inletOutlet"));
        assert!(text.contains("zeroGradient"));

        // The schema is a real object, not an empty shell.
        assert!(value.get("$defs").is_some() || value.get("definitions").is_some());
    }

    #[test]
    fn schema_writes_to_docs_schema_directory() {
        let out_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../docs/schema");
        std::fs::create_dir_all(&out_dir).expect("create docs/schema");
        let out_path = out_dir.join("case-1.json");
        std::fs::write(&out_path, emit_schema()).expect("write docs/schema/case-1.json");
        assert!(out_path.exists());
    }

    #[test]
    fn f64_round_trips_exactly() {
        let case = read_case_jsonc(&example_path()).unwrap();
        let original = case.physics.fluid.nu;
        assert_eq!(original, 1.5e-5);

        let text = serde_json::to_string(&case).unwrap();
        let back: JsonCase = serde_json::from_str(&text).unwrap();
        assert_eq!(back.physics.fluid.nu, original);
        assert_eq!(back.physics.fluid.nu.to_bits(), original.to_bits());

        // The mesh bounds carry a mix of the exact-decimal and
        // not-exactly-representable kind of f64 this test exists to catch.
        assert_eq!(back.mesh.bounds.min[0], case.mesh.bounds.min[0]);
        assert_eq!(back.mesh.bounds.min[0].to_bits(), case.mesh.bounds.min[0].to_bits());
    }
}
