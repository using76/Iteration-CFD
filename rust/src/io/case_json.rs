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

use std::collections::{BTreeMap, BTreeSet};
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
use crate::energy::PrtModel;
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
    /// SPEC-LIT §18/§31: volumetric sources. Empty (the default) is every
    /// case this reader has ever built. See [`JsonSource`] for what a JSONC
    /// case can say that the OpenFOAM `constant/fvSources` route
    /// ([`crate::sources::read_sources`]) cannot express and vice versa.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub sources: Vec<JsonSource>,
}

/// One `sources[]` entry - SPEC-LIT §18's registry, reached from JSONC.
///
/// *DESIGN.* `constant/fvSources` (SPEC-LIT §18, [`crate::sources::read_sources`])
/// already has six term kinds over a box/sphere/all-cells selector; this is
/// deliberately not a second copy of that whole surface. It exists because a
/// PERIODIC case (SPEC-LIT §31.1) has no inlet to prescribe a mass flow from,
/// so the only way left to drive it is a momentum source, and JSONC had no
/// way to say one at all - not even the uniform, whole-domain case a periodic
/// channel needs. `momentumSource` is that one case, reusing
/// [`crate::sources::SourceTerm::BodyForce`] and
/// [`crate::sources::CellSelector::All`] exactly as the OpenFOAM route would
/// build them for `selection all`. A JSONC case that wants a ZONED source
/// (a heat release in a box, say) still has no JSONC way to ask for one -
/// extending this enum with `box`/`sphere` selectors is future work, not a
/// gap this closes.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "type", rename_all = "camelCase", deny_unknown_fields)]
pub enum JsonSource {
    MomentumSource {
        /// Which equation the body force acts on. A body force is a VECTOR
        /// (SPEC-LIT §18's [`crate::sources::SourceTerm::BodyForce`] doc) and
        /// only the momentum equation has a direction for it to point in, so
        /// `"U"` is the only value [`JsonCase::lower`] accepts - present
        /// anyway, spelled out, rather than assumed silently, exactly as the
        /// OpenFOAM `constant/fvSources` route requires a `field` entry.
        field: String,
        #[serde(rename = "bodyForce")]
        body_force: [f64; 3],
    },

    /// SPEC-LIT §35.1: the bulk-temperature thermostat, a volumetric
    /// proportional controller on the domain's own volume-mean `T`. Always
    /// corrects `T` and always acts over the whole mesh - see
    /// [`crate::sources::Thermostat`]'s own doc for why. `tau` omitted
    /// defaults to the domain's flow-through time
    /// ([`crate::sources::flow_through_time`]).
    ///
    /// `weighting` (SPEC-LIT §35.3) says how the controller DISTRIBUTES the
    /// total power it asks for: `"uniform"`, the default, spreads it by
    /// volume, and `"massFlux"` spreads it by the local streamwise mass flux
    /// `rho u . e_hat`, which is what the periodic-fully-developed
    /// decomposition actually calls for. `direction` gives `e_hat`; omitted,
    /// it is taken from the mesh's single cyclic pair, and a mesh with none
    /// or with several is a §13.4 error rather than a guess.
    Thermostat {
        target: f64,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        tau: Option<f64>,
        /// SPEC-LIT §35.3. Omitted is `"uniform"` - deliberately, so every
        /// measurement already recorded with the uniform form stays
        /// reproducible bit for bit (§35.3.6).
        #[serde(default, skip_serializing_if = "Option::is_none")]
        weighting: Option<String>,
        /// SPEC-LIT §35.3.5's `e_hat`. Only meaningful with
        /// `"weighting": "massFlux"`; given alongside `"uniform"` it is a
        /// §13.4 error, since uniform has no direction to use and reading it
        /// and ignoring it is exactly the silent drop §13.4 forbids.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        direction: Option<[f64; 3]>,
    },
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
    /// Per-axis grading toward the axis's own two ends, lowered onto
    /// [`crate::blockgen::GradedAxis`] exactly as `blockgen`'s own cases use
    /// it (`src/blockgen.rs`'s `case_block_spec`, e.g. the channel case's
    /// `b.y.expansion = 20.0; b.y.two_sided = true;`). Absent entirely, or an
    /// axis absent from it, keeps that axis uniform - the pre-grading
    /// behaviour this reader had before, bit for bit (see
    /// `a_case_without_grading_lowers_to_the_same_mesh_as_before`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub grading: Option<JsonGrading>,
    /// SPEC-LIT §31.1: cyclic pairs, each naming two of the six `boundaries`
    /// values. Empty (the default) is the ordinary, uncoupled mesh this
    /// reader has always built. See [`JsonCyclicPair`].
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub cyclic: Vec<JsonCyclicPair>,
}

/// One `mesh.cyclic[]` entry - SPEC-LIT §31.1. `a` and `b` are two of the six
/// `mesh.boundaries` values, and must name the two opposite slots of one
/// axis (`xmin`/`xmax`, `ymin`/`ymax` or `zmin`/`zmax`) - `build_block`
/// checks that, not this type.
///
/// `transform` takes only `"translate"` - the transform
/// [`crate::blockgen::BlockSpec::set_cyclic_axis`] knows how to build a
/// pairing from, implied by the block's own extent along that axis. A
/// rotational pair (`"rotate"`, with an axis and an angle) needs a different
/// face-matching search and a vector transform on `Sf` that nothing here
/// implements yet, so it deserialises fine (it is a *known* setting) and is
/// refused in `build_block` with a SPEC-LIT §13.4 error naming `translate` -
/// exactly like any other recognised-but-unimplemented setting, not a parse
/// failure.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct JsonCyclicPair {
    pub a: String,
    pub b: String,
    pub transform: String,
}

/// [`JsonMesh::grading`]'s three optional per-axis entries. A missing axis is
/// uniform; a present one is validated and lowered by
/// [`apply_axis_grading`].
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields, default)]
pub struct JsonGrading {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub x: Option<JsonGradingAxis>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub y: Option<JsonGradingAxis>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub z: Option<JsonGradingAxis>,
}

/// One axis's worth of [`crate::blockgen::GradedAxis`] grading:
/// `expansion` is the ratio `GradedAxis::expansion` documents (last cell over
/// first for one-sided, centre over wall for `twoSided`) and must be
/// strictly positive - `0` or negative is not a cell-size ratio for any
/// mesh, checked in [`apply_axis_grading`] rather than left for `blockgen` to
/// silently degenerate to uniform (`GradedAxis::default`'s own `!(r > 0)`
/// guard exists so a bad ratio never produces NaN coordinates, not so a bad
/// case file goes unnoticed - SPEC-LIT §13.4).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields, rename_all = "camelCase")]
pub struct JsonGradingAxis {
    pub expansion: f64,
    /// `true`: grade symmetrically toward BOTH ends of the axis (a channel's
    /// two walls); `false` (the default - most cases graded toward only one
    /// end, e.g. a pipe's single wall or a boundary layer growing off one
    /// plate, name that explicitly rather than leaving it to a default that
    /// silently picks a physical setup for them): one-sided, `hi`'s cell
    /// `expansion` times `lo`'s.
    #[serde(default)]
    pub two_sided: bool,
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
    /// SPEC-LIT S37.4: which closure supplies `Pr_t` in S26's
    /// `k_eff = k + rho cp nu_t/Pr_t` - `"constant"` (the default, and what
    /// `Prt` above means on its own) or `"KaysCrawford"` (where `Prt` above
    /// is read as `Pr_t_inf`, the free-stream asymptote). A free string
    /// rather than an enum, for the same reason `turbulence.model` is one:
    /// an unrecognised spelling has to reach [`crate::energy::PrtModel::parse`]
    /// and come back as a S13.4 error NAMING the alternatives, not as
    /// serde's "unknown variant".
    #[serde(rename = "PrtModel", default, skip_serializing_if = "Option::is_none")]
    pub prt_model: Option<String>,
    /// SPEC-LIT §38.7: which closure supplies the LAMINAR viscosity, and its
    /// coefficients. Absent means `Newtonian`, which is `nu` above and is
    /// bitwise the pre-§38 momentum equation.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rheology: Option<JsonRheology>,
    #[serde(rename = "TRef")]
    pub t_ref: f64,
}

/// `physics.fluid.rheology` - SPEC-LIT §38.
///
/// Every coefficient is in the literature's DYNAMIC units and the block
/// carries its own `rho` [kg/m³], because §5's momentum equation is
/// KINEMATIC and the conversion cannot be guessed (§38.4). `rho` is required
/// for every non-Newtonian model.
///
/// The coefficients are `Option` and the model is a free string on purpose.
/// The string has to reach [`crate::rheology::RheologyModel::parse`] and come
/// back as a §13.4 error NAMING the six spellings, not as serde's "unknown
/// variant"; and the options are what let [`JsonCase::lower`] tell "the case
/// wrote tau0" from "the case did not", so a coefficient the named model does
/// not read is REFUSED rather than dropped. That check is
/// [`crate::rheology::RheologyCoeffs::read_keys`], shared verbatim with the
/// OpenFOAM reader so there is one §13.4 contract and not two.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct JsonRheology {
    /// `Newtonian`, `powerLaw`, `CrossPowerLaw`, `BirdCarreau`,
    /// `HerschelBulkley` or `Casson`.
    pub model: String,
    /// Density [kg/m³] - required for every non-Newtonian model (§38.4).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rho: Option<f64>,
    /// Zero-shear viscosity [Pa s].
    #[serde(default, rename = "mu0", skip_serializing_if = "Option::is_none")]
    pub mu0: Option<f64>,
    /// Infinite-shear viscosity [Pa s].
    #[serde(default, rename = "muInf", skip_serializing_if = "Option::is_none")]
    pub mu_inf: Option<f64>,
    /// Casson's plastic viscosity [Pa s].
    #[serde(default, rename = "muC", skip_serializing_if = "Option::is_none")]
    pub mu_c: Option<f64>,
    /// Consistency [Pa s^n].
    #[serde(default, rename = "K", skip_serializing_if = "Option::is_none")]
    pub k: Option<f64>,
    /// Power-law index [-].
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub n: Option<f64>,
    /// Time constant [s].
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub lambda: Option<f64>,
    /// Cross's exponent, or Carreau-Yasuda's `a` [-].
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub a: Option<f64>,
    /// Yield stress [Pa].
    #[serde(default, rename = "tau0", skip_serializing_if = "Option::is_none")]
    pub tau0: Option<f64>,
    /// Papanastasiou's regularisation parameter [s] - §38.3. A NUMERICAL
    /// parameter, and printed as one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub m: Option<f64>,
    /// Clip on the apparent viscosity [Pa s].
    #[serde(default, rename = "muMin", skip_serializing_if = "Option::is_none")]
    pub mu_min: Option<f64>,
    #[serde(default, rename = "muMax", skip_serializing_if = "Option::is_none")]
    pub mu_max: Option<f64>,
    /// Floor on `gdot` before any divide or power [1/s] - §38.3.
    #[serde(default, rename = "gammaDotFloor", skip_serializing_if = "Option::is_none")]
    pub gamma_dot_floor: Option<f64>,
    /// Elementwise relaxation of the viscosity fixed point - §38.5(iv).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub relax: Option<f64>,
}

impl JsonRheology {
    /// The key/value pairs the case actually wrote, in the vocabulary
    /// [`crate::rheology::RheologyCoeffs::read_keys`] speaks.
    fn stated(&self) -> Vec<(&'static str, f64)> {
        let mut v = Vec::new();
        let mut push = |k: &'static str, o: Option<f64>| {
            if let Some(x) = o {
                v.push((k, x));
            }
        };
        push("rho", self.rho);
        push("mu0", self.mu0);
        push("muInf", self.mu_inf);
        push("muC", self.mu_c);
        push("K", self.k);
        push("n", self.n);
        push("lambda", self.lambda);
        push("a", self.a);
        push("tau0", self.tau0);
        push("m", self.m);
        push("muMin", self.mu_min);
        push("muMax", self.mu_max);
        push("gammaDotFloor", self.gamma_dot_floor);
        push("relax", self.relax);
        v
    }

    /// Resolve to [`crate::rheology::RheologyCoeffs`] under §13.4's contract.
    pub fn lower(&self) -> Result<crate::rheology::RheologyCoeffs> {
        use crate::rheology::RheologyCoeffs;

        let mut c = RheologyCoeffs {
            model: crate::rheology::RheologyModel::parse(
                "physics.fluid.rheology.model",
                &self.model,
            )?,
            ..RheologyCoeffs::default()
        };
        let stated = self.stated();

        if c.is_newtonian() {
            if let Some((k, _)) = stated.first() {
                return Err(Error::Config(format!(
                    "physics.fluid.rheology.{k} is set but the model is \
                     `{}`, so it would be read by nothing (SPEC-LIT 13.4). \
                     Name one of: {}",
                    self.model,
                    crate::rheology::RheologyModel::NAMES[1..].join(", ")
                )));
            }
            return Ok(c);
        }

        let present: Vec<String> = stated.iter().map(|(k, _)| (*k).to_string()).collect();
        c.read_keys(&present, "physics.fluid.rheology", |key| {
            stated.iter().find(|(k, _)| *k == key).map(|(_, v)| v.to_string())
        })?;
        c.validate("physics.fluid.rheology")?;
        Ok(c)
    }
}

/// `b = g·(TRef/T - 1)` (density-ratio, `SPEC-LIT` §9) is the only model this
/// solver implements - see [`crate::momentum::BuoyancyCoeffs`]. `boussinesq`
/// is accepted here, so the schema documents that OpenFOAM cases spell it
/// that way too, and [`JsonCase::lower`] rejects it under the §13.4 contract
/// rather than silently linearising a hot plume's `ΔT/T ≈ 3`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub enum BuoyancyModel {
    DensityRatio,
    Boussinesq,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct JsonPhysics {
    pub gravity: [f64; 3],
    pub fluid: JsonFluid,
    pub buoyancy: BuoyancyModel,
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
    /// SPEC-LIT §34.1: `PatchKind::Empty` - the 2-D "this axis does not
    /// exist" declaration the solver has always understood
    /// (`OFPATCH_EMPTY`), newly nameable in JSONC. A CONSTRAINT, not a
    /// boundary condition: legal only on a slot exactly one cell across
    /// (checked in [`build_block`]), and a rule of this kind may not also
    /// carry a per-field BC (checked in [`validate_constraint_rules`]) -
    /// the constraint decides every field, and `field_setup::topology_override`
    /// would silently win over a per-field BC anyway, so naming one is
    /// always a mistake worth catching rather than a redundancy worth
    /// tolerating.
    Empty,
    /// SPEC-LIT §34.1: `PatchKind::Symmetry` - the mirror-plane constraint
    /// the solver's vector-reflection branch has always understood, newly
    /// nameable in JSONC. Same CONSTRAINT rules as [`Self::Empty`], minus
    /// the single-cell check (a symmetry plane has no such restriction).
    Symmetry,
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
    /// SPEC-LIT §32.2's redesigned thermal-wall gate: a wall carrying a fixed
    /// heat FLUX `q` (W/m^2) rather than a temperature -
    /// `crate::field::BcKind::FixedFluxTemperature`. Deliberately the SAME
    /// JSON type on a `wallTreatment: standard` wall-function mesh and a
    /// `lowRe` resolved one - see that `BcKind` variant's own doc for why one
    /// condition is exact on both.
    #[serde(rename = "fixedFluxTemperature")]
    FixedFluxTemperature { q: f64 },
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
    /// The end time the case asks for. Honoured: both case readers turn it
    /// into `TurbulenceControls::n_outer_iterations` (steady) or a step count
    /// (transient). `ofgpu-lowmach` takes its run MODE from `-endTime`/`-deltaT`
    /// instead and prints which of the two is in force (SPEC-LIT S13.4.2).
    #[serde(rename = "endTime")]
    pub end_time: f64,
    /// The time step. Honoured; `-deltaT` on the command line overrides it.
    #[serde(rename = "deltaT")]
    pub delta_t: f64,
    /// `false` (the default) is honoured, because a fixed step is what every
    /// driver that reads a JSONC case does.
    ///
    /// `true` is a SPEC-LIT S13.4 ERROR naming the alternatives: no such
    /// driver adjusts its own step. `ofgpu-vof` is the one adaptive loop in
    /// this crate (`controlDict` `adjustTimeStep` + `maxCo`, or `-maxCo`) and
    /// it takes an OpenFOAM case directory. `-permissive` substitutes a fixed
    /// step of `deltaT` and says so.
    #[serde(rename = "adjustTimeStep", default)]
    pub adjust_time_step: bool,
    /// The Courant ceiling an adaptive step would hold. A SPEC-LIT S13.4
    /// ERROR whenever present, for the same reason as `adjustTimeStep`: it
    /// only means anything to a loop that adjusts its step, and no driver
    /// reading this format has one.
    #[serde(rename = "maxCo", default, skip_serializing_if = "Option::is_none")]
    pub max_co: Option<f64>,
}

// ---- output ----------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
/// The dense voxel grid a renderer reads - SPEC-LIT S44.1.
///
/// Resolved by [`crate::io::output_plan::OutputPlan::from_json`], which is
/// where every S13.4 refusal about these five entries lives; nothing here
/// validates, exactly as nothing else in this file does.
pub struct JsonVisualisation {
    /// `vdb`, `nvdb`, or a comma list of them. A format from `exact`'s
    /// column (`vtu`, `openfoam`) is an error naming `exact`; `usda` is an
    /// error naming `usdScene`.
    pub format: String,
    /// Seconds of physical time between writes. Absent (or `0`) is
    /// "the final state, once" - SPEC-LIT S44.4. Was REQUIRED before S44.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub interval: Option<f64>,
    /// Which cell fields to write, and in what order. Absent is every field
    /// the run has - SPEC-LIT S44.2.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fields: Option<Vec<String>>,
    /// `fp32` (the default) or `fp16` - SPEC-LIT S44.3. This is the ONE
    /// place in the case format where reduced precision is legitimate.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub precision: Option<String>,
    /// Also write a `.usda` scene referencing the volume files.
    #[serde(rename = "usdScene", default)]
    pub usd_scene: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
/// Interchange with the polyhedra preserved - SPEC-LIT S44.1.
pub struct JsonExact {
    /// `vtu` or `openfoam` (`foam`), or a comma list. A volume format is an
    /// error naming `visualisation`.
    pub format: String,
    /// Seconds of physical time between writes; absent is "once, at the
    /// end" - SPEC-LIT S44.4.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub interval: Option<f64>,
    /// **Always a SPEC-LIT S44.3 error.** Present as a field, rather than
    /// left to `deny_unknown_fields`, so the message can say *why* - a lossy
    /// "exact" format is a contradiction in the name - and name
    /// `visualisation.precision`, where reduced precision belongs.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub precision: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
/// This solver's own state, to resume from - SPEC-LIT S44.1/S44.5.
pub struct JsonRestart {
    /// Seconds of physical time between checkpoints; absent is "once, at
    /// the end".
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub interval: Option<f64>,
    /// How many checkpoints to retain. `0` keeps every one of them -
    /// SPEC-LIT S44.5, and the safe reading of a zero is the one that
    /// deletes nothing.
    pub keep: usize,
    /// **Always a SPEC-LIT S44.3 error**, for a sharper reason than
    /// `exact`'s: S5.1's argument for carrying `phi` in a checkpoint is that
    /// a re-derived flux is not the conservative one, and a rounded one is
    /// not either.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub precision: Option<String>,
}

/// What this run writes: three sub-blocks, three purposes - SPEC-LIT S44.
///
/// This block was a S13.4 refusal in its entirety until S44, and the reason
/// is worth keeping: three of its knobs (`visualisation.fields`,
/// `visualisation.precision`, `restart.keep`) had no implementation anywhere
/// in the crate, so honouring `format` and `interval` because they happened
/// to exist and dropping the other three in silence would have manufactured
/// S13.4.1's own defect inside its fix. S44 builds all three
/// ([`crate::io::output_plan::FieldSelection`],
/// [`crate::io::nvdb::Precision`] on both volume writers, and
/// [`crate::restart::Checkpoints`]) and then honours the block whole.
///
/// Resolution, and every S13.4 refusal, is
/// [`crate::io::output_plan::OutputPlan::from_json`]. A case that carries
/// this block must NOT also name `-output`/`-writeInterval`/`-restartWrite`
/// on the command line: those are the same settings said twice, and naming
/// both is an error rather than a silent winner (S44.6).
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
    read_case_jsonc_str(&text, &path.display().to_string())
}

/// [`read_case_jsonc`] on text already in memory - what a test that writes
/// two cases differing in one byte needs.
pub fn read_case_jsonc_str(text: &str, what: &str) -> Result<JsonCase> {
    let value = jsonc_value(text, what)?;
    refuse_held_back_blocks(&value, what)?;
    deserialize_value(value, what)
}

/// The entries this format still SPELLS and this engine does not solve, each
/// refused by name with what it selected and what is here instead - SPEC-LIT
/// §13.4.
///
/// They described a reacting, sooting or radiating medium. This engine has no
/// chemistry, no soot equation and no participating-medium radiation in it,
/// and the honest answer to a case that asks for one is not serde's `unknown
/// field "fire", expected one of ...` beside a list of its siblings. §13.4
/// asks for the name, what it needed, and what is available instead; a case
/// written for the reacting-medium solver is a real document somebody wrote,
/// and it deserves to be told what happened to it rather than diagnosed as a
/// typo.
///
/// It runs BEFORE deserialisation because `deny_unknown_fields` would
/// otherwise answer first and answer worse, and it is an [`Error::Config`]
/// rather than an `io::contract` note because `-permissive` must not be able
/// to waive it: a waived `physics.fire.combustion` is a burner case running
/// at `q''' = 0` and reporting a wall heat balance that closes because there
/// was no fire in it. That is §13.4.1's defect class on exactly the cases
/// whose whole point is the heat.
fn refuse_held_back_blocks(value: &serde_json::Value, what: &str) -> Result<()> {
    fn at<'a>(v: &'a serde_json::Value, path: &str) -> Option<&'a serde_json::Value> {
        let mut node = v;
        for seg in path.split('.') {
            node = node.get(seg)?;
        }
        Some(node)
    }

    // `(entry, what it selected, what is here instead)`.
    let sub: &[(&str, &str, &str)] = &[
        (
            "physics.fire.combustion",
            "a mixing-controlled reaction between transported species",
            "a `sources[]` heat release, or a driver's own `-heaterPower`",
        ),
        (
            "physics.fire.radiation",
            "a PARTICIPATING medium - a radiation field solved on the cells \
             out of an absorption coefficient and a spectral model",
            "`viewFactor` exchange across a TRANSPARENT medium, selected in \
             `constant/radiationProperties` (SPEC-LIT §49/§50/§51)",
        ),
        (
            "physics.fire.soot",
            "a transported soot mass fraction whose source is driven by the \
             reaction",
            "nothing - there is no soot equation here, and no reaction to \
             drive one",
        ),
    ];

    let mut named: Vec<String> = Vec::new();
    for (entry, selected, instead) in sub {
        if at(value, entry).is_some() {
            named.push(format!("{entry} selects {selected}; here instead: {instead}"));
        }
    }
    // The parent only when no child was named: reporting `physics.fire` as
    // well as `physics.fire.combustion` says the same thing twice and buries
    // the specific half.
    if named.is_empty() && at(value, "physics.fire").is_some() {
        named.push(
            "physics.fire is the reacting-medium block; here instead: the \
             low-Mach formulation of SPEC-LIT §25 and the energy equation \
             of §26, which is what is left when nothing reacts"
                .to_string(),
        );
    }
    if at(value, "initial.Y_F").is_some() {
        named.push(
            "initial.Y_F is the ambient fuel mass fraction, and it is what \
             turned the fuel/oxidiser/products species set on; here \
             instead: `crate::species` as a LIBRARY - SPEC-LIT §19's \
             general scalar transport is in this engine, and no case entry \
             selects it"
                .to_string(),
        );
    }
    // A patch rule's own `Y_F` is a condition on that same species set. Named
    // once, not once per matching rule: it is one decision, not N.
    if value
        .get("patches")
        .and_then(|v| v.as_array())
        .is_some_and(|rules| rules.iter().any(|r| r.get("Y_F").is_some()))
    {
        named.push(
            "patches[].Y_F is the fuel mass fraction at a boundary; here \
             instead: nothing - there is no species equation for it to be a \
             condition on"
                .to_string(),
        );
    }

    if named.is_empty() {
        return Ok(());
    }
    Err(Error::Config(format!(
        "{what}: this case names {} that this engine does not solve. \
         SPEC-LIT §13.4 refuses {} by name rather than reading {} and \
         dropping {}:\n  {}\nThe names are still recognised here precisely \
         so that a case written for the reacting-medium solver is told what \
         became of it.",
        if named.len() == 1 { "an entry" } else { "entries" },
        if named.len() == 1 { "it" } else { "them" },
        if named.len() == 1 { "it" } else { "them" },
        if named.len() == 1 { "it" } else { "them" },
        named.join("\n  ")
    )))
}

/// Parse any JSONC document into any `serde` type, with the same comment and
/// trailing-comma handling, the same number-literal rules and the same
/// `serde_path_to_error` diagnostics [`read_case_jsonc`] gets.
///
/// Exists so a SECOND case format (SPEC-LIT §47.4's multi-region conduction
/// case, `crate::io::case_cht`) reads exactly the way this one does rather
/// than growing a second parser with its own quirks. `deny_unknown_fields` on
/// the target type is what turns a mistyped entry into an error instead of a
/// silently ignored one, and both formats set it.
pub fn parse_jsonc_file<T: serde::de::DeserializeOwned>(path: &Path) -> Result<T> {
    let text = std::fs::read_to_string(path).path(path)?;
    parse_jsonc_str(&text, &path.display().to_string())
}

/// [`parse_jsonc_file`] on text already in memory - what a test that writes
/// two cases differing in one byte needs.
pub fn parse_jsonc_str<T: serde::de::DeserializeOwned>(text: &str, what: &str) -> Result<T> {
    deserialize_value(jsonc_value(text, what)?, what)
}

/// The JSONC document as a [`serde_json::Value`], before any type has had an
/// opinion about it. [`refuse_held_back_blocks`] reads this, because
/// `deny_unknown_fields` would otherwise answer first and answer worse.
fn jsonc_value(text: &str, what: &str) -> Result<serde_json::Value> {
    let parsed = jsonc_parser::parse_to_value(text, &parse_options())
        .map_err(|e| Error::Parse { path: what.to_string(), msg: e.to_string() })?;
    let Some(value) = parsed else {
        return Err(Error::Parse {
            path: what.to_string(),
            msg: "empty JSONC document".to_string(),
        });
    };
    jsonc_to_serde(value)
}

/// [`jsonc_value`]'s output, deserialised, with `serde_path_to_error`'s JSON
/// path in the message - WHERE in the file, and not just what.
fn deserialize_value<T: serde::de::DeserializeOwned>(
    value: serde_json::Value,
    what: &str,
) -> Result<T> {
    serde_path_to_error::deserialize(value).map_err(|e| {
        let json_path = e.path().to_string();
        Error::Parse {
            path: what.to_string(),
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
    /// Only meaningful when [`Self::adjust_time_step`] is set, and `None`
    /// where the case named none.
    ///
    /// An `Option` rather than a `Scalar` defaulted to zero, because "the
    /// case did not say" and "the case said zero" are different states and
    /// the refusal in `common::refuse_unimplemented_blocks` has to tell them
    /// apart - SPEC-LIT §13.4. Collapsing them with `unwrap_or(0.0)` is what
    /// this field used to do, and it was read by nobody at all.
    pub max_co: Option<Scalar>,
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
    /// `physics.fluid.PrtModel`, resolved - SPEC-LIT S37.4. Kept beside
    /// [`Self::fluid`] rather than inside it because
    /// [`ScalarTransportCoeffs`] is also what `ofgpu-buoyant`/`ofgpu-plume`
    /// carry, and neither implements S37: a field there would be a setting
    /// those two accepted and ignored, which is the S13.4 defect this
    /// project keeps finding.
    pub prt_model: PrtModel,
    /// `physics.fluid.rheology`, resolved - SPEC-LIT §38. Newtonian unless
    /// the case named a closure.
    pub rheology: crate::rheology::RheologyCoeffs,
    pub buoyancy: BuoyancyCoeffs,
    pub wall: WallFunctionCoeffs,
    pub turbulence_model: Option<String>,
    /// `turbulence.kind`, kept so [`Self::to_case_controls`] can route LES:
    /// the selector reads `simulationType`/`LES/model` from the
    /// `momentum_transport` dictionary, which a JSONC case does not have -
    /// dropping the kind here was how `"kind": "LES"` got accepted by the
    /// schema and then answered with "needs simulationType LES", a setting
    /// the JSONC format does not spell.
    pub turbulence_kind: Option<TurbulenceKind>,

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
    /// `numerics.solvers`, verbatim and ORDERED - the rules themselves, not
    /// just the four equations this layer happened to resolve them for.
    ///
    /// [`Self::p_solver`], [`Self::u_solver`] and `turb.k_solver`/
    /// `turb.epsilon_solver` are the four `lower` resolves eagerly, because
    /// those are the four `CaseControls` has slots for. A driver solving an
    /// equation with no slot - `T` in `ofgpu-lowmach`, anything a later
    /// model adds - had NO way to reach its rule at all, so a case writing
    /// `{ "match": "T", ... }` got `SolverControls::default()` and no
    /// diagnostic. [`Self::solver_for`] is that way.
    pub solvers: Vec<JsonSolverRule>,
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

    pub output: Option<JsonOutput>,

    /// `sources[]`, resolved to the same [`crate::sources::SourceSpec`] the
    /// OpenFOAM `constant/fvSources` route ([`crate::sources::read_sources`])
    /// produces - one registry, two ways to name an entry in it. Empty for
    /// every case that names no `sources` block.
    pub sources: Vec<crate::sources::SourceSpec>,
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

    /// The `numerics.div` entry for `key` **only if the case named it**, with
    /// no fall back to `"default"`.
    ///
    /// [`Self::div_for`] is right for an equation the case is expected to
    /// have an opinion about; this is for one where the DRIVER's own
    /// documented default is better than the case's catch-all `default`
    /// entry - a species convection, whose SPEC-LIT S19 default is bounded
    /// upwind while a case's `"default": "Gauss upwind"` is not bounded.
    /// Falling back there would change the physics of every case that never
    /// mentioned species at all.
    pub fn div_named(&self, key: &str) -> Option<DivEntry> {
        self.div.get(key).copied()
    }

    /// The `numerics.solvers` rule that governs `var`, resolved with the
    /// SAME first-match-wins order [`resolve_patch_rule`] uses - the JSONC
    /// twin of `crate::io::case::read_solver_controls`.
    ///
    /// A `var` no rule matches keeps [`SolverControls::default`], exactly as
    /// an OpenFOAM case with no `solvers/<var>` entry does. Use it for every
    /// equation a driver actually assembles, by that equation's OWN name:
    /// handing the energy equation `p_solver` because `p_solver` was the one
    /// already resolved is the substitution SPEC-LIT §13.4 forbids.
    pub fn solver_for(&self, var: &str) -> Result<SolverControls> {
        solver_for(&self.solvers, var)
    }

    /// `numerics.relaxation`'s entry for `var`, or `fallback`.
    ///
    /// By the field's OWN name, for the same reason [`Self::solver_for`] is:
    /// `numerics.relaxation` is a map keyed per equation, and a driver
    /// reading `U`'s factor for `T` would be reading one equation's setting
    /// for another.
    pub fn relax_for(&self, var: &str, fallback: Scalar) -> Scalar {
        self.relaxation.get(var).copied().unwrap_or(fallback)
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
            rheology: self.rheology,
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
        // `"kind": "LES"` routes through the SAME selector the OpenFOAM
        // format uses, by synthesising the two dictionary entries it reads.
        // The RAS branch needs nothing: the selector falls back to
        // `model_name` there, which is already set above.
        if self.turbulence_kind == Some(TurbulenceKind::Les) {
            let model = self.turbulence_model.as_deref().unwrap_or("");
            let src = format!(
                "simulationType LES;
LES {{ model {model}; }}
"
            );
            match crate::io::dict::FoamDict::parse(&src, "<jsonc turbulence>") {
                Ok(d) => c.momentum_transport = d,
                // Unreachable for the strings this format can produce; a
                // parse failure here would be a bug in the synthesis, not in
                // the case, so surface it as the model name and let the
                // selector produce its own section-13.4 error downstream.
                Err(_) => {}
            }
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
        // SPEC-LIT §34.1: these two are recognised by `PatchKind::from_type`
        // (`src/mesh.rs`) directly, so the JSON kind name and the OpenFOAM
        // type string this reader writes into `BlockSpec.patch_type` are the
        // same word - unlike wall/inlet/open, which fold onto OpenFOAM's own
        // "wall"/"patch" class.
        PatchPresetKind::Empty => "empty",
        PatchPresetKind::Symmetry => "symmetry",
    }
}

/// `mesh.grading.{x,y,z}`, applied to one already-built (uniform) axis.
///
/// `grading` absent for this axis leaves it exactly as `build_block`'s own
/// `axis` closure made it - the bit-identity a case without grading owes
/// (`a_case_without_grading_lowers_to_the_same_mesh_as_before`). A present
/// entry with `expansion <= 0` is SPEC-LIT §13.4: not a menu of alternatives
/// (any positive ratio is physical), so [`unsupported`]'s `available` list is
/// empty and the named fallback is the uniform axis this reader would have
/// built without the `grading` block at all.
fn apply_axis_grading(name: &str, axis: &mut GradedAxis, grading: Option<&JsonGradingAxis>) -> Result<()> {
    let Some(g) = grading else { return Ok(()) };

    if g.expansion > 0.0 {
        axis.expansion = g.expansion as Scalar;
        axis.two_sided = g.two_sided;
        return Ok(());
    }

    let uniform = unsupported(
        &format!("mesh.grading.{name}.expansion"),
        &g.expansion.to_string(),
        &[],
        "uniform grading (expansion = 1, the axis's un-graded default)",
        (1.0 as Scalar, false),
    )?;
    axis.expansion = uniform.0;
    axis.two_sided = uniform.1;
    Ok(())
}

/// SPEC-LIT §34.1: a per-field BC on an `empty`/`symmetry` rule is a §13.4
/// error naming the field - checked statically over every `patches[]` rule,
/// independent of whether this particular case even solves that field. The
/// constraint decides every field for the patch it matches
/// (`field_setup::topology_override` overrides whatever a field spec says
/// there regardless), so a rule that ALSO names one has misunderstood
/// something the reader can catch rather than something worth silently
/// discarding.
fn validate_constraint_rules(patches: &[JsonPatchRule]) -> Result<()> {
    for rule in patches {
        if !matches!(rule.kind, PatchPresetKind::Empty | PatchPresetKind::Symmetry) {
            continue;
        }
        let kind_name = patch_class(rule.kind);
        let reject = |field: &str| -> Error {
            Error::Config(format!(
                "patches: rule \"{}\" has kind \"{kind_name}\" - a CONSTRAINT, not a \
                 boundary condition - but also sets \"{field}\"; the constraint decides \
                 every field on this patch, so remove the \"{field}\" entry",
                rule.pattern
            ))
        };
        if rule.u.is_some() {
            return Err(reject("U"));
        }
        if rule.p.is_some() {
            return Err(reject("p"));
        }
        if rule.t.is_some() {
            return Err(reject("T"));
        }
        if rule.turbulence.is_some() {
            return Err(reject("turbulence"));
        }
        if rule.k.is_some() {
            return Err(reject("k"));
        }
        if rule.epsilon.is_some() {
            return Err(reject("epsilon"));
        }
        if rule.omega.is_some() {
            return Err(reject("omega"));
        }
        if rule.nut.is_some() {
            return Err(reject("nut"));
        }
        if rule.wall_treatment.is_some() {
            return Err(reject("treatment"));
        }
        if rule.ks.is_some() {
            return Err(reject("Ks"));
        }
        if rule.cs.is_some() {
            return Err(reject("Cs"));
        }
    }
    Ok(())
}

fn build_block(mesh: &JsonMesh, patches: &[JsonPatchRule]) -> Result<(BlockSpec, Vec<WindowRegionSpec>)> {
    let MeshKind::Cartesian = mesh.kind;

    validate_constraint_rules(patches)?;

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

    if let Some(g) = &mesh.grading {
        apply_axis_grading("x", &mut block.x, g.x.as_ref())?;
        apply_axis_grading("y", &mut block.y, g.y.as_ref())?;
        apply_axis_grading("z", &mut block.z, g.z.as_ref())?;
    }

    let names = [
        mesh.boundaries.xmin.clone(),
        mesh.boundaries.xmax.clone(),
        mesh.boundaries.ymin.clone(),
        mesh.boundaries.ymax.clone(),
        mesh.boundaries.zmin.clone(),
        mesh.boundaries.zmax.clone(),
    ];

    // SPEC-LIT §31.1: a cyclic slot gets `cyclic` from `set_cyclic_axis`
    // below, not from a `patches[]` rule - `PatchPresetKind` has no `cyclic`
    // variant to resolve to, and a cyclic patch does not need one written.
    let cyclic_slots = cyclic_slot_set(mesh, &names)?;

    for (slot, name) in names.iter().enumerate() {
        block.patch_name[slot] = name.clone();
        if cyclic_slots.contains(&slot) {
            continue;
        }
        let rule = resolve_patch_rule(patches, name)?;
        block.patch_type[slot] = patch_class(rule.kind).to_string();

        // SPEC-LIT §34.1: `empty` is legal only across a single cell - this
        // is already `blockgen`'s own rule for its canned cases (see the
        // "single cell in that direction" check in `case_block_spec`); the
        // JSONC reader must not let a case say `empty` on a slot the mesh
        // builder would then treat as a real, multi-cell boundary.
        if rule.kind == PatchPresetKind::Empty {
            let axis = slot / 2;
            let n = mesh.cells[axis];
            if n != 1 {
                return Err(Error::Config(format!(
                    "mesh.boundaries.{}: patch \"{name}\" is \"empty\" but its axis \
                     ({}) has {n} cell(s), not 1 - an empty patch is only legal across \
                     a single cell",
                    SLOT_NAMES[slot],
                    ["x", "y", "z"][axis],
                )));
            }
        }
    }

    // SPEC-LIT §34.2: every pair is wired up now - `BlockSpec.cyclic` is a
    // list, and `set_cyclic_axis` itself refuses a repeated axis ("an axis
    // may appear in at most one pair"). The §31.1 "exactly one cyclic pair"
    // refusal that used to live here is gone: it was `BlockSpec`'s own
    // limitation, not a policy this reader should still enforce.
    for pair in &mesh.cyclic {
        if pair.transform != "translate" {
            unsupported::<()>(
                "mesh.cyclic[].transform",
                &pair.transform,
                &["translate"],
                "translate",
                (),
            )?;
        }

        let axis = cyclic_axis_of(&names, pair)?;
        block.set_cyclic_axis(axis).map_err(|e| Error::Config(e.to_string()))?;

        // Point 2: a cyclic patch may not also be named by a `patches[]`
        // rule - including an `empty`/`symmetry` constraint rule (§34.2:
        // "a pair and a constraint patch on the same slot is a §13.4 error
        // naming both, because `empty` and `cyclic` are contradictory
        // statements about the same faces"). This is really "no rule may
        // match this name", with one exception: the mandatory catch-all
        // (`resolve_patch_rule`'s own error names the canonical spelling,
        // `".*"`) every OTHER case is expected to end with, which says
        // nothing about THIS patch in particular and is not a contradiction
        // of the pairing.
        for name in [&pair.a, &pair.b] {
            if let Some(rule) = patches.iter().find(|r| {
                r.pattern != ".*"
                    && Regex::new(&r.pattern).map(|re| re.is_match(name)).unwrap_or(false)
            }) {
                return Err(Error::Config(format!(
                    "mesh.cyclic: patch '{name}' is paired with '{}' but is ALSO named by \
                     a patches[] rule (\"match\": \"{}\", \"kind\": \"{:?}\") - a cyclic \
                     patch gets `cyclic` on every field automatically and cannot also \
                     carry a wall/inlet/open/empty/symmetry rule",
                    if name == &pair.a { &pair.b } else { &pair.a },
                    rule.pattern,
                    rule.kind,
                )));
            }
        }
    }
    // NOTE: `mesh_patch_names` (below) still hands a cyclic patch's name to
    // `resolve_patch_rule` for every FIELD's own boundary condition (`U`,
    // `p`, `T`, the turbulence closure, ...) - it will match the mandatory
    // catch-all, since the check just above refuses anything more specific,
    // and so, e.g., `U`'s lowered spec for a cyclic patch reads whatever the
    // catch-all's preset says (`noSlip`, `zeroGradient`, ...), not `cyclic`.
    // That is deliberate rather than an oversight: `field_setup.rs`'s
    // `topology_override` (its own comment: "what the mesh insists on,
    // regardless of what the field file says") forces `BcKind::Cyclic` on
    // every face of a `PatchKind::Cyclic` patch at the point a field is
    // actually built, whatever a lowered `PatchFieldSpec` or an OpenFOAM
    // field file happened to say - the exact mechanism `blockgen`'s own
    // `field.rs::kinds_from_patches` doc comment names ("the mesh has the
    // last word on empty and cyclic"). Special-casing all seven per-field
    // loops below to write `cyclic` explicitly would only change what a
    // round-tripped field FILE looks like, never what the solver does with
    // it.

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

    // One region per SLOT, generalised from the single region this shipped
    // with: a compartment case needs a supply window in the floor and a
    // doorway window in a wall. Two regions on the SAME slot are refused, by
    // `blockgen::boundary_patches` and by name, because a slot split three
    // ways cannot be written as contiguous startFace/nFaces runs.
    block.windows = windows.iter().map(|w| w.window.clone()).collect();

    Ok((block, windows))
}

// ------------------------------------------------------------ cyclic pairs

/// Which axis (0=x, 1=y, 2=z) `pair` names, given the six `mesh.boundaries`
/// values in `-x +x -y +y -z +z` order: `a` and `b` must be two of those six
/// names, and must be the two OPPOSITE slots of one axis, in either order.
fn cyclic_axis_of(names: &[String; 6], pair: &JsonCyclicPair) -> Result<usize> {
    let find = |name: &str| -> Result<usize> {
        names.iter().position(|n| n == name).ok_or_else(|| {
            Error::Config(format!(
                "mesh.cyclic: '{name}' is not one of the six mesh.boundaries values ({})",
                names.join(", ")
            ))
        })
    };
    let (sa, sb) = (find(&pair.a)?, find(&pair.b)?);
    let axis = sa / 2;
    if sa == sb || sb / 2 != axis {
        return Err(Error::Config(format!(
            "mesh.cyclic: '{}' and '{}' are not the two opposite faces of one axis - \
             a cyclic pair has to be xmin/xmax, ymin/ymax or zmin/zmax",
            pair.a, pair.b
        )));
    }
    Ok(axis)
}

/// The boundary slots `mesh.cyclic` claims, as a set - empty when there are
/// no cyclic pairs (SPEC-LIT §34.2 generalises this from at most one pair to
/// any number). "An axis may appear in at most one pair" is enforced here,
/// naming the axis, before any pair reaches [`BlockSpec::set_cyclic_axis`]
/// (which would also catch it, but with a `BlockSpec`-flavoured message
/// rather than one that names the JSON setting). "A patch may appear in at
/// most one pair" is the SAME constraint once pairing is axis-based: each
/// axis's two slots belong to no other axis, so two pairs can only ever
/// collide by naming the same axis.
fn cyclic_slot_set(mesh: &JsonMesh, names: &[String; 6]) -> Result<BTreeSet<usize>> {
    const AXIS_NAMES: [&str; 3] = ["x", "y", "z"];
    let mut slots = BTreeSet::new();
    let mut axis_owner: BTreeMap<usize, &JsonCyclicPair> = BTreeMap::new();

    for pair in &mesh.cyclic {
        let axis = cyclic_axis_of(names, pair)?;
        if let Some(prev) = axis_owner.insert(axis, pair) {
            return Err(Error::Config(format!(
                "mesh.cyclic: axis {} is claimed by two pairs - '{}'/'{}' and '{}'/'{}' \
                 - an axis may appear in at most one cyclic pair",
                AXIS_NAMES[axis], prev.a, prev.b, pair.a, pair.b,
            )));
        }
        slots.insert(2 * axis);
        slots.insert(2 * axis + 1);
    }
    Ok(slots)
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
        ScalarBc::FixedFluxTemperature { q } => {
            let mut s = spec("fixedFluxTemperature");
            s.extra.insert("q".to_string(), q.to_string());
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
        // SPEC-LIT §34.1: `validate_constraint_rules` has already refused a
        // `U` entry on a constraint rule, so this arm is only ever reached
        // with no override - the spec written here is a placeholder in the
        // exact sense `mesh_patch_names`' cyclic-patch comment describes:
        // `field_setup::topology_override` forces `BcKind::Empty`/`Symmetry`
        // on every face of the matching `PatchKind` regardless of what this
        // spec says, so its `type_name` exists for readability, not effect.
        PatchPresetKind::Empty | PatchPresetKind::Symmetry => Ok(spec(patch_class(rule.kind))),
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
        // See `u_spec_for`'s matching arm: a placeholder, overridden by the
        // mesh's own topology at field-build time regardless.
        PatchPresetKind::Empty | PatchPresetKind::Symmetry => Ok(spec(patch_class(rule.kind))),
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
///
/// `turbulence_kind`/`model_name` are only for SPEC-LIT §32's second
/// finding, via [`crate::io::case::validate_low_re_wall_treatment`]: `None`
/// (an LES case - its own `lowRe` row is always valid, checked separately by
/// `les_nut_type`) skips that check entirely.
fn turb_field_spec(
    case_default: WallTreatmentKind,
    rule: &JsonPatchRule,
    field_name: &str,
    turbulence_kind: Option<TurbulenceKind>,
    model_name: Option<&str>,
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
    // SPEC-LIT §34.1: an `empty`/`symmetry` constraint patch needs no
    // per-field condition and no wall-treatment expansion - it is not a
    // wall, and `validate_constraint_rules` has already refused an explicit
    // one here, so this is a placeholder overridden by the mesh's own
    // topology at field-build time, same as `u_spec_for`'s matching arm.
    if matches!(rule.kind, PatchPresetKind::Empty | PatchPresetKind::Symmetry) {
        return Ok(spec(patch_class(rule.kind)));
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
    let treatment = if turbulence_kind == Some(TurbulenceKind::Ras) {
        crate::io::case::validate_low_re_wall_treatment(
            &format!("{setting} wallTreatment (\"lowRe\" together with turbulence.model)"),
            model_name.unwrap_or(""),
            treatment,
        )?
    } else {
        treatment
    };
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
    for w in &block.windows {
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
        // SPEC-LIT S37.4. Absent is `constant`, which is what every case
        // written before S37 existed means; an unrecognised spelling is a
        // S13.4 error naming both, resolved HERE rather than at the point of
        // use so a case that names it is refused before any GPU work starts.
        let prt_model = match &self.physics.fluid.prt_model {
            Some(w) => PrtModel::parse("physics.fluid.PrtModel", w)?,
            None => PrtModel::Constant,
        };
        // SPEC-LIT S38.7. Absent is `Newtonian`, which is what every case
        // written before S38 existed means, and resolving it HERE means an
        // unrecognised model, or a coefficient no model reads, is refused
        // before any GPU work starts.
        let rheology = match &self.physics.fluid.rheology {
            Some(r) => r.lower()?,
            None => crate::rheology::RheologyCoeffs::default(),
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

        // ---- turbulence -------------------------------------------------
        let mut wall = WallFunctionCoeffs::default();
        let (turbulence_model, turbulence_kind) = if let Some(t) = &self.turbulence {
            wall.kappa = t.wall_functions.kappa as Scalar;
            wall.e = t.wall_functions.e as Scalar;
            (Some(t.model.clone()), Some(t.kind))
        } else {
            (None, None)
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

        // Which dissipation variable this case's MODEL transports fills the
        // one dissipation slot of `TurbulenceControls` - the same rule, and
        // the same reason, as `crate::io::case::dissipation_key`. This used
        // to be `"epsilon"` unconditionally for the solver and
        // epsilon-then-omega for the relaxation, so a `kOmega` JSONC case
        // took `solvers[match=epsilon]` for its omega equation.
        let model_name = self
            .turbulence
            .as_ref()
            .map(|t| t.model.as_str())
            .unwrap_or("");
        let diss = crate::io::case::dissipation_from_model(model_name).unwrap_or_else(|| {
            if self.numerics.div.contains_key("div(phi,omega)")
                && !self.numerics.div.contains_key("div(phi,epsilon)")
            {
                "omega"
            } else {
                "epsilon"
            }
        });

        let mut turb = TurbulenceControls::default();
        turb.k_solver = solver_for(&self.numerics.solvers, "k")?;
        turb.epsilon_solver = solver_for(&self.numerics.solvers, diss)?;
        if let Some(v) = self.numerics.relaxation.get("k") {
            turb.k_relax = *v as Scalar;
        }
        if let Some(v) = self.numerics.relaxation.get(diss) {
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

        // The MODEL decides which entry fills the one dissipation slot -
        // `crate::io::case::dissipation_key` records what asking the
        // dictionary instead used to cost. `cases/plume.jsonc` names BOTH
        // `div(phi,epsilon)` and `div(phi,omega)`, exactly as the OpenFOAM
        // case it mirrors does.
        let eps_key = format!("div(phi,{diss})");
        let ee = div.get(&eps_key).copied().unwrap_or(default_div);
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

        let mut algorithm = build_algorithm(&self.numerics.algorithm, turb.n_non_orth_correctors);

        // SPEC-LIT §31.3: `run.endTime`, `numerics.ddt` and `numerics.algorithm`
        // are three settings a case can get individually right and jointly
        // nonsensical - a shipped transient case named `SIMPLE` while being
        // run transiently, and diverged to Inf around step 20.
        crate::io::case::check_transient_algorithm_contract(
            self.run.end_time as Scalar,
            ddt,
            &mut algorithm,
        )?;

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
            max_co: self.run.max_co.map(|v| v as Scalar),
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
                // SPEC-LIT §32's second finding, same check as
                // `turb_field_spec`'s own: `T`'s own `lowRe` row (§29.3,
                // "pins the molecular resistance") is only meaningful when
                // `lowRe` itself is - an LES case (`turbulence_kind ==
                // Les`) skips it, same reasoning as there.
                let treatment = if turbulence_kind == Some(TurbulenceKind::Ras) {
                    crate::io::case::validate_low_re_wall_treatment(
                        &format!(
                            "patches: rule \"{}\" T wallTreatment (\"lowRe\" together with \
                             turbulence.model)",
                            rule.pattern
                        ),
                        turbulence_model.as_deref().unwrap_or(""),
                        treatment,
                    )?
                } else {
                    treatment
                };
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

        // ---- turbulence closure fields ---------------------------------
        // Dimensions match what `crate::field_setup`'s own drivers write
        // back out - see e.g. `src/bin/k_epsilon.rs`'s `out_k.dimensions`.
        let turb_field = |value: Option<f64>, name: &str, dims: &str| -> Result<Option<LoweredScalarField>> {
            let Some(v0) = value else { return Ok(None) };
            let mut boundary = BTreeMap::new();
            for pname in &names {
                let rule = resolve_patch_rule(&self.patches, pname)?;
                boundary.insert(
                    pname.clone(),
                    turb_field_spec(
                        wall_treatment_default,
                        rule,
                        name,
                        turbulence_kind,
                        turbulence_model.as_deref(),
                    )?,
                );
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

        // ---- SPEC-LIT §18/§31: sources[] -----------------------------------
        let mut sources: Vec<crate::sources::SourceSpec> = Vec::new();
        for (i, src) in self.sources.iter().enumerate() {
            match src {
                JsonSource::MomentumSource { field, body_force } => {
                    if field != "U" {
                        return Err(Error::Config(format!(
                            "sources[{i}] (momentumSource): field \"{field}\" - a body \
                             force is a vector and only the momentum equation (\"U\") \
                             has a direction for it to point in (SPEC-LIT §18)"
                        )));
                    }
                    let b = to_vec3(*body_force);
                    if !(b.x.is_finite() && b.y.is_finite() && b.z.is_finite()) {
                        return Err(Error::Config(format!(
                            "sources[{i}] (momentumSource): bodyForce ({} {} {}) is not \
                             finite",
                            b.x, b.y, b.z
                        )));
                    }
                    sources.push(crate::sources::SourceSpec {
                        name: format!("momentumSource{i}"),
                        field: field.clone(),
                        selector: crate::sources::CellSelector::All,
                        term: Some(crate::sources::SourceTerm::BodyForce(b)),
                        heat_release: None,
                    });
                }
                JsonSource::Thermostat {
                    target,
                    tau,
                    weighting,
                    direction,
                } => {
                    if !target.is_finite() {
                        return Err(Error::Config(format!(
                            "sources[{i}] (thermostat): target {target} is not \
                             a finite temperature"
                        )));
                    }
                    if let Some(t) = tau {
                        if !(*t > 0.0) || !t.is_finite() {
                            return Err(Error::Config(format!(
                                "sources[{i}] (thermostat): tau {t} is not a \
                                 usable relaxation time"
                            )));
                        }
                    }
                    // SPEC-LIT §35.3.7. Absent is `uniform`, the default
                    // §35.3.6 keeps deliberately; any other spelling is a
                    // §13.4 error naming the two that exist.
                    let spelled = weighting.as_deref().unwrap_or("uniform");
                    let weighting = match crate::sources::ThermostatWeighting::parse(spelled) {
                        Some(w) => w,
                        None => unsupported(
                            &format!("sources[{i}].weighting"),
                            spelled,
                            &["uniform", "massFlux"],
                            "uniform, SPEC-LIT §35.1's volume-weighted form",
                            crate::sources::ThermostatWeighting::Uniform,
                        )?,
                    };
                    sources.push(crate::sources::SourceSpec {
                        name: format!("thermostat{i}"),
                        field: "T".to_string(),
                        selector: crate::sources::CellSelector::All,
                        term: Some(crate::sources::SourceTerm::Thermostat {
                            target: *target as Scalar,
                            tau: tau.map(|t| t as Scalar),
                            weighting,
                            direction: direction.map(to_vec3),
                        }),
                        heat_release: None,
                    });
                }
            }
        }

        Ok(LoweredCase {
            name: self.name.clone(),
            block,
            windows,
            nu,
            fluid,
            prt_model,
            rheology,
            buoyancy,
            wall,
            turbulence_model,
            turbulence_kind,
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
            solvers: self.numerics.solvers.clone(),
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
            output: self.output.clone(),
            sources,
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
        assert!(!lowered.block.windows.is_empty(), "the burner window should be carved");
        let w = lowered.block.windows.first().unwrap();
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
    //  Task G.1: mesh.grading -> GradedAxis
    // ------------------------------------------------------------------

    /// A minimal, parseable case whose mesh block callers mutate before
    /// lowering - `docs/case-example.json` and `cases/plume.jsonc` both carry
    /// far more machinery (sources, windows, a full patch table) than a grading
    /// test needs to exercise `build_block`.
    fn minimal_case_with_mesh(mesh_extra: &str) -> JsonCase {
        let text = format!(
            r#"{{
                "name": "gradingTest",
                "mesh": {{
                    "kind": "cartesian",
                    "bounds": {{ "min": [0,0,0], "max": [1,1,1] }},
                    "cells": [4, 6, 8],
                    "boundaries": {{
                        "xmin": "xa", "xmax": "xb", "ymin": "ya",
                        "ymax": "yb", "zmin": "za", "zmax": "zb"
                    }}{mesh_extra}
                }},
                "physics": {{
                    "gravity": [0,0,0],
                    "fluid": {{ "nu": 1e-5, "Pr": 0.71, "Prt": 0.85, "TRef": 293.15 }},
                    "buoyancy": "densityRatio"
                }},
                "patches": [ {{ "match": ".*", "kind": "wall" }} ],
                "initial": {{ "U": [0,0,0], "p": 0.0 }},
                "numerics": {{
                    "algorithm": {{ "kind": "SIMPLE" }},
                    "ddt": "steadyState",
                    "div": {{ "default": "Gauss upwind" }},
                    "grad": "Gauss linear",
                    "laplacian": {{ "snGrad": "corrected", "nonOrthogonalCorrectors": 0 }},
                    "solvers": []
                }},
                "run": {{ "endTime": 1.0, "deltaT": 1.0 }}
            }}"#
        );
        // A per-call counter, not just the process id: cargo's test runner
        // is multi-threaded, and every test in this module calls this helper
        // - a shared file name would let two threads race on the same path.
        static COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let n = COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!(
            "case_json_grading_test_{}_{n}.jsonc",
            std::process::id()
        ));
        std::fs::write(&path, text).unwrap();
        let case = read_case_jsonc(&path).expect("minimal grading test case should parse");
        let _ = std::fs::remove_file(&path);
        case
    }

    /// The bit-identity requirement Task G.1 names explicitly: a case with NO
    /// `grading` key lowers to the exact same `HostMesh` as before this
    /// feature existed - hashed here (via `PartialEq` on the whole array, the
    /// strongest available check) rather than merely "n matches".
    #[test]
    fn a_case_without_grading_lowers_to_the_same_mesh_as_before() {
        let with_no_grading_key = minimal_case_with_mesh("");
        let with_explicit_uniform_defaults = minimal_case_with_mesh(
            r#", "grading": {} "#,
        );

        let a = with_no_grading_key.lower().expect("lower");
        let b = with_explicit_uniform_defaults.lower().expect("lower");
        assert_eq!(a.block, b.block, "an absent grading block and an empty one must lower identically");

        // Every axis is exactly `GradedAxis::default`'s shape but for lo/hi/n.
        for axis in [&a.block.x, &a.block.y, &a.block.z] {
            assert_eq!(axis.expansion, 1.0);
            assert!(!axis.two_sided);
        }

        let mesh_a = crate::blockgen::build_mesh(&a.block).expect("build_mesh");
        let mesh_b = crate::blockgen::build_mesh(&b.block).expect("build_mesh");
        assert_eq!(mesh_a.c, mesh_b.c, "cell centres must be bit-identical");
        assert_eq!(mesh_a.v, mesh_b.v, "cell volumes must be bit-identical");

        // Hash the node coordinates (what `graded_nodes` actually computes)
        // against a hand-built uniform axis, so this is checked against the
        // pre-grading behaviour itself, not just internal self-consistency.
        use std::hash::{Hash, Hasher};
        fn hash_mesh(m: &crate::mesh::HostMesh) -> u64 {
            let mut h = std::collections::hash_map::DefaultHasher::new();
            for x in &m.c {
                x.x.to_bits().hash(&mut h);
                x.y.to_bits().hash(&mut h);
                x.z.to_bits().hash(&mut h);
            }
            for v in &m.v {
                v.to_bits().hash(&mut h);
            }
            h.finish()
        }
        let uniform_block = crate::blockgen::BlockSpec {
            x: GradedAxis { lo: 0.0, hi: 1.0, n: 4, ..GradedAxis::default() },
            y: GradedAxis { lo: 0.0, hi: 1.0, n: 6, ..GradedAxis::default() },
            z: GradedAxis { lo: 0.0, hi: 1.0, n: 8, ..GradedAxis::default() },
            ..a.block.clone()
        };
        let mesh_ref = crate::blockgen::build_mesh(&uniform_block).expect("build_mesh");
        assert_eq!(hash_mesh(&mesh_a), hash_mesh(&mesh_ref), "grading-free lowering must match a hand-built uniform mesh");
    }

    // ---- SPEC-LIT §31.1: mesh.cyclic ---------------------------------------

    /// [`minimal_case_with_mesh`], but with the `patches` array named too -
    /// the cyclic/patches-conflict tests need a `patches[]` entry that names
    /// a cyclic patch specifically, which the hardcoded `".*"` catch-all in
    /// `minimal_case_with_mesh` cannot express.
    fn case_with_mesh_and_patches(mesh_extra: &str, patches_json: &str) -> Result<JsonCase> {
        let text = format!(
            r#"{{
                "name": "cyclicTest",
                "mesh": {{
                    "kind": "cartesian",
                    "bounds": {{ "min": [0,0,0], "max": [1,1,1] }},
                    "cells": [4, 6, 8],
                    "boundaries": {{
                        "xmin": "xa", "xmax": "xb", "ymin": "ya",
                        "ymax": "yb", "zmin": "za", "zmax": "zb"
                    }}{mesh_extra}
                }},
                "physics": {{
                    "gravity": [0,0,0],
                    "fluid": {{ "nu": 1e-5, "Pr": 0.71, "Prt": 0.85, "TRef": 293.15 }},
                    "buoyancy": "densityRatio"
                }},
                "patches": {patches_json},
                "initial": {{ "U": [0,0,0], "p": 0.0 }},
                "numerics": {{
                    "algorithm": {{ "kind": "SIMPLE" }},
                    "ddt": "steadyState",
                    "div": {{ "default": "Gauss upwind" }},
                    "grad": "Gauss linear",
                    "laplacian": {{ "snGrad": "corrected", "nonOrthogonalCorrectors": 0 }},
                    "solvers": []
                }},
                "run": {{ "endTime": 1.0, "deltaT": 1.0 }}
            }}"#
        );
        static COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let n = COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!(
            "case_json_cyclic_test_{}_{n}.jsonc",
            std::process::id()
        ));
        std::fs::write(&path, text).unwrap();
        let case = read_case_jsonc(&path);
        let _ = std::fs::remove_file(&path);
        case
    }

    const WALL_CATCH_ALL: &str = r#"[ { "match": ".*", "kind": "wall" } ]"#;

    #[test]
    fn a_cyclic_pair_lowers_to_a_paired_block_spec() {
        let case = case_with_mesh_and_patches(
            r#", "cyclic": [ { "a": "xa", "b": "xb", "transform": "translate" } ]"#,
            WALL_CATCH_ALL,
        )
        .expect("parse");

        let lowered = case.lower().expect("lower");
        assert_eq!(lowered.block.cyclic, vec![0]);
        assert_eq!(lowered.block.patch_type[0], "cyclic");
        assert_eq!(lowered.block.patch_type[1], "cyclic");
        assert_eq!(lowered.block.patch_name[0], "xa");
        assert_eq!(lowered.block.patch_name[1], "xb");
        // Untouched by the pairing.
        assert_eq!(lowered.block.patch_type[2], "wall");
        assert_eq!(lowered.block.patch_type[4], "wall");

        // And it actually builds into a real, checkable mesh.
        let hm = crate::blockgen::build_mesh(&lowered.block).expect("build_mesh");
        assert!(hm.patches.iter().any(|p| p.kind == crate::mesh::PatchKind::Cyclic));
    }

    /// `ya`/`za` are not opposite faces of one axis - a §13.4-shaped error,
    /// not a panic or a silently wrong pairing.
    #[test]
    fn a_cyclic_pair_naming_two_non_opposite_faces_is_an_error() {
        let _g = crate::io::contract::permissive_test_guard();
        crate::io::contract::set_permissive(false);
        let case = case_with_mesh_and_patches(
            r#", "cyclic": [ { "a": "ya", "b": "za", "transform": "translate" } ]"#,
            WALL_CATCH_ALL,
        )
        .expect("parse");
        let err = match case.lower() {
            Err(e) => e.to_string(),
            Ok(_) => panic!("expected lower() to fail"),
        };
        assert!(err.contains("ya"), "{err}");
        assert!(err.contains("za"), "{err}");
    }

    /// SPEC-LIT §31.1: only `translate` exists; `rotate` is a recognised,
    /// unimplemented setting - a §13.4 error naming `translate`, not a parse
    /// failure (the field itself deserialises fine as a plain string).
    #[test]
    fn a_rotate_cyclic_transform_is_a_13_4_error_naming_translate() {
        let _g = crate::io::contract::permissive_test_guard();
        crate::io::contract::set_permissive(false);
        let case = case_with_mesh_and_patches(
            r#", "cyclic": [ { "a": "xa", "b": "xb", "transform": "rotate" } ]"#,
            WALL_CATCH_ALL,
        )
        .expect("parse");
        let err = match case.lower() {
            Err(e) => e.to_string(),
            Ok(_) => panic!("expected lower() to fail"),
        };
        assert!(err.contains("translate"), "{err}");
        assert!(err.contains("rotate"), "{err}");
    }

    /// Point 2 of SPEC-LIT §31.1: a patch named in a cyclic pair may not ALSO
    /// be named by a `patches[]` rule - here `xa` gets its own explicit wall
    /// rule ahead of the catch-all, directly contradicting the pairing.
    #[test]
    fn a_cyclic_patch_named_by_a_patches_rule_is_an_error_naming_both() {
        let _g = crate::io::contract::permissive_test_guard();
        crate::io::contract::set_permissive(false);
        let case = case_with_mesh_and_patches(
            r#", "cyclic": [ { "a": "xa", "b": "xb", "transform": "translate" } ]"#,
            r#"[ { "match": "xa", "kind": "wall" }, { "match": ".*", "kind": "wall" } ]"#,
        )
        .expect("parse");
        let err = match case.lower() {
            Err(e) => e.to_string(),
            Ok(_) => panic!("expected lower() to fail"),
        };
        assert!(err.contains("xa"), "{err}");
        assert!(err.contains("xb"), "{err}");
    }

    /// The mandatory catch-all is not itself a contradiction - every case is
    /// expected to end with one, and it says nothing about `xa`/`xb`
    /// specifically.
    #[test]
    fn the_mandatory_catch_all_does_not_conflict_with_a_cyclic_pair() {
        let case = case_with_mesh_and_patches(
            r#", "cyclic": [ { "a": "xa", "b": "xb", "transform": "translate" } ]"#,
            WALL_CATCH_ALL,
        )
        .expect("parse");
        case.lower().expect("the catch-all alone must not conflict with a cyclic pair");
    }

    /// SPEC-LIT §34.2: the single-pair limit is gone - two pairs (a plane
    /// channel periodic in x and y) both close into the block, and the third
    /// axis is untouched.
    #[test]
    fn two_cyclic_pairs_both_lower_onto_the_block_spec() {
        let case = case_with_mesh_and_patches(
            r#", "cyclic": [
                { "a": "xa", "b": "xb", "transform": "translate" },
                { "a": "ya", "b": "yb", "transform": "translate" }
            ]"#,
            WALL_CATCH_ALL,
        )
        .expect("parse");
        let lowered = case.lower().expect("two pairs must both lower");

        assert_eq!(lowered.block.cyclic, vec![0, 1]);
        assert_eq!(lowered.block.patch_type[0], "cyclic");
        assert_eq!(lowered.block.patch_type[1], "cyclic");
        assert_eq!(lowered.block.patch_type[2], "cyclic");
        assert_eq!(lowered.block.patch_type[3], "cyclic");
        assert_eq!(lowered.block.patch_type[4], "wall", "z is untouched by either pair");
        assert_eq!(lowered.block.patch_type[5], "wall");

        let hm = crate::blockgen::build_mesh(&lowered.block).expect("build_mesh");
        assert_eq!(
            hm.patches.iter().filter(|p| p.kind == crate::mesh::PatchKind::Cyclic).count(),
            4,
            "both pairs' four patches must all come back as PatchKind::Cyclic"
        );
    }

    /// SPEC-LIT §34.2: three pairs is a fully periodic box - every one of the
    /// six patches is cyclic and none is left as the mesh's default wall.
    #[test]
    fn three_cyclic_pairs_close_a_fully_periodic_box() {
        let case = case_with_mesh_and_patches(
            r#", "cyclic": [
                { "a": "xa", "b": "xb", "transform": "translate" },
                { "a": "ya", "b": "yb", "transform": "translate" },
                { "a": "za", "b": "zb", "transform": "translate" }
            ]"#,
            WALL_CATCH_ALL,
        )
        .expect("parse");
        let lowered = case.lower().expect("three pairs must all lower");

        assert_eq!(lowered.block.cyclic, vec![0, 1, 2]);
        assert!(lowered.block.patch_type.iter().all(|t| t == "cyclic"));

        let hm = crate::blockgen::build_mesh(&lowered.block).expect("build_mesh");
        assert_eq!(
            hm.patches.iter().filter(|p| p.kind == crate::mesh::PatchKind::Cyclic).count(),
            6
        );
    }

    /// SPEC-LIT §34.2: "an axis may appear in at most one pair" - here `x` is
    /// claimed twice (once via `xa`/`xb` directly, once via a second pair
    /// that also resolves to axis 0), which must be refused naming the axis
    /// rather than silently pairing whichever came first.
    #[test]
    fn an_axis_named_by_two_cyclic_pairs_is_an_error_naming_the_axis() {
        let _g = crate::io::contract::permissive_test_guard();
        crate::io::contract::set_permissive(false);
        let case = case_with_mesh_and_patches(
            r#", "cyclic": [
                { "a": "xa", "b": "xb", "transform": "translate" },
                { "a": "xa", "b": "xb", "transform": "translate" }
            ]"#,
            WALL_CATCH_ALL,
        )
        .expect("parse");
        let err = match case.lower() {
            Err(e) => e.to_string(),
            Ok(_) => panic!("expected lower() to fail"),
        };
        assert!(err.contains("mesh.cyclic"), "{err}");
        assert!(err.contains('x'), "{err}");
    }

    // ---- SPEC-LIT §34.1: constraint patches (empty/symmetry) --------------

    /// A JSONC case naming the exact shape `blockgen`'s own `Cavity` preset
    /// builds (four walls, `empty` front/back - `case_block_spec`'s
    /// `CaseKind::Cavity` arm) must close to the SAME mesh as an actual
    /// OpenFOAM-format case of that shape, written to disk and read back
    /// through the ordinary `read_poly_mesh` + `build_host_mesh` path - the
    /// same cell-for-cell contract `plume_jsonc_mesh_matches_the_generated_
    /// openfoam_case_exactly` already holds `wall`/`patch` to, extended to
    /// `empty`. This is the 2-D case §34.1's own doc says JSONC could not
    /// write before ("`wall`, `inlet`, `open` and nothing else").
    #[test]
    fn a_2d_jsonc_case_with_empty_patches_matches_its_openfoam_format_twin() {
        let (nx, ny) = (8usize, 6usize);

        let text = format!(
            r#"{{
                "name": "cavity2d",
                "mesh": {{
                    "kind": "cartesian",
                    "bounds": {{ "min": [0,0,0], "max": [0.1,0.1,0.1] }},
                    "cells": [{nx}, {ny}, 1],
                    "boundaries": {{
                        "xmin": "leftWall", "xmax": "rightWall",
                        "ymin": "fixedWall", "ymax": "movingWall",
                        "zmin": "back", "zmax": "front"
                    }}
                }},
                "physics": {{
                    "gravity": [0,0,0],
                    "fluid": {{ "nu": 1e-5, "Pr": 0.71, "Prt": 0.85, "TRef": 293.15 }},
                    "buoyancy": "densityRatio"
                }},
                "patches": [
                    {{ "match": "(back|front)", "kind": "empty" }},
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
                "run": {{ "endTime": 1.0, "deltaT": 1.0 }}
            }}"#
        );
        static COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let n = COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let tmp = std::env::temp_dir();
        let jsonc_path = tmp.join(format!("case_json_empty_2d_{}_{n}.jsonc", std::process::id()));
        std::fs::write(&jsonc_path, text).unwrap();
        let case = read_case_jsonc(&jsonc_path).expect("cavity2d jsonc should parse");
        let _ = std::fs::remove_file(&jsonc_path);

        let lowered = case.lower().expect("cavity2d jsonc should lower");
        assert_eq!(lowered.block.patch_type[4], "empty");
        assert_eq!(lowered.block.patch_type[5], "empty");
        let direct = crate::blockgen::build_mesh(&lowered.block).expect("build_mesh");

        let case_dir = tmp.join(format!("case_json_empty_2d_case_{}_{n}", std::process::id()));
        let _ = std::fs::remove_dir_all(&case_dir);
        crate::blockgen::write_case(&case_dir, crate::blockgen::CaseKind::Cavity, nx, ny, 1)
            .expect("write_case(Cavity) should write an OpenFOAM-format twin");
        let raw = crate::io::polymesh::read_poly_mesh(&case_dir).expect("read the twin back");
        let via_disk = crate::io::polymesh::build_host_mesh(&raw).expect("build_host_mesh");
        let _ = std::fs::remove_dir_all(&case_dir);

        assert_eq!(direct.n_cells, via_disk.n_cells);
        assert_eq!(direct.n_internal_faces, via_disk.n_internal_faces);
        assert_eq!(direct.n_boundary_faces, via_disk.n_boundary_faces);
        assert_eq!(direct.owner, via_disk.owner);
        assert_eq!(direct.neighbour, via_disk.neighbour);
        assert_eq!(direct.v, via_disk.v, "cell volumes must be bit-identical");
        assert_eq!(direct.c, via_disk.c, "cell centres must be bit-identical");
        assert_eq!(direct.sf, via_disk.sf);
        assert_eq!(direct.b_sf, via_disk.b_sf, "boundary face area vectors must be bit-identical");
        assert_eq!(direct.b_cf, via_disk.b_cf);

        assert_eq!(direct.patches.len(), via_disk.patches.len());
        for (a, b) in direct.patches.iter().zip(via_disk.patches.iter()) {
            assert_eq!(a.name, b.name);
            assert_eq!(a.kind, b.kind);
            assert_eq!(a.start, b.start);
            assert_eq!(a.size, b.size);
        }
        assert!(
            direct.patches.iter().any(|p| p.kind == crate::mesh::PatchKind::Empty),
            "the back/front patches must have actually come back as PatchKind::Empty"
        );
    }

    /// SPEC-LIT §34.1: `empty` is only legal across a single cell - here `z`
    /// has 3, and the error must name the slot and the offending count
    /// rather than let a meaningless multi-cell "empty" patch through to the
    /// mesh builder.
    #[test]
    fn empty_on_a_multi_cell_slot_is_refused_naming_the_count() {
        let _g = crate::io::contract::permissive_test_guard();
        crate::io::contract::set_permissive(false);
        let case = case_with_mesh_and_patches(
            "",
            r#"[ { "match": "(za|zb)", "kind": "empty" }, { "match": ".*", "kind": "wall" } ]"#,
        )
        .expect("parse");
        // `case_with_mesh_and_patches`'s fixture mesh has cells: [4, 6, 8] -
        // z has 8 cells, not 1.
        let err = match case.lower() {
            Err(e) => e.to_string(),
            Ok(_) => panic!("expected lower() to fail"),
        };
        assert!(err.contains("empty"), "{err}");
        assert!(err.contains('8'), "{err}");
        assert!(err.contains("za") || err.contains("zmin"), "{err}");
    }

    /// SPEC-LIT §34.1: a per-field BC on a CONSTRAINT rule (`empty` or
    /// `symmetry`) is a §13.4 error naming the field - the constraint
    /// decides every field, so a rule that also sets one has misunderstood
    /// something the reader can catch. Checked for one representative field
    /// of each per-field family (`U`, a plain [`ScalarBc`]; `k`, a
    /// [`TurbBc`]) - `validate_constraint_rules` handles the rest of the
    /// eleven identically.
    #[test]
    fn a_per_field_bc_on_a_constraint_rule_is_refused_naming_the_field() {
        let _g = crate::io::contract::permissive_test_guard();
        crate::io::contract::set_permissive(false);

        let u_case = case_with_mesh_and_patches(
            "",
            r#"[
                { "match": "(za|zb)", "kind": "empty", "U": { "type": "zeroGradient" } },
                { "match": ".*", "kind": "wall" }
            ]"#,
        )
        .expect("parse");
        let err = match u_case.lower() {
            Err(e) => e.to_string(),
            Ok(_) => panic!("expected lower() to fail on a U entry"),
        };
        assert!(err.contains("empty"), "{err}");
        assert!(err.contains('U'), "{err}");

        let k_case = case_with_mesh_and_patches(
            "",
            r#"[
                { "match": "(ya|yb)", "kind": "symmetry", "k": { "type": "zeroGradient" } },
                { "match": ".*", "kind": "wall" }
            ]"#,
        )
        .expect("parse");
        let err = match k_case.lower() {
            Err(e) => e.to_string(),
            Ok(_) => panic!("expected lower() to fail on a k entry"),
        };
        assert!(err.contains("symmetry"), "{err}");
        assert!(err.contains('k'), "{err}");
    }

    /// SPEC-LIT §34.2: "a pair and a constraint patch on the same slot is a
    /// §13.4 error naming both" - here `xa`/`xb` are both a cyclic pair AND
    /// (via the mandatory-catch-all-preceding rule) named `empty`, which is
    /// a direct contradiction about the same faces.
    #[test]
    fn a_cyclic_pair_naming_an_empty_patch_is_an_error_naming_both() {
        let _g = crate::io::contract::permissive_test_guard();
        crate::io::contract::set_permissive(false);
        let case = case_with_mesh_and_patches(
            r#", "cyclic": [ { "a": "xa", "b": "xb", "transform": "translate" } ]"#,
            r#"[ { "match": "xa", "kind": "empty" }, { "match": ".*", "kind": "wall" } ]"#,
        )
        .expect("parse");
        let err = match case.lower() {
            Err(e) => e.to_string(),
            Ok(_) => panic!("expected lower() to fail"),
        };
        assert!(err.contains("xa"), "{err}");
        assert!(err.contains("xb"), "{err}");
    }

    /// `blockgen`'s own cases spell this exact shape
    /// (`b.y.expansion = 20.0; b.y.two_sided = true;`, `src/blockgen.rs`'s
    /// `case_block_spec`); the JSONC reader must lower the mirrored JSON key
    /// onto exactly the same two `GradedAxis` fields, on whichever axis names
    /// it.
    #[test]
    fn grading_lowers_onto_graded_axis_exactly_as_blockgen_cases_use_it() {
        let case = minimal_case_with_mesh(
            r#", "grading": { "y": { "expansion": 20.0, "twoSided": true } } "#,
        );
        let lowered = case.lower().expect("case with grading should lower");

        assert_eq!(lowered.block.y.expansion, 20.0);
        assert!(lowered.block.y.two_sided);
        // x and z are untouched.
        assert_eq!(lowered.block.x.expansion, 1.0);
        assert!(!lowered.block.x.two_sided);
        assert_eq!(lowered.block.z.expansion, 1.0);
        assert!(!lowered.block.z.two_sided);

        // And it actually reaches `graded_nodes`: the y-axis is no longer
        // uniformly spaced.
        let nodes = crate::blockgen::graded_nodes(&lowered.block.y);
        let first_cell = nodes[1] - nodes[0];
        let mid_cell = nodes[nodes.len() / 2] - nodes[nodes.len() / 2 - 1];
        assert!(mid_cell > first_cell * 5.0, "two-sided grading should make the centre cell much larger than the wall cell");
    }

    /// One-sided grading (`twoSided` defaulted to `false`) reaches
    /// `GradedAxis` too - the default is one end plain, not "no grading at
    /// all".
    #[test]
    fn one_sided_grading_defaults_two_sided_to_false() {
        let case = minimal_case_with_mesh(r#", "grading": { "x": { "expansion": 4.0 } } "#);
        let lowered = case.lower().expect("lower");
        assert_eq!(lowered.block.x.expansion, 4.0);
        assert!(!lowered.block.x.two_sided);
    }

    #[test]
    fn zero_or_negative_expansion_is_a_strict_error_naming_the_axis() {
        // Strict-mode behaviour depends on the process-wide permissive flag
        // staying `false` - take the same guard the permissive-toggling
        // tests do so this cannot observe another test's `set_permissive(true)`
        // mid-run.
        let _g = crate::io::contract::permissive_test_guard();
        crate::io::contract::set_permissive(false);

        let case = minimal_case_with_mesh(r#", "grading": { "z": { "expansion": 0.0 } } "#);
        let err = case.lower().err().expect("expected an error").to_string();
        assert!(err.contains("mesh.grading.z.expansion"), "{err}");
        assert!(err.contains("-permissive"), "{err}");

        let case = minimal_case_with_mesh(r#", "grading": { "y": { "expansion": -3.0 } } "#);
        let err = case.lower().err().expect("expected an error").to_string();
        assert!(err.contains("mesh.grading.y.expansion"), "{err}");
        assert!(err.contains("-3"), "{err}");
    }

    #[test]
    fn permissive_substitutes_uniform_for_a_non_physical_expansion() {
        let _g = crate::io::contract::permissive_test_guard();
        crate::io::contract::reset_warnings();
        crate::io::contract::set_permissive(true);

        let case = minimal_case_with_mesh(r#", "grading": { "y": { "expansion": -1.0 } } "#);
        let lowered = case.lower().expect("permissive should substitute, not fail");
        assert_eq!(lowered.block.y.expansion, 1.0);
        assert!(!lowered.block.y.two_sided);

        crate::io::contract::set_permissive(false);
    }

    #[test]
    fn unknown_grading_key_names_the_json_path() {
        let text = r#"{
            "name": "bad",
            "mesh": {
                "kind": "cartesian",
                "bounds": { "min": [0,0,0], "max": [1,1,1] },
                "cells": [4,4,4],
                "boundaries": {
                    "xmin": "a", "xmax": "a", "ymin": "a",
                    "ymax": "a", "zmin": "a", "zmax": "a"
                },
                "grading": { "y": { "expansion": 2.0, "bogus": true } }
            },
            "physics": {
                "gravity": [0,0,0],
                "fluid": { "nu": 1e-5, "Pr": 0.71, "Prt": 0.85, "TRef": 293.15 },
                "buoyancy": "densityRatio"
            },
            "patches": [ { "match": ".*", "kind": "wall" } ],
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

        let path = std::env::temp_dir().join("case_json_unknown_grading_key_test.jsonc");
        std::fs::write(&path, text).unwrap();
        let err = read_case_jsonc(&path).unwrap_err().to_string();
        let _ = std::fs::remove_file(&path);

        assert!(err.contains("mesh.grading.y"), "{err}");
        assert!(err.contains("bogus"), "{err}");
    }

    /// **The shipped schema has to BE the generated one.**
    ///
    /// `docs/schema/case-1.json` is what a case file's `$schema` points a
    /// human's editor at, and it is a copy of [`emit_schema`]'s output. A
    /// copy drifts: when SPEC-LIT S44 regenerated it, the shipped file was
    /// still missing every field two later sections had added months
    /// earlier, so an editor validating against it flagged a valid case as
    /// invalid. That is the documentation half of S13.4.1's defect, and this
    /// is the test that stops it recurring.
    ///
    /// Line endings are normalised because git may check the file out with
    /// CRLF; nothing else is.
    #[test]
    fn the_shipped_schema_is_the_generated_one() {
        let path =
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../docs/schema/case-1.json");
        let shipped = std::fs::read_to_string(&path)
            .unwrap_or_else(|e| panic!("cannot read {}: {e}", path.display()));
        let norm = |s: &str| s.replace("
", "
").trim_end().to_string();
        assert_eq!(
            norm(&shipped),
            norm(&emit_schema()),
            "docs/schema/case-1.json has drifted from emit_schema(); regenerate it"
        );
    }

    #[test]
    fn schema_documents_the_grading_block() {
        let text = emit_schema();
        assert!(text.contains("\"grading\""), "schema missing grading: {text}");
        assert!(text.contains("\"twoSided\""), "schema missing twoSided: {text}");
        assert!(text.contains("\"expansion\""), "schema missing expansion: {text}");
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

    /// `case.lower()`, with the turbulence model switched to `laminar` first
    /// when `wt` is `LowRe` - `plume_case()`'s own model is `kEpsilon`, and
    /// SPEC-LIT §32's second finding
    /// (`crate::io::case::validate_low_re_wall_treatment`) correctly refuses
    /// `lowRe` under it: no model this solver implements is valid at `lowRe`
    /// today (that gate has its own tests, below). The tests using this
    /// helper are exercising a DIFFERENT thing, the PRESET'S ROW SHAPE
    /// (SPEC-LIT §29.1's table) - a purely mechanical string expansion that
    /// does not depend on which model is named - so they sidestep the gate
    /// by naming the one model it always leaves alone (`laminar`, `nu_t = 0`
    /// regardless of wall treatment) rather than by substituting the row
    /// away with `-permissive`, which would defeat the point of the test.
    fn lower_permitting_low_re(mut case: JsonCase, wt: WallTreatmentKind) -> LoweredCase {
        if wt == WallTreatmentKind::LowRe {
            case.turbulence.as_mut().unwrap().model = "laminar".to_string();
        }
        case.lower().unwrap_or_else(|e| panic!("{wt:?} should lower: {e}"))
    }

    /// Each preset expands to exactly its row - SPEC-LIT §29.1's table,
    /// string-level, through the JSONC lowering path.
    #[test]
    fn each_preset_expands_to_exactly_its_row_through_jsonc() {
        for (wt, nut_want, k_want, eps_want, omega_want) in [
            (WallTreatmentKind::Standard, "nutkWallFunction", "kqRWallFunction", "epsilonWallFunction", "omegaWallFunction"),
            (WallTreatmentKind::Spalding, "nutUWallFunction", "kqRWallFunction", "epsilonWallFunction", "omegaWallFunction"),
            (WallTreatmentKind::Rough, "nutkRoughWallFunction", "kqRWallFunction", "epsilonWallFunction", "omegaWallFunction"),
            (WallTreatmentKind::LowRe, "nutLowReWallFunction", "kLowReWallFunction", "fixedValue", "zeroGradient"),
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
            let lowered = lower_permitting_low_re(case, wt);
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
            let lowered = lower_permitting_low_re(case, wt);
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
            let lowered = lower_permitting_low_re(case, wt);
            let p = a_wall_patch_name(&lowered);
            let t = lowered.t_field.unwrap();
            assert_eq!(t.boundary[&p].type_name, "thermalWallFunction", "{wt:?}");
            assert_eq!(t.boundary[&p].value, vec![400.0 as Scalar], "{wt:?}");
        }
    }

    /// SPEC-LIT §32.2's fixed-flux wall, through JSONC: the type stays
    /// `fixedFluxTemperature` (not folded into `thermalWallFunction` or
    /// `fixedGradient`) and `q` round-trips through `PatchFieldSpec::extra`
    /// exactly as `Ks`/`Cs` already do for the rough-wall condition - on
    /// every wall-treatment row, `lowRe` included, since SPEC-LIT §32.2's own
    /// point is that this ONE condition is exact on both a wall-function and
    /// a resolved mesh.
    #[test]
    fn fixed_flux_temperature_carries_its_own_q_through_jsonc() {
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
                r.t = Some(ScalarBc::FixedFluxTemperature { q: 500.0 });
                if wt == WallTreatmentKind::Rough {
                    r.ks = Some(0.001);
                }
            }
            let lowered = lower_permitting_low_re(case, wt);
            let p = a_wall_patch_name(&lowered);
            let t = lowered.t_field.unwrap();
            assert_eq!(t.boundary[&p].type_name, "fixedFluxTemperature", "{wt:?}");
            let q: f64 = t.boundary[&p].extra["q"].parse().unwrap();
            assert!((q - 500.0).abs() < 1e-9, "{wt:?}: q = {q}");
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

    /// SPEC-LIT §32's second finding, wired into the JSONC lowering path:
    /// `wallTreatment lowRe` under `plume_case()`'s own `kEpsilon` (a
    /// high-Reynolds-number closure with no near-wall damping) is refused in
    /// strict mode, naming the model - not a mesh problem, `lowRe` itself is
    /// invalid under this model at any resolution.
    #[test]
    fn lowre_wall_treatment_under_kepsilon_is_refused_naming_the_model() {
        let _g = crate::io::contract::permissive_test_guard();
        crate::io::contract::set_permissive(false);

        let mut case = plume_case();
        case.turbulence.as_mut().unwrap().wall_treatment = WallTreatmentKind::LowRe;
        clear_explicit_turbulence(wall_rule_mut(&mut case));
        let err = match case.lower() {
            Err(e) => e,
            Ok(_) => panic!("lowRe under kEpsilon must be refused"),
        };
        let s = err.to_string();
        assert!(s.contains("kEpsilon"), "{s}");
        assert!(s.contains("standard"), "{s}");
    }

    /// The other direction of the same gate: `-permissive` substitutes
    /// `standard` (the full wall-function row) and the case lowers - `nut`
    /// on the wall ends up `nutkWallFunction`, not `nutLowReWallFunction`.
    #[test]
    fn permissive_substitutes_standard_for_lowre_under_kepsilon() {
        let _g = crate::io::contract::permissive_test_guard();
        crate::io::contract::reset_warnings();
        crate::io::contract::set_permissive(true);

        let mut case = plume_case();
        case.turbulence.as_mut().unwrap().wall_treatment = WallTreatmentKind::LowRe;
        clear_explicit_turbulence(wall_rule_mut(&mut case));
        let lowered = case.lower().expect("-permissive resolves it");
        let p = a_wall_patch_name(&lowered);
        assert_eq!(
            lowered.nut_field.unwrap().boundary[&p].type_name,
            "nutkWallFunction",
            "-permissive must substitute the standard row, not leave lowRe in place"
        );

        crate::io::contract::set_permissive(false);
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
            "fixedValue",
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
    /// `"kind": "LES"` must reach the LES branch of the selector - the
    /// schema advertised it while the lowering dropped it, so a Deardorff
    /// case was answered with "needs `simulationType LES;`", a setting the
    /// JSONC format does not spell.
    #[test]
    fn jsonc_kind_les_routes_to_the_les_selector() {
        let mut case = minimal_case_with_mesh("");
        case.turbulence = Some(JsonTurbulence {
            kind: TurbulenceKind::Les,
            model: "Deardorff".to_string(),
            wall_functions: JsonWallFunctions { kappa: 0.41, e: 9.8 },
            wall_treatment: WallTreatmentKind::Standard,
        });
        let lowered = case.lower().expect("lowers");
        let cc = lowered.to_case_controls();
        let sel = crate::models::registry::select_turbulence_model(&cc)
            .expect("selects without the simulationType complaint");
        assert!(
            format!("{sel:?}").contains("Deardorff"),
            "expected the Deardorff LES selection, got {sel:?}"
        );

        // And the RAS route is untouched: the same case with kind RAS +
        // kOmegaSST still selects SST through model_name.
        let mut ras = minimal_case_with_mesh("");
        ras.turbulence = Some(JsonTurbulence {
            kind: TurbulenceKind::Ras,
            model: "kOmegaSST".to_string(),
            wall_functions: JsonWallFunctions { kappa: 0.41, e: 9.8 },
            wall_treatment: WallTreatmentKind::Standard,
        });
        let cc2 = ras.lower().expect("lowers").to_case_controls();
        let sel2 = crate::models::registry::select_turbulence_model(&cc2)
            .expect("RAS route");
        assert!(format!("{sel2:?}").contains("KOmegaSST"), "got {sel2:?}");
    }

    // ------------------------------------------------------------------
    //  SPEC-LIT §31.3: the transient/algorithm contract, JSONC side
    // ------------------------------------------------------------------

    /// The exact shape a shipped transient case had: `endTime > 0`,
    /// `ddt` not `steadyState`, and `numerics.algorithm.kind` still `SIMPLE`.
    #[test]
    fn a_transient_jsonc_case_naming_simple_is_a_lower_error() {
        let _g = crate::io::contract::permissive_test_guard();
        crate::io::contract::set_permissive(false);

        let mut case = minimal_case_with_mesh("");
        case.numerics.ddt = "Euler".to_string();
        case.run.end_time = 20.0;
        case.run.delta_t = 0.01;
        // numerics.algorithm.kind is already Simple from minimal_case_with_mesh.

        let err = case.lower().err().expect("transient + SIMPLE must be rejected").to_string();
        assert!(err.contains("SIMPLE"), "{err}");
        assert!(err.contains("PISO"), "{err}");
        assert!(err.contains("PIMPLE"), "{err}");
        assert!(err.contains("-permissive"), "{err}");
    }

    /// The same mismatch from the other side: `steadyState` naming `PISO`.
    #[test]
    fn a_steady_jsonc_case_naming_piso_is_a_lower_error() {
        let _g = crate::io::contract::permissive_test_guard();
        crate::io::contract::set_permissive(false);

        let mut case = minimal_case_with_mesh("");
        case.numerics.algorithm.kind = AlgorithmKind::Piso;
        // ddt stays "steadyState" from minimal_case_with_mesh.

        let err = case.lower().err().expect("steady + PISO must be rejected").to_string();
        assert!(err.contains("PISO"), "{err}");
        assert!(err.contains("SIMPLE"), "{err}");
        assert!(err.contains("-permissive"), "{err}");
    }

    /// `-permissive` substitutes `PIMPLE` with one outer corrector, and the
    /// case still lowers.
    #[test]
    fn permissive_substitutes_pimple_and_the_jsonc_case_still_lowers() {
        let _g = crate::io::contract::permissive_test_guard();
        crate::io::contract::set_permissive(true);

        let mut case = minimal_case_with_mesh("");
        case.numerics.ddt = "Euler".to_string();
        case.run.end_time = 20.0;
        case.run.delta_t = 0.01;

        let lowered = case.lower().expect("-permissive must not fail");
        assert_eq!(lowered.algorithm.dict, "PIMPLE");
        assert_eq!(lowered.algorithm.n_outer_correctors, 1);

        crate::io::contract::set_permissive(false);
    }

    /// A transient case naming `PIMPLE` (the shape that fixed
    /// shape) lowers cleanly - the contract does not reject the combination
    /// it exists to require.
    #[test]
    fn a_transient_jsonc_case_naming_pimple_lowers_cleanly() {
        let mut case = minimal_case_with_mesh("");
        case.numerics.algorithm.kind = AlgorithmKind::Pimple;
        case.numerics.algorithm.outer_correctors = Some(1);
        case.numerics.ddt = "Euler".to_string();
        case.run.end_time = 20.0;
        case.run.delta_t = 0.01;

        let lowered = case.lower().expect("transient + PIMPLE must lower");
        assert_eq!(lowered.algorithm.dict, "PIMPLE");
    }

    /// **SPEC-LIT §13.4.** A case written for the reacting-medium solver is
    /// refused BY NAME, and the refusal survives `-permissive`.
    ///
    /// This is [`refuse_held_back_blocks`]'s own gate. The failure it stands
    /// against is not a parse error: it is a burner case read cleanly by an
    /// engine with no chemistry in it, run to completion at `q''' = 0`, and
    /// reporting a wall heat balance that closes because there was no fire.
    /// The names must therefore still be RECOGNISED here - "unknown field" is
    /// a diagnosis of a typo, and this is not one.
    #[test]
    fn a_case_written_for_a_reacting_medium_is_refused_by_name() {
        let _g = crate::io::contract::permissive_test_guard();
        // ON. A note would be waived here; this is an `Error::Config` and is
        // not, for the same reason `RadiationConfig::from_case`'s is not.
        crate::io::contract::set_permissive(true);

        let base = std::fs::read_to_string(example_path()).expect("read the example case");
        let mut got: Vec<(&str, Option<String>)> = Vec::new();
        for (entry, insert, after) in [
            ("physics.fire.combustion", r#""fire": { "combustion": {} },"#, "\"physics\": {"),
            (
                "physics.fire.radiation",
                r#""fire": { "radiation": { "model": "P1", "a": 0.5 } },"#,
                "\"physics\": {",
            ),
            (
                "physics.fire.soot",
                r#""fire": { "soot": { "model": "prescribedYield" } },"#,
                "\"physics\": {",
            ),
            ("initial.Y_F", r#""Y_F": 0.0,"#, "\"initial\": {"),
        ] {
            let at = base.find(after).expect("the example case must have the block");
            let mut text = base.clone();
            text.insert_str(at + after.len(), insert);
            got.push((
                entry,
                read_case_jsonc_str(&text, "fixture").err().map(|e| e.to_string()),
            ));
        }
        // Process-wide: restore before an assertion can panic and leave a
        // later strict-mode test observing it.
        crate::io::contract::set_permissive(false);

        for (entry, e) in got {
            let e = e.unwrap_or_else(|| panic!("{entry} was read and not refused"));
            assert!(e.contains(entry), "the refusal must name {entry}: {e}");
            assert!(
                e.contains("13.4"),
                "the refusal must name the contract it is made under: {e}"
            );
        }
        // And the example case itself, unmodified, is read - or every
        // assertion above would pass for the wrong reason.
        read_case_jsonc_str(&base, "fixture").expect("the example case must still read");
    }

    /// SPEC-LIT §31.3's regression: every `.jsonc` case this project ships
    /// must pass the contract - a shipped transient case used not to.
    #[test]
    fn every_shipped_jsonc_case_passes_the_transient_algorithm_contract() {
        let _g = crate::io::contract::permissive_test_guard();
        crate::io::contract::set_permissive(false);

        let cases_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../cases");
        let entries = std::fs::read_dir(&cases_dir)
            .unwrap_or_else(|e| panic!("{}: {e}", cases_dir.display()));

        let mut checked = 0usize;
        for entry in entries {
            let path = entry.expect("dir entry").path();
            // SPEC-LIT §47.14: a `*.cht.jsonc` is a MULTI-REGION CONDUCTION
            // case (`crate::io::case_cht`), a different document with no
            // `physics`, no `U` and no algorithm at all. It is not this
            // reader's to lower, and its own shipped-case scan is
            // `io::case_cht::tests::every_shipped_cht_case_lowers`.
            if path.file_name().and_then(|f| f.to_str()).is_some_and(|f| f.ends_with(".cht.jsonc")) {
                continue;
            }
            // SPEC-LIT S55.6: and a `*.dc.jsonc` is a DATA-CENTRE ROOM case
            // (`crate::io::case_dc`) - again a different document, with its
            // own `room`, `fans`, `tiles` and `racks` blocks and no `physics`.
            // Its own shipped-case scan is
            // `io::case_dc::tests::the_base_case_lowers`, which reads
            // `cases/coldAisle.dc.jsonc` and lowers it.
            if path.file_name().and_then(|f| f.to_str()).is_some_and(|f| f.ends_with(".dc.jsonc")) {
                continue;
            }
            if path.extension().and_then(|e| e.to_str()) == Some("jsonc") {
                let case = read_case_jsonc(&path)
                    .unwrap_or_else(|e| panic!("{}: {e}", path.display()));
                // STRICT mode, with nothing excepted. Two cases used to be
                // retried under `-permissive` here because they named
                // `wallTreatment lowRe` with `kEpsilon`, which SPEC-LIT §33
                // refuses; they have since been retired to `cases/retired/`
                // (SPEC-LIT §13.4.3), which this walk does not descend into.
                // The escape hatch went with them, so this test now says what
                // `cases/README.md` says: every `.jsonc` at the top level of
                // `cases/` lowers cleanly as shipped. A new case that does not
                // fails here rather than in a licensee's fresh clone.
                case.lower().unwrap_or_else(|e| panic!("{}: {e}", path.display()));
                checked += 1;
            }
        }
        assert!(
            checked >= 4,
            "expected several .jsonc cases under {}, found {checked}",
            cases_dir.display()
        );
    }

    // ---- SPEC-LIT §18/§31.1: sources[] --------------------------------------

    /// [`minimal_case_with_mesh`], with a top-level `sources` array spliced
    /// in - the momentum-source tests need one and no other helper here
    /// takes arbitrary top-level JSON.
    fn minimal_case_with_sources(sources_json: &str) -> Result<JsonCase> {
        let text = format!(
            r#"{{
                "name": "sourcesTest",
                "mesh": {{
                    "kind": "cartesian",
                    "bounds": {{ "min": [0,0,0], "max": [1,1,1] }},
                    "cells": [4, 6, 8],
                    "boundaries": {{
                        "xmin": "xa", "xmax": "xb", "ymin": "ya",
                        "ymax": "yb", "zmin": "za", "zmax": "zb"
                    }}
                }},
                "physics": {{
                    "gravity": [0,0,0],
                    "fluid": {{ "nu": 1e-5, "Pr": 0.71, "Prt": 0.85, "TRef": 293.15 }},
                    "buoyancy": "densityRatio"
                }},
                "patches": [ {{ "match": ".*", "kind": "wall" }} ],
                "initial": {{ "U": [0,0,0], "p": 0.0 }},
                "numerics": {{
                    "algorithm": {{ "kind": "SIMPLE" }},
                    "ddt": "steadyState",
                    "div": {{ "default": "Gauss upwind" }},
                    "grad": "Gauss linear",
                    "laplacian": {{ "snGrad": "corrected", "nonOrthogonalCorrectors": 0 }},
                    "solvers": []
                }},
                "run": {{ "endTime": 1.0, "deltaT": 1.0 }},
                "sources": [{sources_json}]
            }}"#
        );
        static COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let n = COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!(
            "case_json_sources_test_{}_{n}.jsonc",
            std::process::id()
        ));
        std::fs::write(&path, text).unwrap();
        let case = read_case_jsonc(&path);
        let _ = std::fs::remove_file(&path);
        case
    }

    /// SPEC-LIT §37.4 through the JSONC route, end to end: the entry is
    /// recognised, absent means `constant`, and an unrecognised spelling comes
    /// back as a §13.4 error NAMING the alternatives rather than as serde's
    /// "unknown variant" - which is why it is a `String` in [`JsonFluid`] and
    /// not an enum.
    #[test]
    fn the_prt_model_lowers_and_an_unknown_spelling_names_the_alternatives() {
        let _g = crate::io::contract::permissive_test_guard();

        fn case_with_fluid(fluid: &str) -> Result<JsonCase> {
            let text = format!(
                r#"{{
                    "name": "prtModelTest",
                    "mesh": {{
                        "kind": "cartesian",
                        "bounds": {{ "min": [0,0,0], "max": [1,1,1] }},
                        "cells": [4, 6, 8],
                        "boundaries": {{
                            "xmin": "xa", "xmax": "xb", "ymin": "ya",
                            "ymax": "yb", "zmin": "za", "zmax": "zb"
                        }}
                    }},
                    "physics": {{
                        "gravity": [0,0,0],
                        "fluid": {fluid},
                        "buoyancy": "densityRatio"
                    }},
                    "patches": [ {{ "match": ".*", "kind": "wall" }} ],
                    "initial": {{ "U": [0,0,0], "p": 0.0 }},
                    "numerics": {{
                        "algorithm": {{ "kind": "SIMPLE" }},
                        "ddt": "steadyState",
                        "div": {{ "default": "Gauss upwind" }},
                        "grad": "Gauss linear",
                        "laplacian": {{ "snGrad": "corrected", "nonOrthogonalCorrectors": 0 }},
                        "solvers": []
                    }},
                    "run": {{ "endTime": 1.0, "deltaT": 1.0 }}
                }}"#
            );
            static COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
            let n = COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            let path = std::env::temp_dir()
                .join(format!("case_json_prt_test_{}_{n}.jsonc", std::process::id()));
            std::fs::write(&path, text).unwrap();
            let case = read_case_jsonc(&path);
            let _ = std::fs::remove_file(&path);
            case
        }

        // Absent: `constant`, which is what every case written before S37 means.
        let plain = case_with_fluid(r#"{ "nu": 1e-5, "Pr": 0.71, "Prt": 0.85, "TRef": 293.15 }"#)
            .expect("parses")
            .lower()
            .expect("lowers");
        assert_eq!(plain.prt_model, PrtModel::Constant);

        // Named: selected.
        let kc = case_with_fluid(
            r#"{ "nu": 1e-5, "Pr": 0.71, "Prt": 0.85, "PrtModel": "KaysCrawford", "TRef": 293.15 }"#,
        )
        .expect("parses")
        .lower()
        .expect("lowers");
        assert_eq!(kc.prt_model, PrtModel::KaysCrawford);

        // Misspelled: a S13.4 error, naming the entry, the value and the menu.
        let case = case_with_fluid(
            r#"{ "nu": 1e-5, "Pr": 0.71, "Prt": 0.85, "PrtModel": "kaysCrawfordJischa", "TRef": 293.15 }"#,
        )
        .expect("parses");
        let err = match case.lower() {
            Err(e) => e.to_string(),
            Ok(_) => panic!("expected lower() to refuse an unknown PrtModel"),
        };
        assert!(err.contains("physics.fluid.PrtModel"), "{err}");
        assert!(err.contains("kaysCrawfordJischa"), "{err}");
        assert!(err.contains("constant") && err.contains("KaysCrawford"), "{err}");
    }

    /// No `sources` key at all lowers to an empty list - every case this
    /// reader has ever built before this feature existed.
    #[test]
    fn a_case_with_no_sources_key_lowers_to_an_empty_list() {
        let case = minimal_case_with_mesh("");
        let lowered = case.lower().expect("lower");
        assert!(lowered.sources.is_empty());
    }

    /// A `momentumSource` on `U` lowers to exactly the
    /// [`crate::sources::SourceSpec`] the OpenFOAM `constant/fvSources`
    /// `momentumSource`/`selection all` route would build - one registry,
    /// two ways in.
    #[test]
    fn a_momentum_source_lowers_to_a_whole_domain_body_force() {
        let case = minimal_case_with_sources(
            r#"{ "type": "momentumSource", "field": "U", "bodyForce": [3.9, 0.0, -1.5] }"#,
        )
        .expect("parses");
        let lowered = case.lower().expect("a momentumSource on U must lower");
        assert_eq!(lowered.sources.len(), 1);
        let spec = &lowered.sources[0];
        assert_eq!(spec.field, "U");
        assert_eq!(spec.selector, crate::sources::CellSelector::All);
        match spec.term {
            Some(crate::sources::SourceTerm::BodyForce(b)) => {
                assert_eq!((b.x, b.y, b.z), (3.9, 0.0, -1.5));
            }
            other => panic!("expected SourceTerm::BodyForce, got {other:?}"),
        }
    }

    /// SPEC-LIT §18: a body force is a vector and only the momentum equation
    /// has a direction for it to point in - `field` names anything else is a
    /// §13.4 error, not a silent no-op.
    #[test]
    fn a_momentum_source_naming_a_field_other_than_u_is_an_error() {
        let case = minimal_case_with_sources(
            r#"{ "type": "momentumSource", "field": "T", "bodyForce": [1.0, 0.0, 0.0] }"#,
        )
        .expect("parses");
        let err = match case.lower() {
            Err(e) => e.to_string(),
            Ok(_) => panic!("expected lower() to fail for field \"T\""),
        };
        assert!(err.contains("\"T\""), "{err}");
    }

    /// SPEC-LIT §35.1's own worked example: `{"type": "thermostat", "target":
    /// 350.0, "tau": 0.02}` lowers to a `T`-field, whole-domain
    /// `SourceTerm::Thermostat` with both numbers preserved exactly.
    #[test]
    fn a_thermostat_lowers_to_a_whole_domain_t_source() {
        let case = minimal_case_with_sources(
            r#"{ "type": "thermostat", "target": 350.0, "tau": 0.02 }"#,
        )
        .expect("parses");
        let lowered = case.lower().expect("a thermostat must lower");
        assert_eq!(lowered.sources.len(), 1);
        let spec = &lowered.sources[0];
        assert_eq!(spec.field, "T");
        assert_eq!(spec.selector, crate::sources::CellSelector::All);
        match spec.term {
            Some(crate::sources::SourceTerm::Thermostat {
                target,
                tau,
                weighting,
                direction,
            }) => {
                assert_eq!(target, 350.0);
                assert_eq!(tau, Some(0.02));
                // SPEC-LIT §35.3.6: `weighting` omitted is `uniform`.
                assert_eq!(weighting, crate::sources::ThermostatWeighting::Uniform);
                assert_eq!(direction, None);
            }
            other => panic!("expected SourceTerm::Thermostat, got {other:?}"),
        }
    }

    /// `tau` is genuinely optional - SPEC-LIT §35.1's own default (the
    /// domain flow-through time) is resolved by the DRIVER, not the reader,
    /// so `None` here must survive the round trip rather than being filled
    /// in early.
    #[test]
    fn a_thermostat_without_tau_leaves_the_default_to_the_driver() {
        let case = minimal_case_with_sources(r#"{ "type": "thermostat", "target": 350.0 }"#)
            .expect("parses");
        let lowered = case.lower().expect("a thermostat without tau must lower");
        match lowered.sources[0].term {
            Some(crate::sources::SourceTerm::Thermostat { tau, .. }) => assert_eq!(tau, None),
            other => panic!("expected SourceTerm::Thermostat, got {other:?}"),
        }
    }

    /// A non-finite target is refused at `lower`, not left to fail deep
    /// inside the solver later (SPEC-LIT §13.4's shape: fail loudly, at the
    /// point the case said something impossible).
    #[test]
    fn a_thermostat_with_a_non_finite_target_is_an_error() {
        let case = minimal_case_with_sources(r#"{ "type": "thermostat", "target": 1e400 }"#);
        // `1e400` overflows `f64` at JSON parse time, exactly like the
        // existing `bodyForce` overflow test below.
        assert!(case.is_err() || case.unwrap().lower().is_err());
    }

    /// A `tau` of zero or negative is refused - SPEC-LIT §35.1's own
    /// relaxation time must be positive to mean anything.
    #[test]
    fn a_thermostat_with_a_non_positive_tau_is_an_error() {
        let case = minimal_case_with_sources(
            r#"{ "type": "thermostat", "target": 350.0, "tau": -1.0 }"#,
        )
        .expect("parses");
        let err = match case.lower() {
            Err(e) => e.to_string(),
            Ok(_) => panic!("expected lower() to fail for tau -1.0"),
        };
        assert!(err.contains("tau"), "{err}");
    }

    /// SPEC-LIT §35.3.7: the JSONC route carries `weighting` and
    /// `direction` through to the registry.
    #[test]
    fn a_mass_flux_thermostat_lowers_with_its_direction() {
        let case = minimal_case_with_sources(
            r#"{ "type": "thermostat", "target": 293.15, "tau": 0.02,
                 "weighting": "massFlux", "direction": [1, 0, 0] }"#,
        )
        .expect("parses");
        let lowered = case.lower().expect("a massFlux thermostat must lower");
        match lowered.sources[0].term {
            Some(crate::sources::SourceTerm::Thermostat {
                weighting,
                direction,
                ..
            }) => {
                assert_eq!(weighting, crate::sources::ThermostatWeighting::MassFlux);
                assert_eq!(direction, Some(crate::Vec3::new(1.0, 0.0, 0.0)));
            }
            other => panic!("expected SourceTerm::Thermostat, got {other:?}"),
        }
    }

    /// `direction` omitted stays `None` - SPEC-LIT §35.3.5 point 2 resolves
    /// it from the mesh's single cyclic pair, in the DRIVER, where a mesh
    /// exists.
    #[test]
    fn a_mass_flux_thermostat_may_leave_its_direction_to_the_mesh() {
        let case = minimal_case_with_sources(
            r#"{ "type": "thermostat", "target": 293.15, "weighting": "massFlux" }"#,
        )
        .expect("parses");
        let lowered = case.lower().expect("must lower");
        match lowered.sources[0].term {
            Some(crate::sources::SourceTerm::Thermostat {
                weighting,
                direction,
                ..
            }) => {
                assert_eq!(weighting, crate::sources::ThermostatWeighting::MassFlux);
                assert_eq!(direction, None);
            }
            other => panic!("expected SourceTerm::Thermostat, got {other:?}"),
        }
    }

    /// SPEC-LIT §13.4: a `weighting` this solver does not have is refused at
    /// `lower`, naming the two it does.
    #[test]
    fn an_unknown_thermostat_weighting_is_an_error() {
        let _g = crate::io::contract::permissive_test_guard();
        let case = minimal_case_with_sources(
            r#"{ "type": "thermostat", "target": 350.0, "weighting": "bulkVelocity" }"#,
        )
        .expect("parses");
        let err = match case.lower() {
            Err(e) => e.to_string(),
            Ok(_) => panic!("expected lower() to refuse an unknown weighting"),
        };
        assert!(err.contains("bulkVelocity"), "{err}");
        assert!(err.contains("uniform"), "{err}");
        assert!(err.contains("massFlux"), "{err}");
    }

    /// SPEC-LIT §35.3.5: `direction` alongside the default `uniform`
    /// weighting is refused - by `SourceTerm::validate`, which `build` runs
    /// once a mesh exists. The lowering itself records what the case said.
    #[test]
    fn a_direction_on_a_uniform_thermostat_is_refused() {
        let case = minimal_case_with_sources(
            r#"{ "type": "thermostat", "target": 350.0, "direction": [1, 0, 0] }"#,
        )
        .expect("parses");
        let lowered = case.lower().expect("lowers; the refusal is in validate");
        let err = lowered.sources[0]
            .term
            .expect("a term")
            .validate("thermostat0")
            .unwrap_err()
            .to_string();
        assert!(err.contains("uniform"), "{err}");
        assert!(err.contains("massFlux"), "{err}");
    }

    /// A `bodyForce` component past `f64`'s range is refused when the case
    /// is READ, before `lower` ever runs - this reader's own JSON number
    /// parsing already checks finiteness (see [`Self::f64_round_trips_exactly`]'s
    /// neighbourhood), so an out-of-range exponent such as `1e400` never
    /// reaches a `JsonSource` at all. This is the JSONC route's version of
    /// the same finiteness guard [`crate::sources::SourceTerm::validate`]
    /// makes for the OpenFOAM `constant/fvSources` route - caught one step
    /// earlier here, not skipped.
    #[test]
    fn an_out_of_range_body_force_component_is_refused_on_read() {
        let err = match minimal_case_with_sources(
            r#"{ "type": "momentumSource", "field": "U", "bodyForce": [1e400, 0.0, 0.0] }"#,
        ) {
            Err(e) => e.to_string(),
            Ok(_) => panic!("expected read_case_jsonc to refuse an out-of-range bodyForce component"),
        };
        assert!(err.contains("finite"), "{err}");
    }
}
