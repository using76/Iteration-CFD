// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.

//! The multi-region conduction case - SPEC-LIT §46 and §47.4, the case format
//! side.
//!
//! Provenance: ORIGINAL. This is meteor-cfd's own case format, in the JSONC
//! style `crate::io::case_json` established (`docs/05-io-redesign.md`); the
//! physics it names is SPEC-LIT §46's solid energy equation and §47's
//! conjugate interface, and the numbers a user types are the ones the
//! literature cited in those sections defines. Nothing was transcribed from
//! another code's case format. No GPL-licensed source was consulted.
//!
//! # What this format is for, and what it is not
//!
//! A stack of **solid** regions coupled through conformal interfaces, with
//! contact resistances, anisotropic conductivities and volumetric heat
//! sources: a package, a board, a wall assembly. That is the whole of
//! SPEC-LIT §46 plus §47's interface, and it is what
//! [`crate::cht::ConjugateHeat`] solves.
//!
//! **SPEC-LIT §60 added the fluid region this used to refuse.** A region may
//! now say `"kind": "fluid"`, and it then carries a `fluid` block (four
//! constant properties) instead of a `material` one, the case carries a
//! `buoyancy` block and a `numerics.flow` block, and the whole thing is
//! solved by `crate::cht::flow::run_flow_case` - §26's energy equation over
//! §47.4's concatenated mesh (§59) beside §5's SIMPLE loop on the fluid
//! block alone.
//!
//! **SPEC-LIT §79 opened it.** A fluid patch may now say
//! `"kind": "inlet"` (and carry `U`) or `"kind": "outlet"` (and carry
//! `inletOutlet` on `T`), which is the entry §60.2 recorded as not existing -
//! and the reason §60.6's Gate 6, Qu & Mudawar's forced-convection
//! micro-channel, was UNREACHABLE rather than refused. Exactly one of each, or
//! neither; **neither is §60.2's closed cavity, unchanged in every bit**, and
//! a case with no opening still writes a no-slip wall on every non-`empty`
//! fluid patch.
//!
//! One more thing moved with it: `buoyancy` is REQUIRED by a closed cavity,
//! which has nothing else that could drive it, and OPTIONAL once a case names
//! an inlet. Absent, the fluid has constant density `fluid.rho` and no body
//! force - Qu & Mudawar's own assumptions (4) and (6), and the right model for
//! a liquid, which SPEC-LIT §25's `rho = p0/(R_s T)` is not.
//!
//! §47.9's `coupledTemperature` is still refused as a patch entry, because on
//! this format an interface is declared by the `interfaces` block, and a patch
//! that named the condition without an interface behind it would be a setting
//! the case can say and the solver ignores.
//!
//! # The rule that shapes it
//!
//! **Every patch of every region must be named exactly once**, by a `patches`
//! rule or by an `interfaces` entry. Not defaulted, not inferred. An unnamed
//! patch is an error listing the patches that were named and the ones that
//! were not, because "adiabatic unless you say otherwise" is precisely how a
//! case comes to say something the solver ignores.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::blockgen::{self, BlockSpec, GradedAxis};
use crate::cht::flow::{
    Buoyancy, FlowCase, FlowControls, FlowRegion, FluidMaterial, Openings,
};
use crate::cht::{
    Conductivity, InterfaceRequest, PairingTolerances, RegionKind, SolidMaterial,
};
use crate::fv::DivScheme;
use crate::error::{Error, Result};
use crate::field::BcKind;
use crate::io::case::{LinearSolverKind, Preconditioner, SolverControls};
use crate::io::case_json::{JsonBounds, JsonGrading, JsonGradingAxis, JsonOutput};
use crate::io::output_plan::{OutputFormat, OutputPlan};
use crate::io::polymesh::{build_host_mesh, read_poly_mesh, PolyMeshRaw};
use crate::mesh::{HostMesh, PatchKind};
use crate::solid::{BondTreatment, Material, NotBuilt};
use crate::{Label, Scalar, Vec3};

// ==========================================================================
//  1. The document
// ==========================================================================

/// A multi-region conduction case.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ChtCase {
    #[serde(rename = "$schema", default, skip_serializing_if = "Option::is_none")]
    pub schema: Option<String>,
    pub name: String,
    /// At least one. Region order fixes the concatenated cell numbering
    /// (SPEC-LIT §47.4), so it is the case's own decision and not this
    /// reader's.
    pub regions: Vec<ChtRegion>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub interfaces: Vec<ChtInterface>,
    /// SPEC-LIT §9's face body force. **Required by a fluid region and
    /// refused without one** - a body force on a stack of solids is a setting
    /// the solver would ignore, which is the §13.4.1 defect.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub buoyancy: Option<ChtBuoyancy>,
    pub initial: ChtInitial,
    pub run: ChtRun,
    #[serde(default)]
    pub numerics: ChtNumerics,
    /// SPEC-LIT §44.1's `output` block, unchanged - on this format it
    /// selects the per-region VTU write (§96.2). Resolved through
    /// `OutputPlan::from_json` at lowering, with §96.3 row 17's refusals
    /// applied; refused whole on a case with a fluid region (row 18).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output: Option<JsonOutput>,
}

/// SPEC-LIT §9's `constant/g` and `TRef`, as a conjugate case states them.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ChtBuoyancy {
    /// Gravitational acceleration, m/s^2.
    pub g: [f64; 3],
    /// The reference temperature of `b = g(TRef/T - 1)`, K.
    #[serde(rename = "TRef")]
    pub t_ref: f64,
}

/// One region: a block of cells, one material, its own boundary conditions.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields, rename_all = "camelCase")]
pub struct ChtRegion {
    pub name: String,
    /// `"solid"` or `"fluid"` - SPEC-LIT §60.2. Anything else is a §13.4
    /// error listing both. A fluid region must be the FIRST region and there
    /// can be at most one (§47.4's numbering invariant).
    #[serde(default = "solid_kind")]
    pub kind: String,
    pub mesh: ChtRegionMesh,
    /// A **solid** region's material - SPEC-LIT §46.5. Required on a solid
    /// region and refused on a fluid one, which carries [`Self::fluid`]
    /// instead.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub material: Option<ChtMaterial>,
    /// A **fluid** region's four constant properties - SPEC-LIT §60.2.
    /// Required on a fluid region and refused on a solid one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fluid: Option<ChtFluid>,
    /// SPEC-LIT §96.1's `mechanics` block: `Some` on a solid region whose
    /// §95 displacement is solved after the thermal one. Refused on a fluid
    /// region (§96.3 row 8), and legal only with `"mode": "stress"` on
    /// `run` - in both directions (row 14).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mechanics: Option<ChtMechanics>,
    /// Uniform volumetric heat source `q'''`, W/m^3 - SPEC-LIT (S46.1). The
    /// die's own dissipation, in the case this format exists for.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source: Option<f64>,
    /// One rule per patch. Every patch must appear here or in an
    /// `interfaces` entry; see the module doc.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub patches: Vec<ChtPatchRule>,
}

fn solid_kind() -> String {
    "solid".to_string()
}

/// One region's mesh: an axis-aligned block this reader builds, or a
/// polyMesh (or a single-volume `.msh`) read from disk - SPEC-LIT §97.1.
///
/// Untagged, with the block form FIRST, so every document written before §97
/// deserialises exactly as it always did. Each form is a newtype over its own
/// `deny_unknown_fields` struct - not a struct variant - so a mistyped key is
/// refused under either form, and a document carrying both `polyMesh` and
/// `cells` matches neither variant and is a parse error.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(untagged)]
pub enum ChtRegionMesh {
    Block(ChtBlockMesh),
    PolyMesh(ChtPolyMeshRef),
}

/// The block form - what [`ChtRegionMesh`] alone was before §97, field for
/// field unchanged. An axis-aligned block; patch names are the case's, one
/// per face.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ChtBlockMesh {
    pub bounds: JsonBounds,
    pub cells: [u32; 3],
    /// The six face names, `-x +x -y +y -z +z`.
    pub boundaries: ChtBoundaries,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub grading: Option<JsonGrading>,
}

/// The imported form - SPEC-LIT §97.1. `polyMesh` is a path RELATIVE TO THE
/// CASE FILE'S DIRECTORY: a polyMesh directory, or a case root / `constant`
/// holding one - the three probes [`read_poly_mesh`] makes - or a `.msh`
/// file with ONE volume entity.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ChtPolyMeshRef {
    #[serde(rename = "polyMesh")]
    pub poly_mesh: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ChtBoundaries {
    pub xmin: String,
    pub xmax: String,
    pub ymin: String,
    pub ymax: String,
    pub zmin: String,
    pub zmax: String,
}

impl ChtBoundaries {
    fn names(&self) -> [&str; 6] {
        [
            &self.xmin, &self.xmax, &self.ymin, &self.ymax, &self.zmin, &self.zmax,
        ]
    }
}

/// SPEC-LIT §46.5's three entries.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ChtMaterial {
    /// `rho_s`, kg/m^3.
    pub rho: f64,
    /// `c_s`, J/(kg K).
    pub c: f64,
    /// `k_s`: one number for an isotropic material, three for `diag(kx,ky,kz)`
    /// in the MESH axes. Nine is a §13.4 error naming the two that are
    /// implemented - SPEC-LIT §46.4.
    pub kappa: ChtKappa,
}

/// A fluid region's constant properties - SPEC-LIT §60.2.
///
/// Four numbers, and `Pr = mu cp/kappa` is DERIVED from them and printed
/// rather than stated: a case that stated both could contradict itself, and
/// the reader would have to pick a winner.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ChtFluid {
    /// `rho_f` at `buoyancy.TRef`, kg/m^3.
    pub rho: f64,
    /// `c_p`, J/(kg K).
    pub cp: f64,
    /// `k_f`, W/(m K). A **scalar**: an anisotropic fluid conductivity is not
    /// a thing, and three or nine components are a §13.4 error.
    pub kappa: f64,
    /// Dynamic viscosity, Pa s.
    pub mu: f64,
}

/// `kappa` written either way. A user with an isotropic material should not
/// have to type a one-element list, and one with an anisotropic material
/// should not have to type three separate entries.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(untagged)]
pub enum ChtKappa {
    Isotropic(f64),
    Components(Vec<f64>),
}

impl ChtKappa {
    fn values(&self) -> Vec<Scalar> {
        match self {
            Self::Isotropic(k) => vec![*k as Scalar],
            Self::Components(v) => v.iter().map(|x| *x as Scalar).collect(),
        }
    }
}

/// One patch's temperature condition, and - on a fluid region - what the patch
/// IS.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields, rename_all = "camelCase")]
pub struct ChtPatchRule {
    /// The patch name, as `mesh.boundaries` spells it. Exact, not a pattern:
    /// a conduction stack has a handful of named faces and a pattern would
    /// only make it possible to match none of them by accident.
    #[serde(rename = "match")]
    pub match_: String,
    /// `"wall"` (the default), `"inlet"` or `"outlet"` - SPEC-LIT §79.2.
    ///
    /// This is the entry §60.2 said did not exist. Until §79 every non-`empty`
    /// patch of a fluid region was a no-slip wall and the document had no
    /// spelling for anything else, which is why §60.6's Gate 6 was
    /// UNREACHABLE rather than refused. An `inlet` carries [`Self::u`] and a
    /// `fixedValue` `T`; an `outlet` carries neither and takes `inletOutlet`
    /// or `zeroGradient`. Only a FLUID region may say either.
    #[serde(default = "wall_patch_kind")]
    pub kind: String,
    /// The inlet velocity vector, m/s - required on an `inlet` and refused
    /// anywhere else. Uniform over the patch: SPEC-LIT §79.4's
    /// flux-establishment pass takes one normal speed, and a profile is
    /// refused by name rather than averaged.
    #[serde(rename = "U", default, skip_serializing_if = "Option::is_none")]
    pub u: Option<[f64; 3]>,
    #[serde(rename = "T")]
    pub t: ChtScalarBc,
}

fn wall_patch_kind() -> String {
    "wall".to_string()
}

/// What a patch can say about `T` - SPEC-LIT §4's triple, reached by name.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "type", deny_unknown_fields)]
pub enum ChtScalarBc {
    /// A held temperature, K.
    #[serde(rename = "fixedValue")]
    FixedValue { value: f64 },
    /// Adiabatic.
    #[serde(rename = "zeroGradient")]
    ZeroGradient,
    /// A prescribed wall heat flux, W/m^2, positive INTO the solid -
    /// SPEC-LIT §32.2.
    #[serde(rename = "fixedFluxTemperature")]
    FixedFluxTemperature { q: f64 },
    /// SPEC-LIT §79.5's outflow condition: `zeroGradient` while the flux
    /// leaves, `fixedValue inletValue` on any face where it comes back in.
    ///
    /// Legal ONLY on an `outlet` patch. On a wall the flux is identically
    /// zero, so the switch would never fire and the condition would be
    /// `zeroGradient` wearing another name - a setting the case can say and
    /// the solver ignores, which is the §13.4.1 defect.
    #[serde(rename = "inletOutlet")]
    InletOutlet {
        #[serde(rename = "inletValue")]
        inlet_value: f64,
    },
    /// The 2-D front/back plane: the patch contributes to no surface integral
    /// at all.
    ///
    /// Spelled as a `T` condition rather than as a mesh flag because the
    /// format's one rule is that **every patch is named exactly once** (module
    /// doc), and a mesh flag that silently claimed two patches would be
    /// exactly the "adiabatic unless you say otherwise" default that rule
    /// exists to stop. `empty` patches come in opposite pairs and the axis
    /// they lie on must have one cell; `blockgen` refuses anything else,
    /// naming the axis.
    #[serde(rename = "empty")]
    Empty,
}

/// One conformal interface between two regions - SPEC-LIT §47.4/§47.5.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields, rename_all = "camelCase")]
pub struct ChtInterface {
    pub region_a: String,
    pub patch_a: String,
    pub region_b: String,
    pub patch_b: String,
    /// The contact resistance, m^2 K/W. Absent is perfect contact.
    ///
    /// Mutually exclusive with `thicknessLayers`/`kappaLayers`: a case that
    /// writes both has said the same number twice and this reader has no
    /// business deciding which one it meant.
    #[serde(rename = "Rc", default, skip_serializing_if = "Option::is_none")]
    pub rc: Option<f64>,
    /// OpenFOAM's spelling of the same thing, summed by SPEC-LIT (S47.11):
    /// `R_c = sum_i t_i/k_i`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub thickness_layers: Option<Vec<f64>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub kappa_layers: Option<Vec<f64>>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ChtInitial {
    /// Uniform initial temperature, K. Per-region initial fields are not
    /// implemented; a case that needs one should say so and be told, rather
    /// than have this reader guess.
    #[serde(rename = "T")]
    pub t: f64,
}

/// Steady, or a fixed-step transient.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields, rename_all = "camelCase")]
pub struct ChtRun {
    /// `true`: drop `dT/dt` and solve the steady problem - SPEC-LIT §46.1's
    /// quasi-steady solid, which is a control flag and not a second code
    /// path.
    #[serde(default)]
    pub steady: bool,
    /// `"thermal"` (the default - §46's conduction, what this format has
    /// always solved) or `"stress"` (§96.1's thermo-elastic run: after the
    /// thermal solve converges, every solid region carrying a `mechanics`
    /// block has §95's displacement solved on it). Anything else is refused
    /// listing both (§96.3 row 19).
    #[serde(default = "thermal_mode")]
    pub mode: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub end_time: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub delta_t: Option<f64>,
    /// Outer SIMPLE iterations. **Required by a fluid region and refused
    /// without one**; meaningless on a pure-conduction case, where the
    /// coupled system is solved in one pass (SPEC-LIT §47.3).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub iterations: Option<u32>,
}

fn thermal_mode() -> String {
    "thermal".to_string()
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields, rename_all = "camelCase")]
pub struct ChtNumerics {
    /// `PCG` (the default - a pure conduction matrix is symmetric, including
    /// its coupled interface entries, SPEC-LIT §47.2) or `PBiCGStab`.
    pub solver: String,
    /// `DIC`, `DILU` or `diagonal`.
    pub preconditioner: String,
    pub tolerance: f64,
    pub max_iter: u32,
    /// Non-orthogonal corrector passes - SPEC-LIT §2.4. Zero on an
    /// orthogonal mesh, where the correction is identically zero.
    #[serde(default)]
    pub n_non_orthogonal_correctors: u32,
    /// The outer loop's own settings. **Required by a fluid region and
    /// refused without one**, for the same §13.4.1 reason as `buoyancy`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub flow: Option<ChtFlow>,
}

/// SPEC-LIT §60.1's `numerics.flow` block - the SIMPLE loop's own settings.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields, rename_all = "camelCase")]
pub struct ChtFlow {
    /// Momentum's implicit under-relaxation, SPEC-LIT §5.2.
    pub relax_u: f64,
    /// The pressure FIELD's explicit relaxation. Patankar §6.7 pairs it with
    /// `relaxU` as `alpha_p ~ 1 - alpha_U`; SIMPLEC permits `1.0`.
    pub relax_p: f64,
    /// `T`'s implicit under-relaxation.
    pub relax_t: f64,
    /// SIMPLEC (SPEC-LIT §5.3) rather than plain SIMPLE.
    #[serde(default)]
    pub simplec: bool,
    /// `div(phi,U)`, a `divSchemes` entry - SPEC-LIT §11.
    #[serde(default = "linear_scheme")]
    pub div_scheme_u: String,
    /// `div(phi,T)`.
    #[serde(default = "linear_scheme")]
    pub div_scheme_t: String,
    /// Stop when all three initial residuals are below this. `0` runs the
    /// full `run.iterations`.
    #[serde(default)]
    pub residual: f64,
    #[serde(default = "default_flow_tolerance")]
    pub u_tolerance: f64,
    #[serde(default = "default_flow_tolerance")]
    pub p_tolerance: f64,
    #[serde(default = "default_flow_max_iter")]
    pub u_max_iter: u32,
    #[serde(default = "default_flow_max_iter")]
    pub p_max_iter: u32,
}

fn linear_scheme() -> String {
    "Gauss linear".to_string()
}

fn default_flow_tolerance() -> f64 {
    1e-14
}

fn default_flow_max_iter() -> u32 {
    1000
}

impl Default for ChtNumerics {
    fn default() -> Self {
        Self {
            solver: "PCG".to_string(),
            preconditioner: "DIC".to_string(),
            tolerance: 1e-12,
            max_iter: 2000,
            n_non_orthogonal_correctors: 0,
            flow: None,
        }
    }
}

// ==========================================================================
//  1b. The mechanics block - SPEC-LIT §96
// ==========================================================================

/// SPEC-LIT §96.1's `mechanics` block: the elastic half of a thermo-elastic
/// region. `Some` on a solid region whose §95 displacement is to be solved
/// after the thermal one; refused on a fluid one (§96.3 row 8), and legal
/// only with `"mode": "stress"` on `run`, in both directions (row 14).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ChtMechanics {
    /// One elastic material for the whole region - or [`Self::materials`]'s
    /// zone list instead, and never both (§96.3 row 9).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub material: Option<ChtElastic>,
    /// docs/10's R4 zone list: a bonded two-material solid (a bimetal strip)
    /// is ONE region with one entry per material, not two regions.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub materials: Option<Vec<ChtElasticZone>>,
    /// How the zones meet at their shared faces: `"series"` (the default) or
    /// `"linear"`. Only legal with [`Self::materials`] - one material has no
    /// bond face (§96.3 row 13).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bond: Option<String>,
    /// Every non-empty patch of the region's mesh, exactly once - interface
    /// patches included, the module doc's rule carried over whole.
    pub patches: Vec<ChtMechanicalPatchRule>,
    /// The outer loop's own settings. Both knobs optional, both defaulted.
    #[serde(default)]
    pub solver: ChtSolidSolver,
}

/// The elastic constants of one material or one zone - SPEC-LIT §96.1.
///
/// Validated at lowering by [`crate::solid::Material::validate`], which is
/// where `E <= 0`, the `nu` range and the measured 0.45 edge are refused
/// (§96.3 rows 1-2); this struct carries no logic of its own.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ChtElastic {
    /// Young's modulus `E`, Pa.
    #[serde(rename = "E")]
    pub e: f64,
    /// Poisson's ratio, in (-1, 0.5) and at most the measured edge 0.45.
    pub nu: f64,
    /// Linear thermal expansion coefficient `alpha`, 1/K.
    pub alpha: f64,
    /// The stress-free temperature, K: the thermal strain is
    /// `alpha (T - TRef)`, so `alpha > 0` without it is refused (§96.3
    /// row 3), and it without `alpha` is a reference nothing reads (row 4).
    #[serde(rename = "TRef", default, skip_serializing_if = "Option::is_none")]
    pub t_ref: Option<f64>,
    /// ALWAYS refused (§96.3 row 5): nothing in §95's static solve reads a
    /// density. Present as a field, rather than left to
    /// `deny_unknown_fields`, so the message can say WHY - the same idiom
    /// `JsonExact::precision` uses.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rho: Option<f64>,
}

/// One bonded material of a `materials` zone list - docs/10's R4. The zone
/// is the CLOSED box test on the cell CENTROID, and every cell must land in
/// exactly one zone (§96.3 rows 10-11).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ChtElasticZone {
    /// The zone's name, as the refusal and the banner print it.
    pub name: String,
    /// The closed box the zone's cells' centroids must fall in.
    pub bounds: JsonBounds,
    /// The zone's elastic constants.
    pub material: ChtElastic,
}

/// One `mechanics.patches` rule - the [`ChtPatchRule`] shape carried over:
/// the exact patch name, one condition.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ChtMechanicalPatchRule {
    /// The patch name, as `mesh.boundaries` spells it. An interface patch
    /// is named here too - the displacement problem does not read the
    /// thermal `interfaces` block.
    #[serde(rename = "match")]
    pub match_: String,
    /// The mechanical condition on the face.
    pub u: ChtMechanicalBc,
}

/// What a patch can say about `u` - §95's per-component statement, reached
/// by name and lowered into [`LoweredMechanicalBc`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "type", deny_unknown_fields)]
pub enum ChtMechanicalBc {
    /// `u_i = value` per component; `null` leaves that component
    /// traction-free.
    #[serde(rename = "fixedDisplacement")]
    FixedDisplacement {
        /// Per axis; `null` is traction-free in that component.
        value: [Option<f64>; 3],
    },
    /// `(sigma . n)_i = value`, Pa, global axes. `traction [0,0,0]` is a
    /// free surface.
    #[serde(rename = "traction")]
    Traction {
        /// The traction vector, Pa.
        value: [f64; 3],
    },
    /// The normal component fixed 0, the tangential traction 0. The axis
    /// is the patch's slot in `-x +x -y +y -z +z`, divided by two.
    #[serde(rename = "symmetry")]
    Symmetry,
    /// Traction `(0, 0, 0)`.
    #[serde(rename = "free")]
    Free,
}

/// The outer displacement loop's own settings - SPEC-LIT §96.1. Two knobs,
/// both optional and defaulted; the two entries that are NOT offered
/// (`relaxation`, `ddtScheme`) are present as fields so their refusals can
/// say why, exactly as `JsonExact::precision` does.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields, rename_all = "camelCase")]
pub struct ChtSolidSolver {
    /// The outer loop's stop: the residual decades it must fall. Default
    /// `1e-6`.
    #[serde(default = "default_solid_tolerance")]
    pub tolerance: f64,
    /// The outer loop's iteration cap. Default `500`.
    #[serde(default = "default_solid_max_outer")]
    pub max_outer: u32,
    /// ALWAYS refused (§96.3 row 7): the outer loop is Aitken delta-squared
    /// on the increment and its first omega is 1 - a static relaxation
    /// factor is not offered.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub relaxation: Option<f64>,
    /// ALWAYS refused (§96.3 row 6) through §95's inertia refusal:
    /// Newmark, HHT and generalised-alpha are not built.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ddt_scheme: Option<String>,
}

impl Default for ChtSolidSolver {
    fn default() -> Self {
        Self {
            tolerance: default_solid_tolerance(),
            max_outer: default_solid_max_outer(),
            relaxation: None,
            ddt_scheme: None,
        }
    }
}

fn default_solid_tolerance() -> f64 {
    1e-6
}

fn default_solid_max_outer() -> u32 {
    500
}

// ==========================================================================
//  2. Reading
// ==========================================================================

pub fn read_cht_case(path: &Path) -> Result<ChtCase> {
    crate::io::case_json::parse_jsonc_file(path)
}

pub fn parse_cht_case(text: &str, what: &str) -> Result<ChtCase> {
    crate::io::case_json::parse_jsonc_str(text, what)
}

// ==========================================================================
//  3. Lowering
// ==========================================================================

/// A patch's condition, resolved onto SPEC-LIT §4's triple.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum LoweredBc {
    /// `fr = 1`, `refValue = value`.
    FixedValue(Scalar),
    /// `fr = 0`, `refGrad = 0`.
    ZeroGradient,
    /// `fr = 0`, and `refGrad` is written from `q` and the face's own
    /// conductance - SPEC-LIT §32.2.
    FixedFlux(Scalar),
    /// SPEC-LIT §79.5: `refValue = inletValue`, `refGrad = 0`, and `fr`
    /// rewritten from the sign of the face flux every outer iteration by
    /// `field_ops::update_inlet_outlet`.
    InletOutlet(Scalar),
}

impl LoweredBc {
    pub fn kind(self) -> BcKind {
        match self {
            Self::FixedValue(_) => BcKind::FixedValue,
            Self::ZeroGradient => BcKind::ZeroGradient,
            Self::FixedFlux(_) => BcKind::FixedFluxTemperature,
            Self::InletOutlet(_) => BcKind::InletOutlet,
        }
    }
}

/// SPEC-LIT §96.2: one elastic zone of a region's `mechanics` block,
/// validated and in [`crate::solid::Material`]'s own units.
#[derive(Debug, Clone, PartialEq)]
pub struct LoweredElasticZone {
    /// The zone's name, as the case spelled it.
    pub name: String,
    /// The zone's validated constants.
    pub material: crate::solid::Material,
    /// The zone's `TRef`, K - present because §96.3 row 3 required it
    /// whenever `alpha > 0`.
    pub t_ref: Scalar,
}

/// A mechanical patch condition, resolved onto §95's per-component
/// statement.
#[derive(Debug, Clone, PartialEq)]
pub enum LoweredMechanicalBc {
    /// `u_i = value` per component; `None` is traction-free in that
    /// component.
    Fixed([Option<Scalar>; 3]),
    /// `(sigma . n) = value`, Pa, global axes.
    Traction(Vec3),
    /// The normal component fixed 0, the tangential traction 0, on `axis`.
    Symmetry {
        /// The axis the plane is normal to: the patch's slot in
        /// `-x +x -y +y -z +z`, divided by two.
        axis: usize,
    },
    /// Traction `(0, 0, 0)`.
    Free,
}

/// `mechanics.solver`, resolved - SPEC-LIT §96.2.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SolidOuterControls {
    /// The outer loop's stop.
    pub tolerance: Scalar,
    /// The outer loop's iteration cap.
    pub max_outer: usize,
}

/// A region's `mechanics` block, everything resolved - what
/// `crate::solid::case` (SPEC-LIT §96.2) reads.
#[derive(Debug, Clone, PartialEq)]
pub struct LoweredMechanics {
    /// The region's index.
    pub region: usize,
    /// One zone at least - the whole region, when the case named one
    /// `material`.
    pub zones: Vec<LoweredElasticZone>,
    /// `[meshes[region].n_cells]`: the zone of every cell.
    pub zone_of_cell: Vec<usize>,
    /// How zones meet: `series` (the default) or `linear`.
    pub bond: crate::solid::BondTreatment,
    /// `(patch name, condition)`, every non-empty patch exactly once.
    pub patch_bcs: Vec<(String, LoweredMechanicalBc)>,
    /// The outer loop's settings.
    pub solver: SolidOuterControls,
}

/// Everything [`crate::cht`] needs, with every name already resolved.
#[derive(Debug)]
pub struct LoweredChtCase {
    pub name: String,
    pub region_names: Vec<String>,
    pub kinds: Vec<RegionKind>,
    pub meshes: Vec<HostMesh>,
    /// One raw polyMesh per region, in region order - bitwise what
    /// `blockgen::raw_mesh` built the matching `meshes[r]` from. `HostMesh`
    /// keeps no point set and no face polygons (SPEC-LIT §49.3), so the raw
    /// geometry travels with the lowered case instead.
    pub raw: Vec<PolyMeshRaw>,
    /// `[n_regions]`, `Some` exactly on a solid region carrying §96.1's
    /// `mechanics` block. Everything resolved and refused at lowering; read
    /// by `crate::solid::case` (§96.2).
    pub mechanics: Vec<Option<LoweredMechanics>>,
    /// `run.mode == "stress"`: §96.2's thermo-elastic run. Refused without a
    /// region carrying `mechanics`, and refused WITH one in thermal mode.
    pub stress: bool,
    /// The case's `output` block, resolved through
    /// [`crate::io::output_plan::OutputPlan::from_json`] with §96.3 row 17's
    /// refusals applied. `None` when the case says nothing.
    pub output: Option<crate::io::output_plan::OutputPlan>,
    /// `[n_regions]` the conduction entry for every region. A **fluid**
    /// region's entry is a placeholder built from its own `rho`/`cp`/`kappa`
    /// - SPEC-LIT (S59.3) masks every coefficient it produces on a fluid face
    /// away, because a fluid face carries the LIVE `k_eff`.
    pub materials: Vec<SolidMaterial>,
    /// `[n_regions]`, `Some` exactly on the fluid region - SPEC-LIT §60.2.
    pub fluids: Vec<Option<FluidMaterial>>,
    /// SPEC-LIT §9's body force. `Some` exactly when there is a fluid region.
    pub buoyancy: Option<Buoyancy>,
    /// The outer loop's settings. `Some` exactly when there is a fluid region.
    pub flow: Option<FlowControls>,
    /// SPEC-LIT §79.2's inlet/outlet pair, on the fluid region. `None` is
    /// §60.2's closed cavity, unchanged.
    pub openings: Option<Openings>,
    /// `[n_regions]` uniform volumetric source, W/m^3.
    pub sources: Vec<Scalar>,
    pub interfaces: Vec<InterfaceRequest>,
    /// `(region, patch name, condition)`, one per patch that is not an
    /// interface.
    pub patch_bcs: Vec<(usize, String, LoweredBc)>,
    pub initial_t: Scalar,
    pub steady: bool,
    pub end_time: Scalar,
    pub delta_t: Scalar,
    pub solver: SolverControls,
    pub n_non_orthogonal_correctors: usize,
    pub tolerances: PairingTolerances,
}

impl LoweredChtCase {
    /// The region kinds, in order.
    pub fn kinds(&self) -> Vec<RegionKind> {
        self.kinds.clone()
    }

    /// Does this case carry a fluid region - i.e. is it
    /// `crate::cht::flow::run_flow_case`'s business rather than
    /// `crate::cht::run_case`'s?
    pub fn has_fluid(&self) -> bool {
        self.kinds.iter().any(|k| *k == RegionKind::Fluid)
    }

    /// The conjugate fluid/solid case, borrowed out of this one - SPEC-LIT
    /// §60. `None` on a pure-conduction case.
    pub fn flow_case(&self) -> Option<FlowCase<'_>> {
        let flow = self.flow.clone()?;
        Some(FlowCase {
            name: self.name.clone(),
            regions: self
                .region_names
                .iter()
                .zip(&self.kinds)
                .zip(&self.materials)
                .zip(&self.fluids)
                .zip(&self.sources)
                .map(|((((name, kind), mat), fl), src)| FlowRegion {
                    name: name.clone(),
                    kind: *kind,
                    solid: (*kind == RegionKind::Solid).then(|| mat.clone()),
                    fluid: fl.clone(),
                    source: *src,
                })
                .collect(),
            meshes: &self.meshes,
            interfaces: self.interfaces.clone(),
            patch_bcs: self.patch_bcs.clone(),
            buoyancy: self.buoyancy,
            openings: self.openings.clone(),
            initial_t: self.initial_t,
            flow,
            t_solver: self.solver,
            n_non_orthogonal_correctors: self.n_non_orthogonal_correctors,
            tolerances: self.tolerances,
            p0: AMBIENT_PRESSURE,
        })
    }
}

/// The ambient pressure SPEC-LIT §25's gas state is pinned at, Pa.
///
/// Not a case entry. `p0` and the molar mass are the SAME one-parameter
/// family in `rho = p0/(R_s T)` (see `FluidMaterial::gas_properties`), so a
/// case that could set both could contradict its own `fluid.rho` - and the
/// density is the number a reader checks. One standard atmosphere.
pub const AMBIENT_PRESSURE: Scalar = 101_325.0;

impl ChtCase {
    /// [`Self::lower_in`]`(None)`: every all-block case ever written needs no
    /// disk, and this is what it lowers through. A document with a `polyMesh`
    /// region is refused here BY NAME - the path is relative to the case
    /// file's directory, and with no directory there is nothing to resolve it
    /// against (SPEC-LIT §97.2).
    pub fn lower(&self) -> Result<LoweredChtCase> {
        self.lower_in(None)
    }

    /// Resolve every name, build or READ every region mesh, and refuse
    /// everything §13.4 and §97.2 say must be refused. `case_dir` is the
    /// directory holding the case file (`ofgpu-cht` passes
    /// `case_path.parent()`); `None` is [`Self::lower`], block regions only.
    pub fn lower_in(&self, case_dir: Option<&Path>) -> Result<LoweredChtCase> {
        if self.regions.is_empty() {
            return Err(Error::Config(format!(
                "{}: a conduction case needs at least one region",
                self.name
            )));
        }

        let mut index: BTreeMap<&str, usize> = BTreeMap::new();
        for (i, r) in self.regions.iter().enumerate() {
            if index.insert(r.name.as_str(), i).is_some() {
                return Err(Error::Config(format!(
                    "regions: '{}' is declared twice; region names are how \
                     `interfaces` refers to them and must be unique",
                    r.name
                )));
            }
        }

        let mut region_names = Vec::new();
        let mut kinds = Vec::new();
        let mut meshes = Vec::new();
        let mut raws = Vec::new();
        let mut materials = Vec::new();
        let mut fluids: Vec<Option<FluidMaterial>> = Vec::new();
        let mut sources = Vec::new();
        // Which patches of which region have been spoken for, and by what.
        let mut claimed: Vec<BTreeMap<String, &'static str>> = Vec::new();
        // §97.2: each region's own patch names, as the BUILT mesh spells
        // them - the two listing refusals below read these, because an
        // imported region's names are the mesh's, not the document's.
        let mut all_patch_names: Vec<Vec<String>> = Vec::new();
        // SPEC-LIT §96.2: one lowered `mechanics` per region, `None` when
        // the region says nothing.
        let mut mechanics: Vec<Option<LoweredMechanics>> = Vec::new();

        for (i, r) in self.regions.iter().enumerate() {
            let kind = match r.kind.as_str() {
                "solid" => RegionKind::Solid,
                "fluid" => RegionKind::Fluid,
                other => {
                    // `-permissive` cannot substitute here, and the code
                    // says so rather than reaching an `unreachable!()`: a
                    // region is solid or fluid, and there is no third
                    // thing to run instead of the one the case named.
                    crate::io::contract::unsupported(
                        &format!("regions/{}/kind", r.name),
                        other,
                        &["solid", "fluid"],
                        "solid (SPEC-LIT 46's conducting region) or fluid \
                         (SPEC-LIT 60.2's closed buoyant cavity). There is no \
                         third kind",
                        (),
                    )?;
                    return Err(Error::Config(format!(
                        "regions/{}/kind: \"{other}\" is not solid or fluid, \
                         and cannot be substituted even under -permissive - \
                         there is no third thing to run instead",
                        r.name
                    )));
                }
            };

            // SPEC-LIT 47.4's numbering invariant, checked here so the message
            // names the case's own region rather than a cell index.
            if kind == RegionKind::Fluid && i != 0 {
                return Err(Error::Config(format!(
                    "regions/{}: the fluid region is region {i}; it must be the \
                     FIRST region so the fluid block keeps its own cell and \
                     boundary-face numbering in the concatenated thermal mesh \
                     (SPEC-LIT 47.4)",
                    r.name
                )));
            }

            // Which patches are `empty` has to be known BEFORE the mesh is
            // built or read, because it is the mesh's patch TYPE and not a
            // condition written onto it afterwards.
            let empties: Vec<&str> = r
                .patches
                .iter()
                .filter(|p| matches!(p.t, ChtScalarBc::Empty))
                .map(|p| p.match_.as_str())
                .collect();
            // SPEC-LIT §79.2: an opening is the mesh's patch TYPE, exactly as
            // `empty` is, so it has to be known before the block is built. A
            // solid region cannot carry one - it has no flow - and the message
            // names the region rather than a face index.
            let mut flow_patches: Vec<&str> = Vec::new();
            for rule in &r.patches {
                match rule.kind.as_str() {
                    "wall" => {}
                    "inlet" | "outlet" => {
                        if kind != RegionKind::Fluid {
                            return Err(Error::Config(format!(
                                "regions/{}/patches: patch '{}' is an `{}`, but region '{}' is \
                                 SOLID. An opening is a place flow enters or leaves and a \
                                 conducting solid has none (SPEC-LIT 79.2); put it on the fluid \
                                 region",
                                r.name, rule.match_, rule.kind, r.name
                            )));
                        }
                        flow_patches.push(rule.match_.as_str());
                    }
                    other => {
                        // The fallback NAME is deliberately not the name of a
                        // patch kind, because there is no substitution to
                        // make: `-permissive` reaches the refusal below
                        // whatever it says here, so a `fallback_name` reading
                        // "wall" would promise, in the non-permissive
                        // message, a substitution that never happens. This is
                        // §60.3's own idiom for `regions/kind`, with the
                        // promise taken back out of it.
                        crate::io::contract::unsupported(
                            &format!("regions/{}/patches/{}/kind", r.name, rule.match_),
                            other,
                            &["wall", "inlet", "outlet"],
                            "NOTHING - the patch TYPE decides whether the pressure equation owns \
                             the face's flux, so there is no third answer to guess and this is \
                             refused under -permissive too",
                            (),
                        )?;
                        return Err(Error::Config(format!(
                            "regions/{}/patches/{}/kind: \"{other}\" is not wall, inlet or outlet, \
                             and cannot be substituted even under -permissive - the patch TYPE \
                             decides whether the pressure equation owns the face's flux, and there \
                             is no third answer to guess",
                            r.name, rule.match_
                        )));
                    }
                }
            }
            let (mesh, rmesh) =
                build_region_mesh(r, kind, &empties, &flow_patches, case_dir)?;

            // SPEC-LIT §97.2: for an IMPORTED region the patch list lives on
            // disk, not in the document - so the `seen` set, and every
            // refusal below that lists a region's patches, reads the BUILT
            // mesh's patch names. For a block the two lists are the same six
            // names in the same order, so no existing message moves.
            let patch_names: Vec<String> =
                mesh.patches.iter().map(|p| p.name.clone()).collect();
            let mut seen: BTreeMap<String, &'static str> = BTreeMap::new();
            for n in &patch_names {
                seen.entry(n.clone()).or_insert("unnamed");
            }

            // SPEC-LIT §96.2: the region's `mechanics` block, if it says
            // one, lowered against the mesh just built - it is the mesh's
            // own centroids and patch slots the zones and the symmetry axes
            // are measured on.
            let mech = lower_mechanics(i, r, &mesh, &empties, &patch_names)?;

            let (mat, fluid) = match (kind, &r.material, &r.fluid) {
                (RegionKind::Solid, Some(m), None) => {
                    let mat = SolidMaterial {
                        name: r.name.clone(),
                        rho: m.rho as Scalar,
                        c: m.c as Scalar,
                        k: Conductivity::parse(
                            &m.kappa.values(),
                            &format!("regions/{}/material/kappa", r.name),
                        )?,
                    };
                    mat.validate()?;
                    (mat, None)
                }
                (RegionKind::Fluid, None, Some(f)) => {
                    let fl = FluidMaterial {
                        name: r.name.clone(),
                        rho: f.rho as Scalar,
                        cp: f.cp as Scalar,
                        kappa: f.kappa as Scalar,
                        mu: f.mu as Scalar,
                    };
                    fl.validate()?;
                    // The conduction entry a fluid region still needs; every
                    // coefficient it produces on a fluid face is masked away
                    // by SPEC-LIT (S59.3).
                    let mat = SolidMaterial {
                        name: r.name.clone(),
                        rho: fl.rho,
                        c: fl.cp,
                        k: Conductivity::Isotropic(fl.kappa),
                    };
                    (mat, Some(fl))
                }
                (RegionKind::Solid, None, _) => {
                    return Err(Error::Config(format!(
                        "regions/{}: a solid region needs a `material` block \
                         (rho, c, kappa) - SPEC-LIT 46.5",
                        r.name
                    )))
                }
                (RegionKind::Solid, Some(_), Some(_)) => {
                    return Err(Error::Config(format!(
                        "regions/{}: a SOLID region carries `material`, not \
                         `fluid`. The two name different physics (SPEC-LIT 46.5 \
                         against 60.2) and this reader will not choose between \
                         them",
                        r.name
                    )))
                }
                (RegionKind::Fluid, _, None) => {
                    return Err(Error::Config(format!(
                        "regions/{}: a fluid region needs a `fluid` block (rho, \
                         cp, kappa, mu) - SPEC-LIT 60.2",
                        r.name
                    )))
                }
                (RegionKind::Fluid, Some(_), Some(_)) => {
                    return Err(Error::Config(format!(
                        "regions/{}: a FLUID region carries `fluid`, not \
                         `material`. The two name different physics (SPEC-LIT \
                         60.2 against 46.5) and this reader will not choose \
                         between them",
                        r.name
                    )))
                }
            };

            // SPEC-LIT 60.3 refused a source on a fluid region because
            // SPEC-LIT 18's registry was not wired to this format's fluid
            // side, and a source that is read and dropped is the 13.4.1
            // defect. It is wired now: the number lowers into `sources`
            // exactly as a solid region's does, and `run_flow_case` registers
            // every region's onto its own cells of the thermal mesh in
            // `EnergySources` - the same registry, and the same energy
            // balance, a solid region's source reaches. Nothing is refused
            // here any more, and nothing is dropped either.

            region_names.push(r.name.clone());
            kinds.push(kind);
            meshes.push(mesh);
            raws.push(rmesh);
            materials.push(mat);
            fluids.push(fluid);
            sources.push(r.source.unwrap_or(0.0) as Scalar);
            claimed.push(seen);
            all_patch_names.push(patch_names);
            mechanics.push(mech);
        }

        if kinds.iter().filter(|k| **k == RegionKind::Fluid).count() > 1 {
            return Err(Error::Config(
                "regions: more than one fluid region. Multiple fluid regions \
                 coupled through a solid are not implemented (SPEC-LIT 47.4); \
                 mesh them as one fluid region, or couple them through a solid \
                 whose two faces are separate interfaces"
                    .to_string(),
            ));
        }
        let has_fluid = kinds.iter().any(|k| *k == RegionKind::Fluid);

        // ---- interfaces --------------------------------------------------
        let mut interfaces = Vec::new();
        for (i, f) in self.interfaces.iter().enumerate() {
            let ra = *index.get(f.region_a.as_str()).ok_or_else(|| {
                Error::Config(format!(
                    "interfaces[{i}]: no region '{}'. The case declares: {}",
                    f.region_a,
                    region_names.join(", ")
                ))
            })?;
            let rb = *index.get(f.region_b.as_str()).ok_or_else(|| {
                Error::Config(format!(
                    "interfaces[{i}]: no region '{}'. The case declares: {}",
                    f.region_b,
                    region_names.join(", ")
                ))
            })?;

            for (r, patch) in [(ra, &f.patch_a), (rb, &f.patch_b)] {
                match claimed[r].get_mut(patch.as_str()) {
                    None => {
                        return Err(Error::Config(format!(
                            "interfaces[{i}]: region '{}' has no patch '{patch}'. It \
                             has: {}",
                            region_names[r],
                            all_patch_names[r].join(", ")
                        )))
                    }
                    Some(slot) if *slot != "unnamed" => {
                        return Err(Error::Config(format!(
                            "interfaces[{i}]: patch '{patch}' of region '{}' is already \
                             claimed by a {slot}. A patch carries ONE condition \
                             (SPEC-LIT 47.6), so an interface face cannot also have a \
                             `patches` rule",
                            region_names[r]
                        )))
                    }
                    Some(slot) => *slot = "interface",
                }
            }

            let r_c = match (&f.rc, &f.thickness_layers, &f.kappa_layers) {
                (Some(_), Some(_), _) | (Some(_), _, Some(_)) => {
                    return Err(Error::Config(format!(
                        "interfaces[{i}]: both `Rc` and `thicknessLayers`/`kappaLayers` \
                         were given. They are two spellings of the same number \
                         (SPEC-LIT S47.11) and this reader will not choose between them"
                    )))
                }
                (Some(v), None, None) => *v as Scalar,
                (None, Some(t), Some(k)) => crate::cht::layered_resistance(
                    &t.iter().map(|v| *v as Scalar).collect::<Vec<Scalar>>(),
                    &k.iter().map(|v| *v as Scalar).collect::<Vec<Scalar>>(),
                )?,
                (None, Some(_), None) | (None, None, Some(_)) => {
                    return Err(Error::Config(format!(
                        "interfaces[{i}]: `thicknessLayers` and `kappaLayers` name the \
                         same layers and must both be given"
                    )))
                }
                (None, None, None) => 0.0,
            };
            if !(r_c >= 0.0) {
                return Err(Error::Config(format!(
                    "interfaces[{i}]: Rc = {r_c} is negative; a contact resistance \
                     cannot create heat"
                )));
            }

            interfaces.push(InterfaceRequest::new(ra, &f.patch_a, rb, &f.patch_b, r_c));
        }

        // ---- patch rules -------------------------------------------------
        let mut patch_bcs = Vec::new();
        #[allow(clippy::type_complexity)]
        let mut inlets: Vec<(String, String, Option<crate::Vec3>)> = Vec::new();
        let mut outlets: Vec<(String, String)> = Vec::new();
        for (r, region) in self.regions.iter().enumerate() {
            for rule in &region.patches {
                match claimed[r].get_mut(rule.match_.as_str()) {
                    None => {
                        return Err(Error::Config(format!(
                            "regions/{}/patches: no patch '{}'. The region has: {}",
                            region.name,
                            rule.match_,
                            all_patch_names[r].join(", ")
                        )))
                    }
                    Some(slot) if *slot != "unnamed" => {
                        return Err(Error::Config(format!(
                            "regions/{}/patches: patch '{}' is named twice (the second \
                             time after a {slot}). A patch carries ONE condition",
                            region.name, rule.match_
                        )))
                    }
                    Some(slot) => *slot = "patches rule",
                }
                let path = format!("regions/{}/patches/{}", region.name, rule.match_);
                let is_inlet = rule.kind == "inlet";
                let is_outlet = rule.kind == "outlet";

                // SPEC-LIT §79.2, in both directions. A `U` that no opening
                // reads and an inlet with no `U` are the same defect seen from
                // the two sides, and each is refused by name.
                if rule.u.is_some() && !is_inlet {
                    return Err(Error::Config(format!(
                        "{path}/U: a velocity on a `{}` patch is a setting the solver would ignore \
                         (SPEC-LIT 13.4.1) - only an `inlet` prescribes one. A wall is no-slip and \
                         an outlet's velocity is what the pressure equation computes",
                        rule.kind
                    )));
                }
                let opening_u = if is_inlet {
                    let u = rule.u.ok_or_else(|| {
                        Error::Config(format!(
                            "{path}: an `inlet` needs `U` - the velocity, m/s, uniform over the \
                             patch. SPEC-LIT 79.4's flux-establishment pass is built from its \
                             normal component and there is no default this reader is entitled to \
                             invent"
                        ))
                    })?;
                    let v = crate::Vec3::new(u[0] as Scalar, u[1] as Scalar, u[2] as Scalar);
                    if !v.x.is_finite() || !v.y.is_finite() || !v.z.is_finite() {
                        return Err(Error::Config(format!("{path}/U is not finite")));
                    }
                    if !(v.mag_sqr() > 0.0) {
                        return Err(Error::Config(format!(
                            "{path}/U is zero. An inlet through which nothing enters is a no-slip \
                             wall spelled at length; say `\"kind\": \"wall\"` and mean it (SPEC-LIT \
                             79.2)"
                        )));
                    }
                    Some(v)
                } else {
                    None
                };

                // SPEC-LIT §79.5 and §79.10: which `T` each patch kind may carry. Every
                // combination not listed is refused by name, because the ones
                // that are missing are the ones that would be read and then
                // mean nothing.
                match (&rule.t, is_inlet, is_outlet) {
                    (ChtScalarBc::Empty, false, false) => {}
                    (ChtScalarBc::Empty, _, _) => {
                        return Err(Error::Config(format!(
                            "{path}: an `empty` patch contributes to no surface integral at all, so \
                             it cannot also be an opening (SPEC-LIT 79.2)"
                        )))
                    }
                    (ChtScalarBc::InletOutlet { .. }, _, true) => {}
                    (ChtScalarBc::InletOutlet { .. }, _, false) => {
                        return Err(Error::Config(format!(
                            "{path}/T: `inletOutlet` switches on the SIGN of the face flux, and the \
                             flux through a wall or an inlet never changes sign - so on anything \
                             but an `outlet` it is `zeroGradient` (or `fixedValue`) wearing another \
                             name, which is the setting the solver ignores that SPEC-LIT 13.4.1 \
                             exists to stop"
                        )))
                    }
                    (ChtScalarBc::FixedValue { .. }, true, _) => {}
                    (_, true, _) => {
                        return Err(Error::Config(format!(
                            "{path}/T: an `inlet` carries `fixedValue` - the temperature of what \
                             enters. Any other condition leaves the entering enthalpy undetermined \
                             (SPEC-LIT 79.5)"
                        )))
                    }
                    (ChtScalarBc::ZeroGradient, _, true) => {}
                    (_, _, true) => {
                        return Err(Error::Config(format!(
                            "{path}/T: an `outlet` carries `inletOutlet` or `zeroGradient`. A held \
                             temperature at an outlet conducts heat back INTO a domain the flow is \
                             leaving, and a prescribed flux there is a wall condition on a face \
                             that is not a wall (SPEC-LIT 79.5)"
                        )))
                    }
                    _ => {}
                }

                if is_inlet {
                    inlets.push((region.name.clone(), rule.match_.clone(), opening_u));
                }
                if is_outlet {
                    outlets.push((region.name.clone(), rule.match_.clone()));
                }

                patch_bcs.push((r, rule.match_.clone(), lower_bc(&rule.t)));
            }
        }

        // Every patch, named exactly once. An unnamed patch is an error and
        // not a default - see the module doc.
        let mut unnamed = Vec::new();
        for (r, region) in self.regions.iter().enumerate() {
            for (name, by) in &claimed[r] {
                if *by == "unnamed" {
                    unnamed.push(format!("{}:{name}", region.name));
                }
            }
        }
        if !unnamed.is_empty() {
            return Err(Error::Config(format!(
                "these patches carry no condition: {}. Every patch must be named \
                 exactly once, by a `patches` rule or by an `interfaces` entry - a \
                 silent adiabatic default is how a case comes to say something the \
                 solver ignores (SPEC-LIT 13.4)",
                unnamed.join(", ")
            )));
        }

        // ---- SPEC-LIT §79.2's openings -----------------------------------
        //
        // Exactly one of each, or neither. Two inlets would need a pressure
        // level each to decide how the inflow splits, and SPEC-LIT §79.4's
        // flux-establishment pass carries the single Dirichlet reference the
        // outlet supplies - so rather than pick a split silently, the pair is
        // required to be a pair.
        let openings = match (inlets.len(), outlets.len()) {
            (0, 0) => None,
            (1, 1) => {
                let (_, inlet_patch, u) = inlets.remove(0);
                let (_, outlet_patch) = outlets.remove(0);
                if inlet_patch == outlet_patch {
                    return Err(Error::Config(
                        "regions/patches: the inlet and the outlet are the same patch. A patch \
                         carries ONE condition"
                            .to_string(),
                    ));
                }
                Some(Openings {
                    inlet_patch,
                    inlet_velocity: u.unwrap_or_default(),
                    outlet_patch,
                })
            }
            (n_in, n_out) => {
                return Err(Error::Config(format!(
                    "regions/patches: {n_in} inlet(s) and {n_out} outlet(s). SPEC-LIT 79.2 takes \
                     exactly one of each, or neither (the closed cavity of SPEC-LIT 60.2). An inlet \
                     with no outlet drives mass into a domain with no path out of it; a second \
                     opening needs a pressure level of its own to decide how the flow splits, and \
                     SPEC-LIT 79.4's flux-establishment solve carries exactly one Dirichlet \
                     reference"
                )));
            }
        };
        if openings.is_some() && !has_fluid {
            return Err(Error::Config(
                "regions/patches: an opening was named but no region has `\"kind\": \"fluid\"`"
                    .to_string(),
            ));
        }

        // ---- the fluid-only blocks, and SPEC-LIT 60.3's refusals ---------
        //
        // Each of these is a setting the case could write and the solver would
        // ignore, which is the 13.4.1 defect six instances of have been found
        // in this project. Refused in BOTH directions: present without a fluid
        // region, and absent with one.
        //
        // SPEC-LIT §79.6 loosened exactly one row of this: `buoyancy` is
        // required by a CLOSED cavity, which has no other thing to drive it,
        // and OPTIONAL once the case names an inlet, because forced convection
        // is driven by the inlet and Qu & Mudawar's own assumption (6) is that
        // buoyancy is negligible. Absent, the fluid has CONSTANT density -
        // their assumption (4) - and no body force.
        let forced = openings.is_some();
        for (what, present) in [
            ("numerics/flow", self.numerics.flow.is_some()),
            ("run/iterations", self.run.iterations.is_some()),
        ] {
            if present && !has_fluid {
                return Err(Error::Config(format!(
                    "{what} was given but no region has `\"kind\": \"fluid\"`. \
                     Nothing in a stack of conducting solids reads it, and a \
                     setting the solver ignores is exactly what SPEC-LIT 13.4.1 \
                     exists to stop - delete it, or make a region fluid"
                )));
            }
            if !present && has_fluid {
                return Err(Error::Config(format!(
                    "a fluid region needs `{what}`. SPEC-LIT 60.2: a closed \
                     cavity is driven by SPEC-LIT 9's body force and solved by \
                     an outer SIMPLE loop, and neither has a default this \
                     reader is entitled to invent"
                )));
            }
        }

        if self.buoyancy.is_some() && !has_fluid {
            return Err(Error::Config(
                "buoyancy was given but no region has `\"kind\": \"fluid\"`. Nothing in a stack of \
                 conducting solids reads it, and a setting the solver ignores is exactly what \
                 SPEC-LIT 13.4.1 exists to stop - delete it, or make a region fluid"
                    .to_string(),
            ));
        }
        if self.buoyancy.is_none() && has_fluid && !forced {
            return Err(Error::Config(
                "a CLOSED fluid cavity needs `buoyancy`. SPEC-LIT 60.2: every non-`empty` patch of \
                 it is a no-slip wall, so SPEC-LIT 9's body force is the only thing that can drive \
                 any flow at all and there is no default this reader is entitled to invent. A case \
                 that meant conduction should say `kind: solid`; a case that meant FORCED \
                 convection should name an `inlet` and an `outlet` (SPEC-LIT 79.2), and may then \
                 omit `buoyancy` - which makes the fluid's density constant at `fluid.rho`"
                    .to_string(),
            ));
        }

        let buoyancy = match &self.buoyancy {
            Some(b) => {
                let b = Buoyancy {
                    g: crate::Vec3::new(b.g[0] as Scalar, b.g[1] as Scalar, b.g[2] as Scalar),
                    t_ref: b.t_ref as Scalar,
                };
                b.validate()?;
                Some(b)
            }
            None => None,
        };

        let flow = match (&self.numerics.flow, self.run.iterations) {
            (Some(f), Some(n)) => {
                let c = FlowControls {
                    iterations: n as usize,
                    residual: f.residual as Scalar,
                    relax_u: f.relax_u as Scalar,
                    relax_p: f.relax_p as Scalar,
                    relax_t: f.relax_t as Scalar,
                    div_u: lower_div("numerics/flow/divSchemeU", &f.div_scheme_u)?,
                    div_t: crate::io::schemes::DivEntry {
                        scheme: lower_div("numerics/flow/divSchemeT", &f.div_scheme_t)?,
                        bounded: true,
                    },
                    u_solver: SolverControls {
                        solver: LinearSolverKind::PBiCGStab,
                        precon: Preconditioner::Dilu,
                        tolerance: f.u_tolerance as Scalar,
                        rel_tol: 0.01,
                        max_iter: f.u_max_iter as Label,
                        check_interval: 10,
                        ..SolverControls::default()
                    },
                    p_solver: SolverControls {
                        solver: LinearSolverKind::PCG,
                        precon: Preconditioner::Dic,
                        tolerance: f.p_tolerance as Scalar,
                        rel_tol: 0.001,
                        max_iter: f.p_max_iter as Label,
                        check_interval: 10,
                        ..SolverControls::default()
                    },
                    n_non_orth_correctors: self.numerics.n_non_orthogonal_correctors as usize,
                    simplec: f.simplec,
                };
                c.validate()?;
                Some(c)
            }
            _ => None,
        };

        // ---- run and numerics --------------------------------------------
        let (end_time, delta_t) = if self.run.steady {
            if self.run.end_time.is_some() || self.run.delta_t.is_some() {
                return Err(Error::Config(
                    "run: `steady` was set and so was `endTime`/`deltaT`. A steady \
                     solve has no time to end at (SPEC-LIT 46.1's quasi-steady \
                     solid); say one or the other"
                        .to_string(),
                ));
            }
            (0.0, 0.0)
        } else {
            let end = self.run.end_time.ok_or_else(|| {
                Error::Config("run: a transient case needs `endTime`".to_string())
            })?;
            let dt = self.run.delta_t.ok_or_else(|| {
                Error::Config("run: a transient case needs `deltaT`".to_string())
            })?;
            if !(dt > 0.0) || !(end > 0.0) {
                return Err(Error::Config(format!(
                    "run: endTime = {end} and deltaT = {dt} must both be positive"
                )));
            }
            (end as Scalar, dt as Scalar)
        };
        if has_fluid && !self.run.steady {
            return Err(Error::Config(
                "run: a case with a fluid region must be `steady`. SPEC-LIT \
                 59.6: the (rho c) ratio across a fluid/solid interface is \
                 O(1e3) and nothing in this tree gates the time accuracy of a \
                 conjugate FLUID transient - SPEC-LIT 47.12's Gate 3 gates the \
                 solid/solid one and stops there. It is refused rather than run \
                 and believed"
                    .to_string(),
            ));
        }

        let solver = SolverControls {
            solver: lower_solver(&self.numerics.solver)?,
            precon: lower_precon(&self.numerics.preconditioner)?,
            tolerance: self.numerics.tolerance as Scalar,
            rel_tol: 0.0,
            max_iter: self.numerics.max_iter as Label,
            ..SolverControls::default()
        };

        // ---- SPEC-LIT §96.3 rows 14-16, 18-19: the run mode --------------
        let stress = match self.run.mode.as_str() {
            "thermal" => false,
            "stress" => true,
            other => {
                crate::io::contract::unsupported(
                    "run/mode",
                    other,
                    &["thermal", "stress"],
                    "NOTHING - the mode is which physics runs, not a spelling, \
                     and this is refused under -permissive too",
                    (),
                )?;
                return Err(Error::Config(format!(
                    "run/mode: \"{other}\" is neither `thermal` nor `stress`, \
                     and cannot be substituted even under -permissive - there \
                     is no third mode to run instead"
                )));
            }
        };
        if stress && mechanics.iter().all(|m| m.is_none()) {
            return Err(Error::Config(
                "run/mode: \"stress\" names SPEC-LIT §95's displacement solve, \
                 but no region carries a `mechanics` block - nothing would read \
                 it. Add `mechanics` to a solid region, or drop the mode (the \
                 default `thermal` is what this format has always solved)"
                    .to_string(),
            ));
        }
        if !stress {
            for r in &self.regions {
                if r.mechanics.is_some() {
                    return Err(Error::Config(format!(
                        "regions/{}/mechanics: a `mechanics` block needs \
                         `run.mode` \"stress\" - with the default \"thermal\" \
                         nothing would read it, which is the setting the solver \
                         ignores that SPEC-LIT 13.4.1 exists to stop",
                        r.name
                    )));
                }
            }
        }
        if stress && has_fluid {
            return Err(Error::Config(
                "run/mode: \"stress\" on a case with a fluid region - the fluid \
                 side of a thermo-elastic run is docs/10's address 106 (WF-B), \
                 not this format's. Drop the mode, or the fluid region"
                    .to_string(),
            ));
        }
        if stress && !self.run.steady {
            return Err(Error::Config(
                "run/mode: \"stress\" on a transient case - the SPEC-LIT 93 \
                 verdict is a steady residual, and time in a stress run is \
                 docs/10's address 103. Set run.steady to true"
                    .to_string(),
            ));
        }
        if self.output.is_some() && has_fluid {
            return Err(Error::Config(
                "output: a case with a fluid region cannot carry an `output` \
                 block yet - the flow path's VTU is a follow-up, not in this \
                 unit. Delete the block, or make every region solid"
                    .to_string(),
            ));
        }

        // ---- the output block, resolved last (SPEC-LIT §96.3 row 17) -----
        let output = match &self.output {
            Some(o) => {
                let mut plan = OutputPlan::from_json(o)?;
                // A multi-region mesh is not one Cartesian lattice, so there
                // is no voxel grid to sample onto.
                plan.refuse_visualisation_on_a_non_cartesian_mesh(false)?;
                plan.refuse_restart("ofgpu-cht", "run the case again")?;
                let foam = plan
                    .exact
                    .as_ref()
                    .is_some_and(|e| e.formats.contains(&OutputFormat::Foam));
                if foam {
                    return Err(Error::Config(
                        "output/exact/format: one polyMesh per region is \
                         docs/10's address 97 layout, S11 - this run writes \
                         `vtu` only. Say \"format\": \"vtu\""
                            .to_string(),
                    ));
                }
                let interval = plan.exact.as_ref().map_or(0.0, |e| e.interval);
                if interval > 0.0 {
                    if self.run.steady {
                        plan.refuse_interval_when_steady(
                            "ofgpu-cht",
                            "set run.steady to false and give endTime/deltaT; \
                             the thermal transient then writes its final state \
                             only, because ofgpu-cht's run_case returns one \
                             state (SPEC-LIT 47.14)",
                        )?;
                    } else {
                        return Err(Error::Config(
                            "output/exact/interval: a positive interval names a \
                             schedule this run has no clock for - ofgpu-cht's \
                             run_case returns one state, the final one. Drop the \
                             interval (SPEC-LIT 44.4)"
                                .to_string(),
                        ));
                    }
                }
                Some(plan)
            }
            None => None,
        };

        Ok(LoweredChtCase {
            name: self.name.clone(),
            region_names,
            kinds,
            meshes,
            raw: raws,
            mechanics,
            stress,
            output,
            materials,
            fluids,
            buoyancy,
            flow,
            openings,
            sources,
            interfaces,
            patch_bcs,
            initial_t: self.initial.t as Scalar,
            steady: self.run.steady,
            end_time,
            delta_t,
            solver,
            n_non_orthogonal_correctors: self.numerics.n_non_orthogonal_correctors as usize,
            tolerances: PairingTolerances::default(),
        })
    }
}

/// SPEC-LIT §96.2: one region's `mechanics` block, lowered against the mesh
/// `build_region_mesh` just built - every patch name resolved, every zone's
/// cells found, and §96.3's rows 1-13 raised here.
fn lower_mechanics(
    i: usize,
    r: &ChtRegion,
    mesh: &HostMesh,
    empties: &[&str],
    patch_names: &[String],
) -> Result<Option<LoweredMechanics>> {
    let Some(mech) = &r.mechanics else {
        return Ok(None);
    };
    let path = format!("regions/{}/mechanics", r.name);

    // Row 8: a fluid region has no displacement to solve.
    if r.kind != "solid" {
        return Err(Error::Config(format!(
            "{path}: region '{}' is \"fluid\" - a fluid carries `fluid`, not \
             `material`, and has no §95 displacement to solve. `mechanics` \
             belongs on a solid region",
            r.name
        )));
    }

    // Row 9: exactly one spelling of the material.
    if mech.material.is_some() && mech.materials.is_some() {
        return Err(Error::Config(format!(
            "{path}: EXACTLY ONE of `material` and `materials` must be given - \
             both were. They are two spellings of one answer and this reader \
             will not choose between them (SPEC-LIT 96.3 row 9)"
        )));
    }
    if mech.material.is_none() && mech.materials.as_ref().map_or(true, Vec::is_empty)
    {
        return Err(Error::Config(format!(
            "{path}: neither `material` nor `materials` was given - an empty \
             `materials` list is neither (SPEC-LIT 96.3 row 9). Name one \
             `material`, or the zones of a bonded solid"
        )));
    }

    // The zones, validated under their own paths (§96.3 rows 1-5). A single
    // `material` is one zone named after the region.
    let mut zones = Vec::new();
    if let Some(m) = &mech.material {
        let (material, t_ref) = lower_elastic(m, &format!("{path}/material"))?;
        zones.push(LoweredElasticZone {
            name: r.name.clone(),
            material,
            t_ref,
        });
    } else if let Some(list) = &mech.materials {
        for z in list {
            let (material, t_ref) =
                lower_elastic(&z.material, &format!("{path}/materials/{}", z.name))?;
            zones.push(LoweredElasticZone {
                name: z.name.clone(),
                material,
                t_ref,
            });
        }
    }

    // The zone of every cell: the closed box test on the CENTROID, exactly
    // as §96.1 states it. A single material tiles by construction.
    let zone_of_cell = if mech.material.is_some() {
        vec![0usize; mesh.n_cells]
    } else {
        let list: &[ChtElasticZone] = mech.materials.as_deref().unwrap_or(&[]);
        let mut of = vec![usize::MAX; mesh.n_cells];
        for (zi, z) in list.iter().enumerate() {
            for (ci, ctr) in mesh.c.iter().enumerate() {
                let ctrs = [ctr.x, ctr.y, ctr.z];
                let in_box = (0..3).all(|a| {
                    z.bounds.min[a] as Scalar <= ctrs[a] && ctrs[a] <= z.bounds.max[a] as Scalar
                });
                if in_box {
                    if of[ci] != usize::MAX {
                        return Err(Error::Config(format!(
                            "{path}/materials: zones '{}' and '{}' both claim the \
                             cell at ({:.6}, {:.6}, {:.6}) m - a cell is in \
                             exactly one zone (SPEC-LIT 96.3 row 11)",
                            list[of[ci]].name,
                            z.name,
                            ctr.x,
                            ctr.y,
                            ctr.z
                        )));
                    }
                    of[ci] = zi;
                }
            }
        }
        let uncovered: Vec<usize> =
            (0..mesh.n_cells).filter(|ci| of[*ci] == usize::MAX).collect();
        if let Some(first) = uncovered.first() {
            let ctr = &mesh.c[*first];
            return Err(Error::Config(format!(
                "{path}/materials: {} cells are in no zone - the first is the \
                 cell whose centroid is ({:.6}, {:.6}, {:.6}) m. The zones' \
                 bounds must tile the region, closed boxes on the cell \
                 centroids (SPEC-LIT 96.3 row 10)",
                uncovered.len(),
                ctr.x,
                ctr.y,
                ctr.z
            )));
        }
        of
    };

    // Row 13: the bond treatment, only where there is a bond face.
    let bond = match &mech.bond {
        None => BondTreatment::Series,
        Some(word) => {
            if zones.len() < 2 {
                return Err(Error::Config(format!(
                    "{path}/bond: \"{word}\" was given but the region names one \
                     `material` - one material has no bond face. `bond` is only \
                     legal with `materials` (SPEC-LIT 96.3 row 13)"
                )));
            }
            match word.as_str() {
                "series" => BondTreatment::Series,
                "linear" => BondTreatment::Linear,
                other => {
                    crate::io::contract::unsupported(
                        &format!("{path}/bond"),
                        other,
                        &["series", "linear"],
                        "NOTHING - how two zones share a face is physics, not a \
                         spelling, and this is refused under -permissive too",
                        (),
                    )?;
                    return Err(Error::Config(format!(
                        "{path}/bond: \"{other}\" is neither `series` nor \
                         `linear`, and cannot be substituted even under \
                         -permissive"
                    )));
                }
            }
        }
    };

    // Row 12: every non-empty patch, exactly once - the module doc's rule
    // carried over to the displacement statement. The names are the BUILT
    // mesh's (§97.2): for an imported region they are the boundary file's.
    let mut named: Vec<&str> = Vec::new();
    let mut patch_bcs = Vec::new();
    for rule in &mech.patches {
        let name = rule.match_.as_str();
        let slot = patch_names.iter().position(|n| n == name).ok_or_else(|| {
            Error::Config(format!(
                "{path}/patches: no patch '{name}'. The region has: {}",
                patch_names.join(", ")
            ))
        })?;
        if empties.contains(&name) {
            return Err(Error::Config(format!(
                "{path}/patches: patch '{name}' is `empty` - it contributes to \
                 no surface integral at all and carries no mechanical condition. \
                 Name only the region's non-empty patches"
            )));
        }
        if named.contains(&name) {
            return Err(Error::Config(format!(
                "{path}/patches: patch '{name}' is named twice. A patch carries \
                 ONE condition"
            )));
        }
        named.push(name);
        let bc = match &rule.u {
            ChtMechanicalBc::FixedDisplacement { value } => LoweredMechanicalBc::Fixed([
                value[0].map(|v| v as Scalar),
                value[1].map(|v| v as Scalar),
                value[2].map(|v| v as Scalar),
            ]),
            ChtMechanicalBc::Traction { value } => LoweredMechanicalBc::Traction(Vec3::new(
                value[0] as Scalar,
                value[1] as Scalar,
                value[2] as Scalar,
            )),
            // The axis is the slot in `-x +x -y +y -z +z`, divided by two.
            ChtMechanicalBc::Symmetry => LoweredMechanicalBc::Symmetry { axis: slot / 2 },
            ChtMechanicalBc::Free => LoweredMechanicalBc::Free,
        };
        patch_bcs.push((rule.match_.clone(), bc));
    }
    let unnamed: Vec<&str> = patch_names
        .iter()
        .map(|n| n.as_str())
        .filter(|n| !empties.contains(n) && !named.contains(n))
        .collect();
    if !unnamed.is_empty() {
        return Err(Error::Config(format!(
            "{path}/patches: these patches carry no mechanical condition: {}. \
             Every non-empty patch of the region must be named exactly once - \
             the thermal `patches` and `interfaces` blocks do not reach the \
             displacement problem (SPEC-LIT 96.3 row 12)",
            unnamed.join(", ")
        )));
    }

    // Rows 6-7: the two `solver` entries that are not offered, present as
    // fields precisely so these refusals can say why.
    if mech.solver.ddt_scheme.is_some() {
        return Err(crate::solid::refuse(
            NotBuilt::Inertia,
            &format!("{path}/solver/ddtScheme"),
        ));
    }
    if mech.solver.relaxation.is_some() {
        return Err(Error::Config(format!(
            "{path}/solver/relaxation: a static relaxation factor is not \
             offered - the outer loop is Aitken delta-squared on the increment \
             and its first omega is 1 (SPEC-LIT §95). Delete `relaxation`"
        )));
    }
    let solver = SolidOuterControls {
        tolerance: mech.solver.tolerance as Scalar,
        max_outer: mech.solver.max_outer as usize,
    };

    Ok(Some(LoweredMechanics {
        region: i,
        zones,
        zone_of_cell,
        bond,
        patch_bcs,
        solver,
    }))
}

/// One elastic material, validated under its JSON path - SPEC-LIT §96.3
/// rows 1-5. [`crate::solid::Material::validate`] refuses `E <= 0`, the `nu`
/// range and the measured 0.45 edge with its own messages; this wraps it
/// with the path and adds what the case format knows that the struct does
/// not: `alpha < 0`, the `TRef` pairing, and `rho`.
fn lower_elastic(m: &ChtElastic, path: &str) -> Result<(Material, Scalar)> {
    if let Some(rho) = m.rho {
        return Err(Error::Config(format!(
            "{path}/rho = {rho}: nothing in SPEC-LIT 95's static solve reads a \
             density - no inertia, no self-weight; the dynamic solid is \
             docs/10's address 106b. Delete it (the thermal `material.rho` is \
             the region's density)"
        )));
    }
    let mat = Material {
        e: m.e as Scalar,
        nu: m.nu as Scalar,
        alpha: m.alpha as Scalar,
    };
    mat.validate()
        .map_err(|e| Error::Config(format!("{path}: {e}")))?;
    if mat.alpha < 0.0 {
        return Err(Error::Config(format!(
            "{path}/alpha = {:e} is negative - a negative expansion coefficient \
             is a sign error, not a material",
            mat.alpha
        )));
    }
    let t_ref = match m.t_ref {
        Some(t) => {
            if mat.alpha == 0.0 {
                return Err(Error::Config(format!(
                    "{path}/TRef = {t} is given with alpha = 0 - a reference \
                     nothing reads, which is the setting the solver ignores \
                     (SPEC-LIT 13.4.1). Delete TRef, or give the alpha it scales"
                )));
            }
            t as Scalar
        }
        None => {
            if mat.alpha > 0.0 {
                return Err(Error::Config(format!(
                    "{path}/TRef: alpha = {} is given but TRef is not - the \
                     thermal strain is alpha (T - TRef); no TRef, no strain. \
                     Give TRef, the stress-free temperature",
                    mat.alpha
                )));
            }
            0.0
        }
    };
    Ok((mat, t_ref))
}

fn lower_bc(bc: &ChtScalarBc) -> LoweredBc {
    match bc {
        ChtScalarBc::FixedValue { value } => LoweredBc::FixedValue(*value as Scalar),
        ChtScalarBc::ZeroGradient => LoweredBc::ZeroGradient,
        ChtScalarBc::FixedFluxTemperature { q } => LoweredBc::FixedFlux(*q as Scalar),
        ChtScalarBc::InletOutlet { inlet_value } => {
            LoweredBc::InletOutlet(*inlet_value as Scalar)
        }
        // An `empty` patch contributes to no surface integral, so the triple
        // written on it is never read. `run_flow_case` skips those faces by
        // the mesh's own `PatchKind`, which is where the fact lives.
        ChtScalarBc::Empty => LoweredBc::ZeroGradient,
    }
}

/// A `divSchemes` entry, through the same reader every other case uses -
/// SPEC-LIT §11.7, so the menu in the refusal is the crate's one menu.
fn lower_div(setting: &str, text: &str) -> Result<DivScheme> {
    crate::io::schemes::parse_div(setting, text).map(|e| e.scheme)
}

fn lower_solver(name: &str) -> Result<LinearSolverKind> {
    match name {
        "PCG" => Ok(LinearSolverKind::PCG),
        "PBiCGStab" => Ok(LinearSolverKind::PBiCGStab),
        other => crate::io::contract::unsupported(
            "numerics/solver",
            other,
            &["PCG", "PBiCGStab"],
            "PCG - a pure conduction matrix is symmetric, INCLUDING its coupled \
             interface entries (SPEC-LIT 47.2), and `solver::matrix_is_symmetric` \
             now checks both halves (SPEC-LIT 48.3)",
            LinearSolverKind::PCG,
        ),
    }
}

fn lower_precon(name: &str) -> Result<Preconditioner> {
    match name {
        "DIC" => Ok(Preconditioner::Dic),
        "DILU" => Ok(Preconditioner::Dilu),
        "diagonal" | "none" => Ok(Preconditioner::Diagonal),
        other => crate::io::contract::unsupported(
            "numerics/preconditioner",
            other,
            &["DIC", "DILU", "diagonal"],
            "DIC, the incomplete Cholesky factorisation (SPEC-LIT 21)",
            Preconditioner::Dic,
        ),
    }
}

/// Build or read one region's mesh - SPEC-LIT §97.1.
///
/// Block: through `blockgen` exactly as before (`raw_mesh` +
/// `build_host_mesh`). PolyMesh: `resolve_mesh_path` ->
/// `read_poly_mesh` (or `read_msh_with_volumes` when the path ends in
/// `.msh`, case-insensitively) -> `check_imported_patches` ->
/// `build_host_mesh`, the SAME constructor a block region reaches, which is
/// what makes Gate 97-A a bit-for-bit statement rather than a tolerance.
///
/// `empties` and `openings` name the patches that are the mesh's patch TYPE
/// rather than a condition written on afterwards (SPEC-LIT §60.2's 2-D front
/// and back, §79.2's inlet and outlet): an `empty` face contributes to no
/// surface integral at all, which is a property of the topology and not a
/// boundary condition. On the block form a solid region's other faces stay
/// plain `patch`, and a **fluid** region's become `wall` - no-slip walls in
/// the momentum sense (§60.2); on the IMPORTED form the mesh's own types
/// stand, and [`check_imported_patches`] refuses the combinations the case
/// and the mesh can disagree on.
fn build_region_mesh(
    r: &ChtRegion,
    kind: RegionKind,
    empties: &[&str],
    openings: &[&str],
    case_dir: Option<&Path>,
) -> Result<(HostMesh, PolyMeshRaw)> {
    let b = match &r.mesh {
        ChtRegionMesh::Block(b) => b,
        ChtRegionMesh::PolyMesh(pr) => {
            let path = resolve_mesh_path(&r.name, case_dir, &pr.poly_mesh)?;
            let is_msh = path
                .extension()
                .and_then(|e| e.to_str())
                .is_some_and(|e| e.eq_ignore_ascii_case("msh"));
            let raw = if is_msh {
                let (raw, n_vol) = crate::io::msh::read_msh_with_volumes(&path)?;
                if n_vol > 1 {
                    return Err(Error::Config(format!(
                        "regions/{}/mesh/polyMesh: '{}' is a `.msh` with {n_vol} \
                         volume entities. A `.msh` with {n_vol} volumes is several \
                         regions in one file, and this reader keeps one region per \
                         mesh - the volume tags are discarded by `io::msh`, so the \
                         faces the volumes share would come out internal with no \
                         interface patch between them. Write the region layout with \
                         `tools/mesh/regions_from_msh.py` and load it through \
                         `ofgpu-regions` (SPEC-LIT 97)",
                        r.name, pr.poly_mesh
                    )));
                }
                raw
            } else {
                read_poly_mesh(&path)?
            };
            check_imported_patches(&r.name, kind, &raw, empties, openings)?;
            let mesh = build_host_mesh(&raw)?;
            return Ok((mesh, raw));
        }
    };
    let bounds = &b.bounds;
    let axis = |i: usize| -> Result<GradedAxis> {
        let (lo, hi) = (bounds.min[i] as Scalar, bounds.max[i] as Scalar);
        if !(hi > lo) {
            return Err(Error::Config(format!(
                "regions/{}/mesh/bounds: axis {i} runs from {lo} to {hi}",
                r.name
            )));
        }
        if b.cells[i] == 0 {
            return Err(Error::Config(format!(
                "regions/{}/mesh/cells: axis {i} has no cells",
                r.name
            )));
        }
        let mut a = GradedAxis {
            lo,
            hi,
            n: b.cells[i] as usize,
            expansion: 1.0,
            two_sided: false,
        };
        let g = b.grading.as_ref().and_then(|g| match i {
            0 => g.x.as_ref(),
            1 => g.y.as_ref(),
            _ => g.z.as_ref(),
        });
        apply_grading(&r.name, i, &mut a, g)?;
        Ok(a)
    };

    let names = b.boundaries.names();
    let base = if r.kind == "fluid" { "wall" } else { "patch" };
    let n_empty = names.iter().filter(|n| empties.contains(n)).count();
    // `empty` faces come in OPPOSITE pairs, and blockgen's own check
    // ("an empty patch is only legal with a single cell in that direction")
    // catches the cell count. What it cannot catch is a case that made one of
    // a pair empty and left the other a wall, which would put a real wall on
    // one side of a one-cell-thick domain and nothing on the other.
    if n_empty != 0 && n_empty != 2 {
        return Err(Error::Config(format!(
            "regions/{}: {n_empty} patch(es) are `empty`. They come in opposite \
             pairs - both faces of the thin direction, or neither",
            r.name
        )));
    }
    if n_empty == 2 {
        let axis_of = |i: usize| i / 2;
        let mut axes: Vec<usize> = (0..6)
            .filter(|i| empties.contains(&names[*i]))
            .map(axis_of)
            .collect();
        axes.dedup();
        if axes.len() != 1 {
            return Err(Error::Config(format!(
                "regions/{}: the two `empty` patches are not the two faces of \
                 one axis. An empty pair is the front and back of a 2-D case",
                r.name
            )));
        }
    }
    let spec = BlockSpec {
        x: axis(0)?,
        y: axis(1)?,
        z: axis(2)?,
        patch_name: std::array::from_fn(|i| names[i].to_string()),
        patch_type: std::array::from_fn(|i| {
            if empties.contains(&names[i]) {
                "empty".to_string()
            } else if openings.contains(&names[i]) {
                // SPEC-LIT §79.2: an opening is `patch`, not `wall`. The
                // distinction is not cosmetic - `PatchKind::Wall` is what a
                // wall function targets, and an outlet is not a wall.
                "patch".to_string()
            } else {
                base.to_string()
            }
        }),
        windows: Vec::new(),
        cyclic: Vec::new(),
    };
    // `build_mesh` is `build_host_mesh(&raw_mesh(b)?)` by definition
    // (blockgen keeps the two in step under that exact identity); handing
    // the raw mesh back alongside costs nothing and is what the per-region
    // VTU writer and `attach_points` consume.
    let raw = blockgen::raw_mesh(&spec)?;
    let mesh = build_host_mesh(&raw)?;
    Ok((mesh, raw))
}

/// §97.2's path refusals, in this order. On success the JOINED path is
/// returned - not the canonical one, so the reader's own error messages keep
/// printing the path as the case spelled it against the case directory.
fn resolve_mesh_path(region: &str, case_dir: Option<&Path>, p: &str) -> Result<PathBuf> {
    let path = Path::new(p);
    // 1. No directory at all. `lower()` is this shape, and a polyMesh path
    //    is RELATIVE TO THE CASE FILE'S DIRECTORY - there is nothing to
    //    resolve it against.
    let Some(dir) = case_dir else {
        return Err(Error::Config(format!(
            "regions/{region}/mesh/polyMesh: '{p}' is relative to the case file's \
             directory, and this document was lowered without one. Call \
             `ChtCase::lower_in(Some(&case_dir))` - as `ofgpu-cht` does with \
             `case_path.parent()` - to import a polyMesh region"
        )));
    };
    // 0. `Path::new("case.cht.jsonc").parent()` is `Some("")`, which IS the
    //    current directory and says so.
    let dir = if dir.as_os_str().is_empty() { Path::new(".") } else { dir };
    // 2. Absolute.
    if path.is_absolute() {
        return Err(Error::Config(format!(
            "regions/{region}/mesh/polyMesh: '{p}' is absolute. The path must be \
             RELATIVE to the case file's directory - a case that only opens from \
             one absolute location is a case that cannot be moved (SPEC-LIT 97.2)"
        )));
    }
    // 3. Missing.
    let joined = dir.join(path);
    if !joined.exists() {
        return Err(Error::Config(format!(
            "regions/{region}/mesh/polyMesh: '{}' does not exist (case directory \
             '{}'). A polyMesh directory, a case root or `constant` holding one, \
             or a single-volume `.msh` file",
            joined.display(),
            dir.display()
        )));
    }
    // 4. Outside.
    let inside = dir
        .canonicalize()
        .ok()
        .map(|root| joined.canonicalize().ok().is_some_and(|p| p.starts_with(root)))
        .unwrap_or(false);
    if !inside {
        return Err(Error::Config(format!(
            "regions/{region}/mesh/polyMesh: '{}' resolves outside the case \
             directory '{}'. A case is self-contained: its regions' meshes live \
             under the directory the case file is in (SPEC-LIT 97.2)",
            joined.display(),
            dir.display()
        )));
    }
    Ok(joined)
}

/// §97.2's patch-TYPE refusals on an imported region. The patch list is the
/// mesh's own `boundary` file, and the case must agree with it where the two
/// can disagree: an `empty` is a patch type AND a rule, so it is checked in
/// both directions, and an opening is `patch`, not `wall`.
fn check_imported_patches(
    region: &str,
    kind: RegionKind,
    raw: &PolyMeshRaw,
    empties: &[&str],
    openings: &[&str],
) -> Result<()> {
    for p in &raw.patches {
        if matches!(p.kind, PatchKind::Cyclic | PatchKind::Processor) {
            return Err(Error::Config(format!(
                "regions/{region}/mesh/polyMesh: patch '{}' is of type '{}' - a \
                 periodic or decomposed conducting region is not gated; SPEC-LIT \
                 31's cyclic pair on a concatenated thermal mesh is unexercised",
                p.name, p.type_name
            )));
        }
        let is_empty = matches!(p.kind, PatchKind::Empty);
        if is_empty && !empties.contains(&p.name.as_str()) {
            return Err(Error::Config(format!(
                "regions/{region}: patch '{}' is of type `empty` in the mesh but \
                 carries no `empty` rule in the case. The mesh's patch types are \
                 the mesh's own - write {{ \"match\": \"{}\", \"T\": {{ \"type\": \
                 \"empty\" }} }} (SPEC-LIT 97.2)",
                p.name, p.name
            )));
        }
        if !is_empty && empties.contains(&p.name.as_str()) {
            return Err(Error::Config(format!(
                "regions/{region}: patch '{}' carries an `empty` rule but the mesh's \
                 boundary file types it '{}'. An `empty` patch contributes to no \
                 surface integral at all, and only the mesh can make one",
                p.name, p.type_name
            )));
        }
    }
    if kind == RegionKind::Fluid {
        for name in openings {
            let Some(p) = raw.patches.iter().find(|p| p.name == *name) else {
                continue; // named in the mesh or not, the unnamed-patch rule reports it
            };
            if matches!(p.kind, PatchKind::Wall) {
                return Err(Error::Config(format!(
                    "regions/{region}/mesh/polyMesh: patch '{}' is of type `wall`, \
                     and the case names it an opening. An opening is `patch`, not \
                     `wall` - SPEC-LIT 79.2, the same distinction the block build \
                     writes: `PatchKind::Wall` is what a wall function targets, and \
                     an outlet is not a wall",
                    p.name
                )));
            }
        }
    }
    Ok(())
}

fn apply_grading(
    region: &str,
    axis: usize,
    a: &mut GradedAxis,
    g: Option<&JsonGradingAxis>,
) -> Result<()> {
    let Some(g) = g else { return Ok(()) };
    if !(g.expansion > 0.0) {
        return Err(Error::Config(format!(
            "regions/{region}/mesh/grading: axis {axis} has expansion {}, which is \
             not a cell-size ratio",
            g.expansion
        )));
    }
    a.expansion = g.expansion as Scalar;
    a.two_sided = g.two_sided;
    Ok(())
}

/// The JSON Schema for [`ChtCase`], generated from these same types by
/// `schemars` - the same discipline `crate::io::case_json::emit_schema` runs
/// under: the schema cannot disagree with the reader, because it IS the
/// reader's own types, printed (SPEC-LIT §96.1).
pub fn emit_cht_schema() -> String {
    serde_json::to_string_pretty(&schemars::schema_for!(ChtCase)).unwrap_or_default()
}

#[cfg(test)]
mod tests;
