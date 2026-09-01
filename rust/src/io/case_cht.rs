// meteor-cfd - Copyright (c) 2026 주식회사 메테오시뮬레이션 (Meteo Simulation Co., Ltd.)
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
use std::path::Path;

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
use crate::io::case_json::{JsonBounds, JsonGrading, JsonGradingAxis};
use crate::mesh::HostMesh;
use crate::{Label, Scalar};

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

/// An axis-aligned block. Patch names are the case's, one per face.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ChtRegionMesh {
    pub bounds: JsonBounds,
    pub cells: [u32; 3],
    /// The six face names, `-x +x -y +y -z +z`.
    pub boundaries: ChtBoundaries,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub grading: Option<JsonGrading>,
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
    /// anywhere else. Uniform over the patch: SPEC-LIT §79.3's
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
    /// SPEC-LIT §79.4's outflow condition: `zeroGradient` while the flux
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
    /// SPEC-LIT §79.4: `refValue = inletValue`, `refGrad = 0`, and `fr`
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

/// Everything [`crate::cht`] needs, with every name already resolved.
#[derive(Debug)]
pub struct LoweredChtCase {
    pub name: String,
    pub region_names: Vec<String>,
    pub kinds: Vec<RegionKind>,
    pub meshes: Vec<HostMesh>,
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
    /// Resolve every name, build every block, and refuse everything §13.4
    /// says must be refused.
    pub fn lower(&self) -> Result<LoweredChtCase> {
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
        let mut materials = Vec::new();
        let mut fluids: Vec<Option<FluidMaterial>> = Vec::new();
        let mut sources = Vec::new();
        // Which patches of which region have been spoken for, and by what.
        let mut claimed: Vec<BTreeMap<String, &'static str>> = Vec::new();

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

            // Every patch of every region, listed before anything claims one.
            let mut seen: BTreeMap<String, &'static str> = BTreeMap::new();
            for n in r.mesh.boundaries.names() {
                seen.entry(n.to_string()).or_insert("unnamed");
            }

            // Which patches are `empty` has to be known BEFORE the block is
            // built, because it is the mesh's patch TYPE and not a condition
            // written onto it afterwards.
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
                        crate::io::contract::unsupported(
                            &format!("regions/{}/patches/{}/kind", r.name, rule.match_),
                            other,
                            &["wall", "inlet", "outlet"],
                            "wall - SPEC-LIT 60.2's no-slip patch, which is what every fluid patch \
                             was before SPEC-LIT 79",
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
            let mesh = build_region_mesh(r, &empties, &flow_patches)?;

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

            // SPEC-LIT 60.3: SPEC-LIT 18's registry is not wired to this
            // format's fluid side, and a source that is read and dropped is
            // exactly the 13.4.1 defect.
            if kind == RegionKind::Fluid && r.source.is_some() {
                return Err(Error::Config(format!(
                    "regions/{}/source: a volumetric heat source on a FLUID \
                     region is not implemented on this format. SPEC-LIT 18's \
                     registry is what would carry it, and this reader does not \
                     reach it - so the entry is refused rather than read and \
                     dropped (SPEC-LIT 13.4.1). Put the source in a solid \
                     region, or use `ofgpu-fire`",
                    r.name
                )));
            }

            region_names.push(r.name.clone());
            kinds.push(kind);
            meshes.push(mesh);
            materials.push(mat);
            fluids.push(fluid);
            sources.push(r.source.unwrap_or(0.0) as Scalar);
            claimed.push(seen);
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
                            self.regions[r].mesh.boundaries.names().join(", ")
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
                            region.mesh.boundaries.names().join(", ")
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
                             patch. SPEC-LIT 79.3's flux-establishment pass is built from its \
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

                // SPEC-LIT §79.4: which `T` each patch kind may carry. Every
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
                             (SPEC-LIT 79.4)"
                        )))
                    }
                    (ChtScalarBc::ZeroGradient, _, true) => {}
                    (_, _, true) => {
                        return Err(Error::Config(format!(
                            "{path}/T: an `outlet` carries `inletOutlet` or `zeroGradient`. A held \
                             temperature at an outlet conducts heat back INTO a domain the flow is \
                             leaving, and a prescribed flux there is a wall condition on a face \
                             that is not a wall (SPEC-LIT 79.4)"
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
        // level each to decide how the inflow splits, and SPEC-LIT §79.3's
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
                     SPEC-LIT 79.3's flux-establishment solve carries exactly one Dirichlet \
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
        // SPEC-LIT §79.5 loosened exactly one row of this: `buoyancy` is
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

        Ok(LoweredChtCase {
            name: self.name.clone(),
            region_names,
            kinds,
            meshes,
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

/// One region's block, through the same `blockgen` every other case uses.
///
/// `empties` is the patch names the case gave `"T": { "type": "empty" }` -
/// SPEC-LIT §60.2's 2-D front and back. They have to be known here rather than
/// written on afterwards, because `empty` is the mesh's patch TYPE: an
/// `empty` face contributes to no surface integral at all, which is a
/// property of the topology and not a boundary condition.
///
/// A solid region's other faces stay plain `patch`, which is what §47.14's
/// format has always done and what a conduction stack wants. A **fluid**
/// region's become `wall`, because they are no-slip walls in the momentum
/// sense (SPEC-LIT §60.2) and `momFluxIsPrescribed` asks the mesh, not the
/// case.
fn build_region_mesh(r: &ChtRegion, empties: &[&str], openings: &[&str]) -> Result<HostMesh> {
    let b = &r.mesh.bounds;
    let axis = |i: usize| -> Result<GradedAxis> {
        let (lo, hi) = (b.min[i] as Scalar, b.max[i] as Scalar);
        if !(hi > lo) {
            return Err(Error::Config(format!(
                "regions/{}/mesh/bounds: axis {i} runs from {lo} to {hi}",
                r.name
            )));
        }
        if r.mesh.cells[i] == 0 {
            return Err(Error::Config(format!(
                "regions/{}/mesh/cells: axis {i} has no cells",
                r.name
            )));
        }
        let mut a = GradedAxis {
            lo,
            hi,
            n: r.mesh.cells[i] as usize,
            expansion: 1.0,
            two_sided: false,
        };
        let g = r.mesh.grading.as_ref().and_then(|g| match i {
            0 => g.x.as_ref(),
            1 => g.y.as_ref(),
            _ => g.z.as_ref(),
        });
        apply_grading(&r.name, i, &mut a, g)?;
        Ok(a)
    };

    let names = r.mesh.boundaries.names();
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
    blockgen::build_mesh(&spec)
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

#[cfg(test)]
mod tests;
