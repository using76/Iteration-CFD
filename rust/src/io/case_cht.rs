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
//! It is **not** a fluid case. There is no `physics.fluid`, no momentum, no
//! turbulence: a region whose `kind` is anything but `solid` is refused by
//! name. The fluid side of a conjugate problem - the `k_eff Delta` or
//! wall-function conductance of §47.6 - is implemented in `crate::cht` and
//! exercised by its tests, but no *case format* reaches it yet, and this file
//! deliberately does not pretend otherwise. §47.9's `coupledTemperature` is
//! likewise refused as a patch entry here, because on this format an
//! interface is declared by the `interfaces` block, and a patch that named
//! the condition without an interface behind it would be a setting the case
//! can say and the solver ignores - the §13.4.1 defect.
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
use crate::cht::{
    Conductivity, InterfaceRequest, PairingTolerances, RegionKind, SolidMaterial,
};
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
    pub initial: ChtInitial,
    pub run: ChtRun,
    #[serde(default)]
    pub numerics: ChtNumerics,
}

/// One region: a block of cells, one material, its own boundary conditions.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields, rename_all = "camelCase")]
pub struct ChtRegion {
    pub name: String,
    /// `"solid"`. Anything else is a §13.4 error - see the module doc.
    #[serde(default = "solid_kind")]
    pub kind: String,
    pub mesh: ChtRegionMesh,
    pub material: ChtMaterial,
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

/// One patch's temperature condition.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields, rename_all = "camelCase")]
pub struct ChtPatchRule {
    /// The patch name, as `mesh.boundaries` spells it. Exact, not a pattern:
    /// a conduction stack has a handful of named faces and a pattern would
    /// only make it possible to match none of them by accident.
    #[serde(rename = "match")]
    pub match_: String,
    #[serde(rename = "T")]
    pub t: ChtScalarBc,
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
}

impl Default for ChtNumerics {
    fn default() -> Self {
        Self {
            solver: "PCG".to_string(),
            preconditioner: "DIC".to_string(),
            tolerance: 1e-12,
            max_iter: 2000,
            n_non_orthogonal_correctors: 0,
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
}

impl LoweredBc {
    pub fn kind(self) -> BcKind {
        match self {
            Self::FixedValue(_) => BcKind::FixedValue,
            Self::ZeroGradient => BcKind::ZeroGradient,
            Self::FixedFlux(_) => BcKind::FixedFluxTemperature,
        }
    }
}

/// Everything [`crate::cht`] needs, with every name already resolved.
#[derive(Debug)]
pub struct LoweredChtCase {
    pub name: String,
    pub region_names: Vec<String>,
    pub meshes: Vec<HostMesh>,
    pub materials: Vec<SolidMaterial>,
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
    /// The region kinds, in order - all `Solid` on this format.
    pub fn kinds(&self) -> Vec<RegionKind> {
        vec![RegionKind::Solid; self.meshes.len()]
    }
}

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
        let mut meshes = Vec::new();
        let mut materials = Vec::new();
        let mut sources = Vec::new();
        // Which patches of which region have been spoken for, and by what.
        let mut claimed: Vec<BTreeMap<String, &'static str>> = Vec::new();

        for r in &self.regions {
            if r.kind != "solid" {
                return crate::io::contract::unsupported(
                    &format!("regions/{}/kind", r.name),
                    &r.kind,
                    &["solid"],
                    "solid - this case format solves SPEC-LIT 46's SOLID energy \
                     equation over a stack of conducting regions. A fluid region \
                     needs momentum, turbulence and a flux, none of which this \
                     format carries; the conjugate interface itself (SPEC-LIT 47) \
                     is implemented and tested in `crate::cht`, but no case format \
                     reaches its fluid side yet",
                    (),
                )
                .map(|()| unreachable!());
            }

            let mesh = build_region_mesh(r)?;
            let names = r.mesh.boundaries.names();
            let mut seen: BTreeMap<String, &'static str> = BTreeMap::new();
            for n in names {
                seen.entry(n.to_string()).or_insert("unnamed");
            }

            let mat = SolidMaterial {
                name: r.name.clone(),
                rho: r.material.rho as Scalar,
                c: r.material.c as Scalar,
                k: Conductivity::parse(
                    &r.material.kappa.values(),
                    &format!("regions/{}/material/kappa", r.name),
                )?,
            };
            mat.validate()?;

            region_names.push(r.name.clone());
            meshes.push(mesh);
            materials.push(mat);
            sources.push(r.source.unwrap_or(0.0) as Scalar);
            claimed.push(seen);
        }

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
            meshes,
            materials,
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
    }
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
fn build_region_mesh(r: &ChtRegion) -> Result<HostMesh> {
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
    let spec = BlockSpec {
        x: axis(0)?,
        y: axis(1)?,
        z: axis(2)?,
        patch_name: std::array::from_fn(|i| names[i].to_string()),
        // Every face is a plain `patch`. A conduction region has no walls in
        // the momentum sense and no empties: an `empty` patch contributes
        // nothing to any surface integral, which for a conduction case would
        // silently make one direction adiabatic whatever the case said.
        patch_type: std::array::from_fn(|_| "patch".to_string()),
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
