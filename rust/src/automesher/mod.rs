// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.
// Provenance: see PROVENANCE.md. No GPL-licensed source was consulted.

//! The automesher (SPEC-LIT §92): this crate's own mesher, built so that the
//! mesh it emits has already passed its own quality gate. Tranche 1 is the
//! hex-dominant path - octree levels, castellation, snapping, layers; the
//! tetrahedral path of §92.4 is specified there and deliberately not built.
//!
//! This unit ships only the gate. §92.3's seven checks - positive volume,
//! closure, one cell region, non-orthogonality, thickness, gradient
//! conditioning, addressing - are the precondition every later stage of the
//! mesher must hold to, and the refusal a stage that cannot ends the run
//! with. The mesher that has to satisfy them is the rest of §92.
//!
//! Provenance: ORIGINAL - the seven checks are this project's own, stated in
//! SPEC-LIT §92.3 with equations (92.11)-(92.15) and the refusal message
//! format fixed there so tests can assert it. No GPL-licensed source was
//! consulted.

pub mod quality;

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::error::Error;

// ==========================================================================
//  The config tree - SPEC-LIT §92.2's pipeline, one struct per stage
// ==========================================================================

/// The whole automesher config: SPEC-LIT §92.2's hex-dominant pipeline read
/// top to bottom - `input` -> `domain` (stage 0) -> `refinement` (stage 1)
/// -> `castellation` (stage 3) -> `snap` (stage 4) -> `layers` -> `quality`
/// (§92.3's gate) -> `output`. `$schema` is accepted and ignored, exactly as
/// [`crate::io::case_json::JsonCase`] treats it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct AutomeshConfig {
    #[serde(rename = "$schema", default, skip_serializing_if = "Option::is_none")]
    pub schema: Option<String>,
    pub input: InputSpec,
    pub domain: DomainSpec,
    #[serde(default)]
    pub refinement: RefinementSpec,
    #[serde(default)]
    pub castellation: CastellationSpec,
    #[serde(default)]
    pub snap: SnapSpec,
    #[serde(default)]
    pub layers: LayerSpec,
    #[serde(default)]
    pub quality: QualitySpec,
    pub output: OutputSpec,
}

/// `input` - SPEC-LIT §92.2: the surfaces the mesher casts against
/// (stage 3) and snaps to (stage 4).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct InputSpec {
    /// One or more STL files. A `name` overrides the solid names in the file.
    pub surfaces: Vec<SurfaceInput>,
}

/// One `input.surfaces[]` entry - one STL file on disk.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct SurfaceInput {
    pub path: String,
    #[serde(default)]
    pub name: Option<String>,
}

/// `domain` - SPEC-LIT §92.2 stage 0: the background block the octree
/// subdivides rather than replaces. Built as a `blockgen::BlockSpec` over
/// `extent`, and it must contain the surface's bounding box with at least
/// one base cell of margin or the run refuses before any work is done.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct DomainSpec {
    /// `[xlo, xhi, ylo, yhi, zlo, zhi]`, metres.
    pub extent: [f64; 6],
    /// Target background cell size, metres (SPEC-LIT §92.2 stage 0).
    pub base_size: f64,
    /// Per-axis one-sided expansion, last cell / first cell. Default all 1.
    #[serde(default = "grading_one")]
    pub grading: [f64; 3],
}

/// [`DomainSpec::grading`]'s default: no expansion on any axis.
fn grading_one() -> [f64; 3] {
    [1.0; 3]
}

impl Default for DomainSpec {
    fn default() -> Self {
        Self { extent: [0.0; 6], base_size: 0.0, grading: grading_one() }
    }
}

/// `refinement` - SPEC-LIT §92.2 stage 1: the octree levels the distance
/// bands and the feature angle ask for, capped at [`RefinementSpec::max_level`]
/// (eq. 92.1's `l(c)`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct RefinementSpec {
    /// Per-patch distance bands. Empty: no band-driven refinement.
    #[serde(default)]
    pub levels: Vec<RefinementBand>,
    /// Dihedral angle past which a triangulation edge is a feature edge
    /// (eq. 92.2). Default 30 degrees, the value SPEC-LIT §92.2 quotes.
    #[serde(default = "d_feature_angle")]
    pub feature_angle_deg: f64,
    /// The level cap `l(c)` is min'd with (eq. 92.1). SPEC-LIT §74.2 caps
    /// the octree at 6.
    #[serde(default = "d_max_level")]
    pub max_level: u32,
}

/// One `refinement.levels[]` entry - one patch's distance bands.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct RefinementBand {
    /// Patch (STL solid) name this band applies to.
    pub patch: String,
    /// `[[distance_m, level], ...]` - within `distance` of the patch, at
    /// least `level`. SPEC-LIT §92.2 (92.1).
    pub bands: Vec<DistanceBand>,
}

/// One distance band of [`RefinementBand`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct DistanceBand {
    pub distance: f64,
    pub level: u32,
}

fn d_feature_angle() -> f64 {
    30.0
}

fn d_max_level() -> u32 {
    2
}

impl Default for RefinementSpec {
    fn default() -> Self {
        Self {
            levels: Vec::new(),
            feature_angle_deg: d_feature_angle(),
            max_level: d_max_level(),
        }
    }
}

/// `castellation` - SPEC-LIT §92.2 stage 3: parity classification, then the
/// keep-set of eq. (92.4) - the connected component that contains the seed,
/// which is what removes the sealed pockets a mesher leaves under buildings.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct CastellationSpec {
    /// Which connected component of the fluid leaves survives eq. (92.4).
    #[serde(default)]
    pub keep_region: KeepRegion,
    /// The keep point `keep_region = "seed"` needs; ignored otherwise.
    #[serde(default)]
    pub seed_point: Option<[f64; 3]>,
    /// A kept cell with fewer faces than this is dropped - a two-faced cell
    /// is a hole in the addressing, not a control volume.
    #[serde(default = "d_min_faces")]
    pub min_faces: usize,
}

/// [`CastellationSpec::keep_region`]'s two choices - SPEC-LIT §92.2 stage 3.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema, Default)]
#[serde(rename_all = "camelCase")]
pub enum KeepRegion {
    /// The biggest connected component (the default).
    #[default]
    Largest,
    /// The component containing `castellation.seed_point`.
    Seed,
}

fn d_min_faces() -> usize {
    4
}

impl Default for CastellationSpec {
    fn default() -> Self {
        Self { keep_region: KeepRegion::default(), seed_point: None, min_faces: d_min_faces() }
    }
}

/// `snap` - SPEC-LIT §92.2 stage 4: boundary points displaced to the closest
/// point on the surface (92.5), the displacement field smoothed (92.6), and
/// any displacement that would break the §92.3 gate undone (92.7).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct SnapSpec {
    /// Snapping iterations, the `k` of eq. (92.5).
    #[serde(default = "d_snap_iters")]
    pub iterations: usize,
    /// Points closer than this to the surface are considered on it.
    #[serde(default = "d_snap_tol")]
    pub tolerance: f64,
    /// Laplacian smoothing passes over the displacement field, eq. (92.6).
    #[serde(default = "d_smoothing_passes")]
    pub smoothing_passes: usize,
    /// The smoothing relaxation weight, in `[0, 1]`.
    #[serde(default = "d_smoothing")]
    pub smoothing: f64,
    /// How many times a breaking displacement is halved before it is set to
    /// zero - eq. (92.7)'s undo, §92.3's repair step 1.
    #[serde(default = "d_undo_limit")]
    pub undo_limit: usize,
}

fn d_snap_iters() -> usize {
    30
}

fn d_snap_tol() -> f64 {
    1e-3
}

fn d_smoothing_passes() -> usize {
    3
}

fn d_smoothing() -> f64 {
    0.5
}

fn d_undo_limit() -> usize {
    4
}

impl Default for SnapSpec {
    fn default() -> Self {
        Self {
            iterations: d_snap_iters(),
            tolerance: d_snap_tol(),
            smoothing_passes: d_smoothing_passes(),
            smoothing: d_smoothing(),
            undo_limit: d_undo_limit(),
        }
    }
}

/// `layers` - SPEC-LIT §92.2, wall-normal layers, eqs. (92.9)-(92.10).
/// Default: no layers anywhere.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct LayerSpec {
    /// The patches the layers are added to. Empty: none.
    #[serde(default)]
    pub patches: Vec<String>,
    /// The number of layers, eq. (92.9)'s `n`. Zero: no layers.
    #[serde(default)]
    pub n: usize,
    /// The first layer's thickness, metres.
    #[serde(default = "d_first_thickness")]
    pub first_thickness: f64,
    /// The expansion from one layer to the next.
    #[serde(default = "d_growth")]
    pub growth: f64,
    /// The total thickness below which the layers are dropped.
    #[serde(default = "d_min_thickness")]
    pub min_thickness: f64,
    /// Fraction of the local cell size below which the layers are dropped.
    #[serde(default = "d_medial_frac")]
    pub medial_frac: f64,
}

fn d_first_thickness() -> f64 {
    0.05
}

fn d_growth() -> f64 {
    1.3
}

fn d_min_thickness() -> f64 {
    0.1
}

fn d_medial_frac() -> f64 {
    0.5
}

impl Default for LayerSpec {
    fn default() -> Self {
        Self {
            patches: Vec::new(),
            n: 0,
            first_thickness: d_first_thickness(),
            growth: d_growth(),
            min_thickness: d_min_thickness(),
            medial_frac: d_medial_frac(),
        }
    }
}

/// `quality` - SPEC-LIT §92.3's gate thresholds, G2 (92.12) through G6
/// (92.15). G1, G3 and G7 are pass/fail with no knob. A stage of the mesher
/// that cannot hold the mesh inside these ends the run with the refusal
/// §92.3 fixes the message format of.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct QualitySpec {
    /// G2 (92.12): |sum s Sf| / V^(2/3) must stay under this (the crate's
    /// own `mesh::geometry::CLOSURE_LIMIT`).
    #[serde(default = "d_max_closure")]
    pub max_closure: f64,
    /// G4 (92.13): max internal-face non-orthogonality, degrees.
    #[serde(default = "d_max_non_orth")]
    pub max_non_orth_deg: f64,
    /// G4: internal faces past this angle are counted and reported.
    #[serde(default = "d_report_non_orth")]
    pub report_non_orth_deg: f64,
    /// G5 (92.14): tau_c = 3 V_c / A_max(c)^(3/2) must stay at or above this.
    #[serde(default = "d_min_thickness_ratio")]
    pub min_thickness_ratio: f64,
    /// G6 (92.15): cond(T_c) must stay under this.
    #[serde(default = "d_max_cond")]
    pub max_cond: f64,
}

fn d_max_closure() -> f64 {
    1e-10
}

fn d_max_non_orth() -> f64 {
    70.0
}

fn d_report_non_orth() -> f64 {
    60.0
}

fn d_min_thickness_ratio() -> f64 {
    0.05
}

fn d_max_cond() -> f64 {
    1e4
}

impl Default for QualitySpec {
    fn default() -> Self {
        Self {
            max_closure: d_max_closure(),
            max_non_orth_deg: d_max_non_orth(),
            report_non_orth_deg: d_report_non_orth(),
            min_thickness_ratio: d_min_thickness_ratio(),
            max_cond: d_max_cond(),
        }
    }
}

impl QualitySpec {
    /// The gate's own thresholds, as [`quality::check`] takes them. The
    /// config is f64; `Scalar` is f32 under the `single` feature.
    ///
    /// Written by the supervising session, not by the coding agent.
    pub fn thresholds(&self) -> quality::QualityThresholds {
        quality::QualityThresholds {
            max_closure: self.max_closure as crate::Scalar,
            max_non_orth_deg: self.max_non_orth_deg as crate::Scalar,
            report_non_orth_deg: self.report_non_orth_deg as crate::Scalar,
            min_thickness_ratio: self.min_thickness_ratio as crate::Scalar,
            max_cond: self.max_cond as crate::Scalar,
        }
    }
}

/// `output` - where the mesh and the case that uses it are written.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct OutputSpec {
    pub case_dir: String,
    pub name: String,
}

// ==========================================================================
//  Reading, schema, and the checks that need no STL
// ==========================================================================

/// Read a JSONC automesher config. Uses
/// [`crate::io::case_json::parse_jsonc_file`] so comments, trailing commas
/// and `serde_path_to_error` paths work exactly as they do for a case file.
pub fn read_config(path: &std::path::Path) -> crate::error::Result<AutomeshConfig> {
    crate::io::case_json::parse_jsonc_file(path)
}

/// [`read_config`] on text already in memory - what a test that feeds a
/// deliberately mistyped key needs.
pub fn read_config_str(text: &str, what: &str) -> crate::error::Result<AutomeshConfig> {
    crate::io::case_json::parse_jsonc_str(text, what)
}

/// The JSON Schema of [`AutomeshConfig`], generated from the SAME types that
/// parse it - see [`crate::io::case_json::emit_schema`] for the pattern.
pub fn emit_schema() -> String {
    let schema = schemars::schema_for!(AutomeshConfig);
    serde_json::to_string_pretty(&schema).unwrap_or_default()
}

impl AutomeshConfig {
    /// Cheap checks that do not need the STL. Each failure names the FIELD
    /// and the value, so a bad config is refused before any meshing work.
    pub fn validate(&self) -> crate::error::Result<()> {
        let e = &self.domain.extent;
        if !(e[1] > e[0]) {
            return Err(Error::Mesh(format!(
                "domain.extent: x-axis is empty or reversed (xlo = {}, xhi = {})",
                e[0], e[1]
            )));
        }
        if !(e[3] > e[2]) {
            return Err(Error::Mesh(format!(
                "domain.extent: y-axis is empty or reversed (ylo = {}, yhi = {})",
                e[2], e[3]
            )));
        }
        if !(e[5] > e[4]) {
            return Err(Error::Mesh(format!(
                "domain.extent: z-axis is empty or reversed (zlo = {}, zhi = {})",
                e[4], e[5]
            )));
        }
        if !(self.domain.base_size > 0.0) {
            return Err(Error::Mesh(format!(
                "domain.base_size: must be > 0, got {}",
                self.domain.base_size
            )));
        }
        let g = &self.domain.grading;
        if !(g[0] > 0.0 && g[1] > 0.0 && g[2] > 0.0) {
            return Err(Error::Mesh(format!(
                "domain.grading: every axis must be > 0, got [{}, {}, {}]",
                g[0], g[1], g[2]
            )));
        }
        if self.refinement.max_level > 6 {
            return Err(Error::Mesh(format!(
                "refinement.max_level: {} exceeds the {} levels SPEC-LIT §74.2 caps the octree at",
                self.refinement.max_level, 6
            )));
        }
        if !(self.snap.smoothing >= 0.0 && self.snap.smoothing <= 1.0) {
            return Err(Error::Mesh(format!(
                "snap.smoothing: must lie in [0, 1], got {}",
                self.snap.smoothing
            )));
        }
        let q = &self.quality;
        for (field, value) in [
            ("quality.max_closure", q.max_closure),
            ("quality.max_non_orth_deg", q.max_non_orth_deg),
            ("quality.report_non_orth_deg", q.report_non_orth_deg),
            ("quality.min_thickness_ratio", q.min_thickness_ratio),
            ("quality.max_cond", q.max_cond),
        ] {
            if !(value > 0.0) {
                return Err(Error::Mesh(format!(
                    "{field}: threshold must be > 0, got {value}"
                )));
            }
        }
        if self.input.surfaces.is_empty() {
            return Err(Error::Mesh(
                "input.surfaces: at least one STL is required, got none"
                    .to_string(),
            ));
        }
        if !(self.layers.growth > 0.0) {
            return Err(Error::Mesh(format!(
                "layers.growth: must be > 0, got {}",
                self.layers.growth
            )));
        }
        if self.castellation.keep_region == KeepRegion::Seed
            && self.castellation.seed_point.is_none()
        {
            return Err(Error::Mesh(
                "castellation.seed_point: keep_region = \"seed\" needs a seed_point, got none"
                    .to_string(),
            ));
        }
        Ok(())
    }

    /// `(lo, hi)` of `domain.extent` as two points.
    pub fn extent_bounds(&self) -> (crate::Vec3, crate::Vec3) {
        // The config is f64 throughout; `Scalar` is f32 under the `single`
        // feature, so the casts are what make this compile both ways.
        let e = &self.domain.extent;
        let s = |v: f64| v as crate::Scalar;
        (
            crate::Vec3::new(s(e[0]), s(e[2]), s(e[4])),
            crate::Vec3::new(s(e[1]), s(e[3]), s(e[5])),
        )
    }
}

// ==========================================================================
//  Tests
// ==========================================================================

#[cfg(test)]
mod config_tests {
    use super::*;

    fn minimal() -> AutomeshConfig {
        AutomeshConfig {
            schema: None,
            input: InputSpec {
                surfaces: vec![SurfaceInput { path: "site.stl".to_string(), name: None }],
            },
            domain: DomainSpec {
                extent: [0.0, 10.0, 0.0, 10.0, 0.0, 10.0],
                base_size: 1.0,
                grading: [1.0; 3],
            },
            refinement: RefinementSpec::default(),
            castellation: CastellationSpec::default(),
            snap: SnapSpec::default(),
            layers: LayerSpec::default(),
            quality: QualitySpec::default(),
            output: OutputSpec { case_dir: "out".to_string(), name: "site".to_string() },
        }
    }

    fn minimal_text() -> String {
        String::from(
            "{ \"input\": { \"surfaces\": [ { \"path\": \"site.stl\" } ] },\n\
             \"domain\": { \"extent\": [0,10,0,10,0,10], \"base_size\": 1.0 },\n\
             \"output\": { \"case_dir\": \"out\", \"name\": \"site\" } }",
        )
    }

    #[test]
    fn config_round_trips_through_serde() {
        let mut cfg = minimal();
        cfg.schema = Some("automesher.schema.json".to_string());
        cfg.input
            .surfaces
            .push(SurfaceInput { path: "ship.stl".to_string(), name: Some("hull".to_string()) });
        cfg.domain.grading = [1.0, 2.0, 1.0];
        cfg.refinement = RefinementSpec {
            levels: vec![RefinementBand {
                patch: "site".to_string(),
                bands: vec![
                    DistanceBand { distance: 4.0, level: 1 },
                    DistanceBand { distance: 1.0, level: 2 },
                ],
            }],
            feature_angle_deg: 45.0,
            max_level: 3,
        };
        cfg.castellation = CastellationSpec {
            keep_region: KeepRegion::Seed,
            seed_point: Some([1.0, 1.0, 1.0]),
            min_faces: 6,
        };
        cfg.snap.iterations = 10;
        cfg.snap.tolerance = 1e-4;
        cfg.snap.smoothing_passes = 5;
        cfg.snap.smoothing = 0.25;
        cfg.snap.undo_limit = 2;
        cfg.layers = LayerSpec {
            patches: vec!["ground".to_string()],
            n: 3,
            first_thickness: 0.01,
            growth: 1.2,
            min_thickness: 0.05,
            medial_frac: 0.4,
        };
        cfg.quality.max_non_orth_deg = 65.0;
        cfg.quality.report_non_orth_deg = 50.0;
        let json = serde_json::to_string(&cfg).unwrap();
        let back = read_config_str(&json, "round-trip").unwrap();
        assert_eq!(back, cfg);
    }

    #[test]
    fn an_unknown_config_key_is_refused_by_name() {
        let text = minimal_text().replace("\"output\"", "\"refinment\": {}, \"output\"");
        let err = read_config_str(&text, "typo").unwrap_err();
        assert!(err.to_string().contains("refinment"), "{err}");
    }

    #[test]
    fn serde_defaults_match_the_default_impls() {
        let cfg = read_config_str(&minimal_text(), "defaults").unwrap();
        assert_eq!(cfg.refinement, RefinementSpec::default());
        assert_eq!(cfg.castellation, CastellationSpec::default());
        assert_eq!(cfg.snap, SnapSpec::default());
        assert_eq!(cfg.layers, LayerSpec::default());
        assert_eq!(cfg.quality, QualitySpec::default());
        assert_eq!(cfg.domain.grading, grading_one());
    }

    #[test]
    fn validate_refuses_a_bad_extent_and_a_deep_max_level() {
        let mut cfg = minimal();
        cfg.domain.extent = [10.0, 0.0, 0.0, 10.0, 0.0, 10.0];
        let err = cfg.validate().unwrap_err();
        assert!(err.to_string().contains("extent"), "{err}");

        let mut cfg = minimal();
        cfg.refinement.max_level = 7;
        let err = cfg.validate().unwrap_err();
        assert!(err.to_string().contains("max_level"), "{err}");
    }
}
