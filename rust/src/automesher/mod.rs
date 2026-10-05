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

pub mod castellate;
pub mod driver;
pub mod features;
pub mod identity;
pub mod layers;
pub mod octree;
pub mod quality;
pub mod snap;

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::error::Error;
use crate::io::polymesh::check_patch_name;

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
    /// SPEC-LIT §92.16 (92.67): axis-aligned boxes; every leaf whose interior
    /// overlaps a box is refined to at least that box's level. Empty, the
    /// default, is no box refinement and is not serialised.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub boxes: Vec<RefinementBox>,
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
    /// (92.37): a leaf within its own longest edge of one of this patch's
    /// feature edges (92.34) is refined to this level. Zero, the default, is
    /// no feature refinement.
    #[serde(default)]
    pub feature_level: u32,
}

/// One distance band of [`RefinementBand`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct DistanceBand {
    pub distance: f64,
    pub level: u32,
}

/// One entry of `refinement.boxes` - an axis-aligned box, SPEC-LIT §92.16.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct RefinementBox {
    /// `[x, y, z]` lower corner, metres.
    pub min: [f64; 3],
    /// `[x, y, z]` upper corner, metres.
    pub max: [f64; 3],
    /// The level every overlapped leaf is refined to at least (capped at `max_level`).
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
            boxes: Vec::new(),
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
    /// Closed bodies kept as regions of their own. Empty (the default) is
    /// today's castellation exactly: every solid leaf is removed.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub bodies: Vec<BodySpec>,
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

/// [`LayerSpec::terminate`]'s two choices - SPEC-LIT §92.13 (92.73): where a
/// layer stack ends when a point cannot carry it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema, Default)]
#[serde(rename_all = "lowercase")]
pub enum LayerTerminate {
    /// The default: the stack ends per PATCH - a point that cannot carry the
    /// stack costs its whole patch its layers, and the snapped mesh comes
    /// back for it.
    #[default]
    Patch,
    /// The stack ends per FACE: the OUTER ladder moves no point - a
    /// G5-failing KEEP stack merges into one cell of the whole thickness,
    /// every other failing layer face is cut, leaving a step the wall
    /// carries - and the INNER ladder still anchors, tapering the faces
    /// around an anchored point to zero in one wedge cell; only the faces
    /// whose points are all anchored, or that were cut, go without layers.
    Face,
}

/// One `castellation.bodies[]` entry: a closed body the run KEEPS as a region
/// of its own instead of removing it as solid (SPEC-LIT §92.10 (92.23) removes
/// every leaf whose centre is inside the surface; a declared body is the
/// exception). `patches` are the STL solid names that together form ONE
/// closed shell; `name` is the region's name in the layout (docs/10 §C,
/// SPEC-LIT §97) and the prefix of its interface patches `<name>_to_fluid`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct BodySpec {
    pub name: String,
    pub patches: Vec<String>,
}

fn d_min_faces() -> usize {
    4
}

impl Default for CastellationSpec {
    fn default() -> Self {
        Self {
            keep_region: KeepRegion::default(),
            seed_point: None,
            min_faces: d_min_faces(),
            bodies: Vec::new(),
        }
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
    /// The dead band and the convergence test of SPEC-LIT §92.11 (92.28),
    /// as a FRACTION of `domain.base_size`: a point within
    /// `tolerance * base_size` of the surface is on it, and the loop stops
    /// when no point moves further than that.
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
    /// (92.32): a wall patch carrying more than this many times its own
    /// surface area is geometry the cells never resolved, and snapping it
    /// would collapse the one cell that reached it - refused instead.
    #[serde(default = "d_max_area_ratio")]
    pub max_area_ratio: f64,
    /// (92.38): a boundary point whose surface target lies within
    /// `feature_tolerance * base_size` of a feature edge is snapped onto the
    /// edge instead, and onto a corner when it claims one (92.39). Zero
    /// turns the attraction off.
    #[serde(default = "d_feature_tolerance")]
    pub feature_tolerance: f64,
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

fn d_max_area_ratio() -> f64 {
    4.0
}

fn d_feature_tolerance() -> f64 {
    0.5
}

impl Default for SnapSpec {
    fn default() -> Self {
        Self {
            iterations: d_snap_iters(),
            tolerance: d_snap_tol(),
            smoothing_passes: d_smoothing_passes(),
            smoothing: d_smoothing(),
            undo_limit: d_undo_limit(),
            max_area_ratio: d_max_area_ratio(),
            feature_tolerance: d_feature_tolerance(),
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
    /// The fraction of the medial distance `m(i)` the stack may take, eq.
    /// (92.10)/(92.45).
    #[serde(default = "d_medial_frac")]
    pub medial_frac: f64,
    /// (92.41): smoothing passes over the point normals. Zero leaves them
    /// the area-weighted average of (92.40), which is what a flat wall
    /// needs to stay exact.
    #[serde(default = "d_normal_passes")]
    pub normal_passes: usize,
    /// (92.45): the fraction of the LOCAL CELL SIZE the stack may take.
    #[serde(default = "d_cell_frac")]
    pub cell_frac: f64,
    /// (92.41) and (92.46)'s `w`.
    #[serde(default = "d_layer_smoothing")]
    pub smoothing: f64,
    /// (92.46)'s passes into the interior.
    #[serde(default = "d_layer_smoothing_passes")]
    pub smoothing_passes: usize,
    /// (92.47): how many times the thickness is halved before the patch
    /// loses its layers.
    #[serde(default = "d_retreat_limit")]
    pub retreat_limit: usize,
    /// (92.73): where a layer stack ends - `"patch"` drops a whole patch
    /// when one of its points cannot carry the stack; `"face"` anchors the
    /// point and tapers the faces around it to zero in one wedge cell.
    #[serde(default, skip_serializing_if = "terminate_is_patch")]
    #[schemars(default)]
    pub terminate: LayerTerminate,
    /// (92.69): face mode's angle tolerance the junction normals of a point
    /// merge within, degrees. Must lie in (0, 90).
    #[serde(
        default = "d_junction_angle_deg",
        skip_serializing_if = "junction_angle_is_default"
    )]
    #[schemars(default = "d_junction_angle_deg")]
    pub junction_angle_deg: f64,
}

/// [`LayerSpec::terminate`]'s skip predicate: `"patch"` is the default and
/// is not written out.
fn terminate_is_patch(t: &LayerTerminate) -> bool {
    *t == LayerTerminate::Patch
}

/// [`LayerSpec::junction_angle_deg`]'s skip predicate.
fn junction_angle_is_default(v: &f64) -> bool {
    *v == d_junction_angle_deg()
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

fn d_normal_passes() -> usize {
    3
}

fn d_cell_frac() -> f64 {
    0.5
}

fn d_layer_smoothing() -> f64 {
    0.5
}

fn d_layer_smoothing_passes() -> usize {
    4
}

fn d_retreat_limit() -> usize {
    4
}

fn d_junction_angle_deg() -> f64 {
    15.0
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
            normal_passes: d_normal_passes(),
            cell_frac: d_cell_frac(),
            smoothing: d_layer_smoothing(),
            smoothing_passes: d_layer_smoothing_passes(),
            retreat_limit: d_retreat_limit(),
            terminate: LayerTerminate::Patch,
            junction_angle_deg: d_junction_angle_deg(),
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
    /// (92.56): rename the final mesh's patches on the way out — the octree's
    /// `xMin`..`zMax` and the STL's own solid names become the names the case's
    /// boundary conditions are written against. Absent keys leave a patch alone.
    #[serde(default, skip_serializing_if = "std::collections::BTreeMap::is_empty")]
    pub patch_names: std::collections::BTreeMap<String, String>,
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
        if !(self.snap.max_area_ratio >= 1.0) {
            return Err(Error::Mesh(format!(
                "snap.max_area_ratio: must be >= 1, got {}",
                self.snap.max_area_ratio
            )));
        }
        if !(self.snap.smoothing >= 0.0 && self.snap.smoothing <= 1.0) {
            return Err(Error::Mesh(format!(
                "snap.smoothing: must lie in [0, 1], got {}",
                self.snap.smoothing
            )));
        }
        if !(self.snap.feature_tolerance.is_finite() && self.snap.feature_tolerance >= 0.0) {
            return Err(Error::Mesh(format!(
                "snap.feature_tolerance: must be >= 0 and finite, got {}",
                self.snap.feature_tolerance
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
        // (92.69): the merge tolerance is an angle, not a fraction - 0 would
        // merge nothing and 90 would merge everything, and neither is what
        // the face mode's junction dedupe means.
        if !(self.layers.junction_angle_deg > 0.0
            && self.layers.junction_angle_deg < 90.0)
        {
            return Err(Error::Mesh(format!(
                "layers.junction_angle_deg: must lie in (0, 90), got {}",
                self.layers.junction_angle_deg
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
        // The declared bodies of `castellation.bodies`: every refusal names
        // the field path and the name that broke the rule, before any
        // meshing work.
        let box_names = octree::patch_names();
        let mut seen: Vec<&str> = Vec::new();
        for (i, b) in self.castellation.bodies.iter().enumerate() {
            check_patch_name(&b.name, "body name").map_err(|e| {
                Error::Mesh(format!("castellation.bodies[{i}].name: {e}"))
            })?;
            if b.name == "fluid" {
                return Err(Error::Mesh(format!(
                    "castellation.bodies[{i}].name: \"fluid\" is the fluid region's \
                     own name - a body cannot take it"
                )));
            }
            if box_names.iter().any(|q| q == &b.name) {
                return Err(Error::Mesh(format!(
                    "castellation.bodies[{i}].name: \"{}\" is a domain patch name - \
                     the six box patches are {:?}",
                    b.name, box_names
                )));
            }
            if seen.contains(&b.name.as_str()) {
                return Err(Error::Mesh(format!(
                    "castellation.bodies[{i}].name: \"{}\" is declared twice",
                    b.name
                )));
            }
            seen.push(&b.name);
            if b.patches.is_empty() {
                return Err(Error::Mesh(format!(
                    "castellation.bodies[{i}].patches: empty - a body is a set of \
                     STL patch names, got none"
                )));
            }
            for (j, other) in self.castellation.bodies.iter().enumerate() {
                if j >= i {
                    break;
                }
                for p in &b.patches {
                    if other.patches.iter().any(|q| q == p) {
                        return Err(Error::Mesh(format!(
                            "castellation.bodies[{i}].patches: \"{}\" is also a patch \
                             of body \"{}\" - a patch belongs to one body",
                            p, other.name
                        )));
                    }
                }
            }
        }
        // The declared boxes of `refinement.boxes` (§92.16): the same refusal
        // style - the field path, the index and the value that broke the rule,
        // before any meshing work. A level above `max_level` is NOT refused -
        // it is capped, exactly as a band level is (92.1's min(..., max_level)).
        let e = &self.domain.extent;
        for (i, b) in self.refinement.boxes.iter().enumerate() {
            if !b.min.iter().chain(b.max.iter()).all(|v| v.is_finite()) {
                return Err(Error::Mesh(format!(
                    "refinement.boxes[{i}]: every coordinate must be finite, \
                     got min {:?} max {:?}",
                    b.min, b.max
                )));
            }
            for (a, name) in [(0usize, "x"), (1, "y"), (2, "z")] {
                if !(b.max[a] > b.min[a]) {
                    return Err(Error::Mesh(format!(
                        "refinement.boxes[{i}]: {name}-axis is empty or reversed \
                         (min = {}, max = {})",
                        b.min[a], b.max[a]
                    )));
                }
            }
            if b.level == 0 {
                return Err(Error::Mesh(format!(
                    "refinement.boxes[{i}].level: 0 refines nothing - a box asks \
                     for level >= 1"
                )));
            }
            if !b
                .min
                .iter()
                .zip(b.max.iter())
                .enumerate()
                .all(|(a, (lo, hi))| *lo < e[2 * a + 1] && *hi > e[2 * a])
            {
                return Err(Error::Mesh(format!(
                    "refinement.boxes[{i}]: lies outside domain.extent - a box \
                     that overlaps no cell refines nothing"
                )));
            }
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
            output: OutputSpec {
                case_dir: "out".to_string(),
                name: "site".to_string(),
                patch_names: std::collections::BTreeMap::new(),
            },
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
                feature_level: 0,
            }],
            feature_angle_deg: 45.0,
            max_level: 3,
            boxes: vec![RefinementBox { min: [1.0; 3], max: [2.0; 3], level: 1 }],
        };
        cfg.castellation = CastellationSpec {
            keep_region: KeepRegion::Seed,
            seed_point: Some([1.0, 1.0, 1.0]),
            min_faces: 6,
            bodies: vec![BodySpec {
                name: "hull".to_string(),
                patches: vec!["hull_deck".to_string(), "hull_keel".to_string()],
            }],
        };
        cfg.snap.iterations = 10;
        cfg.snap.tolerance = 1e-4;
        cfg.snap.smoothing_passes = 5;
        cfg.snap.smoothing = 0.25;
        cfg.snap.undo_limit = 2;
        cfg.snap.max_area_ratio = 2.5;
        cfg.layers = LayerSpec {
            patches: vec!["ground".to_string()],
            n: 3,
            first_thickness: 0.01,
            growth: 1.2,
            min_thickness: 0.05,
            medial_frac: 0.4,
            normal_passes: 2,
            cell_frac: 0.45,
            smoothing: 0.3,
            smoothing_passes: 5,
            retreat_limit: 3,
            terminate: LayerTerminate::Patch,
            junction_angle_deg: d_junction_angle_deg(),
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

    /// (92.69) and (92.73)'s two keys: they parse, they refuse by name, and
    /// a default config serialises without either.
    #[test]
    fn layers_terminate_and_junction_angle_parse_and_refuse() {
        let spec = |body: &str| {
            let text = minimal_text().replace(
                "\"output\"",
                &format!("\"layers\": {{{body}}}, \"output\""),
            );
            read_config_str(&text, "layers")
        };
        let cfg = spec("\"terminate\": \"face\"").expect("face parses");
        assert_eq!(cfg.layers.terminate, LayerTerminate::Face);
        let err = spec("\"terminate\": \"edge\"").unwrap_err();
        assert!(err.to_string().contains("edge"), "{err}");
        for bad in [0.0, 90.0] {
            let mut cfg = spec("\"junction_angle_deg\": 30.0").expect("30 in range");
            cfg.layers.junction_angle_deg = bad;
            let err = cfg.validate().unwrap_err();
            assert!(
                err.to_string().contains("layers.junction_angle_deg"),
                "{err}"
            );
        }
        let cfg = spec("\"junction_angle_deg\": 30.0").expect("30 parses");
        assert_eq!(cfg.layers.junction_angle_deg, 30.0);
        // A default LayerSpec serialises with neither key.
        let json = serde_json::to_string(&LayerSpec::default()).unwrap();
        assert!(!json.contains("terminate"), "{json}");
        assert!(!json.contains("junction_angle_deg"), "{json}");
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

    /// M6 Run 1: each rule a body can break is refused by the field path and
    /// the name that broke it. A body whose name equals one of its own patch
    /// names is LEGAL (the normal case) and is not in this list.
    #[test]
    fn a_body_that_breaks_a_rule_is_refused_by_name() {
        let body = |name: &str, patches: &[&str]| BodySpec {
            name: name.to_string(),
            patches: patches.iter().map(|p| p.to_string()).collect(),
        };
        let cases: Vec<(Vec<BodySpec>, &str)> = vec![
            (vec![body("bad name", &["hull"])], "bad name"),
            (vec![body("fluid", &["hull"])], "fluid"),
            (vec![body("xMin", &["hull"])], "xMin"),
            (vec![body("blob", &["a"]), body("blob", &["b"])], "blob"),
            (vec![body("hull", &[])], "castellation.bodies[0].patches"),
            (
                vec![body("one", &["p"]), body("two", &["p"])],
                "castellation.bodies[1].patches",
            ),
        ];
        for (bodies, needle) in cases {
            let mut cfg = minimal();
            cfg.castellation.bodies = bodies;
            let err = cfg.validate().unwrap_err();
            let text = err.to_string();
            assert!(text.contains("castellation.bodies["), "{err}");
            assert!(text.contains(needle), "{needle} not in {err}");
        }
        // The default config is the same walk as before the field existed:
        // absent from the re-serialised config, so the summary is too.
        assert!(
            !serde_json::to_string(&minimal()).unwrap().contains("\"bodies\""),
            "an empty bodies list must not serialise"
        );
        // The normal case survives: a body named after its own patch.
        let mut cfg = minimal();
        cfg.castellation.bodies = vec![body("sphere", &["sphere"])];
        cfg.validate().expect("a body may take its patch's name");
    }

    /// SPEC-LIT §92.16: `refinement.boxes` parses, round-trips, and an
    /// empty list serialises to NOTHING, so a config without boxes keeps
    /// the summary JSON it always had.
    #[test]
    fn refinement_boxes_parse_round_trip_and_stay_absent_when_empty() {
        let text = minimal_text().replace(
            "\"output\"",
            "\"refinement\": {\"boxes\": \
             [{\"min\": [1,1,1], \"max\": [2,2,2], \"level\": 1}]}, \"output\"",
        );
        let cfg = read_config_str(&text, "boxes").unwrap();
        assert_eq!(cfg.refinement.boxes.len(), 1, "{cfg:?}");
        assert_eq!(cfg.refinement.boxes[0].min, [1.0, 1.0, 1.0]);
        assert_eq!(cfg.refinement.boxes[0].max, [2.0, 2.0, 2.0]);
        assert_eq!(cfg.refinement.boxes[0].level, 1);
        // The round-trip test's config carries one box through serde.
        let mut cfg = minimal();
        cfg.refinement.boxes =
            vec![RefinementBox { min: [1.0; 3], max: [2.0; 3], level: 2 }];
        let json = serde_json::to_string(&cfg).unwrap();
        let back = read_config_str(&json, "round-trip").unwrap();
        assert_eq!(back, cfg);
        // Empty: absent from the re-serialised config, so the summary is too.
        assert!(
            !serde_json::to_string(&minimal()).unwrap().contains("\"boxes\""),
            "an empty boxes list must not serialise"
        );
        // An unknown key inside a box is refused by name.
        let text = minimal_text().replace(
            "\"output\"",
            "\"refinement\": {\"boxes\": \
             [{\"min\": [1,1,1], \"max\": [2,2,2], \"lvl\": 1}]}, \"output\"",
        );
        let err = read_config_str(&text, "typo").unwrap_err();
        assert!(err.to_string().contains("lvl"), "{err}");
    }

    /// §92.16: each rule a box can break is refused by the field path, the
    /// index and the value. A level above `max_level` is NOT in the list -
    /// it is capped, exactly as a band level is.
    #[test]
    fn a_box_that_breaks_a_rule_is_refused_by_name() {
        let box6 = |min: [f64; 3], max: [f64; 3], level: u32| RefinementBox { min, max, level };
        let cases: Vec<(RefinementBox, &str)> = vec![
            (box6([f64::NAN, 0.0, 0.0], [2.0, 2.0, 2.0], 1),
             "refinement.boxes[0]: every coordinate must be finite"),
            (box6([0.0, 5.0, 0.0], [1.0, 5.0, 1.0], 1),
             "y-axis is empty or reversed"),
            (box6([1.0, 1.0, 1.0], [2.0, 2.0, 2.0], 0), "refinement.boxes[0].level"),
            (box6([11.0, 0.0, 0.0], [12.0, 1.0, 1.0], 1),
             "lies outside domain.extent"),
            // Touching the xhi face only: xlo = 10 is not < xhi = 10.
            (box6([10.0, 0.0, 0.0], [11.0, 1.0, 1.0], 1),
             "lies outside domain.extent"),
        ];
        for (b, needle) in cases {
            let mut cfg = minimal();
            cfg.refinement.boxes = vec![b];
            let err = cfg.validate().unwrap_err();
            let text = err.to_string();
            assert!(text.contains(needle), "{needle} not in {err}");
        }
        // A second box after a good one is named by ITS index.
        let mut cfg = minimal();
        cfg.refinement.boxes = vec![
            box6([1.0, 1.0, 1.0], [2.0, 2.0, 2.0], 1),
            box6([0.0, 0.0, 0.0], [1.0, 0.0, 1.0], 1),
        ];
        let err = cfg.validate().unwrap_err();
        assert!(err.to_string().contains("refinement.boxes[1]"), "{err}");
        // A level above max_level validates: it is capped, not refused.
        let mut cfg = minimal();
        cfg.refinement.boxes = vec![box6([1.0, 1.0, 1.0], [2.0, 2.0, 2.0], 9)];
        cfg.refinement.max_level = 2;
        cfg.validate().expect("a deep level is capped, not refused");
    }
}
