// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.
// Provenance: see PROVENANCE.md. No GPL-licensed source was consulted.

//! The native tetrahedral mesher of SPEC-LIT §116.
//!
//! Precision (§116.2): every coordinate inside this module is [`Point`] =
//! `[f64; 3]`, and every size and length is `f64`, whatever `Scalar` is.
//! The exact predicates of §116.4 carry f64 error bounds, so the
//! tetrahedral path never runs its geometry in f32; only the emitter of
//! §116.16 turns a point into `Scalar`, and it re-checks the rounded cells
//! when it does.
//!
//! [`TetSpec`] is the automesher config's `tet` block (§116.3): the five
//! blocks the tet path adds, every field defaulted, so `"tet": {}` is a
//! complete block. This unit ships the config types only. Later units add
//! these submodules to the module:
//!
//! - `predicates` (§116.4), the exact orientation and insphere predicates;
//! - `perturb` (§116.5), the symbolic perturbation;
//! - `delaunay` (§116.6), the Delaunay kernel;
//! - `order` (§116.7), the insertion order;
//! - `sizefield` (§116.8), the size field h(x);
//! - `surfprep` (§116.9), surface preparation;
//! - `remesh` (§116.10), isotropic surface remeshing;
//! - `recover` (§116.11), conforming boundary recovery;
//! - `carve` (§116.12), the flood-fill carve;
//! - `refine` (§116.13), Delaunay refinement;
//! - `optimise` (§116.14), flips, smoothing, sliver perturbation;
//! - `inflate` (§116.15), prism layers by post-inflation;
//! - `emit` (§116.16), emission, the gate and the f32 rounding check.
//!
//! Provenance: ORIGINAL. No GPL-licensed source was consulted.

use crate::error::Error;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

/// A vertex coordinate: `[f64; 3]` in both builds, whatever `Scalar` is
/// (§116.2). Sizes and lengths are `f64` too; no `tetmesh` function takes
/// or returns `Scalar`, except the emitter of §116.16.
pub type Point = [f64; 3];

/// `tet.size` - SPEC-LIT §116.3: how the size field of §116.8 chooses edge
/// lengths, and the floor under them.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct TetSizeSpec {
    /// Size growth per metre of distance from the surfaces, `g` of (116.7):
    /// the size grows by at most `g - 1` per metre. `1` is a uniform size
    /// and is legal.
    #[serde(default = "d_gradation")]
    pub gradation: f64,
    /// Cells per full turn of the surface's curvature, `n_curv` of (116.9):
    /// `h <= 2 pi / (n_curv kappa)`, so a sphere of radius `R` gets
    /// `h <= 2 pi R / n_curv`. `0` turns the curvature source off.
    #[serde(default = "d_curvature_cells")]
    pub curvature_cells: u32,
    /// Cells across the narrowest gap between surfaces, `n_gap` of (116.10).
    /// `0` turns the proximity source off.
    #[serde(default = "d_gap_cells")]
    pub gap_cells: u32,
    /// Absolute floor on the size, the `h_min` of (116.12), metres. Absent
    /// means `H / 64` (§116.8).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub min_size: Option<f64>,
}

fn d_gradation() -> f64 {
    1.2
}

fn d_curvature_cells() -> u32 {
    16
}

fn d_gap_cells() -> u32 {
    3
}

impl Default for TetSizeSpec {
    fn default() -> Self {
        Self {
            gradation: d_gradation(),
            curvature_cells: d_curvature_cells(),
            gap_cells: d_gap_cells(),
            min_size: None,
        }
    }
}

/// `tet.recover` - SPEC-LIT §116.3: the knobs of the conforming boundary
/// recovery of §116.11.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct TetRecoverSpec {
    /// Input angles narrower than this, degrees, get a protecting ball
    /// (§116.11).
    #[serde(default = "d_protect_angle_deg")]
    pub protect_angle_deg: f64,
    /// The cap on recovery passes. It is reported, never silently passed
    /// (§116.11).
    #[serde(default = "d_max_passes")]
    pub max_passes: usize,
}

fn d_protect_angle_deg() -> f64 {
    90.0
}

fn d_max_passes() -> usize {
    64
}

impl Default for TetRecoverSpec {
    fn default() -> Self {
        Self {
            protect_angle_deg: d_protect_angle_deg(),
            max_passes: d_max_passes(),
        }
    }
}

/// `tet.refine` - SPEC-LIT §116.3: the knobs of the Delaunay refinement of
/// §116.13.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct TetRefineSpec {
    /// The radius-edge bound `B` of (116.17); two or more carries Shewchuk
    /// 1998's termination guarantee.
    #[serde(default = "d_radius_edge")]
    pub radius_edge: f64,
    /// The size factor `c` of (116.18): the circumradius of the regular
    /// tetrahedron with edge `h` is `c h`.
    #[serde(default = "d_size_factor")]
    pub size_factor: f64,
    /// The cap on refinement insertions. It is reported, never silently
    /// passed (§116.13).
    #[serde(default = "d_max_inserted_points")]
    pub max_inserted_points: u64,
}

fn d_radius_edge() -> f64 {
    2.0
}

fn d_size_factor() -> f64 {
    6f64.sqrt() / 4.0
}

fn d_max_inserted_points() -> u64 {
    50_000_000
}

impl Default for TetRefineSpec {
    fn default() -> Self {
        Self {
            radius_edge: d_radius_edge(),
            size_factor: d_size_factor(),
            max_inserted_points: d_max_inserted_points(),
        }
    }
}

/// `tet.optimise` - SPEC-LIT §116.3: the knobs of the optimisation of
/// §116.14 - flips, smoothing, sliver perturbation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct TetOptimiseSpec {
    /// The number of flip/smooth alternations (§116.14).
    #[serde(default = "d_passes")]
    pub passes: usize,
    /// The optimiser's goal for the smallest dihedral angle, degrees. A
    /// target, not a gate: the gates are fixed in SPEC-LIT §116.18.
    #[serde(default = "d_target_min_dihedral_deg")]
    pub target_min_dihedral_deg: f64,
    /// The optimiser's goal for the largest dihedral angle, degrees. A
    /// target, not a gate.
    #[serde(default = "d_target_max_dihedral_deg")]
    pub target_max_dihedral_deg: f64,
    /// Run the sliver perturbation of §116.14.
    #[serde(default = "d_perturb_slivers")]
    pub perturb_slivers: bool,
    /// The seed of that perturbation's deterministic generator (§116.14).
    #[serde(default = "d_perturb_seed")]
    pub perturb_seed: u64,
}

fn d_passes() -> usize {
    8
}

fn d_target_min_dihedral_deg() -> f64 {
    10.0
}

fn d_target_max_dihedral_deg() -> f64 {
    165.0
}

fn d_perturb_slivers() -> bool {
    true
}

fn d_perturb_seed() -> u64 {
    1
}

impl Default for TetOptimiseSpec {
    fn default() -> Self {
        Self {
            passes: d_passes(),
            target_min_dihedral_deg: d_target_min_dihedral_deg(),
            target_max_dihedral_deg: d_target_max_dihedral_deg(),
            perturb_slivers: d_perturb_slivers(),
            perturb_seed: d_perturb_seed(),
        }
    }
}

/// `tet.layers` - SPEC-LIT §116.3: the placement knobs of the prism layers
/// grown by post-inflation (§116.15).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct TetLayerSpec {
    /// The largest ratio of total layer thickness between two wall-adjacent
    /// wall points, `T(p) / T(q)` of (116.24c).
    #[serde(default = "d_max_neighbour_ratio")]
    pub max_neighbour_ratio: f64,
    /// The fraction of the local gap `gap(p)` the total layer thickness may
    /// take, (116.24a).
    #[serde(default = "d_gap_fraction")]
    pub gap_fraction: f64,
    /// The fraction of a concave wall's radius of curvature `rho_c(p)` the
    /// total layer thickness may take, (116.24b).
    #[serde(default = "d_concave_fraction")]
    pub concave_fraction: f64,
}

fn d_max_neighbour_ratio() -> f64 {
    1.5
}

fn d_gap_fraction() -> f64 {
    0.5
}

fn d_concave_fraction() -> f64 {
    0.5
}

impl Default for TetLayerSpec {
    fn default() -> Self {
        Self {
            max_neighbour_ratio: d_max_neighbour_ratio(),
            gap_fraction: d_gap_fraction(),
            concave_fraction: d_concave_fraction(),
        }
    }
}

/// `tet` - the tet path's own config, SPEC-LIT §116.3: the five blocks the
/// tet path adds to the automesher config. Everything else the pipeline
/// needs - `input`, `domain`, `refinement`, `castellation`'s `keep_region`,
/// the layer patch list, `quality`, `output` - is shared with the
/// hex-dominant path. Every field defaults, so `"tet": {}` is complete.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct TetSpec {
    /// Size control: the knobs of §116.8.
    #[serde(default)]
    pub size: TetSizeSpec,
    /// Boundary recovery: §116.11.
    #[serde(default)]
    pub recover: TetRecoverSpec,
    /// Delaunay refinement: §116.13.
    #[serde(default)]
    pub refine: TetRefineSpec,
    /// Optimisation: §116.14.
    #[serde(default)]
    pub optimise: TetOptimiseSpec,
    /// Prism layers by post-inflation: §116.15.
    #[serde(default)]
    pub layers: TetLayerSpec,
}

impl Default for TetSpec {
    fn default() -> Self {
        Self {
            size: TetSizeSpec::default(),
            recover: TetRecoverSpec::default(),
            refine: TetRefineSpec::default(),
            optimise: TetOptimiseSpec::default(),
            layers: TetLayerSpec::default(),
        }
    }
}

impl TetSpec {
    /// Checks §116.3's refusal list and returns the first failure, naming
    /// the field and the value. Every comparison is written so a NaN fails
    /// each check it can reach.
    pub fn validate(&self) -> crate::error::Result<()> {
        if !(self.size.gradation >= 1.0) {
            return Err(Error::Mesh(format!(
                "tet.size.gradation: must be >= 1, got {}",
                self.size.gradation
            )));
        }
        if let Some(v) = self.size.min_size {
            if !(v.is_finite() && v > 0.0) {
                return Err(Error::Mesh(format!(
                    "tet.size.min_size: must be finite and > 0, got {}",
                    v
                )));
            }
        }
        if !(self.recover.protect_angle_deg > 0.0
            && self.recover.protect_angle_deg < 180.0)
        {
            return Err(Error::Mesh(format!(
                "tet.recover.protect_angle_deg: must be in (0, 180), got {}",
                self.recover.protect_angle_deg
            )));
        }
        if self.recover.max_passes == 0 {
            return Err(Error::Mesh(
                "tet.recover.max_passes: must be >= 1, got 0".to_string(),
            ));
        }

        if !(self.refine.radius_edge >= 2.0) {
            return Err(Error::Mesh(format!(
                "tet.refine.radius_edge: must be >= 2 (Shewchuk 1998's termination bound, SPEC-LIT §116.13), got {}",
                self.refine.radius_edge
            )));
        }
        if !(self.refine.size_factor.is_finite() && self.refine.size_factor > 0.0) {
            return Err(Error::Mesh(format!(
                "tet.refine.size_factor: must be finite and > 0, got {}",
                self.refine.size_factor
            )));
        }
        if self.refine.max_inserted_points == 0 {
            return Err(Error::Mesh(
                "tet.refine.max_inserted_points: must be >= 1, got 0".to_string(),
            ));
        }
        if self.optimise.passes == 0 {
            return Err(Error::Mesh(
                "tet.optimise.passes: must be >= 1, got 0".to_string(),
            ));
        }
        let a = self.optimise.target_min_dihedral_deg;
        let b = self.optimise.target_max_dihedral_deg;
        if !(0.0 < a && a < b && b < 180.0) {
            return Err(Error::Mesh(format!(
                "tet.optimise: dihedral targets must satisfy 0 < min < max < 180, got min {}, max {}",
                a, b
            )));
        }

        if !(self.layers.max_neighbour_ratio >= 1.0) {
            return Err(Error::Mesh(format!(
                "tet.layers.max_neighbour_ratio: must be >= 1, got {}",
                self.layers.max_neighbour_ratio
            )));
        }
        if !(self.layers.gap_fraction > 0.0 && self.layers.gap_fraction <= 1.0) {
            return Err(Error::Mesh(format!(
                "tet.layers.gap_fraction: must be in (0, 1], got {}",
                self.layers.gap_fraction
            )));
        }
        if !(self.layers.concave_fraction > 0.0
            && self.layers.concave_fraction <= 1.0)
        {
            return Err(Error::Mesh(format!(
                "tet.layers.concave_fraction: must be in (0, 1], got {}",
                self.layers.concave_fraction
            )));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spec_round_trips_json() {
        let empty: TetSpec = serde_json::from_str("{}").unwrap();
        assert_eq!(empty, TetSpec::default());

        let value = serde_json::to_value(TetSpec::default()).unwrap();
        assert_eq!(
            value,
            serde_json::json!({
                "size": {"gradation": 1.2, "curvature_cells": 16, "gap_cells": 3},
                "recover": {"protect_angle_deg": 90.0, "max_passes": 64},
                "refine": {
                    "radius_edge": 2.0,
                    "size_factor": 0.6123724356957945,
                    "max_inserted_points": 50000000
                },
                "optimise": {
                    "passes": 8,
                    "target_min_dihedral_deg": 10.0,
                    "target_max_dihedral_deg": 165.0,
                    "perturb_slivers": true,
                    "perturb_seed": 1
                },
                "layers": {
                    "max_neighbour_ratio": 1.5,
                    "gap_fraction": 0.5,
                    "concave_fraction": 0.5
                }
            })
        );

        let full = TetSpec {
            size: TetSizeSpec {
                gradation: 1.3,
                curvature_cells: 24,
                gap_cells: 4,
                min_size: Some(0.002),
            },
            recover: TetRecoverSpec { protect_angle_deg: 75.0, max_passes: 32 },
            refine: TetRefineSpec {
                radius_edge: 2.5,
                size_factor: 0.7,
                max_inserted_points: 1_000_000,
            },
            optimise: TetOptimiseSpec {
                passes: 4,
                target_min_dihedral_deg: 12.0,
                target_max_dihedral_deg: 160.0,
                perturb_slivers: false,
                perturb_seed: 42,
            },
            layers: TetLayerSpec {
                max_neighbour_ratio: 1.3,
                gap_fraction: 0.4,
                concave_fraction: 0.25,
            },
        };
        let text = serde_json::to_string(&full).unwrap();
        let back: TetSpec = serde_json::from_str(&text).unwrap();
        assert_eq!(back, full);
        let full_value = serde_json::to_value(&full).unwrap();
        assert_eq!(full_value["size"]["min_size"], serde_json::json!(0.002));
        assert!(full.validate().is_ok());

        let partial: TetSpec =
            serde_json::from_value(serde_json::json!({"size": {"gradation": 1.5}}))
                .unwrap();
        let d = TetSpec::default();
        assert_eq!(partial.size.gradation, 1.5);
        assert_eq!(partial.size.curvature_cells, d.size.curvature_cells);
        assert_eq!(partial.size.gap_cells, d.size.gap_cells);
        assert_eq!(partial.size.min_size, None);
        assert_eq!(partial.recover, d.recover);
        assert_eq!(partial.refine, d.refine);
        assert_eq!(partial.optimise, d.optimise);
        assert_eq!(partial.layers, d.layers);
    }

    #[test]
    fn spec_defaults_are_section_116_values() {
        let d = TetSpec::default();
        assert_eq!(d.size.gradation, 1.2);
        assert_eq!(d.size.curvature_cells, 16);
        assert_eq!(d.size.gap_cells, 3);
        assert_eq!(d.size.min_size, None);
        assert_eq!(d.recover.protect_angle_deg, 90.0);
        assert_eq!(d.recover.max_passes, 64);
        assert_eq!(d.refine.radius_edge, 2.0);
        assert_eq!(d.refine.size_factor, 0.6123724356957945);
        assert_eq!(d.refine.size_factor, 6f64.sqrt() / 4.0);
        assert_eq!(d.refine.max_inserted_points, 50_000_000);
        assert_eq!(d.optimise.passes, 8);
        assert_eq!(d.optimise.target_min_dihedral_deg, 10.0);
        assert_eq!(d.optimise.target_max_dihedral_deg, 165.0);
        assert!(d.optimise.perturb_slivers);
        assert_eq!(d.optimise.perturb_seed, 1);
        assert_eq!(d.layers.max_neighbour_ratio, 1.5);
        assert_eq!(d.layers.gap_fraction, 0.5);
        assert_eq!(d.layers.concave_fraction, 0.5);
        assert!(d.validate().is_ok());
    }

    #[test]
    fn spec_refuses_unknown_keys() {
        let refused = |json: serde_json::Value, key: &str| {
            let err = serde_json::from_value::<TetSpec>(json).unwrap_err();
            let msg = err.to_string();
            assert!(
                msg.contains(&format!("unknown field `{}`", key)),
                "not refused by name: {}",
                msg
            );
        };
        refused(serde_json::json!({"sise": {}}), "sise");
        refused(serde_json::json!({"size": {"gradatoin": 1.3}}), "gradatoin");
        refused(serde_json::json!({"layers": {"n": 5}}), "n");
    }

    #[test]
    fn spec_validate_refusals() {
        fn refused(spec: &TetSpec, msg: &str) {
            assert_eq!(spec.validate().unwrap_err().to_string(), msg);
        }

        {
            let mut s = TetSpec::default();
            s.size.gradation = 0.9;
            refused(&s, "tet.size.gradation: must be >= 1, got 0.9");
        }
        {
            let mut s = TetSpec::default();
            s.size.gradation = f64::NAN;
            refused(&s, "tet.size.gradation: must be >= 1, got NaN");
        }
        {
            let mut s = TetSpec::default();
            s.size.min_size = Some(0.0);
            refused(&s, "tet.size.min_size: must be finite and > 0, got 0");
        }
        {
            let mut s = TetSpec::default();
            s.size.min_size = Some(f64::INFINITY);
            refused(&s, "tet.size.min_size: must be finite and > 0, got inf");
        }
        {
            let mut s = TetSpec::default();
            s.recover.protect_angle_deg = 0.0;
            refused(&s, "tet.recover.protect_angle_deg: must be in (0, 180), got 0");
        }
        {
            let mut s = TetSpec::default();
            s.recover.protect_angle_deg = 180.0;
            refused(&s, "tet.recover.protect_angle_deg: must be in (0, 180), got 180");
        }
        {
            let mut s = TetSpec::default();
            s.recover.max_passes = 0;
            refused(&s, "tet.recover.max_passes: must be >= 1, got 0");
        }

        {
            let mut s = TetSpec::default();
            s.refine.radius_edge = 1.9;
            refused(
                &s,
                "tet.refine.radius_edge: must be >= 2 (Shewchuk 1998's termination bound, SPEC-LIT §116.13), got 1.9",
            );
        }
        {
            let mut s = TetSpec::default();
            s.refine.size_factor = -1.0;
            refused(&s, "tet.refine.size_factor: must be finite and > 0, got -1");
        }
        {
            let mut s = TetSpec::default();
            s.refine.max_inserted_points = 0;
            refused(&s, "tet.refine.max_inserted_points: must be >= 1, got 0");
        }
        {
            let mut s = TetSpec::default();
            s.optimise.passes = 0;
            refused(&s, "tet.optimise.passes: must be >= 1, got 0");
        }
        {
            let mut s = TetSpec::default();
            s.optimise.target_min_dihedral_deg = 20.0;
            s.optimise.target_max_dihedral_deg = 15.0;
            refused(
                &s,
                "tet.optimise: dihedral targets must satisfy 0 < min < max < 180, got min 20, max 15",
            );
        }
        {
            let mut s = TetSpec::default();
            s.optimise.target_min_dihedral_deg = 0.0;
            refused(
                &s,
                "tet.optimise: dihedral targets must satisfy 0 < min < max < 180, got min 0, max 165",
            );
        }

        {
            let mut s = TetSpec::default();
            s.optimise.target_max_dihedral_deg = 180.0;
            refused(
                &s,
                "tet.optimise: dihedral targets must satisfy 0 < min < max < 180, got min 10, max 180",
            );
        }
        {
            let mut s = TetSpec::default();
            s.layers.max_neighbour_ratio = 0.99;
            refused(&s, "tet.layers.max_neighbour_ratio: must be >= 1, got 0.99");
        }
        {
            let mut s = TetSpec::default();
            s.layers.gap_fraction = 0.0;
            refused(&s, "tet.layers.gap_fraction: must be in (0, 1], got 0");
        }
        {
            let mut s = TetSpec::default();
            s.layers.gap_fraction = 1.5;
            refused(&s, "tet.layers.gap_fraction: must be in (0, 1], got 1.5");
        }
        {
            let mut s = TetSpec::default();
            s.layers.concave_fraction = f64::NAN;
            refused(&s, "tet.layers.concave_fraction: must be in (0, 1], got NaN");
        }

        let mut s = TetSpec::default();
        s.size.gradation = 1.0;
        assert!(s.validate().is_ok());
        let mut s = TetSpec::default();
        s.refine.radius_edge = 2.0;
        assert!(s.validate().is_ok());
        let mut s = TetSpec::default();
        s.layers.gap_fraction = 1.0;
        assert!(s.validate().is_ok());
        let mut s = TetSpec::default();
        s.layers.max_neighbour_ratio = 1.0;
        assert!(s.validate().is_ok());
    }

    #[test]
    fn geometry_is_f64_in_both_builds() {
        assert_eq!(std::mem::size_of::<Point>(), 24);
        let p: Point = [0.1, 0.2, 0.3];
        assert_eq!(p[0].to_bits(), 0.1f64.to_bits());
    }

    #[test]
    fn spec_schema_names_every_key() {
        let schema = serde_json::to_string(&schemars::schema_for!(TetSpec)).unwrap();
        let names = [
            "gradation",
            "curvature_cells",
            "gap_cells",
            "min_size",
            "protect_angle_deg",
            "max_passes",
            "radius_edge",
            "size_factor",
            "max_inserted_points",
            "passes",
            "target_min_dihedral_deg",
            "target_max_dihedral_deg",
            "perturb_slivers",
            "perturb_seed",
            "max_neighbour_ratio",
            "gap_fraction",
            "concave_fraction",
        ];
        assert_eq!(names.len(), 17);
        for name in names {
            let quoted = format!("{}{}{}", '"', name, '"');
            assert!(schema.contains(&quoted), "schema lacks {}", name);
        }
    }
}
