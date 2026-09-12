// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.
// Provenance: see PROVENANCE.md. No GPL-licensed source was consulted.

//! The driver - SPEC-LIT §92.14: §92.9's octree, §92.10's castellation,
//! §92.11's snapping and §92.13's layers run in (92.55)'s order by one call,
//! with the stop rule, the progress lines of §92.14, (92.56)'s patch
//! rename and (92.57)'s summary. The stages themselves live in their own
//! files and gate their own output; this file owns only what belongs to no
//! stage - when a run stops, and what the patches of the written case are
//! called.
//!
//! Provenance: ORIGINAL. No GPL-licensed source was consulted.

use std::time::Instant;

use serde_json::json;

use crate::error::{Error, Result};
use crate::io::polymesh::PolyMeshRaw;
use crate::surface::Surface;
use crate::Scalar;

use super::quality::QualityReport;
use super::AutomeshConfig;

// ==========================================================================
//  The stages - (92.55)
// ==========================================================================

/// (92.55)'s stage list - the four stages that return a mesh. Stage 0 (the
/// background block) and stage 2 (the 2:1 balance) are not here: neither
/// returns a mesh of its own.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Stage {
    Octree,
    Castellate,
    Snap,
    Layers,
}

impl Stage {
    /// The name in a banner, a config and the summary.
    pub fn name(self) -> &'static str {
        match self {
            Stage::Octree => "octree",
            Stage::Castellate => "castellate",
            Stage::Snap => "snap",
            Stage::Layers => "layers",
        }
    }

    /// Parse a `-stopAfter` value. "features" parses to [`Stage::Snap`] -
    /// §92.14: the feature attraction runs INSIDE stage 4, so there is no
    /// mesh between them, and a driver that offered a separate stop would be
    /// claiming an intermediate state the mesher does not have.
    pub fn parse(s: &str) -> Result<Stage> {
        match s {
            "octree" => Ok(Stage::Octree),
            "castellate" => Ok(Stage::Castellate),
            "features" | "snap" => Ok(Stage::Snap),
            "layers" => Ok(Stage::Layers),
            _ => Err(Error::Config(format!(
                "stopAfter: unknown stage \"{s}\" - one of \
                 \"octree\", \"castellate\", \"features\", \"snap\", \"layers\""
            ))),
        }
    }

    /// The four, in order.
    pub const ALL: [Stage; 4] = [Stage::Octree, Stage::Castellate, Stage::Snap, Stage::Layers];
}

// ==========================================================================
//  The run's output - §92.14 and (92.57)'s inputs
// ==========================================================================

/// One row of (92.57)'s `stages` array.
#[derive(Debug, Clone)]
pub struct StageTiming {
    pub stage: Stage,
    pub seconds: f64,
    /// The stage's own counts, already formatted as JSON - the shapes are
    /// fixed per stage, and (92.57) splices them at the top level of the
    /// stage's object, next to `stage` and `seconds`.
    pub counts: serde_json::Value,
    /// The stage's own lines for the run log, ready to print (may be empty).
    pub log: String,
}

/// What a whole run produced.
#[derive(Debug, Clone)]
pub struct PipelineOutput {
    /// The mesh of the last stage that ran, with (92.56) already applied.
    pub mesh: PolyMeshRaw,
    /// One row per stage of (92.55) this run did, in order - a stopped run's
    /// vector is shorter, a skipped layers stage still has its row.
    pub stages: Vec<StageTiming>,
    /// The LAST stage's own `QualityReport` - every stage of (92.55) gates
    /// its own output before returning it, so nothing here is re-measured.
    pub quality: QualityReport,
    pub stopped_after: Option<Stage>,
    pub total_seconds: f64,
}

/// §92.14's banner: `=== stage 2/4  castellate ===`. One-based, and the
/// denominator is how many stages THIS run will do - a stopped run's
/// denominator shrinks with it.
fn banner(progress: &mut dyn FnMut(&str), i: usize, n: usize, stage: Stage) {
    progress(&format!("=== stage {}/{}  {} ===", i, n, stage.name()));
}

/// §92.14's elapsed line: `--- castellate: 412.8 s`.
fn elapsed(progress: &mut dyn FnMut(&str), stage: Stage, seconds: f64) {
    progress(&format!("--- {}: {:.1} s", stage.name(), seconds));
}

/// A multi-line report, emitted one line per call and kept for
/// [`StageTiming::log`]. The summaries end in a newline, so the split's
/// empty last piece is dropped.
fn report_lines(progress: &mut dyn FnMut(&str), text: &str) -> String {
    let lines: Vec<&str> = text.split('\n').filter(|l| !l.is_empty()).collect();
    for l in &lines {
        progress(l);
    }
    let mut joined = lines.join("\n");
    if !joined.is_empty() {
        joined.push('\n');
    }
    joined
}

// ==========================================================================
//  The run - §92.14
// ==========================================================================

/// (92.55)'s stop rule, applied: the stages up to and including `stop_after`
/// run, every stage gates its own output before returning it, and (92.56) is
/// applied ONCE, to the mesh that leaves the last stage that ran.
fn finish(
    cfg: &AutomeshConfig,
    stop_after: Option<Stage>,
    mesh: Option<PolyMeshRaw>,
    quality: Option<QualityReport>,
    stages: Vec<StageTiming>,
    t0: Instant,
) -> Result<PipelineOutput> {
    let mut mesh = mesh.expect("a run that ran a stage has a mesh");
    rename_patches(&mut mesh, &cfg.output.patch_names)?;
    Ok(PipelineOutput {
        mesh,
        stages,
        quality: quality.expect("the stage that made the mesh gated it"),
        stopped_after: stop_after,
        total_seconds: t0.elapsed().as_secs_f64(),
    })
}

/// §92.14: run the stages of (92.55) in order, stopping after `stop_after`
/// when it is `Some`. `progress` is called with ONE line at a time (no
/// trailing newline) - the banner before each stage runs, the stage's own
/// report lines as it returns, the elapsed line after - so the caller decides
/// where progress goes and a stage that hangs has already been named.
pub fn run(
    cfg: &AutomeshConfig,
    surf: &Surface,
    stop_after: Option<Stage>,
    progress: &mut dyn FnMut(&str),
) -> Result<PipelineOutput> {
    let t0 = Instant::now();
    let thr = cfg.quality.thresholds();
    let names = super::octree::patch_names();
    let mut stages: Vec<StageTiming> = Vec::new();
    let mut mesh: Option<PolyMeshRaw>;
    let mut quality: Option<QualityReport>;

    // The stop rule: the denominator of every banner is how many stages THIS
    // run will do.
    let n = match stop_after {
        Some(s) => Stage::ALL.iter().position(|x| *x == s).expect("a Stage") + 1,
        None => Stage::ALL.len(),
    };

    // ---- stage 1: octree - the tree everyone downstream needs, kept alive
    // in these locals so no stage re-refines it.
    banner(progress, 1, n, Stage::Octree);
    let started = Instant::now();
    let bg = super::octree::Background::from_domain(&cfg.domain)?;
    let mut tree = super::octree::Octree::uniform(bg.base_n(), cfg.refinement.max_level)?;
    let (refine_splits, balance_splits) =
        super::octree::refine_to_surface(&mut tree, &bg, surf, &cfg.refinement)?;
    let leaf_mesh = super::octree::emit(&tree, &bg, &names)?;
    // The octree is the one stage that carries no gate of its own, so the
    // driver measures its mesh itself - §92.14: a stopped run is not a way
    // to get an ungated mesh out of the mesher.
    let leaf_quality = super::quality::measure(&leaf_mesh, &thr)?;
    let counts = json!({
        "base_n": bg.base_n(),
        "max_level": tree.max_level(),
        "refine_splits": refine_splits,
        "balance_splits": balance_splits,
        "n_leaves": leaf_quality.n_cells,
    });
    let log = report_lines(progress, &leaf_quality.summary());
    let seconds = started.elapsed().as_secs_f64();
    elapsed(progress, Stage::Octree, seconds);
    stages.push(StageTiming { stage: Stage::Octree, seconds, counts, log });
    mesh = Some(leaf_mesh);
    quality = Some(leaf_quality);
    if stop_after == Some(Stage::Octree) {
        return finish(cfg, stop_after, mesh, quality, stages, t0);
    }

    // ---- stage 2: castellate, on the tree stage 1 left behind.
    banner(progress, 2, n, Stage::Castellate);
    let started = Instant::now();
    let cast =
        super::castellate::castellate(&tree, &bg, surf, &names, &cfg.castellation, &thr)?;
    let seconds = started.elapsed().as_secs_f64();
    let mut wall_patches = serde_json::Map::new();
    for (name, faces) in &cast.wall_patches {
        wall_patches.insert(name.clone(), json!(faces));
    }
    let counts = json!({
        "n_cells": cast.report.n_cells,
        "wall_faces": cast.wall_faces,
        "removed": {
            "solid": cast.removed.solid,
            "pinch": cast.removed.pinch,
            "off_region": cast.removed.off_region,
            "starved": cast.removed.starved,
            "unrepaired_pinch": cast.removed.unrepaired_pinch,
        },
        "n_dropped_components": cast.removed.dropped_boxes.len(),
        "wall_patches": wall_patches,
    });
    let log = report_lines(progress, &cast.report.summary());
    elapsed(progress, Stage::Castellate, seconds);
    stages.push(StageTiming { stage: Stage::Castellate, seconds, counts, log });
    mesh = Some(cast.mesh);
    quality = Some(cast.report);
    if stop_after == Some(Stage::Castellate) {
        return finish(cfg, stop_after, mesh, quality, stages, t0);
    }

    // ---- stage 3: snap. The feature attraction §92.12 folded into this
    // stage's own loop is why `features` parses to Snap and stops here too.
    banner(progress, 3, n, Stage::Snap);
    let started = Instant::now();
    let snapped = super::snap::snap(
        mesh.as_ref().expect("castellate ran"),
        surf,
        cfg.domain.base_size as Scalar,
        cfg.refinement.feature_angle_deg as Scalar,
        &cfg.snap,
        &thr,
    )?;
    let seconds = started.elapsed().as_secs_f64();
    let counts = json!({
        "n_boundary_points": snapped.report.n_boundary_points,
        "iterations": snapped.report.iterations,
        "converged": snapped.report.converged,
        "max_step": snapped.report.max_step,
        "max_residual": snapped.report.max_residual,
        "p99_residual": snapped.report.p99_residual,
        "n_scaled_back": snapped.report.n_scaled_back,
        "n_pinned": snapped.report.n_pinned,
        "n_abandoned": snapped.report.n_abandoned,
        "n_feature_edges": snapped.report.n_feature_edges,
        "n_feature_corners": snapped.report.n_feature_corners,
        "n_snapped_to_edge": snapped.report.n_snapped_to_edge,
        "n_snapped_to_corner": snapped.report.n_snapped_to_corner,
    });
    let log = report_lines(progress, &snapped.report.summary());
    elapsed(progress, Stage::Snap, seconds);
    stages.push(StageTiming { stage: Stage::Snap, seconds, counts, log });
    mesh = Some(snapped.mesh);
    quality = Some(snapped.quality);
    if stop_after == Some(Stage::Snap) {
        return finish(cfg, stop_after, mesh, quality, stages, t0);
    }

    // ---- stage 4: layers. An empty layer spec makes `add_layers` a no-op
    // that still costs a gate pass, so the stage is recorded and skipped
    // instead of run.
    banner(progress, 4, n, Stage::Layers);
    if cfg.layers.n == 0 || cfg.layers.patches.is_empty() {
        let counts = json!({
            "skipped": true,
            "n_layer_cells": 0,
            "n_layer_points": 0,
            "n_side_internal": 0,
            "n_side_boundary": 0,
            "n_split_sides": 0,
            "retreats": 0,
            "patches": [],
        });
        let log = report_lines(
            progress,
            "layers: skipped - layers.n is 0 or layers.patches is empty\n",
        );
        elapsed(progress, Stage::Layers, 0.0);
        stages.push(StageTiming { stage: Stage::Layers, seconds: 0.0, counts, log });
    } else {
        let started = Instant::now();
        let lay = super::layers::add_layers(
            mesh.as_ref().expect("snap ran"),
            surf,
            &cfg.layers,
            &thr,
        )?;
        let seconds = started.elapsed().as_secs_f64();
        let patches: Vec<serde_json::Value> =
            lay.report.patches.iter().map(|p| json!({
                "name": p.name,
                "n_layers": p.n_layers,
                "n_faces": p.n_faces,
                "full_area_frac": p.full_area_frac,
                "mean_frac": p.mean_frac,
                "t1_requested": p.t1_requested,
                "t1_mean": p.t1_mean,
                "t1_min": p.t1_min,
                "dropped": p.dropped,
            })).collect();
        let counts = json!({
            "n_layer_cells": lay.report.n_layer_cells,
            "n_layer_points": lay.report.n_layer_points,
            "n_side_internal": lay.report.n_side_internal,
            "n_side_boundary": lay.report.n_side_boundary,
            "n_split_sides": lay.report.n_split_sides,
            "retreats": lay.report.retreats,
            "patches": patches,
        });
        let log = report_lines(progress, &lay.report.summary());
        elapsed(progress, Stage::Layers, seconds);
        stages.push(StageTiming { stage: Stage::Layers, seconds, counts, log });
        mesh = Some(lay.mesh);
        quality = Some(lay.quality);
    }
    finish(cfg, stop_after, mesh, quality, stages, t0)
}

// ==========================================================================
//  (92.56) - the names the case gets
// ==========================================================================

/// (92.56) on its own: rename `mesh.patches` through `map`, refusing
///
/// * a key that names no patch of the mesh - a typo silently ignored is a
///   case wired to a patch that is not there,
/// * a value [`crate::io::polymesh::check_patch_name`] would refuse - the
///   same check the octree's and the surface's own patches went through,
/// * two patches that would end up with the SAME name.
///
/// Nothing is changed when it refuses.
pub fn rename_patches(
    mesh: &mut PolyMeshRaw,
    map: &std::collections::BTreeMap<String, String>,
) -> Result<()> {
    if map.is_empty() {
        return Ok(());
    }
    for from in map.keys() {
        if !mesh.patches.iter().any(|p| p.name == *from) {
            let have: Vec<&str> = mesh.patches.iter().map(|p| p.name.as_str()).collect();
            return Err(Error::Config(format!(
                "output.patch_names: key \"{from}\" names no patch of the mesh \
                 (patches: {})",
                have.join(", ")
            )));
        }
    }
    for to in map.values() {
        if let Err(e) = crate::io::polymesh::check_patch_name(to, "output.patch_names value") {
            return Err(Error::Config(format!("output.patch_names: {e}")));
        }
    }
    // The collision check runs on the FINAL names, so it catches a value
    // that walks into an untouched patch's name as well as two keys that
    // rename onto each other.
    let final_names: Vec<String> = mesh
        .patches
        .iter()
        .map(|p| map.get(&p.name).cloned().unwrap_or_else(|| p.name.clone()))
        .collect();
    for i in 0..final_names.len() {
        if let Some(j) = final_names[..i].iter().position(|n| *n == final_names[i]) {
            return Err(Error::Config(format!(
                "output.patch_names: \"{a}\" and \"{b}\" would both be named \
                 \"{name}\" - two patches cannot share a boundary entry",
                a = mesh.patches[j].name,
                b = mesh.patches[i].name,
                name = final_names[i],
            )));
        }
    }
    for p in &mut mesh.patches {
        if let Some(to) = map.get(&p.name) {
            p.name = to.clone();
        }
    }
    Ok(())
}

// ==========================================================================
//  (92.57) - the summary the run leaves
// ==========================================================================

/// (92.57): the summary object, ready for `serde_json::to_string_pretty`.
/// The patch list is the FINAL one - `out.mesh` carries (92.56)'s names -
/// and `config` is the parsed struct re-serialised, defaults filled in,
/// which is the thing a second run has to match.
pub fn summary_json(
    cfg: &AutomeshConfig,
    config_path: &str,
    surf: &Surface,
    out: &PipelineOutput,
) -> serde_json::Value {
    let stages: Vec<serde_json::Value> = out
        .stages
        .iter()
        .map(|s| {
            let mut o = serde_json::Map::new();
            o.insert("stage".to_string(), json!(s.stage.name()));
            o.insert("seconds".to_string(), json!(s.seconds));
            // The stage's own counts splice in at the top level of the
            // object, next to `stage` and `seconds`.
            if let serde_json::Value::Object(extra) = &s.counts {
                for (k, v) in extra {
                    o.insert(k.clone(), v.clone());
                }
            }
            serde_json::Value::Object(o)
        })
        .collect();

    let (lo, hi) = surf.bbox;
    let mesh_patches: Vec<serde_json::Value> = out
        .mesh
        .patches
        .iter()
        .map(|p| {
            json!({
                "name": p.name,
                // The BOUNDARY FILE's own `type` word, not the enum's name:
                // a reader comparing this list against `constant/polyMesh/
                // boundary` must see the same three strings it does.
                // (Written by the supervising session.)
                "kind": p.type_name,
                "size": p.size,
            })
        })
        .collect();

    let q = &out.quality;
    json!({
        "tool": "ofgpu-automesher",
        "config_path": config_path,
        "name": cfg.output.name,
        "case_dir": cfg.output.case_dir,
        "stopped_after": out.stopped_after.map(|s| json!(s.name())),
        "surface": {
            "n_triangles": surf.tris.len(),
            "n_points": surf.points.len(),
            "bbox": [
                lo.x as f64, lo.y as f64, lo.z as f64,
                hi.x as f64, hi.y as f64, hi.z as f64,
            ],
            "patches": surf.patch_names,
        },
        "stages": stages,
        "mesh": {
            "n_points": out.mesh.points.len(),
            "n_cells": q.n_cells,
            "n_internal_faces": out.mesh.neighbour.len(),
            "n_boundary_faces": out.mesh.faces.len() - out.mesh.neighbour.len(),
            "patches": mesh_patches,
        },
        "quality": {
            "n_cells": q.n_cells,
            "min_volume": q.min_volume as f64,
            "min_volume_cell": q.min_volume_cell,
            "max_closure": q.max_closure as f64,
            "max_closure_cell": q.max_closure_cell,
            "n_regions": q.n_regions,
            "region_sizes": q.region_sizes,
            "max_non_orth_deg": q.max_non_orth_deg as f64,
            "mean_non_orth_deg": q.mean_non_orth_deg as f64,
            "n_non_orth_over_report": q.n_non_orth_over_report,
            "min_thickness_ratio": q.min_thickness_ratio as f64,
            "min_thickness_cell": q.min_thickness_cell,
            "max_cond": q.max_cond as f64,
            "max_cond_cell": q.max_cond_cell,
            "n_duplicate_faces": q.n_duplicate_faces,
            "ldu_ordered": q.ldu_ordered,
            "passed": q.passed(),
        },
        "total_seconds": out.total_seconds,
        "config": serde_json::to_value(cfg).unwrap_or(serde_json::Value::Null),
    })
}

// ==========================================================================
//  Tests
// ==========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;
    use crate::automesher::castellate::tests::box_soup;
    use crate::automesher::{
        CastellationSpec, DistanceBand, DomainSpec, InputSpec, LayerSpec, OutputSpec, QualitySpec,
        RefinementBand, RefinementSpec, SnapSpec, SurfaceInput,
    };
    use crate::surface::Surface;

    /// A 4 m box at base size 1, one level deep, with a 1 m cube at its
    /// centre and two layers on the cube's wall: the smallest config every
    /// stage of (92.55) runs on, end to end.
    fn cube_config() -> AutomeshConfig {
        AutomeshConfig {
            schema: None,
            input: InputSpec {
                surfaces: vec![SurfaceInput { path: "cube.stl".to_string(), name: None }],
            },
            domain: DomainSpec {
                extent: [0.0, 4.0, 0.0, 4.0, 0.0, 4.0],
                base_size: 1.0,
                grading: [1.0; 3],
            },
            refinement: RefinementSpec {
                levels: vec![RefinementBand {
                    patch: "cube".to_string(),
                    bands: vec![DistanceBand { distance: 0.0, level: 1 }],
                    feature_level: 0,
                }],
                feature_angle_deg: 30.0,
                max_level: 1,
            },
            castellation: CastellationSpec::default(),
            snap: SnapSpec::default(),
            layers: LayerSpec {
                patches: vec!["cube".to_string()],
                n: 2,
                first_thickness: 0.02,
                ..LayerSpec::default()
            },
            quality: QualitySpec::default(),
            output: OutputSpec {
                case_dir: "out".to_string(),
                name: "cube".to_string(),
                patch_names: BTreeMap::new(),
            },
        }
    }

    /// The config's surface: the 1 m cube, wound outward, one patch.
    fn cube_surface() -> Surface {
        Surface::from_soup(box_soup([1.3; 3], [2.3; 3]), vec!["cube".to_string()])
            .expect("surface")
    }

    /// Run it, keeping every progress line.
    fn run_recording(
        cfg: &AutomeshConfig,
        surf: &Surface,
        stop_after: Option<Stage>,
    ) -> (PipelineOutput, Vec<String>) {
        let mut lines: Vec<String> = Vec::new();
        let out = run(cfg, surf, stop_after, &mut |l: &str| lines.push(l.to_string()))
            .expect("run");
        (out, lines)
    }

    #[test]
    fn stage_parses_features_as_snap() {
        assert_eq!(Stage::parse("octree").unwrap(), Stage::Octree);
        assert_eq!(Stage::parse("castellate").unwrap(), Stage::Castellate);
        assert_eq!(Stage::parse("snap").unwrap(), Stage::Snap);
        assert_eq!(Stage::parse("layers").unwrap(), Stage::Layers);
        // §92.14: the feature attraction lives inside stage 4, so there is
        // no mesh between them to stop at.
        assert_eq!(Stage::parse("features").unwrap(), Stage::Snap);
        let err = Stage::parse("featur").unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("featur"), "{msg}");
        for spelling in ["octree", "castellate", "features", "snap", "layers"] {
            assert!(msg.contains(spelling), "{msg}");
        }
    }

    #[test]
    fn full_pipeline_runs_every_stage_in_order() {
        let cfg = cube_config();
        let surf = cube_surface();
        let (out, lines) = run_recording(&cfg, &surf, None);
        assert_eq!(out.stopped_after, None);
        let names: Vec<&str> = out.stages.iter().map(|s| s.stage.name()).collect();
        assert_eq!(names, ["octree", "castellate", "snap", "layers"]);
        for s in &out.stages {
            assert!(s.seconds >= 0.0, "{}: {}", s.stage.name(), s.seconds);
        }
        // The last stage's own gate is the one the run reports.
        assert!(out.quality.passed(), "the final mesh must pass the gate");
        // Every banner was emitted before its stage ran, and the elapsed
        // line of each stage follows its banner.
        assert!(
            lines.iter().any(|l| l == "=== stage 1/4  octree ==="),
            "{lines:?}"
        );
        assert!(
            lines.iter().any(|l| l.starts_with("--- layers:")),
            "{lines:?}"
        );
        assert_eq!(
            lines.iter().filter(|l| l.starts_with("=== stage")).count(),
            4
        );
    }

    #[test]
    fn stop_after_octree_writes_the_leaf_mesh() {
        let cfg = cube_config();
        let surf = cube_surface();
        let (out, lines) = run_recording(&cfg, &surf, Some(Stage::Octree));
        assert_eq!(out.stopped_after, Some(Stage::Octree));
        assert_eq!(out.stages.len(), 1);
        assert_eq!(out.stages[0].stage, Stage::Octree);
        // No wall patch exists yet: the six domain patches, and the banner's
        // denominator is this run's own stage count.
        assert_eq!(out.mesh.patches.len(), 6);
        for name in ["xMin", "xMax", "yMin", "yMax", "zMin", "zMax"] {
            assert!(
                out.mesh.patches.iter().any(|p| p.name == name),
                "no {name} in {:?}",
                out.mesh.patches.iter().map(|p| p.name.as_str()).collect::<Vec<_>>()
            );
        }
        // The cell count of the mesh IS the leaf count the stage recorded.
        let leaves = out.stages[0].counts.get("n_leaves").unwrap().as_u64().unwrap();
        assert_eq!(out.quality.n_cells as u64, leaves);
        assert!(
            lines.iter().any(|l| l == "=== stage 1/1  octree ==="),
            "{lines:?}"
        );
        assert!(!lines.iter().any(|l| l.starts_with("--- castellate:")));
    }

    #[test]
    fn features_and_snap_stop_at_the_same_mesh() {
        let cfg = cube_config();
        let surf = cube_surface();
        let (a, _) = run_recording(&cfg, &surf, Some(Stage::parse("features").unwrap()));
        let (b, _) = run_recording(&cfg, &surf, Some(Stage::parse("snap").unwrap()));
        assert_eq!(a.stopped_after, Some(Stage::Snap));
        assert_eq!(b.stopped_after, Some(Stage::Snap));
        assert_eq!(a.mesh.points.len(), b.mesh.points.len());
        for (pa, pb) in a.mesh.points.iter().zip(b.mesh.points.iter()) {
            assert_eq!(pa.x, pb.x);
            assert_eq!(pa.y, pb.y);
            assert_eq!(pa.z, pb.z);
        }
    }

    #[test]
    fn rename_patches_applies_and_refuses() {
        let cfg = cube_config();
        let surf = cube_surface();
        let (stopped, _) = run_recording(&cfg, &surf, Some(Stage::Octree));
        let mut mesh = stopped.mesh;

        // One rename, the rest left alone.
        let mut map = BTreeMap::new();
        map.insert("xMin".to_string(), "west".to_string());
        rename_patches(&mut mesh, &map).unwrap();
        assert!(mesh.patches.iter().any(|p| p.name == "west"));
        assert!(mesh.patches.iter().any(|p| p.name == "xMax"));

        let (stopped, _) = run_recording(&cfg, &surf, Some(Stage::Octree));
        let mut mesh = stopped.mesh;

        // A key that names no patch - a typo must not be silently dropped.
        let mut typo = BTreeMap::new();
        typo.insert("xmin".to_string(), "west".to_string());
        let err = rename_patches(&mut mesh, &typo).unwrap_err();
        assert!(err.to_string().contains("xmin"), "{err}");
        assert!(mesh.patches.iter().any(|p| p.name == "xMin"), "refusal changed nothing");

        // A value that walks into another patch's FINAL name - both source
        // patches are named.
        let mut collide = BTreeMap::new();
        collide.insert("xMin".to_string(), "xMax".to_string());
        let err = rename_patches(&mut mesh, &collide).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("xMin") && msg.contains("xMax"), "{msg}");
        assert!(mesh.patches.iter().any(|p| p.name == "xMin"), "refusal changed nothing");

        // A value the boundary file could not carry - a space breaks the
        // bare-token grammar the reader owns.
        let mut spaced = BTreeMap::new();
        spaced.insert("xMin".to_string(), "my patch".to_string());
        let err = rename_patches(&mut mesh, &spaced).unwrap_err();
        assert!(err.to_string().contains("my patch"), "{err}");
        assert!(mesh.patches.iter().any(|p| p.name == "xMin"), "refusal changed nothing");
    }

    #[test]
    fn summary_json_describes_the_written_mesh() {
        let mut cfg = cube_config();
        cfg.output
            .patch_names
            .insert("xMin".to_string(), "west".to_string());
        let surf = cube_surface();
        let (out, _) = run_recording(&cfg, &surf, None);
        let s = summary_json(&cfg, "cube.automesher.json", &surf, &out);

        // The four stages, in order, each with its own counts spliced in.
        let stages = s["stages"].as_array().unwrap();
        let names: Vec<&str> =
            stages.iter().map(|v| v["stage"].as_str().unwrap()).collect();
        assert_eq!(names, ["octree", "castellate", "snap", "layers"]);
        assert!(stages[0].get("n_leaves").is_some(), "{}", stages[0]);
        assert!(stages[1].get("wall_faces").is_some(), "{}", stages[1]);
        assert!(stages[3].get("n_layer_cells").is_some(), "{}", stages[3]);

        // The mesh block describes the WRITTEN mesh: the final names, after
        // (92.56), and the report's own cell count.
        assert_eq!(
            s["mesh"]["n_cells"].as_u64().unwrap() as usize,
            out.quality.n_cells
        );
        let summary_names: Vec<String> = s["mesh"]["patches"]
            .as_array()
            .unwrap()
            .iter()
            .map(|p| p["name"].as_str().unwrap().to_string())
            .collect();
        let mesh_names: Vec<String> =
            out.mesh.patches.iter().map(|p| p.name.clone()).collect();
        assert_eq!(summary_names, mesh_names);
        assert!(summary_names.iter().any(|n| n == "west"), "{summary_names:?}");

        // A full run stops nowhere.
        assert_eq!(s["stopped_after"], serde_json::Value::Null);
        assert_eq!(s["tool"], "ofgpu-automesher");
        assert_eq!(s["config_path"], "cube.automesher.json");
        assert_eq!(s["surface"]["patches"].as_array().unwrap().len(), 1);

        // `config` is the parsed struct re-serialised: it parses BACK to the
        // config the run actually used.
        let back: AutomeshConfig = serde_json::from_value(s["config"].clone()).unwrap();
        assert_eq!(back, cfg);
    }
}
