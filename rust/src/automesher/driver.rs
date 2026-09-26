// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.
// Provenance: see PROVENANCE.md. No GPL-licensed source was consulted.

//! The driver - SPEC-LIT §92.14: §92.9's octree, §92.10's castellation,
//! §92.11's snapping and §92.13's layers run in (92.55)'s order by one call
//! (§92.15's split between snap and layers when bodies are declared),
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
use super::{AutomeshConfig, LayerSpec};

// ==========================================================================
//  The stages - (92.55)
// ==========================================================================

/// (92.55)'s stage list - the five stages that return a mesh (the split
/// returns the layout; on a run with no body it is recorded and skipped).
/// Stage 0 (the background block) and stage 2 (the 2:1 balance) are not
/// here: neither returns a mesh of its own.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Stage {
    Octree,
    Castellate,
    Snap,
    Split,
    Layers,
}

impl Stage {
    /// The name in a banner, a config and the summary.
    pub fn name(self) -> &'static str {
        match self {
            Stage::Octree => "octree",
            Stage::Castellate => "castellate",
            Stage::Snap => "snap",
            Stage::Split => "split",
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
            "split" => Ok(Stage::Split),
            "layers" => Ok(Stage::Layers),
            _ => Err(Error::Config(format!(
                "stopAfter: unknown stage \"{s}\" - one of \
                 \"octree\", \"castellate\", \"features\", \"snap\", \
                 \"split\", \"layers\""
            ))),
        }
    }

    /// The five, in order.
    pub const ALL: [Stage; 5] =
        [Stage::Octree, Stage::Castellate, Stage::Snap, Stage::Split, Stage::Layers];
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
    /// On a split run this is the FLUID region's mesh (a clone of
    /// `layout.regions[0].mesh`), so every reader of one mesh keeps working.
    pub mesh: PolyMeshRaw,
    /// One row per stage of (92.55) this run did, in order - a stopped run's
    /// vector is shorter, a skipped layers stage still has its row.
    pub stages: Vec<StageTiming>,
    /// The LAST stage's own `QualityReport` - every stage of (92.55) gates
    /// its own output before returning it, so nothing here is re-measured.
    pub quality: QualityReport,
    pub stopped_after: Option<Stage>,
    pub total_seconds: f64,
    /// `None` when `castellation.bodies` is empty or the run stopped before
    /// the split; `Some` after the split stage ran.
    pub layout: Option<LayoutOutput>,
}

/// One region of a split run: its name in the layout, its kind, its mesh
/// after every stage that ran, and the LAST gate it passed.
#[derive(Debug, Clone)]
pub struct RegionOutput {
    pub name: String,
    pub kind: crate::cht::RegionKind,
    pub mesh: PolyMeshRaw,
    pub quality: QualityReport,
}

/// What the split stage produced: `regions[0]` is the fluid (SPEC-LIT
/// §47.4: the fluid is region 0), then the bodies in `castellation.bodies`
/// order, and the interface pairs `split_by_zones` named (`fluid_to_<b>`,
/// `<b>_to_fluid`, docs/10 §C R3).
#[derive(Debug, Clone)]
pub struct LayoutOutput {
    pub regions: Vec<RegionOutput>,
    pub interfaces: Vec<crate::io::regions::SplitInterface>,
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
/// applied ONCE, to the mesh that leaves the last stage that ran. With a
/// layout (the split ran) the rename applies PER REGION, and a key naming an
/// interface patch - or a patch of no region - is refused before anything
/// changes.
fn finish(
    cfg: &AutomeshConfig,
    stop_after: Option<Stage>,
    mesh: Option<PolyMeshRaw>,
    quality: Option<QualityReport>,
    stages: Vec<StageTiming>,
    t0: Instant,
    layout: Option<LayoutOutput>,
) -> Result<PipelineOutput> {
    if let Some(mut layout) = layout {
        for from in cfg.output.patch_names.keys() {
            if let Some(iface) = layout
                .interfaces
                .iter()
                .find(|i| i.patch_a == *from || i.patch_b == *from)
            {
                let (a, b) = (
                    &layout.regions[iface.region_a].name,
                    &layout.regions[iface.region_b].name,
                );
                return Err(Error::Config(format!(
                    "output.patch_names: key \"{from}\" names the interface \
                     patch of regions \"{a}\" and \"{b}\" - interface names \
                     are the layout's contract (docs/10 §C R3) and are not \
                     renamed"
                )));
            }
            if !layout
                .regions
                .iter()
                .any(|r| r.mesh.patches.iter().any(|p| p.name == *from))
            {
                let lists: Vec<String> = layout
                    .regions
                    .iter()
                    .map(|r| {
                        format!(
                            "{}: {:?}",
                            r.name,
                            r.mesh.patches.iter().map(|p| p.name.as_str()).collect::<Vec<_>>()
                        )
                    })
                    .collect();
                return Err(Error::Config(format!(
                    "output.patch_names: key \"{from}\" names no patch of any \
                     region ({})"
                , lists.join("; "))));
            }
        }
        for region in &mut layout.regions {
            let sub: std::collections::BTreeMap<String, String> = cfg
                .output
                .patch_names
                .iter()
                .filter(|(k, _)| region.mesh.patches.iter().any(|p| p.name == **k))
                .map(|(k, v)| (k.clone(), v.clone()))
                .collect();
            rename_patches(&mut region.mesh, &sub)?;
        }
        let mesh = layout.regions[0].mesh.clone();
        let quality = layout.regions[0].quality.clone();
        return Ok(PipelineOutput {
            mesh,
            stages,
            quality,
            stopped_after: stop_after,
            total_seconds: t0.elapsed().as_secs_f64(),
            layout: Some(layout),
        });
    }
    let mut mesh = mesh.expect("a run that ran a stage has a mesh");
    rename_patches(&mut mesh, &cfg.output.patch_names)?;
    Ok(PipelineOutput {
        mesh,
        stages,
        quality: quality.expect("the stage that made the mesh gated it"),
        stopped_after: stop_after,
        total_seconds: t0.elapsed().as_secs_f64(),
        layout: None,
    })
}

/// The split stage's body when `castellation.bodies` is not empty: one
/// standalone mesh per region, the fluid first (SPEC-LIT §47.4), each gated
/// on its own, one log line per region and per interface.
fn split_stage(
    progress: &mut dyn FnMut(&str),
    stages: &mut Vec<StageTiming>,
    mesh: &mut Option<PolyMeshRaw>,
    quality: &mut Option<QualityReport>,
    layout: &mut Option<LayoutOutput>,
    cast_region: &[i32],
    body_names: &[String],
    thr: &super::quality::QualityThresholds,
) -> Result<()> {
    let started = Instant::now();
    let snapped_mesh = mesh.take().expect("snap ran");
    // SPEC-LIT §47.4: the fluid is regions[0], so its zone is first and the
    // bodies follow in `castellation.bodies` order, cell ids ascending.
    let mut zones: Vec<(String, Vec<crate::Label>)> =
        vec![("fluid".to_string(), Vec::new())];
    for (c, r) in cast_region.iter().enumerate() {
        if *r == -1 {
            zones[0].1.push(c as crate::Label);
        }
    }
    for (k, name) in body_names.iter().enumerate() {
        let mut cells = Vec::new();
        for (c, r) in cast_region.iter().enumerate() {
            if *r == k as i32 {
                cells.push(c as crate::Label);
            }
        }
        zones.push((name.clone(), cells));
    }
    let (parts, ifaces) = crate::io::regions::split_by_zones(&snapped_mesh, &zones)?;
    let mut regions: Vec<RegionOutput> = Vec::with_capacity(parts.len());
    let mut region_counts: Vec<serde_json::Value> = Vec::new();
    let mut log = String::new();
    for (i, (name, m)) in parts.into_iter().enumerate() {
        let q = super::quality::check(&m, thr)
            .map_err(|e| Error::Mesh(format!("split: region \"{name}\": {e}")))?;
        let kind_word = if i == 0 { "fluid" } else { "solid" };
        log.push_str(&format!(
            "split: region \"{name}\" ({kind_word}): {} cells, {} patches\n",
            q.n_cells,
            m.patches.len()
        ));
        region_counts.push(json!({
            "name": name,
            "kind": kind_word,
            "n_cells": q.n_cells,
            "n_points": m.points.len(),
            "n_internal_faces": m.neighbour.len(),
            "n_boundary_faces": m.faces.len() - m.neighbour.len(),
            "n_patches": m.patches.len(),
        }));
        let kind = if i == 0 {
            crate::cht::RegionKind::Fluid
        } else {
            crate::cht::RegionKind::Solid
        };
        regions.push(RegionOutput { name, kind, mesh: m, quality: q });
    }
    let interfaces_json: Vec<serde_json::Value> = ifaces
        .iter()
        .map(|i| {
            log.push_str(&format!(
                "split: interface {} <-> {}: {} faces ({} / {})\n",
                regions[i.region_a].name,
                regions[i.region_b].name,
                i.n_faces,
                i.patch_a,
                i.patch_b
            ));
            json!({
                "regions": [regions[i.region_a].name.clone(), regions[i.region_b].name.clone()],
                "patches": [i.patch_a.clone(), i.patch_b.clone()],
                "faces": i.n_faces,
            })
        })
        .collect();
    let counts = json!({
        "n_regions": regions.len(),
        "regions": region_counts,
        "interfaces": interfaces_json,
    });
    let mut log = report_lines(progress, &log);
    for r in &regions {
        log.push_str(&report_lines(progress, &r.quality.summary()));
    }
    let seconds = started.elapsed().as_secs_f64();
    elapsed(progress, Stage::Split, seconds);
    stages.push(StageTiming { stage: Stage::Split, seconds, counts, log });
    *mesh = None;
    *quality = None;
    *layout = Some(LayoutOutput { regions, interfaces: ifaces });
    Ok(())
}

/// One layer row of the summary, (92.50) per patch: the same keys on the
/// single-mesh path and in each region's rows (SPEC-LIT §92.14).
fn layer_patch_json(p: &super::layers::PatchLayers) -> serde_json::Value {
    let tau_ge: Vec<serde_json::Value> = super::layers::TAU_GE_BETAS
        .iter()
        .zip(p.area_frac_tau_ge.iter())
        .map(|(b, f)| json!({ "beta": b, "area_frac": f }))
        .collect();
    json!({
        "name": p.name,
        "n_layers": p.n_layers,
        "n_faces": p.n_faces,
        "area": p.area,
        "full_area_frac": p.full_area_frac,
        "area_frac_tau_ge": tau_ge,
        "mean_frac": p.mean_frac,
        "t1_requested": p.t1_requested,
        "t1_mean": p.t1_mean,
        "t1_min": p.t1_min,
        "dropped": p.dropped,
    })
}

/// The layers stage on a split run: `layers.patches` names the SPLIT patch
/// names (`fluid_to_<body>`, `<body>_to_fluid`), so each region filters the
/// spec down to the patches it carries, a region with none is recorded and
/// skipped, and the stage's counts are the totals SUMMED over the regions
/// plus one row per region.
fn layers_regions_stage(
    progress: &mut dyn FnMut(&str),
    stages: &mut Vec<StageTiming>,
    layout: &mut LayoutOutput,
    cfg: &AutomeshConfig,
    surf: &Surface,
    thr: &super::quality::QualityThresholds,
) -> Result<()> {
    let started = Instant::now();
    // A patch no region carries is a typo or a stale name: refuse naming
    // every region's patch list, before any region is touched.
    for name in &cfg.layers.patches {
        if !layout
            .regions
            .iter()
            .any(|r| r.mesh.patches.iter().any(|p| p.name == *name))
        {
            let lists: Vec<String> = layout
                .regions
                .iter()
                .map(|r| {
                    format!(
                        "{} carries {:?}",
                        r.name,
                        r.mesh.patches.iter().map(|p| p.name.as_str()).collect::<Vec<_>>()
                    )
                })
                .collect();
            return Err(Error::Mesh(format!(
                "layers: patch \"{name}\" is not a patch of any region - {}",
                lists.join("; ")
            )));
        }
    }
    let mut totals = (0usize, 0usize, 0usize, 0usize, 0usize, 0usize);
    let mut rows: Vec<serde_json::Value> = Vec::new();
    let mut log = String::new();
    for r in layout.regions.iter_mut() {
        let mine: Vec<String> = cfg
            .layers
            .patches
            .iter()
            .filter(|p| r.mesh.patches.iter().any(|q| q.name == **p))
            .cloned()
            .collect();
        if mine.is_empty() {
            log.push_str(&format!(
                "layers: region \"{}\": skipped - none of layers.patches is \
                 a patch of this region\n",
                r.name
            ));
            rows.push(json!({
                "name": r.name,
                "skipped": true,
                "n_layer_cells": 0,
                "patches": [],
            }));
            continue;
        }
        let sub = LayerSpec { patches: mine, ..cfg.layers.clone() };
        let lay = super::layers::add_layers(&r.mesh, surf, &sub, thr)?;
        let patches: Vec<serde_json::Value> = lay
            .report
            .patches
            .iter()
            .map(layer_patch_json)
            .collect();
        rows.push(json!({
            "name": r.name,
            "n_layer_cells": lay.report.n_layer_cells,
            "n_layer_points": lay.report.n_layer_points,
            "n_side_internal": lay.report.n_side_internal,
            "n_side_boundary": lay.report.n_side_boundary,
            "n_split_sides": lay.report.n_split_sides,
            "retreats": lay.report.retreats,
            "patches": patches,
        }));
        totals.0 += lay.report.n_layer_cells;
        totals.1 += lay.report.n_layer_points;
        totals.2 += lay.report.n_side_internal;
        totals.3 += lay.report.n_side_boundary;
        totals.4 += lay.report.n_split_sides;
        totals.5 += lay.report.retreats;
        log.push_str(&format!("layers: region \"{}\":\n", r.name));
        log.push_str(&lay.report.summary());
        if !log.ends_with('\n') {
            log.push('\n');
        }
        r.mesh = lay.mesh;
        r.quality = lay.quality;
    }
    let counts = json!({
        "n_layer_cells": totals.0,
        "n_layer_points": totals.1,
        "n_side_internal": totals.2,
        "n_side_boundary": totals.3,
        "n_split_sides": totals.4,
        "retreats": totals.5,
        "regions": rows,
    });
    let log = report_lines(progress, &log);
    let seconds = started.elapsed().as_secs_f64();
    elapsed(progress, Stage::Layers, seconds);
    stages.push(StageTiming { stage: Stage::Layers, seconds, counts, log });
    Ok(())
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
    // The split's inputs, taken from the castellate stage: the region of
    // every cell and the bodies' names. `-1` fluid, `k >= 0` body k; empty
    // when `castellation.bodies` is, which is what keeps the split skipped.
    // The empty initial value is read only when the run stops before the
    // castellate stage fills it.
    #[allow(unused_assignments)]
    let (mut cast_region, mut body_names): (Vec<i32>, Vec<String>) =
        (Vec::new(), Vec::new());
    let mut layout: Option<LayoutOutput> = None;

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
        "gate_passed": leaf_quality.passed(),
        "max_non_orth_deg": leaf_quality.max_non_orth_deg as f64,
    });
    let log = report_lines(progress, &leaf_quality.summary());
    let seconds = started.elapsed().as_secs_f64();
    elapsed(progress, Stage::Octree, seconds);
    stages.push(StageTiming { stage: Stage::Octree, seconds, counts, log });
    mesh = Some(leaf_mesh);
    quality = Some(leaf_quality);
    if stop_after == Some(Stage::Octree) {
        return finish(cfg, stop_after, mesh, quality, stages, t0, layout);
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
    cast_region = cast.region_of_cell;
    body_names = cast.body_names;
    mesh = Some(cast.mesh);
    quality = Some(cast.report);
    if stop_after == Some(Stage::Castellate) {
        return finish(cfg, stop_after, mesh, quality, stages, t0, layout);
    }

    // ---- stage 3: snap. The feature attraction §92.12 folded into this
    // stage's own loop is why `features` parses to Snap and stops here too.
    banner(progress, 3, n, Stage::Snap);
    let started = Instant::now();
    // With a declared body the interface faces' points join the snapped set
    // (§92.11's B), so the interface is moved onto the body's own triangles
    // and stays conformal bit for bit after the split (docs/10 §C R2).
    let snapped = super::snap::snap_regions(
        mesh.as_ref().expect("castellate ran"),
        surf,
        Some(&cast_region),
        cfg.domain.base_size as Scalar,
        cfg.refinement.feature_angle_deg as Scalar,
        &cfg.snap,
        &thr,
    )?;
    let seconds = started.elapsed().as_secs_f64();
    // h_f of docs/15 §D.1's F3: the finest cell, base_size / 2^max_level.
    let h_f = cfg.domain.base_size / 2f64.powi(cfg.refinement.max_level as i32);
    let area_ratio: Vec<serde_json::Value> = snapped
        .report
        .patch_areas
        .iter()
        .map(|p| {
            json!({
                "name": p.name,
                "stl_area_m2": p.stl_area,
                "castellated_area_m2": p.castellated_area,
                "snapped_area_m2": p.snapped_area,
                "castellated_ratio": p.castellated_ratio(),
                "ratio": p.ratio(),
            })
        })
        .collect();
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
        "h_f_m": h_f,
        "p99_over_h": snapped.report.p99_residual as f64 / h_f,
        "max_over_h": snapped.report.max_residual as f64 / h_f,
        "n_pinned_boundary": snapped.report.n_pinned_boundary,
        "area_ratio": area_ratio,
    });
    let log = report_lines(progress, &snapped.report.summary());
    elapsed(progress, Stage::Snap, seconds);
    stages.push(StageTiming { stage: Stage::Snap, seconds, counts, log });
    mesh = Some(snapped.mesh);
    quality = Some(snapped.quality);
    if stop_after == Some(Stage::Snap) {
        return finish(cfg, stop_after, mesh, quality, stages, t0, layout);
    }

    // ---- stage 4: the split. `castellation.bodies` empty is every run
    // before M6: the stage is recorded and skipped, like an empty layer
    // stage.
    banner(progress, 4, n, Stage::Split);
    if cfg.castellation.bodies.is_empty() {
        let counts = json!({
            "skipped": true,
            "n_regions": 1,
        });
        let log = report_lines(
            progress,
            "split: skipped - castellation.bodies is empty\n",
        );
        elapsed(progress, Stage::Split, 0.0);
        stages.push(StageTiming { stage: Stage::Split, seconds: 0.0, counts, log });
    } else {
        split_stage(progress, &mut stages, &mut mesh, &mut quality, &mut layout,
            &cast_region, &body_names, &thr)?;
    }
    if stop_after == Some(Stage::Split) {
        return finish(cfg, stop_after, mesh, quality, stages, t0, layout);
    }

    // ---- stage 5: layers. An empty layer spec makes `add_layers` a no-op
    // that still costs a gate pass, so the stage is recorded and skipped
    // instead of run. On a split run it runs PER REGION.
    banner(progress, 5, n, Stage::Layers);
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
    } else if let Some(lay_out) = layout.as_mut() {
        layers_regions_stage(progress, &mut stages, lay_out, cfg, surf, &thr)?;
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
            lay.report.patches.iter().map(layer_patch_json).collect();
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
    finish(cfg, stop_after, mesh, quality, stages, t0, layout)
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

/// The manifest's word for a region kind (docs/10 §C R4).
fn kind_word(k: crate::cht::RegionKind) -> &'static str {
    match k {
        crate::cht::RegionKind::Fluid => "fluid",
        crate::cht::RegionKind::Solid => "solid",
    }
}

/// (92.57)'s `quality` object - the measured numbers of §92.3's report,
/// gate by gate. The run's own report on a one-mesh run, one region's on a
/// split run.
fn quality_json(q: &QualityReport) -> serde_json::Value {
    json!({
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
    })
}

/// (92.57): the summary object, ready for `serde_json::to_string_pretty`.
/// The patch list is the FINAL one - `out.mesh` carries (92.56)'s names -
/// and `config` is the parsed struct re-serialised, defaults filled in,
/// which is the thing a second run has to match.
pub fn summary_json(
    cfg: &AutomeshConfig,
    config_path: &str,
    surf: &Surface,
    out: &PipelineOutput,
    ident: &crate::automesher::identity::MeshIdentity,
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
    // (92.57)'s `layout` block: `null` unless the split stage ran, and then
    // the written layout's own shape - per region the counts and its OWN
    // gate report, and the interface pairs as the manifest carries them.
    let layout_json = match &out.layout {
        None => serde_json::Value::Null,
        Some(l) => {
            let regions: Vec<serde_json::Value> = l
                .regions
                .iter()
                .map(|r| {
                    json!({
                        "name": r.name,
                        "kind": kind_word(r.kind),
                        "n_points": r.mesh.points.len(),
                        "n_cells": r.quality.n_cells,
                        "n_internal_faces": r.mesh.neighbour.len(),
                        "n_boundary_faces": r.mesh.faces.len() - r.mesh.neighbour.len(),
                        "patches": r.mesh.patches.iter().map(|p| json!({
                            "name": p.name, "kind": p.type_name, "size": p.size,
                        })).collect::<Vec<serde_json::Value>>(),
                        "quality": quality_json(&r.quality),
                    })
                })
                .collect();
            let interfaces: Vec<serde_json::Value> = l
                .interfaces
                .iter()
                .map(|i| {
                    json!({
                        "regions": [l.regions[i.region_a].name, l.regions[i.region_b].name],
                        "patches": [i.patch_a, i.patch_b],
                        "faces": i.n_faces,
                    })
                })
                .collect();
            json!({
                "manifest": "mesh/regions.json",
                "regions": regions,
                "interfaces": interfaces,
            })
        }
    };
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
        "quality": quality_json(q),
        "layout": layout_json,
        "total_seconds": out.total_seconds,
        "config": serde_json::to_value(cfg).unwrap_or(serde_json::Value::Null),
        "identity": ident.to_json(),
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
        BodySpec, CastellationSpec, DistanceBand, DomainSpec, InputSpec, LayerSpec, OutputSpec,
        QualitySpec, RefinementBand, RefinementSpec, SnapSpec, SurfaceInput,
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

    /// `cube_config()` with the cube declared a body and no layers: the
    /// default `layers.patches = ["cube"]` names no REGION patch after the
    /// split (the cube region carries `cube_to_fluid` only), so a run with
    /// layers on is refused at the layer stage - a later test pins that
    /// refusal by name.
    fn body_config() -> AutomeshConfig {
        let mut cfg = cube_config();
        cfg.castellation.bodies =
            vec![BodySpec { name: "cube".to_string(), patches: vec!["cube".to_string()] }];
        cfg.layers = LayerSpec::default();
        cfg
    }

    /// The cube ON the cell planes: `[1, 2.5]^3` in a `[0,4]^3` domain at
    /// level 1 - a 0.5 m lattice puts the cube exactly on cell faces, so the
    /// body region is 3x3x3 cells with 54 interface faces, both sides.
    fn planar_config() -> AutomeshConfig {
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
            castellation: CastellationSpec {
                bodies: vec![BodySpec {
                    name: "cube".to_string(),
                    patches: vec!["cube".to_string()],
                }],
                ..CastellationSpec::default()
            },
            // The castellation keeps the cube ON the cell planes, so the
            // interface already lies on the surface and snap has nothing to
            // move - the default feature attraction and smoothing would
            // still slide the interface points along the surface (their own
            // gates hold, but the points leave the cell planes, which is
            // all the layer extrusion reads: §92.13). This config turns
            // that attraction off and keeps the walls snap-clean.
            snap: SnapSpec {
                feature_tolerance: 0.0,
                smoothing_passes: 0,
                ..SnapSpec::default()
            },
            layers: LayerSpec {
                patches: vec!["fluid_to_cube".to_string(), "cube_to_fluid".to_string()],
                n: 3,
                first_thickness: 0.02,
                normal_passes: 0,
                min_thickness: 0.0,
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

    /// `planar_config`'s surface: the same cube, wound outward, one patch.
    fn planar_surface() -> Surface {
        Surface::from_soup(box_soup([1.0; 3], [2.5; 3]), vec!["cube".to_string()])
            .expect("surface")
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
        assert_eq!(Stage::parse("split").unwrap(), Stage::Split);
        let err = Stage::parse("featur").unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("featur"), "{msg}");
        for spelling in ["octree", "castellate", "features", "snap", "split", "layers"] {
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
        assert_eq!(names, ["octree", "castellate", "snap", "split", "layers"]);
        // The no-body split is recorded and skipped, like an empty layer
        // stage.
        assert_eq!(out.stages[3].counts["skipped"], json!(true));
        for s in &out.stages {
            assert!(s.seconds >= 0.0, "{}: {}", s.stage.name(), s.seconds);
        }
        // The last stage's own gate is the one the run reports.
        assert!(out.quality.passed(), "the final mesh must pass the gate");
        // Every banner was emitted before its stage ran, and the elapsed
        // line of each stage follows its banner.
        assert!(
            lines.iter().any(|l| l == "=== stage 1/5  octree ==="),
            "{lines:?}"
        );
        assert!(
            lines.iter().any(|l| l.starts_with("--- layers:")),
            "{lines:?}"
        );
        assert!(
            lines.iter().any(|l| l == "split: skipped - castellation.bodies is empty"),
            "{lines:?}"
        );
        assert_eq!(
            lines.iter().filter(|l| l.starts_with("=== stage")).count(),
            5
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
        let ident = crate::automesher::identity::MeshIdentity::new(
            "ofgpu-automesher",
            std::path::Path::new(&cfg.output.case_dir),
            &cfg.output.name,
            None,
        );
        let s = summary_json(&cfg, "cube.automesher.json", &surf, &out, &ident);

        // The five stages, in order, each with its own counts spliced in.
        let stages = s["stages"].as_array().unwrap();
        let names: Vec<&str> =
            stages.iter().map(|v| v["stage"].as_str().unwrap()).collect();
        assert_eq!(names, ["octree", "castellate", "snap", "split", "layers"]);
        assert!(stages[0].get("n_leaves").is_some(), "{}", stages[0]);
        assert!(stages[1].get("wall_faces").is_some(), "{}", stages[1]);
        assert!(stages[3].get("skipped").is_some(), "{}", stages[3]);
        assert!(stages[4].get("n_layer_cells").is_some(), "{}", stages[4]);

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

        // A full run stops nowhere, and a no-body run wrote no layout: the
        // split stage ran recorded-and-skipped.
        assert_eq!(s["stopped_after"], serde_json::Value::Null);
        assert!(s["layout"].is_null(), "{}", s["layout"]);
        assert_eq!(s["tool"], "ofgpu-automesher");
        assert_eq!(s["config_path"], "cube.automesher.json");
        assert_eq!(s["surface"]["patches"].as_array().unwrap().len(), 1);

        // `config` is the parsed struct re-serialised: it parses BACK to the
        // config the run actually used.
        let back: AutomeshConfig = serde_json::from_value(s["config"].clone()).unwrap();
        assert_eq!(back, cfg);
    }

    #[test]
    fn summary_json_carries_the_identity_block() {
        let cfg = cube_config();
        let surf = cube_surface();
        let (out, _) = run_recording(&cfg, &surf, None);
        let ident = crate::automesher::identity::MeshIdentity::new(
            "ofgpu-automesher",
            std::path::Path::new("out"),
            "cube",
            Some("r_42".to_string()),
        );
        let s = summary_json(&cfg, "cube.automesher.json", &surf, &out, &ident);
        let i = &s["identity"];
        assert_eq!(i["schema"], 1);
        assert_eq!(i["tool"], "ofgpu-automesher");
        assert_eq!(i["run_id"], "r_42");
        let mid = i["mesh_id"].as_str().expect("mesh_id");
        assert!(mid.starts_with("m_") && mid.len() == 18, "{mid}");
        let wa = i["written_at"].as_str().expect("written_at");
        assert_eq!(wa.len(), 20, "{wa}");
        assert!(wa.ends_with('Z'), "{wa}");
        assert!(i["host"].is_string() || i["host"].is_null(), "{}", i["host"]);
        // The new block disturbed nothing: the mesh numbers and the config
        // round-trip are exactly what they were.
        assert_eq!(s["mesh"]["n_cells"].as_u64().unwrap() as usize, out.quality.n_cells);
        let back: AutomeshConfig = serde_json::from_value(s["config"].clone()).unwrap();
        assert_eq!(back, cfg);
    }

    /// M6 Run 2 requirement 1: the no-body full run is bit for bit what it
    /// was - the mesh leaves the pipeline unchanged through the split stage
    /// being recorded and skipped, and `out.layout` staying `None`.
    #[test]
    fn the_full_run_without_bodies_is_pinned() {
        let cfg = cube_config();
        let surf = cube_surface();
        let (out, _) = run_recording(&cfg, &surf, None);
        const PINNED: u64 = 0xaaab7b6052cce5be;
        let v = crate::automesher::castellate::tests::mesh_fingerprint(&out.mesh);
        eprintln!("fingerprint full-run-no-body = {v:#018x}");
        assert_eq!(v, PINNED);
        assert!(out.layout.is_none(), "a no-body run wrote no layout");
    }

    /// M6 Run 2 requirement 4: with a body declared the split stage runs,
    /// the fluid is regions[0], the body follows it, every region passed
    /// its own gate, and the run's single mesh is the FLUID's.
    #[test]
    fn a_declared_body_becomes_its_own_region() {
        let cfg = body_config();
        let surf = cube_surface();
        let (out, _) = run_recording(&cfg, &surf, None);
        let lay = out.layout.as_ref().expect("a body run splits");
        let names: Vec<&str> = lay.regions.iter().map(|r| r.name.as_str()).collect();
        assert_eq!(names, ["fluid", "cube"]);
        assert_eq!(lay.regions[0].kind, crate::cht::RegionKind::Fluid);
        assert_eq!(lay.regions[1].kind, crate::cht::RegionKind::Solid);
        for r in &lay.regions {
            assert!(r.quality.passed(), "region \"{}\" failed its gate", r.name);
        }
        assert_eq!(lay.regions[1].quality.n_cells, 8);
        assert_eq!(lay.interfaces.len(), 1);
        assert_eq!(lay.interfaces[0].n_faces, 24);
        let fluid_names: Vec<&str> =
            lay.regions[0].mesh.patches.iter().map(|p| p.name.as_str()).collect();
        assert_eq!(
            fluid_names,
            ["xMin", "xMax", "yMin", "yMax", "zMin", "zMax", "fluid_to_cube"]
        );
        let cube_names: Vec<&str> =
            lay.regions[1].mesh.patches.iter().map(|p| p.name.as_str()).collect();
        assert_eq!(cube_names, ["cube_to_fluid"]);
        // `out.mesh` is the fluid region, name for name, and the split
        // stage's row counts the two regions.
        let out_names: Vec<&str> =
            out.mesh.patches.iter().map(|p| p.name.as_str()).collect();
        assert_eq!(out_names, fluid_names);
        assert_eq!(out.stages[3].counts["n_regions"], json!(2));
    }

    /// M6 Run 2 requirement 5: the written layout is §C's tree and passes
    /// R1-R6 with bitwise-conformal interface pairs.
    #[test]
    fn the_layout_of_a_body_run_loads_and_pairs() {
        let cfg = body_config();
        let surf = cube_surface();
        let (out, _) = run_recording(&cfg, &surf, None);
        let lay = out.layout.as_ref().expect("a body run splits");
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        let mut root = std::env::temp_dir();
        root.push(format!("ofgpu-m6-{}-{nanos}", std::process::id()));
        let dir = root.join("mesh");
        let kinds: Vec<crate::cht::RegionKind> =
            lay.regions.iter().map(|r| r.kind).collect();
        let regions: Vec<(String, PolyMeshRaw)> = lay
            .regions
            .iter()
            .map(|r| (r.name.clone(), r.mesh.clone()))
            .collect();
        let manifest = crate::io::regions::write_layout(&dir, &regions, &kinds, &lay.interfaces, None)
            .expect("write_layout");
        let loaded = crate::io::regions::load(&dir.join("regions.json"))
            .expect("the written layout loads");
        println!(
            "pairing worsts: centroid {:.3e}, area {:.3e}, normal {:.3e}",
            loaded.pairing[0].worst_centroid,
            loaded.pairing[0].worst_area,
            loaded.pairing[0].worst_normal
        );
        assert!(loaded.pairing[0].worst_centroid <= 1e-12);
        assert!(loaded.pairing[0].worst_area <= 1e-12);
        assert!(loaded.pairing[0].worst_normal <= 1e-12);
        assert_eq!(loaded.manifest.regions[0].kind, "fluid");
        assert_eq!(loaded.manifest.regions[1].kind, "solid");
        assert_eq!(loaded.manifest.interfaces[0].faces, Some(24));
        assert_eq!(manifest.regions.len(), 2);
        let _ = std::fs::remove_dir_all(&root);
    }

    /// M6 Run 2 requirement 6: `-stopAfter split` stops with the layout and
    /// no layers; `-stopAfter snap` stops with ONE mesh and no layout even
    /// when bodies are declared - the split has not run.
    #[test]
    fn stopping_before_and_after_the_split() {
        let cfg = body_config();
        let surf = cube_surface();
        let (out, _) = run_recording(&cfg, &surf, Some(Stage::Snap));
        assert_eq!(out.stopped_after, Some(Stage::Snap));
        assert!(out.layout.is_none());
        assert_eq!(out.stages.len(), 3);
        // The kept body has no wall patch: the six domain patches only.
        assert_eq!(out.mesh.patches.len(), 6);

        let (out, _) = run_recording(&cfg, &surf, Some(Stage::Split));
        assert_eq!(out.stopped_after, Some(Stage::Split));
        assert!(out.layout.is_some());
        assert_eq!(out.stages.len(), 4);
        assert!(out.layout.as_ref().unwrap().regions[0].quality.passed());
    }

    /// M6 Run 2 requirement 7: (92.56) applies PER REGION on a split run - a
    /// key renames the patch in every region that has it, a key no region
    /// has is refused naming the regions' patch lists, and an interface
    /// patch is never renamed (docs/10 §C R3).
    #[test]
    fn patch_names_apply_per_region_and_never_to_an_interface() {
        let surf = cube_surface();
        let mut cfg = body_config();
        cfg.output.patch_names.insert("xMin".to_string(), "west".to_string());
        let (out, _) = run_recording(&cfg, &surf, None);
        let lay = out.layout.as_ref().expect("a body run splits");
        let fluid: Vec<&str> =
            lay.regions[0].mesh.patches.iter().map(|p| p.name.as_str()).collect();
        let cube: Vec<&str> =
            lay.regions[1].mesh.patches.iter().map(|p| p.name.as_str()).collect();
        assert!(fluid.contains(&"west"), "{fluid:?}");
        assert!(!fluid.contains(&"xMin"), "{fluid:?}");
        // The cube region has no box patch and is untouched.
        assert_eq!(cube, ["cube_to_fluid"]);

        let mut typo = body_config();
        typo.output.patch_names.insert("xmin".to_string(), "west".to_string());
        let err = run(&typo, &surf, None, &mut |_l: &str| {}).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("xmin"), "{msg}");
        assert!(msg.contains("names no patch of any region"), "{msg}");

        let mut iface = body_config();
        iface
            .output
            .patch_names
            .insert("fluid_to_cube".to_string(), "wall_cube".to_string());
        let err = run(&iface, &surf, None, &mut |_l: &str| {}).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("fluid_to_cube"), "{msg}");
        assert!(msg.contains("interface"), "{msg}");
    }

    /// M6 Run 2 requirement 8: with the cube ON the cell planes, both
    /// regions grow 3 x 54 layer cells and the layered layout still pairs
    /// bit for bit across the interface.
    #[test]
    fn layers_grow_on_both_sides_of_the_interface() {
        let cfg = planar_config();
        let surf = planar_surface();
        let (out, _) = run_recording(&cfg, &surf, None);
        let lay = out.layout.as_ref().expect("a body run splits");
        let rows = out.stages[4].counts["regions"].as_array().cloned().unwrap();
        for (i, r) in lay.regions.iter().enumerate() {
            assert!(r.quality.passed(), "region \"{}\" failed its gate", r.name);
            let row = &rows[i];
            println!("region \"{}\" layer row: {row}", r.name);
            assert_eq!(row["n_layer_cells"], json!(3 * 54), "{row}");
            for p in row["patches"].as_array().unwrap() {
                assert!(
                    p["dropped"].is_null(),
                    "region \"{}\" dropped layers: {p}",
                    r.name
                );
            }
        }
        // The layered layout still pairs bit for bit.
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        let mut root = std::env::temp_dir();
        root.push(format!("ofgpu-m6-layers-{}-{nanos}", std::process::id()));
        let dir = root.join("mesh");
        let kinds: Vec<crate::cht::RegionKind> = lay.regions.iter().map(|r| r.kind).collect();
        let regions: Vec<(String, PolyMeshRaw)> = lay
            .regions
            .iter()
            .map(|r| (r.name.clone(), r.mesh.clone()))
            .collect();
        crate::io::regions::write_layout(&dir, &regions, &kinds, &lay.interfaces, None)
            .expect("write_layout");
        let loaded = crate::io::regions::load(&dir.join("regions.json"))
            .expect("the layered layout loads");
        println!(
            "layered pairing worsts: centroid {:.3e}, area {:.3e}, normal {:.3e}",
            loaded.pairing[0].worst_centroid,
            loaded.pairing[0].worst_area,
            loaded.pairing[0].worst_normal
        );
        assert!(loaded.pairing[0].worst_centroid <= 1e-12);
        assert!(loaded.pairing[0].worst_area <= 1e-12);
        assert!(loaded.pairing[0].worst_normal <= 1e-12);
        let _ = std::fs::remove_dir_all(&root);
    }

    /// M6 Run 2 requirement 9: a `layers.patches` entry no region carries is
    /// refused naming the regions' patch lists; an entry one region carries
    /// grows layers THERE and the other region's row says skipped.
    #[test]
    fn a_layer_patch_that_no_region_has_is_refused() {
        let surf = planar_surface();
        // No region has the STL patch name: the split names are the layout's
        // contract, and the refusal lists what each region does carry.
        let mut bad = planar_config();
        bad.layers.patches = vec!["cube".to_string()];
        let err = run(&bad, &surf, None, &mut |_l: &str| {}).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("cube"), "{msg}");
        assert!(msg.contains("fluid_to_cube"), "{msg}");
        assert!(msg.contains("cube_to_fluid"), "{msg}");

        // Fluid side only: the fluid grows, the cube region is skipped.
        let mut fluid_only = planar_config();
        fluid_only.layers.patches = vec!["fluid_to_cube".to_string()];
        let (out, _) = run_recording(&fluid_only, &surf, None);
        let rows = out.stages[4].counts["regions"].as_array().cloned().unwrap();
        assert_eq!(rows[0]["name"], json!("fluid"));
        assert_eq!(rows[0]["n_layer_cells"], json!(3 * 54), "{}", rows[0]);
        assert_eq!(rows[1]["name"], json!("cube"));
        assert_eq!(rows[1]["skipped"], json!(true), "{}", rows[1]);

        // Mirror: the cube side grows, the fluid region is skipped.
        let mut cube_only = planar_config();
        cube_only.layers.patches = vec!["cube_to_fluid".to_string()];
        let (out, _) = run_recording(&cube_only, &surf, None);
        let rows = out.stages[4].counts["regions"].as_array().cloned().unwrap();
        assert_eq!(rows[0]["skipped"], json!(true), "{}", rows[0]);
        assert_eq!(rows[1]["n_layer_cells"], json!(3 * 54), "{}", rows[1]);
        for p in rows[1]["patches"].as_array().unwrap() {
            println!("cube-side layer patch row: {p}");
        }
        assert!(
            rows[1]["patches"]
                .as_array()
                .unwrap()
                .iter()
                .all(|p| p["dropped"].is_null()),
            "{}",
            rows[1]
        );
    }

    /// M6 Run 2 requirement 10: (92.57) gains `layout` - `null` on a no-body
    /// run, the written layout's shape on a body run, with `mesh` and
    /// `quality` the fluid region's.
    #[test]
    fn the_summary_carries_the_layout() {
        let cfg = body_config();
        let surf = cube_surface();
        let (out, _) = run_recording(&cfg, &surf, None);
        let ident = crate::automesher::identity::MeshIdentity::new(
            "ofgpu-automesher",
            std::path::Path::new(&cfg.output.case_dir),
            &cfg.output.name,
            None,
        );
        let s = summary_json(&cfg, "cube.automesher.json", &surf, &out, &ident);
        let l = &s["layout"];
        assert!(l.is_object(), "{l}");
        assert_eq!(l["manifest"], json!("mesh/regions.json"));
        assert_eq!(l["regions"].as_array().unwrap().len(), 2);
        assert_eq!(l["regions"][0]["name"], json!("fluid"));
        assert_eq!(l["regions"][1]["name"], json!("cube"));
        assert_eq!(l["regions"][1]["quality"]["passed"], json!(true));
        assert_eq!(l["interfaces"][0]["faces"], json!(24));
        assert_eq!(
            s["mesh"]["n_cells"].as_u64().unwrap(),
            l["regions"][0]["quality"]["n_cells"].as_u64().unwrap()
        );
        // `config` round-trips, `bodies` included.
        let back: AutomeshConfig = serde_json::from_value(s["config"].clone()).unwrap();
        assert_eq!(back, cfg);
        assert!(serde_json::to_string(&s["config"]).unwrap().contains("\"bodies\""));
    }

    /// The summary reports the numbers a fidelity flag reads - the snap
    /// stage's finest cell, both residuals over it, the boundary-only
    /// pinned count and one area row per patch, and the octree stage's
    /// gate verdict - while every stage's own behaviour stays untouched.
    #[test]
    fn the_summary_reports_the_snap_ratios_and_the_octree_verdict() {
        let cfg = cube_config();
        let surf = cube_surface();
        let (out, _) = run_recording(&cfg, &surf, None);
        let ident = crate::automesher::identity::MeshIdentity::new(
            "ofgpu-automesher",
            std::path::Path::new(&cfg.output.case_dir),
            &cfg.output.name,
            None,
        );
        let s = summary_json(&cfg, "cube.automesher.json", &surf, &out, &ident);
        let st = &s["stages"];
        // Octree: the leaf mesh measured, reported and not refused on.
        assert!(st[0]["gate_passed"].is_boolean(), "{}", st[0]);
        assert!(st[0]["gate_passed"].as_bool().unwrap());
        let mno = st[0]["max_non_orth_deg"].as_f64().unwrap();
        eprintln!("octree max_non_orth_deg {mno}");
        assert!(mno.is_finite() && mno >= 0.0, "max_non_orth_deg {mno}");
        // Snap: this config's finest cell is base 1 over 2^1.
        assert_eq!(st[2]["h_f_m"].as_f64().unwrap(), 0.5);
        let p99 = st[2]["p99_over_h"].as_f64().unwrap().to_bits();
        assert_eq!(p99, (st[2]["p99_residual"].as_f64().unwrap() / 0.5).to_bits());
        let max = st[2]["max_over_h"].as_f64().unwrap().to_bits();
        assert_eq!(max, (st[2]["max_residual"].as_f64().unwrap() / 0.5).to_bits());
        let npb = st[2]["n_pinned_boundary"].as_u64().unwrap();
        assert!(
            npb <= st[2]["n_boundary_points"].as_u64().unwrap(),
            "boundary-only pinned {npb} over the boundary points"
        );
        let rows = st[2]["area_ratio"].as_array().unwrap();
        assert_eq!(rows.len(), 1);
        let row = &rows[0];
        assert_eq!(row["name"].as_str().unwrap(), "cube");
        for key in [
            "name",
            "stl_area_m2",
            "castellated_area_m2",
            "snapped_area_m2",
            "castellated_ratio",
            "ratio",
        ] {
            assert!(row.get(key).is_some(), "missing {key} in {row}");
        }
        let r = row["ratio"].as_f64().unwrap();
        eprintln!("cube area ratio {r}");
        assert!(0.9 < r && r < 1.1, "ratio {r} not in (0.9, 1.1)");
    }

    /// The region rows carry the same `area` bookkeeping as the
    /// single-mesh path: on the on-plane cube every region's layer rows sum
    /// to the STL's area for the patch to 1e-12 relative, and every patch
    /// row publishes its three beta shares at `TAU_GE_BETAS`.
    #[test]
    fn the_region_layer_rows_sum_to_the_cube_area() {
        let cfg = planar_config();
        let surf = planar_surface();
        let (out, _) = run_recording(&cfg, &surf, None);
        let rows = out.stages[4].counts["regions"].as_array().cloned().unwrap();
        let a_stl = surf.patch_area[0];
        for r in &rows {
            assert!(r.get("skipped").is_none(), "{r}");
            let mut sum = 0.0;
            for p in r["patches"].as_array().unwrap() {
                sum += p["area"].as_f64().expect("area is a number");
                let shares = p["area_frac_tau_ge"].as_array().expect("tau shares");
                assert_eq!(shares.len(), 3, "{p}");
                for (i, s) in shares.iter().enumerate() {
                    assert_eq!(
                        s["beta"].as_f64().unwrap(),
                        crate::automesher::layers::TAU_GE_BETAS[i] as f64,
                        "{p}"
                    );
                    let f = s["area_frac"].as_f64().expect("area_frac is a number");
                    let full = p["full_area_frac"].as_f64().unwrap();
                    assert!(full <= f && f <= 1.0 + 1e-12, "{p}");
                }
            }
            eprintln!(
                "region \"{}\" layer rows sum to {sum:.9} (STL area {a_stl:.9})",
                r["name"].as_str().unwrap()
            );
            assert!(((sum - a_stl) / a_stl).abs() <= 1e-12, "{sum} vs {a_stl}");
        }
    }

    /// The single-mesh summary's layer row carries `area` and the three
    /// beta shares beside every key it had before; the cube here straddles
    /// the cell planes, so its row area is printed beside the STL's, not
    /// compared with it.
    #[test]
    fn the_summary_layer_row_carries_its_area_and_tau_shares() {
        let cfg = cube_config();
        let surf = cube_surface();
        let (out, _) = run_recording(&cfg, &surf, None);
        let ident = crate::automesher::identity::MeshIdentity::new(
            "ofgpu-automesher",
            std::path::Path::new(&cfg.output.case_dir),
            &cfg.output.name,
            None,
        );
        let s = summary_json(&cfg, "cube.automesher.json", &surf, &out, &ident);
        let r = &s["stages"][4]["patches"][0];
        assert_eq!(r["name"].as_str().unwrap(), "cube");
        let area = r["area"].as_f64().expect("area is a number");
        assert!(area.is_finite() && area > 0.0, "{area}");
        let a_stl = surf.patch_area[0];
        eprintln!("cube row area {area:.9} vs STL area {a_stl:.9}");
        let shares = r["area_frac_tau_ge"].as_array().expect("tau shares");
        assert_eq!(shares.len(), 3, "{r}");
        let betas = crate::automesher::layers::TAU_GE_BETAS;
        for (beta, s) in betas.iter().zip(shares.iter()) {
            assert_eq!(s["beta"].as_f64().unwrap(), *beta as f64, "{r}");
        }
        for key in [
            "name",
            "n_layers",
            "n_faces",
            "full_area_frac",
            "mean_frac",
            "t1_requested",
            "t1_mean",
            "t1_min",
            "dropped",
        ] {
            assert!(r.get(key).is_some(), "missing {key} in {r}");
        }
    }

    /// `cube_config()` end to end: the mesh the run emits is the one it
    /// emitted when the layer stage's castellated goldens were written, bit
    /// for bit, hashed as they are.
    #[test]
    fn the_cube_config_run_is_golden() {
        const GOLDEN: &str = "3b257be84dc343e6b66432eca0ead5d7e150203a2c5984001e28a4b3fd95d5e3";
        let (out, _) = run_recording(&cube_config(), &cube_surface(), None);
        let got = crate::automesher::layers::tests::castellated_goldens::mesh_sha256(&out.mesh);
        eprintln!("golden cube_config_end_to_end = {got}");
        assert_eq!(got, GOLDEN);
    }
}
