// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.
// Provenance: see PROVENANCE.md. No GPL-licensed source was consulted.
//
// The DST-07 sector-mesh gate (SPEC-LIT §157.3), written ONCE and included
// twice - by `tests/dust_gatemesh.rs`, which runs it as an integration test
// in BOTH builds, and by the lib tests of `src/dust/gatemesh.rs`, where the
// same five tests sit beside the code. Host only: no card is needed.
//
// Provenance: ORIGINAL - the exact frustum geometry is elementary.
// No GPL-licensed source was consulted.

use ofgpu::dust::gatemesh::{build_sector_mesh, cell_radii, check_sector, sector_radii, SectorSpec};
use ofgpu::io::case_json::{read_case_jsonc, read_case_jsonc_str};

/// The Leeds sector: R 176.4 mm, the equal-volume radius of H20's 0.023
/// m³; 0.125 mm cells (n_fine 400) out to 50 mm, growth 1.05 beyond; the
/// 90 deg sector the case ships (the apex pyramid's V^(1/3) = 1.10 dr,
/// §157.8); the generator's own five patch names, the outer square
/// the vessel wall and the four sides symmetry.
fn leeds() -> SectorSpec {
    SectorSpec {
        radius: 0.1764,
        fine_radius: 0.05,
        n_fine: 400,
        growth: 1.05,
        angle_deg: 90.0,
        patch_name: ["outer", "ymin", "ymax", "zmin", "zmax"].map(String::from),
        patch_type: ["wall", "symmetry", "symmetry", "symmetry", "symmetry"].map(String::from),
    }
}

/// A coarse sector: R 1 m, r_f 0.5 m, five uniform cells then growth 1.1,
/// the full 90 deg.
fn coarse() -> SectorSpec {
    SectorSpec {
        radius: 1.0,
        fine_radius: 0.5,
        n_fine: 5,
        growth: 1.1,
        angle_deg: 90.0,
        patch_name: ["outer", "ymin", "ymax", "zmin", "zmax"].map(String::from),
        patch_type: ["wall", "symmetry", "symmetry", "symmetry", "symmetry"].map(String::from),
    }
}

fn rel(a: f64, b: f64) -> f64 {
    (a - b).abs() / b.abs()
}

/// The Leeds case text, with the mesh block's `bounds`, `cells` and any
/// other mesh keys (a leading comma included) taken as arguments - each
/// refusal of §157.3 is then one literal. `extra_mesh` carries the
/// `sector` block itself where the caller wants one.
fn leeds_like(bounds: &str, cells: &str, extra_mesh: &str) -> String {
    format!(
        r#"{{
  "name": "leedsBomb",
  "mesh": {{ "kind": "cartesian", "bounds": {bounds}, "cells": {cells},
    "boundaries": {{ "xmin": "apex", "xmax": "vessel", "ymin": "sideYmin",
      "ymax": "sideYmax", "zmin": "sideZmin", "zmax": "sideZmax" }}{extra_mesh} }},
  "physics": {{ "gravity": [0, 0, 0],
    "fluid": {{ "nu": 1.3636e-5, "Pr": 0.7, "Prt": 0.4, "TRef": 328 }},
    "buoyancy": "densityRatio" }},
  "turbulence": {{ "kind": "RAS", "model": "kOmegaSST",
    "wallFunctions": {{ "kappa": 0.41, "E": 9.8 }} }},
  "patches": [
    {{ "match": "apex", "kind": "symmetry" }},
    {{ "match": "vessel", "kind": "wall",
      "U": {{ "type": "fixedValue", "value": [0, 0, 0] }},
      "p": {{ "type": "zeroGradient" }}, "T": {{ "type": "zeroGradient" }},
      "k": {{ "type": "kqRWallFunction", "value": 3.94 }},
      "omega": {{ "type": "omegaWallFunction", "value": 1101.33 }} }},
    {{ "match": "side.*", "kind": "symmetry" }}
  ],
  "initial": {{ "U": [0, 0, 0], "T": 328, "p": 0.0, "k": 3.94, "omega": 1101.33 }},
  "numerics": {{ "algorithm": {{ "kind": "PIMPLE", "correctors": 2 }}, "ddt": "Euler",
    "div": {{ "default": "Gauss upwind", "div(phi,U)": "Gauss upwind",
      "div(phi,T)": "bounded Gauss upwind" }},
    "grad": "Gauss linear",
    "laplacian": {{ "snGrad": "corrected", "nonOrthogonalCorrectors": 0 }},
    "relaxation": {{ "U": 1.0, "p": 1.0, "T": 1.0 }},
    "solvers": [
      {{ "match": "p", "solver": "PBiCGStab", "preconditioner": "DIC", "tolerance": 1e-13, "relTol": 0.0, "maxIter": 2000 }},
      {{ "match": "U", "solver": "PBiCGStab", "preconditioner": "diagonal", "tolerance": 1e-13, "relTol": 0.0, "maxIter": 2000 }},
      {{ "match": "T", "solver": "PBiCGStab", "preconditioner": "diagonal", "tolerance": 1e-13, "relTol": 0.0, "maxIter": 2000 }},
      {{ "match": "c", "solver": "PBiCGStab", "preconditioner": "diagonal", "tolerance": 1e-13, "relTol": 0.0, "maxIter": 2000 }},
      {{ "match": "k|omega", "solver": "PBiCGStab", "preconditioner": "diagonal", "tolerance": 1e-13, "relTol": 0.0, "maxIter": 2000 }}
    ] }},
  "run": {{ "endTime": 0.03, "deltaT": 5e-6 }},
  "combustion": {{ "model": "FSC", "A": 0.4, "PrT": 0.4,
    "ignition": {{ "point": [0, 0, 0], "sigmaR": 2.5e-4, "W0": 1e13, "t0": 1e-3 }} }},
  "dust": {{ "thermo": "prescribed", "cp": 1011.1, "Wu": 0.0328, "Wb": 0.0297,
    "Tb": 1500, "SL": 0.12,
    "sst": {{ "sust": {{ "kAmb": 3.94, "omegaAmb": 1101.33 }} }} }}
}}"#
    )
}

/// The shipped case's own `mesh.sector` block as `extra_mesh`.
fn sector_extra() -> &'static str {
    r#", "sector": { "angle": 90, "fineRadius": 0.05, "growth": 1.05 }"#
}

/// **SPEC-LIT §157.3 (D).** The Leeds radii: 481 stations, 480 cells, the
/// uniform part bitwise `i * dr` with `x_400 = r_f` and `x_480 = R` exact,
/// the graded spacings the stated first and last, strictly increasing
/// throughout, and g = 1 uniform beyond r_f.
#[test]
fn sector_radii_are_the_stated_grading() {
    let l = leeds();
    let x = sector_radii(&l).unwrap();
    assert_eq!(x.len(), 481, "{} radii", x.len());
    assert_eq!(x.len() - 1, 480, "{} cells", x.len() - 1);
    let dr = 0.05 / 400.0;
    for (i, &xi) in x.iter().enumerate().take(400) {
        assert_eq!(xi.to_bits(), (i as f64 * dr).to_bits(), "x_{i} is not i*dr");
    }
    assert_eq!(x[400].to_bits(), 0.05f64.to_bits(), "x_400 is not r_f");
    assert_eq!(x[480].to_bits(), 0.1764f64.to_bits(), "x_480 is not R");
    for (i, w) in x.windows(2).enumerate() {
        assert!(w[1] > w[0], "not strictly increasing at x_{i}");
    }
    let first = x[401] - x[400];
    let last = x[480] - x[479];
    println!("first graded spacing {first:.9e}, last {last:.9e}");
    assert!(rel(first, 1.301444080e-4) <= 1e-9, "first graded {first}");
    assert!(rel(last, 6.142994674e-3) <= 1e-9, "last graded {last}");
    // g = 1: uniform beyond r_f, every graded spacing the fine dr's scaled
    // self to 1e-12 relative.
    let mut u = leeds();
    u.n_fine = 5;
    u.growth = 1.0;
    let xu = sector_radii(&u).unwrap();
    let spacings: Vec<f64> = xu.windows(2).map(|w| w[1] - w[0]).collect();
    let graded = &spacings[5..];
    let first_graded = graded[0];
    for &s in graded {
        assert!(rel(s, first_graded) <= 1e-12, "g = 1 spacing {s} != {first_graded}");
    }
}

/// **SPEC-LIT §157.3 (D).** Every cell closes to 1e-12 relative and every
/// radial square's area is `4 a² x²` to 1e-12 - on Leeds and on the coarse
/// sector both, the measured values printed.
#[test]
fn sector_cells_close_and_areas_grow_as_r_squared() {
    for (tag, spec, cells) in
        [("leeds", leeds(), 480usize), ("coarse", coarse(), 9usize)]
    {
        let c = check_sector(&spec).unwrap();
        println!(
            "{tag}: closure_max = {:e}, area_r2_max = {:e}, cells = {}",
            c.closure_max, c.area_r2_max, c.cells
        );
        assert!(c.closure_max <= 1e-12, "{tag} closure {}", c.closure_max);
        assert!(c.area_r2_max <= 1e-12, "{tag} area/r2 {}", c.area_r2_max);
        assert_eq!(c.cells, cells, "{tag} cell count");
    }
}

/// **SPEC-LIT §157.3 (D).** The host mesh's own geometry is the exact
/// frusta's: every volume `(4a²/3)(x_b³ - x_a³)`, every centroid x of
/// `cell_radii` with y and z zero, the sum `(4a²/3) R³`, and the five
/// patches `outer` then the four sides.
#[test]
fn sector_host_mesh_matches_the_exact_frusta() {
    let l = leeds();
    let m = build_sector_mesh(&l).unwrap();
    let g = m.geom64();
    assert_eq!(m.n_cells, 480);
    let a = (90f64.to_radians() * 0.5).tan();
    let x = sector_radii(&l).unwrap();
    let xc = cell_radii(&l).unwrap();
    let mut total = 0.0;
    for c in 0..m.n_cells {
        let (xa, xb) = (x[c], x[c + 1]);
        let want = 4.0 * a * a / 3.0 * (xb.powi(3) - xa.powi(3));
        assert!(rel(g.v[c], want) <= 1e-12, "cell {c}: v = {} != {want}", g.v[c]);
        assert!(rel(g.c[c].x, xc[c]) <= 1e-12, "cell {c}: cx = {} != {}", g.c[c].x, xc[c]);
        assert!(g.c[c].y.abs() <= 1e-12 * l.radius, "cell {c}: cy = {}", g.c[c].y);
        assert!(g.c[c].z.abs() <= 1e-12 * l.radius, "cell {c}: cz = {}", g.c[c].z);
        total += g.v[c];
    }
    let want_total = 4.0 * a * a / 3.0 * l.radius.powi(3);
    assert!(rel(total, want_total) <= 1e-12, "sum {total} != {want_total}");
    let names: Vec<&str> = m.patches.iter().map(|p| p.name.as_str()).collect();
    assert_eq!(names, ["outer", "ymin", "ymax", "zmin", "zmax"]);
    let sizes: Vec<usize> = m.patches.iter().map(|p| p.size).collect();
    assert_eq!(sizes, [1, 480, 480, 480, 480]);
}

/// **SPEC-LIT §157.3 (D).** The shipped Leeds case lowers: its
/// `mesh.sector` is the Leeds spec under the case's own patch names and
/// classes, its host mesh is the 480-cell sector with five patches, no
/// lowered U/p/T field carries an `apex` entry, and the dust block came
/// along.
#[test]
fn sector_case_lowers() {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../cases/dust/leedsBomb.jsonc");
    let case = read_case_jsonc(&path).unwrap();
    let lowered = case.lower().unwrap();
    let mut want = leeds();
    want.patch_name = [
        "vessel".to_string(),
        "sideYmin".to_string(),
        "sideYmax".to_string(),
        "sideZmin".to_string(),
        "sideZmax".to_string(),
    ];
    assert_eq!(lowered.sector.as_ref(), Some(&want));
    let m = lowered.build_host_mesh().unwrap();
    assert_eq!(m.n_cells, 480);
    assert_eq!(m.patches.len(), 5);
    assert!(!lowered.u_field.boundary.contains_key("apex"), "U carries apex");
    assert!(!lowered.p_field.boundary.contains_key("apex"), "p carries apex");
    if let Some(t) = &lowered.t_field {
        assert!(!t.boundary.contains_key("apex"), "T carries apex");
    }
    assert!(lowered.dust.is_some(), "the dust block did not lower");
}

/// **SPEC-LIT §157.3 (D).** Every refusal names its JSON path and the
/// section; a case without `mesh.sector` lowers with `sector == None`.
#[test]
fn sector_refusals_name_the_setting() {
    let leeds_bounds = r#"{ "min": [0, 0, 0], "max": [0.1764, 0, 0] }"#;
    let refusals: [(&str, &str, &str, &str); 12] = [
        // (bounds, cells, extra mesh keys, the needle the refusal names)
        (r#"{ "min": [0.001, 0, 0], "max": [0.1764, 0, 0] }"#, "[400, 1, 1]", sector_extra(), "mesh.bounds.min"),
        (r#"{ "min": [0, 0, 0], "max": [0.1764, 0.01, 0] }"#, "[400, 1, 1]", sector_extra(), "mesh.bounds.max"),
        (r#"{ "min": [0, 0, 0], "max": [0.1764, 0, 0.01] }"#, "[400, 1, 1]", sector_extra(), "mesh.bounds.max"),
        (leeds_bounds, "[400, 1, 1]", r#", "sector": { "angle": 60, "fineRadius": 0.2, "growth": 1.05 }"#, "mesh.bounds.max[0]"),
        (leeds_bounds, "[400, 1, 1]", r#", "sector": { "angle": 60, "fineRadius": 0, "growth": 1.05 }"#, "mesh.sector.fineRadius"),
        (leeds_bounds, "[400, 2, 1]", sector_extra(), "mesh.cells"),
        (leeds_bounds, "[400, 1, 2]", sector_extra(), "mesh.cells"),
        (leeds_bounds, "[0, 1, 1]", sector_extra(), "mesh.cells"),
        (leeds_bounds, "[400, 1, 1]", r#", "sector": { "angle": 60, "fineRadius": 0.05, "growth": 1.5 }"#, "mesh.sector.growth"),
        (leeds_bounds, "[400, 1, 1]", r#", "sector": { "angle": 120, "fineRadius": 0.05, "growth": 1.05 }"#, "mesh.sector.angle"),
        (leeds_bounds, "[400, 1, 1]", r#", "sector": { "angle": 0, "fineRadius": 0.05, "growth": 1.05 }"#, "mesh.sector.angle"),
        (leeds_bounds, "[400, 1, 1]", r#", "sector": { "angle": 60, "fineRadius": 0.05, "growth": 1.05 }, "wedge": { "angle": 5 }"#, "mesh.wedge"),
    ];
    // The refusals live in `JsonCase::lower`, so parse AND lower.
    let refused = |text: &str, tag: &str| -> String {
        match read_case_jsonc_str(text, tag).and_then(|c| c.lower()) {
            Ok(_) => panic!("{tag}: the case lowered but must be refused"),
            Err(e) => e.to_string(),
        }
    };
    for (i, (bounds, cells, extra, needle)) in refusals.iter().enumerate() {
        let msg = refused(&leeds_like(bounds, cells, extra), "refusal");
        assert!(msg.contains("157.3"), "refusal {i} misses the section: {msg}");
        assert!(msg.contains(needle), "refusal {i} does not name {needle}: {msg}");
    }
    // The rest of the refusal list, each its own case text.
    for (extra, needle) in [
        (r#", "regions": [ { "name": "win", "on": "xmax",
            "shape": { "kind": "box", "min": [0.1, 0, 0], "max": [0.15, 0, 0] } } ]"#, "mesh.regions"),
        (r#", "cyclic": [ { "a": "sideYmin", "b": "sideYmax", "transform": "translate" } ]"#, "mesh.cyclic"),
        (r#", "grading": { "x": { "expansion": 2.0 } }"#, "mesh.grading"),
    ] {
        // These three ride BESIDE the sector block - that is the refusal.
        let extra = format!("{}{}", sector_extra(), extra);
        let msg = refused(&leeds_like(leeds_bounds, "[400, 1, 1]", &extra), "refusal");
        assert!(msg.contains("157.3"), "{needle} misses the section: {msg}");
        assert!(msg.contains(needle), "{needle} not named: {msg}");
    }
    let motion = r#"  "motion": { "patches": [ { "rule": "fixed", "patch": "vessel" } ] },"#;
    let with_motion = leeds_like(leeds_bounds, "[400, 1, 1]", sector_extra())
        .replace("  \"combustion\"", &format!("{motion}\n  \"combustion\""));
    let msg = refused(&with_motion, "motion");
    assert!(msg.contains("157.3"), "motion refusal misses the section: {msg}");
    assert!(msg.contains("motion"), "motion not named: {msg}");
    // A case WITHOUT mesh.sector lowers exactly as before, sector None.
    let case = read_case_jsonc_str(&leeds_like(leeds_bounds, "[400, 1, 1]", ""), "plain").unwrap();
    assert!(case.lower().unwrap().sector.is_none());
}





