// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.
// Provenance: see PROVENANCE.md. No GPL-licensed source was consulted.

//! DST-11's placeholder-vessel generator, run in BOTH builds on the host
//! alone (no GPU): the vessel dimensions by (158.10a) of SPEC-LIT 158.10,
//! (158.10b) of SPEC-LIT 158.10 and (158.10c) of SPEC-LIT 158.10, the closed
//! wall-shell STL whose signed volume is V_solid of (158.10e) of SPEC-LIT
//! 158.10, the automesher config, the refusals of the membrane plane
//! (158.10d) of SPEC-LIT 158.10, and the sealed-region walk. The geometry is
//! the labelled PLACEHOLDER of `cases/dust/rembeVented.geometry.json` - H22
//! publishes no vessel dimensions (SPEC-LIT 151.7 item 9) - and nothing here
//! is compared with data.
//!
//! The specification was informed by a review of FSCDustFoam (Huang &
//! Lipatnikov, GPL) and no code was copied. No GPL-licensed source was
//! consulted.

use std::collections::HashMap;
use std::path::Path;

use ofgpu::dust::gatemesh::{
    automesher_config, membrane_plane, parse_vessel_geometry, read_vessel_geometry, seal_reach,
    vessel_stl, VesselGeometry,
};
use ofgpu::io::case_json::read_case_jsonc_str;
use ofgpu::types::DVec3;

/// The shipped placeholder file's path and text.
fn shipped_path() -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../cases/dust/rembeVented.geometry.json")
}

fn shipped_text() -> String {
    std::fs::read_to_string(shipped_path()).expect("cases/dust/rembeVented.geometry.json reads")
}

fn shipped() -> VesselGeometry {
    read_vessel_geometry(&shipped_path()).expect("the shipped placeholder geometry parses")
}

/// `got` equals `want` to 1e-14 relative, or to 1e-14 absolute where `want`
/// is zero.
fn close(got: f64, want: f64) -> bool {
    if want == 0.0 {
        got.abs() <= 1.0e-14
    } else {
        ((got - want) / want).abs() <= 1.0e-14
    }
}

/// **SPEC-LIT 158.10 (A).** The placeholder's dimensions by (158.10a) of
/// SPEC-LIT 158.10, (158.10b) of SPEC-LIT 158.10 and (158.10c) of SPEC-LIT
/// 158.10, each to 1e-14 relative; the membrane plane of (158.10d) of
/// SPEC-LIT 158.10 is the base-grid plane x = 5.425; and both probes sit
/// inside the cavity at the quarter points the file states.
#[test]
fn placeholder_dims() {
    let g = shipped();
    assert!(g.placeholder, "the shipped geometry is labelled a placeholder");
    assert!(g.label.starts_with("PLACEHOLDER"), "label: {}", g.label);
    let d = g.dims();
    for (name, got, want) in [
        ("R", d.r, 0.970873525930309),
        ("L", d.l, 3.883494103721236),
        ("r_v", d.r_vent, 0.3989422804014327),
        ("x1", d.x1, 5.325),
        ("x0", d.x0, 1.4415058962787644),
    ] {
        assert!(close(got, want), "{name}: {got} != {want}");
    }
    let plane = membrane_plane(&g).expect("the membrane plane of the placeholder");
    assert!((plane - 5.425).abs() <= 1.0e-12, "membrane plane {plane}");

    assert_eq!(g.probes.len(), 2);
    let (p1, p2) = (&g.probes[0], &g.probes[1]);
    assert_eq!((p1.name.as_str(), p2.name.as_str()), ("P1", "P2"));
    for p in [p1, p2] {
        assert!(
            d.in_cavity(DVec3::new(p.point[0], p.point[1], p.point[2])),
            "probe {} is outside the cavity",
            p.name
        );
    }
    let want1 = [d.x0 + d.l / 4.0, 0.0, 3.0 + d.r / 2.0];
    let want2 = [d.x0 + 3.0 * d.l / 4.0, 0.0, 3.0 - d.r / 2.0];
    for (p, want) in [(p1, want1), (p2, want2)] {
        for k in 0..3 {
            assert!(
                close(p.point[k], want[k]),
                "probe {} component {k}: {} != {}",
                p.name,
                p.point[k],
                want[k]
            );
        }
    }
    // A point on the wall is not in the cavity, and nor is one past the vent end.
    assert!(!d.in_cavity(DVec3::new(d.x0, 0.0, 3.0)), "x0 is the wall, not the cavity");
    assert!(!d.in_cavity(DVec3::new(d.x1 + 0.01, 0.0, 3.0)), "past x1 is the front wall");
    assert!(!d.in_cavity(DVec3::new(3.0, d.r, 3.0)), "rho = R is the wall");
}

/// The text after `vertex ` of one vertex line - the key a vertex is shared
/// by, since every vertex is printed from one f64 triple.
fn vertex_key(line: &str) -> Option<&str> {
    line.trim().strip_prefix("vertex ").map(str::trim)
}

fn parse3(key: &str) -> [f64; 3] {
    let v: Vec<f64> = key.split_whitespace().map(|t| t.parse().expect("a vertex number")).collect();
    assert_eq!(v.len(), 3, "vertex {key}");
    [v[0], v[1], v[2]]
}

/// **SPEC-LIT 158.10 (B).** The STL is ONE closed, outward-oriented solid:
/// every directed edge appears exactly once and its reverse exactly once, the
/// signed volume `sum a . (b x c) / 6` is positive and equals V_solid of
/// (158.10e) of SPEC-LIT 158.10 to 1e-12 relative, and each stored normal is
/// the winding's own unit normal.
#[test]
fn vessel_stl_is_closed_and_oriented() {
    let g = shipped();
    let text = vessel_stl(&g).expect("the STL builds");
    assert!(text.starts_with("solid vessel\n"), "header");
    assert!(text.trim_end().ends_with("endsolid vessel"), "footer");

    let mut facets: Vec<[String; 3]> = Vec::new();
    let mut normals: Vec<[f64; 3]> = Vec::new();
    let mut cur: Vec<String> = Vec::new();
    for line in text.lines() {
        if let Some(rest) = line.trim().strip_prefix("facet normal ") {
            normals.push(parse3(rest));
        }
        if let Some(k) = vertex_key(line) {
            cur.push(k.to_string());
            if cur.len() == 3 {
                facets.push([cur[0].clone(), cur[1].clone(), cur[2].clone()]);
                cur.clear();
            }
        }
    }
    assert!(cur.is_empty(), "a facet with fewer than three vertices");
    let n = g.vessel.segments;
    assert_eq!(facets.len(), 12 * n, "seven pieces of N-gon rings make 12 N facets");
    assert_eq!(normals.len(), facets.len());

    // Every directed edge exactly once, and its reverse exactly once.
    let mut edges: HashMap<(&str, &str), usize> = HashMap::new();
    for f in &facets {
        for e in 0..3 {
            let (a, b) = (f[e].as_str(), f[(e + 1) % 3].as_str());
            assert_ne!(a, b, "a degenerate edge {a}");
            *edges.entry((a, b)).or_insert(0) += 1;
        }
    }
    for (&(a, b), &count) in &edges {
        assert_eq!(count, 1, "the directed edge {a} -> {b} appears {count} times");
        assert_eq!(
            edges.get(&(b, a)).copied(),
            Some(1),
            "the edge {a} -> {b} has no single reverse: the solid is not closed"
        );
    }

    // The signed volume, and each stored normal against its winding.
    let mut vol6 = 0.0f64;
    for (f, stored) in facets.iter().zip(&normals) {
        let (a, b, c) = (parse3(&f[0]), parse3(&f[1]), parse3(&f[2]));
        let cross = |u: [f64; 3], w: [f64; 3]| {
            [u[1] * w[2] - u[2] * w[1], u[2] * w[0] - u[0] * w[2], u[0] * w[1] - u[1] * w[0]]
        };
        vol6 += a[0] * cross(b, c)[0] + a[1] * cross(b, c)[1] + a[2] * cross(b, c)[2];
        let ab = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
        let ac = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
        let nn = cross(ab, ac);
        let m = (nn[0] * nn[0] + nn[1] * nn[1] + nn[2] * nn[2]).sqrt();
        let dot = (nn[0] * stored[0] + nn[1] * stored[1] + nn[2] * stored[2]) / m;
        assert!(dot > 1.0 - 1.0e-12, "the stored normal is not the winding's: dot = {dot}");
    }
    let volume = vol6 / 6.0;
    assert!(volume > 0.0, "the solid is inside out: signed volume {volume}");

    // (158.10e) of SPEC-LIT 158.10.
    let d = g.dims();
    let area = |rho: f64| (n as f64) / 2.0 * rho * rho * (2.0 * std::f64::consts::PI / n as f64).sin();
    let v_solid = area(d.r + d.t) * (d.l + 2.0 * d.t) - area(d.r) * d.l - area(d.r_vent) * d.t;
    assert!(
        ((volume - v_solid) / v_solid).abs() <= 1.0e-12,
        "signed volume {volume} != V_solid {v_solid}"
    );
    assert!(
        ((v_solid - 6.82925363456151) / 6.82925363456151).abs() <= 1.0e-12,
        "V_solid {v_solid} is not the placeholder's 6.82925363456151"
    );
}

/// **SPEC-LIT 158.10 (C).** The automesher config parses with the
/// automesher's own reader, and carries the geometry's extent, base size and
/// max level, and the seed point by the formula - bitwise.
#[test]
fn automesher_config_reads_back() {
    let g = shipped();
    let value = automesher_config(&g, "vessel.stl", "case");
    let text = serde_json::to_string_pretty(&value).expect("the config serialises");
    let cfg = ofgpu::automesher::read_config_str(&text, "rembe cfg").expect("the config reads back");
    let e = g.domain.extent;
    assert_eq!(cfg.domain.extent.map(f64::to_bits), e.map(f64::to_bits), "extent");
    assert_eq!(cfg.domain.base_size.to_bits(), g.domain.base_size.to_bits(), "base size");
    assert_eq!(cfg.refinement.max_level, g.domain.level, "max level");
    let want = [
        e[0] + 0.9137 * (e[1] - e[0]),
        e[2] + 0.5137 * (e[3] - e[2]),
        e[4] + 0.5137 * (e[5] - e[4]),
    ];
    let seed = cfg.castellation.seed_point.expect("a seed point");
    assert_eq!(seed.map(f64::to_bits), want.map(f64::to_bits), "seed point {seed:?} != {want:?}");
    assert_eq!(cfg.input.surfaces.len(), 1);
    assert_eq!(cfg.input.surfaces[0].path, "vessel.stl");
    assert_eq!(cfg.output.case_dir, "case");
    assert_eq!(cfg.layers.n, 0, "no layers");
    assert_eq!(cfg.refinement.levels.len(), 1);
    assert_eq!(cfg.refinement.levels[0].patch, "vessel");
    assert_eq!(cfg.refinement.levels[0].bands.len(), 2);
}

/// **SPEC-LIT 158.10 (D).** Each way to break the placeholder is refused
/// naming its key: the plane off the base grid, a wall thinner than two
/// finest cells, an extent that cuts the shell, too few segments - and an
/// unknown key, which the reader names.
#[test]
fn generator_refusals() {
    let base = shipped_text();
    let edit = |from: &str, to: &str| -> String {
        assert!(base.contains(from), "the shipped text no longer holds {from:?}");
        base.replacen(from, to, 1)
    };
    let geometry = |text: &str| parse_vessel_geometry(text, "refusal").expect("the edit still parses");
    let plane_refusal = |text: String, key: &str| {
        let g = geometry(&text);
        let msg = match membrane_plane(&g) {
            Ok(p) => panic!("{key}: the plane {p} was accepted but must be refused"),
            Err(e) => e.to_string(),
        };
        assert!(msg.contains(key), "{key} is not named: {msg}");
        assert!(msg.contains("158.10"), "{key}: the section is not named: {msg}");
    };

    // ventPlaneX 5.4 is off the base grid (h0 = 0.775).
    plane_refusal(edit("\"ventPlaneX\": 5.425", "\"ventPlaneX\": 5.4"), "vessel.ventPlaneX");
    // shell 0.15 < 2 h_f = 0.19375.
    plane_refusal(edit("\"shell\": 0.2", "\"shell\": 0.15"), "vessel.shell");
    // z_hi = 4.0 cuts the shell's top at z = 4.1709.
    plane_refusal(edit("0.0, 6.355]", "0.0, 4.0]"), "domain.extent");

    // segments 4 is no cylinder.
    let msg = match parse_vessel_geometry(&edit("\"segments\": 48", "\"segments\": 4"), "refusal") {
        Ok(_) => panic!("segments 4 parsed"),
        Err(e) => e.to_string(),
    };
    assert!(msg.contains("vessel.segments"), "{msg}");
    assert!(msg.contains("158.10"), "{msg}");

    // An unknown key inside `vessel` is named.
    let msg = match parse_vessel_geometry(&edit("\"segments\": 48", "\"segments\": 48, \"colour\": 1"), "refusal") {
        Ok(_) => panic!("an unknown key parsed"),
        Err(e) => e.to_string(),
    };
    assert!(msg.contains("colour"), "{msg}");
}

/// A 4x1x1 box as a cartesian JSONC case.
fn block_case() -> String {
    r#"{
        "name": "sealTest",
        "mesh": {
            "kind": "cartesian",
            "bounds": { "min": [0, 0, 0], "max": [4, 1, 1] },
            "cells": [4, 1, 1],
            "boundaries": { "xmin": "xa", "xmax": "xb", "ymin": "ya", "ymax": "yb", "zmin": "za", "zmax": "zb" }
        },
        "physics": {
            "gravity": [0, 0, 0],
            "fluid": { "nu": 1e-5, "Pr": 0.71, "Prt": 0.85, "TRef": 293.15 },
            "buoyancy": "densityRatio"
        },
        "patches": [ { "match": ".*", "kind": "wall" } ],
        "initial": { "U": [0, 0, 0], "p": 0.0 },
        "numerics": {
            "algorithm": { "kind": "SIMPLE" },
            "ddt": "steadyState",
            "div": { "default": "Gauss upwind" },
            "grad": "Gauss linear",
            "laplacian": { "snGrad": "corrected", "nonOrthogonalCorrectors": 0 },
            "solvers": []
        },
        "run": { "endTime": 1.0, "deltaT": 1.0 }
    }"#
    .to_string()
}

/// **SPEC-LIT 158.10 (E).** On a 4x1x1 block the closed membrane at the
/// internal face between cells 1 and 2 keeps cells 0 and 1 apart from 2 and
/// 3; with no membrane all four cells are reached.
#[test]
fn seal_reach_on_a_block() {
    let lowered = read_case_jsonc_str(&block_case(), "sealTest")
        .and_then(|c| c.lower())
        .expect("the block case lowers");
    let hm = lowered.build_host_mesh().expect("the block builds");
    assert_eq!(hm.n_cells, 4);
    let membrane = (0..hm.n_internal_faces)
        .find(|&f| hm.owner[f] as usize == 1 && hm.neighbour[f] as usize == 2)
        .expect("an internal face between cells 1 and 2");
    assert_eq!(seal_reach(&hm, &[membrane], 0), vec![0, 1]);
    assert_eq!(seal_reach(&hm, &[membrane], 3), vec![2, 3], "the other side, from the far seed");
    assert_eq!(seal_reach(&hm, &[], 0), vec![0, 1, 2, 3]);
}
