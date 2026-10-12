// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.
// Provenance: see PROVENANCE.md. No GPL-licensed source was consulted.

//! SPEC-LIT 119.3 - the `wedge` case-JSON kind, through the public API in
//! BOTH builds: the 119.3 JSON example lowers onto the 119.0 mesh with the
//! case's own boundary names (the pipe keeps no `axis` entry, the annulus
//! keeps its inner patch), the derived wedge rides on it at 1e-12, and the
//! refusal list names its JSON path. Geometry is compared in f64 through
//! `geom64` (the derivation is f64 in both builds, SPEC-LIT 118.2), so the
//! bitwise contracts hold in both. No `cfg_attr(... ignore)` anywhere.
//!
//! No GPL-licensed source was consulted.

use ofgpu::blockgen::{build_wedge_mesh, GradedAxis, WedgeSpec};
use ofgpu::io::case_json::read_case_jsonc_str;
use ofgpu::PatchKind;

/// SPEC-LIT 119.3's JSON example, completed to a full case, with the
/// numbers the tests vary as parameters. `angle: None` is no `mesh.wedge`
/// block at all.
fn wedge_case(
    x_bounds: (f64, f64),
    cells: [u32; 3],
    angle: Option<f64>,
    patches: &str,
) -> Result<ofgpu::io::case_json::JsonCase, ofgpu::Error> {
    let wedge_block = match angle {
        Some(a) => format!(r#""wedge": {{ "angle": {a} }},"#),
        None => String::new(),
    };
    let text = format!(
        r#"{{
                "name": "wedgeCaseTest",
                "mesh": {{
                    "kind": "cartesian",
                    "bounds": {{ "min": [{}, 0, 0], "max": [{}, 0, 1] }},
                    "cells": [{}, {}, {}],
                    {wedge_block}
                    "boundaries": {{
                        "xmin": "axis", "xmax": "wall", "ymin": "back",
                        "ymax": "front", "zmin": "inlet", "zmax": "outlet"
                    }}
                }},
                "physics": {{
                    "gravity": [0,0,0],
                    "fluid": {{ "nu": 0.01, "Pr": 0.71, "Prt": 0.85, "TRef": 293.15 }},
                    "buoyancy": "densityRatio"
                }},
                "patches": {patches},
                "initial": {{ "U": [0,0,0], "p": 0.0 }},
                "numerics": {{
                    "algorithm": {{ "kind": "SIMPLE" }},
                    "ddt": "steadyState",
                    "div": {{ "default": "Gauss upwind" }},
                    "grad": "Gauss linear",
                    "laplacian": {{ "snGrad": "corrected", "nonOrthogonalCorrectors": 0 }},
                    "solvers": []
                }},
                "run": {{ "endTime": 1.0, "deltaT": 1.0 }}
            }}"#,
        x_bounds.0, x_bounds.1, cells[0], cells[1], cells[2]
    );
    read_case_jsonc_str(&text, "119.3 wedge case test")
}

const CATCH_ALL: &str =
    r#"[ { "match": "(back|front)", "kind": "wedge" }, { "match": ".*", "kind": "wall" } ]"#;

/// Test 1 through the public API: the 119.3 example lowers to the 119.0
/// pipe, the two wedge sides carry the case's own names, and the geometry
/// is BITWISE the generator's own (compared in f64 through `geom64`).
#[test]
fn a_wedge_case_lowers_to_the_119_0_mesh() {
    let lowered = wedge_case((0.0, 0.5), [8, 1, 4], Some(5.0), CATCH_ALL)
        .expect("parse")
        .lower()
        .expect("lowers");
    assert!(lowered.wedge.is_some());

    let hm = lowered.build_host_mesh().expect("the wedge mesh builds");
    assert_eq!(hm.n_cells, 32);
    let names: Vec<&str> = hm.patches.iter().map(|p| p.name.as_str()).collect();
    assert_eq!(names, vec!["wall", "inlet", "outlet", "back", "front"]);
    assert_eq!(hm.patches[3].kind, PatchKind::Wedge);
    assert_eq!(hm.patches[4].kind, PatchKind::Wedge);
    let theta = hm.wedge.as_ref().expect("119.1's derivation ran").theta;
    assert!(
        (theta - 5f64.to_radians()).abs() <= 1e-12,
        "theta {theta} != {}",
        5f64.to_radians()
    );

    let eq = WedgeSpec {
        r: GradedAxis { lo: 0.0, hi: 0.5, n: 8, ..GradedAxis::default() },
        z: GradedAxis { lo: 0.0, hi: 1.0, n: 4, ..GradedAxis::default() },
        theta_deg: 5.0,
        patch_type: WedgeSpec::default().patch_type,
    };
    let want = build_wedge_mesh(&eq).expect("builds");
    let (g, gw) = (hm.geom64(), want.geom64());
    assert_eq!(g.v.len(), gw.v.len());
    for (i, (a, b)) in g.v.iter().zip(gw.v.iter()).enumerate() {
        assert_eq!(a.to_bits(), b.to_bits(), "volume of cell {i}");
    }
    for (i, (a, b)) in g.c.iter().zip(gw.c.iter()).enumerate() {
        assert_eq!(a.x.to_bits(), b.x.to_bits(), "centre x of cell {i}");
        assert_eq!(a.y.to_bits(), b.y.to_bits(), "centre y of cell {i}");
        assert_eq!(a.z.to_bits(), b.z.to_bits(), "centre z of cell {i}");
    }
    // A pipe has no axis patch: no lowered field carries the xmin name.
    assert!(!lowered.u_field.boundary.contains_key("axis"));
    assert!(!lowered.p_field.boundary.contains_key("axis"));
}

/// Test 2 through the public API: the annulus keeps its inner patch, named
/// as `xmin` names it, and the wedge pair closes the list.
#[test]
fn a_wedge_annulus_keeps_its_inner_patch() {
    let lowered = wedge_case((1.0, 2.0), [8, 1, 4], Some(5.0), CATCH_ALL)
        .expect("parse")
        .lower()
        .expect("lowers");
    let hm = lowered.build_host_mesh().expect("builds");
    assert_eq!(hm.patches.len(), 6);
    assert_eq!(hm.patches[0].name, "axis");
    let names: Vec<&str> = hm.patches.iter().map(|p| p.name.as_str()).collect();
    assert_eq!(names, vec!["axis", "wall", "inlet", "outlet", "back", "front"]);
}

/// Test 4 through the public API: every refusal names its JSON path and the
/// section.
#[test]
fn every_wedge_refusal_names_its_path() {
    let angle = |a: f64| {
        wedge_case((0.0, 0.5), [8, 1, 4], Some(a), CATCH_ALL)
            .expect("parse")
            .lower()
    };
    let plain = |patches: &'static str| {
        wedge_case((0.0, 0.5), [8, 1, 4], Some(5.0), patches)
            .expect("parse")
            .lower()
    };
    let mut refusals: Vec<(String, String, &'static str)> = Vec::new();
    let push = |refusals: &mut Vec<(String, String, &'static str)>,
                label: String,
                r: Result<ofgpu::io::case_json::LoweredCase, ofgpu::Error>,
                path: &'static str| {
        let msg = match r {
            Err(e) => e.to_string(),
            Ok(_) => panic!("{label}: expected a refusal"),
        };
        refusals.push((label, msg, path));
    };

    push(
        &mut refusals,
        "wedge rule without mesh.wedge".to_string(),
        wedge_case((0.0, 0.5), [8, 1, 4], None, CATCH_ALL)
            .expect("parse")
            .lower(),
        "mesh.boundaries.ymin",
    );
    push(
        &mut refusals,
        "wedge rule on xmin".to_string(),
        plain(
            r#"[ { "match": "axis", "kind": "wedge" },
                { "match": "(back|front)", "kind": "wedge" },
                { "match": ".*", "kind": "wall" } ]"#,
        ),
        "mesh.boundaries.xmin",
    );
    push(
        &mut refusals,
        "ymin not a wedge rule".to_string(),
        plain(r#"[ { "match": ".*", "kind": "wall" } ]"#),
        "mesh.boundaries.ymin",
    );
    push(
        &mut refusals,
        "ymax not a wedge rule".to_string(),
        plain(
            r#"[ { "match": "back", "kind": "wedge" },
                { "match": ".*", "kind": "wall" } ]"#,
        ),
        "mesh.boundaries.ymax",
    );
    push(
        &mut refusals,
        "cells[1] = 2".to_string(),
        wedge_case((0.0, 0.5), [8, 2, 4], Some(5.0), CATCH_ALL)
            .expect("parse")
            .lower(),
        "mesh.cells[1]",
    );
    push(
        &mut refusals,
        "bounds min x -0.5".to_string(),
        wedge_case((-0.5, 0.5), [8, 1, 4], Some(5.0), CATCH_ALL)
            .expect("parse")
            .lower(),
        "mesh.bounds",
    );
    for a in [0.0, 180.0, -1.0] {
        push(
            &mut refusals,
            format!("angle {a}"),
            angle(a),
            "mesh.wedge.angle",
        );
    }
    push(
        &mut refusals,
        "a motion block".to_string(),
        read_case_jsonc_str(
            r#"{{
                "name": "wedgeMotionTest",
                "mesh": {{
                    "kind": "cartesian",
                    "bounds": {{ "min": [0, 0, 0], "max": [0.5, 0, 1] }},
                    "cells": [8, 1, 4],
                    "wedge": {{ "angle": 5 }},
                    "boundaries": {{
                        "xmin": "axis", "xmax": "wall", "ymin": "back",
                        "ymax": "front", "zmin": "inlet", "zmax": "outlet"
                    }}
                }},
                "physics": {{
                    "gravity": [0,0,0],
                    "fluid": {{ "nu": 0.01, "Pr": 0.71, "Prt": 0.85, "TRef": 293.15 }},
                    "buoyancy": "densityRatio"
                }},
                "patches": [ { "match": "(back|front)", "kind": "wedge" },
                             { "match": ".*", "kind": "wall" } ],
                "initial": {{ "U": [0,0,0], "p": 0.0 }},
                "numerics": {{
                    "algorithm": {{ "kind": "SIMPLE" }},
                    "ddt": "steadyState",
                    "div": {{ "default": "Gauss upwind" }},
                    "grad": "Gauss linear",
                    "laplacian": {{ "snGrad": "corrected", "nonOrthogonalCorrectors": 0 }},
                    "solvers": []
                }},
                "run": {{ "endTime": 1.0, "deltaT": 1.0 }},
                "motion": {{ "patches": [] }}
            }}"#,
            "119.3 wedge motion refusal test",
        )
        .and_then(|c| c.lower()),
        "motion",
    );

    assert!(refusals.len() >= 8);
    for (label, msg, path) in &refusals {
        assert!(msg.contains(path), "{label}: no path {path} in {msg}");
        assert!(msg.contains("119.3"), "{label}: no section in {msg}");
    }
}
