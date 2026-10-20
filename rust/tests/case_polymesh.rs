// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.
// Provenance: see PROVENANCE.md. No GPL-licensed source was consulted.

//! DST-11's `mesh.kind: "polymesh"` (SPEC-LIT §158.10), run in BOTH builds on
//! the host alone (no GPU): a JSONC case lowers onto a polyMesh read from
//! disk - the same mesh, the same patch names, classes and kinds a cartesian
//! case of the same box gives, bit for bit - and every setting a polyMesh
//! cannot take from a case is refused by its key and the section.
//!
//! The specification was informed by a review of FSCDustFoam (Huang &
//! Lipatnikov, GPL) and no code was copied. No GPL-licensed source was
//! consulted.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use ofgpu::io::case_json::{read_case_jsonc, read_case_jsonc_str};
use ofgpu::io::polymesh::write_poly_mesh_raw;

/// The 4x3x2 box of the tests, six named slots.
const BOX_MESH: &str = r#"{
    "kind": "cartesian",
    "bounds": { "min": [0, 0, 0], "max": [4, 3, 2] },
    "cells": [4, 3, 2],
    "boundaries": { "xmin": "inlet", "xmax": "outlet", "ymin": "floor",
                    "ymax": "lid", "zmin": "sideA", "zmax": "sideB" }
}"#;

/// Rules covering the six slots with four different classes: `patch`,
/// `patch`, `wall` and `symmetry`.
const RULES: &str = r#"[
    { "match": "inlet", "kind": "inlet", "U": { "type": "fixedValue", "value": [1, 0, 0] } },
    { "match": "outlet", "kind": "open" },
    { "match": "side.*", "kind": "symmetry" },
    { "match": ".*", "kind": "wall" }
]"#;

/// A whole case around `mesh` and `rules`, with `extra` top-level keys
/// (each starting with a comma) appended.
fn case_text(mesh: &str, rules: &str, extra: &str) -> String {
    format!(
        r#"{{
        "name": "polymeshTest",
        "mesh": {mesh},
        "physics": {{
            "gravity": [0, 0, 0],
            "fluid": {{ "nu": 1e-5, "Pr": 0.71, "Prt": 0.85, "TRef": 293.15 }},
            "buoyancy": "densityRatio"
        }},
        "patches": {rules},
        "initial": {{ "U": [0, 0, 0], "p": 0.0 }},
        "numerics": {{
            "algorithm": {{ "kind": "SIMPLE" }},
            "ddt": "steadyState",
            "div": {{ "default": "Gauss upwind" }},
            "grad": "Gauss linear",
            "laplacian": {{ "snGrad": "corrected", "nonOrthogonalCorrectors": 0 }},
            "solvers": []
        }},
        "run": {{ "endTime": 1.0, "deltaT": 1.0 }}{extra}
    }}"#
    )
}

/// A fresh, empty directory under the system temp dir, unique to this
/// process and call.
fn temp_dir(tag: &str) -> PathBuf {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    let d = std::env::temp_dir().join(format!("ofgpu_case_polymesh_{tag}_{}_{n}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).expect("the temp dir");
    d
}

/// A path as JSON text: forward slashes, which every platform's reader takes.
fn json_path(p: &Path) -> String {
    p.display().to_string().replace('\\', "/")
}

/// The polymesh mesh block naming `dir`, with `stated` more keys.
fn poly_mesh(dir: &Path, stated: &str) -> String {
    format!(r#"{{ "kind": "polymesh", "path": "{}"{stated} }}"#, json_path(dir))
}

/// Lower `text`, which must be refused; the refusal's message.
fn refused(text: &str) -> String {
    match read_case_jsonc_str(text, "refusal").and_then(|c| c.lower()) {
        Ok(_) => panic!("the case lowered but must be refused:\n{text}"),
        Err(e) => e.to_string(),
    }
}

/// The cartesian box lowered, and its raw mesh written under a fresh temp
/// dir's `constant/polyMesh`; the dir.
fn written_box(tag: &str) -> PathBuf {
    let cart = read_case_jsonc_str(&case_text(BOX_MESH, RULES, ""), "box")
        .and_then(|c| c.lower())
        .expect("the cartesian box lowers");
    let raw = cart.build_raw_mesh().expect("the raw mesh");
    let dir = temp_dir(tag);
    write_poly_mesh_raw(&dir.join("constant/polyMesh"), &raw).expect("the polyMesh writes");
    dir
}

/// **SPEC-LIT §158.10 (A).** A polymesh case builds the mesh it names: the
/// cartesian box's polyMesh, written to disk and named by `mesh.path`, gives
/// the same cells, internal faces, patch names (in order), kinds and type
/// strings, and bitwise-equal volumes and centres, as the cartesian case
/// itself - and the same lowered boundary conditions. A relative `mesh.path`
/// resolves against the case file's directory.
#[test]
fn a_polymesh_case_builds_the_mesh_it_names() {
    // 1. the cartesian case, 2. its mesh on disk.
    let cart_case = read_case_jsonc_str(&case_text(BOX_MESH, RULES, ""), "box").expect("the box parses");
    let text = serde_json::to_string(&cart_case).expect("the case serialises");
    for key in ["\"bounds\"", "\"cells\"", "\"boundaries\""] {
        assert!(text.contains(key), "a cartesian case still serialises {key}");
    }
    assert!(!text.contains("\"path\""), "a cartesian case serialises no path");
    let cart = cart_case.lower().expect("the cartesian box lowers");
    assert!(cart.poly_mesh.is_none(), "a cartesian case carries no polyMesh");
    let raw = cart.build_raw_mesh().expect("the raw mesh");
    let dir = temp_dir("builds");
    write_poly_mesh_raw(&dir.join("constant/polyMesh"), &raw).expect("the polyMesh writes");

    // 3. the polymesh case, the same rules.
    let poly_case = read_case_jsonc_str(&case_text(&poly_mesh(&dir, ""), RULES, ""), "polymesh")
        .expect("the polymesh case parses");
    let poly = poly_case.lower().expect("the polymesh case lowers");
    assert!(poly.poly_mesh.is_some(), "the polymesh case carries its polyMesh");
    assert_eq!(poly.block, ofgpu::blockgen::BlockSpec::default(), "a polymesh case has no block");
    assert!(poly.windows.is_empty(), "a polymesh case has no windows");

    // 4. the two host meshes.
    let (a, b) = (cart.build_host_mesh().expect("cartesian mesh"), poly.build_host_mesh().expect("polymesh mesh"));
    assert_eq!(a.n_cells, 24);
    assert_eq!(b.n_cells, a.n_cells);
    assert_eq!(b.n_internal_faces, a.n_internal_faces);
    assert_eq!(b.patches.len(), a.patches.len());
    for (pa, pb) in a.patches.iter().zip(&b.patches) {
        assert_eq!(pb.name, pa.name, "patch names, in order");
        assert_eq!(pb.kind, pa.kind, "patch {} kind", pa.name);
        assert_eq!(pb.type_name, pa.type_name, "patch {} type", pa.name);
    }
    let names: Vec<&str> = b.patches.iter().map(|p| p.name.as_str()).collect();
    assert_eq!(names, ["inlet", "outlet", "floor", "lid", "sideA", "sideB"]);
    let types: Vec<&str> = b.patches.iter().map(|p| p.type_name.as_str()).collect();
    assert_eq!(types, ["patch", "patch", "wall", "wall", "symmetry", "symmetry"]);
    for (i, (va, vb)) in a.v.iter().zip(&b.v).enumerate() {
        assert_eq!(vb.to_bits(), va.to_bits(), "cell {i} volume");
    }
    for (i, (ca, cb)) in a.c.iter().zip(&b.c).enumerate() {
        assert_eq!(cb.x.to_bits(), ca.x.to_bits(), "cell {i} centre x");
        assert_eq!(cb.y.to_bits(), ca.y.to_bits(), "cell {i} centre y");
        assert_eq!(cb.z.to_bits(), ca.z.to_bits(), "cell {i} centre z");
    }
    assert_eq!(poly.u_field, cart.u_field, "the lowered U boundary conditions");
    assert_eq!(poly.p_field, cart.p_field, "the lowered p boundary conditions");

    // 5. the same case as a FILE, with a RELATIVE path: against the file's
    // own directory (the test process's directory has no polyMesh), both
    // beside the mesh and from a sub-directory.
    let file = dir.join("case.jsonc");
    std::fs::write(&file, case_text(r#"{ "kind": "polymesh", "path": "." }"#, RULES, "")).expect("the case file");
    let lowered = read_case_jsonc(&file).and_then(|c| c.lower()).expect("a relative path lowers");
    assert_eq!(lowered.build_host_mesh().expect("mesh").n_cells, a.n_cells);
    let sub = dir.join("sub");
    std::fs::create_dir_all(&sub).expect("sub dir");
    let file = sub.join("case.jsonc");
    std::fs::write(&file, case_text(r#"{ "kind": "polymesh", "path": ".." }"#, RULES, "")).expect("the case file");
    let lowered = read_case_jsonc(&file).and_then(|c| c.lower()).expect("a parent-relative path lowers");
    assert_eq!(lowered.build_host_mesh().expect("mesh").n_cells, a.n_cells);
    // An absolute path in a file is used as it stands.
    let file = sub.join("abs.jsonc");
    std::fs::write(&file, case_text(&poly_mesh(&dir, ""), RULES, "")).expect("the case file");
    let lowered = read_case_jsonc(&file).and_then(|c| c.lower()).expect("an absolute path lowers");
    assert_eq!(lowered.build_raw_mesh().expect("raw mesh").faces.len(), raw.faces.len());
    let _ = std::fs::remove_dir_all(&dir);
}

/// **SPEC-LIT §158.10 (B).** Every refusal names its key and the section:
/// each setting a polyMesh cannot take (`mesh.bounds`, `mesh.cells`,
/// `mesh.boundaries`, `mesh.regions`, `mesh.grading`, `mesh.cyclic`,
/// `mesh.wedge`, `mesh.sector`, `motion`), a missing `mesh.path`, a directory
/// without a polyMesh, a cyclic patch in the file, a patch a `wedge` rule
/// governs - and, on the cartesian side, a `path` and a missing `bounds`,
/// `cells` or `boundaries`. A polyMesh patch no rule governs gets the
/// existing unmatched-rule refusal.
#[test]
fn polymesh_refusals_name_the_key() {
    let dir = written_box("refusals");

    // Each stated beside "polymesh": the key and the section.
    let stated: [(&str, &str); 8] = [
        (r#", "bounds": { "min": [0, 0, 0], "max": [4, 3, 2] }"#, "mesh.bounds"),
        (r#", "cells": [4, 3, 2]"#, "mesh.cells"),
        (
            r#", "boundaries": { "xmin": "a", "xmax": "b", "ymin": "c", "ymax": "d", "zmin": "e", "zmax": "f" }"#,
            "mesh.boundaries",
        ),
        (
            r#", "regions": [ { "name": "w", "on": "xmin", "shape": { "kind": "box", "min": [0, 0, 0], "max": [0, 1, 1] } } ]"#,
            "mesh.regions",
        ),
        (r#", "grading": { "x": { "expansion": 2.0 } }"#, "mesh.grading"),
        (r#", "cyclic": [ { "a": "inlet", "b": "outlet", "transform": "translate" } ]"#, "mesh.cyclic"),
        (r#", "wedge": { "angle": 5 }"#, "mesh.wedge"),
        (r#", "sector": { "angle": 60, "fineRadius": 0.05, "growth": 1.05 }"#, "mesh.sector"),
    ];
    for (more, key) in stated {
        let msg = refused(&case_text(&poly_mesh(&dir, more), RULES, ""));
        assert!(msg.contains(key), "{key} is not named: {msg}");
        assert!(msg.contains("158.10"), "{key}: the section is not named: {msg}");
    }

    // A motion block.
    let msg = refused(&case_text(
        &poly_mesh(&dir, ""),
        RULES,
        r#", "motion": { "patches": [ { "rule": "fixed", "patch": "floor" } ] }"#,
    ));
    assert!(msg.contains("motion"), "motion is not named: {msg}");
    assert!(msg.contains("158.10"), "{msg}");

    // A missing path.
    let msg = refused(&case_text(r#"{ "kind": "polymesh" }"#, RULES, ""));
    assert!(msg.contains("mesh.path"), "{msg}");
    assert!(msg.contains("158.10"), "{msg}");

    // A directory without a readable polyMesh: the reader's error, prefixed.
    let empty = temp_dir("empty");
    let msg = refused(&case_text(&poly_mesh(&empty, ""), RULES, ""));
    assert!(msg.contains("mesh.path"), "{msg}");
    assert!(msg.contains("polyMesh"), "the reader's own error is carried: {msg}");
    assert!(msg.contains("158.10"), "{msg}");
    let _ = std::fs::remove_dir_all(&empty);

    // A patch that is cyclic in the file: a cartesian case with a cyclic
    // pair, written, then named as a polymesh.
    let cyclic_mesh = BOX_MESH.replace(
        "\"boundaries\"",
        "\"cyclic\": [ { \"a\": \"inlet\", \"b\": \"outlet\", \"transform\": \"translate\" } ], \"boundaries\"",
    );
    let cyclic_case = read_case_jsonc_str(
        &case_text(&cyclic_mesh, r#"[ { "match": ".*", "kind": "wall" } ]"#, ""),
        "cyclic box",
    )
    .and_then(|c| c.lower())
    .expect("the cyclic box lowers");
    let cyclic_dir = temp_dir("cyclic");
    write_poly_mesh_raw(
        &cyclic_dir.join("constant/polyMesh"),
        &cyclic_case.build_raw_mesh().expect("the cyclic raw mesh"),
    )
    .expect("the cyclic polyMesh writes");
    let msg = refused(&case_text(&poly_mesh(&cyclic_dir, ""), RULES, ""));
    assert!(msg.contains("mesh.path"), "{msg}");
    assert!(msg.contains("inlet"), "the cyclic patch is named: {msg}");
    assert!(msg.contains("158.10"), "{msg}");
    let _ = std::fs::remove_dir_all(&cyclic_dir);

    // A patch a `wedge` rule governs.
    let msg = refused(&case_text(
        &poly_mesh(&dir, ""),
        r#"[ { "match": "side.*", "kind": "wedge" }, { "match": ".*", "kind": "wall" } ]"#,
        "",
    ));
    assert!(msg.contains("patches"), "{msg}");
    assert!(msg.contains("sideA"), "the patch is named: {msg}");
    assert!(msg.contains("158.10"), "{msg}");

    // A polyMesh patch no rule governs: the existing refusal.
    let msg = refused(&case_text(
        &poly_mesh(&dir, ""),
        r#"[ { "match": "inlet", "kind": "inlet", "U": { "type": "fixedValue", "value": [1, 0, 0] } },
              { "match": "outlet", "kind": "open" } ]"#,
        "",
    ));
    assert!(msg.contains("no rule matches boundary patch"), "{msg}");
    assert!(msg.contains("floor"), "the unmatched patch is named: {msg}");

    // The cartesian side: a path, and each of bounds, cells, boundaries missing.
    let msg = refused(&case_text(&BOX_MESH.replace("\"cells\"", "\"path\": \"somewhere\", \"cells\""), RULES, ""));
    assert!(msg.contains("mesh.path"), "{msg}");
    assert!(msg.contains("polymesh"), "{msg}");
    assert!(msg.contains("158.10"), "{msg}");
    let without = |key: &str, cut: &str| {
        let mesh = BOX_MESH.replace(cut, "");
        assert_ne!(mesh, BOX_MESH, "the box text still holds {cut:?}");
        let msg = refused(&case_text(&mesh, RULES, ""));
        assert!(msg.contains(key), "{key} is not named: {msg}");
        assert!(msg.contains("required"), "{key}: {msg}");
        assert!(msg.contains("158.10"), "{key}: {msg}");
    };
    without("mesh.bounds", "\"bounds\": { \"min\": [0, 0, 0], \"max\": [4, 3, 2] },");
    without("mesh.cells", "\"cells\": [4, 3, 2],");
    let boundaries_start = BOX_MESH.find("\"boundaries\"").expect("the box has boundaries");
    let boundaries_end = BOX_MESH.rfind('}').expect("the box closes");
    let cut = &BOX_MESH[boundaries_start..boundaries_end];
    let mesh = BOX_MESH.replace(cut, "\"cyclic\": []");
    let msg = refused(&case_text(&mesh, RULES, ""));
    assert!(msg.contains("mesh.boundaries"), "{msg}");
    assert!(msg.contains("158.10"), "{msg}");
    let _ = std::fs::remove_dir_all(&dir);
}
