// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.
// Provenance: see PROVENANCE.md. No GPL-licensed source was consulted.

// TET-12's gates of SPEC-LIT §116.16 and §116.18, written once and included
// both by src/automesher/driver.rs (as automesher::driver::tests) and by
// tests/automesher_tet.rs, so both builds run them. The config is the
// example tools/automesher/examples/box_sphere_tet.json itself, and the
// surface is the sphere `make_box_sphere_stl.py --outward` writes, built
// here the way the script builds it. Every tet-path number is evaluated in
// f64 (§116.2), so the gates are class D (counts, names, refusals) and A
// (the dihedral bars, §116.18), with the same bars in both builds. Under
// `single` §92.3's G2 sums f32 area vectors and refuses the mesh, the miss
// §116.16 records; the test asserts that cause and prints MISSED.
//
// Provenance: ORIGINAL. No GPL-licensed source was consulted.

use ofgpu::automesher::driver::{run, run_tet, tet_summary_json, TET_STAGES};
use ofgpu::automesher::identity::MeshIdentity;
use ofgpu::automesher::{read_config_str, AutomeshConfig, MeshKind};
use ofgpu::surface::stl::parse_stl;
use ofgpu::surface::Surface;

/// The example config, as the repository carries it.
const BOX_SPHERE_TET: &str =
    include_str!("../../../tools/automesher/examples/box_sphere_tet.json");

/// `make_box_sphere_stl.py box_sphere_outward.stl --outward` as ASCII STL:
/// radius 1 at (5, 5, 5), 48 latitude bands of 96 meridians, the caps as
/// fans, each triangle `(a, b, c)` of the script written `(a, c, b)` so its
/// right-hand normal points out, and the coordinates in `%.9e`'s ten
/// significant digits.
fn box_sphere_outward_stl() -> String {
    let (r, c) = (1.0f64, [5.0f64; 3]);
    let (nlat, nlon) = (48usize, 96usize);
    let p = |i: usize, j: usize| -> [f64; 3] {
        let th = std::f64::consts::PI * i as f64 / nlat as f64;
        let ph = 2.0 * std::f64::consts::PI * (j % nlon) as f64 / nlon as f64;
        [c[0] + r * th.sin() * ph.cos(), c[1] + r * th.sin() * ph.sin(), c[2] + r * th.cos()]
    };
    let mut s = String::from("solid sphere\n");
    let mut tri = |a: [f64; 3], b: [f64; 3], cc: [f64; 3]| {
        s.push_str("  facet normal 0 0 0\n    outer loop\n");
        for q in [a, cc, b] {
            s.push_str(&format!("      vertex {:.9e} {:.9e} {:.9e}\n", q[0], q[1], q[2]));
        }
        s.push_str("    endloop\n  endfacet\n");
    };
    for i in 0..nlat {
        for j in 0..nlon {
            let (a, b) = (p(i, j), p(i, j + 1));
            let (cc, d) = (p(i + 1, j + 1), p(i + 1, j));
            if i == 0 {
                tri(a, cc, d);
            } else if i == nlat - 1 {
                tri(a, b, cc);
            } else {
                tri(a, b, cc);
                tri(a, cc, d);
            }
        }
    }
    s.push_str("endsolid sphere\n");
    s
}

fn example() -> (AutomeshConfig, Surface) {
    let cfg = read_config_str(BOX_SPHERE_TET, "box_sphere_tet.json").expect("the example parses");
    cfg.validate().expect("the example validates");
    let surf = parse_stl(box_sphere_outward_stl().as_bytes(), "box_sphere_outward", "box_sphere_outward.stl")
        .expect("the sphere parses");
    (cfg, surf)
}

/// SPEC-LIT §116.16 end to end on `box_sphere_tet.json`: the seven stages
/// in order, §92.3 passed, the dihedral bars of §116.18 (min >= 10, max <=
/// 165 degrees), the patches after (92.56), the summary's `tet` block, the
/// polyMesh written and re-read by `-check`'s path (`read_poly_mesh`, then
/// `quality::measure`) passing again, and 30 s or less in release.
#[test]
fn tet_end_to_end() {
    let (cfg, surf) = example();
    assert_eq!(cfg.kind, MeshKind::Tet);
    assert_eq!(surf.tris.len(), 9024);
    let single = std::mem::size_of::<ofgpu::Scalar>() == 4;
    let mut lines: Vec<String> = Vec::new();
    let t0 = std::time::Instant::now();
    let res = run_tet(&cfg, &surf, &mut |l: &str| lines.push(l.to_string()));
    let seconds = t0.elapsed().as_secs_f64();
    let banners: Vec<&String> = lines.iter().filter(|l| l.starts_with("=== stage ")).collect();
    for (i, b) in banners.iter().enumerate() {
        assert_eq!(**b, format!("=== stage {}/7  {} ===", i + 1, TET_STAGES[i]));
    }
    let out = match res {
        Ok(out) => out,
        Err(err) => {
            // SPEC-LIT 116.16's miss under `single`: G2 sums the f32 area
            // vectors, whose rounding exceeds 1e-10 of V^(2/3).
            let text = err.to_string();
            let head: String = text.chars().take(240).collect();
            println!("tet end to end: 92.3 MISSED under single after {seconds:.1} s - {head}");
            assert!(single, "{err}");
            assert_eq!(banners.len(), 7, "{err}");
            assert!(text.contains("G2 (closure)"), "{err}");
            return;
        }
    };
    let q = &out.quality;
    println!(
        "tet end to end: {} tetrahedra, {} prisms, dihedral {:.4} .. {:.4} deg, {} Steiner points, \
         92.3 PASS (max closure {:.3e}, max non-orthogonality {:.3} deg, min tau {:.4}, max cond {:.3}), \
         {:.1} s",
        out.n_tets,
        out.n_prisms,
        out.min_dihedral_deg,
        out.max_dihedral_deg,
        out.steiner["total"],
        q.max_closure,
        q.max_non_orth_deg,
        q.min_thickness_ratio,
        q.max_cond,
        seconds
    );
    assert_eq!(banners.len(), 7);
    assert!(q.passed());
    assert_eq!(out.n_tets, q.n_cells);
    assert_eq!(out.n_prisms, 0);
    assert_eq!(q.n_regions, 1);
    assert!(out.min_dihedral_deg >= 10.0, "min dihedral {}", out.min_dihedral_deg);
    assert!(out.max_dihedral_deg <= 165.0, "max dihedral {}", out.max_dihedral_deg);
    let names: Vec<&str> = out.mesh.patches.iter().map(|p| p.name.as_str()).collect();
    assert_eq!(names, ["inlet", "outlet", "side_y0", "side_y1", "ground", "top", "sphere"]);
    let types: Vec<&str> = out.mesh.patches.iter().map(|p| p.type_name.as_str()).collect();
    assert_eq!(types, ["patch", "patch", "patch", "patch", "patch", "patch", "wall"]);
    assert!(out.mesh.patches.iter().all(|p| p.size > 0));
    let n_boundary = out.mesh.faces.len() - out.mesh.neighbour.len();
    assert_eq!(out.mesh.patches.iter().map(|p| p.size as usize).sum::<usize>(), n_boundary);
    if !cfg!(debug_assertions) {
        assert!(seconds <= 30.0, "box_sphere_tet.json took {seconds:.1} s, the gate is 30 s");
    }

    // The summary of §116.16.
    let case = std::env::temp_dir().join(format!("ofgpu_tet12_{}", std::process::id()));
    let ident = MeshIdentity::new("ofgpu-automesher", &case, &cfg.output.name, None);
    let s = tet_summary_json(&cfg, "box_sphere_tet.json", &surf, &out, &ident);
    assert_eq!(s["kind"], "tet");
    assert_eq!(s["tet"]["n_tets"], out.n_tets);
    assert_eq!(s["tet"]["n_prisms"], 0);
    assert_eq!(s["tet"]["min_dihedral_deg"], out.min_dihedral_deg);
    assert_eq!(s["tet"]["max_dihedral_deg"], out.max_dihedral_deg);
    let st = &s["tet"]["steiner"];
    assert_eq!(
        st["total"].as_u64().unwrap(),
        st["recover"]["total"].as_u64().unwrap() + st["refine"]["total"].as_u64().unwrap()
    );
    let rows: Vec<&str> = s["stages"].as_array().unwrap().iter().map(|r| r["stage"].as_str().unwrap()).collect();
    assert_eq!(rows, TET_STAGES);
    for name in TET_STAGES {
        assert!(s["tet"]["seconds"][name].as_f64().unwrap() >= 0.0, "{name}");
    }
    assert_eq!(s["quality"]["passed"], true);
    assert_eq!(s["mesh"]["n_cells"], out.n_tets);
    assert_eq!(s["config"]["kind"], "tet");
    assert_eq!(s["layout"], serde_json::Value::Null);

    // `-check`'s path: the written polyMesh is read back and gated again.
    let poly = case.join("constant").join("polyMesh");
    ofgpu::io::polymesh::write_poly_mesh_raw(&poly, &out.mesh).expect("write");
    let raw = ofgpu::io::polymesh::read_poly_mesh(&case).expect("read back");
    let back = ofgpu::automesher::quality::measure(&raw, &cfg.quality.thresholds()).expect("measure");
    let _ = std::fs::remove_dir_all(&case);
    assert!(back.passed(), "{}", back.refusal_text());
    assert_eq!(back.n_cells, q.n_cells);
}

/// TET-12's kind-mismatch refusals at the driver: the hex driver refuses a
/// tet config, the tet driver a hex one, each naming the kind; and the tet
/// path refuses a surface that leaves the domain box and a surface patch
/// named like a domain patch, before any stage runs.
#[test]
fn tet_driver_refusals() {
    let (cfg, surf) = example();
    let mut sink = |_: &str| {};
    let err = run(&cfg, &surf, None, &mut sink).unwrap_err().to_string();
    assert!(err.contains("driver::run: this config's kind is \"tet\""), "{err}");

    let mut hex = cfg.clone();
    hex.kind = MeshKind::HexDominant;
    hex.tet = None;
    let err = run_tet(&hex, &surf, &mut sink).unwrap_err().to_string();
    assert!(err.contains("driver::run_tet: this config's kind is \"hexDominant\""), "{err}");

    let mut small = cfg.clone();
    small.domain.extent = [0.0, 5.5, 0.0, 10.0, 0.0, 10.0];
    let mut lines: Vec<String> = Vec::new();
    let err = run_tet(&small, &surf, &mut |l: &str| lines.push(l.to_string())).unwrap_err().to_string();
    assert!(err.contains("tet/domain: surface point"), "{err}");
    assert!(lines.is_empty(), "{lines:?}");

    let mut named = surf.clone();
    named.patch_names = vec!["xMin".to_string()];
    let mut renamed = cfg.clone();
    renamed.refinement.levels[0].patch = "xMin".to_string();
    let err = run_tet(&renamed, &named, &mut sink).unwrap_err().to_string();
    assert!(err.contains("tet/patches: the surface patch \"xMin\""), "{err}");
}

/// TET-13b's prism layers end to end (SPEC-LIT §116.15, §116.16): the same
/// example with `layers.patches = ["sphere"]`, 3 layers at 0.02 m, growth
/// 1.2. Seven banners still; in f64 §92.3 passes and `n_prisms` is three
/// per sphere-patch face; the summary's `tet` block and the emit row carry
/// the prism counts. Under `single` the G2 miss of §116.16 refuses the run
/// and the test prints MISSED, as `tet_end_to_end` does.
#[test]
fn tet_layers_end_to_end() {
    let (mut cfg, surf) = example();
    assert_eq!(cfg.kind, MeshKind::Tet);
    cfg.layers.patches = vec!["sphere".to_string()];
    cfg.layers.n = 3;
    cfg.layers.first_thickness = 0.02;
    cfg.layers.growth = 1.2;
    cfg.validate().expect("the layered config validates");
    let single = std::mem::size_of::<ofgpu::Scalar>() == 4;
    let mut lines: Vec<String> = Vec::new();
    let t0 = std::time::Instant::now();
    let res = run_tet(&cfg, &surf, &mut |l: &str| lines.push(l.to_string()));
    let seconds = t0.elapsed().as_secs_f64();
    let banners: Vec<&String> = lines.iter().filter(|l| l.starts_with("=== stage ")).collect();
    for (i, b) in banners.iter().enumerate() {
        assert_eq!(**b, format!("=== stage {}/7  {} ===", i + 1, TET_STAGES[i]));
    }
    let out = match res {
        Ok(out) => out,
        Err(err) => {
            let text = err.to_string();
            let head: String = text.chars().take(240).collect();
            println!("tet layers end to end: 92.3 MISSED under single after {seconds:.1} s - {head}");
            assert!(single, "{err}");
            assert_eq!(banners.len(), 7, "{err}");
            assert!(text.contains("G2 (closure)"), "{err}");
            return;
        }
    };
    let q = &out.quality;
    let n_sphere = out
        .mesh
        .patches
        .iter()
        .find(|p| p.name == "sphere")
        .map(|p| p.size as usize)
        .unwrap_or(0);
    println!(
        "tet layers end to end: {} tetrahedra, {} prisms = 3 x {n_sphere} sphere faces, \
         92.3 PASS (max closure {:.3e}), {:.1} s",
        out.n_tets, out.n_prisms, q.max_closure, seconds
    );
    assert_eq!(banners.len(), 7);
    assert!(q.passed());
    assert_eq!(out.n_prisms, 3 * n_sphere);
    assert!(out.n_prisms > 0);

    // The summary of §116.16 with the prism counts.
    let case = std::env::temp_dir().join(format!("ofgpu_tet13b_{}", std::process::id()));
    let ident = MeshIdentity::new("ofgpu-automesher", &case, &cfg.output.name, None);
    let s = tet_summary_json(&cfg, "box_sphere_tet.json", &surf, &out, &ident);
    assert_eq!(s["tet"]["n_prisms"], out.n_prisms);
    let emit_row = s["stages"]
        .as_array()
        .unwrap()
        .iter()
        .find(|r| r["stage"] == "emit")
        .expect("the emit row")
        .clone();
    assert_eq!(emit_row["n_prisms"], out.n_prisms);
    assert!(emit_row.get("n_layer_points").is_some());
    assert!(emit_row.get("inflate_rounds").is_some());
    assert!(emit_row.get("n_cut").is_some());
    println!(
        "tet layers end to end: emit row n_prisms {}, n_layer_points {}, inflate_rounds {}, n_cut {}",
        emit_row["n_prisms"], emit_row["n_layer_points"], emit_row["inflate_rounds"],
        emit_row["n_cut"]
    );
    let _ = std::fs::remove_dir_all(&case);
}

/// TET-13b's driver refusal: a `layers.patches` name that is no surface
/// patch is refused with `tet/layers` (SPEC-LIT §116.15, "In the driver").
#[test]
fn tet_layers_refusal() {
    let (mut cfg, surf) = example();
    cfg.layers.patches = vec!["nope".to_string()];
    cfg.layers.n = 2;
    cfg.validate().expect("the config still validates: the tet path reads layers.patches");
    let mut lines: Vec<String> = Vec::new();
    let err =
        run_tet(&cfg, &surf, &mut |l: &str| lines.push(l.to_string())).unwrap_err().to_string();
    assert!(err.contains("tet/layers"), "{err}");
    assert!(lines.is_empty(), "{lines:?}");
    println!("tet layers refusal: {err}");
}
