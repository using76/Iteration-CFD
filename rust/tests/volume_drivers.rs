// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.
// Provenance: see PROVENANCE.md. No GPL-licensed source was consulted.

//! VIO-05a's and VIO-05b's smoke tests: every driver writes `.vdb` on a mesh
//! that is not a uniform Cartesian box (SPEC-LIT 117.12) - `ofgpu-lowmach`
//! and `ofgpu-k-omega` (VIO-05a), `ofgpu-k-epsilon`, `ofgpu-sa`, `ofgpu-vof`,
//! `ofgpu-buoyant` and `ofgpu-cht`, the last one file per region (VIO-05b).
//!
//! The mesh is `ofgpu-generate-mesh channel 8 6 4 -grading x=2`: graded, so
//! `cartesian::detect` refuses it and the run lays the lattice of 117.3 on
//! its defaults (the command line states no lattice). Each driver runs 20
//! iterations with `-output vdb` and must exit 0, print the lattice of
//! (117.1) - predicted here from the mesh, `h = (min_c V_c)^(1/3)` and
//! `n_a = max(1, ceil(L_a / h - 1e-9))`, not read from the map - and leave a
//! non-empty `.vdb` under `VDB/`. The box-sphere automesher run of the
//! plan's gate is the supervisor's (SPEC-LIT 117.13), because its STL comes
//! from a Python script the test suite does not run.
//!
//! `#[ignore]`: it needs a CUDA device. Run it with `--ignored`.
//!
//! Written from SPEC-LIT section 117 (117.3, 117.7, 117.12).
//! Provenance: ORIGINAL.
//! No GPL-licensed source was consulted.

use std::path::{Path, PathBuf};
use std::process::Command;

use ofgpu::io::polymesh::{build_host_mesh, read_poly_mesh};

fn scratch(name: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("ofgpu-vio05a-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).expect("scratch dir");
    d
}

/// The graded channel case, written by the real generator.
fn graded_channel(dir: &Path) -> PathBuf {
    generated(dir, "channel", &["8", "6", "4", "-grading", "x=2"])
}

/// A case of `kind` written by the real generator with `args`.
fn generated(dir: &Path, kind: &str, args: &[&str]) -> PathBuf {
    let case = dir.join(kind);
    let out = Command::new(env!("CARGO_BIN_EXE_ofgpu-generate-mesh"))
        .arg(kind)
        .arg(&case)
        .args(args)
        .output()
        .expect("run ofgpu-generate-mesh");
    assert!(out.status.success(), "generate-mesh: {}", String::from_utf8_lossy(&out.stderr));
    case
}

/// Swap the generator's `kEpsilon` for `model` in `constant/momentumTransport`.
fn set_model(case: &Path, model: &str) {
    let mt = case.join("constant").join("momentumTransport");
    let text = std::fs::read_to_string(&mt).expect("momentumTransport");
    assert!(text.contains("kEpsilon;"), "the generator's model");
    std::fs::write(&mt, text.replace("kEpsilon;", &format!("{model};"))).expect("write momentumTransport");
}

/// (117.1) on the 117.3 defaults, from the mesh alone: `[n_x, n_y, n_z]`.
fn predicted_lattice(case: &Path) -> [usize; 3] {
    let raw = read_poly_mesh(case).expect("read_poly_mesh");
    let hm = build_host_mesh(&raw).expect("build_host_mesh");
    assert!(ofgpu::pressure::cartesian::detect(&hm).is_err(), "the graded mesh is not a uniform box");
    let v_min = hm.geom64().v.iter().copied().fold(f64::INFINITY, f64::min);
    let h = v_min.cbrt();
    let (mut lo, mut hi) = ([f64::INFINITY; 3], [f64::NEG_INFINITY; 3]);
    for p in &raw.points {
        for (a, x) in [p.x, p.y, p.z].into_iter().enumerate() {
            lo[a] = lo[a].min(x);
            hi[a] = hi[a].max(x);
        }
    }
    let n = [0, 1, 2].map(|a| (((hi[a] - lo[a]) / h - 1e-9).ceil() as usize).max(1));
    assert!(n[0] * n[1] * n[2] <= 8_000_000, "the budget does not bind here");
    n
}

/// Run `exe` on `case` and check the lattice line and the `.vdb`.
fn run_and_check(exe: &str, case: &Path, output: &str, n: [usize; 3]) {
    run_and_check_with(exe, case, &["-iters", "20", "-output", output], n);
}

/// [`run_and_check`] with the driver's own arguments after the case.
fn run_and_check_with(exe: &str, case: &Path, args: &[&str], n: [usize; 3]) {
    let t0 = std::time::Instant::now();
    let out = Command::new(exe)
        .arg(case)
        .args(args)
        .output()
        .expect("run the driver");
    eprintln!("{exe}: {:.1} s", t0.elapsed().as_secs_f64());
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        out.status.success(),
        "{exe} failed: {}\n{stdout}\n{}",
        out.status,
        String::from_utf8_lossy(&out.stderr)
    );
    let want = format!("volume output: lattice {} x {} x {} of h = ", n[0], n[1], n[2]);
    let line = stdout
        .lines()
        .find(|l| l.starts_with("volume output: lattice"))
        .unwrap_or_else(|| panic!("{exe}: no lattice line\n{stdout}"));
    assert!(line.starts_with(&want), "{exe}: {line} (want {want})");
    let voxels = format!(": {} voxels,", n[0] * n[1] * n[2]);
    assert!(line.contains(&voxels), "{exe}: {line} (want {voxels})");
    let vdb: Vec<PathBuf> = std::fs::read_dir(case.join("VDB"))
        .expect("a VDB directory")
        .map(|e| e.expect("entry").path())
        .filter(|p| p.extension().is_some_and(|e| e == "vdb"))
        .collect();
    assert_eq!(vdb.len(), 1, "{exe}: one .vdb: {vdb:?}");
    assert!(std::fs::metadata(&vdb[0]).expect("metadata").len() > 0, "{exe}: a non-empty .vdb");
}

#[test]
#[ignore = "needs a CUDA device"]
fn lowmach_and_k_omega_write_vdb_on_a_graded_mesh() {
    let dir = scratch("drivers");
    let case = graded_channel(&dir);
    let n = predicted_lattice(&case);
    run_and_check(env!("CARGO_BIN_EXE_ofgpu-lowmach"), &case, "foam,vdb", n);

    // ofgpu-k-omega builds only kOmega: the same case with that model.
    let _ = std::fs::remove_dir_all(case.join("VDB"));
    let mt = case.join("constant").join("momentumTransport");
    let text = std::fs::read_to_string(&mt).expect("momentumTransport");
    assert!(text.contains("kEpsilon;"), "the generator's model");
    std::fs::write(&mt, text.replace("kEpsilon;", "kOmega;")).expect("write momentumTransport");
    run_and_check(env!("CARGO_BIN_EXE_ofgpu-k-omega"), &case, "vdb", n);
    let _ = std::fs::remove_dir_all(&dir);
}

/// VIO-05b: `ofgpu-k-epsilon` (the generator's own model) and `ofgpu-sa` on
/// the same graded channel, each with `-output vdb` on the command line.
#[test]
#[ignore = "needs a CUDA device"]
fn k_epsilon_and_sa_write_vdb_on_a_graded_mesh() {
    let dir = scratch("ke-sa");
    let case = graded_channel(&dir);
    let n = predicted_lattice(&case);
    run_and_check(env!("CARGO_BIN_EXE_ofgpu-k-epsilon"), &case, "foam,vdb", n);

    let _ = std::fs::remove_dir_all(case.join("VDB"));
    set_model(&case, "SpalartAllmaras");
    run_and_check(env!("CARGO_BIN_EXE_ofgpu-sa"), &case, "vdb", n);
    let _ = std::fs::remove_dir_all(&dir);
}

/// VIO-05b: `ofgpu-vof` on a graded damBreak, a few time steps.
#[test]
#[ignore = "needs a CUDA device"]
fn vof_writes_vdb_on_a_graded_mesh() {
    let dir = scratch("vof");
    let case = generated(&dir, "damBreak", &["20", "30", "1", "-grading", "x=2"]);
    let n = predicted_lattice(&case);
    let args = ["-endTime", "0.002", "-deltaT", "0.0005", "-output", "foam,vdb"];
    run_and_check_with(env!("CARGO_BIN_EXE_ofgpu-vof"), &case, &args, n);
    let _ = std::fs::remove_dir_all(&dir);
}

/// VIO-05b: `ofgpu-buoyant` on a graded plume, a few iterations.
#[test]
#[ignore = "needs a CUDA device"]
fn buoyant_writes_vdb_on_a_graded_mesh() {
    let dir = scratch("buoyant");
    let case = generated(&dir, "plume", &["8", "8", "8", "-grading", "z=2"]);
    let n = predicted_lattice(&case);
    let args = ["-iters", "6", "-backend", "pbicgstab", "-output", "vdb"];
    run_and_check_with(env!("CARGO_BIN_EXE_ofgpu-buoyant"), &case, &args, n);
    let _ = std::fs::remove_dir_all(&dir);
}

/// VIO-05b: `ofgpu-cht` lays one lattice per region (117.3: the default box
/// is the region's). A two-layer slab whose first region is graded, so it is
/// not a uniform box and gets a lattice of (117.1) on its own points, while
/// the second is a uniform box whose cells are its voxels (117.6). Each
/// region leaves `<case>_jsonc/VDB/<region>_000000.vdb`.
#[test]
#[ignore = "needs a CUDA device"]
fn cht_writes_one_vdb_per_region() {
    let dir = scratch("cht");
    let case = dir.join("slab.cht.jsonc");
    std::fs::write(&case, CHT_SLAB).expect("write the case");
    let t0 = std::time::Instant::now();
    let out = Command::new(env!("CARGO_BIN_EXE_ofgpu-cht")).arg(&case).output().expect("run ofgpu-cht");
    eprintln!("ofgpu-cht: {:.1} s", t0.elapsed().as_secs_f64());
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(out.status.success(), "ofgpu-cht failed: {stdout}\n{}", String::from_utf8_lossy(&out.stderr));
    let lattice = stdout
        .lines()
        .find(|l| l.contains("region 'insulation': volume output: lattice"))
        .unwrap_or_else(|| panic!("no lattice line for the graded region\n{stdout}"));
    // The graded region: 12 x 1 x 1 cells over [0, 0.010] x [0, 0.02]^2 with
    // x expansion 4 (last / first). Its smallest cell is the first, of width
    // dx0 = L r^0 / sum_{i<12} r^(i/11) with r = 4; h = (dx0 dy dz)^(1/3).
    let r: f64 = 4.0;
    let sum: f64 = (0..12).map(|i| r.powf(i as f64 / 11.0)).sum();
    let dx0 = 0.010 / sum;
    let h = (dx0 * 0.02 * 0.02).cbrt();
    let n = [0.010, 0.02, 0.02].map(|l: f64| ((l / h - 1e-9).ceil() as usize).max(1));
    let want = format!("lattice {} x {} x {} of h = ", n[0], n[1], n[2]);
    assert!(lattice.contains(&want), "{lattice} (want {want})");
    assert!(
        stdout.contains("region 'metal': volume output: uniform Cartesian box"),
        "the uniform region writes its cells\n{stdout}"
    );
    let vdb_dir = dir.join("slab.cht_jsonc").join("VDB");
    for region in ["insulation", "metal"] {
        let p = vdb_dir.join(format!("{region}_000000.vdb"));
        assert!(std::fs::metadata(&p).map(|m| m.len() > 0).unwrap_or(false), "a non-empty {}", p.display());
    }
    let _ = std::fs::remove_dir_all(&dir);
}

/// The two-layer slab of `io::case_cht`'s tests, with the first region
/// graded and an `output.visualisation` block asking for `vdb`.
const CHT_SLAB: &str = r#"{
  "name": "twoLayerSlab",
  "regions": [
    {
      "name": "insulation",
      "mesh": {
        "bounds": { "min": [0.0, 0.0, 0.0], "max": [0.010, 0.02, 0.02] },
        "cells": [12, 1, 1], "grading": { "x": { "expansion": 4.0 } },
        "boundaries": {
          "xmin": "hot",  "xmax": "toMetal",
          "ymin": "sideA", "ymax": "sideB",
          "zmin": "frontA", "zmax": "backA"
        }
      },
      "material": { "rho": 2000.0, "c": 800.0, "kappa": 1.4 },
      "patches": [
        { "match": "hot",    "T": { "type": "fixedValue", "value": 380.0 } },
        { "match": "sideA",  "T": { "type": "zeroGradient" } },
        { "match": "sideB",  "T": { "type": "zeroGradient" } },
        { "match": "frontA", "T": { "type": "zeroGradient" } },
        { "match": "backA",  "T": { "type": "zeroGradient" } }
      ]
    },
    {
      "name": "metal",
      "mesh": {
        "bounds": { "min": [0.010, 0.0, 0.0], "max": [0.030, 0.02, 0.02] },
        "cells": [9, 1, 1],
        "boundaries": {
          "xmin": "toInsulation", "xmax": "cold",
          "ymin": "sideC", "ymax": "sideD",
          "zmin": "frontB", "zmax": "backB"
        }
      },
      "material": { "rho": 1000.0, "c": 1200.0, "kappa": 148.0 },
      "patches": [
        { "match": "cold",   "T": { "type": "fixedValue", "value": 300.0 } },
        { "match": "sideC",  "T": { "type": "zeroGradient" } },
        { "match": "sideD",  "T": { "type": "zeroGradient" } },
        { "match": "frontB", "T": { "type": "zeroGradient" } },
        { "match": "backB",  "T": { "type": "zeroGradient" } }
      ]
    }
  ],
  "interfaces": [
    { "regionA": "insulation", "patchA": "toMetal", "regionB": "metal", "patchB": "toInsulation" }
  ],
  "initial": { "T": 340.0 },
  "run": { "steady": true },
  "numerics": { "solver": "PCG", "preconditioner": "DIC", "tolerance": 1e-12, "maxIter": 4000 },
  "output": { "visualisation": { "format": "vdb" } }
}
"#;
