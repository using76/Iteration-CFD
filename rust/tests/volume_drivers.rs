// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.
// Provenance: see PROVENANCE.md. No GPL-licensed source was consulted.

//! VIO-05a's smoke test: `ofgpu-lowmach` and `ofgpu-k-omega` write `.vdb`
//! on a mesh that is not a uniform Cartesian box (SPEC-LIT 117.12).
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
    let case = dir.join("ch");
    let out = Command::new(env!("CARGO_BIN_EXE_ofgpu-generate-mesh"))
        .arg("channel")
        .arg(&case)
        .args(["8", "6", "4", "-grading", "x=2"])
        .output()
        .expect("run ofgpu-generate-mesh");
    assert!(out.status.success(), "generate-mesh: {}", String::from_utf8_lossy(&out.stderr));
    case
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
    let out = Command::new(exe)
        .arg(case)
        .args(["-iters", "20", "-output", output])
        .output()
        .expect("run the driver");
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
