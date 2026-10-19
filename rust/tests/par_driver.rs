// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.
// Provenance: see PROVENANCE.md. No GPL-licensed source was consulted.

//! SPEC-LIT §138.8's gates - the parallel driver `ofgpu-par`, in BOTH builds.
//!
//! An integration test because the library's own test target does not
//! compile under `--features single`, and because the driver is a binary:
//! each test runs `ofgpu-par` as a child process and reads its exit status,
//! its gate lines and its refusals. The refusals run before any device is
//! opened. The GPU test runs the threads transport on a reduced copy of
//! `cases/plume.jsonc` and counts the gate lines, so a gate that stopped
//! running would fail here as surely as a gate that failed.
//!
//! Class D of docs/17 §5.1: exact bytes and exact counts.
//!
//! Provenance: ORIGINAL - the driver under test is this project's own design
//! (SPEC-LIT §138.8, PAR-07). No external source. No GPL-licensed source was
//! consulted.

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

fn exe() -> Command {
    let mut c = Command::new(env!("CARGO_BIN_EXE_ofgpu-par"));
    for var in ["PMI_RANK", "PMI_SIZE", "OFGPU_DEVICES"] {
        c.env_remove(var);
    }
    c
}

fn plume() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../cases/plume.jsonc")
}

fn text(o: &Output) -> (String, String) {
    (
        String::from_utf8_lossy(&o.stdout).into_owned(),
        String::from_utf8_lossy(&o.stderr).into_owned(),
    )
}

/// Runs `ofgpu-par` and returns its stderr, asserting a non-zero exit and
/// that every word in `names` is in the message.
fn refused(cmd: &mut Command, names: &[&str]) -> String {
    let o = cmd.output().expect("run ofgpu-par");
    let (out, err) = text(&o);
    assert!(!o.status.success(), "accepted: {out}{err}");
    assert!(err.starts_with("ofgpu-par: "), "the message: {err}");
    for n in names {
        assert!(err.contains(n), "the refusal does not name `{n}`: {err}");
    }
    assert!(!out.contains("PASS"), "a refused run printed a gate: {out}");
    err
}

/// `-comm mpi` is refused by name: the transport is PAR-06, deferred.
#[test]
fn par_refuses_mpi_by_name() {
    let p = plume();
    refused(
        exe().arg(&p).args(["-np", "2", "-comm", "mpi"]),
        &["-comm mpi", "deferred", "PAR-06", "-comm threads"],
    );
}

/// A PMI launcher's environment is refused by name whatever `-comm` says:
/// `PMI_RANK` and `PMI_SIZE` together, and each alone.
#[test]
fn par_refuses_mpiexec_environment() {
    let p = plume();
    let want = ["mpiexec", "deferred", "PAR-06", "-comm threads"];
    let err = refused(
        exe()
            .arg(&p)
            .args(["-np", "2", "-comm", "threads"])
            .env("PMI_RANK", "0")
            .env("PMI_SIZE", "2"),
        &want,
    );
    assert!(err.contains("PMI_RANK"), "{err}");
    let err = refused(exe().arg(&p).env("PMI_SIZE", "4"), &want);
    assert!(err.contains("PMI_SIZE"), "{err}");
    let err = refused(exe().arg(&p).env("PMI_RANK", "1"), &want);
    assert!(err.contains("PMI_RANK"), "{err}");
}

/// An unknown transport, a zero or malformed rank count and an unknown
/// partitioner are refused by name.
#[test]
fn par_refuses_bad_options() {
    let p = plume();
    refused(exe().arg(&p).args(["-comm", "tcp"]), &["-comm 'tcp'", "threads"]);
    refused(exe().arg(&p).args(["-np", "2,0"]), &["-np '2,0'", "at least 1"]);
    refused(exe().arg(&p).args(["-np", "x"]), &["-np 'x'"]);
    refused(exe().arg(&p).args(["-method", "metis"]), &["'metis'", "hilbert"]);
    refused(exe().arg(&p).arg("-frobnicate"), &["unknown option '-frobnicate'"]);
}

/// **The gate.** `cases/plume.jsonc` at 28 x 12 x 5 cells, `-np 2,3,4
/// -comm threads`: 3 partitioners x (2 + 3 + 4) rotations = 27 rotations,
/// each with a MAP, HALO and REDUCTION line and a PCG and PBiCGStab line,
/// every one PASS, and the verdict PASS with exit status 0.
#[test]
fn par_threads_gates_pass_on_a_small_plume() {
    let src = std::fs::read_to_string(plume()).expect("read cases/plume.jsonc");
    let full = "\"cells\":  [98, 42, 20]";
    assert_eq!(src.matches(full).count(), 1, "the plume case's cell line moved");
    let small = src.replace(full, "\"cells\":  [28, 12, 5]");
    let dir = std::env::temp_dir().join(format!("ofgpu-par-test-{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("temp dir");
    let case = dir.join("plume_small.jsonc");
    std::fs::write(&case, small).expect("write the case");

    let o = exe()
        .arg(&case)
        .args(["-np", "2,3,4", "-comm", "threads", "-quiet"])
        .output()
        .expect("run ofgpu-par");
    let _ = std::fs::remove_dir_all(&dir);
    let (out, err) = text(&o);
    assert!(o.status.success(), "ofgpu-par failed:\n{out}\n{err}");

    let count = |gate: &str| {
        out.lines()
            .filter(|l| l.contains(&format!(" {gate} PASS - ")))
            .count()
    };
    assert_eq!(count("MAP      "), 27, "{out}");
    assert_eq!(count("HALO     "), 27, "{out}");
    assert_eq!(count("REDUCTION"), 27, "{out}");
    assert_eq!(count("PCG      "), 27, "{out}");
    assert_eq!(count("PBiCGStab"), 27, "{out}");
    assert!(!out.contains("FAIL"), "{out}");
    let last = out.lines().last().unwrap_or("");
    assert!(
        last.starts_with("ofgpu-par: PASS - 9 worlds, 27 rotations"),
        "the verdict: {last}"
    );
    // Every HALO line compared ghost cells: a cut with no halo proves nothing.
    for l in out.lines().filter(|l| l.contains(" HALO      PASS")) {
        assert!(!l.contains(" + 0 ghost"), "no ghost cells: {l}");
    }
}

/// SPEC-LIT 139.10 (PAR-14)'s fixtures: a fresh generated cavity case under
/// the temp dir, removed first - the BASE side of every one-setting pair.
fn base_case(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("ofgpu-par14-{tag}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    ofgpu::blockgen::write_case(&dir, ofgpu::blockgen::CaseKind::Cavity, 8, 8, 1)
        .expect("write the base case");
    dir
}

/// `-check` accepts `dir`: exit 0 and the line saying the case takes the
/// decomposed laminar SIMPLE path.
fn accepted(cmd: &mut Command) {
    let o = cmd.output().expect("run ofgpu-par -check");
    let (out, err) = text(&o);
    assert!(o.status.success(), "refused:\n{out}\n{err}");
    assert!(
        out.contains("ofgpu-par -check:"),
        "no -check line:\n{out}\n{err}"
    );
    assert!(!out.contains("PASS"), "a -check run printed a gate: {out}");
}

/// SPEC-LIT 13.4.1's shape, on SPEC-LIT 139.10's refusal list: for every
/// unconverted path the BASE case is accepted by `ofgpu-par <dir> -check`,
/// ONE setting is turned, and the same command is refused naming the row's
/// name and `139.10`. No GPU is needed - the refusals and the loader run
/// before any device is opened.
#[test]
fn par_check_refuses_every_unconverted_path_by_name() {
    // VOF: `0/alpha.water` exists.
    let d = base_case("vof");
    accepted(exe().arg(&d).arg("-check"));
    std::fs::write(d.join("0/alpha.water"), "internalField uniform 0;\n").expect("write");
    refused(exe().arg(&d).arg("-check"), &["VOF", "139.10"]);

    // The dust model: cloudProperties stating `dustCloud`.
    let d = base_case("dust");
    accepted(exe().arg(&d).arg("-check"));
    std::fs::write(d.join("constant/cloudProperties"), "dustCloud { }\n").expect("write");
    refused(exe().arg(&d).arg("-check"), &["dust model", "139.10"]);

    // Parcels: cloudProperties without `dustCloud`.
    let d = base_case("parcels");
    accepted(exe().arg(&d).arg("-check"));
    std::fs::write(d.join("constant/cloudProperties"), "capacity 4;\n").expect("write");
    refused(exe().arg(&d).arg("-check"), &["parcels", "139.10"]);

    // LTS: `ddtSchemes/default localEuler` in the generated fvSchemes.
    let d = base_case("lts");
    accepted(exe().arg(&d).arg("-check"));
    let p = d.join("system/fvSchemes");
    let src = std::fs::read_to_string(&p).expect("read fvSchemes");
    let steady = "default         steadyState;";
    assert_eq!(src.matches(steady).count(), 1, "the generated fvSchemes moved");
    std::fs::write(&p, src.replacen(steady, "default         localEuler;", 1)).expect("write");
    refused(exe().arg(&d).arg("-check"), &["LTS", "139.10"]);

    // AMR: `dynamicFvMesh dynamicRefineFvMesh`.
    let d = base_case("amr");
    accepted(exe().arg(&d).arg("-check"));
    std::fs::write(d.join("constant/dynamicMeshDict"), "dynamicFvMesh dynamicRefineFvMesh;\n")
        .expect("write");
    refused(exe().arg(&d).arg("-check"), &["AMR", "139.10"]);

    // ALE: `dynamicFvMesh` anything else; the static base side is accepted.
    let d = base_case("ale");
    std::fs::write(d.join("constant/dynamicMeshDict"), "dynamicFvMesh staticFvMesh;\n")
        .expect("write");
    accepted(exe().arg(&d).arg("-check"));
    std::fs::write(
        d.join("constant/dynamicMeshDict"),
        "dynamicFvMesh dynamicMotionSolverFvMesh;\n",
    )
    .expect("write");
    refused(exe().arg(&d).arg("-check"), &["ALE", "139.10"]);

    // CHT multi-region.
    let d = base_case("cht");
    accepted(exe().arg(&d).arg("-check"));
    std::fs::write(
        d.join("constant/regionProperties"),
        "regions ( fluid (air) solid (wall) );\n",
    )
    .expect("write");
    refused(exe().arg(&d).arg("-check"), &["CHT multi-region", "139.10"]);

    // Reacting chemistry.
    let d = base_case("chem");
    accepted(exe().arg(&d).arg("-check"));
    std::fs::write(d.join("constant/chemistryProperties"), "chemistry on;\n").expect("write");
    refused(exe().arg(&d).arg("-check"), &["reacting chemistry", "139.10"]);

    // ofgpu-compressible.
    let d = base_case("compressible");
    accepted(exe().arg(&d).arg("-check"));
    std::fs::write(
        d.join("constant/thermophysicalProperties"),
        "thermoType { type hePsiThermo; }\n",
    )
    .expect("write");
    refused(exe().arg(&d).arg("-check"), &["ofgpu-compressible", "139.10"]);

    // Radiation: the model's own name; the `none` base side is accepted.
    let d = base_case("rad-p1");
    std::fs::write(d.join("constant/radiationProperties"), "radiationModel none;\n")
        .expect("write");
    accepted(exe().arg(&d).arg("-check"));
    std::fs::write(d.join("constant/radiationProperties"), "radiationModel P1;\n").expect("write");
    refused(exe().arg(&d).arg("-check"), &["P1", "139.10"]);

    let d = base_case("rad-fvdom");
    accepted(exe().arg(&d).arg("-check"));
    std::fs::write(d.join("constant/radiationProperties"), "radiationModel fvDOM;\n")
        .expect("write");
    refused(exe().arg(&d).arg("-check"), &["fvDOM", "139.10"]);

    // The flags: the pbicgstab base side is accepted, fft refused; the
    // no-flag base side is accepted, -graph refused.
    let d = base_case("backend");
    accepted(exe().arg(&d).arg("-check").args(["-backend", "pbicgstab"]));
    refused(
        exe().arg(&d).arg("-check").args(["-backend", "fft"]),
        &["FFT backend", "139.10"],
    );

    let d = base_case("graph");
    accepted(exe().arg(&d).arg("-check"));
    refused(exe().arg(&d).arg("-check").arg("-graph"), &["CUDA-graph capture", "139.10"]);
}
