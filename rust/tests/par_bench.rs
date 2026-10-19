// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.
// Provenance: see PROVENANCE.md. No GPL-licensed source was consulted.

//! SPEC-LIT §140's gates - the `-bench` harness of `ofgpu-par`, in BOTH
//! builds (`tests/par_driver.rs`'s harness: each test runs `ofgpu-par` as a
//! child process and reads its exit status, its rows and its refusals; the
//! refusals run before any device is opened).
//!
//! Class D of docs/17 §5.1: exact counts and exact strings. The timings
//! themselves are REPORTED, never gated - the one threshold, the plan's
//! 600 s for P = 1..4, was FIXED BEFORE MEASURING and a miss is printed and
//! the test continues (a shared card is slower, not wrong).
//!
//! Provenance: ORIGINAL - the harness under test is this project's own
//! design (SPEC-LIT §140, PAR-16); these tests check its refusals and its
//! time-shared rule. No external source code. No GPL-licensed source was
//! consulted.

use std::path::PathBuf;
use std::process::{Command, Output};

fn exe() -> Command {
    let mut c = Command::new(env!("CARGO_BIN_EXE_ofgpu-par"));
    for var in ["PMI_RANK", "PMI_SIZE", "OFGPU_DEVICES"] {
        c.env_remove(var);
    }
    c
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
    assert!(!out.contains("DONE"), "a refused run measured: {out}");
    err
}

/// A temp directory of this test's own, removed at the end.
fn temp_dir(tag: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("ofgpu-par-bench-{tag}-{}", std::process::id()));
    std::fs::create_dir_all(&d).expect("temp dir");
    d
}

/// Runs one `-bench` command to its JSON file, returns (stdout, stderr, the
/// parsed JSON as text), and removes the directory.
fn bench_to_json(cmd: &mut Command, json: &std::path::Path, dir: &std::path::Path)
-> (String, String, String) {
    std::fs::create_dir_all(dir).expect("temp dir");
    let o = cmd.arg("-json").arg(json).output().expect("run ofgpu-par");
    let (out, err) = text(&o);
    let text = std::fs::read_to_string(json)
        .unwrap_or_else(|e| panic!("read the JSON {}: {e}\n{out}\n{err}", json.display()));
    let _ = std::fs::remove_dir_all(dir);
    assert!(o.status.success(), "ofgpu-par -bench failed:\n{out}\n{err}");
    (out, err, text)
}

/// `-bench speed` is refused by name, before any option is read and before
/// any device is opened: the refusal names both known modes.
#[test]
fn bench_refuses_an_unknown_mode() {
    refused(
        exe().args(["-bench", "speed"]),
        &["-bench 'speed'", "strong", "weak"],
    );
    refused(exe().args(["-bench"]), &["-bench ''", "strong", "weak"]);
}

/// `-bench strong -comm mpi` is refused by name: the MPI transport is
/// PAR-06, deferred by the user.
#[test]
fn bench_refuses_mpi_by_name() {
    refused(
        exe().args(["-bench", "strong", "-comm", "mpi"]),
        &["-comm mpi", "deferred", "PAR-06", "-comm threads"],
    );
}

/// A zero mesh side, rank count, timed-step count or exchange count, and a
/// partitioner that is not one cut (`all`), are refused naming the option,
/// before any device is opened.
#[test]
fn bench_refuses_zero_counts() {
    refused(exe().args(["-bench", "strong", "-n", "0"]), &["-n"]);
    refused(exe().args(["-bench", "strong", "-np", "0"]), &["-np '0'"]);
    refused(exe().args(["-bench", "weak", "-iters", "0"]), &["-iters"]);
    refused(
        exe().args(["-bench", "weak", "-exchanges", "0"]),
        &["-exchanges"],
    );
    refused(
        exe().args(["-bench", "strong", "-method", "all"]),
        &["-method 'all'", "hilbert", "linear", "roundrobin"],
    );
}

/// A reduced strong bench on a real card: P = 1 is the reference with NO
/// speedup and NO efficiency, every statistic carries five positive samples
/// whose range brackets its median and whose interval brackets its mean,
/// the MS-MPI column reads NOT RUN, and - while the machine has fewer than
/// two devices - P = 2 is time-shared and says so on stdout.
#[test]
fn a_reduced_strong_bench_is_time_shared_on_one_device() {
    let dir = temp_dir("strong");
    let json = dir.join("s.json");
    let (out, _err, text) = bench_to_json(
        exe().args([
            "-bench", "strong", "-np", "1,2", "-n", "16", "-warm", "1",
            "-iters", "2", "-exchanges", "10",
        ]),
        &json,
        &dir,
    );
    let v: serde_json::Value =
        serde_json::from_str(&text).expect("the bench JSON parses");
    assert_eq!(v["bench"], "strong");
    assert_eq!(v["transport"], "threads");
    assert_eq!(v["repeats"], 5);
    assert!(
        v["mpi"].as_str().unwrap_or("").starts_with("NOT RUN"),
        "mpi: {}",
        v["mpi"]
    );
    let rows = v["rows"].as_array().expect("rows");
    assert_eq!(rows.len(), 2, "{text}");
    assert_eq!(rows[0]["np"], 1);
    assert_eq!(rows[0]["scaling"], "reference");
    assert!(rows[0]["speedup"].is_null(), "{}", rows[0]);
    assert!(rows[0]["efficiency"].is_null(), "{}", rows[0]);
    for row in rows {
        for key in ["t_iter", "l_ex", "l_ag"] {
            let s = &row[key];
            let samples = s["samples"].as_array().expect("samples");
            assert_eq!(samples.len(), 5, "{key}: {s}");
            assert!(
                samples.iter().all(|x| x.as_f64().unwrap_or(0.0) > 0.0),
                "{key}: {s}"
            );
            let (median, mean) = (s["median"].as_f64().unwrap(), s["mean"].as_f64().unwrap());
            let range = s["range"].as_array().unwrap();
            let ci = s["ci95"].as_array().unwrap();
            assert!(range[0].as_f64().unwrap() <= median && median <= range[1].as_f64().unwrap());
            assert!(ci[0].as_f64().unwrap() <= mean && mean <= ci[1].as_f64().unwrap());
        }
    }
    if v["devices"].as_u64().unwrap_or(0) < 2 {
        assert_eq!(rows[1]["scaling"], "time-shared, not scaling", "{text}");
        assert!(rows[1]["efficiency"].is_null(), "{}", rows[1]);
        assert!(
            out.contains("time-shared, not scaling"),
            "the stdout does not refuse: {out}"
        );
    }
    for row in rows {
        println!(
            "[140] reduced strong: P = {} t_iter median {:.3e} s, L_ex median \
             {:.3e} s, L_ag median {:.3e} s, scaling {}",
            row["np"],
            row["t_iter"]["median"].as_f64().unwrap(),
            row["l_ex"]["median"].as_f64().unwrap(),
            row["l_ag"]["median"].as_f64().unwrap(),
            row["scaling"],
        );
    }
}

/// A device listed TWICE is still ONE device: with `OFGPU_DEVICES=0,0` the
/// weak bench reports `devices` 1, and its P = 2 row - the weak box grown
/// to (2*n) x (1*n) - is time-shared with a null efficiency.
#[test]
fn a_listed_device_twice_is_still_time_shared() {
    let dir = temp_dir("weak");
    let json = dir.join("d.json");
    let (_out, _err, text) = bench_to_json(
        exe()
            .env("OFGPU_DEVICES", "0,0")
            .args([
                "-bench", "weak", "-np", "2", "-n", "8", "-warm", "1",
                "-iters", "1", "-exchanges", "5",
            ]),
        &json,
        &dir,
    );
    let v: serde_json::Value =
        serde_json::from_str(&text).expect("the bench JSON parses");
    assert_eq!(v["devices"], 1, "{text}");
    let rows = v["rows"].as_array().expect("rows");
    assert_eq!(rows.len(), 1, "{text}");
    let row = &rows[0];
    assert_eq!(row["np"], 2);
    assert_eq!(row["nx"], 16, "the weak box at P = 2 is 2n x n: {text}");
    assert_eq!(row["ny"], 8, "{text}");
    assert_eq!(row["cells"], 128, "{text}");
    assert_eq!(row["cells_per_rank"], 64.0, "{text}");
    assert_eq!(row["scaling"], "time-shared, not scaling", "{text}");
    assert!(row["efficiency"].is_null(), "{text}");
    assert!(row["speedup"].is_null(), "{text}");
}

/// The plan's gate (docs/17 §12, row PAR-16): the harness runs P = 1..4
/// strong then weak at the DEFAULTS in under 600 s, the threshold fixed
/// BEFORE measuring. The wall time is the larger of the sum of the two JSON
/// "seconds" and the two child processes' wall clock.
/// A miss is REPORTED and the test continues - a shared card is slower,
/// not wrong - so the assertion is on the exit statuses and the row counts
/// only; the timings are printed, never gated.
#[test]
fn the_harness_runs_p1_to_p4_in_ten_minutes() {
    let dir = temp_dir("gate");
    let a_json = dir.join("a.json");
    let b_json = dir.join("b.json");
    // The wall clock of the two child processes, start-up included, beside
    // the harness's own "seconds"; the gate reads the larger.
    let t0 = std::time::Instant::now();
    let (_out, _err, a_text) =
        bench_to_json(exe().args(["-bench", "strong"]), &a_json, &dir);
    let (_out, _err, b_text) = bench_to_json(exe().args(["-bench", "weak"]), &b_json, &dir);
    let wall = t0.elapsed().as_secs_f64();
    let a: serde_json::Value = serde_json::from_str(&a_text).expect("strong JSON");
    let b: serde_json::Value = serde_json::from_str(&b_text).expect("weak JSON");
    for (v, name) in [(&a, "strong"), (&b, "weak")] {
        assert_eq!(v["rows"].as_array().expect("rows").len(), 4, "{name}: {v}");
    }
    let total = (a["seconds"].as_f64().unwrap() + b["seconds"].as_f64().unwrap()).max(wall);
    for (name, v) in [("strong", &a), ("weak", &b)] {
        for row in v["rows"].as_array().unwrap() {
            println!(
                "[140] harness row: bench {name} P = {} cells {} t_iter median \
                 {:.3e} s, L_ex median {:.3e} s, L_ag median {:.3e} s, scaling {}",
                row["np"],
                row["cells"],
                row["t_iter"]["median"].as_f64().unwrap(),
                row["l_ex"]["median"].as_f64().unwrap(),
                row["l_ag"]["median"].as_f64().unwrap(),
                row["scaling"],
            );
        }
    }
    if total < 600.0 {
        println!("[140] harness P = 1..4: {total:.1} s < 600 s");
    } else {
        println!("[140] MISSED harness P = 1..4: {total:.1} s >= 600 s (shared GPU)");
    }
}
