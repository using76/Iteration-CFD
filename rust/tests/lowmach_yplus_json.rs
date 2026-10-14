// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.
// Provenance: see PROVENANCE.md. No GPL-licensed source was consulted.

//! SPEC-LIT 29.3's note (docs/15 AR-1, AUT-15): `ofgpu-lowmach` writes the
//! per-patch first-cell y+ it prints as `yPlus.json` in the output root.
//! The shipped `cases/channelPeriodicFluxWF.jsonc` is run as the binary
//! itself, and every `y+ at` line on its stdout is read back against the
//! file: the same patches, and each JSON double formatted by the printer's
//! own `%g` (`common::g`) is the printed token. A GPU is needed; without one
//! the test says so and passes.

#[path = "../src/bin/common/mod.rs"]
#[allow(dead_code)]
mod common;

use std::path::PathBuf;
use std::process::Command;

/// `(patch, min, mean, max)` tokens of every `  y+ at NAME: min A | mean B |
/// max C` line, as printed.
fn printed_yplus(stdout: &str) -> Vec<(String, String, String, String)> {
    stdout
        .lines()
        .filter_map(|l| {
            let rest = l.strip_prefix("  y+ at ")?;
            let (name, nums) = rest.split_once(": min ")?;
            let (min, nums) = nums.split_once(" | mean ")?;
            let (mean, max) = nums.split_once(" | max ")?;
            Some((name.to_string(), min.to_string(), mean.to_string(), max.trim().to_string()))
        })
        .collect()
}

#[test]
fn the_yplus_json_equals_the_printed_line_on_channel_periodic_flux_wf() {
    if ofgpu::Gpu::new(0).is_err() {
        eprintln!("no CUDA device: the y+ JSON run needs one; skipped");
        return;
    }
    let shipped = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("cases")
        .join("channelPeriodicFluxWF.jsonc");
    let dir = std::env::temp_dir().join(format!("ofgpu-aut15-yplus-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("scratch dir");
    let case = dir.join("channelPeriodicFluxWF.jsonc");
    std::fs::copy(&shipped, &case).expect("copy the shipped case");

    let out = Command::new(env!("CARGO_BIN_EXE_ofgpu-lowmach"))
        .arg(&case)
        .args(["-iters", "20"])
        .output()
        .expect("run ofgpu-lowmach");
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        out.status.success(),
        "ofgpu-lowmach failed: {}\n{}",
        out.status,
        String::from_utf8_lossy(&out.stderr)
    );

    let printed = printed_yplus(&stdout);
    let names: Vec<&str> = printed.iter().map(|p| p.0.as_str()).collect();
    assert_eq!(names, ["wallHot0", "wallHot1"], "the printed y+ lines");

    let path = common::json_case_output_dir(&case).join("yPlus.json");
    let text = std::fs::read_to_string(&path).expect("yPlus.json beside the run");
    let doc: serde_json::Value = serde_json::from_str(&text).expect("yPlus.json parses");
    assert_eq!(doc["solver"], "ofgpu-lowmach");
    assert_eq!(doc["turbulence"], "kEpsilon");
    let patches = doc["patches"].as_array().expect("patches[]");
    assert_eq!(patches.len(), printed.len(), "one JSON entry per printed line");

    for (j, (name, pmin, pmean, pmax)) in patches.iter().zip(&printed) {
        assert_eq!(j["patch"], name.as_str());
        // 8 x 6 x 1 cells: each y wall carries 8 faces.
        assert_eq!(j["faces"], 8, "{name}: faces");
        let v = |k: &str| j[k].as_f64().unwrap_or_else(|| panic!("{name}: {k} is a number"));
        let (min, mean, max) = (v("min"), v("mean"), v("max"));
        assert_eq!(common::g(min), *pmin, "{name}: min");
        assert_eq!(common::g(mean), *pmean, "{name}: mean");
        assert_eq!(common::g(max), *pmax, "{name}: max");
        assert!(0.0 < min && min <= mean && mean <= max, "{name}: {min} <= {mean} <= {max}");
    }
    let _ = std::fs::remove_dir_all(&dir);
}
