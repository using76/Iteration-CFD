// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.
// Provenance: see PROVENANCE.md. No GPL-licensed source was consulted.

//! SPEC-LIT 120.1's device Function1 through the public API, in BOTH builds:
//! the evaluation is f64 on the device in both, so the device stays bit for
//! bit its host twin, and under `single` a timed patch's entry is the f32
//! cast of the same f64 value, rounded ONCE. No
//! `#[cfg_attr(feature = "single", ignore)]` anywhere.

use std::collections::BTreeMap;

use ofgpu::device::Gpu;
use ofgpu::function1::{
    DeviceF1, DeviceTimedPatch, F1, F1Clock, F1Kernels, F1Target, TimedPatch,
};
use ofgpu::{DevBuf, Scalar};

const SETTING: &str = "p: boundaryField/inlet/uniformValue";

/// A hand-made entry map for [`F1::parse`].
fn ent(entries: &[(&str, &str)]) -> BTreeMap<String, String> {
    entries
        .iter()
        .map(|(k, v)| (k.to_string(), v.to_string()))
        .collect()
}

/// Parse one entry, expecting it present.
fn parse(entries: &BTreeMap<String, String>, n_c: usize) -> F1 {
    F1::parse(entries, "uniformValue", n_c, SETTING, None)
        .unwrap()
        .expect("the entry is present")
}

/// Three of the lib gate's cases - the sine over a 1000 s span, the repeat
/// table overrunning both ends, and the scale - at 1000 arguments each, all
/// three components bitwise in BOTH builds.
#[test]
fn f32_device_equals_host_bitwise() {
    let Ok(gpu) = Gpu::new(0) else { return };
    let k = F1Kernels::new(&gpu).unwrap();

    let sine = parse(&ent(&[
        ("uniformValue/type", "sine"),
        ("uniformValue/amplitude", "2"),
        ("uniformValue/frequency", "3.7"),
        ("uniformValue/start", "0.1"),
        ("uniformValue/level", "0.5"),
    ]), 1);
    let repeat = parse(&ent(&[
        ("uniformValue/type", "table"),
        ("uniformValue/values", "((0 0) (1 2) (3 -2))"),
        ("uniformValue/outOfBounds", "repeat"),
    ]), 1);
    let scale = parse(&ent(&[
        ("uniformValue/type", "scale"),
        ("uniformValue/scale/type", "halfCosineRamp"),
        ("uniformValue/scale/start", "0"),
        ("uniformValue/scale/duration", "2"),
        ("uniformValue/value", "table ((0 (1 0 0)) (2 (3 4 -1)))"),
    ]), 3);

    let cases: [(&str, &F1, f64, f64); 3] = [
        ("sine", &sine, 0.0, 1000.0),
        ("table repeat", &repeat, -10.0, 20.0),
        ("scale", &scale, -1.0, 4.0),
    ];
    for (name, f, lo, hi) in cases {
        let xs: Vec<f64> = (0..1000)
            .map(|kk| lo + (hi - lo) * kk as f64 / 999.0)
            .collect();
        let df = DeviceF1::upload(&gpu, f).unwrap();
        let xdev = gpu.upload(&xs).unwrap();
        let mut out = gpu.zeros::<f64>(3 * xs.len()).unwrap();
        df.eval_many(&gpu, &k, &xdev, &mut out, xs.len()).unwrap();
        let got = gpu.download(&out).unwrap();
        for (kk, &x) in xs.iter().enumerate() {
            let want = f.eval(x);
            for c in 0..3 {
                assert_eq!(
                    got[3 * kk + c].to_bits(),
                    want[c].to_bits(),
                    "{name}: k {kk} x {x}: device {} vs host {} (component {c})",
                    got[3 * kk + c],
                    want[c],
                );
            }
        }
    }
}

/// The scalar sine patch of the lib test, driven through the clock's host
/// write: for twenty steps of 1e-4 s past t = 1000 the device entries equal
/// `host_values` bitwise - which under `single` is the f32 cast of the f64
/// value, rounded once.
#[test]
fn f32_apply_rounds_once() {
    let Ok(gpu) = Gpu::new(0) else { return };
    let k = F1Kernels::new(&gpu).unwrap();

    let sine = parse(&ent(&[
        ("uniformValue/type", "sine"),
        ("uniformValue/amplitude", "2"),
        ("uniformValue/frequency", "3.7"),
        ("uniformValue/start", "0.1"),
        ("uniformValue/level", "0.5"),
    ]), 1);
    let patch = TimedPatch {
        patch: "sine".to_string(),
        start: 1,
        n: 5,
        n_t: 1,
        target: F1Target::RefValue,
        f: sine,
        base: (0..5).map(|i| 0.5 + 0.1 * i as f64).collect(),
        w: (0..5).map(|i| 0.5 + 0.1 * i as f64).collect(),
    };
    let dp = DeviceTimedPatch::upload(&gpu, &patch).unwrap();
    let mut clock = F1Clock::new(&gpu, 1000.0, 1.0e-4).unwrap();
    let mut dst: DevBuf<Scalar> = gpu.upload(&vec![0.0 as Scalar; 8]).unwrap();

    for kk in 0..20 {
        let t = 1000.0 + kk as f64 * 1.0e-4;
        clock.set(&gpu, t, 1.0e-4).unwrap();
        dp.apply_scalar(&gpu, &k, &clock, &mut dst).unwrap();
        let got = gpu.download(&dst).unwrap();
        let want = patch.host_values(t);
        for i in 0..5 {
            assert_eq!(
                got[1 + i].to_bits(),
                want[i].to_bits(),
                "k {kk} t {t}: face {}: device {} vs host {}",
                1 + i,
                got[1 + i],
                want[i],
            );
        }
    }
}
