// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.

//! SPEC-LIT 120.1's host twin, exercised at exact arguments and refused
//! by name - all host only, no GPU.
//!
//! No GPL-licensed source was consulted.

use super::*;

/// A hand-made entry map, so the reader itself is what every test drives.
fn ent(entries: &[(&str, &str)]) -> BTreeMap<String, String> {
    entries
        .iter()
        .map(|(k, v)| (k.to_string(), v.to_string()))
        .collect()
}

/// The setting every message in these tests starts with.
const SETTING: &str = "p: boundaryField/inlet/uniformValue";

/// [`F1::parse`] at the one key these tests use, expecting the entry there.
fn parse(entries: &BTreeMap<String, String>, n_c: usize) -> Result<F1> {
    let f = F1::parse(entries, "uniformValue", n_c, SETTING, None)?
        .expect("the entry is present");
    Ok(f)
}

/// `S2`/`C2` agree with `sin`/`cos` to 2e-15 over [0, 1), hit the quadrant
/// values exactly, and are periodic bit for bit - the last because the
/// range reduction in turns is exact (SPEC-LIT 120.1's gate).
#[test]
fn s2_c2_match_std_to_2e_15() {
    for k in 0..100000 {
        let c = k as f64 / 100000.0;
        let (s, co) = s2c2(c);
        let target = 2.0 * std::f64::consts::PI * c;
        assert!(
            (s - target.sin()).abs() <= 2e-15,
            "c = {c}: {s} vs {}",
            target.sin()
        );
        assert!(
            (co - target.cos()).abs() <= 2e-15,
            "c = {c}: {co} vs {}",
            target.cos()
        );
    }

    assert_eq!(s2c2(0.0), (0.0, 1.0));
    assert_eq!(s2c2(0.25).0, 1.0);
    assert_eq!(s2c2(0.5).1, -1.0);
    assert_eq!(s2c2(0.75).0, -1.0);

    for j in 0..1024 {
        let c = j as f64 / 1024.0;
        assert_eq!(s2c2(c + 3.0).0.to_bits(), s2c2(c).0.to_bits());
        assert_eq!(s2c2(c + 3.0).1.to_bits(), s2c2(c).1.to_bits());
    }
}

/// The table's two schemes and its end rules, exact at these arguments
/// ((120.2)-(120.4)), read through forms 3 and 5 both.
#[test]
fn table_linear_step_and_bounds() {
    // linear, clamp - the defaults, form 3 inline.
    let f = parse(&ent(&[("uniformValue", "table ((0 0) (1 2) (3 -2))")]), 1).unwrap();
    assert_eq!(f.eval(0.5), [1.0, 0.0, 0.0]);
    assert_eq!(f.eval(2.0), [0.0, 0.0, 0.0]);
    assert_eq!(f.eval(3.0), [-2.0, 0.0, 0.0]);
    assert_eq!(f.eval(-1.0), [0.0, 0.0, 0.0]);
    assert_eq!(f.eval(5.0), [-2.0, 0.0, 0.0]);

    // step, form 5.
    let f = parse(
        &ent(&[
            ("uniformValue/type", "table"),
            ("uniformValue/values", "((0 0) (1 2) (3 -2))"),
            ("uniformValue/interpolationScheme", "step"),
        ]),
        1,
    )
    .unwrap();
    assert_eq!(f.eval(0.5)[0], 0.0);
    assert_eq!(f.eval(1.0)[0], 2.0);
    assert_eq!(f.eval(2.9)[0], 2.0);
    assert_eq!(f.eval(3.0)[0], -2.0);

    // repeat: 3.5 wraps to 0.5 and -0.5 wraps to 2.5 by (120.4).
    let f = parse(
        &ent(&[
            ("uniformValue/type", "table"),
            ("uniformValue/values", "((0 0) (1 2) (3 -2))"),
            ("uniformValue/outOfBounds", "repeat"),
        ]),
        1,
    )
    .unwrap();
    assert_eq!(f.eval(3.5)[0], 1.0);
    assert_eq!(f.eval(-0.5)[0], -1.0);

    // A vector table, n_c 3, at 1 -> [2, 2, 0].
    let f = parse(&ent(&[("uniformValue", "table ((0 (1 0 0)) (2 (3 4 0)))")]), 3).unwrap();
    assert_eq!(f.eval(1.0), [2.0, 2.0, 0.0]);
}

/// The five ramps and the two waves at their exact arguments
/// ((120.6)-(120.8)), all read through form-5 dicts.
#[test]
fn ramps_and_waves() {
    let ramp = |kind: &str| {
        parse(
            &ent(&[
                ("uniformValue/type", kind),
                ("uniformValue/start", "1"),
                ("uniformValue/duration", "2"),
            ]),
            1,
        )
        .unwrap()
    };

    let f = ramp("linearRamp");
    assert_eq!(f.eval(0.0)[0], 0.0);
    assert_eq!(f.eval(2.0)[0], 0.5);
    assert_eq!(f.eval(4.0)[0], 1.0);

    let f = ramp("quadraticRamp");
    assert_eq!(f.eval(2.0)[0], 0.25);

    let f = ramp("halfCosineRamp");
    assert_eq!(f.eval(2.0)[0], 0.5);

    let f = ramp("quarterSineRamp");
    assert_eq!(f.eval(3.0)[0], 1.0);

    let f = ramp("quarterCosineRamp");
    assert_eq!(f.eval(3.0)[0], 1.0);
    assert_eq!(f.eval(1.0)[0], 0.0);

    // sine at a quadrant: phi = 0.5*(0.75 - 0.25) = 0.25 turns, so the wave
    // is exactly at its crest (120.6).
    let f = parse(
        &ent(&[
            ("uniformValue/type", "sine"),
            ("uniformValue/amplitude", "2"),
            ("uniformValue/frequency", "0.5"),
            ("uniformValue/start", "0.25"),
            ("uniformValue/level", "1"),
        ]),
        1,
    )
    .unwrap();
    assert_eq!(f.eval(0.75)[0], 3.0);

    // square with markSpace 3: three quarters of a period at +1 (120.7).
    let f = parse(
        &ent(&[
            ("uniformValue/type", "square"),
            ("uniformValue/amplitude", "1"),
            ("uniformValue/frequency", "1"),
            ("uniformValue/level", "0"),
            ("uniformValue/markSpace", "3"),
        ]),
        1,
    )
    .unwrap();
    assert_eq!(f.eval(0.5)[0], 1.0);
    assert_eq!(f.eval(0.8)[0], -1.0);
    assert_eq!(f.eval(1.1)[0], 1.0);

    // A vector sine: F_c = L_c + (A*S_c)*S2(phi), phi = 0.25 turns here.
    let f = parse(
        &ent(&[
            ("uniformValue/type", "sine"),
            ("uniformValue/amplitude", "1"),
            ("uniformValue/frequency", "1"),
            ("uniformValue/level", "(1 0 0)"),
            ("uniformValue/scale", "(0 1 0)"),
        ]),
        3,
    )
    .unwrap();
    assert_eq!(f.eval(0.25), [1.0, 1.0, 0.0]);
}

/// The power series (120.5) sums in k order with exponents by repeated
/// squaring; a non-integer or oversized exponent is refused by name.
#[test]
fn polynomial_has_integer_exponents() {
    let f = parse(
        &ent(&[("uniformValue", "polynomial ((1 0) (2 1) (-0.5 2))")]),
        1,
    )
    .unwrap();
    assert_eq!(f.eval(2.0)[0], 3.0);
    assert_eq!(f.eval(-1.0)[0], -1.5);

    for bad in ["((1 0.5))", "((1 65))"] {
        let raw = format!("polynomial {bad}");
        let e = parse(&ent(&[("uniformValue", &raw)]), 1)
            .unwrap_err()
            .to_string();
        assert!(e.contains("integer exponent"), "{e}");
        assert!(e.contains(SETTING), "{e}");
    }
}

/// `scale` composes a scalar child with a child of the entry's rank
/// ((120.9)), the children read by `F1::parse` at their own keys; a scale
/// inside a scale is refused.
#[test]
fn scale_of_a_ramp() {
    let f = parse(
        &ent(&[
            ("uniformValue/type", "scale"),
            ("uniformValue/scale/type", "linearRamp"),
            ("uniformValue/scale/start", "0"),
            ("uniformValue/scale/duration", "4"),
            ("uniformValue/value", "constant (2 0 0)"),
        ]),
        3,
    )
    .unwrap();
    assert_eq!(f.eval(1.0), [0.5, 0.0, 0.0]);

    let e = parse(
        &ent(&[
            ("uniformValue/type", "scale"),
            ("uniformValue/scale/type", "constant"),
            ("uniformValue/scale/value", "1"),
            ("uniformValue/value/type", "scale"),
            ("uniformValue/value/scale/type", "constant"),
            ("uniformValue/value/scale/value", "1"),
            ("uniformValue/value/value", "constant (1 0 0)"),
        ]),
        3,
    )
    .unwrap_err()
    .to_string();
    assert!(e.contains("scale inside a scale"), "{e}");
    assert!(e.contains(SETTING), "{e}");
}

/// Forms 1-5 all read the same constant, a bare vector reads with n_c 3,
/// and an absent key is Ok(None) (SPEC-LIT 120.1's five forms).
#[test]
fn the_five_forms_read_the_same_constant() {
    let want = Function1::Constant { v: [2.5, 0.0, 0.0] };

    // 1: bare.  2: named.  and `uniform <v>` accepted as a constant too.
    assert_eq!(parse(&ent(&[("uniformValue", "2.5")]), 1).unwrap().f, want);
    assert_eq!(
        parse(&ent(&[("uniformValue", "constant 2.5")]), 1).unwrap().f,
        want
    );
    assert_eq!(parse(&ent(&[("uniformValue", "uniform 2.5")]), 1).unwrap().f, want);

    // 4: the bare type word, parameters in `KCoeffs`.
    assert_eq!(
        parse(
            &ent(&[("uniformValue", "constant"), ("uniformValueCoeffs/value", "2.5")]),
            1
        )
        .unwrap()
        .f,
        want
    );

    // 5: the type and the parameters in one sub-dictionary.
    assert_eq!(
        parse(
            &ent(&[("uniformValue/type", "constant"), ("uniformValue/value", "2.5")]),
            1
        )
        .unwrap()
        .f,
        want
    );

    assert_eq!(
        parse(&ent(&[("uniformValue", "(1 2 3)")]), 3).unwrap().f,
        Function1::Constant { v: [1.0, 2.0, 3.0] }
    );
    assert!(parse(&ent(&[("uniformValue", "2.5")]), 1).unwrap().is_constant());

    assert!(
        F1::parse(&ent(&[("other", "2.5")]), "uniformValue", 1, SETTING, None)
            .unwrap()
            .is_none()
    );
}

/// `tableFile` and `csvFile` both become a table: comments stripped out of
/// the .dat, headers and the doubled separator handled in the .csv, and a
/// missing file refused naming the resolved path (SPEC-LIT 120.1).
#[test]
fn table_file_and_csv_file_are_tables() {
    let dir = std::env::temp_dir().join(format!("ofgpu_f1_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("inlet.dat"), "// c\n( (0 1) /* x */ (2 5) )\n").unwrap();
    std::fs::write(dir.join("inlet.csv"), "t;u;v;w\nunits\n0;;1;0;0\n2;5;0;0\n").unwrap();

    let inline = parse(&ent(&[("uniformValue", "table ((0 1) (2 5))")]), 1).unwrap();
    let f = F1::parse(
        &ent(&[
            ("uniformValue/type", "tableFile"),
            ("uniformValue/file", "<case>/inlet.dat"),
        ]),
        "uniformValue",
        1,
        SETTING,
        Some(&dir),
    )
    .unwrap()
    .unwrap();
    assert_eq!(f.f, inline.f);

    let f = F1::parse(
        &ent(&[
            ("uniformValue/type", "csvFile"),
            ("uniformValue/file", "inlet.csv"),
            ("uniformValue/nHeaderLine", "2"),
            ("uniformValue/refColumn", "0"),
            ("uniformValue/componentColumns", "(1 2 3)"),
            ("uniformValue/separator", ";"),
            ("uniformValue/mergeSeparators", "true"),
        ]),
        "uniformValue",
        3,
        SETTING,
        Some(&dir),
    )
    .unwrap()
    .unwrap();
    match f.f {
        Function1::Table { x, y, interp: Interp::Linear, oob: OutOfBounds::Clamp } => {
            assert_eq!(x, vec![0.0, 2.0]);
            assert_eq!(y, vec![[1.0, 0.0, 0.0], [5.0, 0.0, 0.0]]);
        }
        other => panic!("a csvFile is a table, got {other:?}"),
    }

    let e = F1::parse(
        &ent(&[
            ("uniformValue/type", "tableFile"),
            ("uniformValue/file", "<case>/missing.dat"),
        ]),
        "uniformValue",
        1,
        SETTING,
        Some(&dir),
    )
    .unwrap_err()
    .to_string();
    assert!(e.contains("cannot read"), "{e}");
    assert!(e.contains(SETTING), "{e}");

    let _ = std::fs::remove_dir_all(&dir);
}

/// Out of bounds is settled at setup (SPEC-LIT 120.1): `error` refuses the
/// range naming the setting and the table's ends, `warn` warns once and
/// passes, `clamp` and `repeat` pass whatever the range.
#[test]
fn out_of_bounds_is_settled_at_setup() {
    let table = |oob: &str| {
        parse(
            &ent(&[
                ("uniformValue/type", "table"),
                ("uniformValue/values", "((0 0) (3 3))"),
                ("uniformValue/outOfBounds", oob),
            ]),
            1,
        )
        .unwrap()
    };

    assert!(table("error").check_range(0.0, 3.0).is_ok());
    let e = table("error").check_range(-1.0, 2.0).unwrap_err().to_string();
    assert!(e.contains(SETTING), "{e}");
    assert!(e.contains("outOfBounds error"), "{e}");
    assert!(e.contains("-1") && e.contains("2") && e.contains("0") && e.contains("3"), "{e}");

    let _g = crate::io::contract::permissive_test_guard();
    crate::io::contract::reset_warnings();
    assert!(table("warn").check_range(-1.0, 2.0).is_ok());
    assert!(crate::io::contract::warned(SETTING));
    drop(_g);

    assert!(table("clamp").check_range(-1.0, 2.0).is_ok());
    assert!(table("repeat").check_range(-1.0, 2.0).is_ok());
}

/// Every refusal names the setting and says which - SPEC-LIT 120.1's
/// refused-by-name list, one phrase each.
#[test]
fn refusals_name_the_setting() {
    let refuse = |entries: &BTreeMap<String, String>, n_c: usize, phrase: &str| {
        let e = parse(entries, n_c).unwrap_err().to_string();
        assert!(e.contains(SETTING), "{e}");
        assert!(e.contains(phrase), "{e}");
    };

    refuse(
        &ent(&[("uniformValue/type", "cosine")]),
        1,
        "unknown Function1 type 'cosine'",
    );
    refuse(
        &ent(&[
            ("uniformValue/type", "linearRamp"),
            ("uniformValue/duration", "1"),
        ]),
        3,
        "is a scalar Function1",
    );
    refuse(
        &ent(&[
            ("uniformValue/type", "table"),
            ("uniformValue/values", "((0 0) (1 2))"),
            ("uniformValue/xScale", "2"),
        ]),
        1,
        "xScale",
    );
    refuse(&ent(&[("uniformValue", "table ()")]), 1, "no points");
    refuse(
        &ent(&[("uniformValue", "table ((1 0) (0 1))")]),
        1,
        "strictly increasing",
    );
    refuse(
        &ent(&[
            ("uniformValue/type", "table"),
            ("uniformValue/values", "((0 0))"),
            ("uniformValue/outOfBounds", "repeat"),
        ]),
        1,
        "repeat needs two points",
    );
    refuse(
        &ent(&[
            ("uniformValue/type", "sine"),
            ("uniformValue/amplitude", "1"),
            ("uniformValue/frequency", "0"),
        ]),
        1,
        "frequency must be > 0",
    );
    refuse(
        &ent(&[
            ("uniformValue/type", "linearRamp"),
            ("uniformValue/duration", "-1"),
        ]),
        1,
        "duration must be > 0",
    );
    refuse(
        &ent(&[
            ("uniformValue/type", "square"),
            ("uniformValue/amplitude", "1"),
            ("uniformValue/frequency", "1"),
            ("uniformValue/markSpace", "0"),
        ]),
        1,
        "markSpace must be > 0",
    );
    refuse(
        &ent(&[
            ("uniformValue/type", "csvFile"),
            ("uniformValue/file", "nope.csv"),
            ("uniformValue/componentColumns", "(1)"),
        ]),
        3,
        "componentColumns",
    );
}

// ==========================================================================
//  SPEC-LIT 120.1 on the device: the kernels, the clock, the timed patches,
//  and the bitwise and capture gates
// ==========================================================================

use crate::capture::capture_replays_bitwise;
use crate::device::{DevBuf, Gpu};
use crate::{Scalar, Vec3};

/// One case of the bitwise gate: its name, its rank, its entries, whether
/// they resolve against the case directory, and the argument span.
struct BitwiseCase {
    name: &'static str,
    n_c: usize,
    entries: Vec<(&'static str, &'static str)>,
    from_case_dir: bool,
    lo: f64,
    hi: f64,
}

/// The nineteen cases of the bitwise gate, every type, both table schemes
/// and both device out-of-bounds modes, spanning and overrunning each
/// function's range (SPEC-LIT 120.1's gate).
fn bitwise_cases() -> Vec<BitwiseCase> {
    // Form 5: `uniformValue/type` first, then the parameters at their
    // full `<prefix>/<name>` keys.
    let form5 = |kind: &'static str, params: &[(&'static str, &'static str)]| {
        let mut v = Vec::with_capacity(params.len() + 1);
        v.push(("uniformValue/type", kind));
        v.extend_from_slice(params);
        v
    };
    let mut v: Vec<BitwiseCase> = Vec::new();
    v.push(BitwiseCase {
        name: "constant 2.5", n_c: 1, from_case_dir: false, lo: -1.0, hi: 1.0,
        entries: vec![("uniformValue", "2.5")],
    });
    v.push(BitwiseCase {
        name: "constant (1 -2 3)", n_c: 3, from_case_dir: false, lo: -1.0, hi: 1.0,
        entries: vec![("uniformValue", "(1 -2 3)")],
    });
    v.push(BitwiseCase {
        name: "table linear clamp", n_c: 1, from_case_dir: false, lo: -2.0, hi: 6.0,
        entries: vec![("uniformValue", "table ((0 0) (1 2) (3 -2))")],
    });
    v.push(BitwiseCase {
        name: "table step clamp", n_c: 1, from_case_dir: false, lo: -2.0, hi: 6.0,
        entries: form5("table", &[
            ("uniformValue/values", "((0 0) (1 2) (3 -2))"),
            ("uniformValue/interpolationScheme", "step"),
        ]),
    });
    v.push(BitwiseCase {
        name: "table linear repeat", n_c: 1, from_case_dir: false, lo: -10.0, hi: 20.0,
        entries: form5("table", &[
            ("uniformValue/values", "((0 0) (1 2) (3 -2))"),
            ("uniformValue/outOfBounds", "repeat"),
        ]),
    });
    v.push(BitwiseCase {
        name: "table step repeat", n_c: 1, from_case_dir: false, lo: -10.0, hi: 20.0,
        entries: form5("table", &[
            ("uniformValue/values", "((0 0) (1 2) (3 -2))"),
            ("uniformValue/interpolationScheme", "step"),
            ("uniformValue/outOfBounds", "repeat"),
        ]),
    });
    v.push(BitwiseCase {
        name: "vector table", n_c: 3, from_case_dir: false, lo: -1.0, hi: 3.0,
        entries: vec![("uniformValue", "table ((0 (1 0 0)) (2 (3 4 -1)))")],
    });
    v.push(BitwiseCase {
        name: "tableFile", n_c: 1, from_case_dir: true, lo: -1.0, hi: 3.0,
        entries: vec![("uniformValue/type", "tableFile"), ("uniformValue/file", "<case>/inlet.dat")],
    });
    v.push(BitwiseCase {
        name: "csvFile", n_c: 1, from_case_dir: true, lo: -1.0, hi: 3.0,
        entries: form5("csvFile", &[
            ("uniformValue/file", "inlet.csv"),
            ("uniformValue/nHeaderLine", "2"),
            ("uniformValue/refColumn", "0"),
            ("uniformValue/componentColumns", "(1)"),
            ("uniformValue/separator", ";"),
            ("uniformValue/mergeSeparators", "true"),
        ]),
    });
    v.push(BitwiseCase {
        name: "polynomial", n_c: 1, from_case_dir: false, lo: 0.1, hi: 4.0,
        entries: vec![(
            "uniformValue",
            "polynomial ((1 0) (2 1) (-0.5 2) (0.125 3) (1 -1))",
        )],
    });
    v.push(BitwiseCase {
        name: "sine scalar", n_c: 1, from_case_dir: false, lo: 0.0, hi: 1000.0,
        entries: form5("sine", &[
            ("uniformValue/amplitude", "2"),
            ("uniformValue/frequency", "3.7"),
            ("uniformValue/start", "0.1"),
            ("uniformValue/level", "0.5"),
        ]),
    });
    v.push(BitwiseCase {
        name: "sine vector", n_c: 3, from_case_dir: false, lo: 0.0, hi: 10.0,
        entries: form5("sine", &[
            ("uniformValue/amplitude", "1"),
            ("uniformValue/frequency", "1.3"),
            ("uniformValue/level", "(1 0 0)"),
            ("uniformValue/scale", "(0 1 2)"),
        ]),
    });
    v.push(BitwiseCase {
        name: "square", n_c: 1, from_case_dir: false, lo: -5.0, hi: 5.0,
        entries: form5("square", &[
            ("uniformValue/amplitude", "1"),
            ("uniformValue/frequency", "1.3"),
            ("uniformValue/markSpace", "0.6"),
        ]),
    });
    for kind in [
        "linearRamp",
        "quadraticRamp",
        "halfCosineRamp",
        "quarterSineRamp",
        "quarterCosineRamp",
    ] {
        v.push(BitwiseCase {
            name: kind, n_c: 1, from_case_dir: false, lo: -1.0, hi: 3.0,
            entries: form5(kind, &[
                ("uniformValue/start", "0.2"),
                ("uniformValue/duration", "1.5"),
            ]),
        });
    }
    v.push(BitwiseCase {
        name: "scale of halfCosineRamp and the vector table", n_c: 3,
        from_case_dir: false, lo: -1.0, hi: 4.0,
        entries: form5("scale", &[
            ("uniformValue/scale/type", "halfCosineRamp"),
            ("uniformValue/scale/start", "0"),
            ("uniformValue/scale/duration", "2"),
            ("uniformValue/value", "table ((0 (1 0 0)) (2 (3 4 -1)))"),
        ]),
    });
    v
}

/// SPEC-LIT 120.1's gate: every type, both table schemes and both device
/// out-of-bounds modes at 1000 arguments spanning and overrunning each
/// function's range, device against host, bit for bit - `a*b + c` stays two
/// roundings and every sum keeps its written order on both sides.
#[test]
fn device_equals_host_bitwise_at_1000_points_per_type() {
    let Ok(gpu) = Gpu::new(0) else { return };
    let k = F1Kernels::new(&gpu).unwrap();

    let dir = std::env::temp_dir().join(format!("ofgpu_f1_dev_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("inlet.dat"), "// c\n( (0 1) /* x */ (2 5) )\n").unwrap();
    std::fs::write(dir.join("inlet.csv"), "t;u;v;w\nunits\n0;;1;0;0\n2;5;0;0\n").unwrap();

    let cases = bitwise_cases();
    assert_eq!(cases.len(), 19, "the gate's census");
    for case in &cases {
        let entries = ent(&case.entries);
        let dir_arg = case.from_case_dir.then_some(dir.as_path());
        let f = F1::parse(&entries, "uniformValue", case.n_c, SETTING, dir_arg)
            .unwrap()
            .expect("the entry is present");
        let xs: Vec<f64> = (0..1000)
            .map(|kk| case.lo + (case.hi - case.lo) * kk as f64 / 999.0)
            .collect();
        let df = DeviceF1::upload(&gpu, &f).unwrap();
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
                    "{}: k {kk} x {x}: device {} vs host {} (component {c})",
                    case.name,
                    got[3 * kk + c],
                    want[c],
                );
            }
        }
    }
    println!(
        "device_equals_host_bitwise: {} cases at 1000 points each, all bitwise",
        cases.len()
    );
    let _ = std::fs::remove_dir_all(&dir);
}

/// SPEC-LIT 81's protocol, the gate of the registry row for
/// `src/function1/device.rs`: a `flowRateInletVelocity`-style patch whose
/// rate is `linearRamp`-scaled (SPEC-LIT 120.1's table of conditions), one
/// step = `advance` + apply, captured and replayed three times bit for bit
/// with the time moving.
#[test]
fn the_ramped_inlet_replays_bitwise() {
    let Ok(gpu) = Gpu::new(0) else { return };

    type Rig = (DeviceTimedPatch, F1Clock, DevBuf<Vec3>, F1Kernels);
    let build = || -> crate::error::Result<Rig> {
        let f = F1::parse(
            &ent(&[
                ("uniformValue/type", "scale"),
                ("uniformValue/scale/type", "linearRamp"),
                ("uniformValue/scale/start", "0"),
                ("uniformValue/scale/duration", "1"),
                ("uniformValue/value/type", "constant"),
                ("uniformValue/value/value", "0.3"),
            ]),
            "uniformValue",
            1,
            SETTING,
            None,
        )?
        .expect("the entry is present");
        let (mut base, mut w) = (Vec::new(), Vec::new());
        for i in 0..8 {
            let nrm = if i % 2 == 0 { [-1.0, 0.0, 0.0] } else { [0.0, -1.0, 0.0] };
            for c in 0..3 {
                base.push(0.0);
                w.push(-nrm[c] / 4.0);
            }
        }
        Ok((
            DeviceTimedPatch::upload(&gpu, &TimedPatch {
                patch: "inlet".to_string(),
                start: 2,
                n: 8,
                n_t: 3,
                target: F1Target::RefValue,
                f,
                base,
                w,
            })?,
            F1Clock::new(&gpu, 0.0, 0.125)?,
            gpu.zeros::<Vec3>(12)?,
            F1Kernels::new(&gpu)?,
        ))
    };
    let iterate = |m: &mut Rig| -> crate::error::Result<()> {
        let (dp, clock, vecs, k) = m;
        clock.advance(&gpu, k)?;
        dp.apply_vector(&gpu, k, clock, vecs)
    };
    let state = |m: &Rig| -> crate::error::Result<crate::capture::Snapshot> {
        let (_, clock, vecs, _) = m;
        let cl = gpu.download(&clock.buf)?;
        let vs = gpu.download(vecs)?;
        Ok(vec![
            ("clock", cl.iter().map(|v| *v as Scalar).collect()),
            ("ref_value", vs.iter().flat_map(|v| [v.x, v.y, v.z]).collect()),
        ])
    };
    let report = capture_replays_bitwise(
        &gpu,
        "the ramped inlet (SPEC-LIT 120.1)",
        build,
        iterate,
        state,
    )
    .expect("the ramped inlet must replay bitwise with the time moving");
    println!("{report}");
}

/// The apply kernels against [`TimedPatch::host_values`], bitwise, over 50
/// steps of three patches - a scalar target with a sine F, a vector target
/// with a vector table F, and a vector target with a scalar
/// `quarterSineRamp` F scaling `w = -n/A`. Entries outside the patch stay
/// exactly as they were.
#[test]
fn apply_equals_host_values_bitwise() {
    let Ok(gpu) = Gpu::new(0) else { return };
    let k = F1Kernels::new(&gpu).unwrap();

    let sine = parse(&ent(&[
        ("uniformValue/type", "sine"),
        ("uniformValue/amplitude", "2"),
        ("uniformValue/frequency", "3.7"),
        ("uniformValue/start", "0.1"),
        ("uniformValue/level", "0.5"),
    ]), 1)
    .unwrap();
    let vtable = parse(&ent(&[
        ("uniformValue", "table ((0 (1 0 0)) (2 (3 4 -1)))"),
    ]), 3)
    .unwrap();
    let qsr = parse(&ent(&[
        ("uniformValue/type", "quarterSineRamp"),
        ("uniformValue/start", "0.2"),
        ("uniformValue/duration", "1.5"),
    ]), 1)
    .unwrap();

    let fixed = |n: usize| (0..n).map(|i| 0.5 + 0.1 * i as f64).collect::<Vec<_>>();
    let minus_n_over_a = |n: usize| {
        (0..n)
            .flat_map(|i| {
                let nrm = if i % 2 == 0 { [-1.0, 0.0, 0.0] } else { [0.0, -1.0, 0.0] };
                nrm.map(|v| -v / 4.0)
            })
            .collect::<Vec<_>>()
    };
    let scalar = TimedPatch {
        patch: "sine".to_string(),
        start: 1, n: 5, n_t: 1, target: F1Target::RefValue,
        f: sine, base: fixed(5), w: fixed(5),
    };
    let vector = TimedPatch {
        patch: "vtable".to_string(),
        start: 1, n: 5, n_t: 3, target: F1Target::RefValue,
        f: vtable, base: fixed(15), w: fixed(15),
    };
    let mixed = TimedPatch {
        patch: "qsr".to_string(),
        start: 1, n: 5, n_t: 3, target: F1Target::RefGrad,
        f: qsr, base: fixed(15), w: minus_n_over_a(5),
    };
    let a = DeviceTimedPatch::upload(&gpu, &scalar).unwrap();
    let b = DeviceTimedPatch::upload(&gpu, &vector).unwrap();
    let c = DeviceTimedPatch::upload(&gpu, &mixed).unwrap();

    let guard = -7.0 as Scalar;
    let mut dst_a: DevBuf<Scalar> = gpu.upload(&vec![guard; 8]).unwrap();
    let gvec = Vec3::new(guard, guard, guard);
    let mut dst_b: DevBuf<Vec3> = gpu.upload(&vec![gvec; 8]).unwrap();
    let mut dst_c: DevBuf<Vec3> = gpu.upload(&vec![gvec; 8]).unwrap();
    let mut clock = F1Clock::new(&gpu, 0.0, 0.03).unwrap();

    for _ in 0..50 {
        clock.advance(&gpu, &k).unwrap();
        a.apply_scalar(&gpu, &k, &clock, &mut dst_a).unwrap();
        b.apply_vector(&gpu, &k, &clock, &mut dst_b).unwrap();
        c.apply_vector(&gpu, &k, &clock, &mut dst_c).unwrap();

        let (t, _) = clock.read(&gpu).unwrap();
        let want_a = scalar.host_values(t);
        let want_b = vector.host_values(t);
        let want_c = mixed.host_values(t);
        let ga = gpu.download(&dst_a).unwrap();
        let gb = gpu.download(&dst_b).unwrap();
        let gc = gpu.download(&dst_c).unwrap();
        for i in 0..8 {
            let inside = (1..6).contains(&i);
            if inside {
                assert_eq!(
                    ga[i].to_bits(),
                    want_a[i - 1].to_bits(),
                    "sine: face {i} t {t}"
                );
            } else {
                assert_eq!(ga[i].to_bits(), guard.to_bits(), "sine: guard face {i}");
            }
            for cc in 0..3 {
                let flat = 3 * i + cc;
                let gbi = [gb[i].x, gb[i].y, gb[i].z][cc];
                let gci = [gc[i].x, gc[i].y, gc[i].z][cc];
                if inside {
                    assert_eq!(
                        gbi.to_bits(),
                        want_b[flat - 3].to_bits(),
                        "vtable: entry {flat} t {t}"
                    );
                    assert_eq!(
                        gci.to_bits(),
                        want_c[flat - 3].to_bits(),
                        "qsr: entry {flat} t {t}"
                    );
                } else {
                    assert_eq!(gbi.to_bits(), guard.to_bits(), "vtable: guard {flat}");
                    assert_eq!(gci.to_bits(), guard.to_bits(), "qsr: guard {flat}");
                }
            }
        }
    }
}

/// The clock's device sum is the host's sum, bitwise: a time of 1000 s
/// stepped by 1e-4 s a thousand times picks up the same accumulated
/// rounding on both sides, and lands within 1e-9 of 1000.1.
#[test]
fn clock_advances_like_the_host_sum() {
    let Ok(gpu) = Gpu::new(0) else { return };
    let k = F1Kernels::new(&gpu).unwrap();
    let mut clock = F1Clock::new(&gpu, 1000.0, 1.0e-4).unwrap();
    for _ in 0..1000 {
        clock.advance(&gpu, &k).unwrap();
    }
    let (t, dt) = clock.read(&gpu).unwrap();

    let mut want = 1000.0_f64;
    for _ in 0..1000 {
        want = want + 1.0e-4;
    }
    assert_eq!(t.to_bits(), want.to_bits(), "the device sum is the host sum");
    assert_eq!(dt.to_bits(), 1.0e-4_f64.to_bits(), "dt is untouched");
    assert!(
        (t - 1000.1).abs() < 1.0e-9,
        "t = {t} is not within 1e-9 of 1000.1"
    );
}

/// The upload refuses a rank the target does not have, by name, and the
/// apply kernels refuse the wrong field kind (SPEC-LIT 120.1's table: `n_t`
/// is 1 or 3 and `f.n_c` is `n_t` or 1).
#[test]
fn upload_refuses_mismatched_ranks() {
    let Ok(gpu) = Gpu::new(0) else { return };
    let k = F1Kernels::new(&gpu).unwrap();
    let f1 = parse(&ent(&[("uniformValue", "1.5")]), 1).unwrap();
    let f3 = parse(&ent(&[("uniformValue", "(1 -2 3)")]), 3).unwrap();

    let mk = |f: &F1, n_t: usize, base_len: usize| TimedPatch {
        patch: "p".to_string(),
        start: 0,
        n: 5,
        n_t,
        target: F1Target::RefValue,
        f: f.clone(),
        base: vec![0.0; base_len],
        w: vec![1.0; base_len],
    };

    let e = DeviceTimedPatch::upload(&gpu, &mk(&f1, 2, 5))
        .unwrap_err()
        .to_string();
    assert!(e.contains("not 1 or 3"), "{e}");

    let e = DeviceTimedPatch::upload(&gpu, &mk(&f3, 1, 5))
        .unwrap_err()
        .to_string();
    assert!(e.contains("components where the target has 1"), "{e}");

    let e = DeviceTimedPatch::upload(&gpu, &mk(&f1, 1, 4))
        .unwrap_err()
        .to_string();
    assert!(e.contains("base has 4 entries"), "{e}");

    let bad_w = TimedPatch {
        w: vec![1.0; 6],
        ..mk(&f1, 1, 5)
    };
    let e = DeviceTimedPatch::upload(&gpu, &bad_w)
        .unwrap_err()
        .to_string();
    assert!(e.contains("w has 6 entries"), "{e}");

    let three = DeviceTimedPatch::upload(&gpu, &mk(&f3, 3, 15)).unwrap();
    let one = DeviceTimedPatch::upload(&gpu, &mk(&f1, 1, 5)).unwrap();
    let clock = F1Clock::new(&gpu, 0.0, 0.1).unwrap();
    let mut scalar_dst: DevBuf<Scalar> = gpu.zeros(8).unwrap();
    let mut vec_dst: DevBuf<Vec3> = gpu.zeros(8).unwrap();

    let e = three
        .apply_scalar(&gpu, &k, &clock, &mut scalar_dst)
        .unwrap_err()
        .to_string();
    assert!(e.contains("apply_scalar on a timed patch"), "{e}");
    let e = one
        .apply_vector(&gpu, &k, &clock, &mut vec_dst)
        .unwrap_err()
        .to_string();
    assert!(e.contains("apply_vector on a timed patch"), "{e}");
}

/// A patch uploaded for another field must be refused before the launch: the
/// kernel writes `dst[start .. start + n)` and the device has no bounds
/// check, so a target shorter than `start + n` would be out-of-bounds writes
/// on the device rather than an error (SPEC-LIT 120.1's apply step).
#[test]
fn apply_refuses_a_target_too_short() {
    let Ok(gpu) = Gpu::new(0) else { return };
    let k = F1Kernels::new(&gpu).unwrap();
    let f = parse(
        &ent(&[
            ("uniformValue/type", "linearRamp"),
            ("uniformValue/start", "0"),
            ("uniformValue/duration", "1"),
        ]),
        1,
    )
    .unwrap();
    let clock = F1Clock::new(&gpu, 0.0, 0.1).unwrap();

    let scalar = DeviceTimedPatch::upload(&gpu, &TimedPatch {
        patch: "p".to_string(),
        start: 3,
        n: 4,
        n_t: 1,
        target: F1Target::RefValue,
        f: f.clone(),
        base: vec![0.0; 4],
        w: vec![1.0; 4],
    })
    .unwrap();
    let mut dst: DevBuf<Scalar> = gpu.zeros(5).unwrap();
    let e = scalar
        .apply_scalar(&gpu, &k, &clock, &mut dst)
        .unwrap_err()
        .to_string();
    assert!(e.contains("too short"), "{e}");
    assert!(e.contains("p"), "{e}");
    assert!(e.contains("3") && e.contains("4") && e.contains("5"), "{e}");

    let vector = DeviceTimedPatch::upload(&gpu, &TimedPatch {
        patch: "q".to_string(),
        start: 0,
        n: 4,
        n_t: 3,
        target: F1Target::RefValue,
        f,
        base: vec![0.0; 12],
        w: vec![1.0; 12],
    })
    .unwrap();
    let mut vdst: DevBuf<Vec3> = gpu.zeros(3).unwrap();
    let e = vector
        .apply_vector(&gpu, &k, &clock, &mut vdst)
        .unwrap_err()
        .to_string();
    assert!(e.contains("too short"), "{e}");
}

/// SPEC-LIT 120.1's gate (precision class A for the integral): a
/// `flowRateInletVelocity` patch whose rate is `0.2 * linearRamp(0.25, 0.5)`
/// steps through `t in [0, 1]` at dt = 1/64, the inflow
/// `-sum_f ref_value . Sf` is integrated by the trapezoid rule - exact for a
/// piecewise-linear rate whose kinks fall on steps - and must come out
/// `0.5 * Q0 = 0.1` to 1e-12 relative, with the rate exactly 0 at t = 0 and
/// 0.2 to 1e-12 relative after the ramp.
#[test]
fn ramped_inlet_volume_is_the_closed_form() {
    let Ok(gpu) = Gpu::new(0) else { return };
    let k = F1Kernels::new(&gpu).unwrap();

    let (mut m, points, faces) =
        crate::mesh::topology::tests::box_mesh([2, 2, 1], Vec3::new(0.5, 0.25, 1.0));
    m.compute_geometry(&points, &faces).expect("geometry");
    m.build_cell_face_maps();
    let gm = crate::mesh::GpuMesh::upload(&gpu, &m).expect("upload");

    let mut raw = crate::io::fields::RawVectorField {
        name: "U".into(),
        ..Default::default()
    };
    let mut inlet = crate::io::fields::PatchFieldSpec {
        type_name: "flowRateInletVelocity".into(),
        ..Default::default()
    };
    inlet.dicts.insert(
        "volumetricFlowRate".into(),
        "type scale; scale { type linearRamp; start 0.25; duration 0.5; } \
         value constant 0.2;"
            .into(),
    );
    for p in &m.patches {
        let spec = if p.name == "xmin" {
            inlet.clone()
        } else {
            let mut s = crate::io::fields::PatchFieldSpec {
                type_name: "fixedValue".into(),
                ..Default::default()
            };
            s.value_v = vec![Vec3::ZERO];
            s
        };
        raw.boundary.insert(p.name.clone(), spec);
    }

    let mut f = crate::field::GpuVectorField::zeros(&gpu, &gm, "U").expect("field");
    let timed = crate::field_setup::setup_vector_field_timed(
        &gpu,
        &mut f,
        &raw,
        &m,
        crate::field_setup::BcInputs::default(),
        &crate::field_setup::F1Setup {
            t_start: 0.0,
            t_end: 1.0,
            case_dir: None,
        },
    )
    .expect("setup");
    assert_eq!(timed.len(), 1, "one time-varying patch");

    // The inflow through xmin, in f64 from the downloaded ref_value and the
    // mesh's face areas.
    let q_of = |ref_value: &[Vec3]| -> f64 {
        let p0 = &m.patches[0];
        let mut acc = 0.0;
        for i in 0..p0.size {
            let u = ref_value[p0.start + i];
            let sf = m.b_sf[p0.start + i];
            acc += u.x as f64 * sf.x as f64
                + u.y as f64 * sf.y as f64
                + u.z as f64 * sf.z as f64;
        }
        -acc
    };

    let mut clock = F1Clock::new(&gpu, 0.0, 1.0 / 64.0).unwrap();
    let mut q_prev = q_of(&gpu.download(&f.ref_value).unwrap());
    assert_eq!(q_prev, 0.0, "the ramp is 0 at t = 0");
    let mut integral = 0.0f64;
    for _ in 0..64 {
        clock.advance(&gpu, &k).unwrap();
        timed.apply_vector(&gpu, &k, &clock, &mut f).unwrap();
        let q = q_of(&gpu.download(&f.ref_value).unwrap());
        integral += (1.0 / 64.0) * (q_prev + q) / 2.0;
        q_prev = q;
    }
    println!("ramped_inlet_volume: trapezoid integral = {integral}");
    assert!(
        (integral - 0.1).abs() <= crate::precision::roundoff_tol(1.0e-12) * 0.1,
        "the integral {integral} is not 0.1 to 1e-12 relative"
    );
    assert!(
        (q_prev - 0.2).abs() <= crate::precision::roundoff_tol(1.0e-12) * 0.2,
        "the final rate {} is not 0.2 to 1e-12 relative",
        q_prev
    );
}

