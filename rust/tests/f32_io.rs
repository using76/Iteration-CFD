// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.
// Provenance: see PROVENANCE.md. No GPL-licensed source was consulted.

//! SPEC-LIT §118.6 — field, phi and restart I/O at f32: `phi` is written at
//! THIS BUILD's round-trip digit count (`ROUND_TRIP_DIGITS`, 17 for f64 and
//! 9 under `single`), the `%g` formatter never prints more than that under
//! `single`, the reader parses every number as `f64` and rounds ONCE to
//! `Scalar`, and the `.mcr` restart's `Scalar` <-> f64 seam widens exactly
//! and narrows bit for bit, so a single-build checkpoint restarts bitwise.
//!
//! An integration test because the library's own test target does not
//! compile under `single`; an integration test links the normal (non-test)
//! library, which does. It runs in BOTH builds with the same assertions —
//! precision class D of docs/17 §5.1 rule 4: discrete / bitwise, the SAME
//! assertion in both builds. No GPU needed.
//!
//! Provenance: ORIGINAL — the values are this tree's own generator
//! (SplitMix64's finaliser, Steele, Lea & Flood 2014, public domain, already
//! cited at §66.14, used ONLY as a test-data generator) and this tree's own
//! edge set; no GPL-licensed source was consulted.

use std::collections::BTreeMap;
use std::path::PathBuf;

use ofgpu::io::fields::{
    read_scalar_field, write_scalar_field, write_scalar_field_prec, write_surface_scalar_field,
    PatchFieldSpec, RawScalarField, PHI_PRECISION, ROUND_TRIP_DIGITS,
};
use ofgpu::restart::{
    narrow_scalars, narrow_vectors, read_restart, widen_scalars, widen_vectors, write_restart,
    FieldKind, RestartData, RestartField,
};
use ofgpu::{Scalar, Vec3};

// ==========================================================================
//  Test-data generation
// ==========================================================================

/// SplitMix64's finaliser (Steele, Lea & Flood 2014, public domain, already
/// cited at §66.14) — a test-data generator and nothing else.
struct SplitMix64(u64);

impl SplitMix64 {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
}

/// The build's `Scalar` from a raw bit pattern, so the sample covers every
/// exponent (subnormals included) and both signs.
#[cfg(feature = "single")]
fn scalar_from_bits(u: u64) -> Scalar {
    f32::from_bits(u as u32)
}

#[cfg(not(feature = "single"))]
fn scalar_from_bits(u: u64) -> Scalar {
    f64::from_bits(u)
}

/// `n` finite `Scalar`s drawn as random bit patterns - every exponent,
/// subnormals included, is reached, and the non-finite patterns (about one in
/// 128) are dropped.
fn sample(n: usize, seed: u64) -> Vec<Scalar> {
    let mut g = SplitMix64(seed);
    let mut out = Vec::with_capacity(n);
    while out.len() < n {
        let v = scalar_from_bits(g.next());
        if v.is_finite() {
            out.push(v);
        }
    }
    out
}

/// The values the round trip is most likely to get wrong at: zeros (both
/// signs - the sign bit is data), one, the decimal that looks exact but
/// isn't, one third, the largest and most negative finite values, the
/// smallest normal, the smallest SUBNORMAL, the spacing at one, and, for
/// every power-of-two exponent in range, that power, its predecessor and its
/// successor - all computed in this build's `Scalar`.
fn edge_values() -> Vec<Scalar> {
    let mut v: Vec<Scalar> = vec![
        0.0,
        -0.0,
        1.0,
        -1.0,
        0.1,
        1.0 / 3.0,
        Scalar::MAX,
        -Scalar::MAX,
        Scalar::MIN_POSITIVE,
        Scalar::MIN_POSITIVE * Scalar::EPSILON,
        Scalar::EPSILON,
    ];
    for k in -30..=30i32 {
        let two: Scalar = 2.0;
        let two_k: Scalar = two.powi(k);
        v.push(two_k);
        v.push(two_k * (1.0 - Scalar::EPSILON / 2.0));
        v.push(two_k * (1.0 + Scalar::EPSILON));
    }
    v
}

// ==========================================================================
//  Bitwise comparison
// ==========================================================================

/// This build's `Scalar` bit pattern. Widening to f64 first is exact and
/// injective in both builds, so the widened f64's bits identify an f32 bit
/// for bit too.
#[cfg(feature = "single")]
fn bit_pattern(v: Scalar) -> u32 {
    v.to_bits()
}

#[cfg(not(feature = "single"))]
fn bit_pattern(v: Scalar) -> u64 {
    v.to_bits()
}

fn assert_one_bitwise(what: &str, i: usize, w: Scalar, g: Scalar) {
    if bit_pattern(w) != bit_pattern(g) {
        panic!(
            "{what}: value {i} is not bit for bit: wrote {:e} (bits 0x{:0w$x}), \
             read back {:e} (bits 0x{:0w$x})",
            w,
            bit_pattern(w),
            g,
            bit_pattern(g),
            w = std::mem::size_of::<Scalar>() * 2,
        );
    }
}

fn assert_bitwise_equal(what: &str, want: &[Scalar], got: &[Scalar]) {
    assert_eq!(
        want.len(),
        got.len(),
        "{what}: wrote {} values, read back {}",
        want.len(),
        got.len()
    );
    for (i, (&w, &g)) in want.iter().zip(got.iter()).enumerate() {
        assert_one_bitwise(what, i, w, g);
    }
}

fn assert_vectors_bitwise_equal(what: &str, want: &[Vec3], got: &[Vec3]) {
    assert_eq!(
        want.len(),
        got.len(),
        "{what}: wrote {} vectors, read back {}",
        want.len(),
        got.len()
    );
    for (i, (w, g)) in want.iter().zip(got.iter()).enumerate() {
        for (axis, (wc, gc)) in [(w.x, g.x), (w.y, g.y), (w.z, g.z)].into_iter().enumerate() {
            assert_one_bitwise(&format!("{what}[{i}] axis {axis}"), i, wc, gc);
        }
    }
}

/// Significant decimal digits of one numeric token: everything from the
/// first exponent letter off, then sign and point characters, then leading
/// zeros - so `0.100000001` -> 9, `3.40282347e+38` -> 9, `-0` -> 0, `2.0` -> 2.
fn sig_digits(token: &str) -> usize {
    let mantissa = match token.find(|c| c == 'e' || c == 'E') {
        Some(i) => &token[..i],
        None => token,
    };
    let digits: String = mantissa
        .chars()
        .filter(|c| !matches!(c, '-' | '+' | '.'))
        .collect();
    digits.trim_start_matches('0').len()
}

/// A private directory per test, named with the process id and the test
/// name, so two tests (or two runs) never share a scratch file.
fn scratch_dir(test: &str) -> PathBuf {
    let mut d = std::env::temp_dir();
    d.push(format!("ofgpu_f32_io_{}_{test}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).expect("scratch dir");
    d
}

fn raw_scalar_field(name: &str, internal: Vec<Scalar>) -> RawScalarField {
    RawScalarField {
        name: name.to_string(),
        dimensions: "[0 3 -1 0 0 0 0]".to_string(),
        internal,
        boundary: BTreeMap::new(),
        boundary_patterns: Vec::new(),
    }
}

// ==========================================================================
//  Tests
// ==========================================================================

/// Test 1 - a `phi` of edges plus 65 536 random finite values round-trips
/// through `write_surface_scalar_field` / `read_scalar_field` BIT FOR BIT,
/// and every numeric token the file carries is at the build's round-trip
/// digit count - no token longer, and (the random sample guarantees) one of
/// exactly that length.
#[test]
fn f32_phi_round_trip_is_bitwise() {
    let dir = scratch_dir("phi_round_trip");
    let path = dir.join("phi");

    let internal: Vec<Scalar> = edge_values()
        .into_iter()
        .chain(sample(65536, 0x5EED_0F32_0009_0001))
        .collect();
    let inlet_value = sample(4096, 0x5EED_0F32_0009_0002);
    let walls_value = edge_values();

    let mut raw = raw_scalar_field("phi", internal.clone());
    raw.boundary.insert(
        "inlet".to_string(),
        PatchFieldSpec {
            type_name: "calculated".to_string(),
            value: inlet_value.clone(),
            ..Default::default()
        },
    );
    raw.boundary.insert(
        "walls".to_string(),
        PatchFieldSpec {
            type_name: "calculated".to_string(),
            value: walls_value.clone(),
            ..Default::default()
        },
    );
    raw.boundary.insert(
        "frontAndBack".to_string(),
        PatchFieldSpec {
            type_name: "empty".to_string(),
            ..Default::default()
        },
    );

    write_surface_scalar_field(&path, &raw, "0").expect("write phi");

    let back = read_scalar_field(&path, internal.len()).expect("read phi back");
    assert_bitwise_equal("phi internalField", &internal, &back.internal);

    let inlet = back.boundary.get("inlet").expect("the inlet patch");
    assert_eq!(inlet.type_name, "calculated");
    assert_bitwise_equal("phi inlet value", &inlet_value, &inlet.value);

    let walls = back.boundary.get("walls").expect("the walls patch");
    assert_eq!(walls.type_name, "calculated");
    assert_bitwise_equal("phi walls value", &walls_value, &walls.value);

    let fab = back
        .boundary
        .get("frontAndBack")
        .expect("the frontAndBack patch");
    assert_eq!(fab.type_name, "empty");
    assert!(fab.value.is_empty(), "an empty patch carries no value");

    // Every whitespace/`(`/`)`/`;`-separated token that parses as a number
    // and contains a digit is at ROUND_TRIP_DIGITS significant digits.
    let text = std::fs::read_to_string(&path).expect("read phi text");
    let mut max_sig = 0usize;
    let mut numeric_tokens = 0usize;
    for tok in text.split(|c: char| c.is_whitespace() || c == '(' || c == ')' || c == ';') {
        if tok.is_empty() || !tok.chars().any(|c| c.is_ascii_digit()) {
            continue;
        }
        if tok.parse::<f64>().is_ok() {
            numeric_tokens += 1;
            max_sig = max_sig.max(sig_digits(tok));
        }
    }
    assert_eq!(
        max_sig, ROUND_TRIP_DIGITS,
        "the file's longest numeric token carries {max_sig} significant digits, \
         expected exactly the build's ROUND_TRIP_DIGITS = {ROUND_TRIP_DIGITS}"
    );

    let checked = internal.len() + inlet_value.len() + walls_value.len();
    let bytes = std::fs::metadata(&path).expect("phi file").len();
    println!(
        "phi round trip ({build}): {checked} values checked ({numeric_tokens} numeric tokens \
         in the file), file is {bytes} bytes",
        build = if cfg!(feature = "single") { "single" } else { "f64" },
    );

    let _ = std::fs::remove_dir_all(&dir);
}

/// Test 2 - the round-trip digit count is the build's own (9 under `single`,
/// 17 otherwise), `phi` uses it, and the `%g` formatter answers a caller
/// asking for seventeen with the build's own count under `single` (the f64
/// build has no clamp and keeps its seventeen digits - these assertions pin
/// the f64 bytes).
#[test]
fn phi_is_written_at_the_round_trip_digit_count_of_this_build() {
    #[cfg(feature = "single")]
    {
        assert_eq!(ROUND_TRIP_DIGITS, 9, "the f32 build round-trips at 9 digits");
    }
    #[cfg(not(feature = "single"))]
    {
        assert_eq!(ROUND_TRIP_DIGITS, 17, "the f64 build round-trips at 17 digits");
    }
    assert_eq!(
        PHI_PRECISION, ROUND_TRIP_DIGITS,
        "PHI_PRECISION must be this build's ROUND_TRIP_DIGITS"
    );

    let dir = scratch_dir("digit_count");

    // `phi` at PHI_PRECISION: the three values every digit count is judged
    // against - the not-quite-decimal, one third, and the largest finite.
    let path = dir.join("phi_three");
    let phi = raw_scalar_field(
        "phi",
        vec![0.1, 1.0 / 3.0, Scalar::MAX],
    );
    write_surface_scalar_field(&path, &phi, "0").expect("write phi");

    // An all-equal field collapses to `uniform x;`.
    let path_uniform = dir.join("phi_uniform");
    let phi_uniform = raw_scalar_field("phi", vec![0.1]);
    write_surface_scalar_field(&path_uniform, &phi_uniform, "0").expect("write uniform phi");

    // A `volScalarField` at OpenFOAM's default `writePrecision` of 6 loses
    // digits in BOTH builds - by design.
    let path_vol6 = dir.join("p_precision_6");
    let p = raw_scalar_field("p", vec![0.1, 1.0 / 3.0]);
    write_scalar_field(&path_vol6, &p, "0").expect("write p at 6");

    // The same field at an explicit 17: clamped to nine under `single`,
    // honoured in the f64 build.
    let path_vol17 = dir.join("p_precision_17");
    write_scalar_field_prec(&path_vol17, &p, "0", 17, true).expect("write p at 17");

    let read = |p: &PathBuf| -> String { std::fs::read_to_string(p).expect("read back") };

    #[cfg(feature = "single")]
    {
        let text = read(&path);
        for line in ["0.100000001", "0.333333343", "3.40282347e+38"] {
            assert!(
                text.lines().any(|l| l == line),
                "expected a whole line `{line}` under single, got:\n{text}"
            );
        }
        assert!(
            read(&path_uniform).contains("uniform 0.100000001;"),
            "expected `uniform 0.100000001;` under single, got:\n{}",
            read(&path_uniform)
        );
        let text17 = read(&path_vol17);
        for line in ["0.100000001", "0.333333343"] {
            assert!(
                text17.lines().any(|l| l == line),
                "expected a whole line `{line}` at an asked-for 17 under single, got:\n{text17}"
            );
        }
    }
    #[cfg(not(feature = "single"))]
    {
        let text = read(&path);
        for line in [
            "0.10000000000000001",
            "0.33333333333333331",
            "1.7976931348623157e+308",
        ] {
            assert!(
                text.lines().any(|l| l == line),
                "expected a whole line `{line}` in the f64 build, got:\n{text}"
            );
        }
        assert!(
            read(&path_uniform).contains("uniform 0.10000000000000001;"),
            "expected `uniform 0.10000000000000001;` in the f64 build, got:\n{}",
            read(&path_uniform)
        );
        let text17 = read(&path_vol17);
        for line in ["0.10000000000000001", "0.33333333333333331"] {
            assert!(
                text17.lines().any(|l| l == line),
                "expected a whole line `{line}` at an asked-for 17 in the f64 build, got:\n{text17}"
            );
        }
    }

    // Precision 6 is below the clamp in both builds.
    let text6 = read(&path_vol6);
    for line in ["0.1", "0.333333"] {
        assert!(
            text6.lines().any(|l| l == line),
            "expected a whole line `{line}` at writePrecision 6 in BOTH builds, got:\n{text6}"
        );
    }

    let _ = std::fs::remove_dir_all(&dir);
}

/// Test 3 - the reader's contract, pinned directly: every number in the file
/// is parsed as `f64` and rounded ONCE to `Scalar`, so a value this writer
/// wrote at `ROUND_TRIP_DIGITS` comes back bit for bit. The file is written
/// by hand, with no writer in the loop.
#[test]
fn the_reader_parses_f64_and_rounds_once() {
    let dir = scratch_dir("reader_f64");
    let path = dir.join("phi");

    let text = "FoamFile
{
    version     2.0;
    format      ascii;
    class       surfaceScalarField;
    object      phi;
}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

dimensions      [0 3 -1 0 0 0 0];

internalField   nonuniform List<scalar>
5
(
0.10000000000000001
0.33333333333333331
1e-40
3.4028235e+38
-0
)
;

boundaryField
{
}

// ************************************************************************* //
";
    std::fs::write(&path, text).expect("write the hand-made phi");

    let back = read_scalar_field(&path, 5).expect("read the hand-made phi");

    // Each expected value is the f64 literal rounded ONCE to this build's
    // `Scalar` - the identity in the f64 build, one round-to-nearest under
    // `single`.
    let expected: [Scalar; 5] = [
        0.10000000000000001 as Scalar,
        0.33333333333333331 as Scalar,
        1e-40 as Scalar,
        3.4028235e38 as Scalar,
        -0.0 as Scalar,
    ];
    assert_bitwise_equal("hand-made phi", &expected, &back.internal);

    for (i, v) in back.internal.iter().enumerate() {
        assert!(v.is_finite(), "value {i} ({v:e}) must be finite");
    }
    assert!(
        back.internal[4].is_sign_negative(),
        "the last value must keep its sign bit (it is a negative zero)"
    );

    #[cfg(feature = "single")]
    {
        assert_eq!(
            back.internal[3], Scalar::MAX,
            "3.4028235e+38 must round to f32::MAX under single"
        );
        assert!(
            !back.internal[2].is_normal() && back.internal[2] != 0.0,
            "1e-40 must be a subnormal f32 under single"
        );
    }

    let _ = std::fs::remove_dir_all(&dir);
}

/// Test 4 - the restart seam, through the production functions: header
/// scalars, a cell-scalar field, a cell-vector field and `phi` (surface
/// scalar) written with `widen_*`, read back and narrowed with `narrow_*`,
/// every value bit for bit - the identity in the f64 build, one exact
/// widening and one round-to-nearest under `single`.
#[test]
fn f32_restart_round_trip_is_bitwise() {
    let dir = scratch_dir("restart");
    let path = dir.join("restart.mcr");
    const HASH: u64 = 0x0F32_0009_0000_0001;

    // Header scalars, held in `Scalar` as the drivers hold them.
    let t: Scalar = 1.0 / 3.0;
    let p0: Scalar = 101_325.37;
    let dp0dt: Scalar = -1.0e-7;

    let t_internal = sample(4096, 0x5EED_0F32_0009_0003);
    let t_boundary = sample(1000, 0x5EED_0F32_0009_0004);

    let triples = |src: &[Scalar]| -> Vec<Vec3> {
        src.chunks_exact(3)
            .map(|c| Vec3::new(c[0], c[1], c[2]))
            .collect()
    };
    let u_internal = triples(&sample(3 * 4096, 0x5EED_0F32_0009_0005));
    let u_boundary = triples(&sample(3 * 1000, 0x5EED_0F32_0009_0006));

    let edges = edge_values();
    let mut phi_internal = edges.clone();
    phi_internal.extend(sample(11000 - edges.len(), 0x5EED_0F32_0009_0007));
    let phi_boundary = sample(1000, 0x5EED_0F32_0009_0008);

    let data = RestartData {
        mesh_hash: HASH,
        time: f64::from(t),
        p0: f64::from(p0),
        dp0dt: f64::from(dp0dt),
        n_cells: 4096,
        n_internal: 11000,
        n_boundary: 1000,
        fields: vec![
            RestartField {
                name: "T".to_string(),
                kind: FieldKind::CellScalar,
                internal: widen_scalars(&t_internal),
                boundary: widen_scalars(&t_boundary),
            },
            RestartField {
                name: "U".to_string(),
                kind: FieldKind::CellVector,
                internal: widen_vectors(&u_internal),
                boundary: widen_vectors(&u_boundary),
            },
            RestartField {
                name: "phi".to_string(),
                kind: FieldKind::SurfaceScalar,
                internal: widen_scalars(&phi_internal),
                boundary: widen_scalars(&phi_boundary),
            },
        ],
    };

    write_restart(&path, &data).expect("write restart");
    let rd = read_restart(&path, HASH).expect("read restart");

    // The header scalars come back as the `Scalar`s they were written from.
    assert_bitwise_equal("restart time", &[t], &[rd.time as Scalar]);
    assert_bitwise_equal("restart p0", &[p0], &[rd.p0 as Scalar]);
    assert_bitwise_equal("restart dp0dt", &[dp0dt], &[rd.dp0dt as Scalar]);

    let field = |name: &str| -> &RestartField {
        rd.fields
            .iter()
            .find(|f| f.name == name)
            .unwrap_or_else(|| panic!("restart has no field {name}"))
    };

    let t_back = field("T");
    assert_eq!(t_back.kind, FieldKind::CellScalar);
    assert_bitwise_equal("T internal", &t_internal, &narrow_scalars(&t_back.internal));
    assert_bitwise_equal("T boundary", &t_boundary, &narrow_scalars(&t_back.boundary));

    let u_back = field("U");
    assert_eq!(u_back.kind, FieldKind::CellVector);
    assert_vectors_bitwise_equal("U internal", &u_internal, &narrow_vectors(&u_back.internal));
    assert_vectors_bitwise_equal("U boundary", &u_boundary, &narrow_vectors(&u_back.boundary));

    let phi_back = field("phi");
    assert_eq!(phi_back.kind, FieldKind::SurfaceScalar);
    assert_bitwise_equal(
        "phi internal",
        &phi_internal,
        &narrow_scalars(&phi_back.internal),
    );
    assert_bitwise_equal(
        "phi boundary",
        &phi_boundary,
        &narrow_scalars(&phi_back.boundary),
    );

    // And the seam is the identity on a large random sample.
    let x = sample(65536, 0x5EED_0F32_0009_0009);
    assert_bitwise_equal("widen/narrow identity", &x, &narrow_scalars(&widen_scalars(&x)));

    let _ = std::fs::remove_dir_all(&dir);
}
