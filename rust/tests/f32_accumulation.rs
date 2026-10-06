// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.
// Provenance: see PROVENANCE.md. No GPL-licensed source was consulted.

//! The f64-accumulation tests of SPEC-LIT §118.1: every reduction of
//! `cuda/solver.cu` accumulates in `ofacc` (double) and its partials are
//! `ofacc`, in BOTH builds, and the one rounding happens when stage two
//! stores.
//!
//! This is an integration test, and not a `#[cfg(test)]` module of the
//! library, because the library's own test target does not compile under
//! `single` yet; an integration test links the normal (non-test) library,
//! which does. It runs in BOTH builds with the same assertions - precision
//! class A: an accumulated identity evaluated in f64, whose expected values
//! are exact, so the tolerance is the same in both builds.
//!
//! Provenance: ORIGINAL - the tests and the inputs. The inputs are this
//! tree's own construction: the generator below and the reason each vector
//! defeats a float accumulator are recorded in §118.1.
//! No GPL-licensed source was consulted.

use ofgpu::exactsum::ExactReduction;
use ofgpu::solver::{
    device_dot, device_dot2, device_max_mag, device_sum, device_sum_mag, reduce_partitions,
    SolverKernels,
};
use ofgpu::{Gpu, Scalar};

/// Terms summed. `2^20`, comfortably above the point where a float tree loses
/// unit terms and below the `2^31` the kernels' `oflabel` indexing bounds.
const N: usize = 1 << 20;

const TWO20: u64 = 1 << 20;
const TWO23: u64 = 1 << 23;
const TWO24: u64 = 1 << 24;

/// `Σ_{k < N-1} r_k`.
const SUM_REST: u64 = 4_398_030_967_741;
/// `r_{N-1}`, which closes `R = Σ r_k` to a multiple of `2^20`.
const R_LAST: u64 = 863_299;
/// `R = Σ r_k`, a multiple of `2^20`.
const R_TOTAL: u64 = 4_398_031_831_040;
/// `T = N + R·2^-23`, the exact sum of `x` - and of `sum|z|` and `(x,1)`.
const T_EXACT: f64 = 1_572_862.25;
/// `max r_k`.
const R_MAX: u64 = 8_388_591;
/// `max|z_k| = 1 + R_MAX·2^-23`, exact as a `Scalar` in both builds.
const Z_MAX: f64 = 1.999_997_973_442_077_6;
/// The exact sum of `y`: `N` unit terms beside one `±2^24` pair.
const Y_EXACT: f64 = (N - 2) as f64;

/// Every device test needs a card. Returning `None` makes the test pass
/// vacuously on a machine without one, which is the convention the rest of
/// the crate follows.
fn gpu() -> Option<Gpu> {
    Gpu::new(0).ok()
}

/// The term magnitudes: `r_k < 2^23` for `k < N-1`, and `r_{N-1}` closes the
/// sum to a multiple of `2^20` so that `T` is exactly representable.
///
/// Chosen by simulating `solver.cu`'s exact grid-stride order at `--features
/// single` until an input was found that the float tree misses and an f64
/// tree does not.
fn radii() -> Vec<u64> {
    let mut r: Vec<u64> = (0..N - 1)
        .map(|k| ((k as u64 * 2_654_435_761) & 0xFFFF_FFFF) >> 9)
        .collect();
    let rest: u64 = r.iter().sum();
    r.push((TWO20 - rest % TWO20) % TWO20);
    r
}

/// `x_k = 1 + r_k·2^-23` - a multiple of `2^-23` in `[1, 2)`, so exact as a
/// `Scalar` in BOTH builds.
fn x_field(r: &[u64]) -> Vec<Scalar> {
    r.iter()
        .map(|&rk| (1.0 + rk as f64 * 2f64.powi(-23)) as Scalar)
        .collect()
}

/// `z_k = x_k` for even `k`, `-x_k` for odd `k`: `sum|z| = T` again, with
/// cancellation a naive signed sum would drown in.
fn z_field(r: &[u64]) -> Vec<Scalar> {
    r.iter()
        .enumerate()
        .map(|(k, &rk)| {
            let x = (1.0 + rk as f64 * 2f64.powi(-23)) as Scalar;
            if k % 2 == 0 { x } else { -x }
        })
        .collect()
}

/// `y_k = 1` except `y_0 = 2^24` and `y_{N-1} = -2^24`: at `2^24` a float's
/// spacing is 2, so a float accumulator swallows every unit term it meets
/// there; the f64 tree keeps all of them.
fn y_field() -> Vec<Scalar> {
    (0..N)
        .map(|k| match k {
            0 => TWO24 as Scalar,
            k if k == N - 1 => -(TWO24 as Scalar),
            _ => 1.0,
        })
        .collect()
}

/// `AA = Σ x_k² = Σ (2^23 + r_k)²/2^46`, the numerator exact in `u128` and
/// the division by a power of two exact in f64.
fn aa_exact(r: &[u64]) -> f64 {
    let num: u128 = r
        .iter()
        .map(|&rk| {
            let t = TWO23 as u128 + rk as u128;
            t * t
        })
        .sum();
    num as f64 / (1u128 << 46) as f64
}

/// The generator's recorded constants, asserted so a wrong generator fails
/// loudly rather than silently changing the expected values (§118.1).
fn check_generator(r: &[u64]) {
    assert_eq!(r[0], 0, "r_0");
    assert_eq!(r[1], 5_184_444, "r_1");
    assert_eq!(r[2], 1_980_281, "r_2");
    assert_eq!(r[3], 7_164_726, "r_3");
    assert_eq!(*r.iter().max().unwrap(), R_MAX, "max r_k");
    let rest: u64 = r[..N - 1].iter().sum();
    assert_eq!(rest, SUM_REST, "sum of r_k for k < N-1");
    assert_eq!(r[N - 1], R_LAST, "r_(N-1)");
    let total: u64 = r.iter().sum();
    assert_eq!(total, R_TOTAL, "R = sum r_k");
    assert_eq!(total % TWO20, 0, "R must be a multiple of 2^20");

    // Every x_k is exact as a Scalar in both builds, and so is T.
    for (k, &rk) in r.iter().enumerate() {
        let xk = 1.0 + rk as f64 * 2f64.powi(-23);
        assert_eq!(xk as Scalar as f64, xk, "x_{k} must be exact as a Scalar");
    }
    assert_eq!(T_EXACT as Scalar as f64, T_EXACT, "T must be exact as a Scalar");
    assert_eq!(
        (TWO23 as f64 + R_MAX as f64) / TWO23 as f64,
        Z_MAX,
        "max|z|"
    );

    // y sums to N-2 exactly, in integers.
    let y = y_field();
    let y_sum: i64 = y.iter().map(|&v| v as i64).sum();
    assert_eq!(y_sum as f64, Y_EXACT, "the exact sum of y");

    // AA against its recorded value.
    let aa = aa_exact(r);
    assert!(
        (aa - 2_446_672.163_792_271).abs() < 1e-6,
        "AA = sum x^2: {aa:.17e}"
    );
}

/// Test 1 - `device_sum(x)`: the plain sum, exact only if the accumulation
/// happened in f64 and the single rounding of §118.1 happened at the store.
#[test]
fn f32_sum_of_2e20_terms_is_f64_accurate() {
    let Some(gpu) = gpu() else {
        eprintln!("no GPU");
        return;
    };
    let r = radii();
    check_generator(&r);
    let k = SolverKernels::new(&gpu).expect("solver kernels");
    let x = x_field(&r);
    let dev = gpu.upload(&x).expect("upload x");
    let mut out = gpu.zeros(1).expect("out");
    let mut partials = gpu.zeros(reduce_partitions(N)).expect("partials");
    device_sum(&gpu, &k, &mut out, &dev, &mut partials, N).expect("device_sum");
    let got = gpu.download(&out).expect("download")[0] as f64;
    let rel = ((got - T_EXACT) / T_EXACT).abs();
    assert!(
        rel <= 4.0 * f64::EPSILON,
        "device_sum(x): device {got:.17e}, expected {T_EXACT:.17e}, relative error {rel:.3e}"
    );
    assert_eq!(got, T_EXACT, "device_sum(x): device {got:.17e}, expected {T_EXACT:.17e}");
}

/// Test 2 - `device_sum(y)`: unit terms summed beside `±2^24`, which a float
/// accumulator loses to spacing and an f64 accumulator keeps.
#[test]
fn f32_sum_keeps_unit_terms_beside_2e24() {
    let Some(gpu) = gpu() else {
        eprintln!("no GPU");
        return;
    };
    let k = SolverKernels::new(&gpu).expect("solver kernels");
    let y = y_field();
    let dev = gpu.upload(&y).expect("upload y");
    let mut out = gpu.zeros(1).expect("out");
    let mut partials = gpu.zeros(reduce_partitions(N)).expect("partials");
    device_sum(&gpu, &k, &mut out, &dev, &mut partials, N).expect("device_sum");
    let got = gpu.download(&out).expect("download")[0] as f64;
    let d = got - Y_EXACT;
    assert_eq!(
        got, Y_EXACT,
        "device_sum(y): device {got:.1}, expected {Y_EXACT:.1}, difference {d:.1e}"
    );
}

/// Test 3 - `device_sum_mag(z)` and `device_max_mag(z)`: the magnitude sum is
/// `T` again, and the maximum - exact whatever the order - is `Z_MAX`.
#[test]
fn f32_sum_mag_and_max_mag_of_alternating_terms() {
    let Some(gpu) = gpu() else {
        eprintln!("no GPU");
        return;
    };
    let k = SolverKernels::new(&gpu).expect("solver kernels");
    let z = z_field(&radii());
    let dev = gpu.upload(&z).expect("upload z");
    let mut out = gpu.zeros(1).expect("out");
    let mut partials = gpu.zeros(reduce_partitions(N)).expect("partials");
    device_sum_mag(&gpu, &k, &mut out, &dev, &mut partials, N).expect("device_sum_mag");
    let got_sum = gpu.download(&out).expect("download")[0] as f64;
    assert_eq!(
        got_sum, T_EXACT,
        "device_sum_mag(z): device {got_sum:.17e}, expected {T_EXACT:.17e}"
    );

    let mut out2 = gpu.zeros(1).expect("out2");
    let mut partials2 = gpu.zeros(reduce_partitions(N)).expect("partials2");
    device_max_mag(&gpu, &k, &mut out2, &dev, &mut partials2, N).expect("device_max_mag");
    let got_max = gpu.download(&out2).expect("download")[0] as f64;
    assert_eq!(
        got_max, Z_MAX,
        "device_max_mag(z): device {got_max:.17e}, expected {Z_MAX:.17e}"
    );
}

/// Test 4 - `device_dot(x, 1)` and the fused `device_dot2(x, 1)`: the dot is
/// `T`, and the paired `(x,x)` lands within one output rounding of the exact
/// `AA` plus a bound on the f64 tree.
#[test]
fn f32_dot_and_dot2_accumulate_in_f64() {
    let Some(gpu) = gpu() else {
        eprintln!("no GPU");
        return;
    };
    let k = SolverKernels::new(&gpu).expect("solver kernels");
    let r = radii();
    let x = x_field(&r);
    let ones: Vec<Scalar> = vec![1.0; N];
    let dx = gpu.upload(&x).expect("upload x");
    let d_ones = gpu.upload(&ones).expect("upload ones");
    let mut out = gpu.zeros(1).expect("out");
    let mut partials = gpu.zeros(reduce_partitions(N)).expect("partials");
    device_dot(&gpu, &k, &mut out, &dx, &d_ones, &mut partials, N).expect("device_dot");
    let got_dot = gpu.download(&out).expect("download")[0] as f64;
    assert_eq!(
        got_dot, T_EXACT,
        "device_dot(x, 1): device {got_dot:.17e}, expected {T_EXACT:.17e}"
    );

    let aa = aa_exact(&r);
    let tol = (Scalar::EPSILON as f64 / 2.0 + 64.0 * 2f64.powi(-53)) * aa;
    let mut ab = gpu.zeros(1).expect("ab");
    let mut aa_buf = gpu.zeros(1).expect("aa");
    let mut p_ab = gpu.zeros(reduce_partitions(N)).expect("partials ab");
    let mut p_aa = gpu.zeros(reduce_partitions(N)).expect("partials aa");
    device_dot2(
        &gpu,
        &k,
        &mut ab,
        &mut aa_buf,
        &dx,
        &d_ones,
        &mut p_ab,
        &mut p_aa,
        N,
    )
    .expect("device_dot2");
    let got_ab = gpu.download(&ab).expect("download")[0] as f64;
    assert_eq!(
        got_ab, T_EXACT,
        "device_dot2 ab: device {got_ab:.17e}, expected {T_EXACT:.17e}"
    );
    let got_aa = gpu.download(&aa_buf).expect("download")[0] as f64;
    let d = (got_aa - aa).abs();
    assert!(
        d <= tol,
        "device_dot2 aa: device {got_aa:.17e}, expected {aa:.17e}, \
         difference {d:.3e}, tolerance {tol:.3e}"
    );
}

/// Test 5 - the `exactsum` gather now moves each part's stage-two result
/// through `Acc`: `gathered_sum` matches the host construction over the
/// rounded halves and lands on `T` in both builds, the exact limb sum is `T`,
/// and `max_mag` is `Z_MAX`.
#[test]
fn f32_exactsum_gathers_through_acc() {
    let Some(gpu) = gpu() else {
        eprintln!("no GPU");
        return;
    };
    let k = SolverKernels::new(&gpu).expect("solver kernels");
    let r = radii();
    let x = x_field(&r);
    let z = z_field(&r);

    // Each half its own device buffer of exactly N/2 values.
    let xs = vec![
        gpu.upload(&x[..N / 2]).expect("upload x[0..N/2)"),
        gpu.upload(&x[N / 2..]).expect("upload x[N/2..N)"),
    ];
    let zs = vec![
        gpu.upload(&z[..N / 2]).expect("upload z[0..N/2)"),
        gpu.upload(&z[N / 2..]).expect("upload z[N/2..N)"),
    ];

    // The half sums, exact on the host: every term is (2^23 + r_k)/2^23 and
    // the integer numerator of each half fits f64 exactly.
    let half = |rr: &[u64]| -> f64 {
        let num: u128 = rr.iter().map(|&rk| TWO23 as u128 + rk as u128).sum();
        num as f64 / TWO23 as f64
    };
    let s1 = half(&r[..N / 2]);
    let s2 = half(&r[N / 2..]);
    assert_eq!(s1, 786_432.067_382_812_5, "S1");
    assert_eq!(s2, 786_430.182_617_187_5, "S2");

    let mut er = ExactReduction::new(&gpu, &[N / 2, N / 2]).expect("ExactReduction");

    er.gathered_sum(&gpu, &k, &xs).expect("gathered_sum");
    let got_gather = er.value(&gpu).expect("value") as f64;
    let want_gather = ((s1 as Scalar) as f64 + (s2 as Scalar) as f64) as Scalar as f64;
    assert_eq!(
        got_gather, want_gather,
        "gathered_sum: device {got_gather:.17e}, expected {want_gather:.17e}"
    );
    assert_eq!(
        want_gather, T_EXACT,
        "gathered_sum must land on T exactly: {want_gather:.17e}"
    );

    er.sum(&gpu, &k, &xs).expect("exact sum");
    let got_exact = er.value(&gpu).expect("value") as f64;
    assert_eq!(
        got_exact, T_EXACT,
        "exact sum: device {got_exact:.17e}, expected {T_EXACT:.17e}"
    );

    er.max_mag(&gpu, &k, &zs).expect("max_mag");
    let got_max = er.value(&gpu).expect("value") as f64;
    assert_eq!(
        got_max, Z_MAX,
        "max_mag over the halves: device {got_max:.17e}, expected {Z_MAX:.17e}"
    );
}
