// meteor-cfd - Copyright (c) 2026 주식회사 메테오시뮬레이션 (Meteo Simulation Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.

//! The gate SPEC-LIT §72 exists to pass, and the measurement that says why the
//! cheap construction is not enough.
//!
//! Provenance: ORIGINAL - the tests and the adversarial data. No external
//! source (`PROVENANCE.md`, *GPU plumbing and tooling - original*).
//! No GPL-licensed source was consulted.

use super::*;
use crate::decompose::tests::boxes;
use crate::decompose::{partition, Decomposition, PartitionMethod};
use crate::mesh::HostMesh;
use crate::Label;

/// Every device test needs a card. Returning `None` makes the test pass
/// vacuously on a machine without one, which is the convention the rest of the
/// crate follows.
fn gpu() -> Option<Gpu> {
    Gpu::new(0).ok()
}

/// The message of a refusal. Written out because `unwrap_err` wants the OK
/// type to be `Debug`, and an accumulator holding device buffers is not.
fn err<T>(r: Result<T>) -> String {
    match r {
        Ok(_) => panic!("this was supposed to be refused"),
        Err(e) => e.to_string(),
    }
}

// ==========================================================================
//  Data that makes the difference visible
// ==========================================================================

/// A term with a full-width mantissa, a hundred-binade exponent range and
/// mixed signs.
///
/// Every part of that is load-bearing, and the first version of this function
/// had none of it: terms of the form `(1 + k/8) 2^e` with `|e| <= 20` are all
/// multiples of `2^-23`, so every partial sum a partition can form is
/// *exactly* representable and the gathered-partial construction reproduced
/// the whole mesh's answer in all nine partitions. It passed a gate it does
/// not deserve to pass. A 52-bit mantissa and a hundred binades of spread put
/// a rounding into essentially every addition, which is the regime a real
/// residual lives in.
///
/// The mixing is one round of a 64-bit multiply-xorshift on the index - a pure
/// function, so the data is the same on every machine and every run.
fn nasty(i: usize) -> Scalar {
    let mut x = (i as u64)
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    x ^= x >> 33;
    x = x.wrapping_mul(0xff51_afd7_ed55_8ccd);
    x ^= x >> 33;

    let mant = 1.0 as Scalar + (x >> 12) as Scalar * ldexp(1.0, -52);
    let e = ((x >> 3) % 101) as i32 - 50;
    let s = if x & 1 == 0 { 1.0 } else { -1.0 as Scalar };
    s * ldexp(mant, e)
}

/// A benign field of the global cell id, for the mesh-level gate.
fn field(c: usize) -> Scalar {
    0.25 + 0.0625 * ((c * 37) % 23) as Scalar
}

/// Cut `[0, n)` into `p` contiguous runs.
fn linear_slices(n: usize, p: usize) -> Vec<Vec<Scalar>> {
    let mut out = Vec::new();
    for r in 0..p {
        let a = (r * n) / p;
        let b = ((r + 1) * n) / p;
        out.push((a..b).map(nasty).collect());
    }
    out
}

/// Cut `[0, n)` round robin - the worst partition there is, and the one whose
/// per-part sums have the least in common with the whole.
fn round_robin_slices(n: usize, p: usize) -> Vec<Vec<Scalar>> {
    let mut out = vec![Vec::new(); p];
    for i in 0..n {
        out[i % p].push(nasty(i));
    }
    out
}

/// A deterministic shuffle: cell `i` to part `(i * 7 + i / 3) % p`. Not a
/// contiguous cut and not round robin, so no part is a stride of the whole.
fn shuffled_slices(n: usize, p: usize) -> Vec<Vec<Scalar>> {
    let mut out = vec![Vec::new(); p];
    for i in 0..n {
        out[(i * 7 + i / 3) % p].push(nasty(i));
    }
    out
}

/// Upload one buffer per part, each `pad` values longer than its own data so
/// that the reduction is exercised on a buffer that is longer than the part
/// owns - which is what every field in a decomposed run is.
fn upload_parts(gpu: &Gpu, parts: &[Vec<Scalar>], pad: usize) -> (Vec<DevBuf<Scalar>>, Vec<usize>) {
    let mut bufs = Vec::new();
    let mut ns = Vec::new();
    for v in parts {
        let mut padded = v.clone();
        // Poison the padding. It stands for the halo, which no reduction may
        // ever see: a ghost cell is owned by another part and summing it would
        // count it twice.
        padded.extend(std::iter::repeat_n(1.0e30 as Scalar, pad));
        ns.push(v.len());
        bufs.push(gpu.upload(&padded).expect("upload"));
    }
    (bufs, ns)
}

fn exact_sum_of(gpu: &Gpu, sol: &SolverKernels, parts: &[Vec<Scalar>], pad: usize) -> (Scalar, Vec<i64>) {
    let (bufs, ns) = upload_parts(gpu, parts, pad);
    let mut red = ExactReduction::new(gpu, &ns).expect("new");
    red.sum(gpu, sol, &bufs).expect("sum");
    gpu.sync().expect("sync");
    (red.value(gpu).expect("value"), red.limbs(gpu).expect("limbs"))
}

/// `ExactReduction::non_finite_terms` on a field with one bad value, so the
/// accessor is exercised and not merely present.
fn red_non_finite(gpu: &Gpu, sol: &SolverKernels, bad: Scalar) -> i64 {
    let mut terms: Vec<Scalar> = (0..500).map(nasty).collect();
    terms[137] = bad;
    let (bufs, ns) = upload_parts(gpu, &[terms], 0);
    let mut red = ExactReduction::new(gpu, &ns).expect("new");
    red.sum(gpu, sol, &bufs).expect("sum");
    gpu.sync().expect("sync");
    red.non_finite_terms(gpu).expect("count")
}

fn gathered_sum_of(gpu: &Gpu, sol: &SolverKernels, parts: &[Vec<Scalar>]) -> Scalar {
    let (bufs, ns) = upload_parts(gpu, parts, 0);
    let mut red = ExactReduction::new(gpu, &ns).expect("new");
    red.gathered_sum(gpu, sol, &bufs).expect("gathered");
    gpu.sync().expect("sync");
    red.value(gpu).expect("value")
}

// ==========================================================================
//  1. The layout, and the host twin
// ==========================================================================

/// `W` and `K` are `#define`s on the device and `pub const`s on the host, and
/// nothing but this test makes them the same numbers.
///
/// One term equal to 1.0 makes both readable: the anchor is `M = 1`, so the
/// top limb holds `trunc(1.0 * 2^(W-1)) = 2^(W-1)` and every other limb is
/// zero. Read `W` off the top limb, read `K` off the length.
#[test]
fn the_limb_layout_matches_the_device() {
    let Some(gpu) = gpu() else { return };
    let sol = SolverKernels::new(&gpu).expect("kernels");

    let (v, limbs) = exact_sum_of(&gpu, &sol, &[vec![1.0 as Scalar]], 0);

    assert_eq!(limbs.len(), EXACT_WORDS, "the device writes a different word count");
    assert_eq!(
        limbs[0],
        1i64 << (EXACT_W - 1),
        "the top limb of a single 1.0 pins the limb width; the device is not \
         using W = {EXACT_W}"
    );
    for (k, &m) in limbs.iter().enumerate().skip(1) {
        assert_eq!(
            m, 0,
            "word {k} of an exact power of two must be empty - limbs 1..K              because the value is a power of two, and the last one because the              term is finite"
        );
    }
    assert_eq!(v.to_bits(), (1.0 as Scalar).to_bits(), "sum([1.0]) != 1.0");
}

/// The device accumulator against an independently written host one.
///
/// The host twin accumulates in `i128` and never touches the GPU, so an
/// agreement is evidence about the algorithm rather than about one
/// implementation of it.
#[test]
fn the_accumulator_agrees_with_an_independent_host_implementation() {
    let Some(gpu) = gpu() else { return };
    let sol = SolverKernels::new(&gpu).expect("kernels");

    let cases: Vec<Vec<Scalar>> = vec![
        vec![],
        vec![0.0],
        vec![1.0],
        vec![-1.0, 1.0],
        vec![1.0, ldexp(1.0, -53), ldexp(1.0, -53), ldexp(1.0, -53), ldexp(1.0, -53)],
        (0..1000).map(nasty).collect(),
        (0..77_777).map(nasty).collect(),
        (0..5000).map(|i| field(i) * if i % 2 == 0 { 1.0 } else { -1.0 }).collect(),
    ];

    for (i, terms) in cases.iter().enumerate() {
        let want = host_exact_sum(terms);
        let (got, _) = exact_sum_of(&gpu, &sol, std::slice::from_ref(terms), 3);
        assert_eq!(
            got.to_bits(),
            want.to_bits(),
            "case {i} ({} terms): device {got:e} != host {want:e}",
            terms.len()
        );
    }
}

/// The accumulator is more accurate than the tree it replaces, not less.
///
/// `1 + 4*2^-53` is exactly representable as `1 + 2^-51`; a `double` running
/// sum in ascending order rounds every one of the four additions away and
/// answers 1. This is the smallest case where "reproducible" and "right" point
/// the same way, and it is worth having one.
#[test]
fn the_accumulator_keeps_what_a_running_double_sum_throws_away() {
    let Some(gpu) = gpu() else { return };
    let sol = SolverKernels::new(&gpu).expect("kernels");

    let eps = ldexp(1.0 as Scalar, -53);
    let terms = vec![1.0 as Scalar, eps, eps, eps, eps];

    let mut naive = 0.0 as Scalar;
    for &t in &terms {
        naive += t;
    }
    assert_eq!(naive.to_bits(), (1.0 as Scalar).to_bits(), "the premise moved");

    let (got, _) = exact_sum_of(&gpu, &sol, &[terms], 0);
    assert_eq!(
        got.to_bits(),
        (1.0 as Scalar + ldexp(1.0, -51)).to_bits(),
        "the exact accumulator lost the tail as well: {got:e}"
    );
}

// ==========================================================================
//  2. The gate: the answer does not move
// ==========================================================================

/// The same terms, cut nine different ways, must give the same bits - and the
/// same limbs, which is the stronger statement because it compares the
/// accumulator rather than a rounded view of it.
#[test]
fn an_exact_sum_is_the_same_number_however_the_terms_are_dealt_out() {
    let Some(gpu) = gpu() else { return };
    let sol = SolverKernels::new(&gpu).expect("kernels");

    let n = 20_000;
    let (want, want_limbs) = exact_sum_of(&gpu, &sol, &[(0..n).map(nasty).collect()], 0);

    let mut runs = 0;
    for p in 1..=4 {
        for (name, parts) in [
            ("linear", linear_slices(n, p)),
            ("roundrobin", round_robin_slices(n, p)),
            ("shuffled", shuffled_slices(n, p)),
        ] {
            let total: usize = parts.iter().map(|v| v.len()).sum();
            assert_eq!(total, n, "{name} at P = {p} lost or duplicated a term");
            let (got, limbs) = exact_sum_of(&gpu, &sol, &parts, 5);
            assert_eq!(
                limbs, want_limbs,
                "{name} at P = {p}: the limb totals moved, so the accumulator \
                 itself is partition-dependent"
            );
            assert_eq!(
                got.to_bits(),
                want.to_bits(),
                "{name} at P = {p}: {got:e} != {want:e}"
            );
            runs += 1;
        }
    }
    assert_eq!(runs, 12);
}

/// The measurement the section exists to publish.
///
/// The gathered-partial construction - each part's own `device_sum`, the `P`
/// partials moved and finished by the existing one-block kernel - is what a
/// design note calls guarantee A, and it is what the brief for this work asked
/// for. It is reproducible run to run and it is **not** partition-invariant.
/// This test runs it and the exact accumulator over the identical data and
/// asserts that the first moves and the second does not.
///
/// If this ever stops failing for the gathered construction, the data has gone
/// benign and the test has stopped meaning anything - hence the assertion that
/// it moved, not merely the assertion that the exact one did not.
#[test]
fn the_gathered_partial_construction_moves_and_the_exact_one_does_not() {
    let Some(gpu) = gpu() else { return };
    let sol = SolverKernels::new(&gpu).expect("kernels");

    let n = 20_000;
    let whole: Vec<Scalar> = (0..n).map(nasty).collect();
    let gathered_whole = gathered_sum_of(&gpu, &sol, std::slice::from_ref(&whole));
    let (exact_whole, _) = exact_sum_of(&gpu, &sol, &[whole], 0);

    let mut gathered_moved = 0;
    let mut trials = 0;
    for p in 2..=4 {
        for parts in [
            linear_slices(n, p),
            round_robin_slices(n, p),
            shuffled_slices(n, p),
        ] {
            trials += 1;
            if gathered_sum_of(&gpu, &sol, &parts).to_bits() != gathered_whole.to_bits() {
                gathered_moved += 1;
            }
            // The dot twin of the cheap construction takes the same path and
            // is here so that it is exercised rather than merely present.
            {
                let (bufs, ns) = upload_parts(&gpu, &parts, 0);
                let mut red = ExactReduction::new(&gpu, &ns).expect("new");
                red.gathered_dot(&gpu, &sol, &bufs, &bufs).expect("gathered dot");
                gpu.sync().expect("sync");
                assert!(red.value(&gpu).expect("value") > 0.0, "(x,x) is positive");
            }
            let (exact, _) = exact_sum_of(&gpu, &sol, &parts, 0);
            assert_eq!(
                exact.to_bits(),
                exact_whole.to_bits(),
                "the exact accumulator moved, which is the only thing this \
                 section claims it cannot do"
            );
        }
    }
    assert!(
        gathered_moved > 0,
        "the gathered-partial construction agreed with the whole in all \
         {trials} partitions, so this test is no longer measuring anything - \
         the data has gone benign, not the construction correct"
    );
}

/// Relabelling the parts is the purest permutation there is: the same cells in
/// the same groups, only the group names changed. Nothing may move.
#[test]
fn relabelling_the_parts_changes_nothing() {
    let Some(gpu) = gpu() else { return };
    let sol = SolverKernels::new(&gpu).expect("kernels");

    let n = 9_000;
    for p in 2..=4 {
        let parts = shuffled_slices(n, p);
        let (want, want_limbs) = exact_sum_of(&gpu, &sol, &parts, 0);
        for shift in 1..p {
            let rotated: Vec<Vec<Scalar>> =
                (0..p).map(|r| parts[(r + shift) % p].clone()).collect();
            let (got, limbs) = exact_sum_of(&gpu, &sol, &rotated, 0);
            assert_eq!(limbs, want_limbs, "P = {p}, shift {shift}: limbs moved");
            assert_eq!(got.to_bits(), want.to_bits(), "P = {p}, shift {shift}");
        }
    }
}

/// A part that owns nothing must contribute nothing, and must not be a
/// different code path - a rank with no cells still enters every collective or
/// a real run deadlocks.
#[test]
fn a_part_that_owns_nothing_contributes_nothing() {
    let Some(gpu) = gpu() else { return };
    let sol = SolverKernels::new(&gpu).expect("kernels");

    let n = 3_000;
    let whole: Vec<Scalar> = (0..n).map(nasty).collect();
    let (want, want_limbs) = exact_sum_of(&gpu, &sol, std::slice::from_ref(&whole), 0);

    let with_empties = vec![
        Vec::new(),
        whole[..1000].to_vec(),
        Vec::new(),
        whole[1000..].to_vec(),
        Vec::new(),
    ];
    let (got, limbs) = exact_sum_of(&gpu, &sol, &with_empties, 2);
    assert_eq!(limbs, want_limbs);
    assert_eq!(got.to_bits(), want.to_bits());
}

/// A maximum is exactly order-independent, so `contErr`, the Courant number
/// and every adaptive-`dt` measure in the crate were already
/// partition-invariant. This asserts it rather than asserting it in prose.
#[test]
fn a_maximum_needs_none_of_the_accumulator() {
    let Some(gpu) = gpu() else { return };
    let sol = SolverKernels::new(&gpu).expect("kernels");

    let n = 12_000;
    let want = {
        let (bufs, ns) = upload_parts(&gpu, &[(0..n).map(nasty).collect()], 0);
        let mut red = ExactReduction::new(&gpu, &ns).expect("new");
        red.max_mag(&gpu, &sol, &bufs).expect("max");
        gpu.sync().expect("sync");
        red.value(&gpu).expect("value")
    };
    let host = (0..n).fold(0.0 as Scalar, |m, i| m.max(nasty(i).abs()));
    assert_eq!(want.to_bits(), host.to_bits(), "the max itself is wrong");

    for p in 1..=4 {
        for parts in [
            linear_slices(n, p),
            round_robin_slices(n, p),
            shuffled_slices(n, p),
        ] {
            let (bufs, ns) = upload_parts(&gpu, &parts, 4);
            let mut red = ExactReduction::new(&gpu, &ns).expect("new");
            red.max_mag(&gpu, &sol, &bufs).expect("max");
            gpu.sync().expect("sync");
            assert_eq!(
                red.value(&gpu).expect("value").to_bits(),
                want.to_bits(),
                "P = {p}: a maximum moved, which would mean the gather is not a copy"
            );
        }
    }
}

// ==========================================================================
//  3. The other three shapes
// ==========================================================================

/// `sum_mag`, `dot` and `norm_factor` are the three remaining term expressions
/// in the crate. `dot` and `norm_factor` carry their own anchor kernels, so
/// neither is covered by `sum`'s gate; `sum_mag` shares `sum`'s anchor but not
/// its stage one.
#[test]
fn the_dot_the_magnitude_sum_and_the_normalisation_factor_survive_the_cut_too() {
    let Some(gpu) = gpu() else { return };
    let sol = SolverKernels::new(&gpu).expect("kernels");

    let n = 8_000;
    let a: Vec<Scalar> = (0..n).map(nasty).collect();
    let b: Vec<Scalar> = (0..n).map(|i| nasty(i + 13)).collect();
    let c: Vec<Scalar> = (0..n).map(|i| field(i) - 1.0).collect();

    let deal = |parts: usize, map: &dyn Fn(usize) -> usize| -> Vec<Vec<usize>> {
        let mut out = vec![Vec::new(); parts];
        for i in 0..n {
            out[map(i) % parts].push(i);
        }
        out
    };

    let run = |ids: &[Vec<usize>]| -> (Scalar, Scalar, Scalar) {
        let ns: Vec<usize> = ids.iter().map(|v| v.len()).collect();
        let up = |src: &[Scalar]| -> Vec<DevBuf<Scalar>> {
            ids.iter()
                .map(|v| {
                    let mut d: Vec<Scalar> = v.iter().map(|&i| src[i]).collect();
                    d.push(1.0e30);
                    gpu.upload(&d).expect("upload")
                })
                .collect()
        };
        let (da, db, dc) = (up(&a), up(&b), up(&c));
        let mut red = ExactReduction::new(&gpu, &ns).expect("new");
        red.dot(&gpu, &sol, &da, &db).expect("dot");
        gpu.sync().expect("sync");
        let dot = red.value(&gpu).expect("value");
        red.norm_factor(&gpu, &sol, &da, &db, &dc, 0.0).expect("nf");
        gpu.sync().expect("sync");
        let nf = red.value(&gpu).expect("value");
        red.sum_mag(&gpu, &sol, &da).expect("sum_mag");
        gpu.sync().expect("sync");
        (dot, nf, red.value(&gpu).expect("value"))
    };

    let whole = deal(1, &|i| i);
    let (want_dot, want_nf, want_mag) = run(&whole);

    // The dot product must also BE the dot product: its terms are a[i]*b[i],
    // so the host twin of the accumulator answers it directly.
    let host_dot = host_exact_sum(&(0..n).map(|i| a[i] * b[i]).collect::<Vec<_>>());
    assert_eq!(want_dot.to_bits(), host_dot.to_bits(), "the dot itself is wrong");
    let host_nf = host_exact_sum(
        &(0..n)
            .map(|i| (a[i] - c[i]).abs() + (b[i] - c[i]).abs())
            .collect::<Vec<_>>(),
    );
    assert_eq!(want_nf.to_bits(), host_nf.to_bits(), "the norm factor itself is wrong");
    let host_mag = host_exact_sum(&a.iter().map(|x| x.abs()).collect::<Vec<_>>());
    assert_eq!(want_mag.to_bits(), host_mag.to_bits(), "the magnitude sum itself is wrong");

    for p in 2..=4 {
        let maps: [&dyn Fn(usize) -> usize; 3] =
            [&|i: usize| i, &|i: usize| i * 7 + i / 3, &|i: usize| i / 97];
        for map in maps {
            let (dot, nf, mag) = run(&deal(p, map));
            assert_eq!(dot.to_bits(), want_dot.to_bits(), "dot moved at P = {p}");
            assert_eq!(nf.to_bits(), want_nf.to_bits(), "normFactor moved at P = {p}");
            assert_eq!(mag.to_bits(), want_mag.to_bits(), "sumMag moved at P = {p}");
        }
    }
}

// ==========================================================================
//  4. The gate, through a real decomposition
// ==========================================================================

/// The one that ties this section to §71: a real mesh, cut by the real
/// partitioner, with the field distributed by `split_field` - so the buffers
/// are `n_cells + n_halo` long, the halo is a real halo, and the reduction has
/// to know not to sum it.
///
/// The halo is deliberately poisoned with `1e30` before the reduction runs. A
/// reduction that summed the ghost cells would not merely be wrong, it would
/// be wrong by twenty orders of magnitude, and the anchor would move with it.
#[test]
fn a_reduction_over_a_decomposed_mesh_is_the_undecomposed_reduction() {
    let Some(gpu) = gpu() else { return };
    let sol = SolverKernels::new(&gpu).expect("kernels");

    for cyclic in [false, true] {
        let m: HostMesh = boxes([6, 5, 4], cyclic);
        let n = m.n_cells;
        let global: Vec<Scalar> = (0..n).map(|c| nasty(c) + field(c)).collect();

        let (want, want_limbs) = exact_sum_of(&gpu, &sol, std::slice::from_ref(&global), 0);

        for p in 1..=4usize {
            let base = [
                ("hilbert", partition(&m, p, &PartitionMethod::Hilbert).expect("hilbert")),
                ("linear", partition(&m, p, &PartitionMethod::Linear).expect("linear")),
                (
                    "roundrobin",
                    (0..n).map(|c| (c % p) as Label).collect::<Vec<Label>>(),
                ),
            ];
            for (name, map) in base {
                // ... and the same map with the part labels rotated, which is
                // the permutation the gate is really about: the same cells in
                // the same groups, only the names changed.
                for shift in 0..p {
                    let rotated: Vec<Label> = map
                        .iter()
                        .map(|&r| ((r as usize + shift) % p) as Label)
                        .collect();
                    let d = Decomposition::from_map(&m, p, rotated).expect("decompose");
                    if p > 1 {
                        assert!(
                            d.parts.iter().any(|q| q.n_halo > 0),
                            "{name} at P = {p} produced no halo at all, so the \
                             cut is not being exercised"
                        );
                    }

                    let mut bufs = Vec::new();
                    let mut ns = Vec::new();
                    for q in 0..p {
                        let mut f = d.split_field(q, &global).expect("split");
                        let owned = d.parts[q].mesh.n_cells;
                        assert_eq!(f.len(), d.parts[q].n_local());
                        for h in f.iter_mut().skip(owned) {
                            *h = 1.0e30;
                        }
                        ns.push(owned);
                        bufs.push(gpu.upload(&f).expect("upload"));
                    }

                    let mut red = ExactReduction::new(&gpu, &ns).expect("new");
                    red.sum(&gpu, &sol, &bufs).expect("sum");
                    gpu.sync().expect("sync");
                    assert_eq!(
                        red.limbs(&gpu).expect("limbs"),
                        want_limbs,
                        "cyclic {cyclic}, {name} at P = {p} shift {shift}: limbs moved"
                    );
                    assert_eq!(
                        red.value(&gpu).expect("value").to_bits(),
                        want.to_bits(),
                        "cyclic {cyclic}, {name} at P = {p} shift {shift}"
                    );
                }
            }
        }
    }
}

/// A poisoned field must say so rather than answer plausibly.
///
/// The accumulator has no representation for an infinity: `ilogb(inf)` is
/// `INT_MAX`, and both the anchor arithmetic and the limb split would produce
/// a number with no relation to the input. So it returns a NaN, which is
/// visible, instead of a finite value, which would not be. A NaN term takes
/// the same path.
#[test]
fn a_non_finite_term_poisons_the_answer_visibly() {
    let Some(gpu) = gpu() else { return };
    let sol = SolverKernels::new(&gpu).expect("kernels");

    for bad in [Scalar::INFINITY, Scalar::NEG_INFINITY, Scalar::NAN] {
        let mut terms: Vec<Scalar> = (0..500).map(nasty).collect();
        terms[137] = bad;
        let (got, limbs) = exact_sum_of(&gpu, &sol, &[terms], 0);
        assert!(
            got.is_nan(),
            "a term of {bad} gave {got:e}, which a reader would take for an answer"
        );
        assert_eq!(limbs[EXACT_K], 1, "one non-finite term, counted once");
        assert_eq!(red_non_finite(&gpu, &sol, bad), 1, "the accessor says the same");
        // The host twin must say the same thing, or the agreement test above
        // is only checking the well-behaved half of the domain.
        let mut host_terms: Vec<Scalar> = (0..500).map(nasty).collect();
        host_terms[137] = bad;
        assert!(host_exact_sum(&host_terms).is_nan());
    }

    // And a field of exact zeros is not poisoned - it is zero.
    let (got, _) = exact_sum_of(&gpu, &sol, &[vec![0.0 as Scalar; 500]], 0);
    assert_eq!(got.to_bits(), (0.0 as Scalar).to_bits());
}

// ==========================================================================
//  5. Refusals - SPEC-LIT §13.4
// ==========================================================================

#[test]
fn an_impossible_reduction_is_refused_by_name() {
    let Some(gpu) = gpu() else { return };
    let sol = SolverKernels::new(&gpu).expect("kernels");

    let e = err(ExactReduction::new(&gpu, &[]));
    assert!(e.contains("zero parts"), "{e}");

    let e = err(ExactReduction::new(&gpu, &[EXACT_MAX_TERMS, 1]));
    assert!(e.contains("exceeds the"), "{e}");
    assert!(e.contains("§72.2"), "the refusal must name the section: {e}");

    let mut red = ExactReduction::new(&gpu, &[4, 4]).expect("new");
    let a = gpu.upload(&[1.0 as Scalar; 4]).expect("up");
    let b = gpu.upload(&[1.0 as Scalar; 4]).expect("up");
    let short = gpu.upload(&[1.0 as Scalar; 3]).expect("up");

    let e = red.sum(&gpu, &sol, std::slice::from_ref(&a)).unwrap_err().to_string();
    assert!(e.contains("1 `x` buffer(s) for 2 part(s)"), "{e}");

    let e = red
        .sum(&gpu, &sol, &[a.clone(), short.clone()])
        .unwrap_err()
        .to_string();
    assert!(e.contains("part 1's `x` buffer holds 3 values"), "{e}");
    assert!(e.contains("n_cells + n_halo"), "{e}");

    let e = red
        .dot(&gpu, &sol, &[a.clone(), b.clone()], &[a.clone(), short])
        .unwrap_err()
        .to_string();
    assert!(e.contains("dot: part 1's `b` buffer"), "{e}");

    // And the happy path still works after every refusal, so a refusal has not
    // left the accumulator in a state.
    red.sum(&gpu, &sol, &[a, b]).expect("sum");
    gpu.sync().expect("sync");
    assert_eq!(
        red.value(&gpu).expect("value").to_bits(),
        (8.0 as Scalar).to_bits()
    );
}

// ==========================================================================
//  6. The host twin, on its own
// ==========================================================================

/// `ldexp` and `anchor_exponent` are the two host helpers the twin is built
/// out of, and both have a subnormal case that is easy to get wrong.
#[test]
fn the_host_scaling_helpers_are_exact() {
    for e in -60..=60i32 {
        let want = if e >= 0 {
            (0..e).fold(1.0 as Scalar, |a, _| a * 2.0)
        } else {
            (0..-e).fold(1.0 as Scalar, |a, _| a * 0.5)
        };
        assert_eq!(ldexp(1.0, e).to_bits(), want.to_bits(), "ldexp(1, {e})");
        assert_eq!(anchor_exponent(want), e + 1, "anchor(2^{e})");
    }
    assert_eq!(anchor_exponent(0.0), 0);
    assert_eq!(anchor_exponent(0.75), 0);
    assert_eq!(anchor_exponent(1.0), 1);
    assert_eq!(ldexp(3.0, 0).to_bits(), (3.0 as Scalar).to_bits());
}
