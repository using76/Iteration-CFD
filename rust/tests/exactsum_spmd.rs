// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.
// Provenance: see PROVENANCE.md. No GPL-licensed source was consulted.

//! SPEC-LIT §138.4's gate - the SPMD exact reduction, in BOTH builds.
//!
//! An integration test because the library's own test target does not
//! compile under `--features single`; an integration test links the normal
//! (non-test) library, which does. Every rank is a [`ThreadComm`] thread
//! with its own [`Gpu`] on device 0 and its own [`SolverKernels`], and the
//! references are the in-process [`ExactReduction`] on the main thread's
//! card: a reduction run over `P` ranks must produce the bits the
//! one-process, `P`-part run produces, and the bits the one-part run
//! produces.
//!
//! Class D of docs/17 §5.1: exact bytes. Every value comparison is
//! `to_bits()`, every limb comparison is `==` on the `i64` words, and the
//! count of comparisons is asserted at the end, so a silent narrowing of
//! the gate cannot pass.
//!
//! Provenance: ORIGINAL - the SPMD form under test is this project's own
//! design (SPEC-LIT §138.4, PAR-03) on the accumulator of §72. No external
//! source. No GPL-licensed source was consulted.

use ofgpu::comm::{Comm, ThreadComm};
use ofgpu::device::{DevBuf, Gpu};
use ofgpu::exactsum::{ExactReduction, SpmdReduction, EXACT_MAX_TERMS, EXACT_WORDS, ldexp};
use ofgpu::solver::SolverKernels;
use ofgpu::{Result, Scalar};

/// How many terms the gate deals out.
const N: usize = 20_011;

/// The poison that stands for the halo, appended to every uploaded buffer:
/// a method that reads past the rank's own `n` values fails on it.
const PAD: usize = 3;

/// `norm_factor`'s offset, added once by the cross-rank combine.
const EPS: Scalar = Scalar::MIN_POSITIVE;

/// A term with a full-width mantissa, a hundred-binade exponent range and
/// mixed signs - `exactsum/tests.rs`'s `nasty`, unchanged: a pure function
/// of the index, so the data is the same on every machine and every run.
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

/// The dealing: which part global index `i` belongs to. The SAME map is
/// applied to x, y and z.
fn deal(kind: &str, i: usize, p: usize) -> usize {
    match kind {
        // part r gets [(r*N)/P, ((r+1)*N)/P)
        "contiguous" => {
            let mut r = 0;
            while r + 1 < p && i >= ((r + 1) * N) / p {
                r += 1;
            }
            r
        }
        "round robin" => i % p,
        "shuffled" => (i * 7 + i / 3) % p,
        // contiguous over P-1 parts, with an EMPTY part inserted at index 1
        "one empty" => {
            let mut r = 0;
            while r + 1 < p - 1 && i >= ((r + 1) * N) / (p - 1) {
                r += 1;
            }
            if r >= 1 { r + 1 } else { r }
        }
        other => panic!("unknown dealing {other}"),
    }
}

/// How many terms part `part` owns under the dealing.
fn part_len(kind: &str, part: usize, p: usize) -> usize {
    (0..N).filter(|&i| deal(kind, i, p) == part).count()
}

/// One part's slice of `global` under the dealing, then `PAD` poison values.
fn local(global: &[Scalar], kind: &str, part: usize, p: usize) -> Vec<Scalar> {
    let mut v: Vec<Scalar> = (0..N)
        .filter(|&i| deal(kind, i, p) == part)
        .map(|i| global[i])
        .collect();
    v.extend(std::iter::repeat_n(1.0e30 as Scalar, PAD));
    v
}

/// The three arrays cut into parts: `parts[g][r]` is array `g`'s terms on
/// part `r`, without the padding.
fn cut(globals: &[&[Scalar]], kind: &str, p: usize) -> Vec<Vec<Vec<Scalar>>> {
    let mut parts = vec![vec![Vec::new(); p]; 3];
    for i in 0..N {
        let r = deal(kind, i, p);
        for (g, global) in globals.iter().enumerate() {
            parts[g][r].push(global[i]);
        }
    }
    parts
}

/// Upload one padded buffer per part, in `order`: slot j is part order[j].
fn upload(gpu: &Gpu, parts: &[Vec<Scalar>], order: &[usize]) -> Vec<DevBuf<Scalar>> {
    order
        .iter()
        .map(|&r| {
            let mut v = parts[r].clone();
            v.extend(std::iter::repeat_n(1.0e30 as Scalar, PAD));
            gpu.upload(&v).expect("upload")
        })
        .collect()
}

/// What one run of one operation recorded: the answer's bits, and for the
/// exact constructions also the limbs and the anchor's bits.
struct Rec {
    val: u64,
    limbs: Vec<i64>,
    anchor: u64,
}

/// The nine operations, in the one fixed order both forms run them.
const OPS: [&str; 9] = [
    "sum",
    "sum_mag",
    "dot",
    "norm_factor",
    "max_mag",
    "gathered_sum",
    "gathered_dot",
    "gathered_sum_mag",
    "gathered_norm_factor",
];

/// The record of an exact construction: value bits, limbs, anchor bits.
fn rec_exact(red: &ExactReduction, gpu: &Gpu) -> Rec {
    gpu.sync().expect("sync");
    Rec {
        val: u64::from(red.value(gpu).expect("value").to_bits()),
        limbs: red.limbs(gpu).expect("limbs"),
        anchor: u64::from(red.anchor(gpu).expect("anchor").to_bits()),
    }
}

/// The record of a value-only construction.
fn rec_val(red: &ExactReduction, gpu: &Gpu) -> Rec {
    gpu.sync().expect("sync");
    Rec {
        val: u64::from(red.value(gpu).expect("value").to_bits()),
        limbs: Vec::new(),
        anchor: 0,
    }
}

/// Reference A: the in-process reduction over the parts in `order`, all
/// nine operations, on the main thread's card.
fn reference(
    gpu: &Gpu,
    sol: &SolverKernels,
    lens: &[usize],
    xs: &[DevBuf<Scalar>],
    ys: &[DevBuf<Scalar>],
    zs: &[DevBuf<Scalar>],
) -> Vec<Rec> {
    let mut red = ExactReduction::new(gpu, lens).expect("reference new");
    let mut out = Vec::with_capacity(9);

    red.sum(gpu, sol, xs).expect("reference sum");
    out.push(rec_exact(&red, gpu));
    red.sum_mag(gpu, sol, ys).expect("reference sum_mag");
    out.push(rec_exact(&red, gpu));
    red.dot(gpu, sol, xs, ys).expect("reference dot");
    out.push(rec_exact(&red, gpu));
    red.norm_factor(gpu, sol, xs, ys, zs, EPS)
        .expect("reference norm_factor");
    out.push(rec_exact(&red, gpu));
    red.max_mag(gpu, sol, xs).expect("reference max_mag");
    out.push(rec_val(&red, gpu));
    red.gathered_sum(gpu, sol, xs).expect("reference gathered_sum");
    out.push(rec_val(&red, gpu));
    red.gathered_dot(gpu, sol, xs, ys).expect("reference gathered_dot");
    out.push(rec_val(&red, gpu));
    red.gathered_sum_mag(gpu, sol, ys)
        .expect("reference gathered_sum_mag");
    out.push(rec_val(&red, gpu));
    red.gathered_norm_factor(gpu, sol, xs, ys, zs, EPS)
        .expect("reference gathered_norm_factor");
    out.push(rec_val(&red, gpu));
    out
}

/// What one rank recorded: nine values' bits, the four exact constructions'
/// limbs and anchors, and the handle's counters.
struct RankOut {
    vals: [u64; 9],
    limbs: [[i64; EXACT_WORDS]; 4],
    anchors: [u64; 4],
    allgathers: u64,
    counts: Vec<usize>,
}

/// **The gate.** 20,011 terms of full mantissa spread over 100 binades,
/// `P = 1..4`, four dealings, every rotation of the rank labels: on every
/// rank the five whole-value operations equal the in-process `ExactReduction`
/// AND the one-part answer by bits, the four exact ones also by limbs and
/// anchor, and the four gathered ones equal the in-process gathered answer
/// for the same part order. 1,071 answers are compared, and each handle has
/// made 14 allgathers.
#[test]
fn spmd_exact_reduction_is_the_in_process_one() {
    let Some(gpu0) = Gpu::new(0).ok() else { return };
    let sol0 = SolverKernels::new(&gpu0).expect("solver kernels");

    let x: Vec<Scalar> = (0..N).map(nasty).collect();
    let y: Vec<Scalar> = (0..N).map(|i| nasty(i + 7919)).collect();
    let z: Vec<Scalar> = (0..N).map(|i| nasty(i + 104_729)).collect();

    // ---- Reference B: the one-part answer over the whole arrays ----------
    let mut red_b = ExactReduction::new(&gpu0, &[N]).expect("B new");
    let bx = gpu0.upload(&x).expect("upload");
    let by = gpu0.upload(&y).expect("upload");
    let bz = gpu0.upload(&z).expect("upload");
    let xs = [bx];
    let ys = [by];
    let zs = [bz];
    let mut b = [0u64; 5];
    red_b.sum(&gpu0, &sol0, &xs).expect("B sum");
    b[0] = u64::from(red_b.value(&gpu0).expect("value").to_bits());
    red_b.sum_mag(&gpu0, &sol0, &ys).expect("B sum_mag");
    b[1] = u64::from(red_b.value(&gpu0).expect("value").to_bits());
    red_b.dot(&gpu0, &sol0, &xs, &ys).expect("B dot");
    b[2] = u64::from(red_b.value(&gpu0).expect("value").to_bits());
    red_b
        .norm_factor(&gpu0, &sol0, &xs, &ys, &zs, EPS)
        .expect("B norm_factor");
    b[3] = u64::from(red_b.value(&gpu0).expect("value").to_bits());
    red_b.max_mag(&gpu0, &sol0, &xs).expect("B max_mag");
    b[4] = u64::from(red_b.value(&gpu0).expect("value").to_bits());
    drop(red_b);

    let globals: [&[Scalar]; 3] = [&x, &y, &z];
    let mut comparisons = 0usize;

    for pcount in 1..=4usize {
        let dealings: &[&str] = if pcount == 1 {
            &["contiguous", "round robin", "shuffled"]
        } else {
            &["contiguous", "round robin", "shuffled", "one empty"]
        };
        for &dealing in dealings {
            for rot in 0..pcount {
                // rank r holds part order[r]
                let order: Vec<usize> = (0..pcount).map(|r| (r + rot) % pcount).collect();
                let ctx = format!("P={pcount} {dealing} rotation {rot}");
                let lens: Vec<usize> =
                    order.iter().map(|&r| part_len(dealing, r, pcount)).collect();
                let parts = cut(&globals, dealing, pcount);

                // ---- Reference A: parts in `order`, on the main card -------
                let axs = upload(&gpu0, &parts[0], &order);
                let ays = upload(&gpu0, &parts[1], &order);
                let azs = upload(&gpu0, &parts[2], &order);
                let a = reference(&gpu0, &sol0, &lens, &axs, &ays, &azs);
                drop((axs, ays, azs));

                // ---- the SPMD run -----------------------------------------
                let got = ThreadComm::run(pcount, |comm| -> Result<RankOut> {
                    let gpu = Gpu::new(0)?;
                    let sol = SolverKernels::new(&gpu)?;
                    let part = order[comm.rank()];
                    let n_r = part_len(dealing, part, pcount);
                    let bx = gpu.upload(&local(&x, dealing, part, pcount))?;
                    let by = gpu.upload(&local(&y, dealing, part, pcount))?;
                    let bz = gpu.upload(&local(&z, dealing, part, pcount))?;
                    let mut red = SpmdReduction::new(&gpu, comm, n_r)?;

                    let mut vals = [0u64; 9];
                    let mut limbs = [[0i64; EXACT_WORDS]; 4];
                    let mut anchors = [0u64; 4];

                    red.sum(&gpu, &sol, comm, &bx)?;
                    vals[0] = u64::from(red.value(&gpu)?.to_bits());
                    limbs[0].copy_from_slice(&red.limbs(&gpu)?);
                    anchors[0] = u64::from(red.anchor(&gpu)?.to_bits());

                    red.sum_mag(&gpu, &sol, comm, &by)?;
                    vals[1] = u64::from(red.value(&gpu)?.to_bits());
                    limbs[1].copy_from_slice(&red.limbs(&gpu)?);
                    anchors[1] = u64::from(red.anchor(&gpu)?.to_bits());

                    red.dot(&gpu, &sol, comm, &bx, &by)?;
                    vals[2] = u64::from(red.value(&gpu)?.to_bits());
                    limbs[2].copy_from_slice(&red.limbs(&gpu)?);
                    anchors[2] = u64::from(red.anchor(&gpu)?.to_bits());

                    red.norm_factor(&gpu, &sol, comm, &bx, &by, &bz, EPS)?;
                    vals[3] = u64::from(red.value(&gpu)?.to_bits());
                    limbs[3].copy_from_slice(&red.limbs(&gpu)?);
                    anchors[3] = u64::from(red.anchor(&gpu)?.to_bits());

                    red.max_mag(&gpu, &sol, comm, &bx)?;
                    vals[4] = u64::from(red.value(&gpu)?.to_bits());

                    red.gathered_sum(&gpu, &sol, comm, &bx)?;
                    vals[5] = u64::from(red.value(&gpu)?.to_bits());
                    red.gathered_dot(&gpu, &sol, comm, &bx, &by)?;
                    vals[6] = u64::from(red.value(&gpu)?.to_bits());
                    red.gathered_sum_mag(&gpu, &sol, comm, &by)?;
                    vals[7] = u64::from(red.value(&gpu)?.to_bits());
                    red.gathered_norm_factor(&gpu, &sol, comm, &bx, &by, &bz, EPS)?;
                    vals[8] = u64::from(red.value(&gpu)?.to_bits());
                    gpu.sync()?;

                    Ok(RankOut {
                        vals,
                        limbs,
                        anchors,
                        allgathers: red.n_allgathers(),
                        counts: red.counts().to_vec(),
                    })
                })
                .expect("spmd run");

                // ---- every rank: bits against A and B ----------------------
                for r in 0..pcount {
                    let out = got[r]
                        .as_ref()
                        .unwrap_or_else(|e| panic!("rank {r} failed: {e}"));
                    assert_eq!(out.counts, lens, "counts, {ctx}, rank {r}");
                    assert_eq!(
                        out.allgathers, 14,
                        "allgathers (1 in new, 2 per exact op, 1 elsewhere), \
                         {ctx}, rank {r}"
                    );
                    for (op, name) in OPS.iter().enumerate() {
                        comparisons += 1;
                        assert_eq!(
                            out.vals[op], a[op].val,
                            "{name} value vs in-process, {ctx}, rank {r}"
                        );
                        if op < 4 {
                            assert_eq!(
                                &out.limbs[op][..], &a[op].limbs[..],
                                "{name} limbs vs in-process, {ctx}, rank {r}"
                            );
                            assert_eq!(
                                out.anchors[op], a[op].anchor,
                                "{name} anchor vs in-process, {ctx}, rank {r}"
                            );
                            assert_eq!(
                                out.vals[op], b[op],
                                "{name} value vs the one-part answer, {ctx}, rank {r}"
                            );
                        }
                        if op == 4 {
                            assert_eq!(
                                out.vals[4], b[4],
                                "max_mag value vs the one-part answer, {ctx}, rank {r}"
                            );
                        }
                    }
                }
            }
        }
    }
    assert_eq!(
        comparisons, 1071,
        "one comparison per (rank, operation): 119 rank results times 9 \
         operations"
    );
}

/// One NaN on rank 0 and one infinity on rank 2, `P = 3`: the sum answers
/// NaN on every rank and the non-finite count is 2 on every rank - the
/// count travels with the limbs, so no rank can see a plausible number,
/// and the anchor's `max` cannot hide the NaN (§72, the extra word).
#[test]
fn an_spmd_reduction_counts_a_non_finite_term_on_every_rank() {
    if Gpu::new(0).is_err() {
        return;
    }
    let got = ThreadComm::run(3, |comm| -> Result<(bool, i64)> {
        let gpu = Gpu::new(0)?;
        let sol = SolverKernels::new(&gpu)?;
        let rank = comm.rank();
        let mut terms: Vec<Scalar> =
            (0..300).filter(|&i| i % 3 == rank).map(nasty).collect();
        assert_eq!(terms.len(), 100, "rank {rank}: the round robin must deal 100");
        if rank == 0 {
            terms[17] = Scalar::NAN;
        }
        if rank == 2 {
            terms[5] = Scalar::INFINITY;
        }
        terms.extend(std::iter::repeat_n(1.0e30 as Scalar, PAD));
        let bx = gpu.upload(&terms)?;
        let mut red = SpmdReduction::new(&gpu, comm, 100)?;
        red.sum(&gpu, &sol, comm, &bx)?;
        gpu.sync()?;
        Ok((red.value(&gpu)?.is_nan(), red.non_finite_terms(&gpu)?))
    })
    .expect("spmd run");
    for r in 0..3 {
        let (nan, nft) = got[r]
            .as_ref()
            .unwrap_or_else(|e| panic!("rank {r} failed: {e}"));
        assert!(nan, "rank {r}: the sum must answer NaN");
        assert_eq!(*nft, 2, "rank {r}: one NaN and one infinity");
    }
}

/// The six refusals of §138.4's table, each `Error::Config` starting
/// `exactsum: `. Rows 4-6 return before any collective, so the handle's
/// allgather count is unchanged after each.
#[test]
fn an_spmd_reduction_refuses_by_name() {
    let gpu = match Gpu::new(0) {
        Ok(g) => g,
        Err(_) => return,
    };
    let sol = SolverKernels::new(&gpu).expect("solver kernels");

    // Row 1: rank 1's count does not fit a Label. The counts are gathered
    // first and checked afterwards, so BOTH ranks refuse, and both name
    // rank 1 and the count.
    let got = ThreadComm::run(2, |comm| -> Result<String> {
        let gpu = Gpu::new(0)?;
        let n = if comm.rank() == 0 { 5 } else { 2_147_483_648 };
        match SpmdReduction::new(&gpu, comm, n) {
            Ok(_) => panic!("row 1: rank {} was not refused", comm.rank()),
            Err(e) => Ok(e.to_string()),
        }
    })
    .expect("spmd run");
    for r in 0..2 {
        let msg = got[r]
            .as_ref()
            .unwrap_or_else(|e| panic!("rank {r} failed: {e}"));
        assert!(msg.starts_with("exactsum: "), "row 1 rank {r}: {msg}");
        assert!(msg.contains("rank 1"), "row 1 rank {r}: {msg}");
        assert!(msg.contains("2147483648"), "row 1 rank {r}: {msg}");
    }

    // Row 2: every rank's count fits, but the sum is over the limit.
    let got = ThreadComm::run(3, |comm| -> Result<String> {
        let gpu = Gpu::new(0)?;
        match SpmdReduction::new(&gpu, comm, 2_147_483_647) {
            Ok(_) => panic!("row 2: rank {} was not refused", comm.rank()),
            Err(e) => Ok(e.to_string()),
        }
    })
    .expect("spmd run");
    for r in 0..3 {
        let msg = got[r]
            .as_ref()
            .unwrap_or_else(|e| panic!("rank {r} failed: {e}"));
        assert!(msg.starts_with("exactsum: "), "row 2 rank {r}: {msg}");
        assert!(msg.contains("6442450941"), "row 2 rank {r}: {msg}");
        assert!(
            msg.contains(&EXACT_MAX_TERMS.to_string()),
            "row 2 rank {r}: {msg}"
        );
    }

    // Row 3: a reduction inside a capture. Afterwards, the same call
    // outside the capture is `Ok` - the refusal left nothing behind.
    let comm1 = &ThreadComm::world(1).expect("world")[0];
    let mut red3 = SpmdReduction::new(&gpu, comm1, 10).expect("new");
    let x10 = gpu.zeros::<Scalar>(10).expect("alloc");
    let msg = gpu
        .capture(|_| red3.sum(&gpu, &sol, comm1, &x10))
        .err()
        .expect("row 3 must refuse")
        .to_string();
    assert!(msg.starts_with("exactsum: "), "row 3: {msg}");
    assert!(msg.contains("capture"), "row 3: {msg}");
    assert!(msg.contains("ExactReduction"), "row 3: {msg}");
    red3.sum(&gpu, &sol, comm1, &x10).expect("after the refusal");

    // Rows 4-6 run on a fresh handle: its only allgather is `new`'s, so
    // `n_allgathers()` must still be 1 after each refusal.
    let mut red = SpmdReduction::new(&gpu, comm1, 10).expect("new");

    // Row 4: the world size changed under the handle.
    let w2 = ThreadComm::world(2).expect("world");
    let msg = red
        .max_mag(&gpu, &sol, &w2[0], &x10)
        .err()
        .expect("row 4 must refuse")
        .to_string();
    assert!(msg.starts_with("exactsum: "), "row 4: {msg}");
    assert!(msg.contains("2"), "row 4: {msg}");
    assert!(msg.contains("1"), "row 4: {msg}");
    assert!(msg.contains("ranks"), "row 4: {msg}");
    assert_eq!(red.n_allgathers(), 1, "row 4 refused before any collective");

    // Row 5: the handle belongs to another rank. Two separately made
    // worlds; each rank of a `run` calls with the OTHER handle of its own
    // pair, handed over by index through a Mutex.
    let mut other = ThreadComm::world(2).expect("world");
    let second = other.pop().expect("rank 1");
    let first = other.pop().expect("rank 0");
    let slots: std::sync::Mutex<Vec<Option<ThreadComm>>> =
        std::sync::Mutex::new(vec![Some(first), Some(second)]);
    let got = ThreadComm::run(2, |comm| -> Result<(String, u64)> {
        let gpu = Gpu::new(0)?;
        let sol = SolverKernels::new(&gpu)?;
        let mut red = SpmdReduction::new(&gpu, comm, 10)?;
        let handle = {
            let mut guard = slots.lock().expect("slots");
            guard[1 - comm.rank()].take().expect("the other rank's handle")
        };
        let x10 = gpu.zeros::<Scalar>(10)?;
        let msg = match red.sum(&gpu, &sol, &handle, &x10) {
            Ok(()) => panic!("row 5: rank {} was not refused", comm.rank()),
            Err(e) => e.to_string(),
        };
        Ok((msg, red.n_allgathers()))
    })
    .expect("spmd run");
    for r in 0..2 {
        let (msg, allgathers) = got[r]
            .as_ref()
            .unwrap_or_else(|e| panic!("rank {r} failed: {e}"));
        assert!(msg.starts_with("exactsum: "), "row 5 rank {r}: {msg}");
        assert!(msg.contains("rank 0"), "row 5 rank {r}: {msg}");
        assert!(msg.contains("rank 1"), "row 5 rank {r}: {msg}");
        assert_eq!(
            *allgathers, 1,
            "row 5 rank {r}: refused before any collective"
        );
    }

    // Row 6: a buffer shorter than `n`. The message names the buffer, its
    // length and `n`; the dot with a short SECOND buffer names `b`.
    let short = gpu.zeros::<Scalar>(9).expect("alloc");
    let msg = red
        .sum(&gpu, &sol, comm1, &short)
        .err()
        .expect("row 6 sum must refuse")
        .to_string();
    assert!(msg.starts_with("exactsum: "), "row 6 sum: {msg}");
    assert!(msg.contains("`x`"), "row 6 sum: {msg}");
    assert!(msg.contains("9"), "row 6 sum: {msg}");
    assert!(msg.contains("10"), "row 6 sum: {msg}");
    assert_eq!(red.n_allgathers(), 1, "row 6 sum refused before any collective");

    let msg = red
        .dot(&gpu, &sol, comm1, &x10, &short)
        .err()
        .expect("row 6 dot must refuse")
        .to_string();
    assert!(msg.starts_with("exactsum: "), "row 6 dot: {msg}");
    assert!(msg.contains("`b`"), "row 6 dot: {msg}");
    assert_eq!(red.n_allgathers(), 1, "row 6 dot refused before any collective");
}

