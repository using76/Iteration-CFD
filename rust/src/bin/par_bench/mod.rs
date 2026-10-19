// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.
// Provenance: see PROVENANCE.md. No GPL-licensed source was consulted.

//! `ofgpu-par -bench` - SPEC-LIT §140's measured-scaling harness (PAR-16).
//!
//! Scaling is MEASURED, not modelled. Per rank count `P` and per repeat
//! (REPEATS independent worlds) this harness times three quantities on
//! every rank and takes the MAX over ranks:
//!
//! * `t_iter` - `distflow::cavity_part_timed`'s seconds per outer SIMPLE
//!   iteration, `warm` untimed steps then `iters` timed ones, device synced
//!   and world barriered on both sides of the timed block;
//! * `L_ex` - `SpmdHalo::scalar`'s halo exchange, 5 untimed then
//!   `-exchanges` timed, over the same sync-barrier frame;
//! * `L_ag` - `Comm::allgather` of 8 bytes per rank, the same shape.
//!
//! The five repeat values of each quantity are summarised by
//! equation (140.2) of SPEC-LIT 140.2: the median, the mean, the 95 % t
//! interval of the batch means (Student 1908's quantile at 4 dof) and the
//! range. Speedup and efficiency come from equation (140.1) of SPEC-LIT
//! 140.2 - strong (Amdahl 1967) or weak (Gustafson 1988) - but only while
//! `P <= D`, D the number of DISTINCT device ordinals in the placement:
//! at `P > D` the ranks time-share the card and the row prints
//! [`TIME_SHARED`] (SPEC-LIT 73.6's rule, 140.3). The MS-MPI column is
//! printed and written as [`MPI_NOT_RUN`]: the MPI transport is PAR-06,
//! deferred by the user.
//!
//! Provenance: ORIGINAL - the harness, its statistics and its refusal rule
//! are this project's own design; the DEFINITIONS they implement are cited
//! (Amdahl 1967; Gustafson 1988; Hoefler & Belli 2015; Student 1908) and no
//! external source code was consulted. No GPL-licensed source was
//! consulted.

use std::path::PathBuf;
use std::time::Instant;

use serde_json::json;

use ofgpu::blockgen::{build_case, CaseKind};
use ofgpu::comm::{Comm, Placement, ThreadComm};
use ofgpu::decompose::{Decomposition, PartitionMethod};
use ofgpu::device::Gpu;
use ofgpu::distflow::cavity_part_timed;
use ofgpu::error::{Error, Result};
use ofgpu::halo::SpmdHalo;
use ofgpu::Scalar;

use super::{method_named, refuse_unbuilt_transport, USAGE};

/// The independent repeats behind every statistic (Hoefler & Belli 2015:
/// report the median over repeats, not one number).
pub(crate) const REPEATS: usize = 5;

/// The two-sided 95 % Student t quantile at 4 degrees of freedom, the exact
/// f64 (Student 1908; the table value 2.776 to three decimals).
pub(crate) const T95_DF4: f64 = 2.7764451051977987;

/// The scaling label of a row whose ranks exceed the card: time-sharing is
/// not a scaling measurement (SPEC-LIT 73.6, 140.3).
pub(crate) const TIME_SHARED: &str = "time-shared, not scaling";

/// The MS-MPI cell of every table: the transport is PAR-06, deferred by the
/// user.
pub(crate) const MPI_NOT_RUN: &str =
    "NOT RUN - the MPI transport (PAR-06) is deferred by the user";

/// Five repeats of one quantity at one rank count, summarised by
/// equation (140.2) of SPEC-LIT 140.2.
pub(crate) struct Stats {
    /// The five batch means, in repeat order.
    pub(crate) samples: Vec<f64>,
    /// The sorted copy's middle sample, `x(3)`.
    pub(crate) median: f64,
    /// The sum in input order over 5.
    pub(crate) mean: f64,
    /// `mean +/- t95(4) s / sqrt(5)`, the t interval of the batch means.
    pub(crate) ci95: (f64, f64),
    /// `[min, max]` - an interval covering the median with probability
    /// `1 - 2 (1/2)^5 = 0.9375`, which is NOT a 95 % interval (140.2).
    pub(crate) range: (f64, f64),
}

/// Equation (140.2) of SPEC-LIT 140.2, on five samples in input order: the
/// median of the sorted copy, the mean summed in input order, the sample
/// standard deviation about that mean (divisor 4), the 95 % t interval of
/// the mean, and the range.
pub(crate) fn stats(x: &[f64; REPEATS]) -> Stats {
    let mut sorted = *x;
    sorted.sort_by(|a, b| a.partial_cmp(b).expect("a timing sample is not NaN"));
    let median = sorted[2];
    let mean = (x[0] + x[1] + x[2] + x[3] + x[4]) / REPEATS as f64;
    let mut ss = 0.0;
    for &xi in x {
        ss += (xi - mean) * (xi - mean);
    }
    let s = (ss / 4.0).sqrt();
    let half = T95_DF4 * s / (REPEATS as f64).sqrt();
    let lo = x.iter().cloned().fold(f64::INFINITY, f64::min);
    let hi = x.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    Stats {
        samples: x.to_vec(),
        median,
        mean,
        ci95: (mean - half, mean + half),
        range: (lo, hi),
    }
}

/// Strong (fixed mesh) or weak (fixed cells per rank).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Kind {
    Strong,
    Weak,
}

/// What a rank count's row may claim. `Reference` (P = 1) and `TimeShared`
/// (P > D) carry no speedup and no efficiency, by the rule of 140.3.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum Scaling {
    Reference,
    TimeShared,
    Measured { speedup: f64, efficiency: f64 },
    /// `1 < P <= D` but the run had no P = 1 row to scale against: no claim.
    NoReference,
}

/// Equation (140.1) of SPEC-LIT 140.2. `t1` and `tp` are the MEDIAN
/// per-iteration times at P = 1 and P. Strong scaling divides (Amdahl
/// 1967): `S = t(1)/t(P)`, `E = S/P`. Weak scaling multiplies (Gustafson
/// 1988): `E = t(1)/t(P)`, `S = P E`. P = 1 is the reference and claims
/// nothing; `P > D` distinct devices is time-shared and claims nothing
/// (SPEC-LIT 73.6's rule).
pub(crate) fn scaling(kind: Kind, t1: f64, tp: f64, p: usize, distinct_devices: usize) -> Scaling {
    if p == 1 {
        return Scaling::Reference;
    }
    if p > distinct_devices {
        return Scaling::TimeShared;
    }
    match kind {
        Kind::Strong => {
            let speedup = t1 / tp;
            Scaling::Measured {
                speedup,
                efficiency: speedup / p as f64,
            }
        }
        Kind::Weak => {
            let efficiency = t1 / tp;
            Scaling::Measured {
                speedup: p as f64 * efficiency,
                efficiency,
            }
        }
    }
}

/// The weak box's cell shape at `P` ranks: `a = ceil(sqrt(P))` incremented
/// until `P % a == 0`, then `(a, P/a)`. At `-n N` the weak mesh at P is
/// `build_case(Cavity, a*n, b*n, 1)` - exactly `N*N` cells per rank. The
/// domain stays the generator's 0.1 m square, so at `a != b` (P = 2, 3) the
/// cells are `a/b` : 1 anisotropic; the cell counts per side stay as near
/// equal as an exact divisor lets them.
pub(crate) fn weak_shape(p: usize) -> (usize, usize) {
    let mut a = (p as f64).sqrt().ceil() as usize;
    if a == 0 {
        a = 1;
    }
    while p % a != 0 {
        a += 1;
    }
    (a, p / a)
}

/// The number of DISTINCT device ordinals in the placement:
/// `OFGPU_DEVICES=0,0` still gives 1 - two ranks on one card are
/// time-sharing it, and 140.3 refuses the efficiency for exactly that.
pub(crate) fn distinct(devs: &[usize]) -> usize {
    let mut d = devs.to_vec();
    d.sort_unstable();
    d.dedup();
    d.len()
}

/// The options of `-bench strong|weak`, SPEC-LIT 140's harness. No `<case>`
/// is read; the mesh is a generated cavity box.
pub(crate) struct BenchArgs {
    /// "strong" or "weak" (refused by `run`'s dispatch before this parses).
    pub(crate) mode: &'static str,
    pub(crate) nps: Vec<usize>,
    pub(crate) n: usize,
    pub(crate) warm: usize,
    pub(crate) iters: usize,
    pub(crate) exchanges: usize,
    pub(crate) method: String,
    pub(crate) comm: String,
    pub(crate) json: Option<PathBuf>,
    pub(crate) quiet: bool,
}

/// Parses the bench's options. Every refusal here runs BEFORE any device is
/// opened: the transport through `refuse_unbuilt_transport` (PAR-06), a
/// zero count by the option's own name, an unknown option with the USAGE.
pub(crate) fn parse_bench(mode: &'static str, argv: &[String]) -> Result<BenchArgs> {
    let mut a = BenchArgs {
        mode,
        nps: vec![1, 2, 3, 4],
        n: 32,
        warm: 2,
        iters: 5,
        exchanges: 100,
        method: "hilbert".to_string(),
        comm: "threads".to_string(),
        json: None,
        quiet: false,
    };
    let mut i = 0;
    while i < argv.len() {
        let need = |i: usize| -> Result<String> {
            argv.get(i + 1)
                .cloned()
                .ok_or_else(|| Error::Config(format!("{} needs a value", argv[i])))
        };
        match argv[i].as_str() {
            "-np" => {
                let v = need(i)?;
                a.nps = v
                    .split(',')
                    .map(|s| s.trim().parse::<usize>())
                    .collect::<std::result::Result<_, _>>()
                    .map_err(|e| Error::Config(format!("-np '{v}': {e}")))?;
                if a.nps.iter().any(|&n| n == 0) {
                    return Err(Error::Config(format!(
                        "-np '{v}': a rank count must be at least 1"
                    )));
                }
                i += 2;
            }
            "-n" => {
                a.n = need(i)?
                    .parse()
                    .map_err(|e| Error::Config(format!("-n: {e}")))?;
                if a.n == 0 {
                    return Err(Error::Config(
                        "-n: cells per side must be >= 1".to_string(),
                    ));
                }
                i += 2;
            }
            "-warm" => {
                a.warm = need(i)?
                    .parse()
                    .map_err(|e| Error::Config(format!("-warm: {e}")))?;
                i += 2;
            }
            "-iters" => {
                a.iters = need(i)?
                    .parse()
                    .map_err(|e| Error::Config(format!("-iters: {e}")))?;
                if a.iters == 0 {
                    return Err(Error::Config(
                        "-iters: the timed steps must be at least 1".to_string(),
                    ));
                }
                i += 2;
            }
            "-exchanges" => {
                a.exchanges = need(i)?
                    .parse()
                    .map_err(|e| Error::Config(format!("-exchanges: {e}")))?;
                if a.exchanges == 0 {
                    return Err(Error::Config(
                        "-exchanges: the timed exchanges must be at least 1".to_string(),
                    ));
                }
                i += 2;
            }
            "-method" => {
                a.method = need(i)?;
                i += 2;
            }
            "-comm" => {
                a.comm = need(i)?;
                i += 2;
            }
            "-json" => {
                a.json = Some(PathBuf::from(need(i)?));
                i += 2;
            }
            "-quiet" => {
                a.quiet = true;
                i += 1;
            }
            other => {
                return Err(Error::Config(format!(
                    "unknown option '{other}'\n\n{USAGE}"
                )))
            }
        }
    }
    refuse_unbuilt_transport(&a.comm)?;
    // The partitioner is named before any device opens; `all` is the case
    // gates' sweep and is not a single cut to time.
    if !matches!(a.method.as_str(), "hilbert" | "linear" | "roundrobin") {
        return Err(Error::Config(format!(
            "-method '{}': known for -bench: hilbert, linear, roundrobin",
            a.method
        )));
    }
    Ok(a)
}

/// One rank count's measured row: the mesh it ran, the five repeats of each
/// quantity (each repeat's value the MAX over the ranks), and the scaling
/// claim the rule allows it.
struct Row {
    np: usize,
    nx: usize,
    ny: usize,
    cells: usize,
    t: Stats,
    ex: Stats,
    ag: Stats,
    scaling: Scaling,
}

/// SPEC-LIT 140's harness. Refusals already happened in `parse_bench`; this
/// opens the devices, measures every rank count REPEATS times, prints one
/// line per P, the MS-MPI line, the efficiency refusal when any P > D, and
/// writes the JSON. A measurement is not a gate: the exit status is 0 when
/// the run completed, whatever the numbers say.
pub(crate) fn run_bench(args: &BenchArgs) -> Result<()> {
    let kind = match args.mode {
        "strong" => Kind::Strong,
        _ => Kind::Weak,
    };
    let t_all = Instant::now();
    let device_count = Gpu::device_count()?;
    let placement = Placement::from_env(device_count)?;
    let devices = distinct(placement.devices());
    // The main thread's own context, as the case gates hold one while the
    // worlds open theirs; it also names the card for the report.
    let gpu0 = Gpu::for_rank(&placement, 0)?;
    let device = gpu0.name()?;
    if !args.quiet {
        println!(
            "bench  : {} - the per-iteration time, L_ex and L_ag (SPEC-LIT 140)",
            args.mode
        );
        println!(
            "device : {device} (placement {:?}, {devices} distinct)",
            placement.devices()
        );
        println!(
            "mesh   : strong {0} x {0}; weak a x b per rank, a*b = {0} x {0} cells",
            args.n
        );
        println!(
            "runs   : np {:?}, warm {}, iters {}, exchanges {}, repeats {}",
            args.nps, args.warm, args.iters, args.exchanges, REPEATS
        );
        println!();
    }

    let mut rows: Vec<Row> = Vec::with_capacity(args.nps.len());
    for &np in &args.nps {
        if !args.quiet {
            println!("P = {np:<2} measuring {REPEATS} repeat(s) ...");
        }
        rows.push(bench_one(args, kind, &placement, np)?);
    }
    let secs = t_all.elapsed().as_secs_f64();

    // The rule of 140.3, applied once every median exists: P = 1 is the
    // reference; P > D is time-shared; the rest get equation (140.1)'s S
    // and E against the P = 1 median. A list with no P = 1 ran has no
    // reference to scale against, so its rows claim nothing either and say
    // why (`NoReference`), rather than borrowing the time-shared label.
    let t1 = rows.iter().find(|r| r.np == 1).map(|r| r.t.median);
    for row in &mut rows {
        row.scaling = if row.np == 1 {
            Scaling::Reference
        } else if row.np > devices {
            Scaling::TimeShared
        } else {
            match t1 {
                Some(t1) => scaling(kind, t1, row.t.median, row.np, devices),
                None => Scaling::NoReference,
            }
        };
    }

    for row in &rows {
        print_row(row);
    }
    println!("MS-MPI: {MPI_NOT_RUN}");
    if rows.iter().any(|r| r.scaling == Scaling::TimeShared) {
        println!(
            "efficiency refused: P > D = {devices} device(s) - {TIME_SHARED} \
             (SPEC-LIT 73.6)"
        );
    }
    println!(
        "ofgpu-par -bench {}: DONE - {} rows, {secs:.1} s",
        args.mode,
        rows.len()
    );
    if let Some(path) = &args.json {
        write_json(args, &rows, &device, device_count, devices, secs, path)?;
    }
    Ok(())
}

/// One rank count: its box, its cut, and REPEATS fresh worlds. Each repeat
/// is an INDEPENDENT world (its own `ThreadComm`, its own `Gpu` per rank,
/// its own construction of the part) so a repeat measures what a user's run
/// would, not a warm process's steady state. The repeat's value of every
/// quantity is the MAX over ranks: the world runs at its slowest rank's
/// pace, so that is the number a scaling figure must be honest about.
fn bench_one(
    args: &BenchArgs,
    kind: Kind,
    placement: &Placement,
    np: usize,
) -> Result<Row> {
    let (nx, ny) = match kind {
        Kind::Strong => (args.n, args.n),
        Kind::Weak => {
            let (a, b) = weak_shape(np);
            (a * args.n, b * args.n)
        }
    };
    let hm = build_case(CaseKind::Cavity, nx, ny, 1)?.0;
    let cells = hm.n_cells;
    // P = 1 is the Linear cut, which IS the one-part cut (`run_solve`'s
    // rule); any other P takes the named method.
    let method = if np == 1 {
        PartitionMethod::Linear
    } else {
        method_named(&args.method, &hm, np)?
    };
    let dec = Decomposition::build(&hm, np, &method)?;

    let mut t = [0.0; REPEATS];
    let mut ex = [0.0; REPEATS];
    let mut ag = [0.0; REPEATS];
    for rep in 0..REPEATS {
        let got = ThreadComm::run(np, |comm| -> Result<(f64, f64, f64)> {
            rank_measure(placement, comm, &dec, args.warm, args.iters, args.exchanges)
        })?;
        let got = got
            .into_iter()
            .enumerate()
            .map(|(r, x)| x.map_err(|e| Error::Config(format!("P = {np} rep {rep} rank {r}: {e}"))))
            .collect::<Result<Vec<_>>>()?;
        t[rep] = got.iter().map(|x| x.0).fold(f64::NEG_INFINITY, f64::max);
        ex[rep] = got.iter().map(|x| x.1).fold(f64::NEG_INFINITY, f64::max);
        ag[rep] = got.iter().map(|x| x.2).fold(f64::NEG_INFINITY, f64::max);
    }
    Ok(Row {
        np,
        nx,
        ny,
        cells,
        t: stats(&t),
        ex: stats(&ex),
        ag: stats(&ag),
        scaling: Scaling::TimeShared,
    })
}

/// One rank's three measurements in one fresh world. `t_iter` comes from
/// `cavity_part_timed` (the whole decomposed SIMPLE construction, warm
/// steps then a synced, barriered, timed block); `L_ex` from
/// `SpmdHalo::scalar` over a full-length buffer poisoned with the rank's
/// own value; `L_ag` from `Comm::allgather` of 8 bytes per rank. Both
/// latency blocks are 5 untimed calls, then `gpu.sync` + `comm.barrier`,
/// then the timed calls, then sync + barrier again - the trailing barrier
/// is what makes the last exchange's completion THIS rank's time, at the
/// cost of including it in the mean (140.1's stated bias, the same on both
/// sides and on both transports, so it moves L_ex and L_ag together).
fn rank_measure(
    placement: &Placement,
    comm: &dyn Comm,
    dec: &Decomposition,
    warm: usize,
    iters: usize,
    exchanges: usize,
) -> Result<(f64, f64, f64)> {
    let r = comm.rank();
    let gpu = Gpu::for_rank(placement, r)?;
    let t_it = cavity_part_timed(&gpu, comm, dec, warm, iters)?;

    let part = &dec.parts[r];
    let mut halo = SpmdHalo::new(&gpu, part, comm.size())?;
    let len = part.mesh.n_cells + part.n_halo;
    let mut buf = gpu.upload(&vec![r as Scalar; len])?;
    for _ in 0..5 {
        halo.scalar(&gpu, comm, &mut buf)?;
    }
    gpu.sync()?;
    comm.barrier()?;
    let t0 = Instant::now();
    for _ in 0..exchanges {
        halo.scalar(&gpu, comm, &mut buf)?;
    }
    gpu.sync()?;
    comm.barrier()?;
    let l_ex = t0.elapsed().as_secs_f64() / exchanges as f64;

    let mine = (r as f64).to_le_bytes();
    let mut all = vec![0u8; 8 * comm.size()];
    for _ in 0..5 {
        comm.allgather(&mine, &mut all)?;
    }
    comm.barrier()?;
    let t0 = Instant::now();
    for _ in 0..exchanges {
        comm.allgather(&mine, &mut all)?;
    }
    comm.barrier()?;
    let l_ag = t0.elapsed().as_secs_f64() / exchanges as f64;
    Ok((t_it, l_ex, l_ag))
}

/// One P's line: the medians with their 95 % t intervals, the row's
/// cells-per-rank, and the scaling claim.
fn print_row(row: &Row) {
    let cpr = row.cells as f64 / row.np as f64;
    let cpr = if cpr.fract() == 0.0 {
        format!("{}", cpr as usize)
    } else {
        format!("{cpr:.1}")
    };
    println!(
        "P = {:<2} cells {} ({}/rank)  t_iter median {:.3e} s [95% CI {:.3e}, \
         {:.3e}]  L_ex median {:.3e} s [95% CI {:.3e}, {:.3e}]  L_ag median \
         {:.3e} s [95% CI {:.3e}, {:.3e}]  {}",
        row.np,
        row.cells,
        cpr,
        row.t.median,
        row.t.ci95.0,
        row.t.ci95.1,
        row.ex.median,
        row.ex.ci95.0,
        row.ex.ci95.1,
        row.ag.median,
        row.ag.ci95.0,
        row.ag.ci95.1,
        scaling_label(row.scaling),
    );
}

/// The row's trailing label: "reference", TIME_SHARED, or the measured
/// speedup and efficiency.
fn scaling_label(s: Scaling) -> String {
    match s {
        Scaling::Reference => "reference".to_string(),
        Scaling::TimeShared => TIME_SHARED.to_string(),
        Scaling::Measured { speedup, efficiency } => {
            format!("speedup {speedup:.2}, efficiency {efficiency:.2}")
        }
        Scaling::NoReference => "no P = 1 reference".to_string(),
    }
}

/// The JSON name of a row's scaling claim.
fn scaling_name(s: Scaling) -> &'static str {
    match s {
        Scaling::Reference => "reference",
        Scaling::TimeShared => TIME_SHARED,
        Scaling::Measured { .. } => "measured",
        Scaling::NoReference => "no reference",
    }
}

/// One quantity's five repeats, as JSON (all seconds).
fn stats_json(s: &Stats) -> serde_json::Value {
    json!({
        "samples": s.samples,
        "median": s.median,
        "mean": s.mean,
        "ci95": [s.ci95.0, s.ci95.1],
        "range": [s.range.0, s.range.1],
    })
}

/// The whole run's JSON document, written to `-json FILE`.
fn write_json(
    args: &BenchArgs,
    rows: &[Row],
    device: &str,
    device_count: usize,
    devices: usize,
    secs: f64,
    path: &std::path::Path,
) -> Result<()> {
    let doc = json!({
        "bench": args.mode,
        "transport": "threads",
        "device": device,
        "devices": devices,
        "device_count": device_count,
        "precision": if cfg!(feature = "single") { "f32" } else { "f64" },
        "n": args.n,
        "warm": args.warm,
        "iters": args.iters,
        "repeats": REPEATS,
        "exchanges": args.exchanges,
        "method": args.method,
        "mpi": MPI_NOT_RUN,
        "seconds": secs,
        "rows": rows.iter().map(|r| {
            let (speedup, efficiency) = match r.scaling {
                Scaling::Measured { speedup, efficiency } => (json!(speedup), json!(efficiency)),
                _ => (serde_json::Value::Null, serde_json::Value::Null),
            };
            json!({
                "np": r.np,
                "nx": r.nx,
                "ny": r.ny,
                "cells": r.cells,
                "cells_per_rank": r.cells as f64 / r.np as f64,
                "t_iter": stats_json(&r.t),
                "l_ex": stats_json(&r.ex),
                "l_ag": stats_json(&r.ag),
                "scaling": scaling_name(r.scaling),
                "speedup": speedup,
                "efficiency": efficiency,
            })
        }).collect::<Vec<_>>(),
    });
    let text = serde_json::to_string_pretty(&doc)
        .map_err(|e| Error::Config(format!("-json {}: {e}", path.display())))?;
    std::fs::write(path, text).map_err(|e| Error::Config(format!("-json {}: {e}", path.display())))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Class A: (140.2)'s own f64 rounding, on two permutations of 1..=5
    /// (the order of the sum must not move the mean) and on a flat sample.
    #[test]
    fn stats_of_one_to_five() {
        let s = stats(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        assert_eq!(s.median, 3.0);
        assert_eq!(s.mean, 3.0);
        assert_eq!(s.range, (1.0, 5.0));
        let want = (1.0367568385224393, 4.9632431614775605);
        assert!((s.ci95.0 - want.0).abs() <= 1e-14 * want.0.abs(), "{:?}", s.ci95);
        assert!((s.ci95.1 - want.1).abs() <= 1e-14 * want.1.abs(), "{:?}", s.ci95);

        let s = stats(&[5.0, 1.0, 4.0, 2.0, 3.0]);
        assert_eq!(s.median, 3.0);
        assert_eq!(s.mean, 3.0);
        assert_eq!(s.range, (1.0, 5.0));
        assert!((s.ci95.0 - want.0).abs() <= 1e-14 * want.0.abs(), "{:?}", s.ci95);
        assert!((s.ci95.1 - want.1).abs() <= 1e-14 * want.1.abs(), "{:?}", s.ci95);
    }

    /// The median ignores the outlier the mean must carry, and a constant
    /// sample has a degenerate (exactly equal) interval.
    #[test]
    fn stats_of_a_skewed_and_a_flat_sample() {
        let s = stats(&[10.0, 12.0, 11.0, 30.0, 13.0]);
        assert_eq!(s.median, 12.0);
        assert_eq!(s.mean, 15.2);
        let want = (4.833778650581262, 25.566221349418736);
        assert!((s.ci95.0 - want.0).abs() <= 1e-14 * want.0.abs(), "{:?}", s.ci95);
        assert!((s.ci95.1 - want.1).abs() <= 1e-14 * want.1.abs(), "{:?}", s.ci95);

        let s = stats(&[2.0; REPEATS]);
        assert_eq!(s.ci95, (2.0, 2.0));
    }

    /// Equation (140.1): strong divides, weak multiplies, and the rule of
    /// 140.3 refuses S and E at P = 1 (reference) and at P > D
    /// (time-shared).
    #[test]
    fn strong_weak_and_refused_scaling() {
        assert_eq!(
            scaling(Kind::Strong, 1.0, 0.5, 2, 2),
            Scaling::Measured { speedup: 2.0, efficiency: 1.0 }
        );
        assert_eq!(
            scaling(Kind::Strong, 2.0, 0.625, 4, 4),
            Scaling::Measured { speedup: 3.2, efficiency: 0.8 }
        );
        assert_eq!(
            scaling(Kind::Weak, 1.0, 1.25, 4, 4),
            Scaling::Measured { speedup: 3.2, efficiency: 0.8 }
        );
        assert_eq!(scaling(Kind::Strong, 1.0, 0.5, 2, 1), Scaling::TimeShared);
        assert_eq!(scaling(Kind::Weak, 1.0, 1.0, 4, 2), Scaling::TimeShared);
        assert_eq!(scaling(Kind::Strong, 1.0, 0.5, 1, 1), Scaling::Reference);
        assert_eq!(scaling(Kind::Weak, 1.0, 0.5, 1, 1), Scaling::Reference);
    }

    /// The weak box's shape: nearest divisor pair at or above sqrt(P), so
    /// `a*b = P` exactly; and the distinct-ordinal count behind D.
    #[test]
    fn weak_boxes_and_distinct_devices() {
        assert_eq!(weak_shape(1), (1, 1));
        assert_eq!(weak_shape(2), (2, 1));
        assert_eq!(weak_shape(3), (3, 1));
        assert_eq!(weak_shape(4), (2, 2));
        assert_eq!(weak_shape(6), (3, 2));
        assert_eq!(weak_shape(8), (4, 2));
        assert_eq!(distinct(&[0, 0]), 1);
        assert_eq!(distinct(&[0, 1, 0]), 2);
    }
}

