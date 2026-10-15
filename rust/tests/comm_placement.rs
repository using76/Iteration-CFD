// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.
// Provenance: see PROVENANCE.md. No GPL-licensed source was consulted.

//! SPEC-LIT §138.7's gates - ranks placed on devices, in BOTH builds.
//!
//! An integration test because the library's own test target does not
//! compile under `--features single`. The placement is a pure function, so
//! its test fakes the device counts 1, 2, 4 and 8. The GPU test places four
//! [`ThreadComm`] ranks on device 0 through a one-device placement, each with
//! its own [`Gpu`] (its own stream, event tracking off), and holds §138.5's
//! SPMD PCG to the in-process bits while all four run at once. The
//! two-device test is written for a machine with two GPUs and is ignored
//! here.
//!
//! Class D of docs/17 §5.1: exact bytes and exact counts.
//!
//! Provenance: ORIGINAL - the placement under test is this project's own
//! design (SPEC-LIT §138.7, PAR-05); the solve fixture is
//! `tests/distsolve_spmd.rs`'s. No external source. No GPL-licensed source
//! was consulted.

use std::time::Instant;

use ofgpu::blockgen::{build_case, CaseKind};
use ofgpu::comm::placement::DEVICES_ENV;
use ofgpu::comm::{Comm, Placement, ThreadComm};
use ofgpu::decompose::{Decomposition, PartitionMethod};
use ofgpu::device::{DevBuf, Gpu};
use ofgpu::distsolve::{
    dist_pcg, spmd_pcg, DistPerformance, DistReduce, DistSystem, DistWorkspace, SpmdWorkspace,
};
use ofgpu::halo::{HaloExchange, SpmdHalo};
use ofgpu::ldu::HostLduMatrix;
use ofgpu::mesh::HostMesh;
use ofgpu::solver::{LinearSolverKind, Preconditioner, SolverControls, SolverKernels};
use ofgpu::{Label, Result, Scalar};

/// The solver tolerance: a solver control, not a gate tolerance - every
/// comparison below is exact. `tests/distsolve_spmd.rs`'s values.
#[cfg(feature = "single")]
const TOL: Scalar = 1e-5;
#[cfg(not(feature = "single"))]
const TOL: Scalar = 1e-12;

const RDT: Scalar = 1.5;

fn field(c: usize) -> Scalar {
    0.25 + 0.0625 * ((c * 37) % 23) as Scalar
}

fn wall(c: usize) -> Scalar {
    0.5 + 0.125 * ((c * 11) % 7) as Scalar
}

/// `tests/distsolve_spmd.rs`'s SPD Poisson operator plus a `ddt`, unchanged.
fn poisson(m: &HostMesh) -> HostLduMatrix {
    let mut a = HostLduMatrix::zeros(m);
    for f in 0..m.n_internal_faces {
        let g = (m.mag_sf[f] * m.delta_coeffs[f]) as Scalar;
        a.upper[f] = -g;
        a.lower[f] = -g;
        a.diag[m.owner[f] as usize] += g;
        a.diag[m.neighbour[f] as usize] += g;
    }
    for bf in 0..m.n_boundary_faces {
        let g = (m.b_mag_sf[bf] * m.b_delta_coeffs[bf]) as Scalar;
        let c = m.b_face_cells[bf] as usize;
        a.diag[c] += g;
        if m.b_nbr_cell[bf] >= 0 {
            a.boundary_coeffs[bf] = g;
        } else {
            a.source[c] += g * wall(c);
        }
    }
    for c in 0..m.n_cells {
        let vd = (m.v[c] as Scalar) * RDT;
        a.diag[c] += vd;
        a.source[c] += vd * field(c);
    }
    a
}

fn controls(precon: Preconditioner) -> SolverControls {
    SolverControls {
        solver: LinearSolverKind::PCG,
        precon,
        tolerance: TOL,
        rel_tol: 0.0,
        max_iter: 400,
        min_iter: 0,
        check_interval: 1,
        fixed_iters: false,
        report_residuals: true,
        gamg: ofgpu::io::case::GamgControls::default(),
    }
}

/// The three partitioners of `tests/distsolve_spmd.rs` at `np` parts.
fn cuts(m: &HostMesh, np: usize) -> Vec<(&'static str, Decomposition)> {
    let rr = PartitionMethod::Explicit((0..m.n_cells).map(|c| (c % np) as Label).collect());
    [
        ("hilbert", PartitionMethod::Hilbert),
        ("linear", PartitionMethod::Linear),
        ("roundrobin", rr),
    ]
    .into_iter()
    .map(|(name, method)| (name, Decomposition::build(m, np, &method).expect("cut")))
    .collect()
}

/// The message of a refused placement.
fn refusal(count: usize, value: Option<&str>) -> String {
    match Placement::new(count, value) {
        Ok(p) => panic!(
            "count {count} value {value:?} was accepted as {:?}",
            p.devices()
        ),
        Err(e) => e.to_string(),
    }
}

/// **The pure gate.** `rank mod device_count` at faked counts 1, 2, 4 and 8,
/// the override lists, and every refusal by name.
#[test]
fn placement_maps_ranks_to_devices() {
    let mut checked = 0usize;
    for n in [1usize, 2, 4, 8] {
        let p = Placement::new(n, None).expect("default placement");
        assert!(!p.is_overridden());
        assert_eq!(p.device_count(), n);
        assert_eq!(p.devices(), (0..n).collect::<Vec<_>>().as_slice());
        for rank in 0..32usize {
            assert_eq!(p.device_of(rank), rank % n, "count {n} rank {rank}");
            checked += 1;
        }
    }
    assert_eq!(checked, 128, "ranks checked");

    // The override: the listed ordinals, cycled in rank order.
    let cases: [(usize, &str, [usize; 6]); 5] = [
        (2, "1", [1, 1, 1, 1, 1, 1]),
        (4, "3,1", [3, 1, 3, 1, 3, 1]),
        (4, " 0 , 2 ", [0, 2, 0, 2, 0, 2]),
        (2, "0,0,1", [0, 0, 1, 0, 0, 1]),
        (8, "7,6,5,4,3,2,1,0", [7, 6, 5, 4, 3, 2]),
    ];
    for (n, value, want) in cases {
        let p = Placement::new(n, Some(value)).expect("override placement");
        assert!(p.is_overridden(), "{value}");
        for (rank, d) in want.into_iter().enumerate() {
            assert_eq!(
                p.device_of(rank),
                d,
                "count {n} value '{value}' rank {rank}"
            );
        }
    }

    // Refused by name: every message starts `placement: ` and names what
    // it refuses.
    let rows: [(usize, Option<&str>, &[&str]); 9] = [
        (0, None, &["device count is 0"]),
        (0, Some("0"), &["device count is 0"]),
        (2, Some(""), &[DEVICES_ENV, "empty", "r mod 2"]),
        (2, Some("  "), &[DEVICES_ENV, "empty"]),
        (
            2,
            Some("0,,1"),
            &[DEVICES_ENV, "entry 2 is empty", "'0,,1'"],
        ),
        (
            2,
            Some("0,a"),
            &[DEVICES_ENV, "entry 2 'a'", "not a device ordinal"],
        ),
        (2, Some("-1"), &["entry 1 '-1'", "not a device ordinal"]),
        (2, Some("1.5"), &["entry 1 '1.5'", "not a device ordinal"]),
        (
            4,
            Some("0,4"),
            &[DEVICES_ENV, "entry 2 is device 4", "4 device(s)", "0..3"],
        ),
    ];
    for (n, value, names) in rows {
        let msg = refusal(n, value);
        assert!(msg.contains("placement: "), "count {n} {value:?}: {msg}");
        for name in names {
            assert!(
                msg.contains(name),
                "count {n} {value:?} must name '{name}': {msg}"
            );
        }
    }
}

/// One rank's evidence: the owned field, the performance, the votes, the
/// device ordinal, the stream handle, whether event tracking was off,
/// whether the rank's own `Gpu` owns its field and the main thread's does
/// not, the solve's start and end, and its outer refinement steps (0 in
/// the f64 build; SPEC-LIT §118.12 under `single`).
type RankRun = (
    Vec<Scalar>,
    DistPerformance,
    u64,
    usize,
    u64,
    bool,
    bool,
    Instant,
    Instant,
    usize,
);

/// The outer refinement steps of the workspace's last solve.
#[cfg(feature = "single")]
fn outer_steps(w: &SpmdWorkspace) -> usize {
    w.ir_outer_steps()
}

#[cfg(not(feature = "single"))]
fn outer_steps(_w: &SpmdWorkspace) -> usize {
    0
}

/// The SPMD PCG at `dec.n_parts` ranks, rank `r` opening
/// `Gpu::for_rank(placement, r)`.
#[allow(clippy::too_many_arguments)]
fn spmd_run(
    gpu0: &Gpu,
    placement: &Placement,
    m: &HostMesh,
    a: &HostLduMatrix,
    whole: &[Scalar],
    dec: &Decomposition,
    ctrl: &SolverControls,
) -> Result<Vec<Result<RankRun>>> {
    let np = dec.n_parts;
    ThreadComm::run(np, |comm| -> Result<RankRun> {
        let rank = comm.rank();
        let gpu = Gpu::for_rank(placement, rank)?;
        let sol = SolverKernels::new(&gpu)?;
        let mut halo = SpmdHalo::new(&gpu, &dec.parts[rank], np)?;
        let sys = DistSystem::split_one(&gpu, &mut halo, comm, m, dec, a)?;
        let mut w = SpmdWorkspace::new(&gpu, comm, &sys, DistReduce::Exact)?;
        w.set_flag_vote(true);
        let mut psi: DevBuf<Scalar> = gpu.upload(&dec.split_field(rank, whole)?)?;
        let owns = gpu.owns(&psi) && !gpu0.owns(&psi);
        comm.barrier()?;
        let t0 = Instant::now();
        let perf = spmd_pcg(&gpu, &sol, comm, &mut psi, &sys, &mut halo, &mut w, ctrl)?;
        gpu.sync()?;
        let t1 = Instant::now();
        let stream = gpu.stream().cu_stream() as usize as u64;
        Ok((
            gpu.download(&psi)?,
            perf,
            w.n_flag_votes(),
            gpu.ordinal(),
            stream,
            !gpu.event_tracking(),
            owns,
            t0,
            t1,
            outer_steps(&w),
        ))
    })
}

/// The in-process `dist_pcg` of `dec` on the main thread's card: the
/// per-part fields and the performance.
fn in_process(
    gpu0: &Gpu,
    sol0: &SolverKernels,
    m: &HostMesh,
    a: &HostLduMatrix,
    whole: &[Scalar],
    dec: &Decomposition,
    ctrl: &SolverControls,
) -> (Vec<Vec<Scalar>>, DistPerformance) {
    let mut ex = HaloExchange::new(gpu0, dec).expect("halo");
    let sys = DistSystem::split(gpu0, &mut ex, m, dec, a).expect("split");
    let mut w = DistWorkspace::new(gpu0, &sys, DistReduce::Exact).expect("ws");
    let mut psi: Vec<DevBuf<Scalar>> = (0..dec.n_parts)
        .map(|p| gpu0.upload(&dec.split_field(p, whole).expect("split field")))
        .collect::<Result<_>>()
        .expect("upload");
    let perf = dist_pcg(gpu0, sol0, &mut psi, &sys, &mut ex, &mut w, ctrl).expect("dist_pcg");
    gpu0.sync().expect("sync");
    let fields = psi
        .iter()
        .map(|b| gpu0.download(b).expect("download"))
        .collect();
    (fields, perf)
}

/// Holds one SPMD run to the in-process bits and to the one-part answer;
/// returns the rank results and cells compared.
#[allow(clippy::too_many_arguments)]
fn assert_bitwise(
    at: &str,
    dec: &Decomposition,
    got: &[Result<RankRun>],
    ref_fields: &[Vec<Scalar>],
    ref_perf: &DistPerformance,
    want: &[Scalar],
    want_iters: usize,
) -> (usize, usize) {
    let (mut ranks, mut cells) = (0usize, 0usize);
    for (r, part) in dec.parts.iter().enumerate() {
        let (g, perf, votes, .., outer) = got[r]
            .as_ref()
            .unwrap_or_else(|e| panic!("{at} rank {r}: {e}"));
        ranks += 1;
        for i in 0..part.mesh.n_cells {
            let gi = part.global_cell[i] as usize;
            assert_eq!(
                g[i].to_bits(),
                ref_fields[r][i].to_bits(),
                "vs the in-process part, {at} rank {r} owned {i}"
            );
            assert_eq!(
                g[i].to_bits(),
                want[gi].to_bits(),
                "vs the one-part answer, {at} rank {r} owned {i} (global {gi})"
            );
            cells += 1;
        }
        let at = format!("{at} rank {r}");
        assert_eq!(perf.n_iterations, ref_perf.n_iterations, "iterations, {at}");
        assert_eq!(perf.n_iterations, want_iters, "vs the one-part solve, {at}");
        assert_eq!(perf.n_exchanges, ref_perf.n_exchanges, "exchanges, {at}");
        assert_eq!(perf.n_reductions, ref_perf.n_reductions, "reductions, {at}");
        assert_eq!(perf.converged, ref_perf.converged, "converged, {at}");
        assert_eq!(
            perf.initial_residual.to_bits(),
            ref_perf.initial_residual.to_bits(),
            "initial residual bits, {at}"
        );
        assert_eq!(
            perf.final_residual.to_bits(),
            ref_perf.final_residual.to_bits(),
            "final residual bits, {at}"
        );
        // `n_iterations + 1` unrefined; a refined solve also votes before
        // each inner solve and at each outer verdict (SPEC-LIT §118.12).
        assert_eq!(*votes, (perf.n_iterations + 2 * *outer + 1) as u64, "votes, {at}");
    }
    (ranks, cells)
}

/// **The GPU gate.** A one-device placement puts all four ranks of a
/// `ThreadComm` world on device 0, each with its own `Gpu`. On both
/// `blockgen` boxes, with `Diagonal` and `None` and three partitioners: every
/// rank is on ordinal 0 with event tracking off, the four streams are four,
/// each rank's field is its own `Gpu`'s buffer and not the main thread's,
/// the four solves overlap in time, and every owned cell, counter and
/// residual is bit for bit the in-process `dist_pcg` and the one-part answer
/// (§138.5's gate).
#[test]
fn four_ranks_on_one_device_run_concurrently() {
    let gpu0 = match Gpu::new(0) {
        Ok(g) => g,
        Err(_) => return,
    };
    let sol0 = SolverKernels::new(&gpu0).expect("kernels");
    let placement = Placement::new(1, None).expect("one-device placement");
    let np = 4usize;
    for r in 0..np {
        assert_eq!(placement.device_of(r), 0, "rank {r}");
    }

    let (mut rank_results, mut cells, mut runs) = (0usize, 0usize, 0usize);
    for (nx, ny, nz) in [(5usize, 4usize, 3usize), (8, 6, 5)] {
        let (m, _) = build_case(CaseKind::Big, nx, ny, nz).expect("mesh");
        let a = poisson(&m);
        let whole: Vec<Scalar> = (0..m.n_cells).map(field).collect();
        for precon in [Preconditioner::Diagonal, Preconditioner::None] {
            let ctrl = controls(precon);
            let one = Decomposition::build(&m, 1, &PartitionMethod::Linear).expect("cut");
            let (one_part, wp) = in_process(&gpu0, &sol0, &m, &a, &whole, &one, &ctrl);
            let want = &one_part[0];
            assert!(
                wp.converged,
                "{nx}x{ny}x{nz} {precon:?}: the reference must converge"
            );
            assert!(
                wp.n_iterations > 3,
                "{nx}x{ny}x{nz} {precon:?}: too few iterations"
            );

            for (name, dec) in cuts(&m, np) {
                let at = format!("{nx}x{ny}x{nz} {precon:?} P = {np} {name}");
                assert!(
                    dec.parts.iter().all(|p| p.n_halo > 0),
                    "{at}: a part has no halo"
                );
                let (ref_fields, ref_perf) = in_process(&gpu0, &sol0, &m, &a, &whole, &dec, &ctrl);
                let got = spmd_run(&gpu0, &placement, &m, &a, &whole, &dec, &ctrl)
                    .unwrap_or_else(|e| panic!("{at}: {e}"));
                assert_eq!(got.len(), np, "{at}: rank results");
                let runs_ok: Vec<&RankRun> = got
                    .iter()
                    .enumerate()
                    .map(|(r, g)| g.as_ref().unwrap_or_else(|e| panic!("{at} rank {r}: {e}")))
                    .collect();

                let mut streams: Vec<u64> = Vec::new();
                for (r, run) in runs_ok.iter().enumerate() {
                    let (_, _, _, ordinal, stream, tracking_off, owns, _, _, _) = run;
                    assert_eq!(*ordinal, 0, "{at} rank {r}: device");
                    assert!(*tracking_off, "{at} rank {r}: event tracking is on");
                    assert!(
                        *owns,
                        "{at} rank {r}: the field is not its own Gpu's buffer"
                    );
                    streams.push(*stream);
                }
                streams.push(gpu0.stream().cu_stream() as usize as u64);
                streams.sort_unstable();
                streams.dedup();
                assert_eq!(
                    streams.len(),
                    np + 1,
                    "{at}: the ranks and main share a stream"
                );

                let last_start = runs_ok.iter().map(|r| r.7).max().expect("starts");
                let first_end = runs_ok.iter().map(|r| r.8).min().expect("ends");
                assert!(
                    last_start < first_end,
                    "{at}: the four solves did not overlap"
                );

                let (rr, cc) = assert_bitwise(
                    &at,
                    &dec,
                    &got,
                    &ref_fields,
                    &ref_perf,
                    want,
                    wp.n_iterations,
                );
                rank_results += rr;
                cells += cc;
                runs += 1;
            }
        }
    }
    assert_eq!(runs, 12, "four-rank worlds run");
    assert_eq!(rank_results, 48, "rank results compared");
    assert_eq!(cells, 1800, "owned cells compared");
}

/// The two-device placement: ranks 0 and 1 on devices 0 and 1, then the
/// override `1,0` swapping them; both against the in-process solve on
/// device 0, bit for bit. **Not runnable on this machine** (one GPU): run it
/// with `--ignored` on a machine with two devices of the same architecture,
/// since the bits are the same code's only on the same SASS.
#[test]
#[ignore = "needs 2 GPUs"]
fn peer_placement_two_devices() {
    let n = Gpu::device_count().expect("device count");
    assert!(n >= 2, "needs 2 GPUs: the driver sees {n}");
    let gpu0 = Gpu::new(0).expect("device 0");
    let sol0 = SolverKernels::new(&gpu0).expect("kernels");
    let (m, _) = build_case(CaseKind::Big, 8, 6, 5).expect("mesh");
    let a = poisson(&m);
    let whole: Vec<Scalar> = (0..m.n_cells).map(field).collect();
    let ctrl = controls(Preconditioner::Diagonal);
    let one = Decomposition::build(&m, 1, &PartitionMethod::Linear).expect("cut");
    let (one_part, wp) = in_process(&gpu0, &sol0, &m, &a, &whole, &one, &ctrl);
    let want = &one_part[0];

    let mut rank_results = 0usize;
    for (label, value, devices) in [("default", None, [0usize, 1]), ("1,0", Some("1,0"), [1, 0])] {
        let placement = Placement::new(n, value).expect("placement");
        for (name, dec) in cuts(&m, 2) {
            let at = format!("{label} P = 2 {name}");
            let (ref_fields, ref_perf) = in_process(&gpu0, &sol0, &m, &a, &whole, &dec, &ctrl);
            let got = spmd_run(&gpu0, &placement, &m, &a, &whole, &dec, &ctrl)
                .unwrap_or_else(|e| panic!("{at}: {e}"));
            for (r, g) in got.iter().enumerate() {
                let run = g.as_ref().unwrap_or_else(|e| panic!("{at} rank {r}: {e}"));
                assert_eq!(run.3, devices[r], "{at} rank {r}: device");
            }
            let (rr, _) = assert_bitwise(
                &at,
                &dec,
                &got,
                &ref_fields,
                &ref_perf,
                want,
                wp.n_iterations,
            );
            rank_results += rr;
        }
    }
    assert_eq!(rank_results, 12, "rank results compared");
}

/// `Gpu::for_rank` refuses, by name, a device the driver does not have: a
/// placement built for one device more than the driver sees, with the
/// override naming that extra device.
#[test]
fn for_rank_refuses_a_device_the_driver_lacks() {
    let Ok(n) = Gpu::device_count() else { return };
    if n == 0 {
        return;
    }
    let extra = n.to_string();
    let placement = Placement::new(n + 1, Some(extra.as_str())).expect("placement");
    let msg = match Gpu::for_rank(&placement, 3) {
        Ok(g) => panic!("rank 3 opened device {} of {n}", g.ordinal()),
        Err(e) => e.to_string(),
    };
    for name in [
        "placement: ",
        "rank 3",
        &format!("device {n}"),
        &format!("driver sees {n} device(s)"),
    ] {
        assert!(msg.contains(name), "must name '{name}': {msg}");
    }
}
