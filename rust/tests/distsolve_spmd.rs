// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.
// Provenance: see PROVENANCE.md. No GPL-licensed source was consulted.

//! SPEC-LIT §138.5's and §138.6's gates - the SPMD PCG and PBiCGStab, and
//! the block-local DIC and DILU on a rank, in BOTH builds.
//!
//! An integration test because the library's own test target does not
//! compile under `--features single`; an integration test links the normal
//! (non-test) library, which does. Every rank is a [`ThreadComm`] thread
//! with its own [`Gpu`] on device 0 and its own [`SolverKernels`], and the
//! references are the in-process [`dist_pcg`] and [`dist_pbicgstab`] on the
//! main thread's card: a solve run over `P` ranks must produce the bits the
//! one-process, `P`-part run produces, and - the reduction being exact and
//! the preconditioner one a cut cannot change - the bits the one-part run
//! produces.
//!
//! Class D of docs/17 §5.1: exact bytes and exact counts. Every value
//! comparison is `to_bits()`, and the counts of rank results and owned
//! cells compared are asserted at the end, so a silent narrowing of the
//! gate cannot pass.
//!
//! Provenance: ORIGINAL - the SPMD form under test is this project's own
//! design (SPEC-LIT §138.5, PAR-04a; §138.6, PAR-04b) on §73's kernels and
//! recurrences. No external source. No GPL-licensed source was consulted.

use ofgpu::blockgen::{build_case, CaseKind};
use ofgpu::comm::{Comm, ThreadComm};
use ofgpu::decompose::{Decomposition, PartitionMethod};
use ofgpu::device::{DevBuf, Gpu};
use ofgpu::distsolve::{
    dist_pcg, dist_pbicgstab, spmd_pcg, spmd_pbicgstab, spmd_solve, DistPerformance, DistReduce,
    DistSystem, DistWorkspace, SpmdWorkspace,
};
use ofgpu::halo::{HaloExchange, SpmdHalo};
use ofgpu::ldu::HostLduMatrix;
use ofgpu::mesh::HostMesh;
use ofgpu::solver::{LinearSolverKind, Preconditioner, SolverControls, SolverKernels};
use ofgpu::{Label, Result, Scalar};

/// The solver tolerance: `1e-12` in the f64 build, `1e-5` in the f32 one. A
/// solver control and NOT a gate tolerance - every comparison below is exact.
#[cfg(feature = "single")]
const TOL: Scalar = 1e-5;
#[cfg(not(feature = "single"))]
const TOL: Scalar = 1e-12;

/// `1/dt`, which makes the operator non-singular on an all-Neumann mesh -
/// `src/distsolve/tests.rs`'s fixture, unchanged.
const RDT: Scalar = 1.5;

/// A deterministic initial field of the GLOBAL cell id, so distributing it
/// is a permutation and never a computation.
fn field(c: usize) -> Scalar {
    0.25 + 0.0625 * ((c * 37) % 23) as Scalar
}

/// A deterministic Dirichlet value per boundary cell, likewise.
fn wall(c: usize) -> Scalar {
    0.5 + 0.125 * ((c * 11) % 7) as Scalar
}

/// The SPD Poisson operator plus a `ddt`, from the mesh's own metrics,
/// folded on the host - `src/distsolve/tests.rs`'s fixture, with `as
/// Scalar` where a mesh quantity is not `Scalar`.
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

/// Controls with everything that could hide a difference turned off.
fn controls(
    solver: LinearSolverKind,
    precon: Preconditioner,
    fixed: Option<usize>,
) -> SolverControls {
    SolverControls {
        solver,
        precon,
        tolerance: TOL,
        rel_tol: 0.0,
        max_iter: fixed.map_or(400, |k| k as Label),
        min_iter: 0,
        check_interval: 1,
        fixed_iters: fixed.is_some(),
        report_residuals: true,
        gamg: ofgpu::io::case::GamgControls::default(),
    }
}

/// The round robin: cell `c` to part `c % P`, the worst partition possible.
fn round_robin(n: usize, p: usize) -> PartitionMethod {
    PartitionMethod::Explicit((0..n).map(|c| (c % p) as Label).collect())
}

/// The ten decompositions of B1: `P = 1` with the linear cut alone, then
/// three partitioners at `P = 2, 3, 4`.
fn cuts(m: &HostMesh, np: usize) -> Vec<(String, Decomposition)> {
    if np == 1 {
        return vec![(
            "linear".to_string(),
            Decomposition::build(m, 1, &PartitionMethod::Linear).expect("cut"),
        )];
    }
    [
        ("hilbert", PartitionMethod::Hilbert),
        ("linear", PartitionMethod::Linear),
        ("roundrobin", round_robin(m.n_cells, np)),
    ]
    .into_iter()
    .map(|(name, method)| {
        (
            name.to_string(),
            Decomposition::build(m, np, &method).expect("cut"),
        )
    })
    .collect()
}

/// An SPMD solver as a plain fn pointer, so `match`/`as` arms unify - the
/// two fn items have distinct types.
type SpmdSolve = fn(
    &Gpu,
    &SolverKernels,
    &dyn Comm,
    &mut DevBuf<Scalar>,
    &DistSystem,
    &mut SpmdHalo,
    &mut SpmdWorkspace,
    &SolverControls,
) -> Result<DistPerformance>;

/// One decomposition, both ways: the in-process solver picked by
/// `in_process` over every part on the main thread's card, then the same
/// solve as one rank per part through `spmd`. Colours both sides first when
/// the controls ask for the block-local factorisation. Returns the per-part
/// fields, the performance and the colour counts of the reference and, per
/// rank, the owned field, the performance, the vote count, the colour count
/// and `partition_invariant`'s answer for the controls.
#[allow(clippy::too_many_arguments)]
fn both_ways<I, S>(
    gpu0: &Gpu,
    sol0: &SolverKernels,
    m: &HostMesh,
    a: &HostLduMatrix,
    whole: &[Scalar],
    dec: &Decomposition,
    ctrl: &SolverControls,
    reduce: DistReduce,
    in_process: I,
    spmd: S,
) -> Result<(
    Vec<Vec<Scalar>>,
    DistPerformance,
    Vec<usize>,
    Vec<Result<(Vec<Scalar>, DistPerformance, u64, usize, bool, usize)>>,
)>
where
    I: Fn(
        &Gpu,
        &SolverKernels,
        &mut [DevBuf<Scalar>],
        &DistSystem,
        &mut HaloExchange,
        &mut DistWorkspace,
        &SolverControls,
    ) -> Result<DistPerformance>,
    S: Fn(
            &Gpu,
            &SolverKernels,
            &dyn Comm,
            &mut DevBuf<Scalar>,
            &DistSystem,
            &mut SpmdHalo,
            &mut SpmdWorkspace,
            &SolverControls,
        ) -> Result<DistPerformance>
        + Sync,
{
    let np = dec.n_parts;
    let mut ex = HaloExchange::new(gpu0, dec)?;
    let sys = DistSystem::split(gpu0, &mut ex, m, dec, a)?;
    let mut w = DistWorkspace::new(gpu0, &sys, reduce)?;
    if matches!(ctrl.precon, Preconditioner::Dic | Preconditioner::Dilu) {
        w.colour(gpu0, &sys)?;
    }
    let mut psi: Vec<DevBuf<Scalar>> = (0..np)
        .map(|p| gpu0.upload(&dec.split_field(p, whole)?))
        .collect::<Result<_>>()?;
    let perf = in_process(gpu0, sol0, &mut psi, &sys, &mut ex, &mut w, ctrl)?;
    gpu0.sync()?;
    let colours = w.colours();
    let fields: Vec<Vec<Scalar>> = psi
        .iter()
        .map(|buf| gpu0.download(buf))
        .collect::<Result<_>>()?;
    drop((psi, w, ex, sys));

    let got = ThreadComm::run(
        np,
        |comm| -> Result<(Vec<Scalar>, DistPerformance, u64, usize, bool, usize)> {
        let gpu = Gpu::new(0)?;
        let sol = SolverKernels::new(&gpu)?;
        let rank = comm.rank();
        let mut halo = SpmdHalo::new(&gpu, &dec.parts[rank], np)?;
        let sys = DistSystem::split_one(&gpu, &mut halo, comm, m, dec, a)?;
        let mut w = SpmdWorkspace::new(&gpu, comm, &sys, reduce)?;
        w.set_flag_vote(true);
        if matches!(ctrl.precon, Preconditioner::Dic | Preconditioner::Dilu) {
            w.colour(&gpu, &sys)?;
        }
        let mut psi = gpu.upload(&dec.split_field(rank, whole)?)?;
        let perf = spmd(&gpu, &sol, comm, &mut psi, &sys, &mut halo, &mut w, ctrl)?;
        gpu.sync()?;
        let pinv = w.partition_invariant(ctrl.precon);
        Ok((gpu.download(&psi)?, perf, w.n_flag_votes(), w.colours(), pinv, outer_steps(&w)))
        },
    )?;
    Ok((fields, perf, colours, got))
}

/// **The gate, in both builds.** Two `blockgen` boxes, `Diagonal` and
/// `None`, `P = 1..4`, three partitioners, the exact reduction: on every
/// rank every owned cell is bit for bit the in-process part's answer AND
/// the one-part answer, the performance counters and both residuals' bits
/// equal the in-process solve's, and the flag vote counts `n_iterations +
/// 1` reads and finds nothing to report. Then, on the small box with
/// `Diagonal` at `P = 2..4`, the same against the in-process GATHERED
/// solve - run-invariant only, so the reference is the same decomposition,
/// never the one-part answer.
#[test]
fn an_spmd_pcg_is_the_in_process_pcg_in_both_builds() {
    let gpu0 = match Gpu::new(0) {
        Ok(g) => g,
        Err(_) => return,
    };
    let sol0 = SolverKernels::new(&gpu0).expect("kernels");

    let mut rank_results = 0usize;
    let mut cells = 0usize;

    // ---- the exact leg ---------------------------------------------------
    for (nx, ny, nz) in [(5usize, 4usize, 3usize), (8, 6, 5)] {
        let (m, _) = build_case(CaseKind::Big, nx, ny, nz).expect("mesh");
        let a = poisson(&m);
        let whole: Vec<Scalar> = (0..m.n_cells).map(field).collect();

        for precon in [Preconditioner::Diagonal, Preconditioner::None] {
            let ctrl = controls(LinearSolverKind::PCG, precon, None);

            // Reference ONE: the in-process solve at one part.
            let one = Decomposition::build(&m, 1, &PartitionMethod::Linear).expect("cut");
            let (want, wp) = {
                let mut ex = HaloExchange::new(&gpu0, &one).expect("halo");
                let sys = DistSystem::split(&gpu0, &mut ex, &m, &one, &a).expect("split");
                let mut w = DistWorkspace::new(&gpu0, &sys, DistReduce::Exact).expect("ws");
                let mut psi = vec![gpu0.upload(&whole).expect("upload")];
                let perf = dist_pcg(&gpu0, &sol0, &mut psi, &sys, &mut ex, &mut w, &ctrl)
                    .expect("P = 1");
                gpu0.sync().expect("sync");
                (gpu0.download(&psi[0]).expect("download"), perf)
            };
            assert!(
                wp.converged,
                "{nx}x{ny}x{nz} {precon:?}: the reference solve must converge"
            );
            assert!(
                wp.n_iterations > 3,
                "{nx}x{ny}x{nz} {precon:?}: a one-iteration gate proves nothing"
            );

            for np in 1..=4usize {
                for (name, dec) in cuts(&m, np) {
                    if np > 1 {
                        assert!(dec.n_cut_faces > 0, "P = {np} {name} cut nothing");
                        assert!(
                            dec.parts.iter().all(|p| p.n_halo > 0),
                            "P = {np} {name} left a part without a halo"
                        );
                    }
                    let (ref_fields, ref_perf, _, got) = both_ways(
                        &gpu0,
                        &sol0,
                        &m,
                        &a,
                        &whole,
                        &dec,
                        &ctrl,
                        DistReduce::Exact,
                        dist_pcg,
                        spmd_pcg,
                    )
                    .unwrap_or_else(|e| panic!("{nx}x{ny}x{nz} {precon:?} P = {np} {name}: {e}"));
                    for (r, part) in dec.parts.iter().enumerate() {
                        let (g, perf, votes, _, _, outer) = got[r]
                            .as_ref()
                            .unwrap_or_else(|e| {
                                panic!("{nx}x{ny}x{nz} {precon:?} P = {np} {name} rank {r}: {e}")
                            });
                        rank_results += 1;
                        cells += part.mesh.n_cells;
                        for i in 0..part.mesh.n_cells {
                            let gi = part.global_cell[i] as usize;
                            let at = format!(
                                "{nx}x{ny}x{nz} {precon:?} P = {np} {name} \
                                 rank {r} owned {i} (global {gi})"
                            );
                            assert_eq!(
                                g[i].to_bits(),
                                ref_fields[r][i].to_bits(),
                                "vs the in-process part, {at}"
                            );
                            assert_eq!(
                                g[i].to_bits(),
                                want[gi].to_bits(),
                                "vs the one-part answer, {at}"
                            );
                        }
                        let at =
                            format!("{nx}x{ny}x{nz} {precon:?} P = {np} {name} rank {r}");
                        assert_eq!(perf.n_iterations, ref_perf.n_iterations, "iterations, {at}");
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
                        assert_eq!(perf.n_iterations, wp.n_iterations, "vs Reference ONE, {at}");
                        assert_eq!(*votes, expected_votes(perf, *outer), "votes, {at}");
                    }
                }
            }
        }
    }

    // ---- the gathered leg: run-invariant only, so the reference is the
    // in-process GATHERED solve of the SAME decomposition -----------------
    {
        let (m, _) = build_case(CaseKind::Big, 5, 4, 3).expect("mesh");
        let a = poisson(&m);
        let whole: Vec<Scalar> = (0..m.n_cells).map(field).collect();
        let ctrl = controls(LinearSolverKind::PCG, Preconditioner::Diagonal, None);
        for np in 2..=4usize {
            for (name, dec) in cuts(&m, np) {
                let (ref_fields, ref_perf, _, got) = both_ways(
                    &gpu0,
                    &sol0,
                    &m,
                    &a,
                    &whole,
                    &dec,
                    &ctrl,
                    DistReduce::Gathered,
                    dist_pcg,
                    spmd_pcg,
                )
                .unwrap_or_else(|e| panic!("gathered P = {np} {name}: {e}"));
                for (r, part) in dec.parts.iter().enumerate() {
                    let (g, perf, votes, _, _, outer) = got[r]
                        .as_ref()
                        .unwrap_or_else(|e| panic!("gathered P = {np} {name} rank {r}: {e}"));
                    rank_results += 1;
                    cells += part.mesh.n_cells;
                    for i in 0..part.mesh.n_cells {
                        assert_eq!(
                            g[i].to_bits(),
                            ref_fields[r][i].to_bits(),
                            "gathered vs the in-process gathered part, \
                             P = {np} {name} rank {r} owned {i}"
                        );
                    }
                    let at = format!("gathered P = {np} {name} rank {r}");
                    assert_eq!(perf.n_iterations, ref_perf.n_iterations, "iterations, {at}");
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
                    assert_eq!(*votes, expected_votes(perf, *outer), "votes, {at}");
                }
            }
        }
    }

    assert_eq!(rank_results, 139, "rank results compared");
    assert_eq!(
        cells, 6540,
        "owned cells compared against the in-process part"
    );
}

/// The counts of SPEC-LIT §138.5: at `k` fixed iterations a rank makes
/// `k + 3` exchanges and `2k + 5` reductions, the exact reduction costs two
/// allgathers per reduction and the gathered one one, `new`'s included, and
/// a `fixed_iters` solve reads no flag and takes no vote.
#[test]
fn an_spmd_pcg_counts_its_exchanges_and_collectives() {
    if Gpu::new(0).is_err() {
        return;
    }
    let (m, _) = build_case(CaseKind::Big, 5, 4, 3).expect("mesh");
    let a = poisson(&m);
    let whole: Vec<Scalar> = (0..m.n_cells).map(field).collect();
    let dec = Decomposition::build(&m, 3, &PartitionMethod::Hilbert).expect("cut");
    let ctrl = controls(LinearSolverKind::PCG, Preconditioner::Diagonal, Some(5));

    for (reduce, want_allgathers) in [(DistReduce::Exact, 31u64), (DistReduce::Gathered, 16)] {
        let got = ThreadComm::run(3, |comm| -> Result<(DistPerformance, u64, u64)> {
            let gpu = Gpu::new(0)?;
            let sol = SolverKernels::new(&gpu)?;
            let rank = comm.rank();
            let mut halo = SpmdHalo::new(&gpu, &dec.parts[rank], 3)?;
            let sys = DistSystem::split_one(&gpu, &mut halo, comm, &m, &dec, &a)?;
            let mut w = SpmdWorkspace::new(&gpu, comm, &sys, reduce)?;
            w.set_flag_vote(true);
            let mut psi = gpu.upload(&dec.split_field(rank, &whole)?)?;
            let perf = spmd_pcg(&gpu, &sol, comm, &mut psi, &sys, &mut halo, &mut w, &ctrl)?;
            gpu.sync()?;
            Ok((perf, w.n_flag_votes(), w.n_allgathers()))
        })
        .expect("spmd run");
        for r in 0..3 {
            let (perf, votes, allgathers) = got[r]
                .as_ref()
                .unwrap_or_else(|e| panic!("rank {r} failed: {e}"));
            assert_eq!(perf.n_iterations, 5, "rank {r} {reduce:?}");
            assert_eq!(perf.n_exchanges, 8, "rank {r} {reduce:?}");
            assert_eq!(perf.n_reductions, 15, "rank {r} {reduce:?}");
            assert_eq!(*votes, 0, "a fixed_iters solve reads no flag, rank {r}");
            assert_eq!(*allgathers, want_allgathers, "rank {r} {reduce:?}");
        }
    }
}

/// One rank given a tolerance it meets at once, the other solving on: the
/// flags disagree at the first read, and BOTH ranks are refused with both
/// votes in rank order. No rank is left waiting in a collective. Both
/// solvers take the vote, and each refusal names the solver that took it.
#[test]
fn an_spmd_flag_vote_names_a_disagreement() {
    if Gpu::new(0).is_err() {
        return;
    }
    let (m, _) = build_case(CaseKind::Big, 5, 4, 3).expect("mesh");
    let a = poisson(&m);
    let whole: Vec<Scalar> = (0..m.n_cells).map(field).collect();
    let dec = Decomposition::build(&m, 2, &PartitionMethod::Linear).expect("cut");

    for (who, kind, solve_by) in [
        (
            "spmd_pcg: ",
            LinearSolverKind::PCG,
            spmd_pcg as SpmdSolve,
        ),
        (
            "spmd_pbicgstab: ",
            LinearSolverKind::PBiCGStab,
            spmd_pbicgstab as SpmdSolve,
        ),
    ] {
        let got = ThreadComm::run(2, |comm| -> Result<String> {
            let gpu = Gpu::new(0)?;
            let sol = SolverKernels::new(&gpu)?;
            let rank = comm.rank();
            let mut halo = SpmdHalo::new(&gpu, &dec.parts[rank], 2)?;
            let sys = DistSystem::split_one(&gpu, &mut halo, comm, &m, &dec, &a)?;
            let mut w = SpmdWorkspace::new(&gpu, comm, &sys, DistReduce::Exact)?;
            w.set_flag_vote(true);
            let mut ctrl = controls(kind, Preconditioner::Diagonal, None);
            if rank == 1 {
                ctrl.tolerance = 1e20;
            }
            let mut psi = gpu.upload(&dec.split_field(rank, &whole)?)?;
            match solve_by(&gpu, &sol, comm, &mut psi, &sys, &mut halo, &mut w, &ctrl) {
                Ok(_) => panic!("rank {rank}: the vote should have refused this solve"),
                Err(e) => Ok(e.to_string()),
            }
        })
        .expect("spmd run");
        for r in 0..2 {
            let msg = got[r]
                .as_ref()
                .unwrap_or_else(|e| panic!("{who}rank {r} failed: {e}"));
            assert!(msg.starts_with(who), "rank {r}: {msg}");
            assert!(msg.contains("flag"), "rank {r}: {msg}");
            assert!(msg.contains("[0, 1]"), "rank {r}: {msg}");
        }
    }
}

/// The refusal rows of SPEC-LIT §138.5's table, in order, every row with a
/// block here. No multi-rank block lets ONE rank fail alone: every rank is
/// refused at the same check, BEFORE any collective, because a refusal only
/// one rank sees would leave the other ranks waiting in a collective.
#[test]
fn an_spmd_pcg_refuses_by_name() {
    let gpu = match Gpu::new(0) {
        Ok(g) => g,
        Err(_) => return,
    };
    let sol = SolverKernels::new(&gpu).expect("kernels");
    let (m5, _) = build_case(CaseKind::Big, 5, 4, 3).expect("mesh");
    let (m8, _) = build_case(CaseKind::Big, 8, 6, 5).expect("mesh");
    let a5 = poisson(&m5);
    let a8 = poisson(&m8);
    let d1 = Decomposition::build(&m5, 1, &PartitionMethod::Linear).expect("cut");
    let d2 = Decomposition::build(&m5, 2, &PartitionMethod::Linear).expect("cut");
    let e1 = Decomposition::build(&m8, 1, &PartitionMethod::Linear).expect("cut");

    // split_one, row 1: a 1-rank world for a 2-part decomposition.
    {
        let comm1 = &ThreadComm::world(1).expect("world")[0];
        let mut halo = SpmdHalo::new(&gpu, &d2.parts[0], 2).expect("plan");
        let msg = DistSystem::split_one(&gpu, &mut halo, comm1, &m5, &d2, &a5)
            .err()
            .expect("split_one size must refuse")
            .to_string();
        assert!(msg.starts_with("distsolve: "), "{msg}");
        assert!(msg.contains("1 ranks"), "{msg}");
        assert!(msg.contains("2 parts"), "{msg}");
    }

    // split_one, rows 2-3: the halo plan is the OTHER rank's part. Both
    // ranks refuse at the same check, before any exchange.
    {
        let got = ThreadComm::run(2, |comm| -> Result<String> {
            let gpu = Gpu::new(0)?;
            let rank = comm.rank();
            let mut halo = SpmdHalo::new(&gpu, &d2.parts[1 - rank], 2)?;
            match DistSystem::split_one(&gpu, &mut halo, comm, &m5, &d2, &a5) {
                Ok(_) => panic!("rank {rank}: split_one should have refused"),
                Err(e) => Ok(e.to_string()),
            }
        })
        .expect("spmd run");
        for r in 0..2 {
            let msg = got[r]
                .as_ref()
                .unwrap_or_else(|e| panic!("rank {r} failed: {e}"));
            assert!(msg.starts_with("distsolve: "), "rank {r}: {msg}");
            assert!(msg.contains(&format!("rank {r}")), "rank {r}: {msg}");
            assert!(
                msg.contains(&format!("part {}", 1 - r)),
                "rank {r}: {msg}"
            );
        }
    }

    // SpmdWorkspace::new on a system from DistSystem::split. The refusal
    // runs BEFORE the counts allgather, so no rank could be stranded.
    {
        let comm1 = &ThreadComm::world(1).expect("world")[0];
        let mut ex = HaloExchange::new(&gpu, &d1).expect("halo");
        let sys = DistSystem::split(&gpu, &mut ex, &m5, &d1, &a5).expect("split");
        let msg = SpmdWorkspace::new(&gpu, comm1, &sys, DistReduce::Exact)
            .err()
            .expect("the workspace must refuse a split system")
            .to_string();
        assert!(msg.starts_with("distsolve: "), "{msg}");
        assert!(msg.contains("DistSystem::split_one"), "{msg}");
    }

    // SpmdWorkspace::new, rank row: the system's part is 0 (split_one in a
    // 1-rank world) and the handle is rank 1 of a 2-rank world. The refusal
    // runs BEFORE the counts allgather, so rank 1 of that world is never
    // touched - a single thread, nobody stranded.
    {
        let comm1 = &ThreadComm::world(1).expect("world")[0];
        let mut halo = SpmdHalo::new(&gpu, &d1.parts[0], 1).expect("plan");
        let sys = DistSystem::split_one(&gpu, &mut halo, comm1, &m5, &d1, &a5).expect("split_one");
        let w2 = ThreadComm::world(2).expect("world");
        let msg = SpmdWorkspace::new(&gpu, &w2[1], &sys, DistReduce::Exact)
            .err()
            .expect("the workspace must refuse another rank")
            .to_string();
        assert!(msg.starts_with("distsolve: "), "{msg}");
        assert!(msg.contains("rank 1"), "{msg}");
        assert!(msg.contains("part 0"), "{msg}");
    }

    // DistWorkspace::new on a system from split_one.
    {
        let comm1 = &ThreadComm::world(1).expect("world")[0];
        let mut halo = SpmdHalo::new(&gpu, &d1.parts[0], 1).expect("plan");
        let sys = DistSystem::split_one(&gpu, &mut halo, comm1, &m5, &d1, &a5).expect("split_one");
        let msg = DistWorkspace::new(&gpu, &sys, DistReduce::Exact)
            .err()
            .expect("DistWorkspace::new must refuse a split_one system")
            .to_string();
        assert!(msg.starts_with("distsolve: "), "{msg}");
        assert!(msg.contains("SpmdWorkspace"), "{msg}");
    }

    // spmd_pcg, row 1: inside a CUDA-graph capture. The same call outside
    // the capture is `Ok` - the refusal left nothing behind.
    {
        let comm1 = &ThreadComm::world(1).expect("world")[0];
        let mut halo = SpmdHalo::new(&gpu, &d1.parts[0], 1).expect("plan");
        let sys = DistSystem::split_one(&gpu, &mut halo, comm1, &m5, &d1, &a5).expect("split_one");
        let mut w = SpmdWorkspace::new(&gpu, comm1, &sys, DistReduce::Exact).expect("ws");
        w.set_flag_vote(true);
        let whole: Vec<Scalar> = (0..m5.n_cells).map(field).collect();
        let mut psi = gpu.upload(&whole).expect("upload");
        let ctrl = controls(LinearSolverKind::PCG, Preconditioner::Diagonal, None);
        let msg = gpu
            .capture(|_| {
                spmd_pcg(&gpu, &sol, comm1, &mut psi, &sys, &mut halo, &mut w, &ctrl)
                    .map(|_| ())
            })
            .err()
            .expect("row 1 must refuse")
            .to_string();
        assert!(msg.starts_with("spmd_pcg: "), "{msg}");
        assert!(msg.contains("capture"), "{msg}");
        spmd_pcg(&gpu, &sol, comm1, &mut psi, &sys, &mut halo, &mut w, &ctrl)
            .expect("the same call outside the capture is Ok");
    }

    // spmd_pcg, row 2: a `split` system with a workspace from split_one.
    {
        let comm1 = &ThreadComm::world(1).expect("world")[0];
        let mut ex = HaloExchange::new(&gpu, &d1).expect("halo");
        let sys_split = DistSystem::split(&gpu, &mut ex, &m5, &d1, &a5).expect("split");
        let mut halo_one = SpmdHalo::new(&gpu, &d1.parts[0], 1).expect("plan");
        let sys_one =
            DistSystem::split_one(&gpu, &mut halo_one, comm1, &m5, &d1, &a5).expect("split_one");
        let mut w = SpmdWorkspace::new(&gpu, comm1, &sys_one, DistReduce::Exact).expect("ws");
        let whole: Vec<Scalar> = (0..m5.n_cells).map(field).collect();
        let mut psi = gpu.upload(&whole).expect("upload");
        let ctrl = controls(LinearSolverKind::PCG, Preconditioner::Diagonal, None);
        let msg =
            spmd_pcg(&gpu, &sol, comm1, &mut psi, &sys_split, &mut halo_one, &mut w, &ctrl)
                .err()
                .expect("row 2 must refuse")
                .to_string();
        assert!(msg.starts_with("spmd_pcg: "), "{msg}");
        assert!(msg.contains("DistSystem::split_one"), "{msg}");
    }

    // spmd_pcg, row 3: everything built in a 1-rank world, called with rank
    // 0 of a 2-rank world. The refusal is before anything is sent, so the
    // second rank of that world is never touched.
    {
        let comm1 = &ThreadComm::world(1).expect("world")[0];
        let mut halo = SpmdHalo::new(&gpu, &d1.parts[0], 1).expect("plan");
        let sys = DistSystem::split_one(&gpu, &mut halo, comm1, &m5, &d1, &a5).expect("split_one");
        let mut w = SpmdWorkspace::new(&gpu, comm1, &sys, DistReduce::Exact).expect("ws");
        let whole: Vec<Scalar> = (0..m5.n_cells).map(field).collect();
        let mut psi = gpu.upload(&whole).expect("upload");
        let ctrl = controls(LinearSolverKind::PCG, Preconditioner::Diagonal, None);
        let w2 = ThreadComm::world(2).expect("world");
        let msg = spmd_pcg(&gpu, &sol, &w2[0], &mut psi, &sys, &mut halo, &mut w, &ctrl)
            .err()
            .expect("row 3 must refuse")
            .to_string();
        assert!(msg.starts_with("spmd_pcg: "), "{msg}");
        assert!(msg.contains("2 ranks"), "{msg}");
        assert!(msg.contains("1 ranks"), "{msg}");
    }

    // spmd_pcg, row 4: the right world size, the WRONG handle - rank r calls
    // with handle `1 - r` of a SECOND world. ThreadComm is Send, not Sync,
    // so the second world's handles sit in a mutex and a rank TAKES its
    // borrowed one out. Both ranks are refused at the same check, before
    // anything is sent, so no rank waits.
    {
        let second = std::sync::Mutex::new(
            ThreadComm::world(2)
                .expect("world")
                .into_iter()
                .map(Some)
                .collect::<Vec<_>>(),
        );
        let got = ThreadComm::run(2, |comm| -> Result<String> {
            let gpu = Gpu::new(0)?;
            let sol = SolverKernels::new(&gpu)?;
            let rank = comm.rank();
            let mut halo = SpmdHalo::new(&gpu, &d2.parts[rank], 2)?;
            let sys = DistSystem::split_one(&gpu, &mut halo, comm, &m5, &d2, &a5)?;
            let mut w = SpmdWorkspace::new(&gpu, comm, &sys, DistReduce::Exact)?;
            let mut psi = gpu.zeros::<Scalar>(sys.local()[0])?;
            let other = {
                let mut handles = second.lock().expect("lock");
                handles[1 - rank].take().expect("handle")
            };
            let ctrl = controls(LinearSolverKind::PCG, Preconditioner::Diagonal, None);
            match spmd_pcg(&gpu, &sol, &other, &mut psi, &sys, &mut halo, &mut w, &ctrl) {
                Ok(_) => panic!("rank {rank}: row 4 should have refused"),
                Err(e) => Ok(e.to_string()),
            }
        })
        .expect("spmd run");
        for r in 0..2 {
            let msg = got[r]
                .as_ref()
                .unwrap_or_else(|e| panic!("rank {r} failed: {e}"));
            assert!(msg.starts_with("spmd_pcg: "), "rank {r}: {msg}");
            assert!(
                msg.contains(&format!("rank {}", 1 - r)),
                "rank {r}: {msg}"
            );
            assert!(msg.contains(&format!("part {r}")), "rank {r}: {msg}");
        }
    }

    // spmd_pcg, row 5: the workspace was built for another system - the
    // 5x4x3 one, against a system from the 8x6x5 mesh. Both counts named.
    {
        let comm1 = &ThreadComm::world(1).expect("world")[0];
        let mut halo5 = SpmdHalo::new(&gpu, &d1.parts[0], 1).expect("plan 5");
        let sys5 = DistSystem::split_one(&gpu, &mut halo5, comm1, &m5, &d1, &a5)
            .expect("split_one 5");
        let mut w5 = SpmdWorkspace::new(&gpu, comm1, &sys5, DistReduce::Exact).expect("ws 5");
        let mut halo8 = SpmdHalo::new(&gpu, &e1.parts[0], 1).expect("plan 8");
        let sys8 = DistSystem::split_one(&gpu, &mut halo8, comm1, &m8, &e1, &a8)
            .expect("split_one 8");
        let mut psi = gpu.zeros::<Scalar>(sys8.local()[0]).expect("alloc");
        let ctrl = controls(LinearSolverKind::PCG, Preconditioner::Diagonal, None);
        let msg = spmd_pcg(&gpu, &sol, comm1, &mut psi, &sys8, &mut halo8, &mut w5, &ctrl)
            .err()
            .expect("row 5 must refuse")
            .to_string();
        assert!(msg.starts_with("spmd_pcg: "), "{msg}");
        assert!(msg.contains("60"), "{msg}");
        assert!(msg.contains("240"), "{msg}");
    }

    // spmd_pcg, row 6: the halo plan's cell counts are not the system's -
    // the 5x4x3 system with the 8x6x5 mesh's halo plan.
    {
        let comm1 = &ThreadComm::world(1).expect("world")[0];
        let mut halo5 = SpmdHalo::new(&gpu, &d1.parts[0], 1).expect("plan 5");
        let sys5 = DistSystem::split_one(&gpu, &mut halo5, comm1, &m5, &d1, &a5)
            .expect("split_one 5");
        let mut w5 = SpmdWorkspace::new(&gpu, comm1, &sys5, DistReduce::Exact).expect("ws 5");
        let mut halo8 = SpmdHalo::new(&gpu, &e1.parts[0], 1).expect("plan 8");
        let mut psi = gpu.zeros::<Scalar>(sys5.local()[0]).expect("alloc");
        let ctrl = controls(LinearSolverKind::PCG, Preconditioner::Diagonal, None);
        let msg = spmd_pcg(&gpu, &sol, comm1, &mut psi, &sys5, &mut halo8, &mut w5, &ctrl)
            .err()
            .expect("row 6 must refuse")
            .to_string();
        assert!(msg.starts_with("spmd_pcg: "), "{msg}");
        assert!(msg.contains("60"), "{msg}");
        assert!(msg.contains("240"), "{msg}");
    }

    // spmd_pcg, row 7: BOTH ranks pass a psi without halo room. Both are
    // refused at the same check, before any collective.
    {
        let got = ThreadComm::run(2, |comm| -> Result<String> {
            let gpu = Gpu::new(0)?;
            let sol = SolverKernels::new(&gpu)?;
            let rank = comm.rank();
            let mut halo = SpmdHalo::new(&gpu, &d2.parts[rank], 2)?;
            let sys = DistSystem::split_one(&gpu, &mut halo, comm, &m5, &d2, &a5)?;
            let mut w = SpmdWorkspace::new(&gpu, comm, &sys, DistReduce::Exact)?;
            let owned = sys.owned()[0];
            let mut psi = gpu.zeros::<Scalar>(owned)?;
            let ctrl = controls(LinearSolverKind::PCG, Preconditioner::None, None);
            match spmd_pcg(&gpu, &sol, comm, &mut psi, &sys, &mut halo, &mut w, &ctrl) {
                Ok(_) => panic!("rank {rank}: row 7 should have refused"),
                Err(e) => Ok(e.to_string()),
            }
        })
        .expect("spmd run");
        for (r, part) in d2.parts.iter().enumerate() {
            let msg = got[r]
                .as_ref()
                .unwrap_or_else(|e| panic!("rank {r} failed: {e}"));
            assert!(msg.starts_with("spmd_pcg: "), "rank {r}: {msg}");
            assert!(
                msg.contains(&part.mesh.n_cells.to_string()),
                "rank {r}: {msg}"
            );
            assert!(
                msg.contains(&part.n_local().to_string()),
                "rank {r}: {msg}"
            );
            assert!(msg.contains("lduAmul"), "rank {r}: {msg}");
        }
    }

    // spmd_pcg, row 8: DIC is refused in SPEC-LIT §13.4's voice -
    // Error::Refused naming the setting and the available values, and the
    // colouring that would make it available. After `colour` the same call
    // is `Ok` - the block-local factorisation exists, so nothing is refused.
    {
        let comm1 = &ThreadComm::world(1).expect("world")[0];
        let mut halo = SpmdHalo::new(&gpu, &d1.parts[0], 1).expect("plan");
        let sys = DistSystem::split_one(&gpu, &mut halo, comm1, &m5, &d1, &a5).expect("split_one");
        let mut w = SpmdWorkspace::new(&gpu, comm1, &sys, DistReduce::Exact).expect("ws");
        let mut psi = gpu.zeros::<Scalar>(sys.local()[0]).expect("alloc");
        let ctrl = controls(LinearSolverKind::PCG, Preconditioner::Dic, None);
        let msg = spmd_pcg(&gpu, &sol, comm1, &mut psi, &sys, &mut halo, &mut w, &ctrl)
            .err()
            .expect("row 8 must refuse")
            .to_string();
        assert!(msg.contains("preconditioner"), "{msg}");
        assert!(msg.contains("diagonal"), "{msg}");
        assert!(msg.contains("SpmdWorkspace::colour"), "{msg}");
        w.colour(&gpu, &sys).expect("colour");
        assert!(w.colours() >= 1, "the part has colours after colouring");
        spmd_pcg(&gpu, &sol, comm1, &mut psi, &sys, &mut halo, &mut w, &ctrl)
            .expect("the same call after colour is Ok");
    }
}


/// The flag votes a solve takes at check interval 1: `n_iterations + 1`
/// unrefined. Under `single` the solve is refined (SPEC-LIT §118.12): each
/// of the `outer` inner solves votes once before it iterates and once per
/// iteration (`n_iterations + outer` in all), and each of the `outer + 1`
/// outer verdicts is voted too - `n_iterations + 2 outer + 1`, which is the
/// unrefined count at `outer = 0`.
fn expected_votes(perf: &DistPerformance, outer: usize) -> u64 {
    (perf.n_iterations + 2 * outer + 1) as u64
}

/// The outer refinement steps of the workspace's last solve: 0 in the f64
/// build, which does not refine.
#[cfg(feature = "single")]
fn outer_steps(w: &SpmdWorkspace) -> usize {
    w.ir_outer_steps()
}

#[cfg(not(feature = "single"))]
fn outer_steps(_w: &SpmdWorkspace) -> usize {
    0
}

/// The per-rank comparison every gate below makes: the counters, both
/// residuals' bits, `converged`, and the [`expected_votes`] - plus the
/// owned cells bit for bit the in-process part's and, when `one` is given,
/// bit for bit the ONE-part answer's at the owned cell's global index.
fn assert_rank(
    g: &[Scalar],
    perf: &DistPerformance,
    votes: u64,
    outer: usize,
    global_cell: &[Label],
    ref_field: &[Scalar],
    ref_perf: &DistPerformance,
    one: Option<(&[Scalar], usize)>,
    at: &str,
) {
    for i in 0..ref_field.len() {
        assert_eq!(
            g[i].to_bits(),
            ref_field[i].to_bits(),
            "vs the in-process part, {at} owned {i}"
        );
        if let Some((field, _)) = one {
            let gi = global_cell[i] as usize;
            assert_eq!(
                g[i].to_bits(),
                field[gi].to_bits(),
                "vs the one-part answer, {at} owned {i} (global {gi})"
            );
        }
    }
    assert_eq!(perf.n_iterations, ref_perf.n_iterations, "iterations, {at}");
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
    if let Some((_, iters)) = one {
        assert_eq!(perf.n_iterations, iters, "vs Reference ONE, {at}");
    }
    assert_eq!(votes, expected_votes(perf, outer), "votes, {at}");
}

/// The in-process solver as a plain fn pointer, matching [`SpmdSolve`].
fn dist_by_kind(
    kind: LinearSolverKind,
) -> fn(
    &Gpu,
    &SolverKernels,
    &mut [DevBuf<Scalar>],
    &DistSystem,
    &mut HaloExchange,
    &mut DistWorkspace,
    &SolverControls,
) -> Result<DistPerformance> {
    match kind {
        LinearSolverKind::PCG => dist_pcg,
        _ => dist_pbicgstab,
    }
}

/// **The §138.6 gate, in both builds.** The gate above with PBiCGStab on
/// both sides: two `blockgen` boxes, `Diagonal` and `None`, `P = 1..4`,
/// three partitioners, the exact reduction - on every rank every owned cell
/// is bit for bit the in-process part's answer AND the one-part answer, the
/// performance counters and both residuals' bits equal the in-process
/// solve's, and the flag vote counts `n_iterations + 1` reads and finds
/// nothing to report. Then, on the small box with `Diagonal` at `P = 2..4`,
/// the same against the in-process GATHERED solve - run-invariant only, so
/// the reference is the same decomposition, never the one-part answer.
#[test]
fn an_spmd_pbicgstab_is_the_in_process_pbicgstab_in_both_builds() {
    let gpu0 = match Gpu::new(0) {
        Ok(g) => g,
        Err(_) => return,
    };
    let sol0 = SolverKernels::new(&gpu0).expect("kernels");

    let mut rank_results = 0usize;
    let mut cells = 0usize;

    // ---- the exact leg ---------------------------------------------------
    for (nx, ny, nz) in [(5usize, 4usize, 3usize), (8, 6, 5)] {
        let (m, _) = build_case(CaseKind::Big, nx, ny, nz).expect("mesh");
        let a = poisson(&m);
        let whole: Vec<Scalar> = (0..m.n_cells).map(field).collect();

        for precon in [Preconditioner::Diagonal, Preconditioner::None] {
            let ctrl = controls(LinearSolverKind::PBiCGStab, precon, None);

            // Reference ONE: the in-process solve at one part.
            let one = Decomposition::build(&m, 1, &PartitionMethod::Linear).expect("cut");
            let (want, wp) = {
                let mut ex = HaloExchange::new(&gpu0, &one).expect("halo");
                let sys = DistSystem::split(&gpu0, &mut ex, &m, &one, &a).expect("split");
                let mut w = DistWorkspace::new(&gpu0, &sys, DistReduce::Exact).expect("ws");
                let mut psi = vec![gpu0.upload(&whole).expect("upload")];
                let perf = dist_pbicgstab(&gpu0, &sol0, &mut psi, &sys, &mut ex, &mut w, &ctrl)
                    .expect("P = 1");
                gpu0.sync().expect("sync");
                (gpu0.download(&psi[0]).expect("download"), perf)
            };
            assert!(
                wp.converged,
                "{nx}x{ny}x{nz} {precon:?}: the reference solve must converge"
            );
            assert!(
                wp.n_iterations > 3,
                "{nx}x{ny}x{nz} {precon:?}: a one-iteration gate proves nothing"
            );

            for np in 1..=4usize {
                for (name, dec) in cuts(&m, np) {
                    if np > 1 {
                        assert!(dec.n_cut_faces > 0, "P = {np} {name} cut nothing");
                        assert!(
                            dec.parts.iter().all(|p| p.n_halo > 0),
                            "P = {np} {name} left a part without a halo"
                        );
                    }
                    let (ref_fields, ref_perf, _, got) = both_ways(
                        &gpu0,
                        &sol0,
                        &m,
                        &a,
                        &whole,
                        &dec,
                        &ctrl,
                        DistReduce::Exact,
                        dist_pbicgstab,
                        spmd_pbicgstab,
                    )
                    .unwrap_or_else(|e| {
                        panic!("PBiCGStab {nx}x{ny}x{nz} {precon:?} P = {np} {name}: {e}")
                    });
                    for (r, part) in dec.parts.iter().enumerate() {
                        let (g, perf, votes, _, _, outer) = got[r]
                            .as_ref()
                            .unwrap_or_else(|e| {
                                panic!(
                                    "PBiCGStab {nx}x{ny}x{nz} {precon:?} P = {np} {name} \
                                     rank {r}: {e}"
                                )
                            });
                        rank_results += 1;
                        cells += part.mesh.n_cells;
                        let at = format!(
                            "PBiCGStab {nx}x{ny}x{nz} {precon:?} P = {np} {name} rank {r}"
                        );
                        assert_rank(
                            g,
                            perf,
                            *votes,
                            *outer,
                            &part.global_cell,
                            &ref_fields[r],
                            &ref_perf,
                            Some((&want, wp.n_iterations)),
                            &at,
                        );
                    }
                }
            }
        }
    }

    // ---- the gathered leg: run-invariant only, so the reference is the
    // in-process GATHERED solve of the SAME decomposition -----------------
    {
        let (m, _) = build_case(CaseKind::Big, 5, 4, 3).expect("mesh");
        let a = poisson(&m);
        let whole: Vec<Scalar> = (0..m.n_cells).map(field).collect();
        let ctrl = controls(LinearSolverKind::PBiCGStab, Preconditioner::Diagonal, None);
        for np in 2..=4usize {
            for (name, dec) in cuts(&m, np) {
                let (ref_fields, ref_perf, _, got) = both_ways(
                    &gpu0,
                    &sol0,
                    &m,
                    &a,
                    &whole,
                    &dec,
                    &ctrl,
                    DistReduce::Gathered,
                    dist_pbicgstab,
                    spmd_pbicgstab,
                )
                .unwrap_or_else(|e| panic!("gathered PBiCGStab P = {np} {name}: {e}"));
                for (r, part) in dec.parts.iter().enumerate() {
                    let (g, perf, votes, _, _, outer) = got[r]
                        .as_ref()
                        .unwrap_or_else(|e| panic!("gathered P = {np} {name} rank {r}: {e}"));
                    rank_results += 1;
                    cells += part.mesh.n_cells;
                    let at = format!("gathered PBiCGStab P = {np} {name} rank {r}");
                    assert_rank(
                        g,
                        perf,
                        *votes,
                        *outer,
                        &part.global_cell,
                        &ref_fields[r],
                        &ref_perf,
                        None,
                        &at,
                    );
                }
            }
        }
    }

    assert_eq!(rank_results, 139, "rank results compared");
    assert_eq!(
        cells, 6540,
        "owned cells compared against the in-process part"
    );
}

/// **The §138.6 block-local gate, in both builds.** `PCG + Dic` and
/// `PBiCGStab + Dilu`, both sides coloured, over the two `blockgen` boxes,
/// the exact reduction, THE TEN CUTS: every owned cell, every counter and
/// both residuals' bits equal the in-process block-local solve's, each
/// rank's colour count equals the in-process part's and is at least 1, and
/// `partition_invariant` is `false` on every rank.
///
/// NOT compared with a one-part answer - there is no reason for it to equal
/// one. Instead the not-partition-invariance is MEASURED: on the 8×6×5 box
/// with `Dic`, the ranks at `P >= 2` whose owned field differs from the
/// one-part `Dic` field in at least one bit are counted, and the count must
/// be positive.
#[test]
fn an_spmd_block_local_factorisation_is_the_in_process_one_in_both_builds() {
    let gpu0 = match Gpu::new(0) {
        Ok(g) => g,
        Err(_) => return,
    };
    let sol0 = SolverKernels::new(&gpu0).expect("kernels");

    let mut rank_results = 0usize;
    let mut cells = 0usize;
    let mut differing = 0usize;

    for (nx, ny, nz) in [(5usize, 4usize, 3usize), (8, 6, 5)] {
        let (m, _) = build_case(CaseKind::Big, nx, ny, nz).expect("mesh");
        let a = poisson(&m);
        let whole: Vec<Scalar> = (0..m.n_cells).map(field).collect();

        // The one-part `Dic` field, for the measurement on the big box.
        let one_dic: Option<Vec<Scalar>> = if (nx, ny, nz) == (8, 6, 5) {
            let one = Decomposition::build(&m, 1, &PartitionMethod::Linear).expect("cut");
            let ctrl1 = controls(LinearSolverKind::PCG, Preconditioner::Dic, None);
            let mut ex = HaloExchange::new(&gpu0, &one).expect("halo");
            let sys = DistSystem::split(&gpu0, &mut ex, &m, &one, &a).expect("split");
            let mut w = DistWorkspace::new(&gpu0, &sys, DistReduce::Exact).expect("ws");
            w.colour(&gpu0, &sys).expect("colour");
            let mut psi = vec![gpu0.upload(&whole).expect("upload")];
            let perf = dist_pcg(&gpu0, &sol0, &mut psi, &sys, &mut ex, &mut w, &ctrl1)
                .expect("one-part Dic");
            gpu0.sync().expect("sync");
            assert!(perf.converged, "the one-part Dic solve must converge");
            Some(gpu0.download(&psi[0]).expect("download"))
        } else {
            None
        };

        for (kind, precon) in [
            (LinearSolverKind::PCG, Preconditioner::Dic),
            (LinearSolverKind::PBiCGStab, Preconditioner::Dilu),
        ] {
            let ctrl = controls(kind, precon, None);
            let spmd_by: SpmdSolve = match kind {
                LinearSolverKind::PCG => spmd_pcg,
                _ => spmd_pbicgstab,
            };
            for np in 1..=4usize {
                for (name, dec) in cuts(&m, np) {
                    if np > 1 {
                        assert!(dec.n_cut_faces > 0, "P = {np} {name} cut nothing");
                        assert!(
                            dec.parts.iter().all(|p| p.n_halo > 0),
                            "P = {np} {name} left a part without a halo"
                        );
                    }
                    let (ref_fields, ref_perf, ref_colours, got) = both_ways(
                        &gpu0,
                        &sol0,
                        &m,
                        &a,
                        &whole,
                        &dec,
                        &ctrl,
                        DistReduce::Exact,
                        dist_by_kind(kind),
                        spmd_by,
                    )
                    .unwrap_or_else(|e| {
                        panic!("{nx}x{ny}x{nz} {kind:?}/{precon:?} P = {np} {name}: {e}")
                    });
                    assert!(
                        ref_perf.converged,
                        "{nx}x{ny}x{nz} {kind:?}/{precon:?} P = {np} {name}: the in-process \
                         block-local solve must converge"
                    );

                    for (r, part) in dec.parts.iter().enumerate() {
                        let (g, perf, votes, colours, pinv, outer) = got[r]
                            .as_ref()
                            .unwrap_or_else(|e| {
                                panic!(
                                    "{nx}x{ny}x{nz} {kind:?}/{precon:?} P = {np} {name} \
                                     rank {r}: {e}"
                                )
                            });
                        rank_results += 1;
                        cells += part.mesh.n_cells;
                        let at = format!(
                            "{nx}x{ny}x{nz} {kind:?}/{precon:?} P = {np} {name} rank {r}"
                        );
                        assert_rank(
                            g,
                            perf,
                            *votes,
                            *outer,
                            &part.global_cell,
                            &ref_fields[r],
                            &ref_perf,
                            None,
                            &at,
                        );
                        assert_eq!(*colours, ref_colours[r], "colours, {at}");
                        assert!(*colours >= 1, "a coloured part has colours, {at}");
                        assert!(
                            !pinv,
                            "the block-local factorisation is not partition-invariant, {at}"
                        );
                        if let Some(one) = one_dic.as_deref() {
                            if np > 1 && kind == LinearSolverKind::PCG {
                                let differs = (0..part.mesh.n_cells).any(|i| {
                                    let gi = part.global_cell[i] as usize;
                                    g[i].to_bits() != one[gi].to_bits()
                                });
                                if differs {
                                    differing += 1;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    assert_eq!(rank_results, 112, "rank results compared");
    assert_eq!(
        cells, 6000,
        "owned cells compared against the in-process part"
    );
    assert!(
        differing > 0,
        "at P >= 2 with Dic on 8x6x5 at least one rank's field must differ \
         from the one-part field, or not-partition-invariant is assumed \
         rather than measured"
    );
}

/// The counts of SPEC-LIT §138.6: at 5 fixed iterations, `P = 3`,
/// PBiCGStab makes `2k + 3 = 13` exchanges and `4k + 4 = 24` reductions and
/// PCG makes `k + 3 = 8` and `2k + 5 = 15`; no vote is taken; and the
/// handle counts `1 + 2 (reductions)` allgathers with the exact reduction
/// and `1 + (reductions)` with the gathered one. A preconditioner changes
/// none of these numbers, because no preconditioner here exchanges or
/// reduces - `Dilu` and `Dic` included, on a coloured workspace.
#[test]
fn an_spmd_pbicgstab_counts_its_exchanges_and_collectives() {
    if Gpu::new(0).is_err() {
        return;
    }
    let (m, _) = build_case(CaseKind::Big, 5, 4, 3).expect("mesh");
    let a = poisson(&m);
    let whole: Vec<Scalar> = (0..m.n_cells).map(field).collect();
    let dec = Decomposition::build(&m, 3, &PartitionMethod::Hilbert).expect("cut");

    for (kind, precon, colour, want_ex, want_red, want_exact, want_gathered) in [
        (LinearSolverKind::PBiCGStab, Preconditioner::Diagonal, false, 13usize, 24usize, 49u64, 25u64),
        (LinearSolverKind::PBiCGStab, Preconditioner::Dilu, true, 13, 24, 49, 25),
        (LinearSolverKind::PCG, Preconditioner::Dic, true, 8, 15, 31, 16),
    ] {
        let ctrl = controls(kind, precon, Some(5));
        let spmd_by: SpmdSolve = match kind {
            LinearSolverKind::PCG => spmd_pcg,
            _ => spmd_pbicgstab,
        };
        for (reduce, want_allgathers) in [
            (DistReduce::Exact, want_exact),
            (DistReduce::Gathered, want_gathered),
        ] {
            let got = ThreadComm::run(3, |comm| -> Result<(DistPerformance, u64, u64)> {
                let gpu = Gpu::new(0)?;
                let sol = SolverKernels::new(&gpu)?;
                let rank = comm.rank();
                let mut halo = SpmdHalo::new(&gpu, &dec.parts[rank], 3)?;
                let sys = DistSystem::split_one(&gpu, &mut halo, comm, &m, &dec, &a)?;
                let mut w = SpmdWorkspace::new(&gpu, comm, &sys, reduce)?;
                w.set_flag_vote(true);
                if colour {
                    w.colour(&gpu, &sys)?;
                }
                let mut psi = gpu.upload(&dec.split_field(rank, &whole)?)?;
                let perf = spmd_by(&gpu, &sol, comm, &mut psi, &sys, &mut halo, &mut w, &ctrl)?;
                gpu.sync()?;
                Ok((perf, w.n_flag_votes(), w.n_allgathers()))
            })
            .expect("spmd run");

            for r in 0..3 {
                let (perf, votes, allgathers) = got[r]
                    .as_ref()
                    .unwrap_or_else(|e| panic!("rank {r} failed: {e}"));
                assert_eq!(
                    perf.n_iterations, 5,
                    "rank {r} {kind:?}/{precon:?} {reduce:?}"
                );
                assert_eq!(
                    perf.n_exchanges, want_ex,
                    "rank {r} {kind:?}/{precon:?} {reduce:?}"
                );
                assert_eq!(
                    perf.n_reductions, want_red,
                    "rank {r} {kind:?}/{precon:?} {reduce:?}"
                );
                assert_eq!(
                    *votes, 0,
                    "a fixed_iters solve reads no flag, rank {r} {kind:?}/{precon:?}"
                );
                assert_eq!(
                    *allgathers, want_allgathers,
                    "rank {r} {kind:?}/{precon:?} {reduce:?}"
                );
            }
        }
    }
}

/// The eight `spmd_*` refusal rows of SPEC-LIT §138.5's table, in order,
/// every row with a block here, for ONE solver: `who` is the message
/// prefix the rows must start with and `solve_by` the solver every row
/// calls. No multi-rank block lets ONE rank fail alone: every rank is
/// refused at the same check, BEFORE any collective, because a refusal only
/// one rank sees would leave the other ranks waiting in a collective.
fn spmd_solver_refusals(
    gpu: &Gpu,
    sol: &SolverKernels,
    who: &str,
    precon8: Preconditioner,
    solve_by: SpmdSolve,
) {
    let (m5, _) = build_case(CaseKind::Big, 5, 4, 3).expect("mesh");
    let (m8, _) = build_case(CaseKind::Big, 8, 6, 5).expect("mesh");
    let a5 = poisson(&m5);
    let a8 = poisson(&m8);
    let d1 = Decomposition::build(&m5, 1, &PartitionMethod::Linear).expect("cut");
    let d2 = Decomposition::build(&m5, 2, &PartitionMethod::Linear).expect("cut");
    let e1 = Decomposition::build(&m8, 1, &PartitionMethod::Linear).expect("cut");
    let kind = if who.starts_with("spmd_pbicgstab") {
        LinearSolverKind::PBiCGStab
    } else {
        LinearSolverKind::PCG
    };

    // row 1: inside a CUDA-graph capture. The same call outside the capture
    // is `Ok` - the refusal left nothing behind.
    {
        let comm1 = &ThreadComm::world(1).expect("world")[0];
        let mut halo = SpmdHalo::new(gpu, &d1.parts[0], 1).expect("plan");
        let sys = DistSystem::split_one(gpu, &mut halo, comm1, &m5, &d1, &a5).expect("split_one");
        let mut w = SpmdWorkspace::new(gpu, comm1, &sys, DistReduce::Exact).expect("ws");
        w.set_flag_vote(true);
        let whole: Vec<Scalar> = (0..m5.n_cells).map(field).collect();
        let mut psi = gpu.upload(&whole).expect("upload");
        let ctrl = controls(kind, Preconditioner::Diagonal, None);
        let msg = gpu
            .capture(|_| {
                solve_by(gpu, sol, comm1, &mut psi, &sys, &mut halo, &mut w, &ctrl).map(|_| ())
            })
            .err()
            .expect("row 1 must refuse")
            .to_string();
        assert!(msg.starts_with(who), "{msg}");
        assert!(msg.contains("capture"), "{msg}");
        solve_by(gpu, sol, comm1, &mut psi, &sys, &mut halo, &mut w, &ctrl)
            .expect("the same call outside the capture is Ok");
    }

    // row 2: a `split` system with a workspace from split_one.
    {
        let comm1 = &ThreadComm::world(1).expect("world")[0];
        let mut ex = HaloExchange::new(gpu, &d1).expect("halo");
        let sys_split = DistSystem::split(gpu, &mut ex, &m5, &d1, &a5).expect("split");
        let mut halo_one = SpmdHalo::new(gpu, &d1.parts[0], 1).expect("plan");
        let sys_one =
            DistSystem::split_one(gpu, &mut halo_one, comm1, &m5, &d1, &a5).expect("split_one");
        let mut w = SpmdWorkspace::new(gpu, comm1, &sys_one, DistReduce::Exact).expect("ws");
        let whole: Vec<Scalar> = (0..m5.n_cells).map(field).collect();
        let mut psi = gpu.upload(&whole).expect("upload");
        let ctrl = controls(kind, Preconditioner::Diagonal, None);
        let msg = solve_by(gpu, sol, comm1, &mut psi, &sys_split, &mut halo_one, &mut w, &ctrl)
            .err()
            .expect("row 2 must refuse")
            .to_string();
        assert!(msg.starts_with(who), "{msg}");
        assert!(msg.contains("DistSystem::split_one"), "{msg}");
    }

    // row 3: everything built in a 1-rank world, called with rank 0 of a
    // 2-rank world. The refusal is before anything is sent, so the second
    // rank of that world is never touched.
    {
        let comm1 = &ThreadComm::world(1).expect("world")[0];
        let mut halo = SpmdHalo::new(gpu, &d1.parts[0], 1).expect("plan");
        let sys = DistSystem::split_one(gpu, &mut halo, comm1, &m5, &d1, &a5).expect("split_one");
        let mut w = SpmdWorkspace::new(gpu, comm1, &sys, DistReduce::Exact).expect("ws");
        let whole: Vec<Scalar> = (0..m5.n_cells).map(field).collect();
        let mut psi = gpu.upload(&whole).expect("upload");
        let ctrl = controls(kind, Preconditioner::Diagonal, None);
        let w2 = ThreadComm::world(2).expect("world");
        let msg = solve_by(gpu, sol, &w2[0], &mut psi, &sys, &mut halo, &mut w, &ctrl)
            .err()
            .expect("row 3 must refuse")
            .to_string();
        assert!(msg.starts_with(who), "{msg}");
        assert!(msg.contains("2 ranks"), "{msg}");
        assert!(msg.contains("1 ranks"), "{msg}");
    }

    // row 4: the right world size, the WRONG handle - rank r calls with
    // handle `1 - r` of a SECOND world. ThreadComm is Send, not Sync, so
    // the second world's handles sit in a mutex and a rank TAKES its
    // borrowed one out. Both ranks are refused at the same check, before
    // anything is sent, so no rank waits.
    {
        let second = std::sync::Mutex::new(
            ThreadComm::world(2)
                .expect("world")
                .into_iter()
                .map(Some)
                .collect::<Vec<_>>(),
        );
        let got = ThreadComm::run(2, |comm| -> Result<String> {
            let gpu = Gpu::new(0)?;
            let sol = SolverKernels::new(&gpu)?;
            let rank = comm.rank();
            let mut halo = SpmdHalo::new(&gpu, &d2.parts[rank], 2)?;
            let sys = DistSystem::split_one(&gpu, &mut halo, comm, &m5, &d2, &a5)?;
            let mut w = SpmdWorkspace::new(&gpu, comm, &sys, DistReduce::Exact)?;
            let mut psi = gpu.zeros::<Scalar>(sys.local()[0])?;
            let other = {
                let mut handles = second.lock().expect("lock");
                handles[1 - rank].take().expect("handle")
            };
            let ctrl = controls(kind, Preconditioner::Diagonal, None);
            match solve_by(&gpu, &sol, &other, &mut psi, &sys, &mut halo, &mut w, &ctrl) {
                Ok(_) => panic!("rank {rank}: row 4 should have refused"),
                Err(e) => Ok(e.to_string()),
            }
        })
        .expect("spmd run");
        for r in 0..2 {
            let msg = got[r]
                .as_ref()
                .unwrap_or_else(|e| panic!("rank {r} failed: {e}"));
            assert!(msg.starts_with(who), "rank {r}: {msg}");
            assert!(
                msg.contains(&format!("rank {}", 1 - r)),
                "rank {r}: {msg}"
            );
            assert!(msg.contains(&format!("part {r}")), "rank {r}: {msg}");
        }
    }

    // row 5: the workspace was built for another system - the 5x4x3 one,
    // against a system from the 8x6x5 mesh. Both counts named.
    {
        let comm1 = &ThreadComm::world(1).expect("world")[0];
        let mut halo5 = SpmdHalo::new(gpu, &d1.parts[0], 1).expect("plan 5");
        let sys5 = DistSystem::split_one(gpu, &mut halo5, comm1, &m5, &d1, &a5)
            .expect("split_one 5");
        let mut w5 = SpmdWorkspace::new(gpu, comm1, &sys5, DistReduce::Exact).expect("ws 5");
        let mut halo8 = SpmdHalo::new(gpu, &e1.parts[0], 1).expect("plan 8");
        let sys8 = DistSystem::split_one(gpu, &mut halo8, comm1, &m8, &e1, &a8)
            .expect("split_one 8");
        let mut psi = gpu.zeros::<Scalar>(sys8.local()[0]).expect("alloc");
        let ctrl = controls(kind, Preconditioner::Diagonal, None);
        let msg = solve_by(gpu, sol, comm1, &mut psi, &sys8, &mut halo8, &mut w5, &ctrl)
            .err()
            .expect("row 5 must refuse")
            .to_string();
        assert!(msg.starts_with(who), "{msg}");
        assert!(msg.contains("60"), "{msg}");
        assert!(msg.contains("240"), "{msg}");
    }

    // row 6: the halo plan's cell counts are not the system's - the 5x4x3
    // system with the 8x6x5 mesh's halo plan.
    {
        let comm1 = &ThreadComm::world(1).expect("world")[0];
        let mut halo5 = SpmdHalo::new(gpu, &d1.parts[0], 1).expect("plan 5");
        let sys5 = DistSystem::split_one(gpu, &mut halo5, comm1, &m5, &d1, &a5)
            .expect("split_one 5");
        let mut w5 = SpmdWorkspace::new(gpu, comm1, &sys5, DistReduce::Exact).expect("ws 5");
        let mut halo8 = SpmdHalo::new(gpu, &e1.parts[0], 1).expect("plan 8");
        let mut psi = gpu.zeros::<Scalar>(sys5.local()[0]).expect("alloc");
        let ctrl = controls(kind, Preconditioner::Diagonal, None);
        let msg = solve_by(gpu, sol, comm1, &mut psi, &sys5, &mut halo8, &mut w5, &ctrl)
            .err()
            .expect("row 6 must refuse")
            .to_string();
        assert!(msg.starts_with(who), "{msg}");
        assert!(msg.contains("60"), "{msg}");
        assert!(msg.contains("240"), "{msg}");
    }

    // row 7: BOTH ranks pass a psi without halo room. Both are refused at
    // the same check, before any collective.
    {
        let got = ThreadComm::run(2, |comm| -> Result<String> {
            let gpu = Gpu::new(0)?;
            let sol = SolverKernels::new(&gpu)?;
            let rank = comm.rank();
            let mut halo = SpmdHalo::new(&gpu, &d2.parts[rank], 2)?;
            let sys = DistSystem::split_one(&gpu, &mut halo, comm, &m5, &d2, &a5)?;
            let mut w = SpmdWorkspace::new(&gpu, comm, &sys, DistReduce::Exact)?;
            let owned = sys.owned()[0];
            let mut psi = gpu.zeros::<Scalar>(owned)?;
            let ctrl = controls(kind, Preconditioner::None, None);
            match solve_by(&gpu, &sol, comm, &mut psi, &sys, &mut halo, &mut w, &ctrl) {
                Ok(_) => panic!("rank {rank}: row 7 should have refused"),
                Err(e) => Ok(e.to_string()),
            }
        })
        .expect("spmd run");
        for (r, part) in d2.parts.iter().enumerate() {
            let msg = got[r]
                .as_ref()
                .unwrap_or_else(|e| panic!("rank {r} failed: {e}"));
            assert!(msg.starts_with(who), "rank {r}: {msg}");
            assert!(
                msg.contains(&part.mesh.n_cells.to_string()),
                "rank {r}: {msg}"
            );
            assert!(
                msg.contains(&part.n_local().to_string()),
                "rank {r}: {msg}"
            );
            assert!(msg.contains("lduAmul"), "rank {r}: {msg}");
        }
    }

    // row 8: the block-local factorisation is refused in SPEC-LIT §13.4's
    // voice - Error::Refused naming the setting, the available values and
    // the colouring that would make it available.
    {
        let comm1 = &ThreadComm::world(1).expect("world")[0];
        let mut halo = SpmdHalo::new(gpu, &d1.parts[0], 1).expect("plan");
        let sys = DistSystem::split_one(gpu, &mut halo, comm1, &m5, &d1, &a5).expect("split_one");
        let mut w = SpmdWorkspace::new(gpu, comm1, &sys, DistReduce::Exact).expect("ws");
        let mut psi = gpu.zeros::<Scalar>(sys.local()[0]).expect("alloc");
        let ctrl = controls(kind, precon8, None);
        let msg = solve_by(gpu, sol, comm1, &mut psi, &sys, &mut halo, &mut w, &ctrl)
            .err()
            .expect("row 8 must refuse")
            .to_string();
        assert!(msg.contains("preconditioner"), "{msg}");
        assert!(msg.contains("diagonal"), "{msg}");
        assert!(msg.contains("SpmdWorkspace::colour"), "{msg}");
    }
}

/// The PBiCGStab form of every `spmd_pcg` refusal row, then the three
/// `SpmdWorkspace::colour` rows. The colouring is local - it sends nothing -
/// so its rows run in one-rank worlds and one two-rank world where BOTH
/// ranks refuse at the same local check, before any collective.
#[test]
fn an_spmd_pbicgstab_refuses_by_name() {
    let gpu = match Gpu::new(0) {
        Ok(g) => g,
        Err(_) => return,
    };
    let sol = SolverKernels::new(&gpu).expect("kernels");
    spmd_solver_refusals(
        &gpu,
        &sol,
        "spmd_pbicgstab: ",
        Preconditioner::Dilu,
        spmd_pbicgstab,
    );

    let (m5, _) = build_case(CaseKind::Big, 5, 4, 3).expect("mesh");
    let (m8, _) = build_case(CaseKind::Big, 8, 6, 5).expect("mesh");
    let a5 = poisson(&m5);
    let a8 = poisson(&m8);
    let d1 = Decomposition::build(&m5, 1, &PartitionMethod::Linear).expect("cut");
    let d2 = Decomposition::build(&m5, 2, &PartitionMethod::Linear).expect("cut");
    let e1 = Decomposition::build(&m8, 1, &PartitionMethod::Linear).expect("cut");

    // colour, row 1: a system from `DistSystem::split` is not the one-part
    // system the colouring needs.
    {
        let comm1 = &ThreadComm::world(1).expect("world")[0];
        let mut ex = HaloExchange::new(&gpu, &d1).expect("halo");
        let sys_split = DistSystem::split(&gpu, &mut ex, &m5, &d1, &a5).expect("split");
        let mut halo_one = SpmdHalo::new(&gpu, &d1.parts[0], 1).expect("plan");
        let sys_one =
            DistSystem::split_one(&gpu, &mut halo_one, comm1, &m5, &d1, &a5).expect("split_one");
        let mut w = SpmdWorkspace::new(&gpu, comm1, &sys_one, DistReduce::Exact).expect("ws");
        assert_eq!(w.colours(), 0, "nothing is coloured before `colour`");
        let msg = w
            .colour(&gpu, &sys_split)
            .err()
            .expect("colour must refuse a split system")
            .to_string();
        assert!(msg.starts_with("distsolve: "), "{msg}");
        assert!(msg.contains("DistSystem::split_one"), "{msg}");
    }

    // colour, row 2: the system is the OTHER rank's part. Rank r builds its
    // own world's system, then - with handle `1 - r` of a SECOND world,
    // taken out of a mutex because ThreadComm is Send, not Sync - a second
    // halo and system for part `1 - r`. Both threads build theirs, so the
    // second world's split_one exchanges complete, and then BOTH ranks are
    // refused at colour's local part check, before any collective.
    {
        let second = std::sync::Mutex::new(
            ThreadComm::world(2)
                .expect("world")
                .into_iter()
                .map(Some)
                .collect::<Vec<_>>(),
        );
        let got = ThreadComm::run(2, |comm| -> Result<String> {
            let gpu = Gpu::new(0)?;
            let rank = comm.rank();
            let mut halo = SpmdHalo::new(&gpu, &d2.parts[rank], 2)?;
            let sys = DistSystem::split_one(&gpu, &mut halo, comm, &m5, &d2, &a5)?;
            let mut w = SpmdWorkspace::new(&gpu, comm, &sys, DistReduce::Exact)?;
            let other = {
                let mut handles = second.lock().expect("lock");
                handles[1 - rank].take().expect("handle")
            };
            let mut other_halo = SpmdHalo::new(&gpu, &d2.parts[1 - rank], 2)?;
            let other_sys =
                DistSystem::split_one(&gpu, &mut other_halo, &other, &m5, &d2, &a5)?;
            match w.colour(&gpu, &other_sys) {
                Ok(()) => panic!("rank {rank}: colour should have refused"),
                Err(e) => Ok(e.to_string()),
            }
        })
        .expect("spmd run");
        for r in 0..2 {
            let msg = got[r]
                .as_ref()
                .unwrap_or_else(|e| panic!("rank {r} failed: {e}"));
            assert!(msg.starts_with("distsolve: "), "rank {r}: {msg}");
            assert!(msg.contains(&format!("rank {r}")), "rank {r}: {msg}");
            assert!(
                msg.contains(&format!("part {}", 1 - r)),
                "rank {r}: {msg}"
            );
        }
    }

    // colour, row 3: the workspace was built for the 5x4x3 part, the system
    // is the 8x6x5 one. Both counts named.
    {
        let comm1 = &ThreadComm::world(1).expect("world")[0];
        let mut halo5 = SpmdHalo::new(&gpu, &d1.parts[0], 1).expect("plan 5");
        let sys5 = DistSystem::split_one(&gpu, &mut halo5, comm1, &m5, &d1, &a5)
            .expect("split_one 5");
        let mut w5 = SpmdWorkspace::new(&gpu, comm1, &sys5, DistReduce::Exact).expect("ws 5");
        let mut halo8 = SpmdHalo::new(&gpu, &e1.parts[0], 1).expect("plan 8");
        let sys8 = DistSystem::split_one(&gpu, &mut halo8, comm1, &m8, &e1, &a8)
            .expect("split_one 8");
        let msg = w5
            .colour(&gpu, &sys8)
            .err()
            .expect("colour must refuse another system")
            .to_string();
        assert!(msg.starts_with("distsolve: "), "{msg}");
        assert!(msg.contains("60 owned"), "{msg}");
        assert!(msg.contains("240 owned"), "{msg}");
    }
}

/// `spmd_solve` dispatches on `ctrl.solver` as `dist_solve` does, and
/// refuses `GAMG` by name with `PBiCGStab` as the alternative. At 3 fixed
/// iterations in a one-rank world PCG makes `k + 3 = 6` exchanges and
/// `2k + 5 = 11` reductions, PBiCGStab `2k + 3 = 9` and `4k + 4 = 16`.
#[test]
fn an_spmd_solve_dispatches_and_refuses_gamg() {
    let gpu = match Gpu::new(0) {
        Ok(g) => g,
        Err(_) => return,
    };
    let sol = SolverKernels::new(&gpu).expect("kernels");
    let (m, _) = build_case(CaseKind::Big, 5, 4, 3).expect("mesh");
    let a = poisson(&m);
    let whole: Vec<Scalar> = (0..m.n_cells).map(field).collect();
    let dec = Decomposition::build(&m, 1, &PartitionMethod::Linear).expect("cut");

    for (kind, want_ex, want_red) in [
        (LinearSolverKind::PCG, 6usize, 11usize),
        (LinearSolverKind::PBiCGStab, 9, 16),
    ] {
        let comm1 = &ThreadComm::world(1).expect("world")[0];
        let mut halo = SpmdHalo::new(&gpu, &dec.parts[0], 1).expect("plan");
        let sys = DistSystem::split_one(&gpu, &mut halo, comm1, &m, &dec, &a).expect("split_one");
        let mut w = SpmdWorkspace::new(&gpu, comm1, &sys, DistReduce::Exact).expect("ws");
        let mut psi = gpu.upload(&whole).expect("upload");
        let ctrl = controls(kind, Preconditioner::Diagonal, Some(3));
        let perf = spmd_solve(&gpu, &sol, comm1, &mut psi, &sys, &mut halo, &mut w, &ctrl)
            .expect("the dispatch solves");
        assert_eq!(perf.n_iterations, 3, "{kind:?}");
        assert_eq!(perf.n_exchanges, want_ex, "{kind:?}");
        assert_eq!(perf.n_reductions, want_red, "{kind:?}");
    }

    // GAMG without a distributed cycle attached (SPEC-LIT §141.11) is
    // refused by name, in §13.4's voice, before anything is launched.
    {
        let comm1 = &ThreadComm::world(1).expect("world")[0];
        let mut halo = SpmdHalo::new(&gpu, &dec.parts[0], 1).expect("plan");
        let sys = DistSystem::split_one(&gpu, &mut halo, comm1, &m, &dec, &a).expect("split_one");
        let mut w = SpmdWorkspace::new(&gpu, comm1, &sys, DistReduce::Exact).expect("ws");
        let mut psi = gpu.upload(&whole).expect("upload");
        let mut ctrl = controls(LinearSolverKind::PCG, Preconditioner::Diagonal, Some(3));
        ctrl.solver = LinearSolverKind::Gamg;
        let msg = spmd_solve(&gpu, &sol, comm1, &mut psi, &sys, &mut halo, &mut w, &ctrl)
            .err()
            .expect("GAMG must be refused")
            .to_string();
        assert!(msg.contains("solver"), "{msg}");
        assert!(msg.contains("GAMG"), "{msg}");
        assert!(msg.contains("PBiCGStab"), "{msg}");
    }
}
