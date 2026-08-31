// meteor-cfd - Copyright (c) 2026 주식회사 메테오시뮬레이션 (Meteo Simulation Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.

//! The gate SPEC-LIT §73 exists to pass, and the measurement of what the
//! block-local factorisation costs.
//!
//! Two claims are separated here on purpose, because they are the two
//! guarantees of §72.5 arriving at a solver:
//!
//! * a solve with the **exact** reduction and a **partition-invariant**
//!   preconditioner is bit-for-bit the undecomposed solve, at every part
//!   count, under every partition map and every relabelling - and takes the
//!   same number of iterations, because the convergence test reads a number
//!   that did not move;
//! * a solve with the **block-local** DIC or DILU is a different
//!   preconditioner for every cut, so it is *not*, and the test asserts that
//!   it really does move rather than letting a silently-inert preconditioner
//!   pass the first gate by accident.
//!
//! Provenance: ORIGINAL - the tests, the matrix and the harness. No external
//! source (`PROVENANCE.md`, *GPU plumbing and tooling - original*).
//! No GPL-licensed source was consulted.

use super::*;
use crate::decompose::tests::{boxes, round_robin};
use crate::decompose::PartitionMethod;
use crate::solver::{SolverPerformance, SolverWorkspace};

/// Every device test needs a card. Returning `None` makes the test pass
/// vacuously on a machine without one, which is the convention the rest of the
/// crate follows.
fn gpu() -> Option<Gpu> {
    Gpu::new(0).ok()
}

/// The message of a refusal. Written out because `unwrap_err` wants the OK
/// type to be `Debug`, and a workspace holding device buffers is not.
fn err<T>(r: Result<T>) -> String {
    match r {
        Ok(_) => panic!("this was supposed to be refused"),
        Err(e) => e.to_string(),
    }
}

// ==========================================================================
//  The system under test
// ==========================================================================

/// `1/dt`, which is what makes the operator below non-singular on an
/// all-Neumann mesh and therefore a legitimate PCG problem.
const RDT: Scalar = 1.5;

/// A deterministic initial field of the GLOBAL cell id, so that distributing
/// it is a permutation and never a computation.
fn field(c: usize) -> Scalar {
    0.25 + 0.0625 * ((c * 37) % 23) as Scalar
}

/// A deterministic Dirichlet value per boundary cell, likewise.
fn wall(c: usize) -> Scalar {
    0.5 + 0.125 * ((c * 11) % 7) as Scalar
}

/// A **symmetric positive definite** Poisson operator plus a `ddt`, from the
/// mesh's own metrics, already folded.
///
/// SPD because PCG needs it and because DIC is the Cholesky form; folded on
/// the host because `split_matrix` sets `internal_coeffs = 0` on a cut face,
/// which makes the fold a no-op there and therefore makes folding before or
/// after the split the same matrix (SPEC-LIT §71.6). Folding here rather than
/// on each part also keeps this harness free of the sixteen assembly kernels
/// §70.5 refuses to a decomposed run.
///
/// ```text
/// internal face f:   upper = lower = -g ,  diag[own] += g ,  diag[nbr] += g
/// coupled face bf:   diag[own] += g ,  boundary_coeffs = g
/// Dirichlet face bf: diag[own] += g ,  source[own] += g wall(own)
/// every cell c:      diag += V rdt ,  source += V rdt field(c)
/// ```
///
/// The coupled sign is the one `lduAmul` applies as `sum -= bc * psi_N`, so
/// `bc = g` reproduces the internal face's `-g psi_N` exactly.
fn poisson(m: &HostMesh) -> HostLduMatrix {
    let mut a = HostLduMatrix::zeros(m);
    for f in 0..m.n_internal_faces {
        let g = m.mag_sf[f] * m.delta_coeffs[f];
        a.upper[f] = -g;
        a.lower[f] = -g;
        a.diag[m.owner[f] as usize] += g;
        a.diag[m.neighbour[f] as usize] += g;
    }
    for bf in 0..m.n_boundary_faces {
        let g = m.b_mag_sf[bf] * m.b_delta_coeffs[bf];
        let c = m.b_face_cells[bf] as usize;
        a.diag[c] += g;
        if m.b_nbr_cell[bf] >= 0 {
            a.boundary_coeffs[bf] = g;
        } else {
            a.source[c] += g * wall(c);
        }
    }
    for c in 0..m.n_cells {
        let vd = m.v[c] * RDT;
        a.diag[c] += vd;
        a.source[c] += vd * field(c);
    }
    a
}

/// Controls with everything that could hide a difference turned off.
fn controls(solver: LinearSolverKind, precon: Preconditioner, fixed: Option<usize>) -> SolverControls {
    SolverControls {
        solver,
        precon,
        tolerance: 1e-12,
        rel_tol: 0.0,
        max_iter: fixed.map_or(400, |k| k as Label),
        min_iter: 0,
        check_interval: 1,
        fixed_iters: fixed.is_some(),
        report_residuals: true,
    }
}

/// The undecomposed solve, through the crate's own solver.
fn serial(
    gpu: &Gpu,
    sol: &SolverKernels,
    m: &HostMesh,
    a: &HostLduMatrix,
    ctrl: &SolverControls,
) -> Result<(Vec<Scalar>, SolverPerformance)> {
    let gm = GpuMesh::upload(gpu, m)?;
    let ga = a.upload(gpu)?;
    let mut w = SolverWorkspace::for_mesh(gpu, &gm)?;
    let mut psi = gpu.upload(&(0..m.n_cells).map(field).collect::<Vec<Scalar>>())?;
    let perf = solver::solve(gpu, sol, &mut psi, &ga, &gm, &mut w, ctrl)?;
    gpu.sync()?;
    Ok((gpu.download(&psi)?, perf))
}

/// The decomposed solve, gathered back into whole-mesh order.
fn distributed(
    gpu: &Gpu,
    sol: &SolverKernels,
    m: &HostMesh,
    dec: &Decomposition,
    a: &HostLduMatrix,
    ctrl: &SolverControls,
    reduce: DistReduce,
) -> Result<(Vec<Scalar>, DistPerformance)> {
    let mut ex = HaloExchange::new(gpu, dec)?;
    let sys = DistSystem::split(gpu, &mut ex, m, dec, a)?;
    let mut w = DistWorkspace::new(gpu, &sys, reduce)?;
    if matches!(ctrl.precon, Preconditioner::Dic | Preconditioner::Dilu) {
        w.colour(gpu, &sys)?;
    }

    let whole: Vec<Scalar> = (0..m.n_cells).map(field).collect();
    let mut psi: Vec<DevBuf<Scalar>> = (0..dec.n_parts)
        .map(|p| gpu.upload(&dec.split_field(p, &whole)?))
        .collect::<Result<_>>()?;

    let perf = dist_solve(gpu, sol, &mut psi, &sys, &mut ex, &mut w, ctrl)?;
    gpu.sync()?;
    let per_part: Vec<Vec<Scalar>> = psi.iter().map(|b| gpu.download(b)).collect::<Result<_>>()?;
    Ok((dec.gather_field(&per_part)?, perf))
}

/// Every partition of `m` this file gates on: three methods at every part
/// count, plus the round robin, which cuts almost every face and is the one a
/// good partitioner would never produce.
fn partitions(m: &HostMesh, np: usize) -> Vec<(String, PartitionMethod)> {
    vec![
        ("hilbert".to_string(), PartitionMethod::Hilbert),
        ("linear".to_string(), PartitionMethod::Linear),
        ("roundrobin".to_string(), round_robin(m.n_cells, np)),
    ]
}

/// Bit-for-bit, with the first offender named.
fn same_bits(got: &[Scalar], want: &[Scalar], what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: lengths differ");
    for (c, (g, w)) in got.iter().zip(want).enumerate() {
        assert_eq!(
            g.to_bits(),
            w.to_bits(),
            "{what}: cell {c} is {g:e}, the whole mesh's answer is {w:e} \
             (|d| {:e})",
            (g - w).abs()
        );
    }
}

fn worst_rel(got: &[Scalar], want: &[Scalar]) -> Scalar {
    let mut worst = 0.0 as Scalar;
    for (g, w) in got.iter().zip(want) {
        let d = (g - w).abs();
        let s = w.abs().max(1e-30);
        worst = worst.max(d / s);
    }
    worst
}

fn differs(got: &[Scalar], want: &[Scalar]) -> bool {
    got.iter().zip(want).any(|(g, w)| g.to_bits() != w.to_bits())
}

// ==========================================================================
//  The pin: at one part, the gathered path IS the serial solver
// ==========================================================================

/// Every kernel argument in both recurrences, checked against a solver that is
/// already gated.
///
/// At `P = 1` with the gathered reduction, this module launches the same
/// kernels on the same buffers in the same order as `solver::solve_pcg` and
/// `solver::solve_pbicgstab`, so the answer must be **identical to the bit**.
/// Nothing weaker would catch a transposed argument: a swapped `alpha`/`omega`
/// still converges, just differently.
///
/// PBiCGStab is included even though this module takes `(t,s)` and `(t,t)` as
/// two reductions where the serial solver fuses them into `device_dot2` -
/// because `solDot2Stage1` accumulates each with the identical grid-stride
/// walk and `solSum2Stage2` combines each with the identical loop, so the two
/// constructions are bitwise equal, and this test is what proves that claim
/// rather than asserting it in a comment.
#[test]
fn a_one_part_gathered_solve_is_the_serial_solver() {
    let Some(gpu) = gpu() else { return };
    let sol = SolverKernels::new(&gpu).expect("kernels");

    for cyclic in [false, true] {
        let m = boxes([5, 4, 3], cyclic);
        let a = poisson(&m);
        let dec = Decomposition::build(&m, 1, &PartitionMethod::Linear).expect("cut");

        for (kind, precon) in [
            (LinearSolverKind::PCG, Preconditioner::Diagonal),
            (LinearSolverKind::PCG, Preconditioner::None),
            (LinearSolverKind::PCG, Preconditioner::Dic),
            (LinearSolverKind::PBiCGStab, Preconditioner::Diagonal),
            (LinearSolverKind::PBiCGStab, Preconditioner::Dilu),
        ] {
            let ctrl = controls(kind, precon, Some(9));
            let (want, sp) = serial(&gpu, &sol, &m, &a, &ctrl).expect("serial");
            let (got, dp) =
                distributed(&gpu, &sol, &m, &dec, &a, &ctrl, DistReduce::Gathered).expect("dist");

            same_bits(
                &got,
                &want,
                &format!("cyclic {cyclic}, {precon:?}, one part, gathered"),
            );
            assert_eq!(dp.n_iterations, sp.n_iterations);
            let want_ex = match kind {
                LinearSolverKind::PCG => 9 + 3,
                _ => 2 * 9 + 3,
            };
            assert_eq!(dp.n_exchanges, want_ex, "{kind:?} exchange count");
        }
    }
}

// ==========================================================================
//  The gate
// ==========================================================================

/// **The gate.** A Krylov solve over a decomposition is the undecomposed
/// solve, bit for bit, at every part count under every partitioner.
///
/// The preconditioner is one a cut cannot change (`Diagonal`, and `None`) and
/// the reduction is the exact accumulator, which is precisely the pair
/// [`DistWorkspace::partition_invariant`] answers `true` for. The iteration
/// count is compared too: a convergence test that reads a number which did not
/// move must stop on the same iteration.
///
/// The cyclic mesh is in the list deliberately - it is the only one with
/// coupled boundary faces before it is cut, so it is the only one exercising a
/// halo behind an already-coupled face.
#[test]
fn a_decomposed_krylov_solve_is_the_undecomposed_solve() {
    let Some(gpu) = gpu() else { return };
    let sol = SolverKernels::new(&gpu).expect("kernels");

    for cyclic in [false, true] {
        let m = boxes([5, 4, 3], cyclic);
        let a = poisson(&m);

        for (kind, precon) in [
            (LinearSolverKind::PCG, Preconditioner::Diagonal),
            (LinearSolverKind::PBiCGStab, Preconditioner::Diagonal),
            (LinearSolverKind::PCG, Preconditioner::None),
        ] {
            let ctrl = controls(kind, precon, None);
            let one = Decomposition::build(&m, 1, &PartitionMethod::Linear).expect("cut");
            let (want, wp) =
                distributed(&gpu, &sol, &m, &one, &a, &ctrl, DistReduce::Exact).expect("P = 1");
            assert!(wp.converged, "the reference solve must converge");
            assert!(wp.n_iterations > 3, "a one-iteration gate proves nothing");

            for np in 2..=4 {
                for (name, method) in partitions(&m, np) {
                    let dec = Decomposition::build(&m, np, &method).expect("cut");
                    assert!(
                        dec.n_cut_faces > 0,
                        "P = {np} {name} cut nothing, so the gate would be inert"
                    );
                    assert!(
                        dec.parts.iter().all(|p| p.n_halo > 0),
                        "P = {np} {name} left a part without a halo"
                    );
                    let (got, dp) = distributed(&gpu, &sol, &m, &dec, &a, &ctrl, DistReduce::Exact)
                        .expect("decomposed");
                    same_bits(
                        &got,
                        &want,
                        &format!("cyclic {cyclic}, {kind:?}/{precon:?}, P = {np}, {name}"),
                    );
                    assert_eq!(
                        dp.n_iterations, wp.n_iterations,
                        "cyclic {cyclic}, {kind:?}/{precon:?}, P = {np}, {name}: the \
                         convergence test moved"
                    );
                }
            }
        }
    }
}

/// The purest permutation there is: the same cells in the same groups with
/// only the names changed.
#[test]
fn relabelling_the_parts_changes_no_bit_of_the_solve() {
    let Some(gpu) = gpu() else { return };
    let sol = SolverKernels::new(&gpu).expect("kernels");

    let m = boxes([4, 4, 3], false);
    let a = poisson(&m);
    let ctrl = controls(LinearSolverKind::PCG, Preconditioner::Diagonal, None);

    let one = Decomposition::build(&m, 1, &PartitionMethod::Linear).expect("cut");
    let (want, _) = distributed(&gpu, &sol, &m, &one, &a, &ctrl, DistReduce::Exact).expect("P = 1");

    for np in 2..=4 {
        let base = crate::decompose::partition(&m, np, &PartitionMethod::Hilbert).expect("map");
        for shift in 0..np {
            let rotated: Vec<Label> = base
                .iter()
                .map(|&r| ((r as usize + shift) % np) as Label)
                .collect();
            let dec = Decomposition::from_map(&m, np, rotated).expect("cut");
            let (got, _) =
                distributed(&gpu, &sol, &m, &dec, &a, &ctrl, DistReduce::Exact).expect("dist");
            same_bits(&got, &want, &format!("P = {np}, rotation {shift}"));
        }
    }
}

// ==========================================================================
//  What the block-local factorisation costs
// ==========================================================================

/// The honest half. A block-local DIC or DILU **is** a different
/// preconditioner on every cut, and this asserts that it behaves like one.
///
/// Three things must hold at once, and the third is the one that stops this
/// section from claiming too much:
///
/// 1. [`DistWorkspace::partition_invariant`] answers `false`, so nothing in
///    the crate can assume otherwise;
/// 2. the answer really does move - if it did not, the gate above would be
///    passing for a preconditioner that was silently inert;
/// 3. it still solves: every cut reaches the same solution to well inside the
///    tolerance, so what moved is the path and not the answer.
///
/// The round robin at `P = 4` is the extreme case and is here for it: it
/// leaves a part with almost no interior faces at all, so its "DIC" is
/// arithmetically Jacobi, and the iteration count says so.
#[test]
fn the_block_local_factorisation_moves_the_answer_and_says_so() {
    let Some(gpu) = gpu() else { return };
    let sol = SolverKernels::new(&gpu).expect("kernels");

    let m = boxes([5, 4, 3], false);
    let a = poisson(&m);
    let ctrl = controls(LinearSolverKind::PCG, Preconditioner::Dic, None);

    let one = Decomposition::build(&m, 1, &PartitionMethod::Linear).expect("cut");
    let (want, wp) = distributed(&gpu, &sol, &m, &one, &a, &ctrl, DistReduce::Exact).expect("P = 1");
    assert!(wp.converged);

    // (1) the crate is not allowed to believe this is invariant.
    {
        let mut ex = HaloExchange::new(&gpu, &one).expect("halo");
        let sys = DistSystem::split(&gpu, &mut ex, &m, &one, &a).expect("split");
        let w = DistWorkspace::new(&gpu, &sys, DistReduce::Exact).expect("ws");
        assert!(w.partition_invariant(Preconditioner::Diagonal));
        assert!(w.partition_invariant(Preconditioner::None));
        assert!(!w.partition_invariant(Preconditioner::Dic));
        assert!(!w.partition_invariant(Preconditioner::Dilu));
        let g = DistWorkspace::new(&gpu, &sys, DistReduce::Gathered).expect("ws");
        assert!(!g.partition_invariant(Preconditioner::Diagonal));
    }

    let mut moved = 0usize;
    let mut tried = 0usize;
    for np in [2usize, 4] {
        for (name, method) in partitions(&m, np) {
            let dec = Decomposition::build(&m, np, &method).expect("cut");
            let (got, dp) =
                distributed(&gpu, &sol, &m, &dec, &a, &ctrl, DistReduce::Exact).expect("dist");
            tried += 1;
            if differs(&got, &want) {
                moved += 1;
            }
            // (3) it still solves.
            assert!(dp.converged, "P = {np} {name} did not converge");
            assert!(
                worst_rel(&got, &want) < 1e-6,
                "P = {np} {name}: block-local DIC reached a different answer, \
                 worst relative {:e}",
                worst_rel(&got, &want)
            );
        }
    }
    // (2) it really is a different preconditioner.
    assert!(
        moved > 0,
        "the block-local factorisation changed nothing in {tried} cuts, which \
         would mean the couplings it drops were never there"
    );
}

/// What is actually true about the block-local iteration count, which is less
/// than the obvious guess and was measured before it was asserted.
///
/// The obvious guess is that dropping couplings can only cost iterations, so
/// the count rises monotonically with the part count. **It does not, and this
/// test was written asserting that it did and had to be corrected.** On
/// `cases/gb_800000` the block-local DILU takes 97 iterations on the whole
/// mesh and 87 at `P = 8` - *fewer*. BiCGStab's iteration count is not a
/// monotone functional of preconditioner quality: it is a two-term recurrence
/// whose `omega` can stagnate, and a different preconditioner is a different
/// path, not a longer one. The same is true of CG in principle, though the
/// measured DIC counts do rise.
///
/// What *is* guaranteed is the bracket, and it is what is asserted:
///
/// * the block-local form still preconditions - its count stays at or below
///   the diagonal preconditioner's, because at `P = n_cells` the two are
///   arithmetically the same thing and no cut can be worse than that;
/// * it converges at every part count.
///
/// The numbers themselves are the deliverable and SPEC-LIT §73.5 publishes
/// them from `ofgpu-decompose`, on meshes large enough for a ratio to mean
/// something.
#[test]
fn a_block_local_factorisation_still_preconditions_at_every_part_count() {
    let Some(gpu) = gpu() else { return };
    let sol = SolverKernels::new(&gpu).expect("kernels");

    let m = boxes([6, 5, 4], false);
    let a = poisson(&m);
    let one = Decomposition::build(&m, 1, &PartitionMethod::Linear).expect("cut");

    for (kind, precon) in [
        (LinearSolverKind::PCG, Preconditioner::Dic),
        (LinearSolverKind::PBiCGStab, Preconditioner::Dilu),
    ] {
        let ctrl = controls(kind, precon, None);
        let (_, wp) = distributed(&gpu, &sol, &m, &one, &a, &ctrl, DistReduce::Exact).expect("1");
        assert!(wp.converged);

        // The ceiling: no cut can be worse than having no factorisation at
        // all, because the finest cut IS having none.
        let jctrl = controls(kind, Preconditioner::Diagonal, None);
        let (_, jp) = distributed(&gpu, &sol, &m, &one, &a, &jctrl, DistReduce::Exact).expect("j");
        assert!(
            wp.n_iterations <= jp.n_iterations,
            "{precon:?} on the whole mesh took {} iterations against Jacobi's \
             {} - the factorisation is not preconditioning at all",
            wp.n_iterations,
            jp.n_iterations
        );

        for np in [2usize, 4, 8] {
            let dec = Decomposition::build(&m, np, &PartitionMethod::Hilbert).expect("cut");
            let (_, dp) =
                distributed(&gpu, &sol, &m, &dec, &a, &ctrl, DistReduce::Exact).expect("dist");
            assert!(dp.converged, "{precon:?} at P = {np} did not converge");
            assert!(
                dp.n_iterations <= jp.n_iterations,
                "{precon:?} at P = {np} took {} iterations, more than Jacobi's \
                 {} - dropping couplings cannot be worse than having none",
                dp.n_iterations,
                jp.n_iterations
            );
        }
    }
}

// ==========================================================================
//  The two reduction modes, side by side
// ==========================================================================

/// The gathered construction is run-invariant and no more, and this measures
/// it inside a solver rather than on a bare array.
///
/// The exact half is the hard assertion. The gathered half is reported and
/// asserted only in the direction that cannot be flaky: over enough cuts of a
/// mesh whose dot products run to hundreds of terms, *something* must move -
/// if nothing did, the data would be too benign for the comparison to mean
/// anything, which is the trap SPEC-LIT §72.5 records falling into once.
#[test]
fn the_gathered_reduction_moves_where_the_exact_one_does_not() {
    let Some(gpu) = gpu() else { return };
    let sol = SolverKernels::new(&gpu).expect("kernels");

    let m = boxes([8, 7, 6], false);
    let a = poisson(&m);
    let ctrl = controls(LinearSolverKind::PCG, Preconditioner::Diagonal, Some(12));

    let one = Decomposition::build(&m, 1, &PartitionMethod::Linear).expect("cut");
    let (exact_one, _) =
        distributed(&gpu, &sol, &m, &one, &a, &ctrl, DistReduce::Exact).expect("exact 1");
    let (gath_one, _) =
        distributed(&gpu, &sol, &m, &one, &a, &ctrl, DistReduce::Gathered).expect("gathered 1");

    let mut gathered_moved = 0usize;
    let mut tried = 0usize;
    for np in 2..=4 {
        for (name, method) in partitions(&m, np) {
            let dec = Decomposition::build(&m, np, &method).expect("cut");
            let (e, _) =
                distributed(&gpu, &sol, &m, &dec, &a, &ctrl, DistReduce::Exact).expect("exact");
            let (g, _) = distributed(&gpu, &sol, &m, &dec, &a, &ctrl, DistReduce::Gathered)
                .expect("gathered");
            same_bits(&e, &exact_one, &format!("exact, P = {np}, {name}"));
            tried += 1;
            if differs(&g, &gath_one) {
                gathered_moved += 1;
            }
        }
    }
    assert!(
        gathered_moved > 0,
        "the gathered construction survived all {tried} cuts, which means this \
         problem cannot tell the two constructions apart and the comparison is \
         worthless here"
    );
}

/// The two shapes SPEC-LIT §73 adds to the accumulator are the crate's own
/// reductions when there is only one part.
///
/// `sum_mag` and `norm_factor` had exact twins already; their *gathered* twins
/// are new, and at `P = 1` a gathered construction is one part's own reduction
/// plus a one-element combine, which must be the identity.
#[test]
fn the_new_gathered_shapes_are_the_serial_reductions_at_one_part() {
    let Some(gpu) = gpu() else { return };
    let sol = SolverKernels::new(&gpu).expect("kernels");

    let m = boxes([5, 4, 3], false);
    let a = poisson(&m);
    let gm = GpuMesh::upload(&gpu, &m).expect("mesh");
    let ga = a.upload(&gpu).expect("matrix");
    let n = m.n_cells;

    let x: Vec<Scalar> = (0..n).map(|c| field(c) - 0.6).collect();
    let dx = gpu.upload(&x).expect("x");
    let mut w = SolverWorkspace::for_mesh(&gpu, &gm).expect("ws");
    let mut red = ExactReduction::new(&gpu, &[n]).expect("red");
    let mut out: DevBuf<Scalar> = gpu.zeros(1).expect("out");

    // sum_mag
    solver::device_sum_mag(&gpu, &sol, &mut out, &dx, &mut w.partials, n).expect("plain");
    let xs = [dx];
    red.gathered_sum_mag(&gpu, &sol, &xs).expect("gathered");
    gpu.sync().expect("sync");
    assert_eq!(
        gpu.download(&out).expect("d")[0].to_bits(),
        red.value(&gpu).expect("v").to_bits(),
        "gathered_sum_mag at one part is not device_sum_mag"
    );

    // norm_factor: the plain one leaves A.psi in w.apsi and A.xRef in w.y, so
    // the gathered twin is fed exactly the vectors the plain one reduced.
    let psi = gpu.upload(&x).expect("psi");
    solver::device_norm_factor(&gpu, &sol, &mut w, &psi, &ga, &gm).expect("nf");
    gpu.sync().expect("sync");
    let plain = gpu.download(&w.norm_factor).expect("d")[0];

    // `DevBuf::clone` in cudarc is a device-to-device copy, so these carry the
    // same bits the plain reduction just read.
    let apsi = [w.apsi.clone()];
    let axr = [w.y.clone()];
    let src = [ga.source.clone()];
    red.gathered_norm_factor(&gpu, &sol, &apsi, &src, &axr, solver::NORM_EPS)
        .expect("gathered nf");
    gpu.sync().expect("sync");
    assert_eq!(
        plain.to_bits(),
        red.value(&gpu).expect("v").to_bits(),
        "gathered_norm_factor at one part is not device_norm_factor"
    );
}

// ==========================================================================
//  Communication, counted
// ==========================================================================

/// One exchange per matrix product and no others, and the reduction count the
/// strong-scaling model of SPEC-LIT §73.6 is built on.
///
/// Both are derived in the module header and both are asserted here, because
/// an exchange that crept into an elementwise kernel would cost latency
/// forever and would never show up as a wrong answer.
#[test]
fn an_exchange_happens_once_per_matrix_product_and_no_more() {
    let Some(gpu) = gpu() else { return };
    let sol = SolverKernels::new(&gpu).expect("kernels");

    let m = boxes([4, 4, 3], false);
    let a = poisson(&m);
    let dec = Decomposition::build(&m, 3, &PartitionMethod::Hilbert).expect("cut");

    for k in [1usize, 5, 11] {
        let ctrl = controls(LinearSolverKind::PCG, Preconditioner::Diagonal, Some(k));
        let (_, p) = distributed(&gpu, &sol, &m, &dec, &a, &ctrl, DistReduce::Exact).expect("pcg");
        assert_eq!(p.n_iterations, k);
        // 2 in the normalisation, 1 per iteration, 1 in the epilogue.
        assert_eq!(p.n_exchanges, k + 3, "PCG exchanges at {k} iterations");
        // sum(psi), normFactor, sum|r|, (r,z), then 2 per iteration, then
        // sum|b - A psi|.
        assert_eq!(p.n_reductions, 2 * k + 5, "PCG reductions at {k} iterations");

        let ctrl = controls(LinearSolverKind::PBiCGStab, Preconditioner::Diagonal, Some(k));
        let (_, p) = distributed(&gpu, &sol, &m, &dec, &a, &ctrl, DistReduce::Exact).expect("bicg");
        assert_eq!(p.n_exchanges, 2 * k + 3, "PBiCGStab exchanges at {k}");
        assert_eq!(p.n_reductions, 4 * k + 4, "PBiCGStab reductions at {k}");
    }
}

// ==========================================================================
//  Refusals - SPEC-LIT section 13.4
// ==========================================================================

/// A field buffer with no room for the halo is named, with the length it
/// should have had.
#[test]
fn a_field_buffer_without_room_for_the_halo_is_refused_by_name() {
    let Some(gpu) = gpu() else { return };
    let sol = SolverKernels::new(&gpu).expect("kernels");

    let m = boxes([4, 4, 3], false);
    let a = poisson(&m);
    let dec = Decomposition::build(&m, 2, &PartitionMethod::Hilbert).expect("cut");
    let mut ex = HaloExchange::new(&gpu, &dec).expect("halo");
    let sys = DistSystem::split(&gpu, &mut ex, &m, &dec, &a).expect("split");
    let mut w = DistWorkspace::new(&gpu, &sys, DistReduce::Exact).expect("ws");

    // Owned cells only - exactly the mistake that would read past the end of
    // psi at every cut face.
    let mut psi: Vec<DevBuf<Scalar>> = sys
        .owned()
        .iter()
        .map(|&n| gpu.zeros::<Scalar>(n))
        .collect::<Result<_>>()
        .expect("alloc");
    let ctrl = controls(LinearSolverKind::PCG, Preconditioner::Diagonal, Some(2));
    let msg = err(dist_pcg(&gpu, &sol, &mut psi, &sys, &mut ex, &mut w, &ctrl));
    assert!(msg.contains("dist_pcg"), "{msg}");
    assert!(msg.contains("halo cells"), "{msg}");
    assert!(msg.contains("lduAmul"), "{msg}");
    assert!(
        msg.contains(&format!("{}", sys.local()[0])),
        "the refusal must name the length it wanted: {msg}"
    );

    // And the wrong number of buffers is its own message.
    let mut short: Vec<DevBuf<Scalar>> = vec![gpu.zeros::<Scalar>(sys.local()[0]).expect("a")];
    let msg = err(dist_pcg(&gpu, &sol, &mut short, &sys, &mut ex, &mut w, &ctrl));
    assert!(msg.contains("1 solution buffer(s) for 2 part(s)"), "{msg}");
}

/// `DIC` on a workspace that was never coloured is refused by name, with the
/// alternative and with what to do instead - SPEC-LIT §13.4.
#[test]
fn an_uncoloured_workspace_refuses_dic_by_name() {
    let Some(gpu) = gpu() else { return };

    let m = boxes([4, 4, 3], false);
    let a = poisson(&m);
    let dec = Decomposition::build(&m, 2, &PartitionMethod::Hilbert).expect("cut");
    let mut ex = HaloExchange::new(&gpu, &dec).expect("halo");
    let sys = DistSystem::split(&gpu, &mut ex, &m, &dec, &a).expect("split");
    let mut w = DistWorkspace::new(&gpu, &sys, DistReduce::Exact).expect("ws");

    for p in [Preconditioner::Dic, Preconditioner::Dilu] {
        let msg = err(w.effective_preconditioner(p));
        assert!(msg.contains("preconditioner"), "{msg}");
        assert!(msg.contains(p.name()), "{msg}");
        assert!(msg.contains("DistWorkspace::colour"), "{msg}");
    }
    assert!(w.effective_preconditioner(Preconditioner::Diagonal).is_ok());

    // And after colouring, both are available and every part reports a real
    // colour count.
    w.colour(&gpu, &sys).expect("colour");
    assert!(w.effective_preconditioner(Preconditioner::Dic).is_ok());
    assert!(w.colours().iter().all(|&c| c >= 2), "{:?}", w.colours());
}

/// `GAMG` is not a Krylov method here, and the distributed path says so with
/// the same voice the serial one uses.
#[test]
fn gamg_is_refused_by_name_on_the_distributed_path() {
    let Some(gpu) = gpu() else { return };
    let sol = SolverKernels::new(&gpu).expect("kernels");

    let m = boxes([4, 4, 3], false);
    let a = poisson(&m);
    let dec = Decomposition::build(&m, 2, &PartitionMethod::Hilbert).expect("cut");
    let mut ex = HaloExchange::new(&gpu, &dec).expect("halo");
    let sys = DistSystem::split(&gpu, &mut ex, &m, &dec, &a).expect("split");
    let mut w = DistWorkspace::new(&gpu, &sys, DistReduce::Exact).expect("ws");
    let mut psi = sys.zeros(&gpu).expect("psi");

    let mut ctrl = controls(LinearSolverKind::PCG, Preconditioner::Diagonal, Some(2));
    ctrl.solver = LinearSolverKind::Gamg;
    let msg = err(dist_solve(&gpu, &sol, &mut psi, &sys, &mut ex, &mut w, &ctrl));
    assert!(msg.contains("solver"), "{msg}");
    assert!(msg.contains("GAMG"), "{msg}");
    assert!(msg.contains("PBiCGStab"), "{msg}");
}

/// A workspace built for one decomposition, used with another, is named rather
/// than left to index past the end of something.
#[test]
fn a_workspace_from_another_decomposition_is_refused_by_name() {
    let Some(gpu) = gpu() else { return };
    let sol = SolverKernels::new(&gpu).expect("kernels");

    let m = boxes([4, 4, 3], false);
    let a = poisson(&m);
    let two = Decomposition::build(&m, 2, &PartitionMethod::Hilbert).expect("cut");
    let three = Decomposition::build(&m, 3, &PartitionMethod::Hilbert).expect("cut");

    let mut ex2 = HaloExchange::new(&gpu, &two).expect("halo");
    let mut ex3 = HaloExchange::new(&gpu, &three).expect("halo");
    let sys2 = DistSystem::split(&gpu, &mut ex2, &m, &two, &a).expect("split");
    let sys3 = DistSystem::split(&gpu, &mut ex3, &m, &three, &a).expect("split");
    let mut w2 = DistWorkspace::new(&gpu, &sys2, DistReduce::Exact).expect("ws");
    let mut psi = sys3.zeros(&gpu).expect("psi");

    let ctrl = controls(LinearSolverKind::PCG, Preconditioner::Diagonal, Some(2));
    let msg = err(dist_pcg(&gpu, &sol, &mut psi, &sys3, &mut ex3, &mut w2, &ctrl));
    assert!(msg.contains("2 part(s)") && msg.contains("3"), "{msg}");

    // And an exchange built for the wrong part count is refused at the split.
    let msg = err(DistSystem::split(&gpu, &mut ex2, &m, &three, &a));
    assert!(msg.contains("2 part(s)") && msg.contains("3"), "{msg}");
}

// ==========================================================================
//  It has to solve, not merely reproduce itself
// ==========================================================================

/// A solve that reproduced itself perfectly while computing nonsense would
/// pass every gate above. This is the one that says the answer is right.
///
/// The distributed answer is compared with the crate's own serial solver -
/// not bit for bit, because the two use different accumulators by design, but
/// to well inside the tolerance both were asked for.
#[test]
fn a_distributed_solve_reaches_the_serial_solvers_answer() {
    let Some(gpu) = gpu() else { return };
    let sol = SolverKernels::new(&gpu).expect("kernels");

    for cyclic in [false, true] {
        let m = boxes([5, 4, 3], cyclic);
        let a = poisson(&m);

        for kind in [LinearSolverKind::PCG, LinearSolverKind::PBiCGStab] {
            let ctrl = controls(kind, Preconditioner::Diagonal, None);
            let (want, sp) = serial(&gpu, &sol, &m, &a, &ctrl).expect("serial");
            assert!(sp.converged, "the serial reference must converge");

            for np in [1usize, 3, 4] {
                let dec = Decomposition::build(&m, np, &PartitionMethod::Hilbert).expect("cut");
                let (got, dp) =
                    distributed(&gpu, &sol, &m, &dec, &a, &ctrl, DistReduce::Exact).expect("dist");
                assert!(dp.converged, "cyclic {cyclic} {kind:?} P = {np}");
                let worst = worst_rel(&got, &want);
                assert!(
                    worst < 1e-6,
                    "cyclic {cyclic} {kind:?} P = {np}: worst relative \
                     difference from the serial answer is {worst:e}"
                );
                assert!(
                    dp.final_residual <= 1e-8,
                    "cyclic {cyclic} {kind:?} P = {np}: final residual \
                     {:e}",
                    dp.final_residual
                );
            }
        }
    }
}
