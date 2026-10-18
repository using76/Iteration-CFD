// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.
// Provenance: see PROVENANCE.md. No GPL-licensed source was consulted.

//! SPEC-LIT 139.3 (PAR-10b)'s gate - a part's p and momentum solves routed
//! through the SPMD Krylov, in BOTH builds.
//!
//! An integration test because the library's own test target does not
//! compile under `--features single`; an integration test links the normal
//! (non-test) library, which does. Every rank is a [`ThreadComm`] thread
//! with its own [`Gpu`] on device 0, and the reference is the WHOLE mesh
//! solved by [`solve`] on the main thread's card: a part's solve, routed
//! through [`SpmdPressureBackend`] or [`SpmdMomentumRoute`], must reach the
//! serial answer, and must report residual norms that are the world's - the
//! same bits on every rank - which is what shows the route was taken, a
//! rank-local serial solve reporting rank-local norms instead.
//!
//! The matrices are ASSEMBLED ON THE PART, never split from a whole:
//! `poisson_on` folds the part's own mesh with the cells' GLOBAL ids as the
//! per-cell arguments, so a part's matrix is the fold of its own metrics -
//! the flow path's shape, and the shape `DistSystem::for_rank` +
//! `load_matrix` exist for.
//!
//! Classes (docs/17 §5.1, §12.3 note 5): the Poisson gate is class A at
//! `1e-8` rel in BOTH builds, under `single` the refined solution
//! (SPEC-LIT 118.12) being the f64 answer; the momentum gate is class R,
//! `1e-8` in f64 and tol32 `1e-5` under `single`, the stored field being
//! float.
//!
//! Provenance: ORIGINAL - the routing under test is this project's own
//! design (SPEC-LIT 139.3, PAR-10b) over the SPMD Krylov of SPEC-LIT
//! 138.5/138.6 (PAR-04a/b). No external source. No GPL-licensed source was
//! consulted.

use ofgpu::blockgen::{build_case, CaseKind};
use ofgpu::comm::{Comm, ThreadComm};
use ofgpu::decompose::{Decomposition, PartitionMethod};
use ofgpu::distflow::{RankCtx, SpmdKrylov, SpmdMomentumRoute, SpmdPressureBackend};
use ofgpu::distsolve::{DistSystem, DistReduce};
use ofgpu::field_ops::{correct_boundary_conditions_vector, FieldKernels};
use ofgpu::halo::SpmdHalo;
use ofgpu::io::case::{DivScheme, GamgControls};
use ofgpu::fv::SnGradScheme;
use ofgpu::ldu::HostLduMatrix;
use ofgpu::mesh::{GpuMesh, HostMesh};
use ofgpu::momentum::{BuoyancyCoeffs, ComponentRoute, Momentum, MomentumControls};
use ofgpu::pressure::{PressureBackend, SystemProbe};
use ofgpu::solver::{
    solve, LinearSolverKind, Preconditioner, SolverControls, SolverKernels, SolverWorkspace,
};
use ofgpu::decompose::PartMesh;
use ofgpu::{Gpu, GpuScalarField, GpuSurfaceScalarField, GpuVectorField, Result, Scalar, Vec3};

/// A machine without a card makes every device test pass vacuously, the
/// convention the rest of the crate's tests follow.
fn gpu() -> Option<Gpu> {
    Gpu::new(0).ok()
}

/// Mesh A: an axis-aligned (orthogonal) box.
fn mesh_a() -> HostMesh {
    build_case(CaseKind::Big, 12, 10, 8).expect("mesh A").0
}

/// Mesh B: a non-orthogonal, skewed unit cube.
fn mesh_b() -> HostMesh {
    ofgpu::solid::prototype::jittered_block(10, 0.25).expect("mesh B")
}

/// The round robin: cell `c` to part `c % P`, the worst partition possible.
fn round_robin(n: usize, p: usize) -> PartitionMethod {
    PartitionMethod::Explicit((0..n).map(|c| (c % p) as ofgpu::Label).collect())
}

/// The cuts of test 1: `P = 1` with the linear cut alone, then Hilbert,
/// Linear and round robin.
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

/// A deterministic Dirichlet value per boundary cell, of the GLOBAL id.
fn wall(g: usize) -> Scalar {
    0.5 + 0.125 * ((g * 11) % 7) as Scalar
}

/// The deterministic right-hand side per cell, of the GLOBAL id.
fn rhs(g: usize) -> Scalar {
    1.0 + 0.0625 * ((g * 37) % 23) as Scalar
}

/// The SPD Poisson operator from the mesh's own metrics, folded on the host,
/// WITHOUT the ddt term of `tests/distsolve_spmd.rs`'s fixture: internal
/// faces `g = mag_sf*delta_coeffs`, `upper = lower = -g`, both cells' diag
/// `+= g`; boundary faces `g = b_mag_sf*b_delta_coeffs`, `diag[c] += g`,
/// `boundary_coeffs = g` on a coupled face else `source[c] += g*wall(gid)`;
/// and `source[c] += v[c]*rhs(gid(c))`. `gid` is the GLOBAL cell id of a
/// LOCAL cell, so the whole mesh folds with the identity and a part folds
/// through `part.global_cell` - the part-assembled matrix, assembled on the
/// part and never split from a whole.
///
/// Every coefficient is accumulated in f64 and rounded ONCE at the end. In
/// the f64 build this is the same arithmetic as before (Scalar is f64), so
/// the f64 numbers are unchanged. WHY: under `single` a float running sum
/// makes the part's `diag`/`source` differ from the whole fold's by its
/// SUMMATION ORDER - a part adds a processor face in its boundary loop,
/// after its internal faces, where the whole adds it inside the internal
/// loop - and two float systems that differ by an ulp answer two different
/// questions. That assembly-order gap is PAR-13a's subject, not what this
/// gate measures; one rounding of an f64 sum makes both folds the same
/// float system up to a rounding tie.
fn poisson_on(m: &HostMesh, gid: impl Fn(usize) -> usize) -> HostLduMatrix {
    let mut a = HostLduMatrix::zeros(m);
    let mut d64 = vec![0.0f64; m.n_cells];
    let mut s64 = vec![0.0f64; m.n_cells];
    for f in 0..m.n_internal_faces {
        let g = m.mag_sf[f] as f64 * m.delta_coeffs[f] as f64;
        a.upper[f] = -(g as Scalar);
        a.lower[f] = -(g as Scalar);
        d64[m.owner[f] as usize] += g;
        d64[m.neighbour[f] as usize] += g;
    }
    for bf in 0..m.n_boundary_faces {
        let g = m.b_mag_sf[bf] as f64 * m.b_delta_coeffs[bf] as f64;
        let c = m.b_face_cells[bf] as usize;
        d64[c] += g;
        if m.b_nbr_cell[bf] >= 0 {
            a.boundary_coeffs[bf] = g as Scalar;
        } else {
            s64[c] += g * wall(gid(c)) as f64;
        }
    }
    for c in 0..m.n_cells {
        s64[c] += m.v[c] as f64 * rhs(gid(c)) as f64;
    }
    for c in 0..m.n_cells {
        a.diag[c] = d64[c] as Scalar;
        a.source[c] = s64[c] as Scalar;
    }
    a
}

/// The controls: tolerance `1e-12` in BOTH builds - a solver control and
/// NOT a gate tolerance; the gate below is on the answer, at `1e-8` rel.
fn controls(solver: LinearSolverKind, precon: Preconditioner) -> SolverControls {
    SolverControls {
        solver,
        precon,
        tolerance: 1e-12,
        rel_tol: 0.0,
        max_iter: 2000,
        min_iter: 0,
        check_interval: 1,
        fixed_iters: false,
        report_residuals: true,
        gamg: GamgControls::default(),
    }
}

/// The flag votes a solve takes at check interval 1: `n_iterations + 1`
/// unrefined. Under `single` the solve is refined (SPEC-LIT 118.12): each of
/// the `outer` inner solves votes once before it iterates and once per
/// iteration, and each of the `outer + 1` outer verdicts is voted too -
/// `n_iterations + 2 outer + 1`, the unrefined count at `outer = 0`.
fn expected_votes(n_iterations: usize, outer: usize) -> u64 {
    (n_iterations + 2 * outer + 1) as u64
}

/// The outer refinement steps of the route's last solve: 0 in the f64
/// build, which does not refine.
#[cfg(feature = "single")]
fn outer_steps(k: &SpmdKrylov) -> usize {
    k.workspace().ir_outer_steps()
}

#[cfg(not(feature = "single"))]
fn outer_steps(_k: &SpmdKrylov) -> usize {
    0
}

/// A rank's owned answer, in f64: under `single` the refined solution
/// (SPEC-LIT 118.12) is the answer - precision class A - and its first
/// `n_owned` entries are the owned cells; in f64 it is `psi` itself.
#[cfg(feature = "single")]
fn owned_answer(gpu: &Gpu, k: &SpmdKrylov, psi: &ofgpu::DevBuf<Scalar>, n_owned: usize) -> Vec<f64> {
    if let Some(hi) = k.refined_solution() {
        return gpu.download(hi).expect("download x_hi")[..n_owned].to_vec();
    }
    gpu.download(psi)
        .expect("download psi")[..n_owned]
        .iter()
        .map(|&v| v as f64)
        .collect()
}

#[cfg(not(feature = "single"))]
fn owned_answer(gpu: &Gpu, _k: &SpmdKrylov, psi: &ofgpu::DevBuf<Scalar>, n_owned: usize) -> Vec<f64> {
    gpu.download(psi)
        .expect("download psi")[..n_owned]
        .iter()
        .map(|&v| v as f64)
        .collect()
}

/// The whole mesh's reference answer, in f64: under `single` the refined
/// solution's first `n` entries (class A of docs/17 §5.1, evaluated in f64
/// at the f64 tolerance); in f64 it is `psi`.
#[cfg(feature = "single")]
fn serial_answer(gpu: &Gpu, _psi: &ofgpu::DevBuf<Scalar>, ws: &SolverWorkspace, n: usize) -> Vec<f64> {
    let hi = ws
        .refined_solution()
        .expect("a non-fixed_iters solve under single is refined");
    gpu.download(hi).expect("download x_hi")[..n].to_vec()
}

#[cfg(not(feature = "single"))]
fn serial_answer(gpu: &Gpu, psi: &ofgpu::DevBuf<Scalar>, _ws: &SolverWorkspace, _n: usize) -> Vec<f64> {
    gpu.download(psi)
        .expect("download psi")
        .iter()
        .map(|&v| v as f64)
        .collect()
}

/// The test velocity, a function of the WHOLE mesh's cell centres so that
/// splitting the mesh is a permutation and not a computation - the three
/// exponentials of `tests/distflow.rs`:
///
/// `U = (exp(3 xh + 2 yh + 2.5 zh), exp(-2 xh + 3 yh + 2 zh),
/// exp(2 xh + 2 yh - 3 zh))`, `xh = (x - xmin)/(xmax - xmin)` and friends.
fn smooth_u(m: &HostMesh) -> Vec<Vec3> {
    let (mut lo, mut hi) = (m.c[0], m.c[0]);
    for c in &m.c {
        lo.x = lo.x.min(c.x);
        lo.y = lo.y.min(c.y);
        lo.z = lo.z.min(c.z);
        hi.x = hi.x.max(c.x);
        hi.y = hi.y.max(c.y);
        hi.z = hi.z.max(c.z);
    }
    m.c.iter()
        .map(|&c| {
            let (xh, yh, zh) = (
                (c.x - lo.x) / (hi.x - lo.x),
                (c.y - lo.y) / (hi.y - lo.y),
                (c.z - lo.z) / (hi.z - lo.z),
            );
            Vec3::new(
                (3.0 * xh + 2.0 * yh + 2.5 * zh).exp(),
                (-2.0 * xh + 3.0 * yh + 2.0 * zh).exp(),
                (2.0 * xh + 2.0 * yh - 3.0 * zh).exp(),
            )
        })
        .collect()
}

/// One part's local buffer: owned cells from the global field, through
/// `global_cell`; the halo ZERO.
fn local_owned(part: &PartMesh, us: &[Vec3]) -> Vec<Vec3> {
    let mut h = vec![Vec3::ZERO; part.n_local()];
    for (i, hi) in h.iter_mut().enumerate().take(part.mesh.n_cells) {
        *hi = us[part.global_cell[i] as usize];
    }
    h
}

/// What one rank returns from a routed solve.
struct RankOut {
    ids: Vec<usize>,
    x: Vec<f64>,
    initial: Scalar,
    final_res: Scalar,
    n_iterations: usize,
    converged: bool,
    votes: u64,
    outer: usize,
}

/// The message of an `Err` a call must return - the refusals under test
/// return non-`Debug` handles, so `unwrap_err` is not available.
fn err_msg<T>(r: ofgpu::Result<T>, tag: &str) -> String {
    match r {
        Err(e) => format!("{e}"),
        Ok(_) => panic!("{tag}: the call was expected to refuse"),
    }
}

/// **The gate.** A part-assembled SPD Poisson solved by
/// [`SpmdPressureBackend`] on P = 1, 2, 4 ranks reaches the serial answer:
/// every owned cell within `1e-8` rel of the whole-mesh solve in BOTH
/// builds (class A - under `single` both sides are the refined f64
/// solutions), every rank converged, and `initial_residual`, `final_residual`
/// and `n_iterations` the same bits on every rank - a rank-local serial
/// solve would norm its own cells only, so identical norms everywhere is
/// what shows the route was taken. The flag vote counts
/// `n_iterations + 2 outer + 1` reads on every rank, and the parts' owned
/// cells cover the mesh exactly once.
#[test]
fn a_part_assembled_poisson_reaches_the_serial_answer() {
    let Some(gpu0) = gpu() else { return };
    let sol0 = SolverKernels::new(&gpu0).expect("kernels");
    let mut max_rel = 0.0f64;

    for (name, hm) in [("A 12x10x8 box", mesh_a()), ("B jittered_block(10, 0.25)", mesh_b())] {
        let n = hm.n_cells;
        let a_whole = poisson_on(&hm, |g| g);
        for (solver, precon) in [
            (LinearSolverKind::PCG, Preconditioner::Dic),
            (LinearSolverKind::PCG, Preconditioner::Diagonal),
            (LinearSolverKind::PBiCGStab, Preconditioner::Dilu),
        ] {
            let ctrl = controls(solver, precon);
            let tag = format!("{name} {solver:?}/{precon:?}");

            // ---- the serial reference: the whole fold, on the main card ---
            let (serial, serial_iters) = {
                let gm0 = GpuMesh::upload(&gpu0, &hm).expect("upload");
                let a_dev = a_whole.upload(&gpu0).expect("upload A");
                let mut psi = gpu0.zeros::<Scalar>(n).expect("psi");
                let mut ws = SolverWorkspace::for_mesh(&gpu0, &gm0).expect("ws");
                let p = solve(&gpu0, &sol0, &mut psi, &a_dev, &gm0, &mut ws, &ctrl)
                    .unwrap_or_else(|e| panic!("{tag}: the serial reference failed: {e}"));
                gpu0.sync().expect("sync");
                assert!(p.converged, "{tag}: the serial reference must converge");
                (serial_answer(&gpu0, &psi, &ws, n), p.n_iterations)
            };
            let scale = serial.iter().fold(0.0f64, |m, &v| m.max(v.abs()));
            assert!(scale > 0.0, "{tag}: a degenerate reference scale");

            for np in [1usize, 2, 4] {
                for (pname, dec) in cuts(&hm, np) {
                    let got = ThreadComm::run(np, |comm| -> Result<RankOut> {
                        let gpu = Gpu::new(0)?;
                        let part = &dec.parts[comm.rank()];
                        let mut be = SpmdPressureBackend::new(
                            &gpu, comm, &dec, ctrl, DistReduce::Exact,
                        )?;
                        be.krylov_mut().set_flag_vote(true);
                        let gm = GpuMesh::upload_with_halo(&gpu, &part.mesh, part.n_halo)?;
                        be.setup(&gpu, &part.mesh, &gm, &SystemProbe::default())?;
                        // Assembled ON THE PART, global ids as the fold's
                        // per-cell arguments.
                        let a_part = poisson_on(&part.mesh, |c| part.global_cell[c] as usize);
                        let a_dev = a_part.upload(&gpu)?;
                        let mut psi = gpu.zeros::<Scalar>(part.n_local())?;
                        let perf = be.solve(&gpu, &mut psi, &a_dev, &gm)?;
                        gpu.sync()?;
                        let k = be.krylov();
                        Ok(RankOut {
                            ids: part.global_cell[..part.mesh.n_cells]
                                .iter()
                                .map(|&g| g as usize)
                                .collect(),
                            x: owned_answer(&gpu, k, &psi, part.mesh.n_cells),
                            initial: perf.initial_residual,
                            final_res: perf.final_residual,
                            n_iterations: perf.n_iterations,
                            converged: perf.converged,
                            votes: k.workspace().n_flag_votes(),
                            outer: outer_steps(k),
                        })
                    })
                    .unwrap_or_else(|e| panic!("{tag} P={np} {pname}: {e}"));
                    assert_eq!(got.len(), np, "{tag} P={np} {pname}: ranks");

                    let mut case_rel = 0.0f64;
                    let mut seen = vec![0u64; n];
                    let first = got[0]
                        .as_ref()
                        .unwrap_or_else(|e| panic!("{tag} P={np} {pname} rank 0: {e}"));
                    for (r, part) in dec.parts.iter().enumerate() {
                        let o = got[r]
                            .as_ref()
                            .unwrap_or_else(|e| panic!("{tag} P={np} {pname} rank {r}: {e}"));
                        let at = format!("{tag} P={np} {pname} rank {r}");
                        assert!(o.converged, "{at}: the routed solve must converge");
                        assert_eq!(o.ids.len(), part.mesh.n_cells, "{at}: owned ids");
                        assert_eq!(o.x.len(), part.mesh.n_cells, "{at}: owned answer");
                        for (i, &g) in o.ids.iter().enumerate() {
                            seen[g] += 1;
                            let rel = (o.x[i] as f64 - serial[g] as f64).abs() / scale;
                            case_rel = case_rel.max(rel);
                        }
                        // The reported norms are the WORLD's: the same bits
                        // on every rank.
                        assert_eq!(
                            o.initial.to_bits(),
                            first.initial.to_bits(),
                            "{at}: initial residual bits differ from rank 0"
                        );
                        assert_eq!(
                            o.final_res.to_bits(),
                            first.final_res.to_bits(),
                            "{at}: final residual bits differ from rank 0"
                        );
                        assert_eq!(o.n_iterations, first.n_iterations, "{at}: iterations");
                        assert_eq!(
                            o.votes,
                            expected_votes(o.n_iterations, o.outer),
                            "{at}: flag votes"
                        );
                    }
                    for (g, c) in seen.iter().enumerate() {
                        assert_eq!(*c, 1, "{tag} P={np} {pname}: global cell {g} covered {c} times");
                    }
                    assert!(
                        case_rel <= 1e-8,
                        "{tag} P={np} {pname}: rel {case_rel:.3e} > 1e-8"
                    );
                    // Print-only diagnostic, no assertion: folding each part
                    // on the HOST, how many owned rows' diag or source differ
                    // BITWISE from the whole fold's row at the global cell -
                    // what is left of the assembly-order gap after the
                    // fixture's one f64 rounding (PAR-13a's item).
                    let mut diff_rows = 0usize;
                    for part in &dec.parts {
                        let ap = poisson_on(&part.mesh, |c| part.global_cell[c] as usize);
                        for c in 0..part.mesh.n_cells {
                            let g = part.global_cell[c] as usize;
                            if ap.diag[c].to_bits() != a_whole.diag[g].to_bits()
                                || ap.source[c].to_bits() != a_whole.source[g].to_bits()
                            {
                                diff_rows += 1;
                            }
                        }
                    }
                    println!(
                        "  {tag} P={np} {pname}: rel {case_rel:.3e}, \
                         iters {} (serial {}), \
                         rows differing from the whole fold: {diff_rows}",
                        first.n_iterations, serial_iters
                    );
                    max_rel = max_rel.max(case_rel);
                }
            }
        }
    }
    println!("  max rel over every case: {max_rel:.3e}");
}

/// The gate tolerance of the momentum route: `1e-8` in f64, and under
/// `single` class R of the docs/17 plan's precision rule (its R class:
/// tol32 = max(tol64, min(tol64 2^29, 1e-5)) = 1e-5), the stored field
/// being float.
#[cfg(feature = "single")]
const TOL_U: f64 = 1e-5;
#[cfg(not(feature = "single"))]
const TOL_U: f64 = 1e-8;

/// The momentum predictor's three component solves go through the SPMD
/// route: each component's `n_iterations > 0` and its `initial_residual`,
/// `final_residual` and `n_iterations` are the same bits on every rank - a
/// rank-local serial solve would report rank-local norms - and the owned
/// velocity reaches the whole mesh's within [`TOL_U`] rel.
#[test]
fn a_part_momentum_solve_goes_through_the_spmd_route() {
    let Some(gpu0) = gpu() else { return };
    let hm = mesh_a();
    let n = hm.n_cells;
    let us = smooth_u(&hm);
    let ctrl = MomentumControls {
        nu: 0.01,
        u_solver: controls(LinearSolverKind::PBiCGStab, Preconditioner::Dilu),
        u_relax: 0.7,
        div_scheme: DivScheme::Upwind,
        bounded_convection: false,
        sn_grad: SnGradScheme::Uncorrected,
        variable_viscosity_stress: false,
        ..Default::default()
    };

    // ---- the whole mesh on the main thread --------------------------------
    let whole_u = {
        let gm0 = GpuMesh::upload(&gpu0, &hm).expect("upload");
        let fk0 = FieldKernels::new(&gpu0).expect("fk");
        let mut u = GpuVectorField::zeros(&gpu0, &gm0, "U").expect("U");
        gpu0.write(&mut u.f, &us).expect("write U");
        correct_boundary_conditions_vector(&gpu0, &fk0, &mut u, &gm0).expect("correct U");
        let phi = GpuSurfaceScalarField::zeros(&gpu0, &gm0, "phi").expect("phi");
        let nut = GpuScalarField::zeros(&gpu0, &gm0, "nut").expect("nut");
        let mut mom =
            Momentum::new(&gpu0, &gm0, ctrl, BuoyancyCoeffs::default()).expect("momentum");
        assert!(!mom.has_route(), "the serial momentum must not be routed");
        let perfs = mom
            .solve(&gpu0, &mut u, &phi, &nut)
            .expect("serial momentum solve");
        gpu0.sync().expect("sync");
        assert!(perfs.iter().all(|p| p.n_iterations > 0));
        gpu0.download(&u.f).expect("download U")
    };
    let u_scale = whole_u.iter().fold(0.0f64, |m, v| {
        m.max(v.x.abs() as f64)
            .max(v.y.abs() as f64)
            .max(v.z.abs() as f64)
    });
    assert!(u_scale > 0.0);

    // ---- the routed run: one thread per rank ------------------------------
    let mut max_rel = 0.0f64;
    for np in [2usize, 4] {
        for (pname, method) in [
            ("hilbert", PartitionMethod::Hilbert),
            ("roundrobin", round_robin(n, np)),
        ] {
            let dec = Decomposition::build(&hm, np, &method).expect("cut");
            let got = ThreadComm::run(np, |comm| -> Result<(Vec<usize>, Vec<Vec3>, [(Scalar, Scalar, usize); 3])> {
                let gpu = Gpu::new(0)?;
                let part = &dec.parts[comm.rank()];
                let m = GpuMesh::upload_with_halo(&gpu, &part.mesh, part.n_halo)?;
                let mut ctx = RankCtx::new(&gpu, comm, part, &m)?;
                let fk = FieldKernels::new(&gpu)?;

                // Owned U only, ghosts zero, then the exchange fills them
                // and re-evaluates every boundary face.
                let mut u = GpuVectorField::zeros(&gpu, &m, "U")?;
                gpu.write(&mut u.f, &local_owned(part, &us))?;
                ctx.exchange_field_vector(&gpu, &fk, &mut u, &m)?;

                let phi = GpuSurfaceScalarField::zeros(&gpu, &m, "phi")?;
                let nut = GpuScalarField::zeros(&gpu, &m, "nut")?;
                let mut mom = Momentum::new(&gpu, &m, ctrl, BuoyancyCoeffs::default())?;
                assert!(!mom.has_route(), "a fresh momentum must be unrouted");
                mom.set_route(Some(Box::new(SpmdMomentumRoute::new(
                    &gpu, comm, &dec, DistReduce::Exact,
                )?)));
                assert!(mom.has_route(), "the route did not take");
                let perfs = mom.solve(&gpu, &mut u, &phi, &nut)?;
                gpu.sync()?;
                let uf: Vec<Vec3> = gpu.download(&u.f)?;
                let ids = part.global_cell[..part.mesh.n_cells]
                    .iter()
                    .map(|&g| g as usize)
                    .collect();
                let norms = perfs.map(|p| (p.initial_residual, p.final_residual, p.n_iterations));
                Ok((ids, uf, norms))
            })
            .unwrap_or_else(|e| panic!("momentum P={np} {pname}: {e}"));
            assert_eq!(got.len(), np, "momentum P={np} {pname}: ranks");

            let mut case_rel = 0.0f64;
            let first = got[0]
                .as_ref()
                .unwrap_or_else(|e| panic!("momentum P={np} {pname} rank 0: {e}"));
            for r in 0..np {
                let (ids, uf, norms) = got[r]
                    .as_ref()
                    .unwrap_or_else(|e| panic!("momentum P={np} {pname} rank {r}: {e}"));
                let at = format!("momentum P={np} {pname} rank {r}");
                for (c, (ri, rf, ni)) in norms.iter().enumerate() {
                    assert!(*ni > 0, "{at}: component {c} did no iteration");
                    let (r0i, r0f, n0) = first.2[c];
                    assert_eq!(
                        ri.to_bits(),
                        r0i.to_bits(),
                        "{at}: component {c} initial residual bits differ from rank 0"
                    );
                    assert_eq!(
                        rf.to_bits(),
                        r0f.to_bits(),
                        "{at}: component {c} final residual bits differ from rank 0"
                    );
                    assert_eq!(*ni, n0, "{at}: component {c} iterations differ from rank 0");
                }
                for (i, &g) in ids.iter().enumerate() {
                    let w = whole_u[g];
                    let p = uf[i];
                    for (a, b) in [(p.x, w.x), (p.y, w.y), (p.z, w.z)] {
                        case_rel = case_rel.max((a as f64 - b as f64).abs() / u_scale);
                    }
                }
            }
            for (c, (_, _, ni)) in first.2.iter().enumerate() {
                assert!(*ni > 0, "component {c} did no iteration");
            }
            assert!(
                case_rel <= TOL_U,
                "momentum P={np} {pname}: rel {case_rel:.3e} > {TOL_U}"
            );
            println!(
                "  momentum A P={np} {pname}: rel {case_rel:.3e}, iters \
                 [{}, {}, {}] (tol {TOL_U})",
                first.2[0].2, first.2[1].2, first.2[2].2
            );
            max_rel = max_rel.max(case_rel);
        }
    }
    println!("  momentum max rel: {max_rel:.3e}");
}

/// The refusal table, by name, on a Linear 2-part cut of a small box: GAMG
/// as the pressure controls' solver or preconditioner, a `setup` on the
/// whole mesh, a world smaller than the decomposition, a `load_matrix` of
/// the whole mesh's fold, and GAMG through the momentum route - each
/// message starting where it must and naming what it must.
#[test]
fn the_spmd_routes_refuse_by_name() {
    let Some(_gpu0) = gpu() else { return };
    let hm = build_case(CaseKind::Big, 5, 4, 3).expect("mesh").0;
    let d2 = Decomposition::build(&hm, 2, &PartitionMethod::Linear).expect("cut 2");
    let d3 = Decomposition::build(&hm, 3, &PartitionMethod::Linear).expect("cut 3");
    let a_whole = poisson_on(&hm, |g| g);

    let got = ThreadComm::run(2, |comm| -> Result<usize> {
        let gpu = Gpu::new(0)?;
        let rank = comm.rank();
        let part = &d2.parts[rank];
        let mut checked = 0usize;

        // (a) GAMG as the pressure controls' solver.
        {
            let ctrl = controls(LinearSolverKind::Gamg, Preconditioner::Diagonal);
            let msg = err_msg(
                SpmdPressureBackend::new(&gpu, comm, &d2, ctrl, DistReduce::Exact),
                "(a)",
            );
            assert!(msg.starts_with("distflow: "), "(a): {msg}");
            assert!(msg.contains("GAMG"), "(a): {msg}");
            checked += 1;
        }

        // (b) GAMG as the preconditioner.
        {
            let ctrl = controls(LinearSolverKind::PCG, Preconditioner::Gamg);
            let msg = err_msg(
                SpmdPressureBackend::new(&gpu, comm, &d2, ctrl, DistReduce::Exact),
                "(b)",
            );
            assert!(msg.starts_with("distflow: "), "(b): {msg}");
            assert!(msg.contains("GAMG"), "(b): {msg}");
            checked += 1;
        }

        // (c) A good backend, set up on the WHOLE host mesh.
        {
            let ctrl = controls(LinearSolverKind::PCG, Preconditioner::Diagonal);
            let mut be =
                SpmdPressureBackend::new(&gpu, comm, &d2, ctrl, DistReduce::Exact).expect("be");
            let gm = GpuMesh::upload_with_halo(&gpu, &part.mesh, part.n_halo)?;
            let msg = err_msg(
                be.setup(&gpu, &hm, &gm, &SystemProbe::default()),
                "(c)",
            );
            assert!(msg.starts_with("distflow: "), "(c): {msg}");
            assert!(msg.contains("setup"), "(c): {msg}");
            assert!(
                msg.contains(&hm.n_cells.to_string()),
                "(c): the whole count is missing: {msg}"
            );
            assert!(
                msg.contains(&part.mesh.n_cells.to_string()),
                "(c): the owned count is missing: {msg}"
            );
            checked += 1;
        }

        // (d) `for_rank` on a decomposition with more parts than ranks.
        {
            let msg = err_msg(DistSystem::for_rank(&gpu, comm, &d3), "(d)");
            assert!(msg.starts_with("distsolve: "), "(d): {msg}");
            assert!(msg.contains('3'), "(d): {msg}");
            checked += 1;
        }

        // (e) `for_rank`, then `load_matrix` with the WHOLE mesh's fold.
        {
            let mut halo = SpmdHalo::new(&gpu, part, comm.size())?;
            let mut sys = DistSystem::for_rank(&gpu, comm, &d2)?;
            let a_dev = a_whole.upload(&gpu)?;
            let msg = err_msg(
                sys.load_matrix(&gpu, &mut halo, comm, &a_dev),
                "(e)",
            );
            assert!(msg.starts_with("distsolve: "), "(e): {msg}");
            assert!(
                msg.contains(&hm.n_cells.to_string()),
                "(e): the whole count is missing: {msg}"
            );
            assert!(
                msg.contains(&part.mesh.n_cells.to_string()),
                "(e): the owned count is missing: {msg}"
            );
            checked += 1;
        }

        // (f) The momentum route, asked for GAMG.
        {
            let mut route = SpmdMomentumRoute::new(&gpu, comm, &d2, DistReduce::Exact)?;
            let ctrl = controls(LinearSolverKind::Gamg, Preconditioner::Diagonal);
            let gm = GpuMesh::upload_with_halo(&gpu, &part.mesh, part.n_halo)?;
            let a_dev = a_whole.upload(&gpu)?;
            let mut psi = gpu.zeros::<Scalar>(part.n_local())?;
            let msg = err_msg(
                route.solve_component(&gpu, &mut psi, &a_dev, &gm, &ctrl),
                "(f)",
            );
            assert!(msg.starts_with("distflow: "), "(f): {msg}");
            assert!(msg.contains("GAMG"), "(f): {msg}");
            checked += 1;
        }

        Ok(checked)
    })
    .expect("spmd run");
    for (r, res) in got.iter().enumerate() {
        let c = res.as_ref().unwrap_or_else(|e| panic!("rank {r}: {e}"));
        assert_eq!(*c, 6, "rank {r} checked {c} refusals, not 6");
    }
}
