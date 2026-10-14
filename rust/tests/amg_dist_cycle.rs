// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.
// Provenance: see PROVENANCE.md. No GPL-licensed source was consulted.

//! SPEC-LIT §141.11's gates (AMG-07b) - the distributed GAMG over SPMD
//! ranks, in BOTH builds.
//!
//! An integration test because the library's own test target does not
//! compile under `--features single`. Every rank is a [`ThreadComm`] thread
//! with its own [`Gpu`] on device 0 (multi-GPU runs are deferred by the
//! user). The reference is the serial GAMG of §141.5 on the whole mesh:
//! GAMG-PCG for `spmd_pcg` with `Preconditioner::Gamg`, and the stationary
//! cycle for `spmd_solve` with `LinearSolverKind::Gamg`. The answer gate is
//! class R of the docs/17 precision rules, `1e-8` relative, the iteration
//! gate a count (`<= 1.3x` serial). Since AMG-06a (SPEC-LIT §141.9) both
//! numbers hold in BOTH builds: under `single` every solve here, the
//! stationary `spmd_gamg` included, is refined (§118.12), and the answer
//! compared is the refinement's double solution.
//!
//! Provenance: ORIGINAL - the gates of this project's own design (SPEC-LIT
//! §141.11). No external source. No GPL-licensed source was consulted.

use ofgpu::amg::{DistGamg, Gamg, GamgConfig, HierarchyConfig};
use ofgpu::blockgen::{build_mesh, BlockSpec, GradedAxis};
use ofgpu::comm::{Comm, ThreadComm};
use ofgpu::decompose::{Decomposition, PartitionMethod};
use ofgpu::device::Gpu;
use ofgpu::distsolve::{
    dist_solve, spmd_pcg, spmd_solve, DistPerformance, DistReduce, DistSystem, DistWorkspace,
    SpmdWorkspace,
};
use ofgpu::halo::{HaloExchange, SpmdHalo};
use ofgpu::ldu::HostLduMatrix;
use ofgpu::mesh::{GpuMesh, HostMesh};
use ofgpu::solver::{
    solve, LinearSolverKind, Preconditioner, SolverControls, SolverKernels, SolverPerformance,
    SolverWorkspace,
};
use ofgpu::{Result, Scalar};

/// The answer gate, class R: `1e-8` relative, in both builds - under
/// `single` the refined solves reach the f64 tolerance (docs/17 §12 note 4,
/// AMG-06a; it was `tol32 = 1e-5` before).
const GATE: f64 = 1e-8;

/// The solver tolerance, a solver control: `1e-11` in both builds (it was
/// `1e-5` under `single` while `spmd_gamg` was a float solve that stalls
/// near the f32 floor).
const TOL: Scalar = 1e-11;

/// A uniform box, every patch a wall, the axes of `cyclic` cyclic.
fn box_mesh(nx: usize, ny: usize, nz: usize, cyclic: &[usize]) -> HostMesh {
    let axis = |n: usize| GradedAxis {
        lo: 0.0,
        hi: n as Scalar,
        n,
        expansion: 1.0,
        two_sided: false,
    };
    let mut b = BlockSpec {
        x: axis(nx),
        y: axis(ny),
        z: axis(nz),
        windows: Vec::new(),
        patch_name: BlockSpec::default().patch_name,
        patch_type: ["wall"; 6].map(String::from),
        cyclic: Vec::new(),
    };
    for &a in cyclic {
        b.set_cyclic_axis(a).expect("cyclic axis");
    }
    build_mesh(&b).expect("box mesh")
}

/// The §2 laplacian negated (`sigma = +1`): `upper = lower = -|S_f|
/// delta_f`, a wall folded into the diagonal (Dirichlet zero), a cyclic
/// couple as `boundary_coeffs = |S_b| delta_b`, and `b = 1 + (c mod 7)/7`.
fn poi(m: &HostMesh) -> HostLduMatrix {
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
        a.diag[m.b_face_cells[bf] as usize] += g;
        if m.b_nbr_cell[bf] >= 0 {
            a.boundary_coeffs[bf] = g;
        }
    }
    a.source = (0..m.n_cells)
        .map(|c| 1.0 + (c % 7) as Scalar / 7.0)
        .collect();
    a
}

fn controls(solver: LinearSolverKind, precon: Preconditioner) -> SolverControls {
    SolverControls {
        solver,
        precon,
        tolerance: TOL,
        rel_tol: 0.0,
        max_iter: 500,
        min_iter: 0,
        check_interval: 1,
        fixed_iters: false,
        report_residuals: true,
    }
}

/// The serial GAMG of §141.5 on the whole mesh, default config, from zero.
fn serial(
    gpu: &Gpu,
    k: &SolverKernels,
    m: &HostMesh,
    a: &HostLduMatrix,
    ctrl: &SolverControls,
) -> Result<(SolverPerformance, Vec<f64>)> {
    let mesh = GpuMesh::upload(gpu, m)?;
    let mat = a.upload(gpu)?;
    let mut w = SolverWorkspace::for_mesh(gpu, &mesh)?;
    w.set_gamg(Gamg::new(gpu, m, a, &GamgConfig::default(), false)?)?;
    let mut psi = gpu.zeros(m.n_cells)?;
    let p = solve(gpu, k, &mut psi, &mat, &mesh, &mut w, ctrl)?;
    #[cfg(feature = "single")]
    if w.ir_outer_steps() > 0 {
        if let Some(x) = w.refined_solution() {
            return Ok((p, gpu.download(x)?[..m.n_cells].to_vec()));
        }
    }
    Ok((p, gpu.download(&psi)?.iter().map(|&v| v as f64).collect()))
}

/// What one rank brings back: per control its performance and owned field,
/// the cycle's owned and whole-level sizes, and the workspace's
/// `partition_invariant(Gamg)`.
struct RankOut {
    solves: Vec<(DistPerformance, Vec<f64>)>,
    sizes: Vec<usize>,
    global_sizes: Vec<usize>,
    invariant: bool,
}

/// The rank's owned answer in f64: under `single` the refinement's double
/// solution when the solve was refined (SPEC-LIT §118.12), else `psi`
/// widened.
fn owned_answer(
    gpu: &Gpu,
    w: &SpmdWorkspace,
    psi: &ofgpu::DevBuf<Scalar>,
    n: usize,
) -> Result<Vec<f64>> {
    #[cfg(feature = "single")]
    if w.ir_outer_steps() > 0 {
        if let Some(x) = w.refined_solution() {
            return Ok(gpu.download(x)?[..n].to_vec());
        }
    }
    #[cfg(not(feature = "single"))]
    let _ = w;
    Ok(gpu.download(psi)?[..n].iter().map(|&v| v as f64).collect())
}

/// One rank per part: `DistGamg::new` on the rank's split matrix, attached
/// to its workspace, then every control in order through `spmd_solve`, each
/// from zero.
fn spmd(
    m: &HostMesh,
    a: &HostLduMatrix,
    dec: &Decomposition,
    cfg: &GamgConfig,
    ctrls: &[SolverControls],
) -> Result<Vec<Result<RankOut>>> {
    let np = dec.n_parts;
    ThreadComm::run(np, |comm| -> Result<RankOut> {
        let gpu = Gpu::new(0)?;
        let sol = SolverKernels::new(&gpu)?;
        let r = comm.rank();
        let mut halo = SpmdHalo::new(&gpu, &dec.parts[r], np)?;
        let sys = DistSystem::split_one(&gpu, &mut halo, comm, m, dec, a)?;
        let mut w = SpmdWorkspace::new(&gpu, comm, &sys, DistReduce::Exact)?;
        w.set_flag_vote(true);
        let split = dec.split_matrix(m, r, a)?;
        let g = DistGamg::new(&gpu, comm, &dec.parts[r], &split, cfg, false)?;
        let sizes = g.sizes().to_vec();
        let global_sizes = g.global_sizes().to_vec();
        w.set_gamg(g)?;
        let n = dec.parts[r].mesh.n_cells;
        let mut solves = Vec::new();
        for ctrl in ctrls {
            let mut psi = gpu.zeros(dec.parts[r].n_local())?;
            let perf = spmd_solve(&gpu, &sol, comm, &mut psi, &sys, &mut halo, &mut w, ctrl)?;
            gpu.sync()?;
            solves.push((perf, owned_answer(&gpu, &w, &psi, n)?));
        }
        Ok(RankOut {
            solves,
            sizes,
            global_sizes,
            invariant: w.partition_invariant(Preconditioner::Gamg),
        })
    })
}

/// The ranks' owned fields of solve `k` put back in global order, and the
/// ranks' performances; every rank's error is unwrapped by name.
fn gathered(
    dec: &Decomposition,
    got: &[Result<RankOut>],
    k: usize,
    what: &str,
) -> (Vec<f64>, Vec<DistPerformance>) {
    let mut x = vec![0.0f64; dec.n_global_cells];
    let mut perfs = Vec::new();
    for (r, out) in got.iter().enumerate() {
        let out = out
            .as_ref()
            .unwrap_or_else(|e| panic!("{what}: rank {r} failed: {e}"));
        let (perf, owned) = &out.solves[k];
        for (c, &v) in owned.iter().enumerate() {
            x[dec.parts[r].global_cell[c] as usize] = v;
        }
        perfs.push(perf.clone());
    }
    (x, perfs)
}

/// `max |x - y| / max |y|` in f64.
fn rel_inf(x: &[f64], y: &[f64]) -> f64 {
    let (mut d, mut s) = (0.0f64, 0.0f64);
    for (&a, &b) in x.iter().zip(y) {
        d = d.max((a - b).abs());
        s = s.max(b.abs());
    }
    d / s
}

fn differs(x: &[f64], y: &[f64]) -> bool {
    x.iter().zip(y).any(|(a, b)| a.to_bits() != b.to_bits())
}

/// **The AMG-07b gate**, both builds. POI on a 32^3 box and on a 24x16x12
/// box cyclic in x; `P = 2, 4` ThreadComm ranks; Hilbert and linear cuts;
/// the default config (Chebyshev degree 2, V(2,2)). `spmd_pcg` with
/// `Preconditioner::Gamg` against the serial GAMG-PCG, and `spmd_solve`
/// with `LinearSolverKind::Gamg` against the serial stationary GAMG: every
/// rank converged on the same iteration, the answer within [`GATE`] of the
/// serial answer, and the iterations at most 1.3x the serial count.
#[test]
fn distributed_gamg_matches_serial_gamg_in_both_builds() {
    let gpu0 = match Gpu::new(0) {
        Ok(g) => g,
        Err(_) => return,
    };
    let k0 = SolverKernels::new(&gpu0).expect("kernels");
    let pcg = controls(LinearSolverKind::PCG, Preconditioner::Gamg);
    let gamg = controls(LinearSolverKind::Gamg, Preconditioner::None);
    let mut rows = 0usize;
    for (name, m) in [
        ("POI 32^3", box_mesh(32, 32, 32, &[])),
        ("POI 24x16x12 cyclic x", box_mesh(24, 16, 12, &[0])),
    ] {
        let a = poi(&m);
        let (sp, sx) = serial(&gpu0, &k0, &m, &a, &pcg).expect("serial GAMG-PCG");
        let (gp, gx) = serial(&gpu0, &k0, &m, &a, &gamg).expect("serial GAMG");
        assert!(
            sp.converged && gp.converged,
            "{name}: the serial references must converge"
        );
        assert!(
            sp.n_iterations > 3,
            "{name}: a three-iteration gate proves little"
        );
        for np in [2usize, 4] {
            for (cut, method) in [
                ("hilbert", PartitionMethod::Hilbert),
                ("linear", PartitionMethod::Linear),
            ] {
                let dec = Decomposition::build(&m, np, &method).expect("cut");
                assert!(dec.n_cut_faces > 0, "{name} P = {np} {cut} cut nothing");
                let got = spmd(&m, &a, &dec, &GamgConfig::default(), &[pcg, gamg]).expect("ranks");
                for (k, (what, want_x, want_it)) in [
                    ("GAMG-PCG", &sx, sp.n_iterations),
                    ("GAMG", &gx, gp.n_iterations),
                ]
                .into_iter()
                .enumerate()
                {
                    let at = format!("{name} P = {np} {cut} {what}");
                    let (x, perfs) = gathered(&dec, &got, k, &at);
                    let it = perfs[0].n_iterations;
                    for (r, p) in perfs.iter().enumerate() {
                        assert!(p.converged, "{at}: rank {r} did not converge");
                        assert_eq!(p.n_iterations, it, "{at}: rank {r} stopped elsewhere");
                    }
                    let rel = rel_inf(&x, want_x);
                    let sizes = got[0]
                        .as_ref()
                        .map(|o| o.global_sizes.clone())
                        .unwrap_or_default();
                    eprintln!(
                        "{at}: iterations {it} (serial {want_it}), max rel {rel:.3e}, levels {sizes:?}"
                    );
                    assert!(
                        rel <= GATE,
                        "{at}: the answer is {rel:e} from the serial one, gate {GATE:e}"
                    );
                    assert!(
                        10 * it <= 13 * want_it,
                        "{at}: {it} iterations against the serial {want_it}, over 1.3x"
                    );
                    rows += 1;
                }
            }
        }
    }
    assert_eq!(rows, 16, "every mesh, part count, cut and method is a row");
}

/// **The §73 honesty pattern.** The distributed cycle aggregates
/// block-locally, so every cut is a different preconditioner: (1)
/// `partition_invariant(Gamg)` is `false` on every rank and in one process;
/// (2) the answer really moves under a cut - bit for bit it differs between
/// cuts; (3) it still solves - every cut stays within [`GATE`] of the serial
/// answer, so what moved is the path, not the answer.
#[test]
fn the_distributed_cycle_moves_the_answer_under_a_cut_and_says_so() {
    let gpu0 = match Gpu::new(0) {
        Ok(g) => g,
        Err(_) => return,
    };
    let k0 = SolverKernels::new(&gpu0).expect("kernels");
    let m = box_mesh(16, 12, 10, &[]);
    let a = poi(&m);
    let pcg = controls(LinearSolverKind::PCG, Preconditioner::Gamg);
    let (_, want) = serial(&gpu0, &k0, &m, &a, &pcg).expect("serial");

    {
        let dec = Decomposition::build(&m, 2, &PartitionMethod::Hilbert).expect("cut");
        let mut ex = HaloExchange::new(&gpu0, &dec).expect("halo");
        let sys = DistSystem::split(&gpu0, &mut ex, &m, &dec, &a).expect("split");
        let dw = DistWorkspace::new(&gpu0, &sys, DistReduce::Exact).expect("ws");
        assert!(!dw.partition_invariant(Preconditioner::Gamg));
    }

    let mut fields: Vec<(String, Vec<f64>)> = Vec::new();
    for (np, cut, method) in [
        (2usize, "hilbert", PartitionMethod::Hilbert),
        (2, "linear", PartitionMethod::Linear),
        (4, "hilbert", PartitionMethod::Hilbert),
        (4, "linear", PartitionMethod::Linear),
    ] {
        let dec = Decomposition::build(&m, np, &method).expect("cut");
        let got = spmd(&m, &a, &dec, &GamgConfig::default(), &[pcg]).expect("ranks");
        let at = format!("P = {np} {cut}");
        for (r, out) in got.iter().enumerate() {
            let out = out
                .as_ref()
                .unwrap_or_else(|e| panic!("{at}: rank {r}: {e}"));
            assert!(
                !out.invariant,
                "{at}: rank {r} claims a partition-invariant GAMG"
            );
            assert!(out.sizes.len() >= 2, "{at}: rank {r} has no coarse level");
        }
        let (x, perfs) = gathered(&dec, &got, 0, &at);
        assert!(perfs.iter().all(|p| p.converged), "{at}: did not converge");
        let rel = rel_inf(&x, &want);
        assert!(
            rel <= GATE,
            "{at}: {rel:e} from the serial answer, gate {GATE:e}"
        );
        fields.push((at, x));
    }
    let mut moved = 0usize;
    let mut pairs = 0usize;
    for i in 0..fields.len() {
        for j in i + 1..fields.len() {
            pairs += 1;
            if differs(&fields[i].1, &fields[j].1) {
                moved += 1;
            }
        }
    }
    eprintln!("the answer moved in {moved} of {pairs} pairs of cuts");
    assert!(
        moved > 0,
        "the cut changed nothing in {pairs} pairs, so the hierarchy never depended on it"
    );
}

fn refused<T>(r: Result<T>, needle: &str) {
    match r {
        Ok(_) => panic!("expected a refusal naming {needle:?}"),
        Err(e) => {
            let s = e.to_string();
            assert!(s.contains(needle), "{needle:?} not in: {s}");
        }
    }
}

/// The refusals of §141.11, each by name, on every rank alike.
#[test]
fn a_distributed_gamg_request_that_cannot_be_served_is_refused_by_name() {
    let gpu0 = match Gpu::new(0) {
        Ok(g) => g,
        Err(_) => return,
    };
    let m = box_mesh(8, 6, 4, &[]);
    let a = poi(&m);
    let dec = Decomposition::build(&m, 2, &PartitionMethod::Hilbert).expect("cut");

    // In one process the cycle is refused, naming the SPMD route.
    {
        let sol = SolverKernels::new(&gpu0).expect("kernels");
        let mut ex = HaloExchange::new(&gpu0, &dec).expect("halo");
        let sys = DistSystem::split(&gpu0, &mut ex, &m, &dec, &a).expect("split");
        let mut dw = DistWorkspace::new(&gpu0, &sys, DistReduce::Exact).expect("ws");
        refused(
            dw.effective_preconditioner(Preconditioner::Gamg),
            "SpmdWorkspace::set_gamg",
        );
        let mut psi: Vec<_> = (0..2)
            .map(|p| gpu0.zeros::<Scalar>(dec.parts[p].n_local()).expect("zeros"))
            .collect();
        let ctrl = controls(LinearSolverKind::Gamg, Preconditioner::None);
        refused(
            dist_solve(&gpu0, &sol, &mut psi, &sys, &mut ex, &mut dw, &ctrl),
            "spmd_solve",
        );
    }

    let small = HierarchyConfig {
        n_cells_coarsest: 1000,
        ..GamgConfig::default().hierarchy
    };
    let got = ThreadComm::run(2, |comm| -> Result<Vec<String>> {
        let gpu = Gpu::new(0)?;
        let sol = SolverKernels::new(&gpu)?;
        let r = comm.rank();
        let mut halo = SpmdHalo::new(&gpu, &dec.parts[r], 2)?;
        let sys = DistSystem::split_one(&gpu, &mut halo, comm, &m, &dec, &a)?;
        let split = dec.split_matrix(&m, r, &a)?;
        let mut out = Vec::new();
        let msg = |e: ofgpu::Error| e.to_string();

        // No coarse level: the whole mesh is under n_cells_coarsest.
        let cfg = GamgConfig {
            hierarchy: small,
            ..GamgConfig::default()
        };
        out.push(
            DistGamg::new(&gpu, comm, &dec.parts[r], &split, &cfg, false)
                .err()
                .map(msg)
                .unwrap_or_default(),
        );
        // No smoothing sweeps.
        let cfg = GamgConfig {
            n_pre: 0,
            n_post: 0,
            ..GamgConfig::default()
        };
        out.push(
            DistGamg::new(&gpu, comm, &dec.parts[r], &split, &cfg, false)
                .err()
                .map(msg)
                .unwrap_or_default(),
        );

        // A workspace without a cycle refuses both GAMG requests.
        let mut w = SpmdWorkspace::new(&gpu, comm, &sys, DistReduce::Exact)?;
        let mut psi = gpu.zeros(dec.parts[r].n_local())?;
        let pcg = controls(LinearSolverKind::PCG, Preconditioner::Gamg);
        out.push(
            spmd_pcg(&gpu, &sol, comm, &mut psi, &sys, &mut halo, &mut w, &pcg)
                .err()
                .map(msg)
                .unwrap_or_default(),
        );
        let gamg = controls(LinearSolverKind::Gamg, Preconditioner::None);
        out.push(
            spmd_solve(&gpu, &sol, comm, &mut psi, &sys, &mut halo, &mut w, &gamg)
                .err()
                .map(msg)
                .unwrap_or_default(),
        );

        // An asymmetric cycle under CG.
        let cfg = GamgConfig {
            n_pre: 1,
            n_post: 2,
            ..GamgConfig::default()
        };
        let mut dg = DistGamg::new(&gpu, comm, &dec.parts[r], &split, &cfg, false)?;
        // The cycle inside a CUDA-graph capture, refused before it sends.
        let b = gpu.zeros::<Scalar>(dec.parts[r].mesh.n_cells)?;
        out.push(
            gpu.capture(|_| dg.cycle(&gpu, comm, &sys.mesh[0], &sys.a[0], &mut psi, &b))
                .err()
                .map(msg)
                .unwrap_or_default(),
        );
        w.set_gamg(dg)?;
        out.push(
            spmd_pcg(&gpu, &sol, comm, &mut psi, &sys, &mut halo, &mut w, &pcg)
                .err()
                .map(msg)
                .unwrap_or_default(),
        );
        Ok(out)
    })
    .expect("ranks");
    let needles = [
        "amg::distcycle: the hierarchy has no coarse level",
        "nPreSweeps + nPostSweeps = 0 is refused",
        "SpmdWorkspace::set_gamg",
        "spmd_gamg: solver GAMG was requested",
        "amg::distcycle: a distributed cycle was called inside a CUDA-graph capture",
        "spmd_pcg: GAMG preconditioning with nPreSweeps 1 and nPostSweeps 2",
    ];
    let mut checked = 0usize;
    for (r, res) in got.into_iter().enumerate() {
        let msgs = res.unwrap_or_else(|e| panic!("rank {r}: {e}"));
        assert_eq!(msgs.len(), needles.len(), "rank {r}");
        for (msg, needle) in msgs.iter().zip(needles) {
            assert!(msg.contains(needle), "rank {r}: {needle:?} not in {msg:?}");
            checked += 1;
        }
    }
    assert_eq!(checked, 12);
}
