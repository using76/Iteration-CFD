// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.
// Provenance: see PROVENANCE.md. No GPL-licensed source was consulted.

//! SPEC-LIT §118.4 — the mixed-precision iterative refinement of PCG and
//! PBiCGStab under `single`. Eight tests, the SAME assertions in BOTH
//! builds, over a system whose coefficients are EXACT in f32 (multiples of
//! 2^-12 and smaller dyadic rationals), so both builds hold the very same
//! matrix and the f64 dense solve is the truth for both. The refinement
//! itself is precision class P/A of docs/17 §5.1 rule 4: the answer is the
//! f64 dense solution to an absolute 1e-10 in both builds.
//!
//! An integration test, because the library's own test target does not
//! compile under `single`; an integration test links the normal (non-test)
//! library, which does. A machine without a card passes vacuously, the
//! convention of `tests/f32_accumulation.rs`.
//!
//! Provenance: ORIGINAL — the generator below and the systems are this
//! tree's own construction; the algorithm they exercise is §118.4's, whose
//! literature is cited there (Moler 1967; Carson & Higham 2018).
//! No GPL-licensed source was consulted.

use ofgpu::blockgen::{self, BlockSpec, GradedAxis};
use ofgpu::reference as cpu;
use ofgpu::GpuLduMatrix;
use ofgpu::solver::{
    solve_pbicgstab, solve_pcg, Preconditioner, SolverControls, SolverKernels, SolverWorkspace,
};
use ofgpu::{DevBuf, Gpu, GpuMesh, HostMesh, Result, Scalar};

/// Absolute slack of `max|x - x*|` against the f64 dense solution.
const SLACK: f64 = 1e-10;

/// The tolerance every refinement test asks for, unless it says otherwise.
const SOLVE_TOL: f64 = 1e-14;

/// The SplitMix64 finaliser of `seed * 0x9E3779B97F4A7C15 + k` - the test
/// data generator of SPEC-LIT §66.14, used only to make coefficients.
fn h(seed: u64, k: u64) -> u64 {
    let mut z = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(k);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Every device test needs a card; without one the test passes vacuously.
fn gpu() -> Option<Gpu> {
    Gpu::new(0).ok()
}

/// A uniform 6 x 5 x 4 block on [0,1]^3, every patch a wall, no cyclic - the
/// `spec` shape of `tests/f32_cht.rs`.
fn spec() -> BlockSpec {
    let axis = |n: usize| GradedAxis {
        lo: 0.0,
        hi: 1.0,
        n,
        expansion: 1.0,
        two_sided: false,
    };
    BlockSpec {
        x: axis(6),
        y: axis(5),
        z: axis(4),
        windows: Vec::new(),
        patch_name: BlockSpec::default().patch_name,
        patch_type: ["wall"; 6].map(String::from),
        cyclic: Vec::new(),
    }
}

/// An f64 value stored as the build's `Scalar`: a no-op in the f64 build,
/// the one rounding under `single`. Every coefficient below is chosen so the
/// rounding is exact and both builds hold the same matrix.
#[allow(clippy::useless_conversion)]
fn s(x: f64) -> Scalar {
    x as Scalar
}

/// The test system: an LDU matrix with diagonally dominant rows
/// (`diag_i = d_i + sum |off-diagonal|`, so the matrix is an M-matrix and
/// nonsingular), written straight onto the device - no assembly, so nothing
/// stands between the generator and the solver.
///
/// `asymmetric` makes `upper = -c(1-g)`, `lower = -c(1+g)` instead of both
/// `-c`; `b_scale` multiplies every `b_i` by a power of two, which is exact.
struct System {
    m: HostMesh,
    gm: GpuMesh,
    a: GpuLduMatrix,
    /// The f64 masters of the stored coefficients (exact in `Scalar`).
    diag: Vec<f64>,
    upper: Vec<f64>,
    lower: Vec<f64>,
    /// The f64 right-hand side actually stored (exact in `Scalar`).
    source: Vec<f64>,
    /// `solve_dense64(dense_from_ldu64)` of the same system.
    exact: Vec<f64>,
}

fn make_system(gpu: &Gpu, asymmetric: bool, b_scale: f64) -> Result<System> {
    let m = blockgen::build_mesh(&spec())?;
    let gm = GpuMesh::upload(gpu, &m)?;
    let n = m.n_cells;
    let nf = m.n_internal_faces;

    let mut upper = vec![0.0f64; nf];
    let mut lower = vec![0.0f64; nf];
    let mut diag = vec![0.0f64; n];
    let mut source = vec![0.0f64; n];

    for f in 0..nf {
        let c = ((32 + h(1, f as u64) % 64) as f64) / 64.0;
        let g = (((h(2, f as u64) % 17) as f64) - 8.0) / 16.0;
        let (u, l) = if asymmetric {
            (-c * (1.0 - g), -c * (1.0 + g))
        } else {
            (-c, -c)
        };
        upper[f] = u;
        lower[f] = l;
    }
    for i in 0..n {
        let d = ((8 + h(3, i as u64) % 64) as f64) / 64.0;
        let mut row = d;
        for f in 0..nf {
            if m.owner[f] as usize == i {
                row += upper[f].abs();
            }
            if m.neighbour[f] as usize == i {
                row += lower[f].abs();
            }
        }
        diag[i] = row;
        source[i] = ((((h(4, i as u64) % 8192) as f64) - 4096.0) / 4096.0) * b_scale;
    }

    let mut a = GpuLduMatrix::new(gpu, &gm)?;
    a.zero(gpu)?;
    let up: Vec<Scalar> = upper.iter().map(|v| s(*v)).collect();
    let lo: Vec<Scalar> = lower.iter().map(|v| s(*v)).collect();
    let di: Vec<Scalar> = diag.iter().map(|v| s(*v)).collect();
    let src: Vec<Scalar> = source.iter().map(|v| s(*v)).collect();
    gpu.write(&mut a.upper, &up)?;
    gpu.write(&mut a.lower, &lo)?;
    gpu.write(&mut a.diag, &di)?;
    gpu.write(&mut a.source, &src)?;

    let mut r = cpu::CpuLdu::new(&m);
    r.diag = di;
    r.upper = up;
    r.lower = lo;
    r.source = src;
    let exact = cpu::solve_dense64(cpu::dense_from_ldu64(&r, &m), &source)
        .expect("the reference dense solve");

    Ok(System {
        m,
        gm,
        a,
        diag,
        upper,
        lower,
        source,
        exact,
    })
}

/// The reference's own self-check, independent of [`cpu`]: the f64 row
/// residual `max_i |b_i - (A x*)_i|` of the dense solution, gathered per row
/// the way the device walks a row. No boundary term: every patch is a wall,
/// no cyclic, so the coupled coefficients are zero and bNbrCell < 0.
fn row_residual(sys: &System, x: &[f64]) -> f64 {
    let n = sys.m.n_cells;
    let mut rows = vec![0.0f64; n];
    for c in 0..n {
        rows[c] += sys.diag[c] * x[c];
    }
    for f in 0..sys.m.n_internal_faces {
        let p = sys.m.owner[f] as usize;
        let nb = sys.m.neighbour[f] as usize;
        rows[p] += sys.upper[f] * x[nb];
        rows[nb] += sys.lower[f] * x[p];
    }
    (0..n)
        .map(|c| (sys.source[c] - rows[c]).abs())
        .fold(0.0f64, f64::max)
}

/// The answer read back: the f64 build downloads `psi` itself, `single`
/// downloads the refinement's double solution.
#[cfg(not(feature = "single"))]
fn answer(gpu: &Gpu, _ws: &SolverWorkspace, psi: &DevBuf<Scalar>) -> Result<Vec<f64>> {
    Ok(gpu.download(psi)?)
}

/// The answer read back, `single`: [`ofgpu::solver::SolverWorkspace::
/// refined_solution`], the double `x_hi` the refinement carries.
#[cfg(feature = "single")]
fn answer(gpu: &Gpu, ws: &SolverWorkspace, _psi: &DevBuf<Scalar>) -> Result<Vec<f64>> {
    let Some(x_hi) = ws.refined_solution() else {
        panic!("single: a refined solve must leave a double solution behind");
    };
    Ok(gpu.download(x_hi)?)
}

/// The controls every test starts from.
fn ctrl() -> SolverControls {
    SolverControls {
        tolerance: s(SOLVE_TOL),
        rel_tol: 0.0,
        max_iter: 500,
        min_iter: 0,
        check_interval: 1,
        report_residuals: true,
        ..Default::default()
    }
}

/// The dense-solution agreement every refined solve is held to.
fn agrees(x: &[f64], exact: &[f64]) -> f64 {
    x.iter()
        .zip(exact.iter())
        .map(|(v, e)| (v - e).abs())
        .fold(0.0f64, f64::max)
}

/// 1. The refined PCG on the symmetric system is the f64 dense solve, for
///    Jacobi and for the multi-colour DIC.
#[test]
fn refined_pcg_matches_the_f64_dense_solve() {
    let Some(gpu) = gpu() else { return };
    let sys = make_system(&gpu, false, 1.0).unwrap();
    let self_check = row_residual(&sys, &sys.exact);
    assert!(self_check <= 1e-13, "dense reference self-check {self_check:e}");

    let k = SolverKernels::new(&gpu).unwrap();
    for precon in [Preconditioner::Diagonal, Preconditioner::Dic] {
        let mut ws = SolverWorkspace::for_mesh(&gpu, &sys.gm).unwrap();
        let mut psi: DevBuf<Scalar> = gpu.zeros(sys.m.n_cells).unwrap();
        let ctrl = SolverControls { precon, ..ctrl() };
        let perf = solve_pcg(&gpu, &k, &mut psi, &sys.a, &sys.gm, &mut ws, &ctrl).unwrap();
        gpu.sync().unwrap();

        assert!(perf.converged, "PCG/{precon:?} did not converge");
        let x = answer(&gpu, &ws, &psi).unwrap();
        let worst = agrees(&x, &sys.exact);
        assert!(worst < SLACK, "PCG/{precon:?}: max|x - x*| = {worst:e}");
        assert!(
            f64::from(perf.final_residual) <= SOLVE_TOL,
            "PCG/{precon:?}: final residual {}",
            perf.final_residual
        );
        assert!(
            perf.n_iterations <= 2 * sys.m.n_cells,
            "PCG/{precon:?}: {} sweeps",
            perf.n_iterations
        );

        #[cfg(feature = "single")]
        {
            let steps = ws.ir_outer_steps();
            assert!(
                (1..=6).contains(&steps),
                "PCG/{precon:?}: {steps} outer steps"
            );
            let psi_h: Vec<Scalar> = gpu.download(&psi).unwrap();
            let x_hi = gpu.download(ws.refined_solution().unwrap()).unwrap();
            for (p, x) in psi_h.iter().zip(x_hi.iter()) {
                assert_eq!(*p, *x as f32, "the stored field must be fl32(x_hi)");
            }
        }
    }
}

/// 3. The same system scaled by 2^-100 - far below float's normal range as
///    a residual, though every stored value stays a normal float. The
///    unit-norm correction right-hand side of (118.4.5) is what keeps the
///    solve from underflowing or stalling.
#[test]
fn a_system_at_2e_minus_100_neither_underflows_nor_stalls() {
    let Some(gpu) = gpu() else { return };
    let scale = 2f64.powi(-100);
    let sys = make_system(&gpu, true, scale).unwrap();
    let self_check = row_residual(&sys, &sys.exact);
    assert!(self_check <= 1e-13, "dense reference self-check {self_check:e}");

    let k = SolverKernels::new(&gpu).unwrap();
    let mut ws = SolverWorkspace::for_mesh(&gpu, &sys.gm).unwrap();
    let mut psi: DevBuf<Scalar> = gpu.zeros(sys.m.n_cells).unwrap();
    let perf = solve_pbicgstab(&gpu, &k, &mut psi, &sys.a, &sys.gm, &mut ws, &ctrl()).unwrap();
    gpu.sync().unwrap();

    assert!(perf.converged, "the scaled system did not converge");
    let x = answer(&gpu, &ws, &psi).unwrap();
    let exact_max = sys.exact.iter().fold(0.0f64, |s, v| s.max(v.abs()));
    let worst = agrees(&x, &sys.exact) / exact_max;
    assert!(worst < SLACK, "relative error {worst:e}");
    assert!(
        f64::from(perf.final_residual) <= SOLVE_TOL,
        "final residual {}",
        perf.final_residual
    );
}

/// 4. A field already at the answer takes no sweep at all.
#[test]
fn an_already_converged_field_takes_no_sweep() {
    let Some(gpu) = gpu() else { return };
    let sys = make_system(&gpu, false, 1.0).unwrap();
    let k = SolverKernels::new(&gpu).unwrap();

    let mut ws = SolverWorkspace::for_mesh(&gpu, &sys.gm).unwrap();
    let rounded: Vec<Scalar> = sys.exact.iter().map(|v| s(*v)).collect();
    let mut psi: DevBuf<Scalar> = gpu.zeros(sys.m.n_cells).unwrap();
    gpu.write(&mut psi, &rounded).unwrap();
    let ctrl = SolverControls {
        tolerance: 1e-4,
        ..ctrl()
    };
    let perf = solve_pcg(&gpu, &k, &mut psi, &sys.a, &sys.gm, &mut ws, &ctrl).unwrap();
    gpu.sync().unwrap();

    assert!(perf.converged);
    assert_eq!(perf.n_iterations, 0, "an answered system must not iterate");
    #[cfg(feature = "single")]
    assert_eq!(ws.ir_outer_steps(), 0, "no outer step either");
}

/// 5. A relative tolerance stops at its own reduction, and costs no more
///    sweeps than the absolute solve of the same system.
#[test]
fn a_relative_tolerance_stops_at_its_own_reduction() {
    let Some(gpu) = gpu() else { return };
    let sys = make_system(&gpu, true, 1.0).unwrap();
    let k = SolverKernels::new(&gpu).unwrap();

    // Test 2's Diagonal solve, repeated here so this test owns its number.
    let abs_sweeps = {
        let mut abs_ws = SolverWorkspace::for_mesh(&gpu, &sys.gm).unwrap();
        let mut abs_psi: DevBuf<Scalar> = gpu.zeros(sys.m.n_cells).unwrap();
        let abs = solve_pbicgstab(&gpu, &k, &mut abs_psi, &sys.a, &sys.gm, &mut abs_ws, &ctrl())
            .unwrap();
        gpu.sync().unwrap();
        assert!(abs.converged, "the absolute solve itself must converge");
        abs.n_iterations
    };

    let mut ws = SolverWorkspace::for_mesh(&gpu, &sys.gm).unwrap();
    let mut psi: DevBuf<Scalar> = gpu.zeros(sys.m.n_cells).unwrap();
    let ctrl = SolverControls {
        tolerance: 0.0,
        rel_tol: 1e-6,
        ..ctrl()
    };
    let perf = solve_pbicgstab(&gpu, &k, &mut psi, &sys.a, &sys.gm, &mut ws, &ctrl).unwrap();
    gpu.sync().unwrap();

    assert!(perf.converged);
    let budget = 1e-6 * perf.initial_residual as f64 * (1.0 + 1e-6);
    assert!(
        perf.final_residual as f64 <= budget,
        "final {} against budget {budget:e}",
        perf.final_residual
    );
    assert!(
        perf.n_iterations <= abs_sweeps,
        "rel {} sweeps against abs {abs_sweeps}",
        perf.n_iterations
    );
}

/// 6. A spent budget is reported honestly: not converged.
#[test]
fn a_spent_budget_is_not_converged() {
    let Some(gpu) = gpu() else { return };
    let sys = make_system(&gpu, true, 1.0).unwrap();
    let k = SolverKernels::new(&gpu).unwrap();

    let mut ws = SolverWorkspace::for_mesh(&gpu, &sys.gm).unwrap();
    let mut psi: DevBuf<Scalar> = gpu.zeros(sys.m.n_cells).unwrap();
    let ctrl = SolverControls {
        max_iter: 3,
        ..ctrl()
    };
    let perf = solve_pbicgstab(&gpu, &k, &mut psi, &sys.a, &sys.gm, &mut ws, &ctrl).unwrap();
    gpu.sync().unwrap();

    assert!(!perf.converged, "three sweeps cannot be the answer");
    assert!(perf.n_iterations <= 3);
}

/// 2. The same on the asymmetric system with PBiCGStab, for Jacobi and for
///    the multi-colour DILU.
#[test]
fn refined_pbicgstab_matches_the_f64_dense_solve_on_an_asymmetric_system() {
    let Some(gpu) = gpu() else { return };
    let sys = make_system(&gpu, true, 1.0).unwrap();
    let self_check = row_residual(&sys, &sys.exact);
    assert!(self_check <= 1e-13, "dense reference self-check {self_check:e}");

    let k = SolverKernels::new(&gpu).unwrap();
    for precon in [Preconditioner::Diagonal, Preconditioner::Dilu] {
        let mut ws = SolverWorkspace::for_mesh(&gpu, &sys.gm).unwrap();
        let mut psi: DevBuf<Scalar> = gpu.zeros(sys.m.n_cells).unwrap();
        let ctrl = SolverControls { precon, ..ctrl() };
        let perf = solve_pbicgstab(&gpu, &k, &mut psi, &sys.a, &sys.gm, &mut ws, &ctrl).unwrap();
        gpu.sync().unwrap();

        assert!(perf.converged, "PBiCGStab/{precon:?} did not converge");
        let x = answer(&gpu, &ws, &psi).unwrap();
        let worst = agrees(&x, &sys.exact);
        assert!(worst < SLACK, "PBiCGStab/{precon:?}: max|x - x*| = {worst:e}");
        assert!(
            f64::from(perf.final_residual) <= SOLVE_TOL,
            "PBiCGStab/{precon:?}: final residual {}",
            perf.final_residual
        );
        assert!(
            perf.n_iterations <= 2 * sys.m.n_cells,
            "PBiCGStab/{precon:?}: {} sweeps",
            perf.n_iterations
        );

        #[cfg(feature = "single")]
        {
            let steps = ws.ir_outer_steps();
            assert!(
                (1..=6).contains(&steps),
                "PBiCGStab/{precon:?}: {steps} outer steps"
            );
            let psi_h: Vec<Scalar> = gpu.download(&psi).unwrap();
            let x_hi = gpu.download(ws.refined_solution().unwrap()).unwrap();
            for (p, x) in psi_h.iter().zip(x_hi.iter()) {
                assert_eq!(*p, *x as f32, "the stored field must be fl32(x_hi)");
            }
        }
    }
}

/// 7. A `fixedIters` solve is NOT refined - it is the only kind a CUDA graph
///    may capture (§113.2) - and reports exactly its five sweeps.
#[test]
fn a_fixed_iteration_solve_is_not_refined() {
    let Some(gpu) = gpu() else { return };
    let sys = make_system(&gpu, true, 1.0).unwrap();
    let k = SolverKernels::new(&gpu).unwrap();

    let mut ws = SolverWorkspace::for_mesh(&gpu, &sys.gm).unwrap();
    let mut psi: DevBuf<Scalar> = gpu.zeros(sys.m.n_cells).unwrap();
    let ctrl = SolverControls {
        fixed_iters: true,
        max_iter: 5,
        ..ctrl()
    };
    let perf = solve_pbicgstab(&gpu, &k, &mut psi, &sys.a, &sys.gm, &mut ws, &ctrl).unwrap();
    gpu.sync().unwrap();

    assert_eq!(perf.n_iterations, 5);
    #[cfg(feature = "single")]
    assert_eq!(ws.ir_outer_steps(), 0, "a fixedIters solve is not refined");
}

/// 8. A zero right-hand side is answered at once: R_0 = 0 is converged
///    before any sweep.
#[test]
fn a_zero_right_hand_side_is_converged_at_once() {
    let Some(gpu) = gpu() else { return };
    let sys = make_system(&gpu, false, 0.0).unwrap();
    let k = SolverKernels::new(&gpu).unwrap();

    let mut ws = SolverWorkspace::for_mesh(&gpu, &sys.gm).unwrap();
    let mut psi: DevBuf<Scalar> = gpu.zeros(sys.m.n_cells).unwrap();
    let perf = solve_pcg(&gpu, &k, &mut psi, &sys.a, &sys.gm, &mut ws, &ctrl()).unwrap();
    gpu.sync().unwrap();

    assert!(perf.converged);
    assert_eq!(perf.n_iterations, 0);
    #[cfg(feature = "single")]
    assert_eq!(ws.ir_outer_steps(), 0);
}
