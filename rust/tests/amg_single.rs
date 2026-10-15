// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.
// Provenance: see PROVENANCE.md. No GPL-licensed source was consulted.

//! AMG-06b: the f32 GAMG preconditioner of an f64 solve, the gate table
//! of SPEC-LIT §141.12.1. In the f64 build `GamgPrecision::Single` holds
//! the hierarchy's device copy in `f32`; under `--features single` the
//! hierarchy is already float and the two precisions are one, which the
//! same tests then hold bitwise.
//!
//! Provenance: ORIGINAL - the gates of SPEC-LIT §141.12 on the POI matrix
//! of (141.5.5). No GPL-licensed source was consulted.

use ofgpu::amg::{CycleKind, Gamg, GamgConfig, GamgPrecision, HostRows};
use ofgpu::blockgen::{BlockSpec, GradedAxis, build_mesh};
use ofgpu::device::DevBuf;
use ofgpu::error::Result;
use ofgpu::io::case::{LinearSolverKind, Preconditioner, SolverControls, read_solver_controls};
use ofgpu::io::dict::FoamDict;
use ofgpu::ldu::{GpuLduMatrix, HostLduMatrix};
use ofgpu::mesh::{GpuMesh, HostMesh};
use ofgpu::pressure::{PbicgstabBackend, PressureBackend, SystemProbe};
use ofgpu::solver::{SolverPerformance, SolverWorkspace, solve};
use ofgpu::{Gpu, Scalar};

/// Every device test needs a card; without one the test passes vacuously.
fn gpu() -> Option<Gpu> {
    Gpu::new(0).ok()
}

/// A cube of `n^3` unit cells, every patch a wall.
fn box_mesh(n: usize) -> HostMesh {
    let axis = GradedAxis {
        lo: 0.0,
        hi: n as Scalar,
        n,
        expansion: 1.0,
        two_sided: false,
    };
    let b = BlockSpec {
        x: axis.clone(),
        y: axis.clone(),
        z: axis,
        windows: Vec::new(),
        patch_name: BlockSpec::default().patch_name,
        patch_type: ["wall"; 6].map(String::from),
        cyclic: Vec::new(),
    };
    build_mesh(&b).expect("box mesh")
}

/// `POI(n)` of (141.5.5): the §2 laplacian negated, the wall terms folded
/// into the diagonal, `boundary_coeffs` zero, `b = 1`.
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
        let c = m.b_face_cells[bf] as usize;
        a.diag[c] += m.b_mag_sf[bf] * m.b_delta_coeffs[bf];
    }
    a.source = vec![1.0; m.n_cells];
    a
}

/// The solve's answer in f64: under `single` the refined double solution
/// of §118.4 when the solve was refined (SPEC-LIT 141.9), else `psi`.
fn answer(gpu: &Gpu, w: &SolverWorkspace, psi: &DevBuf<Scalar>, n: usize) -> Result<Vec<f64>> {
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

/// SPEC-LIT §8.4's normalised residual of the stored system, in f64 on
/// the host (the AMG-06a helper of `tests/amg_cycle.rs`).
fn res_8_4(rows: &HostRows, b: &[Scalar], x: &[f64]) -> f64 {
    let n = rows.n_cells;
    let ax = rows.amul(&x[..n]);
    let mean = x[..n].iter().sum::<f64>() / n as f64;
    let axr = rows.amul(&vec![mean; n]);
    let (mut num, mut den) = (0.0f64, f64::MIN_POSITIVE);
    for c in 0..n {
        let bc = b[c] as f64;
        num += (bc - ax[c]).abs();
        den += (ax[c] - axr[c]).abs() + (bc - axr[c]).abs();
    }
    num / den
}

/// One solve from zero with `g` attached; the performance and the answer.
fn one_solve(
    gpu: &Gpu,
    mesh: &GpuMesh,
    mat: &GpuLduMatrix,
    g: Gamg,
    ctrl: &SolverControls,
) -> Result<(SolverPerformance, Vec<f64>)> {
    let k = ofgpu::solver::SolverKernels::new(gpu)?;
    let n = mesh.n_cells;
    let mut w = SolverWorkspace::for_mesh(gpu, mesh)?;
    w.set_gamg(g)?;
    let mut psi = gpu.zeros(n)?;
    let p = solve(gpu, &k, &mut psi, mat, mesh, &mut w, ctrl)?;
    let x = answer(gpu, &w, &psi, n)?;
    Ok((p, x))
}

/// **The AMG-06b gate** (docs/17 §12 row, SPEC-LIT 141.12.1): `POI(128)`,
/// default config, PCG with GAMG at tol `1e-10`, `relTol 0`, from zero,
/// once with the native hierarchy and once with `precision single`.
/// f64 build: both converge, the f32 copy has the f64 hierarchy's level
/// sizes, takes `<= 1.15x` the all-f64 iterations, its answer's §8.4
/// residual on the host is `<= 1e-10`, and the device bytes it holds are
/// `<= 0.55x` the native hierarchy's (141.47). Under `single` the two
/// precisions are one: equal counts, equal bytes, bitwise answers.
#[test]
fn an_f32_gamg_preconditions_an_f64_pcg_on_poisson_128() {
    let Some(gpu) = gpu() else { return };
    let m = box_mesh(128);
    let a = poi(&m);
    let rows = HostRows::of_matrix(&m, &a).expect("host rows");
    let mesh = GpuMesh::upload(&gpu, &m).expect("mesh");
    let mat = a.upload(&gpu).expect("matrix");
    let ctrl = SolverControls {
        solver: LinearSolverKind::PCG,
        precon: Preconditioner::Gamg,
        tolerance: 1e-10,
        rel_tol: 0.0,
        max_iter: 5000,
        check_interval: 1,
        ..SolverControls::default()
    };
    let cfg = GamgConfig::default();
    let build = |p| Gamg::new_with_precision(&gpu, &m, &a, &cfg, p, false).expect("gamg");

    let gn = build(GamgPrecision::Native);
    let (sizes_n, bytes_n, f32_n) = (gn.sizes(), gn.device_bytes(), gn.runs_in_f32());
    let (pn, xn) = one_solve(&gpu, &mesh, &mat, gn, &ctrl).expect("native GAMG-PCG");
    let gs = build(GamgPrecision::Single);
    let (sizes_s, bytes_s, f32_s) = (gs.sizes(), gs.device_bytes(), gs.runs_in_f32());
    let (ps, xs) = one_solve(&gpu, &mesh, &mat, gs, &ctrl).expect("single GAMG-PCG");

    let (rn, rs) = (res_8_4(&rows, &a.source, &xn), res_8_4(&rows, &a.source, &xs));
    let (mut dd, mut nn) = (0.0f64, 0.0f64);
    for (p, q) in xs.iter().zip(&xn) {
        dd += (p - q) * (p - q);
        nn += q * q;
    }
    let diff = (dd / nn).sqrt();
    let rho = bytes_s.total() as f64 / bytes_n.total() as f64;
    let rho_float = (bytes_s.float + bytes_s.acc) as f64 / (bytes_n.float + bytes_n.acc) as f64;
    let mib = |b: usize| b as f64 / (1024.0 * 1024.0);
    eprintln!(
        "POI(128) PCG/GAMG @1e-10: sizes {} | native {} it ({}), single {} it ({}) | ratio {:.4} \
         | 8.4 residual native {rn:.6e}, single {rs:.6e} | answers differ by {diff:.3e} relative",
        sizes_s.iter().map(|s| s.to_string()).collect::<Vec<_>>().join("->"),
        pn.n_iterations,
        if pn.converged { "converged" } else { "NOT converged" },
        ps.n_iterations,
        if ps.converged { "converged" } else { "NOT converged" },
        ps.n_iterations as f64 / pn.n_iterations as f64,
    );
    eprintln!(
        "device bytes (141.47): native {:.2} MiB (float {:.2}, acc {:.4}, label {:.2}) | single \
         {:.2} MiB (float {:.2}, acc {:.4}, label {:.2}) | rho {rho:.4} total, {rho_float:.4} \
         float and acc only",
        mib(bytes_n.total()),
        mib(bytes_n.float),
        mib(bytes_n.acc),
        mib(bytes_n.label),
        mib(bytes_s.total()),
        mib(bytes_s.float),
        mib(bytes_s.acc),
        mib(bytes_s.label),
    );

    assert!(pn.converged && ps.converged, "both PCG solves must converge at tol 1e-10");
    assert_eq!(sizes_n, sizes_s, "the f32 copy is the solve's own hierarchy, level for level");
    assert!(!f32_n, "a native hierarchy never runs on the f32 copy");
    assert!(rn <= 1e-10, "the native answer's 8.4 residual {rn:.3e} is over 1e-10");
    assert!(rs <= 1e-10, "the single answer's 8.4 residual {rs:.3e} is over 1e-10");
    if cfg!(feature = "single") {
        assert!(!f32_s, "under single the hierarchy is already float: no second copy");
        assert_eq!(pn.n_iterations, ps.n_iterations, "under single the two precisions are one");
        assert_eq!(bytes_n, bytes_s, "under single the two precisions hold the same bytes");
        assert!(
            xn.iter().zip(&xs).all(|(p, q)| p.to_bits() == q.to_bits()),
            "under single the two precisions answer bitwise alike"
        );
    } else {
        assert!(f32_s, "precision single must run the f32 copy in the f64 build");
        assert!(
            100 * ps.n_iterations <= 115 * pn.n_iterations,
            "f64 PCG + f32 GAMG took {} iterations, over 1.15x the all-f64 {}",
            ps.n_iterations,
            pn.n_iterations
        );
        assert!(rho <= 0.55, "the f32 hierarchy holds {rho:.4}x the native bytes, over 0.55x");
    }
}

/// `solvers { p { <body> } }`, as an fvSolution file writes it.
fn controls(body: &str) -> Result<SolverControls> {
    let src = format!(
        "solvers\n{{\n    p\n    {{\n{body}\n        tolerance 1e-8;\n        relTol 0;\n        \
         maxIter 2000;\n    }}\n}}\n"
    );
    let d = FoamDict::parse(&src, "system/fvSolution").expect("fvSolution parses");
    let mut sc = SolverControls::default();
    read_solver_controls(&mut sc, &d, "p")?;
    Ok(sc)
}

/// One solve from zero through `PbicgstabBackend` set up with `ctrl`.
fn backend_solve(
    gpu: &Gpu,
    hm: &HostMesh,
    m: &GpuMesh,
    a: &HostLduMatrix,
    ctrl: SolverControls,
) -> (SolverPerformance, Vec<Scalar>) {
    let mut be = PbicgstabBackend::new(ctrl);
    be.setup(gpu, hm, m, &SystemProbe::default()).expect("setup");
    let ga = a.upload(gpu).expect("matrix upload");
    let mut x: DevBuf<Scalar> = gpu.zeros(hm.n_cells).expect("psi");
    let p = be.solve(gpu, &mut x, &ga, m).expect("solve");
    (p, gpu.download(&x).expect("download"))
}

/// SPEC-LIT §13.4.1's pair (§141.12.1): two fvSolution texts that differ
/// in the one line `precision single;`, read by `read_solver_controls`
/// and solved through `PbicgstabBackend` on `POI(16)` at `1e-8`. The key
/// lands in `GamgControls::precision` under PCG/GAMG, under the
/// sub-dictionary form and under the GAMG solver; the default is
/// `native`, written or not. f64 build: the `single` answers differ from
/// the native ones in their bits and every solve converges. Under
/// `single` the two are bitwise one. A value other than `native` and
/// `single` is refused naming the key and both values.
#[test]
fn the_gamg_precision_key_reaches_the_solve() {
    let pcg = "        solver PCG;\n        preconditioner GAMG;";
    let sub = "        solver PCG;\n        preconditioner\n        {\n            preconditioner GAMG;";
    let gamg = "        solver GAMG;";
    let read = |lines: &str| controls(lines).expect(lines);
    assert_eq!(read(pcg).gamg.precision, GamgPrecision::Native);
    assert_eq!(read(&format!("{pcg}\n        precision native;")), read(pcg));
    assert_eq!(read(&format!("{pcg}\n        precision single;")).gamg.precision, GamgPrecision::Single);
    assert_eq!(
        read(&format!("{sub}\n            precision single;\n        }}")).gamg.precision,
        GamgPrecision::Single
    );
    assert_eq!(read(&format!("{gamg}\n        precision single;")).gamg.precision, GamgPrecision::Single);

    let err = controls(&format!("{pcg}\n        precision half;")).expect_err("half is refused");
    let msg = err.to_string();
    for want in ["solvers/p/precision", "half", "native", "single", "141.12"] {
        assert!(msg.contains(want), "the refusal must name {want}: {msg}");
    }

    let Some(gpu) = gpu() else { return };
    let hm = box_mesh(16);
    let m = GpuMesh::upload(&gpu, &hm).expect("mesh upload");
    let a = poi(&hm);
    for lines in [pcg, gamg] {
        let (pn, xn) = backend_solve(&gpu, &hm, &m, &a, read(lines));
        let (ps, xs) = backend_solve(&gpu, &hm, &m, &a, read(&format!("{lines}\n        precision single;")));
        let differing = xn.iter().zip(&xs).filter(|(p, q)| p.to_bits() != q.to_bits()).count();
        println!(
            "{}: native {} it, single {} it, {differing} of {} entries differ in their bits",
            lines.trim().replace('\n', " "),
            pn.n_iterations,
            ps.n_iterations,
            xn.len()
        );
        assert!(pn.converged && ps.converged, "{lines}: both must converge");
        if cfg!(feature = "single") {
            assert_eq!(differing, 0, "{lines}: under single the two precisions are one");
        } else {
            assert!(differing > 0, "{lines}: precision single must reach the solve");
        }
    }
}

/// The f32 copy's refusals (§141.12.1, class D): the K-cycle is refused
/// naming `K-cycle`, `single` and `141.12`, and the Gamg keeps its
/// V-cycle. Under `single` there is no f32 copy and the K-cycle is set.
#[test]
fn the_single_cycle_refusals_are_named() {
    let Some(gpu) = gpu() else { return };
    let m = box_mesh(8);
    let a = poi(&m);
    let mut g =
        Gamg::new_with_precision(&gpu, &m, &a, &GamgConfig::default(), GamgPrecision::Single, false)
            .expect("gamg");
    assert_eq!(g.precision(), GamgPrecision::Single);
    let k = g.set_cycle(&gpu, CycleKind::K { threshold: 0.25 });
    if cfg!(feature = "single") {
        assert!(k.is_ok(), "under single precision single is native and takes a K-cycle");
    } else {
        let msg = k.expect_err("the K-cycle is refused on the f32 copy").to_string();
        for want in ["K-cycle", "single", "141.12"] {
            assert!(msg.contains(want), "the refusal must name {want}: {msg}");
        }
        assert_eq!(g.cycle_kind(), CycleKind::V);
    }
}

/// The stationary GAMG solver on the f32 copy, (141.46): `POI(32)`,
/// `solver GAMG`, tol `1e-10`, `relTol 0`, from zero. Each cycle's
/// residual is formed in the solve's precision, so the f32 levels do not
/// floor it: it converges, the answer's §8.4 residual on the host is
/// `<= 1e-10`, in at most 1.15x the native cycles (both builds).
#[test]
fn the_stationary_single_gamg_converges() {
    let Some(gpu) = gpu() else { return };
    let m = box_mesh(32);
    let a = poi(&m);
    let rows = HostRows::of_matrix(&m, &a).expect("host rows");
    let mesh = GpuMesh::upload(&gpu, &m).expect("mesh");
    let mat = a.upload(&gpu).expect("matrix");
    let ctrl = SolverControls {
        solver: LinearSolverKind::Gamg,
        tolerance: 1e-10,
        rel_tol: 0.0,
        max_iter: 5000,
        check_interval: 1,
        ..SolverControls::default()
    };
    let cfg = GamgConfig::default();
    let mut its = Vec::new();
    for p in [GamgPrecision::Native, GamgPrecision::Single] {
        let g = Gamg::new_with_precision(&gpu, &m, &a, &cfg, p, false).expect("gamg");
        let (perf, x) = one_solve(&gpu, &mesh, &mat, g, &ctrl).expect("GAMG");
        let r = res_8_4(&rows, &a.source, &x);
        println!(
            "POI(32) solver GAMG @1e-10, precision {}: {} cycles, converged {}, 8.4 residual {r:.3e}",
            p.name(),
            perf.n_iterations,
            perf.converged
        );
        assert!(perf.converged, "precision {}: the stationary GAMG must converge", p.name());
        assert!(r <= 1e-10, "precision {}: the 8.4 residual {r:.3e} is over 1e-10", p.name());
        its.push(perf.n_iterations);
    }
    assert!(
        100 * its[1] <= 115 * its[0],
        "the f32 copy took {} cycles, over 1.15x the native {}",
        its[1],
        its[0]
    );
}
