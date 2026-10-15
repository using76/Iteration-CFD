// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.
// Provenance: see PROVENANCE.md. No GPL-licensed source was consulted.

//! AMG-04b: a case can ask for GAMG and its settings and gets them, the
//! gate table of SPEC-LIT §141.7. Each pair is two `fvSolution` texts that
//! differ in ONE line, read by `io::case::read_solver_controls` (the reader
//! every dictionary driver uses) and solved by `pressure::PbicgstabBackend`
//! (the backend every incompressible driver hands its pressure equation
//! to) on `POI(16)` of (141.5.5): SPEC-LIT §13.4.1's pair rule, the setting
//! shown to reach the solve by the iteration count or the hierarchy it
//! changes.
//!
//! An integration test, so the same assertions run in both builds (§118).
//! A machine without a card passes the device rows vacuously, the
//! convention of `tests/amg_cycle.rs`.
//!
//! Provenance: ORIGINAL - the fixtures and the assertions are this tree's
//! own construction, run against SPEC-LIT §141.5 and §141.7. OpenFOAM's
//! GAMG and its sources were not opened; the key names are the case
//! format's, as `fvSolution` files write them. No GPL-licensed source was
//! consulted.

use ofgpu::blockgen::{BlockSpec, GradedAxis, build_mesh};
use ofgpu::device::DevBuf;
use ofgpu::io::case::{
    LinearSolverKind, Preconditioner, SolverControls, read_solver_controls,
};
use ofgpu::io::dict::FoamDict;
use ofgpu::ldu::{GpuLduMatrix, HostLduMatrix};
use ofgpu::mesh::{GpuMesh, HostMesh};
use ofgpu::pressure::{PbicgstabBackend, PressureBackend, SystemProbe};
use ofgpu::solver::SolverPerformance;
use ofgpu::{Gpu, Scalar};

/// Every device test needs a card; without one the test passes vacuously.
fn gpu() -> Option<Gpu> {
    Gpu::new(0).ok()
}

/// A box of unit cells, every patch a wall.
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

/// `POI(n)` of (141.5.5), the diagonal scaled by `s`: `s = 1` is the
/// Dirichlet Poisson matrix, and any other `s` a different matrix of the
/// same pattern. `b = 1`.
fn poi(m: &HostMesh, s: Scalar) -> HostLduMatrix {
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
    for d in &mut a.diag {
        *d *= s;
    }
    a.source = vec![1.0; m.n_cells];
    a
}

/// `solvers { p { <body> } }`, as an fvSolution file writes it.
fn fv_solution(body: &str) -> FoamDict {
    let src = format!("solvers\n{{\n    p\n    {{\n{body}\n    }}\n}}\n");
    FoamDict::parse(&src, "system/fvSolution").expect("fvSolution parses")
}

/// The controls the reader gives `p` for `body`.
fn controls(body: &str) -> ofgpu::error::Result<SolverControls> {
    let mut sc = SolverControls::default();
    read_solver_controls(&mut sc, &fv_solution(body), "p")?;
    Ok(sc)
}

/// What one run of the backend gave: the performance of each solve, the
/// answer of the last, the hierarchy's level sizes and how many it built.
struct Run {
    perf: Vec<SolverPerformance>,
    psi: Vec<Scalar>,
    sizes: Option<Vec<usize>>,
    builds: usize,
}

struct Sys {
    gpu: Gpu,
    hm: HostMesh,
    m: GpuMesh,
}

impl Sys {
    fn new(n: usize) -> Option<Self> {
        let gpu = gpu()?;
        let hm = box_mesh(n);
        let m = GpuMesh::upload(&gpu, &hm).expect("mesh upload");
        Some(Self { gpu, hm, m })
    }

    /// Solve each matrix of `mats` once, from zero, through one backend set
    /// up with `ctrl`.
    fn run(&self, ctrl: SolverControls, mats: &[&HostLduMatrix]) -> Run {
        let mut be = PbicgstabBackend::new(ctrl);
        be.setup(&self.gpu, &self.hm, &self.m, &SystemProbe::default())
            .expect("setup");
        let mut perf = Vec::new();
        let mut psi = Vec::new();
        for a in mats {
            let ga: GpuLduMatrix = a.upload(&self.gpu).expect("matrix upload");
            let mut x: DevBuf<Scalar> = self.gpu.zeros(self.hm.n_cells).expect("psi");
            perf.push(be.solve(&self.gpu, &mut x, &ga, &self.m).expect("solve"));
            psi = self.gpu.download(&x).expect("download");
        }
        Run {
            perf,
            psi,
            sizes: be.gamg_sizes(),
            builds: be.gamg_builds(),
        }
    }
}

/// The common lines of every `p` entry below: tolerance `1e-8`, no
/// relative tolerance, and room to converge.
const TAIL: &str = "        tolerance 1e-8;\n        relTol 0;\n        maxIter 2000;";

fn body(lines: &str) -> String {
    format!("{lines}\n{TAIL}")
}

/// SPEC-LIT §141.7: the solver and the preconditioner a case names are the
/// ones the pressure backend runs. `PBiCGStab`, `PCG` and `GAMG` as the
/// solver, and `GAMG` as the preconditioner of PCG and PBiCGStab, both as
/// a bare word and in OpenFOAM's sub-dictionary form: each converges, a
/// hierarchy is attached exactly when GAMG is named, and the GAMG rows take
/// other iteration counts than the plain Krylov rows.
#[test]
fn the_solver_a_case_asks_for_is_the_solver_it_gets() {
    let rows: [(&str, &str, LinearSolverKind, Preconditioner); 6] = [
        ("PBiCGStab/diagonal", "        solver PBiCGStab;\n        preconditioner diagonal;",
            LinearSolverKind::PBiCGStab, Preconditioner::Diagonal),
        ("PCG/DIC", "        solver PCG;\n        preconditioner DIC;",
            LinearSolverKind::PCG, Preconditioner::Dic),
        ("GAMG", "        solver GAMG;",
            LinearSolverKind::Gamg, Preconditioner::Diagonal),
        ("PCG/GAMG", "        solver PCG;\n        preconditioner GAMG;",
            LinearSolverKind::PCG, Preconditioner::Gamg),
        ("PBiCGStab/GAMG", "        solver PBiCGStab;\n        preconditioner GAMG;",
            LinearSolverKind::PBiCGStab, Preconditioner::Gamg),
        ("PCG/{GAMG}", "        solver PCG;\n        preconditioner\n        {\n            preconditioner GAMG;\n            smoother l1Jacobi;\n        }",
            LinearSolverKind::PCG, Preconditioner::Gamg),
    ];
    for (what, lines, kind, precon) in rows {
        let sc = controls(&body(lines)).expect(what);
        assert_eq!((sc.solver, sc.precon), (kind, precon), "{what}");
    }
    // The sub-dictionary's own settings are read from inside it.
    let sub = controls(&body(rows[5].1)).expect("sub-dictionary");
    assert_eq!(
        ofgpu::io::case::gamg_smoother_name(sub.gamg.cycle.hierarchy.smoother),
        "l1Jacobi"
    );

    let Some(sys) = Sys::new(16) else {
        return;
    };
    let a = poi(&sys.hm, 1.0);
    let mut iters = Vec::new();
    for (what, lines, kind, precon) in rows {
        let r = sys.run(controls(&body(lines)).expect(what), &[&a]);
        let p = &r.perf[0];
        println!("{what}: {} iterations, final residual {:.3e}", p.n_iterations, p.final_residual);
        assert!(p.converged, "{what}: {p:?}");
        let gamg = kind == LinearSolverKind::Gamg || precon == Preconditioner::Gamg;
        assert_eq!(r.builds, usize::from(gamg), "{what}: hierarchies built");
        assert_eq!(r.sizes.is_some(), gamg, "{what}: a hierarchy is attached iff GAMG is named");
        iters.push(p.n_iterations);
    }
    // GAMG as the solver, and as PCG's preconditioner, take other counts
    // than the Krylov rows they replace.
    assert_ne!(iters[2], iters[0], "GAMG vs PBiCGStab/diagonal");
    assert_ne!(iters[3], iters[1], "PCG/GAMG vs PCG/DIC");
    assert_ne!(iters[4], iters[0], "PBiCGStab/GAMG vs PBiCGStab/diagonal");
}

/// SPEC-LIT §13.4.1, one pair per GAMG key: two entries differing in ONE
/// line give a different iteration count (`smoother`, `nPreSweeps`,
/// `nPostSweeps`), a different hierarchy (`nCellsInCoarsestLevel`,
/// `maxLevels`), or a different number of hierarchies built over two
/// solves of one matrix with the same answer bit for bit
/// (`cacheAgglomeration`).
#[test]
fn every_gamg_key_reaches_the_solve() {
    let Some(sys) = Sys::new(16) else {
        return;
    };
    let a = poi(&sys.hm, 1.0);
    let run = |lines: &str, mats: &[&HostLduMatrix]| {
        sys.run(controls(&body(lines)).expect(lines), mats)
    };
    let iters = |r: &Run| r.perf[0].n_iterations;

    // smoother: PCG's preconditioner, V(2,2) with Chebyshev, then l1-Jacobi.
    let x = run("        solver PCG;\n        preconditioner GAMG;\n        smoother Chebyshev;", &[&a]);
    let y = run("        solver PCG;\n        preconditioner GAMG;\n        smoother l1Jacobi;", &[&a]);
    println!("smoother Chebyshev {} / l1Jacobi {}", iters(&x), iters(&y));
    assert!(x.perf[0].converged && y.perf[0].converged);
    assert_ne!(iters(&x), iters(&y), "smoother must move the iteration count");

    // nPreSweeps and nPostSweeps: the standalone GAMG solver, V(2,2)
    // against V(1,2) and V(2,1).
    let v22 = run("        solver GAMG;\n        nPreSweeps 2;\n        nPostSweeps 2;", &[&a]);
    let v12 = run("        solver GAMG;\n        nPreSweeps 1;\n        nPostSweeps 2;", &[&a]);
    let v21 = run("        solver GAMG;\n        nPreSweeps 2;\n        nPostSweeps 1;", &[&a]);
    println!("V(2,2) {} / V(1,2) {} / V(2,1) {}", iters(&v22), iters(&v12), iters(&v21));
    for r in [&v22, &v12, &v21] {
        assert!(r.perf[0].converged, "{:?}", r.perf[0]);
    }
    assert_ne!(iters(&v22), iters(&v12), "nPreSweeps must move the iteration count");
    assert_ne!(iters(&v22), iters(&v21), "nPostSweeps must move the iteration count");

    // nCellsInCoarsestLevel and maxLevels: the hierarchy itself.
    let deep = run("        solver GAMG;\n        nCellsInCoarsestLevel 64;", &[&a]);
    let shallow = run("        solver GAMG;\n        nCellsInCoarsestLevel 512;", &[&a]);
    let two = run("        solver GAMG;\n        maxLevels 2;", &[&a]);
    let full = run("        solver GAMG;\n        maxLevels 50;", &[&a]);
    println!(
        "nCellsInCoarsestLevel 64 {:?} / 512 {:?}; maxLevels 50 {:?} / 2 {:?}",
        deep.sizes, shallow.sizes, full.sizes, two.sizes
    );
    let (d, s) = (deep.sizes.clone().expect("deep"), shallow.sizes.clone().expect("shallow"));
    assert_ne!(d, s, "nCellsInCoarsestLevel must move the hierarchy");
    assert!(*d.last().expect("levels") <= 64 && *s.last().expect("levels") <= 512);
    let (f, t) = (full.sizes.clone().expect("full"), two.sizes.clone().expect("two"));
    assert_ne!(f, t, "maxLevels must move the hierarchy");
    assert_eq!(t.len(), 2, "maxLevels 2 is the fine level and one coarse level");
    for r in [&deep, &shallow, &two, &full] {
        assert!(r.perf[0].converged, "{:?}", r.perf[0]);
    }

    // cacheAgglomeration: two solves of one matrix build one hierarchy
    // with it on and two with it off, to the same answer bit for bit; a
    // changed matrix is rebuilt either way.
    let on = run("        solver GAMG;\n        cacheAgglomeration yes;", &[&a, &a]);
    let off = run("        solver GAMG;\n        cacheAgglomeration no;", &[&a, &a]);
    println!("cacheAgglomeration yes {} builds / no {} builds", on.builds, off.builds);
    assert_eq!((on.builds, off.builds), (1, 2), "cacheAgglomeration must move the builds");
    let bits = |v: &[Scalar]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
    assert_eq!(bits(&on.psi), bits(&off.psi), "the cache must not move the answer");
    let b = poi(&sys.hm, 1.5);
    let changed = run("        solver GAMG;\n        cacheAgglomeration yes;", &[&a, &b]);
    assert_eq!(changed.builds, 2, "a changed matrix must be rebuilt");
    let fresh = run("        solver GAMG;\n        cacheAgglomeration no;", &[&b]);
    assert_eq!(bits(&changed.psi), bits(&fresh.psi), "the rebuilt hierarchy is the fresh one");
}

/// SPEC-LIT §141.7 and §13.4: the unsupported values are refused by name,
/// with the alternative in the message, and a malformed count or switch is
/// refused naming its key.
#[test]
fn unsupported_gamg_values_are_refused_by_name_with_the_alternative() {
    let _guard = ofgpu::io::contract::permissive_test_guard();
    ofgpu::io::contract::set_permissive(false);
    let refused = |lines: &str, needles: &[&str]| {
        let e = controls(&body(lines)).expect_err(lines).to_string();
        for n in needles {
            assert!(e.contains(n), "`{lines}` must name `{n}`: {e}");
        }
    };
    for s in ["GaussSeidel", "DICGaussSeidel", "symGaussSeidel", "DIC"] {
        refused(
            &format!("        solver GAMG;\n        smoother {s};"),
            &["solvers/p/smoother", s, "Chebyshev", "l1Jacobi", "-permissive"],
        );
    }
    refused(
        "        solver GAMG;\n        mergeLevels 2;",
        &["solvers/p/mergeLevels", "\"2\"", "nCellsInCoarsestLevel", "-permissive"],
    );
    refused(
        "        solver GAMG;\n        agglomerator faceAreaPair;",
        &["solvers/p/agglomerator", "faceAreaPair", "pairwise", "remove the entry"],
    );
    // The sub-dictionary form refuses under its own path.
    refused(
        "        solver PCG;\n        preconditioner\n        {\n            preconditioner GAMG;\n            smoother DICGaussSeidel;\n        }",
        &["solvers/p/preconditioner/smoother", "DICGaussSeidel", "Chebyshev"],
    );
    refused(
        "        solver GAMG;\n        nPreSweeps 0;\n        nPostSweeps 0;",
        &["nPreSweeps", "nPostSweeps", "both 0"],
    );
    refused("        solver GAMG;\n        maxLevels 0;", &["solvers/p/maxLevels", "at least 1"]);
    refused(
        "        solver GAMG;\n        nCellsInCoarsestLevel 2.5;",
        &["solvers/p/nCellsInCoarsestLevel", "whole number"],
    );
    refused(
        "        solver GAMG;\n        cacheAgglomeration maybe;",
        &["solvers/p/cacheAgglomeration", "switch"],
    );

    // Honoured: mergeLevels 1 (no merging, what the hierarchy does), and
    // under -permissive a refused smoother becomes the announced Chebyshev.
    let ok = controls(&body("        solver GAMG;\n        mergeLevels 1;")).expect("mergeLevels 1");
    assert_eq!(ok.gamg, ofgpu::io::case::GamgControls::default());
    ofgpu::io::contract::set_permissive(true);
    let p = controls(&body("        solver GAMG;\n        smoother GaussSeidel;"))
        .expect("permissive substitutes");
    assert_eq!(ofgpu::io::case::gamg_smoother_name(p.gamg.cycle.hierarchy.smoother), "Chebyshev");
    ofgpu::io::contract::set_permissive(false);
}
