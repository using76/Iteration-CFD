// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.
// Provenance: see PROVENANCE.md. No GPL-licensed source was consulted.

//! AMG-05: the native GAMG is a pressure backend of its own and the
//! selector measures it (SPEC-LIT §141.8). On `POI(16)` of (141.5.5),
//! `b = 1`: a later matrix is re-gathered onto the kept aggregation, and
//! is the fresh build bit for bit when it would aggregate the same way;
//! `cacheAgglomeration no` builds every changed matrix; the selector's
//! agreement trial is solved to the reference's tolerance, so a GAMG
//! backend under a loose case entry still agrees to `1e-8`, and its row
//! carries a set-up time.
//!
//! An integration test, so the same assertions run in both builds (§118).
//! A machine without a card passes vacuously, as `tests/gamg_case.rs`.
//!
//! Provenance: ORIGINAL - the fixtures and the assertions are this tree's
//! own construction, run against SPEC-LIT §141.8. OpenFOAM's GAMG was not
//! opened. No GPL-licensed source was consulted.

use ofgpu::blockgen::{BlockSpec, GradedAxis, build_mesh};
use ofgpu::device::DevBuf;
use ofgpu::io::case::{LinearSolverKind, Preconditioner, SolverControls};
use ofgpu::ldu::{GpuLduMatrix, HostLduMatrix};
use ofgpu::mesh::{GpuMesh, HostMesh};
use ofgpu::pressure::{
    choose_pressure_backend, default_candidates, GamgBackend, PressureBackend, SystemProbe,
};
use ofgpu::solver::SolverPerformance;
use ofgpu::{Gpu, Scalar};

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

/// `POI(n)` of (141.5.5) with every face coefficient times `w(x_f)` and
/// the diagonal times `s`; `w = 1, s = 1` is the Dirichlet Poisson matrix.
fn poi(m: &HostMesh, s: Scalar, w: impl Fn(Scalar) -> Scalar) -> HostLduMatrix {
    let mut a = HostLduMatrix::zeros(m);
    for f in 0..m.n_internal_faces {
        let g = m.mag_sf[f] * m.delta_coeffs[f] * w(m.cf[f].x);
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

fn ctrl(tol: Scalar, rel: Scalar, cache: bool) -> SolverControls {
    let mut c = SolverControls {
        solver: LinearSolverKind::PBiCGStab,
        precon: Preconditioner::Diagonal,
        tolerance: tol,
        rel_tol: rel,
        max_iter: 2000,
        ..SolverControls::default()
    };
    c.gamg.cache_agglomeration = cache;
    c
}

fn symmetric() -> SystemProbe {
    SystemProbe {
        symmetric: true,
        ..SystemProbe::default()
    }
}

struct Sys {
    gpu: Gpu,
    hm: HostMesh,
    m: GpuMesh,
}

struct Run {
    perf: Vec<SolverPerformance>,
    psi: Vec<Scalar>,
    sizes: Vec<Option<Vec<usize>>>,
    builds: usize,
    regathers: usize,
}

impl Sys {
    fn new(n: usize) -> Option<Self> {
        let gpu = Gpu::new(0).ok()?;
        let hm = box_mesh(n);
        let m = GpuMesh::upload(&gpu, &hm).expect("mesh upload");
        Some(Self { gpu, hm, m })
    }

    /// Each matrix of `mats` solved once, from zero, by one backend.
    fn run(&self, c: SolverControls, mats: &[&HostLduMatrix]) -> Run {
        let mut be = GamgBackend::new(c);
        be.setup(&self.gpu, &self.hm, &self.m, &symmetric()).expect("setup");
        assert_eq!(be.run_controls().solver, LinearSolverKind::PCG);
        assert_eq!(be.run_controls().precon, Preconditioner::Gamg);
        let (mut perf, mut psi, mut sizes) = (Vec::new(), Vec::new(), Vec::new());
        for a in mats {
            let ga: GpuLduMatrix = a.upload(&self.gpu).expect("matrix upload");
            let mut x: DevBuf<Scalar> = self.gpu.zeros(self.hm.n_cells).expect("psi");
            perf.push(be.solve(&self.gpu, &mut x, &ga, &self.m).expect("solve"));
            psi = self.gpu.download(&x).expect("download");
            sizes.push(be.sizes());
        }
        Run { perf, psi, sizes, builds: be.builds(), regathers: be.regathers() }
    }
}

fn bits(v: &[Scalar]) -> Vec<u64> {
    v.iter().map(|x| f64::from(*x).to_bits()).collect()
}

/// SPEC-LIT §141.8: the diagonal alone moves, so the new matrix aggregates
/// as the first did (the strength of §141.1 reads the couplings), and the
/// re-gather is the fresh build of it: one build, one re-gather, the
/// answer bit for bit a fresh backend's on the scaled matrix.
#[test]
fn a_regather_on_the_same_aggregation_is_the_fresh_build() {
    let Some(sys) = Sys::new(16) else { return };
    let a = poi(&sys.hm, 1.0, |_| 1.0);
    let b = poi(&sys.hm, 1.5, |_| 1.0);
    let c = ctrl(1e-8, 0.0, true);
    let two = sys.run(c, &[&a, &b]);
    let fresh = sys.run(c, &[&b]);
    println!(
        "regather: builds {} regathers {} iterations {:?}, fresh {:?}",
        two.builds,
        two.regathers,
        two.perf.iter().map(|p| p.n_iterations).collect::<Vec<_>>(),
        fresh.perf[0].n_iterations
    );
    assert_eq!((two.builds, two.regathers), (1, 1));
    assert_eq!((fresh.builds, fresh.regathers), (1, 0));
    assert!(two.perf.iter().all(|p| p.converged), "{:?}", two.perf);
    assert_eq!(two.sizes[1], fresh.sizes[0], "the kept hierarchy is the fresh one");
    assert_eq!(two.perf[1].n_iterations, fresh.perf[0].n_iterations);
    assert_eq!(bits(&two.psi), bits(&fresh.psi), "re-gathered answer == fresh build");
}

/// SPEC-LIT §141.8: the couplings move (`w = 1 + x/16`), the aggregation
/// is kept and re-gathered with the new values, the level sizes stay those
/// of the first build, and the solve converges to the tolerance. With
/// `cacheAgglomeration no` the same pair costs two builds.
#[test]
fn a_changed_coupling_is_regathered_onto_the_kept_aggregation() {
    let Some(sys) = Sys::new(16) else { return };
    let a = poi(&sys.hm, 1.0, |_| 1.0);
    let b = poi(&sys.hm, 1.0, |x| 1.0 + x / 16.0);
    let on = sys.run(ctrl(1e-8, 0.0, true), &[&a, &b]);
    let off = sys.run(ctrl(1e-8, 0.0, false), &[&a, &b]);
    println!(
        "changed coupling: cache on {:?} iterations, off {:?}; sizes {:?}",
        on.perf.iter().map(|p| p.n_iterations).collect::<Vec<_>>(),
        off.perf.iter().map(|p| p.n_iterations).collect::<Vec<_>>(),
        on.sizes[0]
    );
    assert_eq!((on.builds, on.regathers), (1, 1));
    assert_eq!((off.builds, off.regathers), (2, 0));
    assert_eq!(on.sizes[0], on.sizes[1], "the aggregation is kept");
    for p in on.perf.iter().chain(off.perf.iter()) {
        assert!(p.converged && p.final_residual <= 1e-8, "{p:?}");
    }
    // A second solve of the same matrix reuses it: no build, no re-gather.
    let same = sys.run(ctrl(1e-8, 0.0, true), &[&a, &a]);
    assert_eq!((same.builds, same.regathers), (1, 0));
}

/// SPEC-LIT §141.8: `default_candidates` lists GAMG, and the selector
/// solves its agreement trial to the reference's tolerance: under a loose
/// case entry (`1e-3`, `relTol 0.1`), which would stop well short of
/// `1e-8`, the GAMG row still agrees to `1e-8` and is timed, and it
/// carries a set-up time. Which backend wins is a measurement and is not
/// asserted.
#[test]
fn the_selector_measures_gamg_with_its_set_up() {
    let Some(sys) = Sys::new(16) else { return };
    let a = poi(&sys.hm, 1.0, |_| 1.0);
    let ga = a.upload(&sys.gpu).expect("matrix upload");
    let cands = default_candidates(ctrl(1e-3, 0.1, true));
    let names: Vec<&str> = cands.iter().map(|b| b.name()).collect();
    assert_eq!(names, ["PBiCGStab", "GAMG", "cuFFT", "AMGX"]);
    let (_chosen, choice) =
        choose_pressure_backend(&sys.gpu, &sys.hm, &sys.m, &ga, &symmetric(), cands)
            .expect("selection");
    let report = choice.report();
    println!("{report}");
    let row = choice.considered.iter().position(|c| c.0 == "GAMG").expect("GAMG row");
    assert!(choice.considered[row].1, "{report}");
    assert!(choice.considered[row].2.is_some(), "GAMG is timed: {report}");
    assert!(choice.notes[row].starts_with("agrees to"), "{report}");
    assert!(choice.setup[row].is_some_and(|s| s > 0.0), "{report}");
    assert!(report.contains("set-up"), "{report}");
}
