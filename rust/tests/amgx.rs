// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.
// Provenance: see PROVENANCE.md. No GPL-licensed source was consulted.

//! SPEC-LIT §141.20 - the AMGX pressure backend, opened at run time, against
//! the PBiCGStab reference on the plume-sized Poisson matrix: 98 x 42 x 20
//! cells over 14.64 x 6.24 x 3 m, `Γ = 0.37` times face area, `fixedValue`
//! on `+x`, `zeroGradient` everywhere else, and a deterministic noise source
//! so the answer has no shape the eye can check. The gate is the band of
//! §141.20.1: `1e-8` relative in the f64 build (dDDI), `1e-5` under `single`
//! (dFFI, class D-F32-1).
//!
//! An integration test, because the library's own test target does not
//! compile under `single`; an integration test links the normal (non-test)
//! library, which does. A machine without a card, or without the DLL, passes
//! vacuously - `OFGPU_REQUIRE_AMGX=1` is what turns absence into a failure.
//!
//! Provenance: ORIGINAL - the matrix, the boundary conditions and the noise
//! source are this tree's own construction.
//! No GPL-licensed source was consulted.

#![cfg(feature = "amgx")]

use ofgpu::blockgen::{self, BlockSpec, GradedAxis};
use ofgpu::fv::{self, FvKernels};
use ofgpu::ldu_ops::{self, LduKernels};
use ofgpu::pressure::{amgx, AmgxBackend, PbicgstabBackend, PressureBackend, SystemProbe};
use ofgpu::{BcKind, DevBuf, Gpu, GpuLduMatrix, GpuMesh, GpuScalarField, Label, Scalar};

/// The agreement band of §141.20.1 (class R, D-F32-1): `1e-8` relative with
/// dDDI in the f64 build, `1e-5` with dFFI under `single`.
const TOL: f64 = if cfg!(feature = "single") { 1e-5 } else { 1e-8 };

/// The laplacian coefficient of the plume gate - not 1, so a backend that
/// dropped the coefficient could not hide (the unit tests' GAMMA).
const GAMMA: Scalar = 0.37;

/// Every device test needs a card; without one the test passes vacuously.
fn gpu() -> Option<Gpu> {
    Gpu::new(0).ok()
}

/// A uniform block, every side a plain `patch` - the `box_spec` shape of
/// `tests/f32_boussinesq.rs`, with no wall or empty kind in the way of the
/// boundary conditions this gate puts on `xMax`.
fn box_spec(n: [usize; 3], lo: [Scalar; 3], hi: [Scalar; 3]) -> BlockSpec {
    let axis = |lo: Scalar, hi: Scalar, n: usize| GradedAxis {
        lo,
        hi,
        n,
        expansion: 1.0,
        two_sided: false,
    };
    BlockSpec {
        x: axis(lo[0], hi[0], n[0]),
        y: axis(lo[1], hi[1], n[1]),
        z: axis(lo[2], hi[2], n[2]),
        windows: Vec::new(),
        patch_name: BlockSpec::default().patch_name,
        patch_type: ["patch"; 6].map(String::from),
        cyclic: Vec::new(),
    }
}

/// Deterministic pseudo-random values - the same LCG the unit tests use, so
/// the source term is the same numbers every run and a failure is
/// reproducible.
fn noise(n: usize, seed: u64) -> Vec<Scalar> {
    let mut s = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
    (0..n)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((s >> 33) as f64 / (1u64 << 31) as f64 - 1.0) as Scalar
        })
        .collect()
}

#[test]
fn amgx_agrees_with_pbicgstab_on_the_plume_matrix() {
    let Some(gpu) = gpu() else { return };
    if !amgx::require_loaded().expect("AMGX required") {
        return;
    }

    // The mesh, the plume case's own box.
    let spec = box_spec([98, 42, 20], [0.0; 3], [14.64, 6.24, 3.0]);
    let hm = blockgen::build_mesh(&spec).expect("mesh");
    assert_eq!(hm.n_cells, 82_320);
    let m = GpuMesh::upload(&gpu, &hm).expect("upload");

    // zeroGradient everywhere but `xMax` (+x), the outlet that pins the
    // system, fixedValue with fr = 1 and reference value 0.
    let mut p = GpuScalarField::zeros(&gpu, &m, "p").expect("p");
    let x_max = hm
        .patches
        .iter()
        .position(|patch| patch.name == "xMax")
        .expect("a block has an xMax patch");
    let mut kind = vec![BcKind::ZeroGradient as Label; hm.n_boundary_faces];
    let mut fr = vec![0.0 as Scalar; hm.n_boundary_faces];
    let ref_value = vec![0.0 as Scalar; hm.n_boundary_faces];
    for bf in 0..hm.n_boundary_faces {
        if hm.b_patch[bf] as usize == x_max {
            kind[bf] = BcKind::FixedValue as Label;
            fr[bf] = 1.0;
        }
    }
    gpu.write(&mut p.bc_kind, &kind).expect("bc_kind");
    gpu.write(&mut p.fr, &fr).expect("fr");
    gpu.write(&mut p.ref_value, &ref_value).expect("ref_value");

    // Γ = 0.37 times face area, internal and boundary alike.
    let gamma: Vec<Scalar> = hm.mag_sf.iter().map(|s| GAMMA * s).collect();
    let b_gamma: Vec<Scalar> = hm.b_mag_sf.iter().map(|s| GAMMA * s).collect();
    let gamma = gpu.upload(&gamma).expect("gamma");
    let b_gamma = gpu.upload(&b_gamma).expect("b_gamma");

    // The separable laplacian, plus its boundary contributions.
    let fvk = FvKernels::new(&gpu).expect("fv kernels");
    let lduk = LduKernels::new(&gpu).expect("ldu kernels");
    let mut a = GpuLduMatrix::new(&gpu, &m).expect("matrix");
    a.zero(&gpu).expect("zero");
    fv::fvm_laplacian(&gpu, &fvk, &mut a, &m, &gamma, &b_gamma, &p, 1.0).expect("laplacian");
    ldu_ops::add_boundary_contributions(&gpu, &lduk, &mut a, &m).expect("boundary");

    // A source with no shape the eye can check, added on top of whatever
    // the Dirichlet faces contributed - the fixedValue faces put theirs in
    // already, and the outlet pins the mean, so nothing needs removing.
    let mut src = gpu.download(&a.source).expect("source");
    for (s, e) in src.iter_mut().zip(&noise(hm.n_cells, 20250822)) {
        *s += *e;
    }
    gpu.write(&mut a.source, &src).expect("write source");

    let probe = SystemProbe::probe(&gpu, &hm, &p, &a, &gamma, &b_gamma).expect("probe");

    // AMGX solves it; the PBiCGStab reference decides whether the answer
    // is right.
    let mut am = AmgxBackend::new();
    assert!(
        am.applicable(&probe),
        "the plume matrix must suit AMGX: {}",
        am.why_not(&probe)
    );
    am.setup(&gpu, &hm, &m, &probe).expect("amgx setup");
    let mut psi: DevBuf<Scalar> = gpu.zeros(hm.n_cells).expect("psi");
    let perf = am.solve(&gpu, &mut psi, &a, &m).expect("amgx solve");
    let x_amgx: Vec<Scalar> = gpu.download(&psi).expect("download");

    let mut refb = PbicgstabBackend::reference();
    refb.setup(&gpu, &hm, &m, &probe).expect("ref setup");
    let mut psi_ref: DevBuf<Scalar> = gpu.zeros(hm.n_cells).expect("psi_ref");
    refb.solve(&gpu, &mut psi_ref, &a, &m).expect("ref solve");
    let x_ref: Vec<Scalar> = gpu.download(&psi_ref).expect("download");

    // ‖x_AMGX - x_ref‖∞ / ‖x_ref‖∞, in f64 whichever build.
    let num = x_amgx
        .iter()
        .zip(&x_ref)
        .fold(0.0f64, |m, (x, r)| m.max((*x as f64 - *r as f64).abs()));
    let den = x_ref.iter().fold(0.0f64, |m, r| m.max(r.abs() as f64));
    let rel = num / den;

    let build = if cfg!(feature = "single") { "single" } else { "f64" };
    eprintln!(
        "amgx plume matrix: build={build} iterations={} converged={} rel={rel:.3e}",
        perf.n_iterations, perf.converged
    );

    if !cfg!(feature = "single") {
        // dDDI can be asked for a converged solve; dFFI's tighter residual
        // floor cannot always reach AMGX's own 1e-12, so only f64 asserts it.
        assert!(perf.converged, "AMGX did not converge, rel = {rel:.3e}");
    }
    assert!(
        rel <= TOL,
        "AMGX and PBiCGStab differ by {rel:.3e} relative, above the {TOL:.0e} band"
    );
}

