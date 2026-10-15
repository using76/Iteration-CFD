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
//! (dFFI, class D-F32-1). SPEC-LIT §141.21 adds the re-set-up: a second
//! matrix of the same pattern with other couplings, through one backend,
//! is one full set-up and one re-set-up on the kept structure with reuse on
//! and two full set-ups with it off, each agreeing to the same band.
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
use ofgpu::{
    BcKind, DevBuf, Gpu, GpuLduMatrix, GpuMesh, GpuScalarField, HostMesh, Label, Scalar,
};

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
/// The plume case's own box with `p`'s boundary conditions: zeroGradient
/// everywhere but `xMax` (+x), the outlet that pins the system, fixedValue
/// with fr = 1 and reference value 0.
struct Plume {
    hm: HostMesh,
    m: GpuMesh,
    p: GpuScalarField,
}

fn plume(gpu: &Gpu) -> Plume {
    let spec = box_spec([98, 42, 20], [0.0; 3], [14.64, 6.24, 3.0]);
    let hm = blockgen::build_mesh(&spec).expect("mesh");
    assert_eq!(hm.n_cells, 82_320);
    let m = GpuMesh::upload(gpu, &hm).expect("upload");

    let mut p = GpuScalarField::zeros(gpu, &m, "p").expect("p");
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
    Plume { hm, m, p }
}

/// The separable laplacian with `Γ = 0.37` times face area, internal face
/// `i` further times `face_scale(i)` (boundary faces unscaled), plus its
/// boundary contributions and the noise source, with the probe of it.
fn assemble(
    gpu: &Gpu,
    pl: &Plume,
    face_scale: impl Fn(usize) -> Scalar,
) -> (GpuLduMatrix, SystemProbe) {
    let hm = &pl.hm;
    let gamma: Vec<Scalar> = hm
        .mag_sf
        .iter()
        .enumerate()
        .map(|(i, s)| GAMMA * s * face_scale(i))
        .collect();
    let b_gamma: Vec<Scalar> = hm.b_mag_sf.iter().map(|s| GAMMA * s).collect();
    let gamma = gpu.upload(&gamma).expect("gamma");
    let b_gamma = gpu.upload(&b_gamma).expect("b_gamma");

    let fvk = FvKernels::new(gpu).expect("fv kernels");
    let lduk = LduKernels::new(gpu).expect("ldu kernels");
    let mut a = GpuLduMatrix::new(gpu, &pl.m).expect("matrix");
    a.zero(gpu).expect("zero");
    fv::fvm_laplacian(gpu, &fvk, &mut a, &pl.m, &gamma, &b_gamma, &pl.p, 1.0).expect("laplacian");
    ldu_ops::add_boundary_contributions(gpu, &lduk, &mut a, &pl.m).expect("boundary");

    // A source with no shape the eye can check, added on top of whatever
    // the Dirichlet faces contributed - the fixedValue faces put theirs in
    // already, and the outlet pins the mean, so nothing needs removing.
    let mut src = gpu.download(&a.source).expect("source");
    for (s, e) in src.iter_mut().zip(&noise(hm.n_cells, 20250822)) {
        *s += *e;
    }
    gpu.write(&mut a.source, &src).expect("write source");

    let probe = SystemProbe::probe(gpu, hm, &pl.p, &a, &gamma, &b_gamma).expect("probe");
    (a, probe)
}

/// The PBiCGStab reference's answer, which decides whether a backend's is
/// right.
fn reference(gpu: &Gpu, pl: &Plume, a: &GpuLduMatrix, probe: &SystemProbe) -> Vec<Scalar> {
    let mut refb = PbicgstabBackend::reference();
    refb.setup(gpu, &pl.hm, &pl.m, probe).expect("ref setup");
    let mut psi_ref: DevBuf<Scalar> = gpu.zeros(pl.hm.n_cells).expect("psi_ref");
    refb.solve(gpu, &mut psi_ref, a, &pl.m).expect("ref solve");
    gpu.download(&psi_ref).expect("download")
}

/// `‖x - x_ref‖∞ / ‖x_ref‖∞`, in f64 whichever build.
fn rel_diff(x: &[Scalar], x_ref: &[Scalar]) -> f64 {
    let num = x
        .iter()
        .zip(x_ref)
        .fold(0.0f64, |m, (x, r)| m.max((*x as f64 - *r as f64).abs()));
    let den = x_ref.iter().fold(0.0f64, |m, r| m.max(r.abs() as f64));
    num / den
}

fn build_name() -> &'static str {
    if cfg!(feature = "single") {
        "single"
    } else {
        "f64"
    }
}

#[test]
fn amgx_agrees_with_pbicgstab_on_the_plume_matrix() {
    let Some(gpu) = gpu() else { return };
    if !amgx::require_loaded().expect("AMGX required") {
        return;
    }
    let pl = plume(&gpu);
    let (a, probe) = assemble(&gpu, &pl, |_| 1.0);

    let mut am = AmgxBackend::new();
    assert!(
        am.applicable(&probe),
        "the plume matrix must suit AMGX: {}",
        am.why_not(&probe)
    );
    am.setup(&gpu, &pl.hm, &pl.m, &probe).expect("amgx setup");
    let mut psi: DevBuf<Scalar> = gpu.zeros(pl.hm.n_cells).expect("psi");
    let perf = am.solve(&gpu, &mut psi, &a, &pl.m).expect("amgx solve");
    let x_amgx: Vec<Scalar> = gpu.download(&psi).expect("download");

    let rel = rel_diff(&x_amgx, &reference(&gpu, &pl, &a, &probe));
    eprintln!(
        "amgx plume matrix: build={} iterations={} converged={} rel={rel:.3e}",
        build_name(),
        perf.n_iterations,
        perf.converged
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

/// SPEC-LIT §141.21: a second matrix with the same pattern and different
/// couplings, through ONE backend. With reuse on it is a re-set-up on the
/// kept structure, (141.21b); with reuse off a second full set-up,
/// (141.21c). Either way the answer is the PBiCGStab reference's to the
/// band of §141.20.1. The times are printed, not asserted: one card is
/// shared by three lanes.
#[test]
fn a_resetup_on_the_kept_structure_agrees_with_pbicgstab() {
    let Some(gpu) = gpu() else { return };
    if !amgx::require_loaded().expect("AMGX required") {
        return;
    }
    let pl = plume(&gpu);
    let (a1, probe) = assemble(&gpu, &pl, |_| 1.0);
    // Strongly varying couplings, so the strength of connection - and with
    // it a fresh coarsening - would differ from A1's.
    let (a2, probe2) =
        assemble(&gpu, &pl, |i| 1.0 + 0.5 * ((i * 7919) % 1000) as Scalar / 1000.0);
    let x_ref2 = reference(&gpu, &pl, &a2, &probe2);

    for (reuse, counts) in [(true, (1, 1)), (false, (2, 0))] {
        let mut am = AmgxBackend::new().with_reuse(reuse);
        assert_eq!(am.reuse(), reuse);
        am.setup(&gpu, &pl.hm, &pl.m, &probe).expect("amgx setup");
        let mut psi: DevBuf<Scalar> = gpu.zeros(pl.hm.n_cells).expect("psi");
        let perf1 = am.solve(&gpu, &mut psi, &a1, &pl.m).expect("amgx solve A1");
        let first = am.setup_secs().expect("a set-up time after the first solve");
        let perf2 = am.solve(&gpu, &mut psi, &a2, &pl.m).expect("amgx solve A2");
        let second = am.setup_secs().expect("a set-up time after the second solve");
        let x2: Vec<Scalar> = gpu.download(&psi).expect("download");
        let rel = rel_diff(&x2, &x_ref2);
        eprintln!(
            "amgx re-set-up: build={} reuse={reuse} iterations {} then {} set-up {:.2} ms then \
             {:.2} ms rel={rel:.3e} | {}",
            build_name(),
            perf1.n_iterations,
            perf2.n_iterations,
            1e3 * first,
            1e3 * second,
            am.run_summary().unwrap_or_default()
        );
        assert_eq!(am.setup_counts(), counts, "reuse = {reuse}: (full set-ups, re-set-ups)");
        if !cfg!(feature = "single") {
            assert!(perf1.converged && perf2.converged, "reuse = {reuse}: AMGX did not converge");
        }
        assert!(
            rel <= TOL,
            "reuse = {reuse}: AMGX and PBiCGStab differ by {rel:.3e} relative on A2, above {TOL:.0e}"
        );
    }
}
