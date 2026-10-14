// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.
// Provenance: see PROVENANCE.md. No GPL-licensed source was consulted.

//! SPEC-LIT §118.12 - the iterative refinement of §118.4 on the other
//! backends: the cuFFT direct solve, the pressure selector's probe, and the
//! distributed Krylov of §73 in process and SPMD (§138.5, §138.6). Five
//! tests, the SAME assertions in BOTH builds: the answers are held to f64
//! tolerances (precision class P/A of docs/17 §5.1 rule 4), which the f64
//! build meets as it always did and `single` meets through the refinement.
//!
//! An integration test, because the library's own test target does not
//! compile under `single`. A machine without a card - or, for the cuFFT
//! tests, without a loadable cuFFT - passes vacuously.
//!
//! Provenance: ORIGINAL - the systems and the harness are this tree's own;
//! the method is §118.4's (Moler 1967; Carson & Higham 2018) and the halo
//! split §118.12's (Dekker 1971). No GPL-licensed source was consulted.

use ofgpu::blockgen::{self, BlockSpec, GradedAxis};
use ofgpu::comm::{Comm, ThreadComm};
use ofgpu::decompose::{Decomposition, PartitionMethod};
use ofgpu::distsolve::{
    dist_pbicgstab, dist_pcg, spmd_pbicgstab, spmd_pcg, DistPerformance, DistReduce, DistSystem,
    DistWorkspace, SpmdWorkspace,
};
use ofgpu::fv::{fvm_laplacian, FvKernels};
use ofgpu::halo::{HaloExchange, SpmdHalo};
use ofgpu::ldu::HostLduMatrix;
use ofgpu::ldu_ops::{add_boundary_contributions, LduKernels};
use ofgpu::pressure::fft::cufft_available;
use ofgpu::pressure::{
    cartesian, choose_pressure_backend, FftBackend, PbicgstabBackend, PressureBackend, SystemProbe,
    AGREEMENT_TOL,
};
use ofgpu::reference as cpu;
use ofgpu::solver::{LinearSolverKind, Preconditioner, SolverControls, SolverKernels};
use ofgpu::{
    BcKind, DevBuf, Gpu, GpuLduMatrix, GpuMesh, GpuScalarField, HostMesh, Label, Result, Scalar,
};

/// Relative slack of a solution against its f64 reference.
const SLACK: f64 = 1e-10;

/// The tolerance the distributed solves ask for.
const SOLVE_TOL: f64 = 1e-14;

/// Every device test needs a card; without one the test passes vacuously.
fn gpu() -> Option<Gpu> {
    Gpu::new(0).ok()
}

/// The SplitMix64 finaliser of `seed * 0x9E3779B97F4A7C15 + k`, as in
/// `tests/f32_refinement.rs`.
fn h(seed: u64, k: u64) -> u64 {
    let mut z = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(k);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// An f64 value stored as the build's `Scalar` - exact for every value the
/// generators below make.
#[allow(clippy::useless_conversion)]
fn s(x: f64) -> Scalar {
    x as Scalar
}

fn axis(hi: Scalar, n: usize) -> GradedAxis {
    GradedAxis {
        lo: 0.0,
        hi,
        n,
        expansion: 1.0,
        two_sided: false,
    }
}

fn block(x: GradedAxis, y: GradedAxis, z: GradedAxis, kind: &str) -> BlockSpec {
    BlockSpec {
        x,
        y,
        z,
        windows: Vec::new(),
        patch_name: BlockSpec::default().patch_name,
        patch_type: [kind; 6].map(String::from),
        cyclic: Vec::new(),
    }
}

fn worst_rel(got: &[f64], want: &[f64]) -> f64 {
    let scale = want
        .iter()
        .fold(0.0f64, |m, v| m.max(v.abs()))
        .max(f64::MIN_POSITIVE);
    got.iter()
        .zip(want)
        .fold(0.0f64, |m, (g, w)| m.max((g - w).abs()))
        / scale
}

// ==========================================================================
//  The cuFFT direct solve and the selector
// ==========================================================================

/// A uniform 9 x 6 x 4 box, the laplacian of `0.37 p` with `fixedValue` on
/// -x and +y and `zeroGradient` elsewhere - section 6's "mixed sides" - plus
/// a deterministic source.
struct FftSys {
    hm: HostMesh,
    gm: GpuMesh,
    a: GpuLduMatrix,
    probe: SystemProbe,
}

fn fft_system(gpu: &Gpu) -> Result<FftSys> {
    let hm = blockgen::build_mesh(&block(axis(0.9, 9), axis(0.6, 6), axis(0.4, 4), "patch"))?;
    let gm = GpuMesh::upload(gpu, &hm)?;
    let grid = cartesian::detect(&hm).map_err(|e| ofgpu::Error::Config(e.to_string()))?;

    let mut p = GpuScalarField::zeros(gpu, &gm, "p")?;
    let mut kind = vec![BcKind::ZeroGradient as Label; hm.n_boundary_faces];
    let mut fr = vec![0.0 as Scalar; hm.n_boundary_faces];
    for bf in 0..hm.n_boundary_faces {
        if [0usize, 3].contains(&(grid.b_side[bf] as usize)) {
            kind[bf] = BcKind::FixedValue as Label;
            fr[bf] = 1.0;
        }
    }
    gpu.write(&mut p.bc_kind, &kind)?;
    gpu.write(&mut p.fr, &fr)?;

    let gamma: Vec<Scalar> = hm.mag_sf.iter().map(|v| 0.37 * v).collect();
    let b_gamma: Vec<Scalar> = hm.b_mag_sf.iter().map(|v| 0.37 * v).collect();
    let gamma = gpu.upload(&gamma)?;
    let b_gamma = gpu.upload(&b_gamma)?;
    let fvk = FvKernels::new(gpu)?;
    let lduk = LduKernels::new(gpu)?;
    let mut a = GpuLduMatrix::new(gpu, &gm)?;
    a.zero(gpu)?;
    fvm_laplacian(gpu, &fvk, &mut a, &gm, &gamma, &b_gamma, &p, 1.0)?;
    add_boundary_contributions(gpu, &lduk, &mut a, &gm)?;

    let mut src = gpu.download(&a.source)?;
    for (i, v) in src.iter_mut().enumerate() {
        *v += s((((h(5, i as u64) % 8192) as f64) - 4096.0) / 4096.0);
    }
    gpu.write(&mut a.source, &src)?;
    let probe = SystemProbe::probe(gpu, &hm, &p, &a, &gamma, &b_gamma)?;
    Ok(FftSys { hm, gm, a, probe })
}

/// The dense f64 solve of the DEVICE's own matrix, and that matrix on the
/// host - the truth both builds are held to.
fn dense_truth(gpu: &Gpu, a: &GpuLduMatrix, hm: &HostMesh) -> Result<(Vec<f64>, cpu::CpuLdu)> {
    let mut dev = cpu::CpuLdu::new(hm);
    dev.diag.copy_from_slice(&gpu.download(&a.diag)?);
    dev.upper.copy_from_slice(&gpu.download(&a.upper)?);
    dev.lower.copy_from_slice(&gpu.download(&a.lower)?);
    dev.source.copy_from_slice(&gpu.download(&a.source)?);
    dev.boundary_coeffs
        .copy_from_slice(&gpu.download(&a.boundary_coeffs)?);
    let b64: Vec<f64> = dev.source.iter().map(|v| f64::from(*v)).collect();
    let exact = cpu::solve_dense64(cpu::dense_from_ldu64(&dev, hm), &b64).expect("dense solve");
    Ok((exact, dev))
}

/// The answer of an FFT solve: its refined double solution under `single`,
/// the field itself in the f64 build.
#[cfg(feature = "single")]
fn fft_answer(gpu: &Gpu, b: &FftBackend, psi: &DevBuf<Scalar>) -> Result<Vec<f64>> {
    let x = b
        .refined_solution()
        .expect("single: the FFT solve leaves a double solution");
    let mut v = gpu.download(x)?;
    v.truncate(psi.len());
    Ok(v)
}

#[cfg(not(feature = "single"))]
fn fft_answer(gpu: &Gpu, _b: &FftBackend, psi: &DevBuf<Scalar>) -> Result<Vec<f64>> {
    gpu.download(psi)
}

/// 1. The cuFFT solve is the dense f64 solve of the device's own matrix to
///    `1e-10` relative with an f64-level residual - under `single` through
///    at most six refining transforms.
#[test]
fn fft_refined_matches_the_dense_f64_solve() {
    let Some(gpu) = gpu() else { return };
    if !cufft_available() {
        return;
    }
    let sys = fft_system(&gpu).unwrap();
    let mut fft = FftBackend::new();
    assert!(fft.applicable(&sys.probe), "{}", fft.why_not(&sys.probe));
    fft.setup(&gpu, &sys.hm, &sys.gm, &sys.probe).unwrap();
    let mut psi: DevBuf<Scalar> = gpu.zeros(sys.hm.n_cells).unwrap();
    let perf = fft.solve(&gpu, &mut psi, &sys.a, &sys.gm).unwrap();
    assert!(perf.converged, "the FFT solve reports converged");

    let (exact, dev) = dense_truth(&gpu, &sys.a, &sys.hm).unwrap();
    let x = fft_answer(&gpu, &fft, &psi).unwrap();
    let rel = worst_rel(&x, &exact);
    assert!(rel <= SLACK, "cuFFT vs the dense f64 solve: {rel:e}");
    let res = cpu::residual64(&x, &dev, &sys.hm);
    assert!(res <= 1e-13, "cuFFT residual in f64: {res:e}");
    assert!(
        perf.final_residual <= 1e-13,
        "reported final residual {:e}",
        perf.final_residual
    );

    #[cfg(feature = "single")]
    {
        let outer = fft.ir_outer_steps();
        assert!((1..=6).contains(&outer), "refining transforms: {outer}");
        assert_eq!(perf.n_iterations, outer);
        let stored = gpu.download(&psi).unwrap();
        for (i, (p, v)) in stored.iter().zip(&x).enumerate() {
            assert_eq!(
                p.to_bits(),
                (*v as Scalar).to_bits(),
                "cell {i}: psi is fl32(x)"
            );
        }
    }
}

/// 2. The selector keeps the cuFFT backend: it agrees with the reference to
///    `AGREEMENT_TOL` and leaves a residual below `REFERENCE_RESIDUAL_MAX`.
///    Before §118.12 `single` disqualified it at 4e-7 (two float fields an
///    ulp apart); now the refined answers are compared.
#[test]
fn the_selector_keeps_the_fft_backend() {
    let Some(gpu) = gpu() else { return };
    if !cufft_available() {
        return;
    }
    let sys = fft_system(&gpu).unwrap();
    let cands: Vec<Box<dyn PressureBackend>> = vec![
        Box::new(PbicgstabBackend::reference()),
        Box::new(FftBackend::new()),
    ];
    let (_chosen, choice) =
        choose_pressure_backend(&gpu, &sys.hm, &sys.gm, &sys.a, &sys.probe, cands).unwrap();
    let at = choice
        .considered
        .iter()
        .position(|c| c.0 == "cuFFT")
        .expect("cuFFT was considered");
    assert!(
        choice.considered[at].1 && choice.considered[at].2.is_some(),
        "cuFFT must survive the selector: {}",
        choice.notes[at]
    );
    // The yardstick is f64-grade and the agreement is held to the 1e-8
    // floor itself, not to the 100x-the-reference-residual relaxation a
    // float yardstick forces (SPEC-LIT 118.3's AGREEMENT_TOL row).
    let number = |note: &str, key: &str| -> f64 {
        let at = note
            .find(key)
            .unwrap_or_else(|| panic!("no '{key}' in '{note}'"))
            + key.len();
        note[at..]
            .split_whitespace()
            .next()
            .unwrap()
            .parse()
            .unwrap()
    };
    let ref_res = number(&choice.notes[0], "residual ");
    assert!(
        ref_res <= 1e-12,
        "reference residual {ref_res:e}: {}",
        choice.notes[0]
    );
    let agrees = number(&choice.notes[at], "agrees to ");
    assert!(agrees <= AGREEMENT_TOL as f64, "cuFFT agrees to {agrees:e}");
}

// ==========================================================================
//  The distributed Krylov
// ==========================================================================

/// An M-matrix on a 6 x 5 x 4 wall box whose coefficients are exact in f32
/// (the generator of `tests/f32_refinement.rs`), SPD or asymmetric.
fn dist_matrix(m: &HostMesh, asymmetric: bool) -> HostLduMatrix {
    let mut a = HostLduMatrix::zeros(m);
    let nf = m.n_internal_faces;
    let mut up = vec![0.0f64; nf];
    let mut lo = vec![0.0f64; nf];
    for f in 0..nf {
        let c = ((32 + h(1, f as u64) % 64) as f64) / 64.0;
        let g = (((h(2, f as u64) % 17) as f64) - 8.0) / 16.0;
        let (u, l) = if asymmetric {
            (-c * (1.0 - g), -c * (1.0 + g))
        } else {
            (-c, -c)
        };
        up[f] = u;
        lo[f] = l;
    }
    let mut diag: Vec<f64> = (0..m.n_cells)
        .map(|i| ((8 + h(3, i as u64) % 64) as f64) / 64.0)
        .collect();
    for f in 0..nf {
        diag[m.owner[f] as usize] += up[f].abs();
        diag[m.neighbour[f] as usize] += lo[f].abs();
    }
    for f in 0..nf {
        a.upper[f] = s(up[f]);
        a.lower[f] = s(lo[f]);
    }
    for i in 0..m.n_cells {
        a.diag[i] = s(diag[i]);
        a.source[i] = s((((h(4, i as u64) % 8192) as f64) - 4096.0) / 4096.0);
    }
    a
}

fn dist_mesh() -> HostMesh {
    blockgen::build_mesh(&block(axis(1.0, 6), axis(1.0, 5), axis(1.0, 4), "wall")).unwrap()
}

fn dist_ctrl(kind: LinearSolverKind) -> SolverControls {
    SolverControls {
        solver: kind,
        precon: Preconditioner::Diagonal,
        tolerance: s(SOLVE_TOL),
        rel_tol: 0.0,
        max_iter: 1000,
        min_iter: 0,
        check_interval: 1,
        fixed_iters: false,
        report_residuals: true,
    }
}

/// One part's answer: refined double under `single`, the field in f64.
#[cfg(feature = "single")]
fn part_answers(gpu: &Gpu, w: &DistWorkspace, _psi: &[DevBuf<Scalar>]) -> Vec<Vec<f64>> {
    let xs = w
        .refined_solution()
        .expect("single: a refined distributed solve");
    xs.iter().map(|x| gpu.download(x).unwrap()).collect()
}

#[cfg(not(feature = "single"))]
fn part_answers(gpu: &Gpu, _w: &DistWorkspace, psi: &[DevBuf<Scalar>]) -> Vec<Vec<f64>> {
    psi.iter().map(|x| gpu.download(x).unwrap()).collect()
}

/// The in-process distributed solve, gathered to whole-mesh order.
fn solve_in_process(
    gpu: &Gpu,
    sol: &SolverKernels,
    m: &HostMesh,
    dec: &Decomposition,
    a: &HostLduMatrix,
    ctrl: &SolverControls,
) -> (Vec<f64>, Vec<Vec<f64>>, DistPerformance, usize) {
    let mut ex = HaloExchange::new(gpu, dec).unwrap();
    let sys = DistSystem::split(gpu, &mut ex, m, dec, a).unwrap();
    let mut w = DistWorkspace::new(gpu, &sys, DistReduce::Exact).unwrap();
    let mut psi = sys.zeros(gpu).unwrap();
    let perf = match ctrl.solver {
        LinearSolverKind::PCG => dist_pcg(gpu, sol, &mut psi, &sys, &mut ex, &mut w, ctrl),
        _ => dist_pbicgstab(gpu, sol, &mut psi, &sys, &mut ex, &mut w, ctrl),
    }
    .unwrap();
    let parts = part_answers(gpu, &w, &psi);
    let mut whole = vec![0.0f64; m.n_cells];
    for (p, part) in dec.parts.iter().enumerate() {
        for i in 0..part.mesh.n_cells {
            whole[part.global_cell[i] as usize] = parts[p][i];
        }
    }
    #[cfg(feature = "single")]
    let outer = w.ir_outer_steps();
    #[cfg(not(feature = "single"))]
    let outer = 0;
    (whole, parts, perf, outer)
}

/// 3. **The plan's gate.** A refined distributed PCG (SPD) and PBiCGStab
///    (asymmetric) reach the dense f64 solve to `1e-10` relative at P = 1,
///    2 and 4 under two partitioners, and so agree with each other across
///    the part counts to the same tolerance.
#[test]
fn a_refined_distributed_solve_is_part_count_invariant() {
    let Some(gpu) = gpu() else { return };
    let sol = SolverKernels::new(&gpu).unwrap();
    let m = dist_mesh();
    for (kind, asym) in [
        (LinearSolverKind::PCG, false),
        (LinearSolverKind::PBiCGStab, true),
    ] {
        let a = dist_matrix(&m, asym);
        let mut dev = cpu::CpuLdu::new(&m);
        dev.diag = a.diag.clone();
        dev.upper = a.upper.clone();
        dev.lower = a.lower.clone();
        dev.source = a.source.clone();
        let b64: Vec<f64> = a.source.iter().map(|v| f64::from(*v)).collect();
        let exact = cpu::solve_dense64(cpu::dense_from_ldu64(&dev, &m), &b64).expect("dense");
        let ctrl = dist_ctrl(kind);

        let one = Decomposition::build(&m, 1, &PartitionMethod::Linear).unwrap();
        let (x1, _, p1, _) = solve_in_process(&gpu, &sol, &m, &one, &a, &ctrl);
        assert!(p1.converged, "{kind:?} P = 1: converged");
        for np in [1usize, 2, 4] {
            for (name, method) in [
                ("hilbert", PartitionMethod::Hilbert),
                ("linear", PartitionMethod::Linear),
            ] {
                let dec = Decomposition::build(&m, np, &method).unwrap();
                let (x, _, perf, outer) = solve_in_process(&gpu, &sol, &m, &dec, &a, &ctrl);
                let at = format!("{kind:?} P = {np} {name}");
                assert!(perf.converged, "{at}: converged");
                assert!(
                    perf.final_residual <= 1e-13,
                    "{at}: final residual {:e}",
                    perf.final_residual
                );
                let vs_dense = worst_rel(&x, &exact);
                assert!(
                    vs_dense <= SLACK,
                    "{at}: vs the dense f64 solve {vs_dense:e}"
                );
                let vs_one = worst_rel(&x, &x1);
                assert!(vs_one <= SLACK, "{at}: vs P = 1 {vs_one:e}");
                if cfg!(feature = "single") {
                    assert!((1..=6).contains(&outer), "{at}: outer steps {outer}");
                }
            }
        }
    }
}

/// 4. The SPMD refinement is bit for bit the in-process one: every rank's
///    owned answer, its sweeps, outer steps, exchanges, reductions and
///    reported residuals, at P = 2 and 4.
#[test]
fn an_spmd_refined_solve_is_the_in_process_one() {
    let Some(gpu0) = gpu() else { return };
    let sol0 = SolverKernels::new(&gpu0).unwrap();
    let m = dist_mesh();
    for (kind, asym) in [
        (LinearSolverKind::PCG, false),
        (LinearSolverKind::PBiCGStab, true),
    ] {
        let a = dist_matrix(&m, asym);
        let ctrl = dist_ctrl(kind);
        for np in [2usize, 4] {
            let dec = Decomposition::build(&m, np, &PartitionMethod::Hilbert).unwrap();
            let (_, want, wp, wouter) = solve_in_process(&gpu0, &sol0, &m, &dec, &a, &ctrl);
            let got = ThreadComm::run(np, |comm| -> Result<(Vec<f64>, DistPerformance, usize)> {
                let gpu = Gpu::new(0)?;
                let sol = SolverKernels::new(&gpu)?;
                let rank = comm.rank();
                let mut halo = SpmdHalo::new(&gpu, &dec.parts[rank], np)?;
                let sys = DistSystem::split_one(&gpu, &mut halo, comm, &m, &dec, &a)?;
                let mut w = SpmdWorkspace::new(&gpu, comm, &sys, DistReduce::Exact)?;
                let mut psi: DevBuf<Scalar> = gpu.zeros(sys.local()[0])?;
                let perf = match ctrl.solver {
                    LinearSolverKind::PCG => {
                        spmd_pcg(&gpu, &sol, comm, &mut psi, &sys, &mut halo, &mut w, &ctrl)?
                    }
                    _ => {
                        spmd_pbicgstab(&gpu, &sol, comm, &mut psi, &sys, &mut halo, &mut w, &ctrl)?
                    }
                };
                #[cfg(feature = "single")]
                let (x, outer) = (
                    gpu.download(w.refined_solution().expect("refined"))?,
                    w.ir_outer_steps(),
                );
                #[cfg(not(feature = "single"))]
                let (x, outer) = (gpu.download(&psi)?, 0usize);
                Ok((x, perf, outer))
            })
            .unwrap();
            for (r, part) in dec.parts.iter().enumerate() {
                let (x, perf, outer) = got[r].as_ref().unwrap_or_else(|e| panic!("rank {r}: {e}"));
                let at = format!("{kind:?} P = {np} rank {r}");
                for i in 0..part.mesh.n_cells {
                    assert_eq!(x[i].to_bits(), want[r][i].to_bits(), "{at}: owned cell {i}");
                }
                assert_eq!(perf.n_iterations, wp.n_iterations, "{at}: sweeps");
                assert_eq!(*outer, wouter, "{at}: outer steps");
                assert_eq!(perf.n_exchanges, wp.n_exchanges, "{at}: exchanges");
                assert_eq!(perf.n_reductions, wp.n_reductions, "{at}: reductions");
                assert_eq!(perf.converged, wp.converged, "{at}: converged");
                assert_eq!(
                    perf.final_residual.to_bits(),
                    wp.final_residual.to_bits(),
                    "{at}: final"
                );
            }
        }
    }
}

/// 5. (118.12.1)-(118.12.2) on the device, `single` only: a double split
///    into three floats and joined back is the same double, bit for bit,
///    over `2^-97 <= |x| < 2^126` (and zero comes back as zero, a negative
///    zero as `+0`) - the claim the exact halo of the distributed
///    refinement rests on. The f64 build has no split.
#[test]
fn a_double_crosses_the_float_split_bit_for_bit() {
    #[cfg(feature = "single")]
    {
        use ofgpu::solver::{join3, split3};
        let Some(gpu) = gpu() else { return };
        let k = SolverKernels::new(&gpu).unwrap();
        let mut xs: Vec<f64> = vec![
            0.0,
            -0.0,
            1.0,
            -1.0,
            f64::from(f32::MAX) * 0.5,
            2f64.powi(-97),
        ];
        for i in 0..4000u64 {
            let e = (h(7, i) % 223) as i32 - 97; // exponents -97 ..= 125
            let mant = 1.0 + (h(8, i) >> 12) as f64 / (1u64 << 52) as f64;
            let sign = if h(9, i) % 2 == 0 { 1.0 } else { -1.0 };
            xs.push(sign * mant * 2f64.powi(e));
        }
        let n = xs.len();
        let x = gpu.upload(&xs).unwrap();
        let (mut hi, mut mid, mut lo): (DevBuf<Scalar>, DevBuf<Scalar>, DevBuf<Scalar>) = (
            gpu.zeros(n).unwrap(),
            gpu.zeros(n).unwrap(),
            gpu.zeros(n).unwrap(),
        );
        split3(&gpu, &k, &mut hi, &mut mid, &mut lo, &x, n).unwrap();
        let mut back: DevBuf<f64> = gpu.zeros(n).unwrap();
        join3(&gpu, &k, &mut back, &hi, &mid, &lo, 0, n).unwrap();
        let got = gpu.download(&back).unwrap();
        for (i, (g, w)) in got.iter().zip(&xs).enumerate() {
            if *w == 0.0 {
                assert_eq!(*g, 0.0, "entry {i}: zero");
            } else {
                assert_eq!(g.to_bits(), w.to_bits(), "entry {i}: {w:e} -> {g:e}");
            }
        }
    }
}
