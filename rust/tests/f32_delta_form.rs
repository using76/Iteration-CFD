// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.
// Provenance: see PROVENANCE.md. No GPL-licensed source was consulted.

//! SPEC-LIT §118.5 — the delta form of a transient solve under `single`.
//! Five tests, the SAME assertions in BOTH builds, over a 256-cell slab
//! whose every number (dx, volume, alpha, dt, the 168·2^-15 surface rise)
//! is exact in f32, so both builds hold the very same matrix and the host
//! f64 march is the truth for both. Precision classes: **P** (physics) the
//! slab heats by erfc against Abramowitz & Stegun 7.1.26; **R** (reference)
//! the device agrees with that host march of the same discrete scheme,
//! which shares no code with the device; **D** (double bookkeeping) the
//! stored field is `fl32(level)` of the double level, cell for cell.
//!
//! An integration test, because the library's own test target does not
//! compile under `single`; an integration test links the normal (non-test)
//! library, which does. A machine without a card passes vacuously, the
//! convention of `tests/f32_refinement.rs`.
//!
//! Provenance: ORIGINAL — the slab, the template and the host march are
//! this tree's own construction; the delta form they exercise is §118.5's
//! (Beam & Warming 1978, doi:10.2514/3.60901; Carson & Higham 2018), and
//! erfc is Abramowitz & Stegun 7.1.26 (|err| <= 1.5e-7).
//! No GPL-licensed source was consulted.

use ofgpu::blockgen::{self, BlockSpec, GradedAxis};
use ofgpu::ldu_ops::LduKernels;
use ofgpu::solver::{solve, solve_delta, SolverControls, SolverKernels, SolverWorkspace};
use ofgpu::timescheme::{DeltaLevels, Ddt, DdtScheme, LtsControls};
use ofgpu::{DevBuf, Gpu, GpuLduMatrix, GpuMesh, HostMesh, Result, Scalar};

/// The slab: `dx = 2^-6`, so one cell's volume is `2^-6` (y and z are 1).
const DX: f64 = 1.0 / 64.0;
const VOL: f64 = 1.0 / 64.0;
/// Diffusivity `2^-6`, step `2^-9`, and the run length.
const ALPHA: f64 = 1.0 / 64.0;
const DT: f64 = 1.0 / 512.0;
const STEPS: usize = 2000;
/// The wall temperature: `T0` plus a rise of `168·2^-15`, i.e. about
/// `1.0013e-8·T0` per step on average - one fifth of `u32·|T0|`, which is
/// what the plain form loses and the delta form must not.
const T0: f64 = 256.0;
const DTS: f64 = 168.0 / 32768.0;
const TS: f64 = T0 + DTS;

/// Every device test needs a card; without one the test passes vacuously.
fn gpu() -> Option<Gpu> {
    Gpu::new(0).ok()
}

/// An f64 value stored as the build's `Scalar`: a no-op in the f64 build,
/// the one rounding under `single`. Every number in the slab is chosen so
/// the rounding is exact and both builds hold the same matrix and levels.
#[allow(clippy::unnecessary_cast, clippy::useless_conversion)]
fn s(x: f64) -> Scalar {
    x as Scalar
}

/// The slab block: x over [0,4] in 256 cells, y and z one cell over [0,1],
/// every patch a wall.
fn slab_spec() -> BlockSpec {
    let axis = |lo: f64, hi: f64, n: usize| GradedAxis {
        lo: s(lo),
        hi: s(hi),
        n,
        expansion: 1.0,
        two_sided: false,
    };
    BlockSpec {
        x: axis(0.0, 4.0, 256),
        y: axis(0.0, 1.0, 1),
        z: axis(0.0, 1.0, 1),
        windows: Vec::new(),
        patch_name: BlockSpec::default().patch_name,
        patch_type: ["wall"; 6].map(String::from),
        cyclic: Vec::new(),
    }
}

/// The spatial template `M = -laplacian`, written straight to the device -
/// no assembly, so nothing stands between the generator and the ddt. Every
/// internal face carries -1; `diag_c` is the cell's internal-face count plus
/// 2.0 on the wall cell (the Dirichlet half-cell, folded in by hand); the
/// source is `2·TS` on the wall cell and nothing elsewhere. In units of
/// `gamma·A/dx = 1`. Locate cells by centre, never by index.
struct Slab {
    m: HostMesh,
    gm: GpuMesh,
    a: GpuLduMatrix,
    t_diag: DevBuf<Scalar>,
    t_upper: DevBuf<Scalar>,
    t_lower: DevBuf<Scalar>,
    t_source: DevBuf<Scalar>,
    /// The f64 masters of the stored template, for the host march.
    diag64: Vec<f64>,
    source64: Vec<f64>,
}

/// Build the slab and its template, restored into `a` at the top of every
/// step so a step sees exactly the template plus its own ddt.
#[allow(clippy::useless_conversion)]
fn slab(gpu: &Gpu) -> Result<Slab> {
    let m = blockgen::build_mesh(&slab_spec())?;
    let gm = GpuMesh::upload(gpu, &m)?;
    let n = m.n_cells;
    let nf = m.n_internal_faces;

    let mut faces_of = vec![0.0f64; n];
    for f in 0..nf {
        faces_of[m.owner[f] as usize] += 1.0;
        faces_of[m.neighbour[f] as usize] += 1.0;
    }
    let is_wall: Vec<bool> = m.c.iter().map(|p| f64::from(p.x) < DX).collect();
    let diag64: Vec<f64> = (0..n)
        .map(|c| faces_of[c] + if is_wall[c] { 2.0 } else { 0.0 })
        .collect();
    let source64: Vec<f64> = (0..n).map(|c| if is_wall[c] { 2.0 * TS } else { 0.0 }).collect();

    let upper = vec![s(-1.0); nf];
    let lower = vec![s(-1.0); nf];
    let diag: Vec<Scalar> = diag64.iter().map(|v| s(*v)).collect();
    let source: Vec<Scalar> = source64.iter().map(|v| s(*v)).collect();

    let mut a = GpuLduMatrix::new(gpu, &gm)?;
    a.zero(gpu)?;
    gpu.write(&mut a.diag, &diag)?;
    gpu.write(&mut a.upper, &upper)?;
    gpu.write(&mut a.lower, &lower)?;
    gpu.write(&mut a.source, &source)?;

    Ok(Slab {
        t_diag: gpu.upload(&diag)?,
        t_upper: gpu.upload(&upper)?,
        t_lower: gpu.upload(&lower)?,
        t_source: gpu.upload(&source)?,
        m,
        gm,
        a,
        diag64,
        source64,
    })
}

impl Slab {
    /// Put the template back: a step must see exactly the template plus its
    /// own ddt, so the matrix of the step before it must not leak in.
    fn restore(&mut self, gpu: &Gpu) -> Result<()> {
        self.a.zero(gpu)?;
        gpu.stream().memcpy_dtod(&self.t_diag, &mut self.a.diag)?;
        gpu.stream().memcpy_dtod(&self.t_upper, &mut self.a.upper)?;
        gpu.stream().memcpy_dtod(&self.t_lower, &mut self.a.lower)?;
        gpu.stream().memcpy_dtod(&self.t_source, &mut self.a.source)?;
        Ok(())
    }
}

/// `theta_c = (level_c - T0)/DTS` - the rise in units of the surface step.
fn theta_of(level: &[f64]) -> Vec<f64> {
    level.iter().map(|l| (l - T0) / DTS).collect()
}

/// `erfc(x)` by Abramowitz & Stegun 7.1.26, `|err| <= 1.5e-7` - three orders
/// below the 1e-3 this file asserts against.
fn erfc(x: f64) -> f64 {
    let p = 0.327_591_1;
    let (a1, a2, a3, a4, a5) = (
        0.254_829_592,
        -0.284_496_736,
        1.421_413_741,
        -1.453_152_027,
        1.061_405_429,
    );
    let t = 1.0 / (1.0 + p * x.abs());
    let v = t * (a1 + t * (a2 + t * (a3 + t * (a4 + t * a5)))) * (-(x * x)).exp();
    if x >= 0.0 {
        v
    } else {
        2.0 - v
    }
}

/// The exact theta of the semi-infinite slab, `erfc(x_c / (2·sqrt(alpha·t)))`
/// at `t = STEPS·DT`, at every cell centre.
#[allow(clippy::useless_conversion)]
fn exact_theta(hm: &HostMesh) -> Vec<f64> {
    let t = STEPS as f64 * DT;
    let two_root = 2.0 * (ALPHA * t).sqrt();
    hm.c
        .iter()
        .map(|p| erfc(f64::from(p.x) / two_root))
        .collect()
}

/// `max_i |a_i - b_i|`.
fn max_abs(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(u, v)| (u - v).abs())
        .fold(0.0f64, f64::max)
}

// ==========================================================================
//  The f64 reference march
//
//  The SAME discrete scheme the device runs, marched in f64 on the host by
//  an x-ordered Thomas solve, sharing no code with the device: Euler;
//  backward = Euler on step 0 then `(1.5 x - 2 x0 + 0.5 x00)/DT`; CN at
//  theta 1/2 as `(V/DT + M/2) x = V/DT x0 + b - (M/2) x0`. This is the R
//  class's truth, and its agreement with the exact erfc is the P class's
//  discretisation floor.
// ==========================================================================

/// `(M x)_c` for the slab template: `diag_c·x_c` less each internal-face
/// neighbour, walked off the mesh's own owner/neighbour lists.
fn m_rows(diag: &[f64], x: &[f64], m: &HostMesh) -> Vec<f64> {
    let mut out: Vec<f64> = diag.iter().zip(x).map(|(dv, xv)| dv * xv).collect();
    for f in 0..m.n_internal_faces {
        let (o, nb) = (m.owner[f] as usize, m.neighbour[f] as usize);
        out[o] -= x[nb];
        out[nb] -= x[o];
    }
    out
}

/// Thomas' algorithm on an x-ordered tridiagonal row: `sub[i]·x[i-1] +
/// d[i]·x[i] + sup[i]·x[i+1] = r[i]`.
fn thomas(sub: &[f64], d: &[f64], sup: &[f64], r: &[f64]) -> Vec<f64> {
    let n = d.len();
    let mut cp = vec![0.0; n];
    let mut dp = vec![0.0; n];
    cp[0] = sup[0] / d[0];
    dp[0] = r[0] / d[0];
    for i in 1..n {
        let denom = d[i] - sub[i] * cp[i - 1];
        cp[i] = sup[i] / denom;
        dp[i] = (r[i] - sub[i] * dp[i - 1]) / denom;
    }
    let mut x = vec![0.0; n];
    x[n - 1] = dp[n - 1];
    for i in (0..n - 1).rev() {
        x[i] = dp[i] - cp[i] * x[i + 1];
    }
    x
}

/// The host march of [`DdtScheme`] `scheme` over [`STEPS`] steps, from `T0`.
fn host_march(slab: &Slab, scheme: DdtScheme) -> Vec<f64> {
    let n = slab.m.n_cells;
    let v_dt = VOL / DT;
    let diag = &slab.diag64;
    let src = &slab.source64;
    let mut x = vec![T0; n];
    let mut x0 = vec![T0; n];
    let mut x00 = vec![T0; n];

    for step in 0..STEPS {
        x00.clone_from(&x0);
        x0.clone_from(&x);
        let (sub, sup, d, r) = match (scheme, step == 0) {
            // BDF2's first step IS Euler: psi^{n-2} does not exist yet.
            (DdtScheme::Euler, _) | (DdtScheme::Backward, true) => (
                vec![-1.0; n],
                vec![-1.0; n],
                diag.iter().map(|dv| v_dt + dv).collect::<Vec<_>>(),
                (0..n).map(|c| v_dt * x0[c] + src[c]).collect::<Vec<_>>(),
            ),
            (DdtScheme::Backward, false) => (
                vec![-1.0; n],
                vec![-1.0; n],
                diag.iter().map(|dv| 1.5 * v_dt + dv).collect::<Vec<_>>(),
                (0..n)
                    .map(|c| src[c] + 2.0 * v_dt * x0[c] - 0.5 * v_dt * x00[c])
                    .collect::<Vec<_>>(),
            ),
            (DdtScheme::CrankNicolson(_), _) => {
                let mx0 = m_rows(diag, &x0, &slab.m);
                (
                    vec![-0.5; n],
                    vec![-0.5; n],
                    diag.iter().map(|dv| v_dt + 0.5 * dv).collect::<Vec<_>>(),
                    (0..n)
                        .map(|c| v_dt * x0[c] + src[c] - 0.5 * mx0[c])
                        .collect::<Vec<_>>(),
                )
            }
            (other, _) => unreachable!("the slab marches a transient scheme, not {other:?}"),
        };
        x = thomas(&sub, &d, &sup, &r);
    }
    x
}

// ==========================================================================
//  The device march
// ==========================================================================

/// The controls every slab step asks for.
fn slab_ctrl() -> SolverControls {
    SolverControls {
        tolerance: s(1e-10),
        rel_tol: 0.0,
        max_iter: 1000,
        ..Default::default()
    }
}

/// March the slab on the device through the DELTA form
/// ([`Ddt::add_delta`] + [`solve_delta`]), restoring the template at the top
/// of every step, and hand back the stored field and the double level.
fn march_delta(gpu: &Gpu, slab: &mut Slab, scheme: DdtScheme) -> Result<(Vec<Scalar>, Vec<f64>)> {
    let n = slab.m.n_cells;
    let k = SolverKernels::new(gpu)?;
    let lduk = LduKernels::new(gpu)?;
    let mut ws = SolverWorkspace::for_mesh(gpu, &slab.gm)?;
    let mut psi: DevBuf<Scalar> = gpu.upload(&vec![s(T0); n])?;
    let mut lv = DeltaLevels::new(gpu, &psi, n)?;
    let mut scratch: DevBuf<Scalar> = gpu.zeros(n)?;
    let mut ddt = Ddt::new_with_relax(gpu, &slab.gm, scheme, s(DT), LtsControls::default(), false)?;
    let ctrl = slab_ctrl();

    for _ in 0..STEPS {
        slab.restore(gpu)?;
        lv.store_old_time(gpu)?;
        ddt.add_delta(gpu, &lduk, &k, &mut slab.a, &slab.gm, &lv, &mut scratch, s(1.0))?;
        let perf = solve_delta(gpu, &k, &mut psi, &slab.a, &slab.gm, &mut ws, &ctrl, &mut lv)?;
        assert!(
            perf.converged,
            "the increment solve stagnated at {:e}",
            perf.final_residual
        );
        ddt.advance(s(DT));
    }
    gpu.sync()?;
    Ok((gpu.download(&psi)?, gpu.download(lv.current())?))
}

/// The three-way check every scheme's delta march is held to: (P) against
/// the exact erfc, (R) against the host f64 march of the same scheme, (D)
/// the stored field is `fl32(level)` cell for cell.
fn slab_case(scheme: DdtScheme) {
    let Some(gpu) = gpu() else { return };
    let mut slab = slab(&gpu).unwrap();
    let (psi, level) = march_delta(&gpu, &mut slab, scheme).unwrap();

    let th = theta_of(&level);
    let ex = exact_theta(&slab.m);
    let host = theta_of(&host_march(&slab, scheme));
    let p = max_abs(&th, &ex);
    let r = max_abs(&th, &host);
    eprintln!(
        "{scheme:?}: (P) max|theta - erfc| = {p:e}, (R) max|theta - theta_host| = {r:e}"
    );
    assert!(p <= 1e-3, "(P) {scheme:?}: max|theta - erfc| = {p:e}");
    assert!(r <= 1e-5, "(R) {scheme:?}: max|theta - theta_host| = {r:e}");

    for (c, (p, l)) in psi.iter().zip(&level).enumerate() {
        assert_eq!(*p, s(*l), "(D) {scheme:?} cell {c}: psi is not fl32(level)");
    }
}

/// 1. The delta form heats the slab by about `1.0013e-8·T0` per step under
///    Euler implicit: theta against the exact erfc (P), against the host f64
///    march of the same scheme (R), and the stored field against the double
///    level (D).
#[test]
fn f32_slab_heats_by_1e_8_per_step_euler() {
    slab_case(DdtScheme::Euler);
}

/// 1b. The same on BDF2 - Euler on the first step by construction, then the
///     variable-coefficient three-level form with `a_00 = r²/((1+r)·dt)`,
///     whose `x00` correction the delta right-hand side must carry.
#[test]
fn f32_slab_heats_by_1e_8_per_step_backward() {
    slab_case(DdtScheme::Backward);
}

/// 1c. The same on Crank-Nicolson at theta 1/2: the one scheme whose
///     increment system carries the FULL matrix's `b - M x0` and the
///     theta-scaled operator side, reachable only unrelaxed.
#[test]
fn f32_slab_heats_by_1e_8_per_step_crank_nicolson() {
    slab_case(DdtScheme::CrankNicolson(0.5));
}

/// 2. The PLAIN form - Euler through [`Ddt::add`] + [`solve`] on an f32
///    field rotated by memcpy, the pattern every pre-118.5 march ran - on
///    the same slab. The per-step increment is below `u32·|psi|`, and the
///    stored field's ulp grid loses it: the f64 build tracks the exact erfc,
///    the `single` build does not. This is the documented defect §118.5
///    exists to fix, asserted in BOTH builds.
#[allow(clippy::useless_conversion)]
#[test]
fn the_plain_form_loses_the_increment_only_under_single() {
    let Some(gpu) = gpu() else { return };
    let mut slab = slab(&gpu).unwrap();
    let n = slab.m.n_cells;
    let k = SolverKernels::new(&gpu).unwrap();
    let mut ws = SolverWorkspace::for_mesh(&gpu, &slab.gm).unwrap();
    let mut psi: DevBuf<Scalar> = gpu.upload(&vec![s(T0); n]).unwrap();
    let mut f0: DevBuf<Scalar> = gpu.upload(&vec![s(T0); n]).unwrap();
    let mut f00: DevBuf<Scalar> = gpu.upload(&vec![s(T0); n]).unwrap();
    let mut ddt = Ddt::new_with_relax(
        &gpu,
        &slab.gm,
        DdtScheme::Euler,
        s(DT),
        LtsControls::default(),
        false,
    )
    .unwrap();
    let ctrl = slab_ctrl();

    for _ in 0..STEPS {
        gpu.stream().memcpy_dtod(&f0, &mut f00).unwrap();
        gpu.stream().memcpy_dtod(&psi, &mut f0).unwrap();
        slab.restore(&gpu).unwrap();
        ddt.add(&gpu, &mut slab.a, &slab.gm, &f0, &f00, s(1.0)).unwrap();
        solve(&gpu, &k, &mut psi, &slab.a, &slab.gm, &mut ws, &ctrl).unwrap();
        ddt.advance(s(DT));
    }
    gpu.sync().unwrap();

    let got: Vec<Scalar> = gpu.download(&psi).unwrap();
    let th: Vec<f64> = got.iter().map(|p| (f64::from(*p) - T0) / DTS).collect();
    let e = max_abs(&th, &exact_theta(&slab.m));
    eprintln!("plain form: max|theta - erfc| = {e:e}");
    #[cfg(not(feature = "single"))]
    assert!(e <= 1e-3, "f64 plain form: max|theta - erfc| = {e:e}");
    #[cfg(feature = "single")]
    assert!(
        e >= 0.1,
        "single plain form: the increment was NOT lost, max|theta - erfc| = {e:e}"
    );
}

/// 3. The delta form is for transients: `steadyState` and `localEuler` are
///    refused by name, and the refusal cites SPEC-LIT 118.5.
#[test]
fn add_delta_refuses_steady_and_local_euler() {
    let Some(gpu) = gpu() else { return };
    let mut slab = slab(&gpu).unwrap();
    let n = slab.m.n_cells;
    let k = SolverKernels::new(&gpu).unwrap();
    let lduk = LduKernels::new(&gpu).unwrap();
    let psi: DevBuf<Scalar> = gpu.upload(&vec![s(T0); n]).unwrap();
    let lv = DeltaLevels::new(&gpu, &psi, n).unwrap();
    let mut scratch: DevBuf<Scalar> = gpu.zeros(n).unwrap();

    for scheme in [DdtScheme::SteadyState, DdtScheme::LocalEuler] {
        let ddt =
            Ddt::new_with_relax(&gpu, &slab.gm, scheme, s(DT), LtsControls::default(), false)
                .unwrap();
        let e = ddt
            .add_delta(&gpu, &lduk, &k, &mut slab.a, &slab.gm, &lv, &mut scratch, s(1.0))
            .expect_err("a steady scheme must be refused in the delta form");
        assert!(
            e.to_string().contains("118.5"),
            "{scheme:?}: the refusal must cite SPEC-LIT 118.5: {e}"
        );
    }
}
