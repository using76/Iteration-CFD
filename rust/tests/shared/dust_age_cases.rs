// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.
// Provenance: see PROVENANCE.md. No GPL-licensed source was consulted.

// The DST-20 gates of SPEC-LIT §160.5, written once and included both by
// src/dust/tests_age.rs (as dust::age::tests) and by tests/dust_age.rs, so
// both builds run them. The D rows (bitwise) and the refusal must hold; a
// row that misses prints `MISSED ...` with the measured value and the test
// still passes (the house rule for gate rows). A GPU test returns early, by
// name, on a box without a device.
// The specification was informed by a review of FSCDustFoam (Huang &
// Lipatnikov, GPL) and no code was copied.
// No GPL-licensed source was consulted.

use ofgpu::device::Gpu;
use ofgpu::dust::age::{predict_cell, sources_cell, C_A};
use ofgpu::dust::device::{
    age_predict, age_sources, fsc as launch_fsc, fsc_aged, AgeSourceInputs, FscFields,
    FscInputsDev, FscKernels, ProgressKernels,
};
use ofgpu::dust::fsc::{evaluate, CombustionModel, FscConstants, FscInputs, FscState};
use ofgpu::dust::progress::ProgressEquation;
use ofgpu::dust::psi::{reaction_time, PsiOptions, PsiProblem, PublishedCurve};
use ofgpu::dust::tube1d::{
    run, run_aged, tube_mesh, AgeBudgetRow, AgeOptions, AgeSample, FrozenTurbulence, KappaLaw,
    LateKernel, TubeSpec,
};
use ofgpu::error::Error;
use ofgpu::fv::{DivScheme, Limiter};
use ofgpu::io::case::{Preconditioner, TurbulenceControls};
use ofgpu::timescheme::DdtScheme;
use ofgpu::{GpuMesh, Scalar};

/// Leeds (H20 Tables 1, 4; SPEC-LIT §151.5): u' = 1.62 m/s.
const K0: f64 = 3.94;
const EPS0: f64 = 390.53;
const PR_T: f64 = 0.4;
const S_L_LEEDS: f64 = 0.12;
const RHO_U_LEEDS: f64 = 1.32;
const T_U_LEEDS: f64 = 328.0;

fn kappa_u_leeds() -> f64 {
    1.8e-5 / (1.32 * 0.7)
}

/// A value as the device holds it: rounded to `Scalar`, widened back.
#[allow(clippy::unnecessary_cast)]
fn r(v: f64) -> f64 {
    v as Scalar as f64
}

/// The frozen Leeds turbulence through the k-omega mapping of §154.4:
/// omega = eps/(beta* k), nu_t = k/omega, as the device holds them.
fn frozen() -> FrozenTurbulence {
    let omega = EPS0 / (0.09 * K0);
    FrozenTurbulence { k: r(K0), omega: r(omega), nu_t: r(K0 / omega) }
}

/// The FSC constants of a tube: the published closure's T_b = sigma T_u.
fn constants(sigma: f64, t_u: f64, s_l: f64, kappa_u: f64, t_r: f64) -> FscConstants {
    FscConstants::new(&FscInputs {
        model: CombustionModel::Fsc,
        a: 0.4,
        s_l,
        kappa_u,
        pr_t: PR_T,
        t_r,
        t_b: sigma * t_u,
        kappa_b: sigma * kappa_u,
        t_flacc: None,
    })
    .expect("the tube's FSC constants are admitted")
}

/// The Leeds published-closure t_r (153.2).
fn t_r_leeds() -> f64 {
    let pc = PublishedCurve::new(5.06).expect("curve");
    let s = PsiProblem { curve: &pc, theta_u: 20000.0 / T_U_LEEDS, n: 1.0 }
        .eigen(&PsiOptions::default())
        .expect("psi");
    reaction_time(s.psi, kappa_u_leeds(), S_L_LEEDS).expect("t_r")
}

fn gpu(test: &str) -> Option<Gpu> {
    match Gpu::new(0) {
        Ok(g) => Some(g),
        Err(_) => {
            eprintln!("{test}: skipped, no CUDA device");
            None
        }
    }
}

/// Bitwise equality of a device word and a host word as the build holds
/// them (a D row's bar).
fn bits_eq(a: Scalar, b: Scalar) -> bool {
    if std::mem::size_of::<Scalar>() == 8 {
        (a as f64).to_bits() == (b as f64).to_bits()
    } else {
        (a as f32).to_bits() == (b as f32).to_bits()
    }
}

/// The outer tolerance on max |c~ change| per step (as dust_progress's).
const OUTER_TOL: f64 = if cfg!(feature = "single") { 1e-6 } else { 1e-8 };

/// The R-class bar on a relative clock error: tol64 = 1e-10, and by
/// D-F32-1 the float build's 1e-5.
fn tol_r() -> f64 {
    let tol64: f64 = 1e-10;
    let tol32 = tol64.max((tol64 * 2f64.powi(29)).min(1e-5));
    if cfg!(feature = "single") {
        tol32
    } else {
        tol64
    }
}

/// The c~ equation's controls of the age rows: Euler, limited (van Leer)
/// bounded convection, PBiCGStab with DILU, each solve to its ABSOLUTE
/// tolerance (1e-13, `single` 1e-7) at up to 1000 (`single` 50) iterations.
fn age_controls(dt: f64) -> TurbulenceControls {
    let mut ctrl = TurbulenceControls {
        steady: false,
        delta_t: dt as Scalar,
        k_relax: 1.0,
        ..Default::default()
    };
    ctrl.ddt = DdtScheme::Euler;
    ctrl.div_scheme = DivScheme::Limited(Limiter::VanLeer);
    ctrl.bounded_convection = true;
    ctrl.k_solver.tolerance = if cfg!(feature = "single") { 1e-7 } else { 1e-13 };
    ctrl.k_solver.rel_tol = 0.0;
    ctrl.k_solver.max_iter = if cfg!(feature = "single") { 50 } else { 1000 };
    ctrl.k_solver.precon = Preconditioner::Dilu;
    ctrl.k_solver.report_residuals = false;
    ctrl
}

/// The shared tube of the age rows (§160.5): planar_spec's construction -
/// the Leeds constants, n = 250 on [-0.08, 0.12] m, sigma 5.06, Co_f 0.5,
/// kappa = 0, frozen turbulence, Q off - with the row's controls. Returns
/// the spec and its step.
fn age_spec(t_fd_origin: f64) -> (TubeSpec, f64) {
    let c = constants(5.06, T_U_LEEDS, S_L_LEEDS, kappa_u_leeds(), t_r_leeds());
    let f = frozen();
    let s = FscState { t: 1000.0, k: f.k, omega: f.omega, nu_t: f.nu_t };
    let isp = evaluate(&c, &s, f64::INFINITY, 1.0).u_t_isp;
    let n = 250usize;
    let dx = 0.2 / n as f64;
    let dt = 0.5 * dx / ((5.06 - 1.0) * isp + 5.06 * isp);
    let spec = TubeSpec {
        n_cells: n,
        x_lo: -0.08,
        x_hi: 0.12,
        sigma: 5.06,
        rho_u: RHO_U_LEEDS,
        t_u: T_U_LEEDS,
        kappa: KappaLaw::Zero,
        fsc: c,
        turbulence: Some(frozen()),
        with_q: false,
        ignition: None,
        t_fd_origin,
        dt,
        max_outer: 20,
        outer_tol: OUTER_TOL,
        ctrl: age_controls(dt),
    };
    (spec, dt)
}

/// The cell nearest x = v on an ascending grid.
fn nearest(x: &[f64], v: f64) -> usize {
    let (mut best, mut d) = (0usize, f64::INFINITY);
    for (i, &xi) in x.iter().enumerate() {
        let e = (xi - v).abs();
        if e < d {
            d = e;
            best = i;
        }
    }
    best
}

/// 1: OFF is bitwise the global-clock path (D, must hold). (i) `dustFscAged`
/// with every cell's age equal to a clock g writes the five outputs and the
/// zero count of `dustFsc(t_fd = g, t = g)` bit for bit; (ii) a laminar
/// tube - no turbulence, so the closures do not read the clock - gives the
/// same c~ and cbar bit for bit with the age ON as OFF.
#[test]
fn off_is_global_clock_bitwise() -> Result<(), Error> {
    let Some(g) = gpu("off_is_global_clock_bitwise") else { return Ok(()) };
    let fk = FscKernels::new(&g)?;
    let n = 64usize;

    // (i) 64 cells of varied T (some below the zero rule), k, omega, nu_t
    // (some k = 0), t_flacc = 5e-3.
    let c = FscConstants::new(&FscInputs {
        model: CombustionModel::Fsc,
        a: 0.4,
        s_l: S_L_LEEDS,
        kappa_u: kappa_u_leeds(),
        pr_t: PR_T,
        t_r: t_r_leeds(),
        t_b: 5.06 * T_U_LEEDS,
        kappa_b: 5.06 * kappa_u_leeds(),
        t_flacc: Some(5e-3),
    })
    .expect("the closures' constants are admitted");
    let ts = [200.0, 240.0, 300.0, 328.0, 700.0, 1200.0, 5.06 * T_U_LEEDS, 3000.0];
    let ks = [0.0, 0.0, 0.5, K0, K0, 1.0, 2.0, 0.1];
    let om = [0.0, 10.0, 100.0, EPS0 / (0.09 * K0), 500.0, 50.0, 2000.0, 20.0];
    let nt = [0.0, 1e-4, 1e-3, frozen().nu_t, 1e-2, 1e-3, 5e-4, 2e-3];
    let cells = |t: &[f64]| -> Vec<Scalar> { (0..n).map(|i| t[i % t.len()] as Scalar).collect() };
    let t_b = g.upload(&cells(&ts))?;
    let k_b = g.upload(&cells(&ks))?;
    let w_b = g.upload(&cells(&om))?;
    let nu_b = g.upload(&cells(&nt))?;
    let mut age_b: ofgpu::device::DevBuf<Scalar> = g.zeros(n)?;
    let (mut nz_a, mut nz_b) = (g.zeros(1)?, g.zeros(1)?);
    let outs = |g: &Gpu, n: usize| -> Result<[ofgpu::device::DevBuf<Scalar>; 5], Error> {
        Ok([g.zeros(n)?, g.zeros(n)?, g.zeros(n)?, g.zeros(n)?, g.zeros(n)?])
    };
    for gv in [0.0f64, 1e-4, 3.4e-3, 0.02, 1e30] {
        let gs = r(gv);
        g.write(&mut age_b, &vec![gs as Scalar; n])?;
        let [mut d_a, mut u_a, mut i_a, mut l_a, mut kr_a] = outs(&g, n)?;
        let [mut d_b, mut u_b, mut i_b, mut l_b, mut kr_b] = outs(&g, n)?;
        g.write(&mut nz_a, &[0u64])?;
        fsc_aged(
            &g,
            &fk,
            &c,
            FscFields {
                d_t: &mut d_a,
                u_t: &mut u_a,
                u_t_isp: &mut i_a,
                tau_l: &mut l_a,
                k_rate: &mut kr_a,
            },
            &mut nz_a,
            FscInputsDev { t: &t_b, k: &k_b, omega: &w_b, nu_t: &nu_b },
            &age_b,
            n,
        )?;
        g.write(&mut nz_b, &[0u64])?;
        launch_fsc(
            &g,
            &fk,
            &c,
            FscFields {
                d_t: &mut d_b,
                u_t: &mut u_b,
                u_t_isp: &mut i_b,
                tau_l: &mut l_b,
                k_rate: &mut kr_b,
            },
            &mut nz_b,
            FscInputsDev { t: &t_b, k: &k_b, omega: &w_b, nu_t: &nu_b },
            gs,
            gs,
            n,
        )?;
        let (za, zb) = (g.download(&nz_a)?[0], g.download(&nz_b)?[0]);
        assert_eq!(za, zb, "g = {gs:e}: the zero count differs");
        let names = ["d_t", "u_t", "u_t_isp", "tau_l", "k_rate"];
        for (name, (a, b)) in names.iter().zip([
            (&d_a, &d_b),
            (&u_a, &u_b),
            (&i_a, &i_b),
            (&l_a, &l_b),
            (&kr_a, &kr_b),
        ]) {
            let (va, vb) = (g.download(a)?, g.download(b)?);
            for i in 0..n {
                assert!(
                    bits_eq(va[i], vb[i]),
                    "g = {gs:e}: {name}[{i}] differs: {:e} vs {:e}",
                    va[i] as f64,
                    vb[i] as f64
                );
            }
        }
    }
    println!("off_is_global_clock_bitwise: the aged kernel is dustFsc bit for bit at five clocks");

    // (ii) a LAMINAR tube: no turbulence, so the closures never read the
    // clock, and the c~ path is the same operations in the same order with
    // the age ON as OFF.
    let c = constants(5.06, T_U_LEEDS, S_L_LEEDS, kappa_u_leeds(), t_r_leeds());
    let dt_l = 2e-5;
    let spec = TubeSpec {
        n_cells: 250,
        x_lo: -0.08,
        x_hi: 0.12,
        sigma: 5.06,
        rho_u: RHO_U_LEEDS,
        t_u: T_U_LEEDS,
        kappa: KappaLaw::ConstantRhoKappa { kappa_u: kappa_u_leeds() },
        fsc: c,
        turbulence: None,
        with_q: true,
        ignition: None,
        t_fd_origin: 0.0,
        dt: dt_l,
        max_outer: 20,
        outer_tol: OUTER_TOL,
        ctrl: age_controls(dt_l),
    };
    let c0 = |x: f64| if x < 0.0 { 1.0 } else { 0.0 };
    let plain = run(&g, &spec, &c0, 50.0 * dt_l, &mut |_, _, _| {})?;
    let opts = AgeOptions { c_a: C_A, late: Vec::new() };
    let aged = run_aged(&g, &spec, &opts, &c0, 50.0 * dt_l, &mut |_: &AgeSample<'_>| {})?;
    assert_eq!(plain.steps, aged.run.steps, "the step count");
    assert_eq!(plain.outer_total, aged.run.outer_total, "the outer-iteration count");
    for i in 0..plain.c.len() {
        assert!(
            bits_eq(plain.c[i] as Scalar, aged.run.c[i] as Scalar),
            "c~[{i}] differs: {:e} vs {:e}",
            plain.c[i],
            aged.run.c[i]
        );
        assert!(
            bits_eq(plain.cbar[i] as Scalar, aged.run.cbar[i] as Scalar),
            "cbar[{i}] differs: {:e} vs {:e}",
            plain.cbar[i],
            aged.run.cbar[i]
        );
    }
    println!(
        "off_is_global_clock_bitwise: the laminar tube is bitwise the same with the age on, \
         {} steps, {} outer iterations",
        plain.steps, plain.outer_total
    );
    Ok(())
}

/// 2: the age kernels against their host twins (D): `age_predict` and
/// `age_sources` on 64 varied cells - c~ in {-1e-9, 0, 5e-7, 1e-6, 0.3, 1}
/// - equal `predict_cell`/`sources_cell` bit for bit.
#[test]
fn age_kernels_match_host_twins() -> Result<(), Error> {
    let Some(g) = gpu("age_kernels_match_host_twins") else { return Ok(()) };
    let pk = ProgressKernels::new(&g)?;
    let n = 64usize;
    let cs = [-1e-9f64, 0.0, 5e-7, 1e-6, 0.3, 1.0];
    let (mut c_v, mut a0_v, mut su_v, mut sp_v) =
        (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    let (mut r0_v, mut c0_v, mut r00_v, mut c00_v) =
        (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    for i in 0..n {
        let f = i as f64;
        c_v.push(cs[i % 6] as Scalar);
        a0_v.push((1e-3 * (f + 1.0)) as Scalar);
        su_v.push(((f - 20.0) * 100.0) as Scalar);
        sp_v.push(((f - 40.0) * 50.0) as Scalar);
        r0_v.push((1.32 + 0.01 * f) as Scalar);
        c0_v.push((0.01 * f) as Scalar);
        r00_v.push((1.32 - 0.005 * f) as Scalar);
        c00_v.push((0.02 * f - 0.1) as Scalar);
    }
    let (c_b, a0_b) = (g.upload(&c_v)?, g.upload(&a0_v)?);
    let (su_b, sp_b) = (g.upload(&su_v)?, g.upload(&sp_v)?);
    let (r0_b, c0_b) = (g.upload(&r0_v)?, g.upload(&c0_v)?);
    let (r00_b, c00_b) = (g.upload(&r00_v)?, g.upload(&c00_v)?);
    let (delta, c_a) = (3.4e-3f64, C_A);

    // (160.4) against predict_cell.
    let (mut alpha, mut age) = (g.zeros(n)?, g.zeros(n)?);
    age_predict(&g, &pk, &mut alpha, &mut age, &a0_b, &c_b, delta, c_a, n)?;
    let (al_h, ag_h) = (g.download(&alpha)?, g.download(&age)?);
    for i in 0..n {
        let (s, a) = predict_cell(a0_v[i], c_v[i], delta, c_a);
        assert!(bits_eq(al_h[i], s), "alpha*[{i}] differs: {:e} vs {:e}", al_h[i] as f64, s as f64);
        assert!(bits_eq(ag_h[i], a), "a*[{i}] differs: {:e} vs {:e}", ag_h[i] as f64, a as f64);
    }

    // (160.5)-(160.6) against sources_cell.
    let (w0, w00) = (2.5f64, -1.75f64);
    let (mut sua, mut spa) = (g.zeros(n)?, g.zeros(n)?);
    age_sources(
        &g,
        &pk,
        &mut sua,
        &mut spa,
        &AgeSourceInputs {
            su: &su_b,
            sp: &sp_b,
            c: &c_b,
            rho0: &r0_b,
            c0: &c0_b,
            rho00: &r00_b,
            c00: &c00_b,
            w0,
            w00,
            c_a,
        },
        n,
    )?;
    let (sua_h, spa_h) = (g.download(&sua)?, g.download(&spa)?);
    let mut worst = 0usize;
    for i in 0..n {
        let (s, p) = sources_cell(
            su_v[i], sp_v[i], c_v[i], r0_v[i], c0_v[i], r00_v[i], c00_v[i], w0, w00, c_a,
        );
        if !bits_eq(sua_h[i], s) || !bits_eq(spa_h[i], p) {
            worst += 1;
        }
        assert!(bits_eq(sua_h[i], s), "S_alpha[{i}] differs: {:e} vs {:e}", sua_h[i] as f64, s as f64);
        assert!(bits_eq(spa_h[i], p), "S_P,a[{i}] differs: {:e} vs {:e}", spa_h[i] as f64, p as f64);
    }
    println!("age_kernels_match_host_twins: {n} cells bitwise, {worst} differ");
    Ok(())
}

/// 3: ON with one ignition - the planar kernel at t = 0, the clock's origin
/// t_0 = 1 ms - holds a = t - t_0 wherever c~ > c_a (R, §160.2): at every
/// sample with t >= 2 ms the relative error against the uniform clock, with
/// the band maxima over c~. A miss is a finding (the leading-edge defect of
/// §160.2), printed as MISSED and never a widened band.
#[test]
fn one_ignition_is_uniform_clock() -> Result<(), Error> {
    let Some(g) = gpu("one_ignition_is_uniform_clock") else { return Ok(()) };
    let t0 = 1e-3;
    let (spec, _) = age_spec(t0);
    let opts = AgeOptions { c_a: C_A, late: Vec::new() };
    let tol = tol_r();
    let (mut e_all, mut e_band) = (0.0f64, [0.0f64; 3]);
    let samples = std::cell::Cell::new(0usize);
    run_aged(&g, &spec, &opts, &|x| if x < 0.0 { 1.0 } else { 0.0 }, 0.01, &mut |s: &AgeSample<'_>| {
        samples.set(samples.get() + 1);
        if s.t < 2e-3 {
            return;
        }
        let want = s.t - t0;
        for k in 0..s.x.len() {
            if s.c[k] > C_A {
                let e = (s.age[k] - want).abs() / want;
                e_all = e_all.max(e);
                let b = if s.c[k] <= 1e-4 {
                    0
                } else if s.c[k] <= 1e-2 {
                    1
                } else {
                    2
                };
                e_band[b] = e_band[b].max(e);
            }
        }
    })?;
    println!(
        "one_ignition_is_uniform_clock: {} samples, max e {e_all:.3e}, bar {tol:.0e}; bands \
         (c_a,1e-4] {:+.3e}, (1e-4,1e-2] {:+.3e}, (1e-2,1] {:+.3e}",
        samples.get(),
        e_band[0],
        e_band[1],
        e_band[2]
    );
    if e_all > tol {
        println!("MISSED one ignition (R): measured {e_all:.3e}, bar {tol:.0e}");
        println!(
            "  band maxima: (c_a,1e-4] {:+.3e}, (1e-4,1e-2] {:+.3e}, (1e-2,1] {:+.3e}",
            e_band[0],
            e_band[1],
            e_band[2]
        );
    }
    Ok(())
}

/// What both `two_kernels_two_clocks` and `age_budget_closes` measure: the
/// planar kernel from t = 0 (t_fd_origin 0, c~ = 1 for x < -0.07 m) and a
/// late kernel - a slab c~ = 1 on [0.09, 0.11] m - at t_ig = 5 ms, to
/// 0.03 s. The midpoint is x = 0.01 m: cells left of it are kernel 1's and
/// hold a = t, cells right of it hold a = t - 5 ms.
struct TwoClocks {
    merge: Option<f64>,
    /// The samples in the window `t_ig + dt < t < merge` (every sample
    /// after `t_ig + dt` when the kernels never merge).
    window: usize,
    worst_left: f64,
    worst_right: f64,
    budget: Vec<AgeBudgetRow>,
}

fn two_kernel_run(g: &Gpu) -> Result<TwoClocks, Error> {
    let t_ig = 5e-3;
    let (spec, dt) = age_spec(0.0);
    fn slab(x: f64) -> f64 {
        if (0.09..=0.11).contains(&x) {
            1.0
        } else {
            0.0
        }
    }
    let opts = AgeOptions { c_a: C_A, late: vec![LateKernel { t_ig, profile: slab }] };
    let mut merge: Option<f64> = None;
    let mut window = 0usize;
    let (mut wl, mut wr) = (0.0f64, 0.0f64);
    let aged = run_aged(g, &spec, &opts, &|x| if x < -0.07 { 1.0 } else { 0.0 }, 0.03, &mut |s: &AgeSample<'_>| {
        // The merge: the first sample whose c~ at the cell nearest
        // x = 0.01 m, midway between the kernels, has reached c_a.
        if merge.is_none() && s.c[nearest(s.x, 0.01)] >= C_A {
            merge = Some(s.t);
        }
        // Before the merge each kernel's cells hold their own clock; no
        // merge means every sample after the second kernel's first step.
        if t_ig + dt < s.t && merge.map_or(true, |m| s.t < m) {
            window += 1;
            for k in 0..s.x.len() {
                if s.c[k] <= C_A {
                    continue;
                }
                if s.x[k] < 0.01 {
                    wl = wl.max((s.age[k] - s.t).abs() / s.t);
                } else {
                    wr = wr.max((s.age[k] - (s.t - t_ig)).abs() / (s.t - t_ig));
                }
            }
        }
    })?;
    let budget = aged.budget;
    Ok(TwoClocks { merge, window, worst_left: wl, worst_right: wr, budget })
}

/// 4: ON with two kernels ignited 5 ms apart: each kernel's cells hold
/// t and t - 5 ms until they merge (R). The window between the second
/// kernel's first step and the merge must hold at least 20 samples (D):
/// the c_a level of an implicit Euler tail outruns the flame, so the first
/// layout (kernel 1 on x < 0, kernel 2 on [0.04, 0.06] m) merged at
/// 2.76 ms, before kernel 2 existed. A clock miss is a finding, printed as
/// MISSED.
#[test]
fn two_kernels_two_clocks() -> Result<(), Error> {
    let Some(g) = gpu("two_kernels_two_clocks") else { return Ok(()) };
    let tc = two_kernel_run(&g)?;
    let tol = tol_r();
    match tc.merge {
        Some(m) => println!("two_kernels_two_clocks: merge at {m:.4e} s"),
        None => println!("two_kernels_two_clocks: no merge by 0.03 s"),
    }
    println!(
        "  window: {} samples after t_ig + dt and before the merge",
        tc.window
    );
    println!(
        "  per-side maxima: left {:+.3e}, right {:+.3e}, bar {tol:.0e}",
        tc.worst_left, tc.worst_right
    );
    assert!(
        tc.window >= 20,
        "the window held {} samples, fewer than 20 (SPEC-LIT §160.5)",
        tc.window
    );
    let worst = tc.worst_left.max(tc.worst_right);
    if worst > tol {
        println!("MISSED two clocks (R): measured {worst:.3e}, bar {tol:.0e}");
    }
    Ok(())
}

/// 5: after the merge the alpha budget closes (A, 1e-6 of the summed
/// magnitudes, both builds): every row from the step after the late kernel
/// on within 1e-6 of its scale. A miss is printed as MISSED.
#[test]
fn age_budget_closes() -> Result<(), Error> {
    let Some(g) = gpu("age_budget_closes") else { return Ok(()) };
    let tc = two_kernel_run(&g)?;
    let t_ig = 5e-3;
    let ratio = |r: &AgeBudgetRow| {
        let s = r.scale();
        if s > 0.0 {
            r.residual().abs() / s
        } else {
            0.0
        }
    };
    let (mut worst_all, mut worst_merged) = (0.0f64, 0.0f64);
    for r in &tc.budget {
        let e = ratio(r);
        worst_all = worst_all.max(e);
        if tc.merge.map_or(false, |m| r.t > m) {
            worst_merged = worst_merged.max(e);
        }
    }
    // The gate: every row from the step after the late kernel on.
    let after = tc.budget.iter().position(|r| r.t > t_ig);
    let gate =
        after.map_or(0.0, |k| tc.budget[k + 1..].iter().map(&ratio).fold(0.0f64, f64::max));
    println!(
        "age_budget_closes: max |residual|/scale {worst_all:.3e} overall, {worst_merged:.3e} \
         after the merge, {gate:.3e} from the step after the kernel, bar 1e-6"
    );
    if gate > 1e-6 {
        println!("MISSED age budget (A): measured {gate:.3e}, bar 1e-6");
    }
    Ok(())
}

/// 6: the refusals (D): a second `enable_age`, c_a = 0 and c_a = NaN, each
/// message naming `dust::age`.
#[test]
fn enable_age_refuses() -> Result<(), Error> {
    let Some(g) = gpu("enable_age_refuses") else { return Ok(()) };
    let (spec, _) = age_spec(0.0);
    let hm = tube_mesh(&spec)?;
    let gm = GpuMesh::upload(&g, &hm)?;
    let mut prog = ProgressEquation::new(&g, &hm, &gm, age_controls(spec.dt))?;
    prog.enable_age(&g, 0.0, C_A)?;
    assert!(prog.age_enabled(), "the first enable_age holds");
    let err = prog.enable_age(&g, 0.0, C_A).expect_err("a second enable_age is refused");
    assert!(err.to_string().contains("dust::age"), "{err}");
    let mut fresh = ProgressEquation::new(&g, &hm, &gm, age_controls(spec.dt))?;
    for c_a in [0.0, f64::NAN] {
        let err = fresh.enable_age(&g, 0.0, c_a).expect_err("c_a must be positive and finite");
        assert!(err.to_string().contains("dust::age"), "{err}");
    }
    // A c~ field with a mean/mapped/turbulent boundary rule: the age field
    // carries none of them.
    fresh.field_mut().mean_rules.push(ofgpu::field::patch_mean::PatchMeanRule {
        kind: ofgpu::field::BcKind::FixedMean,
        start: 0,
        size: 1,
        src_start: 0,
        src_size: 0,
        target: ofgpu::Vec3::ZERO,
        fraction: 1.0,
    });
    let err = fresh
        .enable_age(&g, 0.0, C_A)
        .expect_err("a mean-rule c~ field is refused");
    assert!(err.to_string().contains("dust::age"), "{err}");
    println!(
        "enable_age_refuses: a second call, c_a = 0, c_a = NaN and a mean-rule c~ field are \
         all refused by name"
    );
    Ok(())
}
