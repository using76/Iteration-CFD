// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.
// Provenance: see PROVENANCE.md. No GPL-licensed source was consulted.

// The DST-04 gates of SPEC-LIT §155.8, written once and included both by
// src/dust/tests_progress.rs (as dust::progress::tests) and by
// tests/dust_progress.rs, so both builds run them. Every row is class P at
// one tolerance in both builds; the device rows of the ignition kernel are
// 4 ulp in either build. A GPU test returns early, by name, on a box
// without a device.
// The specification was informed by a review of FSCDustFoam (Huang &
// Lipatnikov, GPL) and no code was copied.
// No GPL-licensed source was consulted.

use ofgpu::device::Gpu;
use ofgpu::dust::courant::{check_fixed_step, predicted, CO_F_MAX};
use ofgpu::dust::device::{ignition as launch_ignition, ProgressKernels};
use ofgpu::dust::fsc::{evaluate, CombustionModel, FscConstants, FscInputs, FscState, BETA_STAR};
use ofgpu::dust::ignition::IgnitionSource;
use ofgpu::dust::progress::{PlanarExact, BOUNDS_TOL};
use ofgpu::dust::psi::{reaction_time, PsiOptions, PsiProblem, PublishedCurve};
use ofgpu::dust::thermo::{tm4513_gas_set, EquivalentGas, FormulaInputs};
use ofgpu::dust::tube1d::{run, tube_mesh, FrozenTurbulence, KappaLaw, TubeRun, TubeSpec};
use ofgpu::error::Error;
use ofgpu::fv::{DivScheme, Limiter};
use ofgpu::io::case::TurbulenceControls;
use ofgpu::timescheme::DdtScheme;
use ofgpu::{DVec3, GpuMesh, Scalar};

/// Leeds (H20 Tables 1, 4; SPEC-LIT §151.5): u' = 1.62 m/s.
const K0: f64 = 3.94;
const EPS0: f64 = 390.53;
const PR_T: f64 = 0.4;
const S_L_LEEDS: f64 = 0.12;
const RHO_U_LEEDS: f64 = 1.32;
const T_U_LEEDS: f64 = 328.0;
/// REMBE (H22, SPEC-LIT §151.5): sigma of §152.10, T_0, S_L(0.62) by (152.13).
const SIGMA_REMBE: f64 = 7.04;
const T_U_REMBE: f64 = 273.0;
const S_L_REMBE: f64 = 0.2411076681537468;
/// The planar run's end, s (§155.8).
const T_PLANAR: f64 = 0.02;

fn kappa_u_leeds() -> f64 {
    1.8e-5 / (1.32 * 0.7)
}

/// REMBE's unburned state (SPEC-LIT §151.5): rho_u from the formula-mode
/// gas at p_0 = 101 325 Pa and T_0 (152.9), and kappa_u = mu/(rho_u Pr)
/// with Sutherland air mu = 1.458e-6 T^1.5/(T + 110.4) and Pr = 0.7.
fn rembe_state() -> (f64, f64) {
    let gas = EquivalentGas::formula(
        &FormulaInputs::huang2022_rembe().expect("REMBE inputs"),
        &tm4513_gas_set().expect("gas set"),
    )
    .expect("REMBE gas");
    let rho_u = gas.rho(101_325.0, T_U_REMBE, 0.0).expect("rho_u");
    let mu = 1.458e-6 * T_U_REMBE.powf(1.5) / (T_U_REMBE + 110.4);
    (rho_u, mu / (rho_u * 0.7))
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

/// A value as the device holds it: rounded to `Scalar`, widened back.
#[allow(clippy::unnecessary_cast)]
fn r(v: f64) -> f64 {
    v as Scalar as f64
}

/// The frozen Leeds turbulence through the k-omega mapping of §154.4:
/// omega = eps/(beta* k), nu_t = k/omega, as the device holds them.
fn frozen() -> FrozenTurbulence {
    let omega = EPS0 / (BETA_STAR * K0);
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

/// The planar reference of (155.5) from the host twin of the closures.
fn planar_exact(c: &FscConstants) -> PlanarExact {
    let f = frozen();
    let s = FscState { t: 1000.0, k: f.k, omega: f.omega, nu_t: f.nu_t };
    let dev = evaluate(c, &s, f64::INFINITY, 1.0);
    PlanarExact { d_inf: dev.d_t, tau_l: dev.tau_l, u_t_isp: dev.u_t_isp }
}

/// The c~ equation's controls: Euler, limited (van Leer) bounded
/// convection, PBiCGStab with DILU, each solve to a 100-fold residual cut
/// and at most the build's floor.
fn controls(dt: f64) -> TurbulenceControls {
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
    // Each solve of the outer loop needs only cut its residual 100-fold:
    // the outer tolerance, not the inner one, sets the converged state.
    ctrl.k_solver.rel_tol = 1e-2;
    ctrl.k_solver.max_iter = 2000;
    ctrl.k_solver.precon = ofgpu::io::case::Preconditioner::Dilu;
    ctrl.k_solver.report_residuals = false;
    ctrl
}

/// The outer tolerance on max |c~ change| per step.
const OUTER_TOL: f64 = if cfg!(feature = "single") { 1e-6 } else { 1e-8 };

/// A planar tube in frozen Leeds turbulence (§155.8): Q = 0, kappa = 0, the
/// burned end at -0.08 m, the fresh end at 0.12 m, a step at x = 0 (a node
/// on all three meshes), dt from the PREDICTED Co_f (155.9).
fn planar_spec(sigma: f64, n: usize, co_f: f64) -> (TubeSpec, PlanarExact, f64) {
    // Q is off, so the closure's T_b only has to admit K_b: Leeds' 5.06.
    let c = constants(5.06, T_U_LEEDS, S_L_LEEDS, kappa_u_leeds(), t_r_leeds());
    let ex = planar_exact(&c);
    let dx = 0.2 / n as f64;
    let isp = ex.u_t_isp;
    let dt = co_f * dx / ((sigma - 1.0) * isp + sigma * isp);
    let pred = predicted(dt, dx, (sigma - 1.0) * isp, sigma, isp);
    let spec = TubeSpec {
        n_cells: n,
        x_lo: -0.08,
        x_hi: 0.12,
        sigma,
        rho_u: RHO_U_LEEDS,
        t_u: T_U_LEEDS,
        kappa: KappaLaw::Zero,
        fsc: c,
        turbulence: Some(frozen()),
        with_q: false,
        ignition: None,
        t_fd_origin: 0.0,
        dt,
        max_outer: 20,
        outer_tol: OUTER_TOL,
        ctrl: controls(dt),
    };
    (spec, ex, pred)
}

/// The bounds row of SPEC-LIT 155.8 (D): no cell of any step's result is
/// counted outside [0, 1]. Under `single` the burned plateau of a
/// diffusion-dominated tube drifts above 1 by the float rounding of the
/// matrix diagonal (about eps (1 + 2 D) a step, D = Gamma dt/dx^2), which
/// 155.9 records; that miss is asserted as MISSED in its recorded form, an
/// excursion within 1e-3, and any other count fails.
fn check_bounds(what: &str, run: &TubeRun) {
    if run.out_of_bounds == 0 {
        return;
    }
    let recorded = cfg!(feature = "single") && run.max_excursion <= 1e-3;
    println!(
        "{what}: MISSED: {} cells counted outside [0, 1], the largest excursion {:.2e}{}",
        run.out_of_bounds,
        run.max_excursion,
        if recorded { " (the recorded float finding of SPEC-LIT 155.9)" } else { "" }
    );
    assert!(recorded, "{what}: c~ left [0, 1]: {} cells, excursion {:e}", run.out_of_bounds, run.max_excursion);
}

/// x where cbar crosses `level`, searched from the burned side, by linear
/// interpolation between the two bracketing centres.
fn crossing(x: &[f64], cbar: &[f64], level: f64) -> f64 {
    for s in 0..x.len() - 1 {
        if cbar[s] >= level && cbar[s + 1] < level {
            let f = (cbar[s] - level) / (cbar[s] - cbar[s + 1]);
            return x[s] + f * (x[s + 1] - x[s]);
        }
    }
    f64::NAN
}

/// The three planar errors: x_f and delta_t (155.6) relative and signed,
/// and the profile's Linf.
struct PlanarErrors {
    x_f: f64,
    delta_t: f64,
    linf: f64,
}

fn planar_errors(run: &TubeRun, ex: &PlanarExact) -> PlanarErrors {
    let xf_exact = ex.int_u_t(run.t);
    let dt_exact = ex.delta_t(run.t);
    let xf = crossing(&run.x, &run.cbar, 0.5);
    // max |grad cbar| of (155.6): the largest centre-to-centre difference,
    // its peak placed by the parabola through it and its two neighbours, so
    // the reading does not depend on where the peak falls between samples.
    let g: Vec<f64> = (0..run.x.len() - 1)
        .map(|s| (run.cbar[s + 1] - run.cbar[s]).abs() / (run.x[s + 1] - run.x[s]))
        .collect();
    let k = (1..g.len() - 1).max_by(|&a, &b| g[a].total_cmp(&g[b])).expect("a peak");
    let (gm, g0, gp) = (g[k - 1], g[k], g[k + 1]);
    let curv = gm - 2.0 * g0 + gp;
    let grad = if curv < 0.0 { g0 - 0.125 * (gp - gm) * (gp - gm) / curv } else { g0 };
    let linf = run
        .x
        .iter()
        .zip(&run.cbar)
        .map(|(&x, &cb)| (cb - ex.profile(x, 0.0, run.t)).abs())
        .fold(0.0f64, f64::max);
    PlanarErrors { x_f: xf / xf_exact - 1.0, delta_t: (1.0 / grad) / dt_exact - 1.0, linf }
}

fn planar_case(test: &str, sigma: f64) -> Result<(), Error> {
    let Some(g) = gpu(test) else { return Ok(()) };
    let mut errs: Vec<[f64; 3]> = Vec::new();
    for n in [250usize, 500, 1000] {
        let (spec, ex, _) = planar_spec(sigma, n, CO_F_MAX);
        let run = run(&g, &spec, &|x| if x < 0.0 { 1.0 } else { 0.0 }, T_PLANAR, &mut |_, _, _| {})?;
        let e = planar_errors(&run, &ex);
        println!(
            "{test}: n = {n}: steps {}, outer iterations {} (at most {} a step), x_f err {:+.3e}, \
             delta_t err {:+.3e}, Linf {:.3e}, Co_f max {:.3}, excursion {:.1e}, out of bounds {}",
            run.steps,
            run.outer_total,
            run.max_outer_used,
            e.x_f,
            e.delta_t,
            e.linf,
            run.max_co_f,
            run.max_excursion,
            run.out_of_bounds
        );
        check_bounds(&format!("{test}: n = {n}"), &run);
        errs.push([e.x_f.abs(), e.delta_t.abs(), e.linf]);
    }
    let fine = errs[2];
    assert!(fine[0] <= 0.005, "{test}: x_f error {:.4e} > 0.5 %", fine[0]);
    assert!(fine[1] <= 0.02, "{test}: delta_t error {:.4e} > 2 %", fine[1]);
    assert!(fine[2] <= 0.02, "{test}: profile Linf {:.4e} > 0.02", fine[2]);
    for (k, what) in ["x_f", "delta_t", "Linf"].iter().enumerate() {
        let seq = [errs[0][k], errs[1][k], errs[2][k]];
        if seq[1] <= seq[0] && seq[2] <= seq[1] {
            continue;
        }
        // SPEC-LIT 155.4: a sigma != 1 miss is a finding about H20's claim,
        // recorded in 155.9 and asserted as MISSED, never a widened band.
        // The recorded miss is an error at its floor: every rise ends at an
        // error within 1e-3 (delta_t at sigma = 5.06 in both builds; under
        // `single` also x_f and Linf on the finest mesh, the float floor).
        let rises_at_floor = seq.windows(2).all(|w| w[1] <= w[0] || w[1] <= 1e-3);
        let recorded =
            (cfg!(feature = "single") || (sigma != 1.0 && k == 1)) && rises_at_floor;
        println!(
            "{test}: MISSED: the {what} error is not non-increasing over 250/500/1000 cells: \
             {:.3e}, {:.3e}, {:.3e}{}",
            seq[0],
            seq[1],
            seq[2],
            if recorded { " (a finding SPEC-LIT 155.9 records)" } else { "" }
        );
        assert!(recorded, "{test}: the {what} error rose with the mesh: {seq:?}");
    }
    Ok(())
}

/// 1: sigma = 1 - (155.3) is linear advection-diffusion and (155.5) exact.
#[test]
fn planar_exact_sigma_1() -> Result<(), Error> {
    planar_case("planar_exact_sigma_1", 1.0)
}

/// 2: sigma = 5.06 - H20's exactness claim for the BML flame (§155.4).
#[test]
fn planar_exact_sigma_5_06() -> Result<(), Error> {
    planar_case("planar_exact_sigma_5_06", 5.06)
}

/// A laminar-limit tube (§155.8): D_t = U_t = 0, Q of the published closure
/// with rho kappa = rho_u kappa_u (n = 1), 200 cells per delta_L, the
/// front starting at x = 0 with an exponential preheat, dt moving the
/// front two cells per step (the Euler step's numerical diffusion
/// dt S_L^2/2 is then 1/200 of kappa_u). Returns the front speed over the
/// last half of an 8 delta_L travel.
fn laminar_speed(
    g: &Gpu,
    sigma: f64,
    t_u: f64,
    s_l: f64,
    kappa_u: f64,
    rho_u: f64,
    zfk: bool,
) -> Result<(f64, TubeRun), Error> {
    let pc = PublishedCurve::new(sigma)?;
    let sol = PsiProblem { curve: &pc, theta_u: 20000.0 / t_u, n: 1.0 }.eigen(&PsiOptions::default())?;
    let t_r = if zfk {
        kappa_u / (s_l * s_l * sol.lambda_zfk)
    } else {
        reaction_time(sol.psi, kappa_u, s_l)?
    };
    let d_l = kappa_u / s_l;
    let dx = d_l / 200.0;
    let (behind, travel, ahead) = (4.0, 8.0, 6.0);
    let n = ((behind + travel + ahead) * 200.0) as usize;
    let dt = 2.0 * dx / s_l;
    let spec = TubeSpec {
        n_cells: n,
        x_lo: -behind * d_l,
        x_hi: (travel + ahead) * d_l,
        sigma,
        rho_u,
        t_u,
        kappa: KappaLaw::ConstantRhoKappa { kappa_u },
        fsc: constants(sigma, t_u, s_l, kappa_u, t_r),
        turbulence: None,
        with_q: true,
        ignition: None,
        t_fd_origin: 0.0,
        dt,
        max_outer: 20,
        outer_tol: OUTER_TOL,
        ctrl: controls(dt),
    };
    let t_end = travel * d_l / s_l;
    let mut track: Vec<(f64, f64)> = Vec::new();
    let run = run(
        g,
        &spec,
        &|x| if x < 0.0 { 1.0 } else { (-x / d_l).exp() },
        t_end,
        &mut |t, x, cbar| track.push((t, crossing(x, cbar, 0.5))),
    )?;
    let (t1, x1) = track[track.len() / 2];
    let (t2, x2) = *track.last().expect("samples");
    Ok(((x2 - x1) / (t2 - t1), run))
}

/// 3: the laminar limit propagates at S_L with the regenerated Psi (§153).
#[test]
fn laminar_limit_is_s_l() -> Result<(), Error> {
    let Some(g) = gpu("laminar_limit_is_s_l") else { return Ok(()) };
    let (rho_r, kappa_r) = rembe_state();
    for (name, sigma, t_u, s_l, kappa_u, rho_u) in [
        ("Leeds", 5.06, T_U_LEEDS, S_L_LEEDS, kappa_u_leeds(), RHO_U_LEEDS),
        ("REMBE", SIGMA_REMBE, T_U_REMBE, S_L_REMBE, kappa_r, rho_r),
    ] {
        let (s, run) = laminar_speed(&g, sigma, t_u, s_l, kappa_u, rho_u, false)?;
        let e = s / s_l - 1.0;
        println!(
            "laminar_limit_is_s_l: {name}: speed {s:.6} m/s, S_L {s_l}, rel {e:.3e}, steps {}, \
             outer <= {}, out of bounds {}",
            run.steps, run.max_outer_used, run.out_of_bounds
        );
        println!("  the largest excursion outside [0, 1]: {:.1e}", run.max_excursion);
        check_bounds(name, &run);
        assert!(e.abs() <= 0.01, "{name}: the laminar front runs at {s} m/s, not S_L = {s_l} within 1 %");
    }
    Ok(())
}

/// 4: with t_r from Lambda_ZFK the front runs at (Lambda_ZFK/Lambda)^{1/2} S_L.
#[test]
fn laminar_limit_zfk_speed() -> Result<(), Error> {
    let Some(g) = gpu("laminar_limit_zfk_speed") else { return Ok(()) };
    let (rho_r, kappa_r) = rembe_state();
    for (name, sigma, t_u, s_l, kappa_u, rho_u) in [
        ("Leeds", 5.06, T_U_LEEDS, S_L_LEEDS, kappa_u_leeds(), RHO_U_LEEDS),
        ("REMBE", SIGMA_REMBE, T_U_REMBE, S_L_REMBE, kappa_r, rho_r),
    ] {
        let pc = PublishedCurve::new(sigma)?;
        let sol =
            PsiProblem { curve: &pc, theta_u: 20000.0 / t_u, n: 1.0 }.eigen(&PsiOptions::default())?;
        let want = s_l * (sol.lambda_zfk / sol.lambda).sqrt();
        let (s, run) = laminar_speed(&g, sigma, t_u, s_l, kappa_u, rho_u, true)?;
        let e = s / want - 1.0;
        println!(
            "laminar_limit_zfk_speed: {name}: speed {s:.6} m/s, want {want:.6}, rel {e:.3e}, \
             out of bounds {}",
            run.out_of_bounds
        );
        println!("  the largest excursion outside [0, 1]: {:.1e}", run.max_excursion);
        check_bounds(name, &run);
        assert!(e.abs() <= 0.01, "{name}: the ZFK front runs at {s} m/s, not {want} within 1 %");
    }
    Ok(())
}

/// 5: the planar sigma = 5.06 front speed changes < 0.5 % between Co_f 0.1
/// and 0.4; the measured Co_f stays at or below the prediction.
#[test]
fn flame_courant_independent() -> Result<(), Error> {
    let Some(g) = gpu("flame_courant_independent") else { return Ok(()) };
    let mut speeds = Vec::new();
    for co in [0.1, 0.4] {
        let (spec, _, pred) = planar_spec(5.06, 250, co);
        let run = run(&g, &spec, &|x| if x < 0.0 { 1.0 } else { 0.0 }, T_PLANAR, &mut |_, _, _| {})?;
        let s = crossing(&run.x, &run.cbar, 0.5) / run.t;
        println!(
            "flame_courant_independent: Co_f {co}: predicted {pred:.4}, measured max {:.4}, \
             speed {s:.6} m/s, steps {}",
            run.max_co_f, run.steps
        );
        println!("  the largest excursion outside [0, 1]: {:.1e}", run.max_excursion);
        assert_eq!(run.out_of_bounds, 0, "Co_f {co}: c~ left [0, 1]");
        assert!(run.max_co_f <= pred * (1.0 + 1e-6), "Co_f {co}: measured {} > predicted {pred}", run.max_co_f);
        speeds.push(s);
    }
    let d = (speeds[1] / speeds[0] - 1.0).abs();
    assert!(d < 0.005, "the front speed changed by {d:.4e} between Co_f 0.1 and 0.4");
    Ok(())
}

/// 6: a fixed dt with a predicted Co_f above 1 is refused by name, with the
/// setting and the value in the message (D).
#[test]
fn flame_courant_refused() {
    let (dx, isp, sigma) = (2.0e-4, 1.127, 5.06);
    let dt_ok = 0.4 * dx / ((2.0 * sigma - 1.0) * isp);
    let ok = check_fixed_step(dt_ok, predicted(dt_ok, dx, (sigma - 1.0) * isp, sigma, isp), CO_F_MAX)
        .expect("Co_f 0.4 is admitted");
    assert!((ok - 0.4).abs() < 1e-12, "predicted {ok}");
    let dt = 1.2 * dx / ((2.0 * sigma - 1.0) * isp);
    let co = predicted(dt, dx, (sigma - 1.0) * isp, sigma, isp);
    let err = check_fixed_step(dt, co, CO_F_MAX).expect_err("Co_f 1.2 must be refused");
    let msg = err.to_string();
    println!("flame_courant_refused: {msg}");
    assert!(msg.contains("deltaT"), "{msg}");
    assert!(msg.contains("combustion.maxFlameCourant"), "{msg}");
    assert!(msg.contains("1.2000"), "{msg}");
    assert!(check_fixed_step(dt, co, 0.0).is_err(), "a zero bound is refused");
}

/// The distance in units in the last place between a device word and the
/// host value rounded to `Scalar`; `u64::MAX` across a sign change.
fn ulps(dev: Scalar, host: f64) -> u64 {
    let h = host as Scalar;
    if dev.is_sign_negative() != h.is_sign_negative() {
        return u64::MAX;
    }
    if std::mem::size_of::<Scalar>() == 8 {
        (dev as f64).to_bits().abs_diff((h as f64).to_bits())
    } else {
        u64::from((dev as f32).to_bits().abs_diff((h as f32).to_bits()))
    }
}

/// 7: the ignition kernel (155.4) against its host twin (4 ulp), held after
/// t_0 bit for bit, and c~ at the ignition point >= 0.99 after t_0.
#[test]
fn ignition_kernel_forms() -> Result<(), Error> {
    let Some(g) = gpu("ignition_kernel_forms") else { return Ok(()) };
    let src = IgnitionSource::huang2020(DVec3::new(0.0, 0.0, 0.0));
    let (dt, n) = (1.0e-5, 160usize);
    let c = constants(5.06, T_U_LEEDS, S_L_LEEDS, kappa_u_leeds(), t_r_leeds());
    let spec = TubeSpec {
        n_cells: n,
        x_lo: 0.0,
        x_hi: 4.0e-3,
        sigma: 5.06,
        rho_u: RHO_U_LEEDS,
        t_u: T_U_LEEDS,
        kappa: KappaLaw::ConstantRhoKappa { kappa_u: kappa_u_leeds() },
        fsc: c,
        turbulence: None,
        with_q: false,
        ignition: Some(src),
        t_fd_origin: src.t0,
        dt,
        max_outer: 20,
        outer_tol: OUTER_TOL,
        ctrl: controls(dt),
    };

    // The kernel against the host twin, before and after t_0.
    let hm = tube_mesh(&spec)?;
    let gm = GpuMesh::upload(&g, &hm)?;
    let pk = ProgressKernels::new(&g)?;
    let p = src.local_point(&hm);
    let mut w = g.zeros::<Scalar>(n)?;
    let mut worst = 0u64;
    let mut held = Vec::new();
    for t in [0.0, 0.5e-3, 1.5e-3, 3.0e-3] {
        launch_ignition(&g, &pk, &mut w, &gm.c, &src, p, t, n)?;
        let got = g.download(&w)?;
        for (i, &v) in got.iter().enumerate() {
            worst = worst.max(ulps(v, src.bracket(hm.c[i], p, t)));
            assert!(v == 0.0 || v.abs() >= Scalar::MIN_POSITIVE, "a subnormal W at cell {i}");
        }
        if t > src.t0 {
            held.push(got);
        }
    }
    println!("ignition_kernel_forms: device vs host worst {worst} ulp");
    assert!(worst <= 4, "the ignition kernel is {worst} ulp from its host twin");
    assert!(held[0] == held[1], "W changed after t_0");

    // The tube: c~ at the cell nearest the point after t_0, recovered from
    // cbar by (155.1)-(155.2): c~ = cbar / (sigma - (sigma - 1) cbar).
    let mut c_min_after = 1.0f64;
    let run = run(&g, &spec, &|_| 0.0, 1.2e-3, &mut |t, _, cbar| {
        if t > src.t0 {
            c_min_after = c_min_after.min(cbar[0] / (5.06 - 4.06 * cbar[0]));
        }
    })?;
    println!(
        "ignition_kernel_forms: c~ at the point {:.6} at t = {:.4e} s, min {c_min_after:.6} \
         after t_0, out of bounds {}",
        run.c[0], run.t, run.out_of_bounds
    );
    println!("  the largest excursion outside [0, 1]: {:.1e}", run.max_excursion);
    assert_eq!(run.out_of_bounds, 0, "c~ left [0, 1]");
    assert!(run.c[0] >= 0.99, "c~(r = 0, t > t_0) = {} < 0.99", run.c[0]);
    assert!(c_min_after >= 0.99, "c~ at the point fell to {c_min_after} after t_0");
    Ok(())
}

/// 8: a harsh tube - REMBE's sigma, frozen turbulence, Q, ignition - keeps
/// c~ in [0, 1] and the bounds counter at 0 (D).
#[test]
fn bounded() -> Result<(), Error> {
    let Some(g) = gpu("bounded") else { return Ok(()) };
    let src = IgnitionSource::huang2020(DVec3::new(0.0, 0.0, 0.0));
    let sigma = SIGMA_REMBE;
    let c = constants(sigma, T_U_LEEDS, S_L_LEEDS, kappa_u_leeds(), t_r_leeds());
    let isp = planar_exact(&c).u_t_isp;
    let n = 250usize;
    let dx = 0.05 / n as f64;
    let dt = CO_F_MAX * dx / ((2.0 * sigma - 1.0) * isp);
    let spec = TubeSpec {
        n_cells: n,
        x_lo: 0.0,
        x_hi: 0.05,
        sigma,
        rho_u: RHO_U_LEEDS,
        t_u: T_U_LEEDS,
        kappa: KappaLaw::ConstantRhoKappa { kappa_u: kappa_u_leeds() },
        fsc: c,
        turbulence: Some(frozen()),
        with_q: true,
        ignition: Some(src),
        t_fd_origin: src.t0,
        dt,
        max_outer: 20,
        outer_tol: OUTER_TOL,
        ctrl: controls(dt),
    };
    let run = run(&g, &spec, &|_| 0.0, 6.0e-3, &mut |_, _, _| {})?;
    let lo = run.c.iter().cloned().fold(f64::INFINITY, f64::min);
    let hi = run.c.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    println!(
        "bounded: steps {}, c~ in [{lo:e}, {hi}], x(cbar = 0.5) {:.4e} m, out of bounds {}",
        run.steps,
        crossing(&run.x, &run.cbar, 0.5),
        run.out_of_bounds
    );
    println!("  the largest excursion outside [0, 1]: {:.1e}", run.max_excursion);
    assert_eq!(run.out_of_bounds, 0, "c~ left [0, 1]");
    let tol = f64::from(BOUNDS_TOL).max(OUTER_TOL);
    assert!(lo >= -tol && hi <= 1.0 + tol, "c~ in [{lo}, {hi}]");
    Ok(())
}
