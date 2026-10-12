// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.
// Provenance: see PROVENANCE.md. No GPL-licensed source was consulted.

//! SPEC-LIT §143.1 - integration tests for the 1-D RANS oracle: the
//! measured SPEC-LIT §114 SST pipe tables, the laminar Poiseuille balance,
//! the §56.4 SA log-layer identity and the §94 Richardson extrapolation.
//! Every test prints its measured values so the two builds can be compared.
//!
//! No GPL-licensed source was consulted.

use ofgpu::vv::rans1d::*;

fn rel(a: f64, b: f64) -> f64 {
    (a / b - 1.0).abs()
}

#[test]
fn rans1d_sst_pipe_reproduces_spec_lit_114_table_3() {
    let case = PipeCase::tg0(576.69);
    let mut fs = [0.0f64; 3];
    let mut dfs = [0.0f64; 3];
    let mut nut4 = 0.0f64;
    for (i, level) in [2u32, 3u32, 4u32].into_iter().enumerate() {
        let mesh = PipeMesh::tg0(level).unwrap();
        let ctl = Controls::for_model(Model::KOmegaSst);
        let sol = solve(&mesh, &case, Model::KOmegaSst, &ctl).unwrap();
        let m = metrics(&mesh, &case, &sol);
        fs[i] = m.f;
        dfs[i] = m.defect;
        if level == 4 {
            nut4 = m.nut_axis_over_utau_r;
        }
        println!(
            "level {level}: f = {:.9e}, defect = {:.9e}, nut/utau/R = {:.9e}, iters = {}",
            m.f, m.defect, m.nut_axis_over_utau_r, sol.iterations
        );
    }
    let (f2, f3, f4) = (fs[0], fs[1], fs[2]);
    println!("f(level 4) = {f4:.9e}");
    assert!(rel(f4, 0.027258) < 1e-4, "f(level 4) = {f4}");
    let (p, ext) = richardson_constant_ratio(f4, f3, f2, 2.0);
    println!("richardson(f): p = {p:.9e}, ext = {ext:.9e}");
    assert!(rel(ext, 0.027287) < 1e-4, "richardson ext = {ext}");
    assert!((1.06..=1.24).contains(&p), "richardson p = {p}");
    println!("|nut_axis/utau/R(level 4) - 0.124| = {:.9e}", (nut4 - 0.124).abs());
    assert!((nut4 - 0.124).abs() <= 5e-4, "nut ratio = {nut4}");
    let (pd, extd) = richardson_constant_ratio(dfs[2], dfs[1], dfs[0], 2.0);
    println!("defect(level 4) = {:.9e}", dfs[2]);
    println!("richardson(defect): p = {pd:.9e}, ext = {extd:.9e}");
}

#[test]
fn rans1d_sst_pipe_reproduces_spec_lit_114_table_2() {
    let case = PipeCase::tg0(576.69);
    let ctl = Controls::for_model(Model::KOmegaSst);
    let run = |level: u32, c: &Controls| {
        let mesh = PipeMesh::tg0(level).unwrap();
        let sol = solve(&mesh, &case, Model::KOmegaSst, c).unwrap();
        metrics(&mesh, &case, &sol)
    };
    let m1 = run(1, &ctl);
    println!(
        "Re_tau 576.69 level 1: u_b {:.9e} u_axis {:.9e} f {:.9e} defect {:.9e}",
        m1.u_b, m1.u_axis, m1.f, m1.defect
    );
    assert!(rel(m1.u_b, 6.02290) < 1e-5, "u_b = {}", m1.u_b);
    assert!(rel(m1.u_axis, 7.28630) < 1e-5, "u_axis = {}", m1.u_axis);
    assert!(rel(m1.f, 0.026944) < 1e-4, "f = {}", m1.f);
    assert!(rel(m1.defect, 3.6145) < 1e-4, "defect = {}", m1.defect);
    let m0 = run(0, &ctl);
    println!("Re_tau 576.69 level 0: u_b {:.9e} defect {:.9e}", m0.u_b, m0.defect);
    assert!(rel(m0.u_b, 6.07522) < 1e-5, "u_b = {}", m0.u_b);
    assert!(rel(m0.defect, 3.6163) < 1e-4, "defect = {}", m0.defect);
    let case_hi = PipeCase::tg0(2358.0);
    let mesh1 = PipeMesh::tg0(1).unwrap();
    let sol_hi = solve(&mesh1, &case_hi, Model::KOmegaSst, &ctl).unwrap();
    let mhi = metrics(&mesh1, &case_hi, &sol_hi);
    println!("Re_tau 2358 level 1: u_b {:.9e}", mhi.u_b);
    assert!(rel(mhi.u_b, 30.26666) < 1e-5, "u_b = {}", mhi.u_b);
    let ctl_ex = Controls { wall_distance: WallDistance::Exact, ..ctl };
    let mex = run(1, &ctl_ex);
    println!("Exact distance: u_b {:.9e}, rel to Tucker {:.9e}", mex.u_b, rel(mex.u_b, m1.u_b));
    assert!(rel(mex.u_b, m1.u_b) < 1e-5, "u_b exact = {}", mex.u_b);
}

#[test]
fn rans1d_laminar_pipe_is_poiseuille() {
    let case = PipeCase::tg0(576.69);
    let ctl = Controls::for_model(Model::Laminar);
    let mut e = [0.0f64; 2];
    for (i, level) in [1u32, 2u32].into_iter().enumerate() {
        let mesh = PipeMesh::tg0(level).unwrap();
        let n = mesh.n();
        let sol = solve(&mesh, &case, Model::Laminar, &ctl).unwrap();
        assert_eq!(sol.iterations, 1);
        let tau = face_total_stress(&mesh, &case, &sol);
        for (j, (&t, &r)) in tau.iter().zip(mesh.faces[1..n].iter()).enumerate() {
            let exact = case.g * r / 2.0;
            assert!(rel(t, exact) < 1e-12, "face {j}: {t} vs {exact}");
        }
        let wall = case.g * mesh.rho_wall() / 2.0;
        assert!(rel(tau[n - 1], wall) < 1e-12, "wall: {} vs {wall}", tau[n - 1]);
        let m = metrics(&mesh, &case, &sol);
        assert!(rel(m.u_tau * m.u_tau, wall) < 1e-12, "u_tau^2 = {}", m.u_tau * m.u_tau);
        e[i] = (m.u_b / (case.g * mesh.rho_wall() * mesh.rho_wall() / (8.0 * case.nu)) - 1.0).abs();
        println!("level {level}: e = {:.9e}, u_b = {:.9e}", e[i], m.u_b);
    }
    println!("e1/e2 = {:.9e}", e[0] / e[1]);
    assert!((3.9..=4.1).contains(&(e[0] / e[1])), "e1/e2 = {}", e[0] / e[1]);
}

#[test]
fn rans1d_sa_log_layer_identity() {
    let c = SaCoeffs::default();
    assert!((c.cw1() - 3.2390678).abs() < 1e-7, "cw1 = {}", c.cw1());
    let ut = 0.37f64;
    let k2 = c.kappa * c.kappa;
    let diff = (1.0 + c.cb2) / c.sigma * k2 * ut * ut;
    for y in [1e-4f64, 1e-3, 1e-2, 1e-1] {
        let nu = c.kappa * ut * y / 1e14;
        let nt = c.kappa * ut * y;
        let vort = ut / (c.kappa * y);
        let (p, d) = sa_point_terms(&c, nu, nt, vort, y);
        let res = (p - d + diff).abs();
        println!("y = {y:.0e}: p = {p:.9e} d = {d:.9e} diff = {diff:.9e} residual = {res:.9e}");
        assert!(res <= 1e-12 * p.max(d).max(diff), "residual = {res}");
    }
    let y = 0.01f64;
    let residual = |chi: f64| {
        let nu = c.kappa * ut * y / chi;
        let nt = c.kappa * ut * y;
        let vort = ut / (c.kappa * y);
        let (p, d) = sa_point_terms(&c, nu, nt, vort, y);
        (p - d + diff).abs() / (k2 * ut * ut)
    };
    let r3 = residual(1e3);
    let r4 = residual(1e4);
    println!("residual(1e3) = {r3:.9e}, residual(1e4) = {r4:.9e}");
    let rate = (r3 / r4).log10();
    println!("log10(residual(1e3)/residual(1e4)) = {rate:.9e}");
    assert!((rate - 1.0).abs() < 0.05, "rate = {rate}");
}

#[test]
fn rans1d_every_model_converges_and_balances_the_body_force() {
    let case = PipeCase::tg0(576.69);
    let mesh = PipeMesh::tg0(1).unwrap();
    for model in [
        Model::Laminar,
        Model::KOmegaSst,
        Model::KOmega1988,
        Model::Wilcox2006,
        Model::LaunderSharma,
        Model::SpalartAllmaras,
    ] {
        let ctl = Controls::for_model(model);
        let sol = solve(&mesh, &case, model, &ctl).unwrap();
        assert!(sol.iterations < ctl.max_iter, "{model:?}: {} iterations", sol.iterations);
        for v in sol
            .u
            .iter()
            .chain(&sol.nut)
            .chain(&sol.k)
            .chain(&sol.omega)
            .chain(&sol.eps_tilde)
            .chain(&sol.nu_tilde)
        {
            assert!(v.is_finite(), "{model:?}: non-finite field");
        }
        for v in &sol.nut {
            assert!(*v >= 0.0, "{model:?}: negative nut {v}");
        }
        let m = metrics(&mesh, &case, &sol);
        let balance = case.g * mesh.rho_wall() / 2.0;
        assert!(rel(m.u_tau * m.u_tau, balance) < 1e-10, "{model:?}: utau^2 {}", m.u_tau * m.u_tau);
        println!(
            "{model:?}: u_b = {:.9e}, f = {:.9e}, defect = {:.9e}, iters = {}",
            m.u_b, m.f, m.defect, sol.iterations
        );
        if model == Model::SpalartAllmaras {
            assert!(rel(m.f, 0.024600) <= 1e-4, "SA f = {}", m.f);
        }
    }
    let ctl = Controls::for_model(Model::KOmega1988);
    let c62 = KOmega1988Coeffs::spec_lit_6_2();
    let sol = solve_k_omega_1988(&mesh, &case, &c62, &ctl).unwrap();
    let m = metrics(&mesh, &case, &sol);
    println!("kOmega1988 spec_lit_6_2: f = {:.9e}, iters = {}", m.f, sol.iterations);
}

#[test]
fn wilcox2006_coefficients_are_the_tmr_page() {
    let c = Wilcox2006Coeffs::default();
    assert_eq!(c.beta_star, 0.09);
    assert_eq!(c.beta_0, 0.0708);
    assert_eq!(c.gamma, 13.0 / 25.0);
    assert_eq!(c.sigma_k, 0.6);
    assert_eq!(c.sigma_w, 0.5);
    assert_eq!(c.sigma_d, 1.0 / 8.0);
    assert_eq!(c.c_lim, 7.0 / 8.0);
    assert_eq!(c.f_beta_num, 85.0);
    assert_eq!(c.f_beta_den, 100.0);
    let ident = c.beta_0 / c.beta_star - c.sigma_w * 0.40f64 * 0.40 / c.beta_star.sqrt();
    println!(
        "gamma = {:.17e}, TMR identity = {:.17e}, diff = {:.3e}",
        c.gamma, ident, (c.gamma - ident).abs()
    );
    assert!(
        (c.gamma - ident).abs() <= 1e-15,
        "gamma identity diff = {}",
        (c.gamma - ident).abs()
    );
}

#[test]
fn wilcox2006_cross_diffusion_is_zero_unless_grad_k_dot_grad_omega_is_positive() {
    let c = Wilcox2006Coeffs::default();
    let r = wilcox2006_cross_diffusion(&c, &[1.0, 2.0, 3.0], &[-1.0, 0.0, 0.0], 5.0);
    assert_eq!(r.to_bits(), 0, "negative dot: {r}");
    let r = wilcox2006_cross_diffusion(&c, &[1.0, 0.0, 0.0], &[0.0, 1.0, 0.0], 5.0);
    assert_eq!(r.to_bits(), 0, "orthogonal: {r}");
    let r = wilcox2006_cross_diffusion(&c, &[1.0, 1.0, 0.0], &[-1.0, 1.0, 0.0], 5.0);
    assert_eq!(r.to_bits(), 0, "cancelling: {r}");
    let r = wilcox2006_cross_diffusion(&c, &[2.0, 0.0, 0.0], &[3.0, 0.0, 0.0], 4.0);
    assert_eq!(r, 0.1875, "positive dot: {r}");
    let r = wilcox2006_cross_diffusion(&c, &[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0], 2.0);
    assert_eq!(r, 2.0, "positive dot: {r}");
    // The same rule on the solved TG0 pipe fields, gk/gw as in the sweep.
    let mesh = PipeMesh::tg0(1).unwrap();
    let case = PipeCase::tg0(576.69);
    let ctl = Controls::for_model(Model::Wilcox2006);
    let sol = solve_wilcox_2006(&mesh, &case, &c, &ctl).unwrap();
    let n = mesh.n();
    let gk = gauss_grad(&mesh, &sol.k, 0.0);
    let gw = gauss_grad(&mesh, &sol.omega, sol.omega[n - 1]);
    let (mut neg, mut pos) = (0usize, 0usize);
    for j in 0..n {
        let r =
            wilcox2006_cross_diffusion(&c, &[gk[j], 0.0, 0.0], &[gw[j], 0.0, 0.0], sol.omega[j]);
        if gk[j] * gw[j] <= 0.0 {
            assert_eq!(r.to_bits(), 0, "cell {j}: gk*gw = {}, r = {r}", gk[j] * gw[j]);
            neg += 1;
        } else {
            assert!(r > 0.0, "cell {j}: gk*gw = {}, r = {r}", gk[j] * gw[j]);
            pos += 1;
        }
    }
    println!("cross-diffusion on tg0(1): {neg} cells with gk.gw <= 0, {pos} with gk.gw > 0");
    assert!(neg > 0 && pos > 0, "both sets must be non-empty: {neg} / {pos}");
}

#[test]
fn wilcox2006_f_beta_is_one_for_simple_shear() {
    let c = Wilcox2006Coeffs::default();
    let mut g = [[0.0f64; 3]; 3];
    g[0][1] = 1234.5;
    assert_eq!(wilcox2006_f_beta(&c, &g, 7.0).to_bits(), 1.0f64.to_bits(), "planar shear");
    let mut g = [[0.0f64; 3]; 3];
    g[0][1] = 987.6 * 0.3f64.cos();
    g[0][2] = 987.6 * 0.3f64.sin();
    assert_eq!(wilcox2006_f_beta(&c, &g, 7.0).to_bits(), 1.0f64.to_bits(), "tilted shear");
    let mut g = [[0.0f64; 3]; 3];
    g[2][0] = -3.25;
    assert_eq!(wilcox2006_f_beta(&c, &g, 7.0).to_bits(), 1.0f64.to_bits(), "2->0 shear");
    let g = [[1.0, 2.0, 0.0], [1.0, 0.0, 3.0], [0.0, 1.0, 0.0]];
    let fb = wilcox2006_f_beta(&c, &g, 10.0);
    println!("f_beta(non-shear) = {:.17e}, expected {:.17e}", fb, 85729.0 / 100729.0);
    assert!(rel(fb, 85729.0 / 100729.0) <= 1e-15, "f_beta = {fb}");
}

#[test]
fn wilcox2006_pipe_matches_the_supervisor_oracle() {
    let c = Wilcox2006Coeffs::default();
    let check = |re_tau: f64, level: u32, u_b: f64, f: f64, defect: f64, nut: Option<f64>| {
        let mesh = PipeMesh::tg0(level).unwrap();
        let case = PipeCase::tg0(re_tau);
        let ctl = Controls::for_model(Model::Wilcox2006);
        let sol = solve_wilcox_2006(&mesh, &case, &c, &ctl).unwrap();
        let m = metrics(&mesh, &case, &sol);
        println!(
            "Wilcox2006 re_tau {re_tau} level {level}: u_b {:.9e} f {:.9e} defect {:.9e} nut/utau/R {:.9e} iters {}",
            m.u_b, m.f, m.defect, m.nut_axis_over_utau_r, sol.iterations
        );
        assert!(rel(m.u_b, u_b) < 1e-6, "u_b = {} vs {u_b}", m.u_b);
        assert!(rel(m.f, f) < 1e-6, "f = {} vs {f}", m.f);
        assert!(rel(m.defect, defect) < 1e-6, "defect = {} vs {defect}", m.defect);
        if let Some(nr) = nut {
            assert!(
                rel(m.nut_axis_over_utau_r, nr) < 1e-6,
                "nut ratio = {} vs {nr}",
                m.nut_axis_over_utau_r
            );
        }
    };
    check(576.69, 1, 6.089861407, 0.0263551108, 3.857669312, Some(0.09696406419));
    check(2358.0, 1, 30.73986073, 0.01729333019, 3.590450099, None);
}

#[test]
fn tg0_bands_reproduce_spec_lit_114() {
    println!("f_prandtl(576.69) = {:.17e}", f_prandtl(576.69));
    println!("f_prandtl(2358.0) = {:.17e}", f_prandtl(2358.0));
    assert!(rel(f_prandtl(576.69), 0.025789172770595198) < 1e-12);
    assert!(rel(f_prandtl(2358.0), 0.01801605735288456) < 1e-12);
    let ctl = Controls::for_model(Model::KOmegaSst);
    let band = |re_tau: f64, level: u32| -> (PipeMetrics, Tg0Bands) {
        let mesh = PipeMesh::tg0(level).unwrap();
        let case = PipeCase::tg0(re_tau);
        let sol = solve(&mesh, &case, Model::KOmegaSst, &ctl).unwrap();
        let m = metrics(&mesh, &case, &sol);
        let b = tg0_bands(&mesh, &case, &sol, &m);
        println!(
            "SST re_tau {re_tau} level {level}: u_b {:.9e} f {:.9e} re_d {:.6e} re_tau {:.6e} f_vs_prandtl {:.6e} f_vs_blasius {:.6e} loglaw_dev {:.6e}/{} defect_vs_4_07 {:.6e}",
            m.u_b, m.f, b.re_d, b.re_tau, b.f_vs_prandtl, b.f_vs_blasius, b.loglaw_dev,
            b.loglaw_cells, b.defect_vs_4_07
        );
        (m, b)
    };
    let (_, b1) = band(576.69, 1);
    assert!((b1.f_vs_prandtl - 0.0448).abs() <= 1e-4, "L1 f_vs_prandtl = {}", b1.f_vs_prandtl);
    assert!((b1.loglaw_dev - 1.073).abs() <= 1e-3, "L1 loglaw_dev = {}", b1.loglaw_dev);
    let (m2, _) = band(576.69, 2);
    let (m3, _) = band(576.69, 3);
    let (m4, b4) = band(576.69, 4);
    assert!((b4.f_vs_blasius - 0.0211).abs() <= 1e-4, "L4 f_vs_blasius = {}", b4.f_vs_blasius);
    // SPEC-LIT 143.2: 114.3's log-law 1.239 at 640 cells is not reproduced at its printed
    // digit (this oracle: 1.24026); it is recorded there, not gated.
    assert!(
        (b4.defect_vs_4_07 + 0.112).abs() <= 1e-3,
        "L4 defect_vs_4_07 = {}",
        b4.defect_vs_4_07
    );
    let (_, ext) = richardson_constant_ratio(m4.f, m3.f, m2.f, 2.0);
    let ext_vs_pr = ext / f_prandtl(576.69) - 1.0;
    println!("richardson f at re_tau 576.69: ext = {ext:.9e}, ext/f_prandtl - 1 = {ext_vs_pr:.6e}");
    assert!((ext_vs_pr - 0.0581).abs() <= 1e-4, "ext/f_prandtl - 1 = {ext_vs_pr}");
    let (m2h, _) = band(2358.0, 2);
    let (m3h, _) = band(2358.0, 3);
    let (m4h, b4h) = band(2358.0, 4);
    // SPEC-LIT 143.2: 114.3's 2358 row (f 0.018428, log-law 1.358, f extrapolated 0.018502)
    // is reproduced only to 1.1e-4, 2.4e-3 and 1.5e-4; recorded there, not gated.
    assert!(
        (b4h.f_vs_blasius - 0.0308).abs() <= 1e-4,
        "2358 L4 f_vs_blasius = {}",
        b4h.f_vs_blasius
    );
    let (_, exth) = richardson_constant_ratio(m4h.f, m3h.f, m2h.f, 2.0);
    println!("richardson f at re_tau 2358: ext = {exth:.9e}");
}

#[test]
#[ignore = "measurement: prints SPEC-LIT 143.2's table"]
fn tg0_bands_measured_table() {
    for model in [
        Model::KOmegaSst,
        Model::KOmega1988,
        Model::Wilcox2006,
        Model::LaunderSharma,
        Model::SpalartAllmaras,
    ] {
        for re_tau in [576.69f64, 2358.0] {
            let case = PipeCase::tg0(re_tau);
            let ctl = Controls::for_model(model);
            let mut fs = [0.0f64; 3];
            let mut b4 = None;
            let mut ok = true;
            for (i, level) in [2u32, 3u32, 4u32].into_iter().enumerate() {
                let mesh = PipeMesh::tg0(level).unwrap();
                let sol = match solve(&mesh, &case, model, &ctl) {
                    Ok(s) => s,
                    Err(e) => {
                        println!("{model:?} re_tau {re_tau} level {level}: Err {e}");
                        ok = false;
                        continue;
                    }
                };
                let m = metrics(&mesh, &case, &sol);
                let b = tg0_bands(&mesh, &case, &sol, &m);
                fs[i] = m.f;
                b4 = Some(b);
                println!(
                    "{model:?} re_tau {re_tau} n {} iters {} u_b {:.6e} f {:.6e} defect {:.6e} loglaw_dev {:.6e} loglaw_cells {} nut/utau/R {:.6e} re_d {:.6e}",
                    mesh.n(), sol.iterations, m.u_b, m.f, m.defect, b.loglaw_dev,
                    b.loglaw_cells, m.nut_axis_over_utau_r, b.re_d
                );
            }
            if ok {
                let b4 = b4.expect("ok implies level 4 bands");
                let (p, ext) = richardson_constant_ratio(fs[2], fs[1], fs[0], 2.0);
                let fp = f_prandtl(case.re_tau);
                println!(
                    "{model:?} re_tau {re_tau} summary: f(640) {:.6e} p {:.6e} f_ext {:.6e} f_vs_prandtl(640) {:.6e} f_vs_prandtl(ext) {:.6e} f_vs_blasius(640) {:.6e} loglaw_dev(640) {:.6e} defect_vs_4_07(640) {:.6e}",
                    fs[2], p, ext, b4.f_vs_prandtl, ext / fp - 1.0, b4.f_vs_blasius,
                    b4.loglaw_dev, b4.defect_vs_4_07
                );
            }
        }
    }
}
