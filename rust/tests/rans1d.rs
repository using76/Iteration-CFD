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
