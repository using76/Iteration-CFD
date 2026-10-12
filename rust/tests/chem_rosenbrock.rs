// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.
// Provenance: see PROVENANCE.md. No GPL-licensed source was consulted.

//! CHR-05's gates: the stiff Rosenbrock integrator (SPEC-LIT §132,
//! `src/chem/rosenbrock.rs`) — the coefficient sets' order conditions of
//! §132.2, Robertson (132.11) and HIRES (132.12) against CHR-00c's
//! transcription of the Mazzia & Magherini test-set reference solutions at
//! rtol 1e-8, the observed order on `y' = -y`, and the §132.7 refusals
//! (R1's singular LU, R5's validation, R6's method names). The reference
//! values are tracked answer keys, `reference/ivptestset2008/*.csv`.
//!
//! An integration test, so it runs in both builds: it links the normal
//! library, which is f64 in every build.
//! No GPL-licensed source was consulted.

// The oracle's values are kept to the digits they printed.
#![allow(clippy::excessive_precision)]

use ofgpu::Error;
use ofgpu::chem::rosenbrock::{
    RODAS3, ROS3, Coefficients, Method, RosOptions, fd_jacobian, integrate, lu_factor, lu_solve,
    ros_step,
};

/// The `(component, t_end, y0, y_ref)` rows of a CHR-00c answer key: `#`
/// lines and the header skipped, each column parsed as printed.
fn reference_rows(csv: &str) -> Vec<(usize, f64, f64, f64)> {
    csv.lines()
        .filter(|l| !l.starts_with('#'))
        .skip(1)
        .map(|l| {
            let c: Vec<&str> = l.split(',').collect();
            (
                c[0].trim().parse().expect("component"),
                c[1].trim().parse().expect("t_end"),
                c[2].trim().parse().expect("y0"),
                c[3].trim().parse().expect("y_ref"),
            )
        })
        .collect()
}

/// The four order-3 conditions of SPEC-LIT §132.2 for weights `w`, as
/// signed residuals. With `beta_ij = alpha_ij + gamma_ij`,
/// `beta'_i = sum_{j<i} beta_ij` and `alpha_i = sum_{j<i} alpha_ij`:
/// (1) `sum w = 1`; (2) `sum w_i beta'_i = 1/2 - gamma`;
/// (3) `sum w_i alpha_i^2 = 1/3`;
/// (4) `sum_i w_i sum_j beta_ij beta'_j = 1/6 - gamma + gamma^2`.
/// The embedded pair satisfies (1)-(2) only.
fn order_residuals(c: &Coefficients, w: &[f64; 4]) -> [f64; 4] {
    let s = c.stages;
    let mut beta = [[0.0f64; 4]; 4];
    for i in 0..s {
        for j in 0..i {
            beta[i][j] = c.alpha[i][j] + c.gamma_ij[i][j];
        }
    }
    let bp: Vec<f64> = (0..s).map(|i| (0..i).map(|j| beta[i][j]).sum()).collect();
    let ca: Vec<f64> = (0..s).map(|i| (0..i).map(|j| c.alpha[i][j]).sum()).collect();
    let r1: f64 = w[..s].iter().sum::<f64>() - 1.0;
    let r2: f64 = (0..s).map(|i| w[i] * bp[i]).sum::<f64>() - (0.5 - c.gamma);
    let r3: f64 = (0..s).map(|i| w[i] * ca[i] * ca[i]).sum::<f64>() - 1.0 / 3.0;
    let r4: f64 = (0..s)
        .map(|i| w[i] * (0..s).map(|j| beta[i][j] * bp[j]).sum::<f64>())
        .sum::<f64>()
        - (1.0 / 6.0 - c.gamma + c.gamma * c.gamma);
    [r1, r2, r3, r4]
}

/// (132.11), Robertson's problem.
fn rober_f(y: &[f64], f: &mut [f64]) -> Result<(), Error> {
    f[0] = -0.04 * y[0] + 1.0e4 * y[1] * y[2];
    f[1] = 0.04 * y[0] - 1.0e4 * y[1] * y[2] - 3.0e7 * y[1] * y[1];
    f[2] = 3.0e7 * y[1] * y[1];
    Ok(())
}

/// (132.12), HIRES.
fn hires_f(y: &[f64], f: &mut [f64]) -> Result<(), Error> {
    f[0] = -1.71 * y[0] + 0.43 * y[1] + 8.32 * y[2] + 0.0007;
    f[1] = 1.71 * y[0] - 8.75 * y[1];
    f[2] = -10.03 * y[2] + 0.43 * y[3] + 0.035 * y[4];
    f[3] = 8.32 * y[1] + 1.71 * y[2] - 1.12 * y[3];
    f[4] = -1.745 * y[4] + 0.43 * y[5] + 0.43 * y[6];
    f[5] = -280.0 * y[5] * y[7] + 0.69 * y[3] + 1.71 * y[4] - 0.43 * y[5] + 0.69 * y[6];
    f[6] = 280.0 * y[5] * y[7] - 1.81 * y[6];
    f[7] = -280.0 * y[5] * y[7] + 1.81 * y[6];
    Ok(())
}

/// `y' = -y` on one component.
fn decay_f(y: &[f64], f: &mut [f64]) -> Result<(), Error> {
    f[0] = -y[0];
    Ok(())
}

/// §132.6 row 1 (class D): both built sets satisfy the conditions of
/// §132.2, to 1e-15 absolute, so the tree proves its own coefficients.
#[test]
fn ros_coefficients_satisfy_order_conditions() {
    for c in [&RODAS3, &ROS3] {
        assert_eq!(c.order, 3, "{}: order", c.name);
        assert_eq!(c.embedded_order, 2, "{}: embedded order", c.name);
        assert_eq!(c.stages, if c.name == "RODAS3" { 4 } else { 3 });
        let rb = order_residuals(c, &c.b);
        for (i, r) in rb.iter().enumerate() {
            assert!(
                r.abs() <= 1e-15,
                "{}: b, condition {} residual {:e}",
                c.name,
                i + 1,
                r
            );
        }
        let rh = order_residuals(c, &c.b_hat);
        for (i, r) in rh[..2].iter().enumerate() {
            assert!(
                r.abs() <= 1e-15,
                "{}: b_hat, condition {} residual {:e}",
                c.name,
                i + 1,
                r
            );
        }
        println!(
            "{}: b residuals {:e} {:e} {:e} {:e}, b_hat residuals {:e} {:e}",
            c.name, rb[0], rb[1], rb[2], rb[3], rh[0], rh[1]
        );
    }
}

/// §132.6 row 2 (class P): Robertson (132.11) at rtol 1e-8, atol 1e-20 in
/// every component, against CHR-00c's answer key at t = 1e11, with
/// `y1 + y2 + y3 = 1` to 1e-12. Oracle: RODAS3 2.74e-8 (4519 accepted),
/// ROS3 8.79e-8 (24883 accepted).
#[test]
fn ros_robertson_testset() {
    let rows = reference_rows(include_str!("../../reference/ivptestset2008/rober.csv"));
    assert_eq!(rows.len(), 3, "three components");
    let t_end = rows[0].1;
    let y0: Vec<f64> = rows.iter().map(|r| r.2).collect();
    for method in [Method::Rodas3, Method::Ros3] {
        let mut f = rober_f;
        let mut y = y0.clone();
        let mut opts = RosOptions::new(y.len(), 1e-8, 1e-20);
        opts.method = method;
        opts.h0 = 1e-10;
        let stats = integrate(&mut f, &mut y, 0.0, t_end, &opts)
            .unwrap_or_else(|e| panic!("ROBER {}: integrate failed: {}", method.name(), e));
        let mut worst = 0.0f64;
        for (k, r) in rows.iter().enumerate() {
            worst = worst.max((y[k] - r.3).abs() / r.3.abs());
        }
        println!("ROBER {}: max rel err = {:e} (gate 1e-7), stats {:?}", method.name(), worst, stats);
        assert!(
            worst <= 1e-7,
            "ROBER {}: max rel err {:e} exceeds 1e-7",
            method.name(),
            worst
        );
        let mass: f64 = y[0] + y[1] + y[2];
        assert!(
            (mass - 1.0).abs() <= 1e-12,
            "ROBER {}: y1 + y2 + y3 = {:.15}",
            method.name(),
            mass
        );
    }
}

/// §132.6 row 3 (class P): HIRES (132.12) at rtol 1e-8, atol 1e-20 in
/// every component, against CHR-00c's answer key at t = 321.8122. Oracle:
/// RODAS3 3.09e-9 (12297 accepted), ROS3 3.08e-9 (16105 accepted).
#[test]
fn ros_hires_testset() {
    let rows = reference_rows(include_str!("../../reference/ivptestset2008/hires.csv"));
    assert_eq!(rows.len(), 8, "eight components");
    let t_end = rows[0].1;
    let y0: Vec<f64> = rows.iter().map(|r| r.2).collect();
    for method in [Method::Rodas3, Method::Ros3] {
        let mut f = hires_f;
        let mut y = y0.clone();
        let mut opts = RosOptions::new(y.len(), 1e-8, 1e-20);
        opts.method = method;
        opts.h0 = 1e-10;
        let stats = integrate(&mut f, &mut y, 0.0, t_end, &opts)
            .unwrap_or_else(|e| panic!("HIRES {}: integrate failed: {}", method.name(), e));
        let mut worst = 0.0f64;
        for (k, r) in rows.iter().enumerate() {
            worst = worst.max((y[k] - r.3).abs() / r.3.abs());
        }
        println!("HIRES {}: max rel err = {:e} (gate 1e-7), stats {:?}", method.name(), worst, stats);
        assert!(
            worst <= 1e-7,
            "HIRES {}: max rel err {:e} exceeds 1e-7",
            method.name(),
            worst
        );
    }
}

/// §132.6 row 4 (class P): fixed steps on `y' = -y` over [0, 1] with 40
/// and 80 steps, `p = log2(e_40 / e_80)`, within 0.1 of the method order 3.
/// Oracle: RODAS3 e_40 = 1.1857e-7, e_80 = 1.4895e-8, p = 2.9928; ROS3
/// 1.4669e-7, 1.8470e-8, p = 2.9894. Also one step h = 0.1 from y = 1.
#[test]
fn ros_observed_order() {
    for method in [Method::Rodas3, Method::Ros3] {
        let c = method.coefficients();
        let mut errs = [0.0f64; 2];
        for (i, &n) in [40usize, 80].iter().enumerate() {
            let h = 1.0 / n as f64;
            let atol = vec![1e-3];
            let mut f = decay_f;
            let mut y = vec![1.0f64];
            let mut f_y = vec![0.0f64];
            let mut jac = vec![0.0f64];
            let mut y_new = vec![0.0f64];
            let mut est = vec![0.0f64];
            for _ in 0..n {
                f(&y, &mut f_y).unwrap();
                fd_jacobian(&mut f, &y, &f_y, &atol, &mut jac).unwrap();
                ros_step(c, &mut f, &y, h, &jac, &mut y_new, &mut est).unwrap();
                y.copy_from_slice(&y_new);
            }
            errs[i] = (y[0] - (-1.0f64).exp()).abs();
        }
        let p = (errs[0] / errs[1]).log2();
        println!(
            "{}: e_40 = {:e}, e_80 = {:e}, p = {:.4} (oracle p 2.9928 / 2.9894)",
            method.name(),
            errs[0],
            errs[1],
            p
        );
        assert!(
            (p - 3.0).abs() <= 0.1,
            "{}: observed order p = {:.4} off 3 by more than 0.1",
            method.name(),
            p
        );
        // One step h = 0.1 from y = 1: inside 1e-5 of exp(-0.1), and the
        // embedded estimate is not identically 0.
        let c = method.coefficients();
        let mut f = decay_f;
        let f_y = vec![-1.0f64];
        let mut jac = vec![0.0f64];
        fd_jacobian(&mut f, &[1.0], &f_y, &[1e-3], &mut jac).unwrap();
        let mut y_new = vec![0.0f64];
        let mut est = vec![0.0f64];
        ros_step(c, &mut f, &[1.0], 0.1, &jac, &mut y_new, &mut est).unwrap();
        let exact = (-0.1f64).exp();
        println!(
            "{}: one step h = 0.1: err = {:e}, est = {:e}",
            method.name(),
            (y_new[0] - exact).abs(),
            est[0]
        );
        assert!((y_new[0] - exact).abs() < 1e-5, "{}: one-step error {:e}", method.name(), (y_new[0] - exact).abs());
        assert!(est[0] != 0.0, "{}: embedded estimate is 0", method.name());
    }
}

/// The §132.7 refusals: R6's names, R5's validation, the pass-through of
/// `f`'s error, and R1's singular LU with the pivoted row swap.
#[test]
fn ros_refusals() {
    // R6: "ros4" is named and not built; anything unrecognised names the
    // set; "ros3" parses.
    let e = Method::from_name("ros4").unwrap_err();
    assert!(matches!(e, Error::Refused(_)));
    assert!(e.to_string().contains("ros4"));
    let e = Method::from_name("rk4").unwrap_err();
    assert!(e.to_string().contains("rodas3, ros3"));
    assert_eq!(Method::from_name("ros3").unwrap(), Method::Ros3);

    let mut lin = |y: &[f64], f: &mut [f64]| -> Result<(), Error> {
        f[0] = -y[0];
        Ok(())
    };
    // atol of the wrong length.
    let mut y = vec![1.0f64];
    let mut opts = RosOptions::new(1, 1e-6, 1e-9);
    opts.atol = vec![1e-9, 1e-9];
    assert!(integrate(&mut lin, &mut y, 0.0, 1.0, &opts).is_err());
    // An atol component that is 0.
    let mut opts = RosOptions::new(1, 1e-6, 1e-9);
    opts.atol = vec![0.0];
    assert!(integrate(&mut lin, &mut y, 0.0, 1.0, &opts).is_err());
    // t1 < t0.
    let opts = RosOptions::new(1, 1e-6, 1e-9);
    assert!(integrate(&mut lin, &mut y, 0.0, -1.0, &opts).is_err());
    // t1 == t0 returns at once, with zero steps and y unchanged.
    let opts = RosOptions::new(1, 1e-6, 1e-9);
    let stats = integrate(&mut lin, &mut y, 0.0, 0.0, &opts).unwrap();
    assert_eq!(stats.accepted, 0);
    assert_eq!(y[0], 1.0);
    // A rejection of the very first step divides h by 10, below h_min of
    // 0.05 (§132.3): refused, naming h_min, never forced.
    let mut y = vec![1.0f64];
    let mut opts = RosOptions::new(1, 1e-30, 1e-30);
    opts.h0 = 0.1;
    opts.h_min = 0.05;
    let e = integrate(&mut lin, &mut y, 0.0, 1.0, &opts).unwrap_err();
    assert!(e.to_string().contains("h_min"), "{}", e);
    // A step capped at 1e-3 over [0, 1] needs more than 10 attempts.
    let mut y = vec![1.0f64];
    let mut opts = RosOptions::new(1, 1e-6, 1e-9);
    opts.h_max = 1e-3;
    opts.max_steps = 10;
    let e = integrate(&mut lin, &mut y, 0.0, 1.0, &opts).unwrap_err();
    assert!(e.to_string().contains("max_steps"), "{}", e);
    // `f`'s own error passes through unchanged.
    let mut boom = |_y: &[f64], _f: &mut [f64]| -> Result<(), Error> {
        Err(Error::Config("boom".to_string()))
    };
    let mut y = vec![1.0f64];
    let opts = RosOptions::new(1, 1e-6, 1e-9);
    let e = integrate(&mut boom, &mut y, 0.0, 1.0, &opts).unwrap_err();
    assert!(e.to_string().contains("boom"), "{}", e);
    // R1: an all-zero matrix has a 0 pivot and is singular.
    let mut a = vec![0.0f64; 4];
    let mut piv = vec![0usize; 2];
    assert!(!lu_factor(&mut a, 2, &mut piv));
    // [[0, 2], [3, 4]] needs the row swap; x = (2, 1) exactly.
    let mut a = vec![0.0, 2.0, 3.0, 4.0f64];
    let mut piv = vec![0usize; 2];
    assert!(lu_factor(&mut a, 2, &mut piv));
    let mut b = vec![2.0, 10.0f64];
    lu_solve(&a, 2, &piv, &mut b);
    assert_eq!(b, vec![2.0, 1.0]);
}
