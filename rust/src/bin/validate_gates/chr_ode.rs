// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.

//! **Gate CHR-ODE (SPEC-LIT §132.6, §132.8): RODAS3 and ROS3 on Robertson
//! and HIRES, and the constant-pressure reactor, live.**
//!
//! Ported from `tests/chem_rosenbrock.rs` and `tests/chem_reactor.rs` on
//! the ten-species TM-4513 fixture (§130.7, through its tracked key): both
//! built coefficient sets satisfy the order-3 conditions of §132.2 to 1e-15
//! (class A, with the orders and stage counts pinned discrete); Robertson
//! (132.11) and HIRES (132.12) at rtol 1e-8, atol 1e-20 land on CHR-00c's
//! transcription of the Mazzia & Magherini reference solutions within 1e-7
//! relative (class P), ROBER's mass at 1e-12 (class A); fixed steps on
//! `y' = -y` observe order 3 within 0.1 and one step h = 0.1 lands within
//! 1e-5 of exp(-0.1) with the embedded estimate not identically 0 (class P
//! and discrete); and the constant-pressure reactor of §132.4 lands on
//! §131's `(h, p)` equilibrium - 0.5 K, 1e-6 in Y - and on the supervisor's
//! oracle end state for both mixtures, closes its mass and enthalpy
//! balances, and ignites (class A and discrete), with the ignition delay
//! converging between rtol 1e-6 and 1e-9 and matching the oracle at 1e-3
//! (class A). The reference solutions are the tracked transcriptions
//! `reference/ivptestset2008/*.csv`, read exactly as the tests read them.
//!
//! Written from:
//!   ofgpu `SPEC-LIT.md` §132.6 (the gate table row), §132.2-§132.4 (the
//!     Rosenbrock sets, (132.11)/(132.12), the reactor), §132.8 (its
//!     refusals), §131.4 (the equilibrium it lands on)
//!   tests/chem_rosenbrock.rs - tests `ros_coefficients_satisfy_order_conditions`,
//!     `ros_robertson_testset`, `ros_hires_testset`, `ros_observed_order`;
//!   tests/chem_reactor.rs - tests `reactor_lands_on_equilibrium`,
//!     `reactor_delay_converges`, `reactor_delay_matches_oracle`;
//!     the tests stay as they are
//!   F. Mazzia & C. Magherini, Test Set for IVP Solvers, rel. 2.4 (2008),
//!     problems ROBER and HIRES - as the tracked transcriptions
//! No GPL-licensed source was consulted.

// The oracle's values are kept to the digits they printed.
#![allow(clippy::excessive_precision)]

use ofgpu::chem::reactor::{ConstPressureReactor, ReactorOptions, ReactorReport};
use ofgpu::chem::rosenbrock::{fd_jacobian, integrate, ros_step, Coefficients, Method, RosOptions};
use ofgpu::chem::{rosenbrock::RODAS3, rosenbrock::ROS3, ThermoSet, P_ATM};
use ofgpu::{Error, Result, Scalar};

use crate::{key, Checks};

use super::chr_fixture::{ch4_air, h2_air, mech, tm4513};

/// The gate's scope and registry name.
pub(crate) const GATE: &str = "SPEC-LIT 132.6 Gate CHR-ODE Rosenbrock and the reactor";

/// The `(component, t_end, y0, y_ref)` rows of a CHR-00c answer key: `#`
/// lines and the header skipped, each column parsed as printed -
/// `tests/chem_rosenbrock.rs`'s `reference_rows`, its parse failures
/// returned rather than panicked over.
fn reference_rows(csv: &str) -> Result<Vec<(usize, f64, f64, f64)>> {
    csv.lines()
        .filter(|l| !l.starts_with('#'))
        .skip(1)
        .map(|l| {
            let f: Vec<&str> = l.split(',').collect();
            let cell = |i: usize| -> Result<f64> {
                f[i].trim()
                    .parse::<f64>()
                    .map_err(|e| Error::Config(format!("reference column {i} of '{l}': {e}")))
            };
            let comp: usize = f[0]
                .trim()
                .parse()
                .map_err(|e| Error::Config(format!("reference component of '{l}': {e}")))?;
            Ok((comp, cell(1)?, cell(2)?, cell(3)?))
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
fn rober_f(y: &[f64], f: &mut [f64]) -> Result<()> {
    f[0] = -0.04 * y[0] + 1.0e4 * y[1] * y[2];
    f[1] = 0.04 * y[0] - 1.0e4 * y[1] * y[2] - 3.0e7 * y[1] * y[1];
    f[2] = 3.0e7 * y[1] * y[1];
    Ok(())
}

/// (132.12), HIRES.
fn hires_f(y: &[f64], f: &mut [f64]) -> Result<()> {
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
fn decay_f(y: &[f64], f: &mut [f64]) -> Result<()> {
    f[0] = -y[0];
    Ok(())
}

/// One built method's order conditions, order and stages - a discrete row
/// and eight class A rows per method.
fn order_rows(c: &mut Checks, coeff: &'static Coefficients) {
    c.require(
        &format!("Gate CHR-ODE: {} carries order 3 and embedded order 2 (SPEC-LIT 132.6)", coeff.name),
        coeff.order == 3 && coeff.embedded_order == 2,
    );
    c.require(
        &format!("Gate CHR-ODE: {} carries {} stages (SPEC-LIT 132.6)", coeff.name, coeff.stages),
        coeff.stages == if coeff.name == "RODAS3" { 4 } else { 3 },
    );
    let rb = order_residuals(coeff, &coeff.b);
    for (i, r) in rb.iter().enumerate() {
        c.check_a(
            &format!("Gate CHR-ODE: {} b, order condition {} residual within 1e-15 (SPEC-LIT 132.6)", coeff.name, i + 1),
            r.abs() as Scalar,
            1e-15,
        );
    }
    let rh = order_residuals(coeff, &coeff.b_hat);
    for (i, r) in rh[..2].iter().enumerate() {
        c.check_a(
            &format!("Gate CHR-ODE: {} b_hat, order condition {} residual within 1e-15 (SPEC-LIT 132.6)", coeff.name, i + 1),
            r.abs() as Scalar,
            1e-15,
        );
    }
}

/// The oracle's end state after 1 s at the defaults: (T, [Y_k]) for H2-air.
const ORACLE_END_H2: (f64, [f64; 10]) = (
    2693.827535802454,
    [
        3.0006981624e-03,
        1.7464580520e-02,
        2.1606791437e-01,
        1.5228277595e-02,
        2.6812734917e-03,
        4.4012740327e-04,
        0.0,
        0.0,
        0.0,
        7.4511712845e-01,
    ],
);

/// The oracle's end state after 1 s at the defaults: (T, [Y_k]) for CH4-air.
const ORACLE_END_CH4: (f64, [f64; 10]) = (
    2711.7999623963583,
    [
        1.2156675401e-03,
        2.5724001689e-02,
        1.0424789862e-01,
        1.1890015591e-02,
        3.3205775636e-03,
        2.8356439014e-04,
        3.9808346692e-02,
        8.8844925441e-02,
        1.0309182165e-15,
        7.2466500247e-01,
    ],
);

/// The oracle's ignition delays at rtol 1e-9 over 0.01 s: (H2-air, CH4-air).
const ORACLE_DELAY: (f64, f64) = (2.6552096919429e-3, 2.6042862928263e-3);

/// The 15-reaction reactor fixture of CHR-06. Every reaction is reversible,
/// the reverse from K_c (no REV, no FORD), so the ODE's rest point is the
/// Gibbs equilibrium of §131. Its rate numbers are placeholders, not any
/// paper's.
const REACTOR_FIXTURE_RX: &[&str] = &[
    "H + O2 <=> O + OH                 1.0E+14  -0.5  15000.0",
    "O + H2 <=> H + OH                 1.0E+05   2.5   6000.0",
    "OH + H2 <=> H + H2O               2.0E+08   1.5   3000.0",
    "H + OH + M <=> H2O + M            2.0E+22  -2.0      0.0",
    "  H2O/6.0/ H2/2.0/",
    "O + O + M <=> O2 + M              1.0E+17  -1.0      0.0",
    "H + H (+M) <=> H2 (+M)            1.0E+13   0.0      0.0",
    "  LOW / 5.0E+18 -1.0 0.0 /",
    "  H2O/5.0/",
    "CO + O (+M) <=> CO2 (+M)          2.0E+10   0.0   2500.0",
    "  LOW / 5.0E+14 0.0 3000.0 /",
    "  TROE / 0.6 100.0 1500.0 5000.0 /",
    "O + H2 (+N2) <=> H2O (+N2)        1.0E+12   0.0   1000.0",
    "  LOW / 1.0E+18 -1.0 0.0 /",
    "  TROE / 0.7 200.0 1000.0 /",
    "H + O (+M) <=> OH (+M)            5.0E+12   0.0      0.0",
    "  LOW / 4.0E+18 -1.0 0.0 /",
    "  SRI / 0.5 800.0 200.0 1.2 0.1 /",
    "CO + OH <=> CO2 + H               4.0E+06   1.5   -500.0",
    "OH + OH <=> O + H2O               3.0E+04   2.4  -2000.0",
    "  DUP",
    "OH + OH <=> O + H2O               1.0E+12   0.0   1000.0",
    "  DUP",
    "H2 + O2 <=> OH + OH               1.7E+13   0.0  47780.0",
    "CH4 + O2 <=> CO + H2O + H2        1.0E+12   0.0  30000.0",
    "CH4 + O <=> CO + 2 H2             1.0E+13   0.0   8000.0",
];

/// rel(x, t) = |x - t| / |t|, the class A comparison.
fn rel(x: f64, t: f64) -> f64 {
    (x - t).abs() / t.abs()
}

/// The reactor run's gates, shared by both mixtures - one row each: the end
/// state against §131's `(h, p)` equilibrium (0.5 K, 1e-6 in Y), against
/// the oracle end state when one is given, mass conservation at the clips,
/// enthalpy conservation, and an ignition delay.
fn eq_gates(
    c: &mut Checks,
    set: &ThermoSet,
    rep: &ReactorReport,
    y0: &[f64],
    h0: f64,
    opts: &ReactorOptions,
    oracle: Option<(f64, &[f64])>,
) -> Result<()> {
    let eq = ofgpu::chem::equilibrate_hp(set, h0, P_ATM, y0, &ofgpu::chem::EqOptions::default())?;
    let eq_y = eq.mass_fractions(set);
    let d_eq = rep
        .y
        .iter()
        .zip(&eq_y)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f64, f64::max);
    c.check_a(
        "Gate CHR-ODE: the run lands on the (h, p) equilibrium within 0.5 K (SPEC-LIT 132.6)",
        (rep.temperature - eq.t).abs() as Scalar,
        0.5,
    );
    c.check_a(
        "Gate CHR-ODE: the run's Y lands on the equilibrium's within 1e-6 (SPEC-LIT 132.6)",
        d_eq as Scalar,
        1e-6,
    );
    if let Some((t_or, y_or)) = oracle {
        let d_or = rep
            .y
            .iter()
            .zip(y_or)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f64, f64::max);
        c.check_a(
            "Gate CHR-ODE: the run's T lands on the oracle's end state within 0.5 K (SPEC-LIT 132.6)",
            (rep.temperature - t_or).abs() as Scalar,
            0.5,
        );
        c.check_a(
            "Gate CHR-ODE: the run's Y lands on the oracle's end state within 1e-6 (SPEC-LIT 132.6)",
            d_or as Scalar,
            1e-6,
        );
    }
    let d_sum = (rep.y.iter().sum::<f64>() - y0.iter().sum::<f64>()).abs();
    let h_end = set.h_mass(rep.temperature, &rep.y)?;
    c.check_a(
        "Gate CHR-ODE: the run conserves its mass at the clips (SPEC-LIT 132.6)",
        d_sum as Scalar,
        (1e-12 + rep.clipped as f64 * opts.atol_y) as Scalar,
    );
    c.check_a(
        "Gate CHR-ODE: the run conserves its enthalpy within 1e-6 of |h0| (SPEC-LIT 132.6)",
        ((h_end - h0).abs()) as Scalar,
        (1e-6 * h0.abs()) as Scalar,
    );
    c.require(
        "Gate CHR-ODE: the mixture ignites - a delay is reported (SPEC-LIT 132.6)",
        rep.ignition_delay.is_some(),
    );
    Ok(())
}

/// **Gate CHR-ODE** (SPEC-LIT §132.6, §132.8): the Rosenbrock integrator on
/// the test set and the constant-pressure reactor, live.
pub(crate) fn check_chr_ode(c: &mut Checks) -> Result<()> {
    // ---- the order conditions of 132.2 ---------------------------------------
    order_rows(c, &RODAS3);
    order_rows(c, &ROS3);

    // ---- Robertson (132.11) and HIRES (132.12) against the test set ----------
    let (rober_line, rober_text) = key::load_text("ivptestset2008-rober")?;
    c.note(&rober_line);
    let rober = reference_rows(&rober_text)?;
    c.require(
        "Gate CHR-ODE: the ROBER key carries three components (SPEC-LIT 132.6)",
        rober.len() == 3,
    );
    let (hires_line, hires_text) = key::load_text("ivptestset2008-hires")?;
    c.note(&hires_line);
    let hires = reference_rows(&hires_text)?;
    c.require(
        "Gate CHR-ODE: the HIRES key carries eight components (SPEC-LIT 132.6)",
        hires.len() == 8,
    );
    for (name, rows, rhs) in [
        ("ROBER", &rober, rober_f as fn(&[f64], &mut [f64]) -> Result<()>),
        ("HIRES", &hires, hires_f as fn(&[f64], &mut [f64]) -> Result<()>),
    ] {
        let t_end = rows[0].1;
        let y0: Vec<f64> = rows.iter().map(|r| r.2).collect();
        for method in [Method::Rodas3, Method::Ros3] {
            let mut f = rhs;
            let mut y = y0.clone();
            let mut opts = RosOptions::new(y.len(), 1e-8, 1e-20);
            opts.method = method;
            opts.h0 = 1e-10;
            integrate(&mut f, &mut y, 0.0, t_end, &opts)?;
            let mut worst = 0.0f64;
            for (k, r) in rows.iter().enumerate() {
                worst = worst.max((y[k] - r.3).abs() / r.3.abs());
            }
            c.check_p(
                &format!("Gate CHR-ODE: {name} on {} worst component error within 1e-7 of the test set's reference (SPEC-LIT 132.6)", method.name()),
                worst as Scalar,
                1e-7,
            );
            if name == "ROBER" {
                let mass: f64 = y[0] + y[1] + y[2];
                c.check_a(
                    &format!("Gate CHR-ODE: {name} on {} keeps y1 + y2 + y3 = 1 to 1e-12 (SPEC-LIT 132.6)", method.name()),
                    ((mass - 1.0).abs()) as Scalar,
                    1e-12,
                );
            }
        }
    }

    // ---- the observed order on y' = -y ---------------------------------------
    for method in [Method::Rodas3, Method::Ros3] {
        let coeff = method.coefficients();
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
                f(&y, &mut f_y)?;
                fd_jacobian(&mut f, &y, &f_y, &atol, &mut jac)?;
                ros_step(coeff, &mut f, &y, h, &jac, &mut y_new, &mut est)?;
                y.copy_from_slice(&y_new);
            }
            errs[i] = (y[0] - (-1.0f64).exp()).abs();
        }
        let p = (errs[0] / errs[1]).log2();
        c.check_p(
            &format!("Gate CHR-ODE: {} observes order 3 within 0.1 on y' = -y (SPEC-LIT 132.6)", method.name()),
            ((p - 3.0).abs()) as Scalar,
            0.1,
        );
        // One step h = 0.1 from y = 1: inside 1e-5 of exp(-0.1), and the
        // embedded estimate is not identically 0.
        let mut f = decay_f;
        let f_y = vec![-1.0f64];
        let mut jac = vec![0.0f64];
        fd_jacobian(&mut f, &[1.0], &f_y, &[1e-3], &mut jac)?;
        let mut y_new = vec![0.0f64];
        let mut est = vec![0.0f64];
        ros_step(coeff, &mut f, &[1.0], 0.1, &jac, &mut y_new, &mut est)?;
        let exact = (-0.1f64).exp();
        c.check_p(
            &format!("Gate CHR-ODE: {} one step h = 0.1 lands within 1e-5 of exp(-0.1) (SPEC-LIT 132.6)", method.name()),
            ((y_new[0] - exact).abs()) as Scalar,
            1e-5,
        );
        c.require(
            &format!("Gate CHR-ODE: {} embedded estimate is not identically 0 (SPEC-LIT 132.6)", method.name()),
            est[0] != 0.0,
        );
    }

    // ---- the constant-pressure reactor on the TM-4513 fixture -----------------
    let fset = tm4513(c)?;
    let m = mech(&fset, REACTOR_FIXTURE_RX)?;
    let set = m.thermo();
    let r = ConstPressureReactor::new(&m, P_ATM)?;
    let opts = ReactorOptions::default();
    let y_h2 = h2_air(set);
    let h_h2 = set.h_mass(1000.0, &y_h2)?;
    let rep = r.advance(&y_h2, 1000.0, 1.0, &opts)?;
    let (t_or, y_or) = ORACLE_END_H2;
    eq_gates(c, set, &rep, &y_h2, h_h2, &opts, Some((t_or, &y_or)))?;
    let y_ch4 = ch4_air(set);
    let h_ch4 = set.h_mass(1400.0, &y_ch4)?;
    let rep = r.advance(&y_ch4, 1400.0, 1.0, &opts)?;
    let (t_or, y_or) = ORACLE_END_CH4;
    eq_gates(c, set, &rep, &y_ch4, h_ch4, &opts, Some((t_or, &y_or)))?;

    // ---- the ignition delay converges between rtol 1e-6 and 1e-9 --------------
    let o6 = ReactorOptions {
        rtol: 1e-6,
        ..ReactorOptions::default()
    };
    let o9 = ReactorOptions {
        rtol: 1e-9,
        ..ReactorOptions::default()
    };
    let delay_of = |rep: &ReactorReport, what: &str| -> Result<f64> {
        rep.ignition_delay
            .ok_or_else(|| Error::Config(format!("Gate CHR-ODE: {what} reported no ignition delay")))
    };
    let d6 = r.advance(&y_h2, 1000.0, 0.01, &o6)?;
    let d9 = r.advance(&y_h2, 1000.0, 0.01, &o9)?;
    c.check_a(
        "Gate CHR-ODE: H2-air delay at rtol 1e-6 within 0.5 % of its rtol 1e-9 value (SPEC-LIT 132.6)",
        rel(delay_of(&d6, "H2-air at rtol 1e-6")?, delay_of(&d9, "H2-air at rtol 1e-9")?) as Scalar,
        5e-3,
    );
    let d6 = r.advance(&y_ch4, 1400.0, 0.01, &o6)?;
    let d9 = r.advance(&y_ch4, 1400.0, 0.01, &o9)?;
    c.check_a(
        "Gate CHR-ODE: CH4-air delay at rtol 1e-6 within 0.5 % of its rtol 1e-9 value (SPEC-LIT 132.6)",
        rel(delay_of(&d6, "CH4-air at rtol 1e-6")?, delay_of(&d9, "CH4-air at rtol 1e-9")?) as Scalar,
        5e-3,
    );

    // ---- the ignition delay against the oracle, rtol 1e-9 over 0.01 s ---------
    let rep = r.advance(&y_h2, 1000.0, 0.01, &o9)?;
    c.check_a(
        "Gate CHR-ODE: H2-air ignition delay within 0.1 % of the oracle's (SPEC-LIT 132.6)",
        rel(delay_of(&rep, "H2-air")?, ORACLE_DELAY.0) as Scalar,
        1e-3,
    );
    let rep = r.advance(&y_ch4, 1400.0, 0.01, &o9)?;
    c.check_a(
        "Gate CHR-ODE: CH4-air ignition delay within 0.1 % of the oracle's (SPEC-LIT 132.6)",
        rel(delay_of(&rep, "CH4-air")?, ORACLE_DELAY.1) as Scalar,
        1e-3,
    );
    Ok(())
}


