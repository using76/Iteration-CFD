// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.
// Provenance: see PROVENANCE.md. No GPL-licensed source was consulted.

// The DST-02 psi gates, written once and included both by
// src/dust/tests_psi.rs (as dust::psi::tests) and by tests/dust_psi.rs, so
// both builds run the same nine cases. Host f64 only.
// No GPL-licensed source was consulted.

use ofgpu::dust::psi::{
    documentation_table, format_table, implied_speed_ratio, reaction_time, AdiabaticCurve,
    FlameCurve, PsiOptions, PsiProblem, PublishedCurve,
};
use ofgpu::dust::thermo::{EquivalentGas, FormulaInputs, PrescribedInputs, tm4513_gas_set};
use ofgpu::error::Error;

/// |a/b - 1|.
fn rel(a: f64, b: f64) -> f64 {
    (a / b - 1.0).abs()
}

/// The Leeds eigenproblem (H20): the published closure at sigma = 5.06,
/// theta_u = 20000/328, n = 1 (SPEC-LIT §153.6).
fn leeds<'c>(curve: &'c PublishedCurve) -> PsiProblem<'c> {
    PsiProblem { curve, theta_u: 20000.0 / 328.0, n: 1.0 }
}

/// The REMBE eigenproblem (H22): the published closure at sigma = 7.04,
/// theta_u = 20000/273, n = 1 (SPEC-LIT §153.6).
fn rembe<'c>(curve: &'c PublishedCurve) -> PsiProblem<'c> {
    PsiProblem { curve, theta_u: 20000.0 / 273.0, n: 1.0 }
}

/// 1: the eigenvalue converges with the march, and both cases land on the
/// oracle's Psi, Lambda/Lambda_ZFK and Ze at 40 000 steps (SPEC-LIT §153.5).
#[test]
fn eigen_converges() -> Result<(), Error> {
    let case = |name: &str, p: &PsiProblem, psi_ref: f64, rzfk_ref: f64, ze_ref: f64| {
        let small = p.eigen(&PsiOptions { n_steps: 20_000, ..Default::default() })?;
        let big = p.eigen(&PsiOptions { n_steps: 40_000, ..Default::default() })?;
        let rl = rel(small.lambda, big.lambda);
        println!(
            "{name}: rel(lambda_20000, lambda_40000) = {rl:.3e}, psi(40000) = {:e}, \
             ratio_zfk = {}, ze = {}, bisections = {}",
            big.psi, big.ratio_zfk, big.ze, big.bisections
        );
        assert!(rl <= 1e-10, "{name}: rel(lambda_20000, lambda_40000) = {rl}");
        assert!(
            rel(big.psi, psi_ref) <= 1e-9,
            "{name}: psi = {:e} is not 1e-9 of {psi_ref:e}",
            big.psi
        );
        assert!(
            (big.ratio_zfk - rzfk_ref).abs() <= 1e-8,
            "{name}: ratio_zfk = {} is not 1e-8 of {rzfk_ref}",
            big.ratio_zfk
        );
        assert!(
            (big.ze - ze_ref).abs() <= 1e-12,
            "{name}: ze = {} is not 1e-12 of {ze_ref}",
            big.ze
        );
        assert!(big.bisections <= 200, "{name}: {} bisections", big.bisections);
        Ok::<(), Error>(())
    };
    let pc5 = PublishedCurve::new(5.06)?;
    case("Leeds", &leeds(&pc5), 1.544527979224942e-4, 0.7819266642497452, 9.668990907909674)?;
    let pc7 = PublishedCurve::new(7.04)?;
    case("REMBE", &rembe(&pc7), 3.186768215956455e-4, 0.7721669671167958, 8.928098416734780)?;
    Ok(())
}

/// 2: (153.4) holds to 1e-10 on the converged march, for both cases at the
/// default options and for REMBE at n = 0.5 and n = 1.7 (SPEC-LIT §153.4).
#[test]
fn integral_identity_holds() -> Result<(), Error> {
    let pc7 = PublishedCurve::new(7.04)?;
    let pc5 = PublishedCurve::new(5.06)?;
    let dflt = PsiOptions::default();
    let p = leeds(&pc5);
    let s = p.eigen(&dflt)?;
    println!("Leeds identity_residual = {:e}", s.identity_residual);
    assert!(s.identity_residual.abs() <= 1e-10, "Leeds: {:e}", s.identity_residual);
    let p = rembe(&pc7);
    let s = p.eigen(&dflt)?;
    println!("REMBE identity_residual = {:e}", s.identity_residual);
    assert!(s.identity_residual.abs() <= 1e-10, "REMBE: {:e}", s.identity_residual);
    for (n, psi_ref) in [(0.5, 2.041280308848866e-4), (1.7, 5.959546468606348e-4)] {
        let opt = PsiOptions { n_steps: 20_000, ..Default::default() };
        let p = PsiProblem { curve: &pc7, theta_u: 20000.0 / 273.0, n };
        let s = p.eigen(&opt)?;
        println!("REMBE n = {n}: psi = {:e}, identity_residual = {:e}", s.psi, s.identity_residual);
        assert!(
            rel(s.psi, psi_ref) <= 1e-9,
            "REMBE n = {n}: psi = {:e} is not 1e-9 of {psi_ref:e}",
            s.psi
        );
        assert!(
            s.identity_residual.abs() <= 1e-10,
            "REMBE n = {n}: {:e}",
            s.identity_residual
        );
    }
    Ok(())
}

/// 3: the eigenvalue does not depend on c_cut below 1e-6, the classical
/// cold-boundary difficulty of §153.2, at c_cut in {1e-8, 1e-10, 1e-12}.
#[test]
fn cut_insensitive() -> Result<(), Error> {
    let pc5 = PublishedCurve::new(5.06)?;
    let pc7 = PublishedCurve::new(7.04)?;
    for (name, p) in [("Leeds", leeds(&pc5)), ("REMBE", rembe(&pc7))] {
        let mut lams = Vec::new();
        for &cut in &[1e-8, 1e-10, 1e-12] {
            let opt = PsiOptions { n_steps: 20_000, c_cut: cut, ..Default::default() };
            lams.push(p.eigen(&opt)?.lambda);
        }
        let worst = rel(lams[0], lams[1])
            .max(rel(lams[0], lams[2]))
            .max(rel(lams[1], lams[2]));
        println!("{name}: max pairwise rel(lambda) over c_cut = {worst:.3e}");
        assert!(worst <= 1e-9, "{name}: {worst:.3e}");
    }
    Ok(())
}

/// 4: Lambda/Lambda_ZFK climbs toward 1 as Ze grows at fixed sigma, the
/// asymptote of (153.5), with the oracle's four ratios.
#[test]
fn zfk_limit_monotone() -> Result<(), Error> {
    let refs = [0.7915130837, 0.8831637678, 0.9376024844, 0.9676609830];
    let sigma = 7.04;
    let pc = PublishedCurve::new(sigma)?;
    let opt = PsiOptions { n_steps: 20_000, ..Default::default() };
    let mut gap = Vec::new();
    for (ze, r_ref) in [10.0, 20.0, 40.0, 80.0].into_iter().zip(refs) {
        let theta_u = ze * sigma * sigma / (sigma - 1.0);
        let p = PsiProblem { curve: &pc, theta_u, n: 1.0 };
        let s = p.eigen(&opt)?;
        println!(
            "Ze = {ze}: ratio_zfk = {}, ze = {}, identity_residual = {:e}",
            s.ratio_zfk, s.ze, s.identity_residual
        );
        assert!(
            (s.ze - ze).abs() <= 1e-9 * ze,
            "Ze = {ze}: ze = {}",
            s.ze
        );
        assert!(
            s.ratio_zfk > 0.0 && s.ratio_zfk < 1.0,
            "Ze = {ze}: ratio_zfk = {}",
            s.ratio_zfk
        );
        assert!(
            (s.ratio_zfk - r_ref).abs() <= 1e-8,
            "Ze = {ze}: ratio_zfk = {} is not 1e-8 of {r_ref}",
            s.ratio_zfk
        );
        assert!(
            s.identity_residual.abs() <= 1e-10,
            "Ze = {ze}: {:e}",
            s.identity_residual
        );
        gap.push(1.0 - s.ratio_zfk);
    }
    for w in gap.windows(2) {
        assert!(w[1] < w[0], "|1 - ratio| not strictly decreasing: {} then {}", w[0], w[1]);
    }
    assert!(gap[3] < 0.05, "|1 - ratio| at Ze 80 = {}", gap[3]);
    Ok(())
}

/// 5: the converged trajectory's end slope sits on the saddle quadratic
/// a^2 + a = Lambda k g(1) of §153.2, seeded and measured (§153.5).
#[test]
fn saddle_slope() -> Result<(), Error> {
    let pc5 = PublishedCurve::new(5.06)?;
    let pc7 = PublishedCurve::new(7.04)?;
    for (name, s) in
        [("Leeds", leeds(&pc5).eigen(&PsiOptions::default())?), ("REMBE", rembe(&pc7).eigen(&PsiOptions::default())?)]
    {
        let q = s.lambda * s.k * s.g1;
        let seed = (s.seed_slope * s.seed_slope + s.seed_slope - q).abs();
        let measured = (s.end_slope * s.end_slope + s.end_slope - q).abs();
        println!(
            "{name}: q = {:e}, seed a = {:e} residual = {seed:.3e}, a_m = {:e} \
             residual = {measured:.3e}",
            q, s.seed_slope, s.end_slope
        );
        assert!(seed <= 1e-12 * q, "{name}: seed residual {seed:.3e} > 1e-12 q");
        assert!(measured <= 1e-6 * q, "{name}: measured residual {measured:.3e} > 1e-6 q");
    }
    Ok(())
}

/// 6: the prescribed Leeds gas's adiabatic line is the closed form
/// t = 1 + c(1500/328 - 1), r = (W(c)/0.0328)/t, and its eigen-solution
/// lands on the oracle's Psi, ratio and Ze (SPEC-LIT §153.1).
#[test]
fn adiabatic_line_closed_form() -> Result<(), Error> {
    let gas = EquivalentGas::prescribed(&PrescribedInputs::huang2020_leeds(1000.0))?;
    let curve = AdiabaticCurve::new(&gas, 328.0)?;
    for c in [0.0, 0.25, 0.5, 0.75, 1.0] {
        let (t, r) = curve.state(c)?;
        let t_ref = 1.0 + c * (1500.0 / 328.0 - 1.0);
        let w_c = 1.0 / ((1.0 - c) / 0.0328 + c / 0.0297);
        let r_ref = (w_c / 0.0328) / t_ref;
        println!("c = {c}: t = {t}, r = {r}");
        assert!(rel(t, t_ref) <= 1e-12, "c = {c}: t = {t} not 1e-12 of {t_ref}");
        assert!(rel(r, r_ref) <= 1e-12, "c = {c}: r = {r} not 1e-12 of {r_ref}");
    }
    assert_eq!(curve.state(0.0)?, (1.0, 1.0), "state(0) is not exactly (1, 1)");
    let p = PsiProblem { curve: &curve, theta_u: 20000.0 / 328.0, n: 1.0 };
    let s = p.eigen(&PsiOptions::default())?;
    println!(
        "Leeds adiabatic line: psi = {:e}, ratio_zfk = {}, ze = {}, \
         identity_residual = {:e}",
        s.psi, s.ratio_zfk, s.ze, s.identity_residual
    );
    assert!(rel(s.psi, 7.647549203367135e-5) <= 1e-9, "psi = {:e}", s.psi);
    assert!((s.ratio_zfk - 0.7921150637777358).abs() <= 1e-8, "ratio = {}", s.ratio_zfk);
    assert!((s.ze - 10.41777777777778).abs() <= 1e-12, "ze = {}", s.ze);
    assert!(s.identity_residual.abs() <= 1e-10, "{:e}", s.identity_residual);
    Ok(())
}

/// 7: the formula REMBE gas's own adiabatic line ends exactly at the
/// isobaric burned state, rises strictly in c, and carries its eigenvalue
/// (SPEC-LIT §153.1).
#[test]
fn adiabatic_line_formula_rembe() -> Result<(), Error> {
    let gas_set = tm4513_gas_set()?;
    let gas = EquivalentGas::formula(&FormulaInputs::huang2022_rembe()?, &gas_set)?;
    let b = gas.isobaric_burned(273.0)?;
    let curve = AdiabaticCurve::new(&gas, 273.0)?;
    let mut prev = 0.0;
    for i in 0..=20 {
        let c = i as f64 / 20.0;
        let (t, r) = curve.state(c)?;
        println!("c = {c}: T = {}, r = {r}", t * 273.0);
        if i > 0 {
            assert!(t > prev, "t not strictly increasing at c = {c}");
        }
        if i == 20 {
            assert!(
                rel(t * 273.0, b.t_b) <= 1e-12,
                "T(1) = {} is not 1e-12 of t_b = {}",
                t * 273.0,
                b.t_b
            );
            assert!(rel(r, 1.0 / b.sigma) <= 1e-12, "r(1) = {} not 1e-12 of 1/sigma", r);
        }
        prev = t;
    }
    let p = PsiProblem { curve: &curve, theta_u: 20000.0 / 273.0, n: 1.0 };
    let s = p.eigen(&PsiOptions { n_steps: 20_000, ..Default::default() })?;
    println!(
        "REMBE adiabatic line: psi = {:e}, identity_residual = {:e}",
        s.psi, s.identity_residual
    );
    assert!(s.identity_residual.abs() <= 1e-10, "{:e}", s.identity_residual);
    Ok(())
}

/// 8: the setup pieces — (153.2) exactly, the speed a user-supplied Psi
/// implies (§153.5), the sensitivity of §153.6's n = 0.5 REMBE row, the
/// review table and its Markdown.
#[test]
fn setup_table_and_diagnostic() -> Result<(), Error> {
    let tr = reaction_time(2.0, 3.0, 0.5)?;
    println!("reaction_time(2, 3, 0.5) = {tr}");
    assert_eq!(tr, 48.0, "reaction_time is not exactly 48");
    let pc = PublishedCurve::new(7.04)?;
    let opt = PsiOptions { n_steps: 20_000, ..Default::default() };
    let p05 = PsiProblem { curve: &pc, theta_u: 20000.0 / 273.0, n: 0.5 };
    let p1 = rembe(&pc);
    let s05 = p05.eigen(&opt)?;
    let s1 = p1.eigen(&opt)?;
    let speed = implied_speed_ratio(s05.psi, s1.psi)?;
    println!(
        "REMBE psi n 0.5 = {:e}, n 1 = {:e}, implied speed ratio = {speed} (0.64 S_L)",
        s05.psi, s1.psi
    );
    assert!(rel(speed, 0.6405487222530479) <= 1e-8, "speed ratio = {speed}");
    let rows = documentation_table(&[5.06, 7.04], &[20000.0 / 328.0, 20000.0 / 273.0], &[1.0], &opt)?;
    assert_eq!(rows.len(), 4, "{} rows", rows.len());
    let pc5 = PublishedCurve::new(5.06)?;
    let s_leeds = leeds(&pc5).eigen(&opt)?;
    assert_eq!(rows[0].psi.to_bits(), s_leeds.psi.to_bits(), "row 0 psi differs bitwise");
    assert_eq!(rows[3].psi.to_bits(), s1.psi.to_bits(), "row 3 psi differs bitwise");
    let text = format_table(&rows);
    assert_eq!(text.lines().count(), 4 + 2, "{} lines", text.lines().count());
    println!("{text}");
    Ok(())
}

/// 9: every malformed input is refused with the input, its value and its
/// SPEC-LIT section named.
#[test]
fn refusals() -> Result<(), Error> {
    let gas = EquivalentGas::prescribed(&PrescribedInputs::huang2020_leeds(1000.0))?;
    let pc = PublishedCurve::new(5.06)?;
    assert!(PublishedCurve::new(1.0).is_err(), "sigma = 1 accepted");
    assert!(PublishedCurve::new(f64::NAN).is_err(), "sigma = NaN accepted");
    let good = PsiOptions::default();
    assert!(
        PsiProblem { curve: &pc, theta_u: 0.0, n: 1.0 }.eigen(&good).is_err(),
        "theta_u = 0 accepted"
    );
    assert!(
        PsiProblem { curve: &pc, theta_u: 20000.0 / 328.0, n: f64::NAN }.eigen(&good).is_err(),
        "n = NaN accepted"
    );
    let p = leeds(&pc);
    for (name, opt) in [
        ("n_steps = 30002", PsiOptions { n_steps: 30_002, ..Default::default() }),
        ("c_cut = 1e-5", PsiOptions { n_steps: 20_000, c_cut: 1e-5, ..Default::default() }),
        ("c_cut = 0", PsiOptions { n_steps: 20_000, c_cut: 0.0, ..Default::default() }),
        ("s_max = 10", PsiOptions { n_steps: 20_000, s_max: 10.0, ..Default::default() }),
    ] {
        assert!(p.eigen(&opt).is_err(), "{name} accepted");
    }
    assert!(reaction_time(1.0, 1.0, 0.0).is_err(), "s_l = 0 accepted");
    assert!(implied_speed_ratio(1.0, 0.0).is_err(), "psi_used = 0 accepted");
    assert!(AdiabaticCurve::new(&gas, -1.0).is_err(), "t_u = -1 accepted");
    println!("refusals: all 11 malformed inputs refused");
    Ok(())
}
