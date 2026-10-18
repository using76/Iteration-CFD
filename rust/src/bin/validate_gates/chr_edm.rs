// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.

//! **Gate CHR-EDM (SPEC-LIT §134.5): EDM and PaSR, the library and kernel
//! gate of CHR-14, run live - not a driver run.**
//!
//! EDM and PaSR are not wired into `ofgpu-lowmach` yet (a hand-off row of
//! docs/17-review-later.md), so this section gates the library and the
//! kernels CHR-14 built, exactly as `tests/chem_tci.rs` holds them: the
//! closed, frozen-rho box where (134.2) is linear in the fuel so (134.3) is
//! an exponential, and the product-limited point of §134.2 (class P, 1e-10
//! and 1e-14); the two-step mechanism's element balance over 1000 random
//! states and the refusal of three reactions and of a reversible one
//! (class A, discrete); PaSR's first edge - tau_m = 0 exactly, and kappa = 1
//! returning the laminar `W omega` bitwise on the host and over 64 device
//! cells (discrete); the exact kappa of one first-order reaction and ffr's
//! zero-product edge (class P, discrete); and the device kernels
//! `chemEdmSource` and `chemPasrScale` against their f64 host twins over
//! 4096 random cells fed the values the device saw, the round-off twin of
//! `tci_device_matches_host` at 1e-13 (class R). The mechanisms are built
//! from text over the format-valid THERMO `record` of tests/case_chem.rs,
//! read as data, not from any paper.
//!
//! Written from:
//!   ofgpu `SPEC-LIT.md` §134.5 (the gate table row), §134.1-§134.6 (EDM,
//!     PaSR, the limits, the refusals)
//!   tests/chem_tci.rs - tests `edm_box_closed_form`,
//!     `edm_two_step_conserves_elements`, `pasr_limit_is_laminar`,
//!     `pasr_linear_kappa_exact`, `tci_device_matches_host`; the tests stay
//!     as they are
//! No GPL-licensed source was consulted.

use ofgpu::chem::device::{production_rates, ChemKernels, ChemWork, DeviceMechanism};
use ofgpu::chem::rosenbrock::{integrate, RosOptions};
use ofgpu::chem::tci::{
    edm_source, kappa, pasr_source, pasr_source_device, tau_chem, tau_mix, DeviceEdm,
    DeviceMixing, EdmMechanism, TciKernels,
};
use ofgpu::chem::thermo::ThermoSet;
use ofgpu::chem::{production_rates_tpy, ElementTable, Mechanism, P_ATM, R_GAS};
use ofgpu::io::case_chem::{TauChem, TauMix};
use ofgpu::{Error, Gpu, Result, Scalar};

use crate::Checks;

use super::chr_fixture::SplitMix64;

/// The gate's scope and registry name.
pub(crate) const GATE: &str = "SPEC-LIT 134.5 Gate CHR-EDM EDM and PaSR";

/// One format-valid CHEMKIN THERMO record, `a1 = 3.5` and the rest 0
/// (tests/case_chem.rs's `record`, copied: read as data, not from any
/// paper).
fn record(name: &str, els: &[(&str, u32)]) -> String {
    let mut c1 = format!("{name:<18}TEST  ");
    for (s, n) in els {
        c1.push_str(&format!("{s:<2}{n:>3}"));
    }
    while c1.len() < 44 {
        c1.push(' ');
    }
    c1.push_str(&format!("G{:>10.3}{:>10.3}{:>8.2}      1", 200.0, 6000.0, 1000.0));
    let a = |x: f64| format!("{x:>15.8E}");
    let z = a(0.0);
    format!(
        "{c1}\n{}{z}{z}{z}{z}    2\n{z}{z}{}{z}{z}    3\n{z}{z}{z}{z}                   4\n",
        a(3.5),
        a(3.5)
    )
}

/// The species as a THERMO block, 200/1000/6000 K bounds for all.
fn thermo_text(species: &[(&str, &[(&str, u32)])]) -> String {
    let mut t = String::from("THERMO ALL\n   200.000  1000.000  6000.000\n");
    for (n, e) in species {
        t.push_str(&record(n, e));
    }
    t.push_str("END\n");
    t
}

/// A mechanism from text: the species' records as a set at `P_ATM`, then
/// the ELEMENTS / SPECIES / REACTIONS lines - `tests/chem_tci.rs`'s `mech`.
fn mech(species: &[(&str, &[(&str, u32)])], elements: &str, rx: &[&str]) -> Result<Mechanism> {
    let thermo = thermo_text(species);
    let set = ThermoSet::read_chemkin(&thermo, "test", P_ATM, &ElementTable::standard())?;
    let names: Vec<&str> = species.iter().map(|(n, _)| *n).collect();
    let mut text =
        format!("ELEMENTS {elements} END\nSPECIES {} END\nREACTIONS\n", names.join(" "));
    for r in rx {
        text.push_str(r);
        text.push('\n');
    }
    text.push_str("END\n");
    Mechanism::read_chemkin(&text, "test", Some(&set))
}

/// H2 O2 H2O N2, the elements the H2 fixture's reactions touch.
const H2_SPECIES: &[(&str, &[(&str, u32)])] = &[
    ("H2", &[("H", 2)]),
    ("O2", &[("O", 2)]),
    ("H2O", &[("H", 2), ("O", 1)]),
    ("N2", &[("N", 2)]),
];

/// CH4 O2 CO H2O CO2 N2, the two-step BFER fixture of §134.6's rows.
const CH4_SPECIES: &[(&str, &[(&str, u32)])] = &[
    ("CH4", &[("C", 1), ("H", 4)]),
    ("O2", &[("O", 2)]),
    ("CO", &[("C", 1), ("O", 1)]),
    ("H2O", &[("H", 2), ("O", 1)]),
    ("CO2", &[("C", 1), ("O", 2)]),
    ("N2", &[("N", 2)]),
];

const CH4_RX: &[&str] = &[
    "CH4 + 1.5O2 => CO + 2H2O  1.0E+10 0.0 10000.0",
    "CO + 0.5O2 => CO2  1.0E+10 0.0 10000.0",
];

/// FUEL PROD N2, the one-reaction first-order fixture.
const FP_SPECIES: &[(&str, &[(&str, u32)])] = &[
    ("FUEL", &[("C", 1), ("H", 4)]),
    ("PROD", &[("C", 1), ("H", 4)]),
    ("N2", &[("N", 2)]),
];

/// The one-reaction H2 fixture of §134.6's rows.
fn h2_mech(rx: &[&str]) -> Result<Mechanism> {
    mech(H2_SPECIES, "H O N", rx)
}

/// A refusal's Config text, or empty - the gate matches on the message the
/// way the test matches on the error variant.
fn config_text(r: Result<EdmMechanism>) -> String {
    match r {
        Err(Error::Config(msg)) => msg,
        _ => String::new(),
    }
}

/// **Gate CHR-EDM** (SPEC-LIT §134.5): EDM and PaSR, live.
pub(crate) fn check_chr_edm(c: &mut Checks, gpu: &Gpu) -> Result<()> {
    // ---- the closed-form box of (134.3) ---------------------------------------
    let m = h2_mech(&["2H2 + O2 => 2H2O   1.0E+10 0.0 10000.0"])?;
    let edm = EdmMechanism::new(&m)?;
    let (a, r, rho) = (4.0, 10.0, 1.2);
    let mut rhs = |yy: &[f64], out: &mut [f64]| -> Result<()> {
        let s = edm.source(a, None, r, rho, yy);
        for (o, v) in out.iter_mut().zip(&s) {
            *o = v / rho;
        }
        Ok(())
    };
    for (t, want) in [
        (0.01, 0.006703200460356393),
        (0.05, 0.0013533528323661271),
        (0.1, 0.00018315638888734178),
    ] {
        let mut y = vec![0.01, 0.23, 0.0, 0.76];
        integrate(&mut rhs, &mut y, 0.0, t, &RosOptions::new(4, 1e-12, 1e-20))?;
        let got = y[0];
        c.check_p(
            &format!("Gate CHR-EDM: the closed box's Y_H2 at t = {t} within 1e-10 of (134.3)'s exponential (SPEC-LIT 134.5)"),
            ((got - want) / want).abs() as Scalar,
            1e-10,
        );
    }
    // The product-limited point (B = 0.5): Y_H2O's term wins the minimum,
    // so q is §134.2's product argument alone.
    let w_h2o = m
        .thermo()
        .get("H2O")
        .ok_or_else(|| Error::Config("Gate CHR-EDM: the H2 fixture has no H2O".to_string()))?
        .molar_mass;
    let yy = [0.01, 0.23, 0.001, 0.759];
    let q = edm.progress(a, Some(0.5), r, rho, &yy);
    let want = ((a * r) * rho) * (0.5 * 0.001 / (2.0 * w_h2o));
    c.check_p(
        "Gate CHR-EDM: the product-limited point is §134.2's product argument alone within 1e-14 (SPEC-LIT 134.5)",
        ((q[0] - want) / want).abs() as Scalar,
        1e-14,
    );

    // ---- the two-step mechanism conserves its elements ------------------------
    let m = mech(CH4_SPECIES, "C H O N", CH4_RX)?;
    let edm = EdmMechanism::new(&m)?;
    let sp = m.thermo().species();
    let (a, b, r, rho) = (4.0, Some(0.5), 10.0, 1.0);
    let mut rng = SplitMix64(20261017);
    let mut worst = 0.0f64;
    for _ in 0..1000 {
        let mut y: Vec<f64> = (0..sp.len()).map(|_| rng.u()).collect();
        let s: f64 = y.iter().sum();
        for v in y.iter_mut() {
            *v /= s;
        }
        let q = edm.progress(a, b, r, rho, &y);
        let src = edm.source(a, b, r, rho, &y);
        let sum_q: f64 = q.iter().map(|q| q.abs()).sum();
        for el in ["C", "H", "O"] {
            let e_el: f64 = sp
                .iter()
                .enumerate()
                .map(|(m_i, sp_i)| {
                    let n = sp_i
                        .elements
                        .iter()
                        .find(|(s, _)| s == el)
                        .map_or(0.0, |(_, n)| f64::from(*n));
                    n * src[m_i] / sp_i.molar_mass
                })
                .sum();
            worst = worst.max(e_el.abs() / sum_q);
        }
    }
    c.check_a(
        "Gate CHR-EDM: two-step worst element imbalance over 1000 random states within 1e-14 of sum |q_r| (SPEC-LIT 134.5)",
        worst as Scalar,
        1e-14,
    );
    let three = h2_mech(&[
        "2H2 + O2 => 2H2O  1.0E+10 0.0 10000.0",
        "2H2O => 2H2 + O2  1.0E+10 0.0 10000.0",
        "H2 + 0.5O2 => H2O  1.0E+10 0.0 10000.0",
    ])?;
    c.require(
        "Gate CHR-EDM: three global reactions are refused, naming EDM (SPEC-LIT 134.5)",
        config_text(EdmMechanism::new(&three)).contains("EDM"),
    );
    let rev = h2_mech(&["2H2 + O2 <=> 2H2O   1.0E+10 0.0 10000.0"])?;
    c.require(
        "Gate CHR-EDM: a reversible reaction is refused, naming EDM (SPEC-LIT 134.5)",
        config_text(EdmMechanism::new(&rev)).contains("EDM"),
    );

    // ---- PaSR's first edge: tau_m = 0 is the laminar limit --------------------
    let m = h2_mech(&["2H2 + O2 => 2H2O   1.0E+10 0.0 10000.0"])?;
    let y = [0.02, 0.2, 0.05, 0.73];
    let sp = m.thermo().species();
    let omega = production_rates_tpy(&m, 1500.0, 101325.0, &y)?;
    let wl: Vec<f64> = omega.iter().zip(sp).map(|(o, s)| s.molar_mass * o).collect();
    let tau_m = tau_mix(TauMix::Kolmogorov, None, 0.1, 10.0, 0.0, 0.0);
    c.require(
        "Gate CHR-EDM: kolmogorov at nu = 0 is exactly 0 (SPEC-LIT 134.5)",
        tau_m == 0.0,
    );
    let ydot: Vec<f64> = wl.iter().map(|v| v / 1.2).collect();
    for kind in [TauChem::Sfr, TauChem::Ffr, TauChem::Gfr] {
        let kap = kappa(tau_chem(kind, &y, &ydot), tau_m);
        let out = pasr_source(&wl, kap);
        for (k, (got, want)) in out.iter().zip(&wl).enumerate() {
            c.require(
                &format!("Gate CHR-EDM: kappa = 1 returns the laminar rate bitwise, host {kind:?} species {k} (SPEC-LIT 134.5)"),
                got.to_bits() == want.to_bits(),
            );
        }
    }

    // The device arm: 64 cells of the same state, omega from the device's
    // own production rates.
    let g = gpu;
    let n = 64;
    let ks = m.species().len();
    let ts = vec![1500.0 as Scalar; n];
    let ps = vec![101325.0 as Scalar; n];
    let mut ys = vec![0.0 as Scalar; ks * n];
    for c_i in 0..n {
        for (k, v) in y.iter().enumerate() {
            ys[k * n + c_i] = *v as Scalar;
        }
    }
    let ck = ChemKernels::new(g)?;
    let dm = DeviceMechanism::upload(g, &m)?;
    let mut work = ChemWork::new(g, &dm, n)?;
    let (dt, dp, dy) = (g.upload(&ts)?, g.upload(&ps)?, g.upload(&ys)?);
    let mut domega = g.zeros::<Scalar>(ks * n)?;
    let mut st = g.zeros::<i32>(n)?;
    production_rates(g, &ck, &dm, &mut work, &dt, &dp, &dy, &mut domega, &mut st, n)?;
    g.sync()?;
    c.require(
        "Gate CHR-EDM: the 64 production-rate cells all return status 0 (SPEC-LIT 134.5)",
        g.download(&st)?.iter().all(|&s| s == 0),
    );
    let omega_d = g.download(&domega)?;
    let kern = TciKernels::new(g)?;
    let dk = g.upload(&vec![1.0 as Scalar; n])?;
    let deps = g.upload(&vec![10.0 as Scalar; n])?;
    let dnu = g.upload(&vec![0.0 as Scalar; n])?;
    let drho = g.upload(&vec![1.2 as Scalar; n])?;
    let wv: Vec<f64> = sp.iter().map(|s| s.molar_mass).collect();
    let dw = g.upload(&wv)?;
    let mut dwd = g.zeros::<Scalar>(ks * n)?;
    let mut dst = g.zeros::<i32>(n)?;
    let mix = DeviceMixing::Epsilon { k: &dk, epsilon: &deps };
    pasr_source_device(
        g, &kern, &mix, TauMix::Kolmogorov, TauChem::Sfr, None, &dnu, &dw, &drho, &dy, &domega,
        &mut dwd, &mut dst, n,
    )?;
    g.sync()?;
    let wd = g.download(&dwd)?;
    let sts = g.download(&dst)?;
    let mut bitwise = true;
    for c_i in 0..n {
        bitwise &= sts[c_i] == 0;
        for k in 0..ks {
            let want = (sp[k].molar_mass * omega_d[k * n + c_i] as f64) as Scalar;
            bitwise &= wd[k * n + c_i].to_bits() == want.to_bits();
        }
    }
    c.require(
        "Gate CHR-EDM: kappa = 1 returns the laminar rate bitwise in all 64 device cells (SPEC-LIT 134.5)",
        bitwise,
    );

    // ---- the exact kappa of one first-order reaction, and ffr's edge ----------
    let m = mech(FP_SPECIES, "C H N", &["FUEL => PROD  1000.0 0.0 0.0"])?;
    let y = [0.05, 0.0, 0.95];
    let w_mix = m.thermo().molar_mass_mix(&y)?;
    let rho = 101325.0 * w_mix / (R_GAS * 1000.0);
    let sp = m.thermo().species();
    let omega = production_rates_tpy(&m, 1000.0, 101325.0, &y)?;
    let wl: Vec<f64> = omega.iter().zip(sp).map(|(o, s)| s.molar_mass * o).collect();
    let ydot: Vec<f64> = wl.iter().map(|v| v / rho).collect();
    let tau_c = tau_chem(TauChem::Sfr, &y, &ydot)
        .ok_or_else(|| Error::Config("Gate CHR-EDM: FUEL read as dormant in the Sfr row".to_string()))?;
    let tau_m = tau_mix(TauMix::Integral, Some(1.0), 1.0 / 1000.0, 1.0, 0.0, 0.0);
    let kap = kappa(Some(tau_c), tau_m);
    c.check_p(
        "Gate CHR-EDM: one first-order reaction gives (134.12)'s kappa = 0.5 within 1e-12 (SPEC-LIT 134.5)",
        ((kap - 0.5) / 0.5).abs() as Scalar,
        1e-12,
    );
    let got = pasr_source(&wl, kap)[0] / rho;
    c.check_p(
        "Gate CHR-EDM: W_F omega-bar_F/rho lands on the closed form -25 within 1e-12 (SPEC-LIT 134.5)",
        ((got - -25.0) / -25.0).abs() as Scalar,
        1e-12,
    );
    // `ffr`'s edge: PROD has Y = 0 and is not dormant, so tau_c = 0 and,
    // with tau_m > 0, kappa = 0 exactly (§134.6).
    c.require(
        "Gate CHR-EDM: ffr's zero-product edge gives tau_c exactly 0 (SPEC-LIT 134.5)",
        tau_chem(TauChem::Ffr, &y, &ydot) == Some(0.0),
    );
    c.require(
        "Gate CHR-EDM: ffr's zero stops the reaction - kappa exactly 0 (SPEC-LIT 134.5)",
        kappa(tau_chem(TauChem::Ffr, &y, &ydot), tau_m) == 0.0,
    );

    // ---- the device kernels against their f64 host twins, 4096 cells ----------
    let m = mech(CH4_SPECIES, "C H O N", CH4_RX)?;
    let edm = EdmMechanism::new(&m)?;
    let sp = m.thermo().species();
    let ks = m.species().len();
    let n = 4096;
    let mut rng = SplitMix64(20261017);
    let mut rhos = Vec::with_capacity(n);
    let mut kf = Vec::with_capacity(n);
    let mut es = Vec::with_capacity(n);
    let mut ys = vec![0.0 as Scalar; ks * n];
    for i in 0..n {
        let mut y: Vec<f64> = (0..ks).map(|_| rng.u()).collect();
        let s: f64 = y.iter().sum();
        for v in y.iter_mut() {
            *v /= s;
        }
        for (k, v) in y.iter().enumerate() {
            ys[k * n + i] = *v as Scalar;
        }
        rhos.push((0.2 + 1.8 * rng.u()) as Scalar);
        kf.push((0.1 + 9.9 * rng.u()) as Scalar);
        es.push((1.0 + 999.0 * rng.u()) as Scalar);
    }
    let kern = TciKernels::new(g)?;
    let de = DeviceEdm::upload(g, &edm)?;
    let drho = g.upload(&rhos)?;
    let dk = g.upload(&kf)?;
    let deps = g.upload(&es)?;
    let dy = g.upload(&ys)?;
    let mut dwd = g.zeros::<Scalar>(ks * n)?;
    let mut dst = g.zeros::<i32>(n)?;
    let mix = DeviceMixing::Epsilon { k: &dk, epsilon: &deps };
    edm_source(g, &kern, &de, &mix, 4.0, Some(0.5), &drho, &dy, &mut dwd, &mut dst, n)?;
    g.sync()?;
    let wd = g.download(&dwd)?;
    let sts = g.download(&dst)?;
    let (mut worst, mut refused, mut bad_bits) = (0.0f64, 0usize, false);
    for i in 0..n {
        refused += (sts[i] != 0) as usize;
        let rho_c = rhos[i] as Scalar as f64;
        let k_c = kf[i] as Scalar as f64;
        let e_c = es[i] as Scalar as f64;
        let y_c: Vec<f64> = (0..ks).map(|k| ys[k * n + i] as Scalar as f64).collect();
        let host = edm.source(4.0, Some(0.5), e_c / k_c, rho_c, &y_c);
        let scale = host.iter().fold(0.0f64, |acc, v| acc.max(v.abs()));
        for k in 0..ks {
            let dev = wd[k * n + i] as f64;
            let e = if scale == 0.0 {
                bad_bits |= dev.to_bits() != (host[k] as Scalar as f64).to_bits();
                0.0
            } else {
                (dev - host[k]).abs() / scale
            };
            worst = worst.max(e);
        }
    }
    c.require(
        "Gate CHR-EDM: no chemEdmSource cell is refused over the 4096 (SPEC-LIT 134.5)",
        refused == 0,
    );
    c.require(
        "Gate CHR-EDM: every zero-scale chemEdmSource rate is bitwise the host's (SPEC-LIT 134.5)",
        !bad_bits,
    );
    c.check_roundoff(
        "Gate CHR-EDM: chemEdmSource against its f64 host twin, worst relative error over the 4096 cells (SPEC-LIT 134.5)",
        worst as Scalar,
        1e-13,
    );

    // PaSR, geometric + gfr: the omega comes from the device's own
    // production rates at T = 1500, p = 101325, and the host is fed the
    // downloaded values.
    let dts = g.upload(&vec![1500.0 as Scalar; n])?;
    let dps = g.upload(&vec![101325.0 as Scalar; n])?;
    let mut domega = g.zeros::<Scalar>(ks * n)?;
    let mut pst = g.zeros::<i32>(n)?;
    {
        let ck = ChemKernels::new(g)?;
        let dm = DeviceMechanism::upload(g, &m)?;
        let mut work = ChemWork::new(g, &dm, n)?;
        production_rates(g, &ck, &dm, &mut work, &dts, &dps, &dy, &mut domega, &mut pst, n)?;
        g.sync()?;
    }
    let prod_statuses = g.download(&pst)?;
    c.require(
        "Gate CHR-EDM: the 4096 production-rate cells all return status 0 (SPEC-LIT 134.5)",
        prod_statuses.iter().all(|&s| s == 0),
    );
    let omega_d = g.download(&domega)?;
    let dnu = g.upload(&vec![1.5e-5 as Scalar; n])?;
    let wv: Vec<f64> = sp.iter().map(|s| s.molar_mass).collect();
    let dw = g.upload(&wv)?;
    let mut pwd = g.zeros::<Scalar>(ks * n)?;
    let mut pst2 = g.zeros::<i32>(n)?;
    pasr_source_device(
        g, &kern, &mix, TauMix::Geometric, TauChem::Gfr, None, &dnu, &dw, &drho, &dy, &domega,
        &mut pwd, &mut pst2, n,
    )?;
    g.sync()?;
    let pwd_d = g.download(&pwd)?;
    let sts = g.download(&pst2)?;
    let nu_c = 1.5e-5 as Scalar as f64;
    let (mut worst, mut refused, mut bad_bits) = (0.0f64, 0usize, false);
    for i in 0..n {
        refused += (sts[i] != 0) as usize;
        let rho_c = rhos[i] as Scalar as f64;
        let k_c = kf[i] as Scalar as f64;
        let e_c = es[i] as Scalar as f64;
        let y_c: Vec<f64> = (0..ks).map(|k| ys[k * n + i] as Scalar as f64).collect();
        let tau_m = tau_mix(TauMix::Geometric, None, k_c / e_c, e_c, nu_c, 0.0);
        let mut wl = vec![0.0f64; ks];
        let mut ydot = vec![0.0f64; ks];
        for k in 0..ks {
            wl[k] = wv[k] * omega_d[k * n + i] as f64;
            ydot[k] = wl[k] / rho_c;
        }
        let kap = kappa(tau_chem(TauChem::Gfr, &y_c, &ydot), tau_m);
        let scale = wl.iter().fold(0.0f64, |acc, v| acc.max((kap * v).abs()));
        for k in 0..ks {
            let want = kap * wl[k];
            let dev = pwd_d[k * n + i] as f64;
            let e = if scale == 0.0 {
                bad_bits |= dev.to_bits() != (want as Scalar as f64).to_bits();
                0.0
            } else {
                (dev - want).abs() / scale
            };
            worst = worst.max(e);
        }
    }
    c.require(
        "Gate CHR-EDM: no chemPasrScale cell is refused over the 4096 (SPEC-LIT 134.5)",
        refused == 0,
    );
    c.require(
        "Gate CHR-EDM: every zero-scale chemPasrScale rate is bitwise the host's (SPEC-LIT 134.5)",
        !bad_bits,
    );
    c.check_roundoff(
        "Gate CHR-EDM: chemPasrScale against its f64 host twin, worst relative error over the 4096 cells (SPEC-LIT 134.5)",
        worst as Scalar,
        1e-13,
    );
    Ok(())
}




