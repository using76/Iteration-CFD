// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.
// Provenance: see PROVENANCE.md. No GPL-licensed source was consulted.

//! CHR-14's and CHR-15's gates (SPEC-LIT §134.5, §134.6, §134.7): EDM's
//! closed-form box (`edm_box_closed_form`), the two-step mechanism's
//! element conservation (`edm_two_step_conserves_elements`), the refusal
//! of a case without `A` (`edm_requires_a_and_b`), PaSR's two limits
//! (`pasr_limit_is_laminar` bitwise, `pasr_linear_kappa_exact` at 1e-12),
//! the selector's refusals (`tci_refuses_no_mixing_rate`), the device
//! kernels `chemEdmSource` and `chemPasrScale` against the f64 host twins
//! (`tci_device_matches_host`), and EDC's rows: the cascade constants
//! (`edc_constants_by_variant`), the PSR closed form on the host and on
//! `chemEdcIntegrate` (`edc_psr_closed_form`), the counted cap
//! (`edc_cap_is_counted`), the refused batch variant
//! (`edc_refuses_batch_variant`) and the mean-enthalpy hold
//! (`edc_psr_holds_mean_enthalpy`).
//!
//! An integration test, so every row runs in BOTH builds: the kernels
//! compute in double in both (SPEC-LIT §132.6, §134.6, §134.7), and the
//! f32 build holds the host to the f32-rounded inputs the device saw. The
//! GPU tests return early when the box has no card. The mechanisms are
//! built from text over tests/case_chem.rs's format-valid THERMO
//! `record`.
//! No GPL-licensed source was consulted.

#![allow(clippy::excessive_precision)]

use ofgpu::chem::device::{production_rates, ChemKernels, ChemWork, DeviceMechanism, DeviceReactor};
use ofgpu::chem::mechanism::Mechanism;
use ofgpu::chem::reactor::ReactorOptions;
use ofgpu::chem::rosenbrock::{integrate, RosOptions};
use ofgpu::chem::thermo::ThermoSet;
use ofgpu::chem::tci::{
    edc_constants, edc_host, edc_report, edc_scales, edc_source_device, edm_source, kappa,
    pasr_source, pasr_source_device, select, select_for, tau_chem, tau_mix, Closure, DeviceEdm,
    DeviceMixing, EdcParams, EdcReport, EdcWork, EdmMechanism, TciKernels,
};
use ofgpu::chem::{production_rates_tpy, ElementTable, P_ATM, R_GAS};
use ofgpu::device::Gpu;
use ofgpu::error::Error;
use ofgpu::io::case_chem::{
    lower_blocks, ChemBlocks, EdcVariant, JsonChemistry, JsonCombustion, JsonSpecies,
    LoweredCombustion, MixingKind, TauChem, TauMix,
};
use ofgpu::models::coupled::MixingRate;
use ofgpu::Scalar;

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
/// the ELEMENTS / SPECIES / REACTIONS lines.
fn mech(species: &[(&str, &[(&str, u32)])], elements: &str, rx: &[&str]) -> Mechanism {
    let thermo = thermo_text(species);
    let set = ThermoSet::read_chemkin(&thermo, "test", P_ATM, &ElementTable::standard())
        .expect("the thermo fixture reads");
    let names: Vec<&str> = species.iter().map(|(n, _)| *n).collect();
    let mut text =
        format!("ELEMENTS {elements} END\nSPECIES {} END\nREACTIONS\n", names.join(" "));
    for r in rx {
        text.push_str(r);
        text.push('\n');
    }
    text.push_str("END\n");
    Mechanism::read_chemkin(&text, "test", Some(&set))
        .unwrap_or_else(|e| panic!("the mechanism fixture does not read: {e}"))
}

/// H2 O2 H2O N2, the elements the H2 fixture's reactions touch.
const H2_SPECIES: &[(&str, &[(&str, u32)])] = &[
    ("H2", &[("H", 2)]),
    ("O2", &[("O", 2)]),
    ("H2O", &[("H", 2), ("O", 1)]),
    ("N2", &[("N", 2)]),
];

/// The one-reaction H2 fixture of §134.6's rows.
fn h2_mech(rx: &[&str]) -> Mechanism {
    mech(H2_SPECIES, "H O N", rx)
}

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

/// The process's device, if the box has one: every GPU test returns early
/// rather than failing (tests/chem_device.rs's `gpu`, copied).
fn gpu() -> Option<Gpu> {
    Gpu::new(0).ok()
}

/// SplitMix64, seeded per test; `u()` is uniform on [0, 1) as
/// (x >> 11) 2^-53 (tests/chem_device.rs's generator).
struct SplitMix64(u64);

impl SplitMix64 {
    fn u(&mut self) -> f64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        (z >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }
}

/// The twin gate's tolerance: 1e-13 in f64, 1e-6 in the f32 build, where
/// the host is fed the f32-rounded inputs the device saw.
#[cfg(not(feature = "single"))]
fn tci_tol() -> f64 {
    1.0e-13
}

#[cfg(feature = "single")]
fn tci_tol() -> f64 {
    1.0e-6
}

/// The EDC twin gate's tolerance: 1e-8 in f64, 1e-6 in the f32 build
/// (SPEC-LIT §134.5, §134.7).
#[cfg(not(feature = "single"))]
fn edc_dev_tol() -> f64 {
    1.0e-8
}

#[cfg(feature = "single")]
fn edc_dev_tol() -> f64 {
    1.0e-6
}

/// One format-valid CHEMKIN THERMO record with BOTH a6 slots set to `a6`
/// (§134.7's EXO fixture: the product's enthalpy 5000 K * R per mole below
/// the fuel's, so h/RT carries -5000/T); `record` is this at `a6 = 0`.
fn record_h(name: &str, els: &[(&str, u32)], a6: f64) -> String {
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
    let s6 = a(a6);
    format!(
        "{c1}\n{}{z}{z}{z}{z}    2\n{s6}{z}{}{z}{z}    3\n{z}{z}{s6}{z}                   4\n",
        a(3.5),
        a(3.5)
    )
}

/// FUEL PROD N2, the species of §134.7's LIN and EXO fixtures.
const FP_SPECIES: &[(&str, &[(&str, u32)])] = &[
    ("FUEL", &[("C", 1), ("H", 4)]),
    ("PROD", &[("C", 1), ("H", 4)]),
    ("N2", &[("N", 2)]),
];

/// A mechanism from text with per-species a6: the species' records as a
/// set at `P_ATM`, then the ELEMENTS / SPECIES / REACTIONS lines.
fn mech_h(
    species: &[(&str, &[(&str, u32)], f64)],
    elements: &str,
    rx: &[&str],
) -> Mechanism {
    let mut thermo = String::from("THERMO ALL\n   200.000  1000.000  6000.000\n");
    for (n, e, a6) in species {
        thermo.push_str(&record_h(n, e, *a6));
    }
    thermo.push_str("END\n");
    let set = ThermoSet::read_chemkin(&thermo, "test", P_ATM, &ElementTable::standard())
        .expect("the thermo fixture reads");
    let names: Vec<&str> = species.iter().map(|(n, _, _)| *n).collect();
    let mut text =
        format!("ELEMENTS {elements} END\nSPECIES {} END\nREACTIONS\n", names.join(" "));
    for r in rx {
        text.push_str(r);
        text.push('\n');
    }
    text.push_str("END\n");
    Mechanism::read_chemkin(&text, "test", Some(&set))
        .unwrap_or_else(|e| panic!("the mechanism fixture does not read: {e}"))
}

/// §134.7's lowered parameters: row (a)'s constants, the 0.95 cap and the
/// 20-time-constant PSR span.
fn prm(v: EdcVariant) -> EdcParams {
    EdcParams { variant: v, cd1: 0.135, cd2: 0.5, gamma_max: 0.95, psr_span: 20.0 }
}

/// §134.7's rows run at `rtol 1e-10`.
fn edc_opts() -> ReactorOptions {
    ReactorOptions { rtol: 1.0e-10, atol_y: 1.0e-14, atol_t: 1.0e-8, ..Default::default() }
}

/// SPEC-LIT §134.5 `edm_box_closed_form`: a closed, frozen-rho box with
/// `B` absent and a fuel-lean mixture makes (134.2) linear in the fuel, so
/// (134.3) is `Y_F(t) = Y_F(0) exp(-A eps t / k)`; and one product-limited
/// point checks §134.2's reduction of Magnussen & Hjertager's third
/// argument.
#[test]
fn edm_box_closed_form() {
    let m = h2_mech(&["2H2 + O2 => 2H2O   1.0E+10 0.0 10000.0"]);
    let edm = EdmMechanism::new(&m).expect("one global reaction is EDM-ready");
    let (a, r, rho) = (4.0, 10.0, 1.2);
    let mut rhs = |yy: &[f64], out: &mut [f64]| -> ofgpu::Result<()> {
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
        integrate(&mut rhs, &mut y, 0.0, t, &RosOptions::new(4, 1e-12, 1e-20))
            .expect("the box integrates");
        let got = y[0];
        let e = ((got - want) / want).abs();
        eprintln!("edm_box_closed_form: t = {t}: Y_H2 = {got:e}, relative error {e:.3e}");
        assert!(e <= 1e-10, "t = {t}: Y_H2 = {got:e} against {want:e}, relative error {e:.3e}");
    }
    // The product-limited point (B = 0.5): Y_H2O's term wins the minimum,
    // so q is §134.2's product argument alone.
    let w_h2o = m.thermo().get("H2O").expect("H2O").molar_mass;
    let yy = [0.01, 0.23, 0.001, 0.759];
    let q = edm.progress(a, Some(0.5), r, rho, &yy);
    let want = ((a * r) * rho) * (0.5 * 0.001 / (2.0 * w_h2o));
    let got = q[0];
    let e = ((got - want) / want).abs();
    eprintln!("edm_box_closed_form: product-limited q = {got:e}, relative error {e:.3e}");
    assert!(e <= 1e-14, "product-limited q = {got:e} against {want:e}, relative error {e:.3e}");
}

/// SPEC-LIT §134.5 `edm_two_step_conserves_elements`: (134.2) is a rate of
/// progress multiplied by a balanced stoichiometric vector, so no element
/// is produced or destroyed; and `EdmMechanism::new` refuses three
/// reactions and a reversible one, naming EDM.
#[test]
fn edm_two_step_conserves_elements() {
    let m = mech(CH4_SPECIES, "C H O N", CH4_RX);
    let edm = EdmMechanism::new(&m).expect("two global reactions are EDM-ready");
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
    eprintln!("edm_two_step_conserves_elements: worst |E_e|/sum|q_r| = {worst:.3e}");
    assert!(worst <= 1e-14, "worst |E_e|/sum|q_r| = {worst:.3e} exceeds 1e-14");
    // Three reactions over the H2 set are refused, naming EDM.
    let three = h2_mech(&[
        "2H2 + O2 => 2H2O  1.0E+10 0.0 10000.0",
        "2H2O => 2H2 + O2  1.0E+10 0.0 10000.0",
        "H2 + 0.5O2 => H2O  1.0E+10 0.0 10000.0",
    ]);
    match EdmMechanism::new(&three) {
        Err(Error::Config(msg)) => assert!(msg.contains("EDM"), "{msg}"),
        Err(other) => panic!("three reactions: expected the not-global refusal, got {other:?}"),
        Ok(_) => panic!("three reactions: expected a refusal"),
    }
    let rev = h2_mech(&["2H2 + O2 <=> 2H2O   1.0E+10 0.0 10000.0"]);
    match EdmMechanism::new(&rev) {
        Err(Error::Config(msg)) => assert!(msg.contains("EDM"), "{msg}"),
        Err(other) => panic!("reversible: expected the non-global refusal, got {other:?}"),
        Ok(_) => panic!("reversible: expected a refusal"),
    }
}

/// SPEC-LIT §134.5 `edm_requires_a_and_b`: the lowered case without `A` is
/// refused, naming `combustion.A` and the closed paper (tests/case_chem.rs
/// carries the same refusal in both builds; this is the selector's arm).
#[test]
fn edm_requires_a_and_b() {
    const SPECIES: &str = r#"{ "names": ["H2", "O2", "H2O", "N2"], "inert": "N2",
        "transport": { "Le": { "H2": 0.3 }, "Sct": 0.9 } }"#;
    const CHEMISTRY: &str = r#"{ "mechanism": "mech.inp", "thermo": "therm.dat" }"#;
    let sp: JsonSpecies = serde_json::from_str(SPECIES).unwrap();
    let ch: JsonChemistry = serde_json::from_str(CHEMISTRY).unwrap();
    let co: JsonCombustion = serde_json::from_str(r#"{ "model": "EDM" }"#).unwrap();
    let blocks = ChemBlocks { species: Some(&sp), chemistry: Some(&ch), combustion: Some(&co) };
    match lower_blocks(blocks, Some(MixingKind::Epsilon), false, None) {
        Err(Error::Refused(msg)) => {
            assert!(msg.contains("combustion.A") && msg.contains("Magnussen"), "{msg}")
        }
        other => panic!("EDM without A: expected a refusal, got {other:?}"),
    }
}

/// SPEC-LIT §134.5 `pasr_limit_is_laminar`: `tau_m = 0` is (134.15)'s
/// first edge, and kappa = 1 returns the laminar `W omega` bitwise, on the
/// host and on the device.
#[test]
fn pasr_limit_is_laminar() {
    let m = h2_mech(&["2H2 + O2 => 2H2O   1.0E+10 0.0 10000.0"]);
    let y = [0.02, 0.2, 0.05, 0.73];
    let sp = m.thermo().species();
    let omega = production_rates_tpy(&m, 1500.0, 101325.0, &y).expect("the rates evaluate");
    let wl: Vec<f64> = omega.iter().zip(sp).map(|(o, s)| s.molar_mass * o).collect();
    let tau_m = tau_mix(TauMix::Kolmogorov, None, 0.1, 10.0, 0.0, 0.0);
    assert_eq!(tau_m, 0.0, "kolmogorov at nu = 0 is exactly 0");
    let ydot: Vec<f64> = wl.iter().map(|v| v / 1.2).collect();
    for kind in [TauChem::Sfr, TauChem::Ffr, TauChem::Gfr] {
        let kap = kappa(tau_chem(kind, &y, &ydot), tau_m);
        let out = pasr_source(&wl, kap);
        for (k, (got, want)) in out.iter().zip(&wl).enumerate() {
            assert_eq!(
                got.to_bits(),
                want.to_bits(),
                "species {k}: kappa = 1 must return the laminar rate bitwise"
            );
        }
    }
    eprintln!("pasr_limit_is_laminar: the host holds the laminar rates bitwise");
    // The device arm: 64 cells of the same state, omega from the device's
    // own production rates.
    let Some(g) = gpu() else { return };
    let n = 64;
    let ks = m.species().len();
    let ts = vec![1500.0 as Scalar; n];
    let ps = vec![101325.0 as Scalar; n];
    let mut ys = vec![0.0 as Scalar; ks * n];
    for c in 0..n {
        for (k, v) in y.iter().enumerate() {
            ys[k * n + c] = *v as Scalar;
        }
    }
    let ck = ChemKernels::new(&g).expect("kernels");
    let dm = DeviceMechanism::upload(&g, &m).expect("upload");
    let mut work = ChemWork::new(&g, &dm, n).expect("work");
    let (dt, dp, dy) = (
        g.upload(&ts).expect("upload"),
        g.upload(&ps).expect("upload"),
        g.upload(&ys).expect("upload"),
    );
    let mut domega = g.zeros::<Scalar>(ks * n).expect("alloc");
    let mut st = g.zeros::<i32>(n).expect("alloc");
    production_rates(&g, &ck, &dm, &mut work, &dt, &dp, &dy, &mut domega, &mut st, n)
        .expect("launch");
    g.sync().expect("sync");
    assert!(g.download(&st).expect("download").iter().all(|&s| s == 0), "production statuses");
    let omega_d = g.download(&domega).expect("download");
    let kern = TciKernels::new(&g).expect("tci kernels");
    let dk = g.upload(&vec![1.0 as Scalar; n]).expect("upload");
    let deps = g.upload(&vec![10.0 as Scalar; n]).expect("upload");
    let dnu = g.upload(&vec![0.0 as Scalar; n]).expect("upload");
    let drho = g.upload(&vec![1.2 as Scalar; n]).expect("upload");
    let wv: Vec<f64> = sp.iter().map(|s| s.molar_mass).collect();
    let dw = g.upload(&wv).expect("upload");
    let mut dwd = g.zeros::<Scalar>(ks * n).expect("alloc");
    let mut dst = g.zeros::<i32>(n).expect("alloc");
    let mix = DeviceMixing::Epsilon { k: &dk, epsilon: &deps };
    pasr_source_device(
        &g, &kern, &mix, TauMix::Kolmogorov, TauChem::Sfr, None, &dnu, &dw, &drho, &dy, &domega,
        &mut dwd, &mut dst, n,
    )
    .expect("launch");
    g.sync().expect("sync");
    let wd = g.download(&dwd).expect("download");
    let sts = g.download(&dst).expect("download");
    for c in 0..n {
        assert_eq!(sts[c], 0, "cell {c} refused with status {}", sts[c]);
        for k in 0..ks {
            let want = (sp[k].molar_mass * omega_d[k * n + c] as f64) as Scalar;
            assert_eq!(
                wd[k * n + c].to_bits(),
                want.to_bits(),
                "cell {c} species {k}: the device must return the laminar rate bitwise"
            );
        }
    }
    eprintln!("pasr_limit_is_laminar: 64 device cells hold the laminar rates bitwise");
}

/// SPEC-LIT §134.5 `pasr_linear_kappa_exact`: one first-order reaction
/// gives (134.12)'s kappa exactly, and `ffr`'s zero-product edge (a
/// non-dormant species with Y = 0) gives kappa = 0 exactly.
#[test]
fn pasr_linear_kappa_exact() {
    const FP: &[(&str, &[(&str, u32)])] = &[
        ("FUEL", &[("C", 1), ("H", 4)]),
        ("PROD", &[("C", 1), ("H", 4)]),
        ("N2", &[("N", 2)]),
    ];
    let m = mech(FP, "C H N", &["FUEL => PROD  1000.0 0.0 0.0"]);
    let y = [0.05, 0.0, 0.95];
    let w_mix = m.thermo().molar_mass_mix(&y).expect("W_mix");
    let rho = 101325.0 * w_mix / (R_GAS * 1000.0);
    let sp = m.thermo().species();
    let omega = production_rates_tpy(&m, 1000.0, 101325.0, &y).expect("the rates evaluate");
    let wl: Vec<f64> = omega.iter().zip(sp).map(|(o, s)| s.molar_mass * o).collect();
    let ydot: Vec<f64> = wl.iter().map(|v| v / rho).collect();
    let tau_c = tau_chem(TauChem::Sfr, &y, &ydot).expect("FUEL is not dormant");
    let tau_m = tau_mix(TauMix::Integral, Some(1.0), 1.0 / 1000.0, 1.0, 0.0, 0.0);
    let kap = kappa(Some(tau_c), tau_m);
    let e = ((kap - 0.5) / 0.5).abs();
    eprintln!("pasr_linear_kappa_exact: kappa = {kap}, relative error {e:.3e}");
    assert!(e <= 1e-12, "kappa {kap} against 0.5, relative error {e:.3e}");
    let got = pasr_source(&wl, kap)[0] / rho;
    let e = ((got - -25.0) / -25.0).abs();
    eprintln!("pasr_linear_kappa_exact: W_F omega-bar_F/rho = {got}, relative error {e:.3e}");
    assert!(e <= 1e-12, "W_F omega-bar_F/rho {got} against -25, relative error {e:.3e}");
    // `ffr`'s edge: PROD has Y = 0 and is not dormant, so tau_c = 0 and,
    // with tau_m > 0, kappa = 0 exactly (§134.6).
    assert_eq!(
        tau_chem(TauChem::Ffr, &y, &ydot),
        Some(0.0),
        "PROD has Y = 0 and is not dormant"
    );
    assert_eq!(
        kappa(tau_chem(TauChem::Ffr, &y, &ydot), tau_m),
        0.0,
        "ffr's zero stops the reaction until its products exist"
    );
    eprintln!("pasr_linear_kappa_exact: ffr's zero gives kappa = 0 exactly");
}

/// SPEC-LIT §134.5 `tci_refuses_no_mixing_rate`: §134.1's refusals, each
/// naming the combination.
#[test]
fn tci_refuses_no_mixing_rate() {
    let edc = || {
        LoweredCombustion::Edc {
            variant: EdcVariant::V1989,
            cd1: 0.135,
            cd2: 0.5,
            gamma_max: 0.95,
            psr_span: 20.0,
        }
    };
    let cases: Vec<(LoweredCombustion, &str)> = vec![
        (LoweredCombustion::Edm { a: 4.0, b: None }, "EDM"),
        (edc(), "EDC"),
        (
            LoweredCombustion::Pasr {
                tau_mix: TauMix::Integral,
                tau_chem: TauChem::Sfr,
                c_mix: Some(1.0),
            },
            "PaSR",
        ),
    ];
    for (c, model) in &cases {
        let msg = match select(c, MixingKind::None, true) {
            Err(Error::Config(msg)) => msg,
            other => panic!("{model} with MixingRate::None: expected a refusal, got {other:?}"),
        };
        assert!(
            msg.contains(model) && msg.contains("MixingRate::None"),
            "{model}: the refusal does not name the combination: {msg}"
        );
        let none = MixingRate::None;
        let msg = match select_for(c, &none, None) {
            Err(Error::Config(msg)) => msg,
            other => panic!("{model}: expected a refusal, got {other:?}"),
        };
        assert!(
            msg.contains(model) && msg.contains("MixingRate::None"),
            "{model}: the refusal does not name the combination: {msg}"
        );
        assert!(
            DeviceMixing::from_rate(&none, None).is_err(),
            "{model}: from_rate refuses MixingRate::None"
        );
    }
    eprintln!("tci_refuses_no_mixing_rate: EDM, EDC and PaSR refuse MixingRate::None by name");
    assert_eq!(
        select(&LoweredCombustion::Laminar, MixingKind::None, true).expect("laminar takes None"),
        Closure::Laminar
    );
    let edm = LoweredCombustion::Edm { a: 4.0, b: None };
    match select(&edm, MixingKind::Strain, true) {
        Err(Error::Config(msg)) => assert!(msg.contains("MixingRate::Strain"), "{msg}"),
        other => panic!("EDM with Strain: expected a refusal, got {other:?}"),
    }
    match select(&edc(), MixingKind::Epsilon, true) {
        Ok(Closure::Edc(p)) => assert_eq!(
            p,
            EdcParams {
                variant: EdcVariant::V1989,
                cd1: 0.135,
                cd2: 0.5,
                gamma_max: 0.95,
                psr_span: 20.0
            },
            "EDC under Epsilon lowers to its own parameters"
        ),
        other => panic!("EDC under Epsilon: expected Closure::Edc, got {other:?}"),
    }
    match select(&edc(), MixingKind::Omega, false) {
        Err(Error::Config(msg)) => {
            assert!(msg.contains("k field") && msg.contains("EDC"), "{msg}")
        }
        other => panic!("EDC under Omega without k: expected a refusal, got {other:?}"),
    }
    let pasr =
        LoweredCombustion::Pasr { tau_mix: TauMix::Kolmogorov, tau_chem: TauChem::Sfr, c_mix: None };
    match select(&pasr, MixingKind::Omega, false) {
        Err(Error::Config(msg)) => assert!(msg.contains("k field"), "{msg}"),
        other => panic!("kolmogorov without k: expected a refusal, got {other:?}"),
    }
    assert_eq!(
        select(&edm, MixingKind::Epsilon, true).expect("EDM takes Epsilon"),
        Closure::Edm { a: 4.0, b: None }
    );
}

/// SPEC-LIT §134.5 `tci_device_matches_host`: 4096 random cells,
/// `chemEdmSource` and then `chemPasrScale` (geometric + gfr) against the
/// f64 host twins fed the values the device saw.
#[test]
fn tci_device_matches_host() {
    let Some(g) = gpu() else { return };
    let m = mech(CH4_SPECIES, "C H O N", CH4_RX);
    let edm = EdmMechanism::new(&m).expect("two global reactions are EDM-ready");
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
    let kern = TciKernels::new(&g).expect("tci kernels");
    let de = DeviceEdm::upload(&g, &edm).expect("upload");
    let drho = g.upload(&rhos).expect("upload");
    let dk = g.upload(&kf).expect("upload");
    let deps = g.upload(&es).expect("upload");
    let dy = g.upload(&ys).expect("upload");
    let mut dwd = g.zeros::<Scalar>(ks * n).expect("alloc");
    let mut dst = g.zeros::<i32>(n).expect("alloc");
    let mix = DeviceMixing::Epsilon { k: &dk, epsilon: &deps };
    edm_source(&g, &kern, &de, &mix, 4.0, Some(0.5), &drho, &dy, &mut dwd, &mut dst, n)
        .expect("launch");
    g.sync().expect("sync");
    let wd = g.download(&dwd).expect("download");
    let sts = g.download(&dst).expect("download");
    let (mut worst, mut wc) = (0.0f64, 0usize);
    for i in 0..n {
        assert_eq!(sts[i], 0, "EDM cell {i} refused with status {}", sts[i]);
        let rho_c = rhos[i] as Scalar as f64;
        let k_c = kf[i] as Scalar as f64;
        let e_c = es[i] as Scalar as f64;
        let y_c: Vec<f64> = (0..ks).map(|k| ys[k * n + i] as Scalar as f64).collect();
        let host = edm.source(4.0, Some(0.5), e_c / k_c, rho_c, &y_c);
        let scale = host.iter().fold(0.0f64, |acc, v| acc.max(v.abs()));
        for k in 0..ks {
            let dev = wd[k * n + i] as f64;
            let e = if scale == 0.0 {
                assert_eq!(dev.to_bits(), (host[k] as Scalar as f64).to_bits(), "cell {i} species {k}");
                0.0
            } else {
                (dev - host[k]).abs() / scale
            };
            if e > worst {
                worst = e;
                wc = i;
            }
        }
    }
    eprintln!(
        "tci_device_matches_host: EDM worst relative error {worst:.3e} at cell {wc}; tolerance {:e}",
        tci_tol()
    );
    assert!(worst <= tci_tol(), "EDM worst {worst:.3e} at cell {wc} exceeds {:e}", tci_tol());
    // PaSR, geometric + gfr: the omega comes from the device's own
    // production rates at T = 1500, p = 101325, and the host is fed the
    // downloaded values.
    let dts = g.upload(&vec![1500.0 as Scalar; n]).expect("upload");
    let dps = g.upload(&vec![101325.0 as Scalar; n]).expect("upload");
    let mut domega = g.zeros::<Scalar>(ks * n).expect("alloc");
    let mut pst = g.zeros::<i32>(n).expect("alloc");
    {
        let ck = ChemKernels::new(&g).expect("kernels");
        let dm = DeviceMechanism::upload(&g, &m).expect("upload");
        let mut work = ChemWork::new(&g, &dm, n).expect("work");
        production_rates(&g, &ck, &dm, &mut work, &dts, &dps, &dy, &mut domega, &mut pst, n)
            .expect("launch");
        g.sync().expect("sync");
    }
    assert!(g.download(&pst).expect("download").iter().all(|&s| s == 0), "production statuses");
    let omega_d = g.download(&domega).expect("download");
    let dnu = g.upload(&vec![1.5e-5 as Scalar; n]).expect("upload");
    let wv: Vec<f64> = sp.iter().map(|s| s.molar_mass).collect();
    let dw = g.upload(&wv).expect("upload");
    let mut pwd = g.zeros::<Scalar>(ks * n).expect("alloc");
    let mut pst2 = g.zeros::<i32>(n).expect("alloc");
    pasr_source_device(
        &g, &kern, &mix, TauMix::Geometric, TauChem::Gfr, None, &dnu, &dw, &drho, &dy, &domega,
        &mut pwd, &mut pst2, n,
    )
    .expect("launch");
    g.sync().expect("sync");
    let pwd_d = g.download(&pwd).expect("download");
    let sts = g.download(&pst2).expect("download");
    let nu_c = 1.5e-5 as Scalar as f64;
    (worst, wc) = (0.0, 0);
    for i in 0..n {
        assert_eq!(sts[i], 0, "PaSR cell {i} refused with status {}", sts[i]);
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
                assert_eq!(dev.to_bits(), (want as Scalar as f64).to_bits(), "cell {i} species {k}");
                0.0
            } else {
                (dev - want).abs() / scale
            };
            if e > worst {
                worst = e;
                wc = i;
            }
        }
    }
    eprintln!(
        "tci_device_matches_host: PaSR worst relative error {worst:.3e} at cell {wc}; tolerance {:e}",
        tci_tol()
    );
    assert!(worst <= tci_tol(), "PaSR worst {worst:.3e} at cell {wc} exceeds {:e}", tci_tol());
}

/// SPEC-LIT §134.5 `edc_constants_by_variant`: `C_gamma` and `C_tau` of
/// row (a) against 2.129810347487575 and 0.408248290463863, and `gamma_lambda`,
/// `gamma_star` and `tau_star` at `nu = 1.5e-5, k = 1, eps = 10` against
/// the §134.3 hand table, all to 1e-14 relative, with no cap.
#[test]
fn edc_constants_by_variant() {
    let (cg, ct) = edc_constants(0.135, 0.5);
    let mut worst = 0.0f64;
    for (got, want, name) in
        [(cg, 2.129810347487575, "C_gamma"), (ct, 0.408248290463863, "C_tau")] {
        let e = ((got - want) / want).abs();
        worst = worst.max(e);
        eprintln!("edc_constants_by_variant: {name} = {got}, relative error {e:.3e}");
        assert!(e <= 1e-14, "{name} = {got} against {want}, relative error {e:.3e}");
    }
    let mut checks = |v: EdcVariant, g_star: f64| {
        let s = edc_scales(&prm(v), 1.5e-5, 1.0, 10.0);
        let rows = [
            (s.gamma_lambda, 0.23570226039551584, "gamma_lambda"),
            (s.gamma_star, g_star, "gamma_star"),
            (s.tau_star, 5.0e-4, "tau_star"),
        ];
        for (got, want, name) in rows {
            let e = ((got - want) / want).abs();
            worst = worst.max(e);
            eprintln!("edc_constants_by_variant: {v:?} {name} = {got}, relative error {e:.3e}");
            assert!(e <= 1e-14, "{v:?} {name} = {got} against {want}, relative error {e:.3e}");
        }
        assert!(!s.capped, "{v:?}: gamma_lambda must not be capped here");
    };
    checks(EdcVariant::V1989, 0.013094570021973102);
    checks(EdcVariant::V2005, 0.05555555555555555);
    eprintln!("edc_constants_by_variant: worst relative error {worst:.3e} against 1e-14");
}

/// SPEC-LIT §134.5 `edc_psr_closed_form`: one first-order irreversible
/// reaction makes (134.9) linear, so its steady state is (134.11),
/// `Y*_F = Ytilde_F / (1 + (1 - gamma*) k_r tau*)`, and the mean rate is
/// `- rho gamma* k_r Ytilde_F / (1 + (1 - gamma*) k_r tau*)`. The host twin
/// matches to 1e-8 relative in both variants, `T*` stays 1000 K to 1e-12,
/// and so does the device over 64 identical cells (1e-8 in f64, 1e-6 in
/// f32); no cell is capped, unconverged or refused.
#[test]
fn edc_psr_closed_form() {
    let m = mech(FP_SPECIES, "C H N", &["FUEL => PROD   1000.0 0.0 0.0"]);
    let opts = edc_opts();
    let (p, t_bar, rho_bar, nu, k, eps) = (101325.0, 1000.0, 1.0, 1.5e-5, 1.0, 10.0);
    let y_bar = [0.05, 0.0, 0.95];
    let mut worst = 0.0f64;
    for (v, ys_want, wd_want) in [
        (EdcVariant::V1989, 0.03347946640571598, -0.43839921714794405),
        (EdcVariant::V2005, 0.033962264150943396, -1.8867924528301887),
    ] {
        let cell = edc_host(&m, &prm(v), &opts, p, t_bar, &y_bar, rho_bar, nu, k, eps)
            .expect("the closed form integrates");
        let e = ((cell.y_star[0] - ys_want) / ys_want).abs();
        worst = worst.max(e);
        eprintln!("edc_psr_closed_form: host {v:?} Y*_F = {}, relative error {e:.3e}", cell.y_star[0]);
        assert!(e <= 1e-8, "host {v:?} Y*_F {} against {ys_want}, relative error {e:.3e}", cell.y_star[0]);
        let got = cell.wdot[0] / rho_bar;
        let e = ((got - wd_want) / wd_want).abs();
        worst = worst.max(e);
        eprintln!("edc_psr_closed_form: host {v:?} W_F wb_F/rho = {got}, relative error {e:.3e}");
        assert!(e <= 1e-8, "host {v:?} rate {got} against {wd_want}, relative error {e:.3e}");
        let e = ((cell.t_star - t_bar) / t_bar).abs();
        worst = worst.max(e);
        eprintln!("edc_psr_closed_form: host {v:?} T* = {}, |T* - Ttilde|/Ttilde {e:.3e}", cell.t_star);
        assert!(e <= 1e-12, "host {v:?} T* {} must stay 1000 K, error {e:.3e}", cell.t_star);
        assert!(cell.converged, "{v:?}: the closed form must converge");
    }
    eprintln!("edc_psr_closed_form: host worst relative error {worst:.3e}");
    let Some(g) = gpu() else { return };
    let n = 64usize;
    let ks = m.species().len();
    let kern = TciKernels::new(&g).expect("tci kernels");
    let dm = DeviceMechanism::upload(&g, &m).expect("upload");
    let dr = DeviceReactor::new(&g, &opts).expect("reactor");
    let ts = vec![t_bar as Scalar; n];
    let ps = vec![p as Scalar; n];
    let rs = vec![rho_bar as Scalar; n];
    let ns = vec![nu as Scalar; n];
    let ksbuf = vec![k as Scalar; n];
    let es = vec![eps as Scalar; n];
    let mut ys = vec![0.0 as Scalar; ks * n];
    for c in 0..n {
        for (k_i, v) in y_bar.iter().enumerate() {
            ys[k_i * n + c] = *v as Scalar;
        }
    }
    for (v, ys_want, wd_want) in [
        (EdcVariant::V1989, 0.03347946640571598, -0.43839921714794405),
        (EdcVariant::V2005, 0.033962264150943396, -1.8867924528301887),
    ] {
        let mut work = EdcWork::new(&g, &dm, &dr, n).expect("work");
        let mut wdot = g.zeros::<Scalar>(ks * n).expect("alloc");
        let mut st = g.zeros::<i32>(n).expect("alloc");
        let (dt, dp, drho, dnu, dk, de, dy) = (
            g.upload(&ts).expect("upload"),
            g.upload(&ps).expect("upload"),
            g.upload(&rs).expect("upload"),
            g.upload(&ns).expect("upload"),
            g.upload(&ksbuf).expect("upload"),
            g.upload(&es).expect("upload"),
            g.upload(&ys).expect("upload"),
        );
        let mix = DeviceMixing::Epsilon { k: &dk, epsilon: &de };
        edc_source_device(
            &g, &kern, &dm, &dr, &mut work, &prm(v), &mix, &dnu, &drho, &dt, &dp, &dy,
            &mut wdot, &mut st, n,
        )
        .expect("launch");
        g.sync().expect("sync");
        let rep = edc_report(&g, &work, &st, n).expect("report");
        let state = g.download(work.fine_state()).expect("download");
        let wd = g.download(&wdot).expect("download");
        let sts = g.download(&st).expect("download");
        assert!(sts.iter().all(|&s| s == 0), "{v:?}: statuses {sts:?}");
        for c in 0..n {
            let got = state[c] as f64;
            let e = ((got - ys_want) / ys_want).abs();
            worst = worst.max(e);
            if c == 0 {
                eprintln!("edc_psr_closed_form: dev {v:?} cell {c} Y*_F = {got}, relative error {e:.3e}");
            }
            assert!(e <= edc_dev_tol(), "dev {v:?} cell {c} Y*_F {got} against {ys_want}, error {e:.3e}");
            let got = wd[c] as f64;
            let e = ((got - wd_want) / wd_want).abs();
            worst = worst.max(e);
            if c == 0 {
                eprintln!("edc_psr_closed_form: dev {v:?} cell {c} wdot_F = {got}, relative error {e:.3e}");
            }
            assert!(e <= edc_dev_tol(), "dev {v:?} cell {c} wdot_F {got} against {wd_want}, error {e:.3e}");
        }
        assert_eq!(rep, EdcReport::default(), "{v:?}: the report must be clean");
        eprintln!("edc_psr_closed_form: device {v:?} clean; worst so far {worst:.3e}");
    }
    eprintln!("edc_psr_closed_form: worst relative error {worst:.3e}; tolerance {:e}", edc_dev_tol());
}

/// SPEC-LIT §134.5 `edc_cap_is_counted`: 64 cells with `k = 0.01 + 0.003 c`
/// at `nu = 1.5e-5, eps = 10`; the 18 cells with `k < 0.0616` have
/// `gamma_lambda > 0.95`, the host counts them and each capped cell's
/// `gamma_star` is `0.95^3` (1989) to 1e-14; the device's CAPPED count and
/// its per-cell bits match the host exactly, with nothing refused.
#[test]
fn edc_cap_is_counted() {
    let m = mech(FP_SPECIES, "C H N", &["FUEL => PROD   1000.0 0.0 0.0"]);
    let v = EdcVariant::V1989;
    let (nu, eps) = (1.5e-5, 10.0);
    let n = 64usize;
    let ks = m.species().len();
    let ks_f: Vec<f64> = (0..n).map(|c| 0.01 + 0.003 * c as f64).collect();
    let capped: Vec<bool> =
        ks_f.iter().map(|&k_c| edc_scales(&prm(v), nu, k_c, eps).capped).collect();
    assert_eq!(capped.iter().filter(|&&c| c).count(), 18, "18 cells must exceed the cap");
    for (c, &cap) in capped.iter().enumerate() {
        if cap {
            let s = edc_scales(&prm(v), nu, ks_f[c], eps);
            let e = ((s.gamma_star - 0.95f64.powi(3)) / 0.95f64.powi(3)).abs();
            eprintln!("edc_cap_is_counted: cell {c} capped, gamma_star = {}, error {e:.3e}", s.gamma_star);
            assert!(e <= 1e-14, "cell {c}: gamma_star {} against 0.95^3, error {e:.3e}", s.gamma_star);
        }
    }
    eprintln!("edc_cap_is_counted: host count 18, capped gamma_star = 0.95^3 to 1e-14");
    let Some(g) = gpu() else { return };
    let kern = TciKernels::new(&g).expect("tci kernels");
    let dm = DeviceMechanism::upload(&g, &m).expect("upload");
    let dr = DeviceReactor::new(&g, &edc_opts()).expect("reactor");
    let mut work = EdcWork::new(&g, &dm, &dr, n).expect("work");
    let y_bar = [0.05, 0.0, 0.95];
    let mut ys = vec![0.0 as Scalar; ks * n];
    for c in 0..n {
        for (k_i, v) in y_bar.iter().enumerate() {
            ys[k_i * n + c] = *v as Scalar;
        }
    }
    let dnu = g.upload(&vec![nu as Scalar; n]).expect("upload");
    let drho = g.upload(&vec![1.0 as Scalar; n]).expect("upload");
    let dt = g.upload(&vec![1000.0 as Scalar; n]).expect("upload");
    let dp = g.upload(&vec![101325.0 as Scalar; n]).expect("upload");
    let dk = g.upload(&(ks_f.iter().map(|&x| x as Scalar).collect::<Vec<_>>())).expect("upload");
    let de = g.upload(&vec![eps as Scalar; n]).expect("upload");
    let dy = g.upload(&ys).expect("upload");
    let mut wdot = g.zeros::<Scalar>(ks * n).expect("alloc");
    let mut st = g.zeros::<i32>(n).expect("alloc");
    let mix = DeviceMixing::Epsilon { k: &dk, epsilon: &de };
    edc_source_device(
        &g, &kern, &dm, &dr, &mut work, &prm(v), &mix, &dnu, &drho, &dt, &dp, &dy, &mut wdot,
        &mut st, n,
    )
    .expect("launch");
    g.sync().expect("sync");
    let rep = edc_report(&g, &work, &st, n).expect("report");
    let flags = g.download(work.flags()).expect("download");
    eprintln!(
        "edc_cap_is_counted: device report capped {}, unconverged {}, refused {}",
        rep.capped, rep.unconverged, rep.refused
    );
    assert_eq!(rep.capped, 18, "the device must count the same 18 capped cells");
    assert_eq!(rep.refused, 0, "no cell may be refused");
    for (c, &cap) in capped.iter().enumerate() {
        assert_eq!(
            flags[c] & ofgpu::chem::tci::edc_flag::CAPPED != 0,
            cap,
            "cell {c}: the CAPPED bit must sit exactly on the host-capped cells"
        );
    }
    eprintln!("edc_cap_is_counted: the device's CAPPED bits sit exactly on the host's 18 cells");
}

/// SPEC-LIT §134.5 `edc_refuses_batch_variant`: integrating the fine
/// structures as a batch reactor is Fluent's variant, not (134.9)'s PSR,
/// so the lowering refuses it, naming `batch` and §134.3's reason.
#[test]
fn edc_refuses_batch_variant() {
    const SPECIES: &str = r#"{ "names": ["FUEL", "PROD", "N2"], "inert": "N2" }"#;
    const CHEMISTRY: &str = r#"{ "mechanism": "mech.inp", "thermo": "therm.dat" }"#;
    let sp: JsonSpecies = serde_json::from_str(SPECIES).unwrap();
    let ch: JsonChemistry = serde_json::from_str(CHEMISTRY).unwrap();
    let co: JsonCombustion =
        serde_json::from_str(r#"{ "model": "EDC", "variant": "batch" }"#).unwrap();
    let blocks = ChemBlocks { species: Some(&sp), chemistry: Some(&ch), combustion: Some(&co) };
    match lower_blocks(blocks, Some(MixingKind::Epsilon), false, None) {
        Err(Error::Refused(msg)) => {
            assert!(msg.contains("batch") && msg.contains("134.9"), "{msg}");
            eprintln!("edc_refuses_batch_variant: refused with: {msg}");
        }
        other => panic!("variant batch: expected a refusal, got {other:?}"),
    }
}

/// SPEC-LIT §134.5 `edc_psr_holds_mean_enthalpy`: the fine-structure
/// reactor is adiabatic and isobaric, so (134.16) keeps `h(T*, Y*) = h~`,
/// which for the exothermic fixture means `T* > T~`. On the host to 1e-8
/// of the mean enthalpy scale; over 256 random cells the device matches
/// the host twin fed the values the device saw, to `edc_dev_tol()` of the
/// cell's largest mean rate and of `T~`, statuses 0, and the capped and
/// unconverged counts agree.
#[test]
fn edc_psr_holds_mean_enthalpy() {
    let m = mech_h(
        &[
            ("FUEL", &[("C", 1), ("H", 4)], 0.0),
            ("PROD", &[("C", 1), ("H", 4)], -5000.0),
            ("N2", &[("N", 2)], 0.0),
        ],
        "C H N",
        &["FUEL => PROD   1.0E+06 0.0 10000.0"],
    );
    let opts = edc_opts();
    let (p, t_bar, nu, k, eps) = (101325.0, 1200.0, 1.5e-5, 1.0, 10.0);
    let y_bar = [0.05, 0.0, 0.95];
    let cell = edc_host(&m, &prm(EdcVariant::V1989), &opts, p, t_bar, &y_bar, 1.0, nu, k, eps)
        .expect("the exothermic cell integrates");
    let thermo = m.thermo();
    let scale: f64 = thermo
        .species()
        .iter()
        .zip(y_bar)
        .map(|(s, y)| y * s.h(t_bar).expect("h").abs() / s.molar_mass)
        .sum();
    let dh = (thermo.h_mass(cell.t_star, &cell.y_star).expect("h*")
        - thermo.h_mass(t_bar, &y_bar).expect("h~"))
    .abs();
    eprintln!(
        "edc_psr_holds_mean_enthalpy: host |h* - h~| = {dh:.3e}, scale = {scale:.3e}, T* = {}, residual = {:.3e}",
        cell.t_star, cell.residual
    );
    assert!(dh <= 1e-8 * scale, "|h(T*, Y*) - h~| = {dh:.3e} exceeds 1e-8 x {scale:.3e}");
    assert!(cell.t_star > t_bar, "T* = {} must exceed T~ = {t_bar}", cell.t_star);
    assert!(cell.converged, "the exothermic cell must converge");
    let Some(g) = gpu() else { return };
    let n = 256usize;
    let ks = m.species().len();
    let mut rng = SplitMix64(20261017);
    let (mut ts, mut rhos, mut kf, mut es) =
        (Vec::with_capacity(n), Vec::with_capacity(n), Vec::with_capacity(n), Vec::with_capacity(n));
    let mut ys = vec![0.0 as Scalar; ks * n];
    for c in 0..n {
        let y_f = 0.01 + 0.07 * rng.u();
        let y_p = 0.02 * rng.u();
        for (k_i, v) in [y_f, y_p, 1.0 - y_f - y_p].iter().enumerate() {
            ys[k_i * n + c] = *v as Scalar;
        }
        ts.push((900.0 + 600.0 * rng.u()) as Scalar);
        rhos.push((0.2 + 1.8 * rng.u()) as Scalar);
        kf.push((0.2 + 4.8 * rng.u()) as Scalar);
        es.push((1.0 + 999.0 * rng.u()) as Scalar);
    }
    let kern = TciKernels::new(&g).expect("tci kernels");
    let dm = DeviceMechanism::upload(&g, &m).expect("upload");
    let dr = DeviceReactor::new(&g, &opts).expect("reactor");
    let mut work = EdcWork::new(&g, &dm, &dr, n).expect("work");
    let dnu = g.upload(&vec![nu as Scalar; n]).expect("upload");
    let drho = g.upload(&rhos).expect("upload");
    let dt = g.upload(&ts).expect("upload");
    let dp = g.upload(&vec![p as Scalar; n]).expect("upload");
    let dk = g.upload(&kf).expect("upload");
    let de = g.upload(&es).expect("upload");
    let dy = g.upload(&ys).expect("upload");
    let mut wdot = g.zeros::<Scalar>(ks * n).expect("alloc");
    let mut st = g.zeros::<i32>(n).expect("alloc");
    let mix = DeviceMixing::Epsilon { k: &dk, epsilon: &de };
    edc_source_device(
        &g, &kern, &dm, &dr, &mut work, &prm(EdcVariant::V2005), &mix, &dnu, &drho, &dt, &dp,
        &dy, &mut wdot, &mut st, n,
    )
    .expect("launch");
    g.sync().expect("sync");
    let rep = edc_report(&g, &work, &st, n).expect("report");
    let state = g.download(work.fine_state()).expect("download");
    let wd = g.download(&wdot).expect("download");
    let sts = g.download(&st).expect("download");
    let (mut worst_w, mut worst_t, mut wc) = (0.0f64, 0.0f64, 0usize);
    let (mut host_capped, mut host_unc) = (0usize, 0usize);
    let mut worst_res = (0.0f64, 0usize, 0.0f64);
    for c in 0..n {
        assert_eq!(sts[c], 0, "cell {c} refused with status {}", sts[c]);
        let cell = edc_host(
            &m,
            &prm(EdcVariant::V2005),
            &opts,
            101325.0 as Scalar as f64,
            ts[c] as Scalar as f64,
            &(0..ks).map(|k_i| ys[k_i * n + c] as Scalar as f64).collect::<Vec<_>>(),
            rhos[c] as Scalar as f64,
            nu as Scalar as f64,
            kf[c] as Scalar as f64,
            es[c] as Scalar as f64,
        )
        .expect("the host twin integrates the device's inputs");
        if cell.scales.capped {
            host_capped += 1;
        }
        if !cell.converged {
            host_unc += 1;
        }
        if cell.residual > worst_res.0 {
            worst_res = (cell.residual, c, cell.scales.tau_star * (1.0 - cell.scales.gamma_star));
        }
        let scale = cell.wdot.iter().fold(0.0f64, |a, v| a.max(v.abs()));
        for k_i in 0..ks {
            let host = cell.wdot[k_i];
            let dev = wd[k_i * n + c] as f64;
            let e = if scale == 0.0 {
                assert_eq!(dev.to_bits(), (host as Scalar as f64).to_bits(), "cell {c} species {k_i}");
                0.0
            } else {
                (dev - host).abs() / scale
            };
            if e > worst_w {
                worst_w = e;
                wc = c;
            }
        }
        let e = ((state[ks * n + c] as f64 - cell.t_star).abs()) / (ts[c] as Scalar as f64);
        if e > worst_t {
            worst_t = e;
        }
    }
    eprintln!(
        "edc_psr_holds_mean_enthalpy: worst rate error {worst_w:.3e} at cell {wc}, worst \
         |T*_dev - T*_host|/T~ {worst_t:.3e}; tolerance {:e}",
        edc_dev_tol()
    );
    eprintln!(
        "edc_psr_holds_mean_enthalpy: capped host {} device {}, unconverged host {} device {}",
        host_capped, rep.capped, host_unc, rep.unconverged
    );
    eprintln!(
        "edc_psr_holds_mean_enthalpy: largest host residual (134.17) {:.3e} at cell {} (tau_e {:.3e} s, T~ {} K)",
        worst_res.0, worst_res.1, worst_res.2, ts[worst_res.1]
    );
    assert!(worst_w <= edc_dev_tol(), "worst rate error {worst_w:.3e} at cell {wc}");
    assert!(worst_t <= edc_dev_tol(), "worst T* error {worst_t:.3e}");
    assert_eq!(rep.capped, host_capped, "the capped counts must agree");
    assert_eq!(rep.unconverged, host_unc, "the unconverged counts must agree");
    assert_eq!(rep.refused, 0, "no cell may be refused");
}
