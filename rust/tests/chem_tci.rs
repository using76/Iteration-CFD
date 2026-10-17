// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.
// Provenance: see PROVENANCE.md. No GPL-licensed source was consulted.

//! CHR-14's gates (SPEC-LIT §134.5, §134.6): EDM's closed-form box
//! (`edm_box_closed_form`), the two-step mechanism's element conservation
//! (`edm_two_step_conserves_elements`), the refusal of a case without `A`
//! (`edm_requires_a_and_b`), PaSR's two limits (`pasr_limit_is_laminar`
//! bitwise, `pasr_linear_kappa_exact` at 1e-12), the selector's refusals
//! (`tci_refuses_no_mixing_rate`) and the device kernels `chemEdmSource`
//! and `chemPasrScale` against the f64 host twins
//! (`tci_device_matches_host`).
//!
//! An integration test, so every row runs in BOTH builds: the kernels
//! compute in double in both (SPEC-LIT §132.6, §134.6), and the f32 build
//! holds the host to the f32-rounded inputs the device saw. The GPU tests
//! return early when the box has no card. The mechanisms are built from
//! text over tests/case_chem.rs's format-valid THERMO `record`.
//! No GPL-licensed source was consulted.

#![allow(clippy::excessive_precision)]

use ofgpu::chem::device::{production_rates, ChemKernels, ChemWork, DeviceMechanism};
use ofgpu::chem::mechanism::Mechanism;
use ofgpu::chem::rosenbrock::{integrate, RosOptions};
use ofgpu::chem::thermo::ThermoSet;
use ofgpu::chem::tci::{
    edm_source, kappa, pasr_source, pasr_source_device, select, select_for, tau_chem, tau_mix,
    Closure, DeviceEdm, DeviceMixing, EdmMechanism, TciKernels,
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
        Err(Error::Config(msg)) => assert!(msg.contains("CHR-15"), "{msg}"),
        other => panic!("EDC: expected the not-built refusal, got {other:?}"),
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
