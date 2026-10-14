// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.
// Provenance: see PROVENANCE.md. No GPL-licensed source was consulted.

//! CHR-08's gates: the mixture-averaged transport of SPEC-LIT §133.4 - the
//! TM-4513 eq. (10) fits of `reference/mcbride-gordon-reno1993/
//! table_IV.csv`, the Wilke mixture viscosity (133.8)+(133.10), the
//! mixture conductivity (133.9)+(133.10)+(133.11), the Lewis route, and
//! the device twins of `cuda/chem.cu` against the f64 host. The air gate
//! is Kadoya, Matsunaga & Nagashima 1985 (2 % on mu, 5 % on lambda at the
//! 0.1 MPa row), taken over the FULL ten-species TM-4513 thermo set with
//! the air Y (X_N2/X_O2 = 0.79/0.21) and Y zero elsewhere - the species a
//! zero Y leaves out are not evaluated, like `ThermoSet::cp_mass`.
//!
//! An integration test, so it runs in both builds: the host module is f64
//! in every build, and the device kernels compute in double in every
//! build, the Scalar crossing rounded once (§133.4).
//! No GPL-licensed source was consulted.

#![allow(clippy::excessive_precision)]

use ofgpu::chem::device::{
    status, transport_lewis, transport_mixture, transport_prandtl, ChemKernels, DeviceLewis,
    DeviceTransport, TransportWork,
};
use ofgpu::chem::thermo::{SpeciesThermo, ThermoSet};
use ofgpu::chem::transport::{
    lewis_diffusivity, prandtl_conductivity, read_table_iv_csv, Lewis, SpeciesTransport,
    TransportFit, TransportSet,
};
use ofgpu::chem::R_GAS;
use ofgpu::device::Gpu;
use ofgpu::{Error, Scalar};

/// The process's device, if the box has one: every GPU test returns early
/// rather than failing.
fn gpu() -> Option<Gpu> {
    Gpu::new(0).ok()
}

/// SplitMix64, seeded per test; `u()` is uniform on [0, 1) as (x >> 11) 2^-53.
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

/// The ten TM-4513 thermo records of `reference/mcbride-gordon-reno1993/
/// table_II.csv` (SPEC-LIT §128.6, §130.7).
const TABLE_II: &str = include_str!("../../reference/mcbride-gordon-reno1993/table_II.csv");

/// The ten TM-4513 transport records (SPEC-LIT §133.4).
const TABLE_IV: &str = include_str!("../../reference/mcbride-gordon-reno1993/table_IV.csv");

/// Kadoya, Matsunaga & Nagashima 1985: Table 7 (the eq. (3) constants) and
/// Table 11 (the eq. (5) constants).
const TABLE_7: &str = include_str!("../../reference/kadoya1985/table_7.csv");
const TABLE_11: &str = include_str!("../../reference/kadoya1985/table_11.csv");

/// The printed 0.1 MPa rows of Table 8 (eta) and Table 12 (lambda).
const TABLES_8_12: &str = include_str!("../../reference/kadoya1985/tables_8_12.csv");

/// The table's species codes: name and elements (§128.6's code list).
#[allow(clippy::type_complexity)]
const CODES: &[(&str, &str, &[(&str, u32)])] = &[
    ("1", "N2", &[("N", 2)]),
    ("2", "O2", &[("O", 2)]),
    ("5", "CH4", &[("C", 1), ("H", 4)]),
    ("6", "CO2", &[("C", 1), ("O", 2)]),
    ("7", "H2O", &[("H", 2), ("O", 1)]),
    ("8", "CO", &[("C", 1), ("O", 1)]),
    ("9", "H2", &[("H", 2)]),
    ("10", "OH", &[("O", 1), ("H", 1)]),
    ("11", "O", &[("O", 1)]),
    ("12", "H", &[("H", 1)]),
];

/// The records as a set at `P_ATM`, as tests/chem_device.rs's `tm_set`:
/// the bounds are 200/1000/6000 K for all ten species.
fn tm_set() -> ThermoSet {
    let mut rows: Vec<(String, f64, f64, [f64; 7])> = Vec::new();
    for line in TABLE_II.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') || line.starts_with("species") {
            continue;
        }
        let c: Vec<&str> = line.split(',').collect();
        rows.push((
            c[0].to_string(),
            c[1].parse().unwrap(),
            c[3].parse().unwrap(),
            [
                c[4].parse().unwrap(),
                c[5].parse().unwrap(),
                c[6].parse().unwrap(),
                c[7].parse().unwrap(),
                c[8].parse().unwrap(),
                c[9].parse().unwrap(),
                c[10].parse().unwrap(),
            ],
        ));
    }
    let mut v = Vec::new();
    for (code, name, els) in CODES {
        let pick = |t_lo: f64| {
            rows.iter()
                .find(|(c, t, _, _)| c == code && *t == t_lo)
                .map(|r| (r.2, r.3))
                .unwrap()
        };
        let (mol_weight, upper) = pick(1000.0);
        let (_, lower) = pick(200.0);
        v.push(SpeciesThermo {
            name: name.to_string(),
            elements: els.iter().map(|(s, n)| (s.to_string(), *n)).collect(),
            molar_mass: mol_weight / 1000.0,
            t_low: 200.0,
            t_mid: 1000.0,
            t_high: 6000.0,
            upper,
            lower,
            line: 0,
        });
    }
    ThermoSet::from_species(v, ofgpu::chem::P_ATM, "TM-4513").unwrap()
}

/// The transport records as read: ten species, two viscosity and two
/// conductivity fits each.
fn records() -> Vec<SpeciesTransport> {
    read_table_iv_csv(TABLE_IV, "table_IV.csv").unwrap()
}

/// The transport set over the FULL ten-species thermo set; the air tests
/// carry Y zero outside N2 and O2, and `mixture_ty` does not evaluate a
/// zero-Y species.
fn air_set() -> TransportSet {
    TransportSet::for_thermo(&tm_set(), &records()).unwrap()
}

/// Dry air's mass fractions from X_N2 = 0.79, X_O2 = 0.21: Y_k = X_k W_k /
/// Σ X_j W_j, with W from tm_set.
fn air_y() -> Vec<f64> {
    let thermo = tm_set();
    let wn = thermo.species()[0].molar_mass;
    let wo = thermo.species()[1].molar_mass;
    let s = 0.79 * wn + 0.21 * wo;
    let mut y = vec![0.0; thermo.species().len()];
    y[0] = 0.79 * wn / s;
    y[1] = 0.21 * wo / s;
    y
}

/// One Kadoya property's constants: T* (K), rho* (kg/m^3), the scale (H in
/// 1e-6 Pa s or Lambda in 1e-3 W/(m K)), and the (exponent, coefficient)
/// pairs of the T_r series and the rho_r series.
struct Kadoya {
    t_star: f64,
    rho_star: f64,
    scale: f64,
    expo: Vec<(f64, f64)>,
    dens: Vec<(f64, f64)>,
}

/// Parses table_7.csv's or table_11.csv's format.
fn kadoya(text: &str) -> Kadoya {
    let mut k = Kadoya {
        t_star: 0.0,
        rho_star: 0.0,
        scale: 0.0,
        expo: Vec::new(),
        dens: Vec::new(),
    };
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') || line.starts_with("kind") {
            continue;
        }
        let c: Vec<&str> = line.split(',').collect();
        let kind: i32 = c[0].parse().unwrap();
        let index: f64 = c[1].parse().unwrap();
        let value: f64 = c[2].parse().unwrap();
        match kind {
            0 => match index as i64 {
                1 => k.t_star = value,
                2 => k.rho_star = value,
                _ => k.scale = value,
            },
            1 => k.expo.push((index, value)),
            _ => k.dens.push((index, value)),
        }
    }
    k
}

/// Eq. (3)/(5)'s bracket at T on the 0.1 MPa isobar:
/// `Σ A_i T_r^i + Σ B_i rho_r^i`, with `rho = p M/(R_GAS T)`,
/// p = 0.1e6 Pa and M = 28.9644e-3 kg/mol. Multiply by the scale and its
/// unit (1e-6 for eta, 1e-3 for lambda).
fn kadoya_sum(k: &Kadoya, t: f64) -> f64 {
    let rho = 0.1e6 * 28.9644e-3 / (R_GAS * t);
    let tr = t / k.t_star;
    let rr = rho / k.rho_star;
    let mut e = 0.0;
    for (i, a) in &k.expo {
        e += a * tr.powf(*i);
    }
    for (i, b) in &k.dens {
        e += b * rr.powf(*i);
    }
    e
}

/// The printed 0.1 MPa rows: (t, eta in 1e-6 Pa s, lambda in 1e-3 W/(m K)).
fn printed_rows() -> Vec<(f64, f64, f64)> {
    let mut v = Vec::new();
    for line in TABLES_8_12.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') || line.starts_with('t') {
            continue;
        }
        let c: Vec<&str> = line.split(',').collect();
        v.push((
            c[0].parse().unwrap(),
            c[1].parse().unwrap(),
            c[2].parse().unwrap(),
        ));
    }
    v
}

/// Class R's rule: the f64 build gates at 1e-12; under `single` the
/// tolerance widens by at most 2^29 but never past 1e-5.
fn device_tol() -> f64 {
    if cfg!(feature = "single") {
        (1.0e-12 * (1u64 << 29) as f64).min(1.0e-5).max(1.0e-12)
    } else {
        1.0e-12
    }
}

/// The reader: ten records in file order, two viscosity and two
/// conductivity fits each, and the eq. (10) oracle values (1e-9 relative).
#[test]
fn transport_table_iv_reads() {
    let r = records();
    let names: Vec<&str> = r.iter().map(|s| s.name.as_str()).collect();
    assert_eq!(names, ["N2", "O2", "CH4", "CO2", "H2O", "CO", "H2", "OH", "O", "H"]);
    for sp in &r {
        assert_eq!(sp.viscosity.len(), 2, "{}", sp.name);
        assert_eq!(sp.conductivity.len(), 2, "{}", sp.name);
    }
    let n2 = &r[0];
    let o2 = &r[1];
    let checks = [
        (n2.viscosity_at(300.0).unwrap(), 1.7938539082e-05, "N2 mu(300)"),
        (n2.conductivity_at(300.0).unwrap(), 2.5726943698e-02, "N2 lambda(300)"),
        (o2.viscosity_at(1500.0).unwrap(), 6.4125500576e-05, "O2 mu(1500)"),
        (o2.conductivity_at(1500.0).unwrap(), 1.0125632914e-01, "O2 lambda(1500)"),
    ];
    let mut worst = 0.0f64;
    for (v, oracle, what) in checks {
        let rel = (v / oracle - 1.0).abs();
        worst = worst.max(rel);
        println!("  {what}: {v:.10e} vs {oracle:.10e} (rel {rel:.2e})");
    }
    println!("  table IV reads: worst rel {worst:.3e}");
    assert!(worst <= 1.0e-9, "worst rel {worst:.3e}");
}

/// The air viscosity gate (class P): (133.8)+(133.10) on X_N2/X_O2 =
/// 0.79/0.21 against Kadoya's eq. (3) over 300..1500 K and against every
/// printed Table 8 row at 300..1000 K, within 2 %.
#[test]
fn transport_air_viscosity_vs_kadoya() {
    let set = air_set();
    let y = air_y();
    let k7 = kadoya(TABLE_7);
    let mut worst = 0.0f64;
    let mut mu300 = 0.0;
    for i in 0..25 {
        let t = 300.0 + 50.0 * i as f64;
        let mu = set.mixture_ty(t, &y).unwrap().viscosity;
        if t == 300.0 {
            mu300 = mu;
        }
        let reference = k7.scale * kadoya_sum(&k7, t) * 1.0e-6;
        worst = worst.max((mu / reference - 1.0).abs());
    }
    for (t, eta, _lam) in printed_rows() {
        if t < 300.0 {
            continue;
        }
        let mu = set.mixture_ty(t, &y).unwrap().viscosity;
        worst = worst.max((mu / (eta * 1.0e-6) - 1.0).abs());
    }
    println!(
        "  air viscosity vs Kadoya: worst |mu_mix/ref - 1| = {worst:.3e}; mu_mix(300 K) = {mu300:.4e} Pa s"
    );
    assert!(worst <= 0.02, "worst {worst:.3e} > 0.02");
    assert!((mu300 / 1.8505e-05 - 1.0).abs() <= 0.005);
}

/// The air conductivity gate (class P): (133.9)+(133.10)+(133.11) against
/// Kadoya's eq. (5) over 300..1500 K and against every printed Table 12
/// row at 300..1000 K, within 5 %.
#[test]
fn transport_air_conductivity_vs_kadoya() {
    let set = air_set();
    let y = air_y();
    let k11 = kadoya(TABLE_11);
    let mut worst = 0.0f64;
    let mut lam300 = 0.0;
    for i in 0..25 {
        let t = 300.0 + 50.0 * i as f64;
        let lam = set.mixture_ty(t, &y).unwrap().conductivity;
        if t == 300.0 {
            lam300 = lam;
        }
        let reference = k11.scale * kadoya_sum(&k11, t) * 1.0e-3;
        worst = worst.max((lam / reference - 1.0).abs());
    }
    for (t, _eta, lam) in printed_rows() {
        if t < 300.0 {
            continue;
        }
        let lam_mix = set.mixture_ty(t, &y).unwrap().conductivity;
        worst = worst.max((lam_mix / (lam * 1.0e-3) - 1.0).abs());
    }
    println!(
        "  air conductivity vs Kadoya: worst |lambda_mix/ref - 1| = {worst:.3e}; lambda_mix(300 K) = {lam300:.4e} W/(m K)"
    );
    assert!(worst <= 0.05, "worst {worst:.3e} > 0.05");
    assert!((lam300 / 2.5804e-02 - 1.0).abs() <= 0.005);
}

/// Route (b) is NAMED and NOT BUILT: both entry points refuse by name as
/// `Error::Refused`, with the fit, its authors, the read state, the
/// section and the alternative in the message.
#[test]
fn transport_route_b_named() {
    let set = air_set();
    let y = air_y();
    let errs = [
        set.binary_diffusion(0, 1, 300.0, 101325.0).unwrap_err(),
        set.mixture_diffusivities(300.0, 101325.0, &y).unwrap_err(),
    ];
    for e in errs {
        let Error::Refused(msg) = &e else {
            panic!("route (b) must refuse as Error::Refused, got {e}");
        };
        for part in [
            "route (b)",
            "Neufeld, Janzen & Aziz (1972)",
            "not read",
            "SPEC-LIT 133.4",
            "(`Lewis`)",
            "available alternative",
        ] {
            assert!(msg.contains(part), "`{part}` missing from: {msg}");
        }
    }
}

/// A refused temperature is a named refusal, on the pure species and on
/// the mixture; NaN and zero T are refused before anything is evaluated.
#[test]
fn transport_out_of_range_is_refused() {
    let r = records();
    let n2 = &r[0];
    let set = air_set();
    let y = air_y();
    let msg = format!("{}", n2.viscosity_at(299.0).unwrap_err());
    assert!(
        msg.contains("N2") && msg.contains("viscosity") && msg.contains("299"),
        "{msg}"
    );
    let msg = format!("{}", n2.conductivity_at(5000.5).unwrap_err());
    assert!(
        msg.contains("N2") && msg.contains("conductivity") && msg.contains("5000.5"),
        "{msg}"
    );
    let msg = format!("{}", set.mixture_ty(250.0, &y).unwrap_err());
    assert!(msg.contains("N2"), "{msg}");
    for t in [f64::NAN, 0.0] {
        assert!(set.mixture_ty(t, &y).is_err());
        assert!(n2.viscosity_at(t).is_err());
        assert!(n2.conductivity_at(t).is_err());
    }
}

/// A fit on [t_lo, t_hi] with the N2 300-1000 coefficients' shape.
fn fit(t_lo: f64, t_hi: f64) -> TransportFit {
    TransportFit {
        t_lo,
        t_hi,
        a: 0.60443938,
        b: -43.632704,
        c: -884.41949,
        d: 1.8972150,
    }
}

/// Every record-level and set-level refusal, and the reader's Parse error
/// naming the 1-based line of an unparsable number.
#[test]
fn transport_records_are_validated() {
    let ok = vec![fit(300.0, 5000.0)];
    // the empty viscosity list
    let msg = format!("{}", SpeciesTransport::new("N2", vec![], ok.clone()).unwrap_err());
    assert!(msg.contains("N2") && msg.contains("viscosity"), "{msg}");
    // four fits, one past the three-interval cap
    let four = vec![
        fit(300.0, 1000.0),
        fit(1000.0, 2000.0),
        fit(2000.0, 3000.0),
        fit(3000.0, 4000.0),
    ];
    assert!(SpeciesTransport::new("N2", four, ok.clone()).is_err());
    // a gap between neighbours
    let gap = vec![fit(300.0, 1000.0), fit(1100.0, 5000.0)];
    let msg = format!("{}", SpeciesTransport::new("N2", gap, ok.clone()).unwrap_err());
    assert!(msg.contains("N2") && msg.contains("not contiguous"), "{msg}");
    // t_hi <= t_lo
    assert!(SpeciesTransport::new("N2", vec![fit(1000.0, 300.0)], ok.clone()).is_err());
    // a NaN coefficient
    let nan = TransportFit {
        a: f64::NAN,
        ..fit(300.0, 5000.0)
    };
    assert!(SpeciesTransport::new("N2", vec![nan], ok.clone()).is_err());
    // the conductivity list carries the species and the property word
    let msg = format!("{}", SpeciesTransport::new("O2", ok, vec![]).unwrap_err());
    assert!(msg.contains("O2") && msg.contains("conductivity"), "{msg}");
    // for_thermo: a missing record names the species, a duplicate names it
    let thermo = tm_set();
    let mut recs = records();
    recs.retain(|r| r.name != "CO");
    let msg = format!("{}", TransportSet::for_thermo(&thermo, &recs).unwrap_err());
    assert!(msg.contains("CO"), "{msg}");
    let mut recs = records();
    recs.push(recs[0].clone());
    let msg = format!("{}", TransportSet::for_thermo(&thermo, &recs).unwrap_err());
    assert!(msg.contains("N2"), "{msg}");
    // the reader names the line of a bad number
    let text = "# a comment\n\
                species,name,property,t_lo,t_hi,a,b,c,d\n\
                1,N2,V,300.0,1000.0,0.6,bogus,-884.4,1.9\n";
    match read_table_iv_csv(text, "broken.csv") {
        Err(Error::Parse { path, msg }) => {
            assert_eq!(path, "broken.csv");
            assert!(msg.contains("line 3"), "{msg}");
        }
        other => panic!("expected a Parse error, got {other:?}"),
    }
}

/// The Lewis route's two formulas are exactly the operations SPEC-LIT 133.4
/// spells, and `Lewis::new` refuses a bad Pr or Le.
#[test]
fn transport_lewis_and_prandtl() {
    assert_eq!(
        prandtl_conductivity(1.8e-5, 1005.0, 0.7),
        (1.8e-5 * 1005.0) / 0.7
    );
    assert_eq!(
        lewis_diffusivity(0.025, 1.2, 1005.0, 0.9),
        0.025 / ((1.2 * 1005.0) * 0.9)
    );
    for (pr, le) in [(0.0, vec![1.0]), (f64::NAN, vec![1.0])] {
        assert!(Lewis::new(pr, le).is_err(), "Pr = {pr} must be refused");
    }
    assert!(Lewis::new(0.7, vec![]).is_err());
    let msg = format!("{}", Lewis::new(0.7, vec![1.0, -1.0]).unwrap_err());
    assert!(msg.contains("species 1"), "{msg}");
}

/// Y = N2 only reduces (133.8)/(133.9) to the pure species' values, and
/// the set's mole fractions are bitwise `ThermoSet::mole_fractions`.
#[test]
fn transport_mixture_reduces_to_pure_species() {
    let set = air_set();
    let thermo = tm_set();
    let mut y = vec![0.0; 10];
    y[0] = 1.0;
    for t in [300.0, 1000.0, 2500.0] {
        let mix = set.mixture_ty(t, &y).unwrap();
        let mu = set.species()[0].viscosity_at(t).unwrap();
        let lam = set.species()[0].conductivity_at(t).unwrap();
        let worst = (mix.viscosity / mu - 1.0)
            .abs()
            .max((mix.conductivity / lam - 1.0).abs());
        println!("  pure N2 at {t} K: worst rel {worst:.2e}");
        assert!(worst <= 1.0e-15, "at {t} K: {worst:.3e}");
    }
    let mut rng = SplitMix64(20261014);
    for _ in 0..100 {
        let mut y: Vec<f64> = (0..10).map(|_| rng.u()).collect();
        let s: f64 = y.iter().sum();
        for v in y.iter_mut() {
            *v /= s;
        }
        let a = set.mole_fractions(&y).unwrap();
        let b = thermo.mole_fractions(&y).unwrap();
        for (u, v) in a.iter().zip(&b) {
            assert_eq!(u.to_bits(), v.to_bits(), "mole fractions differ");
        }
    }
}

/// The device twin: 4096 random cells' mu, lambda, lambda_pr and d against
/// the f64 host on the values AS UPLOADED - the Scalar-rounded T, Y, rho
/// and cp converted back to f64 - class R's 1e-12 in the f64 build and
/// 1e-5 under `single`, every status 0.
#[test]
fn transport_device_matches_host() {
    let Some(g) = gpu() else { return };
    let set = air_set();
    let dt = DeviceTransport::upload(&g, &set).unwrap();
    let ck = ChemKernels::new(&g).unwrap();
    let le = [1.0, 1.1, 0.97, 1.4, 0.83, 1.1, 0.3, 0.75, 0.7, 0.18];
    let dl = DeviceLewis::upload(&g, &Lewis::new(0.7, le.to_vec()).unwrap()).unwrap();
    let ks = 10;
    let n = 4096;
    let mut work = TransportWork::new(&g, &dt, n).unwrap();
    let mut rng = SplitMix64(20261013);
    let mut tv = vec![0.0 as Scalar; n];
    let mut yv = vec![0.0 as Scalar; ks * n];
    let mut rhov = vec![0.0 as Scalar; n];
    let mut cpv = vec![0.0 as Scalar; n];
    for c in 0..n {
        let t = 300.0 + 2700.0 * rng.u();
        let mut y: Vec<f64> = (0..ks).map(|_| rng.u()).collect();
        if c % 10 == 0 {
            y[(c / 10) % ks] = 0.0;
        }
        let s: f64 = y.iter().sum();
        for v in y.iter_mut() {
            *v /= s;
        }
        let rho = 0.1 + 9.9 * rng.u();
        let cp = 1000.0 + 1000.0 * rng.u();
        tv[c] = t as Scalar;
        for k in 0..ks {
            yv[k * n + c] = y[k] as Scalar;
        }
        rhov[c] = rho as Scalar;
        cpv[c] = cp as Scalar;
    }
    let t = g.upload(&tv).unwrap();
    let y = g.upload(&yv).unwrap();
    let rho = g.upload(&rhov).unwrap();
    let cp = g.upload(&cpv).unwrap();
    let mut mu = g.zeros::<Scalar>(n).unwrap();
    let mut lambda = g.zeros::<Scalar>(n).unwrap();
    let mut lambda_pr = g.zeros::<Scalar>(n).unwrap();
    let mut d = g.zeros::<Scalar>(ks * n).unwrap();
    let mut st = g.zeros::<i32>(n).unwrap();
    transport_mixture(&g, &ck, &dt, &mut work, &t, &y, &mut mu, &mut lambda, &mut st, n).unwrap();
    transport_prandtl(&g, &ck, &dl, &mu, &cp, &mut lambda_pr, n).unwrap();
    transport_lewis(&g, &ck, &dl, &lambda, &rho, &cp, &mut d, n).unwrap();
    // The host reference on the values AS UPLOADED.
    let as_f64 = |v: &ofgpu::DevBuf<Scalar>| -> Vec<f64> {
        g.download(v).unwrap().into_iter().map(|x| x as f64).collect()
    };
    let t_h = as_f64(&t);
    let y_h = as_f64(&y);
    let rho_h = as_f64(&rho);
    let cp_h = as_f64(&cp);
    let mu_d: Vec<Scalar> = g.download(&mu).unwrap();
    let lam_d: Vec<Scalar> = g.download(&lambda).unwrap();
    let lamp_d: Vec<Scalar> = g.download(&lambda_pr).unwrap();
    let d_d: Vec<Scalar> = g.download(&d).unwrap();
    let st_d: Vec<i32> = g.download(&st).unwrap();
    let tol = device_tol();
    let mut worst = [0.0f64; 4];
    let mut ycol = vec![0.0f64; ks];
    for c in 0..n {
        assert_eq!(st_d[c], status::OK, "cell {c} refused with {}", st_d[c]);
        for k in 0..ks {
            ycol[k] = y_h[k * n + c];
        }
        let mix = set.mixture_ty(t_h[c], &ycol).unwrap();
        worst[0] = worst[0].max((mu_d[c] as f64 / mix.viscosity - 1.0).abs());
        worst[1] = worst[1].max((lam_d[c] as f64 / mix.conductivity - 1.0).abs());
        let lp = prandtl_conductivity(mu_d[c] as f64, cp_h[c], 0.7);
        worst[2] = worst[2].max((lamp_d[c] as f64 / lp - 1.0).abs());
        for k in 0..ks {
            let dh = lewis_diffusivity(lam_d[c] as f64, rho_h[c], cp_h[c], le[k]);
            worst[3] = worst[3].max((d_d[k * n + c] as f64 / dh - 1.0).abs());
        }
    }
    println!(
        "  device transport vs host: worst rel mu = {:.3e}, lambda = {:.3e}, lambda_pr = {:.3e}, d = {:.3e} (tol {tol:.0e})",
        worst[0], worst[1], worst[2], worst[3]
    );
    for (i, w) in worst.iter().enumerate() {
        assert!(*w <= tol, "output {i} worst {w:.3e} > {tol:.3e}");
    }
}

/// The per-cell refusals, in the host's order of checks: NaN T -> 2,
/// T -1 -> 2, a negative Y -> 3, no species present -> 4, T 250 and
/// T 6000 -> 10 outside the fits' range, and a valid cell -> 0 with the
/// host's values; a refused cell's mu and lambda are NaN.
#[test]
fn transport_device_refuses_by_status() {
    let Some(g) = gpu() else { return };
    let set = air_set();
    let dt = DeviceTransport::upload(&g, &set).unwrap();
    let ck = ChemKernels::new(&g).unwrap();
    let ks = 10;
    let n = 7;
    let mut work = TransportWork::new(&g, &dt, n).unwrap();
    let air = air_y();
    let mut tv = vec![0.0 as Scalar; n];
    let mut yv = vec![0.0 as Scalar; ks * n];
    // c0 NaN, c1 -1, c2 a negative Y, c3 all zero, c4 250 K, c5 6000 K,
    // c6 a valid air cell at 1000 K.
    tv[0] = f64::NAN as Scalar;
    tv[1] = -1.0;
    tv[2] = 1000.0;
    tv[3] = 1000.0;
    tv[4] = 250.0;
    tv[5] = 6000.0;
    tv[6] = 1000.0;
    for c in [2usize, 4, 5, 6] {
        for k in 0..ks {
            yv[k * n + c] = air[k] as Scalar;
        }
    }
    yv[5 * n + 2] = -0.1;
    let t = g.upload(&tv).unwrap();
    let y = g.upload(&yv).unwrap();
    let mut mu = g.zeros::<Scalar>(n).unwrap();
    let mut lambda = g.zeros::<Scalar>(n).unwrap();
    let mut st = g.zeros::<i32>(n).unwrap();
    transport_mixture(&g, &ck, &dt, &mut work, &t, &y, &mut mu, &mut lambda, &mut st, n).unwrap();
    let mu_d: Vec<Scalar> = g.download(&mu).unwrap();
    let lam_d: Vec<Scalar> = g.download(&lambda).unwrap();
    let st_d: Vec<i32> = g.download(&st).unwrap();
    let want = [
        status::BAD_TEMPERATURE,
        status::BAD_TEMPERATURE,
        status::BAD_MASS_FRACTION,
        status::NO_SPECIES,
        status::OUTSIDE_TRANSPORT_RANGE,
        status::OUTSIDE_TRANSPORT_RANGE,
        status::OK,
    ];
    assert_eq!(st_d, want);
    for c in 0..n - 1 {
        assert!(mu_d[c].is_nan() && lam_d[c].is_nan(), "cell {c} is not NaN");
    }
    // The valid cell against the host, on the values as uploaded.
    let tol = device_tol();
    let t_h = g.download(&t).unwrap()[6] as f64;
    let y_up = g.download(&y).unwrap();
    let y_h: Vec<f64> = (0..ks).map(|k| y_up[k * n + 6] as f64).collect();
    let mix = set.mixture_ty(t_h, &y_h).unwrap();
    let rel = (mu_d[6] as f64 / mix.viscosity - 1.0)
        .abs()
        .max((lam_d[6] as f64 / mix.conductivity - 1.0).abs());
    println!("  device refusals: the valid cell's worst rel {rel:.3e} (tol {tol:.0e})");
    assert!(rel <= tol, "valid cell rel {rel:.3e} > {tol:.3e}");
}
