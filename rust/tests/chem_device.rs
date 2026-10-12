// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.
// Provenance: see PROVENANCE.md. No GPL-licensed source was consulted.

//! CHR-07a's gates: the device kernels of SPEC-LIT §132.9
//! (`cuda/chem.cu`, `src/chem/device.rs`) against the f64 host twins -
//! `production_rates_tpy` of §130.7, and `ConstPressureReactor::rhs` with
//! `fd_jacobian` of §132.3/§132.8. 4096 random states of §130.7's
//! generator over the ten-species TM-4513 fixture; GRI-Mech 3.0 is
//! user-supplied by path (§129.6) and skips by name when
//! `OFGPU_GRI30_MECH` or `OFGPU_GRI30_THERMO` is unset, and
//! `OFGPU_REQUIRE_MECH=1` turns the skip into a failure.
//!
//! An integration test, so it runs in both builds: the kernels compute in
//! double in both (§132.6). The omega gate's f32 build rounds the inputs
//! to f32 and holds the host to the rounded values too; the Jacobian's
//! buffers are double in both, so its tolerances do not move.
//! No GPL-licensed source was consulted.

#![allow(clippy::excessive_precision)]

use std::env;
use std::fs;

use ofgpu::chem::device::{
    production_rates, reactor_jacobian, status, ChemKernels, ChemWork, DeviceMechanism,
};
use ofgpu::chem::kinetics::{production_rates_tpy, rates_of_progress};
use ofgpu::chem::mechanism::Mechanism;
use ofgpu::chem::rosenbrock::fd_jacobian;
use ofgpu::chem::thermo::{SpeciesThermo, ThermoSet};
use ofgpu::chem::{ConstPressureReactor, ElementTable, P_ATM, R_GAS};
use ofgpu::device::Gpu;
use ofgpu::Scalar;

/// The process's device, if the box has one: every GPU test returns early
/// rather than failing.
fn gpu() -> Option<Gpu> {
    Gpu::new(0).ok()
}

/// SplitMix64, seed 20261012; `u()` is uniform on [0, 1) as (x >> 11) 2^-53.
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
/// Random state i (SPEC-LIT §130.7's generator): T = 300 + 2700 u, then
/// p = 10^(4 + 3 u), then one Y draw per species, with every 10th state's
/// Y_[(i/10) % n] zeroed, normalised to sum 1.
fn random_state(i: usize, rng: &mut SplitMix64, n: usize) -> (f64, f64, Vec<f64>) {
    let t = 300.0 + 2700.0 * rng.u();
    let p = 10f64.powf(4.0 + 3.0 * rng.u());
    let mut y: Vec<f64> = (0..n).map(|_| rng.u()).collect();
    if i % 10 == 0 {
        y[(i / 10) % n] = 0.0;
    }
    let s: f64 = y.iter().sum();
    for v in y.iter_mut() {
        *v /= s;
    }
    (t, p, y)
}

/// The ten TM-4513 records of `reference/mcbride-gordon-reno1993/
/// table_II.csv` (SPEC-LIT §128.6, §130.7).
const TABLE_II: &str = include_str!("../../reference/mcbride-gordon-reno1993/table_II.csv");

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
/// The records as a set at `P_ATM`, as tests/chem_kinetics.rs's `tm_set`:
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
    ThermoSet::from_species(v, P_ATM, "TM-4513").unwrap()
}
/// A mechanism of the given reaction lines over the ten TM-4513 species;
/// the concentration order is H2 O2 H2O OH O H CO CO2 CH4 N2 (0..9).
fn mech(rx: &[&str]) -> Mechanism {
    let mut text: Vec<&str> = vec![
        "ELEMENTS H O C N END",
        "SPECIES H2 O2 H2O OH O H CO CO2 CH4 N2 END",
        "REACTIONS",
    ];
    text.extend_from_slice(rx);
    text.push("END");
    Mechanism::read_chemkin(&text.join("\n"), "kin-fixture", Some(&tm_set()))
        .unwrap_or_else(|e| panic!("the fixture does not read: {e}"))
}

/// The 14-reaction fixture of SPEC-LIT §130.7. Its rate numbers are
/// placeholders, not any paper's.
fn fixture() -> Mechanism {
    mech(&[
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
        "  REV / 2.0E+08 1.5 23000.0 /",
        "CH4 + 2 O2 => CO2 + 2 H2O         1.0E+12   0.0  30000.0",
        "  FORD /CH4 0.2/ FORD /O2 1.3/",
        "2 H2 + O2 => 2 H2O                1.0E+10   0.0  30000.0",
        "OH + OH <=> O + H2O               3.0E+04   2.4  -2000.0",
        "  DUP",
        "OH + OH <=> O + H2O               1.0E+12   0.0   1000.0",
        "  DUP",
    ])
}
fn env_pair(mech_var: &str, thermo_var: &str, test: &str) -> Option<(String, String)> {
    match (env::var(mech_var), env::var(thermo_var)) {
        (Ok(m), Ok(t)) => Some((m, t)),
        (mech, _) => {
            let var = if mech.is_err() { mech_var } else { thermo_var };
            let why = format!(
                "SKIPPED {test}: {var} is not set; GRI-Mech 3.0 is user-supplied by path \
                 (SPEC-LIT §129.6)"
            );
            if env::var("OFGPU_REQUIRE_MECH").as_deref() == Ok("1") {
                panic!("{why}");
            }
            eprintln!("{why}");
            None
        }
    }
}

/// Reads the GRI pair, as tests/chem_kinetics.rs's `read_pair`: the thermo
/// file as its own set at `P_ATM`, then the mechanism with it.
fn read_pair(mech: &str, thermo: &str) -> Mechanism {
    let t = fs::read_to_string(thermo)
        .unwrap_or_else(|e| panic!("the thermo file {thermo} does not read: {e}"));
    let set = ThermoSet::read_chemkin(&t, thermo, P_ATM, &ElementTable::standard())
        .unwrap_or_else(|e| panic!("the thermo file {thermo} does not read: {e}"));
    let m = fs::read_to_string(mech)
        .unwrap_or_else(|e| panic!("the mechanism file {mech} does not read: {e}"));
    Mechanism::read_chemkin(&m, mech, Some(&set))
        .unwrap_or_else(|e| panic!("the mechanism file {mech} does not read: {e}"))
}

/// §132.9's omega gate: 1e-12 against the f64 host; the f32 build's plan
/// band is 1e-6, on inputs the device saw rounded to f32.
#[cfg(not(feature = "single"))]
fn omega_tol() -> f64 {
    1.0e-12
}

#[cfg(feature = "single")]
fn omega_tol() -> f64 {
    1.0e-6
}
/// The species' gross rate (§132.9's measure): `G_k = sum over reactions i
/// of nu (|q_f| + |q_r|)` over every reactant and product entry of k.
fn gross_rates(m: &Mechanism, q: &[(f64, f64)], ks: usize) -> Vec<f64> {
    let mut gross = vec![0.0f64; ks];
    for (r, (q_f, q_r)) in m.reactions().iter().zip(q) {
        let s = q_f.abs() + q_r.abs();
        for (k, nu) in &r.reactants {
            gross[*k] += nu.to_f64() * s;
        }
        for (k, nu) in &r.products {
            gross[*k] += nu.to_f64() * s;
        }
    }
    gross
}

/// The omega twin (§132.9): 4096 random states of §130.7's generator, the
/// device against the host on the inputs the device saw, in the gross
/// rate's scale. A species with G_k = 0 comes back exactly 0.
fn omega_matches_host(m: &Mechanism, label: &str) {
    let Some(g) = gpu() else { return };
    let ks = m.species().len();
    let n = 4096;
    let mut rng = SplitMix64(20261012);
    let mut ts: Vec<Scalar> = Vec::with_capacity(n);
    let mut ps: Vec<Scalar> = Vec::with_capacity(n);
    let mut ys = vec![0.0 as Scalar; ks * n];
    for i in 0..n {
        let (t, p, y) = random_state(i, &mut rng, ks);
        ts.push(t as Scalar);
        ps.push(p as Scalar);
        for (k, v) in y.iter().enumerate() {
            ys[k * n + i] = *v as Scalar;
        }
    }
    let ck = ChemKernels::new(&g).expect("kernels");
    let dm = DeviceMechanism::upload(&g, m).expect("upload");
    let mut work = ChemWork::new(&g, &dm, n).expect("work");
    let dt = g.upload(&ts).expect("upload");
    let dp = g.upload(&ps).expect("upload");
    let dy = g.upload(&ys).expect("upload");
    let mut omega = g.zeros::<Scalar>(ks * n).expect("alloc");
    let mut st = g.zeros::<i32>(n).expect("alloc");
    production_rates(&g, &ck, &dm, &mut work, &dt, &dp, &dy, &mut omega, &mut st, n)
        .expect("launch");
    g.sync().expect("sync");
    let omega_d = g.download(&omega).expect("download");
    let sts = g.download(&st).expect("download");

    let tol = omega_tol();
    let (mut worst, mut wc, mut wk, mut wabs) = (0.0f64, 0usize, 0usize, 0.0f64);
    for i in 0..n {
        assert_eq!(
            sts[i],
            status::OK,
            "{label}: cell {i} refused with status {}",
            sts[i]
        );
        // The host reference on the ROUNDED values: in the f32 build that
        // is what the device was given.
        let t = ts[i] as f64;
        let p = ps[i] as f64;
        let y_r: Vec<f64> = (0..ks).map(|k| ys[k * n + i] as f64).collect();
        let host = production_rates_tpy(m, t, p, &y_r).unwrap_or_else(|e| {
            panic!("{label}: cell {i}: the host refused what the device took: {e}")
        });
        let conc = m.thermo().concentrations(p, t, &y_r).expect("conc");
        let q: Vec<(f64, f64)> = rates_of_progress(m, t, &conc)
            .expect("rates")
            .iter()
            .map(|q_i| (q_i.forward, q_i.reverse))
            .collect();
        let gross = gross_rates(m, &q, ks);
        for k in 0..ks {
            let dev = omega_d[k * n + i] as f64;
            let e = if gross[k] == 0.0 {
                assert_eq!(
                    dev.to_bits(),
                    0.0f64.to_bits(),
                    "{label}: cell {i} species {k}: gross rate 0, device gave {dev:e}"
                );
                0.0
            } else {
                (dev - host[k]).abs() / gross[k]
            };
            if e > worst {
                worst = e;
                wc = i;
                wk = k;
                wabs = dev - host[k];
            }
        }
    }
    println!(
        "{label}: worst relative e {worst:.3e} at cell {wc} species {wk} \
         (absolute difference {wabs:.3e}); tolerance {tol:e}"
    );
    assert!(
        worst <= tol,
        "{label}: worst relative e {worst:.3e} at cell {wc} species {wk} exceeds {tol:e}"
    );
}
/// §132.9's omega twin on the 14-reaction fixture, 4096 random states.
#[test]
fn chem_device_omega_matches_host() {
    let m = fixture();
    omega_matches_host(&m, "fixture");
}

/// The same on GRI-Mech 3.0 when its two paths are set (§129.6).
#[test]
fn chem_device_omega_matches_host_gri30() {
    let Some((mech_path, thermo_path)) = env_pair(
        "OFGPU_GRI30_MECH",
        "OFGPU_GRI30_THERMO",
        "chem_device_omega_matches_host_gri30",
    ) else {
        return;
    };
    let m = read_pair(&mech_path, &thermo_path);
    omega_matches_host(&m, "GRI-30");
}
/// §132.9's Jacobian twin: 4096 random states, f0 and J against `rhs` and
/// `fd_jacobian`, each of a column's two evaluations within the
/// right-hand-side bound at the state it ran on,
/// `|J_d - J_h| delta_j <= 1e-12 (F_i(y) + F_i(y + delta_j e_j))` and
/// `|f0_d - f0_h| <= 1e-12 F_i(y)`, with `F_i = W_i G_i / rho` and
/// `F_T = sum |H_k| G_k / (rho cp)`, both builds (the inputs are double in
/// both). A zero bound is exact equality.
#[test]
fn chem_device_jacobian_matches_host() {
    let Some(g) = gpu() else { return };
    let m = fixture();
    let ks = m.species().len();
    let m1 = ks + 1;
    let n = 4096;
    let mut atol = vec![1.0e-14f64; ks];
    atol.push(1.0e-8);

    // The same 4096 draws as the omega gate, as unrounded f64: the
    // Jacobian's buffers are double in both builds.
    let mut rng = SplitMix64(20261012);
    let mut state = vec![0.0f64; m1 * n];
    let mut ps = vec![0.0f64; n];
    for i in 0..n {
        let (t, p, y) = random_state(i, &mut rng, ks);
        for (k, v) in y.iter().enumerate() {
            state[k * n + i] = *v;
        }
        state[ks * n + i] = t;
        ps[i] = p;
    }

    let ck = ChemKernels::new(&g).expect("kernels");
    let dm = DeviceMechanism::upload(&g, &m).expect("upload");
    let mut work = ChemWork::new(&g, &dm, n).expect("work");
    let dst = g.upload(&state).expect("upload");
    let dp = g.upload(&ps).expect("upload");
    let da = g.upload(&atol).expect("upload");
    let mut f0 = g.zeros::<f64>(m1 * n).expect("alloc");
    let mut jac = g.zeros::<f64>(m1 * m1 * n).expect("alloc");
    let mut st = g.zeros::<i32>(n).expect("alloc");
    reactor_jacobian(&g, &ck, &dm, &mut work, &dst, &dp, &da, &mut f0, &mut jac, &mut st, n)
        .expect("launch");
    g.sync().expect("sync");
    let f0_d = g.download(&f0).expect("download");
    let jac_d = g.download(&jac).expect("download");
    let sts = g.download(&st).expect("download");
    let sp = m.thermo().species();
    let (mut worst_f, mut worst_j) = (0.0f64, 0.0f64);
    let (mut cf, mut cj_i, mut cj_j) = (0usize, 0usize, 0usize);
    for i in 0..n {
        assert_eq!(sts[i], status::OK, "cell {i} refused with status {}", sts[i]);
        let t = state[ks * n + i];
        let p = ps[i];
        let y: Vec<f64> = (0..ks).map(|k| state[k * n + i]).collect();
        let r = ConstPressureReactor::new(&m, p).expect("reactor");
        let mut y_full = y.clone();
        y_full.push(t);
        let mut f0_h = vec![0.0f64; m1];
        r.rhs(&y_full, &mut f0_h).expect("rhs");
        let mut jac_h = vec![0.0f64; m1 * m1];
        fd_jacobian(&mut |yy, f| r.rhs(yy, f), &y_full, &f0_h, &atol, &mut jac_h)
            .expect("fd");

        // The scales, as the reactor computes rho, cp and Y+ (§132.8 R8),
        // at whatever clipped state the evaluation ran on.
        let fscale = |ypv: &[f64], t_v: f64| -> Vec<f64> {
            let s: f64 = ypv.iter().zip(sp).map(|(v, sk)| v / sk.molar_mass).sum();
            let rho = p / (R_GAS * t_v * s);
            let c: Vec<f64> =
                ypv.iter().zip(sp).map(|(v, sk)| rho * v / sk.molar_mass).collect();
            let q: Vec<(f64, f64)> = rates_of_progress(&m, t_v, &c)
                .expect("rates")
                .iter()
                .map(|q_i| (q_i.forward, q_i.reverse))
                .collect();
            let gross = gross_rates(&m, &q, ks);
            let cp = m.thermo().cp_mass(t_v, ypv).expect("cp");
            (0..m1)
                .map(|fi| {
                    if fi < ks {
                        sp[fi].molar_mass * gross[fi] / rho
                    } else {
                        (0..ks)
                            .map(|k| sp[k].h(t_v).expect("h").abs() * gross[k])
                            .sum::<f64>()
                            / (rho * cp)
                    }
                })
                .collect()
        };
        let yp: Vec<f64> = y.iter().map(|v| v.max(0.0)).collect();
        let fs_y = fscale(&yp, t);
        for fi in 0..m1 {
            let fs = fs_y[fi];
            let dev = f0_d[fi * n + i];
            if fs == 0.0 {
                assert_eq!(
                    dev.to_bits(),
                    f0_h[fi].to_bits(),
                    "cell {i} f0[{fi}]: scale 0, device {dev:e} host {}",
                    f0_h[fi]
                );
            } else {
                let e = (dev - f0_h[fi]).abs() / (1.0e-12 * fs);
                if e > worst_f {
                    worst_f = e;
                    cf = i;
                }
            }
        }
        for j in 0..m1 {
            let delta = f64::EPSILON.sqrt() * y_full[j].abs().max(atol[j]);
            // Each of the column's two evaluations is within the
            // right-hand-side bound at the state it ran on (§132.9), so the
            // derived bound is 1e-12 (F_i(y) + F_i(y + delta_j e_j)). A zero
            // gross rate at y does not make the quotient zero: a FORD order
            // on a zeroed species gives a genuine derivative there.
            let mut y_pert = y_full.clone();
            y_pert[j] += delta;
            let yp_pert: Vec<f64> = y_pert[..ks].iter().map(|v| v.max(0.0)).collect();
            let fs_p = fscale(&yp_pert, y_pert[ks]);
            for fi in 0..m1 {
                let bound = 1.0e-12 * (fs_y[fi] + fs_p[fi]);
                let dev = jac_d[(fi * m1 + j) * n + i];
                let host = jac_h[fi * m1 + j];
                if bound == 0.0 {
                    assert_eq!(dev.to_bits(), host.to_bits(), "cell {i} jac[{fi}][{j}]: scale 0");
                    continue;
                }
                let e = (dev - host).abs() * delta / bound;
                if e > worst_j {
                    worst_j = e;
                    cj_i = fi;
                    cj_j = j;
                }
            }
        }
    }
    println!(
        "  jacobian vs host: worst f0 ratio {worst_f:.3e} (cell {cf}), \
         worst J ratio {worst_j:.3e} at [{cj_i}][{cj_j}]"
    );
    assert!(worst_f <= 1.0, "f0 worst ratio {worst_f:.3e} at cell {cf}");
    assert!(
        worst_j <= 1.0,
        "J worst ratio {worst_j:.3e} at [{cj_i}][{cj_j}]"
    );
}
/// §132.9's statuses: 1-5 and the NaN outputs, each in the host twins'
/// order of checks; a clipped negative Y is not refused by the reactor.
#[test]
fn chem_device_refuses_by_status() {
    let Some(g) = gpu() else { return };
    let m = fixture();
    let ks = m.species().len();
    let m1 = ks + 1;
    let uniform: Vec<f64> = vec![1.0 / ks as f64; ks];
    let clipped = |v: f64| {
        let mut y = uniform.clone();
        y[0] = v;
        y
    };

    // ---- production rates: p, T, Y, no species, thermo range -----------
    let prod: Vec<(f64, f64, Vec<f64>, i32)> = vec![
        (100.0, 1.0e5, uniform.clone(), status::OUTSIDE_THERMO_RANGE),
        (f64::NAN, 1.0e5, uniform.clone(), status::BAD_TEMPERATURE),
        (-5.0, 1.0e5, uniform.clone(), status::BAD_TEMPERATURE),
        (1200.0, 0.0, uniform.clone(), status::BAD_PRESSURE),
        (1200.0, 1.0e5, clipped(-0.1), status::BAD_MASS_FRACTION),
        (1200.0, 1.0e5, vec![0.0; ks], status::NO_SPECIES),
        (1200.0, 1.0e5, uniform.clone(), status::OK),
    ];
    let np = prod.len();
    let (mut ts, mut ps) = (Vec::with_capacity(np), Vec::with_capacity(np));
    let mut ys = vec![0.0 as Scalar; ks * np];
    for (i, (t, p, y, _)) in prod.iter().enumerate() {
        ts.push(*t as Scalar);
        ps.push(*p as Scalar);
        for (k, v) in y.iter().enumerate() {
            ys[k * np + i] = *v as Scalar;
        }
    }
    let ck = ChemKernels::new(&g).expect("kernels");
    let dm = DeviceMechanism::upload(&g, &m).expect("upload");
    let mut work = ChemWork::new(&g, &dm, np).expect("work");
    let (dt, dp, dy) = (
        g.upload(&ts).expect("upload"),
        g.upload(&ps).expect("upload"),
        g.upload(&ys).expect("upload"),
    );
    let mut omega = g.zeros::<Scalar>(ks * np).expect("alloc");
    let mut st = g.zeros::<i32>(np).expect("alloc");
    production_rates(&g, &ck, &dm, &mut work, &dt, &dp, &dy, &mut omega, &mut st, np)
        .expect("launch");
    g.sync().expect("sync");
    let omega_d = g.download(&omega).expect("download");
    let sts = g.download(&st).expect("download");
    for (i, (t, p, y, want)) in prod.iter().enumerate() {
        assert_eq!(sts[i], *want, "production cell {i}: status {}", sts[i]);
        if *want == status::OK {
            for k in 0..ks {
                assert!(
                    omega_d[k * np + i].is_finite(),
                    "production cell {i} species {k}: a normal cell's omega is finite"
                );
            }
        } else {
            for k in 0..ks {
                assert!(
                    omega_d[k * np + i].is_nan(),
                    "production cell {i} species {k}: a refused cell writes NaN"
                );
            }
            let y_ref: Vec<f64> = y.iter().map(|v| *v as f64).collect();
            assert!(
                production_rates_tpy(&m, *t, *p, &y_ref).is_err(),
                "production cell {i}: the host twin refuses the same input"
            );
        }
    }
    // ---- the Jacobian: no Y refusal, the reactor clips ------------------
    let jac_cells: Vec<(f64, f64, Vec<f64>, i32)> = vec![
        (100.0, 1.0e5, uniform.clone(), status::OUTSIDE_THERMO_RANGE),
        (-5.0, 1.0e5, uniform.clone(), status::BAD_TEMPERATURE),
        (1200.0, 1.0e5, vec![0.0; ks], status::NO_SPECIES),
        (1200.0, 0.0, uniform.clone(), status::BAD_PRESSURE),
        (1200.0, 1.0e5, uniform.clone(), status::OK),
        (1200.0, 1.0e5, clipped(-1.0e-3), status::OK),
    ];
    let nj = jac_cells.len();
    let mut state = vec![0.0f64; m1 * nj];
    let mut pj = vec![0.0f64; nj];
    for (i, (t, p, y, _)) in jac_cells.iter().enumerate() {
        for (k, v) in y.iter().enumerate() {
            state[k * nj + i] = *v;
        }
        state[ks * nj + i] = *t;
        pj[i] = *p;
    }
    let dstate = g.upload(&state).expect("upload");
    let dp = g.upload(&pj).expect("upload");
    let mut da = vec![1.0e-14f64; ks];
    da.push(1.0e-8);
    let da = g.upload(&da).expect("upload");
    let mut f0 = g.zeros::<f64>(m1 * nj).expect("alloc");
    let mut jac = g.zeros::<f64>(m1 * m1 * nj).expect("alloc");
    let mut st = g.zeros::<i32>(nj).expect("alloc");
    reactor_jacobian(&g, &ck, &dm, &mut work, &dstate, &dp, &da, &mut f0, &mut jac, &mut st, nj)
        .expect("launch");
    g.sync().expect("sync");
    let f0_d = g.download(&f0).expect("download");
    let jac_d = g.download(&jac).expect("download");
    let sts = g.download(&st).expect("download");
    for (i, (_, _, _, want)) in jac_cells.iter().enumerate() {
        assert_eq!(sts[i], *want, "jacobian cell {i}: status {}", sts[i]);
        if *want != status::OK {
            for fi in 0..m1 {
                assert!(
                    f0_d[fi * nj + i].is_nan(),
                    "jacobian cell {i} f0[{fi}]: a refused cell writes NaN"
                );
                for fj in 0..m1 {
                    assert!(
                        jac_d[(fi * m1 + fj) * nj + i].is_nan(),
                        "jacobian cell {i} jac[{fi}][{fj}]: a refused cell writes NaN"
                    );
                }
            }
        }
    }
    // The clipped cell: Y_0 = -1e-3 is not refused, it is clipped to 0
    // (§132.8 R8), and f0 is the host rhs of the same state within the
    // right-hand-side bound.
    let i = nj - 1;
    let t = 1200.0;
    let p = 1.0e5;
    let y = clipped(-1.0e-3);
    let mut y_full = y.clone();
    y_full.push(t);
    let r = ConstPressureReactor::new(&m, p).expect("reactor");
    let mut f0_h = vec![0.0f64; m1];
    r.rhs(&y_full, &mut f0_h).expect("rhs");
    let sp = m.thermo().species();
    let yp: Vec<f64> = y.iter().map(|v| v.max(0.0)).collect();
    let s: f64 = yp.iter().zip(sp).map(|(v, sk)| v / sk.molar_mass).sum();
    let rho = p / (R_GAS * t * s);
    let c: Vec<f64> = yp.iter().zip(sp).map(|(v, sk)| rho * v / sk.molar_mass).collect();
    let q: Vec<(f64, f64)> = rates_of_progress(&m, t, &c)
        .expect("rates")
        .iter()
        .map(|q_i| (q_i.forward, q_i.reverse))
        .collect();
    let gross = gross_rates(&m, &q, ks);
    let cp = m.thermo().cp_mass(t, &yp).expect("cp");
    for fi in 0..m1 {
        let fs = if fi < ks {
            sp[fi].molar_mass * gross[fi] / rho
        } else {
            (0..ks)
                .map(|k| sp[k].h(t).expect("h").abs() * gross[k])
                .sum::<f64>()
                / (rho * cp)
        };
        let diff = (f0_d[fi * nj + i] - f0_h[fi]).abs();
        assert!(
            diff <= 1.0e-12 * fs,
            "clipped cell f0[{fi}]: |difference| {diff:e} exceeds 1e-12 F_i = {:.3e}",
            1.0e-12 * fs
        );
    }
    println!("  refusals: production and jacobian statuses and NaN outputs hold");
}
