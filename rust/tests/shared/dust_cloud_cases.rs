// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.
// Provenance: see PROVENANCE.md. No GPL-licensed source was consulted.

// The DST-15 part 1 gates of SPEC-LIT §159.10, written once and included
// both by src/dust/tests_cloud.rs (as dust::cloud::tests) and by
// tests/dust_cloud.rs, so both builds run them. Everything here is host
// f64 in both builds: the volatile record, the mechanism file, the
// datums and the enthalpy, the gather's ledgers, the sources and the
// cloud's AICC.
// The specification was informed by a review of FSCDustFoam (Huang &
// Lipatnikov, GPL) and no code was copied.
// No GPL-licensed source was consulted.

use ofgpu::chem::mechanism::Mechanism;
use ofgpu::chem::thermo::{ElementTable, P_ATM, R_GAS, SpeciesThermo, ThermoSet};
use ofgpu::dust::cloud::{
    cloud_aicc, gather, particle_enthalpy, sources, volatile_species, CloudCharge, CloudThermo,
    MEMBERS, StageDatums, T_DATUM,
};
use ofgpu::dust::particle::{latent_heat, Drying, ParticleState};
use ofgpu::dust::thermo::StarchFuel;

/// |a - b| relative to |b|.
fn rel(a: f64, b: f64) -> f64 {
    (a - b).abs() / b.abs()
}

/// The volatile mechanism: tm4513's set, volatile1.ck's VOL and its
/// one-step reaction, and the cloud thermo over both.
fn mech_and_th() -> (Mechanism, CloudThermo) {
    let root = env!("CARGO_MANIFEST_DIR");
    let set_text = std::fs::read_to_string(format!("{root}/../cases/mech/tm4513_thermo.dat"))
        .expect("cases/mech/tm4513_thermo.dat reads");
    let mech_text = std::fs::read_to_string(format!("{root}/../cases/mech/volatile1.ck"))
        .expect("cases/mech/volatile1.ck reads");
    let set = ThermoSet::read_chemkin(
        &set_text,
        "cases/mech/tm4513_thermo.dat",
        P_ATM,
        &ElementTable::standard(),
    )
    .expect("tm4513's thermo reads");
    let mech =
        Mechanism::read_chemkin(&mech_text, "cases/mech/volatile1.ck", Some(&set))
            .expect("volatile1.ck reads");
    let th = CloudThermo::from_set(mech.thermo(), &ElementTable::standard())
        .expect("the cloud thermo builds");
    (mech, th)
}

/// An LCG: x' = x*6364136223846793005 + 1442695040888963407, wrapping;
/// u() is uniform on [0, 1) as (x >> 11) 2^-53.
struct Lcg(u64);

impl Lcg {
    fn u(&mut self) -> f64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (self.0 >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }
}

/// Test 4's and test 5's parcel pool: 1000 parcels in 37 cells.
fn gather_inputs() -> (
    Vec<ParticleState>,
    Vec<ParticleState>,
    Vec<f64>,
    Vec<usize>,
    CloudThermo,
    StageDatums,
) {
    let (th, dat) = cloud_and_datums();
    let n = 1000;
    let n_cells = 37;
    let mut rng = Lcg(0x1234_5678_9abc_def0);
    let mut before = Vec::with_capacity(n);
    let mut n_p = Vec::with_capacity(n);
    let mut cell = Vec::with_capacity(n);
    for i in 0..n {
        before.push(ParticleState {
            m_w: rng.u() * 1.0e-12,
            m_v: rng.u() * 1.0e-12,
            m_c: rng.u() * 1.0e-12,
            m_a: rng.u() * 1.0e-12,
            t: 300.0 + 2200.0 * rng.u(),
            d: 1.0e-5,
        });
        n_p.push(1.0e6 * (1.0 + (i % 5) as f64));
        cell.push((i * 7) % n_cells);
    }
    let mut after = before.clone();
    for a in after.iter_mut() {
        a.m_w *= rng.u();
        a.m_v *= rng.u();
        a.m_c *= rng.u();
        a.t += 100.0 * rng.u();
    }
    (before, after, n_p, cell, th, dat)
}

/// The cloud thermo of the mechanism and its (159.18) datums at dH 1e5.
fn cloud_and_datums() -> (CloudThermo, StageDatums) {
    let (_mech, th) = mech_and_th();
    let dr = Drying::water(101_325.0).expect("water at 1 atm");
    let dat = StageDatums::new(&th, &dr, 1.0e5).expect("the stage datums");
    (th, dat)
}

/// Test 1 (D/R): the volatile record of (159.16) is the starch's formula
/// times 50 - the whole counts C300 H394 O249, upper == lower ==
/// [50 a0, 50 a1, 0, 0, 0, 50 a5, 0] exactly, the molar mass from the
/// element table's weights, and per kilogram the starch's cp and h at
/// three temperatures to 1e-13 relative.
#[test]
fn volatile_record_is_the_starch_times_50() {
    let fuel = StarchFuel::huang2022_table1().expect("the starch fuel");
    let vol = volatile_species(&fuel, &ElementTable::standard()).expect("the volatile record");
    println!("  VOL: elements {:?}", vol.elements);
    assert_eq!(
        vol.elements,
        [("C".to_string(), 300), ("H".to_string(), 394), ("O".to_string(), 249)]
    );
    let s = 50.0;
    let want = [s * fuel.a0, s * fuel.a1, 0.0, 0.0, 0.0, s * fuel.a5(), 0.0];
    assert_eq!(vol.upper, want, "the upper coefficients must be 50 times the starch's");
    assert_eq!(vol.lower, want, "the lower coefficients must be 50 times the starch's");
    let w = (300.0 * 12.011 + 394.0 * 1.00794 + 249.0 * 15.9994) / 1000.0;
    println!("  W_VOL = {:.17} kg/mol, |got - hand|/W = {:.3e}", vol.molar_mass, rel(vol.molar_mass, w));
    assert!(rel(vol.molar_mass, w) <= 1.0e-15, "W_VOL is off the element table's weights");
    let w50 = w / 50.0;
    for t in [300.0, 1000.0, 2500.0] {
        let cp = vol.cp(t).expect("cp in range") / vol.molar_mass;
        let h = vol.h(t).expect("h in range") / vol.molar_mass;
        let cp_want = R_GAS * (fuel.a0 + fuel.a1 * t) / w50;
        let h_want = R_GAS * (fuel.a0 * t + fuel.a1 * t * t / 2.0 + fuel.a5()) / w50;
        let (ecp, eh) = (rel(cp, cp_want), rel(h, h_want));
        println!("  T = {t} K: cp/W = {cp:.6}, rel {ecp:.3e}; h/W = {h:.4e}, rel {eh:.3e}");
        assert!(ecp <= 1.0e-13, "cp/W at {t} K is {ecp:.3e} off the starch's (bar 1e-13)");
        assert!(eh <= 1.0e-13, "h/W at {t} K is {eh:.3e} off the starch's (bar 1e-13)");
    }
}

/// Test 2 (D): volatile1.ck reads - the five species in order, one
/// irreversible reaction first order in VOL and O2, VOL's in-file record
/// the same elements and the same 14 coefficients as (159.16) to 5e-9
/// (exactly 0 where that is 0), and the reaction balancing C, H, O and N
/// exactly in the records' element counts and the reaction's Rational
/// coefficients. The rate is printed in the units the reader stores.
#[test]
fn volatile1_mechanism_reads() {
    let (mech, _th) = mech_and_th();
    let names: Vec<&str> = mech.species().iter().map(String::as_str).collect();
    println!("  species {names:?}");
    assert_eq!(names, ["VOL", "O2", "H2O", "CO2", "N2"]);
    assert_eq!(mech.reactions().len(), 1, "one reaction");
    let rx = &mech.reactions()[0];
    let vol0 =
        volatile_species(&StarchFuel::huang2022_table1().unwrap(), &ElementTable::standard())
            .unwrap();
    let vol = mech.thermo().get("VOL").expect("VOL's record in the mechanism");
    assert_eq!(vol.elements, vol0.elements, "VOL's element counts");
    for (got, want, which) in
        [(vol.upper, vol0.upper, "upper"), (vol.lower, vol0.lower, "lower")]
    {
        for (k, (a, b)) in got.iter().zip(want.iter()).enumerate() {
            if *b == 0.0 {
                assert_eq!(*a, 0.0, "{which} coefficient {} must be exactly 0", k + 1);
            } else {
                let e = rel(*a, *b);
                assert!(e <= 5.0e-9, "{which} coefficient {} is {e:.3e} off (bar 5e-9)", k + 1);
            }
        }
    }
    println!("  VOL's 14 coefficients agree with (159.16) to within 5e-9 relative");
    // The exact element balance: sum nu*count over products equals over reactants.
    let side = |list: &[(usize, ofgpu::chem::mechanism::Rational)], el: &str| -> (i128, i128) {
        let (mut n, mut d) = (0i128, 1i128);
        for (k, nu) in list {
            let rec = &mech.thermo().species()[*k];
            let count = rec.elements.iter().find(|(s, _)| s == el).map_or(0, |&(_, c)| c);
            let (tn, td) = (i128::from(nu.num()) * i128::from(count), i128::from(nu.den()));
            n = n * td + tn * d;
            d *= td;
        }
        (n, d)
    };
    for (el, _) in mech.elements() {
        let (pn, pd) = side(&rx.products, el);
        let (rn, rd) = side(&rx.reactants, el);
        assert_eq!(pn * rd, rn * pd, "element {el} does not balance exactly");
    }
    println!("  {} balances C, H, O and N exactly", rx.equation);
    assert!(!rx.reversible, "the reaction is irreversible");
    let named = |list: &[(usize, ofgpu::chem::mechanism::Rational)]| -> Vec<(&str, f64)> {
        list.iter()
            .map(|(k, nu)| (mech.thermo().species()[*k].name.as_str(), nu.to_f64()))
            .collect()
    };
    assert_eq!(named(&rx.reactants), [("VOL", 1.0), ("O2", 274.0)]);
    assert_eq!(named(&rx.products), [("CO2", 300.0), ("H2O", 197.0)]);
    println!(
        "  rate: a = {:.6e} (SI, m^3/mol/s at m = 2), beta = {}, theta = {:.6} K (E = \
         30000 cal/mol)",
        rx.rate.a, rx.rate.beta, rx.rate.theta
    );
    assert!(rel(rx.rate.a, 1.0e14 * 1.0e-6) <= 1.0e-12, "A must be 1e14 cm3/(mol s) in SI");
    assert_eq!(rx.rate.beta, 0.0);
    assert!(rel(rx.rate.theta, 4.184 * 30000.0 / R_GAS) <= 1.0e-12);
}

/// Test 3 (D): the stage datums of (159.18) are the hand values bitwise
/// (h_c = h_a = 0), the particle enthalpy of a five-state is its hand
/// expression bitwise, and moving 1e-13 kg from the volatile at fixed T
/// changes E by that mass's sensible-plus-datum enthalpy to 1e-12
/// relative.
#[test]
fn datums_and_enthalpy() {
    let (th, dat) = cloud_and_datums();
    let dr = Drying::water(101_325.0).expect("water at 1 atm");
    let h_w = th.h_mass(1, T_DATUM).expect("h H2O") - latent_heat(&dr, T_DATUM);
    let h_v = th.h_mass(0, T_DATUM).expect("h VOL") - 1.0e5;
    println!("  h_w = {h_w:.9e}, h_v = {h_v:.9e}, h_c = {:.1}, h_a = {:.1} J/kg", dat.h_c, dat.h_a);
    assert_eq!(dat.h_w, h_w, "h_w must be the hand value bitwise");
    assert_eq!(dat.h_v, h_v, "h_v must be the hand value bitwise");
    assert_eq!(dat.h_c, 0.0);
    assert_eq!(dat.h_a, 0.0);
    let s = ParticleState {
        m_w: 1.0e-13,
        m_v: 5.0e-13,
        m_c: 2.0e-13,
        m_a: 1.0e-13,
        t: 1500.0,
        d: 1.0e-5,
    };
    let cp = 1500.0;
    let e = particle_enthalpy(&s, cp, &dat);
    let hand = (s.m_w + s.m_v + s.m_c + s.m_a) * cp * (s.t - T_DATUM)
        + s.m_w * dat.h_w
        + s.m_v * dat.h_v
        + s.m_c * dat.h_c
        + s.m_a * dat.h_a;
    println!("  E = {e:.12e} J");
    assert_eq!(e, hand, "E must be the hand expression bitwise");
    let dp = 1.0e-13;
    let s2 = ParticleState { m_v: s.m_v - dp, ..s };
    let e2 = particle_enthalpy(&s2, cp, &dat);
    let want = dp * (1500.0 * (1500.0 - T_DATUM) + dat.h_v);
    let err = rel(e - e2, want);
    println!("  E - E' = {:.9e}, hand {want:.9e}, rel {err:.3e}", e - e2);
    assert!(err <= 1.0e-12, "the volatile's enthalpy is {err:.3e} off (bar 1e-12)");
}

/// Test 4 (A): 1000 parcels in 37 cells from a deterministic LCG. The
/// gathered cells sum, per member and in energy, to the hand sums over
/// parcels to 1e-14 relative; the carbon ledger
/// `sum (m_VOL fC_VOL + m_CO2 W_C/W_CO2)` equals `sum n (dv fC_VOL + dc)`,
/// and the oxygen ledger `sum (m_VOL fO_VOL + m_H2O fO_H2O + m_CO2
/// W_O2/W_CO2 + m_O2)` equals the oxygen the particles lost, each to
/// 1e-14 of the flow. A short `after` and a cell 37 are refused by name.
#[test]
fn gather_telescopes() {
    let (before, after, n_p, cell, th, dat) = gather_inputs();
    let cp = 1500.0;
    let n_cells = 37;
    let gains = gather(&before, &after, &n_p, &cell, n_cells, cp, &dat, &th)
        .expect("the gather takes the pool");
    assert_eq!(gains.len(), n_cells);
    let mut hand_m = [0.0f64; 4];
    let mut hand_e = 0.0f64;
    for i in 0..before.len() {
        let (b, a) = (&before[i], &after[i]);
        let w = n_p[i];
        hand_m[0] += w * (b.m_v - a.m_v);
        hand_m[1] += w * (b.m_w - a.m_w);
        hand_m[2] += w * (b.m_c - a.m_c) * (th.w_co2() / th.w_c);
        hand_m[3] -= w * (b.m_c - a.m_c) * (th.o2.molar_mass / th.w_c);
        hand_e += w * (particle_enthalpy(b, cp, &dat) - particle_enthalpy(a, cp, &dat));
    }
    for (k, name) in MEMBERS.iter().enumerate() {
        let got: f64 = gains.iter().map(|g| g.m[k]).sum();
        println!(
            "  member {name}: gathered {got:.10e}, hand {:.10e}, rel {:.3e}",
            hand_m[k],
            rel(got, hand_m[k])
        );
        assert!(rel(got, hand_m[k]) <= 1.0e-14, "member {name} is off the hand sum");
    }
    let w_o = ElementTable::standard().weight("O").expect("O") / 1000.0;
    let count = |sym: &str| {
        th.vol.elements.iter().find(|(s, _)| s == sym).map_or(0.0, |&(_, n)| f64::from(n))
    };
    let f_c_vol = count("C") * th.w_c / th.vol.molar_mass;
    let f_o_vol = count("O") * w_o / th.vol.molar_mass;
    let f_o_h2o = w_o / th.h2o.molar_mass;
    let (mut hand_c, mut hand_o, mut flow_o) = (0.0f64, 0.0f64, 0.0f64);
    for i in 0..before.len() {
        let (b, a) = (&before[i], &after[i]);
        let w = n_p[i];
        let (dv, dw, dc) = (b.m_v - a.m_v, b.m_w - a.m_w, b.m_c - a.m_c);
        hand_c += w * (dv * f_c_vol + dc);
        hand_o += w * (dv * f_o_vol + dw * f_o_h2o);
        flow_o += w * (dv * f_o_vol + dw * f_o_h2o + dc * th.o2.molar_mass / th.w_c);
    }
    let got_c: f64 =
        gains.iter().map(|g| g.m[0] * f_c_vol + g.m[2] * th.w_c / th.w_co2()).sum();
    let got_o: f64 = gains
        .iter()
        .map(|g| {
            g.m[0] * f_o_vol + g.m[1] * f_o_h2o + g.m[2] * th.o2.molar_mass / th.w_co2() + g.m[3]
        })
        .sum();
    let got_e: f64 = gains.iter().map(|g| g.e).sum();
    println!("  carbon: {got_c:.10e} vs {hand_c:.10e}, rel {:.3e}", rel(got_c, hand_c));
    println!("  oxygen: {got_o:.10e} vs {hand_o:.10e} (flow {flow_o:.10e})");
    println!("  energy: {got_e:.10e} vs {hand_e:.10e}, rel {:.3e}", rel(got_e, hand_e));
    assert!(rel(got_c, hand_c) <= 1.0e-14, "the carbon ledger is off");
    assert!(
        (got_o - hand_o).abs() <= 1.0e-14 * flow_o.abs(),
        "the oxygen ledger is off the oxygen moved"
    );
    assert!(rel(got_e, hand_e) <= 1.0e-14, "the energy ledger is off");
    let short = gather(&before, &after[..before.len() - 1], &n_p, &cell, n_cells, cp, &dat, &th)
        .expect_err("a short after is refused");
    println!("  short after: {short}");
    assert!(short.to_string().contains("dust.cloud"));
    let mut bad = cell.clone();
    bad[0] = n_cells;
    let err = gather(&before, &after, &n_p, &bad, n_cells, cp, &dat, &th)
        .expect_err("a cell at n_cells is refused");
    println!("  cell 37: {err}");
    assert!(err.to_string().contains("dust.cloud"));
}

/// Test 5 (R): on test 4's gains with `V_c = 1e-6 (1 + c/37)`,
/// `T_gas[c] = 400 + 50 c` and dt 1e-4, every `mdot` and `q` matches the
/// hand formula to 1e-15 relative (q against the larger of |dE| and
/// |sum m h|, both over V dt). dt 0 is refused by name.
#[test]
fn sources_are_the_t_form() {
    let (before, after, n_p, cell, th, dat) = gather_inputs();
    let cp = 1500.0;
    let n_cells = 37;
    let gains = gather(&before, &after, &n_p, &cell, n_cells, cp, &dat, &th)
        .expect("the gather takes the pool");
    let volume: Vec<f64> = (0..n_cells).map(|c| 1.0e-6 * (1.0 + c as f64 / 37.0)).collect();
    let t_gas: Vec<f64> = (0..n_cells).map(|c| 400.0 + 50.0 * c as f64).collect();
    let dt = 1.0e-4;
    let src = sources(&gains, &volume, &t_gas, dt, &th).expect("the sources build");
    let (mut worst_m, mut worst_q) = (0.0f64, 0.0f64);
    for c in 0..n_cells {
        let vdt = volume[c] * dt;
        let mut mh = 0.0;
        for k in 0..4 {
            worst_m = worst_m.max(rel(src.mdot[k][c], gains[c].m[k] / vdt));
            mh += gains[c].m[k] * th.h_mass(k, t_gas[c]).expect("h in range");
        }
        let scale = gains[c].e.abs().max(mh.abs()) / vdt;
        worst_q = worst_q.max((src.q[c] - (gains[c].e - mh) / vdt).abs() / scale);
    }
    println!("  37 cells at dt {dt}: worst |mdot - hand| rel = {worst_m:.3e}, worst |q - hand|/scale = {worst_q:.3e}");
    assert!(worst_m <= 1.0e-15, "an mdot is {worst_m:.3e} off (bar 1e-15)");
    assert!(worst_q <= 1.0e-15, "a q is {worst_q:.3e} off (bar 1e-15)");
    let zero = sources(&gains, &volume, &t_gas, 0.0, &th).expect_err("dt 0 is refused");
    println!("  dt 0: {zero}");
    assert!(zero.to_string().contains("dust.cloud"));
}

/// Test 6 (A): the box's charge (C 0.012 kg/m3, fractions {0.05, 0.75,
/// 0.15, 0.05}, cp 1500) in air (O2 0.233, N2 0.767) at 1500 K closes on
/// (159.21): p0 is 101325 Pa to 1e-9 relative, U_e(t_v) is u0 to 1e-12,
/// t_v lands in (1500, 6000) K with p_v above p0, and the end gas holds
/// the initial gas's plus the cloud's C, H, O and N to 1e-14 relative. A
/// pure-ash charge does not move: |t_v - 1500| <= 1e-9. A charge of
/// C = 1 kg/m3 is refused, richer than complete conversion allows.
#[test]
fn cloud_aicc_closes() {
    let (mech, th) = mech_and_th();
    let n2 = mech.thermo().get("N2").expect("N2 in the mechanism").clone();
    let w_o2 = th.o2.molar_mass;
    let rho0 = 101_325.0 / (R_GAS * 1500.0 * (0.233 / w_o2 + 0.767 / n2.molar_mass));
    let gas: Vec<(SpeciesThermo, f64)> = vec![
        (th.o2.clone(), 0.233 * rho0),
        (n2, 0.767 * rho0),
        (th.vol.clone(), 0.0),
        (th.h2o.clone(), 0.0),
        (th.co2.clone(), 0.0),
    ];
    let charge = CloudCharge {
        concentration: 0.012,
        fractions: [0.05, 0.75, 0.15, 0.05],
        cp: 1500.0,
        datums: StageDatums::new(&th, &Drying::water(101_325.0).expect("water"), 1.0e5)
            .expect("the stage datums"),
    };
    let a = cloud_aicc(&gas, 1500.0, &charge, &th).expect("the cloud's AICC closes");
    println!("  p0 = {:.6} Pa, u0 = {:.4e} J/m3", a.p0, a.u0);
    assert!((a.p0 - 101_325.0).abs() <= 1.0e-9 * 101_325.0, "p0 = {}", a.p0);
    let u_e = |t: f64| -> f64 {
        let mut sh = 0.0;
        let mut sw = 0.0;
        for ((rec, _), r) in gas.iter().zip(&a.rho_end) {
            sh += r * rec.h(t).expect("h in range") / rec.molar_mass;
            sw += r / rec.molar_mass;
        }
        sh - R_GAS * t * sw
            + charge.concentration * charge.fractions[3] * charge.cp * (t - T_DATUM)
    };
    let err = (u_e(a.t_v) - a.u0).abs() / a.u0.abs().max(1.0);
    println!("  t_v = {:.4} K, p_v = {:.4} Pa, p_v/p0 = {:.9}, |U_e(t_v) - u0|/|u0| = {err:.3e}", a.t_v, a.p_v, a.p_v / a.p0);
    assert!(err <= 1.0e-12, "U_e(t_v) is {err:.3e} off u0 (bar 1e-12)");
    assert!(a.t_v > 1500.0 && a.t_v < 6000.0, "t_v = {} must lie in (1500, 6000)", a.t_v);
    assert!(a.p_v > a.p0, "the AICC burn must raise the pressure");
    // The element ledgers: C, H, O, N of rho_end against the initial gas
    // plus the cloud's (moisture as H2O, volatile as VOL, char as C).
    let std = ElementTable::standard();
    let w = |sym: &str| std.weight(sym).expect(sym) / 1000.0;
    let (wc, wh, wo, wn) = (w("C"), w("H"), w("O"), w("N"));
    let elements_of = |rec: &SpeciesThermo, rho: f64| -> [f64; 4] {
        let mut out = [0.0f64; 4];
        for (sym, n) in &rec.elements {
            let (k, w1) = match sym.as_str() {
                "C" => (0, wc),
                "H" => (1, wh),
                "O" => (2, wo),
                "N" => (3, wn),
                _ => continue,
            };
            out[k] += rho * f64::from(*n) * w1 / rec.molar_mass;
        }
        out
    };
    let count = |sym: &str| {
        th.vol.elements.iter().find(|(s, _)| s == sym).map_or(0.0, |&(_, n)| f64::from(n))
    };
    let (f_c, f_h, f_o) = (
        count("C") * wc / th.vol.molar_mass,
        count("H") * wh / th.vol.molar_mass,
        count("O") * wo / th.vol.molar_mass,
    );
    let cloud = [
        charge.concentration * (charge.fractions[2] + charge.fractions[1] * f_c),
        charge.concentration * (charge.fractions[0] * 2.0 * wh / th.h2o.molar_mass
            + charge.fractions[1] * f_h),
        charge.concentration * (charge.fractions[0] * wo / th.h2o.molar_mass
            + charge.fractions[1] * f_o),
        0.0,
    ];
    for (k, name) in ["C", "H", "O", "N"].iter().enumerate() {
        let want = gas.iter().map(|(rec, r)| elements_of(rec, *r)[k]).sum::<f64>() + cloud[k];
        let got = gas
            .iter()
            .zip(&a.rho_end)
            .map(|((rec, _), r)| elements_of(rec, *r)[k])
            .sum::<f64>();
        println!("  element {name}: end {got:.10e} vs gas+cloud {want:.10e}, rel {:.3e}", rel(got, want));
        assert!(rel(got, want) <= 1.0e-14, "element {name} does not close to 1e-14");
    }
    let ash = CloudCharge { fractions: [0.0, 0.0, 0.0, 1.0], ..charge };
    let b = cloud_aicc(&gas, 1500.0, &ash, &th).expect("pure ash closes at T0");
    println!("  pure ash: t_v = {:.12} K", b.t_v);
    assert!((b.t_v - 1500.0).abs() <= 1.0e-9, "pure ash must not move: t_v = {}", b.t_v);
    let rich = CloudCharge { concentration: 1.0, ..charge };
    let err = cloud_aicc(&gas, 1500.0, &rich, &th).expect_err("C = 1 kg/m3 is refused");
    println!("  C = 1 kg/m3: {err}");
    assert!(err.to_string().contains("dust.cloud"));
}
