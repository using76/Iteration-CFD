// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.
// Provenance: see PROVENANCE.md. No GPL-licensed source was consulted.

// The DST-12 gates of SPEC-LIT §159.5, written once and included both by
// src/dust/tests_particle.rs (as dust::particle::tests) and by
// tests/dust_particle.rs, so both builds run them. The host twin is f64 in
// both builds and is held at the f64 bars in both. The device stores its
// state in `Scalar`, so its closed-form rows are class R: 1e-10 in f64 and
// tol32 = max(tol64, min(tol64 2^29, 1e-5)) = 1e-5 under `single`
// (D-F32-1). The device-vs-host row is 4 ulp in either build, against the
// host run on the device's own rounded inputs. A GPU test returns early, by
// name, on a box without a device. DST-13 burns the char; these gates keep
// it inert (no oxygen, `CharInputs::inert`), and dust_char_cases.rs holds it.
// The specification was informed by a review of FSCDustFoam (Huang &
// Lipatnikov, GPL) and no code was copied.
// No GPL-licensed source was consulted.

use ofgpu::device::{DevBuf, Gpu};
use ofgpu::dust::char::CharInputs;
use ofgpu::dust::device::{particle_step, ExchangeFields, ParticleFields, ParticleGasDev, ParticleKernels};
use ofgpu::dust::particle::{
    bird_factor, phi_g, step, Drying, Exchange, GasAtParcel, ParticleInputs, ParticleProps, ParticleState,
    Stage, R0, SIGMA_SB,
};
use ofgpu::error::Error;
use ofgpu::{Acc, Label, Scalar};

/// The device ulp bar of §159.5, the same count in both builds.
const ULPS: u64 = 4;

/// D-F32-1: a class-R bar in the build's precision.
fn tol(tol64: f64) -> f64 {
    if cfg!(feature = "single") {
        tol64.max((tol64 * 2f64.powi(29)).min(1.0e-5))
    } else {
        tol64
    }
}

/// A value as the device holds it: rounded to `Scalar`, widened back.
#[allow(clippy::unnecessary_cast)]
fn r(v: f64) -> f64 {
    v as Scalar as f64
}

/// SplitMix64; `u()` is uniform on [0, 1) as (x >> 11) 2^-53.
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

/// The parcel properties of the gates: C_p = 1500 J/(kg K), SFOM with
/// A = 1000 1/s and E_a = 30 kJ/mol, C_p,vol = 1100, water at 1 atm, and
/// air-like mu = 1.8e-5 Pa s, k_g = 0.06 W/(m K), c_p,g = 1100 J/(kg K).
fn props(emissivity: f64, devol_dh: f64) -> ParticleProps {
    ParticleProps::new(&ParticleInputs {
        cp: 1500.0,
        emissivity,
        devol_a: 1000.0,
        devol_e_a: 3.0e4,
        devol_dh,
        cp_volatile: 1100.0,
        drying: Drying::water(101_325.0).expect("water at 1 atm"),
        mu: 1.8e-5,
        k_g: 0.06,
        cp_g: 1100.0,
        char_rate: CharInputs::inert(),
    })
    .expect("props")
}

/// A sphere of diameter d and density rho_p, split by mass fractions
/// (moisture, volatile, char), ash the rest.
fn particle(d: f64, rho_p: f64, f: (f64, f64, f64), t: f64) -> ParticleState {
    let m = rho_p * std::f64::consts::PI * d * d * d / 6.0;
    ParticleState { m_w: f.0 * m, m_v: f.1 * m, m_c: f.2 * m, m_a: (1.0 - f.0 - f.1 - f.2) * m, t, d }
}

/// The state rounded to `Scalar` as the device holds it.
fn rounded(s: &ParticleState) -> ParticleState {
    ParticleState { m_w: r(s.m_w), m_v: r(s.m_v), m_c: r(s.m_c), m_a: r(s.m_a), t: r(s.t), d: r(s.d) }
}

fn gpu(test: &str) -> Option<Gpu> {
    match Gpu::new(0) {
        Ok(g) => Some(g),
        Err(_) => {
            eprintln!("{test}: skipped, no CUDA device");
            None
        }
    }
}

fn up(g: &Gpu, v: &[f64]) -> DevBuf<Scalar> {
    let s: Vec<Scalar> = v.iter().map(|&x| x as Scalar).collect();
    g.upload(&s).expect("upload")
}

fn down(g: &Gpu, b: &DevBuf<Scalar>) -> Vec<f64> {
    g.download(b).expect("download").iter().map(|&x| x as f64).collect()
}

/// A device pool of parcels, their accumulators and the gas cells.
struct Pool {
    k: ParticleKernels,
    m_w: DevBuf<Scalar>,
    m_v: DevBuf<Scalar>,
    m_c: DevBuf<Scalar>,
    m_a: DevBuf<Scalar>,
    t: DevBuf<Scalar>,
    d: DevBuf<Scalar>,
    acc: Vec<DevBuf<Acc>>,
    cell: DevBuf<Label>,
    u_rel: DevBuf<Scalar>,
    tg: DevBuf<Scalar>,
    yw: DevBuf<Scalar>,
    rho: DevBuf<Scalar>,
    gr: DevBuf<Scalar>,
    po2: DevBuf<Scalar>,
    n: usize,
    n_cells: usize,
}

impl Pool {
    /// `gas[c]` is cell c; parcel i sits in `cell[i]` with `u_rel[i]`.
    fn new(g: &Gpu, s: &[ParticleState], cell: &[usize], u_rel: &[f64], gas: &[GasAtParcel]) -> Self {
        let n = s.len();
        let col = |f: fn(&ParticleState) -> f64| up(g, &s.iter().map(f).collect::<Vec<_>>());
        let gcol = |f: fn(&GasAtParcel) -> f64| up(g, &gas.iter().map(f).collect::<Vec<_>>());
        let cells: Vec<Label> = cell.iter().map(|&c| c as Label).collect();
        Self {
            k: ParticleKernels::new(g).expect("kernels"),
            m_w: col(|p| p.m_w),
            m_v: col(|p| p.m_v),
            m_c: col(|p| p.m_c),
            m_a: col(|p| p.m_a),
            t: col(|p| p.t),
            d: col(|p| p.d),
            acc: (0..11).map(|_| g.zeros::<Acc>(n).expect("acc")).collect(),
            cell: g.upload(&cells).expect("cells"),
            u_rel: up(g, u_rel),
            tg: gcol(|c| c.t),
            yw: gcol(|c| c.y_w),
            rho: gcol(|c| c.rho),
            gr: gcol(|c| c.g),
            po2: gcol(|c| c.p_o2),
            n,
            n_cells: gas.len(),
        }
    }

    fn step(&mut self, g: &Gpu, p: &ParticleProps, dt: f64) -> Result<(), Error> {
        let [a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10] = &mut self.acc[..] else { unreachable!() };
        particle_step(
            g,
            &self.k,
            p,
            ParticleFields {
                m_w: &mut self.m_w,
                m_v: &mut self.m_v,
                m_c: &mut self.m_c,
                m_a: &self.m_a,
                t: &mut self.t,
                d: &mut self.d,
            },
            ExchangeFields {
                m_w: a0,
                m_v: a1,
                mt_w: a2,
                mt_v: a3,
                q_conv: a4,
                q_rad: a5,
                m_c: a6,
                m_o2: a7,
                m_co2: a8,
                mt_c: a9,
                q_char: a10,
            },
            ParticleGasDev {
                cell: &self.cell,
                u_rel: &self.u_rel,
                t: &self.tg,
                y_w: &self.yw,
                rho: &self.rho,
                g: &self.gr,
                p_o2: &self.po2,
                n_cells: self.n_cells,
            },
            dt,
            self.n,
        )
    }

    fn states(&self, g: &Gpu) -> Vec<ParticleState> {
        let (w, v, c, a, t, d) =
            (down(g, &self.m_w), down(g, &self.m_v), down(g, &self.m_c), down(g, &self.m_a), down(g, &self.t), down(g, &self.d));
        (0..self.n).map(|i| ParticleState { m_w: w[i], m_v: v[i], m_c: c[i], m_a: a[i], t: t[i], d: d[i] }).collect()
    }

    fn exchanges(&self, g: &Gpu) -> Vec<Exchange> {
        let a: Vec<Vec<f64>> = self.acc.iter().map(|b| g.download(b).expect("acc")).collect();
        (0..self.n)
            .map(|i| Exchange {
                m_w: a[0][i],
                m_v: a[1][i],
                mt_w: a[2][i],
                mt_v: a[3][i],
                q_conv: a[4][i],
                q_rad: a[5][i],
                m_c: a[6][i],
                m_o2: a[7][i],
                m_co2: a[8][i],
                mt_c: a[9][i],
                q_char: a[10][i],
            })
            .collect()
    }
}

/// §159.5 row 1, (159.6): T_g = T_p = 900 K, Delta H = 0, no radiation and
/// no moisture, so T_p stays 900 K exactly and m_v(t) = m_v0 e^(-k t). 40
/// steps over k t = 3. Host 1e-10 relative; device 1e-10 (R).
#[test]
fn devol_isothermal_closed_form() {
    let p = props(0.0, 0.0);
    let s0 = particle(50.0e-6, 1500.0, (0.0, 0.6, 0.3), 900.0);
    let gas = GasAtParcel { t: 900.0, y_w: 0.0, rho: 101_325.0 / (287.0 * 900.0), g: 0.0, u_rel: 0.3, p_o2: 0.0 };
    let k = 1000.0 * (-(3.0e4 / R0) / 900.0).exp();
    let n = 40;
    let dt = 3.0 / k / n as f64;
    let mut s = s0;
    let mut given = 0.0;
    let mut worst: f64 = 0.0;
    for j in 1..=n {
        let (s1, ex) = step(&p, &s, &gas, dt).expect("step");
        assert_eq!(s1.t, 900.0, "step {j}: T_p moved off 900 K");
        assert_eq!((s1.m_w, s1.m_c, s1.m_a), (s0.m_w, s0.m_c, s0.m_a), "step {j}: only m_v may move");
        given += ex.m_v;
        s = s1;
        let ex_m = s0.m_v * (-k * dt * j as f64).exp();
        worst = worst.max((s.m_v - ex_m).abs() / ex_m);
    }
    let lost = (s0.m_v - s.m_v - given).abs() / s0.m_v;
    println!("  devolatilisation, host: worst |m_v/exact - 1| = {worst:.3e}; ledger {lost:.3e}");
    assert!(worst <= 1.0e-10, "host m_v is {worst:.3e} from (159.6) (bar 1e-10)");
    assert!(lost <= 1.0e-14, "the volatile given differs from the volatile lost by {lost:.3e}");

    let Some(g) = gpu("devol_isothermal_closed_form") else { return };
    let sr = rounded(&s0);
    let mut pool = Pool::new(&g, &[sr], &[0], &[gas.u_rel], &[gas]);
    let mut worst_d: f64 = 0.0;
    for j in 1..=n {
        pool.step(&g, &p, dt).expect("device step");
        let sd = pool.states(&g)[0];
        assert_eq!(sd.t, r(900.0), "device step {j}: T_p moved off 900 K");
        let ex_m = sr.m_v * (-k * dt * j as f64).exp();
        worst_d = worst_d.max((sd.m_v - ex_m).abs() / ex_m);
    }
    println!("  devolatilisation, device: worst |m_v/exact - 1| = {worst_d:.3e} (bar {:.0e})", tol(1.0e-10));
    assert!(worst_d <= tol(1.0e-10), "device m_v is {worst_d:.3e} from (159.6)");
}

/// §159.5 row 2, (159.6): an ash particle at rest (Re = 0, Nu = 2), no
/// reaction, no radiation: T_p(t) = T_g + (T_0 - T_g) e^(-t/tau_T). 40
/// steps over 3 tau_T; the error is relative to |T_0 - T_g|.
#[test]
fn heatup_closed_form() {
    let p = props(0.0, 0.0);
    let s0 = particle(50.0e-6, 1500.0, (0.0, 0.0, 0.0), 300.0);
    let gas = GasAtParcel { t: 1200.0, y_w: 0.0, rho: 101_325.0 / (287.0 * 1200.0), g: 0.0, u_rel: 0.0, p_o2: 0.0 };
    let tau = |s: &ParticleState| s.mass() * 1500.0 / (2.0 * std::f64::consts::PI * s.d * 0.06);
    let n = 40;
    let dt = 3.0 * tau(&s0) / n as f64;
    let mut s = s0;
    let mut q = 0.0;
    let mut worst: f64 = 0.0;
    for j in 1..=n {
        let (s1, ex) = step(&p, &s, &gas, dt).expect("step");
        q += ex.q_conv;
        s = s1;
        let ex_t = 1200.0 + (300.0 - 1200.0) * (-(dt * j as f64) / tau(&s0)).exp();
        worst = worst.max((s.t - ex_t).abs() / 900.0);
    }
    let heat = (q - s0.mass() * 1500.0 * (s.t - 300.0)).abs() / (s0.mass() * 1500.0 * 900.0);
    println!("  heat-up, host: worst |T - exact|/|T0 - Tg| = {worst:.3e}; energy {heat:.3e}");
    assert!(worst <= 1.0e-10, "host T_p is {worst:.3e} from (159.6) (bar 1e-10)");
    assert!(heat <= 1.0e-12, "the convective heat differs from m C dT by {heat:.3e}");

    let Some(g) = gpu("heatup_closed_form") else { return };
    let sr = rounded(&s0);
    let mut pool = Pool::new(&g, &[sr], &[0], &[0.0], &[gas]);
    let mut worst_d: f64 = 0.0;
    for j in 1..=n {
        pool.step(&g, &p, dt).expect("device step");
        let sd = pool.states(&g)[0];
        let ex_t = 1200.0 + (sr.t - 1200.0) * (-(dt * j as f64) / tau(&sr)).exp();
        worst_d = worst_d.max((sd.t - ex_t).abs() / 900.0);
    }
    println!("  heat-up, device: worst |T - exact|/|T0 - Tg| = {worst_d:.3e} (bar {:.0e})", tol(1.0e-10));
    assert!(worst_d <= tol(1.0e-10), "device T_p is {worst_d:.3e} from (159.6)");
}

/// §159.5 row 3: Bird's factor is 1 at beta = 0, matches its leading terms
/// 1 - beta/2 + beta^2/12 to 1e-12 on both sides of the series switch, and
/// is 1/(e - 1) at beta = 1 to 1e-14 relative. The host evaluates it in f64
/// in both builds; the device's copy is held by `device_matches_host`.
#[test]
fn bird_limits() {
    assert_eq!(bird_factor(0.0), 1.0, "beta = 0");
    let mut worst: f64 = 0.0;
    for k in 0..=60 {
        let b = 10f64.powf(-3.0 - 12.0 * k as f64 / 60.0);
        for beta in [b, -b, b * (1.0 + 1.0e-9), 1.0e-3 * (1.0 + 1.0e-12), 1.0e-3 * (1.0 - 1.0e-12)] {
            let lead = 1.0 - beta / 2.0 + beta * beta / 12.0;
            worst = worst.max((bird_factor(beta) - lead).abs());
        }
    }
    let at_one = (bird_factor(1.0) - 0.581_976_706_869_326_5).abs() / 0.581_976_706_869_326_5;
    println!("  Bird: worst |f - (1 - b/2 + b^2/12)| = {worst:.3e} for |b| <= 1e-3; f(1) rel {at_one:.3e}");
    assert!(worst <= 1.0e-12, "the factor leaves its limit by {worst:.3e} (bar 1e-12)");
    assert!(at_one <= 1.0e-14, "f(1) is {at_one:.3e} from 1/(e - 1) (bar 1e-14)");
    // phi1 and g2 of §159.7 meet across their series switch.
    let (s1, s2) = phi_g(1.0 - 1.0e-15);
    let (c1, c2) = phi_g(1.0);
    assert!((s1 - c1).abs() <= 1.0e-14 && (s2 - c2).abs() <= 1.0e-14, "phi1/g2 jump at x = 1");
    assert_eq!(phi_g(0.0), (1.0, 0.5), "phi1(0) = 1, g2(0) = 1/2");
}

/// The stages a run passed through, in order, and the checks of §159.5
/// row 4 on one step from `a` to `b`.
fn check_order(who: &str, j: usize, a: &ParticleState, b: &ParticleState, seen: &mut Vec<Stage>) {
    if b.m_w > 0.0 {
        assert_eq!(b.m_v, a.m_v, "{who} step {j}: volatiles left while moisture remains");
    }
    if b.m_v > 0.0 || b.m_w > 0.0 {
        assert_eq!(b.m_c, a.m_c, "{who} step {j}: char moved before the volatiles were gone");
    }
    assert!(b.m_w <= a.m_w && b.m_v <= a.m_v, "{who} step {j}: a mass grew");
    let st = b.stage();
    if seen.last() != Some(&st) {
        assert!(!seen.contains(&st), "{who} step {j}: stage {st:?} came back");
        seen.push(st);
    }
}

/// §159.5 row 4 (class D): moisture -> volatiles -> char. A 100 um parcel
/// with 10 % moisture, 50 % volatiles and 30 % char in a 1500 K radiating
/// gas: m_v and m_c do not move while moisture remains, m_c does not move
/// while volatiles remain, m_w ends at exactly 0, and every stage is reached
/// once, in order, on the host and on the device.
#[test]
fn stage_order() {
    let p = props(0.9, 1.0e5);
    let s0 = particle(100.0e-6, 1200.0, (0.1, 0.5, 0.3), 300.0);
    let gas = GasAtParcel {
        t: 1500.0,
        y_w: 0.01,
        rho: 101_325.0 / (287.0 * 1500.0),
        g: 4.0 * SIGMA_SB * 1500f64.powi(4),
        u_rel: 1.0,
        p_o2: 0.0,
    };
    let dt = 1.0e-3;
    let mut seen = vec![s0.stage()];
    let mut s = s0;
    for j in 1..=1000 {
        let (s1, _) = step(&p, &s, &gas, dt).expect("step");
        check_order("host", j, &s, &s1, &mut seen);
        s = s1;
    }
    println!("  stage order, host: {seen:?}; m_w {} m_v {} T_p {:.2} K", s.m_w, s.m_v, s.t);
    assert_eq!(seen, vec![Stage::Moisture, Stage::Volatiles, Stage::Char], "host stages");
    assert_eq!((s.m_w, s.m_v, s.m_c), (0.0, 0.0, s0.m_c), "host end state");

    let Some(g) = gpu("stage_order") else { return };
    let sr = rounded(&s0);
    let mut pool = Pool::new(&g, &[sr], &[0], &[gas.u_rel], &[gas]);
    let mut seen = vec![sr.stage()];
    let mut a = sr;
    for j in 1..=1000 {
        pool.step(&g, &p, dt).expect("device step");
        let b = pool.states(&g)[0];
        check_order("device", j, &a, &b, &mut seen);
        a = b;
    }
    println!("  stage order, device: {seen:?}; T_p {:.2} K", a.t);
    assert_eq!(seen, vec![Stage::Moisture, Stage::Volatiles, Stage::Char], "device stages");
    assert_eq!((a.m_w, a.m_v, a.m_c), (0.0, 0.0, sr.m_c), "device end state");
}

/// The spacing of the floating-point numbers at |x|: of `Scalar` for a
/// state word, of f64 for an accumulator.
fn spacing(x: f64, scalar: bool) -> f64 {
    if scalar && std::mem::size_of::<Scalar>() == 4 {
        let f = x.abs() as f32;
        (f32::from_bits(f.to_bits() + 1) - f) as f64
    } else {
        let f = x.abs();
        f64::from_bits(f.to_bits() + 1) - f
    }
}

/// §159.7's measure: |device - host| in units in the last place of the
/// quantity's operand scale (the state word against the host value rounded
/// to `Scalar`). A mass left after a stage ends inside the step is a
/// difference of the masses before and after, so its own ulps count the
/// cancellation and not the arithmetic; its scale is the mass it came from.
fn ulps_at(dev: f64, host: f64, scale: f64, scalar: bool) -> u64 {
    let h = if scalar { r(host) } else { host };
    if dev == h {
        return 0;
    }
    ((dev - h).abs() / spacing(scale.max(h.abs()), scalar)).ceil() as u64
}

/// §159.5 row 5: 4096 random parcels in 1024 random cells, every stage,
/// sub-cooled and boiling moisture, radiation, steps that end a stage: one
/// device step against the host twin on the same rounded inputs, every
/// state word and every accumulator within 4 ulp of its operand scale
/// (§159.7): a mass of the mass it came from, T_p of itself, m T of
/// m_in T_out, and a heat of the step's integrand (a0 |dT| + eps pi d^2
/// (G/4 + sigma T^4)) dt.
#[test]
fn device_matches_host() {
    let Some(g) = gpu("device_matches_host") else { return };
    let p = props(0.85, 1.0e5);
    let mut rng = SplitMix64(20_261_015);
    let n_cells = 1024;
    let gas: Vec<GasAtParcel> = (0..n_cells)
        .map(|_| {
            let t = r(250.0 + 1950.0 * rng.u());
            GasAtParcel {
                t,
                y_w: r(0.05 * rng.u()),
                rho: r(101_325.0 / (287.0 * t)),
                g: r(4.0 * SIGMA_SB * (2000.0 * rng.u()).powi(4)),
                u_rel: 0.0,
                p_o2: 0.0,
            }
        })
        .collect();
    let n = 4096;
    let mut s = Vec::with_capacity(n);
    let mut cell = Vec::with_capacity(n);
    let mut u_rel = Vec::with_capacity(n);
    for i in 0..n {
        let d = 10.0e-6 + 190.0e-6 * rng.u();
        let rho_p = 500.0 + 1000.0 * rng.u();
        let (fw, t) = match i % 3 {
            0 => (0.001 + 0.2 * rng.u(), 280.0 + 100.0 * rng.u()),
            _ => (0.0, 280.0 + 1520.0 * rng.u()),
        };
        let fv = if i % 5 == 0 { 0.0 } else { 0.7 * rng.u() };
        let fc = 0.25 * rng.u();
        s.push(rounded(&particle(d, rho_p, (fw, fv, fc), t)));
        cell.push((i * 7919) % n_cells);
        u_rel.push(r(5.0 * rng.u()));
    }
    let dt = 2.0e-3;
    let mut pool = Pool::new(&g, &s, &cell, &u_rel, &gas);
    pool.step(&g, &p, dt).expect("device step");
    let sd = pool.states(&g);
    let ed = pool.exchanges(&g);
    let mut worst = [0u64; 9];
    let mut worst_x = [0u64; 9];
    let mut n_crossed = 0usize;
    let (mut ended_w, mut boiling) = (0usize, 0usize);
    for i in 0..n {
        let gc = GasAtParcel { u_rel: u_rel[i], ..gas[cell[i]] };
        let (sh, eh) = step(&p, &s[i], &gc, dt).expect("host step");
        if s[i].m_w > 0.0 && sh.m_w == 0.0 {
            ended_w += 1;
        }
        if s[i].m_w > 0.0 && sh.t == p.drying.t_boil_p {
            boiling += 1;
        }
        let a0 = std::f64::consts::PI * s[i].d * p.k_g * (2.0 + 0.6 * (gc.rho * gc.u_rel * s[i].d / p.mu).sqrt() * p.cbrt_pr);
        let t_hi = s[i].t.max(sh.t).max(gc.t);
        let t_lo = s[i].t.min(sh.t).min(gc.t);
        let ar = std::f64::consts::PI * s[i].d * s[i].d * p.emissivity;
        let q_scale = (a0 * (t_hi - t_lo) + ar * (0.25 * gc.g + SIGMA_SB * t_hi.powi(4))) * dt;
        let row = [
            ulps_at(sd[i].m_w, sh.m_w, s[i].m_w, true),
            ulps_at(sd[i].m_v, sh.m_v, s[i].m_v, true),
            ulps_at(sd[i].t, sh.t, s[i].t, true),
            ulps_at(ed[i].m_w, eh.m_w, s[i].m_w, false),
            ulps_at(ed[i].m_v, eh.m_v, s[i].m_v, false),
            ulps_at(ed[i].mt_w, eh.mt_w, s[i].m_w * sh.t, false),
            ulps_at(ed[i].mt_v, eh.mt_v, s[i].m_v * sh.t, false),
            ulps_at(ed[i].q_conv, eh.q_conv, q_scale, false),
            ulps_at(ed[i].q_rad, eh.q_rad, q_scale, false),
        ];
        // A step in which a sub-cooled particle reaches T_b (§159.7's t_hit)
        // is ill-conditioned: t_hit = -ln(1 - r) tau moves by 1/(1 - r)
        // times any change in r, and the drying splits at t_hit between two
        // rates. Those parcels are their own row.
        let crossed = s[i].m_w > 0.0 && s[i].t < p.drying.t_boil_p && sh.t == p.drying.t_boil_p;
        let w = if crossed {
            n_crossed += 1;
            &mut worst_x
        } else {
            &mut worst
        };
        for (w, v) in w.iter_mut().zip(row) {
            *w = (*w).max(v);
        }
    }
    let names = ["m_w", "m_v", "T_p", "acc m_w", "acc m_v", "acc m_w T", "acc m_v T", "Q_conv", "Q_rad"];
    println!(
        "  device vs host, {} parcels: worst ulps of the operand scale {worst:?}; moisture ended {ended_w}, boiling {boiling}",
        n - n_crossed
    );
    assert!(ended_w > 0 && boiling > 0 && n_crossed > 0, "the random set must end a drying stage, boil and reach T_b");
    for (name, w) in names.iter().zip(worst) {
        assert!(w <= ULPS, "{name} is {w} ulp from its host twin (bar {ULPS})");
    }
    let wx = worst_x.iter().copied().max().unwrap_or(0);
    println!(
        "  device vs host, the {n_crossed} parcels that reach T_b in the step: worst {worst_x:?}, bar {ULPS}: {}",
        if wx <= ULPS { "met" } else { "MISSED - the T_b crossing time is ill-conditioned (SPEC-LIT 159.7)" }
    );
    // A branch taken differently on the device moves a word by orders of
    // magnitude more than this; the crossing parcels must stay below it.
    assert!(wx <= 16 * ULPS, "a crossing parcel is {wx} ulp from its host twin: a branch differs");
    if !cfg!(feature = "single") {
        assert!(wx > ULPS, "the recorded miss is now met ({wx} ulp); update SPEC-LIT 159.7");
    }
}

/// Named refusals, and the moisture is §76's water.
#[test]
fn particle_refusals() {
    let base = ParticleInputs {
        cp: 1500.0,
        emissivity: 0.9,
        devol_a: 1000.0,
        devol_e_a: 3.0e4,
        devol_dh: 1.0e5,
        cp_volatile: 1100.0,
        drying: Drying::water(101_325.0).expect("water"),
        mu: 1.8e-5,
        k_g: 0.06,
        cp_g: 1100.0,
        char_rate: CharInputs::inert(),
    };
    let refused = |i: ParticleInputs, what: &str| match ParticleProps::new(&i) {
        Err(Error::Refused(m)) => assert!(m.contains(what), "{what}: {m}"),
        other => panic!("{what} must be refused, got {other:?}"),
    };
    refused(ParticleInputs { cp: 0.0, ..base }, "cp");
    refused(ParticleInputs { emissivity: 1.5, ..base }, "emissivity");
    refused(ParticleInputs { devol_a: -1.0, ..base }, "devol_a");
    refused(ParticleInputs { mu: f64::NAN, ..base }, "mu");
    refused(ParticleInputs { devol_dh: f64::INFINITY, ..base }, "devol_dh");
    assert!(matches!(Drying::water(1.0e9), Err(Error::Refused(_))), "p_amb beyond T_crit");
    let p = ParticleProps::new(&base).expect("props");
    let s = particle(50.0e-6, 1500.0, (0.1, 0.5, 0.3), 300.0);
    let gas = GasAtParcel { t: 1200.0, y_w: 0.0, rho: 0.3, g: 0.0, u_rel: 0.0, p_o2: 0.0 };
    assert!(matches!(step(&p, &s, &gas, 0.0), Err(Error::Refused(_))), "dt = 0");
    // T_b(1 atm) of Hyland-Wexler within 0.05 K of 373.124 K.
    assert!((p.drying.t_boil_p - 373.124).abs() < 0.05, "T_b = {}", p.drying.t_boil_p);
    let w = ofgpu::parcels::evaporation::LiquidProperties::water();
    assert_eq!(
        [p.drying.w_vap, p.drying.h_v_boil, p.drying.t_crit, p.drying.d_vap, p.drying.cp_vap].map(|v| v as Scalar),
        [w.w_vapour, w.h_v_boil, w.t_crit, w.d_vapour, w.cp_vapour],
        "the moisture is section 76's water"
    );

    let Some(g) = gpu("particle_refusals") else { return };
    let mut pool = Pool::new(&g, &[rounded(&s)], &[0], &[0.0], &[gas]);
    assert!(matches!(pool.step(&g, &p, f64::NAN), Err(Error::Refused(_))), "device dt = NaN");
    pool.n = 2;
    match pool.step(&g, &p, 1.0e-3) {
        Err(Error::Refused(m)) => assert!(m.contains("fewer than n"), "{m}"),
        other => panic!("a short buffer must be refused, got {other:?}"),
    }
}

// ==========================================================================
//  DST-16 part 2, bug A: the all-volatile particle's NaN (SPEC-LIT 159.11)
// ==========================================================================

/// `props(0.9, 0.0)` with the tube's kinetics (159.11): A = 1.5e7 1/s and
/// E_a = 7.4e4 J/mol, dH 0 - the authored inputs, not a starch's measured
/// ones.
fn tube_props() -> ParticleProps {
    ParticleProps::new(&ParticleInputs {
        cp: 1500.0,
        emissivity: 0.9,
        devol_a: 1.5e7,
        devol_e_a: 7.4e4,
        devol_dh: 0.0,
        cp_volatile: 1100.0,
        drying: Drying::water(101_325.0).expect("water at 1 atm"),
        mu: 1.8e-5,
        k_g: 0.06,
        cp_g: 1100.0,
        char_rate: CharInputs::inert(),
    })
    .expect("tube props")
}

/// The tube's frozen gas: 1800 K, no water, no oxygen (the char stays
/// inert), radiation of a 1800 K black enclosure.
fn tube_gas() -> GasAtParcel {
    GasAtParcel {
        t: 1800.0,
        y_w: 0.0,
        rho: 0.196,
        g: 4.0 * SIGMA_SB * 1800.0f64.powi(4),
        u_rel: 0.0,
        p_o2: 0.0,
    }
}

/// A particle of nothing but volatile never meets `STAGE_END`'s fraction -
/// the remaining mass IS `m_v` - so the stage must still end when the
/// remnant drops below the smallest normal double. Before that, (159.4)
/// runs `m_v` into the subnormals, the segment's `tt = tseg / mc`
/// overflows to infinity and `num * tt * p1` is `inf * 0 = NaN`.
#[test]
fn all_volatile_burns_out() {
    let p = tube_props();
    let mut s = particle(1e-6, 1500.0, (0.0, 1.0, 0.0), 300.0);
    let m_v0 = s.m_v;
    let gas = tube_gas();
    let dt = 5.0e-5;
    let mut given = 0.0;
    for j in 1..=400 {
        let (s1, ex) = step(&p, &s, &gas, dt).expect("step");
        for (name, v) in
            [("m_w", s1.m_w), ("m_v", s1.m_v), ("m_c", s1.m_c), ("m_a", s1.m_a), ("t", s1.t), ("d", s1.d)]
        {
            assert!(v.is_finite(), "step {j}: state {name} is not finite ({v:e})");
        }
        for (name, v) in [
            ("m_w", ex.m_w),
            ("m_v", ex.m_v),
            ("mt_w", ex.mt_w),
            ("mt_v", ex.mt_v),
            ("q_conv", ex.q_conv),
            ("q_rad", ex.q_rad),
            ("m_c", ex.m_c),
            ("m_o2", ex.m_o2),
            ("m_co2", ex.m_co2),
            ("mt_c", ex.mt_c),
            ("q_char", ex.q_char),
        ] {
            assert!(v.is_finite(), "step {j}: exchange {name} is not finite ({v:e})");
        }
        given += ex.m_v;
        s = s1;
    }
    assert_eq!(s.m_v, 0.0, "the volatile stage must end at exactly 0");
    let drift = (given - m_v0).abs() / m_v0;
    assert!(drift <= 1e-12, "|sum ex.m_v - m_v0|/m_v0 = {drift:e} against 1e-12");
    println!("  [159.11] all-volatile host: m_v0 {m_v0:.6e} kg in 400 steps, end T {:.3} K", s.t);
}

/// The same particle and gas through the device kernel, one parcel in one
/// cell, 400 launches of dt 5e-5. The kernel's `ofacc` is double in BOTH
/// builds, so before the fix the f64 build turns NaN the same way the host
/// does; under `single` the stored word underflows to 0 first and the test
/// passes there either way.
#[test]
fn device_all_volatile_burns_out() {
    let Some(g) = gpu("device_all_volatile_burns_out") else { return };
    let p = tube_props();
    let s0 = rounded(&particle(1e-6, 1500.0, (0.0, 1.0, 0.0), 300.0));
    let mut gas = tube_gas();
    gas.t = r(gas.t);
    gas.rho = r(gas.rho);
    gas.g = r(gas.g);
    let dt = 5.0e-5;
    let mut pool = Pool::new(&g, &[s0], &[0], &[0.0], &[gas]);
    for j in 1..=400 {
        pool.step(&g, &p, dt).expect("device step");
        let st = pool.states(&g);
        let s = &st[0];
        for (name, v) in
            [("m_w", s.m_w), ("m_v", s.m_v), ("m_c", s.m_c), ("m_a", s.m_a), ("t", s.t), ("d", s.d)]
        {
            assert!(v.is_finite(), "launch {j}: state {name} is not finite ({v:e})");
        }
    }
    let st = pool.states(&g);
    let s = &st[0];
    assert_eq!(s.m_v, 0.0, "the volatile stage must end at exactly 0");
    println!("  [159.11] all-volatile device: end T {:.3} K, m_v {}", s.t, s.m_v);
}
