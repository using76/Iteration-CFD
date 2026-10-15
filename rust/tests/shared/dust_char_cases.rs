// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.
// Provenance: see PROVENANCE.md. No GPL-licensed source was consulted.

// The DST-13 gates of SPEC-LIT §159.5 and §159.8, written once and included
// both by src/dust/tests_char.rs (as dust::char::tests) and by
// tests/dust_char.rs, so both builds run them. The host twin is f64 in both
// builds, but it rounds the char mass to `Scalar` where the state stores
// it (§159.8), and the diameter follows the stored mass; so the closed-form
// rows are class R on the host and the device alike: tol32 = max(tol64,
// min(tol64 2^29, 1e-5)) under `single` (D-F32-1). The element ledger is
// class A in both builds: the carbon given is the difference of two stored
// char words, summed in `Acc`. The retention split is class D: bitwise. A
// GPU test returns early, by name, on a box without a device.
// The specification was informed by a review of FSCDustFoam (Huang &
// Lipatnikov, GPL) and no code was copied.
// No GPL-licensed source was consulted.

use ofgpu::device::{DevBuf, Gpu};
use ofgpu::dust::char::{burn, diff_coef, r_kin, rate, shrink, CharInputs, CharProps, W_C, W_CO2, W_O2};
use ofgpu::dust::device::{particle_step, ExchangeFields, ParticleFields, ParticleGasDev, ParticleKernels};
use ofgpu::dust::particle::{
    step, Drying, Exchange, GasAtParcel, ParticleInputs, ParticleProps, ParticleState, SIGMA_SB,
};
use ofgpu::error::Error;
use ofgpu::{Acc, Label, Scalar};
use std::f64::consts::PI;

/// The device ulp bar, as DST-12's (§159.5), in both builds.
const ULPS: u64 = 4;
/// Air's O2 at 1 atm, Pa.
const P_O2: f64 = 0.21 * 101_325.0;
/// ISL22's heat of C oxidation, J/kg.
const H_C: f64 = 32.9e6;
/// The particle's C_p, J/(kg K).
const CP: f64 = 1500.0;

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

/// DST-12's parcel inputs (C_p 1500, air-like gas, water at 1 atm) with
/// the char reaction `ch`.
fn inputs(emissivity: f64, ch: CharInputs) -> ParticleInputs {
    ParticleInputs {
        cp: CP,
        emissivity,
        devol_a: 1000.0,
        devol_e_a: 3.0e4,
        devol_dh: 1.0e5,
        cp_volatile: 1100.0,
        drying: Drying::water(101_325.0).expect("water at 1 atm"),
        mu: 1.8e-5,
        k_g: 0.06,
        cp_g: 1100.0,
        char_rate: ch,
    }
}

fn props(emissivity: f64, ch: CharInputs) -> ParticleProps {
    ParticleProps::new(&inputs(emissivity, ch)).expect("props")
}

/// A dry, devolatilised sphere: char fraction `f`, ash the rest.
fn solid(d: f64, rho_p: f64, f: f64, t: f64) -> ParticleState {
    let m = rho_p * PI * d * d * d / 6.0;
    ParticleState { m_w: 0.0, m_v: 0.0, m_c: f * m, m_a: (1.0 - f) * m, t, d }
}

/// Air at 1 atm and `t` with O2 at `p_o2` and incident radiation `g`.
fn gas(t: f64, p_o2: f64, g: f64, u_rel: f64) -> GasAtParcel {
    GasAtParcel { t, y_w: 0.0, rho: 101_325.0 / (287.0 * t), g, u_rel, p_o2 }
}

/// The constant density 6 m/(pi d^3) the char stage keeps.
fn density(s: &ParticleState) -> f64 {
    6.0 * s.mass() / (PI * s.d * s.d * s.d)
}

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

/// A device pool: the six state words, the eleven accumulators, and the
/// five gas cell fields.
struct Pool {
    k: ParticleKernels,
    st: Vec<DevBuf<Scalar>>,
    acc: Vec<DevBuf<Acc>>,
    cell: DevBuf<Label>,
    u_rel: DevBuf<Scalar>,
    gas: Vec<DevBuf<Scalar>>,
    n: usize,
    n_cells: usize,
}

impl Pool {
    fn new(g: &Gpu, s: &[ParticleState], cell: &[usize], u_rel: &[f64], gas: &[GasAtParcel]) -> Self {
        let fs: [fn(&ParticleState) -> f64; 6] = [|p| p.m_w, |p| p.m_v, |p| p.m_c, |p| p.m_a, |p| p.t, |p| p.d];
        let fg: [fn(&GasAtParcel) -> f64; 5] = [|c| c.t, |c| c.y_w, |c| c.rho, |c| c.g, |c| c.p_o2];
        let cells: Vec<Label> = cell.iter().map(|&c| c as Label).collect();
        Self {
            k: ParticleKernels::new(g).expect("kernels"),
            st: fs.iter().map(|f| up(g, &s.iter().map(f).collect::<Vec<_>>())).collect(),
            acc: (0..11).map(|_| g.zeros::<Acc>(s.len()).expect("acc")).collect(),
            cell: g.upload(&cells).expect("cells"),
            u_rel: up(g, u_rel),
            gas: fg.iter().map(|f| up(g, &gas.iter().map(f).collect::<Vec<_>>())).collect(),
            n: s.len(),
            n_cells: gas.len(),
        }
    }

    fn step(&mut self, g: &Gpu, p: &ParticleProps, dt: f64) -> Result<(), Error> {
        let [w, v, c, a, t, d] = &mut self.st[..] else { unreachable!() };
        let [a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10] = &mut self.acc[..] else { unreachable!() };
        let [tg, yw, rho, gr, po2] = &self.gas[..] else { unreachable!() };
        particle_step(
            g,
            &self.k,
            p,
            ParticleFields { m_w: w, m_v: v, m_c: c, m_a: a, t, d },
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
                t: tg,
                y_w: yw,
                rho,
                g: gr,
                p_o2: po2,
                n_cells: self.n_cells,
            },
            dt,
            self.n,
        )
    }

    fn states(&self, g: &Gpu) -> Vec<ParticleState> {
        let c: Vec<Vec<f64>> =
            self.st.iter().map(|b| g.download(b).expect("state").iter().map(|&x| x as f64).collect()).collect();
        (0..self.n)
            .map(|i| ParticleState { m_w: c[0][i], m_v: c[1][i], m_c: c[2][i], m_a: c[3][i], t: c[4][i], d: c[5][i] })
            .collect()
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

/// `n` steps of one parcel on the host, or on the device when `dev` is
/// given, calling `each(j, state)` after every step.
fn walk(p: &ParticleProps, s0: ParticleState, gs: GasAtParcel, dt: f64, n: usize, dev: Option<&Gpu>, mut each: impl FnMut(usize, &ParticleState)) -> ParticleState {
    let mut pool = dev.map(|g| Pool::new(g, &[s0], &[0], &[gs.u_rel], &[gs]));
    let mut s = s0;
    for j in 1..=n {
        s = match (&mut pool, dev) {
            (Some(pl), Some(g)) => {
                pl.step(g, p, dt).expect("device step");
                pl.states(g)[0]
            }
            _ => step(p, &s, &gs, dt).expect("step").0,
        };
        each(j, &s);
    }
    s
}

/// §159.5, (159.6): diffusion-limited char (R_kin = 1e12, so C'/(R_kin d)
/// is below 1e-16) on a pure-char 50 um particle at T_p = T_g = 1500 K with
/// no radiation and h_s = 0, so T_p stays 1500 K exactly and d^2 falls by
/// 4 p_O2 C_0 T_m^(3/4)/rho_p per second. 40 steps to 90 % burnout by mass;
/// d^2 within 1e-8 relative of the d^2 law, host and device (R: the host
/// stores the char mass in `Scalar` too).
#[test]
fn d2_law() {
    let p = props(0.0, CharInputs { c0: 5.0e-12, a: 1.0e12, e_a: 0.0, h_c: H_C, h_s: 0.0 });
    let gs = gas(1500.0, P_O2, 0.0, 0.0);
    let run = |s0: ParticleState, dev: Option<&Gpu>| -> (f64, f64) {
        let kd = 4.0 * P_O2 * diff_coef(&p.char_rate, 1500.0, 1500.0) / density(&s0);
        let n = 40;
        let dt = (s0.d * s0.d) * (1.0 - 0.1f64.powf(2.0 / 3.0)) / kd / n as f64;
        let mut worst: f64 = 0.0;
        let end = walk(&p, s0, gs, dt, n, dev, |j, s| {
            assert_eq!(s.t, s0.t, "step {j}: T_p moved off 1500 K");
            let ex = s0.d * s0.d - kd * dt * j as f64;
            worst = worst.max((s.d * s.d - ex).abs() / ex);
        });
        (worst, end.m_c / s0.m_c)
    };
    let s0 = solid(50.0e-6, 1000.0, 1.0, 1500.0);
    let (worst, left) = run(s0, None);
    println!("  d2 law, host: worst |d^2/exact - 1| = {worst:.3e} over 40 steps; char left {left:.6}");
    assert!(worst <= tol(1.0e-8), "host d^2 is {worst:.3e} from (159.6) (bar {:.0e})", tol(1.0e-8));
    assert!((left - 0.1).abs() <= 1.0e-6, "the run must end at 90 % burnout, char left {left}");
    let Some(g) = gpu("d2_law") else { return };
    let (worst_d, _) = run(rounded(&s0), Some(&g));
    println!("  d2 law, device: worst |d^2/exact - 1| = {worst_d:.3e} (bar {:.0e})", tol(1.0e-8));
    assert!(worst_d <= tol(1.0e-8), "device d^2 is {worst_d:.3e} from (159.6)");
}

/// §159.5, (159.6): kinetic-limited char (C_0 = 1e6, so R_kin d/C' is below
/// 1e-15) at T_g = T_p = 1500 K, A = 1 kg/(m2 s Pa), E_a = 80 kJ/mol: d
/// falls linearly at 2 p_O2 R_kin/rho_p. 40 steps to 90 % burnout by mass;
/// d within 1e-10 relative of the line, host and device (R).
#[test]
fn kinetic_linear() {
    let p = props(0.0, CharInputs { c0: 1.0e6, a: 1.0, e_a: 8.0e4, h_c: H_C, h_s: 0.0 });
    let gs = gas(1500.0, P_O2, 0.0, 0.0);
    let run = |s0: ParticleState, dev: Option<&Gpu>| -> (f64, f64) {
        let v = 2.0 * P_O2 * r_kin(&p.char_rate, 1500.0) / density(&s0);
        let n = 40;
        let dt = s0.d * (1.0 - 0.1f64.cbrt()) / v / n as f64;
        let mut worst: f64 = 0.0;
        let end = walk(&p, s0, gs, dt, n, dev, |j, s| {
            assert_eq!(s.t, s0.t, "step {j}: T_p moved off 1500 K");
            let ex = s0.d - v * dt * j as f64;
            worst = worst.max((s.d - ex).abs() / ex);
        });
        (worst, end.m_c / s0.m_c)
    };
    let s0 = solid(50.0e-6, 1000.0, 1.0, 1500.0);
    let (worst, left) = run(s0, None);
    println!("  kinetic, host: worst |d/exact - 1| = {worst:.3e} over 40 steps; char left {left:.6}");
    assert!(worst <= tol(1.0e-10), "host d is {worst:.3e} from the line of (159.6) (bar {:.0e})", tol(1.0e-10));
    assert!((left - 0.1).abs() <= 1.0e-6, "the run must end at 90 % burnout, char left {left}");
    let Some(g) = gpu("kinetic_linear") else { return };
    let (worst_d, _) = run(rounded(&s0), Some(&g));
    println!("  kinetic, device: worst |d/exact - 1| = {worst_d:.3e} (bar {:.0e})", tol(1.0e-10));
    assert!(worst_d <= tol(1.0e-10), "device d is {worst_d:.3e} from the line of (159.6)");
}

/// §159.5: (159.5)'s harmonic combination tends to the diffusion rate
/// pi d p_O2 C' as R_kin/R_diff -> infinity and to the kinetic rate
/// pi d^2 p_O2 R_kin as R_diff/R_kin -> infinity, to 1e-12 at a ratio of
/// 1e16; the shrinkage (159.10) tends to the two closed forms of (159.6) to
/// 1e-12 of d_0; and at finite rates it satisfies its own integral
/// C' D + R_kin (d_0 D - D^2/2) = 2 p_O2 C' R_kin t/rho_p to 1e-13. Host
/// arithmetic, f64 in both builds.
#[test]
fn combined_limits() {
    let (d, tg, tp) = (50.0e-6, 1500.0, 1300.0);
    let ch = |c0, a, e_a| CharProps::new(&CharInputs { c0, a, e_a, h_c: H_C, h_s: 0.3 }).expect("char");
    let base = ch(5.0e-12, 1.0, 8.0e4);
    let cd = diff_coef(&base, tg, tp);
    let rk = r_kin(&base, tg);
    let mut last = (1.0f64, 1.0f64);
    for e in [4, 8, 12, 16] {
        let f = 10f64.powi(e);
        // R_kin = f R_diff, with E_a = 0 so that R_kin is A.
        let to_diff = (rate(&ch(5.0e-12, f * cd / d, 0.0), d, P_O2, tg, tp) / (PI * d * P_O2 * cd) - 1.0).abs();
        // R_diff = f R_kin: C_0 scaled so that C'/d = f R_kin.
        let c0 = f * rk * d / (0.5 * (tg + tp)).powf(0.75);
        let to_kin = (rate(&ch(c0, 1.0, 8.0e4), d, P_O2, tg, tp) / (PI * d * d * P_O2 * rk) - 1.0).abs();
        println!("  limits, ratio 1e{e}: |rate/diffusion - 1| = {to_diff:.3e}, |rate/kinetic - 1| = {to_kin:.3e}");
        assert!(to_diff <= 2.0 / f + 1.0e-15 && to_kin <= 2.0 / f + 1.0e-15, "the combination is off its limits at 1e{e}");
        assert!(to_diff <= last.0 && to_kin <= last.1, "the combination must approach its limits monotonically");
        last = (to_diff, to_kin);
    }
    assert!(last.0 <= 1.0e-12 && last.1 <= 1.0e-12, "the limits at 1e16: {last:?} (bar 1e-12)");
    // The shrinkage (159.10) against the two closed forms at a ratio of 1e16.
    let rho = 1000.0;
    let t_d = 0.5 * d * d * rho / (4.0 * P_O2 * cd);
    let sh_d = shrink(cd, 1.0e16 * cd / d, P_O2, rho, d, t_d);
    let ex_d = d - (d * d - 4.0 * P_O2 * cd * t_d / rho).sqrt();
    let t_k = 0.5 * d * rho / (2.0 * P_O2 * rk);
    let sh_k = shrink(1.0e16 * rk * d, rk, P_O2, rho, d, t_k);
    let ex_k = 2.0 * P_O2 * rk * t_k / rho;
    let (e_d, e_k) = ((sh_d - ex_d).abs() / d, (sh_k - ex_k).abs() / d);
    println!("  shrinkage at 1e16: |D - d2 law|/d_0 = {e_d:.3e}, |D - line|/d_0 = {e_k:.3e}");
    assert!(e_d <= 1.0e-12 && e_k <= 1.0e-12, "the shrinkage is off its limits (bar 1e-12)");
    // At finite rates, its own integral.
    let kk = 2.0 * P_O2 * cd * rk / rho;
    let t_out = d * (cd + rk * 0.5 * d) / kk;
    let mut worst: f64 = 0.0;
    for j in 1..=20 {
        let t = t_out * j as f64 / 21.0;
        let dd = shrink(cd, rk, P_O2, rho, d, t);
        let lhs = cd * dd + rk * (d * dd - 0.5 * dd * dd);
        worst = worst.max((lhs - kk * t).abs() / (kk * t));
    }
    println!("  shrinkage integral, combined rates: worst rel {worst:.3e}");
    assert!(worst <= 1.0e-13, "(159.10) leaves its integral by {worst:.3e} (bar 1e-13)");
}

/// The element ledger of a run: carbon, O2 and CO2 summed.
fn ledger(e: &[Exchange]) -> (f64, f64, f64) {
    e.iter().fold((0.0, 0.0, 0.0), |a, x| (a.0 + x.m_c, a.1 + x.m_o2, a.2 + x.m_co2))
}

/// §159.5 row element_balance (A): an 80 um parcel of 70 % char and 30 %
/// ash and a 30 um pure-char parcel burn out in a radiating 1400 K gas with
/// h_s = 0.3. The carbon given equals the char lost to 1e-14 relative; the
/// O2 taken is W_O2/W_C of it and the CO2 given is their sum, and the CO2
/// carries the carbon and the oxygen back, each to 1e-14. The char ends at
/// exactly 0, the ash is unchanged, the ash particle keeps d_0 (m_a/m_0)^(1/3)
/// to 1e-12 (R on the device) and the pure-char particle ends with no mass,
/// no diameter and a finite T_p. Host and device.
#[test]
fn element_balance() {
    let p = props(0.9, CharInputs { c0: 5.0e-12, a: 1.0, e_a: 8.0e4, h_c: H_C, h_s: 0.3 });
    let gs = gas(1400.0, P_O2, 4.0 * SIGMA_SB * 1400f64.powi(4), 0.5);
    let dt = 1.0e-3;
    let check = |who: &str, s0: &[ParticleState], end: &[ParticleState], ex: &[Vec<Exchange>]| {
        for (k, (a, b)) in s0.iter().zip(end).enumerate() {
            let (c, o2, co2) = ledger(&ex[k]);
            let lost = ((a.m_c - b.m_c) - c).abs() / a.m_c;
            let e_o2 = (o2 - c * (W_O2 / W_C)).abs() / o2;
            let e_co2 = (co2 - (c + o2)).abs() / co2;
            let e_cb = (co2 * (W_C / W_CO2) - c).abs() / c;
            let e_ob = (co2 * (W_O2 / W_CO2) - o2).abs() / o2;
            println!(
                "  element balance, {who} parcel {k}: carbon {lost:.3e}, O2 {e_o2:.3e}, CO2 = C + O2 {e_co2:.3e}, C in CO2 {e_cb:.3e}, O in CO2 {e_ob:.3e}"
            );
            for (name, v) in [("carbon", lost), ("O2", e_o2), ("CO2 mass", e_co2), ("C in CO2", e_cb), ("O in CO2", e_ob)] {
                assert!(v <= 1.0e-14, "{who} parcel {k}: the {name} balance is off by {v:.3e} (bar 1e-14)");
            }
            assert_eq!((b.m_w, b.m_v, b.m_c, b.m_a), (0.0, 0.0, 0.0, a.m_a), "{who} parcel {k}: the char must burn out");
            assert!(b.t.is_finite() && b.t > 0.0, "{who} parcel {k}: T_p = {}", b.t);
            if a.m_a > 0.0 {
                let d_end = a.d * (a.m_a / a.mass()).cbrt();
                let e_d = (b.d - d_end).abs() / d_end;
                let bar = if who == "host" { 1.0e-12 } else { tol(1.0e-12) };
                assert!(e_d <= bar, "{who} parcel {k}: d_end is {e_d:.3e} off d_0 (m_a/m_0)^(1/3)");
            } else {
                assert_eq!(b.d, 0.0, "{who} parcel {k}: a pure-char particle ends with no diameter");
            }
        }
    };
    let s0 = [solid(80.0e-6, 800.0, 0.7, 1400.0), solid(30.0e-6, 800.0, 1.0, 1400.0)];
    let mut ex = vec![Vec::new(), Vec::new()];
    let mut end = s0;
    let mut n_steps = 0;
    for j in 1..=2000 {
        for k in 0..2 {
            let (s1, e) = step(&p, &end[k], &gs, dt).expect("step");
            ex[k].push(e);
            end[k] = s1;
        }
        n_steps = j;
        if end.iter().all(|s| s.m_c == 0.0) {
            break;
        }
    }
    println!("  element balance, host: both burnt out in {n_steps} steps of 1 ms");
    check("host", &s0, &end, &ex);

    let Some(g) = gpu("element_balance") else { return };
    let sr = [rounded(&s0[0]), rounded(&s0[1])];
    let mut pool = Pool::new(&g, &sr, &[0, 0], &[gs.u_rel, gs.u_rel], &[gs]);
    for _ in 0..n_steps + 5 {
        pool.step(&g, &p, dt).expect("device step");
    }
    let e = pool.exchanges(&g);
    check("device", &sr, &pool.states(&g), &[vec![e[0]], vec![e[1]]]);
}

/// §159.5 row retention_split (D): one step of a 60 um parcel with 80 %
/// char in a radiating 1500 K gas. With h_s = 0.3 the gas's share is the
/// word (1 - 0.3) (m_c h_c) bit for bit, and the particle's share,
/// m C (T_1 - T_0) - Q_conv - Q_rad, is 0.3 m_c h_c to 1e-12 (R on the
/// device). With h_s = 1 the gas's share is exactly 0. With h_s = 0, a
/// particle at T_g and no radiation keeps T_p bit for bit and the gas takes
/// m_c h_c bit for bit. Host and device.
#[test]
fn retention_split() {
    let dt = 1.0e-3;
    let s0 = solid(60.0e-6, 800.0, 0.8, 1500.0);
    let hot = gas(1500.0, P_O2, 4.0 * SIGMA_SB * 1500f64.powi(4), 1.0);
    let ch = |h_s| CharInputs { c0: 5.0e-12, a: 1.0, e_a: 8.0e4, h_c: H_C, h_s };
    let g = gpu("retention_split");
    let one = |p: &ParticleProps, s: &ParticleState, gs: &GasAtParcel| -> Vec<(ParticleState, Exchange)> {
        let mut out = vec![step(p, s, gs, dt).expect("step")];
        if let Some(g) = &g {
            let mut pool = Pool::new(g, &[rounded(s)], &[0], &[gs.u_rel], &[*gs]);
            pool.step(g, p, dt).expect("device step");
            out.push((pool.states(g)[0], pool.exchanges(g)[0]));
        }
        out
    };
    let who = ["host", "device"];
    // h_s = 0.3.
    let p = props(0.9, ch(0.3));
    for (k, (s1, e)) in one(&p, &s0, &hot).into_iter().enumerate() {
        let a = if k == 0 { s0 } else { rounded(&s0) };
        assert!(e.m_c > 0.0 && s1.m_c > 0.0, "{}: the char must burn and remain", who[k]);
        assert_eq!(e.q_char, (1.0 - 0.3) * (e.m_c * H_C), "{}: the gas share is not (1 - h_s) m_c h_c bit for bit", who[k]);
        let share = a.mass() * CP * (s1.t - a.t) - e.q_conv - e.q_rad;
        let rel = (share - 0.3 * e.m_c * H_C).abs() / (0.3 * e.m_c * H_C);
        let bar = if k == 0 { 1.0e-12 } else { tol(1.0e-12) };
        println!("  retention, {}: |particle share/(h_s m_c h_c) - 1| = {rel:.3e} (bar {bar:.0e}); T_p {:.2} -> {:.2} K", who[k], a.t, s1.t);
        assert!(rel <= bar, "{}: the particle's share is {rel:.3e} off h_s m_c h_c", who[k]);
    }
    // h_s = 1: all of the heat stays in the particle.
    let p = props(0.9, ch(1.0));
    for (k, (_, e)) in one(&p, &s0, &hot).into_iter().enumerate() {
        assert!(e.m_c > 0.0, "{}: h_s = 1, the char must burn", who[k]);
        assert_eq!(e.q_char, 0.0, "{}: h_s = 1 gives the gas nothing", who[k]);
    }
    // h_s = 0: none of it does.
    let p = props(0.0, ch(0.0));
    let still = gas(1500.0, P_O2, 0.0, 1.0);
    for (k, (s1, e)) in one(&p, &s0, &still).into_iter().enumerate() {
        assert!(e.m_c > 0.0, "{}: h_s = 0, the char must burn", who[k]);
        assert_eq!(s1.t, r(1500.0), "{}: h_s = 0 at T_g must keep T_p bit for bit", who[k]);
        assert_eq!(e.q_char, e.m_c * H_C, "{}: h_s = 0 gives the gas all of m_c h_c bit for bit", who[k]);
    }
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

/// |device - host| in ulps of the operand scale (§159.7's measure).
fn ulps_at(dev: f64, host: f64, scale: f64, scalar: bool) -> u64 {
    let h = if scalar { r(host) } else { host };
    if dev == h {
        return 0;
    }
    ((dev - h).abs() / spacing(scale.max(h.abs()), scalar)).ceil() as u64
}

/// §159.8: 4096 random dry parcels in 1024 cells (gas 600-2200 K, p_O2 up
/// to 25 kPa and none in every eighth cell, radiation), every fourth one
/// pure char: one 5 ms device step against the host twin on the same
/// rounded inputs. Parcels burn kinetic- and diffusion-limited, burn out
/// with ash left or with nothing left, or do not burn. Every state word,
/// the carbon given and the two heats are within 4 ulp of their operand
/// scale: a mass of the char it came from, d of itself, T_p of T plus the
/// retained heat of the char h_s h_c m_c/(m C), and a heat of DST-12's
/// integrand plus h_s m_c h_c. The four words formed from the carbon given
/// (O2, CO2, carbon times the stored T_p, and the gas's heat) are its
/// products on the device bit for bit, as they are on the host.
#[test]
fn char_device_matches_host() {
    let Some(g) = gpu("char_device_matches_host") else { return };
    let p = props(0.85, CharInputs { c0: 5.0e-12, a: 1.0, e_a: 8.0e4, h_c: H_C, h_s: 0.3 });
    let ch = p.char_rate;
    let mut rng = SplitMix64(2_026_101_513);
    let n_cells = 1024;
    let cells: Vec<GasAtParcel> = (0..n_cells)
        .map(|c| {
            let t = r(600.0 + 1600.0 * rng.u());
            let po2 = if c % 8 == 0 { 0.0 } else { r(25.0e3 * rng.u()) };
            let gc = gas(t, po2, r(4.0 * SIGMA_SB * (2000.0 * rng.u()).powi(4)), 0.0);
            GasAtParcel { rho: r(gc.rho), ..gc }
        })
        .collect();
    let n = 4096;
    let mut s = Vec::with_capacity(n);
    let mut cell = Vec::with_capacity(n);
    let mut u_rel = Vec::with_capacity(n);
    for i in 0..n {
        let d = 10.0e-6 + 190.0e-6 * rng.u();
        let f = if i % 4 == 0 { 1.0 } else { 0.05 + 0.9 * rng.u() };
        s.push(rounded(&solid(d, 300.0 + 1200.0 * rng.u(), f, 600.0 + 1600.0 * rng.u())));
        cell.push((i * 7919) % n_cells);
        u_rel.push(r(5.0 * rng.u()));
    }
    let dt = 5.0e-3;
    let mut pool = Pool::new(&g, &s, &cell, &u_rel, &cells);
    pool.step(&g, &p, dt).expect("device step");
    let sd = pool.states(&g);
    let ed = pool.exchanges(&g);
    let mut worst = [0u64; 7];
    let (mut out_ash, mut out_gone, mut partial, mut none) = (0, 0, 0, 0);
    for i in 0..n {
        let gc = GasAtParcel { u_rel: u_rel[i], ..cells[cell[i]] };
        let (sh, eh) = step(&p, &s[i], &gc, dt).expect("host step");
        match (eh.m_c > 0.0, sh.m_c == 0.0, sh.mass() == 0.0) {
            (false, _, _) => none += 1,
            (true, true, true) => out_gone += 1,
            (true, true, false) => out_ash += 1,
            (true, false, _) => partial += 1,
        }
        let a0 = PI * s[i].d * p.k_g * (2.0 + 0.6 * (gc.rho * gc.u_rel * s[i].d / p.mu).sqrt() * p.cbrt_pr);
        let t_hi = s[i].t.max(sh.t).max(gc.t);
        let t_lo = s[i].t.min(sh.t).min(gc.t);
        let ar = PI * s[i].d * s[i].d * p.emissivity;
        let chem = ch.h_s * ch.h_c * s[i].m_c;
        let q_scale = (a0 * (t_hi - t_lo) + ar * (0.25 * gc.g + SIGMA_SB * t_hi.powi(4))) * dt + chem;
        let t_scale = t_hi + chem / (s[i].mass() * CP);
        let row = [
            ulps_at(sd[i].m_c, sh.m_c, s[i].m_c, true),
            ulps_at(sd[i].d, sh.d, s[i].d, true),
            ulps_at(sd[i].t, sh.t, t_scale, true),
            ulps_at(ed[i].m_c, eh.m_c, s[i].m_c, false),
            ulps_at(ed[i].q_conv, eh.q_conv, q_scale, false),
            ulps_at(ed[i].q_rad, eh.q_rad, q_scale, false),
            ulps_at(sd[i].m_a, sh.m_a, s[i].m_a, true),
        ];
        for (w, v) in worst.iter_mut().zip(row) {
            *w = (*w).max(v);
        }
        for (who, e, t) in [("device", &ed[i], sd[i].t), ("host", &eh, r(sh.t))] {
            let formed = [e.m_c * ch.o2_per_c, e.m_c * ch.co2_per_c, e.m_c * t, ch.gas_share * (e.m_c * ch.h_c)];
            assert_eq!([e.m_o2, e.m_co2, e.mt_c, e.q_char], formed, "{who} parcel {i}: O2, CO2, m_c T or q_char is not formed from the carbon given");
        }
    }
    println!(
        "  char device vs host, {n} parcels (burnt out with ash {out_ash}, with nothing left {out_gone}, partly {partial}, not burning {none}): worst ulps {worst:?}"
    );
    assert!(out_ash > 0 && out_gone > 0 && partial > 0 && none > 0, "the random set must cover every char outcome");
    let names = ["m_c", "d", "T_p", "acc m_c", "Q_conv", "Q_rad", "m_a"];
    for (name, w) in names.iter().zip(worst) {
        assert!(w <= ULPS, "{name} is {w} ulp from its host twin (bar {ULPS})");
    }
}

/// Named refusals of the char inputs, through `CharProps` and through
/// `ParticleProps`; and `burn` does nothing without char, oxygen or a rate.
#[test]
fn char_refusals() {
    let base = CharInputs { c0: 5.0e-12, a: 1.0, e_a: 8.0e4, h_c: H_C, h_s: 0.3 };
    for (i, what) in [
        (CharInputs { c0: -1.0, ..base }, "c0"),
        (CharInputs { a: f64::NAN, ..base }, "a ="),
        (CharInputs { e_a: -1.0, ..base }, "e_a"),
        (CharInputs { h_c: f64::INFINITY, ..base }, "h_c"),
        (CharInputs { h_s: 1.5, ..base }, "h_s"),
        (CharInputs { h_s: -0.1, ..base }, "h_s"),
    ] {
        match CharProps::new(&i) {
            Err(Error::Refused(m)) => assert!(m.contains(what) && m.contains("dust::char"), "{what}: {m}"),
            other => panic!("{what} must be refused, got {other:?}"),
        }
        assert!(matches!(ParticleProps::new(&inputs(0.9, i)), Err(Error::Refused(_))), "{what} through ParticleProps");
    }
    let c = CharProps::new(&base).expect("char");
    let inert = CharProps::new(&CharInputs::inert()).expect("inert");
    let m = 1.0e-10;
    assert!(burn(&c, m, 0.0, 50.0e-6, 1500.0, 1500.0, 0.0, 1.0e-3).is_none(), "no oxygen");
    assert!(burn(&c, m, 0.0, 50.0e-6, 1500.0, 1500.0, f64::NAN, 1.0e-3).is_none(), "p_O2 = NaN");
    assert!(burn(&c, 0.0, m, 50.0e-6, 1500.0, 1500.0, P_O2, 1.0e-3).is_none(), "no char");
    assert!(burn(&inert, m, 0.0, 50.0e-6, 1500.0, 1500.0, P_O2, 1.0e-3).is_none(), "inert char");
    assert_eq!(rate(&inert, 50.0e-6, P_O2, 1500.0, 1500.0), 0.0, "an inert char has no rate");
}

/// A parcel that loses all of its mass inside a step (§159.8's stop): a
/// pure-moisture parcel dries out in one step and keeps a finite T_p and
/// finite heats, on the host and the device. Before DST-13 the next segment
/// divided by m C = 0 and returned NaN.
#[test]
fn vanished_particle_stays_finite() {
    let p = props(0.9, CharInputs::inert());
    let d = 10.0e-6;
    let m = 1000.0 * PI * d * d * d / 6.0;
    let s = ParticleState { m_w: m, m_v: 0.0, m_c: 0.0, m_a: 0.0, t: 300.0, d };
    let gs = gas(1500.0, 0.0, 0.0, 0.0);
    let (s1, ex) = step(&p, &s, &gs, 0.1).expect("step");
    assert_eq!(s1.m_w, 0.0, "the moisture must be gone");
    assert!(s1.t.is_finite() && ex.q_conv.is_finite() && ex.q_rad.is_finite(), "host T_p = {}, Q_conv = {}", s1.t, ex.q_conv);
    let Some(g) = gpu("vanished_particle_stays_finite") else { return };
    let mut pool = Pool::new(&g, &[rounded(&s)], &[0], &[0.0], &[gs]);
    pool.step(&g, &p, 0.1).expect("device step");
    let (sd, ed) = (pool.states(&g)[0], pool.exchanges(&g)[0]);
    assert_eq!(sd.m_w, 0.0, "device: the moisture must be gone");
    assert!(sd.t.is_finite() && ed.q_conv.is_finite() && ed.q_rad.is_finite(), "device T_p = {}, Q_conv = {}", sd.t, ed.q_conv);
}
