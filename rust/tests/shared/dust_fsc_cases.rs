// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.
// Provenance: see PROVENANCE.md. No GPL-licensed source was consulted.

// The DST-03 gates of SPEC-LIT §154.9, written once and included both by
// src/dust/tests_fsc.rs (as dust::fsc::tests) and by tests/dust_fsc.rs, so
// both builds run them. The host twins are f64 in both builds; the device
// rows compare a `Scalar` kernel output against the host value rounded to
// `Scalar`, in units in the last place: 4 in either build (D-F32-1's same
// ulps). A GPU test returns early, by name, on a box without a device.
// The specification was informed by a review of FSCDustFoam (Huang &
// Lipatnikov, GPL) and no code was copied.
// No GPL-licensed source was consulted.

use ofgpu::device::{DevBuf, Gpu};
use ofgpu::dust::device::{arrhenius as launch_arrhenius, fsc as launch_fsc};
use ofgpu::dust::device::{FscFields, FscInputsDev, FscKernels};
use ofgpu::dust::fsc::{
    arrhenius, bracket_closed, bracket_series, development_time, evaluate, implicit_split, q_source,
    u_t_isp_identity, zero_exponent, CombustionModel, FscConstants, FscInputs, FscOut, FscState,
    BETA_STAR, C_D,
};
use ofgpu::dust::psi::reaction_time;
use ofgpu::error::Error;
use ofgpu::Scalar;

/// The device ulp bar of §154.9, the same count in both builds.
const ULPS: u64 = 4;

/// Leeds (H20 Tables 1-4, SPEC-LIT §151.5): S_L = 0.12 m/s,
/// kappa_u = mu_u/(rho_u Pr) and kappa_b = mu_b/(rho_b Pr) with Pr = 0.7,
/// Pr_t = 0.4, T_b = 1500 K, and t_r from §153.8's published-closure
/// Psi = 1.5445279792249426e-4 by (153.2).
const S_L: f64 = 0.12;
const PSI_LEEDS: f64 = 1.5445279792249426e-4;

fn kappa_u() -> f64 {
    1.8e-5 / (1.32 * 0.7)
}

fn inputs(model: CombustionModel, t_flacc: Option<f64>) -> FscInputs {
    FscInputs {
        model,
        a: 0.4,
        s_l: S_L,
        kappa_u: kappa_u(),
        pr_t: 0.4,
        t_r: reaction_time(PSI_LEEDS, kappa_u(), S_L).expect("t_r"),
        t_b: 1500.0,
        kappa_b: 4.6e-5 / (0.248 * 0.7),
        t_flacc,
    }
}

fn constants(model: CombustionModel, t_flacc: Option<f64>) -> FscConstants {
    FscConstants::new(&inputs(model, t_flacc)).expect("the Leeds constants are admitted")
}

/// SplitMix64, seed 20261014; `u()` is uniform on [0, 1) as (x >> 11) 2^-53.
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

/// A value as the device sees it: rounded to `Scalar`, widened back.
fn r(v: f64) -> f64 {
    v as Scalar as f64
}

/// One random state (T, c~, k, omega, nu_t, rho), already rounded to
/// `Scalar`: T = 150 + 3350 u, c~ = u, k = 10^(-2 + 4u),
/// omega = 10^(1 + 4u), nu_t = (k/omega)(0.5 + 0.5u) (the SST's limiter at
/// most halving k/omega), rho = 0.2 + 2u.
fn random_state(rng: &mut SplitMix64) -> (FscState, f64, f64) {
    let t = r(150.0 + 3350.0 * rng.u());
    let c = r(rng.u());
    let k = r(10f64.powf(-2.0 + 4.0 * rng.u()));
    let omega = r(10f64.powf(1.0 + 4.0 * rng.u()));
    let nu_t = r((k / omega) * (0.5 + 0.5 * rng.u()));
    let rho = r(0.2 + 2.0 * rng.u());
    (FscState { t, k, omega, nu_t }, c, rho)
}

/// 64 clocks (t_fd, t): group 0 is t_fd = 0; the rest t_fd = 10^(-6 + 7u)
/// s with t = t_fd + 1 ms (t_0 of §151.5), so both sides of t_flacc occur.
fn clocks(rng: &mut SplitMix64) -> Vec<(f64, f64)> {
    let mut out = vec![(0.0, 0.0005)];
    for _ in 1..64 {
        let t_fd = 10f64.powf(-6.0 + 7.0 * rng.u());
        out.push((t_fd, t_fd + 1.0e-3));
    }
    out
}

/// A word's bits, in the build's width.
fn bits(v: Scalar) -> u64 {
    if std::mem::size_of::<Scalar>() == 8 {
        (v as f64).to_bits()
    } else {
        (v as f32).to_bits() as u64
    }
}

/// The distance in units in the last place between a device word and the
/// host value rounded to `Scalar`; `u64::MAX` across a sign change. Two
/// zeros of the same sign are 0 apart.
fn ulps(dev: Scalar, host: f64) -> u64 {
    let h = host as Scalar;
    if dev.is_sign_negative() != h.is_sign_negative() {
        return u64::MAX;
    }
    bits(dev).abs_diff(bits(h))
}

/// A nonzero word below `Scalar::MIN_POSITIVE` in magnitude.
fn subnormal(v: Scalar) -> bool {
    v != 0.0 && v.abs() < Scalar::MIN_POSITIVE
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

/// The five output words of one `dustFsc` launch, and its zero counter.
struct Launch {
    d_t: Vec<Scalar>,
    u_t: Vec<Scalar>,
    u_t_isp: Vec<Scalar>,
    tau_l: Vec<Scalar>,
    k_rate: Vec<Scalar>,
    n_zero: u64,
}

fn run_fsc(g: &Gpu, kern: &FscKernels, c: &FscConstants, s: &[FscState], clock: (f64, f64)) -> Launch {
    let n = s.len();
    let t = up(g, &s.iter().map(|x| x.t).collect::<Vec<_>>());
    let k = up(g, &s.iter().map(|x| x.k).collect::<Vec<_>>());
    let w = up(g, &s.iter().map(|x| x.omega).collect::<Vec<_>>());
    let nut = up(g, &s.iter().map(|x| x.nu_t).collect::<Vec<_>>());
    let mut d_t = g.zeros::<Scalar>(n).expect("zeros");
    let mut u_t = g.zeros::<Scalar>(n).expect("zeros");
    let mut isp = g.zeros::<Scalar>(n).expect("zeros");
    let mut tl = g.zeros::<Scalar>(n).expect("zeros");
    let mut kr = g.zeros::<Scalar>(n).expect("zeros");
    let mut nz = g.zeros::<u64>(1).expect("zeros");
    launch_fsc(
        g,
        kern,
        c,
        FscFields { d_t: &mut d_t, u_t: &mut u_t, u_t_isp: &mut isp, tau_l: &mut tl, k_rate: &mut kr },
        &mut nz,
        FscInputsDev { t: &t, k: &k, omega: &w, nu_t: &nut },
        clock.0,
        clock.1,
        n,
    )
    .expect("dustFsc launches");
    Launch {
        d_t: g.download(&d_t).expect("download"),
        u_t: g.download(&u_t).expect("download"),
        u_t_isp: g.download(&isp).expect("download"),
        tau_l: g.download(&tl).expect("download"),
        k_rate: g.download(&kr).expect("download"),
        n_zero: g.download(&nz).expect("download")[0],
    }
}

/// The worst ulp distance of each of the five outputs, and the host's
/// zero count, over one launch.
fn compare(l: &Launch, host: &[FscOut], worst: &mut [u64; 5]) -> u64 {
    let mut zeroed = 0;
    for (i, h) in host.iter().enumerate() {
        let d = [
            ulps(l.d_t[i], h.d_t),
            ulps(l.u_t[i], h.u_t),
            ulps(l.u_t_isp[i], h.u_t_isp),
            ulps(l.tau_l[i], h.tau_l),
            ulps(l.k_rate[i], h.k_rate),
        ];
        for (w, v) in worst.iter_mut().zip(d) {
            *w = (*w).max(v);
        }
        zeroed += h.zeroed as u64;
    }
    zeroed
}

/// §154.9 row 1: 4096 random states in 64 clocks, FSC with the extended
/// FSC on (t_flacc = 0.15 s) and TFC, plus the no-turbulence edge states
/// of decision D5: every output within 4 ulp of the host twin, and the
/// zero counter equal to the host's prediction.
#[test]
fn device_matches_host() {
    let Some(g) = gpu("dust::fsc::tests::device_matches_host") else { return };
    let kern = FscKernels::new(&g).expect("the DUST module loads");
    let mut rng = SplitMix64(20261014);
    let clk = clocks(&mut rng);
    let states: Vec<Vec<FscState>> =
        (0..64).map(|_| (0..64).map(|_| random_state(&mut rng).0).collect()).collect();
    for (label, c) in [
        ("FSC", constants(CombustionModel::Fsc, Some(0.15))),
        ("TFC", constants(CombustionModel::Tfc, None)),
    ] {
        let mut worst = [0u64; 5];
        let (mut dev_zero, mut host_zero) = (0u64, 0u64);
        let (mut below, mut above, mut accel) = (0usize, 0usize, 0usize);
        for (s, &clock) in states.iter().zip(&clk) {
            let host: Vec<FscOut> = s.iter().map(|x| evaluate(&c, x, clock.0, clock.1)).collect();
            for h in &host {
                let x = if clock.0 > 0.0 { clock.0 / h.tau_l } else { 0.0 };
                if x < 0.5 { below += 1 } else { above += 1 }
            }
            if clock.1 >= 0.15 {
                accel += 1;
            }
            let l = run_fsc(&g, &kern, &c, s, clock);
            host_zero += compare(&l, &host, &mut worst);
            dev_zero += l.n_zero;
        }
        // D5's edges: no k, no omega, and nu_t = 0 with k > 0.
        let edge = [
            FscState { t: 1200.0, k: 0.0, omega: 100.0, nu_t: 0.0 },
            FscState { t: 1200.0, k: 1.0, omega: 0.0, nu_t: 0.0 },
            FscState { t: 1200.0, k: 1.0, omega: 100.0, nu_t: 0.0 },
        ];
        for clock in [(0.0, 0.0005), (0.05, 0.051), (0.3, 0.301)] {
            let host: Vec<FscOut> = edge.iter().map(|x| evaluate(&c, x, clock.0, clock.1)).collect();
            for h in &host {
                for v in [h.d_t, h.u_t, h.u_t_isp, h.tau_l, h.k_rate] {
                    assert!(v.is_finite(), "{label}: an edge state gave {v}");
                }
            }
            let l = run_fsc(&g, &kern, &c, &edge, clock);
            host_zero += compare(&l, &host, &mut worst);
            dev_zero += l.n_zero;
        }
        println!(
            "  {label}: worst ulps D_t {} U_t {} U_t,ISP {} tau_L {} K {}; zeroed {dev_zero} \
             (host {host_zero}); x < 0.5: {below}, x >= 0.5: {above}; clocks past t_flacc {accel}",
            worst[0], worst[1], worst[2], worst[3], worst[4]
        );
        for (name, w) in ["D_t", "U_t", "U_t,ISP", "tau_L", "K"].iter().zip(worst) {
            assert!(w <= ULPS, "{label}: {name} is {w} ulp from its host twin (bar {ULPS})");
        }
        assert_eq!(dev_zero, host_zero, "{label}: the zero counter against the host's prediction");
        assert!(below > 0 && above > 0, "{label}: the states must cover both sides of x = 0.5");
        assert!(accel > 0, "{label}: some clocks must be past t_flacc");
        if label == "FSC" {
            assert!(host_zero > 0, "the zero rule must be exercised");
        }
    }
}

/// A homogeneous state for the host limits: k = 1, omega = 100,
/// nu_t = k/omega, T = 1200 K.
fn homogeneous() -> FscState {
    FscState { t: 1200.0, k: 1.0, omega: 100.0, nu_t: 0.01 }
}

/// §154.9 row 2: D_t/D_t,inf = 1 to 1e-12 at t_fd/tau_L = 1e3, and
/// U_t/U_t,ISP = 1 to 1e-12 at t_fd/tau_L = 1e13. At 1e3, U_t/U_t,ISP is
/// sqrt(1 - 1e-3): the plan row's point, where 1e-12 cannot be reached.
#[test]
fn developed_limits() {
    let c = constants(CombustionModel::Fsc, None);
    let s = homogeneous();
    let tl = evaluate(&c, &s, 0.0, 0.0).tau_l;
    let dinf = s.nu_t / c.pr_t;
    let o3 = evaluate(&c, &s, 1.0e3 * tl, 1.0e3 * tl);
    let o13 = evaluate(&c, &s, 1.0e13 * tl, 1.0e13 * tl);
    let e_d = (o3.d_t / dinf - 1.0).abs();
    let e_u = (o13.u_t / o13.u_t_isp - 1.0).abs();
    let e_u3 = (o3.u_t / o3.u_t_isp / (1.0f64 - 1.0e-3).sqrt() - 1.0).abs();
    println!("  D_t/D_inf - 1 at 1e3: {e_d:.1e}; U_t/U_isp - 1 at 1e13: {e_u:.1e}; at 1e3 against sqrt(0.999): {e_u3:.1e}");
    assert!(e_d <= 1.0e-12, "D_t/D_t,inf at x = 1e3 is off by {e_d:e}");
    assert!(e_u <= 1.0e-12, "U_t/U_t,ISP at x = 1e13 is off by {e_u:e}");
    assert!(e_u3 <= 1.0e-12, "U_t/U_t,ISP at x = 1e3 is not sqrt(0.999): {e_u3:e}");
    // The clock's origin t_0 = 1 ms (§151.5): t_fd = max(t - t_0, 0).
    assert_eq!(development_time(0.0005, 1.0e-3), 0.0);
    assert_eq!(development_time(0.003, 1.0e-3), 0.003 - 1.0e-3);
}

/// §154.9 row 3: at t_fd/tau_L = 1e-4, U_t / [U_t,ISP (t_fd/(2 tau_L))^{1/2}]
/// = 1 to 1e-4; the series and the closed form agree at the switch x = 0.5
/// to 1e-14 (§154.2); and the two forms of (154.4) agree to 1e-14 over
/// 4096 random states (§154.3).
#[test]
fn small_time_series() {
    let c = constants(CombustionModel::Fsc, None);
    let s = homogeneous();
    let tl = evaluate(&c, &s, 0.0, 0.0).tau_l;
    let t_fd = 1.0e-4 * tl;
    let o = evaluate(&c, &s, t_fd, t_fd);
    let x = t_fd / o.tau_l;
    let e_s = (o.u_t / (o.u_t_isp * (x / 2.0).sqrt()) - 1.0).abs();
    let e_sw = (bracket_series(0.5) / bracket_closed(0.5) - 1.0).abs();
    let mut rng = SplitMix64(20261015);
    let mut e_id = 0.0f64;
    for _ in 0..4096 {
        let (st, _, _) = random_state(&mut rng);
        let o = evaluate(&c, &st, 0.0, 0.0);
        let u_prime = ((2.0 * st.k) / 3.0).sqrt();
        let l = (C_D * st.k.sqrt()) / (BETA_STAR * st.omega);
        let id = u_t_isp_identity(c.a, u_prime, S_L, kappa_u(), l);
        e_id = e_id.max((o.u_t_isp / id - 1.0).abs());
    }
    println!(
        "  early series: {e_s:.2e}; switch: {e_sw:.1e}; (154.4) two forms: {e_id:.1e}"
    );
    assert!(e_s <= 1.0e-4, "the early-time series row is off by {e_s:e}");
    assert!(e_sw <= 1.0e-14, "series and closed form disagree at x = 0.5 by {e_sw:e}");
    assert!(e_id <= 1.0e-14, "the two forms of (154.4) disagree by {e_id:e}");
}

/// §154.9 row 4: under TFC, D_t = D_t,inf, U_t = U_t,ISP, and K, Q, S_U
/// and S_P are +0 bitwise, on the host and on the device.
#[test]
fn tfc_has_no_q() {
    let c = constants(CombustionModel::Tfc, None);
    let mut rng = SplitMix64(20261016);
    let mut s = Vec::with_capacity(4096);
    for _ in 0..4096 {
        let (st, cc, rho) = random_state(&mut rng);
        let o = evaluate(&c, &st, 0.3, 0.301);
        assert_eq!(o.d_t.to_bits(), (st.nu_t / c.pr_t).to_bits(), "TFC: D_t is D_t,inf");
        assert_eq!(o.u_t.to_bits(), o.u_t_isp.to_bits(), "TFC: U_t is U_t,ISP");
        let (s_u, s_p) = implicit_split(rho, o.k_rate);
        for v in [o.k_rate, q_source(rho, o.k_rate, cc), s_u, s_p] {
            assert_eq!(v.to_bits(), 0, "TFC: K, Q, S_U and S_P are +0 bitwise");
        }
        assert!(!o.zeroed, "TFC counts nothing");
        s.push(st);
    }
    let Some(g) = gpu("dust::fsc::tests::tfc_has_no_q") else { return };
    let kern = FscKernels::new(&g).expect("the DUST module loads");
    let l = run_fsc(&g, &kern, &c, &s, (0.3, 0.301));
    assert!(l.k_rate.iter().all(|&v| bits(v) == 0), "TFC: the device's K is +0 bitwise");
    assert_eq!(l.n_zero, 0, "TFC: the device counts nothing");
}

/// §154.9 row 5: for t < t_flacc the extended FSC's U_t is bitwise the
/// plain FSC's (T2 is not evaluated), on the host and on the device; past
/// t_flacc it is U_t (t/t_flacc)^{1/2}.
#[test]
fn acceleration_off_before_t_flacc() {
    let on = constants(CombustionModel::Fsc, Some(0.15));
    let off = constants(CombustionModel::Fsc, None);
    let mut rng = SplitMix64(20261017);
    let s: Vec<FscState> = (0..4096).map(|_| random_state(&mut rng).0).collect();
    for (i, st) in s.iter().enumerate() {
        let t = 0.15 * (i as f64 + 0.5) / 4096.0;
        let a = evaluate(&on, st, t, t);
        let b = evaluate(&off, st, t, t);
        assert_eq!(a.u_t.to_bits(), b.u_t.to_bits(), "T2 must be exactly 1 at t = {t} < t_flacc");
        let a = evaluate(&on, st, 0.3, 0.3);
        let b = evaluate(&off, st, 0.3, 0.3);
        assert_eq!(a.u_t.to_bits(), (b.u_t * (0.3f64 / 0.15).sqrt()).to_bits(), "(154.7) past t_flacc");
    }
    let Some(g) = gpu("dust::fsc::tests::acceleration_off_before_t_flacc") else { return };
    let kern = FscKernels::new(&g).expect("the DUST module loads");
    let a = run_fsc(&g, &kern, &on, &s, (0.1, 0.1499));
    let b = run_fsc(&g, &kern, &off, &s, (0.1, 0.1499));
    assert!(
        a.u_t.iter().zip(&b.u_t).all(|(x, y)| bits(*x) == bits(*y)),
        "the device's T2 must be exactly 1 before t_flacc"
    );
    let a = run_fsc(&g, &kern, &on, &s, (0.3, 0.3));
    let b = run_fsc(&g, &kern, &off, &s, (0.3, 0.3));
    assert!(
        a.u_t.iter().zip(&b.u_t).any(|(x, y)| bits(*x) != bits(*y)),
        "past t_flacc the device must accelerate"
    );
}

/// §154.9 row 6: E over T in [150, 3500] K matches the f64 host to 4 ulp
/// wherever the host value is at least FLT_MIN 2^24, is +0 below, no word
/// is subnormal, and the counter equals the host's prediction. Decision
/// D2's floor is exercised by one dustFsc state whose D_t/kappa_b takes K
/// below FLT_MIN. c~ does not enter E: the device never forms (1 - c~)
/// (§154.8).
#[test]
fn arrhenius_f32_safe_no_subnormal() {
    let c = constants(CombustionModel::Fsc, None);
    assert!((zero_exponent() - (-70.70101242)).abs() < 1.0e-8, "-102 ln 2");
    let n = 4096;
    let t: Vec<f64> = (0..n).map(|i| r(150.0 + 3350.0 * i as f64 / (n - 1) as f64)).collect();
    let host: Vec<Option<f64>> = t.iter().map(|&x| arrhenius(&c, x)).collect();
    let floor = 2f64.powi(-102);
    for e in host.iter().flatten() {
        assert!(*e >= floor, "a kept E of {e:e} is below FLT_MIN 2^24");
    }
    let predicted = host.iter().filter(|e| e.is_none()).count() as u64;
    assert!(predicted > 0 && predicted < n as u64, "the sweep must cross the threshold");
    let Some(g) = gpu("dust::fsc::tests::arrhenius_f32_safe_no_subnormal") else { return };
    let kern = FscKernels::new(&g).expect("the DUST module loads");
    let td = up(&g, &t);
    let mut e = g.zeros::<Scalar>(n).expect("zeros");
    let mut nz = g.zeros::<u64>(1).expect("zeros");
    launch_arrhenius(&g, &kern, &c, &mut e, &mut nz, &td, n).expect("dustArrhenius launches");
    let e = g.download(&e).expect("download");
    let counted = g.download(&nz).expect("download")[0];
    let mut worst = 0u64;
    for (i, (d, h)) in e.iter().zip(&host).enumerate() {
        assert!(!subnormal(*d), "E[{i}] = {d:e} at T = {} is subnormal", t[i]);
        match h {
            Some(hv) => worst = worst.max(ulps(*d, *hv)),
            None => assert_eq!(bits(*d), 0, "E[{i}] at T = {} must be +0", t[i]),
        }
    }
    println!("  Arrhenius: worst {worst} ulp; zeroed {counted} (host {predicted}) of {n}");
    assert!(worst <= ULPS, "E is {worst} ulp from the host (bar {ULPS})");
    assert_eq!(counted, predicted, "the zero counter against the host's prediction");

    // D2: T = 240 K keeps E (exponent -70.0), and nu_t = 1e9 m^2/s,
    // developed (t_fd = 1e12 s against tau_L = 3.75e9 s), takes
    // K_b E/(1 + D_t/kappa_b) below FLT_MIN.
    let st = FscState { t: 240.0, k: 1.0, omega: 100.0, nu_t: 1.0e9 };
    let h = evaluate(&c, &st, 1.0e12, 1.0e12);
    assert!(arrhenius(&c, 240.0).is_some() && h.zeroed && h.k_rate.to_bits() == 0, "D2 on the host");
    let mut rng = SplitMix64(20261018);
    let mut s: Vec<FscState> = (0..255).map(|_| random_state(&mut rng).0).collect();
    s.push(st);
    let l = run_fsc(&g, &kern, &c, &s, (1.0e12, 1.0e12));
    let host: Vec<FscOut> = s.iter().map(|x| evaluate(&c, x, 1.0e12, 1.0e12)).collect();
    let want = host.iter().filter(|o| o.zeroed).count() as u64;
    for v in [&l.d_t, &l.u_t, &l.u_t_isp, &l.tau_l, &l.k_rate] {
        assert!(!v.iter().any(|w| subnormal(*w)), "a dustFsc word is subnormal");
    }
    assert_eq!(bits(l.k_rate[255]), 0, "D2: the device writes K below FLT_MIN as +0");
    assert_eq!(l.n_zero, want, "the dustFsc zero counter against the host's prediction");
}

/// The setup's refusals by name: K_b outside [1e-3, 1e12] 1/s (§154.7),
/// t_flacc with TFC (§154.6), and an input that is not positive.
#[test]
fn fsc_refusals() {
    let refused = |i: FscInputs, needle: &str| match FscConstants::new(&i) {
        Err(Error::Refused(m)) => assert!(m.contains(needle), "{m} should name {needle}"),
        other => panic!("expected a refusal naming {needle}, got {other:?}"),
    };
    let mut i = inputs(CombustionModel::Fsc, None);
    i.t_r = 1.0e-25;
    refused(i, "K_b");
    let mut i = inputs(CombustionModel::Fsc, None);
    i.t_r = 1.0e2;
    refused(i, "K_b");
    refused(inputs(CombustionModel::Tfc, Some(0.15)), "TFC");
    let mut i = inputs(CombustionModel::Fsc, None);
    i.s_l = 0.0;
    refused(i, "S_L");
    let mut i = inputs(CombustionModel::Fsc, Some(0.15));
    i.kappa_b = f64::NAN;
    refused(i, "kappa_b");
}
